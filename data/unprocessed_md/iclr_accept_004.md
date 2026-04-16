## **Implicit Regularization for Tubal Tensor Factorizations via Gradient Descent**

**Santhosh Karnik** [* 1] **Anna Veselovska** [* 2 3] **Mark Iwen** [4 5] **Felix Krahmer** [2 3]



**Abstract**


We provide a rigorous analysis of implicit regularization in an overparametrized tensor factorization problem beyond the lazy training regime. For
matrix factorization problems, this phenomenon
has been studied in a number of works. A particular challenge has been to design universal initialization strategies which provably lead to implicit regularization in gradient-descent methods.
At the same time, it has been argued by (Cohen
et al., 2016) that more general classes of neural
networks can be captured by considering tensor
factorizations. However, in the tensor case, implicit regularization has only been rigorously established for gradient flow or in the lazy training
regime. In this paper, we prove the first tensor
result of its kind for gradient descent rather than
gradient flow. We focus on the tubal tensor product and the associated notion of low tubal rank,
encouraged by the relevance of this model for image data. We establish that gradient descent in an
overparametrized tensor factorization model with
a small random initialization exhibits an implicit
bias towards solutions of low tubal rank. Our theoretical findings are illustrated in an extensive set
of numerical simulations show-casing the dynamics predicted by our theory as well as the crucial
role of using a small random initialization.


**1. Introduction**


Analyzing implicit regularization during Neural Network
(NN) training is considered crucial for understanding why


*Equal contribution 1Department of Mathematics, Northeastern University, Boston, USA [2] Department of Mathematics and
Munich Data Science Institute, Technical University of Munich,
Munich, Germany [3] Munich Center for Machine Learning, Munich, Germany [4] Department of Mathematics, Michigan State
University, East Lansing, USA [5] Department of Computational
Mathematics Science and Engineering, Michigan State University, East Lansing, USA. Correspondence to: Anna Veselovska
_<_ anna.veselovska@tum.de _>_ .


_Proceedings_ _of_ _the_ _42_ _[nd]_ _International_ _Conference_ _on_ _Machine_
_Learning_, Vancouver, Canada. PMLR 267, 2025. Copyright 2025
by the author(s).



overparametrization can give rise to superior generalization
capability and lead to strong overall NN performance. Consequently, there has been a recent surge in research aimed at
explaining how gradient-based methods interact with overparameterized models under nonconvex losses (see, e.g.,
(Ma et al., 2018; Ling & Strohmer, 2019)). Notably, recent empirical and theoretical studies have suggested that
gradient-based methods with small random initializations
exhibit a bias towards low-rank solutions in a variety of
models.


For matrix factorization models which represent linear neural networks, a rigorous analysis of implicit bias is available
for both gradient descent (Gunasekar et al., 2018; Stoger &¨
Soltanolkotabi, 2021) and gradient flow (its asymptotic limit
for small step size) (Bah et al., 2022; Chou et al., 2024).
In contrast, for neural networks with nonlinear activation,
there has been a good deal of work done showing that fully
connected layers can be represented by, e.g., tensor train factorizations in (Novikov et al., 2015; Razin et al., 2021). As
a consequence, it has been argued that tensor factorizations
should be considered instead of matrix factorizations (see,
e.g., (Cohen et al., 2016)). For tensor factorization models,
however, results predating 2024 were only available for the
asymptotic regime, i.e., gradient flow. This is perhaps due to
the many additional complications in the tensor setting beyond those in the matrix setting including, e.g, that there are
many different valid notions of tensor rank, each of which
motivates its own equally valid class of tensor factorizations.
For gradient descent applied to the tensor recovery problem,
only a very recent partial analysis by (Liu et al., 2024) currently exists for the tubal factorization model. This analysis
requires that the initialization already well approximates
the solution, only after which the convergence of gradient
descent toward a low tubal-rank solution is shown. Herein
we also focus on the tubal factorization, but establish the
corresponding implicit regularization result without needing
such a strong initialization assumption.


Our work is motivated by recent research showing that the
way neural networks are trained, especially with gradient descent, can lead to solutions with useful structure, even without adding explicit regularization terms. This phenomenon,
known as implicit regularization, has been studied in contexts such as sparse recovery (Vaskevicius et al., 2019) and
low-rank matrix completion (Li et al., 2020), where specific



1


**Implicit Regularization for Tubal Tensors via GD**



network architectures are designed to encourage certain
types of structure in the solutions. However, for tensor recovery problems, most existing work either focuses only on
gradient flow or provides only partial analysis. To the best
of our knowledge, our paper is the first to analyze implicit
bias under gradient descent with small random initialization
for a tensor recovery problem. We focus on the tubal rank
model, which is particularly relevant for applications like
video representation. This opens the door to a broader investigation into how implicit regularization can be used for
structured tensor recovery, how network architectures influence this bias, and what conditions ensure convergence. We
see this work as a starting point for a larger line of research
on implicit regularization in tensor problems.


**Related work:** In deep learning it is common to use more
network parameters than training points. In such overparameterized scenarios there are usually many networks that
achieve zero training error so that the training algorithm
effectively imposes an implicit regularization (bias) on the
solution it computes. In practice, training networks with gradient descent is both common and tends to favor solutions
that generalize well, offering the exploration of how gradient
descent implicitly regularlizes in overparameterized regimes
as one avenue for better understanding the success of deep
learning more widely. As a result, a lot of recent work has
been focussed on understanding the implicit regularization
phenomena of gradient descent in multiple settings. The
first theoretical works in this direction (Gunasekar et al.,
2017; 2018; Geyer et al., 2020; Arora et al., 2019; Soudry
et al., 2018) concentrated on training linear networks and
suggested that during training (stochastic) gradient descent
implicitly converges to a linear network (i.e., a linear function described by a matrix) that’s low rank. Motivated by
specific deep learning tasks, multiple works also investigated implicit bias phenomena in the special cases of sparse
vector and low-rank matrix recovery from underdetermined
measurements via an overparameterized square loss functional, where the vectors and matrices to be reconstructed
were deeply factorized into several vector/matrix factors. In
this setting, these works then showed that the dynamics of
vanilla gradient descent are biased towards sparse/low-rank
solutions, respectively (Chou et al., 2024; 2023; Li et al.,
2022; Kolb et al., 2023).


In the realm of optimization, a substantial body of work has
also emerged that provides guarantees for gradient descent’s
convergence in the nonconvex setting for different problems
such as phase retrieval, matrix completion, and blind deconvolution. Broadly, these findings can be categorized into
two main approaches: smart initialization coupled with local convergence (demonstrating, e.g., local convergence of
descent techniques starting from carefully designed spectral
initializations) (Ma et al., 2018; Tu et al., 2016; Ling &



Strohmer, 2019; Candes et al., 2015); and landscape analysis paired with saddle-escaping algorithms which show,
e.g., that all local minima are global and that saddle points
exhibit strict negative curvature so that (stochastic) gradientbased methods can effectively escape saddles and ensure
convergence to global minimizers (Jin et al., 2017; Ge et al.,
2015; Raginsky et al., 2017).


Notably, several studies (Woodworth et al., 2020; Ghorbani
et al., 2020) have highlighted the importance of the scale
of the training initialization for the generalization and test
performance of modern machine learning architectures. In
fact, a small random initialization followed by (stochastic)
gradient descent is arguably the most widely used training algorithm in contemporary machine learning. And,
stronger generalization performance is typically observed
with smaller-scale initializations. Implicit bias for low-rank
matrix recovery with small random initializations has been
extensively studied in this setting as a result by, e.g., (Stoger¨
& Soltanolkotabi, 2021; Soltanolkotabi et al., 2023; Wind,
2023; Kim & Chung, 2024). These studies have shown that
a small random Gaussian initialization behaves similarly to
a spectral initialization in overparameterized settings. Furthermore, they have shown that gradient descent algorithms
with this initialization tend to converge towards low-rank solutions (i.e., that they demonstrate an implicit regularization
towards low-rank solutions).


Recently, numerous connections between tensor decompositions and training neural networks have also been established by, e.g., (Novikov et al., 2015; Razin et al., 2021;
2022). These studies argue that low-rank tensor factorization helps explain implicit regularization in deep learning,
as well as how properties of real-world data translate this
regularization to generalization. Similar to how matrix factorization can be viewed as a linear neural network (i.e., a
fully connected network with linear activation), tensor factorizations correspond to a specific type of shallow (depthtwo) nonlinear convolutional neural network (Cohen et al.,
2016; Razin et al., 2021). Additionally, (Novikov et al.,
2015) demonstrated that the dense weight matrices of fully
connected layers can be converted to tensor trains while
preserving the layer’s expressive power. These findings
have positioned low-rank tensor factorizations as theoretical surrogates for various neural network learning settings,
thereby enhancing our understanding of implicit regularization and overparameterization, and so further motivating
investigation in this area.


Since no unique definition of tensor rank is available, related
literature concerning implicit bias has naturally split with
respect to the notion of tensor rank being considered: CPrank, Tucker-rank, and tubal-rank, in analogy to the analysis
of algorithms specifically designed for tensor recovery and
completion by, e.g., (Zhang et al., 2019; Hou et al., 2021;



2


**Implicit Regularization for Tubal Tensors via GD**



Figure 1: A low tubal-rank factorization of a threedimensional tensor. Using the (reduced) tubal-SVD, each
three-dimensional tensor _**T**_ _∈_ R _[n][×][m][×][k]_ can be decomposed
into a tubal product of three tensors _**T**_ = _**V**_ _∗_ **Σ** _∗_ _**W**_ _[⊤]_ with
_**V**_ _∈_ R _[n][×][n][×][k]_, _**W**_ _∈_ R _[m][×][m][×][k]_ and the frontal slice diagonal tensor **Σ** _∈_ R _[n][×][m][×][k]_ . Here, the tubal rank of a tensor
is the number of non-zero singular tubes in **Σ** _∈_ R _[n][×][m][×][k]_ .
For example, in the figure, the tubal rank of the tensor is
equal to six.


Kong et al., 2018; Ahmed et al., 2020; Liu et al., 2019; 2020;
Haselby et al., 2024). For the CP-tensor factorization, several results are available for gradient-based methods (Wang
et al., 2020; Ge & Ma, 2017). The first theoretical analysis
of implicit regularization towards low tensor rank under
arbitrarily small initialization was provided considering gradient flow in (Razin et al., 2021). In (Ge et al., 2015), it has
been shown for the orthogonal tensor decomposition problem a simple variant of the stochastic gradient algorithm
is able to leverage a low-rank structure from an arbitrary
starting point. In addition, (Wang et al., 2020) shows that
using gradient descent on an over-parametrized objective
for the CP-rank tensor decomposition problem one could go
beyond the lazy training regime and utilize certain low-rank
structures.


Perhaps most closely related to this paper, very recently
(Liu et al., 2024) analyzed the convergence of factorized
gradient descent for the low-tubal-rank sensing problem,
showing that with carefully designed spectral initialization
the gradient iterates converge to a low-tubal rank tensor.
Although the authors in (Liu et al., 2024) allow for overparametrization, they argue the minimal recovery error can
be achieved when knowing the true rank, thereby leaving
questions concerning the advantages of overparametrization
and small random initializations open.


**Our** **contribution:** Motivated by connections between
tensor rank and non-linear neural network representations,
herein we study the implicit regularization phenomenon for
low tubal-rank tensor recovery. Namely, our objective is to
analyze the recovery process of a tensor with a low tubalrank factorization (Kilmer & Martin, 2011) (see Fig 1) from
a limited number of random linear measurements. More



specifically, we consider tensors of the form _**X**_ _∗_ _**X**_ _[⊤]_ and employ a non-convex method based on the tensor factorization,
minimizing the loss function using gradient descent with a
small random initialization. To the best of our knowledge,
we are the first to investigate the implicit bias phenomenon
for gradient descent with a small random initialization applied to a tensor factorization. Namely, we demonstrate that,
irrespective of the degree of overparameterization, vanilla
gradient descent with a small random initialization applied
to a tubal tensor factorization will consistently converge to
a low tubal-rank solution.


Inspired by recent results for the low-rank matrix sensing
problem by (Stoger & Soltanolkotabi, 2021), we establish¨
that gradient descent iterates with small random initializations can be closely approximated by power method iterations in (Gleich et al., 2013; Kilmer et al., 2013) modulo
normalization, and deduce that after sufficient time the iterates approach a commonly used spectral initialization from
the tubal-rank literature in (Liu et al., 2024). Along the way
we must also overcome, e.g., a challenging intersection between the tensor slices during each gradient descent iterate
which forces a non-trivial convergence analysis.


**Organization:** In Section 2, we define our notation and
present a few basic facts regarding tubal tensors. In Section 3, we state our problem and our main result. In Section 4, we outline the steps of the proof in order to provide
intuition. In Section 5, we show numerical experiments
which demonstrate our theoretical findings. We conclude
the paper in Section 6. The proof of our main result is broken up into several lemmas, which are stated and proven in
the appendix.


**2. Notation and Preliminaries**


Every tensor in this paper will be an order-3 tensor whose
third mode is length _k_ . For such a tensor _**T**_ _∈_ R _[m][×][n][×][k]_, we
define a block-diagonal Fourier domain representation by


_**T**_ = blockdiag( _**T**_ ~~(~~ 1) _, . . .,_ _**T**_ ~~(~~ _k_ )) _∈_ C _mk×nk_


~~(~~ _j_ )
where the _j_ -th block _**T**_ _∈_ C _[m][×][n]_ is defined by
_**T**_ ~~(~~ _j_ )( _i, i′_ ) = [�] _kj_ _[′]_ =1 _**[T]**_ [ (] _[i, i][′][, j][′]_ [)] _[e][−]_ ~~_[√]_~~ _[−]_ [12] _[π]_ [(] _[j][−]_ [1)(] _[j][′][−]_ [1)] _[/k][.]_ [In]
other words, we take the FFT of each tube, and then arrange
the resulting frontal slices into a block-diagonal matrix.


The tubal product (or t-product) of two tubal tensors _**A**_ _∈_
R _[m][×][q][×][k]_ and _**B**_ _∈_ R _[q][×][n][×][k]_ is a tubal tensor _**A**_ _∗_ _**B**_ _∈_
R _[m][×][n][×][k]_ whose tubes are given by



Here, _∗_ denotes the circular convolution operation, i.e., ( _**x**_ _∗_



( _**A**_ _∗_ _**B**_ )( _i, i_ _[′]_ _,_ :) =



_q_

- _**A**_ ( _i, p,_ :) _∗_ _**B**_ ( _p, i_ _[′]_ _,_ :) _._


_p_ =1



3


**Implicit Regularization for Tubal Tensors via GD**



_**y**_ ) _i_ = [�] _j_ _[k]_ =1 _[x][j][y][i][−][j]_ [(mod] _[k]_ [)][.] [One can check that] _**[ A]**_ _[ ∗]_ _**[B]**_ [=]
_**A B**_ .


For any tubal tensor _**T**_ _∈_ R _[m][×][n][×][k]_, its tubal transpose
_**T**_ _[⊤]_ _∈_ R _[n][×][m][×][k]_ is given by ( _**T**_ _[⊤]_ )( _i, i_ _[′]_ _,_ 1) = _**T**_ ( _i_ _[′]_ _, i,_ 1)
and ( _**T**_ _[⊤]_ )( _i, i_ _[′]_ _, j_ ) = _**T**_ ( _i_ _[′]_ _, i, k_ + 2 _−_ _j_ ) for _j_ = 2 _, . . ., k_,
i.e., we take the transpose of each face, and then reverse
the order of frontal slices _j_ = 2 _, . . ., k_ . This ensures that
_**T**_ _[⊤]_ = _**T**_ ~~_⊤_~~ .


For any _n_, the _n_ _×_ _n_ _×_ _k_ identity tensor _**I**_ _∈_ R _[n][×][n][×][k]_

is defined by _**I**_ (: _,_ : _,_ 1) = _In×n_ (identity matrix), and _**I**_ (:
_,_ : _, j_ ) = 0 _n×n_ (zero matrix). An orthogonal tensor _**Q**_ _∈_
R _[n][×][n][×][k]_ satisfies _**Q**_ _∗_ _**Q**_ _[⊤]_ = _**Q**_ _[⊤]_ _∗_ _**Q**_ = _**I**_ . An orthonormal
tensor _**W**_ _∈_ R _[m][×][n][×][k]_ with _m ≥_ _n_ satisfies _**W**_ _[⊤]_ _∗_ _**W**_ = _**I**_ .


The tubal-SVD (Kilmer & Martin, 2011) (or t-SVD) of a
tubal tensor _**T**_ _∈_ R _[m][×][n][×][k]_ is a factorization of the form


_**T**_ = _**U**_ _∗_ **Σ** _∗_ _**V**_ _[⊤]_ (2.1)


where _**U**_ _∈_ R _[m][×][m][×][k]_ and _**V**_ _∈_ R _[n][×][n][×][k]_ are orthogonal, and
each frontal slice of **Σ** _∈_ R _[m][×][n][×][k]_ is diagonal. The t-SVD
of a tensor _**T**_ _∈_ R _[m][×][n][×][k]_ can be computed as follows: (1)
compute the FFT of each tube of _**T**_ to get the frontal slices
~~(~~ _j_ )
_**T**_, _j_ = 1 _, . . ., k_, (2) compute the SVD of each resulting
~~(~~ _j_ ) ~~(~~ _j_ ) ~~(~~ _j_ ) ~~(~~ _j_ ) _⊤_
frontal slice _**T**_ = _U_ Σ _V_, (3) concatenate the
matrices _{U_ ( _j_ ) _}kj_ =1 [into a tubal tensor] _**[U]**_ [�] _[∈]_ [C] _[m][×][m][×][k]_ [and]
take the inverse FFT along mode-3 to obtain _**U**_ _∈_ R _[m][×][m][×][k]_

(and similarly to obtain **Σ** _∈_ R _[m][×][n][×][k]_ and _**V**_ _∈_ R _[n][×][n][×][k]_ ).
The tubal rank of a tensor _**T**_ _∈_ R _[m][×][n][×][k]_ is the number of
non-zero diagonal tubes in the **Σ** tensor of its t-SVD, i.e.,
rank( _**T**_ ) = # _{i_ : **Σ** ( _i, i,_ :) = 0 _}_ . For an illustration of
the t-SVD decomposition, see Figure 1. We also define the
condition number _κ_ ( _**T**_ ) of the tubal tensor _**T**_ _∈_ R _[m][×][n][×][k]_

by

_σ_ 1( _**T**_ )
_κ_ ( _**T**_ ) :=
_σ_ min _{m,n}k_ ( _**T**_ ) _[.]_


Finally, for tubal tensors _**T**_ _∈_ R _[m][×][n][×][k]_ we define
the tensor spectral norm _∥_ _**T**_ _∥_ := _∥_ _**T**_ _∥_ and the tensor
nuclear norm _∥_ _**T**_ _∥∗_ := _∥_ _**T**_ _∥∗_ as the spectral and nuclear norm respectively of the block-diagonal Fourier domain representation _**T**_, and the tensor Frobenius norm
_∥_ _**T**_ _∥_ [2] _F_ [:=][ �] _i_ _[m]_ =1 - _nj_ =1 - _kℓ_ =1 _**[T]**_ [ (] _[i, j, ℓ]_ [)][2] [=] _k_ [1] _[∥]_ _**[T]**_ _[ ∥]_ _F_ [2] as a

scaled version of the Frobenius norm of the block-diagonal
Fourier domain representation _**T**_ .


**3. Main Results**


**Problem Formulation** Let _**X**_ _∈_ R _[n][×][r][×][k]_ have tubal rank
_r_ _≤_ _n_ so that _**X**_ _∗_ _**X**_ _[⊤]_ _∈_ _S_ + _[n][×][n][×][k]_ is a tubal positive
semidefinite tensor with tubal rank _r_ . Let _κ_ = _κ_ ( _**X**_ ) be
the condition number of _**X**_ . Suppose we observe _m_ linear



We will start with a small random initialization _**U**_ 0 _∈_
R _[n][×][R][×][k]_ where each entry is i.i.d. _N_ (0 _,_ _[α]_ _R_ [2] [)][ for some small]

_α >_ 0. Then, the gradient descent iterations are given by


_**U**_ _t_ +1 = _**U**_ _t −_ _µ∇ℓ_ ( _**U**_ _t_ )


          - ��
= _**U**_ _t_ + _µA_ _[∗]_ [�] _**y**_ _−A_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ _∗_ _**U**_ _t_

=   - _**I**_ + _µ_ ( _A_ _[∗]_ _A_ )   - _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ �� _∗_ _**U**_ _t_

(3.4)


for some suitably small stepsize _µ_ _>_ 0. Here
_A_ _[∗]_ : R _[m]_ _→_ _S_ _[n][×][n][×][k]_ denotes the adjoint of _A_ which is
given by _A_ _[∗]_ _**z**_ = [�] _i_ _[m]_ =1 _**[z]**_ _[i]_ _**[A]**_ _[i]_ [.]


Moreover, we say that a measurement operator
_A_ : _S_ _[n][×][n][×][k]_ _→_ R _[m]_ satisfies the Restricted Isometry
Property (RIP) of rank- _r_ with constant _δ_ _>_ 0 (abbreviated
RIP( _r, δ_ )), if we have


(1 _−_ _δ_ ) _∥_ _**Z**_ _∥_ [2] _F_ _[≤∥A]_ [(] _**[Z]**_ [)] _[∥]_ 2 [2] _[≤]_ [(1 +] _[ δ]_ [)] _[∥]_ _**[Z]**_ _[∥]_ _F_ [2] _[,]_


for all _**Z**_ _∈_ _S_ _[n][×][n][×][k]_ with tubal-rank _≤_ _r_ . We note that an
RIP condition is a standard condition in the literature, and
is used in similar works such as (Li et al., 2018; Stoger &¨
Soltanolkotabi, 2021). This condition is necessary to ensure
that there is only one low tubal rank tensor for which the
loss function is zero, and that this tensor could be recovered
stably in the presence of noise.


**Results** We have analyzed the convergence process of the
gradient descent iterates (3.4) in the scenario of small random initialization and overparametrization. Namely, with
the ground truth tensor _**X**_ _∈_ R _[n][×][r][×][k]_, we assume the initialization _**U**_ 0 _∈_ R _[n][×][R][×][k]_ is such that each entry is i.i.d.
_N_ (0 _,_ _[α]_ _R_ [2] [)][ with small scaling parameter] _[ α >]_ [ 0][ and the sec-]

ond dimension _R_ exceeding three timesthe ground truth
dimension _r_ . Below, we present the direct results of our
analysis.



measurements of _**X**_ _∗_ _**X**_ _[⊤]_, that is


_yi_ =    - _**A**_ _i,_ _**X**_ _∗_ _**X**_ _[⊤]_ [�] for _i_ = 1 _, . . ., m_ (3.1)


where each _**A**_ _i_ _∈_ _S_ _[n][×][n][×][k]_ is a tubal-symmetric tensor.
We can write this compactly as _**y**_ = _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ) where
_A_ : _S_ _[n][×][n][×][k]_ _→_ R _[m]_ is the linear measurement operator. We
aim to recover _**X**_ _∗_ _**X**_ _[⊤]_ from our measurements _**y**_ by using
gradient descent to learn an overparameterized factorization.
Specifically, we fix an _R ≥_ _r_ and try to find a _**U**_ _∈_ R _[n][×][R][×][k]_

such that _**U**_ _∗_ _**U**_ _[⊤]_ = _**X**_ _∗_ _**X**_ _[⊤]_ by using gradient descent to
minimize the loss function


        - 2
_ℓ_ ( _**U**_ ) : = _A_ _**U**_ _∗_ _**U**_ _[⊤]_ [�] _−_ _**y**_ (3.2)
��� ���2



=



_m_



_i_ =1



�� �2
_**A**_ _i,_ _**U**_ _∗_ _**U**_ _[⊤]_ [�] _−_ _yi_ _._ (3.3)



4


**Implicit Regularization for Tubal Tensors via GD**



**Theorem** **3.1.** _Suppose_ _we_ _have_ _m_ _linear_ _measurements_
_y_ = _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ) _of_ _a_ _tubal_ _positive_ _semidefinite_ _tensor_
_**X**_ _∗_ _**X**_ _[⊤]_ _∈_ _S_ + _[n][×][n][×][k]_ _where_ _**X**_ _∈_ R _[n][×][r][×][k]_ _has tubal rank_
_r_ _≤_ _n._ _We_ _assume_ _A_ _satisfies_ _RIP_ (2 _r_ + 1 _, δ_ ) _with_ _δ_ _≤_
_cκ_ _[−]_ [4] _r_ _[−]_ [1] _[/]_ [2] _._ _Suppose we fit a model_ _**X**_ _∗_ _**X**_ _[⊤]_ = _**U**_ _∗_ _**U**_ _[⊤]_

_where_ _**U**_ _∈_ R _[n][×][R][×][k]_ _with R ≥_ 3 _r and obtain_ _**U**_ _by running_
_the gradient descent iterations_


_**U**_ _t_ +1 = - _**I**_ + _µ_ ( _A_ _[∗]_ _A_ ) - _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ �� _∗_ _**U**_ _t_



_√_
_with a stepsize µ ≤_ _c_



_with a stepsize µ ≤_ _c_ _kκ_ _[−]_ [4] _∥_ _**X**_ _∥_ [2] _starting from the initial-_

_ization_ _**U**_ 0 _∈_ R _[n][×][R][×][k]_ _where each entry is i.i.d._ _N_ (0 _,_ _[α]_ [2] [)] _[.]_



_ization_ _**U**_ 0 _∈_ R _where each entry is i.i.d._ _N_ (0 _,_ _R_ [)] _[.]_

_Then, if the scale of the initialization satisfies_




- _−_ 16 _κ_ [2]



_σmin_ ( _**X**_ )
_α_ ≲ ~~_√_~~
_κ_ [2] min _{n, R}_


_then after_



_k_




_C_ 2 _κ_ [2] _[√]_ _n_

~~�~~ min _{n, R}_



_,_




- _t_ ≲ _µσmin_ 1( _**X**_ ) [2] [ln] - min _C{_ 1 _n,Rκn_ _}_ [min] �1 _,_ _k_ (min _{κrn,R}−r_ ) - _∥kα_ _**X**_ _∥_ 

_iterations, we have that_


_∥_ _**U**_ - _t_ _∗_ _**U**_ - _[⊤]_ _t_ _[−]_ _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤][∥]_ [2] _F_ ≲
_∥_ _**X**_ _∥_ [2]



constant of _δ_ = _O_ ( _κ_ _[−]_ [4] _r_ _[−]_ [1] _[/]_ [2] ), one needs _m ≥_ _O_ ( _κ_ [8] _r_ [2] _nk_ )
random sub-Gaussian measurements.


Additionally, we acknowledge that the parameter dependence in Theorem 3.1 may initially seem unfamiliar. However, it aligns well with intuition and prior work: when the
tensor is ill-conditioned – i.e., possesses a small tubal singular value – gradient descent without regularization naturally
struggles to recover the rank-one component unless the initialization is sufficiently small. While our bound exhibits
exponential dependence on the condition number, this is
consistent with known results in the matrix setting (e.g., see
Lemma 8.6 in (Stoger & Soltanolkotabi, 2021)).Although¨
the necessity of exponential dependence remains an open
question, it presents a compelling direction for future research. Moreover, our numerical experiments (see Figure 4)
support a polynomial relationship between the test error and
the initialization parameter _α_, and while the empirical degree may differ slightly, our theoretical exponent [21] 16 [appears]

to closely approximate the observed behavior.


**4. Proof Outline**


In this section, we turn our attention to giving an overview
of the key ideas of the proof.


In our analysis, we demonstrate that the trajectory of gradient descent iterations can be approximately divided into two
distinct stages: (I) a spectral stage and (II) a convergence
stage described below.


_(I) The spectral stage._ In the spectral stage, where we show
that the gradient descent starting from random initialization
behaves similarly to spectral initialization, enabling us to
prove that by the end of this stage, the column spaces of
the tensor iterates _**U**_ _t_ (3.4) and the ground truth matrix _**X**_
are sufficiently aligned. Namely, we show that the first
few iterations of the gradient descent algorithm _**U**_ _t_ can be
approximated by the iteration of the tensor power method
modulo normalization (see, e.g.(Gleich et al., 2013)) defined
as


    -    - _∗t_
_**U**_  - _t_ = _**I**_ + _µA_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ) _∗_ _**U**_ 0 _∈_ R _[n][×][R][×][k]_ _._


We call this part of the evolution of the gradient descent
iteration the “spectral stage” since, due to its similarity to
the power method, at the end of this stage the iterates _**U**_ _t_
will be closely aligned with the classical t-SVD spectral
initialization of (Liu et al., 2024).


_(II) The convergence stage_ . In the convergence stage, the
gradient iterates converge approximately to the underlying
low tubal-rank tensor _**X**_ _∗_ _**X**_ _[⊤]_ at a geometric rate until
reaching a certain error floor which is dependent on the
initialization scale.


The cornerstone of the analysis of this stage is the de



_−_ 3 3 
16 (min _{n, R} −_ _r_ ) 8 ~~_√_~~ _C_ 2 _κ_ [2] ~~_[√]_~~ _n_



61 1 _−_ 3
_k_ 32 _r_ 8 _κ_ 16



min _{n,R}_



�21 _κ_ [2] - _α_ - [21] 16
_∥_ _**X**_ _∥_



_holds_ _with_ _probability_ _at_ _least_ 1 _−_ _Cke_ _[−][cR]_ [˜] _._ _Here,_
_c,_ ˜ _c, C, C_ 1 _, C_ 2 _>_ 0 _are fixed numerical constants._


Intuitively, this means that if the initialization is sufficiently
small, gradient descent will approximately recover the low
tubal rank tensor _**X**_ _∗_ _**X**_ _[⊤]_ after _t_ iterations. Note that the

[�]
reconstruction error can be made arbitrarily small by making
the size of the random initialization _α_ arbitrarily small. This
comes at the expense of requiring more iterations. However,
this impact is mild as the number of iterations grows only
logarithmically with respect to _α_ .


Although the above theorem holds for any _R_ _≥_ 3 _r_, it is
perhaps most interesting in the case where _R_ _≥_ _n_ as then
every _n × n × k_ tubal positive semidefinite tensor can be
expressed as _**U**_ _∗_ _**U**_ _[⊤]_ for some _**U**_ _∈_ R _[n][×][R][×][k]_ . Hence, the
learner model does not assume that the ground truth tensor
has low tubal rank, yet gradient descent is able to recover
the ground truth tensor instead of any of the infinitely many
high tubal rank tensors whose measurements match that of
the ground truth tensor.


We note that (Zhang et al., 2019) shows that a random subGaussian measurement operator _A_ : R _[n][×][n][×][k]_ _→_ R _[m]_ will
satisfy the RIP for tubal rank- _r_ tensors with RIP constant _δ_
with high probability if _m ≥_ _O_ ( _rnk/δ_ [2] ). To obtain an RIP



5


**Implicit Regularization for Tubal Tensors via GD**


denote by _**W**_ _t,⊥_ _∈_ R _[R][×]_ [(] _[n][−][r]_ [)] _[×][k]_ a tensor whose tensorcolumn subspace is orthogonal to those of _**W**_ _t_, that is
_∥_ _**W**_ _[⊤]_ _t,⊥_ _[∗]_ _**[W]**_ _[t][∥]_ [=] [0] [and] [its] [projection] [operator] _**[P]**_ _**[W]**_ _t,⊥_
is defined as _**P**_ _**W**_ _t,⊥_ = _**W**_ _t,⊥_ _∗_ _**W**_ _[⊤]_ _t,⊥_ [=] _**[ I]**_ _[−]_ _**[W]**_ _[t]_ _[∗]_ _**[W]**_ _[⊤]_ _t_ [.]


We then decompose the gradient descent iterates (3.4) as
follows


_**U**_ _t_ = _**U**_ _t ∗_ _**W**_ _t ∗_ _**W**_ _[⊤]_ _t_ [+] _**[ U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ (4.1)


referring to the tensors _**U**_ _t ∗_ _**W**_ _t ∗_ _**W**_ _[⊤]_ _t_ [as the signal term]
of the gradient descent iterates, and to the tensors _**U**_ _t_ _∗_
_**W**_ _t,⊥_ _∗_ _**W**_ _[⊤]_ _t,⊥_ [as the noise term.] [The advantage of such a]
decomposition is that the tensor-column space of the noise
term _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _[⊤]_ _t,⊥_ [is orthogonal to the tensor-column]
subspace of the ground truth _**X**_ allowing for a rigorous
analysis of the convergence process of the two components
separately.


At the convergence stage, we show that symmetric tensor
_**U**_ _t_ _∗_ _**W**_ _t_ _∗_ _**W**_ _[⊤]_ _t_ _[∗]_ _**[U]**_ _t_ _[⊤]_ [built from the signal term converges to-]
wards the ground truth tensor _**X**_ _∗_ _**X**_ _[⊤]_, whereas the spectral
norm of the noise term _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥∥_, stays small.


**Additional** **challenges** **in** **the** **tensor** **setting** **vs.** **matrix**
**setting** When coming from the matrix case to the tensor
setting com, there are several important differences and
challenges, which need to be carefully considered and are
described below.



Figure 2: Illustration of (top figure) the two stages of gradient descent algorithm: the spectral alignment stage for
1 _≤_ _t_ ≲ 3000 and the convergence stage 3000 ≲ _t_ and
(bottom figure) more details on the alignment phase for
the gradient descent progress. In the ground truth tensor
_**X**_ _∈_ R _[n][×][r][×][k]_, we set _n_ = 10 _, k_ = 4 _, r_ = 3.


composition of the tensor gradient iterates _**U**_ _t_ into two
components, the so-called “signal” and “noise” terms.
This is done by adapting similar decomposition methods
used in recent works analyzing implicit bias phenomenon
for gradient descent in the matrix setting (see (Stoger¨ &
Soltanolkotabi, 2021; Li et al., 2018)) to our tensor setting. Accordingly, let the tensor-column subspace of the
ground truth tensor _**X**_ _∈_ R _[n][×][r][×][k]_ be denoted by _**V**_ _**X**_ with
the corresponding basis _**V**_ _**X**_ _∈_ R _[n][×][r][×][k]_ . Consider the tensor _**V**_ _**X**_ _∗_ _**U**_ _t_ _∈_ R _[r][×][R][×][k]_ with its t-SVD decomposition
_**V**_ _**X**_ _∗_ _**U**_ _t_ = _**V**_ _t_ _∗_ **Σ** _t_ _∗_ _**W**_ _[⊤]_ _t_ [.] [For] _**[W]**_ _[t]_ _[∈]_ [R] _[R][×][r][×][k]_ [,] [we]




- In contrast to the matrix case, the range and kernel
of a third-order tubal tensor can include overlapping
generator elements (we refrain from using the term
basis, in the sense that knowledge of the multirank
and complimentary tubal scalar of a tensor must be
included to describe the range). Namely, if in the
t-SVD (2.1) of a symmetric tensor _**X**_ the tensor **Σ**
contains _q_ non-invertible tubes – tubes that have zero
elements in the Fourier domain –, then there are _q_
common generators for the range and the kernel of
_**X**_, please see (Kilmer et al., 2013) for more details.
With this phenomenon, the decomposition (C.1) of
the gradient iterates into signal and noise term is not
available for non-invertible tubes, which is why we
need to work with a more intricate notion of condition
number.


- As stated in (Gleich et al., 2013), running the power
method for tubal tensors of dimensions _n_ _×_ _n_ _×_ _k_
is equivalent to running in parallel _k_ independent
matrix power methods in Fourier domain. However,
running gradient descent in the tubal tensor setting
is not equivalent to running _k_ gradient descent
algorithms independently in Fourier space. This
can be easily seen when transforming the measurement operator part of the gradient descent iterates.



6


**Implicit Regularization for Tubal Tensors via GD**


Namely, let as before _y_ = _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ) _∈_ R _[m]_

with _yi_ =              - _**A**_ _i,_ _**X**_ _∗_ _**X**_ _[⊤]_ [�] =              - _**A**_ _i,_ _**X**_ _∗_ _**X**_ _[⊤]_ [�] =

      - _kq_ =1       - _Ai_ [(] _[q]_ [)] _, X_ [(] _[q]_ [)] _X_ [(] _[q]_ [)H][�] _,_ _j_ = 1 _, . . . m_ then
_A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ) = _A_ _[∗]_ ( _y_ ) =        - _mi_ =1 _[y][i]_ _**[A]**_ _[i]_ _∈_
_S_ _[n][×][n][×][k]_ and the for _j_ -th slice in the
Fourier domain, we get _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ) [(] _[j]_ [)] =

      - _mi_ =1       - _kj_ =1 _[A][i]_ [(] _[j]_ [)][ �] _Ai_ [(] _[q]_ [)] _, X_ [(] _[q]_ [)] _X_ [(] _[q]_ [)H][�] _._ This means

that in each Fourier slice _**U**_ _t_ [(] _[j]_ [)] of the gradient descent
iterates (3.4) we have the full information about the
ground truth tensor _**X**_ _∗_ _**X**_ _[⊤]_ and not only about its
_j_ -th slice. In the spectral stage, this fact does not cause
significant difficulties. However, in the convergence
stage, in order to get the global estimates, it requires a
thorough and vigilant analysis of intersections between
the slices in the Fourier domain.



Figure 3: Outcomes of employing gradient descent to minimize the loss function (3.2) with different overparametrization rates. We set _n_ = 10 _, k_ = 4 _, r_ = 3 in the ground truth
tensor _**X**_ _∈_ R _[n][×][r][×][k]_ and for initialization _**U**_ 0 _∈_ R _[n][×][R][×][k]_,
we set the over-rank to _R_ = 10 _,_ 50 _,_ 100 _,_ 200 _,_ 400. For
each _R_ we plot the average over twenty experiments. The
plots for _[∥]_ _**[U]**_ _[t][∗]_ _∥_ _**[U]**_ _**X**_ _t_ _[⊤]_ _∗_ _[−]_ _**X**_ _**[X]**_ _[⊤][∗]_ _∥_ _**[X]**_ _F_ _[ ⊤][∥][F]_, _ℓ_ ( _Ut_ ) and _[∥][σ][r]_ [(] _∥_ _**[U]**_ _σ_ _[t]_ _r_ [)] ( _[−]_ _**X**_ _[σ]_ ) _[r]_ _∥_ [(] 2 _**[X]**_ [)] _[∥]_ [2] are

semi-log plots.



In particular, this required nontrivial estimations, such
as those presented in Lemmas E.4 and E.5, to control
these interactions and provide the respective bounds,
which require control of proximity of the auxiliary
parameter   - _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] �( _j_ ) to the cor
responding _j_ th Fourier slice of _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [via]
the RIP property of the measurement operator _A_ and
aligned matrix subspaces. Another important point
is that one need to choose the learning rate _µ_ and
the initialization scale _α_ carefully for the noise term
_**U**_ _t ∗_ _**W**_ _⊥,t_ to grow slowly enough in each of the tensor slices in order to not allow overtaking the signal
term _**U**_ _t ∗_ _**W**_ _t_ in the norm, see, e.g., Theorem E.1 and
the usage of Lemma E.3 in its proof.


**5. Numerical Experiments**


To verify our theoretical findings, we set multiple numerical
tests: from showing two phases of the gradient descent algorithm to demonstrating the advantages of overparametrization. These experimental results showcase not only the
implicit regularization for the gradient descent algorithm
toward low-tubal-rank tensors but also demonstrate the firmness of our theoretical findings.


Our experiments were conducted on a MacBook Pro
equipped with an Apple M1 processor and 16GB of
memory, using MATLAB 2023a software. The corresponding code is available in our GitHub repository, [https://github.com/AnnaVeselovskaUA/tubal-tensor-](https://github.com/AnnaVeselovskaUA/tubal-tensor-implicit-reg-GD.git)
[implicit-reg-GD.git.](https://github.com/AnnaVeselovskaUA/tubal-tensor-implicit-reg-GD.git)


We generate the ground truth tensor _**T**_ _∈_ R _[n][×][n][×][k]_ with
tubal rank _r_ by _**T**_ = _**X**_ _∗_ _**X**_ _[⊤]_, where the entries of
_**X**_ _∈_ R _[n][×][r][×][k]_ are i.i.d. sampled from a Gaussian distribution _N_ (0 _,_ 1), and then _**X**_ is normalized. The entries of
measurement tensor _**A**_ _i_ are i.i.d. sampled from a Gaussian
distribution _N_ (0 _,_ _m_ [1] [)][.] [In] [the] [following,] [we] [describe] [dif-]



7


**Implicit Regularization for Tubal Tensors via GD**



ferent testing scenarios for recovery of _**T**_ via the gradient
descent algorithm and their outcome. For all the experiments, we set the dimensions to _n_ = 10 _, k_ = 4 _, r_ = 3, the
learning rate _µ_ = 10 _[−]_ [5], and the number of measurements
_m_ = 254.


**Illustration of the two convergence stages.** To illustrate
the convergence process of the gradient iterates, for the
ground truth tensor _**X**_ _∗_ _**X**_ _[⊤]_ _∈_ R _[n][×][n][×][k]_ and its counterpart _**U**_ _t_ _∗_ _**U**_ _[⊤]_ _t_ _∈_ R _[n][×][n][×][k]_ being learned by the gradient
descent, we consider the training error _ℓ_ ( _Ut_ ), the test error
_∥_ _**U**_ _t∗∥_ _**UX**_ _[⊤]_ _t∗_ _[−]_ _**X**_ _**[X]**_ _[⊤][∗]_ _∥_ _**[X]**_ _F_ _[ ⊤][∥][F]_, and the test error for their _r_ th singular

tubes _σr_ ( _**U**_ _t_ ) _, σr_ ( _**X**_ ) _∈_ R _[k]_, _[∥][σ][r]_ [(] _∥_ _**[U]**_ _σ_ _[t]_ _r_ [)] ( _[−]_ _**X**_ _[σ]_ ) _[r]_ _∥_ [(] 2 _**[X]**_ [)] _[∥]_ [2] . Moreover,

we also take into our consideration the tensor subspace _**L**_
spanned by the tensor-columns corresponding to the first
_r_ singular-tubes of the tensor _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ) and denote
by _**L**_ _t_ the tensor-column subspace spanned by the tensorcolumns corresponding to the first _r_ singular tubes _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [.]
We note that although Theorem 3.1 bounded a relative error
with _∥_ _**X**_ _∥_ [2] in the denominator, we use _∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _∥F_ in the
denominator of the relative error for our experiments as it
is a more natural relative error to consider. Furthermore,
since _∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _∥F_ _≥∥_ _**X**_ _∥_ [2], and _∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _∥F_ could be
much larger than _∥_ _**X**_ _∥_ [2] in cases where the singular values
of _**X**_ _∗_ _**X**_ _[⊤]_ vary drastically, the result of Theorem 3.1 is
stronger than if we bounded the more natural Frobenius
norm error. Besides, the qualitative behavior in the numerical simulation will be the same for the two error measures
as generically they will just differ by a dimensional factor.


Figures 2 demonstrates that the convergence analysis can
be divided into two stages: the spectral and the convergence
stage. We see that in the first stage (1 _≤_ _t_ ≲ 3000), the
first _r_ tensor-columns of _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [learn the tensor column]
subspace corresponding to the first _r_ singular-tubes of the
tensor _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ), i.e. the principal angle between the
tensor column subspaces _**L**_ _t_ and _**L**_ becomes small. Namely,
as one can observe in Figure 2 (bottom), the principal angle
between the two subspaces, _∥_ _**V**_ _[⊤]_ _**L**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥]_ [, decreases where]
as the principal angle between _**X**_ and _**L**_ _t_ reaches certain
plateau, see the behavior of _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥]_ [.] [At] [the] [same]

time, test errors _[∥]_ _**[U]**_ _[t][∗]_ _∥_ _**[U]**_ _**X**_ _t_ _[⊤]_ _∗_ _[−]_ _**X**_ _**[X]**_ _[⊤][∗]_ _∥_ _**[X]**_ _F_ _[ ⊤][∥][F]_ and _[∥][σ][r]_ [(] _∥_ _**[U]**_ _σ_ _[t]_ _r_ [)] ( _[−]_ _**X**_ _[σ]_ ) _[r]_ _∥_ [(] 2 _**[X]**_ [)] _[∥]_ [2]

stay large. In the second stage, we see that the test error
_∥_ _**U**_ _t∗∥_ _**UX**_ _[⊤]_ _t∗_ _[−]_ _**X**_ _**[X]**_ _[⊤][∗]_ _∥_ _**[X]**_ _F_ _[ ⊤][∥][F]_ starts decreasing, meaning that the gra
dient descent iterates _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [start converging to] _**[ X]**_ _[∗]_ _**[X]**_ _[ ⊤]_

by learning more about the tensor-column subspace of the
ground truth tensor. At the same time, the test error over
_r_ th singular tube _[∥][σ][r]_ [(] _∥_ _**[U]**_ _σ_ _[t]_ _r_ [)] ( _[−]_ _**X**_ _[σ]_ ) _[r]_ _∥_ [(] 2 _**[X]**_ [)] _[∥]_ [2] starts decreasing too and

as a result converges to zero. We also see that in this stage
the principal angle between _**L**_ _t_ and _**L**_ grows, which is also
intuitive as the tensor-column subspace _**L**_ does not have
the full information about the tensor-column subspace of



the ground truth tensor _**X**_ _∗_ _**X**_ _[⊤]_, and learning more about
_**X**_ _∗_ _**X**_ _[⊤]_ leads to a larger error in terms of principal angles
of the two.


**Depiction** **of** **the** **alignment** **stage.** In this experiment,
we illustrate that gradient descent with small initialization
behaves similarly to the tensor-power method modulo normalization in the first few iterations, bringing the gradient
iterates close to the spectral tubal initialization, used, e.g., in
(Liu et al., 2024). Here, as before _**L**_ denote the tensor subspace spanned by the tensor-columns corresponding to the
first _r_ singular-tubes of tensor _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ) and _**L**_ _t_ is the
tensor-column subspace corresponding to the first _r_ singular
tubes _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [.] [Additionally,] _**[L]**_ [�] _[t]_ [denotes the tensor-column]
subspace spanned by the first _r_ singular-tubes of the ten
sor _**U**_ [�] _t ∗_ _**U**_ [�] _⊤t_ [, where] _**[U]**_ [�] _⊤t_ [=] - _**I**_ + _A_ _[∗]_ _A_ - _**X**_ _∗_ _**X**_ _[⊤]_ [��] _[∗][t]_ _∗_ _**U**_ 0.

In Figure 2 (bottom), we see that _**U**_ _t_ and _**U**_ [�] _t_ learn the
subspace _**L**_ almost at the same rate in the first iterations,
1 _≤_ _t_ ≲ 3000. In the same figure, we observe that also
the angle between _**V**_ _**X**_ and _**L**_ _t_, respectively _**L**_ [�] _t_, decreases
monotonically in the spectral stage. Then at the beginning
of the convergence stage, 3000 ≲ _t_, the angle between _**V**_ _**X**_
and _**L**_ _t_ starts decreasing gradually and converges to zero, as
expected since _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [converges to] _**[ X]**_ _[ ∗]_ _**[X]**_ _[ ⊤]_ [.] [Whereas the]
principal angle between _**L**_ and _**L**_ _t_ growths until it reaches a
certain plateau.


Figure 4: Impact of different initialization scales on the test
and the training error. The data are represented in the log-log
plot. We set _n_ = 10 _, k_ = 4 _, r_ = 3 in the ground truth tensor
_**X**_ _∈_ R _[n][×][r][×][k]_ and for initialization _**U**_ 0 = _α_ _**U**_ _∈_ R _[n][×][R][×][k]_

with _R_ = 200 and different scales of _α_ . The plot depicts the
averaged value for five runs and the bars represent the deviations from the mean value. For illustration, we also depict
the theoretical test error bound obtained in Theorem 3.1. As
one can see, the numerical error resembles the theoretical
21
behavior of _Cn,k,r,κ · α_ 16 .



8


**Implicit Regularization for Tubal Tensors via GD**



**Test and train error under different scales of initializa-**
**tion.** In this experiment, we explore the influence of the
initialization scale, denoted by _α_, on the training and the test
error. With _R_ = 200, we apply gradient descent for various
values of _α_, halting the iterations at _t_ = 3500 in each run.
The results, presented in Figure 4, demonstrate a reduction
in test error as _α_ decreases. Notably, the figure indicates that
the test error follows an almost polynomial relationship with
the initialization scale _α_ . This observation is consistent with
our theoretical predictions, which also forecast a decrease
in test error at a rate of _α_, see Theorem 3.1.


**Impact of different levels of overparameterization on the**
**convergence.** In this numerical analysis, we set _α_ = 10 _[−]_ [7]

and examined the convergence speed of gradient descent
to the ground truth tensor for various overparameterization
rates _R_ . We run the experiment twenty times for each value
of _R_ and plot the averaged values per each iteration. The
results, shown in Figure 3, reveal that increasing the number
of tensor columns _R_, that is, overparameterizing, accelerates
the convergence rate, resulting in fewer iterations to reach
the desired error level. Additionally, overparameterization
reduces the test error and the training error by affecting the
spectral stages.


**6. Conclusion and Outlook**


In this paper, we focused on studying the implicit regularization of tubal tensor factorizations via gradient descent
by showing that with small random initialization and overparametrization, the gradient descent algorithm is biased
towards a low-tubal-rank solution. We have shown that the
first iterations of gradient descent with small random initialization behave similarly to the tensor power method, which
leads to learning in these first iterations the tensor-column
spaces close to the tensor-column space of the ground truth.
We also demonstrate that the implicit regularization from
small random initialization guides the gradient descent iterations toward low-tubal rank solutions that are not only
globally optimal but also generalize well.


**Acknowledgments**


AV and FK acknowledge support by the German Science
Foundation (DFG) in the context of the collaborative research center TR-109, the Emmy Noether junior research
group KR 4512/1-1 and the Bavarian Funding Program
for Initiating International Research Cooperation, as well
as by the Munich Data Science Institute and Munich Center for Machine Learning. SK acknowledges support by
the United States National Science Foundation in the context of the Foundations of Data Science Institute funded by
grant NSF DMS 2022205. MI acknowledges support by
the United States National Science Foundation grants NSF



DMS 2108479 and NSF EDU DGE 2152014.


**Impact Statement**


This paper presents work whose goal is to advance the field
of Machine Learning, and more specifically, the theoretical
understanding of implicit regularization as a tool for structured recovery problems. There are many potential societal
consequences of our work, none which we feel must be
specifically highlighted here.


**References**


Ahmed, T., Raja, H., and Bajwa, W. U. Tensor regression
using low-rank and sparse tucker decompositions. _SIAM_
_Journal on Mathematics of Data Science_, 2(4):944–966,
2020.


Arora, S., Cohen, N., Hu, W., and Luo, Y. Implicit regularization in deep matrix factorization. _Advances in Neural_
_Information Processing Systems_, 32, 2019.


Bah, B., Rauhut, H., Terstiege, U., and Westdickenberg,
M. Learning deep linear neural networks: Riemannian
gradient flows and convergence to global minimizers.
_Information and Inference:_ _A Journal of the IMA_, 11(1):
307–353, 2022.


Candes, E. J., Li, X., and Soltanolkotabi, M. Phase retrieval via wirtinger flow: Theory and algorithms. _IEEE_
_Transactions on Information Theory_, 61(4):1985–2007,
2015.


Chou, H.-H., Maly, J., and Rauhut, H. More is less: inducing sparsity via overparameterization. _Information and_
_Inference:_ _A Journal of the IMA_, 12(3):1437–1460, 2023.


Chou, H.-H., Gieshoff, C., Maly, J., and Rauhut, H. Gradient descent for deep matrix factorization: Dynamics and
implicit bias towards low rank. _Applied and Computa-_
_tional Harmonic Analysis_, 68:101595, 2024.


Cohen, N., Sharir, O., and Shashua, A. On the expressive
power of deep learning: A tensor analysis. In _Conference_
_on learning theory_, pp. 698–728. PMLR, 2016.


Ge, R. and Ma, T. On the optimization landscape of tensor
decompositions. _Advances in neural information process-_
_ing systems_, 30, 2017.


Ge, R., Huang, F., Jin, C., and Yuan, Y. Escaping from saddle points—online stochastic gradient for tensor decomposition. In _Conference on learning theory_, pp. 797–842.
PMLR, 2015.


Geyer, K., Kyrillidis, A., and Kalev, A. Low-rank regularization and solution uniqueness in over-parameterized



9


**Implicit Regularization for Tubal Tensors via GD**



matrix sensing. In _International Conference on Artificial_
_Intelligence and Statistics_, pp. 930–940. PMLR, 2020.


Ghorbani, B., Mei, S., Misiakiewicz, T., and Montanari, A.
When do neural networks outperform kernel methods?
_Advances in Neural Information Processing Systems_, 33:
14820–14830, 2020.


Gleich, D. F., Greif, C., and Varah, J. M. The power and
arnoldi methods in an algebra of circulants. _Numerical_
_Linear Algebra with Applications_, 20(5):809–831, 2013.


Gunasekar, S., Woodworth, B. E., Bhojanapalli, S.,
Neyshabur, B., and Srebro, N. Implicit regularization
in matrix factorization. _Advances in neural information_
_processing systems_, 30, 2017.


Gunasekar, S., Lee, J. D., Soudry, D., and Srebro, N. Implicit bias of gradient descent on linear convolutional
networks. _Advances_ _in_ _neural_ _information_ _processing_
_systems_, 31, 2018.


Haselby, C., Iwen, M., Karnik, S., and Wang, R. Tensor deli:
Tensor completion for low cp-rank tensors via random
sampling, 2024.


Hou, J., Zhang, F., Qiu, H., Wang, J., Wang, Y., and Meng,
D. Robust low-tubal-rank tensor recovery from binary
measurements. _IEEE Transactions on Pattern Analysis_
_and Machine Intelligence_, 44(8):4355–4373, 2021.


Jin, C., Ge, R., Netrapalli, P., Kakade, S. M., and Jordan,
M. I. How to escape saddle points efficiently. In _Interna-_
_tional conference on machine learning_, pp. 1724–1732.
PMLR, 2017.


Kilmer, M. E. and Martin, C. D. Factorization strategies for
third-order tensors. _Linear Algebra and its Applications_,
435(3):641–658, 2011.


Kilmer, M. E., Braman, K., Hao, N., and Hoover, R. C.
Third-order tensors as operators on matrices: A theoretical and computational framework with applications in
imaging. _SIAM Journal on Matrix Analysis and Applica-_
_tions_, 34(1):148–172, 2013.


Kim, D. and Chung, H. W. Rank-1 matrix completion
with gradient descent and small random initialization.
_Advances in Neural Information Processing Systems_, 36,
2024.


Kolb, C., Muller, C. L., Bischl, B., and R¨ ugamer, D. Smooth-¨
ing the edges: A general framework for smooth optimization in sparse regularization using hadamard overparametrization. _arXiv preprint arXiv:2307.03571_, 2023.


Kong, H., Xie, X., and Lin, Z. t-schatten- _p_ norm for lowrank tensor recovery. _IEEE Journal of Selected Topics in_
_Signal Processing_, 12(6):1405–1419, 2018.



Li, Y., Ma, T., and Zhang, H. Algorithmic regularization in
over-parameterized matrix sensing and neural networks
with quadratic activations. In _Conference On Learning_
_Theory_, pp. 2–47. PMLR, 2018.


Li, Z., Luo, Y., and Lyu, K. Towards resolving the implicit
bias of gradient descent for matrix factorization: Greedy
low-rank learning. _arXiv_ _preprint_ _arXiv:2012.09839_,
2020.


Li, Z., You, C., Bhojanapalli, S., Li, D., Rawat, A. S., Reddi,
S. J., Ye, K., Chern, F., Yu, F., Guo, R., et al. The lazy
neuron phenomenon: On emergence of activation sparsity
in transformers. _arXiv preprint arXiv:2210.06313_, 2022.


Ling, S. and Strohmer, T. Regularized gradient descent: a
non-convex recipe for fast joint blind deconvolution and
demixing. _Information and Inference:_ _A Journal of the_
_IMA_, 8(1):1–49, 2019.


Liu, X.-Y., Aeron, S., Aggarwal, V., and Wang, X. Lowtubal-rank tensor completion using alternating minimization. _IEEE Transactions on Information Theory_, 66(3):
1714–1737, 2019.


Liu, X.-Y., Aeron, S., Aggarwal, V., and Wang, X. Lowtubal-rank tensor completion using alternating minimization. _IEEE Transactions on Information Theory_, 66(3):
1714–1737, 2020. doi: 10.1109/TIT.2019.2959980.


Liu, Z., Han, Z., Tang, Y., Zhao, X.-L., and Wang, Y. Lowtubal-rank tensor recovery via factorized gradient descent.
_arXiv preprint arXiv:2401.11940_, 2024.


Ma, C., Wang, K., Chi, Y., and Chen, Y. Implicit regularization in nonconvex statistical estimation: Gradient descent
converges linearly for phase retrieval and matrix completion. In _International Conference on Machine Learning_,
pp. 3345–3354. PMLR, 2018.


Novikov, A., Podoprikhin, D., Osokin, A., and Vetrov, D. P.
Tensorizing neural networks. _Advances in neural infor-_
_mation processing systems_, 28, 2015.


Raginsky, M., Rakhlin, A., and Telgarsky, M. Non-convex
learning via stochastic gradient langevin dynamics: a
nonasymptotic analysis. In _Conference on Learning The-_
_ory_, pp. 1674–1703. PMLR, 2017.


Razin, N., Maman, A., and Cohen, N. Implicit regularization in tensor factorization. In _International Conference_
_on Machine Learning_, pp. 8913–8924. PMLR, 2021.


Razin, N., Maman, A., and Cohen, N. Implicit regularization in hierarchical tensor factorization and deep convolutional neural networks. In _International Conference on_
_Machine Learning_, pp. 18422–18462. PMLR, 2022.



10


**Implicit Regularization for Tubal Tensors via GD**



Rudelson, M. and Vershynin, R. Smallest singular value of a
random rectangular matrix. _Communications on Pure and_
_Applied Mathematics:_ _A Journal Issued by the Courant_
_Institute of Mathematical Sciences_, 62(12):1707–1739,
2009.


Soltanolkotabi, M., Stoger,¨ D., and Xie, C. Implicit balancing and regularization: Generalization and convergence guarantees for overparameterized asymmetric matrix sensing. In _The Thirty Sixth Annual Conference on_
_Learning Theory_, pp. 5140–5142. PMLR, 2023.


Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S., and
Srebro, N. The implicit bias of gradient descent on separable data. _Journal of Machine Learning Research_, 19
(70):1–57, 2018.


Stoger, D. and Soltanolkotabi, M.¨ Small random initialization is akin to spectral learning: Optimization and generalization guarantees for overparameterized low-rank
matrix reconstruction. _Advances in Neural Information_
_Processing Systems_, 34:23831–23843, 2021.


Tao, T. and Vu, V. Random matrices: The distribution of
the smallest singular values. _Geometric And Functional_
_Analysis_, 20:260–297, 2010.


Tu, S., Boczar, R., Simchowitz, M., Soltanolkotabi, M.,
and Recht, B. Low-rank solutions of linear matrix equations via procrustes flow. In _International Conference on_
_Machine Learning_, pp. 964–973. PMLR, 2016.


Vaskevicius, T., Kanade, V., and Rebeschini, P. Implicit
regularization for optimal sparse recovery. _Advances in_
_Neural Information Processing Systems_, 32, 2019.


Vershynin, R. _High-dimensional probability:_ _An introduc-_
_tion with applications in data science_, volume 47. Cambridge university press, 2018.


Wang, X., Wu, C., Lee, J. D., Ma, T., and Ge, R. Beyond
lazy training for over-parameterized tensor decomposition. _Advances in Neural Information Processing Systems_,
33:21934–21944, 2020.


Wedin, P.-A. [˚] Perturbation bounds in connection with singular value decomposition. _BIT Numerical Mathematics_,
12:99–111, 1972.


Wind, J. S. Asymmetric matrix sensing by gradient descent with small random initialization. _arXiv_ _preprint_
_arXiv:2309.01796_, 2023.


Woodworth, B., Gunasekar, S., Lee, J. D., Moroshko, E.,
Savarese, P., Golan, I., Soudry, D., and Srebro, N. Kernel and rich regimes in overparametrized models. In
_Conference on Learning Theory_, pp. 3635–3673. PMLR,
2020.



Zhang, F., Wang, W., Hou, J., Wang, J., and Huang, J.
Tensor restricted isometry property analysis for a large
class of random measurement ensembles. _arXiv preprint_
_arXiv:1906.01198_, 2019.



11


**Implicit Regularization for Tubal Tensors via GD**

# **Supplementary Material**


**A. Outline of Appendices**


For ease of organization, we divide the supplementary material into appendices as follows. In Appendix B, we define some
additional notation, including the angles between two tensor-column subspaces. In Appendix C, we decompose the gradient
descent iterates into a “signal” term and a “noise” term, which will aid us in our analysis. In Appendices D and E, we
analyze the spectral and convergence stages, respectively, of the gradient descent iterations. In Appendix F, we prove our
main result.


To avoid breaking up the flow of our analysis, we put some technical lemmas in the last few appendices instead of in
the previously mentioned appendices. In Appendix G, we prove some properties of measurement operators which satisfy
the restricted isometry property. In Appendix H, we prove some properties of matrices and their subspaces. Finally, in
Appendix I, we prove some properties of random Gaussian tubal tensors.


**B. Additional Notation**


For a tensor _**Y**_ _∈_ R _[n][×][r][×][k]_, we denote its t-SVD by _**Y**_ = _**V**_ _**Y**_ _∗_ **Σ** _**Y**_ _∗_ _**W**_ _[⊤]_ _**Y**_ [with] [the] [two] [orthogonal] [tensor]
_**V**_ _**Y**_ _,_ _**W**_ _**Y**_ _∈_ R _[n][×][r][×][k]_, and the f-diagonal tensor **Σ** _**Y**_ _∈_ R _[r][×][r][×][k]_ . We will refer to _**V**_ _**Y**_ as the tensor-column subspace
of _**Y**_ and by _**V**_ _**Y**_ _⊥_ _∈_ R _[n][×]_ [(] _[n][−][r]_ [)] _[×][k]_ we denote the tensor-column subspace orthogonal to _**V**_ _**Y**_ with its projection operator
_**V**_ _**Y**_ _⊥_ _∗_ _**V**_ _[⊤]_ _**Y**_ _[⊥]_ [=] _[ I −]_ _**[V]**_ _**[Y]**_ _[∗]_ _**[V]**_ _[⊤]_ _**Y**_ [.]

We measure the angles between two tensor-column subspaces _**Y**_ 1 and _**Y**_ 2 by the tensor-spectral norm _∥_ _**V**_ _**Y**_ _⊥_ 1 _[∗]_ _**[V]**_ _**[Y]**_ [2] _[∥]_ [which]
according to (Liu et al., 2019; Gleich et al., 2013; Kilmer & Martin, 2011) is equal to


_∥_ _**V**_ _[⊤]_ _**Y**_ _[⊥]_ 1 _[∗]_ _**[V]**_ _**[Y]**_ [2] _[∥]_ [=] _[ ∥]_ _**[V]**_ _**Y**_ _[⊤][⊥]_ 1 _[∗]_ _**[V]**_ _**[Y]**_ [2] _[∥]_ [=] �� _**V**_ _⊤_ _**Y**_ _[⊥]_ 1 _**[V]**_ _**[Y]**_ [2] �� _._

which means that the largest principal angle between _**Y**_ 1 and _**Y**_ 2 equals to that of these two subspaces represented in the
Fourier domain. In the Fourier domain, since _**V**_ _[⊤]_ _**Y**_ _[⊥]_ 1 _[∈]_ [C][(] _[n][−][r]_ [)] _[k][×][nk]_ [and] _**[ V]**_ _**[Y]**_ [2] _[∈]_ [C] _[nk][×][nk]_ [are block diagonal matrices, it holds]
that







_**V**_ _**Y**_ 2 [(] _[k]_ [)]



_**V**_ _[⊤]_ _**Y**_ _[⊥]_ 1 (1)



�����������



_**V**_ _[⊤]_ _**Y**_ _[⊥]_ 1 (2)



_**V**_ _[⊤]_ _**Y**_ _[⊥]_ 1 ( _k_ )



�� _**V**_ _⊤_ _**Y**_ _[⊥]_ 1 _**[V]**_ _**[Y]**_ [2] �� =



�����������












_..._



_**V**_ _**Y**_ 2 (2)



�� _**V**_ _⊤_ _**Y**_ _[⊥]_ 1 ( _j_ ) _**V**_ _**Y**_ 2 ( _j_ ) [�] 












= max
1 _≤j≤k_



 _**V**_ _**Y**_ 2 (1)









_..._















**C. Signal Decomposition**


Recall that the gradient descent iterates are defined in (3.4) as


_**U**_ _t_ +1 = _**U**_ _t −_ _µ∇ℓ_ ( _**U**_ _t_ )


                     - ��
= _**U**_ _t_ + _µA_ _[∗]_ [�] _**y**_ _−A_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ _∗_ _**U**_ _t_

=           - _**I**_ + _µ_ ( _A_ _[∗]_ _A_ )           - _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ �� _∗_ _**U**_ _t._


For the ground truth tensor _**X**_ _∈_ R _[n][×][r][×][k]_, consider its tensor-column subspace _**V**_ _**X**_ with the corresponding basis _**V**_ _**X**_ _∈_
R _[n][×][r][×][k]_ . Consider the tensor _**V**_ _**X**_ _∗_ _**U**_ _t_ _∈_ R _[r][×][R][×][k]_ with its t-SVD decomposition _**V**_ _**X**_ _∗_ _**U**_ _t_ = _**V**_ _t_ _∗_ **Σ** _t_ _∗_ _**W**_ _[⊤]_ _t_ [.] [For]
_**W**_ _t_ _∈_ R _[R][×][r][×][k]_, we denote by _**W**_ _t,⊥_ _∈_ R _[R][×]_ [(] _[n][−][r]_ [)] _[×][k]_ a tensor whose tensor-column subspace is orthogonal to those of _**W**_ _t_,
that is _∥_ _**W**_ _[⊤]_ _t,⊥_ _[∗]_ _**[W]**_ _[t][∥]_ [=] [0][ and its projection operator] _**[ P]**_ _**[W]**_ _t,⊥_ [is defined as] _**[ P]**_ _**[W]**_ _t,⊥_ [=] _**[ W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ [=] _[ I −]_ _**[W]**_ _[t]_ _[∗]_ _**[W]**_ _[⊤]_ _t_ [.]
We then decompose the gradient descent iterates _**U**_ _t_ as follows

_**U**_ _t_ = _**U**_ _t ∗_ _**W**_ _t ∗_ _**W**_ _[⊤]_ _t_ [+] _**[ U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ (C.1)

We will refer to the tensors _**U**_ _t∗_ _**W**_ _t∗_ _**W**_ _[⊤]_ _t_ [as the signal term of the gradient descent iterates, and the tensors] _**[ U]**_ _[t][∗]_ _**[W]**_ _[t,][⊥][∗]_ _**[W]**_ _t,_ _[⊤]_ _⊥_
will be named as the noise term.


12


**Implicit Regularization for Tubal Tensors via GD**


**Lemma C.1.** _The tensor-column space of the noise term_ _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _[⊤]_ _t,⊥_ _[is orthogonal to the tensor-column subspace]_
_of the_ _**X**_ _, namely_ _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ [= 0] _[.]_ _[Moreover, if]_ _**[ V]**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ _[is full tubal-rank with all invertible singular tubes,]_
_then the signal term_
_**U**_ _t ∗_ _**W**_ _t ∗_ _**W**_ _[⊤]_ _t_

_has tubal-rank r with all invertible singular tubes and the noise term has tubal rank at most R −_ _r._


_Proof._ _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ [=] _**[ V]**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ _[∗]_ [(] _[I −]_ _**[W]**_ _[t]_ _[∗]_ _**[W]**_ _[⊤]_ _t_ [) =] _**[ V]**_ _**X**_ _[⊤]_ _[∗]_ _**[U]**_ _[t]_ _[−]_ _**[V]**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t]_ _[∗]_ _**[W]**_ _[⊤]_ _t_ [= 0] _[ ∈]_ [R] _[r][×][R][×][k]_ [.]
The second part follows fact that if _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [is full tubal rank with all invertible singular tubes then all the slices in the]
Fourier have full rank.


**D. Analysis of the Spectral Stage**


The goal of this section is to show that the first few iterations of the gradient descent algorithm can be approximated by the
iteration of the tensor power method modulo normalization defined as


            -             - _∗t_
_**U**_         - _t_ = _**I**_ + _µA_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ) _∗_ _**U**_ 0 = _**Z**_ _t ∗_ _**U**_ 0 _∈_ R _[n][×][R][×][k]_ _._


                -                 - _∗t_
with the tensor power method iteration _**Z**_ _t_ =: _**I**_ + _µA_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ) _∈_ R _[n][×][n][×][k]_ _._ Moreover, this will result in the

feature that after the first few iterations, the tensor-column span of the signal term _**U**_ _t ∗_ _**W**_ _t ∗_ _**W**_ _[⊤]_ _t_ [becomes aligned with]
the tensor-column span of _**X**_, and that the noise term _**U**_ _t ∗_ _**W**_ _t,⊥_ is relatively small compared to signal term in terms of the
norm, indicating that the signal term dominates the noise term.


For this, let us denote the difference between the power method and the gradient descent iterations by


_**E**_ _t_ := _**U**_ _t −_ _**U**_ [�] _t._ (D.1)


For convenience, throughout this section, we will denote by _**M**_ the tensor _**M**_ := _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ) _∈_ R _[n][×][n][×][k]_, so that
_**U**_ - _t_ = ( _I_ + _µ_ _**M**_ ) _[∗][t]_ _∗_ _**U**_ 0 and _**Z**_ _t_ = ( _I_ + _µ_ _**M**_ ) _[∗][t]_ .


In the first result of this section, the following lemma, we show that _**E**_ _t_ can be made small via an appropriate initialization
scale.

**Lemma D.1.** _Suppose that A_ : _S_ _[n][×][n][×][k]_ _→_ R _[m]_ _satisfies RIP_ (2 _, δ_ 1) _and let t_ _[⋆]_ _be defined as_


              -               _t_ _[⋆]_ = min _j_ _∈_ N : _∥_ _**U**_ [�] _j−_ 1 _−_ _**U**_ _j−_ 1 _∥_ _> ∥_ _**U**_ [�] _j−_ 1 _∥_ _._ (D.2)


_Then for all integers t such that_ 1 _≤_ _t ≤_ _t_ _[⋆]_ _it holds that_



_√_
_∥_ _**E**_ _t∥_ = _∥_ _**U**_ _t −_ _**U**_ [�] _t∥≤_ 8(1 + _δ_ 1



_k_ )� _k_ min _{n, R}_ _[α]_ [3] (D.3)

_∥_ _**M**_ _∥_ _[∥]_ _**[U]**_ _[∥]_ [3][(1 +] _[ µ][∥]_ _**[M]**_ _[∥]_ [)][3] _[t][.]_



_Proof._ Similarly to the matrix case in (Stoger & Soltanolkotabi, 2021), in the tubal tensor case it can be shown that for¨
_t ≥_ 1, the difference tensor _**E**_ _t_ = _**U**_ _t −_ _**U**_ [�] _t_ can be represented as



_**E**_ _t_ = _**U**_ _t −_ _**U**_ [�] _t_ =



_t_
�( _**I**_ + _µ_ _**M**_ ) _[∗]_ [(] _[t][−][j]_ [)] _**E**_ [�] _j_ (D.4)


_j_ =1



with _**E**_ [�] _j_ = _µA_ _[∗]_ _A_ - _**U**_ _j−_ 1 _∗_ _**U**_ _[⊤]_ _j−_ 1� _∗_ _**U**_ _j−_ 1. To estimate _∥_ _**E**_ _t∥_, we will first estimate each summand in (D.4) separately. First,
we can proceed with the following simple estimation


_∥_ ( _**I**_ + _µ_ _**M**_ ) _[∗]_ [(] _[t][−][j]_ [)] _**E**_ [�] _j∥≤∥_ ( _**I**_ + _µ_ _**M**_ ) _∥_ [(] _[t][−][j]_ [)] _∥_ _**E**_ [�] _j∥≤_ �1 + _µ∥_ _**M**_ _∥_ �( _t−j_ ) _∥_ _**E**_         - _j∥._


Now, for _∥_ _**E**_ [�] _j∥_, using the fact that the spectral norm of tubal tensors is sub-multiplicative, we get that


_∥_ _**E**_ [�] _j∥_ = _µ∥A_ _[∗]_ _A_         - _**U**_ _j−_ 1 _∗_ _**U**_ _[⊤]_ _j−_ 1� _∗_ _**U**_ _j−_ 1 _∥≤_ _µ∥A_ _[∗]_ _A_         - _**U**_ _j−_ 1 _∗_ _**U**_ _[⊤]_ _j−_ 1� _∥· ∥_ _**U**_ _j−_ 1 _∥._


13


**Implicit Regularization for Tubal Tensors via GD**



_√_
Since operator _A_ satisfies RIP(2 _, δ_ 1), by Lemma G.3, _A_ also satisfies S2NRIP( _δ_ 1



Since operator _A_ satisfies RIP(2 _, δ_ 1), by Lemma G.3, _A_ also satisfies S2NRIP( _δ_ 1 _k_ ), which provides the following

estimate
_√_ _√_
_∥A_ _[∗]_ _A_       - _**U**_ _j−_ 1 _∗_ _**U**_ _[⊤]_ _j−_ 1� _∥≤_ (1 + _δ_ 1 _k_ ) _∥_ _**U**_ _j−_ 1 _∗_ _**U**_ _[⊤]_ _j−_ 1 _[∥][∗]_ [= (1 +] _[ δ]_ [1] _k_ ) _∥_ _**U**_ _j−_ 1 _∥_ [2] _F_ _[.]_



_√_
_k_ ) _∥_ _**U**_ _j−_ 1 _∗_ _**U**_ _[⊤]_ _j−_ 1 _[∥][∗]_ [= (1 +] _[ δ]_ [1]



_k_ ) _∥_ _**U**_ _j−_ 1 _∥_ [2] _F_ _[.]_



All this together leads to


_√_
_∥_ _**E**_ _t∥_ = _∥_ _**U**_ _t −_ _**U**_ [�] _t∥≤_ _µ_ (1 + _δ_ 1



_t_



_j_ =1



_k_ )



�1 + _µ∥_ _**M**_ _∥_ �( _t−j_ ) _∥_ _**U**_ _j−_ 1 _∥_ 2 _F_ _[∥]_ _**[U]**_ _[j][−]_ [1] _[∥][.]_ (D.5)



From here, we want to bound _∥_ _**E**_ _t∥_ in terms of the initialization scale _α_ and the data-related norm _∥_ _**M**_ _∥_ . For this, we first
use the fact that the tensor Frobenius norm above can be bounded as _∥_ _**U**_ _j−_ 1 _∥F_ _≤_ - _k_ min _{n, R}∥_ _**U**_ _j−_ 1 _∥_ . Then since for
all 1 _≤_ _j_ _≤_ _t_ _[⋆]_ we have _∥_ _**U**_ [�] _j−_ 1 _−_ _**U**_ _j−_ 1 _∥≤∥_ _**U**_ [�] _j−_ 1 _∥_, the spectral norm of _**U**_ _j−_ 1 can be bounded as


_∥_ _**U**_ _j−_ 1 _∥≤∥_ _**U**_ [�] _j−_ 1 _∥_ + _∥_ _**U**_ _j−_ 1 _−_ _**U**_ [�] _j−_ 1 _∥≤_ 2 _∥_ _**U**_ [�] _j−_ 1 _∥._


This gives us the following upper bound



_√_
_∥_ _**E**_ _t∥≤_ 8 _µ_ (1 + _δ_ 1



_k_ )� _k_ min _{n, R}_



_t_
�(1 + _µ∥_ _**M**_ _∥_ ) _[t][−][j]_ _∥_ _**U**_ [�] _j−_ 1 _∥_ [3] _._ (D.6)


_j_ =1



As for iterations of the tensor power method, it holds that


_∥_ _**U**_ [�] _j−_ 1 _∥_ = _∥_ ( _**I**_ + _µ_ _**M**_ ) _[∗]_ [(] _[j][−]_ [1)] _∗_ _**U**_ 0 _∥≤∥_ ( _**I**_ + _µ_ _**M**_ ) _[∗]_ [(] _[j][−]_ [1)] _∥∥_ _**U**_ 0 _∥≤_ (1 + _µ∥_ _**M**_ _∥_ ) _[j][−]_ [1] _∥_ _**U**_ 0 _∥_ = _α_ (1 + _µ∥_ _**M**_ _∥_ ) _[j][−]_ [1] _∥_ _**U**_ _∥,_


we can proceed with (D.6) as follows



_√_ _t_
_∥_ _**E**_ _t∥≤_ 8 _µ_ (1 + _δ_ 1 _k_ )� _k_ min _{n, R}α_ [3] _∥_ _**U**_ _∥_ [3] �(1 + _µ∥_ _**M**_ _∥_ ) _[t]_ [+2] _[j][−]_ [3] _._


_j_ =1



Now, the sum on the right-hand side can be estimated as



_t_




_t_ _t_
�(1 + _µ∥_ _**M**_ _∥_ ) _[t]_ [+2] _[j][−]_ [3] = (1 + _µ∥_ _**M**_ _∥_ ) _[t][−]_ [1] 

_j_ =1 _j_ =1



_t_

- [(1 +] _[ µ][∥]_ _**[M]**_ _[∥]_ [)][2] _[t][ −]_ [1]

(1 + _µ∥_ _**M**_ _∥_ ) [2] _[j][−]_ [2] = (1 + _µ∥_ _**M**_ _∥_ ) _[t][−]_ [1]

(1 + _µ∥_ _**M**_ _∥_ ) [2] _−_ 1

_j_ =1



(1 + _µ∥_ _**M**_ _∥_ ) [2] _−_ 1




[(1 +] _[ µ][∥]_ _**[M]**_ _[∥]_ [)][2] _[t][ −]_ [1]
= (1 + _µ∥_ _**M**_ _∥_ ) _[t][−]_ [1]



_,_
_µ∥_ _**M**_ _∥_




[(1 +] _[ µ][∥]_ _**[M]**_ _[∥]_ [)][2] _[t]_ [1]

_µ∥_ _**M**_ _∥_ (2 + _µ∥_ _**M**_ _∥_ ) _[≤]_ [(1 +] _µ_ _[ µ]_ _∥_ _[∥]_ _**M**_ _**[M]**_ _∥_ _[∥]_ [)][3] _[t]_



which gives us the final estimation for the norm of _**E**_ _t_ as follows



_√_
_∥_ _**E**_ _t∥≤_ 8(1 + _δ_ 1



_k_ )� _k_ min _{n, R}_ _[α]_ [3]

_∥_ _**M**_ _∥_ _[∥]_ _**[U]**_ _[∥]_ [3][(1 +] _[ µ][∥]_ _**[M]**_ _[∥]_ [)][3] _[t]_



and finishes the proof.


The following lemma provides a lower bound for _t_ _[⋆]_, indicating the duration for which the approximation in Lemma D.1
remains valid.

**Lemma D.2.** _Consider tensors_ _**M**_ := _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ) _∈_ R _[n][×][n][×][k]_ _and_ _**U**_ [�] _t_ := ( _**I**_ + _µ_ _**M**_ ) _[∗][t]_ _∗_ _**U**_ 0 _._ _Let_ _**M**_ _∈_ C _[nk][×][nk]_ _be_
_the corresponding block diagonal form of the tensor_ _**M**_ _with the leading eigenvector v_ 1 _∈_ C _[nk]_ _, then_







(D.7)




2 ln (1 + _µ∥_ _**M**_ _∥_ )


14



_t_ _[⋆]_ _≥_








 - H
ln ~~_√_~~ _∥_ _**M**_ _∥·∥_ _**U**_ 0 _v_ 1 _∥ℓ_ 2
8(1+ _δ_ 1 _k_ ) ~~_[√]_~~ _k_ min _{n,R}_



_k_ ) ~~_[√]_~~



_k_ min _{n,R}α_ [3] _∥_ _**U**_ _∥_ [3]


**Implicit Regularization for Tubal Tensors via GD**



_Proof._ Let _**U**_ [�] _t_ _∈_ C _[nk][×][Rk]_ be the corresponding block diagonal form of tensor _**U**_ [�] _t_ . By the definition of the spectral tensor



norm, we have _∥_ _**U**_ [�] _t∥_ = _∥_ _**U**_ [�] _t∥_ and the definition of the matrix norm gives _∥_ _**U**_ [�] _t∥≥_ ��� _**U**_ _t_

version of _**U**_ [�] _t_, the following properties (see, e.g., (Liu et al., 2019)) holds



H
_v_ 1�� _ℓ_ 2 [.] [For] [the] [block] [diagonal]



_t_
_**U**_        - _t_ = ( _**I**_ + _µ_ _**M**_ ) _∗t_ _∗_ _**U**_ 0 = ( _**I**_ + _µ_ _**M**_ ) _∗t ·_ _**U**_ 0 = ( _**I**_ + _µ_ _**M**_ ) _·_ _**U**_ 0 _._ (D.8)


This allows us to proceed as follows

_**U**_      - _t_ H _v_ 1 = �( _**I**_ + _µ_ _**M**_ ) _t ·_ _**U**_ 0�H _v_ 1 = _**U**_ 0H( _**I**_ + _µ_ _**M**_ ) _t_ [H] _v_ 1 = (1 + _µ∥_ _**M**_ _∥_ ) _t_ _**U**_ 0H _v_ 1 _,_


where for the last equality we used the fact that block-diagonal matrix ( _**I**_ + _µ_ _**M**_ ) has the same set of eigenvectors as matrix



_**M**_ . From here, we get _∥_ _**U**_ [�] _t∥≥_ ��� _**U**_ _t_



H H
_v_ 1�� _ℓ_ 2 [= (1 +] _[ µ][∥]_ _**[M]**_ _[∥]_ [)] _[t]_ [��] _**[U]**_ [0] _v_ 1�� _ℓ_ 2 [.] [Then, applying Lemma D.1, the relative error in]



the spectral norm between _**U**_ [�] _t_ and _**U**_ _t_ can be estimated as



_∥_ _**U**_ [�] _t −_ _**U**_ _t∥_ _√_

_≤_ 8(1 + _δ_ 1
_∥_ _**U**_ [�] _t∥_



_∥_ _**U**_ _∥_ [3] (1 + _µ∥_ _**M**_ _∥_ ��)2 _t._


  


_k_ )




~~�~~ _k_ min _{n, R}α_ [3]

_∥_ _**M**_ _∥· ∥_ _**U**_ 0H _v_ 1�� _ℓ_ 2



Setting the bound above to be smaller than 1 and solving for _t_, we get



_k_ min _{n,R}α_ [3] _∥_ _**U**_ _∥_ [3]



ln




- H ~~_√_~~ _∥_ _**M**_ _∥·∥_ _**U**_ 0 _v_ 1� _ℓ_ 2
8(1+ _δ_ 1 _k_ ) ~~_[√]_~~ _k_ min _{n,R_



_k_ ) ~~_[√]_~~



_t <_



_._
2 ln (1 + _µ∥_ _**M**_ _∥_ )



Since _t ∈_ N with _t ≤_ _t_ _[⋆]_ should be such that _[∥]_ _**[U]**_ [�] _[t][−]_ [1] _[−]_ _**[U]**_ _[t][−]_ [1] _[∥]_ _<_ 1, we can choose _t_ _[⋆]_ as the floor-value of the right-hand side

_∥_ _**U**_ [�] _t−_ 1 _∥_
above.



To show that the tensor column subspaces of the tensor power method iterates and the gradient descent iterates are aligned
after the alignment phase, we use the largest principal angle between two tensor-column subspaces as the potential function
for analysis. Borrowing the idea from (Gleich et al., 2013), we will show that the power method iteration in the tensor
domain can be transformed to the classical subspace iteration in the frequency domain.

For this, consider the power method iterates _**U**_ [�] _t_ = ( _**I**_ + _µ_ _**M**_ ) _[∗][t]_ _∗_ _**U**_ 0, the iterates _**Z**_ _t_ = ( _**I**_ + _µ_ _**M**_ ) _[∗][t]_ and the gradient
descent iterates _**U**_ _t_ represented as _**U**_ _t_ = _**U**_ [�] _t_ + _**E**_ _t_ = _**Z**_ _t ∗_ _**U**_ 0 + _**E**_ _t_ . All these tensors have their counterparts in the Fourier
domain, which we will denote respectively as _**U**_ [�] _t_, _**Z**_ _t_ and _**U**_ _t_ .

As before, consider _**M**_ = _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ) _∈_ R _[n][×][n][×][k]_ with its t-SVD _**M**_ = _**V**_ _**M**_ _∗_ **Σ** _**M**_ _∗_ _**W**_ _[⊤]_ _**M**_ [and its Fourier domain]
representative _**M**_ _∈_ C _[nk][×][nk]_ . We denote by _**L**_ _∈_ R _[n][×][r][×][k]_ the tensor column subspace spanned by the tensor columns
corresponding to the first _r_ singular tubes, that is _**L**_ := _**V**_ _**M**_ (: _,_ 1 : _r,_ :) _∈_ R _[n][×][r][×][k]_ . Note that _**L**_ is also the subspace spanned
by the tensor columns corresponding to the first _r_ singular tubes of the tensor _**Z**_ _t_ _∈_ R _[n][×][n][×][k]_ .


By _**L**_ _t_ _∈_ R _[n][×][n][×][k]_ we will donate the tensor-column subspace spanned by the tensor columns corresponding to the first _r_ singular tubes of the gradient descent iterates _**U**_ _t_ = _**Z**_ _t_ _∗_ _**U**_ 0 + _**E**_ _t_ . More concretely,
for _**U**_ _t_ = [�] _s_ _[R]_ =1 _**[V]**_ _**[U]**_ _t_ [(:] _[, s,]_ [ :)] _[ ∗]_ **[Σ]** _**[U]**_ _t_ [(] _[s, s,]_ [ :)] _[ ∗]_ _**[W]**_ _[⊤]_ _**U**_ _t_ [(:] _[, s,]_ [ :)] [and] [the] [corresponding] [Fourier] [domain] [representation] _**[U]**_ _[t]_ [=]

diag( _Ut_ [(1)] _, Ut_ [(2)] _, . . ., Ut_ [(] _[k]_ [)] ), where _Ut_ [(] _[j]_ [)] = [�] _ℓ_ _[σ]_ _ℓ_ [(] _[j]_ [)] _[v]_ _ℓ_ [(] _[j]_ [)] _[w]_ _ℓ_ [(] _[j]_ [)] H = _UU_ ( _jt_ ) [Σ][(] _U_ _[j]_ _t_ [)] _[W]_ [ (] _U_ _[j]_ _t_ [)] H, we define the corresponding new ten
sors _**L**_ _t_ := _**V**_ _**U**_ _t_ (: _,_ 1 : _r,_ :) _∈_ R _[n][×][r][×][k]_ and their Fourier domain representations



diag( _Ut_ [(1)] _, Ut_ [(2)] _, . . ., Ut_ [(] _[k]_ [)] ), where _Ut_ [(] _[j]_ [)] = [�]



_ℓ_ _[σ]_ _ℓ_ [(] _[j]_ [)] _[v]_ _ℓ_ [(] _[j]_ [)] _[w]_ _ℓ_ [(] _[j]_ [)]



H = _UU_ ( _jt_ ) [Σ][(] _U_ _[j]_ _t_ [)] _[W]_ [ (] _U_ _[j]_ _t_ [)]



_**L**_ _t_ = diag( _Lt_ (1) _, Lt_ (2) _, . . ., Lt_ ( _k_ )) (D.9)


**Lemma D.3.** _Consider the tensor iterates_ _**Z**_ _t_ = ( _**I**_ + _µ_ _**M**_ ) _[∗][t]_ _with its block-matrix representation_


_**Z**_ _t_ = _bdiag_ ( _**Z**_ _t_ ) = _diag_ ( _Zt_ (1) _, Zt_ (2) _, . . ., Zt_ ( _k_ )) _._ (D.10)


_and the tensors_


_**E**_ _t_ = _**U**_ _t −_ _**U**_ [�] _t_ _∈_ R _[n][×][R][×][k]_

_**U**_ 0 = _α_ _**U**_ _∈_ R _[n][×][R][×][k]_ _,_ _α >_ 0 _._


15


**Implicit Regularization for Tubal Tensors via GD**


_Assume that for each_ 1 _≤_ _j_ _≤_ _k, it holds that_


_σr_ +1( _Zt_ ( _j_ )) _∥_ _**U**_ _∥_ + _[∥]_ _**[E]**_ _[t][∥]_ _< σr_ ( _Zt_ ( _j_ )) _σmin_ ( _**V**_ _⊤_ _**L**_ _[∗]_ _**[U]**_ [)] _[.]_ (D.11)

_α_


_Then for each_ 1 _≤_ _j_ _≤_ _k, the following two inequalities hold_


_σr_           - _Ut_ ( _j_ ) [�] = _σr_           - _Zt_ ( _j_ ) _U_ 0( _j_ ) + _Et_ ( _j_ ) [�] _≥_ _ασr_ ( _Zt_ ( _j_ )) _σmin_ ( _**V**_ _⊤_ _**L**_ _[∗]_ _**[U]**_ [)] _[ −∥]_ _**[E]**_ _[t][∥][,]_ (D.12)

_σr_ +1� _Ut_ ( _j_ ) [�] = _σr_ +1� _Zt_ ( _j_ ) _U_ 0( _j_ ) + _Et_ ( _j_ ) [�] _≤_ _ασr_ +1( _Zt_ ( _j_ )) _∥_ _**U**_ _∥_ + _∥_ _**E**_ _t∥_ (D.13)


_Moreover, the principal angle between the tensor-column subspaces_ _**L**_ _and_ _**L**_ _t is bounded as follows_

_∥_ _**V**_ _[⊤]_ _**L**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥≤]_ [max] _ασr_ +1( _Zt_ [(] _[j]_ [)] ) _∥_ _**U**_ _∥_ + _∥_ _**E**_ _t∥_ (D.14)
1 _≤j≤k_ _σr_ ( _Zt_ [(] _[j]_ [)] ) _σmin_                    - _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_                    - _−_ _ασr_ +1� _Zt_ [(] _[j]_ [)] ) _∥_ _**U**_ _∥−∥_ _**E**_ _t∥_


_Proof._ For some _t ∈_ N, consider tensor _**Z**_ _t_ = ( _**I**_ + _µ_ _**M**_ ) _[∗][t]_ with its block-matrix representation



 _Zt_ [(1)]


(1) (2) ( _k_ ) 
_**Z**_ _t_ = bdiag( _**Z**_ _t_ ) = diag( _Zt_ _, Zt_ _, . . ., Zt_ ) = 






_Zt_ [(2)]



_..._









 _[.]_



_Zt_ [(] _[k]_ [)]


As we assume the symmetric tensor case scenario, the block-diagonal matrix representation _Zt_ consists of symmetric
matrices _Zt_ [(] _[j]_ [)] _∈_ C _[n][×][n]_ . At the same time, according to (Gleich et al., 2013), the gradient descent tensors _**U**_ _t_ = _**Z**_ _t_ _∗_ _**U**_ 0 + _**E**_ _t_
have their block-diagonal matrix representation



 _Et_ [(1)]



_Et_ [(1)]



_Et_ [(2)]



_**U**_ _t_ = _**Z**_ _t ∗_ _**U**_ 0 + _**E**_ _t_ _⇔_ _**Z**_ _t_ _**U**_ 0 + _**E**_ _t_ =



 _Zt_ [(1)] _U_ 0 [(1)]









_Zt_ [(2)] _U_ 0 [(2)]



_..._









 [+]










_..._









 _[.]_

_Et_ [(] _[k]_ [)]

(D.15)



_Zt_ [(] _[k]_ [)] _U_ 0 [(] _[k]_ [)]


Using Weyl’s inequality in each block, we have


_σr_       - _Zt_ ( _j_ ) _U_ 0( _j_ ) + _Et_ ( _j_ ) [�] _≥_ _σr_       - _Zt_ ( _j_ ) _U_ 0( _j_ ) [�] _−∥Et_ ( _j_ ) _∥≥_ _σr_ �( _V_ _**L**_ ( _j_ ))H _Zt_ ( _j_ ) _U_ 0( _j_ ) [�] _−∥Et_ ( _j_ ) _∥._


Now, for the singular value above we get the following estimation



_σr_ �( _V_ _**L**_ ( _j_ ))H _Zt_ ( _j_ ) _U_ 0( _j_ ) [�] = _σmin_ - _V_ _**L**_ ( _j_ ) [H] _Zt_ ( _j_ ) _V_ _**L**_ [(] _[j]_ [)] _[V]_ _**L**_ [(] _[j]_ [)] H _U_ 0( _j_ ) [�]


            - ( _j_ ) [H] ( _j_ ) ( _j_ ) [�]            - ( _j_ ) [H] ( _j_ ) [�]
_≥_ _σmin_ _V_ _**L**_ _Zt_ _V_ _**L**_ _σmin_ _V_ _**L**_ _U_ 0


( _j_ )                       - ( _j_ ) [H] ( _j_ ) [�] ( _j_ )                       - ( _j_ ) [H] ( _j_ ) [�]
= _σr_ ( _Zt_ ) _σmin_ _V_ _**L**_ _U_ 0 _≥_ _ασr_ ( _Zt_ ) _σmin_ _V_ _**L**_ _U_



= _ασr_ ( _Zt_ ( _j_ )) _σmin_ - _V_ _**L**_ [H]



( _j_ ) _U_ ( _j_ ) [�] _≥_ _ασr_ ( _Zt_ ( _j_ )) _σmin_ - _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ 


where in the last line we used that for each tensor it holds in the Fourier domain _V_ _**L**_ ( _j_ )H = _**V**_ T _**L**_ ( _j_ ).


To show inequality (D.13), we can use Weyl’s bounds and then the Courant-Fisher theorem, which leads to


_σr_ +1� _Zt_ ( _j_ ) _U_ 0( _j_ ) + _Et_ ( _j_ ) [�] _≤_ _σr_ +1� _Zt_ ( _j_ ) _U_ 0( _j_ ) [�] + _∥Et_ ( _j_ ) _∥≤_ _σr_ +1� _Zt_ ( _j_ ) _U_ 0( _j_ ) [�] + _∥_ _**E**_ _t∥_

_≤_ _σr_ +1� _Zt_ ( _j_ ) [�] _∥U_ 0( _j_ ) _∥_ + _∥_ _**E**_ _t∥≤_ _ασr_ +1� _Zt_ ( _j_ ) [�] _∥_ _**U**_ _∥_ + _∥_ _**E**_ _t∥._


Now, for estimation of _∥_ _**V**_ _[⊥]_ _**L**_ _[∗]_ _**[V]**_ _**[L]**_ _t_ _[∥]_ [, let us recall that] _**[ L]**_ [ is the tensor column subspace spanned by the tensor columns]
corresponding to the first _r_ singular tubes of tensor _**Z**_ _t_ = ( _**I**_ _−_ _µ_ _**M**_ ) _[∗][t]_ _∈_ R _[n][×][n][×][k]_, and _**L**_ _t_ is the tensor-column subspace


16


**Implicit Regularization for Tubal Tensors via GD**


spanned by the tensor-columns corresponding to the first _r_ singular tubes of the gradient descent iterates _**U**_ _t_ = _**Z**_ _t_ _∗_ _**U**_ 0 + _**E**_ _t_,
and consider Fourier-domain representation (D.15) of _**U**_ _t_ . Here, for each 1 _≤_ _j_ _≤_ _k_, the matrices _Zt_ [(] _[j]_ [)] _U_ 0 [(] _[j]_ [)] + _Et_ [(] _[j]_ [)] can
be represented as


( _j_ ) ( _j_ ) ( _j_ ) ( _j_ ) ( _j_ ) ( _j_ ) [H] ( _j_ ) ( _j_ ) ( _j_ ) ( _j_ ) [H] ( _j_ ) ( _j_ )
_Zt_ _U_ 0 + _Et_ = _Zt_ _V_ _**L**_ _V_ _**L**_ _U_ 0 + _Zt_ _V_ _**L**_ _⊥_ _V_ _**L**_ _⊥_ _U_ 0 + _Et_ _._ (D.16)

         - ��         -         - ��         -         - ��         _A_         - [(] _[j]_ [)] _A_ [(] _[j]_ [)] _C_ [(] _[j]_ [)]


As the tensor-column space _**V**_ _**L**_ is _r_ -dimensional, each of matrices _V_ _**L**_ [(] _[j]_ [)] has rank _r_, see (Gleich et al., 2013). Since the
matrices _Zt_ [(] _[j]_ [)] can be decomposed as


( _j_ ) ( _j_ ) ( _j_ ) [H] ( _j_ )
_Zt_ = _V_ _**L**_ Σ [(] _**L**_ _[j]_ [)] _[V]_ _**[L]**_ + _V_ _**L**_ _⊥_ Σ [(] _**L**_ _[j][⊥]_ [)] _[V]_ _**[L]**_ _[⊥]_ [(] _[j]_ [)H]



we have that



_Zt_ ( _j_ ) _V_ _**L**_ ( _j_ ) _V_ _**L**_ ( _j_ ) [H] _U_ 0( _j_ ) = _V_ _**L**_ ( _j_ )Σ [(] _**L**_ _[j]_ [)] _[V]_ _**[L]**_ ( _j_ ) [H] _U_ 0( _j_ ) _._ (D.17)



As _U_ 0 [(] _[j]_ [)] _∈_ C _[r][×][R]_ has rank _r_, _V_ _**L**_ [(] _[j]_ [)H] _U_ 0 [(] _[j]_ [)] has rank _r_, which means that the product above has rank _r_ too. Due to (D.17),
we see that

_Zt_ ( _j_ ) _V_ _**L**_ ( _j_ ) _V_ _**L**_ ( _j_ ) [H] _U_ 0( _j_ ) = _V_ _**L**_ ( _j_ ) _V_ _**L**_ ( _j_ ) [H] _Zt_ ( _j_ ) _V_ _**L**_ ( _j_ ) _V_ _**L**_ ( _j_ ) [H] _U_ 0( _j_ ) _,_


which makes _V_ _**L**_ [(] _[j]_ [)] to the column subspace of _Zt_ [(] _[j]_ [)] _V_ _**L**_ [(] _[j]_ [)] _V_ _**L**_ [(] _[j]_ [)H] _U_ 0 [(] _[j]_ [)] . Considering the gap between the singular values
of for matrices _A_ [(] _[j]_ [)] and _A_ [�][(] _[j]_ [)] in (D.16), namely _δ_ [(] _[j]_ [)] = _σr_ ( _A_ [(] _[j]_ [)] ) _−_ _σr_ +1( _A_ [�][(] _[j]_ [)] ), and using Wedin’s sin _θ_ theorem (Wedin,
1972), for each 1 _≤_ _j_ _≤_ _k_ we get


( _j_ )
_∥V_ _**L**_ _⊥_ [(] _[j]_ [)H] _V_ _**L**_ _t_ _∥≤_ _[∥][C]_ _δ_ [(][(] _[j][j]_ [)][)] _[∥]_ _._


( _j_ )
To conduct a further estimation of _∥V_ _**L**_ _⊥_ [(] _[j]_ [)H] _V_ _**L**_ _t_ _∥_, we analyze lower and upper bounds for the denominator and the
numerator above. We start with the denominator first


_δ_ [(] _[j]_ [)] = _σr_ ( _A_ [(] _[j]_ [)] ) _−_ _σr_ +1( _A_ [�][(] _[j]_ [)] )

= _σr_ ( _Zt_ ( _j_ ) _V_ _**L**_ ( _j_ ) _V_ _**L**_ ( _j_ ) [H] _U_ 0( _j_ )) _−_ _σr_ +1( _Zt_ ( _j_ ) _U_ 0( _j_ ) + _Et_ ( _j_ )) _._


Using properties of singular values of the matrix product for the first term above and Weyl’s bound for the second term, we
get


_δ_ [(] _[j]_ [)] _≥_ _σr_ ( _Zt_ ( _j_ )) _σmin_            - _V_ _**L**_ ( _j_ ) [H] _U_ 0( _j_ ) [�] _−_ _σr_ +1� _Zt_ ( _j_ ) _U_ 0( _j_ ) [�] _−∥Et_ ( _j_ )) _∥_


( _j_ )                          -                          -                          - ( _j_ ) ( _j_ ) [�]
_≥_ _σr_ ( _Zt_ ) _σmin_ _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [0] _−_ _σr_ +1 _Zt_ _U_ 0 _−∥_ _**E**_ _t∥._ (D.18)


For the norm of _C_ [(] _[j]_ [)], the following upper bound can be established


( _j_ ) ( _j_ ) ( _j_ ) [H] ( _j_ ) ( _j_ )
_∥C_ [(] _[j]_ [)] _∥≤∥Zt_ _V_ _**L**_ _⊥_ _V_ _**L**_ _⊥_ _U_ 0 _∥_ + _∥Et_ _∥_

( _j_ ) ( _j_ ) ( _j_ ) [H] ( _j_ )
_≤∥Zt_ _V_ _**L**_ _⊥_ _V_ _**L**_ _⊥_ _∥∥U_ 0 _∥_ + _∥_ _**E**_ _t∥_

_≤_ _ασr_ +1( _Zt_ ( _j_ )) _∥_ _**U**_ _∥_ + _∥_ _**E**_ _t∥_ (D.19)


Now, combining bounds (D.18) and (D.19), one obtains that


_∥_ _**V**_ _[⊤]_ _**L**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥]_ [=] [max] ( _j_ ) _∥≤_ max _ασr_ +1( _Zt_ [(] _[j]_ [)] ) _∥_ _**U**_ _∥_ + _∥_ _**E**_ _t∥_ :
1 _≤j≤k_ _[∥][V]_ _**[L]**_ _[⊥]_ [(] _[j]_ [)H] _[V]_ _**[L]**_ _[t]_ 1 _≤j≤k_             -             -             
_σr_ ( _Zt_ [(] _[j]_ [)] ) _σmin_ _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ _−_ _σr_ +1 _Zt_ [(] _[j]_ [)] _U_ [(] _[j]_ [)][�] _−∥_ _**E**_ _t∥_


Using in the denominator the fact that _σr_ +1� _Zt_ [(] _[j]_ [)] _U_ 0 [(] _[j]_ [)][�] _≤_ _ασr_ +1� _Zt_ [(] _[j]_ [)][�] _∥U_ [(] _[j]_ [)] _∥≤_ _ασr_ +1� _Zt_ [(] _[j]_ [)] ) _∥_ _**U**_ _∥_ finishes the proof
of this lemma.


17


**Implicit Regularization for Tubal Tensors via GD**


Further, we consider the gradient descent iterates with its t-SVD



_**U**_ _t_ =



_R_

- _**V**_ _**U**_ _t_ (: _, s,_ :) _∗_ **Σ** _**U**_ _t_ ( _s, s,_ :) _∗_ _**W**_ _[⊤]_ _**U**_ _t_ [(:] _[, s,]_ [ :)]

_s_ =1



and the corresponding Fourier domain representation _**U**_ _t_ = diag( _Ut_ [(1)] _, Ut_ [(2)] _, . . ., Ut_ [(] _[k]_ [)] ), where



_Ut_ [(] _[j]_ [)] = [�] _ℓ_ _[R]_ =1 _[σ]_ _ℓ_ [(] _[j]_ [)] _[v]_ _ℓ_ [(] _[j]_ [)] _[w]_ _ℓ_ [(] _[j]_ [)]



H ( _j_ )
= _VUt_ [Σ] _U_ [(] _[j]_ _t_ [)] _[W]_ [ (] _U_ _[j]_ _t_ [)H] and its signal-noise term decomposition



_**U**_ _t_ = _**U**_ _t ∗_ _**W**_ _t ∗_ _**W**_ _[⊤]_ _t_ [+] _**[ U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ _[.]_


We also define the corresponding new tensors



_r_

_**L**_ _t_ =          - _**V**_ _**U**_ _t_ (: _, s,_ :) _∗_ **Σ** _**U**_ _t_ ( _s, s,_ :) _∗_ _**W**_ _[⊤]_ _**L**_ _t_ [(:] _[, s,]_ [ :)] (D.20)

_s_ =1


_R_

_**N**_ _t_ =        - _**V**_ _**U**_ _t_ (: _, s,_ :) _∗_ **Σ** _**U**_ _t_ ( _s, s,_ :) _∗_ _**W**_ _[⊤]_ _**U**_ _t_ [(:] _[, s,]_ [ :)] (D.21)

_s_ = _r_ +1


and their Fourier domain representations



_**L**_ _t_ = diag( _Lt_ (1) _, Lt_ (2) _, . . ., Lt_ ( _k_ )) _,_ _Lt_ ( _j_ ) =



_r_

- _σℓ_ [(] _[j]_ [)] _[v]_ _ℓ_ [(] _[j]_ [)] _[w]_ _ℓ_ [(] _[j]_ [)]

_ℓ_ =1



H ( _j_ )
= _V_ _**L**_ _t_ [Σ][(] _**L**_ _[j]_ _t_ [)] _[W]_ [ (] _**L**_ _[j]_ _t_ [)H] (D.22)



_**N**_ _t_ = diag( _Nt_ (1) _, Nt_ (2) _, . . ., Nt_ ( _k_ )) _,_ _Nt_ ( _j_ ) =


**Lemma D.4.** _Assume ∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥≤]_ 2 [1] _[.]_ _[Then it holds that]_



_R_

- _σℓ_ [(] _[j]_ [)] _[v]_ _ℓ_ [(] _[j]_ [)] _[w]_ _ℓ_ [(] _[j]_ [)] H = _V_ _**N**_ ( _j t_ ) [Σ][(] _**N**_ _[j]_ [)] _t_ _[W]_ [ (] _**N**_ _[j]_ _t_ [)H] (D.23)

_ℓ_ = _r_ +1



_∥_ _**W**_ _[⊤]_ _**L**_ _[⊥]_ _t_ _[∗]_ _**[W]**_ _[t][∥≤]_ [2] 1 [max] _≤j≤k_




  _σr_ +1 _Ut_ [(] _[j]_ [)][�]

  -  - _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥][.]_ (D.24)
_σr_ _Ut_ [(] _[j]_ [)]



_Proof._ Consider _∥_ _**W**_ [T] _**L**_ _[⊥]_ _t_ _[∗]_ _**[W]**_ _[t][∥]_ [= max][1] _[≤][j][≤][k][ ∥][W]_ _**[L]**_ _t_ _[⊥]_ ( _j_ ) [H] _Wt_ ( _j_ ) _∥_ . For each 1 _≤_ _j_ _≤_ _k_, we can now exploit the results of
Lemma A.1 in (St¨oger & Soltanolkotabi, 2021), to get that



_∥_ ( _W_ _[⊤]_ ( _j_ ) _∥≤_ _[∥]_ [Σ] _**N**_ [(] _[j]_ [)] _t_ _[∥∥][V]_ _**N**_ [H] _t_ ( _j_ ) _V_ _**X**_ ( _j_ ) _∥_
_**L**_ _[⊥]_ _t_ [)][(] _[j]_ [)] _[W][t]_  -  
[(] _[j]_ [)] [(] _[j]_ [)]



_._
2



_**N**_ _t_ _[∥∥]_ - _[V]_ _**N**_ [H] _t_ ( _j_ ) _V_ _**X**_ (� _j_ ) _∥_ and _σmin_ ( _V_ _**X**_ ( _j_ ) _Ut_ ( _j_ )) _≥_ _[σ][min]_ [(] 2 _[L][t]_ [(] _[j]_ [)][)]

_σmin_ _V_ _**X**_ [(] _[j]_ [)] _Ut_ [(] _[j]_ [)]



From here, we can proceed as follows


_∥_ _**W**_ _[⊤]_ _**L**_ _[⊥]_ _t_ _[∗]_ _**[W]**_ _[t][∥]_ [=] 1 [max] _≤j≤k_ _[∥][W]_ [ H] _**L**_ _[⊥]_ _t_



( _j_ ) _Wt_ ( _j_ ) _∥≤_ 2 max _∥_ Σ [(] _**N**_ _[j]_ [)] _t_ _[∥∥][V]_ _**N**_ [H] _t_ ( _j_ ) _V_ _**X**_ ( _j_ ) _∥_
1 _≤j≤k_ _σmin_ ( _Lt_ [(] _[j]_ [)] )



= 2 max _σr_ +1( _Ut_ [(] _[j]_ [)] ) _∥V_ _**N**_ [H] _t_ ( _j_ ) _V_ _**X**_ ( _j_ ) _∥_
1 _≤j≤k_ _σr_ ( _Ut_ [(] _[j]_ [)] )

_σr_ +1� _Ut_ [(] _[j]_ [)][�]
= 2 1max _≤j≤k_ _σr_ - _Ut_ [(] _[j]_ [)][�] _[∥]_ _**[V]**_ _**X**_ _[⊤]_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥][,]_




_[j]_ [)] ) _∥V_ _**N**_ [H] _t_ ( _j_ ) _V_ _**X**_ ( _j_ ) _∥_ _≤_ 2 max _σr_ +1( _Ut_ [(] _[j]_ [)] )

_σr_ ( _Ut_ [(] _[j]_ [)] ) 1 _≤j≤k_ _σr_ ( _Ut_ [(] _[j]_ [)] )



_r_ +1 _t_

_σr_ ( _Ut_ [(] _[j]_ [)] ) _[∥]_ _**[V]**_ _**L**_ _[⊤][⊥]_ _t_ _[∗]_ _**[V]**_ _**[X]**_ _[ ∥]_



_σr_ +1� _Ut_ [(] _[j]_ [)][�]
= 2 max
1 _≤j≤k_   - [(] _[j]_ [)][�]



which concludes the proof.

**Lemma D.5.** _Assume that ∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥≤]_ [1] 8 _[for some][ t][ ≥]_ [1] _[, t][ ∈]_ [N] _[.]_ _[Then for each]_ [ 1] _[ ≤]_ _[j]_ _[≤]_ _[k][, it holds that]_


                - ( _j_ ) [�]                - ( _j_ ) [�]
_σr_ _**U**_ _t ∗_ _**W**_ _t_ _≥_ [1] _**U**_ _t_ (D.25)

2 _[σ][r]_

_σ_ 1( _**U**_ _t ∗_ _**W**_ _t,⊥_ ( _j_ )) _≤_ 2 _σr_ +1( _Ut_ ( _j_ )) _._ (D.26)


18


**Implicit Regularization for Tubal Tensors via GD**


_Moreover, the principal angles between the tensor-column subspaces spanned by_ _**X**_ _and_ _**U**_ _t_ _**W**_ _t_ _can be estimated as follows_


_∥_ _**V**_ _**X**_ _⊥_ _∗_ _**V**_ _**U**_ _t_ _**W**_ _t∥≤_ 7 _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥]_ (D.27)

_∥_ _**U**_ _t ∗_ _**W**_ _t,⊥∥≤_ 2 max ( _j_ )) _._ (D.28)
1 _≤j≤k_ _[σ][r]_ [+1][(] _[U][t]_


_Proof._ We assume that _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥≤]_ 8 [1] [, then due to Lemma D.4, we obtain that]



_∥_ _**W**_ _[⊤]_ _**L**_ _[⊥]_ _t_ _[∗]_ _**[W]**_ _[t][∥≤]_ [2] 1 [max] _≤j≤k_




  _σr_ +1 _Uj_ [(] _[j]_ [)][�]



_r_ +1 _j_

 -  - _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥≤]_ [1] 4
_σr_ _Uj_ [(] _[j]_ [)]



(D.29)
4 _[.]_




       Now, to estimate _σr_ _**U**_ _t ∗_ _**W**_ _t_ [(] _[j]_ [)][�], we see that for each 1 _≤_ _j_ _≤_ _k_, it holds that


       - ( _j_ ) [�][2] �� ( _j_ ) [�][H] ( _j_ ) [�]       - ( _j_ ) [H] ( _j_ ) [H] ( _j_ ) ( _j_ ) [�]
_σr_ _**U**_ _t ∗_ _**W**_ _t_ = _σr_ _**U**_ _t ∗_ _**W**_ _t_ _**U**_ _t ∗_ _**W**_ _t_ = _σr_ _Wt_ _Ut_ _Ut_ _Wt_ (D.30)


Since _Ut_ [(] _[j]_ [)H] _Ut_ [(] _[j]_ [)] = _Lt_ [(] _[j]_ [)H] _Lt_ [(] _[j]_ [)] + _Nt_ [(] _[j]_ [)H] _Nt_ [(] _[j]_ [)], we get that


_σr_       - _**U**_ _t ∗_ _**W**_ _t_ ( _j_ ) [�][2] _≥_ _σr_       - _Wt_ ( _j_ ) [H] _Lt_ ( _j_ ) [H] _Lt_ ( _j_ ) _Wt_ ( _j_ ) [�] = _σr_       - _Wt_ ( _j_ ) [H] _Lt_ ( _j_ ) [�][2]


                  - ( _j_ ) [H] �2                  - ( _j_ ) [�][2]                  - ( _j_ ) [�][2]
_≥_ _σr_ _Wt_ _WLt_ ( _j_ ) _σr_ _Lt_ _≥_ (1 _−∥_ _**W**_ _**L**_ _⊥t_ _[∗]_ _**[W]**_ _t_ _[T]_ _[∥]_ [2][�] _σr_ _Ut_ _,_


where in the last line we used the definition of the principal angle between tensor column subspaces and the corresponding
properties in their Fourier domain slices, namely



_σr_ - _Wt_ ( _j_ ) [H] _WLt_ ( _j_ )�2 = 1 _−∥Wt_ ( _j_ ) [H] _WL⊥_



_L⊥t_ [(] _[j]_ [)] _[∥]_ [2] [= 1] _[ −∥]_ _**[W]**_ _**[L]**_ _[⊥]_ _t_ _[∗]_ _**[W]**_ _t_ _[T]_ _[∥]_ [2] _[.]_



_⊥_ _[≥]_ [1] _[ −]_ [max] ( _j_ ) [H] _W_ _⊥_

_Lt_ [(] _[j]_ [)] _[∥]_ [2] 1 _≤j≤k_ _[∥][W][t]_ _Lt_



8 [1] [, we can see that in the Fourier domain, the subspaces spanned by] _[ V]_ _**X**_ [(] _[j]_ [)] _[⊥]_ _t_ [and]



Due to our assumption _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥≤]_ 8 [1]



_V_ _**L**_ [(] _[j]_ _t_ [)] [=] _[ V]_ _Lt_ [(] _[j]_ [)] [are close enough.] [Then, decomposing] _[ U][t]_ [(] _[j]_ [)] [into two different ways, namely as]



_Ut_ ( _j_ ) =



_R_

- _σℓ_ [(] _[j]_ [)] _[v]_ _ℓ_ [(] _[j]_ [)] _[w]_ _ℓ_ [(] _[j]_ [)]

_ℓ_ =1



H
= _Lt_ ( _j_ ) + _Nt_ ( _j_ )



and as



( _j_ ) ( _j_ ) ( _j_ ) ( _j_ ) [H] ( _j_ ) ( _j_ ) ( _j_ ) [H]
_Ut_ = _Ut_ _Wt_ _Wt_ + _Ut_ _Wt,⊥_ _Wt,⊥_ _,_



according to Lemma H.1, one obtains for each 1 _≤_ _j_ _≤_ _k_ that



_∥V_ [(] _**X**_ _[j]_ [)] _[⊥]_ _t_ H _VUt_ ( _j_ ) _Wt_ ( _j_ ) _∥≤_ 7 _∥V_ ( _**X**_ _j_ ) _[⊥]_ _t_



H ( _j_ )
_V_ _**L**_ _t_ _[∥]_



_∥Ut_ ( _j_ ) _Wt,⊥_ ( _j_ ) _∥≤_ 2 _σr_ +1( _Ut_ ( _j_ )) _,_


where the last inequality is equivalent to _σ_ 1( _**U**_ _t ∗_ _**W**_ _t,⊥_ [(] _[j]_ [)] ) _≤_ 2 _σr_ +1( _Ut_ [(] _[j]_ [)] ). According to the definition of principal angles
between tensor subspaces, this implies that



H ( _j_ )
_V_ _**L**_ _t_ _[∥]_ [= 7] _[∥]_ _**[V]**_ _**X**_ _[⊤]_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥][.]_



_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ [= max] _j_ _∥V_ [(] _**X**_ _[j]_ [)] _[⊥]_ _t_



H _VUt_ ( _j_ ) _Wt_ ( _j_ ) _∥≤_ 7 max _j_ _∥V_ [(] _**X**_ _[j]_ [)] _[⊥]_ _t_



In the same way, _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥∥_ = max _j ∥Ut_ [(] _[j]_ [)] _Wt,⊥_ [(] _[j]_ [)] _∥≤_ 2 max _j σr_ +1( _Ut_ [(] _[j]_ [)] ), which finishes the proof.


**Lemma D.6.** _Consider a tensor_ _**T**_ := _**X**_ _∗_ _**X**_ _[⊤]_ _∈_ _S_ + _[n][×][n][×][k]_ _with tubal rank r_ _≤_ _n._ _Assume that measurement operator A is_
_such that_
_**M**_ = _A_ _[∗]_ _A_ ( _**T**_ ) = _**T**_ + _**E**_ _∈_ _S_ + _[n][×][n][×][k]_


19


**Implicit Regularization for Tubal Tensors via GD**


_and_ _for_ _for_ _each_ 1 _≤_ _j_ _≤_ _k_ _one_ _has_ _∥E_ [(] _[j]_ [)] _∥≤_ _δλr_ ( _**T**_ [(] _[j]_ [)] ) _with_ _δ_ _≤_ 14 _[.]_ _[For]_ _[the]_ _[same]_ _**[M]**_ _[with]_ _[its]_ _[t-SVD]_ _**[M]**_ [=]
_**V**_ _**M**_ _∗_ **Σ** _**M**_ _∗_ _**W**_ _[⊤]_ _**M**_ _[, let]_ _**[ L]**_ _[ ∈]_ [R] _[n][×][r][×][k]_ _[denote the tensor column subspace spanned by the tensor-columns corresponding to]_
_the first r singular tubes, that is_ _**L**_ := _**V**_ _**M**_ (: _,_ 1 : _r,_ :) _∈_ R _[n][×][r][×][k]_ _._


_Then, in each Fourier slice j,_ 1 _≤_ _j_ _≤_ _k, it holds that_


(1 _−_ _δ_ ) _λ_ 1( _T_ [(] _[j]_ [)] ) _≤_ _λ_ 1( _M_ [(] _[j]_ [)] ) _≤_ (1 + _δ_ ) _λ_ 1( _T_ [(] _[j]_ [)] ) (D.31)

_λr_ +1( _M_ [(] _[j]_ [)] ) _≤_ _δλr_ ( _T_ [(] _[j]_ [)] ) (D.32)

_λr_ ( _M_ [(] _[j]_ [)] ) _≥_ (1 _−_ _δ_ ) _λr_ ( _T_ [(] _[j]_ [)] ) _,_ (D.33)


_and_
(1 _−_ _δ_ ) _∥_ _**T**_ _∥≤∥_ _**M**_ _∥≤_ (1 + _δ_ ) _∥_ _**T**_ _∥_ (D.34)


_Moreover, the tensor-column subspaces of_ _**X**_ _and_ _**L**_ _are aligned, namely_


_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[∥≤]_ [2] _[δ]_ (D.35)


_Proof._ Consider tensor _**T**_ := _**X**_ _∗_ _**X**_ _[⊤]_ _∈_ _S_ + _[n][×][n][×][k]_ . Due to the definition of tensor transpose and conjugate symmetry of

Fourier coefficients (Kilmer & Martin, 2011), the Fourier slices of _**T**_ are defined as _T_ [(] _[j]_ [)] = _X_ [(] _[j]_ [)] _X_ [(] _[j]_ [)H] . That is, each face of
_**T**_ is Hermitian and at least positive semidefinite. As we assume that for each _j_, 1 _≤_ _j_ _≤_ _k_, one has _∥Et_ [(] _[j]_ [)] _∥≤_ _δλr_ ( _**T**_ [(] _[j]_ [)] )
using Weyl’s inequality in each of the Fourier slices, we obtain the first three inequalities.


To show that the tensor subspace _**V**_ _**X**_ and _**V**_ _**L**_ are aligned, we use first the definition



H

_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[∥]_ [=] 1 [max] _≤j≤k_ _[∥][V]_ _**X**_ [(] _[j][⊥]_ [)]



_V_ [(] _**L**_ _[j]_ [)] _[∥]_ (D.36)



For the estimation of _∥V_ _**X**_ [H] _[⊥]_ ( _j_ ) _V_ ( _**L**_ _j_ ) _[∥]_ [in] [each] [of] [the] [Fourier] [slices,] [we] [apply] [Wedin’s] [sin Θ] [theorem.] [For] [this,] [denote]

_**L**_ := _**V**_ _**M**_ (: _,_ 1 : _r,_ :) _∈_ R _[n][×][r][×][k]_ and let _V_ [(] _**L**_ _[j]_ [)] [denote the corresponding Fourier slices of] _**[ L]**_ _[ ∈]_ [R] _[n][×][r][×][k]_ [.] [Since in the Fourier]
space, it holds that _M_ [(] _[j]_ [)] = _T_ [(] _[j]_ [)] + _E_ [(] _[j]_ [)] and _V_ [(] _**L**_ _[j]_ [)] [encompasses the first] _[ r]_ [ eigenvectors of] _[ M]_ [(] _[j]_ [)][, from Wedin’s][ sin Θ][ theorem,]
we obtain



H

_∥V_ _**X**_ [(] _[j][⊥]_ [)]



_V_ [(] _**L**_ _[j]_ [)] _[∥≤]_ _[∥][E]_ [(] _[j]_ [)] _[∥]_ _,_

_ξ_ [(] _[j]_ [)]



with _ξ_ [(] _[j]_ [)] := _λr_ ( _T_ [(] _[j]_ [)] ) _−_ _λr_ +1( _M_ [(] _[j]_ [)] ). Using estimate (D.32), _ξ_ [(] _[j]_ [)] can be lower-bounded as


_ξ_ [(] _[j]_ [)] := _λr_ ( _T_ [(] _[j]_ [)] ) _−_ _λr_ +1( _M_ [(] _[j]_ [)] ) _≥_ _λr_ ( _T_ [(] _[j]_ [)] ) _−_ _δλr_ ( _T_ [(] _[j]_ [)] ) = (1 _−_ _δ_ ) _λr_ ( _T_ [(] _[j]_ [)] ) _._


Using the bound the the assumptions that _∥Et_ [(] _[j]_ [)] _∥≤_ _δλr_ ( _**T**_ [(] _[j]_ [)] ) and _δ_ _≤_ [1] 2 [, we get]



_∥V_ _**X**_ [(] _[j][⊥]_ [)]



H _δ_

_V_ [(] _**L**_ _[j]_ [)] _[∥≤]_
1 _−_ _δ_ _[≤]_ [2] _[δ.]_



Coming back to equality (D.36), we obtain the stated bound for the principal angle between the two tensor column
subspaces.


**Lemma D.7.** _Consider a tensor_ _**X**_ _∗_ _**X**_ _[⊤]_ _∈_ _S_ + _[n][×][n][×][k]_ _with tubal rank r_ _≤_ _n._ _Assume that measurement operator A is such_
_that_
_**M**_ = _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ) = _**X**_ _∗_ _**X**_ _[⊤]_ + _**E**_

_and for each, j,_ 1 _≤_ _j_ _≤_ _k, one has ∥E_ [(] _[j]_ [)] _∥≤_ _δλr_ ( _X_ [(] _[j]_ [)] _X_ [(] _[j]_ [)H] ) _with δ_ _≤_ _c_ 1 _._ _Moreover, assume that for difference tensor_
_**E**_ _t_ = _**U**_ _t −_ _**U**_ [�] _t it holds that_



_α_ max ( _j_ )) _∥_ _**U**_ _∥_ + _∥_ _**E**_ _t∥_
1 _≤j≤k_ _[σ][r]_ [+1][(] _[Z][t]_
_γ_ :=

min ( _j_ ))
1 _≤j≤k_ _[σ][r]_ [(] _[Z][t]_


20



1

_≤_ _c_ 2 _κ_ _[−]_ [2] _,_ (D.37)
_ασmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]


**Implicit Regularization for Tubal Tensors via GD**


_where c_ 1 _, c_ 2 _>_ 0 _are sufficiently small absolute constants._ _Then for the signal and noise term of the gradient descent_ (C.1) _,_
_we have_


_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥≤]_ [14(] _[δ]_ [ +] _[ γ]_ [)] (D.38)

_∥_ _**U**_ _t ∗_ _**W**_ _t,⊥∥≤_ _[κ][−]_ [2] [min] ( _j_ )) _σmin_ ( _**V**_ _⊤_ _**L**_ _[∗]_ _**[U]**_ [)] (D.39)

8 _[α]_ 1 _≤j≤k_ _[σ][r]_ [(] _[Z][t]_


_and for each j,_ 1 _≤_ _j_ _≤_ _k, it holds that_


_σmin_ ( _**U**_ _t ∗_ _**W**_ _t_ ( _j_ )) _≥_ [1] [min] ( _j_ )) _σmin_ ( _**V**_ _⊤_ _**L**_ _[∗]_ _**[U]**_ [)] (D.40)

4 _[α]_ 1 _≤j≤k_ _[σ][r]_ [(] _[Z][t]_

_σ_ 1( _**U**_ _t ∗_ _**W**_ _t,⊥_ ( _j_ )) _≤_ _[κ][−]_ [2] [min] ( _j_ )) _σmin_ ( _**V**_ _⊤_ _**L**_ _[∗]_ _**[U]**_ [)] (D.41)

8 _[α]_ 1 _≤j≤k_ _[σ][r]_ [(] _[Z][t]_


_Proof._ To prove the above-stated properties, we will use Lemma D.3. Therefore, we start by checking the conditions of this
lemma. Sufficiently small _c_ 2 and the assumption _γ_ _≤_ _c_ 2 _κ_ _[−]_ [2] allows for _γ_ _≤_ [1] 2 [.] [This means that]



_α_ max ( _j_ )) _∥_ _**U**_ _∥_ + _∥_ _**E**_ _t∥_
1 _≤j≤k_ _[σ][r]_ [+1][(] _[Z][t]_

min ( _j_ ))
1 _≤j≤k_ _[σ][r]_ [(] _[Z][t]_


and in each of the Fourier slices we have



1

_≤_ [1]

2

_ασmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]



1



2



_σr_ +1( _Zt_ ( _j_ )) _∥_ _**U**_ _∥_ + _[∥]_ _**[E]**_ _[t][∥]_



( _j_ )) _σmin_ ( _**V**_ _⊤_ _**L**_ _[∗]_ _**[U]**_ [)] _[,]_
2 _[σ][r]_ [(] _[Z][t]_




_[t][∥]_

_≤_ [1]
_α_ 2



fulfilling the assumption of Lemma D.3. Hence, from Lemma D.3, we conclude that


_∥_ _**V**_ _[⊤]_ _**L**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥≤]_ [max] _ασr_ +1( _Zt_ [(] _[j]_ [)] ) _∥_ _**U**_ _∥_ + _∥_ _**E**_ _t∥_ (D.42)
1 _≤j≤k_ _ασr_ ( _Zt_ [(] _[j]_ [)] ) _σmin_                  - _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_                  - _−_ _ασr_ +1� _Zt_ [(] _[j]_ [)] ) _∥_ _**U**_ _∥−∥_ _**E**_ _t∥_

_α_ max ( _j_ )) _∥_ _**U**_ _∥_ + _∥_ _**E**_ _t∥_
1 _≤j≤k_ _[σ][r]_ [+1][(] _[Z][t]_
_≤_ _,_ (D.43)

_α_ min ( _j_ )) _σmin_           - _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_           - _−_ _α_ max           - _Zt_ ( _j_ )) _∥_ _**U**_ _∥−∥_ _**E**_ _t∥_
1 _≤j≤k_ _[σ][r]_ [(] _[Z][t]_ 1 _≤j≤k_ _[σ][r]_ [+1]


and, moreover, together with Lemma D.5 and the assumption _γ_ _≤_ 2 [1] [we get]

min ( _j_ )) _≥_ _α_ min ( _j_ )) _σmin_ ( _**V**_ _⊤_ _**L**_ _[∗]_ _**[U]**_ [)] _[ −∥]_ _**[E]**_ _[t][∥≥]_ _[α]_ ( _j_ )) _σmin_ ( _**V**_ _⊤_ _**L**_ _[∗]_ _**[U]**_ [)] (D.44)
1 _≤j≤k_ _[σ][r]_ [(] _[U][t]_ 1 _≤j≤k_ _[σ][r]_ [(] _[Z][t]_ 2 1 [min] _≤j≤k_ _[σ][r]_ [(] _[Z][t]_

max ( _j_ )) _≤_ _α_ min ( _j_ )) _∥_ _**U**_ _∥_ + _∥_ _**E**_ _t∥≤_ _αγ_ min ( _j_ )) _σmin_ ( _**V**_ _⊤_ _**L**_ _[∗]_ _**[U]**_ [)] (D.45)
1 _≤j≤k_ _[σ][r]_ [+1][(] _[U][t]_ 1 _≤j≤k_ _[σ][r][σ][r]_ [(] _[Z][t]_ 1 _≤j≤k_ _[σ][r]_ [(] _[Z][t]_


The last two inequalities, allow extend bound (D.42) as follows


_γ_
_∥_ _**V**_ _[⊤]_ _**L**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥≤]_ (D.46)
1 _−_ _γ_


Now, consider the principal angle between _**X**_ and _**L**_ _t_ using its definition

_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥]_ [=] 1 [max] _≤j≤k_ _[∥][V]_ [(] _**X**_ _[j]_ [)] _[⊥]_ H _V_ ( _**L**_ _jt_ ) _[∥]_ [=] 1 [max] _≤j≤k_ _[∥][V]_ [(] _**X**_ _[j]_ [)] _[⊥]_ _[V]_ [(] _**X**_ _[j]_ [) H] _[⊥]_ _[−]_ _[V]_ [(] _**L**_ _[j]_ _t_ [)] _[V]_ [(] _**L**_ _[j]_ _t_ [)H] _[∥]_

_≤_ 1max _≤j≤k_ _[∥][V]_ [(] _**X**_ _[j]_ [)] _[⊥]_ _[V]_ [(] _**X**_ _[j]_ [) H] _[⊥]_ _[−]_ _[V]_ [(] _**L**_ _[j]_ _t_ [)] _[V]_ [(] _**L**_ _[j]_ _t_ [)H] _[∥≤]_ 1 [max] _≤j≤k_ _[∥][V]_ [(] _**X**_ _[j]_ [)] _[⊥]_ _[V]_ [(] _**X**_ _[j]_ [) H] _[⊥]_ _[−]_ _[V]_ [(] _**L**_ _[j]_ [)] _[V]_ [(] _**L**_ _[j]_ [)H] _∥_ + _∥V_ [(] _**L**_ _[j]_ [)] _[V]_ [(] _**L**_ _[j]_ [)H] _−_ _V_ [(] _**L**_ _[j]_ _t_ [)] _[V]_ [(] _**L**_ _[j]_ _t_ [)H] _[∥]_

_≤_ 1max _≤j≤k_ _[∥][V]_ [(] _**X**_ _[j]_ [)] _[⊥]_ _[V]_ [(] _**X**_ _[j]_ [) H] _[⊥]_ _[−]_ _[V]_ [(] _**L**_ _[j]_ [)] _[V]_ [(] _**L**_ _[j]_ [)H] _∥_ + 1max _≤j≤k_ _[∥][V]_ [(] _**L**_ _[j]_ [)] _[V]_ [(] _**L**_ _[j]_ [)H] _−_ _V_ [(] _**L**_ _[j]_ _t_ [)] _[V]_ [(] _**L**_ _[j]_ _t_ [)H] _[∥]_

= _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[∥]_ [+] _[ ∥]_ _**[V]**_ _[⊤]_ _**L**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥]_


21


**Implicit Regularization for Tubal Tensors via GD**


Using the last line above, and inequalities (D.35) and (D.46), we obtain


_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥≤]_ [2(] _[δ]_ [ +] _[ γ]_ [)] _[.]_


From here, allowing _δ_ and _γ_ to be such that _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥≤]_ [1] 8 [, we can use Lemma D.5 to get]


_∥_ _**V**_ _**X**_ _⊥_ _∗_ _**V**_ _**U**_ _t_ _**W**_ _t∥≤_ 7 _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[L]**_ _[t][∥≤]_ [14(] _[δ]_ [ +] _[ γ]_ [)] _[.]_


Furthermore, Lemma D.5 together with inequality (D.45) also results in


_σ_ 1( _**U**_ _t ∗_ _**W**_ _t,⊥_ ( _j_ )) _≤_ 2 _σr_ +1( _Ut_ ( _j_ ))

_≤_ 2 max ( _j_ ))
1 _≤j≤k_ _[σ][r]_ [+1][(] _[U][t]_

_≤_ 2 _γα_ min ( _j_ )) _σ_ min( _**V**_ _⊤_ _**L**_ _[∗]_ _**[U]**_ [)]
1 _≤j≤k_ _[σ][r]_ [(] _[Z][t]_

_≤_ _[κ][−]_ [2] [min] ( _j_ )) _σ_ min( _**V**_ _⊤_ _**L**_ _[∗]_ _**[U]**_ [)]

8 _[α]_ 1 _≤j≤k_ _[σ][r]_ [(] _[Z][t]_


and for the spectral norm of _**U**_ _t ∗_ _**W**_ _t,⊥_ we get

_∥_ _**U**_ _t ∗_ _**W**_ _t,⊥∥≤_ 2 max ( _j_ )) _≤_ _[κ][−]_ [2] [min] ( _j_ )) _σ_ min( _**V**_ _⊤_ _**L**_ _[∗]_ _**[U]**_ [)] _[.]_
1 _≤j≤k_ _[σ][r]_ [+1][(] _[U][t]_ 8 _[α]_ 1 _≤j≤k_ _[σ][r]_ [(] _[Z][t]_


To conclude the proof, we see that Lemma D.5 together with inequality (D.44) provides for each _j_, 1 _≤_ _j_ _≤_ _k_, the following
lower bound




 - ( _j_ ) [�]
_σr_ _**U**_ _t ∗_ _**W**_ _t_ _≥_ [1]




[1] - _**U**_ _t_ ( _j_ ) [�] _≥_ _[α]_

2 _[σ][r]_ 4



( _j_ ) _⊤_

_[α]_ ) _σmin_ ( _**V**_ _**L**_ _[∗]_ _**[U]**_ [)] _[ ≥]_ _[α]_

4 _[σ][r]_ [(] _[Z][t]_ 4



( _j_ )) _σmin_ ( _**V**_ _⊤_ _**L**_ _[∗]_ _**[U]**_ [)] _[.]_
4 1 [min] _≤j≤k_ _[σ][r]_ [(] _[Z][t]_



The following lemma shows that for an appropriately chosen initialization, in the first new iteration, the tensor column
subspaces between the signal term _**U**_ _t ∗_ _**W**_ _t_ and the ground truth tensor _**X**_ become aligned. Moreover, for each 1 _≤_ _j_ _≤_ _k_
there is a solid gap between the smallest singular values of the signal term and the largest singular values of the noise term.

**Lemma D.8.** _Assume A_ : _S_ _[n][×][n][×][k]_ _→_ R _[m]_ _satisfies the S2NRIP_ ( _δ_ 1) _for some constant δ_ 1 _>_ 0 _._ _Also, assume that_


_**M**_ := _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ) = _**X**_ _∗_ _**X**_ _[⊤]_ + _**E**_


_with ∥E_ [(] _[j]_ [)] _∥≤_ _δλr_ ( _X_ [(] _[j]_ [)] _X_ [(] _[j]_ [)H] ) _for each_ 1 _≤_ _j_ _≤_ _k and δ_ _≤_ _c_ 1 _κ_ _[−]_ [2] _._


_Denote by_ _**L**_ _the tensor-columns corresponding to the first r singular tubes in the t-SVD of_ _**M**_ _, that is,_ _**L**_ := _**V**_ _**M**_ (: _,_ 1 : _r,_ :) _∈_
R _[n][×][r][×][k]_ _, and define the initialization_ _**U**_ 0 = _α_ _**U**_ _with the coefficient α such that_



H
min _{σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)] _[,][ ∥]_ _**[U]**_ [0] _v_ 1 _∥ℓ_ 2 _}_ (D.47)




- _−_ 48 _κ_ [2]



_c∥_ _**X**_ _∥_ [2]
_α_ [2] _≤_

12 _k_ �min _{n, R}κ_ [2] _∥_ _**U**_ _∥_ [3]




2 _κ_ [2] _∥_ _**U**_ _∥_ [3]

_c_ 3 _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]



_where v_ 1 _∈_ C _[nk]_ _is the leading eigenvector of matrix_ _**M**_ _∈_ C _[nk][×][nk]_ _._


_Assume that learning rate µ fulfils µ ≤_ _c_ 3 _κ_ _[−]_ [2] _∥_ _**X**_ _∥_ _[−]_ [2] _, then after t⋆_ _iterations with_



1
_t⋆_ _≍_
_µ_ min1 _≤j≤k σr_ ( _X_ [(] _[j]_ [)] ) [2] [ln]




2 _κ_ [2] _∥_ _**U**_ _∥_

_c_ 3 _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]







(D.48)



_it holds that_



_∥_ _**U**_ _t⋆_ _∥≤_ 3 _∥_ _**X**_ _∥_ (D.49)

_∥_ _**V**_ _**X**_ _⊥_ _∗_ _**V**_ _**U**_ _t⋆_ _∗_ _**W**_ _t⋆_ _∥≤_ _cκ_ _[−]_ [2] _._ (D.50)


22


_and for each_ 1 _≤_ _j_ _≤_ _k, we have_



**Implicit Regularization for Tubal Tensors via GD**


    - ( _j_ ) [�]
_σr_ _**U**_ _t⋆_ _∗_ _**W**_ _t⋆_ _≥_ 4 [1] _[αβ]_ (D.51)


    - ( _j_ ) [�]
_σ_ 1 _**U**_ _t⋆_ _∗_ _**W**_ _t⋆,⊥_ _≤_ _[κ]_ 8 _[−]_ [2] _[αβ]_ (D.52)

(D.53)




                2 _κ_ [2] _∥_ _**U**_ _∥_
_where β_ _satisfies σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)] _[ ≤]_ _[β]_ _[≤]_ _[σ][min]_ [(] _**[V]**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)] _c_ 3 _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]




                2 _κ_ [2] _∥_ _**U**_ _∥_
_where β_ _satisfies σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)] _[ ≤]_ _[β]_ _[≤]_ _[σ][min]_ [(] _**[V]**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)] _[⊤]_



�16 _κ_ [2]



_._



_Proof._ For the proof of this lemma, we want to apply Lemma D.7. The first condition of Lemma D.7 is the following



_α_ max ( _j_ )) _∥_ _**U**_ _∥_ + _∥_ _**E**_ _t∥_
1 _≤j≤k_ _[σ][r]_ [+1][(] _[Z][t]_
_γ_ :=

min ( _j_ ))
1 _≤j≤k_ _[σ][r]_ [(] _[Z][t]_


By the definition of _γ_, it is sufficient to show that



1

_≤_ _c_ 2 _κ_ _[−]_ [2] _,_
_ασmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]



1max _≤j≤k_ _[σ][r]_ [+1][(] _[Z][t]_ ( _j_ )) _∥_ _**U**_ _∥≤_ 2 _[c]_ _κ_ [3][2] 1 [min] _≤j≤k_ _[σ][r]_ [(] _[Z][t]_ ( _j_ )) _σmin_ ( _**V**_ _⊤_ _**L**_ _[∗]_ _**[U]**_ [)] (D.54)


and
_∥_ _**E**_ _t∥≤_ 2 _[c]_ _κ_ [3][2] _[α]_ 1 [min] _≤j≤k_ _[σ][r]_ [(] _[Z][t]_ ( _j_ )) _σmin_ ( _**V**_ _⊤_ _**L**_ _[∗]_ _**[U]**_ [)] _[.]_ (D.55)

Since for _**Z**_ _t_ = ( _**I**_ + _µ_ _**M**_ ) _[∗][t]_ the transformation in the Fourier domain leads to the blocks


_Z_ [(] _t_ _[j]_ [)] = (Id + _µM_ [(] _[j]_ [)] ) _[t]_ _,_


this means that inequality (D.54) is equivalent to



2 _κ_ [2] _∥_ _**U**_ _∥_

_≤_
_c_ 3 _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]



 1 + _µ_ min [(] _[j]_ [)][)]

1 _≤j≤k_ _[σ][r]_ [(] _[M]_








1 + _µ_ max [(] _[j]_ [)][)]
1 _≤j≤k_ _[σ][r]_ [+1][(] _[M]_



_t_








_,_



which can be further modified as


ln


Hence, if we take _t⋆_ as follows


_t⋆_ :=






ln





- ��
2 _κ_ [2] _∥_ _**U**_ _∥_

ln

_σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]








2 _κ_ [2] _∥_ _**U**_ _∥_

_σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]







_≤_ _t_ ln



 1 + _µ_ min [(] _[j]_ [)][)]

1 _≤j≤k_ _[σ][r]_ [(] _[M]_




1 + _µ_ max [(] _[j]_ [)][)]
1 _≤j≤k_ _[σ][r]_ [+1][(] _[M]_






 _._



 (D.56)







 1 + _µ_ min [(] _[j]_ [)][)]

1 _≤j≤k_ _[σ][r]_ [(] _[M]_








1 + _µ_ max [(] _[j]_ [)][)]
1 _≤j≤k_ _[σ][r]_ [+1][(] _[M]_



(D.56)




then condition (D.54) will be satisfied in each block in the Fourier domain. For convenience, we will further denote



_ψ_ := ln




- 2 _κ_ [2] _∥_ _**U**_ _∥_

_σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]



_._ (D.57)



For the second part of Lemma D.7’s condition, inequality (D.55), we will use Lemma D.1. To apply this Lemma, the
condition _t⋆_ _≤_ _t_ _[⋆]_ needs to be satisfied. According to Lemma D.2




 - H
ln ~~_√_~~ _∥_ _**M**_ _∥·∥_ _**U**_ 0 _v_ 1 _∥ℓ_ 2
8(1+ _δ_ 1 _k_ ) ~~_[√]_~~ _k_ min _{n,R}_



2 ln (1 + _µ∥_ _**M**_ _∥_ )


23






(D.58)




_t_ _[⋆]_ _≥_







_k_ ) ~~_[√]_~~



_k_ min _{n,R}α_ [3] _∥_ _**U**_ _∥_ [3]


**Implicit Regularization for Tubal Tensors via GD**


For _t⋆_ _≤_ _t_ _[⋆]_ to hold, it will be sufficient to check, e.g., the following condition







_k_ min _{n,R}α_ [3] _∥_ _**U**_ _∥_ [3]



_ψ_

 - 1+ _µ_ min1 _≤j≤k σr_ ( _M_ [(] _[j]_ [)] )
ln
1+ _µ_ max1 _≤j≤k σr_ +1( _M_ [(] _[j]_ [)] )




   - H
ln ~~_√_~~ _∥_ _**M**_ _∥·∥_ _**U**_ 0 _v_ 1 _∥ℓ_ 2
_≤_ [1] 8(1+ _δ_ 1 _k_ ) ~~_[√]_~~ _k_ min _{n,R_

- 2 _[·]_ 2 ln (1 + _µ∥_ _**M**_ _∥_



_k_ ) ~~_[√]_~~



_._
2 ln (1 + _µ∥_ _**M**_ _∥_ )




                            - 1+ _µ_ min1 _≤j≤k σr_ ( _M_ [(] _[j]_ [)] )
To check this condition let us first analyze the expression ln (1 + _µ∥_ _**M**_ _∥_ ) _/_ ln
1+ _µ_ max1 _≤j≤k σr_ +1( _M_ [(] _[j]_ [)] )

_x_
1+ _x_ _[≤]_ [ln(1 +] _[ x]_ [)] _[ ≤]_ _[x]_ [, we can upper bound the above expression as]




                            - 1+ _µ_ min1 _≤j≤k σr_ ( _M_ [(] _[j]_ [)] )
To check this condition let us first analyze the expression ln (1 + _µ∥_ _**M**_ _∥_ ) _/_ ln
1+ _µ_ max1 _≤j≤k σr_ +1( _M_ [(] _[j]_ [)] )




first. Using



ln (1 + _µ∥_ _**M**_ _∥_ )

 - 1+ _µ_ min1 _≤j≤k σr_ ( _M_ [(] _[j]_ [)] )
ln
1+ _µ_ max1 _≤j≤k σr_ +1( _M_ [(] _[j]_ [)] )



_∥_ _**M**_ _∥_ (1 + _µ_ min1 _≤j≤k σr_ ( _M_ [(] _[j]_ [)] ))
_≤_ (D.59)

- min1 _≤j≤k σr_ ( _M_ [(] _[j]_ [)] ) _−_ max1 _≤j≤k σr_ +1( _M_ [(] _[j]_ [)] )



From here, applying the PSD of the tensor representatives in the Fourier domain and the assumptions _δ_ _≤_ 3 [1] [and] _[µ]_ _[≤]_

_c_ 3 _κ_ _[−]_ [2] _∥_ _**X**_ _∥_ _[−]_ [2] and Lemma D.6, we get



�2 [�]



_∥_ _**M**_ _∥_ (1 + min1 _≤j≤k σr_ ( _M_ [(] _[j]_ [)] )) (1 + _δ_ ) _∥_ _**T**_ _∥_
min1 _≤j≤k σr_ ( _M_ [(] _[j]_ [)] ) _−_ max1 _≤j≤k σr_ +1( _M_ [(] _[j]_ [)] ) _[≤]_ (1 _−_ 2 _δ_ ) _λr_ ( _T_ [(] _[j]_ [)] )




- - _λ_ 1( _X_ ( _j_ ))

1 + _c_ 3(1 + _δ_ )
_κ∥_ _**X**_ _∥_








[(1 +] _[ δ]_ [)]
_≤_ _κ_ [2]

(1 _−_ 2 _δ_ ) [(1 +] _[ c]_ [3][(1 +] _[ δ]_ [))] _[ ≤]_ [8] _[κ]_ [2] _[,]_


in the last line, we used the bound on _δ_ and that _c_ 3 can be taken small enough. This means



ln (1 + _µ∥_ _**M**_ _∥_ )

 - 1+ _µ_ min1 _≤j≤k σr_ ( _M_ [(] _[j]_ [)] )
ln
1+ _µ_ max1 _≤j≤k σr_ +1( _M_ [(] _[j]_ [)] )



_≤_ 8 _κ_ [2] _._ (D.60)




Thus, to show that _t⋆_ _≤_ _t_ _[⋆]_, it is sufficient to tune the initialization factor _α_ so that




    - H

_ψ ·_ 32 _κ_ [2] _≤_ ln ~~_√_~~ _∥_ _**M**_ _∥· ∥_ _**U**_ 0 _v_ 1 _∥ℓ_ 2

8(1 + _δ_ 1 _k_ ) ~~�~~ _k_ min _{n, R}α_ [3] _∥_ _**U**_ _∥_ [3]





_._



or using the notation for _ϕ_, this is equivalent to



H
_≤_ ~~_√_~~ _∥_ _**M**_ _∥· ∥_ _**U**_ 0 _v_ 1 _∥ℓ_ 2
8(1 + _δ_ 1 _k_ )� _k_ min _{n, R}α_ [3] _∥_ _**U**_ _∥_ [3]




2 _κ_ [2] _∥_ _**U**_ _∥_

_σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]



�32 _κ_ [2]



H H
Since _∥_ _**U**_ 0 _v_ 1 _∥ℓ_ 2 _/α_ = _∥_ _**U**_ _v_ 1 _∥ℓ_ 2, The last inequality is implied if




- _−_ 32 _κ_ [2] H
_∥_ ~~_√_~~ _**M**_ _∥· ∥_ _**U**_ _v_ 1 _∥ℓ_ 2
8(1 + _δ_ 1 _k_ )� _k_ min _{n, R}∥_ _**U**_ _∥_ [3] _[,]_



_α_ [2] _≤_




2 _κ_ [2] _∥_ _**U**_ _∥_

_σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]



_√_
or if we set _α_ even smaller using the fact that (1 + _δ_ 1



_√_
_k_ )



_√_
_k_ _≤_ (1 +



_√_
_k_ )



_k_ _≤_ 2 _k_ and _∥_ _**M**_ _∥≥_ [2]



or if we set _α_ even smaller using the fact that (1 + _δ_ 1 _k_ ) _k_ _≤_ (1 + _k_ ) _k_ _≤_ 2 _k_ and _∥_ _**M**_ _∥≥_ [2] 3 _[∥]_ _**[X]**_ _[∥]_ [2] [and] [set] [the]

parameter _α_ so that




- _−_ 32 _κ_ [2] H
_∥_ _**X**_ _∥_ [2] _· ∥_ _**U**_ _v_ 1 _∥ℓ_ 2
24 _k_ �min _{n, R}∥_ _**U**_ _∥_ [3] _[.]_



_α_ [2] _≤_




2 _κ_ [2] _∥_ _**U**_ _∥_

_σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]



Hence _t⋆_ _≤_ _t_ _[⋆]_ is satisfied and applying Lemma D.7, we get



_√_
_∥_ _**E**_ _t⋆_ _∥≤_ 8(1 + _δ_ 1



_k_ )� _k_ min _{n, R}_ _[α]_ [3] (D.61)

_∥_ _**M**_ _∥_ _[∥]_ _**[U]**_ _[∥]_ [3][(1 +] _[ µ][∥]_ _**[M]**_ _[∥]_ [)][3] _[t][⋆]_


24


**Implicit Regularization for Tubal Tensors via GD**



_√_

Moreover, using _∥_ _**M**_ _∥≥_ [2] 3 _[∥]_ _**[X]**_ _[∥]_ [2][ from Lemma D.6 with] _[ δ]_ _[≤]_ [1] _[/]_ [3][ and][ (1 +] _[ δ]_ [1]



_√_
_k_ ) _k_ _≤_ 2 _k_, we get



_∥_ _**E**_ _t⋆_ _∥≤_ 12 _k_ �min _{n, R}_ _∥_ _**X**_ _[α]_ [3] _∥_ [2] _[∥]_ _**[U]**_ _[∥]_ [3][(1 +] _[ µ][∥]_ _**[M]**_ _[∥]_ [)][3] _[t][⋆]_


Hence, using that _Zt_ [(] _[j]_ [)] = (Id + _µM_ [(] _[j]_ [)] ) _[t]_ inequality (D.55) will be implied if



12 _k_ �min _{n, R}_ _[α]_ [3]



2 _κ_ [3][2] _[α]_ 1 [min] _≤j≤k_ _[σ][r]_ �(Id + _µM_ [(] _[j]_ [)] ) _[t][⋆]_ [�] _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)] _[,]_




_[α]_

_∥_ _**X**_ _∥_ [2] _[∥]_ _**[U]**_ _[∥]_ [3][(1 +] _[ µ][∥]_ _**[M]**_ _[∥]_ [)][3] _[t][⋆]_ _[≤]_ 2 _[c]_ _κ_ [3]



which is equivalent to



_∥_ _**X**_ _∥_ [2] _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]
_α_ [2] _≤_ _c_ 3



(D.62)
(1 + _µ∥_ _**M**_ _∥_ ) [3] _[t][⋆]_ _[,]_



_∥_ _**X**_ _∥_ [2] _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)] (1 + _µλr_ ( _M_ [(] _[j]_ [)] )) _[t][⋆]_

12 _k_ �min _{n, R}κ_ [2] _∥_ _**U**_ _∥_ [3] (1 + _µ∥_ _**M**_ _∥_ ) [3] _[t][⋆]_



for all _j_ . To proceed further, let us analyze the last factor from above using the definition of _t⋆_ . Note that



(1 + _µλr_ ( _M_ [(] _[j]_ [)] )) _[t][⋆]_ - - 1 + _µλr_ ( _M_ ( _j_ ))

= exp _t⋆_ ln
(1 + _µ∥_ _**M**_ _∥_ ) [3] _[t][⋆]_ (1 + _µ∥_ _**M**_ _∥_ ) [3]



��
_≥_ exp  - _−_ 3 _t⋆_ ln �(1 + _µ∥_ _**M**_ _∥_ ) [3][��]




                 -                 - 1+ _µ_ min1 _≤j≤k σr_ ( _M_ [(] _[j]_ [)]
Now, using the definition of _t⋆_, that is _t⋆_ = _ψ/_ ln 1+ _µ_ max1 _≤j≤k σr_ +1( _M_ [(] _[j]_ [)] )



��
and inequality (D.60), we get



exp        - _−_ 3 _t⋆_ ln �(1 + _µ∥_ _**M**_ _∥_ ) [3][��] _≥_ exp        - _−_ 48 _ψκ_ [2][�] =


Inserting this into inequality (D.62), we get




- - _−_ 48 _κ_ [2]
2 _κ_ [2] _∥_ _**U**_ _∥_

_c_ 3 _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]



(D.63)



_∥_ _**X**_ _∥_ [2] _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]
_α_ [2] _≤_ _c_ 3 12 _k_ ~~�~~ min _{n, R}κ_ [2] _∥_ _**U**_ _∥_ [3]




- - _−_ 48 _κ_ [2]
2 _κ_ [2] _∥_ _**U**_ _∥_

_c_ 3 _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]



_._ (D.64)



For such _α_, we have shown that inequality (D.55) holds, and the condition of Lemma D.7 is fulfilled, which gives us


_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥≤]_ [14(] _[δ]_ [ +] _[ γ]_ [)] _[ ≤]_ _[cκ][−]_ [2] _[,]_ (D.65)


where the last inequality follows from our assumption that _δ_ _≤_ _c_ 1 _κ_ _[−]_ [2] and _µ ≤_ _c_ 3 _κ_ _[−]_ [2] _∥_ _**X**_ _∥_ _[−]_ [2] and from setting the constants
_c_ 1 and _c_ 3 small enough.


Moreover, for each 1 _≤_ _j_ _≤_ _k_, from Lemma D.7 it follows that


_σmin_ ( _**U**_ _t ∗_ _**W**_ _t_ ( _j_ )) _≥_ [1] (D.66)

4 _[αβ,]_

_σ_ 1( _**U**_ _t ∗_ _**W**_ _t,⊥_ ( _j_ )) _≤_ _[κ][−]_ [2] (D.67)

8 _[αβ.]_


where _β_ := min1 _≤j≤k σr_ ( _Zt_ [(] _[j]_ [)] ) _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)][.]


In the remaining part, we will show that _t⋆_, _β_ and _∥_ _**U**_ _t⋆_ _∥_ have the properties stated in the lemma.


Let us start with _t⋆_ . Using the same inequalities for ln(1 + _x_ ) as above and Lemma D.6, one can show



 _µ_ min [(] _[j]_ [)][)]

1 _≤j≤k_ _[σ][r]_ [(] _[M]_
 _≥_




[min]
3 _[µ]_ 1 _≤j≤k_ _[σ][r]_ [(] _[X]_ [(] _[j]_ [)][)][2]











ln



 1 + _µ_ min [(] _[j]_ [)][)]

1 _≤j≤k_ _[σ][r]_ [(] _[M]_




1 + _µ_ max [(] _[j]_ [)][)]
1 _≤j≤k_ _[σ][r]_ [+1][(] _[M]_



1 _≤j≤k_ _−_ _µ_ max [(] _[j]_ [)][)] _[ ≥]_ [2]

1 + _µ_ min [(] _[j]_ [)][)] 1 _≤j≤k_ _[σ][r]_ [+1][(] _[M]_ 3
1 _≤j≤k_ _[σ][r]_ [(] _[M]_



and at the same time



 1 + _µ_ min [(] _[j]_ [)][)]

1 _≤j≤k_ _[σ][r]_ [(] _[M]_








ln



1 + _µ_ max [(] _[j]_ [)][)]
1 _≤j≤k_ _[σ][r]_ [+1][(] _[M]_





  -    _≤_ ln 1 + _µ_ min [(] _[j]_ [)][)] _≤_ _µ_ min [(] _[j]_ [)][)]
1 _≤j≤k_ _[σ][r]_ [(] _[M]_ 1 _≤j≤k_ _[σ][r]_ [(] _[M]_


25


**Implicit Regularization for Tubal Tensors via GD**


_≤_ _µ_ (1 + _δ_ ) min _[≤]_ [4] _[/]_ [3] _[µ]_ [min]
1 _≤j≤k_ _[σ][r]_ [(] _[X]_ [(] _[j]_ [)][)][2] 1 _≤j≤k_ _[σ][r]_ [(] _[X]_ [(] _[j]_ [)][)][2]


which shows that, on the one hand,



1

        - 1+ _µ_ min1 _≤j≤k σr_ ( _M_ [(] _[j]_ [)] )
ln
1+ _µ_ max1 _≤j≤k σr_ +1( _M_ [(] _[j]_ [)] )


and on the other hand


1



1 2
_≤_ [2]

- 3 _µ_ 1 [max] _≤j≤k_ _σr_ ( _X_ [(] _[j]_ [)] ) [2] [=] 3 _µ_ min1 _≤j≤k σr_ ( _X_ [(] _[j]_ [)] ) [2]



3
_≥_

- 4 _µ_ min1 _≤j≤k σr_ ( _X_ [(] _[j]_ [)] ) [2] _[,]_




            - 1+ _µ_ min1 _≤j≤k σr_ ( _M_ [(] _[j]_ [)] )
ln
1+ _µ_ max1 _≤j≤k σr_ +1( _M_ [(] _[j]_ [)] )


which shows the desired properties of _t⋆_ .



Now, we consider _β_ := min1 _≤j≤k σr_ ( _Zt⋆_ ( _j_ )) _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)][.] [By the definition of] _[ Z][t]_ [(] _[j]_ [)] [and inequality (D.60), we get]

�1 + _µσr_ ( _M_ [(] _[j]_ [)] )� _t⋆_ = exp  - _t⋆_ ln(1 + _µσr_ ( _M_ [(] _[j]_ [)] ))� _≤_ exp  - _t⋆_ ln(1 + _µ∥_ _**M**_ _∥_ )�








2 _κ_ [2] _∥_ _**U**_ _∥_

_c_ 3 _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]



�16 _κ_ [2]



_≤_ exp



ln (1 + _µ∥_ _**M**_ _∥_ )
2 _ψ_ 1max _≤j≤k_ - 1+ _µσr_ ( _M_ [(] _[j]_ [)] )



ln - 1+ _µσr_ ( _M_ [(] _[j]_ [)] )
1+ _µσr_ +1( _M_ [(] _[j]_ [)] )





-  _≤_ exp(16 _ψκ_ [2] ) =



_._ (D.68)



Since this holds for all _j_, we have



�16 _κ_ [2]



_β_ _≤_ _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]




2 _κ_ [2] _∥_ _**U**_ _∥_

_c_ 3 _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]



_._



Finally, we come to the properties of _**U**_ _t⋆_ . By the representation _**U**_ _t⋆_ = _**Z**_ _t⋆_ _∗_ _**U**_ 0 + _**E**_ _t⋆_, we get


_∥_ _**U**_ _t⋆_ _∥≤_ _α∥_ _**Z**_ _t⋆_ _∥∥_ _**U**_ _∥_ + _∥_ _**E**_ _t⋆_ _∥._


From (D.55), we get



_∥_ _**E**_ _t∥≤_ _[c]_ [3]




[3] H) _σmax_ ( _**U**_ ) _≤_ _α∥_ _**Z**_ _t∥∥_ _**U**_ _∥,_

2 _κ_ [2] _[α][∥]_ _**[Z]**_ _[t][∥][σ][min]_ [(] _**[V]**_ _**[L]**_



H

_[c]_ [3] _**U**_ ) _≤_ _[c]_ [3]

2 _κ_ [2] _[α][∥]_ _**[Z]**_ _[t][∥][σ][min]_ [(] _**[V]**_ _**[L]**_ 2 _κ_



which allows us to proceed as follows


_∥_ _**U**_ _t⋆_ _∥≤_ 2 _α∥_ _**Z**_ _t⋆_ _∥∥_ _**U**_ _∥≤_ 2 _α_ (1 + _µ∥_ _**M**_ _∥_ ) _[t][⋆]_ _∥_ _**U**_ _∥,_



�16 _κ_ [2]




   -    = 2 _α_ ln _t⋆_ (1 + _µ∥_ _**M**_ _∥_ ) _∥_ _**U**_ _∥≤_ 2 _α∥_ _**U**_ _∥_




2 _κ_ [2] _∥_ _**U**_ _∥_

_c_ 3 _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]




~~�~~
�� _c_ 3 _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]





- _−_ 8 _κ_ [2]



_≤_ 2 _∥_ _**X**_ _∥_



12 _k_ �min _{n, R}κ_ [2] _∥_ _**U**_ _∥_




2 _κ_ [2] _∥_ _**U**_ _∥_

_c_ 3 _σmin_ ( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)]



_≤_ 3 _∥_ _**X**_ _∥,_



where for the second inequality above we used (D.68) and in the last one an upper bound on _α_ from (D.64) has been applied.


The results in Lemma D.8 hold for any initialization _**U**_ . Below, we will use the fact that _**U**_ is a tensor with Gaussian entries.
This yields the following lemma, which shows that with initialization scale _α >_ 0 chosen sufficiently small, the properties
stated in Lemma D.8 hold with high probability.

**Lemma D.9.** _Fix a sufficiently small constant c >_ 0 _._ _Let_ _**U**_ _∈_ R _[n][×][R][×][k]_ _be a random tubal tensor with i.i.d._ _N_ (0 _,_ _R_ [1] [)] _[ entries,]_

_and let ϵ ∈_ (0 _,_ 1) _._ _Assume that A_ : _S_ _[n][×][n][×][k]_ _→_ R _[m]_ _satisfies the S2NRIP_ ( _δ_ 1) _for some constant δ_ 1 _>_ 0 _._ _Also, assume that_


_**M**_ := _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ ) = _**X**_ _∗_ _**X**_ _[⊤]_ + _**E**_


26


**Implicit Regularization for Tubal Tensors via GD**


_with ∥E_ [(] _[j]_ [)] _∥≤_ _δλr_ ( _X_ [(] _[j]_ [)] _X_ [(] _[j]_ [)H] ) _for each_ 1 _≤_ _j_ _≤_ _k, where δ_ _≤_ _c_ 1 _κ_ _[−]_ [2] _._ _Let_ _**U**_ 0 = _α_ _**U**_ _where_



_ϵ∥_ _**X**_ _∥_ [2]

_k_ [2] _n_ [3] _[/]_ [2] _κ_ [2]




- 2 _κ_ [2] _kn_ [3] _[/]_ [2]



_c_ 3 min _{n, R}_ [3] _[/]_ [2] _ϵ_




_−_ 24 _κ_ [2]




_if R ≥_ 3 _r_


_if R <_ 3 _r_



_._



_α_ [2] ≲











_ϵ_ min _{n, R}∥_ _**X**_ _∥_ [2]

_k_ [2] _n_ [3] _[/]_ [2] _κ_ [2]




_−_ 24 _κ_ [2]





- 2 _κ_ 2 _kn_ 3 _/_ 2

_c_ 3 _r_ [1] _[/]_ [2] _ϵ_



_Assume the step size satisfies µ ≤_ _c_ 2 _κ_ _[−]_ [2] _∥_ _**X**_ _∥_ [2] _._ _Then, with probability at least_ 1 _−_ _p where_



_p_ =




_k_ ( _Cϵ_ [˜] ) _[R][−][r]_ [+1] + _ke_ _[−][cR]_ [˜] _if R ≥_ 2 _r_
_kϵ_ [2] + _ke_ _[−][cR]_ [˜] _if R <_ 2 _r_



_the following statement holds._ _After_



_µ_ min1 _≤j≤_ 1 _k σr_ ( _X_ [(] _[j]_ [)] ) [2] [ln] - 2 _κc_ [2] 3 ~~_[√]_~~ _ϵrn_




2 _κ_ [2] _[√]_ _n_



_c_ [2] 3 _ϵrn_ - _if R <_ 3 _r_







_if R ≥_ 3 _r_



_t⋆_ ≲











1
_µ_ min1 _≤j≤k σr_ ( _X_ [(] _[j]_ [)] ) [2] [ln]



_c_ 3 _ϵ_ �min _{n_ ; _R}_



_iterations, it holds that_


_and for each_ 1 _≤_ _j_ _≤_ _k, we have_


_where_


_and_



_√_
_ϵ_



_∥_ _**U**_ _t⋆_ _∥≤_ 3 _∥_ _**X**_ _∥_ (D.69)

_∥_ _**V**_ _**X**_ _⊥_ _∗_ _**V**_ _**U**_ _t⋆_ _∗_ _**W**_ _t⋆_ _∥≤_ _cκ_ _[−]_ [2] _._ (D.70)


  - ( _j_ ) [�]
_σr_ _**U**_ _t⋆_ _∗_ _**W**_ _t⋆_ _≥_ 4 [1] _[αβ]_ (D.71)


 - ( _j_ ) [�]
_σ_ 1 _**U**_ _t⋆_ _∗_ _**W**_ _t⋆,⊥_ _≤_ _[κ]_ 8 _[−]_ [2] _[αβ]_ (D.72)

(D.73)



�16 _κ_ [2]



_k_




2 _κ_ [2] _[√]_ _n_



_c_ 3 _ϵ_ �min _{n_ ; _R}_



_if R ≥_ 3 _r_


_if R <_ 3 _r_



_β_ ≲




 _ϵ√_









- 2 _κ_ 2 _√rn_
_c_ 3 _ϵ_







_√_
_ϵ_ _k_


_r_



�16 _κ_ [2]



_β_ ≳








_ϵ_ _k_ _if R ≥_ 3 _r_

_√_
_ϵ_ _k_







_k_ _._
_if R <_ 3 _r_
_r_



_Proof._ By Lemma I.3, we have that _∥_ _**U**_ _∥_ ≲




- _k_ max _{n, R}_



_{n, R}_ - _kn_

= [probability] [at] [least] [1] _[−]_
_R_ min _{n_ ; _R}_ [with]



_O_ ( _ke_ _[−][c]_ [ max] _[{][n,R][}]_ ). Also, by Lemma I.4, we have that _∥_ _**U**_ ~~_H_~~ _**v**_ 1 _∥ℓ_ 2 = _∥_ _**U**_ _⊤_ _∗_ _**V**_ 1 _∥F_ _≍_ _√_



_O_ ( _ke_ ). Also, by Lemma I.4, we have that _∥_ _**U**_ _**v**_ 1 _∥ℓ_ 2 = _∥_ _**U**_ _∗_ _**V**_ 1 _∥F_ _≍_ _k_ with probability at least

1 _−_ _O_ ( _ke_ _[−][cR]_ ). Since _**U**_ _∈_ R _[n][×][R][×][k]_ has i.i.d. _N_ (0 _,_ [1] [)] [entries] [and] _**[V]**_ _[⊤]_ _[∗]_ _**[V]**_ _**[L]**_ [=] _**[I]**_ [,] [by] [rotational] [invariance,]



1 _−_ _O_ ( _ke_ _[−][cR]_ ). Since _**U**_ _∈_ R _[n][×][R][×][k]_ has i.i.d. _N_ (0 _,_ _R_ [1] [)] [entries] [and] _**[V]**_ _[⊤]_ _**L**_ _[∗]_ _**[V]**_ _**[L]**_ [=] _**[I]**_ [,] [by] [rotational] [invariance,]

_**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ _[∈]_ [R] _[r][×][R][×][k]_ [also has i.i.d.] _[N]_ [(0] _[,]_ _R_ [1] [)][ entries.] [Hence, the lower bound on] _[ σ]_ [min][(] _**[V]**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)][ in Lemma I.2 applies.] [If]



_**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ _[∈]_ [R] _[r][×][R][×][k]_ [also has i.i.d.] _[N]_ [(0] _[,]_ _R_ [1] [)][ entries.] [Hence, the lower bound on] _[ σ]_ [min][(] _**[V]**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)][ in Lemma I.2 applies.] [If]

_r_ _≤_ _R ≤_ 2 _r_, we have



_√_
_σ_ min( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)] _[ ≥]_ ~~_√_~~ _[ϵ]_



_√_

_k_

≳ _[ϵ]_
_rR_ _r_



_k_
~~_√_~~ _[ϵ]_



_k_

_r_



27


**Implicit Regularization for Tubal Tensors via GD**


with probability at least 1 _−_ _kϵ_ [2] . If 2 _r_ _< R <_ 3 _r_, we have



_√_
_σ_ min( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)] _[ ≥]_ _[ϵ]_



_√_

_R −_ (2 _r −_ 1))

≳ _[ϵ]_
_R_ + ~~_[√]_~~ 2 _r −_ 1) _r_



_√_
_k_ (



_R −_ _[√]_ 2 _r −_ 1)
~~_√_~~




_[√]_ 2 _r −_ 1) _√_

_≥_ ~~_√_~~ _[ϵ]_
_R_ _r_



_k_ ( _R −_ (2 _r −_ 1))
~~_√_~~ _[ϵ]_ ~~_√_~~
_r_ ( _R_ + ~~_[√]_~~ 2 _r −_



_k_

_r_



with probability at least 1 _−_ _k_ ( _Cϵ_ ) _[R][−]_ [2] _[r]_ [+1] _−_ _ke_ _[−][cR]_ . If _R ≥_ 3 _r_, we have



_√_
_σ_ min( _**V**_ _[⊤]_ _**L**_ _[∗]_ _**[U]**_ [)] _[ ≥]_ _[ϵ]_



_√_
_k_ (



_R −_ _[√]_ 2 _r −_ 1) _√_
~~_√_~~ = _ϵ_

_R_




 - _k_ 1 _−_ 2 _rR−_ 1



_k_




- _√_
≳ _ϵ_



with probability at least 1 _−_ _k_ ( _Cϵ_ ) _[R][−]_ [2] _[r]_ [+1] _−_ _ke_ _[−][cR]_ .



Therefore, the above bounds on _∥_ _**U**_ _∥_, _∥_ _**U**_ ~~_H_~~ _**v**_ 1 _∥ℓ_ 2, and _σ_ min( _**V**_ _⊤_ _**L**_ _[∗]_ _**[U]**_ [)][ all hold simultaneously with probability at least][ 1] _[ −]_ _[p]_
where



_p_ =




_k_ ( _Cϵ_ [˜] ) _[R][−][r]_ [+1] + _ke_ _[−][cR]_ [˜] if _R ≥_ 2 _r_
_kϵ_ [2] + _ke_ _[−][cR]_ [˜] if _R <_ 2 _r_ _[.]_



Provided that all three of these bounds hold, one can substitute these into Lemma D.8 to obtain the desired result.


**E. Analysis of Convergence Stage**


In this section, we will prove that after passing the spectral stage, _**U**_ _t_ _∗_ _**U**_ _[⊤]_ _t_ [goes into the convergence process towards]
the ground truth tensor _**X**_ _∗_ _**X**_ _[⊤]_ in the Frobenius norm. For this, we will first show that in each of the tensor slices
_σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1(] _[j]_ [)][)][ grows exponentially, see Lemma E.1, whereas the noise terms] _[ ∥]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [+1] _[,][⊥]_ [(] _[j]_ [)] _[∥]_ [,][ 1] _[ ≤]_ _[j]_ _[≤]_ _[k]_ [, grow]
slower, see Lemma E.3. Moreover, in Lemma E.5, we show that the tensor column spaces of the signal term _**U**_ _t ∗_ _**W**_ _t_ and
the ground truth _**X**_ stay aligned. With this, and several auxiliary lemmas in place, we show that


**Lemma E.1.** _Assume that the following conditions hold_


_µ ≤_ _c∥_ _**X**_ _∥_ _[−]_ [2] _κ_ _[−]_ [2]


_∥_ _**U**_ _t∥≤_ 3 _∥_ _**X**_ _∥_

_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥≤]_ _[cκ][−]_ [1]


_and_
_∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥≤]_ _[cσ]_ _min_ [2] [(] _**[X]**_ [)] _[.]_ (E.1)

_Moreover, assume that_ _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ _[has full tubal rank with all invertible t-SVD-singular tubes.]_ _[Then, for each][ j][,]_ [ 1] _[ ≤]_ _[j]_ _[≤]_ _[k][, it]_
_holds that_

_σmin_ ( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] ( _j_ )) _≥_ _σmin_ ( _**V**_ _⊤_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ ( _j_ )) _≥_ _σmin_ ( _**V**_ _⊤_ _**X**_ _[∗]_ _**[U]**_ _[t]_ ( _j_ ))�1 + [1] _min_ [(] _**[X]**_ [)] _[ −]_ _[µσ]_ _min_ [2] [(] _**[V]**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ ( _j_ ))� _._

4 _[µσ]_ [2]


_Proof._ Consider the tensor _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [.] [Using the definition of] _**[ U]**_ _[t]_ [+1] [in terms of] _**[ U]**_ _[t]_ [, we can rewrite it as]


_**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [=] _**[ V]**_ _[⊤]_ _**X**_ _[∗]_       - _I_ + _µA_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)]       - _∗_ _**U**_ _t ∗_ _**W**_ _t._


This representation leads to the following representation of the RHS above in the Fourier domain


_V_ [(] _**X**_ _[j]_ [) H] (Id + _µ_          - _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] �( _j_ )� _U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] := _H_ [(] _[j]_ [)] _._


Note that here - _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] �( _j_ ) can not be represented as an independent slice of measurements of

_X_ [(] _[j]_ [)] _X_ [(] _[j]_ [)H] _−_ _U_ _t_ [(] _[j]_ [)] _[U]_ [(] _t_ _[j]_ [)H] as it involved the information about all the slices 1 _≤_ _j_ _≤_ _k_ .


Due to our assumptions on _∥_ _**U**_ _t∥_ and the tensor spectral norm property, we get


_∥V_ [(] _**X**_ _[j]_ [) H] _U_ _t_ [(] _[j]_ [)] _[∥≤∥][U]_ [(] _t_ _[j]_ [)] _[∥≤∥]_ _**[U]**_ _[t][∥≤]_ [3] _[∥]_ _**[X]**_ _[∥][.]_


28


**Implicit Regularization for Tubal Tensors via GD**


This in turn is leading to
_µ ≤_ _c∥_ _**X**_ _∥_ _[−]_ [2] _κ_ _[−]_ [2] _≤_ _c_ ˜ _∥V_ [(] _**X**_ _[j]_ [) H] _U_ [(] _t_ _[j]_ [)] _[∥][−]_ [2] _[.]_

This property of _µ_ together with the nature of _W_ [(] _t_ _[j]_ [)] and _V_ [(] _**X**_ _[j]_ [)] [coming along from the signal-noise-term decomposition][ (C.1)]
leads to the fulfilled conditions of Lemma H.2. Applying Lemma H.2 to the matrix _H_ [(] _[j]_ [)], the smallest singular value of
matrix _H_ [(] _[j]_ [)] can be estimated as


_σmin_ ( _H_ [(] _[j]_ [)] ) _≥_ �1+ _µσmin_ [2] [(] _[X]_ [(] _[j]_ [)][)] _[−]_ _[µ][∥][P]_ [ (] 1 _[j]_ [)] _[∥−]_ _[µ][∥][P]_ [ (] 2 _[j]_ [)] _[∥−]_ _[µ]_ [2] _[∥][P]_ [ (] 3 _[j]_ [)] _[∥]_ - _σmin_ ( _V_ [(] _**X**_ _[j]_ [) H] _U_ [(] _t_ _[j]_ [)][)] �1 _−_ _µσmin_ [2] [(] _[V]_ [(] _**X**_ _[j]_ [) H] _U_ [(] _t_ _[j]_ [)][)] - _._ (E.2)


with

_∥P_ 1 [(] _[j]_ [)] _[∥≤]_ [4] _[∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[∥]_ [2] _[∥][V]_ [(] _**X**_ _[j]_ [)] _[⊥]_ _[V]_ _U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[∥]_ [2]

_∥P_ 2 [(] _[j]_ [)] _[∥≤]_ [4] ���� _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] �( _j_ ) _−_ _X_ ( _j_ ) _X_ ( _j_ )H + _U_ ( _tj_ ) _[U]_ [(] _t_ _[j]_ [)H] ���

_∥P_ 3 [(] _[j]_ [)] _[∥≤]_ [2] _[∥][X]_ [(] _[j]_ [)] _[∥]_ [2] _[∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[∥]_ [2] _[.]_


Further, we will make the above bounds for _∥Pi_ [(] _[j]_ [)] _∥, i ∈{_ 1 _,_ 2 _,_ 3 _},_ more precise using information about the tensor setting.

First of all since _∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[∥≤∥][U]_ [(] _t_ _[j]_ [)] _[∥≤∥]_ _**[U]**_ _[t][∥≤]_ [3] _[∥]_ _**[X]**_ _[∥]_ [,] [we] [get] _[∥][P]_ [ (] 1 _[j]_ [)] _[∥≤]_ [36] _[∥]_ _**[X]**_ _[∥]_ [2] _[∥][V]_ [(] _**X**_ _[j]_ [)] _[⊥]_ _[V]_ _U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[∥]_ [2][.] [Moreover,]

since _V_ _**X**_ [(] _[j]_ [)] _[⊥]_ _[V]_ _U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] = _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[V]**_ _**[U]**_ _t_ _[∗]_ _**[W]**_ _t_ ( _j_ ) and _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥≤]_ _[cκ][−]_ [1][ due to the assumption, it follows that for each]

_j,_ 1 _≤_ _j_ _≤_ _k_, it holds that _∥V_ [(] _**X**_ _[j]_ [)] _[⊥]_ _[V]_ _U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[∥≤]_ _[cκ][−]_ [1][.] [This allows for the following estimation]

_∥P_ 1 [(] _[j]_ [)] _[∥≤]_ [36] _[∥]_ _**[X]**_ _[∥]_ [2] _[cκ][−]_ [1] _[≤]_ [1] _min_ [(] _**[X]**_ [)] _[,]_

4 _[σ]_ [2]


where the last inequality follows from the fact that _c >_ 0 is small enough.


Before proceeding with _∥P_ 2 [(] _[j]_ [)] _[∥]_ [, consider]

( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [) = (] _[A][∗][A]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[ −]_        - _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_        - _._


The RHS from above has the following slices in the Fourier domain


( _A_ _[∗]_ _A_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)][(] _[j]_ [)] _[ −]_                - _X_ [(] _[j]_ [)] _X_ [(] _[j]_ [)H] _−_ _U_ [(] _t_ _[j]_ [)] _[U]_ [(] _t_ _[j]_ [)H]                - _,_


the norm of which (due to assumption (E.1) and the definition of the tensor spectral norm) can be bounded as


_∥_ ( _A_ _[∗]_ _A_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)][(] _[j]_ [)] _[ −]_ - _X_ [(] _[j]_ [)] _X_ [(] _[j]_ [)H] _−_ _U_ [(] _t_ _[j]_ [)] _[U]_ [(] _t_ _[j]_ [)H] - _∥≤∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥≤]_ _[cσ]_ _min_ [2] [(] _**[X]**_ [)] _[.]_


This leads to the following estimation
_∥P_ 2 [(] _[j]_ [)] _[∥≤]_ [4] _[cσ]_ _min_ [2] [(] _**[X]**_ [)]

To further assess _∥P_ 3 [(] _[j]_ [)] _[∥]_ [, we take into account that matrix] _[ W]_ [(] _t_ _[j]_ [)] is an orthogonal matrix and the assumption _∥_ _**U**_ _t∥≤_ 3 _∥_ _**X**_ _∥_,
which allows for the next bound


_∥P_ 3 [(] _[j]_ [)] _[∥≤]_ [2] _[∥][X]_ [(] _[j]_ [)] _[∥]_ [2] _[∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[∥]_ [2] _[≤]_ [2] _[∥]_ _**[X]**_ _[∥]_ [2] _[∥][U]_ [(] _t_ _[j]_ [)] _[∥]_ [2] _[≤]_ [2] _[∥]_ _**[X]**_ _[∥]_ [2] _[∥]_ _**[U]**_ _[t][∥]_ [2] _[≤]_ [18] _[∥]_ _**[X]**_ _[∥]_ [4] _[.]_


Inserting the newly obtained estimates for _∥Pi_ [(] _[j]_ [)] _∥, i ∈{_ 1 _,_ 2 _,_ 3 _},_ into (E.2), we get

_σmin_ ( _H_ [(] _[j]_ [)] ) _≥_ (1 + _µσmin_ [2] [(] _[X]_ [(] _[j]_ [)][)] _[ −]_ _[µ]_ _min_ [(] _**[X]**_ [)] _[ −]_ [4] _[µcσ]_ _min_ [2] [(] _**[X]**_ [)] _[ −]_ [18] _[µ]_ [2] _[∥]_ _**[X]**_ _[∥]_ [4][)] _[·]_

4 _[σ]_ [2]

_· σmin_ ( _V_ [(] _**X**_ _[j]_ [) H] _U_ [(] _t_ _[j]_ [)][)] �1 _−_ _µσmin_ [2] [(] _[V]_ [(] _**X**_ _[j]_ [) H] _U_ [(] _t_ _[j]_ [)][)]                        
_≥_ (1 + _µσmin_ [2] [(] _**[X]**_ [)] _[ −]_ _[µ]_ _min_ [(] _**[X]**_ [)] _[ −]_ [4] _[µcσ]_ _min_ [2] [(] _**[X]**_ [)] _[ −]_ [18] _[µ]_ [2] _[∥]_ _**[X]**_ _[∥]_ [4][)] _[σ][min]_ [(] _[V]_ [(] _**X**_ _[j]_ [) H] _U_ [(] _t_ _[j]_ [)][)] �1 _−_ _µσmin_ [2] [(] _[V]_ [(] _**X**_ _[j]_ [) H] _U_ [(] _t_ _[j]_ [)][)]    - _._

4 _[σ]_ [2]


Now, according to the assumption on _µ_, we get


_min_ [(] _**[X]**_ [)]
_µ_ [2] _∥_ _**X**_ _∥_ [4] _≤_ _µcκ_ _[−]_ [2] _∥_ _**X**_ _∥_ _[−]_ [2] _∥_ _**X**_ _∥_ [4] = _µc_ _[σ]_ [2] _∥_ _**X**_ _∥_ _[−]_ [2] _∥_ _**X**_ _∥_ [4] = _cµσmin_ [2] [(] _**[X]**_ [)]

_∥_ _**X**_ _∥_ [2]


29


**Implicit Regularization for Tubal Tensors via GD**


Taking _c_ small enough allows for the following estimation

_σmin_ ( _H_ [(] _[j]_ [)] ) _≥_ _σmin_ ( _V_ [(] _**X**_ _[j]_ [) H] _U_ [(] _t_ _[j]_ [)][)] �1 + [1] _min_ [(] _**[X]**_ [)] ��1 _−_ _µσmin_ [2] [(] _[V]_ [(] _**X**_ _[j]_ [) H] _U_ [(] _t_ _[j]_ [)][)]    
2 _[µσ]_ [2]

= _σmin_ ( _V_ [(] _**X**_ _[j]_ [) H] _U_ [(] _t_ _[j]_ [)][)] �1 + [1] _min_ [(] _**[X]**_ [)] �1 _−_ _µσmin_ [2] [(] _[V]_ [(] _**X**_ _[j]_ [) H] _U_ [(] _t_ _[j]_ [)][)]      - _−_ _µσmin_ [2] [(] _[V]_ [(] _**X**_ _[j]_ [) H] _U_ [(] _t_ _[j]_ [)][)]      
2 _[µσ]_ [2]


Now, since _σmin_ ( _V_ [(] _**X**_ _[j]_ [) H] _U_ [(] _t_ _[j]_ [)][)] _[ ≤]_ _[σ][min]_ [(] _[U]_ [(] _t_ _[j]_ [)][)] _[ ≤∥]_ _**[U]**_ _[t][∥≤]_ [3] _[∥]_ _**[X]**_ _[∥]_ [, we have that]

_µσmin_ [2] [(] _[V]_ [(] _**X**_ _[j]_ [) H] _U_ [(] _t_ _[j]_ [)][)] _[ ≤]_ _[µ]_ [9] _[∥]_ _**[X]**_ _[∥]_ [2] _[≤]_ [9] _[cκ][−]_ [2] _[≤]_ [1]

2


due to the fact that _c >_ 0 can be chosen small enough. The last part of Lemma’s proof follows from
_σmin_ ( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1(] _[j]_ [)][)] _[ ≥]_ _[σ][min]_ [(] _**[V]**_ _**X**_ _[⊤]_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [(] _[j]_ [)][)] [and] _[σ][min]_ [(] _**[V]**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [(] _[j]_ [)][)] [=] _[σ][min]_ [(] _[H]_ [(] _[j]_ [)][)][,] [which] [completes]
the argument.


The next two lemmas will allow us to show that in each of the Fourier slices the noise term part of the gradient descent
iterates is growing slower than its signal term part.

**Lemma E.2.** _Assume that µ ≤_ _c_ min - 101 _[∥]_ _**[X]**_ _[∥][−]_ [2] _[,][ ∥]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][−]_ [1][�] _and ∥_ _**U**_ _t∥≤_ 3 _∥_ _**X**_ _∥._ _Moreover,_

_suppose that_ _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ _[has full tubal rank with all invertible t-SVD-tubes and][ ∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥≤]_ _[cκ][−]_ [1] _[ with a sufficiently]_
_small contact c >_ 0 _._ _Then, the principal angle between_ _**V**_ _**X**_ _⊥_ _and_ _**V**_ _**U**_ _t_ +1 _∗_ _**W**_ _t_ _can be bounded as follows_


_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t][∥≤]_ [2] _[∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ [+ 2] _[µ][∥]_ [(] _[A][∗][A]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][.]_


_In particular, it holds that ∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t][∥≤]_ [1] _[/]_ [50] _[.]_


_Proof._ By the definition of _**U**_ _t_ +1, we have


            -             _**U**_ _t_ +1 _∗_ _**W**_ _t_ = _I_ + _µA_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _∗_ _**U**_ _t ∗_ _**W**_ _t_ _∈_ R _[n][×][r][×][k]_ _,_


which allows for the following representation in the Fourier domain



_**U**_ _t_ +1 _∗_ _**W**_ _t_ ( _j_ ) = �Id + _µA_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)][(] _[j]_ [)][�]



_**U**_ _t ∗_ _**W**_ _t_ ( _j_ ) _∈_ C _[n][×][r]_ _,_ 1 _≤_ _j_ _≤_ _k._



Consider the SVD decomposition _**U**_ _t ∗_ _**W**_ _t_ [(] _[j]_ [)] = _V_ _**U**_ _t∗_ _**W**_ _t_ ( _j_ )Σ _**U**_ _t∗_ _**W**_ _t_ ( _j_ ) _W_ _**U**_ [H] _t∗_ _**W**_ _t_ [(] _[j]_ [)] [and denote by] _[ Z]_ [(] _[j]_ [)] [the matrix]


           _Z_ [(] _[j]_ [)] := Id + _µA_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)][(] _[j]_ [)][�] _V_ _**U**_ _t∗_ _**W**_ _t_ ( _j_ ) _∈_ C _[n][×][r]_ _._


Since by assumption _**U**_ _t ∗_ _**W**_ _t_ [(] _[j]_ [)] has full rank (due to full-rankness of _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [, see Lemma C.1), matrix] _[ Z]_ [(] _[j]_ [)][ has the same]
column space as _**U**_ _t_ +1 _∗_ _**W**_ _t_ [(] _[j]_ [)] and the principal angle between tensor subspaces _**V**_ _**X**_ _⊥_ and _**V**_ _**U**_ _t_ +1 _∗_ _**W**_ _t_ can be computed
via _Z_ [(] _[j]_ [)] as

_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t][∥]_ [=] 1 [max] _≤j≤k_ _[∥][V]_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ [(] _**U**_ _[j]_ _t_ [)] +1 _∗_ _**W**_ _t_ _[∥]_ [=] 1 [max] _≤j≤k_ _[∥][V]_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ _**[U]**_ _t_ _[∗]_ _**[W]**_ _t_ [(] _[j]_ [)] _[∥]_ [=] 1 [max] _≤j≤k_ _[∥][V]_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V][Z]_ [(] _[j]_ [)] _[∥][.]_


Now, we will consider each of the terms _∥V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V][Z]_ [(] _[j]_ [)] _[∥]_ [separately and bound them as follows]


[(] _[j]_ [)H]
_∥V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V][Z]_ [(] _[j]_ [)] _[∥≤∥][V]_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V][Z]_ [(] _[j]_ [)][Σ] _[Z]_ [(] _[j]_ [)] _[W]_ [ H] _Z_ [(] _[j]_ [)] _[∥∥]_ [(Σ] _[Z]_ [(] _[j]_ [)] _[W]_ [ H] _Z_ [(] _[j]_ [)][)] _[−]_ [1] _[∥]_ [=] _[∥]_ _σ_ _[V]_ min _**X**_ _[⊥]_ ( _Z_ _[Z]_ [(] _[j]_ [(] _[j]_ [)] ) [)] _[∥]_ _[.]_ (E.3)


Using the definition of _Z_ [(] _[j]_ [)], the norm in the numerator above can be estimated as


_∥V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[Z]_ [(] _[j]_ [)] _[∥≤∥][V]_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ _**[U]**_ _t_ _[∗]_ _**[W]**_ _t_ [(] _[j]_ [)] _[∥]_ [+] _[ µ][∥][V]_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[A][∗][A]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)][(] _[j]_ [)] _[∥]_

_≤∥V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ [(] _**U**_ _[j]_ _t_ [)] _∗_ _**W**_ _t_ _[∥]_ [+] _[ µ][∥A][∗][A]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)][(] _[j]_ [)] _[∥]_

_≤∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ [+] _[ µ][∥A][∗][A]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][.]_


30


**Implicit Regularization for Tubal Tensors via GD**


Using again the definition of _Z_ [(] _[j]_ [)] and Weyl’s inequality, the denominator in (E.3) can be estimated from below as follows


_σ_ min( _Z_ [(] _[j]_ [)] ) _≥_ _σ_ min( _V_ _**U**_ _t∗_ _**W**_ _t_ ( _j_ )) _−_ _µ∥_      - _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)][(] _[j]_ [)][�] _V_ _**U**_ _t∗_ _**W**_ _t_ ( _j_ ) _∥_


_≥_ 1 _−_ _µ∥A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)][(] _[j]_ [)] _[∥≥]_ [1] _[ −]_ _[µ][∥A][∗][A]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_

_≥_ 1 _−_ _µ_ ( _∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_ [+] _[ ∥]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_ [)]

_≥_ 1 _−_ _µ_       - _∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_ [+] _[ ∥]_ _**[X]**_ _[∥]_ [2][ +] _[ ∥]_ _**[U]**_ _[t][∥]_ [2][�]

_≥_ 1 _−_ _µ_       - _∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_ [+ 10] _[∥]_ _**[X]**_ _[∥]_ [2][�] _≥_ [1]

2 _[,]_


where the last inequality follows from the assumption on _µ_ . Now, we can come back to the estimation of _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t][∥]_ [,]
which due to the combination of the above-carried estimated reads as


_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t][∥≤]_ [2] _[∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ [+ 2] _[µ][∥A][∗][A]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_


providing the first result from the Lemma. The second bound stated in the Lemma follows from our assumption on
_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ [and] _[ µ]_ [ and the fact that the constant] _[ c]_ [ is chosen small enough to make] _[ ∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t][∥≤]_ 501 [.]


**Lemma E.3.** _Assume that µ ≤_ _c_ 1 min - 101 _[∥]_ _**[X]**_ _[∥][−]_ [2] _[,][ ∥]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][−]_ [1][�] _and ∥_ _**U**_ _t∥≤_ 3 _∥_ _**X**_ _∥._ _More-_

_over,_ _suppose_ _that_ _tensor_ _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ _[has]_ _[all]_ _[invertible]_ _[t-SVD-tubes]_ _[and]_ _[that]_ _[∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥≤]_ _[c]_ [1] _[κ][−]_ [1] _[,]_ _[with]_
_absolute constant c_ 1 _>_ 0 _chosen small enough._ _Then, it holds that_

_∥_ _**U**_ _t_ +1 _∗_ _**W**_ _t_ +1 _,⊥_ ( _j_ ) _∥≤_ �1 _−_ _[µ]_ 2 _[∥]_ _**[U]**_ _[t][ ∗]_ _**[W]**_ _[t,][⊥]_ ( _j_ ) _∥_ 2 + 9 _µ∥_ _**V**_ _⊤_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t]_ ( _j_ ) _∥∥_ _**X**_ _∥_ 2

+ 2 _µ∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_                     - _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ ( _j_ ) _∥_


_for each j, with_ 1 _≤_ _j_ _≤_ _k._


_Proof._ First, we will consider tensor _**U**_ _t_ +1 _∗_ _**W**_ _t_ +1 _,⊥_ splitting it into two different parts, and then will conduct the
corresponding norm estimations of each Fourier slices.

To begin with, note that for the tensor-column space of _**X**_, that is _**V**_ _**X**_, it holds that _**V**_ _**X**_ _∗_ _**V**_ _[⊤]_ _**X**_ [+] _**[ V]**_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ [=] _**[ I]**_ [(see,]
for example, (Liu et al., 2019)). Using this, we can represent _**U**_ _t_ +1 _∗_ _**W**_ _t_ +1 _,⊥_ as follows


_**U**_ _t_ +1 _∗_ _**W**_ _t_ +1 _,⊥_ = _**V**_ _**X**_ _∗_ _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [+1] _[,][⊥]_ [+] _**[V]**_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t]_ [+1] _[ ∗]_ _**[W]**_ _[t]_ [+1] _[,][⊥]_ [=] _**[ V]**_ _**X**_ _[ ⊥]_ _[∗]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t]_ [+1] _[ ∗]_ _**[W]**_ _[t]_ [+1] _[,][⊥]_ [(E.4)]


where the last equality follows from Lemma C.1 due to the property _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [+1] _[,][⊥]_ [= 0][.]

Now, we split the term _**V**_ _**X**_ _⊥_ _∗_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t]_ [+1] _[ ∗]_ _**[W]**_ _[t]_ [+1] _[,][⊥]_ [into two parts using] _**[ W]**_ _[t][ ∗]_ _**[W]**_ _t_ _[⊤]_ [+] _**[ W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _t,_ _[⊤]_ _⊥_ [=] _**[ I]**_ [, which leads]
to


_**V**_ _**X**_ _⊥_ _∗_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [+1] _[,][⊥]_ [=] _**[ V]**_ _**X**_ _[ ⊥]_ _[∗]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ _[∗]_ _**[W]**_ _t_ _[⊤]_ _[∗]_ _**[W]**_ _[t]_ [+1] _[,][⊥]_ [+] _**[V]**_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _t,_ _[⊤]_ _⊥_ _[∗]_ _**[W]**_ _[t]_ [+1] _[,][⊥]_
(E.5)
To estimate the norm of _**V**_ _**X**_ _⊥_ _∗_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[W]**_ _[t]_ [+1] _[,][⊥]_ [in each slice in the Fourier domain, we will use the above-given]
representation and estimate each of the summands individually. Let us start with the second one. Its _j_ th slice in the Fourier
domain reads as


( _**V**_ _**X**_ _⊥_ _∗_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t]_ [+1] _[ ∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _t,_ _[⊤]_ _⊥_ _[∗]_ _**[W]**_ _[t]_ [+1] _[,][⊥]_ [)][(] _[j]_ [)] [=] _[ V]_ [(] _**X**_ _[j]_ [)] _[⊥]_ _[V]_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[U]_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[,]_ [H] _[W]_ [(] _t_ +1 _[j]_ [)] _,⊥_ _[.]_


Due to the orthogonality of the columns of _V_ [(] _**X**_ _[j]_ [)] _[⊥]_ [,] [it] [holds] [that] _[∥][V]_ [(] _**X**_ _[j]_ [)] _[⊥]_ _[V]_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[U]_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[,]_ [H] _[W]_ [(] _t_ +1 _[j]_ [)] _,⊥_ _[∥]_ =

_∥V_ _**X**_ [(] _[j]_ [)H] _[⊥]_ _[U]_ _t_ [(] +1 _[j]_ [)] _[W]_ _t,_ [(] _[j]_ _⊥_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[,]_ [H] _[W]_ [(] _t_ +1 _[j]_ [)] _,⊥_ _[∥]_ [.] [In the Fourier domain, this allows us to focus on] _[ j]_ [th slices of the last one]


_V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[U]_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[,]_ [H] _[W]_ [(] _t_ +1 _[j]_ [)] _,⊥_ [:=] _[ G]_ [(] 2 _[j]_ [)] _[.]_


31


**Implicit Regularization for Tubal Tensors via GD**



Due to the definition of the gradient descent iterates _**U**_ _t_ +1, we have the following representation for its blocks _U_ [(] _t_ +1 _[j]_ [)] [in the]
Fourier domain



_U_ [(] _t_ +1 _[j]_ [)] [=] �Id + _µ_




- _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] �( _j_ ) [�] _U_ [(] _t_ _[j]_ [)]



To upper bound the norm of _G_ [(] 2 _[j]_ [)][, we want to apply Lemma H.3.] [Due to the assumptions in this lemma that] _**[ V]**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_
has full tubal rank with all invertible t-SVD-tubes and _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥≤]_ _[cκ][−]_ [1][ in addition to the conditions on] _[ µ]_ [ and]
the decomposition of gradient descent iterates into the signal and noise term, the conditions of Lemma H.3 are satisfied
for the choice _Y_ 1 = _U_ [(] _t_ +1 _[j]_ [)] [and] _[ Y]_ [=] _[ U]_ [(] _t_ _[j]_ [)] and _Z_ as _Z_ = - _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] �( _j_ ). This allows to upper-bound the

norm of _G_ 2 [(] _[j]_ [)] as follows



_∥G_ 2 [(] _[j]_ [)] _[∥≤∥][U]_ _t_ [(] _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_ �1 _−_ _µ∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_ [2][ +] _[ µ][∥]_ - _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] �( _j_ ) _−_ ( _X_ ( _j_ ) _X_ ( _j_ )H _−_ _U_ ( _tj_ ) _[U]_ [(] _t_ _[j]_ [)H] ) _∥_ 


+ _µ_ [2][�] _∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[∥]_ [2][ +] _[ ∥]_ - _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] �( _j_ ) _−_ ( _X_ ( _j_ ) _X_ ( _j_ )H _−_ _U_ ( _tj_ ) _[U]_ [(] _t_ _[j]_ [)H] ) _∥_ - _∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_ [3]



Using now the fact that for each _j_ it holds that


_∥_    - _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] �( _j_ ) _−_ ( _X_ ( _j_ ) _X_ ( _j_ )H _−_ _U_ ( _tj_ ) _[U]_ [(] _t_ _[j]_ [)H] ) _∥≤∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_


and that _∥U_ _t_ [(] _[j]_ [)] _[∥≤∥]_ _**[U]**_ _[t][∥≤]_ [3] _[∥]_ _**[X]**_ _[∥]_ [, we can proceed with the bound for the norm of] _[ G]_ [(] 2 _[j]_ [)] as below


_∥G_ [(] 2 _[j]_ [)] _[∥≤∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_ �1 _−_ _µ∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_ [2][ +] _[ µ][∥]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_      
+ _µ_ [2][�] 9 _∥_ _**X**_ _∥_ [2] + _∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_       - _∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_ [3]


Further, using the assumption _µ ≤_ _c_ 1 min - 101 _[∥]_ _**[X]**_ _[∥][−]_ [2] _[,][ ∥]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][−]_ [1][�], we get



_∥G_ 2 [(] _[j]_ [)] _[∥≤∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_ �1 _−_ _µ∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_ [2][ +] _[ µ][∥]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_ - + _[µ]_



= _∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_ �1 _−_ _[µ]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_ [2][ +] _[ µ][∥]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_ - _._

2 _[∥][U]_




[(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_ [3]
2 _[∥][U]_



Now, let us return to the first summand in (E.5), that is _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ _[∗]_ _**[W]**_ _[⊤]_ _t_ _[∗]_ _**[W]**_ _[t]_ [+1] _[,][⊥]_ [.] [Using again the fact that]
_**V**_ _**X**_ _∗_ _**U**_ _t_ +1 _∗_ _**W**_ _t_ +1 _,⊥_ = 0 allows us to rewrite it as


_**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ _[∗]_ _**[W]**_ _[⊤]_ _t_ _[∗]_ _**[W]**_ _[t]_ [+1] _[,][⊥]_ [=] _[ −]_ _**[V]**_ _**X**_ _[⊤]_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ _[∗]_ _**[W]**_ _[t]_ [+1] _[,][⊥]_ (E.6)


Moreover, for the same summand, the corresponding _j_ th slice in the Fourier domain reads as


_V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[U]_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)H] _W_ [(] _t_ +1 _[j]_ [)] _,⊥_ [:=] _[ G]_ [(] 1 _[j]_ [)] _[.]_


Due to relation (E.6) in the tensor domain, in the Fourier domain it holds that


_V_ [(] _**X**_ _[j]_ [)H] _U_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)H] _W_ [(] _t_ +1 _[j]_ [)] _,⊥_ [=] _[ −][V]_ [(] _**X**_ _[j]_ [)H] _U_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _,⊥_ _[,]_


which allows to represent _W_ [(] _t_ _[j]_ [)H] _W_ [(] _t_ +1 _[j]_ [)] _,⊥_ [as]


_W_ [(] _t_ _[j]_ [)H] _W_ [(] _t_ +1 _[j]_ [)] _,⊥_ [=] _[ −]_      - _V_ [(] _**X**_ _[j]_ [)H] _U_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)]      - _−_ 1 _V_ [(] _**X**_ _[j]_ [)H] _U_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _,⊥_ _[.]_


Note that the matrix on the RHS above is invertible due to the assumption that _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [has full tubal rank with all]
invertible t-SVD-tubes. From here, _G_ [(] 1 _[j]_ [)] can be represented as

_G_ [(] 1 _[j]_ [)] = _V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[U]_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)]      - _V_ [(] _**X**_ _[j]_ [)H] _U_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)]      - _−_ 1 _V_ [(] _**X**_ _[j]_ [)H] _U_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _,⊥_ _[.]_


32


**Implicit Regularization for Tubal Tensors via GD**


According to Lemma H.3, the norm of _G_ [(] 1 _[j]_ [)] can be bounded from above as



_∥G_ 1 [(] _[j]_ [)] _[∥≤]_ [2] _[µ]_ - _∥V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ _U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[∥∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[∥]_ [2][ +] _[ ∥]_

_· ∥V_ _**X**_ [(] _[j]_ [)H] _[⊥]_ _[V]_ _U_ [(] _t_ _[j]_ +1 [)] _[W]_ [(] _t_ _[j]_ [)] _[∥∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_




- _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] �( _j_ ) _−_ ( _X_ ( _j_ ) _X_ ( _j_ )H _−_ _U_ ( _tj_ ) _[U]_ [(] _t_ _[j]_ [)H] ) _∥_ - _·_



_≤_ 2 _µ_   - _∥V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ _U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[∥∥][U]_ [(] _t_ _[j]_ [)] _[∥]_ [2][ +] _[ ∥]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_   - _· ∥V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ _U_ [(] _t_ _[j]_ +1 [)] _[W]_ [(] _t_ _[j]_ [)] _[∥∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_

_≤_ 2 _µ_   - _∥V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ _U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[∥∥][U]_ [(] _t_ _[j]_ [)] _[∥]_ [2][ +] _[ ∥]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_   - _· ∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t][∥∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_


Due to _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t][∥≤]_ 501 [from Lemma E.2, the fact that] _[ ∥][U]_ [(] _t_ _[j]_ [)] _[∥≤∥]_ _**[U]**_ _[t][∥]_ [, and our assumption that] _[ ∥]_ _**[U]**_ _[t][∥≤]_ [3] _[∥]_ _**[X]**_ _[∥]_ [,]
the norm of _G_ 1 [(] _[j]_ [)] can be further bounded as

_∥G_ 1 [(] _[j]_ [)] _[∥≤]_ _[µ]_ �9 _∥V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ _U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[∥∥]_ _**[X]**_ _[∥]_ [2][ +] _[ ∥]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_    - _∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_

= _µ_ �9 _∥_ ( _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t]_ [)][(] _[j]_ [)] _[∥∥]_ _**[X]**_ _[∥]_ [2][ +] _[ ∥]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_      - _∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥][.]_


Since due to representation (E.4), it holds that _∥_ - _**U**_ _t_ +1 _∗_ _**W**_ _t_ +1 _,⊥_ �( _j_ ) _∥_ = _∥_ - _**V**_ _**X**_ _⊥_ _∗_ _**U**_ _t_ +1 _∗_ _**W**_ _t_ +1 _,⊥_ �( _j_ ) _∥,_ combining the



inequalities for _∥G_ [(] 1 _[j]_ [)] _[∥]_ [and] _[ ∥][G]_ 2 [(] _[j]_ [)] _[∥]_ [together with] _[ U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] [=]




- _**U**_ _t ∗_ _**W**_ _t,⊥_ �( _j_ ) leads to the final result




- _**U**_ _t_ +1 _∗_ _**W**_ _t_ +1 _,⊥_ �( _j_ ) _∥≤_ �1 _−_ _[µ]_



_∥_



2 _[∥]_




- _**U**_ _t ∗_ _**W**_ _t,⊥_ �( _j_ ) _∥_ 2 + 9 _µ∥_ ( _**V**_ _⊤_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t]_ [)][(] _[j]_ [)] _[∥∥]_ _**[X]**_ _[∥]_ [2]



+ 2 _µ∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_ - _∥_ - _**U**_ _t ∗_ _**W**_ _t,⊥_ �( _j_ ) _∥._



The next lemma shows that the tensors _**W**_ _t_ and _**W**_ _t_ +1 span approximately the same tensor column space.


**Lemma E.4.** _Assume that the following conditions hold_


_∥_ _**U**_ _t∥≤_ 3 _∥_ _**X**_ _∥,_ (E.7)

_µ ≤_ _c∥_ _**X**_ _∥_ _[−]_ [2] _κ_ _[−]_ [2] (E.8)

_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥≤]_ _[cκ][−]_ [1] (E.9)

_∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ ( _j_ ) _∥≤_ 2 _σmin_ ( _**U**_ _t ∗_ _**W**_ _t_ ( _j_ )) _,_ (E.10)

_∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥≤]_ _[cσ]_ _min_ [2] [(] _**[X]**_ [)] _[.]_ (E.11)


_Then it holds that_


_∥_ _**W**_ _[⊤]_ _t,⊥_ _[∗]_ _**[W]**_ _[t]_ [+1] _[∥≤]_ _[µ]_ - 48001 _[σ]_ _min_ [2] [(] _**[X]**_ [)+] _[∥]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t][∥∥]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥][∥]_ - _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ [+4] _[µ][∥]_ [(] _[A][∗][A−I]_ [)(] _**[X]**_ _[ ∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _t_ _[⊤]_ [)] _[∥]_


_and σmin_ ( _**W**_ _[⊤]_ _t_ _[∗]_ _**[W]**_ _[t]_ [+1(] _[j]_ [)][)] _[ ≥]_ 2 [1] _[,]_ [ 1] _[ ≤]_ _[j]_ _[≤]_ _[k][.]_


_Proof._ To bound the norm of _**W**_ _[⊤]_ _t,⊥_ _[∗]_ _**[W]**_ _[t]_ [+1][, we will rewrite] _**[ W]**_ _[⊤]_ _t,⊥_ _[∗]_ _**[W]**_ _[t]_ [+1] [in the Fourier domain with the help of Fourier]
slices of _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [.] [First, note that due to the decomposition of the gradient iterates into the noise and signal term, it holds]
_**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] [=] _**[ V]**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _t_ _[⊤]_ +1 [.] [This allows us to represent the corresponding] _[ j]_ [th Fourier slices of] _**[ V]**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] [as]
_V_ _**X**_ [(] _[j]_ [)H] _U_ _t_ [(] +1 _[j]_ [)] [=] _[ V]_ [(] _**X**_ _[j]_ [)H] _U_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t_ +1 _[j]_ [)H][, which means that for each] _[ j]_ [, the matrices] _[ V]_ [(] _**X**_ _[j]_ [)H] _U_ [(] _t_ +1 _[j]_ [)] [and] _[ V]_ [(] _**X**_ _[j]_ [)H] _U_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t_ +1 _[j]_ [)H]
have the same kernel, and therefore _U_ [(] _t_ +1 _[j]_ [)H] _[V]_ [(] _**X**_ _[j]_ [)] [spans the same subspace as] _[ W]_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t_ +1 _[j]_ [)H] _[U]_ [(] _t_ +1 _[j]_ [)H] _[V]_ [(] _**X**_ _[j]_ [)][.] [Due to this and the]
following representation of the matrices


_U_ [(] _t_ _[j]_ [)] = _U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)H] + _U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)H] (E.12)

_U_ [(] _t_ +1 _[j]_ [)] [=] _[ U]_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t_ +1 _[j]_ [)H] [+] _[ U]_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t_ +1 _[j]_ [)H] _[,]_ (E.13)


33


**Implicit Regularization for Tubal Tensors via GD**


we can apply Lemma H.4 to estimate the norm of _W_ [H] _t,⊥_ _[W]_ [(] _t_ +1 _[j]_ [)] [taking] _[ Y]_ [1] [=] _[ U]_ [(] _t_ +1 _[j]_ [)] [and] _[ Y]_ [=] _[ U]_ [(] _t_ _[j]_ [)] and _Z_ as



_Z_ [(] _[j]_ [)] :=




- _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] �( _j_ ) _._



This gives us the following estimate











_∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[∥∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥∥][V]_ [(] _**X**_ _[j]_ [)H] _VU_ ( _tj_ ) _W_ [(] _t_ _[j]_ [)] _[∥]_ (E.14)



_∥W_ [H] _t,⊥_ _[W]_ [(] _t_ +1 _[j]_ [)] _[∥≤]_ _[µ]_




[(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[∥]_
1 + _µ_ _[∥][Z]_ [(] _[j]_ [)] _[∥∥][U]_



_σ_ min( _V_ [(] _**X**_ _[j]_ [)H] _U_ [(] _t_ +1 _[j]_ [)] [)]




[(] _t_ _[j]_ [)] _[U]_ [(] _t_ _[j]_ [)H] ) _∥_
+ _µ_ _[∥][Z]_ [(] _[j]_ [)] _[ −]_ [(] _[X]_ [(] _[j]_ [)] _[X]_ [(] _[j]_ [)H] _[ −]_ _[U]_ _∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥][.]_

_σ_ min( _V_ [(] _**X**_ _[j]_ [)H] _U_ [(] _t_ +1 _[j]_ [)] [)]



To proceed further with the upper bound above, we will first show that in each Fourier slice it holds that


_σ_ min� _V_ [(] _**X**_ _[j]_ [)H] _U_ [(] _t_ +1 _[j]_ [)]             - _≥_ [1] [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)][)] _[,]_ 1 _≤_ _j_ _≤_ _k._ (E.15)

2 _[σ]_ [min][(] _[U]_


First, note that


_σ_ min� _V_ _**X**_ [(] _[j]_ [)H] _U_ [(] _t_ +1 _[j]_ [)]   - _≥_ _σ_ min� _V_ [(] _**X**_ _[j]_ [)H] _U_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t_ +1 _[j]_ [)]   - = _σ_ min� _V_ [(] _**X**_ _[j]_ [)H] (Id + _µZ_ [(] _[j]_ [)] ) _U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ +1 _[j]_ [)]   

               -               = _σ_ min _V_ [(] _**X**_ _[j]_ [)H] (Id + _µZ_ [(] _[j]_ [)] ) _VU_ ( _tj_ ) _W_ [(] _t_ _[j]_ +1 [)] _[V]_ _U_ [H][(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ +1 [)] _[U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ +1 _[j]_ [)]




   _≥_ _σ_ min _V_ [(] _**X**_ _[j]_ [)H] (Id + _µZ_ [(] _[j]_ [)] ) _VU_ ( _tj_ ) _W_ [(] _t_ _[j]_ +1 [)]




- _· σ_ min� _VU_ [H][(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ +1 [)] _[U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ +1 _[j]_ [)] 



 - _≥_ _σ_ min _V_ [(] _**X**_ _[j]_ [)H] _VU_ ( _tj_ ) _W_ [(] _t_ _[j]_ +1 [)]




- ( _j_ )H - - 
_−_ _µ_ �� _V_ _**X**_ _Z_ [(] _[j]_ [)] _VU_ ( _tj_ ) _W_ [(] _t_ _[j]_ +1 [)] �� _· σ_ min _VU_ [H][(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ +1 [)] _[U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ +1 _[j]_ [)] _._



Due to our assumption (E.9) on the principal angle _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ [and the properties of the tensor slices, we have that]




  _σ_ min _V_ [(] _**X**_ _[j]_ [)H] _VU_ ( _tj_ ) _W_ [(] _t_ _[j]_ +1 [)]




- - _≥_ _σ_ min _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[V]**_ _**[U]**_ _t_ _[∗]_ _**[W]**_ _t_ +1 =




~~�~~



1 _−_ ��� _**V**_ _⊤_ _**X**_ _[∗]_ _**[V]**_ _**[U]**_ _t_ _[∗]_ _**[W]**_ _t_ +1���2 _≥_ [3]



4 _[,]_



where that last inequality can be guaranteed by choosing _c >_ 0 small enough. Thus, to show that relation (E.15) holds we
( _j_ )H 1
need to demonstrate that _µ_ �� _V_ _**X**_ _Z_ [(] _[j]_ [)] _VU_ ( _tj_ ) _W_ [(] _t_ _[j]_ +1 [)] �� be bounded from above by 4 [.] [For this, we will proceed as follows]


( _j_ )H ( _j_ ) ( _j_ ) ( _j_ ) ( _j_ )H ( _j_ ) ( _j_ ) ( _j_ )H ( _j_ )
_µ_ �� _V_ _**X**_ _Z_ [(] _[j]_ [)] _VU_ ( _tj_ ) _W_ [(] _t_ _[j]_ +1 [)] �� _≤_ _µ_ �� _Z_ �� _≤_ _µ_ �� _Z_ _−_ ( _X_ _X_ _−_ _U_ _t_ _[U]_ [(] _t_ _[j]_ [)H] )�� + _µ_ �� _X_ _X_ _−_ _U_ _t_ _[U]_ [(] _t_ _[j]_ [)H] _∥._ (E.16)


By the definition of _Z_ [(] _[j]_ [)], for the first summand from above we have
��� _Z_ ( _j_ ) _−_ ( _X_ ( _j_ ) _X_ ( _j_ )H _−_ _U_ ( _tj_ ) _[U]_ [(] _t_ _[j]_ [)H] )��� = ���� _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] �( _j_ ) _−_ ( _X_ ( _j_ ) _X_ ( _j_ )H _−_ _U_ ( _tj_ ) _[U]_ [(] _t_ _[j]_ [)H] )���

= ���� _I −A_ _[∗]_ _A_ �( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)][(] _[j]_ [)][���]

_≤_ ���� _I −A_ _[∗]_ _A_ �( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] ���


and for the second summand, it holds that


_∥X_ [(] _[j]_ [)] _X_ [(] _[j]_ [)H] _−_ _U_ [(] _t_ _[j]_ [)] _[U]_ [(] _t_ _[j]_ [)H] _∥≤∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ _[∥≤∥]_ _**[X]**_ _[∥]_ [2][ +] _[ ∥]_ _**[U]**_ _[t][∥]_ [2] _[.]_


This allows us to proceed with inequality (E.16) as


( _j_ )H      - 2 2
_µ_ �� _V_ _**X**_ _Z_ [(] _[j]_ [)] _VU_ ( _tj_ ) _W_ [(] _t_ _[j]_ +1 [)] �� _≤_ _µ_ ��� _I −A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] �� + _µ_ ( _∥_ _**X**_ _∥_ + _∥_ _**U**_ _t∥_ )

_≤_ _µ_ ��� _I −A_ _[∗]_ _A_ �( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] �� + 10 _µ∥_ _**X**_ _∥_ 2) _≤_ _µcσ_ min2 [(] _**[X]**_ [) + 11] _[µ][∥]_ _**[X]**_ _[∥]_ [2] _[≤]_ [1]

2 _[,]_


34


**Implicit Regularization for Tubal Tensors via GD**


where in the first line we used assumption (E.7), and in the second assumption(E.11). The third inequality above follows
from our assumption on _µ_ and sufficiently small constant _c >_ 0. This, in turn, shows that relation (E.15) holds and we can
proceed with (E.14) in the following manner











_∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[∥∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥∥][V]_ [(] _**X**_ _[j]_ [)H] _VU_ ( _tj_ ) _W_ [(] _t_ _[j]_ [)] _[∥]_



_∥W_ [H] _t,⊥_ _[W]_ [(] _t_ +1 _[j]_ [)] _[∥≤]_ _[µ]_




[(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[∥]_
1 + 2 _µ_ _[∥][Z]_ [(] _[j]_ [)] _[∥∥][U]_



_σ_ min( _U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)][)]




[(] _t_ _[j]_ [)] _[U]_ [(] _t_ _[j]_ [)H] ) _∥_
+ 2 _µ_ _[∥][Z]_ [(] _[j]_ [)] _[ −]_ [(] _[X]_ [(] _[j]_ [)] _[X]_ [(] _[j]_ [)H] _[ −]_ _[U]_ _∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥][.]_

_σ_ min( _U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)][)]



Now, using assumption (E.10) and the definition of _Z_ [(] _[j]_ [)], we have



_∥W_ [H] _t,⊥_ _[W]_ [(] _t_ +1 _[j]_ [)] _[∥≤]_ _[µ][∥][V]_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ _U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[∥∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[∥∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_



+ 4 _µ∥_




- _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] �( _j_ ) _−_ ( _X_ ( _j_ ) _X_ ( _j_ )H _−_ _U_ ( _tj_ ) _[U]_ [(] _t_ _[j]_ [)H] ) _∥_



+ 4 _µ_ [2] _∥_ - _A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] �( _j_ ) _∥∥U_ ( _tj_ ) _[W]_ [(] _t_ _[j]_ [)] _[∥]_ [2] _[∥][V]_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ _U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[∥]_



_≤_ _µ∥V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ _U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[∥∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[∥∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_

+ 4 _µ∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_

+ 4 _µ_ [2] _∥A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[∥]_ [2] _[∥][V]_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ _U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[∥][.]_


In the last inequality, we used the tensor norm as the maximum norm in each Fourier slice. Note that, similarly to one of the
estimates above, we get


_∥A_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥≤∥]_ _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ _[∥]_ [+] _[ ∥]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_

_≤∥_ _**X**_ _∥_ [2] + _∥_ _**U**_ _t∥_ [2] + _cσ_ min [2] [(] _**[X]**_ [)] _[ ≤]_ [11] _[∥]_ _**[X]**_ _[∥]_ [2] (E.17)


where the last line holds due to the assumption _∥_ _**U**_ _t∥≤_ 3 _∥_ _**X**_ _∥_ and that _c_ is small enough.

Now, since _µ_ _≤_ _c∥_ _**X**_ _∥_ _[−]_ [2] _κ_ _[−]_ [2], _∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[∥≤∥]_ _**[U]**_ _[t][∥≤]_ [3] _[∥]_ _**[X]**_ _[∥]_ [and] _[∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥≤∥]_ _**[U]**_ _[t][∥≤]_ [3] _[∥]_ _**[X]**_ _[∥]_ [,] [constant] _[c]_ _[>]_ [0] [can]
be chosen so that 4 _µ ·_ 11 _∥_ _**X**_ _∥_ [2] _≤_ 48001 _[σ]_ min [2] [(] _**[X]**_ [)][,] [together with][ (E.17)][ and][ (E.11)][ we can proceed with the estimation of]
_W_ [H] _t,⊥_ _[W]_ _t_ [(] +1 _[j]_ [)] [as]

_∥W_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥≤]_ _[µ]_      - 48001 _[σ]_ min [2] [(] _**[X]**_ [) + 9] _[∥]_ _**[X]**_ _[∥]_ [2][�] _∥V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ _U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[∥]_ [+ 4] _[µcσ]_ min [2] [(] _**[X]**_ [)] _[.]_


Using the assumption _µ ≤_ _c∥_ _**X**_ _∥_ _[−]_ [2] and choosing _c >_ 0 small enough, we obtain that _∥W_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥≤]_ 2 [1] [.] [Note that this]

implies that _σ_ min( _**W**_ _[⊤]_ _t_ _[∗]_ _**[W]**_ _[t]_ [+1(] _[j]_ [)][) =] �1 _−∥W_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥]_ [2] _[≥]_ [1] 2 [, which finishes the proof.]


**Lemma E.5.** _Assume that the following conditions hold_


_∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ ( _j_ ) _∥≤_ 2 _σmin_ ( _**U**_ _t ∗_ _**W**_ _t_ ( _j_ )) _,_ (E.18)

_∥_ _**U**_ _t∥≤_ 3 _∥_ _**X**_ _∥,_ (E.19)

_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥≤]_ _[c]_ [˜] (E.20)

_µ ≤_ _c∥_ _**X**_ _∥_ _[−]_ [2] _κ_ _[−]_ [2] (E.21)

_∥_ _**U**_ _t ∗_ _**W**_ _t,⊥∥≤_ _cκ_ _[−]_ [2] _∥_ _**X**_ _∥_ (E.22)

_∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥≤]_ _[cσ]_ _min_ [2] [(] _**[X]**_ [)] _[.]_ (E.23)


_Then the angle between the column space of the signal term_ _**U**_ _t ∗_ _**W**_ _t_ _and column space of_ _**X**_ _stays sufficiently small from_
_one iteration to another, namely_


            -             _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [+1] _[∥≤]_ 1 _−_ _[µ]_ 4 _[σ]_ _min_ [2] [(] _**[X]**_ [)] _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_

+ 150 _µ∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_ [+ 500] _[µ]_ [2] _[∥]_ _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ _[∥]_ [2] _[.]_


35


**Implicit Regularization for Tubal Tensors via GD**


_Proof._ To estimate the principal angle _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [+1] _[∥]_ [, we first investigate the tensor-column subspace of] _**[ U]**_ _[t]_ [+1] _[ ∗]_
_**W**_ _t_ +1. By the definition of _**U**_ _t_ +1 and _**W**_ _t ∗_ _**W**_ _[⊤]_ _t_ [+] _**[ W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _t,_ _[⊤]_ _⊥_ [=] _[ I]_ [, we have]


_**U**_ _t_ +1 _∗_ _**W**_ _t_ +1 =    - _**I**_ + _µ_ ( _A_ _[∗]_ _A_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)]    - _∗_ _**U**_ _t ∗_ _**W**_ _t_ +1

= ( _**I**_ + _µ_ _**Z**_ ) _∗_ _**U**_ _t ∗_ _**W**_ _t ∗_ _**W**_ _[⊤]_ _t_ _[∗]_ _**[W]**_ _[t]_ [+1] [+ (] _**[I]**_ [+] _[ µ]_ _**[Z]**_ [)] _[ ∗]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ _[∗]_ _**[W]**_ _[t]_ [+1] _[.]_


where we use notation _**Z**_ := ( _A_ _[∗]_ _A_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[.]_ [ This allows to represent] _[ j]_ [th slice of] _**[ U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [+1] [in the Fourier]
domain as


_U_ [(] _t_ +1 _[j]_ [)] _[W]_ [(] _t_ +1 _[j]_ [)] [= (Id +] _[ µZ]_ [(] _[j]_ [)][)] _[U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)H] _W_ [(] _t_ +1 _[j]_ [)] [+ (Id +] _[ µZ]_ [(] _[j]_ [)][)] _[U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[.]_


with _Z_ [(] _[j]_ [)] = ( _A_ _[∗]_ _A_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)][(] _[j]_ [)][.] [Because] [of] [this] [representation] [and] [decomposition] [(E.12)][,] [to] [bound] [the]
principal angle between _**U**_ _t_ +1 _∗_ _**W**_ _t_ +1 and _**X**_, we want to apply inequality (H.5) from Lemma H.4, but for this we first
need to check whether for



_U_ [H][(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)H] _W_ [(] _t_ +1 _[j]_ [)] - _−_ 1 _VU_ [H]



_P_ [(] _[j]_ [)] := _U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ _t_ [(] +1 _[j]_ [)] - _V_ [H]



_U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)]



the following applies
_∥µZ_ [(] _[j]_ [)] + _P_ [(] _[j]_ [)] + _µZ_ [(] _[j]_ [)] _P_ [(] _[j]_ [)] _∥≤_ 1 _._


For convenience, we denote _B_ [(] _[j]_ [)] := _µZ_ [(] _[j]_ [)] + _P_ [(] _[j]_ [)] + _µZ_ [(] _[j]_ [)] _P_ [(] _[j]_ [)] . Using the triangular inequality and submultiplicativity of
the norm, we bet the first simple bound on the norm of _B_ [(] _[j]_ [)]


_∥B_ [(] _[j]_ [)] _∥≤_ _µ∥Z_ [(] _[j]_ [)] _∥_ + (1 + _µ∥Z_ [(] _[j]_ [)] _∥_ ) _∥P_ [(] _[j]_ [)] _∥_ (E.24)


Note that _P_ [(] _[j]_ [)] can be rewritten as



_P_ [(] _[j]_ [)] = _U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] - _W_ [(] _t_ _[j]_ [)H] _W_ [(] _t_ +1 _[j]_ [)] - _−_ 1� _V_ [H]



_U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[,]_



_U_ [H][(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] - _−_ 1 _VU_ [H]



which allows for the following estimate of its norm

_∥P_ [(] _[j]_ [)] _∥≤∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥∥][W]_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥]_ ���� _W_ [(] _t_ _[j]_ [)H] _W_ [(] _t_ +1 _[j]_ [)]     - _−_ 1������� _VU_ [H][(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)]     - _−_ 1��� _∥VU_ H [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[∥]_

_∥U_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥∥][W]_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥]_
_≤_ _._

_σ_ min( _W_ [(] _t_ _[j]_ [)H] _W_ [(] _t_ +1 _[j]_ [)] [)] _[ ·][ σ]_ [min][(] _[U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)][)]


From here, using assumption (E.18) and a lower bound on _σ_ min( _W_ [(] _t_ _[j]_ [)H] _W_ [(] _t_ +1 _[j]_ [)] [)][ from Lemma E.4, we get]


_∥P_ [(] _[j]_ [)] _∥≤_ 4 _∥W_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥][.]_ (E.25)


Using this and the definition of _Z_ [(] _[j]_ [)], we have


_∥B_ [(] _[j]_ [)] _∥≤_ _µ∥_ ( _A_ _[∗]_ _A_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)][(] _[j]_ [)] _[∥]_ [+ 4] �1 + _µ∥_ ( _A_ _[∗]_ _A_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)][(] _[j]_ [)] _[∥]_ - _∥W_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥][.]_ (E.26)


Due to the assumption on _µ_, we can bound _µ∥_ ( _A_ _[∗]_ _A_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)][(] _[j]_ [)] _[∥]_ [as follows]


_µ∥_ ( _A_ _[∗]_ _A_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)][(] _[j]_ [)] _[∥≤]_ _[µ][∥]_ [(] _[A][∗][A]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)][(] _[j]_ [)] _[∥]_

_≤_ _µ∥_ ( _I −A_ _[∗]_ _A_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_ [+] _[ µ][∥]_ _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ _[∥]_

_≤_ _µ_ ( _cσ_ min [2] [(] _**[X]**_ [) + 10] _[∥]_ _**[X]**_ _[∥]_ [2][)] _[ ≤]_ [1]


where in the two last inequalities we use assumptions (E.23), (E.19) and (E.21) with the fact for the learning rate constant
_c >_ 0 can be chosen sufficiently small.


36


**Implicit Regularization for Tubal Tensors via GD**


This, in turn, allows us to proceed with inequality (E.26) as


_∥B_ [(] _[j]_ [)] _∥≤_ _µ∥_ ( _A_ _[∗]_ _A_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)][(] _[j]_ [)] _[∥]_ [+ 8] _[∥][W]_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥][.]_ (E.27)


Now, applying the bound on _∥W_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥≤∥]_ _**[W]**_ _t,_ _[⊤]_ _⊥_ _[∗]_ _**[W]**_ _[t]_ [+1] _[∥]_ [from] [Lemma] [E.4] [and] [similar] [transformation] [for]

_∥_ ( _A_ _[∗]_ _A_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)][(] _[j]_ [)] _[∥]_ [as above, we come the following result in (E.27)]


_∥B_ [(] _[j]_ [)] _∥≤_ _µ∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ _[∥]_ [+] _[ µ]_   - 6001 _[σ]_ [min][(] _**[X]**_ [)][2][ + 8] _[∥]_ _**[U]**_ _[t][ ∗]_ _**[W]**_ _[t][∥∥]_ _**[U]**_ _[t][ ∗]_ _**[W]**_ _[t,][⊥][∥]_   - _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_

+ 33 _µ∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_


To show that this bound above can be made smaller than one, we use assumptions (E.22), (E.23) and that _∥_ _**U**_ _t ∗_ _**W**_ _t∥≤_
_∥_ _**U**_ _∥≤_ 2 _∥_ _**X**_ _∥_, which leads to


_∥B_ [(] _[j]_ [)] _∥≤_ _µ∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ _[∥]_ [+] _[ µ]_  - 6001 _[σ]_ [min][(] _**[X]**_ [)][2][ + 8] _[c]_ _[σ]_ [min][(] _**[X]**_ [)] _·_ 3 _∥_ _**X**_ _∥_  - _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ [+ 33] _[µcσ]_ min [2] [(] _**[X]**_ [)]

_κ_ [2]

_≤_ _µ_ 10 _∥_ _**X**_ _∥_ [2] + _µc_ [1] min [(] _**[X]**_ [) + 33] _[µcσ]_ min [2] [(] _**[X]**_ [)] _[ ≤]_ [1] _[,]_

300 _[σ]_ [2]


with the last inequality following from the assumption on _µ_ . In such a way, we check the conditions of Lemma H.4 to be
able to apply inequality (H.5). This gives

_∥V_ _**X**_ [(] _[j][⊥]_ [)H] _[V]_ _U_ [(] _t_ _[j]_ +1 [)] _[W]_ [(] _t_ _[j]_ +1 [)] _[∥≤∥][V]_ _**X**_ [(] _[j][⊥]_ [)H] _[V]_ _U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[∥]_ �1 _−_ _[µ]_ 2 _[σ]_ min [2] [(] _[X]_ [(] _[j]_ [)][) +] _[ µ][∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_  
                       -                       - 2 _∥W_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_
+ _µ∥Z_ [(] _[j]_ [)] _−_ ( _X_ [(] _[j]_ [)] _X_ [(] _[j]_ [)H] _−_ _U_ [(] _t_ _[j]_ [)] _[U]_ [(] _t_ _[j]_ [)H] ) _∥_ + 1 + _µ∥Z_ [(] _[j]_ [)] _∥_

_σ_ min( _W_ [(] _t_ _[j]_ [)H] _W_ [(] _t_ +1 _[j]_ [)] [)] _[σ]_ [min][(] _[U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)][)]



2








+ 57



_∥W_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥∥][U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t,_ _[j]_ _⊥_ [)] _[∥]_
_µ∥Z_ [(] _[j]_ [)] _∥_ + (1 + _µ∥Z_ [(] _[j]_ [)] _∥_ )



_σ_ min( _W_ [(] _t_ _[j]_ [)H] _W_ [(] _t_ +1 _[j]_ [)] [)] _[σ]_ [min][(] _[U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)][)]



_._



Applying again assumption (E.18) and a lower bound on _σ_ min( _W_ [(] _t_ _[j]_ [)H] _W_ [(] _t_ +1 _[j]_ [)] [)][ from Lemma E.4 as for][ (E.25)][, in addition to]
(E.22), we get

_∥V_ _**X**_ [(] _[j][⊥]_ [)H] _[V]_ _U_ [(] _t_ _[j]_ +1 [)] _[W]_ [(] _t_ _[j]_ +1 [)] _[∥≤∥][V]_ _**X**_ [(] _[j][⊥]_ [)H] _[V]_ _U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[∥]_ �1 _−_ _[µ]_ 3 _[σ]_ min [2] [(] _[X]_ [(] _[j]_ [)][)]   - + _µ∥Z_ [(] _[j]_ [)] _−_ ( _X_ [(] _[j]_ [)] _X_ [(] _[j]_ [)H] _−_ _U_ [(] _t_ _[j]_ [)] _[U]_ [(] _t_ _[j]_ [)H] ) _∥_

+ 8�1 + _µ∥Z_ [(] _[j]_ [)] _∥_            - _∥W_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥]_ [+ 57]            - _µ∥Z_ [(] _[j]_ [)] _∥_ + 4�1 + _µ∥Z_ [(] _[j]_ [)] _∥_            - _∥W_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥]_ �2 _._


Now, making �1 + _µ∥Z_ [(] _[j]_ [)] _∥_ - _≤_ 3 by choosing _c >_ 0 small enough and using the properties of the terms involved, the above
inequality gets the following view

_∥V_ _**X**_ [(] _[j][⊥]_ [)H] _[V]_ _U_ [(] _t_ _[j]_ +1 [)] _[W]_ [(] _t_ _[j]_ +1 [)] _[∥≤∥][V]_ _**X**_ [(] _[j][⊥]_ [)H] _[V]_ _U_ [(] _t_ _[j]_ [)] _W_ [(] _t_ _[j]_ [)] _[∥]_ �1 _−_ _[µ]_ 3 _[σ]_ min [2] [(] _**[X]**_ [)]     - + _µ∥_ ( _I −A_ _[∗]_ _A_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_

+ 32 _∥W_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥]_ [+ 57]             - _µ∥Z_ [(] _[j]_ [)] _∥_ + 12 _∥W_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥]_ �2 _._ (E.28)


To proceed further with (E.28), we will first do several auxiliary estimates. We start by bounding the norm _∥W_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥]_ [.]

Since it holds that _∥W_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥≤∥]_ _**[W]**_ _t,_ _[⊤]_ _⊥_ _[∗]_ _**[W]**_ _[t]_ [+1] _[∥]_ [, from Lemma E.4, one gets]


_∥W_ _t,_ [(] _[j]_ _⊥_ [)H] _[W]_ _t_ [(] +1 _[j]_ [)] _[∥≤]_ _[µ]_  - 48001 _[σ]_ min [2] [(] _**[X]**_ [) +] _[ ∥]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t][∥∥]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥][∥]_  - _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_

+ 4 _µ∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_

_≤_ _µ_      - 48001 _[σ]_ min [2] [(] _**[X]**_ [) + 3] _[cσ]_ min [2] [(] _**[X]**_ [)]      - _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ [+ 4] _[µ][∥]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_

_≤_ 24001 _[µσ]_ min [2] [(] _**[X]**_ [)] _[∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ [+ 4] _[µ][∥]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_ (E.29)


37


**Implicit Regularization for Tubal Tensors via GD**


where we use in the second inequality that _∥_ _**U**_ _t ∗_ _**W**_ _t∥≤∥_ _**U**_ _t∥≤_ 3 _∥_ _**X**_ _∥_ and _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥∥≤_ _cκ_ _[−]_ [2] _∥_ _**X**_ _∥_ by assumption,
and in the last line that _c >_ 0 can be chosen small enough. Using this estimate, let us bound from above the squared term in
(E.28) as follows


min [(] _**[X]**_ [)]
_µ∥Z_ [(] _[j]_ [)] _∥_ + 12 _∥W_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥≤]_ _[µ][∥][Z]_ [(] _[j]_ [)] _[∥]_ [+] _[ µ]_ _[σ]_ [2] _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ [+ 48] _[µ][∥]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_

200

min [(] _**[X]**_ [)]
_≤_ _µ∥X_ [(] _[j]_ [)] _X_ [(] _[j]_ [)H] _−_ _U_ [(] _t_ _[j]_ [)] _[U]_ [(] _t_ _[j]_ [)H] _∥_ + _µ_ _[σ]_ [2] _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_

200

+ 49 _µ∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥][.]_


From here, using Jensen’s inequality, we obtain


min [(] _**[X]**_ [)]
( _µ∥Z_ [(] _[j]_ [)] _∥_ + 12 _∥W_ [(] _t,_ _[j]_ _⊥_ [)H] _[W]_ [(] _t_ +1 _[j]_ [)] _[∥]_ [)][2] _[≤]_ [3] _[µ]_ [2] _[∥][X]_ [(] _[j]_ [)] _[X]_ [(] _[j]_ [)H] _[ −]_ _[U]_ [(] _t_ _[j]_ [)] _[U]_ [(] _t_ _[j]_ [)H] _∥_ [2] + 3 _µ_ [2] _[ σ]_ [4] _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ [2]

200 [2]

+ 3 _·_ 49 [2] _µ_ [2] _∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_ [2] _[.]_


Now, we can come back to bounding (E.28) proceeding as follows




            _∥V_ _**X**_ [(] _[j][⊥]_ [)H] _[V]_ _U_ [(] _t_ _[j]_ +1 [)] _[W]_ [(] _t_ _[j]_ +1 [)] _[∥≤∥]_ _**[V]**_ _**X**_ _[⊤]_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ 1 _−_ _[µ]_ 3




[4] _[µ]_ min [(] _**[X]**_ [)] 
300 _[σ]_ [2]




_[µ]_ min [(] _**[X]**_ [) +] [4] _[µ]_

3 _[σ]_ [2] 300



+ 129 _µ∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_

min [(] _**[X]**_ [)]
+ 171 _µ_ [2] _∥X_ [(] _[j]_ [)] _X_ [(] _[j]_ [)H] _−_ _U_ [(] _t_ _[j]_ [)] _[U]_ [(] _t_ _[j]_ [)H] _∥_ [2] + _µ_ [2][ 171] _[σ]_ [4] _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ [2]
200 [2]

+ 171 _·_ 49 [2] _µ_ [2] _∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_ [2]




      _≤∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ 1 _−_ _[µ]_




[4] _[µ]_ min [(] _**[X]**_ [) +] [171]

300 _[σ]_ [2] 200




_[µ]_ min [(] _**[X]**_ [) +] [4] _[µ]_

3 _[σ]_ [2] 300



200 [171][2] _[κ][−]_ [4][�] _[c][ ·][ cµσ]_ min [2] [(] _**[X]**_ [)] 


+ 171 _µ_ [2] _∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ _[∥]_ [2]

+ _µ_ (129 + 171 _·_ 49 [2] _c_ [2] _κ_ _[−]_ [4] ) _∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥][,]_


where for the last inequality we used assumptions (E.23), (E.20) and (E.21), and the properties of the tubal tensor norm.
Now choosing constant _c >_ 0 sufficiently small, we obtain that

_∥V_ _**X**_ [(] _[j][⊥]_ [)H] _[V]_ _U_ [(] _t_ _[j]_ +1 [)] _[W]_ [(] _t_ _[j]_ +1 [)] _[∥≤]_ �1 _−_ _[µ]_ 4 _[σ]_ min [2] [(] _**[X]**_ [)]      - _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ [+ 200] _[µ]_ [2] _[∥]_ _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ _[∥]_ [2]

+ 150 _∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥][.]_


Since the right-hand side of the above inequality is independent of _j_, we obtain the lemma statement.


The following lemma shows that under a mild condition the technical assumption


_∥_ _**U**_ _t_ +1 _∥≤_ 3 _∥_ _**X**_ _∥_


needed in the lemmas above holds.

**Lemma E.6.** _Assume that ∥_ _**U**_ _t∥≤_ 3 _∥_ _**X**_ _∥, µ ≤_ 271 _[∥]_ _**[X]**_ _[∥][−]_ [2] _[ and that linear measurement operator][ A][ is such that]_


_∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥≤∥]_ _**[X]**_ _[∥]_ [2]


_Then for the iteration t_ + 1 _, it also holds ∥_ _**U**_ _t_ +1 _∥≤_ 3 _∥_ _**X**_ _∥._


_Proof._ Consider the gradient iterate


_**U**_ _t_ +1 = _**U**_ _t_ + _µA_ _[∗]_ _A_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[ ∗]_ _**[U]**_ _[t]_

= _**U**_ _t_ + _µ_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[ ∗]_ _**[U]**_ _[t]_ [+] _[ µ]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[ ∗]_ _**[U]**_ _[t]_

= ( _**I**_ _−_ _µ_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[ ∗]_ _**[U]**_ _[t]_ [+] _[ µ]_ _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[∗]_ _**[U]**_ _[t]_ [+] _[ µ]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[ ∗]_ _**[U]**_ _[t][.]_


38


**Implicit Regularization for Tubal Tensors via GD**


To estimate the norm of _**U**_ _t_ +1, we will bound each summand above separately. Due to the assumption on _µ_ and the norm of
_**U**_ _t_, we have _µ ≤_ 271 _[∥]_ _**[X]**_ _[∥][−]_ [2] _[≤]_ [1] 3 _[∥]_ _**[U]**_ _[t][∥][−]_ [2][.] [This allows us to estimate the tensor norm of][ (] _**[I]**_ _[ −]_ _[µ]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[ ∗]_ _**[U]**_ _[t]_ [via the norm]

of matrix block representation in the Fourier domain. Namely, assume that matrix _**U**_ _t_ has the SVD _**U**_ _t_ = _V_ Σ _W_ [H] . Then for
matrix ( _**I**_ _−_ _µ_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[ ∗]_ _**[U]**_ _[t]_ [, we have]

( _**I**_ _−_ _µ_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[ ∗]_ _**[U]**_ _[t]_ [=] _[ V]_ [ Σ] _[W]_ [ H] _[ −]_ _[µV]_ [ Σ] _[W]_ [ H] _[W]_ [Σ] _[V]_ [H] _[V]_ [ Σ] _[W]_ [ H] [=] _[ V]_ [ Σ] _[W]_ [ H] _[ −]_ _[µV]_ [ Σ][3] _[W]_ [ H] [=] _[ V]_ [ (Σ] _[ −]_ _[µ]_ [Σ][3][)] _[W]_ [ H] _[.]_

From here, since _µ ≤_ 271 _[∥]_ _**[X]**_ _[∥][−]_ [2] _[≤]_ [1] 3 _[∥]_ _**[U]**_ _[∥][−]_ [2] and _∥_ _**U**_ _t∥_ = _∥_ _**U**_ _t∥_, it holds that

_∥_ ( _**I**_ _−_ _µ_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[ ∗]_ _**[U]**_ _[t][∥]_ [=] _[ ∥]_ _**[U]**_ _[t][∥−]_ _[µ][∥]_ _**[U]**_ _[t][∥]_ [3] [=] _[ ∥]_ _**[U]**_ _[t][∥]_ [(1] _[ −]_ _[µ][∥]_ _**[U]**_ _[t][∥]_ [2][)][.] Besides, from the submultiplicativity of the
tensor norm and the triangle inequality, we obtain that


_∥_ _**U**_ _t_ +1 _∥≤_ (1 _−_ _µ∥_ _**U**_ _t∥_ [2] + _µ∥_ _**X**_ _∥_ [2] + _µ∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_ [)] _[∥]_ _**[U]**_ _[t][∥]_ (E.30)

_≤_ (1 _−_ _µ∥_ _**U**_ _t∥_ [2] + 2 _µ∥_ _**X**_ _∥_ [2] ) _∥_ _**U**_ _t∥,_ (E.31)


where in the last line we used the assumption on _∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_ [.] [By combining inequality][ (E.31)][ with]
the assumption _µ ≤_ 27 _∥_ 1 _**X**_ _∥_ [2] _[≤]_ 3 _∥_ _**U**_ 1 _∥_ [2] [, we obtain that] _[ ∥]_ _**[U]**_ _[t]_ [+1] _[∥≤]_ [3] _[∥]_ _**[X]**_ _[∥]_ [, which finishes the proof.]


The following lemma shows that _**U**_ _t ∗_ _**W**_ _t ∗_ _**W**_ _[⊤]_ _t_ _[∗]_ _**[U]**_ _t_ _[⊤]_ [converges towards] _**[ X]**_ _[∗]_ _**[X]**_ _[ T]_ [, when projected onto the tensor column]
space of _**X**_ .


**Lemma E.7.** _Assume that the following conditions hold_


_∥_ _**U**_ _t∥≤_ 3 _∥_ _**X**_ _∥_ (E.32)

1
_µ ≤_ _c ·_ ~~_√_~~ _· κ_ _[−]_ [2] _∥_ _**X**_ _∥_ _[−]_ [2] (E.33)

_nk_



_and_



1
_σmin_ ( _**U**_ _t ∗_ _**W**_ _t_ ) _≥_ ~~_√_~~ (E.34)

10 _[σ][min]_ [(] _**[X]**_ [)]

_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥≤]_ _[cκ][−]_ [2] (E.35)


max ��� _**V**_ _⊤_ _**X**_ _[∗]_ [(] _[A][∗][A −I]_ [)(] _**[Y]**_ _[t]_ [)] �� _F_ _[,]_ �� _**V**_ _⊤_ _**U**_ _t∗_ _**W**_ _t_ _[∗]_ [(] _[A][∗][A −I]_ [)(] _**[Y]**_ _[t]_ [)] �� _F_ _[,]_ ��( _A∗A −I_ )( _**Y**_ _t_ )��� _≤_ _κ_ _[−]_ [2] _∥_ _**Y**_ _t∥F_



_with_ _**Y**_ _t_ := _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ _[.]_ _[Then it holds that]_

_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _t_ _[⊤][∥][F]_ _[≤]_ [3] _[∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_ [+] _[ ∥]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ _[∥][F]_ (E.36)


_as well as_


_∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ _[∥][F]_ _[≤]_ [4] _[∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_ [+] _[ ∥]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _t_ _[⊤][∥][F]_ (E.37)


_and_


                  -                   _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ [+1] _[ ∗]_ _**[U]**_ _[⊤]_ _t_ +1 [)] _[∥][F]_ _[≤]_ 1 _−_ _[µ]_ _min_ [(] _**[X]**_ [)] _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_

200 _[σ]_ [2]

_min_ [(] _**[X]**_ [)]
+ _µ_ _[σ]_ [2] _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ _[∥][F]_ (E.38)

100


_Proof._ We start by proving the first inequality (E.38). For this, let us decompose _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _t_ _[⊤]_ [as follows]

_**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _t_ _[⊤]_ [=] _**[ V]**_ _**X**_ _[⊤]_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _t_ _[⊤]_ _[∗]_ _**[V]**_ _**[X]**_ _[∗]_ _**[V]**_ _[⊤]_ _**X**_ [+] _**[ V]**_ _**X**_ _[⊤]_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _t_ _[⊤]_ _[∗]_ _**[V]**_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[,]_


then using the triangle inequality and submultiplicativity of the Frobenius and the spectral norm, we obtain


_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _t_ _[⊤][∥][F]_ _[≤∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _t_ _[⊤]_ _[∗]_ _**[V]**_ _**[X]**_ _[∥][F]_ [+] _[ ∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _t_ _[⊤]_ _[∗]_ _**[V]**_ _**X**_ _[⊥]_ _[∥][F]_

_≤∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[ ∗]_ _**[V]**_ _**[X]**_ _[∥][F]_ [+] _[ ∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _t_ _[⊤]_ _[∗]_ _**[V]**_ _**X**_ _[⊥]_ _[∥][F]_

_≤∥_ _**V**_ _[⊤]_ _**X**_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_ [+] _[ ∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _t_ _[⊤]_ _[∗]_ _**[V]**_ _**X**_ _[⊥]_ _[∥][F]_ _[,]_ (E.39)


39


**Implicit Regularization for Tubal Tensors via GD**


where in the second line, we used the orthogonality of the decomposition. Now, we will work additionally on bounding the
norm of _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _t_ _[⊤]_ _[∗]_ _**[V]**_ _**X**_ _[⊥]_ [to obtain][ (E.38)][.] [Here, we will use the orthogonal decomposition with respect to] _**[ W]**_ _[t]_ [and]
_**W**_ _t,⊥_, which leads to


_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _t_ _[⊤]_ _[∗]_ _**[V]**_ _**X**_ _[⊥]_ _[∥][F]_ _[≤∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[W]**_ _[t][ ∗]_ _**[W]**_ _t_ _[⊤]_ _[∗]_ _**[U]**_ _t_ _[⊤]_ _[∗]_ _**[V]**_ _**X**_ _[⊥]_ _[∥][F]_ [+] _[ ∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _t,_ _[⊤]_ _⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ _[∗]_ _**[V]**_ _**X**_ _[⊥]_ _[∥][F]_

_≤∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[W]**_ _[t][ ∗]_ _**[W]**_ _t_ _[⊤]_ _[∗]_ _**[U]**_ _t_ _[⊤]_ _[∗]_ _**[V]**_ _**X**_ _[⊥]_ _[∥][F]_ [+] _[ ∥]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ _[∥][F]_


Now, for the first term above, we get


_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[W]**_ _[t][ ∗]_ _**[W]**_ _t_ _[⊤]_ _[∗]_ _**[U]**_ _t_ _[⊤]_ _[∗]_ _**[V]**_ _**X**_ _[⊥]_ _[∥][F]_ [=] _[ ∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t]_ _[∗]_ _**[V]**_ _**U**_ _[⊤]_ _t∗_ _**W**_ _t_ _[∗]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t]_ _[∗]_ _**[W]**_ _[⊤]_ _t_ _[∗]_ _**[U]**_ _t_ _[⊤]_ _[∗]_ _**[V]**_ _**X**_ _[⊥]_ _[∥][F]_



=


=


=



_k_

- _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t]_ _[∗]_ _**[V]**_ _**U**_ _[⊤]_ _t∗_ _**W**_ _t_ _[∗]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t]_ _[∗]_ _**[W]**_ _[⊤]_ _t_ _[∗]_ _**[U]**_ _t_ _[⊤]_ _[∗]_ _**[V]**_ _**X**_ _[⊥]_ [(] _[j]_ [)] _[∥][F]_

_j_ =1


_k_

- _∥V_ _**X**_ [(] _[j]_ [)H] _[⊥]_ _[V]_ [(] _**U**_ _[j]_ _t_ [)] _∗_ _**W**_ _t_ _[V]_ [(] _**U**_ _[j]_ _t_ [)H] _∗_ _**W**_ _t_ _[U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)H] _U_ [(] _t_ _[j]_ [)H] _V_ [(] _**X**_ _[j]_ [)] _[⊥]_ _[∥][F]_

_j_ =1



_k_

- _∥V_ _**X**_ [(] _[j]_ [)H] _[⊥]_ _[V]_ [(] _**U**_ _[j]_ _t_ [)] _∗_ _**W**_ _t_

_j_ =1




- _V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ [(] _**U**_ _[j]_ _t_ [)] _∗_ _**W**_ _t_




- _−_ 1 _V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ [(] _**U**_ _[j]_ _t_ [)] _∗_ _**W**_ _t_ _[V]_ [(] _**U**_ _[j]_ _t_ [)H] _∗_ _**W**_ _t_ _[U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)H] _U_ [(] _t_ _[j]_ [)H] _V_ [(] _**X**_ _[j]_ [)] _[⊥]_ _[∥][F]_




- _−_ 1 [�] ���



_k_

- _∥V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ [(] _**U**_ _[j]_ _t_ [)] _∗_ _**W**_ _t_ _[V]_ [(] _**U**_ _[j]_ _t_ [)H] _∗_ _**W**_ _t_ _[U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)H] _U_ [(] _t_ _[j]_ [)H] _V_ [(] _**X**_ _[j]_ [)] _[⊥]_ _[∥][F]_

_j_ =1



_≤_ 1max _≤j≤k_ _[∥][V]_ _**X**_ [(] _[j]_ [)H] _[⊥]_ _[V]_ [(] _**U**_ _[j]_ _t_ [)] _∗_ _**W**_ _t_ _[∥]_ 1 [max] _≤j≤k_




 - _V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ [(] _**U**_ _[j]_ _t_ [)] _∗_ _**W**_ _t_
����



= _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_

_σ_ min( _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t]_ [)]



= _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_

_σ_ min( _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t]_ [)]



_k_

- _∥V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[V]_ [(] _**U**_ _[j]_ _t_ [)] _∗_ _**W**_ _t_ _[V]_ [(] _**U**_ _[j]_ _t_ [)H] _∗_ _**W**_ _t_ _[U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)H] _U_ [(] _t_ _[j]_ [)H] _V_ [(] _**X**_ _[j]_ [)] _[⊥]_ _[∥][F]_

_j_ =1


_k_

- _∥V_ [(] _**X**_ _[j]_ [)H] _[⊥]_ _[U]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)] _[W]_ [(] _t_ _[j]_ [)H] _U_ [(] _t_ _[j]_ [)H] _V_ [(] _**X**_ _[j]_ [)] _[⊥]_ _[∥][F]_

_j_ =1



= _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ _∥_ _**V**_ _**X**_ _⊥_ _∗_ _**U**_ _t ∗_ _**W**_ _t ∗_ _**W**_ _[⊤]_ _t_ _[∗]_ _**[U]**_ _t_ _[⊤]_ _[∗]_ _**[V]**_ _**X**_ _[⊥]_ _[∥][F]_

_σ_ min( _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t]_ [)]



= _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ _∥_ _**V**_ _**X**_ _⊥_ _∗_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ _[∗]_ _**[V]**_ _**X**_ _[⊥]_ _[∥][F]_

_σ_ min( _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t]_ [)]



= _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ _∥_ _**V**_ _**X**_ _⊥_ _∗_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[ ∗]_ _**[V]**_ _**X**_ _[⊥]_ _[∥][F]_

_σ_ min( _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t]_ [)]



_≤_ _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ _∥_ _**V**_ _**X**_ _⊥_ _∗_ ( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥][F]_ _[≤]_ [2] _[∥]_ _**[V]**_ _**X**_ _[⊥]_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_

_σ_ min( _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t]_ [)]



where in the last line we used the assumption (E.35). Them, using just established bound together with (E.39), we get


_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _t_ _[⊤][∥][F]_ _[≤]_ [3] _[∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_ [+] _[ ∥]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ _[∥][F]_ _[.]_


To get inequality (E.37), we use the orthogonal decomposition of _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [with respect to] _**[ V]**_ _**[X]**_ [and] _**[ V]**_ _**X**_ _[⊥]_ [, which]
leads to


_∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ _[∥][F]_ [=] _[ ∥]_ _**[V]**_ _[⊤]_ _**X**_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_ [+] _[ ∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_

= _∥_ _**V**_ _[⊤]_ _**X**_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_ [+] _[ ∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _t_ _[⊤][∥][F]_

_≤_ 4 _∥_ _**V**_ _[⊤]_ _**X**_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_ [+] _[ ∥]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ _[∥][F]_ _[.]_


Inequality (E.38) follows from the two inequalities proved here and Lemma 9.5 in (Stoger & Soltanolkotabi, 2021).¨ The
building stones for this are the properties of the tubal tensor Frobenius norm. Namely, the Frobenius norm of any tubal


40


**Implicit Regularization for Tubal Tensors via GD**


tensor _**T**_ can be represented as the sum of Frobenius norms of each slice in the domain, that is



_∥_ _**T**_ _∥F_ =



_k_

- _∥T_ [(] _[j]_ [)] _∥F_


_j_ =1



_√_
and _∥_ _**T**_ _∥F_ _≤_



_n · k∥_ _**T**_ _∥._ Besides, the Frobenius norm of the product of two tensors _**T**_ and _**P**_ can be bounded as below



_k_

- _∥P_ [(] _[j]_ [)] _∥F_ _≤∥_ _**T**_ _∥∥_ _**P**_ _∥F ._


_j_ =1



_∥_ _**T**_ _∗_ _**P**_ _∥F_ =



_k_

- _∥T_ [(] _[j]_ [)] _P_ [(] _[j]_ [)] _∥F_ _≤_ max [(] _[j]_ [)] _[∥]_

1 _≤j≤k_ _[∥][T]_
_j_ =1



Now, we have collected all the necessary ingredients to prove the main result of this section, which shows that after a
sufficient number of interactions, the relative error between _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [and] _**[ X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ [becomes small.]



_√_
**Theorem E.1.** _Suppose that the stepsize satisfies µ ≤_ _c_ 1



_kκ_ _[−]_ [4] _∥_ _**X**_ _∥_ _[−]_ [2] _for some small c_ 1 _>_ 0 _, and A_ : _S_ _[n][×][n][×][k]_ _→_ R _[m]_



_satisfies_ _RIP_ (2 _r_ + 1 _, δ_ ) _for_ _some_ _constant_ 0 _<_ _δ_ _≤_ _κ_ [4] _c_ ~~_[√]_~~ 1 _r_ _[.]_ _[Set]_ _[γ]_ _[∈]_ [(0] _[,]_ [1] 2



2 [)] _[,]_ _[and]_ _[choose]_ _[a]_ _[number]_ _[of]_ _[iterations]_ _[t][∗]_



_c_ 2 _σmin_ ( _**X**_ )
_such_ _that_ _σmin_ ( _**U**_ _t∗_ _∗_ _**W**_ _t∗_ ) _≥_ _γ._ _Also,_ _assume_ _that_ _∥_ _**U**_ _t∗_ _∗_ _**W**_ _t∗,⊥∥≤_ 2 _γ,_ _∥_ _**U**_ _t∗_ _∥≤_ 3 _∥_ _**X**_ _∥,_ _γ_ _≤_ _κ_ [2] min _{n, R}_ _[,]_ _[and]_

_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t]_ _∗_ _[∗]_ _**[W]**_ _[t]_ _∗_ _[∥≤]_ _[c]_ [2] _[κ][−]_ [2] _[ for some small][ c]_ [2] _[>]_ [ 0] _[.]_ _[Then, after]_



1                  -                  - _κr_

           - _t −_ _t∗_ ≲ _µσmin_ ( _**X**_ ) [2] [ln] min 1 _,_ _k_ (min _{n, R} −_ _r_ )


_additional iterations, we have_




- _∥_ _**X**_ _∥_

_γ_







_∥_ _**U**_        - _t_ _∗_ _**U**_        - _[⊤]_ _t_ _[−]_ _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤][∥][F]_ ≲ _k_ [5] _[/]_ [4] _r_ [1] _[/]_ [8] _κ_ _[−]_ [3] _[/]_ [16] (min _{n, R} −_ _r_ ) [3] _[/]_ [8] _γ_ [21] _[/]_ [16] _∥_ _**X**_ _∥_ _[−]_ [21] _[/]_ [16] _._
_∥_ _**X**_ _∥_ [2]


_Proof._ First, we set

_t_ 1 = min                   - _t ≥_ _t∗_ : _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [)] _[ ≥]_ ~~_√_~~ 110 _[σ]_ [min][(] _**[X]**_ [)]                   - _,_


and then aim to prove that over the iterations _t∗_ _≤_ _t ≤_ _t_ 1, the following hold:




- _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [)] _[ ≥]_ 2 [1]




[1]

8 _[µσ]_ [min][(] _**[X]**_ [)][2][�] _[t][−][t][∗]_



2 [1] _[γ]_ �1 + [1] 8




        - _√_

- _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥∥≤_ 2 _γ_ 1 + 80 _µc_ 2


- _∥_ _**U**_ _t∥≤_ 3 _∥_ _**X**_ _∥_


- _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥≤]_ _[c]_ [2] _[κ][−]_ [2][.]



_kσ_ min( _**X**_ ) [2][�] _[t][−][t][∗]_



Intuitively, this means that over the range _t∗_ _≤_ _t ≤_ _t_ 1, the smallest singular value of the signal term _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [grows at a]
faster rate than the largest singular value of the noise term _**U**_ _t ∗_ _**W**_ _t,⊥_ .


For _t_ = _t∗_, these inequalities hold due to the assumptions of this theorem. Now, suppose they hold for some _t_ between _t∗_
and _t_ 1. We’ll show they also hold for _t_ + 1.


41


**Implicit Regularization for Tubal Tensors via GD**


First, note that we have:


_∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_

= _∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**W**_ _t ∗_ _**W**_ _[⊤]_ _t_ _[∗]_ _**[U]**_ _t_ _[⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_



_≤∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**W**_ _t ∗_ _**W**_ _[⊤]_ _t_ _[∗]_ _**[U]**_ _t_ _[⊤]_ [)] _[∥]_ [+] _[ ∥]_ [(] _[A][∗][A −I]_ [)(] _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_



_√_
( _a_ ) _≤δ_



_√_
_≤δ_

_√_
= _δ_

_√_
_≤δ_



_√_
_kr∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**W**_ _t ∗_ _**W**_ _[⊤]_ _t_ _[∗]_ _**[U]**_ _t_ _[⊤][∥]_ [+] _[ δ]_



_kr_ - _∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _∥_ + _∥_ _**U**_ _t ∗_ _**W**_ _t ∗_ _**W**_ _[⊤]_ _t_ _[∗]_ _**[U]**_ _t_ _[⊤][∥]_ - + _δ√_



_√_
_kr_ - _∥_ _**X**_ _∥_ [2] + _∥_ _**U**_ _t∥_ [2][�] + _δ_

_√_
_kr_ - _∥_ _**X**_ _∥_ [2] + 9 _∥_ _**X**_ _∥_ [2][�] + _δ_



_k∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _t,⊥_ _∗_ _**U**_ _[⊤]_ _t_ _[∥][∗]_



_k∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _t,⊥_ _∗_ _**U**_ _[⊤]_ _t_ _[∥][∗]_



_k∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _t,⊥_ _∗_ _**U**_ _[⊤]_ _t_ _[∥][∗]_



_√_
_kr_ - _∥_ _**X**_ _∥_ [2] + _∥_ _**U**_ _t ∗_ _**W**_ _t∥_ [2][�] + _δ_



_k∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _t,⊥_ _∗_ _**U**_ _[⊤]_ _t_ _[∥][∗]_



_√_
( _b_ ) _≤δ_



_√_
_≤_ 10 _δ_

_√_
_≤_ 10 _δ_



_√_
_kr∥_ _**X**_ _∥_ [2] + _δ_



_√_
_krκ_ [2] _σ_ min( _**X**_ ) [2] + _δ_



_k_ (min _{n, R} −_ _r_ ) _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _t,⊥_ _∗_ _**U**_ _[⊤]_ _t_ _[∥]_



_k_ (min _{n, R} −_ _r_ ) _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥∥_ [2]



_√_
( _c_ ) _≤_ 10 _c_ 1

_√_
( _d_ ) _≤_ 10 _c_ 1

_√_
( _e_ ) _≤_ 40 _c_ 1



_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2] _._



_k_ (min _{n, R} −_ _r_ ) _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥∥_ [2]



_√_
_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2] + 4 _δ_

_√_
_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2] + 8 _δ_



_k_ (min _{n, R} −_ _r_ ) _γ_ [2][ �] 1 + 80 _µc_ 2 _σ_ min( _**X**_ ) [2][�][2(] _[t][−][t][∗]_ [)]

_k_ (min _{n, R} −_ _r_ ) _γ_ [7] _[/]_ [4] _σ_ min( _**X**_ ) [1] _[/]_ [4]



In inequality (a), we used the fact that _A_ satisfi _√_ es RIP(2 _r_ + 1 _, δ_ ) (a _√_ nd hence, RIP( _r_ + 1 _, δ_ ) and RIP(2 _, δ_ )), and thus, by
Lemmas G.2 and G.3, also satisfies S2SRIP( _r, δ_ _kr_ ) and S2NRIP( _δ_ _k_ ). Inequality (b) uses the assumption _∥_ _**U**_ _t∥≤_ 3 _∥_ _**X**_ _∥_

and the fact that _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [has tubal rank at most][ min] _[{][n, R][} −]_ _[r]_ [.] [In inequality (c), we used the assumption]

_c_ 1
_δ_ _≤_ [with] [the] [second] [bulleted] [inequality] [assumed] [by] [the] [inductive] [step.] [Inequality] [(d)] [holds] [due] [to] [the]
_κ_ [4] ~~_[√]_~~ _r_ [along]

definitions of _t_ 1 and _t∗_ and the fact that _t∗_ _≤_ _t ≤_ _t_ 1. Finally, inequality (e) holds due to the assumption _γ_ _≤_ _κ_ [2] _c_ 2min _σ_ min _{_ ( _n,R_ _**X**_ ) _}_ [.]


If _c_ 1 is chosen small enough, the above bound is less than _∥_ _**X**_ _∥_ . Then, along with our other assumptions, we can use
Lemma E.6 to obtain _∥_ _**U**_ _t_ +1 _∥≤_ 3 _∥_ _**X**_ _∥_ .

Next, we can use Lemma E.1 along with the bound _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [)] _[ ≤]_ ~~_√_~~ 110 _[σ]_ [min][(] _**[X]**_ [)][ to obtain]


_σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1][)] _[ ≥]_ _[σ]_ [min][(] _**[V]**_ _**X**_ _[⊤]_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [+1][)]

                -                 _≥_ _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [)] 1 + [1] _**X**_ _[∗]_ _**[U]**_ _[t]_ [)][2]

4 _[µσ]_ [min][(] _**[X]**_ [)][2] _[ −]_ _[µσ]_ [min][(] _**[V]**_ _[⊤]_




     _≥_ _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [)] 1 + [1]




[1] 
10 _[µσ]_ [min][(] _**[X]**_ [)][2]




[1]

4 _[µσ]_ [min][(] _**[X]**_ [)][2] _[ −]_ [1]




     -      _≥_ _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [)] 1 + [1]

8 _[µσ]_ [min][(] _**[X]**_ [)][2]




[1] �1 + [1]

2 _[γ]_ 8

[1] �1 + [1]

2 _[γ]_ 8




[1] 
8 _[µσ]_ [min][(] _**[X]**_ [)][2]



_≥_ [1]




[1] - _t−t∗_ _·_ �1 + [1]

8 _[µσ]_ [min][(] _**[X]**_ [)][2] 8



= [1]




[1] - _t−t∗_ +1

8 _[µσ]_ [min][(] _**[X]**_ [)][2]



Since _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [+1][)] [=] _[σ]_ [min][(] _**[V]**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1][)][,] [which] [is] [positive] [by] [the] [above] [bound,] [all] [the] [singular] [tubes] [of]
_**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [+1] [are invertible.] [Hence, we can apply Lemma E.3 to obtain]


42


**Implicit Regularization for Tubal Tensors via GD**


_∥_ _**U**_ _t_ +1 _∗_ _**W**_ _t_ +1 _,⊥_ ( _j_ ) _∥≤_ �1 _−_ _[µ]_ 2 _[∥]_ _**[U]**_ _[t][ ∗]_ _**[W]**_ _[t,][⊥]_ ( _j_ ) _∥_ 2 + 9 _µ∥_ _**V**_ _⊤_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t]_ ( _j_ ) _∥∥_ _**X**_ _∥_ 2

+ 2 _µ∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_                - _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ ( _j_ ) _∥_




_[µ]_ 1 + 80 _µc_ 2 _√_

2 _[·]_ [ 4] _[γ]_ [2][ �]




 _≤_ 1 _−_ _[µ]_



_kσ_ min( _**X**_ ) [2][�][2(] _[t][−][t][∗]_ [)] + 9 _µc_ 2 _κ_ _[−]_ [2] _∥_ _**X**_ _∥_ [2]



_√_
+ 2 _µ ·_ 40 _c_ 1



_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2][�] _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ ( _j_ ) _∥_




_[µ]_ 1 + 80 _µc_ 2 _√_

2 _[·]_ [ 4] _[γ]_ [2][ �]




 _≤_ 1 _−_ _[µ]_



_kσ_ min( _**X**_ ) [2][�][2(] _[t][−][t][∗]_ [)] + 9 _µc_ 2 _σ_ min( _**X**_ ) [2]



_√_
+ 80 _c_ 1 _µ_



_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2][�] _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ ( _j_ ) _∥_




 - _√_
_≤_ 1 + 80 _c_ 1 _µ_

 - _√_
_≤_ 1 + 80 _c_ 1 _µ_



_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2][�] _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ ( _j_ ) _∥_

_kσ_ min( _**X**_ ) [2][�] _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ ( _j_ ) _∥_




  - _√_
_≤_ 2 _γ_ 1 + 80 _c_ 1 _µ_



_kσ_ min( _**X**_ ) [2][�] _[t][−][t][∗]_ [+1] _,_



where we have used the inductive assumption that the inequalities hold for _t_ along with the fact that _κ_ = _∥_ _**X**_ _∥/σ_ min( _**X**_ ) _≥_ 1.


Next, we will bound the term using Lemma E.5


_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [+1] _[∥]_




 _≤_ 1 _−_ _[µ]_

4

 _≤_ 1 _−_ _[µ]_

4

 _≤_ 1 _−_ _[µ]_

4

 = 1 _−_ _[µ]_

4

 _≤_ 1 _−_ _[µ]_

4

 = 1 _−_ _[µ]_

4

 = 1 _−_ _[µ]_

4

 = 1 _−_ _[µ]_



_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2]




    min [(] _**[X]**_ [)] _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥]_ [+ 150] _[µ][∥]_ [(] _[A][∗][A −I]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥]_ [+ 500] _[µ]_ [2] _[∥]_ _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ _[∥]_ [2]
4 _[σ]_ [2]




_[µ]_ min [(] _**[X]**_ [)] - _c_ 2 _κ_ _[−]_ [2] + 150 _µ ·_ 40 _c_ 1 _√_

4 _[σ]_ [2]



_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2] + 500 _µ_ [2] _·_ ( _∥_ _**X**_ _∥_ [2] + _∥_ _**U**_ _t∥_ [2] )




    - _√_

_[µ]_ min [(] _**[X]**_ [)] _c_ 2 _κ_ _[−]_ [2] + 6000 _µc_ 1

4 _[σ]_ [2]

_[µ]_ min [(] _**[X]**_ [)] - _c_ 2 _κ_ _[−]_ [2] + 6000 _µc_ 1 _√_

4 _[σ]_ [2]

_[µ]_ min [(] _**[X]**_ [)] - _c_ 2 _κ_ _[−]_ [2] + 6000 _µc_ 1 _√_

4 _[σ]_ [2]

_[µ]_ min [(] _**[X]**_ [)] - _c_ 2 _κ_ _[−]_ [2] + 6000 _µc_ 1 _√_

4 _[σ]_ [2]

_[µ]_ min [(] _**[X]**_ [)] - _c_ 2 _κ_ _[−]_ [2] + 6000 _µc_ 1 _√_

4 _[σ]_ [2]



_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2] + 500 _µ_ [2] _·_ ( _∥_ _**X**_ _∥_ [2] + 9 _∥_ _**X**_ _∥_ [2] ) [2]


_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2] + 50000 _µ_ [2] _∥_ _**X**_ _∥_ [4]


_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2] + 50000 _µ · c_ 1 _κ_ _[−]_ [4] _∥_ _**X**_ _∥_ _[−]_ [2] _· ∥_ _**X**_ _∥_ [4]


_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2] + 50000 _µ · c_ 1 _κ_ _[−]_ [4] _∥_ _**X**_ _∥_ [2]


_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2] + 50000 _µ · c_ 1 _κ_ _[−]_ [4] _κ_ [2] _σ_ min( _**X**_ ) [2]




_[µ]_ min [(] _**[X]**_ [)] - _c_ 2 _κ_ _[−]_ [2] + 56000 _µc_ 1 _√_

4 _[σ]_ [2]



Here, we have again used the inductive assumptions along with the fact that _κ_ = _∥_ _**X**_ _∥/σ_ min( _**X**_ ). If we choose _c_ 1 sufficiently
small, we will have _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [+1] _[∥≤]_ _[c]_ [2] _[κ][−]_ [2][.]


Therefore, the four bullet points hold for _t_ + 1, and thus, the induction is complete.


With the above bullet points in mind, we note that



1
~~_√_~~



_**X**_ _[∗]_ _**[U]**_ _[t]_ 1 [)] _[ ≥]_ [1]
10 _[σ]_ [min][(] _**[X]**_ [)] _[ ≥]_ _[σ]_ [min][(] _**[V]**_ _[⊤]_ 2




[1] �1 + [1]

2 _[γ]_ 8




[1] - _t_ 1 _−t∗_ _,_

8 _[µσ]_ [min][(] _**[X]**_ [)][2]



and so,




   10 _[σ]_ [min][(] _**[X]**_ [)]




   _,_
10 _[σ]_ [min][(] _**[X]**_ [)]




    - 2
log ~~_√_~~
_γ_
_t_ 1 _−_ _t∗_ _≤_



10 16 - 2

_≤_ ~~_√_~~

[1] - _µσ_ min( _**X**_ ) [2] [log] _γ_

8 _[µσ]_ [min][(] _**[X]**_ [)][2]




 log 1 + [1]



43


**Implicit Regularization for Tubal Tensors via GD**


where we have used the inequality log(1+1 _x_ ) _[≤]_ _x_ [2] [for][ 0] _[ < x <]_ [ 1][.] [Furthermore, we can bound the norm of the signal term at]

iteration _t_ 1 by




        - _√_
_∥_ _**U**_ _t_ 1 _∗_ _**W**_ _t_ 1 _,⊥∥≤_ 2 _γ_ 1 + 80 _µc_ 2



_kσ_ min( _**X**_ ) [2][�] _[t]_ [1] _[−][t][∗]_




  - 2
_≤_ 2 _γ_ ~~_√_~~



_γ_



2 _[σ]_ [min][(] _**[X]**_ [)]

10 _[·]_ _γ_

2 _[σ]_ [min][(] _**[X]**_ [)]

10 _[·]_ _γ_



�1280 _c_ 2


�1 _/_ 64




  - 2
_≤_ 2 _γ_ ~~_√_~~



_γ_



_≤_ 3 _γ_ [63] _[/]_ [64] _σ_ min( _**X**_ ) [1] _[/]_ [64]

_≤_ 3 _γ_ [7] _[/]_ [8] _σ_ min( _**X**_ ) [1] _[/]_ [8] _,_


where we have used the previous bound on _t_ 1 _−_ _t∗_, the fact that _c_ 2 _>_ 0 can be chosen to be sufficiently small, and the fact
that _σ_ min( _**X**_ ) _≥_ _γ_ .


Next, we set




    - 300    - 5    - _r_ _∥_ _**X**_ _∥_ [7] _[/]_ [4]
_t_ 2 = _t_ 1 + _µσ_ min( _**X**_ ) [2] [ln] 18 _[κ]_ [1] _[/]_ [4] _k_ (min _{n, R} −_ _r_ ) _γ_ [7] _[/]_ [4]



��



_t_ 3 = min    - _t ≥_ _t_ 1 : �� _k_ (min _{n, R} −_ _r_ ) + 1���� _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _⊤t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ ��� _F_ _[≥∥]_ _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ _[∥][F]_    

                   - _t_ = min _{t_ 2 _, t_ 3 _}._


We now aim to show that over the range _t_ 1 _≤_ _t ≤_ [�] _t_, the following inequalities hold:


1

  - _σ_ min( _**U**_ _t ∗_ _**W**_ _t_ ) _≥_ _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [)] _[ ≥]_ ~~_√_~~

10 _[σ]_ [min][(] _**[X]**_ [)]




       - _√_

- _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥∥≤_ 1 + 80 _µc_ 2


- _∥_ _**U**_ _t∥≤_ 3 _∥_ _**X**_ _∥_


- _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥≤]_ _[c]_ [2] _[κ][−]_ [2]



_kσ_ min( _**X**_ ) [2][�] _[t][−][t]_ [1] _∥_ _**U**_ _t_ 1 _∗_ _**W**_ _t_ 1 _,⊥∥_



_√_

- _∥_ _**V**_ _[⊤]_ _**X**_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_ _[≤]_ [10]



_kr_ �1 _−_ 4001 _[µσ]_ [min][(] _**[X]**_ [)][2][�] _[t][−][t]_ [1] _[ ∥]_ _**[X]**_ _[∥]_ [2]



For _t_ = _t_ 1, the first four bullet points follow from what we previously proved via induction. The last one holds since we
trivially have



_√_
_∥_ _**V**_ _[⊤]_ _**X**_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ 1 _[∗]_ _**[U]**_ _t_ _[⊤]_ 1 [)] _[∥][F]_ _[≤]_



_√_
_≤_

_√_
_≤_

_√_
_≤_



_kr∥_ _**V**_ _[⊤]_ _**X**_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ 1 _[∗]_ _**[U]**_ _t_ _[⊤]_ 1 [)] _[∥]_



_√_
_kr∥_ _**X**_ _∥_ [2] +



_kr∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t_ 1 _∗_ _**U**_ _[⊤]_ _t_ 1 _[∥]_



_√_
_kr∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _∥_ +



_kr∥_ _**U**_ _t_ 1 _∗_ _**U**_ _[⊤]_ _t_ 1 _[∥]_



_kr∥_ _**U**_ _t_ 1 _∥_ [2]



_√_
_≤_ 10



_kr∥_ _**X**_ _∥_ [2] _._



Now suppose all the bullet points hold for some integer _t ∈_ [ _t_ 1 _,_ [�] _t −_ 1]. Again, we aim to s _√_ how they all ho _√_ ld for _t_ + 1. In a
similar manner as done before, we can bound _∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥≤]_ [10] _[δ]_ _kr∥_ _**X**_ _∥_ [2] + _δ_ _k_ (min _{n, R} −_



_√_
_kr∥_ _**X**_ _∥_ [2] + _δ_



similar manner as done before, we can bound _∥_ ( _A_ _A −I_ )( _**X**_ _∗_ _**X**_ _−_ _**U**_ _t ∗_ _**U**_ _t_ [)] _[∥≤]_ [10] _[δ]_ _kr∥_ _**X**_ _∥_ [2] + _δ_ _k_ (min _{n, R} −_

_r_ ) _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥∥_ [2], and then continue as follows



44


**Implicit Regularization for Tubal Tensors via GD**



_∥_ ( _A_ _[∗]_ _A −I_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ [)] _[∥]_



_√_
_≤_ 10 _δ_



_√_
_kr∥_ _**X**_ _∥_ [2] + _δ_



_k_ (min _{n, R} −_ _r_ ) _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥∥_ [2]



_≤_ 10 _·_ _c_ 1 _√kr · κ_ [2] _σ_ min( _**X**_ ) [2] + _δ√_
_κ_ [4] ~~_[√]_~~ _r_ _[·]_




       - _√_
_k_ (min _{n, R} −_ _r_ ) 1 + 80 _µc_ 2



_kσ_ min( _**X**_ ) [2][�][2(] _[t][−][t]_ [1][)] _∥_ _**U**_ _t_ 1 _∗_ _**W**_ _t_ 1 _,⊥∥_ [2]




       - _√_
_k_ (min _{n, R} −_ _r_ ) 1 + 80 _µc_ 2 _kσ_ min( _**X**_ ) [2][�][2(] _[t][−][t]_ [1][)] _·_ 9 _γ_ [7] _[/]_ [4] _σ_ min( _**X**_ ) [1] _[/]_ [4]



_√_
_≤_ 10 _c_ 1


_√_
_≤_ 10 _c_ 1


_√_
_≤_ 10 _c_ 1

_√_
_≤_ 40 _c_ 1



_√_
_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2] + _δ_



_√_
_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2] + 9 _δ_


_√_
_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2] + 9 _δ_


_kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2]




       - _√_
_k_ (min _{n, R} −_ _r_ ) 1 + 80 _µc_ 2 _kσ_ min( _**X**_ ) [2][�][2(] _[t]_ [2] _[−][t]_ [1][)] _γ_ [7] _[/]_ [4] _σ_ min( _**X**_ ) [1] _[/]_ [4]




     - 5      - _r_ _∥_ _**X**_ _∥_ [7] _[/]_ [4]
_k_ (min _{n, R} −_ _r_ )
18 _[κ]_ [1] _[/]_ [4] _k_ (min _{n, R} −_ _r_ ) _γ_ [7] _[/]_ [4]




- _O_ ( _c_ 2)
_γ_ [7] _[/]_ [4] _σ_ min( _**X**_ ) [1] _[/]_ [4]



where we have used the bounds _δ_ _≤_ _κ_ [4] _c_ ~~_[√]_~~ 1 _r_ [,] _[∥]_ _**[X]**_ _[∥]_ [=] _[κσ]_ [min][(] _**[X]**_ [)][,] _[∥]_ _**[U]**_ _[t]_ [1] _[∗]_ _**[W]**_ _[t]_ [1] _[,][⊥][∥≤]_ [3] _[γ]_ [7] _[/]_ [8] _[σ]_ [min][(] _**[X]**_ [)][1] _[/]_ [8][,] [along] [with] [the]
inductive assumptions and the definition of _t_ 1.

Next, we note that if _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [)] _[ ≤]_ [1] 2 _[σ]_ [min][(] _**[X]**_ [)][, then we can use Lemma E.1 along with the inductive assumptions to]

obtain


_σ_ min( _**U**_ _t_ +1 _∗_ _**W**_ _t_ +1) _≥_ _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1][)]

_≥_ _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [)]

                 -                 _≥_ _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [)] 1 + [1] _**X**_ _[∗]_ _**[U]**_ _[t]_ [)][2]

4 _[µσ]_ [min][(] _**[X]**_ [)][2] _[ −]_ _[µσ]_ [min][(] _**[V]**_ _[⊤]_




     _≥_ _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [)] 1 + [1]




[1] 
4 _[σ]_ [min][(] _**[X]**_ [)][2]




[1] [1]

4 _[µσ]_ [min][(] _**[X]**_ [)][2] _[ −]_ _[µ][ ·]_ 4



= _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [)]

1
_≥_ ~~_√_~~

10 _[σ]_ [min][(] _**[X]**_ [)]


Alternatively, if _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [)] _[ ≥]_ [1] 2 _[σ]_ [min][(] _**[X]**_ [)][, then we can again use Lemma E.1 along with the inductive assumptions and]

the fact that _µ ≤_ _c_ 1 _κ_ _[−]_ [2] _∥_ _**X**_ _∥_ [2] for sufficiently small _c_ 1 to obtain


_σ_ min( _**U**_ _t_ +1 _∗_ _**W**_ _t_ +1) _≥_ _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1][)]

_≥_ _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [)]

                 -                 _≥_ _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [)] 1 + [1] _**X**_ _[∗]_ _**[U]**_ _[t]_ [)][2]

4 _[µσ]_ [min][(] _**[X]**_ [)][2] _[ −]_ _[µσ]_ [min][(] _**[V]**_ _[⊤]_

_≥_ [1] �1 _−_ _µσ_ min( _**U**_ _t_ ) [2][�]

2 _[σ]_ [min][(] _**[X]**_ [)]

_≥_ [1] �1 _−_ _µ∥_ _**U**_ _t∥_ [2][�]

2 _[σ]_ [min][(] _**[X]**_ [)]

_≥_ [1] �1 _−_ 9 _µ∥_ _**X**_ _∥_ [2][�]

2 _[σ]_ [min][(] _**[X]**_ [)]

_≥_ [1] �1 _−_ 9 _c_ 1 _κ_ _[−]_ [2][�]

2 _[σ]_ [min][(] _**[X]**_ [)]

1
_≥_ ~~_√_~~

10 _[σ]_ [min][(] _**[X]**_ [)]


In either case, we have _σ_ min( _**U**_ _t_ +1 _∗_ _**W**_ _t_ +1) _≥_ _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1][)] _[ ≥]_ ~~_√_~~ 110 _[σ]_ [min][(] _**[X]**_ [)][.]


45


**Implicit Regularization for Tubal Tensors via GD**


Again, since _σ_ min( _**V**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [)] _[≥]_ ~~_√_~~ 110 _[σ]_ [min][(] _**[X]**_ [)] _[>]_ [0][,] [we have that] _**[ V]**_ _[⊤]_ _**X**_ _[∗]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [has full tubal rank with all]

invertible t-SVD singular tubes. Hence, by Lemma E.3, we again can bound




        - _√_
_∥_ _**U**_ _t_ +1 _∗_ _**W**_ _t_ +1 _,⊥∥≤_ 1 + 80 _µc_ 2 _kσ_ min( _**X**_ ) [2][�] _[t]_ [+1] _[−][t]_ [1] _∥_ _**U**_ _t_ 1 _∗_ _**W**_ _t_ 1 _,⊥∥_ _._



In the exact same way as before, we can use Lemma E.6 to establish _∥_ _**U**_ _t_ +1 _∥≤_ 3 _∥_ _**X**_ _∥_, and use Lemma E.7 to establish
_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[W]**_ _[t]_ [+1] _[∥≤]_ _[c]_ [2] _[κ][−]_ [2][.]

To bound _∥_ _**V**_ _[⊤]_ _**X**_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ [+1] _[∗]_ _**[U]**_ _[⊤]_ _t_ +1 [)] _[∥][F]_ [, we will aim to use Lemma E.7.] [By the inductive assumptions, we already]
have _∥_ _**U**_ _t∥≤_ 3 _∥_ _**X**_ _∥_, _σ_ min( _**U**_ _t ∗_ _**W**_ _t_ ) _≥_ ~~_√_~~ 110 _[σ]_ [min][(] _**[X]**_ [)][, and] _[ ∥]_ _**[V]**_ _**X**_ _[⊤]_ _[⊥]_ _[∗]_ _**[V]**_ _**[U]**_ _[t][∗]_ _**[W]**_ _[t][∥≤]_ _[c]_ [2] _[κ][−]_ [2][.] [To derive the remaining condition]

of Lemma E.7, we first split


_∥_ _**V**_ _[⊤]_ _**X**_ _[∗]_ [(] _[I −A][∗][A]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[∗]_ _**[U]**_ _[⊤]_ [)] _[∥][F]_

= _∥_ _**V**_ _[⊤]_ _**X**_ _[∗]_ [(] _[I −A][∗][A]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t]_ _**[W]**_ _[⊤]_ _t_ _[∗]_ _**[U]**_ _t_ _[⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _**[W]**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_

_≤∥_ _**V**_ _[⊤]_ _**X**_ _[∗]_ [(] _[I −A][∗][A]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t]_ _[∗]_ _**[W]**_ _[⊤]_ _t_ _[∗]_ _**[U]**_ _t_ _[⊤]_ [)] _[∥][F]_ [+] _[ ∥]_ _**[V]**_ _[⊤]_ _**X**_ _[∗]_ [(] _[I −A][∗][A]_ [)(] _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_ _[.]_


To bound the first term, we note that _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**W**_ _t ∗_ _**W**_ _[⊤]_ _t_ _[∗]_ _**[U]**_ _t_ _[⊤]_ [is tubal-symmetric with tubal rank at most][ 2] _[r]_ [, so]
we can write it as the sum of two tubal-symmetric tensors _**Z**_ 1 _,_ _**Z**_ 2 _∈_ _S_ _[n][×][n][×][k]_ with tubal rank at most _r_, and then apply
Lemma G.4 to obtain


_∥_ _**V**_ _[⊤]_ _**X**_ _[∗]_ [(] _[I −A][∗][A]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t]_ _[∗]_ _**[W]**_ _[⊤]_ _t_ _[∗]_ _**[U]**_ _t_ _[⊤]_ [)] _[∥][F]_ [=] _[ ∥]_ _**[V]**_ _[⊤]_ _**X**_ _[∗]_ [(] _[I −A][∗][A]_ [)(] _**[Z]**_ [1] [+] _**[ Z]**_ [2][)] _[∥][F]_



_≤∥_ _**V**_ _[⊤]_ _**X**_ _[∗]_ [(] _[I −A][∗][A]_ [)(] _**[Z]**_ [1][)] _[∥][F]_ [+] _[ ∥]_ _**[V]**_ _[⊤]_ _**X**_ _[∗]_ [(] _[I −A][∗][A]_ [)(] _**[Z]**_ [2][)] _[∥][F]_
_≤_ _δ_ ( _∥_ _**Z**_ 1 _∥F_ + _∥_ _**Z**_ 2 _∥F_ )



_√_
_≤_ _δ_

_√_
= _δ_

_√_
_≤_ _δ_



2 _∥_ _**Z**_ 1 + _**Z**_ 2 _∥F_

2 _∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**W**_ _t ∗_ _**W**_ _[⊤]_ _t_ _[∗]_ _**[U]**_ _t_ _[⊤][∥][F]_

2 _∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ _[∥][F]_



For the second piece, we use the symmetric t-SVD to write _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [=][ �] _i_ _**[V]**_ _[i][ ∗]_ _**[s]**_ _[i][ ∗]_ _**[V]**_ _i_ _[⊤]_ [.] [Then, we can]
bound



��

_**V**_ _i ∗_ _**s**_ _i ∗_ _**V**_ _[⊤]_ _i_

_i_



������ _F_



_∥_ _**V**_ _[⊤]_ _**X**_ _[∗]_ [(] _[I −A][∗][A]_ [)(] _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_ [=]



_**V**_ _⊤_ _**X**_ _[∗]_ [(] _[I −A][∗][A]_ [)]
�����



_≤_ 

_i_



��� _**V**_ _⊤_ _**X**_ _[∗]_ [(] _[I −A][∗][A]_ [)] - _**V**_ _i ∗_ _**s**_ _i ∗_ _**V**_ _[⊤]_ _i_ ���� _F_



_≤_ - _δ_ ��� _**V**_ _i ∗_ _**s**_ _i ∗_ _**V**_ _⊤i_ ��� _F_

_i_



= - _δ ∥_ _**s**_ _i∥_ 2

_i_



= _δ_ ��� _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _⊤t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ ��� _∗_

_≤_ _δ_ ~~�~~ _k_ (min _{n, R} −_ _r_ ) ��� _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _⊤t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ ��� _F_
_≤∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ _[∥][F]_ _[,]_


where we have used the fact that _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [has tubal rank] _[ ≤]_ [min] _[{][n, R][} −]_ _[r]_ [ along with the definition of] _[ t]_ [3][.]


46


**Implicit Regularization for Tubal Tensors via GD**


Hence,

_∥_ _**V**_ _[⊤]_ _**X**_ _[∗]_ [(] _[I −A][∗][A]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[∗]_ _**[U]**_ _[⊤]_ [)] _[∥][F]_



_≤∥_ _**V**_ _[⊤]_ _**X**_ _[∗]_ [(] _[I −A][∗][A]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t]_ _[∗]_ _**[W]**_ _[⊤]_ _t_ _[∗]_ _**[U]**_ _t_ _[⊤]_ [)] _[∥][F]_ [+] _[ ∥]_ _**[V]**_ _[⊤]_ _**X**_ _[∗]_ [(] _[I −A][∗][A]_ [)(] _**[U]**_ _[t]_ _[∗]_ _**[W]**_ _[t,][⊥]_ _[∗]_ _**[W]**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_



_√_
_≤δ_



2 _∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ _[∥][F]_ [+] _[ δ][∥]_ _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ _[∥][F]_



_≤cκ_ _[−]_ [2] _∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ _[∥][F]_ _[,]_

where we have used the assumption that _δ_ _≤_ _c_ 1
_κ_ [4] ~~_[√]_~~ _r_ _[≤]_ _[cκ][−]_ [2] _[.]_


Similarly, we can bound

_∥_ _**V**_ _[⊤]_ _**U**_ _t∗_ _**W**_ _t_ _[∗]_ [(] _[I −A][∗][A]_ [)(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[t]_ [)] _[∥][F]_ _[≤]_ _[cκ][−]_ [2] _[∥]_ _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ _[∗]_ _**[U]**_ _[⊤]_ _t_ _[∥][F]_ _[,]_


and
_∥_ ( _I −A_ _[∗]_ _A_ )( _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _t_ ) _∥≤_ _cκ_ _[−]_ [2] _∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_ _t ∗_ _**U**_ _[⊤]_ _t_ _[∥][F]_ _[.]_

Then, by Lemma E.7, we have


                  -                   _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ [+1] _[ ∗]_ _**[U]**_ _[⊤]_ _t_ +1 [)] _[∥][F]_ _[≤]_ 1 _−_ _[µ]_ min [(] _**[X]**_ [)] _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_

200 _[σ]_ [2]

min [(] _**[X]**_ [)]
+ _µ_ _[σ]_ [2] _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _t_ _[⊤][∥][F]_

100



By the inductive assumption,

_√_
_∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_ _[≤]_ [10] _kr_ �1 _−_ 4001 _[µσ]_ [min][(] _**[X]**_ [)][2][�] _[t][−][t]_ [1] _[ ∥]_ _**[X]**_ _[∥]_ [2] _[.]_


Also, using the inductive assumption and the bound from the previous part, we can bound

_∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ _[∥][F]_ _[≤]_    - _k_ (min _{n, R} −_ _r_ ) _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _t_ _[⊤][∥]_

_≤_           - _k_ (min _{n, R} −_ _r_ ) _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥∥_ [2]



_≤_ - _k_ (min _{n, R} −_ _r_ ) �1 + 80 _µc_ 2 _√kσ_ min( _**X**_ ) [2][�][2(] _[t][−][t]_ [1][)] _∥_ _**U**_ _t_ 1 _∗_ _**W**_ _t_ 1 _,⊥∥_ [2]

_≤_ - _k_ (min _{n, R} −_ _r_ ) �1 + 80 _µc_ 2 _√kσ_ min( _**X**_ ) [2][�][2(] _[t][−][t]_ [1][)] _·_ 9 _γ_ [7] _[/]_ [4] _σ_ min( _**X**_ ) [1] _[/]_ [4]



Since _t ≤_ _t_ 2, we have


and thus,



300  - 5  - _r_ _∥_ _**X**_ _∥_ [7] _[/]_ [4]

_kσ_ min( _**X**_ ) [2] [ln] 18 _[κ]_ [1] _[/]_ [4] min _{n, R} −_ _r_ _γ_ [7] _[/]_ [4]



300
_t −_ _t_ 1 _≤_ _t_ 2 _−_ _t_ 1 _≤_ ~~_√_~~
_µ_ _kσ_ min



_γ_ [7] _[/]_ [4]




_,_



_∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _t_ _[⊤][∥][F]_ _[≤]_ - _k_ (min _{n, R} −_ _r_ ) �1 + 80 _µc_ 2 _√kσ_ min( _**X**_ ) [2][�][2(] _[t][−][t]_ [1][)] _·_ 9 _γ_ [7] _[/]_ [4] _σ_ min( _**X**_ ) [1] _[/]_ [4]




 _kr_ 1 _−_ _[µ]_ _∥_ _**X**_ _∥_ [2] _._

400 _[σ]_ [min][(] _**[X]**_ [)][2][�] _[t][−][t]_ [1]



_≤_ [5]

2



_√_



Combining these inequalities yields


                 -                 _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t]_ [+1] _[ ∗]_ _**[U]**_ _[⊤]_ _t_ +1 [)] _[∥][F]_ _[≤]_ 1 _−_ _[µ]_ min [(] _**[X]**_ [)] _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ _[t][ ∗]_ _**[U]**_ _[⊤]_ _t_ [)] _[∥][F]_

200 _[σ]_ [2]

min [(] _**[X]**_ [)]
+ _µ_ _[σ]_ [2] _∥_ _**U**_ _t ∗_ _**W**_ _t,⊥_ _∗_ _**W**_ _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_ _[⊤]_ _t_ _[∥][F]_

100




 - 1  - _t−t_ 1
_kr_ 1 _−_ _∥_ _**X**_ _∥_ [2]
400 _[µσ]_ [min][(] _**[X]**_ [)][2]




 -  - _√_
_≤_ 1 _−_ _[µ]_ min [(] _**[X]**_ [)] _·_ 10

200 _[σ]_ [2]




_[µ]_ min [(] _**[X]**_ [)]  - _·_ 10 _√_

200 _[σ]_ [2]




 _kr_ 1 _−_ _[µ]_ _∥_ _**X**_ _∥_ [2]

400 _[σ]_ [min][(] _**[X]**_ [)][2][�] _[t][−][t]_ [1]



min [(] _**[X]**_ [)]
+ _µ_ _[σ]_ [2] _·_ [5]

100 2



_√_



_√_  - 1  - _t_ +1 _−t_ 1
_≤_ 10 _kr_ 1 _−_ _∥_ _**X**_ _∥_ [2]

400 _[µσ]_ [min][(] _**[X]**_ [)][2]


47


**Implicit Regularization for Tubal Tensors via GD**


Hence, by induction, the five bullet points hold for _t_ + 1.


If [�] _t_ = _t_ 2, then, we can use Lemma E.7, the previous bullet points, and the definition of _t_ 2 to bound


_∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_  - _t ∗_ _**U**_  - _[⊤]_ _t_ _[∥][F]_ _[≤]_ [4] _[∥]_ _**[V]**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ [�] _t_ _[∗]_ _**[U]**_  - _[⊤]_ _t_ [)] _[∥][F]_ [+] _[ ∥]_ _**[U]**_  - _t_ _[∗]_ _**[W]**_  - _t,⊥_ _[∗]_ _**[W]**_  - _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_  - _[⊤]_ _t_ _[∥][F]_



_√_  - 1 �� _t−t_ 1
_≤_ 40 _kr_ 1 _−_ _∥_ _**X**_ _∥_ [2] + [5]

400 _[µσ]_ [min][(] _**[X]**_ [)][2] 2



_√_




 - 1 �� _t−t_ 1
_kr_ 1 _−_ _∥_ _**X**_ _∥_ [2]
400 _[µσ]_ [min][(] _**[X]**_ [)][2]



= [85]

2



_√_ - 1 �� _t−t_ 1

_kr_ 1 _−_ _∥_ _**X**_ _∥_ [2]
400 _[µσ]_ [min][(] _**[X]**_ [)][2]



_√_
≲




 - 5 ~~�~~ _r_ _∥_ _**X**_ _∥_ [7] _[/]_ [4]
_kr_
18 _[κ]_ [1] _[/]_ [4] _k_ (min _{n, R} −_ _r_ ) _γ_ [7] _[/]_ [4]




 - 5 ~~�~~ _r_ _∥_ _**X**_ _∥_ [7] _[/]_ [4]
_kr_
18 _[κ]_ [1] _[/]_ [4] _k_ (min _{n, R} −_ _r_ ) _γ_ [7] _[/]_ [4]




_−_ 3 _/_ 4

_∥_ _**X**_ _∥_ [2]



≲ _k_ [5] _[/]_ [4] _r_ [1] _[/]_ [8] _κ_ _[−]_ [3] _[/]_ [16] (min _{n, R} −_ _r_ ) [3] _[/]_ [8] _γ_ [21] _[/]_ [16] _∥_ _**X**_ _∥_ [11] _[/]_ [16]


If instead we have [�] _t_ = _t_ 3, then


_∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_    - _t ∗_ _**U**_    - _[⊤]_ _t_ _[∥][F]_

_≤_ 4 _∥_ _**V**_ _[⊤]_ _**X**_ _[⊥]_ _[∗]_ [(] _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤]_ _[−]_ _**[U]**_ [�] _t_ _[∗]_ _**[U]**_  - _[⊤]_ _t_ [)] _[∥][F]_ [+] _[ ∥]_ _**[U]**_  - _t_ _[∗]_ _**[W]**_  - _t,⊥_ _[∗]_ _**[W]**_  - _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_  - _[⊤]_ _t_ _[∥][F]_

_≤_ 4 _∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_  - _t ∗_ _**U**_  - _[⊤]_ _t_ _[∥][F]_ [+] _[ ∥]_ _**[U]**_  - _t_ _[∗]_ _**[W]**_  - _t,⊥_ _[∗]_ _**[W]**_  - _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_  - _t_ _[⊤][∥][F]_

_≤_ 4( ~~�~~ _k_ (min _{n, R} −_ _r_ ) + 1) _∥_ _**U**_  - _t ∗_ _**W**_  - _t,⊥_ _∗_ _**W**_  - _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_  - _[⊤]_ _t_ _[∥][F]_ [+] _[ ∥]_ _**[U]**_  - _t_ _[∗]_ _**[W]**_  - _t,⊥_ _[∗]_ _**[W]**_  - _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_  - _[⊤]_ _t_ _[∥][F]_

=4( ~~�~~ _k_ (min _{n, R} −_ _r_ ) + 5) _∥_ _**U**_   - _t ∗_ _**W**_   - _t,⊥_ _∗_ _**W**_   - _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_   - _[⊤]_ _t_ _[∥][F]_

_≤_ 4( ~~�~~ _k_ (min _{n, R} −_ _r_ ) + 5)�min _{n, R} −_ _r∥_ _**U**_  - _t ∗_ _**W**_  - _t,⊥_ _∗_ _**W**_  - _[⊤]_ _t,⊥_ _[∗]_ _**[U]**_  - _[⊤]_ _t_ _[∥]_

_≤_ 4( ~~�~~ _k_ (min _{n, R} −_ _r_ ) + 5)�min _{n, R} −_ _r∥_ _**U**_  - _t ∗_ _**W**_  - _t,⊥∥_ [2]



_≤_ 4( ~~�~~ _k_ (min _{n, R} −_ _r_ ) + 5)� _k_ (min _{n, R} −_ _r_ ) �1 + 80 _µc_ 2 _√_

_≤_ 4( ~~�~~ _k_ (min _{n, R} −_ _r_ ) + 5)� _k_ (min _{n, R} −_ _r_ ) �1 + 80 _µc_ 2 _√_



_kσ_ min( _**X**_ ) [2][�][2(][�] _[t][−][t]_ [1][)] _∥_ _**U**_ _t_ 1 _∗_ _**W**_ _t_ 1 _,⊥∥_ [2]

_kσ_ min( _**X**_ ) [2][�][2(][�] _[t][−][t]_ [1][)] _·_ 9 _γ_ [63] _[/]_ [32] _σ_ min( _**X**_ ) [1] _[/]_ [32]




- _O_ ( _c_ 2)
_γ_ [63] _[/]_ [32] _σ_ min( _**X**_ ) [1] _[/]_ [32]




      - 5 ~~�~~ _r_ _∥_ _**X**_ _∥_ [7] _[/]_ [4]
≲ _k_ (min _{n, R} −_ _r_ )
18 _[κ]_ [1] _[/]_ [4] _k_ (min _{n, R} −_ _r_ ) _γ_ [7] _[/]_ [4]




      - 5 ~~�~~ _r_ _∥_ _**X**_ _∥_ [7] _[/]_ [4]
≲ _k_ (min _{n, R} −_ _r_ )
18 _[κ]_ [1] _[/]_ [4] _k_ (min _{n, R} −_ _r_ ) _γ_ [7] _[/]_ [4]


      - 5 ~~�~~ _r_ _∥_ _**X**_ _∥_ [7] _[/]_ [4]
≲ _k_ (min _{n, R} −_ _r_ )
18 _[κ]_ [1] _[/]_ [4] _k_ (min _{n, R} −_ _r_ ) _γ_ [7] _[/]_ [4]




- _O_ ( _c_ 2) _[∥]_ _**[X]**_ _[∥]_ [1] _[/]_ [32]
_γ_ [21] _[/]_ [16] _γ_ [21] _[/]_ [32]

_κ_ [1] _[/]_ [32]

- _O_ ( _c_ 2) - _∥_ _**X**_ _∥_
_γ_ [21] _[/]_ [16]
min _{n, R}κ_ [3]



�21 _/_ 32 _∥_ _**X**_ _∥_ 1 _/_ 32

_κ_ [1] _[/]_ [32]




      - 5 ~~�~~ _r_ _∥_ _**X**_ _∥_ [7] _[/]_ [4]
≲ _k_ (min _{n, R} −_ _r_ )
18 _[κ]_ [1] _[/]_ [4] _k_ (min _{n, R} −_ _r_ ) _γ_ [7] _[/]_ [4]


      - 5 ~~�~~ _r_ _∥_ _**X**_ _∥_ [7] _[/]_ [4]
≲ _k_ (min _{n, R} −_ _r_ )
18 _[κ]_ [1] _[/]_ [4] _k_ (min _{n, R} −_ _r_ ) _γ_ [7] _[/]_ [4]




      - 5 ~~�~~ _r_ _∥_ _**X**_ _∥_ [7] _[/]_ [4]
≲ _k_ (min _{n, R} −_ _r_ )
18 _[κ]_ [1] _[/]_ [4] _k_ (min _{n, R} −_ _r_ ) _γ_ [7] _[/]_ [4]



≲ _[k]_ [(min] _[{][n, R][} −]_ _[r]_ [)]

min _{n, R}_ [21] _[/]_ [32]




- 5 ~~�~~ _r_ _∥_ _**X**_ _∥_ [7] _[/]_ [4]
18 _[κ]_ [1] _[/]_ [4] _k_ (min _{n, R} −_ _r_ ) _γ_ [7] _[/]_ [4]




- 5 ~~�~~ _r_ _∥_ _**X**_ _∥_ [7] _[/]_ [4]
18 _[κ]_ [1] _[/]_ [4] _k_ (min _{n, R} −_ _r_ ) _γ_ [7] _[/]_ [4]




- _O_ ( _c_ 2)
_γ_ [21] _[/]_ [16] _κ_ _[−]_ [2] _∥_ _**X**_ _∥_ [11] _[/]_ [16]



≲ _k_ [5] _[/]_ [4] _r_ [1] _[/]_ [8] _κ_ _[−]_ [3] _[/]_ [16] (min _{n, R} −_ _r_ ) [3] _[/]_ [8] _γ_ [21] _[/]_ [16] _∥_ _**X**_ _∥_ [11] _[/]_ [16] _._


So in either case, we have


_∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_         - _t ∗_ _**U**_         - _[⊤]_ _t_ _[∥][F]_ [≲] _[k]_ [5] _[/]_ [4] _[r]_ [1] _[/]_ [8] _[κ][−]_ [3] _[/]_ [16][(min] _[{][n, R][} −]_ _[r]_ [)][3] _[/]_ [8] _[γ]_ [21] _[/]_ [16] _[∥]_ _**[X]**_ _[∥]_ [11] _[/]_ [16] _[,]_


and thus,

_∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _−_ _**U**_        - _t_ _∗_ _**U**_        - _[⊤]_ _t_ _[∥][F]_ ≲ _k_ [5] _[/]_ [4] _r_ [1] _[/]_ [8] _κ_ _[−]_ [3] _[/]_ [16] (min _{n, R} −_ _r_ ) [3] _[/]_ [8] _γ_ [21] _[/]_ [16] _∥_ _**X**_ _∥_ _[−]_ [21] _[/]_ [16] _._
_∥_ _**X**_ _∥_ [2]


48


**Implicit Regularization for Tubal Tensors via GD**


Finally, by the definition of _t_, we have that

[�]


   - _t −_ _t∗_ _≤_ _t_ 2 _−_ _t∗_
_≤_ ( _t_ 2 _−_ _t_ 1) + ( _t_ 1 _−_ _t∗_ )



300  - 5  - _r_ _∥_ _**X**_ _∥_ [7] _[/]_ [4]

_kσ_ min( _**X**_ ) [2] [ln] 18 _[κ]_ [1] _[/]_ [4] _k_ (min _{n, R} −_ _r_ ) _γ_ [7] _[/]_ [4]



300
_≤_ ~~_√_~~
_µ_ _kσ_ min



_γ_ [7] _[/]_ [4]




- 16 - 2 + _µσ_ min( _**X**_ ) [2] [log] _γ_ ~~_√_~~ 10 _[σ]_ [min][(] _**[X]**_ [)]




- 16 - 2
+ _µσ_ min( _**X**_ ) [2] [log] _γ_ ~~_√_~~



1          -          - _κr_          - _∥_ _**X**_ _∥_
≲ _µσ_ min( _**X**_ ) [2] [ln] min 1 _,_ _k_ (min _{n, R} −_ _r_ ) _γ_


**F. Proof of Main Result**







Now that our analyses of the spectral stage and the convergence stage are complete, we are ready to combine these pi _√_ eces to
obtain the proof of our main result. Since _A_ satisfies RIP(2 _r_ + 1 _, δ_ ), by Lemma G.2, _A_ also satisfies S2SRIP(2 _r,_ 2 _krδ_ ).

Hence, _**E**_ := ( _I −A_ _[∗]_ _A_ )( _**X**_ _∗_ _**X**_ _[⊤]_ ) satisfies



_√_ _√_
_∥_ _**E**_ _∥_ = _∥_ ( _I −A_ _[∗]_ _A_ )( _**X**_ _∗_ _**X**_ _[⊤]_ ) _∥≤_ 2 _krδ∥_ _**X**_ _∗_ _**X**_ _[⊤]_ _∥≤_



_√_
2 _kr · cκ_ _[−]_ [4] _r_ _[−]_ [1] _[/]_ [2] _· ∥_ _**X**_ _∥_ [2] = _c_ _kκ_ _[−]_ [2] _σ_ min( _**X**_ ) [2] _._



_C_ 1˜ _[e][−]_ [3˜] _[c]_ [,] [we have that with probability at least][ 1] _[ −]_ _[k]_ [( ˜] _[Cϵ]_ [)] _[R][−]_ [2] _[r]_ [+1] _[−]_ _[ke][−][cR]_ [˜] [=]



Then, by applying Lemma D.9, with _ϵ_ = 1




[1]
1 _−_ _ke_ _[−]_ [3˜] _[c]_ [(] _[R][−]_ [2] _[r]_ [+1)] _−_ _ke_ _[−][cR]_ [˜] _≥_ 1 _−_ _ke_ _[−]_ [3˜] _[c][·]_ 3



3 _[R]_ _−_ _ke_ _[−][cR]_ [˜] = 1 _−_ _O_ ( _ke_ _[−][cR]_ [˜] ), after




     
1 2 _κ_ [2] _[√]_ _n_
_t∗_ ≲ _µσ_ min( _**X**_ ) [2] [ln] _c_ ˜3�min _{n_ ; _R}_







iterations, we have


and for each 1 _≤_ _j_ _≤_ _k_, we have


where (since _R ≥_ 3 _r_ and _ϵ_ is a constant),


_√_


By choosing



_∥_ _**U**_ _t⋆_ _∥≤_ 3 _∥_ _**X**_ _∥_ (F.1)

_∥_ _**V**_ _**X**_ _⊥_ _∗_ _**V**_ _**U**_ _t⋆_ _∗_ _**W**_ _t⋆_ _∥≤_ _cκ_ _[−]_ [2] _._ (F.2)


  - ( _j_ ) [�]
_σr_ _**U**_ _t⋆_ _∗_ _**W**_ _t⋆_ _≥_ 4 [1] _[αβ]_ (F.3)


 - ( _j_ ) [�]
_σ_ 1 _**U**_ _t⋆_ _∗_ _**W**_ _t⋆,⊥_ _≤_ _[κ]_ 8 _[−]_ [2] _[αβ]_ (F.4)

(F.5)



_√_
_k_ ≲ _β_ ≲



_k_



�16 _κ_ [2]




2 _κ_ [2] _[√]_ _n_

_c_ ˜3�min _{n_ ; _R}_



_._


- _−_ 16 _κ_ [2]



4 _c_ 2 _σ_ min( _**X**_ )
_α_ ≲ ~~_√_~~
_κ_ [2] min _{n, R}_



_k_




2 _κ_ [2] _[√]_ _n_

_c_ ˜3�min _{n, R}_



_,_



we have



_c_ 2 _σ_ min( _**X**_ )

_γ_ = [1] [≲]

4 _[αβ]_ _κ_ [2] min _{n, R}_ _[.]_



Also, _[κ][−]_ 8 [2] _[αβ]_ [=] 2 _κ_ 1 [2] _[γ]_ _[≤]_ [2] _[γ]_ [holds.] [Therefore, we can apply Theorem E.1, which gives us that after]



1       -       - _κr_

- _t −_ _t∗_ ≲ _µσ_ min( _**X**_ ) [2] [ln] min 1 _,_ _k_ (min _{n, R} −_ _r_ )


49




- _∥_ _**X**_ _∥_

_γ_






**Implicit Regularization for Tubal Tensors via GD**


iterations beyond the first phase, we have

_∥_ _**U**_        - _t_ _∗_ _**U**_        - _[⊤]_ _t_ _[−]_ _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤][∥][F]_ ≲ _k_ [5] _[/]_ [4] _r_ [1] _[/]_ [8] _κ_ _[−]_ [3] _[/]_ [16] (min _{n, R} −_ _r_ ) [3] _[/]_ [8] _γ_ [21] _[/]_ [16] _∥_ _**X**_ _∥_ _[−]_ [21] _[/]_ [16] _._
_∥_ _**X**_ _∥_ [2]


The total amount of iterations is then bounded by




- _t_ = _t∗_ + (� _t −_ _t∗_ )


1
≲ _µσ_ min( _**X**_ ) [2] [ln]


1
≲ _µσ_ min( _**X**_ ) [2] [ln]


1
≲ _µσ_ min( _**X**_ ) [2] [ln]




- 2 _κ_ [2] _[√]_ _n_ - _κr_

_·_ min 1 _,_
_c_ ˜3�min _{n, R}_ _k_ (min _{n, R} −_ _r_ )

- 2 _κ_ [2] _[√]_ _n_ - _κr_

_·_ min 1 _,_
_c_ ˜3�min _{n, R}_ _k_ (min _{n, R} −_ _r_ )




2 _κ_ [2] _[√]_ _n_

_c_ ˜3�min _{n, R}_







1    -    - _κr_
+ _µσ_ min( _**X**_ ) [2] [ln] min 1 _,_ _k_ (min _{n, R} −_ _r_ )




- _∥_ _**X**_ _∥_

_γ_








- _∥_ _**X**_ _∥_

_γ_








- 4 _∥_ _**X**_ _∥_

_αβ_







1    - _C_ 1 _κn_    - _κr_
≲ _µσ_ min( _**X**_ ) [2] [ln] min _{n, R}_ _[·]_ [ min] 1 _,_ _k_ (min _{n, R} −_ _r_ )




- _∥_ _**X**_ _∥_

_kα_




_,_



_√_

where we have used the choice of _γ_ = 4 [1] _[αβ]_ [and the fact that] _[ β]_ [≳]



_k_ . Finally, the error is bounded by



_∥_ _**U**_ - _t_ _∗_ _**U**_ - _[⊤]_ _t_ _[−]_ _**[X]**_ _[∗]_ _**[X]**_ _[ ⊤][∥][F]_ ≲ _k_ [5] _[/]_ [4] _r_ [1] _[/]_ [8] _κ_ _[−]_ [3] _[/]_ [16] (min _{n, R} −_ _r_ ) [3] _[/]_ [8] _γ_ [21] _[/]_ [16] _∥_ _**X**_ _∥_ _[−]_ [21] _[/]_ [16]
_∥_ _**X**_ _∥_ [2]

≲ _k_ [5] _[/]_ [4] _r_ [1] _[/]_ [8] _κ_ _[−]_ [3] _[/]_ [16] (min _{n, R} −_ _r_ ) [3] _[/]_ [8] ( _αβ_ ) [21] _[/]_ [16] _∥_ _**X**_ _∥_ _[−]_ [21] _[/]_ [16]


                  2 _κ_ [2] _[√]_ _n_
≲ _k_ [5] _[/]_ [4] _r_ [1] _[/]_ [8] _κ_ _[−]_ [3] _[/]_ [16] (min _{n, R} −_ _r_ ) [3] _[/]_ [8] _k_ [21] _[/]_ [32]

_c_ ˜3�min _{n, R}_



�21 _κ_ [2] _α_
_∥_ _**X**_ _∥_



�21 _/_ 16




           _C_ 2 _κ_ [2] _[√]_ _n_
≲ _k_ [61] _[/]_ [32] _r_ [1] _[/]_ [8] _κ_ _[−]_ [3] _[/]_ [16] (min _{n, R} −_ _r_ ) [3] _[/]_ [8]
�min _{n, R}_



�21 _κ_ [2] _α_
_∥_ _**X**_ _∥_



�21 _/_ 16
_,_



as desired.


Remark: One could obtain similar results for the cases where _r_ _≤_ _R_ _<_ 2 _r_ and 2 _r_ _≤_ _R_ _<_ 3 _r_ by choosing the parameter
_ϵ ∈_ (0 _,_ 1) appropriately.


**G. Restricted Isometry Property**


In this section, we show that a measurement operator which satisfies the standard restricted isometry property also satisfies
two other variants of the restricted isometry property - a fact which we used in our analysis of the convergence stage.


We say that a measurement operator _A_ : _S_ _[n][×][n][×][k]_ _→_ R _[m]_ satisfies the spectral-to-spectral Restricted Isometry Property of
rank- _r_ with constant _δ_ _>_ 0 (abbreviated S2SRIP( _r, δ_ )) if for all tensors _**Z**_ _∈_ _S_ _[n][×][n][×][k]_ with tubal-rank _≤_ _r_,


_∥_ ( _I −A_ _[∗]_ _A_ )( _**Z**_ ) _∥≤_ _δ∥_ _**Z**_ _∥._


We say that a measurement operator _A_ : _S_ _[n][×][n][×][k]_ _→_ R _[m]_ satisfies the spectral-to-nuclear Restricted Isometry Property with
constant _δ_ _>_ 0 (abbreviated S2NRIP( _δ_ )) if for all tensors _**Z**_ _∈_ _S_ _[n][×][n][×][k]_ with tubal-rank _≤_ _r_,


_∥_ ( _I −A_ _[∗]_ _A_ )( _**Z**_ ) _∥≤_ _δ∥_ _**Z**_ _∥∗._


**Lemma G.1.** _Suppose that A_ : _S_ _[n][×][n][×][k]_ _→_ R _[m]_ _satisfies RIP_ ( _r_ + _r_ _[′]_ _, δ_ ) _with_ 0 _< δ_ _<_ 1 _._ _Then, for any_ _**Z**_ _,_ _**Y**_ _∈_ _S_ _[n][×][n][×][k]_

_with_ rank( _**Z**_ ) _≤_ _r and_ rank( _**Y**_ ) _≤_ _r_ _[′]_ _, we have_


_|⟨_ ( _I −A_ _[∗]_ _A_ )( _**Z**_ ) _,_ _**Y**_ _⟩| ≤_ _δ∥_ _**Z**_ _∥F ∥_ _**Y**_ _∥F ._


50


**Implicit Regularization for Tubal Tensors via GD**


_Proof._ Let _**Y**_ _[′]_ = _[∥]_ _∥_ _**[Z]**_ _**Y**_ _∥_ _[∥]_ _F_ _[F]_ _**[Y]**_ [so that] _[ ∥]_ _**[Y]**_ _[′][∥][F]_ [=] _[ ∥]_ _**[Z]**_ _[∥][F]_ [ .] [Note that] _**[ Z]**_ [+] _**[ Y]**_ _[′]_ _[∈]_ _[S][n][×][n][×][k]_ [and] _**[ Z]**_ _[−]_ _**[Y]**_ _[′]_ _[∈]_ _[S][n][×][n][×][k]_ [both have tubal]

rank _≤_ _r_ + _r_ _[′]_ . Then, by using the identities _∥_ _**a**_ + _**b**_ _∥_ [2] _−∥_ _**a**_ _−_ _**b**_ _∥_ [2] = 4 _⟨_ _**a**_ _,_ _**b**_ _⟩_ and _∥_ _**a**_ + _**b**_ _∥_ [2] + _∥_ _**a**_ _−_ _**b**_ _∥_ [2] = 2 _∥_ _**a**_ _∥_ [2] + 2 _∥_ _**b**_ _∥_ [2]

(which both hold over any inner product space) along with the fact that _A_ satisfies RIP( _r_ + _r_ _[′]_ _, δ_ ), we have:


�( _I −A_ _[∗]_ _A_ )( _**Z**_ ) _,_ _**Y**_ _[′]_ [�] =     - _**Z**_ _,_ _**Y**_ _[′]_ [�] _−_     - _A_ _[∗]_ _A_ ( _**Z**_ ) _,_ _**Y**_ _[′]_ [�]

=         - _**Z**_ _,_ _**Y**_ _[′]_ [�] _−_         - _A_ ( _**Z**_ ) _, A_ ( _**Y**_ _[′]_ )�



= - _**Z**_ _,_ _**Y**_ _[′]_ [�] _−_ [1]

4

_≤_ - _**Z**_ _,_ _**Y**_ _[′]_ [�] _−_ [1]

4

= - _**Z**_ _,_ _**Y**_ _[′]_ [�] _−_ [1]




  - _∥_ _**Z**_ + _**Y**_ _[′]_ _∥_ [2] _F_ [+] _[ ∥]_ _**[Z]**_ _[−]_ _**[Y]**_ _[′][∥]_ [2] _F_  4 _[δ]_




[1] [+] _**[ Y]**_ _[′]_ [)] _[∥]_ [2] 2 [+] [1]

4 _[∥A]_ [(] _**[Z]**_ 4



4 _[∥A]_ [(] _**[Z]**_ _[−]_ _**[Y]**_ _[′]_ [)] _[∥]_ [2] 2




[1] [+] _**[ Y]**_ _[′][∥]_ [2] _F_ [+] [1]

4 [(1] _[ −]_ _[δ]_ [)] _[∥]_ _**[Z]**_ 4



4 [(1 +] _[ δ]_ [)] _[∥]_ _**[Z]**_ _[−]_ _**[Y]**_ _[′][∥]_ [2] _F_




 -  
[1] _∥_ _**Z**_ + _**Y**_ _[′]_ _∥_ [2] _F_ _[−∥]_ _**[Z]**_ _[−]_ _**[Y]**_ _[′][∥]_ [2] _F_ + [1]

4 4



= [1] - _∥_ _**Z**_ _∥_ [2] _F_ [+] _[ ∥]_ _**[Y]**_ _[′][∥]_ _F_ [2] 
2 _[δ]_



= _δ∥_ _**Z**_ _∥F ∥_ _**Y**_ _[′]_ _∥F_


In a similar manner, �( _I −A_ _[∗]_ _A_ )( _**Z**_ ) _,_ _**Y**_ _[′]_ [�] _≥−δ∥_ _**Z**_ _∥F ∥_ _**Y**_ _[′]_ _∥F_ . Hence, ���( _I −A_ _[∗]_ _A_ )( _**Z**_ ) _,_ _**Y**_ _[′]_ [���] _≤_ _δ∥_ _**Z**_ _∥F ∥_ _**Y**_ _[′]_ _∥F_ . Then,
since _Y_ is a scalar multiple of _Y_ _[′]_, we have



_|⟨_ ( _I −A_ _[∗]_ _A_ )( _**Z**_ ) _,_ _**Y**_ _⟩|_ = _[∥]_ _**[Y]**_ _[′][∥][F]_



_∥_ _[∥]_ _**Y**_ _**[Y]**_ _[′][∥]_ _∥_ _[F]_ _F_ _[δ][∥]_ _**[Z]**_ _[∥][F][ ∥]_ _**[Y]**_ _[′][∥][F]_ [=] _[ δ][∥]_ _**[Z]**_ _[∥][F][ ∥]_ _**[Y]**_ _[∥][F][ .]_



_∥_ _[∥]_ _**Y**_ _**[Y]**_ _[′][∥]_ _∥_ _[F]_ _F_ ���( _I −A_ _[∗]_ _A_ )( _**Z**_ ) _,_ _**Y**_ _[′]_ [���] _≤_ _∥_ _[∥]_ _**Y**_ _**[Y]**_ _[′][∥]_ _∥_ _[F]_



**Lemma** **G.** _√_ **2.** _Suppose_ _that_ _A_ : _S_ _[n][×][n][×][k]_ _→_ R _[m]_ _satisfies_ _RIP_ ( _r_ + 1 _, δ_ 1) _,_ _where_ 0 _<_ _δ_ 1 _<_ 1 _._ _Then,_ _A_ _also_ _satisfies_
_S2SRIP_ ( _r,_ _krδ_ 1) _._


_Proof._ Suppose _**Z**_ _∈_ _S_ _[n][×][n][×][k]_ has tubal-rank _r_ . Since ( _I −A_ _[∗]_ _A_ )( _**Z**_ ) is symmetric, its t-SVD is of the form


( _I −A_ _[∗]_ _A_ )( _**Z**_ ) = _**V**_ ( _I−A∗A_ )( _**Z**_ ) _∗_ **Σ** ( _I−A∗A_ )( _**Z**_ ) _∗_ _**V**_ _[⊤]_ ( _I−A_ _[∗]_ _A_ )( _**Z**_ ) _[.]_



~~_√_~~

_−_ 12 _πjℓ_ where
_k_ _[e]_



Now, define _**V**_ = _**V**_ ( _I−A∗A_ )( _**Z**_ )(: _,_ 1 _,_ :) _∈_ R _[n][×]_ [1] _[×][k]_ and let _**s**_ _∈_ R [1] _[×]_ [1] _[×][k]_ be defined by _**s**_ (1 _,_ 1 _, ℓ_ ) = ~~_√_~~ 1




                      _j_ = arg max _j′ |_ **Σ** [�] (1 _,_ 1 _, j_ _[′]_ ) _|_ . With this definition, one can check that ��� ( _I −A_ _[∗]_ _A_ )( _**Z**_ ) _,_ _**V**_ _∗_ _**s**_ _∗_ _**V**_ _[⊤]_ [���] - = _∥_ ( _I −A_ _[∗]_ _A_ )( _**Z**_ ) _∥_ .



Then, since _A_ satisfies RIP( _r_ + 1 _, δ_ 1) and rank( _**Z**_ ) _≤_ _r_ and rank( _**V**_ _∗_ _**s**_ _∗_ _**V**_ _[⊤]_ ) = 1, by Lemma G.1, we have


                   _∥_ ( _I −A_ _[∗]_ _A_ )( _**Z**_ ) _∥_ = ( _I −A_ _[∗]_ _A_ )( _**Z**_ ) _,_ _**V**_ _∗_ _**s**_ _∗_ _**V**_ _[⊤]_ [���]
���                                                                                                    


_≤_ _δ_ 1 _∥_ _**V**_ _∗_ _**s**_ _∗_ _**V**_ _[⊤]_ _∥F ∥_ _**Z**_ _∥F_
= _δ_ 1 _∥_ _**Z**_ _∥F_



_√_
_≤_ _δ_ 1



_kr∥_ _**Z**_ _∥._



_√_
Since the bound _∥_ ( _I −A_ _[∗]_ _A_ )( _**Z**_ ) _∥≤_ _δ_ 1



Since the bo _√_ und _∥_ ( _I −A_ _[∗]_ _A_ )( _**Z**_ ) _∥≤_ _δ_ 1 _kr∥_ _**Z**_ _∥_ holds for any _**Z**_ _∈_ _S_ _[n][×][n][×][k]_ with tubal rank _≤_ _r_, we have that _A_ satisfies

S2SRIP( _r,_ _krδ_ 1).


**Lemma** _√_ **G.3.** _Suppose_ _that_ _A_ : _S_ _[n][×][n][×][k]_ _→_ R _[m]_ _satisfies_ _RIP_ (2 _, δ_ 2) _where_ 0 _<_ _δ_ 2 _<_ 1 _._ _Then,_ _A_ _also_ _satisfies_
_S2NRIP_ ( _kδ_ 2) _._



_krδ_ 1).



**Lemma** _√_ **G.3.** _Suppose_ _that_ _A_ : _S_ _[n][×][n][×][k]_ _→_ R _[m]_ _satisfies_ _RIP_ (2 _, δ_ 2) _where_ 0 _<_ _δ_ 2 _<_ 1 _._ _Then,_ _A_ _also_ _satisfies_
_S2NRIP_ ( _kδ_ 2) _._


_√_
_Proof._ Since _A_ satisfies RIP(2 _, δ_ 2), by Lemma G.2 for _r_ = 1, _A_ satisfies S2SRIP(1 _,_ _kδ_ 2). Now, suppose that _**Z**_ _∈_

_S_ _[n][×][n][×][k]_ . Since _**Z**_ is symmetric, it has a t-SVD in the form



_√_
_Proof._ Since _A_ satisfies RIP(2 _, δ_ 2), by Lemma G.2 for _r_ = 1, _A_ satisfies S2SRIP(1 _,_



_**Z**_ =



_n_

- _**V**_ _i ∗_ _**s**_ _i ∗_ _**V**_ _[⊤]_ _i_ _[.]_

_i_ =1


51


**Implicit Regularization for Tubal Tensors via GD**


Then, since each term _**V**_ _i ∗_ _**s**_ _i ∗_ _**V**_ _[⊤]_ _i_ [is symmetric with tubal rank][ 1][, we have]




- _n_

 - _**V**_ _i ∗_ _**s**_ _i ∗_ _**V**_ _[⊤]_ _i_


_i_ =1



������



_∥_ ( _I −A_ _[∗]_ _A_ )( _**Z**_ ) _∥_ =


=


_≤_


_≤_


=



( _I −A∗A_ )
�����



�����



_n_
�( _I −A_ _[∗]_ _A_ ) - _**V**_ _i ∗_ _**s**_ _i ∗_ _**V**_ _[⊤]_ _i_ - [�]

_i_ =1 ����



_n_




_n_



_i_ =1

_n_



_i_ =1

_n_




���( _I −A∗A_ ) - _**V**_ _i ∗_ _**s**_ _i ∗_ _**V**_ _[⊤]_ _i_ ����



_√_


_√_



_kδ_ 2 ��� _**V**_ _i ∗_ _**s**_ _i ∗_ _**V**_ _⊤i_ ���


_kδ_ 2 _∥_ _**s**_ _i∥_



_i_ =1



_√_
_≤_



_kδ_ 2 _∥_ _**Z**_ _∥∗_



_√_
Since the bound _∥_ ( _I−A_ _[∗]_ _A_ )( _**Z**_ ) _∥≤_



_√_
_kδ_ 2 _∥_ _**Z**_ _∥∗_ holds for any _**Z**_ _∈_ _S_ _[n][×][n][×][k]_, we have that _A_ satisfies S2NRIP(



_kδ_ 2).



**Lemma G.4.** _Suppose A_ : _S_ _[n][×][n][×][k]_ _→_ R _[m]_ _satisfies RIP_ (2 _r, δ_ 3) _, where_ 0 _< δ_ 3 _<_ 1 _, and_ _**V**_ _∈_ R _[n][×][r][×][k]_ _satisfies_ _**V**_ _[⊤]_ _∗_ _**V**_ = _**I**_ _._
_Then, for any_ _**Z**_ _∈_ _S_ _[n][×][n][×][k]_ _with_ rank( _**Z**_ ) _≤_ _r, we have_

_**V**_ _⊤_ _∗_ [( _I −A∗A_ )( _**Z**_ )]
��� ��� _F_ _[≤]_ _[δ]_ [3] _[∥]_ _**[Z]**_ _[∥][F][ .]_


_**V**_ _[⊤]_ _∗_ [( _I−A_ _[∗]_ _A_ )( _**Z**_ )]
_Proof._ Let _**Z**_ _∈_ _S_ _[n][×][n][×][k]_, and let _**Y**_ = _∥_ _**V**_ _[⊤]_ _∗_ [( _I−A_ _[∗]_ _A_ )( _**Z**_ )] _∥F_ _[∈]_ [R] _[r][×][n][×][k]_ [.] [Trivially,] _[∥]_ _**[Y]**_ _[∥][F]_ [=] [1][,] [and] [so,] _[∥]_ _**[V]**_ _[∗]_ _**[Y]**_ _[∥]_ [2] _F_ [=]

       -       _⟨_ _**V**_ _∗_ _**Y**_ _,_ _**V**_ _∗_ _**Y**_ _⟩_ = _**Y**_ _,_ _**V**_ _[⊤]_ _∗_ _**V**_ _∗_ _**Y**_ = _⟨_ _**Y**_ _,_ _**Y**_ _⟩_ = _∥_ _**Y**_ _∥_ [2] _F_ [= 1][.] [Then, by using Lemma G.1, we have that]


_**V**_ _⊤_ _∗_ [( _I −A∗A_ )( _**Z**_ )]         - _**V**_ _[⊤]_ _∗_ [( _I −A_ _[∗]_ _A_ )( _**Z**_ )] _,_ _**Y**_         ��� ��� _F_ [=]

= _⟨_ [( _I −A_ _[∗]_ _A_ )( _**Z**_ )] _,_ _**V**_ _∗_ _**Y**_ _⟩_


_≤_ _δ_ 3 _∥_ _**Z**_ _∥F ∥_ _**V**_ _∗_ _**Y**_ _∥F_
= _δ_ 3 _∥_ _**Z**_ _∥F_


**H. Properties of Aligned Matrix Subspaces**


In this section, we collect some properties of matrices and their subspaces, useful for the proof of the results in the tensor
Fourier domain.

**Lemma** **H.1.** _((Stoger¨_ _&_ _Soltanolkotabi,_ _2021))_ _For_ _some_ _orthogonal_ _matrix_ _X_ _∈_ C _[n][×][r]_ _and_ _some_ _full-rank_ _matrix_
_Y_ _∈_ C _[n][×][R]_ _consider X_ [H] _Y_ = _V_ Σ _W_ [H] _, and the following decomposition of Y_


_Y_ = _Y WW_ [H] + _Y W⊥W⊥_ [H] (H.1)


_with its SVD decomposition Y_ = [�] _i_ _[R]_ =1 _[σ][i][u][i][v]_ _i_ [H] _[and the best rank-][r][ approximation][ Y][r]_ [=][ �] _[r]_ _i_ =1 _[σ][i][u][i][v]_ _i_ [H] _[.]_ _[Then if the distance]_
_between the column subspace of Yr_ _and the subspace spanned by the columns of X_ _is small enough, that is ∥X⊥_ [H] _[V][Y]_ _r_ _[∥≤]_ [1] 8 _[,]_

_then the decomposition_ (H.1) _follows some low-rank approximation properties, namely_


_∥X⊥_ [H] _[V][Y W]_ _[∥≤]_ [7] _[∥][X]_ _⊥_ [H] _[V][Y]_ _r_ _[∥]_ (H.2)

_∥Y W⊥∥≤_ 2 _σr_ +1( _Y_ ) _._ (H.3)


52


**Implicit Regularization for Tubal Tensors via GD**


**Lemma H.2.** _For a matrix X_ _∈_ C _[n][×][r]_ _, r_ _≤_ _n, with its SVD-decomposition X_ = _VX_ Σ _X_ _WX_ [H] _[and some a full-rank matrix]_
_Y_ _∈_ C _[n][×][R]_ _, consider VX_ [H] _[Y]_ [=] _[ V]_ [ Σ] _[W]_ [ H] _[, and the following decomposition of][ Y]_


_Y_ = _Y WW_ [H] + _Y W⊥W⊥_ [H] _[.]_ (H.4)


_Let matrix H_ _∈_ C _[r][×][r]_ _be defined as_
_H_ = _VX_ [H][(Id +] _[ µZ]_ [)] _[Y W]_

_with some Z_ _∈_ C _[n][×][n]_ _, parameter µ_ _≤_ ~~_√_~~ 13 _[∥][V]_ [H] _[Y][ ∥][−]_ [2] _[and][ ∥][V]_ _⊥_ [H] _[V][Y W][ ∥≤]_ _[c]_ [2] _[with sufficiently small constants][ c]_ [1] _[, c]_ [2] _[>]_ [0] _[.]_

_Then H_ _can be represented as follows_


_H_ = (Id + _µ_ Σ [2] _X_ _[−]_ _[µP]_ [1] [+] _[ µP]_ [2] [+] _[ µ]_ [2] _[P]_ [3][)] _[V][X]_ _[Y W]_ [(Id] _[ −]_ _[µW]_ [ H] _[Y]_ [H] _[V][X]_ _[V]_ _X_ [H] _[Y W]_ [)]


_with matrices P_ 1 _, P_ 2 _, P_ 3 _∈_ C _[r][×][r]_ _such that_


_P_ 1 : = _VX_ [H] _[Y Y]_ [H] _[V]_ _X_ _[⊥]_ _[V]_ _X_ [H] _[⊥]_ _[V][Y W]_ [ (] _[V V][Y W]_ [ )] _[−]_ [1][(Id] _[ −]_ _[µV]_ _X_ [H] _[Y Y]_ [H] _[V][X]_ [)] _[−]_ [1]

_P_ 2 : = _VX_ [H][(] _[Z][ −]_ _[XX]_ [H][ +] _[ Y Y]_ [H][)] _[V][Y W]_ [(] _[V]_ _X_ [H] _[V][Y W]_ [)] _[−]_ [1][(Id] _[ −]_ _[µV]_ _X_ [H] _[Y WW]_ [ H] _[Y]_ [H] _[V][X]_ [)] _[−]_ [1]

_P_ 3 : = Σ [2] _X_ _[V]_ _X_ [H] _[Y W]_ [(Id] _[ −]_ _[µW]_ [ H] _[Y]_ [H] _[V][X]_ _[V]_ _X_ [H] _[Y W]_ [)] _[−]_ [1] _[W]_ [ H] _[Y]_ [H] _[V][X]_


_with_


_∥P_ 1 _∥≤_ 4 _∥Y W_ _∥_ [2] _∥VX_ _⊥_ _VY W ∥_ [2]

_∥P_ 2 _∥≤_ 4 _∥Z −_ _XX_ [H] + _Y Y_ [H] _∥_

_∥P_ 3 _∥≤_ 2 _∥X∥_ [2] _∥Y W_ _∥_ [2] _._


_Moreover, it holds that_


_σmin_ ( _H_ ) _≥_ �1 + _µσmin_ [2] [(] _[X]_ [)] _[ −]_ _[µ][∥][P]_ [1] _[∥−]_ _[µ][∥][P]_ [2] _[∥−]_ _[µ]_ [2] _[∥][P]_ [3] _[∥]_      - _σmin_ ( _VX_ [H] _[Y]_ [ )] �1 _−_ _µσmin_ [2] [(] _[V]_ _X_ [H] _[Y]_ [ )]      - _._


_Proof._ The proof of this Lemma follows from Lemma 9.1 in (Stoger & Soltanolkotabi, 2021) by using an independent matrix¨
_Z_ _∈_ C _[n][×][n]_ instead of the matrix _A_ _[∗]_ _A_ ( _XX_ [H] _−_ _Y Y_ [H] ), omitting the assumption _∥Y ∥≤_ 3 _∥X∥_ and updating respectively
the transformation steps.


**Lemma H.3.** _For a matrix X_ _∈_ C _[n][×][r]_ _, r_ _≤_ _n with its SVD-decomposition X_ = _VX_ Σ _X_ _WX_ [H] _[and some full-rank matrix]_
_Y_ _∈_ C _[n][×][R]_ _and Y_ 1 = (Id + _µZ_ ) _Y_ _consider VX_ [H] _[Y]_ [=] _[ V]_ [ Σ] _[W]_ [ H] _[,][ V]_ _X_ [H] _[Y]_ [1] [=] _[ V]_ [1][Σ][1] _[W]_ [ H] 1 _[, and the following decomposition of][ Y]_
_and Y_ 1


_Y_ = _Y WW_ [H] + _Y W⊥W⊥_ [H] _[.]_

_Y_ 1 = _Y_ 1 _W_ 1 _W_ 1 [H] [+] _[ Y]_ [1] _[W]_ [1] _[,][⊥][W]_ [ H] 1 _,⊥_ _[.]_


_Assume that VX_ [H] _[Y]_ [1] _[W]_ _[is invertible,]_ _[which also implies that][ Y]_ [1] _[W]_ _[is has full-rank,]_ _[and that][ ∥][V]_ _X_ [H] _[⊥]_ _[V][Y]_ [1] _[W][ ∥≤]_ 501 _[and][ µ]_ _[≤]_

min - ~~_√_~~ 13 _[∥][V]_ _X_ [H] _[⊥]_ _[Y W][⊥][∥][−]_ [2] _[,]_ [1] 9 _[∥][X][∥][−]_ [2][�] _and_ _moreover,_ _µ_ _is_ _small_ _enough_ _so_ _that_ 0 _⪯_ Id _−_ _µVX_ [H] _[⊥]_ _[Y WW]_ [ H] _[Y]_ [H] _[V][X]_ _[⊥]_ _[⪯]_ [Id] _[.]_

_Consider two matrices_



min - ~~_√_~~ 1




[H] [1]
3 _[∥][V]_ _X_ _[⊥]_ _[Y W][⊥][∥][−]_ [2] _[,]_ 9



_G_ 1 : = _−VX_ [H] _[⊥]_ _[Y]_ [1] _[W]_ [(] _[V]_ _X_ [H] _[Y]_ [1] _[W]_ [)] _[−]_ [1] _[V]_ _X_ [H] _[Y]_ [1] _[W][⊥][W]_ [ H] _⊥_ _[W]_ [1] _[,][⊥]_

_G_ 2 : = _VX_ [H] _[⊥]_ _[Y]_ [1] _[W][⊥][W]_ [ H] _⊥_ _[W]_ [1] _[,][⊥][.]_


_Then these matrices can be represented as_


_G_ 1 = _µVX_ [H] _[⊥]_ _[V][Y]_ 1 _[W]_ [ (] _[V]_ _X_ [H] _[V][Y]_ 1 _[W]_ [)] _[−]_ [1] _[M]_ [1] _[V]_ _X_ [H] _[⊥]_ _[Y W][⊥][W]_ [ H] _⊥_ _[W]_ [1] _[,][⊥]_


_with M_ 1 := _VX_ [H][(] _[ZV][X]_ _[⊥]_ _[−]_ _[XX]_ [H] _[V][X]_ _[⊥]_ [)] _[ and]_


_G_ 2 = �Id _−_ _µM_ 2 + _µM_ 3) _VX_ [H] _[⊥]_ _[Y W][⊥]_ [(Id] _[ −]_ _[µW]_ _⊥_ [ H] _[Y]_ [H] _[Y W][⊥]_ [)] _[ −]_ _[µ]_ [2][(] _[M]_ [2] _[−]_ _[M]_ [3][)] _[V]_ _X_ [H] _[⊥]_ _[Y W][⊥][W]_ [ H] _⊥_ _[Y]_ [H] _[Y W][⊥]_   - _·_

_· W⊥_ [H] _[W]_ [1] _[,][⊥]_


53


**Implicit Regularization for Tubal Tensors via GD**


_with M_ 2 = _VX_ [H] _[⊥]_ _[Y WW]_ [ H] _[Y]_ [H] _[V][X]_ _[⊥]_ _[and][ M]_ [3] [:=] _[ V]_ _X_ [H] _[⊥]_ [(] _[Z][ −]_ [(] _[XX]_ [H] _[ −]_ _[Y Y]_ [H][))] _[V][X]_ _[⊥]_ _[.]_ _[Moreover, the norm of][ G]_ [1] _[and][ G]_ [2] _[can be]_
_bounded respectively as_


_∥G_ 1 _∥≤_ 2 _µ_ ( _∥VX_ [H] _[⊥]_ _[V][Y W][ ∥∥][Y W]_ _[∥]_ [2][ +] _[ ∥][Z][ −]_ [(] _[XX]_ [H] _[ −]_ _[Y Y]_ [H][)] _[∥]_ [)] _[∥][V]_ _X_ [H] _[⊥]_ _[V][Y]_ 1 _[W][ ∥∥][Y W][⊥][∥][,]_

_∥G_ 2 _∥≤∥Y W⊥∥_ �1 _−_ _µ∥Y W⊥∥_ [2] + _µ∥Z −_ ( _XX_ [H] _−_ _Y Y_ [H] ) _∥_       

+ _µ_ [2][�] _∥Y W_ _∥_ [2] + _∥Z −_ ( _XX_ [H] _−_ _Y Y_ [H] ) _∥_       - _∥Y W⊥∥_ [3] _._


_Proof._ The proof of this Lemma follows from Lemma 9.2 in (Stoger¨ & Soltanolkotabi, 2021) by changing the matrix
_A_ _[∗]_ _A_ ( _XX_ [H] _−_ _Y Y_ [H] ) to the independent matrix _Z_ _∈_ C _[n][×][n]_ and taking into account the respective changes without having
the condition _∥Y ∥≤_ 3 _∥X∥_ .


**Lemma H.4.** _For a matrix X_ _∈_ C _[n][×][r]_ _, r_ _≤_ _n with its SVD-decomposition X_ = _VX_ Σ _X_ _WX_ [H] _[and some full-rank matrix]_
_Y_ _∈_ C _[n][×][R]_ _and Y_ 1 := (Id + _µZ_ ) _Y_ _consider VX_ [H] _[Y]_ [=] _[ V]_ [ Σ] _[W]_ [ H] _[,][ V]_ _X_ [H] _[Y]_ [1] [=] _[ V]_ [1][Σ][1] _[W]_ [ H] 1 _[, and the following decomposition of][ Y]_
_and Y_ 1


_Y_ = _Y WW_ [H] + _Y W⊥W⊥_ [H] _[,]_

_Y_ 1 = _Y_ 1 _W_ 1 _W_ 1 [H] [+] _[ Y]_ [1] _[W]_ [1] _[,][⊥][W]_ [ H] 1 _,⊥_ _[.]_


_Then it holds that_




    _∥W⊥_ [H] _[W]_ [1] _[∥≤]_ _[µ]_ 1 + _µ_ _[∥][Z][∥∥][Y W]_ _[∥]_

_σmin_ ( _VX_ [H] _[Y]_ [ )]




- [H][)] _[∥]_
_∥Y W_ _∥∥Y W⊥∥∥VX_ [H] _[⊥]_ _[V][Y W][ ∥]_ [+] _[ µ]_ _[∥][Z][ −]_ [(] _[XX]_ [H] _[ −]_ _[Y Y]_ _∥Y W⊥∥_ (H.5)

_σmin_ ( _VX_ [H] _[Y]_ [ )]



_Moreover, if for P_ := _Y W⊥W⊥_ [H] _[W]_ [1][(] _[V]_ _Y W_ [H] _[Y WW]_ [ H] _[W]_ [1][)] _[−]_ [1] _[V]_ _Y W_ [H] _[the following applies]_


_∥µZ_ + _P_ + _µZP_ _∥≤_ 1 _,_


_then it holds that_

_∥VX_ [H] _[⊥]_ _[V][Y]_ 1 _[W]_ 1 _[∥≤∥][V]_ _X_ [H] _[⊥]_ _[V][Y W][ ∥]_ �1 _−_ _[µ]_ _min_ [(] _[X]_ [) +] _[ µ][∥][Y W][⊥][∥]_      - + _µ∥Z −_ ( _XX_ [H] _−_ _Y Y_ [H] ) _∥_

2 _[σ]_ [2]

2 _∥W⊥_ [H] _[W]_ [1] _[∥∥][Y W][⊥][∥]_
+ (1 + _µ∥Z∥_ ) (H.6)
_σmin_ ( _W_ [H] _W_ 1) _σmin_ ( _Y W_ )




  - _∥W⊥_ [H] _[W]_ [1] _[∥∥][Y W][⊥][∥]_
+ 57 _µ∥Z∥_ + (1 + _µ∥Z∥_ )
_σmin_ ( _W_ [H] _W_ 1) _σmin_ ( _Y W_ )



2




_Proof._ The proof of inequality (H.5) follows from the first part of the proof of Lemma B.3 in (Stoger & Soltanolkotabi,¨
2021). For this one needs to change the matrix _A_ _[∗]_ _A_ ( _XX_ [H] _−_ _Y Y_ [H] ) in (Stoger & Soltanolkotabi, 2021) to an independent¨
matrix _Z_ _∈_ C _[n][×][n]_ and take into account the above-given decomposition of matrices _Y_ and _Y_ 1 and lack of assumptions on _µ_
and the norm of matrix _Z_ . Inequality (H.6) follows from the proof of Lemma 9.3 in (St¨oger & Soltanolkotabi, 2021).


**I. Random Tubal Tensors**


In this section, we derive bounds on the minimum and maximum singular values as well as the Frobenius norm of a random
tubal tensor with i.i.d. Gaussian random entries. In our analysis of the spectral stage, we applied these lemmas to the small
random initialization.


We start with the following proposition from Rudelson and Vershynin (2009), which bounds the smallest singular value of
an _r × R_ random real Gaussian matrix.

**Proposition I.1** ((Rudelson & Vershynin, 2009)) **.** _Let_ _**G**_ _∈_ R _[r][×][R]_ _with r_ _≤_ _R have i.i.d._ _N_ (0 _,_ 1) _entries._ _Then, for every_
_ϵ >_ 0 _, we have_ _√_ _√_
_σmin_ ( _**G**_ ) _≥_ _ϵ_ ( _R −_ _r −_ 1)



_√_
_R −_



_r −_ 1)



_with probability at least_ 1 _−_ ( _Cϵ_ ) _[R][−][r]_ [+1] _−_ _e_ _[−][cR]_ _._ _The constants C, c >_ 0 _are universal._


54


**Implicit Regularization for Tubal Tensors via GD**


Also, the following proposition from Tao and Vu (2010) bounds the smallest singular value of an _r × r_ random complex
Gaussian matrix.

**Proposition I.2** ((Tao & Vu, 2010)) **.** _Let_ _**G**_ _∈_ R _[r][×][r]_ _have i.i.d._ _CN_ (0 _,_ 1) _entries._ _Then, for every ϵ >_ 0 _, we have_


_ϵ_
_σmin_ ( _**G**_ ) _≥_ ~~_√_~~
_r_


_with probability at least_ 1 _−_ _ϵ_ [2] _._


Using these propositions, we can obtain a bound on the smallest singular value of an _r × R_ random complex Gaussian
matrix, provided that _r_ _≤_ _R_ .

**Lemma I.1.** _Let_ _**G**_ _∈_ C _[r][×][R]_ _with r_ _≤_ _R have i.i.d._ _CN_ (0 _,_ 1) _entries._ _Then, for every ϵ >_ 0 _, we have_



_√_
_ϵ_ (



_σmin_ ( _**G**_ ) _≥_








_√_
_ϵ_ ( _R −_ _[√]_ 2 _r −_ 1) _if R >_ 2 _r_

_ϵ_
~~_√_~~ _if r_ _≤_ _R ≤_ 2 _r_
_r_







_with probability at least_

          1 _−_ ( _Cϵ_ ) _[R][−]_ [2] _[r]_ [+1] _−_ _e_ _[−][cR]_ _if R >_ 2 _r_
1 _−_ _ϵ_ [2] _if r_ _≤_ _R ≤_ 2 _r_ _[.]_


_The constants C, c >_ 0 _are universal._



_Proof._ First, suppose _R_ _>_ 2 _r_ . Let _**G**_ = _**U**_ **Σ** _**V**_ _[H]_ be the SVD of _**G**_ where _**U**_ _∈_ C _[r][×][r]_ and _**V**_ _∈_ C _[R][×][R]_ are unitary and
**Σ** _∈_ R _[r][×][R]_ . Then, the following real 2 _r ×_ 2 _R_ matrix has a real SVD of

�Re _{_ _**G**_ _}_ _−_ Im _{_ _**G**_ _}_       - �Re _{_ _**U**_ _}_ _−_ Im _{_ _**U**_ _}_ �� **Σ** 0 ��Re _{_ _**V**_ _}_ _−_ Im _{_ _**V**_ _}_       - _T_

= _._

Im _{_ _**G**_ _}_ Re _{_ _**G**_ _}_ Im _{_ _**U**_ _}_ Re _{_ _**U**_ _}_ 0 **Σ** Im _{_ _**V**_ _}_ Re _{_ _**V**_ _}_




- �Re _{_ _**U**_ _}_ _−_ Im _{_ _**U**_ _}_
=
Im _{_ _**U**_ _}_ Re _{_ _**U**_ _}_



�� **Σ** 0
0 **Σ**



��Re _{_ _**V**_ _}_ _−_ Im _{_ _**V**_ _}_
Im _{_ _**V**_ _}_ Re _{_ _**V**_ _}_




- _T_
_._



By using the fact that for any _**A**_ _∈_ R _[p][×][q]_ with _p ≤_ _q_, _σ_ min( _**A**_ ) [2] = _**x**_ min _∈_ R _[p]_ _∥_ _**A**_ _[T]_ _**x**_ _∥_ [2] 2 [, we have]
_∥_ _**x**_ _∥_ 2=1



_σ_ min( _**G**_ ) [2] = _σ_ min



��Re _{_ _**G**_ _}_ _−_ Im _{_ _**G**_ _}_ ��2
Im _{_ _**G**_ _}_ Re _{_ _**G**_ _}_







�����



2



= min
_**x**_ _∈_ R [2] _[r]_
_∥_ _**x**_ _∥_ 2=1


= min
_**x**_ _∈_ R [2] _[r]_
_∥_ _**x**_ _∥_ 2=1


_≥_ min
_**x**_ _∈_ R [2] _[r]_
_∥_ _**x**_ _∥_ 2=1



�����




 - 2
��� Re _{_ _**G**_ _}_ _[T]_ Im _{_ _**G**_ _}_ _[T]_ [ �] _**x**_ ���2 [+] _**x**_ [min] _∈_ R [2] _[r]_

_∥_ _**x**_ _∥_ 2=1




Re _{_ _**G**_ _}_ _[T]_ Im _{_ _**G**_ _}_ _[T]_




_−_ Im _{_ _**G**_ _}_ _[T]_ Re _{_ _**G**_ _}_ _[T]_



_**x**_



2




 - 2
���� Re _{_ _**G**_ _}_ _[T]_ Im _{_ _**G**_ _}_ _[T]_ [ �] _**x**_ ���



2



2 - 2

2 [+] ��� _−_ Im _{_ _**G**_ _}_ _[T]_ Re _{_ _**G**_ _}_ _[T]_ [ �] _**x**_ ���2








 - 2
��� Re _{_ _**G**_ _}_ _[T]_ Im _{_ _**G**_ _}_ _[T]_ [ �] _**x**_ ���




 - 2
��� Im _{_ _**G**_ _}_ _[T]_ Re _{_ _**G**_ _}_ _[T]_ [ �] _**x**_ ���2




 - 2
��� Im _{_ _**G**_ _}_ _[T]_ Re _{_ _**G**_ _}_ _[T]_ [ �] _**x**_ ���



�� _−_ Im _{_ _**G**_ _}_ ��2
Re _{_ _**G**_ _}_



= _σ_ min



��Re _{_ _**G**_ _}_ ��2
Im _{_ _**G**_ _}_ + _σ_ min



��2
_,_



= 2 _σ_ min



��Re _{_ _**G**_ _}_
Im _{_ _**G**_ _}_



where the last line follows since reordering the rows of a matrix or flipping the sign of some rows doesn’t change the singular
values.



_√_
Since _**G**_ _∈_ C _[r][×][R]_ has i.i.d. _CN_ (0 _,_ 1) entries,


tion I.1, we have that


_σ_ min( _**G**_ ) _≥_ _σ_ min




- _√_ �Re _{_ _**G**_ _}_ �� _√_
2 _≥_ _ϵ_ (
Im _{_ _**G**_ _}_


55



�Re _{_ _**G**_ _}_ 2 _∈_ R [2] _[r][×][R]_ has i.i.d. _N_ (0 _,_ 1) entries. Therefore, by ProposiIm _{_ _**G**_ _}_



_√_
_R −_



2 _r −_ 1)


**Implicit Regularization for Tubal Tensors via GD**


with probability at least 1 _−_ ( _Cϵ_ ) _[R][−]_ [2] _[r]_ [+1] _−_ _e_ _[−][cR]_, as desired.


Next, suppose _r_ _≤_ _R ≤_ 2 _r_ . Let _**G**_ _r×r_ be an _r × r_ submatrix of _**G**_ . Then,


_σ_ min( _**G**_ ) [2] = _**x**_ min _∈_ C _[r]_ _∥_ _**G**_ _[H]_ _**x**_ _∥_ [2] 2 _[≥]_ _**x**_ min _∈_ C _[r]_ _∥_ _**G**_ _[H]_ _r×r_ _**[x]**_ _[∥]_ [2] 2 [=] _[ σ]_ [min][(] _**[G]**_ _[r][×][r]_ [)][2] _[.]_
_∥_ _**x**_ _∥_ 2=1 _∥_ _**x**_ _∥_ 2=1


Hence, by Proposition I.2, we have
_ϵ_
_σ_ min( _**G**_ ) _≥_ _σ_ min( _**G**_ _r×r_ ) _≥_ ~~_√_~~
_r_


with probability at least 1 _−_ _ϵ_ [2] .


Using the above lemma, we can bound the smallest singular value of an _r × R × k_ tubal tensor.

**Lemma I.2.** _Let_ _**G**_ _∈_ R _[r][×][R][×][k]_ _with r_ _≤_ _R have i.i.d._ _N_ (0 _,_ _R_ [1] [)] _[ entries.]_ _[Then, for every][ ϵ >]_ [ 0] _[, we have]_



_√_
_ϵ_


_√_
_ϵ_



_if r_ _≤_ _R ≤_ 2 _r_
_rR_



_√_
_k_ (



_R −_ _[√]_ 2 _r −_ 1)
~~_√_~~ _if R >_ 2 _r_

_R_



_R −_ _[√]_ 2 _r −_ 1)
~~_√_~~



_σmin_ ( _**G**_ ) _≥_











_ϵ_ _k_

~~_√_~~



_with probability at least_

         1 _−_ _k_ ( _Cϵ_ ) _[R][−]_ [2] _[r]_ [+1] _−_ _ke_ _[−][cR]_ _if R >_ 2 _r_
1 _−_ _kϵ_ [2] _if r_ _≤_ _R ≤_ 2 _r_ _[.]_



_Proof._ Since the entries of _**G**_ are i.i.d. _N_ (0 _,_ [1]



_R_ [1] [)][, the entries of] _**[G]**_ [�] [are i.i.d.] _[CN]_ [(0] _[,]_ _R_ _[k]_




      _R_ _[k]_ [)][.] [Hence, each scaled slice] _Rk_



C _[r][×][R]_ for _j_ = 1 _, . . ., k_ has i.i.d. _CN_ (0 _,_ 1) entries. By Lemma I.1, each scaled slice satisfies



( _j_ )

_Rk_ _**[G]**_ [�] _∈_












_√_
_ϵ_ (



_√_
_ϵ_ ( _R −_ _[√]_ 2 _r −_ 1) if _R >_ 2 _r_

_ϵ_
~~_√_~~ if _r_ _≤_ _R ≤_ 2 _r_
_r_



_σ_ min



�� ( _j_ ) [�]
_Rk_ _**[G]**_ [�] _≥_



with probability at least

          1 _−_ ( _Cϵ_ ) _[R][−]_ [2] _[r]_ [+1] _−_ _e_ _[−][cR]_ if _R >_ 2 _r_
1 _−_ _ϵ_ [2] if _r_ _≤_ _R ≤_ 2 _r_ _[.]_


Then, by taking a union bound, we have that



_R −_ _[√]_ 2 _r −_ 1)
~~_√_~~ if _R >_ 2 _r_

_R_



_√_
_ϵ_


_√_
_ϵ_



if _r_ _≤_ _R ≤_ 2 _r_
_rR_



_√_
_k_ (




      - ( _j_ ) [�]

_σ_ min( _**G**_ ) = min _**G**_ - _≥_
1 _≤j≤k_ _[σ]_ [min]











_ϵ_ _k_

~~_√_~~



with probability at least

         1 _−_ _k_ ( _Cϵ_ ) _[R][−]_ [2] _[r]_ [+1] _−_ _ke_ _[−][cR]_ if _R >_ 2 _r_
1 _−_ _kϵ_ [2] if _r_ _≤_ _R ≤_ 2 _r_ _[.]_


The following proposition bounds the operator norm of an _r × R_ random Gaussian matrix.

**Proposition** **I.3** ((Vershynin, 2018)) **.** _Let_ _**U**_ _∈_ C _[n][×][R]_ _have_ _i.i.d._ _CN_ (0 _,_ 1) _entries._ _Then,_ _with_ _probability_ _at_ _least_
1 _−_ _O_ ( _e_ _[−][c]_ [ max] _[{][n,R][}]_ ) _, we have_
_∥_ _**U**_ _∥_ ≲ �max _{n, R}._


Using the above proposition, we can bound the norm of an _n × R × k_ random Gaussian tubal tensor.


56


**Implicit Regularization for Tubal Tensors via GD**


**Lemma I.3.** _Let_ _**U**_ _∈_ R _[n][×][R][×][k]_ _have i.i.d._ _N_ (0 _,_ _R_ [1] [)] _[ entries.]_ _[Then, with probability at least]_ [ 1] _[ −]_ _[O]_ [(] _[ke][−][c]_ [ max] _[{][n,R][}]_ [)] _[, we have]_



_∥_ _**U**_ _∥_ ≲




~~�~~ _k_ max _{n, R}_

_._
_R_



_Proof._ Since the entries of _**U**_ are i.i.d. _N_ (0 _,_ [1]



_R_ [1] [)][, the entries of] _**[U]**_ [�] [are i.i.d.] _[CN]_ [(0] _[,]_ _R_ _[k]_




      _R_ _[k]_ [)][.] [Hence, each scaled slice] _Rk_



C _[r][×][R]_ for _j_ = 1 _, . . ., k_ has i.i.d _CN_ (0 _,_ 1) entries. By Proposition I.3, each scaled slice satisfies
���� ~~�~~ _Rk_ _**[U]**_ [�] ( _j_ ) [��] �� ≲ ~~�~~ max _{n, R}_



( _j_ )

_Rk_ _**[U]**_ [�] _∈_



_Rk_ _**[U]**_ [�] ( _j_ ) [��] �� ≲ ~~�~~ max _{n, R}_




~~�~~
_R_



with probability at least 1 _−_ _O_ ( _e_ _[−][c]_ [ max] _[{][n,R][}]_ ). Then, by taking a union bound, we have that



_∥_ _**U**_ _∥_ = max
1 _≤j≤k_


with probability at least 1 _−_ _O_ ( _ke_ _[−][c]_ [ max] _[{][n,R][}]_ ).




  
( _j_ ) [��] _k_ max _{n, R}_
_**U**_ ≲
����� �� _R_



**Lemma I.4.** _Let_ _**U**_ _∈_ R _[n][×][R][×][k]_ _have i.i.d._ _N_ (0 _,_ _R_ [1] [)] _[ entries.]_ _[Then, for any fixed]_ _**[ V]**_ [1] _[∈]_ [R] _[n][×]_ [1] _[×][k]_ _[with][ ∥]_ _**[V]**_ [1] _[∥]_ [= 1] _[, we have]_



_√_
_∥_ _**U**_ _[⊤]_ _∗_ _**V**_ 1 _∥F_ _≍_ _k_



_with probability at least_ 1 _−_ _O_ ( _ke_ _[−][cR]_ ) _._



_Proof._ Since the entries of _**U**_ are i.i.d. _N_ (0 _,_ [1]



_R_ [1] [)][,] [the] [entries] [of] _**[U]**_ [�] [are] [i.i.d.] _[CN]_ [(0] _[,]_ _R_ _[k]_



_⊤_
_R_ _[k]_ [)][,] [and] [thus,] [the] [entries] [of] _**[U]**_ [�] are



( _j_ )
_R_ _[k]_ [)][.] [Then, for each slice] _[ j]_ [= 1] _[, . . ., k]_ [, each entry of the matrix-vector product][ �] _**U**_ _[⊤]_ [(] _[j]_ [)][ �] _**V**_ 1 _∈_ C _[R]_ is i.i.d.



also i.i.d. _CN_ (0 _,_ _[k]_



( _j_ )
_R_ _[k]_ _[∥]_ _**[V]**_ [�] 1 _[∥]_ _F_ [2] [)][.] [Hence, the quantity]



_CN_ (0 _,_ _[k]_



���




2



2 _R_


_k_



( _j_ )
��� _**U**_ - _[⊤]_ [(] _[j]_ [)][ �] _**V**_ 1




_F_

- ( _j_ )�2
_**V**_ 1
���� ���




���



2



_F_



has a _χ_ [2] (2 _R_ ) distribution. It follows that

( _j_ )
_**U**_              - _[⊤]_ [(] _[j]_ [)][ �] _**V**_ 1
����



( _j_ )
_≍_ _k_ _**V**_ 1
_F_ �����



2



����



2



����



_F_



holds with probability at least 1 _−_ _O_ ( _e_ _[−][cR]_ ). By taking a union bound over all _j_ = 1 _, . . ., k_, we get that



2



_≍_

_F_



2



2
= _**V**_ 1
_F_ ���� ��� _F_



_k_



2
_**U**_ _⊤_ _∗_ _**V**_ 1
��� ���



_k_



2 [1]

_F_ [=] _k_



_k_



_j_ =1



( _j_ )
_**V**_ 1
�����



����



_F_ [=] _[ k,]_
_F_ [=] _[ k][ ∥]_ _**[V]**_ [1] _[∥]_ [2]



2
_**U**_ _[⊤]_ _⊙_ _**V**_ [�] 1
���� ���



2 [1]

_F_ [=] _k_



_k_



_j_ =1



( _j_ )
_**U**_ - _[⊤]_ [(] _[j]_ [)][ �] _**V**_ 1
����



����



_√_
i.e., _∥_ _**U**_ _[⊤]_ _∗_ _**V**_ 1 _∥F_ _≍_



_k_ with probability at least 1 _−_ _O_ ( _ke_ _[−][cR]_ ).


57


