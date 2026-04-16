# **In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**

**Matthew Smart** [1] **Alberto Bietti** [2] **Anirvan M. Sengupta** [2 3 4]



**Abstract**


We introduce in-context denoising, a task that
refines the connection between attention-based architectures and dense associative memory (DAM)
networks, also known as modern Hopfield networks. Using a Bayesian framework, we show theoretically and empirically that certain restricted
denoising problems can be solved optimally even
by a single-layer transformer. We demonstrate
that a trained attention layer processes each denoising prompt by performing a single gradient
descent update on a context-aware DAM energy
landscape, where context tokens serve as associative memories and the query token acts as an
initial state. This one-step update yields better
solutions than exact retrieval of either a context
token or a spurious local minimum, providing a
concrete example of DAM networks extending
beyond the standard retrieval paradigm. Overall,
this work solidifies the link between associative
memory and attention mechanisms first identified
by Ramsauer et al., and demonstrates the relevance of associative memory models in the study
of in-context learning.


**1. Introduction**


The transformer architecture (Vaswani et al., 2017) has
achieved remarkable success across diverse domains, from
natural language processing (Devlin et al., 2019; Brown
et al., 2020; Touvron et al., 2023) to computer vision (Dosovitskiy et al., 2021). Despite their practical success, un

1 Center for Computational Biology, Flatiron Institute, New
York, NY, USA [2] Center for Computational Mathematics, Flatiron Institute, New York, NY, USA [3] Center for Computational Quantum Physics, Flatiron Institute, New York, NY,
USA [4] Department of Physics and Astronomy, Rutgers University, Piscataway, NJ, USA. Correspondence to: Matthew Smart
_<_ msmart@flatironinstitute.org _>_, Anirvan M. Sengupta _<_ anirvans.physics@gmail.com _>_ .


_Proceedings_ _of_ _the_ _42_ _[nd]_ _International_ _Conference_ _on_ _Machine_
_Learning_, Vancouver, Canada. PMLR 267, 2025. Copyright 2025
by the author(s).



derstanding the mechanisms behind transformer-based networks remains an open challenge. This challenge is exacerbated by the growing scale and complexity of modern large
networks. Toward addressing this, researchers studying simplified architectures have identified connections between the
attention operation that is central to transformers and associative memory models (Ramsauer et al., 2021), providing
not only an avenue for understanding how such architectures
encode and retrieve information but also potentially ways to
improve them further.


The most celebrated model for associative memories in systems neuroscience is the so-called Hopfield model (Amari,
1972; Nakano, 1972; Little, 1974; Hopfield, 1982). This
model has a capacity to store “memories” (stable fixed
points of a recurrent update rule) proportional to the number
of nodes (Hopfield, 1982; Amit et al., 1985). In the last
decade, new energy functions (Krotov & Hopfield, 2016;
Demircigil et al., 2017) were proposed for dense associative
memories with much higher capacities. These energy functions are often referred to as modern Hopfield models. Ramsauer et al. (2021) pointed out the similarity between the
one-step update rule of a certain modern Hopfield network
(Demircigil et al., 2017) and the softmax attention layer of
transformers, generating interest in the statistical physics
and systems neuroscience communities (Krotov & Hopfield,
2021; Krotov, 2023; Lucibello & Mezard´, 2024; Millidge
et al., 2022). Recent work has extended this concept to
improve retrieval by incorporating sparsity (Hu et al., 2023;
Wu et al., 2024b; Santos et al., 2024; Wu et al., 2024a), while
others have leveraged associative memory principles to design new energy-based transformer architectures (Hoover
et al., 2023). However, these extensions and the foundational construction in Ramsauer et al. (2021) primarily focus
on the specific task of exact retrieval (converging to a fixed
point), while in practice transformers may tackle many other
tasks.


To explore this connection beyond retrieval, we introduce _in-context_ _denoising_, a task that bridges the behavior of trained transformers and associative memory networks through the lens of in-context learning (ICL). In
standard ICL, a sequence model is trained to infer an unknown function _g_ from contextual examples, predicting



1


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**



_g_ ( _XL_ +1) given a sequence of input-output pairs _E_ =
(( _X_ 1 _, g_ ( _X_ 1)) _, ...,_ ( _XL, g_ ( _XL_ )) _,_ ( _XL_ +1 _, −_ )). Crucially, _g_
is implied solely through the context and differs across
prompts – performant models are therefore said to “learn
_g_ ( _x_ ) in context”. While ICL has been extensively studied in supervised settings (Garg et al., 2022; Zhang et al.,
2024; Akyurek¨ et al., 2023; Reddy, 2024), recent work
suggests that transformers may internally emulate gradient
descent over a context-specific loss function during inference (Von Oswald et al., 2023; Dai et al., 2023; Ahn et al.,
2023). This general perspective aligns with our findings.


In this work, we generalize ICL to an unsupervised setting
where the prompt consists of _L_ samples from a random distribution and the query is a noise-corrupted sample from the
same distribution. This shift allows us to probe how trained
transformers internally approximate Bayes optimal inference, while deepening the connection to associative memory models which are prototypical denoisers. By setting up
this problem in this way, we also attempt to answer a few
questions. One concerns the memorization-generalization
dilemma in denoising: a Hopfield model’s success is usually
measured by successful memory recovery, while in-context
learning may have to solve a completely new problem. Another question has to do with the number of iterations of
the corresponding Hopfield model: why does the Ramsauer
et al. (2021) correspondence involve only one iteration of
Hopfield energy minimization and not many?


**In** **summary,** **our** **contributions** **are** **as** **follows:** In Section 2, we introduce in-context denoising as a framework for
understanding how transformers perform implicit inference
beyond memory retrieval. In Section 3, we establish that
single-layer transformers with one attention head are expressive enough to optimally solve certain denoising problems.
We then empirically demonstrate that standard training from
random weights can recover the Bayes optimal predictors.
The trained attention layers are mapped back to dense associative memory networks in Section 4. Our results refine the
general connection pointed out in previous work, offer new
mechanistic insights into attention, and provide a concrete
example of dense associative memory networks extending
beyond the standard memory retrieval paradigm to solve a
novel in-context learning task.


**2. Problem formulation:** **In-context denoising**


In this section, we describe our general setup. Recurring
common notation is described in Appendix A.1.


**2.1. Setup**


Each task corresponds to a distribution _D_ over the probaiid
bility distribution of data: _pX_ _∼_ _D_ . Let _X_ 1 _, · · ·_ _, XL_ +1 _∼_
_pX_, define the sampling of the tokens. Let the noise cor


ruption be defined by _X_ [˜] _∼_ _p_ noise( _·|XL_ +1). The random
sequence _E_ = ( _X_ 1 _, X_ 2 _, ..., XL,_ _X_ [˜] ) are given as “context”
(input) to a sequence model _F_ ( _·_ ; _θ_ ) which outputs an estimate _X_ [ˆ] _L_ +1 of the original ( _L_ + 1)-th token . The task is
to minimize the expected loss E[ _l_ ( _X_ [ˆ] _L_ +1 _, XL_ +1)] for some
loss function _l_ ( _·, ·_ ). Namely, our problem is to find


min _θ_ E _pX_ _∼D,X_ 1: _L_ +1 _∼pLX_ +1 _,X_ [˜] _∼p_ noise( _·|XL_ +1) [[] _[l]_ [(] _[F]_ [(] _[E, θ]_ [)] _[, X][L]_ [+1][)]] _[.]_

(1)

In practice, we choose _X_ [˜] = _XL_ +1 + _Z_, a pure token corrupted by the addition of isotropic Gaussian noise _Z_ _∼_
_N_ (0 _, σZ_ [2] _[I][n]_ [)][, and our objective function to minimize is the]
mean squared error (MSE) E[ _||X_ [ˆ] _L_ +1 _−_ _XL_ +1 _||_ [2] ].


In the following subsection, we explain the pure token distributions for three specific tasks. These tasks are of course
structured so that a one-layer transformer has the expressivity to capture a solution, which, as _L_ _→∞_, provides
an optimal solution, in some sense. To that end, we derive
Bayes optimal estimators for each of the three tasks, under
the assumption that we know the original distribution _pX_ of
pure tokens. In Section 3, we use these estimators as baselines to evaluate the performance of the denoiser _f_ ( _E, θ_ )
based on a one-layer transformer trained on finite datasets.


**2.2. Task-specific token distributions**


We consider three elementary in-context denoising tasks,
where the data (vectors in R _[n]_ ) comes from:


1. Linear manifolds ( _d_ -dimensional subspaces)


2. Nonlinear manifolds ( _d_ -spheres)


3. Small noise Gaussian mixtures (clusters) where the
component means have fixed norm


Below we describe the task-specific distributions _pX_ and the
process for sampling tokens _{xt}_ . The same corruption process applies to all cases: _X_ [˜] = _XL_ +1 + _Z, Z_ _∼N_ (0 _, σZ_ [2] _[I][n]_ [)][.]


2.2.1. CASE 1 - LINEAR MANIFOLDS


A given training prompt consists of pure tokens sampled
from a random _d_ -dimensional subspace _S_ of R _[n]_ .


  - Let _P_ be the orthogonal projection operator to a random _d_ -dim subspace _S_ of R _[n]_, sampled according to
the uniform measure, induced by the Haar measure on
the coset space _O_ ( _n_ ) _/O_ ( _n −_ _d_ ) _× O_ ( _d_ ), on the Grassmanian _G_ ( _d, n_ ), the manifold of all _d_ -dimensional subspaces of R _[n]_ .


  - Let _Y_ _∼N_ (0 _, σ_ 0 [2] _[I][n]_ [)] [and] [define] _[X]_ [=] _[PY]_ [ ;] [we]
use this procedure to construct the starting sequences
( _X_ 1 _, ..., XL_ +1) of _L_ + 1 independent tokens.



2


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**



(a) Problem formulation (b)



Prompt: Pure tokens from a data distribution and a single corrupted example

(prompts are randomly constructed from a pre-specified task distribution)





Query


Target


Prediction


|Col1|Col2|
|---|---|
||target<br>query|



Case 1:
Linear manifolds



Case 2:
Nonlinear manifolds



Case 3:
Gaussian mixtures


|sample a task from<br>corruption of<br>final token|Col2|a task distribution<br>sample context<br>tokens|
|---|---|---|
|corruption of<br>ﬁnal token<br>sample a task from|||



_Figure 1._ (a) Problem formulation for a general in-context denoising task. (b) The three denoising tasks considered here include instances
of linear and non-linear manifolds as well as Gaussian mixtures. In each case, the task embedding _E_ [(] _[i]_ [)] consists of a sequence of pure
tokens from the data distribution _p_ [(] _X_ _[i]_ [)] _[∼]_ _[D]_ [ where] _[ D]_ [ denotes the task distribution, along with a single query token that has been corrupted]
by Gaussian noise. The objective is to predict the target (i.e. _denoise_ the query) given information contained only in the prompt.



We thus have _pX_ = _N_ (0 _, σ_ 0 [2] _[P]_ [)][, with the Haar distribution]
of _P_ characterizing the task ensemble associated with _D_ .


2.2.2. CASE 2 - NONLINEAR MANIFOLDS


We focus on the case of _d_ -dimensional spheres of fixed
radius _R_ centered at the origin in R _[n]_ .


  - Choose a random _d_ +1-dimensional subspace _V_ of R _[n]_,
sampled according to the uniform measure, as before,
on the Grassmanian _G_ ( _d_ + 1 _, n_ ). The choice of this
random subspace generates the distribution of tasks _D_ .


  - Inside _V_, sample uniformly from the radius _R_ sphere
(once more, a Haar induced measure on a coset space
_O_ ( _d_ + 1) _/O_ ( _d_ )). We use this procedure to construct
input sequences _X_ 1: _L_ +1 = ( _x_ 1 _, ..., xL_ +1) of _L_ + 1
independent tokens.


In practice, we uniformly sample points with fixed norm in
R _[d]_ and embed them in R _[n]_ by concatenating zeros. We then
rotate the points by selecting a random orthogonal matrix
_Q ∈_ R _[n][×][n]_ .


2.2.3. CASE 3 - GAUSSIAN MIXTURES (CLUSTERING)


Pure tokens are sampled from a weighted mixture of
isotropic Gaussians in _n_ -dimensions, _{wa,_ ( _µa, σa_ [2][)] _[}][K]_ _a_ =1 [.]
The density is



zero. The distribution of tasks _D_, is decided by the choice
of _{µa}_ _[K]_ _a_ =1 [.]


For our ideal case, we will consider the limit that the variances go to zero. In that case, the density is simply



_pX_ 0( _x_ ) =



_K_

- _waδ_ ( _x −_ _µa_ ) _._


_a_ =1



_pX_ ( _x_ ) =



_K_

- _waCae_ _[−∥][x][−][µ][a][∥]_ [2] _[/]_ [2] _[σ]_ _a_ [2] _,_


_a_ =1



where _Ca_ = (2 _πσa_ [2][)] _[−][n/]_ [2] [are normalizing constants.] [The]
_µa_ are independently chosen from a uniform distribution on
the radius _R_ sphere of dimension _n −_ 1, centered around



**2.3. Bayes optimal denoising baselines for each case**


The first _L_ tokens in _E_ are “pure samples” from _p_ that
should provide information about the distribution for our
denoising task. Our performance is expected to be no better
than that of the best method, in the case that the token distribution and also the corrupting process are exactly known.
This is where the Bayesian optimal baseline comes in. As
is well-known, the Bayes optimal predictor of a quantity is
given by the posterior mean. We use that fact to compute
the Bayes optimal loss.


In particular, we seek a function _f_ : R _[n]_ _→_ R _[n]_ such that

   E _X,X_ ˜ _∥X_ _−_ _f_ ( _X_ [˜] ) _∥_ [2][�] is minimized. Since the perturba
tion _Z_ is Gaussian, the posterior distribution of _X_, given _X_ [˜]
is

_pX|X_ ˜ [(] _[x][ |]_ _[x]_ [˜][) =] _[ C]_ [(˜] _[x]_ [)] _[p][X]_ [(] _[x]_ [)] _[e][−∥][x][−][x]_ [˜] _[∥]_ [2] _[/]_ [2] _[σ]_ _Z_ [2] _,_


where _C_ (˜ _x_ ) is a normalizing factor (see Appendix A.2 for
more explanation). The following proposition sets up a
baseline to which we expect to compare our results as _L →_
_∞_ . The proof is in Appendix B.1.


**Proposition 1.** _For each task, specified by the input distri-_
_bution pX_ _, and the noise model pX_ ˜ _|X_ _[,]_


    -     -     E _X,X_ ˜ _∥X_ _−_ _f_ ( _X_ [˜] ) _∥_ [2][�] _≥_ E _X_ ˜ Tr Cov( _X_ _|_ _X_ [˜] ) _._ (2)



3


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**



_This lower bound is met when f_ ( _X_ [˜] ) = E[ _X_ _|_ _X_ [˜] ] _._


Thus, the Bayes optimal denoiser is the posterior expectation
for _X_ given _X_ [˜] . The expected loss is found by computing
the posterior sum of variances.


These optimal denoisers can be computed analytically for
both the linear and nonlinear manifold cases (given the
variances and dimensionalities). In the Gaussian mixture
(clustering) case, it depends on the choice of the centroids
which then needs to be averaged over.


**Linear case.** For the linear denoising task, pure samples
_X_ are drawn from an isotropic Gaussian in a restricted
subspace. The following result provides the Bayes optimal
predictor in this case, the proof of which is in Appendix
C.1.

**Proposition 2.** _For pX_ _corresponding to Subsection 2.2.1,_
_the Bayes optimal answer is_

_fopt_ ( _X_ [˜] ) = E[ _X|X_ [˜] ] = _σ_ 0 [2] _P_ _X,_ [˜] (3)
_σ_ 0 [2] [+] _[ σ]_ _Z_ [2]


_and the expected loss is_


    E _∥P_ _X_ [˜] _−_ _XL_ +1 _∥_ [2][�] = _dσ_ 0 [2] _[σ]_ _Z_ [2] _[/]_ [(] _[σ]_ 0 [2] [+] _[ σ]_ _Z_ [2] [)] _[.]_ (4)


Projection


Projection (shrunk)


_Figure 2._ Baseline estimators for the case of random linear manifolds with projection operator _P_ [(] _[i]_ [)] .


**Manifold case.** In the nonlinear manifold denoising problem, we focus on the case of lower dimensional spheres _S_
(e.g. the circle _S_ [1] _⊂_ R [2] ). For such manifolds, the Bayes
optimal answer is given by the following proposition.

**Proposition 3.** _For pX_ _defined as in Subsection 2.2.2, with_
_P_ _being the orthogonal projection operator to V, the d_ + 1
_dimensional_ _linear_ _subspace,_ _with_ _R_ _being_ _the_ _radius_ _of_
_sphere S, the Bayes optimal answer is_


_fopt_ ( _X_ [˜] ) = E[ _X_ _|_ _X_ [˜] ]



_where_ _X_ [˜] _∥_ = _P_ _X_ [˜] _and Iν_ _is the modified Bessel function of_
_the first kind._


**Clustering case.** For clustering with isotropic Gaussian
mixtures _{wa,_ ( _µa, σa_ [2][)] _[}][p]_ _a_ =1 [, the Bayes optimal predictors]
for some important special cases are as follows. See Appendix C.3 for the general case.


**Proposition 4.** _For general isotropic Gaussian model with_
_σa_ = _σ_ 0 _, ||µa||_ = _R for all a_ = 1 _, . . ., K._


_fopt_ ( _X_ [˜] ) = E[ _X|X_ [˜] ]



= _σ_ 0 [2] _X_ ˜ + _σZ_ [2]
_σ_ 0 [2] [+] _[ σ]_ _Z_ [2] _σ_ 0 [2] [+] _[ σ]_ _Z_ [2]


_If σ_ 0 _→_ 0 _,_



_a_ _[w][a][e][⟨][µ][a][,]_ _X_ [˜] _⟩/_ ( _σ_ 0 [2][+] _[σ]_ _Z_ [2] [)] _µa_

- _[⟨][µ][a][,]_ _X_ [˜] _⟩/_ ( _σ_ [2][+] _[σ]_ [2] [)]







_._
_X_ [˜] _⟩/_ ( _σ_ 0 [2][+] _[σ]_ _Z_ [2] [)]
_a_ _[w][a][e][⟨][µ][a][,]_



(7)



_a_ _[w][a][e][⟨][µ][a][,]_ _X_ [˜] _⟩/σZ_ [2] _µa_

- _[⟨][µ][a][,]_ _X_ [˜] _⟩/σ_ [2]




     _fopt_ ( _X_ [˜] ) = E[ _X_ _|_ _X_ [˜] ] =



_._ (8)
_a_ _[w][a][e][⟨][µ][a][,]_ _X_ [˜] _⟩/σZ_ [2]



In all three cases, we notice similarities between the form
of the Bayes optimal predictor, and attention operations in
transformers, a connection which we explore below.


**3. In-context denoising with one-layer**
**transformers – Empirical results**


In this section, we provide simple constructions of one-layer
transformers that approximate (and under certain conditions,
exactly match) the Bayes optimal predictors above.


iid
**Input:** Let _p_ [(1)] _X_ _[, . . ., p]_ [(] _X_ _[N]_ [)] _∼_ _D_, be distributions sampled
for one of the tasks. For each distribution _p_ [(] _X_ _[i]_ [)][, we sample]
_E_ [(] _[i]_ [)] := ( _X_ 1 [(] _[i]_ [)] _[, . . ., X]_ _L_ [(] _[i]_ [)] _[,]_ _[X]_ [˜] [(] _[i]_ [)][)] [taking] [value] [in] [R] _[n][×]_ [(] _[L]_ [+1)]

be an input to a sequence model. We also retain the true
( _L_ + 1)-th token _XL_ [(] _[i]_ +1 [)] [for each] _[ i]_ [.]


**Objective:** Given an input sequence _E_ [(] _[i]_ [)], return the
uncorrupted final token _XL_ [(] _[i]_ +1 [)] [.] We consider the meansquared error loss over a collection of _N_ training pairs,
_{E_ [(] _[i]_ [)] _, XL_ [(] _[i]_ +1 [)] _[}]_ _i_ _[N]_ =1 [,]



_C_ ( _θ_ ) =



_N_

- _∥F_ ( _E_ [(] _[i]_ [)] _, θ_ ) _−_ _x_ [(] _L_ _[i]_ +1 [)] _[∥]_ [2] _[,]_ (9)

_i_ =1



=




- _e_ _[⟨][x,]_ _X_ [˜] _∥⟩/σZ_ [2] _x dSx_
(5)

 - _e_ _[⟨][x,]_ _X_ [˜] _∥⟩/σZ_ [2] _dSx_





_X_ ˜ _∥_
_R_ _,_ (6)

_∥X_ [˜] _∥∥_



where _F_ ( _E_ [(] _[i]_ [)] _, θ_ ) denotes the parametrized function predicting the target final token based on input sequence _E_ [(] _[i]_ [)] .


**3.1. One-layer transformer and the attention between**
**the query and pure tokens**


To motivate our choice of architecture, let us start by discussing the linear case.



_I_ _d_ +1
= 2

_I_ _d−_ 1

2




- _R_ _[∥]_ _X_ [˜] _∥∥_
_σZ_ [2]

- _R_ _[∥]_ _X_ [˜] _∥∥_
_σZ_ [2]



4


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**



There we have _f_ opt( _X_ [˜] ) = _σ_ 0 [2] _σ_ [+] 0 [2] _[σ]_ _Z_ [2] _[P]_ _[X]_ [˜] [.] [Note] [that,] [by] [the]

strong law of large numbers, _P_ [ˆ] = _σ_ 0 [2] 1 _[L]_ - _Lt_ =1 _[X][t][X]_ _t_ _[T]_ [is] [a]
random matrix that almost surely converges component-bycomponent to the orthogonal projection _P_ as _L →∞_, since,
for each _t_, _XtXt_ _[T]_ [has the expectation] _[ σ]_ 0 [2] _[P]_ [and that] _[ X][t]_ [is]
a Gaussian random variable with zero mean and a finite
covariance matrix. So we could propose



The proof of the theorem is in Appendix D.3. See Appendix E, particularly Proposition 6, for consideration of
convergence rates. Note that the condition of _pX_ being
supported on a sphere is not artificial as, in many practical
transformers, pre-norm with RMSNorm gives you inputs on
the sphere, up to learned diagonal multipliers.


Note that the natural form of attention that is suggested
by our formulation of in-context denoising would involve
Gaussian kernels:



(14)
The relation between softmax attention and the Gaussian
kernel has been noted in (Choromanski et al., 2021; Ambrogioni, 2024) and a Gaussian kernel-based attention is
implemented in (Chen et al., 2021). A related Hopfield
energy, with _WK_, _WQ_, and _WP V_ proportional to identity
matrices, is proposed in (Hoover et al., 2024a).


For the linear case, we use linear attention, but that may
not be essential. Informally speaking, the softmax attention model has the capacity to subsume the linear attention
model.

**Proposition 3.2.** _As ϵ →_ 0 _,_



_f_ ( _X_ [˜] ) = _σ_ 0 [2] _P_ ˆ _X_ ˜ = 1
_σ_ 0 [2] [+] _[ σ]_ _Z_ [2] ( _σ_ 0 [2] [+] _[ σ]_ _Z_ [2] [)] _[L]_



_L_

- _Xt⟨Xt,_ _X_ [˜] _⟩._


_t_ =1

(10)



_t_ _[W][P V][ X][t][e][−]_ [1] 2




    _X_ ˆ = _FG_ ( _E, θ_ ) :=




_[W][P V][ X][t][e][−]_ [1] 2 _[||][W][K]_ _[X][t][−][W][Q]_ _X_ [˜] _||_ [2]

- _[−]_ [1] _[||][W][K]_ _[X][t][−][W][Q]_ _X_ [˜] _||_ [2]



_t_ _[e][−]_ 2 [1]



_._

[1] 2 _[||][W][K]_ _[X][t][−][W][Q]_ _X_ [˜] _||_ [2]



We now consider a simplified one-layer linear transformer
(see Appendices D.1 and D.2 for more detailed discussions)
which still has sufficient expressive power to capture our
finite sample approximation to the Bayes optimal answer.
We define

_X_ ˆ = _F_ Lin( _E, θ_ ) := [1] 1: _L_ _[W][KQ][X]_ [˜] (11)

_L_ _[W][P V][ X]_ [1:] _[L][X]_ _[T]_

taking values in R _[n]_, where _X_ 1: _L_ := [ _X_ 1 _, . . ., XL_ ] taking
values in R _[n][×][L]_, with learnable weights _WKQ, WP V_ _∈_
R _[n][×][n]_ abbreviated by _θ_ . Note that, when _WP V_ =
_αIn, WKQ_ = _βIn_, and _αβ_ = _σ_ 0 [2][+] 1 _[σ]_ _Z_ [2] [,] _[ F]_ [(] _[E, θ]_ [)][ should ap-]

proximate the Bayes optimal answer _f_ opt( _X_ [˜] ) as _L_ _→∞_ .
For a detailed discussion of the convergence rate, see Appendix E, in general, and Proposition 5, in particular.


Similarly, we could argue that the second two problems,
the _d_ -dimesional spheres and the _σ_ 0 _→_ 0 zero limit of the
Gaussian mixtures could be addressed by softmax attention


_X_ ˆ = _F_ ( _E, θ_ ) := _WP V X_ 1: _L_ softmax( _X_ 1: _[T]_ _L_ _[W][KQ][X]_ [˜] [)] [(12)]


taking values in R _[n]_ . The function softmax( _z_ ) :=
1

- _ni_ =1 _[e][z][i]_ [(] _[e][z]_ [1] _[, . . ., e][z][n]_ [)] _[T]_ _[∈]_ [R] _[n]_ [is applied column-wise.]

For both problems, namely the spheres and the _σ_ 0 _→_ 0
Gaussian mixtures, we could have _WP V_ = _αIn, WKQ_ =
_βIn_ with _α_ = 1 _, β_ = 1 _/σZ_ [2] [providing] [Bayes] [optimal] [an-]
swers as _L →∞_ .


In fact, we could make a more general statement about
distributions _pX_ where the norm of _X_ is fixed.

**Theorem 3.1.** _If we have a task distribution D so that the_
_support of each pX_ _is the subset of some sphere, centered_
_around the origin, with a pX_ _-dependent radius R, then the_
_function_


      - _L_ _Z_
_t_ =1 _[X][t][e][⟨][X][t][,][x]_ [˜] _[⟩][/σ]_ [2]
_F_ (( _{Xt}_ _[L]_ _t_ =1 _[,]_ [ ˜] _[x]_ [)] _[, θ][∗]_ [) =]  - _Lt_ =1 _[e][⟨][X][t][,][x]_ [˜] _[⟩][/σ]_ _Z_ [2] (13)


_converges almost surely to the Bayes optimal answer fopt_ (˜ _x_ )
_for_ _all_ _x_ ˜ _∈_ R _[n]_ _,_ _as_ _L_ _→∞._ _The_ _optimal_ _parameter_ _θ_ _[∗]_

_refers to WP V_ = _In, WKQ_ = _σ_ 1 _Z_ [2] _[I][n][.]_




 - - 1 - [�]
_F_ _E,_ = [1] _[X]_ [¯]
_ϵ_ _[W][P V][, ϵW][KQ]_ _ϵ_ _[W][P V]_



+ [1]

_L_ _[W][P V]_



_L_

- _Xt_ ( _Xt −_ _X_ [¯] ) _[T]_ _WKQX_ [˜] + _O_ ( _ϵ_ ) _,_ (15)


_t_ =1



_where_ _X_ [¯] = _L_ [1] - _Lt_ =1 _[X][t][ is the empirical mean.]_



See Appendix F for the details of small _WKQ_ expansion
and Appendix F.1 for the proof of Proposition 3.2.


For case 1, note that E[ _Xt_ ] = 0 and covariance of _Xt_ is
finite, _E_ [ _X_ [¯] ] = 0, and _E_ [ _||X_ [¯] _||_ [2] ] = _O_ ( _L_ [1] [)][,] [allowing] [us] [to]

drop _X_ [¯] as _L_ _→∞_ . If, in addition, _ϵ_ is small, only the
second term survives. Thus, _F_ - _E,_ ( [1] _ϵ_ _[W][P V][, ϵW][KQ]_ [)] - starts

to approximate _F_ Lin� _E,_ ( _WP V, WKQ_ )� when _L_ is large
_√_
and _ϵ_ is small, with _ϵ_ _L_ large. We therefore could use the

softmax model for all three cases.


**3.2. Case 1 – Linear manifolds**


The Bayes optimal predictor for the linear denoising task
from Section 2.3 suggests that the linear attention weights
should be scaled identity matrices with their product satisfying _αβ_ = _σ_ 0 [2][+] 1 _[σ]_ _Z_ [2] [.] [Fig.] [3] [shows] [that] [a] [one-layer] [net-]

work of size _n_ = 16 trained on tasks with _σZ_ [2] [=] [1] _[, σ]_ 0 [2] [=]
2 _, d_ = 8 _, L_ = 500 indeed achieves this bound, training to nearly diagonal weights with the appropriate scale
_⟨wKQ_ [(] _[ii]_ [)] _[⟩⟨][w]_ _P V_ [(] _[ii]_ [)] _[⟩]_ [= 0] _[.]_ [327] _[ ≈]_ [1] _[/]_ [3][ (similar weights are learned]
for each seed, up to a sign flip).



5


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**









Epoch Epoch Epoch



(b)



Final weights: linear softmax



Initial weights Final weights ( ≈ diagonal) Initial weights Final weights



_Figure 3._ (a) Training dynamics for the studied cases using one-layer softmax attention (circles) as well as linear attention (triangles).
Solid lines represent the average loss over six seeds, with the shaded area indicating the range for cases 2 and 3. For each case, the
grey dashed baseline indicates the 0-predictor, and the pink line indicates the Bayes optimal predictor. All cases use a context length
of _L_ = 500, ambient dimension _n_ = 16, and are trained with Adam on a dataset of size 800 with batch size 80 and standard weight
initialization _wij_ _∼_ _U_ [ _−_ 1 _/_ ~~_[√]_~~ _n,_ 1 _/_ ~~_[√]_~~ _n_ ]. (b) Final attention weights _WKQ_ and _WP V_ are shown. For each, we indicate the mean of the
diagonal elements. Representative initial weights are displayed for the second and third case.



Fig. 4(a) displays how this bound is approached as the
context length _L_ of training samples is increased. In Fig.
4(b) we study how the performance of a model trained to
denoise random subspaces of dimension _d_ = 8 is affected
by shifts in the subspace dimension at inference time. We
find that when provided sufficient context, such models can
adapt with mild performance loss to solve more challenging
tasks not present in the training set.


It is evident from Fig. 3(a) that the softmax network performs similarly to the linear one for this task. We can understand this through the small argument expansion of the
softmax function mentioned above. The learned weights displayed in Fig. 3(b) indicate that _β_ [softmax] _≈_ 0 _._ 194 becomes
small (note it decreases by a factor _ϵ_ _≈_ 0 _._ 344 relative to
_β_ [linear] ), while the value scale _α_ [softmax] _≈_ 1 _._ 607 becomes
larger by a similar factor _∼_ 1 _/ϵ_ to compensate. Thus, although the optimal denoiser for this case is intuitively expressed through linear self-attention, it can also be achieved
with softmax self-attention in the appropriate limit.


Moreover, we find that when the entire prompt undergoes
a global invertible transformation _A_ = _I_, the optimal attention weights are no longer scaled identity matrices but
acquire a structured form determined by _A_ . Both linear and
softmax attention layers are able to recover this structure
through training; see Appendix H for details and empirical
verification.



**3.3. Case 2 – Nonlinear manifolds**


Fig. 3 (case 2) shows networks of size _n_ = 16 trained
to denoise subspheres of dimension _d_ = 8 and radius
_R_ = 1, with corruption _σZ_ [2] [=] [0] _[.]_ [1] [and] [context] [length]
_L_ = 500. Once again, the network trains to have scaled
identity weights.


We note that although the network nearly achieves the optimal MSE on the test set, the weights appear at first glance
to deviate slightly from the Bayes optimal predictor of Section 2.3, which indicated _WP V_ = _αI_, _WKQ_ = _βI_ with
_α_ = 1 _, β_ = 1 _/σZ_ [2] [.] [To better understand this, we consider a]
coarse-grained MSE loss landscape by scanning over _α_ and
_β_ . See Fig. 6(a) in Appendix G. We find that the 2D loss
landscape has roughly hyperbolic level sets which is suggestive of the linear attention limit, where the weight scales
become constrained by their product _αβ_ . Reflecting the
symmetry of the problem, we also note mirrored negative
solutions (i.e. one could also identify _α_ = _−_ 1, _β_ = _−_ 1 _/σZ_ [2]
from the analysis in Section 2.3). Importantly, the plot
shows that the trained network lies in the same valley of the
loss landscape as the optimal predictor, in agreement with
Fig. 3. Moreover, the shape of the loss landscape suggested
that linear attention might also be applicable to this case,
which we demonstrate and discuss further in Appendix G.


**3.4. Case 3 – Gaussian mixtures**


Figure 3 (case 3) shows networks of size _n_ = 16 trained
to denoise balanced Gaussian mixtures with _p_ = 8 compo


6


Predict










|Performance<br>maintained away L=30<br>from d=8<br>L=50<br>L=500<br>in-cont<br>Train n=16 model:<br>d=8, L=500<br>subspace provi|Col2|
|---|---|
|||



_Figure 4._ (a) Trained linear attention network converges to Bayes optimal estimator as context length increases ( _n_ = 16, _d_ = 8,
_σ_ 0 [2] [=] [2] _[, σ]_ _z_ [2] [=] [1][).] [(b)] [A] [network] [trained] [to] [denoise] [subspaces] [of] [dimension] _[d]_ [=] [8] [can] [accurately] [denoise] [subspaces] [of] [different]
dimensions presented at inference time, given sufficient context.



nents that have isotropic variance _σ_ 0 [2] [=] [0] _[.]_ [02] [and] [centers]
randomly placed on the unit sphere in R _[n]_ . The corruption
magnitude is _σZ_ [2] [= 0] _[.]_ [1][ and context length is] _[ L]_ [ = 500][.] [The]
baselines show the zero predictor (dashed grey line) as well
as the optimum from Proposition (4) (pink) and its _σ_ 0 [2] _[→]_ [0]
approximation Eq. (8) (grey).


The trained weights qualitatively approach the optimal estimator for the zero-variance limit but with a slightly different
scaling: while the scale of _WP V_ is _α ≈_ 1, the _WKQ_ scale is
_β_ _≈_ 5 _._ 127 _<_ 1 _/σZ_ [2] [.] [To study this, we provide a correspond-]
ing plot of the 2D loss landscape in Fig. 6(a) in Appendix G.
While the symmetry of the previous case has been broken
(the context cluster centers _{µa}_ will not satisfy _⟨µ⟩_ = 0),
we again find that the trained network lies in the anticipated
global valley of the MSE loss landscape.


**4. Connection to dense associative memory**
**networks**


In each of the denoising problems studied above, we have
shown analytically and empirically that the optimal weights
of the one-layer transformer are scaled identity matrices
_WP V_ _≈_ _αI, WKQ_ _≈_ _βI_ . In the softmax case, the trained
denoiser can be concisely expressed as


_x_ ˆ = _g_ ( _X_ 1: _L,_ ˜ _x_ ) := _αX_ 1: _L_ softmax( _βX_ 1: _[T]_ _L_ _[x]_ [˜][)] _[,]_


re-written such that _X_ _∈_ R _[n][×][L]_ stores pure context tokens.


We now demonstrate that such denoising corresponds to
one-step gradient descent (with specific step sizes) of energy
models related to dense associative memory networks, also
known as modern Hopfield networks (Ramsauer et al., 2021;
Demircigil et al., 2017; Krotov & Hopfield, 2016).



Consider the energy function:




[1]

2 _α_ _[∥][s][∥]_ [2] _[ −]_ _β_ [1]



_E_ ( _X_ 1: _L, s_ ) = [1]



_β_ [log]




- _L_ 
 - _e_ _[βX]_ _t_ _[T]_ _[s]_


_t_ =1



_,_ (16)



which mirrors the Ramsauer et al. (2021) construction but
with a Lagrange multiplier added to the first term. Figure 5
illustrates this energy landscape for the spherical manifold
case.


Num. steps: 1 Num. steps: 50





_Figure 5._ Gradient descent denoising for the nonlinear manifold
case (spheres) in _n_ = 2 with _d_ = 1. A context-aware dense associative memory network _E_ ( _X_ 1: _L, s_ ) is constructed whose gradient
corresponds to the Bayes optimal update (trained attention layer).
Note that the density of sampled context tokens sculpts the valleys
of the energy landscape. Left: the attention step of a one-layer
transformer trained on the denoising task corresponds to a single
gradient descent step. Right: Iterating the denoising process—as
is conventional for Hopfield networks—can potentially degrade
the estimate by causing it to become query-independent (e.g. converging to a distant minimum). Here _R_ = 1 _, σZ_ [2] [=] [10] _[, L]_ [=] [20]
and _α_ = 1 _, β_ = 1 _/σZ_ [2] [.]



7


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**



An operation inherent to the associative memory perspective
is the recurrent application of a denoising update. Gradient
descent iteration _s_ ( _t_ + 1) = _s_ ( _t_ ) _−_ _γ_ _∇sE_ - _X_ 1: _L, s_ ( _t_ )�

yields




    _s_ ( _t_ + 1) = 1 _−_ _[γ]_

_α_




- _s_ ( _t_ ) + _γX_ 1: _L_ softmax� _βX_ 1: _[T]_ _L_ _[s]_ [(] _[t]_ [)] - _._
(17)



More broadly, our study suggests that trained attention layers can readily adopt structures that facilitate context-aware
associative retrieval. We have also noted preliminary connections between our work and other architectural features
of modern transformers, namely layer normalization and
residual streams, which warrant further study.


In-context denoising and generative modeling both involve
learning about an underlying distribution, suggesting potential relationships between these two tasks. Recently, Pham
et al. (2024) invoked spurious states of the Hopfield model
as a way of understanding how one can move away from
retrieving individual memorized patterns towards generalization via appropriate mixtures of multiple similar “memories”. In our work, one-step updates do not have to land
in a spurious minimum, but we often operate under circumstances where there are such states (see, for example,
the energy landscape in Fig. 5). More generally, analogies between energy-based associative memory and diffusion models have recently been noted (Ambrogioni, 2024;
Hoover et al., 2024b). Lastly, Bayes optimal denoisers play
an important role in the analysis (Ghio et al., 2024) of a
very related generative model that is based on stochastic
interpolants (Albergo & Vanden-Eijnden, 2023). Although
this work focuses on the case where it is possible to sample
enough tokens from the relevant distributions for certain
functions to converge, generative models become important
when the distribution is in a prohibitively high-dimensional
space making direct sampling difficult. Nonetheless, investigating the precise relationship between our work and
different generative modeling approaches would be an interesting direction to pursue.


Overall, this work refines the connection between dense
associative memories and attention layers first identified
in (Ramsauer et al., 2021). While we show that one energy minimization step of a particular DAM (associated
with a trained attention layer) is optimal for the denoising
tasks studied here, it remains an open question whether
multilayer architectures with varying or tied weights could
extend these results to more complex tasks by effectively
performing multiple iterative steps. This aligns with recent studies on in-context learning, which have considered
whether transformers with multiple layers emulate gradient
descent updates on a context-specific objective (Von Oswald
et al., 2023; Shen et al., 2024; Dai et al., 2023; Ahn et al.,
2023), and may provide a bridge to work on emerging architectures guided by associative memory principles (Hoover
et al., 2023). Investigating when and how multilayer attention architectures perform such gradient descent iterations
in a manner that is both context-dependent and informed by
a large training set represents an exciting direction for future research at the intersection of transformer mechanisms,
associative memory retrieval, and in-context learning.



It is now clear that initializing the state to the query _s_ (0) = _x_ ˜
and taking a single step with size _γ_ = _α_ recovers the behavior of the trained attention model (Fig. 5). The attention
mechanism here is thus mechanistically interpretable: the
context tokens _X_ 1: _L_ induce a context-dependent associative memory landscape, while the query acts as an initial
condition for inference-time gradient descent. One could
naturally consider alternative step sizes and recurrent iteration. However, Fig. 5 demonstrates that naive iteration of
Eq. (17) has the potential to degrade performance.


Additional details are provided in Appendix I. In particular, the energy model for linear attention is discussed in
Appendix I.1.


**5. Discussion**


Motivated by the connection between attention mechanisms
and dense associative memories, here we have introduced incontext denoising, a task that distills their relationship. We
first analyze the general problem, deriving Bayes optimal
predictors for certain restricted tasks. We identify that onelayer transformers using either softmax or linearized selfattention are expressive enough to describe these predictors.
We then empirically demonstrate that standard training of
attention layers from random initial weights will readily
converge to scaled identity weights with scales that approach
the derived optima given sufficient context. Accordingly,
the rather minimal transformers studied here can perform
optimal denoising of novel tasks provided at inference time
via self-contained prompts. This work therefore sheds light
on other in-context learning phenomena, a point we return
to below.


While practical transformers differ in various ways from
the minimal models studied here, we note several key connections. Intriguingly, the self-attention heads of trained
transformers sometimes exhibit weights _WKQ_, _WP V_ that
resemble scaled identity matrices, i.e. _cI_ + _ϵ_ with small fluctuations _ϵij_ _∼N_ (0 _, σ_ [2] ), an observation noted in Trockman
& Kolter (2023). This phenomenon motivated their proposal of “mimetic” weight initialization schemes mirroring
this learned structure. Relatedly, connections to associative
memory concepts have been explored in other architectures
(Smart & Zilman, 2021), which enabled data-dependent
weight initialization strategies to be identified and leveraged.



8


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**



**Software and Data**


Python code underlying this work is available at
[https://github.com/mattsmart/in-context-denoising.](https://github.com/mattsmart/in-context-denoising)


**Acknowledgements**


MS acknowledges M. Mezard for very useful feedback on´
an earlier version of this work. AS thanks D. Krotov and
P. Mehta for enlightening discussions on related matters.
Our early work also benefited from AS’s participation in the
deeplearning23 workshop at the Kavli Institute for Theoretical Physics (KITP), which was supported in part by grants
NSF PHY-1748958 and PHY-2309135 to KITP. AS thanks
Y. Bahri and C. Pehlevan for their patience and willingness
to listen to our early ideas at KITP.


**Impact Statement**


This paper presents work whose goal is to advance the field
of Machine Learning. There are many potential societal
consequences of our work, none which we feel must be
specifically highlighted here.


**References**


Ahn, K., Cheng, X., Daneshmand, H., and Sra, S. Transformers learn to implement preconditioned gradient descent for in-context learning. _Advances in Neural Infor-_
_mation Processing Systems_, 36:45614–45650, 2023.


Akyurek,¨ E., Schuurmans, D., Andreas, J., Ma, T., and
Zhou, D. What learning algorithm is in-context learning? investigations with linear models. In _The Eleventh_
_International Conference on Learning Representations_,
2023. [URL https://openreview.net/forum?](https://openreview.net/forum?id=0g0X4H8yN4I)
[id=0g0X4H8yN4I.](https://openreview.net/forum?id=0g0X4H8yN4I)


Albergo, M. S. and Vanden-Eijnden, E. Building normalizing flows with stochastic interpolants. In _The_
_Eleventh_ _International_ _Conference_ _on_ _Learning_ _Repre-_
_sentations_, 2023. [URL https://arxiv.org/abs/](https://arxiv.org/abs/2209.15571)
[2209.15571.](https://arxiv.org/abs/2209.15571)


Amari, S.-I. Learning patterns and pattern sequences by selforganizing nets of threshold elements. _IEEE Transactions_
_on computers_, 100(11):1197–1206, 1972.


Ambrogioni, L. In search of dispersed memories: Generative diffusion models are associative memory networks. _Entropy_, 26(5), 2024. ISSN 1099-4300. doi: 10.
3390/e26050381. URL [https://www.mdpi.com/](https://www.mdpi.com/1099-4300/26/5/381)
[1099-4300/26/5/381.](https://www.mdpi.com/1099-4300/26/5/381)


Amit, D. J., Gutfreund, H., and Sompolinsky, H. Spinglass models of neural networks. _Physical Review A_, 32



(2):1007–1018, 1985. ISSN 10502947. doi: 10.1103/
PhysRevA.32.1007.


Bolle,´ D., Nieuwenhuizen, T. M., Castillo, I. P., and Verbeiren, T. A spherical hopfield model. _Journal of Physics_
_A: Mathematical and General_, 36(41):10269, 2003.


Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D.,
Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G.,
Askell, A., et al. Language models are few-shot learners.
_Advances in neural information processing systems_, 33:
1877–1901, 2020.


Chen, Y., Zeng, Q., Ji, H., and Yang, Y. Skyformer: Remodel self-attention with gaussian kernel and nystrom¨
method. _Advances_ _in_ _Neural_ _Information_ _Processing_
_Systems_, 34:2122–2135, 2021.


Choromanski, K. M., Likhosherstov, V., Dohan, D., Song,
X., Gane, A., Sarlos, T., Hawkins, P., Davis, J. Q., Mohiuddin, A., Kaiser, L., Belanger, D. B., Colwell, L. J.,
and Weller, A. Rethinking attention with performers. In
_International Conference on Learning Representations_,
2021. [URL https://openreview.net/forum?](https://openreview.net/forum?id=Ua6zuk0WRH)
[id=Ua6zuk0WRH.](https://openreview.net/forum?id=Ua6zuk0WRH)


Dai, D., Sun, Y., Dong, L., Hao, Y., Ma, S., Sui, Z., and Wei,
F. Why can gpt learn in-context? language models implicitly perform gradient descent as meta-optimizers, 2023.
[URL https://arxiv.org/abs/2212.10559.](https://arxiv.org/abs/2212.10559)


Demircigil, M., Heusel, J., Lowe,¨ M., Upgang, S., and
Vermet, F. On a model of associative memory with huge
storage capacity. _Journal of Statistical Physics_, 168:288–
299, 2017.


Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert:
Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 Confer-_
_ence of the North American Chapter of the Association for_
_Computational Linguistics:_ _Human Language Technolo-_
_gies, Volume 1 (Long and Short Papers)_, pp. 4171–4186,
2019.


Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn,
D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer,
M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby,
N. An image is worth 16x16 words: Transformers for
image recognition at scale. In _International Conference_
_on_ _Learning_ _Representations_, 2021. URL [https://](https://openreview.net/forum?id=YicbFdNTTy)
[openreview.net/forum?id=YicbFdNTTy.](https://openreview.net/forum?id=YicbFdNTTy)


Fischer, K. H. and Hertz, J. A. _Spin Glasses_ . Cambridge
University Press, 1993.


Garg, S., Tsipras, D., Liang, P. S., and Valiant, G. What
can transformers learn in-context? a case study of simple
function classes. In _Advances_ _in_ _Neural_ _Information_



9


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**



_Processing Systems_, volume 35, pp. 30583–30598, 2022.
[URL https://arxiv.org/abs/2208.01066.](https://arxiv.org/abs/2208.01066)


Ghio, D., Dandi, Y., Krzakala, F., and Zdeborova, L.´ Sampling with flows, diffusion, and autoregressive neural
networks from a spin-glass perspective. _Proceedings of_
_the National Academy of Sciences_, 121(27):e2311810121,
2024.


Gradshteyn, I. S. and Ryzhik, I. M. _Table of Integrals, Series,_
_and_ _Products_ . Elsevier/Academic Press, Amsterdam,
seventh edition, 2007.


Hoeffding, W. Probability inequalities for sums of bounded
random variables. _The collected works of Wassily Hoeffd-_
_ing_, pp. 409–426, 1994.


Hoover, B., Liang, Y., Pham, B., Panda, R., Strobelt,
H., Chau, D. H., Zaki, M. J., and Krotov, D. Energy
transformer. In _Thirty-seventh_ _Conference_ _on_ _Neural_
_Information_ _Processing_ _Systems_, 2023. URL [https:](https://openreview.net/forum?id=MbwVNEx9KS)
[//openreview.net/forum?id=MbwVNEx9KS.](https://openreview.net/forum?id=MbwVNEx9KS)


Hoover, B., Chau, D. H., Strobelt, H., Ram, P., and Krotov,
D. Dense associative memory through the lens of random
features. In _The_ _Thirty-eighth_ _Annual_ _Conference_ _on_
_Neural Information Processing Systems_, 2024a.


Hoover, B., Strobelt, H., Krotov, D., Hoffman, J., Kira, Z.,
and Chau, D. H. Memory in plain sight: Surveying the
uncanny resemblances of associative memories and diffusion models, 2024b. [URL https://arxiv.org/](https://arxiv.org/abs/2309.16750)
[abs/2309.16750.](https://arxiv.org/abs/2309.16750)


Hopfield, J. J. Neural networks and physical systems with
emergent collective computational abilities. _Proceedings_
_of the National Academy of Sciences of the United States_
_of_ _America_, 79(8):2554–2558, 1982. ISSN 00278424.
doi: 10.1073/pnas.79.8.2554.


Hu, J. Y.-C., Yang, D., Wu, D., Xu, C., Chen, B.-Y., and Liu,
H. On sparse modern hopfield model. In _Proceedings of_
_the 37th International Conference on Neural Information_
_Processing Systems_, NIPS ’23, 2023.


Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F.
Transformers are rnns: fast autoregressive transformers
with linear attention. In _Proceedings_ _of_ _the_ _37th_ _Inter-_
_national_ _Conference_ _on_ _Machine_ _Learning_, ICML’20.
JMLR.org, 2020.


Krotov, D. A new frontier for hopfield networks. _Nature_
_Reviews Physics_, 5(7):366–367, 2023.


Krotov, D. and Hopfield, J. J. Dense associative memory for
pattern recognition. In _Advances in Neural Information_
_Processing Systems_, volume 29, 2016.



Krotov, D. and Hopfield, J. J. Large associative memory problem in neurobiology and machine learning. In
_International Conference on Learning Representations_,
2021. [URL https://openreview.net/forum?](https://openreview.net/forum?id=X4y_10OX-hX)
[id=X4y_10OX-hX.](https://openreview.net/forum?id=X4y_10OX-hX)


Little, W. A. The existence of persistent states in the brain.
_Mathematical biosciences_, 19(1-2):101–120, 1974.


Loeve, M.` Probability theory i. _Graduate Texts in Mathe-_
_matics_, 1977.


Lucibello, C. and Mezard,´ M. Exponential capacity of
dense associative memories. _Phys._ _Rev._ _Lett._, 132:
077301, Feb 2024. doi: 10.1103/PhysRevLett.132.
077301. [URL https://link.aps.org/doi/10.](https://link.aps.org/doi/10.1103/PhysRevLett.132.077301)
[1103/PhysRevLett.132.077301.](https://link.aps.org/doi/10.1103/PhysRevLett.132.077301)


Millidge, B., Salvatori, T., Song, Y., Lukasiewicz, T., and
Bogacz, R. Universal hopfield networks: A general framework for single-shot associative memory models. In _In-_
_ternational Conference on Machine Learning_, pp. 15561–
15583. PMLR, 2022.


Nakano, K. Associatron-a model of associative memory.
_IEEE Transactions on Systems, Man, and Cybernetics_, 2:
380–388, 1972.


Pham, B., Raya, G., Negri, M., Zaki, M. J., Ambrogioni,
L., and Krotov, D. Memorization to generalization: The
emergence of diffusion models from associative memory.
In _NeurIPS_ _2024_ _Workshop_ _on_ _Scientific_ _Methods_ _for_
_Understanding Deep Learning_, 2024.


Ramsauer, H., Schafl, B., Lehner, J., Seidl, P., Widrich, M.,¨
Gruber, L., Holzleitner, M., Adler, T., Kreil, D. P., Kopp,
M. K., Klambauer, G., Brandstetter, J., and Hochreiter, S.
Hopfield networks is all you need. In _9th International_
_Conference_ _on_ _Learning_ _Representations,_ _ICLR_ _2021,_
_Virtual Event, Austria, May 3-7, 2021_ . OpenReview.net,
2021. [URL https://openreview.net/forum?](https://openreview.net/forum?id=tL89RnzIiCd)
[id=tL89RnzIiCd.](https://openreview.net/forum?id=tL89RnzIiCd)


Reddy, G. The mechanistic basis of data dependence and
abrupt learning in an in-context classification task. In _The_
_Twelfth International Conference on Learning Represen-_
_tations_, 2024. URL [https://openreview.net/](https://openreview.net/forum?id=aN4Jf6Cx69)
[forum?id=aN4Jf6Cx69.](https://openreview.net/forum?id=aN4Jf6Cx69)


Rigollet, P. and Hutter,¨ J.-C. High-dimensional statistics.
_arXiv preprint arXiv:2310.19244_, 2023.


Santos, S. J. R. D., Niculae, V., Mcnamee, D. C., and Martins, A. Sparse and structured hopfield networks. In
_Proceedings of the 41st International Conference on Ma-_
_chine Learning_, volume 235 of _Proceedings of Machine_
_Learning Research_, pp. 43368–43388. PMLR, 21–27 Jul
[2024. URL https://proceedings.mlr.press/](https://proceedings.mlr.press/v235/santos24a.html)
[v235/santos24a.html.](https://proceedings.mlr.press/v235/santos24a.html)



10


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**


Shen, L., Mishra, A., and Khashabi, D. Position: Do pretrained transformers learn in-context by gradient descent?
In _Proceedings of the 41st International Conference on_
_Machine_ _Learning_, volume 235 of _Proceedings_ _of_ _Ma-_
_chine Learning Research_, pp. 44712–44740. PMLR, 21–
27 Jul 2024. URL [https://proceedings.mlr.](https://proceedings.mlr.press/v235/shen24d.html)
[press/v235/shen24d.html.](https://proceedings.mlr.press/v235/shen24d.html)


Smart, M. and Zilman, A. On the mapping between hopfield networks and restricted boltzmann machines. _In-_
_ternational_ _Conference_ _on_ _Learning_ _Representations_,
2021. [URL https://openreview.net/forum?](https://openreview.net/forum?id=RGJbergVIoO)
[id=RGJbergVIoO.](https://openreview.net/forum?id=RGJbergVIoO)


Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux,
M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E.,`
Azhar, F., et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_, 2023.


Trockman, A. and Kolter, J. Z. Mimetic initialization of selfattention layers. In _Proceedings of the 40th International_
_Conference on Machine Learning_, ICML’23. JMLR.org,
2023.


Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Łukasz Kaiser, and Polosukhin, I. Attention is all you need. In _Advances in Neural Information_
_Processing Systems_, volume 2017-December, 2017.


Von Oswald, J., Niklasson, E., Randazzo, E., Sacramento,
J., Mordvintsev, A., Zhmoginov, A., and Vladymyrov,
M. Transformers learn in-context by gradient descent.
In _International_ _Conference_ _on_ _Machine_ _Learning_, pp.
35151–35174. PMLR, 2023.


Wu, D., Hu, J. Y.-C., Hsiao, T.-Y., and Liu, H. Uniform
memory retrieval with larger capacity for modern hopfield models. In _Proceedings_ _of_ _the_ _41st_ _International_
_Conference on Machine Learning_, ICML’24. JMLR.org,
2024a.


Wu, D., Hu, J. Y.-C., Li, W., Chen, B.-Y., and Liu, H.
Stanhop: Sparse tandem hopfield model for memoryenhanced time series prediction. In _The Twelfth Interna-_
_tional Conference on Learning Representations_, 2024b.
[URL https://arxiv.org/abs/2312.17346.](https://arxiv.org/abs/2312.17346)


Zhang, R., Frei, S., and Bartlett, P. L. Trained transformers learn linear models in-context. _Journal of Machine_
_Learning_ _Research_, 25(49):1–55, 2024. URL [http:](http://jmlr.org/papers/v25/23-1042.html)
[//jmlr.org/papers/v25/23-1042.html.](http://jmlr.org/papers/v25/23-1042.html)


11


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**


**A. Notation**


**A.1. Recurring notation**


  - _n_  - ambient dimension of input tokens.


  - _xt_ _∈_ R _[n]_  - the value of the _t_ -th random input token.


  - _E_ = ( _X_ 1 _, ..., XL,_ _X_ [˜] ) – the random variable input to the sequence model. The “tilde” indicates that the final token has
in some way been corrupted. _E_ takes values ( _x_ 1 _, ..., xL,_ ˜ _x_ ) _∈_ R _[n][×]_ [(] _[L]_ [+1)] . Note: while capital _X_ or _Xi_ here denotes a
random variable, in Section D use _X_ 1: _L_ or simply _X_ to refer to the realized matrix of input tokens.


  - _L_  - context length = number of uncorrupted tokens.


  - _d_  - dimensionality of manifold _S_ that _xt_ are sampled from


  - _N_  - number of training pairs


**A.2. Bayes posterior notation**


  - _pX_ ( _x_ ) is task-dependent (the three scenarios considered here are introduced above).


  - _pX_ ˜ [(˜] _[x]_ [)] [where] _[x]_ [˜] [=] _[x]_ [ +] _[ z]_ [.] [For] [a] [sum] [of] [independent] [random] [variables,] _[Y]_ [=] _[X]_ [1] [+] _[ X]_ [2][,] [their] [pdf] [is] [a] [convolution]
_pY_ ( _y_ ) =   - _pX_ 1( _x_ ) _pX_ 2( _y −_ _x_ ) _dx_ . Thus:


                   _pX_ ˜ [(˜] _[x]_ [) =] _pZ_ ( _z_ ) _pX_ (˜ _x −_ _z_ ) _dz_



= _CZ_




_e_ _[−∥][z][∥]_ [2] _[/]_ [2] _[σ]_ _Z_ [2] _pX_ (˜ _x −_ _z_ ) _dz_



where _CZ_ = (2 _πσZ_ [2] [)] _[−][n/]_ [2][ is a constant.]


- _pX_ ˜ _|X_ [(˜] _[x][ |][ x]_ [)][:] [This is simply]


- _pX|X_ ˜ [(] _[x][ |]_ _[x]_ [˜][)][:] [By Bayes’ theorem, this is]



_pZ_ (˜ _x −_ _x_ ) = _CZe_ _[−∥][x]_ [˜] _[−][x][∥]_ [2] _[/]_ [2] _[σ]_ _Z_ [2] _._



_pX_ ˜ _|X_ [(˜] _[x][ |][ x]_ [)] _[p][X]_ [(] _[x]_ [)]
_pX|X_ ˜ [(] _[x][ |]_ _[x]_ [˜][) =]

_pX_ ˜ [(˜] _[x]_ [)]




- Posterior mean:



_e_ _[−∥][x]_ [˜] _[−][x][∥]_ [2] _[/]_ [2] _[σ]_ _Z_ [2] _pX_ ( _x_ )
=

       - _e_ _[−∥][x]_ [˜] _[−][x][′][∥]_ [2] _[/]_ [2] _[σ]_ _Z_ [2] _pX_ ( _x_ _[′]_ ) _dx_ _[′]_ _[.]_


      E _X|X_ ˜ [[] _[X]_ _[|]_ _[X]_ [˜] [] =] _x pX|X_ ˜ [(] _[x][ |]_ _[X]_ [˜] [)] _[dx]_



=




- _x e_ _[−∥]_ _X_ [˜] _−x∥_ [2] _/_ 2 _σZ_ [2] _pX_ ( _x_ ) _dx_
_._

 - _e_ _[−∥]_ _X_ [˜] _−x∥_ [2] _/_ 2 _σZ_ [2] _pX_ ( _x_ ) _dx_


12


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**


**B. Bayes optimal predictors for square loss**


**B.1. Proof of Proposition 1**


_Proof._ Observe that


E          - _∥X_ _−_ _f_ ( _X_ [˜] ) _∥_ [2][�] = E _X_ ˜ �E _X|X_ ˜          - _∥X_ _−_ _f_ ( _X_ [˜] ) _∥_ [2] _|_ _X_ [˜]          - [�]

= E _X_ ˜ �E _X|X_ ˜                   - _∥X_ _−_ E[ _X_ _|_ _X_ [˜] ] _∥_ [2] _|_ _X_ [˜]                   
+ _∥_ E[ _X_ _|_ _X_ [˜] ] _−_ _f_ ( _X_ [˜] ) _∥_ [2][�]


_≥_ E _X_ ˜ �E _X|X_ ˜              - _∥X_ _−_ E[ _X_ _|_ _X_ [˜] ] _∥_ [2] _|_ _X_ [˜]              - [�]


                         -                         = E _X_ ˜ Tr Cov( _X_ _|_ _X_ [˜] ) _._


Note the final line is independent of _f_ . This inequality becomes an equality when _f_ ( _X_ [˜] ) = E[ _X_ _|_ _X_ [˜] ].


**C. Details of Bayes optimal denoising baselines for each case**


**C.1. Proof of Proposition 2**


_Proof._ The linear denoising task is a special case of the result in Proposition 1. Here, _X_ is an isotropic Gaussian in a
restricted subspace,


_−_ _[∥][x][−][x]_ [˜] _[∥]_ [2]
_pX|X_ ˜ [(] _[x][ |]_ _[x]_ [˜][) =] _[ C]_ [(˜] _[x]_ [)] _[p][X]_ [(] _[x]_ [)] _[e]_ 2 _σZ_ [2]


where _C_ (˜ _x_ ) is a normalizing factor. The noise can be decomposed into parallel and perpendicular parts using the projection
_P_ onto _S_, i.e.


_X_ ˜ = _X_ ˜ _∥_ + _X_ ˜ _⊥_ = _P_ _X_ ˜ + ( _I_ _−_ _P_ ) ˜ _X,_


so that




_[−][x]_ [˜] _[∥]_ [2] _−_ _∥x−x_ ˜ _∥_ _∥_ [2]

2 _σZ_ [2] = _e_ 2 _σZ_ [2]



2 _σZ_ [2] _._




_−_ _[∥][x][−][x]_ [˜] _[∥]_ [2]
_e_ 2 _σZ_ [2]



_−x_ ˜ _∥_ _∥_ [2] _−_ _∥x_ ˜ _⊥_ _∥_ [2]

2 _σZ_ [2] _e_ 2 _σZ_ [2]



Only the first factor matters for _pX|X_ ˜ [(] _[x]_ _[|]_ _[x]_ [˜][)][ since it depends on] _[ x]_ [.] [Then, for] _[ x]_ _[∈]_ _[S]_ [, the linear subspace supporting] _[ p][X]_ [,]
dropping the _x_ independent _x_ ˜ _⊥_ contribution,




_−_ _∥x−x_ ˜ _∥_ _∥_ [2]
_pX_ ( _x_ ) _e_ 2 _σZ_ [2]




_[x][∥]_ [2] _−_ _∥x−x_ ˜ _∥_ _∥_ [2]

2 _σ_ 0 [2] 2 _σZ_ [2]



_−x_ ˜ _∥_ _∥_ [2] _−_ _[∥][x][∥]_ [2]

2 _σZ_ [2] _∝_ _e_ 2 _σ_ 0 [2]



2 _σZ_ [2]



 _∥x −_ _σ_ 0 [2]

_σ_ 0 [2][+] _[σ]_ _Z_ [2] _[x]_ [˜] _[∥][∥]_ [2]
 _−_ [2] [2]






 _._







_∝_ exp



0 _[σ]_ _Z_ [2]
2 _[σ]_ [2]
_σ_ 0 [2][+] _[σ]_ _Z_ [2]



Thus, _f_ ( _X_ [˜] ) = _σ_ 0 [2] _σ_ [+] 0 [2] _[σ]_ _Z_ [2] _X_ ˜ _∥_ = _σ_ 0 [2] _σ_ [+] 0 [2] _[σ]_ _Z_ [2] _[P]_ _[X]_ [˜] [.]

Using _X_ [˜] = _X_ + _Z_, _X_ = _PX_, and the independence of _X_ and _Z_


E� _∥X_ _−_ _σ_ 0 [2] _P_ _X_ [˜] _∥_ [2][�] = E� _∥_ _σZ_ [2] _PX∥_ [2][�] + E� _∥_ _σ_ 0 [2] _PZ∥_ [2][�] = _[σ]_ _Z_ [4] _[dσ]_ 0 [2] [+] _[ σ]_ 0 [4] _[dσ]_ _Z_ [2] = _[dσ]_ 0 [2] _[σ]_ _Z_ [2] _._
_σ_ 0 [2] [+] _[ σ]_ _Z_ [2] _σ_ 0 [2] [+] _[ σ]_ _Z_ [2] _σ_ 0 [2] [+] _[ σ]_ _Z_ [2] ( _σ_ 0 [2] [+] _[ σ]_ _Z_ [2] [)][2] _σ_ 0 [2] [+] _[ σ]_ _Z_ [2]


13


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**


**C.2. Proof of Proposition 3**


_Proof._ In the nonlinear manifold denoising problem, we focus on the case of lower dimensional spheres _S_ (e.g. the circle
_S_ [1] _⊂_ R [2] ). For such manifolds, we have



E[ _X_ _|_ _X_ [˜] = _x_ ˜] =


=




_−_ _∥x−x_ ˜ _∥_ _∥_ [2]

- _e_ 2 _σZ_ [2]




- _e_ _[⟨][x,][x]_ [˜] _[∥][⟩][/σ]_ _Z_ [2] _x dSx_
_._

 - _e_ _[⟨][x,][x]_ [˜] _[∥][⟩][/σ]_ _Z_ [2] _dSx_



_e_ 2 _σZ_ [2] _x pX_ ( _x_ ) _dx_

_−_ _∥x−x_ ˜ _∥_ _∥_ [2]

- _e_ 2 _σZ_ [2] _pX_ ( _x_ ) _dx_



2 _σZ_ [2] _pX_ ( _x_ ) _dx_



We have used the fact that _∥x −_ _x_ ˜ _∥∥_ [2] = _∥x∥_ [2] + _∥x_ ˜ _∥∥_ [2] _−_ 2 _⟨x,_ ˜ _x∥⟩_ and that _∥x∥_ is fixed on the sphere.


The integrals can be evaluated directly once the parameters are specified. If _S_ is a _d_ –sphere of radius _R_, then the optimal
predictor is again a shrunk projection of _x_ ˜ onto _S_,


�0 _π_ _[e][R][∥][x]_ [˜] _[∥][∥]_ [cos] _[ θ/σ]_ _Z_ [2] cos _θ_ sin [(] _[d][−]_ [1)] _θ dθ_
�0 _π_ _[e][R][∥][x]_ [˜] _[∥][∥]_ [cos] _[ θ/σ]_ _Z_ [2] sin [(] _[d][−]_ [1)] _θ dθ_ _R_ _∥_ _[x]_ _x_ [˜] ˜ _[∥]_ _∥∥_



_I d_ +1
= 2

_I d−_ 1

2




- _R_ _[∥][x]_ [˜] _[∥][∥]_

_σZ_ [2]




- _R_ _[∥][x]_ [˜] _[∥][∥]_

_σZ_ [2]

- _R_ _[∥][x]_ [˜] _[∥][∥]_

_σZ_ [2]




- _R_ _[∥][x]_ [˜] _[∥][∥]_

[2]





_R_ _[x]_ [˜] _[∥]_

- _∥x_ ˜ _∥∥_ _[,]_



where we used identities involving _Iν_ ( _y_ ), modified Bessel function of the first kind of order _ν_ (Gradshteyn & Ryzhik, 2007).
The vector _R_ _∥xx_ ˜˜ _∥∥∥_ [is the point on] _[ S]_ [in the direction of] _[ x][∥]_ [.]


**C.3. Proof of Proposition 4**


_Proof._ For the clustering case involving isotropic Gaussian mixtures with parameters _{wa,_ ( _µa, σa_ [2][)] _[}][p]_ _a_ =1 [,]




_−_ _[∥][x][−][x]_ [˜] _[∥]_ [2]

- _e_ 2 _σZ_ [2]



_a_



2 _σZ_ [2] 



- _−_ _[∥][x][−][µα]_ _[∥]_ [2] _waCae_ 2 _σa_ [2] _x dx_



E[ _X_ _|_ _X_ [˜] = _x_ ˜] =




_−_ _[∥][x][−][x]_ [˜] _[∥]_ [2]

- _e_ 2 _σZ_ [2]



_a_



2 _σZ_ [2] 


_,_

- _−_ _[∥][x][−][µa]_ _[∥]_ [2] _waCae_ 2 _σa_ [2] _dx_



where _Ca_ = (2 _πσa_ [2][)] _[−]_ _[n]_ 2 .


We can simplify this expression by completing the square in the exponent and using the fact that the integral of a Gaussian
about its mean is zero. This yields

E[ _X_ _|_ _X_ [˜] = _x_ ˜] = �� _a_ _[w]_ _a_ _[w][a][C][a][C][a][m][a]_                 - _[a]_ �exp(exp( _−−gag_ ) _a dx_ ) _dx_



where we have introduced


with



2

- _σZ_ [+] _[ σ]_ _a_ [2]
_σZ_ [2] _[σ]_ _a_ [2]




- 1
_∥x −_ _mα∥_ [2] +
2( _σZ_ [2] [+] _[ σ]_ _a_ [2][)] _[∥][x]_ [˜] _[ −]_ _[µ][a][∥]_ [2] _[,]_


_a_ _[x]_ [˜][ +] _[ σ]_ _Z_ [2] _[µ][a]_
_ma_ = _[σ]_ [2] _._
_σa_ [2] + _σZ_ [2]


14



_ga_ = [1]

2



_ga_ = [1]


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**


Doing the integrals and using the expressions for _Ca, ma_








_[∥][x]_ [˜] _[−][µ][a][∥]_ [2] �� _σa_ 2 _[x]_ [˜][+] _[σ]_ _Z_ [2] _[µ][a]_ 
2( _σZ_ [2] [+] _[σ]_ _a_ [2] [)] _σa_ [2] + _σZ_ [2]



E[ _X_ _|_ _X_ [˜] = _x_ ˜] =




  - _σZ_ 2 [+] _[σ]_ _a_ [2]   - _n/_ 2   _a_ _[w][a]_ _σa_ [2] exp _−_ 2( _[∥][x]_ [˜] _σ_ _[−]_ _Z_ [2] _[µ]_ [+] _[a][σ][∥]_ _a_ [2][2]




  - _σZ_ 2 [+] _[σ]_ _a_ [2]
_a_ _[w][a]_ _σ_ [2]




_[x]_ [˜] _[µ][a]_   
2( _σZ_ [2] [+] _[σ]_ _a_ [2] [)]







_σ_ [+] _a_ [2] _[σ]_ _a_ [2] - _n/_ 2 exp - _−_ 2( _[∥][x]_ [˜] _σ_ _[−]_ _Z_ [2] _[µ]_ [+] _[a][σ][∥]_ _a_ [2][2]



In the case that the center norms _∥µa∥_ are independent of _a_ and variances _σa_ [2] [=] _[ σ]_ [0][, we have]







E[ _X_ _|_ _X_ [˜] = _x_ ˜] = _σ_ 0 [2] _x_ ˜ + _σZ_ [2]
_σ_ 0 [2] [+] _[ σ]_ _Z_ [2] _σ_ 0 [2] [+] _[ σ]_ _Z_ [2]








- - _⟨x,µ_ ˜ _a⟩_

_a_ _[w][a]_ [ exp] _σZ_ [2] [+] _[σ]_ 0 [2]




    - _⟨x,µ_ ˜ _a⟩_
_a_ _[w][a][µ][a]_ [ exp] _σZ_ [2] [+] _[σ]_ 0 [2]



_._




Note that in the limit that _σ_ 0 _→_ 0, this becomes expressible by one-layer self-attention, since one can simply replace the
matrix of cluster centers _M_ = [ _µ_ 1 _. . . µp_ ] implicit in the expression with the context _X_ 1: _L_ itself,



_a_ _[w][a][e][⟨][µ][α][,]_ _X_ [˜] _⟩/σZ_ [2] _µa_

- _[⟨][µ][α][,]_ _X_ [˜] _⟩/σ_ [2]




   E[ _X_ _|_ _X_ [˜] ] =



_X_ [˜] _⟩/σZ_ [2] _[.]_
_a_ _[w][a][e][⟨][µ][α][,]_



**D. Additional details on attention layers and softmax expansion**


**D.1. Standard self-attention**


Given a sequence of _L_ seq input tokens _xi_ _∈_ R _[n]_ represented as a matrix _X_ _∈_ R _[n][×][L]_ [seq], standard self-attention defines query,
key, and value matrices
_K_ = _WKX, Q_ = _WQX, V_ = _WV X_ (A.1)

where _WK, WQ_ _∈_ R _[n]_ [attn] _[×][n]_ and _WV_ _∈_ R _[n]_ [out] _[×][n]_ . The softmax self-attention map (Vaswani et al., 2017) is then

Attn( _X, WV, WK_ _[T]_ _[W][Q]_ [) :=] _[ V]_ [ softmax][(] _[K]_ _[T][ Q]_ [)] _[ ∈]_ [R] _[n]_ [out] _[×][L]_ [seq] _[.]_ (A.2)


On merging _WK_, _WQ_ into _WKQ_ = _WK_ _[T]_ _[W][Q]_ [:] [The simplification] _[ W][KQ]_ [=] _[ W][ T]_ _K_ _[W][Q]_ [ (made here and elsewhere) is general]
only when _n_ attn _≥_ _n_ ; in that case, the product _WKQ_ can have rank _n_ and thus it is reasonable to work with the combined
matrix. On the other hand, if _n_ attn _< n_, then the rank of their product is at most _n_ attn and thus there are matrices in R _[n][×][n]_

that cannot be expressed as _WK_ _[T]_ _[W][Q]_ [.] [A similar point can be made about] _[ W][P V]_ [ .] [We note that while] _[ n]_ [attn] _[< n]_ [ may be used]
in practical settings, one often also uses multiple heads which when concatenated could be (roughly) viewed as a single
higher-rank head.


We will also use the simplest version of linear attention (Katharopoulos et al., 2020),


1
AttnLin( _X, WV, WK_ _[T]_ _[W][Q]_ [) :=] _V_ ( _K_ _[T]_ _Q_ ) _∈_ R _[n]_ [out] _[×][L]_ [seq] _._ (A.3)
_L_ seq


**D.2. Minimal transformer architecture for denoising**


We now consider a simplified one-layer linear transformer in term of our variable _E_ = ( _X_ 1: _L,_ _X_ [˜] ) taking values in R _[n][×]_ [(] _[L]_ [+1)]

and start with the linear transformer which still has sufficient expressive power to capture our finite sample approximation to
the Bayes optimal answer in the linear case. Inspired by Zhang et al. (2024), we define


AttnLin( _E, WP V, WKQ_ ) := [1] (A.4)

_L_ _[W][P V][ EM]_ [Lin] _[E][T][ W][KQ][E]_



taking values in R _[n][×]_ [(] _[L]_ [+1)] . The additional aspect compared to the last subsection is the masking matrix _M_ Lin _∈_
R [(] _[L]_ [+1)] _[×]_ [(] _[L]_ [+1)] which is of the form



_M_ Lin = �01 _I×LL_ 0 _L_ 0 _×_ 1




 _,_ (A.5)



15


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**


preventing _WP V_ _X_ [˜] from being added to the output.


Note that this more detailed expression is equivalent to the form used in the main text.


_X_ ˆ = _F_ Lin( _E, θ_ ) := [1] 1: _L_ _[W][KQ][X]_ [˜]

_L_ _[W][P V][ X]_ [1:] _[L][X]_ _[T]_


With learnable weights _WKQ, WP V_ _∈_ R _[n][×][n]_ abbreviated by _θ_, we define


_F_ ( _E, θ_ ) := [AttnLin( _E, WP V, WKQ_ )]: _,L_ +1 _._ (A.6)


Note that, when _WP V_ = _αIn, WKQ_ = _βIn_, and _αβ_ = _σ_ 0 [2][+] 1 _[σ]_ _Z_ [2] [,] _[ F]_ [(] _[E, θ]_ [)][ should approximate the Bayes optimal answer]

_f_ opt( _X_ [˜] ) as _L →∞_ .


Similarly, we could argue that the second two problems, the _d_ -dimesional spheres and the _σ_ 0 _→_ 0 zero limit of the Gaussian
mixtures could be addressed by the full softmax attention


Attn( _E, WP V, WKQ_ ) = _WP V E_ softmax( _E_ _[T]_ _WKQE_ + _M_ ) (A.7)


taking values in R _[n][×]_ [(] _[L]_ [+1)] where _M_ _∈_ R [¯] [(] _[L]_ [+1)] _[×]_ [(] _[L]_ [+1)] is a masking matrix of the form



_M_ = - 0 _L×_ ( _L_ +1)
( _−∞_ )11 _×L_ +1




_,_ (A.8)



1
once more, preventing the contribution of _X_ [˜] value to the output. The function softmax( _z_ ) := - _ni_ =1 _[e][z][i]_ [(] _[e][z]_ [1] _[, . . ., e][z][n]_ [)] _[T]_ _[∈]_ [R] _[n]_

is applied column-wise.


We then define
_F_ ( _E, θ_ ) := [Attn( _E, WP V, WKQ_ )]: _,L_ +1 _,_ (A.9)


which is equivalent to the simplified form used in the main text:


_X_ ˆ = _F_ ( _E, θ_ ) := _WP V X_ 1: _L_ softmax( _X_ 1: _[T]_ _L_ _[W][KQ][X]_ [˜] [)] _[.]_


**D.3. Proof of Theorem 3.1**


_Proof._ Let the support of _pX_ be a subset of a sphere, centered around the origin, of radius _R_ . Then the function




    - _L_ _Z_
_t_ =1 _[X][t][e][⟨][X][t][,][x]_ [˜] _[⟩][/σ]_ [2]
_g_ ( _{Xt}_ _[L]_ _t_ =1 _[,]_ [ ˜] _[x]_ [) =] - _Lt_ =1 _[e][⟨][X][t][,][x]_ [˜] _[⟩][/σ]_ _Z_ [2] =



1 - _L_ _Z_
_L_ _t_ =1 _[X][t][e][⟨][X][t][,][x]_ [˜] _[⟩][/σ]_ [2]

_L_ 1 - _Lt_ =1 _[e][⟨][X][t][,][x]_ [˜] _[⟩][/σ]_ _Z_ [2] _._ (A.10)



Both the numerator _L_ 1 - _Lt_ =1 _[X][t][e][⟨][X][t][,][x]_ [˜] _[⟩][/σ]_ _Z_ [2] and the denominator _L_ 1 - _Lt_ =1 _[e][⟨][X][t][,][x]_ [˜] _[⟩][/σ]_ _Z_ [2] are averages of independent and
identically distributed bounded random variables. By the strong law of large numbers, as _L →∞_, the average vector in the
numerator converges to almost surely to - _e_ _[⟨][x,][x]_ [˜] _[∥][⟩][/σ]_ _Z_ [2] _x dpX_ ( _x_ ) for each component, while the average in the denominator
almost surely converges - _e_ _[⟨][x,][x]_ [˜] _[∥][⟩][/σ]_ _Z_ [2] _dpX_ ( _x_ ), which is positive. So, as _L_ _→∞_, the ratio in Eq. A.10 converges almost
surely to

                     - _e_ _[⟨][x,][x]_ [˜] _[∥][⟩][/σ]_ _Z_ [2] _x dpX_ ( _x_ )

                      - _e_ _[⟨][x,][x]_ [˜] _[∥][⟩][/σ]_ _Z_ [2] _dpX_ ( _x_ ) _[,]_


which is the Bayes optimal answer _f_ opt(˜ _x_ ) for all _x_ ˜ _∈_ R _[n]_ .


**E. Further discussion of convergence rates as** _L →∞_ **and the dependence on dimensions**


Our analysis primarily focused on the asymptotic behavior as _L_ _→∞_ using the strong law of large numbers, which
just requires the mean to exist (Loeve`, 1977). However, in the linear example, our tokens are Gaussian, and in the two
nonlinear cases they are bounded. Intuitively, we expect error _O_ ( ~~_√_~~ [1] [In fact, we can give precise results of the form that]

_L_ [)][.]
the probability of the difference between the empirical sum for the ideal weights departing from the expectation by less


16


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**




~~�~~




 _f_ _d,_ ln [1]




[1] 
_δ_



than _C_ (˜ _x_ ) _L_ _δ_ is greater than 1 _−_ _δ_ . The function _C_ of the query vector and the function _f_ depend on the problem.

Interestingly, these bounds depend on _d_, the dimension spanned by the tokens, not the ambient dimension _n_ .



than _C_ (˜ _x_ )



As mentioned before, the results of the previous paragraph refer to the convergence of the finite sample attention expressions
for ideal weights, namely those corresponding to Bayes optimal answer. There is a second source of error associated with
finite sample estimation of weights, which should also get small as _L_ becomes large. Once more the expectation is that the
weights are known to error _O_ ( ~~_√_~~ [1]

_L_ [)][ for well-converged training procedures, although this is more difficult to guarantee or]
quantify analytically. Overall we expect the loss (MSE) to go down inversely with some power of _L_ . Fig. 4(a) provides
some empirical evidence for this relationship, showing how performance improves with increasing context length.


Notice that the one-layer transformer output is a linear combination of the uncorrupted samples. Hence, if the distribution
_pX_ is supported by a _d_ -dimensional linear subspace, the estimate _X_ [ˆ] is also in that subspace. We can therefore look at
convergence restricted to the supporting subspace. Therefore, it is the dimensionality of the supporting subspace that matters.


Let a _d_ -dimensional vector space _V_ be a linear subspace of R _[n]_ . We define the maximum norm for _V_ with respect to some
orthonormal basis _{vi}_ _[d]_ _i_ =1 [in] _[ V]_ [as] _[ ||][x][||][∞][,V]_ [:= max] _[i][∈{]_ [1] _[,...,d][}]_ _[|⟨][v][i][, x][⟩|]_ [ for any] _[ x][ ∈]_ _[V]_ [ .] [The conventional maximum norm for]
R _[n]_, of course, is defined with respect to the standard orthonormal basis _{ej}_ _[n]_ _j_ =1 [.] [Since] _[ |⟨][v][i][, x][⟩| ≤||][x][||][∞][,V]_ [, for all] _[ i]_ [,]



_||x||_ [2] 2 [=]



_d_ _√_
�( _⟨vi, x⟩_ ) [2] _≤_ _d||x||_ [2] _∞,V_ [=] _[⇒||][x][||]_ [2] _[≤]_ _d||x||∞,V ._

_i_ =1



_√_
Then, for any _x_ _∈_ _V_ _⊆_ R _[n]_, _||x||∞_ _≤_



_√_
_d||x||∞,V_, since _|⟨x, ej⟩|_ _≤||x||_ 2 _≤_



Then, for any _x_ _∈_ _V_ _⊆_ R _[n]_, _||x||∞_ _≤_ _d||x||∞,V_, since _|⟨x, ej⟩|_ _≤||x||_ 2 _≤_ _d||x||∞,V_, for all _j_ _∈{_ 1 _. . ., n}_ . Thus,

controlling component-wise error in any orthonormal basis in _V_ controls component-wise error in R _[n]_, in an _n_ -independent
but _d_ -dependent manner. In the following, we give a flavor of how we can analyze finite sample estimate errors in _V_ . The
maximum norm _|| · ||∞_ is to be understood as _|| · ||∞,V_ for some orthonormal basis choice. Here is the result relevant to the
linear case described Subsubsection 2.2.1.



**Proposition 5.** _Let Xt_ _i.i.d∼N_ (0 _, σ_ 0 [2] _[I][d]_ [)] _[, t]_ [ = 1] _[, . . ., L][ and let]_ [ˆΠ :=] _σ_ 0 [2] 1 _[L]_ - _Lt_ =1 _[X][t][X]_ _t_ _[T]_ _[.][ Then, for any][ δ]_ _[∈]_ [(0] _[,]_ [ 1)] _[,]_



_δ_ [)]
_L_




- [�]
_d_ + ln( [2]




[2] [2]

_δ_ [)] _,_ _[d]_ [ + ln(] _δ_
_L_ _L_



��



_Pr_





_||_ Π˜ [ˆ] _x −_ _x_ ˜ _||∞_ _< C||x_ ˜ _||_ 2 max



_for some C_ _>_ 0 _._


_Proof._ We start by bounding the maximum norm of the difference,


_||_ Π˜ [ˆ] _x −_ _x_ ˜ _||∞_ _≤||_ Π˜ [ˆ] _x −_ _x_ ˜ _||_ 2 _≤||_ Π [ˆ] _−_ _Id||_ op _||x||_ 2 _,_


where _|| · ||_ op is the operator norm.


It can be shown that, for any _δ_ _∈_ (0 _,_ 1)



_δ_ [)]
_L_




_>_ 1 _−_ _δ_


_>_ 1 _−_ _δ_







��




- [�]
_d_ + ln( [2]




[2] [2]

_δ_ [)] _,_ _[d]_ [ + ln(] _δ_
_L_ _L_



Pr



_||_ Π [ˆ] _−_ _Id||_ op _< C_ max



for some _C_ _>_ 0 (Rigollet & H¨utter, 2023). Combining with the first bound, we get our result.



_L_ [1] - _Lt_ =1 _[X][t][e][⟨][X][t][,][x]_ [˜] _[∥][⟩][/σ]_ _Z_ [2]



As to the nonlinear cases, the key result of Theorem 3.1 is the convergence of the numerator [1]



to E[ _Xe_ _[⟨][X,][x]_ [˜] _[∥][⟩][/σ]_ _Z_ [2] ] = - _e_ _[⟨][x,][x]_ [˜] _[∥][⟩][/σ]_ _Z_ [2] _x dpX_ ( _x_ ) and the denominator _L_ 1 - _Lt_ =1 _[e][⟨][X][t][,][x]_ [˜] _[∥][⟩][/σ]_ _Z_ [2] to E[ _e_ _[⟨][X,][x]_ [˜] _[∥][⟩][/σ]_ _Z_ [2] ] =

- _e_ _[⟨][x,][x]_ [˜] _[∥][⟩][/σ]_ _Z_ [2] _dpX_ ( _x_ ).


In the following, we assume that the support of _pX_ is inside a vector space _V_ whose dimension we denote by _d_ (instead of
_d_ + 1, as in the sphere problem). In addition, we refer to the projection of the query on _V_ by _x_ ˜ _∈_ _V_, instead of _x_ ˜ _∥_ . As usual,
the maximum norm in _V_ is with respect to some orthonormal basis choice


17


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**


_i.i.d_
**Proposition 6.** _Let Xt_ _∼_ _pX_ _and ||Xt||_ 2 _≤_ _R for t_ = 1 _, . . ., L._


_Then, for any δ_ _∈_ (0 _,_ 1) _,_




- _L_ _e_ _[⟨][X][t][,][x]_ [˜] _[⟩][/σ]_ _Z_ [2] _−_ E[ _e_ _[⟨][X,][x]_ [˜] _[⟩][/σ]_ _Z_ [2] ] _<_ sinh - _R||x_ ˜ _||_ 2

_t_ =1 ��� _σZ_ [2]




- [�] 2 - 2
_L_ [ln] _δ_




- [�]



_Pr_



1
���� _L_



_and_



_≥_ 1 _−_ _δ_


- [�]

_≥_ 1 _−_ _δ._



1

_Pr_

������� _L_



_L_

- _Xte_ _[⟨][X][t][,][x]_ [˜] _[⟩][/σ]_ _Z_ [2] _−_ E[ _Xe_ _[⟨][X,][x]_ [˜] _[⟩][/σ]_ _Z_ [2] ]

������ _∞_ _[< Re]_
_t_ =1



_R||x_ ˜ _||_ 2

_σZ_ [2]




2  - 2 _d_
_L_ [ln] _δ_



_Proof._ We provide the sketch of our proof here, the key ingredient of which is the Hoeffding inequality (Hoeffding, 1994).



_L_ [1] - _Lt_ =1 _[e][⟨][X][t][,][x]_ [˜] _[⟩][/σ]_ _Z_ [2], each term in the sum is bounded above and below by _e±_ _[R][||]_ _σ_ _[x]_ [˜] _Z_ [2] _[||]_ [2]



For the average _L_ [1] - _Lt_ =1 _[e][⟨][X][t][,][x]_ [˜] _[⟩][/σ]_ _Z_, each term in the sum is bounded above and below by _e_ _σZ_ [2] . So, the Hoeffding

inequality leads to



For the average [1]



_L_

- _e_ _[⟨][X][t][,][x]_ [˜] _[⟩][/σ]_ _Z_ [2] _−_ E[ _e_ _[⟨][X,][x]_ [˜] _[⟩][/σ]_ _Z_ [2] ] _≥_ _ϵ_ ] _≤_ 2 exp

���
_t_ =1







_Lϵ_ [2]

_−_



2 sinh [2][ �] _[R][||][x]_ [˜] _[||]_ [2]

[2]



2 _Lϵ_ [2]

_−_

 -  - _R||x_ ˜ _||_ 2  -  -  - [�][2]
exp _−_ exp _−_ _[R][||][x]_ [˜] _[||]_ [2]
_σZ_ [2] _σZ_ [2]








_[x]_ [˜] [2]  
_σZ_ [2]





= 2 exp



Pr



1
���� _L_





_._




        - _Lϵ_ [2]
Setting _δ_ = 2 exp _−_




   -   - _R||x_ ˜ _||_ 2   - [�] 2   - 2   
_[||][x]_ [˜] _[||]_ [2] -, we get _ϵ_ = sinh _σZ_ [2] _L_ [ln] _δ_, which gives our first probabilistic inequality.

_σZ_ [2]



2 sinh [2][ �] _[R][||][x]_ [˜] _[||]_ [2]

[2]



For each component of the vector average [1]



_R||x_ ˜ _||_ 2



_L_ [1] - _Lt_ =1 _[X][t][e][⟨][X][t][,][x]_ [˜] _[⟩][/σ]_ _Z_ [2], the terms in the sum are bounded above and below



by _±R_ _σZ_ [2] . We use similar arguments involving the Hoeffding inequality, combined with the union bound over all _d_

coordinates



by _±R_



1

Pr

������� _L_



_L_

- _Xte_ _[⟨][X][t][,][x]_ [˜] _[⟩][/σ]_ _Z_ [2] _−_ E[ _Xe_ _[⟨][X,][x]_ [˜] _[⟩][/σ]_ _Z_ [2] ]

������ _∞_ _[≥]_ _[ϵ]_ []] _[ ≤]_ [2] _[d]_ [ exp]
_t_ =1



_Lϵ_ [2]

_−_

2 _R_ [2] exp  - 2 _R||x_ ˜ _||_ 2  _σZ_ [2]









_._



Once more, setting the RHS to _δ_ and solving for _ϵ_, we get our second probabilistic inequality.


**F. Limiting behaviors of the softmax function and softmax attention**


**For small argument**


A Taylor expansion of the softmax function at zero gives


softmax( _βv_ ) = [1]                   - 1 _L_ + _βv_ + _O_ ( _β_ [2] )� _,_

_Z_



_i_ �1 + _βvi_ + _O_ ( _β_ [2] ))� = _L_ (1 + _βv_ ¯ + _O_ ( _β_ [2] )) is a normalizing factor, with ¯ _v_ = _L_ [1]



where _Z_ = [�]




[1] 
_L_



where _Z_ = [�] _i_ 1 + _βvi_ + _O_ ( _β_ )) = _L_ (1 + _βv_ ¯ + _O_ ( _β_ )) is a normalizing factor, with ¯ _v_ = _L_ - _i_ _[v][i]_ [.] [The notation] [ 1] _[L]_

stands for an _L_ -dimensional vector of ones.



Thus, we have


**Lemma F.1** (Small argument expansion of softmax) **.** _As β_ _→_ 0 _,_


softmax( _βv_ ) = 1      - 1 _L_ + _βv_ + _O_ ( _β_ [2] )� = [1]      - 1 _L_ + _β_ ( _v −_ _v_ ¯ 1 ) + _O_ ( _β_ [2] )� _._
_L_ (1 + _βv_ ¯ + _O_ ( _β_ [2] )) _L_


**F.1. Proof of Proposition 3.2**


_Proof._

          -          - 1          - [�]
_F_ _E,_ := [1] 1: _L_ _[W][KQ][X]_ [˜] [)] _[.]_
_ϵ_ _[W][P V][, ϵW][KQ]_ _ϵ_ _[W][P V][ X]_ [1:] _[L]_ [softmax(] _[ϵX]_ _[T]_


18


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**


Using Lemma F.1, as _ϵ →_ 0,







_L_




 - - 1 - [�]
_F_ _E,_ = [1]
_ϵ_ _[W][P V][, ϵW][KQ]_ _ϵ_ _[W][P V][ X]_ [1:] _[L]_




1

_L_



1 _L_ + _ϵ_ - _X_ 1: _[T]_ _L_ _[W][KQ][X]_ [˜] _[−]_ [(] [1]



��

- _Xt_ _[T]_ _[W][KQ][X]_ [˜] [)] [1] _[L]_ - + _O_ ( _ϵ_ [2] )

_t_




[1] _[X]_ [¯] [+] [1]

_ϵ_ _[W][P V]_ _L_



= [1]



_L_ _[W][P V]_



_L_

- _Xt_ ( _Xt −_ _X_ [¯] ) _[T]_ _WKQX_ [˜] + _O_ ( _ϵ_ ) _,_ (A.11)


_t_ =1



where _X_ [¯] = _L_ 1 - _Lt_ =1 _[X][t]_ [is] [the] [empirical] [mean] [and] [the] [notation] [1] _[L]_ [emphasizes] [that] [it] [is] [a] [column] [vector] [of] [ones] [with]
dimension _L_ .


**For large argument**


As _β_ _→∞_, the softmax function simply selects the maximum over its inputs (as long as the the maximum is unique):



softmax( _βv_ ) _≈_




1 if _i_ = arg max _j vj,_
0 otherwise _._



In this case, all attention weight is given to a single element, and the others are effectively ignored.


**G. MSE Loss landscape for scaled identity weights**


(a) MSE loss landscape for Fig. 3 (Case 2) (b) MSE loss landscape for Fig. 3 (Case 3)

|heuristic_KQ<br>heuristic_PV<br>trained model<br>heuristic (theory)<br>2D scan min|heuristic_KQ<br>heuristic_PV<br>trained model<br>heuristic (theory)<br>2D scan min|Col3|Col4|Col5|
|---|---|---|---|---|
|heuristic_KQ<br>heuristic_PV<br>trained model<br>heuristic (theory)<br>2D scan min|||||
|heuristic_KQ<br>heuristic_PV<br>trained model<br>heuristic (theory)<br>2D scan min|||||



_Figure 6._ Loss landscape corresponding to Case 2 and Case 3 of Fig. 3. The MSE is numerically evaluated by assuming scaled identity
weights _WKQ_ = _βIn_ (x-axis) and _WP V_ = _αIn_ (y-axis) and scanning over a 50 _×_ 50 grid. The green point corresponds to the heuristic
minimizer identified from the posterior mean. In Case 2 it is exact, while in case 3 it is an approximation that neglects the residual term
(see Proposition 4). The orange point corresponds to the learned weights displayed in Fig. 3(b), while the white point corresponds to the
numerically identified minimum from this 2D scan. These can fluctuate due to the finite context ( _L_ = 500) and sampling ( _N_ = 800 here).
In both panels, it is apparent that the trained weights and the heuristic estimator co-occur in a broad valley (contour) of the loss landscape.


The loss landscapes in Fig. 6 exhibit large, low-cost valleys with a roughly hyperbolic structure that is especially apparent in
Case 2. This indicates a multiplicative tradeoff in the scales of _WKQ_ and _WP V_, which suggests that linear attention might
be applicable here as well. For completeness, Figure 7 shows linear attention performance for both cases, demonstrating that
it performs quite similarly to softmax for sub-sphere denoising, but less well in the Gaussian mixtures case.


**H. Structured optimal weights under prompt transformation**


We find that one-layer transformers can learn to undo arbitrary invertible coordinate transformations that warp the denoising
tasks. Focusing on the subspace denoising case, suppose each prompt is transformed by a fixed invertible square matrix


19


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**


(a) Case 2: Nonlinear manifolds Case 3: Gaussian mixtures





Epoch Epoch

(b) Final weights: linear Final weights: softmax Final weights: linear Final weights: softmax


_Figure 7._ Linear attention performance for Cases 2 and 3. Additional empirical results for the nonlinear manifolds case (left) and the
Gaussian mixtures case (right). (a) Loss dynamics for randomly initialized softmax and linear attention layers. Solid lines represent the
average loss over six seeds, with shaded area indicating the range. Training details and parameters follow Fig. 3(a). (b) Representative
final attention weights for each layer.


_A_, i.e. _E_ = ( _X_ 1: _L,_ ˜ _x_ ) _→_ _E_ _[′]_ = ( _AX_ 1: _L, Ax_ ˜). If the target remains _xL_ +1 in the untransformed space, then the optimal
attention weights are no longer diagonal, but instead take a structured form determined by the transformation matrix:


_WP V_ = _αA_ _[−]_ [1] _,_ _WKQ_ = _β_ ( _AA_ _[T]_ ) _[−]_ [1] _,_ (A.12)

where _αβ_ = _σ_ 0 [2][+] 1 _[σ]_ _Z_ [2] [as before.]


(a)


_Figure 8._ (a) Example transformation _A_ used to globally alter the in-context denoising prompts. (b) Structure of the optimal attention
weights for this transformed subspace-denoising task. (c,d) Empirically, we find that both linear attention and softmax attention layers are
able to learn these structured targets, but with distinct scalings _α, β_ . Final weights after 500 epochs using Adam, random initializations,
and context length _L_ = 500; other parameters follow Fig. 3(a).


Notably, we find that both the linear and softmax attention layers are able to learn these structures; see Fig. 8 for an example.


20


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**


We use the same basic training procedure as the limiting case of _A_ = _I_ (no additional coordinate transformation) assumed
throughout the main text.

Suppose we still work with transformed coordinates _Yt_ = _AXt_ and _Y_ [˜] = _AX_ [˜], but now intend to retrieve _YL_ +1 = _AXL_ +1
in the new coordinate space (rather than _XL_ +1 as above). In this case, we would be dealing with variables with covariance
matrices Σ _∝_ _AA_ _[T]_ . We would need weight matrices that are not simply proportional to identity to deal with the covariance
structure. This is also the case for in-context learning of linear functions when the input has an anisotropic covariance matrix
(Zhang et al., 2024; Ahn et al., 2023). Recall in the original setting, we had the sample covariance E[ _XX_ _[T]_ ] _≡_ Σ _X_ = _σ_ 0 [2] _[P]_
and noise Σ _Z_ _≡_ _σZ_ [2] _[I]_ [, leading to the estimator, Eq.] [(][10][):] _[X]_ [ˆ] [=] ( _σ_ 0 [2][+] 1 _[σ]_ _Z_ [2] [)] _[L]_ - _Lt_ =1 _[X][t][⟨][X][t][,]_ _[X]_ [˜] _[⟩]_ [.] [Here, the sample covariance]

is Σ _Y_ _≡_ _σ_ 0 [2] _[APA][T]_ [,] [and] [the] [noise] _[V]_ _[≡]_ _[AZ]_ [has] [covariance] [Σ] _[V]_ _[≡]_ _[σ]_ _Z_ [2] _[AA][T]_ [ .] [One] [can] [show] [the] [generalized] [solution] [is]
_Y_ ˆ = Σ _Y_ (Σ _Y_ + Σ _V_ ) _[−]_ [1] _Y_ ˜ . Thus, in the transformed coordinates, the denoising estimate is



_L_

_Y_ ˆ = 1 - _Yt⟨A_ _[−]_ [1] _Yt, A_ _[−]_ [1][ ˜] _Y ⟩._ (A.13)
( _σ_ 0 [2] [+] _[ σ]_ _Z_ [2] [)] _[L]_ _t_ =1



For the relationship of this denoising result in _Y_ to energy models, as discussed in Section 4 and Subsection I.1, we need a
modified energy _E_ ( _Y_ 1: _L, s_ ) = 21 _γ_ _[∥][s][∥]_ [2] _[ −]_ 21 _L_ - _Lt_ =1 _[⟨][A][−]_ [1] _[Y][t][, A][−]_ [1] _[s][⟩]_ [2][ and a preconditioner proportional to] _[ AA][T]_ [ .]


**I. Additional comments on the mapping from attention to associative memory models**


**I.1. Linear attention and traditional Hopfield model**


We have considered a trained network with linear attention, relating the query _X_ [˜] and the estimate of the target _X_ [ˆ], of the
form



_X_ ˆ = _f_ ( ˜ _X_ ) := _[γ]_

_L_



_L_





- _Xt⟨Xt,_ _X_ [˜] _⟩_ (A.14)


_t_ =1



with _γ_ = _σ_ 0 [2][+] 1 _[σ]_ _Z_ [2] [.]


With




[1]

2 _γ_ _[∥][s][∥]_ [2] _[ −]_ 2 [1]



_L_





- _XtXt_ _[T]_ [)] _[s]_ (A.15)

_t_ =1



_E_ ( _X_ 1: _L, s_ ) := [1]



2 _L_ _[s][T]_ [ (]



gradient descent iteration _s_ ( _t_ + 1) = _s_ ( _t_ ) _−_ _γ_ _∇sE_ - _X_ 1: _L, s_ ( _t_ )� gives us



_s_ ( _t_ + 1) = _[γ]_

_L_


making the one-step iteration our denoising operation.




- _Xt⟨Xt, s_ ( _t_ ) _⟩_


_t_



We will call this energy function the Naive Spherical Hopfield model for the following reason. For random memory patterns
_X_ 1: _L_, and the query denoting Ising spins _s ∈{−_ 1 _,_ 1 _}_ _[n]_, the so-called Hopfield energy is



_E_ Hopfield( _X_ 1: _L, s_ ) := _−_ [1]



_L_





- _XtXt_ _[T]_ [)] _[s.]_ (A.16)

_t_ =1



2 _L_ _[s][T]_ [ (]



We could relax the Ising nature of the spins by letting _s_ _∈_ R _[n]_, with a constraint _||s||_ [2] = _n_ . This is the spherical model
(Fischer & Hertz, 1993) since the spin vector _s_ lives on a sphere. If we minimize this energy the optimal _s_ would be
aligned with the dominant eigenvector of the matrix _L_ [1] [(][�] _t_ _[L]_ =1 _[X][t][X]_ _t_ _[T]_ [)][ (][Fischer & Hertz][,][ 1993][), and the model will not have]

a retrieval phase (see Bolle et al.´ (2003) for a similar model that does). A soft-constrained variant can also be found in
Section 3.3, Model C of Krotov & Hopfield (2021).


We could reformulate the optimization problem of minimizing the Hopfield energy, subject to _||s||_ [2] = _R_ [2], as




- max - _−_ [1]
_λ_ 2



_L_





- - [�]

_XtXt_ _[T]_ [)] _[s]_ [ +] _[ λ]_ [(] _[s][T][ s][ −]_ _[R]_ [2][)] _._
_t_ =1



arg min
_s∈_ R _[n]_



2 _L_ _[s][T]_ [ (]



21


**In-Context Denoising with One-Layer Transformers:** **Connections between Attention and Associative Memory Retrieval**


The _s_ -dependent part of the Lagrangian, with _λ_ replaced by 21 _γ_ [gives us the energy function in Eq.] [A.15][ which we have]
called the Naive Spherical Hopfield model.




[1] ( _σ_ 0 [2] [+] _[ σ]_ _Z_ [2] [)] _[I][n]_ _[−]_ [1]

2 _[s][T]_ [ �] _L_



_L_ [(]



_E_ ( _X_ 1: _L, s_ ) := [1]




[1]

2 _γ_ _[∥][s][∥]_ [2] _[ −]_ 2 [1]



2 _L_ _[s][T]_ [ (]



_L_





- _XtXt_ _[T]_ [)] _[s]_ [ =] [1]

2

_t_ =1



_L_





- _XtXt_ _[T]_ [)] - _s._ (A.17)

_t_ =1



For _L_ much larger than _n_, _L_ [1] - _Lt_ =1 _[X][t][X]_ _t_ _[T]_ _[≈]_ _[σ]_ 0 [2] _[P]_ [, so its eigenvalues are either 0 or are very close to] _[ σ]_ 0 [2][.] [Hence, for large] _[ L]_

and _σZ_ _>_ 0, this quadratic function is very likely to be positive definite. One-step gradient descent brings _s_ down to the
_d_ -dimensional linear subspace _S_ spanned by the patterns, but repeated gradient descent steps would take _s_ towards zero.


**I.2. Remarks on the softmax attention case (mapping to dense associative memory networks)**


Regarding the mapping discussed in the main text, we note that there is a symmetry condition on the weights _WKQ, WP V_
that is necessary for the softmax update to be interpreted as a gradient descent (i.e. a conservative flow). In general, a flow
_ds/dt_ = _f_ ( _s_ ) is conservative if it can be written as the gradient of a potential, i.e. _f_ ( _s_ ) = _−∇sV_ ( _s_ ) for some _V_ . For this
to hold, the Jacobian of the dynamics _Jf_ ( _s_ ) = _∇sf_ must be symmetric.


The softmax layer studied in the main text is _f_ ( _s_ ) = _WP V X_ softmax( _X_ _[T]_ _WKQs_ ). We will denote _z_ ( _s_ ) = _X_ _[T]_ _WKQ s_ and
_g_ ( _s_ ) = softmax( _z_ ( _s_ )), both in R _[L]_ . Then the Jacobian is


_J_ ( _s_ ) = _WP V X_ _[∂g]_ �diag( _g_ ) _−_ _gg_ _[T]_ [ �] _X_ _[T]_ _WKQ._ (A.18)

_∂s_ [=] _[ W][P V][ X]_


Observe that _Y_ = _X_ �diag( _g_ ) _−_ _gg_ _[T]_ [ �] _X_ _[T]_ is symmetric (keeping in mind that _g_ ( _s_ ) depends on _WKQ_ ). The Jacobian
symmetry requirement _J_ = _J_ _[T]_ therefore places the following constraint on feasible _WKQ, WP V_ :


_WP V_ _Y_ _WKQ_ _[T]_ [=] _[ W][KQ]_ _[Y]_ _[W]_ _P V_ _[ T]_ _[.]_ (A.19)


It is clear that this condition holds for the scaled identity attention weights discussed in the main text. Potentially, it could
allow for more general weights that might arise from non-isotropic denoising tasks to be cast as gradient descent updates.


The mapping discussed in the main text involves discrete gradient descent steps, Eq. (17). In general, this update rule
retains a “residual” term in _s_ ( _t_ ) if we choose a different descent step size _γ_ = _α_ . Thus, taking _K_ recurrent updates could
be viewed as the depthwise propagation of query updates through a _K_ -layer architecture if one were to use tied weights.
Analogous residual streams are commonly utilized in more elaborate transformer architectures to help propagate information
to downstream attention heads.


22


