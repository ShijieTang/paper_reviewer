## Learning with Expected Signatures: Theory and Applications

Lorenzo Lucchese <sup>1</sup> Mikko S. Pakkanen <sup>1</sup> Almut E. D. Veraart <sup>1</sup>

#### Abstract

The expected signature maps a collection of data streams to a lower dimensional representation, with a remarkable property: the resulting feature tensor can fully characterize the data generating distribution. This "model-free" embedding has been successfully leveraged to build multiple domain-agnostic machine learning (ML) algorithms for time series and sequential data. The convergence results proved in this paper bridge the gap between the expected signature's empirical discrete-time estimator and its theoretical continuous-time value, allowing for a more complete probabilistic interpretation of expected signature-based ML methods. Moreover, when the data generating process is a martingale, we suggest a simple modification of the expected signature estimator with significantly lower mean squared error and empirically demonstrate how it can be effectively applied to improve predictive performance.

#### 1. Introduction

The signature transform of a stream of data is an infinite but countable sequence of its "iterated integrals" summarizing the input in a top-down fashion, meaning the informational content of its terms decays factorially. Originally introduced by [Chen](#page-9-0) [\(1954\)](#page-9-0) and serving as a fundamental object of rough path analysis [\(Lyons et al.,](#page-10-0) [2007\)](#page-10-0), the signature

$$\mathbb{S} = \{S(\mathbb{X})_{[0,t]} \in T((\mathbb{R}^d)), \ t \in [0, T]\},$$

of a path <sup>X</sup> <sup>=</sup> {<sup>X</sup>t, t ∈ [0, T]} ∈ <sup>C</sup>([0, T], <sup>R</sup> d ) is a lift (in the sense that it embeds X) to the space of continuous functions over the tensor algebra T((<sup>R</sup> d )) possessing some nice algebraic and geometric properties. When the path is of bounded variation, the signature is defined as the sequence of iterated integrals of <sup>X</sup>, i.e. for <sup>t</sup> ∈ [0, T], k ≥ <sup>0</sup>

$$S^k(\mathbb{X})_{[0,t]} = \int_0^{\infty} \cdots \int_{0 \leq s_1 \leq \dots \leq s_k \leq t} d\mathbf{X}_{s_1} \otimes \cdots \otimes d\mathbf{X}_{s_k}. \quad (1)$$

In many practical applications the path X is taken to be the piecewise linear interpolation of a discrete-time stream of data, which is of bounded variation by construction. Signature-based machine learning (ML) approaches [\(Lyons](#page-10-1) [& McLeod,](#page-10-1) [2024\)](#page-10-1) thus often restrict the theoretical framework to paths in BV([0, T], <sup>R</sup> d ). In this setting, two fundamental properties of the signature that make it a desirable non-parametric feature extraction method for sequential data are the characterization result of [Hambly & Lyons](#page-9-1) [\(2005\)](#page-9-1) and the universality approximation theorem of [Levin](#page-10-2) [et al.](#page-10-2) [\(2016\)](#page-10-2). Moreover, when the path X is understood as a (realization of a) random process with distribution P over BV([0, T], <sup>R</sup> d ), the shuffle property of the signature implies that all moments of the random variable S(X)[0,T] are determined by its expectation

$$\phi(T) := \mathbb{E}[S(\mathbb{X})_{[0,T]}] \in T((\mathbb{R}^d)).$$

A natural question, known as the Hamburger moment problem [\(Fawcett,](#page-9-2) [2003\)](#page-9-2), is thus whether the expectation of the signature characterizes its law (and thus the law of the path). When imposing a probability distribution P on X the assumption of bounded variation paths becomes quite restrictive: Brownian motion, the basic building block of many stochastic models, has paths of infinite variation almost surely. Even if we observe a discrete-time stream of data, we often still would like to define the process X as a latent stochastic process of which we observe the linear interpolation over some partition π of [0, T], hereafter denoted by X π . We hence wish to make sense of the signature of a stochastic process X with paths of unbounded variation. For a given path <sup>X</sup> ∈ <sup>C</sup>([0, T], <sup>R</sup> d ) of finite p-variation, once we "lift" the process to a p-rough path [\(Lyons et al.,](#page-10-0) [2007,](#page-10-0) Definition 3.11) then the signature S of X is uniquely defined<sup>1</sup> . Without delving into the details of rough path theory, for our purposes it suffices to interpret the choice of lift as fixing a notion of integration with respect to X: the higher order signatures terms are then understood as iterated integrals of the path X defined in this sense.

<sup>1</sup>Department of Mathematics, Imperial College London, London, United Kingdom. Correspondence to: Lorenzo Lucchese <lorenzo.lucchese17@imperial.ac.uk, llucchese6@gmail.com>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

<sup>1</sup>This is the first fundamental theorem in the theory of rough paths [\(Lyons et al.,](#page-10-0) [2007,](#page-10-0) Theorem 3.7).

Motivated by the fact that we can only ever observe the process <sup>X</sup> over a discrete partition π of [0, T] we restrict our attention to the class of stochastic processes whose lift (and hence signature) can be approximated by the lift (and hence signature) of the bounded variation path X π . Following the rough path literature we take such approximation in the pvariation metric to define the notion of *canonical geometric stochastic process*, cf. Definition [2.1.](#page-1-0) In [Chevyrev & Lyons](#page-9-3) [\(2016\)](#page-9-3); [Chevyrev & Oberhauser](#page-9-4) [\(2018\)](#page-9-4) the authors provide characterization results for the expected signature of canonical geometric stochastic processes, i.e. conditions under which the map <sup>P</sup> 7→ <sup>E</sup>[S(X)[0,T] ] is injective. Such characterizing property of the expected signature has found practical use in a wide range of applications, ranging from classic ML tasks [\(Lemercier et al.,](#page-9-5) [2021;](#page-9-5) [Triggiano & Romito,](#page-10-3) [2024;](#page-10-3) [Schell & Oberhauser,](#page-10-4) [2023\)](#page-10-4) to mathematical finance [\(Lyons et al.,](#page-10-5) [2021;](#page-10-5) [Futter et al.,](#page-9-6) [2023\)](#page-9-6).

![](_page_1_Figure_3.jpeg)

The expected signature is thus a highly informative quantity and, consequently, methods for computing ϕ(T) have received considerable research interest. Such methods can be broadly categorized into two classes: those employing an analytical approach and those following a statistical one. Analytical methods aim to develop exact formulas for specific classes of models. A first step in this direction was taken in [Ni](#page-10-6) [\(2012,](#page-10-6) Section 4) showing that the expected signature of an Ito diffusion satisfies an explicit partial differential ˆ equation (PDE). This result was subsequently generalized in [Cuchiero et al.](#page-9-7) [\(2023\)](#page-9-7) to the class of signature-SDEs and in [Friz et al.](#page-9-8) [\(2022;](#page-9-8) [2024\)](#page-9-9) to (discontinuous) semimartingales. On the other hand, the statistical approach aims to estimate ϕ(T) directly from observed data, preserving the model-free nature of the expected signature. For a given set of observations X 1,π , . . . , X N,π one can form the estimator

$$\hat{\phi}^{N,\pi}(T) := \frac{1}{N} \sum_{n=1}^N S(\mathbb{X}^{n,\pi})_{[0,T]},$$

as illustrated in Figure [1,](#page-1-1) and study its in-fill |π| → <sup>0</sup> and large-sample <sup>N</sup> → ∞ asymptotics. This line of work includes the explicit results of [Ni](#page-10-6) [\(2012,](#page-10-6) Section 3.2) for Brownian motion and of [Passeggeri](#page-10-7) [\(2020\)](#page-10-7) for fractional Brownian motion with Hurst parameter H > 1/2 as well as the preliminary results in [Friz & Victoir](#page-9-10) [\(2010\)](#page-9-10) for more general semimartingales. Additionally, [Schell & Oberhauser](#page-10-4) [\(2023,](#page-10-4) Section 8) develops asymptotic results for processes of bounded variation. In this work we provide a unifying set of general conditions under which the expected signature estimator ϕˆN,π(T) displays important asymptotic statistical properties, namely consistency and asymptotic normality. Our results allow for irregular[<sup>2</sup>](#page-0-0) observation partitions π – possibly varying across samples – and for dependency

across the samples X 1,π , . . . , X n,π. The first main contribution of this paper is thus to bridge the gap between the empirical expected signature estimator and the expected signature of a latent continuous-time stochastic process, unlocking a more general probabilistic interpretation of several ML algorithms and effectively moving beyond the expected signature as a simple feature extraction method. This naturally leads to the second theoretical contribution: by starting from the continuous-time setting we devise a modification of the expected signature estimator with significantly better finite sample properties when the latent data generating process is a martingale. The superior performance of this modified estimator is empirically verified through various experiments with expected signature-based ML algorithms from the literature.

Figure 1. Estimating the expected signature estimation from a finite collection of discretely-observed paths.

#### 2. Theory

Let <sup>X</sup> <sup>=</sup> {<sup>X</sup>t, t ∈ [0, T]} denote a <sup>d</sup>-dimensional stochastic process over the probability space (Ω, F, <sup>P</sup>).

Definition 2.1. We say X is a canonical geometric stochastic process of rough order p if there exists a sequence of partitions <sup>ρ</sup> with |ρ| → <sup>0</sup> such that the limit in the <sup>p</sup>-variation metric of the canonically lifted linearly interpolated process X ρ exists in probability. Convergence in probability implies almost sure convergence (along a subsequence) and hence we can almost surely define the lift of X as such limit.

*Remark* 2.2*.* The definition of lift suggests this might depend on the choice of the sequence of partitions ρ. In any case, for a wide range of stochastic processes there exist *canonical* lifts that satisfy our definition of canonical geometric rough path. These include:

- Semimartingales: For <sup>p</sup> ∈ (2, 3) any semimartingale can be lifted to a geometric p-rough path by defining the lift via Stratonovich integration; the signature of X then coincides with iterated Stratonovich integrals. For any sequence of partitions ρ the lifts of the linear

<sup>2</sup>Clearly, for the estimation problem to be well-posed, the sequence of partitions needs to be signature defining in the sense of Definition [2.5.](#page-2-0)

interpolations converge in p-variation metric to the Stratonovich lift [\(Friz & Victoir,](#page-9-10) [2010,](#page-9-10) Chapter 14) and hence X is a canonical geometric stochastic process in the sense of Definition [2.1.](#page-1-0)

- Gaussian processes: Many Gaussian processes admit canonical lifts to geometric p-rough paths [\(Friz](#page-9-10) [& Victoir](#page-9-10) [\(2010,](#page-9-10) Theorem 15.34, Definition 15.35) and [Coutin & Qian](#page-9-11) [\(2002\)](#page-9-11)) with the existence criterion for such canonical lifts easily stated in terms of the covariance function. The definition of the lift implicitly requires ρ to be any sequence such that X ρ converges uniformly to X almost surely. For example, fractional Brownian motion with Hurst parameter H > 1/4 can be lifted to a geometric p-rough path with p > 1/H by choosing ρ to be the sequence of dyadic partitions.

In what follows, we will assume the canonical geometric stochastic process X has a *canonical* lift (i.e. a canonical sequence of partitions ρ along which the lift is defined) and unambiguously refer to it as *the* lift of X.

Let <sup>ρ</sup> denote a partition of [0, T] with mesh |ρ| and <sup>X</sup> <sup>ρ</sup> = {X ρ t , t ∈ [0, T]} the linear approximation of <sup>X</sup> over <sup>ρ</sup>, i.e.

$$\mathbf{X}_t^\rho = \mathbf{X}_u + \frac{t-u}{v-u} \mathbf{X}_{u,v}, \quad t \in [u, v] \in \rho,$$

with <sup>X</sup>u,v <sup>=</sup> <sup>X</sup><sup>v</sup> − <sup>X</sup>u. The signature of the bounded variation path X <sup>ρ</sup> up to time <sup>t</sup> ∈ [0, T] is defined by Equation [\(1\)](#page-0-1) through classic Riemann-Stieltjes integration and can thus be computed by

$$S(\mathbb{X}^\rho)_{[0,t]} = \bigotimes_{[u,v] \in \rho_{[0,t]}} \exp_{\otimes} \mathbf{X}_{u,v}^\rho, \quad (2)$$

where ρ[0,t] denotes the restriction of ρ to [0, t]. The canonical lift of X ρ to a (geometric) p-rough path is

$$\left(1, S^1(\mathbb{X}^\rho)_{[0,t]}, \dots, S^{\lfloor p \rfloor}(\mathbb{X}^\rho)_{[0,t]}\right) \in T^{\lfloor p \rfloor}((\mathbb{R}^d)), \quad (3)$$

for <sup>t</sup> ∈ [0, T]. Definition [2.1](#page-1-0) requires that there exists a sequence of partitions ρ for which this sequence of geometric p-rough paths converges in probability in the p-variation metric. A key result from rough path theory is that a geometric p-rough path has a full signature. Fixing the lift of X via Definition [2.1,](#page-1-0) we thus have a uniquely specified signature for X.

Definition 2.3. The signature of a canonical geometric stochastic process X,

$$\mathbb{S} = \{S(\mathbb{X})_{[0,t]} \in T((\mathbb{R}^d)), \ t \in [0, T]\},$$

is defined pathwise (on a set of full measure) as the unique extension of the lift of X to a multiplicative functional of arbitrary order in the sense of [\(Lyons et al.,](#page-10-0) [2007,](#page-10-0) Theorem 3.7). The elements of the signature are the rough iterated integrals of X.

*Remark* 2.4*.* Taking ρ to be a sequence such that Definition [2.1](#page-1-0) holds, by continuity of the extension map [\(Lyons](#page-10-0) [et al.,](#page-10-0) [2007,](#page-10-0) Theorem 3.10), it immediately follows that the signature of X ρ (truncated at level <sup>K</sup> ≥ ⌊p⌋) converges in probability to the signature of X (up to level K) in the p-variation topology. In particular, this implies that, for any finite collection of words I,

$$S^{\mathbf{I}}(\mathbb{X}^\rho)_{[0,t]} \xrightarrow{\mathbb{P}} S^{\mathbf{I}}(\mathbb{X})_{[0,t]}. \quad (4)$$

Similar arguments imply that, when convergence to the lift along ρ holds almost surely in the p-variation metric, then also the higher order signature terms converge almost surely, and, in particular, [\(4\)](#page-2-1) holds in the almost sure limit.

In the following sections, we will be estimating the expected signature at fixed time horizon T > 0. To develop the properties of these estimators, it will be thus sufficient to work with pointwise limits like [\(4\)](#page-2-1) without having to deal with the stronger pathwise p-variation convergence used to define canonical geometric stochastic processes. This mode of convergence will thus be sufficient to consider a sequence of partitions as *signature-defining*.

Definition 2.5. Let <sup>X</sup> <sup>=</sup> {<sup>X</sup>t, t ∈ [0, T]} be a canonical geometric stochastic process, we say that a sequence of partitions <sup>π</sup> of the interval [s, t] ⊆ [0, T] with |π| → <sup>0</sup> is signature-defining if for any collection of words I,

$$S^{\mathbf{I}}(\mathbb{X}^{\pi})_{[s,t]} \xrightarrow{\mathbb{P}} S^{\mathbf{I}}(\mathbb{X})_{[s,t]}, \quad |\pi| \rightarrow 0. \quad (5)$$

#### 2.1. Expected Signature Estimation

In this section, we assume we have access to N copies of X discretely observed over possibly different partitions of the interval [0, T], i.e. each <sup>X</sup> n,πN,n is an observation over πN,n of a continuous-time latent process X <sup>n</sup>, for n = 1, . . . , N. We will focus on two observational schemes:

(ind) Repeatedly observe X through N independent experiments, in which case the "underlying" signatures S(<sup>X</sup> <sup>n</sup>)[0,T] , for n = 1, . . . , N, are independent and identically distributed.

(chop) Chop-up (and shift in time) a single observation of the process {<sup>X</sup>t, t ≥ <sup>0</sup>} over a partition

$$\Pi(N) := \pi_{N,1} \cup \cdots \cup ((N-1)T + \pi_{N,N}),$$

of [0, NT]. In this setting, we assume that the latent sequence {<sup>X</sup> <sup>n</sup>, n ≥ <sup>1</sup>} taking values in <sup>C</sup>([0, T]; <sup>R</sup> d ) is stationary, i.e. for <sup>k</sup> ∈ <sup>N</sup>, <sup>n</sup>1, . . . , n<sup>k</sup> ∈ <sup>N</sup> and <sup>n</sup> ≥ 0,

$$(\mathbb{X}^{n_1}, \dots, \mathbb{X}^{n_k}) \stackrel{\mathcal{L}}{=} (\mathbb{X}^{n_1+n}, \dots, \mathbb{X}^{n_k+n}), \quad (6)$$

and hence the signatures S(<sup>X</sup> <sup>n</sup>)[0,T] form a stationary sequence. This assumption ensures the task of estimating ϕI(T) is well-posed. Note this condition is slightly stronger than necessary but weaker than requiring {<sup>X</sup>t, t ≥ <sup>0</sup>} to be stationary, cf. Proposition [2.13.](#page-4-0)

The first observational framework can be recast in the second by appropriately pasting the X <sup>n</sup>'s into a single process {<sup>X</sup>t, t ≥ <sup>0</sup>}. Going forward we hence focus on the second setting and refer to the large sample asymptotics <sup>N</sup> → ∞ as long-span asymptotics. For any finite collection of words I, we thus consider the estimator

$$\hat{\phi}_{\mathbf{I}}^{\Pi(N)}(T) := \frac{1}{N} \sum_{n=1}^N S^{\mathbf{I}}(\mathbb{X}^{n, \pi_{N, n}})_{[0, T]}. \quad (7)$$

We will be interested in the double asymptotics where, as the number of signature evaluations N increases, the granularity of the discretized paths from which such signatures are computed also increases, i.e.

$$|\Pi(N)| := \max_{1 \leq n \leq N} |\pi_{N,n}| \rightarrow 0, \quad N \rightarrow \infty.$$

We can decompose

$$\begin{aligned} & \hat{\phi}^{\Pi(N)}(T) - \phi_{\mathbf{I}}(T) \\ &= \frac{1}{N} \sum_{n=1}^N (S^{\mathbf{I}}(\mathbb{X}^{n, \pi_{N, n}})_{[0, T]} - S^{\mathbf{I}}(\mathbb{X}^n)_{[0, T]}) \\ &+ \frac{1}{N} \sum_{n=1}^N S^{\mathbf{I}}(\mathbb{X}^n)_{[0, T]} - \mathbb{E} [S^{\mathbf{I}}(\mathbb{X})_{[0, T]}]. \end{aligned} \quad (8)$$

Under suitable conditions, we shall prove ϕˆΠ(N) I (T) is consistent and asymptotically normal for ϕI(T) by showing

- 1. each summand in the first term converges to zero in L <sup>m</sup> in the in-fill asymptotics |<sup>π</sup>N,n| → <sup>0</sup>;
- 2. the second term, when inflated by √ N, converges in distribution to a normal random variable in the large sample asymptotics <sup>N</sup> → ∞.

#### 2.1.1. IN-FILL ASYMPTOTICS

The convergence in probability [\(5\)](#page-2-2) is not sufficient to show consistency of the expected signature estimator. In this section, we thus explore continuity conditions on the process X ensuring the convergence holds in a stronger L <sup>m</sup> sense.

Let {Fs,t, [s, t] ⊆ [0, T]} be a family of sigma-algebras such that, for [u, v] ⊆ [s, t] ⊆ [0, T], Fu,v ⊆ Fs,t and, for [s, t] ⊆ [0, T], <sup>X</sup>s,u is Fs,t-measurable for all <sup>u</sup> ∈ [s, t].

The following continuity assumptions will be used to state the in-fill asymptotics.

Assumption 2.6. For all <sup>0</sup> ≤ s < u < t ≤ <sup>T</sup>,

$$(\mathbf{A}\alpha) \quad \|\mathbf{X}_{s,t}\|_{L^p} \lesssim |t - s|^\alpha.$$

$$(\mathbf{A}\beta) \quad \|\mathbb{E}_{\mathcal{F}_{0,s} \vee \mathcal{F}_{t,T}}[\mathbf{X}_{s,u} \otimes \mathbf{X}_{u,t}]\|_{L^{p/2}} \lesssim |t - s|^\beta.$$

$$(\text{A}\gamma) \quad \begin{aligned} & \|\mathbb{E}_{\mathcal{F}_{0,s} \vee \mathcal{F}_{t,T}} [\mathbf{X}_{s,u} \otimes \mathbf{X}_{u,t}^{\otimes 2}]\|_{L^p/3} \lesssim |t - s|^\gamma, \\ & \|\mathbb{E}_{\mathcal{F}_{0,s} \vee \mathcal{F}_{t,T}} [\mathbf{X}_{s,u}^{\otimes 2} \otimes \mathbf{X}_{u,t}]\|_{L^p/3} \lesssim |t - s|^\gamma. \end{aligned}$$

$$(\text{A}\delta) \quad \|\mathbb{E}_{\mathcal{F}_{0,s}}[\mathbf{X}_{s,t}]\|_{L^p} \lesssim |t - s|^\delta.$$

*Remark* 2.7*.* By the contraction property of the conditional expectation, the strongest form of [\(A](#page-3-0)β), [\(A](#page-3-0)γ) and [\(A](#page-3-0)δ) is obtained by setting Fs,t <sup>=</sup> <sup>σ</sup>(Xs,u, u ∈ [s, t]).

Theorem 2.8. *Let* <sup>k</sup> = maxI∈<sup>I</sup> |I| *and, for* <sup>m</sup> ≥ <sup>2</sup>*, set* p = mk*. Assume* X *is a canonical geometric stochastic process that satisfies one of the following:*

(i) (**A**
$$\alpha$$
) for  $\alpha > 1/2$ ;

(ii) 
$$(\mathbf{A}\alpha), (\mathbf{A}\delta)$$
 for  $\alpha = 1/2, \delta \geq 1$ ;

(iii) 
$$(\mathbf{A}\alpha)$$
,  $(\mathbf{A}\beta)$  for  $\alpha \in (1/3, 1/2]$ ,  $\beta > 1$ ;

(iv) 
$$(\mathbf{A}\alpha), (\mathbf{A}\beta), (\mathbf{A}\gamma) \text{ for } \alpha \in (1/4, 1/3], \beta > 1, \gamma > 1$$
;

*with*

$$\epsilon = \begin{cases} 2\alpha - 1, & \text{if } (i), \\ (2\alpha - 1/2) \wedge (\alpha + \delta - 1), & \text{if } (ii), \\ 3\alpha \wedge \beta - 1, & \text{if } (iii), \\ 4\alpha \wedge \beta \wedge \gamma - 1, & \text{if } (iv), \end{cases} \quad (9)$$

*and consider a signature-defining, cf. Definition [2.5,](#page-2-0) sequence of refining partitions* {<sup>π</sup>n, n ≥ <sup>1</sup>} *of the interval* [0, T] *such that*

$$\sum_{n \geq 1} |\pi_n|^\epsilon < \infty,$$

*then the stronger convergence holds*

$$S^{\mathbf{I}}(\mathbb{X}^{\pi_n})_{[0,T]} \xrightarrow{L \rightarrow \infty} S^{\mathbf{I}}(\mathbb{X})_{[0,T]}, \quad n \rightarrow \infty, \quad (10)$$

*with rate* O( P n′≥n |<sup>π</sup>n′ | ϵ )*.*

*Proof.* See Appendix [B.1.](#page-12-0)

*Remark* 2.9*.* Note that, if {<sup>π</sup>n, n ≥ <sup>1</sup>} is a sequence of dyadic partitions with |<sup>π</sup>n| = 2−<sup>n</sup>T, then

$$\sum_{n \geq 1} |\pi_n|^\epsilon = \sum_{n \geq 1} 2^{-n\epsilon} T^\epsilon = \frac{T^\epsilon}{1 - 2^{-\epsilon}} < \infty,$$

#### 2.1.2. LONG-SPAN ASYMPTOTICS

Theorem 2.10. *Fix* T > <sup>0</sup> *and let* {<sup>X</sup>t, t ≥ <sup>0</sup>} *be a stochastic process such that* X <sup>1</sup> <sup>=</sup> {<sup>X</sup>t, t ∈ [0, T]} *satisfies the assumptions of Theorem [2.8](#page-3-1) with* m > 2*. Assume* {X <sup>n</sup>, n ≥ <sup>1</sup>} *is stationary and ergodic and the sequence of partitions* {Π(N), N ≥ <sup>1</sup>} *is such tha,t for each* <sup>n</sup> ≥ <sup>1</sup>*,* <sup>π</sup>·,n <sup>=</sup> {<sup>π</sup>N,n, N ≥ <sup>n</sup>} *is a signature-defining sequence of refining partitions, and*

$$\sum_{N' \geq N} |\Pi(N')|^\epsilon \rightarrow 0, \quad N \rightarrow \infty. \quad (11)$$

*Then the expected signature estimator* [\(7\)](#page-3-2) *is*

*I. consistent, i.e. 
$$\hat{\phi}_{\mathbf{I}}^{\Pi(N)}(T) \xrightarrow{L^2} \phi_{\mathbf{I}}(T)$$
 as  $N \rightarrow \infty$ .*

*If, moreover,* {<sup>X</sup> <sup>n</sup>, n ≥ <sup>1</sup>} *is strongly mixing with mixing coefficient* {α(n), n ≥ <sup>1</sup>} *such that, for* <sup>ζ</sup> <sup>=</sup> <sup>m</sup> − <sup>2</sup> <sup>&</sup>gt; <sup>0</sup>*,*

$$\sum_{n \geq 1} \alpha(n)^{\zeta/(2+\zeta)} < \infty, \quad (12)$$

*and*

$$\sqrt{N} \sum_{N' \geq N} |\Pi(N')|^\epsilon \rightarrow 0, \quad N \rightarrow \infty, \quad (13)$$

*where* ϵ *is given in Equation* [\(9\)](#page-3-3)*, then the estimator is also*

#### *2. asymptotically normal, i.e.*

$$\sqrt{N} \left( \hat{\phi}_{\mathbf{I}}^{\Pi(N)}(T) - \phi_{\mathbf{I}}(T) \right) \xrightarrow{\mathcal{E}} \mathcal{N}(0, \Sigma_{\mathbf{I}}), \quad N \rightarrow \infty,$$

*as long as* Σ<sup>I</sup> *is strictly positive definite, where*

$$\begin{aligned}\Sigma_{\mathbf{I}} &= \text{Var} \left( S^{\mathbf{I}}(\mathbb{X}^1)_{[0,T]} \right) \\ &+ 2 \sum_{n \geq 2} \text{Cov} \left( S^{\mathbf{I}}(\mathbb{X}^1)_{[0,T]}, S^{\mathbf{I}}(\mathbb{X}^n)_{[0,T]} \right).\end{aligned}$$

*Proof.* See Appendix [B.2.](#page-22-0)

*Remark* 2.11*.* If {Π(N), N ≥ <sup>1</sup>} is a sequence of expanding dyadic refinements, i.e. for each <sup>n</sup> ≥ <sup>1</sup>, <sup>π</sup>·,n is a sequence of dyadic partitions with |<sup>π</sup>N,n| = 2−<sup>N</sup> <sup>T</sup>, <sup>N</sup> ≥ <sup>n</sup>, as in Remark [2.9,](#page-3-4) then |Π(N)| = 2−<sup>N</sup> <sup>T</sup> and, hence,

$$\sqrt{N} \sum_{N' \geq N} |\Pi(N')|^\epsilon = \mathcal{O}(\sqrt{N} 2^{-\epsilon N}) \rightarrow 0, \quad N \rightarrow \infty.$$

Corollary 2.12. *Assume the conditions of Theorem [2.10](#page-4-1) hold with Theorem [2.8.](#page-3-1)(ii) satisfied for some* m > 4 *and for any* T > 0*. Assume furthermore we can characterize the rate of convergence of Theorem [2.10.](#page-4-1)1 as* <sup>ρ</sup>(N) ∼ <sup>N</sup> <sup>−</sup><sup>υ</sup> *for some* <sup>υ</sup> ∈ (0, 1)*. Then the kernel estimator*

$$\hat{\Sigma}_{\mathbf{I}}^{\Pi(N)} = \sum_{|n| \leq h_N} \hat{\Sigma}_{\mathbf{I}}^{n, \Pi(N)},$$

*with* h<sup>N</sup> = Nυ/<sup>2</sup> *, non-overlapping cross-covariances*

$$\hat{\Sigma}_{\mathbf{I}}^{n,\Pi(N)} = \frac{1}{M} \sum_{m=1}^M [S^{\mathbf{I}}(\mathbb{X}^{\pi_{N,(n+1)m-n}})_{[0,T]} - \hat{\phi}_{\mathbf{I}}^{\Pi(N)}(T)] \\ \times [S^{\mathbf{I}}(\mathbb{X}^{\pi_{N,(n+1)m}})_{[0,T]} - \hat{\phi}_{\mathbf{I}}^{\Pi(N)}(T)]^{\mathbf{T}},$$

*for* <sup>M</sup> <sup>=</sup> ⌊N/(<sup>n</sup> + 1)⌋ *and*

$$\hat{\Sigma}_{\mathbf{I}}^{-n,\Pi(N)} := \left(\hat{\Sigma}_{\mathbf{I}}^{n,\Pi(N)}\right)^{\mathbf{T}} \quad n = 1, \dots, N-1,$$

*is consistent for* ΣI*, i.e.* Σ Π(N) I L → <sup>Σ</sup><sup>I</sup> *as* <sup>N</sup> → ∞*, and hence the CLT result of Theorem [2.10](#page-4-1) can be made feasible.*

*Proof.* See Appendix [B.3.](#page-24-0)

Requiring {<sup>X</sup> <sup>n</sup>, n ≥ <sup>1</sup>} to be stationary and ergodic or strongly mixing are high-level conditions. The following results give stronger but easier-to-interpret conditions.

Proposition 2.13. *Fix* T > <sup>0</sup> *and let* {<sup>X</sup>t, t ≥ <sup>0</sup>} *be a stochastic process. Then*[<sup>3</sup>](#page-0-0)

$$\begin{aligned} \{\mathbf{X}_t, t \geq 0\} & \text{ is stationary} \\ \implies \{\mathbf{X}_t, t \geq 0\} & \text{ has jointly stationary increments} \\ \implies \{\mathbf{X}^n, n \geq 1\} & \text{ is stationary.} \end{aligned}$$

*If any of the above holds, and* X 1 *is a canonical geometric stochastic process, then, for any collection of words* I*,*

$$\begin{aligned} \{\mathbf{X}_t, t \geq 0\} &\text{ is strongly mixing} \\ \implies \{\mathbb{X}^n, n \geq 1\} &\text{ is strongly mixing.} \end{aligned}$$

*Proof.* See Appendix [B.4.](#page-27-0)

One might expect a similar statement to hold for ergodicity, but Remark [B.6](#page-27-1) shows that

$$\{\mathbf{X}_t, t \geq 0\}$$
 is ergodic  $\Rightarrow \{\mathbb{X}^n, n \geq 1\}$  is ergodic.

Strong mixing implies ergodicity and hence the second part of Proposition [2.13](#page-4-0) yields a sufficient condition (as far as {X <sup>n</sup>, n ≥ <sup>1</sup>} is concerned) for both the consistency and asymptotic normality results of Theorem [2.10.](#page-4-1) Strong mixing is a somewhat restrictive assumption and hence one might wish to find a set of interpretable conditions weaker than strong mixing ensuring at least consistency of the estimator. The following theorem gives such a condition when {<sup>X</sup>t, t ≥ <sup>0</sup>} is a Gaussian process.

<sup>3</sup>We say {Xt, t ≥ 0} has jointly stationary increments if for all n ∈ N, 0 ≤ s<sup>i</sup> ≤ t<sup>i</sup> with i = 1, . . . , n, and t ≥ 0,

$$(\mathbf{X}_{s_1, t_1}, \dots, \mathbf{X}_{s_n, t_n}) \stackrel{\mathcal{L}}{=} (\mathbf{X}_{t+s_1, t+t_1}, \dots, \mathbf{X}_{t+s_n, t+t_n}). \quad (14)$$

Theorem 2.14. *Fix* T > <sup>0</sup> *and let* {<sup>X</sup>t, t ≥ <sup>0</sup>} *be a Gaussian process such that* <sup>X</sup> <sup>=</sup> {<sup>X</sup>t, t ∈ [0, T]} *is a canonical geometric stochastic process satisfying*[<sup>4</sup>](#page-0-0) [\(A](#page-3-0)α) *with* <sup>α</sup> ≥ <sup>1</sup>/<sup>2</sup> *and* <sup>p</sup> = 2*. Assume the sequence of dyadic partitions of* [0, T] *is signature-defining for* <sup>X</sup> *and for each* <sup>N</sup> ≥ <sup>1</sup> *let* <sup>π</sup>N,n *be the dyadic partition the interval* [0, T] *with mesh* |<sup>π</sup>N,n| = 2−<sup>N</sup> <sup>T</sup>*.*

*Suppose* {<sup>X</sup>t, t ≥ <sup>0</sup>} *has constant mean and timehomogeneous increment covariance, i.e.* ∀u, v, s, t, r ≥ <sup>0</sup>

$$\text{Cov} (\mathbf{X}_{u,v}, \mathbf{X}_{s,t}) = \text{Cov} (\mathbf{X}_{u+r,v+r}, \mathbf{X}_{s+r,t+r}),$$

*satisfying, for some decreasing* <sup>θ</sup> : <sup>R</sup><sup>+</sup> → <sup>R</sup><sup>+</sup> *with* <sup>θ</sup>(t) → <sup>0</sup>, t → ∞ *and* <sup>R</sup> <sup>T</sup> 0 <sup>θ</sup>(t)*d*t < ∞ *and* <sup>m</sup> ∈ <sup>N</sup>*,*

$$(\mathbf{A}\theta) \quad \| \text{Cov} (\mathbf{X}_{u,v}, \mathbf{X}_{s,t}) \| \lesssim \theta(|s - v|)|v - u||t - s|,$$

*for all* <sup>0</sup> ≤ <sup>u</sup> ≤ v < s ≤ <sup>t</sup> *with* |s−v| ≥ <sup>m</sup> 2 (|t−s|+|v−u|)*. Then the expected signature estimator* [\(7\)](#page-3-2) *is consistent, i.e.* ϕˆΠ(N) I (T) → <sup>ϕ</sup>I(T) *as* <sup>N</sup> → ∞*.*

*Proof.* See Appendix [B.5.](#page-29-0)

#### 2.2. Variance Reduction via Martingale Correction

In Section [2.1](#page-2-3) we developed the necessary theory to establish the asymptotic properties of the estimator [\(7\)](#page-3-2) for the statistic ϕ<sup>I</sup> (T) = <sup>E</sup>[S I (X)[0,T] ], for any word I = (i1, . . . , ik). This section aims to find an alternative estimator with better finite sample properties when the process <sup>X</sup> <sup>=</sup> {<sup>X</sup>t, t ∈ [0, T]} is a martingale. We restrict ourselves to the independent observation setting, with the same partition across samples, i.e. πN,n = π for n = 1, . . . , N. We will hence be considering the estimator

$$\hat{\phi}_I^{N,\pi}(T) := \frac{1}{N} \sum_{n=1}^N S^I(\mathbb{X}^{n,\pi})_{[0,T]}, \quad (15)$$

where the X n,π are i.i.d. piecewise linear observations of <sup>X</sup> over the partition[<sup>5</sup>](#page-0-0) π. We introduce the control-variate modification of the estimator [\(15\)](#page-5-0),

$$\hat{\phi}_I^{N,\pi,c}(T) := \frac{1}{N} \sum_{n=1}^N (S_c^I(\mathbb{X}^{n,\pi})_{[0,T]} - cS_c^I(\mathbb{X}^{n,\pi})_{[0,T]}), \quad (16)$$

where, setting I−<sup>1</sup> := (i1, . . . , ik−1),

$$S_c^I(\mathbb{X}^\pi)_{[0,T]} := \sum_{[u,v] \in \pi} S^{I-1}(\mathbb{X}^\pi)_{[0,u]} X_{u,v}^{(i_k)}.$$

The correction term S I c (<sup>X</sup> π )[0,T] is inspired by considering the continuous-time signature

$$S^I(\mathbb{X})_{[0,T]} = \int_0^T S^{I-1}(\mathbb{X})_{[0,s]} \circ dX_s^{(i_k)},$$

where the integral is defined in the Stratonovich sense. To preserve the estimator's unbiasedness while reducing the variance we aim to find a mean-zero control variate S I c (X)[0,T] that is highly correlated with S I (X)[0,T] . A natural candidate is

$$S_c^I(\mathbb{X})_{[0,T]} = \int_0^T S^{I-1}(\mathbb{X})_{[0,s]} \, dX_s^{(i_k)},$$

where the outermost integral is now interpreted in the Itoˆ sense. If X is a square-integrable martingale satisfying the conditions of [Jacod & Shiryaev](#page-9-12) [\(1987,](#page-9-12) Theorem I.4.40), {S I c (X)[0,t] , t ∈ [0, T]} is also a square-integrable martingale with <sup>E</sup>[S I c (X)[0,T] ] = 0. Going back to the discretized setting, we note that, when X is a martingale, the discretized correction term S I c (<sup>X</sup> π )[0,T] is also mean-zero and, hence, the control variate estimator ϕˆN,π,c I (T) has the same bias as ϕˆN,π I (T), but, when picking the optimal[<sup>6</sup>](#page-0-0)

$$c = c_\pi^* := \frac{\text{Cov}(S^I(\mathbb{X}^\pi)_{[0,T]}, S_c^I(\mathbb{X}^\pi)_{[0,T]})}{\text{Var}(S_c^I(\mathbb{X}^\pi)_{[0,T]})},$$

it has reduced variance

$$\text{Var}(\hat{\phi}_I^{N,\pi,c_\pi^*}(T)) = (1 - \rho_{I,\pi}^2) \text{Var}(\hat{\phi}_I^{N,\pi}(T)),$$

where 
$$\rho_{I,\pi} := \text{Corr}(S^I(\mathbb{X}^\pi)_{[0,T]}, S_c^I(\mathbb{X}^\pi)_{[0,T]})$$
.

In practice, to estimate c ∗ π , the most straightforward approach would be to use the sample variance and covariance. In this case the estimator for c ∗ π is the slope of the simple linear regression of {<sup>S</sup> I (<sup>X</sup> n,π)[0,T] , n = 1, . . . , N} against {S I c (<sup>X</sup> n,π)[0,T] , n = 1, . . . , N} or, exploiting the mean zero property of the control,

$$\hat{c}_\pi^* = \frac{\sum_{n=1}^N S_c^I(\mathbb{X}^{n,\pi})_{[0,T]} S_c^I(\mathbb{X}^{n,\pi})_{[0,T]}}{\sum_{n=1}^N S_c^I(\mathbb{X}^{n,\pi})_{[0,T]}^2}.$$

In Appendix [C.2](#page-33-0) we propose an alternative estimator for c ∗ π derived using the properties of the signature.

*Remark* 2.15*.* This variance reduction technique is not limited to processes X that are *full* martingales but can also be applied to *partial* martingales, i.e. X such that only a subset of the components is a martingale. In this case, we can use the control variate expected signature estimator for any word I = (i1, . . . , ik) such that <sup>X</sup> (ik) is a martingale.

<sup>4</sup>When α = 1/2, assume furthermore <sup>X</sup> satisfies [\(A](#page-3-0)δ) with δ ≥ 1 and p = 2k where k = maxI∈<sup>I</sup> |I|.

<sup>5</sup>Note that by [Friz & Victoir](#page-9-10) [\(2010,](#page-9-10) Chapter 14) any sequence of partitions with vanishing mesh size is signature-defining.

<sup>6</sup>We assume throughout Var(S I <sup>c</sup> (<sup>X</sup> π )[0,T ]) ∈ (0, ∞).

Even when the data generating process X is not a martingale, the variance reduction achieved by the corrected estimator [\(16\)](#page-5-1) may outweigh the bias it introduces, leading to better performance – in terms of mean squared error (MSE) – than the classic estimator [\(15\)](#page-5-0). In cases where the underlying process cannot be assumed to be a martingale we thus suggest to treat the martingale correction as a data transformation applicable in the learning pipeline (a model hyper-parameter in a similar spirit to the add-time or the lead-lag transform in the signature context) whose usefulness may be empirically ascertained via cross-validation.

## 3. Applications

#### 3.1. Examples

We now consider a few concrete examples of continuoustime stochastic processes satisfying the assumptions of Theorem [2.10](#page-4-1) and Theorem [2.14.](#page-4-2) Note that BM, CAR and Heston are semimartingales and hence, by Remark [2.2,](#page-1-2) they are canonical geometric stochastic processes such that any sequence of partitions with vanishing mesh size is signature defining. fBm is instead an example of a process that is not a semimartingale but is a canonical geometric stochastic process with dyadic signature-defining sequence of partitions (Remark [2.2\)](#page-1-2). Taking {Π(N), N ≥ <sup>1</sup>} to be a sequence of expanding dyadic partitions thus ensures the observational assumptions of Theorem [2.10](#page-4-1) and Theorem [2.14](#page-4-2) are satisfied by all four processes, cf. Remark [2.11.](#page-4-3)

BM A standard Brownian motion {<sup>B</sup>t, t ≥ <sup>0</sup>}. It can be easily checked it satisfies [\(A](#page-3-0)α) and [\(A](#page-3-0)δ), for any <sup>α</sup> ≥ <sup>1</sup>/2, δ ≥ <sup>1</sup> and <sup>p</sup> ≥ <sup>2</sup>. Moreover, {<sup>B</sup>t, t ≥ <sup>0</sup>} has stationary and independent increments and, hence, the (ind) and (chop) sampling schemes are equivalent: in both cases we can apply[<sup>7</sup>](#page-0-0) Theorem [2.10](#page-4-1) to deduce consistency and asymptotic normality of the expected signature estimator.

fBm A fractional Brownian motion {B<sup>H</sup> t , t ≥ <sup>0</sup>} with Hurst parameter H > 1/2. <sup>B</sup> <sup>H</sup> satisfies [\(A](#page-3-0)α) with α = H (Appendix [E.2.2\)](#page-46-0) and, hence, Assumption [2.6](#page-3-0) is fulfilled. Under (ind) sampling, {<sup>B</sup> H,n, n ≥ <sup>1</sup>} is trivially stationary and strong mixing and, hence, we can apply Theorem [2.10.](#page-4-1) When instead paths are obtained under (chop) we can apply[<sup>8</sup>](#page-0-0) Theorem [2.14,](#page-4-2) cf. Example [E.2.2,](#page-46-0) to deduce consistency.

CAR A bidimensional Continuous-time Autoregressive (CAR) process {<sup>Y</sup>t, t ≥ <sup>0</sup>} of order <sup>p</sup> = 2 driven by a standard Brownian motion with drift <sup>A</sup> = (A1, A2) ∈ (<sup>R</sup> 2×2 ) 2 . The CAR process is defined as the first d = 2 entries of its pd = 4-dimensional state space representation {<sup>X</sup>t, t ≥ <sup>0</sup>}: an Ornstein-Uhlenbeck process with drift and diffusion

$$A_{\mathbf{A}} = \begin{pmatrix} 0_{2 \times 2} & -I_{2 \times 2} \\ A_2 & A_1 \end{pmatrix}, \quad \Sigma = \begin{pmatrix} 0_{2 \times 2} & 0_{2 \times 2} \\ 0_{2 \times 2} & I_{2 \times 2} \end{pmatrix},$$

[\(Lucchese et al.,](#page-10-9) [2023;](#page-10-9) [Marquardt & Stelzer,](#page-10-10) [2007\)](#page-10-10). We can apply the first set of conditions in Appendix [D.1.2](#page-41-0) to deduce that {<sup>X</sup>t, t ≥ <sup>0</sup>} (and hence {<sup>Y</sup>t, t ≥ <sup>0</sup>}) satisfies [\(A](#page-3-0)α) and [\(A](#page-3-0)δ), for <sup>α</sup> = 1/2, <sup>δ</sup> = 1 and any <sup>p</sup> ≥ <sup>2</sup>. Under (ind) sampling we can hence apply Theorem [2.10.](#page-4-1) Moreover, when A<sup>A</sup> has positive real parts of all eigenvalues and the process is started in its stationary distribution, {<sup>X</sup>t, t ≥ <sup>0</sup>} and {<sup>Y</sup>t, t ≥ <sup>0</sup>} are stationary, ergodic and strongly mixing with strong mixing coefficient <sup>α</sup>(t) = O(<sup>e</sup> <sup>−</sup>at), for some a > 0 [\(Marquardt & Stelzer,](#page-10-10) [2007\)](#page-10-10). We can hence apply Proposition [2.13](#page-4-0) to deduce that {<sup>Y</sup> <sup>n</sup>, n ≥ <sup>1</sup>} is stationary and strongly mixing with strong mixing coefficient α(n) = O(e <sup>−</sup>anT ), i.e. satisfying Equation [\(12\)](#page-4-4), for (any) ζ > 0. Under (chop) sampling we can thus apply[<sup>9</sup>](#page-0-0) the consistency and asymptotic normality results of Theorem [2.10.](#page-4-1)

Heston The joint price-variance dynamics of a Heston model under the risk-neutral measure Q with zero interest rate and no dividends, i.e. {(St, Vt), t ≥ <sup>0</sup>} such that

$$\begin{aligned} dS_t &= \sqrt{V_t} S_t dW_t^S, \\ dV_t &= \kappa(\theta - V_t) dt + \xi \sqrt{V_t} dW_t^V, \end{aligned}$$

where {W<sup>S</sup> t , t ≥ <sup>0</sup>} and {W<sup>V</sup> t , t ≥ <sup>0</sup>} are standard Brownian motions with correlation ⟨WS, W<sup>V</sup> ⟩<sup>t</sup> <sup>=</sup> ρt. Under the Feller condition 2κθ > ξ<sup>2</sup> , the variance process is strictly positive (and so is {<sup>S</sup>t, t ≥ <sup>0</sup>}). The Heston model is thus an Ito diffusion with Lipschitz drift <sup>ˆ</sup> <sup>f</sup> : <sup>R</sup><sup>+</sup> ×<sup>R</sup><sup>+</sup> 7→ <sup>R</sup> 2 and <sup>1</sup>/2-Holder continuous diffusion ¨ <sup>σ</sup> : <sup>R</sup>+×<sup>R</sup><sup>+</sup> 7→ <sup>R</sup> 2×2 . We can thus apply the third case of Appendix [D.1.2](#page-41-0) to prove that {(St, Vt), t ≥ <sup>0</sup>} satisfies [\(A](#page-3-0)α) and [\(A](#page-3-0)δ) with <sup>α</sup> = 1/2, δ = 1 and any p > 2 for deterministic initial conditions S<sup>0</sup> = s<sup>0</sup> and V<sup>0</sup> = v0. When paths are sampled under (ind) we can hence apply Theorem [2.10](#page-4-1) to deduce consistency and asymptotic normality of the expected signature estimator.

#### 3.2. Experiments

Quite a wide range of learning algorithms has been developed leveraging the properties of the expected signature. The theory for such algorithms is usually developed under

<sup>7</sup>Brownian motion is a Gaussian process with constant mean function and time-homogeneous covariance of the increments trivially satisfying [\(A](#page-4-2)θ) with θ ≡ 0 and m = 0, it thus also falls under the scope of Theorem [2.14.](#page-4-2)

<sup>8</sup>The increments of fractional Brownian motion are not strongly mixing [\(Mandelbrot & Van Ness,](#page-10-8) [1968\)](#page-10-8) and, hence, we cannot apply the second part of Theorem [2.10](#page-4-1) to deduce asymptotic normality.

<sup>9</sup>The CAR process is a Gaussian process satisfying [\(A](#page-4-2)θ), cf. Appendix [E.2.1,](#page-45-0) and hence also falls under the scope of Theorem [2.14.](#page-4-2)

the assumption of bounded variation paths for the input process X, assumed to be piecewise linear. The results in Section [2.1](#page-2-3) give the theoretical foundation for their probabilistic interpretation when the underlying process X is an, arguably more realistic, continuous-time stochastic process such as the ones discussed in Section [3.1.](#page-6-0) In this section we review a few algorithms from the literature, showcasing the practical relevance of the asymptotic results of Section [2.1](#page-2-3) and the potential improvements achieved by the martingale correction introduced in Section [2.2.](#page-5-2) Code and examples demonstrating the integration of the martingale correction into machine learning algorithms, along with the simulation results from the previous section, are available at [https://github.](https://github.com/lorenzolucchese/esig) [com/lorenzolucchese/esig](https://github.com/lorenzolucchese/esig). The code is designed to be compatible with Python-based ML pipelines, supporting both numpy arrays and torch tensors.

#### 3.2.1. TIME SERIES CLASSIFICATION

The first model we consider, introduced in [Triggiano &](#page-10-3) [Romito](#page-10-3) [\(2024\)](#page-10-3), falls under the general task of time series classification, mapping an input path <sup>x</sup> ∈ <sup>R</sup> d×M<sup>1</sup> to a class label <sup>c</sup> ∈ C. The input stream is interpreted as a discretetime realization of a Gaussian process, whose conditional mean and covariance are learned parametrically. The expected signature of the latent Gaussian process, used as input to a classification layer, is estimated by super-sampling the process. Theorem [2.14](#page-4-2) ensures this approach consistently estimates the expected signature of the latent continuous-time Gaussian process, a fundamental step for the probabilistic interpretation of the algorithm.

We replicate the synthetic data experiments of [Triggiano &](#page-10-3) [Romito](#page-10-3) [\(2024\)](#page-10-3) on the (FBM), (OU) and (Bidim) datasets. The performance on the out-of-sample testing datasets of the Gaussian Process augmented Expected Signature (GPES) classifier with and without martingale correction is reported in Table [1.](#page-7-0) The output of the GPES model is by construction stochastic and, hence, we repeat the evaluation of the model with 10 different seeds. In Table [1](#page-7-0) we report the mean accuracy and standard error of the model with and without martingale correction (MC), as well as the results of an independent samples t-test between their accuracies. The martingale correction significantly improves the performance of the GPES model, a remarkable result considering that most processes in the three datasets are not martingales.

#### 3.2.2. PRICING PATH-DEPENDENT DERIVATIVES

The next application we consider is a purely financial one. The objective is to price (and hedge) path-dependent derivatives by decomposing them into a set of atomic Arrow-Debreu-like securities. Let <sup>X</sup> <sup>=</sup> {<sup>X</sup>t, t ∈ [0, T]} be a price process, i.e. a semimartingale over some probability space. In [Lyons et al.](#page-10-5) [\(2021,](#page-10-5) Proposition 4.5) the authors use the

|                 | Predictive Accuracy [%] |              |              |
|-----------------|-------------------------|--------------|--------------|
|                 | FBM                     | OU           | Bidim        |
| GPES            | 95.62 (0.18)            | 62.20 (0.70) | 79.33 (0.46) |
| GPES-MC         | 95.26 (0.70)            | 88.26 (0.31) | 88.97 (0.44) |
| <i>t</i> -stat  | 1.49                    | −101.92      | −45.52       |
| <i>p</i> -value | 0.15                    | 0.00         | 0.00         |

Table 1. Synthetic data experiments of [Triggiano & Romito](#page-10-3) [\(2024\)](#page-10-3): GPES model without and with martingale correction (MC).

universality of the signature to show that a large class of path-depend payoffs F can be arbitrarily well approximated by a linear payoff on the signature, i.e.

$$\text{price}(F) = \mathbb{E}^{\mathbb{Q}}[Z_T F] \approx \langle f, Z_T \mathbb{E}^{\mathbb{Q}}[S(\hat{X}^{\text{LL}})_{[0,T]}] \rangle,$$

for a set of linear coefficients <sup>f</sup> ∈ <sup>T</sup>((<sup>R</sup> 4 ) ∗ ) where Q is a pricing measure for <sup>X</sup>, Z<sup>T</sup> a deterministic discount factor over [0, T] and <sup>X</sup>ˆ LL denotes the add-time lead-lag transform of X. In Appendix [F.2.2](#page-50-0) we also discuss the corresponding hedging problem.

Given a pricing model Q for X, we can hence price F via Monte Carlo simulations. This provides a classic setting for applying the martingale correction described in Section [2.2](#page-5-2) since, under Q, the (discounted) price process X is a martingale. In Figure [2,](#page-7-1) we compare the finite sample properties of the expected signature estimator with and without martingale correction when the price process is assumed to follow a Brownian motion (BM); in the context of option pricing, this is known as the Bachelier model. Similarly, in Figure [3,](#page-8-0) we plot the densities of the two estimators under the Heston dynamics[<sup>10</sup>](#page-0-0) (Heston). Both figures suggest the martingale correction (blue) materially improves the classic estimator (red), and hence more accurate pricing is achieved by the modified estimator introduced in Section [2.2.](#page-5-2)

![](_page_7_Figure_11.jpeg)

Figure 2. Distributions of expected signature estimators for BM. The y-axis is in log-scale.

<sup>10</sup>In both simulations we fix T = 1 and consider π to be uniform with mesh |π| = 2−⌊N/10⌋+1. This choice ensures the sequences of partitions are signature-defining for both processes and satisfy the conditions necessary for consistency and asymptotic normality, cf. Remark [2.11.](#page-4-3)

![](_page_8_Figure_2.jpeg)

Figure 3. Distributions of expected signature estimators for the Heston process with parameters s<sup>0</sup> = 1, v<sup>0</sup> = 0.1, θ = 0.1, κ = 0.6, ξ = 0.2 and ρ = −0.15. The y-axis is in log-scale.

#### 3.2.3. DISTRIBUTIONAL REGRESSION FOR STREAMS

Introduced in [Lemercier et al.](#page-9-5) [\(2021\)](#page-9-5), the Signature of the pathwise Expected Signature (SES) model aims to learn a map from a collection of paths, understood as an empirical measure on path space, to a scalar value, a task known as distributional regression. Under appropriate conditions, the authors show that linear functionals on the signature of the pathwise expected signature are universal for weakly continuous functions [\(Lemercier et al.,](#page-9-5) [2021,](#page-9-5) Theorem 3.2).

We repeat two of the synthetic data experiments conducted in [Lemercier et al.](#page-9-5) [\(2021\)](#page-9-5), analyzing the performance of the SES model without and with martingale correction (MC). We report the average out-of-sample mean-squared error (MSE) and its standard deviation in Table [2](#page-8-1) and Table [3,](#page-8-2) as well as the t-statistic and p-value of a pairwise t-test between the MSEs of the two models. While the results do not yield statistical significance there still seems to be a mild benefit in using the martingale correction, especially considering that the processes of both experiments are not martingales[<sup>11</sup>](#page-0-0) .

Table 2. Ideal gas experiment of [Lemercier et al.](#page-9-5) [\(2021\)](#page-9-5): SES model without and with martingale correction (MC).

|         |             | Predictive MSE $[\times 10^{-3}]$ |             |           |  |
|---------|-------------|-----------------------------------|-------------|-----------|--|
|         |             | $N = 20$                          | $N = 50$    | $N = 100$ |  |
|         |             |                                   |             |           |  |
|         |             |                                   |             |           |  |
| SES     | 1.49 (0.39) | 0.33 (0.13)                       | 0.20 (0.08) |           |  |
| SES-MC  | 1.26 (0.48) | 0.31 (0.09)                       | 0.19 (0.05) |           |  |
| t-stat  | 0.87        | 0.63                              | 0.29        |           |  |
| p-value | 0.43        | 0.56                              | 0.79        |           |  |

Table 3. Rough volatility experiment of [Lemercier et al.](#page-9-5) [\(2021\)](#page-9-5): SES model without and with martingale correction (MC).

#### 4. Conclusions

In this paper, we established new estimation results for the expected signature, a model-free embedding for collections of data streams. Our consistency and asymptotic normality results bridge the gap between the theoretically "optimal" continuous-time expected signature and the empirical discrete-time estimator that can be computed from data. Moreover, we introduced a simple modification of such an estimator with significantly better finite sample properties under the assumption of martingale observations. Our empirical results suggest that the modified estimator might improve the performance of models employing expected signature computations even when the underlying data generating process is not necessarily a martingale.

#### Acknowledgements

This research has been supported by the EPSRC Centre for Doctoral Training in Mathematics of Random Systems: Analysis, Modelling and Simulation (EP/S023925/1). The authors would like to thank Nicola Muca Cirone and Will Turner for helpful discussions on the topic, as well as the three anonymous reviewers for their insightful comments.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

<sup>11</sup>In the first experiment, when the particle radii are large and collisions are more frequent, one could argue the motion of the gas particles to be amenable to that of pollen grains in water, the original experiment which led to the "discovery" of Brownian motion by Scottish botanist Robert Brown in 1827.

- References Buehler, H., Murray, P., Pakkanen, M. S., and Wood, B. Deep Hedging: Learning to Remove the Drift under Trading Frictions with Minimal Equivalent Near-Martingale Measures, 2022. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2111.07844) [2111.07844](https://arxiv.org/abs/2111.07844). Burkholder, D. L., Davis, B., and Gundy, R. F. Integral inequalities for convex functions of operators on martingales. In Le Cam, L. M., Neyman, J., and Scott,
- E. L. (eds.), *Proceedings of the Sixth Berkeley Symposium on Mathematical Statistics and Probability*, volume 2, pp. 223–240, Berkeley, California, 1972. University of California Press. URL [https://projecteuclid.](https://projecteuclid.org/euclid.bsmsp/1200514221) [org/euclid.bsmsp/1200514221](https://projecteuclid.org/euclid.bsmsp/1200514221). Chen, K.-T. Iterated Integrals and Exponential Homomorphisms. *Proceedings of the London Mathematical Society*, s3–4(1):502–512, 1954. doi: 10.1112/plms/s3-4.1.502. Chevyrev, I. and Lyons, T. Characteristic functions of measures on geometric rough paths. *The Annals of Probability*, 44(6):4049–4082, 2016. ISSN 00911798. Chevyrev, I. and Oberhauser, H. Signature moments to characterize laws of stochastic processes, 2018. URL <https://arxiv.org/abs/1810.10971>. Coutin, L. and Qian, Z. Stochastic analysis, rough path analysis and fractional Brownian motions. *Probability Theory and Related Fields*, 122:108–140, 01 2002. doi: 10.1007/s004400100158. Cuchiero, C., Svaluto-Ferro, S., and Teichmann, J. Signature SDEs from an affine and polynomial perspective, 2023. URL [https://arxiv.org/abs/2302.](https://arxiv.org/abs/2302.01362) [01362](https://arxiv.org/abs/2302.01362). Dragomir, S. S. *Some Gronwall Type Inequalities and Applications*. Nova Science, New York, 2003. Fawcett, T. *Problems in stochastic analysis. Connections between rough paths and non-commutative harmonic analysis*. PhD thesis, University of Oxford, 2003. Friedman, A. *Partial Differential Equations of Parabolic Type*. Prentice-Hall, 1964. Friz, P. K. and Victoir, N. B. *Multidimensional Stochastic Processes as Rough Paths: Theory and Applications*. Cambridge Studies in Advanced Mathematics. Cambridge University Press, 2010. Friz, P. K., Hager, P. P., and Tapia, N. Unified signature cumulants and generalized Magnus expansions. *Forum of Mathematics, Sigma*, 10:e42, 2022. doi: 10.1017/fms. 2022.20. Friz, P. K., Hager, P. P., and Tapia, N. On expected signatures and signature cumulants in semimartingale models, 2024. URL [https://arxiv.org/abs/2408.](https://arxiv.org/abs/2408.05085) [05085](https://arxiv.org/abs/2408.05085). Futter, O., Horvath, B., and Wiese, M. Signature Trading: A Path-Dependent Extension of the Mean-Variance Framework with Exogenous Signals, 2023. URL [https:](https://arxiv.org/abs/2308.15135) [//arxiv.org/abs/2308.15135](https://arxiv.org/abs/2308.15135). Graham, B. Sparse arrays of signatures for online character recognition, 2013. URL [https://arxiv.org/](https://arxiv.org/abs/1308.0371) [abs/1308.0371](https://arxiv.org/abs/1308.0371). Hambly, B. and Lyons, T. Uniqueness for the signature of a path of bounded variation and the reduced path group. *Annals of Mathematics*, 171(1):109–167, 2005. Ibragimov, I. A. Some Limit Theorems for Stationary Processes. *Theory of Probability & Its Applications*, 7(4): 349–382, 1962. doi: 10.1137/1107036. Isserlis, L. On a Formula for the Product-Moment Coefficient of any Order of a Normal Frequency Distribution in any Number of Variables. *Biometrika*, 12(1–2):134–139, 11 1918. ISSN 0006-3444. doi: 10.1093/biomet/12.1-2.
  - 134. Jacod, J. and Shiryaev, A. N. *Limit Theorems for Stochastic Processes*, volume 288 of *Grundlehren der mathematischen Wissenschaften*. Springer Berlin, Heidelberg, 1987. ISBN 9783662025161. doi: 10.1007/ 978-3-662-02514-7. Kallenberg, O. *Foundations of Modern Probability*. Probability theory and stochastic modelling. Springer, 2021. ISBN 9783030618728. Kiraly, F. J. and Oberhauser, H. Kernels for Sequentially Ordered Data. *Journal of Machine Learning Research*, 20(31):1–45, 2019. Kulik, A. *Ergodic Behavior of Markov Processes*. De Gruyter, Berlin, Boston, 2018. ISBN 9783110458930. doi: 10.1515/9783110458930. Le, K. A stochastic sewing lemma and applications. ˆ *Electronic Journal of Probability*, 25:1 – 55, 2020. doi: 10.1214/20-EJP442. Lemercier, M., Salvi, C., Damoulas, T., Bonilla, E. V., and Lyons, T. Distribution regression for sequential data. In *24th International Conference on Artificial Intelligence and Statistics (AISTATS 2021)*, Proceedings of Machine Learning Research, pp. 3754–3762. Journal of Machine Learning Research, 2021.

Levin, D., Lyons, T., and Ni, H. Learning from the past, predicting the statistics for the future, learning an evolving system, 2016. URL [https://arxiv.org/abs/](https://arxiv.org/abs/1309.0260) [1309.0260](https://arxiv.org/abs/1309.0260). Lucchese, L., Pakkanen, M. S., and Veraart, A. E. D. Estimation and Inference for Multivariate Continuoustime Autoregressive Processes, 2023. URL [https:](https://arxiv.org/abs/2307.13020) [//arxiv.org/abs/2307.13020](https://arxiv.org/abs/2307.13020). Lyons, T. and McLeod, A. D. Signature Methods in Machine Learning, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2206.14674) [2206.14674](https://arxiv.org/abs/2206.14674). Lyons, T., Caruana, M., and Levy, T. ´ *Differential Equations Driven by Rough Paths: Ecole D' ´ et´ e de Probabilit ´ es de ´ Saint-Flour XXXIV-2004*. Number no. 1908 in Differential Equations Driven by Rough Paths: Ecole D' ´ et´ e de ´ Probabilites de Saint-Flour XXXIV-2004. Springer, 2007. ´ ISBN 9783540712848. Lyons, T., Nejad, S., and Perez Arribas, I. Non-parametric ´ pricing and hedging of exotic derivatives. *Applied Mathematical Finance*, 27(6):457–494, 2021. Mandelbrot, B. B. and Van Ness, J. W. Fractional Brownian Motions, Fractional Noises and Applications. *SIAM Review*, 10(4):422–437, 1968. ISSN 00361445. URL <http://www.jstor.org/stable/2027184>. Marquardt, T. and Stelzer, R. Multivariate CARMA processes. *Stochastic Processes and their Applications*, 117 (1):96–120, Jan 2007. ISSN 03044149. doi: 10.1016/j. spa.2006.05.014. Newey, W. K. and West, K. D. A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3):703–708, 1987. ISSN 00129682, 14680262. Ni, H. *The expected signature of a stochastic process*. PhD thesis, University of Oxford, 2012. Passeggeri, R. On the signature and cubature of the fractional Brownian motion for H > 1/2. *Stochastic Processes and their Applications*, 130(3):1226–1257, 2020. ISSN 0304-4149. doi: 10.1016/j.spa.2019.04.013. Perez Arribas, I., Goodwin, G. M., Geddes, J. R., Lyons, ´ T., and Saunders, K. E. A. A signature-based machine learning model for distinguishing bipolar disorder and borderline personality disorder. *Translational Psychiatry*, 8, 2018. Salvi, C., Cass, T., Foster, J., Lyons, T., and Yang, W. The Signature Kernel Is the Solution of a Goursat PDE. *SIAM Journal on Mathematics of Data Science*, 3(3):873–899, 2021. doi: 10.1137/20M1366794. Schell, A. and Oberhauser, H. Nonlinear independent component analysis for discrete-time and continuous-time signals. *Annals of Statistics*, 51(2):487–518, 2023. Triggiano, F. and Romito, M. Gaussian Processes Based Data Augmentation and Expected Signature for Time Series Classification. *IEEE Access*, 12:80884–80895, 2024. doi: 10.1109/ACCESS.2024.3408712. Willett, D. W. Nonlinear vector integral equations as contraction mappings. *Archive for Rational Mechanics and Analysis*, 15:79–86, 1964. doi: 10.1007/bf00257405. Xie, Z., Sun, Z., Jin, L., Ni, H., and Lyons, T. Learning Spatial-Semantic Context with Fully Convolutional Recurrent Network for Online Handwritten Chinese Text Recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(8):1903–1917, 2018. doi: 10.1109/TPAMI.2017.2732978.

## Contents of the Appendix

[A Informal Glossary](#page-12-1) [B Proofs of Section](#page-12-2) [2](#page-1-3) [B.1 Proof of Theorem](#page-12-0) [2.8](#page-3-1) [B.1.1 Proof of Theorem](#page-14-0) [2.8](#page-3-1) under *(i)*, *(iii)* or *(iv)* [B.1.2 Proof of Theorem](#page-18-0) [2.8](#page-3-1) under *(ii)* [B.2 Proof of Theorem](#page-22-0) [2.10](#page-4-1) [B.2.1 Proof of Theorem](#page-23-0) [2.10,](#page-4-1) consistency B.2.2 Proof of Theorem [2.10, asymptotic normality](#page-24-1) [B.3 Proof of Corollary](#page-24-0) [2.12](#page-4-5) [B.4 Proof of Proposition](#page-27-0) [2.13](#page-4-0) B.4.1 Proof of Proposition [2.13, stationary implications](#page-27-2) B.4.2 Proof of Proposition [2.13, strong mixing implications](#page-28-0) [B.5 Proof of Theorem](#page-29-0) [2.14](#page-4-2) [C Variance Reduction via Martingale Correction](#page-32-0) [C.1 Martingale Continuity Criterion](#page-33-1) [C.2 Estimating](#page-33-0) c ∗ π [C.3 Proof of Lemma](#page-35-0) [C.1](#page-34-0) [D Ito processes and diffusions](#page-40-0) ˆ [D.1 In-fill conditions](#page-40-1) [D.1.1 Ito processes](#page-40-2) ˆ [D.1.2 Ito diffusions](#page-41-0) ˆ [D.2 Long span conditions](#page-43-0) [D.2.1 Ito diffusions](#page-43-1) ˆ [E Gaussian Processes](#page-45-1) [E.1 Gaussian Processes Continuity Criterion](#page-45-2) [E.2 Gaussian Processes Covariance Decay Condition](#page-45-3) [E.2.1 Ornstein-Uhlenbeck Process](#page-45-0) [E.2.2 Fractional Brownian Motion](#page-46-0) [F Machine Learning Algorithms with Expected Signatures](#page-47-0) [F.1 Martingale Correction in Applications](#page-48-0) [F.2 Algorithms](#page-48-1) [F.2.1 Time Series Classification \(Triggiano & Romito,](#page-48-2) [2024\)](#page-10-3) [F.2.2 Pricing Path-Dependent Derivatives \(Lyons et al.,](#page-50-0) [2021\)](#page-10-5) [F.2.3 Distributional Regression for Streams \(Lemercier et al.,](#page-52-0) [2021\)](#page-9-5) [F.2.4 Systematic Trading \(Futter et al.,](#page-52-1) [2023\)](#page-9-6) [G Controlled Linear Regression](#page-53-0) [G.1 Controlled Ordinary Least Squares \(OLS\) estimation](#page-54-0) [G.2 Simulation study](#page-57-0)

#### A. Informal Glossary

This informal glossary provides accessible explanations of selected technical terms and notational conventions used in this paper, aimed at readers with little or no background in rough path theory. These intuitive definitions are intended to aid the understanding of the theoretical framework presented in Section [2,](#page-1-3) particularly Definition [2.1.](#page-1-0) However, they remain closely tied to more technical definitions – such as those of multiplicative functionals, rough paths, and geometric rough paths – which require a more rigorous exposition of rough path theory. For a concise introduction to rough paths, we refer the reader to [Lyons et al.](#page-10-0) [\(2007\)](#page-10-0), and for a treatment in the stochastic setting, to [Friz & Victoir](#page-9-10) [\(2010\)](#page-9-10).

p-variation The p-variation of a path is a measure of its regularity. For the purpose of our discussion it suffices to note that paths that have finite p-variation for low p are more regular. A bounded variation (BV) path is a path with finite 1-variation (also known as total variation). This regularity ensures there exists a well-defined notion of integral against this path (e.g. a piecewise linear paths or continuously differentiable path) and, hence, we can easily define its signature as in Equation (2). Many interesting stochastic processes (e.g. those driven by Brownian motion) have infinite 1-variation (i.e. are not BV) but have finite p-variation for all p > 2 and, hence, defining their signature requires rough path theory.

Convergence in p-variation Convergence in the p-variation metric/topology is a pathwise mode of convergence (i.e. over all points <sup>t</sup> ∈ [0, T] simultaneously) that is (much) stronger than the pointwise (i.e. at fixed <sup>t</sup> ∈ [0, T]) convergence required to state and prove our results. See, for example, Remark [2.4.](#page-2-4)

Spaces of paths We denote by C([0, T], <sup>R</sup> d ), resp. BV([0, T], <sup>R</sup> d ), the space of <sup>R</sup> d -valued continuous, resp. bounded variation, paths over the interval [0, T].

Mesh of a partition For a partition <sup>π</sup> <sup>=</sup> {0 = <sup>t</sup><sup>0</sup> < t<sup>1</sup> < . . . < T} of [0, T], we define its mesh as |π| = max[s,t]∈<sup>π</sup> |t−s| where the maximum is taken over all sub-intervals of the partition.

Shuffle property The shuffle property of the signature is an algebraic property stating that the product of two signature terms is a linear combination of higher-order signature terms. More precisely, the product of the signature terms corresponding to words <sup>I</sup> and <sup>J</sup> is the sum of all signature terms indexed by words <sup>K</sup> of length |I| <sup>+</sup> |J| obtained by interleaving I and J. In the context of the discussion on page 1 this means that all moments of the signature can be written as linear combinations of higher order expected signature terms.

Signature indexing A word <sup>I</sup> = (i1, . . . , in) with <sup>i</sup>1, . . . , i<sup>n</sup> ∈ {1, . . . , d} is a multi-index used to denote an entry of the signature, i.e. a real-valued number. The length of the word, i.e. |I| <sup>=</sup> <sup>n</sup>, denotes the signature level, i.e. an n-dimensional tensor, to which such entry belongs. For example S I (X)[0,T] , where I = (1, 2), denotes the (1, 2)-entry of the second level of the signature (a matrix), while I = (2, 1, 1) denotes the (2, 1, 1)-entry of the third level of the signature (a three-dimensional tensor).

Stochastic processes A continuous stochastic process <sup>X</sup> <sup>=</sup> {<sup>X</sup>t, t ∈ [0, T]} over a probability space (Ω, F, <sup>P</sup>) is such that, for each <sup>ω</sup> ∈ <sup>Ω</sup>, the realization <sup>X</sup>(ω) = {<sup>X</sup>t(ω), t ∈ [0, T]} ∈ <sup>C</sup>([0, T], <sup>R</sup> d ). If one takes Ω = C([0, T], <sup>R</sup> d ) and <sup>P</sup> a probability measure over this path space then each <sup>ω</sup> ∈ <sup>Ω</sup> denotes a possible path realization of <sup>X</sup>. We thus say a property holds pathwise or almost surely if the set of <sup>ω</sup> ∈ <sup>Ω</sup> for which that property holds has probability one.

Canonical geometric stochastic process We define a canonical geometric stochastic process as a continuous stochastic process whose "higher order structure" can be approximated by the iterated integrals of its piecewise-linear interpolations (in probability in the p-variation metric). Its signature is then defined as the limit of the signatures of its piecewise-linear interpolations, i.e. the iterated integrals given in Equation [\(3\)](#page-2-5). For clarity, we emphasize that *canonicity* here refers to the aforementioned construction of the signature, not to the underlying probability space on which the process is defined.

# B. Proofs of Section [2](#page-1-3)

# B.1. Proof of Theorem [2.8](#page-3-1)

*Sketch of proof. The main idea of the proof is to show the sequence of discretized signatures* {S k (<sup>X</sup> <sup>π</sup><sup>n</sup> )[0,T ] , n ≥ 1} *is Cauchy in* L <sup>m</sup>*. Since* L <sup>m</sup> *is a Banach space this implies the sequence converges in* L <sup>m</sup>*. By uniqueness of limits, we can* *deduce this limit is the same as its* P*-limit, i.e.* S k (X)[0,T ] *. To show the sequence is Cauchy in* L <sup>m</sup> *we proceed inductively on the signature level* k ′ ∈ {1, . . . , k} *under the progressively weaker norm* L mk/k′ *. The main ingredient of the inductive step is a manipulation of the discrete-time signature* [\(2\)](#page-2-6)*, ensuring*

$$S^{k'}(\mathbb{X}^{\pi_n+1})_{[\tau_0, \tau_1]} - S^{k'}(\mathbb{X}^{\pi_n})_{[\tau_0, \tau_1]}, \quad [\tau_0, \tau_1] \subseteq [0, T],$$

*can be written as a sum over time intervals* πn+1,[τ0,τ1] *. The inductive assumption is then verified by bounding this summation using Lemma [B.1](#page-13-0) when a simple Minkowski bound is too weak. We use two different manipulations of the discrete-time signature under assumptions (i), (iii) or (iv) and under assumption (ii): In the former case we use the classic representation given in* [\(2\)](#page-2-6)*, while in the latter we rely on the "causal" representation of Lemma [B.3.](#page-14-1) For clarity of exposition we thus divide the proof of Theorem [2.8](#page-3-1) into two parts.*

We first establish a couple of useful lemmas which will be used repeatedly in the proof of this in-fill asymptotic results. The first is a basic result which is also applied in the proof of the stochastic sewing lemma [\(Leˆ,](#page-9-13) [2020\)](#page-9-13). In the following, let E denote a Banach space.

Lemma B.1. *Let* {<sup>Z</sup>n, n = 1, . . . , N} *be a finite sequence of* <sup>E</sup>*-valued random variables in* <sup>L</sup> <sup>m</sup> *with* <sup>m</sup> ∈ [2, ∞) *and let* {Gn, n = 1, . . . , N} *be a filtration such that, for each* <sup>n</sup> ∈ {1, . . . , N}*, the variables* <sup>Z</sup>1, . . . , Zn−<sup>1</sup> *are* Gn*-measurable. Then*

$$\left\| \sum_{n=1}^N Z_n \right\|_{L^m} \leq \sum_{n=1}^N \|\mathbb{E}_{\mathcal{G}_n}[Z_n]\|_{L^m} + 2C_m \left( \sum_{n=1}^N \|Z_n\|_{L^m}^2 \right)^{1/2}.$$

*Proof.*

$$\begin{aligned} \left\| \sum_{n=1}^N Z_n \right\|_{L^m} &\stackrel{(i)}{\leq} \left\| \sum_{n=1}^N \mathbb{E}_{\mathcal{G}_n}[Z_n] \right\|_{L^m} + \left\| \sum_{n=1}^N (Z_n - \mathbb{E}_{\mathcal{G}_n}[Z_n]) \right\|_{L^m} \\ &\stackrel{(ii)}{\leq} \sum_{n=1}^N \|\mathbb{E}_{\mathcal{G}_n}[Z_n]\|_{L^m} + C_m \left\| \sum_{n=1}^N \|Z_n - \mathbb{E}_{\mathcal{G}_n}[Z_n]\|^2 \right\|_{L^{m/2}}^{1/2} \\ &\stackrel{(iii)}{\leq} \sum_{n=1}^N \|\mathbb{E}_{\mathcal{G}_n}[Z_n]\|_{L^m} + C_m \left( \sum_{n=1}^N \|Z_n - \mathbb{E}_{\mathcal{G}_n}[Z_n]\|_{L^m}^2 \right)^{1/2} \\ &\stackrel{(iv)}{\leq} \sum_{n=1}^N \|\mathbb{E}_{\mathcal{G}_n}[Z_n]\|_{L^m} + C_m \left( \sum_{n=1}^N (\|Z_n\|_{L^m} + \|\mathbb{E}_{\mathcal{G}_n}[Z_n]\|_{L^m})^2 \right)^{1/2} \\ &\stackrel{(v)}{\leq} \sum_{n=1}^N \|\mathbb{E}_{\mathcal{G}_n}[Z_n]\|_{L^m} + 2C_m \left( \sum_{n=1}^N \|Z_n\|_{L^m}^2 \right)^{1/2}, \end{aligned}$$

by using in (i) the triangle inequality, in (ii) triangle inequality and the Burkholder-Davis-Gundy (BDG) inequality [\(Burkholder et al.,](#page-9-14) [1972\)](#page-9-14) applied to the martingale {<sup>M</sup>n, n = 1, . . . , N} with <sup>M</sup><sup>n</sup> <sup>=</sup> P<sup>n</sup> <sup>i</sup>=1(Z<sup>i</sup> − <sup>E</sup>G<sup>i</sup> [Z<sup>i</sup> ]), in (iii) and (iv) the triangle inequality and in (v) the contraction property of conditional expectation.

Lemma B.2. *Let* p, p1, . . . , p<sup>l</sup> ∈ (0, ∞) ∪ {+∞} *be such that* <sup>p</sup> −1 <sup>1</sup> + . . . + p −1 <sup>l</sup> = p −1 *, then, for any set of tensors* <sup>A</sup><sup>1</sup> ∈ <sup>L</sup> <sup>p</sup><sup>1</sup> ((<sup>R</sup> d ) <sup>⊗</sup>k<sup>1</sup> ), . . . , <sup>A</sup><sup>l</sup> ∈ <sup>L</sup> <sup>p</sup><sup>l</sup> ((<sup>R</sup> d ) <sup>⊗</sup>k<sup>l</sup> )*,*

$$\|\mathbf{A}_1 \otimes \cdots \otimes \mathbf{A}_l\|_{L^p} \lesssim d^l \|\mathbf{A}_1\|_{L^{p_1}} \cdots \|\mathbf{A}_l\|_{L^{p_l}}.$$

*Proof.*

$$\begin{aligned} \|\mathbf{A}_1 \otimes \cdots \otimes \mathbf{A}_l\|_{L^p} &\leq \sum_{(w_1, \dots, w_l) \in \mathcal{W}_{k_1 + \dots + k_l}} \|A_1^{w_1} \cdots A_l^{w_l}\|_{L^p} \\ &\stackrel{(*)}{\leq} \sum_{(w_1, \dots, w_l) \in \mathcal{W}_{k_1 + \dots + k_l}} \|A_1^{w_1}\|_{L^{p_1}} \cdots \|A_l^{w_l}\|_{L^{p_l}} \end{aligned}$$

$$\leq d^{k_1+\dots+k_l} \|\mathbf{A}_1\|_{L^{p_1}} \cdots \|\mathbf{A}_l\|_{L^{p_l}},$$

where W<sup>k</sup> <sup>=</sup> {1, . . . , d} <sup>k</sup> denotes the set of words of length <sup>k</sup> and, in (∗), we applied the classical Holder inequality. ¨

We also prove a useful lemma that allows us to write the k-th level signature of a piecewise linear path as a "causal" sum of lower order signature terms, i.e. preserving time order. This will allow us to derive an in-fill result with assumptions on the regularity of <sup>E</sup>F0,s [Xs,t], a more natural object than <sup>E</sup>F0,s∨Ft,T [Xs,u ⊗ <sup>X</sup>u,t], when <sup>α</sup> = 1/2, i.e. Theorem [2.8](#page-3-1) under *(ii)*.

Lemma B.3. *Let* <sup>π</sup> *be a partition of* [0, T] *and let* <sup>τ</sup> ∈ <sup>π</sup>*. Then, for* <sup>k</sup> ≥ <sup>0</sup>*, we can write*

$$S^{k+1}(\mathbb{X}^\pi)_{[0,\tau]} = \sum_{i=0}^k \frac{1}{(1+i)!} \sum_{[u,v] \in \pi_{[0,\tau]}} S^{k-i}(\mathbb{X}^\pi)_{[0,u]} \otimes \mathbf{X}_{u,v}^{\otimes(i+1)}.$$

*Proof.* Note that for <sup>k</sup> ≥ <sup>0</sup>,

$$\begin{aligned} S^{k+1}(\mathbb{X}^\pi)_{[0,\tau]} &= \sum_{[u,v] \in \pi_{[0,\tau]}} [S^{k+1}(\mathbb{X}^\pi)_{[0,v]} - S^{k+1}(\mathbb{X}^\pi)_{[0,u]}] \\ &\stackrel{(*)}{=} \sum_{[u,v] \in \pi_{[0,\tau]}} \left[ \sum_{i=0}^{k+1} S^{k+1-i}(\mathbb{X}^\pi)_{[0,u]} \otimes \frac{\mathbf{X}_{u,v}^{\otimes i}}{i!} - S^{k+1}(\mathbb{X}^\pi)_{[0,u]} \right] \\ &= \sum_{[u,v] \in \pi_{[0,\tau]}} \sum_{i=1}^{k+1} S^{k+1-i}(\mathbb{X}^\pi)_{[0,u]} \otimes \frac{\mathbf{X}_{u,v}^{\otimes i}}{i!} \\ &= \sum_{i=0}^k \frac{1}{(1+i)!} \sum_{[u,v] \in \pi_{[0,\tau]}} S^{k-i}(\mathbb{X}^\pi)_{[0,u]} \otimes \mathbf{X}_{u,v}^{\otimes (1+i)}, \end{aligned}$$

where, in (∗), we use Chen's relation and <sup>S</sup>(<sup>X</sup> π )[u,v] = exp<sup>⊗</sup> Xu,v since <sup>X</sup> π is linear over [u, v] ∈ <sup>π</sup>.

#### B.1.1. PROOF OF THEOREM [2.8](#page-3-1) UNDER *(i)*, *(iii)* OR *(iv)*

Denote by {<sup>π</sup>n, n ≥ <sup>1</sup>} the signature-defining sequence of refining partitions of the interval [0, T]. Without loss of generality, we can consider {<sup>π</sup>n, n ≥ <sup>1</sup>} to be such that <sup>π</sup>n+1 is obtained from <sup>π</sup><sup>n</sup> by adding at most one refinement in each sub-interval, i.e., for each [s, t] ∈ <sup>π</sup>n, either [s, t] ∈ <sup>π</sup>n+1 or [s, u], [u, t] ∈ <sup>π</sup>n+1, for <sup>u</sup> ∈ (s, t). If not, one can consider a super-sequence satisfying this property and then pass to the original subsequence.

In the following, for any <sup>n</sup> ≥ <sup>1</sup> and [s, t] ∈ <sup>π</sup>n, denote by <sup>π</sup>n,[s,t] the restriction of π<sup>n</sup> to [s, t] and, abusing notation slightly, S(<sup>X</sup> <sup>π</sup><sup>n</sup> )[s,t] = S(<sup>X</sup> <sup>π</sup>n,[s,t] )[s,t] .

Let [τ0, τ1] ∈ <sup>π</sup><sup>N</sup> , for <sup>N</sup> ≥ <sup>1</sup>, and note that, for any <sup>k</sup> ≥ <sup>2</sup> and <sup>n</sup> ≥ <sup>N</sup>, we can write

$$S^k(\mathbb{X}^{\pi_{n+1}})_{[\tau_0, \tau_1]} - S^k(\mathbb{X}^{\pi_n})_{[\tau_0, \tau_1]} = \sum_{[s, t] \in \pi_{n, [\tau_0, \tau_1]}} [S^k(\mathbb{X}^{\pi_{n, t}})_{[\tau_0, \tau_1]} - S^k(\mathbb{X}^{\pi_{n, s}})_{[\tau_0, \tau_1]}], \quad (17)$$

where the partitions <sup>π</sup>n,s are defined as <sup>π</sup>n,s <sup>=</sup> <sup>π</sup>n+1,[0,s] ∪ <sup>π</sup>n,[s,T] , i.e., for each [s, t] ∈ <sup>π</sup>n, the partitions <sup>π</sup>n,s and <sup>π</sup>n,t differ by at most one point <sup>u</sup> ∈ (s, t). Using Chen's relation and the definition of the tensor product, we can write for each [s, t] ∈ <sup>π</sup><sup>n</sup> with refinement <sup>u</sup> ∈ (s, t),

$$\begin{aligned} S^k(\mathbb{X}^{\pi_n, t})_{[\tau_0, \tau_1]} - S^k(\mathbb{X}^{\pi_n, s})_{[\tau_0, \tau_1]} \\ = \sum_{\substack{i_1, i_2, i_3 \geq 0 \\ i_1 + i_2 + i_3 = k}} S^{i_1}(\mathbb{X}^{\pi_{n+1}})_{[\tau_0, s]} \otimes [S^{i_2}(\mathbb{X}^{\pi_{n+1}})_{[s, t]} - S^{i_2}(\mathbb{X}^{\pi_n})_{[s, t]}] \otimes S^{i_3}(\mathbb{X}^{\pi_n})_{[t, \tau_1]}. \end{aligned} \quad (18)$$

$$S^{i_2}(\mathbb{X}^{\pi_{n+1}})_{[s,t]} - S^{i_2}(\mathbb{X}^{\pi_n})_{[s,t]} = 0,$$

and applying again Chen's relation when <sup>i</sup><sup>2</sup> ≥ <sup>2</sup> yields

$$S^{i_2}(\mathbb{X}^{\pi_{n+1}})_{[s,t]} = \sum_{j=0}^{i_2} S^j(\mathbb{X}^{\pi_{n+1}})_{[s,u]} \otimes S^{i_2-j}(\mathbb{X}^{\pi_{n+1}})_{[u,t]} = \frac{1}{i_2!} \sum_{j=0}^{i_2} \binom{i_2}{j} \mathbf{X}_{s,u}^{\otimes j} \otimes \mathbf{X}_{u,t}^{\otimes (i_2-j)},$$

where we used the fact that, if <sup>Y</sup> is linear over [s, t], then S(Y)[s,t] = exp<sup>⊗</sup> Ys,t, which also implies

$$S^{i_2}(\mathbb{X}^{\pi_n})_{[s,t]} = \frac{\mathbf{X}_{s,t}^{\otimes i_2}}{i_2!} = \frac{(\mathbf{X}_{s,t} + \mathbf{X}_{u,t})^{\otimes i_2}}{i_2!} = \frac{1}{i_2!} \sum_{\mathcal{I} \in \{0,1\}^{i_2}} \bigotimes_{i \in \mathcal{I}} \left( \mathbf{X}_{s,u}^{\otimes i} \otimes \mathbf{X}_{u,t}^{\otimes (1-i)} \right),$$

denoting by I ∈ {0, <sup>1</sup>} <sup>i</sup><sup>2</sup> a binary number of length <sup>i</sup><sup>2</sup> with |I| <sup>=</sup> P <sup>i</sup>∈I i and recalling that x <sup>⊗</sup><sup>0</sup> = 1, x <sup>⊗</sup><sup>1</sup> = x, for any <sup>x</sup> ∈ <sup>R</sup> d . We hence have that

$$S^{i2}(\mathbb{X}^{\pi_{n+1}})_{[s,t]} - S^{i2}(\mathbb{X}^{\pi_n})_{[s,t]} = \sum_{\mathcal{I} \in \{0,1\}^{i2}} C_{\mathcal{I}} \bigotimes_{i \in \mathcal{I}} (\mathbf{X}_{s,u}^{\otimes i} \otimes \mathbf{X}_{u,t}^{\otimes (1-i)}),$$

where for I ∈ {0, <sup>1</sup>} i2 ,

$$C_{\mathcal{I}} = \begin{cases} \frac{1}{i_2!} \left[ \binom{i_2}{|\mathcal{I}|} - 1 \right], & \text{if } \mathcal{I} = (1, \dots, 1, 0, \dots, 0), \\ -\frac{1}{i_2!}, & \text{otherwise.} \end{cases}$$

Plugging this into Equation [\(17\)](#page-14-2) via [\(18\)](#page-14-3) and noting that <sup>C</sup><sup>I</sup> = 0 for I ∈ {(0, . . . , 0),(1, . . . , 1)}, we can write, for any <sup>N</sup> ≥ <sup>1</sup>, [τ0, τ1] ∈ <sup>π</sup><sup>N</sup> , n ≥ <sup>N</sup> and <sup>k</sup> ≥ <sup>2</sup>,

$$\begin{aligned} & S^k(\mathbb{X}^{\pi_{n+1}})_{[\tau_0, \tau_1]} - S^k(\mathbb{X}^{\pi_n})_{[\tau_0, \tau_1]} \\ &= \sum_{\substack{[s, t] \in \pi_{n, [\tau_0, \tau_1]} \\ u \in (s, t)}} \sum_{\substack{i_1, i_2 \geq 0, i_2 \geq 2 \\ i_1 + i_2 + i_3 = k}} \sum_{\mathcal{I} \in \{0, 1\}^{i_2}} C_{\mathcal{I}} S^{i_1}(\mathbb{X}^{\pi_{n+1}})_{[\tau_0, s]} \otimes \bigotimes_{i \in \mathcal{I}} \left( \mathbf{X}_{s, u}^{\otimes i} \otimes \mathbf{X}_{u, t}^{\otimes (1-i)} \right) \otimes S^{i_3}(\mathbb{X}^{\pi_n})_{[t, \tau_1]}. \end{aligned}$$

We now proceed inductively to show that, for any <sup>i</sup> ∈ {1, . . . , k} and any [τ0, τ1] ∈ <sup>π</sup><sup>N</sup> with <sup>N</sup> ≥ <sup>1</sup>, the sequence {S i (<sup>X</sup> <sup>π</sup><sup>n</sup> )[τ0,τ1] , n ≥ <sup>N</sup>} converges in <sup>L</sup> mk/i with rate O( P n′≥n |<sup>π</sup>n′ | ϵ ) and

$$\sup_{N \geq 1} \sup_{[\tau_0, \tau_1] \in \pi_N} \|S^i(\mathbb{X}^{\pi_N})_{[\tau_0, \tau_1]}\|_{L^{mk/i}} < \infty. \quad (19)$$

k ′ = 1. Note that for [τ0, τ1] ∈ <sup>π</sup><sup>N</sup> with <sup>N</sup> ≥ <sup>1</sup> one has <sup>S</sup> (<sup>X</sup> <sup>π</sup><sup>n</sup> )[τ0,τ1] = X<sup>τ</sup>0,τ<sup>1</sup> , for all <sup>n</sup> ≥ <sup>N</sup>, and

$$\|\mathbf{X}_{\tau_0, \tau_1}\|_{L^{mk}} \lesssim |\tau_1 - \tau_0|^\alpha \leq T^\alpha < \infty,$$

by Assumption [\(A](#page-3-0)α). Hence S 1 (X)[0,T] <sup>=</sup> <sup>X</sup>0,T ∈ <sup>L</sup> mk and the statement holds trivially.

Assume the inductive hypothesis holds for all <sup>i</sup> ∈ {1, . . . , k′} with <sup>k</sup> ′ ∈ {1, . . . , k − <sup>1</sup>}. Then, for each [τ0, τ1] ∈ <sup>π</sup><sup>N</sup> with <sup>N</sup> ≥ <sup>1</sup> and <sup>n</sup> ≥ <sup>N</sup>, let

$$\begin{aligned} & \left\| S^{k'+1}(\mathbb{X}^{\pi_{n+1}})_{[\tau_0, \tau_1]} - S^{k'+1}(\mathbb{X}^{\pi_n})_{[\tau_0, \tau_1]} \right\|_{L^{mk/(k'+1)}} \\ & \leq \sum_{\substack{i_1, i_2 \geq 0, i_2 \geq 2 \\ i_1 + i_2 + i_3 = k' + 1}} \sum_{\substack{\mathcal{I} \in \{0,1\}^{i_2} \\ \mathcal{I} \neq (0, \dots, 0), (1, \dots, 1)}} |C_{\mathcal{I}}| \left\| \sum_{\substack{[s, t] \in \pi_n, [\tau_0, \tau_1] \\ u \in (s, t)}} Z_{[s, t]}^{\mathcal{I}} \right\|_{L^{mk/(k'+1)}}, \end{aligned} \quad (20)$$

where, for each [τ0, τ1] ∈ <sup>π</sup><sup>N</sup> , <sup>π</sup><sup>n</sup> with <sup>n</sup> ≥ <sup>N</sup>, <sup>i</sup>1, i<sup>3</sup> ≥ <sup>0</sup>, i<sup>2</sup> ≥ <sup>2</sup> with <sup>i</sup><sup>1</sup> <sup>+</sup> <sup>i</sup><sup>2</sup> <sup>+</sup> <sup>i</sup><sup>3</sup> <sup>=</sup> <sup>k</sup> ′ + 1 and I ∈ {0, <sup>1</sup>} i2 , we define

$$Z_{[s,t]}^{\mathcal{I}} := S^{i_1}(\mathbb{X}^{\pi_{n+1}})_{[\tau_0,s]} \otimes \bigotimes_{i \in \mathcal{I}} \left( \mathbf{X}_{s,u}^{\otimes i} \otimes \mathbf{X}_{u,t}^{\otimes (1-i)} \right) \otimes S^{i_3}(\mathbb{X}^{\pi_n})_{[t,\tau_1]}, \quad [s,t] \in \pi_{n,[\tau_0,\tau_1]} \text{ with } u \in (s,t),$$

keeping only the dependence on I for notational convenience. Note that, by applying Lemma [B.2,](#page-13-1) the inductive hypothesis and Assumption [\(A](#page-3-0)α),

$$\| \mathbb{Z}_{[s,t]}^{\mathcal{I}} \|_{L^{mk/(k'+1)}} \lesssim \| S^{i_1} (\mathbb{X}^{\pi_{n+1}})_{[\tau_0,s]} \|_{L^{mk/i_1}} \| \mathbf{X}_{s,u} \|_{L^{mk}}^{|\mathcal{I}|} \| \mathbf{X}_{u,t} \|_{L^{mk}}^{i_2-|\mathcal{I}|} \| S^{i_3} (\mathbb{X}^{\pi_n})_{[t,\tau_1]} \|_{L^{mk/i_3}} \lesssim |t-s|^{i_2\alpha},$$

and, hence, each Z I [s,t] ∈ <sup>L</sup> mk/(k ′+1). Moreover, by a simple application of the triangle inequality,

$$\left\| \sum_{\substack{[s,t] \in \pi_{n,[\tau_0,\tau_1]} \\ u \in (s,t)}} Z_{[s,t]}^T \right\|_{L^{mk/(k'+1)}} \lesssim \sum_{\substack{[s,t] \in \pi_{n,[\tau_0,\tau_1]} \\ u \in (s,t)}} |t-s|^{i_2\alpha}. \quad (21)$$

Assumption (i) Hence, if α > <sup>1</sup>/2, we have for each [τ0, τ1] ∈ <sup>π</sup><sup>N</sup> , <sup>π</sup><sup>n</sup> with <sup>n</sup> ≥ <sup>N</sup>, <sup>i</sup>1, i<sup>3</sup> ≥ <sup>0</sup>, i<sup>2</sup> ≥ <sup>2</sup> with i<sup>1</sup> + i<sup>2</sup> + i<sup>3</sup> = k ′ + 1 and I ∈ {0, <sup>1</sup>} i2 ,

$$\left\| \sum_{\substack{[s,t] \in \pi_{n,[\tau_0,\tau_1]} \\ u \in (s,t)}} Z_{[s,t]}^T \right\|_{L^{mk/(k'+1)}} \lesssim |\pi_n|^{2\alpha-1} |\tau_1 - \tau_0|. \quad (22)$$

Assumption (iii) If <sup>α</sup> ∈ (1/3, <sup>1</sup>/2] note that if I is such that <sup>i</sup><sup>2</sup> ≥ <sup>3</sup>, then

$$\left\| \sum_{\substack{[s,t] \in \pi_{n,[\tau_0,\tau_1]} \\ u \in (s,t)}} Z_{[s,t]}^T \right\|_{L^{mk}/(k'+1)} \lesssim |\pi_n|^{3\alpha-1} |\tau_1 - \tau_0|, \quad (23)$$

but if <sup>i</sup><sup>2</sup> = 2 then the bound [\(21\)](#page-16-0) is not strong enough. We can instead apply Lemma [B.1](#page-13-0) with the filtration {G[s,t] , [s, t] ∈ <sup>π</sup>n} defined by

$$\mathcal{G}_{[s,t]} := \mathcal{F}_s \vee \sigma(\mathbf{X}_{v,w}, [v, w] \in \pi_{n,[t,\tau]}),$$

by noting that each Z I [v,w] with <sup>w</sup> ≤ <sup>s</sup> is G[s,t] -measurable and mk/(k ′ + 1) ≥ <sup>2</sup> for all <sup>k</sup> ′ + 1 ≤ <sup>k</sup>. This implies

$$\begin{aligned} \left\| \sum_{\substack{[s,t] \in \pi_{n,[\tau_0,\tau_1]} \\ u \in (s,t)}} Z_{[s,t]}^{\mathcal{I}} \right\|_{L^{mk/(k'+1)}} &\leq \sum_{\substack{[s,t] \in \pi_{n,[\tau_0,\tau_1]} \\ u \in (s,t)}} \left\| \mathbb{E}_{[s,t]} [Z_{[s,t]}^{\mathcal{I}}] \right\|_{L^{mk/(k'+1)}} + \left( \sum_{\substack{[s,t] \in \pi_{n,[\tau_0,\tau_1]} \\ u \in (s,t)}} \|Z_{[s,t]}^{\mathcal{I}}\|_{L^{mk/(k'+1)}}^2 \right)^{1/2} \\ &\leq \sum_{\substack{[s,t] \in \pi_{n,[\tau_0,\tau_1]} \\ u \in (s,t)}} |t - s|^{\beta} + \left( \sum_{\substack{[s,t] \in \pi_{n,[\tau_0,\tau_1]} \\ u \in (s,t)}} |t - s|^{4\alpha} \right)^{1/2} \\ &\leq |\pi_n|^{(\beta-1) \wedge (2\alpha-1/2)} \left( |\tau_1 - \tau_0| + |\tau_1 - \tau_0|^{1/2} \right), \end{aligned} \tag{24}$$

where we used the fact that for I ∈ {(0, 1),(1, 0)},

$$\begin{aligned} & \left\| \mathbb{E}_{\mathcal{G}_{[s,t]}} [Z_{[s,t]}^{\mathcal{I}}] \right\|_{L^{mk/(k'+1)}} \\ & \stackrel{(i)}{=} \left\| S^{i_1} (\mathbb{X}^{\pi_{n+1}})_{[\tau_0,s]} \otimes \mathbb{E}_{\mathcal{G}_{[s,t]}} \left[ \bigotimes_{i \in \mathcal{I}} \left( \mathbf{X}_{s,u}^{\otimes i} \otimes \mathbf{X}_{u,t}^{\otimes (1-i)} \right) \right] \otimes S^{i_3} (\mathbb{X}^{\pi_n})_{[t,\tau_1]} \right\|_{L^{mk/(k'+1)}} \\ & \stackrel{(ii)}{\leq} \left\| S^{i_1} (\mathbb{X}^{\pi_{n+1}})_{[\tau_0,s]} \right\|_{L^{mk/i_1}} \left\| \mathbb{E}_{\mathcal{G}_{[s,t]}} \left[ \bigotimes_{i \in \mathcal{I}} \left( \mathbf{X}_{s,u}^{\otimes i} \otimes \mathbf{X}_{u,t}^{\otimes (1-i)} \right) \right] \right\|_{L^{mk/2}} \left\| S^{i_3} (\mathbb{X}^{\pi_n})_{[t,\tau_1]} \right\|_{L^{mk/i_3}} \\ & \stackrel{(iii)}{\lesssim} \left\| \mathbb{E}_{\mathcal{G}_{[s,t]}} [\mathbf{X}_{s,u} \otimes \mathbf{X}_{u,t}] \right\|_{L^{mk/2}} \\ & \stackrel{(iv)}{\lesssim} \left\| \mathbb{E}_{\mathcal{F}_{0,s} \vee \mathcal{F}_{t,T}} [\mathbf{X}_{s,u} \otimes \mathbf{X}_{u,t}] \right\|_{L^{mk/2}} \end{aligned}$$

$$\stackrel{(v)}{\lesssim} |t - s|^\beta,$$

by using in (i) measurability of S <sup>i</sup><sup>1</sup> (<sup>X</sup> <sup>π</sup>n+1 )[τ0,s] and S <sup>i</sup><sup>3</sup> (<sup>X</sup> <sup>π</sup><sup>n</sup> )[t,τ1] with respect to G[s,t] , in (ii) Holder inequality for ¨ tensors Lemma [B.2,](#page-13-1) in (iii) the inductive assumption [\(19\)](#page-15-0) and the fact that ∥<sup>A</sup> ⊗ <sup>B</sup>∥ <sup>=</sup> ∥<sup>B</sup> ⊗ <sup>A</sup>∥ for any <sup>A</sup>, <sup>B</sup> ∈ <sup>R</sup> d , in (iv) the tower property and the contractive property of conditional expectation applied to G[s,t] ⊆ F0,s ∨ Ft,T and in (v) Assumption [\(A](#page-3-0)β). Combining bound [\(24\)](#page-16-1) when <sup>i</sup><sup>2</sup> = 2 and bound [\(23\)](#page-16-2) when <sup>i</sup><sup>2</sup> ≥ <sup>3</sup> with <sup>α</sup> ∈ (1/3, <sup>1</sup>/2], it follows that for each [τ0, τ1] ∈ <sup>π</sup><sup>N</sup> , <sup>π</sup><sup>n</sup> with <sup>n</sup> ≥ <sup>N</sup>, <sup>i</sup>1, i<sup>3</sup> ≥ <sup>0</sup>, i<sup>2</sup> ≥ <sup>2</sup> with <sup>i</sup><sup>1</sup> <sup>+</sup> <sup>i</sup><sup>2</sup> <sup>+</sup> <sup>i</sup><sup>3</sup> <sup>=</sup> <sup>k</sup> ′ + 1 and I ∈ {0, <sup>1</sup>} i2 ,

$$\left\| \sum_{\substack{[s,t] \in \pi_n, [\tau_0, \tau_1] \\ u \in (s,t)}} Z_{[s,t]}^T \right\|_{L^{mk/(k'+1)}} \lesssim |\pi_n|^{(\beta-1)\wedge(3\alpha-1)} |\tau_1 - \tau_0|. \quad (25)$$

Assumption (iv) A similar reasoning can be applied when <sup>α</sup> ∈ (1/4, <sup>1</sup>/3] (and <sup>k</sup> ≥ <sup>3</sup>) by considering the cases <sup>i</sup><sup>2</sup> ≥ <sup>4</sup>, <sup>i</sup><sup>2</sup> = 3 and <sup>i</sup><sup>2</sup> = 2 separately. The case <sup>i</sup><sup>2</sup> ≥ <sup>4</sup> follows directly from [\(21\)](#page-16-0), the case <sup>i</sup><sup>2</sup> = 2 follows from [\(24\)](#page-16-1) and the case i<sup>2</sup> = 3 can be shown in the same way as i<sup>2</sup> = 2 with the only difference being that we require Assumption [\(A](#page-3-0)γ) to show that, for I ∈ {(0, <sup>0</sup>, 1),(0, <sup>1</sup>, 0),(1, <sup>0</sup>, 0)},

$$\left\| \mathbb{E}_{\mathcal{G}_{[s,t]}} [Z_{[s,t]}^T] \right\|_{L^{mk/(k'+1)}} \lesssim \left\| \mathbb{E}_{\mathcal{F}_{0,s \vee \mathcal{F}_{t,T}}} [\mathbf{X}_{s,u} \otimes \mathbf{X}_{u,t}^{\otimes 2}] \right\|_{L^{mk/3}} \lesssim |t-s|^\gamma,$$

and, for I ∈ {(0, <sup>1</sup>, 1),(1, <sup>0</sup>, 1),(1, <sup>1</sup>, 0)},

$$\left\| \mathbb{E}_{\mathcal{G}_{[s,t]}}[Z_{[s,t]}^T] \right\|_{L^{mk/(k'+1)}} \lesssim \left\| \mathbb{E}_{\mathcal{F}_{0,s} \vee \mathcal{F}_{t,T}}[\mathbf{X}_{s,u}^{\otimes 2} \otimes \mathbf{X}_{u,t}] \right\|_{L^{mk/3}} \lesssim |t-s|^\gamma,$$

so that applying again Lemma [B.1,](#page-13-0)

$$\begin{aligned} \left\| \sum_{\substack{[s,t] \in \pi_{n,[\tau_0,\tau_1]} \\ u \in (s,t)}} Z_{[s,t]}^{\mathcal{I}} \right\|_{L^{mk/(k'+1)}} &\leq \sum_{\substack{[s,t] \in \pi_{n,[\tau_0,\tau_1]} \\ u \in (s,t)}} \left\| \mathbb{E}_{\mathcal{G}_{[s,t]}} [Z_{[s,t]}^{\mathcal{I}}] \right\|_{L^{mk/(k'+1)}} + \left( \sum_{\substack{[s,t] \in \pi_{n,[\tau_0,\tau_1]} \\ u \in (s,t)}} \|Z_{[s,t]}^{\mathcal{I}}\|_{L^{mk/(k'+1)}}^2 \right)^{1/2} \\ &\leq \sum_{\substack{[s,t] \in \pi_{n,[\tau_0,\tau_1]} \\ u \in (s,t)}} |t-s|^{\gamma} + \left( \sum_{\substack{[s,t] \in \pi_{n,[\tau_0,\tau_1]} \\ u \in (s,t)}} |t-s|^{6\alpha} \right)^{1/2} \\ &\leq |\pi_n|^{(\gamma-1)\wedge(3\alpha-1/2)} \left( |\tau_1 - \tau_0| + |\tau_1 - \tau_0|^{1/2} \right). \end{aligned} \quad (26)$$

Combining the cases <sup>i</sup><sup>2</sup> = 2, <sup>i</sup><sup>2</sup> = 3 and <sup>i</sup><sup>2</sup> ≥ <sup>4</sup> when <sup>α</sup> ∈ (1/4, <sup>1</sup>/3] yields, for each [τ0, τ1] ∈ <sup>π</sup><sup>N</sup> , <sup>π</sup><sup>n</sup> with <sup>n</sup> ≥ <sup>N</sup>, <sup>i</sup>1, i<sup>3</sup> ≥ <sup>0</sup>, i<sup>2</sup> ≥ <sup>2</sup> with <sup>i</sup><sup>1</sup> <sup>+</sup> <sup>i</sup><sup>2</sup> <sup>+</sup> <sup>i</sup><sup>3</sup> <sup>=</sup> <sup>k</sup> ′ + 1 and I ∈ {0, <sup>1</sup>} i2 ,

$$\left\| \sum_{\substack{[s,t] \in \pi_n, [\tau_0, \tau_1] \\ u \in (s, t)}} Z_{[s,t]}^T \right\|_{L^{mk}(k'+1)} \lesssim |\pi_n|^{(\beta-1)\wedge(\gamma-1)\wedge(4\alpha-1)} |\tau_1 - \tau_0|. \quad (27)$$

Defining ϵ as in Equation [\(9\)](#page-3-3), we can plug bounds [\(22\)](#page-16-3), [\(25\)](#page-17-0) and [\(27\)](#page-17-1) into Equation [\(20\)](#page-15-1) to deduce that

$$\left\| S^{k'+1}(\mathbb{X}^{\pi_{n+1}})_{[\tau_0, \tau_1]} - S^{k'+1}(\mathbb{X}^{\pi_n})_{[\tau_0, \tau_1]} \right\|_{L^{mk/(k'+1)}} \lesssim |\tau_1 - \tau_0| |\pi_n|^\epsilon,$$

and, hence, under the assumption that P n≥1 |πn| <sup>ϵ</sup> <sup>&</sup>lt; ∞, for any [τ0, τ1] ∈ <sup>π</sup><sup>N</sup> with <sup>N</sup> ≥ <sup>1</sup>, the sequence {S k ′+1(<sup>X</sup> <sup>π</sup><sup>n</sup> )[τ0,τ1] , n ≥ <sup>N</sup>} is Cauchy in <sup>L</sup> mk/(k ′+1). Since L mk/(k ′+1) is a Banach space the sequence converges in L mk/(k ′+1) to S k ′+1(X)[τ0,τ1] ∈ <sup>L</sup> mk/(k ′+1) (by uniqueness of limits) with rate

$$\left\| S^{k'+1}(\mathbb{X})[\tau_0, \tau_1] - S^{k'+1}(\mathbb{X}^{\pi_n})[\tau_0, \tau_1] \right\|_{L^{mk/(k'+1)}}$$

$$\leq \sum_{n' \geq n} \left\| S^{k'+1}(\mathbb{X}^{\pi_{n'+1}})_{[\tau_0, \tau_1]} - S^{k'+1}(\mathbb{X}^{\pi_{n'}})_{[\tau_0, \tau_1]} \right\|_{L^{mk/(k'+1)}} \lesssim |\tau_1 - \tau_0| \sum_{n' \geq n} |\pi_{n'}|^{\epsilon}.$$

And, to complete the inductive step for k ′ + 1, note that for all <sup>N</sup> ≥ <sup>1</sup> and [τ0, τ1] ∈ <sup>π</sup><sup>N</sup> ,

$$\begin{aligned} \left\| S^{k'+1}(\mathbb{X}^{\pi_N})_{[\tau_0, \tau_1]} \right\|_{L^{mk/(k'+1)}} &\lesssim \left\| S^{k'+1}(\mathbb{X})_{[\tau_0, \tau_1]} \right\|_{L^{mk/(k'+1)}} + |\tau_1 - \tau_0| \sum_{n' \geq N} |\pi_{n'}|^\epsilon \\ &\lesssim \left\| S^{k'+1}(\mathbb{X})_{[\tau_0, \tau_1]} \right\|_{L^{mk/(k'+1)}} + T \sum_{n' \geq 1} |\pi_{n'}|^\epsilon. \end{aligned}$$

#### B.1.2. PROOF OF THEOREM [2.8](#page-3-1) UNDER *(ii)*

In what follows we shall simplify notation and denote <sup>E</sup>F0,t by <sup>E</sup>t.

Denote by {<sup>π</sup>n, n ≥ <sup>1</sup>} the signature-defining sequence of refining partitions of the interval [0, T]. Without loss of generality, we can consider {<sup>π</sup>n, n ≥ <sup>1</sup>} to be such that <sup>π</sup>n+1 is obtained from <sup>π</sup><sup>n</sup> by adding at most one refinement in each sub-interval, i.e. for each [s, t] ∈ <sup>π</sup><sup>n</sup> either [s, t] ∈ <sup>π</sup>n+1 or [s, u], [u, t] ∈ <sup>π</sup>n+1 for <sup>u</sup> ∈ (s, t). If not, one can consider a super-sequence satisfying this property and then pass to the original subsequence.

We start by showing inductively that, for any <sup>i</sup> ∈ {1, . . . , k},

$$\sup_{n \geq 1} \sup_{\tau \in \pi_n} \|S^i(\mathbb{X}^{\pi_n})_{[0,\tau]}\|_{L^{mk/i}} < \infty. \quad (28)$$

Note that the case <sup>i</sup> = 1 is trivial since, for any <sup>n</sup> ≥ <sup>1</sup> and <sup>τ</sup> ∈ <sup>π</sup>n, by [\(A](#page-3-0)α)

$$\|S^1(\mathbb{X}^{\pi_n})_{[0,\tau]}\|_{L^{mk}} = \|\mathbf{X}_{0,\tau}\|_{L^{mk}} \lesssim \tau^\alpha \lesssim T^\alpha.$$

Next, for the inductive step, assume that [\(28\)](#page-18-1) holds for all <sup>i</sup> ∈ {1, . . . k′} with <sup>k</sup> ′ ≤ <sup>k</sup> − <sup>1</sup>. Then, by using Lemma [B.3,](#page-14-1) we can bound for any <sup>n</sup> ≥ <sup>1</sup> and <sup>τ</sup> ∈ <sup>π</sup>n,

$$\begin{aligned} & \|S^{k'+1}(\mathbb{X}^{\pi_n})_{[0,\tau]}\|_{L^{mk/(k'+1)}} \\ & \stackrel{(i)}{\leq} \sum_{i=0}^{k'} \frac{1}{(1+i)!} \left\| \sum_{[u,v] \in \pi_{n,[0,\tau]}} S^{k'-i}(\mathbb{X}^{\pi_n})_{[0,u]} \otimes \mathbf{X}_{u,v}^{\otimes(i+1)} \right\|_{L^{mk/(k'+1)}} \\ & \stackrel{(ii)}{\leq} \sum_{[u,v] \in \pi_{n,[0,\tau]}} \left\| S^{k'}(\mathbb{X}^{\pi_n})_{[0,u]} \otimes \mathbb{E}_u[\mathbf{X}_{u,v}] \right\|_{L^{mk/(k'+1)}} \\ & \quad + \left( \sum_{[u,v] \in \pi_{n,[0,\tau]}} \left\| S^{k'}(\mathbb{X}^{\pi_n})_{[0,u]} \otimes \mathbf{X}_{u,v} \right\|_{L^{mk/(k'+1)}}^2 \right)^{1/2} \\ & \quad + \sum_{i=1}^{k'} \frac{1}{(1+i)!} \sum_{[u,v] \in \pi_{n,[0,\tau]}} \left\| S^{k'-i}(\mathbb{X}^{\pi_n})_{[0,u]} \otimes \mathbf{X}_{u,v}^{\otimes(i+1)} \right\|_{L^{mk/(k'+1)}} \\ & \stackrel{(iii)}{\lesssim} \sum_{[u,v] \in \pi_{n,[0,\tau]}} \|\mathbb{E}_u[\mathbf{X}_{u,v}]\|_{L^{mk}} + \left( \sum_{[u,v] \in \pi_{n,[0,\tau]}} \|\mathbf{X}_{u,v}\|_{L^p}^2 \right)^{1/2} + \sum_{i=1}^{k'} \sum_{[u,v] \in \pi_{n,[0,\tau]}} \|\mathbf{X}_{u,v}\|_{L^{mk}}^{i+1} \\ & \stackrel{(iv)}{\lesssim} \sum_{[u,v] \in \pi_{n,[0,\tau]}} |v-u|^\delta + \left( \sum_{[u,v] \in \pi_{n,[0,\tau]}} |v-u|^{2\alpha} \right)^{1/2} + \sum_{i=1}^{k'} \sum_{[u,v] \in \pi_{n,[0,\tau]}} |v-u|^{(i+1)\alpha} \\ & \stackrel{(v)}{\lesssim} \tau + \sqrt{\tau} + \tau \lesssim T + \sqrt{T}, \end{aligned}$$

where in (i) we applied the triangle inequality, in (ii) we bounded the i = 0 term by applying Lemma [B.1](#page-13-0) to the sequence of random variables

$$Z_{[u,v]} := S^{k'}(\mathbb{X}^{\pi_n})_{[0,u]} \otimes \mathbf{X}_{u,v} \in L^{mk/(k'+1)},$$

with filtration {Fu, [u, v] ∈ <sup>π</sup>n,[0,τ]} and we bounded the <sup>i</sup> = 1, . . . , k′ terms by applying the triangle inequality, in (iii) we applied the Holder inequality given in Lemma ¨ [B.2](#page-13-1) and the inductive hypothesis Equation [\(28\)](#page-18-1) for all signature levels up to k ′ , in (iv) we used Assumptions [\(A](#page-3-0)α) and [\(A](#page-3-0)δ).

Proceeding again by induction, we will show the conclusion of the theorem holds by proving the stronger statement: For each <sup>i</sup> ∈ {1, . . . , k}, for all <sup>N</sup> ≥ <sup>1</sup>, <sup>τ</sup> ∈ <sup>π</sup><sup>N</sup> and <sup>n</sup> ≥ <sup>N</sup>,

$$\|S^i(\mathbb{X}^{\pi_{n+1}})_{[0,\tau]} - S^i(\mathbb{X}^{\pi_n})_{[0,\tau]}\|_{L^{m_k/i}} \lesssim |\pi_n|^\epsilon. \quad (29)$$

The case k ′ = 1 is again trivial since, for all <sup>N</sup> ≥ <sup>1</sup>, <sup>τ</sup> ∈ <sup>π</sup><sup>N</sup> and <sup>n</sup> ≥ <sup>N</sup>, <sup>S</sup> 1 (<sup>X</sup> <sup>π</sup>n+1 )[0,τ] = X0,τ , and hence

$$\|S^1(\mathbb{X}^{\pi_{n+1}})_{[0,\tau]} - S^1(\mathbb{X}^{\pi_n})_{[0,\tau]}\|_{L^{mk}} = 0.$$

For the inductive step, assume Equation [\(29\)](#page-19-0) holds for all <sup>i</sup> ∈ {1, . . . , k′} with <sup>k</sup> ′ ≤ <sup>k</sup>. Fix <sup>N</sup> ≥ <sup>1</sup>, <sup>τ</sup> ∈ <sup>π</sup><sup>N</sup> and <sup>n</sup> ≥ <sup>N</sup>, then we can write the telescoping sum

$$S^{k'+1}(\mathbb{X}^{\pi_{n+1}})_{[0,\tau]} - S^{k'+1}(\mathbb{X}^{\pi_n})_{[0,\tau]} = \sum_{[s,t] \in \pi_{n,[0,\tau]}} \left[ S^{k'+1}(\mathbb{X}^{\pi_{n,t}})_{[0,\tau]} - S^{k'+1}(\mathbb{X}^{\pi_{n,s}})_{[0,\tau]} \right], \quad (30)$$

where the partitions <sup>π</sup>n,s are defined as <sup>π</sup>n,s <sup>=</sup> <sup>π</sup>n+1,[0,s] ∪ <sup>π</sup>n,[s,T] , i.e. for each [s, t] ∈ <sup>π</sup>n, the partitions <sup>π</sup>n,s and <sup>π</sup>n,t differ by at most one point <sup>u</sup> ∈ (s, t). Note that, for each [s, t] ∈ <sup>π</sup><sup>n</sup> with refinement <sup>u</sup> ∈ (s, t), we can apply Lemma [B.3](#page-14-1) to write

$$\begin{aligned} & S^{k'+1}(\mathbb{X}^{\pi_{n,t}})_{[0,\tau]} - S^{k'+1}(\mathbb{X}^{\pi_{n,s}})_{[0,\tau]} \\ &= \sum_{i=0}^{k'} \frac{1}{(1+i)!} \left\{ S^{k'-i}(\mathbb{X}^{\pi_{n,t}})_{[0,s]} \otimes \mathbf{X}_{s,u}^{\otimes(1+i)} + S^{k'-i}(\mathbb{X}^{\pi_{n,t}})_{[0,u]} \otimes \mathbf{X}_{u,t}^{\otimes(1+i)} - S^{k'-i}(\mathbb{X}^{\pi_{n,s}})_{[0,s]} \otimes \mathbf{X}_{s,t}^{\otimes(1+i)} \right. \\ &\quad \left. + \sum_{[v,w] \in \pi_{n,[t,\tau]}} \left[ S^{k'-i}(\mathbb{X}^{\pi_{n,t}})_{[0,v]} - S^{k'-i}(\mathbb{X}^{\pi_{n,s}})_{[0,v]} \right] \otimes \mathbf{X}_{v,w}^{\otimes(1+i)} \right\} \\ &= \sum_{i=0}^{k'} \frac{1}{(1+i)!} \left\{ S^{k'-i}(\mathbb{X}^{\pi_{n+1}})_{[0,s]} \otimes \left[ \mathbf{X}_{s,u}^{\otimes(1+i)} - \mathbf{X}_{s,t}^{\otimes(1+i)} \right] + \sum_{j=0}^{k'-i} S^{k'-i-j}(\mathbb{X}^{\pi_{n+1}})_{[0,s]} \otimes \frac{\mathbf{X}_{s,u}^{\otimes j}}{j!} \otimes \mathbf{X}_{u,t}^{\otimes(1+i)} \right. \\ &\quad \left. + \sum_{[v,w] \in \pi_{n,[t,\tau]}} \left[ S^{k'-i}(\mathbb{X}^{\pi_{n,t}})_{[0,v]} - S^{k'-i}(\mathbb{X}^{\pi_{n,s}})_{[0,v]} \right] \otimes \mathbf{X}_{v,w}^{\otimes(1+i)} \right\} \\ &= \sum_{i=0}^{k'} \frac{1}{(1+i)!} \left\{ S^{k'-i}(\mathbb{X}^{\pi_{n+1}})_{[0,s]} \otimes \left[ \mathbf{X}_{s,u}^{\otimes(1+i)} + \mathbf{X}_{u,t}^{\otimes(1+i)} - \mathbf{X}_{s,t}^{\otimes(1+i)} \right] \right. \\ &\quad \left. + \sum_{j=0}^{k'-i-1} \frac{1}{(1+j)!} S^{k'-i-j-1}(\mathbb{X}^{\pi_{n+1}})_{[0,s]} \otimes \mathbf{X}_{s,u}^{\otimes(1+j)} \otimes \mathbf{X}_{u,t}^{\otimes(1+i)} \right. \\ &\quad \left. + \sum_{[v,w] \in \pi_{n,[t,\tau]}} \left[ S^{k'-i}(\mathbb{X}^{\pi_{n,t}})_{[0,v]} - S^{k'-i}(\mathbb{X}^{\pi_{n,s}})_{[0,v]} \right] \otimes \mathbf{X}_{v,w}^{\otimes(1+i)} \right\} \\ &= \sum_{i=0}^{k'} \frac{1}{(1+i)!} \left\{ -S^{k'-i}(\mathbb{X}^{\pi_{n+1}})_{[0,s]} \otimes \sum_{\substack{\mathcal{I} \in \{0,1\}^{1+i} \\ \mathcal{I} \neq (0,\dots,0), (1,\dots,1)}} \bigotimes_{l \in \mathcal{I}} \left( \mathbf{X}_{s,u}^{\otimes l} \otimes \mathbf{X}_{u,t}^{\otimes(1-l)} \right) \right. \\ &\quad \left. + \sum_{j=0}^{k'-i-1} \frac{1}{(1+j)!} S^{k'-i-j-1}(\mathbb{X}^{\pi_{n+1}})_{[0,s]} \otimes \mathbf{X}_{s,u}^{\otimes(1+j)} \otimes \mathbf{X}_{u,t}^{\otimes(1+i)} \right\} \end{aligned}$$

$$+ \sum_{[v,w] \in \pi_n, [t,r]} \left[ S^{k'-i}(\mathbb{X}^{\pi_{n,t}})_{[0,v]} - S^{k'-i}(\mathbb{X}^{\pi_{n,s}})_{[0,v]} \right] \otimes \mathbf{X}_{v,w}^{\otimes(1+i)} \Biggr\},$$

by noting that, for all [v, w] ∈ <sup>π</sup><sup>n</sup> with <sup>w</sup> ≤ <sup>s</sup>, <sup>S</sup>(<sup>X</sup> <sup>π</sup>n,t )[0,v] = S(<sup>X</sup> <sup>π</sup>n,s )[0,v] = S(<sup>X</sup> <sup>π</sup>n+1 )[0,v] and when applying Chen's relation to S(<sup>X</sup> <sup>π</sup>n,t )[0,u] , setting S(<sup>X</sup> <sup>π</sup>n,t )[s,u] = exp<sup>⊗</sup> Xs,u since <sup>X</sup> <sup>π</sup>n,t is linear over [s, u] ∈ <sup>π</sup>n,t. Plugging this expression into Equation [\(30\)](#page-19-1) and exchanging the orders of the summations we obtain

$$\begin{aligned} & S^{k'+1}(\mathbb{X}^{\pi_{n+1}})_{[0,\tau]} - S^{k'+1}(\mathbb{X}^{\pi_n})_{[0,\tau]} \\ &= \sum_{i=0}^{k'} \frac{1}{(1+i)!} \left\{ - \sum_{\substack{\mathcal{I} \in \{0,1\}^{1+i} \\ \mathcal{I} \neq (0,\dots,0), (1,\dots,1)}} \sum_{\substack{[s,t] \in \pi_{n,[0,\tau]} \\ u \in (s,t)}} S^{k'-i}(\mathbb{X}^{\pi_{n+1}})_{[0,s]} \otimes \bigotimes_{l \in \mathcal{I}} \left( \mathbf{X}_{s,u}^{\otimes l} \otimes \mathbf{X}_{u,t}^{\otimes (1-l)} \right) \right. \\ &\quad + \sum_{j=0}^{k'-i-1} \frac{1}{(1+j)!} \sum_{\substack{[s,t] \in \pi_{n,[0,\tau]} \\ u \in (s,t)}} S^{k'-i-j-1}(\mathbb{X}^{\pi_{n+1}})_{[0,s]} \otimes \mathbf{X}_{s,u}^{\otimes (1+j)} \otimes \mathbf{X}_{u,t}^{\otimes (1+i)} \\ &\quad \left. + \sum_{[v,w] \in \pi_{n,[0,\tau]}} \left( \sum_{[s,t] \in \pi_{n,[0,v]}} \left[ S^{k'-i}(\mathbb{X}^{\pi_{n,t}})_{[0,v]} - S^{k'-i}(\mathbb{X}^{\pi_{n,s}})_{[0,v]} \right] \right) \otimes \mathbf{X}_{v,w}^{\otimes (1+i)} \right\} \\ &= \sum_{i=0}^{k'} \frac{1}{(1+i)!} \left\{ - \sum_{\substack{\mathcal{I} \in \{0,1\}^{1+i} \\ \mathcal{I} \neq (0,\dots,0), (1,\dots,1)}} \sum_{\substack{[s,t] \in \pi_{n,[0,\tau]} \\ u \in (s,t)}} S^{k'-i}(\mathbb{X}^{\pi_{n+1}})_{[0,s]} \otimes \bigotimes_{l \in \mathcal{I}} \left( \mathbf{X}_{s,u}^{\otimes l} \otimes \mathbf{X}_{u,t}^{\otimes (1-l)} \right) \right. \\ &\quad + \sum_{j=0}^{k'-i-1} \frac{1}{(1+j)!} \sum_{\substack{[s,t] \in \pi_{n,[0,\tau]} \\ u \in (s,t)}} S^{k'-i-j-1}(\mathbb{X}^{\pi_{n+1}})_{[0,s]} \otimes \mathbf{X}_{s,u}^{\otimes (1+j)} \otimes \mathbf{X}_{u,t}^{\otimes (1+i)} \\ &\quad \left. + \sum_{[v,w] \in \pi_{n,[0,\tau]}} \left[ S^{k'-i}(\mathbb{X}^{\pi_{n+1}})_{[0,v]} - S^{k'-i}(\mathbb{X}^{\pi_n})_{[0,v]} \right] \otimes \mathbf{X}_{v,w}^{\otimes (1+i)} \right\} \\ &= \sum_{i=0}^{k'} \frac{1}{(1+i)!} \left\{ \sum_{\substack{\mathcal{I} \in \{0,1\}^{1+i} \\ \mathcal{I} \neq (0,\dots,0), (1,\dots,1)}} \sum_{\substack{[s,t] \in \pi_{n,[0,\tau]} \\ u \in (s,t)}} Z_{[s,t]}^{1,\mathcal{I}} + \sum_{j=0}^{k'-i-1} \frac{1}{(1+j)!} \sum_{\substack{[s,t] \in \pi_{n,[0,\tau]} \\ u \in (s,t)}} Z_{[s,t]}^{2,i,j} + \sum_{[v,w] \in \pi_{n,[0,\tau]}} Z_{[v,w]}^{3,i} \right\}, \end{aligned}$$

We thus proceed to bound each of the summation terms over πn,[0,τ] using Lemma [B.1](#page-13-0) and Assumptions [\(A](#page-3-0)α) and [\(A](#page-3-0)δ).

Let I ∈ {0, <sup>1</sup>} 1+<sup>i</sup> with I ̸= (0, . . . , 0),(1, . . . , 1) with <sup>i</sup> ∈ {1, . . . , k′}. Note that for all [s, t] ∈ <sup>π</sup>n,[0,τ]

$$\|Z_{[s,t]}^{1,\mathcal{I}}\|_{L^{mk/(k'+1)}} \leq \left\| S^{k'-i}(\mathbb{X}^{\pi_{n+1}})_{[0,s]} \right\|_{L^{mk/(k'-i)}} \|\mathbf{X}_{s,u}\|_{L^{mk}}^{|\mathcal{I}|} \|\mathbf{X}_{u,t}\|_{L^{mk}}^{1+i-|\mathcal{I}|} \lesssim |t-s|^{(1+i)\alpha}, \quad (31)$$

by applying Lemma [B.2,](#page-13-1) the uniform bound [\(28\)](#page-18-1) and Assumption [\(A](#page-3-0)α), hence Z 1,I [s,t] ∈ L mk/(k ′+1). When i = 1 we can thus apply Lemma [B.1](#page-13-0) to the sequence {<sup>Z</sup> 1,I [s,t] , [s, t] ∈ <sup>π</sup>n,[0,τ] , u ∈ (s, t)} with filtration {Fu, [s, t] ∈ <sup>π</sup>n,[0,τ] , u ∈ (s, t)} to bound

$$\begin{aligned} \left\| \sum_{\substack{[s,t] \in \pi_n, [0, \tau] \\ u \in (s, t)}} Z_{[s,t]}^{1,\mathcal{I}} \right\|_{L^{mk/(k'+1)}} &\leq \sum_{\substack{[s,t] \in \pi_n, [0, \tau] \\ u \in (s, t)}} \|\mathbb{E}_u [Z_{[s,t]}^{1,\mathcal{I}}]\|_{L^{mk/(k'+1)}} + \left( \sum_{\substack{[s,t] \in \pi_n, [0, \tau] \\ u \in (s, t)}} \|Z_{[s,t]}^{1,\mathcal{I}}\|_{L^{mk/(k'+1)}}^2 \right)^{1/2} \\ &\lesssim \sum_{\substack{[s,t] \in \pi_n, [0, \tau] \\ u \in (s, t)}} |t - s|^{\alpha+\delta} + \left( \sum_{\substack{[s,t] \in \pi_n, [0, \tau] \\ u \in (s, t)}} |t - s|^{4\alpha} \right)^{1/2} \\ &\lesssim |\pi_n|^{\alpha+\delta-1} \tau + |\pi_n|^{2\alpha-1/2} \sqrt{\tau} \lesssim |\pi_n|^\epsilon (\tau + \sqrt{\tau}), \end{aligned} \quad (32)$$

where we used Equation [\(31\)](#page-20-0) with i = 1 and

$$\begin{aligned} \|\mathbb{E}_u[Z_{[s,t]}^{1,\mathcal{I}}]\|_{L^{mk/(k'+1)}} &= \left\| S^{k'-i}(\mathbb{X}^{\pi_{n+1}})_{[0,s]} \otimes \mathbf{X}_{s,u} \otimes \mathbb{E}_u[\mathbf{X}_{u,t}] \right\|_{L^{mk/(k'+1)}} \\ &\leq \left\| S^{k'-1}(\mathbb{X}^{\pi_{n+1}})_{[0,s]} \right\|_{L^{mk/(k'-1)}} \|\mathbf{X}_{s,u}\|_{L^{mk}} \|\mathbb{E}_u[\mathbf{X}_{u,t}]\|_{L^{mk}} \lesssim |t-s|^{\alpha+\delta}, \end{aligned}$$

by applying Lemma [B.2,](#page-13-1) the uniform bound [\(28\)](#page-18-1) and Assumptions [\(A](#page-3-0)α) and [\(A](#page-3-0)δ). When <sup>i</sup> ≥ <sup>2</sup> we can directly apply the triangle inequality and Equation [\(31\)](#page-20-0) to bound

$$\begin{aligned} \left\| \sum_{\substack{[s,t] \in \pi_{n_*[0,\tau]} \\ u \in (s,t)}} Z_{[s,t]}^{1,\mathcal{I}} \right\|_{L^{mk/(k'+1)}} &\leq \sum_{\substack{[s,t] \in \pi_{n_*[0,\tau]} \\ u \in (s,t)}} \|Z_{[s,t]}^{1,\mathcal{I}}\|_{L^{mk/(k'+1)}} \\ &\lesssim \sum_{\substack{[s,t] \in \pi_{n_*[0,\tau]} \\ u \in (s,t)}} |t - s|^{(1+i)\alpha} \lesssim |\pi_n|^{3\alpha-1} \tau \lesssim |\pi_n|^\epsilon \tau. \end{aligned} \quad (33)$$

Next, let <sup>i</sup> ∈ {0, . . . , k′} and <sup>j</sup> ∈ {0, . . . , k′ − <sup>i</sup> − <sup>1</sup>}. We can proceed exactly as for <sup>Z</sup> 1,I [s,t] (applying Lemma [B.1](#page-13-0) when <sup>i</sup> <sup>=</sup> <sup>j</sup> = 0 or the triangle inequality when <sup>i</sup> <sup>+</sup> <sup>j</sup> ≥ <sup>1</sup>) to show that under Assumptions [\(A](#page-3-0)α) and [\(A](#page-3-0)δ),

$$\left\| \sum_{\substack{[s,t] \in \pi_{n,[0,\tau]} \\ u \in (s,t)}} Z_{[s,t]}^{2,i,j} \right\|_{L^{mk/(k'+1)}} \lesssim |\pi_n|^\epsilon (\tau + \sqrt{\tau}). \quad (34)$$

Finally, let <sup>i</sup> ∈ {0, . . . , k′}. We proceed in a similar way as for <sup>Z</sup> 1,I [s,t] and Z 2,i,j [s,t] but using the inductive hypothesis [\(29\)](#page-19-0) instead of the bound [\(28\)](#page-18-1). Note that for all [s, t] ∈ <sup>π</sup>n,[0,τ]

$$\|Z_{[s,t]}^{3,i}\|_{L^{mk/(k'+1)}} \leq \left\| S^{k'-i}(\mathbb{X}^{\pi_{n+1}})_{[0,s]} - S^{k'-i}(\mathbb{X}^{\pi_n})_{[0,s]} \right\|_{L^{mk/(k'-i)}} \|\mathbf{X}_{s,t}\|_{L^{mk}}^{1+i} \lesssim |\pi_n|^\epsilon |t-s|^{(1+i)\alpha}, \quad (35)$$

by applying Lemma [B.2,](#page-13-1) the inductive hypothesis [\(29\)](#page-19-0) and Assumption [\(A](#page-3-0)α), hence Z 3,i [s,t] ∈ L mk/(k ′+1). When i = 0 we can hence apply Lemma [B.1](#page-13-0) to the sequence {<sup>Z</sup> 3,i [s,t] , [s, t] ∈ <sup>π</sup>n,[0,τ]} with filtration {Fs, [s, t] ∈ <sup>π</sup>n,[0,τ]} to bound

$$\begin{aligned} \left\| \sum_{[s,t] \in \pi_{n,[0,\tau]}} Z_{[s,t]}^{3,i} \right\|_{L^{mk/(k'+1)}} &\leq \sum_{[s,t] \in \pi_{n,[0,\tau]}} \|\mathbb{E}_s[Z_{[s,t]}^{3,i}]\|_{L^{mk/(k'+1)}} + \left( \sum_{[s,t] \in \pi_{n,[0,\tau]}} \|Z_{[s,t]}^{3,i}\|_{L^{mk/(k'+1)}}^2 \right)^{1/2} \\ &\lesssim \sum_{[s,t] \in \pi_{n,[0,\tau]}} |\pi_n|^\epsilon |t-s|^\delta + \left( \sum_{[s,t] \in \pi_{n,[0,\tau]}} |\pi_n|^{2\epsilon} |t-s|^{2\alpha} \right)^{1/2} \\ &\lesssim |\pi_n|^\epsilon (\tau + \sqrt{\tau}), \end{aligned} \tag{36}$$

where we used Equation [\(35\)](#page-21-0) with i = 0 and

$$\begin{aligned} \|\mathbb{E}_s[Z_{[s,t]}^{3,i}]\|_{L^{mk/(k'+1)}} &= \left\| \left( S^{k'}(\mathbb{X}^{\pi_{n+1}})_{[0,s]} - S^{k'}(\mathbb{X}^{\pi_n})_{[0,s]} \right) \otimes \mathbb{E}_s[\mathbf{X}_s,t] \right\|_{L^{mk/(k'+1)}} \\ &\leq \left\| S^{k'}(\mathbb{X}^{\pi_{n+1}})_{[0,s]} - S^{k'}(\mathbb{X}^{\pi_n})_{[0,s]} \right\|_{L^{mk/k'}} \|\mathbb{E}_s[\mathbf{X}_s,t]\|_{L^{mk}} \lesssim |\pi_n|^\epsilon |t - s|^\delta, \end{aligned}$$

by applying Lemma [B.2,](#page-13-1) the inductive hypothesis [\(29\)](#page-19-0) and Assumption [\(A](#page-3-0)δ). When <sup>i</sup> ≥ <sup>1</sup>, we can instead directly apply the triangle inequality and Equation [\(35\)](#page-21-0) to bound

$$\left\| \sum_{[s,t] \in \pi_{n,[0,\tau]}} Z_{[s,t]}^{3,i} \right\|_{L^{mk/(k'+1)}} \leq \sum_{[s,t] \in \pi_{n,[0,\tau]}} \|Z_{[s,t]}^{3,i}\|_{L^{mk/(k'+1)}} \lesssim \sum_{[s,t] \in \pi_{n,[0,\tau]}} |\pi_n|^\epsilon |t-s|^{(1+i)\alpha} \lesssim |\pi_n|^\epsilon \tau. \quad (37)$$

Combining bounds [\(32\)](#page-20-1), [\(33\)](#page-21-1), [\(34\)](#page-21-2), [\(36\)](#page-21-3) and [\(37\)](#page-21-4) yields

$$\begin{aligned} & \left\| S^{k'+1}(\mathbb{X}^{\pi_{n+1}})_{[0,\tau]} - S^{k'+1}(\mathbb{X}^{\pi_n})_{[0,\tau]} \right\|_{L^{mk/(k'+1)}} \\ & \leq \sum_{i=0}^{k'} \frac{1}{(1+i)!} \left\{ \sum_{\substack{\mathcal{I} \in \{0,1\}^{1+i} \\ \mathcal{I} \neq (0,\dots,0), (1,\dots,1)}} \left\| \sum_{\substack{[s,t] \in \pi_n, [0,\tau] \\ u \in (s,t)}} Z_{[s,t]}^{1,\mathcal{I}} \right\|_{L^{mk/(k'+1)}} \right. \\ & \quad + \sum_{j=0}^{k'-i-1} \frac{1}{(1+j)!} \left\| \sum_{\substack{[s,t] \in \pi_n, [0,\tau] \\ u \in (s,t)}} Z_{[s,t]}^{2,i,j} \right\|_{L^{mk/(k'+1)}} \\ & \quad \left. + \left\| \sum_{[s,t] \in \pi_n, [0,\tau]} Z_{[s,t]}^{3,i} \right\|_{L^{mk/(k'+1)}} \right\} \\ & \lesssim |\pi_n|^\epsilon (\tau + \sqrt{\tau}) \lesssim |\pi_n|^\epsilon (T + \sqrt{T}), \end{aligned}$$

which proves [\(29\)](#page-19-0) with i = k ′ + 1, completing the inductive step.

Setting i = k, N = 1 and τ = T in [\(29\)](#page-19-0) yields

$$\|S^k(\mathbb{X}^{\pi_{n+1}})_{[0,T]} - S^k(\mathbb{X}^{\pi_n})_{[0,T]}\|_{L^m} \lesssim |\pi_n|^\epsilon, \quad n \geq 1,$$

and hence, assuming P n≥1 |πn| <sup>ϵ</sup> <sup>&</sup>lt; ∞, the sequence {<sup>S</sup> k (<sup>X</sup> <sup>π</sup><sup>n</sup> )[0,T] , n ≥} is Cauchy in <sup>L</sup> <sup>m</sup>. Since L <sup>m</sup> is a Banach space, the sequence converges in L <sup>m</sup> to S k (X)[0,T] ∈ <sup>L</sup> <sup>m</sup> (by uniqueness of limits) with rate

$$\|S^k(\mathbb{X}^{\pi_n})_{[0,T]} - S^k(\mathbb{X})_{[0,T]}\|_{L^m} \leq \sum_{n' \geq n} \|S^k(\mathbb{X}^{\pi_{n'+1}})_{[0,T]} - S^k(\mathbb{X}^{\pi_{n'}})_{[0,T]}\|_{L^m} \lesssim \sum_{n' \geq n} |\pi_n|^\epsilon.$$

*Remark* B.4*.* When {<sup>π</sup>n, n ≥ <sup>1</sup>} is a sequence of refining partitions with <sup>P</sup> n≥1 |πn| <sup>ϵ</sup> <sup>&</sup>lt; ∞, the proof actually yields the following (stronger) result

$$\sup_{\tau \in \pi_n} \|S^k(\mathbb{X}^{\pi_n})_{[0,\tau]} - S^k(\mathbb{X})_{[0,\tau]}\|_{L^m} \lesssim \sum_{n' \geq n} |\pi_{n'}|^\epsilon \rightarrow 0, \quad n \rightarrow \infty.$$

#### B.2. Proof of Theorem [2.10](#page-4-1)

*Sketch of proof. The proof of this result relies on decomposition* [\(8\)](#page-3-5)*. We can combine Theorem [2.8](#page-3-1) with assumption* [\(11\)](#page-4-6) *to show the first term in the decomposition vanishes in* L <sup>m</sup>*, for* m > 2*. To show the full consistency result it thus suffices to show the second term in* [\(8\)](#page-3-5) *also vanishes in* L *, which follows by Birkhoff's ergodic theorem under the stated assumptions. Similarly, for the asymptotic normality result, we combine Theorem [2.8](#page-3-1) with assumption* [\(13\)](#page-4-7) *to show the first term in the decomposition vanishes in* L <sup>m</sup> *when inflated by* √ N*. The asymptotic normality of the second term can then be obtained by a simple application of a central limit theorem (CLT) for dependent random variables.*

Under the assumption that {<sup>X</sup> <sup>n</sup>, n ≥ <sup>1</sup>} is stationary and <sup>X</sup> 1 satisfies the assumptions of Theorem [2.8](#page-3-1) with m > 2, we have that, for each <sup>n</sup> ≥ <sup>1</sup>,

$$S^{\mathbf{I}}(\mathbb{X}^{n, \pi_{N, n}})_{[0, T]} \xrightarrow{L^m} S^{\mathbf{I}}(\mathbb{X})_{[0, T]}, \quad N \rightarrow \infty,$$

with rate O( P <sup>N</sup>′≥<sup>N</sup> |<sup>π</sup>N′ ,n| ϵ ). And hence

$$\left\| \frac{1}{N} \sum_{n=1}^N (S^{\mathbf{I}}(\mathbb{X}^{n,\pi_{N,n}})_{[0,T]} - S^{\mathbf{I}}(\mathbb{X}^n)_{[0,T]}) \right\|_{L^m} \lesssim \max_{1 \leq n \leq N} \sum_{N' \geq N} |\pi_{N',n}|^\epsilon \leq \sum_{N' \geq N} |\Pi(N')|^\epsilon,$$

since |Π(N′ )| = max1≤n≤N′ |<sup>π</sup>N′ ,n|. Under assumption [\(11\)](#page-4-6), we have thus established the first term in decomposition [\(8\)](#page-3-5) vanishes in L <sup>m</sup> as <sup>N</sup> → ∞ and therefore can focus on showing the second term also vanishes. Note that, under the stronger

assumption [\(13\)](#page-4-7), a similar reasoning can be applied when "blowing up" decomposition [\(8\)](#page-3-5) by √ N, and hence it suffices to show the second term, when rescaled by √ N, converges to a Gaussian random variable to establish the asymptotic normality result.

We first prove the following somewhat technical result. In what follows we abuse notation slightly and write S I (·)[0,T] to denote S (·).

Proposition B.5. *Let* <sup>X</sup> <sup>=</sup> {<sup>X</sup>t, t ∈ [0, T]} *denote a canonical geometric stochastic process defined on the probability space* (Ω[0,T] = C([0, T]; <sup>R</sup> d ), B[0,T] , <sup>P</sup>[0,T]) *by* <sup>X</sup><sup>t</sup> <sup>=</sup> <sup>ω</sup>[0,T](t) *for* <sup>t</sup> ∈ [0, T] *and* <sup>ω</sup>[0,T] ∈ <sup>Ω</sup>[0,T] *. Then for any collection of words* <sup>I</sup> *there exists a measurable map* S I : C([0, T]; <sup>R</sup> d ) → <sup>R</sup> |I| *such that*

$$\mathcal{S}^{\mathbf{I}}(\omega_{[0,T]}) = \mathcal{S}^{\mathbf{I}}(\mathbb{X})_{[0,T]}, \quad \mathbb{P}_{[0,T]} - \text{a.s.}$$

*Proof.* By Remark [2.4](#page-2-4) every canonical geometric stochastic process has at least one signature-defining sequence of partitions over any [s, t] ⊆ [0, T] given by the sequence <sup>ρ</sup> in Definition [2.1.](#page-1-0) Moreover, by passing to a subsequence if necessary, this also guarantees the existence of a sequence of partitions ρ<sup>∗</sup> along which [\(5\)](#page-2-2) holds almost surely. Hence, there exists a Borel set Ω ′ [0,T] ∈ B[0,T] and a sequence of partitions <sup>ρ</sup> <sup>∗</sup> with vanishing mesh such that for all <sup>ω</sup>[0,T] ∈ <sup>Ω</sup> ′ [0,T] ,

$$S^{\mathbf{I}}\left(\omega_{[0,T]}^{\rho_*}\right)_{[0,T]} \rightarrow S^{\mathbf{I}}\left(\omega_{[0,T]}\right)_{[0,T]}, \quad |\rho_*| \rightarrow 0,$$

and <sup>P</sup>[0,T](Ω′ [0,T] ) = 1. For every partition ρ<sup>∗</sup> the map

$$\omega_{[0,T]} \in \Omega'_{[0,T]} \mapsto S^{\mathbf{I}}\left(\omega_{[0,T]}^{\rho_*}\right)_{[0,T]} \in \mathbb{R}^{|\mathbf{I}|},$$

is Ω ′ [0,T] ∩ B[0,T] -measurable (by measurability of the sums and products of coordinate maps appearing in the discretized signature) and hence also

$$\omega_{[0,T]} \in \Omega'_{[0,T]} \mapsto S^{\mathbf{I}}\left(\omega_{[0,T]}\right)_{[0,T]} \in \mathbb{R}^{|\mathbf{I}|},$$

is Ω ′ [0,T] ∩ B[0,T] -measurable (by measurability of the pointwise limit of measurable functions). We can hence extend S I : Ω′ [0,T] → <sup>R</sup> |I| to a measurable map on the whole of Ω[0,T] = C([0, T]; <sup>R</sup> d ).

#### B.2.1. PROOF OF THEOREM [2.10,](#page-4-1) CONSISTENCY

The consistency result then follows by a simple application of Birkhoff's ergodic theorem [\(Kallenberg,](#page-9-15) [2021,](#page-9-15) Theorem 25.6). Let (Ω = <sup>C</sup>([0, ∞); <sup>R</sup> d ), F <sup>=</sup> B[0,∞) , <sup>P</sup>) denote the canonical space on which {<sup>X</sup>t, t ≥ <sup>0</sup>} is defined, i.e. <sup>P</sup> is the law of {<sup>X</sup>t, t ≥ <sup>0</sup>}. Consider {<sup>X</sup> <sup>n</sup>, n ≥ <sup>1</sup>} as the sequence of <sup>C</sup>([0, T]; <sup>R</sup> d )-valued random variables on the space[<sup>12</sup>](#page-0-0) (C([0, T]; <sup>R</sup> d )<sup>∞</sup>, B<sup>∞</sup> [0,T] , <sup>P</sup><sup>∞</sup> [0,T] ), where the probability measure <sup>P</sup><sup>∞</sup> [0,T] is obtained by pushing forward P by the measurable mapping

$$\omega \in \Omega \mapsto \{\omega_{[0,T]}^n, n \geq 1\} \in C([0,T]; \mathbb{R}^d)^\infty,$$

where

$$\omega_{[0,T]}^n := \{\omega((n-1)T+t) - \omega((n-1)T), t \in [0,T]\}.$$

If {<sup>X</sup> <sup>n</sup>, n ≥ <sup>1</sup>} is stationary and <sup>X</sup> <sup>=</sup> <sup>X</sup> 1 is a canonical geometric stochastic process, then each X <sup>n</sup> is also a canonical geometric stochastic process on (C([0, T]; <sup>R</sup> d ), B[0,T] , <sup>P</sup>∗<sup>X</sup> <sup>n</sup>) with signature given by the same measurable mapping S I : C([0, T]; <sup>R</sup> d ) → <sup>R</sup> <sup>|</sup>I<sup>|</sup> given in Proposition [B.5.](#page-23-1) We can then apply Birkhoff's ergodic theorem [\(Kallenberg,](#page-9-15) [2021,](#page-9-15) Theorem 25.6) to conclude

$$\frac{1}{N} \sum_{n=1}^N S^I(\mathbb{X}^n)_{[0,T]} \xrightarrow{\mathbb{P}\text{-a.s.}} \mathbb{E}[S^I(\mathbb{X})_{[0,T]}], \quad N \rightarrow \infty.$$

$$\mathcal{B}(E^\infty) = \otimes_{n \geq 1} \mathcal{B}(E) := \sigma(\cup_{n \geq 1} (\mathcal{B}(E)^n \times E^\infty)) ,$$

<sup>12</sup>Recall that if (E, B(E)) is a Borel measurable space and we equip E <sup>∞</sup>, i.e. the space of sequences with values in E, with the product topology then

#### B.2.2. PROOF OF THEOREM [2.10,](#page-4-1) ASYMPTOTIC NORMALITY

We apply the dependent central limit theorem (CLT) given in [Ibragimov](#page-9-16) [\(1962,](#page-9-16) Theorem 1.7) and extended to the multivariate setting via the Cramer-Wold theorem. To apply this result we require ´ {<sup>S</sup> (<sup>X</sup> <sup>n</sup>)[0,T] , n ≥ <sup>1</sup>} to be stationary with S I (<sup>X</sup> <sup>n</sup>)[0,T] ∈ <sup>L</sup> 2+ζ and strongly mixing with strong mixing coefficient <sup>α</sup>(n), n ∈ <sup>N</sup> satisfying

$$\sum_{n \geq 1} \alpha(n)^{\zeta/(2+\zeta)} < \infty,$$

so that

$$\Sigma_{\mathbf{I}} = \text{Var} \left( S^{\mathbf{I}}(\mathbb{X}^1)_{[0,T]} \right) + 2 \sum_{n \geq 2} \text{Cov} \left( S^{\mathbf{I}}(\mathbb{X}^1)_{[0,T]}, S^{\mathbf{I}}(\mathbb{X}^n)_{[0,T]} \right) < \infty,$$

and, if Σ<sup>I</sup> is strictly positive definite,

$$\sqrt{N} \left( \frac{1}{N} \sum_{n=1}^N S^{\mathbf{I}}(\mathbb{X}^n)_{[0,T]} - \mathbb{E}[S^{\mathbf{I}}(\mathbb{X})_{[0,T]}] \right) \xrightarrow{\mathcal{L}} \mathcal{N}(0, \Sigma_{\mathbf{I}}), \quad N \rightarrow \infty.$$

Note that S I (<sup>X</sup> <sup>n</sup>)[0,T] ∈ <sup>L</sup> 2+ζ for ζ > 0 is immediately obtained by applying Theorem [2.8](#page-3-1) with m > 2. By measurability of S I : <sup>C</sup>([0, T]; <sup>R</sup>) → <sup>R</sup> (Proposition [B.5\)](#page-23-1), <sup>σ</sup>(<sup>S</sup> I (<sup>X</sup> <sup>n</sup>)) ⊆ <sup>σ</sup>(<sup>X</sup> <sup>n</sup>), for all <sup>n</sup> ≥ <sup>1</sup>, which implies {<sup>S</sup> I (<sup>X</sup> <sup>n</sup>)[0,T] , n ≥ <sup>1</sup>} is also strongly mixing with strong mixing coefficient at most <sup>α</sup>(n), n ∈ <sup>N</sup>. The assumptions of Theorem [2.10](#page-4-1) are thus sufficient to deduce asymptotic normality of the expected signature estimator.

### B.3. Proof of Corollary [2.12](#page-4-5)

*Sketch of proof. In order to show that the CLT result of Theorem [2.10](#page-4-1) can be made feasible we need to prove the kernel estimator* Σˆ Π(N) I *is consistent for the long-run covariance matrix* ΣI*. To do so, we first show that the kernelized estimator is consistent for* Σ<sup>I</sup> *if, for each* n ≥ 0*, the cross-covariance estimator term* Σˆ n,Π(N) I *is consistent for* Σ n *(with a "fast enough" convergence rate). To show consistency of each cross-covariance term, we introduce an auxiliary process* Y n *obtained by appropriately "stitching" the processes* X 1 , . . . , X 1+n *together. This choice of* Y n *, along with the shuffle property of the expected signature, implies that each term in* Σ n <sup>I</sup> *can be expressed as a combination of terms from the expected signature of* Y n *. The rest of the proof is thus devoted to showing that, under the assumptions of this Corollary, the process* Y n *satisfies the conditions of Theorem [2.10.](#page-4-1)1, ensuring we can consistently estimate its signature terms and, in turn, the entries of* Σ n

To make the CLT feasible one requires a consistent estimator for the long-run covariance of the sequence of random variables {S I (<sup>X</sup> <sup>n</sup>)[0,T] , n ≥ <sup>1</sup>},

$$\Sigma_{\mathbf{I}} = \Sigma_{\mathbf{I}}^0 + 2 \sum_{n \geq 1} \Sigma_{\mathbf{I}}^n, \quad \text{where} \quad \Sigma_{\mathbf{I}}^n = \text{Cov} \left( S^{\mathbf{I}}(\mathbb{X}^1)_{[0,T]}, S^{\mathbf{I}}(\mathbb{X}^{1+n})_{[0,T]} \right), \quad n \geq 0.$$

We consider the *non-overlapping* sample (cross-)covariances of the sequence {<sup>S</sup> I (<sup>X</sup> n,πN,n )[0,T] , n = 1, . . . , N}, i.e. for |n| ≤ <sup>N</sup> − <sup>1</sup>,

$$\hat{\Sigma}_{\mathbf{I}}^{n,\Pi(N)} = \frac{1}{[N/(n+1)]} \sum_{m=1}^{[N/(n+1)]} \left( S^{\mathbf{I}}(\mathbb{X}^{\pi_{N,(n+1)m-n}})_{[0,T]} - \hat{\phi}_{\mathbf{I}}^{\Pi(N)}(T) \right) \left( S^{\mathbf{I}}(\mathbb{X}^{\pi_{N,(n+1)m}})_{[0,T]} - \hat{\phi}_{\mathbf{I}}^{\Pi(N)}(T) \right)^T,$$

as estimators for Σ n I . Note that, for a fixed observation partition Π(N), we are only able to estimate Σ n I up to <sup>n</sup> <sup>=</sup> <sup>N</sup> − <sup>1</sup> with the quality of the estimator decreasing as n increases[<sup>13</sup>](#page-0-0). A natural choice would thus be to put less weight on Σˆ n,Π(N) I

$$\frac{1}{N-n} \sum_{m=1}^{N-n} \left( S^{\mathbf{I}}(\mathbb{X}^{\pi_{N,m}})_{[0,T]} - \hat{\phi}_{\mathbf{I}}^{\Pi(N)}(T) \right) \left( S^{\mathbf{I}}(\mathbb{X}^{\pi_{N,m+n}})_{[0,T]} - \hat{\phi}_{\mathbf{I}}^{\Pi(N)}(T) \right)^{\mathbf{T}},$$

<sup>13</sup>In this context, one might exploit the available data more efficiently by considering the full sample cross-covariances of the sequence {S I (<sup>X</sup> n,πN,n )[0,T ] , n = 1, . . . , N}, i.e. for |n| ≤ N − 1,

with n large than on Σˆ n,Π(N) <sup>I</sup> with small n. To do so, one can consider the kernel estimator

$$\hat{\Sigma}_{\mathbf{I}}^{\Pi(N)} = \sum_{n=-(N-1)}^{N-1} k\left(\frac{n}{h_N}\right) \hat{\Sigma}_{\mathbf{I}}^{n,\Pi(N)}, \quad \hat{\Sigma}_{\mathbf{I}}^{-n,\Pi(N)} := \left(\hat{\Sigma}_{\mathbf{I}}^{n,\Pi(N)}\right)^T, \text{ for } n = 1, \dots, N-1,$$

where <sup>k</sup> : <sup>R</sup> → [0, 1] is a decreasing kernel function continuous at zero with with <sup>k</sup>(0) = 1 and <sup>h</sup><sup>N</sup> is an appropriately chosen band-width parameter[<sup>14</sup>](#page-0-0). In what follows, and in the statement of Corollary [2.12,](#page-4-5) we set k to be the truncation kernel k(x) = <sup>1</sup>[−1,1](x) for simplicity, but other choices, such as the Bartlett kernel, might lead to better finite sample properties. For each I, J ∈ <sup>I</sup>, if Σˆ n,Π(N) I,J is consistent for Σ n I,J in L <sup>2</sup> with monotonically decreasing rate <sup>r</sup>(M) → <sup>0</sup> as the *effective* sample size <sup>M</sup> <sup>=</sup> ⌊N/(<sup>n</sup> + 1)⌋ → ∞, then

$$\begin{aligned} \|\hat{\Sigma}_{I,J}^{\Pi(N)} - \Sigma_{I,J}\|_{L^2} &\leq \sum_{|n|\leq h_N} \left\| \hat{\Sigma}_{I,J}^{n,\Pi(N)} - \Sigma_{I,J}^n \right\|_{L^2} + \left| \sum_{|n|>h_N} \Sigma_{I,J}^n \right| \\ &\leq \sum_{|n|\leq h_N} r(\lfloor N/(n+1) \rfloor) + \left| \sum_{|n|>h_N} \Sigma_{I,J}^n \right|. \end{aligned}$$

If the band-width <sup>h</sup><sup>N</sup> → ∞ as <sup>N</sup> → ∞, then <sup>Σ</sup><sup>I</sup> <sup>&</sup>lt; ∞ ensures the second term vanishes. If, moreover, we set the rate at which <sup>h</sup><sup>N</sup> → ∞ to be slow enough to ensure also the first term converges to zero, then consistency of the estimators Σˆ n,Π(N) I,J , for <sup>n</sup> ≥ <sup>0</sup>, is inherited by Σˆ Π(N) I,J . Under the assumption <sup>r</sup>(M) ∼ <sup>M</sup>−<sup>υ</sup> for <sup>υ</sup> ∈ (0, 1), one can set <sup>h</sup><sup>N</sup> <sup>=</sup> <sup>N</sup>υ/<sup>2</sup> . We can hence focus on determining under which conditions, other than those of Theorem [2.10](#page-4-1).2, the estimator Σˆ n,Π(N) I,J is consistent for Σ n I,J , for any <sup>n</sup> ≥ <sup>0</sup> and I, J ∈ <sup>I</sup>.

Note that, we can apply the shuffle identity to write, for I, J ∈ <sup>I</sup>,

$$\hat{\Sigma}_{I,J}^{0,\Pi(N)} = \sum_{K \in I \cup J} \hat{\phi}_K^{\Pi(N)}(T) - \hat{\phi}_I^{\Pi(N)}(T) \hat{\phi}_J^{\Pi(N)}(T),$$

and show consistency of this estimator for Σ I,J by applying the consistency result for the estimator of the expected signature terms to <sup>K</sup> ∈ <sup>I</sup> <sup>J</sup> for I, J ∈ <sup>I</sup>. We now attempt to apply a similar approach to the cross-covariance terms. To do so, we introduce an R nd-valued auxiliary process[<sup>15</sup>](#page-0-0) defined as <sup>Y</sup> <sup>n</sup> <sup>=</sup> {Y<sup>n</sup> t , t ∈ [0,(<sup>n</sup> + 1)T]}, where

$$\mathbf{Y}_t^n = \begin{pmatrix} \mathbf{X}_{t \wedge T}^1 \\ \mathbf{X}_{(t-T)^+ \wedge T}^2 \\ \vdots \\ \mathbf{X}_{(t-nT)^+ \wedge T}^{n+1} \end{pmatrix}, \quad t \in [0, (n+1)T] \iff \mathbf{Y}_t^n = \begin{pmatrix} \mathbf{X}_T^1 \\ \vdots \\ \mathbf{X}_T^i \\ \mathbf{X}_s^{i+1} \\ 0 \\ \vdots \\ 0 \end{pmatrix}, \quad t = iT + s, s \in [0, T], 0 \leq i \leq n.$$

By construction, we have that, for any two words I, J ∈ <sup>I</sup> over the letters {1, . . . , d},

$$S^I(\mathbb{X}^1)_{[0,T]} = S^I(\mathbb{Y}^n)_{[0,(n+1)T]} \quad \text{and} \quad S^J(\mathbb{X}^{n+1})_{[0,T]} = S^{nd+J}(\mathbb{Y}^n)_{[0,(n+1)T]}.$$

I.e. we have re-written the two signature components of X and X <sup>n</sup> over [0, T] as time-overlapping signature components of Y <sup>n</sup> over [0,(n + 1)T]. We can thus apply the shuffle product to deduce

$$S^I(\mathbb{X}^1)_{[0,T]}S^J(\mathbb{X}^n)_{[0,T]} = S^I(\mathbb{Y}^n)_{[0,(n+1)T]}S^{nd+J}(\mathbb{Y}^n)_{[0,(n+1)T]} = \sum_{K \in I \cup \{nd+J\}} S^K(\mathbb{Y}^n)_{[0,(n+1)T]}.$$

as estimators for Σ n <sup>I</sup> with n ≥ 0. The reason for using the non-overlapping sample covariance will become apparent once we introduce the process Y n , which will be used to show consistency of the estimator Σˆ n,Π(N) for Σ n <sup>I</sup> by re-applying Theorem [2.10](#page-4-1).1. To show consistency of the full sample covariance estimator we would instead require the generalization of such result for time-overlapping expected signature estimators.

<sup>14</sup>In the presence of conditional heteroskedasticity such estimators for the long-run covariance are known as Heteroskedasticity and Autocorrelation Consistent (HAC) estimators [\(Newey & West,](#page-10-11) [1987\)](#page-10-11). Note that decomposing the estimation of 2Σ<sup>n</sup> I into the sum of Σˆ n,Π(N) and its transpose ensures the resulting long-run covariance estimator Σˆ Π(N) I is symmetric.

I <sup>15</sup>We would like to thank Nicola Muca Cirone for suggesting this clever trick.

We can thus re-write the (I, J)-th entry of Σ n I as

$$\begin{aligned} \Sigma_{I,J}^n &= \mathbb{E} [S^I(\mathbb{X}^1)_{[0,T]} S^J(\mathbb{X}^{1+n})_{[0,T]}] - \mathbb{E} [S^I(\mathbb{X}^1)_{[0,T]}] \mathbb{E} [S^J(\mathbb{X}^{1+n})_{[0,T]}] \\ &= \mathbb{E} [S^I(\mathbb{Y}^n)_{[0,(n+1)T]} S^{nd+J}(\mathbb{Y}^n)_{[0,(n+1)T]}] - \mathbb{E} [S^I(\mathbb{X})_{[0,T]}] \mathbb{E} [S^J(\mathbb{X})_{[0,T]}] \\ &= \sum_{K \in I \cup \{nd+J\}} \mathbb{E} [S^K(\mathbb{Y}^n)_{[0,(n+1)T]}] - \mathbb{E} [S^I(\mathbb{X})_{[0,T]}] \mathbb{E} [S^J(\mathbb{X})_{[0,T]}] \\ &= \sum_{K \in I \cup \{nd+J\}} \psi_K((n+1)T) - \phi_I(T) \phi_J(T), \end{aligned}$$

and estimate it by

$$\hat{\Sigma}_{I,J}^{n,\Pi(N)} = \sum_{K \in I \cup W(nd+J)} \hat{\psi}_K^{\Pi(N;n)}((n+1)T) - \hat{\phi}_I^{\Pi(N)}(T) \hat{\phi}_J^{\Pi(N)}(T),$$

where, setting <sup>M</sup> <sup>=</sup> ⌊N/(<sup>n</sup> + 1)⌋,

$$\hat{\psi}_K^{\Pi(N;n)}((n+1)T) := \frac{1}{M} \sum_{m=1}^M S^K(\mathbb{Y}^{\pi_{N,m;n}})_{[0,(n+1)T]},$$

and

$$\Pi(N; n) = \pi_{N,1;n} \cup \dots \cup ((M-1)(n+1)T + \pi_{N,M;n}),$$

where for each m = 1, . . . , M,

$$\pi_{N,m;n} = \pi_{N,(m-1)(n+1)+1} \cup \dots \cup (nT + \pi_{N,m(n+1)}),$$

partitions [0,(n + 1)T] and

$$|\Pi(N; n)| := \max_{1 \leq m \leq M} |\pi_{N, m; n}| = \max_{1 \leq m \leq M} \max_{1 \leq i \leq (n+1)} |\pi_{N, (m-1)(n+1)+i}| \leq |\Pi(N)|.$$

In order to apply the consistency result of Theorem [2.10](#page-4-1) to ψˆΠ(N;n) <sup>K</sup> (nT), we need to understand under which additional conditions on {<sup>X</sup>t, t ≥ <sup>0</sup>}, other than those of Theorem [2.10,](#page-4-1) the process {Y<sup>n</sup> t , t ≥ <sup>0</sup>}, defined by

$$\mathbf{Y}_t^n = \sum_{m \geq 1} \begin{pmatrix} \mathbf{X}_{(t-(m-1)T)^+ \wedge T}^m \\ \mathbf{X}_{(t-mT)^+ \wedge T}^{m+1} \\ \dots \\ \mathbf{X}_{(t-(m+n-1)T)^+ \wedge T}^{m+n} \end{pmatrix}, \quad t \geq 0,$$

satisfies itself the assumptions of Theorem [2.10.](#page-4-1) Note that canonical geometricity of the stochastic process X (with liftdefining sequence of partitions <sup>ρ</sup>, |ρ| → <sup>0</sup>) and stationarity of {<sup>X</sup> <sup>m</sup>, m ≥ <sup>1</sup>} imply that each <sup>X</sup> <sup>m</sup> is a canonical geometric stochastic process and, hence, so is Y <sup>n</sup> (with lift-defining sequence of partitions ρ <sup>n</sup> <sup>=</sup> <sup>ρ</sup>∪(<sup>T</sup> <sup>+</sup>ρ)∪· · ·∪(nT <sup>+</sup>ρ), |ρ| → <sup>0</sup>). Moreover, stationarity and ergodicity of {<sup>X</sup> <sup>m</sup>, m ≥ <sup>1</sup>} imply stationarity and ergodicity of {(<sup>X</sup> <sup>m</sup>, . . . , <sup>X</sup> <sup>m</sup>+<sup>n</sup>), m ≥ <sup>1</sup>} and hence of the measurable transformation {<sup>Y</sup> n,m = f(<sup>X</sup> <sup>m</sup>, . . . , <sup>X</sup> <sup>m</sup>+<sup>n</sup>), m ≥ <sup>1</sup>}.

For fixed <sup>n</sup> ≥ <sup>0</sup> and <sup>m</sup> ≥ <sup>1</sup>, each <sup>π</sup>·,m;<sup>n</sup> <sup>=</sup> {<sup>π</sup>N,m;n, ⌊N/(<sup>n</sup> + 1)⌋ ≥ <sup>m</sup>} is a signature-defining sequence for <sup>Y</sup> <sup>n</sup> = <sup>Y</sup> n,1 over [0,(<sup>n</sup> + 1)T] since each <sup>π</sup>·,n <sup>=</sup> {<sup>π</sup>N,n, N ≥ <sup>n</sup>} is a signature-defining sequence for <sup>X</sup> over [0, T]. Moreover, by construction of Π(N; <sup>n</sup>), |Π(N; <sup>n</sup>)| ≤ |Π(N)| → <sup>0</sup> whenever |Π(N)| → <sup>0</sup> as <sup>N</sup> → ∞.

It thus remains to check that Y <sup>n</sup> = <sup>Y</sup> n,1 satisfies [\(A](#page-3-0)α) and [\(A](#page-3-0)δ) over [0,(<sup>n</sup> + 1)T] with <sup>α</sup> ≥ <sup>1</sup>/2, <sup>δ</sup> ≥ <sup>1</sup> and p > <sup>4</sup><sup>k</sup> with <sup>k</sup> = maxI∈<sup>I</sup> |I|. Clearly, as we are now considering the process over an arbitrarily long time span [0,(<sup>n</sup> + 1)T], we will require {<sup>X</sup>t, t ≥ <sup>0</sup>} to satisfy [\(A](#page-3-0)α) and [\(A](#page-3-0)δ) with <sup>α</sup> ≥ <sup>1</sup>/2, <sup>δ</sup> ≥ <sup>1</sup> and p > <sup>4</sup><sup>k</sup> not just over [0, T] but for any <sup>0</sup> ≤ <sup>s</sup> ≤ <sup>t</sup> with |<sup>t</sup> − <sup>s</sup>| ≤ <sup>T</sup>. This condition is automatically fulfilled when {<sup>X</sup>t, t ≥ <sup>0</sup>} is stationary (or, more generally, has jointly stationary increments) and satisfies [\(A](#page-3-0)α) and [\(A](#page-3-0)δ) over [0, T] with <sup>α</sup> ≥ <sup>1</sup>/2, <sup>δ</sup> ≥ <sup>1</sup> and p > <sup>4</sup><sup>k</sup> with <sup>k</sup> = maxI∈<sup>I</sup> |I|. It turns out that these conditions, only slightly stronger than those already required in Theorem [2.10](#page-4-1) for the asymptotic normality of ϕˆΠ(N) I (T), are sufficient to show that <sup>ψ</sup>ˆΠ(N;n) <sup>K</sup> ((<sup>n</sup> + 1)T) is consistent for <sup>ψ</sup>K((<sup>n</sup> + 1)T), and thus Σˆ n,Π(N) I,J is consistent for Σ n I,J for any <sup>n</sup> ≥ <sup>0</sup> and I, J ∈ <sup>I</sup>.

To show [\(A](#page-3-0)α) holds for Y <sup>n</sup> with the same <sup>α</sup> ≥ <sup>1</sup>/<sup>2</sup> as {<sup>X</sup>t, t ≥ <sup>0</sup>} note that for <sup>0</sup> ≤ <sup>s</sup> ≤ <sup>t</sup> ≤ (<sup>n</sup> + 1)T,

$$\begin{aligned} \|\mathbf{Y}^n_{s,t}\|_{L^p} &\leq \sum_{i=0}^n \|\mathbf{X}_{(s-iT)^+ \wedge T, (t-iT)^+ \wedge T}^{i+1}\|_{L^p} \\ &\leq \sum_{i=0}^n \|\mathbf{X}_{iT+(s-iT)^+ \wedge T, iT+(t-iT)^+ \wedge T}\|_{L^p} \\ &\lesssim \sum_{i=0}^n |(t-iT)^+ \wedge T - (s-iT)^+ \wedge T|^\alpha \lesssim |t-s|^\alpha. \end{aligned}$$

Next, to show that [\(A](#page-3-0)δ) holds for Y <sup>n</sup> with the same <sup>δ</sup> ≥ <sup>1</sup> as {<sup>X</sup>t, t ≥ <sup>0</sup>}. Note that for <sup>0</sup> ≤ <sup>s</sup> ≤ <sup>t</sup> ≤ (<sup>n</sup> + 1)T,

$$\begin{aligned}
\|\mathbb{E}_s[\mathbf{Y}_{s,t}^n]\|_{L^p} &\leq \sum_{i=0}^n \|\mathbb{E}_s[\mathbf{X}_{(s-iT)+\wedge T, (t-iT)+\wedge T}^{i+1}]\|_{L^p} \\
&\leq \sum_{i=0}^n \|\mathbb{E}_s[\mathbf{X}_{iT+(s-iT)+\wedge T, iT+(t-iT)+\wedge T}]\|_{L^p} \\
&\leq \sum_{i=0}^n \mathbf{1}_{[0, (i+1)T]}(s) \|\mathbb{E}_{iT+(s-iT)+\wedge T}[\mathbf{X}_{iT+(s-iT)+\wedge T, iT+(t-iT)+\wedge T}]\|_{L^p} \\
&\leq \sum_{i=0}^n |(t-iT)^+ \wedge T - (s-iT)^+ \wedge T|^\delta \lesssim |t-s|^\delta.
\end{aligned}$$

#### B.4. Proof of Proposition [2.13](#page-4-0)

#### B.4.1. PROOF OF PROPOSITION [2.13,](#page-4-0) STATIONARY IMPLICATIONS

The first implication follows directly from the definitions of stationarity and joint stationarity of the increments. It remains to show the latter implies stationarity of {<sup>X</sup> <sup>n</sup>, n ≥ <sup>1</sup>}, i.e. Equation [\(14\)](#page-4-8) implies Equation [\(6\)](#page-2-7). Under joint stationarity of the increments, for all <sup>k</sup> ∈ <sup>N</sup>, <sup>n</sup>1, . . . , n<sup>k</sup> ∈ <sup>N</sup> and <sup>n</sup> ≥ <sup>0</sup>,

$$\begin{aligned}\mathbb{P}(\mathbb{X}^{n_1} \in A_1, \dots, \mathbb{X}^{n_k} \in A_k) \\ &= \mathbb{P}(\mathbf{X}_{(n_1-1)T, (n_1-1)T+t_1^1} \in B_1^1, \dots, \mathbf{X}_{(n_k-1)T, (n_k-1)T+t_{m_k}^k} \in B_{m_k}^k), \\ &= \mathbb{P}(\mathbf{X}_{(n_1+n-1)T, (n_1+n-1)T+t_1^1} \in B_1^1, \dots, \mathbf{X}_{(n_k+n-1)T, (n_k+n-1)T+t_{m_k}^k} \in B_{m_k}^k), \\ &= \mathbb{P}(\mathbb{X}^{n_1+n} \in A_1, \dots, \mathbb{X}^{n_k+n} \in A_k),\end{aligned}$$

for all A1, . . . , A<sup>k</sup> cylinder sets of the form

$$A_j = \{\omega_{[0,T]} \in C([0,T]; \mathbb{R}^d) : \omega(t_1^j) \in B_1^j, \dots, \omega(t_{m_j}^j) \in B_{m_j}^j\},$$

for B j , . . . , B<sup>j</sup> m<sup>j</sup> ∈ B(<sup>R</sup> d ), t1, . . . , t<sup>j</sup> m<sup>j</sup> ∈ [0, T], <sup>m</sup><sup>j</sup> ≥ <sup>1</sup>. Noting that the collection of the sets <sup>A</sup><sup>1</sup> ×. . .×<sup>A</sup><sup>k</sup> is a semi-ring that generates[<sup>16</sup>](#page-0-0) the <sup>σ</sup>-algebra B k [0,T] , we can apply Caratheodory's extension theorem to conclude that [\(6\)](#page-2-7) holds.

$$\mathcal{B}_I = \sigma \left( \omega \in C(I; \mathbb{R}^d) : \omega(t_1) \in A_1, \dots, \omega(t_n) \in A_n, \ t_1, \dots, t_n \in I, \ A_1, \dots, A_n \in \mathcal{B}(\mathbb{R}^d), \ n \geq 1 \right).$$

Moreover, if (E, B(E)) is a Borel measurable space and we equip E <sup>k</sup> with the product topology, then

$$\mathcal{B}(E^k) = \mathcal{B}(E)^k := \sigma(A_1 \times \cdots \times A_k, A_1, \dots, A_k \in \mathcal{B}(E)) = \sigma(A_1 \times \cdots \times A_k, A_1, \dots, A_k \in \mathcal{G}),$$

<sup>16</sup>Recall that for a set I ⊆ <sup>R</sup>+, the Borel σ-algebra B<sup>I</sup> := B(C(I; <sup>R</sup> d )) (w.r.t. the topology induced by ∥ · ∥∞) can be equivalently defined by

*Remark* B.6*.* One might expect a similar statement as Proposition [2.13](#page-4-0) for ergodicity, but the following counterexample shows that

$$\{\mathbf{X}_t, t \geq 0\}$$
 is ergodic  $\Rightarrow \{\mathbb{X}^n, n \geq 1\}$  is ergodic.

Let {<sup>X</sup>t, t ≥ <sup>0</sup>} be an <sup>R</sup>-valued process such that

$$X_t = \sin\left(\frac{\pi t}{T} + \phi\right), \quad t \geq 0,$$

and <sup>ϕ</sup> ∼ <sup>U</sup>([0, <sup>2</sup>π]), inducing a probability measure <sup>P</sup><sup>∞</sup> [0,T] on (Ω<sup>∞</sup> [0,T] , B<sup>∞</sup> [0,T] ) where Ω[0,T] = (C([0, T], <sup>R</sup>). The process is stationary and ergodic. Stationarity of {<sup>X</sup>t, t ≥ <sup>0</sup>} implies stationarity of {<sup>X</sup> <sup>n</sup>, n ≥ <sup>1</sup>} by Proposition [2.13.](#page-4-0) But the shift-invariant set

$$I = \prod_{n \geq 1} \{\omega_{[0,T]}^n \in I_{\geq 0} \cup I_{\leq 0}\} \in \mathcal{B}_{[0,T]}^\infty,$$

where <sup>I</sup>≥0, I≤<sup>0</sup> ∈ B[0,T] are given by

$$\begin{aligned} I_{\geq 0} &:= \{\omega_{[0,T]} : \omega_{[0,T]}(t) \geq 0, \forall t \in [0,T]\}, \\ I_{\leq 0} &:= \{\omega_{[0,T]} : \omega_{[0,T]}(t) \leq 0, \forall t \in [0,T]\}, \end{aligned}$$

has measure <sup>P</sup><sup>∞</sup> [0,T] (I) = <sup>P</sup>(<sup>ϕ</sup> ∈ [π/2, π] ∪ [3π/2, <sup>2</sup>π]) = 1/<sup>2</sup> ∈ { / <sup>0</sup>, <sup>1</sup>} and hence {<sup>X</sup> <sup>n</sup>, n ≥ <sup>1</sup>} is not ergodic.

### B.4.2. PROOF OF PROPOSITION [2.13,](#page-4-0) STRONG MIXING IMPLICATIONS

We start by showing that strong mixing of {<sup>X</sup>t, t ≥ <sup>0</sup>} with strong mixing coefficient <sup>α</sup>(s), s ∈ <sup>R</sup><sup>+</sup> implies strong mixing of the progressive increment process {X<sup>T</sup> t , t ≥ <sup>0</sup>} where

$$\mathbf{X}_t^T := \mathbf{X}_{\lfloor t/T \rfloor T, t}, \quad t \geq 0,$$

with strong mixing coefficient α ′ (s) ≤ <sup>α</sup>(<sup>s</sup> − <sup>2</sup>T), s ≥ <sup>2</sup>T. This follows immediately from the definition of strong mixing and the fact that for <sup>t</sup> ≥ <sup>0</sup>,

$$\sigma(\mathbf{X}_u^T, u \leq t) \subseteq \sigma(\mathbf{X}_u, u \leq t),$$

and for <sup>s</sup> ≥ <sup>2</sup>T,

$$\sigma(\mathbf{X}_u^T, u \geq t + s) \subseteq \sigma(\mathbf{X}_u, u \geq \lfloor(t + s)/T\rfloor T) \subseteq \sigma(\mathbf{X}_u, u \geq t + (s - 2T)).$$

Next, we show that strong mixing of {X<sup>T</sup> t , t ≥ <sup>0</sup>} with strong mixing coefficient <sup>α</sup> ′ (s), s ∈ <sup>R</sup><sup>+</sup> implies strong mixing of {X <sup>n</sup>, n ≥ <sup>1</sup>} with strong mixing coefficient <sup>α</sup> ′′(n) ≤ <sup>α</sup> ′ ((<sup>n</sup> − 1)T), n ∈ <sup>N</sup>, and thus <sup>α</sup> ′′(n) ≤ <sup>α</sup>((<sup>n</sup> − 3)T), n ≥ <sup>3</sup>. Let

$$\mathcal{X}_a^b := \sigma(\mathbf{X}_u^T, u \in [a, b]),$$

for a, b ∈ <sup>R</sup><sup>+</sup> with <sup>a</sup> ≤ <sup>b</sup> and let

$$\mathcal{S}_m^n := \sigma(\mathbb{X}^l, m \leq l \leq n),$$

for m, n ∈ <sup>N</sup> with <sup>m</sup> ≤ <sup>n</sup>. Then, by definition, for each <sup>n</sup> ∈ <sup>N</sup>,

$$\mathcal{X}_{(n-1)T}^{nT} = \sigma(\mathbf{X}_{(n-1)T,(n-1)T+t}, t \in [0, T]) = \mathcal{S}_n^n.$$

Thus for any m, n ∈ <sup>N</sup> with <sup>m</sup> ≤ <sup>n</sup>

$$\mathcal{S}_m^n = \mathcal{X}_{(m-1)T}^{nT}.$$

Letting <sup>k</sup> ∈ <sup>N</sup> and <sup>A</sup> ∈ S<sup>k</sup> −∞, B ∈ S<sup>∞</sup> <sup>k</sup>+<sup>n</sup> we thus have

$$|\mathbb{P}(A \cap B) - \mathbb{P}(A)\mathbb{P}(B)| \leq \alpha'((n-1)T) \rightarrow 0, \quad k \rightarrow \infty.$$

#### B.5. Proof of Theorem [2.14](#page-4-2)

*Sketch of proof. As in the proof of Theorem [2.10,](#page-4-1) c.f. Appendix [B.2,](#page-22-0) we can apply the in-fill result given in Theorem [2.8](#page-3-1) to show the first term in decomposition* [\(8\)](#page-3-5) *vanishes. For the second term, we show the sequence of random variables* {S I (<sup>X</sup> n )[0,T ] , n ≥ 1} *satisfies a (weak) law of large numbers by verifying the auto-covariance decay condition* [\(38\)](#page-29-1)*. For each fixed lag* n ≥ 1*, we start by bounding the auto-covariance of lagged discretized signatures* {S I (<sup>X</sup> ρ,n)[0,T ] , n ≥ 1}*. This step crucially relies on the Gaussian assumption when using Isserlis' theorem to compute the expectation of the product of (arbitrarily many) path increments. The required auto-covariance decay condition is then obtained in the in-fill limit along a sequence of signature-defining partitions* ρ *by an application of Theorem [2.8.](#page-3-1)*

Note that for any <sup>n</sup> ∈ <sup>N</sup>, <sup>0</sup> ≤ <sup>s</sup><sup>i</sup> ≤ <sup>t</sup><sup>i</sup> for <sup>i</sup> = 1, . . . , n and <sup>t</sup> ≥ <sup>0</sup>

$$(\mathbf{X}_{t+s_1, t+t_1}, \dots, \mathbf{X}_{t+s_n, t+t_n}) \stackrel{\mathcal{L}}{=} (\mathbf{X}_{s_1, t_1}, \dots, \mathbf{X}_{s_n, t_n}),$$

since both vectors are normally distributed with means

$$\mathbb{E}[\mathbf{X}_{t+s_i, t+t_i}] = \mathbb{E}[\mathbf{X}_{s_i, t_i}] = 0,$$

for i = 1, . . . , n and covariances

$$\text{Cov}(\mathbf{X}_{t+s_i, t+t_i}, \mathbf{X}_{t+s_j, t+t_j}) = \text{Cov}(\mathbf{X}_{s_i, t_i}, \mathbf{X}_{s_j, t_j}) = C(|t_i - s_i|, |s_j - t_i|, |t_j - s_j|),$$

for i, j = 1, . . . , n. This implies {<sup>X</sup>t, t ≥ <sup>0</sup>} has jointly stationary increments and hence, by Proposition [2.13,](#page-4-0) {<sup>X</sup> <sup>n</sup>, n ≥ <sup>1</sup>} is stationary. By Proposition [B.5](#page-23-1) the sequence {<sup>S</sup> (<sup>X</sup> <sup>n</sup>), n ≥ <sup>1</sup>} is defined <sup>P</sup>-a.s. and is stationary.

Note that by Appendix [E.1,](#page-45-2) <sup>X</sup> satisfies [\(A](#page-3-0)α) for any <sup>p</sup> ≥ <sup>2</sup>. When <sup>α</sup> = 1/2, it also satisfies [\(A](#page-3-0)δ) with <sup>δ</sup> ≥ <sup>1</sup> for all <sup>p</sup> ≥ <sup>2</sup> by assumption. We can thus apply Theorem [2.8](#page-3-1) to obtain an L 2 in-fill asymptotic result along a sequence of dyadic refinements. By the discussion at the start of Appendix [B.2,](#page-22-0) we can thus focus on showing the weak law of large numbers holds for {S I (<sup>X</sup> <sup>n</sup>), n ≥ <sup>1</sup>}.

To apply the weak law of large number for dependent random variables, note that:

- 1. S (<sup>X</sup> <sup>n</sup>)[0,T] ∈ <sup>L</sup> , by the in-fill asymptotic result.
- 2. To show Cov S I (<sup>X</sup> <sup>n</sup>)[0,T] , S<sup>I</sup> (<sup>X</sup> <sup>m</sup>)[0,T] → <sup>0</sup> as |<sup>m</sup> − <sup>n</sup>| → ∞, by stationarity of {<sup>X</sup>n, n ≥ <sup>1</sup>}, it is sufficient to show that

$$\text{Cov} \left( S^{k'}(\mathbb{X}^1)_{[0,T]}, S^{k'}(\mathbb{X}^n)_{[0,T]} \right) \rightarrow 0, \quad n \rightarrow \infty, \quad (38)$$

for any <sup>1</sup> ≤ <sup>k</sup> ′ ≤ <sup>k</sup> = maxI∈<sup>I</sup> |I|.

To show [\(38\)](#page-29-1), we wish to find a bound on Cov(S k ′ (<sup>X</sup> 1 )[0,T] , S<sup>k</sup> ′ (<sup>X</sup> <sup>n</sup>)[0,T]) vanishing to zero as <sup>n</sup> → ∞. Note that, by the in-fill asymptotic result, along the signature-defining sequence of dyadic refinements, for all <sup>n</sup> ≥ <sup>1</sup> and <sup>1</sup> ≤ <sup>k</sup> ′ ≤ <sup>k</sup>,

$$\text{Cov} \left( S^{k'}(\mathbb{X}^{\rho,1})_{[0,T]}, S^{k'}(\mathbb{X}^{\rho,n})_{[0,T]} \right) \rightarrow \text{Cov} \left( S^{k'}(\mathbb{X}^1)_{[0,T]}, S^{k'}(\mathbb{X}^n)_{[0,T]} \right), \quad |\rho| \rightarrow 0. \quad (39)$$

To bound Cov(S k (<sup>X</sup> 1 )[0,T] , S<sup>k</sup> (<sup>X</sup> <sup>n</sup>)[0,T]), we can hence start by bounding the covariance between the discretized signatures Cov(S k (<sup>X</sup> ρ,1 )[0,T] , S<sup>k</sup> (<sup>X</sup> ρ,n)[0,T]) and then let |ρ| → <sup>0</sup>.

Let ρ be a dyadic partition of [0, T] and note the form of the discretized signature

$$S(\mathbb{X}^\rho)_{[0,T]} = \bigotimes_{[u,v] \in \rho} \exp_{\otimes} \mathbf{X}_{u,v},$$

implies

$$S^{k'}(\mathbb{X}^\rho)_{[0,T]} = \sum_{i_\rho \in \mathcal{M}_\rho^{k'}} \bigotimes_{[u,v] \in \rho} \frac{\mathbf{X}_{[u,v]}^{\otimes i_\rho[u,v]}}{i_\rho[u,v]!},$$

where

$$\mathcal{M}_\rho^{k'} := \{i_\rho := \{i_{[u,v]}\}_{[u,v] \in \rho} : 0 \leq i_{[u,v]} \leq k', [u,v] \in \rho, \sum_{[u,v] \in \rho} i_{[u,v]} = k'\},$$

is the set of multindices over ρ with sum of components k ′ . Note that we can rewrite this as

$$\begin{aligned} S^{k'}(\mathbb{X}^\rho)_{[0,T]} &= \sum_{j=1}^{k'} \sum_{\substack{i_1, \dots, i_j \geq 1 \\ i_1 + \dots + i_j = k'}} \sum_{[u_1, v_1] < \dots < [u_j, v_j] \in \rho} \bigotimes_{l=1}^j \frac{\mathbf{X}_{u_l, v_l}^{\otimes i_l}}{i_l!} \\ &= \sum_{j=1}^{k'} \sum_{\substack{i_1, \dots, i_j \geq 1 \\ i_1 + \dots + i_j = k'}} \sum_{[u_1, v_1] < \dots < [u_j, v_j] \in \rho} \left[ \prod_{l=1}^j \frac{1}{i_l!} \right] \bigotimes_{x=1}^{k'} \mathbf{X}_{u_{l_x}, v_{l_x}}, \end{aligned}$$

where we first split the sum over the number of non-zero i[u,v] for each <sup>i</sup><sup>ρ</sup> ∈ M<sup>k</sup> ′ ρ and then rewrite the tensor product by introducing (l1, · · · , lk′ ) = (1, . . . , <sup>1</sup>, <sup>2</sup>, . . . , <sup>2</sup>, . . . , j, . . . , j), where the index <sup>1</sup> is repeated <sup>i</sup><sup>1</sup> times, the index <sup>2</sup> is repeated i<sup>2</sup> times, and so on. We introduce the notation [u1, v1] <m′ [u2, v2] denoting at least m′ intervals between [u1, v1] and [u2, v2] in ρ. When m′ = 0, we equivalently write < or <<sup>0</sup> to denote the interval [u2, v2] being after [u1, v1] in ρ.

We then group the above summations over the intervals [u1, v1], . . . , [u<sup>j</sup> , v<sup>j</sup> ] over all possible combinations of time intervals with at least one pair less than <sup>m</sup> steps away in the partition <sup>ρ</sup>. To do so, introduce the set[<sup>17</sup>](#page-0-0) variable I ⊕<sup>0</sup> J ∈ S<sup>j</sup> ,j,m where I ∈ {1} × {0, <sup>1</sup>} j−1 is such that |I| <sup>=</sup> I<sup>1</sup> <sup>+</sup> . . . <sup>+</sup> I<sup>j</sup> <sup>=</sup> <sup>j</sup> ′ and J ∈ {1, . . . , m} j−j where m is such that [\(A](#page-4-2)θ) holds. We can then recursively define, for l = 1, . . . , j,

$$[u_l, v_l](\mathcal{I} \oplus_0 \mathcal{J}) = \begin{cases} \text{interval } \mathcal{J} \text{ steps after } [u_{l-1}, v_{l-1}](\mathcal{I} \oplus_0 \mathcal{J}) & \text{if } \mathcal{I}_l = 0, \\ [u|_{\mathcal{I}_{1:l}}], v|_{\mathcal{I}_{1:l}}] & \text{if } \mathcal{I}_l = 1, \end{cases}$$

where |I1:<sup>l</sup> | <sup>=</sup> I<sup>1</sup> <sup>+</sup> . . . <sup>+</sup> I<sup>l</sup> , and write

$$S^{k'}(\mathbb{X}^\rho)_{[0,T]} = \sum_{j=1}^{k'} \sum_{\substack{i_1, \dots, i_j \geq 1 \\ i_1 + \dots + i_j = k'}} \sum_{j'=1}^j \sum_{\mathcal{I} \oplus 0 \in \mathcal{J} \in \mathcal{S}_{j', j, m}} \sum_{[u_1, v_1] <_{m_1} \dots <_{m_{j'-1}} [u_{j'}, v_{j'}] \in \rho} \left[ \prod_{l=1}^j \frac{1}{i_l!} \right] \bigotimes_{x=1}^{k'} \mathbf{x}_{[u_{l_x}, v_{l_x}](\mathcal{I} \oplus 0 \mathcal{J})},$$

where, for each l ′ = 1, . . . , j′ − <sup>1</sup>, we have [u<sup>l</sup> ′ , v<sup>l</sup> ′ ] <<sup>m</sup><sup>l</sup> [u<sup>l</sup> ′+1, v<sup>l</sup> ′+1], i.e. there are at least m<sup>l</sup> ′ intervals between [u<sup>l</sup> ′ , v<sup>l</sup> ′ ] and [u<sup>l</sup> ′+1, v<sup>l</sup> ′+1] in ρ, where

$$m_{l'} = m_{l'}(\mathcal{I} \oplus_0 \mathcal{J}) = \sum_{l=1}^j \mathcal{J}_l \mathbb{1}_{\{|\mathcal{I}_{1:l}|=l', \mathcal{I}_l=0\}}.$$

The structure of the tensor products in the summation is thus

$$\mathbf{X}_{u_1, v_1}^{\otimes i_1} \otimes \mathbf{X}_{u_1, v_1}^{\otimes i_2} \otimes \cdots \otimes \mathbf{X}_{u_1, v_1}^{\otimes i_{n_1}} \otimes \mathbf{X}_{u_2, v_2}^{\otimes i_{n_1}+1} \otimes \mathbf{X}_{u_2, v_2}^{\otimes i_{n_1}+2} \otimes \cdots \otimes \mathbf{X}_{u_2, v_2}^{\otimes i_{n_1}+n_2} \otimes \cdots,$$

where [u1, v1], [u 2 1 , v<sup>2</sup> 1 ], · · · , [<sup>u</sup> n<sup>1</sup> 1 , v n<sup>1</sup> 1 ] are all less than m intervals apart and [u2, v2] is at least m intervals after [u n<sup>1</sup> 1 , v n<sup>1</sup> 1 (and so on)[<sup>18</sup>](#page-0-0) .

Let <sup>n</sup> ≥ <sup>3</sup> and define <sup>ρ</sup><sup>n</sup> = (<sup>n</sup> − 1)<sup>T</sup> <sup>+</sup> <sup>ρ</sup>. Then, for each word <sup>I</sup> = (w1, . . . , wk′ ) of length <sup>k</sup> ′ , we can write

$$\text{Cov} \left( S^I(\mathbb{X}^{\rho,1})_{[0,T]}, S^I(\mathbb{X}^{\rho,n})_{[0,T]} \right)$$

$$S^2(\mathbb{X}^\rho)_{[0,T]} = \sum_{[u_1,v_1] \in \rho} \frac{\mathbf{X}_{u_1,v_1}^{\otimes 2}}{2!} + \sum_{[u_1,v_1] \in \rho} \mathbf{X}_{u_1,v_1} \otimes \mathbf{X}_{v_1,w_1} + \sum_{[u_1,v_1] \in \rho} \sum_{[u_2,v_2] \in \rho} \mathbf{X}_{u_1,v_1} \otimes \mathbf{X}_{u_2,v_2},$$

<sup>17</sup>Here I ⊕<sup>0</sup> J denotes pairing elements of J to the elements of I equal to 0.

<sup>18</sup>To help intuitive understanding consider the case where k ′ = 2 and m = 1, then the above expression reduces to

$$\begin{aligned} &= \sum_{j_1, j_2=1}^{k'} \sum_{\substack{i_1, \dots, i_{j_1} \geq 1 \\ i_1 + \dots + i_{j_1} = k'}} \sum_{\substack{e_1, \dots, e_{j_2} \geq 1 \\ e_1 + \dots + e_{j_2} = k'}} \sum_{j'_1=1}^{j_1} \sum_{j'_2=1}^{j_2} \sum_{\mathcal{I}_1 \oplus 0 \mathcal{J}_1 \in S_{j'_1, j_1, m}} \sum_{\mathcal{I}_2 \oplus 0 \mathcal{J}_2 \in S_{j'_2, j_2, m}} \\ &\quad \sum_{[u_1, v_1] < m_1^1 \cdots < m_1^{j_1'} - 1} \sum_{[u_{j'_1}, v_{j'_1}] \in \rho} \left[ \prod_{l=1}^{j_1} \frac{1}{i_l!} \right] \left[ \prod_{l=1}^{j_2} \frac{1}{e_l!} \right] \text{Cov} \left( \prod_{x=1}^{k'} X_{[u_{l_x}^1, v_{l_x}^1]}(\mathcal{I}_1 \oplus 0 \mathcal{J}_1), \prod_{x=1}^{k'} X_{[s_{l_x}^2, t_{l_x}^2]}(\mathcal{I}_2 \oplus 0 \mathcal{J}_2) \right) \\ &= \sum_{j_1, j_2=1}^{k'} \sum_{\substack{i_1, \dots, i_{j_1} \geq 1 \\ i_1 + \dots + i_{j_1} = k'}} \sum_{\substack{e_1, \dots, e_{j_2} \geq 1 \\ e_1 + \dots + e_{j_2} = k'}} \sum_{j'_1=1}^{j_1} \sum_{j'_2=1}^{j_2} \sum_{\mathcal{I}_1 \oplus 0 \mathcal{J}_1 \in S_{j'_1, j_1, m}} \sum_{\mathcal{I}_2 \oplus 0 \mathcal{J}_2 \in S_{j'_2, j_2, m}} \sum_{[u_1, v_1] < m_1^1 \cdots < m_1^{j_1'} - 1} \sum_{[u_{j'_1}, v_{j'_1}] \in \rho} \left[ \prod_{l=1}^{j_1} \frac{1}{i_l!} \right] \left[ \prod_{l=1}^{j_2} \frac{1}{e_l!} \right] \\ &\quad \sum_{p \in MP_{(2, k')}^2} \prod_{\{(\delta_1, x_1), (\delta_2, x_2)\} \in p} \text{Cov} \left( X_{[u_{l_x^1}^1, v_{l_x^1}^1]}(\mathcal{I}_1 \oplus 0 \mathcal{J}_1) X_{[s_{l_x^2}^1, t_{l_x^2}^1]}(\mathcal{I}_2 \oplus 0 \mathcal{J}_2), X_{[u_{l_x^2}^1, v_{l_x^2}^1]}(\mathcal{I}_1 \oplus 0 \mathcal{J}_1) X_{[s_{l_x^2}^2, t_{l_x^2}^2]}(\mathcal{I}_2 \oplus 0 \mathcal{J}_2) \right), \end{aligned}$$

by using the fact that, for two collections of mean-zero normal random variables (Z0,1, . . . , Z0,k′ ) and (Z1,1, . . . , Z1,k′ ), we can apply Isserlis' theorem [\(Isserlis,](#page-9-17) [1918\)](#page-9-17) to show

$$\begin{aligned} & \text{Cov} (Z_{0,1} \cdots Z_{0,k'}, Z_{1,1} \cdots Z_{1,k'}) \\ &= \mathbb{E} [Z_{0,1} \cdots Z_{0,k'} Z_{1,1} \cdots Z_{1,k'}] - \mathbb{E} [Z_{0,1} \cdots Z_{0,k'}] \mathbb{E} [Z_{1,1} \cdots Z_{1,k'}] \\ &= \sum_{p \in P_{(2,k')}^2} \prod_{\{(i_1, i_2), (j_1, j_2)\} \in p} \mathbb{E} [Z_{i_1, i_2} Z_{j_1, j_2}] - \left( \sum_{q \in P_{k'}^2} \prod_{\{i, j\} \in q} \mathbb{E} [Z_{0,i} Z_{0,j}] \right) \left( \sum_{r \in P_{k'}^2} \prod_{\{i, j\} \in r} \mathbb{E} [Z_{1,i} Z_{1,j}] \right) \\ &= \sum_{p \in MP_{(2,k')}^2} \prod_{\{(i_1, i_2), (j_1, j_2)\} \in p} \mathbb{E} [Z_{i_1, i_2} Z_{j_1, j_2}] \\ &= \sum_{p \in MP_{(2,k')}^2} \prod_{\{(i_1, i_2), (j_1, j_2)\} \in p} \text{Cov} (Z_{i_1, i_2}, Z_{j_1, j_2}), \end{aligned}$$

where P 2 (2,k′) denotes the set of all the pairings of {0, <sup>1</sup>} × {1, . . . , k′}, <sup>P</sup> 2 <sup>k</sup>′ denotes the set of all the pairings of {1, . . . , k′} and MP<sup>2</sup> (2,k′) denotes the set of all the pairings of {0, <sup>1</sup>} × {1, . . . , k′} that contain at least one "mixed" pair, i.e. for all <sup>p</sup> ∈ MP<sup>2</sup> (2,k′) there exist {(i1, i2),(j1, j2)} ∈ <sup>p</sup> such that <sup>i</sup><sup>1</sup> ̸<sup>=</sup> <sup>j</sup>1.

Note that each [u<sup>l</sup> ′ , v<sup>l</sup> ′ ] for l ′ = 1, . . . , j′ 1 appears in at least one covariance term. We proceed by cases:

- If [u<sup>l</sup> ′ , v<sup>l</sup> ′ ] appears in a pair with an interval [u∗, v∗] such that [u<sup>l</sup> ′−1, v<sup>l</sup> ′−1] < [u∗, v∗] < [u<sup>l</sup> ′+1, v<sup>l</sup> ′+1], then

$$|\text{Cov}(X_{[u_{l'}, v_{l'}]}^{(w)}, X_{[u_*, v_*]}^{(q)})| \lesssim |v_{l'} - u_{l'}|,$$

by applying Assumption [\(A](#page-3-0)α) with <sup>α</sup> ≥ <sup>1</sup>/<sup>2</sup> and the fact that <sup>ρ</sup> is dyadic (and hence uniform).

• If [u<sup>l</sup> ′ , v<sup>l</sup> ′ ] appears in a pair with an interval [u∗, v∗] such that [u∗, v∗] ≤ [u<sup>l</sup> ′−1, v<sup>l</sup> ′−1], then we have |<sup>u</sup><sup>l</sup> ′ − <sup>v</sup>∗| ≥ where [v1, w1] is the interval right-contiguous to [u1, v1] in ρ. If k ′ = 2 and m = 2, instead

$$S^2(\mathbb{X}^\rho)_{[0,T]} = \sum_{[u_1,v_1] \in \rho} \frac{\mathbf{X}_{u_1,v_1}^{\otimes 2}}{2!} + \sum_{[u_1,v_1] \in \rho} \mathbf{X}_{u_1,v_1} \otimes \mathbf{X}_{v_1,w_1} + \sum_{[u_1,v_1] \in \rho} \mathbf{X}_{u_1,v_1} \otimes \mathbf{X}_{w_1,z_1} + \sum_{[u_1,v_1] \in \rho} \sum_{[u_1,v_1] \in \rho} \mathbf{X}_{u_1,v_1} \otimes \mathbf{X}_{u_2,v_2},$$

m/2(|<sup>v</sup><sup>∗</sup> − <sup>u</sup>∗| <sup>+</sup> |<sup>v</sup><sup>l</sup> ′ − <sup>u</sup><sup>l</sup> ′ |) and

$$|\text{Cov}(X_{[u_{l'}, v_{l'}]}^{(w)}, X_{[u_*, v_*]}^{(q)})| \lesssim \theta(|u_{l'} - v_*|)|v_* - u_*||v_{l'} - u_{l'}| \lesssim \theta(|u_{l'} - v_{l'-1}|)|v_* - u_*||v_{l'} - u_{l'}|,$$

by applying [\(A](#page-4-2)θ).

- Similarly, if [u<sup>l</sup> ′ , v<sup>l</sup> ′ ] appears in a pair with an interval [u∗, v∗] such that [u<sup>l</sup> ′+1, v<sup>l</sup> ′+1] ≤ [u∗, v∗], then we have |<sup>u</sup><sup>∗</sup> − <sup>v</sup><sup>l</sup> ′ | ≥ m/2(|<sup>v</sup><sup>l</sup> ′ − <sup>u</sup><sup>l</sup> ′ | <sup>+</sup> |<sup>v</sup><sup>∗</sup> − <sup>u</sup>∗|) and

$$|\text{Cov}(X_{[u_{l'}, v_{l'}]}^{(w)}, X_{[u_*, v_*]}^{(q)})| \lesssim \theta(|u_* - v_{l'}|)|v_{l'} - u_{l'}||v_* - u_*| \lesssim \theta(|u_{l'+1} - v_{l'}|)|v_{l'} - u_{l'}||v_* - u_*|.$$

- If [u<sup>l</sup> ′ , v<sup>l</sup> ′ ] appears in a pair with an interval [s∗, t∗] ∈ <sup>ρ</sup>n, then

$$|\text{Cov}(X_{[u_{l'}, v_{l'}]}^{(w)}, X_{[s_*, t_*]}^{(q)})| \lesssim \theta((n-2)T)|v_{l'} - u_{l'}| |t_* - s_*|,$$

by applying [\(A](#page-4-2)θ).

A similar reasoning applies to each [s<sup>l</sup> ′ , t<sup>l</sup> ′ ], for l ′ = 1, . . . , j′ 2 . Noting that at least one pairing is mixed across ρ and ρ<sup>n</sup> we can hence bound

$$\begin{aligned} & |\text{Cov}\left(S^I(\mathbb{X}^{\rho,1})_{[0,T]}, S^I(\mathbb{X}^{\rho,n})_{[0,T]}\right)| \\ & \lesssim \sum_{j_1, j_2=1}^{k'} \sum_{\substack{i_1, \dots, i_{j_1} \geq 1 \\ i_1 + \dots + i_{j_1} = k'}} \sum_{\substack{e_1, \dots, e_{j_2} \geq 1 \\ e_1 + \dots + e_{j_2} = k'}} \sum_{j_1'=1}^{j_1} \sum_{j_2'=1}^{j_2} \sum_{\substack{\mathcal{I}_1 \oplus 0 \\ \mathcal{I}_2 \oplus 0}} \mathcal{J}_1 \in S_{j_1', j_1, m} [u_1, v_1] <_{m_1^{-1}} \dots <_{m_1^{-1}} \sum_{[u_{j_1'}, v_{j_1'}] \in \rho} \left[ \prod_{l=1}^{j_1} \frac{1}{i_l!} \right] \left[ \prod_{l=1}^{j_2} \frac{1}{e_l!} \right] \\ & \quad \sum_{p \in MP_{(2,k')}^2} \theta((n-2)T) |v_1 - u_1| \theta(|u_2 - v_1|) |v_2 - u_2| \dots \theta(|u_{j_1'} - v_{j_1'-1}|) |v_{j_1'} - u_{j_1'}| \\ & \quad \times |t_1 - s_1| \theta(|s_2 - t_1|) |t_2 - s_2| \dots \theta(|s_{j_2'} - t_{j_2'-1}|) |t_{j_2'} - s_{j_2'}| \\ & \lesssim \sum_{j_1, j_2=1}^{k'} \sum_{\substack{i_1, \dots, i_{j_1} \geq 1 \\ i_1 + \dots + i_{j_1} = k'}} \sum_{\substack{e_1, \dots, e_{j_2} \geq 1 \\ e_1 + \dots + e_{j_2} = k'}} \sum_{j_1'=1}^{j_1} \sum_{j_2'=1}^{j_2} \theta((n-2)T) \\ & \quad \times \left( \sum_{[u_1, v_1] \in \rho} |v_1 - u_1| \right) \left( \sum_{[u_2, v_2] \in \rho} \theta(|u_2|) |v_2 - u_2| \right) \dots \left( \sum_{[u_{j_1'}, v_{j_1'}] \in \rho} \theta(|u_{j_1'}|) |v_{j_1'} - u_{j_1'}| \right) \\ & \quad \times \left( \sum_{[s_1, t_1] \in \rho_n} |t_1 - s_1| \right) \left( \sum_{[s_2, t_2] \in \rho_n} \theta(|s_2|) t_2 - s_2 \right) \dots \left( \sum_{[s_{j_2'}, t_{j_2'}] \in \rho_n} \theta(|s_{j_2'}|) |t_{j_2'} - s_{j_2'}| \right) \\ & \rightarrow \theta((n-2)T) \left( \sum_{j_1, j_2=1}^{k'} \sum_{\substack{i_1, \dots, i_{j_1} \geq 1 \\ i_1 + \dots + i_{j_1} = k'}} \sum_{\substack{e_1, \dots, e_{j_2} \geq 1 \\ e_1 + \dots + e_{j_2} = k'}} \sum_{j_1'=1}^{j_1} \sum_{j_2'=1}^{j_2} T^2 \left( \int_0^T \theta(t) dt \right)^{j_1' + j_2' - 2} \right), \quad |\rho| \rightarrow 0. \end{aligned}$$

Combining this bound with [\(39\)](#page-29-2), we can conclude that, for all <sup>n</sup> ≥ <sup>3</sup>,

$$|\text{Cov} (S^I(\mathbb{X}^1)_{[0,T]}, S^I(\mathbb{X}^n)_{[0,T]})| \lesssim \theta((n-2)T) \rightarrow 0, \quad n \rightarrow \infty,$$

i.e. we have shown [\(38\)](#page-29-1). Hence, we can apply the weak law of large numbers of dependent random variables and the in-fill asymptotics to obtain the desired consistency result.

## C. Variance Reduction via Martingale Correction

In this Section [2.2](#page-5-2) we considered a single control obtained by substituting the outermost Stratonovich integral with an Ito integral. In principle, for a word of length <sup>ˆ</sup> |I| <sup>=</sup> <sup>k</sup> ≥ <sup>2</sup>, one could consider <sup>2</sup> <sup>k</sup>−<sup>2</sup> distinct controls: for any subset I ⊆ {2, . . . , k − <sup>1</sup>}, one can obtain a control by changing each of the integrals with index in I ∪ {k} to Ito integrals. One <sup>ˆ</sup> can then apply the controlled linear regression estimator (with only the intercept term as regressor) described in Appendix [G.](#page-53-0) This family of controls will likely be highly correlated and hence:

- the improvements provided by each additional control might be quite marginal compared to the considerable increase in computational cost;
- the estimator of the (inverse) variance matrix of the controls, needed to make the estimator feasible, might be quite unstable.

Hence, for clarity of exposition and computational ease, throughout the rest of this work we only consider the control variate estimator introduced in Section [2.2,](#page-5-2) i.e. for a fixed word I

$$\hat{\phi}_I^{N,\pi,c}(T) := \frac{1}{N} \sum_{n=1}^N \left( S^I(\mathbb{X}^{n,\pi})_{[0,T]} - cS_c^I(\mathbb{X}^{n,\pi})_{[0,T]} \right), \quad (40)$$

where I−<sup>1</sup> := (i1, . . . , ik−1) and

$$S_c^I(\mathbb{X}^\pi)_{[0,T]} := \sum_{[u,v] \in \pi} S^{I-1}(\mathbb{X}^\pi)_{[0,u]} X_{u,v}^{(i_k)}.$$

#### C.1. Martingale Continuity Criterion

If X is a martingale, then, by the Burkholder-Davis-Gundy (BDG) inequality [\(Burkholder et al.,](#page-9-14) [1972\)](#page-9-14), we can write a stronger version of assumptions [\(A](#page-3-0)α) in terms of the quadratic variation of <sup>X</sup>, for all <sup>0</sup> ≤ <sup>s</sup> ≤ <sup>t</sup> ≤ <sup>T</sup>

$$(\text{A}\alpha.\text{M}) \quad \|\langle \mathbf{X} \rangle_{s,t}\|_{L^{p/2}} \lesssim |t - s|^{2\alpha}.$$

Note that for many martingales assumption (Aα[.M\)](#page-33-1) holds with α = 1/2 and hence, since [\(A](#page-3-0)δ) holds trivially, we can usually apply Theorem [2.8](#page-3-1) under *(ii)*. Some non-trivial degenerate cases exist: For example, consider a one-dimensional mean-zero Gaussian martingale over [0, 1] with covariance function <sup>C</sup>(<sup>s</sup> ∧ <sup>t</sup>) where <sup>C</sup> is the Cantor function. This process has quadratic variation ⟨X⟩<sup>t</sup> <sup>=</sup> <sup>C</sup>(t) and hence – since the Cantor function is Holder continuous with H ¨ older exponent ¨ log<sup>3</sup> (2) – assumption (Aα[.M\)](#page-33-1) is satisfied with α = 1 2 log<sup>3</sup> <sup>2</sup> ∈ (1/4, <sup>1</sup>/3]. In this case, one can easily check that [\(A](#page-3-0)β) and [\(A](#page-3-0)γ) hold by combining the independent increments and martingale property of X.

#### C.2. Estimating c ∗ π

In Section [2.2,](#page-5-2) we considered the following estimator for c ∗ π :

$$\hat{c}_{\pi,1}^* = \frac{\sum_{n=1}^N S_c^I(\mathbb{X}^n, \pi)_{[0,T]} S_c^I(\mathbb{X}^n, \pi)_{[0,T]}}{\sum_{n=1}^N S_c^I(\mathbb{X}^n, \pi)_{[0,T]}^2}.$$

Alternatively, we can exploit the explicit form of the covariance and variance of the infeasible estimator and approximate

$$\begin{aligned} \text{Cov}(S^I(\mathbb{X}^\pi)_{[0,T]}, S_c^I(\mathbb{X}^\pi)_{[0,T]}) &\approx \text{Cov}(S^I(\mathbb{X})_{[0,T]}, S_c^I(\mathbb{X})_{[0,T]}) \\ &= \sum_{J \in I \cup I} \psi_J(T) - \frac{1}{2} \sum_{J \in I \cup I - 2 * ((i_{k-1}, i_k))} \psi_J(T) \\ &\approx \sum_{J \in I \cup I} \hat{\psi}_J^N(T) - \frac{1}{2} \sum_{J \in I \cup I - 2 * ((i_{k-1}, i_k))} \hat{\psi}_J^N(T) \\ &\approx \sum_{J \in I \cup I} \hat{\psi}_J^{N, \pi}(T) - \frac{1}{2} \sum_{J \in I \cup I - 2 * ((i_{k-1}, i_k))} \hat{\psi}_J^{N, \pi}(T) \\ &\approx \sum_{J \in I \cup I} \hat{\psi}_J^{N, \pi, \prime}(T) - \frac{1}{2} \sum_{J \in I \cup I - 2 * ((i_{k-1}, i_k))} \hat{\psi}_J^{N, \pi, \prime}(T), \end{aligned}$$

where

$$\hat{\psi}_J^N(T) := \frac{1}{N} \sum_{n=1}^N S^J((\mathbb{X}, \langle \mathbb{X} \rangle)^n)_{[0,T]}, \quad \hat{\psi}_J^{N,\pi}(T) := \frac{1}{N} \sum_{n=1}^N S^J((\mathbb{X}, \langle \mathbb{X} \rangle)^{n,\pi})_{[0,T]},$$

and

$$\hat{\psi}_J^{N,\pi,'}(T) := \frac{1}{N} \sum_{n=1}^N S^J((\mathbb{X}^{n,\pi}, \langle \hat{\mathbb{X}} \rangle^{n,\pi}))_{[0,T]},$$

with ⟨X<sup>ˆ</sup>⟩ <sup>π</sup> <sup>=</sup> {⟨X<sup>ˆ</sup> ⟩ π t , t ∈ [0, T]} defined as

$$\langle \hat{\mathbf{X}} \rangle_t^\pi = \sum_{[u', v'] \in \pi[0, u]} \mathbf{X}_{u', v'}^2 + \frac{t - u}{v - u} \mathbf{X}_{u, v}^2, \quad t \in [u, v].$$

Similarly, we can approximate

$$\text{Var}(S_c^I(\mathbb{X}^\pi)_{[0,T]}) \approx \text{Var}(S_c^I(\mathbb{X})_{[0,T]}) \approx \sum_{J \in I_{-1} \cup I_{-1}} \hat{\psi}_{J*((i_k, i_k))}^{N,\pi}(T) \approx \sum_{J \in I_{-1} \cup I_{-1}} \hat{\psi}_{J*((i_k, i_k))}^{N,\pi'}(T),$$

and hence we define the second estimator for c ∗ π as

$$\begin{aligned} \hat{c}_{\pi,2}^* &= \frac{\sum_{J \in I \cup I} \hat{\psi}_J^{N,\pi,'}(T) - \frac{1}{2} \sum_{J \in I \cup I_{-2}*((i_{k-1},i_k))} \hat{\psi}_J^{N,\pi,'}(T)}{\sum_{J \in I_{-1} \cup I_{-1}} \hat{\psi}_{J*((i_k,i_k))}^{N,\pi,'}(T)} \\ &= \frac{\sum_{n=1}^N \left( \sum_{J \in I \cup I} S^J((\mathbb{X}^{n,\pi}, \langle \hat{\mathbb{X}} \rangle^{n,\pi}))_{[0,T]} - \frac{1}{2} \sum_{J \in I \cup I_{-2}*((i_{k-1},i_k))} S^J((\mathbb{X}^{n,\pi}, \langle \hat{\mathbb{X}} \rangle^{n,\pi}))_{[0,T]} \right)}{\sum_{n=1}^N \sum_{J \in I_{-1} \cup I_{-1}} S^{J*((i_k,i_k))}((\mathbb{X}^{n,\pi}, \langle \hat{\mathbb{X}} \rangle^{n,\pi}))_{[0,T]}} \end{aligned}$$

Whether estimator cˆ ∗ π,<sup>1</sup> or estimator cˆ ∗ π,2 is more precise, in terms of MSE, depends on the properties of the process X and the expected signature word I being estimated by [\(40\)](#page-33-2).

Lemma C.1. *Let* <sup>X</sup> <sup>=</sup> {<sup>X</sup>t, t ∈ [0, T]} *be a square-integrable martingale satisfying Assumption* (Aα[.M\)](#page-33-1)*, for some* <sup>α</sup> ≥ <sup>1</sup>/<sup>2</sup> *and* <sup>p</sup> = 4<sup>k</sup> *where* <sup>k</sup> <sup>=</sup> |I|*. Assume that* <sup>π</sup> *is part of a sequence of refining partitions with mesh vanishing fast enough, i.e.*

$$\sum_{n \geq 1} |\pi_n|^{2\alpha-1/2} < \infty.$$

*Then the difference between the mean-square errors of the two estimators* cˆ ∗ π,<sup>1</sup> *and* cˆ ∗ π,2 *for* c ∗ π *is approximately*

$$\mathbb{E}[(\hat{c}_{\pi,2}^* - c_\pi^*)^2] - \mathbb{E}[(\hat{c}_{\pi,1}^* - c_\pi^*)^2] \approx \frac{1}{N} \frac{\mu_Y}{\mu_Z^3} \left( \frac{\mu_Y}{\mu_Z} (\mathbb{E}[Z_2^2] - \mathbb{E}[Z_1^2]) - 2(\mathbb{E}[YZ_2] - \mathbb{E}[YZ_1]) \right),$$

*where*

$$Y = S^I(\mathbb{X})_{[0,T]}S_c^I(\mathbb{X})_{[0,T]}, \quad Z_1 = S_c^I(\mathbb{X})_{[0,T]}^2, \quad Z_2 = \sum_{J \in I_{-1} \cup I_{-1}} S^{J*((i_k, i_k))}((\mathbb{X}, \langle \mathbb{X} \rangle))_{[0,T]},$$

*and* µ<sup>Y</sup> = <sup>E</sup>[Y ]*,* µ<sup>Z</sup> = <sup>E</sup>[Z1] = <sup>E</sup>[Z2]*.*

*Proof.* See Appendix [C.3.](#page-35-0)

In practical applications, the above expression cannot be evaluated exactly, but we can approximate it by its sample estimate

$$\mathbb{E}[(\hat{c}_{\pi,2}^* - c_\pi^*)^2] - \mathbb{E}[(\hat{c}_{\pi,1}^* - c_\pi^*)^2] \propto \frac{1}{N^2} \sum_{n=1}^N \left( \frac{\bar{\mu}_Y}{\bar{\mu}_Z} ((Z_{2,n}^\pi)^2 - (Z_{1,n}^\pi)^2) - \sum_{j=1}^2 (Y_{j,n}^\pi Z_{2,n}^\pi - Y_{j,n}^\pi Z_{1,n}^\pi) \right).$$

where

$$\bar{\mu}_Y = \frac{1}{2N} \left( \sum_{n=1}^N Y_{1,n}^\pi + \sum_{n=1}^N Y_{2,n}^\pi \right), \quad \bar{\mu}_Z = \frac{1}{2N} \left( \sum_{n=1}^N Z_{1,n}^\pi + \sum_{n=1}^N Z_{2,n}^\pi \right),$$

and Y π 1,n, Z<sup>π</sup> <sup>1</sup>,n, Y <sup>π</sup> 2,n, Z<sup>π</sup> <sup>2</sup>,n are given in Appendix [C.3.](#page-35-0)

Another important discriminant when choosing between estimators cˆ ∗ π,1 and cˆ ∗ π,2 is usually computational cost. To compute cˆ ∗ π,1 it suffices to regress {<sup>S</sup> I (<sup>X</sup> n,π)[0,T] , n = 1, . . . , N} against {<sup>S</sup> I c (<sup>X</sup> n,π)[0,T] , n = 1, . . . , N}. Both samples need to be computed to evaluate the control-variate estimator [\(40\)](#page-33-2) and thus the extra computational cost of cˆ ∗ π,1 is just the cost of a simple linear regression with sample size <sup>N</sup>, namely O(N). On the other hand, to compute <sup>c</sup><sup>ˆ</sup> ∗ π,2 , one needs to compute all the higher-order expected signature estimates ψˆN,π,′ J (T) with <sup>J</sup> ∈ <sup>I</sup> <sup>I</sup>, <sup>J</sup> ∈ <sup>I</sup> <sup>I</sup>−<sup>2</sup> ∗ ((ik−1, ik)) and <sup>J</sup> <sup>=</sup> <sup>J</sup> ′ ∗ ((ik, ik)) for J ′ ∈ <sup>I</sup>−<sup>1</sup> <sup>I</sup>−1, which has (naive) extra computational cost O(|π| <sup>−</sup><sup>1</sup>T k(d <sup>2</sup><sup>k</sup> + (d + 1)<sup>2</sup>k−<sup>1</sup> )) when parallelizing across the <sup>N</sup> samples. In the in-fill limit, |π| <sup>−</sup><sup>1</sup><sup>T</sup> ≫ <sup>N</sup> and, hence, computing <sup>c</sup><sup>ˆ</sup> ∗ π,2 is significantly more expensive than computing cˆ ∗ π,1 .

#### C.3. Proof of Lemma [C.1](#page-34-0)

*Sketch of proof. To compare the two estimators* cˆ ∗ π,<sup>1</sup> *and* cˆ ∗ π,<sup>2</sup> *we exploit their structure as ratio estimators based on i.i.d. observations of numerator random variables* Y π <sup>1</sup> , Y <sup>π</sup> <sup>2</sup> *and denominator random variables* Z π <sup>1</sup> , Z<sup>π</sup> <sup>2</sup> *respectively. We first show that the two estimators are both (biased) estimators for* c ∗ <sup>π</sup> = c ∗ π,<sup>1</sup> = c ∗ π,<sup>2</sup> *where*

$$c_{\pi,1}^* = \frac{\mathbb{E}[Y_1^\pi]}{\mathbb{E}[Z_1^\pi]} \quad \text{and} \quad c_{\pi,2}^* = \frac{\mathbb{E}[Y_2^\pi]}{\mathbb{E}[Z_2^\pi]}.$$

*The first equality is trivial while the second requires several applications of Theorem [2.8](#page-3-1) which are detailed in Lemma [C.3.](#page-37-0) We then derive a simple formula for the mean squared error of ratio estimators in terms of the means and variances of the numerator and denominator random variables. Taking the limit* |π| ↓ 0 *and applying Theorem [2.8](#page-3-1) to show the second order statistics of* Y π <sup>1</sup> , Y <sup>π</sup> <sup>2</sup> , Z<sup>π</sup> <sup>1</sup> , Z<sup>π</sup> <sup>2</sup> *converge to those of* Y1, Y2, Z1, Z<sup>2</sup> *yields the desired result.*

Note that both cˆ ∗ π,1 and cˆ ∗ π,2 are ratio estimators of the form

$$\hat{C}_{\pi,j}^* = \frac{\bar{Y}_j^\pi}{\bar{Z}_j^\pi} = \frac{\sum_{n=1}^N Y_{j,n}^\pi}{\sum_{n=1}^N Z_{j,n}^\pi},$$

for random variables Y π j,1 , . . . , Y <sup>π</sup> j,N and Z π j,1 , . . . , Z<sup>π</sup> j,N with j = 1, 2 given by i.i.d. copies of

$$Y_1^\pi = S^I(\mathbb{X}^\pi)_{[0,T]}S_c^I(\mathbb{X}^\pi)_{[0,T]}, \quad Z_1^\pi = S_c^I(\mathbb{X}^\pi)_{[0,T]}^2,$$

and

$$\begin{aligned} Y_2^\pi &= \sum_{J \in I \cup I} S^J((\mathbb{X}^\pi, \langle \hat{\mathbb{X}} \rangle^\pi))_{[0, T]} - \frac{1}{2} \sum_{J \in I \cup I - 2 * ((i_{k-1}, i_k))} S^J((\mathbb{X}^\pi, \langle \hat{\mathbb{X}} \rangle^\pi))_{[0, T]}, \\ Z_2^\pi &= \sum_{J \in I - 1 \cup I - 1} S^{J * ((i_k, i_k))}((\mathbb{X}^\pi, \langle \hat{\mathbb{X}} \rangle^\pi))_{[0, T]}. \end{aligned}$$

By the standard theory of ratio estimators, these are biased estimators for

$$C_{\pi,j}^* = \frac{\mathbb{E}[Y_j^\pi]}{\mathbb{E}[Z_j^\pi]},$$

and applying a first-order Taylor expansion[<sup>19</sup>](#page-0-0) we can approximate the mean squared error as

$$\begin{aligned} \mathbb{E}[(\hat{c}_{\pi,j}^* - c_{\pi,j}^*)^2] &\approx \frac{1}{\mathbb{E}[Z_j^\pi]^2} \text{Var}(\bar{Y}_j^\pi) + \frac{\mathbb{E}[Y_j^\pi]^2}{\mathbb{E}[Z_j^\pi]^4} \text{Var}(\bar{Z}_j^\pi) - 2 \frac{\mathbb{E}[Y_j^\pi]}{\mathbb{E}[Z_j^\pi]^3} \text{Cov}(\bar{Y}_j^\pi, \bar{Z}_j^\pi) \\ &\approx \frac{1}{N} \left( \frac{1}{\mathbb{E}[Z_j^\pi]^2} \text{Var}(Y_j^\pi) + \frac{\mathbb{E}[Y_j^\pi]^2}{\mathbb{E}[Z_j^\pi]^4} \text{Var}(Z_j^\pi) - 2 \frac{\mathbb{E}[Y_j^\pi]}{\mathbb{E}[Z_j^\pi]^3} \text{Cov}(Y_j^\pi, Z_j^\pi) \right) \end{aligned}$$

.

Note that

$$c_{\pi,1}^* = \frac{\mathbb{E}[S^I(\mathbb{X}^\pi)_{[0,T]} S_c^I(\mathbb{X}^\pi)_{[0,T]}]}{\mathbb{E}[S_c^I(\mathbb{X}^\pi)_{[0,T]}^2]} = \frac{\text{Cov}(S^I(\mathbb{X}^\pi)_{[0,T]}, S_c^I(\mathbb{X}^\pi)_{[0,T]})}{\text{Var}(S_c^I(\mathbb{X}^\pi)_{[0,T]}^2)} = c_\pi^*,$$

and, as |π| → <sup>0</sup>,

$$\begin{aligned} c_{\pi,2}^* &= \frac{\sum_{J \in I \cup I} \mathbb{E}[S^J((\mathbb{X}^\pi, \langle \hat{\mathbb{X}})^\pi)]_{[0,T]}] - \frac{1}{2} \sum_{J \in I \cup I - 2*((i_{k-1}, i_k))} \mathbb{E}[S^J((\mathbb{X}^\pi, \langle \hat{\mathbb{X}})^\pi)]_{[0,T]}]}{\sum_{J \in I - 1 \cup I - 1} \mathbb{E}[S^{J*((i_k, i_k))}((\mathbb{X}^\pi, \langle \hat{\mathbb{X}})^\pi)]_{[0,T]}]} \\ &\stackrel{(46)}{\approx} \frac{\sum_{J \in I \cup I} \mathbb{E}[S^J((\mathbb{X}, \langle \mathbb{X})^\pi)]_{[0,T]}] - \frac{1}{2} \sum_{J \in I \cup I - 2*((i_{k-1}, i_k))} \mathbb{E}[S^J((\mathbb{X}, \langle \mathbb{X})^\pi)]_{[0,T]}]}{\sum_{J \in I - 1 \cup I - 1} \mathbb{E}[S^{J*((i_k, i_k))}((\mathbb{X}, \langle \mathbb{X})^\pi)]_{[0,T]}]} \\ &\stackrel{(43)}{\approx} \frac{\sum_{J \in I \cup I} \mathbb{E}[S^J((\mathbb{X}, \langle \mathbb{X}))]_{[0,T]}] - \frac{1}{2} \sum_{J \in I \cup I - 2*((i_{k-1}, i_k))} \mathbb{E}[S^J((\mathbb{X}, \langle \mathbb{X}))]_{[0,T]}]}{\sum_{J \in I - 1 \cup I - 1} \mathbb{E}[S^{J*((i_k, i_k))}((\mathbb{X}, \langle \mathbb{X}))]_{[0,T]}]} \\ &= \frac{\text{Cov}(S^I(\mathbb{X})_{[0,T]}, S_c^I(\mathbb{X})_{[0,T]})}{\text{Var}(S_c^I(\mathbb{X})_{[0,T]}^2)} \\ &\stackrel{(41),(42)}{\approx} \frac{\text{Cov}(S^I(\mathbb{X}^\pi)_{[0,T]}, S_c^I(\mathbb{X}^\pi)_{[0,T]})}{\text{Var}(S_c^I(\mathbb{X}^\pi)_{[0,T]}^2)} = c_\pi^*. \end{aligned}$$

We refer to Lemma [C.3](#page-37-0) for the rigorous justification of the approximations [\(41\)](#page-38-1), [\(42\)](#page-38-2), [\(43\)](#page-38-0), [\(46\)](#page-40-3). Moreover, as |π| → <sup>0</sup>, by [\(41\)](#page-38-1) and [\(42\)](#page-38-2),

$$Y_1^\pi \xrightarrow{L^2} S^I(\mathbb{X})_{[0,T]} S_c^I(\mathbb{X})_{[0,T]} =: Y_1, \quad Z_1^\pi \xrightarrow{L^2} S_c^I(\mathbb{X})_{[0,T]}^2 =: Z_1,$$

and by combining [\(43\)](#page-38-0) and [\(46\)](#page-40-3),

$$\begin{aligned} Y_2^\pi &\xrightarrow{L_2^\pi} \sum_{J \in I \cup I \setminus I} S^J((\mathbb{X}, \langle \mathbb{X} \rangle))_{[0, T]} - \frac{1}{2} \sum_{J \in I \cup I \setminus I - 2*((i_{k-1}, i_k))} S^J((\mathbb{X}, \langle \mathbb{X} \rangle))_{[0, T]} =: Y_2, \\ Z_2^\pi &\xrightarrow{L_2^\pi} \sum_{J \in I - 1 \cup I \setminus I} S^{J*((i_k, i_k))}((\mathbb{X}, \langle \mathbb{X} \rangle))_{[0, T]} =: Z_2. \end{aligned}$$

Note that <sup>Y</sup><sup>1</sup> <sup>=</sup> <sup>Y</sup><sup>2</sup> =: <sup>Y</sup> but <sup>Z</sup><sup>1</sup> ̸<sup>=</sup> <sup>Z</sup><sup>2</sup> even though <sup>E</sup>[Z1] = <sup>E</sup>[Z2] =: <sup>µ</sup>Z. When |π| is small we can thus approximate the MSEs of cˆ ∗ π,1 and cˆ ∗ π,<sup>2</sup> with respect to c ∗ π as

$$\mathbb{E}[(\hat{c}_{\pi,1}^* - c_\pi^*)^2] \approx \frac{1}{N} \left( \frac{1}{\mu_Z^2} \sigma_Y^2 + \frac{\mu_Y^2}{\mu_Z^4} \text{Var}(Z_1) - 2 \frac{\mu_Y}{\mu_Z^3} \text{Cov}(Y, Z_1) \right),$$

$$\begin{aligned} \hat{c}_{\pi,j} &= c_{\pi,j}^* \left( 1 + \frac{\bar{Y}_j^\pi - \mathbb{E}[Y_j^\pi]}{\mathbb{E}[Y_j^\pi]} \right) \left( 1 + \frac{\bar{Z}_j^\pi - \mathbb{E}[Z_j^\pi]}{\mathbb{E}[Z_j^\pi]} \right)^{-1} \\ &= c_{\pi,j}^* \left( 1 + \frac{\bar{Y}_j^\pi - \mathbb{E}[Y_j^\pi]}{\mathbb{E}[Y_j^\pi]} \right) \left( 1 - \frac{\bar{Z}_j^\pi - \mathbb{E}[Z_j^\pi]}{\mathbb{E}[Z_j^\pi]} + \mathcal{O} \left( \left( \frac{\bar{Z}_j^\pi - \mathbb{E}[Z_j^\pi]}{\mathbb{E}[Z_j^\pi]} \right)^2 \right) \right) \\ &= c_{\pi,j}^* + \frac{1}{\mathbb{E}[Z_j^\pi]} (\bar{Y}_j^\pi - \mathbb{E}[Y_j^\pi]) - \frac{\mathbb{E}[Y_j^\pi]}{\mathbb{E}[Z_j^\pi]^2} (\bar{Z}_j^\pi - \mathbb{E}[Z_j^\pi]) + \mathcal{O} \left( \left( \frac{\bar{Y}_j^\pi - \mathbb{E}[Y_j^\pi]}{\mathbb{E}[Y_j^\pi]} \right) \left( \frac{\bar{Z}_j^\pi - \mathbb{E}[Z_j^\pi]}{\mathbb{E}[Z_j^\pi]} \right) \right). \end{aligned}$$

For the approximation to be rigorous one needs to assume the probability of the set {ω : |Z¯<sup>π</sup> <sup>j</sup> (ω) − <sup>E</sup>[Z π ]| ≥ |E[Z π ]|} approaches 0 faster than the speed at which the MSE conditional on this set explodes as N → ∞.

<sup>19</sup>On the set |Z¯<sup>π</sup> <sup>j</sup> − <sup>E</sup>[Z π j ]| < |E[Z π j ]|,

and

$$\mathbb{E}[(\hat{c}_{\pi,2}^* - c_\pi^*)^2] \approx \frac{1}{N} \left( \frac{1}{\mu_Z^2} \sigma_Y^2 + \frac{\mu_Y^2}{\mu_Z^4} \text{Var}(Z_2) - 2 \frac{\mu_Y}{\mu_Z^3} \text{Cov}(Y, Z_2) \right),$$

which differ by

$$\mathbb{E}[(\hat{c}_{\pi,2}^* - c_\pi^*)^2] - \mathbb{E}[(\hat{c}_{\pi,1}^* - c_\pi^*)^2] \approx \frac{1}{N} \frac{\mu_Y}{\mu_Z^3} \left( \frac{\mu_Y}{\mu_Z} (\mathbb{E}[Z_2^2] - \mathbb{E}[Z_1^2]) - 2(\mathbb{E}[YZ_2] - \mathbb{E}[YZ_1]) \right).$$

*Remark* C.2*.* Note that we can "mix" the two estimators and form

$$\hat{c}_{\pi,2,1}^* = \frac{\bar{Y}_2^\pi}{\bar{Z}_1^\pi} \quad \text{and} \quad \hat{c}_{\pi,1,2}^* = \frac{\bar{Y}_1^\pi}{\bar{Z}_2^\pi}.$$

The discussion above ensures that, as |π| → <sup>0</sup>, <sup>c</sup><sup>ˆ</sup> ∗ π,2,1 and cˆ ∗ π,1,<sup>2</sup> have the same MSEs as cˆ ∗ π,1,<sup>1</sup> = ˆcπ,<sup>1</sup> and cˆ ∗ π,2,<sup>2</sup> = ˆc ∗ π,2 , respectively.

Lemma C.3. *Assume that* <sup>X</sup> *satisfies* (Aα[.M\)](#page-33-1) *for some* <sup>α</sup> ≥ <sup>1</sup>/<sup>2</sup> *and* <sup>p</sup> = 4<sup>k</sup> *where* <sup>k</sup> <sup>=</sup> |I|*. Assume that* <sup>π</sup> *is part of a sequence of refining partitions with mesh vanishing fast enough, i.e.*

$$\sum_{n \geq 1} |\pi_n|^{2\alpha-1/2} < \infty.$$

*Then the approximations* [\(41\)](#page-38-1)*,* [\(42\)](#page-38-2)*,* [\(43\)](#page-38-0)*,* [\(46\)](#page-40-3) *hold as* |π| → <sup>0</sup>*.*

*Proof.* By Appendix [C.1](#page-33-1) <sup>X</sup> satisfies [\(A](#page-3-0)α) with <sup>α</sup> ≥ <sup>1</sup>/<sup>2</sup> and <sup>p</sup> = 4k. Note that since <sup>X</sup> is a martingale [\(A](#page-3-0)δ) holds trivially and hence we can set ϵ = 1/2.

We can apply Theorem [2.8](#page-3-1) to deduce that

$$S^I(\mathbb{X}^\pi)_{[0,T]} \xrightarrow{L^4} S^I(\mathbb{X})_{[0,T]}, \quad |\pi| \rightarrow 0.$$

Moreover,

$$S_c^I(\mathbb{X}^\pi)_{[0,T]} \xrightarrow{L^4} S_c^I(\mathbb{X})_{[0,T]}, \quad |\pi| \rightarrow 0.$$

since

$$\begin{aligned} & \|S_c^I(\mathbb{X})_{[0,T]} - S_c^I(\mathbb{X}^\pi)_{[0,T]}\|_{L^4} \\ & \stackrel{(i)}{\leq} \left\| S_c^I(\mathbb{X})_{[0,T]} - \sum_{[u,v] \in \pi} S^{I-1}(\mathbb{X})_{[0,u]} X_{u,v}^{(i_k)} \right\|_{L^4} + \left\| \sum_{[u,v] \in \pi} (S^{I-1}(\mathbb{X})_{[0,u]} - S^{I-1}(\mathbb{X}^\pi)_{[0,u]}) X_{u,v}^{(i_k)} \right\|_{L^4} \\ & \stackrel{(ii)}{\leq} \left\| \int_0^T S^{I-1}(\mathbb{X})_{[0,s]} dX_s^{(i_k)} - \sum_{[u,v] \in \pi} S^{I-1}(\mathbb{X})_{[0,u]} X_{u,v}^{(i_k)} \right\|_{L^4} \\ & \quad + 2C_2 \left( \sum_{[u,v] \in \pi} \|S^{I-1}(\mathbb{X})_{[0,u]} - S^{I-1}(\mathbb{X}^\pi)_{[0,u]}\|_{L^{2k/(k-1)}}^2 \|X_{u,v}^{(i_k)}\|_{L^{2k}}^2 \right)^{1/2} \\ & \stackrel{(iii)}{\lesssim} \left\| \int_0^T S^{I-1}(\mathbb{X})_{[0,s]} dX_s^{(i_k)} - \sum_{[u,v] \in \pi} S^{I-1}(\mathbb{X})_{[0,u]} X_{u,v}^{(i_k)} \right\|_{L^4} \\ & \quad + 2C_2 \sup_{u \in \pi} \left\| S^{I-1}(\mathbb{X})_{[0,u]} - S^{I-1}(\mathbb{X}^\pi)_{[0,u]} \right\|_{L^{2k/(k-1)}} \left( \sum_{[u,v] \in \pi} |v - u|^{2\alpha} \right)^{1/2} \\ & \xrightarrow{(iv)} 0, \quad |\pi| \rightarrow 0, \end{aligned}$$

by applying in (i) the triangle inequality, in (ii) Lemma [B.1](#page-13-0) with the natural filtration of <sup>X</sup>, in (iii) Theorem [2.8](#page-3-1) to the word <sup>I</sup>−<sup>1</sup> with <sup>m</sup> = 2k/(<sup>k</sup> − 1) and in (iv) the definition of the Ito integral and Remark <sup>ˆ</sup> [B.4.](#page-22-1)

Under these assumptions, we thus have

$$S^I(\mathbb{X}^\pi)_{[0,T]}S_c^I(\mathbb{X}^\pi)_{[0,T]} \xrightarrow{L^2} S^I(\mathbb{X})_{[0,T]}S_c^I(\mathbb{X})_{[0,T]}, \quad |\pi| \rightarrow 0, \quad (41)$$

and

$$S_c^I(\mathbb{X}^\pi)_{[0,T]}^2 \xrightarrow{L^2} S_c^I(\mathbb{X})_{[0,T]}^2, \quad |\pi| \rightarrow 0. \quad (42)$$

Next, we consider the approximation [\(43\)](#page-38-0). Note that <sup>X</sup> satisfies [\(A](#page-3-0)α) (and trivially [\(A](#page-3-0)δ)) with <sup>α</sup> ≥ <sup>1</sup>/<sup>2</sup> and <sup>p</sup> = 4<sup>k</sup> while ⟨X⟩ satisfies [\(A](#page-3-0)α) and [\(A](#page-3-0)δ) both with exponent <sup>2</sup><sup>α</sup> ≥ <sup>1</sup> and <sup>p</sup> = 2k. By using a slightly more general version[<sup>20</sup>](#page-0-0) of Theorem [2.8](#page-3-1) applied to the process (X,⟨X⟩), for any word <sup>J</sup> ∈ <sup>I</sup> <sup>I</sup> or <sup>J</sup> ∈ <sup>I</sup> <sup>I</sup>−<sup>2</sup> ∗ ((ik−1, ik)) or <sup>J</sup> <sup>=</sup> <sup>J</sup> ′ ∗ ((ik, ik)) with J ′ ∈ <sup>I</sup>−<sup>1</sup> <sup>I</sup>−<sup>1</sup> we have

$$S^J((\mathbb{X}, \langle \mathbb{X} \rangle)^\pi)_{[0, T]} \xrightarrow{L^2} S^J((\mathbb{X}, \langle \mathbb{X} \rangle))_{[0, T]}, \quad |\pi| \rightarrow 0, \quad (43)$$

and hence for fixed <sup>N</sup> ≥ <sup>1</sup>,

$$\hat{\psi}_J^{N,\pi}(T) \xrightarrow{L^2} \hat{\psi}_J^N(T), \quad |\pi| \rightarrow 0.$$

Finally, making approximation [\(46\)](#page-40-3) rigorous is a bit more challenging. Let us start by considering the case where ⟨X⟩ only appears in the outermost integral, i.e. <sup>J</sup> = (j1, . . . , jk′ ), where <sup>j</sup>1, . . . , jk′−<sup>1</sup> ∈ {1, . . . , d} and <sup>j</sup>k′ ∈ {(1, 1), . . . ,(1, d), . . . ,(d, d)} for some <sup>k</sup> ′ ∈ {1, . . . , <sup>2</sup><sup>k</sup> − <sup>1</sup>}. Then, for any <sup>τ</sup> ∈ <sup>π</sup>,

$$\begin{aligned} & \left\| S^J((\mathbb{X}, \langle \mathbb{X} \rangle)^\pi)_{[0, \tau]} - S^J((\mathbb{X}^\pi, \langle \hat{\mathbb{X}}^\pi \rangle)_{[0, \tau]}) \right\|_{L^{4k/(k'+1)}} \\ & \stackrel{(i)}{=} \left\| \sum_{i=1}^{k'} \frac{1}{i!} \sum_{[u, v] \in \pi_{[0, \tau]}} S^{J-i}(\mathbb{X}^\pi)_{[0, u]} X_{u, v}^{(j_{k'}-i+1)} \cdots X_{u, v}^{(j_{k'}-1)} \left( \langle X \rangle_{u, v}^{(j_{k'})} - \langle \hat{X} \rangle_{u, v}^{(j_{k'})} \right) \right\|_{L^{4k/(k'+1)}} \\ & \stackrel{(ii)}{\leq} \sum_{i=1}^{k'} \frac{1}{i!} \left\| \sum_{[u, v] \in \pi_{[0, \tau]}} S^{J-i}(\mathbb{X}^\pi)_{[0, u]} X_{u, v}^{(j_{k'}-i+1)} \cdots X_{u, v}^{(j_{k'}-1)} \left( \langle X \rangle_{u, v}^{(j_{k'})} - \left( X_{u, v}^{(j_{k'})} \right)^2 \right) \right\|_{L^{4k/(k'+1)}} \\ & \stackrel{(iii)}{\leq} \left( \sum_{[u, v] \in \pi_{[0, \tau]}} \left\| S^{J-1}(\mathbb{X}^\pi)_{[0, u]} \left( \langle X \rangle_{u, v}^{(j_{k'})} - \left( X_{u, v}^{(j_{k'})} \right)^2 \right) \right\|_{L^{4k/(k'+1)}}^2 \right)^{1/2} \\ & \quad + \sum_{i=2}^{k'} \frac{1}{i!} \sum_{[u, v] \in \pi_{[0, \tau]}} \left\| S^{J-i}(\mathbb{X}^\pi)_{[0, u]} X_{u, v}^{(j_{k'}-i+1)} \cdots X_{u, v}^{(j_{k'}-1)} \left( \langle X \rangle_{u, v}^{(j_{k'})} - \left( X_{u, v}^{(j_{k'})} \right)^2 \right) \right\|_{L^{4k/(k'+1)}} \\ & \stackrel{(iii)}{\leq} \left( \sum_{[u, v] \in \pi_{[0, \tau]}} \left\| S^{J-1}(\mathbb{X}^\pi)_{[0, u]} \left( \langle X \rangle_{u, v}^{(j_{k'})} - \left( X_{u, v}^{(j_{k'})} \right)^2 \right) \right\|_{L^{4k/(k'+1)}}^2 \right)^{1/2} \\ & \quad + \sum_{i=2}^{k'} \frac{1}{i!} \sum_{[u, v] \in \pi_{[0, \tau]}} \left\| S^{J-i}(\mathbb{X}^\pi)_{[0, u]} X_{u, v}^{(j_{k'}-i+1)} \cdots X_{u, v}^{(j_{k'}-1)} \left( \langle X \rangle_{u, v}^{(j_{k'})} - \left( X_{u, v}^{(j_{k'})} \right)^2 \right) \right\|_{L^{4k/(k'+1)}} \\ & \stackrel{(iv)}{\leq} \sup_{u \in \pi_{[0, \tau]}} \left\| S^{J-1}(\mathbb{X}^\pi)_{[0, u]} \right\|_{L^{4k/(k'-1)}} \left( \sum_{[u, v] \in \pi_{[0, \tau]}} \|\langle \mathbf{X} \rangle_{u, v} - \mathbf{X}_{u, v}^{\otimes 2}\|_{L^{2k}}^2 \right)^{1/2} \\ & \quad + \sum_{i=2}^{k'} \frac{1}{i!} \sup_{u \in \pi_{[0, \tau]}} \left\| S^{J-i}(\mathbb{X}^\pi)_{[0, u]} \right\|_{L^{4k/(k'-i)}} \sum_{[u, v] \in \pi_{[0, \tau]}} \|\mathbf{X}_{u, v}\|_{L^{4k}}^{i-1} \|\langle \mathbf{X} \rangle_{u, v} - \mathbf{X}_{u, v}^{\otimes 2}\|_{L^{2k}} \\ & \stackrel{(v)}{\lesssim} |\pi|^{4\alpha-1} \sqrt{\tau} + |\pi| \tau \rightarrow 0, \quad |\pi| \rightarrow 0, \end{aligned}$$

<sup>20</sup>To estimate the expected signature term corresponding to the word I = (i1, . . . , ik), it is sufficient to require *(i)*, *(ii)*, *(iii)*, or *(iv)* to be satisfied by X (i1) , . . . , X(ik) with p1, . . . , p<sup>k</sup> such that p −1 <sup>1</sup> + · · · + p −1 <sup>k</sup> ≤ <sup>m</sup>−<sup>1</sup> . In Theorem [2.8](#page-3-1) we considered the case where p<sup>1</sup> = · · · = p<sup>k</sup> = p for clarity.

by using in (i) Lemma [B.3,](#page-14-1) in (ii) the triangle inequality and the definition of ⟨X<sup>ˆ</sup>⟩ π , in (iii) Lemma [B.1](#page-13-0) with the natural filtration of <sup>X</sup> (with respect to which <sup>X</sup> − ⟨X⟩ is a martingale) for the <sup>i</sup> = 1 term and the triangle inequality for the i = 2, . . . , j terms, in (iv) Holder inequality, in ¨ (v) Remark [B.4](#page-22-1) applied to <sup>X</sup> and J−<sup>i</sup> with p = 4k, m = 4k/(k ′ − <sup>i</sup>) and |<sup>J</sup>−<sup>i</sup> | <sup>=</sup> <sup>k</sup> ′ − <sup>i</sup> for <sup>i</sup> = 1, . . . , k′ and

$$\|\langle \mathbf{X} \rangle_{u,v} - \mathbf{X}_{u,v}^{\otimes 2}\|_{L^{2k}} \leq \|\langle \mathbf{X} \rangle_{u,v}\|_{L^{2k}} + \|\mathbf{X}_{u,v}\|_{L^{4k}}^2 \leq |v - u|^{2\alpha}, \quad (44)$$

by assumption (Aα[.M\)](#page-33-1) with <sup>α</sup> ≥ <sup>1</sup>/<sup>2</sup> and <sup>p</sup> = 4k. We have thus shown

$$\sup_{\tau \in \pi} \left\| S^J((\mathbb{X}, \langle \mathbb{X} \rangle)^\pi)_{[0, \tau]} - S^J((\mathbb{X}^\pi, \langle \hat{\mathbb{X}} \rangle^\pi)_{[0, \tau]}) \right\|_{L^{4k_f/(k'_f+1)}} \lesssim |\pi|. \quad (45)$$

This extends to any <sup>J</sup> = (j1, . . . , jk′ , . . . , jk′+m′ ) with <sup>m</sup>′ ≥ <sup>0</sup> and <sup>1</sup> ≤ <sup>k</sup> ′ , k′ <sup>+</sup> <sup>m</sup>′ ≤ <sup>2</sup><sup>k</sup> − <sup>1</sup>, where, as above, <sup>j</sup>k′ ∈ {(1, 1), . . . ,(1, d), . . . ,(d, d)} and <sup>j</sup><sup>i</sup> ∈ {1, . . . , d} for all other <sup>i</sup> ̸<sup>=</sup> <sup>k</sup> ′ . We proceed inductively on <sup>m</sup>′ = 0, . . . , <sup>2</sup><sup>k</sup> − <sup>k</sup> ′ − <sup>1</sup> to show Equation [\(45\)](#page-39-0) holds for any J of this form in the L 4k/(k ′+m′+1) norm. The case m′ = 0 has been covered above. Assume [\(45\)](#page-39-0) holds for <sup>J</sup>k′+<sup>m</sup> with <sup>m</sup> ∈ {0, . . . , m′} and <sup>0</sup> ≤ <sup>m</sup>′ ≤ <sup>2</sup><sup>k</sup> − <sup>k</sup> ′ − <sup>1</sup>, then

$$\begin{aligned} & \left\| S^{J_{k'+m'+1}}((\mathbb{X}, \langle \mathbb{X} \rangle)^\pi)_{[0,\tau]} - S^{J_{k'+m'+1}}((\mathbb{X}^\pi, \langle \hat{\mathbb{X}} \rangle^\pi))_{[0,\tau]} \right\|_{L^{4k/(k'+m'+2)}} \\ & \stackrel{(i)}{=} \left\| \sum_{i=1}^{m'+1} \frac{1}{i!} \sum_{[u,v] \in \pi_{[0,\tau]}} \left[ S^{J_{k'+m'+1-i}}((\mathbb{X}, \langle \mathbb{X} \rangle)^\pi)_{[0,u]} - S^{J_{k'+m'+1-i}}((\mathbb{X}^\pi, \langle \hat{\mathbb{X}} \rangle^\pi))_{[0,u]} \right] X_{u,v}^{(j_{k'+m'+2-i})} \dots X_{u,v}^{(j_{k'+m'+1})} \right. \\ & \quad + \sum_{i=m'+2}^{k'+m'+1} \frac{1}{i!} \sum_{[u,v] \in \pi_{[0,\tau]}} S^{J_{k'+m'+1-i}}(\mathbb{X}^\pi)_{[0,u]} \\ & \quad \times X_{u,v}^{(j_{k'+m'+2-i})} \dots \left( \langle X \rangle_{u,v}^{(j_{k'})} - \langle \hat{X} \rangle_{u,v}^{\pi,(j_{k'})} \right) \dots X_{u,v}^{(j_{k'+m'+1})} \left\|_{L^{4k/(k'+m'+2)}} \right. \\ & \stackrel{(ii)}{\leq} \left\| \sum_{[u,v] \in \pi_{[0,\tau]}} \left[ S^{J_{k'+m'}}((\mathbb{X}, \langle \mathbb{X} \rangle)^\pi)_{[0,u]} - S^{J_{k'+m'}}((\mathbb{X}^\pi, \langle \hat{\mathbb{X}} \rangle^\pi))_{[0,u]} \right] X_{u,v}^{(j_{k'+m'+1})} \right\|_{L^{4k/(k'+m'+2)}} \\ & \quad + \sum_{i=2}^{m'+1} \frac{1}{i!} \left\| \sum_{[u,v] \in \pi_{[0,\tau]}} \left[ S^{J_{k'+m'+1-i}}((\mathbb{X}, \langle \mathbb{X} \rangle)^\pi)_{[0,u]} - S^{J_{k'+m'+1-i}}((\mathbb{X}^\pi, \langle \hat{\mathbb{X}} \rangle^\pi))_{[0,u]} \right] \right. \\ & \quad \times X_{u,v}^{(j_{k'+m'+2-i})} \dots X_{u,v}^{(j_{k'+m'+1})} \left\|_{L^{4k/(k'+m'+2)}} \right. \\ & \quad + \sum_{i=m'+2}^{k'+m'+1} \frac{1}{i!} \left\| \sum_{[u,v] \in \pi_{[0,\tau]}} S^{J_{k'+m'+1-i}}(\mathbb{X}^\pi)_{[0,u]} \right. \\ & \quad \times X_{u,v}^{(j_{k'+m'+2-i})} \dots \left( \langle X \rangle_{u,v}^{(j_{k'})} - \left( X_{u,v}^{(j_{k'})} \right)^2 \right) \dots X_{u,v}^{(j_{k'+m'+1})} \left\|_{L^{4k/(k'+m'+2)}} \right. \\ & \stackrel{(iii)}{\leq} \sup_{u \in \pi_{[0,\tau]}} \left\| S^{J_{k'+m'}}((\mathbb{X}, \langle \mathbb{X} \rangle)^\pi)_{[0,u]} - S^{J_{k'+m'}}((\mathbb{X}^\pi, \langle \hat{\mathbb{X}} \rangle^\pi))_{[0,u]} \right\|_{L^{4k/(k'+m'+1)}} \left( \sum_{[u,v] \in \pi_{[0,\tau]}} \|\mathbf{X}_{u,v}\|_{L^{4k}}^2 \right)^{1/2} \\ & \quad + \sum_{i=2}^{m'+1} \frac{1}{i!} \sup_{u \in \pi_{[0,\tau]}} \left\| S^{J_{k'+m'+1-i}}((\mathbb{X}, \langle \mathbb{X} \rangle)^\pi)_{[0,u]} - S^{J_{k'+m'+1-i}}((\mathbb{X}^\pi, \langle \hat{\mathbb{X}} \rangle^\pi))_{[0,u]} \right\|_{L^{4k/(k'+m'-i+2)}} \\ & \quad \times \left( \sum_{[u,v] \in \pi_{[0,\tau]}} \|\mathbf{X}_{u,v}\|_{L^{4k}}^i \right) \\ & \quad + \sum_{i=m'+2}^{k'+m'+1} \frac{1}{i!} \sup_{u \in \pi_{[0,\tau]}} \left\| S^{J_{k'+m'+1-i}}(\mathbb{X}^\pi)_{[0,u]} \right\|_{L^{4k/(k'+m'+1-i)}} \left( \sum_{[u,v] \in \pi_{[0,\tau]}} \|\mathbf{X}_{u,v}\|_{L^{4k}}^{i-1} \|\langle \mathbf{X} \rangle_{u,v} - \mathbf{X}_{u,v}^{\otimes 2}\|_{L^{2k}} \right) \\ & \stackrel{(v)}{\lesssim} |\pi| \sqrt{\tau} + |\pi| \tau \rightarrow 0, |\pi| \rightarrow 0, \end{aligned}$$

where in (i) we have used Lemma [B.3,](#page-14-1) in (ii) triangle inequality, in (iii) Lemma [B.1](#page-13-0) with the natural filtration of <sup>X</sup> for the i = 1 term, traingle inequality for the i = 2, . . . , k′ + m′ + 1 terms and Holder inequality across all terms, in ¨ (iv) the inductive hypothesis for Jk′+<sup>m</sup> with m = 0, . . . , m′ for the first two sups, Remark [B.4](#page-22-1) applied to <sup>X</sup> and J<sup>i</sup> with p = 4k, <sup>m</sup> = 4k/i and |<sup>J</sup><sup>i</sup> | <sup>=</sup> <sup>i</sup> for <sup>i</sup> = 1, . . . , k′ − <sup>1</sup> for the third sup and bounds [\(44\)](#page-39-1) and [\(A](#page-3-0)α) with <sup>p</sup> = 4<sup>k</sup> and <sup>α</sup> ≥ <sup>1</sup>/<sup>2</sup> for the summations over π[0,τ] .

We have thus shown that for any word <sup>J</sup> ∈ <sup>I</sup> <sup>I</sup> or <sup>J</sup> ∈ <sup>I</sup> <sup>I</sup>−<sup>2</sup> ∗ ((ik−1, ik)) or <sup>J</sup> <sup>=</sup> <sup>J</sup> ′ ∗ ((ik, ik)) for <sup>J</sup> ′ ∈ <sup>I</sup>−<sup>1</sup> <sup>I</sup>−<sup>1</sup> we have

$$S^J((\mathbb{X}, \langle \mathbb{X} \rangle^\pi)_{[0,T]} - S^J((\mathbb{X}^\pi, \langle \hat{\mathbb{X}} \rangle^\pi)_{[0,T]} \xrightarrow{L^2} 0, \quad |\pi| \rightarrow 0, \quad (46)$$

and hence for fixed <sup>N</sup> ≥ <sup>1</sup>,

$$\hat{\psi}_J^{N,\pi}(T) - \hat{\psi}_J^{N,\pi,'}(T) \xrightarrow{L^2} 0, \quad |\pi| \rightarrow 0.$$

#### D. Ito processes and diffusions ˆ

In this section we consider Ito processes and It ˆ o diffusions: two common classes of models for continuous-time stochastic ˆ processes. We start by providing sufficient conditions ensuring Assumption [\(2.6\)](#page-3-0), needed for the in-fill asymptotics, holds. We then focus on time-homogeneous Ito diffusions and discuss general conditions under which these processes are ˆ stationary and strongly mixing, ensuring stationarity and strong mixing of {<sup>X</sup> <sup>n</sup>, n ≥ <sup>1</sup>} under (chop) observations (cf. Proposition [2.13\)](#page-4-0).

#### D.1. In-fill conditions

#### D.1.1. ITO PROCESSES ˆ

We consider the case where X is an Ito process, i.e. satisfies ˆ

$$\mathbf{X}_t = \mathbf{X}_0 + \int_0^t \mathbf{b}_s \, ds + \int_0^t V_s \, d\mathbf{W}_s, \quad t \in [0, T],$$

where <sup>b</sup> <sup>=</sup> {<sup>b</sup>t, t ∈ [0, T]} and <sup>V</sup> <sup>=</sup> {<sup>V</sup>t, t ∈ [0, T]} are progressively measurable <sup>d</sup>- and <sup>d</sup> × <sup>q</sup>-dimensional processes such that

$$\sup_{s \in [0, T]} \|\mathbf{b}_s\|_{L^p}, \quad \sup_{s \in [0, T]} \|V_s\|_{L^p} < \infty, \quad (47)$$

<sup>W</sup> <sup>=</sup> {<sup>W</sup>t, t ≥ <sup>0</sup>} is a <sup>q</sup>-dimensional Brownian motion and <sup>X</sup><sup>0</sup> ∈ <sup>L</sup> p . The assumptions on b and V imply that for all <sup>0</sup> ≤ <sup>s</sup> ≤ <sup>t</sup> ≤ <sup>T</sup>,

$$\left\| \int_s^t \mathbf{b}_u du \right\|_{L^p} \leq \int_s^t \|\mathbf{b}_u\|_{L^p} du \quad (48)$$

$$\leq \left( \sup_{u \in [0, T]} \|\mathbf{b}_u\|_{L^p} \right) |t - s|, \quad (49)$$

by Minkowski's integral inequality, and

$$\begin{aligned} \left\| \int_s^t V_u d\mathbf{W}_u \right\|_{L^p} &\lesssim \mathbb{E} \left[ \left( \text{tr} \int_s^t V_u V_u^T du \right)^{p/2} \right]^{1/p} = \left\| \int_s^t \|V_u\|^2 du \right\|_{L^{p/2}}^{1/2} \\ &\leq \left( \int_s^t \|V_u\|_{L^p}^2 du \right)^{1/2} \end{aligned} \quad (50)$$

$$\leq \left( \sup_{u \in [0, T]} \|V_u\|_{L^p} \right) |t - s|^{1/2}, \quad (51)$$

by Burkholder-Davis-Gundy (BDG) inequality [\(Burkholder et al.,](#page-9-14) [1972\)](#page-9-14), the formula for the quadratic variation of the Itoˆ integral, and Minkowski integral inequality. We can show [\(A](#page-3-0)α) holds with <sup>α</sup> = 1/<sup>2</sup> by noting that for all <sup>0</sup> ≤ <sup>s</sup> ≤ <sup>t</sup> ≤ <sup>T</sup>,

$$\|\mathbf{X}_{s,t}\|_{L^p} \leq \left\| \int_s^t \mathbf{b}_u du \right\|_{L^p} + \left\| \int_s^t V_u d\mathbf{W}_u \right\|_{L^p} \lesssim |t-s| + |t-s|^{1/2} \lesssim |t-s|^{1/2},$$

by combining bounds [\(49\)](#page-40-4) and [\(51\)](#page-40-5).

Next, we can show [\(A](#page-3-0)δ) holds with <sup>δ</sup> = 1 by noting that for all <sup>0</sup> ≤ <sup>s</sup> ≤ <sup>t</sup> ≤ <sup>T</sup>,

$$\|\mathbb{E}_s[\mathbf{X}_{s,t}]\|_{L^p} = \left\| \mathbb{E}_s \left[ \int_s^t \mathbf{b}_u du \right] \right\|_{L^p} \leq \left\| \int_s^t \mathbf{b}_u du \right\|_{L^p} \lesssim |t-s|,$$

where we use the martingale property of Ito integrals, contractive property of conditional expectation and the bound ( ˆ [49\)](#page-40-4).

#### D.1.2. ITO DIFFUSIONS ˆ

Next, assume X is a (possibly time-inhomogeneous) Ito diffusion, i.e. satisfies ˆ

$$d\mathbf{X}_t = f(t, \mathbf{X}_t)dt + \sigma(t, \mathbf{X}_t)d\mathbf{W}_t, \quad t \in [0, T],$$

where <sup>W</sup> <sup>=</sup> {<sup>W</sup>t, t ∈ [0, T]} is a <sup>q</sup>-dimensional Brownian motion, <sup>f</sup> : [0, T] × <sup>R</sup> <sup>d</sup> → <sup>R</sup> d , σ : [0, T] × <sup>R</sup> <sup>d</sup> → <sup>R</sup> d×q and <sup>X</sup><sup>0</sup> ∈ <sup>L</sup> p . Ito diffusions form a subclass of It ˆ o processes, which we already covered in Appendix ˆ [D.1.1.](#page-40-2) Here, we give conditions specific to Ito diffusions – i.e. in terms of ˆ f and σ – which imply condition [\(47\)](#page-40-6).

- If <sup>f</sup> and <sup>σ</sup> are uniformly bounded on [0, T] × <sup>R</sup> d , then condition [\(47\)](#page-40-6) immediately holds.
- Assume <sup>f</sup> and <sup>σ</sup> are Lipschitz continuous, i.e. for all s, t ∈ [0, T] and <sup>x</sup>, <sup>y</sup> ∈ <sup>R</sup> d ,

$$\begin{aligned}\|f(t, \mathbf{x}) - f(s, \mathbf{y})\| &\leq K_f\|(t, \mathbf{x}) - (s, \mathbf{y})\| \leq K_f(|t - s| + \|\mathbf{x} - \mathbf{y}\|), \\ \|\sigma(t, \mathbf{x}) - \sigma(s, \mathbf{y})\| &\leq K_\sigma\|(t, \mathbf{x}) - (s, \mathbf{y})\| \leq K_\sigma(|t - s| + \|\mathbf{x} - \mathbf{y}\|).\end{aligned}$$

Then, for all <sup>0</sup> ≤ <sup>s</sup> ≤ <sup>t</sup> ≤ <sup>T</sup>, we can bound

$$\begin{aligned} & \|\mathbf{X}_{s,t}\|_{L^p} \\ &= \left\| \int_s^t f(u, \mathbf{X}_u) du + \int_s^t \sigma(u, \mathbf{X}_u) d\mathbf{W}_u \right\|_{L^p} \\ &\stackrel{(i)}{\lesssim} \int_s^t \|f(u, \mathbf{X}_u)\|_{L^p} du + \left( \int_s^t \|\sigma(u, \mathbf{X}_u)\|_{L^p}^2 du \right)^{1/2} \\ &\stackrel{(ii)}{\lesssim} \|f(s, \mathbf{X}_s)\|_{L^p} (t-s) + \int_s^t \|f(u, \mathbf{X}_u) - f(s, \mathbf{X}_s)\|_{L^p} du \\ &\quad + \|\sigma(s, \mathbf{X}_s)\|_{L^p} (t-s)^{1/2} + \left( \int_s^t \|\sigma(u, \mathbf{X}_u) - \sigma(s, \mathbf{X}_s)\|_{L^p}^2 du \right)^{1/2} \\ &\stackrel{(iii)}{\lesssim} \left[ \|f(0, \mathbf{0})\| + K_f(s + \|\mathbf{X}_s\|_{L^p}) \right] (t-s) + K_f \frac{(t-s)^2}{2} + \int_s^t K_f \|\mathbf{X}_{s,u}\|_{L^p} du \\ &\quad + \left[ \|\sigma(0, \mathbf{0})\| + K_\sigma(s + \|\mathbf{X}_s\|_{L^p}) \right] (t-s)^{1/2} + K_\sigma \left( \int_s^t \|(u, \mathbf{X}_u) - (s, \mathbf{X}_s)\|_{L^p}^2 du \right)^{1/2} \\ &\stackrel{(iv)}{\lesssim} \left[ \|f(0, \mathbf{0})\| + K_f(s + \|\mathbf{X}_s\|_{L^p}) \right] (t-s) + K_f \frac{(t-s)^2}{2} + \int_s^t K_f \|\mathbf{X}_{s,u}\|_{L^p} du \\ &\quad + \left[ \|\sigma(0, \mathbf{0})\| + K_\sigma(s + \|\mathbf{X}_s\|_{L^p}) \right] (t-s)^{1/2} + K_\sigma \left( \int_s^t |u-s|^2 du \right)^{1/2} \\ &\quad + K_\sigma \left( \int_s^t \|\mathbf{X}_{s,u}\|_{L^p}^2 du \right)^{1/2} \\ &\stackrel{(v)}{\lesssim} (1 \vee \|\mathbf{X}_s\|_{L^p}) (t-s)^{1/2} + \int_s^t \|\mathbf{X}_{s,u}\|_{L^p} du + \left( \int_s^t \|\mathbf{X}_{s,u}\|_{L^p}^2 du \right)^{1/2} \\ &\stackrel{(vi)}{\lesssim} (1 \vee \|\mathbf{X}_s\|_{L^p}) (t-s)^{1/2} + \left( \int_s^t \|\mathbf{X}_{s,u}\|_{L^p}^2 du \right)^{1/2}, \end{aligned}$$

by using in (i) triangle inequality and Equations [\(48\)](#page-40-7) and [\(50\)](#page-40-8), in (ii) triangle inequality, in (iii) Lipschitzianity of f and <sup>σ</sup>, in (iv) triangle inequality, in (v) for all <sup>0</sup> ≤ <sup>s</sup> ≤ <sup>t</sup> ≤ <sup>T</sup>, <sup>s</sup> ≤ <sup>T</sup> <sup>≲</sup> <sup>1</sup> and (t−s) <sup>1</sup>/2+<sup>ϵ</sup> ≤ <sup>T</sup> ϵ (t−s) <sup>1</sup>/<sup>2</sup> <sup>≲</sup> (t−s) 1/2 for <sup>ϵ</sup> ≥ <sup>0</sup>, and in (vi) Jensen's inequality. Setting <sup>s</sup> = 0, since we assume <sup>X</sup><sup>0</sup> ∈ <sup>L</sup> p , this is

$$\|\mathbf{X}_{0,t}\|_{L^p} \lesssim t^{1/2} + \left( \int_0^t \|\mathbf{X}_{0,u}\|_{L^p}^2 du \right)^{1/2},$$

and we can apply [Willett](#page-10-12) [\(1964,](#page-10-12) Lemma 2.2), a nonlinear generalization of the Gronwall inequality, to deduce for all <sup>0</sup> ≤ <sup>t</sup> ≤ <sup>T</sup>,

$$\begin{aligned} \|\mathbf{X}_{0,t}\|_{L^p} &\lesssim t^{1/2} + \frac{\left(\int_0^t \exp\{-Cs\} s \, ds\right)^{1/2}}{1 - \sqrt{1 - \exp\{-Ct\}}} \\ &\lesssim T^{1/2} + \frac{T}{1 - \sqrt{1 - \exp\{-CT\}}} < \infty. \end{aligned}$$

We can hence show the condition for Ito processes <sup>ˆ</sup> [\(47\)](#page-40-6) holds by noting that <sup>X</sup><sup>0</sup> ∈ <sup>L</sup> p and Lipschitzianity of f and σ imply

$$\begin{aligned}\|f(t, \mathbf{X}_t)\|_{L^p} &\leq \|f(0, \mathbf{0})\| + K_f(|t| + \|\mathbf{X}_0\|_{L^p} + \|\mathbf{X}_{0,t}\|_{L^p}) < \infty, \\ \|\sigma(t, \mathbf{X}_t)\|_{L^p} &\leq \|\sigma(0, \mathbf{0})\| + K_\sigma(|t| + \|\mathbf{X}_0\|_{L^p} + \|\mathbf{X}_{0,t}\|_{L^p}) < \infty,\end{aligned}$$

uniformly in <sup>t</sup> ∈ [0, T], and hence [\(A](#page-3-0)α) and [\(A](#page-3-0)δ) hold with <sup>α</sup> = 1/<sup>2</sup> and <sup>δ</sup> = 1, respectively.

- Assume f and σ are time-homogeneous such that f is Lipschitz continuous and σ is 1/2-Holder continuous, i.e. for ¨ all <sup>x</sup>, <sup>y</sup> ∈ <sup>R</sup> d ,

$$\begin{aligned}\|f(\mathbf{x}) - f(\mathbf{y})\| &\leq K_f\|\mathbf{x} - \mathbf{y}\|, \\ \|\sigma(\mathbf{x}) - \sigma(\mathbf{y})\| &\leq K_\sigma\|\mathbf{x} - \mathbf{y}\|^{1/2}.\end{aligned}$$

Then, for all <sup>0</sup> ≤ <sup>s</sup> ≤ <sup>t</sup> ≤ <sup>T</sup>, we can bound

$$\begin{aligned} \|\mathbf{X}_{s,t}\|_{L^p} &= \left\| \int_s^t f(\mathbf{X}_u) du + \int_s^t \sigma(\mathbf{X}_u) d\mathbf{W}_u \right\|_{L^p} \\ &\stackrel{(i)}{\lesssim} \int_s^t \|f(\mathbf{X}_u)\|_{L^p} du + \left( \int_s^t \|\sigma(\mathbf{X}_u)\|_{L^p}^2 du \right)^{1/2} \\ &\stackrel{(ii)}{\lesssim} \|f(\mathbf{X}_s)\|_{L^p} (t-s) + \int_s^t \|f(\mathbf{X}_u) - f(\mathbf{X}_s)\|_{L^p} du \\ &\quad + \|\sigma(\mathbf{X}_s)\|_{L^p} (t-s)^{1/2} + \left( \int_s^t \|\sigma(\mathbf{X}_u) - \sigma(\mathbf{X}_s)\|_{L^p}^2 du \right)^{1/2} \\ &\stackrel{(iii)}{\lesssim} \left[ \|f(\mathbf{0})\| + K_f \|\mathbf{X}_s\|_{L^p} \right] (t-s) + \int_s^t K_f \|\mathbf{X}_{s,u}\|_{L^p} du \\ &\quad + \left[ \|\sigma(\mathbf{0})\| + K_\sigma \|\mathbf{X}_s\|_{L^p}^{1/2} \right] (t-s)^{1/2} + K_\sigma^{1/2} \left( \int_s^t \|\mathbf{X}_{s,u}\|_{L^p} du \right)^{1/2} \\ &\stackrel{(iv)}{\lesssim} \left( 1 \vee \|\mathbf{X}_s\|_{L^p} \vee \|\mathbf{X}_s\|_{L^p}^{1/2} \right) (t-s)^{1/2} + \int_s^t \|\mathbf{X}_{s,u}\|_{L^p}^2 du + \left( \int_s^t \|\mathbf{X}_{s,u}\|_{L^p} du \right)^{1/2}, \end{aligned}$$

by proceeding as in the previous case. Setting <sup>s</sup> = 0, since we assume <sup>X</sup><sup>0</sup> ∈ <sup>L</sup> p , this is

$$\|\mathbf{X}_{0,t}\|_{L^p} \lesssim t^{1/2} + \int_0^t \|\mathbf{X}_{0,u}\|_{L^p} \, du + \left( \int_0^t \|\mathbf{X}_{0,u}\|_{L^p} \, du \right)^{1/2},$$

and we can apply [Dragomir](#page-9-18) [\(2003,](#page-9-18) Theorem 41), another nonlinear generalization of the Gronwall inequality, to deduce for all <sup>0</sup> ≤ <sup>t</sup> ≤ <sup>T</sup>,

$$\|\mathbf{X}_{0,t}\|_{L^p} \lesssim f(t) < \infty.$$

We can hence show the condition for Ito processes <sup>ˆ</sup> [\(47\)](#page-40-6) holds by noting that <sup>X</sup><sup>0</sup> ∈ <sup>L</sup> p and the conditions on f and σ imply

$$\begin{aligned}\|f(\mathbf{X}_t)\|_{L^p} &\leq \|f(\mathbf{0})\| + K_f(\|\mathbf{X}_0\|_{L^p} + \|\mathbf{X}_{0,t}\|_{L^p}) < \infty, \\ \|\sigma(\mathbf{X}_t)\|_{L^p} &\leq \|\sigma(\mathbf{0})\| + K_\sigma(\|\mathbf{X}_0\|_{L^p} + \|\mathbf{X}_{0,t}\|_{L^p})^{1/2} < \infty,\end{aligned}$$

uniformly in <sup>t</sup> ∈ [0, T], and hence [\(A](#page-3-0)α) and [\(A](#page-3-0)δ) hold with <sup>α</sup> = 1/<sup>2</sup> and <sup>δ</sup> = 1 respectively.

## D.2. Long span conditions

## D.2.1. ITO DIFFUSIONS ˆ

When developing conditions ensuring stationarity and ergodicity of an Ito diffusion it is natural to restrict <sup>ˆ</sup> {<sup>X</sup>t, t ≥ <sup>0</sup>} to the case where it is time-homogeneous, i.e. satisfies

$$d\mathbf{X}_t = f(\mathbf{X}_t)dt + \sigma(\mathbf{X}_t)d\mathbf{W}_t, \quad t \geq 0,$$

where <sup>W</sup> <sup>=</sup> {<sup>W</sup>t, t ≥ <sup>0</sup>} is a <sup>q</sup>-dimensional Brownian motion, <sup>f</sup> : <sup>R</sup> <sup>d</sup> → <sup>R</sup> d and σ : R <sup>d</sup> → <sup>R</sup> d×q . Assume:

- The diffusion coefficient σ : R <sup>d</sup> → <sup>R</sup> d×q is Lipschitz continuous and Σ := σσ<sup>T</sup> : <sup>R</sup> <sup>d</sup> → <sup>R</sup> d×d is bounded and uniformly elliptic, i.e.

$$\inf_{x \in \mathbb{R}^d, \xi \in \mathbb{R}^d \setminus \{0\}} \frac{\langle \xi, \Sigma(x)\xi \rangle}{\|\xi\|^2} > 0.$$

This is a classic PDE condition which ensures the transition densities are "nice", i.e. continuous and bounded away from zero [\(Friedman,](#page-9-19) [1964\)](#page-9-19).

- The drift f : R <sup>d</sup> → <sup>R</sup> d is Lipschitz continuous and has negative radial part at ∞, i.e.

$$\limsup_{\|x\| \rightarrow \infty} \left\langle f(x), \frac{x}{\|x\|^{\kappa+1}} \right\rangle =: -C_\kappa \in [-\infty, 0),$$

pushing the process towards the origin with strength[<sup>21</sup>](#page-0-0) controlled by <sup>κ</sup> ∈ [−1, ∞). When <sup>κ</sup> <sup>=</sup> −<sup>1</sup> assume further that 2C−<sup>1</sup> > supx∈R<sup>d</sup> Tr Σ(x) and define

$$\eta_{f,\Sigma}^* := \begin{cases} \infty, & \text{if } \kappa > 0, \\ 2C_0/\|\Sigma\|, & \text{if } \kappa = 0, \\ 2C_\kappa/((1 + \kappa)\|\Sigma\|), & \text{if } \kappa \in (-1, 0), \\ (2C_{-1} - \sup_{x \in \mathbb{R}^d} \text{Tr } \Sigma(x)) / \|\Sigma\|, & \text{if } \kappa = -1, \end{cases}$$

where |||Σ||| := supx∈R<sup>d</sup> ∥Σ(x)∥.

These conditions are enough to ensure there exists a unique invariant probability measure µ on R <sup>d</sup> with

$$\begin{cases} \int_{\mathbb{R}^d} e^{\eta \|x\|^2} \mu(dx) < \infty, & \text{if } \kappa \geq 0, \\ \int_{\mathbb{R}^d} e^{\eta \|x\|^{1+\kappa}} \mu(dx) < \infty, & \text{if } \kappa \in (-1, 0), \\ \int_{\mathbb{R}^d} \|x\|^n \mu(dx) < \infty, & \text{if } \kappa = -1, \end{cases}$$

<sup>21</sup>Note that, for large <sup>∥</sup>x∥, one has D f(x), x ∥x∥ E ≈ −Cκ∥x∥ κ , and hence the strength of the pull grows as ∥x∥ increases when κ > 0 and decays as ∥x∥ increases when κ < 0.

for all <sup>η</sup> ∈ (0, η<sup>∗</sup> f,Σ), such that for any <sup>x</sup> ∈ <sup>R</sup> d the transition probabilities[<sup>22</sup>](#page-0-0) {<sup>P</sup>t(x, ·), t ≥ <sup>0</sup>} converge to <sup>µ</sup> in total variation distance with rates

$$\|P_t(x, \cdot) - \mu\|_{\text{TV}} \leq \begin{cases} c_1 e^{-c_2 t} (e^{\eta \|x\|} + c_3), & \text{if } \kappa \geq 0, \\ c_1 e^{-c_2 t^{(1+\kappa)/(1-\kappa)}} (e^{\eta \|x\|^{1+\kappa}} + c_3), & \text{if } \kappa \in (-1, 0), \\ c_1 (1 + c_2 t)^{-\eta/2} (\|x\|^\eta + c_3), & \text{if } \kappa = -1, \end{cases}$$

with c1, c2, c<sup>3</sup> > 0 [\(Kulik,](#page-9-20) [2018,](#page-9-20) Theorem 3.3.4, 3.3.5 and 3.3.6).

Assuming <sup>X</sup><sup>0</sup> ∼ <sup>µ</sup>, the Ito diffusion <sup>ˆ</sup> {<sup>X</sup>t, t ≥ <sup>0</sup>} defines a stationary Markov process with <sup>X</sup><sup>0</sup> ∈ <sup>L</sup> p for all <sup>p</sup> ≥ <sup>2</sup> when κ > −<sup>1</sup> and for all <sup>2</sup> ≤ p < η<sup>∗</sup> f,<sup>Σ</sup> when <sup>κ</sup> <sup>=</sup> −<sup>1</sup>. Recall that, by the discussion in Appendix [D.1.2,](#page-41-0) when <sup>f</sup> and <sup>σ</sup> are Lipschitz continuous, it is enough to have <sup>X</sup><sup>0</sup> ∈ <sup>L</sup> p to ensure the process <sup>X</sup> <sup>=</sup> {<sup>X</sup>t, t ∈ [0, T]} satisfies Assumptions [\(A](#page-3-0)α) and [\(A](#page-3-0)δ) with <sup>α</sup> = 1/<sup>2</sup> and <sup>δ</sup> = 1, implying <sup>ϵ</sup> = 1/2. By Proposition [2.13](#page-4-0) the chain {<sup>X</sup> <sup>n</sup>, n ≥ <sup>1</sup>} is stationary and, hence, it remains to establish strong mixing of {<sup>X</sup> <sup>n</sup>, n ≥ <sup>1</sup>} to apply Theorem [2.10.](#page-4-1) The strong mixing coefficient of the stationary Markov process {<sup>X</sup>t, t ≥ <sup>0</sup>} can be easily[<sup>23</sup>](#page-0-0) bounded by

$$\alpha(t) \leq \int_{\mathbb{R}^d} \|P_t(x, \cdot) - \mu\|_{\text{TV}} \mu(dx) \lesssim \begin{cases} e^{-c_2 t}, & \text{if } \kappa \geq 0, \\ e^{-c_2 t^{(1+\kappa)/(1-\kappa)}}, & \text{if } \kappa \in (-1, 0), \\ (1 + c_2 t)^{-\eta/2}, & \text{if } \kappa = -1, \end{cases}$$

and hence the process is strongly mixing. By Proposition [2.13](#page-4-0) the chain {<sup>X</sup> <sup>n</sup>, n ≥ <sup>1</sup>} is also strongly mixing with coefficient α ′′(n) ≤ <sup>α</sup>((n−3)T), <sup>n</sup> ≥ <sup>3</sup>. It follows immediately that {<sup>X</sup> <sup>n</sup>, n ≥ <sup>1</sup>} is ergodic and hence, we can apply Theorem [2.10](#page-4-1).<sup>1</sup> to deduce that, letting |Π(N)| → <sup>0</sup> as <sup>N</sup> → ∞ the expected signature estimator [\(7\)](#page-3-2) is consistent for any expected signature term when κ > −<sup>1</sup> and for all expected signature terms with |I| <sup>&</sup>lt; 2 η ∗ f,<sup>Σ</sup> when <sup>κ</sup> <sup>=</sup> −<sup>1</sup>.

Finally, we note that for any ζ > 0

$$\sum_{n \geq 0} \alpha''(n)^{\zeta/(2+\zeta)} \lesssim \begin{cases} \sum_{n \geq 1} e^{-c_2 n T \zeta/(2+\zeta)} < \infty, & \text{if } \kappa \geq 0, \\ \sum_{n \geq 1} e^{-c_2(nT)^{(1+\kappa)/(1-\kappa)}} \zeta/(2+\zeta) < \infty, & \text{if } \kappa \in (-1, 0), \\ \sum_{n \geq 1} (1 + c_2 n T)^{-\eta \zeta/(4+2\zeta)} & = \infty, & \text{if } \kappa = -1, \end{cases}$$

and hence, if κ > −<sup>1</sup> and Π(N) is a sequence of expanding dyadic refinements, we can apply Theorem [2.10](#page-4-1).<sup>2</sup> to show that the expected signature estimator [\(7\)](#page-3-2) is also asymptotically normal.

$$P_t(x, A) := \mathbb{P}(\mathbf{X}_t \in A | \mathbf{X}_0 = x) = \mathbb{P}(\mathbf{X}_{s+t} \in A | \mathbf{X}_s = x),$$

$$\begin{aligned}
|\mathbb{P}(A \cap B) - \mathbb{P}(A)\mathbb{P}(B)| &= |\mathbb{E}[\mathbb{1}_A \mathbb{1}_B] - \mathbb{E}[\mathbb{1}_A]\mathbb{E}[\mathbb{1}_B]| \\
&= |\mathbb{E}[\mathbb{1}_A \mathbb{E}[\mathbb{E}[\mathbb{1}_B | \mathcal{F}_{-\infty}^{t+s}] | \mathcal{F}_{-\infty}^t]] - \mathbb{E}[\mathbb{1}_A \mathbb{E}[\mathbb{1}_B]]| \\
&= \left| \mathbb{E} \left[ \mathbb{1}_A \left( \int_{\mathbb{R}^d} \mathbb{E}[\mathbb{1}_B | \mathbf{X}_{t+s}] | \mathbf{X}_t \right] - \mathbb{E}[\mathbb{E}[\mathbb{1}_B | \mathbf{X}_{t+s}]) \right) \right] \right| \\
&= \left| \mathbb{E} \left[ \mathbb{1}_A \left( \int_{\mathbb{R}^d} h_B(x) P_s(\mathbf{X}_t, dx) - \int_{\mathbb{R}^d} h_B(x) \mu(dx) \right) \right) \right] \\
&\leq \mathbb{E} \left[ \mathbb{1}_A \left| \int_{\mathbb{R}^d} h_B(x) (P_s(\mathbf{X}_t, dx) - \mu(dx)) \right| \right] \\
&\leq \mathbb{E} [\|P_s(\mathbf{X}_t, \cdot) - \mu\|_{\text{TV}}] \\
&= \int_{\mathbb{R}^d} \|P_s(x, \cdot) - \mu\|_{\text{TV}} \mu(dx),
\end{aligned}$$

<sup>22</sup>For the time-homogeneous Ito diffusion ˆ {Xt, t ≥ 0} these are defined by

for all t, s ≥ 0, x ∈ R d , A ∈ B(<sup>R</sup> d ).

<sup>23</sup>If {Xt, t ≥ 0}} is a stationary Markov process with stationary distribution µ and A ∈ F<sup>t</sup> −∞, B ∈ F<sup>∞</sup>t+<sup>s</sup> for t, s ≥ 0,

#### E. Gaussian Processes

We first exploit the properties of Gaussian random variables to show that, if a Gaussian process satisfies [\(A](#page-3-0)α) for p = 2 and some α > <sup>1</sup>/4, then it satisfies [\(A](#page-3-0)α) for any <sup>p</sup> ≥ <sup>2</sup> and the same <sup>α</sup>. Next, we show that two common examples of Gaussian processes, Ornstein-Uhlenbeck processes and fractional Brownian motion, satisfy the assumptions of Theorem [2.14,](#page-4-2) i.e. [\(A](#page-3-0)α) with <sup>p</sup> = 2 and <sup>α</sup> ≥ <sup>1</sup>/2, [\(A](#page-3-0)δ) when <sup>α</sup> = 1/<sup>2</sup> and [\(A](#page-4-2)θ) for some decreasing <sup>θ</sup> : <sup>R</sup><sup>+</sup> → <sup>R</sup><sup>+</sup> with <sup>θ</sup>(t) → <sup>0</sup>, t → ∞ and R <sup>T</sup> 0 <sup>θ</sup>(t)dt < ∞ and <sup>m</sup> ∈ <sup>N</sup>.

#### E.1. Gaussian Processes Continuity Criterion

Let <sup>X</sup> be a Gaussian process with mean function <sup>µ</sup> : [0, T] → <sup>R</sup> d . If <sup>X</sup> satisfies [\(A](#page-3-0)α) with p = 2 and α > 1/4 and µ is <sup>α</sup>-Holder continuous, then it satisfies ¨ [\(A](#page-3-0)α) for any <sup>p</sup> ≥ <sup>2</sup> and exponent <sup>α</sup>. Note that, by the inclusion of norms in <sup>L</sup> p spaces, it suffices to show this holds for arbitrarily large p. Choosing p = 2q even, we can write

$$\begin{aligned} \|\mathbf{X}_{s,t}\|_{L^{2q}} &\leq \|\mathbf{X}_{s,t} - \mu_{s,t}\|_{L^{2q}} + \|\mu_{s,t}\| \\ &\lesssim \left( \mathbb{E} \left[ \left( \sum_{i=1}^d |X_{s,t}^{(i)} - \mu_{s,t}^{(i)}|^2 \right)^q \right] \right)^{1/2q} + |t - s|^\alpha \\ &= \left( \sum_{\substack{q_1+\dots+q_d=q \\ q_1,\dots,q_d \geq 0}} \binom{q}{q_1,\dots,q_d} \mathbb{E} \left[ \left| X_{s,t}^{(1)} - \mu_{s,t}^{(1)} \right|^{2q_1} \dots \left| X_{s,t}^{(d)} - \mu_{s,t}^{(d)} \right|^{2q_d} \right] \right)^{1/2q} + |t - s|^\alpha \\ &\stackrel{(i)}{=} \left( \sum_{\substack{q_1+\dots+q_d=q \\ q_1,\dots,q_d \geq 0}} \binom{q}{q_1,\dots,q_d} \sum_{p \in P_{2q_1,\dots,2q_d}^2} \prod_{\{i,j\} \in p} \text{Cov} \left( X_{s,t}^{(i)}, X_{s,t}^{(j)} \right) \right)^{1/2q} + |t - s|^\alpha \\ &\stackrel{(ii)}{\lesssim} \left( \sum_{\substack{q_1+\dots+q_d=q \\ q_1,\dots,q_d \geq 0}} \binom{q}{q_1,\dots,q_d} \sum_{p \in P_{2q_1,\dots,2m_k}^2} \prod_{\{i,j\} \in p} |t - s|^{2\alpha} \right)^{1/2q} \lesssim |t - s|^\alpha, \end{aligned}$$

where in (i) we apply Isserlis' theorem [\(Isserlis,](#page-9-17) [1918\)](#page-9-17) denoting by P 2 2q1,...,2q<sup>d</sup> the set of all the pairings of <sup>S</sup> <sup>=</sup> {1} <sup>2</sup>q<sup>1</sup> ∪ {2} <sup>2</sup>q<sup>2</sup> ∪· · ·∪ {d} <sup>2</sup>q<sup>d</sup> , i.e. all distinct ways of partitioning S into q<sup>1</sup> +. . .+q<sup>d</sup> = q pairs, and in (ii) the fact that Assumption [\(A](#page-3-0)α) with p = 2 and α > 1/4 implies for all i, j = 1, . . . , d,

$$\begin{aligned} \text{Cov}(X_{s,t}^{(i)}, X_{s,t}^{(j)}) &\leq |\text{Cov}(X_{s,t}^{(i)}, X_{s,t}^{(j)})| \\ &\leq |\mathbb{E}[X_{s,t}^{(i)} X_{s,t}^{(j)}]| + |\mu_{s,t}^{(i)} \mu_{s,t}^{(j)}| \\ &\leq \mathbb{E}[|X_{s,t}^{(i)} X_{s,t}^{(j)}|] + |\mu_{s,t}^{(i)}| |\mu_{s,t}^{(j)}| \\ &\leq \mathbb{E}[|X_{s,t}^{(i)}|^2]^{1/2} \mathbb{E}[|X_{s,t}^{(j)}|^2]^{1/2} + |\mu_{s,t}^{(i)}| |\mu_{s,t}^{(j)}| \\ &\leq \|\mathbf{X}_{s,t}\|^2 + \|\mu_{s,t}\|^2 \lesssim |t - s|^{2\alpha}. \end{aligned}$$

#### E.2. Gaussian Processes Covariance Decay Condition

#### E.2.1. ORNSTEIN-UHLENBECK PROCESS

If {<sup>X</sup>t, t ≥ <sup>0</sup>} is a stationary mean-zero <sup>d</sup>-dimensional Ornstein-Uhlenbeck process, i.e. a mean-zero Gaussian process with covariance

$$C(s, t) := \text{Cov}(\mathbf{X}_s, \mathbf{X}_t) = e^{-A|t-s|} \Sigma,$$

where Σ = Var(Xt) and the drift matrix parameter <sup>A</sup> ∈ <sup>R</sup> <sup>d</sup>×<sup>d</sup> has positive real parts of all eigenvalues, then we can show that:

- [\(A](#page-3-0)α) holds with <sup>α</sup> = 1/<sup>2</sup> and <sup>p</sup> = 2. Note that for all <sup>0</sup> ≤ <sup>s</sup> ≤ <sup>t</sup> ≤ <sup>T</sup>,

$$\|\mathbf{X}_{s,t}\|_{L^2}^2 = \mathbb{E}[\text{tr}(\mathbf{X}_{s,t} \otimes \mathbf{X}_{s,t})]$$

$$\begin{aligned} &= \text{tr}(\text{Var}(\mathbf{X}_t) + \text{Var}(\mathbf{X}_s) - 2\text{Cov}(\mathbf{X}_s, \mathbf{X}_t)) \\ &= 2 \text{tr}((I_d - e^{-A|t-s|})\Sigma) \\ &\leq 2 \|I_d - e^{-A|t-s|}\| \|\Sigma\| \\ &\leq 2 (\|A\| \|t - s\| e^{\|A\| \|t-s\|}) \|\Sigma\| \\ &\lesssim |t - s|. \end{aligned}$$

- [\(A](#page-3-0)δ) holds for any <sup>p</sup> ≥ <sup>2</sup> with <sup>δ</sup> = 1. Using the integral representation of the OU process, one can easily verify that for all <sup>0</sup> ≤ <sup>s</sup> ≤ <sup>t</sup> ≤ <sup>T</sup>, <sup>E</sup>s[Xt] := <sup>E</sup>[Xt|Fs] = <sup>e</sup> <sup>−</sup>A|t−s|Xs, where {Ft, t ∈ [0, T]} is the natural filtration of <sup>X</sup>. Then, for all <sup>0</sup> ≤ <sup>s</sup> ≤ <sup>t</sup> ≤ <sup>T</sup>,

$$\|\mathbb{E}_s[\mathbf{X}_{s,t}]\|_{L^p} = \|(e^{-A|t-s|} - I_d)\mathbf{X}_s\|_{L^p} \leq \|I_d - e^{-A|t-u|}\| \|\mathbf{X}_s\|_{L^p} \lesssim |t-s|.$$

- The covariance of the increments is homogeneous since, for all u, v, s, t ≥ <sup>0</sup>,

$$\begin{aligned} \text{Cov}(\mathbf{X}_{u,v}, \mathbf{X}_{s,t}) &= \text{Cov}(\mathbf{X}_v, \mathbf{X}_t) - \text{Cov}(\mathbf{X}_v, \mathbf{X}_s) - \text{Cov}(\mathbf{X}_u, \mathbf{X}_t) + \text{Cov}(\mathbf{X}_u, \mathbf{X}_s) \\ &= (e^{-A|t-v|} - e^{-A|s-v|} - e^{-A|t-u|} + e^{-A|s-u|}) \Sigma, \end{aligned}$$

depends only on the relative distances |<sup>t</sup> − <sup>v</sup>|, |<sup>s</sup> − <sup>v</sup>|, |<sup>t</sup> − <sup>u</sup>| and |<sup>s</sup> − <sup>u</sup>|.

- The covariance of the increments satisfies Assumption [\(A](#page-4-2)θ) with m = 0 and θ(t) = e <sup>−</sup>λA<sup>t</sup> where λ<sup>A</sup> is a constant depending on the drift matrix <sup>A</sup> ∈ <sup>R</sup> d×d . For all <sup>0</sup> ≤ <sup>u</sup> ≤ v < s ≤ <sup>t</sup>,

$$\begin{aligned} \|\text{Cov}(\mathbf{X}_{u,v}, \mathbf{X}_{s,t})\| &= \|e^{-A|s-v|}(e^{-A|t-s|} - I_d - e^{-A|t-s|-A|v-u|} + e^{-A|v-u|})\Sigma\| \\ &= \|e^{-A|s-v|}(e^{-A|t-s|} - I_d)(I_d - e^{-A|v-u|})\Sigma\| \\ &= \|e^{-A|s-v|}\| \|I_d - e^{-A|t-s|}\| \|I_d - e^{-A|v-u|}\| \|\Sigma\| \\ &\lesssim e^{-\lambda_A|s-v|}|t-s||v-u|, \end{aligned}$$

where, in the last step, we use the fact that <sup>A</sup> ∈ <sup>R</sup> <sup>d</sup>×<sup>d</sup> has positive real parts of all eigenvalues to find <sup>λ</sup><sup>A</sup> ∈ (0, minλ∈σ(A) Re(λ)) such that ∥<sup>e</sup> <sup>−</sup>At∥ <sup>≲</sup> <sup>e</sup> −λAt for all <sup>t</sup> ≥ <sup>0</sup>.

Note that [\(A](#page-3-0)α) and [\(A](#page-3-0)δ) could have been alternatively established by noting that the OU process is an Ito diffusion with ˆ Lipschitz continuous coefficients and applying the results of Appendix [D.1.2.](#page-41-0)

## E.2.2. FRACTIONAL BROWNIAN MOTION

If {X<sup>H</sup> t , t ≥ <sup>0</sup>} is a (one-dimensional) fractional Brownian motion with Hurst parameter H > <sup>1</sup>/2, i.e. a mean-zero Gaussian process with covariance

$$C^H(s, t) := \text{Cov}(X_s^H, X_t^H) = \frac{1}{2}(|t|^{2H} + |s|^{2H} - |t - s|^{2H}),$$

then we can show that:

- [\(A](#page-3-0)α) holds with <sup>α</sup> <sup>=</sup> H > <sup>1</sup>/<sup>2</sup> (and <sup>p</sup> = 2) since ∥X<sup>H</sup> s,t∥L<sup>2</sup> <sup>=</sup> |<sup>t</sup> − <sup>s</sup>|
- H.
- The covariance of the increments is homogeneous since for all u, v, s, t ≥ <sup>0</sup>,

$$\begin{aligned} \text{Cov}(X_{u,v}^H, X_{s,t}^H) &= \text{Cov}(X_{v,t}^H, X_{s,t}^H) - \text{Cov}(X_{v,s}^H, X_{s,t}^H) - \text{Cov}(X_{u,t}^H, X_{t,t}^H) + \text{Cov}(X_{u,s}^H, X_{s,t}^H) \\ &= \frac{1}{2}(|s - v|^{2H} + |t - u|^{2H} - |t - v|^{2H} - |s - u|^{2H}), \end{aligned}$$

- The covariance of the increments satisfies Assumption [\(A](#page-4-2)θ) with m = 3 and θ(t) = t 2H−2 . For all <sup>0</sup> ≤ <sup>u</sup> ≤ v < s ≤ <sup>t</sup> with |<sup>s</sup> − <sup>v</sup>| ≥ <sup>3</sup>/2(|<sup>t</sup> − <sup>s</sup>| <sup>+</sup> |<sup>v</sup> − <sup>u</sup>|),

$$\begin{aligned}
& |\text{Cov} (X_{u,v}^H, X_{s,t}^H)| \\
&= \frac{1}{2} ||s - v|^{2H} + |t - u|^{2H} - |t - v|^{2H} - |s - u|^{2H}| \\
&= \frac{1}{2} ||s - v|^{2H} \\
&\quad + \sum_{n=0}^{\infty} \frac{(2H) \cdots (2H - n + 1)}{n!} |s - v|^{2H-n} (|t - u| - |s - v|)^n \\
&\quad - \sum_{n=0}^{\infty} \frac{(2H) \cdots (2H - n + 1)}{n!} |s - v|^{2H-n} (|t - v| - |s - v|)^n \\
&\quad - \sum_{n=0}^{\infty} \frac{(2H) \cdots (2H - n + 1)}{n!} |s - v|^{2H-n} (|s - u| - |s - v|)^n \Big| \\
&= \frac{1}{2} \left| \sum_{n=2}^{\infty} \frac{(2H) \cdots (2H - n + 1)}{n!} |s - v|^{2H-n} \left[ (|t - s| + |v - u|)^n - |t - s|^n - |v - u|^n \right] \right| \\
&= \frac{1}{2} |s - v|^{2H-2} \left| \sum_{n=2}^{\infty} \frac{(2H) \cdots (2H - n + 1)}{n!} |s - v|^{-(n-2)} \sum_{j=1}^{n-1} \binom{n}{j} |t - s|^j |v - u|^{n-j} \right| \\
&\leq \frac{1}{2} |s - v|^{2H-2} |t - s| |v - u| \\
&\quad \times \sum_{n=2}^{\infty} \frac{|(2H) \cdots (2H - n + 1)|}{(n - 2)!} |s - v|^{-(n-2)} \sum_{j=0}^{n-2} \binom{n - 2}{j} |t - s|^j |v - u|^{(n-2)-j} \\
&\leq \frac{1}{2} |s - v|^{2H-2} |t - s| |v - u| |2H| |2H - 1| \sum_{n=0}^{\infty} |s - v|^{-n} (|t - s| + |v - u|)^n \\
&\leq \frac{1}{2} |s - v|^{2H-2} |t - s| |v - u| |2H| |2H - 1| \left( 1 - \frac{|t - s| + |v - u|}{|s - v|} \right)^{-1} \\
&\leq |2H| |2H - 1| |s - v|^{2H-2} |t - s| |v - u|,
\end{aligned}$$

by using the fact that that <sup>x</sup> 7→ <sup>f</sup>(x) = <sup>x</sup> <sup>2</sup><sup>H</sup> is analytic for any x > <sup>0</sup> and |<sup>s</sup> − <sup>v</sup>| ≥ <sup>3</sup>/2(|<sup>t</sup> − <sup>s</sup>| <sup>+</sup> |<sup>v</sup> − <sup>u</sup>|).

## F. Machine Learning Algorithms with Expected Signatures

Signatures of paths have found widespread use in the machine learning community, with applications ranging from character recognition [\(Graham,](#page-9-21) [2013;](#page-9-21) [Xie et al.,](#page-10-13) [2018\)](#page-10-13) to medical diagnosis [\(Perez Arribas et al.](#page-10-14) ´ , [2018\)](#page-10-14). Taking the signature of a stream of data is essentially a feature extraction method mapping raw stream-like data to a lower-dimensional but highly-informative latent space. The theoretical foundations for their efficacy range from the characterization result of [Hambly & Lyons](#page-9-1) [\(2005\)](#page-9-1) to the universal approximation theorem [\(Levin et al.,](#page-10-2) [2016,](#page-10-2) Theorem 3.1). As discussed in the introduction, when dealing with collections of paths, the characterization results of [Fawcett](#page-9-2) [\(2003\)](#page-9-2), [Chevyrev & Lyons](#page-9-3) [\(2016\)](#page-9-3) and [Chevyrev & Oberhauser](#page-9-4) [\(2018\)](#page-9-4) give strong theoretical justification for the use of the expected signature as a feature extraction method.

While dealing with a collection of paths is arguably a less common setting than a single stream of data, the literature still provides a wide range of machine learning algorithms leveraging expected signatures[<sup>24</sup>](#page-0-0). These cover many different tasks (from distributional regression to generative modeling) and applications (from ECG classification to option pricing). In this section, we review five of these machine learning algorithms discussing how the martingale correction introduced in Section [2.2](#page-5-2) can be applied to the expected signature computation step to improve performance. Before diving into each

<sup>24</sup>The GPES algorithms, discussed in Section [3.2.1,](#page-7-2) actually takes as input a single stream of data and applies a data augmentation technique to form a collection of paths.

specific application in detail, we discuss some general considerations on the use of the martingale correction term in practice.

#### F.1. Martingale Correction in Applications

In Section [2.2,](#page-5-2) we considered the same framework as in the rest of this work, namely the setting where X is a continuous-time stochastic process and X π is a piecewise-linear interpolation of the discrete-time observation of such process along the partition π. It is important to note that, while some applications have a natural underlying latent continuous-time model [\(Lyons et al.,](#page-10-5) [2021\)](#page-10-5) others do not [\(Lemercier et al.,](#page-9-5) [2021\)](#page-9-5). In any case, we do not necessarily require the "background" continuous-time model X to be defined to apply the control-variate estimator [\(40\)](#page-33-2) with c = ˆc ∗ <sup>1</sup>,π. Letting <sup>π</sup> <sup>=</sup> {0 = <sup>t</sup><sup>0</sup> <sup>&</sup>lt; <sup>t</sup><sup>1</sup> <sup>&</sup>lt; · · · < t<sup>M</sup> <sup>=</sup> <sup>T</sup>}, one can easily see that it is sufficient for the discrete-time process {<sup>X</sup><sup>m</sup> <sup>=</sup> <sup>X</sup><sup>t</sup>m, m = 0, . . . , M} to be a discrete-time martingale with respect to F<sup>m</sup> <sup>=</sup> <sup>σ</sup>(X1, . . . , <sup>X</sup>m) for the control variate estimator to have the same bias but lower variance than the naive expected signature estimator. In what follows all path observations are inevitably sampled at discrete points in time and hence, abusing notation slightly, in some places we drop the dependence on the partition π. Whether X is a continuous-time process or a sequence of observations in discrete time should be clear from the context.

Machine learning methods based on signature methods often apply augmentations to the raw streams of data before computing the signature, cf. [Lyons & McLeod](#page-10-1) [\(2024,](#page-10-1) Section 2.5). For example, a path augmentation which is often found to improve model performance is the lead-lag transform. Combining the previous observation on discrete-time martingales with Remark [2.15,](#page-5-3) we can easily see that the control variate expected signature estimator can also be employed when the lead-lag augmentation is applied to the raw data, i.e. when the <sup>d</sup>-dimensional discrete-time martingale {<sup>X</sup>m, m = 1, . . . , M} is embedded into the 2d-dimensional process

$$\mathbb{X}' = \{(\mathbf{X}_1, \mathbf{X}_1), (\mathbf{X}_2, \mathbf{X}_1), (\mathbf{X}_2, \mathbf{X}_2), \dots, (\mathbf{X}_M, \mathbf{X}_{M-1}), (\mathbf{X}_M, \mathbf{X}_M)\} = \{\mathbf{X}'_{m'}, m' = 2, \dots, 2M + 2\}.$$

Note that for each m′ = 2, . . . , 2M + 2, if m′ = 2m then X′ <sup>m</sup>′ = (Xm, <sup>X</sup>m), <sup>X</sup>′ <sup>m</sup>′+1 = (Xm+1, Xm) and

$$\mathbb{E}[\mathbf{X}'_{m'+1} | \mathbf{X}'_1, \dots, \mathbf{X}'_{m'}] = (\mathbb{E}[\mathbf{X}_{m+1} | \mathbf{X}_1, \dots, \mathbf{X}_m], \mathbf{X}_m) = (\mathbf{X}_m, \mathbf{X}_m),$$

and if m′ = 2m + 1 then X′ <sup>m</sup>′ = (Xm+1, <sup>X</sup>m), <sup>X</sup>′ <sup>m</sup>′+1 = (Xm+1, Xm+1) and

$$\mathbb{E}[\mathbf{X}'_{m'+1} | \mathbf{X}'_1, \dots, \mathbf{X}'_{m'}] = (\mathbf{X}_{m+1}, \mathbf{X}_{m+1}).$$

and hence the leading components, i.e. the first d entries of X ′ , form a discrete-time martingale with respect to the natural filtration of X ′ . By Remark [2.15,](#page-5-3) we can hence apply the control variate expected signature estimator [\(40\)](#page-33-2) for any word <sup>I</sup> = (i1, . . . , ik) ∈ {1, . . . , <sup>2</sup>d} <sup>k</sup> with <sup>i</sup><sup>k</sup> ∈ {1, . . . , d}.

Finally, in some applications, we may not have a strong prior on whether the process being modeled is a martingale or not. In this case, we may consider the martingale correction as a model configuration hyperparameter to be tuned, just like the lead-lag path augmentation discussed above. In the model training phase we can then apply a cross-validation procedure to learn whether applying the martingale correction to (some of) the expected signature terms improves the performance of the model.

*Remark* F.1*.* Both the signature transform and the expected signature transform are general methods applicable to any machine learning task dealing with (collections of) streams of data. These can thus always be used as out-of-the-box feature extraction methods when little domain knowledge is available. On the other hand, when task-specific information is known, incorporating such knowledge in the machine learning model will most likely improve performance.

*Remark* F.2*.* It is important to note there is also a wide range of machine learning methods based on signature kernels [\(Kiraly & Oberhauser,](#page-9-22) [2019;](#page-9-22) [Chevyrev & Oberhauser,](#page-9-4) [2018;](#page-9-4) [Lemercier et al.,](#page-9-5) [2021;](#page-9-5) [Salvi et al.,](#page-10-15) [2021\)](#page-10-15). This bypasses the need to explicitly estimate the expected signature and hence we cannot directly apply the martingale correction developed in Section [2.2.](#page-5-2)

#### F.2. Algorithms

#### F.2.1. TIME SERIES CLASSIFICATION (T[RIGGIANO](#page-10-3) & ROMITO, [2024\)](#page-10-3)

In the Gaussian Process augmented Expected Signature (GPES) classifier, the input stream <sup>x</sup> ∈ <sup>R</sup> d×M<sup>1</sup> is interpreted as a discrete-time realization of a Gaussian process <sup>X</sup> ∼ GP(µ(t), Σ(t)) at points <sup>π</sup><sup>1</sup> <sup>=</sup> {0 = <sup>t</sup><sup>1</sup> < . . . < t<sup>M</sup><sup>1</sup> <sup>=</sup> <sup>T</sup>}, i.e. a realization of X π . Values of the process over a fixed set of in-fill points <sup>π</sup><sup>2</sup> <sup>=</sup> {<sup>s</sup><sup>1</sup> < . . . < s<sup>M</sup><sup>2</sup> } can thus be

sampled[<sup>25</sup>](#page-0-0) from the conditional distribution of <sup>X</sup> <sup>π</sup><sup>2</sup> given the input <sup>X</sup> π<sup>1</sup> , i.e. the conditional distribution X π<sup>2</sup> |X <sup>π</sup><sup>1</sup> = <sup>x</sup> ∼ N (µx,π1,π<sup>2</sup> , Σx,π1,π<sup>2</sup> ). The expected signature of the process <sup>X</sup> is then estimated from a collection of samples X 1,π1∪π<sup>2</sup> , . . . , X N,π1∪π<sup>2</sup> such that <sup>X</sup> n,π<sup>1</sup> = x and <sup>X</sup> n,π<sup>2</sup> ∼ N (µx,π1,π<sup>2</sup> , Σx,π1,π<sup>2</sup> ) for n = 1, . . . , N. In [Triggiano &](#page-10-3) [Romito](#page-10-3) [\(2024\)](#page-10-3), the authors emphasize the theoretical and empirical importance of the tensor normalization introduced in [Chevyrev & Oberhauser](#page-9-4) [\(2018\)](#page-9-4), ensuring the resulting expected robust signature is characteristic for a larger class of processes. An important component of the GPES model is thus the (truncated at level k) tensor normalization λ<sup>C</sup> : T <sup>K</sup>(<sup>R</sup> d ) → <sup>T</sup> <sup>K</sup>(<sup>R</sup> d ), controlled by the hyperparameter C. For more details on the effect of the tensor normalization procedure and a sensitivity analysis[<sup>26</sup>](#page-0-0) with respect to the hyper-parameter C we refer to [Triggiano & Romito](#page-10-3) [\(2024\)](#page-10-3). When applying the martingale correction to the GPES model[<sup>27</sup>](#page-0-0) we subtract the correction term cˆ ∗ <sup>1</sup>S I c (<sup>X</sup> k,π1∪π<sup>2</sup> )[0,T] to each S I (<sup>X</sup> k,π1∪π<sup>2</sup> )[0,T] , i.e. *before* taking the empirical expectation over the paths. This modification of the original algorithm highlighted in green in Algorithm [1.](#page-49-0) The final layer of the GPES model then maps the expected signature to a class by a combination of a linear transformation and a softmax output activation. The forward pass through the GPES model is summarized in Algorithm [1.](#page-49-0)

Algorithm 1 Gaussian Process augmented Expected Signature (GPES) classifier, forward pass

hyperparameters Signature truncation level <sup>k</sup> ∈ <sup>N</sup>, tensor normalization parameter <sup>C</sup> ∈ <sup>R</sup>+, data augmentation size <sup>N</sup> ∈ <sup>N</sup>, in-fill partition <sup>π</sup><sup>2</sup> ∈ <sup>∆</sup> M<sup>2</sup> [0,T] s.t. <sup>M</sup><sup>2</sup> ∈ <sup>N</sup>.

parameters Biases <sup>b</sup><sup>µ</sup> ∈ <sup>R</sup> <sup>d</sup><sup>µ</sup> , <sup>b</sup><sup>Σ</sup> ∈ <sup>R</sup> <sup>d</sup><sup>Σ</sup> , <sup>b</sup>out ∈ <sup>R</sup> <sup>d</sup>out and weights <sup>W</sup><sup>µ</sup> ∈ <sup>R</sup> dµ×din , <sup>W</sup><sup>Σ</sup> ∈ <sup>R</sup> dΣ×din , <sup>W</sup>out ∈ <sup>R</sup> dout×dsig where <sup>d</sup>in ← (d M<sup>1</sup> <sup>+</sup> <sup>M</sup><sup>1</sup> <sup>+</sup> <sup>M</sup>2), <sup>d</sup><sup>µ</sup> ← d M2, <sup>d</sup><sup>Σ</sup> ← d M2(d M<sup>2</sup> + 1)/2, <sup>d</sup>sig ← (<sup>d</sup> <sup>+</sup> . . . <sup>+</sup> <sup>d</sup> k ) and <sup>d</sup>out ← |C|. M<sup>1</sup>

- input <sup>x</sup> ∈ <sup>R</sup> <sup>d</sup>×M<sup>1</sup> and <sup>π</sup><sup>1</sup> ∈ <sup>∆</sup> [0,T] . 1: <sup>µ</sup>x,π1,π<sup>2</sup> ← <sup>b</sup><sup>µ</sup> <sup>+</sup> <sup>W</sup>µ(x, π1, π2). 2: <sup>L</sup>x,π1,π<sup>2</sup> ← <sup>b</sup><sup>Σ</sup> <sup>+</sup> <sup>W</sup>Σ(x, π1, π2) and <sup>Σ</sup>x,π1,π<sup>2</sup> ← <sup>L</sup>x,π1,π<sup>2</sup> <sup>L</sup> T x,π1,π<sup>2</sup> . 3: for <sup>n</sup> ∈ {1, . . . , N} do 4: X n,π<sup>1</sup> ← <sup>x</sup>. 5: Sample X n,π<sup>2</sup> ∼ N (µx,π1,π<sup>2</sup> , Σx,π1,π<sup>2</sup> ). 6: Signature of X n,π1∪π<sup>2</sup> : S <sup>n</sup> = S ≤k (<sup>X</sup> n,π1∪π<sup>2</sup> )[0,T]−c<sup>ˆ</sup> ∗ <sup>1</sup>S ≤k c (<sup>X</sup> n,π1∪π<sup>2</sup> )[0,T] ∈ <sup>R</sup> dsig . 7: Tensor normalization: S <sup>n</sup> ← <sup>λ</sup><sup>C</sup> (<sup>S</sup> <sup>n</sup>). 8: end for 9: Expected signature ES ← 1 N P<sup>N</sup> <sup>n</sup>=1 S
- n. output <sup>c</sup><sup>ˆ</sup> ← softmax(bout <sup>+</sup> <sup>W</sup>outES).

Note that, unlike classic Gaussian process regression where the prior mean is assumed to be constant <sup>µ</sup>(t) ≡ <sup>µ</sup> and the prior covariance function

$$\Sigma : [0, T] \rightarrow \mathbb{R}^{d \times d},$$

is parameterized by a kernel and posteriors are computed by combining standard properties of the multivariate normal distribution and the kernel trick[<sup>28</sup>](#page-0-0), in the GPES model the conditional mean and covariance functions

$$(\mathbf{x}, \pi_1, \pi_2) \in \mathbb{R}^{d \times M_1} \times \Delta_{[0, T]}^{M_1} \times \Delta_{[0, T]}^{M_2} \cong \mathbb{R}^{d M_1 + M_1 + M_2} \mapsto \begin{cases} \mu_{\mathbf{x}, \pi_1, \pi_2} \in \mathbb{R}^{d \times M_2} \cong \mathbb{R}^d M_2, \\ \Sigma_{\mathbf{x}, \pi_1, \pi_2} \in \mathcal{M}(\mathbb{R}^{d \times M_2}) \cong \mathbb{R}^d M_2 \times d M_2, \end{cases}$$

are parametrized by linear transformations

$$\begin{aligned}\mu_{\mathbf{x}, \pi_1, \pi_2} &= \mathbf{b}_\mu + W_\mu(\mathbf{x}, \pi_1, \pi_2), \\ \Sigma_{\mathbf{x}, \pi_1, \pi_2} &= L_{\mathbf{x}, \pi_1, \pi_2} L_{\mathbf{x}, \pi_1, \pi_2}^T, \quad L_{\mathbf{x}, \pi_1, \pi_2} = \mathbf{b}_\Sigma + W_\Sigma(\mathbf{x}, \pi_1, \pi_2),\end{aligned}$$

<sup>25</sup>Super-sampling the input data to a collection of realizations from a Gaussian process, can be effectively understood as a regularization by noise technique.

<sup>26</sup>Choosing a very large value of C is in practice equivalent to not applying a tensor normalization.

<sup>27</sup>Note that the GPES algorithm estimates the expected signature conditional on <sup>X</sup> n,π<sup>1</sup> = x, so technically the martingale correction is biasing the estimator.

<sup>28</sup>The Gaussian process regression model is then fitted by tuning the kernel hyperparameters (either via maximum likelihood or cross-validation).

where Lx,π1,π<sup>2</sup> is a lower triangular matrix. The parameters bµ, bΣ, Wµ, W<sup>Σ</sup> are learned along with the output layer parameters bout, Wout in the training phase via numerical optimization, in [Triggiano & Romito](#page-10-3) [\(2024\)](#page-10-3) the authors use simple stochastic gradient descent (SGD). If the timestamps π<sup>1</sup> of the observations x are not provided, they need to be fixed and can be regarded as hyperparameters of the model. The (way the) in-fill partition (is chosen) is instead always chosen a-priori, with a natural choice for π<sup>2</sup> being the set of mid-points of π1. Other model hyperparameters include the signature truncation level k, the size of the augmentation N and the constant C controlling the strength of the tensor normalization, as well as the training procedure's hyperparameters (learning rate, batch size etc.).

In Section [3.2.1](#page-7-2) we replicate the synthetic data experiments of [Triggiano & Romito](#page-10-3) [\(2024\)](#page-10-3). These consist of three datasets:

(FBM) Two equally balanced classes with samples generated according to a standard Brownian motion and a fractional Brownian motion with Hurst parameter H = 0.26 (both in dimension d = 1).

(OU) Two equally balanced classes with samples generated according to two different Ornstein-Uhlenbeck (OU) processes (both in dimension d = 1).

(Bidim) Six equally balanced classes with samples generated according to six different bi-dimensional stochastic processes (d = 2).

When fitting the models we take the optimal hyperparameters cross-validated by [Triggiano & Romito](#page-10-3) [\(2024\)](#page-10-3) [<sup>29</sup>](#page-0-0) and apply cross-validated SGD to the training dataset. That is, we use 80% of the training dataset to iterate through SGD parameter updates, while keeping the remaining 20% of the training dataset (the validation set) to determine when the procedure has converged without overfitting. As described in [\(Triggiano & Romito,](#page-10-3) [2024\)](#page-10-3) the presence of the tensor normalization step often leads to exploding gradients in the training procedure. We thus repeat the SGD routine over 5 different parameter initializations and pick the model with best validation performance.

F.2.2. PRICING PATH-DEPENDENT DERIVATIVES (L[YONS ET AL](#page-10-5)., [2021\)](#page-10-5)

While the authors of [Lyons et al.](#page-10-5) [\(2021\)](#page-10-5) consider a more general setting, for brevity, we focus on the case where the (discounted) price process <sup>X</sup> is assumed to be a semimartingale. Let <sup>X</sup> <sup>=</sup> {<sup>X</sup>t, t ∈ [0, T]} be a semimartingale on the probability space (Ω = <sup>C</sup>([0, T], <sup>R</sup>), F, <sup>P</sup>) and denote by Ωˆ LL T the set of realized (time and lead-lag augmented) price signatures, i.e.

$$\hat{\Omega}_T^{\text{LL}} = \{S(\hat{\mathbf{X}}^{\text{LL}})_{[0,T]} \in T((\mathbb{R}^4)) : \hat{\mathbf{X}} = \{(t, \mathbf{X}_t), t \geq 0\}\},$$

where we refer to [Lyons et al.](#page-10-5) [\(2021,](#page-10-5) Definition 2.14 and Example 2.15) for the definition of the lead-lag augmentation but, for the purposes of our discussion, it is sufficient to note that it is uniquely defined through Stratonovich integration. The authors then consider the market (Ωˆ LL T , B(Ωˆ LL T ), {Ft, t ∈ [0, T]}, <sup>P</sup>ˆLL) where <sup>P</sup>ˆLL is the push-forward of <sup>P</sup> onto Ωˆ LL T . By defining the set of derivative payoffs as all measurable F : Ωˆ LL <sup>T</sup> → <sup>R</sup>, i.e. for a given price realization <sup>X</sup> the holder of the derivative receives <sup>F</sup>(S(X<sup>ˆ</sup> LL)[0,T]), in [Lyons et al.](#page-10-5) [\(2021,](#page-10-5) Proposition 4.5) the authors use the universality of the signature to show that any *continuous* payoff F can be arbitrarily well approximated by a linear payoff[<sup>30</sup>](#page-0-0). In particular, this implies the price of any such F can be decomposed as

$$\mathbb{E}^{\mathbb{Q}}[Z_T F(S(\hat{\mathbb{X}}^{\text{LL}})_{[0,T]})] \approx \langle f, Z_T \mathbb{E}^{\mathbb{Q}}[S(\hat{\mathbb{X}}^{\text{LL}})_{[0,T]}] \rangle,$$

for a set of linear coefficients <sup>f</sup> ∈ <sup>T</sup>((<sup>R</sup> 4 ) ∗ ) where Q is a pricing measure for <sup>X</sup> and Z<sup>T</sup> is a deterministic discount factor over [0, T]. The set of signature payoffs {<sup>S</sup> I (X<sup>ˆ</sup> LL)[0,T] , I ∈ W({1, <sup>2</sup>, <sup>3</sup>, <sup>4</sup>})} can thus be understood as a set of Arrow-Debreu securities spanning the set of continuous path-dependent derivatives F. Similarly, in [Lyons et al.](#page-10-5) [\(2021,](#page-10-5) Proposition 4.6) the authors show that linear trading strategies are dense in the space of admissible trading strategies A, here defined as the set of all continuous functions <sup>θ</sup> : <sup>S</sup>(Xˆ)[0,t] 7→ <sup>θ</sup>(S(Xˆ)[0,t]) over the stopped at <sup>t</sup> ∈ [0, T] time-augmented price path signatures, i.e.

$$\theta(S(\hat{\mathbb{X}})_{[0,t]}) \approx \langle \ell, S(\hat{\mathbb{X}})_{[0,t]} \rangle, \quad \forall t \in [0, T],$$

<sup>29</sup>The only hyperparameter we modify is the truncation level k which we set to 4 for computational reasons (in the original paper the optimal value was found to be 5 or 6, depending on the dataset). The hyperparameters an the training and testing routines used to produce the results in Table [1](#page-7-0) can be found at <https://github.com/lorenzolucchese/gp-esig-classifier>.

<sup>30</sup>Note that the approximation of F by f does not depend on the choice of probability measure <sup>P</sup>, i.e. it is a pathwise density result. In Algorithm [2](#page-51-0) and Algorithm [3](#page-51-1) we thus assume that f has been estimated offline (i.e. for any model) by linearly regressing F(ω) against ω for a large set of ω ∈ Ωˆ LL <sup>T</sup> .

for a set of linear coefficients <sup>ℓ</sup> ∈ <sup>T</sup>((<sup>R</sup> 2 ) ∗ ). These two results are then combined in [Lyons et al.](#page-10-5) [\(2021,](#page-10-5) Theorem 4.7) to show that the solution of the quadratic <sup>P</sup>-hedging problem[<sup>31</sup>](#page-0-0)

$$\theta^* = \operatorname{argmin}_{\theta \in \mathcal{A}} \mathbb{E}^{\mathbb{P}} \left[ \left( F(S(\hat{\mathbf{X}}^{\text{LL}})_{[0,T]}) - p_0 - \int_0^T \theta(S(\hat{\mathbf{X}})_{[0,t]}) d\mathbf{X}_t \right)^2 \right], \quad (52)$$

can be arbitrarily well approximated by the solution of a linear signature quadratic hedging problem, i.e.

$$\theta^*(S(\hat{\mathbb{X}})_{[0,t]}) \approx \langle \ell^*, S(\hat{\mathbb{X}})_{[0,t]} \rangle, \quad t \in [0, T],$$

where ℓ <sup>∗</sup> ∈ <sup>T</sup>((<sup>R</sup> 2 ) ∗ ) can be computed by

$$\ell^* = \operatorname{argmin}_{\ell \in T((\mathbb{R}^2)^*)} \langle (f - p_0 \emptyset + \ell \mathbf{4})^{\sqcup 2}, \mathbb{E}^{\mathbb{P}}[S(\hat{\mathbf{X}}^{\text{LL}})_{[0,T]}] \rangle.$$

Algorithm 2 Pricing Path-Dependent Derivatives with Expected Signatures

hyperparameters Signature truncation levels <sup>k</sup>, number of Monte Carlo samples <sup>N</sup> ∈ <sup>N</sup>. parameters Risk-neutral measure <sup>Q</sup>, deterministic discount factor <sup>Z</sup><sup>T</sup> ∈ <sup>R</sup> <sup>+</sup>, linear approximator <sup>f</sup> ∈ <sup>T</sup> k ((<sup>R</sup> 4 ) ∗ ).

input Derivative payoff F : Ωˆ LL <sup>T</sup> → <sup>R</sup>.

1: Sample N trajectories X 1 , . . . , X <sup>N</sup> ∼ <sup>Q</sup>. 2: Compute time-augmented lead-lag transforms <sup>X</sup><sup>ˆ</sup> n,LL for <sup>n</sup> ∈ {1, . . . , N}. 3: Compute S <sup>n</sup> ∈ <sup>T</sup> k ((<sup>R</sup> 4 )) s.t. S n <sup>I</sup> ← <sup>S</sup> I (X<sup>ˆ</sup> n,LL)[0,T] , |I| ≤ <sup>k</sup> for <sup>n</sup> ∈ {1, . . . , N}. 4: Estimate <sup>Φ</sup> ∈ <sup>T</sup> k ((<sup>R</sup> 4 )) s.t. <sup>Φ</sup><sup>I</sup> ← <sup>ϕ</sup>ˆN,cˆ<sup>1</sup> I (T), |I| ≤ <sup>k</sup> from {X<sup>ˆ</sup> n,LL} N <sup>n</sup>=1.

output Price <sup>p</sup> <sup>=</sup> ⟨f, Z<sup>T</sup> <sup>Φ</sup>⟩.

Algorithm 3 Hedging Path-Dependent Derivatives with Expected Signatures

hyperparameters Signature truncation levels <sup>k</sup>, number of Monte Carlo samples <sup>N</sup> ∈ <sup>N</sup>. parameters Real-world measure <sup>P</sup>, initial capital <sup>p</sup><sup>0</sup> ∈ <sup>R</sup>, linear approximator <sup>f</sup> ∈ <sup>T</sup> k ((<sup>R</sup> 4 ) ∗ ).

input Derivative payoff F : Ωˆ LL <sup>T</sup> → <sup>R</sup>.

1: Sample N trajectories X 1 , . . . , X <sup>N</sup> ∼ <sup>P</sup>. 2: Compute time-augmented lead-lag transforms <sup>X</sup><sup>ˆ</sup> n,LL for <sup>n</sup> ∈ {1, . . . , N}. 3: Compute S <sup>n</sup> ∈ <sup>T</sup> k ((<sup>R</sup> 4 )) s.t. S n <sup>I</sup> ← <sup>S</sup> I (X<sup>ˆ</sup> n,LL)[0,T] , |I| ≤ <sup>k</sup> for <sup>n</sup> ∈ {1, . . . , N}. 4: Estimate <sup>Φ</sup> ∈ <sup>T</sup> k ((<sup>R</sup> 4 )) s.t. <sup>Φ</sup><sup>I</sup> ← <sup>ϕ</sup>ˆN,cˆ<sup>1</sup> I (T), |I| ≤ <sup>k</sup> from {X<sup>ˆ</sup> n,LL} N <sup>n</sup>=1. 5: ˆℓ <sup>∗</sup> ← infℓ∈T⌊k/2⌋((R<sup>2</sup>) ∗) ⟨(<sup>f</sup> − <sup>p</sup>0<sup>∅</sup> <sup>+</sup> <sup>ℓ</sup>4) 2 , <sup>Φ</sup>⟩.

output Hedging strategy <sup>t</sup> 7→ ⟨<sup>ˆ</sup><sup>ℓ</sup> ∗ , S(Xˆ)[0,t]⟩, t ∈ [0, T].

These theoretical results suggest both a pricing and a hedging strategy for path-dependent derivatives based on expected signatures[<sup>32</sup>](#page-0-0), summarized in Algorithm [2](#page-51-0) and Algorithm [3.](#page-51-1) Both algorithms make use of expected signature estimation via Monte Carlo simulations, an approach that provides a classic setting for applying the martingale correction described in Section [2.2](#page-5-2) (recall that by Remark [2.15](#page-5-3) and the Lead-Lag discussion in Appendix [F.1](#page-48-0) we apply the correction only to signature terms with the process X appearing in the outer integral). Note that price processes under P (as considered in the hedging setting) are usually not martingales and hence it is not clear whether the variance reduction for the Monte Carlo estimator of the expected signature obtained via the martingale correction offsets the introduced bias. On the other hand, under Q, the fundamental theorem of option pricing ensures the discounted asset price process X is a (local) martingale and hence the martingale correction is exact.

<sup>31</sup>For conciseness here we only discuss the quadratic hedging problem, in [Lyons et al.](#page-10-5) [\(2021\)](#page-10-5) the authors obtain results for general polynomials which allows them to also approximate the optimal hedge under exponential utility.

<sup>32</sup>As discussed in [Perez Arribas et al.](#page-10-14) ´ [\(2018,](#page-10-14) Section 6.1) the expected signature under the measure Q can alternatively be estimated in a model-free way from the market prices of a large enough set of exotic derivatives, yielding the implied expected signature. Here we assume the measures P and Q have been appropriately calibrated to market data.

#### F.2.3. DISTRIBUTIONAL REGRESSION FOR STREAMS (L[EMERCIER ET AL](#page-9-5)., [2021\)](#page-9-5)

The machine learning model we consider in Section [3.2.3](#page-8-3) is perhaps the most natural one when working with expected signatures. Introduced in [Lemercier et al.](#page-9-5) [\(2021\)](#page-9-5), the Signature of the pathwise Expected Signature (SES) model aims to learn a map from a collection of paths X , . . . , X <sup>N</sup> ∈ X ⊆ Lip([0, T], <sup>R</sup> d ), understood as an empirical measure on path space µ = 1 N P<sup>N</sup> <sup>n</sup>=1 <sup>δ</sup>X<sup>n</sup> ∈ P(X ), to a corresponding scalar value <sup>f</sup>(µ) ∈ <sup>R</sup>. The task of learning such function <sup>f</sup> : P(X ) → <sup>R</sup> from a finite number of (possibly noisy) observations {(µ<sup>i</sup> , yi)}i∈Itrain is known as distributional regression. Under appropriate conditions, by combining the characterizing property of the expected signature and the universality of the signature, the authors show that linear functionals on the signature of the pathwise expected signature are universal for weakly continuous functions <sup>f</sup> : P(X ) → <sup>R</sup> [\(Lemercier et al.,](#page-9-5) [2021,](#page-9-5) Theorem 3.2). The training and testing of the SES model is summarized in Algorithm [4](#page-52-2) with the step at which can apply the martingale correction highlighted in green.

Algorithm 4 Signature of pathwise Expected Signature (SES), training and testing

hyperparameters Signature truncation levels: <sup>k</sup>1, k<sup>2</sup> ∈ <sup>N</sup>. Linear regression regularizers. input {({<sup>X</sup> <sup>n</sup>}n∈N<sup>i</sup> , yi)}i∈Itrain , {({<sup>X</sup> <sup>n</sup>}n∈N<sup>i</sup> , yi)}i∈Itest where <sup>X</sup> <sup>n</sup> ∈ <sup>R</sup> <sup>d</sup>×<sup>M</sup> and <sup>y</sup><sup>i</sup> ∈ <sup>R</sup>. 1: <sup>d</sup><sup>1</sup> ← <sup>d</sup> <sup>+</sup> · · · <sup>+</sup> <sup>d</sup> <sup>k</sup><sup>1</sup> and <sup>d</sup><sup>2</sup> ← <sup>d</sup><sup>1</sup> <sup>+</sup> · · · <sup>+</sup> <sup>d</sup> k<sup>2</sup> 1 . 2: for <sup>i</sup> ∈ Itrain ∪ Itest do 3: Pathwise expected signature of {<sup>X</sup> <sup>n</sup>}n∈N<sup>i</sup> :Φ <sup>i</sup> ∈ <sup>R</sup> <sup>d</sup>1×<sup>M</sup>,Φ i I,m ←ϕˆNi,c<sup>ˆ</sup> I (tm), |I|≤<sup>k</sup>1, <sup>1</sup>≤m≤<sup>M</sup>. 4: Signature of Φ i : S <sup>i</sup> ← <sup>S</sup> <sup>≤</sup>k<sup>2</sup> (Φ<sup>i</sup> )[0,T] ∈ <sup>R</sup> d<sup>2</sup> . 5: end for 6: Fit linear regression: βˆ = (βˆ <sup>0</sup>, . . . , βˆ d<sup>2</sup> ) ← LinearRegressionFit({(<sup>S</sup> i , yi)}i∈Itrain ). 7: Predict using fitted linear regression: {yˆi}i∈Itest ← LinearRegressionPredict(βˆ, {<sup>S</sup> <sup>i</sup>}i∈Itest). output Performance metric: <sup>L</sup>({yˆi}i∈Itest , {<sup>y</sup>i}i∈Itest).

In Section [3.2.3](#page-8-3) we repeat two of the synthetic data experiments conducted in [Lemercier et al.](#page-9-5) [\(2021\)](#page-9-5), analyzing the performance of the SES model without and with martingale correction (MC). In the first experiment [\(Lemercier et al.,](#page-9-5) [2021,](#page-9-5) Section 5.2), the task is to infer the temperature of an ideal gas from the paths of N = 20 particles moving in a box. The dynamics of the system are inevitably linked to the radius of the particles, with larger particle radii resulting in more frequent collisions. Two settings[<sup>33</sup>](#page-0-0) are therefore considered, one with smaller particle radii <sup>r</sup><sup>1</sup> = 0.<sup>35</sup> × p<sup>3</sup> V /N and one with larger particle radii <sup>r</sup><sup>2</sup> = 0.<sup>65</sup> × p<sup>3</sup> V /N. The second experiment [\(Lemercier et al.,](#page-9-5) [2021,](#page-9-5) Section 5.3) concerns the estimation of the mean-reversion parameter in a rough volatility model. More precisely, the task is to infer the value of <sup>a</sup> ∈ [10<sup>−</sup><sup>6</sup> , 1] from a sample {<sup>σ</sup> n π} N <sup>n</sup>=1 of (discretely observed) paths σ <sup>n</sup> <sup>=</sup> {<sup>σ</sup> n t , t ∈ <sup>π</sup>} over the partition <sup>π</sup> <sup>=</sup> {0, <sup>0</sup>.01, . . . , <sup>2</sup>} with continuous-time dynamics

$$dZ_t = -a(Z_t - \mu)dt + \nu dB_t^H, \quad \sigma_t = \exp Z_t, \quad t \in [0, 2],$$

where {B<sup>H</sup> t , t ∈ [0, 2]} is a fractional Brownian motion with Hurst parameter <sup>H</sup> = 0.2, <sup>µ</sup> = 0.5, <sup>ν</sup> = 0.<sup>3</sup> and <sup>Z</sup><sup>0</sup> = 0.5. The performance of the model is evaluated with increasingly large collections of paths as inputs, i.e. N = 20, 50, 100. As expected, the model becomes more accurate as the number of paths increases. We refer to [Lemercier et al.](#page-9-5) [\(2021\)](#page-9-5) for more details on the two experimental setups.

In both experiments, we keep the same training-evaluation pipeline as the one considered in the original paper, namely nested k-fold cross-validation with 5 outer folds for evaluation and 3 inner folds for hyperparameter selection (including the signature truncation k<sup>1</sup> and a Lasso regularization parameter). The code used to produce the results of Table [2](#page-8-1) and Table [3](#page-8-2) is available at <https://github.com/lorenzolucchese/distribution-regression-streams>.

#### F.2.4. SYSTEMATIC TRADING (F[UTTER ET AL](#page-9-6)., [2023\)](#page-9-6)

The last application we consider is also motivated by a financial application and can be understood as a natural extension of the quadratic hedging problem [\(52\)](#page-51-2). In [Futter et al.](#page-9-6) [\(2023\)](#page-9-6) the authors consider the same setup as in [Lyons et al.](#page-10-5) [\(2021\)](#page-10-5) but allow the trading strategy <sup>θ</sup> ∈ A to depend on the signature of the augmented process

$$\hat{\mathbb{Z}} = \{(t, \mathbf{X}_t, \mathbf{f}_t), t \in [0, T]\} \in C([0, T], \mathbb{R}^{1+d+q}),$$

<sup>33</sup>V = 3cm<sup>3</sup> denotes the volume of the box in which the particles are moving.

where <sup>X</sup> <sup>=</sup> {<sup>X</sup>t, t ∈ [0, T]} ∈ <sup>C</sup>([0, T], <sup>R</sup> d ) is a <sup>d</sup>-dimensional price process and <sup>F</sup> <sup>=</sup> {<sup>f</sup>t, t ∈ [0, T]} ∈ <sup>C</sup>([0, T], <sup>R</sup> q ) is a set of q observable but not tradable factors (or signals) influencing X, and to trade in d assets. Again motivated by a universality result of the signature the authors approximate any such trading strategy by a linear one, i.e.

$$\theta_i(S(\mathbb{Z})_{[0,t]}) \approx \langle \ell_i, S(\mathbb{Z})_{[0,t]} \rangle, \quad i \in \{1, \dots, d\}, \quad t \in [0, T],$$

for some set of linear functionals <sup>ℓ</sup>1, . . . , ℓ<sup>d</sup> ∈ <sup>T</sup>((<sup>R</sup> 1+d+q )∗). The authors then show that an explicit solution to the path-dependent mean-variance problem

$$\ell_1^*, \dots, \ell_d^* = \begin{array}{l} \operatorname{argmin}_{\ell_1, \dots, \ell_d \in T^{k((\mathbb{R}^{1+d+q})^*)}} \mathbb{E}^{\mathbb{P}}[\text{PnL}_T], \\ \text{Var}(\text{PnL}_T) \leq \Delta \end{array} \quad \text{where} \quad \text{PnL}_T = \sum_{i=1}^d \int_0^T \langle \ell_i, S(\hat{Z})_{[0,t]} \rangle dX_t^i,$$

for arbitrary truncation level <sup>k</sup> ∈ <sup>N</sup> can be read from the entries of

$$\frac{1}{2\lambda_{\Delta}} \Sigma_{\text{sig}}^{-1} \mu_{\text{sig}} \in \mathbb{R}^{d_{\text{sig}}}, \quad d_{\text{sig}} = d + \dots + d(1 + d + q)^k,$$

where <sup>µ</sup>sig ∈ <sup>R</sup> <sup>d</sup>sig and <sup>Σ</sup>sig ∈ <sup>R</sup> <sup>d</sup>sig×dsig and <sup>λ</sup><sup>∆</sup> ∈ <sup>R</sup><sup>+</sup> only depend[<sup>34</sup>](#page-0-0) on the expected signature <sup>E</sup> [S(ZˆLL)[0,T] ].

A standard way of applying the sig-trading strategy in practice is given in Algorithm [5.](#page-53-1) Note that in real financial markets price and signal paths cannot be resampled and hence the collection {(<sup>X</sup> <sup>n</sup>, <sup>F</sup> n)} N <sup>n</sup>=1 can only be obtained by chopping-andshifting a single long observation {(Xt,ft), t ∈ [0, NT]}. In this respect, the sig-trading algorithm provides a striking example of a setting where the sampling scheme cannot be considered i.i.d. and hence one needs to resort to the results of Theorem [2.10](#page-4-1) to obtain theoretical guarantees for the expected signature estimator.

A silent but fundamental assumption[<sup>35</sup>](#page-0-0) is that the market dynamics of the collection of *past* price-factor paths {(<sup>X</sup> <sup>n</sup>, <sup>F</sup> n)} N n=1 used to estimate the expected signature will be the same as those of the *future* price-factor process (<sup>X</sup> ∗ , F ∗ ) to which the trading strategy will be applied. Using the martingale correction for estimating (some of the entries of) the expected signature, induces a bias to "ignore" the drift component of the signature term. For example, the first level of the martingale-corrected expected signature is always zero. This might be a desirable feature to avoid over-fitting the trading strategy to spurious drifts in the data (for example in the price processes X [\(Buehler et al.,](#page-9-23) [2022\)](#page-9-23)) while capturing higher order effects. As with other applications discussed in this section, the usefulness of the martingale correction in Algorithm [5](#page-53-1) can be empirically cross-validated.

Algorithm 5 Signature Trading

hyperparameters Signature truncation level <sup>k</sup>, maximum variance <sup>∆</sup> ∈ <sup>R</sup>+. input Collection of price-factor paths {(<sup>X</sup> <sup>n</sup>, <sup>F</sup> n)} N <sup>n</sup>=1 where <sup>X</sup> <sup>n</sup> <sup>=</sup> {X<sup>n</sup> t , t ∈ [0, T]} ∈ <sup>C</sup>([0, T], <sup>R</sup> d ) and <sup>F</sup> <sup>n</sup> <sup>=</sup> {<sup>f</sup> n t , t ∈ q

[0, T]} ∈ <sup>C</sup>([0, T], <sup>R</sup> ) for <sup>n</sup> ∈ {1, . . . , N}. 1: Set <sup>Z</sup><sup>ˆ</sup> ← {(t, <sup>X</sup>t,ft), t ∈ [0, T]}. 2: Compute time-augmented lead-lag transforms <sup>Z</sup>ˆn,LL for <sup>n</sup> ∈ {1, . . . , N}. 3: Estimate <sup>Φ</sup> ∈ <sup>T</sup> k ((<sup>R</sup> 1+d+q )) s.t. <sup>Φ</sup><sup>I</sup> ← <sup>ϕ</sup>ˆN,cˆ<sup>1</sup> I (T), |I| ≤ <sup>k</sup> from {Zˆn,LL} N <sup>n</sup>=1. 4: Compute λ∆, µsig, Σsig from corresponding entries of Φ. 5: Extract ˆℓ ∗ i from corresponding entries of (2λ∆Σsig) <sup>−</sup><sup>1</sup>µsig for <sup>i</sup> ∈ {1, . . . , d}. output Trading strategy <sup>t</sup> 7→ ⟨<sup>ˆ</sup><sup>ℓ</sup> ∗ i , S(Zˆ<sup>∗</sup> )[0,t]⟩, i ∈ {1, . . . , d}, t ∈ [0, T] for new (<sup>X</sup> ∗ , F ∗ ).

## G. Controlled Linear Regression

In this section, we introduce the notion of controlled linear regression. The main rationale is to exploit as much information as possible in the training phase to make the coefficient estimators as precise as possible. We start by considering the following linear model

$$y = \beta_1 x_1 + \dots + \beta_p x_p + \epsilon = \mathbf{x}^T \boldsymbol{\beta} + \epsilon,$$

<sup>34</sup>As in the mean-variance optimal portfolio, the variance scaling parameter λ<sup>∆</sup> ∈ <sup>R</sup><sup>+</sup> also depends on the target variance ∆.

<sup>35</sup>Clearly, this assumption is not specific to the sig-trading strategy but applies to any trading strategy that tries to learn patterns from the past to profit in the future.

and assume we observe the training data {(yn, xn,1, . . . , xn,p), n = 1, . . . , N}, i.e.

$$y_n = \beta_1 x_{n,1} + \dots + \beta_p x_{n,p} + \epsilon_n = \mathbf{x}_n^T \boldsymbol{\beta} + \epsilon_n, \quad n = 1, \dots, N,$$

where, given the design matrix <sup>X</sup> ∈ <sup>R</sup> N×p , the errors {<sup>ϵ</sup>n, n = 1, . . . , N} are

- mean zero, i.e. <sup>E</sup>[ϵn|X] = 0 for <sup>n</sup> = 1, . . . , N,
- homoskedastic, i.e. <sup>E</sup>[ϵ 2 n |X] = <sup>σ</sup> <sup>2</sup> ∈ [0, ∞) for <sup>n</sup> = 1, . . . , N, and
- uncorrelated, i.e. <sup>E</sup>[ϵnϵm|X] = 0 for n, m = 1, . . . , N with <sup>n</sup> ̸<sup>=</sup> <sup>m</sup>.

For any test observation <sup>x</sup><sup>∗</sup> = (x∗,1, . . . , x∗,p) ∈ <sup>R</sup> <sup>p</sup> we thus have the best possible prediction (in terms of MSE) for y<sup>∗</sup> is

$$\mathbb{E}[y_* | \mathbf{x}_*] = \mathbf{x}_*^T \boldsymbol{\beta} =: \hat{y}_*(\boldsymbol{\beta}),$$

and plugging in an estimator βˆ for β yields the predictor

$$\hat{y}_*(\hat{\beta}) = \mathbf{x}_*^T \hat{\beta}.$$

Note that, under the assumptions introduced above, the mean squared error of such a predictor can be decomposed as

$$\begin{aligned}\mathbb{E} \left[ (\hat{y}_*(\hat{\beta}) - y_*)^2 | \mathbf{X}, \mathbf{x}_* \right] &= \mathbb{E} \left[ \epsilon_*^2 | \mathbf{X}, \mathbf{x}_* \right] + \mathbb{E} \left[ (\hat{y}_*(\hat{\beta}) - \hat{y}_*(\beta))^2 | \mathbf{X}, \mathbf{x}_* \right] \\ &= \sigma^2 + \mathbb{E} \left[ (\hat{y}_*(\hat{\beta}) - \hat{y}_*(\beta))^2 | \mathbf{X}, \mathbf{x}_* \right].\end{aligned}$$

Under the assumptions discussed above minimizing the mean squared error of the predictor relative to the target y<sup>∗</sup> is thus equivalent to minimizing the mean squared error of the predictor relative to the infeasible best prediction yˆ∗(β).

## G.1. Controlled Ordinary Least Squares (OLS) estimation

Classic OLS estimation The usual OLS estimator for <sup>β</sup> = (β1, . . . , βp) ∈ <sup>R</sup> p is given by[<sup>36</sup>](#page-0-0)

$$\hat{\beta}_X = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y},$$

which, by the Gauss-Markov theorem, is known to be the best linear unbiased estimator (BLUE): for any <sup>λ</sup> = (λ1, . . . , λp) ∈ R p ,

$$\mathbb{E} \left[ (\lambda^\top \hat{\beta}_\mathbf{x} - \lambda^\top \beta)^2 | \mathbf{X} \right] = \min_{\tilde{\beta}_\mathbf{x} \in \text{LUE}(\mathbf{X}, \mathbf{y})} \mathbb{E} \left[ (\lambda^\top \tilde{\beta}_\mathbf{x} - \lambda^\top \beta) | \mathbf{X} \right],$$

where LUE(X, <sup>y</sup>) is the set of all linear and unbiased estimator for <sup>β</sup>, i.e. <sup>β</sup>˜<sup>X</sup> <sup>=</sup> <sup>C</sup>(X)<sup>y</sup> for some <sup>X</sup>-measurable matrix <sup>C</sup>(X) ∈ <sup>R</sup> <sup>p</sup>×<sup>N</sup> and <sup>E</sup>[β˜X|X] = <sup>β</sup>. By applying the BLUE property, we can show that <sup>y</sup>ˆ∗(βˆX) is the best[<sup>37</sup>](#page-0-0) predictor across all predictors formed from linear and unbiased estimators, i.e.

$$\mathbb{E} \left[ (\hat{y}_*(\hat{\beta}_{\mathbf{X}}) - \hat{y}_*(\beta))^2 | \mathbf{X}, \mathbf{x}_* \right] \leq \mathbb{E} \left[ (\hat{y}_*(\tilde{\beta}_{\mathbf{X}}) - \hat{y}_*(\beta))^2 | \mathbf{X}, \mathbf{x}_* \right],$$

for all <sup>β</sup>˜<sup>X</sup> ∈ LUE(X, <sup>y</sup>). Note that here we applied the mean-zero uncorrelated errors assumption.

Controlled OLS estimation Let us now assume we can additionally observe the "control" random variables {<sup>z</sup><sup>n</sup> <sup>=</sup> (zn,1, . . . , zn,k) ∈ <sup>R</sup> k , n = 1, . . . , N}. We shall assume the controls are available for training, i.e. when estimating <sup>β</sup>, but for predicting, i.e. when forecasting <sup>y</sup><sup>∗</sup> we will have access to <sup>x</sup><sup>∗</sup> but not to <sup>z</sup>∗. Given the original design matrix <sup>X</sup> ∈ <sup>R</sup> N×p , we now assume the errors and the controls are jointly

- mean zero, i.e. 
  $$\mathbb{E}[(\epsilon_n, \mathbf{z}_n) | \mathbf{X}] = \mathbf{0} \in \mathbb{R}^{k+1}$$
   for  $n = 1, \dots, N$ ,

<sup>36</sup>Here, and in all other estimators discussed in this section, the dependence on y is dropped from the notation.

<sup>37</sup>In terms of mean squared error (MSE). Recall that for unbiased estimators, the MSE is equal to the estimator's variance.

- homoskedastic, i.e. <sup>E</sup>[(ϵn, zn) ⊗2 |X] = Σ ∈ <sup>R</sup> (k+1)×(k+1) for n = 1, . . . , N for some Σ symmetric positive definite,
- uncorrelated, i.e. <sup>E</sup>[(ϵn, <sup>z</sup>n)⊗(ϵm, <sup>z</sup>m)|X] = <sup>0</sup> ∈ <sup>R</sup> (k+1)×(k+1) for n, m = 1, . . . , N with <sup>n</sup> ̸<sup>=</sup> <sup>m</sup>.

In what follows we partition

$$\Sigma = \begin{pmatrix} \sigma^2 & \Sigma_{y,\mathbf{z}} \\ \Sigma_{\mathbf{z},y} & \Sigma_{\mathbf{z}} \end{pmatrix},$$

*Remark* G.1*.* Throughout the whole section, unless stated otherwise, we consider the original design matrix X and the new observation <sup>x</sup><sup>∗</sup> ∈ <sup>R</sup> p to be fixed with random controls Z and errors ϵ.

As discussed above, fixing the design matrix <sup>X</sup> ∈ <sup>R</sup> N×p and a test observation <sup>x</sup><sup>∗</sup> = (x∗,1, . . . , x∗,p) ∈ <sup>R</sup> p , the predictor

$$\hat{y}_*(\hat{\beta}_{\mathbf{X}}) = \mathbf{x}_*^T\hat{\beta}_{\mathbf{X}},$$

is unbiased for the statistic yˆ∗(β) = x T <sup>∗</sup><sup>β</sup> ∈ <sup>R</sup>. We hence introduce the control variate predictor

$$\hat{y}_*(\hat{\beta}_{\mathbf{X}}, \mathbf{Z}, \boldsymbol{\lambda}) = \hat{y}_*(\hat{\beta}_{\mathbf{X}}) + \boldsymbol{\lambda}_1^T \mathbf{Z} \boldsymbol{\lambda}_2,$$

where <sup>Z</sup> ∈ <sup>R</sup> N×k is the control design matrix while <sup>λ</sup><sup>1</sup> ∈ <sup>R</sup> <sup>N</sup> and <sup>λ</sup><sup>2</sup> ∈ <sup>R</sup> k are measurable in X and x∗. Under the assumptions discussed above the controlled predictor can be shown to be unbiased and attains a minimum[<sup>38</sup>](#page-0-0) in variance when

$$\boldsymbol{\lambda}_1^* = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_* \quad \text{and} \quad \boldsymbol{\lambda}_2^* = -\Sigma_{\mathbf{z}}^{-1} \Sigma_{\mathbf{z},y}.$$

$$\begin{aligned} \text{Var}(\hat{y}_*(\hat{\beta}_{\mathbf{X}}, \mathbf{Z}, \boldsymbol{\lambda}) | \mathbf{X}, \mathbf{x}_*) &= \text{Var}(\mathbf{x}_*^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T (\mathbf{X} \beta + \boldsymbol{\epsilon}) + \boldsymbol{\lambda}_1^T \mathbf{Z} \boldsymbol{\lambda}_2 | \mathbf{X}, \mathbf{x}_*) \\ &= \mathbf{x}_*^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbb{E}[\boldsymbol{\epsilon} \boldsymbol{\epsilon}^T | \mathbf{X}, \mathbf{x}_*] \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_* + 2 \mathbf{x}_*^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbb{E}[\boldsymbol{\epsilon} \boldsymbol{\lambda}_2^T \mathbf{Z}^T | \mathbf{X}, \mathbf{x}_*] \boldsymbol{\lambda}_1 + \boldsymbol{\lambda}_1^T \mathbb{E}[\mathbf{Z} \boldsymbol{\lambda}_2 \boldsymbol{\lambda}_2^T \mathbf{Z}^T | \mathbf{X}, \mathbf{x}_*] \boldsymbol{\lambda}_1 \\ &= \sigma^2 \mathbf{x}_*^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_* + 2 \boldsymbol{\lambda}_2^T \Sigma_{\mathbf{z}, \mathbf{y}} \mathbf{x}_*^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \boldsymbol{\lambda}_1 + \boldsymbol{\lambda}_2^T \Sigma_{\mathbf{z}} \boldsymbol{\lambda}_2 \boldsymbol{\lambda}_1^T \boldsymbol{\lambda}_1. \end{aligned}$$

Setting partial derivatives in λ<sup>1</sup> and λ<sup>2</sup> equal to zero

$$\begin{aligned} \partial_{\lambda_1} \text{Var}(\hat{y}_*(\hat{\beta}_{\mathbf{X}}, \mathbf{Z}, \boldsymbol{\lambda}) | \mathbf{X}, \mathbf{x}_*) &= 2\lambda_2^{\text{T}} \Sigma_{\mathbf{Z}, y} \mathbf{x}_*^{\text{T}} (\mathbf{X}^{\text{T}} \mathbf{X})^{-1} \mathbf{X}^{\text{T}} + 2\lambda_2^{\text{T}} \Sigma_{\mathbf{Z}} \boldsymbol{\lambda}_2 \boldsymbol{\lambda}_1 = 2\lambda_2^{\text{T}} (\Sigma_{\mathbf{Z}, y} \mathbf{x}_*^{\text{T}} (\mathbf{X}^{\text{T}} \mathbf{X})^{-1} \mathbf{X}^{\text{T}} + \Sigma_{\mathbf{Z}} \boldsymbol{\lambda}_2 \boldsymbol{\lambda}_1) = 0, \\ \partial_{\lambda_2} \text{Var}(\hat{y}_*(\hat{\beta}_{\mathbf{X}}, \mathbf{Z}, \boldsymbol{\lambda}) | \mathbf{X}, \mathbf{x}_*) &= 2\lambda_1^{\text{T}} \mathbf{X} (\mathbf{X}^{\text{T}} \mathbf{X})^{-1} \mathbf{x}_* \Sigma_{\mathbf{Z}, y} + 2\lambda_1^{\text{T}} \boldsymbol{\lambda}_1 \Sigma_{\mathbf{Z}} \boldsymbol{\lambda}_2 = 2\lambda_1^{\text{T}} (\mathbf{X} (\mathbf{X}^{\text{T}} \mathbf{X})^{-1} \mathbf{x}_* \Sigma_{\mathbf{Z}, y} + \boldsymbol{\lambda}_1 \Sigma_{\mathbf{Z}} \boldsymbol{\lambda}_2) = 0, \end{aligned}$$

yields as non-trivial (i.e. such that λ1,λ<sup>2</sup> ̸= 0) stationary point λ ∗ <sup>1</sup> = X(X<sup>T</sup>X) <sup>−</sup><sup>1</sup>x<sup>∗</sup> and λ ∗ <sup>2</sup> = −Σ −1 <sup>z</sup> Σz,y. By taking second derivatives we can compute the Hessian at the stationary point to be

$$\partial_\lambda \partial_{\lambda^T} \text{Var}(\hat{y}_*(\hat{\beta}_{\mathbf{X}}, \mathbf{Z}, \boldsymbol{\lambda}) | \mathbf{X}, \mathbf{x}_*) \Big|_{\lambda=\lambda_*} = 2 \begin{pmatrix} \Sigma_{y, \mathbf{z}} \Sigma_{\mathbf{z}}^{-1} \Sigma_{\mathbf{z}, y} I_{N \times N} & -\mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_* \Sigma_{y, \mathbf{z}} \\ -\Sigma_{\mathbf{z}, y} \mathbf{x}_* (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T & \mathbf{x}_* (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_* \Sigma_{\mathbf{z}} \end{pmatrix},$$

which is positive (semi)definite since Σ is positive definite and hence λ ∗ is a minimum. To show the assumption that Σ is positive definite implies the Hessian is positive (semi)definite we use the following result from linear algebra theory: a symmetric block matrix

$$\mathbf{M} = \begin{pmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{B}^T & \mathbf{D} \end{pmatrix},$$

where A is square is positive (semi)definite if and only A and D − B <sup>T</sup>A−<sup>1</sup>B are positive (semi)definite. Applying this result to the Hessian at λ<sup>∗</sup> note that A = Σy,zΣ −1 <sup>z</sup> Σz,yIN×<sup>N</sup> is trivially positive (semi)definite since Σy,zΣ −1 <sup>z</sup> Σz,y ≥ 0 by positive definitness of Σz, and

$$\mathbf{D} - \mathbf{B}^T \mathbf{A}^{-1} \mathbf{B} = \mathbf{x}_*(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_* \left( \Sigma_{\mathbf{z}} - \frac{\Sigma_{\mathbf{z},y} \Sigma_{y,\mathbf{z}}}{\Sigma_{y,\mathbf{z}} \Sigma_{\mathbf{z}}^{-1} \Sigma_{\mathbf{z},y}} \right),$$

is positive (semi)definite since x∗(X<sup>T</sup>X) <sup>−</sup><sup>1</sup>x<sup>∗</sup> ≥ 0 by positive definitness of X<sup>T</sup>X and Σ<sup>z</sup> − (Σy,zΣ −1 <sup>z</sup> Σz,y) <sup>−</sup><sup>1</sup>Σz,yΣy,<sup>z</sup> is positive (semi)definite by applying the converse of the previous statement to the positive (semi)definite matrix

$$\mathbf{M}' = \begin{pmatrix} \Sigma_{y,\mathbf{z}} \Sigma_{\mathbf{z}}^{-1} \Sigma_{\mathbf{z},y} & \Sigma_{y,\mathbf{z}} \\ \Sigma_{\mathbf{z},y} & \Sigma_{\mathbf{z}} \end{pmatrix}.$$

<sup>38</sup>Note that since Z and ϵ are assumed to be jointly spherical given X (and x∗) the variance of the controlled predictor is given by

For any <sup>x</sup><sup>∗</sup> = (x∗,1, . . . , x∗,p) ∈ <sup>R</sup> <sup>p</sup> we thus have

$$\hat{y}_*(\hat{\beta}_{\mathbf{X}}, \mathbf{Z}, \boldsymbol{\lambda}^*) = \mathbf{x}_*^{\text{T}}(\mathbf{X}^{\text{T}}\mathbf{X})^{-1}\mathbf{X}^{\text{T}}\mathbf{y} - \mathbf{x}_*^{\text{T}}(\mathbf{X}^{\text{T}}\mathbf{X})^{-1}\mathbf{X}^{\text{T}}\mathbf{Z}\Sigma_{\mathbf{z}, \mathbf{y}}^{-1}\Sigma_{\mathbf{z}, \mathbf{y}} = \mathbf{x}_*^{\text{T}}\hat{\beta}_{\mathbf{X}, \mathbf{Z}, \Sigma} = \hat{y}_*(\hat{\beta}_{\mathbf{X}, \mathbf{Z}, \Sigma}),$$

yielding the infeasible (in the sense that it depends on the unknown quantity Σ) estimator

$$\hat{\beta}_{\mathbf{X}, \mathbf{Z}, \Sigma} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T (\mathbf{y} - \mathbf{Z} \Sigma_{\mathbf{z}}^{-1} \Sigma_{\mathbf{z}, \mathbf{y}}) = \hat{\beta}_{\mathbf{X}} - (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Z} \Sigma_{\mathbf{z}}^{-1} \Sigma_{\mathbf{z}, \mathbf{y}}.$$

Note that, given <sup>X</sup>, <sup>β</sup>ˆX,Z,<sup>Σ</sup> is unbiased (by mean zero property of the controls). Moreover, for any test observation <sup>x</sup><sup>∗</sup> = (x∗,1, . . . , x∗,p) ∈ <sup>R</sup> p ,

$$\begin{aligned}
& \text{Var}(\mathbf{x}_*^T \hat{\beta}_{\mathbf{X}, \mathbf{Z}, \Sigma} | \mathbf{X}, \mathbf{x}_*) \\
&= \text{Var}(\mathbf{x}_*^T \hat{\beta}_{\mathbf{X}} | \mathbf{X}, \mathbf{x}_*) + \text{Var}(\mathbf{x}_*^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Z} \Sigma_{\mathbf{Z}}^{-1} \Sigma_{\mathbf{Z}, y} | \mathbf{X}, \mathbf{x}_*) \\
&\quad - 2 \text{Cov}(\mathbf{x}_*^T \hat{\beta}_{\mathbf{X}}, \mathbf{x}_*^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Z} \Sigma_{\mathbf{Z}}^{-1} \Sigma_{\mathbf{Z}, y} | \mathbf{X}, \mathbf{x}_*) \\
&= \sigma^2 \mathbf{x}_*^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_* + \Sigma_{y, \mathbf{z}} \Sigma_{\mathbf{z}}^{-1} \Sigma_{\mathbf{z}, y} \mathbf{x}_*^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_* - 2 \Sigma_{y, \mathbf{z}} \Sigma_{\mathbf{z}}^{-1} \Sigma_{\mathbf{z}, y} \mathbf{x}_*^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_* \\
&= (\sigma^2 - \Sigma_{y, \mathbf{z}} \Sigma_{\mathbf{z}}^{-1} \Sigma_{\mathbf{z}, y}) \mathbf{x}_*^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_* \\
&\leq \sigma^2 \mathbf{x}_*^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_* = \text{Var}(\mathbf{x}_*^T \hat{\beta}_{\mathbf{X}} | \mathbf{X}, \mathbf{x}_*),
\end{aligned}$$

with equality iff Σz,y = 0. In other words, as long as the controls are correlated with the target, we obtain a better prediction by using <sup>β</sup>ˆX,Z,<sup>Σ</sup> instead of the OLS estimator <sup>β</sup>ˆ<sup>X</sup> and the quality of the prediction increases as the correlation between <sup>y</sup> and z increases. The variance reduction factor is constant across test observations and is given by

$$\frac{\text{Var}(\mathbf{x}_*^\top \hat{\boldsymbol{\beta}}_{\mathbf{X}, \mathbf{Z}, \Sigma} | \mathbf{X}, \mathbf{x}_*)}{\text{Var}(\mathbf{x}_*^\top \hat{\boldsymbol{\beta}}_{\mathbf{X}} | \mathbf{X}, \mathbf{x}_*)} = \left( 1 - \frac{\Sigma_{y, \mathbf{y}} \Sigma_{\mathbf{z}}^{-1} \Sigma_{\mathbf{z}, \mathbf{y}}}{\sigma^2} \right).$$

*Remark* G.2*.* Note that when <sup>X</sup> <sup>=</sup> <sup>1</sup> ∈ <sup>R</sup> <sup>N</sup> , <sup>x</sup><sup>∗</sup> = 1 and <sup>Z</sup> ∈ <sup>R</sup> <sup>N</sup> , we estimate µ<sup>∗</sup> = <sup>E</sup>[y] with the OLS estimator µˆ<sup>∗</sup> = βˆ = y¯ and the simplest control variate estimator

$$\hat{\mu}_*^c = \bar{\mathbf{y}} - \frac{\text{Cov}(y, z)}{\text{Var}(z)} \bar{\mathbf{z}},$$

since λ ∗ <sup>1</sup> = 1 N <sup>1</sup> ∈ <sup>R</sup> <sup>N</sup> and λ ∗ <sup>2</sup> <sup>=</sup> − Cov(y, z) Var(z) , which has reduced variance by a factor of (1 − Corr(y, z)).

In practice, the correlation matrix <sup>Σ</sup> is usually unknown; hence, to make the estimator <sup>β</sup>ˆX,Z,<sup>Σ</sup> feasible, we need to estimate it. Under the assumptions discussed above, the most natural candidate is given by the sample estimates

$$\hat{\Sigma}_{\mathbf{z},y} = \frac{1}{N}\mathbf{Z}^T\mathbf{y} \quad \text{and} \quad \hat{\Sigma}_{\mathbf{z}} = \frac{1}{N}\mathbf{Z}^T\mathbf{Z},$$

yielding the feasible estimator

$$\hat{\beta}_{\mathbf{X}, \mathbf{Z}, \hat{\Sigma}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T (I - \mathbf{Z}(\mathbf{Z}^T \mathbf{Z})^{-1} \mathbf{Z}^T) \mathbf{y}.$$

This can be equivalently understood as regressing y on the control Z (i.e. projecting y onto the space spanned by Z) and then regressing the residual onto X. The resulting estimator will likely be biased (given X), with its exact finite sample properties depending on the distribution of (X, <sup>Z</sup>)|<sup>ϵ</sup>.

When ϵ depends linearly on Z, i.e.

$$\epsilon = Z\alpha + \eta,$$

we can write

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{Z}\boldsymbol{\alpha} + \boldsymbol{\eta},$$

and hence we know that the joint OLS estimator obtained from the design matrix (X Z) ∈ <sup>R</sup> N×(p+k) , i.e.

$$\begin{pmatrix} \hat{\beta}_{\mathbf{X},\mathbf{Z}} \\ \hat{\alpha}_{\mathbf{X},\mathbf{Z}} \end{pmatrix} = \begin{bmatrix} (\mathbf{X}^T) & (\mathbf{X} & \mathbf{Z}) \end{bmatrix}^{-1} \begin{pmatrix} \mathbf{X}^T \\ \mathbf{Z}^T \end{pmatrix} \mathbf{y} = \begin{pmatrix} \mathbf{X}^T \mathbf{X} & \mathbf{X}^T \mathbf{Z} \\ \mathbf{Z}^T \mathbf{X} & \mathbf{Z}^T \mathbf{Z} \end{pmatrix}^{-1} \begin{pmatrix} \mathbf{X}^T \mathbf{y} \\ \mathbf{Z}^T \mathbf{y} \end{pmatrix}.$$

By using some simple algebraic manipulations for block matrices, we can extract the first p entries of the joint OLS estimator, i.e. the estimator for β, as

$$\hat{\beta}_{\mathbf{X},\mathbf{Z}} = \hat{\beta}_{\mathbf{X},\mathbf{Z},\hat{\Sigma}'} = \hat{\beta} - (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Z}(\mathbf{Z}^T\mathbf{Z} - \mathbf{Z}^T\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Z})^{-1}(\mathbf{Z}^T\mathbf{y} - \mathbf{Z}^T\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}),$$

i.e. <sup>β</sup>ˆX,Z,<sup>Σ</sup> with <sup>Σ</sup> estimated by

$$\hat{\Sigma}'_{\mathbf{z}, \mathbf{y}} = \frac{1}{N} \mathbf{Z}^T (I - \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T) \mathbf{y} \quad \text{and} \quad \hat{\Sigma}'_{\mathbf{z}} = \frac{1}{N} \mathbf{Z}^T (I - \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T) \mathbf{Z}.$$

Note these are the covariance and variance estimators obtained when projecting Z onto the orthogonal complement of the space spanned by X. Under the assumptions introduced above, these are also unbiased for Σz,y and Σz. This provides a second feasible controlled estimator for β.

If we fix both X and Z and assume η satisfies the Gauss-Markov conditions, then the joint OLS estimator is the BLUE for (β, <sup>α</sup>). Extending the classic OLS estimator <sup>β</sup>ˆ<sup>X</sup> to (βˆX, <sup>α</sup>ˆX,Z) we obtain another linear and unbiased estimator for (β, <sup>α</sup>). It follows from the Gauss-Markov theorem that for any <sup>x</sup><sup>∗</sup> ∈ <sup>R</sup> p , setting <sup>λ</sup> = (x∗, <sup>E</sup>[z∗|X, <sup>Z</sup>, <sup>x</sup>∗] = <sup>0</sup>), one has

$$\mathbb{E} \left[ (\hat{y}_*(\hat{\beta}_{\mathbf{X}, \mathbf{Z}}) - \hat{y}_*(\beta))^2 \middle| \mathbf{X}, \mathbf{Z}, \mathbf{x}_* \right] \leq \mathbb{E} \left[ (\hat{y}_*(\hat{\beta}_{\mathbf{X}}) - \hat{y}_*(\beta))^2 \middle| \mathbf{X}, \mathbf{Z}, \mathbf{x}_* \right],$$

and hence, by the tower property of conditional expectation,

$$\mathbb{E} \left[ (\hat{y}_*(\hat{\beta}_{\mathbf{X}, \mathbf{Z}}) - \hat{y}_*(\beta))^2 | \mathbf{X}, \mathbf{x}_* \right] \leq \mathbb{E} \left[ (\hat{y}_*(\hat{\beta}_{\mathbf{X}}) - \hat{y}_*(\beta))^2 | \mathbf{X}, \mathbf{x}_* \right].$$

Using the forms of the variances of βˆ<sup>X</sup> and βˆX,<sup>Z</sup> we can compute

$$\begin{aligned} \mathbb{E} \left[ (\hat{y}_*(\hat{\mathbf{\beta}}_{\mathbf{X}, \mathbf{Z}}) - \hat{y}_*(\boldsymbol{\beta}))^2 | \mathbf{X}, \mathbf{x}_* \right] &= \sigma^2 \mathbf{x}_*^T \mathbb{E}[(\mathbf{X}^T (I - \mathbf{Z}(\mathbf{Z}^T \mathbf{Z})^{-1} \mathbf{Z}^T) \mathbf{X})^{-1} | \mathbf{X}] \mathbf{x}_*, \\ \mathbb{E} \left[ (\hat{y}_*(\hat{\mathbf{\beta}}_{\mathbf{X}}) - \hat{y}_*(\boldsymbol{\beta}))^2 | \mathbf{X}, \mathbf{x}_* \right] &= \sigma^2 \mathbf{x}_*^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_*, \end{aligned}$$

and thus quantify the MSE reduction factor as

$$\frac{\mathbb{E} \left[ (\hat{y}_*(\hat{\beta}_{\mathbf{X},\mathbf{Z}}) - \hat{y}_*(\boldsymbol{\beta}))^2 | \mathbf{X}, \mathbf{x}_* \right]}{\mathbb{E} \left[ (\hat{y}_*(\hat{\beta}_{\mathbf{X}}) - \hat{y}_*(\boldsymbol{\beta}))^2 | \mathbf{X}, \mathbf{x}_* \right]} = \frac{\mathbf{x}_*^T \mathbb{E} [(\mathbf{X}^T (I - \mathbf{Z}(\mathbf{Z}^T \mathbf{Z})^{-1} \mathbf{Z}^T) \mathbf{X})^{-1} | \mathbf{X}] \mathbf{x}_*}{\mathbf{x}_*^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_*},$$

which we note does not depend on σ 2 .

Given X and Z, augmenting the first feasible controlled estimator βˆ <sup>X</sup>,Z,Σˆ to an estimator for (β, α), yields a linear but biased (at least in the β components) estimator. Whether βˆ <sup>X</sup>,Z,Σˆ or <sup>β</sup>ˆX,<sup>Z</sup> yields a better predictor cannot thus be deduced from the Gauss-Markov theorem. As we will see in the numerical experiments discussed in the next section, which estimator performs better depends on the properties of the data generating process.

#### G.2. Simulation study

In the previous section we introduced two feasible control estimators, βˆ <sup>X</sup>,Z,Σˆ and <sup>β</sup>ˆX,Z, for the parameters <sup>β</sup> ∈ <sup>R</sup> p . We showed that when

$$\epsilon = Z\alpha + \eta,$$

and <sup>η</sup>|Z, <sup>X</sup> is mean-zero, uncorrelated and homoskedastic then <sup>y</sup>ˆ∗(βˆX,Z) is always a better predictor than the one formed by the classic OLS estimator <sup>y</sup>ˆ∗(βˆX). This leaves unanswered the question of how <sup>β</sup><sup>ˆ</sup> <sup>X</sup>,Z,Σˆ performs relative to <sup>β</sup>ˆX,<sup>Z</sup> (and <sup>β</sup>ˆX). We address this question empirically in Table [<sup>4</sup>](#page-58-0) with the following experimental setup.

We consider N = 1, 000 i.i.d. samples from the model

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \epsilon,$$

with <sup>β</sup><sup>0</sup> <sup>=</sup> −1, β<sup>1</sup> = 6, β<sup>2</sup> = 8 and

$$\epsilon = \sigma(\rho z + \sqrt{1 - \rho^2}\eta),$$

with <sup>η</sup> ∼ N (0, 1) ⊥ <sup>z</sup> ∼ N (0, 1) independently of <sup>X</sup>. We thus have Corr(y, z|x) = Corr(ϵ, z) = <sup>ρ</sup> and <sup>ϵ</sup> ∼ N (0, σ<sup>2</sup> ) ⊥ <sup>X</sup>. We fix <sup>N</sup> = 1, <sup>000</sup> samples <sup>x</sup><sup>1</sup> ∼ N (0, 1) ⊥ <sup>x</sup><sup>2</sup> ∼ N (1, 1) to obtain <sup>X</sup> ∈ <sup>R</sup> N×3 and evaluate the performance of the predictor on the new observation x ∗ <sup>1</sup> = 0, x<sup>∗</sup> <sup>2</sup> = 1. For each estimator, we report an estimate of the root mean square error

$$\text{RMSE}(\hat{y}_*(\hat{\beta}), \hat{y}_*(\beta)) = \sqrt{\mathbb{E} \left[ (\hat{y}_*(\hat{\beta}) - \hat{y}_*(\beta))^2 \middle| \mathbf{X}, \mathbf{x}_* \right]},$$

obtained over 10, 000 Monte Carlo samples (i.e. keeping X and x <sup>∗</sup> fixed and resampling <sup>Z</sup> ∈ <sup>R</sup> <sup>N</sup> and <sup>η</sup> ∈ <sup>R</sup> <sup>N</sup> ). We highlight with a single asterisk (\*) RMSEs that are lower than the uncontrolled OLS estimator's RMSE with high statistical significance (t-test p-value across the Monte Carlo samples less than 0.001). When one of the two proposed estimators outperforms the other with high statistical significance we highlight the corresponding RMSE with double asterisks (\*\*).

| σ ρ  | β ˆ X RMSE | RMSE     | β ˆ X , Z , Σˆ (% of β ˆ | X ) RMSE | β ˆ X , Z (% of β ˆ | X ) RMSE | β ˆ X , Z , Σ (% of β ˆ X ) |
|------|------------|----------|--------------------------|----------|---------------------|----------|-----------------------------|
| 0.00 | 0.1572     | 0.1579   | 100.46%                  | 0.1573   | 100.03%             | 0.1572   | 100.00%                     |
| 0.25 | 0.1588     | 0.1550*  | 97.57%                   | 0.1538** | 96.86%              | 0.1489   | 93.75%                      |
| 0.50 | 0.1581     | 0.1368*  | 86.48%                   | 0.1358** | 85.86%              | 0.1186   | 75.00%                      |
| 0.75 | 0.1592     | 0.1052*  | 66.08%                   | 0.1043** | 65.52%              | 0.0697   | 43.75%                      |
| 1.00 | 0.1606     | 0.0164*  | 10.21%                   | 0.0000** | 0.01%               | 0.0000   | 0.00%                       |
| 0.00 | 0.3144     | 0.3146   | 100.06%                  | 0.3145   | 100.03%             | 0.3144   | 100.00%                     |
| 0.25 | 0.3177     | 0.3082*  | 97.03%                   | 0.3077** | 96.86%              | 0.2978   | 93.75%                      |
| 0.50 | 0.3163     | 0.2719*  | 85.97%                   | 0.2715*  | 85.86%              | 0.2372   | 75.00%                      |
| 0.75 | 0.3185     | 0.2088*  | 65.57%                   | 0.2086*  | 65.52%              | 0.1393   | 43.75%                      |
| 1.00 | 0.3211     | 0.0164*  | 5.11%                    | 0.0000** | 0.00%               | 0.0000   | 0.00%                       |
| 0.00 | 0.6289     | 0.6286   | 99.96%                   | 0.6291   | 100.03%             | 0.6289   | 100.00%                     |
| 0.25 | 0.6353     | 0.6154*  | 96.86%                   | 0.6154*  | 96.86%              | 0.5956   | 93.75%                      |
| 0.50 | 0.6325     | 0.5429*  | 85.83%                   | 0.5431*  | 85.86%              | 0.4744   | 75.00%                      |
| 0.75 | 0.6369     | 0.4169*  | 65.46%                   | 0.4173*  | 65.52%              | 0.2786   | 43.75%                      |
| 1.00 | 0.6422     | 0.0164*  | 2.55%                    | 0.0000** | 0.00%               | 0.0000   | 0.00%                       |
| 0.00 | 1.2578     | 1.2569   | 99.93%                   | 1.2582   | 100.03%             | 1.2578   | 100.00%                     |
| 0.25 | 1.2707     | 1.2300** | 96.80%                   | 1.2307*  | 96.86%              | 1.1913   | 93.75%                      |
| 0.50 | 1.2651     | 1.0853** | 85.79%                   | 1.0862*  | 85.86%              | 0.9488   | 75.00%                      |
| 0.75 | 1.2738     | 0.8336** | 65.44%                   | 0.8346*  | 65.52%              | 0.5573   | 43.75%                      |
| 1.00 | 1.2845     | 0.0164*  | 1.28%                    | 0.0000** | 0.00%               | 0.0000   | 0.00%                       |

Table 4. RMSEs of the classic OLS estimator <sup>β</sup>ˆX, the two feasible control estimators <sup>β</sup><sup>ˆ</sup> <sup>X</sup>,Z,Σˆ and <sup>β</sup>ˆX,Z, and the infeasible optimal control estimator <sup>β</sup>ˆX,Z,Σ. A single asterisk (\*) indicates feasible RMSEs that are lower than the uncontrolled classic OLS estimator's RMSE with high statistical significance (t-test p-value across the 10,000 Monte Carlo samples less than 0.001). A double asterisk (\*\*) indicates the feasible control estimator's RMSE is lower than the other feasible control estimator's RMSE with high statistical significance.

As expected, as the correlation between the control and the target ρ increases, the performance gain obtained by using either of the feasible control estimators increases. The joint-OLS estimator outperforms the standard OLS estimator by the same amount across different signal-to-noise ratios while βˆ <sup>X</sup>,Z,Σˆ 's relative performance changes. Comparing <sup>β</sup><sup>ˆ</sup> <sup>X</sup>,Z,Σˆ and <sup>β</sup>ˆX,<sup>Z</sup> we note the results suggest that for high signal-to-noise ratios[<sup>39</sup>](#page-0-0) the joint-OLS estimator <sup>β</sup>ˆX,<sup>Z</sup> slightly outperforms <sup>β</sup><sup>ˆ</sup> X,Z,Σˆ while for lower signal-to-noise ratios the latter estimator performs marginally better. Consider the two edge cases:

<sup>39</sup>Here we define the signal-to-noise ratio as std(x <sup>T</sup>β)/σ where std(x <sup>T</sup>β) measures the variablity of the explainable part of the model (signal) and σ measures the variability in the error ϵ (noise). In Table [4](#page-58-0) the signal component of the model is kept fixed and hence higher σ denotes lower signal-to-noise ratio.

- when there is no error, i.e. σ = 0, βˆ <sup>X</sup>,Z,Σˆ performs worse than <sup>β</sup>ˆ<sup>X</sup> (<sup>=</sup> <sup>β</sup>ˆX,Z) as it is adding uninformative variability to the estimator;
- when there is no signal, i.e. <sup>X</sup> <sup>=</sup> <sup>1</sup> ∈ <sup>R</sup> <sup>N</sup> , βˆ <sup>X</sup>,Z,Σˆ and <sup>β</sup>ˆX,<sup>Z</sup> both reduce to the classic control variates estimators, cf. Remark [G.2,](#page-56-0) but the former is more precise as it estimates λ ∗ <sup>2</sup> <sup>=</sup> −<sup>Σ</sup> −1 <sup>z</sup> Σz,y using the knowledge that the control is mean zero.

Next, we investigate the empirical performance of βˆ <sup>X</sup>,Z,Σˆ and <sup>β</sup>ˆX,<sup>Z</sup> when the dependency between <sup>Z</sup> and <sup>ϵ</sup> is non-linear but (Z, ϵ)|<sup>X</sup> are still jointly mean-zero, uncorrelated (across samples) and homoskedastic. Keeping the same design matrix <sup>X</sup> as in Table [<sup>4</sup>](#page-58-0) and the same parameters <sup>β</sup> = (−1, <sup>6</sup>, 8) we now let

$$\epsilon = \sigma \kappa f(z) + \sigma \sqrt{1 - \kappa^2} \eta,$$

with <sup>η</sup> ∼ N (0, 1) ⊥ <sup>z</sup> ∼ N (0, 1) independently of <sup>X</sup>. By choosing <sup>f</sup>(z) such that <sup>E</sup>[f(z)] = 0 and <sup>E</sup>[f(z) 2 ] = 1 we have <sup>E</sup>[ϵ] = 0 and <sup>E</sup>[ϵ 2 ] = σ . Moreover, choosing <sup>κ</sup> <sup>=</sup> Cov(z, f(z))<sup>−</sup><sup>1</sup><sup>ρ</sup> ensures that Corr(y, z|x) = Corr(ϵ, z) = <sup>ρ</sup>. We investigate the following three dependence functions:

$$(\text{sq}) \quad f(z) = \frac{z^2 + z - 1}{\sqrt{3}};$$

$$(\text{cube}) \quad f(z) = \frac{z^3}{\sqrt{15}};$$

$$(\exp) \quad f(z) = \frac{e^z - \sqrt{e}}{\sqrt{e^2 - e}}.$$

The code used to produce the results of Table [4](#page-58-0) and Table [5](#page-60-0) can be found at [https://github.com/](https://github.com/lorenzolucchese/controlled-linear-regression) [lorenzolucchese/controlled-linear-regression](https://github.com/lorenzolucchese/controlled-linear-regression).

We also experimented with multiplicative noise (<sup>ϵ</sup> ∝ <sup>f</sup>(z) <sup>η</sup>), heavier tailed errors (<sup>η</sup> ∼ <sup>t</sup>3) and different dataset sizes (<sup>N</sup> ∈ {100, <sup>1000</sup>, <sup>10000</sup>}). The results are similar to the ones reported in Table [<sup>4</sup>](#page-58-0) and Table [5:](#page-60-0) the two control estimators outperform the classic OLS estimator as long as ρ > 0 while the differences in performance between the two control estimators are statistically significant but small.

| f ρ  | β ˆ X RMSE | RMSE    | β ˆ X , Z , Σˆ (% of β ˆ | X ) RMSE | β ˆ X , Z (% of β ˆ | X ) RMSE | β ˆ X , Z , Σ (% of β ˆ X ) |
|------|------------|---------|--------------------------|----------|---------------------|----------|-----------------------------|
| 0.00 | 0.3144     | 0.3146  | 100.06%                  | 0.3145   | 100.03%             | 0.3144   | 100.00%                     |
| 0.25 | 0.3174     | 0.3084* | 97.14%                   | 0.3075** | 96.88%              | 0.2976   | 93.75%                      |
| 0.50 | 0.3202     | 0.2772* | 86.57%                   | 0.2761** | 86.25%              | 0.2401   | 75.00%                      |
| 0.00 | 0.3144     | 0.3146  | 100.06%                  | 0.3145   | 100.03%             | 0.3144   | 100.00%                     |
| 0.25 | 0.3184     | 0.3088* | 97.00%                   | 0.3083*  | 96.83%              | 0.2985   | 93.75%                      |
| 0.50 | 0.3184     | 0.2736* | 85.95%                   | 0.2734*  | 85.86%              | 0.2388   | 75.00%                      |
| 0.75 | 0.3222     | 0.2125* | 65.96%                   | 0.2123*  | 65.90%              | 0.1409   | 43.75%                      |
| 0.00 | 0.3144     | 0.3146  | 100.06%                  | 0.3145   | 100.03%             | 0.3144   | 100.00%                     |
| 0.25 | 0.3178     | 0.3085* | 97.07%                   | 0.3078** | 96.85%              | 0.2979   | 93.75%                      |
| 0.50 | 0.3182     | 0.2741* | 86.13%                   | 0.2734** | 85.92%              | 0.2387   | 75.00%                      |
| 0.75 | 0.3191     | 0.2091* | 65.53%                   | 0.2081** | 65.20%              | 0.1396   | 43.75%                      |

Table 5. RMSEs of the classic OLS estimator <sup>β</sup>ˆX, the two feasible control estimators <sup>β</sup><sup>ˆ</sup> <sup>X</sup>,Z,Σˆ and <sup>β</sup>ˆX,Z, and the infeasible optimal control estimator <sup>β</sup>ˆX,Z,Σ. A single asterisk (\*) indicates feasible RMSEs that are lower than the uncontrolled classic OLS estimator's RMSE with high statistical significance (t-test p-value across the 10,000 Monte Carlo samples less than 0.001). A double asterisk (\*\*) indicates the feasible control estimator's RMSE is lower than the other feasible control estimator's RMSE with high statistical significance. Empty values indicate setups that are not achievable for the given correlation ρ and dependence function f.