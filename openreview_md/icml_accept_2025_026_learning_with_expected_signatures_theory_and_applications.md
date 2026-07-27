# 

Lorenzo Lucchese 1 Mikko S. Pakkanen 1 **Almut E. D. Veraart** 1

## Abstract

of iterated integrals of X, i.e. for t ∈ [0, T], k ≥ 0 The expected signature maps a collection of data streams to a lower dimensional representation, with a remarkable property: the resulting feature tensor can fully characterize the data generating distribution. This "model-free" embedding has been successfully leveraged to build multiple domain-agnostic machine learning (ML) algorithms for time series and sequential data. The convergence results proved in this paper bridge the gap between the expected signature's empirical discrete-time estimator and its theoretical continuous-time value, allowing for a more complete probabilistic interpretation of expected signature-based ML methods. Moreover, when the data generating process is a martingale, we suggest a simple modification of the expected signature estimator with significantly lower mean squared error and empirically demonstrate how it can be effectively applied to improve predictive performance.

## 1. Introduction

The signature transform of a stream of data is an infinite but countable sequence of its "iterated integrals" summarizing the input in a top-down fashion, meaning the informational content of its terms decays factorially. Originally introduced by Chen (1954) and serving as a fundamental object of rough path analysis (Lyons et al., 2007), the signature

$${\mathfrak{S}}=\{S({\mathfrak{X}})_{[0,t]}\in T(({\mathbb{R}}^{d})),\;t\in[0,T]\},$$

of a path X = {Xt, t ∈ [0, T]} ∈ C([0, T], R
d) is a lift
(in the sense that it embeds X) to the space of continuous functions over the tensor algebra T((R
d)) possessing some nice algebraic and geometric properties. When the path is of bounded variation, the signature is defined as the sequence 1Department of Mathematics, Imperial College London, London, United Kingdom. Correspondence to: Lorenzo Lucchese <lorenzo.lucchese17@imperial.ac.uk, llucchese6@gmail.com>.

$$S^{k}(\mathbb{X})_{[0,t]}=\int\cdots\int\mathrm{d}\mathbf{X}_{s_{1}}\otimes\cdots\otimes\mathrm{d}\mathbf{X}_{s_{k}}.\tag{1}$$

In many practical applications the path X is taken to be the piecewise linear interpolation of a discrete-time stream of data, which is of bounded variation by construction. Signature-based machine learning (ML) approaches (Lyons & McLeod, 2024) thus often restrict the theoretical framework to paths in BV([0, T], R
d). In this setting, two fundamental properties of the signature that make it a desirable non-parametric feature extraction method for sequential data are the characterization result of Hambly & Lyons (2005) and the universality approximation theorem of Levin et al. (2016). Moreover, when the path X is understood as a (realization of a) random process with distribution P over BV([0, T], R
d), the shuffle property of the signature implies that all moments of the random variable S(X)[0,T]
are determined by its expectation

$$\phi(T):=\mathbb{E}[S]$$

## Φ(T) := E[S(X)[0,T]] ∈ T((R D)).

A natural question, known as the Hamburger moment problem (Fawcett, 2003), is thus whether the expectation of the signature characterizes its law (and thus the law of the path). When imposing a probability distribution P on X
the assumption of bounded variation paths becomes quite restrictive: Brownian motion, the basic building block of many stochastic models, has paths of infinite variation almost surely. Even if we observe a discrete-time stream of data, we often still would like to define the process X as a latent stochastic process of which we observe the linear interpolation over some partition π of [0, T], hereafter denoted by X
π. We hence wish to make sense of the signature of a stochastic process X with paths of unbounded variation.

For a given path X ∈ C([0, T], R
d) of finite p-variation, once we "lift" the process to a p-rough path (Lyons et al.,
2007, Definition 3.11) then the signature S of X is uniquely defined1. Without delving into the details of rough path theory, for our purposes it suffices to interpret the choice of lift as fixing a notion of integration with respect to X: the higher order signatures terms are then understood as iterated integrals of the path X defined in this sense.

1 Motivated by the fact that we can only ever observe the process X over a discrete partition π of [0, T] we restrict our attention to the class of stochastic processes whose lift (and hence signature) can be approximated by the lift (and hence signature) of the bounded variation path X
π. Following the rough path literature we take such approximation in the pvariation metric to define the notion of canonical geometric stochastic process, cf. Definition 2.1. In Chevyrev & Lyons (2016); Chevyrev & Oberhauser (2018) the authors provide characterization results for the expected signature of canonical geometric stochastic processes, i.e. conditions under which the map P 7→ E[S(X)[0,T]] is injective. Such characterizing property of the expected signature has found practical use in a wide range of applications, ranging from classic ML tasks (Lemercier et al., 2021; Triggiano & Romito, 2024; Schell & Oberhauser, 2023) to mathematical finance (Lyons et al., 2021; Futter et al., 2023). The expected signature is thus a highly informative quantity and, consequently, methods for computing ϕ(T) have received considerable research interest. Such methods can be broadly categorized into two classes: those employing an analytical approach and those following a statistical one. Analytical methods aim to develop exact formulas for specific classes of models. A first step in this direction was taken in Ni (2012, Section 4) showing that the expected signature of an Ito diffusion satisfies an explicit partial differential ˆ equation (PDE). This result was subsequently generalized in Cuchiero et al. (2023) to the class of signature-SDEs and in Friz et al. (2022; 2024) to (discontinuous) semimartingales. On the other hand, the statistical approach aims to estimate ϕ(T) directly from observed data, preserving the model-free nature of the expected signature. For a given set of observations X
1,π*, . . . ,* X
N,π one can form the estimator

$$\hat{\phi}^{N,\pi}(T):=\frac{1}{N}\sum_{n=1}^{N}S(\mathbb{X}^{n,\pi})_{[0,T]},$$

as illustrated in Figure 1, and study its in-fill |π| → 0 and large-sample N → ∞ asymptotics. This line of work includes the explicit results of Ni (2012, Section 3.2) for Brownian motion and of Passeggeri (2020) for fractional Brownian motion with Hurst parameter H > 1/2 as well as the preliminary results in Friz & Victoir (2010) for more general semimartingales. Additionally, Schell & Oberhauser (2023, Section 8) develops asymptotic results for processes of bounded variation. In this work we provide a unifying set of general conditions under which the expected signature estimator ϕˆN,π(T) displays important asymptotic statistical properties, namely consistency and asymptotic normality.

Our results allow for irregular2 observation partitions π
- possibly varying across samples - and for dependency 2Clearly, for the estimation problem to be well-posed, the sequence of partitions needs to be signature defining in the sense of Definition 2.5.

across the samples X
1,π*, . . . ,* X
n,π. The first main contribution of this paper is thus to bridge the gap between the empirical expected signature estimator and the expected signature of a latent continuous-time stochastic process, unlocking a more general probabilistic interpretation of several ML algorithms and effectively moving beyond the expected signature as a simple feature extraction method. This naturally leads to the second theoretical contribution: by starting from the continuous-time setting we devise a modification of the expected signature estimator with significantly better finite sample properties when the latent data generating process is a martingale. The superior performance of this modified estimator is empirically verified through various experiments with expected signature-based ML algorithms from the literature.

## 2. Theory

Let X = {Xt, t ∈ [0, T]} denote a d-dimensional stochastic process over the probability space (Ω, F, P).

Definition 2.1. We say X is a canonical geometric stochastic process of rough order p if there exists a sequence of partitions ρ with |ρ| → 0 such that the limit in the p-variation metric of the canonically lifted linearly interpolated process X 
ρexists in probability. Convergence in probability implies almost sure convergence (along a subsequence) and hence we can almost surely define the lift of X as such limit.

Remark 2.2. The definition of lift suggests this might depend on the choice of the sequence of partitions ρ. In any case, for a wide range of stochastic processes there exist canonical lifts that satisfy our definition of canonical geometric rough path. These include:
- Semimartingales: For p ∈ (2, 3) any semimartingale can be lifted to a geometric p-rough path by defining the lift via Stratonovich integration; the signature of X
 then coincides with iterated Stratonovich integrals.

For any sequence of partitions ρ the lifts of the linear interpolations converge in p-variation metric to the Stratonovich lift (Friz & Victoir, 2010, Chapter 14) and hence X is a canonical geometric stochastic process in the sense of Definition 2.1.

- Gaussian processes: Many Gaussian processes admit canonical lifts to geometric p-rough paths (Friz & Victoir (2010, Theorem 15.34, Definition 15.35) and Coutin & Qian (2002)) with the existence criterion for such canonical lifts easily stated in terms of the covariance function. The definition of the lift implicitly requires ρ to be any sequence such that X
ρconverges uniformly to X almost surely. For example, fractional Brownian motion with Hurst parameter H > 1/4 can be lifted to a geometric p-rough path with p > 1/H by choosing ρ to be the sequence of dyadic partitions.

In what follows, we will assume the canonical geometric stochastic process X has a *canonical* lift (i.e. a canonical sequence of partitions ρ along which the lift is defined) and unambiguously refer to it as the lift of X.

Let ρ denote a partition of [0, T] with mesh |ρ| and X
ρ =
{X
ρ t, t ∈ [0, T]} the linear approximation of X over ρ, i.e.

$$\mathbf{X}_{t}^{\rho}=\mathbf{X}_{u}+{\frac{t-u}{v-u}}\mathbf{X}_{u,v},\quad t\in[u,v]\in\rho,$$

with Xu,v = Xv − Xu. The signature of the bounded variation path X
ρ up to time t ∈ [0, T] is defined by Equation (1)
through classic Riemann-Stieltjes integration and can thus be computed by

$$S(\mathbb{X}^{\rho})_{[0,t]}=\bigotimes_{[u,v]\in\rho_{[0,t]}}\exp_{\otimes}\mathbf{X}_{u,v}^{\rho},\tag{2}$$

where ρ[0,t] denotes the restriction of ρ to [0, t]. The canonical lift of X
ρto a (geometric) p-rough path is

$$\left(1,\,S^{1}(\mathbb{X}^{\rho})_{[0,t]},\,\ldots,S^{[p]}(\mathbb{X}^{\rho})_{[0,t]}\right)\in T^{[p]}\left((\mathbb{R}^{d})\right),\tag{3}$$

for t ∈ [0, T]. Definition 2.1 requires that there exists a sequence of partitions ρ for which this sequence of geometric p-rough paths converges in probability in the p-variation metric. A key result from rough path theory is that a geometric p-rough path has a full signature. Fixing the lift
of X via Definition 2.1, we thus have a uniquely specified signature for X.
Definition 2.3. The signature of a canonical geometric
stochastic process X,
$$\mathbb{S}=\{S(\mathbb{X})_{[0,t]}\in T((\mathbb{R}^{d})),\;t\in[0,T]\},$$
is defined pathwise (on a set of full measure) as the unique
extension of the lift of X to a multiplicative functional of
arbitrary order in the sense of (Lyons et al., 2007, Theorem 3.7). The elements of the signature are the rough
iterated integrals of X.

Remark 2.4. Taking ρ to be a sequence such that Definition 2.1 holds, by continuity of the extension map (Lyons et al., 2007, Theorem 3.10), it immediately follows that the signature of X
ρ(truncated at level K ≥ ⌊p⌋) converges in probability to the signature of X (up to level K) in the p-variation topology. In particular, this implies that, for any finite collection of words I,

$$S^{\mathbf{I}}(\mathbb{X}^{\rho})_{[0,t]}\stackrel{\mathbb{P}}{\longrightarrow}S^{\mathbf{I}}(\mathbb{X})_{[0,t]}.$$
(4)  $\frac{1}{2}$ .............................. 

I(X)[0,t]. (4)
Similar arguments imply that, when convergence to the lift along ρ holds almost surely in the p-variation metric, then also the higher order signature terms converge almost surely, and, in particular, (4) holds in the almost sure limit. In the following sections, we will be estimating the expected signature at fixed time horizon T > 0. To develop the properties of these estimators, it will be thus sufficient to work with pointwise limits like (4) without having to deal with the stronger pathwise p-variation convergence used to define canonical geometric stochastic processes. This mode of convergence will thus be sufficient to consider a sequence of partitions as *signature-defining*.

Definition 2.5. Let X = {Xt, t ∈ [0, T]} be a canonical geometric stochastic process, we say that a sequence of partitions π of the interval [s, t] ⊆ [0, T] with |π| → 0 is signature-defining if for any collection of words I,

$$S^{\mathbf{I}}(\mathbb{X}^{\pi})_{[s,t]}\stackrel{{\mathbb{P}}}{{\rightarrow}}S^{\mathbf{I}}(\mathbb{X})_{[s,t]},\quad|\pi|\to0.\tag{5}$$

## 2.1. Expected Signature Estimation

In this section, we assume we have access to N copies of X
discretely observed over possibly different partitions of the interval [0, T], i.e. each X
n,πN,n is an observation over πN,n of a continuous-time latent process X
n, for n = 1*, . . . , N*.

We will focus on two observational schemes:
(ind) Repeatedly observe X through N independent experiments, in which case the "underlying" signatures S(X
n)[0,T], for n = 1*, . . . , N*, are independent and identically distributed.

(chop) Chop-up (and shift in time) a single observation of the process {Xt, t ≥ 0} over a partition

$$\Pi(N):=\pi_{N,1}\cup\cdots\cup((N-1)T+\pi_{N,N}),$$

of [0*, NT*]. In this setting, we assume that the latent sequence {X
n, n ≥ 1} taking values in C([0, T]; R
d)
is stationary, i.e. for k ∈ N, n1*, . . . , n*k ∈ N and n ≥
0,

$$(\mathbb{X}^{n_{1}},\ldots,\mathbb{X}^{n_{k}})\stackrel{{\mathcal{L}}}{{=}}(\mathbb{X}^{n_{1}+n},\ldots,\mathbb{X}^{n_{k}+n}),\tag{6}$$

and hence the signatures S(X
n)[0,T]form a stationary sequence. This assumption ensures the task of estimating ϕI(T) is well-posed. Note this condition is slightly stronger than necessary but weaker than requiring {Xt, t ≥ 0} to be stationary, cf. Proposition 2.13.

The first observational framework can be recast in the second by appropriately pasting the X
n's into a single process
{Xt, t ≥ 0}. Going forward we hence focus on the second setting and refer to the large sample asymptotics N → ∞
as long-span asymptotics. For any finite collection of words I, we thus consider the estimator

$$\hat{\phi}_{\bf I}^{\Pi(N)}(T):=\frac{1}{N}\sum_{n=1}^{N}S^{\bf I}(\mathbb{X}^{n,\pi_{N},n})_{[0,T]}.\tag{7}$$

We will be interested in the double asymptotics where, as the number of signature evaluations N increases, the granularity of the discretized paths from which such signatures are computed also increases, i.e.

$$|\Pi(N)|:=\operatorname*{max}_{1\leq n\leq N}|\pi_{N,n}|\to0,\quad N\to\infty.$$

We can decompose

$$\hat{\phi}_{\bf I}^{\Pi(N)}(T)-\phi_{\bf I}(T)$$ $$=\frac{1}{N}\sum_{n=1}^{N}\left(S^{\bf I}(\mathbb{X}^{n,\pi_{N,n}})_{[0,T]}-S^{\bf I}(\mathbb{X}^{n})_{[0,T]}\right)$$ $$+\frac{1}{N}\sum_{n=1}^{N}S^{\bf I}(\mathbb{X}^{n})_{[0,T]}-\mathbb{E}\left[S^{\bf I}(\mathbb{X})_{[0,T]}\right].$$

Under suitable conditions, we shall prove ϕˆΠ(N)
I(T) is consistent and asymptotically normal for ϕI(T) by showing 1. each summand in the first term converges to zero in L
m in the in-fill asymptotics |πN,n| → 0; 2. the second term, when inflated by √N, converges in distribution to a normal random variable in the large sample asymptotics N → ∞.

## 2.1.1. In-Fill Asymptotics

The convergence in probability (5) is not sufficient to show consistency of the expected signature estimator. In this section, we thus explore continuity conditions on the process X ensuring the convergence holds in a stronger L
m sense.

Let {Fs,t, [*s, t*] ⊆ [0, T]} be a family of sigma-algebras such that, for [u, v] ⊆ [*s, t*] ⊆ [0, T], Fu,v ⊆ Fs,t and, for [s, t] ⊆ [0, T], Xs,u is Fs,t-measurable for all u ∈ [*s, t*].

The following continuity assumptions will be used to state the in-fill asymptotics.

Assumption 2.6. For all 0 ≤ *s < u < t* ≤ T,

$$\|\mathbf{X}_{s,t}\|_{L^{p}}\leq|t-s|^{\alpha}.$$
(A$\beta$) $\|\mathbb{E}_{\mathcal{F}_{0,s}\lor\mathcal{F}_{t,T}}[\mathbf{X}_{s,u}\otimes\mathbf{X}_{u,t}]\|_{L^{p/2}}\leq|t-s|^{\beta}$.  
(A$\gamma$) $\|\mathbb{E}_{\mathcal{F}_{0,s}\lor\mathcal{F}_{t,T}}[\mathbf{X}_{s,u}\otimes\mathbf{X}_{u,t}^{\otimes2}]\|_{L^{p/3}}\leq|t-s|^{\gamma}$, $\|\mathbb{E}_{\mathcal{F}_{0,s}\lor\mathcal{F}_{t,T}}[\mathbf{X}_{s,u}^{\otimes2}\otimes\mathbf{X}_{u,t}]\|_{L^{p/3}}\leq|t-s|^{\gamma}$.  
$$(\mathbf{A}{\boldsymbol{\delta}})\ \ \|\mathbb{E}_{{\mathcal{F}}_{0,s}}[\mathbf{X}_{s,t}]\|_{L^{p}}\leq|t-s|^{\delta}.$$

Remark 2.7. By the contraction property of the conditional expectation, the strongest form of (Aβ), (Aγ) and (Aδ) is obtained by setting Fs,t = σ(Xs,u, u ∈ [s, t]).

Theorem 2.8. Let k = maxI∈I |I| and, for m ≥ 2*, set* p = mk. Assume X is a canonical geometric stochastic process that satisfies one of the following:

$$(i)\ (\mathbf{A}\alpha)\,f o r\,\alpha>1/2;$$
(ii) (A$\alpha$), (A$\delta$) for $\alpha=1/2,\delta=1$;
(iii) (A$\alpha$), (A$\beta$) for $\alpha\in(1/3,1/2],\beta>1$;
(iv) (Aa), (A3), (A7) for 
$$\beta),(\mathbf{A}\gamma)\,f o r\,\alpha\in(1/4,1/3],\beta>1,\gamma>1;$$

with

$$\epsilon={\begin{cases}2\alpha-1,&{\mathrm{if}}\,(i),\\ (2\alpha-1/2)\land(\alpha+\delta-1),&{\mathrm{if}}\,(i i),\\ 3\alpha\land\beta-1,&{\mathrm{if}}\,(i i i),\\ 4\alpha\land\beta\land\gamma-1,&{\mathrm{if}}\,(i v),\end{cases}}$$
$$(9)$$
$$\square$$

and consider a signature-defining, cf. Definition 2.5, sequence of refining partitions {πn, n ≥ 1} *of the interval*
[0, T] *such that*

$$\sum_{n\geq1}|\pi_{n}|^{\epsilon}<\infty,$$

then the stronger convergence holds

$$S^{\bf I}(\mathbb{X}^{\pi_{n}})_{[0,T]}\stackrel{{L^{m}}}{{\rightarrow}}S^{\bf I}(\mathbb{X})_{[0,T]},\quad n\rightarrow\infty,\tag{10}$$  $$\mathbb{X}(\mathbb{X}_{n}\stackrel{{\rm i}}{{\rightarrow}}\mathbb{X}_{n})_{[0,T]}\stackrel{{L^{m}}}{{\rightarrow}}S^{\bf I}(\mathbb{X})_{[0,T]},\quad n\rightarrow\infty,\tag{10}$$
with rate O(Pn′≥n|πn′ |
ϵ).
Proof. See Appendix B.1.

Remark 2.9. Note that, if {πn, n ≥ 1} is a sequence of dyadic partitions with |πn| = 2−nT, then

$$\sum_{n\geq1}|\pi_{n}|^{\epsilon}=\sum_{n\geq1}2^{-n\epsilon}T^{\epsilon}={\frac{T^{\epsilon}}{1-2^{-\epsilon}}}<\infty,$$

and the rate of convergence is O(2−nϵ).

2.1.2. LONG-SPAN ASYMPTOTICS
Theorem 2.10. Fix T > 0 and let {Xt, t ≥ 0} *be a* stochastic process such that X
1 = {Xt, t ∈ [0, T]} satisfies the assumptions of Theorem 2.8 with m > 2*. Assume*
{X
n, n ≥ 1} *is stationary and ergodic and the sequence* of partitions {Π(N), N ≥ 1} *is such tha,t for each* n ≥ 1, π·,n = {πN,n, N ≥ n} is a signature-defining sequence of refining partitions, and

$$\sum_{N^{\prime}\geq N}|\Pi(N^{\prime})|^{\epsilon}\to0,\quad N\to\infty.$$

Then the expected signature estimator (7) is

1. _consistent, i.e. $\phi_{\bf I}^{\Pi(N)}(T)\stackrel{{L^{2}}}{{\rightarrow}}\phi_{\bf I}(T)$ as $N\rightarrow\infty$._
If, moreover, {X
n, n ≥ 1} is strongly mixing with mixing coefficient {α(n), n ≥ 1} *such that, for* ζ = m − 2 > 0,

$$\sum_{n\geq1}\alpha(n)^{\zeta/(2+\zeta)}<\infty,$$
and√NX
$$\sqrt{N}\sum_{N^{\prime}\geq N}|\Pi(N^{\prime})|^{\epsilon}\to0,\quad N\to\infty,\tag{13}$$

where ϵ is given in Equation (9), then the estimator is also 2. *asymptotically normal, i.e.*

$$\sqrt{N}\left(\hat{\phi}_{\mathbf{I}}^{\Pi(N)}(T)-\phi_{\mathbf{I}}(T)\right)\stackrel{\mathcal{L}}{\rightarrow}\mathcal{N}(0,\Sigma_{\mathbf{I}}),\quad N\rightarrow\infty,$$

as long as ΣI *is strictly positive definite, where*

$$\begin{array}{c}{{\Sigma_{\bf I}=\mathrm{Var}\left(S^{\bf I}(\mathbb{X}^{1})_{[0,T]}\right)}}\\ {{\qquad+2\sum_{n\geq2}\mathrm{Cov}\left(S^{\bf I}(\mathbb{X}^{1})_{[0,T]},S^{\bf I}(\mathbb{X}^{n})_{[0,T]}\right).}}\end{array}$$

Proof. See Appendix B.2.

Remark 2.11. If {Π(N), N ≥ 1} is a sequence of expanding dyadic refinements, i.e. for each n ≥ 1, π·,n is a sequence of dyadic partitions with |πN,n| = 2−N T, N ≥ n, as in Remark 2.9, then |Π(N)| = 2−N T and, hence,

$$\sqrt{N}\sum_{N^{\prime}\geq N}|\Pi(N^{\prime})|^{\epsilon}={\mathcal{O}}(\sqrt{N}2^{-\epsilon N})\to0,\quad N\to\infty.$$

Corollary 2.12. Assume the conditions of Theorem 2.10 hold with Theorem 2.8.(ii) satisfied for some m > 4 *and for* any T > 0*. Assume furthermore we can characterize the* rate of convergence of Theorem 2.10.1 as ρ(N) ∼ N −υfor some υ ∈ (0, 1)*. Then the kernel estimator*

$${\hat{\Sigma}}_{\bf I}^{\Pi(N)}=\sum_{|n|\leq h_{N}}{\hat{\Sigma}}_{\bf I}^{n,\Pi(N)},$$

with hN = Nυ/2*, non-overlapping cross-covariances*

$$\begin{array}{c}{{\hat{\Sigma}_{\bf I}^{n,\Pi(N)}=\frac{1}{M}\sum_{m=1}^{M}[S^{\bf I}(\mathbb{X}^{\pi_{N,(n+1)m-n}})_{[0,T]}-\hat{\phi}_{\bf I}^{\Pi(N)}(T)]}}\\ {{\qquad\qquad\qquad\times[S^{\bf I}(\mathbb{X}^{\pi_{N,(n+1)m}})_{[0,T]}-\hat{\phi}_{\bf I}^{\Pi(N)}(T)]_{\bf I}^{\sf T}}}\end{array}$$
$\square$
$${\mathit{t o r\,M=\lfloor N/(n+1)\rfloor\,a n d}}$$
$${\hat{\Sigma}}_{\mathbf{I}}^{-n,\Pi(N)}:=\left({\hat{\Sigma}}_{\mathbf{I}}^{n,\Pi(N)}\right)^{\mathrm{T}}\quad n=1,\ldots,N-1,$$
$$(11)^{\frac{1}{2}}$$

is consistent for ΣI*, i.e.* Σ
Π(N)
I
L
2
→ ΣI as N → ∞, and hence the CLT result of Theorem 2.10 can be made feasible. Proof. See Appendix B.3.

$$(12)$$

Requiring {X
n, n ≥ 1} to be stationary and ergodic or strongly mixing are high-level conditions. The following results give stronger but easier-to-interpret conditions.

Proposition 2.13. Fix T > 0 and let {Xt, t ≥ 0} be a stochastic process. Then3

$$\{\mathbf{X}_{t},\;t\geq0\}{\mathrm{~is~stationary}}$$

=⇒ {Xt, t ≥ 0} *has jointly stationary increments*
=⇒ {X
n, n ≥ 1} is stationary.

If any of the above holds, and X
1is a canonical geometric stochastic process, then, for any collection of words I,

$\{\mathbf{X}_{t},\,t\geq0\}$ is strongly mixing_  $\Longrightarrow\ \{\mathbf{X}^{n},\ n\geq1\}$ _is strongly mixing._
Proof. See Appendix B.4. One might expect a similar statement to hold for ergodicity, but Remark B.6 shows that

$\{{\bf X}_{t},\,t\geq0\}$ is ergodic $\ \Longrightarrow\ \{{\mathbb{X}}^{n},\ n\geq1\}$ is ergodic.  
Strong mixing implies ergodicity and hence the second part of Proposition 2.13 yields a sufficient condition (as far as
{X
n, n ≥ 1} is concerned) for both the consistency and asymptotic normality results of Theorem 2.10. Strong mixing is a somewhat restrictive assumption and hence one might wish to find a set of interpretable conditions weaker than strong mixing ensuring at least consistency of the estimator. The following theorem gives such a condition when
{Xt, t ≥ 0} is a Gaussian process.

3We say {Xt, t ≥ 0} has jointly stationary increments if for all n ∈ N, 0 ≤ si ≤ ti with i = 1*, . . . , n,* and t ≥ 0,

$$(\mathbf{X}_{s_{1},t_{1}},\ldots,\mathbf{X}_{s_{n},t_{n}})\stackrel{{\mathcal{L}}}{{=}}(\mathbf{X}_{t+s_{1},t+t_{1}},\ldots,\mathbf{X}_{t+s_{n},t+t_{n}}).\tag{14}$$

Theorem 2.14. Fix T > 0 and let {Xt, t ≥ 0} *be a* Gaussian process such that X = {Xt, t ∈ [0, T]} *is a* canonical geometric stochastic process satisfying4(Aα)
with α ≥ 1/2 and p = 2*. Assume the sequence of dyadic* partitions of [0, T] is signature-defining for X *and for each* N ≥ 1 let πN,n be the dyadic partition the interval [0, T]
with mesh |πN,n| = 2−N T.

Suppose {Xt, t ≥ 0} has constant mean and timehomogeneous increment covariance, i.e. ∀*u, v, s, t, r* ≥ 0

$$\operatorname{Cov}\left(\mathbf{X}_{u,v},\mathbf{X}_{s,t}\right)=\operatorname{Cov}\left(\mathbf{X}_{u+r,v+r},\mathbf{X}_{s+r,t+r}\right)$$

satisfying, for some decreasing θ : R+ → R+ *with* θ(t) →
0, t → ∞ and R T
0 θ(t)dt < ∞ and m ∈ N,

$$(\mathbf{A}\theta)\;\;\;\|\mathrm{Cov}\,(\mathbf{X}_{u,v},\mathbf{X}_{s,t})\,\|\lesssim\theta(|s-v|)|v-u||t-s|,$$

for all 0 ≤ u ≤ v < s ≤ t with |s−v| ≥ 
m 2
(|t−s|+|v−u|).

Then the expected signature estimator (7) *is consistent, i.e.*
ϕˆΠ(N)
I(T)
P

→ ϕI(T) as N → ∞.

Proof. See Appendix B.5.

## 2.2. Variance Reduction Via Martingale Correction

In Section 2.1 we developed the necessary theory to establish the asymptotic properties of the estimator (7) for the statistic ϕI (T) = E[S
I(X)[0,T]], for any word I =
(i1*, . . . , i*k). This section aims to find an alternative estimator with better finite sample properties when the process X
 = {Xt, t ∈ [0, T]} is a martingale. We restrict ourselves to the independent observation setting, with the same partition across samples, i.e. πN,n = π for n = 1*, . . . , N*. We will hence be considering the estimator

$$\hat{\phi}_{I}^{N,\pi}(T):=\frac{1}{N}\sum_{n=1}^{N}S^{I}(\mathbb{X}^{n,\pi})_{[0,T]},\tag{15}$$

where the X
n,π are i.i.d. piecewise linear observations of X
 over the partition5 π. We introduce the control-variate modification of the estimator (15),

$$\hat{\phi}_{I}^{N,\pi,c}(T):=\frac{1}{N}\sum_{n=1}^{N}\left(S^{I}(\mathbb{X}^{n,\pi})_{[0,T]}-cS_{c}^{I}(\mathbb{X}^{n,\pi})_{[0,T]}\right),\tag{16}$$  where $\hat{\phi}_{I}^{N,\pi,c}(T):=\frac{1}{N}\sum_{n=1}^{N}\left(S^{I}(\mathbb{X}^{n,\pi})_{[0,T]}-cS_{c}^{I}(\mathbb{X}^{n,\pi})_{[0,T]}\right),$
$$\mathbf{\Phi}_{1}:=(i_{1},\ldots,i_{k-1}),$$

where, setting I−1 := (i1*, . . . , i*k−1),

$$S_{c}^{I}(\mathbb{X}^{\pi})_{[0,T]}:=\sum_{[u,v]\in\pi}S^{I_{-1}}(\mathbb{X}^{\pi})_{[0,u]}X_{u,v}^{(i_{k})}.$$

The correction term S
Ic(X
π)[0,T]is inspired by considering the continuous-time signature

$$S^{I}(\mathbb{X})_{[0,T]}=\int_{0}^{T}S^{I-1}(\mathbb{X})_{[0,s]}\circ\mathrm{d}X_{s}^{(i_{k})},$$

where the integral is defined in the Stratonovich sense. To preserve the estimator's unbiasedness while reducing the variance we aim to find a mean-zero control variate S
I
c(X)[0,T]that is highly correlated with S
I(X)[0,T]. A natural candidate is

$$S_{c}^{I}(\mathbb{X})_{[0,T]}=\int_{0}^{T}S^{I-1}(\mathbb{X})_{[0,s]}\,\mathrm{d}X_{s}^{(i_{k})},$$

where the outermost integral is now interpreted in the Itoˆ
sense. If X is a square-integrable martingale satisfying the conditions of Jacod & Shiryaev (1987, Theorem I.4.40),
{S
I
c(X)[0,t], t ∈ [0, T]} is also a square-integrable martingale with E[S
I
c(X)[0,T]] = 0. Going back to the discretized setting, we note that, when X is a martingale, the discretized correction term S
Ic(X
π)[0,T]is also mean-zero and, hence, the control variate estimator ϕˆ*N,π,c* I(T) has the same bias as ϕˆN,π I(T), but, when picking the optimal6

$$c=c_{\pi}^{*}:=\frac{\mathrm{Cov}(S^{I}(\mathbb{X}^{\pi})_{[0,T]},S_{c}^{I}(\mathbb{X}^{\pi})_{[0,T]})}{\mathrm{Var}(S_{c}^{I}(\mathbb{X}^{\pi})_{[0,T]})},$$

it has reduced variance

$$\mathrm{Var}(\hat{\phi}_{I}^{N,\pi,c_{\pi}^{*}}(T))=(1-\rho_{I,\pi}^{2})\mathrm{Var}(\hat{\phi}_{I}^{N,\pi}(T)),$$
$${\mathrm{where}}\quad\rho_{I,\pi}:={\mathrm{Corr}}(S^{I}(\mathbb{X}^{\pi})_{[0,T]},S_{c}^{I}(\mathbb{X}^{\pi})_{[0,T]}).$$

In practice, to estimate c
∗π, the most straightforward approach would be to use the sample variance and covariance. In this case the estimator for c
∗
πis the slope of the simple linear regression of {S
I(X
n,π)[0,T], n = 1*, . . . , N*} against
{S
I
c(X
n,π)[0,T], n = 1*, . . . , N*} or, exploiting the mean zero property of the control,

$$\hat{c}_{\pi}^{*}=\frac{\sum_{n=1}^{N}S^{I}(\mathbb{X}^{n,\pi})_{[0,T]}S_{c}^{I}(\mathbb{X}^{n,\pi})_{[0,T]}}{\sum_{n=1}^{N}S_{c}^{I}(\mathbb{X}^{n,\pi})_{[0,T]}^{2}}.$$

In Appendix C.2 we propose an alternative estimator for c
∗π derived using the properties of the signature. Remark 2.15. This variance reduction technique is not limited to processes X that are *full* martingales but can also be applied to *partial* martingales, i.e. X such that only a subset of the components is a martingale. In this case, we can use the control variate expected signature estimator for any word I = (i1*, . . . , i*k) such that X
(ik)is a martingale.

6We assume throughout Var(S
I
c (X
π)[0,T ]) ∈ (0, ∞).

Even when the data generating process X is not a martingale, the variance reduction achieved by the corrected estimator (16) may outweigh the bias it introduces, leading to better performance - in terms of mean squared error (MSE) - than the classic estimator (15). In cases where the underlying process cannot be assumed to be a martingale we thus suggest to treat the martingale correction as a data transformation applicable in the learning pipeline (a model hyper-parameter in a similar spirit to the add-time or the lead-lag transform in the signature context) whose usefulness may be empirically ascertained via cross-validation.

## 3. Applications 3.1. Examples

We now consider a few concrete examples of continuoustime stochastic processes satisfying the assumptions of Theorem 2.10 and Theorem 2.14. Note that BM, CAR and Heston are semimartingales and hence, by Remark 2.2, they are canonical geometric stochastic processes such that any sequence of partitions with vanishing mesh size is signature defining. fBm is instead an example of a process that is not a semimartingale but is a canonical geometric stochastic process with dyadic signature-defining sequence of partitions
(Remark 2.2). Taking {Π(N), N ≥ 1} to be a sequence of expanding dyadic partitions thus ensures the observational assumptions of Theorem 2.10 and Theorem 2.14 are satisfied by all four processes, cf. Remark 2.11.

BM A standard Brownian motion {Bt, t ≥ 0}. It can be easily checked it satisfies (Aα) and (Aδ), for any α ≥ 1/2, δ ≥ 1 and p ≥ 2. Moreover, {Bt, t ≥ 0} has stationary and independent increments and, hence, the (ind) and (chop) sampling schemes are equivalent: in both cases we can apply7 Theorem 2.10 to deduce consistency and asymptotic normality of the expected signature estimator.

fBm A fractional Brownian motion {BH
t, t ≥ 0} with Hurst parameter H > 1/2. B
H satisfies (Aα) with α = H
(Appendix E.2.2) and, hence, Assumption 2.6 is fulfilled.

Under (ind) sampling, {B
H,n, n ≥ 1} is trivially stationary and strong mixing and, hence, we can apply Theorem 2.10.

When instead paths are obtained under (chop) we can apply8 Theorem 2.14, cf. Example E.2.2, to deduce consistency.

7Brownian motion is a Gaussian process with constant mean function and time-homogeneous covariance of the increments trivially satisfying (Aθ) with θ ≡ 0 and m = 0, it thus also falls under the scope of Theorem 2.14.

8The increments of fractional Brownian motion are not strongly mixing (Mandelbrot & Van Ness, 1968) and, hence, we cannot apply the second part of Theorem 2.10 to deduce asymptotic normality.

CAR A bidimensional Continuous-time Autoregressive
(CAR) process {Yt, t ≥ 0} of order p = 2 driven by a standard Brownian motion with drift A = (A1, A2) ∈
(R
2×2)
2. The CAR process is defined as the first d = 2 entries of its pd = 4-dimensional state space representation
{Xt, t ≥ 0}: an Ornstein-Uhlenbeck process with drift and diffusion

$$A_{\mathbf{A}}=\begin{pmatrix}0_{2\times2}&-I_{2\times2}\\ A_{2}&A_{1}\end{pmatrix},\quad\Sigma=\begin{pmatrix}0_{2\times2}&0_{2\times2}\\ 0_{2\times2}&I_{2\times2}\end{pmatrix},$$

(Lucchese et al., 2023; Marquardt & Stelzer, 2007). We can apply the first set of conditions in Appendix D.1.2 to deduce that {Xt, t ≥ 0} (and hence {Yt, t ≥ 0}) satisfies (Aα)
and (Aδ), for α = 1/2, δ = 1 and any p ≥ 2. Under (ind)
sampling we can hence apply Theorem 2.10. Moreover, when AA has positive real parts of all eigenvalues and the process is started in its stationary distribution, {Xt, t ≥ 0} and {Yt, t ≥ 0} are stationary, ergodic and strongly mixing with strong mixing coefficient α(t) = O(e
−at), for some a > 0 (Marquardt & Stelzer, 2007). We can hence apply Proposition 2.13 to deduce that {Y
n, n ≥ 1} is stationary and strongly mixing with strong mixing coefficient α(n) =
O(e
−anT ), i.e. satisfying Equation (12), for (any) ζ > 0.

Under (chop) sampling we can thus apply9the consistency and asymptotic normality results of Theorem 2.10.

Heston The joint price-variance dynamics of a Heston model under the risk-neutral measure Q with zero interest rate and no dividends, i.e. {(St, Vt), t ≥ 0} such that

$$\begin{array}{l}{{\mathrm{d}S_{t}=\sqrt{V_{t}}S_{t}\mathrm{d}W_{t}^{S},}}\\ {{\mathrm{d}V_{t}=\kappa(\theta-V_{t})\mathrm{d}t+\xi\sqrt{V_{t}}\mathrm{d}W_{t}^{V},}}\end{array}$$

where {WS
t, t ≥ 0} and {WV
t, t ≥ 0} are standard Brownian motions with correlation ⟨WS, WV⟩t = ρt. Under the Feller condition 2*κθ > ξ*2, the variance process is strictly positive (and so is {St, t ≥ 0}). The Heston model is thus an Ito diffusion with Lipschitz drift ˆ f : R+ ×R+ 7→ R
2and 1/2-Holder continuous diffusion ¨ σ : R+×R+ 7→ R
2×2. We can thus apply the third case of Appendix D.1.2 to prove that
{(St, Vt), t ≥ 0} satisfies (Aα) and (Aδ) with α = 1/2, δ = 1 and any p > 2 for deterministic initial conditions S0 = s0 and V0 = v0. When paths are sampled under (ind)
we can hence apply Theorem 2.10 to deduce consistency and asymptotic normality of the expected signature estimator.

## 3.2. Experiments

Quite a wide range of learning algorithms has been developed leveraging the properties of the expected signature. The theory for such algorithms is usually developed under 9The CAR process is a Gaussian process satisfying (Aθ), cf.

Appendix E.2.1, and hence also falls under the scope of Theorem 2.14.

7 the assumption of bounded variation paths for the input process X, assumed to be piecewise linear. The results in Section 2.1 give the theoretical foundation for their probabilistic interpretation when the underlying process X is an, arguably more realistic, continuous-time stochastic process such as the ones discussed in Section 3.1. In this section we review a few algorithms from the literature, showcasing the practical relevance of the asymptotic results of Section 2.1 and the potential improvements achieved by the martingale correction introduced in Section 2.2. Code and examples demonstrating the integration of the martingale correction into machine learning algorithms, along with the simulation results from the previous section, are available at https://github. com/lorenzolucchese/esig. The code is designed to be compatible with Python-based ML pipelines, supporting both numpy arrays and torch tensors.

## 3.2.1. Time Series Classification

The first model we consider, introduced in Triggiano & Romito (2024), falls under the general task of time series classification, mapping an input path x ∈ R
d×M1to a class label c ∈ C. The input stream is interpreted as a discretetime realization of a Gaussian process, whose conditional mean and covariance are learned parametrically. The expected signature of the latent Gaussian process, used as input to a classification layer, is estimated by super-sampling the process. Theorem 2.14 ensures this approach consistently estimates the expected signature of the latent continuous-time Gaussian process, a fundamental step for the probabilistic interpretation of the algorithm. We replicate the synthetic data experiments of Triggiano &
Romito (2024) on the (FBM), (OU) and (Bidim) datasets.

The performance on the out-of-sample testing datasets of the Gaussian Process augmented Expected Signature (GPES) classifier with and without martingale correction is reported in Table 1. The output of the GPES model is by construction stochastic and, hence, we repeat the evaluation of the model with 10 different seeds. In Table 1 we report the mean accuracy and standard error of the model with and without martingale correction (MC), as well as the results of an independent samples t-test between their accuracies. The martingale correction significantly improves the performance of the GPES model, a remarkable result considering that most processes in the three datasets are not martingales.

## 3.2.2. Pricing Path-Dependent Derivatives

The next application we consider is a purely financial one. The objective is to price (and hedge) path-dependent derivatives by decomposing them into a set of atomic Arrow-
Debreu-like securities. Let X = {Xt, t ∈ [0, T]} be a price process, i.e. a semimartingale over some probability space. In Lyons et al. (2021, Proposition 4.5) the authors use the

| Predictive Accuracy [%]   |              |              |              |
|---------------------------|--------------|--------------|--------------|
| FBM                       | OU           | Bidim        |              |
| GPES                      | 95.62 (0.18) | 62.20 (0.70) | 79.33 (0.46) |
| GPES-MC                   | 95.26 (0.70) | 88.26 (0.31) | 88.97 (0.44) |
| t-stat                    | 1.49         | −101.92      | −45.52       |
| p-value                   | 0.15         | 0.00         | 0.00         |

Table 1. Synthetic data experiments of Triggiano & Romito (2024): GPES model without and with martingale correction (MC).

universality of the signature to show that a large class of path-depend payoffs F can be arbitrarily well approximated by a linear payoff on the signature, i.e.

$$\operatorname{price}(F)=\mathbb{E}^{\mathbb{Q}}[Z_{T}F]\approx\langle f,Z_{T}\mathbb{E}^{\mathbb{Q}}[S({\hat{\mathbb{X}}}^{\mathrm{LL}})_{[0,T]}]\rangle,$$

for a set of linear coefficients f ∈ T((R
4)
∗) where Q is a pricing measure for X, ZT a deterministic discount factor over [0, T] and Xˆ LL denotes the add-time lead-lag transform of X. In Appendix F.2.2 we also discuss the corresponding hedging problem.

Given a pricing model Q for X, we can hence price F via Monte Carlo simulations. This provides a classic setting for applying the martingale correction described in Section 2.2 since, under Q, the (discounted) price process X is a martingale. In Figure 2, we compare the finite sample properties of the expected signature estimator with and without martingale correction when the price process is assumed to follow a Brownian motion (BM); in the context of option pricing, this is known as the Bachelier model. Similarly, in Figure 3, we plot the densities of the two estimators under the Heston dynamics10 (Heston). Both figures suggest the martingale correction (blue) materially improves the classic estimator (red), and hence more accurate pricing is achieved by the modified estimator introduced in Section 2.2.

0.5 1.0 I 
= (
0, 0)
N = 20 N = 40 N = 80 0.5 1.0 0.5 1.0
−0.5 0.0 0.5 I 
= (
1, 1, 0)
−0.5 0.0 0.5 −0.5 0.0 0.5 φ ˆΠ(N)
I (T) (iid) φ ˆΠ(N),c I (T) (iid) φI(T)
0.0 0.1 0.2 0.3 I 
=
 (0, 0)
N = 20 N = 40 N = 80 0.0 0.1 0.2 0.3 0.0 0.1 0.2 0.3
−0.001 0.000 0.001 I 
=
 (1
, 1
, 0)
−0.001 0.000 0.001 −0.001 0.000 0.001 φ ˆΠ(N)
I (T) (iid) φ ˆΠ(N),c I (T) (iid)

## 3.2.3. Distributional Regression For Streams

Introduced in Lemercier et al. (2021), the Signature of the pathwise Expected Signature (SES) model aims to learn a map from a collection of paths, understood as an empirical measure on path space, to a scalar value, a task known as distributional regression. Under appropriate conditions, the authors show that linear functionals on the signature of the pathwise expected signature are universal for weakly continuous functions (Lemercier et al., 2021, Theorem 3.2). We repeat two of the synthetic data experiments conducted in Lemercier et al. (2021), analyzing the performance of the SES model without and with martingale correction (MC). We report the average out-of-sample mean-squared error
(MSE) and its standard deviation in Table 2 and Table 3, as well as the t-statistic and p-value of a pairwise t-test between the MSEs of the two models. While the results do not yield statistical significance there still seems to be a mild benefit in using the martingale correction, especially considering that the processes of both experiments are not martingales11.

| Predictive MSE [×10−2 ]   |                     |             |
|---------------------------|---------------------|-------------|
| r1 = 0.35 × p3 V /N       | r2 = 0.65 × p3 V /N |             |
| SES                       | 1.27 (0.23)         | 0.09 (0.03) |
| SES-MC                    | 1.31 (0.45)         | 0.07 (0.02) |
| t-stat                    | −0.29               | 1.41        |
| p-value                   | 0.79                | 0.23        |

Table 2. Ideal gas experiment of Lemercier et al. (2021): SES model without and with martingale correction (MC).

Table 3. Rough volatility experiment of Lemercier et al. (2021): SES model without and with martingale correction (MC).

| Predictive MSE [×10−3 ]   |             |             |             |
|---------------------------|-------------|-------------|-------------|
| N = 20                    | N = 50      | N = 100     |             |
| SES                       | 1.49 (0.39) | 0.33 (0.13) | 0.20 (0.08) |
| SES-MC                    | 1.26 (0.48) | 0.31 (0.09) | 0.19 (0.05) |
| t-stat                    | 0.87        | 0.63        | 0.29        |
| p-value                   | 0.43        | 0.56        | 0.79        |

## 4. Conclusions

In this paper, we established new estimation results for the expected signature, a model-free embedding for collections of data streams. Our consistency and asymptotic normality results bridge the gap between the theoretically
"optimal" continuous-time expected signature and the empirical discrete-time estimator that can be computed from data. Moreover, we introduced a simple modification of such an estimator with significantly better finite sample properties under the assumption of martingale observations. Our empirical results suggest that the modified estimator might improve the performance of models employing expected signature computations even when the underlying data generating process is not necessarily a martingale.

## Acknowledgements

This research has been supported by the EPSRC Centre for Doctoral Training in Mathematics of Random Systems: Analysis, Modelling and Simulation (EP/S023925/1). The authors would like to thank Nicola Muca Cirone and Will Turner for helpful discussions on the topic, as well as the three anonymous reviewers for their insightful comments.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Friz, P. K., Hager, P. P., and Tapia, N. On expected signatures and signature cumulants in semimartingale models, 2024. URL https://arxiv.org/abs/2408. 05085.

Buehler, H., Murray, P., Pakkanen, M. S., and Wood, B.

Deep Hedging: Learning to Remove the Drift under Trading Frictions with Minimal Equivalent Near-Martingale Measures, 2022. URL https://arxiv.org/abs/ 2111.07844.

Futter, O., Horvath, B., and Wiese, M. Signature Trading: A Path-Dependent Extension of the Mean-Variance Framework with Exogenous Signals, 2023. URL https: //arxiv.org/abs/2308.15135.

Burkholder, D. L., Davis, B., and Gundy, R. F. Integral inequalities for convex functions of operators on martingales. In Le Cam, L. M., Neyman, J., and Scott, E. L. (eds.), Proceedings of the Sixth Berkeley Symposium on Mathematical Statistics and Probability, volume 2, pp. 223–240, Berkeley, California, 1972. University of California Press. URL https://projecteuclid. org/euclid.bsmsp/1200514221.

Graham, B. Sparse arrays of signatures for online character recognition, 2013. URL https://arxiv.org/ abs/1308.0371.

Hambly, B. and Lyons, T. Uniqueness for the signature of a path of bounded variation and the reduced path group. Annals of Mathematics, 171(1):109–167, 2005.

Chen, K.-T. Iterated Integrals and Exponential Homomorphisms. *Proceedings of the London Mathematical Society*, s3–4(1):502–512, 1954. doi: 10.1112/plms/s3-4.1.502.

Ibragimov, I. A. Some Limit Theorems for Stationary Processes. *Theory of Probability & Its Applications*, 7(4):
349–382, 1962. doi: 10.1137/1107036.

Chevyrev, I. and Lyons, T. Characteristic functions of measures on geometric rough paths. The Annals of Probability, 44(6):4049–4082, 2016. ISSN 00911798.

Isserlis, L. On a Formula for the Product-Moment Coefficient of any Order of a Normal Frequency Distribution in any Number of Variables. *Biometrika*, 12(1–2):134–139, 11 1918. ISSN 0006-3444. doi: 10.1093/biomet/12.1-2. 134.

Chevyrev, I. and Oberhauser, H. Signature moments to characterize laws of stochastic processes, 2018. URL
https://arxiv.org/abs/1810.10971.

Coutin, L. and Qian, Z. Stochastic analysis, rough path analysis and fractional Brownian motions. Probability Theory and Related Fields, 122:108–140, 01 2002. doi: 10.1007/s004400100158.

Jacod, J. and Shiryaev, A. N. Limit Theorems for Stochastic Processes, volume 288 of Grundlehren der mathematischen Wissenschaften. Springer Berlin, Heidelberg, 1987. ISBN 9783662025161. doi: 10.1007/ 978-3-662-02514-7.

Cuchiero, C., Svaluto-Ferro, S., and Teichmann, J. Signature SDEs from an affine and polynomial perspective, 2023. URL https://arxiv.org/abs/2302. 01362.

Kallenberg, O. *Foundations of Modern Probability*. Probability theory and stochastic modelling. Springer, 2021.

ISBN 9783030618728.

Dragomir, S. S. Some Gronwall Type Inequalities and Applications. Nova Science, New York, 2003.

Kiraly, F. J. and Oberhauser, H. Kernels for Sequentially Ordered Data. *Journal of Machine Learning Research*,
20(31):1–45, 2019.

Fawcett, T. Problems in stochastic analysis. Connections between rough paths and non-commutative harmonic analysis. PhD thesis, University of Oxford, 2003.

Kulik, A. *Ergodic Behavior of Markov Processes*. De Gruyter, Berlin, Boston, 2018. ISBN 9783110458930.

doi: 10.1515/9783110458930.

Friedman, A. Partial Differential Equations of Parabolic Type. Prentice-Hall, 1964.

Le, K. A stochastic sewing lemma and applications. ˆ Electronic Journal of Probability, 25:1 - 55, 2020. doi:
10.1214/20-EJP442.

Friz, P. K. and Victoir, N. B. Multidimensional Stochastic Processes as Rough Paths: Theory and Applications. Cambridge Studies in Advanced Mathematics. Cambridge University Press, 2010.

Lemercier, M., Salvi, C., Damoulas, T., Bonilla, E. V., and Lyons, T. Distribution regression for sequential data. In 24th International Conference on Artificial Intelligence and Statistics (AISTATS 2021), Proceedings of Machine Learning Research, pp. 3754–3762. Journal of Machine Learning Research, 2021.

Friz, P. K., Hager, P. P., and Tapia, N. Unified signature cumulants and generalized Magnus expansions. Forum of Mathematics, Sigma, 10:e42, 2022. doi: 10.1017/fms.

2022.20.

Levin, D., Lyons, T., and Ni, H. Learning from the past, predicting the statistics for the future, learning an evolving system, 2016. URL https://arxiv.org/abs/ 1309.0260.

Lucchese, L., Pakkanen, M. S., and Veraart, A. E. D.

Estimation and Inference for Multivariate Continuoustime Autoregressive Processes, 2023. URL https: //arxiv.org/abs/2307.13020.

Lyons, T. and McLeod, A. D. Signature Methods in Machine Learning, 2024. URL https://arxiv.org/abs/ 2206.14674.

Lyons, T., Caruana, M., and Levy, T. ´ Differential Equations Driven by Rough Paths: Ecole D' ´ et´ e de Probabilit ´ es de ´
Saint-Flour XXXIV-2004. Number no. 1908 in Differential Equations Driven by Rough Paths: Ecole D' ´ et´ e de ´
Probabilites de Saint-Flour XXXIV-2004. Springer, 2007. ´ ISBN 9783540712848.

Lyons, T., Nejad, S., and Perez Arribas, I. Non-parametric ´
pricing and hedging of exotic derivatives. Applied Mathematical Finance, 27(6):457–494, 2021.

Mandelbrot, B. B. and Van Ness, J. W. Fractional Brownian Motions, Fractional Noises and Applications. SIAM Review, 10(4):422–437, 1968. ISSN 00361445. URL
http://www.jstor.org/stable/2027184.

Marquardt, T. and Stelzer, R. Multivariate CARMA processes. *Stochastic Processes and their Applications*, 117 (1):96–120, Jan 2007. ISSN 03044149. doi: 10.1016/j. spa.2006.05.014.

Newey, W. K. and West, K. D. A Simple, Positive Semi-
Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3):703–708, 1987. ISSN 00129682, 14680262.

Ni, H. *The expected signature of a stochastic process*. PhD
thesis, University of Oxford, 2012.

Passeggeri, R. On the signature and cubature of the fractional Brownian motion for H > 1/2. Stochastic Processes and their Applications, 130(3):1226–1257, 2020. ISSN 0304-4149. doi: 10.1016/j.spa.2019.04.013.

Perez Arribas, I., Goodwin, G. M., Geddes, J. R., Lyons, ´
T., and Saunders, K. E. A. A signature-based machine learning model for distinguishing bipolar disorder and borderline personality disorder. *Translational Psychiatry*, 8, 2018.

Salvi, C., Cass, T., Foster, J., Lyons, T., and Yang, W. The Signature Kernel Is the Solution of a Goursat PDE. SIAM
Journal on Mathematics of Data Science, 3(3):873–899, 2021. doi: 10.1137/20M1366794.

Schell, A. and Oberhauser, H. Nonlinear independent component analysis for discrete-time and continuous-time signals. *Annals of Statistics*, 51(2):487–518, 2023.

Triggiano, F. and Romito, M. Gaussian Processes Based Data Augmentation and Expected Signature for Time Series Classification. *IEEE Access*, 12:80884–80895, 2024. doi: 10.1109/ACCESS.2024.3408712.

Willett, D. W. Nonlinear vector integral equations as contraction mappings. Archive for Rational Mechanics and Analysis, 15:79–86, 1964. doi: 10.1007/bf00257405.

Xie, Z., Sun, Z., Jin, L., Ni, H., and Lyons, T. Learning Spatial-Semantic Context with Fully Convolutional Recurrent Network for Online Handwritten Chinese Text Recognition. *IEEE Transactions on Pattern Analysis* and Machine Intelligence, 40(8):1903–1917, 2018. doi:
10.1109/TPAMI.2017.2732978.

## Contents Of The Appendix

A Informal Glossary B Proofs of Section 2 B.1 Proof of Theorem 2.8 B.1.1 Proof of Theorem 2.8 under (i), (iii) or *(iv)* B.1.2 Proof of Theorem 2.8 under *(ii)*
B.2 Proof of Theorem 2.10 B.2.1 Proof of Theorem 2.10, consistency B.2.2 Proof of Theorem 2.10, asymptotic normality B.3 Proof of Corollary 2.12 B.4 Proof of Proposition 2.13 B.4.1 Proof of Proposition 2.13, stationary implications B.4.2 Proof of Proposition 2.13, strong mixing implications B.5 Proof of Theorem 2.14 C Variance Reduction via Martingale Correction C.1 Martingale Continuity Criterion C.2 Estimating c
∗π C.3 Proof of Lemma C.1 D Ito processes and diffusions ˆ
D.1 In-fill conditions D.1.1 Ito processes ˆ D.1.2 Ito diffusions ˆ
D.2 Long span conditions D.2.1 Ito diffusions ˆ
E Gaussian Processes E.1 Gaussian Processes Continuity Criterion E.2 Gaussian Processes Covariance Decay Condition E.2.1 Ornstein-Uhlenbeck Process E.2.2 Fractional Brownian Motion F Machine Learning Algorithms with Expected Signatures F.1 Martingale Correction in Applications F.2 Algorithms F.2.1 Time Series Classification (Triggiano & Romito, 2024) F.2.2 Pricing Path-Dependent Derivatives (Lyons et al., 2021) F.2.3 Distributional Regression for Streams (Lemercier et al., 2021) F.2.4 Systematic Trading (Futter et al., 2023)
G Controlled Linear Regression G.1 Controlled Ordinary Least Squares (OLS) estimation G.2 Simulation study

## A. Informal Glossary

This informal glossary provides accessible explanations of selected technical terms and notational conventions used in this paper, aimed at readers with little or no background in rough path theory. These intuitive definitions are intended to aid the understanding of the theoretical framework presented in Section 2, particularly Definition 2.1. However, they remain closely tied to more technical definitions - such as those of multiplicative functionals, rough paths, and geometric rough paths - which require a more rigorous exposition of rough path theory. For a concise introduction to rough paths, we refer the reader to Lyons et al. (2007), and for a treatment in the stochastic setting, to Friz & Victoir (2010).

p**-variation** The p-variation of a path is a measure of its regularity. For the purpose of our discussion it suffices to note that paths that have finite p-variation for low p are more regular. A bounded variation (BV) path is a path with finite 1-variation
(also known as total variation). This regularity ensures there exists a well-defined notion of integral against this path (e.g. a piecewise linear paths or continuously differentiable path) and, hence, we can easily define its signature as in Equation (2).

Many interesting stochastic processes (e.g. those driven by Brownian motion) have infinite 1-variation (i.e. are not BV) but have finite p-variation for all p > 2 and, hence, defining their signature requires rough path theory. Convergence in p**-variation** Convergence in the p-variation metric/topology is a pathwise mode of convergence (i.e. over all points t ∈ [0, T] simultaneously) that is (much) stronger than the pointwise (i.e. at fixed t ∈ [0, T]) convergence required to state and prove our results. See, for example, Remark 2.4.

Spaces of paths We denote by C([0, T], R
d), resp. BV([0, T], R
d), the space of R
d-valued continuous, resp. bounded variation, paths over the interval [0, T].

Mesh of a partition For a partition π = {0 = t0 < t1 *< . . . < T*} of [0, T], we define its mesh as |π| = max[s,t]∈π |t−s| where the maximum is taken over all sub-intervals of the partition. Shuffle property The shuffle property of the signature is an algebraic property stating that the product of two signature terms is a linear combination of higher-order signature terms. More precisely, the product of the signature terms corresponding to words I and J is the sum of all signature terms indexed by words K of length |I| + |J| obtained by interleaving I and J. In the context of the discussion on page 1 this means that all moments of the signature can be written as linear combinations of higher order expected signature terms.

Signature indexing A word I = (i1*, . . . , i*n) with i1, . . . , in ∈ {1*, . . . , d*} is a multi-index used to denote an entry of the signature, i.e. a real-valued number. The length of the word, i.e. |I| = n, denotes the signature level, i.e. an n-dimensional tensor, to which such entry belongs. For example S
I(X)[0,T], where I = (1, 2), denotes the (1, 2)-entry of the second level of the signature (a matrix), while I = (2, 1, 1) denotes the (2, 1, 1)-entry of the third level of the signature (a three-dimensional tensor).

Stochastic processes A continuous stochastic process X = {Xt, t ∈ [0, T]} over a probability space (Ω, F, P) is such that, for each ω ∈ Ω, the realization X(ω) = {Xt(ω), t ∈ [0, T]} ∈ C([0, T], R
d). If one takes Ω = C([0, T], R
d) and P a probability measure over this path space then each ω ∈ Ω denotes a possible path realization of X. We thus say a property holds pathwise or almost surely if the set of ω ∈ Ω for which that property holds has probability one.

Canonical geometric stochastic process We define a canonical geometric stochastic process as a continuous stochastic process whose "higher order structure" can be approximated by the iterated integrals of its piecewise-linear interpolations (in probability in the p-variation metric). Its signature is then defined as the limit of the signatures of its piecewise-linear interpolations, i.e. the iterated integrals given in Equation (3). For clarity, we emphasize that *canonicity* here refers to the aforementioned construction of the signature, not to the underlying probability space on which the process is defined.

## B. Proofs Of Section 2 B.1. Proof Of Theorem 2.8

Sketch of proof. *The main idea of the proof is to show the sequence of discretized signatures* {S
k(X
πn )[0,T ], n ≥ 1} is Cauchy in L
m*. Since* L
m *is a Banach space this implies the sequence converges in* L
m. By uniqueness of limits, we can deduce this limit is the same as its P*-limit, i.e.* S
k(X)[0,T ]*. To show the sequence is Cauchy in* L
m we proceed inductively on the signature level k
′ ∈ {1, . . . , k} *under the progressively weaker norm* L
mk/k′*. The main ingredient of the inductive step is* a manipulation of the discrete-time signature (2)*, ensuring* S
k
′(X
πn+1 )[τ0,τ1] − S
k
′(X
πn)[τ0,τ1], [τ0, τ1] ⊆ [0, T],
can be written as a sum over time intervals πn+1,[τ0,τ1]*. The inductive assumption is then verified by bounding this summation* using Lemma B.1 when a simple Minkowski bound is too weak. We use two different manipulations of the discrete-time signature under assumptions (i), (iii) or (iv) and under assumption (ii): In the former case we use the classic representation given in (2), while in the latter we rely on the "causal" representation of Lemma *B.3. For clarity of exposition we thus divide* the proof of Theorem 2.8 *into two parts.*
We first establish a couple of useful lemmas which will be used repeatedly in the proof of this in-fill asymptotic results. The first is a basic result which is also applied in the proof of the stochastic sewing lemma (Leˆ, 2020). In the following, let E denote a Banach space.

Lemma B.1. Let {Zn, n = 1, . . . , N} be a finite sequence of E*-valued random variables in* L
m with m ∈ [2, ∞) *and let*
{Gn, n = 1, . . . , N} be a filtration such that, for each n ∈ {1, . . . , N}, the variables Z1, . . . , Zn−1 are Gn-measurable.

Then

$$\left\|\sum_{n=1}^{N}Z_{n}\right\|_{L^{m}}\leq\sum_{n=1}^{N}\|\mathbb{E}_{\mathcal{G}_{n}}[Z_{n}]\|_{L^{m}}+2C_{m}\left(\sum_{n=1}^{N}\|Z_{n}\|_{L^{m}}^{2}\right)^{1/2}$$
.
Proof.

$$\left\|\sum_{n=1}^{N}Z_{n}\right\|_{L^{m n}}$$
(i) ≤  X N n=1 EGn [Zn] Lm +  X N n=1 (Zn − EGn [Zn]) Lm n=1 ∥Zn − EGn [Zn]∥ 2  1/2 (ii) ≤X N n=1 ∥EGn [Zn]∥Lm + Cm  X N Lm/2 n=1 ∥Zn − EGn[Zn]∥ 2Lm !1/2 (iii) ≤X N n=1 ∥EGn[Zn]∥Lm + Cm  X N n=1 (∥Zn∥Lm + ∥EGn[Zn]∥Lm) 2 !1/2 (iv) ≤X N n=1 ∥EGn[Zn]∥Lm + Cm  X N n=1 ∥Zn∥ 2 Lm !1/2 , (v) ≤ X N n=1 ∥EGn[Zn]∥Lm + 2Cm  X N

by using in (i) the triangle inequality, in (ii) triangle inequality and the Burkholder-Davis-Gundy (BDG) inequality
(Burkholder et al., 1972) applied to the martingale {Mn, n = 1*, . . . , N*} with Mn =Pn i=1(Zi − EGi[Zi]), in (iii) and
(iv) the triangle inequality and in (v) the contraction property of conditional expectation.

Lemma B.2. Let p, p1, . . . , pl ∈ (0, ∞) ∪ {+∞} *be such that* p
−1 1 + *. . .* + p
−1 l = p
−1*, then, for any set of tensors* A1 ∈ L
p1 ((R
d)
⊗k1 )*, . . . ,* Al ∈ L
pl ((R
d)
⊗kl ),

$\|\mathbf{A}_{1}\otimes\cdots\otimes\mathbf{A}_{l}\|_{L^{p}}\leq d^{l}\left\|\mathbf{A}_{1}\right\|_{L^{p_{1}}}\cdots\left\|\mathbf{A}_{l}\right\|_{L^{p_{l}}}$
Proof.

$\|\mathbf{A}_{1}\otimes\cdots\otimes\mathbf{A}_{l}\|_{L^{p}}\leq\sum_{(w_{1},...,w_{l})\in\mathcal{W}_{k_{1}+...+k_{l}}}\|A_{1}^{w_{1}}\cdots A_{l}^{w_{l}}\|_{L^{p}}$  $$\stackrel{{(*)}}{{\leq}}\sum_{(w_{1},...,w_{l})\in\mathcal{W}_{k_{1}+...+k_{l}}}\|A_{1}^{w_{1}}\|_{L^{p_{1}}}\cdots\|A_{l}^{w_{l}}\|_{L^{p_{l}}}$$
14

≤ d
$$+k_{l}\parallel\mathbf{A}_{1}\parallel_{L^{p_{1}}}\cdot\cdot\cdot\parallel\mathbf{A}_{l}\parallel_{L^{p_{l}}},$$
where Wk = {1*, . . . , d*}
k denotes the set of words of length k and, in (∗), we applied the classical Holder inequality. ¨
We also prove a useful lemma that allows us to write the k-th level signature of a piecewise linear path as a "causal" sum of lower order signature terms, i.e. preserving time order. This will allow us to derive an in-fill result with assumptions on the regularity of EF0,s[Xs,t], a more natural object than EF0,s∨Ft,T [Xs,u ⊗ Xu,t], when α = 1/2, i.e. Theorem 2.8 under *(ii)*.

Lemma B.3. Let π be a partition of [0, T] and let τ ∈ π. Then, for k ≥ 0*, we can write*

$$S^{k+1}(\mathbb{X}^{\pi})_{[0,\tau]}=\sum_{i=0}^{k}\frac{1}{(1+i)!}\sum_{[u,v]\in\pi_{[0,\tau]}}S^{k-i}(\mathbb{X}^{\pi})_{[0,u]}\otimes\mathbf{X}_{u,v}^{\otimes(i+1)}.$$

Proof. Note that for k ≥ 0,

S k+1(X π)[0,τ] =X [u,v]∈π[0,τ] -S k+1(X π)[0,v] − S k+1(X π)[0,u]  [u,v]∈π[0,τ] "kX +1 i=0 S k+1−i(X π)[0,u] ⊗ X⊗i u,v i!− S k+1(X π)[0,u] # (∗) =X =X [u,v]∈π[0,τ] k X +1 i=1 S k+1−i(X π)[0,u] ⊗ X⊗i u,v i! = X k 1 (1 + i)! X [u,v]∈π[0,τ] S k−i(X π)[0,u] ⊗ X⊗(1+i) u,v , i=0
$$\square$$
where, in (∗), we use Chen's relation and S(X
π)[u,v] = exp⊗ Xu,v since X
πis linear over [*u, v*] ∈ π.

B.1.1. PROOF OF THEOREM 2.8 UNDER (i), (iii) OR *(iv)*
Denote by {πn, n ≥ 1} the signature-defining sequence of refining partitions of the interval [0, T]. Without loss of generality, we can consider {πn, n ≥ 1} to be such that πn+1 is obtained from πn by adding at most one refinement in each sub-interval, i.e., for each [*s, t*] ∈ πn, either [s, t] ∈ πn+1 or [*s, u*], [u, t] ∈ πn+1, for u ∈ (*s, t*). If not, one can consider a super-sequence satisfying this property and then pass to the original subsequence.

In the following, for any n ≥ 1 and [*s, t*] ∈ πn, denote by πn,[s,t]the restriction of πn to [*s, t*] and, abusing notation slightly, S(X
πn )[s,t] = S(X
πn,[s,t] )[s,t].

Let [τ0, τ1] ∈ πN , for N ≥ 1, and note that, for any k ≥ 2 and n ≥ N, we can write

$$S^{k}(\mathbb{X}^{\pi_{n+1}})_{[\tau_{0},\tau_{1}]}-S^{k}(\mathbb{X}^{\pi_{n}})_{[\tau_{0},\tau_{1}]}=\sum_{[\tau_{n}]\in(\pi_{n},[\tau_{0},\tau_{1}]}[S^{k}(\mathbb{X}^{\pi_{n},\tau_{1}})_{[\tau_{0},\tau_{1}]}-S^{k}(\mathbb{X}^{\pi_{n},\tau_{1}})_{[\tau_{0},\tau_{1}]}]\;,\tag{17}$$

where the partitions πn,s are defined as πn,s = πn+1,[0,s] ∪ πn,[s,T], i.e., for each [*s, t*] ∈ πn, the partitions πn,s and πn,t differ by at most one point u ∈ (*s, t*). Using Chen's relation and the definition of the tensor product, we can write for each [s, t] ∈ πn with refinement u ∈ (s, t),

$$S^{k}(\mathbb{X}^{\pi_{n,s}})_{[\pi_{n},\pi_{1}]}-S^{k}(\mathbb{X}^{\pi_{n,s}})_{[\pi_{n},\pi_{1}]}\\ =\sum_{\begin{subarray}{c}i_{1},i_{2},i_{3}\geq0\\ i_{1}+i_{2}+i_{3}=k\end{subarray}}S^{i_{1}}(\mathbb{X}^{\pi_{n+1}})_{[\pi_{n},s]}\otimes\left[S^{i_{2}}(\mathbb{X}^{\pi_{n+1}})_{[\pi,s]}-S^{i_{2}}(\mathbb{X}^{\pi_{n}})_{[\pi,s]}\right]\otimes S^{i_{3}}(\mathbb{X}^{\pi_{n}})_{[\pi,\pi_{1}]}.\tag{18}$$
Note that, for i2 ∈ {0, 1},S
$$S^{i_{2}}(\Im^{\pi_{n+1}})_{[s,t]}-S^{i_{2}}(\Im^{\pi_{n}})_{[s,t]}=0,$$

15 and applying again Chen's relation when i2 ≥ 2 yields

$$S^{i_{2}}(\mathbb{X}^{\pi_{n+1}})_{[s,t]}=\sum_{j=0}^{i_{2}}S^{j}(\mathbb{X}^{\pi_{n+1}})_{[s,u]}\otimes S^{i_{2}-j}(\mathbb{X}^{\pi_{n+1}})_{[u,t]}=\frac{1}{i_{2}!}\sum_{j=0}^{i_{2}}{\binom{i_{2}}{j}}\mathbb{X}^{\otimes j}_{s,u}\otimes\mathbb{X}^{\otimes(i_{2}-j)},$$

where we used the fact that, if Y is linear over [*s, t*], then S(Y)[s,t] = exp⊗ Ys,t, which also implies

$$S^{i_{2}}(\mathbf{X}^{\pi_{n}})_{[s,t]}={\frac{\mathbf{X}_{s,t}^{\otimes i_{2}}}{i_{2}!}}={\frac{(\mathbf{X}_{s,u}+\mathbf{X}_{u,t})^{\otimes i_{2}}}{i_{2}!}}={\frac{1}{i_{2}!}}\sum_{\mathcal{I}\in\{0,1\}^{i_{2}}}\bigotimes_{i\in\mathcal{I}}\left(\mathbf{X}_{s,u}^{\otimes i}\otimes\mathbf{X}_{u,t}^{\otimes(1-i)}\right),$$

denoting by *I ∈ {*0, 1}
i2 a binary number of length i2 with |I| =Pi∈I i and recalling that x
⊗0 = 1, x
⊗1 = x, for any x ∈ R
d. We hence have that

$$S^{i_{2}}(\mathbb{X}^{\pi_{n+1}})_{[s,t]}-S^{i_{2}}(\mathbb{X}^{\pi_{n}})_{[s,t]}=\sum_{\mathcal{I}\in\{0,1\}^{i_{2}}}C_{\mathcal{I}}\bigotimes_{i\in\mathcal{I}}\left(\mathbf{X}_{s,u}^{\otimes i}\otimes\mathbf{X}_{u,t}^{\otimes(1-i)}\right),$$

where for *I ∈ {*0, 1}
i2,

$$C_{\mathcal{I}}=\begin{cases}\frac{1}{i_{2}!}\left[\left(\begin{matrix}i_{2}\\ |\mathcal{I}|\end{matrix}\right)-1\right],&\text{if}\mathcal{I}=(1,\ldots,1,0,\ldots,0),\\ -\frac{1}{i_{2}!},&\text{otherwise.}\end{cases}$$

Plugging this into Equation (17) via (18) and noting that CI = 0 for I ∈ {(0, . . . , 0),(1*, . . . ,* 1)}, we can write, for any N ≥ 1, [τ0, τ1] ∈ πN , n ≥ N and k ≥ 2,

S k(X πn+1 )[τ0,τ1] − S k(X πn )[τ0,τ1] I∈{0,1} i2 I̸=(0,...,0),(1,...,1) CI S i1(X πn+1 )[τ0,s] ⊗O i∈I X⊗i s,u ⊗ X ⊗(1−i) u,t  ⊗ S i3(X πn )[t,τ1]. =X X X i1,i3≥0,i2≥2 i1+i2+i3=k [s,t]∈πn,[τ0,τ1] u∈(s,t)
We now proceed inductively to show that, for any i ∈ {1*, . . . , k*} and any [τ0, τ1] ∈ πN with N ≥ 1, the sequence
{S
i(X
πn )[τ0,τ1], n ≥ N} converges in L
mk/i with rate O(Pn′≥n|πn′ | ϵ) and

$$\sup_{N\geq1}\sup_{[\tau_{0},\tau_{1}]\in\pi_{N}}\|S^{i}(\mathbb{X}^{\pi_{N}})_{[\tau_{0},\tau_{1}]}\|_{L^{mk/i}}<\infty.\tag{19}$$

k
′ = 1. Note that for [τ0, τ1] ∈ πN with N ≥ 1 one has S
1(X
πn )[τ0,τ1] = Xτ0,τ1
, for all n ≥ N, and

$$\|\mathbf{X}_{\tau_{0},\tau_{1}}\|_{L^{m k}}\lesssim|\tau_{1}-\tau_{0}|^{\alpha}\leq T^{\alpha}<\infty,$$

by Assumption (Aα). Hence S
1(X)[0,T] = X0,T ∈ L
mk and the statement holds trivially.

Assume the inductive hypothesis holds for all i ∈ {1*, . . . , k*′} with k
′ ∈ {1*, . . . , k* − 1}. Then, for each [τ0, τ1] ∈ πN with
N ≥ 1 and n ≥ N, let
S k ′+1(X πn+1 )[τ0,τ1]−S k ′+1(X πn )[τ0,τ1] Lmk/(k′+1) I∈{0,1} i2 I̸=(0,...,0),(1,...,1) |CI| X [s,t]∈πn,[τ0,τ1] u∈(s,t) Z I [s,t] Lmk/(k′+1) , (20) ≤X i1,i3≥0,i2≥2 i1+i2+i3=k ′+1 X
where, for each [τ0, τ1] ∈ πN , πn with n ≥ N, i1, i3 ≥ 0, i2 ≥ 2 with i1 + i2 + i3 = k
′ + 1 and *I ∈ {*0, 1}
i2, we define

$Z^{\mathbb{T}}_{[s,t]}:=S^{i_{1}}(\mathbb{X}^{\pi_{n+1}})_{[\eta_{n},t]}\otimes\bigotimes_{t\in\mathbb{T}}\left(\mathbb{X}^{\otimes i}_{s,u}\otimes\mathbb{X}^{\otimes(1-u)}_{u,t}\right)\otimes S^{i_{2}}(\mathbb{X}^{\pi_{n}})_{[t,\tau_{1}]},\quad[s,t]\in\pi_{n,[\eta_{n},\tau_{1}]}$ with $u\in(s,t)$,
keeping only the dependence on I for notational convenience. Note that, by applying Lemma B.2, the inductive hypothesis and Assumption (Aα),

$\|Z^{T}_{[s,t]}\|_{L^{\infty,1/(s^{2}+1)}}\lesssim\|S^{s_{t}}(\mathbb{X}^{\pi_{n+1}})_{[\eta,s]}\|_{L^{\infty,1/s_{1}}}\|\mathbb{X}_{s,u}\|_{L^{\infty,1}}^{21}\|\mathbb{X}_{u,s}\|_{L^{\infty,1}}^{s_{t}-1/2}\|S^{s_{u}}(\mathbb{X}^{\pi_{n}})_{[t,\tau]}\|_{L^{\infty,1/s_{1}}}\lesssim|t-s|^{1/2},$
and, hence, each Z
I
[s,t] ∈ L
mk/(k
′+1). Moreover, by a simple application of the triangle inequality,

$$\left\|\sum_{\begin{subarray}{c}t\in\pi_{+},\,t\in\tau_{1},\,\tau_{1}\\ u\in(s,t)\end{subarray}}Z_{[s,t]}^{T}\right\|_{L^{m,k/(k^{\prime}+1)}}\leq\sum_{\begin{subarray}{c}[s,t]\in\pi_{+},\,\tau_{0},\,\tau_{1}\\ u\in(s,t)\end{subarray}}|t-s|^{i\,2\,\alpha}.$$
$$(21)$$
$$(22)$$

Assumption (i) Hence, if α > 1/2, we have for each [τ0, τ1] ∈ πN , πn with n ≥ N, i1, i3 ≥ 0, i2 ≥ 2 with i1 + i2 + i3 = k
′ + 1 and *I ∈ {*0, 1}
i2,

$$\left\|\sum_{\begin{subarray}{c}[s,t]\in\pi_{+10},\ \tau_{1}]\\ u\in(s,t)\end{subarray}}Z_{[s,t]}^{\tau}\right\|_{L^{m,k}/(k^{\prime}+1)}\lesssim|\pi_{n}|^{2\alpha-1}|\tau_{1}-\tau_{0}|.$$

Assumption (iii) If α ∈ (1/3, 1/2] note that if I is such that i2 ≥ 3, then

$$\left\|\sum_{\begin{subarray}{c}[s,t]\in\pi_{n},[\tau_{0},\tau_{1}]\\ u\in(s,t)\end{subarray}}Z_{[s,t]}^{T}\right\|_{L^{m,k}/(k^{\prime}+1)}\leq|\pi_{n}|^{3\alpha-1}|\tau_{1}-\tau_{0}|,$$

but if i2 = 2 then the bound (21) is not strong enough. We can instead apply Lemma B.1 with the filtration {G[s,t], [*s, t*] ∈ πn} defined by G[s,t]:= Fs ∨ σ(Xv,w, [v, w] ∈ πn,[t,τ]),
by noting that each Z
I
[v,w] with w ≤ s is G[s,t]-measurable and mk/(k
′ + 1) ≥ 2 for all k
′ + 1 ≤ k. This implies

$$(23)$$
∥Z I [s,t]∥ 2 Lmk/(k′+1) !1/2 X [s,t]∈πn,[τ0,τ1] u∈(s,t) Z I [s,t] Lmk/(k′+1) ≤X [s,t]∈πn,[τ0,τ1] u∈(s,t) EG[s,t] [Z I [s,t]] Lmk/(k′+1) +  X [s,t]∈πn,[τ0,τ1] u∈(s,t) |t − s| 4α !1/2 |t − s| β +  X ≤X [s,t]∈πn,[τ0,τ1] u∈(s,t) [s,t]∈πn,[τ0,τ1] u∈(s,t) ≤ |πn| (β−1)∧(2α−1/2) |τ1 − τ0| + |τ1 − τ0| 1/2, (24)
$\mathbf{u}\cdot\mathbf{u}$
where we used the fact that for *I ∈ {*(0, 1),(1, 0)},

EG[s,t]
[Z
I [s,t] ] Lmk/(k′+1) (i) =  S i1(X πn+1 )[τ0,s] ⊗ EG[s,t] "O i∈I X⊗i s,u ⊗ X ⊗(1−i) u,t  #⊗ S i3(X πn )[t,τ1] Lmk/(k′+1) (ii) ≤S i1(X πn+1 )[τ0,s] Lmk/i1  EG[s,t] "O i∈I X⊗i s,u ⊗ X ⊗(1−i) u,t  #Lmk/2 S i3(X πn )[t,τ1] Lmk/i3 (iii) ≲ EG[s,t] [Xs,u ⊗ Xu,t]Lmk/2 (iv) ≲ EF0,s∨Ft,T [Xs,u ⊗ Xu,t]Lmk/2 17
(v)
≲ |t − s| β, by using in (i) measurability of S
i1 (X
πn+1 )[τ0,s] and S
i3 (X
πn )[t,τ1] with respect to G[s,t], in (ii) Holder inequality for ¨
tensors Lemma B.2, in (iii) the inductive assumption (19) and the fact that ∥A ⊗ B∥ = ∥B ⊗ A∥ for any A, B ∈ R
d, in
(iv) the tower property and the contractive property of conditional expectation applied to G[s,t] ⊆ F0,s ∨ Ft,T and in (v)
Assumption (Aβ). Combining bound (24) when i2 = 2 and bound (23) when i2 ≥ 3 with α ∈ (1/3, 1/2], it follows that for each [τ0, τ1] ∈ πN , πn with n ≥ N, i1, i3 ≥ 0, i2 ≥ 2 with i1 + i2 + i3 = k
′ + 1 and *I ∈ {*0, 1}
i2,

$$\left\|\sum_{\begin{subarray}{c}\left[s,t\right]\in\pi_{n,1}\cap\pi_{1}\\ u\in\left(s,t\right)\end{subarray}}Z_{\left[\pi,t\right]}^{T}\right\|_{L^{m,k}/\left(L^{s}+1\right)}\lesssim|\pi_{n}|^{(\beta-1)\wedge(3\alpha-1)}|\tau_{1}-\tau_{0}|.\tag{25}$$

Assumption (iv) A similar reasoning can be applied when α ∈ (1/4, 1/3] (and k ≥ 3) by considering the cases i2 ≥ 4, i2 = 3 and i2 = 2 separately. The case i2 ≥ 4 follows directly from (21), the case i2 = 2 follows from (24) and the case i2 = 3 can be shown in the same way as i2 = 2 with the only difference being that we require Assumption (Aγ) to show that, for *I ∈ {*(0, 0, 1),(0, 1, 0),(1, 0, 0)},

$$\left\|\mathbb{E}_{\mathcal{G}_{[s,t]}}[Z_{[s,t]}^{\mathcal{I}}]\right\|_{L^{m k/(k^{\prime}+1)}}\lesssim\left\|\mathbb{E}_{\mathcal{F}_{0,s}\lor\mathcal{F}_{t,T}}\left[\mathbf{X}_{s,u}\otimes\mathbf{X}_{u,t}^{\otimes2}\right]\right\|_{L^{m k/3}}\lesssim|t-s|^{\gamma},$$

and, for *I ∈ {*(0, 1, 1),(1, 0, 1),(1, 1, 0)},

$$\left\|\mathbb{E}_{\mathcal{G}_{[s,t]}}[Z_{[s,t]}^{T}]\right\|_{L^{m k/(k^{\prime}+1)}}\lesssim\left\|\mathbb{E}_{\mathcal{F}_{0,s}\lor\mathcal{F}_{t,T}}\left[\mathbf{X}_{s,u}^{\otimes2}\otimes\mathbf{X}_{u,t}\right]\right\|_{L^{m k/3}}\lesssim|t-s|^{\gamma},$$

so that applying again Lemma B.1,

∥Z I [s,t]∥ 2 Lmk/(k′+1) !1/2 X [s,t]∈πn,[τ0,τ1] u∈(s,t) Z I [s,t] Lmk/(k′+1) ≤X [s,t]∈πn,[τ0,τ1] u∈(s,t) EG[s,t] [Z I [s,t] ] Lmk/(k′+1) +  X [s,t]∈πn,[τ0,τ1] u∈(s,t) |t − s| 6α !1/2 |t − s| γ +  X ≤X [s,t]∈πn,[τ0,τ1] u∈(s,t) [s,t]∈πn,[τ0,τ1] u∈(s,t) ≤ |πn| (γ−1)∧(3α−1/2) |τ1 − τ0| + |τ1 − τ0| 1/2. (26)
$$(27)$$
Combining the cases i2 = 2, i2 = 3 and i2 ≥ 4 when α ∈ (1/4, 1/3] yields, for each [τ0, τ1] ∈ πN , πn with n ≥ N, i1, i3 ≥ 0, i2 ≥ 2 with i1 + i2 + i3 = k
′ + 1 and *I ∈ {*0, 1}
i2,

X [s,t]∈πn,[τ0,τ1] u∈(s,t) Z I [s,t] Lmk/(k′+1) ≲  |πn| (β−1)∧(γ−1)∧(4α−1)|τ1 − τ0|. (27)
Defining ϵ as in Equation (9), we can plug bounds (22), (25) and (27) into Equation (20) to deduce that

$$\left\|S^{k^{\prime}+1}(\mathbb{X}^{\pi_{n+1}})_{[\tau_{0},\tau_{1}]}-S^{k^{\prime}+1}(\mathbb{X}^{\pi_{n}})_{[\tau_{0},\tau_{1}]}\right\|_{L^{m k/(k^{\prime}+1)}}\lesssim|\tau_{1}-\tau_{0}||\pi_{n}|^{\epsilon},$$

and, hence, under the assumption that Pn≥1|πn| ϵ < ∞, for any [τ0, τ1] ∈ πN with N ≥ 1, the sequence
{S
k
′+1(X
πn )[τ0,τ1], n ≥ N} is Cauchy in L
mk/(k
′+1). Since L
mk/(k
′+1) is a Banach space the sequence converges in L
mk/(k
′+1) to S
k
′+1(X)[τ0,τ1] ∈ L
mk/(k
′+1) (by uniqueness of limits) with rate

$$\left\|S^{k^{\prime}+1}(\mathbb{X})_{[\tau_{0},\tau_{1}]}-S^{k^{\prime}+1}(\mathbb{X}^{\pi_{n}})_{[\tau_{0},\tau_{1}]}\right\|_{L^{m k/(k^{\prime}+1)}}$$
≤ X n′≥n S k ′+1(X πn′+1 )[τ0,τ1] − S k ′+1(X πn′)[τ0,τ1] Lmk/(k′+1) ≲ |τ1 − τ0| X n′≥n |πn′ | ϵ.
And, to complete the inductive step for k
′ + 1, note that for all N ≥ 1 and [τ0, τ1] ∈ πN ,

$$\left\|S^{k^{\prime}+1}(\mathbb{X}^{\pi_{N}})_{[\tau_{0},\tau_{1}]}\right\|_{L^{m,k/(k^{\prime}+1)}}\leqslant\left\|S^{k^{\prime}+1}(\mathbb{X})_{[\tau_{0},\tau_{1}]}\right\|_{L^{m,k/(k^{\prime}+1)}}+|\tau_{1}-\tau_{0}|\sum_{n^{\prime}\geq N}|\pi_{n^{\prime}}|^{\epsilon}$$ $$\lesssim\left\|S^{k^{\prime}+1}(\mathbb{X})_{[\tau_{0},\tau_{1}]}\right\|_{L^{m,k/(k^{\prime}+1)}}+T\sum_{n^{\prime}\geq1}|\pi_{n^{\prime}}|^{\epsilon}.$$

$$\square$$

## B.1.2. Proof Of Theorem 2.8 Under (Ii)

In what follows we shall simplify notation and denote EF0,t by Et.

Denote by {πn, n ≥ 1} the signature-defining sequence of refining partitions of the interval [0, T]. Without loss of generality, we can consider {πn, n ≥ 1} to be such that πn+1 is obtained from πn by adding at most one refinement in each sub-interval, i.e. for each [*s, t*] ∈ πn either [s, t] ∈ πn+1 or [*s, u*], [u, t] ∈ πn+1 for u ∈ (*s, t*). If not, one can consider a super-sequence satisfying this property and then pass to the original subsequence.

We start by showing inductively that, for any i ∈ {1*, . . . , k*},

$$\sup_{n\geq1}\sup_{\tau\in\pi_{n}}\|S^{i}(\mathbb{X}^{\pi_{n}})_{[0,\tau]}\|_{L^{m_{k}/i}}<\infty.\tag{28}$$

Note that the case i = 1 is trivial since, for any n ≥ 1 and τ ∈ πn, by (Aα)

$$\|S^{1}(\mathbb{X}^{\pi_{n}})_{[0,\tau]}\|_{L^{m k}}=\|\mathbf{X}_{0,\tau}\|_{L^{m k}}\lesssim\tau^{\alpha}\lesssim T^{\alpha}.$$

Next, for the inductive step, assume that (28) holds for all i ∈ {1*, . . . k*′} with k
′ ≤ k − 1. Then, by using Lemma B.3, we can bound for any n ≥ 1 and τ ∈ πn,

∥S
k
′+1(X
πn )[0,τ]∥Lmk/(k′+1)
1
(1 + i)!
X
[u,v]∈πn,[0,τ]
S
k
′−i(X
πn )[0,u] ⊗ X⊗(i+1)
u,v
Lmk/(k′+1)
(ii)
≤X
[u,v]∈πn,[0,τ]
S
k
′(X
πn )[0,u] ⊗ Eu[Xu,v]
Lmk/(k′+1)
(i)
≤
k
X
′
i=0

1/2
+

X
[u,v]∈πn,[0,τ]
S
k
′(X
πn )[0,u] ⊗ Xu,v

2
Lmk/(k′+1)

+
k
X
′
1
(1 + i)!
X
[u,v]∈πn,[0,τ]
S
k
′−i(X
πn )[0,u] ⊗ X⊗(i+1)
u,v
Lmk/(k′+1)
i=1
[u,v]∈πn,[0,τ]
∥Xu,v∥
2
Lp
!1/2
+
k
X
′
(iii)
≲
X
[u,v]∈πn,[0,τ]
∥Eu[Xu,v]∥Lmk +
 X
i=1
X
[u,v]∈πn,[0,τ]
∥Xu,v∥
i+1
Lmk
[u,v]∈πn,[0,τ]
|v − u|
2α
!1/2+
k
X
′
(iv)
≲
X
[u,v]∈πn,[0,τ]
|v − u|
δ +
 X
i=1
X
[u,v]∈πn,[0,τ]
|v − u|
(i+1)α
(v)
≲ τ +
√τ + τ ≲ T +
√T ,
where in (i) we applied the triangle inequality, in (ii) we bounded the i = 0 term by applying Lemma B.1 to the sequence of random variables Z[u,v]:= S
k
′(X
πn )[0,u] ⊗ Xu,v ∈ L
mk/(k
′+1),
with filtration {Fu, [*u, v*] ∈ πn,[0,τ]} and we bounded the i = 1*, . . . , k*′terms by applying the triangle inequality, in (iii)
we applied the Holder inequality given in Lemma ¨ B.2 and the inductive hypothesis Equation (28) for all signature levels up to k
′, in (iv) we used Assumptions (Aα) and (Aδ).

Proceeding again by induction, we will show the conclusion of the theorem holds by proving the stronger statement: For each i ∈ {1*, . . . , k*}, for all N ≥ 1, τ ∈ πN and n ≥ N,

$$\|S^{i}(\mathbb{X}^{\pi_{n+1}})_{[0,\tau]}-S^{i}(\mathbb{X}^{\pi_{n}})_{[0,\tau]}\|_{L^{m k/i}}\stackrel{<}{\sim}|\pi_{n}|^{\epsilon}.$$
ϵ. (29)
The case k
′ = 1 is again trivial since, for all N ≥ 1, τ ∈ πN and n ≥ N, S
1(X
πn+1 )[0,τ] = X0,τ , and hence

$$(29)$$

$$\|S^{1}(\mathbb{X}^{\pi_{n+1}})_{[0,\tau]}-S^{1}(\mathbb{X}^{\pi_{n}})_{[0,\tau]}\|_{L^{m k}}=0.$$

For the inductive step, assume Equation (29) holds for all i ∈ {1*, . . . , k*′} with k
′ ≤ k. Fix N ≥ 1, τ ∈ πN and n ≥ N,
then we can write the telescoping sum

$$S^{k^{\prime}+1}(\mathbb{X}^{n_{n+1}})_{[0,\tau]}-S^{k^{\prime}+1}(\mathbb{X}^{n_{n}})_{[0,\tau]}=\sum_{[\sigma,\,t]\in\pi_{n,[0,\tau]}}\left[S^{k^{\prime}+1}(\mathbb{X}^{n_{n},\,s})_{[0,\tau]}-S^{k^{\prime}+1}(\mathbb{X}^{n_{n},\,s})_{[0,\tau]}\right]\,,\tag{30}$$

where the partitions πn,s are defined as πn,s = πn+1,[0,s] ∪ πn,[s,T], i.e. for each [s, t] ∈ πn, the partitions πn,s and πn,t differ by at most one point u ∈ (*s, t*). Note that, for each [*s, t*] ∈ πn with refinement u ∈ (*s, t*), we can apply Lemma B.3 to write

S
k
′+1(X
πn,t )[0,τ] − S
k
′+1(X
πn,s )[0,τ]
=
k
X
′
1
(1 + i)!
(
S
k
′−i(X
πn,t )[0,s] ⊗ X⊗(1+i)
s,u + S
k
′−i(X
πn,t )[0,u] ⊗ X
⊗(1+i)
u,t − S
k
′−i(X
πn,s )[0,s] ⊗ X
⊗(1+i)
s,t
i=0
[v,w]∈πn,[t,τ]
hS
k
′−i(X
πn,t )[0,v] − S
k
′−i(X
πn,s )[0,v]
i⊗ X⊗(1+i)
v,w 
)
+X
=
k
X
′
1
(1 + i)!
(S
k
′−i(X
πn+1 )[0,s] ⊗
hX⊗(1+i)
s,u − X
⊗(1+i)
s,t i+
k
X
′−i
j=0
S
k
′−i−j(X
πn+1 )[0,s] ⊗
X⊗j
s,u
j!⊗ X
⊗(1+i)
u,t
i=0
[v,w]∈πn,[t,τ]
hS
k
′−i(X
πn,t )[0,v] − S
k
′−i(X
πn,s )[0,v]
i⊗ X⊗(1+i)
v,w 
)
+X
=
k
X
′
1
(1 + i)!
(
S
k
′−i(X
πn+1 )[0,s] ⊗
hX⊗(1+i)
s,u + X
⊗(1+i)
u,t − X
⊗(1+i)
s,t 
i
i=0
+
k
′
X
−i−1
1
(1 + j)!S
k
′−i−j−1(X
πn+1 )[0,s] ⊗ X⊗(1+j)
s,u ⊗ X
⊗(1+i)
u,t
j=0
+X [v,w]∈πn,[t,τ] hS k ′−i(X πn,t )[0,v] − S k ′−i(X πn,s )[0,v] = k X ′ 1 (1 + i)! ( − S k ′−i(X πn+1 )[0,s] ⊗X I∈{0,1} 1+i I̸=(0,...,0),(1,...,1) O l∈I X⊗l s,u ⊗ X ⊗(1−l) u,t   i=0 + k ′ X −i−1 1 (1 + j)!S k ′−i−j−1(X πn+1 )[0,s] ⊗ X⊗(1+j) s,u ⊗ X ⊗(1+i) u,t j=0 20
i⊗ X⊗(1+i)
v,w 
)