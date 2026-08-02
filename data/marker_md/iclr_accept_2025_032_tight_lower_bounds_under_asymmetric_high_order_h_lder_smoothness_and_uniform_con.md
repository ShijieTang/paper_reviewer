# TIGHT LOWER BOUNDS UNDER ASYMMETRIC HIGH-ORDER HOLDER ¨ SMOOTHNESS AND UNIFORM CON-VEXITY

Cedar Site Bai Department of Computer Science Purdue University West Lafayette, IN, USA bai123@purdue.edu

Brian Bullins Department of Computer Science Purdue University West Lafayette, IN, USA bbullins@purdue.edu

#### ABSTRACT

In this paper, we provide tight lower bounds for the oracle complexity of minimizing high-order Holder smooth and uniformly convex functions. Specifi- ¨ cally, for a function whose p th-order derivatives are Holder continuous with ¨ degree ν and parameter H, and that is uniformly convex with degree q and parameter σ, we focus on two asymmetric cases: (1) q > p + ν, and (2) q < p + ν. Given up to p th-order oracle access, we establish worst-case oracle complexities of Ω <sup>H</sup> σ <sup>2</sup> 3(p+ν)−2 σ ϵ 2(q−p−ν) <sup>q</sup>(3(p+ν)−2) in the first case with an ℓ∞-ball-truncated-Gaussian smoothed hard function and Ω <sup>H</sup> σ <sup>2</sup> 3(p+ν)−<sup>2</sup> + log log σ p+ν H<sup>q</sup> 1 <sup>p</sup>+ν−<sup>q</sup> 1 ϵ in the second case, for reaching an ϵ-approximate solution in terms of the optimality gap. Our analysis generalizes previous lower bounds for functions under first- and second-order smoothness as well as those for uniformly convex functions, and furthermore our results match the corresponding upper bounds in this general setting.

# 1 INTRODUCTION

With the advancement in computational power, high-order optimization methods (p th-order with p ≥ 2) are gaining more attention for their merit of faster convergence and higher precision. Consequently, uniformly convex problems (with degree q) have become a recent focus, particularly the subproblems of some high-order optimization methods. The subproblem of the cubic-regularized Newton (p = 2, q = 3) [\(Nesterov & Polyak,](#page-11-0) [2006\)](#page-11-0) is an example, as are methods of even higher orders (p ≥ 3, q ≥ 4) [\(Zhu & Cartis,](#page-11-1) [2022\)](#page-11-1).

Although these problems are high-order smooth by definition, a lower-order algorithm may be employed to obtain an approximate solution. For instance, solving the subproblem of cubic-regularized (i.e., q = 3) Newton with gradient descent (accessing first-order oracle, i.e., p = 1), or, more generally, approximately solving the subproblem of (q − 1)th-order Taylor descent [\(Bubeck et al.,](#page-10-0) [2019\)](#page-10-0) (which typically contains a regularization term to the power of q) with lower-order oracle access, introduces an asymmetry between the algorithm's oracle access order and the degree of uniform convexity (q > p + 1).

Conversely, a lower-degree regularization can be paired with a higher-order smooth function. This enables methods that access higher-order oracles, which leads to the opposite asymmetry (q < p + 1). Examples include the objective function of logistic regression, which is known to be infinite-order smooth. Coupled with standard ℓ2-regularization, the problem can be analyzed as a p th-order smooth and strongly convex (q = 2) problem, e.g., p = 2 with access to the Hessian matrix, p = 3 accessing the third-order derivative tensor.

In addressing specific instances of this asymmetry, previous works established some upper bounds [\(Gasnikov et al.,](#page-10-1) [2019;](#page-10-1) [Song et al.,](#page-11-2) [2021\)](#page-11-2) and lower bounds [\(Arjevani et al.,](#page-9-0) [2019;](#page-9-0) [Kornowski](#page-10-2) [& Shamir,](#page-10-2) [2020;](#page-10-2) [Doikov,](#page-10-3) [2022;](#page-10-3) [Thomsen & Doikov,](#page-11-3) [2024\)](#page-11-3) for the oracle complexity. Notably, [Song et al.](#page-11-2) [\(2021\)](#page-11-2) proposed a unified acceleration framework for functions that are p th-order Holder smooth with degree ¨ ν, and uniformly convex with degree q, providing upper bounds for any combination of p, q, and ν. For the case where q > p + ν, they show an oracle complexity of O <sup>H</sup> σ <sup>2</sup> 3(p+ν)−2 σ ϵ 2(q−p−ν) <sup>q</sup>(3(p+ν)−2) , and for the case where q < p + ν, the complexity is O <sup>H</sup> σ <sup>2</sup> 3(p+ν)−<sup>2</sup> + log log σ p+ν H<sup>q</sup> 1 <sup>p</sup>+ν−<sup>q</sup> 1 ϵ . To the best of our knowledge, no lower bounds exist in this general setting, particularly with Holder smoothness and uniform convexity. ¨

In this paper, we provide matching lower bounds to the upper bounds in [\(Song et al.,](#page-11-2) [2021\)](#page-11-2) for these asymmetric cases. Specifically, we establish Ω <sup>H</sup> σ <sup>2</sup> 3(p+ν)−2 σ ϵ 2(q−p−ν) <sup>q</sup>(3(p+ν)−2) for q > p + ν and Ω <sup>H</sup> σ <sup>2</sup> 3(p+ν)−<sup>2</sup> + log log σ p+ν H<sup>q</sup> 1 <sup>p</sup>+ν−<sup>q</sup> 1 ϵ for q < p <sup>+</sup> <sup>ν</sup>. For the q > p <sup>+</sup> <sup>ν</sup> case, we adopt the framework proposed by [\(Guzman & Nemirovski](#page-10-4) ´ , [2015\)](#page-10-4), utilizing a smoothing operator to generate a high-order smooth function. We propose the use of ℓ∞-ball-truncated Gaussian smoothing, which, as we later justify, is novelly designed to achieve the optimal rate and be compatible with both high-order smooth and uniformly convex settings. Both the truncated Gaussian smoothing and the construction of the ℓ<sup>∞</sup> ball are crucial to improve upon the sub-optimal derivation using uniform smoothing within an ℓ<sup>2</sup> ball in [\(Agarwal & Hazan,](#page-9-1) [2018\)](#page-9-1). Our results generalize the lower bounds in [\(Doikov,](#page-10-3) [2022;](#page-10-3) [Thomsen & Doikov,](#page-11-3) [2024\)](#page-11-3) to higher-order and Holder smooth settings. For the ¨ q < p + ν case, we adopt Nesterov's framework [\(Nesterov et al.,](#page-11-4) [2018\)](#page-11-4) and generalize the lower bounds in [\(Arjevani et al.,](#page-9-0) [2019;](#page-9-0) [Kornowski & Shamir,](#page-10-2) [2020\)](#page-10-2) to include Holder smooth and uniformly ¨ convex settings.

# 2 RELATED WORK

Upper Bounds. [Doikov & Nesterov](#page-10-5) [\(2021\)](#page-10-5) showcase the upper bound for uniformly convex functions with Holder-continuous Hessian via cubic regularized Newton method, but the rate is not ¨ optimal. For higher order result, [Bubeck et al.](#page-10-0) [\(2019\)](#page-10-0) and [Jiang et al.](#page-10-6) [\(2019\)](#page-10-6) established a near optimal upper bound of O˜ ϵ − <sup>2</sup> <sup>3</sup>p+1 in the simpler case of ν = 1 without uniform convexity. [Gasnikov et al.](#page-10-1) [\(2019\)](#page-10-1) achieve the same near-optimal rate, but also consider uniform convexity, and by the restarting mechanism, derive the rate that for q > p + 1 as well, generalizing the upper bounds established in second-order [\(Monteiro & Svaiter,](#page-11-5) [2013\)](#page-11-5) and matching the lower bounds later derived in [\(Kornowski](#page-10-2) [& Shamir,](#page-10-2) [2020\)](#page-10-2). [Kovalev & Gasnikov](#page-10-7) [\(2022\)](#page-10-7) closed the log 1 ϵ gap, but does not consider uniform convexity or Holder smoothness. For minimizing uniformly convex functions, ¨ [Juditsky & Nesterov](#page-10-8) [\(2014\)](#page-10-8) and [Roulet & d'Aspremont](#page-11-6) [\(2017\)](#page-11-6) study the complexity of first-order methods. Recently, [Song et al.](#page-11-2) [\(2021\)](#page-11-2) establish the most general upper bounds for arbitrary combinations of the order of Holder smoothness and the degree of uniform convexity, which include the rates for both ¨ q > p + ν and q < p + ν cases.

Lower Bounds. [Agarwal & Hazan](#page-9-1) [\(2018\)](#page-9-1) proved for p th-order smooth convex functions an Ω ϵ − <sup>2</sup> <sup>5</sup>p+1 lower bound based on constructing the hard function with randomized smoothing uniformly over a unit ball. But their rate is not optimal due to the extra dimension factor appearing in the smoothness constant due to the uniform randomized smoothing. [Garg et al.](#page-10-9) [\(2021\)](#page-10-9) added softmax smoothing prior to randomized smoothing, achieving a near-optimal rate of Ω ϵ − <sup>2</sup> <sup>3</sup>p+1 for randomized and quantum algorithms. Separately, [Arjevani et al.](#page-9-0) [\(2019\)](#page-9-0) also established the optimal lower bound of Ω ϵ − <sup>2</sup> <sup>3</sup>p+1 with the Nesterov's hard function construction approach. Furthermore, for the asymmetric case of q < p + 1, [Arjevani et al.](#page-9-0) [\(2019\)](#page-9-0) proved the lower bound of Ω <sup>H</sup> σ 2 <sup>7</sup> + log log σ 3 <sup>H</sup><sup>2</sup> ϵ −1 for the <sup>p</sup> = 2 and <sup>q</sup> = 2 case, and the result is later generalized to the p th order in [\(Kornowski & Shamir,](#page-10-2) [2020\)](#page-10-2). No q > 2 uniformly convex settings were considered in these works. For the case of q > p + ν, lower bounds for uniformly convex functions for q ≥ 3 are limited to the first-order smoothness setting where p = 1 [\(Juditsky & Nesterov,](#page-10-8) [2014;](#page-10-8) [Doikov,](#page-10-3) [2022;](#page-10-3) [Thomsen & Doikov,](#page-11-3) [2024\)](#page-11-3). No lower bounds for uniformly convex functions were established, to our knowledge, in the high-order setting.

#### 3 PRELIMINARIES AND SETTINGS

Notations. We use [n] to represent the set {1, 2, ..., n}. We use ∥ · ∥ to denote an ℓ<sup>2</sup> operator norm. We use ∇ for gradients, ∂ for subgradients, and ⟨·, ·⟩ for inner products. Related to the algorithm, bold lower letters for vectors (e.g., x, y), and with subscript, the vectors in different iterations (e.g., x<sup>T</sup> ). We use regular lower letters for scalars, and with subscript, a coordinate of a vector (e.g., xi). Depending on the context, we use capital letters for a matrix or a random variable. We use ϕ for the probability density function of the standard normal or the standard multivariate normal (MVN), and Φ for the cumulative (density) function of standard normal or MVN. We further overuse the notation of ϕ[·,·] Φ[·,·] for their truncated counterparts for the normal distribution (standard normal if not specified with parameters), and ϕ∥·∥∞≤· Φ∥·∥∞≤· for the MVN truncated within an ℓ<sup>∞</sup> ball.

#### 3.1 DEFINITIONS

Definition 1 (High-order Smoothness). *For* p ∈ Z <sup>+</sup>*, a function* f : <sup>R</sup> <sup>d</sup> → <sup>R</sup> *is* p th*-order smooth or whose* p th*- derivatives are* Lp*-Lipschitz if for* L<sup>p</sup> > 0*,* ∀ x, y ∈ <sup>R</sup> d *,* ∥∇<sup>p</sup>f(x) − ∇<sup>p</sup>f(y)∥ ≤ Lp∥x − y∥*.*

Definition 2 (High-order Holder Smoothness) ¨ . *For* p ∈ Z <sup>+</sup>*, a function* f : <sup>R</sup> <sup>d</sup> → <sup>R</sup> *is* p th  *order Holder smooth or has H ¨ older continuous ¨* p th*-order derivatives if for* ν ∈ (0, 1] *and* H > 0*,* ∀ x, y ∈ R d *,* ∥∇<sup>p</sup>f(x) − ∇<sup>p</sup>f(y)∥ ≤ H∥x − y∥ ν *.*

Definition 3. (Uniform Convexity [\(Nesterov et al.,](#page-11-4) [2018,](#page-11-4) Section 4.2.2)) *For integer* q ≥ 2 *and* σ > 0*, a function* f : R <sup>d</sup> → <sup>R</sup> *is uniformly convex with degree* q *and modulus* σ *if* ∀ x, y ∈ <sup>R</sup> d *,* f(y)−f(x)−⟨∇f(x), y − x⟩ ≥ <sup>σ</sup> q ∥y−x∥ q *, or the function satisfies* ⟨∇f(y) − ∇f(x), y − x⟩ ≥ σ∥y − x∥ q *.*

### 4 LOWER BOUND FOR THE q > p + ν CASE

The derivation of the lower bound is to find such a function by construction that satisfies the uniformly convex and Holder smooth conditions and requires at least a certain amount of iterations to reach an ¨ ϵ-approximate solution. The general steps follow from the framework of showing lower complexity bounds for smooth convex optimization [\(Guzman & Nemirovski](#page-10-4) ´ , [2015\)](#page-10-4), which originates from [\(Nemirovskii & Nesterov,](#page-11-7) [1985\)](#page-11-7) and serves as the basis for results in various follow-up settings [\(Agarwal & Hazan,](#page-9-1) [2018;](#page-9-1) [Garg et al.,](#page-10-9) [2021;](#page-10-9) [Doikov,](#page-10-3) [2022\)](#page-10-3). The construction starts from a nonsmooth function, then smooths the function with some smoothing operator (e.g. Moreau envelope in [\(Guzman & Nemirovski](#page-10-4) ´ , [2015;](#page-10-4) [Doikov,](#page-10-3) [2022\)](#page-10-3), randomized smoothing uniformly within a ball in [\(Agarwal & Hazan,](#page-9-1) [2018;](#page-9-1) [Garg et al.,](#page-10-9) [2021\)](#page-10-9)). We design a truncated Gaussian smoothing operator within the ℓ<sup>∞</sup> ball and start the derivation by stating its formal definition and key properties.

#### 4.1 TRUNCATED GAUSSIAN SMOOTHING

Definition 4 (Truncated Gaussian Smoothing). *For* f : R <sup>d</sup> → <sup>R</sup> *and a parameter* ρ > 0*, define the truncated Gaussian smoothing operator* Sρ[f] : (<sup>R</sup> <sup>d</sup> → <sup>R</sup>) → (<sup>R</sup> <sup>d</sup> → <sup>R</sup>) *as*

$$S_\rho[f](\mathbf{x}) = \mathbb{E}_V[f(\mathbf{x} + \rho V)]$$

*where* V *is a* d*-dimensional random variable that follows the standard multivariate normal (MVN) distribution truncated within a unit ball. That is, the probability density function (PDF) of* V *is*

$$\mathbb{P}[V = \mathbf{v}] = \frac{1}{Z(d)(2\pi)^{\frac{d}{2}}} \exp \left\{ -\frac{\mathbf{v}^\top \mathbf{v}}{2} \right\} \mathbb{I}[\|\mathbf{v}\|_\infty \leq 1],$$

*in which* <sup>I</sup>[·] = 1 *if* · *is true* 0 *otherwise is the indicator function and* Z(d) *is the normalizing factor, i.e., the cumulative distribution within the* d*-dimensional unit* ℓ∞*-ball [\(Cartinhour,](#page-10-10) [1990\)](#page-10-10).*

*We denote* f<sup>ρ</sup> = Sρ[f]*, and use the shorthand notation for the function that applied the smoothing operator for* p *times:* f p <sup>ρ</sup> = S p ρ [f] = Sρ[· · · [Sρ[f]] · · · ] *for* p *times.*

Now we justify the choice of truncated Gaussian smoothing for the construction of hard function. We notice that [Agarwal & Hazan](#page-9-1) [\(2018\)](#page-9-1) choose randomized smoothing uniformly over a unit ℓ2-ball, which by their Lemma 2.3 that the smoothed function is O(d)-smooth (which in fact can be tightened to O( √ d) by [\(Yousefian et al.,](#page-11-8) [2012;](#page-11-8) [Duchi et al.,](#page-10-11) [2012,](#page-10-11) Lemma 8)) where d is the dimension of the variable. Since the number of iteration T ∈ O(d), their result O T − <sup>2</sup> <sup>5</sup>p+1 is sub-optimal by an extra T comparing to the tight lower bound O T − <sup>2</sup> <sup>3</sup>p+1 [\(Arjevani et al.,](#page-9-0) [2019\)](#page-9-0). Therefore we search for a smoothing operator with Lipschitz constant being *dimension-free*. We notice that Gaussian smoothing [\(Duchi et al.,](#page-10-11) [2012,](#page-10-11) Lemma 9), softmax smoothing [\(Bullins,](#page-10-12) [2020,](#page-10-12) Lemma 7), and Moreau smoothing [\(Doikov,](#page-10-3) [2022,](#page-10-3) Lemma 1) are such operators.

Yet as the reader will later see in the proof that the converging points are generated through a sequence of functions, instead of those generated from one hard function. For these two sequences of points to be identical so that the lower bound is indeed for optimizing the hard function constructed, we need the smoothing operator to be *local*, that is, accessing information within *some neighborhood* of the queried point, e.g., a unit ℓ2-ball in [\(Doikov,](#page-10-3) [2022\)](#page-10-3). Unfortunately, Gaussian smoothing and softmax smoothing need access to global information.

For Moreau smoothing that indeed depends on local information, it's successfully applied in proving the lower bound in the first-order setting [\(Doikov,](#page-10-3) [2022\)](#page-10-3), but is not suited for the high-order setting. First, one may attempt the extension of Moreau smoothing with a p th-power regularization, yet it can be shown that the function is not p th-order smooth. Next, one may try to apply Moreau smoothing p times, yet unlike randomized smoothing in [\(Agarwal & Hazan,](#page-9-1) [2018\)](#page-9-1), the Lipschitz constant does not raise to the p th-power with the number of times the smoothing operator is applied, which leads to the same rate as in the first order. Observing the proof of [\(Agarwal & Hazan,](#page-9-1) [2018,](#page-9-1) Corollary 2.4), this is in essence due to the fact that the minimization in Moreau smoothing does not commute with derivative, whereas the expectation in randomized smoothing does.

We then come up with the idea of a truncated multivariate Gaussian smoothing operator that is (i) local (ii) smooth with a dimension-free constant (iii) p th-order smooth with smoothness constant raising to the p th power as well. Initially, we applied the Gaussian smoothing truncated within a unit ball in ℓ<sup>2</sup> by default. We noticed later, however, that the marginal distribution of unit-ℓ2-ball truncated multivariate Gaussian is not the truncated standard normal between [−1, 1], but with an extra d-dependent normalizing constant, which adds the d-dependency to the smoothness constant of the hard function.

To ensure a dimension-free smoothness constant, we instead apply the multivariate Gaussian smoothing truncated within an ℓ<sup>∞</sup> ball, a.k.a., the hypercube with edge length 2, whose marginal distribution is indeed the truncated standard normal between [−1, 1] [\(Cartinhour,](#page-10-10) [1990\)](#page-10-10). The following lemma characterizes these desired properties including convexity, continuity, approximation, and smoothness, with proof deferred to Appendix [A.1.](#page-12-0)

Lemma 1. *Given a L-Lipschitz function* f*, the function* f p <sup>ρ</sup> = Sρ[· · · [Sρ[f]] · · · ] *satisfies*

- *(i) If* f *is convex,* f p ρ *is convex and* L*-Lipschitz with respect to the* ℓ<sup>2</sup> *norm.*
- *(ii) If* f *is convex,* f(x) ≤ f p ρ
- (x) ≤ f(x) + <sup>5</sup><sup>p</sup> 4 Lρ√
  - d*.*
- *(iii)* ∀i ∈ [p]*,* ∀x, x ′ ∈ <sup>R</sup> d *,* ∥∇<sup>i</sup>f p
- (x) − ∇<sup>i</sup>f p (x ′ L∥x − x ′∥*.*

$$(iii) \quad \forall i \in [p], \forall \mathbf{x}, \mathbf{x}' \in \mathbb{R}^d, \|\nabla^i f_\rho^p(\mathbf{x}) - \nabla^i f_\rho^p(\mathbf{x}')\| \leq \left(\frac{2}{\rho}\right)^i L \|\mathbf{x} - \mathbf{x}'\|.$$

4.2 THE LOWER BOUND: FUNCTION CONSTRUCTION AND TRAJECTORY GENERATION

Theorem 1. *For any* T*-step* ( √ d − 1 ≤ T ≤ d) *deterministic algorithm* A *with oracle access up to the* p th *order, there exists a convex function* f(x) *whose* p th*-order derivative is Holder continuous of ¨ degree* ν *with modulus* H *and a corresponding* F(x) = f(x) + <sup>σ</sup> q ∥x∥ <sup>q</sup> *with regularization that is uniformly convex of degree* q *with modulus* σ*, such that* q > p + ν*, it takes*

$$T \in \Omega \left( \left( \frac{H}{\sigma} \right)^{\frac{2}{3(p+\nu)-2}} \left( \frac{\sigma}{\epsilon} \right)^{\frac{2(q-p-\nu)}{q(3(p+\nu)-2)}} \right)$$

4.2.1 FUNCTION CONSTRUCTION WITH TRUNCATED GAUSSIAN SMOOTHING

*1. Non-smooth Function Construction.* We first construct the function

$$g_t(\mathbf{x}) = \max_{1 \leq k \leq t} r_k(\mathbf{x}) \quad \text{where} \quad \forall k \in [T], r_k(\mathbf{x}) = \xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x} \rangle - (k-1)\delta.$$

ξ<sup>k</sup> ∈ {−1, 1}, e is the standard basis, α is a permutation of [T], and δ > 0 is some parameter that we will choose later. Lemma [2](#page-4-0) characterizes the properties of g<sup>t</sup> with proof in Appendix [A.2.](#page-16-0)

Lemma 2. ∀ t ∈ [T]*,* g<sup>t</sup> *is convex and* 1*-Lipschitz with respect to the* ℓ∞*-norm, and also the* ℓ2*-norm.*

*2. Truncate Gaussian Smoothing.* Next, we smooth the function gt(x) with truncate Gaussian smoothing as in Definition [4.](#page-2-0) Given a parameter ρ > 0 and p ∈ Z +,

$$G_t(\mathbf{x}) = S_\rho^p[g_t](\mathbf{x})$$

Based on Lemma [1,](#page-3-0) we show that Gt(x) satisfies the following lemma, with proof in Appendix [A.2.](#page-16-0) Lemma 3. ∀ t ∈ [T]*,* ∀ x, y ∈ <sup>R</sup> d *,*

- (i) 
  $$G_t(\mathbf{x})$$
   is convex and 1-Lipschitz, i.e.,  $G_t(\mathbf{x}) - G_t(\mathbf{y}) \leq \|\mathbf{x} - \mathbf{y}\|$ .
- (ii)  $g_t(\mathbf{x}) \leq G_t(\mathbf{x}) \leq g_t(\mathbf{x}) + \frac{5}{4}p\rho\sqrt{d}$ .

*(ii)* gt(x) ≤ Gt(x) ≤ gt(x) + <sup>5</sup>

$$(ii) \quad g_t(\mathbf{x}) \leq G_t(\mathbf{x}) \leq g_t(\mathbf{x}) + \frac{5}{4}p\rho\sqrt{d}.$$

$$(iii) \quad \text{For some fixed } p \in \mathbb{Z}^+, \forall i \in [p], \|\nabla^i G_t(\mathbf{x}) - \nabla^i G_t(\mathbf{y})\| \leq \left(\frac{2}{\rho}\right)^i \|\mathbf{x} - \mathbf{y}\|.$$

*3. Adding Uniform Convexity.* Now that the constructed function Gt(x) is all-order smooth, we add to it the uniformly convex regularization. We define

$$f_t(\mathbf{x}) = \beta G_t(\mathbf{x}) \quad f(\mathbf{x}) = f_T(\mathbf{x})$$

$$F_t(\mathbf{x}) = f_t(\mathbf{x}) + d_q(\mathbf{x}) \quad \text{for} \quad d_q(\mathbf{x}) = \frac{\sigma}{q} \|\mathbf{x}\|^q, \quad \mathbf{x} \in \mathcal{Q} \quad F(\mathbf{x}) = F_T(\mathbf{x}),$$

where β > 0 is a parameter that we will choose later, Q = {x : ∥x∥<sup>2</sup> ≤ D} [1](#page-4-1) for D ≤ <sup>H</sup> 2 <sup>1</sup>−νC <sup>1</sup> q−p−ν and C = σ(q − 1) × · · · × (q − p).

Lemma 4. *For* F(x) = f<sup>T</sup> (x) + dq(x) *where* dq(x) = <sup>σ</sup> q x q *and* x ∈ Q*,*

- *(i)* F *is uniformly convex function with degree* q *and modulus* σ > 0*.*
- *(ii)* F(x) *is* p th*-order Holder smooth with parameter ¨* H = 2 <sup>p</sup>+1β <sup>ρ</sup><sup>p</sup>+ν−<sup>1</sup> *,* ∀ p ∈ <sup>Z</sup> +*.*

Therefore, by Lemma [4,](#page-4-2) the function constructed satisfies the desired uniform convexity and highorder smoothness conditions. Next, we characterize with Lemma [5](#page-4-3) the upper and lower bounds of the constructed function which will be used in the proof later.

Lemma 5. *For* R(x) = β maxk∈[T] ξ<sup>k</sup> eα(k) , x + σ q ∥x∥ q *, we have*

$$R(\mathbf{x}) - \beta(T-1)\delta \leq F(\mathbf{x}) \leq R(\mathbf{x}) + \frac{5}{4}p\beta\rho\sqrt{d}.$$

#### 4.2.2 CONVERGENCE TRAJECTORY GENERATION

*4. Trajectory Generation Procedure.* The trajectory is generated following a standard T-step iterative procedure same as outlined in [\(Guzman & Nemirovski](#page-10-4) ´ , [2015;](#page-10-4) [Doikov,](#page-10-3) [2022\)](#page-10-3):

· For t = 1, x<sup>1</sup> is the first point of the trajectory and is chosen by initialization of some algorithm A, independent of F. Subsequently, choose

$$\alpha(1) \in \arg \max_{k \in [T]} |\langle \mathbf{e}_{\alpha(k)}, \mathbf{x}_1 \rangle| \quad \xi_1 = \text{sign} \left( \langle \mathbf{e}_{\alpha(1)}, \mathbf{x}_1 \rangle \right),$$

after which a fixed F1(x) is generated.

<sup>1</sup>We would note that for the q > p + ν case, F is guaranteed to be p th-order smooth only in the bounded domain as constructed, since the regularization term dq(x) may not be p th-order smooth on <sup>R</sup> d . The construction is inspired by that in [\(Juditsky & Nesterov,](#page-10-8) [2014\)](#page-10-8). This is not explicitly discussed in [\(Song et al.,](#page-11-2) [2021;](#page-11-2) [Doikov,](#page-10-3) [2022;](#page-10-3) [Thomsen & Doikov,](#page-11-3) [2024\)](#page-11-3).

· For 2 ≤ t ≤ T, at the beginning of each such iteration, we have access to x1, · · · , xt−1, the function Ft−1, and its gradient information, which we denote as It−1(x) = {Ft−1, ∇Ft−1, · · · , ∇<sup>p</sup>Ft−1}. The algorithm A generates the next point with this information: x<sup>t</sup> = A(It−1(x1), · · · , It−1(xt−1)). Then choose

$$\alpha(t) \in \arg \max_{k \in [T] \setminus \{\alpha(i): i < t\}} |\langle \mathbf{e}_{\alpha(k)}, \mathbf{x}_t \rangle| \quad \xi_t = \text{sign}(\langle \mathbf{e}_{\alpha(t)}, \mathbf{x}_t \rangle)$$

after which a fixed Ft(x) is generated for the next iteration.

*5. Indistinguishability of* F<sup>t</sup> *and* F *for Trajectory Generation.* It's important to note that the trajectory x1, · · · , x<sup>T</sup> is generated based on *a sequence of functions* F1, · · · , F<sup>T</sup> , whereas our object of analysis should be just *one hard function* F = F<sup>T</sup> . Here we show:

Lemma 6. *The trajectory* x1, · · · , x<sup>T</sup> *generated by applying an algorithm* A *iteratively on the sequence of functions* F1, · · · , F<sup>T</sup> *, with up to* p th*-order oracle access, is the same as the trajectory generated applying* A *directly on* F *when oracle access pertains only local information within an* ℓ∞*-ball with radius* δ/2*.*

*Proof.* The idea is to show that ∀ 2 ≤ t ≤ T, the function g<sup>t</sup> coincides with g<sup>T</sup> (so that F<sup>t</sup> coincides with F<sup>T</sup> in terms of generating xt+1, i.e., I<sup>t</sup> = I<sup>T</sup> ) under some mild conditions. Similar proof can be found in [\(Guzman & Nemirovski](#page-10-4) ´ , [2015;](#page-10-4) [Doikov,](#page-10-3) [2022,](#page-10-3) Section 3). By construction, ∀ t ∈ [T],

$$g_t(\mathbf{x}) = \max_{1 \leq k \leq t} r_k(\mathbf{x}) = \max \left\{ \max_{1 \leq k \leq s} r_k(\mathbf{x}), \max_{s < k \leq t} r_k(\mathbf{x}) \right\} = \max \left\{ g_s(\mathbf{x}), \max_{s < k \leq t} r_k(\mathbf{x}) \right\}$$

Furthermore, α(s) ∈ arg maxk∈[T]\{α(i):i<s} eα(k) , x<sup>s</sup>   and ξ<sup>s</sup> = sign eα(s) , x<sup>s</sup> , therefore

$$\begin{aligned} g_s(\mathbf{x}_s) &= \max_{1 \leq k \leq s} \xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x}_s \rangle - (k-1)\delta \geq \max_{1 \leq k \leq s} \xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x}_s \rangle - (s-1)\delta \\ &\geq |\langle \mathbf{e}_{\alpha(s)}, \mathbf{x}_s \rangle| - (s-1)\delta \geq \max_{s < k \leq t} \xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x}_s \rangle - (s-1)\delta \\ &\geq \max_{s < k \leq t} \xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x}_s \rangle - (k-1)\delta + \delta \quad (k, s \in \mathbb{Z}^+, k > s \implies k \geq s+1) \end{aligned}$$

If we limit the information access within an ℓ∞-ball with radius δ/2 when searching for the next point xs+1 from xs, we then establish a local region ∀x, ∥x − xs∥<sup>∞</sup> ≤ δ 2 . Further by Lemma [2](#page-4-0) that g<sup>s</sup> (also ξ<sup>k</sup> eα(k) , x ) is 1-Lipschitz with respect to the ℓ<sup>∞</sup> norm, we have ∀ k such that s < k ≤ t,

$$\begin{aligned} g_s(\mathbf{x}_s) &\geq \xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x}_s \rangle - (k-1)\delta + 2\|\mathbf{x} - \mathbf{x}_s\|_\infty \\ &\geq \xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x}_s \rangle - (k-1)\delta + [g_s(\mathbf{x}_s) - g_s(\mathbf{x})] + [\xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x} \rangle - \xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x}_s \rangle], \end{aligned}$$

which implies that gs(x) ≥ maxs<k≤<sup>t</sup> ξ<sup>k</sup> eα(k) , x −(k−1)δ = maxs<k≤<sup>t</sup> rk(x). This concludes that ∀ x such that ∥x − xs∥<sup>∞</sup> ≤ δ 2 , gt(x) = max {gs(x), maxs<k≤<sup>t</sup> rk(x)} = gs(x), which further implies Ft(x) = Fs(x). Letting t = T we have ∀ t ∈ [T], Ft(x) = F<sup>T</sup> (x) for ∥x − xt∥<sup>∞</sup> ≤ δ 2 .

#### 4.2.3 LOWER BOUND DERIVATION

*6. Bounding the Optimality Gap.* The following lemma bounds optimality gap, whose proof is based on Lemma [5,](#page-4-3) and is presented in Appendix [A.2.](#page-16-0)

**Lemma 7.** 
$$F(\mathbf{x}_T) - F(\mathbf{x}^*) \geq -\beta(T-1)\delta - \frac{5}{4}p\beta\rho\sqrt{d} + \frac{q-1}{q} \left( \frac{\beta^q}{\sigma T^{\frac{q}{2}}} \right)^{\frac{1}{q-1}}$$
.

*7. Setting the parameters.* By Definition [4](#page-2-0) and Lemma [14](#page-12-1) (i), we know that Sρ[gt](x), ∇Sρ[gt](x) depends on the value of gt(x) within an ℓ∞-ball of radius ρ. Therefore inductively, we see that for F(x) = βS<sup>p</sup> ρ [g<sup>T</sup> ](x) + <sup>σ</sup> q ∥x∥ q , F(x), ∇F(x), · · · , ∇<sup>p</sup>F(x) depends on the value of F(x) within an ℓ∞-ball of radius pρ. For our construction to hold, we also need Ft(x) and F(x) to be indistinguishable ∀ t ∈ [T], which is true within an ℓ∞-ball of radius δ/2.

Therefore, we set δ = 2pρ, so that for the purpose of oracle access at x<sup>t</sup> (computing (high-order) gradients of F), it's indistinguishable to replace F(x) with Ft(x), and the sequence generated as in Section [4.2.2](#page-4-4) is the same as that directly applying some p th-order algorithm A on F(x). In other words, F(x) and the generated x<sup>T</sup> serve as valid components for deriving the lower bound.

As a result, F(x<sup>T</sup> )−F(x ∗ ) ≥ −2pβρ(T −1+ <sup>5</sup> 8 √ d)+ <sup>q</sup>−<sup>1</sup> q β q σT 1 q−1 . Let T ≥ √ d−1 ≥ 5 8 √ d−1, then F(x<sup>T</sup> ) − F(x ∗ ) ≥ −4pβρT + q−1 q β σT 1 q−1 . By letting 4pβρT = q−1 2q β σT 1 q−1 , we solve for ρ = q−1 8pq σ − <sup>1</sup> <sup>q</sup>−<sup>1</sup> β <sup>q</sup>−<sup>1</sup> T 2−3q 2(q−1) = cqσ − <sup>1</sup> <sup>q</sup>−<sup>1</sup> β 1 <sup>q</sup>−<sup>1</sup> T 2−3q 2(q−1) , in which c<sup>q</sup> = q−1 8pq , and at the same time,

$$F(\mathbf{x}_T) - F(\mathbf{x}^*) \geq \frac{q-1}{2q} \left( \frac{\beta^q}{\sigma T^{\frac{q}{2}}} \right)^{\frac{1}{q-1}}. \quad (1)$$

By the construction of F(x) and Lemma [4,](#page-4-2) we know that F(x) is p th-order Holder smooth with ¨ parameter H = 2 <sup>p</sup>+1β <sup>ρ</sup><sup>p</sup>+ν−<sup>1</sup> . Plugging in the value of ρ, we have

$$H = 2^{p+1} c_q^{-(p+\nu-1)} \sigma^{\frac{p+\nu-1}{q-1}} \beta^{-\frac{p-q+\nu}{q-1}} T^{\frac{(p+\nu-1)(3q-2)}{2(q-1)}},$$

equivalently, β = Hcp+ν−<sup>1</sup> <sup>q</sup> σ <sup>−</sup> <sup>p</sup>+ν−<sup>1</sup> q−1 2 <sup>p</sup>+1 !<sup>−</sup> q−1 p−q+ν T (p+ν−1)(3q−2) 2(p−q+ν) . Plugging the value of β back into Eq. [\(1\)](#page-6-0), we have

$$\begin{aligned} F(\mathbf{x}_T) - F(\mathbf{x}^*) &\geq \frac{q-1}{2q} \sigma^{-\frac{1}{q-1}} \left( \frac{H c_q^{p+\nu-1} \sigma^{-\frac{p+\nu-1}{q-1}}}{2^{p+1}} \right)^{-\frac{q}{p-q+\nu}} T^{\frac{q[3(p+\nu)-2]}{2(p-q+\nu)}} \\ &= 4p \left( \frac{q-1}{8pq} \right)^{\frac{(p+\nu)(1-q)}{p-q+\nu}} \sigma \left( \frac{H \sigma^{-1}}{2^{p+1}} \right)^{-\frac{q}{p-q+\nu}} T^{\frac{q[3(p+\nu)-2]}{2(p-q+\nu)}}. \end{aligned}$$

We complete the proof for Theorem [1](#page-3-1) by letting F(x<sup>T</sup> ) − F(x ∗ ) ≤ ϵ, from which we solve for T ≥ 2 (2p+2ν+pq−q)/q<sup>−</sup> <sup>2</sup> 3(p+ν)−2 p 2(q−p−ν) <sup>q</sup>[3(p+ν)−2] q−1 <sup>8</sup>pq 2(p+ν)(q−1) <sup>q</sup>[3(p+ν)−2] <sup>H</sup> σ <sup>2</sup> 3(p+ν)−2 σ ϵ 2(q−p−ν) q[3(p+ν)−2] .

# 5 LOWER BOUND FOR THE q < p + ν CASE

Theorem 2. *For any* T*-step deterministic algorithm* A *with oracle access up to the* p th *order, there exists a convex function* f(x) *whose* p th*-order derivative is Holder continuous of degree ¨* ν *with modulus* H *and a corresponding* F(x) = f(x) + <sup>σ</sup> q ∥x∥ <sup>q</sup> *with regularization that is uniformly convex of degree* q *with modulus* σ*, such that* q < p + ν*, it takes*

$$T \in \Omega \left( \left( \frac{H}{\sigma} \right)^{\frac{2}{3(p+\nu)-2}} + \log \log \left( \left( \frac{\sigma^{p+\nu}}{H^q} \right)^{\frac{1}{p+\nu-q}} \frac{1}{\epsilon} \right) \right)$$

*steps to reach an* ϵ*-approximate solution* x<sup>T</sup> *satisfying* F(x<sup>T</sup> ) − F(x ∗ ) ≤ ϵ*.*

*Proof.* Similar to all other lower bound proofs, we construct such a function that satisfies the uniformly convex and Holder smooth conditions and show that it requires at least the number of ¨ iterations stated in the theorem. The construction is generally based on Nesterov's hard function [\(Nesterov et al.,](#page-11-4) [2018\)](#page-11-4), and generalizes the construction in [\(Arjevani et al.,](#page-9-0) [2019\)](#page-9-0) to higher-order and the construction in [\(Kornowski & Shamir,](#page-10-2) [2020\)](#page-10-2) to Holder smooth functions as well as uniformly ¨ convex functions.

## 5.1 FUNCTION CONSTRUCTION BASED ON NESTEROV'S HARD FUNCTION

A direct generalization of Nestrov's construction for first- and second-order lower bounds [\(Nesterov](#page-11-4) [et al.,](#page-11-4) [2018,](#page-11-4) Section 2.1.2, 4.3.1) to the p th-order Holder smooth setting takes the form ¨ ˜f(x) = p+ν P<sup>T</sup>˜ <sup>i</sup>=1 |x<sup>i</sup> − xi+1| <sup>p</sup>+<sup>ν</sup> − γx<sup>1</sup> <sup>+</sup> σ˜ q ∥x∥ q , for q < p + ν, ν ∈ [0, 1], which is uniformly convex by the regularization. We further add a coefficient so that the function p th-order Holder smooth with ¨ the desired parameter H and further on top of this a set of orthogonal basis v<sup>i</sup> , ∀ i ∈ [T˜] to limit the access of coordinates through the iterations:

$$f(\mathbf{x}) = \frac{H}{2^{p+\nu+1}(p+\nu-1)!} \left( \frac{1}{p+\nu} \sum_{i=1}^{\tilde{T}} |\langle \mathbf{v}_i, \mathbf{x} \rangle - \langle \mathbf{v}_{i+1}, \mathbf{x} \rangle|^{p+\nu} - \gamma \langle \mathbf{v}_1, \mathbf{x} \rangle \right) + \frac{\sigma}{q} \|\mathbf{x}\|^q,$$

for σ = Hσ˜ <sup>p</sup>+ν+1(p+ν−1)! , or equivalently, σ˜ = <sup>p</sup>+ν+1(p+ν−1)!σ H . v<sup>i</sup> is chosen iteratively to be orthogonal to x1, · · · , x<sup>i</sup> and v1, · · · , vi−1. Similar to [\(Arjevani et al.,](#page-9-0) [2019,](#page-9-0) Lemma 7), one can show that the oracle information of f(xi), ∀ i ≤ t does not depend on vt+1, · · · , vT˜, so that the iterative construction of v<sup>i</sup> is valid, i.e., does not affect the x<sup>i</sup> generated running an algorithm on f. Now we characterize the relation between ˜f and f.

Lemma 8. x <sup>∗</sup> = arg min<sup>x</sup> f(x)*,* y = arg min<sup>x</sup> ˜f(x)*. (i)* ∀i ∈ [T˜]*,* ⟨v<sup>i</sup> , x ∗ ⟩ = y<sup>i</sup> *. (ii)* ∥x <sup>∗</sup>∥ = ∥y∥*.*

Next, we characterize the convexity and smoothness of the constructed function. Specifically, we can show with the proof in Appendix [B](#page-19-0) that f satisfies the following lemma.

Lemma 9. f(x) *is (i) uniformly convex with degree* q *and parameter* σ*. (ii)* p th*-order Holder smooth ¨ with degree* ν *and parameter* H*.*

The analysis of [\(Nesterov et al.,](#page-11-4) [2018\)](#page-11-4) then derives the lower bound based on the closed-form optimal solution that minimizes the hard function. For our generalized construction of f, however, the closed-form solution is hard to obtain. As in [\(Arjevani et al.,](#page-9-0) [2019\)](#page-9-0), we instead analyze some properties of f for each of these lower bounds. For simplicity, we state the properties for function ˜f, and since f is simply a scaling of ˜f, the properties also apply to f with a difference of constants. To prove Theorem [2,](#page-6-1) we show separately for the <sup>H</sup> σ <sup>2</sup> 3(p+ν)−2 term and the log log σ p+ν H<sup>q</sup> 1 <sup>p</sup>+ν−<sup>q</sup> 1 ϵ term. The derivation is largely based on some key lemmas whose complete proof is in Appendix [B.](#page-19-0)

$$5.2 \quad T \in \Omega \left( \left( \frac{H}{\sigma} \right)^{\frac{2}{3(p+\nu)-2}} \right)$$

Since we cannot solve for a closed form solution from arg min<sup>x</sup> ˜f(x), we need to alternatively bound the solution in a relative scale. One key observation is that the coordinates of the optimal solution form a decreasing sequence [\(Arjevani et al.,](#page-9-0) [2019;](#page-9-0) [Carmon et al.,](#page-10-13) [2021\)](#page-10-13), and their relative relation can be characterized as in Lemma [10](#page-7-0) (i) utilizing the first-order optimality condition. Based on the properties of each coordinate, one can relate them to the norm of the optimal solution as in Lemma [10](#page-7-0) (iii).

Lemma 10. *For* y = arg min<sup>x</sup> ˜f(x)*,*

$$(i) \ \forall t \in [\tilde{T}], y_t \geq y_1 - (t-1)\gamma^{\frac{1}{p+\nu-1}}.$$

$$(i) \quad \forall t \in [T], y_t \geq y_1 - (t-1)\gamma^{\frac{p+\nu-1}{2}}.$$

$$(ii) \quad \text{For } \tilde{T} = \left\lfloor \frac{y_1}{\gamma^{\frac{p+\nu-1}{2}}} + 1 \right\rfloor, y_t \geq \gamma^{\frac{1}{p+\nu-1}} + \sqrt{\frac{2\gamma^{\frac{p+\nu}{p+\nu-1}}}{\sigma \|\mathbf{y}\|^{q-2}}}.$$

$$(iii) \quad \text{For } \gamma \geq \tilde{\sigma}^{\frac{p+\nu-1}{p+\nu-2}} \|\mathbf{y}\|^{\frac{(p+\nu-1)(q-2)}{p+\nu-2}}, \forall t \in [\tilde{T}], y_t \geq \frac{\gamma^{\frac{p+\nu}{2(p+\nu-1)}}}{2^{p+\nu+1} \tilde{\sigma}^{\frac{1}{2}} \|\mathbf{y}\|^{\frac{q-2}{q-2}}} + \left(\frac{1}{2} - i\right) \gamma^{\frac{1}{p+\nu-1}}.$$

Then the bound on the norm of the optimal solution can be established in the following lemma.

**Lemma 11.** 
$$\|y\| \leq \frac{2}{2^{3q-2}} \gamma \frac{3(p+\nu)-2}{(p+\nu-1)(3q-2)} \cdot \frac{3}{\tilde{\sigma}^{\frac{3}{3q-2}}}$$
.

The final step is to relate this norm to the optimality gap with the property of uniform convexity. By Lemma [8](#page-7-1) and Lemma [10](#page-7-0) (iii), ⟨v<sup>T</sup> , x ∗ ⟩ = y<sup>T</sup> ≥ γ p+ν 2(p+ν−1) 2 <sup>p</sup>+ν+1σ˜ 1 <sup>2</sup> ∥y∥ q−2 2 + <sup>2</sup> − T γ <sup>p</sup>+ν−<sup>1</sup> . Therefore, with v<sup>T</sup> and x<sup>T</sup> orthogonal to each other by construction,

$$\begin{aligned} f(\mathbf{x}_T) - f(\mathbf{x}^*) &\geq \frac{\sigma}{q} \|\mathbf{x}_T - \mathbf{x}^*\|^q = \frac{\sigma}{q} \left( \sum_{i=1}^{\bar{T}} (\langle \mathbf{v}_i, \mathbf{x}_T - \mathbf{x}^* \rangle)^2 \right)^{\frac{q}{2}} \geq \frac{\sigma}{q} \left( (\langle \mathbf{v}_T, \mathbf{x}_T - \mathbf{x}^* \rangle)^2 \right)^{\frac{q}{2}} \\ &= \frac{\sigma}{q} (\langle \mathbf{v}_T, \mathbf{x}^* \rangle)^q \geq \frac{\sigma}{q} \left( \frac{\gamma^{\frac{p+\nu}{2(p+\nu-1)}}}{2^{p+\nu+1} \tilde{\sigma}^{\frac{1}{2}} \|\mathbf{y}\|^{\frac{q-2}{2}}} + \left( \frac{1}{2} - T \right) \gamma^{\frac{1}{p+\nu-1}} \right)^q \end{aligned}$$

In order to achieve f(x<sup>T</sup> ) − f(x ∗ ) ≤ ϵ, we have <sup>σ</sup> q γ p+ν 2(p+ν−1) 2 <sup>p</sup>+ν+1σ˜ 1 <sup>2</sup> ∥y∥ q−2 + <sup>2</sup> − T γ 1 p+ν−1 q ≤ ϵ, from which we can solve for T ≥ γ p+ν−2 2(p+ν−1) 2 <sup>p</sup>+ν+1σ˜ 1 <sup>2</sup> ∥y∥ q−2 + 1 <sup>2</sup> − qϵ σγ q p+ν−1 1 . For ϵ ≤ γ q <sup>p</sup>+ν−<sup>1</sup> σ 2 <sup>q</sup>q , we have <sup>1</sup> <sup>2</sup> − qϵ σγ q p+ν−1 1 q ≥ 0. Therefore, T ≥ γ p+ν−2 2(p+ν−1) <sup>p</sup>+ν+1σ˜ 1 <sup>2</sup> ∥y∥ q−2 .

By Lemma [11,](#page-7-2) we know that ∥y∥ ≤ <sup>2</sup> 2 <sup>3</sup>q−<sup>2</sup> γ 3(p+ν)−2 (p+ν−1)(3q−2) σ˜ 3 3q−2 . Therefore, for x<sup>0</sup> = 0, by Lemma [8,](#page-7-1) ∥x<sup>0</sup> − x <sup>∗</sup>∥ = ∥x <sup>∗</sup>∥ = ∥y∥ ≤ <sup>2</sup> 2 <sup>3</sup>q−<sup>2</sup> γ 3(p+ν)−2 (p+ν−1)(3q−2) σ˜ 3q−2 . To satisfy the condition ∥x<sup>0</sup> − x <sup>∗</sup>∥ ≤ D, we let 2 2 <sup>3</sup>q−<sup>2</sup> γ 3(p+ν)−2 (p+ν−1)(3q−2) σ˜ 3 3q−2 ≤ D, then we can solve for γ ≤ 2 − 2(p+ν−1) 3(p+ν)−<sup>2</sup> D (p+ν−1)(3q−2) 3(p+ν)−<sup>2</sup> σ˜ 3(p+ν−1) 3(p+ν)−<sup>2</sup> . Plug this as well as ∥y∥ ≤ D into the lower bound on T we have

$$T \geq \frac{\left(2^{-\frac{2(p+\nu-1)}{3(p+\nu)-2}} D^{\frac{(p+\nu-1)(3q-2)}{3(p+\nu)-2}} \tilde{\sigma}^{\frac{3(p+\nu-1)}{3(p+\nu)-2}}\right)^{\frac{p+\nu-2}{2(p+\nu-1)}}}{2^{p+\nu+1} \tilde{\sigma}^{\frac{1}{2}} D^{\frac{q-2}{2}}} = 2^{-\frac{p+\nu-2}{3(p+\nu)-2} - (p+\nu+1)} D^{\frac{2(p+\nu-q-1)}{3(p+\nu)-2}} \tilde{\sigma}^{-\frac{2}{3(p+\nu)-2}}$$

Plugging in σ˜ = 2 <sup>p</sup>+ν+1(p+ν−1)!σ H , we have T ∈ Ω <sup>H</sup> σ <sup>2</sup> 3(p+ν)−2 .

Plugging in 
$$\tilde{\sigma} = \frac{2^{p+\nu+1}(p+\nu-1)!\sigma}{H}$$
, we have  $T \in \Omega\left(\left(\frac{H}{\sigma}\right)^{\frac{2}{3(p+\nu)-2}}\right)$ 

$$5.3 \quad T \in \Omega \left( \log \log \left( \left( \frac{\sigma^{p+\nu}}{H^q} \right)^{\frac{1}{p+\nu-q}} \frac{1}{\epsilon} \right) \right)$$

For the log log term, we follow a similar narrative as in Section [5.2,](#page-7-3) starting from characterizing the per-coordinate relation of the optimal solution.

Lemma 12. *For* y = arg min<sup>x</sup> ˜f(x)*, let* <sup>t</sup><sup>1</sup> ∈ [T˜] *be such that* <sup>y</sup><sup>t</sup><sup>1</sup> <sup>&</sup>gt; p+ν−1 σ˜ 1 <sup>p</sup>+ν−<sup>2</sup> ∥y∥ q−2 <sup>p</sup>+ν−<sup>2</sup> *and* y<sup>t</sup>1+1 ≤ p+ν−1 σ˜ 1 <sup>p</sup>+ν−<sup>2</sup> ∥y∥ q−2 <sup>p</sup>+ν−<sup>2</sup> *. Then*

- (i) 
  $$\forall i \in [\tilde{T}], y_i = y_{i+1} + \left( \tilde{\sigma} \|\mathbf{y}\|^{q-2} \sum_{j=i+1}^{\tilde{T}} y_j \right)^{\frac{1}{p+\nu-1}}$$
   and  $y_{i+1} \leq \frac{1}{\tilde{\sigma} \|\mathbf{y}\|^{q-2}} y_i^p$
- (ii)  $\forall i \geq t_1, \left( \frac{1}{c_{p,\nu}} \right)^{p+\nu-1} \frac{1}{\tilde{\sigma} \|\mathbf{y}\|^{q-2}} y_i^{p+\nu-1} \leq y_{i+1}$  where  $c_{p,\nu}$  is a constant depending on  $p, \nu$ ,
- (iii)  $\forall i \leq \tilde{T} - t_1, y_{t_1+i} \geq \left( \frac{1}{c_{p,\nu}} \right)^{\frac{(p+\nu-1)((p+\nu-1)^i-1)}{p+\nu-2}} (\tilde{\sigma} \|\mathbf{y}\|^{q-2})^{\frac{1}{p+\nu-2}} (p + \nu - 1)^{-(p+\nu-1)^i}$ .

Next, we derive the bound on the norm ∥x<sup>T</sup> −x ∗∥ q from the coordinate-wise properties in Lemma [12](#page-8-0) with the basis defined for f. When constructing the function, we choose H ≥ 2 <sup>p</sup>+ν+1(p + ν − 1)!σ so that σ˜ ≤ 1. Then for basis vector v<sup>t</sup>1+<sup>i</sup> , by Lemma [8](#page-7-1) and Lemma [12](#page-8-0) (iii),

$$\begin{aligned} \langle \mathbf{v}_{t_1+i}, \mathbf{x}^* \rangle &= y_{t_1+i} \geq \left( \frac{1}{c_{p,\nu}} \right)^{\frac{(p+\nu-1)((p+\nu-1)^i-1)}{p+\nu-2}} \cdot \bar{\sigma}^{\frac{1}{p+\nu-q}} \|\mathbf{y}\|^{\frac{q-2}{p+\nu-2}} \cdot (p+\nu-1)^{-(p+\nu-1)^i} \\ &= \left( \frac{1}{c_{p,\nu}} \right)^{\frac{(p+\nu-1)((p+\nu-1)^i-1)}{p+\nu-2}} \cdot \left( \frac{2^{p+\nu+1}(p+\nu-1)!\sigma}{H} \right)^{\frac{1}{p+\nu-q}} \|\mathbf{x}_0 - \mathbf{x}^*\|^{\frac{q-2}{p+\nu-2}} \cdot (p+\nu-1)^{-(p+\nu-1)^i}, \end{aligned}$$

for x<sup>0</sup> = 0, in which the first inequality follows from Lemma [12](#page-8-0) (iii), and then the fact that for q ≥ 2 and σ˜ ≤ 1, σ˜ <sup>p</sup>+ν−<sup>q</sup> ≤ σ˜ <sup>p</sup>+ν−<sup>2</sup> . For t<sup>1</sup> ≤ T˜ 2 ,

$$\begin{aligned} \|\mathbf{x}_T - \mathbf{x}^*\|^q &= (\|\mathbf{x}_T - \mathbf{x}^*\|^2)^{\frac{q}{2}} \geq \left( \sum_{i=1}^{\tilde{T}} (\langle \mathbf{v}_i, \mathbf{x}_T - \mathbf{x}^* \rangle)^2 \right)^{\frac{q}{2}} \geq \left( (\langle \mathbf{v}_{t_1+T}, \mathbf{x}_T - \mathbf{x}^* \rangle)^2 \right)^{\frac{q}{2}} \\ &= \left( (\langle \mathbf{v}_{t_1+T}, \mathbf{x}^* \rangle)^2 \right)^{\frac{q}{2}} = (\langle \mathbf{v}_{t_1+T}, \mathbf{x}^* \rangle)^q \end{aligned}$$

where the equality in the second line follows from the fact that by construction, v<sup>t</sup>1+<sup>T</sup> and x<sup>T</sup> are orthogonal.

Finally, with uniform convexity, we have

$$\begin{aligned} f(\mathbf{x}_T) - f(\mathbf{x}^*) &\geq \frac{\sigma}{q} \|\mathbf{x}_T - \mathbf{x}^*\|^q \\ &\geq \left( \frac{1}{c_{p,\nu}} \right)^{\frac{q(p+\nu-1)((p+\nu-1)^T-1)}{p+\nu-2}} \cdot \frac{\sigma}{q} \left( \frac{2^{p+\nu+1}(p+\nu-1)!\sigma}{H} \right)^{\frac{q}{p+\nu-q}} \|\mathbf{x}_0 - \mathbf{x}^*\|^{\frac{q(q-2)}{p+\nu-2}} \cdot (p+\nu-1)^{-q(p+\nu-1)^T} \\ &= c_{p,q,\nu} \cdot \frac{\sigma^{\frac{p+\nu}{p+\nu-q}}}{L_p^{\frac{q}{p+\nu-q}}} \cdot (p+\nu-1)^{-q(p+\nu-1)^T} \end{aligned}$$

for cp,q,ν =

2 (p+ν+1)q

<sup>p</sup>+ν−<sup>q</sup> ((p+ν−1)!)

q p+ν−q

q

<sup>c</sup>p,ν <sup>q</sup>(p+ν−1)((p+ν−1)<sup>T</sup> <sup>−</sup>1) p+ν−2

D q(q−2)

<sup>p</sup>+ν−<sup>2</sup> in which ∥x0−x

<sup>∗</sup>∥ ≤ D.

In order to achieve f(x<sup>T</sup> ) − f(x

∗

) ≤ ϵ, we have cp,q,ν ·

σ p+ν p+ν−q

L q p+ν−q ·(p + ν − 1)<sup>−</sup>q(p+ν−1)<sup>T</sup>

≤ ϵ, from

which we solve for T ≥ log logp+ν−<sup>1</sup>

 cp,q,ν σ p+q−1 p−1 L p−1 · ϵ !

+ logp+ν−<sup>1</sup>

 q 

, which completes the

proof for Theorem [2](#page-6-1) combined with the result in Section [5.2.](#page-7-3)

# 6 CONCLUSION AND FUTURE WORK

We provide tight lower bounds for minimizing functions with asymmetric high-order Holder smooth- ¨ ness and uniform convexity. Specifically, we show that the oracle complexity is lower bounded by Ω <sup>H</sup> σ <sup>2</sup> 3(p+ν)−2 σ ϵ 2(q−p−ν) <sup>q</sup>(3(p+ν)−2) for the q > p + ν case with the construction of a ℓ∞-balltruncated-Gaussian smoothed hard function, and Ω <sup>H</sup> σ <sup>2</sup> 3(p+ν)−<sup>2</sup> + log log σ p+ν H<sup>q</sup> 1 <sup>p</sup>+ν−<sup>q</sup> 1 ϵ for the q < p + ν case. Both lower bounds match the corresponding upper bounds in the general setting.

We note that the lower bounds for the q > p + ν case and the q < p + ν case are derived based on two different frameworks. The first lower bound based on Nemirovski's max function can be directly extended to hold for randomized algorithms based on "robust-zero-chain" arguments by [\(Carmon](#page-10-14) [et al.,](#page-10-14) [2020;](#page-10-14) [2021\)](#page-10-13). The second lower bound based on Nesterov's function, which is not a robust zero-chain, holds only for deterministic/zero-respecting algorithms.

We further note that the lower bound for the q = p + ν case is not included in this paper. Proposing a unified framework for all three cases as well as generalizing the results to work for randomized algorithms would be of great interest, which we leave for future work.

## REFERENCES


[1] Naman Agarwal and Elad Hazan. Lower bounds for higher-order convex optimization. In *Conference*

[2] *On Learning Theory*, pp. 774–792. PMLR, 2018. (Cited on pages [2,](#page-1-0) [3,](#page-2-1) and [4.](#page-3-2)) Yossi Arjevani, Ohad Shamir, and Ron Shiff. Oracle complexity of second-order methods for smooth convex optimization. *Mathematical Programming*, 178(1):327–360, 2019. (Cited on pages [1,](#page-0-0) [2,](#page-1-0) [4,](#page-3-2) [7,](#page-6-2) [8,](#page-7-4) [20,](#page-19-1) and [21.](#page-20-0))

[3] Dimitri P Bertsekas. Stochastic optimization problems with nondifferentiable cost functionals. *Journal of Optimization Theory and Applications*, 12(2):218–231, 1973. (Cited on page [14.](#page-13-0)) Zygmunt William Birnbaum and FC Andrews. On sums of symmetrically truncated normal random variables. *The Annals of Mathematical Statistics*, 20(3):458–461, 1949. (Cited on page [16.](#page-15-0)) Stephen P Boyd and Lieven Vandenberghe. *Convex optimization*. Cambridge university press, 2004. (Cited on page [14.](#page-13-0)) Sebastien Bubeck, Qijia Jiang, Yin Tat Lee, Yuanzhi Li, and Aaron Sidford. Near-optimal method ´ for highly smooth convex optimization. In *Conference on Learning Theory*, pp. 492–507. PMLR, 2019. (Cited on pages [1](#page-0-0) and [2.](#page-1-0)) Brian Bullins. Highly smooth minimization of non-smooth problems. In *Conference on Learning Theory*, pp. 988–1030. PMLR, 2020. (Cited on page [4.](#page-3-2)) Yair Carmon, John C Duchi, Oliver Hinder, and Aaron Sidford. Lower bounds for finding stationary points i. *Mathematical Programming*, 184(1):71–120, 2020. (Cited on page [10.](#page-9-2)) Yair Carmon, John C Duchi, Oliver Hinder, and Aaron Sidford. Lower bounds for finding stationary points ii: first-order methods. *Mathematical Programming*, 185(1):315–355, 2021. (Cited on pages [8](#page-7-4) and [10.](#page-9-2)) Jack Cartinhour. One-dimensional marginal density functions of a truncated multivariate normal density function. *Communications in Statistics-Theory and Methods*, 19(1):197–203, 1990. (Cited on pages [3,](#page-2-1) [4,](#page-3-2) and [13.](#page-12-2)) Hao Chen, Lanshan Han, and Alvin Lim. A note on the sum of non-identically distributed doubly truncated normal distributions. *arXiv preprint arXiv:2008.07954*, 2020. (Cited on pages [15](#page-14-0) and [16.](#page-15-0)) Nikita Doikov. Lower complexity bounds for minimizing regularized functions. *arXiv preprint arXiv:2202.04545*, 2022. (Cited on pages [2,](#page-1-0) [3,](#page-2-1) [4,](#page-3-2) [5,](#page-4-5) [6,](#page-5-0) and [19.](#page-18-0)) Nikita Doikov and Yurii Nesterov. Minimizing uniformly convex functions by cubic regularization of newton method. *Journal of Optimization Theory and Applications*, 189(1):317–339, 2021. (Cited on page [2.](#page-1-0)) John C Duchi, Peter L Bartlett, and Martin J Wainwright. Randomized smoothing for stochastic optimization. *SIAM Journal on Optimization*, 22(2):674–701, 2012. (Cited on pages [4,](#page-3-2) [14,](#page-13-0) and [15.](#page-14-0)) Ankit Garg, Robin Kothari, Praneeth Netrapalli, and Suhail Sherif. Near-optimal lower bounds for convex optimization for all orders of smoothness. *Advances in Neural Information Processing Systems*, 34:29874–29884, 2021. (Cited on pages [2](#page-1-0) and [3.](#page-2-1)) Alexander Gasnikov, Pavel Dvurechensky, Eduard Gorbunov, Evgeniya Vorontsova, Daniil Selikhanovych, and Cesar A Uribe. Optimal tensor methods in smooth convex and uniformly ´ convexoptimization. In *Conference on Learning Theory*, pp. 1374–1391. PMLR, 2019. (Cited on pages [1](#page-0-0) and [2.](#page-1-0)) Cristobal Guzm ´ an and Arkadi Nemirovski. On lower complexity bounds for large-scale smooth ´ convex optimization. *Journal of Complexity*, 31(1):1–14, 2015. (Cited on pages [2,](#page-1-0) [3,](#page-2-1) [5,](#page-4-5) and [6.](#page-5-0)) Bo Jiang, Haoyue Wang, and Shuzhong Zhang. An optimal high-order tensor method for convex optimization. In *Conference on Learning Theory*, pp. 1799–1801. PMLR, 2019. (Cited on page [2.](#page-1-0)) Anatoli Juditsky and Yuri Nesterov. Primal-dual subgradient methods for minimizing uniformly convex functions. *arXiv preprint arXiv:1401.1792*, 2014. (Cited on pages [2](#page-1-0) and [5.](#page-4-5)) Guy Kornowski and Ohad Shamir. High-order oracle complexity of smooth and strongly convex optimization. *arXiv preprint arXiv:2010.06642*, 2020. (Cited on pages [1,](#page-0-0) [2,](#page-1-0) and [7.](#page-6-2)) Dmitry Kovalev and Alexander Gasnikov. The first optimal acceleration of high-order methods in smooth convex optimization. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), *Advances in Neural Information Processing Systems*, 2022. URL <https://openreview.net/forum?id=YgmiL2Ur01P>. (Cited on page [2.](#page-1-0))

[4] Hariharan Lakshmanan and Daniela Pucci De Farias. Decentralized resource allocation in dynamic networks of agents. *SIAM Journal on Optimization*, 19(2):911–940, 2008. (Cited on page [15.](#page-14-0)) Renato DC Monteiro and Benar Fux Svaiter. An accelerated hybrid proximal extragradient method for convex optimization and its implications to second-order methods. *SIAM Journal on Optimization*, 23(2):1092–1125, 2013. (Cited on page [2.](#page-1-0)) Arkaddii S Nemirovskii and Yu E Nesterov. Optimal methods of smooth convex minimization. *USSR Computational Mathematics and Mathematical Physics*, 25(2):21–30, 1985. (Cited on page [3.](#page-2-1)) Yurii Nesterov and Boris T Polyak. Cubic regularization of newton method and its global performance. *Mathematical programming*, 108(1):177–205, 2006. (Cited on page [1.](#page-0-0)) Yurii Nesterov and Vladimir Spokoiny. Random gradient-free minimization of convex functions. *Foundations of Computational Mathematics*, 17(2):527–566, 2017. (Cited on page [15.](#page-14-0)) Yurii Nesterov et al. *Lectures on convex optimization*, volume 137. Springer, 2018. (Cited on pages [2,](#page-1-0) [3,](#page-2-1) [7,](#page-6-2) [8,](#page-7-4) and [18.](#page-17-0)) Vincent Roulet and Alexandre d'Aspremont. Sharpness, restart and acceleration. *Advances in Neural Information Processing Systems*, 30, 2017. (Cited on page [2.](#page-1-0)) Chaobing Song, Yong Jiang, and Yi Ma. Unified acceleration of high-order algorithms under general holder continuity. *SIAM Journal on Optimization*, 31(3):1797–1826, 2021. (Cited on pages [1,](#page-0-0) [2,](#page-1-0) and [5.](#page-4-5)) Daniel Berg Thomsen and Nikita Doikov. Complexity of minimizing regularized convex quadratic functions. *arXiv preprint arXiv:2404.17543*, 2024. (Cited on pages [2,](#page-1-0) [3,](#page-2-1) and [5.](#page-4-5)) Farzad Yousefian, Angelia Nedic, and Uday V Shanbhag. On stochastic gradient and subgradient ´ methods with adaptive steplength sequences. *Automatica*, 48(1):56–67, 2012. (Cited on page [4.](#page-3-2)) Wenqi Zhu and Coralia Cartis. Quartic polynomial sub-problem solutions in tensor methods for nonconvex optimization. In *NeurIPS 2022 Workshop*, 2022. (Cited on page [1.](#page-0-0))
# Appendices

# A PROOF FOR TECHNICAL LEMMAS IN SECTION [4](#page-2-2)

#### A.1 PROPERTIES OF TRUNCATED GAUSSIAN SMOOTHING

Lemma 13 (ℓ∞-Ball Truncated Gaussian and Its Marginal Distribution). *For standard MVN truncated in a unit* ℓ∞*-ball as defined in Definition [4:](#page-2-0)*

$$\mathbb{P}[V = \mathbf{v}] = \frac{1}{Z(2\pi)^{\frac{d}{2}}} \exp \left\{ -\frac{\mathbf{v}^\top \mathbf{v}}{2} \right\} \mathbb{I}[\|\mathbf{v}\|_\infty \leq 1],$$

- *(i) The cumulative distribution within the* ℓ∞*-ball, i.e., the normalizing factor* Z(d) = [Φ(1) − Φ(−1)]<sup>d</sup> *.*
- *(ii) The marginal distribution is a standard normal truncated within* [−1, 1]*.*

*Proof.* (i) By Eq. (3) in [\(Cartinhour,](#page-10-10) [1990\)](#page-10-10), we know that

$$\begin{aligned} Z(d) &= \int_{\|\mathbf{v}\|_\infty \leq 1} \frac{1}{(2\pi)^{\frac{d}{2}}} \exp \left\{ -\frac{\mathbf{v}^\top \mathbf{v}}{2} \right\} d\mathbf{v} \\ &= \underbrace{\int_{-1}^1 \cdots \int_{-1}^1}_{d\text{-time integration, one for each coordinate}} \frac{1}{(2\pi)^{\frac{d}{2}}} \exp \left\{ -\frac{\sum_{i=1}^d v_i^2}{2} \right\} dv_1 \cdots dv_d \\ &= \prod_{i=1}^d \int_{-1}^1 \frac{1}{\sqrt{2\pi}} \exp \left\{ -\frac{v_i^2}{2} \right\} dv_i \\ &= [\Phi(1) - \Phi(-1)]^d. \end{aligned}$$

(ii) By Eq. (4) and (16) in [\(Cartinhour,](#page-10-10) [1990\)](#page-10-10), ∀i ∈ [d],

$$\begin{aligned}\mathbb{P}[V_i] &= \frac{\exp\left\{-\frac{v_i^2}{2}\right\}}{[\Phi(1) - \Phi(-1)]^d \sqrt{2\pi}} \underbrace{\int_{-1}^1 \dots \int_{-1}^1}_{d-1\text{-time integration}} \frac{\exp\left\{-\frac{\sum_{j \neq i} v_j^2}{2}\right\}}{(2\pi)^{\frac{d-1}{2}}} dv_1 \dots dv_{i-1} dv_{i+1} \dots dv_i \\ &= \frac{\exp\left\{-\frac{v_i^2}{2}\right\}}{[\Phi(1) - \Phi(-1)]^d \sqrt{2\pi}} \prod_{j \neq i} \int_{-1}^1 \frac{1}{\sqrt{2\pi}} \exp\left\{-\frac{v_j^2}{2}\right\} dv_j \\ &= \frac{\exp\left\{-\frac{v_i^2}{2}\right\}}{[\Phi(1) - \Phi(-1)]^d \sqrt{2\pi}} [\Phi(1) - \Phi(-1)]^{d-1} \\ &= \frac{1}{[\Phi(1) - \Phi(-1)] \sqrt{2\pi}} \exp\left\{-\frac{v_i^2}{2}\right\} \\ &= \frac{1}{\sqrt{2\pi} \int_{-1}^1 \frac{1}{\sqrt{2\pi}} \exp\left\{-\frac{v_i^2}{2}\right\} dv_i} \exp\left\{-\frac{v_i^2}{2}\right\},\end{aligned}$$

if −1 ≤ V<sup>i</sup> ≤ 1, otherwise <sup>P</sup> [V<sup>i</sup> ] = 0. Therefore, V<sup>i</sup> follows the truncated standard normal distribution within [−1, 1].

Lemma 14 (Properties of Truncated Gaussian Smoothing). *For a function* f : R <sup>d</sup> → <sup>R</sup> *that is* L*-Lipschitz with respect to the* ℓ<sup>2</sup> *norm, then* ∀ x ∈ <sup>R</sup> d *,*

- *(i) If* f *is convex and non-differentiable in a set with Lebesgue measure* 0*, then* f<sup>ρ</sup> *is continuously differentiable and* ∇fρ(x) = <sup>E</sup><sup>V</sup> [∂f(x + ρV )] *for some random variable* V *.*
- *(ii) If* f *is convex,* f<sup>ρ</sup> *is convex and* L*-Lipschitz with respect to the* ℓ<sup>2</sup> *norm.*

*(iii) If* f *is convex,* f(x) ≤ fρ(x) ≤ f(x) + <sup>5</sup>

4 Lρ√ d*.*

*(iv)* ∇f<sup>ρ</sup> *is* <sup>2</sup>

ρ

L*-Lipschitz, i.e.,* f<sup>ρ</sup> *is* <sup>2</sup>

ρ

L*-smooth.*

*Proof.* The proof of this lemma is based on that of Lemma 9 in [\(Duchi et al.,](#page-10-11) [2012\)](#page-10-11).

(i) The differentiability is established in [\(Bertsekas,](#page-10-15) [1973,](#page-10-15) Proposition 2.3), and ∇fρ(x) = <sup>E</sup><sup>V</sup> [∂f(x+ ρV )] in [\(Bertsekas,](#page-10-15) [1973,](#page-10-15) Proposition 2.2).

(ii) Expectation preserves convexity [\(Boyd & Vandenberghe,](#page-10-16) [2004,](#page-10-16) Section 3.2.1), therefore, given that f is convex, by definition, f<sup>ρ</sup> is also convex. For Lipschitz continuity, by the second part of (i) and Jensen's inequality, we have

$$\begin{aligned}\|\nabla f_\rho(\mathbf{x})\| &= \|\mathbb{E}_V[\partial f(\mathbf{x} + \rho V)]\| \\ &\leq \mathbb{E}_V[\|\partial f(\mathbf{x} + \rho V)\|].\end{aligned}$$

Given that f is L-Lipschitz over R <sup>d</sup> with respect to the ℓ<sup>2</sup> norm, it is implied that ∀x ∈ <sup>R</sup> d , ∥∂f(x)∥ ≤ L. As a result, ∥∇fρ(x)∥ ≤ <sup>E</sup>[L] ≤ L which further implies that f<sup>ρ</sup> is L-Lipschitz with respect to the ℓ<sup>2</sup> norm.

(iii) For fρ(x) = <sup>E</sup><sup>V</sup> [f(x + ρV )], <sup>E</sup><sup>V</sup> [V ] = 0 by construction. And since smoothing preserves convexity, fρ(x) is also convex. For the lower bound, using Jensen's inequality,

$$\begin{aligned} f(\mathbf{x}) &= f(\mathbf{x} + \rho \mathbb{E}_V[V]) \\ &= f(\mathbb{E}_V[\mathbf{x} + \rho V]) \\ &\leq \mathbb{E}_V[f(\mathbf{x} + \rho V)] \\ &= f_\rho(\mathbf{x}). \end{aligned}$$

For the upper bound, since f is L-Lipschitz in ℓ2-norm, f(x + ρV ) − f(x) ≤ L∥ρV ∥. Therefore,

$$\begin{aligned} f_\rho(\mathbf{x}) &= \mathbb{E}_V[f(\mathbf{x} + \rho V)] \\ &\leq \mathbb{E}_V[f(\mathbf{x}) + L\rho \|V\|] \\ &= f(\mathbf{x}) + L\rho \mathbb{E} \left[ \sqrt{\sum_{i=1}^d V_i^2} \right] \\ &\leq f(\mathbf{x}) + L\rho \sqrt{\sum_{i=1}^d \mathbb{E}[V_i^2]}. \end{aligned}$$

By Lemma [13](#page-12-3) (ii), V<sup>i</sup> follows the standard normal distribution truncated within [−1, 1]. Therefore, let Φ(·) denote the cumulative distribution function of standard normal distribution, then

$$\begin{aligned}\mathbb{E} [V_i^2] &= \int_{-1}^1 \frac{\phi(\tau)}{\Phi(1) - \Phi(-1)} \tau^2 d\tau \\ &= \frac{1}{\Phi(1) - \Phi(-1)} \int_{-1}^1 \phi(\tau) \tau^2 d\tau \\ &\leq \frac{1}{\Phi(1) - \Phi(-1)} \int_{-\infty}^{\infty} \phi(\tau) \tau^2 d\tau \\ &= \frac{\mathbb{E} [U_i^2]}{\Phi(1) - \Phi(-1)}\end{aligned}$$

for U<sup>i</sup> ∼ N (0, 1), ∀i ∈ [d]. Then for U = [U1, · · · , Ud] <sup>⊤</sup>, U follows the standard MVN distribution and

$$\begin{aligned} f_\rho(\mathbf{x}) &\leq f(\mathbf{x}) + L\rho \sqrt{\frac{\mathbb{E} \left[ \sum_{i=1}^d U_i^2 \right]}{\Phi(1) - \Phi(-1)}} \\ &= f(\mathbf{x}) + L\rho \sqrt{\frac{\mathbb{E} [\|U\|^2]}{\Phi(1) - \Phi(-1)}}. \end{aligned}$$

E -∥U∥ 2 is the second moment of the standard MVN, which is bounded by the dimension d [\(Nesterov](#page-11-9) [& Spokoiny,](#page-11-9) [2017,](#page-11-9) Lemma 1). We know that Φ(1) − Φ(−1) ≈ 0.6827. As a result, we have

$$f_\rho(\mathbf{x}) \leq f(\mathbf{x}) + \frac{5}{4}L\rho\sqrt{d}.$$

(iv) The proof of this lemma follows that of Lemma 3.3 point 3 in [\(Lakshmanan & De Farias,](#page-11-10) [2008\)](#page-11-10), also seen in that of Lemma 9 (iii) in [\(Duchi et al.,](#page-10-11) [2012\)](#page-10-11). Denote the PDF of the unit ℓ∞-ball-truncated standard MVN as ϕ∥·∥∞≤1(·; 0, 1). Then for fρ(x) = <sup>E</sup><sup>V</sup> [f(x+ρV )], ρV has PDF ϕ∥·∥∞≤ρ(·; 0, ρ<sup>2</sup> ) by Lemma 2 (v) in [\(Chen et al.,](#page-10-17) [2020\)](#page-10-17). By [\(Duchi et al.,](#page-10-11) [2012,](#page-10-11) Lemma 11), ∀ x, x ′ ∈ <sup>R</sup> d , for Z from ϕ∥·∥∞≤ρ(·; 0, ρ<sup>2</sup> ),

$$\|\nabla f_\rho(\mathbf{x}) - \nabla f_\rho(\mathbf{x}')\|_2 \leq L \underbrace{\int |\phi_{\|\cdot\|_\infty \leq \rho}(\mathbf{z} - \mathbf{x}; 0, \rho^2) - \phi_{\|\cdot\|_\infty \leq \rho}(\mathbf{z} - \mathbf{x}'; 0, \rho^2)| \, d\mathbf{z}}_I.$$

Now we bound the integral. Note that ∀ x, ϕ∥·∥∞≤ρ(x; 0, ρ<sup>2</sup> ) is a truncated MVN symmetrically centered at the origin, consequently, is strictly decreasing with respect to ∥x∥<sup>2</sup> . As a result, ϕ∥·∥∞≤ρ(z − x; 0, ρ<sup>2</sup> ) ≥ ϕ∥·∥∞≤ρ(z − x ′ ; 0, ρ<sup>2</sup> ) if and only if ∥z − x∥<sup>2</sup> ≤ ∥z − x ′∥2. Therefore,

$$\begin{aligned} I &= 2 \int_{\|\mathbf{z}-\mathbf{x}\|_2 \leq \|\mathbf{z}-\mathbf{x}'\|_2} (\phi_{\|\cdot\|_\infty} \leq \rho(\mathbf{z} - \mathbf{x}; 0, \rho^2) - \phi_{\|\cdot\|_\infty} \leq \rho(\mathbf{z} - \mathbf{x}'; 0, \rho^2)) d\mathbf{z} \\ &= 2 \int_{\|\mathbf{z}-\mathbf{x}\|_2 \leq \|\mathbf{z}-\mathbf{x}'\|_2} \phi_{\|\cdot\|_\infty} \leq \rho(\mathbf{z} - \mathbf{x}; 0, \rho^2) d\mathbf{z} - 2 \int_{\|\mathbf{z}-\mathbf{x}\|_2 \leq \|\mathbf{z}-\mathbf{x}'\|_2} \phi_{\|\cdot\|_\infty} \leq \rho(\mathbf{z} - \mathbf{x}'; 0, \rho^2) d\mathbf{z}. \end{aligned}$$

Denote y = z − x and y ′ = z − x ′ , then

$$\begin{aligned} I &= 2 \int_{\|\mathbf{y}\|_2 \leq \|\mathbf{y} - (\mathbf{x}' - \mathbf{x})\|_2} \phi_{\|\cdot\|_\infty \leq \rho}(\mathbf{y}; 0, \rho^2) d\mathbf{y} - 2 \int_{\|\mathbf{y}'\|_2 \geq \|\mathbf{y}' - (\mathbf{x} - \mathbf{x}')\|_2} \phi_{\|\cdot\|_\infty \leq \rho}(\mathbf{y}'; 0, \rho^2) d\mathbf{y}' \\ &= 2\mathbb{P}_{\phi_{\|\cdot\|_\infty \leq \rho}} [\|Z\|_2 \leq \|Z - (\mathbf{x}' - \mathbf{x})\|_2] - 2\mathbb{P}_{\phi_{\|\cdot\|_\infty \leq \rho}} [\|Z'\|_2 \geq \|Z' - (\mathbf{x} - \mathbf{x}')\|_2] \\ &= 2\mathbb{P}_{\phi_{\|\cdot\|_\infty \leq \rho}} [\|Z\|_2^2 \leq \|Z - (\mathbf{x}' - \mathbf{x})\|_2^2] - 2\mathbb{P}_{\phi_{\|\cdot\|_\infty \leq \rho}} [\|Z'\|_2^2 \geq \|Z' - (\mathbf{x} - \mathbf{x}')\|_2^2] \\ &= 2\mathbb{P}_{\phi_{\|\cdot\|_\infty \leq \rho}} [2\langle Z, \mathbf{x}' - \mathbf{x} \rangle \leq \|\mathbf{x}' - \mathbf{x}\|_2^2] - 2\mathbb{P}_{\phi_{\|\cdot\|_\infty \leq \rho}} [2\langle Z', \mathbf{x} - \mathbf{x}' \rangle \geq \|\mathbf{x} - \mathbf{x}'\|_2^2] \\ &= 2\mathbb{P}_{\phi_{\|\cdot\|_\infty \leq \rho}} \left[ \left\langle Z, \frac{\mathbf{x}' - \mathbf{x}}{\|\mathbf{x}' - \mathbf{x}\|_2} \right\rangle \leq \frac{\|\mathbf{x}' - \mathbf{x}\|_2}{2} \right] - 2\mathbb{P}_{\phi_{\|\cdot\|_\infty \leq \rho}} \left[ \left\langle Z', \frac{\mathbf{x} - \mathbf{x}'}{\|\mathbf{x} - \mathbf{x}'\|_2} \right\rangle \geq \frac{\|\mathbf{x} - \mathbf{x}'\|_2}{2} \right] \end{aligned}$$

Denote W = D Z, <sup>x</sup> ′−x ∥x′−x∥<sup>2</sup> E and W′ = D Z ′ , x−x ′ ∥x−x′∥<sup>2</sup> E . Since <sup>x</sup> ′−x ∥x′−x∥<sup>2</sup> and <sup>x</sup>−<sup>x</sup> ′ ∥x−x′∥<sup>2</sup> are normalized vectors, W and W′ follow the one-dimensional distribution projected onto a plane along some direction from the truncated multivariate Gaussian, which is symmetrically centered at the origin. Therefore, by symmetry,

$$\begin{aligned} I &= 2\mathbb{P} \left[ W \leq \frac{\|\mathbf{x}' - \mathbf{x}\|_2}{2} \right] - 2\mathbb{P} \left[ W' \geq \frac{\|\mathbf{x} - \mathbf{x}'\|_2}{2} \right] \\ &= 2\mathbb{P} \left[ W \leq -\frac{\|\mathbf{x}' - \mathbf{x}\|_2}{2} \right] + 2\mathbb{P} \left[ -\frac{\|\mathbf{x}' - \mathbf{x}\|_2}{2} \leq W \leq \frac{\|\mathbf{x}' - \mathbf{x}\|_2}{2} \right] - 2\mathbb{P} \left[ W' \geq \frac{\|\mathbf{x} - \mathbf{x}'\|_2}{2} \right] \\ &= 2\mathbb{P} \left[ -\frac{\|\mathbf{x}' - \mathbf{x}\|_2}{2} \leq W \leq \frac{\|\mathbf{x}' - \mathbf{x}\|_2}{2} \right] \end{aligned}$$

As we later upper bound the integration by the peak of this distribution, we know by the geometry of ℓ∞-ball that the projection onto the diagonal yields the highest peak, and that is when W = √ d P<sup>d</sup> <sup>i</sup>=1 Z<sup>i</sup> for Z<sup>i</sup> being the marginal of Z that follows the truncated Gaussian distribution on [−ρ, ρ] by Lemma [<sup>13</sup>](#page-12-3) (ii). And further by Lemma 2 (v) in [\(Chen et al.,](#page-10-17) [2020\)](#page-10-17), √ Z<sup>i</sup> d is also a truncated Gaussian whose PDF is <sup>ϕ</sup>[<sup>−</sup> <sup>√</sup><sup>ρ</sup> d , √ρ (w; 0, ρ d ). As a result, W is the sum of independent identically

Importantly, the PDF is strictly decreasing with respect to ∥·∥2, not ∥·∥∞, no matter in which norm the truncation is done, as long as centered at the origin.

distributed (i.i.d.) truncated Gaussian variables. By Theorem 3 in [\(Chen et al.,](#page-10-17) [2020\)](#page-10-17) and E.q. (4.2) in [\(Birnbaum & Andrews,](#page-10-18) [1949\)](#page-10-18) we know the sum of truncated Gaussian variables converges to a normal distribution for large d. As a result, W ∼ P<sup>d</sup> <sup>i</sup>=1 Var [Z<sup>i</sup> N (0, 1). Knowing from the CDF of truncated Gaussian that ∀ i ∈ [d], Var [Z<sup>i</sup> ] = <sup>σ</sup> d 1 − ϕ(1)+ϕ(−1) Φ(1)−Φ(−1) − ϕ(1)−ϕ(−1) Φ(1)−Φ(−1)<sup>2</sup> = 0.7089 <sup>ρ</sup> d , we have P<sup>d</sup> <sup>i</sup>=1 Var [Z<sup>i</sup> = 0.7089ρ 2

$$\mathbb{P} \left[ -\frac{\|\mathbf{x}' - \mathbf{x}\|_2}{2} \leq W \leq \frac{\|\mathbf{x}' - \mathbf{x}\|_2}{2} \right] = \frac{1}{\sqrt{2\pi}\sqrt{0.7089}\rho} \int_{-\frac{\|\mathbf{x}' - \mathbf{x}\|_2}{2}}^{\frac{\|\mathbf{x}' - \mathbf{x}\|_2}{2}} \exp\{-\frac{w^2}{2 \times 0.7089\rho^2}\} dw$$

Furthermore, since the PDF takes its peak at w = 0, we have

$$\begin{aligned} I &\leq 2 \times \frac{2}{\sqrt{2\pi}\rho} \int_{-\frac{\|\mathbf{x}' - \mathbf{x}\|_2}{2}}^{\frac{\|\mathbf{x}' - \mathbf{x}\|_2}{2}} dw \\ &= \frac{4\|\mathbf{x}' - \mathbf{x}\|_2}{\sqrt{2\pi}\rho} \end{aligned}$$

Therefore,

$$\begin{aligned}\|\nabla f_\rho(\mathbf{x}) - \nabla f_\rho(\mathbf{x}')\|_2 &\leq LI \\ &\leq \frac{2L}{\rho}\|\mathbf{x}' - \mathbf{x}\|_2.\end{aligned}$$

Lemma 1. *Given a L-Lipschitz function* f*, the function* f p <sup>ρ</sup> = Sρ[· · · [Sρ[f]] · · · ] *satisfies*

- *(i) If* f *is convex,* f p ρ *is convex and* L*-Lipschitz with respect to the* ℓ<sup>2</sup> *norm.*
- *(ii) If* f *is convex,* f(x) ≤ f p ρ
- (x) ≤ f(x) + <sup>5</sup><sup>p</sup> 4 Lρ√
  - d*.*
- *(iii)* ∀i ∈ [p]*,* ∀x, x ′ ∈ <sup>R</sup> d *,* ∥∇<sup>i</sup>f p ρ
- (x) − ∇<sup>i</sup>f p ρ (x ′ )∥ ≤ 2 ρ i L∥x − x ′∥*.*

*Proof.* The proof of this lemma relies on inductively applying Lemma [14](#page-12-1) and we provide formal proof by induction.

(i) The base case p = 1 holds directly by Lemma [14](#page-12-1) (ii). Then we state the hypothesis that for p = k, f k ρ is convex and L-Lipschitz with respect to the ℓ<sup>2</sup> norm. For the induction step, we have, by definition, f k+1 <sup>ρ</sup> = Sρ[f k ρ ] where f k ρ is convex and L-Lipschitz with respect to the ℓ<sup>2</sup> norm by our hypothesis, with which f k ρ satisfies the condition of Lemma [14.](#page-12-1) Then by Lemma [14](#page-12-1) (ii), f k+1 ρ is convex and L-Lipschitz with respect to the ℓ<sup>2</sup> norm.

(ii) The base case p = 1 holds directly by Lemma [14](#page-12-1) (iii). Then we state the hypothesis that for p = k, f(x) ≤ f k ρ (x) ≤ f(x) + <sup>5</sup><sup>k</sup> 4 Lρ√ d holds. From the result of (i), we know that f k ρ satisfies the condition of Lemma [14.](#page-12-1) Therefore, applying [14](#page-12-1) (iii) to the function f k ρ (x) we have for the lower bound

$$f_\rho^{k+1}(\mathbf{x}) \geq f_\rho^k(\mathbf{x}) \geq f(\mathbf{x})$$

and for the lower bound

$$f_\rho^{k+1}(\mathbf{x}) \leq f_\rho^k(\mathbf{x}) + \frac{5}{4}L\rho\sqrt{d} \leq f(\mathbf{x}) + \frac{5k}{4}L\rho\sqrt{d} + \frac{5}{4}L\rho\sqrt{d} = f(\mathbf{x}) + \frac{5(k+1)}{4}L\rho\sqrt{d}$$

which completes the induction step.

(iii) The base case p = 1 holds for i = 0 by Lemma [14](#page-12-1) (ii) and for i = 1 by Lemma [14](#page-12-1) (iv). Now we state the inductive hypothesis that for p = k, it holds that ∀ x, x ′ ∈ <sup>R</sup>,

$$\forall i \in [k], \|\nabla^i f_\rho^k(\mathbf{x}) - \nabla^i f_\rho^k(\mathbf{x}')\| \leq \left(\frac{2}{\rho}\right)^i L\|\mathbf{x} - \mathbf{x}'\|.$$

That is, ∀ i ∈ [k], the function ∇<sup>i</sup>f k ρ is ρ i L -Lipschitz. Then for p = k + 1, ∀i ∈ [k + 1],

$$\begin{aligned} \|\nabla^i f_\rho^{k+1}(\mathbf{x}) - \nabla^i f_\rho^{k+1}(\mathbf{x}')\| &= \|\nabla^i S_\rho[f_\rho^k](\mathbf{x}) - \nabla^i S_\rho[f_\rho^k](\mathbf{x}')\| \\ &= \|S_\rho[\nabla^i f_\rho^k](\mathbf{x}) - S_\rho[\nabla^i f_\rho^k](\mathbf{x}')\| \\ &= \|\mathbb{E}_V[\nabla^i f_\rho^k(\mathbf{x} + \rho V)] - \mathbb{E}_V[\nabla^i f_\rho^k(\mathbf{x}' + \rho V)]\| \\ &= \|\mathbb{E}_V[\nabla^i f_\rho^k(\mathbf{x} + \rho V)] - \nabla^i f_\rho^k(\mathbf{x}' + \rho V)\| \\ &\leq \mathbb{E}_V[\|\nabla^i f_\rho^k(\mathbf{x} + \rho V) - \nabla^i f_\rho^k(\mathbf{x}' + \rho V)\|] \end{aligned}$$

where the first equality holds by definition, the second equality by the fact that expectation and derivative commute for differentiable functions, and the last inequality by the Jensen's.

For i < k + 1, we can directly apply Lemma [14](#page-12-1) (iv), with the hypothesis as the condition, on the function ∇<sup>i</sup>f k ρ , to establish the result that ∇<sup>i</sup>f k ρ is smooth with parameter ρ i L. Therefore,

$$\begin{aligned} \|\nabla^i f_\rho^{k+1}(\mathbf{x}) - \nabla^i f_\rho^{k+1}(\mathbf{x}')\| &\leq \mathbb{E}_V [\|\nabla^i f_\rho^k(\mathbf{x} + \rho V) - \nabla^i f_\rho^k(\mathbf{x}' + \rho V)\|] \\ &\leq \mathbb{E}_V \left[ \left( \frac{2}{\rho} \right)^i L \|\mathbf{x} - \mathbf{x}'\| \right] \\ &= \left( \frac{2}{\rho} \right)^i L \|\mathbf{x} - \mathbf{x}'\| \end{aligned}$$

For i = k + 1, we have from our i < k + 1 case that the function ∇<sup>k</sup>f k+1 ρ is 2 ρ k L -Lipschitz. We can therefore apply Lemma [14](#page-12-1) (iv) on ∇<sup>k</sup>f k+1 ρ and claim that it's also smooth with parameter ρ · ρ k L = ρ <sup>k</sup>+1 L. That is,

$$\|\nabla [\nabla^k f_\rho^{k+1}] (\mathbf{x}) - \nabla [\nabla^k f_\rho^{k+1}] (\mathbf{x}')\| \leq \left(\frac{2}{\rho}\right)^{k+1} L \|\mathbf{x} - \mathbf{x}'\|,$$

which completes the proof.

### A.2 PROPERTIES OF THE CONSTRUCTED HARD FUNCTION

Lemma 2. ∀ t ∈ [T]*,* g<sup>t</sup> *is convex and* 1*-Lipschitz with respect to the* ℓ∞*-norm, and also the* ℓ2*-norm.*

*Proof.* (1) For convexity, by definition we have

$$g_t(\mathbf{x}) = \max_{1 \leq k \leq t} r_k(\mathbf{x}) \quad \text{where} \quad \forall k \in [T], r_k(\mathbf{x}) = \xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x} \rangle - (k-1)\delta,$$

Since rk(x) is linear in x, rk(x) is convex. Then gt(x) is the maximum of convex functions which is also convex.

(2) To show Lipschitzness, ∀ x, y ∈ R d , without the loss of generality, denote

$$k_1 = \arg \max_{1 \leq k \leq t} r_k(\mathbf{x}) \qquad k_2 = \arg \max_{1 \leq k \leq t} r_k(\mathbf{y}).$$

Therefore,

$$g_t(\mathbf{x}) = \xi_{k_1} x_{\alpha(k_1)} - (k_1 - 1)\delta \qquad g_t(\mathbf{y}) = \xi_{k_2} y_{\alpha(k_2)} - (k_2 - 1)\delta.$$

Since

$$\begin{aligned} g_t(\mathbf{y}) &= \xi_{k_2} y_{\alpha(k_2)} - (k_2 - 1)\delta \\ &= \max_{1 \leq k \leq t} \xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x} \rangle - (k - 1)\delta \\ &\geq \xi_{k_1} y_{\alpha(k_1)} - (k_1 - 1)\delta, \end{aligned}$$

we have

$$\begin{aligned} g_t(\mathbf{x}) - g_t(\mathbf{y}) &\leq (\xi_{k_1} x_{\alpha(k_1)} - (k_1 - 1)\delta) - (\xi_{k_1} y_{\alpha(k_1)} - (k_1 - 1)\delta) \\ &\leq |x_{\alpha(k_1)} - y_{\alpha(k_1)}| \\ &\leq \max_{1 \leq i \leq d} |x_i - y_i| \\ &= \|\mathbf{x} - \mathbf{y}\|_{\infty} \\ &\leq \|\mathbf{x} - \mathbf{y}\|_2, \end{aligned}$$

where the last two inequalities show Lipschitzness in ℓ<sup>∞</sup> and ℓ<sup>2</sup> norm respectively.

Lemma 3. ∀ t ∈ [T]*,* ∀ x, y ∈ <sup>R</sup> d *,*

- (i) 
  $$G_t(\mathbf{x})$$
   is convex and  $I$ -Lipschitz, i.e.,  $G_t(\mathbf{x}) - G_t(\mathbf{y}) \leq \|\mathbf{x} - \mathbf{y}\|$ .
- (ii)  $g_t(\mathbf{x}) \leq G_t(\mathbf{x}) \leq g_t(\mathbf{x}) + \frac{5}{4}p\rho\sqrt{d}$ .

*(iii) For some fixed* p ∈ Z <sup>+</sup>*,* <sup>∀</sup> <sup>i</sup> <sup>∈</sup> [p]*,* ∥∇<sup>i</sup>Gt(x) − ∇<sup>i</sup>Gt(y)∥ ≤ ρ i ∥x − y∥*.*

*Proof.* The proof follows directly from that for Lemma [1.](#page-3-0)

Lemma 4. *For* F(x) = f<sup>T</sup> (x) + dq(x) *where* dq(x) = <sup>σ</sup> q x q *and* x ∈ Q*,*

- *(i)* F *is uniformly convex function with degree* q *and modulus* σ > 0*.*
- *(ii)* F(x) *is* p th*-order Holder smooth with parameter ¨* H = 2 <sup>p</sup>+1β <sup>ρ</sup><sup>p</sup>+ν−<sup>1</sup> *,* ∀ p ∈ <sup>Z</sup> +*.*

*Proof.* (i) It is shown in [\(Nesterov et al.,](#page-11-4) [2018,](#page-11-4) Section 4.2.2) that <sup>σ</sup> q x q is uniformly convex with degree q and parameter σ. By Lemma [3](#page-4-6) (i), G<sup>T</sup> is convex, therefore f is also convex, so that ∀ x, y ∈ <sup>R</sup>, ⟨∇f(x) − ∇f(y), x − y⟩ ≥ 0. Therefore, by Definition [3,](#page-2-3) D ∇( σ q x q ) − ∇( σ q y q ), x − y E ≥ σ∥x − y∥ q . Adding them together we get ⟨∇F(x) − ∇F(y), x − y⟩ ≥ σ∥x − y∥ q , which shows that F(x) is uniformly convex function with degree q and modulus σ > 0.

(ii) From Lemma [3](#page-4-6) (iii) and Definition [1,](#page-2-4) we know that f is p th-order smooth with parameter L<sup>p</sup> = β ρ p , ∀ p ∈ Z <sup>+</sup>, i.e., ∀ x, y ∈ Q ⊂ <sup>R</sup> d ,

$$\|\nabla^p f(\mathbf{x}) - \nabla^p f(\mathbf{y})\| \leq \beta \left(\frac{2}{\rho}\right)^p \|\mathbf{x} - \mathbf{y}\|.$$

Also, ∇<sup>p</sup>−<sup>1</sup>f is β ρ p−1 -Lipschitz, which implies that ∀ x ∈ R d , ∥∇<sup>p</sup>f(x)∥ ≤ β 2 ρ p−1 . Then we have ∀ x, y ∈ Q,

$$\begin{aligned} \|\nabla^p f(\mathbf{x}) - \nabla^p f(\mathbf{y})\| &= \|\nabla^p f(\mathbf{x}) - \nabla^p f(\mathbf{y})\|^\nu \|\nabla^p f(\mathbf{x}) - \nabla^p f(\mathbf{y})\|^{1-\nu} \\ &\leq \|\nabla^p f(\mathbf{x}) - \nabla^p f(\mathbf{y})\|^\nu (\|\nabla^p f(\mathbf{x})\| + \|\nabla^p f(\mathbf{y})\|)^{1-\nu} \\ &\leq \left(\frac{2}{\rho}\right)^{p\nu} \beta^\nu \|\mathbf{x} - \mathbf{y}\|^\nu \left(2\beta \left(\frac{2}{\rho}\right)^{p-1}\right)^{1-\nu} \\ &= \frac{2^p \beta}{\rho^{p+\nu-1}} \|\mathbf{x} - \mathbf{y}\|^\nu. \end{aligned}$$

By letting H = 2 <sup>p</sup>+1β <sup>ρ</sup><sup>p</sup>+ν−<sup>1</sup> , we can conclude that f is p th-order Holder smooth with parameter ¨ H 2 .

Furthermore, for dq(x), by definition, Q = {x : ∥x∥<sup>2</sup> ≤ D} for D ≤ <sup>H</sup> 2 <sup>1</sup>−νC <sup>1</sup> q−p−ν and C = σ(q − 1) × · · · × (q − p). As a result,

$$\|\nabla^{p+1} d_q(\mathbf{x})\| = \sigma(q-1) \times \cdots \times (q-p) \|\mathbf{x}\|^{q-p-1} \leq C \cdot D^{q-p-1}.$$

$$\|\nabla^p d_q(\mathbf{x}) - \nabla^p d_q(\mathbf{y})\| \leq C \cdot D^{q-p-1} \|\mathbf{x} - \mathbf{y}\|.$$

Given that ∥x − y∥ = ∥x − y∥ <sup>1</sup>−<sup>ν</sup>∥x − y∥ <sup>ν</sup> ≤ (∥x∥ + ∥y∥) <sup>1</sup>−<sup>ν</sup>∥x − y∥ <sup>ν</sup> ≤ (2D) <sup>1</sup>−<sup>ν</sup>∥x − y∥ ν , we have

$$\|\nabla^p d_q(\mathbf{x}) - \nabla^p d_q(\mathbf{y})\| \leq 2^{1-\nu} C \cdot D^{q-p-\nu} \|\mathbf{x} - \mathbf{y}\|^\nu \leq \frac{H}{2} \|\mathbf{x} - \mathbf{y}\|^\nu.$$

That is, dq(x) is p th-order Holder smooth with parameter ¨ H on domain Q. Since f is also p th-order Holder smooth with parameter ¨ H 2 , we conclude that F = f + d<sup>q</sup> is p th-order Holder smooth with ¨ parameter H on domain Q.

Lemma 5. *For* R(x) = β maxk∈[T] ξ<sup>k</sup> eα(k) , x + σ q ∥x∥ q *, we have*

$$R(\mathbf{x}) - \beta(T-1)\delta \leq F(\mathbf{x}) \leq R(\mathbf{x}) + \frac{5}{4}p\beta\rho\sqrt{d}.$$

*Proof.* Since F(x) is constructed with softmax smoothing, we are now able to characterize it with the properties in Lemma [3.](#page-4-6) F(x) can be upper bounded using the second inequality of Lemma [3](#page-4-6) (ii):

$$\begin{aligned} F(\mathbf{x}) &= \beta G_T(\mathbf{x}) + \frac{\sigma}{q} \|\mathbf{x}\|^q \\ &\leq \beta g_T(\mathbf{x}) + \frac{5}{4} p \beta \rho \sqrt{d} + \frac{\sigma}{q} \|\mathbf{x}\|^q \\ &= \beta \max_{k \in [T]} \{\xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x} \rangle - (k-1)\delta\} + \frac{5}{4} p \beta \rho \sqrt{d} + \frac{\sigma}{q} \|\mathbf{x}\|^q \\ &\leq \beta \max_{k \in [T]} \xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x} \rangle + \frac{5}{4} p \beta \rho \sqrt{d} + \frac{\sigma}{q} \|\mathbf{x}\|^q. \end{aligned}$$

F(x) can be lower bounded using the first inequality of Lemma [3](#page-4-6) (ii):

$$\begin{aligned} F(\mathbf{x}) &= \beta G_T(\mathbf{x}) + \frac{\sigma}{q} \|\mathbf{x}\|^q \\ &\geq \beta g_T(\mathbf{x}) + \frac{\sigma}{q} \|\mathbf{x}\|^q \\ &= \beta \max_{k \in [T]} \{ \xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x} \rangle - (k-1)\delta \} + \frac{\sigma}{q} \|\mathbf{x}\|^q \\ &\geq \beta \max_{k \in [T]} \xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x} \rangle - (T-1)\delta + \frac{\sigma}{q} \|\mathbf{x}\|^q. \end{aligned}$$

Lemma 7. F(x<sup>T</sup> ) − F(x ∗ ) ≥ −β(T − 1)δ − 4 pβρ√ d + q−1 q β q σT q 1 q−1

*.*

*Proof.*

$$\begin{aligned} F(\mathbf{x}^*) &= \min_{\mathbf{x}} F(\mathbf{x}) \\ &\leq \min_{\mathbf{x}} R(\mathbf{x}) + \frac{5}{4} p \beta \rho \sqrt{d} \\ &= \min_{\mathbf{x}} \left\{ \beta \max_{k \in [T]} \xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x} \rangle + \frac{\sigma}{q} \|\mathbf{x}\|^q \right\} + \frac{5}{4} p \beta \rho \sqrt{d}, \end{aligned}$$

Define γ = maxk∈[T] ξ<sup>k</sup> eα(k) , x  . Then by symmetry [\(Doikov,](#page-10-3) [2022\)](#page-10-3),

$$\|\mathbf{x}\|^q = T^{\frac{q}{2}} \gamma^q.$$

As a result,

$$\begin{aligned} \min_{\mathbf{x}} R(\mathbf{x}) &= \min_{\mathbf{x}} \left\{ \beta \max_{k \in [T]} \xi_k \langle \mathbf{e}_{\alpha(k)}, \mathbf{x} \rangle + \frac{\sigma}{q} \|\mathbf{x}\|^q \right\} \\ &= \min_{\gamma > 0} \{ -\beta \gamma + \frac{\sigma}{q} T^{\frac{q}{2}} \gamma^q \} \\ &= -\frac{q-1}{q} \left( \frac{\beta^q}{\sigma T^{\frac{q}{2}}} \right)^{\frac{1}{q-1}}. \end{aligned}$$

Therefore,

$$F(\mathbf{x}^*) \leq -\frac{q-1}{q} \left( \frac{\beta^q}{\sigma T^{\frac{q}{2}}} \right)^{\frac{1}{q-1}} + \frac{5}{4} p \beta \rho \sqrt{d}.$$

Furthermore, for some x<sup>T</sup> generated following some algorithm A along some trajectory, by definition,

$$\begin{aligned} g_T(\mathbf{x}_T) &\geq |\langle e_{\alpha(T)}, \mathbf{x}_T \rangle| - (T-1)\delta \\ &\geq -(T-1)\delta. \end{aligned}$$

Therefore,

$$\begin{aligned} F(\mathbf{x}_T) &= f(\mathbf{x}_T) + \frac{\sigma}{q} \|\mathbf{x}_T\|^q \\ &\geq f(\mathbf{x}_T) \\ &= \beta G_T(\mathbf{x}_T) \\ &\geq \beta g_T(\mathbf{x}_T) && \text{(Lemma 3 (ii))} \\ &\geq -\beta(T-1)\delta. \end{aligned}$$

Given the upper bound on F(x ∗ ), we have

$$F(\mathbf{x}_T) - F(\mathbf{x}^*) \geq -\beta(T-1)\delta - \frac{5}{4}p\beta\rho\sqrt{d} + \frac{q-1}{q} \left( \frac{\beta^q}{\sigma T^{\frac{q}{2}}} \right)^{\frac{1}{q-1}}.$$

# B PROOF FOR TECHNICAL LEMMAS IN SECTION [5](#page-6-3)

Lemma 8. x <sup>∗</sup> = arg min<sup>x</sup> f(x)*,* y = arg min<sup>x</sup> ˜f(x)*. (i)* ∀i ∈ [T˜]*,* ⟨v<sup>i</sup> , x ∗ ⟩ = y<sup>i</sup> *. (ii)* ∥x <sup>∗</sup>∥ = ∥y∥*.*

*Proof.* (i) By definition, <sup>f</sup> is a scaling and rotation of ˜f. Since <sup>v</sup>1, · · · , <sup>v</sup>T˜, we can write for <sup>V</sup> = [v1, · · · , <sup>v</sup>T˜], <sup>f</sup>(x) = <sup>H</sup> 2 <sup>p</sup>+ν+1(p+ν−1)! ˜f(V x). Therefore,

$$\begin{aligned} \mathbf{y} &= \arg \min_{\mathbf{x}} \tilde{f}(\mathbf{x}) \\ &= V \arg \min_{\mathbf{x}} \tilde{f}(V\mathbf{x}) \\ &= V \arg \min_{\mathbf{x}} f(\mathbf{x}) \\ &= V\mathbf{x}^*. \end{aligned}$$

(ii) This can be shown in the same way as [\(Arjevani et al.,](#page-9-0) [2019,](#page-9-0) Lemma 6).

Lemma 9. f(x) *is (i) uniformly convex with degree* q *and parameter* σ*. (ii)* p th*-order Holder smooth ¨ with degree* ν *and parameter* H*.*

*Proof.* (i) The proof is similar to that for Lemma [4](#page-4-2) (i).

(ii) Without the loss of generality, let the basis that defines f be the standard basis. ∀ i ∈ [T˜], denote ei the i th vector in the standard basis. Denote function g(x) = <sup>1</sup> p+ν |x| p+ν . g (p) (x), the p th-order derivative of g(x) is (p + ν − 1)!x ν if p is odd, (p + ν − 1)!|x| ν is even. Let d<sup>i</sup> = e<sup>i</sup> − ei+1, then

$$f(\mathbf{x}) = \frac{H}{2^{p+\nu+1}(p+\nu-1)!} \left( \frac{1}{p+\nu} \sum_{i=1}^{\tilde{T}} g(\langle \mathbf{d}_i, \mathbf{x} \rangle) - \gamma x_1 \right) + \frac{\sigma}{q} \|\mathbf{x}\|^q$$

Since q < p + ν, then q ≤ p. Therefore, ∀ x, y ∈ R d ,

$$\begin{aligned} \|\nabla^p f(\mathbf{x}) - \nabla^p f(\mathbf{y})\| &= \frac{H}{2^{p+\nu+1}(p+\nu-1)!} \left\| \sum_{i=1}^{\tilde{T}} \left[ g^{(p)}(\langle \mathbf{d}_i, \mathbf{x} \rangle) - g^{(p)}(\langle \mathbf{d}_i, \mathbf{y} \rangle) \right] [\mathbf{d}_i]^p \right\| \\ &\leq \frac{H(p+\nu-1)!}{2^{p+\nu+1}(p+\nu-1)!} \left\| \sum_{i=1}^{\tilde{T}} |\langle \mathbf{d}_i, \mathbf{x} - \mathbf{y} \rangle|^\nu [\mathbf{d}_i]^p \right\| \\ &\leq \frac{H}{2^{p+\nu+1}} \sqrt{2} \|\mathbf{x} - \mathbf{y}\|^\nu \left\| \sum_{i=1}^{\tilde{T}} [\mathbf{d}_i]^p \right\| \\ &\leq \frac{H}{2^{p+\nu+1}} \sqrt{2} \|\mathbf{x} - \mathbf{y}\|^\nu 2^p \\ &\leq H \|\mathbf{x} - \mathbf{y}\|^\nu. \end{aligned}$$

Lemma 15. *For* y = arg min<sup>x</sup> ˜f(x)*,*

$$(i) \ y_1 \geq y_2 \geq \cdots \geq y_{\tilde{T}} \geq 0.$$

$$(ii) y_{t+1} = y_t - \left( \gamma - \tilde{\sigma} \|\mathbf{y}\|^{q-2} \sum_{j=1}^t y_j \right)^{\frac{1}{p+\nu-1}}$$

$$(iii) \quad \sum_{i=1}^{\tilde{T}} y_i = \frac{\gamma}{\tilde{\sigma} \|\mathbf{y}\|^{q-2}}.$$

*Proof.* (i) The proof is similar to [\(Arjevani et al.,](#page-9-0) [2019,](#page-9-0) Lemma 1), relying on the fact that ˜f is strictly convex, which holds true for our higher-order construction as well, since the function is uniformly convex.

(ii)

$$\nabla \tilde{f}(y) = \begin{bmatrix} |y_1 - y_2|^{p+\nu-2} (y_1 - y_2) - \gamma + \tilde{\sigma} \|\mathbf{y}\|^{q-2} y_1 \\ |y_2 - y_1|^{p+\nu-2} (y_2 - y_1) + |y_2 - y_3|^{p+\nu-2} (y_2 - y_3) + \tilde{\sigma} \|\mathbf{y}\|^{q-2} y_2 \\ \vdots \\ |y_{\tilde{T}-1} - y_{\tilde{T}-2}|^{p+\nu-2} (y_{\tilde{T}-1} - y_{\tilde{T}-2}) + |y_{\tilde{T}-1} - y_{\tilde{T}}|^{p+\nu-2} (y_{\tilde{T}-1} - y_{\tilde{T}}) + \tilde{\sigma} \|\mathbf{y}\|^{q-2} y_{\tilde{T}-1} \\ |y_{\tilde{T}} - y_{\tilde{T}-1}|^{p+\nu-2} (y_{\tilde{T}} - y_{\tilde{T}-1}) + \tilde{\sigma} \|\mathbf{y}\|^{q-2} y_{\tilde{T}} \end{bmatrix}$$

Given that <sup>y</sup><sup>1</sup> <sup>≥</sup> <sup>y</sup><sup>2</sup> ≥ · · · ≥ <sup>y</sup>T˜ <sup>≥</sup> <sup>0</sup>, we have <sup>∀</sup><sup>i</sup> <sup>∈</sup> [T˜ <sup>−</sup> 1], |y<sup>i</sup> <sup>−</sup> <sup>y</sup>i+1| <sup>=</sup> <sup>y</sup><sup>i</sup> <sup>−</sup> <sup>y</sup>i+1. Therefore, with ∇ ˜f(y) = 0, we have

$$(y_1 - y_2)^{p+\nu-1} = \gamma - \tilde{\sigma}\|\mathbf{y}\|^{q-2}y_1, \quad (2)$$

$$(y_i - y_2)^{p+\nu-1} = \gamma - \sigma \|\mathbf{y}\|^{q-2} y_1, \quad (2)$$

$$(y_{i-1} - y_i)^{p+\nu-1} = (y_i - y_{i+1})^{p+\nu-1} + \tilde{\sigma} \|\mathbf{y}\|^{q-2} y_i, \quad 2 \leq i \leq \tilde{T} - 1, \quad (3)$$

$$(y_{\tilde{T}-1} - y_{\tilde{T}})^{p+\nu-1} = \tilde{\sigma} \|\mathbf{y}\|^{q-2} y_{\tilde{T}}. \quad (4)$$

Summing Eq. [\(2\)](#page-20-1) and [\(3\)](#page-20-2), we have

$$(y_i - y_{i+1})^{p+\nu-1} = \gamma - \tilde{\sigma} \|\mathbf{y}\|^{q-2} \sum_{j=1}^i y_j,$$

which completes the proof.

(iii) We know from part (ii) that

$$(y_{\tilde{T}-1} - y_{\tilde{T}})^{p+\nu-1} = \gamma - \tilde{\sigma} \|\mathbf{y}\|^{q-2} \sum_{j=1}^{\tilde{T}-1} y_j.$$

Lemma 11. ∥y∥ ≤ <sup>2</sup> 2 <sup>3</sup>q−<sup>2</sup> γ 3(p+ν)−2 (p+ν−1)(3q−2) σ˜ 3q−2 *.*

**Lemma 11.** 
$$\|y\| \leq \frac{2}{2q-2} \gamma \frac{3(p+\nu)-2}{(p+\nu-1)(3q-2)} \cdot \frac{3}{\tilde{\sigma}^{\frac{3}{3q-2}}}$$
.

*Proof.* By Lemma [15](#page-20-4) (iii) and Lemma [10](#page-7-0) (ii),

$$\begin{aligned} \|\mathbf{y}\|_2^2 &\leq \|\mathbf{y}\|_1 \|\mathbf{y}\|_\infty \\ &= \max_{i \in [\bar{T}]} |y_i| \times \sum_{i=1}^d y_i \\ &= y_1 \times \sum_{i=1}^{\bar{T}} y_i \\ &\leq \left( \gamma \frac{1}{p^{+\nu-1}} + \sqrt{\frac{2\gamma \frac{p^{+\nu}}{p^{+\nu-1}}}{\tilde{\sigma} \|\mathbf{y}\|^{q-2}}} \right) \times \frac{\gamma}{\tilde{\sigma} \|\mathbf{y}\|^{q-2}} \\ &= \left( 1 + \sqrt{\frac{2\gamma \frac{p^{+\nu-2}}{p^{+\nu-1}}}{\tilde{\sigma} \|\mathbf{y}\|^{q-2}}} \right) \frac{\gamma \frac{p^{+\nu}}{p^{+\nu-1}}}{\tilde{\sigma} \|\mathbf{y}\|^{q-2}} \end{aligned}$$

Let γ ≥ 3˜σ∥y∥ q−2 <sup>p</sup>+ν−<sup>1</sup> p+ν−2 , then we have <sup>γ</sup> p+ν−2 p+ν−1 3˜σ∥y∥<sup>q</sup>−<sup>2</sup> <sup>≥</sup> <sup>1</sup> and moreover r γ p+ν−2 p+ν−1 3˜σ∥y∥<sup>q</sup>−<sup>2</sup> ≥ 1, so that we can merge the terms as follows:

$$\begin{aligned} \|\mathbf{y}\|^2 &\leq \left( \sqrt{\frac{\gamma^{\frac{p+\nu-2}{p}}}{3\tilde{\sigma}\|\mathbf{y}\|^{q-2}}} + \sqrt{\frac{2\gamma^{\frac{p+\nu-2}{p}}}{\tilde{\sigma}\|\mathbf{y}\|^{q-2}}} \right) \frac{\gamma^{\frac{p+\nu}{p+\nu-1}}}{\tilde{\sigma}\|\mathbf{y}\|^{q-2}} \\ &= \left( \sqrt{\frac{1}{3} + \sqrt{2}} \right) \frac{\gamma^{\frac{p+\nu-2}{2(p+\nu-1)}} + \frac{p+\nu}{p+\nu-1}}{\tilde{\sigma}^{\frac{3}{2}}\|\mathbf{y}\|^{\frac{3(q-2)}{2}}} \\ &\leq \frac{2\gamma^{\frac{3(p+\nu)-2}{2(p+\nu-1)}}}{\tilde{\sigma}^{\frac{3}{2}}\|\mathbf{y}\|^{\frac{3(q-2)}{2}}}. \end{aligned}$$

We can solve for <sup>∥</sup>y∥ ≤ 2γ 3(p+ν)−2 2(p+ν−1) σ˜ 2 ! 2 3q−2 = 2 2 <sup>3</sup>q−<sup>2</sup> γ 3(p+ν)−2 (p+ν−1)(3q−2) σ˜ 3q−2 .

Lemma 10. *For* y = arg min<sup>x</sup> ˜f(x)*,*

$$(i) \ \forall t \in [\tilde{T}], y_t \geq y_1 - (t-1)\gamma^{\frac{1}{p+\nu-1}}.$$

$$(i) \quad \forall t \in [T], y_t \geq y_1 - (t-1)\gamma^{\frac{1}{p+\nu-1}}.$$

$$(ii) \quad \text{For } \tilde{T} = \left\lfloor \frac{y_1}{\gamma^{\frac{1}{p+\nu-1}}} + 1 \right\rfloor, y_1 \leq \gamma^{\frac{1}{p+\nu-1}} + \sqrt{\frac{2\gamma^{\frac{p+\nu}{p+\nu-1}}}{\tilde{\sigma}\|y\|^{q-2}}}.$$

$$(iii) \text{ For } \gamma \geq \tilde{\sigma}^{\frac{p+\nu-1}{p+\nu-2}} \|\mathbf{y}\|^{\frac{(p+\nu-1)(q-2)}{p+\nu-2}}, \forall t \in [\tilde{T}], y_t \geq \frac{\gamma^{\frac{p+\nu}{2(p+\nu-1)}}}{2^{p+\nu+1}\tilde{\sigma}^{\frac{1}{2}} \|\mathbf{y}\|^{\frac{q-2}{q-2}}} + \left(\frac{1}{2} - i\right) \gamma^{\frac{1}{p+\nu-1}}.$$

*Proof.* (i) By Lemma [15](#page-20-4) (ii), ∀ i ∈ [T˜]

$$\begin{aligned} y_i &= y_{i-1} - \left( \gamma - \tilde{\sigma} \|\mathbf{y}\|^{q-2} \sum_{j=1}^{i-1} y_j \right)^{\frac{1}{p+\nu-1}} \\ &\geq y_{i-1} - \gamma^{\frac{1}{p+\nu-1}} \\ &\geq y_1 - (i-1)\gamma^{\frac{1}{p+\nu-1}}, \end{aligned}$$

in which the first inequality follows from Lemma [15](#page-20-4) (i) that ∀i ∈ [T˜], y<sup>i</sup> ≥ 0, and the second inequality follows from applying the first inequality recursively.

(ii) It follows from part (i) that

$$\sum_{i=1}^{\tilde{T}} y_i \geq \sum_{i=1}^{\tilde{T}} \max \left\{ 0, y_1 - (i-1)\gamma^{\frac{1}{p+\nu-1}} \right\}.$$

For T˜ = y<sup>1</sup> γ p+ν−1 + 1 ≤ y<sup>1</sup> γ 1 p+ν−1 + 1, we always have y<sup>1</sup> − (T˜ − 1)γ 1 <sup>p</sup>+ν−<sup>1</sup> ≥ 0. Consequently, ∀i ∈ [T˜], y<sup>1</sup> − (i − 1)γ <sup>p</sup>+ν−<sup>1</sup> ≥ 0. Therefore,

$$\begin{aligned} \sum_{i=1}^{\bar{T}} y_i &\geq \sum_{i=1}^{\bar{T}} y_1 - (i-1)\gamma^{\frac{1}{p+\nu-1}} \\ &= \sum_{i=1}^{} y_1 - (i-1)\gamma^{\frac{1}{p+\nu-1}} \\ &= \left\lfloor \frac{y_1}{\gamma^{\frac{1}{p+\nu-1}}} + 1 \right\rfloor \cdot y_1 - \gamma^{\frac{1}{p+\nu-1}} \cdot \frac{\left\lfloor \frac{y_1}{\gamma^{\frac{1}{p+\nu-1}}} + 1 \right\rfloor \left( \left\lfloor \frac{y_1}{\gamma^{\frac{1}{p+\nu-1}}} + 1 \right\rfloor - 1 \right)}{2} \\ &\geq \frac{y_1}{\gamma^{\frac{1}{p+\nu-1}}} \cdot y_1 - \gamma^{\frac{1}{p+\nu-1}} \cdot \frac{\left( \frac{y_1}{\gamma^{\frac{1}{p+\nu-1}}} + 1 \right) \left( \frac{y_1}{\gamma^{\frac{1}{p+\nu-1}}} + 1 - 1 \right)}{2} \\ &= \frac{y_1^2}{\gamma^{\frac{1}{p+\nu-1}}} - \frac{y_1^2}{2\gamma^{\frac{1}{p+\nu-1}}} - \frac{y_1}{2} \\ &= \frac{y_1}{2} \left( \frac{y_1}{\gamma^{\frac{1}{p+\nu-1}}} - 1 \right). \end{aligned}$$

Combining with Lemma [<sup>15</sup>](#page-20-4) (iii) that P<sup>T</sup>˜ <sup>i</sup>=1 y<sup>i</sup> = γ <sup>σ</sup>˜∥y∥<sup>q</sup>−<sup>2</sup> , we have

$$\frac{\gamma}{\tilde{\sigma}\|\mathbf{y}\|^{q-2}} = \sum_{i=1}^{\tilde{T}} y_i \geq \frac{y_1}{2} \left( \frac{y_1}{\gamma^{\frac{1}{q+\nu-1}}} - 1 \right).$$

Equivalently,

$$y_1^2 - \gamma \frac{1}{p+\nu-1} y_1 - \frac{2\gamma}{\tilde{\sigma} \|\mathbf{y}\|^{q-2}} \leq 0.$$

By the quadratic formula, we have

$$\begin{aligned} y_1 &\leq \frac{\gamma^{\frac{1}{p+\nu-1}} + \sqrt{\gamma^{\frac{2}{p+\nu-1}} + \frac{8\gamma^{\frac{p+\nu}{p+\nu-1}}}{\tilde{\sigma}\|\mathbf{y}\|^{q-2}}}{2} \\ &\leq \frac{\gamma^{\frac{1}{p+\nu-1}} + \sqrt{\gamma^{\frac{2}{p+\nu-1}} + \sqrt{\frac{8\gamma^{\frac{p+\nu}{p+\nu-1}}}{\tilde{\sigma}\|\mathbf{y}\|^{q-2}}}}{2} \\ &= \gamma^{\frac{1}{p+\nu-1}} + \sqrt{\frac{2\gamma^{\frac{p+\nu}{p+\nu-1}}}{\tilde{\sigma}\|\mathbf{y}\|^{q-2}}} \end{aligned}$$

(iii) Since P<sup>T</sup>˜ <sup>i</sup>=1 y<sup>i</sup> = γ <sup>σ</sup>˜∥y∥<sup>q</sup>−<sup>2</sup> , <sup>∃</sup> <sup>t</sup><sup>0</sup> <sup>∈</sup> [T˜] such that <sup>P</sup><sup>t</sup><sup>0</sup> <sup>i</sup>=1 y<sup>i</sup> > (1 − 1 2 <sup>p</sup>+ν−<sup>1</sup> ) γ <sup>σ</sup>˜∥y∥<sup>q</sup>−<sup>2</sup> and ∀t < t0, P<sup>t</sup> <sup>i</sup>=1 y<sup>i</sup> ≤ (1 − 2 <sup>p</sup>+ν−<sup>1</sup> ) γ <sup>σ</sup>˜∥y∥<sup>q</sup>−<sup>2</sup> . Then ∀ i < t0, we can merge the terms in Lemma [15](#page-20-4) (ii) as

follows:

$$\begin{aligned} y_{i+1} &= y_i - \left( \gamma - \tilde{\sigma} \|\mathbf{y}\|^{q-2} \sum_{j=1}^i y_j \right)^{\frac{1}{p+\nu-1}} \\ &\leq y_i - \left( \gamma - \left( 1 - \frac{1}{2^{p+\nu-1}} \right) \gamma \right)^{\frac{1}{p+\nu-1}} \\ &= y_i - \frac{\gamma^{\frac{1}{p+\nu-1}}}{2}. \end{aligned}$$

Applying this relation recursively, we have

$$y_{t0} \leq y_{t0-1} - \frac{\gamma^{\frac{1}{p+\nu-1}}}{2} \leq \dots \leq y_1 - (t_0 - 1) \frac{\gamma^{\frac{1}{p+\nu-1}}}{2}.$$

Given that y<sup>t</sup><sup>0</sup> ≥ <sup>0</sup>, this yields y<sup>1</sup> ≥ (t<sup>0</sup> − 1) <sup>γ</sup> 1 p+ν−1 .

Now we characterize <sup>t</sup>0. By definition, we have P<sup>t</sup><sup>0</sup> <sup>i</sup>=1 y<sup>i</sup> > (1 − 2 <sup>p</sup>+ν−<sup>1</sup> ) γ <sup>σ</sup>˜∥y∥<sup>q</sup>−<sup>2</sup> . In the meantime,

$$\begin{aligned} \sum_{i=1}^{t_0} y_i &\leq \sum_{i=1}^{t_0} y_1 \\ &= t_0 y_1 \\ &\leq t_0 \left( \gamma^{\frac{1}{p+\nu-1}} + \sqrt{\frac{2\gamma^{\frac{p+\nu}{p+\nu-1}}}{\tilde{\sigma} \|\mathbf{y}\|^{q-2}}} \right), \end{aligned}$$

where the first inequality follows from Lemma [15](#page-20-4) (i) and the second from part (ii). Together, we have (1 − 2 <sup>p</sup>+ν−<sup>1</sup> ) γ <sup>σ</sup>˜∥y∥<sup>q</sup>−<sup>2</sup> < t<sup>0</sup> γ 1 <sup>p</sup>+ν−<sup>1</sup> + r 2γ p+ν p+ν−1 σ˜∥y∥<sup>q</sup>−<sup>2</sup> ! , from which we solve for

$$t_0 > \frac{(2^{p+\nu-1} - 1)\gamma^{\frac{p+\nu-2}{p+\nu-1}}}{2^{p+\nu-1} \left( \tilde{\sigma} \|\mathbf{y}\|^{q-2} + \sqrt{2\tilde{\sigma}\gamma^{\frac{p+\nu-2}{p+\nu-1}} \|\mathbf{y}\|^{q-2}} \right)}$$

Plugging this characterization of t<sup>0</sup> back in,

$$\begin{aligned} y_1 &\geq (t_0 - 1) \frac{\gamma_{\frac{p+\nu-1}{2}}}{2} \\ &> \left( \frac{(2^{p+\nu-1} - 1)\gamma_{\frac{p+\nu-2}{p+\nu-1}}}{2^{p+\nu-1} \left( \tilde{\sigma} \|\mathbf{y}\|^{q-2} + \sqrt{2\tilde{\sigma}\gamma_{\frac{p+\nu-2}{p+\nu-1}} \|\mathbf{y}\|^{q-2}} \right)} - 1 \right) \frac{\gamma_{\frac{1}{p+\nu-1}}}{2} \\ &= \frac{(2^{p+\nu-1} - 1)\gamma}{2^{p+\nu+1} \left( \tilde{\sigma} \|\mathbf{y}\|^{q-2} + \sqrt{2\tilde{\sigma}\gamma_{\frac{p+\nu-2}{p+\nu-1}} \|\mathbf{y}\|^{q-2}} \right)} - \frac{\gamma_{\frac{1}{p+\nu-1}}}{2}. \end{aligned}$$

Finally, plugging this into the result from part (i), ∀i ∈ [T˜],

$$\begin{aligned} y_i &\geq y_1 - (i-1)\gamma^{\frac{1}{p+\nu-1}} \\ &\geq \frac{(2^{p+\nu-1} - 1)\gamma}{2^{p+\nu+1} \left( \tilde{\sigma} \|\mathbf{y}\|^{q-2} + \sqrt{2\tilde{\sigma}\gamma^{\frac{p+\nu-2}{p+\nu-1}} \|\mathbf{y}\|^{q-2}} \right)} - \frac{\gamma^{\frac{1}{p+\nu-1}}}{2} - (i-1)\gamma^{\frac{1}{p+\nu-1}} \\ &= \frac{(2^{p+\nu-1} - 1)\gamma}{2^{p+\nu+1} \left( \tilde{\sigma} \|\mathbf{y}\|^{q-2} + \sqrt{2\tilde{\sigma}\gamma^{\frac{p+\nu-2}{p+\nu-1}} \|\mathbf{y}\|^{q-2}} \right)} + \left( \frac{1}{2} - i \right) \gamma^{\frac{1}{p+\nu-1}} \end{aligned}$$

By letting γ ≥ σ˜ p+ν−1 <sup>p</sup>+ν−<sup>2</sup> ∥y∥ (p+ν−1)(q−2) <sup>p</sup>+ν−<sup>2</sup> as stated in the condition, we have γ p+ν−2 <sup>p</sup>+ν−<sup>1</sup> ≥ σ˜∥y∥ q−2 and are able to merge the terms as follows:

$$\begin{aligned} y_i &\geq \frac{(2^{p+\nu-1} - 1)\gamma}{2^{p+\nu+1} \left( (\tilde{\sigma}\|\mathbf{y}\|^{q-2})^{\frac{1}{2}} \cdot (\tilde{\sigma}\|\mathbf{y}\|^{q-2})^{\frac{1}{2}} + \sqrt{2} (\tilde{\sigma}\|\mathbf{y}\|^{q-2})^{\frac{1}{2}} \cdot \left( \gamma^{\frac{p+\nu-2}{p+\nu-1}} \right)^{\frac{1}{2}} \right)} + \left( \frac{1}{2} - i \right) \gamma^{\frac{1}{p+\nu-1}} \\ &\geq \frac{(2^{p+\nu-1} - 1)\gamma}{2^{p+\nu+1} \left( (\tilde{\sigma}\|\mathbf{y}\|^{q-2})^{\frac{1}{2}} \cdot \left( \gamma^{\frac{p+\nu-2}{p+\nu-1}} \right)^{\frac{1}{2}} + \sqrt{2} (\tilde{\sigma}\|\mathbf{y}\|^{q-2})^{\frac{1}{2}} \cdot \left( \gamma^{\frac{p+\nu-2}{p+\nu-1}} \right)^{\frac{1}{2}} \right)} + \left( \frac{1}{2} - i \right) \gamma^{\frac{1}{p+\nu-1}} \\ &\geq \frac{(2^{p+\nu-1} - 1)\gamma}{2^{p+\nu+1} \left( (1 + \sqrt{2}) (\tilde{\sigma}\|\mathbf{y}\|^{q-2})^{\frac{1}{2}} \cdot \left( \gamma^{\frac{p+\nu-2}{p+\nu-1}} \right)^{\frac{1}{2}} \right)} + \left( \frac{1}{2} - i \right) \gamma^{\frac{1}{p+\nu-1}} \\ &\geq \frac{\gamma^{\frac{p+\nu}{2(p+\nu-1)}}}{2^{p+\nu+1} \tilde{\sigma}^{\frac{1}{2}} \|\mathbf{y}\|^{\frac{q-2}{2}}} + \left( \frac{1}{2} - i \right) \gamma^{\frac{1}{p+\nu-1}} \end{aligned}$$

Lemma 12. *For* y = arg min<sup>x</sup> ˜f(x)*, let* <sup>t</sup><sup>1</sup> ∈ [T˜] *be such that* <sup>y</sup><sup>t</sup><sup>1</sup> <sup>&</sup>gt; p+ν−1 σ˜ 1 <sup>p</sup>+ν−<sup>2</sup> ∥y∥ q−2 <sup>p</sup>+ν−<sup>2</sup> *and* y<sup>t</sup>1+1 ≤ p+ν−1 σ˜ 1 <sup>p</sup>+ν−<sup>2</sup> ∥y∥ q−2 <sup>p</sup>+ν−<sup>2</sup> *. Then*

- (i) 
  $$\forall i \in [\tilde{T}], y_i = y_{i+1} + \left( \tilde{\sigma} \|\mathbf{y}\|^{q-2} \sum_{j=i+1}^{\tilde{T}} y_j \right)^{\frac{1}{p+\nu-1}}$$
   and  $y_{i+1} \leq \frac{1}{\tilde{\sigma} \|\mathbf{y}\|^{q-2}} y_i^p$
- (ii)  $\forall i \geq t_1, \left( \frac{1}{c_{p,\nu}} \right)^{p+\nu-1} \frac{1}{\tilde{\sigma} \|\mathbf{y}\|^{q-2}} y_i^{p+\nu-1} \leq y_{i+1}$  where  $c_{p,\nu}$  is a constant depending on  $p, \nu$ ,
- (iii)  $\forall i \leq \tilde{T} - t_1, y_{t_1+i} \geq \left( \frac{1}{c_{p,\nu}} \right)^{\frac{(p+\nu-1)((p+\nu-1)^i-1)}{p+\nu-2}} (\tilde{\sigma} \|\mathbf{y}\|^{q-2})^{\frac{1}{p+\nu-2}} (p + \nu - 1)^{-(p+\nu-1)^i}$ .

*Proof.* (i) Starting from Lemma [15](#page-20-4) (ii), ∀i ∈ [T˜],

$$\begin{aligned} y_i &= y_{i+1} + \left( \gamma - \tilde{\sigma} \|\mathbf{y}\|^{q-2} \sum_{j=1}^i y_j \right)^{\frac{1}{p+\nu-1}} \\ &= y_{i+1} + \left( \gamma - \tilde{\sigma} \|\mathbf{y}\|^{q-2} \sum_{j=1}^{\tilde{T}} y_j + \tilde{\sigma} \|\mathbf{y}\|^{q-2} \sum_{j=i+1}^{\tilde{T}} y_j \right)^{\frac{1}{p+\nu-1}} \\ &= y_{i+1} + \left( \gamma - \tilde{\sigma} \|\mathbf{y}\|^{q-2} \cdot \frac{\gamma}{\tilde{\sigma} \|\mathbf{y}\|^{q-2}} + \tilde{\sigma} \|\mathbf{y}\|^{q-2} \sum_{j=i+1}^{\tilde{T}} y_j \right)^{\frac{1}{p+\nu-1}} \quad (\text{Lemma 15 (iii)}) \\ &= y_{i+1} + \left( \tilde{\sigma} \|\mathbf{y}\|^{q-2} \sum_{j=i+1}^{\tilde{T}} y_j \right)^{\frac{1}{p+\nu-1}}. \end{aligned}$$

Since xi+1 ≥ 0, we have

$$\begin{aligned} y_i &\geq \left( \tilde{\sigma} \|\mathbf{y}\|^{q-2} \sum_{j=i+1}^{\tilde{T}} y_j \right)^{\frac{1}{p+\nu-1}} \\ &\geq (\tilde{\sigma} \|\mathbf{y}\|^{q-2} y_{i+1})^{\frac{1}{p+\nu-1}}, \end{aligned}$$

equivalently,

$$y_{i+1} \leq \frac{1}{\tilde{\sigma} \|\mathbf{y}\|^{q-2}} y_i^{p+\nu-1}.$$

(ii)

$$\begin{aligned} \sum_{j=i+1}^{\tilde{T}} y_j &= y_{i+1} + y_{i+2} + \cdots + y_{\tilde{T}} \\ &\leq y_{i+1} + \frac{1}{\tilde{\sigma} \|\mathbf{y}\|^{q-2}} y_{i+1}^{p+\nu-1} + \frac{1}{(\tilde{\sigma} \|\mathbf{y}\|^{q-2})^{(p+\nu-1)+1}} y_{i+1}^{(p+\nu-1)^2} + \cdots \\ &\quad + \frac{1}{(\tilde{\sigma} \|\mathbf{y}\|^{q-2})^{\sum_{j=0}^{\tilde{T}-i-2} (p+\nu-1)^j}} y_{j+1}^{(p+\nu-1)^{\tilde{T}-i-1}} \\ &= y_{i+1} \sum_{j=0}^{\tilde{T}-i-1} \left( \frac{y_{i+1}}{\tilde{\sigma} \frac{1}{p+\nu-2} \|\mathbf{y}\|^{\frac{q-2}{p+\nu-2}}} \right)^{(p+\nu-1)^j-1}. \end{aligned}$$

Given that i ≥ t1, then i + 1 ≥ t<sup>1</sup> + 1 and by Lemma [15](#page-20-4) (i) and part (ii) of this lemma, yi+1 ≤ y<sup>t</sup>1+1 ≤ p+ν−1 σ˜ 1 <sup>p</sup>+ν−<sup>2</sup> ∥y∥ q−2 <sup>p</sup>+ν−<sup>2</sup> . Therefore,

$$\begin{aligned} \sum_{j=i+1}^{\tilde{T}} y_j &\leq y_{i+1} \sum_{j=0}^{\tilde{T}-i-1} \left( \frac{\frac{1}{p+\nu-1} \tilde{\sigma} \frac{1}{p+\nu-2} \|\mathbf{y}\|_{\frac{q-2}{p+\nu-2}}^2}{\tilde{\sigma} \frac{1}{p+\nu-2} \|\mathbf{y}\|_{\frac{q-2}{p+\nu-2}}^2} \right)^{(p+\nu-1)^j} \\ &= (p+\nu-1) y_{i+1} \sum_{j=0}^{\tilde{T}-i-1} \frac{1}{(p+\nu-1)^{(p+\nu-1)^j}} \\ &\leq \frac{p+\nu-1}{(p+\nu-2)^2} y_{i+1} \end{aligned}$$

With this, we go back to part (i), for yi+1 ≤ 1 p+ν−1 σ˜ 1 <sup>p</sup>+ν−<sup>2</sup> ∥y∥ q−2 <sup>p</sup>+ν−<sup>2</sup> ,

$$\begin{aligned} y_i &= y_{i+1} + \left( \tilde{\sigma} \|\mathbf{y}\|^{q-2} \sum_{j=i+1}^{\tilde{T}} y_j \right)^{\frac{1}{p+\nu-1}} \\ &\leq y_{i+1} + \left( \frac{p+\nu-1}{(p+\nu-2)^2} \tilde{\sigma} \|\mathbf{y}\|^{q-2} y_{i+1} \right)^{\frac{1}{p+\nu-1}} \\ &= y_{i+1}^{\frac{1}{p+\nu-1}} y_{i+1}^{\frac{p+\nu-2}{p+\nu-1}} + \left( \frac{p+\nu-1}{(p+\nu-2)^2} \tilde{\sigma} \|\mathbf{y}\|^{q-2} y_{i+1} \right)^{\frac{1}{p+\nu-1}} \\ &\leq y_{i+1}^{\frac{1}{p+\nu-1}} \left( \frac{1}{p+\nu-1} \tilde{\sigma}^{\frac{1}{p+\nu-2}} \|\mathbf{y}\|^{\frac{q-2}{p+\nu-2}} \right)^{\frac{p+\nu-2}{p+\nu-1}} + \left( \frac{p+\nu-1}{(p+\nu-2)^2} \tilde{\sigma} \|\mathbf{y}\|^{q-2} y_{i+1} \right)^{\frac{1}{p+\nu-1}} \\ &= \left( ((p+\nu-1)^{\frac{2-p-\nu}{p+\nu-1}} + \left( \frac{p+\nu-1}{(p+\nu-2)^2} \right)^{\frac{1}{p+\nu-1}} \right) (\tilde{\sigma} \|\mathbf{y}\|^{q-2} y_{i+1})^{\frac{1}{p+\nu-1}} \\ &= c_{p,\nu} (\tilde{\sigma} \|\mathbf{y}\|^{q-2} y_{i+1})^{\frac{1}{p+\nu-1}} \end{aligned}$$

$$\text{for } c_{p,\nu} = ((p+\nu-1)^{\frac{2-p-\nu}{p+\nu-1}} + \left(\frac{p+\nu-1}{(p+\nu-2)^2}\right)^{\frac{1}{p+\nu-1}}). \text{ Therefore, } y_{i+1} \geq \left(\frac{1}{c_{p,\nu}}\right)^{p+\nu-1} \frac{1}{\tilde{\sigma}\|\mathbf{y}\|^{q-2}} y_i^{p+\nu-1},$$

(iii) ∀i ≤ T˜ − t1, t<sup>1</sup> + i ≥ t1, therefore, applying part (ii) recursively yields

$$\begin{aligned} y_{t_1+i} &\geq \left( \frac{1}{c_{p,\nu}} \right)^{p+\nu-1} \frac{(y_{t_1+i-1})^{p+\nu-1}}{\tilde{\sigma} \|\mathbf{y}\|^{q-2}} \\ &\geq \left( \frac{1}{c_{p,\nu}} \right)^{(p+\nu-1)^2+(p+\nu-1)} \frac{(y_{t_1+i-2})^{(p+\nu-1)^2}}{(\tilde{\sigma} \|\mathbf{y}\|^{q-2})^{p+\nu}} \\ &\geq \dots \\ &\geq \left( \frac{1}{c_{p,\nu}} \right)^{\frac{(p+\nu-1)((p+\nu-1)^i-1)}{p+\nu-2}} \frac{y_{t_1}^{(p+\nu-1)^i}}{(\tilde{\sigma} \|\mathbf{y}\|^{q-2})^{\frac{(p+\nu-1)^i-1}{p+\nu-2}}} \end{aligned}$$

By the definition of t1, we know that y<sup>t</sup><sup>1</sup> > 1 p+ν−1 σ˜ <sup>p</sup>+ν−<sup>2</sup> ∥y∥ q−2 <sup>p</sup>+ν−<sup>2</sup> . Thus

$$\begin{aligned} y_{t+1} &\geq \left( \frac{1}{c_{p,\nu}} \right)^{\frac{(p+\nu-1)((p+\nu-1)^{i-1})}{(p+\nu-1)-1}} \frac{\left( \frac{1}{p+\nu-1} \tilde{\sigma} \frac{1}{p+\nu-2} \|\mathbf{y}\|^{\frac{q-2}{p+\nu-2}} \right)^{(p+\nu-1)^i}}{(\tilde{\sigma} \|\mathbf{y}\|^{q-2})^{\frac{(p+\nu-1)^{i-1}}{p+\nu-2}}} \\ &= \left( \frac{1}{c_{p,\nu}} \right)^{\frac{(p+\nu-1)((p+\nu-1)^{i-1})}{p+\nu-2}} (\tilde{\sigma} \|\mathbf{y}\|^{q-2})^{\frac{1}{p+\nu-2}} (p+\nu-1)^{-(p+\nu-1)^i} \end{aligned}$$