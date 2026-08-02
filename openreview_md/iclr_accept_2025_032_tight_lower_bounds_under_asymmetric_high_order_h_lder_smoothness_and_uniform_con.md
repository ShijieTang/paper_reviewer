# Tight Lower Bounds Under Asymmetric High- Order Holder ¨ Smoothness And Uniform Con- Vexity

Cedar Site Bai Department of Computer Science Purdue University West Lafayette, IN, USA bai123@purdue.edu Brian Bullins Department of Computer Science Purdue University West Lafayette, IN, USA bbullins@purdue.edu

## Abstract

In this paper, we provide tight lower bounds for the oracle complexity of minimizing high-order Holder smooth and uniformly convex functions. Specifi- ¨ cally, for a function whose p th-order derivatives are Holder continuous with ¨
degree ν and parameter H, and that is uniformly convex with degree q and parameter σ, we focus on two asymmetric cases: (1) *q > p* + ν, and (2) *q < p* + ν. Given up to p th-order oracle access, we establish worst-case oracle complexities of Ω
 H
σ 2 3(p+ν)−2σϵ 2(q−p−ν)
q(3(p+ν)−2) 
in the first case with an ℓ∞-ball-truncated-Gaussian smoothed hard function and Ω
 Hσ 2 3(p+ν)−2 + log log σ p+ν Hq 1 p+ν−q 1 ϵ in the second case, for reaching an ϵ-approximate solution in terms of the optimality gap. Our analysis generalizes previous lower bounds for functions under first- and second-order smoothness as well as those for uniformly convex functions, and furthermore our results match the corresponding upper bounds in this general setting.

## 1 Introduction

With the advancement in computational power, high-order optimization methods (p th-order with p ≥
2) are gaining more attention for their merit of faster convergence and higher precision. Consequently, uniformly convex problems (with degree q) have become a recent focus, particularly the subproblems of some high-order optimization methods. The subproblem of the cubic-regularized Newton (p = 2, q = 3) (Nesterov & Polyak, 2006) is an example, as are methods of even higher orders (p ≥ 3, q ≥ 4) (Zhu & Cartis, 2022). Although these problems are high-order smooth by definition, a lower-order algorithm may be employed to obtain an approximate solution. For instance, solving the subproblem of cubic-regularized (i.e., q = 3) Newton with gradient descent (accessing first-order oracle, i.e., p = 1), or, more generally, approximately solving the subproblem of (q − 1)th-order Taylor descent (Bubeck et al., 2019)
(which typically contains a regularization term to the power of q) with lower-order oracle access, introduces an asymmetry between the algorithm's oracle access order and the degree of uniform convexity (*q > p* + 1).

Conversely, a lower-degree regularization can be paired with a higher-order smooth function. This enables methods that access higher-order oracles, which leads to the opposite asymmetry (*q < p* + 1). Examples include the objective function of logistic regression, which is known to be infinite-order smooth. Coupled with standard ℓ2-regularization, the problem can be analyzed as a p th-order smooth and strongly convex (q = 2) problem, e.g., p = 2 with access to the Hessian matrix, p = 3 accessing the third-order derivative tensor. In addressing specific instances of this asymmetry, previous works established some upper bounds (Gasnikov et al., 2019; Song et al., 2021) and lower bounds (Arjevani et al., 2019; Kornowski 1
& Shamir, 2020; Doikov, 2022; Thomsen & Doikov, 2024) for the oracle complexity. Notably, Song et al. (2021) proposed a unified acceleration framework for functions that are p th-order Holder smooth with degree ¨ ν, and uniformly convex with degree q, providing upper bounds for any combination of p, q, and ν. For the case where *q > p* + ν, they show an oracle complexity of O
 H
σ 2 3(p+ν)−2σ ϵ 2(q−p−ν)
q(3(p+ν)−2) 
, and for the case where *q < p* + ν, the complexity is O
 Hσ 2 3(p+ν)−2 + log log σ p+ν Hq 1 p+ν−q 1 ϵ
. To the best of our knowledge, no lower bounds exist in this general setting, particularly with Holder smoothness and uniform convexity. ¨ In this paper, we provide matching lower bounds to the upper bounds in (Song et al., 2021) for these asymmetric cases. Specifically, we establish Ω
 H
σ 2 3(p+ν)−2σϵ 2(q−p−ν)
q(3(p+ν)−2) 
for *q > p* + ν and Ω
 H
σ 2 3(p+ν)−2 + log log σ p+ν Hq 1 p+ν−q 1 ϵ for *q < p* + ν. For the *q > p* + ν case, we adopt the framework proposed by (Guzman & Nemirovski ´ , 2015), utilizing a smoothing operator to generate a high-order smooth function. We propose the use of ℓ∞-ball-truncated Gaussian smoothing, which, as we later justify, is novelly designed to achieve the optimal rate and be compatible with both high-order smooth and uniformly convex settings. Both the truncated Gaussian smoothing and the construction of the ℓ∞ ball are crucial to improve upon the sub-optimal derivation using uniform smoothing within an ℓ2 ball in (Agarwal & Hazan, 2018). Our results generalize the lower bounds in (Doikov, 2022; Thomsen & Doikov, 2024) to higher-order and Holder smooth settings. For the ¨ q < p + ν case, we adopt Nesterov's framework (Nesterov et al., 2018) and generalize the lower bounds in (Arjevani et al., 2019; Kornowski & Shamir, 2020) to include Holder smooth and uniformly ¨ convex settings.

## 2 Related Work

Upper Bounds. Doikov & Nesterov (2021) showcase the upper bound for uniformly convex functions with Holder-continuous Hessian via cubic regularized Newton method, but the rate is not ¨ optimal. For higher order result, Bubeck et al. (2019) and Jiang et al. (2019) established a near optimal upper bound of O˜ϵ
− 2 3p+1 in the simpler case of ν = 1 without uniform convexity. Gasnikov et al.

(2019) achieve the same near-optimal rate, but also consider uniform convexity, and by the restarting mechanism, derive the rate that for *q > p* + 1 as well, generalizing the upper bounds established in second-order (Monteiro & Svaiter, 2013) and matching the lower bounds later derived in (Kornowski
& Shamir, 2020). Kovalev & Gasnikov (2022) closed the log 1ϵ gap, but does not consider uniform convexity or Holder smoothness. For minimizing uniformly convex functions, ¨ Juditsky & Nesterov (2014) and Roulet & d'Aspremont (2017) study the complexity of first-order methods. Recently, Song et al. (2021) establish the most general upper bounds for arbitrary combinations of the order of Holder smoothness and the degree of uniform convexity, which include the rates for both ¨ *q > p* + ν and *q < p* + ν cases. Lower Bounds. Agarwal & Hazan (2018) proved for p th-order smooth convex functions an Ω
ϵ
− 2 5p+1 lower bound based on constructing the hard function with randomized smoothing uniformly over a unit ball. But their rate is not optimal due to the extra dimension factor appearing in the smoothness constant due to the uniform randomized smoothing. Garg et al. (2021) added softmax smoothing prior to randomized smoothing, achieving a near-optimal rate of Ω
ϵ
− 2 3p+1 for randomized and quantum algorithms. Separately, Arjevani et al. (2019) also established the optimal lower bound of Ω
ϵ
− 2 3p+1 with the Nesterov's hard function construction approach. Furthermore, for the asymmetric case of *q < p* + 1, Arjevani et al. (2019) proved the lower bound of Ω
 H
σ 27 + log log σ 3 H2 ϵ
−1 for the p = 2 and q = 2 case, and the result is later generalized to the p th order in (Kornowski & Shamir, 2020). No q > 2 uniformly convex settings were considered in these works. For the case of *q > p* + ν, lower bounds for uniformly convex functions for q ≥ 3 are limited to the first-order smoothness setting where p = 1 (Juditsky & Nesterov, 2014; Doikov, 2022; Thomsen & Doikov, 2024). No lower bounds for uniformly convex functions were established, to our knowledge, in the high-order setting.

## 3 Preliminaries And Settings

Notations. We use [n] to represent the set {1, 2*, ..., n*}. We use *∥ · ∥* to denote an ℓ2 operator norm.

We use ∇ for gradients, ∂ for subgradients, and ⟨·, ·⟩ for inner products. Related to the algorithm, bold lower letters for vectors (e.g., x, y), and with subscript, the vectors in different iterations (e.g., xT ). We use regular lower letters for scalars, and with subscript, a coordinate of a vector (e.g., xi). Depending on the context, we use capital letters for a matrix or a random variable. We use ϕ for the probability density function of the standard normal or the standard multivariate normal (MVN),
and Φ for the cumulative (density) function of standard normal or MVN. We further overuse the notation of ϕ[·,·] Φ[·,·]for their truncated counterparts for the normal distribution (standard normal if not specified with parameters), and ϕ∥·∥∞≤· Φ∥·∥∞≤· for the MVN truncated within an ℓ∞ ball.

## 3.1 Definitions

Definition 1 (High-order Smoothness). For p ∈ Z
+*, a function* f : R
d → R is p th-order smooth or whose p th- derivatives are Lp*-Lipschitz if for* Lp > 0, ∀ x, y ∈ R
d, ∥∇pf(x) − ∇pf(y)∥ ≤
Lp∥x − y∥.

Definition 2 (High-order Holder Smoothness) ¨ . For p ∈ Z
+*, a function* f : R
d → R is p thorder Holder smooth or has H ¨ *older continuous* ¨ p th-order derivatives if for ν ∈ (0, 1] and H > 0,
∀ x, y ∈ R
d, ∥∇pf(x) − ∇pf(y)∥ ≤ H∥x − y∥
ν.

Definition 3. (Uniform Convexity (Nesterov et al., 2018, Section 4.2.2)) For integer q ≥ 2 and σ > 0*, a function* f : R
d → R is uniformly convex with degree q *and modulus* σ if ∀ x, y ∈ R
d, f(y)−f(x)−⟨∇f(x), y − x⟩ ≥ σq
∥y−x∥
q*, or the function satisfies* ⟨∇f(y) − ∇f(x), y − x⟩ ≥
σ∥y − x∥
q.

## 4 Lower Bound For The Q > P + Ν Case

The derivation of the lower bound is to find such a function by construction that satisfies the uniformly convex and Holder smooth conditions and requires at least a certain amount of iterations to reach an ¨
ϵ-approximate solution. The general steps follow from the framework of showing lower complexity bounds for smooth convex optimization (Guzman & Nemirovski ´ , 2015), which originates from (Nemirovskii & Nesterov, 1985) and serves as the basis for results in various follow-up settings (Agarwal & Hazan, 2018; Garg et al., 2021; Doikov, 2022). The construction starts from a nonsmooth function, then smooths the function with some smoothing operator (e.g. Moreau envelope in
(Guzman & Nemirovski ´ , 2015; Doikov, 2022), randomized smoothing uniformly within a ball in (Agarwal & Hazan, 2018; Garg et al., 2021)). We design a truncated Gaussian smoothing operator within the ℓ∞ ball and start the derivation by stating its formal definition and key properties.

## 4.1 Truncated Gaussian Smoothing

Definition 4 (Truncated Gaussian Smoothing). For f : R
d → R and a parameter ρ > 0, define the truncated Gaussian smoothing operator Sρ[f] : (R
d → R) → (R
d → R) as Sρ[f](x) = EV [f(x + ρV )]
where V is a d-dimensional random variable that follows the standard multivariate normal (MVN) distribution truncated within a unit ball. That is, the probability density function (PDF) of V is

$$\mathbb{P}[V=\mathbf{v}]={\frac{1}{Z(d)(2\pi)^{\frac{d}{2}}}}\exp\left\{-{\frac{\mathbf{v}^{\top}\mathbf{v}}{2}}\right\}\mathbb{I}_{[\|\mathbf{v}\|_{\infty}\leq1]},$$

in which I[·] = 1 if · is true 0 otherwise is the indicator function and Z(d) is the normalizing factor, i.e., the cumulative distribution within the d-dimensional unit ℓ∞-ball (Cartinhour, *1990).*
We denote fρ = Sρ[f], and use the shorthand notation for the function that applied the smoothing operator for p *times:* f p ρ = S
p ρ[f] = Sρ[· · · [Sρ[f]] · · · ] for p times.

Now we justify the choice of truncated Gaussian smoothing for the construction of hard function. We notice that Agarwal & Hazan (2018) choose randomized smoothing uniformly over a unit ℓ2-ball, which by their Lemma 2.3 that the smoothed function is O(d)-smooth (which in fact can be tightened to O(
√d) by (Yousefian et al., 2012; Duchi et al., 2012, Lemma 8)) where d is the dimension of the variable. Since the number of iteration T ∈ O(d), their result OT
− 2 5p+1 is sub-optimal by an extra T comparing to the tight lower bound OT
− 2 3p+1 (Arjevani et al., 2019). Therefore we search for a smoothing operator with Lipschitz constant being *dimension-free*. We notice that Gaussian smoothing (Duchi et al., 2012, Lemma 9), softmax smoothing (Bullins, 2020, Lemma 7), and Moreau smoothing (Doikov, 2022, Lemma 1) are such operators.

Yet as the reader will later see in the proof that the converging points are generated through a sequence of functions, instead of those generated from one hard function. For these two sequences of points to be identical so that the lower bound is indeed for optimizing the hard function constructed, we need the smoothing operator to be *local*, that is, accessing information within *some neighborhood* of the queried point, e.g., a unit ℓ2-ball in (Doikov, 2022). Unfortunately, Gaussian smoothing and softmax smoothing need access to global information. For Moreau smoothing that indeed depends on local information, it's successfully applied in proving the lower bound in the first-order setting (Doikov, 2022), but is not suited for the high-order setting. First, one may attempt the extension of Moreau smoothing with a p th-power regularization, yet it can be shown that the function is not p th-order smooth. Next, one may try to apply Moreau smoothing p times, yet unlike randomized smoothing in (Agarwal & Hazan, 2018), the Lipschitz constant does not raise to the p th-power with the number of times the smoothing operator is applied, which leads to the same rate as in the first order. Observing the proof of (Agarwal & Hazan, 2018, Corollary 2.4), this is in essence due to the fact that the minimization in Moreau smoothing does not commute with derivative, whereas the expectation in randomized smoothing does. We then come up with the idea of a truncated multivariate Gaussian smoothing operator that is (i)
local (ii) smooth with a dimension-free constant (iii) p th-order smooth with smoothness constant raising to the p th power as well. Initially, we applied the Gaussian smoothing truncated within a unit ball in ℓ2 by default. We noticed later, however, that the marginal distribution of unit-ℓ2-ball truncated multivariate Gaussian is not the truncated standard normal between [−1, 1], but with an extra d-dependent normalizing constant, which adds the d-dependency to the smoothness constant of the hard function. To ensure a dimension-free smoothness constant, we instead apply the multivariate Gaussian smoothing truncated within an ℓ∞ ball, a.k.a., the hypercube with edge length 2, whose marginal distribution is indeed the truncated standard normal between [−1, 1] (Cartinhour, 1990). The following lemma characterizes these desired properties including convexity, continuity, approximation, and smoothness, with proof deferred to Appendix A.1. Lemma 1. Given a L-Lipschitz function f*, the function* f p ρ = Sρ[· · · [Sρ[f]] · · · ] satisfies
(i) If f *is convex,* f p ρis convex and L-Lipschitz with respect to the ℓ2 norm.

(ii) If f *is convex,* f(x) ≤ f p ρ(x) ≤ f(x) + 
5p 4 Lρ√d.

(iii) ∀i ∈ [p], ∀x, x
′ ∈ R
d, ∥∇if p ρ(x) − ∇if p ρ(x
′)∥ ≤ 2ρ iL∥x − x

$$-\mathbf{x}^{\prime}\rVert.$$

4.2 THE LOWER BOUND: FUNCTION CONSTRUCTION AND TRAJECTORY GENERATION Theorem 1. For any T*-step* (
√d − 1 ≤ T ≤ d) deterministic algorithm A *with oracle access up to* the p th order, there exists a convex function f(x) *whose* p th-order derivative is Holder continuous of ¨
degree ν with modulus H *and a corresponding* F(x) = f(x) + 
σ q
∥x∥
q *with regularization that is* uniformly convex of degree q with modulus σ, such that q > p + ν*, it takes*

$L\parallel$. 
$$T\in\Omega\left(\left({\frac{H}{\sigma}}\right)^{{\frac{2}{3(p+\nu)-2}}}\left({\frac{\sigma}{\epsilon}}\right)^{{\frac{2(q-p-\nu)}{q(3(p+\nu)-2)}}}\right)$$
!
steps to reach an ϵ-approximate solution xT *satisfying* F(xT ) − F(x
∗) ≤ ϵ.

Proof. We begin the proof by constructing the hard function.

4.2.1 FUNCTION CONSTRUCTION WITH TRUNCATED GAUSSIAN SMOOTHING
1. Non-smooth Function Construction. We first construct the function gt(x) = max 1≤k≤t rk(x) where ∀ k ∈ [T], rk(x) = ξkeα(k), x− (k − 1)δ.

ξk *∈ {−*1, 1}, e is the standard basis, α is a permutation of [T], and δ > 0 is some parameter that we will choose later. Lemma 2 characterizes the properties of gt with proof in Appendix A.2. Lemma 2. ∀ t ∈ [T], gt is convex and 1-Lipschitz with respect to the ℓ∞-norm, and also the ℓ2*-norm.* 2. Truncate Gaussian Smoothing. Next, we smooth the function gt(x) with truncate Gaussian smoothing as in Definition 4. Given a parameter ρ > 0 and p ∈ Z
+,
Gt(x) = S
p ρ[gt](x)

$\uparrow$ 3. 
Based on Lemma 1, we show that Gt(x) satisfies the following lemma, with proof in Appendix A.2.

Lemma 3. ∀ t ∈ [T], ∀ x, y ∈ R
d,
(i) Gt(x) is convex and 1-Lipschitz, i.e., Gt(x) − Gt(y) ≤ ∥x − y∥.

(ii) gt(x) ≤ Gt(x) ≤ gt(x) + 54 pρ√d.

(iii) *For some fixed* p ∈ Z
+, ∀ i ∈ [p], ∥∇iGt(x) − ∇iGt(y)∥ ≤ 2ρ i∥x − y∥.

3. Adding Uniform Convexity. Now that the constructed function Gt(x) is all-order smooth, we add to it the uniformly convex regularization. We define

$$\begin{array}{l l l}{{}}&{{}}&{{f_{t}(\mathbf{x})=\beta G_{t}(\mathbf{x})}}\\ {{}}&{{}}&{{F_{t}(\mathbf{x})=f_{t}(\mathbf{x})+d_{q}(\mathbf{x})\quad{\mathrm{~for~}}\quad d_{q}(\mathbf{x})={\frac{\sigma}{q}}\|\mathbf{x}\|^{q},\quad\mathbf{x}\in\mathbb{Q}}}\end{array}$$
$$\begin{array}{r}{f(\mathbf{x})=f_{T}(\mathbf{x})}\\ {F(\mathbf{x})=F_{T}(\mathbf{x}),}\end{array}$$
q, x ∈ Q F(x) = FT (x),
where β > 0 is a parameter that we will choose later, Q = {x : ∥x∥2 ≤ D}
1for D ≤ H
2 1−νC
 1 q−p−ν and C = σ(q − 1) *× · · · ×* (q − p).

Lemma 4. For F(x) = fT (x) + dq(x) *where* dq(x) = σ q x qand x ∈ Q,
(i) F is uniformly convex function with degree q and modulus σ > 0.

(ii) F(x) is p th*-order Holder smooth with parameter* ¨ H =
2 p+1β ρp+ν−1 , ∀ p ∈ Z
+.

Therefore, by Lemma 4, the function constructed satisfies the desired uniform convexity and highorder smoothness conditions. Next, we characterize with Lemma 5 the upper and lower bounds of the constructed function which will be used in the proof later.

Lemma 5. For R(x) = β maxk∈[T] ξkeα(k), x+
σ q
∥x∥
q*, we have*

$$R(\mathbf{x})-\beta(T-1)\delta\leq F(\mathbf{x})\leq R(\mathbf{x})+{\frac{5}{4}}p\beta\rho{\sqrt{d}}.$$

4.2.2 CONVERGENCE TRAJECTORY GENERATION
4. Trajectory Generation Procedure. The trajectory is generated following a standard T-step iterative procedure same as outlined in (Guzman & Nemirovski ´ , 2015; Doikov, 2022):
· For t = 1, x1 is the first point of the trajectory and is chosen by initialization of some algorithm A, independent of F. Subsequently, choose α(1) ∈ arg max k∈[T]
eα(k), x1 ξ1 = sign eα(1), x1 ,
after which a fixed F1(x) is generated.

· For 2 ≤ t ≤ T, at the beginning of each such iteration, we have access to x1, *· · ·* , xt−1, the function Ft−1, and its gradient information, which we denote as It−1(x) =
{Ft−1, ∇Ft−1, *· · ·* , ∇pFt−1}. The algorithm A generates the next point with this information:
xt = A(It−1(x1), *· · ·* , It−1(xt−1)). Then choose

$\alpha(t)\in\operatorname*{arg\,max}_{k\in[T]\setminus\{\alpha(i):i<t\}}\left|\left\langle\mathbf{e}_{\alpha(k)},\,\mathbf{x}_{t}\right\rangle\right|\qquad\qquad\xi_{t}=\operatorname*{sign}\left(\left\langle\mathbf{e}_{\alpha(t)},\,\mathbf{x}_{t}\right\rangle\right).$
after which a fixed Ft(x) is generated for the next iteration.

5. Indistinguishability of Ft and F *for Trajectory Generation.* It's important to note that the trajectory x1, *· · ·* , xT is generated based on a sequence of functions F1, · · · , FT , whereas our object of analysis should be just *one hard function* F = FT . Here we show:
Lemma 6. The trajectory x1, · · · , xT generated by applying an algorithm A iteratively on the sequence of functions F1, · · · , FT *, with up to* p th-order oracle access, is the same as the trajectory generated applying A directly on F *when oracle access pertains only local information within an* ℓ∞*-ball with radius* δ/2.

Proof. The idea is to show that ∀ 2 ≤ t ≤ T, the function gt coincides with gT (so that Ft coincides with FT in terms of generating xt+1, i.e., It = IT ) under some mild conditions. Similar proof can be found in (Guzman & Nemirovski ´ , 2015; Doikov, 2022, Section 3). By construction, ∀ t ∈ [T],

$$g_{t}(\mathbf{x})=\max_{1\leq k\leq t}r_{k}(\mathbf{x})=\max\left\{\max_{1\leq k\leq s}r_{k}(\mathbf{x}),\max_{s<k\leq t}r_{k}(\mathbf{x})\right\}=\max\left\{g_{s}(\mathbf{x}),\max_{s<k\leq t}r_{k}(\mathbf{x})\right\}.$$
Furthermore, $\alpha(s)\in\arg\max_{k\in[T]\smallsetminus\{\alpha(i):i<s\}}\left|\left\langle\mathbf{e}_{\alpha(k)},\,\mathbf{x}_{s}\right\rangle\right|$ and $\xi_{s}=\operatorname{sign}\left(\left\langle\mathbf{e}_{\alpha(s)},\,\mathbf{x}_{s}\right\rangle\right)$, therefore,
$$g_{s}(\mathbf{x}_{s})=\max_{1\leq k\leq s}\xi_{k}\left\langle\mathbf{e}_{\alpha(k)},\,\mathbf{x}_{s}\right\rangle-(k-1)\delta\geq\max_{1\leq k\leq s}\xi_{k}\left\langle\mathbf{e}_{\alpha(k)},\,\mathbf{x}_{s}\right\rangle-(s-1)\delta$$ $$\geq\left|\left\langle\mathbf{e}_{\alpha(s)},\,\mathbf{x}_{s}\right\rangle\right|-(s-1)\delta\geq\max_{s<k\leq t}\xi_{k}\left\langle\mathbf{e}_{\alpha(k)},\,\mathbf{x}_{s}\right\rangle-(s-1)\delta$$ $$\geq\max_{s<k\leq t}\xi_{k}\left\langle\mathbf{e}_{\alpha(k)},\,\mathbf{x}_{s}\right\rangle-(k-1)\delta+\delta\qquad\qquad(k,s\in\mathbb{Z}^{+},k>s\implies k\geq s+1)\.$$

If we limit the information access within an ℓ∞-ball with radius δ/2 when searching for the next point xs+1 from xs, we then establish a local region ∀x, ∥x − xs∥∞ ≤
δ 2
. Further by Lemma 2 that gs (also ξkeα(k), x) is 1-Lipschitz with respect to the ℓ∞ norm, we have ∀ k such that s < k ≤ t,

$$g_{s}({\bf x}_{s})\geq\xi_{k}\left\langle{\bf e}_{\alpha(k)},\,{\bf x}_{s}\right\rangle-(k-1)\delta+2\|{\bf x}-{\bf x}_{s}\|_{\infty}$$ $$\geq\xi_{k}\left\langle{\bf e}_{\alpha(k)},\,{\bf x}_{s}\right\rangle-(k-1)\delta+\left[g_{s}({\bf x}_{s})-g_{s}({\bf x})\right]+\left[\xi_{k}\left\langle{\bf e}_{\alpha(k)},\,{\bf x}\right\rangle-\xi_{k}\left\langle{\bf e}_{\alpha(k)},\,{\bf x}_{s}\right\rangle\right],$$

which implies that gs(x) ≥ maxs<k≤t ξkeα(k), x−(k−1)δ = maxs<k≤t rk(x). This concludes that ∀ x such that ∥x − xs∥∞ ≤
δ 2
, gt(x) = max {gs(x), maxs<k≤t rk(x)} = gs(x), which further implies Ft(x) = Fs(x). Letting t = T we have ∀ t ∈ [T], Ft(x) = FT (x) for ∥x − xt∥∞ ≤
δ 2
.

## 4.2.3 Lower Bound Derivation

6. Bounding the Optimality Gap. The following lemma bounds optimality gap, whose proof is based on Lemma 5, and is presented in Appendix A.2.

Lemma 7. F(xT ) − F(x
∗) ≥ −β(T − 1)δ −
5 4 pβρ√d +
q−1 q β q σT
q 2 1 q−1.

7. Setting the parameters. By Definition 4 and Lemma 14 (i), we know that Sρ[gt](x), ∇Sρ[gt](x) depends on the value of gt(x) within an ℓ∞-ball of radius ρ. Therefore inductively, we see that for F(x) = βSp ρ[gT ](x) + σq
∥x∥
q, F(x), ∇F(x), *· · ·* , ∇pF(x) depends on the value of F(x)
within an ℓ∞-ball of radius pρ. For our construction to hold, we also need Ft(x) and F(x) to be indistinguishable ∀ t ∈ [T], which is true within an ℓ∞-ball of radius δ/2. Therefore, we set δ = 2pρ, so that for the purpose of oracle access at xt (computing (high-order) gradients of F), it's indistinguishable to replace F(x) with Ft(x), and the sequence generated as in Section 4.2.2 is the same as that directly applying some p th-order algorithm A on F(x). In other words, F(x) and the generated xT serve as valid components for deriving the lower bound.

As a result, F(xT )−F(x ∗) ≥ −2pβρ(T −1+ 58 √d)+ q−1 q β q σT q 2  1 q−1. Let T ≥ √d−1 ≥ 5 8 √d−1, then F(xT ) − F(x ∗) ≥ −4pβρT + q−1 q β q σT q 2  1 q−1. By letting 4pβρT = q−1 2q β q σT q 2  1 q−1, we solve for ρ = q−1 8pq σ − 1 q−1 β 1 q−1 T 2−3q 2(q−1) = cqσ − 1 q−1 β1 q−1 T 2−3q 2(q−1) , in which cq = q−1 8pq , and at the same time,
$\frac{1}{q-1}\,T^{\frac{2-3q}{2(q-1)}}=c_{q}\sigma^{-\frac{1}{q-1}}\,\beta^{\frac{1}{q-1}}\,T^{\frac{2-3q}{2(q-1)}}$, in which $c_{q}=\frac{q-1}{8pq}$, and at the  $$F({\bf x}_{T})-F({\bf x}^{*})\geq\frac{q-1}{2q}\left(\frac{\beta^{q}}{\sigma T^{\frac{2}{2}}}\right)^{\frac{1}{q-1}}.\tag{1}$$
By the construction of F(x) and Lemma 4, we know that F(x) is p th-order Holder smooth with ¨
parameter H =
2 p+1β ρp+ν−1 . Plugging in the value of ρ, we have

$$H=2^{p+1}c_{q}^{-(p+\nu-1)}\sigma^{\frac{p+\nu-1}{q-1}}\beta^{-\frac{p-q+\nu}{q-1}}T^{\frac{(p+\nu-1)(3q-2)}{2(q-1)}},$$
$$\Omega=2^{-\frac{1}{2}}\epsilon_{0}\left(\frac{\mu_{0}\left(\mu_{0}\right)}{\sigma^{2}}\right)^{\frac{1}{2}}\left(\frac{\mu_{0}\left(\mu_{0}\right)}{\sigma^{2}}\right)^{\frac{1}{2}}\,,$$  equivalently, $\beta=\left(\frac{\mu_{0}e^{\mu_{0}+\epsilon_{0}}-\frac{1+\mu_{0}\epsilon_{0}}{\sigma^{2}}}{2^{\mu_{0}+\epsilon_{0}}}\right)^{-\frac{\mu_{0}\epsilon_{0}}{\sigma^{2}}}T^{\frac{(\mu_{0}+\epsilon_{0})(3\mu_{0}-2)}{2(\mu_{0}+\epsilon_{0})}}$. Plugging the value of $\beta$ back into Eq. (1), we have 
$$F(\mathbf{x}_{T})-F(\mathbf{x}^{*})\geq\frac{q-1}{2q}\sigma^{-\frac{1}{q-1}}\left(\frac{H\sigma^{p+\nu-1}\sigma^{-\frac{p+\nu-1}{q-1}}}{2^{p+1}}\right)^{-\frac{q}{p-q+\nu}}T^{\frac{q(5p+\nu)-2}{2(p-q+\nu)}}$$ $$=4p\left(\frac{q-1}{8pq}\right)^{\frac{(p+\nu)(1-q)}{p-q+\nu}}\sigma\left(\frac{H\sigma^{-1}}{2^{p+1}}\right)^{-\frac{q}{p-q+\nu}}T^{\frac{q(3(p+\nu)-2)}{2(p-q+\nu)}}.$$

We complete the proof for Theorem 1 by letting F(xT ) − F(x
∗) ≤ ϵ, from which we solve for T ≥2
(2p+2ν+pq−q)/q− 2 3(p+ν)−2p 2(q−p−ν)
q[3(p+ν)−2] q−1 8pq 
 
2(p+ν)(q−1)
q[3(p+ν)−2]  H
σ 2 3(p+ν)−2σϵ
 
2(q−p−ν)
q[3(p+ν)−2].

## 5 Lower Bound For The Q < P + Ν Case

Theorem 2. For any T-step deterministic algorithm A *with oracle access up to the* p th order, there exists a convex function f(x) *whose* p th-order derivative is Holder continuous of degree ¨ ν with modulus H *and a corresponding* F(x) = f(x) + σ q
∥x∥
q with regularization that is uniformly convex of degree q with modulus σ, such that q < p + ν*, it takes*

$$\begin{array}{l}{{T\in\Omega\left(\left(\frac{H}{\sigma}\right)^{\frac{2}{3(p+\nu)-2}}+\log\log\left(\left(\frac{\sigma^{p+\nu}}{H^{q}}\right)^{\frac{1}{p+\nu-q}}\frac{1}{\epsilon}\right)\right)}}\end{array}$$

steps to reach an ϵ-approximate solution xT *satisfying* F(xT ) − F(x
∗) ≤ ϵ.

Proof. Similar to all other lower bound proofs, we construct such a function that satisfies the uniformly convex and Holder smooth conditions and show that it requires at least the number of ¨ iterations stated in the theorem. The construction is generally based on Nesterov's hard function
(Nesterov et al., 2018), and generalizes the construction in (Arjevani et al., 2019) to higher-order and the construction in (Kornowski & Shamir, 2020) to Holder smooth functions as well as uniformly ¨ convex functions.

## 5.1 Function Construction Based On Nesterov'S Hard Function

A direct generalization of Nestrov's construction for first- and second-order lower bounds (Nesterov et al., 2018, Section 2.1.2, 4.3.1) to the p th-order Holder smooth setting takes the form ¨
˜f(x) =
1 p+ν PT˜
i=1 |xi − xi+1| p+ν − γx1 +
σ˜
q
∥x∥
q, for *q < p* + ν, ν ∈ [0, 1], which is uniformly convex by the regularization. We further add a coefficient so that the function p th-order Holder smooth with ¨
the desired parameter H and further on top of this a set of orthogonal basis vi, ∀ i ∈ [T˜] to limit the access of coordinates through the iterations:

$$f(\mathbf{x})={\frac{H}{2^{p+\nu+1}(p+\nu-1)!}}\left({\frac{1}{p+\nu}}\sum_{i=1}^{\tilde{T}}|\langle\mathbf{v}_{i},\,\mathbf{x}\rangle-\langle\mathbf{v}_{i+1},\,\mathbf{x}\rangle|^{p+\nu}-\gamma\,\langle\mathbf{v}_{1},\,\mathbf{x}\rangle\right)+{\frac{\sigma}{q}}\|\mathbf{x}\|^{q},$$

for σ =Hσ˜
2 p+ν+1(p+ν−1)! , or equivalently, σ˜ =
2 p+ν+1(p+ν−1)!σ H. viis chosen iteratively to be orthogonal to x1, *· · ·* , xi and v1, *· · ·* , vi−1. Similar to (Arjevani et al., 2019, Lemma 7), one can show that the oracle information of f(xi), ∀ i ≤ t does not depend on vt+1, *· · ·* , vT˜, so that the iterative construction of viis valid, i.e., does not affect the xi generated running an algorithm on f.

Now we characterize the relation between ˜f and f.

Lemma 8. x
∗ = arg minx f(x), y = arg minx
˜f(x)*. (i)* ∀i ∈ [T˜], ⟨vi, x
∗⟩ = yi*. (ii)* ∥x
∗∥ = ∥y∥.

Next, we characterize the convexity and smoothness of the constructed function. Specifically, we can show with the proof in Appendix B that f satisfies the following lemma. Lemma 9. f(x) is (i) uniformly convex with degree q and parameter σ*. (ii)* p th*-order Holder smooth* ¨
with degree ν *and parameter* H. The analysis of (Nesterov et al., 2018) then derives the lower bound based on the closed-form optimal solution that minimizes the hard function. For our generalized construction of f, however, the closed-form solution is hard to obtain. As in (Arjevani et al., 2019), we instead analyze some properties of f for each of these lower bounds. For simplicity, we state the properties for function ˜f, and since f is simply a scaling of ˜f, the properties also apply to f with a difference of constants. To prove Theorem 2, we show separately for the  Hσ 2 3(p+ν)−2term and the log log σ p+ν Hq 1 p+ν−q 1 ϵ term. The derivation is largely based on some key lemmas whose complete proof is in Appendix B.

$$\begin{array}{r l}{{\bf5.2}}&{{}T\in\Omega\left(\left({\frac{H}{\sigma}}\right)^{\frac{2}{3(p+\nu)-2}}\right)}\end{array}$$

Since we cannot solve for a closed form solution from arg minx
˜f(x), we need to alternatively bound the solution in a relative scale. One key observation is that the coordinates of the optimal solution form a decreasing sequence (Arjevani et al., 2019; Carmon et al., 2021), and their relative relation can be characterized as in Lemma 10 (i) utilizing the first-order optimality condition. Based on the properties of each coordinate, one can relate them to the norm of the optimal solution as in Lemma 10 (iii).

Lemma 10. For y = arg minx
˜f(x),

$\mathbf{n_x}\,f(\mathbf{x})$,
$\frac{1}{2}$ ................ 
(i) ∀t ∈ [T˜], yt ≥ y1 − (t − 1)γ1 p+ν−1 . (ii) For T˜ = y1 γ1 p+ν−1 + 1, y1 ≤ γ1 p+ν−1 + r2γ p+ν p+ν−1 σ˜∥y∥q−2 . (iii) For γ ≥ σ˜ p+ν−1 p+ν−2 ∥y∥ (p+ν−1)(q−2) p+ν−2 , ∀t ∈ [T˜], yt ≥γ p+ν 2(p+ν−1) 2p+ν+1σ˜ 1 2 ∥y∥ q−2 2 +12 − iγ1 p+ν−1 .
Then the bound on the norm of the optimal solution can be established in the following lemma.

 ##### Lemma 11.$\ \|\mathbf{y}\|\leq\frac{2\frac{2}{3q-2}\gamma\frac{3(p+\nu)-2}{(p+\nu-1)(3q-2)}}{\hat{\sigma}\frac{3}{3q-2}}$. 
.
The final step is to relate this norm to the optimality gap with the property of uniform convexity. By Lemma 8 and Lemma 10 (iii), ⟨vT , x
∗⟩ = yT ≥γ p+ν 2(p+ν−1)
2 p+ν+1σ˜
12 ∥y∥
q−2 2
+12 − Tγ1 p+ν−1 . Therefore, with vT and xT orthogonal to each other by construction,

 q 2 ≥ σ q (⟨vT , xT − x ∗⟩) 2 q2 f(xT ) − f(x ∗) ≥ σ q ∥xT − x ∗∥ q = σ q  i=1 (⟨vi, xT − x ∗⟩) 2  X T˜ = σ q (⟨vT , x ∗⟩) q ≥ σ q  γp+ν 2(p+ν−1) 2 p+ν+1σ˜ 1 2 ∥y∥ q−2 2 + 1 2 − T γ1 p+ν−1 !q
In order to achieve f(xT ) − f(x
∗) ≤ ϵ, we have σq
γp+ν
2(p+ν−1)
2
p+ν+1σ˜
12 ∥y∥
q−2
2
+12 − Tγ1
p+ν−1
q≤ ϵ,
from which we can solve for T ≥γ
p+ν−2
2(p+ν−1)
2
p+ν+1σ˜
12 ∥y∥
q−2
2
+
1
2 
−
qϵ
σγq
p+ν−1
1q. For ϵ ≤
γq
p+ν−1 σ
2
qq, we
have 12 −
qϵ
σγq
p+ν−1
1q≥ 0. Therefore, T ≥γ
p+ν−2
2(p+ν−1)
2p+ν+1σ˜
1
2 ∥y∥
q−2
2
.
By Lemma 11, we know that ∥y∥ ≤ 
22
3q−2 γ3(p+ν)−2
(p+ν−*1)(3*q−2)
σ˜3
3q−2. Therefore, for x0 = 0, by Lemma 8,
∥x0 − x
∗∥ = ∥x
∗∥ = ∥y∥ ≤ 2
2
3q−2 γ3(p+ν)−2
(p+ν−*1)(3*q−2)
σ˜3
3q−2
. To satisfy the condition ∥x0 − x
∗∥ ≤ D, we let
22
3q−2 γ3(p+ν)−2
(p+ν−1)(3q−2)

```
σ˜3
 3q−2
           ≤ D, then we can solve for γ ≤ 2
                                      −
                                       2(p+ν−1)
                                       3(p+ν)−2 D
                                               (p+ν−1)(3q−2)
                                                 3(p+ν)−2 σ˜
                                                           3(p+ν−1)
                                                           3(p+ν)−2 . Plug this

```

as well as ∥y∥ ≤ D into the lower bound on T we have

T ≥ 2 − 2(p+ν−1) 3(p+ν)−2 D (p+ν−1)(3q−2) 3(p+ν)−2 σ˜ 3(p+ν−1) 3(p+ν)−2 p+ν−2 2(p+ν−1) 2 p+ν+1σ˜ 1 2 D q−2 2 = 2− p+ν−2 3(p+ν)−2 −(p+ν+1)D 2(p+ν−q−1) 3(p+ν)−2 σ˜ − 2 3(p+ν)−2
Plugging in σ˜ =
2 p+ν+1(p+ν−1)!σ H, we have T ∈ Ω
 H
σ 2 3(p+ν)−2.

$$\begin{array}{r l}{{5.3}}&{{}T\in\Omega\left(\log\log\left(\left({\frac{\sigma^{p+\nu}}{H^{q}}}\right)^{\frac{1}{p+\nu-q}}{\frac{1}{\epsilon}}\right)\right)}\end{array}$$

For the log log term, we follow a similar narrative as in Section 5.2, starting from characterizing the per-coordinate relation of the optimal solution.

Lemma 12. For y = arg minx
˜f(x), let t1 ∈ [T˜] *be such that* yt1 >1 p+ν−1 σ˜1 p+ν−2 ∥y∥
q−2 p+ν−2 and yt1+1 ≤1 p+ν−1 σ˜1 p+ν−2 ∥y∥
q−2 p+ν−2 . Then

(i) ∀i ∈ [T˜], yi = yi+1 + σ˜∥y∥ q−2 PT˜ j=i+1 yj  1 p+ν−1and yi+1 ≤1 σ˜∥y∥q−2 y p i (ii) ∀i ≥ t1, 1 cp,ν  p+ν−11 σ˜∥y∥q−2 y p+ν−1 i ≤ yi+1 where cp,ν is a constant depending on p, ν. (iii) ∀i ≤ T˜ − t1, yt1+i ≥ 1 cp,ν  (p+ν−1)((p+ν−1)i−1) p+ν−2 σ˜∥y∥ q−2 1 p+ν−2(p + ν − 1)−(p+ν−1)i.
Next, we derive the bound on the norm ∥xT −x
∗∥
qfrom the coordinate-wise properties in Lemma 12 with the basis defined for f. When constructing the function, we choose H ≥ 2 p+ν+1(p + ν − 1)!σ so that σ˜ ≤ 1. Then for basis vector vt1+i, by Lemma 8 and Lemma 12 (iii),

$$\langle\mathbf{v}_{t_{1}+i\cdot i},\mathbf{x}^{\prime}\rangle=y_{t_{1}+i\cdot i}\geq\left(\frac{1}{c_{p,\nu}}\right)^{\frac{(p+\nu-1)(p+\nu-1)^{i}-1)}{p+\nu-1}}\cdot\partial^{\frac{1}{p+\nu-1}}\|\mathbf{y}\|^{\frac{p-2}{p+\nu-2}}\cdot(p+\nu-1)^{-(p+\nu-1)^{i}}$$ $$=\left(\frac{1}{c_{p,\nu}}\right)^{\frac{(p+\nu-1)(p+\nu-1)^{i}-1)}{p+\nu-2}}\cdot\left(\frac{2^{p+\nu+1}(p+\nu-1)!\sigma}{H}\right)^{\frac{p-2}{p+\nu-1}}\|\mathbf{x}_{0}-\mathbf{x}^{\prime}\|^{\frac{p-2}{p+\nu-2}}\cdot(p+\nu-1)^{-(p+\nu-1)^{i}},$$

for x0 = 0, in which the first inequality follows from Lemma 12 (iii), and then the fact that for q ≥ 2
and σ˜ ≤ 1, σ˜1
p+ν−q ≤ σ˜1
p+ν−2 . For t1 ≤
T˜
$$\|\mathbf{x}_{T}-\mathbf{x}^{*}\|^{q}=(\|\mathbf{x}_{T}-\mathbf{x}^{*}\|^{2})^{\frac{q}{2}}\geq\left(\sum_{i=1}^{T}\left(\left(\mathbf{v}_{i},\mathbf{x}_{T}-\mathbf{x}^{*}\right)\right)^{2}\right)^{\frac{q}{2}}\geq\left(\left(\left(\mathbf{v}_{t_{1}+T},\mathbf{x}_{T}-\mathbf{x}^{*}\right)\right)^{2}\right)^{\frac{q}{2}}$$ $$=\left(\left(\left(\mathbf{v}_{t_{1}+T},\mathbf{x}^{*}\right)\right)^{2}\right)^{\frac{q}{2}}=\left(\left(\mathbf{v}_{t_{1}+T},\mathbf{x}^{*}\right)\right)^{q}$$  here the equality in the second line follows from the fact that by construction, $\mathbf{v}_{t_{1}+T}$ and $\mathbf{x}_{T}$
,
where the equality in the second line follows from the fact that by construction, vt1+T and xT are
orthogonal.
Finally, with uniform convexity, we have
f(xT ) − f(x
∗) ≥
σ
q
∥xT − x
∗∥
q
≥
1
cp,ν 
q(p+ν−*1)((*p+ν−1)T −1)
p+ν−2
·
σ
q
2
p+ν+1(p + ν − 1)!σ
H
 
q
p+ν−q
∥x0 − x
∗∥
q(q−2)
p+ν−2 · (p + ν − 1)−q(p+ν−1)T
= c*p,q,ν* ·
σ
p+ν
p+ν−q
L
q
p+ν−q
p
· (p + ν − 1)−q(p+ν−1)T
for c*p,q,ν* =
2
(p+ν+1)q
p+ν−q ((p+ν−1)!)q
p+ν−q
q
1
cp,ν 
 q(p+ν−1)((p+ν−1)T −1)
p+ν−2D
q(q−2)
p+ν−2 in which ∥x0−x
∗∥ ≤ D.
In order to achieve f(xT ) − f(x
∗) ≤ ϵ, we have c*p,q,ν* ·
σ
p+ν
p+ν−q
L
q
p+ν−q
p
·(p + ν − 1)−q(p+ν−1)T≤ ϵ, from
which we solve for T ≥ log logp+ν−1
 
c*p,q,ν*
σ
p+q−1
p−1
L
q
p−1
p
·
1
ϵ
!
+ logp+ν−1
1
q
, which completes the
proof for Theorem 2 combined with the result in Section 5.2.

## 6 Conclusion And Future Work

We provide tight lower bounds for minimizing functions with asymmetric high-order Holder smooth- ¨
ness and uniform convexity. Specifically, we show that the oracle complexity is lower bounded by Ω
 H
σ 2 3(p+ν)−2σϵ
 
2(q−p−ν)
q(3(p+ν)−2) for the *q > p* + ν case with the construction of a ℓ∞-balltruncated-Gaussian smoothed hard function, and Ω
 Hσ 2 3(p+ν)−2 + log log σ p+ν Hq 1 p+ν−q 1 ϵ for the *q < p* + ν case. Both lower bounds match the corresponding upper bounds in the general setting.

We note that the lower bounds for the *q > p* + ν case and the *q < p* + ν case are derived based on two different frameworks. The first lower bound based on Nemirovski's max function can be directly extended to hold for randomized algorithms based on "robust-zero-chain" arguments by (Carmon et al., 2020; 2021). The second lower bound based on Nesterov's function, which is not a robust zero-chain, holds only for deterministic/zero-respecting algorithms. We further note that the lower bound for the q = p + ν case is not included in this paper. Proposing a unified framework for all three cases as well as generalizing the results to work for randomized algorithms would be of great interest, which we leave for future work.

## References

Naman Agarwal and Elad Hazan. Lower bounds for higher-order convex optimization. In Conference On Learning Theory, pp. 774–792. PMLR, 2018. (Cited on pages 2, 3, and 4.)
Yossi Arjevani, Ohad Shamir, and Ron Shiff. Oracle complexity of second-order methods for smooth convex optimization. *Mathematical Programming*, 178(1):327–360, 2019. (Cited on pages 1, 2, 4, 7, 8, 20, and 21.)
Dimitri P Bertsekas. Stochastic optimization problems with nondifferentiable cost functionals.

Journal of Optimization Theory and Applications, 12(2):218–231, 1973. (Cited on page 14.)
Zygmunt William Birnbaum and FC Andrews. On sums of symmetrically truncated normal random variables. *The Annals of Mathematical Statistics*, 20(3):458–461, 1949. (Cited on page 16.)
Stephen P Boyd and Lieven Vandenberghe. *Convex optimization*. Cambridge university press, 2004.

(Cited on page 14.)
Sebastien Bubeck, Qijia Jiang, Yin Tat Lee, Yuanzhi Li, and Aaron Sidford. Near-optimal method ´
for highly smooth convex optimization. In *Conference on Learning Theory*, pp. 492–507. PMLR, 2019. (Cited on pages 1 and 2.)
Brian Bullins. Highly smooth minimization of non-smooth problems. In Conference on Learning Theory, pp. 988–1030. PMLR, 2020. (Cited on page 4.)
Yair Carmon, John C Duchi, Oliver Hinder, and Aaron Sidford. Lower bounds for finding stationary points i. *Mathematical Programming*, 184(1):71–120, 2020. (Cited on page 10.)
Yair Carmon, John C Duchi, Oliver Hinder, and Aaron Sidford. Lower bounds for finding stationary points ii: first-order methods. *Mathematical Programming*, 185(1):315–355, 2021. (Cited on pages 8 and 10.)
Jack Cartinhour. One-dimensional marginal density functions of a truncated multivariate normal density function. *Communications in Statistics-Theory and Methods*, 19(1):197–203, 1990. (Cited on pages 3, 4, and 13.)
Hao Chen, Lanshan Han, and Alvin Lim. A note on the sum of non-identically distributed doubly truncated normal distributions. *arXiv preprint arXiv:2008.07954*, 2020. (Cited on pages 15 and 16.)
Nikita Doikov. Lower complexity bounds for minimizing regularized functions. arXiv preprint arXiv:2202.04545, 2022. (Cited on pages 2, 3, 4, 5, 6, and 19.)
Nikita Doikov and Yurii Nesterov. Minimizing uniformly convex functions by cubic regularization of newton method. *Journal of Optimization Theory and Applications*, 189(1):317–339, 2021. (Cited on page 2.)
John C Duchi, Peter L Bartlett, and Martin J Wainwright. Randomized smoothing for stochastic optimization. *SIAM Journal on Optimization*, 22(2):674–701, 2012. (Cited on pages 4, 14, and 15.)
Ankit Garg, Robin Kothari, Praneeth Netrapalli, and Suhail Sherif. Near-optimal lower bounds for convex optimization for all orders of smoothness. Advances in Neural Information Processing Systems, 34:29874–29884, 2021. (Cited on pages 2 and 3.)
Alexander Gasnikov, Pavel Dvurechensky, Eduard Gorbunov, Evgeniya Vorontsova, Daniil Selikhanovych, and Cesar A Uribe. Optimal tensor methods in smooth convex and uniformly ´
convexoptimization. In *Conference on Learning Theory*, pp. 1374–1391. PMLR, 2019. (Cited on pages 1 and 2.)
Cristobal Guzm ´ an and Arkadi Nemirovski. On lower complexity bounds for large-scale smooth ´
convex optimization. *Journal of Complexity*, 31(1):1–14, 2015. (Cited on pages 2, 3, 5, and 6.)
Bo Jiang, Haoyue Wang, and Shuzhong Zhang. An optimal high-order tensor method for convex optimization. In *Conference on Learning Theory*, pp. 1799–1801. PMLR, 2019. (Cited on page 2.)
Anatoli Juditsky and Yuri Nesterov. Primal-dual subgradient methods for minimizing uniformly convex functions. *arXiv preprint arXiv:1401.1792*, 2014. (Cited on pages 2 and 5.)
Guy Kornowski and Ohad Shamir. High-order oracle complexity of smooth and strongly convex optimization. *arXiv preprint arXiv:2010.06642*, 2020. (Cited on pages 1, 2, and 7.)
Dmitry Kovalev and Alexander Gasnikov. The first optimal acceleration of high-order methods in smooth convex optimization. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), *Advances in Neural Information Processing Systems*, 2022. URL https://openreview.net/forum?id=YgmiL2Ur01P. (Cited on page 2.)
Hariharan Lakshmanan and Daniela Pucci De Farias. Decentralized resource allocation in dynamic networks of agents. *SIAM Journal on Optimization*, 19(2):911–940, 2008. (Cited on page 15.)
Renato DC Monteiro and Benar Fux Svaiter. An accelerated hybrid proximal extragradient method for convex optimization and its implications to second-order methods. *SIAM Journal on Optimization*, 23(2):1092–1125, 2013. (Cited on page 2.)
Arkaddii S Nemirovskii and Yu E Nesterov. Optimal methods of smooth convex minimization. USSR
Computational Mathematics and Mathematical Physics, 25(2):21–30, 1985. (Cited on page 3.)
Yurii Nesterov and Boris T Polyak. Cubic regularization of newton method and its global performance.

Mathematical programming, 108(1):177–205, 2006. (Cited on page 1.)
Yurii Nesterov and Vladimir Spokoiny. Random gradient-free minimization of convex functions.

Foundations of Computational Mathematics, 17(2):527–566, 2017. (Cited on page 15.)
Yurii Nesterov et al. *Lectures on convex optimization*, volume 137. Springer, 2018. (Cited on pages 2, 3, 7, 8, and 18.)
Vincent Roulet and Alexandre d'Aspremont. Sharpness, restart and acceleration. Advances in Neural Information Processing Systems, 30, 2017. (Cited on page 2.)
Chaobing Song, Yong Jiang, and Yi Ma. Unified acceleration of high-order algorithms under general holder continuity. *SIAM Journal on Optimization*, 31(3):1797–1826, 2021. (Cited on pages 1, 2, and 5.)
Daniel Berg Thomsen and Nikita Doikov. Complexity of minimizing regularized convex quadratic functions. *arXiv preprint arXiv:2404.17543*, 2024. (Cited on pages 2, 3, and 5.)
Farzad Yousefian, Angelia Nedic, and Uday V Shanbhag. On stochastic gradient and subgradient ´
methods with adaptive steplength sequences. *Automatica*, 48(1):56–67, 2012. (Cited on page 4.)
Wenqi Zhu and Coralia Cartis. Quartic polynomial sub-problem solutions in tensor methods for nonconvex optimization. In *NeurIPS 2022 Workshop*, 2022. (Cited on page 1.)

## Appendices

A PROOF FOR TECHNICAL LEMMAS IN SECTION 4 A.1 PROPERTIES OF TRUNCATED GAUSSIAN SMOOTHING
Lemma 13 (ℓ∞-Ball Truncated Gaussian and Its Marginal Distribution). For standard MVN truncated in a unit ℓ∞*-ball as defined in Definition* 4:

$$\mathbb{P}[V=\mathbf{v}]={\frac{1}{Z(2\pi)^{\frac{d}{2}}}}\exp\left\{-{\frac{\mathbf{v}^{\top}\mathbf{v}}{2}}\right\}\mathbb{I}_{[\|\mathbf{v}\|_{\infty}\leq1]},$$

(i) The cumulative distribution within the ℓ∞*-ball, i.e., the normalizing factor* Z(d) =
[Φ(1) − Φ(−1)]d.

(ii) The marginal distribution is a standard normal truncated within [−1, 1].

By Eq. (3) in (**Cartinhour**, **1990**), we know that  $$Z(d)=\int_{\|\mathbf{v}\|_{\infty}\leq1}\frac{1}{(2\pi)^{\frac{d}{2}}}\exp\left\{-\frac{\mathbf{v}^{\top}\mathbf{v}}{2}\right\}d\mathbf{v}$$ $$=\underbrace{\int_{-1}^{1}\cdots\int_{-1}^{1}}_{d\text{-time integration,one for each coordinate}}\frac{1}{(2\pi)^{\frac{d}{2}}}\exp\left\{-\frac{\sum_{i=1}^{d}v_{i}^{2}}{2}\right\}dv_{1}\cdots dv_{d}$$ $$=\prod_{i=1}^{d}\int_{-1}^{1}\frac{1}{\sqrt{2\pi}}\exp\left\{-\frac{v_{i}^{2}}{2}\right\}dv_{i}$$ $$=\left[\Phi(1)-\Phi(-1)\right]^{d}.$$
(ii) By Eq. (4) and (16) in (Cartinhour, 1990), ∀i ∈ [d],

[Φ(1) − Φ(−1)]d  √2π Z 1 −1 · · ·  Z 1 −1 | {z } d − 1-time integration exp − Pj̸=i v 2 j 2  P [Vi] = exp n− v 2 i 2 o (2π) d−1 2 dv1 · · · dvi−1dvi+1 · · · dvd = exp n− v 2 i 2 o [Φ(1) − Φ(−1)]d  √2π Y j̸=i Z 1 −1 1 √2π exp  ( − v 2 j 2 ) dvj = exp n− v 2 i 2 o [Φ(1) − Φ(−1)]d  √2π [Φ(1) − Φ(−1)]d−1 =1 [Φ(1) − Φ(−1)]  √2π exp  − v 2 i 2  =1 √2πR 1−1 √ 1 2π exp n− v 2 i 2 odvi exp − v 2 i 2 ,
if −1 ≤ Vi ≤ 1, otherwise P [Vi] = 0. Therefore, Vi follows the truncated standard normal distribution within [−1, 1].

Lemma 14 (Properties of Truncated Gaussian Smoothing). *For a function* f : R
d → R *that is* L-Lipschitz with respect to the ℓ2 *norm, then* ∀ x ∈ R
d,
(i) If f is convex and non-differentiable in a set with Lebesgue measure 0, then fρ *is continuously* differentiable and ∇fρ(x) = EV [∂f(x + ρV )] *for some random variable* V .

(ii) If f is convex, fρ is convex and L-Lipschitz with respect to the ℓ2 *norm.*
(iii) If f *is convex,* f(x) ≤ fρ(x) ≤ f(x) + 54 Lρ√d.

(iv) ∇fρ is 2ρ L*-Lipschitz, i.e.,* fρ is 2ρ L-smooth.

Proof. The proof of this lemma is based on that of Lemma 9 in (Duchi et al., 2012).

(i) The differentiability is established in (Bertsekas, 1973, Proposition 2.3), and ∇fρ(x) = EV [∂f(x+
ρV )] in (Bertsekas, 1973, Proposition 2.2). (ii) Expectation preserves convexity (Boyd & Vandenberghe, 2004, Section 3.2.1), therefore, given that f is convex, by definition, fρ is also convex. For Lipschitz continuity, by the second part of (i)
and Jensen's inequality, we have

$$\begin{array}{c}{{\|\nabla f_{\rho}(\mathbf{x})\|=\|\mathbb{E}_{V}[\partial f(\mathbf{x}+\rho V)]\|}}\\ {{\leq\mathbb{E}_{V}[\|\partial f(\mathbf{x}+\rho V)\|].}}\end{array}$$
Given that f is L-Lipschitz over R
d with respect to the ℓ2 norm, it is implied that ∀x ∈ R
d,
∥∂f(x)∥ ≤ L. As a result, ∥∇fρ(x)∥ ≤ E[L] ≤ L which further implies that fρ is L-Lipschitz with
respect to the ℓ2 norm.
(iii) For fρ(x) = EV [f(x + ρV )], EV [V ] = 0 by construction. And since smoothing preserves
convexity, fρ(x) is also convex. For the lower bound, using Jensen's inequality,
For the upper bound, since $f$ is $L$-Lipschitz in $\ell_{2}$-norm, $f(\mathbf{x}+\rho V)-f(\mathbf{x})\leq L\|\rho V\|$. Therefore,  $$f_{\rho}(\mathbf{x})=\mathbb{E}_{V}[f(\mathbf{x}+\rho V)]$$ $$\leq\mathbb{E}_{V}[f(\mathbf{x})+L\rho\|V\|]$$ $$=f(\mathbf{x})+L\rho\mathbb{E}\left[\sqrt{\sum_{i=1}^{d}V_{i}^{2}}\right]$$ $$\leq f(\mathbf{x})+L\rho\sqrt{\sum_{i=1}^{d}\mathbb{E}\left[V_{i}^{2}\right]}.$$
$$\begin{array}{r l}{f(\mathbf{x})=f(\mathbf{x}+\rho\mathbb{E}_{V}[V])}\\ {=f(\mathbb{E}_{V}[\mathbf{x}+\rho V])}\\ {\leq\mathbb{E}_{V}[f(\mathbf{x}+\rho V)]}\\ {=f_{\rho}(\mathbf{x}).}\end{array}$$

By Lemma 13 (ii), Vi follows the standard normal distribution truncated within [−1, 1]. Therefore, let Φ(·) denote the cumulative distribution function of standard normal distribution, then

$$\mathbb{E}\left[V_{i}^{2}\right]=\int_{-1}^{1}\frac{\phi(\tau)}{\Phi(1)-\Phi(-1)}\tau^{2}d\tau$$ $$=\frac{1}{\Phi(1)-\Phi(-1)}\int_{-1}^{1}\phi(\tau)\tau^{2}d\tau$$ $$\leq\frac{1}{\Phi(1)-\Phi(-1)}\int_{-\infty}^{1}\phi(\tau)\tau^{2}d\tau$$ $$=\frac{\mathbb{E}\left[U_{i}^{2}\right]}{\Phi(1)-\Phi(-1)}$$  for $U_{i}\sim\mathcal{N}(0,1)$, $\forall i\in[d]$. Then for $U=[U_{1},\cdots,U_{d}]^{\top}$, $U$ follows the standard MVN distribution and 
$$f_{\rho}(\mathbf{x})\leq f(\mathbf{x})+L\rho{\sqrt{\frac{\mathbb{E}\left[\sum_{i=1}^{d}U_{i}^{2}\right]}{\Phi(1)-\Phi(-1)}}}$$ $$=f(\mathbf{x})+L\rho{\sqrt{\frac{\mathbb{E}\left[\left\|U\right\|^{2}\right]}{\Phi(1)-\Phi(-1)}}}.$$

E
-∥U∥
2is the second moment of the standard MVN, which is bounded by the dimension d (Nesterov
& Spokoiny, 2017, Lemma 1). We know that Φ(1) − Φ(−1) ≈ 0.6827. As a result, we have

$$f_{\rho}(\mathbf{x})\leq f(\mathbf{x})+{\frac{5}{4}}L\rho{\sqrt{d}}.$$

(iv) The proof of this lemma follows that of Lemma 3.3 point 3 in (Lakshmanan & De Farias, 2008),
also seen in that of Lemma 9 (iii) in (Duchi et al., 2012). Denote the PDF of the unit ℓ∞-ball-truncated standard MVN as ϕ∥·∥∞≤1(·; 0, 1). Then for fρ(x) = EV [f(x+ρV )], ρV has PDF ϕ∥·∥∞≤ρ(·; 0, ρ2)
by Lemma 2 (v) in (Chen et al., 2020). By (Duchi et al., 2012, Lemma 11), ∀ x, x
′ ∈ R
d, for Z from ϕ∥·∥∞≤ρ(·; 0, ρ2),

$$\|\nabla f_{\rho}(\mathbf{x})-\nabla f_{\rho}(\mathbf{x}^{\prime})\|_{2}\leq L\underbrace{\int|\phi_{\|\cdot\|_{\infty}\leq\rho}(\mathbf{z}-\mathbf{x};0,\rho^{2})-\phi_{\|\cdot\|_{\infty}\leq\rho}(\mathbf{z}-\mathbf{x}^{\prime};0,\rho^{2})\big|\,d\mathbf{z}}_{\mathbf{z}}.$$
| {z }
I
Now we bound the integral. Note that ∀ x, ϕ∥·∥∞≤ρ(x; 0, ρ2) is a truncated MVN symmetrically centered at the origin, consequently, is strictly decreasing with respect to ∥x∥2 2. As a result, ϕ∥·∥∞≤ρ(z − x; 0, ρ2) ≥ ϕ∥·∥∞≤ρ(z − x
′; 0, ρ2) if and only if ∥z − x∥2 ≤ ∥z − x
′∥2. Therefore,

I = 2  Z ∥z−x∥2≤∥z−x′∥2 ϕ∥·∥∞≤ρ(z − x; 0, ρ2) − ϕ∥·∥∞≤ρ(z − x ′; 0, ρ2)dz = 2  Z ∥z−x∥2≤∥z−x′∥2 ϕ∥·∥∞≤ρ(z − x; 0, ρ2)dz − 2 Z ∥z−x∥2≤∥z−x′∥2 ϕ∥·∥∞≤ρ(z − x ′; 0, ρ2)dz. Denote y = z − x and y ′ = z − x ′, then
I = 2 
Z
∥y∥2≤∥y−(x′−x)∥2
ϕ∥·∥∞≤ρ(y; 0, ρ2)dy − 2
Z
∥y′∥2≥∥y′−(x−x′)∥2
ϕ∥·∥∞≤ρ(y
′; 0, ρ2)dy
′
= 2Pϕ∥·∥∞≤ρ[∥Z∥2 ≤ ∥Z − (x
′ − x)∥2] − 2Pϕ*∥·∥∞≤*ρ[∥Z
′∥2 ≥ ∥Z
′ − (x − x
′)∥2]
= 2Pϕ*∥·∥∞≤*ρ
-∥Z∥
22 ≤ ∥Z − (x
′ − x)∥
22
− 2Pϕ*∥·∥∞≤*ρ
-∥Z
′∥
22 ≥ ∥Z
′ − (x − x
′)∥
22

= 2Pϕ*∥·∥∞≤*ρ
-2 ⟨Z, x
′ − x*⟩ ≤ ∥*x
′ − x∥
2
2
− 2Pϕ*∥·∥∞≤*ρ
-2 ⟨Z
′, x − x
′*⟩ ≥ ∥*x − x
′∥
2
2

= 2Pϕ*∥·∥∞≤*ρ
Z, x
′ − x
∥x′ − x∥2
≤
∥x
′ − x∥2
2
− 2Pϕ*∥·∥∞≤*ρ
Z
′,x − x
′
∥x − x′∥2
≥
∥x − x
′∥2
2

Denote W =
DZ, x
′−x
∥x′−x∥2
Eand W′ =
DZ
′,x−x
′
∥x−x′∥2
E. Since x
′−x
∥x′−x∥2
and x−x
′
∥x−x′∥2
are normalized
vectors, W and W′follow the one-dimensional distribution projected onto a plane along some
direction from the truncated multivariate Gaussian, which is symmetrically centered at the origin.
Therefore, by symmetry,
Therefore, by symmetry,  $I=2\mathbb{P}\left[W\leq\dfrac{\|\mathbf{x}'-\mathbf{x}\|_2}{2}\right]-2\mathbb{P}\left[W'\geq\dfrac{\|\mathbf{x}-\mathbf{x}'\|_2}{2}\right]$  $=2\mathbb{P}\left[W\leq-\dfrac{\|\mathbf{x}'-\mathbf{x}\|_2}{2}\right]+2\mathbb{P}\left[-\dfrac{\|\mathbf{x}'-\mathbf{x}\|_2}{2}\leq W\leq\dfrac{\|\mathbf{x}'-\mathbf{x}\|_2}{2}\right]-2\mathbb{P}\left[W'\geq\dfrac{\|\mathbf{x}-\mathbf{x}'\|_2}{2}\right]$ $=2\mathbb{P}\left[-\dfrac{\|\mathbf{x}'-\mathbf{x}\|_2}{2}\leq W\leq\dfrac{\|\mathbf{x}'-\mathbf{x}\|_2}{2}\right]$
As we later upper bound the integration by the peak of this distribution, we know by the geometry of ℓ∞-ball that the projection onto the diagonal yields the highest peak, and that is when W =
√
1 d Pd i=1 Zi for Zi being the marginal of Z that follows the truncated Gaussian distribution on
[−*ρ, ρ*] by Lemma 13 (ii). And further by Lemma 2 (v) in (Chen et al., 2020), √
Zi d is also a truncated Gaussian whose PDF is ϕ[− √ρd
, √ρd
](w; 0, ρ 2 d
). As a result, W is the sum of independent identically distributed (i.i.d.) truncated Gaussian variables. By Theorem 3 in (Chen et al., 2020) and E.q. (4.2) in (Birnbaum & Andrews, 1949) we know the sum of truncated Gaussian variables converges to a normal distribution for large d. As a result, W ∼
Pd i=1 Var [Zi]
N (0, 1). Knowing from the CDF of truncated Gaussian that ∀ i ∈ [d], Var [Zi] = σ 2 d 1 −
ϕ(1)+ϕ(−1)
Φ(1)−Φ(−1) −
ϕ(1)−ϕ(−1)
Φ(1)−Φ(−1)2= 0.7089 ρ 2 d
,
we have Pd i=1 Var [Zi]
= 0.7089ρ 2

$$\mathbb{P}\left[-\frac{\|\mathbf{x}^{\prime}-\mathbf{x}\|_{2}}{2}\leq W\leq\frac{\|\mathbf{x}^{\prime}-\mathbf{x}\|_{2}}{2}\right]=\frac{1}{\sqrt{2\pi}\sqrt{0.7089}\rho}\int_{-\frac{\|\mathbf{x}^{\prime}-\mathbf{x}\|_{2}}{2}}^{\frac{\|\mathbf{x}^{\prime}-\mathbf{x}\|_{2}}{2}}\exp\{-\frac{w^{2}}{2\times0.7089\rho^{2}}\}dw\,$$  Furthermore, since the PDF takes its peak at $w=0$, we have 

Furthermore, since the PDF takes its peak at w = 0, we have

 So far peak at $\omega=0$, we have  $ I\leq2\times\frac{2}{\sqrt{2\pi}\rho}\int_{-\frac{\|\mathbf{x}^{\prime}-\mathbf{x}\|_{2}}{2}}^{\frac{\|\mathbf{x}^{\prime}-\mathbf{x}\|_{2}}{2}}dw$  $ =\frac{4\|\mathbf{x}^{\prime}-\mathbf{x}\|_{2}}{\sqrt{2\pi}\rho}$  ... 
Therefore,

$$\begin{array}{c}{{\|\nabla f_{\rho}(\mathbf{x})-\nabla f_{\rho}(\mathbf{x}^{\prime})\|_{2}\leq L I}}\\ {{\leq\frac{2L}{\rho}\|\mathbf{x}^{\prime}-\mathbf{x}\|_{2}.}}\end{array}$$

Lemma 1. Given a L-Lipschitz function f*, the function* f p ρ = Sρ[· · · [Sρ[f]] · · · ] *satisfies*
(i) If f *is convex,* f p ρis convex and L-Lipschitz with respect to the ℓ2 *norm.*
(ii) If f *is convex,* f(x) ≤ f p ρ
(x) ≤ f(x) + 5p 4 Lρ√d.

(iii) ∀i ∈ [p], ∀x, x
′ ∈ R
d, ∥∇if p ρ(x) − ∇if p ρ(x
′)∥ ≤ 2ρ iL∥x − x
′∥.

Proof. The proof of this lemma relies on inductively applying Lemma 14 and we provide formal proof by induction.

(i) The base case p = 1 holds directly by Lemma 14 (ii). Then we state the hypothesis that for p = k, f k ρis convex and L-Lipschitz with respect to the ℓ2 norm. For the induction step, we have, by definition, f k+1 ρ = Sρ[f k ρ] where f k ρis convex and L-Lipschitz with respect to the ℓ2 norm by our hypothesis, with which f k ρsatisfies the condition of Lemma 14. Then by Lemma 14 (ii), f k+1 ρis convex and L-Lipschitz with respect to the ℓ2 norm.

(ii) The base case p = 1 holds directly by Lemma 14 (iii). Then we state the hypothesis that for p = k, f(x) ≤ f k ρ(x) ≤ f(x) + 5k 4 Lρ√d holds. From the result of (i), we know that f k ρsatisfies the condition of Lemma 14. Therefore, applying 14 (iii) to the function f k ρ
(x) we have for the lower bound f k+1 ρ(x) ≥ f k ρ(x) ≥ f(x)
and for the lower bound

$$f_{\rho}^{k+1}({\bf x})\leq f_{\rho}^{k}({\bf x})+\frac{5}{4}L\rho\sqrt{d}\leq f({\bf x})+\frac{5k}{4}L\rho\sqrt{d}+\frac{5}{4}L\rho\sqrt{d}=f({\bf x})+\frac{5(k+1)}{4}L\rho\sqrt{d}$$
which completes the induction step.
(iii) The base case p = 1 holds for i = 0 by Lemma 14 (ii) and for i = 1 by Lemma 14 (iv). Now we
state the inductive hypothesis that for p = k, it holds that ∀ x, x
′ ∈ R,
$$\forall\,i\in[k],\ \|\nabla^{i}f_{\rho}^{k}(\mathbf{x})-\nabla^{i}f_{\rho}^{k}(\mathbf{x}^{\prime})\|\leq\left({\frac{2}{\rho}}\right)^{i}L\|\mathbf{x}-\mathbf{x}^{\prime}\|.$$
That is, ∀ i ∈ [k], the function ∇if k ρis 2ρ iL -Lipschitz. Then for p = k + 1, ∀i ∈ [k + 1], ∥∇if k+1 ρ(x) − ∇if k+1 ρ(x ′)∥ = ∥∇iSρ[f k ρ](x) − ∇iSρ[f k ρ](x ′)∥ = ∥Sρ[∇if k ρ](x) − Sρ[∇if k ρ](x ′)∥ = ∥EV [∇if k ρ(x + ρV )] − EV [∇if k ρ(x ′ + ρV )]∥ = ∥EV [∇if k ρ (x + ρV ) − ∇if k ρ (x ′ + ρV )]∥ ≤ EV [∥∇if k ρ(x + ρV ) − ∇if k ρ(x ′ + ρV )∥] where the first equality holds by definition, the second equality by the fact that expectation and
derivative commute for differentiable functions, and the last inequality by the Jensen's. For *i < k* + 1, we can directly apply Lemma 14 (iv), with the hypothesis as the condition, on the function ∇if k ρ, to establish the result that ∇if k ρis smooth with parameter 2ρ iL. Therefore,

$$\|\nabla^{i}f_{\rho}^{k+1}(\mathbf{x})-\nabla^{i}f_{\rho}^{k+1}(\mathbf{x}^{\prime})\|\leq\mathbb{E}_{V}[\|\nabla^{i}f_{\rho}^{k}(\mathbf{x}+\rho V)-\nabla^{i}f_{\rho}^{k}(\mathbf{x}^{\prime}+\rho V)\|]$$ $$\leq\mathbb{E}_{V}\left[\left(\frac{2}{\rho}\right)^{i}L\|\mathbf{x}-\mathbf{x}^{\prime}\|\right]$$ $$=\left(\frac{2}{\rho}\right)^{i}L\|\mathbf{x}-\mathbf{x}^{\prime}\|$$
$$\square$$

For i = k + 1, we have from our *i < k* + 1 case that the function ∇kf k+1 ρis 2ρ kL
-Lipschitz.

We can therefore apply Lemma 14 (iv) on ∇kf k+1 ρand claim that it's also smooth with parameter 2 ρ
·
2 ρ kL =
2 ρ k+1L. That is,

$$\left\|\nabla\left[\nabla^{k}f_{\rho}^{k+1}\right](\mathbf{x})-\nabla\left[\nabla^{k}f_{\rho}^{k+1}\right](\mathbf{x}^{\prime})\right\|\leq\left(\frac{2}{\rho}\right)^{k+1}L\|\mathbf{x}-\mathbf{x}^{\prime}\|,$$  is the proof. 
which completes the proof. A.2 PROPERTIES OF THE CONSTRUCTED HARD FUNCTION
Lemma 2. ∀ t ∈ [T], gt is convex and 1-Lipschitz with respect to the ℓ∞-norm, and also the ℓ2*-norm.*
Proof. (1) For convexity, by definition we have

$r_{k}(\mathbf{x})=\max_{1\leq k\leq t}r_{k}(\mathbf{x})\qquad where\qquad\forall\ k\in[T],r_{k}(\mathbf{x})=\xi_{k}\left\langle\mathbf{e}_{\alpha(k)},\,\mathbf{x}\right\rangle-(k-1)\delta,$
Since rk(x) is linear in x, rk(x) is convex. Then gt(x) is the maximum of convex functions which is also convex.

(2) To show Lipschitzness, ∀ x, y ∈ R
d, without the loss of generality, denote

  **Lemma, $\tau_{k,y}\in\mathbb{R}^{n}$, where $\tau_{k}$ is a graph, and $k_{1}=\arg\max\limits_{1\leq k\leq t}r_{k}(\mathbf{x})$**  $$k_{2}=\arg\max\limits_{1\leq k\leq t}r_{k}(\mathbf{y}).$$
Therefore,

$$g_{t}\left(\mathbf{x}\right)=\xi_{k_{1}}x_{\alpha(k_{1})}-(k_{1}-1)\delta\qquad\qquad g_{t}\left(\mathbf{y}\right)=\xi_{k_{2}}y_{\alpha(k_{2})}-(k_{2}-1)\delta.$$

Since

$g_{t}\left(\mathbf{y}\right)=\xi_{k_{2}}y_{\alpha(k_{2})}-(k_{2}-1)\delta$  $=\max\limits_{1\leq k\leq t}\xi_{k}\left\langle\mathbf{e}_{\alpha(k)},\,\mathbf{x}\right\rangle-(k-1)\delta$  $\geq\xi_{k_{1}}y_{\alpha(k_{1})}-(k_{1}-1)\delta$,
we have

we have  $$g_{t}\left(\mathbf{x}\right)-g_{t}\left(\mathbf{y}\right)\leq\left(\xi_{k_{1}}x_{\alpha(k_{1})}-(k_{1}-1)\delta\right)-\left(\xi_{k_{1}}y_{\alpha(k_{1})}-(k_{1}-1)\delta\right)$$ $$\leq|x_{\alpha(k_{1})}-y_{\alpha(k_{1})}|$$ $$\leq\max_{1\leq i\leq d}|x_{i}-y_{i}|$$ $$=\left\|\mathbf{x}-\mathbf{y}\right\|_{\infty}$$ $$\leq\left\|\mathbf{x}-\mathbf{y}\right\|_{2},$$  where the last two inequalities show Lipschitzness in $\ell_{\infty}$ and $\ell_{2}$ norm respectively.  
$$\square$$
Lemma 3. ∀ t ∈ [T], ∀ x, y ∈ R
d,
(i) Gt(x) is convex and 1-Lipschitz, i.e., Gt(x) − Gt(y) ≤ ∥x − y∥.

(ii) gt(x) ≤ Gt(x) ≤ gt(x) + 54 pρ√d.

(iii) *For some fixed* p ∈ Z
+, ∀ i ∈ [p], ∥∇iGt(x) − ∇iGt(y)∥ ≤ 2ρ

$$:)-\nabla^{i}G_{t}(\mathbf{y})\|\leq\left({\frac{2}{\rho}}\right)^{i}\|\mathbf{x}-\mathbf{y}\|.$$

Proof. The proof follows directly from that for Lemma 1.

Lemma 4. For F(x) = fT (x) + dq(x) *where* dq(x) = σ q x qand x ∈ Q,
(i) F is uniformly convex function with degree q and modulus σ > 0.

(ii) F(x) is p th*-order Holder smooth with parameter* ¨ H =
2 p+1β ρp+ν−1 , ∀ p ∈ Z
+.

Proof. (i) It is shown in (Nesterov et al., 2018, Section 4.2.2) that σq x qis uniformly convex with de-

gree q and parameter σ. By Lemma 3 (i), GT is convex, therefore f is also convex, so that ∀ x, y ∈ R,
⟨∇f(x) − ∇f(y), x − y⟩ ≥ 0. Therefore, by Definition 3,
D∇(
σ
q
x
q) − ∇(
σ
q
y
q), x − y
E≥
σ∥x − y∥
q. Adding them together we get ⟨∇F(x) − ∇F(y), x − y⟩ ≥ σ∥x − y∥
q, which shows
that F(x) is uniformly convex function with degree q and modulus σ > 0. (ii) From Lemma 3 (iii) and Definition 1, we know that f is p
th-order smooth with parameter
Lp = β 2ρ +, i.e., ∀ x, y ∈ Q ⊂ R d, ∥∇pf(x) − ∇pf(y)∥ ≤ β 2 ρ p∥x − y∥. Also, ∇p−1f is β 2 ρ p−1-Lipschitz, which implies that ∀ x ∈ R d, ∥∇pf(x)∥ ≤ β 2 ρ p−1. Then we have ∀ x, y ∈ Q, ∥∇pf(x) − ∇pf(y)∥ = ∥∇pf(x) − ∇pf(y)∥ ν∥∇pf(x) − ∇pf(y)∥ 1−ν ≤ ∥∇pf(x) − ∇pf(y)∥ ν(∥∇pf(x)∥ + ∥∇pf(y)∥) 1−ν ≤ 2 ρ pνβ ν∥x − y∥ ν   2β 2 ρ p−1!1−ν =2 pβ ρ p+ν−1 ∥x − y∥ ν. By letting H = 2 p+1β th-order Holder smooth with parameter ¨ H .
p, ∀ p ∈ Z
$\square$
ρp+ν−1 , we can conclude that f is p 2 Furthermore, for dq(x), by definition, Q = {x : ∥x∥2 ≤ D} for D ≤ H
2 1−νC
 1 q−p−νand C =
σ(q − 1) *× · · · ×* (q − p). As a result,
∥∇p+1dq(x)∥ = σ(q − 1) *× · · · ×* (q − p)∥x∥
q−p−1 ≤ C · Dq−p−1.

This indicates that dq(x) is p th-order smooth with parameter C · Dq−p−1, which is equivalent to
∀ x, y ∈ Q,

$$\|\nabla^{p}d_{q}({\bf x})-\nabla^{p}d_{q}({\bf y})\|\leq C\cdot D^{q-p-1}\|{\bf x}-{\bf y}\|.$$
Given that ∥x − y∥ = ∥x − y∥
1−ν∥x − y∥
ν ≤ (∥x∥ + ∥y∥)
1−ν∥x − y∥
ν ≤ (2D)
1−ν∥x − y∥
ν,
we have
$\|\mathbf{x}\quad\mathbf{y}\|=\|\mathbf{x}\quad\mathbf{y}\|\quad\|\mathbf{x}\quad\mathbf{y}\|\quad\leq(\|\mathbf{x}\|+\|\mathbf{y}\|)\quad\|\mathbf{x}\quad\mathbf{y}\|\quad\leq(\mathbf{z}\mathbf{z})$  $$\|\nabla^{p}d_{q}(\mathbf{x})-\nabla^{p}d_{q}(\mathbf{y})\|\leq2^{1-\nu}C\cdot D^{q-p-\nu}\|\mathbf{x}-\mathbf{y}\|^{\nu}\leq\frac{H}{2}\|\mathbf{x}-\mathbf{y}\|^{\nu}.$$  $\mathbf{x}$ is $\quad\mathbf{y}$-independent.  
That is, dq(x) is p th-order Holder smooth with parameter ¨
2 on domain Q. Since f is also p th-order Holder smooth with parameter ¨
H
2
, we conclude that F = f + dq is p th-order Holder smooth with ¨
parameter H on domain Q.

Lemma 5. For R(x) = β maxk∈[T] ξkeα(k), x+
$=\beta\max_{k\in[T]}\xi_{k}\left\langle\mathbf{e}_{\alpha(k)},\ \mathbf{x}\right\rangle+\frac{\sigma}{q}\|\mathbf{x}\|^{q},$ we have_  $R(\mathbf{x})-\beta(T-1)\delta\leq F(\mathbf{x})\leq R(\mathbf{x})+\frac{5}{4}p\beta\rho\sqrt{d}.$
Proof. Since F(x) is constructed with softmax smoothing, we are now able to characterize it with
the properties in Lemma 3. F(x) can be upper bounded using the second inequality of Lemma 3 (ii): F(x) = βGT (x) +  σ q ∥x∥ q ≤ βgT (x) +  5 4 pβρ√d + σ q ∥x∥ q = β max k∈[T] ξkeα(k), x− (k − 1)δ	+ 5 4 pβρ√d + σ q ∥x∥ q ≤ β max k∈[T] ξkeα(k), x+ 5 4 pβρ√d + σ q ∥x∥ q. F(x) can be lower bounded using the first inequality of Lemma 3 (ii): F(x) = βGT (x) +  σ q ∥x∥ q ≥ βgT (x) +  σ q ∥x∥ q = β max k∈[T] ξk eα(k), x− (k − 1)δ	+ σ q ∥x∥ q ≥ β max k∈[T] ξkeα(k), x− (T − 1)δ + σ q ∥x∥ q.
$$\square$$
  **Lemma 7**.: $F({\bf x}_{T})-F({\bf x}^{*})\geq-\beta(T-1)\delta-\frac{5}{4}p\beta\rho\sqrt{d}+\frac{q-1}{q}\left(\frac{\beta^{q}}{\sigma T^{\frac{2}{2}}}\right)^{\frac{1}{q-1}}$.  
Proof.
$$F(\mathbf{x}^{*})=\operatorname*{min}_{\mathbf{x}}F(\mathbf{x})$$
$$\leq\min_{\mathbf{x}}R(\mathbf{x})+\frac{5}{4}p\beta\rho\sqrt{d}$$ $$=\min_{\mathbf{x}}\left\{\beta\max_{k\in[T]}\xi_{k}\left\langle\mathbf{e}_{\alpha(k)},\,\mathbf{x}\right\rangle+\frac{\sigma}{q}\|\mathbf{x}\|^{q}\right\}+\frac{5}{4}p\beta\rho\sqrt{d}.$$  Define $\gamma=\left|\max_{k\in[T]}\xi_{k}\left\langle\mathbf{e}_{\alpha(k)},\,\mathbf{x}\right\rangle\right|$. Then by symmetry (**Dokov**, **2022**),  $$\|\mathbf{x}\|^{q}=T^{\frac{q}{2}}\gamma^{q}.$$
As a result,
$$\min_{\mathbf{x}}R(\mathbf{x})=\min_{\mathbf{x}}\left\{\beta\max_{k\in[T]}\xi_{k}\left\langle\mathbf{e}_{\alpha(k)},\,\mathbf{x}\right\rangle+\frac{\sigma}{q}\|\mathbf{x}\|^{q}\right\}$$ $$=\min_{\gamma>0}\{-\beta\gamma+\frac{\sigma}{q}T^{\frac{q}{2}}\gamma^{q}\}$$ $$=-\frac{q-1}{q}\left(\frac{\beta^{q}}{\sigma T^{\frac{q}{2}}}\right)^{\frac{1}{q-1}}.$$
Therefore,

$$F(\mathbf{x}^{*})\leq-{\frac{q-1}{q}}\left({\frac{\beta^{q}}{\sigma T^{\frac{q}{2}}}}\right)^{\frac{1}{q-1}}+{\frac{5}{4}}p\beta\rho{\sqrt{d}}.$$

Furthermore, for some xT generated following some algorithm A along some trajectory, by definition,

$$\begin{array}{r l}{g_{T}(\mathbf{x}_{T})\geq\left|\left\langle e_{\alpha(T)},\,\mathbf{x}_{T}\right\rangle\right|-(T-1)\delta}\\ {\geq-(T-1)\delta.}\end{array}$$

Therefore,

$$F(\mathbf{x}_{T})=f(\mathbf{x}_{T})+{\frac{\sigma}{q}}\|\mathbf{x}_{T}\|^{q}$$
$\geq f(\mathbf{x}_{T})$  $=\beta G_{T}(\mathbf{x}_{T})$  $\geq\beta g_{T}(\mathbf{x}_{T})$  $\geq-\beta(T-1)\delta$.  
Given the upper bound on F(x
∗), we have

$$F({\bf x}_{T})-F({\bf x}^{*})\geq-\beta(T-1)\delta-\frac{5}{4}p\beta\rho\sqrt{d}+\frac{q-1}{q}\left(\frac{\beta^{q}}{\sigma T^{\frac{q}{2}}}\right)^{\frac{1}{q-1}}.$$

## B Proof For Technical Lemmas In Section 5

Lemma 8. x
∗ = arg minxf(x), y = arg minx
˜f(x)*. (i)* ∀i ∈ [T˜], ⟨vi, x
∗⟩ = yi*. (ii)* ∥x
∗∥ = ∥y∥.

Proof. (i) By definition, f is a scaling and rotation of ˜f. Since v1, *· · ·* , vT˜, we can write for V = [v1, *· · ·* , vT˜], f(x) = H
2 p+ν+1(p+ν−1)!

˜f(V x). Therefore,

$$\square$$
 -  $\mathbf{y}=\operatorname*{arg\,min}_{\mathbf{x}}\tilde{f}(\mathbf{x})$  $=V\operatorname*{arg\,min}_{\mathbf{x}}\tilde{f}(V\mathbf{x})$  $=V\operatorname*{arg\,min}_{\mathbf{x}}f(\mathbf{x})$  $=V\mathbf{x^*}$. 
(ii) This can be shown in the same way as (Arjevani et al., 2019, Lemma 6). Lemma 9. f(x) is (i) uniformly convex with degree q and parameter σ*. (ii)* p th-order Holder smooth ¨
with degree ν and parameter H.

Proof. (i) The proof is similar to that for Lemma 4 (i).

(ii) Without the loss of generality, let the basis that defines f be the standard basis. ∀ i ∈ [T˜], denote eithe i th vector in the standard basis. Denote function g(x) = 1 p+ν |x| p+ν. g
(p)(x), the p th-order derivative of g(x) is (p + ν − 1)!x νif p is odd, (p + ν − 1)!|x| νis even. Let di = ei − ei+1, then

$$f(\mathbf{x})={\frac{H}{2^{p+\nu+1}(p+\nu-1)!}}\left({\frac{1}{p+\nu}}\sum_{i=1}^{\tilde{T}}g(\langle\mathbf{d}_{i},\,\mathbf{x}\rangle)-\gamma x_{1}\right)+{\frac{\sigma}{q}}\|\mathbf{x}\|^{q}$$