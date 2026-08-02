# 

Ilias Diakonikolas 1 Daniel M. Kane 2 **Sushrut Karmalkar** 3 4 Jasper C.H. Lee 5 **Thanasis Pittas** 1

## Abstract

We study the complexity of learning k-mixtures of Gaussians (k-GMMs) on R
d. This task is known to have complexity d Ω(k)in full generality.

To circumvent this exponential lower bound on the number of components, research has focused on learning families of GMMs satisfying additional structural properties. A natural assumption posits that the component weights are not exponentially small and that the components have the same unknown covariance. Recent work gave a d O(log(1/wmin))-time algorithm for this class of GMMs, where wmin is the minimum weight. Our first main result is a Statistical Query (SQ) lower bound showing that this quasi-polynomial upper bound is essentially best possible, even for the special case of uniform weights. Specifically, we show that it is SQ-hard to distinguish between such a mixture and the standard Gaussian. We further explore how the distribution of weights affects the complexity of this task. Our second main result is a quasi-polynomial upper bound for the aforementioned testing task when most of the weights are uniform while a small fraction of the weights are potentially arbitrary.

## 1. Introduction

Learning mixture models in high dimensions is a classic and fundamental task with applications in a plethora of domains, such as bioinformatics, astrophysics, and marketing (Lindsay, 1995; Garc´ıa-Escudero et al., 2010); see Titterington et al. (1985) for an extensive list of applications. The prototypical case is that of Gaussian Mixture Models (GMMs)
which is one of the most studied problems in statistics and machine learning, with a large body of research over the past few decades, e.g., Vempala & Wang (2002); Kannan et al.

(2005); Achlioptas & McSherry (2005) - see Appendix A
for a detailed literature review.

The setup is as follows: the learning algorithm observes i.i.d. samples from a k-component GMM model (k-GMM)
in R
d, P =Pk i=1 wiN (µi, Σi), and the goal is to either learn the mixture in total variation distance, learn its parameters, or cluster samples from the GMM correctly. The first task is known to be information-theoretically feasible with poly(d, k) samples, as are the second and third, provided the components are sufficiently well-separated, however known algorithms often require more. While the particular case of spherical mixtures (i.e., Σi = I) can be learned in poly(*d, k*) time and samples, (Liu & Li, 2022; Diakonikolas & Kane, 2024), the best-known algorithms for learning arbitrary GMMs (i.e. with arbitrary weights, and arbitrary and different component covariances) have sample complexity that scales with d O(k)(Bakshi et al., 2022). In this paper, we are concerned with an intermediate regime between the two extremes, where the components share an unknown but common covariance matrix. Diakonikolas et al. (2017) showed that for such mixtures, any sub-exponential time algorithm in the *Statistical Query* (SQ) model requires a sample complexity of d Ω(k). The SQ model consists of algorithms that, instead of drawing samples from the data distribution, make queries to approximate expectations of bounded functions (formally defined in Definition 1.2). The hard instances they proposed are "parallel pancakes" GMMs—mixtures of pairwise-separated Gaussians whose component means are collinear along an unknown direction v, with arbitrary variance in the v-direction and identity covariance in the orthogonal subspace. This will be formally defined in Problem 1.1. Bruna et al. (2021); Gupte et al. (2022) further extended the hardness result to general algorithms but under cryptographic assumptions; similar hardness results were also shown for sum-of-squares algorithms (Diakonikolas et al., 2024). While these results together might suggest that k-GMM learning is fully understood algorithmically, the current theory remains unsatisfactory, in the following sense: the 1 hard instances developed in Diakonikolas et al. (2017) have rather ill-conditioned mixing weights—some of the mixing weights are 1/ poly(k), but others can be as small as 2
−k.

A natural question then is: is it possible to improve the complexity of learning algorithms when all weights are more naturally conditioned, i.e., wi ≥ 1/ poly(k) for all i?

This question was considered in Buhai & Steurer (2023); Anderson et al. (2024), which study GMMs that have a minimum mixing weight wmin ≥ 1/ poly(k) and unknown but common covariance across components. Under the assumption that the mixture components are separated in total variation distance, they provide an algorithm that can correctly cluster 99% of the points, using time and sample complexity d log(1/wmin) ≤ d O(log k). In particular, their results apply to parallel pancake instances, showing that it is possible to circumvent the d Ω(k)(SQ) lower bound under mixing weight assumptions. These prior results on learning mixtures with restricted mixing weights serve as motivation and the starting point of the present work. In particular, the first question we study is:
Is it possible to substantially improve the algorithm of Anderson et al. *(2024) to a* poly(d, k) time algorithm for parallel pancakes when each wi ≥ 1/ poly(k)?

Our first main result rules out this possibility for SQ
algorithms. Specifically, we show in Theorem 1.3 that even when the mixing weights are uniform, any SQ algorithm for such instances requires d Ω(log k)complexity. In fact, the lower bound holds even for the more basic task of distinguishing between a k-GMM from that family and N (0, I). Our second question stems from the fact that the algorithm in Anderson et al. (2024) has complexity d log(1/wmin),
meaning that a single point with arbitrarily small weight (e.g., 2
−k) can result in d kcomplexity.

What is the correct complexity dependence on wmin for learning k*-component parallel pancakes?*
Specifically, we consider again the testing problem of distinguishing between a k-parallel-pancake GMM and N (0, I), but where k
′ ≤ k components can have arbitrary weights while the remaining k − k
′ points must have uniform weights. We show that this mixing weight restriction implies that the testing problem can be solved with time and sample (kd)
O(k
′+log k) + (log k)/wmin - an inverse-linear dependence on wmin instead of quasi-polynomial as suggested by the Anderson et al. (2024) result. While this testing upper bound does not imply a general learning algorithm k-GMM, it serves as a first step in understanding the nuances of the computational landscape of GMMs with respect to the assumptions on the mixing weights.

The technical core for both our main results is to determine the maximum number m of moments that k-parallelpancake GMMs can match with N (0, I). Our SQ lower bound (Theorem 1.3) comes from showing that m = Ω(log k), by employing a result from design theory. Our second, algorithmic result (Theorem 1.4) critically builds on an impossibility-of-moment-matching argument (Proposition 4.1), showing that if there are k
′ ≤ k arbitrary weights in the k-GMM, then m must be O(log(k) + k
′). We show this through a novel proof strategy that bounds the ratio of expectations of appropriately chosen non-negative polynomials that vanish on the points with arbitrary weights.

## 1.1. Our Results

We first formally state the hypothesis testing problem which requires the algorithm to distinguish between a k-parallel pancake and the standard Gaussian N (0, I).

Problem 1.1 (Parallel Pancakes Testing Problem). One has (i.i.d. sample or SQ) access to a distribution D where either:
- (Null Hypothesis) D = N (0, I).

- (Alternative Hypothesis) D is a Gaussian mixture of the form Pi∈[k] wiN (vµi, I − δvv⊤), for some unit vector v ∈ Sd−1, centers µi ∈ R, and weights wi ≥ 0 for i ∈ [k], with Pi∈[k] wi = 1. That is, D is a k-GMM with collinear centers and variance 1 − δ along the direction of the centers and 1 in every orthogonal direction.

The goal is to distinguish between the two cases. Before presenting our first main result, we recall the definition of SQ algorithms. These algorithms, instead of directly accessing samples, query expectations of bounded functions of the distribution. The SQ model, introduced in (Kearns, 1998), has since been extensively studied in various contexts (Feldman, 2016). Many supervised learning algorithms, and several known machine learning techniques are implementable using SQs (Feldman et al., 2017a;b).

Definition 1.2 (STAT Oracle). Let D be a distribution on R
d. A statistical query is a bounded function f : R
d→[−1, 1]. Given f and an accuracy parameter τ > 0, STAT(τ ) returns a v ∈ R such that |v − Ex∼D[f(x)]| ≤ τ .

Since a call to STAT(τ ) can be simulated in the standard PAC model by averaging 1/τ 2samples, τ serves as the SQ model's analog to sample complexity. An informationcomputation tradeoff in the SQ model states that any SQ algorithm for a given problem must either make a large number of queries or at least one query with very fine accuracy (which informally implies a tradeoff between sample complexity and runtime in the standard PAC model). We are now ready to state our first main result. Theorem 1.3 (SQ Lower Bound for Uniform Weights). Let C be a sufficiently large absolute constant, k > C and d ≥ (log k log d)
2 be integers. If we further restrict the alternative hypothesis in Problem 1.1 *to have* wi = 1/k for all i ∈ [k]*, any SQ algorithm requires either* 2 d Ω(1)queries or at least one query of accuracy d
−Ω(log k).

Remarks Buhai & Steurer (2023); Anderson et al. (2024)
presented an algorithm for solving Problem 1.1 using d O(log k)time and samples (e.g., Theorem 1.1 in the first paper, which was the first to achieve this). Our Theorem 1.3 shows that this complexity is best possible. Notably their work requires the components to be statistically separated, but this is something that we can also ensure by taking δ sufficiently small (since δ does not affect the complexity lower bound). We now move to our second main result.

Theorem 1.4 (Testing Algorithm for Parallel Pancakes).

Consider the version of the parallel pancakes hypothesis testing problem (Problem *1.1), where* k
′ ≤ k *of the weights* wiin the Gaussian mixture are unconstrained and the remaining k − k
′ *are assumed to be equal to each other.*
There is an algorithm for that problem which draws n = O
(*kd/δ*)
O(k
′+log(k)) + log(k)/wminsamples (where δ is as in Problem 1.1 and wmin = mini∈[k] wi*is the smallest* weight), has runtime polynomial in n, d, and it outputs the correct hypothesis with probability at least 0.99. The algorithm is based on estimating the first O(k
′ + log k)
moment tensors through the empirical tensors, and thus it is also naturally expressible in the SQ model.

Remarks A single component with arbitrarily small weight can make the complexity in (Buhai & Steurer, 2023; Anderson et al., 2024) blow up quasipolynomially. By contrast, our algorithm can handle any number of such points, and the complexity interpolates smoothly between the alluniform and the fully general weights cases.

## 1.2. Overview Of Techniques

For Theorem 1.3, it suffices to show existence of a onedimensional distribution (corresponding to the projection along the hidden direction v in the parallel pancakes mixture in Problem 1.1) that matches a lot of moments with N (0, 1) and is thus hard to distinguish. Concretely, the goal is to show the existence of a set S ⊂ R of size k such that Ex∼S[x i] = Ex∼N(0,1)[x i] for all i = 1*, . . . , t*, where t = Ω(log k) and x ∼ S denotes the uniform distribution on S. Once established, the theorem follows from standard SQ theory: convolving this discrete distribution with a narrow Gaussian yields a k-GMM B that still matches the first t moments with N (0, 1). A standard result from (Diakonikolas et al., 2023) then shows that hiding B along an unknown direction is hard to distinguish from N (0, I). Fortunately, the desired moment-matching construction, known as a t-design, has been well-studied. Kane (2015) shows that designs of small size to match the moments of a distribution Q exist when the support of Q is "pathconnected". The design's size is upper bounded by the number K, which is defined to be the supremum of the ratio supx∈X p(x)
| infx∈X p(x)| taken over all degree-t zero-mean polynomials p. Thus, it suffices to show K = 2O(t)to prove Theorem 1.3. However, since the Gaussian distribution has unbounded support, there are (many) polynomials p where supx∈R p(x) is infinite while the infimum is clearly finite.

To address this, we can instead consider another distribution Q supported on an interval I of length O(
√t), which also matches the first t moments with N (0, 1) (Lemma 3.3 from Diakonikolas et al. (2017)). Thus, by creating a design to match Q we only need to bound K with X = I, which is now possible (Lemma 3.5). Specifically, by expressing p in the Hermite basis, we can show that supx∈I p(x) ≤ Ex∼N(0,1)[|p(x)|]2O(t). Additionally, by Gaussian anti-concentration, we can show that any zero-mean polynomial p of degree t has a 2
−O(t) probability of being less than −2
−O(t) Ex∼N(0,1)[|p(x)|]. This shows that |infx∈I p(x)| ≥ Ex∼N(0,1)[|p(x)|]2−O(t).

We can also show that this bound is tight: if A is the uniform distribution on k points, it cannot match more than O(log k) moments with N (0, 1). The argument is that for any non-negative function f, Ex∼A[f 2(x)]
Ex∼A[f(x)]2 ≤ k.

If A matches the first 4t moments with N (0, 1), setting f(x) = x 2t makes the ratio 2 Ω(t), implying t = O(log k).

In fact, we can extend the result to non-uniform distributions where all but k
′ points in the support have equal weight, showing that such distributions cannot match more than O(log(k)+k
′) moments with N (0, 1) (Proposition 4.1). As we will explain later, this will lead to the testing algorithm in Theorem 1.4. The first step towards Proposition 4.1 is to show that, if all but k
′ points have weight at least w0, then it is impossible to match more than O(log(1/w0)+k
′) moments with N (0, 1)
(Proposition 4.2). The proof relies on extending the idea of the previous paragraph that any non-negative function f which vanishes (i.e. gives value zero) on the k
′ points in question satisfies Ex∼A[f 2(x)]
Ex∼A[f(x)]2 ≤ 1/w20
. We specifically use f(x) = x tp(x), where p(x) = (x − µ1)
2*. . .*(x − µk′ )
2 and µ1*, . . . , µ*k′ are the points in the support of A with unrestricted weights. The goal then is to show that the ratio r :=Ex∼A[f 2(x)]
Ex∼A[f(x)]2 is at least 2 Ω(t−k
′) - combining this with our earlier lower bound r ≤ 1/w2 0implies t = O(log(1/w0)+k
′)). To bound r, we assume A matches Θ(t + k
′) moments, allowing us to replace Ex∼A[·] with Ex∼N(0,1)[·] in the definition of r. Since p(x) has degree 2k
′, we show p(x) ≥ 2
−O(k
′) Ex∼N(0,1)[p 2(x)]1/2 near x =
√2t (Corollary 4.5). The contribution to Ex∼N(0,1)[f 2(x)] from x ∈ [0.9
√2t, 1.1
√2t] will then be at least (3.6t)
t Ex∼N(0,1)[p 2(x)]2−O(k
′)(see Equations (7) and (8) for the full calculations). Meanwhile, by Holder's inequality, ¨ Ex∼N(0,1)[f(x)] ≤
Ex∼N(0,1)[x 3t/2]
2/3 Ex∼N(0,1)[p 3(x)]1/3, which by hypercontractivity is at most (
3t 2e
)
t/2 Ex∼N(0,1)[p(x)
2]
1/22 O(k
′).

Combining these bounds establishes r ≥ 2 Ω(t−k
′)and proves Proposition 4.2. We can also argue that the previous paragraph's result can always be used with w0 ≥ 2
−O(k
′)/k (Proposition 4.1).

By considering the same polynomial p which vanishes at the points with unconstrained weights, we can combine the hypercontractivity of p with the Cauchy-Schwarz inequality to derive a lower bound on the total weight of the equalweight points: Pi≥k′+1 wi ≥ 2
−O(k
′). This immediately implies w0 ≥ 2
−O(k
′)/k.

We have shown that any discrete distribution with k
′
arbitrary weights and k − k
′equal weights cannot match more than O(log(k) + k
′) moments with N (0, 1).

This result extends to approximate moment matching within error 2
−O(t), and holds even after convolving the distribution with a Gaussian (cf. Lemma D.3). For the parallel pancakes testing problem, this implies that for some i ≤ O(log(k) + k
′) the i-th order tensor of the GMM in the alternative hypothesis differs significantly from that of N (0, I). This gap can be detected by estimating the tensor via averaging samples (an operation that has complexity d Θ(m)), leading to the testing algorithm of Theorem 1.4.

## 2. Preliminaries

We present only the essential preliminaries here; see Appendix B for a full version.

Notation We use Z+ for positive integers, R
+
0for nonnegative reals, and [n]
def 
= {1*, . . . , n*}. We use x ⊗ y for the tensor product of two vectors. For a random variable x following distribution D, we write x ∼ D and E[x] for its expectation. The Gaussian distribution with mean µ and covariance Σ is N (µ, Σ), and Pr(E) denotes the probability of event E. The indicator function of E is 1(E).

The Lp norm of an R-valued random variable x is ∥x∥p =
E[|x| p]
1/p, and for a function f : R
d → R, it is ∥f∥p =
Ex∼N(0,I)[|f(x)| p]
1/p. We use a ≲ b to indicate a ≤ Cb for an absolute constant C > 0 independent of a and b. Hermite Analysis In this paper, we use the normalized probabilist's Hermite polynomials, which form an orthonormal basis of L
2:= {f : Ex∼N(0,1)[f 2(x)] < ∞} with respect to the Gaussian measure, i.e., RR 
hk(x)hm(x)e
−x 2/2 dx = 
√2π1(k = m). Every function f ∈ L
2can be uniquely expressed as f(x) = P∞
i=0 aihi(x).

Probability Facts The first fact below follows from direct calculations, the second from the Carbery-Wright inequality, and the last from Holder's inequality combined ¨ with Fact 2.3. Fact 2.1 (Gaussian Moments). E
x∼N(0,1)
[x t] ≲ (t/e)
t/2 ∀t≥0.

Fact 2.2. For every polynomial of degree r and every ϵ > 0, Prx∼N(0,1) (|p(x)| ≤ ϵ∥p∥1) ≤ O(rϵ1/r).

Fact 2.3 (Gaussian Hypercontractivity). If p *is a degree* r polynomial and k > 2*, then* ∥p∥k ≤ (k − 1)r/2∥p∥2.

Fact 2.4. For any polynomial p *of degree* r,
∥p∥1
∥p∥2
≥ 3
−r.

Arithmetic Mean-Geometric Mean Inequality We record the following continuous analog of the Arithmetic Mean- Geometric Mean (AM-GM) inequality. We refer to Appendix B for a more detailed discussion.

Fact 2.5 (Continuous AM-GM Inequality). Let f : R →
R 
+ 0 be a function, and let I ⊆ R *be a finite interval. If* f(x)
and ln f(x) are integrable on I*, then the following holds:*
1 |I| RI
f(x)dx ≥ exp 1 |I| RI
ln f(x)dx
.

Non-Gaussian Component Analysis The parallel pancakes Problem 1.1 is a special case of the following problem. Problem 2.6 (Non-Gaussian Component Analysis
(NGCA)). Let B be a distribution on R. For a unit vector v, we denote by PB,v the distribution with the density PB,v(x) := B(v
⊤x)ϕ⊥v(x), where ϕ⊥v(x) = exp −∥x − (v
⊤x)v∥
22/2/(2π)
(d−1)/2, i.e.,
the distribution that coincides with B on the direction v and is standard Gaussian in every orthogonal direction. We define the following hypothesis testing problem:
- H0: The data distribution is N (0, Id). - H1: The data distribution is PB,v, for some vector v ∈
S
d−1in the unit sphere.

It is known that solving Problem 2.6 when B matches the first m moments with N (0, 1) requires at least d Ω(m)complexity in the statistical query model (Proposition B.8).

## 3. The Uniform Weights Case

In this section we prove the following proposition, which is sufficient for showing our first result, Theorem 1.3. Proposition 3.1. For each k that is larger than a sufficiently large absolute constant, there exists a set S of k *points in* R
such that the uniform distribution over S *matches the first* Ω(log k) *moments with* N (0, 1).

Given the above, Theorem 1.3 follows directly from standard SQ theory. The details are provided in Appendices B.5 and C, but the steps are summarized as follows: Let A be the uniform distribution on the set S from Proposition 3.1. We can define the distribution B to be what one obtains by first drawing a sample from A, rescaling it by 1/
√δ and adding Gaussian noise N (0, 1 − δ). This operation preserves moment matching and makes B a GMM. The NGCA Problem 2.6 with that B then becomes equivalent to the parallel pancakes Problem 1.1. Since B matches m = Ω(log k) moments with N (0, 1), its standard SQ hardness state that its complexity is d Ω(m) = d Ω(log k)(Proposition B.8). We refer to Appendix C for the details of this paragraph. In the remainder, we focus on proving Proposition 3.1 by leveraging a result on designs theory from Kane (2015). The original result in Kane (2015) is highly general and applies to a wide range of topological, path-connected design problems. However, as we will only use the theorem for intervals, we present here a specialized version tailored to this case.

Fact 3.2 (see Theorem 4 in Kane (2015)). Let t ∈ Z+ be an integer, I ⊂ R be an interval and Q be a distribution on I. Let Wt *be the vector space of all polynomials of* degree at most t, and Vt be the vector space of polynomials p of degree at most t *with* Ex∼Q[p(x)] = 0*. Define* Kt = supp∈V \{0}
supx∈I p(x)
| infx∈I p(x)|
. Then for every integer n > (t − 1)(Kt + 1) there exists a set S ⊂ I of n points such that 1 |S| Px∈S
p(x) = Ex∼Q[p(x)] *for all* p ∈ Wt.

Our goal is to show that Kt = 2O(t)for Q = N (0, 1),
which would directly imply Theorem 1.3. However, as noted in Section 1.2, Kt may be infinite when I = R. To address this, we use a distribution Q supported on a bounded interval of R that matches the first t moments of N (0, 1). Applying Fact 3.2 with this Q also suffices for establish Theorem 1.3.

Lemma 3.3 (Gaussian Quadrature (Lemma 4.3 in Diakonikolas et al. (2017))). There is a discrete distribution Q on the real line, supported on t *points, that agrees with* N (0, 1) on the first 2t − 1 moments. All points x in the support of Q *have* |x| = O(
√t).

We start with an anti-concentration property of Gaussian polynomials that will be useful for bounding the numerator in the definition of Kt.

Lemma 3.4. Let C be a sufficiently large constant. For every polynomial p : R → R of degree at most t *with* Ex∼N(0,1)[p(x)] = 0 and for every ϵ > 0 *it holds*

$$\Pr_{x\sim{\mathcal{N}}(0,1)}(p(x)>\epsilon\|p\|_{1})\geq\left(\frac{1}{2}\frac{\|p\|_{1}}{\|p\|_{2}}(1-C t\epsilon^{1+1/t})\right)^{2}\;.$$

Proof. Denote by ϕ(x) the pdf of N (0, 1). We have the following (each step is explained below):

$$\|p\|_{1}\!=\!\int_{p(x)>0}\!\!p(x)\mathrm{d}x-\!\int_{p(x)\leq0}\!\!p(x)\mathrm{d}x=2\!\int_{p(x)>0}\!\!p(x)\phi(x)\mathrm{d}x$$
= 2 Z p(x)≥ϵ∥p∥1 p(x)ϕ(x) dx + Z 0≤p(x)<ϵ∥p∥1 p(x)ϕ(x)dx ! ≤ 2∥p∥2 Pr x∼N(0,1) (p(x) ≥ ϵ∥p∥1) 1/2 + 2ϵ∥p∥1 Pr x∼N(0,1) (|p(x)| ≤ ϵ∥p∥1) ≤ 2∥p∥2 Pr x∼N(0,1) (p(x) ≥ ϵ∥p∥1) 1/2 + 2ϵ∥p∥1Ctϵ1/t,
where the first line used that Ex∼N(0,1)[p(x)] = 0, the penultimate inequality used the Cauchy–Schwarz inequality for the first term and the bound p(x) ≤ ϵ∥p∥1 for the second term, and the last line used the Carbery- Wright inequality (Fact 2.2). Rearranging, we obtain pPrx∼N(0,1)(p(x) > ϵ∥p∥1) ≥12
∥p∥1
∥p∥2
(1 − 2Ctϵ1+1/t).

We rename the constant 2C to C. We now bound Kt from Fact 3.2 with I=[−C
√t,C√t].

Lemma 3.5. Let t > C be an integer, where C is a sufficiently large constant, and define I = [−C
√t, +C
√t].

For every polynomial p of degree at most t *with* Ex∼N(0,1)[p(x)] = 0 *it holds* supx∈I p(x)
| infx∈I p(x)| ≤ 2 O(t).

Proof. It suffices to upper bound the numerator by 2 O(t)∥p∥1 and lower bound the denominator by 2
−O(t)∥p∥1.

Upper bound on numerator We require the following:
Fact 3.6 (Krasikov (2004)). For the k-th normalized probabilist's Hermite polynomial hk*, we have* supx∈R h 2k
(x)e
−x 2/2 = O(k
−1/6).

Consider a polynomial p which has degree at most t and satisfies Ex∼N(0,1)[p(x)] = 0. We first expand the polynomial in the Hermite basis: p(x) = Ptk=1 akhk(x),
where the summation starts from k = 1 because a0 =
Ex∼N(0,1)[p(x)] = 0. For any x ∈ I we have (the first step uses Cauchy–Schwarz inequality):

$$|p(x)|=\left|\sum_{k=1}^{t}a_{k}h_{k}(x)\right|\leq\sqrt{\sum_{k=1}^{t}a_{k}^{2}}\sqrt{\sum_{k=1}^{t}h_{k}^{2}(x)}$$ $$\leq\|p\|_{2}\sqrt{\sum_{k=1}^{t}e^{x^{2}/2}k^{-1/6}}\qquad\qquad\text{(by Fact3.6)}$$ $$\leq\|p\|_{2}\,2^{O(t)}\sqrt{\sum_{k=1}^{t}k^{-1/6}}\quad\text{(using$|x|=O(\sqrt{k})$)}$$ $$\leq\|p\|_{2}\,2^{O(t)}t^{O(1)}\quad\text{(using$\sum_{k=1}^{t}k^{-1/6}=t^{O(1)}$)}$$ $$\leq\|p\|_{2}\,2^{O(t)}\leq\|p\|_{1}2^{O(t)}\quad\text{(using Fact2.4)}$$

Lower bound on the denominator From Lemma 3.4 with 5
−p in place of p and ϵ = 2−t, and Fact 2.4 we get that

$$\Pr_{x\sim\mathcal{N}(0,1)}(p(x)<-2^{-t}\|p\|_{1})\geq\left(\frac{1}{2}\frac{\|p\|_{1}}{\|p\|_{2}}(1-Ct2^{-t-1})\right)^{2}$$ $$\geq\left(\frac{1}{2}3^{-t}(1-Ct2^{-t-1})\right)^{2}>2^{-4t}\.$$

where we used that t is big enough so that C
t 2t <0.5. Then,

$$\Pr_{x\sim{\mathcal{N}}(0,1)}(p(x)<-2^{-t}\|p\|_{1},x\in I)$$
$$\begin{array}{l}{{x{\sim}{\mathcal{N}}(0,1)}}\\ {{\geq\quad\underset{x{\sim}{\mathcal{N}}(0,1)}{{\mathbf{Pr}}}\left(p(x)<-2^{-t}\|p\|_{1}\right)-\underset{x{\sim}{\mathcal{N}}(0,1)}{{\mathbf{Pr}}}\left(x\not\in I\right)}}\\ {{\geq2^{-4t}-2^{-100t}>0\;,}}\end{array}$$

where in the last line we used that I = [−C
√t, +C
√t] for a large constant C. We have thus shown that infx∈I p(x) ≤
−2
−t∥p∥1 and therefore | infx∈I p(x)| > 2
−t∥p∥1.

## 4. The Mostly Equal Weights Case

This section focuses on Theorem 1.4 and is organized as follows. The key structural result is the following impossibility of moment matching: if A is a distribution on k points, with k
′ points having unconstrained weights and the remaining k − k
′equal, then A cannot match more than O(log k + k
′)
moments with the standard Gaussian. Proposition 4.1. Let k
′ < k be positive integers, and let A
be a discrete distribution on k points in R*. Suppose* k−k
′ of the points have equal probability masses, while the remaining k
′ points have unrestricted probability masses. Denote by m *the highest degree for which every degree-*m′ ≤ m polynomial g *satisfies*Ex∼A[g(x)] − Ex∼N(0,1)[g(x)] ≤
2
−C·m∥g∥2, then m *must satisfy* m ≤ O(log k) + O(k
′).

Section 4.1 explains how Proposition 4.1 leads to a testing algorithm (the full proof appears in Appendix D.1). Section 4.2 provides the proof of Proposition 4.1.

## 4.1. Proof Sketch Of Theorem 1.4

Consider the parallel pancakes problem from Theorem 1.4, which is equivalent to the NGCA problem (Problem 2.6) with the 1-d GMM B =Pi∈[k] wiN (µi, 1−δ).

If B approximately matches m moments of N (0, 1), we aim to show that m≤O(log k+k
′), enabling a testing algorithm that detects significant deviations in moment tensors. Specifically, suppose every polynomial p of degree m′≤m with ∥p∥2=1 satisfiesEx∼B[p(x)]− Ex∼N(0,1)[p(x)] ≤(δ/2)Cm for some large constant C ≫ 1. Now, consider the discrete distribution A, which assigns weight wito each center µi/
√δ. By Lemma D.3, A also approximately matches the moments of N (0, 1), but with an error of 2
−O(m)instead of (δ/2)O(m).

Then Proposition 4.1 yields m ≤ O(log k+k
′), as desired.

We just showed that there is a polynomial p of degree at most m = O(log(k) + k
′), where the expectations under B and N (0, 1) differ significantly: λ :=
Ex∼B[p(x)] − Ex∼N(0,1)[p(x)] > (δ/2)Cm. An averaging argument further implies that a gap holds even for some monomial x i. Lifting this to the d-dimensional parallel pancakes, we have the moment tensor gap Ex∼PB,v [x
⊗i] −
Ex∼N(0,I)[x
⊗i] = ±λv⊗i.

The Frobenius norm of the gap is λ > (δ/2)Cm, implying that between the (expected) moment tensors, at least one entry differs by at least ϵ := λ/dm = (d/δ)
(C−1)m. We will test by searching for such an entry in the empirical tensor. Algorithm 1 Testing Algorithm (simplified)
1: **Input**: *k, n*. **Output**: Hˆ ∈ {H0, H1}.

2: For i = 1, 2, 3*, . . . , C* · (log(k) + k
′) do 3: Draw x1*, . . . , x*n ∼ D.

4: Define M ← 1n Pn i=1 x
⊗i.

5: Define M′:= Ex∼N(0,I)[x
⊗i].

6: If ∃a=(i1*, . . . , j*i) such that |Ma−M′a|>d−Cmλm 7: **then** Output H1 and terminate. 8: **Return** H0.

The tester above is a simplified version. However, it is not fully correct, as we must ensure the concentration of the empirical tensor to bound the sample complexity. The Gaussian N (0, I)'s empirical tensor is well-concentrated. While the parallel pancake's tensor might not concentrate well, this happens only when there is a Gaussian component much farther than 
√d from the origin - otherwise every sample from the parallel pancake is within O(
√d) in norm in high probability, and the empirical tensor is entrywise well-concentrated (e.g. by Hoeffding). This is also a testable condition: with ≫ log(k)/wmin samples, we will be able to check if every component is centered at most O(
√d)
from the origin. The full version of the algorithm with this additional preliminary check, along with its correctness proof, are provided in Appendix D.1.

## 4.2. Proof Of Proposition 4.1

We now show Proposition 4.1. We will actually show a slightly different version below, where k
′ of the points have arbitrary weights and the rest have weight at least w0.

Proposition 4.2. Let C be a sufficiently large constant, and let k
′ < k be positive integers. Let A be a discrete distribution on k points in R with probability masses w1*, . . . , w*k, where wi ≥ w0 for i = k
′ + 1, . . . , k *(i.e., the last* k − k
′
weights are lower bounded by w0*, while the first* k
′ weights are unrestricted). Let m *be the largest degree such that* every polynomial g of degree m′ ≤ m *satisfies*

$$\left|\begin{array}{c}\mbox{\bf E}\left[g(x)\right]-\mbox{\bf E}\\ x\sim\mathcal{N}(0,1)\end{array}\right|\leq w_{0}\,2^{-C\cdot m}\|g\|_{2}.\tag{1}$$

Then m ≤ O(log(1/w0)) + O(k
′).

Proposition 4.1 can be derived from this via the following observation (shown in Appendix D.2): in the setting of Proposition 4.2, let p(x) = Qk
′
i=1(x − µi),
where µ1*, . . . , µ*k′ are the points in the support of A with the unconstrained weights. Then, the weights of the k − k
′ points with uniform weights is always Pk i=k′+1 wi ≥
Ex∼A[p(x)]2 Ex∼A[p2(x)] 
≳ 
∥p∥
2 1
∥p∥
2 2
≥ 3
−2k
′, where the first step uses Cauchy-Schwarz inequality, the second uses the (approximate) moment matching, and the third is a consequence of Gaussian hypercontractivity (Fact 2.4).

This means that every such weight is wi ≥ 3
−2k
′/k, which when plugged into Proposition 4.2 gives Proposition 4.1. We now focus on showing Proposition 4.2. We will follow a top-down presentation, starting with the proof strategy and concluding with a derivation of the necessary bounds. We will use a reparameterization m = 2t + 4k
′ with t even.

The goal is to show that if A is assumed to approximately match the first m = 2t + 4k
′ moments with N (0, 1)
(in the sense of Equation (1)), then t must be at most O(log(1/w0) + k
′). Let µ1*, . . . , µ*k be the points on which A is supported, where the first k
′ points are the ones with the unrestricted weights, and consider the polynomial f(x) = x tp(x), where p(x) = (x − µ1)
2(x − µ2)
2*· · ·*(x − µk′ )
2.

The proof strategy is the following: if the expectation of f under A approximately matches that of N (0, 1), then the value of f on every point µi cannot be too large, which will cause the expectations of f 2to deviate.

Because of Equation (1) with g(x) = f 2(x), we have:

$$\begin{array}{r l}{{}}&{{}\sum_{i=k^{\prime}+1}^{k}\ w_{i}\,\mu_{i}^{t}\,p(\mu_{i})=\mathbf{\Sigma}_{x\sim A}\left[x^{t}p(x)\right]}\\ {{}}&{{}}\\ {{}}&{{}\leq\mathbf{\Sigma}_{x\sim{\mathcal{N}}(0,1)}\left[x^{t}p(x)\right]+w_{0}{\frac{\|x^{t}p(x)\|_{2}}{2C(2t+4k^{\prime})}}\ .}\end{array}$$
. (3)
This, together with the lower bound wi ≥ w0 for the points i = k
′ + 1*, . . . , k* and the fact that t is even, implies that for all i = k
′ + 1*, . . . , k* it holds

$$\mu_{i}^{t}\,p(\mu_{i})\leq\frac{1}{w_{0}}\operatorname*{\mathbf{E}}_{x\sim\mathcal{N}(0,1)}\left[x^{t}p(x)\right]+\frac{\|x^{t}p(x)\|_{2}}{2^{C(2t+4k^{\prime})}}\;.$$
. (4)
We now examine the expectations of the square of f(x). Because of Equation (1) with g(x) = f 2(x), we have

$$\begin{array}{c}{{\mathbf{E}}}\\ {{x{\sim}{\mathcal{N}}(0,1)}}\end{array}\left[x^{2t}p^{2}(x)\right]\leq\mathop{\mathbf{E}}_{x{\sim}A}\left[x^{2t}p^{2}(x)\right]+\frac{\|x^{2t}p^{2}(x)\|_{2}}{2^{C(2t+4k^{\prime})}}\ \ ,$$ $$=\sum_{i=k^{\prime}+1}^{k}w_{i}\ \left(\mu_{i}^{t}\,p(\mu_{i})\right)^{2}+\frac{\|x^{2t}p^{2}(x)\|_{2}}{2^{C(2t+4k^{\prime})}}\ \ .$$

Next, we can combine this with Equation (4), divide both sides by Ex∼N(0,1) [x tp(x)]2(and use Pk i=k′+1 wi ≤ 1) to obtain the following, where λ := 2−C(2t+4k
′):

$$\frac{\mathbf{E}_{x\sim\mathcal{N}(0,1)}\left[x^{2t}p^{2}(x)\right]}{\mathbf{E}_{x\sim\mathcal{N}(0,1)}\left[x^{t}p(x)\right]^{2}}\leq\left(\frac{1}{w_{0}}\right)^{2}$$ $$+\lambda^{2}\frac{\mathbf{E}_{x\sim\mathcal{N}(0,1)}\left[x^{2t}p^{2}(x)\right]}{\mathbf{E}_{x\sim\mathcal{N}(0,1)}\left[x^{t}p(x)\right]^{2}}+\lambda\frac{\mathbf{E}_{x\sim\mathcal{N}(0,1)}\left[x^{4t}p^{4}(x)\right]^{\frac{1}{2}}}{\mathbf{E}_{x\sim\mathcal{N}(0,1)}\left[x^{t}p(x)\right]^{2}}.$$

Let us simplify this inequality. Let r be the ratio on the LHS. The second term on the RHS is λ 2· r. The third term is at most 3 t+2k
′λr, by applying Gaussian hypercontractivity (Fact 2.3) to the polynomial x tp(x). Thus, the inequality becomes r(1 − λ 2 − λ3 t+2k
′) ≲ 1/w20. Since λ = 2−C(2t+4k
′)for large constant C, the expression inside the parentheses is greater than 0.5. Therefore, the inequality implies that r ≲ 1/w20.

The next step is to establish a lower bound for r, specifically to show that r ≥ 2 Ω(t)/2 O(k
′). If this can be done, combining the two bounds 2 Ω(t)/2 O(k
′) ≤ 1/w20and taking logarithms yields t = O(log(1/w0)) + O(k
′), completing the proof of Proposition 4.2.

## 4.2.1. Lower Bounding The Ratio R

We want to establish the following, which was the missing piece in the proof of Proposition 4.2 above.

Lemma 4.3. Let p : R → R
+ 0 be a polynomial of the form p(x) = (x − µ1)
2(x − µ2)
2*· · ·*(x − µk′ )
2*. Then*

$$\frac{\mathbf{E}_{x\sim{\mathcal{N}}(0,1)}\left[x^{2t}p^{2}(x)\right]}{\mathbf{E}_{x\sim{\mathcal{N}}(0,1)}\left[x^{t}p(x)\right]^{2}}\stackrel{>}{\sim}\frac{2^{\Omega(t)}}{2^{O(k^{\prime})}}\ .$$

$$(2)$$
$$(3)$$

The most difficult part involves lower bounding the numerator. To this end, we will show the following bound:
Lemma 4.4. Let p : R → R *be a polynomial of the form* p(x) = (x−µ1)(x−µ2)· · ·(x−µk′ ) where µ1*, . . . , µ*k′ ∈
R*, and define* I := [0.9
√2t, 1.1
√2t]. For every t > 0 and for every µ1, . . . , µk′ ∈ R*, the following holds:*

$$\exp\left(\frac{1}{|I|}\int_{x\in I}\ln|p(x)|\mathrm{d}x\right)\geq\max_{y\in\mathbb{R}:|y|\leq\sqrt{t}}\frac{|p(y)|}{2^{O(k^{\prime})}}\;.\tag{5}$$
$$(4)$$

We will actually apply Lemma 4.4 after taking expectations of both sides. This version is presented below, and its proof follows by taking expectations and performing some manipulations (see Appendix D.3 for a detailed proof).

Corollary 4.5. Let p : R → R *be a polynomial of the form* p(x) = (x−µ1)(x−µ2)· · ·(x−µk′ ) where µ1*, . . . , µ*k′ ∈
R *are arbitrary parameters. Define* I = [0.9
√2t, 1.1
√2t].

For all t ≥ 1 *we have* exp 1 |I| Rx∈I
ln |p(x)|dx
≥
∥p∥2 2O(k′).

To see why the above bound is needed to prove Lemma 4.3, we will first prove Lemma 4.3 assuming Corollary 4.5. Then, we will prove Lemma 4.4. Proof of Lemma *4.3.* First, for the denominator, we have the following:

$$\begin{array}{c}\mbox{\bf E}\\ x\sim{\cal N}(0,1)\end{array}[x^{t}p(x)]\leq\mbox{\bf E}\\ x\sim{\cal N}(0,1)\end{array}[p^{3}(x)]^{1/3}\mbox{\bf E}\\ x\sim{\cal N}(0,1)\end{array}[x^{3t/2}]^{2/3}$$ $$\begin{array}{c}\mbox{\bf\lesssim}\|p\|_{3}\left(\frac{3t}{2e}\right)^{t/2}\leq2^{k^{\prime}}\|p\|_{2}\left(\frac{3t}{2e}\right)^{t/2}\,,\end{array}\tag{6}$$

where the first step uses Holder's inequality, the second step ¨ uses the Gaussian moments bound (Fact 2.1), and the final step uses Gaussian hypercontractivity (Fact 2.3). We now lower bound the numerator. Define I := [0.9
√2t, 1.1
√2t]. We have the following (see below for explanations of each step):

$$\begin{split}&\mathbf{E}\\ x\sim&\mathcal{N}(0,1)\left[x^{2t}p^{2}(x)\right]\gtrsim\int_{-\infty}^{+\infty}x^{2t}e^{-x^{2}/2}p^{2}(x)\,\mathrm{d}x\\ &\geq\int_{x\in I}x^{2t}e^{-x^{2}/2}p^{2}(x)\,\mathrm{d}x\\ &\geq(1.62t)^{t}e^{-0.81t}\int_{x\in I}p^{2}(x)\,\mathrm{d}x\\ &=(1.62t)^{t}e^{-0.81t}|I|\left(\frac{1}{|I|}\int_{x\in I}p^{2}(x)\,\mathrm{d}x\right)\\ &\geq(1.62t)^{t}e^{-0.81t}\left(\frac{1}{|I|}\int_{x\in I}p^{2}(x)\,\mathrm{d}x\right),\end{split}\tag{7}$$

where the third inequality uses that minx∈I x 2te
−x 2/2 ≥
(1.62t)
te
−0.81t, and the final inequality uses that |I| =
0.2
√2t = Ω(1). We now focus on the root mean square term 1 |I| Rx∈I
p 2(x) dx, which we will bound using the AM-
GM inequality (Fact 2.5) and the geometric mean bound from Lemma 4.4. The first step below applies Fact 2.5 with f(x*) :=* p 2(x), and the next step uses Corollary 4.5.

$$\frac{1}{|I|}\int_{x\in I}p^{2}(x)\,\mathrm{d}x\geq\exp\left(\frac{1}{|I|}\int_{x\in I}\ln|p(x)|\,\mathrm{d}x\right)^{2}\geq\frac{\|p\|_{2}^{2}}{2O(k^{\prime})}.$$

Combining with Equation (7), we obtain the following bound for the numerator:

$$\begin{array}{c}\mbox{E}\\ x\sim\mathcal{N}(0,1)\end{array}\left[x^{2t}p^{2}(x)\right]\stackrel{{\sim}}{{\sim}}(1.62t)^{t}e^{-0.81t}\frac{\|p\|_{2}^{2}}{2O(k^{\prime})}.\tag{8}$$

Combining Equation (6) and Equation (8), we conclude

$${\frac{\mathbf{E}_{x\sim{\mathcal{N}}(0,1)}\left[x^{2t}p^{2}(x)\right]}{\mathbf{E}_{x\sim{\mathcal{N}}(0,1)}\left[x^{t}p(x)\right]^{2}}}{\overset{\sim}{\sim}}{\frac{(1.62)^{t}}{({\frac{1.5}{e}})^{t}}}{e^{0.81t}}{\frac{\sim}{2O(k^{\prime})}}{\overset{\sim}{\geq}}{\frac{(1.3)^{t}}{2O(k^{\prime})}}.\ \square$$

We conclude this section by proving Lemma 4.4.

Proof of Lemma *4.4.* Fix an arbitrary y ∈ R with |y| ≤ 
√t.

First, note that by the property of logarithms and sums, we can write the left hand side as

$$\exp\left(\sum_{i=1}^{k^{\prime}}{\frac{1}{|I|}}\int_{x\in I}\ln|x-\mu_{i}|\mathrm{d}x\right)$$
 .
In order to show Equation (5), it suffices to work with each term and show the following for each i ∈ [k
′]:

$${\frac{1}{|I|}}\int_{x\in I}\ln|x-\mu_{i}|\geq\ln|y-\mu_{i}|-O(1)\ .$$

Equivalently, it suffices to show that Equation (5) holds for every linear polynomial of the form p(x) = x − a. Therefore, the goal for the rest of this proof is to show that

$$\exp\left(\frac{1}{|I|}\int_{x\in I}\ln|x-a|\mathrm{d}x\right)\geq|y-a|/O(1)\;,\tag{9}$$

holds for every a ∈ R and y ∈ R with |y| ≤ 
√t. We will examine two cases. Case 1 The first case is when the root a of the polynomial is outside the interval I. In this case, we can show that |x − a|/|y − a| = Θ(1), which implies ln |x − a| ≥ ln |y − a|−O(1), and the desired conclusion (Equation (9)) follows by integrating both sides and applying the exp(·) function. To show the earlier claim that |x − a|/|y − a| = Θ(1), we can consider the following sub-cases:
1. Case a ≥ 1.1
√2t (i.e., a is to the right of I): Suppose a = 1.1
√2t+u for some non-negative u. Then, a−x =
(1.1
√2t−x)+u = Θ(√t)+u and a−y = (1.1
√2t−
y)+u = Θ(√t)+u. Therefore, for any u ≥ 0, the ratio |x − a|/|y − a| = (Θ(√t) + u)/(Θ(√t) + u) = Θ(1).

2. The cases a < −
√t and a ∈ [
√t, 0.9
√2t] can be shown in a similar manner.

Case 2 Suppose that the root a of the polynomial p lies within the interval I. In that case, we can show via derivative analysis that f(a) := 1 |I| Rx∈I
ln |x−a| dx for a ∈ I is minimized at the midpoint of I, i.e., at a =
√2t, and confirm that f(
√2t) ≥
√t/20 = Ω(|y − a|). These calculations are provided in Appendix D.3.

## Conclusions And Future Work

Our work makes progress in understanding the complexity of learning parallel pancake GMMs, in terms of both lower and upper bounds. We establish the tightness of existing algorithms for uniform weights and provide an improved testing algorithm for uneven weights. A number of interesting open problems remain:
- Can we extend our testing algorithm to learning the unknown direction of the parallel pancakes? More broadly, can we characterize the complexity of learning GMMs with common covariance and not necessarily collinear means as a function of the weights distribution?

- Can we obtain an algorithm with quasi-polynomial (i.e.,
d O(log(1/wmin)) complexity for GMMs whose components have unknown (and potentially different) covariances?

## Impact Statement

This work is theoretical in nature and focuses on advancing fundamental knowledge. As such, it does not directly raise any societal or ethical concerns that warrant special consideration.

## Acknowledgments

Ilias Diakonikolas was supported by NSF Medium Award CCF-2107079 and an H.I. Romnes Faculty Fellowship. Daniel M. Kane was supported by NSF Medium Award CCF-2107547 and NSF Award CCF-1553288 (CAREER). The work of Jasper C.H. Lee was done in part while he was at UW Madison, supported by NSF Medium Award CCF-2107079.

## References

Achlioptas, D. and McSherry, F. On spectral learning of mixtures of distributions. In Proc. 18th Annual Conference on Learning Theory (COLT), 2005.

Anderson, P., Bafna, M., Buhai, R. D., Kothari, P. K., and Steurer, D. Dimension reduction via sum-of-squares and improved clustering algorithms for non-spherical mixtures. *arXiv preprint arXiv:2411.12438*, 2024.

Andrews, G. E., Askey, R., and Roy, R. *Special Functions*.

1999.

Arora, S. and Kannan, R. Learning mixtures of arbitrary Gaussians. In Proc. 33rd Annual ACM Symposium on Theory of Computing (STOC), 2001.

Bakshi, A. and Kothari, P. K. Outlier-robust clustering of non-spherical mixtures. *arXiv:2005.02970*, 2020.

Bakshi, A., Diakonikolas, I., Hopkins, S. B., Kane, D., Karmalkar, S., and Kothari, P. K. Outlier-robust clustering of gaussians and other non-spherical mixtures. In Foundations of Computer Science (FOCS), 2020.

Bakshi, A., Diakonikolas, I., Jia, H., Kane, D. M., Kothari, P. K., and Vempala, S. S. Robustly learning mixtures of k arbitrary gaussians. In Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing (STOC 2022), 2022. Also available as arXiv:2012.02119.

Belkin, M. and Sinha, K. Polynomial learning of distribution families. *SIAM Journal on Computing*, 44(4):889–911, 2015.

Bogachev, V. *Gaussian Measures*. 1998.

Brubaker, S. C. and Vempala, S. S. Isotropic pca and affineinvariant clustering. Building Bridges: Between Mathematics and Computer Science, pp. 241–281, 2008.

Bruna, J., Regev, O., Song, M. J., and Tang, Y. Continuous LWE. In *Symposium on Theory of Computing (STOC)*, 2021.

Buhai, R. and Steurer, D. Beyond parallel pancakes: Quasipolynomial time guarantees for non-spherical gaussian mixtures. In The Thirty Sixth Annual Conference on Learning Theory, pp. 548–611. PMLR, 2023.

Carbery, A. and Wright, J. Distributional and l q norm inequalities for polynomials over convex bodies in R
n.

Mathematical Research Letters, 8(3):233–248, 2001. ISSN 10732780, 1945001X. doi: 10.4310/MRL.2001.v8. n3.a1.

Dasgupta, S. Learning mixtures of gaussians. In Foundations of Computer Science (FOCS), 1999.

Daskalakis, C. and Kamath, G. Faster and sample nearoptimal algorithms for proper learning mixtures of gaussians. In *Conference on Learning Theory*, pp. 1183–1213. PMLR, 2014.

Diakonikolas, I. and Kane, D. M. Implicit high-order moment tensor estimation and learning latent variable models. *arXiv preprint arXiv:2411.15669*, 2024.

Diakonikolas, I., Kane, D. M., and Stewart, A. Statistical query lower bounds for robust estimation of highdimensional gaussians and gaussian mixtures. In *Proc.* 58th IEEE Symposium on Foundations of Computer Science (FOCS), 2017. doi: 10.1109/FOCS.2017.16.

Diakonikolas, I., Hopkins, S. B., Kane, D., and Karmalkar, S. Robustly learning any clusterable mixture of gaussians. arXiv:2005.06417, 2020.

Diakonikolas, I., Kane, D. M., Kongsgaard, D., Li, J., and Tian, K. Clustering mixture models in almost-linear time via list-decodable mean estimation. In Proc. 54th Annual ACM Symposium on Theory of Computing (STOC), 2022.

Diakonikolas, I., Kane, D., Ren, L., and Sun, Y. Sq lower bounds for non-gaussian component analysis with weaker assumptions. *Advances in Neural Information Processing* Systems, 2023.

Diakonikolas, I., Karmalkar, S., Pang, S., and Potechin, A.

Sum-of-squares lower bounds for non-gaussian component analysis. In 2024 IEEE 65th Annual Symposium on Foundations of Computer Science (FOCS), pp. 949–958. IEEE, 2024.

Feldman, V. Statistical query learning. In Encyclopedia of Algorithms, pp. 2090–2095. Springer New York, 2016.

Feldman, V., Grigorescu, E., Reyzin, L., Vempala, S. S.,
and Xiao, Y. Statistical algorithms and a lower bound for detecting planted cliques. *Journal of the ACM*, 64(2), 2017a. doi: 10.1145/3046674.

Feldman, V., Guzman, C., and Vempala, S. S. Statistical query algorithms for mean vector estimation and stochastic convex optimization. 2017b.

Garc´ıa-Escudero, L., Gordaliza, A., Matran, C., and Mayo- ´
Iscar, A. A review of robust clustering methods. Advances in Data Analysis and Classification, 4(2):89–109, 2010. doi: 10.1007/s11634-010-0064-5.

Gupte, A., Vafa, N., and Vaikuntanathan, V. Continuous LWE is as hard as LWE & applications to learning gaussian mixtures. *CoRR*, abs/2204.02550, 2022.

Hardt, M. and Price, E. Tight bounds for learning a mixture of two gaussians. In Proceedings of the forty-seventh annual ACM symposium on Theory of computing, pp. 753–760, 2015.

Hopkins, S. B. and Li, J. Mixture models, robustness, and sum of squares proofs. In Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing
(STOC), pp. 1021–1034, 2018.

Kane, D. Small designs for path-connected spaces and path-connected homogeneous spaces. Transactions of the American Mathematical Society, 2015.

Kannan, R., Salmasian, H., and Vempala, S. The spectral method for general mixture models. In *Proc. 18th Annual* Conference on Learning Theory (COLT), 2005.

Kearns, M. J. Efficient noise-tolerant learning from statistical queries. *Journal of the ACM*, 45(6):983–1006, 1998.

Kothari, P. K. and Steurer, D. Outlier-robust momentestimation via sum-of-squares. *arXiv preprint* arXiv:1711.11581, 2017.

Kothari, P. K., Steinhardt, J., and Steurer, D. Robust moment estimation and improved clustering via sum of squares. In Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing (STOC), pp. 1035–1046, 2018.

Krasikov, I. New bounds on the Hermite polynomials. *arXiv* preprint math/0401310, 2004.

Lindsay, B. Mixture models: theory, geometry and applications. Institute for Mathematical Statistics, 1995.

Liu, A. and Li, J. Clustering mixtures with almost optimal separation in polynomial time. In Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing (STOC), pp. 1248–1261, 2022.

Liu, A. and Moitra, A. Settling the robust learnability of mixtures of gaussians. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing (STOC), pp. 518–531, 2021.

Liu, A. and Moitra, A. Learning gmms with nearly optimal robustness guarantees. In Proc. 35th Annual Conference on Learning Theory (COLT), 2022.

Moitra, A. and Valiant, G. Settling the polynomial learnability of mixtures of gaussians. In Proceedings of the 51st Annual IEEE Symposium on Foundations of Computer Science (FOCS), pp. 93–102, 2010.

Nelson, E. The free markoff field. *Journal of Functional* Analysis, 1973.

O'Donnell, R. *Analysis of Boolean Functions*. 2014. Pearson, K. Contributions to the mathematical theory of evolution. *Philosophical Transactions of the Royal Society* of London. Series A, 185:71–110, 1894.

Suresh, A. T., Orlitsky, A., Acharya, J., and Jafarpour, A.

Near-optimal-sample estimators for spherical gaussian mixtures. *Advances in Neural Information Processing* Systems, 27, 2014.

Szego, G. ¨ *Orthogonal Polynomials*, volume XXIII of American Mathematical Society Colloquium Publications. American Mathematical Society, 1989.

Titterington, D., Smith, A., and Makov, U. Statistical Analysis of Finite Mixture Distributions. Wiley, New York, 1985.

Vempala, S. and Wang, G. A spectral algorithm for learning mixtures of distributions. In Proceedings of the 43rd Annual IEEE Symposium on Foundations of Computer Science (FOCS), pp. 113–122, 2002.

## Supplementary Material

Organization Appendix A discusses additional related work on Gaussian mixture models, Appendix B provides the full version of the preliminaries needed for the technical proofs, Appendix C provides the missing details from the proof of our first main result, Theorem 1.3, and Appendix D provides the missing details from our second main result, Theorem 1.4.

## A. Additional Related Work

Learning Gaussian Mixture Models (GMMs) is one of the most studied problems in statistics, dating back to Pearson (1894).

Over the years, a plethora of works has explored this area (Dasgupta, 1999; Arora & Kannan, 2001; Vempala & Wang, 2002; Achlioptas & McSherry, 2005; Kannan et al., 2005; Brubaker & Vempala, 2008; Moitra & Valiant, 2010; Belkin & Sinha, 2015; Suresh et al., 2014; Daskalakis & Kamath, 2014; Hardt & Price, 2015; Diakonikolas et al., 2020; Bakshi et al.,
2020; Diakonikolas et al., 2022; Liu & Moitra, 2021; Bakshi et al., 2022). Here, we provide a brief exposition of part of this literature, though this is not a comprehensive survey. A key starting point was Dasgupta (1999), which studied learning GMMs with well-separated, spherical covariance components. Subsequent work by Vempala & Wang (2002); Achlioptas & McSherry (2005); Kannan et al. (2005) improved the separation condition to be dimension-independent. Hopkins & Li (2018) and Kothari et al. (2018) later refined the separation assumption to the information-theoretic limit, achieving this with quasi-polynomial-time algorithms. Notably, most of the aforementioned works extend beyond spherical Gaussians; however, they measure the pairwise mean separation between components relative to the largest eigenvalue of the components' covariance matrices. More recently, Liu & Li (2022) improved the runtime to polynomial for spherical Gaussians. The case of arbitrary Gaussians with unknown component covariances has also been extensively studied (Belkin & Sinha, 2015; Moitra & Valiant, 2010; Bakshi & Kothari, 2020; Diakonikolas et al., 2020; Liu & Moitra, 2022; Bakshi et al., 2022). While the first works in this list had complexities doubly exponential in k, the number of components, the most recent have reduced this to d O(k). As noted in Section 1, hardness results from Diakonikolas et al. (2017); Bruna et al. (2021); Gupte et al. (2022) showed this complexity is necessary. However, certain special cases of GMMs can circumvent these lower bounds. This was the focus of Buhai & Steurer (2023) and Anderson et al. (2024), who studied GMMs with a minimum mixing weight wmin ≥ 1/ poly(k) and unknown but shared covariance across components. These papers motivated us to study whether further improvements are possible for this family of GMMs.

## B. Additional Preliminaries B.1. Notation

We use Z+ to denote positive integers. For n ∈ Z+ we denote [n]
def = {1*, . . . , n*}. We denote by R
+ 0 the set of all non-negative positive real numbers. We denote by exp(x) = e xthe exponential function, and by ln(x) the natural logarithm
(the logarithm with base e). We write x⊗y to denote the tensor product between two vectors *x, y*. The tensor product of two vectors u ∈ R
m and v ∈ R
n is a matrix u ⊗ v ∈ R
m×n defined such that (u ⊗ v)ij = uivj , where each entry is the product of the corresponding components of u and v. This can be extended to product between more than two vectors. We denote x
⊗kthe k-fold tensor product of the vector x with itself. If A is a tensor, ∥A∥F denotes its Frobenius norm, which is the Euclidean norm of the vector obtained by stacking all entries of the tensor into a single vector. We write x ∼ D for a random variable x following the distribution D and use E[x] for its expectation. We use N (µ, Σ) to denote the Gaussian distribution with mean µ and covariance matrix Σ. We write Pr(E) for the probability of an event E. We denote by 1(E) the indicator function of the event E. The Lp norm of a (R-valued) random variable x is defined to be ∥x∥p = E[|x| p]
1/p. The Lp norm of a function f : R
d → R is defined to be the Lp norm of the random variable f(x), i.e., ∥f∥p = Ex∼N(0,I)[|f(x)| p]
1/p.

We use a ≲ b to denote that there exists an absolute universal constant C > 0 (independent of the variables or parameters on which a and b depend) such that a ≤ Cb. Sometimes, we will also use the O(·), Ω(·), Θ(·) notation with the standard meaning.

## B.2. Hermite Analysis

Hermite polynomials form a complete orthogonal basis of the vector space L
2(R, N (0, 1)) of all functions f : R → R
such that Ex∼N(0,1)[f 2(x)] < ∞. We will use the *normalized probabilist's* Hermite polynomials, which have unit norm and are pairwise orthogonal with respect to the Gaussian measure, i.e., RR 
hk(x)hm(x)e
−x 2/2dx =
√2π1(k = m). These polynomials are the ones obtained by Gram-Schmidt orthonormalization of the basis {1, x, x2*, . . .*} with respect to the inner product ⟨*f, g*⟩N(0,1) := Ex∼N(0,1)[f(x)g(x)]. Every function f ∈ L
2(R, N (0, 1)) can be uniquely written as f(x) = P∞
i=0 aihi(x) and we have limn→∞ Ex∼N(0,1)[(f(x) −Pn i=0 aihi(x))2] = 0 (see Andrews et al. (1999) for a more detailed exposition of Hermite polynomial's properties). We have the following closed form formula (see, e.g., Szego¨ (1989)):

$$h_{n}(x)=\sqrt{n!}\,\sum_{j=0}^{\lfloor n/2\rfloor}\,\frac{(-1)^{j}}{j!(n-2j)!2^{j}}x^{n-2j}\ .$$

$$(10)$$
n−2j. (10)
We will use the following fact, stating that the largest coefficient in the above expansion cannot be too large.

Fact B.1 (Upper Bound on Hermite Polynomial Coefficients). Let hn(x) *denote the normalized probabilist's Hermite* polynomial of order n*. In the monomial expansion* hn(x) = Pn j=1 ajx j, it holds |aj | ≤ 2 O(n)*for all* j ∈ [n].

Proof. This follows by Equation (10). The j-th coefficient is

√n!(−1)j/(j!(n − 2j)!2j)
 ≤
√n!/(j!(n − 2j)!2j). Then one can use the elementary inequalities e(k/e)
k ≤ k! ≤ ek(k/e)
kto bound the factorials that appear in the numerator and denominator and optimize the resulting function. The derivative analysis of the resulting function gives two points of zero derivative: j =
1 4
(1 + 2n −
√1 + 4n) and j =
1 4
(1 + 2n +
√1 + 4n). For each point, it can be checked that the function is smaller than 2 n in the limit n → ∞.

Definition B.2 (Ornstein-Uhlenbeck Operator). For a ρ ∈ [−1, 1], we define the *Ornstein-Uhlenbeck* (or *Gaussian noise*)
operator Uρ as the operator that maps a distribution F on R to the distribution of the random variable ρX +p1 − ρ 2Z,
where X ∼ F and Z ∼ N (0, 1) independently of X. A well-known property of *Ornstein–Uhlenbeck* operator is that it operates diagonally with respect to Hermite polynomials.

Fact B.3 (see, e.g., O'Donnell (2014)). For any normalized Hermite polynomial hi, any distribution F on R*, and* δ ∈ [−1, 1],
it holds that EX∼UρF [hi(X)] = ρ i EX∼F [hi(X)].

B.3. Properties of Polynomials Under the Gaussian Measure Fact 2.1 (Gaussian Moments). E
x∼N(0,1)
[x t] ≲ (t/e)
t/2 ∀t≥0.

Fact B.4 (Carbery-Wright Inequality (Carbery & Wright, 2001)). There is an absolute constant C *such that the following* holds. Let *q, γ* ∈ R
+
0, µ ∈ R
d, Σ ∈ R
d×dsuch that Σ *is symmetric PSD and* p : R
d → R *be a multivariate polynomial of* degree at most r*. Then*

$$\Pr_{r\sim{\mathcal{N}}(\mu,\Sigma)}\left(|p(x)|\leq\gamma\right)\leq{\frac{C q\gamma^{1/r}}{\left(\mathbf{E}_{z\sim{\mathcal{N}}(\mu,\Sigma)}\left[|p(z)|^{q/r}\right]\right)^{1/q}}}\ .$$
.
The way that we will apply this is with the following choice of parameters: d = 1, µ = 0, Σ = 1, q = r and γ = ϵ∥p∥1, where ϵ is a new parameter. This gives:
Fact 2.2. For every polynomial of degree r and every ϵ > 0, Prx∼N(0,1) (|p(x)| ≤ ϵ∥p∥1) ≤ O(rϵ1/r).

The following inequality can be easily derived using Holder's inequality. ¨
Fact B.5. ∥x∥2 ≤ ∥x∥
1/3 1∥x∥
2/3 4*for any random variable.*
The following inequality is the Gaussian Hypercontractivity property (see, e.g., Bogachev (1998); Nelson (1973))
Fact 2.3 (Gaussian Hypercontractivity). If p is a degree r polynomial and k > 2*, then* ∥p∥k ≤ (k − 1)r/2∥p∥2.

In particular we will use the above in the following way: Fact 2.4. For any polynomial p *of degree* r,
∥p∥1
∥p∥2
≥ 3
−r.

Proof. ∥p∥1 ≥
∥p∥
3 2
∥p∥
2 4
= ∥p∥2
∥p∥2
∥p∥4 2≥ 3
−t∥p∥2, where the first step used Fact B.5 and the last step used Gaussian Hypercontractivtiy (Fact 2.3) with k = 4. Rearranging completes the proof. Fact B.6 (Gaussian Norm Concentration). If x ∼ N (µ, I), with probability 1 − τ *we have that*

$$\left|\left\|x\right\|^{2}-\left(\left\|\mu\right\|^{2}+d\right)\right|\lesssim\log{\frac{1}{\tau}}+\left({\sqrt{d}}+\left\|\mu\right\|\right){\sqrt{\log{\frac{1}{\tau}}}}\ .$$

## B.4. Arithmetic Mean-Geometric Mean Inequality

In this paper, we will use a continuous analog of the *Arithmetic Mean-Geometric Mean* (AM-GM) inequality. The continuous analog for the arithmetic mean of a sequence 1n Pn i=1 xiis what one obtains by replacing the summation with its continuous counterpart. Specifically, the arithmetic mean of a function f : R → R over an interval I is defined as: (1/|I|)RI
f(x)dx.

The geometric mean of a discrete sequence is Qn i=1 x 1/n i. Its generalization relies on the property that ln Qn i=1 x 1/n i =
1 n Pn i=1 ln xi (assuming xi > 0). By replacing the summation with an integral, the generalization of the geometric mean of a function f over an interval I is: exp 1 |I| RI
ln f(x)dx
. The continuous analog of the AM-GM inequality thus is the following statement. The proof follows directly from Jensen's inequality:
Fact 2.5 (Continuous AM-GM Inequality). Let f : R → R
+ 0 be a function, and let I ⊆ R be a finite interval. If f(x) and ln f(x) are integrable on I*, then the following holds:* 1 |I| RI
f(x)dx ≥ exp 1 |I| RI
ln f(x)dx
.

## B.5. Statistical Query Lower Bounds Background

We first restate the definition of the non-Gaussian component analysis (NGCA) hypothesis testing problem.

Problem 2.6 (Non-Gaussian Component Analysis (NGCA)). Let B be a distribution on R. For a unit vector v, we denote by PB,v the distribution with the density PB,v(x) := B(v
⊤x)ϕ⊥v(x), where ϕ⊥v(x) =
exp −∥x − (v
⊤x)v∥
22/2/(2π)
(d−1)/2, i.e., the distribution that coincides with B on the direction v and is standard Gaussian in every orthogonal direction. We define the following hypothesis testing problem:
- H0: The data distribution is N (0, Id).

- H1: The data distribution is PB,v, for some vector v ∈ Sd−1in the unit sphere.

Condition B.7 (Approximate moment matching). Let m ∈ Z+. The distribution B on R is such that Ex∼B[x i] −
Ex∼N(0,1)[x i]| ≤ ν for all i ∈ [m].

A known result is that the NGCA problem of Problem 2.6 is hard in the SQ model if B matches a lot of moments with the standard Gaussian. This was shown in (Diakonikolas et al., 2017) and was later strengthened in Diakonikolas et al. (2023). The following is Theorem 1.5 in Diakonikolas et al. (2023) using λ = 1/2 and c = (1 − λ)/*8 = 1*/16.

Proposition B.8 (Theorem 1.5 in Diakonikolas et al. (2023)). Let d, m *be positive integers with* d ≥ (m log d)
2. Any SQ
algorithm that solves Problem 2.6 for a distribution B satisfying Condition B.7 *requires either* 2 d Ω(1)many queries or at least one query with accuracy d
−m/16 + (1 + o(1))ν.

## C. Omitted Details From Section 3

We restate and prove Theorem 1.3.

Theorem 1.3 (SQ Lower Bound for Uniform Weights). Let C be a sufficiently large absolute constant, k > C and d ≥ (log k log d)
2 be integers. If we further restrict the alternative hypothesis in Problem 1.1 to have wi = 1/k *for all* i ∈ [k]*, any SQ algorithm requires either* 2 d Ω(1) *queries or at least one query of accuracy* d
−Ω(log k).

Proof. Let S be the set from Proposition 4.2 and A be the uniform distribution on S. That is, A is a discrete distribution supported on k points and is guaranteed to match the first m = Ω(log k) moments with N (0, 1). Let B = UρA be the distribution which is obtained by applying the Ornstein-Uhlenbeck operator (Definition B.2) with ρ =
√δ. Then B is a

$$(11)$$

k-GMM with uniform weights and variance 1 − δ for each component. Moreover, for every t = 0, 1*, . . . , m* we have the following for the i-th Hermite polynomial

$$\mathbf{E}_{x\sim D_{\rho}}[h_{i}(x)]=\mathbf{E}_{x\sim U_{\rho}A}[h_{i}(x)]=\rho^{i}\mathbf{E}_{x\sim A}[h_{i}(x)]=\rho^{i}\mathbf{E}_{x\sim\mathcal{N}(0,1)}[h_{i}(x)]\;,$$

where the above uses Fact B.3 and the moment matching property of A. Since Ex∼N(0,1)[hi(x)] = 1 for i = 0 and Ex∼N(0,1)[hi(x)] = 0 for all i > 0, Equation (11) means that Ex∼B[hi(x)] = Ex∼N(0,1)[hi(x)], i.e., B matches the first m moments with N (0, 1). An application of Proposition B.8 with ν = 0 shows that the NGCA Problem 2.6 that uses the distribution B from above has SQ complexity d Ω(log k). Noting that this problem is equivalent to Problem 1.1 completes the proof of Theorem 1.3.

We conclude by addressing an edge case. The proof above implicitly assumes that the set S contains distinct points (as otherwise, the weights in the corresponding GMM might not all be exactly 1/k). Here, we argue that Theorem 1.3 still holds even if S contains duplicates. Specifically, one can perturb each point in S by a at most an arbitrarily small amount ∆, ensuring that the points become distinct and that the moments in the resulting GMM distribution B are being matched up to an error of ν rather than exactly (note that for any ν we can find a perturbation so that the moment gap is no more than ν). The SQ lower bound from Proposition B.8 then implies that we either require 2 d Ω(1) queries or at least one query with accuracy d
−m/16 + (1 + o*(1))*ν. By choosing ∆ appropriately small, we can ensure that *ν < d*−m/16.

## D. Omitted Details From Section 4 D.1. Omitted Details From Section 4.1

The lemma below shows that if the approximate momement matching condition is violated, then it has to be violated by a monomial (up to a small deterioration of parameters).

Lemma D.1. Let C be a sufficiently large absolute constant. If there exists a polynomial g : R → R of degree r and unit norm (Ex∼N(0,1)[g 2(x)] = 1*) such that*

$$\left|\operatorname{\mathbf{E}}_{x\sim{\cal A}}[g(x)]-\operatorname*{\mathbf{E}}_{x\sim{\mathcal N}(0,1)}[g(x)]\right|>\gamma\;,$$

then there exists a monomial x i with i ≤ r *for which*

$$\left|\mathbf{E}_{x\sim{\cal A}}[x^{i}]-\mathbf{E}_{x\sim{\cal N}(0,1)}[x^{i}]\right|>2^{-C\cdot r}\gamma\ .$$

Proof. We will show this by contradiction. Suppose that every monomial of degree i ≤ r satisfies
Ex∼A[x i] − Ex∼N(0,1)[x i] ≤ 2
−Crγ. Then, if we expand g(x) in the hermite basis, i.e., g(x) = Pr i=1 aihi(x), we have

$$\left|{}_{x\sim A}[g(x)]-{\bf E}_{\sim N(0,1)}[g(x)]\right|\leq\sum_{i=1}^{r}|a_{i}|\left|{\bf E}_{x\sim A}[h_{i}(x)]-{\bf E}_{x\sim N(0,1)}[h_{i}(x)]\right|$$ $$\leq\sqrt{\sum_{i=1}^{r}|a_{i}|^{2}}\sqrt{\sum_{i=1}^{r}\left|{\bf E}_{x\sim A}[h_{i}(x)]-{\bf E}_{x\sim N(0,1)}[h_{i}(x)]\right|^{2}}\,\tag{12}$$

where the second step uses Cauchy-Schwarz inequality. To further upper bound this, first note that pPr i=1 |ai| 2 = ∥g∥2 = 1, by Parseval's identity and our assumption of unit norm. For the other factor above, we can write hi(x) = Pij=1 bijx jand use the fact that |bij | ≤ 2 O(i)(Fact B.1). Then,

$$\begin{array}{l}{{\left|\mathbf{E}_{\!\!\sim\!A}[h_{i}(x)]-\mathbf{E}_{\!\!\sim\!\!A}[h_{i}(x)]\right|\leq\sum_{j=1}^{i}\left|b_{i j}\right|\left|\mathbf{E}_{\!\!\sim\!\!A}[x^{i}]-\mathbf{E}_{\!\!\sim\!\!A}[x^{i}]\right|}}\\ {{\leq2^{O(r)}\sum_{j=1}^{i}\left|\mathbf{E}_{\!\!\sim\!\!A}[x^{i}]-\mathbf{E}_{\!\!\sim\!\!A}[x^{i}]\right|\leq2^{O(r)}r2^{-C r}\gamma<2^{-C r/2}\gamma\;,}}\end{array}$$

14 Plugging that in Equation (12), we obtain | Ex∼A[g(x)] − EN(0,1)[g(x)]| ≤ 
√r2
−Cr/2*γ < γ*, which gives the desired contradiction. The lemma below provides a testing algorithm for the NGCA problem (Problem 2.6) in the special case where the distribution B is k-GMM for which a moment of order at most me is guaranteed to be significantly different than the corresponding moment of N (0, 1). Lemma D.2 (Testing Algorithm for Parallel Pancakes when the m-th Moment Deviates). Let B *be a Gaussian mixture on* R
 *of the form* B =Pk i=1 wiN (µi, σ2), where σ ∈ (0, 1) and wi ≥ wmin for all i ∈ [k]*. For a decreasing sequence* λm, denote by m the biggest integer such that every degree-m′ ≤ m polynomial g *satisfies*

$$\left|\mathbf{E}_{x\sim{\cal{B}}}[g(x)]-\mathbf{\Sigma}_{x\sim{\cal{N}}(0,1)}[g(x)]\right|\leq\lambda_{m}\sqrt{\mathbf{\Sigma}_{x\sim{\cal{N}}(0,1)}[g^{2}(x)]}\;,$$
$$(13)$$

2(x)] , (13)
Consider the non-Gaussian component analysis hypothesis testing Problem 2.6. Let me be any upper bound for m *i.e.,*
m ≤ me . There is an algorithm that takes as input me and wmin*, draws* n =
(md e )
O(me )λ
−O(1)
me + log(k)w
−1 minlog(1/τ )
samples, and distinguishes correctly between H0 and H1 with probability 1 − τ . Moreover, the runtime of the algorithm is polynomial in n and d.

Proof. We will do the proof for me = m. The proof trivially extends to any me bigger than m. For degree m + 1, there exists a polynomial g that violates the condition in Equation (13). By Lemma D.1, there exists a monomial x i with i ≤ m + 1 such that

$$\widetilde{\lambda}=\left|\mathbf{E}_{x\sim B}[x^{i}]-\mathbf{E}_{\mathcal{N}(0,1)}[x^{i}]\right|>2^{-C\cdot m}\lambda_{m}\ .$$

For the corresponding d-dimensional distributions PB,v (defined in Problem 2.6) and N (0, I), we have

$$\mathbf{E}_{x\sim P_{B,v}}[x^{\otimes i}]-\mathbf{E}_{x\sim{\mathcal{N}}(0,I)}[x^{\otimes i}]=\pm{\tilde{\lambda}}v^{\otimes i}\ .$$

Thus, the Frobenius norm is

$$\left\|\operatorname*{\mathbf{E}}_{x\sim P_{B,v}}[x^{\otimes i}]-\operatorname*{\mathbf{E}}_{x\sim\mathcal{N}(0,I)}[x^{\otimes i}]\right\|_{\mathrm{F}}=\widetilde{\lambda}>2^{-C\cdot m}\lambda_{m}\;.$$

This means that at least one entry in the difference of the two tensors has gap at least ϵ := d
−(m+1)2
−C·mλm. The idea for the testing algorithm is to approximate every entry of Ex∼PB,v [x
⊗i] − Ex∼N(0,I)[x
⊗i] up to absolute error ϵ/100, and test whether some entry is bigger than ϵ/2. This is done in Algorithm 2 and Algorithm 3. Algorithm 2 Testing Algorithm 1: **Input**: k, me ∈ Z+, wmin ∈ (0, 1].

2: **Output**: Hˆ ∈ {H0, H1}.

3: for i = 1, 2, 3*, . . . ,* me + 1 do 4: Run Algorithm 3 with input k, i, *m, w* e min repetitively log((me + 1)/τ ) times and let Hˆ be the most frequent output.

5: if Hˆ = H1 **then**
6: Return H1 7: Return H0.

We start with the correctness of the sub-routine, Algorithm 3. We say that that the output of Algorithm 3 is "successful" if it always agrees with the true hypothesis, with the exception of the following case, where mistakes are permitted: this case is when the true hypothesis is H1, the data distribution satisfies maxi∈[k] ∥µi∥2 ≤ C
√d (recall that µi's are the centers of the k-GMM distribution B) andEx∼PB,v [x
⊗i] − Ex∼N(0,I)[x
⊗i]F
≤ 2
−Cmλm. We will show that the output of Algorithm 3 is indeed "successful" in this sense with constant probability.

Algorithm 3 Checking the i-th order tensor mismatch 1: **Input**: k, me ∈ Z+, i ∈ Z+, wmin ∈ (0, 1].

2: **Output**: Hˆ ∈ {H0, H1}.

3: Define n = (md e )
Cme λ
−C
me + log(k)w
−1 min for sufficiently large C.

4: Draw x1*, . . . , x*n i.i.d. from the data distribution.

5: if there exists i ∈ [n] with ∥xi∥2 > C√d **then**
6: Output H1 and terminate. 7: **else** 8: Form the empirical tensor M = Ex∼S[x
⊗i].

9: Let M′ denote the Gaussian tensor Ex∼N(0,I)[x
⊗i].

10: if there is an entry in Mi1*,...,j*i with |Mi1*,...,j*i − M′
i1*,...,j*i | > d−Cme λme **then**
11: Output H1 and terminate.

12: Return H0.

Having that claim established, Lemma D.2 follows straightforwardly: First, note that the probability of success can be amplified to 1 − τ by repeating the subroutine log(1/τ ) times and taking the majority vote. Second, if the true hypothesis is H1, there exists an i such thatEx∼PB,v [x
⊗i] − Ex∼N(0,I)[x
⊗i]F
> 2
−Cmλm. Combined with the claim of the previous paragraph about Algorithm 3, this ensures that running Algorithm 3 for that i will be H1, as desired. Similarly, under H0, the output is always H0, which guarantees that the output of Algorithm 2 matches the true hypothesis.

We now move to showing the claim that Algorithm 3 is "successful" with constant probability. We examine the following cases:
Case 1 The true hypothesis is H0. In this case, the data distribution is D = N (0, I). By Gaussian norm concentration
(Fact B.6) we have Prx1*,...,x*n∼N(0,I)[maxi ∥xi∥ > C√d log n] < 0.01. This means that Algorithm 3 will enter line 7.

Then, by standard entry-wise concentration of Gaussian tensors (see e.g., Fact 5.6 and Equation (5.4) in (Kothari & Steurer, 2017)) we have that if *n > d*C
′m/λ2m for C
′ ≫ C, we will have ∥ Ex∼N(0,I)[x
⊗i] − Ex∼S[x
⊗i]∥∞ < d−Cmλm and thus the condition in 10 will be false, resulting in the algorithm to output H0.

Case 2 The hypothesis under effect is H1 and maxi∈[k] ∥µi∥2 > C√d log n. The claim is that for log(k)/wmin samples, with high constant probability, one sample from every component will be observed, and the sample that comes from the component with ∥µi∥2 > C√d log n will satisfy ∥x∥ > C√d log n by Fact B.6. To see the first part of the claim, fix an i ∈ [k]. With 10/wmin samples, one sample from i will be observed with at least 0.9 probability. We can boost that probability to 1 − 1/k by repeating log(k) times. Then, by union bound, this means that one sample from each component is observed with constant probability.

Case 3 The hypothesis under effect is H1 and maxi∈[k] ∥µi∥2 ≤ C
√d log n. In this case the data distribution is a k-GMM where the center of every Gaussian component is bounded in norm by most R = C
√d log n. By Gaussian norm concentration, if x1*, . . . , x*n are points drawn from that GMM, then with constant probability we will have ∥xi∥2 ≤
2C
√d log n. Therefore the algorithm will enter 7, and because of the bound ∥xi∥2 ≤ 2C
√d log n, we can treat the distribution as bounded and use Hoeffding bound for the tensor concentration. That application of Hoeffding's inequality shows that if *n > R*C
′md C
′m/λC
′
m then the estimation error is at most d
−Cmλm. Thus, in this case, the algorithm will output H1 if and only ifEx∼A[x
⊗i] − Ex∼N(0,1)[x
⊗i]F
≤ 2
−Cmλm.

This completes the proof of the claim. Our main result, Theorem 1.4 will be based on Lemma D.2 and our impossibility of matching result, Proposition 4.2 that will allow us to use me = O(log(k) + k
′) in Lemma D.2. However, Proposition 4.2 concerns only discrete distributions, while the parallel pancakes uses a Gaussian mixture. In order to bridge this difference, we show the following lemma, which states that the impossibility of moment matching can be indeed extended to Gaussian mixtures.

Lemma D.3. Let P be a Gaussian mixture distribution on R *of the form* B =Pk i=1 wiN (µi, 1 − δ)*, where* wi > 0 with Pk i=1 wi = 1 are the weights of each component, µ1, . . . , µk ∈ R are the centers and δ ∈ (0, 1] *is the parameter* associated with the common variance of the components. Suppose that for every polynomial of degree at most m′ and Ex∼N(0,1)[p 2(x)] = 1 *the following holds*

$$\left|\mathop{\bf E}_{x\sim B}[p(x)]-\mathop{\bf E}_{x\sim{\cal N}(0,1)}[p(x)]\right|\leq\lambda\;.\tag{1}$$
$$(14)$$

Then, if A denotes the discrete distribution on {µ1/
√*δ, . . . , µ*i/
√δ} that assigns mass wi*to the point* µi/
√δ for i ∈ [k],
the following is true: For every polynomial with degree at most m′ and Ex∼N(0,1)[p 2(x)] = 1 *it holds*

$$\left|\mathop{\bf E}_{x\sim A}[p(x)]-\mathop{\bf E}_{x\sim{\cal N}(0,1)}[p(x)]\right|\leq\sqrt{m^{\prime}}\,\lambda\,\delta^{-m^{\prime}/2}\;.$$

Proof. We can write the distribution B as the result of applying the Ornstein-Uhlenbeck (Definition B.2) operator to A, i.e., B = UρA with ρ =
√δ. By Fact B.3, we have the following for every i = 1, 2*, . . .*:

$$(15)$$
$$\mathbf{E}_{x\sim A}[h_{i}(x)]=\delta^{-i/2}\;\mathbf{E}_{x\sim B}[h_{i}(x)]\;.$$
$$(16)$$
[hi(x)] . (16)
Fix i ∈ [m′]. Using the above and the fact that B matches approximately the m first moments with N (0, 1) (in the sense of Equation (14)) we have the following for the gap between the expectations of the Hermite polynomial hi under A and N (0, 1):

 E x∼A [hi(x)] − E x∼N(0,1) [hi(x)]  =  δ −i/2 E x∼B [hi(x)] − E x∼N(0,1) [hi(x)] (using Equation (16)) = δ −i/2 E x∼B [hi(x)]  (using Ex∼N(0,1)[hi(x)] = 0 for i ≥ 1) ≤ δ −i/2E x∼N(0,1) [hi(x)]  + λ (using Equation (14)) = δ −i/2λ (using Ex∼N(0,1)[hi(x)] = 0 for i ≥ 1)
For the special case i = 0, we have exact matching, Ex∼A[h0(x)] = Ex∼N(0,1)[h0(x)] since h0(x) = 1. Now, in order to show Equation (15), consider a general polynomial p(x) with Ex∼N(0,1)[p 2(x)] = 1. Expanding in the Hermite basis, we can write p(x) = Pi∈[m′]
aihi(x) with Pi∈[m′]
a 2 i = 1 (which means that Ex∼N(0,1)[p 2(x)] = 1 by Parseval's identity). We have

$$\left|\mathbf{E}_{x\sim A}[p(x)]-\mathbf{E}_{x\sim\mathcal{N}(0,1)}[p(x)]\right|\leq\sqrt{\sum_{i=1}^{m^{\prime}}a_{i}^{2}}\sqrt{\sum_{i=1}^{m^{\prime}}\left|\mathbf{E}_{x\sim A}[h_{i}(x)]-\mathbf{E}_{x\sim\mathcal{N}(0,1)}[h_{i}(x)]\right|^{2}}$$ $$\leq\sqrt{m^{\prime}}\ \max_{i\in[m^{\prime}]}\left|\mathbf{E}_{\sim A}[h_{i}(x)]-\mathbf{E}_{x\sim\mathcal{N}(0,1)}[h_{i}(x)]\right|$$ $$\leq\sqrt{m^{\prime}}\ \delta^{-m^{\prime}/2}\ \lambda\.$$

We now combine the previous statements to show our main theorem.

Theorem 1.4 (Testing Algorithm for Parallel Pancakes). Consider the version of the parallel pancakes hypothesis testing problem (Problem *1.1), where* k
′ ≤ k of the weights wiin the Gaussian mixture are unconstrained and the remaining k − k
′ *are assumed to be equal to each other. There is an algorithm for that problem which draws* n = O
(kd/δ)
O(k
′+log(k)) + log(k)/wminsamples (where δ is as in Problem 1.1 and wmin = mini∈[k] wi*is the* smallest weight), has runtime polynomial in n, d, and it outputs the correct hypothesis with probability at least 0.99. Proof. First, we note that the parallel pancakes testing problem of interest is a special case of the non-Gaussian component analysis Problem 2.6 where B =Pi∈[k] wiN (µi, 1 − δ), where wi, µi and δ the ones from Problem 1.1, in particular k
′ of the wi's are unconstrained and the rest are assumed to be uniform.

The proof consists of two parts: The first part argues that this one-dimensional distribution B cannot match approximately more than the first m = O(log k +k
′) moments with N (0, I) (the approximate moment matching will be quantified shortly).

Then, for the second part, we can show that since the m + 1 moment deviates significantly from that of N (0, I), estimating the empirical moment tensor of order m + 1 and comparing with the one from N (0, I) yields a successful test. We now proceed with the quantification. Let C be a sufficiently large constant, and define m to be the largest integer such that for every polynomial p of degree m′ ≤ m and ∥p∥2 = 1 we have

$$\left|\mathop{\bf E}_{x\sim A}[p(x)]-\mathop{\bf E}_{x\sim{\cal N}(0,1)}[p(x)]\right|\leq\lambda_{m}\.$$
$$(17)$$
$$(18)$$

To prove our claim by contradiction, suppose that *m > C*(k
′ + log k). For each degree m′ ≤ m We will use Lemma D.3 with λ = (δ/2)Cm. The application of Lemma D.3 yields that the discrete distribution A supported on a scaled version of the centers µi and using the same weights wi approximately matches the same m first moments with N (0, 1), i.e., for every polynomial p of degree m′ ≤ m and ∥p∥2 = 1 we have

$$\left|\mathop{\bf E}_{x\sim D}[p(x)]-\mathop{\bf E}_{x\sim{\cal N}(0,1)}[p(x)]\right|\leq\sqrt{m}\lambda_{m}\delta^{-m/2}\leq2^{-C m/2}\;.$$

where the last step uses that λ *= (2*δ)
−Cm.

The conclusion of Equation (18) contradicts Proposition 4.1. This is because the discrete distribution D from above, is of the form that Proposition 4.1 considers: supported on k points, with k
′ of the points having arbitrary mass and the remaining k − k
′ having equal masses.

Thus far, we have shown that if m is the largest degree for which all moments m′ ≤ m of the distribution A match with N (0, 1) in the sense of Equation (17), then m = O(log(k) + k
′).

The result then follows by Lemma D.2 with me = C(log(k) + k
′) for a sufficiently large constant C, σ 2 = 1 − δ, and λme = (δ/2)Cme.

## D.2. Omitted Details From Section 4.2

We restate and prove a version of Proposition 4.2 which does not involve minimum weight w0 of points with equal weights.

Proposition 4.1. Let k
′ < k be positive integers, and let A be a discrete distribution on k points in R*. Suppose* k − k
′ of the points have equal probability masses, while the remaining k
′ points have unrestricted probability masses. Denote by m the highest degree for which every degree-m′ ≤ m polynomial g *satisfies*Ex∼A[g(x)] − Ex∼N(0,1)[g(x)] ≤ 2
−C·m∥g∥2, then m *must satisfy* m ≤ O(log k) + O(k
′).

Proof. Suppose that the order m is bigger than C log k + Ck′. If C is sufficiently large, we will show that this moment matching is impossible. Let µ1*, . . . , µ*k be the points on which A is supported, and by w1*, . . . , w*k the probability masses of the points. Without loss of generality, assume that the first k
′ points are the ones which do not have any restriction on their probability mass, and the remaining k − k
′are the points with equal probability masses (wi = wj for all i, j ∈ {k
′ + 1*, . . . , k*} with i ̸= j). Let p(x) = (x − µ1)*· · ·*(x − µk′ ) be the polynomial whose roots are the first k
′ points.

We will show the following series of inequalities (we use the notation ∥p∥r = Ex∼N(0,1)[|p(x)| r]
1/r):

$$\sum_{i=k^{\prime}+1}^{k}w_{i}\geq\left(\frac{{\bf E}_{x\sim A}[p(x)]}{{\bf E}_{x\sim A}[p^{2}(x)]^{1/2}}\right)^{2}\asymp\left(\frac{\|p\|_{1}}{\|p\|_{2}}\right)^{2}\geq3^{-2k^{\prime}}\.$$

$\square$
$$(19)^{\frac{1}{2}}$$

The third step is Fact 2.4. To see how the first step is derived, let E be the event that i ∈ {k
′ + 1*, . . . , k*}. Then

$$\|p\|_{1}=\mathop{\mathbb{E}}_{\mu_{1}\sim A}[p(\mu_{i})]=\mathop{\mathbb{E}}_{\mu_{1}\sim A}[p(\mu_{i})1(\mathcal{E})]\leq\sqrt{\mathop{\mathbb{E}}_{\mu_{1}\sim A}[p^{2}(\mu_{i})]}\sqrt{\mathbf{Pr}[\mathcal{E}]}=\|p\|_{2}\sqrt{\mathbf{Pr}[\mathcal{E}]}=\|p\|_{2}\sqrt{\sum_{i=k^{\prime}+1}^{k}w_{i}}\;,$$

where the first step above uses the fact that µ1*, . . . , µ*k′ are roots of p. Rearranging gives Pk i=k′ wi ≥ (∥p∥1/∥p∥2)
2.

It remains to show the second step in Equation (19), which is due to the approximate moment matching property: Let λm := 2−Cm to save space. For the numerator, we have Ex∼A[p(x)] ≥ ∥p∥1 − λm∥p∥2 ≥ ∥p∥1(1 − λm3 k
′) ≥ ∥p∥1/2, where the first step uses the approximate moment matching, the second step uses Fact 2.4 and the last part uses that λm := w02
−Cm with C being sufficiently large constant and *m > k*′. We can work similarly for the denominator to get Ex∼A[p 2(x)] ≤ ∥p∥
22 + λm∥p∥
24 ≤ ∥p∥
22(1 + λm3 k
′) ≤ 2∥p∥
22, where we used Fact 2.3 in the penultimate step. Combining the bounds for numerator and denominator conclude the proof of the second step in Equation (19). We can now conclude the proof of Theorem 1.4. Since we have assumed that the weights for the last k − k
′ points are equal to each other, Equation (19) implies that mini=k′+1*,...,k* wi ≥ 3
−2k
′/k. Using Proposition 4.2 with w0 = 3−2k
′/k concludes the proof.

## D.3. Omitted Details From Section **4.2.1**

We restate and prove the following corollary of Lemma 4.4.

Corollary 4.5. Let p : R → R *be a polynomial of the form* p(x) = (x − µ1)(x − µ2)· · ·(x − µk′ ) where µ1*, . . . , µ*k′ ∈ R
are arbitrary parameters. Define I = [0.9
√2t, 1.1
√2t]. For all t ≥ 1 *we have* exp 1 |I| Rx∈I
ln |p(x)|dx
≥∥p∥2 2O(k′).

Proof. We can multiply both sides of the conclusion of Lemma 4.4 (Equation (5)) with the Gaussian density e
−y 2/2/
√2π and then integrate both sides. This yields

$$\int_{-\sqrt{t}}^{\sqrt{t}}\frac{1}{\sqrt{2\pi}}e^{-y^{2}/2}\exp\left(\frac{1}{|I|}\int_{x\in I}\ln|p(x)|\mathrm{d}x\right)\mathrm{d}y\geq\int_{-\sqrt{t}}^{\sqrt{t}}\frac{|p(y)|}{2^{O(k^{\prime})}}\frac{1}{\sqrt{2\pi}}e^{-y^{2}/2}\mathrm{d}y\.$$  We is $\Theta\left(\exp\left((1/|I|)\int_{x\in I}\ln|p(x)|\mathrm{d}x\right)\right).$ The right hand side is 
The left hand side is Θexp (1/|I|)Rx∈I
1 √2π Z  √t − √t e −y 2/2 |p(y)| 2O(k′) dy = E y∼N(0,1) [|p(y)|] − E y∼N(0,1) [|p(y)|1(|y| > √t)]2 −O(k ′) ≥   E y∼N(0,1) [|p(y)|] − ∥p∥2 rPr y∼N(0,1) [|y| > t] ! 2 −O(k ′)(using the Cauchy–Schwarz inequality) ≥ E y∼N(0,1) [|p(y)|] − ∥p∥2e −t 2/22 −O(k ′) ≥ ∥p∥2 − ∥p∥2e −t 2/22 −O(k ′)(using Fact 2.4) = ∥p∥2/2 O(k ′). (using t ≥ 1)
We now restate Lemma 4.4 and provide the complete proof which includes the details that were missing from Section 4.2.1.

Corollary 4.5. Let p : R → R *be a polynomial of the form* p(x) = (x − µ1)(x − µ2)· · ·(x − µk′ ) where µ1*, . . . , µ*k′ ∈ R
are arbitrary parameters. Define I = [0.9
√2t, 1.1
√2t]. For all t ≥ 1 *we have* exp 1 |I| Rx∈I
ln |p(x)|dx
≥∥p∥2 2O(k′).

Proof. Fix an arbitrary y ∈ R with |y| ≤ 
√t. First, note that by the property of logarithms and sums, we can write the left hand side as

$$\exp\left(\sum_{i=1}^{k^{\prime}}{\frac{1}{|I|}}\int_{x\in I}\ln|x-\mu_{i}|\mathrm{d}x\right)$$
 .
19 In order to show Equation (5), it suffices to work with each term and show the following for each i ∈ [k
′]:

$${\frac{1}{|I|}}\int_{x\in I}\ln|x-\mu_{i}|\geq\ln|y-\mu_{i}|-O(1)\;.$$
$$(20)$$

Equivalently, it suffices to show that Equation (5) holds for every linear polynomial of the form p(x) = x − a. Therefore, the goal for the rest of this proof is to show that

$$\exp\left(\frac{1}{|I|}\int_{x\in I}\ln|x-a|\mathrm{d}x\right)\geq|y-a|/O(1)\;,$$

holds for every a ∈ R and y ∈ R with |y| ≤ 
√t. We will examine two cases.

Case 1 The first case is when the root a of the polynomial is outside the interval I. In this case, we can show that |x − a|/|y − a| = Θ(1), which implies ln |x − a| ≥ ln |y − a| − O(1), and the desired conclusion (Equation (20)) follows by integrating both sides and applying the exp(·) function. To show the earlier claim that |x − a|/|y − a| = Θ(1), we can consider the following sub-cases:
1. Case a ≥ 1.1
√2t (i.e., a is to the right of I): Suppose a = 1.1
√2t + u for some non-negative u. Then, a − x =
(1.1
√2t − x) + u = Θ(√t) + u and a − y = (1.1
√2t − y) + u = Θ(√t) + u. Therefore, for any u ≥ 0, the ratio |x − a|/|y − a| = (Θ(√t) + u)/(Θ(√t) + u) = Θ(1).

2. The cases a < −
√t and a ∈ [
√t, 0.9
√2t] can be shown in a similar manner.

Case 2 The complementary case is when the root a of the polynomial p belongs in the interval I. In that case,

$${\frac{1}{|I|}}\int_{x\in I}\!\!\ln|x\!-\!a|\mathrm{d}x={\frac{1}{|I|}}\int_{a}^{1.1{\sqrt{2t}}}\!\!\ln(x\!-\!a)\mathrm{d}x+{\frac{1}{|I|}}\int_{0.9{\sqrt{2t}}}^{a}\!\!\ln(a\!-\!x)\mathrm{d}x\ .$$

Define A := 1 0.2
√2t R 1.1
√2t aln(x − a)dx and B := 1 0.2
√2t R a 0.9
√2t ln(a − x)dx. We will work with each integral separately.

For A, we have the following (after a change of variable in the integral):

$$A=\frac{1}{0.2\sqrt{2t}}\int_{0}^{1.1\sqrt{2t}-a}\ln z\;\mathrm{d}z=\frac{1}{0.2\sqrt{2t}}\left[-z+z\ln z\right]_{z=0}^{z=1.1\sqrt{2t}-a}$$ $$=-\left(5.5-\frac{a}{0.2\sqrt{2t}}\right)+\left(5.5-\frac{a}{0.2\sqrt{2t}}\right)\ln\left(1.1\sqrt{2t}-a\right).$$

Recalling that we have assumed a ∈ [0.9
√2t, 1.1
√2t], we can rewrite the above as A = −C1 + C1 ln(1.1
√2t − a), where C1 = 5.5 −a 0.2
√2t
∈ [0, 1].

We now work with the integral defined as A previously in a similar way:

$$B=\frac{1}{0.2\sqrt{2t}}\int_{0}^{a-0.9\sqrt{2t}}\ln z\;\mathrm{d}z=\frac{1}{0.2\sqrt{2t}}\left[-z+z\ln z\right]_{z=0}^{z=a-0.9\sqrt{2t}}$$ $$=-\left(\frac{a}{0.2\sqrt{2t}}-4.5\right)+\left(\frac{a}{0.2\sqrt{2t}}-4.5\right)\ln(a-0.9\sqrt{2t})\;.$$

Taking into consideration that a ∈ [0.9
√2t, 1.1
√2t] the above can be written as B = −C2 + C2 ln(a − 0.9
√2t), where C2 =a 0.2
√2t
− 4.5 ∈ [0, 1].

Combining the bounds for A and B together with the definitions C1 = 5.5 −a 0.2
√2t and C2 =a 0.2
√2t
− 4.5, we obtain exp(A + B) = exp(f(a) − 1), where f(a) is the function

$$f(a):=\left(5.5-{\frac{a}{0.2{\sqrt{2t}}}}\right)\ln(1.1{\sqrt{2t}}-a)+\left({\frac{a}{0.2{\sqrt{2t}}}}-4.5\right)\ln(a-0.9{\sqrt{2t}})\;,$$