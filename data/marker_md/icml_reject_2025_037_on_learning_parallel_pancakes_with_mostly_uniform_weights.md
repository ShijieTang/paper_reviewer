# On Learning Parallel Pancakes with Mostly Uniform Weights

Ilias Diakonikolas <sup>1</sup> Daniel M. Kane <sup>2</sup> Sushrut Karmalkar 3 4 Jasper C.H. Lee <sup>5</sup> Thanasis Pittas <sup>1</sup>

# Abstract

We study the complexity of learning k-mixtures of Gaussians (k-GMMs) on R d . This task is known to have complexity d Ω(k) in full generality. To circumvent this exponential lower bound on the number of components, research has focused on learning families of GMMs satisfying additional structural properties. A natural assumption posits that the component weights are not exponentially small and that the components have the same unknown covariance. Recent work gave a d <sup>O</sup>(log(1/wmin))-time algorithm for this class of GMMs, where wmin is the minimum weight. Our first main result is a Statistical Query (SQ) lower bound showing that this quasi-polynomial upper bound is essentially best possible, even for the special case of uniform weights. Specifically, we show that it is SQ-hard to distinguish between such a mixture and the standard Gaussian. We further explore how the distribution of weights affects the complexity of this task. Our second main result is a quasi-polynomial upper bound for the aforementioned testing task when most of the weights are uniform while a small fraction of the weights are potentially arbitrary.

# 1. Introduction

Learning mixture models in high dimensions is a classic and fundamental task with applications in a plethora of domains, such as bioinformatics, astrophysics, and marketing [\(Lind](#page-9-0)[say,](#page-9-0) [1995;](#page-9-0) [Garc´ıa-Escudero et al.,](#page-9-1) [2010\)](#page-9-1); see [Titterington](#page-9-2) [et al.](#page-9-2) [\(1985\)](#page-9-2) for an extensive list of applications. The prototypical case is that of Gaussian Mixture Models (GMMs)

which is one of the most studied problems in statistics and machine learning, with a large body of research over the past few decades, e.g., [Vempala & Wang](#page-9-3) [\(2002\)](#page-9-3); [Kannan et al.](#page-9-4) [\(2005\)](#page-9-4); [Achlioptas & McSherry](#page-8-0) [\(2005\)](#page-8-0) — see Appendix [A](#page-10-0) for a detailed literature review.

The setup is as follows: the learning algorithm observes i.i.d. samples from a k-component GMM model (k-GMM) in R d , P = P<sup>k</sup> <sup>i</sup>=1 wiN (µ<sup>i</sup> , Σi), and the goal is to either learn the mixture in total variation distance, learn its parameters, or cluster samples from the GMM correctly. The first task is known to be information-theoretically feasible with poly(d, k) samples, as are the second and third, provided the components are sufficiently well-separated, however known algorithms often require more. While the particular case of spherical mixtures (i.e., Σ<sup>i</sup> = I) can be learned in poly(d, k) time and samples, [\(Liu & Li,](#page-9-5) [2022;](#page-9-5) [Diakonikolas](#page-8-1) [& Kane,](#page-8-1) [2024\)](#page-8-1), the best-known algorithms for learning arbitrary GMMs (i.e. with arbitrary weights, and arbitrary and different component covariances) have sample complexity that scales with d O(k) [\(Bakshi et al.,](#page-8-2) [2022\)](#page-8-2). In this paper, we are concerned with an intermediate regime between the two extremes, where the components share an unknown but common covariance matrix.

[Diakonikolas et al.](#page-8-3) [\(2017\)](#page-8-3) showed that for such mixtures, any sub-exponential time algorithm in the *Statistical Query* (SQ) model requires a sample complexity of d Ω(k) . The SQ model consists of algorithms that, instead of drawing samples from the data distribution, make queries to approximate expectations of bounded functions (formally defined in Definition [1.2\)](#page-1-0).

The hard instances they proposed are "parallel pancakes" GMMs—mixtures of pairwise-separated Gaussians whose component means are collinear along an unknown direction v, with arbitrary variance in the v-direction and identity covariance in the orthogonal subspace. This will be formally defined in Problem [1.1.](#page-1-1) [Bruna et al.](#page-8-4) [\(2021\)](#page-8-4); [Gupte et al.](#page-9-6) [\(2022\)](#page-9-6) further extended the hardness result to general algorithms but under cryptographic assumptions; similar hardness results were also shown for sum-of-squares algorithms [\(Diakonikolas et al.,](#page-9-7) [2024\)](#page-9-7).

While these results together might suggest that k-GMM learning is fully understood algorithmically, the current theory remains unsatisfactory, in the following sense: the

Author names in alphabetical order <sup>1</sup>Department of Computer Sciences, University of Wisconsin Madison, Madison, United States <sup>2</sup>University of California San Diego, San Diego, United States <sup>3</sup>Microsoft Research, Cambridge, England <sup>4</sup> Some of this work was done while the author was a postdoctoral researcher at UW-Madison. <sup>5</sup>University of California Davis, Davis, United States. Correspondence to: Thanasis Pittas <pittas@wisc.edu>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

hard instances developed in [Diakonikolas et al.](#page-8-3) [\(2017\)](#page-8-3) have rather ill-conditioned mixing weights—some of the mixing weights are 1/ poly(k), but others can be as small as 2 −k . A natural question then is: is it possible to improve the complexity of learning algorithms when all weights are more naturally conditioned, i.e., w<sup>i</sup> ≥ 1/ poly(k) for all i?

This question was considered in [Buhai & Steurer](#page-8-5) [\(2023\)](#page-8-5); [Anderson et al.](#page-8-6) [\(2024\)](#page-8-6), which study GMMs that have a minimum mixing weight wmin ≥ 1/ poly(k) and unknown but common covariance across components. Under the assumption that the mixture components are separated in total variation distance, they provide an algorithm that can correctly cluster 99% of the points, using time and sample complexity d log(1/wmin) ≤ d O(log k) . In particular, their results apply to parallel pancake instances, showing that it is possible to circumvent the d Ω(k) (SQ) lower bound under mixing weight assumptions.

These prior results on learning mixtures with restricted mixing weights serve as motivation and the starting point of the present work. In particular, the first question we study is:

*Is it possible to substantially improve the algorithm of [Anderson et al.](#page-8-6) [\(2024\)](#page-8-6) to a* poly(d, k) *time algorithm for parallel pancakes when each* w<sup>i</sup> ≥ 1/ poly(k)*?*

Our first main result rules out this possibility for SQ algorithms. Specifically, we show in Theorem [1.3](#page-1-2) that even when the mixing weights are uniform, any SQ algorithm for such instances requires d Ω(log k) complexity. In fact, the lower bound holds even for the more basic task of distinguishing between a k-GMM from that family and N (0, I).

Our second question stems from the fact that the algorithm in [Anderson et al.](#page-8-6) [\(2024\)](#page-8-6) has complexity d log(1/wmin) , meaning that a single point with arbitrarily small weight (e.g., 2 −k ) can result in d k complexity.

*What is the correct complexity dependence on* wmin *for learning* k*-component parallel pancakes?*

Specifically, we consider again the testing problem of distinguishing between a k-parallel-pancake GMM and N (0, I), but where k ′ ≤ k components can have arbitrary weights while the remaining k − k ′ points must have uniform weights. We show that this mixing weight restriction implies that the testing problem can be solved with time and sample (kd) O(k ′+log <sup>k</sup>) + (log <sup>k</sup>)/wmin — an inverse-linear dependence on wmin instead of quasi-polynomial as suggested by the [Anderson et al.](#page-8-6) [\(2024\)](#page-8-6) result. While this testing upper bound does not imply a general learning algorithm k-GMM, it serves as a first step in understanding the nuances of the computational landscape of GMMs with respect to the assumptions on the mixing weights.

The technical core for both our main results is to deter-

mine the maximum number m of moments that k-parallelpancake GMMs can match with N (0, I). Our SQ lower bound (Theorem [1.3\)](#page-1-2) comes from showing that m = Ω(log k), by employing a result from design theory. Our second, algorithmic result (Theorem [1.4\)](#page-2-0) critically builds on an impossibility-of-moment-matching argument (Proposition [4.1\)](#page-5-0), showing that if there are k ′ ≤ k arbitrary weights in the k-GMM, then m must be O(log(k) + k ′ ). We show this through a novel proof strategy that bounds the ratio of expectations of appropriately chosen non-negative polynomials that vanish on the points with arbitrary weights.

#### 1.1. Our Results

We first formally state the hypothesis testing problem which requires the algorithm to distinguish between a k-parallel pancake and the standard Gaussian N (0, I).

Problem 1.1 (Parallel Pancakes Testing Problem). One has (i.i.d. sample or SQ) access to a distribution D where either:

- (Null Hypothesis) D = N (0, I).
- (Alternative Hypothesis) D is a Gaussian mixture of the form P <sup>i</sup>∈[k] wiN (vµ<sup>i</sup> , I − δvv<sup>⊤</sup>), for some unit vector v ∈ S<sup>d</sup>−<sup>1</sup> , centers µ<sup>i</sup> ∈ <sup>R</sup>, and weights w<sup>i</sup> ≥ 0 for i ∈ [k], with P <sup>i</sup>∈[k] w<sup>i</sup> = 1. That is, D is a k-GMM with collinear centers and variance 1 − δ along the direction of the centers and 1 in every orthogonal direction.

The goal is to distinguish between the two cases.

Before presenting our first main result, we recall the definition of SQ algorithms. These algorithms, instead of directly accessing samples, query expectations of bounded functions of the distribution. The SQ model, introduced in [\(Kearns,](#page-9-8) [1998\)](#page-9-8), has since been extensively studied in various contexts [\(Feldman,](#page-9-9) [2016\)](#page-9-9). Many supervised learning algorithms, and several known machine learning techniques are implementable using SQs [\(Feldman et al.,](#page-9-10) [2017a](#page-9-10)[;b\)](#page-9-11).

Definition 1.2 (STAT Oracle). Let D be a distribution on R d . A statistical query is a bounded function f : R <sup>d</sup>→[−1, 1]. Given f and an accuracy parameter τ > 0, STAT(τ ) returns a v ∈ <sup>R</sup> such that |v − Ex∼D[f(x)]| ≤ τ .

Since a call to STAT(τ ) can be simulated in the standard PAC model by averaging 1/τ <sup>2</sup> samples, τ serves as the SQ model's analog to sample complexity. An *informationcomputation* tradeoff in the SQ model states that any SQ algorithm for a given problem must either make a large number of queries or at least one query with very fine accuracy (which informally implies a tradeoff between sample complexity and runtime in the standard PAC model). We are now ready to state our first main result.

Theorem 1.3 (SQ Lower Bound for Uniform Weights). *Let* C *be a sufficiently large absolute constant,* k > C *and* d ≥ (log k log d) <sup>2</sup> *be integers. If we further restrict the* *alternative hypothesis in Problem [1.1](#page-1-1) to have* w<sup>i</sup> = 1/k *for all* i ∈ [k]*, any SQ algorithm requires either* 2 d Ω(1) *queries or at least one query of accuracy* d −Ω(log k) *.*

Remarks [Buhai & Steurer](#page-8-5) [\(2023\)](#page-8-5); [Anderson et al.](#page-8-6) [\(2024\)](#page-8-6) presented an algorithm for solving Problem [1.1](#page-1-1) using d O(log k) time and samples (e.g., Theorem 1.1 in the first paper, which was the first to achieve this). Our Theorem [1.3](#page-1-2) shows that this complexity is best possible. Notably their work requires the components to be statistically separated, but this is something that we can also ensure by taking δ sufficiently small (since δ does not affect the complexity lower bound).

We now move to our second main result.

Theorem 1.4 (Testing Algorithm for Parallel Pancakes). *Consider the version of the parallel pancakes hypothesis testing problem (Problem [1.1\)](#page-1-1), where* k ′ ≤ k *of the weights* w<sup>i</sup> *in the Gaussian mixture are unconstrained and the remaining* k − k ′ *are assumed to be equal to each other. There is an algorithm for that problem which draws* n = O (kd/δ) O(k ′+log(k)) + log(k)/wmin *samples (where* δ *is as in Problem [1.1](#page-1-1) and* wmin = mini∈[k] w<sup>i</sup> *is the smallest weight), has runtime polynomial in* n, d*, and it outputs the correct hypothesis with probability at least* 0.99*.*

The algorithm is based on estimating the first O(k ′ + log k) moment tensors through the empirical tensors, and thus it is also naturally expressible in the SQ model.

Remarks A single component with arbitrarily small weight can make the complexity in [\(Buhai & Steurer,](#page-8-5) [2023;](#page-8-5) [Anderson et al.,](#page-8-6) [2024\)](#page-8-6) blow up quasipolynomially. By contrast, our algorithm can handle any number of such points, and the complexity interpolates smoothly between the alluniform and the fully general weights cases.

## 1.2. Overview of Techniques

For Theorem [1.3,](#page-1-2) it suffices to show existence of a onedimensional distribution (corresponding to the projection along the hidden direction v in the parallel pancakes mixture in Problem [1.1\)](#page-1-1) that matches a lot of moments with N (0, 1) and is thus hard to distinguish. Concretely, the goal is to show the existence of a set S ⊂ R of size k such that Ex∼S[x i ] = Ex∼N(0,1)[x i ] for all i = 1, . . . , t, where t = Ω(log k) and x ∼ S denotes the uniform distribution on S. Once established, the theorem follows from standard SQ theory: convolving this discrete distribution with a narrow Gaussian yields a k-GMM B that still matches the first t moments with N (0, 1). A standard result from [\(Diakonikolas et al.,](#page-8-7) [2023\)](#page-8-7) then shows that hiding B along an unknown direction is hard to distinguish from N (0, I).

Fortunately, the desired moment-matching construction,

known as a t-design, has been well-studied. [Kane](#page-9-12) [\(2015\)](#page-9-12) shows that designs of small size to match the moments of a distribution Q exist when the support of Q is "pathconnected". The design's size is upper bounded by the number K, which is defined to be the supremum of the ratio supx∈<sup>X</sup> <sup>p</sup>(x) | infx∈<sup>X</sup> p(x)| taken over all degree-t zero-mean polynomials p. Thus, it suffices to show K = 2O(t) to prove Theorem [1.3.](#page-1-2) However, since the Gaussian distribution has unbounded support, there are (many) polynomials p where supx∈<sup>R</sup> p(x) is infinite while the infimum is clearly finite.

To address this, we can instead consider another distribution Q supported on an interval I of length O( √ t), which also matches the first t moments with N (0, 1) (Lemma [3.3](#page-4-0) from [Diakonikolas et al.](#page-8-3) [\(2017\)](#page-8-3)). Thus, by creating a design to match Q we only need to bound K with X = I, which is now possible (Lemma [3.5\)](#page-4-1). Specifically, by expressing p in the Hermite basis, we can show that supx∈<sup>I</sup> p(x) ≤ Ex∼N(0,1)[|p(x)|]2<sup>O</sup>(t) . Additionally, by Gaussian anti-concentration, we can show that any zero-mean polynomial p of degree t has a 2 <sup>−</sup>O(t) probability of being less than −2 <sup>−</sup>O(t) Ex∼N(0,1)[|p(x)|]. This shows that |infx∈<sup>I</sup> p(x)| ≥ Ex∼N(0,1)[|p(x)|]2<sup>−</sup>O(t) .

We can also show that this bound is tight: if A is the uniform distribution on k points, it cannot match more than O(log k) moments with N (0, 1). The argument is that for any non-negative function f, Ex∼A[f (x)] <sup>E</sup>x∼A[f(x)]<sup>2</sup> ≤ <sup>k</sup>. If A matches the first 4t moments with N (0, 1), setting f(x) = x <sup>2</sup><sup>t</sup> makes the ratio 2 Ω(t) , implying t = O(log k).

In fact, we can extend the result to non-uniform distributions where all but k ′ points in the support have equal weight, showing that such distributions cannot match more than O(log(k)+k ′ ) moments with N (0, 1) (Proposition [4.1\)](#page-5-0). As we will explain later, this will lead to the testing algorithm in Theorem [1.4.](#page-2-0)

The first step towards Proposition [4.1](#page-5-0) is to show that, if all but k ′ points have weight at least w0, then it is impossible to match more than O(log(1/w0)+k ′ ) moments with N (0, 1) (Proposition [4.2\)](#page-5-1). The proof relies on extending the idea of the previous paragraph that any non-negative function f which vanishes (i.e. gives value zero) on the k ′ points in question satisfies <sup>E</sup>x∼A[<sup>f</sup> (x)] <sup>E</sup>x∼A[f(x)]<sup>2</sup> <sup>≤</sup> <sup>1</sup>/w<sup>2</sup> 0 . We specifically use f(x) = x <sup>t</sup>p(x), where p(x) = (x − µ1) 2 . . .(x − µk′ ) 2 and µ1, . . . , µk′ are the points in the support of A with unrestricted weights. The goal then is to show that the ratio r := Ex∼A[f (x)] <sup>E</sup>x∼A[f(x)]<sup>2</sup> is at least 2 Ω(t−k ′ ) — combining this with our earlier lower bound r ≤ 1/w<sup>2</sup> implies t = O(log(1/w0)+k ′ )). To bound r, we assume A matches Θ(t + k ′ ) moments, allowing us to replace Ex∼A[·] with Ex∼N(0,1)[·] in the definition of r. Since p(x) has degree 2k ′ , we show p(x) ≥ 2 −O(k ) Ex∼N(0,1)[p 2 (x)]<sup>1</sup>/<sup>2</sup>

near x = √ 2t (Corollary [4.5\)](#page-6-0). The contribution to Ex∼N(0,1)[f 2 (x)] from x ∈ [0.9 √ 2t, 1.1 √ 2t] will then be at least (3.6t) <sup>t</sup> Ex∼N(0,1)[p 2 (x)]2<sup>−</sup>O(<sup>k</sup> ′ ) (see Equations [\(7\)](#page-7-0) and [\(8\)](#page-7-1) for the full calculations). Meanwhile, by Holder's inequality, ¨ Ex∼N(0,1)[f(x)] ≤ Ex∼N(0,1)[x 3t/2 <sup>2</sup>/<sup>3</sup> Ex∼N(0,1)[p 3 (x)]<sup>1</sup>/<sup>3</sup> , which by hypercontractivity is at most ( 3t 2e ) t/<sup>2</sup> Ex∼N(0,1)[p(x) <sup>1</sup>/<sup>2</sup>2 O(k ) . Combining these bounds establishes r ≥ 2 Ω(t−k ′ ) and proves Proposition [4.2.](#page-5-1)

We can also argue that the previous paragraph's result can always be used with w<sup>0</sup> ≥ 2 −O(k )/k (Proposition [4.1\)](#page-5-0). By considering the same polynomial p which vanishes at the points with unconstrained weights, we can combine the hypercontractivity of p with the Cauchy-Schwarz inequality to derive a lower bound on the total weight of the equalweight points: P <sup>i</sup>≥k′+1 w<sup>i</sup> ≥ 2 −O(k ) . This immediately implies w<sup>0</sup> ≥ 2 −O(k ′ )/k.

We have shown that any discrete distribution with k ′ arbitrary weights and k − k ′ equal weights cannot match more than O(log(k) + k ′ ) moments with N (0, 1). This result extends to approximate moment matching within error 2 −O(t) , and holds even after convolving the distribution with a Gaussian (cf. Lemma [D.3\)](#page-15-0). For the parallel pancakes testing problem, this implies that for some i ≤ O(log(k) + k ′ ) the i-th order tensor of the GMM in the alternative hypothesis differs significantly from that of N (0, I). This gap can be detected by estimating the tensor via averaging samples (an operation that has complexity d Θ(m) ), leading to the testing algorithm of Theorem [1.4.](#page-2-0)

# 2. Preliminaries

We present only the essential preliminaries here; see Appendix [B](#page-10-1) for a full version.

Notation We use Z<sup>+</sup> for positive integers, R + 0 for nonnegative reals, and [n] def <sup>=</sup> {1, . . . , n}. We use <sup>x</sup> ⊗ <sup>y</sup> for the tensor product of two vectors. For a random variable x following distribution D, we write x ∼ D and E[x] for its expectation. The Gaussian distribution with mean µ and covariance Σ is N (µ, Σ), and Pr(E) denotes the probability of event E. The indicator function of E is 1(E). The L<sup>p</sup> norm of an <sup>R</sup>-valued random variable x is ∥x∥<sup>p</sup> = E[|x| p <sup>1</sup>/p, and for a function f : <sup>R</sup> <sup>d</sup> → <sup>R</sup>, it is ∥f∥<sup>p</sup> = Ex∼N(0,I) [|f(x)| p <sup>1</sup>/p. We use a ≲ b to indicate a ≤ Cb for an absolute constant C > 0 independent of a and b.

Hermite Analysis In this paper, we use the *normalized probabilist's* Hermite polynomials, which form an orthonormal basis of L 2 := {f : Ex∼N(0,1)[f 2 (x)] < ∞} with respect to the Gaussian measure, i.e., R hk(x)hm(x)e −x <sup>2</sup>/<sup>2</sup> dx <sup>=</sup> <sup>√</sup> 2π1(k = m). Every function f ∈ L 2 can be uniquely

expressed as f(x) = P<sup>∞</sup> <sup>i</sup>=0 aihi(x).

Probability Facts The first fact below follows from direct calculations, the second from the Carbery-Wright inequality, and the last from Holder's inequality combined ¨ with Fact [2.3.](#page-3-0)

Fact 2.1 (Gaussian Moments). E x∼N(0,1) [x t ] ≲ (t/e) t/<sup>2</sup> ∀t≥0*.*

Fact 2.2. *For every polynomial of degree* r *and every* ϵ > 0*,* Prx∼N(0,1) (|p(x)| ≤ ϵ∥p∥1) ≤ O(rϵ<sup>1</sup>/r)*.*

Fact 2.3 (Gaussian Hypercontractivity). *If* p *is a degree* r *polynomial and* k > 2*, then* ∥p∥<sup>k</sup> ≤ (k − 1)r/<sup>2</sup>∥p∥2*.*

Fact 2.4. *For any polynomial* p *of degree* r*,* ∥p∥<sup>1</sup> ∥p∥<sup>2</sup> ≥ 3 −r *.*

Arithmetic Mean-Geometric Mean Inequality We record the following continuous analog of the *Arithmetic Mean-Geometric Mean* (AM-GM) inequality. We refer to Appendix [B](#page-10-1) for a more detailed discussion.

Fact 2.5 (Continuous AM-GM Inequality). *Let* f : R → R + 0 *be a function, and let* I ⊆ <sup>R</sup> *be a finite interval. If* f(x) *and* ln f(x) *are integrable on* I*, then the following holds:* |I| R I <sup>f</sup>(x)d<sup>x</sup> <sup>≥</sup> exp 1 |I| R I ln f(x)dx *.*

Non-Gaussian Component Analysis The parallel pancakes Problem [1.1](#page-1-1) is a special case of the following problem.

Problem 2.6 (Non-Gaussian Component Analysis (NGCA)). Let B be a distribution on R. For a unit vector v, we denote by PB,v the distribution with the density PB,v(x) := B(v <sup>⊤</sup>x)ϕ⊥v(x), where ϕ⊥v(x) = exp −∥x − (v <sup>⊤</sup>x)v∥ 2 <sup>2</sup>/2 /(2π) (d−1)/2 , i.e., the distribution that coincides with B on the direction v and is standard Gaussian in every orthogonal direction. We define the following hypothesis testing problem:

- H0: The data distribution is N (0, Id).
- H1: The data distribution is PB,v, for some vector v ∈ S d−1 in the unit sphere.

It is known that solving Problem [2.6](#page-3-1) when B matches the first m moments with N (0, 1) requires at least d Ω(m) complexity in the statistical query model (Proposition [B.8\)](#page-12-0).

# 3. The Uniform Weights Case

In this section we prove the following proposition, which is sufficient for showing our first result, Theorem [1.3.](#page-1-2)

Proposition 3.1. *For each* k *that is larger than a sufficiently large absolute constant, there exists a set* S *of* k *points in* R *such that the uniform distribution over* S *matches the first* Ω(log k) *moments with* N (0, 1)*.*

Given the above, Theorem [1.3](#page-1-2) follows directly from standard SQ theory. The details are provided in Appendices [B.5](#page-12-1)

and [C,](#page-12-2) but the steps are summarized as follows: Let A be the uniform distribution on the set S from Proposition [3.1.](#page-3-2) We can define the distribution B to be what one obtains by first drawing a sample from A, rescaling it by 1/ √ δ and adding Gaussian noise N (0, 1 − δ). This operation preserves moment matching and makes B a GMM. The NGCA Problem [2.6](#page-3-1) with that B then becomes equivalent to the parallel pancakes Problem [1.1.](#page-1-1) Since B matches m = Ω(log k) moments with N (0, 1), its standard SQ hardness state that its complexity is d Ω(m) = d Ω(log k) (Proposition [B.8\)](#page-12-0). We refer to Appendix [C](#page-12-2) for the details of this paragraph.

In the remainder, we focus on proving Proposition [3.1](#page-3-2) by leveraging a result on designs theory from [Kane](#page-9-12) [\(2015\)](#page-9-12). The original result in [Kane](#page-9-12) [\(2015\)](#page-9-12) is highly general and applies to a wide range of topological, path-connected design problems. However, as we will only use the theorem for intervals, we present here a specialized version tailored to this case.

Fact 3.2 (see Theorem 4 in [Kane](#page-9-12) [\(2015\)](#page-9-12)). *Let* t ∈ Z<sup>+</sup> *be an integer,* I ⊂ R *be an interval and* Q *be a distribution on* I*. Let* W<sup>t</sup> *be the vector space of all polynomials of degree at most* t*, and* V<sup>t</sup> *be the vector space of polynomials* p *of degree at most* t *with* Ex∼Q[p(x)] = 0*. Define* K<sup>t</sup> = supp∈<sup>V</sup> \{0} supx∈<sup>I</sup> p(x) | infx∈<sup>I</sup> p(x)| *. Then for every integer* n > (t − 1)(K<sup>t</sup> + 1) *there exists a set* S ⊂ I *of* n *points such that* <sup>1</sup> |S| P x∈S p(x) = Ex∼Q[p(x)] *for all* p ∈ Wt*.*

Our goal is to show that K<sup>t</sup> = 2O(t) for Q = N (0, 1), which would directly imply Theorem [1.3.](#page-1-2) However, as noted in Section [1.2,](#page-2-1) K<sup>t</sup> may be infinite when I = <sup>R</sup>. To address this, we use a distribution Q supported on a bounded interval of <sup>R</sup> that matches the first t moments of N (0, 1). Applying Fact [3.2](#page-4-2) with this Q also suffices for establish Theorem [1.3.](#page-1-2)

Lemma 3.3 (Gaussian Quadrature (Lemma 4.3 in [Di](#page-8-3)[akonikolas et al.](#page-8-3) [\(2017\)](#page-8-3))). *There is a discrete distribution* Q *on the real line, supported on* t *points, that agrees with* N (0, 1) *on the first* 2t − 1 *moments. All points* x *in the support of* Q *have* |x| = O( √ t)*.*

We start with an anti-concentration property of Gaussian polynomials that will be useful for bounding the numerator in the definition of Kt.

Lemma 3.4. *Let* C *be a sufficiently large constant. For every polynomial* p : R → R *of degree at most* t *with* Ex∼N(0,1)[p(x)] = 0 *and for every* ϵ > 0 *it holds*

$$\mathbf{Pr}_{x \sim \mathcal{N}(0,1)}(p(x) > \epsilon \|p\|_1) \geq \left( \frac{1}{2} \frac{\|p\|_1}{\|p\|_2} (1 - Ct\epsilon^{1+1/t}) \right)^2.$$

*Proof.* Denote by ϕ(x) the pdf of N (0, 1). We have the following (each step is explained below):

$$\|p\|_1 = \int_{p(x)>0} p(x)\phi(x)dx - \int_{p(x)\leq 0} p(x)\phi(x)dx = 2 \int_{p(x)>0} p(x)\phi(x)dx$$

$$\begin{aligned} &= 2 \left( \int_{p(x) \geq \epsilon \|p\|_1} p(x) \phi(x) \, dx + \int_{0 \leq p(x) < \epsilon \|p\|_1} p(x) \phi(x) \, dx \right) \\ &\leq 2 \|p\|_2 \mathbf{Pr}_{x \sim \mathcal{N}(0,1)} (p(x) \geq \epsilon \|p\|_1)^{1/2} \\ &\quad + 2\epsilon \|p\|_1 \mathbf{Pr}_{x \sim \mathcal{N}(0,1)} (|p(x)| \leq \epsilon \|p\|_1) \\ &\leq 2 \|p\|_2 \mathbf{Pr}_{x \sim \mathcal{N}(0,1)} (p(x) \geq \epsilon \|p\|_1)^{1/2} + 2\epsilon \|p\|_1 C t \epsilon^{1/t}, \end{aligned}$$

where the first line used that Ex∼N(0,1)[p(x)] = 0, the penultimate inequality used the Cauchy–Schwarz inequality for the first term and the bound p(x) ≤ ϵ∥p∥<sup>1</sup> for the second term, and the last line used the Carbery-Wright inequality (Fact [2.2\)](#page-3-3). Rearranging, we obtain p Prx∼N(0,1)(p(x) > ϵ∥p∥1) ≥ ∥p∥<sup>1</sup> ∥p∥<sup>2</sup> (1 − 2Ctϵ1+1/t). We rename the constant 2C to C.

We now bound K<sup>t</sup> from Fact [3.2](#page-4-2) with I=[−C √ t,C√ t].

Lemma 3.5. *Let* t > C *be an integer, where* C *is a sufficiently large constant, and define* I = [−C √ t, +C √ t]*. For every polynomial* p *of degree at most* t *with* <sup>E</sup>x∼N(0,1)[p(x)] = 0 *it holds* supx∈<sup>I</sup> <sup>p</sup>(x) <sup>|</sup> infx∈<sup>I</sup> <sup>p</sup>(x)<sup>|</sup> ≤ <sup>2</sup> O(t) *.*

*Proof.* It suffices to upper bound the numerator by 2 <sup>O</sup>(t)∥p∥<sup>1</sup> and lower bound the denominator by 2 <sup>−</sup>O(t)∥p∥1.

Upper bound on numerator We require the following:

Fact 3.6 [\(Krasikov](#page-9-13) [\(2004\)](#page-9-13)). *For the* k*-th normalized probabilist's Hermite polynomial* hk*, we have* supx∈<sup>R</sup> h 2 k (x)e −x <sup>2</sup>/<sup>2</sup> = O(k −1/6 )*.*

Consider a polynomial p which has degree at most t and satisfies Ex∼N(0,1)[p(x)] = 0. We first expand the polynomial in the Hermite basis: p(x) = P<sup>t</sup> <sup>k</sup>=1 akhk(x), where the summation starts from k = 1 because a<sup>0</sup> = Ex∼N(0,1)[p(x)] = 0. For any x ∈ I we have (the first step uses Cauchy–Schwarz inequality):

$$\begin{aligned} |p(x)| &= \left| \sum_{k=1}^t a_k h_k(x) \right| \leq \sqrt{\sum_{k=1}^t a_k^2} \sqrt{\sum_{k=1}^t h_k^2(x)} \\ &\lesssim \|p\|_2 \sqrt{\sum_{k=1}^t e^{x^2/2} k^{-1/6}} \quad (\text{by Fact 3.6}) \\ &\leq \|p\|_2 2^{O(t)} \sqrt{\sum_{k=1}^t k^{-1/6}} \quad (\text{using } |x| = O(\sqrt{k})) \\ &\leq \|p\|_2 2^{O(t)} t^{O(1)} \quad (\text{using } \sum_{k=1}^t k^{-1/6} = t^{O(1)}) \\ &\leq \|p\|_2 2^{O(t)} \leq \|p\|_1 2^{O(t)}. \quad (\text{using Fact 2.4}) \end{aligned}$$

−p in place of p and ϵ = 2−<sup>t</sup> , and Fact [2.4](#page-3-4) we get that

$$\begin{aligned} \mathbf{P}_{x \sim \mathcal{N}(0,1)}(p(x) < -2^{-t} \|p\|_1) &\geq \left( \frac{1}{2} \frac{\|p\|_1}{\|p\|_2} (1 - Ct2^{-t-1}) \right)^2 \\ &\geq \left( \frac{1}{2} 3^{-t} (1 - Ct2^{-t-1}) \right)^2 > 2^{-4t} . \end{aligned}$$

where we used that t is big enough so that C t 2 <sup>t</sup> <0.5. Then,

$$\begin{aligned} & \mathbf{P}_{x \sim \mathcal{N}(0,1)} (p(x) < -2^{-t} \|p\|_1, x \in I) \\ & \geq \frac{\mathbf{P}_{x \sim \mathcal{N}(0,1)}}{} (p(x) < -2^{-t} \|p\|_1) - \frac{\mathbf{P}_{x \sim \mathcal{N}(0,1)}}{} (x \notin I) \\ & \geq 2^{-4t} - 2^{-100t} > 0 , \end{aligned}$$

where in the last line we used that I = [−C √ t, +C √ t] for a large constant C. We have thus shown that infx∈<sup>I</sup> p(x) ≤ −2 <sup>−</sup><sup>t</sup>∥p∥<sup>1</sup> and therefore | infx∈<sup>I</sup> p(x)| > 2 <sup>−</sup><sup>t</sup>∥p∥1.

# 4. The Mostly Equal Weights Case

This section focuses on Theorem [1.4](#page-2-0) and is organized as follows. The key structural result is the following impossibility of moment matching: if A is a distribution on k points, with k ′ points having unconstrained weights and the remaining k − k ′ equal, then A cannot match more than O(log k + k ′ ) moments with the standard Gaussian.

Proposition 4.1. *Let* k ′ < k *be positive integers, and let* A *be a discrete distribution on* k *points in* R*. Suppose* k−k ′ *of the points have equal probability masses, while the remaining* k ′ *points have unrestricted probability masses. Denote by* m *the highest degree for which every degree-*m′ ≤ m *polynomial* g *satisfies* Ex∼A[g(x)] − Ex∼N(0,1)[g(x)]  ≤ 2 <sup>−</sup>C·<sup>m</sup>∥g∥2*, then* m *must satisfy* m ≤ O(log k) + O(k ′ )*.*

Section [4.1](#page-5-2) explains how Proposition [4.1](#page-5-0) leads to a testing algorithm (the full proof appears in Appendix [D.1\)](#page-13-0). Section [4.2](#page-5-3) provides the proof of Proposition [4.1.](#page-5-0)

#### 4.1. Proof Sketch of Theorem [1.4](#page-2-0)

Consider the parallel pancakes problem from Theorem [1.4,](#page-2-0) which is equivalent to the NGCA problem (Problem [2.6\)](#page-3-1) with the 1-d GMM B = P <sup>i</sup>∈[k] wiN (µ<sup>i</sup> , 1−δ). If B approximately matches m moments of N (0, 1), we aim to show that m≤O(log k+k ′ ), enabling a testing algorithm that detects significant deviations in moment tensors. Specifically, suppose every polynomial p of degree m′≤m with ∥p∥2=1 satisfies Ex∼B[p(x)]− Ex∼N(0,1)[p(x)]  ≤(δ/2)Cm for some large constant C ≫ 1. Now, consider the discrete distribution A, which assigns weight w<sup>i</sup> to each center µi/ √ δ. By Lemma [D.3,](#page-15-0) A also approximately matches the moments of N (0, 1), but with an error of 2 −O(m) instead of (δ/2)<sup>O</sup>(m) . Then Proposition [4.1](#page-5-0) yields m ≤ O(log k+k ′ ), as desired.

We just showed that there is a polynomial p of degree at most m = O(log(k) + k ′ ), where the expectations under B and N (0, 1) differ significantly: λ := Ex∼B[p(x)] − Ex∼N(0,1)[p(x)]  > (δ/2)Cm. An averaging argument further implies that a gap holds even for some monomial x i . Lifting this to the d-dimensional parallel pancakes, we have the moment tensor gap Ex∼PB,v [x ⊗i ] − Ex∼N(0,I) [x ⊗i ] = ±λv⊗<sup>i</sup> .

The Frobenius norm of the gap is λ > (δ/2)Cm, implying that between the (expected) moment tensors, at least one entry differs by at least ϵ := λ/d<sup>m</sup> = (d/δ) (C−1)<sup>m</sup>. We will test by searching for such an entry in the empirical tensor.

Algorithm 1 Testing Algorithm (simplified)

1: Input: k, n. Output: Hˆ ∈ {H0, H1}. 2: For i = 1, 2, 3, . . . , C · (log(k) + k ′ ) do 3: Draw x1, . . . , x<sup>n</sup> ∼ D. 4: Define M ← <sup>1</sup> n P<sup>n</sup> <sup>i</sup>=1 x ⊗i . 5: Define M′ := Ex∼N(0,I) [x ⊗i ]. 6: If ∃a=(i1, . . . , ji) such that |Ma−M′ a |>d−Cmλ<sup>m</sup> 7: then Output H<sup>1</sup> and terminate. 8: Return H0.

The tester above is a simplified version. However, it is not fully correct, as we must ensure the concentration of the empirical tensor to bound the sample complexity. The Gaussian N (0, I)'s empirical tensor is well-concentrated. While the parallel pancake's tensor might not concentrate well, this happens only when there is a Gaussian component much farther than √ d from the origin — otherwise every sample from the parallel pancake is within O( √ d) in norm in high probability, and the empirical tensor is entrywise well-concentrated (e.g. by Hoeffding). This is also a testable condition: with ≫ log(k)/wmin samples, we will be able to check if every component is centered at most O( √ d) from the origin. The full version of the algorithm with this additional preliminary check, along with its correctness proof, are provided in Appendix [D.1.](#page-13-0)

### 4.2. Proof of Proposition [4.1](#page-5-0)

We now show Proposition [4.1.](#page-5-0) We will actually show a slightly different version below, where k ′ of the points have arbitrary weights and the rest have weight at least w0.

Proposition 4.2. *Let* C *be a sufficiently large constant, and let* k ′ < k *be positive integers. Let* A *be a discrete distribution on* k *points in* <sup>R</sup> *with probability masses* w1, . . . , wk*, where* w<sup>i</sup> ≥ w<sup>0</sup> *for* i = k ′ + 1, . . . , k *(i.e., the last* k − k ′ *weights are lower bounded by* w0*, while the first* k ′ *weights are unrestricted). Let* m *be the largest degree such that every polynomial* g *of degree* m′ ≤ m *satisfies*

$$\left| \mathbf{E}_{x \sim A} [g(x)] - \mathbf{E}_{x \sim \mathcal{N}(0,1)} [g(x)] \right| \leq w_0 2^{-C \cdot m} \|g\|_2. \quad (1)$$

*Then* m ≤ O(log(1/w0)) + O(k ′ )*.*

Proposition [4.1](#page-5-0) can be derived from this via the following observation (shown in Appendix [D.2\)](#page-17-0): in the setting of Proposition [4.2,](#page-5-1) let p(x) = Q<sup>k</sup> <sup>i</sup>=1(x − µi), where µ1, . . . , µk′ are the points in the support of A with the unconstrained weights. Then, the weights of the k − k ′ points with uniform weights is always P<sup>k</sup> <sup>i</sup>=k′+1 w<sup>i</sup> ≥ Ex∼A[p(x)]<sup>2</sup> <sup>E</sup>x∼A[p<sup>2</sup>(x)] ≳ ∥p∥ 2 1 ∥p∥ ≥ 3 −2k ′ , where the first step uses Cauchy-Schwarz inequality, the second uses the (approximate) moment matching, and the third is a consequence of Gaussian hypercontractivity (Fact [2.4\)](#page-3-4). This means that every such weight is w<sup>i</sup> ≥ 3 −2k /k, which when plugged into Proposition [4.2](#page-5-1) gives Proposition [4.1.](#page-5-0)

We now focus on showing Proposition [4.2.](#page-5-1) We will follow a top-down presentation, starting with the proof strategy and concluding with a derivation of the necessary bounds.

We will use a reparameterization m = 2t + 4k ′ with t even. The goal is to show that if A is assumed to approximately match the first m = 2t + 4k ′ moments with N (0, 1) (in the sense of Equation [\(1\)](#page-5-4)), then t must be at most O(log(1/w0) + k ′ ). Let µ1, . . . , µ<sup>k</sup> be the points on which A is supported, where the first k ′ points are the ones with the unrestricted weights, and consider the polynomial f(x) = x <sup>t</sup>p(x), where p(x) = (x − µ1) 2 (x − µ2) 2 · · ·(x − µk′ ) 2 . The proof strategy is the following: if the expectation of f under A approximately matches that of N (0, 1), then the value of f on every point µ<sup>i</sup> cannot be too large, which will cause the expectations of f 2 to deviate.

Because of Equation [\(1\)](#page-5-4) with g(x) = f 2 (x), we have:

$$\sum_{i=k'+1}^k w_i \mu_i^t p(\mu_i) = \mathbf{E}_{x \sim A} [x^t p(x)] \quad (2)$$

$$\leq \mathbf{E}_{x \sim \mathcal{N}(0,1)} [x^t p(x)] + w_0 \frac{\|x^t p(x)\|_2}{2^{C(2t+4k')}}. \quad (3)$$

This, together with the lower bound w<sup>i</sup> ≥ w<sup>0</sup> for the points i = k ′ + 1, . . . , k and the fact that t is even, implies that for all i = k ′ + 1, . . . , k it holds

$$\mu_i^t p(\mu_i) \leq \frac{1}{w_0} \mathbf{E}_{x \sim \mathcal{N}(0,1)} [x^t p(x)] + \frac{\|x^t p(x)\|_2}{2^{C(2t+4k')}}. \quad (4)$$

We now examine the expectations of the square of f(x). Because of Equation [\(1\)](#page-5-4) with g(x) = f 2 (x), we have

$$\begin{aligned} x \sim \mathcal{N}(0, 1) & \left[ x^{2t} p^2(x) \right] \leq \mathbf{E}_{x \sim \mathcal{N}} \left[ x^{2t} p^2(x) \right] + \frac{\|x^{2t} p^2(x)\|_2}{2^{C(2t+4k')}} \\ & = \sum_{i=k'+1}^k w_i (\mu_i^t p(\mu_i))^2 + \frac{\|x^{2t} p^2(x)\|_2}{2^{C(2t+4k')}}. \end{aligned}$$

Next, we can combine this with Equation [\(4\)](#page-6-1), divide both sides by Ex∼N(0,1) [x <sup>t</sup>p(x)]<sup>2</sup> (and use P<sup>k</sup> <sup>i</sup>=k′+1 w<sup>i</sup> ≤ 1) to obtain the following, where λ := 2−C(2t+4<sup>k</sup> ′ ) :

$$\begin{aligned} \frac{\mathbf{E}_{x\sim\mathcal{N}(0,1)} [x^{2t}p^2(x)]}{\mathbf{E}_{x\sim\mathcal{N}(0,1)} [x^t p(x)]^2} &\leq \left( \frac{1}{w_0} \right)^2 \\ &+ \lambda^2 \frac{\mathbf{E}_{x\sim\mathcal{N}(0,1)} [x^{2t}p^2(x)]}{\mathbf{E}_{x\sim\mathcal{N}(0,1)} [x^t p(x)]^2} + \lambda \frac{\mathbf{E}_{x\sim\mathcal{N}(0,1)} [x^{4t}p^4(x)]^{\frac{1}{2}}}{\mathbf{E}_{x\sim\mathcal{N}(0,1)} [x^t p(x)]^2}. \end{aligned}$$

Let us simplify this inequality. Let r be the ratio on the LHS. The second term on the RHS is λ 2 · r. The third term is at most 3 t+2k λr, by applying Gaussian hypercontractivity (Fact [2.3\)](#page-3-0) to the polynomial x <sup>t</sup>p(x). Thus, the inequality becomes r(1 − λ <sup>2</sup> − λ3 t+2k ′ ) ≲ 1/w<sup>2</sup> . Since λ = 2−C(2t+4<sup>k</sup> ′ ) for large constant C, the expression inside the parentheses is greater than 0.5. Therefore, the inequality implies that r ≲ 1/w<sup>2</sup> 0 .

The next step is to establish a lower bound for r, specifically to show that r ≥ 2 Ω(t)/2 O(k ) . If this can be done, combining the two bounds 2 Ω(t)/2 O(k ) ≤ 1/w<sup>2</sup> 0 and taking logarithms yields t = O(log(1/w0)) + O(k ′ ), completing the proof of Proposition [4.2.](#page-5-1)

### 4.2.1. LOWER BOUNDING THE RATIO r

We want to establish the following, which was the missing piece in the proof of Proposition [4.2](#page-5-1) above.

Lemma 4.3. *Let* p : R → R + *be a polynomial of the form* p(x) = (x − µ1) 2 (x − µ2) 2 · · ·(x − µk′ ) 2 *. Then*

$$\frac{\mathbf{E}_{x\sim\mathcal{N}(0,1)} [x^{2t} p^2(x)]}{\mathbf{E}_{x\sim\mathcal{N}(0,1)} [x^t p(x)]^2} \gtrsim \frac{2^{\Omega(t)}}{2^{O(k')}}.$$

The most difficult part involves lower bounding the numerator. To this end, we will show the following bound:

Lemma 4.4. *Let* p : R → R *be a polynomial of the form* p(x) = (x−µ1)(x−µ2)· · ·(x−µk′ ) *where* µ1, . . . , µk′ ∈ <sup>R</sup>*, and define* I := [0.9 √ 2t, 1.1 √ 2t]*. For every* t > 0 *and for every* µ1, . . . , µk′ ∈ <sup>R</sup>*, the following holds:*

$$\exp\left(\frac{1}{|I|} \int_{x \in I} \ln |p(x)| dx\right) \geq \max_{y \in \mathbb{R}: |y| \leq \sqrt{t}} \frac{|p(y)|}{2^{O(k')}}. \quad (5)$$

We will actually apply Lemma [4.4](#page-6-2) after taking expectations of both sides. This version is presented below, and its proof follows by taking expectations and performing some manipulations (see Appendix [D.3](#page-18-0) for a detailed proof).

Corollary 4.5. *Let* p : R → R *be a polynomial of the form* p(x) = (x−µ1)(x−µ2)· · ·(x−µk′ ) *where* µ1, . . . , µk′ ∈ <sup>R</sup> *are arbitrary parameters. Define* I = [0.9 √ 2t, 1.1 √ 2t]*. For all* <sup>t</sup> <sup>≥</sup> <sup>1</sup> *we have* exp 1 |I| R x∈I ln |p(x)|dx ≥ ∥p∥<sup>2</sup> 2O(k′) *.* To see why the above bound is needed to prove Lemma [4.3,](#page-6-3) we will first prove Lemma [4.3](#page-6-3) assuming Corollary [4.5.](#page-6-0) Then, we will prove Lemma [4.4.](#page-6-2)

*Proof of Lemma [4.3.](#page-6-3)* First, for the denominator, we have the following:

$$\begin{aligned} x_{\sim \mathcal{N}(0,1)}[x^t p(x)] &\leq x_{\sim \mathcal{N}(0,1)}[p^3(x)]^{1/3} x_{\sim \mathcal{N}(0,1)}[x^{3t/2}]^{2/3} \\ &\lesssim \|p\|_3 \left( \frac{3t}{2e} \right)^{t/2} \lesssim 2^{k'} \|p\|_2 \left( \frac{3t}{2e} \right)^{t/2}, \end{aligned} \quad (6)$$

where the first step uses Holder's inequality, the second step ¨ uses the Gaussian moments bound (Fact [2.1\)](#page-3-5), and the final step uses Gaussian hypercontractivity (Fact [2.3\)](#page-3-0).

We now lower bound the numerator. Define I := [0.9 √ 2t, 1.1 √ 2t]. We have the following (see below for explanations of each step):

$$\begin{aligned} x \sim \mathcal{N}(0,1) \quad & \left[ x^{2t} p^2(x) \right] \gtrsim \int_{-\infty}^{+\infty} x^{2t} e^{-x^2/2} p^2(x) \, dx \\ & \geq \int_{x \in I} x^{2t} e^{-x^2/2} p^2(x) \, dx \\ & \geq (1.62t)^t e^{-0.81t} \int_{x \in I} p^2(x) \, dx \\ & = (1.62t)^t e^{-0.81t} |I| \left( \frac{1}{|I|} \int_{x \in I} p^2(x) \, dx \right) \\ & \gtrsim (1.62t)^t e^{-0.81t} \left( \frac{1}{|I|} \int_{x \in I} p^2(x) \, dx \right), \quad (7) \end{aligned}$$

where the third inequality uses that minx∈<sup>I</sup> x 2t e −x <sup>2</sup>/<sup>2</sup> ≥ (1.62t) t e −0.81t , and the final inequality uses that |I| = 0.2 √ 2t = Ω(1). We now focus on the root mean square term <sup>1</sup> |I| R x∈I p 2 (x) dx, which we will bound using the AM-GM inequality (Fact [2.5\)](#page-3-6) and the geometric mean bound from Lemma [4.4.](#page-6-2) The first step below applies Fact [2.5](#page-3-6) with f(x) := p 2 (x), and the next step uses Corollary [4.5.](#page-6-0)

$$\frac{1}{|I|} \int_{x \in I} p^2(x) \, dx \geq \exp \left( \frac{1}{|I|} \int_{x \in I} \ln |p(x)| \, dx \right)^2 \geq \frac{\|p\|_2^2}{2^{O(k')}}.$$

Combining with Equation [\(7\)](#page-7-0), we obtain the following bound for the numerator:

$$x \sim \mathbf{E}_{\mathcal{N}(0,1)} [x^{2t} p^2(x)] \gtrsim (1.62t)^t e^{-0.81t} \frac{\|p\|_2^2}{2^{O(k')}}. \quad (8)$$

Combining Equation [\(6\)](#page-7-2) and Equation [\(8\)](#page-7-1), we conclude

$$\frac{\mathbf{E}_{x\sim\mathcal{N}(0,1)} [x^{2t}p^2(x)]}{\mathbf{E}_{x\sim\mathcal{N}(0,1)} [x^t p(x)]^2} \gtrsim \frac{(1.62)^t}{(\frac{1.5}{e})^t e^{0.81t} 2^{O(k')}} \geq \frac{(1.3)^t}{2^{O(k')}} \cdot \square$$

We conclude this section by proving Lemma [4.4.](#page-6-2)

*Proof of Lemma [4.4.](#page-6-2)* Fix an arbitrary <sup>y</sup> <sup>∈</sup> <sup>R</sup> with <sup>|</sup>y| ≤ √ t. First, note that by the property of logarithms and sums, we can write the left hand side as

$$\exp \left( \sum_{i=1}^{k'} \frac{1}{|I|} \int_{x \in I} \ln |x - \mu_i| dx \right).$$

In order to show Equation [\(5\)](#page-6-4), it suffices to work with each term and show the following for each i ∈ [k ′ ]:

$$\frac{1}{|I|} \int_{x \in I} \ln |x - \mu_i| \geq \ln |y - \mu_i| - O(1) .$$

Equivalently, it suffices to show that Equation [\(5\)](#page-6-4) holds for every linear polynomial of the form p(x) = x − a. Therefore, the goal for the rest of this proof is to show that

$$\exp\left(\frac{1}{|I|} \int_{x \in I} \ln |x - a| dx\right) \geq |y - a|/O(1), \quad (9)$$

holds for every <sup>a</sup> <sup>∈</sup> <sup>R</sup> and <sup>y</sup> <sup>∈</sup> <sup>R</sup> with <sup>|</sup>y| ≤ √ t. We will examine two cases.

Case 1 The first case is when the root a of the polynomial is outside the interval I. In this case, we can show that |x − a|/|y − a| = Θ(1), which implies ln |x − a| ≥ ln |y − a|−O(1), and the desired conclusion (Equation [\(9\)](#page-7-3)) follows by integrating both sides and applying the exp(·) function.

To show the earlier claim that |x − a|/|y − a| = Θ(1), we can consider the following sub-cases:

- 1. Case a ≥ 1.1 √ 2t (i.e., a is to the right of I): Suppose a = 1.1 √ 2t+u for some non-negative u. Then, a−x = (1.1 √ <sup>2</sup>t−x)+<sup>u</sup> = Θ(√ t)+u and a−y = (1.1 √ 2t− <sup>y</sup>)+<sup>u</sup> = Θ(√ t)+u. Therefore, for any u ≥ 0, the ratio <sup>|</sup><sup>x</sup> <sup>−</sup> <sup>a</sup>|/|<sup>y</sup> <sup>−</sup> <sup>a</sup><sup>|</sup> = (Θ(√
  - <sup>t</sup>) + <sup>u</sup>)/(Θ(√
    - t) + u) = Θ(1).
- 2. The cases a < − √ t and a ∈ [ √ t, 0.9 √ 2t] can be shown in a similar manner.

Case 2 Suppose that the root a of the polynomial p lies within the interval I. In that case, we can show via derivative analysis that f(a) := <sup>1</sup> |I| R x∈I ln |x−a| dx for a ∈ I is minimized at the midpoint of I, i.e., at a = √ 2t, and confirm that f( √ 2t) ≥ √ t/20 = Ω(|y − a|). These calculations are provided in Appendix [D.3.](#page-18-0)

# Conclusions and Future Work

Our work makes progress in understanding the complexity of learning parallel pancake GMMs, in terms of both lower and upper bounds. We establish the tightness of existing algorithms for uniform weights and provide an improved testing algorithm for uneven weights. A number of interesting open problems remain:

- Can we extend our testing algorithm to learning the unknown direction of the parallel pancakes? More broadly, can we characterize the complexity of learning GMMs with common covariance and not necessarily collinear means as a function of the weights distribution?
- Can we obtain an algorithm with quasi-polynomial (i.e., d O(log(1/wmin) ) complexity for GMMs whose components have unknown (and potentially different) covariances? Impact Statement This work is theoretical in nature and focuses on advancing fundamental knowledge. As such, it does not directly raise any societal or ethical concerns that warrant special consideration. Acknowledgments Ilias Diakonikolas was supported by NSF Medium Award CCF-2107079 and an H.I. Romnes Faculty Fellowship. Daniel M. Kane was supported by NSF Medium Award CCF-2107547 and NSF Award CCF-1553288 (CAREER). The work of Jasper C.H. Lee was done in part while he was at UW Madison, supported by NSF Medium Award CCF-2107079. References Achlioptas, D. and McSherry, F. On spectral learning of mixtures of distributions. In *Proc. 18th Annual Conference on Learning Theory (COLT)*, 2005. Anderson, P., Bafna, M., Buhai, R. D., Kothari, P. K., and Steurer, D. Dimension reduction via sum-of-squares and improved clustering algorithms for non-spherical mixtures. *arXiv preprint arXiv:2411.12438*, 2024. Andrews, G. E., Askey, R., and Roy, R. *Special Functions*. 1999. Arora, S. and Kannan, R. Learning mixtures of arbitrary Gaussians. In *Proc. 33rd Annual ACM Symposium on Theory of Computing (STOC)*, 2001. Bakshi, A. and Kothari, P. K. Outlier-robust clustering of non-spherical mixtures. *arXiv:2005.02970*, 2020. Bakshi, A., Diakonikolas, I., Hopkins, S. B., Kane, D., Karmalkar, S., and Kothari, P. K. Outlier-robust clustering of gaussians and other non-spherical mixtures. In *Foundations of Computer Science (FOCS)*, 2020. Bakshi, A., Diakonikolas, I., Jia, H., Kane, D. M., Kothari,
- P. K., and Vempala, S. S. Robustly learning mixtures of k arbitrary gaussians. In *Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing (STOC 2022)*, 2022. Also available as arXiv:2012.02119. Belkin, M. and Sinha, K. Polynomial learning of distribution families. *SIAM Journal on Computing*, 44(4):889–911, 2015. Bogachev, V. *Gaussian Measures*. 1998. Brubaker, S. C. and Vempala, S. S. Isotropic pca and affineinvariant clustering. *Building Bridges: Between Mathematics and Computer Science*, pp. 241–281, 2008. Bruna, J., Regev, O., Song, M. J., and Tang, Y. Continuous LWE. In *Symposium on Theory of Computing (STOC)*, 2021. Buhai, R. and Steurer, D. Beyond parallel pancakes: Quasipolynomial time guarantees for non-spherical gaussian mixtures. In *The Thirty Sixth Annual Conference on Learning Theory*, pp. 548–611. PMLR, 2023. Carbery, A. and Wright, J. Distributional and l <sup>q</sup> norm inequalities for polynomials over convex bodies in R
  - n. *Mathematical Research Letters*, 8(3):233–248, 2001. ISSN 10732780, 1945001X. doi: 10.4310/MRL.2001.v8. n3.a1. Dasgupta, S. Learning mixtures of gaussians. In *Foundations of Computer Science (FOCS)*, 1999. Daskalakis, C. and Kamath, G. Faster and sample nearoptimal algorithms for proper learning mixtures of gaussians. In *Conference on Learning Theory*, pp. 1183–1213. PMLR, 2014. Diakonikolas, I. and Kane, D. M. Implicit high-order moment tensor estimation and learning latent variable models. *arXiv preprint arXiv:2411.15669*, 2024. Diakonikolas, I., Kane, D. M., and Stewart, A. Statistical query lower bounds for robust estimation of highdimensional gaussians and gaussian mixtures. In *Proc. 58th IEEE Symposium on Foundations of Computer Science (FOCS)*, 2017. doi: 10.1109/FOCS.2017.16. Diakonikolas, I., Hopkins, S. B., Kane, D., and Karmalkar,
  - S. Robustly learning any clusterable mixture of gaussians. *arXiv:2005.06417*, 2020. Diakonikolas, I., Kane, D. M., Kongsgaard, D., Li, J., and Tian, K. Clustering mixture models in almost-linear time via list-decodable mean estimation. In *Proc. 54th Annual ACM Symposium on Theory of Computing (STOC)*, 2022. Diakonikolas, I., Kane, D., Ren, L., and Sun, Y. Sq lower bounds for non-gaussian component analysis with weaker assumptions. *Advances in Neural Information Processing Systems*, 2023.

Diakonikolas, I., Karmalkar, S., Pang, S., and Potechin, A. Sum-of-squares lower bounds for non-gaussian component analysis. In *2024 IEEE 65th Annual Symposium on Foundations of Computer Science (FOCS)*, pp. 949–958. IEEE, 2024. Feldman, V. Statistical query learning. In *Encyclopedia of Algorithms*, pp. 2090–2095. Springer New York, 2016. Feldman, V., Grigorescu, E., Reyzin, L., Vempala, S. S., and Xiao, Y. Statistical algorithms and a lower bound for detecting planted cliques. *Journal of the ACM*, 64(2), 2017a. doi: 10.1145/3046674. Feldman, V., Guzman, C., and Vempala, S. S. Statistical query algorithms for mean vector estimation and stochastic convex optimization. 2017b. Garc´ıa-Escudero, L., Gordaliza, A., Matran, C., and Mayo- ´ Iscar, A. A review of robust clustering methods. *Advances in Data Analysis and Classification*, 4(2):89–109, 2010. doi: 10.1007/s11634-010-0064-5. Gupte, A., Vafa, N., and Vaikuntanathan, V. Continuous LWE is as hard as LWE & applications to learning gaussian mixtures. *CoRR*, abs/2204.02550, 2022. Hardt, M. and Price, E. Tight bounds for learning a mixture of two gaussians. In *Proceedings of the forty-seventh annual ACM symposium on Theory of computing*, pp. 753–760, 2015. Hopkins, S. B. and Li, J. Mixture models, robustness, and sum of squares proofs. In *Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing (STOC)*, pp. 1021–1034, 2018. Kane, D. Small designs for path-connected spaces and path-connected homogeneous spaces. *Transactions of the American Mathematical Society*, 2015. Kannan, R., Salmasian, H., and Vempala, S. The spectral method for general mixture models. In *Proc. 18th Annual Conference on Learning Theory (COLT)*, 2005. Kearns, M. J. Efficient noise-tolerant learning from statistical queries. *Journal of the ACM*, 45(6):983–1006, 1998. Kothari, P. K. and Steurer, D. Outlier-robust momentestimation via sum-of-squares. *arXiv preprint arXiv:1711.11581*, 2017. Kothari, P. K., Steinhardt, J., and Steurer, D. Robust moment estimation and improved clustering via sum of squares. In *Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing (STOC)*, pp. 1035–1046, 2018. Krasikov, I. New bounds on the Hermite polynomials. *arXiv preprint math/0401310*, 2004. Lindsay, B. *Mixture models: theory, geometry and applications*. Institute for Mathematical Statistics, 1995. Liu, A. and Li, J. Clustering mixtures with almost optimal separation in polynomial time. In *Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing (STOC)*, pp. 1248–1261, 2022. Liu, A. and Moitra, A. Settling the robust learnability of mixtures of gaussians. In *Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing (STOC)*, pp. 518–531, 2021. Liu, A. and Moitra, A. Learning gmms with nearly optimal robustness guarantees. In *Proc. 35th Annual Conference on Learning Theory (COLT)*, 2022. Moitra, A. and Valiant, G. Settling the polynomial learnability of mixtures of gaussians. In *Proceedings of the 51st Annual IEEE Symposium on Foundations of Computer Science (FOCS)*, pp. 93–102, 2010. Nelson, E. The free markoff field. *Journal of Functional Analysis*, 1973. O'Donnell, R. *Analysis of Boolean Functions*. 2014. Pearson, K. Contributions to the mathematical theory of evolution. *Philosophical Transactions of the Royal Society of London. Series A*, 185:71–110, 1894. Suresh, A. T., Orlitsky, A., Acharya, J., and Jafarpour, A. Near-optimal-sample estimators for spherical gaussian mixtures. *Advances in Neural Information Processing Systems*, 27, 2014. Szego, G. ¨ *Orthogonal Polynomials*, volume XXIII of *American Mathematical Society Colloquium Publications*. American Mathematical Society, 1989. Titterington, D., Smith, A., and Makov, U. *Statistical Analysis of Finite Mixture Distributions*. Wiley, New York, 1985. Vempala, S. and Wang, G. A spectral algorithm for learning mixtures of distributions. In *Proceedings of the 43rd Annual IEEE Symposium on Foundations of Computer Science (FOCS)*, pp. 113–122, 2002.

## Supplementary Material

Organization Appendix [A](#page-10-0) discusses additional related work on Gaussian mixture models, Appendix [B](#page-10-1) provides the full version of the preliminaries needed for the technical proofs, Appendix [C](#page-12-2) provides the missing details from the proof of our first main result, Theorem [1.3,](#page-1-2) and Appendix [D](#page-13-1) provides the missing details from our second main result, Theorem [1.4.](#page-2-0)

# A. Additional Related Work

Learning Gaussian Mixture Models (GMMs) is one of the most studied problems in statistics, dating back to [Pearson](#page-9-14) [\(1894\)](#page-9-14). Over the years, a plethora of works has explored this area [\(Dasgupta,](#page-8-8) [1999;](#page-8-8) [Arora & Kannan,](#page-8-9) [2001;](#page-8-9) [Vempala & Wang,](#page-9-3) [2002;](#page-9-3) [Achlioptas & McSherry,](#page-8-0) [2005;](#page-8-0) [Kannan et al.,](#page-9-4) [2005;](#page-9-4) [Brubaker & Vempala,](#page-8-10) [2008;](#page-8-10) [Moitra & Valiant,](#page-9-15) [2010;](#page-9-15) [Belkin &](#page-8-11) [Sinha,](#page-8-11) [2015;](#page-8-11) [Suresh et al.,](#page-9-16) [2014;](#page-9-16) [Daskalakis & Kamath,](#page-8-12) [2014;](#page-8-12) [Hardt & Price,](#page-9-17) [2015;](#page-9-17) [Diakonikolas et al.,](#page-8-13) [2020;](#page-8-13) [Bakshi et al.,](#page-8-14) [2020;](#page-8-14) [Diakonikolas et al.,](#page-8-15) [2022;](#page-8-15) [Liu & Moitra,](#page-9-18) [2021;](#page-9-18) [Bakshi et al.,](#page-8-2) [2022\)](#page-8-2). Here, we provide a brief exposition of part of this literature, though this is not a comprehensive survey.

A key starting point was [Dasgupta](#page-8-8) [\(1999\)](#page-8-8), which studied learning GMMs with well-separated, spherical covariance components. Subsequent work by [Vempala & Wang](#page-9-3) [\(2002\)](#page-9-3); [Achlioptas & McSherry](#page-8-0) [\(2005\)](#page-8-0); [Kannan et al.](#page-9-4) [\(2005\)](#page-9-4) improved the separation condition to be dimension-independent. [Hopkins & Li](#page-9-19) [\(2018\)](#page-9-19) and [Kothari et al.](#page-9-20) [\(2018\)](#page-9-20) later refined the separation assumption to the information-theoretic limit, achieving this with quasi-polynomial-time algorithms. Notably, most of the aforementioned works extend beyond spherical Gaussians; however, they measure the pairwise mean separation between components relative to the largest eigenvalue of the components' covariance matrices. More recently, [Liu & Li](#page-9-5) [\(2022\)](#page-9-5) improved the runtime to polynomial for spherical Gaussians.

The case of arbitrary Gaussians with unknown component covariances has also been extensively studied [\(Belkin & Sinha,](#page-8-11) [2015;](#page-8-11) [Moitra & Valiant,](#page-9-15) [2010;](#page-9-15) [Bakshi & Kothari,](#page-8-16) [2020;](#page-8-16) [Diakonikolas et al.,](#page-8-13) [2020;](#page-8-13) [Liu & Moitra,](#page-9-21) [2022;](#page-9-21) [Bakshi et al.,](#page-8-2) [2022\)](#page-8-2). While the first works in this list had complexities doubly exponential in k, the number of components, the most recent have reduced this to d O(k) . As noted in Section [1,](#page-0-0) hardness results from [Diakonikolas et al.](#page-8-3) [\(2017\)](#page-8-3); [Bruna et al.](#page-8-4) [\(2021\)](#page-8-4); [Gupte](#page-9-6) [et al.](#page-9-6) [\(2022\)](#page-9-6) showed this complexity is necessary. However, certain special cases of GMMs can circumvent these lower bounds. This was the focus of [Buhai & Steurer](#page-8-5) [\(2023\)](#page-8-5) and [Anderson et al.](#page-8-6) [\(2024\)](#page-8-6), who studied GMMs with a minimum mixing weight wmin ≥ 1/ poly(k) and unknown but shared covariance across components. These papers motivated us to study whether further improvements are possible for this family of GMMs.

# B. Additional Preliminaries

## B.1. Notation

We use <sup>Z</sup><sup>+</sup> to denote positive integers. For n ∈ <sup>Z</sup><sup>+</sup> we denote [n] def <sup>=</sup> {1, . . . , n}. We denote by <sup>R</sup> + 0 the set of all non-negative positive real numbers. We denote by exp(x) = e x the exponential function, and by ln(x) the natural logarithm (the logarithm with base e). We write x⊗y to denote the tensor product between two vectors x, y. The tensor product of two vectors u ∈ R <sup>m</sup> and v ∈ <sup>R</sup> <sup>n</sup> is a matrix u ⊗ v ∈ <sup>R</sup> <sup>m</sup>×<sup>n</sup> defined such that (u ⊗ v)ij = uiv<sup>j</sup> , where each entry is the product of the corresponding components of u and v. This can be extended to product between more than two vectors. We denote x ⊗k the k-fold tensor product of the vector x with itself. If A is a tensor, ∥A∥<sup>F</sup> denotes its Frobenius norm, which is the Euclidean norm of the vector obtained by stacking all entries of the tensor into a single vector. We write x ∼ D for a random variable x following the distribution D and use E[x] for its expectation. We use N (µ, Σ) to denote the Gaussian distribution with mean µ and covariance matrix Σ. We write Pr(E) for the probability of an event E. We denote by 1(E) the indicator function of the event E. The L<sup>p</sup> norm of a (R-valued) random variable x is defined to be ∥x∥<sup>p</sup> = E[|x| p <sup>1</sup>/p. The L<sup>p</sup> norm of a function f : R <sup>d</sup> → <sup>R</sup> is defined to be the L<sup>p</sup> norm of the random variable f(x), i.e., ∥f∥<sup>p</sup> = Ex∼N(0,I) [|f(x)| p 1/p .

We use a ≲ b to denote that there exists an absolute universal constant C > 0 (independent of the variables or parameters on which a and b depend) such that a ≤ Cb. Sometimes, we will also use the O(·), Ω(·), Θ(·) notation with the standard meaning.

## B.2. Hermite Analysis

Hermite polynomials form a complete orthogonal basis of the vector space L 2 (R, N (0, 1)) of all functions f : <sup>R</sup> → <sup>R</sup> such that Ex∼N(0,1)[f 2 (x)] < ∞. We will use the *normalized probabilist's* Hermite polynomials, which have unit norm and are pairwise orthogonal with respect to the Gaussian measure, i.e., R hk(x)hm(x)e −x <sup>2</sup>/<sup>2</sup>dx = √ 2π1(k = m). These polynomials are the ones obtained by Gram-Schmidt orthonormalization of the basis {1, x, x<sup>2</sup> , . . .} with respect to the inner product ⟨f, g⟩<sup>N</sup>(0,1) := Ex∼N(0,1)[f(x)g(x)]. Every function f ∈ L 2 (R, N (0, 1)) can be uniquely written as f(x) = P<sup>∞</sup> <sup>i</sup>=0 aihi(x) and we have limn→∞ Ex∼N(0,1)[(f(x) − P<sup>n</sup> <sup>i</sup>=0 <sup>a</sup>ihi(x))<sup>2</sup> ] = 0 (see [Andrews et al.](#page-8-17) [\(1999\)](#page-8-17) for a more detailed exposition of Hermite polynomial's properties). We have the following closed form formula (see, e.g., [Szego¨](#page-9-22) [\(1989\)](#page-9-22)):

$$h_n(x) = \sqrt{n!} \sum_{j=0}^{\lfloor n/2 \rfloor} \frac{(-1)^j}{j!(n-2j)!2^j} x^{n-2j}. \quad (10)$$

We will use the following fact, stating that the largest coefficient in the above expansion cannot be too large.

Fact B.1 (Upper Bound on Hermite Polynomial Coefficients). *Let* hn(x) *denote the normalized probabilist's Hermite polynomial of order* <sup>n</sup>*. In the monomial expansion* <sup>h</sup>n(x) = P<sup>n</sup> <sup>j</sup>=1 ajx j *, it holds* |a<sup>j</sup> | ≤ 2 O(n) *for all* j ∈ [n]*.*

*Proof.* This follows by Equation [\(10\)](#page-11-0). The j-th coefficient is √ n!(−1)<sup>j</sup>/(j!(n − 2j)!2<sup>j</sup> )  ≤ √ n!/(j!(n − 2j)!2<sup>j</sup> ). Then one can use the elementary inequalities e(k/e) <sup>k</sup> ≤ k! ≤ ek(k/e) k to bound the factorials that appear in the numerator and denominator and optimize the resulting function. The derivative analysis of the resulting function gives two points of zero derivative: j = 4 (1 + 2n − √ 1 + 4n) and j = 4 (1 + 2n + √ 1 + 4n). For each point, it can be checked that the function is smaller than 2 <sup>n</sup> in the limit n → ∞.

Definition B.2 (Ornstein-Uhlenbeck Operator). For a ρ ∈ [−1, 1], we define the *Ornstein-Uhlenbeck* (or *Gaussian noise*) operator U<sup>ρ</sup> as the operator that maps a distribution F on <sup>R</sup> to the distribution of the random variable ρX + p 1 − ρ <sup>2</sup>Z, where X ∼ F and Z ∼ N (0, 1) independently of X.

A well-known property of *Ornstein–Uhlenbeck* operator is that it operates diagonally with respect to Hermite polynomials.

Fact B.3 (see, e.g., [O'Donnell](#page-9-23) [\(2014\)](#page-9-23)). *For any normalized Hermite polynomial* h<sup>i</sup> *, any distribution* F *on* <sup>R</sup>*, and* δ ∈ [−1, 1]*, it holds that* EX∼Uρ<sup>F</sup> [hi(X)] = ρ <sup>i</sup> EX∼<sup>F</sup> [hi(X)]*.*

# B.3. Properties of Polynomials Under the Gaussian Measure

Fact 2.1 (Gaussian Moments). E x∼N(0,1) [x t ] ≲ (t/e) t/<sup>2</sup> ∀t≥0*.*

Fact B.4 (Carbery-Wright Inequality [\(Carbery & Wright,](#page-8-18) [2001\)](#page-8-18)). *There is an absolute constant* C *such that the following holds. Let* q, γ ∈ R + 0 *,* µ ∈ R d , Σ ∈ R d×d *such that* Σ *is symmetric PSD and* p : R <sup>d</sup> → <sup>R</sup> *be a multivariate polynomial of degree at most* r*. Then*

$$\mathbf{Pr}_{x \sim \mathcal{N}(\mu, \Sigma)} (|p(x)| \leq \gamma) \leq \frac{Cq\gamma^{1/r}}{(\mathbf{E}_{z \sim \mathcal{N}(\mu, \Sigma)} [|p(z)|^{q/r}])^{1/q}}.$$

The way that we will apply this is with the following choice of parameters: d = 1, µ = 0, Σ = 1, q = r and γ = ϵ∥p∥1, where ϵ is a new parameter. This gives:

Fact 2.2. *For every polynomial of degree* r *and every* ϵ > 0*,* Prx∼N(0,1) (|p(x)| ≤ ϵ∥p∥1) ≤ O(rϵ<sup>1</sup>/r)*.*

The following inequality can be easily derived using Holder's inequality. ¨

Fact B.5. ∥x∥<sup>2</sup> ≤ ∥x∥ 1/3 1 ∥x∥ 2/3 4 *for any random variable.*

The following inequality is the Gaussian Hypercontractivity property (see, e.g., [Bogachev](#page-8-19) [\(1998\)](#page-8-19); [Nelson](#page-9-24) [\(1973\)](#page-9-24))

Fact 2.3 (Gaussian Hypercontractivity). *If* p *is a degree* r *polynomial and* k > 2*, then* ∥p∥<sup>k</sup> ≤ (k − 1)r/<sup>2</sup>∥p∥2*.*

In particular we will use the above in the following way:

*Proof.* ∥p∥<sup>1</sup> ≥ ∥p∥ 2 ∥p∥ 4 = ∥p∥<sup>2</sup> ∥p∥<sup>2</sup> ∥p∥<sup>4</sup> 2 ≥ 3 <sup>−</sup><sup>t</sup>∥p∥2, where the first step used Fact [B.5](#page-11-1) and the last step used Gaussian Hypercontractivtiy (Fact [2.3\)](#page-3-0) with k = 4. Rearranging completes the proof.

Fact B.6 (Gaussian Norm Concentration). *If* x ∼ N (µ, I)*, with probability* 1 − τ *we have that*

$$|\|x\|^2 - (\|\mu\|^2 + d)| \lesssim \log \frac{1}{\tau} + (\sqrt{d} + \|\mu\|) \sqrt{\log \frac{1}{\tau}}.$$

### B.4. Arithmetic Mean-Geometric Mean Inequality

In this paper, we will use a continuous analog of the *Arithmetic Mean-Geometric Mean* (AM-GM) inequality. The continuous analog for the arithmetic mean of a sequence <sup>1</sup> n P<sup>n</sup> <sup>i</sup>=1 x<sup>i</sup> is what one obtains by replacing the summation with its continuous counterpart. Specifically, the arithmetic mean of a function f : <sup>R</sup> → <sup>R</sup> over an interval I is defined as: (1/|I|) R I f(x)dx.

The geometric mean of a discrete sequence is Q<sup>n</sup> <sup>i</sup>=1 x 1/n i . Its generalization relies on the property that ln Q<sup>n</sup> <sup>i</sup>=1 x 1/n <sup>i</sup> = n P<sup>n</sup> <sup>i</sup>=1 ln x<sup>i</sup> (assuming x<sup>i</sup> > 0). By replacing the summation with an integral, the generalization of the geometric mean of a function <sup>f</sup> over an interval <sup>I</sup> is: exp |I| R I ln f(x)dx . The continuous analog of the AM-GM inequality thus is the following statement. The proof follows directly from Jensen's inequality:

Fact 2.5 (Continuous AM-GM Inequality). *Let* f : R → R + 0 *be a function, and let* I ⊆ <sup>R</sup> *be a finite interval. If* f(x) *and* ln f(x) *are integrable on* I*, then the following holds:* <sup>1</sup> |I| R I <sup>f</sup>(x)d<sup>x</sup> <sup>≥</sup> exp 1 |I| R I ln f(x)dx *.*

## B.5. Statistical Query Lower Bounds Background

We first restate the definition of the non-Gaussian component analysis (NGCA) hypothesis testing problem.

Problem 2.6 (Non-Gaussian Component Analysis (NGCA)). Let B be a distribution on R. For a unit vector v, we denote by PB,v the distribution with the density PB,v(x) := B(v <sup>⊤</sup>x)ϕ⊥v(x), where ϕ⊥v(x) = exp −∥x − (v <sup>⊤</sup>x)v∥ 2 <sup>2</sup>/2 /(2π) (d−1)/2 , i.e., the distribution that coincides with B on the direction v and is standard Gaussian in every orthogonal direction. We define the following hypothesis testing problem:

- H0: The data distribution is N (0, Id).
- H1: The data distribution is PB,v, for some vector v ∈ S<sup>d</sup>−<sup>1</sup> in the unit sphere.

Condition B.7 (Approximate moment matching). Let m ∈ <sup>Z</sup>+. The distribution B on <sup>R</sup> is such that Ex∼B[x i ] − Ex∼N(0,1)[x i ]| ≤ ν for all i ∈ [m].

A known result is that the NGCA problem of Problem [2.6](#page-3-1) is hard in the SQ model if B matches a lot of moments with the standard Gaussian. This was shown in [\(Diakonikolas et al.,](#page-8-3) [2017\)](#page-8-3) and was later strengthened in [Diakonikolas et al.](#page-8-7) [\(2023\)](#page-8-7). The following is Theorem 1.5 in [Diakonikolas et al.](#page-8-7) [\(2023\)](#page-8-7) using λ = 1/2 and c = (1 − λ)/8 = 1/16.

Proposition B.8 (Theorem 1.5 in [Diakonikolas et al.](#page-8-7) [\(2023\)](#page-8-7)). *Let* d, m *be positive integers with* d ≥ (m log d) 2 *. Any SQ algorithm that solves Problem [2.6](#page-3-1) for a distribution* B *satisfying Condition [B.7](#page-12-3) requires either* 2 d Ω(1) *many queries or at least one query with accuracy* d <sup>−</sup>m/<sup>16</sup> + (1 + o(1))ν*.*

# C. Omitted Details from Section [3](#page-3-7)

We restate and prove Theorem [1.3.](#page-1-2)

Theorem 1.3 (SQ Lower Bound for Uniform Weights). *Let* C *be a sufficiently large absolute constant,* k > C *and* d ≥ (log k log d) <sup>2</sup> *be integers. If we further restrict the alternative hypothesis in Problem [1.1](#page-1-1) to have* w<sup>i</sup> = 1/k *for all* i ∈ [k]*, any SQ algorithm requires either* 2 d Ω(1) *queries or at least one query of accuracy* d −Ω(log k) *.*

*Proof.* Let S be the set from Proposition [4.2](#page-5-1) and A be the uniform distribution on S. That is, A is a discrete distribution supported on k points and is guaranteed to match the first m = Ω(log k) moments with N (0, 1). Let B = UρA be the distribution which is obtained by applying the Ornstein-Uhlenbeck operator (Definition [B.2\)](#page-11-2) with ρ = √ δ. Then B is a k-GMM with uniform weights and variance 1 − δ for each component. Moreover, for every t = 0, 1, . . . , m we have the following for the i-th Hermite polynomial

$$\mathbf{E}_{x \sim B}[h_i(x)] = \frac{\mathbf{E}_{x \sim U_\rho A}[h_i(x)]}{\mathbf{E}_{x \sim A}[h_i(x)]} = \rho^i \frac{\mathbf{E}_{x \sim N(0,1)}[h_i(x)]}{\mathbf{E}_{x \sim N(0,1)}[h_i(x)]}, \quad (11)$$

where the above uses Fact [B.3](#page-11-3) and the moment matching property of A. Since Ex∼N(0,1)[hi(x)] = 1 for i = 0 and Ex∼N(0,1)[hi(x)] = 0 for all i > 0, Equation [\(11\)](#page-13-2) means that Ex∼B[hi(x)] = Ex∼N(0,1)[hi(x)], i.e., B matches the first m moments with N (0, 1).

An application of Proposition [B.8](#page-12-0) with ν = 0 shows that the NGCA Problem [2.6](#page-3-1) that uses the distribution B from above has SQ complexity d Ω(log k) . Noting that this problem is equivalent to Problem [1.1](#page-1-1) completes the proof of Theorem [1.3.](#page-1-2)

We conclude by addressing an edge case. The proof above implicitly assumes that the set S contains distinct points (as otherwise, the weights in the corresponding GMM might not all be exactly 1/k). Here, we argue that Theorem [1.3](#page-1-2) still holds even if S contains duplicates. Specifically, one can perturb each point in S by a at most an arbitrarily small amount ∆, ensuring that the points become distinct and that the moments in the resulting GMM distribution B are being matched up to an error of ν rather than exactly (note that for any ν we can find a perturbation so that the moment gap is no more than ν). The SQ lower bound from Proposition [B.8](#page-12-0) then implies that we either require 2 d Ω(1) queries or at least one query with accuracy d <sup>−</sup>m/<sup>16</sup> + (1 + o(1))ν. By choosing ∆ appropriately small, we can ensure that ν < d−m/<sup>16</sup> .

# D. Omitted Details from Section [4](#page-5-5)

### D.1. Omitted Details from Section [4.1](#page-5-2)

The lemma below shows that if the approximate momement matching condition is violated, then it has to be violated by a monomial (up to a small deterioration of parameters).

Lemma D.1. *Let* C *be a sufficiently large absolute constant. If there exists a polynomial* g : R → R *of degree* r *and unit norm (*Ex∼N(0,1)[g 2 (x)] = 1*) such that*

$$\left| \mathbf{E}_{x \sim A}[g(x)] - \mathbf{E}_{x \sim \mathcal{N}(0,1)}[g(x)] \right| > \gamma,$$

*then there exists a monomial* x <sup>i</sup> *with* i ≤ r *for which*

$$\left| \mathbf{E}_{x \sim A}[x^i] - \mathbf{E}_{x \sim \mathcal{N}(0,1)}[x^i] \right| > 2^{-C \cdot r} \gamma.$$

*Proof.* We will show this by contradiction. Suppose that every monomial of degree i ≤ r satisfies Ex∼A[x i ] − Ex∼N(0,1)[x i ]  ≤ 2 <sup>−</sup>Crγ. Then, if we expand g(x) in the hermite basis, i.e., g(x) = P<sup>r</sup> <sup>i</sup>=1 aihi(x), we have

$$\begin{aligned} \left| \mathbf{E}_{x \sim A} [g(x)] - \mathbf{E}_{x \sim \mathcal{N}(0,1)} [g(x)] \right| &\leq \sum_{i=1}^r |a_i| \left| \mathbf{E}_{x \sim A} [h_i(x)] - \mathbf{E}_{x \sim \mathcal{N}(0,1)} [h_i(x)] \right| \\ &\leq \sqrt{\sum_{i=1}^r |a_i|^2} \sqrt{\sum_{i=1}^r \left| \mathbf{E}_{x \sim A} [h_i(x)] - \mathbf{E}_{x \sim \mathcal{N}(0,1)} [h_i(x)] \right|^2}, \end{aligned} \quad (12)$$

where the second step uses Cauchy-Schwarz inequality. To further upper bound this, first note that pP<sup>r</sup> <sup>i</sup>=1 |a<sup>i</sup> <sup>2</sup> = ∥g∥<sup>2</sup> = 1, by Parseval's identity and our assumption of unit norm. For the other factor above, we can write <sup>h</sup>i(x) = P<sup>i</sup> <sup>j</sup>=1 bijx j and use the fact that |bij | ≤ 2 O(i) (Fact [B.1\)](#page-11-4). Then,

$$\begin{aligned} \left| \mathbf{E}_{x \sim A}[h_i(x)] - \mathbf{E}_{\mathcal{N}(0,1)}[h_i(x)] \right| &\leq \sum_{j=1}^i |b_{ij}| \left| \mathbf{E}_{x \sim A}[x^i] - \mathbf{E}_{\mathcal{N}(0,1)}[x^i] \right| \\ &\leq 2^{O(r)} \sum_{j=1}^i \left| \mathbf{E}_{x \sim A}[x^i] - \mathbf{E}_{\mathcal{N}(0,1)}[x^i] \right| \leq 2^{O(r)} r 2^{-Cr} \gamma < 2^{-Cr/2} \gamma, \end{aligned}$$

Plugging that in Equation [\(12\)](#page-13-3), we obtain <sup>|</sup> <sup>E</sup>x∼A[g(x)] <sup>−</sup> <sup>E</sup>N(0,1)[g(x)]| ≤ √ r2 <sup>−</sup>Cr/<sup>2</sup>γ < γ, which gives the desired contradiction.

The lemma below provides a testing algorithm for the NGCA problem (Problem [2.6\)](#page-3-1) in the special case where the distribution <sup>B</sup> is <sup>k</sup>-GMM for which a moment of order at most <sup>m</sup>e is guaranteed to be significantly different than the corresponding moment of N (0, 1).

Lemma D.2 (Testing Algorithm for Parallel Pancakes when the m-th Moment Deviates). *Let* B *be a Gaussian mixture on* R *of the form* B = P<sup>k</sup> <sup>i</sup>=1 wiN (µ<sup>i</sup> , σ<sup>2</sup> )*, where* σ ∈ (0, 1) *and* w<sup>i</sup> ≥ wmin *for all* i ∈ [k]*. For a decreasing sequence* λm*, denote by* m *the biggest integer such that every degree-*m′ ≤ m *polynomial* g *satisfies*

$$\left| \mathbf{E}_{x \sim B} [g(x)] - \mathbf{E}_{x \sim \mathcal{N}(0,1)} [g(x)] \right| \leq \lambda_m \sqrt{\mathbf{E}_{x \sim \mathcal{N}(0,1)} [g^2(x)]}, \quad (13)$$

*Consider the non-Gaussian component analysis hypothesis testing Problem [2.6.](#page-3-1) Let* <sup>m</sup>e *be any upper bound for* <sup>m</sup> *i.e.,* <sup>m</sup> <sup>≤</sup> <sup>m</sup>e *. There is an algorithm that takes as input* <sup>m</sup>e *and* <sup>w</sup>min*, draws* <sup>n</sup> <sup>=</sup> (md e ) <sup>O</sup>(me )λ −O(1) <sup>m</sup><sup>e</sup> + log(k)<sup>w</sup> −1 min log(1/τ ) *samples, and distinguishes correctly between* H<sup>0</sup> *and* H<sup>1</sup> *with probability* 1 − τ *. Moreover, the runtime of the algorithm is polynomial in* n *and* d*.*

*Proof.* We will do the proof for <sup>m</sup>e <sup>=</sup> <sup>m</sup>. The proof trivially extends to any <sup>m</sup>e bigger than <sup>m</sup>. For degree <sup>m</sup> + 1, there exists a polynomial g that violates the condition in Equation [\(13\)](#page-14-0). By Lemma [D.1,](#page-13-4) there exists a monomial x <sup>i</sup> with i ≤ m + 1 such that

$$\tilde{\lambda} = \left| \mathbf{E}_{x \sim B}[x^i] - \mathbf{E}_{\mathcal{N}(0,1)}[x^i] \right| > 2^{-C \cdot m} \lambda_m .$$

For the corresponding d-dimensional distributions PB,v (defined in Problem [2.6\)](#page-3-1) and N (0, I), we have

$$\mathbf{E}_{x \sim P_{B,v}}[x^{\otimes i}] - \mathbf{E}_{x \sim \mathcal{N}(0,I)}[x^{\otimes i}] = \pm \tilde{\lambda} v^{\otimes i}.$$

Thus, the Frobenius norm is

$$\left\| \begin{matrix} \mathbf{E} [x^{\otimes i}] - \mathbf{E}_{x \sim \mathcal{N}(0, I)} [x^{\otimes i}] \end{matrix} \right\|_{\text{F}} = \tilde{\lambda} > 2^{-C \cdot m} \lambda_m .$$

This means that at least one entry in the difference of the two tensors has gap at least ϵ := d <sup>−</sup>(m+1)2 <sup>−</sup>C·<sup>m</sup>λm. The idea for the testing algorithm is to approximate every entry of Ex∼PB,v [x ⊗i ] − Ex∼N(0,I) [x ⊗i ] up to absolute error ϵ/100, and test whether some entry is bigger than ϵ/2. This is done in Algorithm [2](#page-14-1) and Algorithm [3.](#page-15-1)

Algorithm 2 Testing Algorithm

|    | Algorithm | 2      | Testing Algorithm                                                                                                       |
|----|-----------|--------|-------------------------------------------------------------------------------------------------------------------------|
| 1: | Input :   | k, m e | ∈ Z + , w min ∈ (0 , 1]                                                                                                 |
| 2: | Output    | : H ˆ  | ∈ { H 0 , H 1 }                                                                                                         |
| 3: | for i =   | 1 , 2  | , 3 , , m e + 1 do                                                                                                      |
| 4: | Run       |        | Algorithm 3 with input k, i, m, w e min repetitively log(( m e + 1) /τ ) times and let H ˆ be the most frequent output. |
| 5: | if H ˆ    | = H    | 1 then                                                                                                                  |
| 6: |           | Return | H 1                                                                                                                     |
| 7: | Return    | H 0    |                                                                                                                         |

We start with the correctness of the sub-routine, Algorithm [3.](#page-15-1) We say that that the output of Algorithm [3](#page-15-1) is "successful" if it always agrees with the true hypothesis, with the exception of the following case, where mistakes are permitted: this case is when the true hypothesis is H1, the data distribution satisfies maxi∈[k] ∥µi∥<sup>2</sup> ≤ C √ d (recall that µi's are the centers of the k-GMM distribution B) and Ex∼PB,v [x ⊗i ] − Ex∼N(0,I) [x ⊗i F ≤ 2 <sup>−</sup>Cmλm. We will show that the output of Algorithm [3](#page-15-1) is indeed "successful" in this sense with constant probability.

Algorithm 3 Checking the i-th order tensor mismatch

1: Input: k, <sup>m</sup>e <sup>∈</sup> <sup>Z</sup>+, i <sup>∈</sup> <sup>Z</sup>+, wmin <sup>∈</sup> (0, 1]. 2: Output: Hˆ ∈ {H0, H1}. 3: Define <sup>n</sup> = (md e ) Cme λ −C <sup>m</sup><sup>e</sup> + log(k)<sup>w</sup> −1 min for sufficiently large C. 4: Draw x1, . . . , x<sup>n</sup> i.i.d. from the data distribution. 5: if there exists <sup>i</sup> <sup>∈</sup> [n] with <sup>∥</sup>xi∥<sup>2</sup> > C√ d then 6: Output H<sup>1</sup> and terminate. 7: else 8: Form the empirical tensor M = Ex∼S[x ⊗i ]. 9: Let M′ denote the Gaussian tensor Ex∼N(0,I) [x ⊗i ]. 10: if there is an entry in M<sup>i</sup>1,...,j<sup>i</sup> with |M<sup>i</sup>1,...,j<sup>i</sup> − M′ i1,...,j<sup>i</sup> <sup>|</sup> > d−Cm<sup>e</sup> <sup>λ</sup>m<sup>e</sup> then 11: Output H<sup>1</sup> and terminate. 12: Return H0.

Having that claim established, Lemma [D.2](#page-14-2) follows straightforwardly: First, note that the probability of success can be amplified to 1 − τ by repeating the subroutine log(1/τ ) times and taking the majority vote. Second, if the true hypothesis is H1, there exists an i such that Ex∼PB,v [x ⊗i ] − Ex∼N(0,I) [x ⊗i F > 2 <sup>−</sup>Cmλm. Combined with the claim of the previous paragraph about Algorithm [3,](#page-15-1) this ensures that running Algorithm [3](#page-15-1) for that i will be H1, as desired. Similarly, under H0, the output is always H0, which guarantees that the output of Algorithm [2](#page-14-1) matches the true hypothesis.

We now move to showing the claim that Algorithm [3](#page-15-1) is "successful" with constant probability. We examine the following cases:

Case 1 The true hypothesis is H0. In this case, the data distribution is D = N (0, I). By Gaussian norm concentration (Fact [B.6\)](#page-12-4) we have Pr<sup>x</sup>1,...,xn∼N(0,I) [max<sup>i</sup> <sup>∥</sup>xi<sup>∥</sup> > C√ d log n] < 0.01. This means that Algorithm [3](#page-15-1) will enter line [7.](#page-15-2) Then, by standard entry-wise concentration of Gaussian tensors (see e.g., Fact 5.6 and Equation (5.4) in [\(Kothari & Steurer,](#page-9-25) [2017\)](#page-9-25)) we have that if n > d<sup>C</sup> ′<sup>m</sup>/λ<sup>2</sup> <sup>m</sup> for C ′ ≫ C, we will have ∥ Ex∼N(0,I) [x ⊗i ] − Ex∼S[x ⊗i ]∥<sup>∞</sup> < d−Cmλ<sup>m</sup> and thus the condition in [10](#page-15-3) will be false, resulting in the algorithm to output H0.

Case 2 The hypothesis under effect is <sup>H</sup><sup>1</sup> and maxi∈[k] <sup>∥</sup>µi∥<sup>2</sup> > C√ d log n. The claim is that for log(k)/wmin samples, with high constant probability, one sample from every component will be observed, and the sample that comes from the component with <sup>∥</sup>µi∥<sup>2</sup> > C√ <sup>d</sup> log <sup>n</sup> will satisfy <sup>∥</sup>x<sup>∥</sup> > C√ d log n by Fact [B.6.](#page-12-4) To see the first part of the claim, fix an i ∈ [k]. With 10/wmin samples, one sample from i will be observed with at least 0.9 probability. We can boost that probability to 1 − 1/k by repeating log(k) times. Then, by union bound, this means that one sample from each component is observed with constant probability.

Case 3 The hypothesis under effect is H<sup>1</sup> and maxi∈[k] ∥µi∥<sup>2</sup> ≤ C √ d log n. In this case the data distribution is a k-GMM where the center of every Gaussian component is bounded in norm by most R = C √ d log n. By Gaussian norm concentration, if x1, . . . , x<sup>n</sup> are points drawn from that GMM, then with constant probability we will have ∥xi∥<sup>2</sup> ≤ 2C √ d log n. Therefore the algorithm will enter [7,](#page-15-2) and because of the bound ∥xi∥<sup>2</sup> ≤ 2C √ d log n, we can treat the distribution as bounded and use Hoeffding bound for the tensor concentration. That application of Hoeffding's inequality shows that if n > R<sup>C</sup> ′<sup>m</sup>d C ′<sup>m</sup>/λ<sup>C</sup> <sup>m</sup> then the estimation error is at most d <sup>−</sup>Cmλm. Thus, in this case, the algorithm will output H<sup>1</sup> if and only if Ex∼A[x ⊗i ] − Ex∼N(0,1)[x ⊗i F ≤ 2 <sup>−</sup>Cmλm.

This completes the proof of the claim.

Our main result, Theorem [1.4](#page-2-0) will be based on Lemma [D.2](#page-14-2) and our impossibility of matching result, Proposition [4.2](#page-5-1) that will allow us to use <sup>m</sup>e <sup>=</sup> <sup>O</sup>(log(k) + <sup>k</sup> ′ ) in Lemma [D.2.](#page-14-2) However, Proposition [4.2](#page-5-1) concerns only discrete distributions, while the parallel pancakes uses a Gaussian mixture. In order to bridge this difference, we show the following lemma, which states that the impossibility of moment matching can be indeed extended to Gaussian mixtures.

Lemma D.3. *Let* P *be a Gaussian mixture distribution on* R *of the form* B = P<sup>k</sup> <sup>i</sup>=1 wiN (µ<sup>i</sup> , 1 − δ)*, where* w<sup>i</sup> > 0 *with* P<sup>k</sup> <sup>i</sup>=1 w<sup>i</sup> = 1 *are the weights of each component,* µ1, . . . , µ<sup>k</sup> ∈ <sup>R</sup> *are the centers and* δ ∈ (0, 1] *is the parameter* *associated with the common variance of the components. Suppose that for every polynomial of degree at most* m′ *and* Ex∼N(0,1)[p 2 (x)] = 1 *the following holds*

$$\left| \mathbf{E}_{x \sim B} [p(x)] - \mathbf{E}_{x \sim \mathcal{N}(0,1)} [p(x)] \right| \leq \lambda. \quad (14)$$

*Then, if* A *denotes the discrete distribution on* {µ1/ √ δ, . . . , µi/ √ δ} *that assigns mass* w<sup>i</sup> *to the point* µi/ √ δ *for* i ∈ [k]*, the following is true: For every polynomial with degree at most* m′ *and* Ex∼N(0,1)[p (x)] = 1 *it holds*

$$\left| \mathbf{E}_{x \sim A} [p(x)] - \mathbf{E}_{x \sim \mathcal{N}(0,1)} [p(x)] \right| \leq \sqrt{m'} \lambda \delta^{-m'/2}. \quad (15)$$

*Proof.* We can write the distribution B as the result of applying the Ornstein-Uhlenbeck (Definition [B.2\)](#page-11-2) operator to A, i.e., B = UρA with ρ = √ δ. By Fact [B.3,](#page-11-3) we have the following for every i = 1, 2, . . .:

$$\mathbf{E}_{x \sim A}[h_i(x)] = \delta^{-i/2} \mathbf{E}_{x \sim B}[h_i(x)]. \quad (16)$$

Fix i ∈ [m′ ]. Using the above and the fact that B matches approximately the m first moments with N (0, 1) (in the sense of Equation [\(14\)](#page-16-0)) we have the following for the gap between the expectations of the Hermite polynomial h<sup>i</sup> under A and N (0, 1):

$$\begin{aligned} \left| \mathbf{E}_{x \sim \mathcal{A}} [h_i(x)] - \mathbf{E}_{x \sim \mathcal{N}(0,1)} [h_i(x)] \right| &= \left| \delta^{-i/2} \mathbf{E}_{x \sim \mathcal{B}} [h_i(x)] - \mathbf{E}_{x \sim \mathcal{N}(0,1)} [h_i(x)] \right| \quad (\text{using Equation (16)}) \\ &= \left| \delta^{-i/2} \mathbf{E}_{x \sim \mathcal{B}} [h_i(x)] \right| \quad (\text{using } \mathbf{E}_{x \sim \mathcal{N}(0,1)} [h_i(x)] = 0 \text{ for } i \geq 1) \\ &\leq \delta^{-i/2} \left( \left| \mathbf{E}_{x \sim \mathcal{N}(0,1)} [h_i(x)] \right| + \lambda \right) \quad (\text{using Equation (14)}) \\ &= \delta^{-i/2} \lambda \quad (\text{using } \mathbf{E}_{x \sim \mathcal{N}(0,1)} [h_i(x)] = 0 \text{ for } i \geq 1) \end{aligned}$$

For the special case i = 0, we have exact matching, Ex∼A[h0(x)] = Ex∼N(0,1)[h0(x)] since h0(x) = 1.

Now, in order to show Equation [\(15\)](#page-16-2), consider a general polynomial p(x) with Ex∼N(0,1)[p 2 (x)] = 1. Expanding in the Hermite basis, we can write p(x) = P i∈[m′ aihi(x) with P i∈[m′ a 2 <sup>i</sup> = 1 (which means that Ex∼N(0,1)[p 2 (x)] = 1 by Parseval's identity). We have

$$\begin{aligned} \left| \mathbf{E}_{x \sim A}[p(x)] - \mathbf{E}_{x \sim \mathcal{N}(0,1)}[p(x)] \right| &\leq \sqrt{\sum_{i=1}^{m'} a_i^2} \sqrt{\sum_{i=1}^{m'} \left| \mathbf{E}_{x \sim A}[h_i(x)] - \mathbf{E}_{x \sim \mathcal{N}(0,1)}[h_i(x)] \right|^2} \\ &\leq \sqrt{m'} \max_{i \in [m']} \left| \mathbf{E}_{x \sim A}[h_i(x)] - \mathbf{E}_{x \sim \mathcal{N}(0,1)}[h_i(x)] \right| \\ &\leq \sqrt{m'} \delta^{-m'/2} \lambda. \end{aligned}$$

We now combine the previous statements to show our main theorem.

Theorem 1.4 (Testing Algorithm for Parallel Pancakes). *Consider the version of the parallel pancakes hypothesis testing problem (Problem [1.1\)](#page-1-1), where* k ′ ≤ k *of the weights* w<sup>i</sup> *in the Gaussian mixture are unconstrained and the remaining* k − k ′ *are assumed to be equal to each other. There is an algorithm for that problem which draws* n = O (kd/δ) O(k ′+log(k)) + log(k)/wmin *samples (where* δ *is as in Problem [1.1](#page-1-1) and* wmin = mini∈[k] w<sup>i</sup> *is the smallest weight), has runtime polynomial in* n, d*, and it outputs the correct hypothesis with probability at least* 0.99*.*

*Proof.* First, we note that the parallel pancakes testing problem of interest is a special case of the non-Gaussian component analysis Problem [2.6](#page-3-1) where B = P <sup>i</sup>∈[k] wiN (µ<sup>i</sup> , 1 − δ), where w<sup>i</sup> , µ<sup>i</sup> and δ the ones from Problem [1.1,](#page-1-1) in particular k ′ of the wi's are unconstrained and the rest are assumed to be uniform.

The proof consists of two parts: The first part argues that this one-dimensional distribution B cannot match approximately more than the first m = O(log k +k ′ ) moments with N (0, I) (the approximate moment matching will be quantified shortly). Then, for the second part, we can show that since the m + 1 moment deviates significantly from that of N (0, I), estimating the empirical moment tensor of order m + 1 and comparing with the one from N (0, I) yields a successful test.

We now proceed with the quantification. Let C be a sufficiently large constant, and define m to be the largest integer such that for every polynomial p of degree m′ ≤ m and ∥p∥<sup>2</sup> = 1 we have

$$\left| \mathbf{E}_{x \sim A} [p(x)] - \mathbf{E}_{x \sim \mathcal{N}(0,1)} [p(x)] \right| \leq \lambda_m . \quad (17)$$

To prove our claim by contradiction, suppose that m > C(k ′ + log k). For each degree m′ ≤ m We will use Lemma [D.3](#page-15-0) with λ = (δ/2)Cm. The application of Lemma [D.3](#page-15-0) yields that the discrete distribution A supported on a scaled version of the centers µ<sup>i</sup> and using the same weights w<sup>i</sup> approximately matches the same m first moments with N (0, 1), i.e., for every polynomial p of degree m′ ≤ m and ∥p∥<sup>2</sup> = 1 we have

$$\left| \mathbf{E}_{x \sim D} [p(x)] - \mathbf{E}_{x \sim \mathcal{N}(0,1)} [p(x)] \right| \leq \sqrt{m} \lambda_m \delta^{-m/2} \leq 2^{-Cm/2}. \quad (18)$$

where the last step uses that λ = (2δ) <sup>−</sup>Cm.

The conclusion of Equation [\(18\)](#page-17-1) contradicts Proposition [4.1.](#page-5-0) This is because the discrete distribution D from above, is of the form that Proposition [4.1](#page-5-0) considers: supported on k points, with k ′ of the points having arbitrary mass and the remaining k − k ′ having equal masses.

Thus far, we have shown that if m is the largest degree for which all moments m′ ≤ m of the distribution A match with N (0, 1) in the sense of Equation [\(17\)](#page-17-2), then m = O(log(k) + k ′ ).

The result then follows by Lemma [D.2](#page-14-2) with <sup>m</sup>e <sup>=</sup> <sup>C</sup>(log(k) + <sup>k</sup> ′ ) for a sufficiently large constant C, σ <sup>2</sup> = 1 − δ, and <sup>λ</sup>m<sup>e</sup> = (δ/2)Cm<sup>e</sup> .

## D.2. Omitted Details from Section [4.2](#page-5-3)

We restate and prove a version of Proposition [4.2](#page-5-1) which does not involve minimum weight w<sup>0</sup> of points with equal weights.

Proposition 4.1. *Let* k ′ < k *be positive integers, and let* A *be a discrete distribution on* k *points in* <sup>R</sup>*. Suppose* k − k ′ *of the points have equal probability masses, while the remaining* k ′ *points have unrestricted probability masses. Denote by* m *the highest degree for which every degree-*m′ ≤ m *polynomial* g *satisfies* Ex∼A[g(x)] − Ex∼N(0,1)[g(x)]  ≤ 2 <sup>−</sup>C·<sup>m</sup>∥g∥2*, then* m *must satisfy* m ≤ O(log k) + O(k ′ )*.*

*Proof.* Suppose that the order m is bigger than C log k + Ck′ . If C is sufficiently large, we will show that this moment matching is impossible.

Let µ1, . . . , µ<sup>k</sup> be the points on which A is supported, and by w1, . . . , w<sup>k</sup> the probability masses of the points. Without loss of generality, assume that the first k ′ points are the ones which do not have any restriction on their probability mass, and the remaining k − k ′ are the points with equal probability masses (w<sup>i</sup> = w<sup>j</sup> for all i, j ∈ {k ′ + 1, . . . , k} with i ̸= j). Let p(x) = (x − µ1)· · ·(x − µk′ ) be the polynomial whose roots are the first k ′ points.

We will show the following series of inequalities (we use the notation ∥p∥<sup>r</sup> = Ex∼N(0,1)[|p(x)| r <sup>1</sup>/r):

$$\sum_{i=k'+1}^k w_i \geq \left( \frac{\mathbf{E}_{x \sim A}[p(x)]}{\mathbf{E}_{x \sim A}[p^2(x)]^{1/2}} \right)^2 \gtrsim \left( \frac{\|p\|_1}{\|p\|_2} \right)^2 \geq 3^{-2k'}. \quad (19)$$

The third step is Fact [2.4.](#page-3-4) To see how the first step is derived, let E be the event that i ∈ {k ′ + 1, . . . , k}. Then

$$\|p\|_1 = \mathbf{E}_{\mu_i \sim A}[p(\mu_i)] = \mathbf{E}_{\mu_i \sim A}[p(\mu_i) \mathbb{1}(\mathcal{E})] \leq \sqrt{\mathbf{E}_{\mu_i \sim A}[p^2(\mu_i)] \sqrt{\mathbf{Pr}[\mathcal{E}]}} = \|p\|_2 \sqrt{\mathbf{Pr}[\mathcal{E}]} = \|p\|_2 \sqrt{\sum_{i=k'+1}^k w_i},$$

where the first step above uses the fact that µ1, . . . , µk′ are roots of <sup>p</sup>. Rearranging gives P<sup>k</sup> <sup>i</sup>=k′ w<sup>i</sup> ≥ (∥p∥1/∥p∥2) 2 .

It remains to show the second step in Equation [\(19\)](#page-17-3), which is due to the approximate moment matching property: Let λ<sup>m</sup> := 2−Cm to save space. For the numerator, we have Ex∼A[p(x)] ≥ ∥p∥<sup>1</sup> − λm∥p∥<sup>2</sup> ≥ ∥p∥1(1 − λm3 k ) ≥ ∥p∥1/2, where the first step uses the approximate moment matching, the second step uses Fact [2.4](#page-3-4) and the last part uses that λ<sup>m</sup> := w02 <sup>−</sup>Cm with C being sufficiently large constant and m > k′ . We can work similarly for the denominator to get Ex∼A[p 2 (x)] ≤ ∥p∥ 2 <sup>2</sup> + λm∥p∥ <sup>4</sup> ≤ ∥p∥ 2 2 (1 + λm3 k ) ≤ 2∥p∥ 2 , where we used Fact [2.3](#page-3-0) in the penultimate step. Combining the bounds for numerator and denominator conclude the proof of the second step in Equation [\(19\)](#page-17-3).

We can now conclude the proof of Theorem [1.4.](#page-2-0) Since we have assumed that the weights for the last k − k ′ points are equal to each other, Equation [\(19\)](#page-17-3) implies that mini=k′+1,...,k w<sup>i</sup> ≥ 3 −2k ′ /k. Using Proposition [4.2](#page-5-1) with w<sup>0</sup> = 3−2<sup>k</sup> ′ /k concludes the proof.

## D.3. Omitted Details from Section [4.2.1](#page-6-5)

We restate and prove the following corollary of Lemma [4.4.](#page-6-2)

Corollary 4.5. *Let* p : <sup>R</sup> → <sup>R</sup> *be a polynomial of the form* p(x) = (x − µ1)(x − µ2)· · ·(x − µk′ ) *where* µ1, . . . , µk′ ∈ <sup>R</sup> *are arbitrary parameters. Define* I = [0.9 √ 2t, 1.1 √ <sup>2</sup>t]*. For all* <sup>t</sup> <sup>≥</sup> <sup>1</sup> *we have* exp |I| R x∈I ln |p(x)|dx ≥ ∥p∥<sup>2</sup> 2O(k′) *.*

*Proof.* We can multiply both sides of the conclusion of Lemma [4.4](#page-6-2) (Equation [\(5\)](#page-6-4)) with the Gaussian density e −y <sup>2</sup>/<sup>2</sup>/ √ 2π and then integrate both sides. This yields

$$\int_{-\sqrt{t}}^{\sqrt{t}} \frac{1}{\sqrt{2\pi}} e^{-y^2/2} \exp\left(\frac{1}{|I|} \int_{x \in I} \ln |p(x)| dx\right) dy \geq \int_{-\sqrt{t}}^{\sqrt{t}} \frac{|p(y)|}{2^{O(k')}} \frac{1}{\sqrt{2\pi}} e^{-y^2/2} dy.$$

The left hand side is Θ exp (1/|I|) R x∈I ln |p(x)|dx . The right hand side is

$$\begin{aligned} \frac{1}{\sqrt{2\pi}} \int_{-\sqrt{t}}^{\sqrt{t}} e^{-y^2/2} \frac{|p(y)|}{2^{O(k')}} dy &= \left( y_{\sim \mathcal{N}(0,1)} [|p(y)|] - y_{\sim \mathcal{N}(0,1)} [|p(y)| \mathbb{1}(|y| > \sqrt{t})] \right) 2^{-O(k')} \\ &\geq \left( y_{\sim \mathcal{N}(0,1)} [|p(y)|] - \|p\|_2 \sqrt{y_{\sim \mathcal{N}(0,1)} [|y| > t]} \right) 2^{-O(k')} \quad (\text{using the Cauchy-Schwarz inequality}) \\ &\geq \left( y_{\sim \mathcal{N}(0,1)} [|p(y)|] - \|p\|_2 e^{-t^2/2} \right) 2^{-O(k')} \\ &\geq \left( \|p\|_2 - \|p\|_2 e^{-t^2/2} \right) 2^{-O(k')} \quad (\text{using Fact 2.4}) \\ &= \|p\|_2 / 2^{O(k')}. \quad (\text{using } t \geq 1) \end{aligned}$$

We now restate Lemma [4.4](#page-6-2) and provide the complete proof which includes the details that were missing from Section [4.2.1.](#page-6-5) Corollary 4.5. *Let* p : <sup>R</sup> → <sup>R</sup> *be a polynomial of the form* p(x) = (x − µ1)(x − µ2)· · ·(x − µk′ ) *where* µ1, . . . , µk′ ∈ <sup>R</sup> *are arbitrary parameters. Define* I = [0.9 √ 2t, 1.1 √ <sup>2</sup>t]*. For all* <sup>t</sup> <sup>≥</sup> <sup>1</sup> *we have* exp 1 |I| R x∈I ln |p(x)|dx ≥ ∥p∥<sup>2</sup> 2O(k′) *.*

*Proof.* Fix an arbitrary <sup>y</sup> <sup>∈</sup> <sup>R</sup> with <sup>|</sup>y| ≤ √ t. First, note that by the property of logarithms and sums, we can write the left hand side as

$$\exp \left( \sum_{i=1}^{k'} \frac{1}{|I|} \int_{x \in I} \ln |x - \mu_i| dx \right).$$

In order to show Equation [\(5\)](#page-6-4), it suffices to work with each term and show the following for each i ∈ [k ′ ]:

$$\frac{1}{|I|} \int_{x \in I} \ln |x - \mu_i| \geq \ln |y - \mu_i| - O(1) .$$

Equivalently, it suffices to show that Equation [\(5\)](#page-6-4) holds for every linear polynomial of the form p(x) = x − a. Therefore, the goal for the rest of this proof is to show that

$$\exp\left(\frac{1}{|I|} \int_{x \in I} \ln |x - a| dx\right) \geq |y - a|/O(1) , \quad (20)$$

holds for every <sup>a</sup> <sup>∈</sup> <sup>R</sup> and <sup>y</sup> <sup>∈</sup> <sup>R</sup> with <sup>|</sup>y| ≤ √ t. We will examine two cases.

Case 1 The first case is when the root a of the polynomial is outside the interval I. In this case, we can show that |x − a|/|y − a| = Θ(1), which implies ln |x − a| ≥ ln |y − a| − O(1), and the desired conclusion (Equation [\(20\)](#page-19-0)) follows by integrating both sides and applying the exp(·) function.

To show the earlier claim that |x − a|/|y − a| = Θ(1), we can consider the following sub-cases:

- 1. Case a ≥ 1.1 √ 2t (i.e., a is to the right of I): Suppose a = 1.1 √ 2t + u for some non-negative u. Then, a − x = (1.1 √ <sup>2</sup><sup>t</sup> <sup>−</sup> <sup>x</sup>) + <sup>u</sup> = Θ(√
  - t) + u and a − y = (1.1 √ <sup>2</sup><sup>t</sup> <sup>−</sup> <sup>y</sup>) + <sup>u</sup> = Θ(√
  - t) + u. Therefore, for any u ≥ 0, the ratio <sup>|</sup><sup>x</sup> <sup>−</sup> <sup>a</sup>|/|<sup>y</sup> <sup>−</sup> <sup>a</sup><sup>|</sup> = (Θ(√
    - <sup>t</sup>) + <sup>u</sup>)/(Θ(√
      - t) + u) = Θ(1).
- 2. The cases a < − √ t and a ∈ [ √ t, 0.9 √ 2t] can be shown in a similar manner.

Case 2 The complementary case is when the root a of the polynomial p belongs in the interval I. In that case,

$$\frac{1}{|I|} \int_{x \in I} \ln |x-a| dx = \frac{1}{|I|} \int_a^{1.1\sqrt{2t}} \ln(x-a) dx + \frac{1}{|I|} \int_{0.9\sqrt{2t}}^a \ln(a-x) dx.$$

Define A := <sup>1</sup> 0.2 √ 2t R <sup>1</sup>.<sup>1</sup> √ 2t a ln(x − a)dx and B := <sup>1</sup> 0.2 √ 2t R a 0.9 √ 2t ln(a − x)dx. We will work with each integral separately. For A, we have the following (after a change of variable in the integral):

$$\begin{aligned} A &= \frac{1}{0.2\sqrt{2t}} \int_0^{1.1\sqrt{2t}-a} \ln z \, dz = \frac{1}{0.2\sqrt{2t}} [-z + z \ln z]_{z=0}^{z=1.1\sqrt{2t}-a} \\ &= - \left( 5.5 - \frac{a}{0.2\sqrt{2t}} \right) + \left( 5.5 - \frac{a}{0.2\sqrt{2t}} \right) \ln \left( 1.1\sqrt{2t}-a \right). \end{aligned}$$

Recalling that we have assumed a ∈ [0.9 √ 2t, 1.1 √ 2t], we can rewrite the above as A = −C<sup>1</sup> + C<sup>1</sup> ln(1.1 √ 2t − a), where C<sup>1</sup> = 5.5 − a 0.2 √ 2t ∈ [0, 1].

We now work with the integral defined as A previously in a similar way:

$$\begin{aligned} B &= \frac{1}{0.2\sqrt{2t}} \int_0^{a-0.9\sqrt{2t}} \ln z \, dz = \frac{1}{0.2\sqrt{2t}} [-z + z \ln z]_{z=0}^{z=a-0.9\sqrt{2t}} \\ &= - \left( \frac{a}{0.2\sqrt{2t}} - 4.5 \right) + \left( \frac{a}{0.2\sqrt{2t}} - 4.5 \right) \ln(a - 0.9\sqrt{2t}). \end{aligned}$$

Taking into consideration that a ∈ [0.9 √ 2t, 1.1 √ 2t] the above can be written as B = −C<sup>2</sup> + C<sup>2</sup> ln(a − 0.9 √ 2t), where C<sup>2</sup> = a 0.2 √ 2t − 4.5 ∈ [0, 1].

Combining the bounds for A and B together with the definitions C<sup>1</sup> = 5.5 − a 0.2 √ 2t and C<sup>2</sup> = a 0.2 √ 2t − 4.5, we obtain exp(A + B) = exp(f(a) − 1), where f(a) is the function

$$f(a) := \left( 5.5 - \frac{a}{0.2\sqrt{2t}} \right) \ln(1.1\sqrt{2t} - a) + \left( \frac{a}{0.2\sqrt{2t}} - 4.5 \right) \ln(a - 0.9\sqrt{2t}) ,$$

We can verify through derivative analysis that the minimum is achieved at the midpoint of I, i.e., for a = √ 2t:

$$f'(a) = \frac{5}{\sqrt{2t}} \left( \ln(10a - 9\sqrt{2t}) - \ln(11\sqrt{2t} - 10a) \right).$$

It is easy to see that f ′ ( √ 2t) = 0. Furthermore, the second derivative is f ′′(a) = 1/(t/50 − (a − √ 2t) 2 ), which is non-negative for all a ∈ I. Thus, the only minimizer in I is a = √ 2t. For that point, exp(A + B) becomes:

$$\exp(A + B) \geq \exp\left(\frac{\ln(t/50)}{2} - 1\right) \geq \frac{\sqrt{t}}{20} \geq \frac{|y - a|}{52},$$

where we used <sup>|</sup><sup>y</sup> <sup>−</sup> <sup>a</sup>| ≤ |y<sup>|</sup> <sup>+</sup> <sup>|</sup>a| ≤ √ t + 1.1 √ 2t < 2.6 √ t.