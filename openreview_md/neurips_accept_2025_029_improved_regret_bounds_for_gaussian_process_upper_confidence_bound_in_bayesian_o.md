# Improved Regret Bounds For Gaussian Process Upper Confidence Bound In Bayesian Optimization

Shogo Iwazaki LY Corporation Tokyo, Japan siwazaki@lycorp.co.jp

## Abstract

This paper addresses the Bayesian optimization problem (also referred to as the Bayesian setting of the Gaussian process bandit), where the learner seeks to minimize the regret under a function drawn from a known Gaussian process (GP). Under a Matern kernel with a certain degree of smoothness, we show that the Gaussian ´
process upper confidence bound (GP-UCB) algorithm achieves e(
√) cumulative regret with high probability. Furthermore, our analysis yields (
√︁ ln2 ) regret under a squared exponential kernel. These results fill the gap between the existing regret upper bound for GP-UCB and the best-known bound provided by Scarlett [46]. The key idea in our proof is to capture the concentration behavior of the input sequence realized by GP-UCB, enabling a more refined analysis of the GP's information gain.

## 1 Introduction

We study the Bayesian optimization (BO) problem, where the learner seeks to minimize the regret under a random function drawn from a known Gaussian process (GP) [18, 19]. Throughout this paper, we focus on the GP-UCB algorithm [51], which combines the posterior distribution of GP with the optimism principle. Due to its simple algorithm construction and general theoretical framework provided by Srinivas et al. [51], GP-UCB has played an important role in the advancement of the BO field. On the other hand, our theoretical understanding of the performance of GP-UCB has not been improved from [51] in the Bayesian setting, while its frequentist counterpart is studied in several existing works [11, 61]. Specifically, the current regret upper bound for GP-UCB, as provided by Srinivas et al. [51], is known to be worse than that of the algorithm in [46], which achieves stateof-the-art (
√
 ln) cumulative regret. Then, the natural question is whether there is further room for improvement in the existing regret upper bound of GP-UCB. This paper provides an affirmative answer to this question by showing that GP-UCB achieves e(
√) regret with high probability.

Contribution. We summarize our contributions as follows.

- We show that the GP-UCB proposed by Srinivas et al. [51] achieves e(
√) regret with high probability under a Matern kernel with a certain degree of smoothness (precise condition ´
is provided in Theorem 3). Here, e(·) is the order notation that hides polylogarithmic dependence. This result is comparable to state-of-the-art (
√
 ln) regret provided by Scarlett [46] up to a polylogarithmic factor and strictly improves upon the existing e(
+
2+ )
upper bound of GP-UCB [51, 58]. Here,  and  denote the dimension of the input domain and smoothness parameter, respectively.

- Furthermore, for a squared exponential kernel, we establish 
√︁
 ln2 cumulative regret of GP-UCB. This improves the existing 
√︁
 ln+2 upper bound provided by Srinivas et al. [51] for any  ≥ 1.

- The key idea behind our analysis is to refine the existing information gain bounds by leveraging algorithm-dependent behavior and sample path properties of the GP. We also discuss the applicability of this technique to other algorithms and settings in Section 4.

## 1.1 Related Works

BO has been extensively studied in the past few decades. Some of them are constructed so as to maximize the utility-based acquisition function defined through the GP posterior, including expected improvement [37], knowledge gradient [17], and the entropy-based algorithms [24]. The theoretical aspect of BO has also been actively studied through the lens of the bandit algorithms, such as GP- UCB [51], Thompson sampling [43], and information directed sampling [44]. In contrast to the noisy observation setting, which these algorithms focus on, algorithms for the noise-free setting form a separate line of research [14, 23, 32]. Extensions of these algorithms to more advanced settings have also been well-studied, e.g., contextual [34], parallel observation [15], high-dimensional [29], timevarying [6], and multi-fidelity setting [30]. Unlike the Bayesian assumption on the objective function adopted in this paper, existing works also extensively study the frequentist assumption of the function, which is also referred to as the frequentist setting of BO or GP bandits [7, 9, 11, 26, 35, 45, 47, 56, 59].

Among the existing studies, [46] is closely related to this paper, which propose a successive elimination-based algorithm and shows an (
√
 ln) upper bound and an Ω(
√) lower bound of the cumulative regret for a one-dimensional BO problem. The fundamental theoretical assumptions and the high-level idea of our analysis are built on the proof provided by Scarlett [46]. Following [46], Wang et al. [60] also proves similar regret guarantees under the one-dimensional Brownian motion. In addition to [46], some parts of our analysis are inspired by the technique leveraged in [8, 28]. Firstly, Cai et al. [8] studies the GP-UCB algorithm through a relaxed version of regret, which is called *lenient regret*. In our analysis, the cumulative regret is decomposed into the lenient regretbased term, and we leverage their technique to analyze it. Secondly, Janz et al. [28] proposed the input partitioning-based algorithm for obtaining a superior regret in the frequentist setting. Roughly speaking, the high-level idea of their analysis is based on the fact that tighter information gain bounds can be obtained within a properly shrinking partition of the input. The key idea provided in Section 3.1 is motivated by this fact, while our analysis itself is substantially different from that in [28].

## 2 Preliminaries

Let  : X → R be a black-box objective function whose input domain X is X := [0, ]
 with some  > 0. At each step  ∈ N+, the learner chooses a query point x ∈ X, and then receives a noisy observation  =  (x) + . Here, is a mean-zero noise random variable. We consider a Bayesian setting, where the objective function  and the noise sequence () are drawn from a known zero-mean Gaussian process (GP) and a Gaussian distribution, respectively. We formally describe it using the following assumptions.

Assumption 1. Let  : X × X → R *be the known positive definite kernel with* ∀x ∈ X,  (x, x) ≤ 1. Then, assume  ∼ GP (0, )*, where* GP (0, ) denotes the mean-zero GP characterized by the covariance function .

Assumption 2. The noise sequence () ∈N+*is mutually independent. Furthermore, assume*  ∼
N (0, 2), where  > 0 *is the known constant. Here,* N (, 2) *denotes the Gaussian distribution* with mean  *and variance* 2.

These are standard sets of assumptions in the existing theory of BO [43, 51]. Specifically, in Assumption 1, we focus on the following squared exponential (SE) kernel SE and Matern kernel ´
Algorithm 1 Gaussian process upper confidence bound Require: Kernel , confidence width parameters (
1/2

) ∈N+.

1: for  = 1, 2*, . . .* do 2: x ← arg maxx∈X (x; X−1, y−1) + 
1/2
 (x; X−1).

3: Observe  and update the posterior mean and variance.

4: **end for**

$k_{\text{Mathem}}$:  $$k_{\text{SE}}(\mathbf{x},\widehat{\mathbf{x}})=\exp\left(-\frac{\|\mathbf{x}-\widehat{\mathbf{x}}\|_{2}^{2}}{2\ell^{2}}\right),\ \ k_{\text{Mathem}}(\mathbf{x},\widehat{\mathbf{x}})=\frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}\|\mathbf{x}-\widehat{\mathbf{x}}\|_{2}}{\ell}\right)^{\nu}J_{\nu}\left(\frac{\sqrt{2\nu}\|\mathbf{x}-\widehat{\mathbf{x}}\|_{2}}{\ell}\right),\tag{1}$$

$$(2)$$
$$({\mathfrak{I}})$$
where ℓ > 0 and  > 0 are the known lengthscale and smoothness parameters, respectively. In addition,  (·) and Γ(·) respectively denote modified Bessel and Gamma functions. Under Assumptions 1 and 2, the learner can infer the function  through the GP posterior distribution. Let
H

:= (x 
, )≤ be the history that the learner obtained up to the end of step . Given H, the
posterior distribution of  is again GP, whose posterior mean and variance are respectively defined
as
$$\begin{array}{c}{{\mu(x;{\bf X}_{t},y_{t})=k({\bf X}_{t},x)^{\top}({\bf K}({\bf X}_{t},{\bf X}_{t})+\sigma^{2}{\bf I}_{t})^{-1}y_{t},}}\\ {{\sigma^{2}(x;{\bf X}_{t})=k(x,x)-k({\bf X}_{t},x)^{\top}({\bf K}({\bf X}_{t},{\bf X}_{t})+\sigma^{2}{\bf I}_{t})^{-1}k_{t}({\bf X}_{t},x),}}\end{array}$$
 x), (3)
where k(X 
, x) := [ (x, xe)]x∈X
and y

:= (1*, . . . ,* )
⊤ are the -dimensional kernel and output vectors, respectively. Here, we set X = (x1*, . . . ,* x). Furthermore, K(X
,
 X) := [ (x, xe)]x,xe∈X
and I denote  × -gram matrix and  × -identity matrix, respectively.

Learner's goal. Under the total step size  ∈ N+, the learner's goal is to minimize the cumulative regret  :=Í
 ∈ []
 (x
∗) −  (x), where x
∗ ∈ argmaxx∈X  (x) and [] = {1*, . . . ,* }.

Maximum information gain. To quantify the regret, the existing theory utilizes the following information-theoretic quantity  (X) arising from GP:

$$\gamma_{T}(X)=\sup_{\mathbf{x}_{1},\ldots,\mathbf{x}_{T}\in X}I(\mathbf{X}_{T}),\text{where}I(\mathbf{X}_{T})=\frac{1}{2}\ln\det(\mathbf{I}_{T}+\sigma^{-2}\mathbf{K}(\mathbf{X}_{T},\mathbf{X}_{T})).\tag{4}$$

The quantity  (X) is referred to as the *maximum information gain* (MIG) over X [51], since (X) equals the mutual information between the function values (  (x))
 ∈ [] and the outputs () ∈ []
under Assumptions 1 and 2, and the input sequence X = (x1*, . . . ,* x). MIG plays a vital role in the theoretical analysis of BO, and its increasing speed is analyzed in several commonly used kernels.

For example,  (X) = (ln+1 ) as  → ∞ under  = SE [51]. For the notational convenience, we also define (X) = ⌈⌉ (X) for any non-integer  > 0.

Probabilistic property of GP sample path. The existing theory of GP-UCB under the Bayesian setting utilizes the regularity conditions of the realized sample path of GP. We summarize the existing known properties of the GP sample path in the following lemmas.

Lemma 1 (Lipchitz condition of sample path, e.g., [51]). *Suppose*  = SE or  = Matern ´ *with*
 >
 2. Assume Assumption 1. Then, there exist the constants ,  > 0 *such that*

$$\forall L>0,\;\mathbb{P}\left(\forall x,{\widetilde{\mathbf{x}}}\in\mathcal{X},\;|f(\mathbf{x})-f({\widetilde{\mathbf{x}}})|\leq L\|\mathbf{x}-{\widetilde{\mathbf{x}}}\|_{1}\right)\geq1-d a\exp\left(-{\frac{L^{2}}{b^{2}}}\right).$$
 (5)
Lemma 2 (Sample path condition for the global maximizer, e.g., [13, 14, 46]). *Suppose*  = SE or = Matern ´ with  > 2*. Assume Assumption 1. Then, for any* GP ∈ (0, 1), there exist the strictly positive constants gap, sup, quad, quad > 0 such that the following statements simultaneously hold with probability at least 1 − GP:
1. The function  *has a unique maximizer* x
∗ ∈ X *such that*  (x
∗) >  (xe
∗) + gap holds for any local maximizer xe
∗ ∈ X of  .

$$(S)$$

2. *The sup-norm of the sample path is bounded as* ∥  ∥∞ ≤ sup. 3. The function  *satisfies* ∀x ∈ B2 (quad; x
∗),  (x
∗) − quad ∥x
∗ − x∥
2 2
≥  (x)*, where* B2 (quad; x
∗) := {x ∈ X | ∥x
∗ −x∥2 ≤ quad} is the L2-ball on X*, whose radius and center* are quad and x
∗*, respectively.*
Lemma 1 states that the sample path  of GP is a Lipschitz function with high probability. This property is leveraged in the theory of GP-UCB to control the discretization error arising from the confidence bound construction in the continuous input domain. As described in [51], Lemma 1 is a direct consequence of Theorem 5 in [21] under the existence of fourth-order mixed partial derivatives of the kernel, which are satisfied under  = SE and  = Matern ´ with  > 2 1. Lemma 2 specifies the regularity condition of  related to the maximizer x
∗. Here, property 1 is implied from the fact that the GP-sample path has a unique maximizer almost surely under SE and Matern ´ [e.g., Lemma 2.6 in 33]. Property 2 is implied from, e.g., the compactness of X and the almost-sure continuity of the sample path under SE and Matern ´. Property 3 also holds automatically under  = SE and

 = Matern ´ with  > 2 and is used in existing works. See Theorem 5 in [13], Assumption 3 in [46],
and the discussions provided by them for further details. Note that the properties in Lemma 2 are not used in the existing proof of GP-UCB in [51]. As described in the next section, we analyze the realized input sequence X of GP-UCB by relating it to conditions in Lemma 2. Summary of existing analysis of GP-UCB. We briefly summarize the existing analysis of GP- UCB (Algorithm 1) provided by Srinivas et al. [51]. Based on Assumptions 1 and 2, we can construct the high-probability confidence bound of the underlying function value  (x) for each x and  ∈ N+ through the posterior distribution of  (x). Specifically, by choosing a properly designed finite representative input set X ⊂ X and taking into account the discretization error with Lemma 1, Srinivas et al. [51] showed the following events hold simultaneously with probability at least 1 − :
1. **Confidence bound.** For any  ∈ N+, the function value at the queried point x satisfies
(x; X−1, y−1) − 
1/2
 (x; X−1) ≤  (x). Furthermore, for any  ∈ N+, any function value  (x) on X satisfies  (x) ≤ (x; X−1, y−1) + 
1/2
 (x; X−1).

2. **Discretization error.** The discretization error arising from Xis at most 1/
2. Namely, |  (x) −  ( [x])| ≤ 1/
2 holds for any x ∈ X and  ∈ N+, where [x] denotes one of the closest points of x on X.

In the above statements, 1/2 is chosen based on the constants ,  in Lemma 1 and the length  of X, and is defined as

$$\beta_{t}=2\ln{\frac{2t^{2}\pi^{2}}{3\delta}}+2d\ln\left(t^{2}d b r{\sqrt{\ln{\frac{4d a}{\delta}}}}\right).$$

 (6)
The above two events and the UCB-selection rule for ximply

$$R_{T}=\sum_{t=1}^{T}f(\mathbf{x}^{*})-f([\mathbf{x}^{*}]_{t})+\sum_{t=1}^{T}f([\mathbf{x}^{*}]_{t})-f(\mathbf{x}_{t})\leq\frac{\pi^{2}}{6}+2\beta_{T}^{1/2}\sum_{t=1}^{T}\sigma(\mathbf{x}_{t};\mathbf{X}_{t-1}).\tag{7}$$  In the above expression, the upper bound $\sum_{t=1}^{T}f(\mathbf{x}^{*})-f([\mathbf{x}^{*}]_{t})\leq\sum_{t=1}^{T}1/2^{t}\leq\pi^{2}/6$ follows from the $\mathbf{x}$-independent (linearization over). The lower limit $\mathbf{x}^{T}$ is $f([\mathbf{x}]_{t})-f([\mathbf{x}]_{t})+2\beta_{T}^{1/2}\mathbf{x}^{T}-(\mathbf{x}_{t}-\mathbf{X}_{t-1})$

$$(6)$$
$$\mathbf{\Phi}(T)$$
$$(8)$$

second event (discretization error). The inequality Í
=1

 ( [x
∗]) −  (x) ≤ 2 Í
=1 
(x; X−1)
also follows from the first event (confidence bound) and the definition of x. See the proof of Theorem 2 in [51] for details. The above inequality suggests that the regret upper bound of GP-UCB
depends on the sum of the posterior standard deviations Í
=1 
(x; X−1). Srinivas et al. [51]
provides the upper bound of this term by leveraging the information gain (X) as follows:

$$\sum_{t=1}^{T}\sigma(\mathbf{x}_{t};\mathbf{X}_{t-1})\leq{\sqrt{C T I(\mathbf{X}_{T})}}\leq{\sqrt{C T\gamma_{T}(X)}},$$

where $C\;=\;\frac{}{\ln0}$  . 
ln(1+−2)
. From Eqs. (7) and (8), we conclude that the regret upper bound of GP-
UCB is 
√︁ 
 (X)with probability at least 1 − . By combining the explicit upper bound of  (X) [51, 58], we also obtain 
√︁
 ln+2 and e
+
2+
regret upper bounds for SE and Matern kernels, respectively. ´

## 3 Improved Regret Bound For Gp-Ucb

The following theorem presents our main result: a new regret upper bound for GP-UCB. Theorem 3 (Improved regret upper bound for GP-UCB). *Suppose Assumptions 1 and 2 hold. Set* 
 = SE or  = Matern ´ with  > 2. Furthermore, assume that , , ℓ,  *, and* 2 are fixed constants.

Fix any GP ∈ (0, 1), and set the confidence width parameter  *of GP-UCB as defined in Eq.* (6) with any fixed  ∈ (0, 1 − GP). Then, with probability at least 1 − GP − *, the cumulative regret of* GP-UCB (Algorithm 1) satisfies

$$R_{T}=\begin{cases}\widetilde{O}\left(\sqrt{T}\right)&\text{if}k=k_{\text{Matfern}}\text{with}2\nu+d\leq\nu^{2},\\ O\left(\sqrt{T\ln^{2}T}\right)&\text{if}k=k_{\text{SE}}.\end{cases}$$

$$(9)$$

The hidden constants in the above expressions may depend on ln(1/), , , ℓ, , 2*, and the constants* sup, gap, quad, quad corresponding with GP*, which are guaranteed to exist by Lemma 2.*
We would like to note the following three aspects of our results. First, the constants associated with the sample path properties defined in Lemma 2 are used solely for analyzing the regret. On the other hand, the existing algorithm provided by Scarlett [46], which shows the same e(
√) regret as ours, requires prior information about these constants for the algorithm run. This is often unrealistic in practice. Secondly, our result does not imply the upper bound of Bayesian expected regret E[].

The main issue is that the dependence of the constants in Lemma 2 on GP is not explicitly known.

We leave future work to break this limitation; however, note that the same limitation exists in the algorithm provided by Scarlett [46]. Thirdly, our results in Theorem 3 only focus on the dependence of the total step size  in the regret. Therefore, we cannot claim any improvements of the regret on the dependence of the other parameters. For example, compared to the existing  = (
√︁
 ln+2 )
regret under  = SE, our regret upper bound  = (
√︁
 ln2 ) indeed avoids the dependence of in the logarithmic factor; however, under the joint limit of  and  (,  → ∞), it easily behaves super-linearly even under the slowly increasing  (e.g.,  = Θ(ln ln)) due to the hidden constants in the regret.

## 3.1 Intuitive Explanation Of Our Analysis

Before we describe the proof, we provide an intuitive explanation of why GP-UCB achieves a tighter regret than the existing (√︁  (X)) upper bound. The motivation for our new analysis comes from the observation that the upper bound of the information gain: (X) ≤  (X) in Eq. (8) is not always tight depending on the specific realization of the input sequence X. To see this, let us observe the following two simple extreme cases of X where the inequality (X) ≤  (X) is loose and tight:
- **Case I:** (X) ≤  (X) **is loose**: Let us assume all the input is equal to the unique maximizer x
∗(namely, ∀ ∈ [], x = x
∗). Then, when the kernel function satisfies
∀x ∈ X,  (x, x) = 1 as with SE and Matern ´, we have:

$$I({\bf X}_{T})=\frac{1}{2}\ln\det({\bf I}_{T}+\sigma^{-2}{\bf K}({\bf X}_{T},{\bf X}_{T}))=\frac{1}{2}\sum_{i=1}^{T}\ln(1+\sigma^{-2}\lambda_{i})=\frac{1}{2}\ln(1+\sigma^{-2}T),\tag{10}$$

where is the -th eigenvalue of K(X, X) = 11⊤ with 1 = (1*, . . . ,* 1)
⊤ ∈ R

. The third equation uses the fact that 11⊤ is rank 1, and its unique non-zero eigenvalue is .

- **Case II:** (X) ≤ 
 (X) **is tight**: Let us assume that (x) is the same as the input sequence generated by the maximum variance reduction (MVR) algorithm (namely, ∀ ∈ [], x ∈
argmaxx∈X (x; X−1)) [51, 56]. Then, from the discussion in Sections 2 and 5 in [51], we already know that 
 (X) ≤ (1 − 1/)
−1(X). This suggests that (X) ≤ 
 (X) is tight up to a constant factor when X is realized by MVR.

0 25 50 75 100 125 150 175 200 Number of time steps 0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 0.0 0.2 0.4 0.6 0.8 1.0
−1 0 1 2 In f o r m a ti o n g ai n I(
X

T)
GP-UCB
Case I
Case II (MVR)
f(x
)

C

o u n t o f r e ali z e d xt GP-UCB MVR
0.0 0.2 0.4 0.6 0.8 1.0 x 0 50 100
From Case I, we observe that (X) satisfies Θ(ln) ≤ (X) ≤ 
 (X) depending on X. Furthermore, by comparing the input sequences in cases I and II, we expect that (X) becomes small if X
concentrates around the neighborhood of x
∗, while (X) becomes large if X spreads over the entire input domain X. Then, from the fact that the worst-case regret of GP-UCB increases sub-linearly with the speed of (√︁  (X)), we can deduce that the input sequence X of GP-UCB will eventually concentrate around the maximizer x
∗if x
∗is unique and ∥  ∥∞ is not extremely small2.

We provide an illustrative image in Figure 1. Our proof is designed so as to capture the above intuition that (X) could be improved from 
 (X) to Θ(ln) under "favorable" sample path  .

## 3.2 Proof Of Theorem 3

Let A be an event such that the two high-probability events of the original GP-UCB proof (described in the last paragraph in Section 2) and Lemma 2 with the confidence level GP simultaneously hold. Note that event A occurs with probability at least 1 − GP −  from the union bound. Therefore, it is enough to prove our upper bound under A. To encode the high-level idea in the previous section, we need to capture the concentration behavior of the input sequence X around the maximizer x
∗. From this motivation, given some constant  > 0, we decompose the regret as  = 
(1)

() + 
(2)

(),
where:

$$R_{T}^{(1)}(\varepsilon)=\sum_{t\in\mathcal{T}(\varepsilon)}f(\mathbf{x}^{*})-f(\mathbf{x}_{t}),\ R_{T}^{(2)}(\varepsilon)=\sum_{t\in\mathcal{T}^{\varepsilon}(\varepsilon)}f(\mathbf{x}^{*})-f(\mathbf{x}_{t}).\tag{11}$$

We set T () = { ∈ [] |  (x
∗) −  (x) > } and T
() = [] \ T () in the above definition.

A key observation is that, if we set sufficiently small  depending on the constants in Lemma 2, the inputs (x) in 
(2)

() (namely, inputs (x) such that  (x
∗) −  (x) ≤  holds) are on the locally quadratic region around the maximizer x
∗ due to conditions 1 and 3 in Lemma 2. The formal descriptions are provided in Lemma 20 in Appendix C. This fact is originally leveraged in [46] to analyze the successive elimination-based algorithm. In the analysis of GP-UCB, it enables us to analyze the behavior of the sub-input sequence {x|  (x
∗) −  (x) ≤ } through the regularity constant quad. Below, we formally give the upper bound for 
(2)

().

```
Lemma 4 (General upper bound of 
                          (2)
                         
                          ). Suppose (x) ∈ []is the input query sequence realized
by the GP-UCB algorithm. Furthermore, let 
                                
                                is the upper bound of MIG (X) such that 
                                                                
                                                                / is
non-increasing on [, ∞) with some  ∈ N+
                              3. Then, under event A, we have

```

$$R_{T}^{(2)}(\varepsilon)\leq2c_{\text{gap}}\overline{T}+\frac{\pi^{2}}{3}\left(\log_{2}T+1\right)+\frac{2\sqrt{2C\beta_{T}T}}{\sqrt{2}-1}\max_{i\in\{T\}}\sqrt{\gamma_{(T/2^{i-1})}\left(\mathbb{B}_{2}\left(\sqrt{c_{\text{gap}}^{-1}\eta_{i}};\mathbf{x}^{*}\right)\right)},$$  _where $C=2/\text{ln}(1+\sigma^{-2})$, $\tilde{i}=\left\lfloor\log_{2}\frac{T}{T}\right\rfloor+1$, $\eta_{i}=\frac{2\left(2\sqrt{C\beta_{T}(T/2^{i-1})\overline{\gamma_{T/2^{i-1}}}+\frac{\pi^{2}}{6}}\right)}{(T/2^{i})}$, and $\varepsilon=\min\{c_{\text{gap}},c_{\text{gap}}\rho_{\text{quad}}^{2}\}$._

We give the full proof in Appendix A.1. Here, the dominant term in the above lemma is given as:

$$R_{T}^{(2)}(\varepsilon)=\widetilde{O}\left(\max_{i}\sqrt{T\gamma_{(T/2^{i-1})}\left(\mathcal{B}_{2}\left(\sqrt{c_{\text{quad}}^{-1}\eta_{i}};\mathbf{x}^{*}\right)\right)}\right).\tag{12}$$

Note that is decreasing as the time index /2
−1 of MIG increases. In other words, the input domain B2
√︃
−1 quad; x
∗of MIG shrinks as the time index /2
−1increases. This property is beneficial for obtaining a tighter upper bound than that from the existing technique. For example, under  = Matern ´ with 2 +  ≤ 
2, we can confirm that the dominant polynomial term in MIG
is canceled out by the shrinking of the input domain in MIG. Namely, we can obtain the following result under  = Matern ´:

$$\max_{i}\gamma_{(T/2^{i-1})}\left(\mathcal{B}_{2}\left(\sqrt{c_{\text{quad}}^{-1}\eta_{i};\boldsymbol{x}^{*}}\right)\right)=\widetilde{O}(1)\;\;(\text{as}\;\;T\rightarrow\infty),\tag{13}$$

which leads to 
(2)

() = e(
√). This strictly improves the trivial upper bound 
(2)

() =
e(√︁  (X)) under  = Matern ´. The formal descriptions are given in the next lemma.

Lemma 5 (Upper bound of 
(2)

under SE and Matern ´ ). *Suppose* (x) ∈ []*is the input sequence* realized by the GP-UCB algorithm. Furthermore,  *is set as that in Lemma 4. Then, under event* A,

$$R_{T}^{(2)}(\varepsilon)=\begin{cases}\widetilde{O}(\sqrt{T})&\text{if$k=k_{\text{Matern}}$with$2\nu+d\leq\nu^{2}$,}\\ O\left(\sqrt{T\ln^{2}T}\right)&\text{if$k=k_{\text{SE}}$.}\end{cases}\tag{14}$$

The full proof is given in Appendix A.2. The remaining interest is the upper bound of 
(1)
().

The definition of 
(1)

() is the same as the *lenient regret* [8], which is known to be smaller than the original regret  in GP-UCB. Although Cai et al. [8] studies the frequentist setting, their proof strategy is also applicable to the Bayesian setting as described in Section 3.4 in [8]. The following lemma provides the formal statement about the upper bound of 
(1)

().

Lemma 6 (Upper bound of 
(1)

, adaptation of the proof of Theorem 1 in [8]). *Fix any*  > 0.

Suppose  = SE or  = Matern ´*. Then, when running GP-UCB,* 
(1)

() = e(1) holds under event A.

We provide the proof in Appendix A.3 for completeness. For both kernels, 
(1)

() is dominated by the upper bound of 
(2)

(). Finally, we obtain the desired results by aggregating the inequalities in Lemmas 5 and 6.

## 4 Discussions

Below, we discuss the limitations of our results and outline possible directions for future research.

- **Optimality.** Based on the Ω(
√) lower bound on the expected regret provided by Scarlett
[46], we conjecture that our e(
√) high-probability regret bound for GP-UCB is nearoptimal. However, it is not straightforward to extend the lower bound for the expected regret in [46] to a high probability result. Specifically, the lower bound in [46] is quantified by a mutual information term (Lemma 4 in [46]); however, to our knowledge, the technique used to handle this term appears to be specific to the expected regret setting. We believe that the rigorous optimality argument for the Bayesian high probability regret is an important direction for future research.

- **Smoothness condition.** In our result for the Matern kernel, we require an additional ´
smoothness constraint to obtain a e(
√) regret bound4 To overcome this issue in our proof, we believe that we need stronger regularity conditions on the sample path around the maximizer than those assumed in Lemma 2.

- **Extension to the expected regret.** Our regret bounds involve regularity constants that depend on the sample path. However, to our knowledge, there is no existing research that rigorously analyzes how these constants depend on the confidence level GP. This makes it difficult to obtain the expected regret guarantees as with the original GP-UCB, whose expected regret bounds are established by properly decreasing the confidence level as a function of  (e.g., [40, 53]). To overcome this issue, further analysis for Lemma 2, or another idea to quantify the sample path regularities, is required.

- **Extension to other algorithms.** One limitation of our technique is its restricted applicability to other algorithms. To apply our proof, at least the algorithm should satisfy the following two conditions: (i) on any index subset, the sub-linear cumulative regret is obtained with high probability (Lemma 21), and (ii) the high probability lenient regret bound is provided (Lemma 6). The existing analysis of the other major algorithms in the Bayesian setting (e.g., Thompson sampling [43], information directed sampling [44]) does not provide these properties. Nevertheless, we believe that the high-level ideas in our proof (see Section 3.1) could be beneficial for future refined analyses of other algorithms.

- **Instance dependent analysis in the frequentist setting.** As described in the footnote in Section 3.1, we believe that our analysis does not improve the worst-case regret upper bound in the frequentist setting. On the other hand, our technique can be applied to the instance-dependent analysis [49] for GP-UCB. We expect that our proof strategy could yield a e(
√) instance-dependent regret for GP-UCB by replacing the sample path condition 3 in Lemma 2 with the *growth condition* (Definition 4 in [49]) of the function. It is an interesting direction for future research.

## 5 Conclusion

We provide a refined analysis of GP-UCB in the BO problem. For both SE and Matern kernels, ´ our results improve upon existing regret guarantees and fill the gap between the existing regret of GP-UCB and the current best upper bound in [46]. The core idea of our analysis is to capture the shrinking behavior of the input sequence by relating it to the worst-case upper bound and the sample path regularity conditions. Although our current analysis is limited to GP-UCB in the Bayesian setting, we believe it lays the foundation for several promising future research directions.

## Acknowledgments

We thank Jonathan Scarlett and Shion Takeno for their valuable comments on revising the manuscript.

## References

[2] Douglas Azevedo and Valdir Antonio Menegatto. Sharp estimates for eigenvalues of integral operators generated by dot product kernels on the sphere. *Journal of Approximation Theory*, 2014.

[3] Francis Bach. Breaking the curse of dimensionality with convex neural networks. *Journal of* Machine Learning Research, 2017.

[4] Felix Berkenkamp, Angela P Schoellig, and Andreas Krause. No-regret Bayesian optimization with unknown hyperparameters. *Journal of Machine Learning Research*, 2019.

[5] Alberto Bietti and Francis Bach. Deep equals shallow for relu networks in kernel regimes.

International Conference on Learning Representations, 2021.

[6] Ilija Bogunovic, Jonathan Scarlett, and Volkan Cevher. Time-varying Gaussian process bandit optimization. In Proc. International Conference on Artificial Intelligence and Statistics (AISTATS), 2016.

[7] Adam D Bull. Convergence rates of efficient global optimization algorithms. Journal of Machine Learning Research, 2011.

[8] Xu Cai, Selwyn Gomes, and Jonathan Scarlett. Lenient regret and good-action identification in Gaussian process bandits. In *International Conference on Machine Learning*, pages 1183–1192. PMLR, 2021.

[9] Romain Camilleri, Kevin Jamieson, and Julian Katz-Samuels. High-dimensional experimental design and kernel bandits. In *Proc. International Conference on Machine Learning (ICML)*, 2021.

[10] Alexandre Capone, Armin Lederer, and Sandra Hirche. Gaussian process uniform error bounds with unknown hyperparameters for safety-critical applications. In International Conference on Machine Learning, 2022.

[11] Sayak Ray Chowdhury and Aditya Gopalan. On kernelized multi-armed bandits. In Proc.

International Conference on Machine Learning (ICML), 2017.

[12] Andreas Christmann and Ingo Steinwart. Support vector machines. 2008. [13] Nando de Freitas, Alex Smola, and Masrour Zoghi. Regret bounds for deterministic Gaussian process bandits. *arXiv preprint arXiv:1203.2177*, 2012.

[14] Nando De Freitas, Alex J. Smola, and Masrour Zoghi. Exponential regret bounds for Gaussian process bandits with deterministic observations. In Proceedings of the 29th International Conference on International Conference on Machine Learning, page 955–962. Omnipress, 2012.

[15] Thomas Desautels, Andreas Krause, and Joel W. Burdick. Parallelizing exploration-exploitation tradeoffs in Gaussian process bandit optimization. *Journal of Machine Learning Research*, 2014.

[16] Costas Efthimiou and Christopher Frye. *Spherical harmonics in p dimensions*. World Scientific, 2014.

[17] Peter Frazier, Warren Powell, and Savas Dayanik. The knowledge-gradient policy for correlated normal beliefs. *INFORMS journal on Computing*, 21(4):599–613, 2009.

[18] Peter I Frazier. A tutorial on Bayesian optimization. *arXiv preprint arXiv:1807.02811*, 2018. [19] Roman Garnett. *Bayesian optimization*. Cambridge University Press, 2023. [20] Amnon Geifman, Abhay Yadav, Yoni Kasten, Meirav Galun, David Jacobs, and Basri Ronen. On the similarity between the Laplace and neural tangent kernels. *Advances in Neural Information* Processing Systems, 2020.

[21] Subhashis Ghosal and Anindya Roy. Posterior consistency of Gaussian process prior for nonparametric binary regression. 2006.

[22] Andrew Gray and George Ballard Mathews. A treatise on Bessel functions and their applications to physics. Macmillan, 1895.

[23] Steffen Gr¨unewalder, Jean-Yves Audibert, Manfred Opper, and John Shawe-Taylor. Regret ¨
bounds for Gaussian process bandit problems. In *Proc. International Conference on Artificial* Intelligence and Statistics (AISTATS). JMLR Workshop and Conference Proceedings, 2010.

[24] Philipp Hennig and Christian J Schuler. Entropy search for information-efficient global optimization. *The Journal of Machine Learning Research*, 13(1):1809–1837, 2012.

[25] Shogo Iwazaki and Shinya Suzumura. No-regret bandit exploration based on soft tree ensemble model. *Advances in Neural Information Processing Systems*, 2024.

[26] Shogo Iwazaki and Shion Takeno. Improved regret analysis in Gaussian process bandits: Optimality for noiseless reward, RKHS norm, and non-stationary variance. In Proc. International Conference on Machine Learning (ICML), 2025.

[27] David Janz. *Sequential decision making with feature-linear models*. PhD thesis, 2022. [28] David Janz, David Burt, and Javier Gonzalez. Bandit optimisation of functions in the Matern ´
kernel RKHS. In Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of *Proceedings of Machine Learning Research*, pages 2486–2495. PMLR, 2020.

[29] Kirthevasan Kandasamy, Jeff Schneider, and Barnabas Poczos. High dimensional Bayesian optimisation and bandits via additive models. In Proc. International Conference on Machine Learning (ICML), 2015.

[30] Kirthevasan Kandasamy, Gautam Dasarathy, Junier Oliva, Jeff Schneider, and Barnabas Poczos.

Multi-fidelity Gaussian process bandit optimisation. *Journal of Artificial Intelligence Research*, 2019.

[31] Parnian Kassraie and Andreas Krause. Neural contextual bandits without regret. In International Conference on Artificial Intelligence and Statistics, pages 240–278. PMLR, 2022.

[32] Kenji Kawaguchi, Leslie P Kaelbling, and Tomas Lozano-P ´ erez. Bayesian optimization with ´
exponential convergence. *Advances in neural information processing systems*, 28, 2015.

[33] Jeankyung Kim and David Pollard. Cube root asymptotics. *The Annals of Statistics*, pages 191–219, 1990.

[34] Andreas Krause and Cheng Ong. Contextual Gaussian process bandit optimization. In *Proc.*
Neural Information Processing Systems (NeurIPS), 2011.

[35] Zihan Li and Jonathan Scarlett. Gaussian process bandit optimization with few batches. In Proc. International Conference on Artificial Intelligence and Statistics (AISTATS), 2022.

[36] Ha Quang Minh, Partha Niyogi, and Yuan Yao. Mercer's theorem, feature maps, and smoothing.

In *International Conference on Computational Learning Theory*. Springer, 2006.

[37] Jonas Mockus. On Bayesian methods for seeking the extremum. In ˇ Optimization Techniques IFIP Technical Conference Novosibirsk, July 1–7, 1974 6, pages 400–404. Springer, 1975.

[38] Francis J Narcowich and Joseph D Ward. Scattered data interpolation on spheres: error estimates and locally supported basis functions. *SIAM Journal on Mathematical Analysis*, 33 (6):1393–1410, 2002.

[39] Francis J Narcowich, Xinping Sun, and Joseph D Ward. Approximation power of RBFs and their associated SBFs: a connection. *Advances in Computational Mathematics*, 2007.

[40] Biswajit Paria, Kirthevasan Kandasamy, and Barnabas P ´ oczos. A flexible framework for ´
multi-objective Bayesian optimization using random scalarizations. In *Uncertainty in Artificial* Intelligence, 2020.

[41] Carl Edward Rasmussen and Christopher K. I. Williams. *Gaussian Processes for Machine* Learning (Adaptive Computation and Machine Learning). The MIT Press, 2005.

[42] Gabriel Riutort-Mayol, Paul-Christian B¨urkner, Michael R Andersen, Arno Solin, and Aki Vehtari. Practical hilbert space approximate Bayesian Gaussian processes for probabilistic programming. *Statistics and Computing*, 33(1):17, 2023.

[43] Daniel Russo and Benjamin Van Roy. Learning to optimize via posterior sampling. Mathematics of Operations Research, 39(4):1221–1243, 2014.

[44] Daniel Russo and Benjamin Van Roy. Learning to optimize via information-directed sampling.

Advances in neural information processing systems, 27, 2014.

[45] Sudeep Salgia, Sattar Vakili, and Qing Zhao. Random exploration in Bayesian optimization: Order-optimal regret and computational efficiency. In Proc. International Conference on Machine Learning (ICML), 2024.

[46] Jonathan Scarlett. Tight regret bounds for Bayesian optimization in one dimension. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 4500–4508. PMLR, 2018.

[47] Jonathan Scarlett, Ilija Bogunovic, and Volkan Cevher. Lower bounds on regret for noisy Gaussian process bandit optimization. In *Proc. Conference on Learning Theory (COLT)*, 2017.

[48] Meyer Scetbon and Zaid Harchaoui. A spectral analysis of dot-product kernels. In *International* conference on artificial intelligence and statistics, 2021.

[49] Shubhanshu Shekhar and Tara Javidi. Instance dependent regret analysis of kernelized bandits.

In *International Conference on Machine Learning*, 2022.

[50] Arno Solin and Simo Sarkk ¨ a. Hilbert space methods for reduced-rank gaussian process regres- ¨
sion. *Statistics and Computing*, 2020.

[51] Niranjan Srinivas, Andreas Krause, Sham Kakade, and Matthias Seeger. Gaussian process optimization in the bandit setting: No regret and experimental design. In *Proc. International* Conference on Machine Learning (ICML), 2010.

[52] Michael L Stein. *Interpolation of spatial data: some theory for kriging*. Springer Science &
Business Media, 1999.

[53] Shion Takeno, Yu Inatsu, and Masayuki Karasuyama. Randomized Gaussian process upper confidence bound with tighter Bayesian regret bounds. In *Proceedings of the 40th International* Conference on Machine Learning, volume 202 of *Proceedings of Machine Learning Research*, pages 33490–33515. PMLR, 2023.

[54] Filip Tronarp, Toni Karvonen, and Simo Sarkk ¨ a. Mixture representation of the mat ¨ ern class ´
with applications in state space approximations and Bayesian quadrature. In 2018 IEEE 28th International Workshop on Machine Learning for Signal Processing (MLSP), 2018.

[55] Sattar Vakili and Julia Olkhovskaya. Kernelized reinforcement learning with order optimal regret bounds. *Advances in Neural Information Processing Systems*, 2023.

[56] Sattar Vakili, Nacime Bouziani, Sepehr Jalali, Alberto Bernacchia, and Da shan Shiu. Optimal order simple regret for Gaussian process bandits. In Proc. Neural Information Processing Systems (NeurIPS), 2021.

[57] Sattar Vakili, Michael Bromberg, Jezabel Garcia, Da-shan Shiu, and Alberto Bernacchia. Uniform generalization bounds for overparameterized neural networks. arXiv preprint arXiv:2109.06099, 2021.

[58] Sattar Vakili, Kia Khezeli, and Victor Picheny. On information gain and regret bounds in Gaussian process bandits. In *Proc. International Conference on Artificial Intelligence and* Statistics (AISTATS), 2021.

[59] Michal Valko, Nathan Korda, Remi Munos, Ilias Flaounas, and Nello Cristianini. Finite-time ´
analysis of kernelised contextual bandits. In Proceedings of the Twenty-Ninth Conference on Uncertainty in Artificial Intelligence, UAI'13, page 654–663. AUAI Press, 2013.

[60] Zexin Wang, Vincent YF Tan, and Jonathan Scarlett. Tight regret bounds for noisy optimization of a Brownian motion. *IEEE Transactions on Signal Processing*, 70:1072–1087, 2022.

[61] Justin Whitehouse, Zhiwei Steven Wu, and Aaditya Ramdas. Improved self-normalized concentration in Hilbert spaces: Sublinear regret for GP-UCB. *Proc. Neural Information Processing* Systems (NeurIPS), 2023.

[62] Yun Yang, Anirban Bhattacharya, and Debdeep Pati. Frequentist coverage and sup-norm convergence rate in gaussian process regression. *arXiv preprint arXiv:1708.04753*, 2017.

[63] Fuzhen Zhang. *Matrix theory: basic results and techniques*. Springer Science & Business Media, 2011.

## A Proofs In Section 3 A.1 Proof Of Lemma 4

Proof. From Lemma 21, we have the following upper bound for any index set T ⊂ [] under A:

$$\sum_{t\in\mathcal{T}}f(x^{*})-f(x_{t})\leq2{\sqrt{C\beta_{T}|\mathcal{T}|\overline{{\gamma}}_{|\mathcal{T}|}}}+{\frac{\pi^{2}}{6}}.$$

Here, for any  such that /2
−1 ≥ , we set () as

$$\eta_{i}=\frac{2\left(2\sqrt{C\beta_{T}(T/2^{i-1})\overline{{{Y}}}_{T/2^{i-1}}+\frac{\pi^{2}}{6}}\right)}{(T/2^{i-1})}.$$

$$(15)$$
$$(16)$$
$$(17)$$
$$(18)$$
$$(19)$$

As described in the proof below, these () are designed so that we can obtain the upper bound of |T ()| in a dyadic manner. Here, we consider the upper bound of |T ()| based on the worst-case upper bound in Eq. (15). From the definition of T () and Eq. (15) with T = [], the condition |T (1)|1 ≤ 2√︁  + 
2/6 must be satisfied; otherwise, we have Í ∈ []
 (x
∗) −  (x) ≥
Í ∈ T (1 )
 (x
∗) −  (x) ≥ |T (1)|1 > 2√︁  + 
2/6, which contradicts worst-case upper bound in Eq. (15). Therefore, we can obtain the following upper bound:

$$|{\mathcal{T}}(\eta_{1})|\leq\operatorname*{max}\left\{t\leq T\mid t\eta_{1}\leq2{\sqrt{C\beta_{T}T{\overline{{\gamma}}}_{T}}}+{\frac{\pi^{2}}{6}}\right\}={\frac{T}{2}}.$$

Furthermore, since is monotonic due to the condition about 
, we have 1 ≤ 2, which implies T (2) ⊂ T (1). From Eq. (15) with T = T (1), Eq. (17), and T (2) ⊂ T (1), we further obtain

$$|{\mathcal{T}}(\eta_{2})|\leq\operatorname*{max}\left\{t\leq T/2\mid t\eta_{2}\leq2{\sqrt{C\beta_{T}(T/2){\overline{{\gamma}}}_{(T/2)}}}+{\frac{\pi^{2}}{6}}\right\}={\frac{T}{4}}.$$

. (18)
Similarly to |T (2)|, we have T (3) ⊂ T (2) and

$$|{\mathcal{T}}(\eta_{3})|\leq\operatorname*{max}\left\{t\leq T/4\mid t\eta_{3}\leq2{\sqrt{C\beta_{T}(T/4){\overline{{\gamma}}}_{(T/4)}}}+{\frac{\pi^{2}}{6}}\right\}={\frac{T}{8}}.$$
. (19)

```
By repeating this argument  times while /2
                                           −1 ≥  holds, we have the following inequality for any
 ≤ ⌊log2
         
          ⌋ + 1:

```

$$|{\mathcal{T}}(\eta_{i})|\leq\operatorname*{max}\left\{t\leq T/2^{i-1}\mid t\eta_{i}\leq{\sqrt{C\beta_{T}(T/2^{i-1}){\overline{{\nabla}}}_{(T/2^{i-1})}}}+{\frac{\pi^{2}}{6}}\right\}={\frac{T}{2^{i}}}.$$

. (20)
Then, we have

 (2)  () =∑︁  ∈ T ( )  (x ∗) −  (x) (21) =∑︁  ∈ T ( )∩T (1 )  (x ∗) −  (x) + ∑︁  ∈ T ( )∩T (1 )  (x ∗) −  (x) (22) =∑︁  ∈ T ( )∩T (1 )∩T (2 )  (x ∗) −  (x) + ∑︁  ∈ T ( )∩T (1 )∩T (2 )  (x ∗) −  (x) +∑︁  ∈ T ( )∩T (1 )  (x ∗) −  (x) (23) =∑︁  ∈ T ( )∩T (2 )  (x ∗) −  (x) + ∑︁ 2 =1 ∑︁  ∈ T ( )∩T (−1 )∩T ( )  (x ∗) −  (x), (24) where the last line follows from T (2) ⊂ T (1). In the above inequality, we define T (0) as

$$(20)$$
T (0) = [] for notational convenience. By repeatedly applying the above decomposition, we
obtain

$$\sum_{t\in\mathcal{T}^{c}(\varepsilon)\cap\mathcal{T}(\eta_{2})}f(\mathbf{x}^{*})-f(\mathbf{x}_{t})+\sum_{i=1}^{2}\sum_{t\in\mathcal{T}^{c}(\varepsilon)\cap\mathcal{T}(\eta_{i-1})\cap\mathcal{T}^{c}(\eta_{i})}f(\mathbf{x}^{*})-f(\mathbf{x}_{t})$$ $$=\sum_{t\in\mathcal{T}^{c}(\varepsilon)\cap\mathcal{T}(\eta_{2})}f(\mathbf{x}^{*})-f(\mathbf{x}_{t})+\sum_{i=1}^{3}\sum_{t\in\mathcal{T}^{c}(\varepsilon)\cap\mathcal{T}(\eta_{i-1})\cap\mathcal{T}^{c}(\eta_{i})}f(\mathbf{x}^{*})-f(\mathbf{x}_{t})$$ $$\vdots$$
$$(25)$$  $$(26)$$
$$=\sum_{t\in\mathcal{T}^{c}(\kappa)\cap\mathcal{T}(\eta)}f(\mathbf{x}^{*})-f(\mathbf{x}_{t})+\sum_{i=1}^{7}\sum_{t\in\mathcal{T}^{c}(\kappa)\cap\mathcal{T}(\eta_{i-1})\cap\mathcal{T}^{c}(\eta_{i})}f(\mathbf{x}^{*})-f(\mathbf{x}_{t}),$$  where $\overline{i}=\left\lfloor\log_{2}\frac{7}{L}\right\rfloor+1$. Regarding the first term in Eq. (27), we have 

$$(27)$$
$$(28)$$

$$\sum_{t\in\mathcal{T}^{\prime}(\kappa)\setminus\mathcal{T}(\eta)}f(\mathbf{x}^{*})-f(\mathbf{x}_{t})\leq2c_{\sup}|\mathcal{T}(\eta_{t})|\leq2c_{\sup}\overline{T},\tag{28}$$  where the last inequality follows from $|\mathcal{T}(\eta_{t})|\leq\overline{T}$, which is implied by $|\mathcal{T}(\eta_{t})|\leq T/2^{\overline{t}}$ from 

Eq. (20) and the definition of . Next, regarding the second term in Eq. (27), we first define T and X()as T = T
() ∩ T (−1) ∩ T() and X() = {x|  ∈ T }, respectively. Then, by applying Lemma 21 with T = T

∑︁ , we have  ∈ T ( )∩T (−1 )∩T ( )  (x ∗) −  (x) = ∑︁  ∈ T  (x ∗) −  (x) (29) ≤ 2 √︃  |T |(X()) +  2 6(30) ≤ 2 √︃  |T|| T |(X()) +  2 6(31) ≤ 2 √︃  |T (−1)|| T (−1 ) |(X()) +  2 ≤ 2 √︃  (/2 −1)(/2 −1) (X()) +  2 6 where the third inequality follows from |T| ≤ |T (−1)|, and the last inequality follows from Eq. (20).

$$(29)$$

6(32)
, (33)
By aggregating Eqs. (27), (28), and (33), we obtain the following inequality under A:

 (2) () ≤ 2sup + 2 ∑︁  =1 √︃  (/2 −1)(/2 −1)(X()) +  2 6 (34) ≤ 2sup +  2 3 log2  + 1+ 2√︁  ∑︁  1 2 (−1)/2 √︃(/2 −1) (X()) (35) =1 ≤ 2sup +  2 3 log2  + 1+ 2√︁2  √2 − 1max ∈ [] √︃(/2 −1) (X()). (36) The last line follows from Í=11 2 (−1)/2 ≤Í∞ =11 2 (−1)/2 =1 1−1/ √2 = √2 √2−1 . The last part of the proof is

$$(34)$$
(35)  $\binom{36}{2}$  . 

to specify the radius of the ball B2 (·; x
∗) such that X()is included in it.
Conversion of the sub-optimality gap into the upper bound input radius. From condition 3 in Lemma 2, the definition of T

(), , and Lemma 20, we have x ∈ B2 (quad; x
∗) for any x ∈ X().

This implies ∀x ∈ X(),  (x
∗) −  (x) ≥ quad ∥x − x
∗∥
2 2 from condition 3 in Lemma 2. Since ∀x ∈
X(),  (x
∗) −  (x) ≤  from T ⊂ T(), we have  ≥ quad ∥x−x
∗∥
2 2 ⇔
√︃

−1 quad ≥ ∥x−x
∗∥2, which implies X() ⊂ B2 (
√︃

−1 quad; x
∗). Therefore, we have

$$\gamma_{(T/2^{i-1})}(X^{(i)})\leq\gamma_{(T/2^{i-1})}\left(\mathcal{B}_{2}\left(\sqrt{c_{\text{quad}}^{-1}\eta_{i}};\mathbf{x}^{*}\right)\right).\tag{37}$$

14 Finally, combining Eq. (35) with Eq. (37), we have

$$R_{T}^{(2)}(\varepsilon)\leq2c_{\text{sup}}\overline{T}+\frac{\pi^{2}}{3}\left(\log_{2}T+1\right)+\frac{2\sqrt{2C\beta\pi T}}{\sqrt{2}-1}\max_{i\in\{\overline{i}\}}\sqrt{\gamma_{(T/2^{i-1})}\left(\beta_{2}\left(\sqrt{c_{\text{quad}}^{-1}\eta_{i}\colon\pi^{\star}}\right)\right)}.\tag{38}$$

(39)  (40)  $\binom{40}{40}$  . 
(41)  $\binom{42}{42}$  . 

## A.2 Proof Of Lemma 5

To prove Lemma 5, we require the upper bound of MIG with the explicit dependence on the radius of the input domain. In Corollary 8 in Appendix B, we provide it with a full proof. Below, we establish the proof of Lemma 5 based on Corollary 8.

When  = Matern ´. Set Mat > 0 as the constant such that the following inequalities hold:

$$\begin{array}{l}{{\forall t\geq2,\gamma_{t}(X)\leq C_{\mathrm{Mat}^{d}}\frac{d}{\gamma_{t}d}\,\ln\frac{4\pi d}{\gamma_{t}d}\,t,}}\\ {{\forall t\geq2,\forall\eta>0,\gamma_{t}\left(\{x\in\mathbb{R}^{d}\mid\|x\|_{2}\leq\eta\}\right)\leq C_{\mathrm{Mat}}\left(\eta^{\frac{2\pi d}{\gamma_{t}d}\,\frac{d}{t}\,\log\frac{4\pi d}{\gamma_{t}d}}\,t+\ln^{2}t\right).}}\end{array}$$

 (40)
The existence of Mat is guaranteed by the upper bound of MIG established in Corollary 85. Note that Mat is the constant that may depend on *, ℓ, ,* , and 2. Furthermore, we set 
 
= Mat 2+ ln 4+
2+ .

For function () := /, we have

$$g^{\prime}\left(t\right)=-\frac{2\nu C_{\text{Max}}}{2\nu+d}t^{-\frac{2\nu}{2\nu+d}-1}\ln\frac{\frac{4\nu+d}{2\nu+d}}{2\nu+d}t+C_{\text{Max}}\frac{4\nu+d}{2\nu+d}t^{-\frac{2\nu}{2\nu+d}-1}\ln\frac{\frac{2\nu}{2\nu+d}}{2\nu+d}t$$ $$=\frac{C_{\text{Max}}}{2\nu+d}t^{-\frac{2\nu}{2\nu+d}-1}(\ln\frac{\frac{2\nu}{2\nu+d}}{2\nu+d}\ t)\left(-2\nu\ln t+4\nu+d\right).$$

From the above expression, if 2 ln  ≥ 4 +  ⇔  ≥ exp(2 + /(2)), / is non-increasing.

Therefore, we set  = ⌈exp(2 + /(2))⌉, which is independent of . Here, for any  > 0 and  ≥ 2, we have

$\gamma_{t}\left(\mathcal{B}_{2}\left(\eta;\boldsymbol{x}^{*}\right)\right)\leq\gamma_{t}\left(\left\{\boldsymbol{x}\in\mathbb{R}^{d}\ |\ \|\boldsymbol{x}-\boldsymbol{x}^{*}\|_{2}\leq\eta\right\}\right)$  $$=\gamma_{t}\left(\left\{\boldsymbol{x}\in\mathbb{R}^{d}\ |\ \|\boldsymbol{x}\|_{2}\leq\eta\right\}\right)$$ $$\leq C_{\text{Mat}}\left(\eta^{\frac{2\omega_{d}}{2\omega_{d}}}t^{\frac{d}{2\omega_{d}}}\ln^{\frac{4\omega_{d}}{2\omega_{d}}}t+\ln^{2}t\right),$$

$$(43)$$
$$(44)$$

$$(45)$$

where the second line follows from the fact that Matern ´is the stationary kernel (namely, Matern ´ is transition invariant against any shift of inputs). Regarding in Lemma 4, by setting  as

 = /2
−1, we have

  = 2 2√︁  +  2 6   = 4√︁  +  2 3(47) = 4 √︃ Mat 2+  ln 4+ 2+   +  2 3(48) = 4√︁Mat  − 2+  ln 4+ 4+2  +  2 3 ≤ eMat√︁  − 2+ ln 4+ 4+2  ,  (50)

$$(47)$$
$$(48)$$

$$(46)$$

$$(49)$$

$$(50)$$

```
where eMat > 0 is a sufficiently large constant such that eMat
                                                           √
                                                               
                                                                  −
                                                                   2+
                                                                 
                                                                  ln
                                                                         4+
                                                                        4+2 
                                                                              
                                                                              ≥

4√︁Mat
          
            −
              
             2+
            
            ln
                   4+
                  4+2 
                        
                        +
                            
                             2
                            3
                               for any  ≥ 2. Note that we can choose eMat > 0 with-
out depending on . From Eqs. (45) and (50), for any , we have

```

/2 −1 B2 √︃ −1 quad; x ∗  (51) ≤ Mat   −  2+ quad   2+   2+  ln 4+ 2+   + ln2  (52) ≤ Mat   −  2+ quad e  2+ Mat    2(2+)   − 2+  ln 4+ 4+2    2+ 2+  ln 4+ 2+   + ln2 

. (53)
Furthermore, by noting condition 2 +  ≤ 
2, we have

$$\left(T_{i}^{-\frac{\nu}{2\nu+d}}\,\ln\frac{4\nu+d}{4\nu+d}\,T_{i}\right)^{\frac{\nu d}{2\nu+d}}\,T_{i}^{\frac{d}{2\nu+d}}\,T_{i}=\widetilde{O}\left(T_{i}^{-\frac{\nu^{2}d}{(2\nu+d)^{2}}+\frac{d}{2\nu+d}}\right)$$ $$=\widetilde{O}\left(T_{i}^{\frac{d(2\nu+d)-\nu^{2}d}{(2\nu+d)^{2}}}\right)$$ $$=\widetilde{O}\left(T_{i}^{\frac{d(2\nu+d-\nu^{2})}{(2\nu+d)^{2}}}\right)$$ $$=\widetilde{O}(1).$$

$$(S4)$$
$$(S1)$$
$$(52)$$
$$(\mathbf{53})$$

$$(\mathbf{55})$$
$$({\mathsf{S}}7)$$

$$(56)$$

From the above inequalities, we have /2
−1 B2
√︃ 
−1 quad; x
∗  = e(1). Therefore, Lemma 4 implies

$$R_{T}^{(2)}(\varepsilon)\leq2c_{\sup}\overline{T}+\frac{\pi^{2}}{3}\left(\log_{2}T+1\right)+\frac{2\sqrt{2C\beta_{T}T}}{\sqrt{2}-1}\times\widetilde{O}(1)$$ $$=\widetilde{O}(\sqrt{T}).$$

When  = SE. The proof for  = SE is not as straightforward as the proof for  = Matern ´. Specifically, we have to choose a proper  so as to obtain an (ln) upper bound of MIG. Let SE > 0 be the constant such that the following inequalities hold:

$\forall t\geq2,\gamma_{t}(X)\leq C_{\text{SE}}\ln^{d+1}t,$  $\forall t\geq2,\forall\eta\in(0,\sqrt{\frac{2t^{2}}{e^{2}c_{d}}}),\gamma_{t}(\{x\in\mathbb{R}^{d}\mid\|x\|_{2}\leq\eta\})\leq C_{\text{SE}}\left(\frac{\ln^{d+1}t}{\ln^{d}\left(\frac{2t^{2}}{\eta^{2}ec_{d}}\right)}+\ln T\right).$  It is easy to see that $C_{\text{SE}}$ is not stable. On all $\eta\in\mathbb{R}$, the above inequality is the set 

$$(58)$$
$$(59)$$
$$(61)$$
$$(63)$$
(64)  $\binom{65}{2}$  . 

 (61)
The existence of such SE is guaranteed by Corollary 8. In the above inequalities,  is the constant defined in Corollary 8. We also set as 
 = SE ln+1. We choose  later such that we can leverage the second statement in the above inequalities. Under  = SE, we have

  = 2 2√︁  +  2 6   = 4√︁   +  2 3(63) = 4 √︃ SE ln+1   +  2 3(64) = 4√︁SE  − 1 2  ln +1 2  +  2 3(65) ≤ eSE√︁  − 1 2  ln +1 2  ,  (66)

$$(62)$$

$$(66)$$

```
where eSE > 0 is a sufficiently large constant such that eSE
                                                              √
                                                                  
                                                                     −
                                                                      1
                                                                      2
                                                                    
                                                                     ln
                                                                         +1
                                                                          2 
                                                                             
                                                                              ≥ 4√︁SE
         
           −
             1
             2
           
           ln
                +1
                2 
                    
                    +
                        
                        2
                       3
                          for any  ≥ 2. Hereafter, we define 
                                                         
                                                         := eSE
                                                               √
                                                                  
                                                                     −
                                                                      1
                                                                      2
                                                                     
                                                                     ln
                                                                         +1
                                                                          2 
                                                                             
                                                                             .

```

Then, to apply Eq. (61), we consider the lower bound of  such that √︃
−1 quad <√︁2ℓ 2/(
2) hold.

From the condition √︃
−1 quad <√︁2ℓ 2/(
2), we have

$$\sqrt{c_{\text{quad}}^{-1}\overline{\overline{\eta}}_{i}}<\sqrt{\frac{2\ell^{2}}{e^{2}c_{d}}}\Leftrightarrow c_{\text{quad}}^{-1}\frac{e^{2}c_{d}}{2\ell^{2}}\widetilde{C}_{\text{SE}}\sqrt{\beta_{T}}\ln^{\frac{d+1}{2}}T_{i}<T_{i}^{\frac{1}{2}}$$ $$\Leftrightarrow c_{\text{quad}}^{-1}\frac{e^{2}c_{d}}{2\ell^{2}}\widetilde{C}_{\text{SE}}\sqrt{\beta_{T}}\ln^{\frac{d+1}{2}}T<T_{i}^{\frac{1}{2}}$$ $$\Leftrightarrow\left(\frac{e^{2}c_{d}\widetilde{C}_{\text{SE}}}{2\ell^{2}c_{\text{quad}}}\right)^{2}\beta_{T}\ln^{d+1}T<T_{i}.$$

 

(67)  $$\begin{array}{l}\left(68\right)\end{array}$$ = $$\begin{array}{l}\left(69\right)\end{array}$$ . 

From the above inequality, we set  such that

$$\left(\frac{e^{2}c_{d}\widetilde{C}_{\rm SE}}{2\ell^{2}c_{\rm quad}}\right)^{2}\beta_{T}\ln^{d+1}T<\overline{T}.$$  and (70)
Then, from  ≥  and Eqs. (67), and (70),

$$\gamma_{T/2^{i-1}}\left(\mathcal{B}_{2}\left(\sqrt{c_{\mathrm{quad}}^{-1}\eta_{i}};\boldsymbol{x}^{*}\right)\right)\leq\gamma_{T_{i}}\left(\left\{\boldsymbol{x}\in\mathbb{R}^{d}\mid\|\boldsymbol{x}\|_{2}\leq\sqrt{c_{\mathrm{quad}}^{-1}\eta_{i}}\right\}\right)$$ $$\leq C_{\mathrm{SE}}\left(\frac{\ln^{d+1}T_{i}}{\ln^{d}\left(\frac{2c_{\mathrm{quad}}\ell^{2}}{\overline{\eta}_{i}e c_{d}}\right)}+\ln T\right).$$

$$(70)^{\frac{1}{2}}$$
$$\quad(71)$$  $$\quad(72)$$
$$(73)^{\frac{1}{2}}$$
 (74)  $$\begin{array}{l}~~~~~~~~~~~~~~\end{array}$$ (75)  $$\begin{array}{l}~~~~~~~~~~~~~~\end{array}$$ (76)  $$\begin{array}{l}~~~~~~~~~~~~~~\end{array}$$ (77)  $$\begin{array}{l}~~~~~~~~~~~~~~\end{array}$$ (77)  ... 

Based on Eq. (72), we further consider the lower bound of  such that

$$\frac{\ln^{d+1}T_{i}}{d^{d}\left(\frac{2c_{\rm qma}\ell^{2}}{\overline{\eta}_{i}ec_{d}}\right)}=O(\ln T).\tag{1}$$

For the condition in Eq. (73), we have

$$\frac{2c_{\mathrm{quad}}\ell^{2}}{\overline{{{\eta}}}_{i}e c_{d}}\geq T_{i}^{1/4}$$

 ⇔
2quadℓ 2 eSE √ −1/2  ln +1 2  ≥  1/4  (74) ⇔  1/4  ≥ eSE √ ln +1 2  2quadℓ 2(75) ⇐  1/4  ≥ eSE √  ln +1 2  2quadℓ 2(76) ⇔   ≥  eSE √  ln +1 2  2quadℓ 2 !4.  (77)

Hence, if $\overline{T}\geq\left(\frac{ec_{d}\,\bar{C}_{\mathrm{SE}}\sqrt{\beta_{T}}\ln\frac{d+1}{2}\,T}{2c_{\mathrm{quad}}\ell^{2}}\right)^{4}$, we have $\ell$. 

$$C_{\mathrm{SE}}\left(\frac{\ln^{d+1}T_{i}}{\ln^{d}\left(\frac{2c_{\mathrm{out}}\ell^{2}}{\eta_{i}e_{d}}\right)}+\ln T\right)\leq C_{\mathrm{SE}}\left(\frac{\ln^{d+1}T_{i}}{4^{-d}\ln^{d}T_{i}}+\ln T\right)$$ $$\leq C_{\mathrm{SE}}\left(4^{d}\ln T+\ln T\right).$$
(78)  $\binom{79}{5}$  (79)  . 
17 By aggregating the conditions (70) and (77), we set  as the smallest natural number such that the following inequalities hold:

$$\overline{{{T}}}\geq\left(\frac{e^{2}c_{d}\widetilde{C}_{\mathrm{SE}}}{2\ell^{2}c_{\mathrm{quad}}}\right)^{2}\beta_{T}\ln^{d+1}T,\;\;\mathrm{and}\;\;\overline{{{T}}}\geq\left(\frac{e c_{d}}{2c_{\mathrm{quad}}\ell^{2}}\right)^{4}\widetilde{C}_{\mathrm{SE}}^{4}\beta_{T}^{2}\ln^{2(d+1)}T.$$

Then, from Eqs. (72) and (79), we have

$$(80)$$
$$\sqrt{\gamma_{(T/2^{i-1})}\left(\mathcal{B}_{2}\left(\sqrt{c_{\mathrm{quad}}^{-1}\eta_{i};x^{*}}\right)\right)}=O(\sqrt{\ln T}).$$

Finally, by noting  = (ln2+4 ), we obtain the following result from Lemma 4:

$$(81)$$
$$R_{T}^{(2)}(\varepsilon)=O\left(\ln^{2d+4}T+\sqrt{T\ln^{2}T}\right).$$
$$(82)$$
$$\square$$

Since  is a fixed constant, the above equation implies 
(2)

() = (
√︁ ln2 ). □

## A.3 Proof Of Lemma 6

Proof. From the upper bound of the discretization error in event A, we have ∀ ≥√︁2/, ∀x ∈
X, |  (x) −  ( [x])| ≤ /2. Here, we set T () = { ∈ N+ |  ≥√︁2/}. By relying on the standard argument of MIG [51], we observe the following inequality for any realizations and  > 0:

$$\operatorname*{min}_{t\in{\mathcal{T}}(\varepsilon)\cap{\underline{{T}}}(\varepsilon)}\sigma(x_{t};\mathbf{X}_{t-1})\leq{\sqrt{\frac{C\gamma|{\mathcal{T}}(\varepsilon)\cap{\underline{{T}}}(\varepsilon)|(X)}{|{\mathcal{T}}(\varepsilon)\cap{\underline{{T}}}(\varepsilon)|}}},$$
$$(83)$$
$$(84)$$
$$(85)$$
$$(86)$$
$$(87)$$

where T () = { ∈ [] |  (x
∗) −  (x) > } and  = 2/ln(1 + 
−2). Under A, we further have the following inequalities for anye ∈ argmin ∈ T (  )∩T( ) (x; X−1)
6:
where the second inequality follows from the definition of T (), and the last inequality follows from e ∈ T () and event A. Therefore, under A, the inequality −

2
+ 2
√︂
e |T (  )∩T() | (X )
| T (  )∩T( ) | ≥ 0 must hold; otherwise, (xe; Xe−1; ye−1
) + 
1/2 e
(xe; Xe−1
) < ( [x
∗]
e; Xe−1; ye−1
) + 
1/2 e
( [x
∗]
e; Xe−1
),
which contradicts xe ∈ argmaxx∈X (x; Xe−1; ye−1
) + 
1/2 e
(x; Xe−1
). This further implies

$$|{\mathcal{T}}(\epsilon)\cap{\mathcal{T}}(\epsilon)|\leq{\frac{16C\beta_{T}\gamma_{|{\mathcal{T}}(\epsilon)\cap{\mathcal{T}}(\epsilon)|}(X)}{\varepsilon^{2}}}\leq{\frac{16C\beta_{T}\gamma_{|{\mathcal{T}}(\epsilon)\cap{\mathcal{T}}(\epsilon)|}(X)}{\varepsilon^{2}}}$$

(xe; Xe−1; ye−1 ) +  1/2 e (xe; Xe−1 ) (84) = (xe; Xe−1; ye−1 ) −  1/2 e (xe; Xe−1 ) + 2 1/2 e (xe; Xe−1 ) (85) ≤  (xe ) + 2 1/2 e (xe; Xe−1 ) (86) <  (x ∗) −  + 2 √︄  e| T (  )∩T( ) |(X) |T () ∩ T ()| (87) ≤ |  (x ∗) −  ( [x ∗]e)| +  ( [x ∗]e) −  + 2 √︄  e| T (  )∩T( ) |(X) |T () ∩ T ()| (88) ≤ ( [x ∗]e; Xe−1; ye−1 ) +  1/2 e ( [x ∗]e; Xe−1 ) − 2 + 2 √︄  e| T (  )∩T( ) |(X) |T () ∩ T ()| , (89)
$$(88)$$
$$(89)$$
$$(90)$$

for any  > 0. Furthermore,

$$R_{T}^{(1)}(\varepsilon)=\sum_{t\in\mathcal{T}(\varepsilon)}f(\mathbf{x}^{*})-f(\mathbf{x}_{t})$$ $$=2c_{\sup}\sqrt{\frac{2}{\varepsilon}}+\sum_{t\in\mathcal{T}(\varepsilon)\cap\underline{\mathcal{I}}(\varepsilon)}f(\mathbf{x}^{*})-f(\mathbf{x}_{t})$$ $$\leq2c_{\sup}\sqrt{\frac{2}{\varepsilon}}+\frac{\pi^{2}}{6}+2\sqrt{C\beta_{T}|\mathcal{T}(\varepsilon)\cap\underline{\mathcal{I}}(\varepsilon)|\gamma_{|\mathcal{T}(\varepsilon)\cap\underline{\mathcal{I}}(\varepsilon)|}(\mathcal{X})}$$  In the above equation, the last inequality follows from Lemma 21. The proof is 
$$(91)$$
$$(92)$$

$$(93)$$

for any  > 0. In the above expressions, the last inequality follows from Lemma 21. The remaining part of the proof is to substitute the quantity |T () ∩ T ()| in Eq. (93) into its upper bound, which is deduced from Eq. (90) depending on the kernel.

For  = SE. Under  = SE, we crudely take the upper bound of |T () ∩ T ()| as

$$|{\mathcal{T}}(\epsilon)\cap{\mathcal{T}}(\epsilon)|\leq{\frac{16C\beta_{T}\gamma_{|{\mathcal{T}}(\epsilon)\cap{\mathcal{T}}(\epsilon)|}(X)}{\epsilon^{2}}}\leq{\frac{16C\beta_{T}\gamma_{T}(X)}{\epsilon^{2}}}.$$

The above upper bound implies |T () ∩ T ()| =  (  (X)). Since  (X) = (ln+1 ) under = SE, Eq. (93) implies

93) implies  $$R_{T}^{(1)}(\varepsilon)\leq2c_{\sup}\sqrt{\frac{2}{\varepsilon}}+\frac{\pi^{2}}{6}+O\left(\sqrt{\beta_{T}(\beta_{T}\gamma_{T}(X))\ln^{d+1}(\beta_{T}\gamma_{T}(X))}\right)$$ $$=O\left(\beta_{T}\sqrt{(\ln^{d+1}T)\ln^{d+1}(\ln^{d+2}T)}\right)$$ $$=O\left(\sqrt{(\ln T)^{d+3}(\ln\ln T)^{d+1}}\right)$$ $$=\widetilde{O}(1).$$

$$(94)$$

For  = Matern ´. Set Mat > 0 as the constant such that the following inequality holds:

(95)  $\binom{96}{2}$  . 
$$(97)$$
$$\forall t\geq2,\gamma_{t}(X)\leq C_{\mathrm{Mat}}t^{\frac{d}{2\nu+d}}\ln^{\frac{4\nu+d}{2\nu+d}}t.$$
 $$\left({100}\right)$$  $$\left({101}\right)$$  $$\left({102}\right)$$  $$\left({103}\right)$$  ... 
2+ . (99)
The existence of Mat is guaranteed by the upper bound of MIG established in Corollary 8. Then, if |T () ∩ T ()| ≥ 2 holds, Eq. (90) implies

|T () ∩ T ()| ≤ 16 Mat|T () ∩ T ()|   2+ ln 4+ 2+ |T () ∩ T ()|  2(100) ⇒ |T () ∩ T ()| ≤ 16 Mat|T () ∩ T ()|  2+ ln 4+ 2+   2(101) ⇔ |T () ∩ T ()| 2 2+ ≤ 16 Mat ln 4+ 2+   2(102) ⇔ |T () ∩ T ()| ≤ 16 Mat ln 4+ 2+   2 !1+  2 . (103)

Therefore, we have |T () ∩ T ()| = e(1) under fixed , , and . Hence, from Eq. (93), we obtain

(1)

() = e(1). □

## B Information Gain Upper Bound

Our analysis requires the upper bound of MIG with explicit dependence on the radius of the input domain. Several existing works [4, 27, 28] established such a result by extending the proof in [51].

However, the proof strategy in [51] result in e(
(+1)
2+(+1) ) upper bound of MIG in Matern kernel, ´
which is strictly worse than the best achievable e( 2+ ) upper bound. Vakili et al. [58] shows e( 2+ ) upper bound of MIG with  > 1/2 under the uniform boundness assumption of the eigenfunctions. Furthermore, the following work [55] shows  ({x ∈ R

| ∥x∥2 ≤ }) = e( 2 2+ 

2+ )
for any radius  > 0 if there exist eigenfunctions uniformly bounded without depending on  > 0. Some of the related results supports the uniform boundness assumption under  = 1 [27, 62], or under the approximated version of the original Matern kernel [42, 50]; however, to our knowledge, ´ we are not aware of any literature that rigorously support uniform boundness assumption under the general compact input domain with  ≥ 2 and  > 1/2. See Chapter 4.4 in [27] for the detailed discussion. Therefore, this section's goal is twofold: (i) prove e( 2+ ) upper bound as claimed in
[58] without relying on the uniform boundness assumption, and (ii) clarify the explicit dependence on the input radius in the upper bound proved in (i). Below, we formally describe our MIG upper bound.

Theorem 7. *Fix any*  ∈ N+, 
2 >
 0, and  ∈ N+*. Let us assume* X = {x ∈ R

| ∥x∥2 ≤ 1}. Then,

 - *For $k=k_{\rm SE}$, $\gamma_T(X)$ satisfies*. 
$$\gamma_{T}(X)\leq\frac{C_{d}^{(1)}}{\theta^{d}}\ln^{d+1}\left(1+\frac{T}{\sigma^{2}}\right)+\ln\left(1+\frac{T}{\sigma^{2}}\right)+C_{d}^{(2)}\exp\left(-\frac{2}{\theta}+\frac{1}{\theta^{2}}\right)$$  _if $\theta\leq e^{2}c_{d}$ and $T/(e-1)\geq\sigma^{2}$. Furthermore, for any $\theta>e^{2}c_{d}$, we have_  $$\gamma_{T}(X)\leq\frac{C_{d}^{(3)}}{\ln^{d}\left(\frac{\theta}{ec_{d}}\right)}\ln^{d+1}\left(1+\frac{T}{\sigma^{2}}\right)+C_{d}^{(4)}\ln\left(1+\frac{T}{\sigma^{2}}\right)+C_{d}^{(5)}.$$

(104)
$2\ell^{2}$ _and_ $c_{d}=\max\left\{1,\exp\left(\frac{1}{e}\left(\frac{d}{2}-1\right)\right)\right\}$_._ _For_ $\ell^{(5)}>0$ _are the constants only depending on_ $d$

 (105)

```
Here, we set  = 2ℓ
                     2 and  = max n1, exp 1
                                               
                                                2
                                                  − 1
                                                      o. Furthermore,

 (1)
 
   , (2)
     
       , (3)
          
            , (4)
              
                , (5)
                  
                     > 0 are the constants only depending on .

```

- For  = Matern ´ *with*  > 1/2,  (X) *satisfies*

$$(104)$$
$$(105)$$
$$\gamma_{T}(X)\leq C(T,\nu,\sigma^{2})\overline{{{\gamma}}}_{T}+C$$
$$(106)^{\frac{1}{2}}$$
 
+  (106)

```
where (, , 2) = max n1, log2
                            1 +
                                Γ()
                                 
                                   ln 
                                      
                                        2
                                      2
                                         +
                                            1
                                             
                                              log2
                                                  
                                                      2
                                                   Γ() 2
                                                         + 1
                                                             o. Here,  > 0

```

and  > 0 are the constant that only depends on  > 0, and an absolute constant, respectively. Furthermore, 

*is defined as*

$$\overline{\gamma}_{T}=C_{d,\,\nu}^{(1)}\ln\left(1+\frac{2T}{\sigma^{2}}\right)+C_{d,\,\nu}^{(2)}\left(\frac{T}{\sigma^{2}\ell^{2\nu}}\right)^{\frac{d}{2\nu+d}}\ln^{\frac{2\nu}{2\nu+d}}\left(1+\frac{2T}{\sigma^{2}}\right),\tag{107}$$

where 
(1)
,, (2)
, 
>
 0 are the constants only depending on  and .

We also obtain the following corollary by adjusting the lengthscale parameter ℓ > 0 based on the radius of the input domain.

Corollary 8. Fix any  ∈ N+, 
2 > 0,  ∈ N+,  > 0*. Let us assume* X = {x ∈ R

| ∥x∥2 ≤ }.

Then,
- For  = SE,  (X) *satisfies*

$$\gamma_{T}(X)\leq\frac{C_{d}^{(3)}}{\ln^{d}\left(\frac{2\ell^{2}}{\eta^{2}e c_{d}}\right)}\ln^{d+1}\left(1+\frac{T}{\sigma^{2}}\right)+C_{d}^{(4)}\,\ln\left(1+\frac{T}{\sigma^{2}}\right)+C_{d}^{(5)}\,.$$

 (108)
if 2ℓ 2/
2 > 2.

- For  = Matern ´ *with*  > 1/2,  (X) *satisfies Eq.* (106)*, with*

$$\overline{\gamma}_{T}=C_{d,\nu}^{(1)}\ln\left(1+\frac{2T}{\sigma^{2}}\right)+C_{d,\nu}^{(2)}\eta^{\frac{2\nu d}{2\nu+d}}\left(\frac{T}{\sigma^{2}\ell^{2\nu}}\right)^{\frac{d}{2\nu+d}}\ln^{\frac{2\nu}{2\nu+d}}\left(1+\frac{2T}{\sigma^{2}}\right).\tag{109}$$

$$(108)$$