# Tighter Cmi-Based Generalization Bounds Via Stochastic Projection And Quantization

Milad Sefidgaran 1, Kimia Nadjahi 2**, Abdellatif Zaidi** 1,3 1Paris Research Center, Huawei Technologies France 2 CNRS, ENS Paris, France 3 Universite Gustave Eiffel, France ´
milad.sefidgaran2@huawei.com, kimia.nadjahi@ens.fr, abdellatif.zaidi@univ-eiffel.fr

## Abstract

In this paper, we leverage stochastic projection and lossy compression to establish new conditional mutual information (CMI) bounds on the generalization error of statistical learning algorithms. It is shown that these bounds are generally tighter than the existing ones. In particular, we prove that for certain problem instances for which existing MI and CMI bounds were recently shown in Attias et al. [2024] and Livni [2023] to become vacuous or fail to describe the right generalization behavior, our bounds yield suitable generalization guarantees of the order of O(1/
√n), where n is the size of the training dataset. Furthermore, we use our bounds to investigate the problem of data "memorization" raised in those works, and which asserts that there are learning problem instances for which any learning algorithm that has good prediction there exist distributions under which the algorithm must "memorize" a big fraction of the training dataset. We show that for every learning algorithm, there exists an auxiliary algorithm that does not memorize and which yields comparable generalization error for any data distribution. In part, this shows that memorization is not necessary for good generalization.

## 1 Introduction

One of the major problems in statistical learning theory consists in understanding what really drives the generalization error of learning algorithms. That is, what makes an algorithm trained on a given dataset continue to perform well on unseen data samples. Historically, this fundamental question has been studied independently in various lines of work, using seemingly unconnected tools. This includes VC-dimension theory [1], Rademacher complexity approaches [2], stability-based analysis [3] and, more recently, intrinsic-dimension [4–8] and information-theoretic approaches [9–21]. It is only until recently that the above various approaches were shown to be possibly unified [22, 23] using a *variable-length* compressibility technique, which is rate-distortion-theoretic in nature. In the context of statistical learning theory perhaps one can date back information-theoretic approaches to the PAC-Bayes bounds of McAllester [24, 25], which were then followed by various extensions and ramifications [26–39]. The mutual information (MI) bounds of [9] and [10] have the advantages to be relatively simpler comparatively and of offering somewhat clearer insights into the question of generalization. Roughly, such bounds suggest that a learning algorithm generalizes better as its output model reveals less information about the training data samples, where the amount of revealed information is measured in terms of the Shannon mutual information.

However, MI-based bounds are also known to sometimes take large (infinite) values and become vacuous, such as for continuous data and deterministic models. This shortcoming has been identified in a number of works, including [40, 41]. The issue was believed to be resolved by the introduction in [12] of the important framework of conditional mutual information (CMI). The CMI setting introduces a "super-sample" construction in which an auxiliary "ghost sample" is used in conjunction with the training sample; and a sequence of Bernoulli random variables determines which data samples among the super-sample were used for the training. It is shown that a bound on the generalization error involves the mutual information between the Bernoulli random variables and the hypothesis (e.g., model parameters), conditionally given the super-sample [12, Theorem 2]. Because the entropy of Bernoulli random variables is bounded, the resulting bound is bounded. Many follow-up works have proposed extensions and improvements of the original CMI bounds, including using *randomized subset* and *individual sample* techniques, disintegration, and fast-rate variations in regimes in which the empirical risk is small - See [42] for more on this. CMI-type bounds were largely believed to be exempt from the aforementioned limitations of MI bounds until it was recently reported that examples can be constructed for which the standard1 CMI-based bound and its individual-sample variant fail [14, 43, 46]. The (counter-) examples of [46] are in the context of Stochastic Convex Optimization (SCO) problems; and those of [43] involve carefully constructed Convex-Lipschitz-bounded (CLB) and Convex-set-Strongly convex-Lipschitz (CSL) instance problems. These limitations were sometimes extrapolated to the extent of even questioning the utility of informationtheoretic bounds for the analysis of the generalization error of statistical learning algorithms more generally [47]. In this context, we mention [23, Appendix A] in which it was shown that, when applied to the counter-example of [47], a lossy version of MI bounds yields generalization bounds that are of order O(1/n), instead of Ω(1) in the case of standard (lossless) MI bounds.2 The idea of lossy compression was also used in [49]. In this paper, essentially, we show that the aforementioned limitations are in fact not inherent to the CMI framework; and, actually, the CMI framework can be adjusted slightly by the incorporation of a suitable stochastic projection and a suitable lossy compression to cope with those issues. Also, leveraging the utility of CMI and membership inference to study the problem of memorization and its relationship to generalization in machine learning, we use our results to revisit the necessity of memorization for SCO problems claimed in [43]. We show that memorization is not necessary for good generalization; and, as such, the result contributes to a better understanding of what role memorization plays in machine learning, a problem which is yet to be fully understood. Specifically, our contributions are as follows.

- We introduce stochastic projection in conjunction with lossy compression in the CMI framework, and we use them to establish a new CMI-based bound that is generally tighter than the CMI bounds of [12].

- We show that, in sharp contrast with classic CMI-based bounds which fail when applied to the aforementioned CLB, CSL and SCO problem instances of [43, 46] and may not even decay with the number of training samples, our new CMI bound yields meaningful results and decays with the number of training samples as O(1/
√n).

- By applying them to generalized linear stochastic (non-convex) optimization problems, in the appendices we demonstrate that our bounds remain non-vacuous even beyond the convex case previously studied in [50]. The generalization is shown to come at the expense of a slower decay with n in our case; namely, O(1/
√4 n) instead of O(1/
√n) if the functions are convex as in [50].

- We leverage the key ingredients of stochastic projection and lossy compression in the framework of CMI to study the "memorization" issue identified and studied in [43]. Specifically, [43] has demonstrated that, for a given problem instance and every ε-learner algorithm, there exists a data distribution under which the algorithm "memorizes" the training samples. We show that for any learning algorithm A that memorizes the training data, one can find (via stochastic projection and lossy compression) an alternate learning algorithm A˜ with comparable generalization error and that does not memorize the training data for any data distribution. In part, this means that memorization is not necessary for good generalization in SCO.

- In the appendices, we use our general bound to study the generalization error of subspace training algorithms. Specifically, we investigate the setting in which the training is performed using SGD or SGLD; and we derive new bounds based on the differential entropy of Gaussian mixture distributions. This entropy depends on the gradient difference for the training and test datasets, the noise power, the learning rate, and the uncertainty of the index of the training dataset within the super-dataset.

## 2 Notation And Background

Let Z be some random variable with unknown distribution µ and taking values in some alphabet Z. Let Sn ≜ (Z1, . . . , Zn) ∈ Znbe a set of n data samples drawn uniformly from the distribution µ, *i.e.,* Sn ∼ PSn = µ
⊗n. In the framework of statistical learning, a (possibly) stochastic learning algorithm A: Z
n → W takes the training dataset Sn as input and returns a hypothesis W *∈ W ⊆* R
D. We assume that A is *randomized*, in the sense that its output W ≜ A(Sn) is a random variable distributed according to PW|Sn
. We denote the distribution induced on (Sn, W) as PSn,W = PW|Sn ⊗PSn = PW|Sn ⊗µ
⊗n.

For a given function ℓ : *Z × W →* R, the loss incurred by using a hypothesis w ∈ W for a sample z is evaluated as *ℓ(z, w*). A statistical learning algorithm seeks to find a hypothesis w whose population risk R(w) ≜ EZ∼µ[ℓ(*Z, w*)] is minimal. However, since the data distribution µ is unknown, direct computation of the population risk R(w) is not possible. Instead, one resorts to minimizing the empirical risk Rb(sn, w) ≜
1 n Pn i=1 ℓ(zi, w) or a regularized version of it. Throughout, if sn is known from the context, we will use the shorthand notation Rbn(w) ≡ Rb(sn, w).

The *generalization error* induced by a specific choice of hypothesis w ∈ W and dataset sn is evaluated as gen(sn, w) ≜ R(w) − Rbn(w);
and the expected *generalization error* of the learning algorithm A is obtained by taking the expectation over all possible choices of (sn, w), as

$\mathbf{a}\left(\mathbf{a}\right)$
gen(µ, A) ≜ EP*Sn,W* [gen(Sn, W)] = EPSn,W [R(W) − Rbn(W)].

## 2.1 Conditional Mutual Information Framework

Let S˜ ∈ Zn×2be a super-sample composed of 2n data points Zi,j that are drawn uniformly from the distribution µ, where j ∈ {0, 1} and i ∈ [n]. Also, let J = (J1, . . . , Jn) ∈ {0, 1}
nbe a vector of n independent Bernoulli(1/2) random variables, all drawn independently from S˜. Let S˜J = {Z1,J1, Z2,J2, . . . , Zn,Jn }. In what follows, S˜J plays the role of the training dataset Sn, S˜ \ S˜J
plays the role of a test or "ghost" dataset S
′
n and S˜ is a shuffled version of the union of the two. For an algorithm A : Z
n → W, its CMI with respect to the data distribution µ is defined as

$$P_{S_{n},W}[{\mathcal{R}}(W)-{\widehat{\mathcal{R}}}_{n}(W)].$$
$${\mathsf{C M I}}(\mu,{\mathcal{A}})\triangleq\mathrm{I}({\mathcal{A}}({\tilde{\mathbf{S_{J}}}});\mathbf{J}|{\tilde{\mathbf{S}}})\,.$$

The CMI captures the information that the output hypothesis of the algorithm A trained on S˜J provides about the membership vector J given the super-sample S˜. Equivalently, the CMI measures the extent to which the training and test datasets are distinguishable given the shuffled version of the union of the two, as well as the trained model. In its simplest form, it is shown in [12] that the generalization error of an algorithm for a bounded loss in the range [0, 1] can be upper-bounded as

$$\operatorname{gen}(\mu,{\mathcal{A}})\leq{\sqrt{\frac{2}{n}\operatorname{CMI}(\mu,{\mathcal{A}})}}.$$

Furthermore, for a Convex-Lipschitz-Bounded (CLB) whose formal definition will follow, the generalization error of A was shown in [47] to be upper-bounded as

$$\operatorname{gen}(\mu,{\mathcal{A}})\leq L R{\sqrt{\frac{8}{n}}}{\mathsf{C M I}}(\mu,{\mathcal{A}}).$$
$$(1)$$
CMI(µ, A). (1)
Definition 1 (SCO Problem). A stochastic convex optimization (SCO) problem is a triple (W, Z, ℓ)*, where* W ∈ R
D is a convex set and ℓ(z, ·): W → R is a convex function for every z ∈ Z.

Definition 2 (Convex-Lipschitz-Bounded (CLB)). An SCO problem is called CLB if i) *for every* w ∈ W, ∥w∥ ≤ R, and ii) the loss function is convex and L*-Lipschitz,* i.e., ∀z ∈ Z, ∀w1, w2 ∈ W : |*ℓ(z, w*1) − ℓ(z, w2)| ≤ L∥w2 − w1∥. We denote this subclass of SCO problems by CL,R.

## 3 New Cmi-Based Bounds Via Stochastic Projection And Lossy Compression

While the CMI-based bounds are known to be generally tighter than the corresponding MI ones and even tight in some settings [12, 14], they can become vacuous in some cases. This includes the Stochastic Convex Optimization (SCO) examples constructed in the recent works [43, 46], which we will discuss in more detail in Section 4. For these (counter-)examples, it was shown in [43, 46] that CMI-type bounds do not vanish, so they fail to accurately describe the generalization error. In this section, we show that such limitations are not inherent to the CMI framework. In fact, by combining *stochastic projection* with lossy compression (analogously to [49], which addressed the MI case), we derive new CMI-based bounds that do not suffer from such limitations. For instance, when applied to the SCO examples of [43], we show in Section 4 that our new bounds resolve the limitations of other known CMI-based bounds as identified therein. These bounds are also shown in the appendices to apply to the analysis of the generalization error for subspace training algorithms trained with SGD or SGLD. Our new bounds involve two main ingredients, stochastic projection and *lossy compression*.

Stochastic projection. Let Θ ∈ R
D×dbe a random matrix with entries distributed according to some joint distribution PΘ, chosen independently of S˜, In our approach, similar to [49], instead of considering the hypothesis W *∈ W ⊆* R
D which lies in a D-dimensional space, we consider its *projection* Θ
⊤W ∈ R
d onto a smaller d-dimensional space, with d ≪ D.

Lossy Compression. Let ϵ ∈ R be given. An ϵ-lossy algorithm is a (possibly) stochastic map Aˆ: Z
n ×
R 
D×d → Wˆ that maps a pair (Sn, Θ) to a compressed hypothesis or model Wˆ ∈ W ⊆ ˆ R
dgenerated according to some conditional kernel PWˆ |Sn,Θ that satisfies

$$\mathbb{E}_{P_{S_{n},W}P_{\Theta}P_{W|S_{n},\Theta}}\left[\mathrm{gen}(S_{n},W)-\mathrm{gen}(S_{n},\Theta\hat{W})\right]\leq\epsilon.$$

This constraint guarantees that, when projected back onto the original hypothesis space of dimension D,
the compressed model Wˆ has an average generalization error which is within at most ϵ from that of the original model W. In a sense, one works with a compressed model Wˆ which lies in a much smaller dimension space, but with the guarantee that this causes almost no increase in the generalization error. In effect, the *auxiliary* projected-back model ΘWˆ substitutes the original model W.

The concept of a lossy algorithm, also referred to as a "surrogate" or "compressed" algorithm, was introduced in [37, 51, 52] and shown therein to be key to obtaining tighter, non-vacuous, generalization bounds. In this paper, we consider a particular lossy algorithm that involves a suitable stochastic projection followed by quantization. Specifically, we constrain the general conditional PWˆ |Sn,Θ to take the specific form PWˆ |Θ⊤W , where W = A(Sn). Formally, one imposes the Markov chain (Sn, *Θ, W*)−Θ
⊤W −Wˆ
or equivalently PWˆ |Sn,Θ,W = PWˆ |Θ⊤W . In other words, we let Aˆ(Sn, Θ) = A˜(Θ⊤A(Sn)), where A˜: R
d → Wˆ is defined via the Markov kernel PWˆ |Θ⊤A(Sn)
.

Our generalization bounds that will follow are expressed in terms of *disintegrated* CMI, defined as follows.

Let a super-sample S˜ and a stochastic projection matrix Θ be given. The *disintegrated* CMI of an algorithm Aˆ: Z
n → Wˆ is defined as CMIΘ(S˜, Aˆ) ≜ I
S˜,Θ(Aˆ(S˜J, Θ); J),
where Aˆ(S˜J, Θ) = A˜(Θ⊤A(S˜J)) = Wˆ and I
S˜,Θ(Aˆ(S˜J, Θ); J) is the CMI given an instance of S˜ and Θ, computed using the joint distribution PJ ⊗ PW|S˜J
⊗ PWˆ |Θ⊤W , with PJ = Bern(1/2)⊗n.

The next theorem states our main generalization bound and is proved in Appendix E. Theorem 1. *Let a learning algorithm* A: Z
n → W where W ⊆ R
D *be given. Then, for every* ϵ ∈ R,
every d ∈ N*, and every* projected model quantization set W ⊆ ˆ R
d*, we have*

$$\mathbb{C M}^{\mathbb{G}}$$
$${\bar{\mathbf{3}}},{\hat{A}}$$
$${\hat{A}}({\hat{\mathbf{S_{J}}}},\Theta);$$
$\Gamma=\frac{\pi}{2}$. 
$$\mathrm{gen}(\mu,\mathcal{A})\leq\inf_{P_{W|\Theta}\top_{W}}\inf_{P_{\Theta}}\mathbb{E}_{P_{\bar{\Theta}}}P_{\Theta}\left[\sqrt{\frac{2\Delta\ell_{\mathrm{f\bar{\Theta}}}(\hat{\mathbf{S}},\Theta)}{n}\mathsf{CM}^{\Theta}(\hat{\mathbf{S}},\hat{\mathcal{A}})}\right]+\epsilon,\tag{2}$$

where Wˆ ∈ Wˆ , Θ ∈ R
D×d, the infima are over all arbitrary choices of Markov kernel PWˆ |Θ⊤W and distribution PΘ *that satisfy the following distortion criterion:*

$$\mathbb{E}_{P_{S_{n},\,W}P_{\Theta}P_{\hat{W}|\Theta^{\top}W}}\left[\mathrm{gen}(S_{n},W)-\mathrm{gen}(S_{n},\Theta\hat{W})\right]\leq\epsilon,$$
$$,\Theta)\;i s\;g i w e n\;b y$$

and the term ∆ℓwˆ(S˜, Θ) *is given by*

$$\Delta\ell_{\hat{w}}(\hat{\mathbf{S}},\Theta):=\mathbb{E}_{P_{W|\mathbf{S}}P_{W|\Theta}\top_{W}}\left[{\frac{1}{n}}\sum\nolimits_{i\in[n]}(\ell(Z_{i,0},\Theta\hat{W})-\ell(Z_{i,1},\Theta\hat{W}))^{2}\right].$$
$$(3)$$
$$\quad(4)$$

4 Observe that PW|S˜ = EPJ[PW|S˜J
]. Also, if ℓ(·, ·) ∈ [0, C] for some non-negative constant C ∈ R+,
then it is easy to see that the term ∆ℓwˆ(S˜, Θ) is bounded from the above as ∆ℓwˆ(S˜, Θ) ≤ C
2.

The result of Theorem 1 essentially means that the generalization error of the original model is upper bounded by a term that depends on the CMI of the auxiliary model Wˆ plus an additional distortion term that quantifies the generalization gap between the auxiliary and original models. The rationale is that, although the (worst-case) CMI term still depends on the dimension d after stochastic projection, this dimension corresponds to a subspace of the original hypothesis space and can be chosen arbitrarily small in order to guarantee that the bound vanishes with n. Also, the term in left-hand-side (LHS) of equation 3 represents the average distortion (measured by the difference of induced generalization errors) between the original model and the one obtained after projecting back the auxiliary compressed model onto the original hypothesis space. The analysis of this term may seem non-easy; but as visible from the proof, it is not.

This is because, defined as a difference term, its analysis does not necessitate accounting for statistical dependencies between S and W. Instead, one only needs to account for the effect of the following sources of randomness: i) the stochastic projection matrix, ii) the quantization noise, and **iii)** discrepancies between the empirical measure of S and the true unknown distribution µ. As shown in the proofs, the analysis of the distortion term involves the use of classic concentration inequalities. Furthermore, the construction of Wˆ allows us to consider the worst-case bound for the CMI-terms of the RHS of equation 2 without losing the order-wise optimality in certain cases.

We close this section by noting that it is well known that CMI-type bounds can be improved by application of suitable techniques such as random-subset or *individual sample* techniques or in order to get fast rates O(1/n) for small empirical risk regimes, see, e.g., [20, 53, 54]. These same techniques can be applied straightforwardly to our bound of Theorem 1 to get improved ones. For the sake of brevity, we do not elaborate on this here; and we refer the reader to the supplements where a single-datum version of Theorem 1 is provided.

## 4 Application To Resolving Recently Raised Limitations Of Classic Cmi Bounds

Prior works [43, 46] have recently reported carefully constructed counter-example learning problems and have shown that classic MI-based and CMI-based bounds fail to provide meaningful results when applied to them. In this section, we show that the careful addition of our stochastic projection along with our lossy compression resolves those issues, in the sense that the resulting new bound (our Theorem 1), which is still of CMI-type, now yields meaningful results when applied to those counter-examples. In essence, the improvement is brought up by: (i) noticing that the aforementioned negative results for standard CMI-based generalization error bounds rely heavily on that the dimension of the hypothesis space grows fast with n (over-parameterized regime), e.g., as Ω(n 4log n) in the considered counter-examples of [43], which calls for suitable projection onto a smaller dimension space in which this does not hold, and (ii) properly accounting for the distortion induced in the generalization error after projection back to the original high dimensional space.

First, we recall briefly the counterexamples mentioned in [43] and [46]; and, for each of them, we show how our bound of Theorem 1 applies successfully to it. Recall the definitions of a stochastic convex optimization (SCO) problem and a Convex-Lipschitz-Bounded (CLB) SCO problem as given, respectively, in Definition 1 and Definition 2.

Definition 3 (ε-learner for SCO). Fix ϵ > 0. For a given SCO problem (W, Z, ℓ), A = {An}n≥1 is called an ε-learner algorithm with sample complexity N : R × R → N *if the following holds: for every* δ ∈ (0, 1] and n ≥ N(ε, δ) we have that for every µ ∈ M1(Z), where M1(Z) denotes the set of probability measures on Z, with probability at least 1 − δ *over* Sn ∼ µ
⊗nand internal randomness of A,
R(An(Sn)) − min w∈W
R(w) ≤ ε. (5)

## 4.1 Counter-Example Of Attias Et Al. [2024] For Clb Class

Denote by BD(ν) the D-dimensional ball of radius ν ∈ R+.

Definition 4 (Problem instance P
(D)
cvx ). Let L, R ∈ R+, *Z ⊆ B*D(1), and W = BD(R). Define the loss function ℓ : *Z × W →* R as ℓc*(z, w*) = −L⟨*w, z*⟩.

We denote this SCO problem instance as P
(D)
cvx . It is easy to see that this optimization problem belongs to the subclass CL,R of SCO problems as defined in Definition 2.

For this (counter-) example learning problem, [43] have shown that for every ε-learner there exists a data distribution for which the CMI bound of equation 1 for the optimal sample complexity, which is Θ
 LR
ε 2as shown in [50], scales just as Θ(LR). For instance, that CMI-bound on the generalization error does not decay with the size n of the training dataset!

Theorem 2 (CMI-accuracy tradeoff, [43, Theorems 4.1 and 5.2]). Let ε0 ∈ (0, 1) be a universal constant. Consider the above defined P
(D)
cvx problem instance with parameters (L, R). Consider any ϵ ≤ ϵ0 and for any algorithm A = {An}n∈N that ε*-learns* P
(D)
cvx with sample complexity N(·, ·)*. Then, the following* holds: i. For every δ ≤ ε, n ≥ N(ε, δ)*, and* D = Ω n 4log(n),
3there exists a set *Z ⊆ B*D(1)
and a data distribution µ ∈ M1(Z), denoted as µp∗ *, such that* CMI(µ, An) = Ω  LR
ε 2. ii. In particular, considering the optimal sample complexity N(*ε, δ*) = Θ L2R2 ε2*, the CMI generalization* bound of equation 1 equals LRp8CMI(µ, An)/N(*ε, δ*) = Θ(LR).

For this example, it was further shown [43, Corollary 5.6] that application of the *individual sample* technique of [55, 56] (which is traditionally used to avoid the unbounded-ness issue as instance of so called randomized-subset techniques wherein the linearity of the expectation operator is used to obtain an average bound for the loss on randomly chosen subsets of the training set rather than the loss averaged over the full training set) actually yields the very same bound order-wise; and, thus, it does not resolve the issue for this counter-example.

Furthermore, as shown in [43, Equation 1], the expectation of the LHS of equation 5 can be bounded as

$$\mathbb{E}\left[\mathcal{R}(\mathcal{A}_{n}(S_{n}))\right]-\min_{w\in\mathcal{W}}\mathcal{R}(w)\leq LR\sqrt{\frac{\text{SCMI}(\mu,\mathcal{A}_{n})}{n}}+\mathbb{E}\left[\widehat{\mathcal{R}}_{n}(\mathcal{A}_{n}(S_{n}))-\min_{w\in\mathcal{W}}\widehat{\mathcal{R}}_{n}(w)\right].\tag{6}$$

Thus, while the LHS of this inequality is bounded from above by ε by assumption, its right-hand side (RHS) is Θ(LR) by Theorem 2. This means that the CMI bound of equation 1 fails to describe well the excess error of the LHS. In [43], this was even somewhat extrapolated to negatively answer the question about "*whether the excess error decomposition using CMI can accurately capture the worst-case excess* error of optimal algorithms for SCOs". The above applies for any ε-learner of the problem instance P
(D)
cvx when Z = {±1/
√D}
D and µp∗ (z) =
QD
k=1 1+√Dzkp
∗ k 2
,
4for all z = (z1*, . . . , z*D), where p
∗ = (p
∗1*, . . . , p*∗D) ∈ [−1, 1]D.

The next theorem shows that when applied to the aforementioned counter-example, our new CMI-bound of Theorem 1 does not suffer from those shortcomings. Also, this holds true for: (i) arbitrary values of the dimension D ∈ N including n-dependent ones, **(ii)** arbitrary learning algorithms (including the ε-learners of P
(D)
cvx ), **(iii)** arbitrary choices of *Z ⊆ B*D(1) and **(iv)** arbitrary data distributions µ.

Theorem 3. *For every learning algorithm* A: Z
n → W *of the instance* P
(D)
cvx defined as in Definition 4, the generalization bound of Theorem 1 yields

$$\operatorname{gen}(\mu,{\mathcal{A}})\leq{\frac{8L R}{\sqrt{n}}}.$$
_In particular, setting $N(\varepsilon,\delta)=\Theta\left(\frac{L^{2}R^{2}}{\varepsilon^{2}}\right)$ for $\varepsilon$-learner algorithms we get_  $$\text{gen}(\mu,\mathcal{A})=\mathcal{O}\left(\varepsilon\right).$$

The proof of Theorem 3 is deferred to Appendix F.2.

Some remarks are in order. First, while when applied to the studied counter-example the CMI bound of equation 1 yields a bound of the order Θ(LR), i.e., one that does not decay with n, our new CMI-
based bound of Theorem 1 yields one that decays with n as O(LR/√n). Second, when specialized to the case of ε-learner algorithms and considering the sample complexity Θ
 LR
ε 2, we get a bound on the generalization error of the order O (ε). Using this bound, we can write

$$\mathbb{E}_{P_{S_{n},\,w}}\left[\mathcal{R}(\mathcal{A}_{n}(S_{n}))\right]-\min_{w\in\mathcal{V}}\mathcal{R}(w)\leq\mathcal{O}\left(\varepsilon\right)+\mathbb{E}_{P_{S_{n},\,w}}\left[\widehat{\mathcal{R}}_{n}(\mathcal{A}_{n}(S_{n}))-\min_{w\in\mathcal{V}}\widehat{\mathcal{R}}_{n}(w)\right].\tag{7}$$

Contrasting with equation 6 and noticing that if the second term of the summation of the RHS of equation 7 (optimization error) is small then both sides of equation 7 are O(ϵ), it is clear that now the excess error decomposition using our new CMI-based bound can accurately capture the worst-case excess error.

Third, as it can be seen from the proof, stochastic projection onto a one-dimensional space, i.e., d = 1, is sufficient to get the result of Theorem 3. In essence, this is the main reason why, in sharp contrast with projection- and lossy-compression-free CMI-bounds, ours of Theorem 1 does not become vacuous.

That is, one can reduce the effective dimension of the model for the studied example even if the original dimension D is allowed to grow with n as Ω(n 4log(n)) as judiciously chosen in[43] for the purpose of making classic CMI-based bounds fail. Furthermore, it is worth noting that, for this problem, the projection is performed using the famous Johnson-Lindenstrauss [57] dimension reduction algorithm. Since this dimension reduction technique is "lossy", controlling the induced distortion is critical. To do so, we introduce an additional lossy compression step by adding independent noise in the lower-dimensional space. This approach is reminiscent of lossy source coding and allows to obtain possibly tighter bounds on the quantized, projected model. Finally, we mention that for bigger class problem instances or for the memorization problem of Section 5, projection onto one-dimensional spaces may not be enough to get the desired order O*(LR/*√n). In Appendix B, it will be shown that for generalized linear stochastic optimization problems, one may need d = Θ(√n). Similarly, in Section 5 and Appendix C, projections with d = n 2r−1, r < 1 and d = Θ(log n) are used.

## 4.2 Counter-Example Of Attias Et Al. [2024] For Csl Class

The question of whether classic CMI-bounds and individual-sample versions thereof may still fail if one considers more structured subclasses of SCO problems was raised (and answered positively!) in Attias et al. [43]. For convenience, we recall the following two definitions. Definition 5 (Convex set-Strongly Convex-Lipschitz (CSL)). An SCO problem is called CSL if i) the loss function is L-Lipschitz, and ii) the loss function is λ*-strongly convex,* i.e., ∀z ∈ Z, ∀w1, w2 ∈ W : ℓ(z, w2) ≥ ℓ(*z, w*1) + ⟨∂ℓ(z, w1), w2 − w1⟩ +
λ 2
∥w2 − w1∥
2, where ∂ℓ(z, w1) *is the subgradient* of ℓ(z, ·) at w1. We denote this subclass by CL,λ.

Definition 6 (Problem instance P
(D)
scvx). Let λ, R ∈ R+, *Z ⊆ B*D(1), and W = BD(R)*. Define the loss* function ℓ : *Z × W →* R as ℓsc(z, w) = −Lc⟨*w, z*⟩ +
λ 2
∥w∥
2*. We denote this SCO problem as* P
(D)
scvx, which belongs to CL,λ*, with* L = Lc + λR. Setting λ = Lc = R = 1, D = Ω(n 4log(n)), δ = O(1/n2), Z = {±1/
√D}
D and for a particular data distribution that is carefully chosen therein (not reproduced here for brevity), [43, Theorem 4.2] states that for any learning algorithm that ε-learns the problem instance P
(D)
scvx,

$$\operatorname{CM}(\mu,{\mathcal{A}}_{n})=\Omega\left({\frac{1}{\varepsilon}}\right).$$

Moreover, the application of the individual-sample technique does not result in better decay of the bound order-wise [43, Corollary 5.7].

Noticing that (i) the loss ℓsc(z, w) = −Lc⟨*w, z*⟩ +
λ 2
∥w∥
2considered in Definition 6 differs from that ℓsc*(z, w*) = −L⟨*w, z*⟩ of Definition 4 essentially through the added squared magnitude of the model and (ii) that addition does not alter the generalization error of a given learning algorithm, then it is easy to see that Theorem 3 also applies for the problem P
(D)
scvx at hand; and, in this case, it gives a bound of the order O(1/
√n). This is stated in the next proposition, which is proved in Appendix F.3.

Proposition 1. *For every learning algorithm* A: Z
n → W *of the instance* P
(D)
scvx defined as in Definition 6 the generalization bound of Theorem 1 yields

$$\operatorname{gen}(\mu,{\mathcal{A}})\leq{\frac{8L_{c}R}{\sqrt{n}}}.$$
.
In particular, choosing Lc = R = λ = 1 and setting N(*ε, δ*) = cε for some non-negative constant c ∈ R+
for the ERM algorithm (which is an ε*-learner - see, e.g., [50, Theorem 6]), one gets* gen(µ, A) = O (
√ε).

## 4.3 Counter-Example Of Livni [2023]

The counter-example of [46] is the same as the problem instance of Definition 4, with the one difference that the loss function is taken to be the squared distance instead of the inner product, i.e., ℓ(*z, w*) =
−L∥w − x∥
2, for some non-negative constant L ∈ R+. Livni [46] has shown that the MI bound of [11]
(which is a single-datum bound) fails and becomes vacuous when evaluated for this particular learning problem. However, since ℓ(z, w) = −L∥x∥
2 − L∥w∥
2 + 2L⟨*w, x*⟩ and noticing that the squared norm terms do not alter the generalization error relative to when computed for a loss function given by only the inner-product term, it follows that Theorem 3 still applies and gives a bound of the order O(1/
√n)
for this problem instance. In addition, for the optimal sample complexity, the bound is O(ε). In essence, this means that unlike the MI bound of [11], our new CMI-based bound of Theorem 1 does not become vacuous when applied to the problem at hand.

In Appendix B, we apply the bound of Theorem 1 to a wider family of generalized linear stochastic optimization problems. In particular, we show that no counter-example could be found for which the bound of Theorem 1 does not vanish, even if one considers the bigger class of generalized linear stochastic optimization problems in place of the SCO class problems of [43].

## 5 Memorization

Loosely speaking, a learning algorithm is said to "memorize" if by only observing its output model, an adversary can correctly guess elements of the training data among a given super-sample. For the CLB and CSL subclasses of problems studied in Section 4, Attias et al. [43] showed that there are problem instances for which, for any ε-learner algorithm, there exists a data distribution under which the learning algorithm
"memorizes" most of the training data. This is obtained by designing an adversary capable of identifying a significant fraction of the training samples.

In this section, we show that given a learning algorithm A that memorizes the training samples, one can find (via stochastic projection and lossy compression) an alternate learning algorithm A˜ with comparable generalization error and that does not memorize the training data.5 Definition 7 (Recall Game [43, Definition 4.3]). Given A = {An}n≥1*, let* Q: R
D *× Z × M*1(Z) →
{0, 1} *be an adversary for the following game. For* i ∈ [n]*, given a fresh data point* Z
′i ∼ µ *independent* of (Zi, W), let Zi,1 = Zi and Zi,0 = Z
′i. Then, the adversary is given Zi,Ki, where Ki ∼ *Bern*(1/2) is independent of other random variables. The adversary declares Kˆi ≜ Q(W, Zi,Ki, µ) *as its guess of* Ki.

The game consists of n rounds. At each round i ∈ [n], a pair (Zi,0, Zi,1) is considered and the adversary makes two independent guesses: one for the sample Zi,0, the other for Zi,1. Definition 8 (Soundness and recall [43, Definition 4.4]). Consider the setup of Definition 7. Assume that the adversary plays the game in n *rounds. For every round* i ∈ [n], the adversary plays two times, independently of each other, using respectively (W, Zi,0, µ) and (W, Zi,1, µ) *as input. Then, for a given* ξ ∈ [0, 1], the adversary is said to be ξ*-sound if* P (∃ i ∈ [n]: Q(W, Zi,0, µ) = 1) ≤ ξ. Also, the adversary certifies the recall of m samples with probability q ∈ [0, 1] if P
Pi∈[n] Q(W, Zi,1, µ) ≥ m
≥ q.

If both conditions are met, we say that the adversary (m, q, ξ)*-traces the data.*
Clearly, the concept of (m, q, ξ)-*tracing* the data by an adversary is most interesting for values of (*m, q, ξ*) that are such that: ξ is small (i.e., the adversary makes accurate predictions), m is large and q is nonnegligible (i.e., the adversary can recall a significant part of the training data). As Lemma 1, which is stated in Appendix C.1, asserts, certain values of (*m, q, ξ*) can be attained even by a "dummy" adversary that makes guesses without even looking at the given data sample. For the problem instance P
(D)
cvx , Attias et al. [43] have shown that, for every ϵ-learner algorithm, there exist a distribution and an adversary that is capable of identifying a significant portion of the training data. Theorem 4 ([43, Theorem 4.5]). *Consider the* P
(D)
cvx *problem instance of Definition 4 with* L = R = 1.

Fix arbitrary ξ ∈ (0, 1] *and let* Z = {±1/
√D}
D. Let ε0 ∈ (0, 1) be a universal constant. Let ε > 0 such that ε < ε0, δ < ε. Then, given any ε-learner algorithm A with sample complexity *N(ε, δ*) =
Θ(log(1/δ)/ε2), there exist a data distribution µp∗ and an adversary such that for n = N(ε, δ) and D = Ω(n 4log(n/ξ))*, the adversary* Ω(1/ε2), 1/3, ξ*-traces the data.*
5The memorization problem has also been studied in [58] via some examples in which the data distribution µ is not fixed and comes from a meta-distribution, i.e. µ ∼ Pµ. Instead of using the recall game, [58] measured the amount of memorization by I(S; W|µ). A key implication of Theorem 4 is that, for some fixed q > 0, the result holds even when ξ ∈ (0, 1] is arbitrarily small and m *= Ω(n*) (by choosing ε = O(1/
√n)). In other words, for the considered class of problems Pcvx with data drawn from µp∗ , the constructed adversary can provably trace an arbitrarily large part of the training dataset.

We show that the stochastic projection and lossy compression techniques used in the CMI framework can partially mitigate this memorization issue, in a sense that will be made precise in Theorem 8. To this end, we first establish a general result on memorization.

Theorem 5. Consider any learning algorithm A = {An}n≥1 *such that* CMI(µ, An) = o(n). Then, for any adversary for this learning algorithm that (m, q, ξ)*-traces the data, the following holds:* i) m = o(n)
or ξ ≥ q, ii) if, for some α ∈ (0, 1) and n0 ∈ N
∗, m ≥ αn for every n ≥ n0, then for any ϵ ∈ (0, α) it holds that: P
Pi∈[n] Q(W, Zi,0, µ) ≥ m′≥ (α − ϵ)q, where m′ =ϵ 1/q+ϵ−α n − o(n) = Ω(n).

Theorem 5, whose proof is provided in Appendix G.1, applies to any learning problems. In particular, it is not limited to P
(D)
cvx or the CLB subclass. The argument relies on Fano's inequality for approximate recovery [59, Theorem 2]. We construct a suitable estimator of the index set J based on the adversary's guesses, and we show that if this estimator can correctly recover a fraction c > 12 of the membership indices J, then CMI(µ, An) = Θ(n). Theorem 5 i) means that if the CMI of a learning algorithm is of order o(n), then any adversary that recalls a non-negligible fraction of the training dataset with some probability q (*i.e.,* , m = Θ(n)) is q-sound at best. This means that, in this regime, no adversary can do better than a dummy one that makes random guesses independently of the data (See Lemma 1 in Appendix C.1 for what is attainable by a dummy adversary). Theorem 5 ii) means that if an adversary recalls Ω(n) training samples with some probability, then it must also incorrectly guess the membership of Ω(n) test samples with some non-negligible probability. Next, we use the result of Theorem 5 for P
(D)
cvx to show that while the output model W of any ε-learner algorithm must memorize a significant fraction of the data (for some distribution) as asserted in Theorem 4 the auxiliary model ΘWˆ (which is obtained through suitable stochastic projection and lossy compression),
achieves comparable generalization error *without* memorizing the data!

Theorem 6. *Consider the* P
(D)
cvx *problem instance of Definition 4 with* L = R = 1. For every r > 0, every *Z ⊆ B*D(1) *and every learning algorithm* A: Z
n → R
D, there exists another (compressed)
algorithm A
∗: Z
n → R
D*, defined as* A
∗(Sn) ≜ ΘA˜(Θ⊤A(Sn)) = ΘWˆ *, where the projection matrix* Θ ∈ R
D×d, d = 500r log(n), is distributed according to some distribution PΘ independent of (Sn, W),
such that for any data distribution µ*, the following conditions are met simultaneously:*
i) the generalization error of the auxiliary model ΘWˆ *satisfies*

$$\left|\Xi_{P S_{n},W}P_{\Theta}P_{W|\Theta^{\top}W}\left[\mathrm{gen}(S_{n},W)-\mathrm{gen}(S_{n},\Theta\bar{W})\right]\right|=\mathcal{O}\left(n^{-r}\right),$$
$$(8)$$
−r, (8)
ii) if there exists an adversary that by having access to both Θ and Wˆ (and hence ΘWˆ ) (*m, q, ξ*)-
traces the data, then it must be that: a) m = o(n) or ξ ≥ q, and b) *if, for some* α ∈ (0, 1)
and n0 ∈ N
∗, m ≥ αn for every n ≥ n0*, then for any* ϵ ∈ (0, α) *it holds that:*
P 
Pi∈[n] Q(ΘW , Z ˆ i,0, µ) ≥ m′≥ (α − ϵ)q, where m′ =ϵ 1/q+ϵ−α n − o(n) = Ω(n).

Theorem 6, proved in Appendix G.2, holds for Θ being stochastic and shared with the adversary. In essence, it asserts that for any algorithm A(S) = W, one can construct a suitable projected-quantized model Aˆ(S, Θ) = Wˆ from which no adversary would be able to trace the data, for any data distribution µ. It is appealing to contrast this result with that of [43, Theorem 4.5] on the necessity of memorization. Consider the SCO instance problem with O(1) convex-Lipschitz loss defined over the ball of radius one in R
D considered in [43, Theorem 4.5] and let an ε-learner algorithm A with output model W and sample complexity N(*ε, δ*) = Θ(log(1/δ)/ε2) with D = Ω(n 4log(n/ξ)) be given. The result of [43, Theorem 4.5] states that there exists a data distribution for which the algorithm A must memorize a big fraction of the training data. Applied to this particular instance problem, Theorem 6 asserts that if a random Θ is chosen and shared with the adversary then the auxiliary model ΘWˆ has the following guarantees:
(i) for any data distribution, no adversary can trace the data, and (ii) on average over Θ the associated generalization error is arbitrarily close to that of the original model W. At first glance, this may seem to contradict the necessity of memorization stated in [43, Theorem 4.5]. It is important to note, however, that the auxiliary algorithm does not satisfy the conditions required in [43, Theorem 4.5]; and, so, the latter does not apply to ΘWˆ . In particular, while [43, Theorem 4.5] requires the model to be bounded, in our construction for every w we have EW , ˆ Θ
-ΘWˆ≈ w but EW , ˆ Θ
h
ΘWˆ
2iincreases roughly as D
d
(see Lemma 2 in Appendix C.4.1). As discussed after Lemma 2, this causes EW , ˆ Θ
h
ΘWˆ
2ito grow as Ω(n 3) when D = Ω(n 4log(n/ξ)), i.e., it becomes arbitrarily large as n increases. Intuitively, this is what prevents an adversary from guessing correctly whether a sample has (or not) been used for training, and which makes some key proof steps of Attias et al. fail when applied to the auxiliary model ΘWˆ . These steps are discussed in detail in Appendix C.4.2. A somewhat weaker version of Theorem 6, which is stated in Theorem 8 in Appendix C.2, holds for the projection matrix Θ being *deterministic*. In a sense, it provides a stronger guarantee on the generalization error of the auxiliary model, in that the closeness to the performance of the original model holds now for the given Θ and not only in average over Θ as in Theorem 6. However, this comes at the expense of the auxiliary algorithm being dependent on the data distribution. A consequence of this is that the result does not preclude the existence of other distributions for which there would exist adversaries capable of tracing the data. Moreover, in Theorem 9 in Appendix C.3, we show that a similar result holds if one considers the closeness in terms of the population risk, instead of the generalization error.

Summarizing, neither of the results of Theorem 6 and Theorem 8 contradict those of [43]. In essence, they assert that for any learning algorithm A one can find an alternate auxiliary algorithm via stochastic projection combined with lossy compression for which no adversary would be able to trace the data; and, yet, the found auxiliary algorithm has generalization error that is arbitrarily close to that of the original model. Appendix C.3 extends this closeness to the population risk.

## 6 Implications And Concluding Remarks Sample-Compression Schemes

Formally, a learning algorithm is a sample compression scheme of size k ∈ N if there exists a pair of mappings (*ϕ, ψ*) such that for all samples S = (Z1*, . . . , Z*n) of size n ≥ k, the map ϕ compresses the sample into a length-k sequence which the map ψ uses to reconstruct the output of the algorithm, i.e., A(S) = *ψ(ϕ(S*)). Steinke and Zakynthinou [12] establish that if an algorithm An is a samplecompression scheme (*ϕ, ψ*) of size k, then it must be that the associated CMI is bounded from above as CMI(An) ≤ k log(2n). The finding of [43] that, for certain SCO problem instances, every ε-learner algorithm must have CMI that blows up with n (faster than n) was used therein to refute the existence of such sample-compression schemes for the studied SCO problems. The results of this paper may constitute a path to obtaining such schemes when the definition is extended to involve approximate reconstruction (in terms of induced generalization error) instead of the strict An(·) = ψ(ϕ(·)) of Littlesone and Warmuth [60].

## Fingerprinting Codes And Privacy Attacks

In [61], the authors study the problem of designing privacy attacks on mean estimators that expose a fraction of the training data. They show that a well-designed adversary can guess membership of the training samples from the output of every algorithm that estimates mean with high precision. Our results suggest that stochastic projection and lossy compression might be useful to construct differentially private codes that prevent such fingerprinting type attacks. For instance, while noise would naturally be one constituent of the recipe in this context, its injection in a suitable smaller subspace of the summary statistics might be the key enabler of privacy guarantees in such contexts.

## Concluding Remarks

In this work, we revisit recent limitations identified in conditional mutual information-based generalization bounds. By incorporating stochastic projections and lossy compression mechanisms into the CMI framework, we derive bounds that remain informative in stochastic convex optimization, thereby offering a new perspective on the results in [43, 46]. Our approach also provides a constructive resolution to the memorization phenomenon described in [43], by showing that for any algorithm and data distribution, one can construct an alternative model that does not trace training data while achieving comparable generalization. Like prior work on information-theoretic bounds, our analysis applies to stochastic convex optimization. A natural, open question is whether and how these results can be extended to more general learning settings. Another key direction is to translate our theoretical findings into actionable design principles for learning algorithms with controlled generalization and compressibility.

## Acknowledgments

The authors thank the anonymous reviewers for their many insightful comments and suggestions. Their feedback and the ensuing discussions led to the alternative variants of Theorem 8 (*i.e.*, Theorem 6 and Theorem 9), and greatly shaped some of the paper's discussions. Kimia Nadjahi would also like to thank Mahdi Haghifam for the helpful discussions.

## References

[1] VN Vapnik and A Ya Chervonenkis. On the uniform convergence of relative frequencies of events to their probabilities. *Theory of Probability and its Applications*, 16(2):264, 1971.

[2] Peter L Bartlett, Olivier Bousquet, and Shahar Mendelson. Local rademacher complexities. Annals of Statistics, pages 1497–1537, 2005.

[3] Shai Shalev-Shwartz, Ohad Shamir, Nathan Srebro, and Karthik Sridharan. Learnability, stability and uniform convergence. *The Journal of Machine Learning Research*, 11:2635–2670, 2010.

[4] Umut S¸ims¸ekli, Ozan Sener, George Deligiannidis, and Murat A Erdogdu. Hausdorff dimension, heavy tails, and generalization in neural networks. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, *Advances in Neural Information Processing Systems*, volume 33, pages 5138–5151. Curran Associates, Inc., 2020.

[5] Tolga Birdal, Aaron Lou, Leonidas Guibas, and Umut S¸ims¸ekli. Intrinsic dimension, persistent homology and generalization in neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

[6] Liam Hodgkinson, Umut Simsekli, Rajiv Khanna, and Michael Mahoney. Generalization bounds using lower tail exponents in stochastic optimizers. In *International Conference on Machine Learning*,
pages 8774–8795. PMLR, 2022.

[7] Soon Hoe Lim, Yijun Wan, and Umut S¸ims¸ekli. Chaotic regularization and heavy-tailed limits for deterministic gradient descent. *arXiv preprint arXiv:2205.11361*, 2022.

[8] Yijun Wan, Melih Barsbey, Abdellatif Zaidi, and Umut S¸ims¸ekli. Implicit compressibility of overparametrized neural networks trained with heavy-tailed SGD. In Proceedings of the 41st International Conference on Machine Learning, pages 49845–49866, 2024.

[9] Daniel Russo and James Zou. Controlling bias in adaptive data analysis using information theory. In Arthur Gretton and Christian C. Robert, editors, *Proceedings of the 19th International Conference* on Artificial Intelligence and Statistics, volume 51 of *Proceedings of Machine Learning Research*, pages 1232–1240, Cadiz, Spain, 09–11 May 2016. PMLR.

[10] Aolin Xu and Maxim Raginsky. Information-theoretic analysis of generalization capability of learning algorithms. *Advances in Neural Information Processing Systems*, 30, 2017.

[11] Yuheng Bu, Shaofeng Zou, and Venugopal V. Veeravalli. Tightening mutual information-based bounds on generalization error. *IEEE Journal on Selected Areas in Information Theory*, 1(1): 121–130, May 2020. ISSN 2641-8770.

[12] Thomas Steinke and Lydia Zakynthinou. Reasoning about generalization via conditional mutual information. In Jacob Abernethy and Shivani Agarwal, editors, Proceedings of Thirty Third Conference on Learning Theory, volume 125 of *Proceedings of Machine Learning Research*, pages 3437–3452. PMLR, 09–12 Jul 2020.

[13] Amedeo Roberto Esposito, Michael Gastpar, and Ibrahim Issa. Generalization error bounds via Renyi-, ´ f-divergences and maximal leakage, 2020.

[14] Mahdi Haghifam, Gintare Karolina Dziugaite, Shay Moran, and Daniel M. Roy. Towards a unified information-theoretic framework for generalization. In Thirty-Fifth Conference on Neural Information Processing Systems, 2021.

[15] Gergely Neu, Gintare Karolina Dziugaite, Mahdi Haghifam, and Daniel M. Roy. Informationtheoretic generalization bounds for stochastic gradient descent, 2021.

[16] Gholamali Aminian, Yuheng Bu, Laura Toni, Miguel Rodrigues, and Gregory Wornell. An exact characterization of the generalization error for the gibbs algorithm. Advances in Neural Information Processing Systems, 34:8106–8118, 2021.

[17] Ruida Zhou, Chao Tian, and Tie Liu. Individually conditional individual mutual information bound on generalization error. *IEEE Transactions on Information Theory*, 68(5):3304–3316, 2022. doi: 10.1109/TIT.2022.3144615.

[18] Gabor Lugosi and Gergely Neu. Generalization bounds via convex analysis. In ´ Conference on Learning Theory, pages 3524–3546. PMLR, 2022.

[19] Saeed Masiha, Amin Gohari, and Mohammad Hossein Yassaee. f-divergences and their applications in lossy compression and bounding generalization error. *IEEE Transactions on Information Theory*, 2023.

[20] Hrayr Harutyunyan, Maxim Raginsky, Greg Ver Steeg, and Aram Galstyan. Information-theoretic generalization bounds for black-box learning algorithms. Advances in Neural Information Processing Systems, 34, 2021.

[21] Fredrik Hellstrom and Giuseppe Durisi. A new family of generalization bounds using samplewise ¨
evaluated cmi. *Advances in Neural Information Processing Systems*, 35:10108–10121, 2022.

[22] Milad Sefidgaran, Romain Chor, and Abdellatif Zaidi. Rate-distortion theoretic bounds on generalization error for distributed learning. *Advances in Neural Information Processing Systems*, 35: 19687–19702, 2022.

[23] Milad Sefidgaran and Abdellatif Zaidi. Data-dependent generalization bounds via variable-size compressibility. *IEEE Transactions on Information Theory*, 2024.

[24] David A McAllester. Some PAC-Bayesian theorems. In Proceedings of the eleventh annual conference on Computational learning theory, pages 230–234, 1998.

[25] David A McAllester. PAC-Bayesian model averaging. In Proceedings of the twelfth annual conference on Computational learning theory, pages 164–170, 1999.

[26] Matthias Seeger. PAC-Bayesian generalisation error bounds for gaussian process classification. Journal of machine learning research, 3(Oct):233–269, 2002.

[27] John Langford and Rich Caruana. (not) bounding the true error. Advances in Neural Information Processing Systems, 14, 2001.

[28] Olivier Catoni. A PAC-Bayesian approach to adaptive classification. *preprint*, 840, 2003.

[29] Andreas Maurer. A note on the pac bayesian theorem. *arXiv preprint cs/0411099*, 2004. [30] Pascal Germain, Alexandre Lacasse, Franc¸ois Laviolette, and Mario Marchand. PAC-Bayesian learning of linear classifiers. In Proceedings of the 26th Annual International Conference on Machine Learning, pages 353–360, 2009.

[31] Ilya O Tolstikhin and Yevgeny Seldin. PAC-Bayes-empirical-bernstein inequality. Advances in Neural Information Processing Systems, 26, 2013.

[32] Luc Begin, Pascal Germain, Franc¸ois Laviolette, and Jean-Francis Roy. PAC-Bayesian bounds based ´
on the renyi divergence. In ´ *Artificial Intelligence and Statistics*, pages 435–444. PMLR, 2016.

[33] Niklas Thiemann, Christian Igel, Olivier Wintenberger, and Yevgeny Seldin. A strongly quasiconvex PAC-Bayesian bound. In *International Conference on Algorithmic Learning Theory*, pages 466–492. PMLR, 2017.

[34] Gintare Karolina Dziugaite and Daniel M Roy. Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data. arXiv preprint arXiv:1703.11008, 2017.

[35] Behnam Neyshabur, Srinadh Bhojanapalli, and Nathan Srebro. A PAC-Bayesian approach to spectrally-normalized margin bounds for neural networks, 2018.

[36] Omar Rivasplata, Ilja Kuzborskij, Csaba Szepesvari, and John Shawe-Taylor. PAC-Bayes analysis ´
beyond the usual bounds. *Advances in Neural Information Processing Systems*, 33:16833–16845, 2020.

[37] Jeffrey Negrea, Gintare Karolina Dziugaite, and Daniel Roy. In defense of uniform convergence:
Generalization via derandomization with an application to interpolating predictors. In *International* Conference on Machine Learning, pages 7263–7272. PMLR, 2020.

[38] Jeffrey Negrea, Mahdi Haghifam, Gintare Karolina Dziugaite, Ashish Khisti, and Daniel M. Roy.

Information-theoretic generalization bounds for SGLD via data-dependent estimates, 2020.

[39] Paul Viallard, Pascal Germain, Amaury Habrard, and Emilie Morvant. A general framework for the disintegration of PAC-Bayesian bounds. *arXiv preprint arXiv:2102.08649*, 2021.

[40] Raef Bassily, Shay Moran, Ido Nachum, Jonathan Shafer, and Amir Yehudayoff. Learners that use little information. In *Algorithmic Learning Theory*, pages 25–55. PMLR, 2018.

[41] Ido Nachum, Jonathan Shafer, and Amir Yehudayoff. A direct sum result for the information complexity of learning. In *Conference On Learning Theory*, pages 1547–1568. PMLR, 2018.

[42] Fredrik Hellstrom, Giuseppe Durisi, Benjamin Guedj, Maxim Raginsky, et al. Generalization bounds: ¨
Perspectives from information theory and PAC-Bayes. Foundations and Trends® in Machine Learning, 18(1):1–223, 2025.

[43] Idan Attias, Gintare Karolina Dziugaite, Mahdi Haghifam, Roi Livni, and Daniel M. Roy. Information complexity of stochastic convex optimization: Applications to generalization, memorization, and tracing. In *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pages 2035–2068. PMLR, 21–27 Jul 2024.

[44] Hassan Hafez-Kolahi, Zeinab Golgooni, Shohreh Kasaei, and Mahdieh Soleymani. Conditioning and processing: Techniques to improve information-theoretic generalization bounds. *Advances in Neural* Information Processing Systems, 33:16457–16467, 2020.

[45] Ziqiao Wang and Yongyi Mao. Tighter information-theoretic generalization bounds from supersamples. In *Proceedings of the 40th International Conference on Machine Learning*, pages 36111–36137, 2023.

[46] Roi Livni. Information theoretic lower bounds for information theoretic upper bounds. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS '23, Red Hook, NY, USA, 2023. Curran Associates Inc.

[47] Mahdi Haghifam, Borja Rodr´ıguez-Galvez, Ragnar Thobaben, Mikael Skoglund, Daniel M Roy, and ´
Gintare Karolina Dziugaite. Limitations of information-theoretic generalization bounds for gradient descent methods in stochastic convex optimization. In *International Conference on Algorithmic* Learning Theory, pages 663–706. PMLR, 2023.

[48] Ziqiao Wang and Yongyi Mao. Sample-conditioned hypothesis stability sharpens informationtheoretic generalization bounds. *Advances in Neural Information Processing Systems*, 36:49513– 49541, 2023.

[49] Kimia Nadjahi, Kristjan Greenewald, Rickard Bruel Gabrielsson, and Justin Solomon. Slicing mutual ¨
information generalization bounds for neural networks. In International Conference on Machine Learning, pages 37213–37236. PMLR, 2024.

[50] Shai Shalev-Shwartz, Ohad Shamir, Nathan Srebro, and Karthik Sridharan. Stochastic convex optimization. In *COLT*, volume 2, number 4, page 5, 2009.

[51] Yuheng Bu, Weihao Gao, Shaofeng Zou, and Venugopal Veeravalli. Information-theoretic understanding of population risk improvement with model compression. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 3300–3307, 2020.

[52] Milad Sefidgaran, Amin Gohari, Gael Richard, and Umut Simsekli. Rate-distortion theoretic generalization bounds for stochastic learning algorithms. In *Conference on Learning Theory*, pages 4416–4463. PMLR, 2022.

[53] Peter Grunwald, Thomas Steinke, and Lydia Zakynthinou. PAC-Bayes, mac-bayes and conditional mutual information: Fast rate bounds that handle general vc classes. In *Conference on Learning* Theory, pages 2217–2247. PMLR, 2021.

[54] Milad Sefidgaran, Abdellatif Zaidi, and Piotr Krasnowski. Minimum description length and generalization guarantees for representation learning. In *Thirty-seventh Conference on Neural Information* Processing Systems (NeurIPS), 2023.

[55] Borja Rodr´ıguez-Galvez, Germ ´ an Bassi, Ragnar Thobaben, and Mikael Skoglund. On random subset ´
generalization error bounds and the stochastic gradient langevin dynamics algorithm. In *2020 IEEE* Information Theory Workshop (ITW), pages 1–5. IEEE, 2021.

[56] Ruida Zhou, Chao Tian, and Tie Liu. Individually conditional individual mutual information bound on generalization error. *IEEE Transactions on Information Theory*, 68(5):3304–3316, 2022.

[57] William B Johnson and Joram Lindenstrauss. Extensions of lipschitz mappings into a hilbert space 26. *Contemporary mathematics*, 26:28, 1984.

[58] Gavin Brown, Mark Bun, Vitaly Feldman, Adam Smith, and Kunal Talwar. When is memorization of irrelevant training data necessary for high-accuracy learning? In Proceedings of the 53rd annual ACM SIGACT symposium on theory of computing, pages 123–132, 2021.

[59] Jonathan Scarlett and Volkan Cevher. An introductory guide to fano's inequality with applications in statistical estimation. *arXiv preprint arXiv:1901.00555*, 2019.

[60] Nick Littlestone and Manfred Warmuth. Relating data compression and learnability. *Citeseer*, 1986.

[61] Cynthia Dwork, Adam Smith, Thomas Steinke, Jonathan Ullman, and Salil Vadhan. Robust traceability from trace amounts. In 2015 IEEE 56th Annual Symposium on Foundations of Computer Science, pages 650–669, 2015. doi: 10.1109/FOCS.2015.46.

[62] Michel Ledoux and Michel Talagrand. *Probability in Banach Spaces: isoperimetry and processes*.

Springer Science & Business Media, 2013.

[63] Ankit Pensia, Varun Jog, and Po-Ling Loh. Generalization error bounds for noisy, iterative algorithms. *2018 IEEE International Symposium on Information Theory (ISIT)*, pages 546–550, 2018.

[64] Mahdi Haghifam, Jeffrey Negrea, Ashish Khisti, Daniel M Roy, and Gintare Karolina Dziugaite.

Sharpened generalization bounds based on conditional mutual information and an application to noisy, iterative algorithms. *Advances in Neural Information Processing Systems*, 33:9925–9935, 2020.

[65] Borja Rodr´ıguez Galvez, Germ ´ an Bassi, Ragnar Thobaben, and Mikael Skoglund. On random sub- ´
set generalization error bounds and the stochastic gradient langevin dynamics algorithm. *CoRR*, abs/2010.10994, 2020.

[66] Hao Wang, Yizhe Huang, Rui Gao, and Flavio Calmon. Analyzing the generalization capability of SGLD using properties of gaussian channels. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, *Advances in Neural Information Processing Systems*, volume 34, pages 24222–24234. Curran Associates, Inc., 2021.

[67] Hao Wang, Rui Gao, and Flavio P Calmon. Generalization bounds for noisy iterative algorithms using properties of additive noise channels. *Journal of machine learning research*, 24(26):1–43, 2023.

[68] Sejun Park, Umut Simsekli, and Murat A Erdogdu. Generalization bounds for stochastic gradient descent via localized ε-covers. *Advances in Neural Information Processing Systems*, 35:2790–2802, 2022.

[69] Aymeric Dieuleveut, Alain Durmus, and Francis Bach. Bridging the gap between constant step size stochastic gradient descent and Markov chains, 2018.

[70] Leo Kozachkov, Patrick M Wensing, and Jean-Jacques Slotine. Generalization in supervised learning through riemannian contraction. *arXiv preprint arXiv:2201.06656*, 2022.

[71] Allan Grønlund, Lior Kamma, and Kasper Green Larsen. Near-tight margin-based generalization bounds for support vector machines. In Hal Daume III and Aarti Singh, editors, ´ Proceedings of the 37th International Conference on Machine Learning, volume 119 of *Proceedings of Machine* Learning Research, pages 3779–3788. PMLR, 13–18 Jul 2020.

[72] Jean Gallier. *Discrete mathematics*. Springer Science & Business Media, 2011. [73] Robert G Gallager. *Information theory and reliable communication*, volume 588. Springer, 1968. [74] Ziqiao Wang and Yongyi Mao. On the generalization of models trained with SGD: Informationtheoretic bounds and implications, 2021.

## Neurips Paper Checklist 1. **Claims**

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]
Justification: We proved several theoretical results showing the effectiveness of the projection and quantization technique and discussed it in detail. In particular, we showed how this can be used to resolve the recently raised concerns on the information-theoretic bounds. Guidelines:
- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes] Justification: We clearly stated the problem instances and classes for which we demonstrated that this approach results in good generalization bounds. We also stated all assumptions needed for each result.

## Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper.

- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

3. **Theory assumptions and proofs**
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [Yes]

Justification: In this paper, we stated all results rigorously, along with the assumptions used and detailed proofs in the supplements. The proofs are rigorous with enough details provided for the reader to follow.

Guidelines:
- The answer NA means that the paper does not include theoretical results. - All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

- All assumptions should be clearly stated or referenced in the statement of any theorems.

- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. **Experimental Result Reproducibility**

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [NA]
Justification: Our work is a theoretical paper with rigorously proven claims, and does not involve any experiment.

Guidelines:
- The answer NA means that the paper does not include experiments. - If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. **Open Access To Data And Code**

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA]
Justification: Our work does not involve any experiment. Guidelines:
- The answer NA means that paper does not include experiments requiring code.

- Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/
guides/CodeSubmissionPolicy) for more details.

- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

- At submission time, to preserve anonymity, the authors should release anonymized versions
(if applicable).

- Providing as much information as possible in supplemental material (appended to the paper)
is recommended, but including URLs to data and code is permitted.

## 6. **Experimental Setting/Details**

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA] Justification: Our work does not involve any experiment.

Guidelines:
- The answer NA means that the paper does not include experiments. - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. **Experiment Statistical Significance**

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [NA] Justification: Our work does not involve any experiment. Guidelines:
- The answer NA means that the paper does not include experiments. - The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).

- It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. **Experiments Compute Resources**

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA] Justification:Our work does not involve any experiment.

Guidelines:
- The answer NA means that the paper does not include experiments. - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. **Code Of Ethics**

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes]
Justification: Our work is a theoretical paper on learning theory and does not violate any code of ethics.

Guidelines:
- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. **Broader Impacts**

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA] Justification: Our work is a theoretical paper on learning theory and does not have any direct negative societal impact.

Guidelines:
- The answer NA means that there is no societal impact of the work performed. - If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses (e.g.,
disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. **Safeguards**

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA] Justification: Our work does not involve any experiment. Guidelines:
- The answer NA means that the paper poses no such risks. - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. **Licenses For Existing Assets**

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [NA] Justification: Our work does not involve any experiment.

Guidelines:
- The answer NA means that the paper does not use existing assets.

- The authors should cite the original paper that produced the code package or dataset. - The authors should state which version of the asset is used and, if possible, include a URL. - The name of the license (e.g., CC-BY 4.0) should be included for each asset.

- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. **New Assets**

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA]
Justification: Our work does not involve any experiment or any new asset. Guidelines:
- The answer NA means that the paper does not release new assets.

- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

- The paper should discuss whether and how consent was obtained from people whose asset is used.

- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. **Crowdsourcing And Research With Human Subjects**

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA]
Justification: Our work is a theoretical paper on learning theory. Guidelines:
- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

15. **Institutional review board (IRB) approvals or equivalent for research with human subjects**
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA]
Justification: Our work does not involve crowd sourcing nor any research with human subjects. Guidelines:
- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

16. **Declaration of LLM usage**