# Tighter CMI-Based Generalization Bounds via Stochastic Projection and Quantization

Milad Sefidgaran <sup>1</sup> , Kimia Nadjahi <sup>2</sup> , Abdellatif Zaidi <sup>1</sup>,<sup>3</sup>

Paris Research Center, Huawei Technologies France

<sup>2</sup> CNRS, ENS Paris, France <sup>3</sup> Universite Gustave Eiffel, France ´

milad.sefidgaran2@huawei.com, kimia.nadjahi@ens.fr, abdellatif.zaidi@univ-eiffel.fr

# Abstract

In this paper, we leverage stochastic projection and lossy compression to establish new conditional mutual information (CMI) bounds on the generalization error of statistical learning algorithms. It is shown that these bounds are generally tighter than the existing ones. In particular, we prove that for certain problem instances for which existing MI and CMI bounds were recently shown in Attias et al. [2024] and Livni [2023] to become vacuous or fail to describe the right generalization behavior, our bounds yield suitable generalization guarantees of the order of O(1/ √ n), where n is the size of the training dataset. Furthermore, we use our bounds to investigate the problem of data "memorization" raised in those works, and which asserts that there are learning problem instances for which any learning algorithm that has good prediction there exist distributions under which the algorithm must "memorize" a big fraction of the training dataset. We show that for every learning algorithm, there exists an auxiliary algorithm that does *not* memorize and which yields comparable generalization error for any data distribution. In part, this shows that memorization is not necessary for good generalization.

# 1 Introduction

One of the major problems in statistical learning theory consists in understanding what really drives the generalization error of learning algorithms. That is, what makes an algorithm trained on a given dataset continue to perform well on unseen data samples. Historically, this fundamental question has been studied independently in various lines of work, using seemingly unconnected tools. This includes VC-dimension theory [\[1\]](#page-10-0), Rademacher complexity approaches [\[2\]](#page-10-1), stability-based analysis [\[3\]](#page-10-2) and, more recently, intrinsic-dimension [\[4](#page-10-3)[–8\]](#page-10-4) and information-theoretic approaches [\[9](#page-10-5)[–21\]](#page-11-0). It is only until recently that the above various approaches were shown to be possibly unified [\[22,](#page-11-1) [23\]](#page-11-2) using a *variable-length* compressibility technique, which is rate-distortion-theoretic in nature.

In the context of statistical learning theory perhaps one can date back information-theoretic approaches to the PAC-Bayes bounds of McAllester [\[24,](#page-11-3) [25\]](#page-11-4), which were then followed by various extensions and ramifications [\[26–](#page-11-5)[39\]](#page-12-0). The mutual information (MI) bounds of [\[9\]](#page-10-5) and [\[10\]](#page-10-6) have the advantages to be relatively simpler comparatively and of offering somewhat clearer insights into the question of generalization. Roughly, such bounds suggest that a learning algorithm generalizes better as its output model reveals less information about the training data samples, where the amount of revealed information is measured in terms of the Shannon mutual information.

However, MI-based bounds are also known to sometimes take large (infinite) values and become vacuous, such as for continuous data and deterministic models. This shortcoming has been identified in a number of works, including [\[40,](#page-12-1) [41\]](#page-12-2). The issue was believed to be resolved by the introduction in [\[12\]](#page-10-7) of the important framework of conditional mutual information (CMI). The CMI setting introduces a "super-sample" construction in which an auxiliary "ghost sample" is used in conjunction with the training sample; and a sequence of Bernoulli random variables determines which data samples among the super-sample were used for the training. It is shown that a bound on the generalization error involves the mutual information between the Bernoulli random variables and the hypothesis (e.g., model parameters), conditionally given the super-sample [\[12,](#page-10-7) Theorem 2]. Because the entropy of Bernoulli random variables is bounded, the resulting bound is bounded. Many follow-up works have proposed extensions and improvements of the original CMI bounds, including using *randomized subset* and *individual sample* techniques, disintegration, and fast-rate variations in regimes in which the empirical risk is small – See [\[42\]](#page-12-3) for more on this.

CMI-type bounds were largely believed to be exempt from the aforementioned limitations of MI bounds until it was recently reported that examples can be constructed for which the standard[<sup>1</sup>](#page-0-0) CMI-based bound and its individual-sample variant fail [\[14,](#page-10-8) [43,](#page-12-4) [46\]](#page-12-5). The (counter-) examples of [\[46\]](#page-12-5) are in the context of Stochastic Convex Optimization (SCO) problems; and those of [\[43\]](#page-12-4) involve carefully constructed Convex-Lipschitz-bounded (CLB) and Convex-set-Strongly convex-Lipschitz (CSL) instance problems. These limitations were sometimes extrapolated to the extent of even questioning the utility of informationtheoretic bounds for the analysis of the generalization error of statistical learning algorithms more generally [\[47\]](#page-12-6). In this context, we mention [\[23,](#page-11-2) Appendix A] in which it was shown that, when applied to the counter-example of [\[47\]](#page-12-6), a lossy version of MI bounds yields generalization bounds that are of order O(1/n), instead of Ω(1) in the case of standard (lossless) MI bounds.[<sup>2</sup>](#page-0-0) The idea of lossy compression was also used in [\[49\]](#page-12-7).

In this paper, essentially, we show that the aforementioned limitations are in fact *not* inherent to the CMI framework; and, actually, the CMI framework can be adjusted slightly by the incorporation of a suitable stochastic projection and a suitable lossy compression to cope with those issues. Also, leveraging the utility of CMI and membership inference to study the problem of memorization and its relationship to generalization in machine learning, we use our results to revisit the necessity of memorization for SCO problems claimed in [\[43\]](#page-12-4). We show that memorization is *not* necessary for good generalization; and, as such, the result contributes to a better understanding of what role memorization plays in machine learning, a problem which is yet to be fully understood. Specifically, our contributions are as follows.

- We introduce stochastic projection in conjunction with lossy compression in the CMI framework, and we use them to establish a new CMI-based bound that is generally tighter than the CMI bounds of [\[12\]](#page-10-7).
- We show that, in sharp contrast with classic CMI-based bounds which fail when applied to the aforementioned CLB, CSL and SCO problem instances of [\[43,](#page-12-4) [46\]](#page-12-5) and may not even decay with the number of training samples, our new CMI bound yields meaningful results and decays with the number of training samples as O(1/ √ n).
- By applying them to generalized linear stochastic (non-convex) optimization problems, in the appendices we demonstrate that our bounds remain non-vacuous even beyond the convex case previously studied in [\[50\]](#page-12-8). The generalization is shown to come at the expense of a slower decay with n in our case; namely, O(1/ √<sup>4</sup> <sup>n</sup>) instead of <sup>O</sup>(1/ √
  - n) if the functions are convex as in [\[50\]](#page-12-8).
- We leverage the key ingredients of stochastic projection and lossy compression in the framework of CMI to study the "memorization" issue identified and studied in [\[43\]](#page-12-4). Specifically, [\[43\]](#page-12-4) has demonstrated that, for a given problem instance and every ε-learner algorithm, there exists a data distribution under which the algorithm "memorizes" the training samples. We show that for any learning algorithm A that memorizes the training data, one can find (via stochastic projection and lossy compression) an alternate learning algorithm A˜ with comparable generalization error and that does *not* memorize the training data for any data distribution. In part, this means that memorization is *not* necessary for good generalization in SCO.
- In the appendices, we use our general bound to study the generalization error of subspace training algorithms. Specifically, we investigate the setting in which the training is performed using SGD or SGLD; and we derive new bounds based on the differential entropy of Gaussian mixture distributions. This entropy depends on the gradient difference for the training and test datasets, the noise power, the learning rate, and the uncertainty of the index of the training dataset within the super-dataset.

<sup>1</sup>The authors of [\[43\]](#page-12-4) do not evaluate the performance of variants of CMI such as chained CMI [\[44\]](#page-12-9), evaluated CMI and f-CMI [\[20,](#page-11-6) [21,](#page-11-0) [45\]](#page-12-10) on their counter-example.

<sup>2</sup>The counterexample of [\[47\]](#page-12-6) has also been addressed by Wang and Mao [\[48\]](#page-12-11) using a different technique called "Sample-Conditioned Hypothesis Stability".

#### 2 Notation and Background

Let Z be some random variable with unknown distribution µ and taking values in some alphabet Z. Let S<sup>n</sup> ≜ (Z1, . . . , Zn) ∈ Z<sup>n</sup> be a set of n data samples drawn uniformly from the distribution µ, *i.e.,* S<sup>n</sup> ∼ PS<sup>n</sup> = µ ⊗n . In the framework of statistical learning, a (possibly) stochastic learning algorithm A: Z <sup>n</sup> → W takes the training dataset S<sup>n</sup> as input and returns a hypothesis W ∈ W ⊆ <sup>R</sup> <sup>D</sup>. We assume that A is *randomized*, in the sense that its output W ≜ A(Sn) is a random variable distributed according to PW|S<sup>n</sup> . We denote the distribution induced on (Sn, W) as PSn,W = PW|S<sup>n</sup> ⊗PS<sup>n</sup> = PW|S<sup>n</sup> ⊗µ ⊗n .

For a given function ℓ : Z × W → R, the loss incurred by using a hypothesis w ∈ W for a sample z is evaluated as ℓ(z, w). A statistical learning algorithm seeks to find a hypothesis w whose *population risk* R(w) ≜ <sup>E</sup>Z∼µ[ℓ(Z, w)] is minimal. However, since the data distribution µ is unknown, direct computation of the population risk R(w) is not possible. Instead, one resorts to minimizing the *empirical risk* Rb(sn, w) <sup>≜</sup> 1 n P<sup>n</sup> <sup>i</sup>=1 ℓ(zi, w) or a regularized version of it. Throughout, if s<sup>n</sup> is known from the context, we will use the shorthand notation Rbn(w) ≡ Rb(sn, w).

The *generalization error* induced by a specific choice of hypothesis w ∈ W and dataset s<sup>n</sup> is evaluated as

$$\text{gen}(s_n, w) \triangleq \mathcal{R}(w) - \widehat{\mathcal{R}}_n(w);$$

and the expected *generalization error* of the learning algorithm A is obtained by taking the expectation over all possible choices of (sn, w), as

$$\text{gen}(\mu, \mathcal{A}) \triangleq \mathbb{E}_{P_{S_n, W}}[\text{gen}(S_n, W)] = \mathbb{E}_{P_{S_n, W}}[\mathcal{R}(W) - \hat{\mathcal{R}}_n(W)].$$

#### 2.1 Conditional Mutual Information Framework

Let S˜ ∈ Z<sup>n</sup>×<sup>2</sup> be a super-sample composed of 2n data points Zi,j that are drawn uniformly from the distribution µ, where j ∈ {0, 1} and i ∈ [n]. Also, let J = (J1, . . . , Jn) ∈ {0, 1} n be a vector of n independent Bernoulli(1/2) random variables, all drawn independently from S˜. Let S˜ <sup>J</sup> = {Z1,J<sup>1</sup> , Z2,J<sup>2</sup> , . . . , Zn,J<sup>n</sup> }. In what follows, <sup>S</sup>˜ <sup>J</sup> plays the role of the training dataset Sn, S˜ \ S˜ J plays the role of a test or "ghost" dataset S ′ <sup>n</sup> and S˜ is a shuffled version of the union of the two. For an algorithm A : Z <sup>n</sup> → W, its CMI with respect to the data distribution µ is defined as

$$\text{CMI}(\mu, \mathcal{A}) \triangleq \text{I}(\mathcal{A}(\tilde{\mathbf{S}}_J); \mathbf{J}|\tilde{\mathbf{S}}) .$$

The CMI captures the information that the output hypothesis of the algorithm A trained on S˜ <sup>J</sup> provides about the membership vector J given the super-sample S˜. Equivalently, the CMI measures the extent to which the training and test datasets are distinguishable given the shuffled version of the union of the two, as well as the trained model. In its simplest form, it is shown in [\[12\]](#page-10-7) that the generalization error of an algorithm for a bounded loss in the range [0, 1] can be upper-bounded as

$$\text{gen}(\mu, \mathcal{A}) \leq \sqrt{\frac{2}{n}} \text{CMI}(\mu, \mathcal{A}).$$

Furthermore, for a Convex-Lipschitz-Bounded (CLB) whose formal definition will follow, the generalization error of A was shown in [\[47\]](#page-12-6) to be upper-bounded as

$$\text{gen}(\mu, \mathcal{A}) \leq LR \sqrt{\frac{8}{n} \text{CMI}(\mu, \mathcal{A})}. \quad (1)$$

Definition 1 (SCO Problem). *A stochastic convex optimization (SCO) problem is a triple* (W, Z, ℓ)*, where* W ∈ R <sup>D</sup> *is a convex set and* ℓ(z, ·): W → <sup>R</sup> *is a convex function for every* z ∈ Z*.*

Definition 2 (Convex-Lipschitz-Bounded (CLB)). *An SCO problem is called CLB if i) for every* w ∈ W*,* ∥w∥ ≤ R*, and ii) the loss function is convex and* L*-Lipschitz,* i.e., ∀z ∈ Z*,* ∀w1, w<sup>2</sup> ∈ W : |ℓ(z, w1) − ℓ(z, w2)| ≤ L∥w<sup>2</sup> − w1∥*. We denote this subclass of SCO problems by* CL,R*.*

# 3 New CMI-based bounds via stochastic projection and lossy compression

While the CMI-based bounds are known to be generally tighter than the corresponding MI ones and even tight in some settings [\[12,](#page-10-7) [14\]](#page-10-8), they can become vacuous in some cases. This includes the Stochastic Convex Optimization (SCO) examples constructed in the recent works [\[43,](#page-12-4) [46\]](#page-12-5), which we will discuss in more detail in Section [4.](#page-4-0) For these (counter-)examples, it was shown in [\[43,](#page-12-4) [46\]](#page-12-5) that CMI-type bounds do not vanish, so they fail to accurately describe the generalization error. In this section, we show that such limitations are *not* inherent to the CMI framework. In fact, by combining *stochastic projection* with *lossy compression* (analogously to [\[49\]](#page-12-7), which addressed the MI case), we derive new CMI-based bounds that *do not* suffer from such limitations. For instance, when applied to the SCO examples of [\[43\]](#page-12-4), we show in Section [4](#page-4-0) that our new bounds resolve the limitations of other known CMI-based bounds as identified therein. These bounds are also shown in the appendices to apply to the analysis of the generalization error for subspace training algorithms trained with SGD or SGLD.

Our new bounds involve two main ingredients, *stochastic projection* and *lossy compression*.

Stochastic projection. Let Θ ∈ R D×d be a random matrix with entries distributed according to some joint distribution PΘ, chosen independently of S˜, In our approach, similar to [\[49\]](#page-12-7), instead of considering the hypothesis W ∈ W ⊆ R <sup>D</sup> which lies in a D-dimensional space, we consider its *projection* Θ <sup>⊤</sup>W ∈ <sup>R</sup> d onto a smaller d-dimensional space, with d ≪ D.

Lossy Compression. Let ϵ ∈ <sup>R</sup> be given. An ϵ-lossy algorithm is a (possibly) stochastic map Aˆ: Z <sup>n</sup> × R <sup>D</sup>×<sup>d</sup> → Wˆ that maps a pair (Sn, Θ) to a compressed hypothesis or model Wˆ ∈ W ⊆ ˆ <sup>R</sup> d generated according to some conditional kernel PW<sup>ˆ</sup> <sup>|</sup>Sn,<sup>Θ</sup> that satisfies

$$\mathbb{E}_{P_{S_n, W} P_\Theta P_{\hat{W}|S_n, \Theta}} \left[ \text{gen}(S_n, W) - \text{gen}(S_n, \Theta \hat{W}) \right] \leq \epsilon.$$

This constraint guarantees that, when projected back onto the original hypothesis space of dimension D, the compressed model Wˆ has an average generalization error which is within at most ϵ from that of the original model W. In a sense, one works with a compressed model Wˆ which lies in a much smaller dimension space, but with the guarantee that this causes almost no increase in the generalization error. In effect, the *auxiliary* projected-back model ΘWˆ substitutes the original model W.

The concept of a lossy algorithm, also referred to as a "surrogate" or "compressed" algorithm, was introduced in [\[37,](#page-11-7) [51,](#page-12-12) [52\]](#page-12-13) and shown therein to be key to obtaining tighter, non-vacuous, generalization bounds. In this paper, we consider a particular lossy algorithm that involves a suitable stochastic projection followed by quantization. Specifically, we constrain the general conditional PW<sup>ˆ</sup> <sup>|</sup>Sn,<sup>Θ</sup> to take the specific form PW<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> , where W = A(Sn). Formally, one imposes the Markov chain (Sn, Θ, W)−Θ <sup>⊤</sup>W −Wˆ or equivalently <sup>P</sup>W<sup>ˆ</sup> <sup>|</sup>Sn,Θ,W <sup>=</sup> <sup>P</sup>W<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> . In other words, we let <sup>A</sup>ˆ(Sn, Θ) = <sup>A</sup>˜(Θ<sup>⊤</sup>A(Sn)), where A˜: <sup>R</sup> <sup>d</sup> <sup>→</sup> <sup>W</sup><sup>ˆ</sup> is defined via the Markov kernel <sup>P</sup>W<sup>ˆ</sup> <sup>|</sup>Θ⊤A(Sn) .

Our generalization bounds that will follow are expressed in terms of *disintegrated* CMI, defined as follows. Let a super-sample S˜ and a stochastic projection matrix Θ be given. The *disintegrated* CMI of an algorithm Aˆ: Z <sup>n</sup> → Wˆ is defined as

$$\text{CMI}^\Theta(\tilde{\mathbf{S}}, \hat{\mathcal{A}}) \triangleq I^{\tilde{\mathbf{S}}, \Theta}(\hat{\mathcal{A}}(\tilde{\mathbf{S}}_{\mathbf{J}}, \Theta); \mathbf{J}) ,$$

where Aˆ(S˜ <sup>J</sup>, Θ) = A˜(Θ<sup>⊤</sup>A(S˜ <sup>J</sup>)) = Wˆ and I <sup>S</sup>˜,<sup>Θ</sup>(Aˆ(S˜ <sup>J</sup>, Θ); J) is the CMI given an instance of S˜ and Θ, computed using the joint distribution <sup>P</sup><sup>J</sup> ⊗ <sup>P</sup>W|S˜<sup>J</sup> <sup>⊗</sup> <sup>P</sup>W<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> , with <sup>P</sup><sup>J</sup> <sup>=</sup> Bern(1/2)<sup>⊗</sup><sup>n</sup> .

The next theorem states our main generalization bound and is proved in Appendix [E.](#page-29-0)

Theorem 1. *Let a learning algorithm* A: Z <sup>n</sup> → W *where* W ⊆ <sup>R</sup> <sup>D</sup> *be given. Then, for every* ϵ ∈ <sup>R</sup>*, every* d ∈ <sup>N</sup>*, and every* projected model quantization *set* W ⊆ ˆ <sup>R</sup> d *, we have*

$$\text{gen}(\mu, \mathcal{A}) \leq \inf_{P_{\hat{W}|\Theta^\top W}} \inf_{P_\Theta} \mathbb{E}_{P_{\tilde{\mathbf{S}}} P_\Theta} \left[ \sqrt{\frac{2\Delta \ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta)}{n} \text{CMI}^\Theta(\tilde{\mathbf{S}}, \hat{\mathcal{A}})} \right] + \epsilon, \quad (2)$$

*where* Wˆ ∈ Wˆ *,* Θ ∈ <sup>R</sup> D×d *, the infima are over all arbitrary choices of Markov kernel* PW<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> *and distribution* P<sup>Θ</sup> *that satisfy the following distortion criterion:*

$$\mathbb{E}_{P_{S_n, W} P_{\Theta} P_{\hat{W}|\Theta^{\top} W}} \left[ \text{gen}(S_n, W) - \text{gen}(S_n, \Theta \hat{W}) \right] \leq \epsilon, \quad (3)$$

*and the term* ∆ℓwˆ(S˜, Θ) *is given by*

$$\Delta \ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta) := \mathbb{E}_{P_{W|\tilde{\mathbf{S}}} P_{\hat{W}|\Theta^\top W}} \left[ \frac{1}{n} \sum_{i \in [n]} (\ell(Z_{i,0}, \Theta \hat{W}) - \ell(Z_{i,1}, \Theta \hat{W}))^2 \right]. \quad (4)$$

Observe that PW|S˜ = <sup>E</sup>P<sup>J</sup> [PW|S˜<sup>J</sup> ]. Also, if ℓ(·, ·) ∈ [0, C] for some non-negative constant C ∈ <sup>R</sup>+, then it is easy to see that the term ∆ℓwˆ(S˜, Θ) is bounded from the above as ∆ℓwˆ(S˜, Θ) ≤ C 2 .

The result of Theorem [1](#page-3-0) essentially means that the generalization error of the original model is upper bounded by a term that depends on the CMI of the auxiliary model Wˆ plus an additional distortion term that quantifies the generalization gap between the auxiliary and original models. The rationale is that, although the (worst-case) CMI term still depends on the dimension d after stochastic projection, this dimension corresponds to a subspace of the original hypothesis space and can be chosen arbitrarily small in order to guarantee that the bound vanishes with n. Also, the term in left-hand-side (LHS) of equation [3](#page-3-1) represents the average distortion (measured by the difference of induced generalization errors) between the original model and the one obtained after projecting back the auxiliary compressed model onto the original hypothesis space. The analysis of this term may seem non-easy; but as visible from the proof, it is not. This is because, defined as a difference term, its analysis does not necessitate accounting for statistical dependencies between S and W. Instead, one only needs to account for the effect of the following sources of randomness: i) the stochastic projection matrix, ii) the quantization noise, and iii) discrepancies between the empirical measure of S and the true unknown distribution µ. As shown in the proofs, the analysis of the distortion term involves the use of classic concentration inequalities. Furthermore, the construction of Wˆ allows us to consider the worst-case bound for the CMI-terms of the RHS of equation [2](#page-3-2) without losing the order-wise optimality in certain cases.

We close this section by noting that it is well known that CMI-type bounds can be improved by application of suitable techniques such as *random-subset* or *individual sample* techniques or in order to get fast rates O(1/n) for small empirical risk regimes, see, e.g., [\[20,](#page-11-6) [53,](#page-12-14) [54\]](#page-12-15). These same techniques can be applied straightforwardly to our bound of Theorem [1](#page-3-0) to get improved ones. For the sake of brevity, we do not elaborate on this here; and we refer the reader to the supplements where a single-datum version of Theorem [1](#page-3-0) is provided.

# 4 Application to resolving recently raised limitations of classic CMI bounds

Prior works [\[43,](#page-12-4) [46\]](#page-12-5) have recently reported carefully constructed counter-example learning problems and have shown that classic MI-based and CMI-based bounds fail to provide meaningful results when applied to them. In this section, we show that the careful addition of our stochastic projection along with our lossy compression resolves those issues, in the sense that the resulting new bound (our Theorem [1\)](#page-3-0), which is still of CMI-type, now yields meaningful results when applied to those counter-examples. In essence, the improvement is brought up by: (i) noticing that the aforementioned negative results for standard CMI-based generalization error bounds rely heavily on that the dimension of the hypothesis space grows fast with n (over-parameterized regime), e.g., as Ω(n 4 log n) in the considered counter-examples of [\[43\]](#page-12-4), which calls for suitable projection onto a smaller dimension space in which this does not hold, and (ii) properly accounting for the distortion induced in the generalization error after projection back to the original high dimensional space.

First, we recall briefly the counterexamples mentioned in [\[43\]](#page-12-4) and [\[46\]](#page-12-5); and, for each of them, we show how our bound of Theorem [1](#page-3-0) applies successfully to it. Recall the definitions of a stochastic convex optimization (SCO) problem and a Convex-Lipschitz-Bounded (CLB) SCO problem as given, respectively, in Definition [1](#page-2-0) and Definition [2.](#page-2-1)

Definition 3 (ε-learner for SCO). *Fix* ϵ > 0*. For a given SCO problem* (W, Z, ℓ)*,* A = {An}n≥<sup>1</sup> *is called an* ε*-learner algorithm with sample complexity* N : R × R → N *if the following holds: for every* δ ∈ (0, 1] *and* n ≥ N(ε, δ) *we have that for every* µ ∈ M1(Z)*, where* M1(Z) *denotes the set of probability measures on* Z*, with probability at least* 1 − δ *over* S<sup>n</sup> ∼ µ ⊗n *and internal randomness of* A*,*

$$\mathcal{R}(\mathcal{A}_n(S_n)) - \min_{w \in \mathcal{W}} \mathcal{R}(w) \leq \varepsilon. \quad (5)$$

#### 4.1 Counter-example of Attias et al. [2024] for CLB class

Denote by BD(ν) the D-dimensional ball of radius ν ∈ <sup>R</sup>+.

Definition 4 (Problem instance P (D) cvx ). *Let* L, R ∈ <sup>R</sup>+*,* Z ⊆ BD(1)*, and* W = BD(R)*. Define the loss function* ℓ : Z × W → R *as*

$$\ell_c(z, w) = -L\langle w, z \rangle.$$

*We denote this SCO problem instance as* P (D) cvx *. It is easy to see that this optimization problem belongs to the subclass* CL,R *of SCO problems as defined in Definition [2.](#page-2-1)*

For this (counter-) example learning problem, [\[43\]](#page-12-4) have shown that for every ε-learner there exists a data distribution for which the CMI bound of equation [1](#page-2-2) for the optimal sample complexity, which is Θ LR ε 2 as shown in [\[50\]](#page-12-8), scales just as Θ(LR). For instance, that CMI-bound on the generalization error does *not* decay with the size n of the training dataset!

Theorem 2 (CMI-accuracy tradeoff, [\[43,](#page-12-4) Theorems 4.1 and 5.2]). *Let* ε<sup>0</sup> ∈ (0, 1) *be a universal constant. Consider the above defined* P (D) cvx *problem instance with parameters* (L, R)*. Consider any* ϵ ≤ ϵ<sup>0</sup> *and for any algorithm* A = {An}n∈<sup>N</sup> *that* ε*-learns* P (D) cvx *with sample complexity* N(·, ·)*. Then, the following holds: i. For every* δ ≤ ε*,* n ≥ N(ε, δ)*, and* D = Ω n 4 log(n) *,* [3](#page-0-0) *there exists a set* Z ⊆ BD(1) *and a data distribution* <sup>µ</sup> ∈ M1(Z)*, denoted as* <sup>µ</sup>p<sup>∗</sup> *, such that* CMI(µ, <sup>A</sup>n) = Ω LR ε 2 *. ii. In particular, considering the optimal sample complexity* <sup>N</sup>(ε, δ) = Θ L2R<sup>2</sup> ε<sup>2</sup> *, the CMI generalization bound of equation [1](#page-2-2) equals* LRp 8CMI(µ, An)/N(ε, δ) = Θ(LR)*.*

For this example, it was further shown [\[43,](#page-12-4) Corollary 5.6] that application of the *individual sample* technique of [\[55,](#page-12-16) [56\]](#page-12-17) (which is traditionally used to avoid the unbounded-ness issue as instance of so called *randomized-subset* techniques wherein the linearity of the expectation operator is used to obtain an average bound for the loss on randomly chosen subsets of the training set rather than the loss averaged over the full training set) actually yields the very same bound order-wise; and, thus, it does not resolve the issue for this counter-example.

Furthermore, as shown in [\[43,](#page-12-4) Equation 1], the expectation of the LHS of equation [5](#page-4-1) can be bounded as

$$\mathbb{E}[\mathcal{R}(\mathcal{A}_n(S_n))] - \min_{w \in \mathcal{W}} \mathcal{R}(w) \leq LR \sqrt{\frac{8\text{CMI}(\mu, \mathcal{A}_n)}{n}} + \mathbb{E} \left[ \hat{\mathcal{R}}_n(\mathcal{A}_n(S_n)) - \min_{w \in \mathcal{W}} \hat{\mathcal{R}}_n(w) \right]. \quad (6)$$

Thus, while the LHS of this inequality is bounded from above by ε by assumption, its right-hand side (RHS) is Θ(LR) by Theorem [2.](#page-5-0) This means that the CMI bound of equation [1](#page-2-2) fails to describe well the excess error of the LHS. In [\[43\]](#page-12-4), this was even somewhat extrapolated to negatively answer the question about "*whether the excess error decomposition using CMI can accurately capture the worst-case excess error of optimal algorithms for SCOs*".

The above applies for any ε-learner of the problem instance P (D) cvx when Z = {±1/ √ D} <sup>D</sup> and µp<sup>∗</sup> (z) = Q<sup>D</sup> <sup>k</sup>=1 1+√ Dzkp k 2 , [4](#page-0-0) for all z = (z1, . . . , zD), where p <sup>∗</sup> = (p ∗ 1, . . . , p<sup>∗</sup> <sup>D</sup>) ∈ [−1, 1]<sup>D</sup>.

The next theorem shows that when applied to the aforementioned counter-example, our new CMI-bound of Theorem [1](#page-3-0) does *not* suffer from those shortcomings. Also, this holds true for: (i) arbitrary values of the dimension D ∈ N including n-dependent ones, (ii) arbitrary learning algorithms (including the ε-learners of P (D) cvx ), (iii) arbitrary choices of Z ⊆ BD(1) and (iv) arbitrary data distributions µ.

Theorem 3. *For every learning algorithm* A: Z <sup>n</sup> → W *of the instance* P (D) cvx *defined as in Definition [4,](#page-4-2) the generalization bound of Theorem [1](#page-3-0) yields*

$$\text{gen}(\mu, \mathcal{A}) \leq \frac{8LR}{\sqrt{n}}.$$

*In particular, setting* <sup>N</sup>(ε, δ) = Θ L2R<sup>2</sup> ε<sup>2</sup> *for* ε*-learner algorithms we get*

gen(
$$\mu, \mathcal{A}$$
) =  $\mathcal{O}(\varepsilon)$ .

The proof of Theorem [3](#page-5-1) is deferred to Appendix [F.2.](#page-32-0)

Some remarks are in order. First, while when applied to the studied counter-example the CMI bound of equation [1](#page-2-2) yields a bound of the order Θ(LR), i.e., one that does *not* decay with n, our new CMIbased bound of Theorem [1](#page-3-0) yields one that decays with <sup>n</sup> as <sup>O</sup>(LR/√ n). Second, when specialized to the

<sup>3</sup>The arXiv version of [\[43\]](#page-12-4) requires a smaller increase of D with n; namely, D = Ω n 2 log(n) . Here, we consider values of D that are mentioned in the published PMLR version of the document, i.e., D = Ω n 4 log(n) ; but the approach and results that will follow also hold for D = Ω n 2 log(n) .

<sup>4</sup> In the construction of [\[43\]](#page-12-4), by changing n, the data distribution changes, but, for better readability, we drop such dependence in the notation.

case of ε-learner algorithms and considering the sample complexity Θ LR ε 2 , we get a bound on the generalization error of the order O (ε). Using this bound, we can write

$$\mathbb{E}_{P_{S_n,W}} [\mathcal{R}(\mathcal{A}_n(S_n))] - \min_{w \in \mathcal{W}} \mathcal{R}(w) \leq \mathcal{O}(\varepsilon) + \mathbb{E}_{P_{S_n,W}} \left[ \hat{\mathcal{R}}_n(\mathcal{A}_n(S_n)) - \min_{w \in \mathcal{W}} \hat{\mathcal{R}}_n(w) \right]. \quad (7)$$

Contrasting with equation [6](#page-4-3) and noticing that if the second term of the summation of the RHS of equation [7](#page-5-2) (optimization error) is small then both sides of equation [7](#page-5-2) are O(ϵ), it is clear that now the excess error decomposition using our new CMI-based bound can accurately capture the worst-case excess error. Third, as it can be seen from the proof, stochastic projection onto a one-dimensional space, i.e., d = 1, is sufficient to get the result of Theorem [3.](#page-5-1) In essence, this is the main reason why, in sharp contrast with projection- and lossy-compression-free CMI-bounds, ours of Theorem [1](#page-3-0) does *not* become vacuous. That is, one can reduce the effective dimension of the model for the studied example even if the original dimension D is allowed to grow with n as Ω(n 4 log(n)) as judiciously chosen in[\[43\]](#page-12-4) for the purpose of making classic CMI-based bounds fail. Furthermore, it is worth noting that, for this problem, the projection is performed using the famous Johnson-Lindenstrauss [\[57\]](#page-12-18) dimension reduction algorithm. Since this dimension reduction technique is "lossy", controlling the induced distortion is critical. To do so, we introduce an additional lossy compression step by adding independent noise in the lower-dimensional space. This approach is reminiscent of lossy source coding and allows to obtain possibly tighter bounds on the quantized, projected model. Finally, we mention that for bigger class problem instances or for the memorization problem of Section [5,](#page-7-0) projection onto one-dimensional spaces may not be enough to get the desired order <sup>O</sup>(LR/√ n). In Appendix [B,](#page-22-0) it will be shown that for generalized linear stochastic optimization problems, one may need <sup>d</sup> = Θ(√ n). Similarly, in Section [5](#page-7-0) and Appendix [C,](#page-23-0) projections with d = n 2r−1 , r < 1 and d = Θ(log n) are used.

#### 4.2 Counter-example of Attias et al. [2024] for CSL class

The question of whether classic CMI-bounds and individual-sample versions thereof may still fail if one considers more structured subclasses of SCO problems was raised (and answered positively!) in Attias et al. [\[43\]](#page-12-4). For convenience, we recall the following two definitions.

Definition 5 (Convex set-Strongly Convex-Lipschitz (CSL)). *An SCO problem is called CSL if i) the loss function is* L*-Lipschitz, and ii) the loss function is* λ*-strongly convex,* i.e., ∀z ∈ Z*,* ∀w1, w<sup>2</sup> ∈ W : ℓ(z, w2) ≥ ℓ(z, w1) + ⟨∂ℓ(z, w1), w<sup>2</sup> − w1⟩ + λ 2 ∥w<sup>2</sup> − w1∥ 2 *, where* ∂ℓ(z, w1) *is the subgradient of* ℓ(z, ·) *at* w1*. We denote this subclass by* CL,λ*.*

Definition 6 (Problem instance P (D) scvx). *Let* λ, R ∈ <sup>R</sup>+*,* Z ⊆ BD(1)*, and* W = BD(R)*. Define the loss function* ℓ : Z × W → <sup>R</sup> *as* ℓsc(z, w) = −Lc⟨w, z⟩ + λ 2 ∥w∥ 2 *. We denote this SCO problem as* P (D) scvx*, which belongs to* CL,λ*, with* L = L<sup>c</sup> + λR*.*

Setting λ = L<sup>c</sup> = R = 1, D = Ω(n 4 log(n)), δ = O(1/n<sup>2</sup> ), Z = {±1/ √ D} <sup>D</sup> and for a particular data distribution that is carefully chosen therein (not reproduced here for brevity), [\[43,](#page-12-4) Theorem 4.2] states that for any learning algorithm that ε-learns the problem instance P (D) scvx,

$$\text{CMI}(\mu, \mathcal{A}_n) = \Omega\left(\frac{1}{\varepsilon}\right).$$

Moreover, the application of the individual-sample technique does not result in better decay of the bound order-wise [\[43,](#page-12-4) Corollary 5.7].

Noticing that (i) the loss ℓsc(z, w) = −Lc⟨w, z⟩ + λ 2 ∥w∥ 2 considered in Definition [6](#page-6-0) differs from that ℓsc(z, w) = −L⟨w, z⟩ of Definition [4](#page-4-2) essentially through the added squared magnitude of the model and (ii) that addition does not alter the generalization error of a given learning algorithm, then it is easy to see that Theorem [3](#page-5-1) also applies for the problem P (D) scvx at hand; and, in this case, it gives a bound of the order O(1/ √ n). This is stated in the next proposition, which is proved in Appendix [F.3.](#page-35-0)

Proposition 1. *For every learning algorithm* A: Z <sup>n</sup> → W *of the instance* P (D) scvx *defined as in Definition [6](#page-6-0) the generalization bound of Theorem [1](#page-3-0) yields*

$$\text{gen}(\mu, \mathcal{A}) \leq \frac{8L_c R}{\sqrt{n}}.$$

*In particular, choosing* L<sup>c</sup> = R = λ = 1 *and setting* N(ε, δ) = <sup>c</sup> ε *for some non-negative constant* c ∈ <sup>R</sup><sup>+</sup> *for the ERM algorithm (which is an* ε*-learner – see, e.g., [\[50,](#page-12-8) Theorem 6]), one gets* gen(µ, A) = O ( √ ε)*.*

#### 4.3 Counter-example of Livni [2023]

The counter-example of [\[46\]](#page-12-5) is the same as the problem instance of Definition [4,](#page-4-2) with the one difference that the loss function is taken to be the squared distance instead of the inner product, *i.e.,* ℓ(z, w) = −L∥w − x∥ 2 , for some non-negative constant L ∈ <sup>R</sup>+. Livni [\[46\]](#page-12-5) has shown that the MI bound of [\[11\]](#page-10-9) (which is a single-datum bound) fails and becomes vacuous when evaluated for this particular learning problem. However, since ℓ(z, w) = −L∥x∥ <sup>2</sup> − L∥w∥ <sup>2</sup> + 2L⟨w, x⟩ and noticing that the squared norm terms do not alter the generalization error relative to when computed for a loss function given by only the inner-product term, it follows that Theorem [3](#page-5-1) still applies and gives a bound of the order O(1/ √ n) for this problem instance. In addition, for the optimal sample complexity, the bound is O(ε). In essence, this means that unlike the MI bound of [\[11\]](#page-10-9), our new CMI-based bound of Theorem [1](#page-3-0) does not become vacuous when applied to the problem at hand.

In Appendix [B,](#page-22-0) we apply the bound of Theorem [1](#page-3-0) to a wider family of generalized linear stochastic optimization problems. In particular, we show that no counter-example could be found for which the bound of Theorem [1](#page-3-0) does not vanish, even if one considers the bigger class of generalized linear stochastic optimization problems in place of the SCO class problems of [\[43\]](#page-12-4).

# 5 Memorization

Loosely speaking, a learning algorithm is said to "memorize" if by only observing its output model, an adversary can correctly guess elements of the training data among a given super-sample. For the CLB and CSL subclasses of problems studied in Section [4,](#page-4-0) Attias et al. [\[43\]](#page-12-4) showed that there are problem instances for which, for any ε-learner algorithm, there exists a data distribution under which the learning algorithm "memorizes" most of the training data. This is obtained by designing an adversary capable of identifying a significant fraction of the training samples.

In this section, we show that given a learning algorithm A that memorizes the training samples, one can find (via stochastic projection and lossy compression) an alternate learning algorithm A˜ with comparable generalization error and that does *not* memorize the training data.[<sup>5</sup>](#page-0-0)

Definition 7 (Recall Game [\[43,](#page-12-4) Definition 4.3]). *Given* A = {An}n≥1*, let* Q: <sup>R</sup> <sup>D</sup> × Z × M1(Z) → {0, 1} *be an adversary for the following game. For* i ∈ [n]*, given a fresh data point* Z ′ <sup>i</sup> ∼ µ *independent of* (Zi, W)*, let* Zi,<sup>1</sup> = Z<sup>i</sup> *and* Zi,<sup>0</sup> = Z ′ i *. Then, the adversary is given* Zi,K<sup>i</sup> *, where* K<sup>i</sup> ∼ *Bern*(1/2) *is independent of other random variables. The adversary declares* <sup>K</sup>ˆ<sup>i</sup> <sup>≜</sup> Q(W, Zi,K<sup>i</sup> , µ) *as its guess of* Ki*.*

The game consists of n rounds. At each round i ∈ [n], a pair (Zi,0, Zi,1) is considered and the adversary makes two independent guesses: one for the sample Zi,0, the other for Zi,1.

Definition 8 (Soundness and recall [\[43,](#page-12-4) Definition 4.4]). *Consider the setup of Definition [7.](#page-7-1) Assume that the adversary plays the game in* n *rounds. For every round* i ∈ [n]*, the adversary plays two times, independently of each other, using respectively* (W, Zi,0, µ) *and* (W, Zi,1, µ) *as input. Then, for a given* ξ ∈ [0, 1]*, the adversary is said to be* ξ*-sound if* <sup>P</sup> (∃ i ∈ [n]: Q(W, Zi,0, µ) = 1) ≤ ξ*. Also, the adversary certifies the recall of* m *samples with probability* q ∈ [0, 1] *if* <sup>P</sup> P <sup>i</sup>∈[n] Q(W, Zi,1, µ) ≥ m ≥ q*. If both conditions are met, we say that the adversary* (m, q, ξ)*-traces the data.*

Clearly, the concept of (m, q, ξ)-*tracing* the data by an adversary is most interesting for values of (m, q, ξ) that are such that: ξ is small (i.e., the adversary makes accurate predictions), m is large and q is nonnegligible (i.e., the adversary can recall a significant part of the training data). As Lemma [1,](#page-23-1) which is stated in Appendix [C.1,](#page-23-2) asserts, certain values of (m, q, ξ) can be attained even by a "dummy" adversary that makes guesses without even looking at the given data sample.

For the problem instance P (D) cvx , Attias et al. [\[43\]](#page-12-4) have shown that, for every ϵ-learner algorithm, there exist a distribution and an adversary that is capable of identifying a significant portion of the training data.

Theorem 4 ([\[43,](#page-12-4) Theorem 4.5]). *Consider the* P (D) cvx *problem instance of Definition [4](#page-4-2) with* L = R = 1*. Fix arbitrary* ξ ∈ (0, 1] *and let* Z = {±1/ √ D} <sup>D</sup>*. Let* ε<sup>0</sup> ∈ (0, 1) *be a universal constant. Let* ε > 0 *such that* ε < ε0*,* δ < ε*. Then, given any* ε*-learner algorithm* A *with sample complexity* N(ε, δ) = Θ(log(1/δ)/ε<sup>2</sup> )*, there exist a data distribution* µp<sup>∗</sup> *and an adversary such that for* n = N(ε, δ) *and* D = Ω(n 4 log(n/ξ))*, the adversary* Ω(1/ε<sup>2</sup> ), 1/3, ξ *-traces the data.*

<sup>5</sup>The memorization problem has also been studied in [\[58\]](#page-13-0) via some examples in which the data distribution µ is not fixed and comes from a meta-distribution, i.e. µ ∼ Pµ. Instead of using the recall game, [\[58\]](#page-13-0) measured the amount of memorization by I(S; W|µ).

A key implication of Theorem [4](#page-7-2) is that, for some fixed q > 0, the result holds even when ξ ∈ (0, 1] is arbitrarily small and m = Ω(n) (by choosing ε = O(1/ √ n)). In other words, for the considered class of problems Pcvx with data drawn from µp<sup>∗</sup> , the constructed adversary can provably trace an arbitrarily large part of the training dataset.

We show that the stochastic projection and lossy compression techniques used in the CMI framework can partially mitigate this memorization issue, in a sense that will be made precise in Theorem [8.](#page-24-0) To this end, we first establish a general result on memorization.

Theorem 5. *Consider any learning algorithm* A = {An}n≥<sup>1</sup> *such that* CMI(µ, An) = o(n)*. Then, for any adversary for this learning algorithm that* (m, q, ξ)*-traces the data, the following holds: i)* m = o(n) *or* ξ ≥ q*, ii) if, for some* α ∈ (0, 1) *and* n<sup>0</sup> ∈ N ∗ *,* m ≥ αn *for every* n ≥ n0*, then for any* ϵ ∈ (0, α) *it holds that:* P P <sup>i</sup>∈[n] <sup>Q</sup>(W, Zi,0, µ) <sup>≥</sup> <sup>m</sup>′ ≥ (α − ϵ)q*, where* m′ = ϵ 1/q+ϵ−α n − o(n) = Ω(n)*.*

Theorem [5,](#page-8-0) whose proof is provided in Appendix [G.1,](#page-40-0) applies to *any* learning problems. In particular, it is not limited to P (D) cvx or the CLB subclass. The argument relies on Fano's inequality for approximate recovery [\[59,](#page-13-1) Theorem 2]. We construct a suitable estimator of the index set J based on the adversary's guesses, and we show that if this estimator can correctly recover a fraction c > <sup>1</sup> 2 of the membership indices J, then CMI(µ, An) = Θ(n).

Theorem [5](#page-8-0) *i)* means that if the CMI of a learning algorithm is of order o(n), then any adversary that recalls a non-negligible fraction of the training dataset with some probability q (*i.e.,* , m = Θ(n)) is q-sound at best. This means that, in this regime, no adversary can do better than a dummy one that makes random guesses independently of the data (See Lemma [1](#page-23-1) in Appendix [C.1](#page-23-2) for what is attainable by a dummy adversary). Theorem [5](#page-8-0) *ii)* means that if an adversary recalls Ω(n) training samples with some probability, then it must also incorrectly guess the membership of Ω(n) test samples with some non-negligible probability.

Next, we use the result of Theorem [5](#page-8-0) for P (D) cvx to show that while the output model W of any ε-learner algorithm must memorize a significant fraction of the data (for some distribution) as asserted in Theorem [4](#page-7-2) the auxiliary model ΘWˆ (which is obtained through suitable stochastic projection and lossy compression), achieves comparable generalization error *without* memorizing the data!

Theorem 6. *Consider the* P (D) cvx *problem instance of Definition [4](#page-4-2) with* L = R = 1*. For every* r > 0*, every* Z ⊆ BD(1) *and every learning algorithm* A: Z <sup>n</sup> → <sup>R</sup> <sup>D</sup>*, there exists another (compressed) algorithm* A ∗ : Z <sup>n</sup> → <sup>R</sup> <sup>D</sup>*, defined as* A ∗ (Sn) ≜ ΘA˜(Θ<sup>⊤</sup>A(Sn)) = ΘWˆ *, where the projection matrix* Θ ∈ R D×d *,* d = 500r log(n)*, is distributed according to some distribution* P<sup>Θ</sup> *independent of* (Sn, W)*, such that for any data distribution* µ*, the following conditions are met simultaneously:*

*i) the generalization error of the auxiliary model* ΘWˆ *satisfies*

$$\left| \mathbb{E} P_{S_n, W} P_{\Theta} P_{W|\Theta^{\top} W} \left[ \text{gen}(S_n, W) - \text{gen}(S_n, \Theta \hat{W}) \right] \right| = \mathcal{O} \left( n^{-r} \right), \quad (8)$$

- *ii) if there exists an adversary that by having access to both* Θ *and* Wˆ *(and hence* ΘWˆ *)* (m, q, ξ) *traces the data, then it must be that: a)* m = o(n) *or* ξ ≥ q*, and b) if, for some* α ∈ (0, 1) *and* n<sup>0</sup> ∈ N ∗ *,* m ≥ αn *for every* n ≥ n0*, then for any* ϵ ∈ (0, α) *it holds that:* P P <sup>i</sup>∈[n] <sup>Q</sup>(ΘW , Z <sup>ˆ</sup> i,0, µ) <sup>≥</sup> <sup>m</sup>′ ≥ (α − ϵ)q*, where* m′ = ϵ 1/q+ϵ−α n − o(n) = Ω(n)*.*

Theorem [6,](#page-8-1) proved in Appendix [G.2,](#page-45-0) holds for Θ being stochastic and shared with the adversary. In essence, it asserts that for any algorithm A(S) = W, one can construct a suitable projected-quantized model Aˆ(S, Θ) = Wˆ from which no adversary would be able to trace the data, for any data distribution µ. It is appealing to contrast this result with that of [\[43,](#page-12-4) Theorem 4.5] on the necessity of memorization. Consider the SCO instance problem with O(1) convex-Lipschitz loss defined over the ball of radius one in R <sup>D</sup> considered in [\[43,](#page-12-4) Theorem 4.5] and let an ε-learner algorithm A with output model W and sample complexity N(ε, δ) = Θ(log(1/δ)/ε<sup>2</sup> ) with D = Ω(n 4 log(n/ξ)) be given. The result of [\[43,](#page-12-4) Theorem 4.5] states that there exists a data distribution for which the algorithm A must memorize a big fraction of the training data. Applied to this particular instance problem, Theorem [6](#page-8-1) asserts that if a random Θ is chosen and shared with the adversary then the auxiliary model ΘWˆ has the following guarantees: (i) for any data distribution, no adversary can trace the data, and (ii) on average over Θ the associated generalization error is arbitrarily close to that of the original model W. At first glance, this may seem to contradict the necessity of memorization stated in [\[43,](#page-12-4) Theorem 4.5]. It is important to note, however, that the auxiliary algorithm does not satisfy the conditions required in [\[43,](#page-12-4) Theorem 4.5]; and, so, the latter does not apply to ΘWˆ . In particular, while [\[43,](#page-12-4) Theorem 4.5] requires the model to be bounded, in our construction for every w we have <sup>E</sup>W , <sup>ˆ</sup> <sup>Θ</sup> -ΘWˆ ≈ w but <sup>E</sup>W , <sup>ˆ</sup> <sup>Θ</sup> h <sup>Θ</sup>W<sup>ˆ</sup> 2 i increases roughly as <sup>D</sup> d (see Lemma [2](#page-25-0) in Appendix [C.4.1\)](#page-25-1). As discussed after Lemma [2,](#page-25-0) this causes <sup>E</sup>W , <sup>ˆ</sup> <sup>Θ</sup> h <sup>Θ</sup>W<sup>ˆ</sup> 2 i to grow as Ω(n 3 ) when D = Ω(n 4 log(n/ξ)), i.e., it becomes arbitrarily large as n increases. Intuitively, this is what prevents an adversary from guessing correctly whether a sample has (or not) been used for training, and which makes some key proof steps of Attias et al. fail when applied to the auxiliary model ΘWˆ . These steps are discussed in detail in Appendix [C.4.2.](#page-25-2)

A somewhat weaker version of Theorem [6,](#page-8-1) which is stated in Theorem [8](#page-24-0) in Appendix [C.2,](#page-24-1) holds for the projection matrix Θ being *deterministic*. In a sense, it provides a stronger guarantee on the generalization error of the auxiliary model, in that the closeness to the performance of the original model holds now for the given Θ and not only in average over Θ as in Theorem [6.](#page-8-1) However, this comes at the expense of the auxiliary algorithm being dependent on the data distribution. A consequence of this is that the result does not preclude the existence of other distributions for which there would exist adversaries capable of tracing the data. Moreover, in Theorem [9](#page-24-2) in Appendix [C.3,](#page-24-3) we show that a similar result holds if one considers the closeness in terms of the population risk, instead of the generalization error.

Summarizing, neither of the results of Theorem [6](#page-8-1) and Theorem [8](#page-24-0) contradict those of [\[43\]](#page-12-4). In essence, they assert that for any learning algorithm A one can find an alternate auxiliary algorithm via stochastic projection combined with lossy compression for which no adversary would be able to trace the data; and, yet, the found auxiliary algorithm has generalization error that is arbitrarily close to that of the original model. Appendix [C.3](#page-24-3) extends this closeness to the population risk.

# 6 Implications and Concluding Remarks

#### Sample-compression schemes

Formally, a learning algorithm is a sample compression scheme of size k ∈ N if there exists a pair of mappings (ϕ, ψ) such that for all samples S = (Z1, . . . , Zn) of size n ≥ k, the map ϕ compresses the sample into a length-k sequence which the map ψ uses to reconstruct the output of the algorithm, i.e., A(S) = ψ(ϕ(S)). Steinke and Zakynthinou [\[12\]](#page-10-7) establish that if an algorithm A<sup>n</sup> is a samplecompression scheme (ϕ, ψ) of size k, then it must be that the associated CMI is bounded from above as CMI(An) ≤ k log(2n). The finding of [\[43\]](#page-12-4) that, for certain SCO problem instances, every ε-learner algorithm must have CMI that blows up with n (faster than n) was used therein to refute the existence of such sample-compression schemes for the studied SCO problems. The results of this paper may constitute a path to obtaining such schemes when the definition is extended to involve approximate reconstruction (in terms of induced generalization error) instead of the strict An(·) = ψ(ϕ(·)) of Littlesone and Warmuth [\[60\]](#page-13-2).

#### Fingerprinting codes and privacy attacks

In [\[61\]](#page-13-3), the authors study the problem of designing privacy attacks on mean estimators that expose a fraction of the training data. They show that a well-designed adversary can guess membership of the training samples from the output of every algorithm that estimates mean with high precision. Our results suggest that stochastic projection and lossy compression might be useful to construct differentially private codes that prevent such fingerprinting type attacks. For instance, while noise would naturally be one constituent of the recipe in this context, its injection in a suitable smaller subspace of the summary statistics might be the key enabler of privacy guarantees in such contexts.

#### Concluding remarks

In this work, we revisit recent limitations identified in conditional mutual information-based generalization bounds. By incorporating stochastic projections and lossy compression mechanisms into the CMI framework, we derive bounds that remain informative in stochastic convex optimization, thereby offering a new perspective on the results in [\[43,](#page-12-4) [46\]](#page-12-5). Our approach also provides a constructive resolution to the memorization phenomenon described in [\[43\]](#page-12-4), by showing that for any algorithm and data distribution, one can construct an alternative model that does not trace training data while achieving comparable generalization.

Like prior work on information-theoretic bounds, our analysis applies to stochastic convex optimization. A natural, open question is whether and how these results can be extended to more general learning settings. Another key direction is to translate our theoretical findings into actionable design principles for learning algorithms with controlled generalization and compressibility.

# Acknowledgments

The authors thank the anonymous reviewers for their many insightful comments and suggestions. Their feedback and the ensuing discussions led to the alternative variants of Theorem [8](#page-24-0) (*i.e.*, Theorem [6](#page-8-1) and Theorem [9\)](#page-24-2), and greatly shaped some of the paper's discussions. Kimia Nadjahi would also like to thank Mahdi Haghifam for the helpful discussions.

# References


[1] VN Vapnik and A Ya Chervonenkis. On the uniform convergence of relative frequencies of events to their probabilities. *Theory of Probability and its Applications*, 16(2):264, 1971. [2] Peter L Bartlett, Olivier Bousquet, and Shahar Mendelson. Local rademacher complexities. *Annals of Statistics*, pages 1497–1537, 2005. [3] Shai Shalev-Shwartz, Ohad Shamir, Nathan Srebro, and Karthik Sridharan. Learnability, stability and uniform convergence. *The Journal of Machine Learning Research*, 11:2635–2670, 2010. [4] Umut S¸ims¸ekli, Ozan Sener, George Deligiannidis, and Murat A Erdogdu. Hausdorff dimension, heavy tails, and generalization in neural networks. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, *Advances in Neural Information Processing Systems*, volume 33, pages 5138–5151. Curran Associates, Inc., 2020. [5] Tolga Birdal, Aaron Lou, Leonidas Guibas, and Umut S¸ims¸ekli. Intrinsic dimension, persistent homology and generalization in neural networks. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2021. [6] Liam Hodgkinson, Umut Simsekli, Rajiv Khanna, and Michael Mahoney. Generalization bounds using lower tail exponents in stochastic optimizers. In *International Conference on Machine Learning*, pages 8774–8795. PMLR, 2022. [7] Soon Hoe Lim, Yijun Wan, and Umut S¸ims¸ekli. Chaotic regularization and heavy-tailed limits for deterministic gradient descent. *arXiv preprint arXiv:2205.11361*, 2022. [8] Yijun Wan, Melih Barsbey, Abdellatif Zaidi, and Umut S¸ims¸ekli. Implicit compressibility of overparametrized neural networks trained with heavy-tailed SGD. In *Proceedings of the 41st International Conference on Machine Learning*, pages 49845–49866, 2024. [9] Daniel Russo and James Zou. Controlling bias in adaptive data analysis using information theory. In Arthur Gretton and Christian C. Robert, editors, *Proceedings of the 19th International Conference on Artificial Intelligence and Statistics*, volume 51 of *Proceedings of Machine Learning Research*, pages 1232–1240, Cadiz, Spain, 09–11 May 2016. PMLR. [10] Aolin Xu and Maxim Raginsky. Information-theoretic analysis of generalization capability of learning algorithms. *Advances in Neural Information Processing Systems*, 30, 2017. [11] Yuheng Bu, Shaofeng Zou, and Venugopal V. Veeravalli. Tightening mutual information-based bounds on generalization error. *IEEE Journal on Selected Areas in Information Theory*, 1(1): 121–130, May 2020. ISSN 2641-8770. [12] Thomas Steinke and Lydia Zakynthinou. Reasoning about generalization via conditional mutual information. In Jacob Abernethy and Shivani Agarwal, editors, *Proceedings of Thirty Third Conference on Learning Theory*, volume 125 of *Proceedings of Machine Learning Research*, pages 3437–3452. PMLR, 09–12 Jul 2020. [13] Amedeo Roberto Esposito, Michael Gastpar, and Ibrahim Issa. Generalization error bounds via Renyi-, ´ f-divergences and maximal leakage, 2020. [14] Mahdi Haghifam, Gintare Karolina Dziugaite, Shay Moran, and Daniel M. Roy. Towards a unified information-theoretic framework for generalization. In *Thirty-Fifth Conference on Neural Information Processing Systems*, 2021. [15] Gergely Neu, Gintare Karolina Dziugaite, Mahdi Haghifam, and Daniel M. Roy. Informationtheoretic generalization bounds for stochastic gradient descent, 2021. [16] Gholamali Aminian, Yuheng Bu, Laura Toni, Miguel Rodrigues, and Gregory Wornell. An exact characterization of the generalization error for the gibbs algorithm. *Advances in Neural Information Processing Systems*, 34:8106–8118, 2021.

[17] Ruida Zhou, Chao Tian, and Tie Liu. Individually conditional individual mutual information bound on generalization error. *IEEE Transactions on Information Theory*, 68(5):3304–3316, 2022. doi: 10.1109/TIT.2022.3144615. [18] Gabor Lugosi and Gergely Neu. Generalization bounds via convex analysis. In ´ *Conference on Learning Theory*, pages 3524–3546. PMLR, 2022. [19] Saeed Masiha, Amin Gohari, and Mohammad Hossein Yassaee. f-divergences and their applications in lossy compression and bounding generalization error. *IEEE Transactions on Information Theory*, 2023. [20] Hrayr Harutyunyan, Maxim Raginsky, Greg Ver Steeg, and Aram Galstyan. Information-theoretic generalization bounds for black-box learning algorithms. *Advances in Neural Information Processing Systems*, 34, 2021. [21] Fredrik Hellstrom and Giuseppe Durisi. A new family of generalization bounds using samplewise ¨ evaluated cmi. *Advances in Neural Information Processing Systems*, 35:10108–10121, 2022. [22] Milad Sefidgaran, Romain Chor, and Abdellatif Zaidi. Rate-distortion theoretic bounds on generalization error for distributed learning. *Advances in Neural Information Processing Systems*, 35: 19687–19702, 2022. [23] Milad Sefidgaran and Abdellatif Zaidi. Data-dependent generalization bounds via variable-size compressibility. *IEEE Transactions on Information Theory*, 2024. [24] David A McAllester. Some PAC-Bayesian theorems. In *Proceedings of the eleventh annual conference on Computational learning theory*, pages 230–234, 1998. [25] David A McAllester. PAC-Bayesian model averaging. In *Proceedings of the twelfth annual conference on Computational learning theory*, pages 164–170, 1999. [26] Matthias Seeger. PAC-Bayesian generalisation error bounds for gaussian process classification. *Journal of machine learning research*, 3(Oct):233–269, 2002. [27] John Langford and Rich Caruana. (not) bounding the true error. *Advances in Neural Information Processing Systems*, 14, 2001. [28] Olivier Catoni. A PAC-Bayesian approach to adaptive classification. *preprint*, 840, 2003. [29] Andreas Maurer. A note on the pac bayesian theorem. *arXiv preprint cs/0411099*, 2004. [30] Pascal Germain, Alexandre Lacasse, Franc¸ois Laviolette, and Mario Marchand. PAC-Bayesian learning of linear classifiers. In *Proceedings of the 26th Annual International Conference on Machine Learning*, pages 353–360, 2009. [31] Ilya O Tolstikhin and Yevgeny Seldin. PAC-Bayes-empirical-bernstein inequality. *Advances in Neural Information Processing Systems*, 26, 2013. [32] Luc Begin, Pascal Germain, Franc¸ois Laviolette, and Jean-Francis Roy. PAC-Bayesian bounds based ´ on the renyi divergence. In ´ *Artificial Intelligence and Statistics*, pages 435–444. PMLR, 2016. [33] Niklas Thiemann, Christian Igel, Olivier Wintenberger, and Yevgeny Seldin. A strongly quasiconvex PAC-Bayesian bound. In *International Conference on Algorithmic Learning Theory*, pages 466–492. PMLR, 2017. [34] Gintare Karolina Dziugaite and Daniel M Roy. Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data. *arXiv preprint arXiv:1703.11008*, 2017. [35] Behnam Neyshabur, Srinadh Bhojanapalli, and Nathan Srebro. A PAC-Bayesian approach to spectrally-normalized margin bounds for neural networks, 2018. [36] Omar Rivasplata, Ilja Kuzborskij, Csaba Szepesvari, and John Shawe-Taylor. PAC-Bayes analysis ´ beyond the usual bounds. *Advances in Neural Information Processing Systems*, 33:16833–16845, 2020. [37] Jeffrey Negrea, Gintare Karolina Dziugaite, and Daniel Roy. In defense of uniform convergence: Generalization via derandomization with an application to interpolating predictors. In *International Conference on Machine Learning*, pages 7263–7272. PMLR, 2020. [38] Jeffrey Negrea, Mahdi Haghifam, Gintare Karolina Dziugaite, Ashish Khisti, and Daniel M. Roy. Information-theoretic generalization bounds for SGLD via data-dependent estimates, 2020.

[39] Paul Viallard, Pascal Germain, Amaury Habrard, and Emilie Morvant. A general framework for the disintegration of PAC-Bayesian bounds. *arXiv preprint arXiv:2102.08649*, 2021. [40] Raef Bassily, Shay Moran, Ido Nachum, Jonathan Shafer, and Amir Yehudayoff. Learners that use little information. In *Algorithmic Learning Theory*, pages 25–55. PMLR, 2018. [41] Ido Nachum, Jonathan Shafer, and Amir Yehudayoff. A direct sum result for the information complexity of learning. In *Conference On Learning Theory*, pages 1547–1568. PMLR, 2018. [42] Fredrik Hellstrom, Giuseppe Durisi, Benjamin Guedj, Maxim Raginsky, et al. Generalization bounds: ¨ Perspectives from information theory and PAC-Bayes. *Foundations and Trends® in Machine Learning*, 18(1):1–223, 2025. [43] Idan Attias, Gintare Karolina Dziugaite, Mahdi Haghifam, Roi Livni, and Daniel M. Roy. Information complexity of stochastic convex optimization: Applications to generalization, memorization, and tracing. In *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pages 2035–2068. PMLR, 21–27 Jul 2024. [44] Hassan Hafez-Kolahi, Zeinab Golgooni, Shohreh Kasaei, and Mahdieh Soleymani. Conditioning and processing: Techniques to improve information-theoretic generalization bounds. *Advances in Neural Information Processing Systems*, 33:16457–16467, 2020. [45] Ziqiao Wang and Yongyi Mao. Tighter information-theoretic generalization bounds from supersamples. In *Proceedings of the 40th International Conference on Machine Learning*, pages 36111–36137, 2023. [46] Roi Livni. Information theoretic lower bounds for information theoretic upper bounds. In *Proceedings of the 37th International Conference on Neural Information Processing Systems*, NIPS '23, Red Hook, NY, USA, 2023. Curran Associates Inc. [47] Mahdi Haghifam, Borja Rodr´ıguez-Galvez, Ragnar Thobaben, Mikael Skoglund, Daniel M Roy, and ´ Gintare Karolina Dziugaite. Limitations of information-theoretic generalization bounds for gradient descent methods in stochastic convex optimization. In *International Conference on Algorithmic Learning Theory*, pages 663–706. PMLR, 2023. [48] Ziqiao Wang and Yongyi Mao. Sample-conditioned hypothesis stability sharpens informationtheoretic generalization bounds. *Advances in Neural Information Processing Systems*, 36:49513– 49541, 2023. [49] Kimia Nadjahi, Kristjan Greenewald, Rickard Bruel Gabrielsson, and Justin Solomon. Slicing mutual ¨ information generalization bounds for neural networks. In *International Conference on Machine Learning*, pages 37213–37236. PMLR, 2024. [50] Shai Shalev-Shwartz, Ohad Shamir, Nathan Srebro, and Karthik Sridharan. Stochastic convex optimization. In *COLT*, volume 2, number 4, page 5, 2009. [51] Yuheng Bu, Weihao Gao, Shaofeng Zou, and Venugopal Veeravalli. Information-theoretic understanding of population risk improvement with model compression. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 34, pages 3300–3307, 2020. [52] Milad Sefidgaran, Amin Gohari, Gael Richard, and Umut Simsekli. Rate-distortion theoretic generalization bounds for stochastic learning algorithms. In *Conference on Learning Theory*, pages 4416–4463. PMLR, 2022. [53] Peter Grunwald, Thomas Steinke, and Lydia Zakynthinou. PAC-Bayes, mac-bayes and conditional mutual information: Fast rate bounds that handle general vc classes. In *Conference on Learning Theory*, pages 2217–2247. PMLR, 2021. [54] Milad Sefidgaran, Abdellatif Zaidi, and Piotr Krasnowski. Minimum description length and generalization guarantees for representation learning. In *Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)*, 2023. [55] Borja Rodr´ıguez-Galvez, Germ ´ an Bassi, Ragnar Thobaben, and Mikael Skoglund. On random subset ´ generalization error bounds and the stochastic gradient langevin dynamics algorithm. In *2020 IEEE Information Theory Workshop (ITW)*, pages 1–5. IEEE, 2021. [56] Ruida Zhou, Chao Tian, and Tie Liu. Individually conditional individual mutual information bound on generalization error. *IEEE Transactions on Information Theory*, 68(5):3304–3316, 2022. [57] William B Johnson and Joram Lindenstrauss. Extensions of lipschitz mappings into a hilbert space
  - 26. *Contemporary mathematics*, 26:28, 1984.

[58] Gavin Brown, Mark Bun, Vitaly Feldman, Adam Smith, and Kunal Talwar. When is memorization of irrelevant training data necessary for high-accuracy learning? In *Proceedings of the 53rd annual ACM SIGACT symposium on theory of computing*, pages 123–132, 2021. [59] Jonathan Scarlett and Volkan Cevher. An introductory guide to fano's inequality with applications in statistical estimation. *arXiv preprint arXiv:1901.00555*, 2019. [60] Nick Littlestone and Manfred Warmuth. Relating data compression and learnability. *Citeseer*, 1986. [61] Cynthia Dwork, Adam Smith, Thomas Steinke, Jonathan Ullman, and Salil Vadhan. Robust traceability from trace amounts. In *2015 IEEE 56th Annual Symposium on Foundations of Computer Science*, pages 650–669, 2015. doi: 10.1109/FOCS.2015.46. [62] Michel Ledoux and Michel Talagrand. *Probability in Banach Spaces: isoperimetry and processes*. Springer Science & Business Media, 2013. [63] Ankit Pensia, Varun Jog, and Po-Ling Loh. Generalization error bounds for noisy, iterative algorithms. *2018 IEEE International Symposium on Information Theory (ISIT)*, pages 546–550, 2018. [64] Mahdi Haghifam, Jeffrey Negrea, Ashish Khisti, Daniel M Roy, and Gintare Karolina Dziugaite. Sharpened generalization bounds based on conditional mutual information and an application to noisy, iterative algorithms. *Advances in Neural Information Processing Systems*, 33:9925–9935, 2020. [65] Borja Rodr´ıguez Galvez, Germ ´ an Bassi, Ragnar Thobaben, and Mikael Skoglund. On random sub- ´ set generalization error bounds and the stochastic gradient langevin dynamics algorithm. *CoRR*, abs/2010.10994, 2020. [66] Hao Wang, Yizhe Huang, Rui Gao, and Flavio Calmon. Analyzing the generalization capability of SGLD using properties of gaussian channels. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, *Advances in Neural Information Processing Systems*, volume 34, pages 24222–24234. Curran Associates, Inc., 2021. [67] Hao Wang, Rui Gao, and Flavio P Calmon. Generalization bounds for noisy iterative algorithms using properties of additive noise channels. *Journal of machine learning research*, 24(26):1–43, 2023. [68] Sejun Park, Umut Simsekli, and Murat A Erdogdu. Generalization bounds for stochastic gradient descent via localized ε-covers. *Advances in Neural Information Processing Systems*, 35:2790–2802, 2022. [69] Aymeric Dieuleveut, Alain Durmus, and Francis Bach. Bridging the gap between constant step size stochastic gradient descent and Markov chains, 2018. [70] Leo Kozachkov, Patrick M Wensing, and Jean-Jacques Slotine. Generalization in supervised learning through riemannian contraction. *arXiv preprint arXiv:2201.06656*, 2022. [71] Allan Grønlund, Lior Kamma, and Kasper Green Larsen. Near-tight margin-based generalization bounds for support vector machines. In Hal Daume III and Aarti Singh, editors, ´ *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pages 3779–3788. PMLR, 13–18 Jul 2020. [72] Jean Gallier. *Discrete mathematics*. Springer Science & Business Media, 2011. [73] Robert G Gallager. *Information theory and reliable communication*, volume 588. Springer, 1968. [74] Ziqiao Wang and Yongyi Mao. On the generalization of models trained with SGD: Informationtheoretic bounds and implications, 2021.
# NeurIPS Paper Checklist

#### 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We proved several theoretical results showing the effectiveness of the projection and quantization technique and discussed it in detail. In particular, we showed how this can be used to resolve the recently raised concerns on the information-theoretic bounds.

#### Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

#### 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We clearly stated the problem instances and classes for which we demonstrated that this approach results in good generalization bounds. We also stated all assumptions needed for each result.

# Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: In this paper, we stated all results rigorously, along with the assumptions used and detailed proofs in the supplements. The proofs are rigorous with enough details provided for the reader to follow.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

#### 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: Our work is a theoretical paper with rigorously proven claims, and does not involve any experiment.

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
  - (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
  - (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
  - (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
  - (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case

of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

#### 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: Our work does not involve any experiment.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ([https://nips.cc/public/](https://nips.cc/public/guides/CodeSubmissionPolicy) [guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ([https:](https://nips.cc/public/guides/CodeSubmissionPolicy) [//nips.cc/public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

#### 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: Our work does not involve any experiment.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

#### 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: Our work does not involve any experiment.

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

#### 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification:Our work does not involve any experiment.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

#### 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

Answer: [Yes]

Justification: Our work is a theoretical paper on learning theory and does not violate any code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

#### 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our work is a theoretical paper on learning theory and does not have any direct negative societal impact.

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: Our work does not involve any experiment.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: Our work does not involve any experiment.

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <paperswithcode.com/datasets> has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

#### 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: Our work does not involve any experiment or any new asset.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

#### 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our work is a theoretical paper on learning theory.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our work does not involve crowd sourcing nor any research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We have not used LLMs for this work.

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy (<https://neurips.cc/Conferences/2025/LLM>) for what should or should not be described.

# Appendices

The appendices are organized as follows:

- In Appendix [A,](#page-21-0) we present some extensions of Theorem [1,](#page-3-0) that are used in the subsequent sections.
- The results of Section [4](#page-4-0) have been extended to a wider family of generalized linear stochastic optimization problems in Appendix [B.](#page-22-0)
- Further results on memorization are presented in Appendix [C.](#page-23-0) In particular
  - In Appendix [C.1,](#page-23-2) we discuss what values of (m, q, ξ) can be achieved by a "dummy adversary".
  - In Appendix [C.2,](#page-24-1) we consider the case where the projection matrix Θ is fixed and shared with the adversary.
  - In Appendix [C.3,](#page-24-3) we discuss how to provide guarantees on the closeness in terms of the population risk between the projected-quantized model to the original model.
  - In Appendix [C.4,](#page-25-3) we provide technical lemmas used in the main text on reconciliation of our results with those of [\[43\]](#page-12-4).
- The generalization error of subspace training algorithms is investigated in Appendix [D.](#page-26-0) In particular, in Appendix [D.1,](#page-27-0) we develop generalization bounds for the case where iterative optimization algorithms such as SGD and SGLD are used for the optimization of the subspace training algorithms.
- The proof of Theorem [1](#page-3-0) is presented in Appendix [E.](#page-29-0)
- In Appendix [F,](#page-31-0) we present the proofs of the results presented in Section [4](#page-4-0) and Appendix [B](#page-22-0) regarding the applications of Theorem [1](#page-3-0) to resolving recently raised limitations of classic CMI bounds. In particular,
  - a general Johnson-Lindenstrauss projection scheme JL(d, cw, ν) is introduced in Appendix [F.1,](#page-31-1) which is used in the following subsections, with different choices of (d, cw, ν),
  - Theorem [3](#page-5-1) is proved in Appendix [F.2,](#page-32-0)
  - Proposition [1](#page-6-1) is proved in Appendix [F.3,](#page-35-0)
  - Theorem [7](#page-23-3) is proved in Appendix [F.4,](#page-35-1)
  - and Lemma [4](#page-38-0) is proved in Appendix [F.5.](#page-39-0)
- Appendix [G](#page-40-1) contains the proofs of the results in Section [5](#page-7-0) and Appendix [C,](#page-23-0) about the memorization. More precisely,
  - Theorem [5](#page-8-0) is proved in Appendix [G.1,](#page-40-0)
  - Theorem [6](#page-8-1) is proved in Appendix [G.2,](#page-45-0)
  - Lemma [1](#page-23-1) is proved in Appendix [G.3,](#page-45-1)
  - Theorem [8](#page-24-0) is proved in Appendix [G.4,](#page-46-0)
  - Theorem [9](#page-24-2) is proved in Appendix [G.5,](#page-48-0)
  - Lemma [2](#page-25-0) is proved in Appendix [G.6,](#page-49-0)
  - and Lemma [5](#page-44-0) is proved in Appendix [G.7.](#page-52-0)
- Lastly, Appendix [H](#page-54-0) contains the proofs of the results of Appendix [D](#page-26-0) on the generalization error of subspace training algorithms when trained using SGD or SGLD. More precisely,
  - Lemma [3](#page-27-1) is proved in Appendix [H.1,](#page-54-1)
  - Theorem [10](#page-28-0) is proved in Appendix [H.2,](#page-55-0)
  - Theorem [13](#page-28-1) is proved in Appendix [H.3,](#page-58-0)
  - and Lemma [6](#page-58-1) is proved Appendix [H.4.](#page-61-0)

# A Extensions of Theorem [1](#page-3-0)

As mentioned in Section [3,](#page-2-3) Theorem [1](#page-3-0) can be improved in several ways, similar to those proposed in [\[20,](#page-11-6) [53,](#page-12-14) [54\]](#page-12-15). Here, we state only the single-datum version of Theorem [1,](#page-3-0) which is used in Appendix [D,](#page-26-0) followed by a remark about extending Theorem [1](#page-3-0) and its corollary to more general lossy compression algorithms. Denote

$$\mathbf{J}_{-i} = \mathbf{J}_{[n] \setminus \{i\}}, \quad \tilde{\mathbf{S}}_{-i} \triangleq \tilde{\mathbf{S}}_{[n] \setminus \{i\}, [2]} = \tilde{\mathbf{S}} \setminus \{Z_{i,0}, Z_{i,1}\}.$$

Corollary 1. *Consider the setup of Theorem [1.](#page-3-0) Then,*

$$\text{gen}(\mu, \mathcal{A}) \leq \inf_{P_{\tilde{W}|\Theta}} \inf_{P_{\Theta}} \frac{1}{n} \sum_{i \in [n]} \mathbb{E}_{P_{\tilde{S}} P_{\Theta}} \left[ \sqrt{2\Delta \ell_{\tilde{w}, i}(\tilde{\mathbf{S}}, \Theta) \text{CMI}_i^{\Theta}(\tilde{\mathbf{S}}, \hat{\mathcal{A}})} \right] + \epsilon, \quad (9)$$

*and*

$$\text{gen}(\mu, \mathcal{A}) \leq \inf_{P_{\tilde{W}|\Theta^{\top}}} \inf_{P_{\Theta}} \frac{1}{n} \sum_{i \in [n]} \mathbb{E}_{P_{\tilde{S}} P_{\Theta} P_{J_{-i}}} \left[ \sqrt{2 \Delta \ell_{\hat{w}, i} (\tilde{\mathbf{S}}, \Theta) \text{CMI}_{i, J_{-i}}^{\Theta} (\tilde{\mathbf{S}}, \hat{\mathcal{A}})} \right] + \epsilon, \quad (10)$$

*where the infima are over* PW<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> *and* P<sup>Θ</sup> *that satisfy the distortion criterion*

$$\mathbb{E}_{P_{S_n, W} P_{\Theta P_{\hat{W}|\Theta} \tau_W}} \left[ \text{gen}(S_n, W) - \text{gen}(S_n, \Theta \hat{W}) \right] \leq \epsilon, \quad (11)$$

*and where*

$$\begin{aligned} \text{CMI}_i^{\Theta}(\tilde{\mathbf{S}}, \hat{\mathcal{A}}) &\triangleq |\tilde{\mathbf{S}}^{\cdot, \Theta}(\hat{\mathcal{A}}(\tilde{\mathbf{S}}_{\mathbf{J}}, \Theta); J_i)|, \\ \text{CMI}_{i, \mathbf{J}_{-i}}^{\Theta}(\tilde{\mathbf{S}}, \hat{\mathcal{A}}) &\triangleq |\tilde{\mathbf{S}}^{\cdot, \mathbf{J}_{-i}, \Theta}(\hat{\mathcal{A}}(\tilde{\mathbf{S}}_{\mathbf{J}}, \Theta); J_i)|, \\ \Delta \ell_{\hat{w}, i}(\tilde{\mathbf{S}}, \Theta) &\triangleq \mathbb{E}_{P_{W|\tilde{\mathbf{S}}} P_{\hat{W}|\Theta^{-T}}} \left[ (\ell(Z_{i,0}, \Theta \hat{W}) - \ell(Z_{i,1}, \Theta \hat{W}))^2 \right]. \end{aligned}$$

To derive inequality [9,](#page-22-1) first note that by equation [11,](#page-22-2) it is sufficient to show that

$$\text{gen}(\mu, \hat{A}) \leq \inf_{P_{W|\Theta} \top W} \inf_{P_{\Theta}} \frac{1}{n} \sum_{i \in [n]} \mathbb{E}_{P_{S_{\Theta}}} \left[ \sqrt{2 \Delta \ell_{\hat{w}_i}(\tilde{\mathbf{S}}, \Theta) \text{CMI}_i^{\Theta}(\tilde{\mathbf{S}}, \hat{A})} \right].$$

Next, using the linearity of the expectation, we can write

$$\begin{aligned}\mathbb{E}[\text{gen}(S_n, \hat{W})] &= \frac{1}{n} \sum_{i \in [n]} \mathbb{E}[\text{gen}(\{Z_i\}, \hat{W})] \\ &= \frac{1}{n} \sum_{i \in [n]} \mathbb{E}_{\hat{\mathbf{S}}_{-i}} \left[ \mathbb{E}_{Z_i, \hat{W}}[\text{gen}(\{Z_i\}, \hat{W})] \right].\end{aligned}\tag{12}$$

Then applying Theorem [1](#page-3-0) for each of the terms <sup>E</sup>Zi,W<sup>ˆ</sup> [gen({Zi}, <sup>W</sup><sup>ˆ</sup> )] yields equation [9.](#page-22-1)

The inequality [10](#page-22-3) can be achieved similarly, by considering

$$\mathbb{E}[\text{gen}(S_n, \hat{W})] = \frac{1}{n} \sum_{i \in [n]} \mathbb{E}_{\tilde{\mathbf{S}}_{-i}, \mathbf{J}_{-i}} \left[ \mathbb{E}_{Z_i, \hat{W}}[\text{gen}(\{Z_i\}, \hat{W})] \right],$$

instead of equation [12.](#page-22-4)

The results of Theorem [1](#page-3-0) and, consequently, Corollary [1,](#page-21-1) are valid for a broader class of learning algorithms, A, and lossy compression algorithms, Aˆ, as discussed in the remark below and shown in the proof of Theorem [1](#page-3-0) in Appendix [E.](#page-29-0)

Remark 1. *As shown in Appendix [E,](#page-29-0) the bounds of Theorem [1](#page-3-0) and consequently Corollary [1](#page-21-1) hold if the learning algorithm* A *is aware of the projection matrix* Θ*,* i.e., *if* A: Z <sup>n</sup> × <sup>R</sup> <sup>D</sup>×<sup>d</sup> → W *takes both the dataset* S *and the projection matrix* Θ *as input in order to learn the model* W*. Moreover, the results of Theorem [1](#page-3-0) and Corollary [1](#page-21-1) are valid if the quantization step can also depend on* S*,* Θ *and* A(S, Θ)*. In this general case,* Wˆ = Aˆ(S, Θ) = A˜(Θ, S, A(S, Θ)) = Wˆ *. This setting trivially includes the case in which* A: Z <sup>n</sup> → W *and the quantization depends only on* Θ <sup>⊤</sup>A(S, Θ)*. For the ease of the exposition, we found it better not to state the result in its most general form.*

# B Generalized linear stochastic optimization problems

In this section, we show that our bound of Theorem [1](#page-3-0) can be applied successfully to get useful bounds on the generalization error of a family of generalized linear stochastic optimization problems that is wider than the ones considered previously in related prior art.

Definition 9 (Generalized linear stochastic optimization). *Let* L, B, R ∈ <sup>R</sup><sup>+</sup> *and* W = BD(R)*. Define the loss function* ℓgl : Z × W → <sup>R</sup> *as*

$$\ell_{gl}(z, w) = g\left(\langle w, \phi(z) \rangle, z\right) + r(w),$$

*where* g : <sup>R</sup> × Z → <sup>R</sup> *is* L*-Lipschitz with respect to the first argument,* ϕ: Z → BD(B) *and* r : W → <sup>R</sup> *is some arbitrary function. Denote this problem as* P (D) glso*.*

This class of problems is larger than the one considered in [\[50\]](#page-12-8). For instance, while the results of [\[50\]](#page-12-8) require the L-Lipschitz function g(·, ·) and the function r(·) to be both convex to hold, our next theorem applies to arbitrary L-Lipschitz functions g(·, ·) and arbitrary functions r(·).

Theorem 7. *For every learning algorithm* A: Z <sup>n</sup> → W *of the instance problem* P (D) glso *defined in Definition [9,](#page-22-5) the generalization bound of Theorem [1](#page-3-0) yields*

$$\text{gen}(\mu, \mathcal{A}) = \mathcal{O}\left(\frac{LRB}{\sqrt{n}}\right).$$

The proof, stated in Appendix [F.4,](#page-35-1) is based on Theorem [1.](#page-3-0) In order to find a proper stochastic projection and quantization, we use the Johnson-Lindenstrauss (JL) dimensional reduction transformation in a space of dimension d. Then, we apply lossy compression to the projected model. Thanks to the combined projection-quantization, the disintegrated CMI can be bounded easily in the d-dimensional space. However, there are two main caveats to using the JL Lemma directly. First, one needs to bound the term ∆ℓwˆ(S˜, Θ) (see equation [4\)](#page-3-3). This is particularly difficult since the JL Lemma does not guarantee distance preservation in the original space of dimension D after projecting back the quantized model. Second, bounding the distortion term is less easy than in Theorem [3,](#page-5-1) since using the Lipschitz property requires bounding the absolute value of the difference between inner products of the original and projected-quantized models. In essence, this is the reason why, by opposition to JL transformation for which it suffices to take d = log(n), here one needs a higher-dimensional projection space comparatively, with d = √ n.

Theorem [7](#page-23-3) shows that no counter-example could be found for which the bound of Theorem [1](#page-3-0) does not vanish, even if one considers the bigger class of generalized linear stochastic optimization problems of Definition [9](#page-22-5) in place of the SCO class problems of [\[43\]](#page-12-4). The convergence rate O(1/ √<sup>4</sup> <sup>n</sup>) of Theorem [7](#page-23-3) is, however, not optimal. A better rate, O(1/ √ n), seems to be achievable using Rademacher analysis and Talagrand's contraction lemma [\[62\]](#page-13-4). Using a more refined analysis, the same rate might be possible to achieve using our Theorem [1.](#page-3-0) More precisely, in the part of the current proof of Theorem [7](#page-23-3) that analyses the distortion term, we do *not* account for the discrepancy between the empirical measure of S and the true distribution µ; and, instead, we consider a worst-case scenario. A finer analysis that takes such discrepancy into account should lead to a sharper expected concentration bound for the distortion term, and, so, a better rate.

#### C Further results on memorization

In this section, we provide further results on memorization. In Appendix [C.1,](#page-23-2) we show that even a "dummy" adversary can trace the data for some values of (m, q, ξ). In Appendix [C.2,](#page-24-1) we study the case where the projection matrix Θ is deterministic. In Appendix [C.3,](#page-24-3) we provide another variant of Theorem [8,](#page-24-0) in which we can guarantee the closeness of the projected-quantized model to the original model in terms of population risk (instead of the generalization error considered in Theorem [8\)](#page-24-0). Finally, in Appendix [C.4,](#page-25-3) we present some technical lemmas used in the discussions of Section [5](#page-7-0) on the relation of our results with those established in [\[43\]](#page-12-4).

#### C.1 Dummy adversary

In this section, we show that certain values of (m, q, ξ) are attainable by a "dummy" adversary who makes guesses without even looking at the given data sample.

Lemma 1. *Given a learning algorithm* A<sup>n</sup> : Z <sup>n</sup> → W*, there exists an adversary that* (m, q, ξ)*-traces the data for some* m ∈ [0, n] *and* q, ξ ∈ [0, 1] *if one of the following conditions holds: i)* ξ ≥ q*, or ii) there exists an* α ∈ [0, 1 − ξ] ∩ [0, 1) *such that* <sup>n</sup> q 1 − <sup>1</sup>−<sup>α</sup> + r 2n log 1 1−q/(1−α) + m <sup>n</sup> ≤ 1*.*

This lemma, proved in Appendix [G.3,](#page-45-1) implies in particular that even a dummy adversary can (m, q, ξ) trace the data in several cases: when ξ is small, when q is large, or when ξ is small and q is large, provided that m = o(n).

#### C.2 Deterministic projection

In this section, we show that in Theorem [6,](#page-8-1) one can allow Θ to be deterministic. However, this comes at the cost of being specific to a given data distribution.

Theorem 8. *Consider the* P (D) cvx *problem instance of Definition [4](#page-4-2) with* L = R = 1*. For every* r < 1*, every* Z ⊆ BD(1)*, every data distribution* µ*, and every learning algorithm* A*, there exist a projection matrix* Θ ∈ R <sup>D</sup>×<sup>d</sup> *with* d = ⌈n 2r−1 ⌉*, a Markov Kernel* PW<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> *and a* compression algorithm A ∗ <sup>Θ</sup> : Z <sup>n</sup> → <sup>R</sup> d *, defined as* A ∗ <sup>Θ</sup>(Sn) <sup>≜</sup> A˜(Θ<sup>⊤</sup>A(Sn)) = <sup>W</sup><sup>ˆ</sup> *, such that the following conditions are met simultaneously:*

- *i) the generalization error of the auxiliary model* ΘWˆ *satisfies*

$$\left| \mathbb{E}_{P_{S_n, W} P_{\hat{W} | \Theta \top W}} \left[ \text{gen}(S_n, W) - \text{gen}(S_n, \Theta \hat{W}) \right] \right| = \mathcal{O}\left(n^{-r}\right), \quad (13)$$

*where the expectation is taken over* (Sn, W, <sup>W</sup><sup>ˆ</sup> ) <sup>∼</sup> <sup>P</sup>Sn,W <sup>P</sup>W<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> *.*

- *ii) if there exists an adversary that by having access to both* Θ *and* Wˆ *(and hence* ΘWˆ *)* (m, q, ξ) *traces the data, then it must be that: a)* m = o(n) *or* ξ ≥ q*, and b) if, for some* α ∈ (0, 1) *and* n<sup>0</sup> ∈ N ∗ *,* m ≥ αn *for every* n ≥ n0*, then for any* ϵ ∈ (0, α) *it holds that:* P P <sup>i</sup>∈[n] <sup>Q</sup>(ΘW , Z <sup>ˆ</sup> i,0, µ) <sup>≥</sup> <sup>m</sup>′ ≥ (α − ϵ)q*, where* m′ = ϵ 1/q+ϵ−α n − o(n) = Ω(n)*.*

As shown in the proof in Appendix [G.4,](#page-46-0) the constraint on the difference generalization error can be replaced with one with a faster decay with n, namely

$$\mathbb{E}_{P_{S_n, W} P_{\hat{W} | \Theta^\top W}} \left[ \text{gen}(S_n, W) - \text{gen}(S_n, \Theta \hat{W}) \right] = \mathcal{O} \left( n^{-r} \right)$$

for some <sup>r</sup> ∈ [R] and <sup>d</sup> = 500<sup>r</sup> log(n). Also, if <sup>n</sup> <sup>=</sup> <sup>N</sup>(ε, δ), then m, m′ = Ω 1/ε<sup>2</sup> , which means that any adversary who (m, q, ξ)-traces the training data is deemed to misclassify any arbitrary big part of the test samples.

For the proof of Theorem [8,](#page-24-0) we first apply the projection-quantization approach of Theorem [3.](#page-5-1) Then, for a proper Θ that satisfies the distortion criterion of equation [13](#page-24-4) and for which the CMI is o(n) we apply Theorem [8.](#page-24-0) Note two important differences with Theorem [3.](#page-5-1) First, because one now deals with *absolute value* of the average difference of generalization errors one also needs to lower bound the average distortion. Also, for r > 1/2 a faster convergence rate of O(n −r ) is required. This renders the analysis trickier and requires projection on a space of dimension n 2r−1 .

#### C.3 Guarantees on the population risk

In this section, we demonstrate that the closeness guarantee of the projected-quantized model and the original model can also be provided in terms of population risk.

Theorem 9. *Consider the* P (D) cvx *problem instance of Definition [4](#page-4-2) with* L = R = 1*. For every* r < 1/2*, every* Z ⊆ BD(1)*, every data distribution* µ*, and every learning algorithm* A*, there exist a projection matrix* Θ ∈ R <sup>D</sup>×<sup>d</sup> *with* d = ⌈n 2r ⌉*, a Markov Kernel* PW<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> *and a* compression algorithm A ∗ <sup>Θ</sup> : Z <sup>n</sup> → <sup>R</sup> d *, defined as* A ∗ <sup>Θ</sup>(Sn) <sup>≜</sup> A˜(Θ<sup>⊤</sup>A(Sn)) = <sup>W</sup><sup>ˆ</sup> *, such that the following conditions are met simultaneously:*

- *i) the generalization error of the auxiliary model* ΘWˆ *satisfies*

$$|\mathbb{E}_{P_{S_n, W} P_{\hat{W}|\Theta^\top W}}[\mathcal{L}(W) - \mathcal{L}(\Theta\hat{W})]| = \mathcal{O}\left(n^{-r}\right),$$

*where the expectation is taken over* (Sn, W, <sup>W</sup><sup>ˆ</sup> ) <sup>∼</sup> <sup>P</sup>Sn,W <sup>P</sup>W<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> *.*

- *ii) if there exists an adversary that by having access to both* Θ *and* Wˆ *(and hence* ΘWˆ *)* (m, q, ξ) *traces the data, then it must be that: a) Either* m = o(n) *or* ξ ≥ q*, and b) if* m = Ω(n) *then there exists* m′ = Ω(n) *and* q ′ ∈ (0, 1] *such that for sufficiently large* n*, it holds that* P P <sup>i</sup>∈[n] <sup>Q</sup>(ΘW , Z <sup>ˆ</sup> i,0, µ) <sup>≥</sup> <sup>m</sup>′ ≥ q ′ *.*

This result is proved in Appendix [G.5.](#page-48-0) Furthermore, similarly to Theorem [8,](#page-24-0) the constraint on the difference of population risks can be replaced with one with a faster decay with n, namely

$$\mathbb{E}_{P_{S_n, W} P_{\hat{W}|\Theta^\top W}} \left[ \mathcal{L}(W) - \mathcal{L}(\Theta \hat{W}) \right] = \mathcal{O} \left( n^{-r} \right), \quad (14)$$

for some r ∈ [R] and d = 500r log(n).

### C.4 Reconciliation with results of Attias et al. 2024

In this section, we provide the technical lemma showing that the norm two of the projected-quantized model, used in our results, is unbounded. Furthermore, we discuss in detail the steps of the proofs in [\[43\]](#page-12-4) where this bounded assumption is needed.

#### C.4.1 Uboundedness of the norm two of the projected-quantized model

In this section, for the projected-quantized algorithm ΘWˆ , used in Theorem [8](#page-24-0) and Theorem [6,](#page-8-1) we show that <sup>E</sup>W , <sup>ˆ</sup> <sup>Θ</sup> ΘWˆ 2 blows-up with n when D/d grows with n. This lemma is proved in Appendix [G.6.](#page-49-0)

Lemma 2. *Consider the JL*(d, cw, ν) *transformation described in Appendix [F.1,](#page-31-1) with some* d ∈ N +*,* c<sup>w</sup> ∈ h 1, p 5/4 *, and* ν ∈ (0, 1]*. Then, for every* w ∈ W*,*

$$\mathbb{E}_{\Theta, V_\nu} \left[ \| \Theta \hat{W} \|^2 \right] \geq \left( \frac{D + d + 1}{d} \right) \|w\|^2 - \sqrt{\frac{(D + d + 3)(D + d + 5)(d + 2)}{d^3}} \|w\|^2 e^{-0.1d(c_w^2 - 1)^2} - \frac{D\nu^2}{d}.$$

Consider ∥w∥ = 1 and let D = n 4 log(n/ξ) as considered in [\[43,](#page-12-4) Theorem 4.5]. We note that the notation d used in [\[43\]](#page-12-4) corresponds to the notation D in this paper.

Then, considering the constructions used for Theorem [8](#page-24-0) and Theorem [6,](#page-8-1) we have c<sup>w</sup> = 1.1 and ν = 0.4. Moreover, d is chosen either as

$$d = 500r \log(n),$$

or

$$d = n^{2r-1}, \quad r < 1.$$

Using Lemma [2](#page-25-0) with these choices give

$$\mathbb{E}_{\Theta, V_\nu} \left[ \left\| \Theta \hat{W} \right\|^2 \right] = \Omega \left( n^4 \right).$$

and

$$\mathbb{E}_{\Theta, V_\nu} \left[ \left\| \Theta \hat{W} \right\|^2 \right] = \Omega \left( n^{4-2r} \log(n) \right) = \Omega \left( n^3 \log(n) \right),$$

respectively. Hence, in both cases <sup>E</sup>Θ,V<sup>ν</sup> ΘWˆ 2 grows at least as fast as Ω(n 3 ).

#### C.4.2 Details of needed boundedness assumption in Attias et al.

As discussed before, [\[43,](#page-12-4) Theorem 4.1] and [\[43,](#page-12-4) Theorem 4.5] require the model to be bounded. However, as shown in the previous section, this assumption does not hold for the projected-quantized algorithm ΘWˆ when D/d and d grow with n. In this section, we discuss precisely where the bounded model assumption is necessary in the proofs of the impossibility results of [\[43\]](#page-12-4).

- Proof of [\[43,](#page-12-4) Theorem 4.1] and recall analysis in the proof of [\[43,](#page-12-4) Theorem 4.5], in part, relies on an established upper bound Ω(1/ε<sup>2</sup> ) on the term <sup>E</sup>[|I|], where the set I is the subset of columns of supersample such that one of the samples has a large correlation with the output of the algorithm and the other one has small correlation with the output of the algorithm.

- To establish this upper bound, in the last inequality of Page 19 of [\[43\]](#page-12-4), it is assumed that the norm of the model is bounded. Now, when working with the model ΘWˆ , the right-hand side of this inequality needs to be replaced by the D/d-dependent quantity 8ε 4n 2 D <sup>d</sup> + 2ε <sup>2</sup> = Ω(n 5 ) when D = Ω(n 4 ) and d = o(n). This has to be contrasted with the actual bound 8ε 4n <sup>2</sup> + 2ε <sup>2</sup> when the bounded model's norm assumption holds. Thus, one important issue is that, this quantity now being non-negligible, the LHS of (9) can no longer be lowerbounded by the RHS of the inequality (9).
- Another step, used for establishing the upper bound on <sup>E</sup>[|I|], is the step that upper bounds <sup>P</sup> (E c ) = O(1/n<sup>2</sup> ), for the event E defined on top of Page 19 of [\[43\]](#page-12-4). In this case again, in the set of equations before equation (12), it is assumed that the norm of the model is bounded to derive ∥Aˆθ 2 ∥ ≤ 144<sup>2</sup> ε 4 . However, since norm two of ΘWˆ is Ω(n 3 ), then these steps are ot valid and hence the analysis does not give <sup>P</sup> (E c ) = O(1/n<sup>2</sup> ) anymore.
- Another proof step of [\[43,](#page-12-4) Theorem 4.1], used also in the soundness analysis in the proof of [\[43,](#page-12-4) Theorem 4.5], relies on upper bounds for the error event G c , defined on [\[43,](#page-12-4) Page 18] as the probability that the correlation between the model output and the held-out samples is significant. These upper bounds, [\[43,](#page-12-4) Equations 11] in the proof of [\[43,](#page-12-4) Theorem 4.1] and also on [\[43,](#page-12-4) 29] in the soundness analysis in the proof of [\[43,](#page-12-4) Theorem 4.5], are based on an application of [\[43,](#page-12-4) Lemma B.8] and by assuming that the norm two of the model is bounded by 1. These steps again fail if the norm two of the model grows as Ω(D/d) = Ω(n ).

### D Random subspace training algorithms

The generalization bounds of Theorem [1](#page-3-0) and Corollary [1](#page-21-1) apply to any arbitrary learning algorithm. In this section, we show how this bound can be applied to random subspace training algorithms. Then, we consider the case where they are trained using an iterative optimization algorithm.

Let St(d, D) = {Θ ∈ <sup>R</sup> D×d : Θ⊤Θ = Id} be the Stiefel manifold, equipped with the uniform distribution PΘ. Moreover, for a given Θ ∈ <sup>R</sup> D×d , let W<sup>Θ</sup>,d ≜ {w ∈ <sup>R</sup> <sup>D</sup> : ∃w ′ ∈ <sup>R</sup> d s.t. w = Θw ′ }. Random subspace training algorithms first randomly generate an instance of Θ according to PΘ, which is kept frozen during training. A random subspace training algorithm A (d) : Z <sup>n</sup> × <sup>R</sup> <sup>D</sup>×<sup>d</sup> → W<sup>Θ</sup>,d is a learning algorithm that takes the dataset S and the projection matrix Θ as input, and chooses a model W ∈ W<sup>Θ</sup>,d, by choosing a W′ ∈ <sup>R</sup> d .

In other words, A (d) (S, Θ) = ΘW′ , or alternatively, since Θ <sup>⊤</sup>Θ = Id, W′ = Θ<sup>⊤</sup>A (d) (S, Θ). Hence, using Corollary [1](#page-21-1) and by noting Remark [1,](#page-22-6) we can obtain the following result.

Corollary 2. *Consider a random subspace training algorithm and a loss function* ℓ : Z × R <sup>D</sup> → [0, C]*. Then, for any* ϵ ∈ <sup>R</sup> *and the quantization set* W ⊆ ˆ <sup>R</sup> d *, we have*

$$\text{gen}(\mu, \mathcal{A}^{(d)}) \leq \inf_{P_{\hat{W}|W', \Theta, S}} \mathbb{E}_{P_{\Theta} P_{\hat{S}}} \left[ \frac{C}{n} \sum_{i \in [n]} \sqrt{2 \text{CMI}_i^{\Theta}(\hat{\mathbf{S}}, \hat{W})} \right] + \epsilon,$$

*and*

$$\text{gen}(\mu, \mathcal{A}^{(d)}) \leq \inf_{P_{\tilde{W}|W', \Theta, S}} \mathbb{E}_{P_{\Theta} P_{\tilde{S}} P_{\mathbf{J}_{-i}}} \left[ \frac{C}{n} \sum_{i \in [n]} \sqrt{2 \text{CMI}_{i, \mathbf{J}_{-i}}^{\Theta}(\tilde{\mathbf{S}}, \hat{W})} \right] + \epsilon, \quad (15)$$

*where* <sup>W</sup><sup>ˆ</sup> <sup>∈</sup> <sup>W</sup><sup>ˆ</sup> *and the infimum are over all Markov kernels* <sup>P</sup>W<sup>ˆ</sup> <sup>|</sup>W′ ,S,<sup>Θ</sup> *that satisfies the following distortion criterion:*

$$\mathbb{E}_{P_S P_\Theta P_{W'|S,\Theta} P_{\hat{W}|W',S,\Theta}} \left[ \text{gen}(S, \Theta W') - \text{gen}(S, \Theta \hat{W}) \right] \leq \epsilon. \quad (16)$$

This bound is used in the following subsection, when SGD or SGLD are used for random subspace training. Note that the above bound includes the case of Wˆ = W′ and ϵ = 0, which results in the lossless bounds of gen(µ, A (d) ) ≤ <sup>E</sup>PS˜P<sup>Θ</sup> C n P i∈[n] q 2CMI<sup>Θ</sup> i (S˜, W′) and gen(µ, A (d) ) ≤ <sup>E</sup>PS˜PΘPJ−<sup>i</sup> h C n P i∈[n] q 2CMI<sup>Θ</sup> i,J−<sup>i</sup> (S˜, W′) i .

The results presented in the next section are extensions and improvements in some aspects, upon previous work on bounding the generalization error of SGLD without projection [\[38,](#page-11-8) [63](#page-13-5)[–67\]](#page-13-6).

#### D.1 Generalization bounds for SGD and SGLD Algorithms

In this section, we consider subspace training algorithms that are trained using an iterative optimization algorithm such as *mini-batch* Stochastic Gradient Descent (SGD) or Stochastic Gradient Langevin dynamics (SGLD).

Let b ∈ N bet the mini-batch size, and let

$$V_t \triangleq \{i_{t,1}, \dots, i_{t,b}\},$$

be the sample indices chosen at time t ∈ [T], *i.e.,* given S˜ ∈ Z<sup>n</sup>×<sup>2</sup> and J = (J1, . . . , Jn), the chosen indices at time t are S˜ <sup>V</sup>t,<sup>J</sup> <sup>≜</sup> <sup>S</sup>˜ <sup>V</sup>t,JVt <sup>≜</sup> n Z<sup>i</sup>t,1,Jit,<sup>1</sup> , . . . , Z<sup>i</sup>t,b,Jit,bo . Furthermore, denote

$$\widehat{\mathcal{R}}(V_t, W) \triangleq \frac{1}{b} \sum_{i \in V_t} \ell(Z_i, J_i, W).$$

We use also the notation V ≜ (V1, . . . , V<sup>T</sup> ) and recall that J−<sup>i</sup> ≜ J[n]\{i}.

The considered noisy iterative optimization algorithm consists of the following steps:

- *(Initialization)* Sample Θ ∈ R D×d and set the initial model's parameters to W<sup>0</sup> = ΘW′ <sup>0</sup>, where W′ <sup>0</sup> ∈ <sup>R</sup> d .
- *(Iterate)* For t ∈ [T], apply the update rule

$$W'_t = \text{Proj} \left\{ W'_{t-1} - \eta_t \nabla_{w'} \widehat{\mathcal{R}}(V_t, \Theta W'_{t-1}) + \sigma_t \varepsilon_t \right\}, \quad (17)$$

with η<sup>t</sup> > 0 (the learning rate), σ<sup>t</sup> ≥ 0 (the variance of the Gaussian noise), and ε<sup>t</sup> ∼ N (0d, Id) (the isotropic Gaussian noise). Here, the projection is an optional operator often used to keep the norm of the model parameters bounded.

• *(Output)* Return the final hypothesis W<sup>T</sup> = ΘW′ T .

Note that here, we train on a subspace of dimension d < D defined by Θ (randomly picked at initialization and fixed during training). Note also that when σ<sup>t</sup> = 0 for all t ∈ [T], this algorithm reduces to the minibatch SGD (with projection).

#### D.1.1 Mutual information of a mixture of two Gaussians and the component

To state our results, we start by defining two useful functions. Suppose that

$$X = (1 - J)Y_1 + JY_2,$$

where (J, Y1, Y2) are independent real-valued random variables defined as follows: J ∼ Bern(p), Y<sup>1</sup> ∼ N (0, 1), and Y<sup>2</sup> ∼ N (a, 1), for some a ∈ <sup>R</sup>. Then, it is easy to show that I(X; J) = f(a, p), where the function f : <sup>R</sup> × [0, 1] → [0, log 2] is defined as[<sup>6</sup>](#page-0-0)

$$f(a, p) \triangleq h(g_{a,p}(x)) - \log(\sqrt{2\pi e}) = -\mathbb{E}_{g_{a,p}(x)} [\log(g_{a,p}(x))] - \log(\sqrt{2\pi e}). \quad (18)$$

Here, ga,p : <sup>R</sup>×[0, 1] → <sup>R</sup><sup>+</sup> is defined as a mixture of two scalar Gaussian distributions with probabilities p and 1 − p:

$$g_{a,p}(x) \triangleq \frac{1}{\sqrt{2\pi}} \left( pe^{-\frac{x^2}{2}} + (1-p)e^{-\frac{(x-a)^2}{2}} \right). \quad (19)$$

The following lemma, proved in the supplements, establishes some properties of the function f(a, p).

Lemma 3. *i) For every* p ∈ [0, 1]*,* f(0, p) = 0*. ii) For every* p ∈ [0, 1]*,* f(a, p) = f(−a, p) *and* f(a, p) *is an strictly increasing function of* a *in the range* [0, ∞)*. iii)* lima→∞ f(a, p) = log(2)hb(p)*. iv) For every* a ∈ <sup>R</sup>*,* f(a, p) = f(a, 1 − p) *and for* a ̸= 0*,* f(a, p) *is strictly increasing with respect to* p *in the range* [0, 1/2]*.*

<sup>6</sup>All logarithms are considered to have the base of e.

#### D.1.2 Lossless generalization bound

We start by stating our bound in its simplest form.

Theorem 10. *Suppose that* ℓ ∈ [0, C]*. Then, the generalization error of a random subspace training algorithm, optimized using iterations defined in [17,](#page-27-2) is upper-bounded as*

$$\text{gen}(\mu, \mathcal{A}^{(d)}) \leq \frac{C\sqrt{2}}{n} \sum_{i \in [n]} \mathbb{E}_{\mathbf{S}, \Theta, \mathbf{V}, \mathbf{J}_{-i}} \left[ \sqrt{\sum_{t: i \in V_t} \mathbb{E}_{p_{t,i}, \Delta_{t,i}} \left[ f\left(\frac{\eta_t}{b\sigma_t} \Delta_{t,i}, p_{t,i}\right) \right]} \right],$$

*where*

$$\begin{aligned} \Delta_{t,i} &\triangleq \|\nabla_{w'} \ell(\Theta W'_{t-1}, Z_{i,0}) - \nabla_{w'} \ell(\Theta W'_{t-1}, Z_{i,1})\|, \\ p_{t,i} &\triangleq \mathbb{P}\left(J_i = 0 \mid \tilde{\mathbf{S}}, \Theta, \mathbf{V}, \mathbf{J}_{-i}, W'_{t-1}, \{W'_r, W'_{r-1} : r < t, i \in V_r\}\right). \end{aligned} \quad (20)$$

This result is proved in Appendix [H.2.](#page-55-0)

In the bound of equation [20,](#page-28-2) the term f η<sup>t</sup> σ<sup>t</sup> <sup>∆</sup>t,i, pt,i is an increasing function with respect to <sup>η</sup><sup>t</sup> σ<sup>t</sup> , ∆t,i, and a decreasing function with respect to |pt,i − 1/2|. As t increases, the learning algorithm "memorizes" more of the dataset; therefore, |pt,i − 1/2| increases and thus these terms decrease. Furthermore, the learning rate decreases, causing this term to decrease more.

Note that by Lemma [3,](#page-27-1) f(·, p) is maximized for p = 1 2 . Hence, a simpler upper bound from Theorem [10](#page-28-0) can be achieved by replacing pt,i by <sup>1</sup> 2 .

#### D.1.3 Lossy generalization bound

The bound of Theorem [10](#page-28-0) has a clear shortcoming; whenever <sup>η</sup><sup>t</sup> σ<sup>t</sup> is very small, the bound becomes loose. In particular, for SGD where σ<sup>t</sup> = 0, the bound becomes vacuous. In this section, to overcome this issue, we consider a lossy version of the above bound. While the lossy bound can be stated without any further assumptions, for a more concrete bound, we make the following assumptions.

Assumption 11 (Lipschitzness). *The loss function is* L*-Lipschitz,* i.e., *for any* w ′ 1, w′ <sup>2</sup> ∈ <sup>R</sup> d *, any* z ∈ Z*, and any* Θ ∈ St(d, D)*, we have* |ℓ z, Θw ′ 1 − ℓ z, Θw ′ 2 | ≤ L∥w ′ <sup>1</sup> − w ′ <sup>2</sup>∥*.*

Note that since Θ <sup>⊤</sup>Θ = Id, then ∥w ′ <sup>1</sup> − w ′ <sup>2</sup>∥ = ∥Θw ′ <sup>1</sup> − Θw ′ <sup>2</sup>∥.

Assumption 12 (Contractivity). *There exists some* α ∈ R <sup>+</sup>*, such that for any* w ′ 1, w′ <sup>2</sup> ∈ W′ *,* z ∈ Z*, and* Θ ∈ St(d, D)*, we have*

$$\|(w'_1 - \eta \nabla_{w'} \ell(z, \Theta w'_1)) - (w'_2 - \eta \nabla_{w'} \ell(z, \Theta w'_2))\| \leq \alpha \|w'_1 - w'_2\|.$$

*Whenever* α < 1*, we say the projected* SGLD *is* α*-contractive.*

Similar assumptions have been used in previous works, such as [\[68\]](#page-13-7). In fact, the contractivity property of SGD has been theoretically proved under certain conditions, such as when the loss function is smooth and strongly convex [\[68](#page-13-7)[–70\]](#page-13-8).

In addition to being sensitive to cases where <sup>η</sup><sup>t</sup> σ<sup>t</sup> is very small, the bound of Theorem [10](#page-28-0) does not account for the "forgetting" effect of the iterative optimization algorithms: the information obtained by W′ <sup>T</sup> about J<sup>i</sup> in the initial iterations will eventually fade out, as T increases. To account for this effect, similar to [\[66,](#page-13-9) [67\]](#page-13-6), we assume that W′ = BD(R), [7](#page-0-0) for some R ∈ <sup>R</sup>+.

Theorem 13. *Suppose that* ℓ ∈ [0, C]*,* W′ = BD(R)*, for some* R ∈ R+*, and Assumptions [11](#page-28-3) and [12](#page-28-4) hold with constants* L ∈ <sup>R</sup><sup>+</sup> *and* α ≤ 1*, respectively. Then, for any set of* {νt}t∈[<sup>T</sup> ] *, such that* ν<sup>t</sup> ∈ <sup>R</sup>+*, the generalization error of a random subspace training algorithm, optimized using iterations defined in [17,](#page-27-2) is upper bounded as*

$$\begin{aligned} \text{gen}(\mu, \mathcal{A}^{(d)}) &\leq \frac{C\sqrt{2}}{n} \sum_{i \in [n]} \mathbb{E}_{\tilde{\mathbf{S}}, \Theta, \mathbf{V}, \mathbf{J}_{-i}} \left[ \sqrt{\sum_{i: i \in V_t} A_{t,i} \mathbb{E}_{\hat{p}_{t,i}, \hat{\Delta}_{t,i}} \left[ f\left(\frac{\eta_t}{b\hat{\sigma}_t} \hat{\Delta}_{t,i}, \hat{p}_{t,i}\right) \right]} \right] \\ &\quad + \frac{2\sqrt{2}\mathcal{L}\Gamma\left(\frac{d+1}{2}\right)}{\Gamma\left(\frac{d}{2}\right)} \sum_{t \in [T]} \nu_t \alpha^{T-t}, \end{aligned} \quad (21)$$

<sup>7</sup> In this setup, for w ′ ∈ W′ , Proj {w ′ } = w ′ and otherwise Proj {w ′ } = <sup>∥</sup>w′∥w ′ .

*where*

$$\begin{aligned} \hat{\Delta}_{t,i} &\triangleq \left\| \nabla_{w'} \ell \left( \Theta \hat{W}_{t-1}, Z_{i,0} \right) - \nabla_{w'} \ell \left( \Theta \hat{W}_{t-1}, Z_{i,1} \right) \right\|, \\ \hat{p}_{t,i} &\triangleq \mathbb{P} \left( J_i = 0 \mid \tilde{\mathbf{S}}, \Theta, \mathbf{V}, \mathbf{J}_{-i}, \hat{W}_{t-1} \right), \\ \hat{\sigma}_t &\triangleq \sqrt{\sigma_t^2 + \nu_t^2}, \\ q_t &\triangleq 1 - 2\Phi \left( \frac{R + \eta_t \mathcal{L}}{\hat{\sigma}_t} \right), \\ A_{t,i} &\triangleq \prod_{r \in [t+1:T]: i \notin V_r} q_r, \end{aligned}$$

*where* Wˆ <sup>t</sup> *are random variables that satisfy*

$$\left\| \hat{W}_t - W'_t \right\| \leq \sum_{r \in [t]} \alpha^{t-r} \nu_r \left\| \varepsilon'_r \right\|,$$

*for* ε ′ <sup>t</sup> ∼ N (0d, Id)*, which is an auxiliary additional noise, independent of all other random variables, and where* Φ(x) ≜ R <sup>∞</sup> x √1 2π exp(−y 2 /2)dy *is the Gaussian complementary cumulative distribution function (CCDF).*

This theorem is proved in Appendix [H.3.](#page-58-0) Here, we discuss some remarks.

First, the "gained" information from the initial iterations fades as T → ∞, when q<sup>t</sup> < 1 (note that always q<sup>t</sup> ≤ 1).

Second, we note that, unlike in Theorem [10,](#page-28-0) where pt,i depends on all past iterations in which sample i is used, in this theorem, pˆt,i depends only on the immediate past iteration. It can be shown that a similar result can be achieved for Theorem [13](#page-28-1) *i.e.,* allowing pˆt,i to depend on all past iterations, at the expense of replacing all {qt}<sup>t</sup> by 1.

Third, it can be observed that if ∀t ∈ [T]: ν<sup>t</sup> = 0, we recover Theorem [10,](#page-28-0) except for the definition of pt,i, that can be adjusted at the expense of replacing all {qt}<sup>t</sup> by 1, as explained above. Furthermore, by increasing νt, the second term in equation [21,](#page-28-5) i.e. the "distortion" term, increases; but the first "rate" term decreases since f √ <sup>η</sup><sup>t</sup> σ<sup>2</sup> <sup>t</sup> +ν ∆ˆ t,i, <sup>p</sup>ˆt,i decreases. Therefore, in general, the lossy bound can outperform the lossless bound. In particular, for SGD, *i.e.,* when σ<sup>t</sup> = 0, the lossless bound and previous works (for the case of no projection) [\[38,](#page-11-8) [63–](#page-13-5)[67\]](#page-13-6) become vacuous, while the lossy bound does not.

Lastly, to achieve this bound, we considered a sequence of parallel "perturbed" iterations. In each of these auxiliary iterations, we introduced an additional independent noise νtε ′ t , where ε ′ <sup>t</sup> ∼ N (0d,Id). It can be seen that for the contractive SGD/SGLD, the effect of added perturbation in the initial iterations vanishes as T → ∞. Therefore, once again, it can be seen that the effect of the increase in mutual information from the initial iterations eventually fades.

# E Proof of Theorem [1](#page-3-0)

We prove the theorem in its most general form stated in Remark [1.](#page-22-6) This means that we assume that the learning algorithm A is also aware of the projection matrix Θ, *i.e.,* A: Z <sup>n</sup> × <sup>R</sup> <sup>D</sup>×<sup>d</sup> → W takes both the dataset S<sup>n</sup> and the projection matrix Θ as input to learn W. Moreover, we allow the quantization step to depend on S, Θ, and A(Sn, Θ). In this general case, Wˆ = Aˆ(S, Θ) = A˜(Θ, Sn, A(S, Θ)). We denote this general compressed algorithm by PW<sup>ˆ</sup> <sup>|</sup>Sn,W,<sup>Θ</sup>. Note that PW<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> is a special case of this more general setup.

Fix some <sup>ϵ</sup> <sup>∈</sup> <sup>R</sup> and the quantization set <sup>W</sup><sup>ˆ</sup> . Consider any Markov kernel <sup>P</sup>W<sup>ˆ</sup> <sup>|</sup>Sn,W,<sup>Θ</sup> and <sup>P</sup><sup>Θ</sup> that satisfy the following distortion criterion:

$$\mathbb{E}_{P_{S_n} P_{\Theta} P_{W, \hat{W} | S_n, \Theta}} \left[ \text{gen}(S_n, W) - \text{gen}(S_n, \Theta \hat{W}) \right] \leq \epsilon.$$

Using this condition, it is sufficient to show that

$$\text{gen}(\mu, \hat{\mathcal{A}}) = \mathbb{E}_{P_{S_n} P_\Theta P_{W, \hat{W} | S_n, \Theta}} \left[ \text{gen}(S_n, \Theta \hat{W}) \right] \leq \mathbb{E}_{P_{\tilde{S}} P_\Theta} \left[ \sqrt{\frac{2\Delta \ell_{\hat{W}}(\tilde{\mathbf{S}}, \Theta)}{n}} \text{CMI}^\Theta(\tilde{\mathbf{S}}, \hat{\mathcal{A}}) \right],$$

where

$$\Delta \ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta) := \mathbb{E}_{P_{W|\tilde{\mathbf{S}}, \Theta}} \left[ \frac{1}{n} \sum_{i \in [n]} (\ell(Z_{i,0}, \Theta \hat{W}) - \ell(Z_{i,1}, \Theta \hat{W}))^2 \right].$$

Denote the marginal distribution of (Sn, <sup>Θ</sup>, <sup>W</sup><sup>ˆ</sup> ) under <sup>P</sup>S<sup>n</sup> <sup>P</sup>ΘPW,W<sup>ˆ</sup> <sup>|</sup>Sn,<sup>Θ</sup> by <sup>P</sup>Sn,Θ,W<sup>ˆ</sup> and conditional distribution of <sup>W</sup><sup>ˆ</sup> given (Sn, Θ) by <sup>P</sup>W<sup>ˆ</sup> <sup>|</sup>Sn,Θ. Hence, <sup>P</sup>Sn,Θ,W<sup>ˆ</sup> <sup>=</sup> <sup>P</sup>S<sup>n</sup> <sup>P</sup>ΘPW<sup>ˆ</sup> <sup>|</sup>Sn,<sup>Θ</sup> and

$$\begin{aligned} \text{gen}(\mu, \hat{A}) &= \mathbb{E}_{P_{S_n} P_\Theta P_{\hat{W}|S_n, \Theta}} \left[ \text{gen}(S_n, \Theta \hat{W}) \right] \\ &= \mathbb{E}_{P_{\tilde{S}} P_\Theta P_{J_{\hat{W}|\tilde{S}_n, \Theta}}} \left[ \hat{\mathcal{R}}(\tilde{\mathbf{S}}_J^c, \Theta \hat{W}) - \hat{\mathcal{R}}(\tilde{\mathbf{S}}_J, \Theta \hat{W}) \right]. \end{aligned}$$

It is hence sufficient to show that for any S˜ and Θ,

$$\mathbb{E}_{P_{\mathbf{J}} P_{W|\tilde{\mathbf{S}}_{\mathbf{J}}, \Theta} \left[ \widehat{\mathcal{R}}(\tilde{\mathbf{S}}_{\mathbf{J}^c}, \Theta \widehat{W}) - \widehat{\mathcal{R}}(\tilde{\mathbf{S}}_{\mathbf{J}}, \Theta \widehat{W}) \right] \leq \sqrt{\frac{2 \Delta \ell_{\widehat{W}}(\tilde{\mathbf{S}}, \Theta)}{n} \text{CMI}^{\Theta}(\tilde{\mathbf{S}}, \hat{\mathcal{A}})}.$$

Denote <sup>P</sup>W<sup>ˆ</sup> <sup>|</sup>S˜,<sup>Θ</sup> <sup>≜</sup> <sup>E</sup>P<sup>J</sup> h PW<sup>ˆ</sup> <sup>|</sup>S˜J,<sup>Θ</sup> i and <sup>P</sup>J,W<sup>ˆ</sup> <sup>|</sup>S˜,<sup>Θ</sup> <sup>≜</sup> <sup>P</sup>JPW<sup>ˆ</sup> <sup>|</sup>S˜J,<sup>Θ</sup> <sup>≜</sup> <sup>P</sup>J|S˜,Θ,W<sup>ˆ</sup> <sup>P</sup>W<sup>ˆ</sup> <sup>|</sup>S˜,<sup>Θ</sup> be the conditional distributions of (J, <sup>W</sup><sup>ˆ</sup> ) given (S˜, Θ). Note that the marginal distribution of <sup>J</sup> under <sup>P</sup>J,W<sup>ˆ</sup> <sup>|</sup>S˜,<sup>Θ</sup> is PJ, *i.e.,*

$$\mathbb{E}_{P_{\hat{W}|\tilde{\mathbf{s}}, \Theta}} \left[ P_{\mathbf{J}|\tilde{\mathbf{s}}, \Theta, \hat{W}} \right] = P_{\mathbf{J}}.$$

Now, fix some λ ̸= 0 that will be determined later. We have

$$\begin{aligned} & \mathbb{E}_{P_{\mathbf{J}|\tilde{\mathbf{S}}, \hat{W}, \Theta}} \left[ \hat{\mathcal{R}}(\tilde{\mathbf{S}}_{\mathbf{J}^c}, \Theta \hat{W}) - \hat{\mathcal{R}}(\tilde{\mathbf{S}}_{\mathbf{J}}, \Theta \hat{W}) \right] \\ & \stackrel{(a)}{\leq} \frac{1}{\lambda} D_{KL} \left( P_{\mathbf{J}|\tilde{\mathbf{S}}, \hat{W}, \Theta} \|P_{\mathbf{J}}\| + \frac{1}{\lambda} \log \left( \mathbb{E}_{P_{\mathbf{J}}} \left[ e^{\lambda(\hat{\mathcal{R}}(\tilde{\mathbf{S}}_{\mathbf{J}^c}, \Theta \hat{W}) - \hat{\mathcal{R}}(\tilde{\mathbf{S}}_{\mathbf{J}}, \Theta \hat{W}))} \right] \right) \right) \\ & = \frac{1}{\lambda} D_{KL} \left( P_{\mathbf{J}|\tilde{\mathbf{S}}, \hat{W}, \Theta} \|P_{\mathbf{J}}\| + \frac{1}{\lambda} \log \left( \mathbb{E}_{P_{\mathbf{J}}} \left[ e^{\frac{\lambda}{n} \sum_{i \in [n]} (-1)^{J_i} (\ell(Z_{i,0}, \Theta \hat{W}) - \ell(Z_{i,1}, \Theta \hat{W}))} \right] \right) \right) \\ & \stackrel{(b)}{\leq} \frac{1}{\lambda} D_{KL} \left( P_{\mathbf{J}|\tilde{\mathbf{S}}, \hat{W}, \Theta} \|P_{\mathbf{J}}\| + \frac{1}{\lambda} \sum_{i \in [n]} \frac{\lambda^2 (\ell(Z_{i,0}, \Theta \hat{W}) - \ell(Z_{i,1}, \Theta \hat{W}))^2}{2n^2} \right). \end{aligned}$$

where (a) follows from Donsker-Varadhan's inequality and (b) by the inequality <sup>1</sup> 2 (e <sup>−</sup><sup>x</sup> + e x ) ≤ e x <sup>2</sup>/2 . Hence,

$$\begin{aligned} & \mathbb{E}_{P_{\mathbf{J}} P_{\hat{W}|\tilde{\mathbf{S}}_{\mathbf{J}}, \Theta} \left[ \hat{\mathcal{R}}(\tilde{\mathbf{S}}_{\mathbf{J}^c}, \Theta\hat{W}) - \hat{\mathcal{R}}(\tilde{\mathbf{S}}_{\mathbf{J}}, \Theta\hat{W}) \right] \\ &= \mathbb{E}_{P_{\hat{W}|\tilde{\mathbf{S}}, \Theta} P_{\mathbf{J}|\tilde{\mathbf{S}}, \hat{W}, \Theta}} \left[ \hat{\mathcal{R}}(\tilde{\mathbf{S}}_{\mathbf{J}^c}, \Theta\hat{W}) - \hat{\mathcal{R}}(\tilde{\mathbf{S}}_{\mathbf{J}}, \Theta\hat{W}) \right] \\ &\leq \frac{1}{\lambda} D_{KL} \left( P_{\hat{W}|\tilde{\mathbf{S}}, \Theta} P_{\mathbf{J}|\tilde{\mathbf{S}}, \hat{W}, \Theta} \|P_{\mathbf{J}} P_{\hat{W}|\tilde{\mathbf{S}}, \Theta}\right) \\ &\quad + \frac{\lambda}{2n} \mathbb{E}_{P_{\hat{W}|\tilde{\mathbf{S}}, \Theta}} \left[ \frac{1}{n} \sum_{i \in [n]} (\ell(Z_{i,0}, \Theta\hat{W}) - \ell(Z_{i,1}, \Theta\hat{W}))^2 \right] \\ &= \frac{1}{\lambda} D_{KL} \left( P_{\hat{W}|\tilde{\mathbf{S}}, \Theta} P_{\mathbf{J}|\tilde{\mathbf{S}}, \hat{W}, \Theta} \|P_{\mathbf{J}} P_{\hat{W}|\tilde{\mathbf{S}}, \Theta}\right) + \frac{\lambda \Delta \ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta)}{2n} \\ &= \frac{1}{\lambda} D_{KL} \left( P_{\mathbf{J} P_{\hat{W}|\tilde{\mathbf{S}}_{\mathbf{J}}, \Theta}} \|P_{\mathbf{J}} P_{\hat{W}|\tilde{\mathbf{S}}, \Theta}\right) + \frac{\lambda \Delta \ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta)}{2n} \\ &= \frac{1}{\lambda} \text{CMI}^\Theta(\tilde{\mathbf{S}}, \hat{\mathcal{A}}) + \frac{\lambda \Delta \ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta)}{2n} \\ &\leq \sqrt{\frac{2 \Delta \ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta) \text{CMI}^\Theta(\tilde{\mathbf{S}}, \hat{\mathcal{A}})}{n}}, \end{aligned}$$

where the last step is followed by letting

$$\lambda \triangleq \sqrt{\frac{2nCMI^\Theta(\tilde{\mathbf{S}}, \hat{\mathcal{A}})}{\Delta \ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta)}}.$$

# F Proofs of Section [4](#page-4-0) and Appendix [B:](#page-22-0) Application to raised limitations of CMI bounds

For the proofs of Section [4](#page-4-0) and Appendix [B,](#page-22-0) we always consider the normalized setup, *i.e.,* R = 1, L = 1 (for Theorem [3\)](#page-5-1), L<sup>c</sup> = 1 (for Proposition [1\)](#page-6-1), and B = 1 (for Theorem [7\)](#page-23-3). The proof applies for arbitrary values of (R, L, Lc, B), by simply scaling the constants.

All proofs are based on Theorem [1,](#page-3-0) with a particular class of choices of P<sup>Θ</sup> and PW<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> , called the choices from the scheme JL(d, cw, ν) for some d ∈ N, c<sup>w</sup> ∈ h 1, p 5/4 , and ν ∈ (0, 1], described in Appendix [F.1.](#page-31-1) For a given JL(d, cw, ν), we then use Theorem [1](#page-3-0) for some suitable ϵ ∈ <sup>R</sup>:

$$\text{gen}(\mu, \mathcal{A}) \leq \mathbb{E}_{P_{\tilde{S}} P_\Theta} \left[ \sqrt{\frac{2\Delta \ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta)}{n}} \text{CMI}^\Theta(\tilde{\mathbf{S}}, \hat{\mathcal{A}}) \right] + \epsilon. \quad (22)$$

Recall that the term ∆ℓwˆ(S˜, Θ) is defined as

$$\Delta \ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta) := \mathbb{E}_{P_{W|\tilde{\mathbf{S}}} P_{\hat{W}|\Theta^\top W}} \left[ \frac{1}{n} \sum_{i \in [n]} (\ell(Z_{i,0}, \Theta \hat{W}) - \ell(Z_{i,1}, \Theta \hat{W}))^2 \right],$$

and the choices of P<sup>Θ</sup> and PW<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> should satisfy the distortion criterion

$$\mathbb{E}_{P_{S_n, W} P_{\Theta} P_{\hat{W}|\Theta^\top W}} \left[ \text{gen}(S_n, W) - \text{gen}(S_n, \Theta \hat{W}) \right] \leq \epsilon. \quad (23)$$

For brevity, we often use the notation

$$\Delta(W, \Theta\hat{W}; S_n) := \text{gen}(S_n, W) - \text{gen}(S_n, \Theta\hat{W}).$$

Furthermore, denote the D-dimensional ball of radius ν ∈ <sup>R</sup><sup>+</sup> and center w ∈ <sup>R</sup> <sup>D</sup> by BD(w, ν). If w = 0D, for simplicity we write BD(0D, ν) ≡ BD(ν), where 0<sup>D</sup> designates the all-zero vector in <sup>R</sup> D.

#### F.1 Johnson-Lindenstrauss projection scheme

Fix some constant c<sup>w</sup> ∈ 1, q5 4 and ν ∈ (0, 1]. Let d ∈ N ∗ and Θ be a matrix of size D × d whose elements are i.i.d. samples from N (0, 1/d). For a given Θ and W = A(Sn), in the scheme JL(d, cw, ν), let

$$U := \begin{cases} \Theta^\top W, & \text{if } \|\Theta^\top W\| \leq c_w, \\ \mathbf{0}_d, & \text{otherwise.} \end{cases} \quad (24)$$

Let V<sup>ν</sup> be a random variable that takes value uniformly over B<sup>d</sup> (ν). Let Wˆ ∈ Wˆ = Bd(c<sup>w</sup> + ν) be defined as

$$\hat{W} = U + V_\nu. \quad (25)$$

This means that Wˆ is a random variable that takes value uniformly over B<sup>d</sup> (U, ν):

$$\hat{W} \sim \text{Unif}(\mathcal{B}_d(U, \nu)).$$

In other words, we define Wˆ as a quantization of W′ = Θ⊤W obtained as follows: if ∥Θ <sup>⊤</sup>W∥ ≤ cw, then Wˆ is uniformly sampled from B<sup>d</sup> Θ <sup>⊤</sup>W, ν ; otherwise, Wˆ is uniformly sampled from B<sup>d</sup> (ν). Such quantization has been previously used in [\[22\]](#page-11-1) to establish a generalization bound for the distributed SVM learning algorithm.

Disintegrated CMI bound: The disintegrated CMI bound CMI<sup>Θ</sup>(S˜, Aˆ) in the scheme JL(d, cw, ν) can be upper bounded as

$$\begin{aligned}
\text{CMI}^\Theta(\tilde{\mathbf{S}}, \hat{\mathcal{A}}) &= h^{\tilde{\mathbf{S}}, \Theta}(\hat{W}) - h^{\tilde{\mathbf{S}}, \Theta}(\hat{W} | \mathbf{J}) \\
&\stackrel{(a)}{\leq} h^{\tilde{\mathbf{S}}, \Theta}(\hat{W}) - h^{\tilde{\mathbf{S}}, \Theta}(\hat{W} | \mathbf{J}, W) \\
&\stackrel{(b)}{=} h^{\tilde{\mathbf{S}}, \Theta}(\hat{W}) - h(\hat{W} | \Theta^\top W) \\
&\stackrel{(c)}{\leq} \log(\text{Volume}(\mathcal{B}_d(c_w + \nu))) - \log(\text{Volume}(\mathcal{B}_d(\nu))) \\
&= d \log\left(\frac{c_w + \nu}{\nu}\right), \tag{26}
\end{aligned}$$

where

- h <sup>S</sup>˜,<sup>Θ</sup>(W<sup>ˆ</sup> ) is the differential entropy of <sup>W</sup><sup>ˆ</sup> <sup>∼</sup> <sup>P</sup>W<sup>ˆ</sup> <sup>|</sup>S˜,Θ, <sup>h</sup> <sup>S</sup>˜,<sup>Θ</sup>(W<sup>ˆ</sup> |J) = <sup>E</sup><sup>J</sup> h h S˜,Θ,J (Wˆ ) i , and h S˜,Θ,J (W<sup>ˆ</sup> ) is the differential entropy of <sup>W</sup><sup>ˆ</sup> <sup>∼</sup> <sup>P</sup>W<sup>ˆ</sup> <sup>|</sup>S˜,Θ,<sup>J</sup> ,
- (a) follows from the fact that conditioning does not increase the entropy,
- (b) yields due to Markov chain Wˆ − Θ <sup>⊤</sup>W − (S˜, Θ, J, W),
- and (c) holds since i) ∥Wˆ ∥ ≤ c<sup>w</sup> + ν by construction and hence h <sup>S</sup>˜,<sup>Θ</sup>(W<sup>ˆ</sup> ) is upper bounded by the differential entropy of a random variable taking value uniformly over Bd(c<sup>w</sup> + ν), and ii) since given Θ <sup>⊤</sup>W, Wˆ is chosen uniformly over a d-dimensional ball either around 0<sup>d</sup> or Θ <sup>⊤</sup>W, depending on ∥Θ <sup>⊤</sup>W∥.

#### F.2 Proof of Theorem [3](#page-5-1)

As explained in Appendix [F,](#page-31-0) we consider the case L = R = 1, and use Theorem [1](#page-3-0) using the JL(d, cw, ν) transformation described in Appendix [F.1,](#page-31-1) with some d ∈ N <sup>+</sup>, c<sup>w</sup> ∈ h 1, p 5/4 , and ν ∈ (0, 1]. To do so, we start by bounding CMI<sup>Θ</sup>(S˜, <sup>A</sup>ˆ), the distortion equation [23,](#page-31-2) and <sup>E</sup>S˜,<sup>Θ</sup>[∆ℓwˆ(S˜, Θ)].

Bound on the disintegrated CMI: It is shown in equation [26](#page-31-3) that

$$\text{CMI}^\Theta(\tilde{\mathbf{S}}, \hat{\mathcal{A}}) \leq d \log \left( \frac{c_w + \nu}{\nu} \right). \quad (27)$$

Bound on the distortion: Next, we bound the distortion term. By definition, and using the linearity of expectation, we obtain

$$\begin{aligned} \Delta(W, \Theta\hat{W}; S_n) &= \text{gen}(S_n, W) - \text{gen}(\Theta\hat{W}) \\ &= \mathbb{E}_{Z \sim \mu}[-\langle W, Z \rangle] + \frac{1}{n} \sum_{i=1}^n \langle W, Z_i \rangle + \mathbb{E}_{Z \sim \mu}[\langle \Theta\hat{W}, Z \rangle] - \frac{1}{n} \sum_{i=1}^n \langle \Theta\hat{W}, Z_i \rangle \\ &= -\langle W, \mathbb{E}_{Z \sim \mu}[Z] - \frac{1}{n} \sum_{i=1}^n Z_i \rangle + \langle \Theta\hat{W}, \mathbb{E}_{Z \sim \mu}[Z] - \frac{1}{n} \sum_{i=1}^n Z_i \rangle \\ &= -\langle W, \bar{Z} \rangle + \langle \hat{W}, \Theta^\top \bar{Z} \rangle, \end{aligned} \tag{28}$$

where Z¯ <sup>≜</sup> <sup>E</sup>Z∼µ[Z] − n P<sup>n</sup> <sup>i</sup>=1 Zi.

Additionally, since for any (x, y) ∈ <sup>R</sup> <sup>D</sup> × <sup>R</sup> <sup>D</sup>, <sup>E</sup>Θ[⟨Θ <sup>⊤</sup>x, Θ <sup>⊤</sup>y⟩] = ⟨x, y⟩, then

$$\begin{aligned}\mathbb{E}_{\hat{W}, \Theta, W, S_n} [\Delta(W, \Theta \hat{W}; S_n)] &= \mathbb{E}_{\hat{W}, \Theta, W, S_n} [-\langle W, \bar{Z} \rangle + \langle \hat{W}, \Theta^\top \bar{Z} \rangle] \\ &= \mathbb{E}_{\hat{W}, \Theta, W, S_n} [-\langle \Theta^\top W, \Theta^\top \bar{Z} \rangle + \langle \hat{W}, \Theta^\top \bar{Z} \rangle] \\ &= \mathbb{E}_{\hat{W}, \Theta, W, S_n} [\langle \hat{W} - \Theta^\top W, \Theta^\top \bar{Z} \rangle].\end{aligned}$$

Let E be the event that ∥Θ <sup>⊤</sup>W∥ > c<sup>w</sup> and denote by E c the complementary event of E. By the law of total expectation,

$$\mathbb{E}_{\hat{W}, \Theta, W, S_n} [\Delta(W, \Theta\hat{W}; S_n)] = \mathbb{E}[\langle \hat{W} - \Theta^\top W, \Theta^\top \bar{Z} \rangle \mid \mathcal{E}] \mathbb{P}(\mathcal{E}) + \mathbb{E}[\langle \hat{W} - \Theta^\top W, \Theta^\top \bar{Z} \rangle \mid \mathcal{E}^c] \mathbb{P}(\mathcal{E}^c). \quad (29)$$

By definition of Wˆ , <sup>E</sup>[Wˆ ] = 0 under E, <sup>E</sup>[Wˆ ] = Θ<sup>⊤</sup>W otherwise. Therefore, equation [29](#page-32-1) can be simplified as

$$\begin{aligned}\mathbb{E}_{\hat{W}, \Theta, W, S_n} [\Delta(W, \Theta \hat{W}; S_n)] &= \mathbb{E}[-\langle \Theta^\top W, \Theta^\top \bar{Z} \rangle \mid \mathcal{E}] \mathbb{P}(\mathcal{E}) \\ &= \mathbb{E}[-\langle \Theta^\top W, \Theta^\top \bar{Z} \rangle \mathbb{1}\{\mathcal{E}\}] \\ &\leq \mathbb{E}[\|\Theta^\top W\| \|\Theta^\top \bar{Z}\| \mathbb{1}\{\mathcal{E}\}] \\ &\leq \mathbb{E}[\|\Theta^\top \bar{Z}\|^2]^{1/2} \mathbb{E}[\|\Theta^\top W\|^4]^{1/4} \mathbb{E}[\mathbb{1}\{\mathcal{E}\}]^{1/4}, \end{aligned} \tag{30}$$

where equation [30](#page-32-2) follows from Cauchy-Schwarz inequality, and equation [31](#page-32-3) results from Holder's in- ¨ equality.

- Since the elements of Θ ∈ R D×d are i.i.d. from N (0, 1/d), then for any fixed vector x ∈ R <sup>D</sup>, each entry of √ dΘ⊤x ∥x∥ is an independent random variable distributed according to N (0, 1). Hence, V<sup>x</sup> = √ dΘ⊤x ∥x∥ 2 is a chi-squared random variable with d-degrees of freedom, and we have

$$\mathbb{E}[V_x] = d.$$

This concludes that for any z¯,

$$\mathbb{E} \left[ \|\Theta^\top \bar{z}\|^2 \right] = \|\bar{z}\|^2.$$

- Moreover, since V<sup>x</sup> is a chi-squared distribution with d-degrees of freedom, we have that

$$\mathbb{E}[V_x^2] = \mathbb{E}[V_x]^2 + \mathbb{E}[(V_x - \mathbb{E}[V_x])^2] = d^2 + 2d.$$

Hence for every w ∈ W,

$$\begin{aligned}\mathbb{E}_{\Theta} \left[ \|\Theta^\top w\|^4 \right] &= \frac{\|w\|^4}{d^2} \mathbb{E}_{\Theta} \left[ \left\| \frac{\sqrt{d}\Theta^\top w}{\|w\|} \right\|^4 \right] \\ &= \frac{\|w\|^4}{d^2} \mathbb{E}_{\Theta} \left[ V_w^2 \right] \\ &= \left( 1 + \frac{2}{d} \right) \|w\|^4.\end{aligned}$$

- By [\[71,](#page-13-10) Lemma 9], for any w ∈ BD(1), if c<sup>w</sup> ∈ [1, √ 5 2 ),

$$\mathbb{P}(\mathcal{E}) \leq e^{-0.21d(c_w^2-1)^2}. \quad (32)$$

More precisely by [\[71,](#page-13-10) Lemma 9] we have for any t ∈ [0, 1/4) and any w ∈ Bd(1),

$$\mathbb{P}\left(\|\Theta^\top w\|^2 - \|w\|^2 > t\|w\|^2\right) \leq e^{-0.21dt^2}$$

.

We note that this inequality is a "single-sided" tail bound version of [\[71,](#page-13-10) Lemma 9] (while therein stated as a "double-sided" tail bound). This explains why RHS of the inequality in [\[71,](#page-13-10) Lemma 9] is 2e −0.21dt<sup>2</sup> , while here we have e −0.21dt<sup>2</sup> .

Next, note that (t + 1)∥w∥ <sup>2</sup> ≤ (t + 1), hence

$$\mathbb{P} \left( \|\Theta^\top w\|^2 - \|w\|^2 > t\|w\|^2 \right) = \mathbb{P} \left( \|\Theta^\top w\|^2 > (t+1)\|w\|^2 \right) \geq \mathbb{P} \left( \|\Theta^\top w\|^2 > t+1 \right).$$

Thus, by letting t = c 2 <sup>w</sup> − 1 for c<sup>w</sup> ∈ [1, p 5/4), we have

$$\mathbb{P}\left(\|\Theta^\top w\| \geq c_w\right) \leq e^{-0.21dt^2} = e^{-0.21d(c_w^2-1)^2}.$$

Combining the above upper bounds on <sup>E</sup>[∥Θ <sup>⊤</sup>Z¯∥ ], <sup>E</sup>[∥Θ <sup>⊤</sup>W∥ 4 ], and <sup>E</sup>[1{E}], we obtain,

$$\begin{aligned}\mathbb{E}_{\hat{W},\Theta,W,S_n}[\Delta(W,\Theta\hat{W};S_n)] &\leq \mathbb{E}[\|\bar{Z}\|^2]^{1/2} \mathbb{E}[\|\Theta^\top W\|^4]^{1/4} e^{-\frac{0.21}{4}d(c_w^2-1)^2} \\ &\leq \mathbb{E}[\|\bar{Z}\|^2]^{1/2} \mathbb{E}_W[\|W\|^2 + \frac{2}{d}\|W\|^4]^{1/4} e^{-\frac{0.21}{4}d(c_w^2-1)^2} \\ &\leq \mathbb{E}[\|\bar{Z}\|^2]^{1/2} \left(1 + \frac{2}{d}\right)^{1/4} e^{-\frac{0.21}{4}d(c_w^2-1)^2},\end{aligned}\tag{33}$$

It remains then to upper bound <sup>E</sup>[∥Z¯∥ 2 ]. By definition of Z¯ and the linearity of expectation, we have

$$\begin{aligned}\mathbb{E}[\|\bar{Z}\|^2] &= \mathbb{E}[\|\mathbb{E}[Z] - \frac{1}{n} \sum_{i=1}^n Z_i\|^2] \\ &= \mathbb{E}[\|\frac{1}{n} \sum_{i=1}^n (\mathbb{E}[Z] - Z_i)\|^2] \\ &= \frac{1}{n^2} \mathbb{E} \left[ \left( \sum_{i=1}^n (\mathbb{E}[Z] - Z_i) \right)^\top \left( \sum_{j=1}^n (\mathbb{E}[Z] - Z_j) \right) \right] \\ &= \frac{1}{n^2} \mathbb{E} \left[ \sum_{i=1}^n \sum_{j=1}^n (\mathbb{E}[Z] - Z_i)^\top (\mathbb{E}[Z] - Z_j) \right] \\ &= \frac{1}{n^2} \mathbb{E} \left[ \sum_{i=1}^n (\mathbb{E}[Z] - Z_i)^\top (\mathbb{E}[Z] - Z_i) + \sum_{i \neq j} (\mathbb{E}[Z] - Z_i)^\top (\mathbb{E}[Z] - Z_j) \right] \\ &= \frac{1}{n^2} \mathbb{E} \left[ \sum_{i=1}^n \|\mathbb{E}[Z] - Z_i\|^2 + \sum_{i \neq j} \text{Cov}(Z_i, Z_j) \right] \\ &= \frac{1}{n^2} \mathbb{E} \left[ \sum_{i=1}^n \|\mathbb{E}[Z] - Z_i\|^2 \right] \tag{34}\end{aligned}$$

$$\leq \frac{4}{n}, \quad (35)$$

where equation [34](#page-34-0) results from Cov(Zi, Z<sup>j</sup> ) = 0 for i ̸= j since Zi, Z<sup>j</sup> are independent, and equation [35](#page-34-1) follows from Z ⊆ BD(1) (thus, for any i, ∥E[Z] − Zi∥ ≤ 2).

Combining equation [33](#page-33-0) and equation [34,](#page-34-0) we conclude that the distortion is bounded by

$$\mathbb{E}_{\hat{W}, \Theta, W, S_n} [\Delta(W, \Theta \hat{W}; S_n)] \leq \frac{2}{\sqrt{n}} \left(1 + \frac{2}{d}\right)^{1/4} e^{-\frac{0.21}{4} d(c_w^2 - 1)^2}. \quad (36)$$

Bound on <sup>E</sup>S˜,<sup>Θ</sup>[∆ℓwˆ(S˜, Θ)]: We have

$$\begin{aligned}\mathbb{E}_{P_{\tilde{S}} P_{\Theta}}[\Delta \ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta)] &:= \mathbb{E}_{P_{\tilde{S}} P_{\Theta} P_{W|\tilde{S}} P_{\hat{W}|\Theta^{\top} W}} \left[ \frac{1}{n} \sum_{i \in [n]} (\ell(Z_{i,0}, \Theta \hat{W}) - \ell(Z_{i,1}, \Theta \hat{W}))^2 \right] \\ &= \mathbb{E}_{P_{\tilde{S}} P_{\Theta} P_{W|\tilde{S}} P_{\hat{W}|\Theta^{\top} W}} \left[ \frac{1}{n} \sum_{i \in [n]} \left\langle \hat{W}, \Theta^{\top} (Z_{i,0} - Z_{i,1}) \right\rangle^2 \right] \\ &\stackrel{(a)}{\leq} \mathbb{E}_{P_{\tilde{S}} P_{\Theta} P_{W|\tilde{S}} P_{\hat{W}|\Theta^{\top} W}} \left[ \frac{1}{n} \sum_{i \in [n]} \|\Theta^{\top} (Z_{i,0} - Z_{i,1})\|^2 \|\hat{W}\|^2 \right] \\ &\stackrel{(b)}{\leq} (c_w + \nu)^2 \mathbb{E}_{P_{\tilde{S}} P_{\Theta}} \left[ \frac{1}{n} \sum_{i \in [n]} \|\Theta^{\top} (Z_{i,0} - Z_{i,1})\|^2 \right] \\ &\stackrel{(c)}{\leq} 4(c_w + \nu)^2,\end{aligned} \tag{37}$$

where (a) follows by Cauchy–Schwarz inequality, (b) is derived since ∥wˆ∥ ≤ (c<sup>w</sup> + ν), and (c) since for any fixed <sup>z</sup>, each entry of <sup>Θ</sup>⊤<sup>z</sup> ∥z∥ is an independent random variable distributed according to N (0, 1 d ) and hence

$$\mathbb{E}_\Theta \left[ \left\| \Theta^\top z \right\|^2 \right] = \|z\|^2 \leq 4,$$

Generalization Bound: Now, let

$$\epsilon := \frac{2}{\sqrt{n}} \left(1 + \frac{2}{d}\right)^{1/4} e^{-\frac{0.21}{4} d(c_w^2 - 1)^2}.$$

Inequality [36](#page-34-2) shows that the above choices of P<sup>Θ</sup> and PW<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> (according to the scheme JL(d, cw, ν)) satisfy the distortion criterion equation [23.](#page-31-2) Hence, equation [22](#page-31-4) gives

$$\begin{aligned} \text{gen}(\mu, \mathcal{A}) &\leq \mathbb{E}_{P_{\mathbb{S}} P_{\Theta}} \left[ \sqrt{\frac{2\Delta \ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta)}{n} \text{CMI}^{\Theta}(\tilde{\mathbf{S}}, \hat{\mathcal{A}})} \right] + \frac{2}{\sqrt{n}} \left( 1 + \frac{2}{d} \right)^{1/4} e^{-\frac{0.21}{4} d(c_w^2 - 1)^2} \\ &\stackrel{(a)}{\leq} \mathbb{E}_{P_{\mathbb{S}} P_{\Theta}} \left[ \sqrt{\frac{2\Delta \ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta)}{n} d \log \left( \frac{c_w + \nu}{\nu} \right)} \right] + \frac{2}{\sqrt{n}} \left( 1 + \frac{2}{d} \right)^{1/4} e^{-\frac{0.21}{4} d(c_w^2 - 1)^2} \\ &\stackrel{(b)}{\leq} \sqrt{\frac{2d \mathbb{E}_{P_{\mathbb{S}} P_{\Theta}} \left[ \Delta \ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta) \right]}{n}} \log \left( \frac{c_w + \nu}{\nu} \right) + \frac{2}{\sqrt{n}} \left( 1 + \frac{2}{d} \right)^{1/4} e^{-\frac{0.21}{4} d(c_w^2 - 1)^2} \\ &\stackrel{(c)}{\leq} \sqrt{\frac{8d(c_w + \nu)^2}{n} \log \left( \frac{c_w + \nu}{\nu} \right)} + \frac{2}{\sqrt{n}} \left( 1 + \frac{2}{d} \right)^{1/4} e^{-\frac{0.21}{4} d(c_w^2 - 1)^2}, \end{aligned}$$

where (a) is achieved using equation [27,](#page-32-4) (b) by Jensen inequality and due to the concavity of the function √ x, and (c) is derived using equation [37.](#page-34-3)

The proof is completed by letting

$$d=1, \quad c_w=1, \quad \nu=0.4.$$

#### F.3 Proof of Proposition [1](#page-6-1)

As explained in Appendix [F,](#page-31-0) it is sufficient to consider the case L<sup>c</sup> = R = 1. We have

$$\begin{aligned} \text{gen}(\mu, \mathcal{A}) &= \mathbb{E}_{P_{S_n, W}}[\mathcal{R}(W) - \hat{\mathcal{R}}_n(W)] \\ &= \frac{1}{n} \sum_{i \in [n]} \mathbb{E}_{P_{S_n, W}}[\mathbb{E}_{Z \sim \mu}[\ell_{sc}(Z, W)] - \ell_{sc}(Z_i, W)] \\ &\stackrel{(a)}{=} \frac{1}{n} \sum_{i \in [n]} \mathbb{E}_{P_{S_n, W}}[\mathbb{E}_{Z \sim \mu}[-\langle W, Z \rangle] + \langle W, Z_i \rangle] \\ &\stackrel{(b)}{=} \frac{1}{n} \sum_{i \in [n]} \mathbb{E}_{P_{S_n, W}}[\mathbb{E}_{Z \sim \mu}[\ell_c(Z, W)]] - \ell_c(Z_i, W)] \\ &\stackrel{(c)}{\leq} \frac{8}{\sqrt{n}}, \end{aligned}$$

where (a) by definition of ℓsc(z, w) = −⟨w, z⟩+ λ 2 ∥w∥ 2 by Definition [6,](#page-6-0) (b) holds since by Definition [4,](#page-4-2) we have ℓc(z, w) = −L⟨w, z⟩, and (c) follows by Theorem [3.](#page-5-1)

#### F.4 Proof of Theorem [7](#page-23-3)

As explained in Appendix [F,](#page-31-0) we consider the case L = R − B = 1. First, note that similar to the proof of Proposition [1,](#page-6-1) the generalization error does not change, if we consider the loss function ℓglm(z, w) ≜ g (⟨w, ϕ(z)⟩, z) − g (0, z) instead of ℓgl(z, w) = g (⟨w, ϕ(z)⟩, z) + r(w). More precisely,

$$\begin{aligned} \text{gen}(\mu, \mathcal{A}) &= \mathbb{E}_{P_{S_n, W}} [\mathcal{R}(W) - \hat{\mathcal{R}}_n(W)] \\ &= \frac{1}{n} \sum_{i \in [n]} \mathbb{E}_{P_{S_n, W}} [\mathbb{E}_{Z \sim \mu}[\ell_{gl}(Z, W)] - \ell_{gl}(Z_i, W)] \\ &= \frac{1}{n} \sum_{i \in [n]} \mathbb{E}_{P_{S_n, W}} [\mathbb{E}_{Z \sim \mu}[g(\langle W, \phi(Z) \rangle, Z) + r(W)] - g(\langle W, \phi(Z_i) \rangle, Z_i) - r(W)] \\ &= \frac{1}{n} \sum_{i \in [n]} \mathbb{E}_{P_{S_n, W}} [\mathbb{E}_{Z \sim \mu}[g(\langle W, \phi(Z) \rangle, Z)] - g(\langle W, \phi(Z_i) \rangle, Z_i)] \end{aligned}$$

$$\begin{aligned} &\stackrel{(a)}{=} \frac{1}{n} \sum_{i \in [n]} \mathbb{E}_{P_{S_n, W}} [\mathbb{E}_{Z \sim \mu}[g(\langle W, \phi(Z) \rangle, Z) - g(0, Z)] - g(\langle W, \phi(Z_i) \rangle, Z_i) + g(0, Z_i)] \\ &= \frac{1}{n} \sum_{i \in [n]} \mathbb{E}_{P_{S_n, W}} [\mathbb{E}_{Z \sim \mu}[\ell_{glm}(Z, W)] - \ell_{glm}(Z_i, W)], \end{aligned}$$

where (a) follows since <sup>E</sup>Z∼µ[g(0, Z)] = <sup>E</sup>Zi∼µ[g(0, Zi)].

Hence, for the rest of the proof, we consider the generalization with respect to the following loss function:

$$\ell_{glm}(z, w) \triangleq g\left(\langle w, \phi(z) \rangle, z\right) - g\left(0, z\right).$$

Note that due the Lipschitzness of the function g(·, ·) with respect to its first argument, for every z ∈ Z and w ∈ W, we have

$$|\ell_{glm}(z, w)| = |g(\langle w, \phi(z) \rangle, z) - g(0, z)| \leq |\langle w, \phi(z) \rangle|. \quad (38)$$

Furthermore since ∥w∥, ∥ϕ(z)∥ ≤ 1, using Cauchy-Schwarz inequality yields

$$|\ell_{glm}(z, w)| \leq 1.$$

Now, we proceed to establish a generalization bound with respect to the loss function ℓglm(z, w). We use Theorem [1](#page-3-0) with the JL(d, cw, ν) transformation described in Appendix [F.1,](#page-31-1) for some d ∈ N <sup>+</sup>, c<sup>w</sup> ∈ h 1, p 5/4 , and ν ∈ (0, 1]. To do so, We start by bounding CMI<sup>Θ</sup>(S˜, Aˆ), the distortion equation [23,](#page-31-2) and <sup>E</sup>S˜,<sup>Θ</sup>[∆ℓwˆ(S˜, Θ)].

Bound on the disintegrated CMI: It is shown in equation [26](#page-31-3) that

$$\text{CMI}^\Theta(\tilde{\mathbf{S}}, \hat{\mathcal{A}}) = \leq d \log \left( \frac{c_w + \nu}{\nu} \right).$$

Bound on the distortion: Next, we bound the distortion term.

∆(W, ΘWˆ ; Sn) =gen(Sn, W) − gen(Sn, ΘWˆ )

$$\begin{aligned}
& = \text{gen}(S_n, W) - \text{gen}(S_n, \Theta \hat{W}) \\
& = \mathbb{E}_{Z \sim \mu}[\ell_{glm}(Z, W)] - \frac{1}{n} \sum_{i=1}^n \ell_{glm}(Z_i, W) - \mathbb{E}_{Z \sim \mu}[\ell_{glm}(Z, \Theta \hat{W})] + \frac{1}{n} \sum_{i=1}^n \ell_{glm}(Z_i, \Theta \hat{W}) \\
& = \mathbb{E}_{Z \sim \mu}[g(\langle W, \phi(Z) \rangle, Z)] - \frac{1}{n} \sum_{i=1}^n g(\langle W, \phi(Z_i) \rangle, Z_i) \\
& \quad - \mathbb{E}_{Z \sim \mu}[g(\langle \Theta \hat{W}, \phi(Z) \rangle, Z)] + \frac{1}{n} \sum_{i=1}^n g(\langle \Theta \hat{W}, \phi(Z_i) \rangle, Z_i) \\
& \leq \mathbb{E}_{Z \sim \mu}[|g(\langle W, \phi(Z) \rangle, Z) - g(\langle \Theta \hat{W}, \phi(Z) \rangle, Z)|] \\
& \quad + \frac{1}{n} \sum_{i=1}^n |g(\langle W, \phi(Z_i) \rangle, Z_i) - g(\langle \Theta \hat{W}, \phi(Z_i) \rangle, Z_i)| \\
& \stackrel{(a)}{\leq} \mathbb{E}_{Z \sim \mu}[|\langle W - \Theta \hat{W}, \phi(Z) \rangle|] + \frac{1}{n} \sum_{i \in [n]} |\langle W - \Theta \hat{W}, \phi(Z_i) \rangle|, \tag{39}
\end{aligned}$$

where (a) holds due to Lipschitzness of the function g with respect to its first argument.

Hence,

$$\begin{aligned}\mathbb{E}_{\hat{W}, \Theta, W, S_n} \left[ \Delta(W, \Theta\hat{W}; S_n) \right] &\leq 2 \sup_{z, w} \mathbb{E}_{\hat{W}, \Theta \sim P_\Theta} \mathbb{P}_{\hat{W} | \Theta^\top w} \left[ \left\langle w - \Theta\hat{W}, \phi(z) \right\rangle \right] \\ &= 2 \sup_{z, w} \mathbb{E}_{\hat{W}, \Theta \sim P_\Theta} \mathbb{P}_{\hat{W} | \Theta^\top w} \left[ \left\langle w, \phi(z) \right\rangle - \left\langle \hat{W}, \Theta^\top \phi(z) \right\rangle \right] \\ &\leq 2 \sup_{z, w} \left( \mathbb{E}_\Theta \left[ \left\langle w, \phi(z) \right\rangle - \left\langle U, \Theta^\top \phi(z) \right\rangle \right] + \mathbb{E}_{V_\nu, \Theta} \left[ \left\langle V_\nu, \Theta^\top \phi(z) \right\rangle \right] \right],\end{aligned}\tag{40}$$

where the last step follows since by equation [25,](#page-31-5) Wˆ = U + Vν.

In the rest, we fix z and w and upper bound each of the terms in the right-hand side of equation [40:](#page-36-0)

$$\begin{aligned} C_1 &\triangleq \mathbb{E}_{\Theta \sim P_\Theta} \left[ \left\langle \langle w, \phi(z) \rangle - \langle U, \Theta^\top \phi(z) \rangle \right\rangle \right], \\ C_2 &\triangleq \mathbb{E}_{V_\nu, \Theta \sim \text{Uniform}(\mathcal{B}_d(\nu))P_\Theta} \left[ \left\langle \langle V_\nu, \Theta^\top \phi(z) \rangle \right\rangle \right]. \end{aligned}$$

Let E be the event that ∥Θ <sup>⊤</sup>W∥ > c<sup>w</sup> and denote by E c the complementary event of E.

• We start by bounding C1.

$$\begin{aligned} C_1 &= \mathbb{E}_\Theta \left[ \left| \langle w, \phi(z) \rangle - \langle U, \Theta^\top \phi(z) \rangle \right| \mathbb{1}\{\mathcal{E}\} \right] + \mathbb{E}_\Theta \left[ \left| \langle w, \phi(z) \rangle - \langle U, \Theta^\top \phi(z) \rangle \right| \mathbb{1}\{\mathcal{E}^c\} \right] \\ &\stackrel{(a)}{=} \mathbb{E}_\Theta \left[ \left| \langle w, \phi(z) \rangle - \langle \mathbf{0}_d, \Theta^\top \phi(z) \rangle \right| \mathbb{1}\{\mathcal{E}\} \right] + \mathbb{E}_\Theta \left[ \left| \langle w, \phi(z) \rangle - \langle \Theta^\top w, \Theta^\top \phi(z) \rangle \right| \mathbb{1}\{\mathcal{E}^c\} \right] \\ &\leq \mathbb{E}_\Theta [|\langle w, \phi(z) \rangle| \mathbb{1}\{\mathcal{E}\}] + \mathbb{E}_\Theta \left[ \left| \langle w, \phi(z) \rangle - \langle \Theta^\top w, \Theta^\top \phi(z) \rangle \right| \right] \\ &\stackrel{(b)}{\leq} \mathbb{E}_\Theta [\mathbb{1}\{\mathcal{E}\}] + \mathbb{E}_\Theta \left[ \left| \langle w, \phi(z) \rangle - \langle \Theta^\top w, \Theta^\top \phi(z) \rangle \right| \right] \\ &\stackrel{(c)}{\leq} e^{-0.21d(1-c_w^2)^2} + \mathbb{E}_\Theta \left[ \left| \langle w, \phi(z) \rangle - \langle \Theta^\top w, \Theta^\top \phi(z) \rangle \right| \right], \end{aligned} \tag{41}$$

where (a) holds since by equation [24,](#page-31-6) under E, U = 0d, and under E c , U = Θ⊤W, (b) is derived since ∥w∥, ∥ϕ(z)∥ ≤ 1 and hence, Cauchy-Schwarz inequality yields |⟨w, ϕ(z)⟩| ≤ 1, and (c) derived by equation [32.](#page-33-1)

Thus, to bound C1, it remained to bound E<sup>Θ</sup> -⟨w, ϕ(z)⟩ − Θ <sup>⊤</sup>w, Θ <sup>⊤</sup>ϕ(z)  . We use a trick borrowed from [\[71,](#page-13-10) Proof of Theorem 9]. Note that ∥w∥, ∥ϕ(z)∥ ≤ 1. Hence, to upper bound E<sup>Θ</sup> -⟨w, ϕ(z)⟩ − Θ <sup>⊤</sup>w, Θ <sup>⊤</sup>ϕ(z)  , it is sufficient to consider the case where ∥w∥ = ∥ϕ(z)∥ = 1. Let

$$v \triangleq w - \langle w, \phi(z) \rangle \phi(z),$$

$$\hat{v} \triangleq \frac{v}{\|v\|}.$$

It is easy to verify that ⟨v, ϕ(z)⟩ = 0. Hence, since ϕ(z) ⊥ v, we have

$$\|v\| = \sqrt{\|w\|^2 - \langle w, \phi(z) \rangle^2 \|\phi(z)\|^2} = \sqrt{1 - \langle w, \phi(z) \rangle^2} \leq 1.$$

Now, for every r ∈ [d], denote the r'th row of Θ <sup>⊤</sup> ∈ <sup>R</sup> <sup>d</sup>×<sup>D</sup> by T<sup>r</sup> and let

$$\begin{aligned} X_r &\triangleq \langle T_r, \phi(z) \rangle, \\ Y_r &\triangleq \langle T_r, \hat{v} \rangle. \end{aligned}$$

Since ϕ(z) ⊥ v and since the Gaussian distributions are rotationally invariant, we have that X1, . . . , Xd, Y1, . . . , Y<sup>d</sup> are i.i.d. Gaussian random variables distributed according to N (0, 1/d).

Hence, using the identity w = v + ⟨w, ϕ(z)⟩ϕ(z), we can write

$$\begin{aligned} \left| \langle w, \phi(z) \rangle - \langle \Theta^\top w, \Theta^\top \phi(z) \rangle \right| &= \left| \langle w, \phi(z) \rangle - \langle \Theta^\top (v + \langle w, \phi(z) \rangle \phi(z)), \Theta^\top \phi(z) \rangle \right| \\ &= \left| \langle w, \phi(z) \rangle \left( 1 - \left\| \Theta^\top \phi(z) \right\|^2 \right) - \|v\| \langle \Theta^\top \hat{v}, \Theta^\top \phi(z) \rangle \right| \\ &\stackrel{(a)}{\leq} \left| \left\| \Theta^\top \phi(z) \right\|^2 - 1 \right| + \left| \langle \Theta^\top \hat{v}, \Theta^\top \phi(z) \rangle \right| \\ &= \left| \left\| \Theta^\top \phi(z) \right\|^2 - 1 \right| + \left| \sum_{r \in [d]} X_r Y_r \right|, \end{aligned} \quad (42)$$

where (a) is derived using the inequalities ∥⟨w, ϕ(z)⟩ ≤ 1 and ∥v∥ ≤ 1.

We bound the expectation over Θ of each of these terms, denoted respectively as

$$C_{1,1} \triangleq \mathbb{E}_\Theta \left[ \left\| \Theta^\top \phi(z) \right\|^2 - 1 \right],$$

$$C_{1,2} \triangleq \mathbb{E}_\Theta \left[ \sum_{r \in [d]} X_r Y_r \right].$$

– Note that the distribution of

$$d \left\| \Theta^\top \phi(z) \right\|^2,$$

is a chi-squared distribution χ 2 (d) with d-degrees of freedom. Moreover, asymptotically as d → ∞, χ 2 (d) converges to N (d, 2d). Equivalently, asymptotically, χ 2 (d) − d → N (0, 2d). Combining this asymptotic behavior with the fact that for a Gaussian random variable Z ∼ N (0, σ<sup>2</sup> ), with σ ∈ <sup>R</sup>+, we have that <sup>E</sup>[|Z|] = σ q 2 π , yield

$$C_{1,1} \leq \mathcal{O}\left(\frac{1}{\sqrt{d}}\right). \quad (43)$$

– To bound the term C1,2, notice that P <sup>r</sup>∈[d] XrY<sup>r</sup> converges to a random variable with Gaussian distribution N (0, 1/d), as d → ∞. Hence, once again using the fact that for a Gaussian random variable Z ∼ N (0, σ<sup>2</sup> ), <sup>E</sup>[|Z|] = σ q 2 π , yield

$$C_{1,2} \triangleq \mathbb{E}_\Theta \left[ \left| \sum_{r \in [d]} X_r Y_r \right| \right] = \mathcal{O} \left( \frac{\nu}{\sqrt{d}} \right).$$

Combining equation [41,](#page-37-0) equation [42,](#page-37-1) and equation [43](#page-38-1) gives

$$C_1 \triangleq \mathbb{E}_{\Theta \sim P_\Theta} \left[ \left| \langle w, \phi(z) \rangle - \langle U, \Theta^\top \phi(z) \rangle \right| \right] \leq e^{-0.21d(1-c_w^2)^2} + \mathcal{O} \left( \frac{\|\phi(z)\| \|w\|}{\sqrt{d}} \right) \quad (44)$$

$$\leq e^{-0.21d(1-c_w^2)^2} + \mathcal{O}\left(\frac{1}{\sqrt{d}}\right). \quad (45)$$

• Now to bound C2, let V<sup>ν</sup> = (Vν,1, . . . , Vν,d).

$$\begin{aligned}
C_2 &= \mathbb{E}_{\Theta \sim P_{\Theta}} \mathbb{E}_{V_{\nu} \sim \text{Uniform}(\mathcal{B}_d(\nu))} \left[ \left\langle V_{\nu}, \Theta^{\top} \phi(z) \right\rangle \right] \\
&\stackrel{(a)}{=} \mathbb{E}_{\Theta \sim P_{\Theta}} \mathbb{E}_{V_{\nu} \sim \text{Uniform}(\mathcal{B}_d(\nu))} \left[ |V_{\nu}| \|\Theta^{\top} \phi(z)\| \right] \\
&= \mathbb{E}_{\Theta \sim P_{\Theta}} \left[ \|\Theta^{\top} \phi(z)\| \right] \mathbb{E}_{V_{\nu} \sim \text{Uniform}(\mathcal{B}_d(\nu))} [|V_{\nu}| 1] \\
&\stackrel{(b)}{\leq} \mathbb{E}_{V_{\nu} \sim \text{Uniform}(\mathcal{B}_d(\nu))} [|V_{\nu}| 1] \\
&\stackrel{(c)}{=} \frac{\nu \Gamma \left( \frac{d+1}{2} + \frac{1}{2} \right)}{\sqrt{\pi} \Gamma \left( \frac{d+1}{2} + 1 \right)} \\
&\stackrel{(d)}{\leq} \frac{\nu \sqrt{2}}{\sqrt{\pi(d+1)}}, \tag{46}
\end{aligned}$$

where (a) holds by the symmetry of the distribution of Vν, (b) holds since <sup>E</sup>Θ∼P<sup>Θ</sup> -∥Θ <sup>⊤</sup>ϕ(z)∥ ≤ <sup>E</sup>Θ∼P<sup>Θ</sup> ∥Θ <sup>⊤</sup>ϕ(z)∥ <sup>1</sup>/<sup>2</sup> = ∥ϕ(z)∥ ≤ 1, (c) holds by Lemma [4,](#page-38-0) proved in Appendix [F.5,](#page-39-0) and (d) holds since by using Gautschi's inequality we have Γ(x+1/2) Γ(x+1) <sup>≤</sup> √<sup>1</sup> x .

Lemma 4. *Let* V<sup>ν</sup> = (Vν,1, . . . , Vν,d) ∼ *Uniform*(Bd(ν))*. Then,* <sup>E</sup>Vν∼*Uniform*(Bd(ν)) [|Vν,1|] = <sup>ν</sup>Γ( d+2 2 ) √πΓ( d+3 ) *.*

Combining equation [39.](#page-36-1) equation [45,](#page-38-2) and equation [46](#page-38-3) gives

$$\mathbb{E}_{\hat{W}, \Theta, W, S_n} \left[ \Delta(W, \Theta \hat{W}; S_n) \right] \leq e^{-0.21d(1-c_w^2)^2} + \mathcal{O}\left(\frac{1}{\sqrt{d}}\right). \quad (47)$$

Bound on <sup>E</sup>S˜,<sup>Θ</sup>[∆ℓwˆ(S˜, Θ)]: We have

$$\begin{aligned} |\ell_{glm}(z, \Theta\hat{w})| &\stackrel{(a)}{\leq} |\langle \Theta\hat{w}, \phi(z) \rangle| \\ &= \left| \langle \hat{w}, \Theta^\top \phi(z) \rangle \right| \\ &\leq \|\hat{w}\| \|\Theta^\top \phi(z)\| \\ &\stackrel{(b)}{\leq} (c_w + \nu) \|\Theta^\top \phi(z)\|, \end{aligned} \tag{48}$$

where (a) holds by equation [38](#page-36-2) and (b) since by construction ∥wˆ∥ ≤ c<sup>w</sup> + ν.

Hence,

$$\begin{aligned} \mathbb{E}_{\tilde{\mathbf{S}}, \Theta}[\Delta \ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta)] &:= \mathbb{E}_{P_{\tilde{\mathbf{S}}} P_{\Theta} P_{W|\tilde{\mathbf{S}}} P_{\hat{W}|\Theta^\top W}} \left[ \frac{1}{n} \sum_{i \in [n]} (\ell_{glm}(Z_{i,0}, \Theta \hat{W}) - \ell_{glm}(Z_{i,1}, \Theta \hat{W}))^2 \right] \\ &\stackrel{(a)}{\leq} (c_w + \nu)^2 \mathbb{E}_{P_{\tilde{\mathbf{S}}} P_{\Theta} P_{W|\tilde{\mathbf{S}}} P_{\hat{W}|\Theta^\top W}} \left[ \frac{1}{n} \sum_{i \in [n]} (\|\Theta^\top \phi(Z_{i,0})\| + \|\Theta^\top \phi(Z_{i,1})\|)^2 \right] \\ &= (c_w + \nu)^2 \mathbb{E}_{P_{\tilde{\mathbf{S}}} P_{\Theta}} \left[ \frac{1}{n} \sum_{i \in [n]} (\|\Theta^\top \phi(Z_{i,0})\| + \|\Theta^\top \phi(Z_{i,1})\|)^2 \right] \\ &= 4(c_w + \nu)^2 \sup_z \mathbb{E}_{P_{\Theta}} \left[ \|\Theta^\top \phi(z)\|^2 \right] \\ &\stackrel{(b)}{=} 4(c_w + \nu)^2 \sup_z \|\phi(z)\|^2 \\ &\leq 4(c_w + \nu)^2, \end{aligned}$$

where

- (a) follows from equation [48,](#page-38-4)
- (b) since for any fixed <sup>z</sup>, each entry of <sup>Θ</sup>⊤<sup>z</sup> ∥z∥ is an independent random variable distributed according to N (0, 1 d ) and hence

$$\mathbb{E}_\Theta \left[ \|\Theta^\top z\|^2 \right] = \|z\|^2.$$

Generalization Bound: Now, using Theorem [1](#page-3-0) for the above choices of P<sup>Θ</sup> and PW<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> (according to the scheme JL(d, cw, ν)) gives

$$\begin{aligned} \text{gen}(\mu, \mathcal{A}) &\leq \mathbb{E}_{\tilde{\mathbf{S}}, \Theta} \left[ \sqrt{\frac{2\Delta\ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta)}{n} \text{CMI}^\Theta(\tilde{\mathbf{S}}, \hat{\mathcal{A}})} \right] + \mathbb{E}_{\hat{W}, \Theta, W, S_n} \left[ \Delta(W, \Theta\hat{W}; S_n) \right] \\ &\stackrel{(a)}{\leq} \mathbb{E}_{\tilde{\mathbf{S}}, \Theta} \left[ \sqrt{\frac{2\Delta\ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta)}{n} d \log \left( \frac{c_w + \nu}{\nu} \right)} \right] + e^{-0.21d(1-c_w^2)^2} + \mathcal{O} \left( \frac{1}{\sqrt{d}} \right) \\ &\stackrel{(b)}{\leq} \sqrt{\frac{2d \mathbb{E}_{\tilde{\mathbf{S}}, \Theta} \left[ \Delta\ell_{\hat{w}}(\tilde{\mathbf{S}}, \Theta) \right]}{n}} \log \left( \frac{c_w + \nu}{\nu} \right) + e^{-0.21d(1-c_w^2)^2} + \mathcal{O} \left( \frac{1}{\sqrt{d}} \right) \\ &\stackrel{(c)}{\leq} \sqrt{\frac{8d(c_w + \nu)^2}{n} \log \left( \frac{c_w + \nu}{\nu} \right)} + e^{-0.21d(1-c_w^2)^2} + \mathcal{O} \left( \frac{1}{\sqrt{d}} \right), \end{aligned}$$

where (a) is achieved using equation [27](#page-32-4) and equation [47,](#page-38-5) (b) by Jensen inequality and due to the concavity of the function √ x, and (c) is derived using equation [37.](#page-34-3)

The proof is completed by letting

$$d = \sqrt{n}, \quad c_w = 1.1, \quad \nu = 0.5.$$

#### F.5 Proof of Lemma [4](#page-38-0)

Note that

$$\mathbb{E}_{V_\nu \sim \text{Uniform}(\mathcal{B}_d(\nu))} [|V_{\nu,1}|] = \nu \mathbb{E}_{X \sim \text{Uniform}(\mathcal{B}_d(1))} [|X_1|] ,$$

where X = (X1, . . . , Xd) ∼ Uniform(Bd(1)). Hence, it is sufficient to show that <sup>E</sup>X∼Uniform(Bd(1)) [|X1|] = <sup>Γ</sup>( d+2 ) √πΓ( d+3 ) .

First, we compute the marginal distribution of X1. Note that

$$\begin{aligned} f_{X_1}(x_1) &= \frac{1}{\text{Volume}(\mathcal{B}_d(1))} \int_{x_2=-\sqrt{1-x_1^2}}^{\sqrt{1-x_1^2}} \dots \int_{x_d=-\sqrt{1-x_1^2-\dots-x_{d-1}^2}}^{\sqrt{1-x_1^2-\dots-x_{d-1}^2}} dx_2 \dots dx_d \\ &= \frac{\text{Volume}\left(\mathcal{B}_{d-1}\left(\sqrt{1-x_1^2}\right)\right)}{\text{Volume}(\mathcal{B}_d(1))} \\ &= \frac{\Gamma\left(\frac{d+2}{2}\right)}{\sqrt{\pi}\Gamma\left(\frac{d+1}{2}\right)} \left(1 - x_1^2\right)^{\frac{d-1}{2}}. \end{aligned}$$

Now, we have

$$\begin{aligned}\mathbb{E} X \sim \text{Uniform}(\mathcal{B}_d(1)) \quad [|X_1|] &= E_{X_1 \sim f_{X_1}} [|X_1|] \\ &= \frac{2\Gamma\left(\frac{d+2}{2}\right)}{\sqrt{\pi}\Gamma\left(\frac{d+1}{2}\right)} \int_{x_1=0}^1 x_1 (1-x_1^2)^{\frac{d-1}{2}} dx_1 \\ &\stackrel{(a)}{=} \frac{\Gamma\left(\frac{d+2}{2}\right)}{\sqrt{\pi}\Gamma\left(\frac{d+1}{2}\right)} \int_{u=0}^1 (1-u)^{\frac{d-1}{2}} du \\ &\stackrel{(b)}{=} \frac{\Gamma\left(\frac{d+2}{2}\right)}{\sqrt{\pi}\Gamma\left(\frac{d+1}{2}\right)} \text{Beta}(1, (d+1)/2) \\ &= \frac{\Gamma\left(\frac{d+2}{2}\right)}{\sqrt{\pi}\Gamma\left(\frac{d+1}{2}\right)} \times \frac{\Gamma(1)\Gamma\left(\frac{d+1}{2}\right)}{\Gamma\left(\frac{d+3}{2}\right)} \\ &= \frac{\Gamma\left(\frac{d+2}{2}\right)}{\sqrt{\pi}\Gamma\left(\frac{d+3}{2}\right)},\end{aligned}$$

where (a) is achieved by letting u = x 2 <sup>1</sup> and in (b), Beta(·, ·) is the Beta function.

# G Proofs of Section [5](#page-7-0) and Appendix [C:](#page-23-0) Memorization

In this section, we provide the proofs of Section [5](#page-7-0) and Appendix [C.](#page-23-0) Recall that for a given Ki, the adversary outputs its guess of <sup>K</sup><sup>i</sup> as <sup>K</sup>ˆ<sup>i</sup> <sup>≜</sup> Q(W, Zi,K<sup>i</sup> , µ). Throughout the proofs and for better readability, we sometimes denote Kˆ<sup>i</sup> = 1 by Kˆ<sup>i</sup> = 'in' and Kˆ<sup>i</sup> = 0 by Kˆ<sup>i</sup> = 'not in', referring to the semantic meaning that the given Zi,K<sup>i</sup> is part of the training dataset or not.

#### G.1 Proof of Theorem [5](#page-8-0)

We prove each part separately. As stated in the beginning of Appendix [G,](#page-40-1) throughout the proofs and for better readability, we sometimes denote Kˆ<sup>i</sup> = 1 by Kˆ<sup>i</sup> = 'in' and Kˆ<sup>i</sup> = 0 by Kˆ<sup>i</sup> = 'not in', referring to the semantic meaning that the given Zi,K<sup>i</sup> is part of the training dataset or not.

#### G.1.1 Part i.

We prove the result by contradiction. Suppose that there exists an adversary for the algorithm A that is ξ-sound and certifies a recall of m samples with probability q, where ξ < q and m = Ω(n). As before, we denote the output of the learning algorithm by An(Sn) = W.

Recall that S˜ <sup>J</sup> = {Z1,J<sup>1</sup> , Z2,J<sup>2</sup> , . . . , Zn,J<sup>n</sup> } is the training dataset <sup>S</sup><sup>n</sup> and <sup>S</sup>˜ \ <sup>S</sup>˜ <sup>J</sup> is the test dataset S ′ n.

Define Jˆ<sup>i</sup> ∈ {0, 1} as follows:

$$\hat{J}_i = \begin{cases} 0, & \text{if } \mathcal{Q}(\hat{W}, Z_{i,0}, \mu) = \text{'in' and } \mathcal{Q}(\hat{W}, Z_{i,1}, \mu) = \text{'not in'}, \\ 1, & \text{if } \mathcal{Q}(\hat{W}, Z_{i,0}, \mu) = \text{'not in' and } \mathcal{Q}(\hat{W}, Z_{i,1}, \mu) = \text{'in'}, \\ U_i, & \text{otherwise,} \end{cases}$$

where U<sup>i</sup> ∼ Bern(1/2) is a binary uniform random variable, independent of other random variables.

$$\mathbb{P}\left(\exists i \in [n]: \mathcal{Q}(W, Z_i, J_i^c, \mu) = '\text{in}'\right) \leq \xi,$$

and an adversary certifying a recall of m samples means that,

$$\mathbb{P} \left( \sum_{i \in [n]} \mathbb{1}\{\mathcal{Q}(W, Z_i, J_i, \mu) = '\text{in}'\} \geq m \right) \geq q.$$

Since we assumed m = Ω(n), there exists c<sup>1</sup> ∈ (0, 1] and n<sup>0</sup> ∈ N such that, for all n ≥ n0, m ≥ c1n. The second condition then yields,

$$\mathbb{P} \left( \sum_{i \in [n]} \mathbb{1}\{\mathcal{Q}(W, Z_i, J_i, \mu) = '\text{in}'\} \geq c_1 n \right) \geq q.$$

Define the Hamming distance d<sup>H</sup> : {0, 1} <sup>n</sup> × {0, 1} <sup>n</sup> → [n] between binary vectors J and Jˆ as

$$d_H\left(\mathbf{J},\hat{\mathbf{J}}\right) = \sum_{i \in [n]} \mathbb{1}\{J_i \neq \hat{J}_i\}.$$

Next, we use Fano's inequality with approximate recovery [\[59,](#page-13-1) Theorem 2]. Let t = n n 2 1 − c<sup>1</sup> 2 and denote

$$\begin{aligned} P_{e_t} &\triangleq \mathbb{P}\left(d_H\left(\mathbf{J}, \hat{\mathbf{J}}\right) > nt\right), \\ N_{\hat{\mathbf{j}}} &\triangleq \sum_{\mathbf{j} \in \{0,1\}^n} \mathbb{1}\left\{d_H(\hat{\mathbf{j}}, \hat{\mathbf{j}}) \leq nt\right\}. \end{aligned}$$

Note that N<sup>ˆ</sup><sup>j</sup> is the same for all ˆj ∈ {0, 1} n . Indeed, dH(j, ˆj) = dH(j ⊕ a, ˆj ⊕ a), where ⊕ denotes the modulo two summation, for any a ∈ {0, 1} n , and P <sup>j</sup>∈{0,1}<sup>n</sup> <sup>1</sup> n dH(j ⊕ a, <sup>ˆ</sup><sup>j</sup> <sup>⊕</sup> <sup>a</sup>) <sup>≤</sup> nto = P <sup>j</sup>∈{0,1}<sup>n</sup> <sup>1</sup> n dH(j, <sup>ˆ</sup><sup>j</sup> <sup>⊕</sup> <sup>a</sup>) <sup>≤</sup> nto . Hence, N<sup>ˆ</sup><sup>j</sup> = N<sup>ˆ</sup>j⊕<sup>a</sup> for any <sup>a</sup>, and the maximum over <sup>ˆ</sup><sup>j</sup> of <sup>N</sup><sup>ˆ</sup><sup>j</sup> is equal to N<sup>1</sup><sup>n</sup> .

With these notations, we have

$$\begin{aligned} o(n) &\stackrel{(a)}{=} \mathbf{I}(\mathbf{J}; W|\tilde{\mathbf{S}}) \\ &\stackrel{(b)}{=} \mathbf{I}(\mathbf{J}; W|\tilde{\mathbf{S}}, \mathbf{K}) \\ &\stackrel{(c)}{=} \mathbf{I}(\mathbf{J}; W, \hat{\mathbf{J}}|\tilde{\mathbf{S}}, \mathbf{K}) \\ &\stackrel{(d)}{\geq} \mathbf{I}(\mathbf{J}; \hat{\mathbf{J}}|\tilde{\mathbf{S}}, \mathbf{K}) \\ &\stackrel{(e)}{\geq} \mathbf{I}(\mathbf{J}; \hat{\mathbf{J}}) \\ &\stackrel{(f)}{\geq} (1 - P_{e_t}) \log \left( \frac{2^n}{N_{1_n}} \right) - \log(2) \\ &\stackrel{(g)}{\geq} n (1 - P_{e_t}) (1 - h_b(t)) - (1 - P_{e_t}) \log(c_3) - \log(2), \end{aligned}$$

where (a) follows by the assumption of the theorem, (b) results from K is independent of (W, S˜, J), (c) results from I(J; Jˆ|W, S˜, K) = 0 since Jˆ is a function of (W, S˜, K), (d) results from I(J; W, Jˆ|S˜, K) = I(J; Jˆ|S˜, K) + I(J; W|S˜, K, Jˆ) ≥ I(J; Jˆ|S˜, K) (by the positivity of mutual information), (e) is due to the identities below,

$$\mathsf{I}(\mathbf{J}; \hat{\mathbf{J}}|\tilde{\mathbf{S}}, \mathbf{K}) = H(\mathbf{J}) - H(\mathbf{J}|\hat{\mathbf{J}}, \tilde{\mathbf{S}}, \mathbf{K}) \geq H(\mathbf{J}) - H(\mathbf{J}|\hat{\mathbf{J}}) = \mathsf{I}(\mathbf{J}; \hat{\mathbf{J}}),$$

(f) results from applying Fano's inequality with approximate recovery [\[59,](#page-13-1) Theorem 2], and (g) is derived using the claim, proved later below, that N<sup>1</sup><sup>n</sup> ≤ c32 nhb(t) for some constant c<sup>3</sup> ∈ <sup>R</sup><sup>+</sup> and for n sufficiently large.

Note that t = n n 1 − c<sup>1</sup> 2 <sup>&</sup>lt; <sup>1</sup>/<sup>2</sup> and as <sup>n</sup> → ∞, <sup>t</sup> <sup>→</sup> <sup>1</sup>−c1/<sup>2</sup> <sup>2</sup> < 1/2. Hence, since hb(x) is a continuous function of x ∈ [0, 1], 1 − hb(t) converges to the constant 1 − h<sup>b</sup> 1−c1/2 2 > 0. Hence, if we show that for sufficiently large n, 1 − P<sup>e</sup><sup>t</sup> > 0, we obtain a contradiction. Since the left-hand side is of order o(n), which is greater than the right-hand side, which is Ω(n), and the proof is complete.

Hence, it remains to show for n sufficiently large, Claim i) N<sup>1</sup><sup>n</sup> ≤ c32 nhb(t) for some constant c<sup>3</sup> ∈ <sup>R</sup>+, and Claim ii) P<sup>e</sup><sup>t</sup> < 1.

#### Proof of Claim i)

We have

$$\begin{aligned}
N_{\mathbf{1}_n} &= \sum_{\mathbf{j} \in \{0,1\}^n} \mathbb{1}\{d_H(\mathbf{j}, \mathbf{1}_n) \leq nt\} \\
&= \sum_{i=0}^{nt} \binom{n}{i} \\
&\stackrel{(a)}{\leq} \sum_{i=0}^{nt} \binom{n'}{i} \\
&\stackrel{(b)}{\leq} 2^{n'-1} \frac{\binom{n'}{nt+1}}{\binom{n'}{\frac{n'}{2}}} \\
&\stackrel{(c)}{\leq} 2^{n'h_b((nt+1)/n')} \sqrt{\frac{1}{4\pi(nt/n' + 1/n')(1 - nt/n' - 1/n')}} \\
&\stackrel{(d)}{\leq} c_3 2^{nh_b(t)},
\end{aligned} \tag{49}$$

where n ′ = 2 <sup>n</sup> 2 and c<sup>3</sup> ∈ <sup>R</sup>+, (a) results from n ′ ≥ n, (b) follows from applying [\[72,](#page-13-11) Proposition 5.18][<sup>8</sup>](#page-0-0) (n ′ is even and nt ≤ n ′ /2 − 1), (c) is derived using the relation

$$e^{mh_b(j/m)} \sqrt{\frac{m}{8j(m-j)}} \leq \binom{m}{j} \leq e^{mh_b(j/m)} \sqrt{\frac{m}{2\pi j(m-j)}},$$

which is valid for any m ∈ N and 1 ≤ j ≤ m − 1 (see [\[73,](#page-13-12) Exercise 5.8.a]), and (d) holds for sufficiently large n, using n ≤ n ′ ≤ n + 1.

Proof of Claim ii) Define the following events: E<sup>1</sup> ≜ ∃i ∈ [n]: Q(W, Zi,J<sup>c</sup> , µ) = 'in' , E<sup>2</sup> ≜ nP i∈[n] <sup>1</sup>{Q(W, Zi,J<sup>i</sup> , µ) = 'in'} < c1n o . Then, we have

$$\begin{aligned}
& P_{e_t} \triangleq \mathbb{P} \left( d_H \left( \mathbf{J}, \hat{\mathbf{J}} \right) > nt \right) \\
& = \mathbb{P} \left( d_H \left( \mathbf{J}, \hat{\mathbf{J}} \right) > nt, \mathcal{E}_1^c, \mathcal{E}_2^c \right) + \mathbb{P} \left( \mathcal{E}_1, \mathcal{E}_2 \right) \\
& \stackrel{(a)}{=} \mathbb{P} \left( \sum_{i \in [n]} \mathbb{1} \left\{ U_i \neq J_i \right\} \mathbb{1} \left\{ \mathcal{Q}(W, Z_{i, J_i}, \mu) = '\text{not in}' \right\} > nt, \mathcal{E}_1^c, \mathcal{E}_2^c \right) + \mathbb{P} \left( \mathcal{E}_1, \mathcal{E}_2 \right) \\
& \stackrel{(b)}{\leq} \sum_{r \in [\lceil n(1-c_1) \rceil]} \mathbb{P} \left( \sum_{i \in [n]} \mathbb{1} \left\{ U_i \neq J_i \right\} \mathbb{1} \left\{ \mathcal{Q}(W, Z_{i, J_i}, \mu) = '\text{not in}' \right\} > nt, \right. \\
& \quad \left. \sum_{i \in [n]} \mathbb{1} \left\{ \mathcal{Q}(W, Z_{i, J_i}, \mu) = '\text{not in}' \right\} = r, \mathcal{E}_1^c, \mathcal{E}_2^c \right) + \mathbb{P} \left( \mathcal{E}_1, \mathcal{E}_2 \right) \\
& \leq \sum_{r \in [\lceil n(1-c_1) \rceil]} \mathbb{P} \left( \sum_{i \in [n]} \mathbb{1} \left\{ U_i \neq J_i \right\} \mathbb{1} \left\{ \mathcal{Q}(W, Z_{i, J_i}, \mu) = '\text{not in}' \right\} > nt, \right. \\
& \quad \left. \sum_{i \in [n]} \mathbb{1} \left\{ \mathcal{Q}(W, Z_{i, J_i}, \mu) = '\text{not in}' \right\} = r \right) + \mathbb{P} \left( \mathcal{E}_1, \mathcal{E}_2 \right) \\
& = \sum_{r \in [\lceil n(1-c_1) \rceil]} \mathbb{P} \left( \sum_{i \in [n]} \mathbb{1} \left\{ U_i \neq J_i \right\} \mathbb{1} \left\{ \mathcal{Q}(W, Z_{i, J_i}, \mu) = '\text{not in}' \right\} > nt \mid \sum_{i \in [n]} \mathbb{1} \left\{ \mathcal{Q}(W, Z_{i, J_i}, \mu) = '\text{not in}' \right\} = r \right)
\end{aligned}$$

<sup>8</sup> See also <https://mathoverflow.net/questions/17202/sum-of-the-first-k-binomial-coefficients-for-fixed-n> for a reformulation.

$$\begin{aligned} & \times \mathbb{P} \left( \sum_{i \in [n]} \mathbb{1} \{ \mathcal{Q}(W, Z_i, J_i, \mu) = '\text{not in}' \} = r \right) \\ & + \mathbb{P} (\mathcal{E}_1, \mathcal{E}_2) \\ \stackrel{(c)}{=} & \sum_{r \in [nt, \lceil n(1-c_1) \rceil]} \mathbb{P} \left( \sum_{i \in [n]} \mathbb{1} \{ U_i \neq J_i \} \mathbb{1} \{ \mathcal{Q}(W, Z_i, J_i, \mu) = '\text{not in}' \} > nt \mid \sum_{i \in [n]} \mathbb{1} \{ \mathcal{Q}(W, Z_i, J_i, \mu) = '\text{not in}' \} = r \right) \\ & \times \mathbb{P} \left( \sum_{i \in [n]} \mathbb{1} \{ \mathcal{Q}(W, Z_i, J_i, \mu) = '\text{not in}' \} = r \right) \\ & + \mathbb{P} (\mathcal{E}_1, \mathcal{E}_2) \\ \stackrel{(d)}{=} & \sum_{r \in [nt, \lceil n(1-c_1) \rceil]} e^{-2r \left( \frac{nt}{r} - \frac{1}{2} \right)^2} \mathbb{P} \left( \sum_{i \in [n]} \mathbb{1} \{ \mathcal{Q}(W, Z_i, J_i, \mu) = '\text{not in}' \} = r \right) + \mathbb{P} (\mathcal{E}_1, \mathcal{E}_2) \\ & \leq \max_{r \in [nt, \lceil n(1-c_1) \rceil]} e^{-2r \left( \frac{nt}{r} - \frac{1}{2} \right)^2} + \mathbb{P} (\mathcal{E}_1, \mathcal{E}_2) \\ & \stackrel{(e)}{=} e^{-2\lceil n(1-c_1) \rceil \left( \frac{nt}{\lceil n(1-c_1) \rceil} - \frac{1}{2} \right)^2} + \mathbb{P} (\mathcal{E}_1, \mathcal{E}_2) \\ & \stackrel{(f)}{\leq} e^{-2\lceil n(1-c_1) \rceil \left( \frac{nt}{\lceil n(1-c_1) \rceil} - \frac{1}{2} \right)^2} + \mathbb{P} (\mathcal{E}_1) + \mathbb{P} (\mathcal{E}_2) \\ & \leq e^{-2\lceil n(1-c_1) \rceil \left( \frac{nt}{\lceil n(1-c_1) \rceil} - \frac{1}{2} \right)^2} + \xi + 1 - q, \end{aligned}$$

and we justify the main steps hereafter:

- (a) holds since under the event E c <sup>1</sup>, we have that ∀i ∈ [n], <sup>1</sup>{Q(W, Zi,J<sup>c</sup> i , µ) = 'in'} = 0 and also whenever i) both Q(W, Zi,J<sup>c</sup> , µ) = 'not in' and Q(W, Zi,J<sup>i</sup> , µ) = 'not in', Jˆ<sup>i</sup> is chosen as U<sup>i</sup> and hence the Hamming difference of the i'th coordinate is <sup>1</sup> {U<sup>i</sup> ̸= Ji}, and ii) when Q(W, Zi,J<sup>c</sup> , µ) = 'not in' and Q(W, Zi,J<sup>i</sup> , µ) = 'in', Jˆ<sup>i</sup> is chosen as J<sup>i</sup> and hence the Hamming difference of the i'th coordinate is 0.
- (b) holds since under the event E c <sup>2</sup>, we have that P i∈[n] <sup>1</sup>{Q(W, Zi,J<sup>i</sup> , µ) = 'not in'} ≤ n(1 − c1),
- (c) holds since for r < nt, the probability is zero,
- (d) holds by Hoeffding's inequality for the independent uniform random variables <sup>1</sup>{U<sup>i</sup> ̸= Ji} and since nt > n(1 − c1/2)/2 ≥ r/2 for n sufficiently large,
- (e) holds for n large enough since,

$$\begin{aligned} \log \left( \max_{r \in [nt, \lceil n(1-c_1) \rceil]} e^{-2r \left( \frac{nt}{r} - \frac{1}{2} \right)^2} \right) &= - \min_{r \in [nt, \lceil n(1-c_1) \rceil]} 2r \left( \frac{nt}{r} - \frac{1}{2} \right)^2 \\ &= - \min_{\frac{x}{nt} \in [1, \frac{\lceil n(1-c_1) \rceil}{nt}]} 2nt \frac{r}{nt} \left( \frac{nt}{r} - \frac{1}{2} \right)^2 \\ &= -2nt \min_{x \in [1, \frac{\lceil n(1-c_1) \rceil}{nt}]} x \left( \frac{1}{x} - \frac{1}{2} \right)^2 \\ &= -2nt \min_{x \in [1, \frac{\lceil n(1-c_1) \rceil}{nt}]} \left( \frac{1}{x} - 1 + \frac{x}{4} \right) \\ &\stackrel{(*)}{=} -2nt \left( \frac{nt}{\lceil n(1-c_1) \rceil} - 1 + \frac{\lceil n(1-c_1) \rceil}{4nt} \right) \\ &= -2\lceil n(1-c_1) \rceil \left( \frac{nt}{\lceil n(1-c_1) \rceil} - \frac{1}{2} \right)^2, \end{aligned}$$

where (∗) is derived since i) for n sufficiently large, ⌈n(1−c1)⌉ nt = ⌈n(1−c1)⌉ (1− c1 which is less than 2 for n large, and ii) since <sup>x</sup> <sup>−</sup> 1 + <sup>x</sup> 4 is decreasing in the range (0, 2],

- (f) results from <sup>P</sup> (E1, E2) ≤ <sup>P</sup> (E1) + <sup>P</sup> (E2).

Since for sufficiently large n, e −⌈n(1−c1)⌉ nt ⌈n(1−c1)⌉ <sup>−</sup> <sup>1</sup> 2 (which converges to e − nc<sup>2</sup> 1 8(1−c1) ) gets sufficiently small, hence, if ξ < q, then P<sup>e</sup><sup>t</sup> < 1. This completes the proof of Claim ii), and hence of Part i).

#### G.1.2 Part ii.

Similarly to Part i) (Appendix [G.1.1\)](#page-40-2), we will prove the result by contradiction: assume that there exists an adversary for A such that

$$\mathbb{P}\left(\exists i \in [n]: \mathcal{Q}(W, Z_i, J_i^c, \mu) = '\text{in}'\right) \leq \xi,$$

and

$$\mathbb{P} \left( \sum_{i \in [n]} \mathbb{1}\{\mathcal{Q}(W, Z_i, J_i, \mu) = '\text{in}'\} \geq \alpha n \right) \geq q.$$

This also gives,

$$\mathbb{P} \left( \sum_{i \in [n]} \mathbb{1}\{\mathcal{Q}(W, Z_i, J_i, \mu) = '\text{not in}'\} \geq n(1 - \alpha) \right) \leq 1 - q. \quad (50)$$

In our proof, we allow the adversary to be stochastic. We denote expectations and probabilities with respect to the adversary's randomness (which is independent of all other random variables) by <sup>E</sup>Q[·] and <sup>P</sup>Q[·], where needed. The main part of the proof relies on the following lemma, which we state below but prove later (in Appendix [G.7\)](#page-52-0) for better readability.

Lemma 5. *The following holds.*

$$\mathbb{E}_{W, \tilde{\mathbf{S}}, \mathbf{J}, \mathcal{Q}} \left[ \sum_{i \in [n]} (\mathbb{1} \{ \mathcal{Q}(W, Z_i, J_i^c, \mu) = 'not in' \} - \mathbb{1} \{ \mathcal{Q}(W, Z_i, J_i, \mu) = 'not in' \}) \right] = o(n).$$

By Lemma [5,](#page-44-0) we have

$$\begin{aligned} \mathbb{E}_{W, \hat{\mathbf{S}}, \mathbf{J}, \mathcal{Q}} \left[ \sum_{i \in [n]} \mathbb{1} \{ \mathcal{Q}(W, Z_i, J_i^c, \mu) = ' \text{ not in} ' \} \right] \\ = o(n) + \mathbb{E}_{W, \hat{\mathbf{S}}, \mathbf{J}, \mathcal{Q}} \left[ \sum_{i \in [n]} \mathbb{1} \{ \mathcal{Q}(W, Z_i, J_i, \mu) = ' \text{ not in} ' \} \right] \\ \stackrel{(a)}{\leq} o(n) + n(1-q) + n(1-\alpha)q \\ = o(n) + n(1-\alpha q), \end{aligned}$$

where (a) holds using [\(50\)](#page-44-1) and P i∈[n] <sup>1</sup>{Q(W, Zi,J<sup>i</sup> , µ) = 'not in'} ≤ n.

Hence, using Markov's inequality,

$$\mathbb{P}\left(\sum_{i \in [n]} \mathbb{1} \left\{ \mathcal{Q}(W, Z_i, J_i^c, \mu) = '\text{not in}' \right\} \geq n - m' \right) \leq \frac{o(n) + n(1 - \alpha q)}{n - m'},$$

or equivalently,

$$\mathbb{P}\left(\sum_{i \in [n]} \mathbb{1}\{\mathcal{Q}(W, Z_i, J_i^c, \mu) = '\text{in}'\} \geq m'\right) \geq 1 - \frac{o(n) + n(1 - \alpha q)}{n - m'}.$$

Hence, for any

$$q' \in (0, \alpha q), \quad m' = n - \frac{o(n) + n(1 - \alpha q)}{1 - q'} = \frac{n(\alpha q - q' - o(1))}{1 - q'},$$

we have

$$\mathbb{P}\left(\sum_{i \in [n]} \mathbb{1} \left\{ \mathcal{Q}(W, Z_{i, J_i^c}, \mu) = '\text{in}' \right\} \geq m' \right) \geq q'.$$

Hence, by varying q ′ over the interval (0, αq), the ratio m′ /n changes asymptotically from 0 to αq. In other words, if n is sufficiently large, then for any

$$\epsilon \in (0, \alpha), \quad m' = \left( \frac{\epsilon}{1/q + \epsilon - \alpha} \right) n - o(n) = \Omega(n),$$

we have

$$\mathbb{P}\left(\sum_{i \in [n]} \mathbb{1}\{\mathcal{Q}(W, Z_i, J_i^c, \mu) = 'in'\} \geq m'\right) \geq (\alpha - \epsilon)q.$$

This completes the proof of Part ii).

#### G.2 Proof of Theorem [6](#page-8-1)

To prove Theorem [6,](#page-8-1) we show that for any learning algorithm A: Z → R <sup>D</sup>, the projected-quantized algorithm, defined as

$$\mathcal{A}^*(S_n) \triangleq \Theta \tilde{\mathcal{A}}(\Theta^\top \mathcal{A}(S_n)) = \Theta \hat{W},$$

satisfies equation [8](#page-8-2) and

$$\text{CMI}(\mu, \mathcal{A}^*(S_n)) \leq \mathbb{E}_\Theta \left[ \text{CMI}^\Theta(\mu, \mathcal{A}^*(S_n)) \right] = o(n), \quad (51)$$

for any distribution µ. Having shown this, applying Theorem [5](#page-8-0) completes the proof.

Fix any arbitrary distribution µ. Consider the construction of JL(d, cw, ν), described in Appendix [F.1.](#page-31-1) It is shown in equation [26](#page-31-3) that

$$\text{CMI}^\Theta(\tilde{\mathbf{S}}, \tilde{\mathcal{A}}) \leq d \log \left( \frac{c_w + \nu}{\nu} \right),$$

which, together with the data-processing inequality, yield

$$\text{CMI}^\Theta(\mu, \mathcal{A}^*(S_n)) \leq d \log \left( \frac{c_w + \nu}{\nu} \right). \quad (52)$$

Furthermore, similar to equation [36,](#page-34-2) where it is shown that

$$\mathbb{E}_{P_{S_n, W} P_{\Theta} P_{\hat{W}|\Theta^{\top} W}} \left[ \text{gen}(S_n, W) - \text{gen}(S_n, \Theta \hat{W}) \right] \leq \frac{2}{\sqrt{n}} \left( 1 + \frac{2}{d} \right)^{1/4} e^{-\frac{0.21d}{4} d(c_w^2 - 1)^2},$$

it can be shown that

$$\left| \mathbb{E}_{P_{S_n, W} P_\Theta P_{\hat{W} | \Theta^\top W}} \left[ \text{gen}(S_n, W) - \text{gen}(S_n, \Theta \hat{W}) \right] \right| \leq \frac{2}{\sqrt{n}} \left( 1 + \frac{2}{d} \right)^{1/4} e^{-\frac{0.21}{4} d (c_w^2 - 1)^2}. \quad (53)$$

Plugging the choices

$$d = 500r \log(n), \quad c_w = 1.1, \quad \nu = 0.4,$$

in equation [52](#page-45-2) and equation [53](#page-45-3) result equation [51](#page-44-2) and equation [8,](#page-8-2) which completes the proof.

#### G.3 Proof of Lemma [1](#page-23-1)

If m = 0, then consider an adversary that always outputs Q(W, Z, µ) = 0, for any Z ∈ Z.

In the following, we assume that m = nm′ ̸= 0. Let V ∈ {0, 1} be a binary random variable, independent of all other random variables, such that P(V = 0) = α. For example, if there exists a set B ⊆ W such that P(W ∈ B) = α, then the adversary can set V = <sup>1</sup>{W /∈ B}.

Consider an adversary that first picks a random V . If V = 0, then for any Z ∈ Z, it declares Q(W, Z, µ) = 0. Otherwise (*i.e.,* V = 1), it declares Q(W, Z, µ) = 0 with probability r<sup>n</sup> and Q(W, Z, µ) = 1 with probability 1 − rn, independently of (W, Z, µ).

If V = 0, the adversary never recalls m samples with any positive probability

$$\begin{aligned} \mathbb{P}\left(\sum_{i \in [n]} \mathcal{Q}(W, Z_{i,1}, \mu) \geq m\right) &= \mathbb{P}\left(\sum_{i \in [n]} \mathcal{Q}(W, Z_{i,1}, \mu) \geq m, V = 1\right) \\ &= (1 - \alpha)\mathbb{P}\left(\sum_{i \in [n]} \mathcal{Q}(W, Z_{i,1}, \mu) \geq m \mid V = 1\right). \end{aligned}$$

Moreover,

$$\begin{aligned}\mathbb{P}(\exists i \in [n] : \mathcal{Q}(W, Z_{i,0}, \mu) = 1) &= \mathbb{P}(\exists i \in [n] : \mathcal{Q}(W, Z_{i,0}, \mu) = 1, V = 1) \\ &= (1 - \alpha)\mathbb{P}(\exists i \in [n] : \mathcal{Q}(W, Z_{i,0}, \mu) = 1 \mid V = 1).\end{aligned}$$

Using the above two relations, this adversary is ξ-sound and recalls m samples with probability q if, restricting to V = 1, the adversary is <sup>ξ</sup> (1−α) -sound and recalls m samples with probability <sup>q</sup> (1−α) . For the adversary to be <sup>ξ</sup> (1−α) -sound given V = 1, we should have <sup>P</sup> (∀i ∈ [n], Q(W, Zi,0, µ) = 0) ≥ 1− ξ (1−α) . Hence, this adversary is ξ-sound if and only if

$$r_n^n \geq 1 - \frac{\xi}{(1-\alpha)},$$

therefore,

$$r_n \geq \sqrt{1 - \frac{\xi}{(1 - \alpha)}}.$$

Next, when V = 1, to find the probability of recalling m = nm′ samples with probability <sup>q</sup> (1−α) , note that the probability of Q(W, Zi,1, µ) = 1 is equal to (1 − rn). We consider two cases:

- i. If r<sup>n</sup> = 0, P P <sup>i</sup>∈[n] <sup>Q</sup>(W, Zi,1, µ) <sup>≥</sup> <sup>m</sup>|<sup>V</sup> = 1 = 1.
- ii. If m′ < 1 − rn, using Hoeffding's inequality, we have

$$\mathbb{P} \left( \sum_{i \in [n]} \mathcal{Q}(W, Z_{i,1}, \mu) \geq m | V = 1 \right) \geq 1 - e^{-2n(m' + r_n - 1)^2}.$$

Considering these two cases separately,

- i. We should find a value of α such that <sup>q</sup> (1−α) ≤ 1 and 0 ≥ <sup>n</sup> q 1 − (1−α) . Both conditions are satisfied for α = 1 − ξ, if ξ ≥ q.
- ii. It is sufficient to find a value for r<sup>n</sup> such that m′ < (1 − rn), r<sup>n</sup> ≥ <sup>n</sup> q 1 − (1−α) and 1 − e −2n(m′+rn−1)<sup>2</sup> ≥ q . If, 1 − m′ − s 1 log 1− <sup>q</sup> ≥ 0, then let

(1−α)

2n

(1−α)

$$r_n \triangleq 1 - m' - \sqrt{\frac{1}{2n} \log \left( \frac{1}{1 - \frac{q}{(1-\alpha)}} \right)}.$$

It satisfies the first condition and the recall condition. Lastly, the soundness condition is satisfied if for sufficiently large n, we have

$$\sqrt[n]{1 - \frac{\xi}{(1-\alpha)}} + \sqrt{\frac{1}{2n} \log \left( \frac{1}{1 - \frac{q}{(1-\alpha)}} \right)} + \frac{m}{n} \leq 1.$$

#### G.4 Proof of Theorem [8](#page-24-0)

We prove the theorem and the comment after it, separately.

In the first case and to prove Theorem [8,](#page-24-0) we show that for every r < 1 there exists a projection matrix Θ ∈ R <sup>D</sup>×<sup>d</sup> with d = ⌈n 2r−1 ⌉, a Markov Kernel PW<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> and a *compression algorithm* A ∗ <sup>Θ</sup>,n : Z <sup>n</sup> → <sup>R</sup> d , defined as A ∗ <sup>Θ</sup>,n(Sn) <sup>≜</sup> A˜(Θ<sup>⊤</sup>A(Sn)) = <sup>W</sup><sup>ˆ</sup> , such that

$$\left| \mathbb{E}_{P_{S_n, W} P_{\hat{W} | \Theta^\top W}} \left[ \text{gen}(S_n, W) - \text{gen}(S_n, \Theta \hat{W}) \right] \right| = \mathcal{O}\left(n^{-r}\right),$$

$$\text{CMI}(\mu, \mathcal{A}_{\Theta, n}^*) = o(n).$$

Having shown this, then applying Theorem [5](#page-8-0) completes the proof.

In the second case, we show that for every r ∈ R, there exist a projection matrix Θ ∈ R <sup>D</sup>×<sup>d</sup> with d = ⌈r log(n)⌉, a Markov Kernel PW<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> and a *compression algorithm* A ∗ <sup>Θ</sup>,n : Z <sup>n</sup> → <sup>R</sup> d , defined as A ∗ <sup>Θ</sup>,n(Sn) <sup>≜</sup> A˜(Θ<sup>⊤</sup>A(Sn)) = <sup>W</sup><sup>ˆ</sup> , such that

$$\mathbb{E} P_{S_n, W} P_{\hat{W} | \Theta^\top W} \left[ \text{gen}(S_n, W) - \text{gen}(S_n, \Theta \hat{W}) \right] = \mathcal{O} \left( n^{-r} \right),$$

and

$$\text{CMI}(\mu, \mathcal{A}_{\Theta, n}^*) = o(n).$$

Having shown this, again applying Theorem [5](#page-8-0) completes the proof.

Hence, it remains to show the existence of such projection matrices Θ ∈ R D×d , Markov Kernels PW<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> and compression algorithms A ∗ <sup>Θ</sup>,n : Z <sup>n</sup> → <sup>R</sup> d , for each of the above cases.

#### G.4.1 Case i.

Consider the construction of JL(d, cw, ν), described in Appendix [F.1.](#page-31-1) It is shown in equation [26](#page-31-3) that

$$\text{CMI}^\Theta(\tilde{\mathbf{S}}, \hat{\mathcal{A}}) = \leq d \log \left( \frac{c_w + \nu}{\nu} \right).$$

Hence, for any fixed Θ,

$$\text{CMI}^\Theta(\mu, \hat{\mathcal{A}}) = \leq d \log \left( \frac{c_w + \nu}{\nu} \right).$$

Now, let

$$\Delta(W, \Theta\hat{W}; S_n) := \text{gen}(S_n, W) - \text{gen}(S_n, \Theta\hat{W}).$$

We show that for any r < 1, letting

$$d = n^{2r-1}, \quad , c_w = 1.1, \quad , \nu = 0.4,$$

results in

$$\mathfrak{E}_1 \triangleq \mathbb{E}_\Theta \left[ \left\| \mathbb{E}_{\hat{W}, W, S_n} \left[ \Delta(W, \Theta \hat{W}; S_n) \right] \right\| \right] = \mathcal{O} \left( \frac{1}{n^r} \right). \quad (54)$$

Having shown this, it's easy to see that there exists a Θ, for which simultaneously

$$\left| \mathbb{E}_{\hat{W}, W, S_n} \left[ \Delta(W, \Theta \hat{W}; S_n) \right] \right| = \mathcal{O} \left( \frac{1}{n^r} \right),$$

and

$$\text{CMI}^{\ominus}(\mu, \hat{A}) = o(n).$$

Fix this matrix Θ ∈ R D×d and the Markov Kernel PW<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> induced by that. Choosing the overall algorithms as A ∗ <sup>Θ</sup>,n : Z <sup>n</sup> → <sup>R</sup> d completes the proof.

Hence, it remains to show that equation [54](#page-46-1) holds. By equation [28,](#page-32-5) we have

$$\Delta(W, \Theta \hat{W}; S_n) = -\langle W, \bar{Z} \rangle + \langle \hat{W}, \Theta^\top \bar{Z} \rangle,$$

where Z¯ <sup>≜</sup> <sup>E</sup>Z∼µ[Z] − 1 P<sup>n</sup> <sup>i</sup>=1 <sup>Z</sup>i. Recall that <sup>W</sup><sup>ˆ</sup> <sup>=</sup> <sup>U</sup> <sup>+</sup> <sup>V</sup>ν, where <sup>E</sup>V<sup>ν</sup> [Vν] = 0. Hence,

$$\begin{aligned} \mathbb{E}_{Z \sim \mu[Z]} &= \frac{1}{n} \sum_{i=1}^n Z_i. \text{ Recall that } W = U + V_W, \text{ where } \mathbb{E}_{V_W}[V_W] = 0. \text{ Hence, } \\ \mathbb{E}_{\Theta} \left[ \left\| \mathbb{E}_{\hat{W}, W, S_n} \left[ \Delta(W, \Theta \hat{W}; S_n) \right] \right\| \right] &= \mathbb{E}_{\Theta} \left[ \left\| \mathbb{E}_{W, S_n} \left[ -\langle W, \bar{Z} \rangle + \langle U, \Theta^{\top} \bar{Z} \rangle \right] \right\| \right] \\ &\leq \mathbb{E}_{\Theta, W, S_n} \left[ \left\| -\langle W, \bar{Z} \rangle + \langle U, \Theta^{\top} \bar{Z} \rangle \right\| \right] \\ &\leq \mathbb{E}_{S_n, W} \mathbb{E}_{\Theta} \left[ \left\| -\langle w, \bar{Z} \rangle + \langle U, \Theta^{\top} \bar{Z} \rangle \right\| \right]. \end{aligned}$$

Combining above equation with equation [44](#page-38-6) for ϕ(Z¯) = Z¯ gives

$$\mathbb{E}_{\Theta} \left[ \left\| \mathbb{E}_{\hat{W}, W, S_n} \left[ \Delta(W, \Theta \hat{W}; S_n) \right] \right\| \right] \leq e^{-0.21d(1-e_w^2)^2} + \mathbb{E}_{S_n} \left[ \|\bar{Z}\| \right] \mathcal{O}\left(\frac{1}{\sqrt{d}}\right).$$

Next, we know by equation [35](#page-34-1) that <sup>E</sup>[∥Z¯∥] ≤ <sup>E</sup>[∥Z¯<sup>2</sup> <sup>1</sup>/<sup>2</sup> <sup>≤</sup> √<sup>2</sup> n . Hence,

$$\mathbb{E}_\Theta \left[ \left[ \mathbb{E}_{\hat{W}, W, S_n} \left[ \Delta(W, \Theta \hat{W}; S_n) \right] \right] \right] \leq e^{-0.2 d(1-c_w^2)^2} + \mathcal{O}\left(\frac{1}{\sqrt{dn}}\right),$$

$$d = n^{2r-1}$$
,  $c_w = 1.1$ ,  $\nu = 0.4$ .

#### G.4.2 Case ii.

Consider the construction of JL(d, cw, ν), described in Appendix [F.1.](#page-31-1) It is shown in equation [26](#page-31-3) that

$$\text{CMI}^\Theta(\tilde{\mathbf{S}}, \hat{\mathcal{A}}) = \leq d \log \left( \frac{c_w + \nu}{\nu} \right).$$

Hence, for any fixed Θ,

$$\text{CMI}^\Theta(\mu, \hat{A}) \leq d \log \left( \frac{c_w + \nu}{\nu} \right). \quad (55)$$

Furthermore, it is shown in equation [36](#page-34-2) that

$$\mathbb{E}_{P_{S_n, W} P_{\Theta} P_{W|\Theta^{\top} W}} \left[ \text{gen}(S_n, W) - \text{gen}(S_n, \Theta \hat{W}) \right] \leq \frac{2}{\sqrt{n}} \left( 1 + \frac{2}{d} \right)^{1/4} e^{-\frac{0.21}{4} d (c_w^2 - 1)^2}.$$

Hence, there exists at least one Θ for which

$$\mathbb{E}_{P_{S_n, W} P_{\hat{W} | \Theta^\top W}} \left[ \text{gen}(S_n, W) - \text{gen}(S_n, \Theta \hat{W}) \right] \leq \frac{2}{\sqrt{n}} \left( 1 + \frac{2}{d} \right)^{1/4} e^{-\frac{0.21}{4} d(c_w^2 - 1)^2}. \quad (56)$$

Choose this matrix Θ ∈ R D×d and the Markov Kernel PW<sup>ˆ</sup> <sup>|</sup>Θ⊤<sup>W</sup> induced by that. Call the overall algorithms as A ∗ <sup>Θ</sup>,n : Z <sup>n</sup> → <sup>R</sup> d , with the choices

$$d = 500r \log(n), \quad c_w = 1.1, \quad \nu = 0.4.$$

Plugging these constants in equation [55](#page-48-1) and equation [56](#page-48-2) completes the proof.

# G.5 Proof of Theorem [9](#page-24-2)

We first provide the proof of Theorem [9.](#page-24-2) The proof for the comment after the theorem, i.e., to show equation [14](#page-24-5) instead of equation [13,](#page-24-4) then follows similarly to the below proof, in a similar manner shown in the Case ii part of the proof of Theorem [8.](#page-24-0)

To prove Theorem [9,](#page-24-2) we follow the Case i part of the proof of Theorem [8,](#page-24-0) with a slight modification: Z¯ is replaced by <sup>Z</sup>, which results in convergence rates roughly √ n larger than the current ones. For the sake of completeness, we provide the proof.

Let

$$\Delta \mathcal{L}(W, \Theta \hat{W}) := \mathcal{R}(W) - \mathcal{R}(\Theta \hat{W}).$$

Following similarly to the Case i part of the proof of Theorem [8,](#page-24-0) it is sufficient to show that for any r < 1/2, letting

$$d = n^{2r}, \quad , c_w = 1.1, \quad , \nu = 0.4,$$

results in

$$\mathfrak{E}_1 \triangleq \mathbb{E}_\Theta \left[ \left\| \mathbb{E}_{\hat{W}, W} \left[ \Delta \mathcal{L}(W, \Theta \hat{W}) \right] \right\| \right] = \mathcal{O} \left( \frac{1}{n^r} \right). \quad (57)$$

Hence, it remains to show that equation [57](#page-48-3) holds. We have

$$\begin{aligned}\Delta\mathcal{L}(W, \Theta\hat{W}) &= -\mathbb{E}_{Z \sim \mu} \left[ \langle W, Z \rangle + \langle \Theta\hat{W}, Z \rangle \right] \\ &= -\mathbb{E}_{Z \sim \mu} \left[ \langle W, Z \rangle + \langle \hat{W}, \Theta^\top Z \rangle \right] \\ &= -\langle W, \mathbb{E}_{Z \sim \mu}[Z] \rangle + \langle \hat{W}, \Theta^\top \mathbb{E}_{Z \sim \mu}[Z] \rangle.\end{aligned}$$

Denote <sup>z</sup>˜ <sup>≜</sup> <sup>E</sup>Z∼µ[Z]. Hence, since <sup>W</sup><sup>ˆ</sup> <sup>=</sup> <sup>U</sup> <sup>+</sup> <sup>V</sup>ν, where <sup>E</sup>V<sup>ν</sup> [Vν] = 0, we have

$$\begin{aligned} \mathbb{E}_\Theta \left[ \left\| \mathbb{E}_{\hat{W}, W} \left[ \Delta \mathcal{L}(W, \Theta \hat{W}) \right] \right\| \right] &= \mathbb{E}_\Theta \left[ \left\| \mathbb{E}_W \left[ -\langle W, \tilde{z} \rangle + \langle U, \Theta^\top \tilde{z} \rangle \right] \right\| \right] \\ &\leq \mathbb{E}_{\Theta, W} \left[ \left\| \langle W, \tilde{z} \rangle - \langle U, \Theta^\top \tilde{z} \rangle \right\| \right]. \end{aligned}$$

Combining the above equation with equation [44,](#page-38-6) and by replacing ϕ(Z¯) by z˜, gives

$$\mathbb{E}_{\Theta} \left[ \left\| \mathbb{E}_{\hat{W}, W} \left[ \Delta \mathcal{L}(W, \Theta \hat{W}) \right] \right\| \right] \leq e^{-0.21 d(c_w^2 - 1)^2} + \mathcal{O}\left(\frac{1}{\sqrt{d}}\right).$$

$$d = n^{2r}, \quad c_w = 1.1, \quad \nu = 0.4.$$

#### G.6 Proof of Lemma [2](#page-25-0)

Consider the the JL(d, cw, ν) transformation described in Appendix [F.1](#page-31-1) with some d ∈ N <sup>+</sup>, c<sup>w</sup> ∈ h 1, p 5/4 , and ν ∈ (0, 1]. Recall that Wˆ = U + Vν, where V<sup>ν</sup> be a random variable that takes value uniformly over B<sup>d</sup> (ν) and

$$U := \begin{cases} \Theta^\top w, & \text{if } \|\Theta^\top w\| \leq c_w, \\ \mathbf{0}_d, & \text{otherwise.} \end{cases}$$

Let E be the event that ∥Θ <sup>⊤</sup>w∥ > c<sup>w</sup> and denote by E c the complementary event of E. We have

$$\begin{aligned} \mathbb{E}_{\Theta, V_\nu} \left[ \left\| \Theta \hat{W} \right\|^2 \right] &\stackrel{(a)}{\geq} \mathbb{E}_\Theta \left[ \|\Theta U\|^2 \right] - \mathbb{E}_{\Theta, V_\nu} \left[ \|\Theta V_\nu\|^2 \right] \\ &\stackrel{(b)}{=} \mathbb{E}_\Theta \left[ \|\Theta U\|^2 \right] - \frac{D}{d} \mathbb{E}_{V_\nu} \left[ \|V_\nu\|^2 \right] \\ &\stackrel{(c)}{\geq} \mathbb{E}_\Theta \left[ \|\Theta U\|^2 \right] - \frac{D\nu^2}{d} \\ &\stackrel{(d)}{=} \mathbb{E}_\Theta \left[ \left\| \Theta \Theta^\top w \right\|^2 \mathbb{1} \{ \mathcal{E}^c \} \right] - \frac{D\nu^2}{d} \\ &= \mathbb{E}_\Theta \left[ \left\| \Theta \Theta^\top w \right\|^2 \right] - \mathbb{E}_\Theta \left[ \left\| \Theta \Theta^\top w \right\|^2 \mathbb{1} \{ \mathcal{E} \} \right] - \frac{D\nu^2}{d} \\ &\stackrel{(e)}{\geq} \mathbb{E}_\Theta \left[ \left\| \Theta \Theta^\top w \right\|^2 \right] - \mathbb{E}_\Theta \left[ \left\| \Theta \Theta^\top w \right\|^4 \right]^{1/2} \mathbb{E}_\Theta [\mathbb{1} \{ \mathcal{E} \}]^{1/2} - \frac{D\nu^2}{d} \\ &\stackrel{(f)}{\geq} \mathbb{E}_\Theta \left[ \left\| \Theta \Theta^\top w \right\|^2 \right] - \mathbb{E}_\Theta \left[ \left\| \Theta \Theta^\top w \right\|^4 \right]^{1/2} e^{-0.1d(c_w^2 - 1)^2} - \frac{D\nu^2}{d} \\ &\stackrel{(g)}{=} \left( \frac{D+d+1}{d} \right) \|w\|^2 - \sqrt{\frac{(D+d+3)(D+d+5)(d+2)}{d^3}} \|w\|^2 e^{-0.1d(c_w^2 - 1)^2} - \frac{D\nu^2}{d}, \end{aligned}$$

where

- (a) follows by the triangle inequality,
- (b) follows by noting that each element of Θ is i.i.d. with distribution N (0, 1/d),
- (c) holds since V<sup>ν</sup> ∈ B<sup>d</sup> (ν),
- (d) is derived by the definition of U and E,
- (e) follows using Cauchy-Schwarz inequality,
- (f) is derived in equation [32,](#page-33-1)
- and (g) followed by following relations

$$\mathbb{E}_\Theta \left[ \left\| \Theta \Theta^\top w \right\|^2 \right] = \left( \frac{D + d + 1}{d} \right) \|w\|^2,$$

$$\mathbb{E}_\Theta \left[ \left\| \Theta \Theta^\top w \right\|^4 \right] = \frac{(D + d + 3)(D + d + 5)(d + 2)}{d^3} \|w\|^4,$$

shown below.

*Proof of norm two.* Note that E<sup>Θ</sup> h ΘΘ⊤<sup>w</sup> i scales with ∥w∥. Hence, it suffices to assume that ∥w∥ = 1. Next, first we show that E<sup>Θ</sup> h ΘΘ⊤<sup>w</sup> 2 i is the same for any w with ∥w∥ = 1.

For any w ∈ R <sup>D</sup>, there exists an orthonormal matrix Q ∈ <sup>R</sup> <sup>D</sup>×<sup>D</sup> such that QQ<sup>⊤</sup> = I<sup>D</sup> and Qw = e<sup>1</sup> ≜ [1, 0, 0, · · · , 0]<sup>⊤</sup>. This matrix can be constructed by letting the first row as w <sup>⊤</sup>, and choosing the other rows orthogonal to w <sup>⊤</sup>. Next, by letting Θ ′ = QΘ, we can write

$$\begin{aligned} \left\| \Theta \Theta^\top w \right\|^2 &= w^\top \Theta \Theta^\top \Theta \Theta^\top w \\ &= e_1^\top Q \Theta \Theta^\top \Theta \Theta^\top Q^\top e_1 \\ &= e_1^\top \Theta' \Theta'^\top \Theta' \Theta'^\top e_1 \\ &= \left\| \Theta' \Theta'^\top e_1 \right\|^2. \end{aligned}$$

The result follows by noting that E h Θ ′Θ ′⊤e<sup>1</sup> 2 i = E h ΘΘ⊤e<sup>1</sup> 2 i , since the distribution of Θ is rotationally invariant.

Hence, it is sufficient to compute E h ΘΘ⊤e<sup>1</sup> i . Denote the elements of Θ by θi,j , where i ∈ [D], j ∈ [d]. Then, simple algebra gives

$$\mathbb{E} \left[ \left\| \Theta \Theta^\top e_1 \right\|^2 \right] = \mathbb{E} \left[ \sum_{i \in [D]} \sum_{j,j' \in [d]^2} \theta_{i,j} \theta_{i',j'} \theta_{1,j} \theta_{1,j'} \right].$$

We know that for θ ∼ N (0, 1/d),

$$\mathbb{E}[\theta^m] = 0 \text{ for odd } m , \quad \mathbb{E}[\theta^2] = \frac{1}{d}, \quad \mathbb{E}[\theta^4] = \frac{3}{d^2}.$$

Then, it suffices to consider terms in the expansions that are non-zero, i.e. the terms where only even norms of each random variable appear. We consider all such cases:

- 1. i ̸= 1: D − 1 choices 1.1. j = j ′ : d choices and and the expectation of each term equals <sup>1</sup> <sup>d</sup><sup>2</sup> .
- 2. i = 1: 1 choice 2.1. j = j ′ : d choices and and the expectation of each term equals <sup>3</sup> <sup>d</sup><sup>2</sup> . 2.2. j ̸= j ′ : d(d − 1) choices and and the expectation of each term equals <sup>1</sup> <sup>d</sup><sup>2</sup> .

Summing all terms and factorizing properly gives

$$\begin{aligned}\mathbb{E} \left[ \left\| \Theta \Theta^\top e_1 \right\|^2 \right] &= \mathbb{E} \left[ \sum_{i \in [D]} \sum_{j, j' \in [d]^2} \theta_{i, j} \theta_{i', j'} \theta_{1, j} \theta_{1, j'} \right] \\ &= \frac{D + d + 1}{d}.\end{aligned}$$

*Proof of norm four.* Note that E<sup>Θ</sup> h ΘΘ⊤<sup>w</sup> 4 i scales with ∥w∥. Hence, it suffices to assume that ∥w∥ = 1. Next, similar to the proof of norm two, it can be shown that E<sup>Θ</sup> h ΘΘ⊤<sup>w</sup> 4 i is the same for any w with ∥w∥ = 1. Hence, it is sufficient to compute <sup>E</sup> h ΘΘ⊤e<sup>1</sup> 4 i . Denote the elements of Θ by θi,j , where i ∈ [D], j ∈ [d]. Then, simple algebra gives

$$\mathbb{E} \left[ \left\| \Theta \Theta^\top e_1 \right\|^4 \right] = \mathbb{E} \left[ \sum_{i,i' \in [D]^2} \sum_{j_1,j_2,j'_1,j'_2 \in [d]^4} \theta_{i,j_1} \theta_{i,j_2} \theta_{i',j'_1} \theta_{i',j'_2} \theta_{1,j_1} \theta_{1,j_2} \theta_{1,j'_1} \theta_{1,j'_2} \right].$$

We know that for θ ∼ N (0, 1/d),

$$\mathbb{E}[\theta^m] = 0 \text{ for odd } m, \quad \mathbb{E}[\theta^2] = \frac{1}{d}, \quad \mathbb{E}[\theta^4] = \frac{3}{d^2}, \quad \mathbb{E}[\theta^6] = \frac{15}{d^3}, \quad \mathbb{E}[\theta^8] = \frac{105}{d^4}.$$

Then, it suffices to consider terms in the expansions that are non-zero, i.e. the terms where only even norms of each random variable appear. We consider all such cases:

- 1. i = i ′ ̸= 1: D − 1 choices 1.1. j<sup>1</sup> = j<sup>2</sup> = j ′ <sup>1</sup> = j ′ <sup>2</sup>: d choices, and the expectation of each term equals <sup>9</sup> <sup>d</sup><sup>4</sup> . 1.2. Two of (j1, j2, j′ 1, j′
  - <sup>2</sup>) are the same, and two others as well, with a different value: 3d(d−1) choices, and the expectation of each term equals <sup>1</sup> <sup>d</sup><sup>4</sup> .

Hence, the sum of the expectation of the terms for this case equals:

$$3d^{-3}(D-1)(d+2).$$

- 2. i, i′ ̸= 1 and i ̸= i ′ : (D − 1)(D − 2) choices 2.1. j<sup>1</sup> = j<sup>2</sup> = j ′ <sup>1</sup> = j ′ <sup>2</sup>: d choices, and the expectation of each term equals <sup>3</sup> <sup>d</sup><sup>4</sup> . 2.2. j<sup>1</sup> = j<sup>2</sup> and different from j ′ <sup>1</sup> = j ′ <sup>2</sup>: d(d − 1) choices and the expectation of each term equals <sup>1</sup> <sup>d</sup><sup>4</sup> .

Hence, the sum of the expectation of the terms for this case equals:

$$d^{-3}(D-1)(D-2)(d+2).$$

- 3. i = 1 and i ̸= 1 or i ′ = 1 and i ̸= 1: 2(D − 1) choices 3.1. j<sup>1</sup> = j<sup>2</sup> = j ′ <sup>1</sup> = j ′ <sup>2</sup>: d choices, and the expectation of each term equals <sup>15</sup> <sup>d</sup><sup>4</sup> . 3.2. j<sup>1</sup> = j<sup>2</sup> and different from j ′ <sup>1</sup> = j ′ <sup>2</sup>: d(d − 1) choices and the expectation of each term equals <sup>3</sup> <sup>d</sup><sup>4</sup> . 3.3. j<sup>1</sup> different from j ′ <sup>1</sup> = j ′ <sup>2</sup> = j2: d(d − 1) choices and the expectation of each term equals 3 <sup>d</sup><sup>4</sup> . 3.4. j<sup>2</sup> different from j ′ <sup>1</sup> = j ′ <sup>2</sup> = j1: d(d − 1) choices and the expectation of each term equals <sup>d</sup><sup>4</sup> . 3.5. j<sup>1</sup> ̸= j<sup>2</sup> and both different from j ′ <sup>1</sup> = j ′ <sup>2</sup>: d(d − 1)(d − 2) choices and the expectation of each term equals <sup>1</sup> <sup>d</sup><sup>4</sup> .

′

Hence, the sum of the expectation of the terms for this case equals:

$$2d^{-3}(D-1)(15+9(d-1)+(d-1)(d-2)=2d^{-3}(D-1)(d+2)(d+4).$$

#### 4. i = i ′ = 1 : 1 choice

- 4.1. j<sup>1</sup> = j<sup>2</sup> = j ′ <sup>1</sup> = j ′ <sup>2</sup>: d choices, and the expectation of each term equals <sup>105</sup> <sup>d</sup><sup>4</sup> . 4.2. Exactly three of the indices among (j1, j2, j′ 1, j′
- <sup>2</sup>) are the same: 4d(d − 1) choices and the expectation of each term equals <sup>15</sup> <sup>d</sup><sup>4</sup> . 4.3. Two of (j1, j2, j′ 1, j′
- <sup>2</sup>) are the same, and two others as well, with a different value: 3d(d−1) choices, and the expectation of each term equals <sup>9</sup> <sup>d</sup><sup>4</sup> . 4.4. There are exactly two same indices among (j1, j2, j′ 1, j′ <sup>2</sup>): 6d(d − 1)(d − 2) choices and the expectation of each term equals <sup>3</sup> <sup>d</sup><sup>4</sup> . 4.5. All indices among (j1, j2, j′ 1, j′
  - <sup>2</sup>) are different: d(d − 1)(d − 2)(d − 3) choices and the expectation of each term equals <sup>1</sup> <sup>d</sup><sup>4</sup> .

Hence, the sum of the expectation of the terms for this case equals:

$$\begin{aligned} d^{-3}(105 + 60(d - 1) + 27(d - 1) + 18(d - 1)(d - 2) + (d - 1)(d - 2)(d - 3)) \\ = d^{-3}(d + 2)(d + 4)(d + 6). \end{aligned}$$

Finally, summing all terms and factorizing properly gives

$$\mathbb{E} \left[ \left\| \Theta \Theta^\top e_1 \right\|^4 \right] = \mathbb{E} \left[ \sum_{i,i' \in [D]^2} \sum_{j_1, j_2, j'_1, j'_2 \in [d]^4} \theta_{i,j_1} \theta_{i,j_2} \theta_{i',j'_1} \theta_{i',j'_2} \theta_{1,j_1} \theta_{1,j_2} \theta_{1,j'_1} \theta_{1,j'_2} \right]$$

$$= \frac{(D + d + 3)(D + d + 5)(d + 2)}{d^3}.$$

#### G.7 Proof of Lemma [5](#page-44-0)

To prove this lemma, we show the below stronger result:

$$\sum_{i \in [n]} \left| \mathbb{E}_{W, \tilde{\mathbf{s}}_{i, [2]}, J_i, \mathcal{Q}} \left[ \mathbb{1} \left\{ \mathcal{Q}(W, Z_i, J_i^c, \mu) = 'in' \right\} - \mathbb{1} \left\{ \mathcal{Q}(W, Z_i, J_i, \mu) = 'in' \right\} \right] \right| = o(n),$$

which results also

$$\mathbb{E}_{W, \tilde{\mathbf{S}}, \mathbf{J}, \mathcal{Q}} \left[ \sum_{i \in [n]} (\mathbb{1} \{ \mathcal{Q}(W, Z_i, J_i^e, \mu) = '\text{not in}' \} - \mathbb{1} \{ \mathcal{Q}(W, Z_i, J_i, \mu) = '\text{not in}' \} ) \right] = o(n).$$

For a given (W, Zi,j ), denote

$$P_{\mathcal{Q}} \left( \mathcal{Q}(W, Z_{i,j}, \mu) = \text{'in'} \right) = \mathbb{E}_{\mathcal{Q}} \left[ \mathbb{1} \left\{ \mathcal{Q}(W, Z_{i,j}, \mu) = \text{'in'} \right\} \right] \triangleq p(W, Z_{i,j}),$$

where the probability and expectation with respect to Q refer to the stochasticity of the adversary. Note that p (W, Zi,j ) is a measurable function of (W, Zi,j ).

For r ∈ {0, 1, . . . , 2 <sup>n</sup> − 1}, denote its binary representation as r = (br,1, . . . , br,n), where br,i ∈ {0, 1}. Now, consider 2 n *auxiliary* estimators, indexed by r ∈ {0, 1, . . . , 2 <sup>n</sup> − 1} and defined as follows. The estimator r, for the i-th sample, by having access to (W, Zi,0, Zi,1) estimates J<sup>i</sup> as

$$\hat{J}_i = \begin{cases} 0, & \text{with probability } \frac{1 + (-1)^{b_{r,i}} p(W, Z_{i,0}) - (-1)^{b_{r,i}} p(W, Z_{i,1})}{2} \\ 1, & \text{with probability } \frac{1 - (-1)^{b_{r,i}} p(W, Z_{i,0}) + (-1)^{b_{r,i}} p(W, Z_{i,1})}{2} \end{cases}$$

Note that each of these estimators makes its estimations only by having access to (W, Zi,0, Zi,1).

Define the Hamming distance d<sup>H</sup> : {0, 1} <sup>n</sup> × {0, 1} <sup>n</sup> → [n] between binary vectors J and Jˆ as

$$d_H\left(\mathbf{J}, \hat{\mathbf{J}}\right) = \sum_{i \in [n]} \mathbb{1}\{J_i \neq \hat{J}_i\}.$$

We now compute the expectation of <sup>d</sup>H(J, <sup>J</sup>ˆ) for the r-th estimator, *i.e.,* <sup>E</sup>W,S˜,J,J<sup>ˆ</sup> h dH(J, Jˆ) i . Note that due to the symmetry of S˜, we can only consider the case where J = (1, 1, . . . , 1) := 1n.

$$\begin{aligned}\mathbb{E}_{W, \tilde{\mathbf{S}}, \mathbf{J}, \hat{\mathbf{j}}} \left[ d_H(\mathbf{J}, \hat{\mathbf{J}}) \right] &= \mathbb{E}_{W, \tilde{\mathbf{S}}, \hat{\mathbf{j}} | \mathbf{J} = \mathbf{1}_n} \left[ d_H(\mathbf{1}_n, \hat{\mathbf{J}}) \right] \\ &= \sum_{i \in [n]} \mathbb{E}_{W, \tilde{\mathbf{S}}_{i, [2]}, \hat{J}_i | J_i = 1} \left[ d_H(1, \hat{J}_i) \right] \\ &= \sum_{i \in [n]} \mathbb{E}_{W, \tilde{\mathbf{S}}_{i, [2]} | J_i = 1} \left[ \frac{1 + (-1)^{b_{r, i}} p(W, Z_{i, 0}) - (-1)^{b_{r, i}} p(W, Z_{i, 1})}{2} \right] \\ &= \frac{n}{2} + \frac{1}{2} \sum_{i \in [n]} (-1)^{b_{r, i}} \mathbb{E}_{W, \tilde{\mathbf{S}}_{i, [2]} | J_i = 1} [p(W, Z_{i, 0}) - p(W, Z_{i, 1})] \\ &= \frac{n}{2} + \frac{1}{2} \sum_{i \in [n]} (-1)^{b_{r, i}} \mathbb{E}_{W, \tilde{\mathbf{S}}_{i, [2]}, J_i} [p(W, Z_i, J_i^c) - p(W, Z_i, J_i)] \\ &= \frac{n}{2} + \frac{1}{2} \sum_{i \in [n]} (-1)^{b_{r, i}} \mathbb{E}_{W, \tilde{\mathbf{S}}_{i, [2]}, J_i, \mathcal{Q}} \left[ \mathbb{1} \left\{ \mathcal{Q}(W, Z_i, J_i^c, \mu) = \text{'in'} \right\} \right. \\ &\quad \left. - \mathbb{1} \left\{ \mathcal{Q}(W, Z_i, J_i, \mu) = \text{'in'} \right\} \right]\end{aligned}$$

Then, there exists an estimator r ∗ , for which

$$\mathbb{E}_{W, \tilde{\mathbf{S}}, \mathbf{J}, \hat{\mathbf{J}}} \left[ d_H(\mathbf{J}, \hat{\mathbf{J}}) \right] = \frac{n}{2} - \frac{1}{2} \sum_{i \in [n]} \left| \mathbb{E}_{W, \tilde{\mathbf{S}}_{i, [2]}}, J_i, \mathcal{Q} \left[ \mathbb{1} \left\{ \mathcal{Q}(W, Z_i, J_i^c, \mu) = '\text{in}' \right\} - \mathbb{1} \left\{ \mathcal{Q}(W, Z_i, J_i, \mu) = '\text{in}' \right\} \right] \right|.$$

Now, suppose by contradiction that

$$\sum_{i \in [n]} \left| \mathbb{E}_{W, \tilde{\mathbf{s}}_{i, [2]}, J_i, \mathcal{Q}} \left[ \mathbb{1} \left\{ \mathcal{Q}(W, Z_i, J_i^c, \mu) = '\text{in}' \right\} - \mathbb{1} \left\{ \mathcal{Q}(W, Z_i, J_i, \mu) = '\text{in}' \right\} \right] \right|,$$

is not o(n). This means that there exists some b<sup>1</sup> ∈ <sup>R</sup>+ and a sequence {ai}i∈<sup>N</sup> such that limi→∞ a<sup>i</sup> = ∞ and limiting n to this subsequence, we have

$$\sum_{i \in [n]} \left| \mathbb{E}_{W, \tilde{\mathbf{S}}_{i, [2]}, J_i, \mathcal{Q}} \left[ \mathbb{1} \left\{ \mathcal{Q}(W, Z_i, J_i^c, \mu) = \text{'in'} \right\} - \mathbb{1} \left\{ \mathcal{Q}(W, Z_i, J_i, \mu) = \text{'in'} \right\} \right] \right| \geq 2b_1 n.$$

Without loss of generality, we can assume that b<sup>1</sup> ∈ (0, 1/4). Then, for the estimator r ∗ ,

$$\mathbb{E}_{W, \tilde{\mathbf{S}}, \mathbf{J}, \hat{\mathbf{J}}} \left[ d_H(\mathbf{J}, \hat{\mathbf{J}}) \right] \leq \frac{n(1 - 2b_1)}{2}.$$

Next, we use Fano's inequality with approximate recovery [\[59,](#page-13-1) Theorem 2]. Let t = n j (1−b1)n 2 k and denote

$$\begin{aligned} P_{e_t} &\triangleq \mathbb{P}\left(d_H\left(\mathbf{J}, \hat{\mathbf{J}}\right) > nt\right), \\ N_{\hat{\mathbf{j}}} &\triangleq \sum_{\mathbf{j} \in \{0,1\}^n} \mathbb{1}\left\{d_H(\mathbf{j}, \hat{\mathbf{j}}) \leq nt\right\}. \end{aligned}$$

It is easy to not that N<sup>ˆ</sup><sup>j</sup> is the same for all ˆj ∈ {0, 1} n . Hence, the maximum over <sup>ˆ</sup><sup>j</sup> of <sup>N</sup><sup>ˆ</sup><sup>j</sup> is equal to N<sup>1</sup><sup>n</sup> . With these notations, we have

$$\begin{aligned} o(n) &\stackrel{(a)}{\geq} \mathbf{I}(\mathbf{J}; W|\tilde{\mathbf{S}}) \\ &\stackrel{(b)}{=} \mathbf{I}(\mathbf{J}; W, \hat{\mathbf{J}}|\tilde{\mathbf{S}}) \\ &\stackrel{(c)}{\geq} \mathbf{I}(\mathbf{J}; \hat{\mathbf{J}}|\tilde{\mathbf{S}}) \\ &\stackrel{(d)}{\geq} \mathbf{I}(\mathbf{J}; \hat{\mathbf{J}}) \\ &\stackrel{(e)}{\geq} (1 - P_{e_t}) \log \left( \frac{2^n}{N_{\mathbf{1}_n}} \right) - \log(2) \\ &\stackrel{(f)}{\geq} n (1 - P_{e_t}) (1 - h_b(t)) - (1 - P_{e_t}) \log(3) - \log(2), \end{aligned}$$

where (a) is by construction of W and as shown in the proof of Theorem [3,](#page-5-1) (b) is derived since Jˆ is a function of (W, S˜), (c) is derived due to positivity of the mutual information, (d) is derived due to the below relations

$$\mathbf{l}(\mathbf{J}; \hat{\mathbf{J}}|\tilde{\mathbf{S}}) = H(\mathbf{J}) - H(\mathbf{J}|\hat{\mathbf{J}}, \tilde{\mathbf{S}}) \geq H(\mathbf{J}) - H(\mathbf{J}|\hat{\mathbf{J}}) = \mathbf{l}(\mathbf{J}; \hat{\mathbf{J}}),$$

(e) is derived using [\[59,](#page-13-1) Theorem 2], and (f) is derived using the claim, proved later below, that N<sup>1</sup><sup>n</sup> ≤ c32 nhb(t) for some constant c ∈ <sup>R</sup><sup>+</sup> and for n sufficiently large.

Note that t = 1 n j (1−b1)n k < 1/2 and as n → ∞, 1−hb(t) converges to the constant 1−h<sup>b</sup> 1−b<sup>1</sup> 2 > 0. Hence, if we show that for sufficiently large n, 1 − P<sup>e</sup><sup>t</sup> > 1 − b2, for some constant b<sup>2</sup> ∈ (0, 1), the contradiction is achieved. Since the left-hand side is of order o(n), which is greater than the right-hand side, which is Ω(n), and the proof is complete.

Hence, it remains to show for n sufficiently larg i) N<sup>1</sup><sup>n</sup> ≤ c32 nhb(t) for some constant c ∈ <sup>R</sup><sup>+</sup> and ii) P<sup>e</sup><sup>t</sup> < b2, for some constant b<sup>2</sup> ∈ (0, 1).

Proof of Claim i) This is shown in equation [49.](#page-41-0)

Proof of Claim ii) Using Markov's inequality, we have

$$\begin{aligned} P_{e_t} &\triangleq \mathbb{P}\left(d_H(\mathbf{J}, \hat{\mathbf{J}}) > nt\right) \\ &\leq \frac{\mathbb{E}_{W, \tilde{\mathbf{S}}, \mathbf{J}, \hat{\mathbf{J}}}\left[d_H(\mathbf{J}, \hat{\mathbf{J}})\right]}{nt} \\ &\leq \frac{(1 - 2b_1)}{2t} \\ &\leq \frac{1 - 2b_1}{1 - b_1 + 1/n} \end{aligned}$$

$$\begin{aligned} &= 1 - \frac{b_1 - \frac{1}{n}}{1 - b_1 + 1/n} \\ &\leq b_2, \end{aligned}$$

for some constant b<sup>2</sup> ∈ (1/2, 1) and n sufficiently large (or a<sup>i</sup> sufficiently large).

This completes the proof of the lemma.

# H Proofs of Appendix [D:](#page-26-0) Random subspace training algorithms

#### H.1 Proof of Lemma [3](#page-27-1)

Part i. For a = 0,

$$g_{a,p}(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}},$$

which is a standard Gaussian distribution. Hence, <sup>h</sup>(ga,p(x)) = log(√ 2πe) and f(a, p) = 0.

Part ii. The relation f(a, p) = f(−a, p) is trivial since by the symmetry of the distribution ga,p. To show the increasing behavior with respect to a, consider 0 ≤ a ′ < a and some p ∈ [0, 1]. We show f(a ′ , p) < f(a, p). For a > 0, let

$$X_1 = Y_1 + Ja, \quad X_2 = \frac{1}{a}X_1 = \frac{1}{a}Y_1 + J,$$

where Y<sup>1</sup> ∼ N (0, 1) is independent of J ∼ Bern(p). Then, it is easy to verify that

$$l(X_2; J) = l(X_1; J) = f(a, p). \quad (58)$$

Now let σ ≜ q a a′ <sup>2</sup> <sup>−</sup> <sup>1</sup> and define

$$X_3 = X_2 + \frac{1}{a} Y_2 = \frac{1}{a} (Y_1 + Y_2) + J, \quad (59)$$

where Y<sup>2</sup> ∼ N 0, σ<sup>2</sup> is independent of other random variables. Note that Y<sup>3</sup> ≜ a (Y1+Y2) a is independent of J and distributed according to N (0, 1). Hence, we can write

$$X_3 = \frac{1}{a'} Y_3 + J. \quad (60)$$

Now, we have

$$\begin{aligned} f(a, p) &\stackrel{(a)}{=} \mathbf{1}(X_2, J) \\ &\stackrel{(b)}{<} \mathbf{1}(X_3; J) \\ &\stackrel{(c)}{=} f(a', p), \end{aligned}$$

where (a) follows from equation [58,](#page-54-2) (b) from equation [59](#page-54-3) and the strong data processing inequality, and (c) from equation [60.](#page-54-4) This completes the proof of the strictly increasing behavior with respect to a in the range [0, ∞).

Part iii. Denote <sup>Q</sup>1(x) := √<sup>1</sup> 2π e − <sup>x</sup> 2 <sup>2</sup> and <sup>Q</sup>2(x) := √<sup>1</sup> 2π e − (x−a) 2 <sup>2</sup> . Note that ga,p(x) = pQ1(x) + (1 − p)Q2(x). Hence, h(ga,p(x)) = −pEQ<sup>1</sup> [log(ga,p(x))] − (1 − p)<sup>E</sup>Q<sup>2</sup> [log(ga,p(x))]. Now, considering the limit to infinity, we have

$$\begin{aligned} \lim_{a \rightarrow \infty} h(g_{a,p}(x)) &= -p\mathbb{E}_{Q_1}[\log(pQ_1(x))] - (1-p) \lim_{a \rightarrow \infty} \mathbb{E}_{Q_2}[\log((1-p)Q_2(x))] \\ &= -p\log(p) - (1-p)\log(1-p) - p\mathbb{E}_{Q_1}[\log(Q_1(x))] - (1-p) \lim_{a \rightarrow \infty} \mathbb{E}_{Q_2}[\log(Q_2(x))] \\ &\stackrel{(a)}{=} \log(2)h_b(p) + \frac{1}{2}\log(2\pi e), \end{aligned}$$

where (a) is deduced by noting that both Q<sup>1</sup> and Q<sup>2</sup> are Gaussian distributions with variance 1 and hence, their differential entropy is equal to <sup>1</sup> 2 log(2πe).

This concludes that lima→∞ f(a, p) = hb(p).

Part iv. f(a, p) = f(a, 1 − p) is trivial since by the symmetry of the distribution ga,p.

To show the strictly increasing behavior with respect to p, consider 0 ≤ p<sup>1</sup> < p<sup>2</sup> ≤ 1/2. Let

$$X_1 = Y + J_1 a,$$

where Y ∼ N (0, 1) is independent of J<sup>1</sup> ∼ Bern(p1). Then, due to Part ii,

$$l(X_1; J_1) = f(a, p_1) = h(g_{a,p}(x)) - \log(\sqrt{2\pi e}). \quad (61)$$

Moreover, note that

$$h(X_1) = h(g_{a,p_1}(x)) = h(g_{a,1-p_1}(x)). \quad (62)$$

Let Z ∼ Bern(q) be independent of other random variables for some q ∈ (0, 1) that will be determined later. Let

$$X_2 \triangleq Y + Va,$$

where V = |J<sup>1</sup> − Z|. Note that V ∼ Bern(p1q + (1 − p1)(1 − q)) is independent of Y .

Now, on the one hand, we have

$$h(X_2|V) = h(Y) = h(X_1|J_1). \quad (63)$$

On the other hand,

$$\begin{aligned} h(X_2) &\stackrel{(a)}{>} h(X_2|Z) \\ &= h(X_2|Z = 0) q + h(X_2|Z = 1) (1 - q) \\ &= h(Y + |J_1|a) q + h(Y + |J_1 - 1|a) (1 - q) \\ &\stackrel{(b)}{=} h(Y + J_1a) q + h(Y + J'_1a) (1 - q) \\ &\stackrel{(c)}{=} h(g_{a,p_1}(x)) q + h(g_{a,1-p_1}(x)) (1 - q) \\ &\stackrel{(d)}{=} h(g_{a,p_1}(x))) \\ &\stackrel{(e)}{=} h(X_1), \end{aligned}$$

where (a) is derived by strong data processing inequality and since p<sup>1</sup> ∈ [0, 1/2) and q ∈ (0, 1), (b) is derived for J ′ ∼ Bern(1 − p1) independent of Y , and steps (c), (d), (e) are derived using equation [62.](#page-55-1)

Hence, combining equation [61,](#page-55-2) equation [63,](#page-55-3) and equation [63,](#page-55-3) we have

$$\begin{aligned} f(a, p_1) &= l(X_1; J_1) \\ &< l(X_2; V) \\ &= f(a, p_1 q + (1 - p_1)(1 - q)). \end{aligned}$$

The proof completes by find a q ∈ [0, 1] such that p1q + (1 − p1)(1 − q) = p2. To show that such q exist, first denote e<sup>p</sup><sup>1</sup> (q) := p1q + (1 − p1)(1 − q). Now, note that e<sup>p</sup><sup>1</sup> (1) = p<sup>1</sup> < p<sup>2</sup> and e<sup>p</sup><sup>1</sup> (0) = 1 − p<sup>1</sup> > <sup>2</sup> ≥ p2. Hence, there exists a q <sup>∗</sup> ∈ (0, 1) such that ep(q ∗ ) = p2. This completes the proof of this part.

# H.2 Proof of Theorem [10](#page-28-0)

Recall that

$$V_t \triangleq \{i_{t,1}, \dots, i_{t,b}\},$$

is the set of sample indices chosen at time t ∈ [T], chosen independently of any other random variables. Hence,

$$\text{gen}(\mu, \mathcal{A}^{(d)}) = \mathbb{E}_{\mathbf{V}} \left[ \text{gen}(\mu, \mathcal{A}_{\mathbf{V}}^{(d)}) \right],$$

The proof consists of bounding each of the conditional mutual information terms

$$\text{CMI}_{\mathbf{V},i,\mathbf{J}-i}^{\Theta}(\tilde{\mathbf{S}}, W') \triangleq \Gamma^{\tilde{\mathbf{S}}, \mathbf{J}-i, \Theta}(\mathcal{A}_{\mathbf{V}}^{(d)}(\tilde{\mathbf{S}}_{\mathbf{J}}, \Theta); J_i), \quad i \in [n],$$

and then using the bound [15](#page-26-1) of Corollary [2,](#page-26-2) with Aˆ (d) <sup>V</sup> = A (d) <sup>V</sup> and ϵ = 0.

It is sufficient then to show that for a fixed V and every fixed i ∈ [n], we have that

$$\text{CMI}_{\mathbf{V}, \mathbf{J}_{-i}, i}^\Theta(\tilde{\mathbf{S}}, W') \leq \sum_{t: i \in V_t} \mathbb{E}_{p_{t,i}, \Delta_{t,i}} \left[ f \left( \frac{\eta_t}{b\sigma_t} \Delta_{t,i}, p_{t,i} \right) \right], \quad (64)$$

where

$$\begin{aligned} \Delta_{t,i} &\triangleq \|\nabla_{w'} \ell (\Theta W'_{t-1}, Z_{i,0}) - \nabla_{w'} \ell (\Theta W'_{t-1}, Z_{i,1})\|, \\ p_{t,i} &\triangleq \mathbb{P} \left( J_i = 0 \mid \tilde{\mathbf{S}}, \Theta, \mathbf{V}, \mathbf{J}_{-i}, W'_{t-1}, \{W'_r, W'_{r-1} : r < t, i \in V_r\} \right). \end{aligned}$$

For a fixed i ∈ [n], if {t: i ∈ Vt} is an empty set, then the final model is independent of J<sup>i</sup> and hence CMI<sup>Θ</sup> V,i,J−<sup>i</sup> (S˜, W′ ) = 0, which completes the proof. Now, assume that this set is not empty. For ease of notation, suppose that

$$\{t: i \in V_t\} = \{t_1, \dots, t_M\},$$

where 1 ≤ t<sup>1</sup> < t<sup>2</sup> < · · · < t<sup>M</sup> ≤ T.

Then, for a fixed V,

$$\begin{aligned}
\text{CMI}^{\Theta}_{\mathbf{V}, \mathbf{J}-i, i}(\tilde{\mathbf{S}}, W') &\triangleq |\tilde{\mathbf{S}}, \mathbf{J}-i, \Theta(\mathcal{A}_{\mathbf{V}}^{(d)}(\tilde{\mathbf{S}}\mathbf{J}, \Theta); J_i)| \\
&= |\tilde{\mathbf{S}}, \mathbf{J}-i, \Theta, \mathbf{V}(W'_T; J_i)| \\
&\stackrel{(a)}{\leq} |\tilde{\mathbf{S}}, \mathbf{J}-i, \Theta, \mathbf{V}(W'_{t_M}, W'_{t_M-1}, W'_{t_{M-1}}, W'_{t_{M-1}-1}, \dots, W'_{t_1}, W'_{t_1-1}; J_i)| \\
&\stackrel{(b)}{=} \sum_{m \in [M]} |\tilde{\mathbf{S}}, \mathbf{J}-i, \Theta, \mathbf{V}(W'_{t_m}, W'_{t_m-1}; J_i | W'_{t_{m-1}}, W'_{t_{m-1}-1}, \dots, W'_{t_1}, W'_{t_1-1})| \\
&\stackrel{(c)}{=} \sum_{m \in [M]} |\tilde{\mathbf{S}}, \mathbf{J}-i, \Theta, \mathbf{V}(W'_{t_m}; J_i | W'_{t_m-1}, W'_{t_m-1}, W'_{t_{m-1}-1}, \dots, W'_{t_1}, W'_{t_1-1})|,
\end{aligned}$$

where (a) holds since by the data processing inequality I <sup>S</sup>˜,J−i,Θ,<sup>V</sup>(W′ <sup>T</sup> ; Ji) ≤ I <sup>S</sup>˜,J−i,Θ,<sup>V</sup>(W′ <sup>t</sup><sup>M</sup> ; Ji) and I <sup>S</sup>˜,J−i,Θ,<sup>V</sup>(W′ <sup>t</sup><sup>M</sup> ; Ji) ≤ I <sup>S</sup>˜,J−i,Θ,<sup>V</sup>(W′ <sup>t</sup><sup>M</sup> , W′ <sup>t</sup><sup>M</sup> <sup>−</sup>1, W′ tM−<sup>1</sup> , W′ <sup>t</sup>M−1−1, · · · , W′ t1 , W′ <sup>t</sup>1−1; Ji) by the non-negativity of the mutual information, (b) is derived using the chain rule for the mutual information and by using the convention that when m = 1, the conditioning part {W′ tm−<sup>1</sup> , W′ <sup>t</sup>m−1−1, · · · , W′ t1 , W′ <sup>t</sup>1−1} is an empty set, and (c) is derived since I <sup>S</sup>˜,J−i,Θ,<sup>V</sup>(W′ <sup>t</sup>m−1; <sup>J</sup>i|, W′ tm−<sup>1</sup> , W′ <sup>t</sup>m−1−1, · · · , W′ t1 , W′ <sup>t</sup>1−1) = 0.

Consider a fixed value of (W′ <sup>t</sup>m−1, W′ tm−<sup>1</sup> , W′ <sup>t</sup>m−1−1, · · · , W′ t1 , W′ <sup>t</sup>1−1) and let

$$\mathcal{F}_m \triangleq \left\{ \tilde{\mathbf{S}}, \Theta, \mathbf{V}, \mathbf{J}_{-i}, W'_{t_m-1}, W'_{t_{m-1}}, W'_{t_{m-1}-1}, \dots, W'_{t_1}, W'_{t_1-1} \right\}.$$

Note that

$$\begin{aligned} p_{t_m,i} &\triangleq \mathbb{P}\left(J_i = 0 \mid \tilde{\mathbf{S}}, \Theta, \mathbf{V}, \mathbf{J}_{-i}, W'_{t_{m-1}}, \{W'_r, W'_{r-1} : r < t_m, i \in V_r\}\right) \\ &= \mathbb{P}(J_i = 0 \mid \mathcal{F}_m). \end{aligned}$$

Hence, it is sufficient to show that

$$|{}^{F_m}(W'_{t_m}; J_i) \leq f\left(\frac{\eta_{t_m}}{b\sigma_{t_m}}\Delta_{t_m,i}, p_{t_m,i}\right). \quad (65)$$

Recall that

$$W'_{t_m} = \text{Proj} \left\{ W'_{t_m-1} - \eta_{t_m} \nabla_{w'} \widehat{\mathcal{R}}(V_{t_m}, \Theta W'_{t_m-1}) + \sigma_{t_m} \varepsilon_{t_m} \right\},$$

where Rb(V<sup>t</sup><sup>m</sup> , W) <sup>≜</sup> 1 b P i ′∈Vtm ℓ Z<sup>i</sup> ,J<sup>i</sup> , W . Denote

$$\widehat{\mathcal{R}}_{-i}(V_{t_m}, W) \triangleq \frac{1}{b} \sum_{\substack{i': \\ i' \in V_{t_m}, i' \neq i}} \ell(Z_{i', J_{i'}}, W).$$

Furthermore, denote

$$\begin{aligned} \tilde{W}_{t_m} &\triangleq W'_{t_{m-1}} - \eta_{t_m} \nabla_{w'} \hat{\mathcal{R}}(V_{t_m}, \Theta W'_{t_{m-1}}) + \sigma_{t_m} \varepsilon_{t_m} \\ &= W'_{t_{m-1}} - \eta_{t_m} \nabla_{w'} \hat{\mathcal{R}}_{-i}(V_{t_m}, \Theta W'_{t_{m-1}}) - \frac{\eta_{t_m}}{b} \nabla_{w'} \ell(Z_i, J_i, \Theta W'_{t_{m-1}}) + \sigma_{t_m} \varepsilon_{t_m}, \end{aligned}$$

where the last line holds since by assumption i ∈ V<sup>t</sup><sup>m</sup> .

Using the data processing inequality, we have that

$$|\mathcal{F}_m(W'_{t_m}; J_i) \leq |\mathcal{F}_m(\tilde{W}_{t_m}; J_i)|.$$

Hence, it is sufficient to show that

$$|^{F_m}(\tilde{W}_{t_m}; J_i) \leq f\left(\frac{\eta_{t_m}}{b\sigma_{t_m}}\Delta_{t_m,i}, p_{t_m,i}\right). \quad (66)$$

Note that

$$|^{\mathcal{F}_m}(\tilde{W}_{t_m}; J_i) = |^{\mathcal{F}_m}(\tilde{W}_{t_m}/\sigma_{t_m}; J_i) = h^{\mathcal{F}_m}(\tilde{W}_{t_m}/\sigma_{t_m}) - h^{\mathcal{F}_m}(\tilde{W}_{t_m}/\sigma_{t_m}| J_i). \quad (67)$$

To compute each of the two terms in right-side of equation [67,](#page-57-0) first we derive the marginal and conditional distributions of <sup>1</sup> σtm <sup>W</sup>˜ <sup>t</sup><sup>m</sup> .

- Given F<sup>m</sup> and given J<sup>i</sup> = 0,

$$\frac{1}{\sigma_{t_m}} \tilde{W}_{t_m} = \frac{1}{\sigma_{t_m}} W'_{t_{m-1}} - \frac{\eta_{t_m}}{\sigma_{t_m}} \nabla_{w'} \hat{\mathcal{R}}_{-i}(V_{t_m}, \Theta W'_{t_{m-1}}) - \frac{\eta_{t_m}}{\sigma_{t_m}} \nabla_{w'} \ell(Z_{i,0}, \Theta W'_{t_{m-1}}) + \varepsilon_{t_m}.$$

Hence, given F<sup>m</sup> and given J<sup>i</sup> = 0, 1 σtm <sup>W</sup>˜ <sup>t</sup><sup>m</sup> is distributed as

$$\frac{1}{\sigma_{t_m}} \tilde{W}_{t_m} \sim \tilde{P}_0 \triangleq \mathcal{N}(\mu_0, \text{I}_d), \quad (68)$$

where

$$\mu_0 \triangleq \frac{1}{\sigma_{t_m}} W'_{t_m-1} - \frac{\eta_{t_m}}{\sigma_{t_m}} \nabla_{w'} \widehat{\mathcal{R}}_{-i}(V_{t_m}, \Theta W'_{t_m-1}) - \frac{\eta_{t_m}}{b \sigma_{t_m}} \nabla_{w'} \ell (Z_{i,0}, \Theta W'_{t_m-1}).$$

- Similarly, given F<sup>m</sup> and given J<sup>i</sup> = 1, σtm <sup>W</sup>˜ <sup>t</sup><sup>m</sup> is distributed as

$$\frac{1}{\sigma_{t_m}} \tilde{W}_{t_m} \sim \tilde{P}_1 \triangleq \mathcal{N}(\mu_1, I_d), \quad (69)$$

where

$$\mu_1 \triangleq \frac{1}{\sigma_{t_m}} W'_{t_m-1} - \frac{\eta_{t_m}}{\sigma_{t_m}} \nabla_{w'} \widehat{\mathcal{R}}_{-i}(V_{t_m}, \Theta W'_{t_m-1}) - \frac{\eta_{t_m}}{\sigma_{t_m}} \nabla_{w'} \ell(Z_{i,1}, \Theta W'_{t_m-1}).$$

- Lastly, since P J<sup>i</sup> = 0 F<sup>m</sup> = p<sup>t</sup>m,i, then given Fm, σtm <sup>W</sup>˜ <sup>t</sup><sup>m</sup> is distributed as

$$\begin{aligned} \frac{1}{\sigma_{t_m}} \tilde{W}_{t_m} &\sim \tilde{P} \triangleq p_{t_m,i} \tilde{P}_0 + (1 - p_{t_m,i}) \tilde{P}_1 \\ &= p_{t_m,i} \mathcal{N}(\mu_0, \text{Id}) + (1 - p_{t_m,i}) \mathcal{N}(\mu_1, \text{Id}). \end{aligned}$$

Now, we compute each of the two terms of h <sup>F</sup><sup>m</sup> (W˜ <sup>t</sup><sup>m</sup> /σ<sup>t</sup><sup>m</sup> ) and <sup>h</sup> <sup>F</sup><sup>m</sup> (W˜ <sup>t</sup><sup>m</sup> /σ<sup>t</sup><sup>m</sup> |Ji):

- The term h <sup>F</sup><sup>m</sup> (W˜ <sup>t</sup><sup>m</sup> /σ<sup>t</sup><sup>m</sup> ) equals the differential entropy <sup>h</sup>(P˜). Since the differential entropy is invariant under the shift and since also the Gaussian distributions P˜<sup>0</sup> and P˜<sup>1</sup> are invariant under the rotation, h(P˜) is equal to the entropy of the distribution Q˜, defined as

$$\mathbf{Q} \triangleq p_{t_m,i}\mathcal{N}(\mathbf{0}_d, \mathbf{I}_d) + (1 - p_{t_m,i})\mathcal{N}(\mathbf{a}_d, \mathbf{I}_d),$$

where

$$\mathbf{a}_d = \left( \frac{\eta_{t_m}}{b\sigma_{t_m}} \mu, 0, 0, \dots, 0 \right) \in \mathbb{R}^d,$$

and

$$\begin{aligned} \mu &\triangleq \frac{b\sigma_{t_m}}{\eta_{t_m}} \|\mu_1 - \mu_0\| \\ &= \|\nabla_{w'} \ell (\Theta W'_{t_m-1}, Z_{i,0}) - \nabla_{w'} \ell (\Theta W'_{t_m-1}, Z_{i,1})\| \\ &= \Delta_{t_m, i}. \end{aligned}$$

Note that ∥ad∥ = ∥µ<sup>1</sup> − µ0∥.

Furthermore, we can write

$$\mathbf{Q} = Q_1 \otimes Q_2 \otimes \cdots \otimes Q_d, \quad (70)$$

where

$$Q_1 = p_{t_m,i}\mathcal{N}(0,1) + (1 - p_{t_m,i})\mathcal{N}\left(\frac{\eta_{t_m}}{b\sigma_{t_m}}\Delta_{t_m,i},1\right),$$

and for r ∈ {2, 3, . . . , d},

$$Q_i = \mathcal{N}(0, 1).$$

Hence,

$$\begin{aligned} h^{\mathcal{F}_m}(\tilde{W}_{t_m}/\sigma_{t_m}) &= h(\tilde{P}) \\ &= h(\mathbf{Q}) \\ &\stackrel{(a)}{=} \sum_{r \in [d]} h(Q_r) \\ &\stackrel{(b)}{=} h(Q_1) + (d-1) \log(\sqrt{2\pi e}) \\ &\stackrel{(c)}{=} h(g_{a_1, p_{t_m, i}}(x)) + (d-1) \log(\sqrt{2\pi e}) \\ &\stackrel{(d)}{=} f\left(\frac{\eta_{t_m}}{b\sigma_{t_m}} \Delta_{t_m, i}, p_{t_m, i}\right) + d \log(\sqrt{2\pi e}), \end{aligned} \tag{71}$$

where (a) is derived by equation [70,](#page-58-2) (b) holds since the distributions Q2, . . . , Q<sup>d</sup> are scalar standard Gaussian distributions, (c) is derived for a<sup>1</sup> ≜ ηtm bσtm ∆<sup>t</sup>m,i and by the definition of ga,p(·) in [19,](#page-27-3) and (d) by the definition of f(a, p) in [18.](#page-27-4)

- To compute h <sup>F</sup><sup>m</sup> (W˜ <sup>t</sup><sup>m</sup> /σ<sup>t</sup><sup>m</sup> |Ji), note that for each value of <sup>J</sup>i, due to equation [68](#page-57-1) and equation [69,](#page-57-2) the conditional distribution of <sup>1</sup> σtm <sup>W</sup>˜ <sup>t</sup><sup>m</sup> is a multivariate Gaussian distribution with covariance <sup>I</sup>d. Hence,

$$h^{\mathcal{F}_m}(\tilde{W}_{t_m}/\sigma_{t_m}|J_i) = d \log(\sqrt{2\pi e}). \quad (72)$$

Combining equation [71](#page-58-3) and equation [72](#page-58-4) gives equation [66](#page-56-0) which completes the proof.

# H.3 Proof of Theorem [13](#page-28-1)

Recall that

$$W'_t = \text{Proj} \left\{ W'_{t-1} - \eta_t \nabla_{w'} \widehat{\mathcal{R}}(V_t, \Theta W'_{t-1}) + \sigma_t \varepsilon_t \right\},$$

where Rb(Vt, W) <sup>≜</sup> b P i∈V<sup>t</sup> ℓ (Zi,J<sup>i</sup> , W).

In the proof, to define the lossy compression algorithm PW<sup>ˆ</sup> <sup>|</sup>W′ ,Θ,S of Corollary [2,](#page-26-2) we introduce auxiliary optimization iterations n Wˆ <sup>t</sup> o , as follows. Let Wˆ <sup>0</sup> = W′ <sup>0</sup>, and for t ∈ [T], let

$$\hat{W}_t = \text{Proj} \left\{ \hat{W}_{t-1} - \eta_t \nabla_{\hat{w}} \hat{\mathcal{R}}(V_t, \Theta \hat{W}_{t-1}) + \sigma_t \varepsilon_t + \nu_t \varepsilon'_t \right\}, \quad (73)$$

where ε ′ <sup>t</sup> ∼ N (0d, Id) is an additional noise, independent of all other random variables.

In the following Lemma, proved in Appendix [H.4,](#page-61-0) we show that, this choice of PW<sup>ˆ</sup> <sup>|</sup>W′ ,Θ,S,<sup>V</sup> satisfies the distortion term equation [16:](#page-26-3)

$$\mathbb{E}_{P_S P_\Theta P_{\mathbf{V}} P_{W'_T|S,\Theta,\mathbf{v}} P_{\hat{W}_T|W'_T,S,\Theta,\mathbf{v}}} \left[ \text{gen}(S, \Theta W'_T) - \text{gen}(S, \Theta \hat{W}_T) \right] \leq \epsilon,$$

for

$$\epsilon := \frac{2\sqrt{2}\mathfrak{L}\Gamma((d+1)/2)}{\Gamma(d/2)} \sum_{t \in [T]} \alpha^{T-t} \nu_t.$$

Lemma 6. *The following inequalities holds:*

$$\left\| \hat{W}_t - W'_t \right\| \leq \sum_{r \in [t]} \alpha^{t-r} \nu_r \left\| \varepsilon'_r \right\|,$$

*and*

$$\mathbb{E} P_S P_\Theta P_V P_{W'_T | S, \Theta, \mathbf{v}} P_{\hat{W}_T | W'_T, S, \Theta, \mathbf{v}} \left[ \text{gen}(S, \Theta W'_T) - \text{gen}(S, \Theta \hat{W}_T) \right] \leq \frac{2\sqrt{2} \mathfrak{L} \Gamma((d+1)/2)}{\Gamma(d/2)} \sum_{t \in [T]} \alpha^{T-t} \nu_t.$$

Hence, it is sufficient to show that

$$\begin{aligned} & \mathbb{E}_{P_S} P_\Theta P_V P_{\hat{W}_{T|S,\Theta,\mathbf{V}}} \left[ \text{gen}(S, \Theta \hat{W}_T) \right] \\ & \leq \frac{C\sqrt{2}}{n} \sum_{i \in [n]} \mathbb{E}_{\mathbf{S}, \Theta, \mathbf{V}, \mathbf{J}_{-i}} \left[ \sqrt{\sum_{t: i \in V_t} A_{t,i} \mathbb{E}_{\hat{p}_{t,i}, \hat{\Delta}_{t,i}} \left[ f \left( \frac{\eta_t}{b\sqrt{\sigma_t^2 + \nu_t^2}} \hat{\Delta}_{t,i}, \hat{p}_{t,i} \right) \right]} \right]. \end{aligned}$$

Note that the iterations defined in equation [73](#page-58-5) are equivalent in distribution to the following iterations:

$$\hat{W}_t = \text{Proj} \left\{ \hat{W}_{t-1} - \eta_t \nabla_{\hat{w}} \hat{\mathcal{R}}(V_t, \Theta \hat{W}_{t-1}) + \hat{\sigma}_t \tilde{\epsilon}_t \right\},$$

where ε˜<sup>t</sup> ∼ N (0d, Id) is independent of all other random variables and

$$\hat{\sigma}_t \triangleq \sqrt{\sigma_t^2 + \nu_t^2}.$$

Similar to the proof of Theorem [10,](#page-28-0) and by using Corollary [2,](#page-26-2) it is sufficient to show that for a fixed V and every fixed i ∈ [n], we have that

$$\text{CMI}^\Theta_{\nabla, \mathbf{J}_{-i}, i}(\tilde{\mathbf{S}}, \hat{W}_T) \leq \sum_{t: i \in V_t} A_{t,i} \mathbb{E}_{\hat{p}_{t,i}, \hat{\Delta}_{t,i}} \left[ f \left( \frac{\eta_t}{\hat{\sigma}_t} \hat{\Delta}_{t,i}, \hat{p}_{t,i} \right) \right],$$

where

$$\begin{aligned} \text{CMI}^\Theta_{\mathbf{V}, \mathbf{J}_{-i}, i}(\tilde{\mathbf{S}}, \hat{W}_T) &\triangleq |\tilde{\mathbf{S}}, \mathbf{J}_{-i}, \Theta, \mathbf{V}(\hat{W}_T; J_i), \\ &\hat{\Delta}_{t,i} \triangleq \left\| \nabla_{w'\ell} \left( \Theta \hat{W}_{t-1}, Z_{i,0} \right) - \nabla_{w'\ell} \left( \Theta \hat{W}_{t-1}, Z_{i,1} \right) \right\|, \\ &\hat{p}_{t,i} \triangleq \mathbb{P} \left( J_i = 0 \mid \tilde{\mathbf{S}}, \Theta, \mathbf{V}, \mathbf{J}_{-i}, \hat{W}_{t-1} \right), \\ A_{t,i} &:= \prod_{r \in [t+1:T]: i \notin V_r} q_r. \end{aligned}$$

For a fixed i ∈ [n], if {t: i ∈ Vt} is an empty set, then the final model is independent of J<sup>i</sup> and hence CMI<sup>Θ</sup> V,i,J−<sup>i</sup> (S˜, Wˆ <sup>T</sup> ) = 0, which completes the proof. Now, assume that this set is not empty. For ease of notation, suppose that

$$\{t: i \in V_t\} = \{t_1, \dots, t_M\},$$

where 1 ≤ t<sup>1</sup> < t<sup>2</sup> < · · · < t<sup>M</sup> ≤ T.

We show by induction on m ∈ [M] that, we have

$$\text{CMI}_{\nabla}^{\Theta}, \mathbf{J}_{-i}, i (\tilde{\mathbf{S}}, \hat{W}_{t_m}) \leq \sum_{k \leq m} A_{t_k, i}^{t_m} \mathbb{E}_{\hat{p}_{t_k, i}, \hat{\Delta}_{t_k, i}} \left[ f \left( \frac{\eta_{t_k}}{b\hat{\sigma}_{t_k}} \hat{\Delta}_{t_k, i}, \hat{p}_{t_k, i} \right) \right], \quad (74)$$

where ∆ˆ t,i and pˆt,i are defined as above and

$$A_{t,i}^{t'} := \prod_{r \in [t+1:t'] : i \notin V_r} q_r,$$

Once this claim is shown, then we have

$$\begin{aligned} |\tilde{\mathbf{s}}, \mathbf{J}_{-i}, \Theta, \mathbf{V}(\hat{W}_T; J_i)| &\stackrel{(a)}{\leq} \left( \prod_{r=t_M+1}^T q_r \right) |\tilde{\mathbf{s}}, \mathbf{J}_{-i}, \Theta, \mathbf{V}(\hat{W}_{t_M}; J_i)| \\ &\stackrel{(b)}{\leq} \left( \prod_{r=t_M+1}^T q_r \right) \sum_{k \leq M} A_{t_k, i}^{t_M} \mathbb{E}_{\hat{p}_{t_k, i}, \hat{\Delta}_{t_k, i}} \left[ f \left( \frac{\eta_{t_k}}{b\hat{\sigma}_{t_k}} \hat{\Delta}_{t_k, i}, \hat{p}_{t_k, i} \right) \right] \\ &\stackrel{(b)}{=} \sum_{k \leq M} A_{t_k, i} \mathbb{E}_{\hat{p}_{t_k, i}, \hat{\Delta}_{t_k, i}} \left[ f \left( \frac{\eta_{t_k}}{b\hat{\sigma}_{t_k}} \hat{\Delta}_{t_k, i}, \hat{p}_{t_k, i} \right) \right], \end{aligned}$$

where

- (a) is achieved by repeated using of [\[74,](#page-13-13) Lemma 4],
- (b) is derived using equation [74,](#page-59-0)
- and (c) holds by definitions of A<sup>t</sup>k,i = A T <sup>t</sup>k,i and A t<sup>M</sup> <sup>t</sup>k,i.

Hence, it remains to show that equation [74](#page-59-0) holds by induction.

Consider the base of the induction m = 1. Note that A t1 <sup>t</sup>1,i = 1. Hence, the result follows using the proof of Theorem [10;](#page-28-0) more precisely using equation [64](#page-56-1) with <sup>W</sup>′ → <sup>W</sup><sup>ˆ</sup> <sup>t</sup><sup>1</sup> , ∆t,i → ∆ˆ t,i, pt,i → pˆt,i, and σ<sup>t</sup> → σˆt.

Now, suppose that the result holds for m = N ≤ M − 1, *i.e.,*

$$\text{CM}_{\mathbf{V}}^{\Theta}, \mathbf{J}_{-i}, i(\tilde{\mathbf{S}}, \hat{W}_{t_N}) \leq \sum_{r \in [N]} A_{t_r, i}^{t_N} \mathbb{E}_{\hat{p}_{t_r, i}, \hat{\Delta}_{t_k, i}} \left[ f \left( \frac{\eta_{t_r}}{b\hat{\sigma}_{t_r}} \hat{\Delta}_{t_r, i}, \hat{p}_{t_r, i} \right) \right], \quad (75)$$

where

$$A_{t_r,i}^{t_N} := \prod_{t \in [t_r+1:t_N] : i \notin V_t} q_t.$$

We show that it also holds for m = N + 1 ≤ M.

We have I

$$\begin{aligned}
|\tilde{\mathbf{S}}, \mathbf{J}_{-i}, \Theta, \mathbf{V}(\hat{W}_{t_{N+1}}; J_i) &\leq |\tilde{\mathbf{S}}, \mathbf{J}_{-i}, \Theta, \mathbf{V}(\hat{W}_{t_{N+1}}, \hat{W}_{t_{N+1}-1}; J_i)| \\
&= |\tilde{\mathbf{S}}, \mathbf{J}_{-i}, \Theta, \mathbf{V}(\hat{W}_{t_{N+1}}; J_i | \hat{W}_{t_{N+1}-1}) + |\tilde{\mathbf{S}}, \mathbf{J}_{-i}, \Theta, \mathbf{V}(\hat{W}_{t_{N+1}-1}; J_i)| \\
&\stackrel{(a)}{\leq} \mathbb{E}_{\hat{p}_{t_{N+1}, i}} \left[ f \left( \frac{\eta_{t_{N+1}}}{b \hat{\sigma}_{t_{N+1}}} \hat{\Delta}_{t_{N+1}, i}, \hat{p}_{t_{N+1}, i} \right) \right] + |\tilde{\mathbf{S}}, \mathbf{J}_{-i}, \Theta, \mathbf{V}(\hat{W}_{t_{N+1}-1}; J_i)| \\
&\stackrel{(b)}{\leq} \mathbb{E}_{\hat{p}_{t_{N+1}, i}} \left[ f \left( \frac{\eta_{t_{N+1}}}{b \hat{\sigma}_{t_{N+1}}} \hat{\Delta}_{t_{N+1}, i}, \hat{p}_{t_{N+1}, i} \right) \right] \\
&\quad + \left( \prod_{r=t_N+1}^{t_{N+1}-1} q_r \right) |\tilde{\mathbf{S}}, \mathbf{J}_{-i}, \Theta, \mathbf{V}(\hat{W}_{t_N}; J_i)| \\
&\stackrel{(c)}{\leq} \mathbb{E}_{\hat{p}_{t_{N+1}, i}} \left[ f \left( \frac{\eta_{t_{N+1}}}{b \hat{\sigma}_{t_{N+1}}} \hat{\Delta}_{t_{N+1}, i}, \hat{p}_{t_{N+1}, i} \right) \right] \\
&\quad + \left( \prod_{r=t_N+1}^{t_{N+1}-1} q_r \right) \sum_{r \in [N]} A_{t_r, i}^{t_N} \mathbb{E}_{\hat{p}_{t_r, i}} \left[ f \left( \frac{\eta_{t_r}}{b \hat{\sigma}_{t_r}} \hat{\Delta}_{t_r, i}, \hat{p}_{t_r, i} \right) \right] \\
&\stackrel{(d)}{\cong} \sum_{r \in [N+1]} A_{t_r, i}^{t_{N+1}} \mathbb{E}_{\hat{p}_{t_r, i}} \left[ f \left( \frac{\eta_{t_r}}{b \hat{\sigma}_{t_r}} \hat{\Delta}_{t_r, i}, \hat{p}_{t_r, i} \right) \right],
\end{aligned}$$

where

- (a) is derived using the proof of Theorem [10;](#page-28-0) more precisely using equation [65](#page-56-2) with W′ <sup>t</sup><sup>m</sup> → <sup>W</sup><sup>ˆ</sup> <sup>t</sup>N+1 , ∆<sup>t</sup>m,i → ∆ˆ <sup>t</sup>N+1,i, <sup>p</sup><sup>t</sup>m,i → <sup>p</sup>ˆ<sup>t</sup>N+1,i, <sup>σ</sup><sup>t</sup><sup>m</sup> → <sup>σ</sup>ˆ<sup>t</sup>N+1 , and by considering

$$\mathcal{F}_m \rightarrow \left\{ \tilde{\mathbf{S}}, \Theta, \mathbf{V}, \mathbf{J}_{-i}, W'_{t_{N+1}-1} \right\}.$$

- (b) is derived by repeated using of [\[74,](#page-13-13) Lemma 4],
- (c) holds by the assumption of the induction [75,](#page-60-0)
- and (d) by definition of A t<sup>N</sup> t,i and A tN+1 t,i .

This completes the proof of the theorem.

# H.4 Proof of Lemma [6](#page-58-1)

To prove the result, we show first what

$$\left\| \hat{W}_t - W'_t \right\| \leq \sum_{r \in [t]} \alpha^{t-r} \nu_r \left\| \varepsilon'_r \right\|, \quad (76)$$

using induction over t ∈ [T]. Then, using the Lipschitzness property of the loss function, we have that

$$\begin{aligned} \mathbb{E}_{P_S P_\Theta P_{\mathbf{V}} P_{W'_T|S,\Theta,\mathbf{V}} P_{\hat{W}_T|W'_T,S,\Theta,\mathbf{V}}} \left[ \text{gen}(S, \Theta W'_T) - \text{gen}(S, \Theta \hat{W}_T) \right] &\leq 2 \mathbb{E} \left[ \sum_{r \in [T]} \alpha^{T-r} \nu_r \| \varepsilon'_r \| \right] \\ &\stackrel{(a)}{=} \frac{2\sqrt{2} \mathfrak{L} \Gamma((d+1)/2)}{\Gamma(d/2)} \sum_{r \in [T]} \alpha^{T-r} \nu_r, \end{aligned}$$

where (a) is obtained using the fact that if Z ∼ N (0,Id), then ∥Z∥ has a chi-distribution, whose mean is equal to √ 2 Γ((d+1)/2) Γ(d/2) .

For t = 1,

$$\begin{aligned} \left\| \hat{W}_1 - W'_1 \right\| &\stackrel{(a)}{\leq} \left\| \left( W'_0 - \eta_1 \nabla_{w'} \hat{\mathcal{R}}(V_1, \Theta W'_0) + \sigma_1 \varepsilon_1 + \nu_1 \varepsilon'_1 \right) - \left( W'_0 - \eta_1 \nabla_{w'} \hat{\mathcal{R}}(V_1, \Theta W'_0) + \sigma_1 \varepsilon_1 \right) \right\| \\ &= \nu_1 \left\| \varepsilon'_1 \right\|, \end{aligned}$$

where (a) is derived since for any w ′ 1, w′ <sup>2</sup> ∈ <sup>R</sup> d , Proj w ′ 1 − Proj w ′ 2 ≤ w ′ <sup>1</sup> − w ′ , by the contraction property of the projection. This shows the base of the induction.

Suppose that equation [76](#page-61-1) holds for t = t ′ . Now, we show that it also holds for t = t ′ + 1.

$$\begin{aligned} \left\| \hat{W}_{t'+1} - W'_{t'+1} \right\| &\leq \left\| \left( \hat{W}_{t'} - \eta_{t'+1} \nabla_{w'} \hat{\mathcal{R}}(V_{t'+1}, \Theta \hat{W}_{t'}) + \sigma_{t'+1} \varepsilon_{t'+1} + \nu_{t'+1} \varepsilon'_{t'+1} \right) \right. \\ &\quad \left. - \left( W'_{t'} - \eta_{t'+1} \nabla_{w'} \hat{\mathcal{R}}(V_{t'+1}, \Theta W'_{t'}) + \sigma_{t'+1} \varepsilon_{t'+1} \right) \right\| \\ &= \left\| \left( \hat{W}_{t'} - \eta_{t'+1} \nabla_{w'} \hat{\mathcal{R}}(V_{t'+1}, \Theta \hat{W}_{t'}) \right) - \left( W'_{t'} - \eta_{t'+1} \nabla_{w'} \hat{\mathcal{R}}(V_{t'+1}, \Theta W'_{t'}) \right) \right. \\ &\quad \left. + \nu_{t'+1} \varepsilon'_{t'+1} \right\| \\ &\stackrel{(a)}{\leq} \left\| \left( \hat{W}_{t'} - \eta_{t'+1} \nabla_{w'} \hat{\mathcal{R}}(V_{t'+1}, \Theta \hat{W}_{t'}) \right) - \left( W'_{t'} - \eta_{t'+1} \nabla_{w'} \hat{\mathcal{R}}(V_{t'+1}, \Theta W'_{t'}) \right) \right\| \\ &\quad + \left\| \nu_{t'+1} \varepsilon'_{t'+1} \right\| \\ &\stackrel{(b)}{\leq} \alpha \left\| \hat{W}_{t'} - W'_{t'} \right\| + \nu_{t'+1} \left\| \varepsilon'_{t'+1} \right\| \\ &\stackrel{(c)}{\leq} \alpha \sum_{r \in [t']} \alpha^{t'-r} \nu_r \left\| \varepsilon'_r \right\| + \nu_{t'+1} \left\| \varepsilon'_{t'+1} \right\| \\ &= \sum_{r \in [t'+1]} \alpha^{(t'+1)-r} \nu_r \left\| \varepsilon'_r \right\|, \end{aligned}$$

where (a) is derived using the triangle inequality, (b) using the contractility assumption, and (c) using the assumption of the induction. This completes the proof of the induction and the proof of the lemma.