**006**

**009**

**019**

**024**

**029 030**

**032**

**034**

**036**

# THE RATE-DISTORTION-PERCEPTION TRADE-OFF WITH ALGORITHMIC REALISM

Anonymous authors Paper under double-blind review

# ABSTRACT

Realism constraints (or constraints on perceptual quality) have received considerable recent attention within the context of lossy compression, particularly of images. Theoretical studies of lossy compression indicate that high-rate common randomness between the compressor and the decompressor is a valuable resource for achieving realism. On the other hand, the utility of significant amounts of common randomness at test time has not been noted in practice. We offer an explanation for this discrepancy by considering a realism constraint that requires satisfying a universal critic that inspects realizations of individual compressed images, or batches thereof. We characterize the optimal rate-distortion-perception trade-off under such a realism constraint, and show that it is asymptotically achievable without any common randomness, unless the batch size is impractically large.

# 1 INTRODUCTION

Realism, or perceptual quality, of reconstructed signals is a long-standing open challenge in lossy compression, particularly for image/video compression [\(Eckert & Bradley, 1998;](#page-9-0) [Wu et al., 2012\)](#page-10-0). It has received renewed interest in the recent years due to the remarkable progress in image generation models and neural compression techniques. The idea is that reconstructed images should be indistinguishable to humans from naturally occurring ones in addition to having a high pixel-level fidelity to the original source. This ensures that reconstructed images are free of obvious artifacts such as blocking, blurriness, etc.

The idea that the output of the decoder should resemble the source in a statistical sense is not new. Advanced Audio Coding (AAC), for instance, includes a provision to add high-frequency noise to the output so that its power spectrum resembles that of the source [\(Sayood, 2012\)](#page-10-1). But the idea has received renewed attention with the emergence of adversarial loss functions in learned compression [\(Santurkar et al., 2018;](#page-10-2) [Tschannen et al., 2018;](#page-10-3) [Agustsson et al., 2019;](#page-9-1) [Blau & Michaeli,](#page-9-2) [2019\)](#page-9-2). In practice, this has proven to be a powerful method for ensuring that reconstructed images have high perceptual quality [\(Agustsson et al., 2019;](#page-9-1) [Mentzer et al., 2020;](#page-10-4) [He et al., 2022a;](#page-9-3) [Iwai et al.,](#page-9-4) [2024\)](#page-9-4). Adversarial loss functions can in many cases be viewed as variational forms of statistical divergences. Thus one can think of constraining the distribution of reconstructions to be close to that of the source according to some divergence, in addition to requiring that each reconstructed image be close to its respective source according to conventional notions of distortion.

Rate-distortion theory characterizes the optimal trade-off between rate and distortion in lossy compression [\(Pearlman & Said, 2011;](#page-10-5) [Sayood, 2012\)](#page-10-1). The fundamental object in the theory is the *rate-distortion function*, for a given source distribution p<sup>X</sup> :

$$\Delta \in [0, \infty) \mapsto R^{(0)}(\Delta) := \min_{\substack{p_Y|_X \text{ s.t.} \\ \mathbb{E}_p[d(X, Y)] \leq \Delta}} I_p(X; Y), \quad (1)$$

$$\Delta \in [0, \infty) \mapsto R^{(0)}(\Delta) := \min_{\substack{p_{Y|X} \text{ s.t.} \\ \mathbb{E}_p[d(X, Y)] \leq \Delta}} I_p(X; Y), \quad (1)$$

where pX,Y is defined as p<sup>X</sup> · p<sup>Y</sup> <sup>|</sup>X. This function has been shown to describe the optimal trade-off between rate and distortion under a variety of assumptions. [Blau & Michaeli](#page-9-2) [\(2019\)](#page-9-2) postulated an augmented form that includes a *distribution matching* constraint, which they call the *rate-distortionperception* (RDP) function

$$(\Delta, \lambda) \in [0, \infty)^2 \mapsto R^{(1)}(\Delta, \lambda) := \min_{\substack{p_{Y|X} \text{ s.t.} \\ \mathcal{D}(p_X, p_Y) \leq \lambda, \\ \mathbb{E}_p[d(X, Y)] \leq \Delta}} I_p(X; Y), \quad (2)$$

$$(\Delta, \lambda) \in [0, \infty)^2 \mapsto R^{(1)}(\Delta, \lambda) := \min_{\substack{p_Y \mid \text{s.t.} \\ \mathcal{D}(p_X, p_Y) \leq \lambda, \\ \mathbb{E}_p[d(X, Y)] \leq \Delta}} I_p(X; Y), \quad (2)$$

**059**

**061**

**064**

**067**

**069 070 071**

**074**

**079**

**089 090 091**

**094**

**104 105 106** where D can be any divergence between distributions. This function has likewise been shown to describe the optimal trade-off between rate, distortion, and realism under a variety of assumptions [\(Theis & Wagner;](#page-10-6) [Chen et al., 2022\)](#page-9-5). Curiously, however, these results show that substantial amounts of high-quality common randomness are needed to meet the R(1)(·, 0) bound [\(Saldi et al.,](#page-10-7) [2015;](#page-10-7) [Wagner, 2022;](#page-10-8) [Chen et al., 2022\)](#page-9-5) (see also [Xu et al.](#page-10-9) [\(2023\)](#page-10-9)). The exception is the case in which the realism constraint is imposed in a very weak form, namely that the histograms of the source and reconstruction images should be close on a per-realization basis [\(Chen et al., 2022\)](#page-9-5). Note that common pseudorandomness, say generated from a shared seed, does not qualify as common randomness for the purposes of the above results.

On the other hand, the theoretical prediction that lossy compression schemes would benefit from substantial amounts of high-quality common randomness between the encoder and decoder has not been observed in practice. To the best of our knowledge, there exist compression schemes [\(Agustsson](#page-9-6) [et al., 2023;](#page-9-6) [He et al., 2022a;](#page-9-3) [Hoogeboom et al., 2023;](#page-9-7) [Ghouse et al., 2023;](#page-9-8) [Mentzer et al., 2020;](#page-10-4) [Yang & Mandt, 2023\)](#page-10-10), considered as state-of-the-art, that do not involve any common randomness. While it is possible that future designs will find common randomness to be a valuable resource, it seems more likely that the discrepancy between the theoretical prediction and practical experience lies with a flaw with the theoretical models.

Consider a communication system for which a strong realism constraint is imposed: the distribution of the reconstructions must be close to the distribution of natural images, say, in Wasserstein or total variation distance (TVD). If the source distribution is continuous, then the code cannot be deterministic, for otherwise the reconstruction distribution would be supported on a countable set (corresponding to the set of received bit strings). Thus some amount of randomization is required to meet the constraint. The decoder can randomize its output in a way that "spreads" the point masses out to form a continuous distribution, but adding independent noise at the decoder inevitably degrades the distortion. Common randomness is useful because it allows the discrete reconstruction points to be dispersed to form a continuous distribution without less overall distortion. This is the basis for the finding that common randomness is a useful resource for compression under realism constraints [\(Theis & Agustsson, 2021\)](#page-10-11).[<sup>1</sup>](#page-1-0)

The above reasoning is evidently sensitive to the nature of the realism constraint. If we simply require that each reconstructed image appear realistic in its own right, without reference to the reconstruction ensemble, then the spreading process mentioned above is unnecessary. It follows that there would be no need for randomization. This is relevant because human observers, who are the ultimate arbiters of realism in practice, are adept at identifying unrealistic features of individual images. Yet it is difficult for human observers to distinguish between a continuous ensemble and one that is discrete with a very large support set, since doing so would require viewing (and remembering) many images. In short, human critics are very good at spotting unrealistic aspects of individual images but are expected to be poor at detecting subtle ensemble-level differences.

This suggests posing the realism constraint in a way that better captures the relative strengths and weaknesses of human critics. The aforementioned strong realism constraint has also been challenged in the context of other problems, such as generative modeling [\(Theis, 2024\)](#page-10-12). We consider a novel formulation of the lossy compression problem in which the goal is to satisfy a critic that is incredibly discriminating when viewing individual images. In fact, a reconstructed image is declared unrealistic if there exists some computable test, no matter how complex, that can distinguish it from the set of typical source images (see Definition [3.5](#page-4-0) to follow). At the same time, we assume that the critic can glean information about the ensemble only by inspecting batches of individual samples. Under this formulation, we show that the rate-distortion-perception function R(1)(·, 0) in [\(2\)](#page-0-0) is achievable without common randomness unless the batch size is unreasonably high—on par with the number of possible outputs of the decoder (Theorems [4.1](#page-5-0) and [4.2\)](#page-6-0). If common randomness is not needed to fool this critic, it should not be needed to fool any weaker (and more practical) critic, since the stronger critic subsumes the weaker one. This is akin to how in cryptography one might prove security guarantees assuming a very strong adversary, stronger than can be implemented in practice. The fact that the adversary cannot be practically implemented is a strength of our approach. It is notable that there exist compressors that can satisfy such discriminating critics at all. It is all the more notable that

It is now apparent why sharing a pseudorandom seed is insufficient, as this would expand the number of distinct reproductions by a multiplicative factor equal to the number of possible values of the seed, which is relatively small if the seed is short.

**114 115**

**117**

**119**

**127**

**129 130**

**134**

**136**

such critics can be satisfied while achieving the rate-distortion-perception function R(1)(·, 0) in [\(2\)](#page-0-0), which is the most optimistic rate-distortion trade-off possible under the circumstances. Conversely, we show that common randomness is indeed beneficial if the batch size is extremely large, larger than would ever occur in practice (Theorem [4.4\)](#page-7-0). In this regime, our realism measure reduces to a divergence and common randomness is again useful. These two results clarify that common randomness is indeed useful, consistent with theoretical predictions, but only in regimes that do not occur in practice, consistent with the current state of the experimental literature. Our results show the existence of optimal schemes which do not involve any common randomness at test time, but there may exist other optimal schemes, which rely on common randomness at test time, as well as learned schemes relying on common randomness at training time.

In Section [2,](#page-2-0) we provide some background on the formalism for critics in algorithmic information theory. In Section [3,](#page-3-0) we introduce our new formalism for the RDP trade-off. In Section [4,](#page-5-1) we state our main results, namely Theorems [4.1,](#page-5-0) [4.2,](#page-6-0) and [4.4.](#page-7-0) All proofs are deferred to the appendices.

### 2 BACKGROUND

#### 2.1 NOTATION

Calligraphic letters such as X denote sets, except in p U J , which denotes the uniform distribution over set J . The cardinality of a finite set X is denoted |X |. We denote by [a] the set {1, ..., ⌊a⌋} and by {0, 1} ∗ the set of non-empty finite strings of 0's and 1's. Given a real number τ, we denote by ⌊τ ⌋ (resp. ⌈τ ⌉) the largest (resp. smallest) integer less (resp. greater) than or equal to τ. We use x1:<sup>n</sup> to denote a finite sequence (x1, ..., xn), and x (n,b) to denote a batch {x (k) 1:n }k∈[b] of b strings, each being of length n. We abbreviate x (1,b) with x (b) . The length of a string x is denoted by l(x).

We denote the set of (strictly) positive reals by <sup>R</sup>+, the set of (strictly) positive integers by N, the set of rational numbers by Q, and the Borel σ-algebra of <sup>R</sup> by B(R). The closure of a set A is denoted by cl(A). We use ≡ to denote equality of distributions, and Ip(X; Y ) to denote the mutual information between random variables X and Y with respect to joint distribution pX,Y . Logarithms are in base 2. The total variation distance between distributions p and q on a finite set X is defined by

$$\|p - q\|_{TV} := \frac{1}{2} \sum_{x \in \mathcal{X}} |p(x) - q(x)|.$$

For any nonempty finite set X , and any distribution p on X , we denote by p ⊗∗ the function defined on {0, 1} ∗ , which is null outside of ∪n∈NX <sup>n</sup>, and such that for every n ∈ <sup>N</sup>, the restriction of p ⊗∗ on X n is p <sup>⊗</sup><sup>n</sup>. For a finite set X , the empirical distribution of a sequence x1:n∈X <sup>n</sup> is denoted <sup>P</sup> emp X (x1:n). Given a distribution P<sup>X</sup>1:<sup>n</sup> on X <sup>n</sup>, we denote by Pˆ<sup>X</sup> [X1:n] the *average marginal distribution* of random string X1:n, i.e., the distribution on X defined by:

$$\hat{P}_{\mathcal{X}}[X_{1:n}] := \frac{1}{n} \sum_{t=1}^n P_{X_t}.$$

#### 2.2 LOSSY COMPRESSION ALGORITHMS WITHOUT COMMON RANDOMNESS

The performance of practical lossy compression schemes in terms of realism (or perceptual quality) is generally measured with well established metrics such as FID [\(Heusel et al., 2017\)](#page-9-9), LPIPS [\(Zhang](#page-10-13) [et al., 2018\)](#page-10-13), PieAPP [\(Prashnani et al., 2018\)](#page-10-14), and DISTS [\(Ding et al., 2022\)](#page-9-10). Distortion is often measured with PSNR. According to these metrics, the following lossy compression algorithms are state-of-the-art. In particular, these schemes achieve visually pleasing reconstructions at very low compression rates. None of these algorithms make use of common randomness. The schemes in [Mentzer et al.](#page-10-4) [\(2020\)](#page-10-4), [He et al.](#page-9-3) [\(2022a\)](#page-9-3), and [Agustsson et al.](#page-9-6) [\(2023\)](#page-9-6) were obtained by training with an adversarial loss, a method inspired from generative adversarial networks (GANs). The former combines a conditional GAN with the scale hyperprior method of [Ballé et al.](#page-9-11) [\(2018\)](#page-9-11). The latter is an extension of the ELIC scheme [\(He et al., 2022b\)](#page-9-12), which is state-of-the-art in terms of rate and distortion. The loss function of the latter was augmented, in particular, with an adversarial term and an LPIPS term. The method proposed in [Agustsson et al.](#page-9-6) [\(2023\)](#page-9-6) is inspired from [He et al.](#page-9-12) [\(2022b\)](#page-9-12) and [Mentzer et al.](#page-10-4) [\(2020\)](#page-10-4). The schemes in [Yang & Mandt](#page-10-10) [\(2023\)](#page-10-10), [Ghouse et al.](#page-9-8) [\(2023\)](#page-9-8), and

**166 167**

**169**

**171**

**179 180 181**

**204**

[Hoogeboom et al.](#page-9-7) [\(2023\)](#page-9-7) rely on diffusion models. The first uses a diffusion model conditioned on quantized latents. The two other schemes first train an autoencoder for rate and distortion, then train a diffusion model which improves the visual quality of the latter's output. The fact that none of these state-of-the-art algorithms make use of common randomness supports the theoretical results derived in the present paper.

#### 2.3 BACKGROUND ON ALGORITHMIC INFORMATION THEORY

The theory of p-critics and universal critics has recently been brought to the attention of the machine vision community via [Theis](#page-10-12) [\(2024\)](#page-10-12). We refer to it for readers interested in a high-level and insightful presentation of the topic and its usefulness in diverse machine learning tasks (generative modeling, outlier detection). Relevant background on computability theory is provided in Appendix [A.](#page-11-0) Throughout the paper, we assume that the source X follows a distribution p<sup>X</sup> on a finite set X , and that p<sup>X</sup> is a computable function from X to (0, 1). We identify every element of X with a string of 0's and 1's, via an injection from X to {0, 1} s , for some s ∈ N. For example, if X is a set of images of a given resolution, then one can identify each image with the corresponding output from a fixed-length lossless compressor. The following definition is substantially close to [Li & Vitányi](#page-9-13) [\(2019,](#page-9-13) Definition 4.3.8). See also in [Li & Vitányi](#page-9-13) [\(2019,](#page-9-13) Lemma 4.3.5).

Definition 2.1. *Consider a finite set* X , *identified with a subset of* {0, 1} s . *Let* p *be a distribution on* X *such that* ∀x ∈ X , p(x) > 0. *A* p*-critic is a function* δ : X → <sup>R</sup>, *such that*

$$\sum_{x \in \mathcal{X}} p(x) 2^{\delta(x)} \leq 1. \quad (3)$$

*A* p ⊗∗ *-critic is a function* δ : ∪n∈NX <sup>n</sup> → <sup>R</sup>, *such that for every input dimension* n ∈ <sup>N</sup>, *we have*

$$\sum_{x \in \mathcal{X}^n} p^{\otimes n}(x) 2^{\delta(x)} \leq 1. \quad (4)$$

The notion of p ⊗∗-critic in Definition [2.1](#page-3-1) is used to study an asymptotic regime in Section [3.2.](#page-4-1) Note that for any probability distribution π on <sup>N</sup>, the mixture p˜ := P n∈N π(n)p <sup>⊗</sup><sup>n</sup> is a probability measure. By multiplying [\(4\)](#page-3-2) by πn, and summing over n, we obtain

$$\sum_{x \in \cup_{n \in \mathbb{N}} \mathcal{X}^n} \tilde{p}(x) 2^{\delta(x)} \leq 1. \quad (5)$$

Hence, a p-critic (resp. p ⊗∗-critic) is akin to a log-likelihood ratio: given a p-critic (resp. p ⊗∗-critic) δ, setting q : x 7→ p(x)2<sup>δ</sup>(x) (resp. q : x 7→ p˜(x)2<sup>δ</sup>(x) ) gives

$$\forall x \in \mathcal{X} \text{ s.t. } p(x) > 0, \delta(x) = \log \left( \frac{q(x)}{p(x)} \right) \text{ (resp. } \log \left( \frac{q(x)}{\tilde{p}(x)} \right) \text{)}, \quad \text{and} \quad \sum_{x \in \mathcal{X}} q(x) \leq 1. \quad (6)$$

Links to hypothesis testing are discussed in [Theis](#page-10-12) [\(2024\)](#page-10-12), where a sample x is deemed unrealistic if the likelihood ratio is large enough. Hence, intuitively, δ(x) can be considered as a measure of *realism deficiency* of x. The strength of this theory lies in the existence of objects (critics, measures) having a so-called *universality property*. For the purpose of clarity, we defer such results to Appendix [A,](#page-11-0) as they are only used in our proofs.

### 3 NEW MODEL FOR THE RATE-DISTORTION-PERCEPTION TRADE-OFF

#### 3.1 THE ONE-SHOT SETTING

We consider a function d : X ×X → [0, ∞) called the distortion measure. A compression scheme can be randomized, and potentially leverage available common randomness J between the encoder and the decoder, as depicted in Figure [1](#page-4-2) and formalized in the following definition.

Definition 3.1. *Given non-negative reals* R *and* Rc, *an* (R, Rc) *code is a privately randomized encoder and decoder couple* (F, G) *consisting of a conditional distribution* FM|X,J *from* X × [2<sup>R</sup><sup>c</sup> *to* [2<sup>R</sup>], *and a conditional distribution* G<sup>Y</sup> <sup>|</sup>M,J *from* [2<sup>R</sup>] × [2<sup>R</sup><sup>c</sup> ] *to* X . *Variables* M *and* Y *are called the message and reconstruction, respectively, and distribution*

$$P := p_X \cdot p_{[2^{R_C}]}^{\mathcal{U}} \cdot F_{M|X,J} \cdot G_{Y|M,J}$$

**224**

**233 234**

**236 237**

**254**

**256**

**259**

**269**

![](_page_4_Diagram_1.jpeg)

Figure 1: The system model for the one-shot setting.

*is called the distribution induced by the code. Moreover, such a code is said to be deterministic if* R<sup>c</sup> = 0 *and mappings* F, G *are deterministic.*

We propose a new RDP trade-off, formalized in the following two definitions.

Definition 3.2. *We extend* d *into an additive distortion measure on batches of elements of* X *: for all* B∈N,

$$\forall (\mathbf{x}^{(B)}, \mathbf{y}^{(B)}) \in \mathcal{X}^B \times \mathcal{X}^B, \quad d(\mathbf{x}^{(B)}, \mathbf{y}^{(B)}) := \frac{1}{B} \sum_{k=1}^B d(x^{(k)}, y^{(k)}).$$

Definition 3.3. *Consider a positive integer* B, *and a* p ⊗B <sup>X</sup> *-critic* δ. *A tuple* (R, ∆, C) *is said to be* δ*-achievable with algorithmic realism if there exists some* R<sup>c</sup> ∈ <sup>R</sup>≥<sup>0</sup> *and an* (R, Rc) *code such that the distribution* P *induced by the code satisfies*

$$\mathbb{E}_{P \otimes B} [d(\mathbf{X}^{(B)}, \mathbf{Y}^{(B)})] \leq \Delta \text{ and} \quad (7)$$

$$\mathbb{E}_{P \otimes B}[\delta(\mathbf{Y}^{(B)})] \leq C, \quad (8)$$

*where* X(B) *denotes a batch of* B *i.i.d. source samples, and* Y(B) *the batch of corresponding reconstructions produced by the code (with each source sample being compressed separately). If the code is deterministic, then we say that* (R, ∆, C) *is* δ*-achievable with a deterministic code.*

The main difference with the original RDP trade-off of [Blau & Michaeli](#page-9-2) [\(2019\)](#page-9-2) pertains to the realism constraint. In the latter formulation, the realism constraint is D(pX, P<sup>Y</sup> ) ≤ C, where D is some divergence. Intuitively, that constraint corresponds to the special case of infinite batch size in the RDP trade-off proposed in Definition [3.3,](#page-4-3) since the discrete distributions p<sup>X</sup> and P<sup>Y</sup> can be approximated arbitrarily well using a large enough number of samples. In that sense, our proposed RDP framework generalizes the original one, through involving elements of practical realism metrics, such as the number B of samples which are inspected, and a scoring function δ which is required to be approximable via an algorithm. Theorem [4.4](#page-7-0) to follow constitutes a rigorous statement of this intuition. We provide achievable points in the sense of Definition [3.3](#page-4-3) in Section [4.2.](#page-6-1) In the next section, we define an asymptotic notion of achievability.

#### 3.2 ASYMPTOTIC SETTING

In order to derive insight into the corresponding RDP trade-off, we study a special case, which is typical in the information theory literature. We consider the compression of a source distributed according to p ⊗n <sup>X</sup> , with n a large integer. More precisely, we study the RDP trade-off in asymptotic settings where both n and the batch size go to infinity.

The extension of d into an *additive distortion measure* on finite sequences, and batches of finite sequences, follows from Definition [3.2.](#page-3-3) The setup is depicted in Figure [2.](#page-5-2) Given a coding scheme, each item in a batch of source samples is compressed separately, and realism is measured based on the resulting batch of reconstructions. This is formalized in the definition below.

Definition 3.4. *Given* R, R<sup>c</sup> ≥ 0, *and* n ∈ <sup>N</sup>, *a* (n, R, Rc) *code is a privately randomized encoder and decoder couple* (F (n) , G(n) ) *consisting of a mapping* F (n) <sup>M</sup>|X1:n,J *from* X <sup>n</sup> × [2nR<sup>c</sup> ] *to* [2nR] *and a mapping* G (n) <sup>Y</sup>1:n|M,J *from* [2nR<sup>c</sup> ] × [2nR] *to* X <sup>n</sup>. *Moreover, such a code is said to be fully deterministic if* R<sup>c</sup> = 0 *and both* F (n) *and* G(n) *are deterministic. The distribution induced by the code is*

$$P^{(n)} := p_X^{\otimes n} \cdot p_{[2^n R_c]}^{\mathcal{U}} \cdot F_{M|X_{1:n}, J}^{(n)} \cdot G_{Y_{1:n}|M, J}^{(n)},$$

**289 290 291**

**294**

**301**

**304**

**306**

**309**

**314 315 316**

**318 319**

![](_page_5_Diagram_1.jpeg)

Figure 2: The system model for the asymptotic setting. Index k ranges from 1 to the batch size. The same encoder-decoder pair is used to process each source sample in the batch.

*and variable* Y1:<sup>n</sup> *is called the reconstruction.*

We define asymptotic achievability as follows. See Appendix [A](#page-11-0) for background on notions of computability.

#### Definition 3.5.

*A quadruplet* (R, Rc, {Bn}n≥1, ∆) *is said to be asymptotically achievable with algorithmic realism if for any* ε > 0, *there exists a sequence of codes* {(F (n) , G(n) )}n, *the* n*-th being* (n, R + ε, Rc), *such that the sequence* {P (n)}<sup>n</sup> *of distributions induced by the codes satisfies*

$$\limsup_{n \rightarrow \infty} \mathbb{E}_{(P(n)) \otimes B_n} [d(\mathbf{X}^{(n, B_n)}, \mathbf{Y}^{(n, B_n)})] \leq \Delta + \varepsilon, \quad (9)$$

*and for any lower semi-computable* p ⊗∗ <sup>X</sup> *-critic* δ,

$$\sup_{n \in \mathbb{N}} \mathbb{E}_{(P(n)) \otimes B_n} [\delta(\mathbf{Y}^{(n, B_n)})] < \infty. \quad (10)$$

*We say that* (R, {Bn}n≥1, ∆) *is achievable with a fully deterministic scheme if for each* n, *the code* (F (n) , G(n) ) *is fully deterministic.*

Constraint [\(10\)](#page-5-3) is very stringent: a single compression scheme is to satisfy a performance guarantee for every lower semi-computable p ⊗∗ <sup>X</sup> -critic (i.e. every relevant one). The motivation for the specific form of [\(10\)](#page-5-3) is firstly from the algorithmic information theory literature: it is stated in [Li & Vitányi](#page-9-13) [\(2019,](#page-9-13) p.140) that a sample from a large set, identified to a long string of 0's and 1's of some length k, is realistic if its realism deficiency is small compared to k. The constraint in [\(10\)](#page-5-3) is at least as stringent, since in our asymptotic setting, each x1:<sup>n</sup> ∈ X <sup>n</sup> is identified with a string of length linear in n, while we require the realism deficiency to be bounded. Moreover, consider the following simple example. Assume X = {0, 1}, and p<sup>X</sup> is a Bernoulli distribution B(ρ). Consider the 0-1 distortion (also called Hamming distortion), and some distortion level ∆ < min(ρ, 1 − ρ). Then, for large enough n, the classical rate-distortion optimal code appearing in the information theory literature produces reconstructions having a frequency of 1's of roughly (ρ − ∆)/(1 − 2∆) [\(Cover & Thomas,](#page-9-14) [2006,](#page-9-14) Sections 10.3.1 and 10.5), i.e. different from ρ (if ρ ̸= 1/2 and ∆ > 0). Then, for the p ⊗∗ <sup>X</sup> -critic appearing in Appendix [G](#page-20-0) (Claim [G.1\)](#page-20-1), which involves the frequency of occurrence of a pattern, the expected score diverges as n goes to infinity. Hence, the constraint in [\(10\)](#page-5-3) is not satisfied by such a code, optimized only for rate and distortion, but not for realism. This concludes the definitions for our setup. In the next sections, we present our results, in the one-shot setting and in asymptotic settings.

# 4 RESULTS

#### 4.1 LOW BATCH SIZE REGIME

The following theorem states that R(1)(·, 0), defined in [\(2\)](#page-0-0), which naturaly arises in the distribution matching formalism, also characterizes the optimal trade-off in our asymptotic setting, when the batch size is not impractically large.

Theorem 4.1. *Consider a sequence* {Bn}n≥<sup>1</sup> *of positive integers such that*

$$\log(B_n)/n \xrightarrow{n \rightarrow \infty} 0. \quad (11)$$

*For any* ∆ ∈ <sup>R</sup>+, *let* R(∆) *be the infimum of rates* R *such that there exists* R<sup>c</sup> ∈ <sup>R</sup>≥<sup>0</sup> *such that* (R, Rc, {Bn}n≥1, ∆) *is asymptotically achievable with algorithmic realism. Moreover, for*

**329**

**334**

**354 355 356**

**358 359**

**361**

**364**

**369**

*any* ∆ ∈ <sup>R</sup>+, *let* R∗(∆) *be the infimum of rates* R *such that* (R, {Bn}n≥1, ∆) *is asymptotically achievable with algorithmic realism with fully deterministic codes. Then, we have*

$$\forall \Delta \in \mathbb{R}_+ \text{ s.t. } R^{(1)}(\Delta, 0) < H_p(X), \text{ we have } R(\Delta) = R_*(\Delta) = R^{(1)}(\Delta, 0). \quad (12)$$

The proof is provided in Appendices [C](#page-15-0) and [D.](#page-17-0) The strength of this result lies in how stringent constraint [\(10\)](#page-5-3) is: a single compression scheme satisfies a performance guarantee for every relevant p ⊗∗ <sup>X</sup> -critic, and deterministic schemes are sufficient. Moreover, one can find such a scheme for any batch size sequence which is sub-exponential in the dimension n of the source, i.e. for all regimes where the batch size is not impractically large. To prove the achievability direction of Theorem [4.1,](#page-5-0) we leverage the existence of a *universal* p ⊗∗ <sup>X</sup> -critic δ<sup>0</sup> (see Appendix [A.2\)](#page-11-1), which is one of the great successes of algorithmic information theory. Indeed, it is sufficient to construct a scheme which achieves [\(10\)](#page-5-3) only for such a δ0, which is more sensitive than all relevant p ⊗∗ <sup>X</sup> -critics. It is a very strong critic, stronger than can be implemented in practice, which is another strength of Theorem [4.1.](#page-5-0)

#### 4.2 ONE-SHOT ACHIEVABLE POINTS

For theoretical interest, we provide a family of points which are achievable, in the sense of Definition [3.3,](#page-4-3) without any statistical assumption on the source distribution pX. For the sake of gleaning intuition, one can consider the following example.

- X is a finite set of images, e.g. the set of all images of a given resolution, with a finite range for pixels (finite precision).
- d is the mean squared error between pixel values.
- B is the number of images inspected by the critic at a time.
- R<sup>1</sup> is the number of bits into which a given image is compressed.

Theorem 4.2. *Consider a finite set* X *such that* |X | ≥ 2, *a computable distribution* p<sup>X</sup> *on* X *such that* ∀x ∈ X , pX(x)>0, *a positive integer* B, *some* R > log(B)/ log(X ), *some* ∆ ∈ <sup>R</sup>+, *and a* p ⊗B <sup>X</sup> *-critic* δ. *Consider any conditional transition kernel* p<sup>Y</sup> <sup>|</sup><sup>X</sup> *from* X *to* X *satisfying*

$$p_Y \equiv p_X, \quad \mathbb{E}_p[d(X, Y)] \leq \Delta. \quad (13)$$

*Then, for any* ε ∈ (0, ∆/2), *and any* γ > 0, *the triplet* (R1, ∆1, C1) *is* δ*-achievable, with a* (R1, 0) *code, where*

$$R_1 := R \log(|\mathcal{X}|) \quad (14)$$

$$\Delta_1 := \Delta + \varepsilon + \frac{6\Delta}{\varepsilon} \max(d) \cdot \eta_{R,\gamma} \quad (15)$$

$$C_1 := \frac{3\Delta}{\varepsilon} \left[ \frac{B^2}{[2^{R_1}]} + 2B\eta_{R,\gamma} \right] \cdot \max_x B \log \frac{1}{p_X(x)} \quad (16)$$

$$\eta_{R,\gamma} := p(\mathcal{A}_{R,\gamma}) + 2^{-\gamma \log(|\mathcal{X}|)/2} \quad (17)$$

$$\mathcal{A}_{R,\gamma} := \left\{ (x, y) \in \mathcal{X}^2 \mid \log\left(\frac{p_X x_Y(x, y)}{p_X(x)p_Y(y)}\right) - \log(\lfloor 2^{R_1} \rfloor) > -\gamma \log(|\mathcal{X}|) \right\}, \quad (18)$$

*with the convention* 0/0 := 1.

The proof is provided in Appendix [B.](#page-12-0) The term B<sup>2</sup>/⌊2 <sup>R</sup><sup>1</sup> ⌋ is an upper bound on the probability that two source samples in the batch are compressed into the same message. This is related to the so-called *birthday paradox* (see Appendix [I\)](#page-23-0). The term max<sup>x</sup> B log(1/pX(x)) is an upper bound on the output of δ, which follows from Definition [2.1.](#page-3-1)

Theorem [4.2](#page-6-0) provides insights on the asymptotic regime of Theorem [4.1.](#page-5-0) Consider the limit of large |X |, with fixed R, ∆, ε, γ, and with log(B) = o(log |X |). We know that

$$\mathbb{E}_p \left[ \log \left( \frac{p_{X,Y}(x,y)}{p_X(x)p_Y(y)} \right) \right] = I_p(X; Y). \quad (19)$$

Hence, if this log-likelihood ratio concentrates well, and if R<sup>1</sup> > Ip(X; Y ), as in the definition of R(1)(·, 0) in [\(2\)](#page-0-0), then p(AR,γ) is small for small enough γ. In such an asymptotic regime, we obtain

**381**

**384**

**386**

∆<sup>1</sup> ≈ ∆, and C<sup>1</sup> = O(1). Therefore, the assumption in Theorem [4.1,](#page-5-0) that the source is of the form p ⊗n <sup>X</sup> for some large n, is only used to ensure fast concentration of the log-likelihood ratio. Hence, Theorem [4.1](#page-5-0) can be extended to a larger set of sources. In the next section, we present our last main result, which pertains to an asymptotic regime with large batch size.

#### 4.3 GENERALIZING THE DISTRIBUTION MATCHING FORMALISM

In this section, we present a result which connects our proposed formalism for the RDP trade-off to the distribution matching formalism of [Blau & Michaeli](#page-9-2) [\(2019\)](#page-9-2), and concludes our findings regarding the role of common randomness.

#### 4.3.1 BACKGROUND

Under the distribution matching formalism for the RDP trade-off, the natural asymptotic notion of achievability is as follows.

Definition 4.3. *[\(Saldi et al., 2015;](#page-10-7) [Blau & Michaeli, 2019\)](#page-9-2)*

*A quadruplet* (R, Rc, {Bn}n≥1, ∆) *is said to be asymptotically achievable with near-perfect realism if for any* ε > 0, *there exists a sequence of codes* {(F (n) , G(n) )}n, *the* n*-th being* (n, R + ε, Rc), *such that the sequence* {P (n)}<sup>n</sup> *of distributions induced by the codes satisfies*

$$\limsup_{n \rightarrow \infty} \mathbb{E}_{P(n)} [d(X_{1:n}, Y_{1:n})] \leq \Delta + \varepsilon,$$

$$\|P_{Y_{1:n}}^{(n)} - p_X^{\otimes n}\|_{TV} \xrightarrow{n \rightarrow \infty} 0. \quad (20)$$

The TVD in [\(20\)](#page-7-1) is directly related to the performance of the optimal hypothesis tester between the reconstruction distribution P (n) Y1:<sup>n</sup> , and the source distribution p ⊗n <sup>X</sup> [\(Blau & Michaeli, 2019\)](#page-9-2).

Replacing [\(20\)](#page-7-1) with

$$\exists N \in \mathbb{N}, \forall n \geq N, P_{Y_{1:n}}^{(n)} \equiv p_X^{\otimes n} \quad (21)$$

gives the notion of asymptotic *achievability with perfect realism*. It was shown that these two notions are equivalent for finite-valued sources [\(Saldi et al., 2015\)](#page-10-7), as well as for continuous sources under mild assumptions [\(Saldi et al., 2015;](#page-10-7) [Wagner, 2022\)](#page-10-8).

#### 4.3.2 CONNECTION TO OUR FORMALISM

As stated in the theorem below, in a certain large batch size regime, asymptotic achievability with algorithmic realism (Definition [3.5\)](#page-4-0) is equivalent to asymptotic achievability with near-perfect realism (Definition [4.3\)](#page-7-2). The proof is provided in Appendix [E.](#page-18-0)

Theorem 4.4. *Consider a computable increasing sequence* {Bn}n≥<sup>1</sup> *of positive integers such that*

$$\frac{B_n}{|\mathcal{X}|^n} \rightarrow \infty. \quad (22)$$

*Then, for any* R<sup>c</sup> ∈ <sup>R</sup>≥0, *and any* (R, ∆) ∈ (<sup>R</sup>+) 2 , *tuple* (R, Rc, {Bn}n≥1, ∆) *is asymptotically achievable with algorithmic realism if and only if* (R, Rc, ∆) *is asymptotically achievable with near-perfect realism, if and only if* (R, Rc, ∆) *is asymptotically achievable with perfect realism.*

Hence, Theorem [4.4,](#page-7-0) similarly to the finding in [Theis](#page-10-12) [\(2024\)](#page-10-12), shows that for large batch size, our formalism is equivalent to the distribution matching formalism. Hence, the former is a generalization of the latter. Moreover, Theorem [4.4](#page-7-0) and prior work on the distribution matching formalism [\(Saldi](#page-10-7) [et al., 2015;](#page-10-7) [Wagner, 2022;](#page-10-8) [Chen et al., 2022\)](#page-9-5) imply that common randomness is useful when the size of the batch inspected by the critic is extremely large.

# 5 DISCUSSION

Theorem [4.1](#page-5-0) states that common randomness does not improve the trade-off under our formalism, in all regimes where the batch size is not impractically large with respect to the dimension n of the

source. Theorem [4.4](#page-7-0) states that common randomness is useful — consistent with prior theoretical predictions — when the batch size is extremely large. Thus, Theorems [4.1](#page-5-0) and [4.4](#page-7-0) indicate that, in order to understand the role of randomization in lossy compression with realism constraints, the focus should be shifted to the size of the batch inspected by the critic. A continuation of our work could be to investigate realism metrics, where particular attention would be given to the choice of the batch size. This could lead to highlighting specific strengths and weaknesses of existing realism metrics. It may also inspire a critical assessment of the relative performance of existing compression schemes, depending on the choice of realism metric. Another continuation could be to more precisely characterize the amount of randomness needed as a function of the batch size. Furthermore, possible extensions of our setup include compression with side information, and other distributed settings.

**504**

**506**

**509**

**514 515 516**

**518 519**

**524**

**529**

**539**

# REFERENCES


[1] E. Agustsson, M. Tschannen, F. Mentzer, R. Timofte, and L. Van Gool. Generative Adversarial Networks for Extreme Learned Image Compression. In *IEEE/CVF International Conference on Computer Vision*, 2019.

[2] E. Agustsson, D. Minnen, G. Toderici, and F. Mentzer. Multi-Realism Image Compression with a Conditional Generator. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023. Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, and Nick Johnston. Variational image compression with a scale hyperprior. In *International Conference on Learning Representations*, 2018.

[3] Y. Blau and T. Michaeli. Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff. In *36th International Conference on Machine Learning*, 2019.

[4] C. L. Canonne. A short note on learning discrete distributions, 2020. arxiv:2002.11457.

[5] J. Chen, L. Yu, J. Wang, W. Shi, Y. Ge, and W. Tong. On the Rate-Distortion-Perception Function. *IEEE Journal on Selected Areas in Information Theory*, 3(4), 2022. ISSN 2641-8770. doi: 10.1109/JSAIT.2022.3231820. T.M. Cover and J.A. Thomas. *Elements of Information Theory*. Wiley-Interscience. Wiley, 2006. ISBN 9780471748816.

[6] P. Cuff. Distributed Channel Synthesis. *IEEE Transactions on Information Theory*, 59(11), 2013. ISSN 1557-9654. doi: 10.1109/TIT.2013.2279330. Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli. Image Quality Assessment: Unifying Structure and Texture Similarity. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2022.

[7] M. P. Eckert and A. P. Bradley. Perceptual quality metrics applied to still image compression. *Signal Processing*, 70(3):177–200, 1998.

[8] N. F. Ghouse, J. Petersen, A. Wiggers, T. Xu, and G. Sautière. A Residual Diffusion Model for High Perceptual Quality Codec Augmentation, 2023. arxiv:2301.05489.

[9] Y. Hamdi, A. B. Wagner, and D. Gündüz. The Rate-Distortion-Perception Trade-off: the Role of Private Randomness. In *IEEE International Symposium on Information Theory (ISIT)*, 2024.

[10] D. He, Z. Yang, H. Yu, T. Xu, J. Luo, Y. Chen, C. Gao, X. Shi, H. Qin, and Y. Wang. PO-ELIC: Perception-Oriented Efficient Learned Image Coding. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2022a. Dailan He, Ziming Yang, Weikun Peng, Rui Ma, Hongwei Qin, and Yan Wang. ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022b. Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In *Advances in Neural Information Processing Systems*, 2017.

[11] E. Hoogeboom, E. Agustsson, F. Mentzer, L. Versari, G. Toderici, and L. Theis. High-fidelity image compression with score-based generative models, 2023. arXiv:2305.18231.

[12] S. Iwai, T. Miyazaki, and S. Omachi. Controlling rate, distortion, and realism: Towards a single comprehensive neural image compression model. In *IEEE/CVF Winter Conference on Applications of Computer Vision*, 2024.

[13] M. Li and P. Vitányi. *An Introduction to Kolmogorov Complexity and Its Applications*. Texts in Computer Science. Springer International Publishing, Cham, 4th edition, 2019. ISBN 9783030112981. doi: 10.1007/978-3-030-11298-1.

[14] **540 541 542 543 544 545 546 547 548 549 554 555 556 559 561 564 569 571 572 573 574 579**

[15] F. Mentzer, G. D. Toderici, M. Tschannen, and E. Agustsson. High-Fidelity Generative Image Compression. *Advances in Neural Information Processing Systems*, 33, 2020.

[16] W. A. Pearlman and A. Said. *Digital Signal Compression: Principles and Practice*. Cambridge University Press, Cambridge (England), 2011. Ekta Prashnani, Hong Cai, Yasamin Mostofi, and Pradeep Sen. PieAPP: Perceptual Image-Error Assessment Through Pairwise Preference. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2018.

[17] N. Saldi, T. Linder, and S. Yüksel. Output Constrained Lossy Source Coding With Limited Common Randomness. *IEEE Transactions on Information Theory*, 61(9), 2015. doi: 10.1109/TIT.2015. 2450721.

[18] S. Santurkar, D. Budden, and N. Shavit. Generative Compression. In *Picture Coding Symposium*, 2018.

[19] K. Sayood. *Introduction to Data Compression*. Morgan Kaufmann, Waltham, MA (United States of America), 4th edition, 2012. ISBN 9780124160002.

[20] L. Theis. Position: What makes an image realistic? In *Forty-first International Conference on Machine Learning*, 2024.

[21] L. Theis and E. Agustsson. On the advantages of stochastic encoders. In *Neural Compression: From Information Theory to Applications – workshop at the International Conference on Learning Representations*, 2021.

[22] L. Theis and A. B. Wagner. A coding theorem for the rate-distortion-perception function. In *Neural Compression: From Information Theory to Applications – workshop at the International Conference on Learning Representations 2021*.

[23] M. Tschannen, E. Agustsson, and M. Lucic. Deep Generative Models for Distribution-Preserving Lossy Compression. In *NeurIPS*, 2018.

[24] A. B. Wagner. The Rate-Distortion-Perception Tradeoff: The Role of Common Randomness, 2022. arXiv:2202.04147.

[25] H. R. Wu, W. Lin, and L. J. Karam. An overview of perceptual processing for digital pictures. In *IEEE International Conference on Multimedia and Expo Workshops*, 2012.

[26] T. Xu, Q. Zhang, Y. Li, D. He, Z. Wang, Y. Wang, H. Qin, Y. Wang, J. Liu, and Y. Zhang. Conditional perceptual quality preserving image compression, 2023. arXiv:2308.08154.

[27] R. Yang and S. Mandt. Lossy Image Compression with Conditional Diffusion Models. In *NeurIPS*, 2023. Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2018.

[28] **604 605 606**

[29] **608 609**

[30] **614 615**

[31] **617**

[32] **619**

[33] **629**

[34] **634**

[35] **636**
# A FURTHER BACKGROUND ON ALGORITHMIC INFORMATION THEORY

#### A.1 COMPUTABILITY

This definition matches [Li & Vitányi](#page-9-13) [\(2019,](#page-9-13) Definition 1.7.4), except for the definition of a computable real number, which we adapted from [Li & Vitányi](#page-9-13) [\(2019,](#page-9-13) Exercise 1.7.22), and for the definition of a computable set, which matches that of [Li & Vitányi](#page-9-13) [\(2019,](#page-9-13) page 32).

Definition A.1. *Consider a subset* E *of* <sup>N</sup>≥0. *A map* f *from* E *into* <sup>N</sup> 3 ≥0 *is said to be computable if it corresponds to a Turing machine [\(Li & Vitányi, 2019,](#page-9-13) Section 1.7.1). This notion extends to functions having as domain other common countable sets, such as* N k ≥0 *for* k ∈ N, *and* {0, 1} ∗ , *or any subset thereof, by identifying elements of these sets with non-negative integers via some reference bijections. Consider a computable map* f *from a subset* E *of* <sup>N</sup>≥<sup>0</sup> *into* {0, 1} × <sup>N</sup>≥<sup>0</sup> × <sup>N</sup>. *Then, composing* f *with* (s, a, b) 7→ (2s − 1)a/b *yields a map from* E *to* Q, *which is said to be a computable map from* E *to* Q. *A map* f *from a subset* E *of* <sup>N</sup>≥<sup>0</sup> *into* <sup>R</sup> *is said to be lower semi-computable if there exists a computable function* φ *from* E × N *into* Q, *such that*

$$\forall x \in \mathcal{E}, \varphi(x, k) \xrightarrow[k \rightarrow \infty]{} f(x), \quad \text{and} \quad \forall x \in \mathcal{E}, \forall k \in \mathbb{N}, \quad \varphi(x, k+1) \geq \varphi(x, k).$$

*Moreover,* f *is said to be a computable map from* E *to* R *if both* f *and* −f *are lower semi-computable. A real number* λ *is said to be computable if the constant function* f : <sup>N</sup>≥<sup>0</sup> → <sup>R</sup>, n 7→ λ *is a computable function from* <sup>N</sup>≥<sup>0</sup> *to* <sup>R</sup>. *A (possibly infinite) subset* X *of* <sup>N</sup>≥0, *is said to be computable if there exists a computable function* f *from* <sup>N</sup>≥<sup>0</sup> *to* {0, 1}, *which returns* 1 *if its input is in* X , *and* 0 *otherwise.*

The following lemma allows to construct (semi-)computable functions. Its proof is deferred to Appendix [K.](#page-24-0)

Lemma A.2. *Let* E *denote a non-empty subset of* <sup>N</sup>≥0, *and let* f *and* g *denote functions from* E *to* <sup>R</sup>. *(i) If* f *and* g *are both lower semi-computable, then functions* f + g, ⌈f⌉, *and* 2 <sup>f</sup> *are lower semicomputable. If, in addition,* f *and* g *only take non-negative values, then* fg *and* 2 <sup>f</sup> /(3+f) <sup>2</sup> *are lower semi-computable. If, in addition,* f *only takes positive values, then* log(f) *is lower semi-computable. (ii) If* f *and* g *are both computable, then functions* f + g, fg, *and* |f| *are computable. If, in addition,* f *only takes positive values, then functions* 1/f, *and* f <sup>1</sup>/b *are computable, for any positive integer* b. *(iii) Let* X *be a computable finite subset of* {0, 1} ∗ . *If* f *is a lower semi-computable function from* {0, 1} ∗ *into* <sup>R</sup>, *then the function* ˜f : {0, 1} <sup>∗</sup> → <sup>R</sup> *which is null outside of* ∪n∈NX <sup>n</sup>, *and is defined by*

$$\forall x \in \cup_{n \in \mathbb{N}} \mathcal{X}^n, \quad \tilde{f}(x) = \sum_{y \in \mathcal{X}^l(x)} f(y),$$

*is lower semi-computable. Moreover, if* p *is a lower semi-computable probability measure on* X , *then* p ⊗∗ *is lower semi-computable.*

#### A.2 UNIVERSAL CRITICS AND SEMI-MEASURES

Definition A.3. *Given a finite set* W, *a function* f : W → [0, 1] *is a semi-measure if*

$$\sum_{w \in \mathcal{W}} f(w) \leq 1.$$

*It is said to be a lower semi-computable semi-measure if* f *is a semi-measure and* f *is lower semi-computable.*

The following theorem, corresponds to Definition 4.3.2, Equation (4.2), and Theorems 4.3.1 and 4.3.3 in [Li & Vitányi](#page-9-13) [\(2019\)](#page-9-13). It introduces the notion of *universal* p ⊗∗*-critic*, used in [Theis](#page-10-12) [\(2024\)](#page-10-12). The mixture m therein can be used as a prior distribution, which has been shown to be relevant in machine learning applications involving realism, such as outlier detection and generative modeling [\(Theis, 2024\)](#page-10-12).

Theorem A.4. *Consider a finite set* X , *each element of which is identified with a string in* {0, 1} s , *for some* s ∈ N. *Let* p *be a computable distribution on* X *such that* ∀x ∈ X , p(x) > 0. *Then, there*

**654**

**656**

**659 660 661**

**664**

**669**

**674**

**684**

**686**

**689 690 691**

*exists a* p ⊗∗*-critic* δ<sup>0</sup> *(which is not necessarily lower semi-computable), such that for any lower semi-computable* p ⊗∗*-critic* δ, *there exists a constant* c<sup>δ</sup> *such that*

$$\forall x \in \bigcup_{n \in \mathbb{N}} \mathcal{X}^n, \quad \delta_0(x) \geq \delta(x) - c_\delta. \quad (23)$$

*Any* p ⊗∗*-critic satisfying* [\(23\)](#page-12-1) *is called a universal* p ⊗∗*-critic.*

Since our definitions are slightly different from the classical ones, we provide a proof of Theorem [A.4](#page-11-2) in Appendix [J.](#page-23-1) Such a critic δ<sup>0</sup> is one of the best measures of realism deficiency according to p, in the limit of arbitrarily long strings. If a critic δ identifies a certain amount of deficiency in a given string, then δ<sup>0</sup> will identify at least as much deficiency, up to some additive constant. Intuitively, δ<sup>0</sup> is sensitive to all properties of randomness according to p. The existence of such a δ<sup>0</sup> constitutes a remarkable property of the set of all lower semi-computable p ⊗∗ <sup>X</sup> -critics (which is infinite).

Remark A.5. *[\(Li & Vitányi, 2019,](#page-9-13) Theorem 4.3.3) The universal semi-measure* m *can be chosen in such a way that*

$$\forall x \in \{0, 1\}^*, \quad | -\log(\mathbf{m}(x)) - K(x) | \leq c, \quad (24)$$

*for some constant* c, *where* K *is the Kolmogorov complexity [\(Li & Vitányi, 2019,](#page-9-13) Section 3.1). Property* [\(24\)](#page-12-2) *constitutes a strong result, since the Kolmogorov complexity is only defined up to a constant -we omit the corresponding details, for the purpose of clarity. The map* x 7→ log(1/p(x)) − K(x) *is sometimes considered to be an approximation of a universal* p ⊗∗ *critic, see, e.g., [Theis](#page-10-12) [\(2024\)](#page-10-12), and Appendix [J.](#page-23-1)*

# B PROOF OF THEOREM [4.2](#page-6-0)

#### B.1 OUTLINE

To show the achievability of a tuple (R1, ∆1, C1), it is not necessary to construct an explicit compression scheme: it is sufficient to prove the abstract existence of such a scheme. To that end, we consider a set of random reconstructions, and study its realism properties in Section [B.2.](#page-12-3) Then, we show the existence of a suitable choice of realizations of the latter reconstructions in Section [B.3.](#page-13-0) In Section [B.4,](#page-15-1) we prove Theorem [4.2](#page-6-0) by proposing a compression scheme achieving a close-to-uniform sampling from the set of reconstructions. For the remainder of Section [B,](#page-12-0) we fix a finite set X such that |X | ≥ 2, a computable distribution p<sup>X</sup> on X such that ∀x ∈ X , pX(x)>0, a positive integer B, and a p ⊗B <sup>X</sup> -critic δ.

#### B.2 REALISM PERFORMANCE OF A UNIFORMLY SAMPLED BATCH OF RANDOM RECONSTRUCTIONS

#### B.2.1 RANDOM CANDIDATE RECONSTRUCTIONS

Given a positive real R1, let C be a family of ⌊2 <sup>R</sup><sup>1</sup> ⌋ i.i.d. variables, each sampled from pX. The m-th variable is denoted y(C, m). We denote their joint distribution by QC. Given a realization c of C, we consider a batch y (B) of B elements of c, sampled uniformly with replacement. Then, we compute the batch's realism score δ(y (B) ). This is formalized in the following lemma, which gives an upper bound of the expected score with respect to QC.

Lemma B.1. *Consider a positive real* R<sup>1</sup> ∈ (log(B), ∞), *and the following pmf.*

$$\begin{aligned} \mathcal{Q}_{\mathcal{C}, \mathbf{M}^{(B)}, \mathbf{Y}^{(B)}} \left\{ \{y(m')\}_{m' \in [\lfloor 2^{R_1} \rfloor]}, \mathbf{m}^{(B)}, \mathbf{y}^{(B)} \right\} \\ := \left( \prod_{m'=1}^{\lfloor 2^{R_1} \rfloor} p_X(y(m')) \right) \cdot \frac{1}{\lfloor 2^{R_1} \rfloor^B} \cdot \prod_{k=1}^B \mathbf{1}_{y(k)=y(m(k))}. \end{aligned} \quad (25)$$

*Then, we have*

$$\mathbb{E}_Q[\delta(\mathbf{Y}^{(B)})] \leq \frac{B^2}{[2^{R_1}]} \max_x B \log \frac{1}{p_X(x)}. \quad (26)$$

**706**

**709**

**719**

**721**

**724**

**736**

**754**

#### B.2.2 REALISM PERFORMANCE

Claim B.2. *Since* R<sup>1</sup> > log(B), *a simple bound yields,*

$$(p_{[2^{R_1}]}^U)^{\otimes B}(M^{(1)}, \dots, M^{(B)}) \geq 1 - \frac{B^2}{[2^{R_1}]}.$$

See Appendix [I](#page-23-0) for a proof. From the definition (Section [B.2.1\)](#page-12-5) of Q, for any E ∈ B(R),

$$\begin{aligned} Q\left(\left\{\delta_0(\{y(\mathcal{C}, M^{(k)})\}_{k \in [B]}) \in \mathcal{E}\right\} \right. \\ \left. \left| \left\{M^{(1)}, \dots, M^{(B)} \text{ 2 by 2 distinct}\right\} \right) \right. \\ \left. = p_X^{\otimes B}\left(\delta_0(\mathbf{X}^{(B)}) \in \mathcal{E}\right). \right) \end{aligned} \tag{27}$$

Therefore, we have

$$\begin{aligned} & \mathbb{E}_Q[\delta(\{y(\mathcal{C}, M^{(k)})\}_{k \in [B]})] \\ &= \sum_{\mathbf{m}^{(B)}} \mathbb{E}_Q[\mathbf{1}_{\mathbf{M}^{(B)}=\mathbf{m}^{(B)}} \delta(\{y(\mathcal{C}, m^{(k)})\}_{k \in [B]})] \\ &= \sum_{\mathbf{m}^{(B)}} \mathbb{E}_Q[\mathbf{1}_{\mathbf{M}^{(B)}=\mathbf{m}^{(B)}}] \mathbb{E}_Q[\delta(\{y(\mathcal{C}, m^{(k)})\}_{k \in [B]})] \\ &= \sum_{\{m^{(k)}\}_{k \in [B]} \text{ 2 by } 2 \neq 2} (p_{[[2^{R_1}]]}^{\mathcal{U}})^{\otimes B} (\mathbf{M}^{(B)}=\mathbf{m}^{(B)}) \mathbb{E}_{p_X^{\otimes B}}[\delta(\mathbf{X}^{(B)})] \\ &+ \sum_{\{m^{(k)}\}_{k \in [B]} \text{ not } 2 \text{ by } 2 \neq 2} (p_{[[2^{R_1}]]}^{\mathcal{U}})^{\otimes B} (\mathbf{M}^{(B)}=\mathbf{m}^{(B)}) \mathbb{E}_Q[\delta(\{y(\mathcal{C}, m^{(k)})\}_{k \in [B]})] \\ &\leq \mathbb{E}_{p_X^{\otimes B}}[\delta(\mathbf{X}^{(B)})] + \max(\delta) (p_{[[2^{R_1}]]}^{\mathcal{U}})^{\otimes B} (M^{(1)}, \dots, M^{(B)} \text{ not } 2 \text{ by } 2 \neq 2) \\ &\leq \mathbb{E}_{p_X^{\otimes B}}[\delta(\mathbf{X}^{(B)})] + \frac{B^2}{[2^{R_1}]} \max_x B \log \frac{1}{p_X(x)}, \end{aligned} \tag{28}$$

where [\(28\)](#page-13-1) follows from Claim [B.2](#page-13-2) and [\(3\)](#page-3-4).

Claim B.3. *For any distribution* p *on a finite set, any* p*-critic* δ *satisfies*

$$\mathbb{E}_p[\delta(X)] \leq 0. \quad (29)$$

*Proof.* By setting q : x 7→ p(x) · 2 δ(x) , we can write

$$\forall x \in \mathcal{X} \text{ s.t. } p(x) > 0, \delta(x) = \log\left(\frac{q(x)}{p(x)}\right), \text{ with } 0 < \sum_{x \in \mathcal{X}} q(x) \leq 1. \quad (30)$$

We denote the latter sum by q(X ). Then, q/q(X ) is a probability distribution on X , and we have

$$\mathbb{E}_p[\delta(X)] \leq \mathbb{E}_p \left[ \log \left( \frac{q(X)/q(\mathcal{X})}{p(X)} \right) \mathbf{1}_{p(X)>0} \right] = -KL(p||q/q(\mathcal{X})) \leq 0. \quad (31)$$

This concludes the proof of Lemma [B.1.](#page-12-4)

#### B.3 FURTHER PROPERTIES OF A UNIFORMLY SAMPLED BATCH

Proposition B.4. *Consider a finite set* X *such that* |X | ≥ 2, *a distribution* p<sup>X</sup> *on* X *such that* ∀x ∈ X , pX(x)>0, *a positive integer* B, *some* R > log(B)/ log(|X |), *some* ∆ ∈ <sup>R</sup>+, *and a* p ⊗B <sup>X</sup> *-critic* δ. *Consider any conditional transition kernel* p<sup>Y</sup> <sup>|</sup><sup>X</sup> *from* X *to* X *satisfying*

$$p_Y \equiv p_X, \quad \mathbb{E}_p[d(X, Y)] \leq \Delta. \quad (32)$$

**759**

**761**

**764**

**766**

**769**

**779 780 781**

**784**

**804 805 806**

*Then, for any* ε ∈ (0, ∆/2), *and any* γ > 0, *there exists a family* {y(m)}m∈[⌊2R<sup>1</sup> ⌋] , *denoted* c, *of elements of* X , *such that distribution*

$$Q_{M,Y,X}(m, y, x) := \frac{1}{\lfloor 2R_1 \rfloor} \cdot (\mathbf{1}_{y=y(m)}) \cdot p_{X|Y=y(m)}(x) \quad (33)$$

*satisfies*

$$\|Q_X - p_X\|_{TV} \leq \frac{3\Delta}{\varepsilon} [p(\mathcal{A}_{R,\gamma}) + 2^{-\gamma \log(|\mathcal{X}|)/2}] \quad (34)$$

$$\mathbb{E}_{Q^{\otimes B}}[d(\mathbf{X}^{(B)}, \mathbf{Y}^{(B)})] \leq \Delta + \varepsilon \quad (35)$$

$$\mathbb{E}_{Q \otimes B}[\delta(\mathbf{Y}^{(B)})] \leq \frac{3\Delta}{\epsilon} \cdot \frac{B^2}{[2^{R_1}]} \max_x B \log \frac{1}{p_X(x)}, \quad (36)$$

*where* R<sup>1</sup> = R log |X|, *and*

$$\mathcal{A}_{R,\gamma} := \left\{ (x, y) \in \mathcal{X}^2 \mid \log\left(\frac{p_{X,Y}(x,y)}{p_X(x)p_Y(y)}\right) - \log(\lfloor 2^{R_1} \rfloor) > -\gamma \log(|\mathcal{X}|) \right\}. \quad (37)$$

*Proof.* Fix some R > log(B)/ log(|X |), some ∆ > 0, some ε ∈ (0, ∆/2), some γ > 0, and a conditional transition kernel p<sup>Y</sup> <sup>|</sup><sup>X</sup> from X to X satisfying

$$p_Y \equiv p_X, \quad \mathbb{E}_p[d(X, Y)] \leq \Delta. \quad (38)$$

Define R<sup>1</sup> = R log |X |. We apply Lemma [B.1,](#page-12-4) and use the notation therein. Then, from Markov's inequality, we have

$$Q_C\left(\mathbb{E}_Q[\delta(\mathbf{Y}^{(B)})|\mathcal{C}] \geq \frac{3\Delta}{\varepsilon} \frac{B^2}{[2^{R_1}]} \max_x B \log \frac{1}{p_X(x)}\right) \leq \frac{\varepsilon}{3\Delta}. \quad (39)$$

We extend distribution Q as follows.

$$Q_{\mathcal{C}, \mathbf{M}^{(B)}, \mathbf{Y}^{(B)}, \mathbf{X}^{(B)}} \left( \{y(m')\}_{m' \in [\lfloor 2^{R_1} \rfloor]}, \mathbf{m}^{(B)}, \mathbf{y}^{(B)}, \mathbf{x}^{(B)} \right) :=$$

$$Q_{\mathcal{C}, \mathbf{M}^{(B)}, \mathbf{Y}^{(B)}} \left( \{y(m')\}_{m' \in [\lfloor 2^{R_1} \rfloor]}, \mathbf{m}^{(B)}, \mathbf{y}^{(B)}\right) \cdot \prod_{k=1}^B p_{X|Y=y(m^{(k)})}(x^{(k)}). \quad (40)$$

Distribution QC,M(1),Y (1),X(1) corresponds to the setting of [Cuff](#page-9-15) [\(2013,](#page-9-15) Theorem VII.1), known as the soft covering lemma. Since p<sup>Y</sup> ≡ pX, the latter lemma yields that for any τ ∈ <sup>R</sup>,

$$\mathbb{E}_{\mathcal{C}} [\|Q_{X^{(1)}|\mathcal{C}} - p_X\|_{TV}] \leq p(\mathcal{A}_{\tau}) + 2^{\tau/2}, \quad (41)$$

where

$$\mathcal{A}_\tau := \{(x, y) \mid \log(p_{Y|X=x}(y)/p_X(y)) - \log(\lfloor 2^{R_1} \rfloor) > \tau\}. \quad (42)$$

We choose τ = −γ log |X |. Then, A<sup>τ</sup> = AR,γ, with the notation of Proposition [B.4.](#page-13-3) Hence, from [\(41\)](#page-14-0) and Markov's inequality, we have

$$Qc \left( \|Q_{X^{(1)}|C} - p_X\|_{TV} \geq \frac{3\Delta}{\varepsilon} [p(\mathcal{A}_{R,\gamma}) + 2^{-\gamma \log(|\mathcal{X}|)/2}] \right) \leq \frac{\varepsilon}{3\Delta}. \quad (43)$$

By construction, we have QY(B),X(B) ≡ p ⊗B Y,X. Therefore, from [\(38\)](#page-14-1), and the additivity of d, we have

$$\mathbb{E}_Q[d(\mathbf{X}^{(B)}, \mathbf{Y}^{(B)})] \leq \Delta. \quad (44)$$

Therefore, from Markov's inequality,

$$Q_C\left(\mathbb{E}_Q[d(\mathbf{X}^{(B)}, \mathbf{Y}^{(B)})|\mathcal{C}] \geq \Delta + \varepsilon\right) \leq \frac{\Delta}{\Delta + \varepsilon} = 1 - \frac{\varepsilon}{\Delta} \cdot \frac{1}{1 + \varepsilon/\Delta} < 1 - \frac{2\varepsilon}{3\Delta}, \quad (45)$$

where we have used the fact that ε ∈ (0, ∆/2). From a union bound and [\(39\)](#page-14-2), [\(43\)](#page-14-3), and [\(45\)](#page-14-4) there exists a realization c<sup>∗</sup> of C such that none of the corresponding events hold. Since, by construction,

$$Q_{\mathbf{M}^{(B)}, \mathbf{Y}^{(B)}, \mathbf{X}^{(B)} | \mathcal{C}=c*} \equiv Q_{M^{(1)}, Y^{(1)}, X^{(1)} | \mathcal{C}=c*}^{\otimes B},$$

**814 815**

**817**

**819**

**829**

**834**

**836**

**854**

**856**

#### B.4 PROOF OF THEOREM [4.2](#page-6-0)

Fix some R > log(B)/ log(|X |), some ∆ > 0, some ε ∈ (0, ∆/2), some γ > 0, and a conditional transition kernel p<sup>Y</sup> <sup>|</sup><sup>X</sup> from X to X satisfying

$$p_Y \equiv p_X, \quad \mathbb{E}_p[d(X, Y)] \leq \Delta. \quad (46)$$

Define R<sup>1</sup> = R log |X |. Then, we can apply Proposition [B.4.](#page-13-3) We use the notation from the latter.

### B.4.1 COMPRESSION SCHEME ACHIEVING CLOSE-TO-UNIFORM SAMPLING

We define the following distribution PX,Y,M, which differs from Q in having the correct marginal for X :

$$P_{X,M,Y} := p_X \cdot Q_{M,Y|X}. \quad (47)$$

Therefore, from [\(33\)](#page-14-5), distribution P satisfies Markov chain X−M−Y. Hence, it defines a (R1, 0) code. From Lemma [H.2](#page-22-0) (Appendix [H\)](#page-22-1), comparing P with Q reduces to comparing marginals, i.e. to [\(34\)](#page-14-6) :

$$\begin{aligned} \|P_{M,X,Y} - Q_{M,X,Y}\|_{TV} &= \|P_X - Q_X\|_{TV} \\ &= \|p_X - Q_X\|_{TV} \leq \frac{3\Delta}{\varepsilon} [p(\mathcal{A}_{R,\gamma}) + 2^{-\gamma \log(|\mathcal{X}|)/2}]. \end{aligned} \quad (48)$$

Since d is additive, we have

$$\mathbb{E}_{(P) \otimes B} [d(\mathbf{X}^{(B)}, \mathbf{Y}^{(B)})] = \mathbb{E}_P [d(X, Y)] \text{ and}$$

$$\mathbb{E}_{(P)\otimes B}[d(\mathbf{X}^{(B)}, \mathbf{Y}^{(B)})] = \mathbb{E}_P[d(X, Y)] \text{ and }$$

$$\mathbb{E}_{(Q)\otimes B}[d(\mathbf{X}^{(B)}, \mathbf{Y}^{(B)})] = \mathbb{E}_Q[d(X, Y)].$$

Since d is bounded, then we can apply Lemma [H.3](#page-22-2) (Appendix [H\)](#page-22-1). Then, from [\(48\)](#page-15-2), and Lemma [H.1](#page-22-3) with W = (X, Y ), we have

$$\begin{aligned}\mathbb{E}_{\mathbf{P} \otimes \mathbf{B}} [d(\mathbf{X}^{(B)}, \mathbf{Y}^{(B)})] &\leq \mathbb{E}_{\mathbf{Q} \otimes \mathbf{B}} [d(\mathbf{X}^{(B)}, \mathbf{Y}^{(B)})] + \frac{6\Delta}{\varepsilon} \max(d) [p(\mathcal{A}_{R,\gamma}) + 2^{-\gamma \log(|\mathcal{X}|)/2}] \\ &\leq \Delta + \varepsilon + \frac{6\Delta}{\varepsilon} \max(d) [p(\mathcal{A}_{R,\gamma}) + 2^{-\gamma \log(|\mathcal{X}|)/2}],\end{aligned}\tag{49}$$

where the last inequality follows from [\(35\)](#page-14-7). Moving to the realism performance, we have the following property of the TVD - see Appendix [H:](#page-22-1)

Claim B.5. *Given any two distributions* P *and* Q *on the same finite alphabet, we have, for any* B ∈ N,

$$\left\| P^{\otimes B} - Q^{\otimes B} \right\|_{TV} \leq B \left\| P - Q \right\|_{TV}.$$

From Lemma [H.3,](#page-22-2) Claim [B.5,](#page-15-3) [\(48\)](#page-15-2), and Lemma [H.1](#page-22-3) with W = Y(B) , we have,

$$\begin{aligned} \mathbb{E}_{\mathcal{D} \otimes \mathcal{B}} [\delta(\mathbf{Y}^{(B)})] &\leq \mathbb{E}_{\mathcal{Q} \otimes \mathcal{B}} [\delta(\mathbf{Y}^{(B)})] + \frac{6B\Delta}{\varepsilon} \max(\delta) [p(\mathcal{A}_{R,\gamma}) + 2^{-\gamma \log(|\mathcal{X}|)/2}] \\ &\leq \frac{3\Delta}{\varepsilon} \cdot \frac{B^2}{[2R_1]} \max_x B \log \frac{1}{p_X(x)} + \frac{6B\Delta}{\varepsilon} [p(\mathcal{A}_{R,\gamma}) + 2^{-\gamma \log(|\mathcal{X}|)/2}] \cdot \max_x B \log \frac{1}{p_X(x)}. \quad (50) \end{aligned}$$

This concludes the proof.

# C ACHIEVABILITY OF THEOREM [4.1](#page-5-0)

Consider some ∆ ∈ <sup>R</sup><sup>+</sup> such that R(1)(∆, 0) < Hp(X), and a sequence {Bn}n≥<sup>1</sup> of positive integers such that

$$\log(B_n)/n \xrightarrow{n \rightarrow \infty} 0. \quad (51)$$

$$p_Y \equiv p_X, \quad \mathbb{E}_p[d(X, Y)] \leq \Delta, \quad R \geq I_p(X; Y) + \varepsilon. \quad (52)$$

**869**

**874**

**884**

**886**

**889 890 891**

**904**

**906**

**909**

We use the powerful result of Theorem [A.4](#page-11-2) regarding the existence of a so-called *universal critic*. From Definition [2.1,](#page-3-1) for every n ∈ <sup>N</sup>, the restriction of δ<sup>0</sup> to X nB<sup>n</sup> is a p ⊗nB<sup>n</sup> <sup>X</sup> -critic. Moreover, from [\(51\)](#page-15-4), for large enough n, we have nR > log(Bn). Then, for large enough n, we can apply Theorem [4.2](#page-6-0) for set X <sup>n</sup>, distribution p ⊗n <sup>X</sup> , transition kernel Q<sup>n</sup> <sup>t</sup>=1 p<sup>Y</sup> |X, batch size Bn, critic δ0, rate nR/ log(|X <sup>n</sup>|), and constants ∆, ε, γ. This gives that, for every n large enough, there is a (n, R, 0) code, inducing a distribution P (n) such that

$$\begin{aligned} \mathbb{E}_{(P(n)) \otimes B_n} [d(\mathbf{X}^{(n, B_n)}, \mathbf{Y}^{(n, B_n)})] &\leq \Delta + \varepsilon + \frac{6\Delta}{\varepsilon} \max(d) [p(\mathcal{A}_{R, \gamma}^{(n)}) + 2^{-\gamma n \log(|\mathcal{X}|)/2}], \quad (53) \\ \mathbb{E}_{(P(n)) \otimes B_n} [\delta_0(\mathbf{Y}^{(n, B_n)})] &\leq \\ \frac{3\Delta}{\varepsilon} \left[ \frac{B_n^2}{[2^{2nR}]} \max_x n B_n \log \frac{1}{p_X(x)} + 2B_n [p(\mathcal{A}_{R, \gamma}^{(n)}) + 2^{-\gamma n \log(|\mathcal{X}|)/2}] \cdot \max_x n B_n \log \frac{1}{p_X(x)} \right], \quad (54) \end{aligned}$$

where

$$\mathcal{A}_{R,\gamma}^{(n)} := \left\{ (x_{1:n}, y_{1:n}) \in (\mathcal{X}^n)^2 \mid \sum_{t=1}^n \log \left( \frac{p_{X,Y}(x_t, y_t)}{p_X(x_t)p_Y(y_t)} \right) - \log(\lfloor 2^{nR} \rfloor) > -\gamma n \log(|\mathcal{X}|) \right\}, \quad (55)$$

with the convention 0/0 := 1. From [\(52\)](#page-15-5), log(⌊2 nR⌋)/n − γ log(|X |) > Ip(X; Y ) for large enough n. Then, since X is finite, we have, from Hoeffding's inequality,

$$p(\mathcal{A}_{R,\gamma}^{(n)}) = O(e^{-\kappa n}), \quad (56)$$

for some κ > 0. Hence, from [\(51\)](#page-15-4), [\(53\)](#page-16-0), [\(54\)](#page-16-1), and Theorem [A.4,](#page-11-2) we have

$$\limsup_{n \rightarrow \infty} \mathbb{E}_{(P(n)) \otimes B_n} [d(\mathbf{X}^{(n, B_n)}, \mathbf{Y}^{(n, B_n)})] \leq \Delta + \varepsilon, \quad (57)$$

and for any lower semi-computable p ⊗∗ <sup>X</sup> -critic δ,

$$\sup_{n \in \mathbb{N}} \mathbb{E}_{(P^{(n)}) \otimes B_n} [\delta(\mathbf{Y}^{(n, B_n)})] < \infty. \quad (58)$$

From the proof of Theorem [4.2,](#page-6-0) we know that P (n) has a deterministic decoder. Hence, it only remains to derandomize the encoder of P (n) . We denote its decoder by m 7→ y1:n(m). The following claim is a slight modification of [Hamdi et al.](#page-9-16) [\(2024,](#page-9-16) Proposition 4). We provide details in Section [C.1.](#page-17-1)

Claim C.1. *There exists a sequence of deterministic maps*

$$\begin{aligned} f^{(n)} : \mathcal{X}^n &\rightarrow [2^{nR}], \quad \text{such that} \\ \|\hat{P}_{\mathcal{X}^2}^{(n)}[X^n, y_{1:n}(M)] - \hat{P}_{\mathcal{X}^2}^{(n)}[X^n, y_{1:n}(M)]\|_{TV} &\xrightarrow{n \rightarrow \infty} 0, \\ \liminf_{n \rightarrow \infty} \frac{-1}{n} \log \|\hat{P}_M^{(n)} - P_M^{(n)}\|_{TV} &> 0, \quad \text{where} \end{aligned} \tag{59}$$

$$\hat{P}_{X^n, M}^{(n)} := p_X^{\otimes n} \cdot \mathbf{1}_{M=f^{(n)}(X^n)}.$$

Then, from [\(51\)](#page-15-4) and Claim [B.5,](#page-15-3) we have

$$\liminf_{n \rightarrow \infty} \frac{-1}{n} \log \|(\tilde{P}^{(n)})_M^{\otimes B_n} - (P^{(n)})_M^{\otimes B_n}\|_{TV} > 0. \quad (60)$$

Thus, from Lemma [H.3](#page-22-2) and [\(3\)](#page-3-4), we have

$$|\mathbb{E}_{(\bar{P}(n) \otimes B_n} [\delta(\mathbf{Y}^{(n, B_n)})] - \mathbb{E}_{(P(n) \otimes B_n} [\delta(\mathbf{Y}^{(n, B_n)})])|_{n \rightarrow \infty} = 0. \quad (61)$$

Moreover, since d is bounded, then from Lemma [H.3,](#page-22-2) we obtain

$$\|\mathbb{E}_{(P(n))\otimes B_n} [d(\mathbf{X}^{(n,B_n)}, \mathbf{Y}^{(n,B_n)})] - \mathbb{E}_{(P(n))\otimes B_n} [d(\mathbf{X}^{(n,B_n)}, \mathbf{Y}^{(n,B_n)})]\|_{n \rightarrow \infty} 0. \quad (62)$$

Since this analysis is valid for any ε ∈ (0, R − R(1)(∆, 0)), then tuple (R, {Bn}n≥1, ∆) is asymptotically achievable with algorithmic realism with fully deterministic codes. This being true for any R ∈ (R(1)(∆, 0), Hp(X)), we have

$$R(\Delta) \leq R_*(\Delta) \leq R^{(1)}(\Delta, 0),$$

**924**

**929**

**954**

**956**

**959**

**961**

#### C.1 ENCODER DERANDOMIZATION

We show that Claim [C.1](#page-16-2) follows from [Hamdi et al.](#page-9-16) [\(2024,](#page-9-16) Proposition 4), and its proof. We can apply that result directly, since R < Hp(X) and X is finite. This would give all properties in Claim [C.1,](#page-16-2) except for the exponential decay in [\(59\)](#page-16-3). To obtain the latter, it is sufficient to adapt the proof of [Hamdi et al.](#page-9-16) [\(2024,](#page-9-16) Proposition 4), by replacing the use of the law of large numbers with the use of Hoeffding's inequality, and using [Cuff](#page-9-15) [\(2013,](#page-9-15) Theorem VII.1) with τ = −nγ, for small enough γ.

### D CONVERSE OF THEOREM [4.1](#page-5-0)

From standard information-theoretic arguments, we have the following result - see Appendix [F](#page-20-2) for a proof.

Lemma D.1. *Consider a triplet* (R, Rc, ∆) *and a sequence of codes, the* n*-th being* (n, R, Rc), *inducing a sequence* {P (n) X1:n,J,M,Y1:<sup>n</sup> }n≥<sup>1</sup> *of distributions such that*

$$\limsup_{n \rightarrow \infty} \mathbb{E}_{(P(n)) \otimes b_n} [d(\mathbf{X}^{(n,b_n)}, \mathbf{Y}^{(n,b_n)})] \leq \Delta, \quad (63)$$

*for some sequence* {bn}n≥<sup>1</sup> *of positive integers. For every* n ≥ 1, *let* T (n) *denote a uniform variable on* [nbn] *independent from all other random variables. Then, there exists a conditional distribution* p<sup>Y</sup> <sup>|</sup><sup>X</sup> *and an increasing sequence* {ni}i≥<sup>1</sup> *of positive integers such that*

$$(P^{(n_i)})_{X_{T^{(n_i)}}, Y_{T^{(n_i)}}}^{\otimes b_{n_i}} \xrightarrow{i \rightarrow \infty} p_{X,Y} \quad (64)$$

$$\Delta \geq \mathbb{E}_p[d(X, Y)] \quad (65)$$

$$R \geq I_p(X; Y), \quad (66)$$

*where* pX,Y *refers to* p<sup>X</sup> · p<sup>Y</sup> <sup>|</sup>X.

### D.1 CONVERSE PROOF

Consider some ∆ ∈ <sup>R</sup><sup>+</sup> such that R(1)(∆, 0) < Hp(X), and a sequence {Bn}n≥<sup>1</sup> of positive integers such that

$$\log(B_n)/n \xrightarrow{n \rightarrow \infty} 0. \quad (67)$$

We know that R∗(∆) ≥ R(∆), and prove that R(∆) ≥ R(1)(∆, 0). Consider a couple (R, ∆) ∈ <sup>R</sup> 2 +, and some R<sup>c</sup> ∈ <sup>R</sup>≥<sup>0</sup> such that (R, Rc, {Bn}n≥1, ∆) is asymptotically achievable with algorithmic realism. Fix ε > 0. Then, there exists a sequence of codes, the n-th being (n, R, Rc), inducing a sequence {P (n) X1:n,J,M,Y1:<sup>n</sup> }<sup>n</sup> of distributions such that

$$\limsup_{n \rightarrow \infty} \mathbb{E}_{(P(n)) \otimes B_n} [d(\mathbf{X}^{(n, B_n)}, \mathbf{Y}^{(n, B_n)})] \leq \Delta + \varepsilon, \quad (68)$$

and for any lower semi-computable p ⊗∗ <sup>X</sup> -critic δ,

$$\sup_{n \in \mathbb{N}} \mathbb{E}_{(P^{(n)}) \otimes B_n} [\delta(\mathbf{Y}^{(n, B_n)})] < \infty. \quad (69)$$

Then, Lemma [D.1](#page-17-2) applies, with b<sup>n</sup> = Bn, for all n, with R + ε instead of R, and ∆ + ε instead of ∆. Then, there exists a conditional distribution p<sup>Y</sup> <sup>|</sup><sup>X</sup> and an increasing sequence {ni}i≥<sup>1</sup> of positive integers such that

$$(P^{(n_i)})_{X_{T^{(n_i)}}, Y_{T^{(n_i)}}}^{\otimes b_{n_i}} \xrightarrow{i \rightarrow \infty} p_{X,Y} \quad (70)$$

$$\Delta + \varepsilon \geq \mathbb{E}_p[d(X, Y)] \quad (71)$$

$$R + \varepsilon \geq I_p(X; Y), \quad (72)$$

where for any n ∈ N, variable T (n) is uniformly distributed on [nBn], and independent from all other random variables. We prove that p<sup>Y</sup> ≡ pX. Fix e<sup>0</sup> ∈ X . Consider the computable p ⊗∗ <sup>X</sup> -critic δ from Claim [G.1,](#page-20-1) with q therein taken to be pX. Then, from [\(69\)](#page-17-3),

$$\sup_{n \in \mathbb{N}} \mathbb{E}_{(P(n)) \otimes B_n} [\delta(\mathbf{Y}^{(n, B_n)}) - 2 \log(\delta(\mathbf{Y}^{(n, B_n)}) + 3)] < \infty. \quad (73)$$

**979**

**989 990 991**

**994**

**1011**

**1014 1015**

Thus,

$$\sup_{n \in \mathbb{N}} \mathbb{E}_{(P(n)) \otimes B_n} [\delta(\mathbf{Y}^{(n, B_n)})] < \infty, \quad \text{and} \quad \mathbb{E}_{(P(n)) \otimes B_n} [\delta(\mathbf{Y}^{(n, B_n)})] - \frac{1}{2} \log(nB_n) \xrightarrow{n \rightarrow \infty} -\infty.$$

Thus, the frequency of e<sup>0</sup> in a batch of reconstructions converges in L<sup>1</sup> norm to pX(e0). Hence, the expected frequencies converge to pX(e0). This rewrites as

$$(P^{(n)})_{Y_{T^{(n)}}}^{\otimes B_n}(e_0) \rightarrow p_X(e_0). \quad (74)$$

This is true for any e<sup>0</sup> in X . Thus, from [\(70\)](#page-17-4), p<sup>Y</sup> ≡ pX. Hence, from [\(71\)](#page-17-5) and [\(72\)](#page-17-6), we have

$$R + \varepsilon \geq R^{(1)}(\Delta + \varepsilon, 0). \quad (75)$$

This being true for any ε > 0, and since R(1)(·, 0) is convex -thus continuous- on (0, ∞), we have

$$R \geq R^{(1)}(\Delta, 0). \quad (76)$$

This being true for any R ∈ <sup>R</sup><sup>+</sup> such that there exists R<sup>c</sup> ∈ <sup>R</sup>≥<sup>0</sup> such that (R, Rc, {Bn}n≥1, ∆) is asymptotically achievable with algorithmic realism, we have

$$R(\Delta) \geq R^{(1)}(\Delta, 0), \quad (77)$$

as desired.

# E PROOF OF THEOREM [4.4](#page-7-0)

Consider an increasing sequence {Bn}n≥<sup>1</sup> of positive integers such that

$$\frac{B_n}{|\mathcal{X}|^n} \rightarrow \infty, \quad (78)$$

some R<sup>c</sup> ∈ <sup>R</sup>≥0, and some (R, ∆) ∈ (<sup>R</sup>+) 2 such that tuple (R, Rc, ∆) is asymptotically achievable with near-perfect realism. From Theorem 1 in [Wagner](#page-10-8) [\(2022\)](#page-10-8), (R, Rc, ∆) achievable with *perfect realism*, i.e. satisfying the properties in Definition [4.3,](#page-7-2) with [\(20\)](#page-7-1) replaced with

$$\exists N \in \mathbb{N}, \forall n \geq N, P_{Y_{1:n}}^{(n)} \equiv p_X^{\otimes n}. \quad (79)$$

Fix ε > 0, and a corresponding sequence of (n, R + ε, Rc) codes. Denote by P (n) the distribution induces by the n-th code. Then, there exists an integer N<sup>ε</sup> such that

$$\limsup_{n \rightarrow \infty} \mathbb{E}_{P^{(n)}} [d(X_{1:n}, Y_{1:n})] \leq \Delta + \varepsilon, \quad (80)$$

$$\forall n \geq N_\varepsilon, (P_{Y_{1:n}}^{(n)})^{\otimes B_n} \equiv p_X^{\otimes n B_n}. \quad (81)$$

From [\(80\)](#page-18-1), [\(81\)](#page-18-2), Claim [B.3,](#page-13-4) and the additivity of the distortion measure d, we have

$$\limsup_{n \rightarrow \infty} \mathbb{E}_{(P^{(n)}) \otimes B_n} [d(\mathbf{X}^{(n, B_n)}, \mathbf{Y}^{(n, B_n)})] \leq \Delta + \varepsilon, \quad (82)$$

and for any lower semi-computable p ⊗∗ <sup>X</sup> -critic δ,

$$\sup_{n \in \mathbb{N}} \mathbb{E}_{(P(n)) \otimes B_n} [\delta(\mathbf{Y}^{(n, B_n)})] < \infty. \quad (83)$$

Since this analysis is valid for every ε > 0, then (R, Rc, {Bn}n≥1, ∆) is asymptotically achievable with algorithmic realism. Moving to the converse, consider a computable increasing sequence {Bn}n≥<sup>1</sup> of positive integers such that

$$\frac{B_n}{|\mathcal{X}|^n} \rightarrow \infty, \quad (84)$$

some R<sup>c</sup> ∈ <sup>R</sup>≥0, and some (R, ∆) ∈ (<sup>R</sup>+) 2 such that tuple (R, Rc, ∆) is asymptotically achievable with algorithmic realism. Fix ε > 0. Then, there exists a sequence of codes, the n-th being (n, R + ε, Rc), such that the sequence {P (n)}<sup>n</sup> of distributions induced by the codes satisfies

$$\limsup_{n \rightarrow \infty} \mathbb{E}_{(P(n)) \otimes B_n} [d(\mathbf{X}^{(n, B_n)}, \mathbf{Y}^{(n, B_n)})] \leq \Delta + \varepsilon, \quad \text{and} \quad (85)$$

$$\sup_{n \in \mathbb{N}} \mathbb{E}_{(P(n)) \otimes B_n} [\delta(\mathbf{Y}^{(n, B_n)})] < \infty. \quad (86)$$

**1029**

**1034**

**1054**

**1056**

**1071**

Lemma E.1. *[\(Canonne, 2020\)](#page-9-17) There exists a positive integer* λ *such that for any* k ∈ N, *any distribution* q *on some finite set* W *of size* k, *any* ε, η > 0, *and any integer* b *satisfying*

$$b \geq \lambda \cdot \frac{k + \log(1/\eta)}{\varepsilon^2}, \quad (87)$$

*we have*

$$q^{\otimes b} \left( \|\mathbb{P}_{\mathcal{W}}^{emp}[W^b] - q\|_{TV} \geq \varepsilon \right) \leq \eta. \quad (88)$$

For every n ∈ N, define

$$C_n := \left\lceil \left( \frac{B_n}{|\mathcal{X}|^n} \right)^{\frac{1}{3}} \right\rceil. \quad (89)$$

Since X is finite, {Cn}n≥<sup>1</sup> is a computable sequence of positive integers. Moreover, from [\(84\)](#page-18-3), we have

$$C_n \xrightarrow{n \rightarrow \infty} \infty. \quad (90)$$

Choosing, for every n ∈ <sup>N</sup>, η = 1/3 and ε = 1/Cn, then from Lemma [E.1](#page-19-0) and [\(90\)](#page-19-1) we have, for large enough n,

$$(P^{(n)})^{\otimes B_n} \left( \left\| \mathbb{P}_{\mathcal{X}^n}^{\text{emp}} [\mathbf{Y}^{(n, B_n)}] \right\|_{TV} - P_{Y_{1:n}}^{(n)} \right\|_{TV} \geq \frac{1}{C_n} \right) \leq \frac{1}{3}. \quad (91)$$

Consider the computable sequence of positive integers defined by

$$\forall n \in \mathbb{N}, A_n := \left\lceil \left( \frac{B_n}{|\mathcal{X}|^n} \right)^{\frac{4}{9}} \right\rceil. \quad (92)$$

Since {Bn}n≥<sup>1</sup> is increasing, then for any t ∈ <sup>N</sup>, there exists a unique integer n ∈ <sup>N</sup>≥<sup>0</sup> such that

$$t \in [nB_n, (n+1)B_{n+1}),$$

with the definition B<sup>0</sup> := 0. We define δ : ∪t∈NX <sup>t</sup> → <sup>N</sup>≥<sup>0</sup> as follows. For any integer t ∈ [1, B1), and any x1:<sup>t</sup> ∈ X <sup>t</sup> , let δ(x) := 0. For any n ∈ <sup>N</sup>, any t ∈ [nBn,(n + 1)Bn+1), and any x1:<sup>t</sup> ∈ X <sup>t</sup> , let

$$\delta(x_{1:t}) := \left[ A_n \|\mathbb{P}_{\mathcal{X}^n}^{\text{emp}}[x_{1:nB_n}] - p_X^{\otimes n}\|_{TV} \right]. \quad (93)$$

Claim E.2. *From Lemma [E.1](#page-19-0) and* [\(90\)](#page-19-1)*, there exists a positive integer* L *such that* δ−2 log(δ+ 3)−L *is a lower semi-computable* p ⊗∗ <sup>X</sup> *-critic.*

We provide a proof in Appendix [G.2.](#page-21-0) Then, we can apply [\(86\)](#page-18-4) to critic δ − 2 log(δ + 3) − L, and get,

$$\sup_{n \in \mathbb{N}} \mathbb{E}_{(P(n)) \otimes B_n} [\delta(\mathbf{Y}^{(n, B_n)}) - 2 \log(\delta(\mathbf{Y}^{(n, B_n)}) + 3) - L] < \infty. \quad (94)$$

Thus,

$$\sup_{n \in \mathbb{N}} \mathbb{E}_{(P(n)) \otimes B_n} [\delta(\mathbf{Y}^{(n, B_n)})] < \infty, \quad \text{and} \quad (P^{(n)})^{\otimes B_n}(\delta(\mathbf{Y}^{(n, B_n)})) \geq C_n \xrightarrow{n \rightarrow \infty} 0,$$

because {Cn}n≥<sup>1</sup> tends to infinity. Combining this with [\(91\)](#page-19-2) through a union bound, we obtain, from the triangle inequality for the TVD,

$$(P^{(n)})^{\otimes B_n} (\|P_{Y_{1:n}}^{(n)} - p_X^{\otimes n}\|_{TV} \leq \frac{C_n}{A_n} + \frac{1}{C_n}) > 0,$$

for large enough n. The above event does not depend on the random batch, hence the corresponding inequality is true, for large enough n. Since {Cn}n≥<sup>1</sup> tends to infinity and since from [\(84\)](#page-18-3), [\(89\)](#page-19-3), and [\(92\)](#page-19-4), we have Cn/A<sup>n</sup> → 0, then we obtain

$$\|P_{Y_{1:n}}^{(n)} - p_X^{\otimes n}\|_{TV} \xrightarrow{n \rightarrow \infty} 0. \quad (95)$$

**1099**

**1104**

**1106**

**1109**

**1119**

#### F STANDARD CONVERSE ARGUMENTS

Here, we provide a proof of Lemma [D.1](#page-17-2) (Appendix [D\)](#page-17-0). The sequence of distributions (P (n) ) ⊗b<sup>n</sup> X<sup>T</sup> ,Y<sup>T</sup> can be seen as a bounded sequence in R 2 2s , thus it admits a converging subsequence:

$$(P^{(n_i)})_{X_T, Y_T}^{\otimes b_n} \xrightarrow{i \rightarrow \infty} p_{X, Y}. \quad (96)$$

Since d is bounded, we have

$$\mathbb{E}_{(P(n_i)) \otimes b_n} [d(X_T, Y_T)] \xrightarrow{i \rightarrow \infty} \mathbb{E}_p[d(X, Y)]. \quad (97)$$

Since d is additive, we have, for any n ∈ N,

$$\begin{aligned} \mathbb{E}_{(P^{(n)})\otimes b_n} [d(\mathbf{X}^{(n,b_n)}, \mathbf{Y}^{(n,b_n)})] \\ = \mathbb{E}_{(P^{(n)})\otimes b_n} [d(X_T, Y_T)]. \end{aligned} \quad (98)$$

From [\(63\)](#page-17-7), [\(97\)](#page-20-3) and [\(98\)](#page-20-4), we have ∆ ≥ <sup>E</sup>p[d(X, Y )]. Secondly, distribution P (n) satisfies

$$\begin{aligned}
nb_n R &\geq H(\{m^{(k)}\}_{k \in [b_n]} | \{J^{(k)}\}_{k \in [b_n]}) \\
&\geq I(\{m^{(k)}\}_{k \in [b_n]}; \mathbf{X}^{(n,b_n)} | \{J^{(k)}\}_{k \in [b_n]}) \\
&= I(\{m^{(k)}\}_{k \in [b_n]}; \{J^{(k)}\}_{k \in [b_n]}; \mathbf{X}^{(n,b_n)}) \\
&\geq I(\mathbf{Y}^{(n,b_n)}; \mathbf{X}^{(n,b_n)}) \\
&\geq \sum_{k=1}^{b_n} \sum_{t=1}^n I(Y_t^{(k)}; X_t^{(k)}) \\
&= nb_n I(Y_T; X_T | T) \\
&= nb_n I(T, Y_T; X_T) \\
&\geq nb_n I(Y_T; X_T).
\end{aligned}$$

Therefore, from [\(96\)](#page-20-5), and by continuity of mutual information on the set of distributions on ({0, 1} s ) 2 , we have R ≥ Ip(X; Y ).

# G FREQUENCY CRITICS

#### G.1 CRITIC INVOLVING THE FREQUENCY OF A SPECIFIC PATTERN

The following claim, and its proof, are inspired from [Li & Vitányi](#page-9-13) [\(2019,](#page-9-13) Lemma 4.3.5 & Exercise 2.4.1).

Claim G.1. *Consider a finite set* X , *identified with a subset of* {0, 1} s . *Let* q *be a distribution on* X *such that* ∀x ∈ X , q(x) > 0. *Let* e<sup>0</sup> *be any string in* X *, considered as a pattern of interest. For any* n ∈ <sup>N</sup> *and any* x1:<sup>n</sup> ∈ X <sup>n</sup>, *let* S(x1:n) *denote the number of occurrences of* e<sup>0</sup> *in* x1:n. *Define map* δ:∪n∈<sup>N</sup> → <sup>N</sup>≥<sup>0</sup> *by*

$$\forall n \in \mathbb{N}, \forall x_{1:n} \in \mathcal{X}^n, \quad x_{1:n} \mapsto \left\lceil \log \left[ |S(x_{1:n}) - q(e_0)n| / \sqrt{n} \right] \right\rceil. \quad (99)$$

*Then,* δ − 2 log(δ + 3) *is a computable* q ⊗∗*-critic.*

*Proof.* From Lemma [A.2,](#page-11-3) δ is lower semi-computable. Since δ − 2 log(δ + 3) = log(2<sup>δ</sup>/(δ + 3)<sup>2</sup> ), then by Lemma [A.2,](#page-11-3) δ − 2 log(δ + 3) is lower semi-computable. For any (n, C) ∈ N 2 , and any x1:<sup>n</sup> ∈ X <sup>n</sup>, we have:

$$\begin{aligned} \{\delta(x_{1:n}) \geq C\} &= \left\{ \left\lceil \log \left\lceil |S(x_{1:n}) - nq(e_0)| / \sqrt{n} \right\rceil \right\rceil \geq C \right\} \\ &= \left\{ \log \left\lceil |S(x_{1:n}) - nq(e_0)| / \sqrt{n} \right\rceil > C - 1 \right\} \\ &= \left\{ \left\lceil |S(x_{1:n}) - nq(e_0)| / \sqrt{n} \right\rceil > 2^{C-1} \right\} \\ &= \left\{ |S(x_{1:n}) - nq(e_0)| / \sqrt{n} > 2^{C-1} \right\}. \end{aligned}$$

**1154**

**1159**

**1171**

**1174 1175**

**1177**

From this and Chebyshev inequality, we obtain:

$$\begin{aligned} q^{\otimes n}(\delta(X_{1:n}) \geq C) &\leq \mathbb{E}_{q^{\otimes n}} \left[ (S(X_{1:n}) - nq(e_0))^2 / 4^{C-1} \right. \\ &\quad \left. = (q(e_0) - q(e_0)^2) / 4^{C-1} \right. \\ &\leq 4^{-C} \\ &\leq 2^{-C}, \end{aligned} \tag{100}$$

where [\(100\)](#page-21-1) comes from the fact that S(X1:n) follows a binomial distribution B(n, q(e0)). Thus,

$$\mathbb{E}_{q^{\otimes n}}[\mathbf{1}_{\delta(X_{1:n})=C}] \leq 2^{-C}. \quad (101)$$

Therefore,

$$\mathbb{E}_{q^{\otimes n}}[\mathbf{1}_{[\delta(X_{1:n})=C} \cdot 2^{\delta(X_{1:n})-2\log(\delta(X_{1:n})+3)}] \leq \frac{1}{(C+3)^2}. \quad (102)$$

This also holds for C = 0. Summing over C ∈ <sup>N</sup>≥<sup>0</sup> gives, for any n ∈ <sup>N</sup>,

$$\sum_{x_{1:n} \in \mathcal{X}^n} q^{\otimes n}(x_{1:n}) \cdot 2^{\delta(x_{1:n}) - 2 \log(\delta(x_{1:n}) + 1) - 1} \leq 1. \quad (103)$$

Hence, we have that δ − 2 log(δ + 3) is a lower semi-computable q ⊗∗-critic.

G.2 CRITIC INVOLVING AN EMPIRICAL DISTRIBUTION

#### We provide a proof of Claim [E.2.](#page-19-5)

Claim G.2. *The map* f : ∪t∈NX <sup>t</sup> → <sup>R</sup> *defined by* ∀t ∈ [1, B1) ∩ <sup>N</sup>, ∀x1:<sup>t</sup> ∈ X <sup>t</sup> , f(x1:t) := 0, *and* ∀n ∈ <sup>N</sup>, ∀t ∈ [nBn,(n + 1)Bn+1) ∩ <sup>N</sup>, ∀x1:<sup>t</sup> ∈ X <sup>t</sup> , f(x1:t) := <sup>P</sup> *emp* <sup>X</sup> <sup>n</sup> [x1:nB<sup>n</sup> ] − p ⊗n T V *(104) is computable.*

*Proof.* Since there exists s ∈ N such that X ⊆ {0, 1} s , then, given some x ∈ ∪t∈NX <sup>t</sup> → <sup>R</sup>, one can compute the unique corresponding t via a Turing machine. Moreover, since {Bn}n≥<sup>1</sup> is computable, one can further compute the unique n such that t ∈ [nBn,(n + 1)Bn+1) via a Turing machine, as well as the empirical probability appearing in [\(104\)](#page-21-2). For any k ∈ <sup>N</sup>, and any x<sup>0</sup> ∈ X , one can call the rational-valued computable upper and lower approximations of p at point (x0, k). Then, one can go over all y1:<sup>n</sup> ∈ X <sup>n</sup>, and use the explicit constructions from the proof of Lemma [A.2](#page-11-3) regarding the product, sum, and absolute value, yielding rational-valued computable upper and lower approximations of f.

We know that Ann≥<sup>1</sup> is computable. From Lemma [A.2,](#page-11-3) the product of two computable functions is computable, thus lower semi-computable, and the ceiling function preserves semi-computability. Therefore, δ is lower semi-computable. Since δ − 2 log(δ + 3) = log(2<sup>δ</sup>/(δ + 3)<sup>2</sup> ), then by Lemma [A.2,](#page-11-3) for any positive integer L, function δ − 2 log(δ + 3) − L is lower semi-computable. It remains to prove that a certain choice of L yields a p ⊗∗ <sup>X</sup> -critic. From [\(84\)](#page-18-3) and [\(92\)](#page-19-4), there exists N<sup>0</sup> ∈ <sup>N</sup> such that

$$\forall n \geq N_0, B_n \geq \lambda(|\mathcal{X}|^n + 2)A_n^2. \quad (105)$$

For any n ≥ N0, any C ≥ 2, any integer t ∈ [nBn,(n + 1)Bn+1), and any x1:<sup>t</sup> ∈ X <sup>t</sup> , we have:

$$\begin{aligned} \{\delta(x_{1:t}) \geq C\} &= \left\{ \left[ A_n \|\mathbb{P}_{\mathcal{X}^n}^{\text{emp}}[x_{1:n}B_n] - p^{\otimes n}\|_{TV} \right] \geq C \right\} \\ &= \left\{ A_n \|\mathbb{P}_{\mathcal{X}^n}^{\text{emp}}[x_{1:n}B_n] - p^{\otimes n}\|_{TV} > C - 1 \right\}. \end{aligned}$$

From this, [\(105\)](#page-21-3), and Lemma [E.1,](#page-19-0) with distribution p ⊗n <sup>X</sup> , and parameters b = Bn,

ε = (C−1)/An, η = 2−<sup>C</sup> we obtain,

$$\forall t \geq [N_0 B_{N_0}, \infty) \cap \mathbb{N}, \forall C \in \mathbb{N}_{\geq 2}, \quad p_X^{\otimes t}(\delta(X_{1:t}) \geq C) \leq 2^{-C}.$$

$$\forall t \geq [N_0 B_{N_0}, \infty) \cap \mathbb{N}, \forall C \in \mathbb{N}_{\geq 2}, \quad \mathbb{E}_{p_X^{\otimes t}} [\mathbf{1}_{\delta(X_{1:t})=C} \cdot 2^{\delta(X_{1:t})-2 \log(\delta(X_{1:t})+3)}] \leq \frac{1}{(C+3)^2}.$$

**1224**

**1227**

**1229**

This also holds for C ∈ {0, 1}. Summing over C ∈ <sup>N</sup>≥<sup>0</sup> gives,

$$\forall t \geq [N_0 B_{N_0}, \infty) \cap \mathbb{N}, \quad \sum_{x_{1:t} \in \mathcal{X}^t} p_X^{\otimes t}(x_{1:t}) \cdot 2^{\delta(x_{1:t}) - 2 \log(\delta(x_{1:t}) + 3)} \leq 1. \quad (106)$$

In order to extend this to all positive integers t, it is sufficient to multiply by 2 <sup>−</sup><sup>L</sup> for some L large enough. Therefore, there exists L ∈ N such that δ − 2 log(δ + 3) − L is a lower semi-computable p ⊗∗ <sup>X</sup> -critic. This concludes the proof.

# H ON THE TOTAL VARIATION DISTANCE

#### H.1 SOME LEMMAS

Lemma H.1. *Let* Π *and* Γ *be two distributions on a set* W × L. *Then*

$$\|\Pi_W - \Gamma_W\|_{TV} \leq \|\Pi_{W,L} - \Gamma_{W,L}\|_{TV}.$$

Lemma H.2. *Let* Π *and* Γ *be two distributions on a set* W ×L. *Then when using the same conditional probability kernel* ΠL|<sup>W</sup> , *we have*

$$\|\Pi_W \Pi_{L|W} - \Gamma_W \Pi_{L|W}\|_{TV} = \|\Pi_W - \Gamma_W\|_{TV}.$$

Lemma H.3. *Let* Π *and* Γ *be two distributions on a set* W, *and* f : W → R *be a bounded function. Then,*

$$|\mathbb{E}_\Pi[f] - \mathbb{E}_\Gamma[f]| \leq 2 \max |f| \cdot \|\Pi - \Gamma\|_{TV}.$$

# H.2 PROOF OF CLAIM [B.5](#page-15-3)

Let P and Q be any two distributions on the same alphabet. Fix a positive integer B. Then, we have, with the convention Π ⊗ Γ <sup>⊗</sup><sup>0</sup> ≡ Π,

$$\begin{aligned} \|P^B - Q^B\|_{TV} &= \left\| \sum_{k=1}^B (P^{\otimes(B-k+1)} \otimes Q^{\otimes(k-1)} - P^{\otimes(B-k)} \otimes Q^{\otimes k}) \right\|_{TV} \\ &\leq \sum_{k=1}^B \|P^{\otimes(B-k+1)} \otimes Q^{\otimes(k-1)} - P^{\otimes(B-k)} \otimes Q^{\otimes k}\|_{TV} \end{aligned} \quad (107)$$

$$\begin{aligned} &\leq \sum_{k=1}^B \|P^{\otimes(B-k)} \otimes P \otimes Q^{\otimes(k-1)} - P^{\otimes(B-k)} \otimes Q \otimes Q^{\otimes(k-1)}\|_{TV} \\ &\leq \sum_{k=1}^B \|P - Q\|_{TV} = B\|P - Q\|_{TV}, \end{aligned} \quad (108)$$

**1267**

**1281**

**1284**

**1287**

#### I THE BIRTHDAY PARADOX

We provide a proof of Claim [B.2.](#page-13-2) We have

$$\begin{aligned}
 (p_{[\lfloor 2^{R_1} \rfloor}^U)^{\otimes B}(M^{(1)}, \dots, M^{(B)} \text{ 2 by 2 distinct}) &= \prod_{k=1}^B \frac{\lfloor 2^{R_1} \rfloor - k + 1}{\lfloor 2^{R_1} \rfloor} \\
 &\geq \frac{(\lfloor 2^{R_1} \rfloor - B + 1)^B}{\lfloor 2^{R_1} \rfloor^B} \\
 &\geq \left(1 - \frac{B-1}{\lfloor 2^{R_1} \rfloor}\right)^B \\
 &\geq 1 - \frac{B(B-1)}{\lfloor 2^{R_1} \rfloor} \\
 &\geq 1 - \frac{B^2}{\lfloor 2^{R_1} \rfloor},
 \end{aligned} \tag{109}$$

where [\(109\)](#page-23-2) follows from Bernoulli's inequality, since R<sup>1</sup> > log(B).

#### J EXISTENCE OF A UNIVERSAL p ⊗∗ -CRITIC

We provide a proof of Theorem [A.4.](#page-11-2) From [Li & Vitányi](#page-9-13) [\(2019,](#page-9-13) Theorem 4.3.1), there exists a sequence {qn}n≥<sup>1</sup> containing all lower semi-computable semi-measures on {0, 1} ∗ , and a sequence {πn}n≥<sup>1</sup> of (strictly) positive reals, such that the mixture defined by

$$\mathbf{m} := \sum_{n \geq 1} \pi_n q_n \quad (110)$$

is a lower semi-computable semi-measure on {0, 1} ∗ . For every n ∈ N, let m(X <sup>n</sup>) denote

$$\sum_{x_{1:n} \in \mathcal{X}^n} \mathbf{m}(x_{1:n}).$$

From [\(110\)](#page-23-3), we have ∀x ∈ {0, 1} ∗ , m(x) > 0. Moreover, ∀x<sup>0</sup> ∈ X , p(x0) > 0, thus ∀x ∈ ∪n∈NX <sup>n</sup>, p⊗∗(x) > 0. Define function δ0, by

$$\forall n \in \mathbb{N}, \forall x_{1:n} \in \mathcal{X}^n, \delta_0(x_{1:n}) := \log \left( \frac{\mathbf{m}(x_{1:n})}{\mathbf{m}(\mathcal{X}^n)p^{\otimes n}(x_{1:n})} \right). \quad (111)$$

Fix any lower semi-computable p ⊗∗-critic δ. Define map q<sup>δ</sup> : {0, 1} <sup>∗</sup> → <sup>R</sup> by

$$\forall x \in \cup_{n \in \mathbb{N}} \mathcal{X}^n, \quad q_\delta(x) := \mathbf{m}(\mathcal{X}^{l(x)}) 2^{\delta(x)} p^{\otimes *}(x),$$

and x 7→ 0 elsewhere. From Lemma [A.2](#page-11-3) (iii), the function which is null outside of ∪n∈NX n, and defined by x 7→ m(X l(x) ) on ∪n∈NX <sup>n</sup>, is lower semi-computable. Moreover, x 7→ 2 δ(x) and x 7→ p ⊗∗(x) are lower semi-computable by Lemma [A.2](#page-11-3) (i) and (iii) respectively. Thus, q<sup>δ</sup> is the product three non-negative lower semi-computable functions. Hence, q<sup>δ</sup> is lower semi-computable by Lemma [A.2](#page-11-3) (i). Moreover, we have

$$\begin{aligned} \sum_{x \in \{0,1\}^*} q_\delta(x) &= \sum_{n \in \mathbb{N}} \mathbf{m}(\mathcal{X}^n) \sum_{x \in \mathcal{X}^n} 2^{\delta(x)} p^{\otimes n}(x) \\ &\leq \sum_{n \in \mathbb{N}} \mathbf{m}(\mathcal{X}^n) \\ &\leq 1, \end{aligned} \tag{112} \tag{113}$$

where [\(112\)](#page-23-4) follows from the definition of a p ⊗∗-critic; and [\(113\)](#page-23-5) follows from the fact that m is a semi-measure. Therefore, q<sup>δ</sup> is a lower semi-computable semi-measure. Thus, from [\(110\)](#page-23-3), we have m ≥ π<sup>q</sup><sup>δ</sup> qδ, for some positive real π<sup>q</sup><sup>δ</sup> . In order to derive [\(23\)](#page-12-1), fix x ∈ ∪n∈NX <sup>n</sup>, and denote l(x) by

**1317**

**1319**

**1321**

**1324**

**1334**

n. From [\(110\)](#page-23-3), we have m(x) > 0. Therefore, since ∀x<sup>0</sup> ∈ X , p(x0) > 0, we have qδ(x) > 0. Thus, from [\(111\)](#page-23-6), we have

$$\begin{aligned}\delta_0(x) &= \log \left( \frac{\mathbf{m}(x_{1:n})}{\mathbf{m}(\mathcal{X}^n)p^{\otimes n}(x_{1:n})} \right) \\ &\geq \log \left( \frac{\pi_{q_\delta} q(x_{1:n})}{\mathbf{m}(\mathcal{X}^n)p^{\otimes n}(x_{1:n})} \right) \\ &= \log(\pi_{q_\delta}) + \delta(x).\end{aligned}$$

This is true for any lower semi-computable p ⊗∗-critic δ, and any x ∈ ∪n∈NX <sup>n</sup>. Since log(π<sup>q</sup><sup>δ</sup> ) does not depend on x, then property [\(23\)](#page-12-1) holds. This concludes the proof.

# K ADDITIONAL SEMI-COMPUTABILITY ARGUMENTS

We provide a proof of Lemma [A.2.](#page-11-3) If f is lower semi-computable, we denote by (x, k) 7→ φf,<sup>−</sup>(x, k) a computable function from E to Q, monotonically approaching f from below, in the sense of Definition [A.1.](#page-11-4) If f is upper semi-computable, then φf,+(x, k) denotes a function of the form φ−f,<sup>−</sup>(x, k), which monotonically approaches f from above.

#### K.1 ASSUME THAT f AND g ARE COMPUTABLE

#### K.1.1 f + g

Function φf,<sup>−</sup> + φg,<sup>−</sup> is a computable function from E × <sup>N</sup> to Q, which monotonically approaches f + g from below. Similarly, φf,<sup>+</sup> + φg,<sup>+</sup> constitutes a computable rational upper approximation.

# K.1.2 |f|

We construct φ|f|,<sup>−</sup> as follows. Let x ∈ E and k ∈ <sup>N</sup>. If φf,<sup>−</sup>(x, k) ≥ 0, return |φf,<sup>−</sup>(x, k)|. Otherwise, if φf,+(x, k) ≤ 0, return |φf,+(x, k)|. Otherwise, return 0. We define φ|f|,+(x, k) as

$$\max \left( |\varphi_{f,-}(x, k)|, |\varphi_{f,+}(x, k)| \right).$$

Straightforwardly, this implies that |f| is computable.

### K.1.3 fg

Define φf g,<sup>−</sup>(x, k) as follows. If φf,<sup>−</sup>(x, k) ≥ 0 and φg,<sup>−</sup>(x, k) ≥ 0, then return φf,<sup>−</sup>(x, k)φg,<sup>−</sup>(x, k). Otherwise, if φf,+(x, k) ≤ 0 and φg,+(x, k) ≤ 0, then return φf,+(x, k)φg,+(x, k). Otherwise, return

$$-\max \left( |\varphi_{f,-}(x, k)|, |\varphi_{f,+}(x, k)| \right) \max \left( |\varphi_{g,-}(x, k)|, |\varphi_{g,+}(x, k)| \right).$$

Define φf g,<sup>+</sup> as −φ(−f)g,−.

K.2 SUPPOSE THAT f IS COMPUTABLE AND ONLY TAKES POSITIVE VALUES

### K.2.1 1/f

Define φ1/f,−(x, k) as 1/φf,+(x, k). Compute k1(x), the smallest positive integer k such that φf,<sup>−</sup>(x, k) > 0. For all integers k ∈ [1, k1(x)], define φ1/f,+(x, k) as 1/φf,<sup>−</sup>(x, k1(x)). For all integers k ∈ (k1(x), ∞), define φ1/f,+(x, k) as 1/φf,<sup>−</sup>(x, k).

#### K.2.2 f 1/b

Compute k1(x), the smallest positive integer k such that φf,<sup>−</sup>(x, k) > 0. For all integers k ∈ [1, k1(x)), define φ<sup>f</sup> <sup>1</sup>/b,<sup>−</sup>(x, k) := 0 and φ<sup>f</sup> <sup>1</sup>/b,+(x, k) = ⌈φf,+(x, k)⌉. Consider an integer k ∈ [k1(x), ∞). Compute the greatest integer m such that (m/2 k ) <sup>b</sup> ≤ φf,<sup>−</sup>(x, k). Then, define

**1354**

**1371**

**1374**

φ<sup>f</sup> <sup>1</sup>/b,<sup>−</sup>(x, k) := m/2 k . Therefore, we have

$$\forall k \geq k_1(x), \quad 0 \leq \varphi_{f,-}(x, k)^{1/b} - \varphi_{f^{1/b},-}(x, k) < \frac{1}{2^k}. \quad (114)$$

From [\(114\)](#page-25-0), and since the b-th root function and k 7→ φf,<sup>−</sup>(x, k) are both non-decreasing, we have

$$\forall k \geq k_1(x) + 1, \quad \varphi_{f^{1/b},-}(x, k-1) \leq \varphi_{f,-}(x, k)^{1/b}. \quad (115)$$

Since φ<sup>f</sup> <sup>1</sup>/b,<sup>−</sup>(x, k − 1) can also be written in the form m′/2 k , then, from the maximality of the integer m appearing in the construction of φ<sup>f</sup> <sup>1</sup>/b,<sup>−</sup>(x, k), we have

$$\forall k \geq k_1(x) + 1, \quad \varphi_{f^{1/b},-}(x, k-1) \leq \varphi_{f^{1/b},-}(x, k). \quad (116)$$

This also holds for all integers k ∈ [2, k1(x) + 1). Properties [\(114\)](#page-25-0) and [\(116\)](#page-25-1) imply that f <sup>1</sup>/b is lower semi-computable. We prove upper semi-computability similarly, using the smallest integer m˜ such that ( ˜m/2 k ) <sup>b</sup> ≥ φf,+(x, k), and setting φ<sup>f</sup> <sup>1</sup>/b,+(x, k) := ˜m/2 k .

### K.3 ASSUME THAT f AND g ARE LOWER SEMI-COMPUTABLE

# K.3.1 f + g

Function φf,<sup>−</sup> + φg,<sup>−</sup> is a computable function from E × <sup>N</sup> to Q, which monotonically approaches f + g from below.

# K.3.2 ⌈f⌉

Define φ⌈f⌉,<sup>−</sup> as ⌈φf,<sup>−</sup>⌉.

#### K.3.3 2 f

Fix x ∈ E and k ∈ <sup>N</sup>. Let a ∈ <sup>Z</sup> and b ∈ <sup>N</sup> such that φf,<sup>−</sup>(x, k) = a/b. Compute the greatest integer m such that (m/2 k ) <sup>b</sup> ≤ 2 a . Then, define φ<sup>2</sup> <sup>f</sup> ,<sup>−</sup>(x, k) := m/2 k . Therefore, we have

$$0 \leq 2^{\varphi_f, -(x,k)} - \varphi_{2f, -}(x, k) < \frac{1}{2^k}. \quad (117)$$

From [\(117\)](#page-25-2), and since the exponential function and k 7→ φf,<sup>−</sup>(x, k) are both non-decreasing, we have

$$\forall k \geq 2, \varphi_{2f,-}(x, k-1) \leq 2^{\varphi_f,-(x,k)}. \quad (118)$$

Since φ<sup>2</sup> <sup>f</sup> ,<sup>−</sup>(x, k −1) can also be written in the form m′/2 k , then, from the maximality of the integer m appearing in the construction of φ<sup>2</sup> <sup>f</sup> ,<sup>−</sup>(x, k), we have

$$\forall k \geq 2, \varphi_{2f,-}(x, k-1) \leq \varphi_{2f,-}(x, k). \quad (119)$$

Properties [\(117\)](#page-25-2) and [\(119\)](#page-25-3) imply that 2 f is lower semi-computable.

K.4 ASSUME THAT f AND g ARE SEMI-COMPUTABLE AND NON-NEGATIVE

#### K.4.1 fg

If φf,<sup>−</sup>(x, k) ≥ 0 and φg,<sup>−</sup>(x, k) ≥ 0, return φf,<sup>−</sup>(x, k)φg,<sup>−</sup>(x, k). Otherwise, return 0.

#### K.4.2 2 <sup>f</sup> /(3 + f) 2

There exists a real ε ∈ (0, 1) such that u 7→ 2 <sup>u</sup>/(3 + u) 2 is non-decreasing on (−ε, ∞). Fix x ∈ E. Compute k1(x), the smallest positive integer k such that φf,<sup>−</sup>(x, k) > −ε. For all integers k ∈ [1, k1(x)), define φ<sup>2</sup> <sup>f</sup> /(3+f) <sup>2</sup>,<sup>−</sup>(x, k) := 0. Fix an integer k ≥ k1(x). Let a ∈ <sup>Z</sup> and b ∈ <sup>N</sup> such that φf,<sup>−</sup>(x, k) = a/b. Compute the greatest integer m such that (m/2 k ) <sup>b</sup> ≤ 2 <sup>a</sup>/(3 + a/b) 2b . Then, define φ<sup>2</sup> <sup>f</sup> /(3+f) <sup>2</sup>,<sup>−</sup>(x, k) := m/2 k . Therefore, we have

$$\forall k \geq k_1(x), \quad 0 \leq \frac{2\varphi_{f,-}(x,k)}{(3 + \varphi_{f,-}(x,k))^2} - \varphi_{2f}/(3+f)^2, -(x,k) < \frac{1}{2^k}. \quad (120)$$

From [\(120\)](#page-25-4), and since k 7→ φf,<sup>−</sup>(x, k) is non-decreasing, and u 7→ 2 <sup>u</sup>/(3 + u) is non-decreasing on (−ε, ∞), we have

$$\forall k \geq k_1(x) + 1, \quad \varphi_{2f/(3+f)^2, -}(x, k-1) \leq \frac{2\varphi_{f, -}(x, k)}{(3 + \varphi_{f, -}(x, k))^2}. \quad (121)$$

Since φ<sup>2</sup> <sup>f</sup> /(3+f) <sup>2</sup>,<sup>−</sup>(x, k − 1) can also be written in the form m′/2 k , then, from the maximality of the integer m appearing in the construction of φ<sup>2</sup> <sup>f</sup> /(3+f) <sup>2</sup>,<sup>−</sup>(x, k), we have

$$\forall k \geq k_1(x) + 1, \quad \varphi_{2f/(3+f)^2}, -(x, k-1) \leq \varphi_{2f/(3+f)^2}, -(x, k). \quad (122)$$

This is also true for all integers k ∈ [2, k1(x) + 1). Properties [\(120\)](#page-25-4) and [\(122\)](#page-26-0) imply that 2 <sup>f</sup> /(3 +f) 2 is lower semi-computable.

### K.4.3 log(f)

Assume that f only takes positive values. Fix x ∈ E. Compute k1(x), the smallest positive integer k such that φf,<sup>−</sup>(x, k) > 0. Fix an integer k ≥ k1(x). Compute the largest integer m such that 2 <sup>m</sup> ≤ φf,<sup>−</sup>(x, k) 2 k . Then, define φlog(f),<sup>−</sup>(x, k) := m/2 k . For all integers k ∈ [1, k1(x)), define φlog(f),<sup>−</sup>(x, k) as φlog(f),<sup>−</sup>(x, k1(x)). Therefore, we have

$$\forall k \geq k_1(x), \quad 0 \leq \log(\varphi_{f,-}(x, k)) - \varphi_{\log(f),-}(x, k) < \frac{1}{2^k}. \quad (123)$$

From [\(123\)](#page-26-1), and since the logarithm and k 7→ φf,<sup>−</sup>(x, k) are both non-decreasing, we have

$$\forall k \geq k_1(x) + 1, \quad \varphi_{\log(f),-}(x, k-1) \leq \log(\varphi_{f,-}(x, k)). \quad (124)$$

Since φlog(f),<sup>−</sup>(x, k − 1) can also be written in the form m′/2 k , then, from the maximality of the integer m appearing in the construction of φlog(f),<sup>−</sup>(x, k), we have

$$\forall k \geq k_1(x) + 1, \quad \varphi_{\log(f), -}(x, k-1) \leq \varphi_{\log(f), -}(x, k). \quad (125)$$

This also holds for all integers k ∈ [2, k1(x) + 1). Properties [\(123\)](#page-26-1) and [\(125\)](#page-26-2) imply that log(f) is lower semi-computable.

#### K.5 FUNCTIONS OF FINITE BINARY STRINGS

Let X be a finite computable subset of {0, 1} ∗ , and f be a lower semi-computable function from {0, 1} ∗ into R.

Lemma K.1. ∪n∈NX <sup>n</sup> *is a computable set.*

*Proof.* By Definition [A.1,](#page-11-4) it is sufficient to construct a computable function τ from {0, 1} ∗ to {0, 1}, which returns 1 if its input is in ∪n∈NX <sup>n</sup>, and 0 otherwise. Since X is computable, there exists a computable function τ<sup>0</sup> from {0, 1} ∗ to {0, 1}, which returns 1 if its input is in X , and 0 otherwise. Fix x ∈ {0, 1} ∗ . Define τ (x) as follows. Enumerate all partitions of x into consecutive sub-strings. For each, call τ<sup>0</sup> on every sub-string. If for some partition, the output of τ<sup>0</sup> is 1 for every sub-string, then return 1. Otherwise, return 0.

Hereafter, we use the notation τ defined in the above proof.

#### K.5.1 PARTIAL SUMS

Consider the function ˜f : {0, 1} <sup>∗</sup> → <sup>R</sup> which is null outside of ∪n∈NX <sup>n</sup>, and is defined by

$$\forall x \in \cup_{n \in \mathbb{N}} \mathcal{X}^n, \quad \tilde{f}(x) = \sum_{y \in \mathcal{X}^l(x)} f(y).$$

Fix x ∈ {0, 1} ∗ and k ∈ <sup>N</sup>. Define φf ,˜ <sup>−</sup>(x, k) as follows. Compute τ (x). If it is null, return 0. Otherwise: compute l(x), and for each y in X l(x) , compute φf,<sup>−</sup>(y, k), then return

$$\sum_{y \in \mathcal{X}^{l(x)}} \varphi_{f,-}(y, k).$$

Fix some x ∈ ∪n∈NX <sup>n</sup>. The (finite) set of indices of the above sum does not depend on k. Therefore, since for each y ∈ {0, 1} <sup>∗</sup> we have φf,<sup>−</sup>(y, k) → k→∞ f(y), we get

$$\forall x \in \cup_{n \in \mathbb{N}} \mathcal{X}^n, \quad \varphi_{\tilde{f}, -}(x, k) \xrightarrow{k \rightarrow \infty} \tilde{f}(x). \quad (126)$$

Similarly, since for any y ∈ {0, 1} ∗ and any k ≥ 1, we have φf,<sup>−</sup>(y, k) ≤ φf,<sup>−</sup>(y, k + 1), then we have

$$\forall x \in \cup_{n \in \mathbb{N}} \mathcal{X}^n, \forall k \in \mathbb{N}, \quad \varphi_{\tilde{f}, -}(x, k) \leq \varphi_{\tilde{f}, -}(x, k + 1). \quad (127)$$

Properties [\(126\)](#page-27-0) and [\(127\)](#page-27-1) also hold for finite strings outside of ∪n∈NX <sup>n</sup>. Thus, ˜f is lower semicomputable.

#### K.5.2 PRODUCT DISTRIBUTION

Let p be a lower semi-computable probability measure on X . Fix x ∈ {0, 1} ∗ and k ∈ N. Define φp⊗∗,<sup>−</sup>(x, k) as follows. Compute τ (x). If it is null, return 0. Otherwise, proceed as follows. Compute l(x). We write x as x1:l(x) , with x<sup>t</sup> ∈ X for any integer t in [1, l(x)]. Compute and return

$$\prod_{t=1}^{l(x)} \varphi_{p,-}(x_t, k).$$

Fix some x ∈ ∪n∈NX <sup>n</sup>. The (finite) set of indices of the above product does not depend on k. Therefore, since for each y ∈ X , we have φp,<sup>−</sup>(y, k) → k→∞ p(y), we get

$$\forall x \in \cup_{n \in \mathbb{N}} \mathcal{X}^n, \quad \varphi_{p^{\otimes *}, -}(x, k) \xrightarrow{k \rightarrow \infty} p^{\otimes *}(x). \quad (128)$$

Similarly, since for any y ∈ X , and any k ≥ 1, we have φp,<sup>−</sup>(y, k) ≤ φp,<sup>−</sup>(y, k + 1), and since p is non-negative, then we have

$$\forall x \in \cup_{n \in \mathbb{N}} \mathcal{X}^n, \forall k \in \mathbb{N}, \quad \varphi_{p^{\otimes *}, -}(x, k) \leq \varphi_{p^{\otimes *}, -}(x, k + 1). \quad (129)$$

Properties [\(128\)](#page-27-2) and [\(129\)](#page-27-3) also hold for finite strings outside of ∪n∈NX <sup>n</sup>. Thus, p ⊗∗ is lower semi-computable.