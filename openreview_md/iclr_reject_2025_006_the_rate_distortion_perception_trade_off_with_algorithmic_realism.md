000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Realism constraints (or constraints on perceptual quality) have received considerable recent attention within the context of lossy compression, particularly of images. Theoretical studies of lossy compression indicate that high-rate common randomness between the compressor and the decompressor is a valuable resource for achieving realism. On the other hand, the utility of significant amounts of common randomness at test time has not been noted in practice. We offer an explanation for this discrepancy by considering a realism constraint that requires satisfying a universal critic that inspects realizations of individual compressed images, or batches thereof. We characterize the optimal rate-distortion-perception trade-off under such a realism constraint, and show that it is asymptotically achievable without any common randomness, unless the batch size is impractically large.

## 1 Introduction

Realism, or perceptual quality, of reconstructed signals is a long-standing open challenge in lossy compression, particularly for image/video compression (Eckert & Bradley, 1998; Wu et al., 2012). It has received renewed interest in the recent years due to the remarkable progress in image generation models and neural compression techniques. The idea is that reconstructed images should be indistinguishable to humans from naturally occurring ones in addition to having a high pixel-level fidelity to the original source. This ensures that reconstructed images are free of obvious artifacts such as blocking, blurriness, etc. The idea that the output of the decoder should resemble the source in a statistical sense is not new. Advanced Audio Coding (AAC), for instance, includes a provision to add high-frequency noise to the output so that its power spectrum resembles that of the source (Sayood, 2012). But the idea has received renewed attention with the emergence of adversarial loss functions in learned compression (Santurkar et al., 2018; Tschannen et al., 2018; Agustsson et al., 2019; Blau & Michaeli,
2019). In practice, this has proven to be a powerful method for ensuring that reconstructed images
have high perceptual quality (Agustsson et al., 2019; Mentzer et al., 2020; He et al., 2022a; Iwai et al.,
2024). Adversarial loss functions can in many cases be viewed as variational forms of statistical
divergences. Thus one can think of constraining the distribution of reconstructions to be close to that of the source according to some divergence, in addition to requiring that each reconstructed image be close to its respective source according to conventional notions of distortion. Rate-distortion theory characterizes the optimal trade-off between rate and distortion in lossy compression (Pearlman & Said, 2011; Sayood, 2012). The fundamental object in the theory is the
rate-distortion function, for a given source distribution pX :
$$\Delta\in[0,\infty)\mapsto R^{(0)}(\Delta):=\operatorname*{min}_{\begin{array}{l}{p_{Y|X\mathrm{~s.t.}}}\\ {\mathbb{E}_{p}[d(X,Y)]\leq\Delta}\end{array}}I_{p}(X;Y),$$
Ip(X; Y ), (1)
Anonymous authors Paper under double-blind review

## Abstract

where pX,Y is defined as pX · pY |X. This function has been shown to describe the optimal trade-off
between rate and distortion under a variety of assumptions. Blau & Michaeli (2019) postulated an augmented form that includes a *distribution matching* constraint, which they call the rate-distortionperception (RDP) function
$$(\Delta,\lambda)\in[0,\infty)^{2}\mapsto R^{(1)}(\Delta,\lambda):=$$
(1)(∆, λ) := min
Ip(X; Y ), (2)
$$\begin{array}{r l}{{\operatorname*{min}_{p_{Y\mid X\mathrm{~s.t.}}}}}&{{I_{p}(X;Y),}}\\ {{\mathcal{D}(p_{X,p_{Y}}){\leq}\lambda,}}\\ {{\mathbb{E}_{p}[d(X,Y)]{\leq}\Delta}}\end{array}$$
$$(1)$$

$\eqref{eq:walpha}$. 
1

# The Rate-Distortion-Perception Trade-Off With Algorithmic Realism

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 where D can be any divergence between distributions. This function has likewise been shown to describe the optimal trade-off between rate, distortion, and realism under a variety of assumptions (Theis & Wagner; Chen et al., 2022). Curiously, however, these results show that substantial amounts of high-quality common randomness are needed to meet the R(1)(·, 0) bound (Saldi et al.,
2015; Wagner, 2022; Chen et al., 2022) (see also Xu et al. (2023)). The exception is the case in which the realism constraint is imposed in a very weak form, namely that the histograms of the source and reconstruction images should be close on a per-realization basis (Chen et al., 2022). Note that common pseudorandomness, say generated from a shared seed, does not qualify as common randomness for the purposes of the above results. On the other hand, the theoretical prediction that lossy compression schemes would benefit from substantial amounts of high-quality common randomness between the encoder and decoder has not been observed in practice. To the best of our knowledge, there exist compression schemes (Agustsson et al., 2023; He et al., 2022a; Hoogeboom et al., 2023; Ghouse et al., 2023; Mentzer et al., 2020; Yang & Mandt, 2023), considered as state-of-the-art, that do not involve any common randomness.

While it is possible that future designs will find common randomness to be a valuable resource, it seems more likely that the discrepancy between the theoretical prediction and practical experience lies with a flaw with the theoretical models. Consider a communication system for which a strong realism constraint is imposed: the distribution of the reconstructions must be close to the distribution of natural images, say, in Wasserstein or total variation distance (TVD). If the source distribution is continuous, then the code cannot be deterministic, for otherwise the reconstruction distribution would be supported on a countable set (corresponding to the set of received bit strings). Thus some amount of randomization is required to meet the constraint. The decoder can randomize its output in a way that "spreads" the point masses out to form a continuous distribution, but adding independent noise at the decoder inevitably degrades the distortion. Common randomness is useful because it allows the discrete reconstruction points to be dispersed to form a continuous distribution without less overall distortion. This is the basis for the finding that common randomness is a useful resource for compression under realism constraints (Theis & Agustsson, 2021).1 The above reasoning is evidently sensitive to the nature of the realism constraint. If we simply require that each reconstructed image appear realistic in its own right, without reference to the reconstruction ensemble, then the spreading process mentioned above is unnecessary. It follows that there would be no need for randomization. This is relevant because human observers, who are the ultimate arbiters of realism in practice, are adept at identifying unrealistic features of individual images. Yet it is difficult for human observers to distinguish between a continuous ensemble and one that is discrete with a very large support set, since doing so would require viewing (and remembering) many images. In short, human critics are very good at spotting unrealistic aspects of individual images but are expected to be poor at detecting subtle ensemble-level differences. This suggests posing the realism constraint in a way that better captures the relative strengths and weaknesses of human critics. The aforementioned strong realism constraint has also been challenged in the context of other problems, such as generative modeling (Theis, 2024). We consider a novel formulation of the lossy compression problem in which the goal is to satisfy a critic that is incredibly discriminating when viewing individual images. In fact, a reconstructed image is declared unrealistic if there exists some computable test, no matter how complex, that can distinguish it from the set of typical source images (see Definition 3.5 to follow). At the same time, we assume that the critic can glean information about the ensemble only by inspecting batches of individual samples. Under this formulation, we show that the rate-distortion-perception function R(1)(·, 0) in (2) is achievable without common randomness unless the batch size is unreasonably high—on par with the number of possible outputs of the decoder (Theorems 4.1 and 4.2). If common randomness is not needed to fool this critic, it should not be needed to fool any weaker (and more practical) critic, since the stronger critic subsumes the weaker one. This is akin to how in cryptography one might prove security guarantees assuming a very strong adversary, stronger than can be implemented in practice. The fact that the adversary cannot be practically implemented is a strength of our approach. It is notable that there exist compressors that can satisfy such discriminating critics at all. It is all the more notable that 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 such critics can be satisfied while achieving the rate-distortion-perception function R(1)(·, 0) in (2),
which is the most optimistic rate-distortion trade-off possible under the circumstances. Conversely, we show that common randomness is indeed beneficial if the batch size is extremely large, larger than would ever occur in practice (Theorem 4.4). In this regime, our realism measure reduces to a divergence and common randomness is again useful. These two results clarify that common randomness is indeed useful, consistent with theoretical predictions, but only in regimes that do not occur in practice, consistent with the current state of the experimental literature. Our results show the existence of optimal schemes which do not involve any common randomness at test time, but there may exist other optimal schemes, which rely on common randomness at test time, as well as learned schemes relying on common randomness at training time. In Section 2, we provide some background on the formalism for critics in algorithmic information theory. In Section 3, we introduce our new formalism for the RDP trade-off. In Section 4, we state our main results, namely Theorems 4.1, 4.2, and 4.4. All proofs are deferred to the appendices.

## 2 Background 2.1 Notation

We denote the set of (strictly) positive reals by R+, the set of (strictly) positive integers by N, the set of rational numbers by Q, and the Borel σ-algebra of R by B(R). The closure of a set A is denoted by cl(A). We use ≡ to denote equality of distributions, and Ip(X; Y ) to denote the mutual information between random variables X and Y with respect to joint distribution pX,Y . Logarithms are in base 2.

The total variation distance between distributions p and q on a finite set X is defined by

$$\|p-q\|_{T V}:={\frac{1}{2}}\sum_{x\in{\mathcal{X}}}|p(x)-q(x)|.$$

For any nonempty finite set X , and any distribution p on X , we denote by p
⊗∗ the function defined on
{0, 1}
∗, which is null outside of ∪n∈NX
n, and such that for every n ∈ N, the restriction of p
⊗∗ on X
n is p
⊗n. For a finite set X , the empirical distribution of a sequence x1:n∈X n is denoted P
emp X(x1:n).

Given a distribution PX1:non X
n, we denote by PˆX [X1:n] the *average marginal distribution* of random string X1:n, i.e., the distribution on X defined by:

$${\hat{P}}_{\mathcal{X}}[X_{1:n}]:={\frac{1}{n}}\sum_{t=1}^{n}P_{X_{t}}$$

## 2.2 Lossy Compression Algorithms Without Common Randomness

The performance of practical lossy compression schemes in terms of realism (or perceptual quality) is generally measured with well established metrics such as FID (Heusel et al., 2017), LPIPS (Zhang et al., 2018), PieAPP (Prashnani et al., 2018), and DISTS (Ding et al., 2022). Distortion is often measured with PSNR. According to these metrics, the following lossy compression algorithms are state-of-the-art. In particular, these schemes achieve visually pleasing reconstructions at very low compression rates. None of these algorithms make use of common randomness. The schemes in Mentzer et al. (2020), He et al. (2022a), and Agustsson et al. (2023) were obtained by training with an adversarial loss, a method inspired from generative adversarial networks (GANs). The former combines a conditional GAN with the scale hyperprior method of Ballé et al. (2018). The latter is an extension of the ELIC scheme (He et al., 2022b), which is state-of-the-art in terms of rate and distortion. The loss function of the latter was augmented, in particular, with an adversarial term and an LPIPS term. The method proposed in Agustsson et al. (2023) is inspired from He et al. (2022b) and Mentzer et al. (2020). The schemes in Yang & Mandt (2023), Ghouse et al. (2023), and Calligraphic letters such as X denote sets, except in p UJ, which denotes the uniform distribution over set J . The cardinality of a finite set X is denoted *|X |*. We denote by [a] the set {1*, ...,* ⌊a⌋} and by {0, 1}
∗the set of non-empty finite strings of 0's and 1's. Given a real number τ, we denote by ⌊τ ⌋
(resp. ⌈τ ⌉) the largest (resp. smallest) integer less (resp. greater) than or equal to τ. We use x1:n to denote a finite sequence (x1*, ..., x*n), and x
(n,b)to denote a batch {x
(k)
1:n}k∈[b] of b strings, each being of length n. We abbreviate x
(1,b) with x
(b). The length of a string x is denoted by l(x).

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

## 3 New Model For The Rate-Distortion-Perception Trade-Off 3.1 The One-Shot Setting

We consider a function d : *X ×X →* [0, ∞) called the distortion measure. A compression scheme can be randomized, and potentially leverage available common randomness J between the encoder and the decoder, as depicted in Figure 1 and formalized in the following definition. Hoogeboom et al. (2023) rely on diffusion models. The first uses a diffusion model conditioned on quantized latents. The two other schemes first train an autoencoder for rate and distortion, then train a diffusion model which improves the visual quality of the latter's output. The fact that none of these state-of-the-art algorithms make use of common randomness supports the theoretical results derived in the present paper.

## 2.3 Background On Algorithmic Information Theory

The theory of p-critics and universal critics has recently been brought to the attention of the machine vision community via Theis (2024). We refer to it for readers interested in a high-level and insightful presentation of the topic and its usefulness in diverse machine learning tasks (generative modeling, outlier detection). Relevant background on computability theory is provided in Appendix A. Throughout the paper, we assume that the source X follows a distribution pX on a finite set X , and that pX
is a computable function from X to (0, 1). We identify every element of X with a string of 0's and 1's, via an injection from X to {0, 1}
s, for some s ∈ N. For example, if X is a set of images of a
given resolution, then one can identify each image with the corresponding output from a fixed-length lossless compressor. The following definition is substantially close to Li & Vitányi (2019, Definition 4.3.8). See also in Li & Vitányi (2019, Lemma 4.3.5). Definition 2.1. Consider a finite set X , *identified with a subset of* {0, 1}
s. Let p *be a distribution on*
X such that ∀x ∈ X , p(x) > 0. A p-critic is a function δ : X → R, *such that*
$$\sum_{x\in{\mathcal{X}}}p(x)2^{\delta(x)}\leq1.$$
p(x)2δ(x) ≤ 1. (3)
$\mathbb{R}^{n}\to\mathbb{R}$, such that for every input dimensions $\sum_{i}\delta^{n}(x_{i})\delta^{(x)}_{i}<1$._
x∈X n
p
⊗n(x)2δ(x) ≤ 1. (4)
$$({\mathfrak{I}})$$
$\left(\mathcal{A}\right)$. 
$$({\mathfrak{H}})$$
X
The notion of p
⊗∗-critic in Definition 2.1 is used to study an asymptotic regime in Section 3.2.

Note that for any probability distribution π on N, the mixture p˜ := Pn∈N
π(n)p
⊗n is a probability measure. By multiplying (4) by πn, and summing over n, we obtain

$$\sum_{\xi\in\bigcup_{n\in\mathbb{N}}\mathcal{X}^{n}}\tilde{p}(x)2^{\delta(x)}\leq1.$$

Hence, a p-critic (resp. p
⊗∗-critic) is akin to a log-likelihood ratio: given a p-critic (resp. p
⊗∗-critic)
δ, setting q : x 7→ p(x)2δ(x)(resp. q : x 7→ p˜(x)2δ(x)) gives

$\forall x\in\mathcal{X}$ s.t. $p(x)>0,\ \delta(x)=\log\left(\frac{q(x)}{p(x)}\right)$ (resp. $\log\left(\frac{q(x)}{\tilde{p}(x)}\right)$), and $\sum_{x\in\mathcal{X}}q(x)\leq1$. (6)
Links to hypothesis testing are discussed in Theis (2024), where a sample x is deemed unrealistic if the likelihood ratio is large enough. Hence, intuitively, δ(x) can be considered as a measure of realism deficiency of x. The strength of this theory lies in the existence of objects (critics, measures) having a so-called *universality property*. For the purpose of clarity, we defer such results to Appendix A, as they are only used in our proofs.

Definition 3.1. Given non-negative reals R and Rc, an (R, Rc) *code is a privately randomized* encoder and decoder couple (F, G) consisting of a conditional distribution FM|X,J from X × [2Rc]
to [2R], and a conditional distribution GY |M,J from [2R] × [2Rc] to X . Variables M and Y are called the message and reconstruction, respectively, and distribution

$$P:=p_{X}\cdot p_{[2^{R_{c}}]}^{H}\cdot F_{M|X,J}\cdot G_{Y|M,J}$$
# $A\;p^{\otimes*}$ -*critic is a function $\delta:\cup$*  . 
n → R, such that for every input dimension n ∈ N, *we have* 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

$$\left(7\right)$$
$(\mathfrak{L})$
 ∈ [2!!]
 Encoder Decoder
 ∈ [2!]

is called the distribution induced by the code. Moreover, such a code is said to be deterministic if Rc = 0 and mappings F, G *are deterministic.*
We propose a new RDP trade-off, formalized in the following two definitions. Definition 3.2. We extend d into an additive distortion measure on batches of elements of X *: for all* B∈N,

$$\forall(\mathbf{x}^{(B)},\mathbf{y}^{(B)})\in{\mathcal{X}}^{B}\times{\mathcal{X}}^{B},\quad d(\mathbf{x}$$
$${\mathfrak{x}}^{(B)},{\mathfrak{y}}^{(B)}):={\frac{1}{B}}{\sum_{k=1}^{B}}d(x^{(k)},y^{(k)}).$$

Definition 3.3. Consider a positive integer B, *and a* p
⊗B
X -critic δ. *A tuple* (R, ∆, C) is said to be
δ-achievable with algorithmic realism if there exists some Rc ∈ R≥0 *and an* (R, Rc) code such that
the distribution P *induced by the code satisfies*
$\mathbb{E}_{P\otimes B}\left[d(\mathbf{X}^{(B)},\mathbf{Y}^{(B)})\right]\leq\Delta$ and $\mathbb{E}_{P\otimes B}\left[\delta(\mathbf{Y}^{(B)})\right]\leq C$,
where X(B) denotes a batch of B i.i.d. source samples, and Y(B)the batch of corresponding
reconstructions produced by the code (with each source sample being compressed separately). If the code is deterministic, then we say that (R, ∆, C) is δ*-achievable with a deterministic code.* The main difference with the original RDP trade-off of Blau & Michaeli (2019) pertains to the realism constraint. In the latter formulation, the realism constraint is D(pX, PY ) ≤ C, where D is some divergence. Intuitively, that constraint corresponds to the special case of infinite batch size in the RDP trade-off proposed in Definition 3.3, since the discrete distributions pX and PY can be approximated arbitrarily well using a large enough number of samples. In that sense, our proposed RDP framework generalizes the original one, through involving elements of practical realism metrics, such as the number B of samples which are inspected, and a scoring function δ which is required to be approximable via an algorithm. Theorem 4.4 to follow constitutes a rigorous statement of this intuition. We provide achievable points in the sense of Definition 3.3 in Section 4.2. In the next section, we define an asymptotic notion of achievability.

## 3.2 Asymptotic Setting

In order to derive insight into the corresponding RDP trade-off, we study a special case, which is typical in the information theory literature. We consider the compression of a source distributed according to p
⊗n X , with n a large integer. More precisely, we study the RDP trade-off in asymptotic settings where both n and the batch size go to infinity. The extension of d into an *additive distortion measure* on finite sequences, and batches of finite sequences, follows from Definition 3.2. The setup is depicted in Figure 2. Given a coding scheme, each item in a batch of source samples is compressed separately, and realism is measured based on the resulting batch of reconstructions. This is formalized in the definition below.

Definition 3.4. Given R, Rc ≥ 0, and n ∈ N, a (n, R, Rc) code is a privately randomized encoder and decoder couple (F
(n), G(n)) *consisting of a mapping* F
(n)
M|X1:n,J from X
n × [2nRc] to [2nR]
and a mapping G
(n)
Y1:n|M,J *from* [2nRc] × [2nR] to X
n. Moreover, such a code is said to be fully deterministic if Rc = 0 *and both* F
(n) and G(n) are deterministic. The distribution induced by the code is

$$P^{(n)}:=p_{X}^{\otimes n}\cdot p_{[2^{n R_{c}}]}^{\mathcal{U}}\cdot F_{M|X_{1:n},J}^{(n)}\cdot G_{Y_{1:n}|M,J}^{(n)},$$

and variable Y1:n *is called the reconstruction.*

 ∈ [2
]
 Encoder 1:
() Decoder

$$(9)$$

$$(I0)$$

1:
()
 ∈ [2
]
Figure 2: The system model for the asymptotic setting. Index k ranges from 1 to the batch size. The same encoder-decoder pair is used to process each source sample in the batch.

We define asymptotic achievability as follows. See Appendix A for background on notions of computability. Definition 3.5.

A quadruplet (R, Rc, {Bn}n≥1, ∆) is said to be asymptotically achievable with algorithmic realism if for any ε > 0, *there exists a sequence of codes* {(F
(n), G(n))}n, the n-th being (n, R + *ε, R*c),
such that the sequence {P
(n)}n *of distributions induced by the codes satisfies*

$\lim\sup\mathbb{E}_{(P^{(n)})\otimes B_{n}}\left[d(\mathbf{X}^{(n,B_{n})},\mathbf{Y}^{(n,B_{n})})\right]\leq\Delta+\varepsilon$, $n\rightarrow\infty$
E(P (n))⊗Bn-d(X(n,Bn), Y(n,Bn))≤ ∆ + ε, (9)
and for any lower semi-computable p
⊗∗
X *-critic* δ,
$$\operatorname*{sup}_{n\in\mathbb{N}}\mathbb{E}_{(P^{(n)})^{\otimes B_{n}}}\left[\delta(\mathbf{Y}^{(n,B_{n})})\right]<\infty.$$
E(P (n))⊗Bn-δ(Y(n,Bn))< ∞. *(10)*
We say that (R, {Bn}n≥1, ∆) is achievable with a fully deterministic scheme if for each n, *the code*
(F
(n), G(n)) *is fully deterministic.*
270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 4 Results 4.1 Low Batch Size Regime

For any ∆ ∈ R+, let R(∆) be the infimum of rates R such that there exists Rc ∈ R≥0 such that (R, Rc, {Bn}n≥1, ∆) is asymptotically achievable with algorithmic realism. Moreover, for

$$(I I)$$

Constraint (10) is very stringent: a single compression scheme is to satisfy a performance guarantee for every lower semi-computable p
⊗∗
X -critic (i.e. every relevant one). The motivation for the specific form of (10) is firstly from the algorithmic information theory literature: it is stated in Li & Vitányi
(2019, p.140) that a sample from a large set, identified to a long string of 0's and 1's of some length k, is realistic if its realism deficiency is small compared to k. The constraint in (10) is at least as stringent, since in our asymptotic setting, each x1:n ∈ X n is identified with a string of length linear in n, while we require the realism deficiency to be bounded. Moreover, consider the following simple example. Assume X = {0, 1}, and pX is a Bernoulli distribution B(ρ). Consider the 0-1 distortion
(also called Hamming distortion), and some distortion level ∆ < min(ρ, 1 − ρ). Then, for large enough n, the classical rate-distortion optimal code appearing in the information theory literature produces reconstructions having a frequency of 1's of roughly (ρ − ∆)/(1 − 2∆) (Cover & Thomas, 2006, Sections 10.3.1 and 10.5), i.e. different from ρ (if ρ ̸= 1/2 and ∆ > 0). Then, for the p
⊗∗
X -critic appearing in Appendix G (Claim G.1), which involves the frequency of occurrence of a pattern, the expected score diverges as n goes to infinity. Hence, the constraint in (10) is not satisfied by such a code, optimized only for rate and distortion, but not for realism. This concludes the definitions for our setup. In the next sections, we present our results, in the one-shot setting and in asymptotic settings.

The following theorem states that R(1)(·, 0), defined in (2), which naturaly arises in the distribution
matching formalism, also characterizes the optimal trade-off in our asymptotic setting, when the
batch size is not impractically large.
Theorem 4.1. Consider a sequence {Bn}n≥1 *of positive integers such that*
$$\log(B_{n})/n\underset{n\to\infty}{\longrightarrow}0.$$
0. (11)
324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

## 4.2 One-Shot Achievable Points

For theoretical interest, we provide a family of points which are achievable, in the sense of Definition 3.3, without any statistical assumption on the source distribution pX. For the sake of gleaning intuition, one can consider the following example.

- X is a finite set of images, e.g. the set of all images of a given resolution, with a finite range for pixels (finite precision).

- d is the mean squared error between pixel values.

- B is the number of images inspected by the critic at a time.

- R1 is the number of bits into which a given image is compressed.

Theorem 4.2. Consider a finite set X such that |X | ≥ 2, a computable distribution pX on X *such* that ∀x ∈ X , pX(x)>0, a positive integer B, some R > *log(*B)/ log(X ), some ∆ ∈ R+, *and a* p
⊗B
X -critic δ. Consider any conditional transition kernel pY |X from X to X *satisfying* pY ≡ pX, Ep[d(X, Y )] ≤ ∆. (13)
Then, for any ε ∈ (0, ∆/2), and any γ > 0, the triplet (R1, ∆1, C1) is δ*-achievable, with a* (R1, 0)
code, where

$$R_{1}:=R\log(|\mathcal{X}|)$$ $$\Delta_{1}:=\Delta+\varepsilon+\frac{6\Delta}{\varepsilon}\max(d)\cdot\eta_{R,\gamma}$$ $$C_{1}:=\frac{3\Delta}{\varepsilon}\bigg{[}\frac{B^{2}}{|2^{R_{1}}|}+2B\eta_{R,\gamma}\bigg{]}\cdot\max_{x}B\log\frac{1}{p_{\mathcal{X}}(x)}$$ $$\eta_{R,\gamma}:=p(\mathcal{A}_{R,\gamma})+2^{-\gamma\log(|\mathcal{X}|)/2}$$ $$\mathcal{A}_{R,\gamma}:=\Big{\{}(x,y)\in\mathcal{X}^{2}\mid\log\Big{(}\frac{p_{\mathcal{X}}(x,y)}{p_{\mathcal{X}}(x)p_{\mathcal{Y}}(y)}\Big{)}-\log(|2^{R_{1}}|)>-\gamma\log(|\mathcal{X}|)\Big{\}},$$
$$p_{Y}\equiv p_{X},\quad\mathbb{E}_{p}[d(X,Y)]\leq\Delta.$$
$$(I{\mathcal{J}})$$

$$(I4)$$
$$(I S)$$
$$(I6)$$
$$(I7)$$
$$(I8)$$
o, *(18)*
The proof is provided in Appendix B. The term B2/⌊2 R1 ⌋ is an upper bound on the probability that two source samples in the batch are compressed into the same message. This is related to the so-called *birthday paradox* (see Appendix I). The term maxx B log(1/pX(x)) is an upper bound on the output of δ, which follows from Definition 2.1. Theorem 4.2 provides insights on the asymptotic regime of Theorem 4.1. Consider the limit of large |X |, with fixed R, ∆*, ε, γ,* and with log(B) = o(log *|X |*). We know that

$$\mathbb{E}_{p}\Big[\log\Big({\frac{p_{X,Y}(x,y)}{p_{X}(x)p_{Y}(y)}}\Big)\Big]=I_{p}(X;Y).$$

Hence, if this log-likelihood ratio concentrates well, and if R1 > Ip(X; Y ), as in the definition of R(1)(·, 0) in (2), then p(AR,γ) is small for small enough γ. In such an asymptotic regime, we obtain any ∆ ∈ R+, let R∗(∆) be the infimum of rates R such that (R, {Bn}n≥1, ∆) is asymptotically achievable with algorithmic realism with fully deterministic codes. Then, we have

$$\Delta,0).\qquad(I2)$$
$\forall\Delta\in\mathbb{R}_{+}$ s.t. $R^{(1)}(\Delta,0)<H_{p}(X),$ we have $R(\Delta)=R_{*}(\Delta)=R^{(1)}(\Delta,0)$.  
The proof is provided in Appendices C and D. The strength of this result lies in how stringent constraint (10) is: a single compression scheme satisfies a performance guarantee for every relevant p
⊗∗
X -critic, and deterministic schemes are sufficient. Moreover, one can find such a scheme for any batch size sequence which is sub-exponential in the dimension n of the source, i.e. for all regimes where the batch size is not impractically large. To prove the achievability direction of Theorem 4.1, we leverage the existence of a *universal* p
⊗∗
X -critic δ0 (see Appendix A.2), which is one of the great successes of algorithmic information theory. Indeed, it is sufficient to construct a scheme which achieves (10) only for such a δ0, which is more sensitive than all relevant p
⊗∗
X -critics. It is a very strong critic, stronger than can be implemented in practice, which is another strength of Theorem 4.1.

$$(19)$$

with the convention 0/0 := 1.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

## 5 Discussion

∆1 ≈ ∆, and C1 = O(1). Therefore, the assumption in Theorem 4.1, that the source is of the form p
⊗n X for some large n, is only used to ensure fast concentration of the log-likelihood ratio. Hence, Theorem 4.1 can be extended to a larger set of sources. In the next section, we present our last main result, which pertains to an asymptotic regime with large batch size.

## 4.3 Generalizing The Distribution Matching Formalism

In this section, we present a result which connects our proposed formalism for the RDP trade-off to the distribution matching formalism of Blau & Michaeli (2019), and concludes our findings regarding the role of common randomness.

## 4.3.1 Background

Under the distribution matching formalism for the RDP trade-off, the natural asymptotic notion of achievability is as follows. Definition 4.3. (Saldi et al., 2015; Blau & Michaeli, 2019)
A quadruplet (R, Rc, {Bn}n≥1, ∆) *is said to be asymptotically achievable with near-perfect realism* if for any ε > 0, *there exists a sequence of codes* {(F
(n), G(n))}n, the n-th being (n, R + *ε, R*c),
such that the sequence {P
(n)}n *of distributions induced by the codes satisfies*

$\lim\sup\mathbb{E}_{P(n)}\left[d(X_{1:n},Y_{1:n})\right]\leq\Delta+\varepsilon$, $n\to\infty$
The TVD in (20) is directly related to the performance of the optimal hypothesis tester between the reconstruction distribution P
(n)
Y1:n
, and the source distribution p
⊗n X (Blau & Michaeli, 2019).

Replacing (20) with

$$\exists N\in\mathbb{N},\forall n\geq N,\;P_{Y_{1:n}}^{(n)}\equiv p_{X}^{\otimes n}$$
$\eqref{eq:walpha}$
X (21)
gives the notion of asymptotic *achievability with perfect realism*. It was shown that these two notions are equivalent for finite-valued sources (Saldi et al., 2015), as well as for continuous sources under mild assumptions (Saldi et al., 2015; Wagner, 2022).

## 4.3.2 Connection To Our Formalism

As stated in the theorem below, in a certain large batch size regime, asymptotic achievability with algorithmic realism (Definition 3.5) is equivalent to asymptotic achievability with near-perfect realism (Definition 4.3). The proof is provided in Appendix E. Theorem 4.4. Consider a computable increasing sequence {Bn}n≥1 *of positive integers such that*
$${\frac{B_{n}}{|{\mathcal{X}}|^{n}}}\to\infty.$$
→ ∞. (22)
Theorem 4.1 states that common randomness does not improve the trade-off under our formalism, in all regimes where the batch size is not impractically large with respect to the dimension n of the

$$(22)$$
$\mathbf{a}$

$$\|P_{Y_{1:n}}^{(n)}-p_{X}^{\otimes n}\|_{T V}\,\underset{n\to\infty}{\longrightarrow}\,0.$$
$$(20)$$
0. *(20)*
Then, for any Rc ∈ R≥0, *and any* (R, ∆) ∈ (R+)
2, tuple (R, Rc, {Bn}n≥1, ∆) is asymptotically achievable with algorithmic realism if and only if (R, Rc, ∆) is asymptotically achievable with near-perfect realism, if and only if (R, Rc, ∆) *is asymptotically achievable with perfect realism.* Hence, Theorem 4.4, similarly to the finding in Theis (2024), shows that for large batch size, our formalism is equivalent to the distribution matching formalism. Hence, the former is a generalization of the latter. Moreover, Theorem 4.4 and prior work on the distribution matching formalism (Saldi et al., 2015; Wagner, 2022; Chen et al., 2022) imply that common randomness is useful when the size of the batch inspected by the critic is extremely large.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 source. Theorem 4.4 states that common randomness is useful - consistent with prior theoretical predictions - when the batch size is extremely large. Thus, Theorems 4.1 and 4.4 indicate that, in order to understand the role of randomization in lossy compression with realism constraints, the focus should be shifted to the size of the batch inspected by the critic. A continuation of our work could be to investigate realism metrics, where particular attention would be given to the choice of the batch size. This could lead to highlighting specific strengths and weaknesses of existing realism metrics. It may also inspire a critical assessment of the relative performance of existing compression schemes, depending on the choice of realism metric. Another continuation could be to more precisely characterize the amount of randomness needed as a function of the batch size. Furthermore, possible extensions of our setup include compression with side information, and other distributed settings.

## References

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 E. Agustsson, M. Tschannen, F. Mentzer, R. Timofte, and L. Van Gool. Generative Adversarial Networks for Extreme Learned Image Compression. In IEEE/CVF International Conference on Computer Vision, 2019.

E. Agustsson, D. Minnen, G. Toderici, and F. Mentzer. Multi-Realism Image Compression with a Conditional Generator. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*
(CVPR), 2023.

Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, and Nick Johnston. Variational image compression with a scale hyperprior. In *International Conference on Learning Representations*, 2018.

Y. Blau and T. Michaeli. Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff.

In *36th International Conference on Machine Learning*, 2019.

C. L. Canonne. A short note on learning discrete distributions, 2020. arxiv:2002.11457. J. Chen, L. Yu, J. Wang, W. Shi, Y. Ge, and W. Tong. On the Rate-Distortion-Perception Function.

IEEE Journal on Selected Areas in Information Theory, 3(4), 2022. ISSN 2641-8770. doi:
10.1109/JSAIT.2022.3231820.

T.M. Cover and J.A. Thomas. *Elements of Information Theory*. Wiley-Interscience. Wiley, 2006.

ISBN 9780471748816.

P. Cuff. Distributed Channel Synthesis. *IEEE Transactions on Information Theory*, 59(11), 2013.

ISSN 1557-9654. doi: 10.1109/TIT.2013.2279330.

Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli. Image Quality Assessment: Unifying Structure and Texture Similarity. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2022.

M. P. Eckert and A. P. Bradley. Perceptual quality metrics applied to still image compression. Signal Processing, 70(3):177–200, 1998.

N. F. Ghouse, J. Petersen, A. Wiggers, T. Xu, and G. Sautière. A Residual Diffusion Model for High Perceptual Quality Codec Augmentation, 2023. arxiv:2301.05489.

Y. Hamdi, A. B. Wagner, and D. Gündüz. The Rate-Distortion-Perception Trade-off: the Role of Private Randomness. In *IEEE International Symposium on Information Theory (ISIT)*, 2024.

D. He, Z. Yang, H. Yu, T. Xu, J. Luo, Y. Chen, C. Gao, X. Shi, H. Qin, and Y. Wang. PO-ELIC:
Perception-Oriented Efficient Learned Image Coding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2022a.

Dailan He, Ziming Yang, Weikun Peng, Rui Ma, Hongwei Qin, and Yan Wang. ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022b.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter.

GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In Advances in Neural Information Processing Systems, 2017.

E. Hoogeboom, E. Agustsson, F. Mentzer, L. Versari, G. Toderici, and L. Theis. High-fidelity image compression with score-based generative models, 2023. arXiv:2305.18231.

S. Iwai, T. Miyazaki, and S. Omachi. Controlling rate, distortion, and realism: Towards a single comprehensive neural image compression model. In IEEE/CVF Winter Conference on Applications of Computer Vision, 2024.

M. Li and P. Vitányi. *An Introduction to Kolmogorov Complexity and Its Applications*. Texts in Computer Science. Springer International Publishing, Cham, 4th edition, 2019. ISBN 9783030112981. doi: 10.1007/978-3-030-11298-1.

F. Mentzer, G. D. Toderici, M. Tschannen, and E. Agustsson. High-Fidelity Generative Image Compression. *Advances in Neural Information Processing Systems*, 33, 2020.

W. A. Pearlman and A. Said. *Digital Signal Compression: Principles and Practice*. Cambridge University Press, Cambridge (England), 2011.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Ekta Prashnani, Hong Cai, Yasamin Mostofi, and Pradeep Sen. PieAPP: Perceptual Image-Error Assessment Through Pairwise Preference. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.

N. Saldi, T. Linder, and S. Yüksel. Output Constrained Lossy Source Coding With Limited Common Randomness. *IEEE Transactions on Information Theory*, 61(9), 2015. doi: 10.1109/TIT.2015. 2450721.

S. Santurkar, D. Budden, and N. Shavit. Generative Compression. In *Picture Coding Symposium*,
2018.

K. Sayood. *Introduction to Data Compression*. Morgan Kaufmann, Waltham, MA (United States of America), 4th edition, 2012. ISBN 9780124160002.

L. Theis. Position: What makes an image realistic? In Forty-first International Conference on Machine Learning, 2024.

L. Theis and E. Agustsson. On the advantages of stochastic encoders. In *Neural Compression:*
From Information Theory to Applications - workshop at the International Conference on Learning Representations, 2021.

L. Theis and A. B. Wagner. A coding theorem for the rate-distortion-perception function. In Neural Compression: From Information Theory to Applications - workshop at the International Conference on Learning Representations 2021.

M. Tschannen, E. Agustsson, and M. Lucic. Deep Generative Models for Distribution-Preserving Lossy Compression. In *NeurIPS*, 2018.

A. B. Wagner. The Rate-Distortion-Perception Tradeoff: The Role of Common Randomness, 2022.

arXiv:2202.04147.

H. R. Wu, W. Lin, and L. J. Karam. An overview of perceptual processing for digital pictures. In IEEE International Conference on Multimedia and Expo Workshops, 2012.

T. Xu, Q. Zhang, Y. Li, D. He, Z. Wang, Y. Wang, H. Qin, Y. Wang, J. Liu, and Y. Zhang. Conditional perceptual quality preserving image compression, 2023. arXiv:2308.08154.

R. Yang and S. Mandt. Lossy Image Compression with Conditional Diffusion Models. In *NeurIPS*,
2023.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In *IEEE/CVF Conference on Computer* Vision and Pattern Recognition, 2018.

## A Further Background On Algorithmic Information Theory A.1 Computability

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 This definition matches Li & Vitányi (2019, Definition 1.7.4), except for the definition of a computable real number, which we adapted from Li & Vitányi (2019, Exercise 1.7.22), and for the definition of a computable set, which matches that of Li & Vitányi (2019, page 32).

Definition A.1. Consider a subset E of N≥0. A map f from E *into* N
3≥0 is said to be computable if it corresponds to a Turing machine (Li & Vitányi, 2019, Section 1.7.1). This notion extends to functions having as domain other common countable sets, such as N
k≥0 for k ∈ N, and {0, 1}
∗, or any subset thereof, by identifying elements of these sets with non-negative integers via some reference bijections.

Consider a computable map f from a subset E of N≥0 into {0, 1} × N≥0 × N. *Then, composing* f with (s, a, b) 7→ (2s − 1)a/b yields a map from E to Q, *which is said to be a computable map from* E to Q. A map f from a subset E of N≥0 into R is said to be lower semi-computable if there exists a computable function φ from E × N into Q, *such that*

$$\forall x\in{\mathcal{E}},\varphi(x,k)\underset{k\to\infty}{\to}f(x),$$
$$T k\in\mathbb{N},\ \ \varphi(x,k+1)$$

f(x), and ∀x ∈ E, ∀k ∈ N, φ(*x, k* + 1) ≥ φ(*x, k*).

Moreover, f is said to be a computable map from E to R if both f and −f are lower semi-computable.

A real number λ is said to be computable if the constant function f : N≥0 → R, n 7→ λ is a computable function from N≥0 to R. A (possibly infinite) subset X of N≥0, is said to be computable if there exists a computable function f from N≥0 to {0, 1}, which returns 1 if its input is in X , and 0 otherwise. The following lemma allows to construct (semi-)computable functions. Its proof is deferred to Appendix K.

Lemma A.2. Let E denote a non-empty subset of N≥0, and let f and g *denote functions from* E to R.

(i) If f and g are both lower semi-computable, then functions f + g, ⌈f⌉, and 2 f are lower semicomputable. If, in addition, f and g only take non-negative values, then fg and 2 f /(3+f)
2 are lower semi-computable. If, in addition, f *only takes positive values, then* log(f) is lower semi-computable.

(ii) If f and g are both computable, then functions f + g, fg, and |f| *are computable. If, in addition,*
f only takes positive values, then functions 1/f, and f 1/b are computable, for any positive integer b.

(iii) Let X *be a computable finite subset of* {0, 1}
∗. If f *is a lower semi-computable function from*
{0, 1}
∗into R, *then the function* ˜f : {0, 1}
∗ → R *which is null outside of* ∪n∈NX
n, *and is defined by*

$$\forall x\in\cup_{n\in\mathbb{N}}{\mathcal{X}}^{n},\ \ {\tilde{f}}(x)=\sum_{y\in{\mathcal{X}}^{l(x)}}f(y),$$

is lower semi-computable. Moreover, if p is a lower semi-computable probability measure on X , *then* p
⊗∗ *is lower semi-computable.*
A.2 UNIVERSAL CRITICS AND SEMI-MEASURES Definition A.3. Given a finite set W, a function f : W → [0, 1] *is a semi-measure if*

$$\sum_{w\in{\mathcal{W}}}f(w)\leq1.$$

It is said to be a lower semi-computable semi-measure if f is a semi-measure and f is lower semi-computable.

The following theorem, corresponds to Definition 4.3.2, Equation (4.2), and Theorems 4.3.1 and 4.3.3 in Li & Vitányi (2019). It introduces the notion of *universal* p
⊗∗*-critic*, used in Theis (2024).

The mixture m therein can be used as a prior distribution, which has been shown to be relevant in machine learning applications involving realism, such as outlier detection and generative modeling (Theis, 2024). Theorem A.4. Consider a finite set X , *each element of which is identified with a string in* {0, 1}
s, for some s ∈ N. Let p be a computable distribution on X such that ∀x ∈ X , p(x) > 0. *Then, there* 648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701

Since our definitions are slightly different from the classical ones, we provide a proof of Theorem A.4 in Appendix J. Such a critic δ0 is one of the best measures of realism deficiency according to p,
in the limit of arbitrarily long strings. If a critic δ identifies a certain amount of deficiency in a given string, then δ0 will identify at least as much deficiency, up to some additive constant. Intuitively, δ0 is sensitive to all properties of randomness according to p. The existence of such a δ0 constitutes a remarkable property of the set of all lower semi-computable p
⊗∗
X -critics (which is infinite).
Remark A.5. (Li & Vitányi, 2019, Theorem 4.3.3) The universal semi-measure m can be chosen in such a way that
∀x ∈ {0, 1} ∗, | − log(m(x)) − K(x)| ≤ c, (24)

## For Some Constant C, Where K *Is The Kolmogorov Complexity (Li & Vitányi, 2019, Section 3.1). Property* (24) Constitutes A Strong Result, Since The Kolmogorov Complexity Is Only Defined Up To A Constant
-We Omit The Corresponding Details, For The Purpose Of Clarity. The Map X 7→ Log(1/P(X)) − K(X) Is
Sometimes Considered To Be An Approximation Of A Universal P
⊗∗ Critic, See, E.G., Theis (2024), And
Appendix J. B Proof Of Theorem 4.2 B.1 Outline

To show the achievability of a tuple (R1, ∆1, C1), it is not necessary to construct an explicit compression scheme: it is sufficient to prove the abstract existence of such a scheme. To that end, we consider a set of random reconstructions, and study its realism properties in Section B.2. Then, we show the existence of a suitable choice of realizations of the latter reconstructions in Section B.3. In Section B.4, we prove Theorem 4.2 by proposing a compression scheme achieving a close-to-uniform sampling from the set of reconstructions. For the remainder of Section B, we fix a finite set X such that *|X | ≥* 2, a computable distribution pX on X such that ∀x ∈ X , pX(x)>0, a positive integer B, and a p
⊗B
X -critic δ.

## B.2 Realism Performance Of A Uniformly Sampled Batch Of Random Reconstructions B.2.1 Random Candidate Reconstructions

Given a positive real R1, let C be a family of ⌊2
R1 ⌋ i.i.d. variables, each sampled from pX. The m-th
variable is denoted y(C, m). We denote their joint distribution by QC. Given a realization c of C, we consider a batch y
(B) of B elements of c, sampled uniformly with replacement. Then, we compute
the batch's realism score δ(y
(B)). This is formalized in the following lemma, which gives an upper
bound of the expected score with respect to QC. Lemma B.1. *Consider a positive real* R1 ∈ (log(B), ∞), *and the following pmf.*
$$Q_{\mathcal{C},\mathbf{M}^{(B)},\mathbf{Y}^{(B)}}\;(\{y(m^{\prime})\}_{m^{\prime}\in[[\,2^{R_{1}}\,]\,]},\mathbf{m}^{(B)},\mathbf{y}^{(B)})$$
(B)
$$\mathbf{1}=\left(\prod_{m^{\prime}=1}^{\lfloor2^{R_{1}}\rfloor}p_{X}(y(m^{\prime}))\right)\cdot{\frac{1}{\lfloor2^{R_{1}}\rfloor^{B}}}\cdot\prod_{k=1}^{B}\mathbf{1}_{y^{(k)}=y(m^{(k)})}.$$
$\textsf{IPLED}$ $\textsf{BATCH}$ OF RANDOM
$$(25)$$
$$(26)$$
$$\mathbb{E}_{Q}[\delta(\mathbf{Y}^{(B)})]\leq{\frac{B^{2}}{\lfloor2^{R_{1}}\rfloor}}\operatorname*{max}_{x}B\log{\frac{1}{p_{X}(x)}}.$$
The remainder of Section B.2 is dedicated to the proof of Lemma B.1. Fix R1 > log(B).

exists a p
⊗∗-critic δ0 (which is not necessarily lower semi-computable), such that for any lower semi-computable p
⊗∗-critic δ, there exists a constant cδ *such that*

$$\forall x\in\bigcup_{n\in\mathbb{N}}{\mathcal{X}}^{n},\quad\delta_{0}(x)\geq\delta(x)-c_{\delta}.$$
$$(23)$$

$$(24)$$
n, δ0(x) ≥ δ(x) − cδ. *(23)*
$\square$ is a subset of $\mathbb{R}^n$. 
Any p
⊗∗*-critic satisfying* (23) *is called a universal* p

$$r s a l\;p^{\otimes*}\text{-}c r i t i c.$$

Then, we have B.2.2 REALISM PERFORMANCE
Claim B.2. Since R1 > log(B), *a simple bound yields,*
Therefore, we have 702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 Proof. By setting q : x 7→ p(x) · 2 δ(x), we can write

$$\mathbf{\tau}\mapsto p(x)\cdot z$$
$$\forall x\in{\mathcal{X}}\;\mathrm{s.t.}\;p(x)>0,\;\delta(x)=\log{\Big(}{\frac{q(x)}{p(x)}}{\Big)},{\mathrm{~with~}}\;0<\sum_{x\in{\mathcal{X}}}q(x)\leq1.$$

We denote the latter sum by q(X ). Then, q/q(X ) is a probability distribution on X , and we have This concludes the proof of Lemma B.1. B.3 FURTHER PROPERTIES OF A UNIFORMLY SAMPLED BATCH
Proposition B.4. Consider a finite set X such that |X | ≥ 2, a distribution pX on X *such that*
∀x ∈ X , pX(x)>0, a positive integer B, some R > log(B)/ log(|X |), some ∆ ∈ R+, *and a* p
⊗B
X -critic δ. Consider any conditional transition kernel pY |X from X to X *satisfying* pY ≡ pX, Ep[d(X, Y )] ≤ ∆. (32)

$$p_{Y}\equiv p_{X},\quad\mathbb{E}_{p}[d(X,Y)]\leq\Delta.$$
$$(32)$$
EQ[δ({y(C, M(k))}k∈[B])] =X m(B) EQ[1M(B)=m(B) δ({y(C, m(k))}k∈[B])] =X m(B) EQ[1M(B)=m(B) ]EQ[δ({y(C, m(k))}k∈[B])] =X {m(k)}k∈[B] 2 by 2 ̸= (p U [⌊2R1 ⌋] ) ⊗B(M(B)=m(B))Ep ⊗B X [δ(X(B))] +X {m(k)}k∈[B] not 2 by 2 ̸= (p U [⌊2R1 ⌋] ) ⊗B(M(B)=m(B))EQ[δ({y(C, m(k))}k∈[B])] ≤ Ep ⊗B X [δ(X(B))] + max(δ)(p U [⌊2R1 ⌋] ) ⊗B(M(1), ..., M(B)not 2 by 2 ̸=) ≤ Ep ⊗B X [δ(X(B))] + B2 ⌊2R1 ⌋ max x B log 1 pX(x) , (28)
where (28) follows from Claim B.2 and (3). Claim B.3. For any distribution p on a finite set, any p-critic δ *satisfies*

$$(28)$$
$$(29)$$
$$\mathbb{E}_{p}[\delta(X)]\leq0.$$
$$(30)^{\frac{1}{2}}$$
$$(31)$$
Ep[δ(X)] ≤ 0. (29)
$$\mathbb{E}_{p}[\delta(X)]\leq\mathbb{E}_{p}\Big[\log\Big(\frac{q(X)/q(\mathcal{X})}{p(X)}\Big)\mathbf{1}_{p(X)>0}\Big]=-K L(p||q/q(\mathcal{X}))\leq0.$$
$$\begin{array}{c}{{Q\Big(\Big\{\delta_{0}\big(\big\{y(\mathcal{C},M^{(k)})\big\}_{k\in[B]}\big)\in\mathcal{E}\Big\}}}\\ {{\Big[\Big\{M^{(1)},...,M^{(B)}\ 2\ \mathrm{by\ 2\ distinct}\Big\}\Big)}}\\ {{=p_{X}^{\otimes B}\Big(\delta_{0}\big(\mathbf{X}^{(B)}\big)\in\mathcal{E}\Big).}}\end{array}$$
(27) $$\frac{1}{2}$$ . 
X(B)∈ E. (27)
See Appendix I for a proof. From the definition (Section B.2.1) of Q, for any E ∈ B(R),

$(p_{[[2^{R_{1}}]]}^{H})^{\otimes B}(M^{(1)},...,M^{(B)}$ 2 by 2 distinct) $\geq1-\frac{B^{2}}{\left[2^{R_{1}}\right]}$.  
.
756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 We choose τ = −γ log *|X |*. Then, Aτ = AR,γ, with the notation of Proposition B.4. Hence, from
(41) and Markov's inequality, we have

$$Q_{\mathcal{C}}\Big(\|Q_{X^{(1)}|\mathcal{C}}-p_{X}\|_{T V}\geq\frac{3\Delta}{\varepsilon}[p(\mathcal{A}_{R,\gamma})+2^{-\gamma\log(|X|)/2}]\Big)\leq\frac{\varepsilon}{3\Delta}.$$
3∆. (43)
By construction, we have QY(B),X(B) ≡ p
⊗B
Y,X. Therefore, from (38), and the additivity of d, we have
EQ[d(X(B), Y(B))] ≤ ∆. (44)
Therefore, from Markov's inequality,
where we have used the fact that ε ∈ (0, ∆/2). From a union bound and (39), (43), and (45) there exists a realization c∗ of C such that none of the corresponding events hold. Since, by construction, this concludes the proof of Proposition B.4.

$$Q_{\mathbf{M}^{(B)},\mathbf{Y}^{(B)},\mathbf{X}^{(B)}|{\mathcal{C}}=c*}\equiv Q_{M^{(1)},Y^{(1)},X^{(1)}}^{\otimes B}|{\mathcal{C}}=c*},$$
$$\|Q_{X}-p_{X}\|_{TV}\leq\frac{3\Delta}{\varepsilon}[p({\cal A}_{R,\gamma})+2^{-\gamma\log(|{\cal X}|)/2}]$$ $$\mathbb{E}_{Q^{\otimes B}}[d({\bf X}^{(B)},{\bf Y}^{(B)})]\leq\Delta+\varepsilon$$ $$\mathbb{E}_{Q^{\otimes B}}[\delta({\bf Y}^{(B)})]\leq\frac{3\Delta}{\varepsilon}\cdot\frac{B^{2}}{[2^{R_{1}}]}\max_{x}B\log\frac{1}{p_{X}(x)},$$

$$(33)$$
$$(34)$$
$$(35)$$
$$(36)^{\frac{1}{2}}$$

where R1 = R log |X|, and

$${\mathcal{A}}_{R,\gamma}:=\Big\{(x,y)\in{\mathcal{X}}^{2}\mid\,\log\Big({\frac{p_{X,Y}(x,y)}{p_{X}(x)p_{Y}(y)}}\Big)-\log(\lfloor2^{R_{1}}\rfloor)>-\gamma\log(\vert{\mathcal{X}}\vert)\Big\}.$$
o. (37)
Proof. Fix some R > log(B)/ log(*|X |*), some ∆ > 0, some ε ∈ (0, ∆/2), some γ > 0, and a conditional transition kernel pY |X from X to X satisfying

$$p_{Y}\equiv p_{X},\quad\mathbb{E}_{p}[d(X,Y)]\leq\Delta.$$

Define R1 = R log *|X |*. We apply Lemma B.1, and use the notation therein. Then, from Markov's inequality, we have

$$Q c\left(\mathbb{E}_{Q}[\delta(\mathbf{Y}^{(B)})|{\mathcal{C}}]\geq{\frac{3\Delta}{\varepsilon}}{\frac{B^{2}}{\lfloor2^{R_{1}}\rfloor}}\operatorname*{max}_{x}B\log{\frac{1}{p_{X}(x)}}\right)\leq{\frac{\varepsilon}{3\Delta}}.$$
We extend distribution Q as follows.
$Q_{\mathcal{C},\mathbf{M}^{(B)},\mathbf{Y}^{(B)},\mathbf{X}^{(B)}}\left(\{y(m^{\prime})\}_{m^{\prime}\in[[2^{R_{1}}]]},\mathbf{m}^{(B)},\mathbf{y}^{(B)},\mathbf{x}^{(B)}\right)$
$${\mathfrak{p}}|{\mathcal{C}}-p_{X}\|_{2}$$
$${\bf\Phi}^{(i)},{\bf y}^{(B)},{\bf x}^{(B)}):=$$
$Q_{\mathcal{C},\mathbf{M}^{(B)},\mathbf{Y}^{(B)}}\left(\{y(m^{\prime})\}_{m^{\prime}\in[\lfloor2^{n_{1}}\rfloor]},\mathbf{m}^{(B)},\mathbf{y}^{(B)}\right)\cdot\prod_{k=1}^{B}p_{X\mid Y=y(m^{(k)})}(x^{(k)}).$
$$({\mathfrak{I}}{\mathfrak{I}})$$
$$(39)$$
$$(40)$$
$$(41)$$
$$(43)$$

Distribution QC,M(1),Y (1),X(1) corresponds to the setting of Cuff (2013, Theorem VII.1), known as
the soft covering lemma. Since pY ≡ pX, the latter lemma yields that for any τ ∈ R,
EC-∥QX(1)|C − pX∥T V ≤ p(Aτ ) + 2τ/2, (41)
where
$\mathcal{A}_{\tau}:=\{(x,y)\mid\log(p_{Y|X=x}(y)/p_{X}(y))-\log(\lfloor2^{R_{1}}\rfloor)>\tau\}$.  
R1 ⌋) > τ}. (42)
Then, for any ε ∈ (0, ∆/2), and any γ > 0, there exists a family {y(m)}m∈[⌊2R1 ⌋], denoted c, of elements of X , *such that distribution*

$$Q_{M,Y,X}\left(m,y,x\right):=\ {\frac{1}{\lfloor2^{R_{1}}\rfloor}}\cdot\left(\mathbf{1}_{y=y(m)}\right)\cdot p_{X|Y=y(m)}(x)$$
· pX|Y =y(m)(x) (33)
$$Q_{\mathcal{C}}\Big{(}\mathbb{E}_{Q}[d(\mathbf{X}^{(B)},\mathbf{Y}^{(B)})|\mathcal{C}]\geq\Delta+\varepsilon\Big{)}\leq\frac{\Delta}{\Delta+\varepsilon}=1-\frac{\varepsilon}{\Delta}\cdot\frac{1}{1+\varepsilon/\Delta}<1-\frac{2\varepsilon}{3\Delta},\tag{45}$$

satisfies B.4 PROOF OF THEOREM 4.2 B.4.1 COMPRESSION SCHEME ACHIEVING CLOSE-TO-UNIFORM SAMPLING
810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 Since d is additive, we have E(P )⊗B [d(X(B), Y(B))] = EP [d(*X, Y* )] and

$$(46)^{\frac{1}{2}}$$
 $P_{X,M,Y}:=p_X\cdot Q_{M,Y|X}.$  - $P$ satisfies Markov chain $Y=M$. 
$$(48)$$
$$\mathbb{E}_{(Q)^{\otimes B}}[d(\mathbf{X}^{(B)},\mathbf{Y}^{(B)})]=\mathbb{E}_{Q}[d(X,Y)].$$
$$(49)^{\frac{1}{2}}$$

Since d is bounded, then we can apply Lemma H.3 (Appendix H). Then, from (48), and Lemma H.1 with W = (*X, Y* ), we have

$$\mathbb{E}_{P^{0\otimes B}}[d(\mathbf{X}^{(B)},\mathbf{Y}^{(B)})]\leq\mathbb{E}_{Q^{\otimes B}}[d(\mathbf{X}^{(B)},\mathbf{Y}^{(B)})]+\frac{6\Delta}{\varepsilon}\max(d)[p(\mathcal{A}_{R,\gamma})+2^{-\gamma\log(|\mathcal{X}|)/2}]$$ $$\leq\Delta+\varepsilon+\frac{6\Delta}{\varepsilon}\max(d)[p(\mathcal{A}_{R,\gamma})+2^{-\gamma\log(|\mathcal{X}|)/2}],\tag{6.3}$$

where the last inequality follows from (35). Moving to the realism performance, we have the following property of the TVD - see Appendix H: Claim B.5. Given any two distributions P and Q *on the same finite alphabet, we have, for any* B ∈ N,

$$\left\|P^{\otimes B}-Q^{\otimes B}\right\|_{T V}\leq B\|P-Q\|_{T V}.$$

From Lemma H.3, Claim B.5, (48), and Lemma H.1 with W = Y(B), we have,

$$\mathbb{E}_{P^{\otimes B}}[\delta(\mathbf{Y}^{(B)})]\leq\mathbb{E}_{Q^{\otimes B}}[\delta(\mathbf{Y}^{(B)})]+\frac{6B\Delta}{\varepsilon}\max(\delta)[p(\mathcal{A}_{R,\gamma})+2^{-\gamma\log(|\mathcal{X}|)/2}]$$ $$\leq\frac{3\Delta}{\varepsilon}\cdot\frac{B^{2}}{|2^{R_{1}}|}\max_{z}B\log\frac{1}{p_{X}(x)}+\frac{6B\Delta}{\varepsilon}[p(\mathcal{A}_{R,\gamma})+2^{-\gamma\log(|\mathcal{X}|)/2}]\cdot\max_{x}B\log\frac{1}{p_{X}(x)}.\tag{50}$$  his concludes the proof.  

## C Achievability Of Theorem 4.1

Consider some ∆ ∈ R+ such that R(1)(∆, 0) < Hp(X), and a sequence {Bn}n≥1 of positive integers such that

$\log(B_{n})/n\underset{n\rightarrow\infty}{\longrightarrow}0$.  
$$p_{Y}\equiv p_{X},\;\mathbb{E}_{p}[d(X,Y)]\leq\Delta,\;R\geq I_{p}(X;Y)+\varepsilon.$$
Fix R ∈ (R(1)(∆, 0), Hp(X)), ε ∈ (0, R − R(1)(∆, 0)), and γ ∈ (0*, ε/* log(|X |)). Then, there exists pY |X such that pY ≡ pX, Ep[d(X, Y )] ≤ ∆, R ≥ Ip(X; Y ) + ε. (52)
We define the following distribution P*X,Y,M*, which differs from Q in having the correct marginal for X :
PX,M,Y := pX · *QM, Y* |X . (47)
Therefore, from (33), distribution P satisfies Markov chain X−M−Y. Hence, it defines a (R1, 0) code. From Lemma H.2 (Appendix H), comparing P with Q reduces to comparing marginals, i.e. to (34) :

$$\left\|P_{M,X,Y}-Q_{M,X,Y}\right\|_{TV}=\left\|P_{X}-Q_{X}\right\|_{TV}$$ $$=\left\|p_{X}-Q_{X}\right\|_{TV}\leq\frac{3\Delta}{\varepsilon}[p(\mathcal{A}_{R,\gamma})+2-\gamma\log(|X|)/2].$$
$${\mathrm{REM~}}4.1$$
$$(S1)$$
$$(S2)$$

Fix some R > log(B)/ log(*|X |*), some ∆ > 0, some ε ∈ (0, ∆/2), some γ > 0, and a conditional
transition kernel pY |X from X to X satisfying
$$p_{Y}\equiv p_{X},\quad\mathbb{E}_{p}[d(X,Y)]\leq\Delta.$$
Define R1 = R log *|X |*. Then, we can apply Proposition B.4. We use the notation from the latter.

864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 where

$$\mathbb{E}_{(P^{(n)})^{\oplus B_{n}}}\left[d(\mathbf{X}^{(n,B_{n})},\mathbf{Y}^{(n,B_{n})})\right]\leq\Delta+\varepsilon+\frac{6\Delta}{\varepsilon}\max(d)[p(\mathcal{A}_{R,\gamma}^{(n)})+2^{-\gamma n\log(|X|)/2}],\tag{53}$$ $$\mathbb{E}_{(P^{(n)})^{\oplus B_{n}}}\left[\delta_{0}(\mathbf{Y}^{(n,B_{n})})\right]\leq$$ $$\frac{3\Delta}{\varepsilon}\left[\frac{B_{n}^{2}}{(2^{n}H}\max_{x}nB_{n}\log\frac{1}{p_{X}(x)}+2B_{n}[p(\mathcal{A}_{R,\gamma}^{(n)})+2^{-\gamma n\log(|X|)/2}]\cdot\max_{x}nB_{n}\log\frac{1}{p_{X}(x)}\right],\tag{54}$$  where
$$\mathcal{A}_{R,\gamma}^{(n)}:=\Big{\{}(x_{1:n},y_{1:n})\in(\mathcal{X}^{n})^{2}\mid\sum_{t=1}^{n}\log\Big{(}\frac{p_{X,Y}(x_{t},y_{t})}{p_{X}(x_{t})p_{Y}(y_{t})}\Big{)}-\log([2^{nR}])>-\gamma n\log(|\mathcal{X}|)\Big{\}},\tag{55}$$

with the convention 0/0 := 1. From (52), log(⌊2 nR⌋)/n − γ log(|X |) > Ip(X; Y ) for large enough n. Then, since X is finite, we have, from Hoeffding's inequality,

$$p({\mathcal{A}}_{R,\gamma}^{(n)})=O(e^{-\kappa n}),$$
$$(56)$$
$$(58)$$
−κn), (56)
and for any lower semi-computable p
⊗∗
X -critic δ,
  **Remark 1**: **A** **and** $\alpha$**,**  $$\sup_{n\in\mathbb{N}}\mathbb{E}_{(P^{(n)})\otimes B_{n}}\left[\delta(\mathbf{Y}^{(n,B_{n})})\right]<\infty.$$

From the proof of Theorem 4.2, we know that P
(n) has a deterministic decoder. Hence, it only remains to derandomize the encoder of P
(n). We denote its decoder by m 7→ y1:n(m). The following claim is a slight modification of Hamdi et al. (2024, Proposition 4). We provide details in Section C.1. Claim C.1. *There exists a sequence of deterministic maps*

$$f^{(n)}:{\cal X}^{n}\rightarrow[2^{nR}],\quad\mbox{such that}$$  $$\|\hat{\tilde{P}}^{(n)}_{{\cal X}^{2}}[X^{n},y_{1:n}(M)]-\hat{P}^{(n)}_{{\cal X}^{2}}[X^{n},y_{1:n}(M)]\|_{TV}\stackrel{{\longrightarrow}}{{n\rightarrow\infty}}0,$$  $$\liminf_{n\rightarrow\infty}\frac{-1}{n}\log\|\hat{P}^{(n)}_{M}-P^{(n)}_{M}\|_{TV}>0,\quad\mbox{where}$$  $$\tilde{P}^{(n)}_{X^{n},M}:=p^{\otimes n}_{X}\cdot{\bf1}_{M=f^{(n)}}(X^{n})\cdot$$
$$(59)$$
$$\lim\inf_{n\to\infty}\frac{-1}{n}\log\left\|(\bar{P}^{(n)})_{M}^{\otimes B_{n}}-(P^{(n)})_{M}^{\otimes B_{n}}\right\|_{TV}>0.$$  In $\log\left(\Omega\right)$ and $\log\left(\Omega\right)$ are $\log\left(\Omega\right)$ and $\log\left(\Omega\right)$ are $\log\left(\Omega\right)$.  
Thus, from Lemma H.3 and (3), we have
0. (61)
$$|\mathbb{E}_{(\hat{P}^{(n)})^{\otimes B_{n}}}\left[\delta(\mathbf{Y}^{(n,B_{n})})\right]-\mathbb{E}_{(P^{(n)})^{\otimes B_{n}}}\left[\delta(\mathbf{Y}^{(n,B_{n})})\right]|\underset{n\rightarrow\infty}{\longrightarrow}0.$$

Moreover, since d is bounded, then from Lemma H.3, we obtain

$$R(\Delta)\leq R_{*}(\Delta)\leq R^{(1)}(\Delta,0),$$
$$(60)$$
$$(61)$$

We use the powerful result of Theorem A.4 regarding the existence of a so-called *universal critic*.

From Definition 2.1, for every n ∈ N, the restriction of δ0 to X
nBn is a p
⊗nBn X -critic. Moreover, from (51), for large enough n, we have *nR >* log(Bn). Then, for large enough n, we can apply Theorem 4.2 for set X
n, distribution p
⊗n X , transition kernel Qn t=1 pY |X, batch size Bn, critic δ0, rate nR/ log(|X n|), and constants ∆*, ε, γ.* This gives that, for every n large enough, there is a (*n, R,* 0)
code, inducing a distribution P
(n)such that

1.3 and (3), we have 
$$\left|\mathbb{E}_{(P^{(n)})^{\otimes B_{n}}}\left[d(\mathbf{X}^{(n,B_{n})},\mathbf{Y}^{(n,B_{n})})\right]-\mathbb{E}_{(P^{(n)})^{\otimes B_{n}}}\left[d(\mathbf{X}^{(n,B_{n})},\mathbf{Y}^{(n,B_{n})})\right]\right|\underset{n\rightarrow\infty}{\longrightarrow}0.$$
0. (62)
Since this analysis is valid for any ε ∈ (0, R − R(1)(∆, 0)), then tuple (R, {Bn}n≥1, ∆) is asymptotically achievable with algorithmic realism with fully deterministic codes. This being true for any R ∈ (R(1)(∆, 0), Hp(X)), we have as desired. Then, from (51) and Claim B.5, we have

for some $\kappa>0$. Hence, from (51), (53), (54), and Theorem A.4, we have  $$\limsup_{n\to\infty}\mathbb{E}_{(P^{(n)})\otimes B_{n}}\left[d\big{(}\mathbf{X}^{(n,B_{n})},\mathbf{Y}^{(n,B_{n})}\big{)}\right]\leq\Delta+\varepsilon,$$
n→∞
E(P (n))⊗Bn-d(X(n,Bn), Y(n,Bn))≤ ∆ + ε, (57)

## C.1 Encoder Derandomization

918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971

## D Converse Of Theorem 4.1

We show that Claim C.1 follows from Hamdi et al. (2024, Proposition 4), and its proof. We can apply that result directly, since *R < H*p(X) and X is finite. This would give all properties in Claim C.1, except for the exponential decay in (59). To obtain the latter, it is sufficient to adapt the proof of Hamdi et al. (2024, Proposition 4), by replacing the use of the law of large numbers with the use of Hoeffding's inequality, and using Cuff (2013, Theorem VII.1) with τ = −nγ, for small enough γ.

From standard information-theoretic arguments, we have the following result - see Appendix F for a proof.

Lemma D.1. Consider a triplet (R, Rc, ∆) and a sequence of codes, the n-th being (*n, R, R*c),
inducing a sequence {P
(n)
X1:n*,J,M,Y*1:n
}n≥1 *of distributions such that*

$\lim\sup\mathbb{E}_{(P^{(n)})\otimes b_{n}}\left[d(\mathbf{X}^{(n,b_{n})},\mathbf{Y}^{(n,b_{n})})\right]\leq\Delta$, $n\rightarrow\infty$
$$(63)$$
$$(64)$$

$$(65)$$
for some sequence {bn}n≥1 of positive integers. For every n ≥ 1, let T
(n) denote a uniform variable on [nbn] *independent from all other random variables. Then, there exists a conditional distribution* pY |X and an increasing sequence {ni}i≥1 *of positive integers such that*

(P (ni)) ⊗bni XT (ni) ,YT (ni) −→ i→∞ pX,Y (64) ∆ ≥ Ep[d(X, Y )] (65) R ≥ Ip(X; Y ), (66)
where pX,Y *refers to* pX · pY |X.
We know that R∗(∆) ≥ R(∆), and prove that R(∆) ≥ R(1)(∆, 0). Consider a couple (R, ∆) ∈ R
2+,
and some Rc ∈ R≥0 such that (*R, R*c, {Bn}n≥1, ∆) is asymptotically achievable with algorithmic realism. Fix ε > 0. Then, there exists a sequence of codes, the n-th being (*n, R, R*c), inducing a sequence {P
(n)
X1:n*,J,M,Y*1:n
}n of distributions such that

$$(67)$$
$$\operatorname*{lim}_{n\to\infty}\mathbb{E}_{(P^{(n)})^{\otimes B_{n}}}\left[d(\mathbf{X}^{(n,B_{n})},\mathbf{Y}^{(n,B_{n})})\right]\leq\Delta+\varepsilon,$$
$$(68)$$

Then, Lemma D.1 applies, with bn = Bn, for all n, with R + ε instead of R, and ∆ + ε instead of
∆. Then, there exists a conditional distribution pY |X and an increasing sequence {ni}i≥1 of positive integers such that

(P (ni)) ⊗bni XT (ni) ,YT (ni) −→ i→∞ pX,Y (70) ∆ + ε ≥ Ep[d(X, Y )] (71) R + ε ≥ Ip(X; Y ), (72)
where for any n ∈ N, variable T
(n)is uniformly distributed on [nBn], and independent from all other random variables. We prove that pY ≡ pX. Fix e0 ∈ X . Consider the computable p
⊗∗
X -critic δ from Claim G.1, with q therein taken to be pX. Then, from (69),

$$\operatorname*{sup}_{n\in\mathbb{N}}\mathbb{E}_{(P^{(n)})^{\otimes B_{n}}}\left[\delta{\big(}\mathbf{Y}^{(n,B_{n})}{\big)}-2\log(\delta{\big(}\mathbf{Y}^{(n,B_{n})}{\big)}+3{\big)}\right]<\infty.$$

and for any lower semi-computable p
⊗∗
X -critic δ,
$$\operatorname*{sup}_{n\in\mathbb{N}}\mathbb{E}_{(P^{(n)})\otimes B_{n}}\left[\delta(\mathbf{Y}^{(n,B_{n})})\right]<\infty.$$
E(P (n))⊗Bn-δ(Y(n,Bn))< ∞. (69)
$$(69)$$
$$(70)$$

$$(71)$$
$\left(72\right)$. 
$$(73)^{\frac{1}{2}}$$
Consider some ∆ ∈ R+ such that R(1)(∆, 0) < Hp(X), and a sequence {Bn}n≥1 of positive integers such that

$\log(B_{n})/n\underset{n\rightarrow\infty}{\longrightarrow}0$.  
D.1 CONVERSE PROOF Thus,

## E Proof Of Theorem 4.4

972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025

$$\sup_{n\in\mathbb{N}}\mathbb{E}_{(P^{(n)})\otimes B_{n}}\left[\delta(\mathbf{Y}^{(n,B_{n})})\right]<\infty,\ \ \text{and}\ \ \mathbb{E}_{(P^{(n)})\otimes B_{n}}\left[\delta(\mathbf{Y}^{(n,B_{n})})-\frac{1}{2}\log(nB_{n})\right]\underset{n\to\infty}{\longrightarrow}-\infty.$$
$$(74)$$

Thus, the frequency of e0 in a batch of reconstructions converges in L1 norm to pX(e0). Hence, the expected frequencies converge to pX(e0). This rewrites as
(P
(n))
⊗Bn
YT (n)
(e0) → pX(e0). (74)
This is true for any e0 in X . Thus, from (70), pY ≡ pX. Hence, from (71) and (72), we have
R + ε ≥ R
(1)(∆ + ε, 0). (75)
This being true for any ε > 0, and since R(1)(·, 0) is convex -thus continuous- on (0, ∞), we have
R ≥ R
(1)(∆, 0). (76)
This being true for any R ∈ R+ such that there exists Rc ∈ R≥0 such that (*R, R*c, {Bn}n≥1, ∆) is
asymptotically achievable with algorithmic realism, we have
$$R(\Delta)\geq R^{(1)}(\Delta,0),$$
(1)(∆, 0), (77)
as desired.

some Rc ∈ R≥0, and some (R, ∆) ∈ (R+)
2such that tuple (*R, R*c, ∆) is asymptotically achievable with near-perfect realism. From Theorem 1 in Wagner (2022), (*R, R*c, ∆) achievable with perfect realism, i.e. satisfying the properties in Definition 4.3, with (20) replaced with
∃N ∈ N, ∀n ≥ *N, P*(n)
Y1:n
≡ p
⊗n X . (79)
Fix ε > 0, and a corresponding sequence of (n, R + *ε, R*c) codes. Denote by P
(n)the distribution induces by the n-th code. Then, there exists an integer Nε such that lim sup n→∞
EP (n)-d(X1:n, Y1:n)≤ ∆ + ε, (80)

$$(77)$$
$$(78)$$
$\mathbf{M}$
$\angle1\,=\,-\,\frac{1}{2}$ . 
$\mathbf{P}\in\mathbf{R}$
$$\forall n\geq N_{\varepsilon},\ (P_{Y_{1:n}}^{(n)})^{\otimes B_{n}}\equiv p_{X}^{\otimes n B_{n}}.$$
$$({\boldsymbol{\delta}}1)$$
$$(83)$$

X . (81)
From (80), (81), Claim B.3, and the additivity of the distortion measure d, we have lim sup n→∞
E(P (n))⊗Bn-d(X(n,Bn), Y(n,Bn))≤ ∆ + ε, (82)
Since this analysis is valid for every ε > 0, then (*R, R*c, {Bn}n≥1, ∆) is asymptotically achievable with algorithmic realism. Moving to the converse, consider a computable increasing sequence
{Bn}n≥1 of positive integers such that

$$\frac{B_{n}}{|{\cal X}|^{n}}\rightarrow\infty,$$

and for any lower semi-computable p
⊗∗
X -critic δ,

$$\pm(P^{\dagger})$$
$\pi$)) $\mathbb{R}^d$
$$\operatorname*{sup}_{n\in\mathbb{N}}\mathbb{E}_{(P^{(n)})^{\otimes B_{n}}}\left[\delta(\mathbf{Y}^{(n,B_{n})})\right]<\infty.$$
E(P (n))⊗Bn-δ(Y(n,Bn))< ∞. (86)
$$(84)$$
$$(n,B_{n})\}\leq\Delta+\varepsilon,\;\;\mathrm{and}$$
$$(86)^{\frac{1}{2}}$$

some Rc ∈ R≥0, and some (R, ∆) ∈ (R+)
2such that tuple (*R, R*c, ∆) is asymptotically achievable with algorithmic realism. Fix ε > 0. Then, there exists a sequence of codes, the n-th being
(n, R + *ε, R*c), such that the sequence {P
(n)}n of distributions induced by the codes satisfies lim sup n→∞
E(P (n))⊗Bn-d(X(n,Bn), Y(n,Bn))≤ ∆ + ε, and (85)

Consider an increasing sequence {Bn}n≥1 of positive integers such that
$$\frac{B_{n}}{|\mathcal{X}|^{n}}\to\infty,$$
$$\operatorname{Hint}\,\sigma_{+}$$
$$\operatorname*{update}p_{X}\operatorname*{\mathsf{CHIC}}\delta,$$ $$\operatorname*{sup}_{n\in\mathbb{N}}\mathbb{E}_{(P^{(n)})^{\otimes B_{n}}}\left[\delta(\mathbf{Y}^{(n,B_{n})})\right]<\infty.$$
and for any lower semi-computable p
⊗∗
X -critic δ, we have 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 Thus, Choosing, for every n ∈ N, η = 1/3 and ε = 1/Cn, then from Lemma E.1 and (90) we have, for large enough n,

$$(P^{(n)})^{\otimes B_{n}}\Big(\|\mathbb{P}_{\chi^{n}}^{\mathrm{emp}}[\mathbf{Y}^{(n,B_{n})}]-P_{Y_{1:n}}^{(n)}\|_{T V}\geq{\frac{1}{C_{n}}}\Big)\leq{\frac{1}{3}}.$$
. (91)
Consider the computable sequence of positive integers defined by

$$(91)$$
$\forall n\in\mathbb{N},\,A_{n}:=\left[\left(\frac{B_{n}}{|\,\chi^{\prime}|n}\right)^{\frac{4}{9}}\right]$.  
|X |n
4
Since {Bn}n≥1 is increasing, then for any t ∈ N, there exists a unique integer n ∈ N≥0 such that

$$(92)$$
$$93)$$
$$(94)$$

t ∈ [nBn,(n + 1)Bn+1),
with the definition B0 := 0. We define δ : ∪t∈NX
t → N≥0 as follows. For any integer t ∈ [1, B1),
and any x1:t ∈ X t, let δ(x) := 0. For any n ∈ N, any t ∈ [nBn,(n + 1)Bn+1), and any x1:t ∈ X t, let

$$\delta(x_{1:t}):=\left[A_{n}\left\|\mathbb{P}_{X^{n}}^{\mathrm{emp}}[x_{1:nB_{n}}]-p_{X}^{\otimes n}\right\|_{TV}\right].\tag{1}$$
proof in Appendix G.2. Then, we can apply (86) to critic $\delta-2\log(\delta+3)-L$, and  $$\sup_{n\in\mathbb{N}}\mathbb{E}_{(P^{(n)})\otimes B_{n}}\left[\delta(\mathbf{Y}^{(n,B_{n})})-2\log(\delta(\mathbf{Y}^{(n,B_{n})})+3)-L\right]<\infty.$$
$$\operatorname*{sup}_{n\in\mathbb{N}}\mathbb{E}_{(P^{(n)})^{\otimes B_{n}}}\left[\delta(\mathbf{Y}^{(n,B_{n})})\right]<\infty,\;\;\mathrm{and}\;\;(P^{(n)})^{\otimes B_{n}}\left(\delta(\mathbf{Y}^{(n,B_{n})})\geq C_{n}\right)\underset{n\to\infty}{\longrightarrow}0,$$

because {Cn}n≥1 tends to infinity. Combining this with (91) through a union bound, we obtain, from the triangle inequality for the TVD,

$$(P^{(n)})^{\otimes B_{n}}\big(\big\|P_{Y_{1:n}}^{(n)}-p_{X}^{\otimes n}\big\|_{T V}\leq\frac{C_{n}}{A_{n}}+\frac{1}{C_{n}}\big)>0,$$

for large enough n. The above event does not depend on the random batch, hence the corresponding
inequality is true, for large enough n. Since {Cn}n≥1 tends to infinity and since from (84), (89), and (92), we have Cn/An → 0, then we obtain
$$\left\|P_{Y_{1:n}}^{(n)}-p_{X}^{\otimes n}\right\|_{T V\ \,n\rightarrow\infty}0.$$
0. (95)
Hence, from (85) and the additivity of d, we have that (*R, R*c, ∆) is asymptotically achievable with near-perfect realism. This concludes the proof.

$$(95)$$

$$b\geq\lambda\cdot{\frac{k+\log(1/\eta)}{\varepsilon^{2}}},$$

$$(87)$$
2, (87)
$$q^{\otimes b}\Bigl(\bigl\|\mathbb{P}_{\mathcal{W}}^{e m p}[W^{b}]-q\bigr\|_{T V}\geq\varepsilon\Bigr)\leq\eta.$$
$$(88)$$
$$C_{n}:=\left[\left({\frac{B_{n}}{|{\mathcal{X}}|^{n}}}\right)^{\frac{1}{3}}\right].\tag{1}$$
$$(89)$$

$$(90)$$

Since X is finite, {Cn}n≥1 is a computable sequence of positive integers. Moreover, from (84), we
have
$$C_{n}\ {\xrightarrow[n\to\infty]{}}\ \infty.$$
∞. (90)
For every n ∈ N, define Lemma E.1. (Canonne, 2020) There exists a positive integer λ such that for any k ∈ N, any distribution q on some finite set W of size k, any ε, η > 0, and any integer b *satisfying* Claim E.2. *From Lemma E.1 and* (90), there exists a positive integer L *such that* δ−2 log(δ+ 3)−L is a lower semi-computable p
⊗∗
X *-critic.*