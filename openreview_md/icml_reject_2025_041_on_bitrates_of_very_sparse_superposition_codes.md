# On Bitrates Of Very Sparse Superposition Codes

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054

## Abstract

Sparse *autoencoders* have been used to interpret activity inside large language models as "superposition codes" for sparse, high-dimensional signals. The encoder layers of these autoencoders use simple methods, which we will call "onestep estimates," to read latent sparse signals from vectors of hidden neuron activations. This work investigates the reliability of one-step estimates on a generic family of sparse inference problems. We show that these estimates are remarkably inefficient from the point of view of coding theory: even in a "very sparse" regime, they are only reliable when the dimension of the code exceeds the entropy of the latent signal by a factor of 2.7 dimensions per bit. In comparison, a very naive iterative method called matching pursuit can read superposition codes given just 1.3 dimensions per bit. This opens the question of whether neural networks can achieve similar bitrates in their internal representations.

## 1. Introduction

If each neuron in a neural network signaled a meaningful "feature" of its input, we could hope to reverse-engineer the network's overall behavior on a neuron-by-neuron basis. However, individual neurons of real-world networks often lack clear interpretations. For example, both language models and vision models have been found to learn neurons that correlate simultaneously with apparently unrelated features. (See for example (Nguyen et al., 2016), (Zhang & Wang, 2023) and (Olah et al., 2020).) The difficulty of interpreting a network in terms of its local activity—and in particular, the appearance of socalled "polysemantic neurons"—is not surprising from a connectionist viewpoint. Since at least the 1980s, proponents of neural networks have argued that these systems 1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

Anonymous Authors1 Figure 1. A coarse code representing a point on a plane. Each "neuron," drawn as a red or blue square, encodes whether the point belongs to an associated "receptive field." Although no neuron gives specific information on the position of the point, the overall code determines its position with reasonable accuracy.

may naturally use distributed representations—coding schemes where individual features are represented by patterns spread over many neurons, and conversely where each neuron carries information on many features. (This term was apparently coined in (Rumelhart et al., 1986), Chapter 3.) In contrast, a *local* representation would dedicate each neuron to a single feature. (See (Thorpe, 1989) for a general discussion of local and distributed codes.) Figure 1 illustrates a classic example of a coarse code, one kind of distributed representation. It is not clear how deep neural networks learn to represent information in their hidden layers or to what extent this information can be interpreted. However, should "interpretable features" exist, the connectivist viewpoint makes it natural that they would be stored with non-local codes.

This is a common assumption in interpretability research today; for example, when (Meng et al., 2022) intervened on an MLP layer of a language model to "edit" a factual association, both the "subject" and the "fact" were modeled as vectors of neuron activations rather than as individual neurons. How can we infer latent features learned by a neural network? One simple proposal is to model an activation vector x as a linear projection x = Fy of some high-dimensional and *sparse* vector y of latent fea1 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 tures. We refer to the columns of F as codewords and the whole matrix F as a dictionary. Since x is a linear superposition of codewords, we will call it a superposition code for y. The task of inferring the sparse vector y from x is known as sparse reconstruction, and the task of inferring the dictionary F from a distribution over x is called dictionary learning. Both of these problems have been studied in the field of compressive sensing, although with different applications in mind. (See (Elad, 2010) for a review of classic work in the context of signal and image processing.) Already in 2015, (Faruqui et al., 2015) used a dictionary learning method to derive sparse latent codes for word embeddings and argued that these latents were more interpretable than the original embedding dimensions. More recently, a series of works beginning with (Yun et al., 2021) have applied dictionary learning to the internal representations of transformer-based language models. (Cunningham et al., 2023) suggested the use of sparse autoencoders (SAEs) and (Templeton et al., 2024; Gao et al.,
2024) scaled sparse autoencoders to production-size large language models. Sparse latents learned by SAEs are often highly intepretable, and (Templeton et al., 2024) showed that intervening at the level of features allows "steering" language models in predictable ways. However, even SAEs with very high-dimensional latents suffer from an apparently irreducible reconstruction error (Gao et al., 2024). Understanding the limitations of SAEs—and dictionary learning in general—is an important open question in interpretability (Sharkey et al., 2025).

## 2. Contributions

To infer a latent representation y from an activation vector x, sparse autoencoders use an estimate like yˆ(x) = σ(Gx)
for some learnable matrix G: RN×d and some simple nonlinear thresholding function σ. Meanwhile, the literature on compressive sensing is concerned mainly with *iterative* methods for sparse inference. Throughout this paper, we will refer to autoencoder estimates as "one-step estimates." It is natural that iterative methods for sparse reconstruction will perform more reliably than one-step estimates, but the nature of this gap is not obvious in general. Informally speaking, how bad are one-step estimates?

In this work, we answer this question in a toy scenario designed to model the "very sparse" latents learned by sparse autoencoders in practice. Our main contributions are the following.

1. We prove a theoretical guarantee on the performance of one-step methods and indicate simple "rules of thumb" that hold in practice. (See Section 3.3.)

8192 Matching pursuit 1 2 3 4 5 D
im en sion s p er bi t Top-k d Threshold 4096 0 20 40 60 80 100 k 512 1024 2048
2. We show empirically that the gap between one-step methods and iterative methods is significant, even for very sparse *latents*. In comparison to a simple method called matching pursuit, one-step methods require the dimension d of the superposition code to be larger by a constant factor. (See Section 3.5.)
From the point of coding theory, one natural measure for the efficiency of a sparse recovery method is its *bitrate*: that is, the ratio H/d between the entropy H of the latent signal and the minimum dimension d of the code x = Fy needed to recover y. In this language, matching pursuit can decode "very sparse" superposition codes at a rate of roughly one bit per dimension. On the other hand, one-step methods require upwards of 2.7 dimensions per bit. This rate increases as y becomes less sparse; for a latent vector y ∈ R220with 100 non-zero entries, one-step estimates require about 5 dimensions per *bit.* (See Figure 2.) How "efficient," in terms of bitrate, are the codes used by real neural networks? On one hand, it would not make sense for a network to use a code that requires a lengthy iterative decoding process before it can be used. On the other hand, it may still be possible for a network to learn to use codes that are "too efficient" to be entirely decoded by a one-step estimate. Overall, we hope this question informs future work on modeling distributed representations.

## 3. Encoding Sets With Superposition Codes

Given a large number N, consider a map F that "encodes" each subset y ⊆ [N] = {1, . . . , N} by a linear combination

$$x=F y=\sum_{i\in Y}f_{i}\in\mathbb{R}^{d},$$

where the vectors {fi ∈ Rd : i ∈ [N]} are chosen in advance and where the dimension d of the encoding is expected to be much smaller than N. As above, we call the vectors fi codewords for the elements of [N] and call the image Fy a superposition code for the set y. It will often be useful to view y as a vector in {0, 1}
Nwith coefficients

$$y_{i}={\begin{cases}1:i\in y\\ 0:{\mathrm{otherwise}}\end{cases}}$$

and view F as a matrix of column vectors [f1 . . . fN ],
called the dictionary. For simplicity, we'll model our subset as a random variable Y uniformly distributed over the subsets of some fixed size k ≪ N.

Ultimately, we are interested in understanding what might limit the success of SAEs and how other sparse dictionary learning methods may be designed. As a first step, this work addresses the following question. Question 1. When can Y be reliably decoded from the superposition code X = FY with the methods used by sparse autoencoders? Can other computationally efficient methods do significantly *better?*
110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 Specifically, we're interested in understanding how large the dimension d needs to be as a function of (N, k) for some class of method to recover Y, assuming the dictionary F is known. (We do not study the problem of learning the dictionary.) Since Y is a discrete variable, we will focus on conditions for *exact* recovery. We'll also focus on a regime where Y resembles the very sparse latent representations learned by sparse autoencoders trained on large language models. (Gao et al., 2024) discusses scaling the number of latent features on the order of N = 220 with sparsity on the order of k = 28, so we use this as our reference.

To map vectors of activations to latent sparse representations—in our language, to infer X from Y —sparse autoencoders essentially employ one-layer networks. For example, (Templeton et al., 2024) used a ReLU unit to estimate each coefficient of Y. Since the coefficients Yi in our toy scenario are either 0 or 1, a natural analog would be a thresholding rule of the form

$${\hat{Y}}_{i}(x)={\begin{cases}1:\langle\lambda_{i},X\rangle\geq1\\ 0:{\mathrm{otherwise}}\end{cases}}$$

Since the number k of non-zero coefficients is known beforehand, we can also choose the threshold adaptively so that only k of the Yˆi are non-zero. This is called top-k decoding. (Gao et al., 2024) showed that, in practice, top-k autoencoders perform better than their ReLU variants. We refer to both approaches as "one-step estimates." On the other hand, the field of compressive sensing offers a vast literature on *iterative* methods to recover a sparse vector from a linear projection. It is known that, in general, iterative methods are much more reliable than one-step estimates. Indeed, the first *iteration* of an iterative shrinkage method (see Chapter 6 of (Elad, 2010)) is formally identical to the kind of ReLU network employed by (Templeton et al., 2024). However, to our knowledge, a comparison of one-step estimates with iterative methods in the very sparse regime encountered by sparse autoencoders has so far been lacking. The following sections are organized as follows.

- Section 3.1 reviews some basic ideas from information theory and introduces bitrate as a measurement for the efficiency of an inference method.

- Section 3.2 reviews the idea of a matched filter and motivates the two one-step estimates we will consider.

- Section 3.3 studies the reliability of one-step estimates when the dictionary F is random.

- Section 3.4 argues that random dictionaries are "almost optimal" when k ≪ N.

- Section 3.5 discusses the empirical performance of an iterative method called matching pursuit.

## 3.1. Information Theory Bounds

In practice, each dimension of the superposition code FY carries a finite amount of information on the set Y. At best, the information that one dimension can store is determined by the number of states in its numeric datatype—a 16 bit floating point can store nearly 16 bits, and so on. However, under the moderate assumption that the projection F X can still be decoded after the addition of a certain level of white noise, classic results from information theory put more realistic bounds on the dimension of our encoding.

Proposition 1. For a given dictionary F ∈ Rd×N , *suppose* there exists a decoding map D so *that*

$$D(F Y+Z)=Y$$

with probability at least (1−p), where Z is a vector of i.i.d.

Gaussians with variance VZ. Suppose additionally that the We begin by describing the "toy scenario" to be studied.

maximum variance of any coefficient of the *code* X = FY if VX. *Define* Then (See Appendix A for a standard proof.) When p is small and lnNk is large, this means roughly that the "bitrate"

$$R=\log_{2}{\binom{N}{k}}\,{\Big/}\,d$$

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 cannot exceed the "channel capacity" C/ ln 2. (We alternate between bits and nats as convenient.) On the other hand, a classic result of information theory is that, as some block size parameter goes to infinity, there exist arbitrarily reliable coding schemes that essentially meet the channel capacity. In the remainder of this work, we will measure the minimum dimension d required for a certain inference method to recover Y in terms of the corresponding bitrate R. Of course, it will be useful to have an estimate for the entropy lnNk
. For fixed k,Nk is a polynomial in N and so

$$\ln\binom{N}{k}=k\ln N+O(1).$$

Indeed, k ln N is the entropy of an array of k elements of
[N] drawn with replacement. A better estimate for the entropy of a small subset is

$$\ln\binom{N}{k}\leq k\ln(e N/k)=k\ln N-k\ln k+k.$$

In fact, when k is small compared to N, k ln(*eN/k*) is an extremely good approximation. For example, where f ∈ Rd is known but the "noise term" Z ∈ Rd is an unobserved Gaussian vector. In signal processing, the problem of recovering an unobserved variable from a noisy process is known as *filtering*. In a linear system with Gaussian noise, like Equation (1), optimal filtering can be done using a linear function of the measurement data. Specifically, suppose Z has mean zero and non-singular covariance Σ, and define an inner product by 〈v, w〉Σ = xT Σ−1y. Then the posterior of S conditional on X is determined by the function

$$\lambda(X)={\frac{\langle f,X\rangle_{\Sigma}}{\|f\|_{\Sigma}^{2}}},$$

which we will call the matched filter for S. If S ∈ {0, 1}
is a binary variable, a routine calculations shows that the log odds of the posterior on S is given by

$$\begin{array}{r l}{\ln{\frac{\mathrm{P}(S=1|X=x)}{\mathrm{P}(S=0|X=x)}}}\\ {{}}&{{}=\rho\left(\lambda(x)-{\frac{1}{2}}\right)+\ln{\frac{\mathrm{P}(S=1)}{\mathrm{P}(S=0)}},}\end{array}$$
$$(2)^{\frac{1}{2}}$$
, (2)
where ρ = f2Σ is the "signal-to-noise ratio" of the filter λ. See Appendix D for a review. We now return to our original problem. Let's focus on estimating just one scalar Yi from the sum

$$X=Y_{i}f_{i}+\sum_{j\neq i}Y_{j}f_{j}.$$

The "noise term" here is not Gaussian, and the exact Bayesian posterior on Yi turns out to be intractable in gene ral. However, we can try to estimate Yi by approximating j∕=i *Yjfi* by a Gaussian vector of the same covariance.

The corresponding matched filter for Yi can be understood as a kind of least squares estimate.

In the following, let us assume that the codewords fi ∈ Rd are unit vectors. (It is natural for all the codewords fi to have the same magnitude if each coefficient Yi needs to be encoded with the same precision, as they do in our scenario.) If we assume further that the empirical distribution over codewords fi is approximately isotropic, then the matched filter for Yi is approximately

$$\lambda_{i}(X)=\langle f_{i},X\rangle.$$

(If the distribution over codewords is not isotropic, we can first apply a linear transformation to "whiten" the distribution of X.) A one-step estimate is an estimate for Y that relies directly on the matched filters λi. From Equation (2), the maximum likelihood estimate for Yi under our simplified Gaussian model is 1 if

$$\langle f_{i},X\rangle\geq{\frac{1}{\rho}}\ln{\frac{\mathrm{P}(Y_{i}=1)}{\mathrm{P}(Y_{i}=0)}}+{\frac{1}{2}}$$
$$(1)$$

4

## 3.2. Matched Filters And One-Step Estimates

Now, we turn to the problem of decoding a superposition code. Let's begin by reviewing the simpler problem of inferring a random scalar S from a sum

$$X=S f+Z$$

holds with a relative error of about 0.3%. (See Appendix B for a discussion of this estimate.)

$$\ln\left(\!\!\!{\binom{2^{20}}{2^{8}}}\!\!\right)\approx2^{8}\ln(2^{20}e/2^{8})=128(1+12\ln2)$$
$$C={\frac{1}{2}}\ln\left(1+{\frac{V_{X}}{V_{Z}}}\right).$$
$$d\geq C^{-1}\left((1-p)\ln\binom{N}{k}-\ln2\right).$$

and 0 otherwise. If we assume the signal-to-noise ratio ρ is very large, the decision boundary becomes approximately 1/2. This leads to the simpler of the two one-step estimates that we will consider. Definition 1. Given X = FY, the threshold *decoding* is

$${\hat{Y}}_{i}={\begin{cases}1:\langle f_{i},X\rangle\geq1/2\\ 0:o t h e r w i s e.\end{cases}}$$

On the other hand, if we know (or guess) the size k of the set Y in advance, the following is a natural way to make use of that information. (In the context of sparse autoencoders, this method was introduced by (Makhzani & Frey, 2014).)
Definition 2. Given X = FY, the top-k decoding is the set Yˆ of k elements whose codewords fi have largest inner products with X. (Ties are broken *arbitrarily.)*
Note that whenever threshold decoding succeeds at recovering Y, top-k decoding succeeds as well.

## 3.3. One-Step Estimates With Random Codewords

In this section, we show rigorously that one-step estimates are reliable so long as d = Ω(k ln N) and the dictionary F is random. Our theoretical results agree with numerical experiments, and we find that remarkably simple "rules of thumb" govern the performance of one-step estimates in practice. (See Figure 3.)
If inner products 〈fi, fj 〉 between distinct codewords are
"small enough" in some sense, then the matched filters
〈fi, X〉 will be reliable and we can expect one-step estimates to succeed. Indeed,

$$\langle f_{i},X\rangle=\left\langle f_{i},\sum_{j}Y_{j}f_{j}\right\rangle=\sum_{j}Y_{j}\langle f_{i},f_{j}\rangle$$ $$=Y_{i}+\sum_{j\neq i}Y_{j}\langle f_{i},f_{j}\rangle,$$
$$({\mathfrak{I}})$$

where the total "crosstalk" ξi is a sum of either (k − 1) or k inner products 〈fi, fj 〉.

One simple way to produce a dictionary of almostorthogonal codewords is to choose them randomly. For example, the following fact is representative of many similar results in high-dimensional geometry.

Proposition 2. Let d > 2−2(2 ln N + ln p−1), and let

$$\{F_{1},\ldots,F_{N}\}\subseteq\{-1/{\sqrt{d}},1/{\sqrt{d}}\}^{d}$$

be random vectors with independent, uniformly *distributed* entries. Then |〈Fi, Fj 〉| <  for all i ∕= j with probability at least (1 − p).

See Appendix C for a review. Let's call a pair (v, w) of vectors "-orthogonal" when |〈v, w〉| < . When all codewords are pairwise -
orthogonal in the sense of Proposition 2, the crosstalk ξi in Equation (3) is bounded strictly by k in absolute value. Putting  = k/2 gives the following corollary.

Corollary 1. Let d ≥ 8k2(2 ln N + ln p−1), and let F ∈ Rd×N be a dictionary of random codewords in the conditions of Proposition 2. Then with probability at *least*
(1 − p), every k-element subset Y ⊆ [N] is recovered from its superposition code FY by threshold *decoding.* For fixed k, we conclude that the dimension d of our codewords only needs to grow as Ω(ln N). However, the factor of 16k2 turns out to be very pessimistic; in practice, for almost all sets to be reliably encoded, we only need Ω(k ln N) dimensions.

Proposition 3. Let F ∈ Rd×N be a Rademacher dictionary in the conditions above. Fix a k-element set y ∈ [N] and *some* p ∈ (0, 1). If

$$d\geq8k(\ln N+\ln p^{-1}),$$

then y is accurately recovered from the random *variable* X = Fy by threshold decoding with probability at *least*
(1 − p).

As a heuristic guide for this result, consider the crosstalk ξi encountered by a matched filter 〈fi, X〉. If we view the other (N − 1) codewords as random Rademacher vectors Fj , we find that each inner product 〈fi, Fj 〉 is a sum of d independent Rademacher variables scaled to have total variance 1/d. It follows that the variance of ξi is at most k/d. To keep the power of this crosstalk below some fixed threshold, we conclude that d must grow linearly with respect to k. For a full proof, see Appendix E.

Note that, unlike Corollary 1, Proposition 3 does not guarantee that any *fixed* dictionary can reliably encode many sets y. However, we can easily derive such a guarantee with a Markov inequality.

Corollary 2. Let F ∈ Rd×N be a Rademacher dictionary as above and let , p > 0. If

$$d\geq8k(\ln N+\ln(\epsilon p)^{-1}),$$

then with probability at least (1 − p) it is true that at *least*
(1 − )Nk subsets y are accurately decoded from their images X = Fy by threshold *decoding.* The prediction of Proposition 3 agrees well with numerical experiments, graphed in Figure 3. In fact, even as N varies over several orders of magnitude, the slightly weaker condition d ≥ 8k ln N characterizes the regime where the 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 set Y can be decoded with reasonably high probability by threshold decoding. Top-k decoding performs significantly better but admits a similar "rule of thumb": for all values of N trialed,

$$d=4k\ln k N$$

is very close to the smallest dimension needed for top-k decoding to succeed with high probability. See Appendix F for an informal derivation of this bound.

## 3.4. Are Random Codewords Optimal?

So far, we've considered the performance of threshold and top-k decodings at recovering a subset from a superposition code with a *random* dictionary F. One natural question is whether we can do better if the dictionary is optimized. Of course, when d ≥ N, we can make the codewords fi exactly orthogonal. For this reason, the performance of onestep decodings shown in the top row of Figure 3 is much worse than is possible; we never need more than N dimensions to store a latent vector of dimension N. However, when the ratio d/N is small—say, smaller than 1/10—we conjecture that optimizing the dictionary gives practically no improvement over a random initialization. Unfortunately, we are not aware of a theoretical justification for this fact. To see why this may be true, recall the "crosstalk" terms

$$\xi_{i}=\sum_{j\neq i}Y_{j}\langle f_{i},f_{j}\rangle$$

from the previous section. For each i, this is a sum of between k and (k − 1) numbers drawn without replacement from the sequence

## (〈Fi, Fj 〉)J∕=I.

Let's fix the dictionary F and consider the empirical distribution defined by this sequence. Suppose this distribution has zero mean and variance

$$\gamma_{i}(F)=\frac{1}{N-1}\sum_{j\neq i}\langle f_{i},f_{j}\rangle^{2}.$$

When k is moderately large but much smaller than N, we expect the crosstalk ξi to behave like a centered Gaussian with variance *kγi.* Specifically, we expect that the probability of its tail events with respect to the random set Y will be governed by the product kγi. If we assume that tail events for the different variables ξi are "sufficiently independent,"
we conclude overall that the typical value of γi(F) is the limiting factor for the reliability of one-step estimates.

threshold, N = 28 top-k, N = 28 8192 8192 50 100 k 1024 2048 4096 50 100 k 1024 2048 4096 d d threshold, N = 212 top-k, N = 212 8192 8192 50 100 k 1024 2048 4096 50 100 k 1024 2048 4096 d d threshold, N = 216 top-k, N = 216 8192 8192 d 50 100 k 1024 2048 4096 d 50 100 k 1024 2048 4096 threshold, N = 220 top-k, N = 220 8192 8192 50 100 k 1024 2048 4096 50 100 k 1024 2048 4096 d d 0.00 0.25 0.50 0.75 1.00 Success rate
Figure 3. Empirical performance of threshold decoding (left) and top-k decoding (right) at the problem of recovering a kelement subset of [N] from a projection into d dimensions by a Rademacher random matrix. In the left column, we plot the relation d = 8k ln N. On the right, we plot d = 4k ln(kN) and its lower bound of d = 4k ln N.

275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 A dictionary chosen to have smaller "interference scales" γi would, in particular, have smaller average squared interference

$$\gamma(F)={\frac{1}{N}}\sum_{i=1}^{N}\gamma_{i}={\binom{n}{2}}^{-1}\sum_{i\neq j}\langle f_{i},f_{j}\rangle^{2}.$$

For a random dictionary F, γ(F) equals 1/d in expectation. Can we decrease this value significantly by optimization?

330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 Using projected gradient descent, we minimized γ(F) subject to the constraint of maintaining unit norm codewords. We tested dictionaries with between N = 64 and 65536 codewords and with codeword dimensions between d = 16 and 1024. In each case, we initialized with a random Rademacher dictionary and optimized to convergence with standard criteria. Our results are plotted in Figure 4. As d approaches N, we find that the optimal value γopt of γ(F) converges to 0, as expected. On the other hand, when d ≪ N, γopt is very close to 1/d, its expected value under a random initialization. For example, with N = 216
(not plotted), the optimal value of γ(F) is indistinguishable from 1/d on a log-log plot.

Furthermore, we find a striking regularity. Empirically, the ratio γopt/d−1 = dγopt between the optimal value of γ and its expected value at initialization turns out to be a function of the relative dimension *d/N.* Since this holds as N ranges over several orders of magnitude, it is natural to believe it may hold in general. Claim 1. For given (N, d), the optimal value of γ(F) for a dictionary F ∈ Rd×N of unit norm *codewords* is

$$\gamma_{o p t}(N,d)=\frac{\kappa(d/N)}{d}$$

for some function κ. Furthermore, κ(r) is close to 1 for small *values* of r.

If true, this means that the values γi(F) governing the scale of crosstalk suffered by matched filters can't be made significantly smaller than 1/d when d ≤ N for small .

We're not aware of theoretical resultsin this direction. Note in particular that this is not obviously related to work on sphere packing (see (Cohn & Zhao, 2014)) since we are interested in the *scale* of the distribution of inner products rather than in maximum values.

## 3.5. Comparison With Compressive Sensing

Together, Section 3.3 and Section 3.4 provide strong evidence that when k ≤ N, one-step estimates need nearly d ≥ 4k ln N dimensions to read a subset from a superposition code, even when the dictionary F is optimized. In the sense of Section 3.1, this means that the "bitrate" of a

N
64 128 256 512 1024 4096 24 25 26 27 28 29 210 Embedding dimension (d)
2°12 2°10 2°8 2°6 2°4 O
pt im al 
∞

2°6 2°5 2°4 2°3 2°2 2°1 20 Ratio d/N
2°4 2°3 2°2 2°1 20 21 Rat io 
∞o pt
/
∞i nit
superposition code is at most

R = log2 N k (4k ln N) ≤ k log2 (eN/k) 4k ln N =  1 4 ln 2 1 −  ln k − 1 ln N
$$(4)$$
(4)
bits per dimension. (Note that 4 ln 2 > 2.7.) There are several ways to interpret this conclusion. On one hand, it means that one-step estimates are "asymptotically inefficient" in terms of required bitrate when k is moderately large compared to N. More specifically, in a regime where N goes to infinity but ln k/ ln N converges to 1, we predict that one-step estimates only succeed when the bitrate R converges to zero.

In particular, one-step estimates are asymptotically inefficient when k/N ≥  for some positive . Indeed, to have d ≥ 4k ln N we would need d = Ω(N ln(N)), while the entropy of Y grows no faster than O(N). On the other hand, a hallmark result of compressive sensing implies that, when k/N ≤ , the vector y can be recovered from its image Fy under a random projection by a certain *convex* optimization problem so long as d ≥ κ()N for some constant κ(); for example, see (Candes & Tao, 2005). The failure of our one-step estimates in this particular regime is easy to prove. On the other hand, in a sparser regime where ln k/ ln N < for some  < 1, it follows from Proposition 3 that onestep estimates are "information-efficient" in the sense that they can be decoded from superposition codes with bitrate larger than some positive δ. However, it is also of interest to have *non-asymptotic* information on the required bitrate. From Equation (4) we find that one-step estimates need at least 2.7 bits per dimension. Can iterative methods do better?

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 There is an extensive literature on theory of compressive sensing. (Reeves et al., 2019) shows that, in our language, superposition codes with a random dictionary are essentially *optimal* in the information-theoretic sense when ideal maximum-likelihood inference is used as the decoder. A series of earlier works (Joseph & Barron, 2012; 2014; Rush et al., 2017) on superposition codes also showed that, under some conditions on y, certain decoding schemes admit bitrates up to theoretical channel capacity in the presence of Gaussian noise. However, to our knowledge, practical guarantees on the performance of iterative methods are not available for our range of k and N. Figure 5 shows the results of a numerical experiment using an iterative method called matching *pursuit*, first suggested in (Bergeaud & Mallat, 1995). This is a simple "greedy" algorithm that initializes y = 0 and, at each of k iterations, increments the index of y whose corresponding codeword has largest inner product with x − Fy.

We find empirically that matching pursuit far outperforms top-k decoding for the range of N and k considered earlier. Remarkably, decoding tendsto be successful with even odds when d = k log2(*eN/k),* meaning that matching pursuit requires only slightly more than one dimension per bit.

When d ≥ 1.3 log2(*eN/k),* decoding is very reliable when N = 216 and N = 220.

Previous work showed that sparse autoencoders can help learn interpretable representations of the activity inside a neural network. However, the success of these methods is limited for reasons that are not yet well understood. In this work, we have identified one point of view that might explain their limited success. In a toy scenario,

## 4. Conclusions And Future Work

N = 28 N = 212 2048 2048 50 100 k 256 512 1024 50 100 k 256 512 1024 d d N = 216 N = 220 2048 2048 d 50 100 k 256 512 1024 d 50 100 k 256 512 1024 0.00 0.25 0.50 0.75 1.00 Success rate
we showed that the simple estimates these models use to infer sparse representations are less "efficient," in an information-theoretic sense, than a simple iterative method. This is true even when the signal to be inferred is extremely sparse. To our knowledge, this kind of explicit, non-asymptotic comparison was not previously available in the literature. Of course, we do not suggest that the latent signal stored by a typical neural representation is well-modeled as a uniformly random k-sparse subset. However, the "bitrate gap" between one-step estimates and matching pursuit opens a natural question: how much information can neural networks typically encode in their internal activity? Can they, like matching pursuit, read around one bit of mutual information from each neuron? If they can, our findings suggest that sparse autoencoders may be fundamentally unable to decode their representations. Overall, we hope the point of view of coding efficiency helps inspire better interpretability methods in future work.

## 5. Impact Statement

This paper considers basic problems that may be relevant to interpretability of neural networks. We do not feel that any broader societal consequences need to be highlighted.

## References

Bergeaud, F. and Mallat, S. Matching pursuit of images. In Proceedings., International Conference on Image Processing, volume 1, pp. 53–56 vol.1, October 1995. doi: 10.1109/ICIP.1995.529037.

Candes, E. and Tao, T. Decoding by linear programming. IEEE Transactions on Information *Theory*, 51 (12):4203–4215, December 2005. ISSN 1557-9654. doi: 10.1109/TIT.2005.858979. Conference Name: IEEE Transactions on Information Theory.

Cohn, H. and Zhao, Y. Sphere packing bounds via spherical codes. Duke Mathematical *Journal*, 163(10), July 2014. ISSN 0012-7094. doi: 10.1215/00127094-2738857. arXiv:1212.5966 [math].

Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L. Sparse Autoencoders Find Highly Interpretable Features in Language Models, October 2023. arXiv:2309.08600.

Dasgupta, S. and Gupta, A. An elementary proof of a theorem of Johnson and Lindenstrauss. Random *Structures* & *Algorithms*, 22(1):60–65, January 2003. ISSN 10429832, 1098-2418. doi: 10.1002/rsa.10073.

Elad, M. Sparse and redundant representations: from theory to applications in signal and image *processing*. Springer, New York, 2010. ISBN 978-1-4419-7010-7 978-1-4419-7011-4. OCLC: ocn646114450.

Faruqui, M., Tsvetkov, Y., Yogatama, D., Dyer, C., and Smith, N. A. Sparse Overcomplete Word Vector Representations. In Zong, C. and Strube, M. (eds.), Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th *International* Joint Conference on Natural Language Processing (Volume 1: Long *Papers)*, pp. 1491–1500, Beijing, China, July 2015. Association for Computational Linguistics. doi: 10.3115/v1/P15-1144.

Foucart, S. and Rauhut, H. Sparse Recovery with Random Matrices. In Foucart, S. and Rauhut, H. (eds.), A Mathematical Introduction to Compressive *Sensing*, pp. 271–310. Springer, New York, NY, 2013. ISBN 978-08176-4948-7. doi: 10.1007/978-0-8176-4948-7 9.

Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., and Wu, J.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Scaling and evaluating sparse autoencoders, June 2024. arXiv:2406.04093 [cs].

Joseph, A. and Barron, A. R. Least Squares Superposition Codes of Moderate Dictionary Size Are Reliable at Rates up to Capacity. IEEE Transactions on Information Theory, 58(5):2541–2557, May 2012. ISSN 1557-9654. doi: 10.1109/TIT.2012.2184847. Conference Name: IEEE Transactions on Information Theory.

Joseph, A. and Barron, A. R. Fast Sparse Superposition Codes Have Near Exponential Error Probability for
$R<{\cal C}$. IEEE Trans. Inf. *Theor.*, 60(2):919–942, February 2014. ISSN 0018-9448. doi: 10.1109/TIT. 2013.2289865.

Makhzani, A. and Frey, B. k-Sparse Autoencoders, March 2014. arXiv:1312.5663.

Meng, K., Bau, D., Andonian, A., and Belinkov, Y. Locating and editing factual associations in GPT. Advances in Neural Information Processing *Systems*, 35:17359– 17372, 2022.

Nguyen, A., Yosinski, J., and Clune, J. Multifaceted Feature Visualization: Uncovering the Different Types of Features Learned By Each Neuron in Deep Neural Networks, May 2016. arXiv:1602.03616 [cs].

Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., and Carter, S. Zoom In: An Introduction to Circuits. *Distill*, 5(3):10.23915/distill.00024.001, March 2020. ISSN 2476-0757. doi: 10.23915/distill.00024. 001.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V.,
Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, . Scikit-learn: Machine Learning in Python. Journal of Machine Learning *Research*, 12(85):2825–2830, 2011. ISSN 1533-7928.

Reeves, G., Xu, J., and Zadik, I. The All-or-Nothing Phenomenon in Sparse Linear Regression. In *Proceedings* of the Thirty-Second Conference on Learning *Theory*, pp. 2652–2663. PMLR, June 2019. ISSN: 2640-3498.

Rumelhart, D. E., McClelland, J. L., and AU. Parallel Distributed Processing: Explorations in the Microstructure of Cognition: *Foundations*. The MIT Press, 1986. ISBN 978-0-262-29140-8. doi: 10.7551/mitpress/5236. 001.0001.

Rush, C., Greig, A., and Venkataramanan, R. Capacityachieving Sparse Superposition Codes via Approximate Message Passing Decoding. January 2017. ISSN 00189448. doi: 10.17863/CAM.8183. Publisher: IEEE.

Sharkey, L., Chughtai, B., Batson, J., Lindsey, J., Wu, J., Bushnaq, L., Goldowsky-Dill, N., Heimersheim, S., Ortega, A., Bloom, J., Biderman, S., Garriga-Alonso, A., Conmy, A., Nanda, N., Rumbelow, J., Wattenberg, M., Schoots, N., Miller, J., Michaud, E. J., Casper, S., Tegmark, M., Saunders, W., Bau, D., Todd, E., Geiger, A., Geva, M., Hoogland, J., Murfet, D., and McGrath, T. Open Problems in Mechanistic Interpretability, January 2025. arXiv:2501.16496 [cs].

Templeton, A., Conerly, T., Marcus, J., Lindsey, J.,
Bricken, T., Chen, B., Pearce, A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N. L., McDougall, C., MacDiarmid, M., Freeman, C. D., Sumers, T. R., Rees, E., Batson, J., Jermyn, A., Carter, S., Olah, C., and Henighan, T. Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. Transformer Circuits *Thread*, May 2024.

Thomas, M. and Joy, A. T. Elements of information *theory*.

Wiley-Interscience, 2006.

Thorpe, S. Local vs. Distributed Coding. *Intellectica*, 8
(2):3–40, 1989. doi: 10.3406/intel.1989.873. Publisher: Perse - Portail des revues scientifiques en SHS.

Vershynin, R. High-Dimensional Probability: An Introduction with Applications in Data *Science*. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, 2018. ISBN 978-1108-41519-4. doi: 10.1017/9781108231596.

Yun, Z., Chen, Y., Olshausen, B., and LeCun, Y. Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors. In Agirre, E., Apidianaki, M., and Vuli, I. (eds.), Proceedings of Deep Learning Inside Out (DeeLIO): The 2nd Workshop on Knowledge Extraction and Integration for Deep Learning *Architectures*, pp. 1–10, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.deelio-1.1.

Zhang, C. and Wang, Y. A sample survey study of polysemantic neurons in deep CNNs. In International Conference on Computer Graphics, Artificial Intelligence, and Data Processing (ICCAID *2022)*, volume 12604, pp. 849–855. SPIE, May 2023. doi: 10.1117/12.2674650.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549

## A. Proof Of Proposition 1

We restate Proposition 1 for convenience.

Proposition. For a given dictionary F ∈ Rd×N , suppose there exists a decoding map D so *that*

$$D(F Y+Z)=Y$$

550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 with probability at least (1 − p), where Z a vector of *i.i.d.*
Gaussians with variance VZ. Suppose additionally that the maximum variance of any coefficient of the code X = FY if VX. *Define* Then

$$d\geq C^{-1}\left((1-p)\ln\binom{N}{k}-\ln2\right)$$

Proof. By results on the capacity of Gaussian channels (see (Thomas & Joy, 2006), Chapter 9) we can bound the mutual information between X and X + Z as

$$I(X,X+Z)\leq{\frac{d}{2}}\ln(1+\rho)$$

where ρ is an upper bound for the signal-to-noise ratio of each entry of X + Z. In our case, we can put ρ = *VX/VZ.*
Now, let D is a decoding in the conditions above. Then a relaxation of Fano's inequality shows

$$I(Y,D(F Y+Z))\geq(1-p)\ln\binom{N}{k}-\ln2.$$

But since I(Y, FY +Z) ≥ I(Y, D(FY *+Z)),* we conclude that overall

$${\frac{d}{2}}\ln\left(1+{\frac{V_{X}}{V_{Z}}}\right)\geq(1-p)\ln\left(\!\!\!\begin{array}{c}{{N}}\\ {{k}}\end{array}\!\!\right)-\ln2.$$

## B. Estimates For The Binomial Coefficient

To estimate lnNk
, it is helpful to first remember the elementary inequalities Taking logarithms gives

$$k\ln(N/k)\leq\ln\binom{N}{k}\leq k\ln(e N/k),$$
and so $\ln\binom{N}{k}=k\ln(N/k)+O(k)$... 
In this work, we claimed the upper bound k *ln(eN/k*) is a very good approximation when k ≪ N. To see why, substitute the leading-order Stirling approximation ln n! =
n ln *n−n+O*(ln n) into the binomial coefficient to obtain

$$\ln\left(\begin{matrix}N\\ k\end{matrix}\right)=(N-k)\ln\left(\frac{N}{N-k}\right)$$ $$+\,k\ln\left(\frac{N}{k}\right)+O(\ln N).$$

Putting s = k/N, this simplifies to:

$$\ln\binom{N}{k}=h(s)N+O(\ln N),$$

where
$$h(s)=-s\ln s-(1-s)\ln(1-s)$$
$\downarrow$ . 
is the binary entropy function. For small s, note that
$$h(s)=-s\ln s+s+O(s^{2}),$$
and so overall

$$\ln\binom{N}{k}=k\ln N-k\ln k+k+O(s^{2}N)+O(\ln N).$$

In a regime where s = k/N converges to 0, we find that the estimate lnNk
≈ k ln(eN/k) is almost optimal in the sense that

$$\ln\binom{N}{k}=(k+O(1))\ln N-k\ln k+(1+o(1))k.$$

There is also a natural way to see this approximation from the point of view of coding theory. Consider a random subset Y ⊆ [N] where each element is included independently with probability s = k/N. Then the entropy of Y is

$H(Y)=h(s)N=sN\ln s^{-1}+sN+O(s^{2}N)$, $=k\ln(eN/k)+O(s^{2}N)$,
the leading term of which matches our estimate for lnNk
.

## C. Review Of Chernoff Bounds

The results of Section 3.3 rely on well-known facts about tails of independent sums of "sub-Gaussian" distributions. Many references are available on this topic; for example, see Chapter 2 of (Vershynin, 2018). For completeness, here we provide an essentially self-contained proof of Proposition 2 based on the Chernoff bound for a sum of Rademacher variables. Given a random variable X, define the cumulant generating function KX(λ) as

$$K_{X}(\lambda)=\ln\operatorname{E}\exp(\lambda X).$$
$$C={\frac{1}{2}}\ln\left(1+{\frac{V_{X}}{V_{Z}}}\right).$$
$$\left({\frac{N}{k}}\right)^{k}\leq\binom{N}{k}\leq\left({\frac{e N}{k}}\right)^{k}.$$

For example, the cumulant generating function of a unit Gaussian Z is KZ(λ) = λ2/2. Chernoff *bounds* are the following upper bounds on the probability of the tail event X ≥ a in terms of the cumulant generating function.

Proposition 4. For λ > 0, suppose KX(λ) exists. *Then*

$$\ln\operatorname{P}(X\geq a)\leq-\lambda a+K_{X}(\lambda).$$

Proof. By a Markov inequality,

P(X ≥ a) = P(eλX ≥ eλa) ≤ E exp(λX − λa) = exp(−λa + KX(λ)).
For a unit Gaussian, this gives

$$\ln\mathrm{P}(Z\geq a)\leq-\lambda a+\frac{1}{2}\lambda^{2}.$$

Minimizing with respect to λ then gives

$$\ln\mathrm{P}(Z\geq a)\leq-{\frac{1}{2}}a^{2}.$$

In fact, this is the best possible leading-order term; by wellknown bounds on Mills ratios, 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659

For any variable with |X| ≤ 1, it is relatively easy to show
that
$$K_{X}(\lambda)\leq{\frac{\lambda^{2}}{2}}.$$
.

For us, it is enough to know that this holds for the cumulant generating function KX(λ) = cosh(λ) of a Rademacher variable. It follows that the same bound holds for a sum Xn of n independent Rademachers scaled by 1/
√n:

$$K_{X_{n}/{\sqrt{n}}}(\lambda)=n\cosh(\lambda/{\sqrt{n}})\leq{\frac{\lambda^{2}}{2}}$$
.
Therefore, for a > 0, we can bound the tail of Xn in exactly the way that we would bound the tail of a Gaussian with standard deviation √n:

$$\ln\mathrm{P}(X_{n}\geq a)=\ln\mathrm{P}(X_{n}/{\sqrt{n}}\geq a/{\sqrt{n}})\leq-{\frac{a^{2}}{2n}}.$$

This gives us the tool we need to prove Proposition 2, restated here for convenience.

Proposition. Let d > 2−2(2 ln N + ln p−1), and let

$$\{F_{1},\ldots,F_{N}\}\subseteq\{-1/{\sqrt{d}},1/{\sqrt{d}}\}^{d}$$

be random vectors with independent, uniformly distributed entries. Then |〈Fi, Fj 〉| <  for all i ∕= j with *probability* at *least* (1 − p). Proof. Each inner product I = 〈Fi, Fj 〉 is distributed like a sum of d Rademacher variables scaled by 1/d. By the Chernoff bound above, we have that

$$\ln\mathrm{P}(I\geq\epsilon)=\mathrm{P}(X_{d}/d\geq\epsilon)\leq-{\frac{d^{2}\epsilon^{2}}{2d}}=-{\frac{1}{2}}d\epsilon^{2}.$$
$\geq\epsilon$) = P($I\leq-\epsilon$), and so by a union. 
By symmetry P(I ≥ ) = P(I ≤ −), and so by a union bound

$$\ln\mathrm{P}(|\langle F_{i},F_{j}\rangle|\geq\epsilon)\leq\ln(2\,\mathrm{P}(I\geq\epsilon))\leq-\frac{1}{2}d\epsilon^{2}+\ln2.$$

To conclude that |〈Fi, Fj 〉| <  for allN2
 < N2/2 pairs of vectors with probability at least 1 − p by a union bound, it suffices that

$$-\frac{1}{2}d\epsilon^{2}+\ln2\leq\ln\frac{p}{N^{2}/2}$$ $$=-2\ln N+\ln2+\ln p,$$

which is equivalent to the condition on d above. The interested reader should also compare this result to the Johnson-Lindenstrauss lemma, which is proved in a very similar way. (See (Dasgupta & Gupta, 2003) for a proof, or the last section of (Foucart & Rauhut, 2013) for a discussion of the JL lemma with some broader context.)

## D. Review Of Matched Filters

Consider the problem of inferring a scalar S from the sum

$$X=S f+Z$$

where f ∈ Rn and Z is a Gaussian variable independent from S. Suppose for simplicity that Z has non-singular covariance Σ, so that − ln p(z) = 1/2z2Σ where

$$\|z\|_{\Sigma}^{2}=z^{T}\Sigma^{-1}z.$$

Then a routine calculation shows that

$-\ln p(S=s|X=x)$. 
− ln p(S = s|X = x)
$$=C(x)-\ln p(s)+\frac{1}{2}\left(s-\frac{\langle f,x\rangle_{\Sigma}}{\|f\|_{\Sigma}^{2}}\right)^{2}\|f\|_{\Sigma}^{2}\tag{5}$$
where C(x) is a constant depending only on x and 〈−, −〉Σ
is the inner product associated with the norm −Σ. In particular, the distribution of S conditional on X is only a

$$\mathrm{P}(Z\geq a)=-{\frac{1}{2}}a^{2}-\ln a+O(1).$$

Now, let Xn be a sum of independent Rademacher variables, each uniformly distributed over {−1, 1}. We intuitively expect Xn/
√n to be distributed like a unit Gaussian for large n, and so we may hope that P(Xn/
√n ≥ a) is similarly bounded as a function of a. A Chernoff bound lets us formalize this.

function of the inner product 〈f, *X〉Σ.* The matched filter for S is the linear function

$$\lambda(X)={\frac{\langle f,X\rangle_{\Sigma}}{\|f\|_{\Sigma}^{2}}},$$

and can be understood as providing the maximum likelihood estimate for S conditional on X under a uniform improper prior. The quality of our matched filter is measured by its signalto-noise ratio (SNR)

$$\rho=\frac{(\lambda(f))^{2}}{\mathrm{Var}_{Z}\,\lambda(Z)}=\|f\|_{\Sigma}^{2}.$$

Up to a scalar, λ can be characterized as the linear function that maximizes this quantity. Under an improper prior, Equation (5) shows the posterior distribution on S conditional on X is Gaussian with mean λ(X) and precision ρ.

## E. Proof Of Proposition 3

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 We return to the proof of Proposition 3, restated here for convenience.

Proposition. Let F ∈ Rd×N be a Rademacher *dictionary* in the conditions above. Fix a k-element set y ∈ [N] and some p ∈ (0, 1). If

$$d\geq8k(\ln N+\ln p^{-1}),$$

then y is accurately recovered from the random *variable* X = Fy by threshold decoding with probability at *least*
(1 − p).

Proof. Where X1, X2, . . . is a sequence of independent Rademacher variables of unit variance, denote

$$b(d,r)=\mathrm{P}\left(\sum_{i=1}^{d}X_{i}\geq{\sqrt{d}}r\right).$$

By a Chernoff bound, we know that

$$\ln b(d,r)\leq-\frac{1}{2}r^{2}$$

r2 (6)
holds uniformly over d. Now, consider a dictionary F in the conditions above, and let us view its codewords Fi as random vectors. Note that we can assume w.l.o.g. that y = {1, ..., k}, so that X = Fy = F1 + · · · + Fk.

Suppose that we apply threshold decoding with threshold τ, so that

$${\hat{Y}}_{i}={\begin{cases}1:\langle F_{i},X\rangle\geq\tau\\ 0:{\mathrm{otherwise}}.\end{cases}}$$

For i = 1, . . . , k, let Ai denote the event that yi = 1 ∕= *Yˆi.*
Then

$$\mathrm{P}(A_{i})=\mathrm{P}(\langle F_{i},X\rangle<\tau)$$
$$=\mathrm{P}\left(\sum_{\begin{subarray}{l}{j\neq i}\\ {j=1}\end{subarray}}^{k}\langle F_{i},F_{j}\rangle<\tau-1\right)$$

 .
The sum above is distributed like a sum of (k − 1)d independent Rademacher variables scaled by 1/d. Overall,

$$\begin{split}\text{P}(A_{i})&=\text{P}\left(\frac{1}{d}\sum_{i=1}^{(k-1)d}X_{i}\geq1-\tau\right)\\ &=b\left((k-1)d,(1-\tau)\sqrt{\frac{d}{k-1}}\right).\end{split}$$
.
Similarly, for i = k+1, . . . , N, let Bi denote the event that yi is not correctly inferred. Then the same reasoning shows

$z\,\mathrm{P}(B_{i})=\mathrm{P}(\langle F_{i},F_{1}+\cdots+F_{k}\rangle>\tau)$  $$=\mathrm{P}\left(\frac{1}{d}\sum_{i=1}^{kd}X_{i}\geq\tau\right)=b\left(kd,\tau\sqrt{\frac{d}{k}}\right)$$
.
Overall, using Equation (6), we have

$$\begin{array}{c}{{\mathrm{P}(A_{i})\leq\exp\left(-{\frac{(1-\tau)^{2}}{2}}\cdot{\frac{d}{k-1}}\right)}}\\ {{\leq\exp\left(-{\frac{(1-\tau)^{2}}{2}}\cdot{\frac{d}{k}}\right)}}\end{array}$$

and

$$\mathrm{P}(B_{i})\leq\exp\left(-{\frac{\tau^{2}}{2}}\cdot{\frac{d}{k}}\right).$$

With τ = 1/2, the probability of failure is bounded as

$$\mathrm{P}\left(\bigcup_{i=1}^{k}A_{i}\cup\bigcup_{i=k+1}^{N}B_{i}\right)\leq\sum_{i=1}^{k}\mathrm{P}(A_{i})+\sum_{i=k+1}^{N}\mathrm{P}(B_{i})$$ $$\leq k\exp\left(-\frac{d}{8k}\right)+(N-k)\exp\left(-\frac{d}{8k}\right)$$ $$=N\exp\left(-\frac{d}{8k}\right).$$
$$(6)$$

Setting this bound less than p and rearranging proves the theorem.

## F. Possible Extensions Of Proposition 3

In practice, the numerical experiments reported in Section 3.3 show that threshold decoding succeeds with little more than d = 8k ln N dimensions. In fact, it is likely possible to prove the conclusion of Proposition 3 under slightly milder conditions by using a refinement of the Chernoff bound. For example, recall from Appendix C that the actual probability of a Gaussian tail event Z ≥ a is

$$\ln\mathrm{P}(Z\geq a)=-\frac{1}{2}a^{2}-\ln a+O(1),$$

which is slightly less than −1/2a2 for large a. (Note that, when d satisfies the conditions of Proposition 3, the parame√
ter a used in the Chernoff bound grows on the order of ln N.)
Numerical experiments also showed that top-k decoding succeeds with only slightly more than 4k ln(kN) dimensions. We believe it is also possible to prove a bound to justify this empirical observation.

To see how, let us denote Ai,j for the event that

$$\langle F_{i},X\rangle\geq\langle F_{j}X\rangle.$$

Then top-k decoding succeeds so long as no event Ai,j holds for i ∈ {k + 1, . . . , N} and j ∈ {1, . . . , k}. Each event is identically distributed, so by a union bound we conclude that top-k decoding succeeds with probability at least (1 − p) if

$$\ln\operatorname{P}(\langle F_{k+1},X\rangle\geq\langle F_{1},X\rangle)\leq\ln p-\ln(k(N-k)).$$

Both inner products above have variance 1/d and are, in some sense, approximately independent. We therefore expect that their difference can be approximated Gaussian variable with variance 2/d. A Chernoff bound would then give

$$\ln\mathrm{P}(\langle F_{k+1},X\rangle-\langle F_{1},X\rangle\geq0)\leq-\frac{\sqrt{d/2}^{2}}{2}=-\frac{d}{4}.$$

In terms of d, this means we need only

$$\begin{array}{r l}{d\geq4(\ln(k(N-k))+\ln p^{-1})}\\ {\ }&{{}\approx4(\ln(k N)+\ln p^{-1}).}\end{array}$$

Again, we expect that improving the Chernoff bound with lower-order terms would show that only slightly more than 4k ln(kN) dimensions are enough.

## G. Empirical Results On Basis Pursuit Denoising

We used the implementation of LASSO regression available in sklearn (Pedregosa et al., 2011) to infer sparse subsets of {1, . . . , 216} from superposition codes by minimizing the objective

$${\frac{1}{2d}}\|x-F{\hat{y}}\|_{2}^{2}+10^{-5}\|{\hat{y}}\|_{1}$$

715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769

1024 0.0 0.2 0.4 0.6 0.8 1.0 S
ucc es s rat e 512 d 256 20 40 60 80 100 k
Figure 6. Empirical performance of basis pursuit decoding for N = 216. The bold line plots d = k log2(*eN/k),* and the dotted line plot d = 0.8k log2(*eN/k*).

with respect to yˆ. In compressive sensing, this is known as basis pursuit denoising (BPDN). Results are graphed in Figure 6. Compared to the performance of matching pursuit shown in Figure 5, we find that BPDN can recover a subset from even fewer dimensions; around 0.8 bits per dimension are enough.