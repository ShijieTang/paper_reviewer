000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 In neural compression, vector quantization (VQ) is usually replaced by a differentiable approximation during training for gradient backpropagation. However, prior approximation methods face two main issues: 1) the train-test mismatch between differentiable approximation and actual quantization, and 2) the suboptimal encoder gradients for rate-distortion (RD) optimization. In this paper, we first provide new finds about how approximation methods influence the RD optimization in neural compression, and then propose a new solution based on these finds. Specifically, if a neural compressor is regarded as a source-space VQ, we find that the encoder implicitly determines the quantization boundaries, and the decoder determines the quantization centers. Suboptimal approximation methods lead to suboptimal gradients for RD optimization of quantization boundaries and centers. Therefore, to address the first issue, we propose an encode-decoder alternating optimization strategy. The encoder is optimized with differentiable approximation, and the decoder is optimized with actual quantization to avoid the train-test mismatch of quantization centers. To address the second issue, we propose a spherenoise based stochastic approximation method. During encoder optimization, VQ
is replaced with a uniform sphere noise centered at the input vector. When the input vector is located at the quantization boundary, the encoder gradient is closer to the difference in RD loss between adjacent quantization centers, facilitating better encoder optimization. We name the combination of optimization strategy and approximation method as Alternating Optimized Stochastic Vector Quantization. Experimental results on various vector sources and natural images demonstrate the effectiveness of our method.

## 1 Introduction

Quantization is a classical lossy compression technique. In theory, vector quantization (VQ) Gersho & Gray (1992) can achieve optimal rate-distortion (RD) performance in source coding. However, the exponentially increasing complexity of VQ and its non-differentiable nature limit its practical use in neural compression Balle et al. (2017); Ball ´ e et al. (2020); Lu et al. (2019); Li et al. (2021), ´ particularly for high-dimensional data. The complexity issue can be addressed by simplifying VQ to scalar quantization Balle et al. (2020), multistage VQ Feng et al. (2023); Zhu et al. (2022) or lattice ´ VQ Zhang & Wu (2023). In this paper, we focus on tackling the non-differentiability issue of VQ for end-to-end RD optimization. In neural compression, quantization is performed in the latent space of an autoencoder. Since quantization is non-differentiable, optimizing the learnable encoder transform presents a significant challenge. A typical solution is to introduce a differentiable approximation of quantization during training, such as additive uniform noise Balle et al. (2017) and straight-through estimator (STE) Bengio ´
et al. (2013). However, prior works mainly focus on a special case of VQ, *i.e.*, uniform scalar quantization. For general vector quantization, the optimization problem remains unresolved and primarily involves two issues. The first issue is train-test mismatch. Differentiable approximations Agustsson et al. (2017a); Zhu et al. (2022) often differ from actual quantization, resulting in mismatch in the decoder's reconstruction between training and testing. The second issue is the suboptimality of encoder gradients. Although previous approximation methods are differentiable, the gradients backpropagated to the encoder remain suboptimal under the RD criterion.

# Alternating Optimized Stochastic Vector Quantization In Neural Compression

Anonymous authors Paper under double-blind review

## Abstract

1 In this paper, we aim to design an VQ approximation method for end-to-end RD optimization in neural compression. Since the gradients for the encoder and decoder vary depending on the approximation method, the first step is to understand how they influence RD performance. By interpreting a neural compressor as a source-space vector quantizer, we show that the encoder function implicitly determines the quantization boundaries, and the decoder function determines the quantization centers. Suboptimal boundaries and centers directly lead to suboptimal RD performance in lossy compression. Thus, the encoder gradient at the boundaries and the decoder's gradient at the centers are key factors influencing compression performance. In theory, entropy-constrained vector quantization (ECVQ) Chou et al. (1989) has the optimal quantization boundaries and centers. To address the train-test mismatch issue, we propose an encode-decoder alternating optimization strategy. When optimizing the quantization centers, the encoder is fixed, and the decoder and codebook are optimized using actual quantization. When optimizing the quantization boundaries, the decoder and codebook are fixed, and the encoder is optimized using the approximation method. These two steps alternate during training, ensuring consistent decoder reconstruction while allowing gradients to be backpropagated to the encoder. To address the issue of suboptimal encoder gradients, we first provide gradient analysis and argue that prior approximation methods are suboptimal for RD performance due to two reasons: 1)
discontinuous encoder gradients result in non-smooth quantization boundaries, and 2) the encoder gradients at boundaries should align with the RD loss differences when quantizing to nearby centers. In theoretically optimal ECVQ, the RD loss for an input vector at the quantization boundary is equal when quantized to the two neighboring centers. Therefore, if the encoder gradient at the boundary closely approximates the loss difference between neighboring centers, it will help the encoder learn better quantization boundaries. Based on this analysis, we propose a sphere-noise based stochastic approximation method. This quantization approximation follows a uniform sphere distribution centered at the input vector, with the radius of the hypersphere equal to the distance between the input vector and the nearest quantization center. We further demonstrate that the encoder gradient is equivalent to the integral of the RD function over the surface of the high-dimensional sphere. When the input vector lies at the quantization boundary, the gradient is closer to the difference in RD loss between adjacent quantization centers, leading to more effective encoder optimization. By combining the proposed alternating optimization strategy and shere-noise based stochas- tic approximation, we propose a new method named Alternating Optimized Stochastic Vector Quantization for end-to-end RD optimization. We provide comprehensive experiments and analysis on various vector sources. Experimental results on neural image compression further demonstrate the effectiveness of the proposed method.

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107

## 2 Related Work

Most existing works in neural compression follows the structure of nonlinear transform coding Balle´ et al. (2020), with a pair of learnable transform, an entropy model and a vector quantizer in latent space. As shown in Figure 1, the encoder transform ga maps the input vector x into latent vector y, which is then quantized by a quantizer yˆ = Qy(y) = Qdy
(Qey
(y)). Qeyis the quantization encoder that maps y to discrete index i, and the quantization decoder Qd y maps i to quantized vector yˆ. The entropy model pi is used to model the distribution of index i for entropy coding. The optimization target is to minimize the RD loss L = R + λD, where R = Ex[− log pi(i)] = Ex[− log pyˆ(yˆ)] is rate and D = Exd(x, gs(yˆ)) is distortion. λ is a coefficient controlling the RD trade-off and d is a distortion metric.

The vector quantization Q is usually simplified to uniform quantization, *e.g.*,, rounding to the nearest integer, where yˆ = ⌊y⌉ = i. Most previous approximation methods are designed for uniform quantization. propose to add uniform noise on y during training. use straight-though estimator (STE) that copies gradient from yˆ to y to enable the training of encoder. Both and propose stochastic rounding that randomly quantizes y into two nearest integers, where anneals stochastic rounding to rounding during training. Agustsson & Theis (2020) propose a soft quantizer that smoothly interpolate between uniform noise and rounding. propose to optimize encoder with additive uniform noise and optimize decoder with rounding to reduce train-test mismatch. propose a two-stage strategy which first uses uniform noise for pre-training and then uses rounding for decoder finetuning. Among these methods, we can observe that additive uniform noise perform well for encoder 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 optimization and rounding or annealing based rounding perform well for decoder optimization. In Section 3.2, we provide an explanation for this observation based gradient analysis. For general vector quantization, the approximation design is more complicated. Agustsson et al. (2017a) propose a smooth approximation of vector quantization which is annealed to hard quantization during training. Zhu et al. (2022) replace vector quantization with a stochastic approximation that randomly quantize y to different codewords in the codebook. In VQVAE Van Den Oord et al. (2017), the authors use STE passing gradient from decoder to encoder, and introduce a VQ distance loss between y and codewords for the optimization of codebook and encoder. These method can only be optimized with distortion loss, where the rate is determined by codebook size. To achieve joint RD optimization, Feng et al. (2023) further improve the approximation in VQVAE with entropyconstrained vector quantization (ECVQ) Chou et al. (1989) in latent space. However, based on the analysis in Section 3.2, we argue that the encoder or decoder learned by previous methods are suboptimal in terms of RD performance.

## 3.1 Interpreting Neural Compression As Vector Quantization

From VQ definition Gersho & Gray (1992), a vector quantizer Qx of size N partitions the input vector space R
kinto N regions or cells. The region corresponding to the codeword ci ∈ Cx denote as Si, where i belongs to a index set I and Cx is the codebook. Siis defined as:

## 3 Gradient Analysis In Neural Compression

In Section 2, we introduce the common architecture of neural compression and several VQ approximation methods. However, it is unclear about the impact of these approximation methods on the RD optimization. In the following, we first show that optimizing encoder and decoder is equivalent to optimizing the quantization boundaries and centers. Then we provide gradient analysis at boundaries and centers, showing the suboptimality of existing approximation methods of vector quantization.

Here, the regions satisfied SN
i=1 Si = R
kand Si ∩ Sj = ∅ for all i ̸= j.This implies that all the regions form a partition of the k-dimensional Euclidean space R
k. The quantization boundaries are the partition boundaries, and the quantization centers are the codewords. The quantization boundaries and centers determine the RD performance.

For a neural compressor shown in Figure 1, we can regard the whole process xˆ = gs◦Qdy ◦Qey ◦ga(x)
as a vector quantization process xˆ = Qx(x) in source space. For the quantizer Qx, the quantization

$$S_{i}=\{\mathbf{x}\in\mathbb{R}^{k}\mid Q_{x}(\mathbf{x})=\mathbf{c}_{i}\}$$
k| Qx(x) = ci} (1)
162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 UQ-AUN We start with uniform quantization for simplicity. Additive uniform noise (AUN) Balle´ et al. (2017); Balle et al. (2018a) is one of the most popular method to approximate uniform quan- ´ tization during training. The rounding result yˆ = ⌊y⌉ is replaced with y˜ = y + u, where u is encoder is Qex = Qey ◦ ga and the quantization decoder is Qdx = gs ◦ Qdy. It is important to know how a neural compressor determines the quantization boundaries and centers of Qx in source space. In fact, the boundaries of Qx is determined by boundaries of Qy and encoder transform ga; the centers of Qx is determined by centers of Qy and decoder transform gs. Assuming input vector x is transformed into y and quantized to yˆ = ei ∈ Cy, where Cy is codebook of Qy. We can define the latent space region Ai partitioned by Qy as Ai = {y ∈ R
ky | Qy(y) = ei}. As y = ga(x) and xˆ = gs(yˆ), we have:

$$S_{i}=\{\mathbf{x}\in\mathbb{R}^{k}\mid g_{a}(\mathbf{x})\in A_{i}\}$$
k| ga(x) ∈ Ai} (2)
$$(2)$$
$$\mathbf{c}_{i}=g_{s}(\mathbf{e}_{i})$$
$$({\mathfrak{I}})$$
ci = gs(ei) (3)

## 3.2 Gradient Analysis

During training, the quantized latent vector yˆ is replaced with a approximation y˜, and xˆ is changed to x˜ = gs(y˜). With the per sample RD loss l = − log pyˆ(yˆ) + λd(x, gs(yˆ)), we care about the encoder gradient E [∂l/∂y] and decoder gradient E [*∂l/∂*x˜].

In fact, according to Section 3.1, if we fix the encoder (*i.e.*, fix the quantization boundaries) and optimize decoder with test-time quantization yˆ, making the decoder gradient E [*∂l/∂*xˆ] towards zero will lead to optimal optimization result of quantization centers. Therefore, when encoder is fixed, the best approximation y˜ to optimize decoder is yˆ itself. In this section, we focus on analyzing the encoder gradient with different approximation methods for learning quantization boundaries.

Since the quantization boundaries are uniquely determined by regions, the boundaries of Qx depend only on the encoder transform ga and the boundaries of Qy. The quantization centers of Qx depend only on the decoder transform gs and the centers of Qy. Moreover, if y lies on the boundary between two adjacent regions Ai and Aj , then x will be on the boundary between Si and Sj . These finds show that optimizing the encoder and decoder is equivalent to optimizing the quantization boundaries and centers, providing insights on the design of approximation methods.

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 sampled from uniform noise U
-−
1 2
,
1 2 ky. The encoder gradient of the scalar y1 in y is:

$\mathfrak{I}$ 13. 
$$(4)$$

$$({\boldsymbol{S}})$$
Eu
$$\left[{\frac{\partial l}{\partial{\bar{y}}_{1}}}\right]$$
=
$$\int_{y_{1}-0.5}^{y_{1}+0.5}\cdots\int_{y_{k}-0.5}^{y_{k}+0.5}\frac{\partial l(\vec{y})}{\partial\tilde{y}_{1}}\mathrm{d}\tilde{y}_{1}\cdots\mathrm{d}\tilde{y}_{k}$$ $$\int_{y_{2}-0.5}^{y_{2}+0.5}\cdots\int_{y_{k}-0.5}^{y_{k}+0.5}l(y_{1}+0.5,\tilde{y}_{2},\cdots,\tilde{y}_{k})-l(y_{1}-0.5,\tilde{y}_{2},\cdots,\tilde{y}_{k})\mathrm{d}\tilde{y}_{2}\cdots\mathrm{d}\tilde{y}_{k}$$
  **Table ($n=1$), the second ground state is $l(y_{1}+0.5)-l(y_{1}-0.5)$**
=
If y˜1 is on the quantization boundaries, such as y1 = n + 0.5, n ∈ Z, the encoder gradient of y1 is related to loss differences when y1 is quantized to two nearby centers n and n + 1. In the case of a one-dimensional source (k = 1), the encoder gradient is simplified to:
l(y1 + 0.5) − l(y1 − 0.5) (5)
In Section 3.1, we show that if y lies at the boundary between two regions in the latent space, then x is similarly positioned at the boundary of two corresponding regions in the source space. Consequently, when the gradient approaches zero at quantization boundaries, we have l(n) = l(n + 1), which aligns perfectly with the boundary definition in optimal ECVQ Chou et al. (1989), given the quantization centers. In ECVQ, the loss of quantizing to two nearby centers is equal when x is at the boundaries. This is why NTC Balle et al. (2020) achieves near-optimal performance on ´ 1-dimensional sources. In Figure 2 (right), we illustrate the encoder gradients on a 1D Gaussian source. The gradients labeled "analytical" are calculated using Equation 4, while the unlabeled ones represent the averaged gradients over samples. The averaged gradients are smooth and closely match the theoretical results. Additionally, we show the encoder-decoder mapping function in Figure 2 (left). The encoder transform ga and decoder transform gs are not inverse functions of each other, leading to rate-constrained quantization results in source space (similar to ECVQ), where quantization boundaries are not at the center of two nearby quantization centers.

UQ-STE STE Bengio et al. (2013) is also an popular approximation. The value of y˜ is the same as the value of yˆ but with modified gradient, where dy˜/dy = 1. We can represent it as y˜ = y + sg [yˆ − y], where sg is the operation of stopping gradient. The encoder gradient for STE is equal to E [*∂l/∂*yˆ]. The gradient is discontinuous at boundaries because the yˆ suddenly changes from one quantization center to another one, as shown in Figure 3 (right). Moreover, the sum of the gradients on both sides of the boundary equals the difference in the derivatives of the RD loss, which can cause the RD optimization to get trapped in local optima. In Figure 3 (left), We show that the quantization boundaries optimized with UQ-STE are nonsmooth and suboptimal for RD performance.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 4.2 Stochastic Vector Quantization For Encoder Optimization

Consider a stochastic vector quantization, where the output y˜ belongs to a conditional distribution q(y˜ | y). We assume dy˜/dy = 1 and encoder gradient of the scalar y1 in y is:

$$\mathbb{E}_{\tilde{\mathbf{y}}}\left[{\frac{\partial l}{\partial y_{1}}}\right]=\mathbb{E}_{\tilde{\mathbf{y}}}\left[{\frac{\partial l}{\partial{\dot{y}}_{1}}}\right]=\int\limits_{\omega}q({\tilde{\mathbf{y}}}\mid\mathbf{y}){\frac{\partial l}{\partial{\dot{y}}_{1}}}\;\mathrm{d}{\dot{y}}_{1}\mathrm{d}{\tilde{y}}_{2}\cdots\mathrm{d}{\tilde{y}}_{k}$$

VQ-STE Since STE on uniform quantization does not define a way to optimize the codebook, it cannot be directly used in vector quantization with learnbale codebook. To simultaneously optimize the encoding transform and the codebook, VQVAE Van Den Oord et al. (2017); Razavi et al. (2019)
introduce additional VQ distance loss Dvq in latent space. The distance loss Dvq = Exd1(y, ei)
is calculated between the latent vector y and the corresponding codeword ei, where d1 is a VQ
distance metric in latent space. To ensure end-to-end RD optimization, Feng et al. (2023) further introduce ECVQ and additional rate loss. The loss function is as:

$$L_{1}=R+\lambda D+\beta D_{v q},$$
$$(6)$$
L1 = R + λD + βDvq, (6)
where β controls the trade-off between d and d1. Figure 4 illustrates that, unlike UQ-AUN, VQ-STE
does not optimize the encoder based on the difference in RD loss. Instead, it optimizes the encoder by balancing the latent-space VQ distance loss Dvq with the distortion loss D. This results in the encoder-decoder mapping becoming an identity mapping for 1D sources. The latent-space ECVQ in VQ-STE is equivalent to a source-space ECVQ. However, the issue of discontinuous gradients persists when Dvq and D cannot be properly balanced.

## 4 The Proposed Method 4.1 Encoder-Decoder Alternating Optimization

As shown in Figure 5, to address the train-test mismatch issues, this paper proposes an alternating optimization strategy for the encoder and decoder. When optimizing the quantization centers, the encoder is fixed, and the actual quantized values y˜ are used to generate the reconstruction x˜, after which the RD loss is computed to optimize the decoder and codebook. When optimizing the quantization boundaries, the decoder and codebook are fixed, and the quantization approximation y˜ is used to generate the reconstruction x˜, with the RD loss used to optimize the encoder. These two steps alternate during training. The entropy model is optimized during the first step.

$$\left(T\right)$$

6 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

Step2: Encoder Optimization Step1: Decoder & Codebook & Prior Optimization
ω is the integration area in R
k. Let q(y˜ | y) be a uniform sphere distribution centered at y. The radius of the hypersphere is equal to ∥y − yˆ∥. Therefore, the encoder gradient is as:

$$\int\frac{1}{V(\omega)}\frac{\partial l(\tilde{\mathbf{y}})}{\partial\tilde{y}_{1}}\;\mathrm{d}\tilde{y}_{1}\mathrm{d}\tilde{y}_{2}\cdots\mathrm{d}\tilde{y}_{k},\tag{1}$$

where V (ω) is the volume of hypersphere, and 1/V (ω) is the density because y˜ is uniformly distributed. According to the generalized Stokes theorem, we have the gradient as:

$${\frac{1}{V(\omega)}}\int\limits_{\partial\omega}l({\tilde{\mathbf{y}}})\mathrm{d}{\bar{y}}_{2}\cdot\cdot\cdot\mathrm{d}{\tilde{y}}_{k},$$
$$(8)$$
$$({\mathfrak{g}})$$

l(y˜)d˜y2 *· · ·* d˜yk,(9)
Therefore, the encoder gradient is the integration of loss function on the surface of the sphere. When y) is at the boundaries, both the nearby two quantization centers ei and ej are on the surface, due to ∥y − yˆ∥ = ∥y − ei∥ = ∥y − ej∥. In fact, the proposed approximation is a generalization of additive uniform noise. If q(y˜ | y) is uniform distributed within a unit hypercube centered at y with volume V (ω) = 1, the gradient will be the same as that in Equation 4.

## 5 Experiments 5.1 Setup

Source Data For vector sources, we conduct tests on 1-dimensional Gaussian sources, 2dimensional Boomerang sources, and 8-dimensional Laplace sources. For natural image sources, we train on the train2017 dataset from COCO Lin et al. (2014), which contains 118,287 images.

The training images are randomly cropped into 256×256 patches. The evaluation dataset is the Kodak dataset Kodak (1993), consisting of 24 images with a resolution of 768×512 pixels. Evaluation Metrics For vector sources, we use the following metric to measure distortion: −10 log(MSE(x, ˆx)), where MSE is the mean squared error. The bitrate is measured as bits per dimension (bpd). For natural images, the quality of the reconstructed images is evaluated using peak signal-to-noise ratio (PSNR) in the RGB color space, and the bitrate is assessed in bits per pixel (bpp). Both the distortion metrics d and d1 are mean squared error. Additionally, the BD- rate Bjontegaard (2001a) is employed to evaluate the average RD performance gain. Implementation Details For the model on low-dimensional vector sources, both the encoder and decoder transforms are constructed from Resblocks. The dimension of the latent-space vector is equal to the dimension of the source-space vector. For the model on image sources, the encoder and

$\square$
378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

## 5.2 Results And Analysis

decoder transforms follow the same structure as in the factorized model Balle et al. (2018b), with ´ the number of channels in the convolutional layers set to 192. For the entropy model, we use the factorized entropy model Balle et al. (2018b) when training with ´ UQ-AUN and UQ-STE. When training with VQ-STE and the proposed method, we employ the discrete entropy model Van Den Oord et al. (2017); Feng et al. (2023), which consists of a softmax function and learnable logits. For 1D, 2D, and 4D vector quantization, the codebook sizes are set to 256, 4096, and 32768, respectively. Since the codebook size required for vector quantization beyond 4 dimensions becomes excessively large without affecting performance, the experiments in this paper mainly focus on optimizing vector quantization for dimensions 4 and below.

We use the Adam optimizer Kingma & Ba (2014) for optimization, with a batch size of 1024 for low-dimensional vector sources and a batch size of 8 for image sources. In this section, we present the experimental results of the proposed method on different data sources, compare its performance with other methods, and conduct a series of ablation studies and analyses.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

| UQ-AUN   | VQ-STE-1d   | VQ-STE-2d   | VQ-STE-4d   | Ours-1d   | Ours-2d   | Ours-4d   | BPG444   |
|----------|-------------|-------------|-------------|-----------|-----------|-----------|----------|
| 0.0      | 1.15        | 2.42        | 0.16        | -5.58     | -7.20     | -9.39     | -26.16   |

## 5.2.1 Low-Dimensional Vector Sources

1D Gaussian Source For the 1D Gaussian source, the proposed method achieves performance very close to that of UQ-AUN. Here, we mainly showcase the encoder-decoder mapping function and the encoder gradient results for analysis. As shown in Figure 6, although the encoder gradients and encoder transform are not as smooth as those of UQ-AUN, the decoder transform remains sufficiently smooth and is able to learn nearly optimal mapping functions. Compared to UQ-AUN, the proposed method ensures train-test mismatch and is applicable to high-dimensional vector quantization. Additionally, compared to the UQ-STE method, the proposed method ensures optimal RD performance when determining quantization boundaries, avoiding the discontinuity in gradients. 2D Boomerang Source For the 2D Boomerang source, we present the RD performance and the visualized quantization results of the proposed method. As shown in Figure 7, Ours-1d, UQ-AUN and VQ-STE-1d are there neural compressors, where the dimension of the latent-space vector quantizers is 1, *i.e.*,, scalar quantizers. The difference lies in that UQ-AUN uses uniform scalar quantization, while VQ-STE-1d and Ours-1d are scalar quantizers with learnable codebooks. It can be observed that VQ-STE-1d has a significant performance drop compared to UQ-AUN, the main reason for which is analyzed in Section 3.2. In contrast, the proposed method achieves results comparable to NTC. 8D Laplace Source The experimental results on the 8D Laplace source are shown in Figure 8 (left). We performed both 1D and 4D vector quantization using the proposed optimization strategy. It can be observed that, even with scalar quantization, the performance of VQ-STE-1d, which uses the optimization strategy from previous work Feng et al. (2023), is slightly inferior to that of UQ- AUN. When the VQ dimension increases, the performance of VQ-STE-4d shows a significant drop.

In contrast, the proposed method (Ours-1d) maintains performance on par with UQ-AUN in scalar quantization. As the quantization dimension increases to 4, Ours-4d shows improvements at higher bitrates, confirming its effectiveness. Notably, the performance of 8D ECVQ plateaus beyond 1.75 bpd due to its codebook size being insufficient to meet bitrate demands. At this rate point, the codebook size reaches 409,600. Due to the exponential growth in codebook size with increasing bitrates, further expansion becomes impractical.

## 5.2.2 Natural Images

We also validate the effectiveness of the proposed method on the Kodak image dataset. Since the proposed alternating optimization strategy is only applicable to single-layer quantization and unconditional entropy models, we did not test on the state-of-the-art multi-layer quantization models for image compression. Instead, we tested the 1D, 2D, and 4D vector quantization results on the singlelayer Factorized model Balle et al. (2018b). The vector quantization is performed along the channel ´ dimension. For example, in the case of 4D quantization, the 192 × 1 × 1 channel vector is divided into 48 sub-vectors of size 4 × 1 × 1, and vector quantization is performed on each sub-vector using 48 different codebooks, with the codebooks shared across the spatial domain. This quantization method only removes redundancy in the channel domain and does not address spatial redundancy. The RD performance curve on the Kodak dataset is shown in the right column of Figure 8. Table 1 presents the BD-rate results Bjontegaard (2001b) with UQ-AUN as the baseline. It can be observed that the proposed method achieves steady performance improvements as the quantization dimension increases, while VQ-STE shows no significant improvement and even some performance degradation. Additionally, Ours-1d performs significantly better than UQ-AUN, primarily because alternating optimization resolves the train-test mismatch issue. Table 2 presents a series of ablation experiments on the Kodak dataset. To verify the effectiveness of the alternating optimization (A1), we directly fed the quantization approximation results into the decoder and used an additional loss to constrain the learning of the codebook. However, the model without the alternating optimization strategy experienced training collapse, demonstrating the importance of alternating optimization for stable convergence. Retaining the alternating optimization strategy, we replaced the proposed stochastic vector quantization method with two other approaches, including: soft-to-hard vector quantization Agustsson et al. (2017b) (A2), and probabilistic vector quantization Zhu et al. (2022) based on Gumbel Softmax Maddison et al. (2017) (A3). The rate of these methods is controlled by adjusting the codebook size. It can be observed that, with the same transform structures and optimization strategy, the proposed sphere-noise based stochastic approximation achieves better RD performance compared to other VQ approximation. Table 2: Abaltion studies on Kodak dataset in terms of PSNR. The benchmark is UQ-AUN (Factorized model Balle et al. (2018b)), with lower values indicating better performance. ´

| BD-rate                                  |       |
|------------------------------------------|-------|
| UQ-AUN                                   | 0.0   |
| Ours-4d                                  | -9.39 |
| A1: Ours-4d w/o alternating optimization | NaN   |
| A2: Ours-4d + Agustsson et al. (2017b)   | 25.31 |
| A3: Ours-4d + Zhu et al. (2022)          | 16.25 |

In this paper, we propose a method named Alternating Optimized Stochastic Vector Quantization to address the RD optimization issue in vector quantization based neural compression. We propose an encode-decoder alternating optimization strategy. The encoder is optimized with differentiable approximation, and the decoder is optimized with actual quantization to avoid the train-test mismatch of quantization centers. For better encoder optimization, we propose a sphere-noise based stochastic approximation method. During encoder optimization, VQ is replaced with a uniform sphere noise centered at the input vector. When the input vector is located at the quantization boundary, the encoder gradient is closer to the difference in RD loss between adjacent quantization centers, facilitating better encoder optimization. We provide a thorough analysis using toy vector sources and demonstrate through extensive experiments on neural image compression that our proposed method achieves a significant performance gain.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

## 6 Conclusion References

Eirikur Agustsson, Fabian Mentzer, Michael Tschannen, Lukas Cavigelli, Radu Timofte, Luca Benini, and Luc V Gool. Soft-to-hard vector quantization for end-to-end learning compressible representations. *Advances in neural information processing systems*, 30, 2017a.

Eirikur Agustsson, Fabian Mentzer, Michael Tschannen, Lukas Cavigelli, Radu Timofte, Luca Benini, and Luc V Gool. Soft-to-hard vector quantization for end-to-end learning compressible representations. In *Advances in Neural Information Processing Systems 30*, pp. 1141–1151, 2017b.

Johannes Balle, Valero Laparra, and Eero P. Simoncelli. End-to-end optimized image compression. ´
In *5th International Conference on Learning Representations, ICLR 2017*, 2017.

Johannes Balle, David Minnen, Saurabh Singh, Sung Jin Hwang, and Nick Johnston. Variational ´
image compression with a scale hyperprior. *arXiv preprint arXiv:1802.01436*, 2018a.

Johannes Balle, David Minnen, Saurabh Singh, Sung Jin Hwang, and Nick Johnston. Variational ´
image compression with a scale hyperprior. *arXiv preprint arXiv:1802.01436*, 2018b.

Johannes Balle, Philip A Chou, David Minnen, Saurabh Singh, Nick Johnston, Eirikur Agustsson, ´
Sung Jin Hwang, and George Toderici. Nonlinear transform coding. IEEE Journal of Selected Topics in Signal Processing, 15(2):339–353, 2020.

Gisle Bjontegaard. Calculation of average psnr differences between rd-curves. *VCEG-M33*, 2001a. Gisle Bjontegaard. Calculation of average psnr differences between rd-curves. *VCEG-M33*, 2001b.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Runsen Feng, Zongyu Guo, Weiping Li, and Zhibo Chen. Nvtc: Nonlinear vector transform coding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6101–6110, 2023. doi: 10.1109/CVPR52729.2023.00591.

Allen Gersho and Robert M Gray. *Vector quantization and signal compression*. 1992. Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Eastman Kodak. Kodak Lossless True Color Image Suite (PhotoCD PCD0992). http://r0k.

us/graphics/kodak/, 1993.

Jiahao Li, Bin Li, and Yan Lu. Deep contextual video compression. *arXiv preprint* arXiv:2109.15047, 2021.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ´ European conference on computer vision, pp. 740–755. Springer, 2014.

Guo Lu, Wanli Ouyang, Dong Xu, Xiaoyun Zhang, Chunlei Cai, and Zhiyong Gao. Dvc: An endto-end deep video compression framework. In *Proceedings of the IEEE Conference on Computer* Vision and Pattern Recognition, pp. 11006–11015, 2019.

Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. *Advances in neural information processing systems*, 32, 2019.

Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The concrete distribution: A continuous relaxation of discrete random variables, 2017. URL https://arxiv.org/abs/1611.00712.

Eirikur Agustsson and Lucas Theis. Universally quantized neural compression. *Advances in neural* information processing systems, 33:12367–12376, 2020.

Yoshua Bengio, Nicholas Leonard, and Aaron Courville. Estimating or propagating gradients ´
through stochastic neurons for conditional computation. *arXiv preprint arXiv:1308.3432*, 2013.

Philip A Chou, Tom Lookabaugh, and Robert M Gray. Entropy-constrained vector quantization.

IEEE Transactions on acoustics, speech, and signal processing, 37(1):31–42, 1989.

## A Appendix

You may include other additional sections here.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Xiaosu Zhu, Jingkuan Song, Lianli Gao, Feng Zheng, and Heng Tao Shen. Unified multivariate gaussian mixture for efficient neural image compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17612–17621, 2022.

Xi Zhang and Xiaolin Wu. Lvqac: Lattice vector quantization coupled with spatially adaptive companding for efficient learned image compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10239–10248, June 2023.

Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. *Advances in* neural information processing systems, 30, 2017.