# PROGRESSIVE COMPRESSION WITH UNIVERSALLY QUANTIZED DIFFUSION MODELS

Yibo Yang<sup>∗</sup> Justus C. Will<sup>∗</sup> Stephan Mandt Department of Computer Science University of California, Irvine {yibo.yang, jcwill, mandt}@uci.edu

### ABSTRACT

Diffusion probabilistic models have achieved mainstream success in many generative modeling tasks, from image generation to inverse problem solving. A distinct feature of these models is that they correspond to deep hierarchical latent variable models optimizing a variational evidence lower bound (ELBO) on the data likelihood. Drawing on a basic connection between likelihood modeling and compression, we explore the potential of diffusion models for progressive coding, resulting in a sequence of bits that can be incrementally transmitted and decoded with progressively improving reconstruction quality. Unlike prior work based on Gaussian diffusion or conditional diffusion models, we propose a new form of diffusion model with uniform noise in the forward process, whose negative ELBO corresponds to the end-to-end compression cost using universal quantization. We obtain promising first results on image compression, achieving competitive ratedistortion and rate-realism results on a wide range of bit-rates with a single model, bringing neural codecs a step closer to practical deployment. Our code can be found at <https://github.com/mandt-lab/uqdm>.

### 1 INTRODUCTION

A diffusion probabilistic model can be equivalently viewed as a deep latent-variable model [\(Sohl-](#page-11-0)[Dickstein et al.,](#page-11-0) [2015;](#page-11-0) [Ho et al.,](#page-11-1) [2020;](#page-11-1) [Kingma et al.,](#page-11-2) [2021\)](#page-11-2), a cascade of denoising autoencoders that perform score matching at different noise levels [\(Vincent,](#page-12-0) [2011;](#page-12-0) [Song & Ermon,](#page-11-3) [2019\)](#page-11-3), or a neural SDE [\(Song et al.,](#page-12-1) [2021b\)](#page-12-1). Here we take the latent-variable model view and explore the potential of diffusion models for communicating information. Given the strong performance of these models on likelihood estimation [\(Kingma et al.,](#page-11-2) [2021;](#page-11-2) [Nichol & Dhariwal,](#page-11-4) [2021\)](#page-11-4), it is natural to ask whether they also excel in the closely related task of data compression [\(MacKay,](#page-11-5) [2003;](#page-11-5) [Yang et al.,](#page-12-2) [2023\)](#page-12-2).

[Ho et al.](#page-11-1) [\(2020\)](#page-11-1); [Theis et al.](#page-12-3) [\(2022\)](#page-12-3) first suggested a progressive compression method based on an unconditional diffusion model and demonstrated its strong potential for data compression. Such a *progressive* codec is desirable as it allows us to decode data reconstructions from partial bit-streams, starting from lossy reconstructions at low bit-rates to perfect (lossless) reconstructions at high bitrates, all with a single model. The ability to decode intermediate reconstructions without having to wait for all bits to be received is a highly useful feature present in many traditional codecs, such as JPEG. The use of diffusion models has the additional advantage that we can, in theory, obtain perfectly realistic reconstructions [\(Theis et al.,](#page-12-3) [2022\)](#page-12-3), even at ultra-low bit-rates. Unfortunately, the proposed method requires the communication of Gaussian samples across many steps, which remains intractable because the exponential runtime complexity of channel simulation [\(Goc & Flamich,](#page-11-6) [2024\)](#page-11-6).

In this work, we take first steps towards a diffusion-based progressive codec that is computationally tractable. The key idea is to replace Gaussian distributions in the forward process with suitable *uniform* distributions and adjust the reverse process distributions accordingly. These modifications allow the application of universal quantization [\(Zamir & Feder,](#page-12-4) [1992\)](#page-12-4) for simulating uniform noise channels, avoiding the intractability of Gaussian channel simulation in [\(Theis et al.,](#page-12-3) [2022\)](#page-12-3).

<sup>∗</sup>Equal contribution

![](_page_1_Figure_1.jpeg)

Figure 1: Example reconstructions from several traditional and neural codecs, chosen at roughly similar bitrates. At high bitrates, our UQDM method preserves details (e.g. shape and color pattern of the spider, or sharpness of the calligraphy) better than other neural codecs. Note that among the methods considered here, only ours and CTC [\(Jeon et al.,](#page-11-7) [2023\)](#page-11-7) implement progressive coding.

- 1. We introduce a new form of diffusion model, Universally Quantized Diffusion Model (UQDM), that is suitable for end-to-end learned progressive data compression. Unlike in the closely-related Gaussian diffusion model [\(Kingma et al.,](#page-11-2) [2021\)](#page-11-2), compression with UQDM is performed efficiently with universal quantization, avoiding the generally exponential runtime of relative entropy coding [\(Agustsson & Theis,](#page-10-0) [2020;](#page-10-0) [Goc & Flamich,](#page-11-6) [2024\)](#page-11-6).
- 2. We investigate design choices of UQDM, specifying its forward and reverse processes largely by matching the moments of those in Gaussian diffusion, and obtain the best results when we learn the reverse-process variance as inspired by [Nichol & Dhariwal](#page-11-4) [\(2021\)](#page-11-4).
- 3. We provide theoretical insight into UQDM in relation to VDM, and derive the continuoustime limit of its forward process approaching that of the Gaussian diffusion. These results may inspire future research in improving the modeling formalism and training efficiency.
- 4. We apply UQDM to image compression, and obtain competitive rate-distortion and raterealism results which exceed existing progressive codecs at a wide range of bit-rates (up to lossless compression), all with a single model. Our results demonstrate, for the first time, the high potential of an unconditional diffusion model as a practical progressive codec.

#### 2 BACKGROUND

Diffusion models Diffusion probabilistic models learn to model data by inverting a Gaussian noising process. Following the discrete-time setup of VDM [\(Kingma et al.,](#page-11-2) [2021\)](#page-11-2), the forward noising process begins with a data observation x and defines a sequence of increasingly noisy latent variables z<sup>t</sup> with a conditional Gaussian distribution,

$$q(\mathbf{z}_t|\mathbf{x}) = \mathcal{N}(\alpha_t\mathbf{x}, \sigma_t^2\mathbf{I}), \quad t = 0, 1, \dots, T.$$

Here α<sup>t</sup> and σ t are positive scalar-valued functions of time, with a strictly monotonically increasing *signal-to-noise-ratio* SNR(t) := α 2 <sup>t</sup> /σ<sup>2</sup> t . The *variance-preserving* process of DDPM [\(Ho et al.,](#page-11-1) [2020\)](#page-11-1) corresponds to the choice α 2 <sup>t</sup> = 1 − σ 2 t . The reverse-time generative model is defined by a collection of conditional distributions p(zt−1|zt), a prior p(z<sup>T</sup> ) = N (0, I), and likelihood model p(x|z0). The conditional distributions p(zt−1|zt) := q(zt−1|zt, x = xˆθ(zt, t)) are chosen to have the same distributional form as the "forward posterior" distribution q(zt−1|zt, x), with x estimated from its noisy version z<sup>t</sup> through the learned *denoising model* xˆθ. Further details on the forward and

backward processes can be found in Appendix [A](#page-13-0) and [B.](#page-16-0) Throughout the paper the logarithms use base 2. The model is trained by minimizing the negative ELBO (Evidence Lower BOund),

$$\mathcal{L}(\mathbf{x}) = \underbrace{\text{KL}(q(\mathbf{z}_T|\mathbf{x}) \| p(\mathbf{z}_T))}_{:=L_T} + \underbrace{\mathbb{E}[-\log p(\mathbf{x}|\mathbf{z}_0)]}_{:=L_{\mathbf{x}|\mathbf{z}_0}} + \sum_{t=1}^T \underbrace{\mathbb{E}[\text{KL}(q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x}) \| p(\mathbf{z}_{t-1}|\mathbf{z}_t))]}_{:=L_{t-1}}, \quad (1)$$

where the expectations are taken with respect to the forward process q(z0:<sup>T</sup> |x). [Kingma et al.](#page-11-2) [\(2021\)](#page-11-2) showed that a larger T corresponds to a tighter bound on the marginal likelihood log p(x), and as T → ∞ the loss approaches the loss of a class of continuous-time diffusion models that includes the ones considered by [Song et al.](#page-12-1) [\(2021b\)](#page-12-1).

Relative Entropy Coding (REC) Relative Entropy Coding (REC) deals with the problem of efficiently communicating a single sample from a target distribution q using a coding distribution p. Suppose two parties in communication have access to a common "prior" distribution p and pseudo-random number generators with a common seed; a Relative Entropy Coding (REC) method [\(Flamich et al.,](#page-10-1) [2020\)](#page-10-1) allows the sender to transmit a sample z ∼ q using close to KL(q ∥ p) bits on average. If q arises from a conditional distribution, e.g., q<sup>x</sup> = q(z | x) is the inference distribution of a VAE (which can be viewed as a noisy *channel*), a *reverse channel coding* or *channel simulation* [\(Theis & Ahmed,](#page-12-5) [2022\)](#page-12-5) algorithm then allows the sender to transmit z ∼ q<sup>x</sup> with x ∼ p(x) using close to <sup>E</sup>x∼p(x) [KL(q(z | x) ∥ p(z))] bits on average. At a high level, a typical REC method works as follows. The sender generates a (possibly large) number of candidate z samples from the prior p,

$$\mathbf{z}_n \sim p, \quad n = 1, 2, 3, \dots,$$

and appropriately chooses an index K such that z<sup>K</sup> is a fair sample from the target distribution, i.e., z<sup>K</sup> ∼ q. The chosen index K ∈ <sup>N</sup> is then converted to binary and transmitted to the receiver. The receiver recovers z<sup>K</sup> by drawing the same sequence of z candidates from p (made possible by using a pseudo-random number generator with the same seed as the sender) and stopping at the Kth one.

A major challenge of REC algorithms is that their computational complexity generally scales exponentially with the amount of information being communicated [\(Agustsson & Theis,](#page-10-0) [2020;](#page-10-0) [Goc](#page-11-6) [& Flamich,](#page-11-6) [2024\)](#page-11-6). As an example, the MRC algorithm [\(Cuff,](#page-10-2) [2008;](#page-10-2) [Havasi et al.,](#page-11-8) [2018\)](#page-11-8) draws M candidate samples and selects K ∈ {1, 2, , ..., M} with a probability proportional to the importance weights, <sup>q</sup>(zn)/p(zn), n = 1, ..., M; similarly to importance sampling, M needs to be on the order of 2 KL(q∥p) for z<sup>K</sup> to be (approximately) a fair sample from q, thus requiring a number of drawn samples that scales exponentially with the relative entropy KL(q∥p) (the cost of transmitting K is thus log M ≈ KL(q∥p) bits). The exponential complexity prevents, e.g., naively communicating the entire latent tensor z in a Gaussian VAE for lossy compression, as the relative entropy KL(q(z|x) ∥ p(z)) easily exceeds thousands of bits, even for a small image. This difficulty can be partly remedied by performing REC on sub-problems with lower dimensions [\(Flamich et al.,](#page-10-1) [2020;](#page-10-1) [2022\)](#page-10-3) for which computationally viable REC algorithms exist [\(Flamich et al.,](#page-11-9) [2024;](#page-11-9) [Flamich,](#page-10-4) [2024\)](#page-10-4), but at the expense of worse bitrate efficiency due to the accumulation of codelength overhead across the dimensions.

Progressive Coding with Diffusion A *progressive* compression algorithm allows for lossy reconstructions with improving quality as more bits are sent, up till a lossless reconstruction. This results in variable-rate compression with a single bitstream, and is highly desirable in practical applications.

As we will explain, the NELBO of a diffusion model (eq. [\(1\)](#page-2-0)) naturally corresponds to the *lossless* coding cost of a progressive codec, which can be optimized end-to-end on the data distribution of interest. Given a trained diffusion model, a REC algorithm, and a data point x, we can perform progressive compression as follows [\(Ho et al.,](#page-11-1) [2020;](#page-11-1) [Theis et al.,](#page-12-3) [2022\)](#page-12-3): Initially, at time T, the sender transmits a sample of q(z<sup>T</sup> |x) under the prior p(z<sup>T</sup> ), using L<sup>T</sup> bits on average. At each subsequent time step t, the sender transmits a sample of q(zt−1|zt, x) given the previously transmitted zt, under the (conditional) prior p(zt−1|zt), using approximately Lt−<sup>1</sup> bits. Finally, given z<sup>0</sup> at t = 0, x can be transmitted losslessly under the model p(x|z0) by an entropy coding algorithm (e.g., arithmetic coding), with a codelength close to Lx|z<sup>0</sup> bits [\(Polyanskiy & Wu,](#page-11-10) [2022,](#page-11-10) Chapter 13.1). Thus, the overall cost of losslessly compressing x sums up to L(x) bits, as in the NELBO in eq. [\(1\)](#page-2-0). Crucially, at any time t, the receiver can use the most-recently-received z<sup>t</sup> to obtain a *lossy* data reconstruction xˆt. For this, several options are possible: [Ho et al.](#page-11-1) [\(2020\)](#page-11-1) consider using the diffusion model's denoising prediction xˆθ(zt), while [Theis et al.](#page-12-3) [\(2022\)](#page-12-3) consider sampling xˆ<sup>t</sup> ∼ p(x|zt), either by ancestral

sampling or a probability flow ODE [\(Song et al.,](#page-12-1) [2021b\)](#page-12-1). Note that if the reverse generative model captures the data distribution perfectly, then xˆ<sup>t</sup> ∼ p(x|zt) follows the same marginal distribution as the data and has the desirable property of *perfect realism*, i.e., being indistinguishable from real data [\(Theis et al.,](#page-12-3) [2022\)](#page-12-3).

Universal Quantization Although general-purpose REC algorithms suffer from exponential runtime [\(Agustsson & Theis,](#page-10-0) [2020;](#page-10-0) [Goc & Flamich,](#page-11-6) [2024\)](#page-11-6), efficient REC algorithms exist if we are willing to restrict the kinds of target and coding distributions allowed [\(Flamich et al.,](#page-10-3) [2022;](#page-10-3) [2024\)](#page-11-9). Here, we focus on the special case where the target distribution q is given by a uniform noise channel, which is solved efficiently by Universal Quantization (UQ) [\(Roberts,](#page-11-11) [1962;](#page-11-11) [Zamir & Feder,](#page-12-4) [1992;](#page-12-4) [Agustsson & Theis,](#page-10-0) [2020\)](#page-10-0). Specifically, suppose we (the sender) have access to a scalar r.v. Y ∼ p<sup>Y</sup> , and would like to communicate a noise-perturbed version of it,

$$\tilde{Y} = Y + U,$$

where U ∼ U(−∆/2, <sup>∆</sup>/2)is an independent r.v. with a uniform distribution on the interval [−∆/2, <sup>∆</sup>/2]. UQ accomplishes this as follows: *Step 1.* Perturb Y by adding another independent noise U ′ ∼ U(−∆/2, <sup>∆</sup>/2), and quantize the result to the closet quantization point K on a uniform grid of width ∆, i.e., computing K := ∆⌊ Y +U ∆ ⌉ where ⌊·⌉ denotes rounding to the nearest integer. *Step 2.* Entropy code and transmit K under the conditional distribution of K given U ′ . *Step 3.* The receiver draws the same U ′ by using the same random number generator and obtains a reconstruction Yˆ := K − U ′ = ∆⌊ Y +U ∆ ⌉ − U ′ . [Zamir & Feder](#page-12-4) [\(1992\)](#page-12-4) showed that Yˆ indeed has the same distribution as Y˜ , and the entropy coding cost of K is related to the differential entropy of Y˜ via

$$H[K|U'] = I(Y; \tilde{Y}) = h(\tilde{Y}) - \log(\Delta).$$

In the above, the optimal entropy coding distribution <sup>P</sup>(K|U ′ = u ′ ) is obtained by discretizing pY˜ := p<sup>Y</sup> ⋆ U(−∆/2, <sup>∆</sup>/2) on a grid of width ∆ and offset by U ′ = u ′ [\(Zamir & Feder,](#page-12-4) [1992\)](#page-12-4), where ⋆ denotes convolution. If the true pY˜ is unknown, we can replace it with a surrogate density model fθ(˜y) during entropy coding and incur a higher coding cost,

$$\mathbb{E}_{y \sim P_Y}[\text{KL}(u(\cdot|y) \parallel f_\theta(\cdot))] \geq I(Y; \tilde{Y}), \quad (2)$$

where u(·|y) denotes the density function of the uniform noise channel qY˜ <sup>|</sup><sup>Y</sup> <sup>=</sup><sup>y</sup> = U(y−∆/2, y+<sup>∆</sup>/2). It can be shown that the optimal choice of f<sup>θ</sup> is the convolution of p<sup>Y</sup> with U(−∆/2, <sup>∆</sup>/2). Therefore, as in prior work [\(Agustsson & Theis,](#page-10-0) [2020;](#page-10-0) [Balle et al.](#page-10-5) ´ , [2018\)](#page-10-5), we will choose f<sup>θ</sup> to have the form of another underlying density model g<sup>θ</sup> convolved with uniform noise, i.e.

$$f_\theta(\cdot) = g_\theta(\cdot) \star \mathcal{U}(\cdot; -\Delta/2, \Delta/2). \quad (3)$$

### 3 UNIVERSALLY QUANTIZED DIFFUSION MODELS

We follow the same conceptual framework of progressive compression with diffusion models as in [\(Ho et al.,](#page-11-1) [2020;](#page-11-1) [Theis et al.,](#page-12-3) [2022\)](#page-12-3), reviewed in the previous section. While [Theis et al.](#page-12-3) [\(2022\)](#page-12-3) use Gaussian diffusion, relying on the communication of Gaussian samples which remains intractable in higher dimensions, we want to apply UQ to similarly achieve a compression cost given by the NELBO, while remaining computationally efficient. We therefore introduce a new model with a modified forward process and reverse process, which we term *universally quantized diffusion model* (UQDM), substituting Gaussian noise channels for uniform noise channels.

#### 3.1 FORWARD PROCESS

The forward process of a standard diffusion model is often given by the transition kernel q(zt+1|zt) [\(Ho et al.,](#page-11-1) [2020\)](#page-11-1) or perturbation kernel q(zt|x) [\(Kingma et al.,](#page-11-2) [2021\)](#page-11-2), which in turn determines the conditional (reverse-time) distributions q(z<sup>T</sup> |x) and {q(zt−1|zt, x)|t = 1, ..., T} appearing in the NELBO in eq. [\(1\)](#page-2-0). As we are interested in operationalizing and optimizing the coding cost associated with eq. [\(1\)](#page-2-0), we will directly specify these conditional distributions to be compatible with UQ, rather than deriving them from a transition/perturbation kernel. We thus specify the forward process with

the same factorization as in DDIM [\(Song et al.,](#page-11-12) [2021a\)](#page-11-12) via q(z0:<sup>T</sup> |x) = q(z<sup>T</sup> |x) Q<sup>T</sup> <sup>t</sup>=1 q(zt−1|zt, x), and consider a discrete-time non-Markovian process as follows,

$$\begin{cases} q(\mathbf{z}_T|\mathbf{x}) := \mathcal{N}(\alpha_T \mathbf{x}, \sigma_T^2 \mathbf{I}), \\ q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x}) := \mathcal{U}\left(b(t)\mathbf{z}_t + c(t)\mathbf{x} - \frac{\Delta(t)}{2}, b(t)\mathbf{z}_t + c(t)\mathbf{x} + \frac{\Delta(t)}{2}\right), t = 1, 2, \dots, T, \end{cases} \quad (4)$$

where b(t), c(t), and ∆(t) are scalar-valued functions of time. Note that unlike in Gaussian diffusion, our q(zt−1|zt, x) is chosen to be a uniform distribution so that it can be efficiently simulated with UQ (as a result, our q(zt|x) for any t ̸= T does not admit a simple distributional form). There is freedom in these choices of the forward process, but for simplicity we base them closely on the Gaussian case: we choose a standard isotropic Gaussian q(z<sup>T</sup> |x), and set b(t), c(t), ∆(t) so that q(zt−1|zt, x) has the same mean and variance as in the Gaussian case (see Appendix [A](#page-13-0) for more details):

$$b(t) = \frac{\alpha_t}{\alpha_{t-1}} \frac{\sigma_{t-1}^2}{\sigma_t^2}, \quad c(t) = \sigma_{t|t-1}^2 \frac{\alpha_{t-1}}{\sigma_t^2}, \quad \Delta(t) = \sqrt{12} \sigma_{t|t-1} \frac{\sigma_{t-1}}{\sigma_t}, \quad \text{with } \sigma_{t|t-1}^2 := \sigma_t^2 - \frac{\alpha_t^2}{\alpha_{t-1}^2} \sigma_{t-1}^2.$$

We note here that q(zt|z<sup>T</sup> , x) can be written as a sum of uniform distributions, which as we increase T → ∞, converges in distribution to a Gaussian by the Central Limit Theorem. This implies that q(zt|x) also converges to a Gaussian for every t, and that our forward process has the same underlying continuous-time limit as in VDM [\(Kingma et al.,](#page-11-2) [2021\)](#page-11-2). We give the precise statement and a proof in Appendix [A.3.](#page-14-0)

As in VDM [\(Kingma et al.,](#page-11-2) [2021\)](#page-11-2), the forward process schedules (i.e., α<sup>t</sup> and σt, as well as b(t), c(t), ∆(t)) can be learned end-to-end, e.g., by parameterizing σ 2 <sup>t</sup> = sigmoid(ϕ(t)), where ϕ is a monotonic neural network. We did not find this to yield significant improvements compared to using a linear noise schedule similar to the one in [Kingma et al.](#page-11-2) [\(2021\)](#page-11-2).

#### 3.2 BACKWARD PROCESS

Analogously to the Gaussian case, we want to define a conditional distribution p(zt−1|zt) that leverages a denoising model xˆ<sup>t</sup> = xˆθ(zt, t) and closely matches the forward "posterior" q(zt−1|zt, x). In our case, the forward "posterior" corresponds to a uniform noise channel with width ∆(t), i.e., zt−<sup>1</sup> = b(t)z<sup>t</sup> + c(t)x + ∆(t)ut, u<sup>t</sup> ∼ U(−1/2, <sup>1</sup>/2); to simulate it with UQ, we choose a density model for zt−<sup>1</sup> with the same form as the convolution in eq. [\(3\)](#page-3-0). Specifically, we let

$$p(\mathbf{z}_{t-1}|\mathbf{z}_t) = g_\theta(\mathbf{z}_{t-1}; \mathbf{z}_t, t) \star \mathcal{U}(-\Delta^{(t)}/2, \Delta^{(t)}/2), \quad (5)$$

where gθ(zt−1; zt, t) is a learned density chosen to match q(zt−1|zt, x). Recall in Gaussian diffusion [\(Kingma et al.,](#page-11-2) [2021\)](#page-11-2), p(zt−1|zt) is chosen to be a Gaussian of the form q(zt−1|zt, x = xˆθ(zt;t)), i.e., the same as q(zt−1|zt, x) but with the original data x replaced by a denoised prediction x = xˆθ(zt;t). For simplicity, we base g<sup>θ</sup> closely on the choice of p(zt−1|zt) in Gaussian diffusion, e.g.,

$$g_\theta(\mathbf{z}_{t-1}; \mathbf{z}_t, t) = \mathcal{N}(b(t)\mathbf{z}_t + c(t)\hat{\mathbf{x}}_\theta(\mathbf{z}_t; t), \sigma_Q^2(t)\mathbf{I}) \quad (6)$$

or a logistic distribution with the same mean and variance,

$$g_\theta(\mathbf{z}_{t-1}; \mathbf{z}_t, t) = \text{Logistic } (b(t)\mathbf{z}_t + c(t)\hat{\mathbf{x}}_\theta(\mathbf{z}_t; t), \sigma_Q^2(t)\mathbf{I}). \quad (7)$$

where σ 2 <sup>Q</sup>(t) is the variance of the Gaussian forward "posterior", and we use the same noise-prediction network for xˆ<sup>θ</sup> as in [\(Kingma et al.,](#page-11-2) [2021\)](#page-11-2). We found the Gaussian and logistic distributions to give similar results, but the logistic to be numerically more stable and therefore adopt it in all our experiments.

Inspired by [\(Nichol & Dhariwal,](#page-11-4) [2021\)](#page-11-4), we found that learning a per-coordinate variance in the reverse process to significantly improve the log-likelihood, which we demonstrate in Sec. [5.](#page-6-0) In practice, this is implemented by doubling the output dimension of the score network to also compute a tensor of scaling factors sθ(zt), so that the variance of g<sup>θ</sup> is σ 2 <sup>θ</sup> = σ 2 <sup>Q</sup>(t) ⊙ sθ(zt). Refer to Appendix [B.2](#page-16-1) for a more detailed analysis of the log-likelihood and how a learned variance is beneficial.

We note that other possibilities for g<sup>θ</sup> exist besides Gaussian or logistic, e.g., mixture distributions [\(Cheng et al.,](#page-10-6) [2020\)](#page-10-6), which trade off higher computation cost for increased modeling power. Analyzing the time reversal of the our forward process, similarly to [\(Song et al.,](#page-11-12) [2021a\)](#page-11-12), may also suggest better choices of the reverse-time density model gθ. We leave these explorations to future work.

We adopt the same form of categorical likelihood model p(x|z0) as in VDM [\(Kingma et al.,](#page-11-2) [2021\)](#page-11-2), as well as the use of Fourier features.

| Algorithm 1     | Encoding        |            |           |     |   |     |      |   |     |   |   |     |                                                  |
|-----------------|-----------------|------------|-----------|-----|---|-----|------|---|-----|---|---|-----|--------------------------------------------------|
| z T ∼ p ( z T ) |                 |            |           |     |   |     |      |   |     |   |   |     |                                                  |
| for t = T,      | , 2 , 1 do      |            |           |     |   |     |      |   |     |   |   |     |                                                  |
| Let ∆ t         | = ∆( t ) , µ    | Q =        | b ( t     | ) z | t |     | +    | c | (   | t | ) | x   |                                                  |
| Compute         | the             | parameters | of        |     | p | (   | z    | t | − 1 |   |   | z t | )                                                |
| ▷ Send z        | t − 1 ∼ q (     | z t −      | 1   z t , | x ) |   |     | with |   |     |   |   | UQ: |                                                  |
| u t ∼ U (       | − 1 / 2 , 1 / 2 | )          |           |     |   |     |      |   |     |   |   |     |                                                  |
| k t = ∆ t       | ⌊               |            |           |     |   |     |      |   |     |   |   |     |                                                  |
|                 | µ Q             |            |           |     |   |     |      |   |     |   |   |     |                                                  |
|                 | ∆ t             |            |           |     |   |     |      |   |     |   |   |     |                                                  |
|                 | + u             | t ⌉        |           |     |   |     |      |   |     |   |   |     |                                                  |
| Derive          | entropy         | model      | p (       | k   | z | t , |      | u | t   | ) |   | by  | dis                                             |
| cretizing p (   | z t − 1   z t ) |            |           |     |   |     |      |   |     |   |   |     |                                                  |
| Entropy-encode  |                 | k t        | under     | p ( | k |     | z    | t | ,   | u | t | )   |                                                  |
| z t − 1 =       | k t − ∆ t u     | t          |           |     |   |     |      |   |     |   |   |     |                                                  |
| end for         |                 |            |           |     |   |     |      |   |     |   |   |     |                                                  |
| Entropy-encode  | x with          | p          | ( x   z 0 | )   |   |     |      |   |     |   |   |     |                                                  |
|                 |                 |            |           |     |   |     |      |   |     |   |   |     | Algorithm 2 Decoding                             |
|                 |                 |            |           |     |   |     |      |   |     |   |   |     | z T ∼ p ( z T ) ▷ Using shared seed              |
|                 |                 |            |           |     |   |     |      |   |     |   |   |     | for t = T, , 2 , 1 do                            |
|                 |                 |            |           |     |   |     |      |   |     |   |   |     | Let ∆ t = ∆( t )                                 |
|                 |                 |            |           |     |   |     |      |   |     |   |   |     | Compute the parameters of p ( z t − 1   z t )    |
|                 |                 |            |           |     |   |     |      |   |     |   |   |     | u t ∼ U ( − 1 / 2 , 1 / 2 ) ▷ Using shared seed  |
|                 |                 |            |           |     |   |     |      |   |     |   |   |     | Derive entropy model p ( k   z t , u t ) by dis |
|                 |                 |            |           |     |   |     |      |   |     |   |   |     | cretizing p ( z t − 1   z t )                    |
|                 |                 |            |           |     |   |     |      |   |     |   |   |     | Entropy-decode k t under p ( k   z t , u t )     |
|                 |                 |            |           |     |   |     |      |   |     |   |   |     | z t − 1 = k t − ∆ t u t                          |
|                 |                 |            |           |     |   |     |      |   |     |   |   |     | x ˆ t = x ˆ θ ( z t − 1 ; t − 1) ▷ Lossy         |
|                 |                 |            |           |     |   |     |      |   |     |   |   |     | end for                                          |
|                 |                 |            |           |     |   |     |      |   |     |   |   |     | Entropy-decode x with p ( x   z 0 ) ▷ Lossless   |

### 3.3 PROGRESSIVE CODING

Given a UQDM trained on the NELBO in eq. [\(1\)](#page-2-0), we can use it for progressive compression similarly to [\(Ho et al.,](#page-11-1) [2020;](#page-11-1) [Theis et al.,](#page-12-3) [2022\)](#page-12-3), outlined in Section [2.](#page-1-0)

The initial step t = T involves transmitting a Gaussian z<sup>T</sup> . Since we do not assume access to an efficient REC scheme for the Gaussian channel, we will instead draw the same z<sup>T</sup> ∼ p(z<sup>T</sup> ) = N (0, I) on both the encoder and decoder side, with the help of a shared pseudo-random seed.[<sup>1</sup>](#page-5-0) To avoid a train/compression mismatch, we therefore always ensure q(z<sup>T</sup> |x) ≈ p(z<sup>T</sup> ) and hence L<sup>T</sup> ≈ 0. At any subsequent step t, instead of sampling zt−<sup>1</sup> = b(t)z<sup>t</sup> + c(t)x + ∆(t)u ′ t as in training, we apply UQ to communicate the "forward posterior" mean vector µ<sup>Q</sup> := b(t)z<sup>t</sup> + c(t)x. Specifically, given zt, the sender computes µ<sup>Q</sup> and the parameters of p(zt−1|zt) (by evaluating the score network), draws a pseudo-random noise u<sup>t</sup> ∼ U(−1/2, <sup>1</sup>/2), quantizes µ<sup>Q</sup> to k<sup>t</sup> = ∆t⌊ µ<sup>Q</sup> ∆<sup>t</sup> + ut⌉ where ∆<sup>t</sup> := ∆(t), derives an entropy model p(k|zt, ut) (by discretizing p(zt−1|zt) on a grid of width ∆<sup>t</sup> and offset by ut), and entropy-encodes k<sup>t</sup> under p(k|zt, ut). The receiver draws the same pseudorandom u<sup>t</sup> ∼ U(−1/2, <sup>1</sup>/2), entropy-decodes k<sup>t</sup> under the same entropy model p(k|zt, ut), and computes zt−<sup>1</sup> = k<sup>t</sup> − ∆tu<sup>t</sup> and (optionally) a lossy reconstruction xˆ<sup>t</sup> from zt−1. Finally, after having transmitted z<sup>0</sup> when t = 1, x is losslessly compressed using the entropy model p(x|z0). Pseudocode can be found in Algorithms [1](#page-5-1) and [2.](#page-5-2) Note that we can replace the denoised prediction xˆ<sup>t</sup> = xˆθ(zt−1;t − 1) with more sophisticated ways to obtain lossy reconstructions such as flowbased reconstruction or ancestral sampling [\(Theis et al.,](#page-12-3) [2022\)](#page-12-3). As our method is progressive, the algorithm can be stopped at any time and the most recent lossy reconstruction be used as the output. Compared to compression with VDM [\(Theis et al.,](#page-12-3) [2022\)](#page-12-3), the main difference is that we transmit zt−<sup>1</sup> ∼ q(zt−1|zt, x) under p(zt−1|zt) using UQ instead of Gaussian channel simulation; the overall computation complexity is now dominated by the evaluation of the denoising network xˆ<sup>θ</sup> (for computing the parameters of p(zt−1|zt)), which scales linearly with the number of time steps.

We implemented the progressive codec using tensorflow-compression [\(Balle et al.](#page-10-7) ´ ), and found the actual file size to be within 3% of the theoretical NELBO.

### 4 RELATED WORK

Diffusion models [\(Sohl-Dickstein et al.,](#page-11-0) [2015\)](#page-11-0) have achieved impressive results on image generation [\(Ho et al.,](#page-11-1) [2020;](#page-11-1) [Song et al.,](#page-11-12) [2021a\)](#page-11-12) and density estimation [\(Kingma et al.,](#page-11-2) [2021;](#page-11-2) [Nichol & Dhariwal,](#page-11-4) [2021\)](#page-11-4). Our work is closely based on the latent-variable formalism of diffusion models [\(Ho et al.,](#page-11-1)

<sup>1</sup>This corresponds to a trivial REC problem where a sample from q = p can be transmitted using KL(q∥p) = 0 bits.

[2020;](#page-11-1) [Kingma et al.,](#page-11-2) [2021\)](#page-11-2), with our forward and backward processes adapted from the Gaussian case. Our forward process is non-Markovian like DDIM [\(Song et al.,](#page-11-12) [2021a\)](#page-11-12), and our reverse process uses learned variance, inspired by [\(Nichol & Dhariwal,](#page-11-4) [2021\)](#page-11-4). Recent research has focused on efficient sampling [\(Song et al.,](#page-11-12) [2021a;](#page-11-12) [Pandey et al.,](#page-11-13) [2023\)](#page-11-13) and better scalability via latent diffusion [\(Rombach](#page-11-14) [et al.,](#page-11-14) [2022\)](#page-11-14), consistency models [\(Song et al.,](#page-12-6) [2023\)](#page-12-6), and distillation [\(Sauer et al.,](#page-11-15) [2024\)](#page-11-15), whereas we focus on the compression task. Related to our approach, cold diffusion [\(Bansal et al.,](#page-10-8) [2024\)](#page-10-8) showed that alternative forward processes other than the Gaussian still produce good image generation results.

Several diffusion-based neural compression methods exist, but they use conditional diffusion models [\(Yang & Mandt,](#page-12-7) [2023;](#page-12-7) [Careil et al.,](#page-10-9) [2023;](#page-10-9) [Hoogeboom et al.,](#page-11-16) [2023\)](#page-11-16) which do not permit progressive decoding. Furthermore, they are also less flexible as a separate model has to be trained for each bitrate. Progressive neural compression has so far been mostly achieved by combining non-linear transform coding (for example using a VAE) with progressive quantization schemes. Such methods include PLONQ [\(Lu et al.,](#page-11-17) [2021\)](#page-11-17), which uses nested quantization, DPICT [\(Lee et al.,](#page-11-18) [2022\)](#page-11-18) and its extension CTC [\(Jeon et al.,](#page-11-7) [2023\)](#page-11-7), which use trit-plane coding, and DeepHQ [\(Lee et al.,](#page-11-19) [2024\)](#page-11-19) which uses a learned quantization scheme. Finally, codecs based on hierarchical VAEs [\(Townsend et al.,](#page-12-8) [2024;](#page-12-8) [Duan et al.,](#page-10-10) [2023\)](#page-10-10) are closely related but do not directly target the realism criterion.

### 5 EXPERIMENTS

We train UQDM end-to-end by directly optimizing the NELBO loss eq. [\(1\)](#page-2-0), summing up L<sup>t</sup> across all time steps. This involves simulating the entire forward process {z0, ..., z<sup>T</sup> } according to eq. [\(4\)](#page-4-0) and can be computationally expensive when T is large but can be avoided by using a Monte-Carlo estimate based on a single L<sup>t</sup> as in the diffusion literature [\(Ho et al.,](#page-11-1) [2020\)](#page-11-1). We found a small T (< 10) to give the best compression performance, and therefore leave the investigation of training with a single-step Monte-Carlo objective to future work. Note that this would require sampling from the marginal distribution q(zt|x), which becomes approximately Gaussian for large t (see Sec. [3.1\)](#page-3-1).

When considering the progressive compression performance of VDM and UQDM, we consider three ways of computing progressive reconstructions from zt: denoise, where xˆ = xˆθ(zt;t) is the prediction from the denoising network; ancestral, where xˆ ∼ p(x|zt) is drawn by ancestral sampling; and flow-based where xˆ ∼ p(x|zt) is computed deterministically using the probability flow ODE as in [\(Theis et al.,](#page-12-3) [2022\)](#page-12-3). In VDM, the probability flow ODE produces the same trajectory of marginal distributions as ancestral sampling, but gives improved lossy compression performance [\(Theis et al.,](#page-12-3) [2022\)](#page-12-3). In the case of UQDM, we apply the same update equations and observe similar benefits, likely due to the continuous-time equivalence of the underlying processes of UQDM and VDM. See Appendix [B.3](#page-17-0) for details. Note that DiffC-A and DiffC-F [\(Theis et al.,](#page-12-3) [2022\)](#page-12-3) directly correspond to our VDM results with ancestral and flow-based reconstructions.

In all experiments involving VDM and UQDM, we always use the same denoising U-net architecture for both, except UQDM uses twice as many output dimensions to additionally predict the reverseprocess variance (see Sec. [3\)](#page-3-2). We refer to Appendix Sec. [C](#page-19-0) for further experiment details.

#### 5.1 SWIRL TOY DATA

We obtain initial insights into the behavior of our proposed UQDM by experimenting on toy swirl data (see Appendix [C.1](#page-19-1) for details) and comparing with the hypothetical performance of VDM [\(Kingma](#page-11-2) [et al.,](#page-11-2) [2021\)](#page-11-2).

First, we train UQDM end-to-end for various values of T ∈ {3, 4, 5, 10, 15, 20, 30}, with and without learning the reverse process variance. For comparison, we also train a single VDM with T = 1000, but compute the progressive-coding NELBO eq. [\(1\)](#page-2-0) using different T. Fig. [2](#page-7-0) plots the resulting NELBO values, corresponding to the bits-per-dimension cost of lossless compression. We observe that for UQDM, learning the reverse-process variance significantly improves the NELBO across all T, and a higher T is not necessarily better. In fact, there seems to be an optimal T ≈ 5, for which we obtain a bpd of around 8. The theoretical performance of VDM, by comparison, monotonically improves with T (green curve) until it converges to a bpd of 5.8 at T = 1000, as consistent with theory [\(Kingma et al.,](#page-11-2) [2021\)](#page-11-2). We also tried initializing a UQDM without learned reverse-process variances to use the pre-trained VDM weights; interestingly, this resulted in very similar performance to the end-to-end trained result (blue curve), and further finetuning gave little to no improvement.

![](_page_7_Figure_1.jpeg)

Figure 2: Results on swirl data. The VDM curves correspond to the hypothetical performance of REC that remains computationally intractable. Left: Lossless compression rates v.s. the choice of T, for UQDM with/without learned reverse-process variance (blue/orange) and VDM (green). For UQDM, learning the reverse-process variance significantly improved the NELBO, and an optimal T ≈ 5. Middle, Right: Progressive lossy compression performance for VDM and UQDM, measured in fidelity (PSNR) v.s. bit-rate (middle), or realism (sliced Wasserstein distance) v.s. bit-rate (right).

![](_page_7_Figure_3.jpeg)

Figure 3: Progressive lossy compression performance of UQDM on the CIFAR10 dataset, comparing fidelity (PSNR) and realism (FID) with bit-rate per pixel (bpp), using either ancestral sampling or denoised prediction to obtain progressive reconstructions as indicated. The VDM curve corresponds to hypothetical performance of REC that is computationally intractable. We achieve better fidelity and realism than JPEG and JPEG2000 across all bit-rates and than BPG in the high bit-rate regime.

This suggests that a pretrained VDM can already be used for progressive compression with UQ via our moment-matching scheme (see Section [3\)](#page-3-2), although the compression performance will be much worse compared to end-to-end trained UQDM with learned reverse-process variances.

We then examine the lossy compression performance of progressive coding. Here, we train UQDM end-to-end with learned reverse-process variances, and perform progressive reconstruction by ancestral sampling. Figure [2](#page-7-0) plots the results in fidelity v.s. bit-rate and realism v.s. bit-rate. For reference, we also show the theoretical performance of VDM using T = 100 discretization steps, assuming a hypothetical REC algorithm that operates with no overhead. The results are consistent with those on lossless compression, with a similar performance ranking for T among UQDM, and a gap remains to the hypothetical performance of VDM.

Finally, we examine the quality of unconditional samples from UQDM with varying T. Although our earlier results indicate worse compression performance for T > 5, Figure [7](#page-18-0) shows that UQDM's sample quality monotonically improves with increasing T.

#### 5.2 CIFAR10

Next, we apply our method to natural images. We start with the CIFAR10 dataset containing 32 × 32 images. We train a baseline VDM model with a smaller architecture than that used by [Kingma et al.](#page-11-2) [\(2021\)](#page-11-2), converging to around 3 bits per dimension. We use the noise schedule σ 2 <sup>t</sup> = σ(γt) where γ<sup>t</sup> is linear in t with learned endpoints γ<sup>T</sup> and γ0. For our UQDM model we empirically find that T ≈ 4

![](_page_8_Figure_1.jpeg)

Figure 4: Progressive lossy compression performance of UQDM on the Imagenet64 dataset, comparing fidelity (PSNR) and realism (FID) with bit-rate per pixel (bpp), using either ancestral sampling or the denoised prediction to obtain progressive reconstructions as indicated. The VDM curve corresponds to hypothetical performance of REC that remains computationally intractable. While the reconstruction quality of other codecs like CDC or BPG plateaus at higher bit-rates, our method continues to gradually improve fidelity and realism even at higher bit-rates where it achieves the best results of any baseline. We beat compression performance of JPEG, JPEG2000, and CTC across all bit-rates. Note that only UQDM, CTC, and JPEG2000 implement progressive coding.

yields the best trade-off between bit-rate and reconstruction quality. We train our model end-to-end on the progressive coding NELBO eq. [\(1\)](#page-2-0) with learned reverse-process variances.

We compare against the wavelet-based codecs JPEG, JPEG2000, and BPG [\(Bellard,](#page-10-11) [2018\)](#page-10-11). For JPEG and BPG we use a fixed set of quality levels and encode the images independently, for JPEG2000 we instead use its progressive compression mode that allows us to set the approximate size reduction in each quality layer and obtain a rate-distortion curve from one bit-stream.

As shown in Figure [3,](#page-7-1) we consistently outperform both JPEG and JPEG2000 over all bit-rates and metrics. Even though BPG, a competitive non-progressive codec optimized for rate-distortion performance, achieves better reconstruction fidelity (as measured in PSNR) in the low bit-rate regime, our method closely matches BPG in realism (as measured in FID) and even beats BPG in PSNR at higher bit-rates. The theoretical performance of compression with Gaussian diffusion (e.g., VDM) [\(Theis et al.,](#page-12-3) [2022\)](#page-12-3), especially with a high number of steps such as T = 1000, is currently computationally infeasible, both due to the large number of neural function evaluations required, and due the intractable runtime of REC algorithms in the Gaussian case. Still, for reference we report theoretical results both for T = 1000 and T = 20, where the latter uses a smaller and more practical number of diffusion/progressive reconstruction steps.

#### 5.3 IMAGENET 64 × 64

Finally, we present results on the ImageNet 64 × 64 dataset. We train a baseline VDM model with the same architecture as in [\(Kingma et al.,](#page-11-2) [2021\)](#page-11-2), reproducing their reported BPD of around 3.4; we train a UQDM of the same architecture with learned reverse-process variances and T = 4. In addition to the baselines described in the previous section, we also compare with CTC [\(Jeon et al.,](#page-11-7) [2023\)](#page-11-7), a recent progressive neural codec, and CDC [\(Yang & Mandt,](#page-12-7) [2023\)](#page-12-7), a non-progressive neural codec based on a conditional diffusion model that can trade-off between distortion and realism via a hyperparameter p. We separately report results for both p = 0, which purely optimizes the conditional diffusion objective, and p = 0.9, which prioritizes more realistic reconstructions that also jointly minimizes a perceptual loss. For CTC we use pre-trained model checkpoints from the official implementation [\(Jeon et al.,](#page-11-7) [2023\)](#page-11-7); for CDC we fix the architecture but train a new model for each bit-rate v.s. reconstruction quality/realism trade-off.

The results are shown in Figure [4.](#page-8-0) When obtaining progressive reconstructions from denoised predictions, UQDM again outperforms both JPEG and JPEG2000. Our results are comparable to, if not slightly better than, CTC, and even though the reconstruction quality of other codecs plateaus

![](_page_9_Figure_1.jpeg)

![](_page_9_Picture_2.jpeg)

Figure 5: Example progressive reconstructions from UQDM trained with T = 4, obtained with denoised prediction (left) or ancestral sampling (right). The latter avoids blurriness but introduces graininess at low bit-rates, likely because the UQDM is unable to completely capture the data distribution and achieve perfect realism (perfect realism is also difficult to achieve also for Gaussian diffusion, as seen in the rate-realism plot of [\(Theis et al.,](#page-12-3) [2022\)](#page-12-3)). Flow-based reconstructions are qualitatively similar to the denoising-based reconstructions and can be found in Figure [8.](#page-18-1)

at higher bit-rates, our method continues to improve quality and realism gradually, even at higher bit-rates. Refer to Figures [1,](#page-1-1) [5](#page-9-0) and [8](#page-18-1) for qualitative results demonstrating progressive coding and comparison across codecs. At high bit-rates, UQDM preserves details better than other neural codecs. UQDM with denoised predictions tends to introduce blurriness, while ancestral sampling introduces graininess at low bit-rates, likely because the UQDM is unable to completely capture the data distribution and achieve perfect realism. Flow-based denoising matches the distortion of denoised predictions but achieves significantly higher realism as measured by FID. We note that the ideal of perfect realism (i.e., achieving 0 divergence between the data distribution and model's distribution) remains a challenge even for state-of-the-art diffusion models.

### 6 DISCUSSION

In this paper, we presented a new progressive coding scheme based on a novel adaptation of the standard diffusion model. Our universally quantized diffusion model (UQDM) implements the idea of progressive compression with an unconditional diffusion model [\(Theis et al.,](#page-12-3) [2022\)](#page-12-3) but bypasses the intractability of Gaussian channel simulation by using universal quantization [\(Zamir & Feder,](#page-12-4) [1992\)](#page-12-4) instead. We present promising first results that match or outperform classic and neural compression baselines, including a recent progressive neural image compression method [\(Jeon et al.,](#page-11-7) [2023\)](#page-11-7). Given the practical advantages of a progressive neural codec – allowing for dynamic trade-offs between rate, distortion and computation, support for both lossy and lossless compression, and potential for high realism, all in a single model – our approach brings neural compression a step closer towards real-world deployment.

Future work may further improve our approach to close the performance gap to Gaussian diffusion; the latter represents the ideal lossy compression performance under a perfect realism constraint for an approximately Gaussian-distributed data source [\(Theis et al.,](#page-12-3) [2022\)](#page-12-3). This may require more sophisticated methods for computing progressive reconstructions that can achieve higher quality with fewer steps, or exploring different parameterizations of the forward and reverse processes with better theoretical properties. Finally, we expect further improvement in computation efficiency and scalability when combining our method with ideas such as latent diffusion [\(Rombach et al.,](#page-11-14) [2022\)](#page-11-14), distillation [\(Sauer et al.,](#page-11-15) [2024\)](#page-11-15), or consistency models [\(Song et al.,](#page-12-6) [2023\)](#page-12-6).

# ETHICS STATEMENT

Our work focuses on the methodology of a learning-based data compression method, and thus has no direct ethical implications. The deployment of neural lossy compression however carries with it risks of miscommunication and misrepresentation [\(Yang et al.,](#page-12-2) [2023\)](#page-12-2), and needs to carefully analyzed and mitigated with future research.

# REPRODUCIBILITY STATEMENT

We include proofs for all theoretical results introduced in the main text in Appendix [A](#page-13-0) and [B.](#page-16-0) We include further experimental and implementation details (including model architectures and other hyperparameter choices) in Appendix [C.](#page-19-0) Our code can be found at [https://github.com/](https://github.com/mandt-lab/uqdm) [mandt-lab/uqdm](https://github.com/mandt-lab/uqdm).

### ACKNOWLEDGMENTS

Justus Will and Yibo Yang acknowledge support from the HPI Research Center in Machine Learning and Data Science at UC Irvine. Stephan Mandt acknowledges support from the National Science Foundation (NSF) under an NSF CAREER Award IIS-2047418 and IIS-2007719, the NSF LEAP Center, by the Department of Energy under grant DE-SC0022331, the IARPA WRIVA program, the Hasso Plattner Research Center at UCI, the Chan Zuckerberg Initiative, and gifts from Qualcomm and Disney. We thank Kushagra Pandey for feedback on the manuscript.

### REFERENCES


[1] Eirikur Agustsson and Lucas Theis. Universally Quantized Neural Compression. *NeurIPS*, 2020. Johannes Balle, David Minnen, Saurabh Singh, Sung Jin Hwang, and Nick Johnston. Variational ´ Image Compression with a Scale Hyperprior. *ICLR*, 2018. Johannes Balle, Sung Jin Hwang, Nick Johnston, and David Minnen. Tensorflow-compression: Data ´ compression in tensorflow. URL <https://github.com/tensorflow/compression>. Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Cold Diffusion: Inverting Arbitrary Image Transforms without Noise. *NeurIPS*, 2024. Fabrice Bellard. Bpg image format, 2018. URL <https://bellard.org/bpg/>. Marlene Careil, Matthew J Muckley, Jakob Verbeek, and St ` ephane Lathuili ´ ere. Towards Image ` Compression with Perfect Realism at Ultra-low Bitrates. *ICLR*, 2023. Zhengxue Cheng, Heming Sun, Masaru Takeuchi, and Jiro Katto. Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 7939–7948, 2020. Paul Cuff. Communication requirements for generating correlated random variables. In *2008 IEEE International Symposium on Information Theory*, pp. 1393–1397. IEEE, 2008. Zhihao Duan, Ming Lu, Zhan Ma, and Fengqing Zhu. Lossy Image Compression with Quantized Hierarchical VAEs. In *IEEE/CVF Winter Conference on Applications of Computer Vision*, pp. 198–207, 2023. Rick Durrett. *Probability: theory and examples*, volume 49. Cambridge university press, 2019. Gergely Flamich. Greedy Poisson Rejection Sampling. *NeurIPS*, 2024. Gergely Flamich, Marton Havasi, and Jose Miguel Hern ´ andez-Lobato. Compressing Images by ´ Encoding their Latent Representations with Relative Entropy Coding. *NeurIPS*, 2020. Gergely Flamich, Stratis Markou, and Jose Miguel Hern ´ andez-Lobato. Fast Relative Entropy Coding ´ with A\* Coding. *ICML*, 2022.

[2] Gergely Flamich, Stratis Markou, and Jose Miguel Hern ´ andez-Lobato. Faster Relative Entropy ´ Coding with Greedy Rejection Coding. *NeurIPS*, 2024. Daniel Goc and Gergely Flamich. On Channel Simulation with Causal Rejection Samplers. *arXiv preprint arXiv:2401.16579*, 2024. Marton Havasi, Robert Peharz, and Jose Miguel Hern ´ andez-Lobato. Minimal random code learning: ´ Getting bits back from compressed model parameters. *arXiv preprint arXiv:1810.00440*, 2018. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. *NeurIPS*, 2020. Emiel Hoogeboom, Eirikur Agustsson, Fabian Mentzer, Luca Versari, George Toderici, and Lucas Theis. High-Fidelity Image Compression with Score-Based Generative Models. *arXiv preprint arXiv:2305.18231*, 2023. Seungmin Jeon, Kwang Pyo Choi, Youngo Park, and Chang-Su Kim. Context-Based Trit-Plane Coding for Progressive Image Compression. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 14348–14357, 2023. Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational Diffusion Models. *NeurIPS*, 2021. Jae-Han Lee, Seungmin Jeon, Kwang Pyo Choi, Youngo Park, and Chang-Su Kim. DPICT: Deep Progressive Image Compression using Trit-Planes. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 16113–16122, 2022. Jooyoung Lee, Se Yoon Jeong, and Munchurl Kim. DeepHQ: Learned Hierarchical Quantizer for Progressive Deep Image Coding. *arXiv preprint arXiv:2408.12150*, 2024. Yadong Lu, Yinhao Zhu, Yang Yang, Amir Said, and Taco S Cohen. Progressive Neural Image Compression with Nested Quantization and Latent Ordering. In *IEEE International Conference on Image Processing*, pp. 539–543, 2021. David JC MacKay. *Information Theory, Inference and Learning Algorithms*. Cambridge University Press, 2003. Alexander Quinn Nichol and Prafulla Dhariwal. Improved Denoising Diffusion Probabilistic Models. *ICML*, 2021. Kushagra Pandey, Maja Rudolph, and Stephan Mandt. Efficient Integrators for Diffusion Generative Models. *ICLR*, 2023. Yury Polyanskiy and Yihong Wu. Information theory: From coding to learning. *Book draft*, 2022. Lawrence Roberts. Picture Coding using Pseudo-Random Noise. *IRE Transactions on Information Theory*, pp. 145–154, 1962. Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ¨ Resolution Image Synthesis with Latent Diffusion Models. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 10684–10695, 2022. Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation. *arXiv preprint arXiv:2403.12015*, 2024. Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep Unsupervised Learning using Nonequilibrium Thermodynamics. *ICML*, 2015. Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models. *ICLR*, 2021a. Yang Song and Stefano Ermon. Generative Modeling by Estimating Gradients of the Data Distribution. *NeurIPS*, 2019.

[3] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-Based Generative Modeling through Stochastic Differential Equations. *ICLR*, 2021b. Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In *International Conference on Learning Representations*, 2021c. Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency Models. *ICML*, 2023. Lucas Theis and Noureldin Y Ahmed. Algorithms for the Communication of Samples. *ICML*, 2022. Lucas Theis, Tim Salimans, Matthew D Hoffman, and Fabian Mentzer. Lossy Compression with Gaussian Diffusion. *arXiv preprint arXiv:2206.08889*, 2022. James Townsend, Thomas Bird, Julius Kunze, and David Barber. HiLLoc: Lossless Image Compression with Hierarchical Latent Variable Models. *ICLR*, 2024. Pascal Vincent. A Connection Between Score Matching and Denoising Autoencoders. *Neural Computation*, pp. 1661–1674, 2011. Ruihan Yang and Stephan Mandt. Lossy Image Compression with Conditional Diffusion Models. *NeurIPS*, 2023. Yibo Yang, Stephan Mandt, Lucas Theis, et al. An Introduction to Neural Data Compression. *Foundations and Trends in Computer Graphics and Vision*, pp. 113–200, 2023.

[4] R. Zamir and M. Feder. On Universal Quantization by Randomized Uniform/Lattice Quantizers. *IEEE Transactions on Information Theory*, pp. 428–436, 1992.
### APPENDIX

### A FORWARD PROCESS DETAILS

### A.1 GAUSSIAN (DDPM/VDM)

For completeness and reference, we restate the forward process and related conditionals given in [\(Kingma et al.,](#page-11-2) [2021\)](#page-11-2). The forward process is defined by

$$q(\mathbf{z}_t|\mathbf{x}) := \mathcal{N}(\alpha_t \mathbf{x}, \sigma_t^2 \mathbf{I}),$$

where α<sup>t</sup> and σ 2 t are positive scalar-valued functions of t. As in [\(Kingma et al.,](#page-11-2) [2021\)](#page-11-2), we define the following notation shorthand which are used in the rest of the appendix: for any s < t, let

$$\alpha_{t|s} := \frac{\alpha_t}{\alpha_s}, \quad \sigma_{t|s}^2 := \sigma_t^2 - \frac{\alpha_t^2}{\alpha_s^2} \sigma_s^2, \quad b_{t|s} := \frac{\alpha_t}{\alpha_s} \frac{\sigma_s^2}{\sigma_t^2}, \quad c_{t|s} := \sigma_{t|s}^2 \frac{\alpha_s}{\sigma_t^2}, \quad \beta_{t|s} := \sigma_{t|s} \frac{\sigma_s}{\sigma_t}.$$

By properties of the Gaussian distribution, it can be shown that for any 0 ≤ s < t ≤ T,

$$q(\mathbf{z}_t | \mathbf{z}_s) = \mathcal{N}(\alpha_{t|s} \mathbf{x}, \sigma_{t|s}^2 \mathbf{I}),$$

$$q(\mathbf{z}_s | \mathbf{z}_t, \mathbf{x}) = \mathcal{N}(b_{t|s} \mathbf{z}_t + c_{t|s} \mathbf{x}, \beta_{t|s}^2 \mathbf{I}),$$

In particular,

$$\begin{aligned} q(\mathbf{z}_{t-1} | \mathbf{z}_t, \mathbf{x}) &= \mathcal{N}(b_{t|t-1} \mathbf{z}_t + c_{t|t-1} \mathbf{x}, \beta_{t|t-1}^2 \mathbf{I}), \\ q(\mathbf{z}_t | \mathbf{z}_T, \mathbf{x}) &= \mathcal{N}(b_{T|t} \mathbf{z}_t + c_{T|t} \mathbf{x}, \beta_{T|t}^2 \mathbf{I}), \end{aligned}$$

and we can use the reparameterization trick to write

$$\begin{aligned}\mathbf{z}_{t-1} &= b_{t|t-1} \mathbf{z}_t + c_{t|t-1} \mathbf{x} + \beta_{t|t-1} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \\ \mathbf{z}_t &= b_{T|t} \mathbf{z}_T + c_{T|t} \mathbf{x} + \beta_{T|t} \epsilon_T, \quad \epsilon_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})\end{aligned}$$

### A.2 UNIFORM (OURS)

Our forward process is specified by q(z<sup>T</sup> |x) and q(zt−1|zt, x) for each t, and closely follows that of the Gaussian diffusion. We set q(z<sup>T</sup> |x) to be the same as in the Gaussian case, i.e.,

$$q(\mathbf{z}_T|\mathbf{x}) := \mathcal{N}(\alpha_T \mathbf{x}, \sigma_T^2 \mathbf{I}),$$

and q(zt−1|zt, x) to be a uniform with the same mean and variance as in the Gaussian case, such that

$$q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x}) := \mathcal{U}(b_{t|t-1}\mathbf{z}_t + c_{t|t-1}\mathbf{x} - \sqrt{3}\beta_{t|t-1}, b_{t|t-1}\mathbf{z}_t + c_{t|t-1}\mathbf{x} + \sqrt{3}\beta_{t|t-1}),$$

or in other words,

$$\mathbf{z}_{t-1} = b_{t|t-1}\mathbf{z}_t + c_{t|t-1}\mathbf{x} + \sqrt{12}\beta_{t|t-1}\mathbf{u}_t, \quad \mathbf{u}_t \sim \mathcal{U}(-1/2, 1/2).$$

In the notation of eq. [\(4\)](#page-4-0) this corresponds to letting <sup>b</sup>(t) = <sup>b</sup>t|t−1, <sup>c</sup>(t) = <sup>c</sup>t|t−1, ∆(t) = √ 12βt|t−1. It follows by algebraic manipulation that

$$\mathbf{z}_t = b_{T|t} \mathbf{z}_T + c_{T|t} \mathbf{x} + \underbrace{\sum_{v=t+1}^T \sqrt{12} \delta_{v|t} \mathbf{u}_v}_{:=\omega_t}, \quad (8)$$

where

$$\mathbf{u}_v \sim \mathcal{U}(-1/2, 1/2), v = t + 1, \dots, T$$

are independent uniform noise variables, and

$$\delta_{v|t} := \beta_{v|v-1} \prod_{j=t+1}^{v-1} b_{j|j-1} = \frac{\sigma_t^2}{\alpha_t} \sqrt{\text{SNR}(v-1) - \text{SNR}(v)},$$

where

$$\text{SNR}(s) := \frac{\alpha_s^2}{\sigma_s^2}.$$

It can be verified that

$$\mathbb{E}[\boldsymbol{\omega}_t] = \mathbf{0},$$

$$\text{Var}(\boldsymbol{\omega}_t) = \sum_{v=t+1}^T \delta_{v|t}^2 \mathbf{I} = \frac{\sigma_t^4}{\alpha_t^2} [\text{SNR}(t) - \text{SNR}(T)] \mathbf{I} = \beta_{T|t}^2 \mathbf{I},$$

or in other words, at any step t our forward-process "posterior" distribution q(zt|z<sup>T</sup> , x) has the same mean and variance as in the Gaussian case.

### A.3 CONVERGENCE TO THE GAUSSIAN CASE

We show that both forward processes are equivalent in the continuous-time limit. To allow comparison across different number of steps T, we suppose that α<sup>t</sup> and σ<sup>t</sup> are obtained from continuous-time schedules α(·) : [0, 1] → <sup>R</sup> <sup>+</sup> and σ(·) : [0, 1] → <sup>R</sup> <sup>+</sup> (which were fixed ahead of time), such that α<sup>t</sup> := α(t/T) and σ<sup>t</sup> := σ(t/T) for t = 0, . . . , T, for any choice of T. As in VDM [\(Kingma et al.,](#page-11-2) [2021\)](#page-11-2), we assume that the continuous-time signal-to-noise ratio snr(·) := α(·) <sup>2</sup>/σ(·) 2 is strictly monotonically decreasing.

To obtain the continuous-time limit, we hold the "continuous" time ρ := <sup>t</sup> T fixed for some ρ ∈ [0, 1), and let T → ∞ (or equivalently, let the time discretization <sup>1</sup> <sup>T</sup> → 0). We note that the quantities bT|<sup>t</sup> , cT|<sup>t</sup> , β 2 T|t only depend on ρ, and are thus well-defined when we hold ρ fixed and let T → ∞:

$$\begin{aligned}
b_{T|t} &= \frac{\alpha_T}{\alpha_t} \frac{\sigma_t^2}{\sigma_T^2} = \frac{\alpha(1)}{\alpha(\rho)} \frac{\sigma^2(\rho)}{\sigma^2(1)}, \\
c_{T|t} &= \left( \sigma^2(1) - \frac{\alpha^2(1)}{\alpha^2(\rho)} \sigma^2(\rho) \right) \frac{\alpha(\rho)}{\sigma^2(1)}, \\
\beta_{T|t}^2 &= \left( \sigma^2(1) - \frac{\alpha^2(1)}{\alpha^2(\rho)} \sigma^2(\rho) \right) \frac{\sigma^2(\rho)}{\sigma^2(1)} = \frac{\sigma^4(\rho)}{\alpha^2(\rho)} (\text{snr}(\rho) - \text{snr}(1)).
\end{aligned}$$

We start by showing that our q(zt|z<sup>T</sup> , x) converges to the corresponding Gaussian distribution in VDM in the continuous-time limit, which in turn implies the convergence of our q(zt|x) to the corresponding Gaussian distribution in VDM.

#### Theorem A.1.

*For every fixed* ρ := <sup>t</sup> T ∈ [0, 1)*,* q(zt|z<sup>T</sup> , x) <sup>d</sup>−→ N (bT|<sup>t</sup> <sup>z</sup><sup>T</sup> <sup>+</sup> <sup>c</sup>T|<sup>t</sup> <sup>x</sup>, β<sup>2</sup> T|t I) *as* T → ∞*.*

#### *Proof.*

Recall the following fact in the forward process of UQDM (see eq. [\(8\)](#page-13-1)):

$$\mathbf{z}_t = b_{T|t} \mathbf{z}_T + c_{T|t} \mathbf{x} + \underbrace{\sum_{v=t+1}^T \sqrt{12} \delta_{v|t} \mathbf{u}_v}_{:=\omega_t}, \quad (9)$$

where

$$\mathbf{u}_v \sim \mathcal{U}(-1/2, 1/2), v = t + 1, \dots, T$$

are independent uniform noise variables, and

$$\delta_{v|t} := \beta_{v|v-1} \prod_{j=t+1}^{v-1} b_{j|j-1} = \frac{\sigma_t^2}{\alpha_t} \sqrt{\text{SNR}(v-1) - \text{SNR}(v)},$$

$$\text{SNR}(s) := \frac{\alpha_s^2}{\sigma_s^2}.$$

It therefore suffices to show that ω<sup>t</sup> converges in distribution to N (0, β<sup>2</sup> T|t I) in the continuous-time limit. Since the different coordinates of ω<sup>t</sup> are independent, we focus on a single coordinate and study the continuous-time limit of a scalar Ωt, given by a sum of scaled uniform variables,

$$\Omega_t := \sum_{v=t+1}^T \left( \frac{\sqrt{12}\sigma^2(\rho)}{\alpha(\rho)} \sqrt{\text{snr}\left(\frac{v-1}{T}\right) - \text{snr}\left(\frac{v}{T}\right)} \right) U_v \quad (10)$$

$$= \sum_{j=1}^n \left( \frac{\sqrt{12}\sigma^2(\rho)}{\alpha(\rho)} \sqrt{\operatorname{snr}\left(\rho + \frac{j-1}{T}\right)} - \operatorname{snr}\left(\rho + \frac{j}{T}\right) \right) U_j \quad (11)$$

where U<sup>j</sup> 's are i.i.d. U(−1/2, <sup>1</sup>/2) variables, and in the last step we set n := n(T) = T − t and switched the summation index to j = v − t.

Define a triangular array of variables by

$$X_{n,j} = \left( \frac{\sqrt{12}\sigma^2(\rho)}{\alpha(\rho)} \sqrt{\operatorname{snr}(\rho + \frac{j-1}{T})} - \operatorname{snr}(\rho + \frac{j}{T}) \right) U_j,$$

for j = 1, 2, ..., n and for n ∈ N <sup>+</sup>. For each n, {Xn,j}j=1,2,...,n are independent variables with <sup>E</sup>[Xn,j ] = 0, and it can be verified that

$$\sum_{j=1}^n \mathbb{E}[X_{n,j}^2] = \text{Var}(\Omega_t) = \beta_{T|t}^2 = \frac{\sigma^4(\rho)}{\alpha^2(\rho)} (\text{snr}(\rho) - \text{snr}(1)).$$

To apply the Lindeberg-Feller central limit theorem [\(Durrett,](#page-10-12) [2019,](#page-10-12) Theorem 3.4.10) to Ω<sup>t</sup> = Xn,<sup>1</sup> + ... + Xn,n, it remains to verify the condition

$$\forall \epsilon > 0, \lim_{n \rightarrow \infty} \sum_{j=1}^n \mathbb{E}[X_{n,j}^2 \mathbf{1}\{|X_{n,j}| > \epsilon\}] = 0.$$

Let ϵ > 0. Since snr(·) is continuous on a compact domain [0, 1], it is also uniformly continuous; then there exists a δ such that

$$|\text{snr}(x_1) - \text{snr}(x_2)| < \left( \frac{\epsilon \alpha(\rho)}{\sqrt{12}\sigma^2(\rho)} \right)^2, \quad \forall x_1, x_2, |x_1 - x_2| < \delta. \quad (12)$$

Let T (and thus n = T − t) become sufficiently large such that <sup>1</sup> <sup>T</sup> < δ. Then, for all such T (and thus n) sufficiently large, and for all j, it holds that 1{|Xn,j | > ϵ} = 0 almost everywhere:

$$\mathbb{P}(|X_{n,j}| > \epsilon) = \mathbb{P}\left(\left(\frac{\sqrt{12}\sigma^2(\rho)}{\alpha(\rho)}\sqrt{\text{snr}(\rho + \frac{j-1}{T})} - \text{snr}(\rho + \frac{j}{T})\right) |U_j| > \epsilon\right) \quad (13)$$

$$= \mathbb{P} \left( |U_j| > \frac{\epsilon \alpha(\rho)}{\sqrt{12}\sigma^2(\rho)} \frac{1}{\sqrt{\operatorname{snr}(\rho + \frac{j-1}{T}) - \operatorname{snr}(\rho + \frac{j}{T})}} \right) \quad (14)$$

$$\begin{aligned} & \text{by eq. (12)} \\ & \leq \mathbb{P}(|U_j| > 1) \\ & = 0 \end{aligned} \tag{15}$$
(16)

since U<sup>j</sup> ∼ U(−1/2, <sup>1</sup>/2), and it follows that

$$\mathbb{E}[X_{n,j}^2 \mathbf{1}\{|X_{n,j}| > \epsilon\}] = 0$$

for all j for all sufficiently large n. We conclude by the Lindeberg-Feller theorem that

$$\Omega_t = X_{n,1} + \dots + X_{n,n} \xrightarrow{d} \mathcal{N}(0, \beta_{T|t}^2)$$

#### Corollary A.1.1.

*If we assume* σ<sup>T</sup> *and* α<sup>T</sup> *to be constants, then for every* t*,* q(zt|x) <sup>d</sup>−→ N (αtx, σ<sup>2</sup> t I) *as* T → ∞*, that is, our forward model approaches the Gaussian forward process of VDM with an increasing number of diffusion steps.*

*Proof.* As q(z<sup>T</sup> |x) = N (α<sup>T</sup> x, σ<sup>2</sup> T I) does not depend on T, the joint distribution q(zt, z<sup>T</sup> |x) = q(zt|z<sup>T</sup> , x)q(z<sup>T</sup> |x) converges in distribution, which in turn implies convergence of q(zt|x). The statement then follows from the identity

$$\mathcal{N}(\mathbf{z}_t; \alpha_t \mathbf{x}, \sigma_t^2 \mathbf{I}) = \int \mathcal{N}(\mathbf{z}_t; b_{T|t} \mathbf{z}_T + c_{T|t} \mathbf{x}, \beta_{T|t}^2 \mathbf{I}) \mathcal{N}(\mathbf{z}_T; \alpha_T \mathbf{x}, \sigma_T^2 \mathbf{I}) d\mathbf{z}_T.$$

### B BACKWARD PROCESS DETAILS AND RATE ESTIMATES

### B.1 GAUSSIAN (DDPM/VDM)

[Kingma et al.](#page-11-2) [\(2021\)](#page-11-2) set p(zt−1|zt) := q(zt−1|zt, x = xˆt) = N (bt|t−<sup>1</sup> z<sup>t</sup> + ct|t−<sup>1</sup> xˆt, β<sup>2</sup> t|t−1 I) which yields

$$\begin{aligned} L_{t-1} &= \text{KL}(\mathcal{N}(b_{t|t-1} \mathbf{z}_t + c_{t|t-1} \mathbf{x}, \beta_{t|t-1}^2 \mathbf{I}) \| \mathcal{N}(b_{t|t-1} \mathbf{z}_t + c_{t|t-1} \hat{\mathbf{x}}_t, \beta_{t|t-1}^2 \mathbf{I})) \\ &= \frac{1}{2} \frac{c_{t|t-1}^2}{\beta_{t|t-1}^2} \|\mathbf{x} - \hat{\mathbf{x}}_t\|_2^2 = \frac{1}{2} (\text{SNR}(t-1) - \text{SNR}(t)) \|\mathbf{x} - \hat{\mathbf{x}}_t\|_2^2. \end{aligned}$$

We have that Lt−<sup>1</sup> → 0 as T → ∞, due to the continuity of SNR(· /T) = snr(·) = α(·) <sup>2</sup>/σ(·) 2 .

### B.2 UNIFORM (OURS)

Recall that we choose each coordinate of the reverse-process model p(zt−1|zt) to have the density

$$\begin{aligned}
p(\mathbf{z}_{t-1} | \mathbf{z}_t)_i &:= g_t(z) \star \mathcal{U}(z; -\Delta_t/2, \Delta_t/2) \\
&= \frac{1}{\Delta_t} \int_{z-\Delta_t/2}^{z+\Delta_t/2} g_t(z) dz = \frac{1}{\Delta_t} (G_t(z + \Delta_t/2) - G_t(z - \Delta_t/2)),
\end{aligned}$$

where G<sup>t</sup> and g<sup>t</sup> are the cdf and pdf of a distribution with mean µˆ<sup>t</sup> := bt|t−1z + ct|t−1xˆ and variance σ 2 g , z := (zt)<sup>i</sup> , x := x<sup>i</sup> , and xˆ := xˆθ(zt;t)<sup>i</sup> . Using the shorthand µ<sup>t</sup> := bt|t−1z + ct|t−1x we can derive the rate associated with the ith coordinate

$$\begin{aligned} L_{t-1} &= \text{KL}(\mathcal{U}(z; \mu_t - \Delta_t/2, \mu_t + \Delta_t/2) \parallel g_t(z) \star \mathcal{U}(z; -\Delta/2_t, \Delta/2_t)) \\ &= \frac{1}{\Delta_t} \int_{\mu_t - \Delta_t/2}^{\mu_t + \Delta_t/2} \log \frac{\frac{1}{\Delta_t} \mathbf{1}_{[\mu_t - \Delta_t/2, \mu_t + \Delta_t/2]}(z)}{\frac{1}{\Delta_t} (G_t(z + \Delta_t/2) - G_t(z - \Delta_t/2))} dz \\ &= \frac{1}{\Delta_t} \int_{-\Delta_t/2}^{\Delta_t/2} - \underbrace{\log(G_t(z + \mu_t + \Delta_t/2) - G_t(z + \mu_t - \Delta_t/2))}_{:= h(z)} dz. \end{aligned}$$

To gain some intuition for this rate, note that h(z) is lowest when most of the probability mass of G<sup>t</sup> is concentrated tightly around z + µt, which is the case when |µ<sup>t</sup> − µˆt| is small. Specifically, if G<sup>t</sup> is in a distributional family with a standardized cdf G<sup>0</sup> such that Gt(z) = G0((z − µˆt)/σg) then

$$G_t(z + \mu_t + \Delta_t/2) - G_t(z + \mu_t - \Delta_t/2) \rightarrow \begin{cases} 1 & \text{if } |z + \mu_t - \hat{\mu}_t| < \Delta_t/2 \\ G_0(0) & \text{if } |z - \mu_t - \hat{\mu}_t| = \Delta_t/2 \\ 0 & \text{else} \end{cases}$$

as σ<sup>g</sup> → 0. Thus, if |µ<sup>t</sup> − µˆt| ≪ <sup>∆</sup>t/2, we obtain improved bit-rates for σ<sup>g</sup> that are small (relative to ∆t). On the other hand, as almost certainly |µ<sup>t</sup> − µˆt| > 0, we can't choose arbitrarily small σ<sup>g</sup> because in that case both max(−h(−<sup>∆</sup>t/2), −h(<sup>∆</sup>t/2)) → ∞ and Lt−<sup>1</sup> → ∞ as σ<sup>g</sup> → 0. This further motivates the merit of learning the backwards variances as σ 2 <sup>g</sup> = sθ(z)β 2 <sup>t</sup>|t−<sup>1</sup> <sup>=</sup> <sup>s</sup>θ(z)∆<sup>2</sup> <sup>t</sup> /12, allowing them to adapt to |µ<sup>t</sup> − µˆt|. Conversely, by the mean value theorem, there exists one c ∈ (−<sup>∆</sup>t/2, <sup>∆</sup>t/2) so that

$$G_t(z + \mu_t + \Delta_t/2) - G_t(z + \mu_t - \Delta_t/2) = \Delta_t g_t(z + \mu_t + c) \approx \Delta_t g_t(z + \mu_t)$$

where the last approximation becomes more accurate for larger σg. If we further assume that G<sup>t</sup> is Gaussian (or sufficiently similar) h(t) becomes approximately quadratic. In that case we study

$$h(z) \approx \left(1 - \frac{4z^2}{\Delta_t^2}\right) h(0) + \frac{2z^2 - \Delta_t z}{\Delta_t^2} h(-\Delta_t/2) + \frac{2z^2 + \Delta_t z}{\Delta_t^2} h(\Delta_t/2),$$

a quadratic function that exactly matches h at values z ∈ {−<sup>∆</sup>t/2, 0, <sup>∆</sup>t/2}. Finally, this results in

$$\begin{aligned} L_{t-1} &\approx \frac{1}{\Delta_t} \left[ \frac{2}{\Delta_t^2} (h(-\Delta_t/2) + h(\Delta_t/2) - 2h(0)) \int_{-\Delta_t/2}^{\Delta_t/2} z^2 dz + \frac{1}{\Delta_t} (h(\Delta_t/2) - h(\Delta_t/2)) \int_{-\Delta_t/2}^{\Delta_t/2} z dz + \Delta_t h(0) \right] \\ &= -\frac{1}{6} [4h(0) + h(-\Delta_t/2) + h(\Delta_t/2)] \geq \frac{1}{3} \log(2), \end{aligned}$$

where the last equality uses h(z) ≤ 0 and h(−<sup>∆</sup>t/2) + h(<sup>∆</sup>t/2) ≤ log(0.25) which follow from the fact that G<sup>t</sup> is a cdf. Empirically we note that this estimate is very accurate as long as σ 2 <sup>g</sup> ≥ β 2 t|t−1 , demonstrating that simply matching moments as in VDM will occur a constant overhead for each diffusion step. As seen in Figure [2,](#page-7-0) this can be partly mitigated with smaller σ 2 <sup>g</sup> but increasing the number of diffusion steps T might still lead to an increase in ELBO. Numerical integration of Lt−<sup>1</sup> confirms that if σ 2 g is close to the optimal choice of σ<sup>g</sup> ≈ |µ<sup>t</sup> − µˆt|, Lt−<sup>1</sup> → 0 as T → ∞ as in the Gaussian case.

#### B.3 FLOW-BASED RECONSTRUCTIONS

Given an intermediate latent zt, ancestral sampling yields an intermediate lossy reconstruction xˆ ∼ p(x|zt) that requires us to repeatedly sample from the conditional p(zt−1|zt) until finally obtaining a reconstruction from z<sup>0</sup> with the help of p(x|z0). This is equivalent to approximately solving a reverse SDE [\(Song et al.,](#page-12-9) [2021c\)](#page-12-9) and introduces additional noise during inference, which can make reconstructions grainy for diffusion models with a small number of steps, as can be seen in Figure [5.](#page-9-0) [Song et al.](#page-12-9) [\(2021c\)](#page-12-9) further note that an alternative approximate solution to the SDE can be obtained by deterministically reversing a "probability-flow" ODE (see also [Theis et al.](#page-12-3) [\(2022\)](#page-12-3)). Specifically, this involves repeatedly evaluating zt−<sup>1</sup> = f(zt, t), where f for VDM is defined as

$$f(\mathbf{z}_t, t) = \frac{\alpha_{t-1}}{\alpha_t} \mathbf{z}_t + \left( \sigma_{t-1} - \frac{\alpha_{t-1}}{\alpha_t} \sigma_t \right) \hat{\epsilon}_t = \frac{\sigma_{t-1}}{\sigma_t} \mathbf{z}_t + \left( \alpha_{t-1} - \frac{\sigma_{t-1}}{\sigma_t} \alpha_t \right) \hat{\mathbf{x}}_t, \quad (17)$$

recovering the same process defined in [\(Song et al.,](#page-11-12) [2021a\)](#page-11-12). The equivalence of the continuous limit in Corollary [A.1.1,](#page-15-1) suggests that the discrete-time backward processes of UQDM and VDM are similar enough in the sense that eq. [\(17\)](#page-17-1) also approximately solves the implied reverse SDE of UQDM. Thus we use eq. [\(17\)](#page-17-1) to obtain flow-based reconstructions for both VDM and UQDM.

![](_page_18_Figure_1.jpeg)

Figure 7: Unconditional samples from UQDM models trained with varying T on the swirl dataset. The sample quality improves with larger T; however the compression performance becomes worse after T > 5, as discussed in Section [5.](#page-6-0)

![](_page_18_Figure_3.jpeg)

Figure 6: Left: 1000 samples from the toy swirl source. Right: Additional results on swirl data. We examined the compression performance of applying universal quantization to a pre-trained VDM model; conceptually this is equivalent to When using fixed reverse-process variances, we can directly re-use weights from a pretrained VDM model (orange), which achieves comparable results to training a UQDM model from scratch, even for a smaller number of timesteps.

![](_page_18_Figure_5.jpeg)

Figure 8: Additional results on ImageNet 64x64 data. Left: Example progressive reconstructions from UQDM trained with T = 4, obtained with flow-based denoising, as in Figure [5.](#page-9-0) Flow-based reconstructions achieve similar distortion (as meassured with PSNR) than denoised predictions at higher fidelity (as meassured with FID). Right: Ablation of the influence of model size on validation loss. Bars are labeled with the number of parameters for each model. Increasing the size of the denoising network allows for smaller bitrates.

### C ADDITIONAL EXPERIMENTAL RESULTS

#### C.1 SWIRL DATA

We use the swirl data from the codebase of [\(Kingma et al.,](#page-11-2) [2021\)](#page-11-2); Figure [6](#page-18-2) shows 1000 samples from the toy data source. We use the same denoisng network xˆ<sup>θ</sup> as in the official implementation,[<sup>2</sup>](#page-19-2) which consists of 2 hidden layers with 512 units each. Figure [6](#page-18-2) highlights the consequence of Corollary [A.1.1:](#page-15-1) Because VDM and UQDM share the same continuous limit, we can use the weights of a pretrained VDM to obtain comparable UQDM results as a UQDM model that has been trained from scratch.

#### C.2 CIFAR10

We use a scaled-down version of the denoising network from the VDM paper [\(Kingma et al.,](#page-11-2) [2021\)](#page-11-2) for faster experimentation. We use a U-Net of depth 8, consisting of 8 ResNet blocks in the forward direction and 9 ResNet blocks in the reverse direction, with a single attention layer and two additional ResNet blocks in the middle. We keep the number of channels constant throughout at 128.

We verified that our UQDM implementation based on tensorflow-compression achieves file size close the theoretical NELBO. When compressing a single 32x32 CIFAR image, we observe file size overhead ≤ 3% of the theoretical NELBO. In terms of computation speed, it takes our model with fixed reverse-process variance less than 1 second to encode or decode a CIFAR image, either on CPU or GPU,[<sup>3</sup>](#page-19-3) likely because the very few neural-network evaluations required (T = 4). For our model with learned reverse-process variance, however, it takes about 5 minutes to compress or decompress a CIFAR image, with nearly all of the compute time spent on a single CPU core. This is because with learned reverse-process variance, each latent dimension has a different predicted variance, and a separate CDF table needs to be built for each latent dimension during entropy coding; the tensorflow-compression library builds the CDF table for each coordinate in a naive for-loop rather than in parallel. Thus we expect the coding speed to be dramatically faster with a parallel implementation of entropy coding, e.g., using the DietGPU[<sup>4</sup>](#page-19-4) library.

#### C.3 IMAGENET 64 × 64

We use the same denoising network as in the VDM paper [\(Kingma et al.,](#page-11-2) [2021\)](#page-11-2). We use a U-Net of depth 64, consisting of 64 ResNet blocks in the forward direction and 65 ResNet blocks in the reverse direction, with a single attention layer and two additional ResNet blocks in the middle. We keep the number of channels constant throughout at 256. To investigate the impact of the size of the denoising network, in addition to this configuration with 237M parameters we call UQDM-big, we also run experiments with three smaller networks with 32 ResNet blocks and 128 channels (UQDMmedium, 122M parameters), 8 ResNet blocks and 64 channels (UQDM-small, 2M parameters), and 1 ResNet block and 32 channels (UQDM-tiny, 127K parameters), respectively. Smaller network are significantly faster and more resource-efficient but will naturally suffer from higher bitrates, as can be seen in Figure [8.](#page-18-1)

The required number of FLOPS per pixel for encoding and decoding is strongly dominated by the number of neural function evaluations (NFE) of our denoising network which depends on how soon we stop the encoding and decoding process. For lossless compression we have to multiple the FLOPS per NFE with T which is equal to 4 in our case. For lossy compression after t steps, with lossy reconstructions obtained through a denoised prediction, we obtain the required FLOPS for encoding and decoding by multiplying with t and t + 1 respectively. The FLOPS per NFE depend on the network size, our investigated model size require 389K, 2.3M, 105M, and 204M FLOPS per pixel, in order from smallest to biggest model.

<sup>2</sup>[https://github.com/google-research/vdm/blob/main/colab/2D\\_VDM\\_Example.](https://github.com/google-research/vdm/blob/main/colab/2D_VDM_Example.ipynb) [ipynb](https://github.com/google-research/vdm/blob/main/colab/2D_VDM_Example.ipynb)

<sup>3</sup>Around 0.6 s for encoding and 0.5 s for decoding on Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz CPU; 0.5 s for encoding and 0.3 s for decoding on a single Quadro RTX 8000 GPU.

<sup>4</sup><https://github.com/facebookresearch/dietgpu>

Figures [9](#page-21-0) and [10](#page-21-0) show more example reconstructions from several traditional and neural codecs, similar to Figure [1.](#page-1-1) At lower bitrates the artifacts each compression codecs introduces become more visible.

![](_page_21_Picture_1.jpeg)

![](_page_21_Figure_2.jpeg)

Figure 9: Additional example reconstructions , chosen at roughly similar (high) bitrates.

![](_page_21_Picture_4.jpeg)

![](_page_21_Figure_5.jpeg)

Figure 10: Additional example reconstructions , chosen at roughly similar (low) bitrates.