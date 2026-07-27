000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Extending Flexibility Of Image Coding En- Hancement Framework For Iots

Anonymous authors Paper under double-blind review

## Abstract

Neural image compression, necessary in various edge-device scenarios, suffers from its heavy encode-decode structures and inflexible compression level switch.

The primary issue is that the computational and storage capabilities of edge devices are weaker than those of servers, preventing them from handling the same amount of computation and storage. One solution is to downsample images and reconstruct them on the receiver side; however, current methods uniformly downsample the image and limit flexibility in compression levels. We take a step to break up this paradigm by proposing a conditional uniform-based sampler that allows for flexible image size reduction and reconstruction. Building on this, we introduce a lightweight transformer-based reconstruction structure to further reduce the reconstruction load on the receiver side. Extensive evaluations conducted on a real-world testbed demonstrate multiple advantages of our system over existing compression techniques, especially in terms of adaptability to different compression levels, computational efficiency, and image reconstruction quality.

## 1 Introduction

The need for advanced lossy image compression is raised by the explosive development of edge devices equipped with high-resolution cameras, such as industrial-inspections (George et al., 2019), wildlife observation (wil, 2023), and autonomous driving (Ananthanarayanan et al., 2017). Neural- Network (NN) based compressor can satisfy this need, which outperform traditional image compression techniques like JPEG (Group, 1986) and BPG (Bellard, 2014). However, due to its heavy, symmetric encoding and decoding structure and inflexible compression rate adjustment, current NN-based methods have not yielded practical use on resource-constrained edge devices (Dasari et al., 2022). Given the paucity of computational ability on edge devices in general (Fut, 2020; ope, 2018; aws, 2023; Li et al., 2023a;b), a huge gap would exist in the edge compression/decompress and transmission latency. As shown in Fig. 1a, encoding an image can take as long as 18 seconds on high-end devices like the NVIDIA Jetson TX2. Downsampling image size at the sender and restoring it on the receiver is one way to alleviate this problem (Yin et al., 2023; Cheng et al., 2024). However, these solutions usually employ super-resolution, which uniformly downsample and restore images to fixed sizes, lacking flexibility for dynamic and complex compression needs in real-world applications. We take a fresh look at this problem and introduce Easz, a lightweight compression enhancement framework that operates efficiently at the edge-sender with near-zero computational demand, while also maintaining efficiency on the receiver. Easz is compatible with all existing compression algorithms. The intuition of Easz is an implicit assumption undermined in current solutions: the image need to be uniformly downsampled. Easz includes an erase-and-squeeze process, which relaxes this assumption by designing a conditional uniform-based sampler. This technique provides a more adaptable and fine-grained compression level but also loses the chance to employ efficient reconstruction through convolution or the fast Fourier transform techniques. We then propose a receiver-side lightweight transformer architecture for efficient, high-quality reconstruction of erased patches. This involves a two-stage image patchify process to limit the scope of attention correlation calculations and a four-layer transformer model for pixel-level local image reconstruction. As shown in Figure 1b, Easz surpasses both the NN-based compressor and the traditional compressor. The key contributions of this paper are:
1

Balle et al.

Balle et al. Minnen et al.

Cheng et al.

11600 18015 17952 Edge Encoding Time(ms)
103 104 15 20 30 Easz 10 4 MBT
Valu es (l og sc ale)
Co mpressi on Ratio Cheng-Anchor Bmshj-factorized Bmshj-hyperprior Gap BPG

1361 10 3 WEBP
552 413 151 286374 162 JPEG
163 152 Transmission Latency (ms) Load Latency (ms) Edge Encode Latency (ms)
(a)
(b)
- Generalized Erase-and-Squeeze Process: A new paradigm is introduced that offers more refined and flexible image reduction ratios;
- Receiver-Side Lightweight Transformer Architecture: A lightweight(8.7MB) transformer architecture is designed for efficient and high-quality reconstruction of erased patches;
- Compatibility with Existing Algorithms: Easz is compatible with all existing image compression algorithms and can also function independently;
- Enhanced Compression Flexibility and Efficiency: Easz offers significant compression flexibility and efficiency improvements. For the sender-side, Easz requires almost no additional computational cost with a controllable compression ratio, and on the recieverside, Easz's reconstruction model is also lightweight, making it well-suited for real-world applications with varying and complex compression needs.

## 2 Related Work

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 Learned-based image compression is experiencing significant growth, with advancements in endto-end training, hyperprior structures, entropy models, and encoder-decoder improvements Notable developments include the introduction of auto-regressive components (Minnen et al., 2018), Gaussian Mixture Models for probability estimation (Cheng et al., 2020), and a general-purpose lossless compression paradigm using lightweight neural networks (Mao et al., 2022b;a; 2023). Attention mechanisms have been incorporated through Informer (Jun-Hyuk et al., 2022), while Transformers and Swin architectures are replacing traditional CNNs in encoding/decoding tasks (Yinhao Zhu and Cohen, 2022; He et al., 2021). Despite progress, real-world applications still face challenges such as inflexibility in switching models and high latency at the edge. Deep-learning-based compression methods take about 1∼20 second per image (512x768) on NVIDIA Jetson TX2, and many real-life endpoints are less potent than the TX2
(considering Raspberry Pi 4) but still need to compress images. A primary issue is that most NN-
based image compressors require a model switch when changing compression levels. One approach involves downsampling images at the edge and using super-resolution techniques to reconstruct them on the server (Yin et al., 2023; Cheng et al., 2024). These methods reduce computational load at the edge, but applying super-resolution directly in this context results in an inflexible downsizing rate and can degrade reconstruction performance (Laroche et al., 2023; Jin et al., 2022).

## 3 System Design

The paper presents Easz, a novel edge-optimized image compression framework. It applies the erase-and-squeeze technique at the sender with a lightweight transformer-powered reconstruction 2

## 3.1 Implicit Assumption In Previous Methods

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

## 3.2 Erase-And-Squeeze Algorithm 3.2.1 Problem Formulation

Let X ∈ R
H×W×C be a high-resolution image of an arbitrary size *H, W, C*. A sampler G takes X
as input and computes a downsampled image Xˆ = G(X), where Xˆ ∈ R
h×w×C .

on the receiver side, outperforming conventional codecs like JPEG and other neural network-driven compressors. The default compressor used is JPEG due to its common use and prevalence. The whole framework is illustrated in Fig. 2. Next, we'll present our design step-by-step following the dataflow shown in Fig. 2. A detailed flexibility analysis is presented in §3.2.4. Previous image enhancement frameworks usually employ super-resolution as the downsamplereconstruction technique (Yin et al., 2023; Cheng et al., 2024). We point out that this introduces an implicit assumption and limits its flexibility. The standard super-resolution (SR) model with multiple degradations typically posits that the lowresolution image is a degraded representation of a high-resolution image, characterized explicitly as a blurry, noisy, and sub-sampled version of the original.

$$y=(x\circledast k)\downarrow_{s}+e$$

## Y = (X ⊛ K) ↓S +Ε With Ε ∼ N (0, Σ2)

In this formulation, let x denote the high-resolution image, y represent its low-resolution counterpart, k be the blur kernel, ↓s signify the subsampling operator with scale factor s, and ϵ denote the additive noise. This model operates under the assumption that the blur kernel is uniform across the entire image, allowing for efficient computation of the low-resolution image through convolution or fast Fourier transform techniques, as highlighted in recent studies (Laroche et al., 2023; Jin et al., 2022).

However, when this model is directly implemented within an edge image-enhancement framework, the implicit assumption of uniformity introduces a constraint on the downsampling ratio, which in turn restricts the framework's flexibility. This limitation underscores the need for more adaptable techniques to accommodate varying degradation patterns, limiting the framework's flexibility. Next, we will explain how to relax this assumption. The uniformly downsampled assumption is critical for applying efficient super-resolution-based reconstruction through convolution or fast Fourier transform techniques. By challenging this assumption, direct random pixel prediction becomes costly (see §3.3.2). We then propose a two-stage patchify process with a lightweight transformer to solve this problem.

Whole structure Erase-and-squeeze Un-erased Sub-Patch (bxb)
Erased Sub-Patch( bxb)
nxn nx(n-b)
nxn Erase **Squeeze**
Compressor Image Patch *Squeezed Image Patch* **Edge**
Server Reconstruction nx(n-b) nxn Decoder Un-erased sub-patch vector Encoder Decompressor Decompressed Squeezed Image Patch Reconstructed Image Patch Zero vector When compress, we transmit a "partially erased" image, which will lead to directly compression ratio increase. When decompress, we recover the erased components utilizing auto-encoder. This will enhance all kinds of current compressors, with no extra computational costs during compression stage. 
Consider a coordinate system such that X[*u, v*] is the pixel value of X where u, v ∈ [0, H − 1] and
[0, W − 1], respectively. Xˆ [*i, j*] is the pixel value of Xˆ at coordinates (*i, j*) for i ∈ {1, 2, . . . h}, j ∈
{1, 2*, . . . w*}. Essentially, the sampler G computes a mapping between (*i, j*) and (u, v). Practically, sampler G contains two functions g 0, g1	such that:

$${\hat{\mathbf{X}}}[i,j]:=\mathbf{X}\left[g^{0}(i,j),g^{1}(i,j)\right]$$

The uniform approach will have a sampler Gu =g 0 u(i, j) = (i − 1)/(h − 1), g1u(*i, j*) = (j − 1)/(w − 1)	.

3.2.2 CONDITIONAL UNIFORM-BASED SAMPLER
162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 In this section, we aim to propose an effective sampler that challenges the implicit assumption discussed in Section 3.1. We treat each pixel as a sampling unit, though this can also be extended to patches (See §3.2.4). We initially introduce a Uniform-based sampler for row-based random sampling and subsequently impose constraints on it. Row-based sampling is employed to ensure that the sampled image can be reassembled into a rectangular format.

Random Sampler Definition. The random sampler Gr computes a mapping between the coordinates
(i, j) of the downsampled image Xˆ and the relative coordinates (*u, v*) of the original image X. In this sampler, each row i of the image is processed sequentially, and within each row, the column coordinate j is selected using a uniform random sampler. The sampler is defined by two functions, g 1 r, which governs the random column selection from X, while g 0r(i) represents the current row:

## Xˆ [I, J] = X[G 0 R (I), G1 R (I, J)]

where g 0r(i) is a deterministic function representing row selection, and g 1 r(*i, j*) is a random mapping for the column coordinate within row i. Uniform Random Selection in Each Row. To ensure uniform random sampling within each row of the original image X, the row coordinate g 0 r(i) is fixed as i, and the column selection function g 1 r(*i, j*) samples a pixel uniformly from the width W for each row i. The random sampler Gr is described as follows:

$$G_{r}=\left\{g_{r}^{0}(i)=i,\quad g_{r}^{1}(i,j)=\mathrm{Uniform}(0,W-1)\right\}$$

Here, the row index i remains fixed for each row, and the uniform random sampler g 1 r(i, j) selects random column coordinates for each pixel j in row i. This ensures uniform random selection across columns within each row while maintaining a structured row-based approach. Applying this sampler directly results in poor compression ratios and reconstruction performance, as shown in Fig. 3(a). Quantitative analysis reveals that this issue stems from the adjacent sampled areas (see Fig. 3(b)). To address this problem, we introduce two constraints on the sampler.

Constraints for Row-based Random Sampling. When sampling from a matrix X ∈ R
H×W , where each row is sampled T times, the new sample xi,t+1 is subject to the following conditions:
1. Intra-row constraint (avoid proximity to previous samples in the same row):

g
1
r(*i, t* + 1) ∼ Uniform(0, W − 1) subject tog
$$\left|g_{r}^{1}(i,t+1)-g_{r}^{1}(i,t)\right|>\delta$$
Here, δ is a threshold distance that ensures the newly sampled column g
1
r(*i, t* + 1) is sufficiently
distant from the previously selected columns {g
1
r(i, 0)*, . . . , g*1r(i, t)}. This constraint guarantees a
diverse selection of columns within each row, preventing the samples from clustering too closely together. 2. Inter-row constraint (minimize adjacency to prior samples from the preceding row):
g 1 r(*i, t* + 1) ∼ Uniform(0, W − 1) subject tog 1 r(*i, t* + 1) − g 1 r(i − 1, T) > ∆
Similarly, ∆ represents a minimum separation between the newly sampled column in row i and the previously selected columns {g 1 r(i − 1, 0)*, . . . , g*1r(i − 1, T)} from row i − 1. This prevents adjacent 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 3.2.4 FLEXIBILITY ANALYSIS By relaxing the uniform sampling assumption outlined in §3.1, Easz can no longer utilize a superresolution method for reconstruction, as the low-level continuous information becomes fragmented.

- Both horizontal and vertical erase and squeeze methods are viable, and they may slightly influence the subsequent image compression results, depending on the shape of the image. We will discuss these differences in the experimental section.

Auto-encoder structure on decoding stage
- After decompressing and retaining the squeezed portion, we proceed to reconstruct the erased parts to restore the entire image. - The model we utilize consists of two main components: an encoder and a decoder. The encoder takes the existing non-erased blocks and encodes them into a high-dimensional feature space. Then, these high-dimensional feature vectors are combined with some blank vectors and fed into the decoder. The decoder utilizes these high-dimensional feature vectors to make predictions on the blank vectors and generate the reconstructed erased parts.

- In summary, the encoder captures the essential information from the non-erased blocks, and the decoder leverages this information to infer and reconstruct the erased portions, ultimately restoring the complete image. 

n-b n b b Patchify Mapping n n Random, T=1 Random, T=2 Random, T=3 Org image **Squeezed**
Decod er blo ckn Enco der b loc k Enc od er b loc k De cod er bl ock Layernorm Un-erased subpatch vector n Zero vector Zero filled **Neighbor filled**
(b)
Proposed, T=1 Proposed, T=2 **Proposed, T=3**
(a)
(a) (b) (c)

$$\mathrm{subject~to~}$$

rows from sampling nearby columns, ensuring that the selection process avoids redundancy across rows. Under these constraints, the row-based random sampler can be formalized as:

$$G_{r}=\left\{g_{r}^{0}(i)=i,\quad g_{r}^{1}(i,t+1)=\left\{\begin{array}{l l}{{\mathrm{clip\,(Uniform(0,}W-1))}}\\ {{|g_{r}^{1}(i,t+1)-g_{r}^{1}(i,t)|>\delta}}\\ {{|g_{r}^{1}(i,t+1)-g_{r}^{1}(i-1,T)|>\Delta}}\end{array}\right.\right.$$

where the random column selection g 1 r
(*i, t* + 1) is adjusted dynamically to satisfy both the intra-row and inter-row constraints. This ensures a well-distributed sampling process across the entire matrix, balancing randomness with structured diversity.

 # 2.3    Erasse AND SQUEE  $$\newcommand{\vecs}[1]{\overset{\rightharpoonup}{\mathbf{#1}}}$$  $$\newcommand{\vecd}[1]{\overset{-\!-\!\rightharpoonup}{\vphantom{a}\smash{#1}}}$$
To handle the unsampled locations in the downsampled image Xˆ , we define a binary mask M of the same size as Xˆ where each entry is set as follows:

## M[I, J] = 1 If (I, J) Is Sampled, Else 0

The mask ratio, which determines the proportion of the image that is sampled, is controlled by the patch size p and the sampled size T. Specifically, the choice of patch size influences the granularity of the sampling, while the sampled size T dictates how many patches are included in the reconstruction for each row. By tuning these parameters, the framework can effectively balance the level of reconstruction difficulty and the computational load, enhancing model performance across various datasets. The mask will be sent with the compressed image for the receiver to decode.

The next step is to squeeze the non-zero (sampled) locations together to form a smaller image Xsqueezed ∈ R
h
′×w
′×C , where h
′ < h and w
′ < w represent the dimensions of the squeezed image.

This can be achieved by filtering out the zero entries from X:

$$\mathbf{X}_{\mathrm{squeezed}}[i^{\prime},j^{\prime}]=\mathbf{X}[i,j]\quad{\mathrm{for~all~}}(i,j){\mathrm{~where~}}\mathbf{M}[i,j]=1$$
$$\operatorname{L}_{\mathrm{L}}\operatorname{T}\operatorname{S}\mathrm{I}\otimes$$

After erase-and-squeeze process, the squeezed image Xsqueezed would be encoded using existing compressors like JPEG, BPG, etc to get a compressed form Xˆsqueezed, as illustrated in Fig. 2.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 However, this shift creates new opportunities for flexibility. By controlling the sample size T, Easz can provide a more adaptable and fine-grained compression level compared to directly applying super-resolution techniques. Erase level. Given an image with dimensions H × W × C, the overall compression ratio can be understood as comprising two components: 1) The ratio achieved through image size reduction. 2) The ratio obtained from the subsequent compression algorithm. The image size reduction is primarily controlled by the sampling size T applied to each row. Thus, the reduction ratios can be expressed as 1 W ,
2 W ,
3 W *, . . .*. It's important to note that this proposed method can be similarly transposed to other axes, such as the columns. So far, our discussion has focused on pixel-level sampling. However, the sample-erase-squeeze unit can be extended to operate on *patches*. By adopting this extension, we introduce another parameter that influences the reduction ratio: the patch size p. Consequently, experiments conducted with varying patch sizes will be detailed in §4.4. Under this patch-level sampling framework, the reduction ratios would be expressed as p 2 W ,
2p 2 W ,
3p 2 W *, . . .*. In contrast to traditional super-resolution methods, which typically offer a single reduction pattern for a model, our sampler provides significant flexibility, enabling adaptation to various real-world applications. Model switching and mask transferring. Switching models and transferring masks might be burdensome. As would introduced in §3.3.2, we present a lightweight transformer-based process capable of performing direct pixel-level reconstruction. This process is designed to handle any erase ratio since it is trained under this setting. Consequently, there is no need to prepare a separate model for each erase ratio or to switch models during compression ratio adjustments. Another consideration is mask transferring. In our design, the mask is applied to small sub-patches (discussed in §3.3.2) created through a proposed two-stage image patching process. For instance, if the sub-patch size is 32 × 32, then the corresponding mask size would be 128 bytes. This same mask is used for all sub-patches. Thus, the transmission of this size is not a concern. As mentioned in the beginning, by utilizing the proposed erase-and-squeeze technique, the blurkernel-based super-resolution method is no longer suitable for reconstruction. We then introduce a lightweight transformer architecture to directly conduct pixel prediction to address this issue.

## 3.3 Reconstruction

Given the squeezed compressed image Xˆsqueezed, we introduced a Masked-Image-Modeling process to perform the pixel-level reconstruction on the receiver side. The reconstruction of the squeezed image back to the original image is approached through a framework reminiscent of masked image modeling (MIM). Specifically, the input is Xˆsqueezed. Y = T (Xˆsqueezed). The autoencoder model G(·)
takes the corrupted images Xˆsqueezed as input and aims to generate a prediction Yˆ , optimizing the model by minimizing the difference between the prediction and the target:

## Yˆ = G(XˆSqueezed), L = D(X, Yˆ )

This paper focuses on parametric target generation strategies utilizing a transformer structure for image reconstruction tasks. A significant challenge arises from directly predicting pixel values using transformers due to the quadratic complexity of attention mechanisms.

## 3.3.1 Complexity Analysis.

Predicting pixel values for each element in the image, particularly in high-resolution images, proves to be computationally expensive. Let the transformer model's attention mechanism operate with complexity O(n 2· d), where n is the number of tokens (patches or pixels) and d is the dimensionality of the token embeddings. This results in costly computation when n is large, especially for pixel-level operations.

For example, a 256x256 grayscale image would require 4, 294, 967, 296 × d*model* calculations if each pixel were treated as a token. By employing the two-stage patchify process (with n = 32 and b = 1), this number is reduced by 256 times to 16, 777, 216 × d*model* calculations. Detailed analysis would be in Appendix B. The reduction in complexity comes from performing attention operations within each patch rather than across the whole image. 3.3.2 TWO-STAGE PATCHIFYING AND LIGHTWEIGHT RESTORATION. Motivated by this observation, we adopt a *two-stage patchifying process* followed by a lightweight transformer to improve computational efficiency.

1. First Stage: Initial Image Patchifying: Given the image X ∈ R
h×w×C , we divide the image into non-overlapping patches of size ph × pw. For each patch Pm,n, the extraction follows:
Pm,n = X[m · ph : (m + 1) · ph − 1, n · pw : (n + 1) · pw − 1, :]
324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

where $m=0,\ldots,\left\lfloor\frac{h}{p_{h}}\right\rfloor-1$ and $n=0,\ldots,\left\lfloor\frac{w}{p_{w}}\right\rfloor-1$.  
2. Second Stage: Subdividing into Sub-Patches: Each patch Pm,n is further divided into sub-patches of size Sh × Sw to facilitate fine-grained prediction:
Pm,n,k,l = Pm,n[k · Sh : (k + 1) · Sh − 1, l · Sw : (l + 1) · Sw − 1, :]

$$n\left[k+S_{h}:(k+1)\cdot S_{h}\right]$$
where $k=0,\ldots,\left\lfloor\frac{p_{h}}{S_{h}}\right\rfloor-1$ and $l=0,\ldots,\left\lfloor\frac{p_{w}}{S_{w}}\right\rfloor-1$.  
3. Lightweight Transformer for Sub-Patch Restoration: A lightweight transformer model, denoted as Tlight(·), with complexity O(m2· d) (where m is the number of sub-patches), is employed for restoring the missing pixel values within each sub-patch. Fig. 3(c) illustrates the architecture of our efficient transformer-based network on the server side. The encoder and decoder are composed of two transformer blocks, each containing three layernorms, one attention layer, and one feedforward layer. The model size is significantly reduced to 8.4MB. Given the downsampled input sub-patch P*m,n,k,l*, the predicted sub-patch Pˆ*m,n,k,l* is generated as:

$$)\cdot p_{w}-1,:]$$
$$s_{h}-1,l\cdot S_{w}:(l+1)\cdot S_{w}-1,:]$$
$$\hat{P}_{m,n,k,l}=\mathcal{T}_{\mathrm{light}}(P_{m,n,k,l})$$

Through this reconstruction process, the original content of the image is effectively restored from the masked and squeezed representation. We adopt LPIPS (Zhang et al., 2018), a well-known perceptual loss, along with L1 loss as training loss as L. Note that the encoder and decoder can work with a variety of input erase ratios, which is controlled by p and T (See §3.2.4), and hence, we do not need to train a separate model for each erase ratio. Sub-patches can execute in parallel both for encoder and decoder due to the nature of transformer block (Vaswani et al., 2017).

## 4 Experiments 4.1 Experimental Setting

Training setting. The experiments consist of two phases: offline pretraining and online testing. In the pretraining phase, a specific loss function is used with these hyperparameters: learning rate of 2.8e-4, erase ratio of 0.25, batch size of 4096, and weight decay of 0.05. Randomly generated erase masks are applied for model robustness during this stage. A consistent mask is utilized for online testing on both edge and server sides.

Hardware platforms. Our framework is implemented with ∼1000 lines of Python. We use an NVIDIA Jetson TX2 as the edge device and a desktop with Intel i7-9700K CPU and RTX 2080Ti GPU as the server, which are connected to a Wi-Fi router and communicate via TCP.

Datasets. During the offline pretraining phase, the CIFAR-10 (Alex and Geoffrey, 2009) dataset is employed to pre-train the model, enabling it to acquire generative capabilities. In the testing phase, two common image compression datasets, Kodak (Company, 1993) and CLIC (Workshop and on Learned Image Compression, 2022), are utilized to assess the generative performance of the proposed method. Metrics. Since removing content would negatively impact reference-based metrics such as PSNR and SSIM (a trend also observed in other downsampled-and-super-resolution methods), we employ nonreference perceptual metrics for comparison with different compression techniques: Brisque (Mittal 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 et al., 2012), Pi (Blau et al., 2018), and Tres (Golestaneh et al., 2022). To benchmark against other super-resolution approaches, we include PSNR and SSIM to demonstrate the superiority of our method. Furthermore, we have conducted image classification experiments on the reconstructed images to showcase the proposed compression method's robustness in handling image analytic tasks. Compression performance is evaluated by bits per pixel (BPP). Baselines. We use four compression methods as baselines to demonstrate the effectiveness of the proposed method: JPEG, BPG, MBT(Minnen et al., 2018), and Cheng-Anchor. Among these, MBT and Cheng-Anchor are two neural-network-based compression methods.

Easz MBT Cheng 0 1 2 3 0 5000 10000 15000 20000 Easz MBT
Cheng Erase&Squeeze Compression Transmit Decomp Recon Po w er 
(
W
)

GPU Power CPU Power 1.05 1.93 1.98 M
e m ory (
G
B) 
2 1 Easz MBT Cheng 0

## Figure 4: Efficiency Evaluation On Nvidia Jetson Tx2.

We first report the latency breakdown of Easz with other neural network-based compression methods, using a Jetson TX2 for compression and a server for decompression. We repeat the runs 24 times and report the average on Fig. 4a. We observe that Erase-and-Squeeze only takes up 0.7% of the end-to-end latency, which induces minimal overhead on the edge device and proves the efficiency of Easz' design. While both MBT and Cheng-Anchor are too compute-intensive to run the compression on the edge side. As expected, the reconstruction in Easz takes the longest time, accounting for 74% of the latency. We argue that it can be significantly improved by upgrading to a datacenter-class GPU, such as the A100, instead of the RTX 2080Ti. Resource consumption is another critical consideration when it comes to resource-constrained edge devices. To assess this, we measure three key metrics - CPU power, GPU power, and memory footprint - using the Tegrastats Utility (teg, 2023) on the Jetson TX2. As illustrated in Fig. 4, our findings reveal that Easz excels in all metrics compared to other NN-based compression methods. Specifically, in contrast to MBT and Cheng-Anchor, Easz achieves a remarkable 71.3% and 59.9%
reduction in total power consumption. It's noteworthy that Easz does not utilize any GPU power on the edge device, attributed to its lightweight yet effective erase-and-squeeze design. Furthermore, Easz reduces memory footprint by 45.8% and 47.1%, respectively. These results underscore the advantage of deploying Easz on wimpy edge devices.

## 4.3 Comparison With Super-Resolution Methods

We compare the reconstruction effect of Easz with state-of-the-art super-resolution methods to demonstrate Easz's effectiveness. As shown in Tab. 1, Easz outperforms super-resolution in pixellevel reconstruction metrics while having a much more flexible reduction ability. Note that Easz uses a model of only 8.7MB, while other models are 67MB.

Table 1: Comparison with Super-Resolution on Kodak Dataset.

Metrics Easz SwinIR realESRGAN BSRGAN
PSNR 28.96 24.86 24.85 25.35 MS_SSIM 0.96 0.94 0.93 0.94 Recon Model Size 8.7MB 67MB 67MB 67MB
Experimental results demonstrate that Easz surpasses traditional super-resolution in terms of PSNR and SSIM metrics when reducing pixel count equivalently. Figure 5 illustrates a comparison of image detail reconstruction, where Easz and other super-resolution all perform 2x reconstruction. It is evident that Easz better preserves image details; the children's faces are clearer, and the characters 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

ESRGAN BSRGAN SwinIR Easz
are more recognizable. In contrast, the super-resolution reconstructed image is unsatisfactory. More quantitative results are shown in Appendix E

## 4.4 Ablation Study

Effectiveness of proposed sampler. Fig. 6a and Fig. 6b compares the proposed erase mask generation method, the random erase mask method, and the baseline(JPEG and BPG) throughout the entire pipeline. It can be observed that the proposed erase mask generation method achieves better BPP at the same quality level on both JPEG and BPG, further substantiating the effectiveness of the proposed method.

Ms e 1e 3 Patch4 Patch2 Patch1 Los s v alu e 1e 2 Patch=4 Patch=2 Patch=1 0 250500750 1000 1250 1500 17502000 Epochs 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 0.0 0.1 0.2 0.3 0.4 0.5 0.6 Infer Time 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 0.000.250.500.751.001.251.501.752.00 BPP
10 15 20 25 30 35 40 45 Bri sq ue BPG +Easz
+random Bri sq ue JPEG +Easz +random 1 2 3 4 5 BPP
5 10 15 20 25
Figure 6: (a)(b): Comparison between Easz with proposed mask strategy, Easz with random mask strategy, and conventional compression baselines (JPEG and BPG). (c) Patch size and erase ratio's impact on MSE(↓ better). (d) MSE(↓ better) during Easz fine-tuning process with patch size=1,2,4 on Kodak dataset. Patch size selection. Fig. 6c examines the effects of two hyperparameters, erase block size (1, 2, and 4) and erase ratio (10% to 50%), on compression rate and quality. As the erase ratio increases, MSE rises, indicating lower reconstruction quality. Smaller patch sizes yield better reconstruction due to higher local correlations. Patch size=2 offers a balance between speed—being six times faster than size=1—and quality—with only a slight difference in MSE. Doubling the patch size from 2 to 4 also doubles both speed and MSE. The recommendation is to use smaller patch sizes for practical applications but consider size=2 for additional speed needs.

Effectiveness of fine-tuning. Our model, after pretraining on the CIFAR dataset for 5000 epochs, can be applied to various image compression tasks due to its ability to recognize similarities in local image features. Typically, models are first pre-trained on a large dataset and then fine-tuned for specific tasks. We tested if fine-tuning our pretrained model with the Kodak dataset would be beneficial and found that it indeed improves performance by reducing losses across different patch sizes (1x1, 2x2, and 4x4), as shown in Fig. 6d. This suggests that online fine-tuning of pre-trained models could further enhance compression effectiveness in real-world applications.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

| Metrics       | JPEG      | BPG    | MBT       | Cheng-anchor   |           |       |           |       |
|---------------|-----------|--------|-----------|----------------|-----------|-------|-----------|-------|
| Org           | +Proposed | Org    | +Proposed | Org            | +Proposed | Org   | +Proposed |       |
| BPP           | 0.412     | 0.411  | 0.382     | 0.410          | 0.433     | 0.389 | 0.418     | 0.402 |
| Brisque 43.06 | 22.34     | 30.675 | 23.27     | 28.13          | 18.63     | 29.16 | 20.51     |       |
| Pi            | 4.84      | 3.33   | 3.07      | 3.04           | 3.01      | 3.00  | 3.11      | 3.05  |
| Tres          | 77.62     | 86.26  | 83.55     | 85.88          | 84.14     | 88.03 | 88.53     | 89.80 |
| BPP           | 0.306     | 0.307  | 0.308     | 0.293          | 0.308     | 0.292 | 0.287     | 0.267 |
| Brisque 60.51 | 23.63     | 39.95  | 25.27     | 32.20          | 18.37     | 35.42 | 21.55     |       |
| Pi            | 8.51      | 5.02   | 4.85      | 4.66           | 4.33      | 4.35  | 4.58      | 4.50  |
| Tres          | 50.65     | 63.69  | 65.14     | 67.08          | 73.54     | 78.30 | 82.91     | 83.95 |

4.6 END-TO-END COMPRESSION PERFORMANCE

Lat en cy 
(m s)
1e4 0.2 0.4 0.6 0.8 1.0 1.2 BPP
20 30 40 50 60 JPEG Easz MBT Cheng 0.2 0.4 0.60.81.0 1.2 BPP
3 4 5 6 7 0.2 0.4 0.6 0.8 1.0 1.2 BPP
60 65 70 75 80 85 90 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 BPP
0.5 1.0 1.5 2.0 2.5 JPEG Easz MBT

Cheng Bri sq ue MBT
Cheng Easz Tre s JPEG Easz MBT

Cheng Pi
(a) Brisque (↓ better).

(b) Pi (↓ better).

(c) Tres (↑ better).

(d) End-to-end Latency.
Figure 7: Compression performance of Easz, JPEG, MBT and Cheng on three perceptual metrics (a-c). Fig. 7d evaluates the end-to-end latency on our testbed.

In this experiment, we use JPEG+Easz as the baseline and observe changes in three perceptual metrics at different bitrates (BPP). Notably, JPEG alone underperforms compared to two deeplearning compression methods at all compression levels. However, with Easz enhancement, JPEG shows a marked improvement. For the BRISQUE metric specifically, JPEG+Easz exceeds both deep-learning methods. Regarding the Pi metric, JPEG+Easz matches the performance of these methods. With the Tres metric, while JPEG+Easz outdoes MBT, it falls short of Cheng-anchor's results. Overall, Easz boosts JPEG to compete effectively with other state-of-the-art deep-learning compression techniques in each perceptual measure. Easz also outperforms two neural network-based methods in latency, with an average end-to-end latency of 2568ms across different bitrates per pixel, marking an 89% reduction compared to MBT and Cheng's methods.

## 5 Conclusion

This paper proposes Easz, which addresses the challenges of edge image compression and transmission latency by introducing a novel erase-and-squeeze technique that enhances flexibility and efficiency. By relaxing the conventional requirement for uniform downsampling, Easz allows for adaptable compression levels tailored to dynamic real-world applications. The implementation of a lightweight transformer architecture on the receiver side ensures high-quality reconstruction of erased image patches without imposing significant computational demands. Moreover, Easz's compatibility with existing compression algorithms makes it a versatile solution for modern edge devices. Our real-world evaluation in an edge-server testbed demonstrates Easz's improvement in compression performance and efficiency, emphasizing its potential for real-world applications. To evaluate how well Easz works with leading compressors, we incorporated it into four established methods: JPEG and BPG (traditional compressors), as well as MBT and Cheng-anchor (neural network-based compressors). We used two datasets, Kodak and CLIC, to test the resilience of Easz across different types of image data. For the Kodak dataset, we aimed for a bit-per-pixel (BPP) rate of approximately 0.4; for the CLIC dataset, we targeted a BPP of around 0.3 to gauge Easz's efficacy at varying levels of compression. The results showing how each baseline method performs on its own and when combined with Easz —are detailed in Tables 2.

Table 2: Compression Performance Enhancement on Kodak Dataset and Clic Dataset.

## 4.5 Improvement On Existing Compressors References

2018. AI and Compute. https://openai.com/blog/ai-and-compute/
2020. The Future of Computing is Distributed. https://www.datanami.com/2020/02/
26/the-futureof-computing-is-distributed/ [Accessed on October 14, 2023].

2023. AWS Outposts. https://aws.amazon.com/outposts/ 2023. Bringing Cutting-Edge Technology to Wildlife Conservation. https://www.

wildlifeinsights.org/.

2023. Tegrastats Utility. https://docs.nvidia.com/drive/drive_os_5.1.6.1L/
nvvib_docs/DRIVE_OS_Linux_SDK_Development_Guide/Utilities/util_ tegrastats.html.

Krizhevsky Alex and Hinton Geoffrey. 2009. Learning multiple layers of features from tiny images.

2009 (2009).

Ganesh Ananthanarayanan, Victor Bahl, Peter Bodik, Krishna Chintalapudi, Matthai Philipose, Lenin Ravindranath Sivalingam, and Sudipta Sinha. 2017. Real-Time Video Analytics - The Killer App for Edge Computing. *IEEE Computer* (October 2017).

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, and Nick Johnston. 2018. Variational image compression with a scale hyperprior. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net.

Fabrice Bellard. 2014. BPG. https://bellard.org/bpg/. Yochai Blau, Roey Mechrez, Radu Timofte, Tomer Michaeli, and Lihi Zelnik-Manor. 2018. The 2018 PIRM challenge on perceptual image super-resolution. In Proceedings of the European Conference on Computer Vision (ECCV) Workshops. 0–0.

Yihua Cheng, Ziyi Zhang, Hanchen Li, Anton Arapin, Yue Zhang, Qizheng Zhang, Yuhan Liu, Kuntai Du, Xu Zhang, Francis Y Yan, et al. 2024. {GRACE}:{Loss-Resilient}{Real-Time} Video through Neural Codecs. In *21st USENIX Symposium on Networked Systems Design and* Implementation (NSDI 24). 509–531.

Zhengxue Cheng, Heming Sun, Masaru Takeuchi, and Jiro Katto. 2020. Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules. In *Proceedings of the* IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

He Dailan, Zheng Yaoyan, Sun Baocheng, Wang Yan, and Qin Hongwei. 2017. Checkerboard context model for efficient learned image compression. In *International Conference on Learning* Representations. 1–27.

Mallesham Dasari, Kumara Kahatapitiya, Samir R Das, Aruna Balasubramanian, and Dimitris Samaras. 2022. Swift: Adaptive video streaming with layered neural codecs. In *19th USENIX* Symposium on Networked Systems Design and Implementation (NSDI 22). 103–118.

S Alireza Golestaneh, Saba Dadsetan, and Kris M Kitani. 2022. No-reference image quality assessment via transformers, relative ranking and self-consistency. In Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision. 1220–1230.

JPEG Group. 1986. JPEG. https://jpeg.org/. Eastman Kodak Company. 1993. Kodak Lossless True Color Image Suite. https://r0k.us/
graphics/kodak/.

Shilpa George, Junjue Wang, Mihir Bala, Thomas Eiszler, Padmanabhan Pillai, and Mahadev Satyanarayanan. 2019. Towards Drone-Sourced Live Video Analytics for the Construction Industry. In *Proceedings of the 20th International Workshop on Mobile Computing Systems and Applications*. ACM, 3–8.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Dailan He, Yaoyan Zheng, Baocheng Sun, Yan Wang, and Hongwei Qin. 2021. Checkerboard context model for efficient learned image compression. In *Proceedings of the IEEE/CVF Conference on* Computer Vision and Pattern Recognition. 14771–14780.

Chen Jin, Ryutaro Tanno, Thomy Mertzanidou, Eleftheria Panagiotaki, and Daniel C Alexander.

2022. Learning to downsample for segmentation of ultra-high resolution images. *ICLR* (2022).

Kim Jun-Hyuk, Heo Byeongho, and Lee Jong-Seok. 2022. Joint global and local hierarchical priors for learned image compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5992–6001.

Charles Laroche, Andrés Almansa, and Matias Tassano. 2023. Deep model-based super-resolution with non-uniform blur. In *Proceedings of the IEEE/CVF winter conference on applications of* computer vision. 1797–1808.

Jingzong Li, Yik Hong Cai, Libin Liu, Yu Mao, Chun Jason Xue, and Hong Xu. 2023a. Moby:
Empowering 2D Models for Efficient Point Cloud Analytics on the Edge. In *Proc. ACM MM*.

Jingzong Li, Libin Liu, Hong Xu, Shudeng Wu, and Chun Jason Xue. 2023b. Cross-Camera Inference on the Constrained Edge. In *Proc. IEEE INFOCOM*.

Yu Mao, Yufei Cui, Tei-Wei Kuo, and Chun Jason Xue. 2022a. Accelerating General-Purpose Lossless Compression via Simple and Scalable Parameterization. In *Proceedings of the 30th ACM* International Conference on Multimedia. 3205–3213.

Yu Mao, Yufei Cui, Tei-Wei Kuo, and Chun Jason Xue. 2022b. TRACE: A Fast Transformerbased General-Purpose Lossless Compressor. In *Proceedings of the ACM Web Conference 2022*.

1829–1838.

Yu Mao, Jingzong Li, Yufei Cui, and Jason Chun Xue. 2023. Faster and Stronger Lossless Compression with Optimized Autoregressive Framework. In 2023 60th ACM/IEEE Design Automation Conference (DAC). 1–6.

David Minnen, Johannes Ballé, and George Toderici. 2018. Joint Autoregressive and Hierarchical Priors for Learned Image Compression. In Advances in Neural Information Processing Systems 31:
Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, 3-8 December 2018, Montréal, Canada, Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicolò Cesa-Bianchi, and Roman Garnett (Eds.). 10794–10803.

Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik. 2012. No-reference image quality assessment in the spatial domain. *IEEE Transactions on image processing* 21, 12 (2012), 4695–
4708.

K. Simonyan and A. Zisserman. 2015. Very deep convolutional networks for large-scale image recognition. In *3rd International Conference on Learning Representations (ICLR 2015)*. 1–14.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).

Workshop and Challenge on Learned Image Compression. 2022. Challenge on Learned Image Compression. https://compression.cc/.

Guanghao Yin, Zefan Qu, Xinyang Jiang, Shan Jiang, Zhenhua Han, Ningxin Zheng, Xiaohong Liu, Huan Yang, Yuqing Yang, Dongsheng Li, et al. 2023. Online Streaming Video Super-Resolution with Convolutional Look-Up Table. *arXiv preprint arXiv:2303.00334* (2023).

Yang Yang Yinhao Zhu and Taco Cohen. 2022. Transformer-based transform coding. In *International* Conference on Learning Representations. 1–35.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition. 586–595.

## A Detailed Efficiency Report For Nn-Based Compressors

648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 To further illustrate the challenges faced when deploying current neural network-based compressors on edge devices, this section provides a detailed report on the performance of the four compression methods: b-facDailan et al. (2017), b-hyperBallé et al. (2018), MBTMinnen et al. (2018), Cheng- AnchorCheng et al. (2020) on actual edge devices. This area has been seldom explored in previous research. Table 3 shows the Edge FLOPs per image (512x768), model size for a single compression level, and loading time for five deep learning compression methods on NVIDIA Jetson TX2. It is important to note that the model size is only for one compression level; therefore, actual storage requirements are calculated by multiplying the given number by the number of compression levels. For example, Cheng-Anchor has six levels of compression, thus requiring a total storage space of 110MB*6=660MB. It's worth mentioning that although quantization can improve both space occupancy and operational efficiency of models, this approach involves performance, and it's wellknown that quantization presents issues: its deployment on edge devices is complex and difficult to standardize. In contrast, Easz offers an easily deployable framework.

Table 3: Details of representative NN-based compression methods. Note that the loading and compression times are measured on an NVIDIA Jetson TX2, and FLOPs are evaluated with (512×768) image size.

| NN-based methods      | b-fac   | b-hyper   | MBT   | Cheng-Anchor   |
|-----------------------|---------|-----------|-------|----------------|
| Edge-FLOPs            | 36G     | 36G       | 36G   | 145G           |
| Model size (MB)       | 28      | 46        | 118   | 110            |
| Loading time (ms)     | 286     | 552       | 1361  | 1600           |
| Compression time (ms) | 374     | 413       | 17952 | 18015          |

## B Two-Stage Image Patchifying Analysis

Consider a grayscale image of size 256 × 256 with a patch size of 1 × 1. This results in 256 × 256 = 65, 536 pixels. In an attention-based model, treating each pixel as a token, the computational complexity for self-attention is given by:

$$O((h\times w)^{2}\times d_{\mathrm{model}})$$

where h and w are the height and width of the image, and dmodel is the model dimension. For a 256 × 256 image, the calculation becomes:

$$O(65536^{2}\times d_{\mathrm{model}})=O(4,294,967,296\times d_{\mathrm{model}})$$

This level of computation is prohibitively expensive on modern hardware, and the complexity increases rapidly for higher-resolution images.

To reduce the computational load, we employ a two-stage patchify process. The first stage divides the image into non-overlapping patches of size n × n. The number of patches is:

$\dfrac{h\times w}{w}$ . 
$$\overline{{n^{2}}}$$
$$O\left(\left({\frac{h\times w}{n^{2}}}\right)^{2}\right)=O\left({\frac{(h\times w)^{2}}{n^{4}}}\right)$$
This reduces the number of tokens, thus lowering the complexity. However, further refinement can be achieved with a second stage of patchification. Each patch is treated as a token, and the complexity of attention at this stage is reduced to: Each n×n patch is subdivided into b×b sub-patches in the second stage. The number of sub-patches is: This results in a total of:

$$\frac{n^{2}}{b^{2}}$$

sub-patches, and the attention complexity at this stage becomes:

$$h\times w$$
b
$\overline{h^2}$. 
$$O\left({\frac{h\times w}{n^{2}}}\times{\frac{n^{4}}{b^{4}}}\right)=O\left({\frac{(h\times w)\times n^{2}}{b^{4}}}\right)$$

to:
702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755

## D Loss

Thus, the complexity is much lower than the original full-image attention, even for very small sub-patches (e.g., when b = 1). The final complexity is:

$$O(h\times w\times n^{2})$$

For the example of a 256 × 256 image, where n = 32 and b = 1, the complexity reduces from:

$$O(65536^{2}\times d_{\mathrm{model}})=O(4,294,967,296\times d_{\mathrm{model}})$$
$$O(16,777,216\times d_{\mathrm{model}})$$

This represents a 256-fold reduction in computational complexity compared to the original.

## C Inference For Easz

During the inference stage, the procedure starts from receiving a set of un-erased sub-patches U = {u1, u2*, . . . , u*m}, and a set of zero sub-patches Uˆ = {uˆ1, uˆ2*, . . . ,* uˆk} is firstly added. {U,Uˆ}
is then mapped to corresponding position using M. Afterward, we project this combined set {U,Uˆ} into the embedding space {F, Fˆ}, feed it into the encoder to obtain feature representations and reconstruct Pˆ.

The mapping timing is the critical difference between the training and inference stages. The mapping is applied to the feature representations F in the training stage. However, during the inference stage, the mapping is performed on the sub-patches before they are sent into the encoder. This difference is due to the use of positional embedding. In the training process, the positional embedding is applied before the erase operation because the sub-patches remain in their original positions at that stage. On the other hand, during the inference phase, the received un-erased sub-patches need to be mapped back to their original positions to incorporate the positional embedding effectively.

To minimize the difference between the original image P and the reconstructed image Pˆ, we adopt LPIPS Zhang et al. (2018), a well-known perceptual loss, and L1 loss as training loss. The final loss function is shown as follows:

## E.1 Comparison With Other Super-Resolution Method

756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809

## E.2 Quantitive Result Of Easz Enhanced Compressor

In this section, we provide additional quantitive results for the Easz enhanced compressor and the original decompressed image to show its effectiveness.

## F.2 Limitations

$$L_{1}(x,y)=\frac{1}{N}\sum_{i=1}^{N}|x_{i}-y_{i}|$$  $$\text{LPIPS}(x,y)=\sum_{i=1}^{N}w_{i}\cdot d_{i}(x,y)$$  $$\text{Loss}(x,y)=\text{L1}(x,y)+\lambda*\text{LPIPS}(x,y)$$
$$(1)$$
$$(2)$$
$$({\mathfrak{I}})$$

Where x and y are two images being compared, N is the number of layers in the feature extraction model, which is decided as VGG Simonyan and Zisserman (2015), di(*x, y*) is the distance between the feature maps of the two images at layer i, and wiis a layer-dependent weighting factor. λ is chosen as 0.3 in our experiments.

## E Additional Quantitive Results

This section provides additional quantitive results for Easz compared with other super-resolution methods. From Fig. 8, it is evident that compared to the super-resolution method, Easz and the original image share more similar details, whereas super-resolution introduces more imaginative elements. The sys's-attributed to sys's direct pixel generation feature, making it better suited for use in conjunction with compression.

## F Discussion And Limitations F.1 Additional Opportunities

Online Finetuning.In the edge-server scenario, conducting online training using real data presents practical challenges due to the server's unavailability of the original images. To enable online training on the server-side model with real data, it is crucial to meticulously choose the images for transmission and devise an efficient and well-designed method to maximize effectiveness. Consequently, these features aid traditional compressors in improving the performance of subsequent image analysis tasks, particularly in scenarios with low-quality transmission. This observation opens an intriguing avenue for future research: How can we generate erased blocks more effectively to assist traditional compression methods better? This question could lead to exciting developments in the context of this paper. Semantic erase mask generation for specific scenarios. A more promising direction for generating erase masks is to consider the semantic information of the image. This involves erasing only less significant regions. In this paper, such methods were not adopted to avoid additional computation on the edge. However, we still believe this is a viable direction for future research.

A fundamental assumption of Easz is that the server needs to be equipped with GPUs for efficient neural network processing. While common, these accelerators can increase costs and energy consumption. Moreover, if the reconstruction model is not well-trained, it may impact the overall performance of Easz. Additionally, on extremely low-power edge devices where other compression methods are not available, sys's compression capabilities, while still usable, might have limitations compared to more specialized alternatives.

810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863

Ground Truth Easz BSRGAN Real-ESRGAN SwinIR
JPEG Easz+JPEG