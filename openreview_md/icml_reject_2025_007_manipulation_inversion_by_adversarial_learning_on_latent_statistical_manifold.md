# Manipulation Inversion By Adversarial Learning On Latent Statistical Manifold

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054

## 1. Introduction

Generative adversarial networks (GANs) have been playing as the cutting-edge deep generative models for generating realistic content (Sauer et al., 2022; Kang et al., 2023), which also popularises its application to various tasks such as image/video compression (Mentzer et al., 2020; 2022), superresolution (Wang et al., 2021; 2018), enhancement (Galteri et al., 2019), to name but a few. Compared with existing deep generative models, the merit of GANs arises from the distinct intrinsic nature of directly generating highly realistic images from low-dimensional random noise, thus capable of depicting the complicated high-dimensional data from the 1

## Anonymous Authors1 Abstract

The inversion of generative adversarial network (GAN) is able to investigate rich semantics within the generative models, thus receiving increasing research efforts most recently. Existing GAN inversion methods focus on reconstructing images, with relatively less focus on improving the editing realism, the most important criterion for evaluating the semantics achieved by inversion. In this paper, we systematically investigate the latent generating space and prove that both the realism of editing and accuracy of reconstruction can be unified under the umbrella of the inversion against manipulations. Motivated by this, we propose to establish the generating space as latent probabilistic models, followed by the developed statistical manifold to minimise the distribution discrepancy. Based on the manifold, we further propose an adversarial learning strategy to avoid the excessive enumeration when calculating the manipulation inversion metric. We may also need to point out that the proposed method is universal to different architectures, as a novel plugin inversion method. We comprehensively evaluate our method across different types of network architectures, comparing it against the state-of-theart inversion methods. The experimental results demonstrate that our method is able to achieve superior performances on both reconstruction accuracy and realism of editing.

low-dimensional continuous latent generating space. This merit also enables the latent generating space to possess rich and precise semantics (Shen et al., 2020b; Hark ¨ onen et al. ¨ , 2020), as the potentially well-behaved proxy for representing the real-world scenarios.

Since almost all the existing GANs uni-directionally generate images from the latent space, the way to invert images back into the latent generating space of GANs is the prerequisite before we start to investigate the rich semantics from real-world scenarios. This essentially requires carefully embedding into the latent space to ensure both the accuracy of reconstruction and realism of editing, which are oftentimes trade-off with the other (Wang et al., 2022; Dinh et al., 2022; Tov et al., 2021; Yao et al., 2022). Existing methods address this trade-off by restoring the semantics in the latent space and reconstructing the details in the middle-layer features, given a pre-trained (or slightly fine-tuned) generator (Wang et al., 2022; Dinh et al., 2022; Yao et al., 2022). However, existing GAN inversion methods essentially focus on pointwise estimation against the latent representation for both inverting images and semantics, without considering the characteristics of arising local curvature, thus suffering from the incompleteness regarding editing based on the estimated point. Thus, the improvement on the editing performance, the key to depicting the latent space characteristics, is still of an *ad hoc* manner (Xia et al., 2022).

Indeed, the preferred GAN inversion is able to embed arbitrary realistic images into the latent space, followed by accurate reconstruction based on the embedding; this is expected to still hold when embedding and restoring edited images, given the fact that the realism of editing is another criteria of inverting GANs. Therefore, the desirable embedding is capable of accurately inverting both the original image and its arbitrarily edited counterparts, which as shall be proved in this paper, is equivalent to the capability of precisely restoring the edited image back to the original image; we name this operation as *inverting manipulations* that essentially poses more stringent requirements against inverting images and semantics of GANs. Unfortunately, although the concept has been preliminarily mentioned in a few works, by either evaluation metrics (Tov et al., 2021) or auxiliary cycle consistency regularisation (Pehlivan et al., 2023), inverting manipulations is still yet to be systematically investigated by far.

055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 In this paper, we set out the first attempt to invert arbitrary manipulations upon GANs, so as to optimise both the latent representation and its corresponding local curvature, as illustrated in Fig. 1. More specifically, we first systematically analyse the characteristics of GANs, including in-depth analysis on the local optimum and curvatures within the generating space. In light of the analysis, we propose to embed each inverting image into an individual distribution, in which randomly sampling from the distribution operates as variants within the same identity of images, including semantic editing and non-semantic nuisance noise to reflect the local curvature. We then establish the statistical manifold for the GAN generating space based on the Cramer-Rao metric, and optimisation on the manifold improves both the image reconstruction and manipulation inversion.

Therefore, we propose to optimise the inversion of manipulation based on the established manifold, the goal that cannot be achieved by the *de facto* point-wise reconstruction by almost all inversion methods. To further relieve the excessive enumeration of random samples for inverting arbitrary manipulations, we propose an adversarial strategy to efficiently reduce the searching trials during the optimisation procedure. This way, we are able to unify the optimisation of manipulation inversion problem, under an efficient end-to-end distribution alignment within the latent space in practice. Consequently, experimental results verify the superior performance of our method in precisely inverting the manipulation, as well as on the accuracy of reconstruction and the quality of editing.

## 2. Related Works

Since the StyleGAN architecture has been exhibiting the state-of-the-art generation performances in various scenarios, existing GAN inversion methods mainly focus on the StyleGAN architecture (Karras et al., 2019; 2020; 2021), in which the images are generated sequentially from the

 … �

…

…
�
Inverter Generator Generator  Inverter Generator  **Inverter**
(, ) (�, )
Input Manip. Inv.

Input Manip. Inv.

Input Manip. Inv.

Input Manip. Inv.

Input Manip. Inv.

Input Manip. Inv.

Edit Edit Rec Edit Edit Rec Edit Edit Rec Edit Edit Rec Edit Edit Rec Edit Edit Rec
random noise z, the style code w and the transformed style codes w+. We thus name the corresponding spaces as Z, W and W+, respectively.

Inversion on Images: Existing methods regarding inverting StyleGANs can be generally categorised into three groups, the optimisation-oriented, encoder-based, and hybrid methods. Based on either gradient descent solvers (Yeh et al., 2017; Zhu et al., 2016; Fang & Schwing, 2019) or gradientfree strategies (Huh et al., 2020; Abdal et al., 2019; 2020), the optimisation-oriented methods exhaustively seek the best latent representation for each image , at the cost of heavy computational complexity . On the other hand, the encoder-based methods focus on achieving universal inversion, with the goal to learn general solutions regarding image inversion. The hierarchical encoder architecture is typically employed to embed multiple scales into transformed styles W+ (Richardson et al., 2021) . Advanced inversion methods accommodate the reconstruction-editing trade-off by a two-phase strategy, in which the first phase aims to retain the editing ability in the W (or W+) space, and additional modules are developed in the second phase so as to compensate the reconstruction error (Wang et al., 2022; Dinh et al., 2022; Li et al., 2023; Pehlivan et al., 2023) . The above encoders can also be combined with the optimisationoriented methods, in which the encoders provide a welldefined initialisation for the optimisation-oriented methods (Zhu et al., 2016; Hussein et al., 2020; Roich et al., 2022; Alaluf et al., 2022). However, all the above methods mainly focus on inverting images, which fall short in retaining the local curvature and thus inevitably exhibit deficiency on inverting manipulations of StyleGANs. Inversion on Latent Representations: Since the latent spaces of StyleGANs including Z and W spaces have been proved to possess rich semantics (Hark ¨ onen et al. ¨ , 2020; Shen et al., 2020b; Abdal et al., 2019), we also witnessed several recent inversion methods that regularise the align110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 ment within the latent spaces (Tov et al., 2021; Bau et al., 2019; Zhu et al., 2020; 2024). Latent space regularisation can find its roots in bi-direction generation of training GANs, by either catering for theoretical completeness (Li et al., 2022) or practice benefits (Ding et al., 2020; Dumoulin et al., 2016). However, since their primary goal focuses on the generation quality, these methods still suffer from inaccurate restoration of semantics and reconstruction of images. Regarding inversion based on pre-trained GANs, in addition to reconstructing images by pixel-wise loss, the E4E method (Tov et al., 2021) also develops a discriminator in the W space, so as to regularise the latent representations from the trained encoder to be similar to the original generating space, and correspondingly proposed a metric called latent editing consistency to measure the editing capability. On the other hand, Bau *et al.* (Bau et al., 2019) proposed to pretrain an encoder by inverting the randomly sampled latent representation in Z space, which is then used for the following layer-wise optimisation. Zhu *et al.* (Zhu et al., 2020) further proposed an optimisation-oriented method, which inverts images with the assistant of in-domain image prior in the Z space. However, for the encoder-based methods, it is obvious that without any additional constraints, the inversion in the latent space is ill-posed, since the minimisation of ||f(x)−f(g(f(x)))||22can find its bad local minimum at 0 for any surjection f(x) = c, where f is the encoder to be optimised, x represents the input image, g denotes the fixed generator and c is any constant. More importantly, existing methods in the latent space are based on the pointwise estimation. As shall be shown shortly, the point-wise estimation, oftentimes taken for granted, is proved to be insufficient in indicating the semantic discrepancy.

## 3. Analysis On Latent Generating Space

Compared with existing GAN models, the StyleGAN architecture has achieved the state-of-the-art performance when generating realistic and diversifying images (Karras et al., 2020; Sauer et al., 2022); this essentially ensures the richness of semantics in the latent generating space (Abdal et al., 2019; 2020). Therefore, we mainly focus on analysing the latent space of StyleGAN, by revealing several important findings and properties that motivate our follow-up method for inverting the manipulations. Finding 1: *There exist multiple local domains in the latent* space that correspond to the same person identities. We first analyse the correspondence between the latent space and generated images, based on the StyleGAN model. More specifically, our analysis is based on the officially released model (Karras et al., 2020), which is adopted in almost all inversion methods based on StyleGAN. By inspecting the generator, we essentially find that the domains of w ∈ W
and w+ ∈ W+ obtained by sampling z from the standard 

dom.

 B
origin +smile +glasses **+makeup**
dom
. A 
do m. B
origin +smile +glasses **+makeup**
do m. A 
dom
. B 
dom. B 
do m. A 
do m. A 
(a) Local Domains (b) Anisotropy Property α Im ag e MSE0.072 0.0750.088 0.081 0.093 0.095 0.086 α Im ag e MSE0.072 0.0750.088 0.081 0.093 0.095 0.086 0.0024 0.051 MSE↓
0.0024 0.051 MSE↓
(c) Inconsistent Variation Figure 2. In-depth analysis on the lantent generating space of Style- GAN. (a) represents multiple local domains corresponding to the same person identities. (b) represents the anisotropy property across different directions. (c) denotes the inconsistency between the latent space and image space.

Gaussian distribution (named as the *sampling domain*) are not well aligned with those obtained from inverting methods (named as the *inverting domain*). The average l2 norm distance between those two domains is much larger than that between the generated images from the two domains, exhibiting that those two domains are well separated in the latent space. More importantly, both of the two domains, even with their interpolations, are able to reconstruct extremely similar images with the same person identity, as illustrated in Fig. 2-(a). This reveals that the similarity of two images may not be sufficient to guarantee the closeness in the latent space. On the other hand, the similar images across the interpolation indicate that certain direction in the latent space cannot alter the image semantics, which is further analysed by the following findings. Finding 2: *When manipulated by directions with the same* scale, the generated images exhibit distinct anisotropy across semantics. The rich semantics within the low-dimensional latent space allow for flexible manipulations. We investigate the impact of manipulations on the W space, the *de facto* choice for the majority GAN inversion methods (Richardson et al., 2021; Tov et al., 2021; Alaluf et al., 2022; Dinh et al., 2022; Alaluf et al., 2021; Hu et al., 2022; Pehlivan et al., 2023; Wang et al., 2022; Yao et al., 2022). We represent the generation from the W space as {g(w) : w *∈ W}*, and analyse the manipulated generation g(w+v) for arbitrary v that satisfies both ||v||22 = β and (w+v) ∈ W. Please note that β is the constant that restricts the scale of manipulating direction v to possess the same length of the l2-norm. We then plot sets of g(w + v) and g(w) for different v in Fig. 2-(b). From this figure, when manipulated by the 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 same scale vector within the W space, we can conclude that the variations of generated images are distinct, in which certain random directions eventually anisotropically change generated person identities. In other words, the variation of an image within the same person identity essentially corresponds to a curved latent space. Finding 3: When sequentially edited by a fixed semantic direction, the generated images exhibit inconsistent variation against the unedited image. Given the fact that GAN inversion searches for the best representation in the latent space to minimise the discrepancy between generated and input images, we then analyse the relationship between the deviation in the latent space and the variation of the corresponding generated images.

More specifically, we investigate the deviation on the W
space by directions with explicit semantics, which can be obtained by InterfaceGAN (Shen et al., 2020a) for the face images and GANSpace (Hark ¨ onen et al. ¨ , 2020) for the car and church images. Given a normalised semantic direction
{e : ||e||2 = 1}, we are able to calculate the variation of generated images by ||g(w) − g(w + α · e)||22, in which w ∈ W, α ∈ R
1 denotes the deviation scale, g(·) denotes the generation process and the variation is evaluated by the MSE metric *|| · ||*22. We illustrate in Fig. 2-(c) regarding the MSE values between edited and unedited images, along with the change of scale α in the latent space. From this figure, we can conclude that when increasing the scales given a direction, the generated images, although still possessing the same identity, exhibit inconsistent MSE values, sometimes even have decreased MSE results. In other words, given fixed (or slightly fine-tuned) generators, minimising MSE on images may even result into the increase of deviation in the latent space, thus preventing from finding the best latent representation. In contrast, considering the curvature within the latent space is beneficial to achieve the global optima.

## 4. Methodology 4.1. Latent Manipulation Inversion

Basically, GAN inversion seeks to accurately restore realistic images, whereas *Finding 1* indicates that directly inverting images can result into sub-optimal results due to multiple local domains in the latent space. Since the latent space of StyleGAN possesses rich semantics, we propose to restrict the inversion within the latent space to ensure the consistency on semantics. Indeed, the inversion essentially requires the bijection from the encoder to the generator for real-world images, i.e., x = g(f(x)), which is equivalent to the bijection from the generator to the encoder within a certain local domain, namely, w = f(g(w)). Therefore, performing the inversion within the latent space can also contribute to improving the inversion for restored images.

The other important criteria of inversion is the quality of semantics of the inverted latent feature, i.e., retaining the realism of editing. Given the fact that the GAN inversion restores realistic images, the preferred GAN inversion thus has to restore arbitrarily edited images. When Assumption 1 exists, **Lemma** 4.2 ensures the equivalence between inverting arbitrarily edited images and inverting arbitrary manipulation in the latent space, thus providing a new way of improving the GAN inversion. Assumption 4.1. The generator of StyleGAN is locally Lipschitz and operates as continuous mapping from the latent space W to the image space (Arjovsky et al., 2017). More importantly, there exists a local domain that the generation is injective, which is also the prerequisite for GAN inversion.

Lemma 4.2. Let g(·) *denote the pre-trained generator, and* f(·) to represent the inversion encoder. Given an arbitrary latent feature w from an image w = f(x) *and direction* v ∈ Bϵ(w), where Bϵ(w) *represents an open ball of* w with radius ϵ*, we represent the edited image by* xe = g(w + v). Then, the arbitrarily edited image xe can be precisely inverted, i.e., xe = g(f(xe)), if and only if we are able to invert the manipulation, i.e., f(xe) − v = f(x).

Proof. Please refer to the Appendix-C for the proof.

Besides performing the inversion for the latent features, we thus propose to invert the manipulation within the latent space, which basically calls for the consistency of local curvature surrounding the inverted latent feature. Given a style code w, manipulation inversion can be formally achieved by minimising the follow objective:

$$\min_{f}{\cal L}_{r}=\min_{f}\int||f(g({\bf w}+\beta\frac{{\bf v}}{||{\bf v}||_{2}}))-\beta\frac{{\bf v}}{||{\bf v}||_{2}}-{\bf w}||_{2}^{2}d{\bf v},\tag{1}$$

where v/||v||2 denotes unit manipulation and β denotes the constant scale to retain within the same identity. In (1),
recall that g(·) denotes the fixed generator and f(·) represents the inversion encoder to be optimised. As proved by Lemma 4.2, minimising (1) essentially ensures the ability of restoring arbitrarily edited images.

## 4.2. Latent Statistical Manifold

We propose to align the distributions within the latent space, named as distribution preserving embedding (DPE), as a well-defined proxy of local curvature. This essentially requires to establish the latent probabilistic model for w. As analysed by *Finding 2*, the latent style features w exhibit anisotropic property across image semantics, and we thus cannot rely on the isotropic Gaussian assumption that is employed in typical settings. Correspondingly, we extend the Gaussian model in (Wulff & Torralba, 2020) to the widely applied factor model for latent codes, as follows,

$$\mathbf{w}=\mathbf{S}^{T}\mathbf{n}+{\boldsymbol{\epsilon}}+\mathbf{c},$$
T n + ϵ + c, (2)
where c relates to the conditions given by the inversion encoder f(x) and mapping network h(z), S is the projection matrix, n and ϵ denote two independent random variables that satisfy Gaussian distributions. Please note that h(z) is fixed and f(x) is encouraged to approach h(z), such that the manipulation inversion in the latent style code is optimised.

220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 More importantly, since the style code feature w ∈ W possesses rich semantics as reflected by *Findings 1&2&3*, we choose S as the semantic matrix, in which each column of S represents one semantic direction. As proved in various works (Shen et al., 2020b), all the latent vectors corresponding to the same attribute of generated images should be reachable through the direct path between them, and the direct path is chosen as one semantic direction in S. In this way, n ∼ N (0, I) and S
T n randomly combines various semantic directions to generate the style code, which also facilitates diverse and complete semantics of generated images. On the other hand, ϵ essentially represents the nuisance noise that denotes the randomness of generated images whilst not altering the semantics. This can be established by ϵ = J
T η, whereby η ∼ N (0, I) denotes the random Gaussian noise on the image, and J is the Jacobian matrix when generating images from w ∈ W, which maps the nuisance noise at the image side to latent style codes.

Therefore, according to (2), we are able to model the distributions output from the inversion encoder f(x) and mapping network h(z), as N (f(x), S
T S + J
T J) and
N (h(z), S
T S + J
T J), respectively. This way, we can minimise the distance between the two Gaussian distributions, so as to accommodate the manipulation inversion in (1) by random directions v/||v||. More importantly, f(x) and h(z) now represent two distributions. In other words, given two Gaussian distributions, the way to optimise f(x) shall follow the shortest path given by the distribution discrepancy between N (f(x), S
T S + J
T J) and N (h(z), S
T S + J
T J).
This naturally motivates us to establish the statistical manifold Mw for two Gaussian distributions, by the Cramer-Rao
distance (Amari, 2016), in which the Riemannian metric is
Mar, 2016), in which the fundamental model  $ds^2=d\mathbf{w}^T(\mathbf{S}^T\mathbf{S}+\mathbf{J}^T\mathbf{J})^{-1}d\mathbf{w}$  . 
−1dw (3)
More importantly, taking advantages of the equivalence against the inner product of directional derivative on the Riemannian manifold and within the Euclidean space (Absil et al., 2008), we are able to calculate the Riemannian gradient on the established statistical manifold Mw. More specifically, given any smooth loss function ϕ(w) and any directional derivative dξ, we can calculate the gradient on the Riemannian manifold, i.e., Riemannian manifold, by the following equivalence
∇rϕ(w)
T(S
T S + J
T J)
−1dξ = ∇eϕ(w)
Tdξ, (4)
where ∇rϕ(w) denotes the Riemannian gradient on Mw and ∇eϕ(w) is the Euclidean gradient. As (4) holds for arbitrary directional derivative, we can choose linear independent directional derivatives dξ to compose a full-rank matrix Λ. Then, we have

$$\nabla_{r}\phi({\bf w})^{T}({\bf S}^{T}{\bf S}+{\bf J}^{T}{\bf J})^{-1}{\bf\Lambda}=\nabla_{r}\phi({\bf w})^{T}{\bf\Lambda},\tag{5}$$

such that the Riemannian gradient is obtained by

$$\nabla_{r}\phi(\mathbf{w})=\nabla_{e}\phi(\mathbf{w})^{T}(\mathbf{S}^{T}\mathbf{S}+\mathbf{J}^{T}\mathbf{J}).$$
$$(6)$$
T J). (6)
In practice, we follow (Shen et al., 2020b) to calculate the semantic matrix S and (Ramesh et al., 2018) to calculate the Jacobian matrix J.

## 4.3. Adversarial Learning To Invert Manipulation

The remaining task is to invert the manipulation by minimising (1), based on the established manifold Mw. More importantly, the inversion of manipulation requires the excessive numeration on the manipulation direction v to calculate the integration. This, however, is intractable in practice. Although matching the distributions in the latent style code space can contribute to the manipulation inversion, an accurate inversion still requires to enumerate v based on (1). To relieve this issue, we propose to use the adversarial learning to choose the "best" direction that maximises (1), namely,

$$({\mathfrak{I}}{\mathfrak{I}})$$
$${\cal L}_{r}^{*}=\operatorname*{max}_{\mathbf{v}}||f(g(\mathbf{w}+\beta\frac{\mathbf{v}}{||\mathbf{v}||_{2}}))-\beta\frac{\mathbf{v}}{||\mathbf{v}||_{2}}-\mathbf{w}||_{2}^{2}.\tag{7}$$

The adversarial learning L
∗ressentially operates as an upper bound of (1). This way, minimising L
∗rcan ensure the minimisation of (1) to retain the manipulation inversion. To achieve the adversarial learning, we formulate the properties of the loss function ψ, which is represented as:

$$\psi({\bf w},{\bf v})=||f(g({\bf w}+\beta\frac{{\bf v}}{||{\bf v}||_{2}}))-\beta\frac{{\bf v}}{||{\bf v}||_{2}}-{\bf w}||_{2}^{2}\tag{8}$$

275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329

| car scenarios. The best performance is highlighted in red and the second-best performance in blue. Human Face Church   | Cars                                                                                         |        |        |        |        |        |        |        |        |        |        |        |
|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Method                                                                                                                 | MSE ↓ LPIPS ↓ SSIM ↑ MS-SSIM ↑ MSE ↓ LPIPS ↓ SSIM ↑ MS-SSIM ↑ MSE ↓ LPIPS ↓ SSIM ↑ MS-SSIM ↑ |        |        |        |        |        |        |        |        |        |        |        |
| pSp (Richardson et al., 2021)                                                                                          | 0.0500                                                                                       | 0.3656 | 0.5875 | 0.7196 | 0.1103 | 0.8279 | 0.3879 | 0.4226 | 0.7339 | 1.6028 | 0.2148 | 0.0564 |
| E4E (Tov et al., 2021)                                                                                                 | 0.0665                                                                                       | 0.4299 | 0.5668 | 0.6795 | 0.1713 | 1.0159 | 0.3604 | 0.3085 | 0.5290 | 1.2436 | 0.2394 | 0.0591 |
| ReStylepSp (Alaluf et al., 2021)                                                                                       | 0.0402                                                                                       | 0.2701 | 0.6057 | 0.7608 | 0.1822 | 0.7230 | 0.3589 | 0.4177 | 0.1493 | 0.6181 | 0.4922 | 0.5770 |
| ReStyleE4E (Alaluf et al., 2021)                                                                                       | 0.0602                                                                                       | 0.3961 | 0.5698 | 0.7022 | 0.2593 | 1.0308 | 0.3053 | 0.2636 | 0.2922 | 0.8751 | 0.4372 | 0.4539 |
| HyperInverter (Dinh et al., 2022) 0.0262                                                                               | 0.1645                                                                                       | 0.6594 | 0.8190 | 0.0921 | 0.3815 | 0.4248 | 0.6034 | -      | -      | -      | -      |        |
| HFGI (Wang et al., 2022)                                                                                               | 0.0446                                                                                       | 0.3198 | 0.5817 | 0.7481 | 0.1566 | 0.9032 | 0.3642 | 0.3811 | -      | -      | -      | -      |
| FSE (Yao et al., 2022)                                                                                                 | 0.0223                                                                                       | 0.1839 | 0.7115 | 0.8625 | 0.0573 | 0.3275 | 0.4883 | 0.7236 | 0.0772 | 0.3617 | 0.5399 | 0.7092 |
| E2Style (Wei et al., 2022)                                                                                             | 0.0481                                                                                       | 0.4148 | 0.6253 | 0.7590 | 0.0554 | 0.3097 | 0.5244 | 0.7538 | -      | -      | -      | -      |
| StyleRes (Pehlivan et al., 2023)                                                                                       | 0.0366                                                                                       | 0.5707 | 0.6440 | 0.7205 | -      | -      | -      | -      | -      | -      | -      | -      |
| Ours                                                                                                                   | 0.0139                                                                                       | 0.1263 | 0.7414 | 0.8931 | 0.0458 | 0.2691 | 0.5437 | 0.7847 | 0.0486 | 0.3001 | 0.5987 | 0.7726 |

Then, we define the distance D(w, v) = ||ψ(w, v) −
ψ(w, 0)||22, and approximate this using a Taylor expansion. The virtual editing reaches a maximum v
∗through the power iteration method applied to the principal eigenvector of the Hessian (Golub & der Vorst, 2000).

In practice, to optimise the manipulation inversion within a well-behaved latent space, we employ a fixed encoder as the projector to generate w, as illustrated in Fig. 3. We then calculate ψ and update w. After obtaining w′ = w + v
∗,
we optimise the inverter by minimizing the objective in (1), ultimately generating the final images. Therefore, our final loss becomes

$${\mathcal{L}}={\mathcal{L}}_{r}+\lambda_{1}{\mathcal{L}}_{j}+\lambda_{2}{\mathcal{L}}_{s}+\lambda_{3}{\mathcal{L}}_{\mathrm{ori}}$$

where λ1, λ2 and λ3 are hyperparameters and are empirically set to 1.0, 0.8 and 3.0, respectively, while Lori represents the original inversion loss, i.e., Lori = Lmse + λo1Llpips + λo2Lid.

## 5. Experiment 5.1. Experimental Settings

Dataset: Our experimental evaluations were performed based on various scenarios. For the widely tested human face scenarios, we employed the high-quality face dataset, i.e., Flickr-Faces-HQ Dataset (FFHQ) (Karras et al., 2019) dataset for training, and evaluated based on the CelebA- HQ (Karras et al., 2018) dataset for the inversion. Both resolution for the FFHQ and CelebA-HQ datasets is 1024 × 1024. We also evaluated our method for the car scenario, based on the 512 × 512 images within the Stanford Cars
(Krause et al., 2013) to serve for training and evaluation with the official split. We further evaluated the challenging scenery images by the church scenario, including 256×256 images within the LSUN Church dataset (Yu et al., 2015) and also followed the official data splitting strategy.

Baselines We compared our method with state-of-the-art image inversion methods, including classical methods such as pixel2style2pixel (pSp) (Richardson et al., 2021) and encoder for editing (E4E) (Tov et al., 2021), and most recent methods such as residual-based StyleGAN (ReStyle) (Alaluf et al., 2021) and E2Style (Wei et al., 2022), HFGI (Wang et al., 2022), HyperInverter (Dinh et al., 2022), FeatureStyleEncoder (FSE) (Yao et al., 2022) and StyleRes (Pehlivan et al., 2023), which benefit from multi-stage and multi-level information. Literately, we used the official pretrained weights and configurations released by the authors to perform our evaluation experiments. For the Stanford Car dataset and LSUN Church dataset, several methods are omitted from comparisons when the models were not released. When evaluating the editing and manipulation, another important criterion for GAN inversion, we run extensive experiments leveraging InterfaceGAN (Shen et al., 2020a) for the human face images and GANSpace (Hark ¨ onen et al. ¨ , 2020) for the car and church images to ensure diverse editing directions. More specifically, for the face images, we adopted the edit direction from the previous method (Yao et al., 2022), using the smiling, eyeglasses and heavy makeup boundaries trained by InterFaceGAN. For the car and church images, we computed PCA directions following the official GANSpace implementation (Hark ¨ onen et al. ¨ , 2020).

$$({\mathfrak{g}})$$

Implementation details In our experiments, the pretrained StyleGAN generator was directly sourced from the Style- GAN2 repository (Karras et al., 2020). We then employed the same pretrained encoder as a fixed component for implementing the projector. We adopted the backbone design from FSE (Yao et al., 2022). Then, we followed the previous encoder-based methods (Richardson et al., 2021; Tov et al., 2021; Alaluf et al., 2021), with the Ranger optimizer, which combined the Lookahead (Zhang et al., 2019) and the Rectified Adam (Liu et al., 2019) optimizer for training. we set the learning rate and other parameters as lr = 0.0001, β1 = 0.95, β2 = 0.999. To guarantee the image domain not varying a lot for some special point, we set the λo1 = 0.8 and λo2 = 0.1 for the image domain loss.

All evaluation experiments were conducted using a single NVIDIA GeForce RTX 4090 GPU.

pSp [32] E4E [37] ReStylepSp
[4] ReStylee4e Input [4] HyperInv. [10] HFGI [38] FSE [44] E2Style [41] Ours StyleRes [30]
±
Smili ng
±
Ey eglas se s 
±
M
ake up Figure 4. Results of manipulation inversion across multiple editing directions on human faces. The notations including ± indicates that the image is first edited by the manipulation in the latent generating space, followed by the inversion to restore the original input.
330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384

## 5.2. Comparisons On Manipulation Inversion

As mentioned in this paper, the inversion of manipulation can well reflect the realism of inversion. We first compared our method for the manipulation inversion and the results are reported in Table 1. As can be seen from this table, our method significantly outperforms other approaches in terms of manipulation inversion across both domains, demonstrating substantial improvements in all evaluation metrics. Additional comparisons on editing realism are provided in Appendix-A. We further show subjective results in Fig. 4, whereas our method exhibits superior performances during manipulation inversion and achieves the best manipulation inversion accuracy. Additional qualitative results can be found in Appendix-B.

## 5.3. Comparisons On Reconstruction Accuracy

As proved in the Sec. 4.1, the manipulation inversion optimises both the reconstruction and editing aspects. We thus systematically conduct a series of experiments to underscore the alignment between manipulation inversion and existing evaluation metrics of GAN inversion, substantiating their compatibility and effectiveness. More specifically, we conducted experiments on different challenging scenarios which are the *de facto* choice to evaluate the performances for GAN inversion tasks. Fig. 6 illustrates the samples of our reconstruction results and the comparison with existing baseline methods is provided in Table 2. Again, our method outperforms other encoder-based methods for reconstruction accuracy in all scenarios, exhibiting the superior perfor-

| formance is highlighted in red and the second-best in blue. Method MSE ↓ LPIPS ↓ SSIM ↑ MS-SSIM ↑ pSp (Richardson et al., 2021) 0.0497 0.2790 0.6218 0.7208 E4E (Tov et al., 2021) 0.0663 0.3510 0.5985 0.6812 ReStylepSp (Alaluf et al., 2021) 0.0401 0.2050 0.6394 0.7613 ReStyleE4E (Alaluf et al., 2021) 0.0600 0.3232 0.6015 0.7032 HyperInverter (Dinh et al., 2022) 0.0256 0.1481 0.6722 0.8105 HFGI (Wang et al., 2022) 0.0445 0.1803 0.6937 0.7495 FSE (Yao et al., 2022) 0.0215 0.0990 0.7550 0.8678 E2Style (Wei et al., 2022) 0.0481 0.3037 0.6595 0.7591 Ours 0.0133 0.0810 0.7670 0.8971   |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

mances of our method that focuses on aligning distributions in the latent space to invert the arbitrary manipulation.

## 5.4. Ablation Study

To demonstrate the impact of each component in our method, we conducted step-by-step experiments to validate the effectiveness of the additional metrics and operations within the latent space. The MSE constraint on the latent space is referred to as the *latent restriction*. We denote Σ = J
T J
as the *Jacobian component* and Σ = S
T S as the semantic component. The term *Adversarial learning* refers to the adversarial training proposed in Sec. 4.3. We report the results in Table 3. Notably, the latent restriction alone significantly improves the manipulation inversion results, as evidenced by the LPIPS metric and other metrics. Furthermore, the Jacobian component introduces additional constraints, effectively aligning transformations in both the image and latent domains. This alignment leads to substantial improvement in the metrics, indicating enhanced consistency and accu385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439

Figure 6. Illustration on the reconstruction accuracy of our method for GAN inversion.

racy in the inversion process. Our method, by sequentially incorporating all the components, consistently improves the inversion accuracy of manipulation, thus proving the effectiveness of each component.

Table 3. Ablation study evaluations on manipulation inversion against latent restriction, Jacobian component, semantic component and adversarial learning, which are 4 key components in the proposed method.

MSE ↓ LPIPS ↓ SSIM ↑ **MS-SSIM** ↑

Baseline 0.0223 0.1839 0.7115 0.8625

+ Latent restriction 0.0212 0.1491 0.7217 0.8678 + Jacobian component 0.0154 0.1277 0.7354 0.8694 + Semantic component 0.0144 0.1265 0.7403 0.8926 + Adversarial learning 0.0139 0.1263 0.7414 **0.8931**

## 5.5. Compatibility On Different Architectures

To demonstrate the universal property of our method, especially for the applicability across various encoder-based methods, we integrated different encoder types into our experiments, including a simple latent encoder (i.e., pSp), a two-phase encoder utilizing the shallow feature (i.e., HFGI), a multi-stage method incorporating the shallow feature (i.e.,
E2Style) and the state-of-the-art result (i.e., FSE). Our distri-

Input Manip. Manip. Inv Input Manip. Manip. Inv Input Manip. Manip. Inv Input Manip. Manip. Inv Input Manip. Manip. Inv Input Manip. Manip. Inv Input Manip. Manip. Inv Input Manip. Manip. Inv pSp [32] Enhanced pSp HFGI [38]
pSp [32] Enhanced pSp HFGI [38]
Enhanced HFGI
Enhanced HFGI
pSp (Richardson et al., 2021) Enhanced pSp HFGI (Wang et al., 2022) Enhanced HFGI
E2Style [41] Enhanced E2Style FSE [44] Enhanced FSE
E2Style (
E2Style [41]
Wei et al., 2022) Enhanced E2Style FSE (
Enhanced E2Style FSE [44]
Yao et al., 2022) Enhanced FSE
Enhanced FSE
Figure 5. Comparisons between existing typical architectures and those enhanced by our method, including pSp (Richardson et al., 2021), HFGI (Wang et al., 2022), E2Style (Wei et al., 2022), and FSE (Yao et al., 2022). For each architecture, we show the input image, the manipulated image, and the inversion result. The enhanced versions demonstrate the improvements achieved by integrating our method.

Table 4. The results of existing architectures enhanced by our method. The best results are highlighted in **Bold**.

MSE ↓ LPIPS ↓ SSIM ↑ **MS-SSIM** ↑
pSp (Richardson et al., 2021) 0.0500 0.3656 0.5875 0.7196 Enhanced pSp **0.0478 0.3473 0.5934 0.7290**
HFGI (Wang et al., 2022) 0.0446 0.3198 0.5817 0.7481 Enhanced HFGI **0.0315 0.2618 0.6201 0.7905**
E2Style (Tov et al., 2021) 0.0481 0.4148 0.6253 0.7590 Enhanced E2Style **0.0453 0.3497 0.6271 0.7650**
FSE (Yao et al., 2022) 0.0223 0.1839 0.7115 0.8625 Enhanced FSE **0.0139 0.1263 0.7414 0.8931**
bution estimation training was systematically applied to realign the latent code within their respective latent spaces. All the training strategies were the same, with the exception of HFGI. Given the fact that HFGI refines images exclusively in the second stage, we adapt its methodology by initially training an E4E encoder, followed by training the secondstage consultation encoder using the default procedure of HFGI. The results are reported in Table 4, which exhibits the consistent improvements when using our method. Additionally, subjective results in Fig. 5 illustrate the effectiveness of integrating our method across different architectures.

## 6. Conclusion

In this paper, we have systematically analysed the latent generating space of generative adversial network (GAN), by realising that the local curvature exists when inverting images. Motivated by this, we have proposed a new strategy, namely, inverting manipulations, instead of inverting images, by modelling the latent space as probabilistic models, and corespondingly establishing the statistical manifold. We then further proposed an adversarial training method to achieve efficient optimisation on calculating the manipulation inversion loss. The proposed method can flexibly act as plugin method to improve the inversion performances on different architectures. Experimental results have demonstrated superior performances on both reconstruction accuracy and editing realism.

R

e c o n I

n p u t R

e c o n I

n p u t

## Broader Impact

The proposed method bridges GAN inversion and statistical manifold theory to unify reconstruction accuracy and editing realism, offering a novel perspective for high-fidelity image manipulation. By establishing a latent statistical manifold and adversarial optimization, our framework serves as a universal plugin for diverse GAN architectures, enabling seamless integration into applications such as medical imaging restoration, artistic content generation, and video compression. This universality reduces the need for architecture-specific adaptations, broadening its adoption in cross-domain tasks. Furthermore, our adversarial strategy for minimizing manipulation inversion metrics introduces a computationally efficient paradigm for latent space optimization. This could inspire future research in unsupervised representation learning, particularly in scenarios requiring robustness to semantic perturbations, such as domain adaptation or anomaly detection. However, enhanced editing realism may lower the barrier for generating deceptive content (e.g., deepfakes). To mitigate misuse risks, we advocate for ethical guidelines and detection frameworks to accompany such advancements. Future work should explore embedding traceability mechanisms within the latent space and fostering public awareness of synthetic media risks. By balancing innovation and responsibility, our method aims to advance generative technologies while safeguarding societal trust.

## References

Abdal, R., Qin, Y., and Wonka, P. Image2stylegan: How to embed images into the stylegan latent space? In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4432–4441, 2019.

Abdal, R., Qin, Y., and Wonka, P. Image2stylegan++: How to edit the embedded images? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8296–8305, 2020.

Absil, P.-A., Mahony, R., and Sepulchre, R. *Optimization* Algorithms on Matrix Manifolds. Princeton University Press, 2008.

Alaluf, Y., Patashnik, O., and Cohen-Or, D. Restyle: A
residual-based stylegan encoder via iterative refinement. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 6711–6720, 2021.

Alaluf, Y., Tov, O., Mokady, R., Gal, R., and Bermano, A. Hyperstyle: Stylegan inversion with hypernetworks for real image editing. In *Proceedings of the IEEE/CVF*
440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18511–18521, 2022.

Amari, S. *Information Geometry and Its Applications*, volume 194. Springer, 2016.

Arjovsky, M., Chintala, S., and Bottou, L. Wasserstein generative adversarial networks. In *Proceedings of the* International Conference on Machine Learning (ICML), pp. 214–223. PMLR, 2017.

Bau, D., Zhu, J.-Y., Wulff, J., Peebles, W., Strobelt, H.,
Zhou, B., and Torralba, A. Seeing what a gan cannot generate. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4502–4511, 2019.

Ding, R., Guo, G., Yan, X., Chen, B., Liu, Z., and He, X.

Bigan: Collaborative filtering with bidirectional generative adversarial networks. In *Proceedings of the 2020* SIAM International Conference on Data Mining (SDM), pp. 82–90. SIAM, 2020.

Dinh, T. M., Tran, A. T., Nguyen, R., and Hua, B.-S. Hyperinverter: Improving stylegan inversion via hypernetwork. In *Proceedings of the IEEE/CVF Conference on* Computer Vision and Pattern Recognition (CVPR), pp. 11389–11398, 2022.

Dumoulin, V., Belghazi, I., Poole, B., Mastropietro, O.,
Lamb, A., Arjovsky, M., and Courville, A. Adversarially learned inference. *arXiv preprint arXiv:1606.00704*, 2016.

Fang, T. and Schwing, A. Co-generation with gans using ais based hmc. Advances in Neural Information Processing Systems, 32, 2019.

Galteri, L., Seidenari, L., Bertini, M., Uricchio, T., and Bimbo, A. D. Fast video quality enhancement using gans. In Proceedings of the 27th ACM International Conference on Multimedia, pp. 1065–1067, 2019.

Golub, G. H. and der Vorst, H. A. V. Eigenvalue computation in the 20th century. Journal of Computational and Applied Mathematics, 123(1-2):35–65, 2000.

Hark ¨ onen, E., Hertzmann, A., Lehtinen, J., and Paris, S. ¨
Ganspace: Discovering interpretable gan controls. Advances in Neural Information Processing Systems, 33:
9841–9850, 2020.

Hu, X., Huang, Q., Shi, Z., Li, S., Gao, C., Sun, L., and Li, Q. Style transformer for image inversion and editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp.

11337–11346, 2022.

Huh, M., Zhang, R., Zhu, J.-Y., and Paris, S. Transforming and projecting images into class-conditional generative networks. In *European Conference on Computer Vision* (ECCV), pp. 17–34. Springer, 2020.

Hussein, S. A., Tirer, T., and Giryes, R. Image-adaptive gan based reconstruction. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 3121–3129, 2020.

Kang, M., Zhu, J.-Y., Zhang, R., Park, J., Shechtman, E.,
Paris, S., and Park, T. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10124–10134, 2023.

Karras, T., Aila, T., Laine, S., and Lehtinen, J. Progressive growing of gans for improved quality, stability, and variation. In *Proceedings of the International Conference on* Learning Representations (ICLR), 2018.

Karras, T., Laine, S., and Aila, T. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4401–4410, 2019.

Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J.,
and Aila, T. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8110–8119, 2020.

Karras, T., Aittala, M., Laine, S., Hark ¨ onen, E., Hellsten, ¨
J., Lehtinen, J., and Aila, T. Alias-free generative adversarial networks. In Proceedings of the 35th Annual Conference on Neural Information Processing Systems
(NeurIPS 2021), pp. 852–863, 2021.

Krause, J., Stark, M., Deng, J., and Fei-Fei, L. 3d object representations for fine-grained categorization. In Proceedings of the IEEE International Conference on Computer Vision Workshops (ICCV Workshops), pp. 554–561, 2013.

Li, B., Ma, T., Zhang, P., Hua, M., Liu, W., He, Q., and Yi, Z.

Reganie: Rectifying gan inversion errors for accurate real image editing. In *Proceedings of the AAAI Conference on* Artificial Intelligence, volume 37, pp. 1269–1277, 2023.

Li, S., Yu, Z., Xiang, M., and Mandic, D. Reciprocal gan through characteristic functions (rcf-gan). IEEE Transactions on Pattern Analysis and Machine Intelligence, 45 (2):2246–2263, 2022.

Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., and Han, J. On the variance of the adaptive learning rate and beyond. In *Proceedings of the International Conference* on Learning Representations (ICLR), 2019.

Mentzer, F., Toderici, G. D., Tschannen, M., and Agustsson, E. High-fidelity generative image compression. *Advances* in Neural Information Processing Systems, 33:11913– 11924, 2020.

Mentzer, F., Agustsson, E., Balle, J., Minnen, D., Johnston, ´
N., and Toderici, G. Neural video compression using gans for detail synthesis and propagation. In *European* Conference on Computer Vision (ECCV), pp. 562–578. Springer, 2022.

Pehlivan, H., Dalva, Y., and Dundar, A. Styleres: Transforming the residuals for real image editing with stylegan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1828–1837, 2023.

Ramesh, A., Choi, Y., and LeCun, Y. A spectral regularizer for unsupervised disentanglement. arXiv preprint arXiv:1812.01161, 2018.

Richardson, E., Alaluf, Y., Patashnik, O., Nitzan, Y., Azar, Y., Shapiro, S., and Cohen-Or, D. Encoding in style: A stylegan encoder for image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2287–2296, 2021.

Roich, D., Mokady, R., Bermano, A. H., and Cohen-Or, D. Pivotal tuning for latent-based editing of real images. ACM Transactions on Graphics (TOG), 42(1):1–13, 2022.

Sauer, A., Schwarz, K., and Geiger, A. Stylegan-xl: Scaling stylegan to large diverse datasets. In ACM SIGGRAPH 2022 Conference Proceedings, pp. 1–10, 2022.

Shen, Y., Gu, J., Tang, X., and Zhou, B. Interpreting the latent space of gans for semantic face editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9243–9252, 2020a.

Shen, Y., Yang, C., Tang, X., and Zhou, B. Interfacegan:
Interpreting the disentangled face representation learned by gans. *IEEE Transactions on Pattern Analysis and* Machine Intelligence, 44(4):2004–2018, 2020b.

Tov, O., Alaluf, Y., Nitzan, Y., Patashnik, O., and Cohen-Or, D. Designing an encoder for stylegan image manipulation. ACM Transactions on Graphics (TOG), 40(4):1–14, 2021.

Wang, T., Zhang, Y., Fan, Y., Wang, J., and Chen, Q. Highfidelity gan inversion for image attribute editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11379–11388, 2022.

Wang, X., Yu, K., Wu, S., Gu, J., and Liu, Y. Esrgan: Enhanced super-resolution generative adversarial networks.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 In *Proceedings of the European Conference on Computer* Vision (ECCV) Workshops, pp. 1–10, 2018.

Wang, X., Xie, L., Dong, C., and Shan, Y. Real-esrgan:
Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1905–1914, 2021.

Wei, T., Chen, D., Zhou, W., Liao, J., Zhang, W., Yuan, L.,
Hua, G., and Yu, N. E2style: Improve the efficiency and effectiveness of stylegan inversion. IEEE Transactions on Image Processing, 31:3267–3280, 2022.

Wulff, J. and Torralba, A. Improving inversion and generation diversity in stylegan using a gaussianized latent space. *arXiv preprint arXiv:2009.06529*, 2020.

Xia, W., Zhang, Y., Yang, Y., Xue, J.-H., Zhou, B., and Yang, M.-H. Gan inversion: A survey. *IEEE Transactions on* Pattern Analysis and Machine Intelligence, 2022.

Yao, X., Newson, A., Gousseau, Y., and Hellier, P. A
style-based gan encoder for high fidelity reconstruction of images and videos. In European Conference on Computer Vision (ECCV), pp. 581–597. Springer, 2022.

Yeh, R. A., Chen, C., Lim, T. Y., Schwing, A. G., Hasegawa-
Johnson, M., and Do, M. N. Semantic image inpainting with deep generative models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5485–5493, 2017.

Yu, F., Seff, A., Zhang, Y., Song, S., Funkhouser, T., and Xiao, J. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015.

Zhang, M., Lucas, J., Ba, J., and Hinton, G. E. Lookahead optimizer: k steps forward, 1 step back. In Proceedings of the Neural Information Processing Systems (NeurIPS), 2019.

Zhu, J., Shen, Y., Zhao, D., and Zhou, B. In-domain gan inversion for real image editing. In European Conference on Computer Vision (ECCV), pp. 592–608. Springer, 2020.

Zhu, J., Shen, Y., Xu, Y., Zhao, D., Chen, Q., and Zhou, B.

In-domain gan inversion for faithful reconstruction and editability. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(5):1056–1069, 2024.

Zhu, J.-Y., Krahenb ¨ uhl, P., Shechtman, E., and Efros, A. A. ¨
Generative visual manipulation on the natural image manifold. In *European Conference on Computer Vision* (ECCV), pp. 597–613. Springer, 2016.

550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604

## A. Edit Realism Result

Figure 8. Samples of edit results in car domain on the Stanford Cars (Krause et al., 2013) dataset.

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 As proved in the paper, the manipulation inversion optimises both the reconstruction and editing aspects. The inversion of manipulation can well reflect the realism of inversion. We first exhibiting the comparisons on editing realism result.

The editing realism result are shown in the Fig. 7 and 8. We show two edit directions of each image domain. For the face domain, we use the editing direction from InterfaceGAN (Shen et al., 2020b), for the car domain, we adapt the direction from GANSpace (Hark ¨ onen et al. ¨ , 2020).

Input Input 
+ 
Blu e S
ky 
+ Ora ng e 
+ 
Sum li gh t

+ G
ras s

Figure 7. Samples of edit results in face domain on the CelebA-HQ (Karras et al., 2018) dataset.

Input Inp ut 
+ 
Sm ile 
+ Bl ue S
ky 
+ E
yeg la ss 
+ S
u ml ig ht 
Our experimental evaluations were performed based on various scenarios. For the widely tested human face scenarios, we evaluated based on the CelebA-HQ (Karras et al., 2018) dataset for the inversion. We also evaluated our method for the car scenario, based on the 512 × 512 images within the Stanford Cars (Krause et al., 2013) to serve for training and evaluation with the official split.

+ Eye gl as s 

## B. Manipulation Inversion Result C. Proof Of Lemma 4.2

The proof of the Lemma 4.2 is listed below.

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 We further show subjective results in Fig. 9, where our method exhibits superior stability during manipulation inversion and achieves the best reconstruction.

Input + Grass ± Grass Input + Sunset ± **Sunset**
Figure 9. Manipulation inversion results in car domain on the Stanford Cars (Krause et al., 2013) dataset.
Lemma C.1 (Lemma 4.2). Let g(·) denote the pre-trained generator, and f(·) to represent the inversion encoder. Given an arbitrary latent feature w from an image w = f(x) and direction v ∈ Bϵ(w), where Bϵ(w) *represents an open ball of* w with radius ϵ, we represent the edited image by xe = g(w + v). Then, the arbitrarily edited image xe can be precisely inverted, i.e., xe = g(f(xe)), if and only if we are able to invert the manipulation, i.e., f(xe) − v = f(x).

Proof. We can prove the equivalence between inverting arbitrarily edited images and inverting manipulation, through sufficiency and necessity.

- Sufficiency: If any edited images can be inverted, we have xe = g(f(xe)) for any v ∈ Bϵ(w). On the other hand, xe is generated by xe = g(w + v). Then, we arrive at g(f(xe)) = xe = g(w + v).

As the generator g is continuous and acts as injection mapping, we can safely remove g(·) and thus have Input +Blue Sky ±Blue Sky Input +Sunlight **±Sunlight**
f(xe) = w + v Therefore, we prove the sufficiency f(x) = w = f(xe) − v for any v ∈ Bϵ(w).

- Necessity: Given f(xe) − v = f(x) for any v ∈ Bϵ(w), we thus have xe = g(f(xe)) = g(f(x) + v)
which obtains xe = g(w + v). This proves that any edited image can be precisely inverted.

This completes the proof.