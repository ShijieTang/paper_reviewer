000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Anonymous authors Paper under double-blind review

## Abstract

Test-Time Adaptation (TTA) methods use unlabeled test data to dynamically adjust models in response to distribution changes. However, existing TTA methods are not tailored for practical use on edge devices with limited computational capacity, resulting in a latency-accuracy trade-off. To address this problem, we propose SNAP-TTA, a sparse TTA framework that significantly reduces adaptation frequency and data usage, delivering latency reductions proportional to adaptation rate. It achieves competitive accuracy even with an adaptation rate as low as 0.01, demonstrating its ability to adapt infrequently while utilizing only a small portion of the data compared to full adaptation. Our approach involves (i) Class and Domain Representative Memory (CnDRM), which identifies key samples that are both class-representative and domain-representative to facilitate adaptation with minimal data, and (ii) Inference-only Batch-aware Memory Normalization (IoBMN), which leverages representative samples to adjust normalization layers on-the-fly during inference, aligning the model effectively to changing domains. When combined with five state-of-the-art TTA algorithms, SNAP-TTA maintains the performances of these methods even with much-reduced adaptation rates from 0.01 to 0.5, making it suitable for edge devices serving latency-sensitive applications.

## 1 Introduction

Deep learning models often suffer from performance degradation under domain shifts caused by environmental changes or noise (Quinonero-Candela et al., 2008). Test-Time Adaptation (TTA) ˜ offers a promising solution for domain shifts by utilizing only unlabeled test data without requiring source data. While TTA algorithms have advanced in complexity to improve accuracy in data streams (Wang et al., 2021; Niu et al., 2022; Wang et al., 2022; Yuan et al., 2023; Niu et al., 2023; Song et al., 2023), they are typically designed for resource-rich servers, overlooking the computational and memory limitations crucial for real-world deployment. Operations such as backpropagation, data augmentation, and model ensembling (Wang et al., 2022; Yuan et al., 2023; Zhang et al., 2022) result in substantial latency and memory consumption, making state-of-the-art (SOTA) TTA methods inefficient for practical use (Section 2). For edge devices with limited computational power, such as mobile devices or IoT sensors, the adaptation latency from TTA methods becomes a critical bottleneck, particularly in latency-sensitive applications such as autonomous driving and real-time health monitoring. Moreover, the model must keep up with the data stream in those applications, but high computational overhead could cause it to miss critical samples, resulting in inference lags and reduced accuracy. This issue is exacerbated with fast data streams, such as high-frame-rate videos or high-performance sensors. For example, even a slight delay in processing sensor data can lead to dangerous situations in autonomous driving. A high adaptation latency that accumulates with each batch not only undermines real-time performance but also limits the potential of TTA algorithms in latency-sensitive applications. In online TTA scenarios that require rapid response to incoming data streams on resourceconstrained devices, *Sparse TTA (STTA)*, which adapts occasionally rather than at every batch, can offer a practical solution by reducing the adaption overhead. However, na¨ıve STTA may result in performance degradation as it utilizes far less data (e.g., 0.1) for model adaptation (Figure 1). The

# Snap-Tta: Sparse Test-Time Adaptation For Latency-Sensitive Applications

1

Online data stream t Original TTA
Delay Adapt batch Timeline Sparse TTA
Latency: 6.7s Acc: 81%
TTA on Edge devices Pre-trained Model Adapt batch Timeline Latency: 2.1s Acc: 69%
: Adaptation : Inference
*Adaptation Rate: 0.33
effectiveness of STTA hinges on selecting proper samples from a large pool, ensuring that the model maintains adequate performance with fewer updates (detailed analysis in Section 4). Conventional TTA approaches that adopt sampling strategies are designed for non-i.i.d data (Gong et al., 2022; Niu et al., 2023; Yuan et al., 2023) or noisy data (Gong et al., 2023). They do not aim for data efficiency and thus yield high sample usage for updates. While EATA (Niu et al., 2022) excludes unreliable samples and utilizes fewer samples, it suffers from performance degradation when attempting more aggressive reductions. Data-efficient deep learning demonstrated that selecting easy, class-representative samples is effective when the sampling ratio is low (e.g., below 0.4) (Xia et al., 2022; Choi et al., 2024). However, these methods rely on ground-truth label information, which is typically unavailable in TTA scenarios. We propose **SNAP-TTA**: Sparse Network Adaptation for Practical Test-Time Adaptation, a lowlatency TTA framework designed for resource-constrained devices. SNAP-TTA addresses the challenge of balancing adaptation accuracy with computational efficiency in STTA, where only a small subset of data is used for updates. To that end, SNAP-TTA has two key technical enablers: First, it introduces a sampling strategy that combines *class-representative* and *domain-representative* samples. This approach enables the model to adapt effectively to domain shifts even with minimal data. Class and Domain Representative Memory (CnDRM) selects these critical samples by using pseudo-label confidence in a prediction-balanced manner for class-representative samples, and by identifying the domain-representative samples closest to the center of the target domain's feature embedding (Section 3.1). Second, Inference-only Batch-aware Memory Normalization (IoBMN) refines the normalization process during inference by utilizing CnDRM's class-domain representative statistics, leveraging the representativeness of these selected samples to correct skewed feature distributions at each inference step. This ensures that the model effectively adapts to domain shifts without back-propagation, maintaining alignment with the evolving data distribution (Section 3.2). These two components are integrated to perform adaptation, minimizing accuracy drop and latency in real-world domain-shifted scenarios. SNAP-TTA is designed to work together with existing TTA methods orthogonally; thus, we evaluated SNAP-TTA integrated with existing SOTA TTA algorithms under diverse adaptation rates. Specifically, we evaluated SNAP-TTA with five SOTA TTA algorithms (Tent(Wang et al., 2021), EATA(Niu et al., 2022), SAR(Niu et al., 2023),CoTTA(Wang et al., 2022), and RoTTA(Yuan et al., 2023)) on three common TTA benchmarks (CIFAR10-C, CIFAR100-C (Hendrycks & Dietterich, 2019a), and ImageNet-C (Hendrycks & Dietterich, 2019b)). SNAP-TTA effectively reduces latency while minimizing performance drops in existing TTA methods. For instance, on our implementation in Raspberry Pi 4(Raspberry Pi Foundation, 2019) testbed, SNAP-TTA achieved up to 87.5%
latency reduction at an adaptation rate of 0.1. In CIFAR10-C, SNAP-TTA-integrated methods consistently outperformed their original counterparts, showing up to 13.38% accuracy gain for CoTTA at an adaptation rate of 0.01. In addition, SNAP-TTA integration performed comparable accuracy to the original TTA methods under full adaptation settings. For instance, it achieved 77.12%∼81.74% accuracy for Tent at various adaptation rates, whereas the full adaptation accuracy was 80.43% in CIFAR10-C.

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107

## 2 Preliminaries 108

109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 We focus on the Test-Time Adaptation (TTA) latency challenges specific to edge devices, highlighting the constraints of adapting models in real-time environments with limited resources. Detailed related works are in Appendix A. Test-Time Adaptation and Its Latency Challenge on Edge Devices. In unsupervised domain adaptation, the source domain data DS = X
S , Y is drawn from the distribution PS (x, y), while the target domain data DT = X
T, Y follows PT (x, y), typically without known labels yj . Given a pre-trained model f(·; Θ) on the source domain DS , test-time adaptation (TTA) (Wang et al., 2021) adjusts the model to the target distribution PT using only target instances xj , updating the parameters Θ to reduce domain discrepancy. When applied to resource-constrained devices, however, current TTA approaches face significant latency challenges. In real-time applications that require rapid inference, online TTA becomes impractical due to the need for adaptation at every batch (Figure 4, detailed latency tracking reported in Appendix E.3). Our experiment on Raspberry Pi 4 (Raspberry Pi Foundation, 2019) showed a minimum of 3.83 seconds latency per batch for existing TTA methods. This indicates existing methods could not handle real-time applications with fast data streams and strict latency requirements, such as autonomous driving (Tampuu et al., 2024; Liu et al., 2023). TTA methods such as CoTTA use computationally intensive operations such as data augmentations and ensemble models at the cost of increased latency. Relatively lightweight algorithms incur non-negligible latency from adaptation processes such as backpropagation, which becomes bottlenecks in resource-constrained devices without the parallel processing capabilities and memory bandwidth of GPUs. A recent work (Alfarra et al., 2024), recognizing latency as a problem, proposed a TTA evaluation protocol that penalizes methods that are slower than the data stream rate. Instead of penalizing a model for being slow, we utilize Sparse TTA, where the model actively chooses to adapt at sparse intervals for the goal of maintaining a real-time inference rate. As real deployments involve devices with different computational capabilities and data streams of varying speeds, we believe a framework that effectively maintains various TTA methods' performance across different latency requirements is crucial. SNAP-TTA framework resolves the high latency and inefficiency issue of existing Test-Time Adaptation (TTA) methods. By introducing a Sparse TTA (STTA) strategy combined with a novel sampling method, SNAP-TTA minimizes adaptation delays while maintaining accuracy. The overall system, illustrated in Figure 2, consists of two primary components: (i) Class and Domain Representative Memory (CnDRM) for efficient sampling and (ii) Inference-only Batch-aware Memory Normalization (IoBMN) to correct feature distribution shifts during inference. Together, these components enable effective STTA with minimal computational overhead.

## 3 Methodology

Sparse Test-Time Adaptation and Adaptation rates. Sparse Test-Time Adaptation (STTA) aims to efficiently adapt models by reducing both the frequency of updates and the number of samples used per update, which is essential for minimizing latency in edge devices. The concept of adaptation rate plays a central role in STTA, as it controls both the update frequency and the number of data points used. Unlike Original Test-Time Adaptation (TTA), which uses full batches of data and can create significant computational overhead, STTA employs an adaptation rate to limit updates and data usage proportionally, thus introducing sparsity (Figure 1).

By adjusting the *adaptation rate*, STTA can minimize latency and computational costs while maintaining adaptation performance. This rate defines how sparsely updates occur and the proportion of samples used for updates compared to the Original TTA, enabling efficient model adjustments to distribution shifts. The balance between adaptation accuracy and computational efficiency makes STTA particularly suitable for environments that demand both quick responses and minimal resource usage.

Online data stream t
① ② ③ ④ ⑤ ⑥ ⑦ ⑧
Adapt Samples Timeline Latency Acc
(b) Inference-only Batch aware Memory Normalization
(a) Sparse adaptation via Class-Domain Representative Memory High Confidence Low Confidence Utilize Memory Normalization (MN) stats
&
Correct via Inference-only BN (IoBN) stats ҧm, തm IBt IoBN
MN
IoBMN = MN + S(MN-IoBN) 
S: soft shrink 1
(μd, σd)
2 ҧ, ത 3 Domain Centroid Close sample Domain Centroid Far sample CnDRM

## 3.1 Class And Domain Representative Memory (Cndrm)

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 CnDRM is a core component of SNAP-TTA that addresses the challenges of efficient data sampling for STTA. In STTA, the adaptation rate directly impacts the number of samples used, necessitating a careful sampling strategy to optimize performance with minimal data. Given this limited sampling ratio, CnDRM selects both class and domain-representative samples to maintain model performance while minimizing adaptation overhead. Motivation. Data sampling is crucial in data-efficient deep learning, especially when working with a limited number of samples. In high data sampling ratio scenarios, score-based methods prioritize difficult or rare samples, often achieving performance comparable to full-dataset training.

However, when the sampling ratio is low, selecting easy and class-representative samples becomes more effective (Choi et al., 2024). This method selects samples that minimize differences in loss gradients or curvature, ensuring that the generalizability is retained even with fewer samples. Similarly, the Moderate Coreset (Xia et al., 2022) paper demonstrates that at low sampling ratios of 0.2 to 0.4, the distance from the class center significantly impacts performance, with samples closer to the center being particularly effective in scenarios with high label noise. In the STTA setting, where ground truth labels are unavailable and the probability of incorrect predictions is high, selecting representative samples based on potentially incorrect predictions resembles a high label noise situation. Therefore, selecting class-representative easy samples could provide some benefit to STTA. However, if the model must perform STTA at an even lower adaptation rate (e.g., 0.1) due to the latency limits, selecting class-representative samples alone would be insufficient (Table 4). Unlike traditional classification tasks, STTA is an unsupervised domain adaptation, which requires identifying target domain-representative samples that reflect the distributional shift between the source and target domains. In these cases, we argue that focusing on domain-representative instances is just as crucial, as selecting samples that best capture the domain shift can help the model retain generalizability with minimal data. Therefore, selecting both class-representative and domainrepresentative samples could enhance STTA performance in low-data environments, where each sample must contribute significantly to model adaptation. Critera 1: Class Representation. CnDRM selects samples with higher confidence scores to avoid the issues caused by low-confidence samples. Low-confidence samples are typically located near decision boundaries and are more likely to carry incorrect pseudo-labels. This strategy ensures that the adaptation process is guided by stable learning signals, which is important in the absence of ground-truth labels. By focusing on high-confidence samples, CnDRM mitigates the risk of propagating errors resulting from incorrect pseudo-labels, thereby supporting more effective and stable adaptation (Details in Appendix E.2). The confidence score C(x) for each sample x is calculated as: C(x) = maxy∈Y p(y|x; Θ) where p(y|x; Θ) is the softmax probability for class y. Only samples with confidence above a predefined threshold τ*conf* are retained. For a balanced representa216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

Close Far PCA 
Feat ure 2 Wasserstein distance (Instance Domain centroid)
PCA Feature 1 PCA Feature 1 Sampled Instance All test data (KDE) 
PCA 
Feat ure 2 Accuracy: 26.65 % **Accuracy: 18.52 %**

## Critera 2: Domain Representation.

In addition to class-representative sampling, CnDRM selects domainrepresentative samples to facilitate adaptation to new domain conditions. Building on the efficient classrepresentative sampling criteria, we argue that selecting samples close to the domain centroid would enhance performance in STTA. Our preliminary experiment results validate improved performance when selecting samples near the centroid (Figure 3). For ImageNet-C Gaussian noise, TTA with the closest 20% of samples achieved 26.65% accuracy, whereas the farthest 20% showed a lower accuracy of 18.52%. As early layers in deep learning models tend to retain domain-specific features (Zeiler & Fergus, 2014; Lee et al., 2018; Segu et al., 2023), we utilize the hidden features of early layers to identify domain-representative samples (Appendix E.1). We use the feature statistics (mean and variance) of the first normalization layer to evaluate domain representation. This choice is made as domain discrepancies can be effectively reduced through normalization adjustments (Nado et al., 2020; Schneider et al., 2020). Domain discrepancies in hidden features are substantially reduced after passing through a single normalization layer, significantly minimizing domain shift differences (Li et al., 2016). While deeper layers provide detailed information, using the first layer balances capturing domain-specific information and maintaining computational efficiency.

The domain centroid c*domain* is computed using a momentum-based update of batch statistics from the normalization layer: µdomain ← (1 − β)µ*domain* + βµt and σ 2*domain* ← (1 − β)σ 2*domain* +
βσ2 t, where µt and σ 2 tare the mean and variance of the current batch t, and β is the momentum parameter. In our preliminary study, we found that using only the mean and standard deviation values before the first normalization was sufficient to calculate the domain centroid. The sampled instances effectively represented the domain and were correctly positioned in the embedding space for each criterion (Figure 3). To determine domain-representative samples, CnDRM calculates the Wasserstein distance between each sample's feature statistics and the domain centroid. The Wasserstein distance measures the similarity between two distributions by considering their mean and variance, evaluating how well a sample represents the domain. It is useful for capturing domain characteristics, leading to its wide use in domain generalization (Segu et al., 2023). For each sample xt, the feature statistics (µxt
, σxt
)
are taken from the input to the normalization layer, and the Wasserstein distance W(xt, c*domain*) is given by:
Figure 3: Samping visualization and accuracy comparison between the closest 20% and farthest 20% samples from the domain centroid (based on Wasserstein distance) on ImageNet-C Gaussian noise.

RAND
KDE
66.08

$$W(\mathbf{x_{t}},\mathbf{c}_{d o m a i n})={\sqrt{(\mu_{\mathbf{x_{t}}}-\mu_{d o m a i n})^{2}+(\sigma_{\mathbf{x_{t}}}-\sigma_{d o m a i n})^{2}}}$$
2. (1)
tion across diverse classes, CnDRM selects these high-confidence samples in a prediction-balanced manner. This balance helps maintain the model's overall classification capability and prevents bias towards certain classes when only a low sample ratio is available for adaptation. By leveraging both high confidence and prediction balance, CnDRM effectively selects class-representative samples that are diverse and reliable, even without access to ground-truth labels.

Memory Management Algorithm. The memory management in CnDRM maintains efficiency without introducing additional overhead. To achieve this, the memory size is kept equal to the batch size for minimal resource usage. Within this fixed memory, samples are managed by balancing the number of samples per class based on predictions so that each class remains well-represented. For domain adaptation, samples in memory are periodically replaced with new samples that are closer to the domain centroid and meet the confidence threshold to retain only the most class-domain representative samples. Algorithm 1 has details.

$\left(\mathbb{I}\right)$. 
270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 3.2 Inference-Only Batch-Aware Memory Normalization (Iobmn)

Motivation. In Sparse Test-Time Adaptation (STTA) scenarios, models must adapt to domain shifts despite having limited opportunities for updates. In this setting, maintaining robust performance becomes challenging as the stored memory statistics, derived from representative adaptation batches, may not fully align with subsequent inference batches, especially when updates are skipped. This can lead to a potential mismatch between the stored statistics and the current data distribution. Traditional normalization methods, which solely rely on test batches' statistics, struggle to address these subtle shifts effectively. To tackle this issue, we introduce the Inference-only Batch-aware Memory Normalization (IoBMN) module, which leverages the robustness of class-domain representative statistics while dynamically adjusting for mismatches that arise in skipped batches. By primarily basing normalization on stable, representative memory statistics and selectively adapting with recent inference data, IoBMN efficiently corrects for distributional shifts, ensuring both robustness and adaptability in STTA conditions. This approach significantly enhances model stability in sparse adaptation scenarios, as shown in our ablation study in Section 4.

Approach. Given a feature map f ∈ R
B×C×L, where B is the batch size, C is the number of channels, and L is the number of spatial locations, the batch-wise statistics µ¯c and σ¯
2 c for the c-th channel are calculated as follows:

$$\bar{\mu}_{c}=\frac{1}{B\times L}\sum_{b=1}^{B}\sum_{l=1}^{L}f_{b,c,l},\quad\bar{\sigma}_{c}^{2}=\frac{1}{B\times L}\sum_{b=1}^{B}\sum_{l=1}^{L}(f_{b,c,l}-\mu_{b,c}),$$
$$\left(2\right)$$

$$({\mathfrak{I}})$$

$$(4)$$
(fb,c,l − µb,c), (2)
where µ¯m and σ¯
2m are calculated from the most recent adapted CnDRM samples in the same way with Equation 2, using the memory capacity M with m representing the memory. We assume that µm and σ 2m follow the *sampling distribution* of the feature map size L and memory capacity M.

The corresponding variances for the memory mean µm and variance σ 2m are calculated as:

$s_{\mu_m}^2:=\dfrac{\bar{\sigma}_m^2}{C\times M},\quad s_{\sigma_m^2}^2:=\dfrac{2\bar{\sigma}_m^4}{C\times M-1}.$  we to adapt efficiently to the current inference batch. 
. (3)
For the normalization process to adapt efficiently to the current inference batch statistics, IoBMN
corrects (¯µm, σ¯
2m) only when µ¯c (and σ¯
2 c
) significantly differ from µ¯m (and σ¯
2m) through soft shrinkage function:
µ
IoBMN
m = ¯µm + Sλ(¯µc − µ¯m; αsµm), (σ
IoBMN
m )
2 = ¯σ
2
$$\stackrel{\mathrm{s}}{{}_{n}}+S_{\lambda}(\bar{\sigma}_{c}^{2}-\bar{\sigma}_{m}^{2};\alpha s_{\sigma_{m}^{2}}),$$
), (4)
Require: test data stream xt, memory M with capacity N, confidence threshold τ*conf* , sample unit for memory s, adaptation rate 1/k 1: for batch b ∈ {1*, . . . , B*} do 2: Yˆb ← f(b; Θ)
3: for each sample xt in batch b do 4: yˆt ← Yˆb[t]
5: confidence ← C(xt; Θ)
6: ct(µxt
, σxt
) ← mean and variance of early hidden feature 7: wxt ← W(xt, c*domain*)
8: if confidence > τ*conf* **then** ▷ Class-representative samples 9: Add st(xt, yˆt, ct, wxt
) to M ▷ Add samples in prediction-balanced manner 10: if |M| > N **then** 11: L
∗ ← class with most samples in M
12: if yˆt ∈/ L
∗**then** ▷ Removes domain-centroid farthest sample 13: smax dist ← arg maxsi∈M∧yˆi∈L∗ wxi 14: **else**
15: smax dist ← arg maxsi∈M∧yˆi=ˆyt wxi 16: Remove smax dist from M 17: cdomain ← (1 − β)c*domain* + βct ▷ Update domain-centroid 18: Recalculate wsifor all siin M
19: if b mod k == 0 **then** ▷ Adaptation occurs every k batches 20: Update model Θ using samples in M

## Algorithm 1 Class And Domain Representative Memory (Cndrm)

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 where α ≥ 0 in IoBMN controls the reliance on the normalization layer statistics. A larger α gives more weight to the last adapted memory normalization statistics, whereas a smaller α emphasizes the current inference batch normalization statistics. The soft shrinkage function Sλ(x; λ) is defined as:

$$S_{\lambda}(x;\lambda)={\begin{cases}x-\lambda&{\mathrm{if~}}x>\lambda,\\ x+\lambda&{\mathrm{if~}}x<-\lambda,a n d\\ 0&{\mathrm{~otherwise,}}\end{cases}}$$
$$\mathbf{(5)}$$

where λ is the threshold, s is a scaling factor, and x is the input. The function allows for proportional adjustments based on the magnitude of the values, where smaller values are adjusted less and larger values more, preserving the critical information inherent in the adapted memory normalization statistics.

Finally, the output of the IoBMN for each feature f*b,c,l* is computed as:

$$\mathrm{IoBMN}(f_{b,c,l};\bar{\mu}_{m},\bar{\sigma}_{m}^{2},\mu_{m}^{\mathrm{loBMN}},(\sigma_{m}^{\mathrm{loBMN}})^{2}):=\gamma\cdot\frac{f_{b,c,l}-\mu_{m}^{\mathrm{loBMN}}}{\sqrt{(\sigma_{m}^{\mathrm{loBMN}})^{2}+\epsilon}}+\beta,$$
+ β, (5)
where γ and β are learnable affine parameters of normalization layer, and ϵ is a small constant added for numerical stability. In our experiments, we chose α = 4 to effectively handle various out-ofdistribution scenarios. The parameter s is a hyperparameter that determines the degree of adjustment desired and can be tuned based on specific requirements. IoBMN utilizes CnDRM's class-domain representative statistics and adjusts them based on the current inferencing batch statistics. This dual-statistic approach allows IoBMN to correct the outdated and skewed distribution of the memory, ensuring alignment with the data distribution at each inference point. By leveraging the statistics of the data used during model update points, IoBMN adapts effectively without significant computational overhead. Additionally, this method mitigates the performance degradation caused by the prolonged intervals between adaptations so that the model remains well-aligned with the evolving data distribution.

## 4 Experiments

This section outlines our experimental setup and presents the results obtained under various STTA settings. Refer to Appendix B for further details. Scenario. We examined how different adaptation rates affect performance to simulate a scenario requiring a certain latency threshold for latency-sensitive applications. We varied the adaptation rate to observe its impact on both model accuracy and latency. The main evaluation was run with diverse adaptation rates (0.01, 0.03, 0.05, 0.1, 0.3, and 0.5). We report the average accuracy and standard deviation from three random seeds. Latency measurement was done on our Raspberry Pi 4 (Raspberry Pi Foundation, 2019) testbed. Dataset and Model. We used three standard TTA benchmarks: CIFAR10-C, **CIFAR100-** C (Hendrycks & Dietterich, 2019a) and **ImageNet-C** (Hendrycks & Dietterich, 2019b). These datasets include 15 different types of corruption with five levels of severity, and we used the highest one. CIFAR10-C/CIFAR100-C has 10,000 test data with 10/100 classes, and ImageNet-C has 50,000 test data with 1,000 classes for each corruption. We employed **ResNet18** (He et al., 2016) as the backbone network, utilizing models pre-trained on CIFAR10 and CIFAR100 (Krizhevsky & Hinton, 2009). We also use **ResNet50** (He et al., 2016) and ViT (Dosovitskiy, 2020) pre-trained on ImageNet (Deng et al., 2009) from the TorchVision (maintainers & contributors, 2016) library.

Baselines. SNAP-TTA is designed to integrate with existing TTA algorithms. Therefore, testing existing *TTA algorithms under different adaptation rates* serves as our baseline (implementation details including hyperparameters are in Appendix B.1). We selected five SOTA TTA algorithms: (i) **Tent** (Wang et al., 2021) updates only BN affine parameters, (ii) **CoTTA** (Wang et al., 2022) updates the entire model parameters using a teacher-student framework, (iii) **EATA** (Niu et al., 2022), (iv) SAR(Niu et al., 2023), and (v) **RoTTA**(Yuan et al., 2023). For efficiency evaluation, we compared our method against **BN stats** (Nado et al., 2020; Schneider et al., 2020).

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

RoTTA 20.60 22.83 19.81 10.46 10.10 21.31 31.83 39.66 32.09 46.08 62.22 20.27 42.54 47.47 40.67 31.20 87.00 Tent 23.63 25.18 24.80 21.81 20.97 34.11 43.60 41.44 36.98 52.66 64.21 22.74 **48.96** 53.46 46.80 37.42 27.34

+ SNAP 26.60 28.21 27.94 24.37 22.39 36.45 44.36 42.64 38.54 52.91 64.26 **33.47** 48.58 53.90 47.41 **39.47** 28.84

CoTTA 11.74 12.74 12.68 11.77 11.62 22.64 34.97 31.05 29.81 44.24 62.12 13.73 40.31 45.19 36.71 28.09 205.22

+ SNAP 15.26 16.00 15.83 13.81 14.13 24.84 36.46 32.58 31.73 46.04 63.52 15.69 42.18 46.74 38.00 **30.19** 208.10

EATA 27.35 29.03 28.62 23.94 23.45 37.21 46.18 44.05 39.19 54.52 64.54 32.20 51.22 55.00 49.27 40.38 20.27

+ SNAP 29.48 31.20 30.69 26.68 25.90 38.24 46.60 44.62 39.31 54.82 64.44 32.87 51.41 55.41 49.78 **41.43** 22.16

SAR 28.12 29.30 29.63 22.37 23.88 39.34 45.36 45.69 36.73 54.91 64.11 10.96 52.22 55.76 49.60 39.20 36.44

+ SNAP 32.63 34.69 34.26 28.91 27.96 43.51 47.79 48.27 42.41 56.45 64.77 32.76 53.74 57.21 51.67 **43.80** 38.01

RoTTA 16.90 17.88 17.25 12.89 12.51 23.96 35.26 36.26 32.32 47.25 63.98 17.46 42.77 48.21 39.35 30.95 59.32

0.3

+ SNAP 18.63 19.94 19.35 14.88 14.34 25.88 36.47 37.13 33.32 47.74 63.96 19.08 42.98 48.73 40.27 **32.18** 60.31

Tent 22.00 23.51 23.07 19.38 18.86 32.15 42.29 39.70 34.33 51.62 63.70 15.79 47.74 52.35 45.54 35.47 18.01

+ SNAP 26.21 27.85 27.50 23.62 22.73 36.01 44.11 42.19 38.15 52.95 64.57 30.23 48.56 53.71 47.09 **39.03** 18.76

CoTTA 10.97 11.92 11.98 11.45 11.38 22.39 34.96 30.88 29.89 44.09 61.96 13.08 40.20 45.27 36.71 27.81 161.98

+ SNAP 15.13 16.03 15.91 13.86 14.02 24.90 36.51 32.56 31.81 46.02 63.60 15.69 41.94 46.78 38.03 **30.19** 163.24

EATA 22.43 23.78 23.26 19.38 19.42 32.18 43.22 40.65 36.64 52.38 63.87 24.59 48.13 52.89 46.33 36.61 16.00

+ SNAP 26.10 27.29 27.13 22.38 22.15 33.45 43.92 40.96 36.68 52.71 63.77 27.93 48.47 53.23 47.46 **38.24** 17.45

SAR 26.12 27.56 26.93 22.51 23.35 36.03 44.48 43.19 37.26 53.82 64.15 19.87 50.78 54.78 48.43 38.62 21.39

+ SNAP 30.28 31.97 31.30 26.67 26.31 39.66 46.08 45.43 40.26 54.76 64.62 36.12 51.26 55.42 49.63 **41.99** 23.99

RoTTA 14.77 15.59 15.33 13.17 13.19 23.85 35.38 32.73 30.77 45.22 63.08 15.62 41.05 46.15 37.19 29.54 45.98

0.1

+ SNAP 15.35 16.20 16.01 13.67 13.66 24.27 35.62 33.04 31.02 45.38 62.95 15.96 41.06 46.17 37.44 **29.85** 47.47

Tent 23.77 24.65 24.44 20.54 20.27 32.73 43.57 40.82 35.92 52.78 63.82 15.95 49.33 53.46 47.19 36.62 16.93

+ SNAP 29.12 30.46 30.30 25.77 25.22 38.21 46.14 44.29 39.95 54.65 65.47 33.81 50.83 55.59 49.21 **41.27** 17.55

CoTTA 11.03 11.91 11.75 11.03 11.20 22.30 34.98 30.87 29.78 43.99 61.87 12.92 40.26 45.23 36.63 27.72 152.94

+ SNAP 15.22 15.97 15.93 13.91 14.05 24.87 36.48 32.60 31.65 46.09 63.59 15.67 42.00 46.71 37.96 **30.18** 153.34

EATA 19.53 20.65 20.72 16.74 16.96 29.11 41.22 37.96 34.84 50.75 63.29 19.86 45.92 51.15 44.13 34.19 15.82

+ SNAP 22.83 23.95 23.62 19.43 19.70 30.34 41.59 38.06 35.06 50.98 63.30 23.72 46.26 51.52 45.46 **35.72** 16.44

SAR 23.25 24.23 23.66 19.98 20.38 33.05 43.04 40.73 36.06 52.61 64.09 20.17 49.00 53.35 46.73 36.69 19.98

+ SNAP 27.54 29.03 28.66 24.05 23.42 36.28 44.12 42.89 38.54 53.24 64.25 31.83 48.79 54.04 47.80 **39.63** 20.94

RoTTA 14.42 15.22 15.02 13.25 13.31 23.79 35.27 32.09 30.43 44.71 **62.64** 15.24 40.63 45.55 36.75 29.22 43.32

0.05

+ SNAP 14.65 15.48 15.29 13.43 13.45 23.93 35.33 32.18 30.53 **44.71** 62.58 15.41 40.64 45.55 36.81 **29.33** 44.71

Table 1: STTA classification accuracy (%) and latency per batch (s) comparing with and without SNAP-TTA on ImageNet-C through Adaptation Rates (AR) (0.3, 0.1, and 0.05).AR is the ratio of the number of backpropagation occurrences to the total, and thus represents the reduction in adaptation latency compared to full adaptation (AR=1). More results on diverse AR (0.5, 0.03 and 0.01) are on Appendix C.1. **Bold** numbers are the highest accuracy.

AR Methods Gau. Shot Imp. Def. Gla. Mot. Zoom Snow Fro. Fog Brit. Cont. Elas. Pix. JPEG Avg. Lat.

Source 3.00 3.70 2.64 17.90 9.74 14.72 22.45 16.60 23.06 24.00 59.11 5.37 16.50 20.88 32.63 18.15 16.60 BN stats 14.29 15.06 14.89 13.30 13.38 23.78 35.22 31.78 30.26 44.40 62.39 15.14 40.42 45.25 36.53 29.00 17.36 Tent 27.03 28.98 28.64 24.66 23.63 38.70 45.77 44.82 38.06 54.59 64.61 16.84 51.64 55.54 49.38 39.53 38.33 CoTTA 13.12 13.98 13.94 12.44 12.18 23.74 35.22 31.78 30.26 44.40 62.40 15.13 40.42 45.26 36.53 28.72 300.23 EATA 29.62 31.79 31.17 26.89 26.30 40.65 47.44 46.29 40.78 55.57 64.97 38.02 52.66 56.03 50.26 42.56 31.98 SAR 17.49 22.04 21.21 11.62 12.60 39.76 44.13 45.98 29.39 55.13 63.71 17.34 52.31 56.09 49.35 35.21 78.15 1

Table 2: STTA classification accuracy (%) and latency per batch (s) comparing with and without SNAP-TTA on CIFAR10/100-C at Adaptation Rate 0.1. Numbers in parentheses represent the performance difference of SNAP-TTA compared to full adaptation **Bold** numbers are the highest accuracy. More results on other adaptation rates are in Appendix C.2 and C.3.

Methods Gau. Shot Imp. Def. Gla. Mot. Zoom Snow Fro. Fog Brit. Cont. Elas. Pix. JPEG Avg. Lat.

CIFAR10-C

Tent 67.32 69.39 60.69 85.34 63.82 83.52 84.70 79.68 77.79 83.75 88.53 83.12 75.18 77.82 71.47 76.81 (-3.62) 2.80 (-29.47%)

+ SNAP 70.22 71.48 63.08 87.35 65.74 85.89 86.38 81.93 80.00 85.62 90.34 87.47 76.44 79.63 72.72 **78.95 (1.48)** 3.08 (-22.42%)

CoTTA 59.11 60.26 56.07 72.23 56.77 73.55 72.20 68.05 66.68 72.88 77.66 65.95 65.67 64.12 65.16 66.42 (-11.58) 4.92 (-93.14%)

+ SNAP 71.70 73.54 66.70 85.16 66.83 84.30 84.88 81.02 80.61 84.20 89.84 81.71 76.60 79.66 75.71 **78.83 (+0.83)** 4.93 (-93.12%)

EATA 66.65 68.96 59.73 84.93 63.26 83.10 84.53 79.28 77.46 83.48 88.12 82.46 74.49 77.48 70.43 76.29 (-5.27) 2.52 (-35.88%)

+ SNAP 69.29 70.49 61.71 87.32 65.48 85.96 86.64 81.44 79.56 85.47 90.50 86.84 76.32 79.64 72.51 **78.61 (-2.95)** 2.87 (-26.97%)

SAR 66.11 68.18 59.15 84.91 62.87 82.33 84.27 79.23 77.58 83.21 88.29 82.60 74.65 75.92 70.79 76.01 (-3.04) 2.85 (-50.43%)

+ SNAP 67.76 70.68 60.82 86.78 64.73 85.29 86.22 80.82 79.30 84.95 91.33 86.59 75.72 78.72 71.24 **78.06(-0.99)** 2.98 (-48.17%)

RoTTA 63.12 64.84 56.72 84.49 62.15 82.53 83.84 78.03 76.13 82.88 87.48 81.49 73.75 76.04 68.24 74.78 (-2.22) 2.91 (-50.93%)

+ SNAP 65.35 66.99 58.09 86.77 63.63 85.47 86.01 80.54 78.38 84.99 90.00 85.99 75.67 78.14 70.09 **77.07 (+0.07)** 2.94 (-50.42%)

CIFAR100-C

Tent 43.55 44.25 37.95 62.56 41.80 59.45 62.13 53.04 51.60 56.76 64.60 61.19 51.01 56.42 46.28 52.84 (-2.92) 3.34 (-27.49%)

+ SNAP 46.51 47.68 39.92 65.39 44.14 63.29 64.53 55.20 55.55 59.71 68.05 64.90 53.91 59.28 49.58 **55.84 (+0.08)** 3.67 (-19.17%)

CoTTA 28.53 29.53 26.45 42.19 30.34 44.69 41.88 34.44 33.93 39.03 45.49 31.17 37.25 36.17 36.84 35.86 (-13.53) 4.94 (-93.40%)

+ SNAP 41.72 42.62 37.46 58.43 41.24 57.33 57.96 50.34 51.17 52.29 63.59 51.32 49.68 54.78 47.89 **50.52 (+1.13)** 4.95 (-93.38%)

EATA 38.41 39.03 32.29 61.07 38.45 58.21 60.62 49.59 49.19 54.23 62.88 57.39 49.00 53.01 42.05 49.70 (-1.04) 3.13 (-27.17%)

+ SNAP 40.62 41.53 34.31 64.08 40.29 61.32 63.04 52.00 51.77 56.85 65.98 61.96 51.05 55.67 44.80 **52.35 (+1.61)** 3.51 (-17.50%)

SAR 43.92 45.28 38.64 63.36 42.58 60.36 62.78 53.39 52.23 57.54 65.41 60.88 52.07 56.80 47.16 53.49 (-4.45) 2.95 (-56.16%)

+ SNAP 46.29 47.60 39.95 65.26 44.00 63.09 64.97 55.08 55.17 59.73 68.13 64.72 53.84 58.98 49.54 **55.76 (-2.18)** 3.09 (-53.73%)

RoTTA 36.28 37.12 31.38 61.20 38.36 58.26 60.30 49.20 48.21 53.54 62.80 56.78 49.61 52.28 41.26 49.11 (-2.44) 2.96 (-55.92%)

+ SNAP 37.83 38.42 32.38 63.73 39.72 61.32 62.58 51.38 51.18 55.61 65.70 61.39 51.36 54.51 42.85 **51.33 (-0.22)** 2.99 (-55.41%)

Overall performance across various adaptation rates. Table 1, 2 and Appedix C summarize the performance comparison of baseline state-of-the-art (SOTA) TTA methods and SNAP-TTA integration across various adaptation rates (0.01 to 0.5) on CIFAR10/100-C and ImageNet-C. These results reveal that while Sparse TTA achieves a substantial reduction in adaptation latency up to 87.5% conventional SOTA algorithms suffer significant accuracy degradation under sparse adaptation settings (Table 3, Figure 4). In contrast, SNAP-TTA demonstrates a robust ability to mitigate this performance drop. Leveraging minimal updates with only a few samples, SNAP-TTA consistently outperforms baseline methods and shows competitive accuracy even when compared to fully adapted models. Furthermore, in certain scenarios, SNAP-TTA achieves accuracy gains over the 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

+ SNAP-TTA
Tent 55.76 55.84
+ SNAP-TTA
EATA
50.74 52.35
+ SNAP-TTA
SAR
57.94 55.76 51.55 51.33
+ SNAP-TTA
RoTTA
+ SNAP-TTA
CoTTA
49.39 50.70 Acc.

Figure 4: Latency and accuracy comparison of original TTA methods and their SNAP- TTA integration on CIFAR100-C. SNAP- TTA significantly enhances the efficiency.
Table 3: Latency reduction and accuracy gaps of SNAP-TTA (adaptation rate 0.1) compared by original TTA, tested on Raspberry Pi 4. Performance averaged over 15 CIFAR10-C corruptions. Numbers in parentheses represent the performance difference of SNAP-TTA compared to full adaptation.

Latency per batch (s) Accuracy (%) Methods Original TTA **SNAP-TTA** naive STTA **SNAP-TTA**
Tent 3.97 **2.20 (-44.0%)** 76.81 (-3.62) **78.95 (-1.48)** CoTTA 71.68 **8.96 (-87.5%)** 66.42 (-11.58) **78.83 (+0.83)** EATA 3.93 **2.18 (-44.6%)** 76.29 (-5.27) **78.61 (-2.95)** SAR 5.75 **2.30 (-60.1%)** 76.01 (-3.04) **78.06 (-0.99)**
RoTTA 5.93 **2.25 (-62.0%)** 74.78 (-2.27) **77.07 (+0.07)**
original counterparts, highlighting its adaptability and effectiveness. These results underscore the capability of SNAP-TTA to balance efficiency and performance, providing a significant advantage in sparse adaptation scenarios while maintaining or even enhancing classification accuracy. This validates the effectiveness of utilizing class-domain representative samples in the STTA setting. Furthermore, Figure 5 shows more computationally complex and latency-intensive methods such as CoTTA tend to have greater performance gain when integrated with SNAP-TTA. This is because methods that update the entire model parameters are more susceptible to the influence of specific adaptation samples, leading to significant performance drops under sparse update conditions, which SNAP-TTA's CnDRM and IoBMN effectively mitigate. In addition, adaptation rates of 0.5 or 0.3, which represent relatively high adaptation frequencies, sometimes can achieves even better performance with SNAP-TTA than the original TTA, despite in the STTA setting. This is likely because the sampling rate was not critically low but rather comparable to that of existing data-efficient methods such as EATA (Niu et al., 2022), allowing SNAP-TTA to achieve performance gains similar to various sampling-based TTA methods (Niu et al., 2022; 2023; Gong et al., 2022; 2023) using fewer yet effective samples. Overall, SNAP-TTA significantly reduced the average latency per batch while effectively maintaining accuracy, highlighting its benefits for resource-constrained environments. More details on all other adaptation rates are reported in Appendix C.

0.1 acc Full Naïve SNAP-TTA

Table 4: Classification accuracy (%) comparison of ablative settings on the STTA (adaptation rate 0.1). Performance averaged over 15 CIFAR10-C corruptions.

Methods Tent CoTTA EATA SAR RoTTA

Na¨ıve 76.81 66.42 76.29 76.01 74.78 Random 77.08 65.61 76.59 76.33 75.01 LowEntropy 75.66 63.19 74.89 74.41 72.60 CRM 77.77 65.71 77.18 74.36 75.27 CnDRM 77.46 77.69 77.17 76.85 75.64 CnDRM+EMA 78.02 72.19 77.05 76.84 76.18

CnDRM+IoBMN 78.95 78.83 78.61 78.06 **77.07**

Original SAR SAR with SNAP-TTA
60 65 70 75 80 60 65 70 75 80

| CIFAR10-C corruptions. Methods Tent CoTTA                 | EATA   | SAR   | RoTTA   |       |       |
|-----------------------------------------------------------|--------|-------|---------|-------|-------|
| Na¨ıve                                                    | 76.81  | 66.42 | 76.29   | 76.01 | 74.78 |
| Random                                                    | 77.08  | 65.61 | 76.59   | 76.33 | 75.01 |
| LowEntropy                                                | 75.66  | 63.19 | 74.89   | 74.41 | 72.60 |
| CRM                                                       | 77.77  | 65.71 | 77.18   | 74.36 | 75.27 |
| CnDRM                                                     | 77.46  | 77.69 | 77.17   | 76.85 | 75.64 |
| CnDRM+EMA                                                 | 78.02  | 72.19 | 77.05   | 76.84 | 76.18 |
| CnDRM+IoBMN 78.95                                         | 78.83  | 78.61 | 78.06   | 77.07 |       |
| latency AR 1 (full adaptation) 0.5 0.3 0.1 0.05 0.03 0.01 |        |       |         |       |       |

1 0.5 0.3 0.1 0.05 0.03 0.01 Accu racy 
(%)
1 0.5 0.3 0.1 0.05 0.03 0.01 Accu racy 
(%)
Adaptation Rate Adaptation Rate Original CoTTA CoTTA with SNAP-TTA
Accu racy 
(%)
Figure 5: Classification accuracy on CIFAR10-C with varying adaptation rates. SNAP-TTA consistently mitigates accuracy drop across all rates.

65.00 70.00 75.00 80.00 85.00 Contribution of individual components of SNAP-TTA. We conducted an ablative evaluation to understand the effects of the individual components of SNAP-TTA (Table 4; more results on diverse adaptation rates and datasets are on Appendix D). CRM denotes prediction-balanced sampling with a confidence threshold (same as the Class-Representative criteria of CnDRM), and CnDRM denotes both Class and Domain Representative sampling (the first component of SNAP-TTA). For inference, the default uses test batch normalization statistics, EMA uses the exponential moving average of the test batch, and IoBMN uses memory samples' statistics corrected to match that of the test batch (the second component of SNAP-TTA). Contrary to the hypothesis that low-entropy samples are beneficial for TTA (Niu et al., 2022; 2023), LowEntropy performed worse than Rand for STTA. This can be attributed to the limited updates of STTA, resulting in poor or longer convergence times due to low entropy minimization loss.

CRM, originally used for data-efficient supervised deep learning (Choi et al., 2024; Xia et al., 2022), performed better than Rand. However, as CRM on TTA inevitably relies on uncertain pseudo labels instead of the ground truth, its performance remains lower than utilizing domain representative features (CnDRM) (note that TTA is unsupervised domain adaptation rather than training from scratch (Xia et al., 2022)). The highest accuracy was achieved when inference was performed us486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

| Methods                                              | Gau.                                                  | Shot   | Imp.   | Def.   | Gla.   | Mot.   | Zoom   | Snow              | Fro.   | Fog   | Brit.       | Cont.       | Elas.   | Pix.   | JPEG   | Avg.   |
|------------------------------------------------------|-------------------------------------------------------|--------|--------|--------|--------|--------|--------|-------------------|--------|-------|-------------|-------------|---------|--------|--------|--------|
| Tent                                                 | 40.56                                                 | 41.30  | 41.69  | 35.76  | 31.81  | 42.01  | 38.02  | 44.33 53.53 20.69 | 72.41  | 30.42 | 45.87 51.95 | 56.11 43.10 |         |        |        |        |
| + SNAP-TTA 40.98 41.72 42.18 37.16 32.30 42.89 38.44 | 46.19 52.50 53.11 72.25 39.25 46.77 51.53 55.99 46.22 |        |        |        |        |        |        |                   |        |       |             |             |         |        |        |        |
| EATA                                                 | 20.12                                                 | 21.52  | 21.40  | 20.90  | 23.42  | 15.71  | 18.00  | 16.12             | 28.35  | 22.24 | 35.97       | 11.33       | 19.78   | 20.22  | 19.99  | 21.00  |
| + SNAP-TTA 40.74 43.22 43.11 40.63 44.59 51.58 50.63 | 54.77 58.32 61.50 73.91 33.85 60.19 63.35 63.01 52.23 |        |        |        |        |        |        |                   |        |       |             |             |         |        |        |        |
| SAR                                                  | 21.45                                                 | 23.02  | 23.17  | 23.67  | 24.64  | 15.98  | 14.62  | 7.70              | 31.49  | 8.94  | 41.33       | 6.82        | 17.35   | 22.39  | 22.49  | 20.34  |
| + SNAP-TTA 37.59 38.27 36.78 38.58 39.99 49.00 45.77 | 43.96 56.61 59.96 73.02 19.69 54.30 61.16 61.85 47.77 |        |        |        |        |        |        |                   |        |       |             |             |         |        |        |        |

ing IoBMN, which primarily utilizes memory statistics and only shifts slightly to the test batch on demand. These results collectively indicate that utilizing CnDRM and IoBMN of SNAP-TTA enhances performance in a low-latency STTA scenario. Validation of SNAP-TTA on Vision Transformer (ViT) based Model. To validate the effectiveness of SNAP-TTA on the Vision Transformer (ViT) (Dosovitskiy, 2020), we conducted experiments on ImageNet-C with adaptation rate of 0.1. Since ViT uses layer normalization (LN), we adjusted CnDRM and IoBMN to use LN from instances, demonstrating that the core concepts of selecting domain-representative samples and mitigating shift in normalization statistics can be applied effectively to a different normalization type (details in Appendix F.3). The results in Table 5 confirm consistent accuracy gains of SNAP-TTA with significant latency decrease, regardless of model and normalization types.

## 5 Discussion And Conclusion

Limitations and future work. Our work could be optimized for more realistic data streams, such as continuous domain adaptation scenarios (Appendix F.2). For instance, the adaptation rate can be dynamically altered based on the need for adaptation (i.e., the data distribution just changed). Additionally, while SNAP-TTA employed a fixed confidence threshold in CnDRM as a safeguard to filter noisy samples, its adaptability could be improved. Dynamically adjusting the threshold based on data characteristics presents a promising direction for future research to enhance sampling efficiency and overall performance. Moreover, while we focused on reducing adaptation latency, memory overhead is another concern. We note that SNAP-TTA introduces negligible additional memory overhead, as detailed in the Appendix E.4, where related analysis and tracking information from real-device experiments are provided. Additionally, we demonstrate in the Appendix E.5 that SNAP-TTA can be effectively used alongside memory-efficient TTA methods such as MECTA (Hong et al., 2023), showcasing its compatibility and practicality. Future works could further explore optimizing SNAP-TTA for both latency and memory. Conclusion We raised the overlooked issue of latency of TTA methods, which is particularly relevant for applications on resource-constrained edge devices. To this end, we propose SNAP-TTA, a Sparse TTA (STTA) framework that could be applied to existing TTA methods to significantly reduce their latency while maintaining competitive accuracy. For effective performance in an STTA setting, we utilize class-domain representative memory of samples for adaptation. Furthermore, we optimize inference by adapting normalization layers using representative samples to account for domain shifts. Extensive experiments and ablative studies demonstrate SNAP-TTA's effectiveness in latency and adaptation accuracy.

## Reproducibility Statement

Details of the experiments, including datasets, scenarios, and hyperparameters for reproducibility, are provided in the Appendix B. Additionally, we share the link (https://anonymous.4open.science/r/SNAPTTA-DD0E) of an anonymous repository containing our source code and instructions to validate the reproducibility.

## References

Motasem Alfarra, Hani Itani, Alejandro Pardo, Shyma Yaser Alhuwaider, Merey Ramazanova, Juan Camilo Perez, Zhipeng Cai, Matthias Muller, and Bernard Ghanem. Evaluation of test- ¨ time adaptation under computational time constraints. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 976–991. PMLR, 21–27 Jul 2024. URL https: //proceedings.mlr.press/v235/alfarra24a.html.

Hoyong Choi, Nohyun Ki, and Hye Won Chung. Bws: Best window selection based on sample scores for data pruning across broad ranges. *arXiv preprint arXiv:2406.03057*, 2024.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition*, pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale.

arXiv preprint arXiv:2010.11929, 2020.

Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization for efficiently improving generalization. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=6Tm1mposlrM.

Taesik Gong, Jongheon Jeong, Taewon Kim, Yewon Kim, Jinwoo Shin, and Sung-Ju Lee. NOTE:
Robust continual test-time adaptation against temporal correlation. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=E9HNxrCFZPV.

Taesik Gong, Yewon Kim, Taeckyung Lee, Sorn Chottananurak, and Sung-Ju Lee. SoTTA: Robust test-time adaptation on noisy data streams. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In *International Conference on Learning Representations*, 2019a. URL https://openreview.net/forum?id=HJz6tiCqYm.

Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. *arXiv preprint arXiv:1903.12261*, 2019b.

Junyuan Hong, Lingjuan Lyu, Jiayu Zhou, and Michael Spranger. Mecta: Memory-economic continual test-time model adaptation. In *International Conference on Learning Representations*, 2023. URL https://openreview.net/pdf?id=N92hjSf5NNh.

Ziheng Jiang, Chiyuan Zhang, Kunal Talwar, and Michael C Mozer. Characterizing structural regularities of labeled data in overparameterized models. In Marina Meila and Tong Zhang (eds.), *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of Proceedings of Machine Learning Research, pp. 5034–5044. PMLR, 18–24 Jul 2021. URL
https://proceedings.mlr.press/v139/jiang21k.html.

Diederick P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In Doina Precup and Yee Whye Teh (eds.), Proceedings of the 34th International Conference on Machine Learning, volume 70 of *Proceedings of Machine Learning Research*, pp. 1885–1894. PMLR, 06–11 Aug 2017. URL https://proceedings.mlr.press/v70/koh17a. html.

A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. Master's thesis, Department of Computer Science, University of Toronto, 2009.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. *Advances in neural information processing* systems, 31, 2018.

Yanghao Li, Naiyan Wang, Jianping Shi, Jiaying Liu, and Xiaodi Hou. Revisiting batch normalization for practical domain adaptation, 2016. URL https://arxiv.org/abs/1603.04779.

Ji Lin, Wei-Ming Chen, Yujun Lin, john cohn, Chuang Gan, and Song Han. Mcunet: Tiny deep learning on iot devices. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 11711–11722. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper_files/ paper/2020/file/86c51678350f656dcc7f490a43946ee5-Paper.pdf.

Haolan Liu, Zixuan Wang, and Jishen Zhao. Cola: Characterizing and optimizing the tail latency for safe level-4 autonomous vehicle systems. *arXiv preprint arXiv:2305.07147*, 2023.

Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations (ICLR), 2017.

TorchVision maintainers and contributors. Torchvision: Pytorch's computer vision library. https:
//github.com/pytorch/vision, 2016.

Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. Coresets for data-efficient training of machine learning models. In Hal Daume III and Aarti Singh (eds.), ´ *Proceedings of the 37th* International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 6950–6960. PMLR, 13–18 Jul 2020. URL https://proceedings.mlr. press/v119/mirzasoleiman20a.html.

Zachary Nado, Shreyas Padhy, D Sculley, Alexander D'Amour, Balaji Lakshminarayanan, and Jasper Snoek. Evaluating prediction-time batch normalization for robustness under covariate shift. *arXiv preprint arXiv:2006.10963*, 2020.

Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, and Mingkui Tan. Efficient test-time model adaptation without forgetting. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), *Proceedings of the 39th* International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 16888–16905. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr. press/v162/niu22a.html.

Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, and Mingkui Tan. Towards stable test-time adaptation in dynamic wild world. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum? id=g2YraF75Tj.

NVIDIA Corporation. *NVIDIA Jetson Nano*, 2019. URL https://developer.nvidia.

com/embedded/jetson-nano. Accessed: 2024-11-20.

Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. Deep learning on a data diet:
Finding important examples early in training. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.), *Advances in Neural Information Processing Systems*, 2021. URL https://openreview.net/forum?id=Uj7pF-D-YvT.

Geoff Pleiss, Tianyi Zhang, Ethan Elenberg, and Kilian Q Weinberger. Identifying mislabeled data using the area under the margin ranking. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 17044–17056. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/ file/c6102b3727b2a7d8b1bb6981147081ef-Paper.pdf.

Omead Pooladzandi, David Davini, and Baharan Mirzasoleiman. Adaptive second order coresets for data-efficient machine learning. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of *Proceedings of Machine Learning Research*, pp. 17848–17869. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/ v162/pooladzandi22a.html.

Joaquin Quinonero-Candela, Masashi Sugiyama, Anton Schwaighofer, and Neil D Lawrence. ˜
Dataset shift in machine learning. Mit Press, 2008.

Raspberry Pi Foundation. *Raspberry Pi 4 Model B*, 2019. URL https://www.raspberrypi.

com/products/raspberry-pi-4-model-b/. Accessed: 2024-11-20.

Raspberry Pi Foundation. *Raspberry Pi Zero 2 W*, 2021. URL https://www.raspberrypi.

com/products/raspberry-pi-zero-2-w/. Accessed: 2024-11-20.

Steffen Schneider, Evgenia Rusak, Luisa Eck, Oliver Bringmann, Wieland Brendel, and Matthias Bethge. Improving robustness against common corruptions by covariate shift adaptation. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 11539–11551. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/ 85690f81aadc1749175c187784afc9ee-Paper.pdf.

648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 Mattia Segu, Alessio Tonioni, and Federico Tombari. Batch normalization embeddings for deep domain generalization. *Pattern Recognition*, 135:109115, 2023.

Junha Song, Jungsoo Lee, In So Kweon, and Sungha Choi. Ecotta: Memory-efficient continual test-time adaptation via self-distilled regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11920–11929, 2023.

Ardi Tampuu, Kristjan Roosild, and Ilmar Uduste. The effects of speed and delays on test-time performance of end-to-end self-driving. *Sensors*, 24(6):1963, 2024.

Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J. Gordon. An empirical study of example forgetting during deep neural network learning. In *International Conference on Learning Representations*, 2019. URL https:// openreview.net/forum?id=BJlxm30cKm.

Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully test-time adaptation by entropy minimization. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=uXl3bZLkr3c.

Qin Wang, Olga Fink, Luc Van Gool, and Dengxin Dai. Continual test-time domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7201–7211, June 2022.

Xiaobo Xia, Jiale Liu, Jun Yu, Xu Shen, Bo Han, and Tongliang Liu. Moderate coreset: A universal method of data selection for real-world data-efficient deep learning. In *The Eleventh International* Conference on Learning Representations, 2022.

Shuo Yang, Zeke Xie, Hanyu Peng, Min Xu, Mingming Sun, and Ping Li. Dataset pruning: Reducing training data by examining generalization influence. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=
4wZiAXD29TQ.

Longhui Yuan, Binhui Xie, and Shuang Li. Robust test-time adaptation in dynamic scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15922–15932, June 2023.

Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I 13, pp. 818–833. Springer, 2014.

Marvin Zhang, Sergey Levine, and Chelsea Finn. Memo: Test time robustness via adaptation and augmentation. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), *Advances in Neural Information Processing* Systems, volume 35, pp. 38629–38642. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/ fc28053a08f59fccb48b11f2e31e81c7-Paper-Conference.pdf.

702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809

## A Related Work

Test-time adaptation. Test-time adaptation (TTA) aims to improve model performance on Out-of- Distribution (OOD) data by using only the unlabeled test data stream to adapt the model. Test-time normalization (Nado et al., 2020; Schneider et al., 2020) adjusts the batch normalization (BN) statistics using test data to improve performance. Other works mainly involve updating the parameters of the model during test-time. Tent (Wang et al., 2021) adapts the affine parameters of the BN layers to minimize the entropy of its predictions. EATA (Niu et al., 2022) builds upon Tent, sampling reliable and non-redundant samples and utilizing an anti-forgetting regularizer for efficiency. Other works introduce more complex schemes, primarily to improve robustness against more practical test-time scenarios. CoTTA (Wang et al., 2022) addresses a continually changing test-time environment by using weight-averaged and augmentation-averaged predictions with stochastic restoring. SAR (Niu et al., 2023) filters samples with large and noisy gradients to stabilize the model during wilder test-time scenarios. RoTTA (Yuan et al., 2023) targets a practical test-time setting of changing distributions and correlative sampling by introducing a memory bank and a teacher-student model. Test-time adaptation on edge devices. TTA on edge devices primarily inherit the challenges of on-device learning: limited memory and increased latency from general resource constraints (Lin et al., 2020). Several memory-efficient TTA works have been proposed in this regard. MECTA (Hong et al., 2023) aims to reduce the memory consumption of gradient-based TTA, proposing an adaptive normalization layer to reduce the intermediate caches for backpropagation. Another work EcoTTA (Song et al., 2023) proposes memory-efficient continual TTA by adapting lightweight meta networks instead of the originals to reduce the size of intermediate activations. Despite works to promote memory-efficiency, the latency of TTA, especially on resource-constrained edge devices, has been generally overlooked. While many adaptation-based TTA (Wang et al., 2021; Niu et al., 2022; 2023; Yuan et al., 2023) update only the affine parameters for general time and memory concerns, they still involve computationally-heavy operations every batch, which can lead to high latency on edge devices. A recent work (Alfarra et al., 2024) introduces a more realistic TTA evaluation protocol that penalizes slow TTA methods by providing them with fewer samples for adaptation. We build on from this notion, proposing a sparse TTA setting to reduce the latency of existing TTA methods, but at a minimal cost to performance.

## B Experiment Details

All experiments presented in this paper were conducted using three random seeds (0, 1, 2), and we report the average accuracies along with their corresponding standard deviations. To ensure efficiency in experimentation, accuracy measurements were obtained using NVIDIA GeForce RTX 3090 GPUs, as the performance differences attributable to the random seed are negligible. Latency measurements were conducted on a Raspberry Pi 4 (Raspberry Pi Foundation, 2019), equipped with a Quad-core Cortex-A72 (ARM v8) 64-bit SoC @ 1.8GHz CPU and 4GB RAM. Data-efficient deep learning. Data-efficient deep learning methods enable deep learning models to achieve competitive performance with less data. Among these methods, data selection, or data sampling, involves utilizing a small subset of the training data in an attempt to match that of fulldataset training. A branch of data-selection is score-based selection, which scores each sample based on some predefined metric, such as a sample's influence (Koh & Liang, 2017), difficulty (Toneva et al., 2019; Paul et al., 2021), prediction confidence (Pleiss et al., 2020), or consistency (Jiang et al., 2021), and selects samples with scores in a certain range. Another set of data-selection methods involve optimization-based selection, which formulates an optimization problem to find a optimal subset that can best approximate full-dataset training (Mirzasoleiman et al., 2020; Yang et al., 2023; Pooladzandi et al., 2022). While these approaches work well in their preconceived settings, they generally suffer performance drop as their settings change, such as a change in sampling ratio. More recent works like the Moderate Coreset (Xia et al., 2022) proposes a more robust selection approach by using the distance of a sample to the class center as a score criterion, for an effective representation of the dataset. While our proposed sparse TTA setting is more challenging than the conventional data-efficient setting, as we cannot access ground truths labels nor make assumptions regarding the model, we utilize similar ideas of representative sampling as motivation for our method.

## B.1 Baseline Implementation Details

810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 In this study, we utilized the official implementations of the baseline methods. To ensure consistency, we adopted the reported best hyperparameters documented in the respective papers or source code repositories as much as possible. Also, we present information about the implementation specifics of the baseline methods and provide a comprehensive overview of our experimental setup, including detailed descriptions of the employed hyperparameters. We adopt hyperparameters from the original papers or the official code of the baselines for consistency. To assess the generality of SNAP-TTA, the test batch sizes were set to 16 for all baseline methods to ensure a fair comparison. To minimize overhead and maintain consistency with inference batches, we set the size of CnDRM equal to the batch size. TTA is conducted in an online manner, with adaptation or inference performed per batch. When there was a conflict between the implementation of SNAP-TTA and certain components of the existing baseline methods, we prioritized SNAP-TTA's features for fair evaluation at the STTA setting. For Tent (Wang et al., 2021), we update the BN affine parameters using the SGD optimizer (Loshchilov & Hutter, 2017) with a learning rate of l = 1e − 3 for CIFAR10/100C and l = 1e − 4 for ImageNet-C. For separate experimentation on the ViT, we used a learning rate of l = 2e − 4. For CoTTA (Wang et al., 2022), we update all model parameters using the Adam optimizer (Kingma & Ba, 2015) with a learning rate of l = 1e − 4. Furthermore, we set CoTTA's teacher model EMA factor to α = 0.99, the restoration factor to p = 0.1, and the anchor probability to pth = 0.9. For EATA (Niu et al., 2022), we use the SGD optimizer with a learning rate of l = 1e − 4. We set the entropy threshold as E0 = 0.4 × ln |N|, where N is the total number of classes. For SAR (Niu et al., 2023), we use SAM (Foret et al., 2021) with the base optimizer as SGD with a learning rate of l = 1e − 3. For fair evaluation, we replaced the sample filtering scheme with SNAP-TTA's CnDRM. For RoTTA (Yuan et al., 2023), we use the SGD optimizer with a learning rate of l = 1e−3. For fair evaluation, we replaced RoTTA's RBN and CSTU with SNAP-TTA's Cn-
DRM and IoBMN. For the teacher-student structure, we set the teacher model's exponential moving average update rate as v = 1e − 3.

Finally, we list the hyperparameters specific to the components of SNAP-TTA. The confidence threshold for CnDRM τ*conf* is set to 0.4 for CIFAR10-C, 0.45 for CIFAR100-C, and 0.5 for ImageNet-C. The entropy threshold for our ablation study τ*entr* is set to log(10)×0.40 for CIFAR10-
C and log(100) × 0.40 for CIFAR100-C, as referenced in a previous work using entropy-based filtering (Niu et al., 2022). Additionally, the parameters for the soft shrinkage function in IoBMN are fixed with α = 4 for Tent, CoTTA, SAR, RoTTA, and α = 2 for EATA.

## C Detailed Experiment Results

In this section, we provide detailed experimental results for the performance comparison of SNAP- TTA across a wide range of adaptation rates. We evaluated the performance on CIFAR10-C, CIFAR100-C, and ImageNet-C datasets with adaptation rates of 0.01, 0.03, 0.05, 0.1, 0.3, and 0.5, and across five state-of-the-art (SOTA) TTA algorithms: Tent, EATA, SAR, CoTTA, and RoTTA. This comprehensive evaluation resulted in a total of 150 combinations (3 datasets, 6 adaptation rates, 5 algorithms). The results demonstrate that, regardless of the adaptation rate, dataset, or the TTA algorithm, integrating SNAP-TTA consistently outperforms the baseline methods. Specifically, SNAP-TTA achieved the highest accuracy across nearly all of these 150 combinations, effectively demonstrating its robustness in both high and low adaptation settings. For CIFAR10-C and CIFAR100-C, SNAP-TTA showed substantial performance improvements compared to the baseline, even at very low adaptation rates (e.g., 0.01 and 0.05). Similarly, for ImageNet-C, SNAP-TTA maintained superior accuracy across diverse corruption types. These results highlight that SNAP-TTA effectively balances adaptation and latency, ensuring optimal performance even when the adaptation rate is sparse and regardless of the underlying TTA algorithm. This consistent superiority across all 150 combinations underscores SNAP-TTA's suitability for practical, real-world applications on resource-constrained devices. C.1 IMAGENET-C Table 6: STTA classification accuracy (%) comparing with and without SNAP-TTA on ImageNet-C through Adaptation Rates(AR) (0.5, 0.3, and 0.1), including results for full adaptation (AR=1). **Bold** numbers are the highest accuracy.

AR Methods **Gau. Shot Imp. Def. Gla. Mot. Zoom Snow Fro. Fog Brit. Cont. Elas. Pix. JPEG Avg.**

3.00 3.70 2.64 17.90 9.74 14.72 22.45 16.60 23.06 24.00 59.11 5.37 16.50 20.88 32.63 18.15 Source ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 14.29 15.06 14.89 13.30 13.38 23.78 35.22 31.78 30.26 44.40 62.39 15.14 40.42 45.25 36.53 29.07 BN stats ±0.05 ±0.02 ±0.08 ±0.08 ±0.08 ±0.05 ±0.06 ±0.04 ±0.07 ±0.14 ±0.11 ±0.05 ±0.10 ±0.04 ±0.16 ±0.07 27.03 28.98 28.64 24.66 23.63 38.70 45.77 44.82 38.06 54.59 64.61 16.84 51.64 55.54 49.38 39.53 Tent ±0.05 ±0.08 ±0.29 ±0.27 ±0.25 ±0.10 ±0.12 ±0.08 ±0.35 ±0.08 ±0.10 ±1.51 ±0.10 ±0.15 ±0.07 ±0.24 13.12 13.98 13.94 12.44 12.18 23.74 35.22 31.78 30.26 44.40 62.40 15.13 40.42 45.26 36.53 28.72 CoTTA ±0.08 ±0.07 ±0.01 ±0.10 ±0.04 ±0.04 ±0.06 ±0.05 ±0.06 ±0.14 ±0.11 ±0.03 ±0.10 ±0.04 ±0.16 ±0.07 29.62 31.79 31.17 26.89 26.30 40.65 47.44 46.29 40.78 55.57 64.97 38.02 52.66 56.03 50.26 42.56 EATA ±0.02 ±0.09 ±0.19 ±0.03 ±0.15 ±0.12 ±0.06 ±0.09 ±0.05 ±0.08 ±0.08 ±0.08 ±0.20 ±0.04 ±0.16 ±0.10 17.49 22.04 21.21 11.62 12.60 39.76 44.13 45.98 29.39 55.13 63.71 17.34 52.31 56.09 49.35 35.21 SAR ±0.40 ±1.44 ±0.96 ±0.72 ±0.97 ±0.63 ±0.11 ±0.23 ±0.30 ±0.20 ±0.08 ±0.61 ±0.08 ±0.18 ±0.13 ±0.47

20.60 22.83 19.81 10.46 10.10 21.31 31.83 39.66 32.09 46.08 62.22 20.27 42.54 47.47 40.67 31.20

RoTTA ±0.07 ±0.09 ±0.24 ±0.04 ±0.26 ±0.27 ±0.23 ±0.18 ±0.18 ±0.23 ±0.27 ±0.49 ±0.29 ±0.23 ±0.10 ±0.21

25.24 26.86 26.35 23.26 22.41 35.99 44.60 42.96 37.68 53.60 64.40 21.35 50.23 54.32 47.93 38.48 Tent ±0.10 ±0.27 ±0.08 ±0.06 ±0.05 ±0.09 ±0.10 ±0.13 ±0.17 ±0.15 ±0.12 ±0.94 ±0.12 ±0.15 ±0.04 ±0.17

28.05 29.97 29.39 25.73 23.39 38.49 45.65 44.21 39.57 53.90 64.52 34.39 49.99 54.88 48.72 40.72 + SNAP-TTA ±0.00 ±0.04 ±0.19 ±0.15 ±0.06 ±0.17 ±0.03 ±0.09 ±0.10 ±0.10 ±0.09 ±1.83 ±0.14 ±0.07 ±0.09 **±0.21**

11.99 13.04 12.86 11.90 11.64 22.92 35.06 31.20 29.97 44.28 62.16 14.02 40.39 45.29 36.58 28.22 CoTTA ±0.13 ±0.20 ±0.10 ±0.07 ±0.07 ±0.02 ±0.06 ±0.09 ±0.06 ±0.07 ±0.07 ±0.09 ±0.05 ±0.09 ±0.12 ±0.09

15.16 15.96 15.86 13.98 14.13 24.69 36.51 32.59 31.71 45.98 63.62 15.72 42.05 46.71 37.93 30.17 + SNAP-TTA ±0.14 ±0.02 ±0.14 ±0.04 ±0.00 ±0.09 ±0.07 ±0.16 ±0.06 ±0.09 ±0.05 ±0.04 ±0.09 ±0.24 ±0.14 **±0.09**

28.62 30.12 29.94 25.34 24.48 38.94 46.85 45.20 **40.03** 55.04 64.84 34.48 52.06 55.57 49.85 41.42 EATA ±0.10 ±0.10 ±0.14 ±0.20 ±0.44 ±0.10 ±0.25 ±0.12 **±0.01** ±0.06 ±0.07 ±0.41 ±0.24 ±0.13 ±0.05 ±0.16

30.00 31.88 31.47 26.93 26.64 39.16 47.23 **45.36** 39.75 55.30 64.52 33.75 52.29 55.66 50.48 42.03 + SNAP-TTA ±0.29 ±0.17 ±0.13 ±0.21 ±0.28 ±0.15 ±0.07 **±0.13** ±0.14 ±0.14 ±0.10 ±0.07 ±0.09 ±0.18 ±0.08 **±0.15**

26.74 28.56 28.77 19.90 21.50 39.97 44.98 45.95 34.22 55.04 63.93 6.58 52.50 55.98 49.71 38.29 SAR ±0.25 ±1.75 ±0.13 ±0.21 ±0.38 ±0.10 ±0.12 ±0.17 ±0.80 ±0.05 ±0.03 ±0.64 ±0.10 ±0.19 ±0.09 ±0.33

31.58 33.22 33.77 26.47 26.26 44.01 47.94 48.77 42.51 56.96 64.86 28.31 54.23 57.55 51.90 43.22 + SNAP-TTA ±0.38 ±2.44 ±0.56 ±1.69 ±0.94 ±0.10 ±0.04 ±0.12 ±0.09 ±0.13 ±0.10 ±10.99 ±0.08 ±0.16 ±0.19 **±1.20**

18.17 19.59 18.49 12.32 11.79 23.56 34.62 37.84 32.91 47.86 63.94 18.68 43.21 48.54 40.20 31.45 RoTTA ±0.05 ±0.03 ±0.10 ±0.11 ±0.13 ±0.15 ±0.14 ±0.11 ±0.06 ±0.05 ±0.16 ±0.42 ±0.08 ±0.23 ±0.23 ±0.14

20.43 22.03 21.05 15.47 14.49 26.36 36.46 38.98 34.15 48.41 64.02 20.74 43.66 49.16 41.05 **33.10**

+ SNAP-TTA ±0.03 ±0.08 ±0.11 ±0.11 ±0.07 ±0.06 ±0.10 ±0.09 ±0.12 ±0.13 ±0.13 ±0.23 ±0.10 ±0.10 ±0.15 **±0.11**

23.63 25.18 24.80 21.81 20.97 34.11 43.60 41.44 36.98 52.66 64.21 22.74 48.96 53.46 46.80 37.42 Tent ±0.08 ±0.37 ±0.28 ±0.02 ±0.18 ±0.07 ±0.04 ±0.05 ±0.04 ±0.15 ±0.13 ±0.04 ±0.16 ±0.07 ±0.09 ±0.12

26.60 28.21 27.94 24.37 22.39 36.45 44.36 42.64 38.54 52.91 64.26 33.47 48.58 53.90 47.41 39.47 + SNAP-TTA ±0.20 ±0.19 ±0.33 ±0.36 ±0.12 ±0.07 ±0.13 ±0.07 ±0.15 ±0.06 ±0.10 ±0.44 ±0.10 ±0.14 ±0.11 **±0.17**

11.74 12.74 12.68 11.77 11.62 22.64 34.97 31.05 29.81 44.24 62.12 13.73 40.31 45.19 36.71 28.09 CoTTA ±0.09 ±0.06 ±0.07 ±0.17 ±0.14 ±0.14 ±0.07 ±0.01 ±0.13 ±0.05 ±0.06 ±0.02 ±0.15 ±0.08 ±0.09 ±0.09

15.26 16.00 15.83 13.81 14.13 24.84 36.46 32.58 31.73 46.04 63.52 15.69 42.18 46.74 38.00 30.19 + SNAP-TTA ±0.16 ±0.09 ±0.06 ±0.04 ±0.01 ±0.03 ±0.13 ±0.03 ±0.08 ±0.21 ±0.06 ±0.08 ±0.07 ±0.05 ±0.14 **±0.08**

27.35 29.03 28.62 23.94 23.45 37.21 46.18 44.05 39.19 54.52 64.54 32.20 51.22 55.00 49.27 40.38 EATA ±0.04 ±0.15 ±0.27 ±0.06 ±0.60 ±0.30 ±0.13 ±0.20 ±0.22 ±0.01 ±0.06 ±0.62 ±0.16 ±0.10 ±0.21 ±0.21

29.48 31.20 30.69 26.68 25.90 38.24 46.60 44.62 39.31 54.82 64.44 32.87 51.41 55.41 49.78 41.43 + SNAP-TTA ±0.14 ±0.04 ±0.11 ±0.14 ±0.25 ±0.01 ±0.22 ±0.06 ±0.19 ±0.06 ±0.13 ±0.29 ±0.25 ±0.06 ±0.14 **±0.14**

28.12 29.30 29.63 22.37 23.88 39.34 45.36 45.69 36.73 54.91 64.11 10.96 52.22 55.76 49.60 39.20 SAR ±0.13 ±0.89 ±0.17 ±0.47 ±0.33 ±0.18 ±0.11 ±0.18 ±0.79 ±0.07 ±0.02 ±1.33 ±0.19 ±0.13 ±0.08 ±0.34

32.63 34.69 34.26 28.91 27.96 43.51 47.79 48.27 42.41 56.45 64.77 32.76 53.74 57.21 51.67 43.80 + SNAP-TTA ±0.11 ±0.23 ±0.18 ±0.27 ±0.29 ±0.14 ±0.03 ±0.11 ±0.13 ±0.09 ±0.07 ±3.04 ±0.13 ±0.28 ±0.12 **±0.35**

16.90 17.88 17.25 12.89 12.51 23.96 35.26 36.26 32.32 47.25 63.98 17.46 42.77 48.21 39.35 30.95 RoTTA ±0.15 ±0.11 ±0.08 ±0.17 ±0.05 ±0.03 ±0.16 ±0.01 ±0.07 ±0.02 ±0.13 ±0.18 ±0.09 ±0.24 ±0.15 ±0.11

18.63 19.94 19.35 14.88 14.34 25.88 36.47 37.13 33.32 47.74 63.96 19.08 42.98 48.73 40.27 **32.18**

+ SNAP-TTA ±0.07 ±0.08 ±0.06 ±0.08 ±0.05 ±0.03 ±0.03 ±0.02 ±0.11 ±0.17 ±0.06 ±0.21 ±0.07 ±0.17 ±0.20 **±0.09**

22.00 23.51 23.07 19.38 18.86 32.15 42.29 39.70 34.33 51.62 63.70 15.79 47.74 52.35 45.54 35.47 Tent ±3.47 ±3.92 ±3.85 ±2.30 ±2.06 ±3.40 ±2.45 ±3.27 ±0.60 ±2.30 ±0.29 ±4.61 ±2.84 ±2.27 ±2.98 ±2.71

26.21 27.85 27.50 23.62 22.73 36.01 44.11 42.19 38.15 52.95 64.57 30.23 48.56 53.71 47.09 39.03 + SNAP-TTA ±4.92 ±5.36 ±5.30 ±4.23 ±4.11 ±5.57 ±3.72 ±4.49 ±3.37 ±3.47 ±1.18 ±5.15 ±4.29 ±3.31 ±4.09 **±4.17**

10.97 11.92 11.98 11.45 11.38 22.39 34.96 30.88 29.89 44.09 61.96 13.08 40.20 45.27 36.71 27.81 CoTTA ±0.32 ±0.32 ±0.18 ±0.04 ±0.34 ±0.02 ±0.15 ±0.14 ±0.09 ±0.23 ±0.05 ±0.28 ±0.18 ±0.16 ±0.10 ±0.17

15.13 16.03 15.91 13.86 14.02 24.90 36.51 32.56 31.81 46.02 63.60 15.69 41.94 46.78 38.03 30.19 + SNAP-TTA ±0.06 ±0.09 ±0.04 ±0.00 ±0.07 ±0.05 ±0.05 ±0.06 ±0.12 ±0.06 ±0.10 ±0.04 ±0.09 ±0.09 ±0.12 **±0.07**

22.43 23.78 23.26 19.38 19.42 32.18 43.22 40.65 36.64 52.38 63.87 24.59 48.13 52.89 46.33 36.61 EATA ±0.05 ±0.16 ±0.43 ±0.26 ±0.51 ±0.31 ±0.19 ±0.15 ±0.16 ±0.27 ±0.05 ±1.52 ±0.40 ±0.12 ±0.14 ±0.32

26.10 27.29 27.13 22.38 22.15 33.45 43.92 40.96 36.68 52.71 63.77 27.93 48.47 53.23 47.46 38.24 + SNAP-TTA ±0.09 ±0.13 ±0.20 ±0.32 ±0.14 ±0.27 ±0.08 ±0.16 ±0.01 ±0.09 ±0.10 ±0.18 ±0.24 ±0.10 ±0.17 **±0.15**

26.12 27.56 26.93 22.51 23.35 36.03 44.48 43.19 37.26 53.82 64.15 19.87 50.78 54.78 48.43 38.62 SAR ±0.17 ±0.01 ±0.11 ±0.24 ±0.21 ±0.21 ±0.09 ±0.09 ±0.32 ±0.21 ±0.11 ±2.10 ±0.12 ±0.18 ±0.07 ±0.28

30.28 31.97 31.30 26.67 26.31 39.66 46.08 45.43 40.26 54.76 64.62 36.12 51.26 55.42 49.63 41.99 + SNAP-TTA ±0.16 ±0.24 ±0.12 ±0.34 ±0.37 ±0.25 ±0.04 ±0.09 ±0.13 ±0.23 ±0.05 ±0.67 ±0.06 ±0.20 ±0.06 **±0.20**

14.77 15.59 15.33 13.17 13.19 23.85 35.38 32.73 30.77 45.22 63.08 15.62 41.05 46.15 37.19 29.54 RoTTA ±0.04 ±0.04 ±0.04 ±0.07 ±0.10 ±0.05 ±0.05 ±0.03 ±0.04 ±0.15 ±0.12 ±0.02 ±0.10 ±0.07 ±0.13 ±0.07

15.35 16.20 16.01 13.67 13.66 24.27 35.62 33.04 31.02 45.38 62.95 15.96 41.06 46.17 37.44 **29.85**

| 876 877 878 879 880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917   | 1 0.5 0.3 0.1   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|

+ SNAP-TTA ±0.03 ±0.01 ±0.07 ±0.09 ±0.07 ±0.03 ±0.01 ±0.07 ±0.04 ±0.11 ±0.08 ±0.08 ±0.11 ±0.07 ±0.19 **±0.07**

Table 7: STTA classification accuracy (%) comparing with and without SNAP-TTA on ImageNet-C through Adaptation Rates(AR) (0.05, 0.03, and 0.01). **Bold** numbers are the highest accuracy.

918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971

AR Methods **Gau. Shot Imp. Def. Gla. Mot. Zoom Snow Fro. Fog Brit. Cont. Elas. Pix. JPEG Avg.**

23.77 24.65 24.44 20.54 20.27 32.73 43.57 40.82 35.92 52.78 63.82 15.95 49.33 53.46 47.19 36.62 Tent ±0.40 ±0.43 ±0.58 ±0.70 ±0.69 ±0.30 ±0.14 ±0.15 ±0.33 ±0.12 ±0.02 ±1.18 ±0.18 ±0.09 ±0.03 ±0.35

29.12 30.46 30.30 25.77 25.22 38.21 46.14 44.29 39.95 54.65 65.47 33.81 50.83 55.59 49.21 41.27 + SNAP-TTA ±0.09 ±0.22 ±0.48 ±0.20 ±0.23 ±0.43 ±0.00 ±0.13 ±0.07 ±0.15 ±0.09 ±1.10 ±0.13 ±0.10 ±0.03 **±0.23**

11.03 11.91 11.75 11.03 11.20 22.30 34.98 30.87 29.78 43.99 61.87 12.92 40.26 45.23 36.63 27.72 CoTTA ±0.30 ±0.57 ±0.33 ±0.24 ±0.46 ±0.18 ±0.05 ±0.08 ±0.01 ±0.11 ±0.06 ±0.36 ±0.19 ±0.17 ±0.07 ±0.21

15.22 15.97 15.93 13.91 14.05 24.87 36.48 32.60 31.65 46.09 63.59 15.67 42.00 46.71 37.96 30.18 + SNAP-TTA ±0.08 ±0.11 ±0.03 ±0.06 ±0.12 ±0.04 ±0.00 ±0.07 ±0.04 ±0.03 ±0.07 ±0.05 ±0.03 ±0.09 ±0.09 **±0.06**

19.53 20.65 20.72 16.74 16.96 29.11 41.22 37.96 34.84 50.75 63.29 19.86 45.92 51.15 44.13 34.19 EATA ±0.31 ±0.66 ±0.75 ±0.41 ±0.58 ±0.49 ±0.27 ±0.18 ±0.23 ±0.21 ±0.13 ±1.26 ±0.35 ±0.17 ±0.09 ±0.41

22.83 23.95 23.62 19.43 19.70 30.34 41.59 38.06 35.06 50.98 63.30 23.72 46.26 51.52 45.46 35.72 + SNAP-TTA ±0.10 ±0.34 ±0.30 ±0.09 ±0.19 ±0.56 ±0.08 ±0.11 ±0.21 ±0.18 ±0.13 ±0.30 ±0.16 ±0.16 ±0.18 **±0.21**

23.25 24.23 23.66 19.98 20.38 33.05 43.04 40.73 36.06 52.61 64.09 20.17 49.00 53.35 46.73 36.69 SAR ±0.21 ±0.34 ±0.30 ±0.09 ±0.16 ±0.30 ±0.16 ±0.02 ±0.12 ±0.09 ±0.07 ±0.84 ±0.11 ±0.10 ±0.11 ±0.20

27.54 29.03 28.66 24.05 23.42 36.28 44.12 42.89 38.54 53.24 64.25 31.83 48.79 54.04 47.80 39.63 + SNAP-TTA ±0.16 ±0.05 ±0.04 ±0.16 ±0.08 ±0.12 ±0.10 ±0.11 ±0.07 ±0.07 ±0.05 ±0.24 ±0.23 ±0.19 ±0.08 **±0.12**

14.42 15.22 15.02 13.25 13.31 23.79 35.27 32.09 30.43 44.71 62.64 15.24 40.63 45.55 36.75 29.22 RoTTA ±0.06 ±0.05 ±0.10 ±0.11 ±0.07 ±0.03 ±0.08 ±0.05 ±0.07 ±0.13 ±0.14 ±0.09 ±0.10 ±0.07 ±0.16 ±0.09

14.65 15.48 15.29 13.43 13.45 23.93 35.33 32.18 30.53 44.71 62.58 15.41 40.64 45.55 36.81 **29.33**

| 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955   | 0.05 0.03 0.01   |
|---------------------------------------------------------------------------------------------------------------------------------------------------|------------------|

+ SNAP-TTA ±0.06 ±0.02 ±0.08 ±0.09 ±0.09 ±0.03 ±0.05 ±0.04 ±0.05 ±0.16 ±0.10 ±0.04 ±0.09 ±0.10 ±0.14 **±0.08**

21.76 22.76 22.58 19.06 18.90 30.85 42.34 38.94 35.53 51.58 63.42 18.61 47.96 52.41 45.56 35.48 Tent ±0.17 ±0.35 ±0.17 ±0.04 ±0.12 ±0.22 ±0.12 ±0.26 ±0.31 ±0.18 ±0.11 ±0.91 ±0.26 ±0.21 ±0.08 ±0.23

26.42 28.20 27.81 23.79 22.82 35.77 44.80 42.37 38.81 53.34 64.95 30.05 49.28 54.16 47.57 39.34 + SNAP-TTA ±0.14 ±0.26 ±0.37 ±0.46 ±0.21 ±0.11 ±0.16 ±0.34 ±0.14 ±0.06 ±0.11 ±0.62 ±0.17 ±0.09 ±0.08 **±0.22**

10.61 12.36 11.78 11.66 11.32 22.25 35.01 30.88 29.84 44.09 61.83 12.92 40.26 45.20 36.58 27.77 CoTTA ±0.18 ±0.36 ±0.57 ±0.57 ±0.26 ±0.11 ±0.18 ±0.24 ±0.07 ±0.11 ±0.16 ±0.12 ±0.19 ±0.11 ±0.09 ±0.22

15.29 16.02 16.00 13.99 14.06 24.78 36.54 32.62 31.70 46.01 63.49 15.69 42.05 46.75 37.97 30.20 + SNAP-TTA ±0.08 ±0.07 ±0.09 ±0.07 ±0.11 ±0.05 ±0.07 ±0.06 ±0.08 ±0.01 ±0.04 ±0.04 ±0.18 ±0.19 ±0.08 **±0.08**

17.17 18.34 17.94 14.48 15.04 26.31 39.47 35.51 33.41 49.16 **63.06** 18.01 44.16 49.90 42.47 32.30 EATA ±0.41 ±0.19 ±0.36 ±0.82 ±0.22 ±0.25 ±0.33 ±0.50 ±0.33 ±0.19 **±0.05** ±0.88 ±0.31 ±0.09 ±0.31 ±0.35

20.75 21.87 21.28 17.34 17.90 28.08 39.84 36.27 33.54 **49.50** 63.04 20.86 44.68 49.97 43.53 33.90 + SNAP-TTA ±0.32 ±0.41 ±0.35 ±0.30 ±0.34 ±0.34 ±0.16 ±0.13 ±0.11 **±0.12** ±0.07 ±0.33 ±0.28 ±0.13 ±0.03 **±0.23**

20.38 21.34 21.18 18.24 18.28 30.56 41.63 38.57 35.23 51.19 63.74 20.40 47.32 52.02 44.81 34.99 SAR ±0.10 ±0.14 ±0.36 ±0.18 ±0.27 ±0.08 ±0.12 ±0.17 ±0.28 ±0.22 ±0.04 ±0.20 ±0.09 ±0.09 ±0.19 ±0.17

25.11 26.27 26.00 22.02 21.25 33.51 42.86 40.83 37.09 51.87 63.83 28.36 47.19 52.63 45.80 37.64 + SNAP-TTA ±0.23 ±0.31 ±0.10 ±0.49 ±0.56 ±0.31 ±0.14 ±0.16 ±0.21 ±0.18 ±0.10 ±0.29 ±0.34 ±0.06 ±0.30 **±0.25**

14.36 15.12 14.95 13.30 13.34 23.78 35.23 31.89 30.33 44.52 62.48 15.20 40.50 45.36 36.63 29.13 RoTTA ±0.04 ±0.03 ±0.08 ±0.08 ±0.08 ±0.04 ±0.05 ±0.04 ±0.07 ±0.11 ±0.12 ±0.01 ±0.11 ±0.07 ±0.17 ±0.07

14.45 15.21 15.06 13.35 13.42 23.83 35.26 31.92 30.36 44.53 62.47 15.27 40.50 45.39 36.65 **29.18**

+ SNAP-TTA ±0.04 ±0.02 ±0.08 ±0.08 ±0.07 ±0.04 ±0.06 ±0.02 ±0.08 ±0.10 ±0.09 ±0.04 ±0.10 ±0.08 ±0.16 **±0.07**

17.09 17.70 17.69 14.91 15.25 25.23 38.66 34.15 32.28 48.14 62.65 15.76 43.44 49.14 41.18 31.55 Tent ±0.14 ±0.10 ±0.13 ±0.23 ±0.09 ±0.25 ±0.27 ±0.27 ±0.21 ±0.21 ±0.16 ±0.48 ±0.23 ±0.04 ±0.10 ±0.19

20.66 21.73 21.55 18.46 18.28 29.88 40.63 36.97 34.89 49.85 64.29 22.64 45.13 50.77 43.17 34.59 + SNAP-TTA ±0.02 ±0.12 ±0.18 ±0.34 ±0.33 ±0.12 ±0.14 ±0.21 ±0.10 ±0.26 ±0.10 ±0.14 ±0.29 ±0.07 ±0.51 **±0.19**

11.11 13.24 11.86 10.85 10.97 22.18 34.96 30.88 29.63 44.09 61.71 12.81 40.16 45.14 36.73 27.75 CoTTA ±0.61 ±0.12 ±0.65 ±0.59 ±0.98 ±0.05 ±0.18 ±0.14 ±0.21 ±0.21 ±0.22 ±0.53 ±0.20 ±0.22 ±0.12 ±0.34

15.09 16.00 15.83 13.84 14.06 24.70 36.47 32.59 31.66 46.10 63.62 15.60 42.03 46.74 38.17 30.17 + SNAP-TTA ±0.04 ±0.09 ±0.14 ±0.09 ±0.02 ±0.07 ±0.02 ±0.11 ±0.03 ±0.15 ±0.07 ±0.06 ±0.10 ±0.01 ±0.20 **±0.08**

14.85 15.61 15.69 13.26 13.37 23.72 36.18 32.57 **31.14** 46.06 **62.35** 13.88 41.91 47.00 38.88 29.76 EATA ±0.13 ±0.21 ±0.21 ±0.04 ±0.06 ±0.19 ±0.13 ±0.09 **±0.06** ±0.29 **±0.09** ±0.35 ±0.17 ±0.15 ±0.09 ±0.15

16.73 17.55 17.30 14.35 14.64 24.13 36.83 **32.81** 31.09 **46.63** 62.20 15.26 42.34 47.44 39.81 30.61 + SNAP-TTA ±0.12 ±0.10 ±0.19 ±0.09 ±0.10 ±0.36 ±0.23 **±0.08** ±0.10 **±0.19** ±0.16 ±0.54 ±0.12 ±0.18 ±0.34 **±0.19**

16.08 17.04 16.69 14.72 14.78 25.92 37.85 34.07 32.25 47.66 63.15 17.20 43.05 48.78 40.14 31.29 SAR ±0.08 ±0.07 ±0.10 ±0.16 ±0.12 ±0.13 ±0.05 ±0.24 ±0.11 ±0.13 ±0.05 ±0.15 ±0.20 ±0.09 ±0.20 ±0.13

18.89 19.45 19.70 16.70 16.55 27.69 38.57 35.34 33.09 48.08 63.04 20.39 42.95 48.76 40.99 32.68 + SNAP-TTA ±0.15 ±0.15 ±0.12 ±0.14 ±0.15 ±0.16 ±0.11 ±0.22 ±0.09 ±0.31 ±0.07 ±0.12 ±0.29 ±0.26 ±0.33 **±0.18**

14.30 15.06 14.89 13.30 13.37 23.78 35.22 31.79 30.27 44.40 62.40 15.16 40.42 45.27 36.54 29.08 RoTTA ±0.05 ±0.03 ±0.07 ±0.07 ±0.08 ±0.04 ±0.06 ±0.04 ±0.06 ±0.14 ±0.11 ±0.06 ±0.10 ±0.05 ±0.16 ±0.07

14.30 15.07 14.92 13.30 13.38 23.78 35.22 31.78 30.26 44.41 62.40 15.15 40.43 45.27 36.54 **29.08**

+ SNAP-TTA ±0.06 ±0.03 ±0.08 ±0.08 ±0.07 ±0.04 ±0.06 ±0.04 ±0.07 ±0.14 ±0.11 ±0.05 ±0.09 ±0.04 ±0.15 **±0.07**

C.2 CIFAR10-C Table 8: STTA classification accuracy (%) comparing with and without SNAP-TTA on CIFAR10-C through Adaptation Rates(AR) (0.5, 0.3, and 0.1), including results for full adaptation (AR=1). **Bold** numbers are the highest accuracy.

AR Methods **Gau. Shot Imp. Def. Gla. Mot. Zoom Snow Fro. Fog Brit. Cont. Elas. Pix. JPEG Avg.**

22.13 29.25 22.53 54.54 55.10 67.45 64.37 78.25 69.93 74.26 91.29 35.45 77.20 46.56 73.38 57.45 Source ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00 ±0.00

63.72 65.67 57.14 84.99 62.72 83.86 84.26 78.98 76.95 83.32 88.46 84.60 73.96 76.61 68.79 75.60 BN stats ±0.48 ±0.12 ±0.25 ±0.31 ±0.23 ±0.48 ±0.30 ±0.30 ±0.08 ±0.17 ±0.16 ±0.17 ±0.18 ±0.02 ±0.42 ±0.24 73.66 76.18 68.04 86.61 67.12 85.73 86.24 82.34 81.56 86.02 89.99 87.16 76.40 82.95 76.45 80.43 Tent ±0.88 ±0.94 ±1.32 ±0.50 ±0.76 ±0.38 ±0.09 ±0.94 ±0.64 ±0.18 ±0.16 ±2.50 ±0.82 ±0.15 ±0.46 ±0.71 71.95 73.97 67.03 83.91 66.75 82.64 83.34 79.92 79.49 82.41 88.39 80.14 75.38 79.24 75.42 78.00 CoTTA ±0.32 ±0.48 ±0.66 ±0.20 ±0.08 ±0.34 ±0.19 ±0.09 ±0.13 ±0.23 ±0.18 ±0.17 ±0.09 ±0.07 ±0.25 ±0.23 75.82 77.61 69.63 87.14 69.41 85.96 87.08 83.42 82.28 86.58 90.40 89.26 77.62 83.35 77.77 81.56 EATA ±0.50 ±0.27 ±0.87 ±0.29 ±0.68 ±0.39 ±0.27 ±0.38 ±0.29 ±0.41 ±0.17 ±0.39 ±0.28 ±0.32 ±0.20 ±0.38 73.52 74.03 65.45 85.69 65.01 84.63 85.01 81.47 80.91 84.18 88.70 86.23 74.94 81.20 74.84 79.05 SAR ±1.53 ±0.46 ±1.81 ±0.37 ±0.35 ±0.53 ±0.34 ±0.37 ±0.72 ±0.09 ±0.12 ±0.16 ±0.03 ±0.28 ±0.69 ±0.52

66.54 68.60 60.27 85.73 64.84 84.68 85.01 80.15 78.02 84.13 89.00 84.91 75.06 77.96 70.12 77.00

RoTTA ±0.46 ±0.23 ±0.46 ±0.35 ±0.63 ±0.36 ±0.45 ±0.56 ±0.06 ±0.09 ±0.27 ±0.19 ±0.15 ±0.16 ±0.36 ±0.32

73.44 75.93 67.18 86.52 67.28 85.25 86.23 82.24 80.35 85.39 89.80 87.77 77.00 82.08 75.58 80.14 Tent ±0.61 ±0.44 ±0.78 ±0.17 ±1.78 ±0.49 ±0.42 ±0.77 ±0.14 ±0.20 ±0.28 ±0.27 ±0.65 ±0.68 ±0.60 ±0.55

75.17 77.66 68.78 88.25 69.18 87.11 88.19 84.21 82.72 87.34 91.63 86.30 78.76 83.43 77.28 81.74 + SNAP-TTA ±0.00 ±0.78 ±1.26 ±0.38 ±0.51 ±0.18 ±0.13 ±0.29 ±0.45 ±0.51 ±0.12 ±1.07 ±0.28 ±0.18 ±0.50 **±0.44**

65.08 66.67 61.30 77.50 61.36 77.70 77.37 74.05 72.86 77.43 82.69 72.44 70.52 70.94 69.79 71.85 CoTTA ±0.26 ±0.21 ±0.16 ±0.48 ±0.15 ±0.37 ±0.37 ±0.22 ±0.44 ±0.19 ±0.30 ±0.72 ±0.07 ±0.27 ±0.10 ±0.29

71.89 74.18 66.92 85.46 67.57 84.27 84.91 81.10 80.62 84.06 90.16 82.14 76.75 80.23 75.98 79.08 + SNAP-TTA ±0.45 ±0.33 ±0.19 ±0.32 ±0.26 ±0.22 ±0.18 ±0.09 ±0.46 ±0.24 ±0.17 ±0.33 ±0.16 ±0.38 ±0.50 **±0.28**

73.95 75.82 68.00 86.83 67.83 85.27 86.48 82.63 80.99 85.45 89.86 87.61 77.01 82.13 76.11 80.40 EATA ±0.22 ±0.18 ±0.70 ±0.25 ±0.50 ±0.39 ±0.15 ±0.50 ±0.05 ±0.16 ±0.18 ±0.53 ±0.31 ±0.18 ±0.45 ±0.32

74.85 77.63 68.43 88.53 69.70 87.19 88.16 83.87 82.84 87.18 91.54 89.62 78.91 83.76 77.36 81.97 + SNAP-TTA ±0.51 ±0.46 ±0.43 ±0.17 ±0.69 ±0.35 ±0.18 ±0.42 ±0.33 ±0.15 ±0.12 ±0.38 ±0.48 ±0.14 ±0.22 **±0.33**

69.10 72.37 63.22 85.18 64.30 83.94 85.07 80.11 79.64 83.91 88.64 84.21 75.70 79.10 72.92 77.83 SAR ±1.63 ±1.05 ±0.44 ±0.25 ±1.02 ±0.12 ±0.45 ±0.17 ±0.60 ±0.37 ±0.10 ±0.30 ±0.34 ±0.52 ±0.09 ±0.50

73.98 75.48 66.41 86.63 68.15 85.50 86.53 81.62 80.20 85.06 91.46 87.04 77.22 81.16 75.53 80.13 + SNAP-TTA ±0.48 ±0.65 ±1.26 ±0.15 ±0.07 ±0.15 ±0.10 ±0.39 ±0.17 ±0.27 ±0.03 ±0.11 ±0.45 ±0.27 ±0.23 **±0.32**

65.02 66.84 58.38 85.26 63.51 83.81 84.66 79.26 76.76 83.46 88.27 83.47 74.43 77.39 69.13 75.98 RoTTA ±0.04 ±0.52 ±0.33 ±0.42 ±0.18 ±0.15 ±0.20 ±0.29 ±0.49 ±0.21 ±0.04 ±0.05 ±0.16 ±0.29 ±0.41 ±0.25

66.03 68.09 58.88 87.09 64.55 85.70 86.48 80.97 78.87 85.29 90.28 86.22 76.05 78.76 70.51 **77.58**

| 983 984 985 986 987 988 989 990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025   | 1 0.5 0.3 0.1   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|

+ SNAP-TTA ±0.14 ±0.15 ±0.06 ±0.27 ±0.07 ±0.03 ±0.02 ±0.22 ±0.20 ±0.22 ±0.13 ±0.10 ±0.22 ±0.22 ±0.35 **±0.16**

71.18 74.06 65.44 85.93 66.01 84.37 85.90 81.31 79.80 84.80 89.58 84.01 75.96 80.46 74.09 78.86 Tent ±0.99 ±0.80 ±1.17 ±0.28 ±0.97 ±0.14 ±0.17 ±0.40 ±0.09 ±0.25 ±0.23 ±0.30 ±0.30 ±0.39 ±0.54 ±0.47

74.95 77.29 67.59 88.27 67.46 86.97 87.64 83.46 82.45 86.72 91.22 87.79 78.26 82.61 75.79 81.23 + SNAP-TTA ±0.84 ±0.55 ±0.46 ±0.27 ±0.26 ±0.21 ±0.16 ±0.40 ±0.19 ±0.19 ±0.21 ±0.98 ±0.35 ±0.38 ±0.32 **±0.39**

63.01 64.38 58.95 75.43 59.65 76.08 75.47 71.75 70.33 75.52 80.94 70.53 68.75 67.87 67.55 69.75 CoTTA ±0.12 ±0.64 ±0.74 ±0.61 ±0.48 ±0.58 ±0.16 ±0.55 ±0.48 ±0.32 ±0.49 ±0.51 ±0.65 ±0.30 ±0.37 ±0.47

71.39 73.57 66.29 85.22 66.71 84.20 84.64 80.77 80.56 84.06 89.85 81.86 76.48 79.94 75.69 78.75 + SNAP-TTA ±0.31 ±0.27 ±0.10 ±0.22 ±0.19 ±0.18 ±0.13 ±0.21 ±0.32 ±0.15 ±0.17 ±0.08 ±0.07 ±0.24 ±0.27 **±0.19**

70.98 73.70 65.73 86.01 66.71 84.36 86.10 80.92 79.87 84.48 89.29 86.33 76.19 80.66 73.98 79.02 EATA ±1.05 ±0.28 ±1.68 ±0.35 ±0.81 ±0.23 ±0.38 ±0.47 ±0.09 ±0.04 ±0.19 ±0.31 ±0.20 ±0.58 ±0.52 ±0.48

74.19 76.64 67.89 87.93 68.56 87.08 87.89 83.56 82.20 86.60 91.11 88.94 78.10 83.03 75.83 81.30 + SNAP-TTA ±0.38 ±0.68 ±0.19 ±0.25 ±0.20 ±0.05 ±0.34 ±0.30 ±0.25 ±0.23 ±0.22 ±0.61 ±0.14 ±0.20 ±0.43 **±0.30**

69.10 72.37 63.22 85.18 64.30 83.94 85.07 80.11 79.64 83.91 88.64 84.21 75.70 79.10 72.92 77.83 SAR ±1.63 ±1.05 ±0.44 ±0.25 ±1.02 ±0.12 ±0.45 ±0.17 ±0.60 ±0.37 ±0.10 ±0.30 ±0.34 ±0.52 ±0.09 ±0.50

72.72 75.25 65.78 86.53 66.19 85.53 86.40 81.61 80.53 85.08 91.41 86.74 77.23 81.00 74.52 79.77 + SNAP-TTA ±0.94 ±0.30 ±1.06 ±0.16 ±0.60 ±0.26 ±0.27 ±0.45 ±0.64 ±0.23 ±0.14 ±0.08 ±0.41 ±0.37 ±1.04 **±0.46**

64.09 66.07 57.58 84.97 62.66 83.06 84.08 78.60 76.40 82.86 88.03 83.21 74.14 76.35 68.70 75.39 RoTTA ±0.44 ±0.13 ±0.63 ±0.20 ±0.15 ±0.18 ±0.17 ±0.34 ±0.36 ±0.05 ±0.22 ±0.24 ±0.58 ±0.47 ±0.17 ±0.29

65.83 67.57 58.39 86.97 64.22 85.63 86.39 80.75 78.90 85.21 90.19 85.92 75.92 78.91 70.42 **77.41**

+ SNAP-TTA ±0.18 ±0.19 ±0.29 ±0.33 ±0.16 ±0.18 ±0.09 ±0.15 ±0.08 ±0.17 ±0.16 ±0.21 ±0.09 ±0.05 ±0.37 **±0.18**

67.32 69.39 60.69 85.34 63.82 83.52 84.70 79.68 77.79 83.75 88.53 83.12 75.18 77.82 71.47 76.81 Tent ±0.93 ±0.96 ±0.36 ±0.24 ±0.41 ±0.13 ±0.15 ±0.41 ±0.50 ±0.08 ±0.49 ±0.66 ±0.68 ±0.69 ±0.44 ±0.48

70.22 71.48 63.08 87.35 65.74 85.89 86.38 81.93 80.00 85.62 90.34 87.47 76.44 79.63 72.72 78.95 + SNAP-TTA ±0.44 ±0.91 ±0.04 ±0.20 ±0.26 ±0.25 ±0.32 ±0.33 ±0.21 ±0.14 ±0.22 ±0.11 ±0.12 ±0.14 ±0.39 **±0.27**

59.11 60.26 56.07 72.23 56.77 73.55 72.20 68.05 66.68 72.88 77.66 65.95 65.67 64.12 65.16 66.42 CoTTA ±0.43 ±0.56 ±0.65 ±0.69 ±0.64 ±0.68 ±0.94 ±0.63 ±0.52 ±0.56 ±1.15 ±1.17 ±0.83 ±0.95 ±0.58 ±0.73

71.70 73.54 66.70 85.16 66.83 84.30 84.88 81.02 80.61 84.20 89.84 81.71 76.60 79.66 75.71 78.83 + SNAP-TTA ±0.40 ±0.21 ±0.02 ±0.19 ±0.39 ±0.08 ±0.20 ±0.25 ±0.24 ±0.23 ±0.08 ±0.20 ±0.20 ±0.14 ±0.25 **±0.20**

66.65 68.96 59.73 84.93 63.26 83.10 84.53 79.28 77.46 83.48 88.12 82.46 74.49 77.48 70.43 76.29 EATA ±0.43 ±0.47 ±0.15 ±0.27 ±0.36 ±0.24 ±0.15 ±0.44 ±0.42 ±0.13 ±0.09 ±0.24 ±0.20 ±0.69 ±0.25 ±0.30

69.29 70.49 61.71 87.32 65.48 85.96 86.64 81.44 79.56 85.47 90.50 86.84 76.32 79.64 72.51 78.61 + SNAP-TTA ±0.39 ±0.57 ±0.37 ±0.42 ±0.38 ±0.29 ±0.21 ±0.34 ±0.47 ±0.23 ±0.38 ±0.36 ±0.21 ±0.12 ±0.32 **±0.34**

66.11 68.18 59.15 84.91 62.87 82.33 84.27 79.23 77.58 83.21 88.29 82.60 74.65 75.92 70.79 76.01 SAR ±0.59 ±0.83 ±0.72 ±0.45 ±0.27 ±0.60 ±0.13 ±0.32 ±0.43 ±0.18 ±0.09 ±0.57 ±0.46 ±0.77 ±0.40 ±0.45

67.76 70.68 60.82 86.78 64.73 85.29 86.22 80.82 79.30 84.95 91.33 86.59 75.72 78.72 71.24 78.06 + SNAP-TTA ±0.22 ±0.14 ±1.08 ±0.26 ±0.43 ±0.10 ±0.11 ±0.23 ±0.48 ±0.28 ±0.17 ±0.14 ±0.26 ±0.35 ±0.46 **±0.31**

63.12 64.84 56.72 84.49 62.15 82.53 83.84 78.03 76.13 82.88 87.48 81.49 73.75 76.04 68.24 74.78 RoTTA ±0.33 ±0.21 ±0.30 ±0.04 ±0.17 ±0.30 ±0.02 ±0.29 ±0.71 ±0.16 ±0.08 ±0.11 ±0.14 ±0.29 ±0.27 ±0.23

65.35 66.99 58.09 86.77 63.63 85.47 86.01 80.54 78.38 84.99 90.00 85.99 75.67 78.14 70.09 **77.07**

+ SNAP-TTA ±0.20 ±0.15 ±0.18 ±0.18 ±0.18 ±0.13 ±0.21 ±0.11 ±0.24 ±0.43 ±0.23 ±0.03 ±0.17 ±0.06 ±0.23 **±0.18**

Table 9: STTA classification accuracy (%) comparing with and without SNAP-TTA on CIFAR10-C through Adaptation Rates(AR) (0.05, 0.03, and 0.01). **Bold** numbers are the highest accuracy.

1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079

AR Methods **Gau. Shot Imp. Def. Gla. Mot. Zoom Snow Fro. Fog Brit. Cont. Elas. Pix. JPEG Avg.**

64.65 67.08 58.48 85.00 62.61 82.76 84.63 79.01 77.66 83.32 88.00 82.34 74.16 77.11 69.40 75.75 Tent ±0.55 ±0.58 ±0.42 ±0.60 ±0.44 ±0.70 ±0.55 ±0.74 ±0.91 ±0.48 ±0.56 ±0.93 ±0.10 ±0.60 ±0.48 ±0.57

67.71 69.84 59.53 87.10 64.66 85.73 86.35 80.68 78.92 85.60 90.19 86.72 76.16 78.86 70.95 77.93 + SNAP-TTA ±0.38 ±0.82 ±1.10 ±0.15 ±0.25 ±0.20 ±0.20 ±0.23 ±0.14 ±0.08 ±0.31 ±0.20 ±0.17 ±0.42 ±0.30 **±0.33**

59.27 61.18 56.33 72.22 57.37 74.27 72.61 70.03 68.68 74.82 79.72 65.57 66.92 64.13 65.25 67.22 CoTTA ±0.66 ±1.12 ±0.06 ±1.43 ±1.10 ±1.46 ±1.11 ±1.02 ±0.92 ±1.09 ±1.07 ±1.38 ±1.14 ±1.27 ±0.98 ±1.05

71.42 73.31 65.91 85.23 67.01 84.19 84.91 80.80 80.56 84.19 90.00 82.09 76.31 79.79 75.18 78.73 + SNAP-TTA ±0.29 ±0.12 ±0.13 ±0.11 ±0.21 ±0.20 ±0.14 ±0.19 ±0.34 ±0.14 ±0.23 ±0.35 ±0.05 ±0.29 ±0.21 **±0.20**

64.68 67.01 58.07 84.90 62.56 82.64 84.57 78.77 77.16 83.09 87.80 81.62 74.05 76.99 69.31 75.55 EATA ±0.31 ±0.37 ±0.24 ±0.54 ±0.33 ±0.67 ±0.61 ±0.71 ±0.92 ±0.44 ±0.47 ±0.59 ±0.28 ±0.41 ±0.71 ±0.51

67.36 68.73 59.35 87.05 64.36 85.62 86.48 81.31 78.73 85.33 90.03 86.31 76.04 78.79 70.90 77.76 + SNAP-TTA ±0.33 ±0.26 ±0.37 ±0.22 ±0.18 ±0.18 ±0.25 ±0.24 ±0.22 ±0.15 ±0.24 ±0.07 ±0.12 ±0.27 ±0.38 **±0.23**

64.79 66.32 57.58 84.66 62.46 81.42 84.13 78.87 77.20 82.62 88.10 82.12 74.04 75.38 69.13 75.25 SAR ±0.13 ±0.86 ±0.69 ±0.72 ±0.26 ±1.52 ±0.34 ±0.26 ±0.81 ±1.24 ±0.41 ±0.74 ±0.05 ±0.80 ±0.52 ±0.62

66.00 68.85 58.47 86.54 63.06 85.26 86.13 80.38 78.17 85.17 90.93 85.96 75.27 77.37 70.61 77.21 + SNAP-TTA ±0.17 ±0.75 ±0.42 ±0.25 ±0.28 ±0.09 ±0.38 ±0.09 ±0.27 ±0.13 ±0.36 ±0.20 ±0.31 ±0.28 ±0.30 **±0.29**

63.21 64.87 56.60 84.64 62.16 82.31 84.13 78.16 76.39 82.90 87.44 81.47 73.59 76.02 68.09 74.80 RoTTA ±0.37 ±0.62 ±0.28 ±0.52 ±0.31 ±0.63 ±0.56 ±0.71 ±0.95 ±0.62 ±0.46 ±0.65 ±0.42 ±0.40 ±0.33 ±0.52

65.28 66.91 57.88 86.75 63.51 85.48 86.17 80.46 78.38 85.24 89.99 85.82 75.66 77.98 70.15 **77.05**

| 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063   | 0.05 0.03 0.01   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|

+ SNAP-TTA ±0.32 ±0.22 ±0.06 ±0.25 ±0.13 ±0.13 ±0.10 ±0.23 ±0.26 ±0.13 ±0.23 ±0.03 ±0.16 ±0.19 ±0.29 **±0.18**

64.36 66.21 57.65 84.73 62.95 83.07 84.50 78.46 76.99 83.00 88.07 82.62 73.93 76.50 68.82 75.46 Tent ±0.43 ±0.16 ±1.01 ±0.48 ±0.52 ±0.50 ±0.32 ±0.82 ±0.32 ±0.36 ±0.43 ±0.34 ±0.23 ±0.46 ±0.48 ±0.46

66.32 68.38 59.00 86.93 64.04 85.58 86.35 80.78 78.68 85.34 90.08 86.19 75.77 78.37 70.49 77.49 + SNAP-TTA ±0.61 ±0.71 ±0.52 ±0.19 ±0.24 ±0.34 ±0.05 ±0.10 ±0.02 ±0.05 ±0.10 ±0.31 ±0.05 ±0.06 ±0.08 **±0.23**

60.38 61.26 56.71 72.44 57.58 74.64 72.73 69.68 68.34 74.64 79.52 67.28 67.42 64.89 66.19 67.58 CoTTA ±1.71 ±1.94 ±2.47 ±2.23 ±1.85 ±1.74 ±2.61 ±2.03 ±2.02 ±2.52 ±2.37 ±1.89 ±1.77 ±0.79 ±1.73 ±1.98

71.12 73.68 66.34 85.30 66.64 84.25 84.55 80.88 80.11 84.06 89.89 81.98 76.27 79.77 75.35 78.68 + SNAP-TTA ±0.47 ±0.29 ±0.24 ±0.01 ±0.12 ±0.34 ±0.13 ±0.15 ±0.15 ±0.14 ±0.14 ±0.37 ±0.19 ±0.26 ±0.08 **±0.21**

63.99 65.95 57.39 84.71 62.66 83.11 84.44 78.42 76.63 82.97 88.00 82.55 73.85 76.46 68.91 75.34 EATA ±0.87 ±0.44 ±1.05 ±0.48 ±0.62 ±0.52 ±0.33 ±0.75 ±0.26 ±0.26 ±0.47 ±0.34 ±0.33 ±0.29 ±0.56 ±0.50

66.16 67.60 58.81 86.95 64.06 85.49 86.34 80.79 78.65 85.24 90.09 86.23 75.88 78.48 70.56 77.42 + SNAP-TTA ±0.03 ±0.41 ±0.36 ±0.13 ±0.17 ±0.36 ±0.08 ±0.01 ±0.25 ±0.13 ±0.12 ±0.08 ±0.18 ±0.10 ±0.47 **±0.19**

63.72 65.75 57.89 84.37 62.45 81.47 82.46 78.32 76.79 81.93 88.60 82.72 73.89 74.55 68.79 74.91 SAR ±0.46 ±0.29 ±0.65 ±0.81 ±0.69 ±1.61 ±2.95 ±0.81 ±0.24 ±1.33 ±0.68 ±0.29 ±0.43 ±0.98 ±0.61 ±0.85

65.40 67.68 58.37 86.72 63.11 85.10 86.18 79.93 78.05 84.92 90.93 85.58 75.30 77.22 69.97 76.96 + SNAP-TTA ±0.33 ±0.60 ±0.45 ±0.18 ±0.16 ±0.16 ±0.29 ±0.17 ±0.31 ±0.22 ±0.35 ±0.14 ±0.14 ±0.30 ±0.30 **±0.27**

63.36 65.10 56.64 84.62 62.41 82.96 84.35 78.10 76.42 82.69 87.90 82.34 73.56 76.09 68.39 75.00 RoTTA ±0.80 ±0.55 ±0.56 ±0.49 ±0.79 ±0.67 ±0.43 ±0.80 ±0.23 ±0.25 ±0.53 ±0.32 ±0.25 ±0.44 ±0.31 ±0.50

65.27 67.05 58.05 86.79 63.48 85.46 86.25 80.39 78.34 85.19 90.10 85.94 75.67 78.04 69.75 **77.05**

+ SNAP-TTA ±0.32 ±0.19 ±0.22 ±0.21 ±0.18 ±0.33 ±0.09 ±0.08 ±0.15 ±0.10 ±0.16 ±0.08 ±0.12 ±0.09 ±0.27 **±0.17**

62.43 64.13 55.85 84.03 62.21 82.47 83.87 77.71 76.55 82.75 87.35 81.83 73.24 75.34 67.73 74.50 Tent ±1.70 ±1.51 ±1.35 ±1.07 ±1.20 ±0.88 ±0.93 ±0.66 ±0.18 ±0.14 ±1.11 ±1.81 ±1.33 ±1.18 ±1.50 ±1.10

65.51 67.26 58.05 86.89 63.53 85.44 85.97 80.58 78.35 85.12 90.09 85.86 75.66 78.38 70.12 77.12 + SNAP-TTA ±0.24 ±0.31 ±0.34 ±0.28 ±0.07 ±0.33 ±0.20 ±0.12 ±0.12 ±0.16 ±0.21 ±0.11 ±0.08 ±0.21 ±0.33 **±0.21**

59.75 59.44 54.47 71.12 57.11 72.47 72.83 66.05 65.14 69.75 75.12 64.31 66.22 62.65 64.76 65.41 CoTTA ±4.69 ±6.21 ±5.57 ±5.10 ±4.35 ±4.52 ±4.80 ±7.60 ±7.65 ±9.79 ±6.79 ±6.46 ±4.50 ±5.27 ±5.36 ±5.91

71.79 73.61 65.98 85.34 66.76 84.26 84.93 80.64 80.38 83.94 89.98 82.47 76.48 79.61 75.60 78.79 + SNAP-TTA ±0.22 ±0.29 ±0.58 ±0.36 ±0.26 ±0.12 ±0.21 ±0.45 ±0.30 ±0.42 ±0.08 ±0.64 ±0.26 ±0.24 ±0.29 **±0.31**

62.36 63.92 55.73 84.05 62.24 82.38 83.90 77.66 76.48 82.67 87.34 81.82 73.30 75.31 67.76 74.46 EATA ±1.73 ±1.66 ±1.39 ±1.10 ±1.18 ±0.85 ±0.93 ±0.72 ±0.15 ±0.17 ±1.12 ±1.81 ±1.24 ±1.20 ±1.52 ±1.12

65.49 67.19 57.93 86.92 63.65 85.42 85.97 80.46 78.13 85.07 90.03 85.87 75.69 78.20 70.03 77.07 + SNAP-TTA ±0.29 ±0.04 ±0.40 ±0.41 ±0.18 ±0.28 ±0.24 ±0.18 ±0.27 ±0.13 ±0.10 ±0.20 ±0.11 ±0.13 ±0.46 **±0.23**

62.50 64.13 55.65 82.30 62.22 77.21 80.11 77.66 76.75 79.12 89.45 81.97 73.39 69.39 67.83 73.31 SAR ±1.69 ±1.83 ±1.38 ±3.37 ±1.21 ±6.27 ±6.19 ±0.80 ±0.34 ±3.28 ±1.79 ±1.97 ±1.21 ±5.48 ±1.65 ±2.57

65.06 66.93 57.66 86.76 62.78 85.05 85.94 79.95 77.62 84.65 90.72 85.48 75.34 75.72 69.61 76.62 + SNAP-TTA ±0.17 ±0.11 ±0.51 ±0.29 ±0.24 ±0.21 ±0.48 ±0.18 ±0.37 ±0.21 ±0.62 ±0.35 ±0.13 ±1.35 ±0.25 **±0.36**

62.25 63.71 55.59 84.05 62.17 82.32 83.86 77.56 76.39 82.64 87.27 81.75 73.21 75.15 67.75 74.38 RoTTA ±1.65 ±1.68 ±1.46 ±1.12 ±1.37 ±0.83 ±0.90 ±0.75 ±0.24 ±0.10 ±1.12 ±1.82 ±1.21 ±1.27 ±1.48 ±1.13

65.32 66.94 57.85 86.91 63.44 85.32 85.98 80.49 78.22 85.04 90.01 85.77 75.75 78.15 70.06 **77.02**

+ SNAP-TTA ±0.25 ±0.12 ±0.29 ±0.31 ±0.24 ±0.22 ±0.14 ±0.24 ±0.20 ±0.15 ±0.06 ±0.24 ±0.11 ±0.07 ±0.47 **±0.21**