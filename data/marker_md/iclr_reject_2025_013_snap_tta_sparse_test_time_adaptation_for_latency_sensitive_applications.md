**006**

**009**

**019**

**024**

**029 030**

**032**

**034**

**036**

# SNAP-TTA: SPARSE TEST-TIME ADAPTATION FOR LATENCY-SENSITIVE APPLICATIONS

Anonymous authors Paper under double-blind review

#### ABSTRACT

Test-Time Adaptation (TTA) methods use unlabeled test data to dynamically adjust models in response to distribution changes. However, existing TTA methods are not tailored for practical use on edge devices with limited computational capacity, resulting in a latency-accuracy trade-off. To address this problem, we propose SNAP-TTA, a sparse TTA framework that significantly reduces adaptation frequency and data usage, delivering latency reductions proportional to adaptation rate. It achieves competitive accuracy even with an adaptation rate as low as 0.01, demonstrating its ability to adapt infrequently while utilizing only a small portion of the data compared to full adaptation. Our approach involves (i) Class and Domain Representative Memory (CnDRM), which identifies key samples that are both class-representative and domain-representative to facilitate adaptation with minimal data, and (ii) Inference-only Batch-aware Memory Normalization (IoBMN), which leverages representative samples to adjust normalization layers on-the-fly during inference, aligning the model effectively to changing domains. When combined with five state-of-the-art TTA algorithms, SNAP-TTA maintains the performances of these methods even with much-reduced adaptation rates from 0.01 to 0.5, making it suitable for edge devices serving latency-sensitive applications.

# 1 INTRODUCTION

Deep learning models often suffer from performance degradation under domain shifts caused by environmental changes or noise [\(Quinonero-Candela et al., 2008\)](#page-12-0). Test-Time Adaptation (TTA) ˜ offers a promising solution for domain shifts by utilizing only unlabeled test data without requiring source data. While TTA algorithms have advanced in complexity to improve accuracy in data streams [\(Wang et al., 2021;](#page-12-1) [Niu et al., 2022;](#page-11-0) [Wang et al., 2022;](#page-12-2) [Yuan et al., 2023;](#page-12-3) [Niu et al., 2023;](#page-11-1) [Song et al., 2023\)](#page-12-4), they are typically designed for resource-rich servers, overlooking the computational and memory limitations crucial for real-world deployment. Operations such as backpropagation, data augmentation, and model ensembling [\(Wang et al., 2022;](#page-12-2) [Yuan et al., 2023;](#page-12-3) [Zhang et al.,](#page-13-0) [2022\)](#page-13-0) result in substantial latency and memory consumption, making state-of-the-art (SOTA) TTA methods inefficient for practical use (Section [2\)](#page-2-0).

For edge devices with limited computational power, such as mobile devices or IoT sensors, the adaptation latency from TTA methods becomes a critical bottleneck, particularly in latency-sensitive applications such as autonomous driving and real-time health monitoring. Moreover, the model must keep up with the data stream in those applications, but high computational overhead could cause it to miss critical samples, resulting in inference lags and reduced accuracy. This issue is exacerbated with fast data streams, such as high-frame-rate videos or high-performance sensors. For example, even a slight delay in processing sensor data can lead to dangerous situations in autonomous driving. A high adaptation latency that accumulates with each batch not only undermines real-time performance but also limits the potential of TTA algorithms in latency-sensitive applications.

In online TTA scenarios that require rapid response to incoming data streams on resourceconstrained devices, *Sparse TTA (STTA)*, which adapts occasionally rather than at every batch, can offer a practical solution by reducing the adaption overhead. However, na¨ıve STTA may result in performance degradation as it utilizes far less data (e.g., 0.1) for model adaptation (Figure [1\)](#page-1-0). The

**059**

**061**

**064**

**067**

**069 070 071**

**074**

**079**

**089 090 091**

**094**

**104 105 106**

![](_page_1_Diagram_1.jpeg)

Figure 1: Comparison of average latency per batch and classification accuracy between the Original TTA and Sparse TTA approaches on edge devices processing an online data stream. With an adaptation rate of 0.33, adaptation occurs once every three batches, reducing latency relative to the adaptation rate but leading to a significant accuracy drop than fully adapting original TTA.

effectiveness of STTA hinges on selecting proper samples from a large pool, ensuring that the model maintains adequate performance with fewer updates (detailed analysis in Section [4\)](#page-6-0).

Conventional TTA approaches that adopt sampling strategies are designed for non-i.i.d data [\(Gong](#page-10-0) [et al., 2022;](#page-10-0) [Niu et al., 2023;](#page-11-1) [Yuan et al., 2023\)](#page-12-3) or noisy data [\(Gong et al., 2023\)](#page-10-1). They do not aim for data efficiency and thus yield high sample usage for updates. While EATA [\(Niu et al., 2022\)](#page-11-0) excludes unreliable samples and utilizes fewer samples, it suffers from performance degradation when attempting more aggressive reductions. Data-efficient deep learning demonstrated that selecting easy, class-representative samples is effective when the sampling ratio is low (e.g., below 0.4) [\(Xia](#page-12-5) [et al., 2022;](#page-12-5) [Choi et al., 2024\)](#page-10-2). However, these methods rely on ground-truth label information, which is typically unavailable in TTA scenarios.

We propose SNAP-TTA: Sparse Network Adaptation for Practical Test-Time Adaptation, a lowlatency TTA framework designed for resource-constrained devices. SNAP-TTA addresses the challenge of balancing adaptation accuracy with computational efficiency in STTA, where only a small subset of data is used for updates. To that end, SNAP-TTA has two key technical enablers: First, it introduces a sampling strategy that combines *class-representative* and *domain-representative* samples. This approach enables the model to adapt effectively to domain shifts even with minimal data. Class and Domain Representative Memory (CnDRM) selects these critical samples by using pseudo-label confidence in a prediction-balanced manner for class-representative samples, and by identifying the domain-representative samples closest to the center of the target domain's feature embedding (Section [3.1\)](#page-3-0). Second, Inference-only Batch-aware Memory Normalization (IoBMN) refines the normalization process during inference by utilizing CnDRM's class-domain representative statistics, leveraging the representativeness of these selected samples to correct skewed feature distributions at each inference step. This ensures that the model effectively adapts to domain shifts without back-propagation, maintaining alignment with the evolving data distribution (Section [3.2\)](#page-5-0). These two components are integrated to perform adaptation, minimizing accuracy drop and latency in real-world domain-shifted scenarios.

SNAP-TTA is designed to work together with existing TTA methods orthogonally; thus, we evaluated SNAP-TTA integrated with existing SOTA TTA algorithms under diverse adaptation rates. Specifically, we evaluated SNAP-TTA with five SOTA TTA algorithms (Tent[\(Wang et al., 2021\)](#page-12-1), EATA[\(Niu et al., 2022\)](#page-11-0), SAR[\(Niu et al., 2023\)](#page-11-1),CoTTA[\(Wang et al., 2022\)](#page-12-2), and RoTTA[\(Yuan et al.,](#page-12-3) [2023\)](#page-12-3)) on three common TTA benchmarks (CIFAR10-C, CIFAR100-C [\(Hendrycks & Dietterich,](#page-10-3) [2019a\)](#page-10-3), and ImageNet-C [\(Hendrycks & Dietterich, 2019b\)](#page-10-4)). SNAP-TTA effectively reduces latency while minimizing performance drops in existing TTA methods. For instance, on our implementation in Raspberry Pi 4[\(Raspberry Pi Foundation, 2019\)](#page-12-6) testbed, SNAP-TTA achieved up to 87.5% latency reduction at an adaptation rate of 0.1. In CIFAR10-C, SNAP-TTA-integrated methods consistently outperformed their original counterparts, showing up to 13.38% accuracy gain for CoTTA at an adaptation rate of 0.01. In addition, SNAP-TTA integration performed comparable accuracy to the original TTA methods under full adaptation settings. For instance, it achieved 77.12%∼81.74% accuracy for Tent at various adaptation rates, whereas the full adaptation accuracy was 80.43% in CIFAR10-C.

**114 115**

**117**

**119**

**127**

**129 130**

**134**

**136**

### 2 PRELIMINARIES

We focus on the Test-Time Adaptation (TTA) latency challenges specific to edge devices, highlighting the constraints of adapting models in real-time environments with limited resources. Detailed related works are in Appendix [A.](#page-14-0)

Test-Time Adaptation and Its Latency Challenge on Edge Devices. In unsupervised domain adaptation, the source domain data D<sup>S</sup> = X <sup>S</sup> , Y is drawn from the distribution P<sup>S</sup> (x, y), while the target domain data D<sup>T</sup> = X T , Y follows P<sup>T</sup> (x, y), typically without known labels y<sup>j</sup> . Given a pre-trained model f(·; Θ) on the source domain D<sup>S</sup> , test-time adaptation (TTA) [\(Wang et al.,](#page-12-1) [2021\)](#page-12-1) adjusts the model to the target distribution P<sup>T</sup> using only target instances x<sup>j</sup> , updating the parameters Θ to reduce domain discrepancy.

When applied to resource-constrained devices, however, current TTA approaches face significant latency challenges. In real-time applications that require rapid inference, online TTA becomes impractical due to the need for adaptation at every batch (Figure [4,](#page-8-0) detailed latency tracking reported in Appendix [E.3\)](#page-23-0). Our experiment on Raspberry Pi 4 [\(Raspberry Pi Foundation, 2019\)](#page-12-6) showed a minimum of 3.83 seconds latency per batch for existing TTA methods. This indicates existing methods could not handle real-time applications with fast data streams and strict latency requirements, such as autonomous driving [\(Tampuu et al., 2024;](#page-12-7) [Liu et al., 2023\)](#page-11-2). TTA methods such as CoTTA use computationally intensive operations such as data augmentations and ensemble models at the cost of increased latency. Relatively lightweight algorithms incur non-negligible latency from adaptation processes such as backpropagation, which becomes bottlenecks in resource-constrained devices without the parallel processing capabilities and memory bandwidth of GPUs.

A recent work [\(Alfarra et al., 2024\)](#page-10-5), recognizing latency as a problem, proposed a TTA evaluation protocol that penalizes methods that are slower than the data stream rate. Instead of penalizing a model for being slow, we utilize Sparse TTA, where the model actively chooses to adapt at sparse intervals for the goal of maintaining a real-time inference rate. As real deployments involve devices with different computational capabilities and data streams of varying speeds, we believe a framework that effectively maintains various TTA methods' performance across different latency requirements is crucial.

Sparse Test-Time Adaptation and Adaptation rates. Sparse Test-Time Adaptation (STTA) aims to efficiently adapt models by reducing both the frequency of updates and the number of samples used per update, which is essential for minimizing latency in edge devices. The concept of adaptation rate plays a central role in STTA, as it controls both the update frequency and the number of data points used. Unlike Original Test-Time Adaptation (TTA), which uses full batches of data and can create significant computational overhead, STTA employs an adaptation rate to limit updates and data usage proportionally, thus introducing sparsity (Figure [1\)](#page-1-0).

By adjusting the *adaptation rate*, STTA can minimize latency and computational costs while maintaining adaptation performance. This rate defines how sparsely updates occur and the proportion of samples used for updates compared to the Original TTA, enabling efficient model adjustments to distribution shifts. The balance between adaptation accuracy and computational efficiency makes STTA particularly suitable for environments that demand both quick responses and minimal resource usage.

# 3 METHODOLOGY

SNAP-TTA framework resolves the high latency and inefficiency issue of existing Test-Time Adaptation (TTA) methods. By introducing a Sparse TTA (STTA) strategy combined with a novel sampling method, SNAP-TTA minimizes adaptation delays while maintaining accuracy. The overall system, illustrated in Figure [2,](#page-3-1) consists of two primary components: (i) Class and Domain Representative Memory (CnDRM) for efficient sampling and (ii) Inference-only Batch-aware Memory Normalization (IoBMN) to correct feature distribution shifts during inference. Together, these components enable effective STTA with minimal computational overhead.

**166 167**

**169**

**171**

**204**

**206**

![](_page_3_Diagram_1.jpeg)

Figure 2: Design overview of SNAP-TTA. The framework consists of two primary components: (a) Class and Domain Representative Memory (CnDRM), which efficiently selects representative samples to minimize adaptation overhead, and (b) Inference-only Batch-aware Memory Normalization (IoBMN), which corrects feature distribution shifts during inference. Together, these components implement the Sparse TTA (STTA) strategy, reducing latency while maintaining model accuracy.

#### 3.1 CLASS AND DOMAIN REPRESENTATIVE MEMORY (CNDRM)

CnDRM is a core component of SNAP-TTA that addresses the challenges of efficient data sampling for STTA. In STTA, the adaptation rate directly impacts the number of samples used, necessitating a careful sampling strategy to optimize performance with minimal data. Given this limited sampling ratio, CnDRM selects both class and domain-representative samples to maintain model performance while minimizing adaptation overhead.

Motivation. Data sampling is crucial in data-efficient deep learning, especially when working with a limited number of samples. In high data sampling ratio scenarios, score-based methods prioritize difficult or rare samples, often achieving performance comparable to full-dataset training. However, when the sampling ratio is low, selecting easy and class-representative samples becomes more effective [\(Choi et al., 2024\)](#page-10-2). This method selects samples that minimize differences in loss gradients or curvature, ensuring that the generalizability is retained even with fewer samples. Similarly, the Moderate Coreset [\(Xia et al., 2022\)](#page-12-5) paper demonstrates that at low sampling ratios of 0.2 to 0.4, the distance from the class center significantly impacts performance, with samples closer to the center being particularly effective in scenarios with high label noise. In the STTA setting, where ground truth labels are unavailable and the probability of incorrect predictions is high, selecting representative samples based on potentially incorrect predictions resembles a high label noise situation. Therefore, selecting class-representative easy samples could provide some benefit to STTA.

However, if the model must perform STTA at an even lower adaptation rate (e.g., 0.1) due to the latency limits, selecting class-representative samples alone would be insufficient (Table [4\)](#page-8-1). Unlike traditional classification tasks, STTA is an unsupervised domain adaptation, which requires identifying target domain-representative samples that reflect the distributional shift between the source and target domains. In these cases, we argue that focusing on domain-representative instances is just as crucial, as selecting samples that best capture the domain shift can help the model retain generalizability with minimal data. Therefore, selecting both class-representative and domainrepresentative samples could enhance STTA performance in low-data environments, where each sample must contribute significantly to model adaptation.

Critera 1: Class Representation. CnDRM selects samples with higher confidence scores to avoid the issues caused by low-confidence samples. Low-confidence samples are typically located near decision boundaries and are more likely to carry incorrect pseudo-labels. This strategy ensures that the adaptation process is guided by stable learning signals, which is important in the absence of ground-truth labels. By focusing on high-confidence samples, CnDRM mitigates the risk of propagating errors resulting from incorrect pseudo-labels, thereby supporting more effective and stable adaptation (Details in Appendix [E.2\)](#page-23-1). The confidence score C(x) for each sample x is calculated as: C(x) = maxy∈Y p(y|x; Θ) where p(y|x; Θ) is the softmax probability for class y. Only samples with confidence above a predefined threshold τconf are retained. For a balanced representa-

**224**

**236 237**

**254**

**256**

**259**

**269**

tion across diverse classes, CnDRM selects these high-confidence samples in a prediction-balanced manner. This balance helps maintain the model's overall classification capability and prevents bias towards certain classes when only a low sample ratio is available for adaptation. By leveraging both high confidence and prediction balance, CnDRM effectively selects class-representative samples that are diverse and reliable, even without access to ground-truth labels.

> KDE 66.08

![](_page_4_Figure_2.jpeg)

Figure 3: Samping visualization and accuracy comparison between the closest 20% and farthest 20% samples from the domain centroid (based on Wasserstein distance) on ImageNet-C Gaussian noise.

*the domain centroid* would enhance performance in STTA. Our preliminary experiment results validate improved performance when selecting samples near the centroid (Figure [3\)](#page-4-0).

For ImageNet-C Gaussian noise, TTA with the closest 20% of samples achieved 26.65% accuracy,

whereas the farthest 20% showed a lower accuracy of 18.52%.

As early layers in deep learning models tend to retain domain-specific features [\(Zeiler & Fergus,](#page-12-8) [2014;](#page-12-8) [Lee et al., 2018;](#page-11-3) [Segu et al., 2023\)](#page-12-9), we utilize the hidden features of early layers to identify domain-representative samples (Appendix [E.1\)](#page-23-2). We use the feature statistics (mean and variance) of the first normalization layer to evaluate domain representation. This choice is made as domain discrepancies can be effectively reduced through normalization adjustments [\(Nado et al., 2020;](#page-11-4) [Schnei](#page-12-10)[der et al., 2020\)](#page-12-10). Domain discrepancies in hidden features are substantially reduced after passing through a single normalization layer, significantly minimizing domain shift differences [\(Li et al.,](#page-11-5) [2016\)](#page-11-5). While deeper layers provide detailed information, using the first layer balances capturing domain-specific information and maintaining computational efficiency.

The domain centroid cdomain is computed using a momentum-based update of batch statistics from the normalization layer: µdomain ← (1 − β)µdomain + βµ<sup>t</sup> and σ 2 domain ← (1 − β)σ 2 domain + βσ<sup>2</sup> t , where µ<sup>t</sup> and σ 2 t are the mean and variance of the current batch t, and β is the momentum parameter. In our preliminary study, we found that using only the mean and standard deviation values before the first normalization was sufficient to calculate the domain centroid. The sampled instances effectively represented the domain and were correctly positioned in the embedding space for each criterion (Figure [3\)](#page-4-0).

To determine domain-representative samples, CnDRM calculates the Wasserstein distance between each sample's feature statistics and the domain centroid. The Wasserstein distance measures the similarity between two distributions by considering their mean and variance, evaluating how well a sample represents the domain. It is useful for capturing domain characteristics, leading to its wide use in domain generalization [\(Segu et al., 2023\)](#page-12-9). For each sample xt, the feature statistics (µ<sup>x</sup><sup>t</sup> , σ<sup>x</sup><sup>t</sup> ) are taken from the input to the normalization layer, and the Wasserstein distance W(xt, cdomain) is given by:

$$W(\mathbf{x}_t, \mathbf{c}_{domain}) = \sqrt{(\mu_{\mathbf{x}_t} - \mu_{domain})^2 + (\sigma_{\mathbf{x}_t} - \sigma_{domain})^2}. \quad (1)$$

Memory Management Algorithm. The memory management in CnDRM maintains efficiency without introducing additional overhead. To achieve this, the memory size is kept equal to the batch size for minimal resource usage. Within this fixed memory, samples are managed by balancing the number of samples per class based on predictions so that each class remains well-represented. For domain adaptation, samples in memory are periodically replaced with new samples that are closer to the domain centroid and meet the confidence threshold to retain only the most class-domain representative samples. Algorithm [1](#page-5-1) has details.

**289 290 291**

**294**

**301**

**304**

**306**

**309**

**314 315**

**318 319**

**321**

Algorithm 1 Class and Domain Representative Memory (CnDRM)

Require: test data stream xt, memory M with capacity N, confidence threshold τconf , sample unit for memory s, adaptation rate 1/k

1: for batch b ∈ {1, . . . , B} do 2: Yˆ <sup>b</sup> ← f(b; Θ) 3: for each sample x<sup>t</sup> in batch b do 4: yˆ<sup>t</sup> ← Yˆ <sup>b</sup>[t] 5: confidence ← C(xt; Θ) 6: ct(µ<sup>x</sup><sup>t</sup> , σ<sup>x</sup><sup>t</sup> ) ← mean and variance of early hidden feature 7: w<sup>x</sup><sup>t</sup> ← W(xt, cdomain) 8: if confidence > τconf then ▷ Class-representative samples 9: Add st(xt, yˆt, ct, w<sup>x</sup><sup>t</sup> ) to M ▷ Add samples in prediction-balanced manner 10: if |M| > N then 11: L <sup>∗</sup> ← class with most samples in M 12: if yˆ<sup>t</sup> ∈/ L ∗ then ▷ Removes domain-centroid farthest sample 13: smax dist ← arg max<sup>s</sup>i∈M∧yˆi∈L<sup>∗</sup> w<sup>x</sup><sup>i</sup> 14: else 15: smax dist ← arg max<sup>s</sup>i∈M∧yˆi=ˆy<sup>t</sup> w<sup>x</sup><sup>i</sup> 16: Remove smax dist from M 17: cdomain ← (1 − β)cdomain + βc<sup>t</sup> ▷ Update domain-centroid 18: Recalculate w<sup>s</sup><sup>i</sup> for all s<sup>i</sup> in M 19: if b mod k == 0 then ▷ Adaptation occurs every k batches 20: Update model Θ using samples in M

### 3.2 INFERENCE-ONLY BATCH-AWARE MEMORY NORMALIZATION (IOBMN)

Motivation. In Sparse Test-Time Adaptation (STTA) scenarios, models must adapt to domain shifts despite having limited opportunities for updates. In this setting, maintaining robust performance becomes challenging as the stored memory statistics, derived from representative adaptation batches, may not fully align with subsequent inference batches, especially when updates are skipped. This can lead to a potential mismatch between the stored statistics and the current data distribution. Traditional normalization methods, which solely rely on test batches' statistics, struggle to address these subtle shifts effectively. To tackle this issue, we introduce the Inference-only Batch-aware Memory Normalization (IoBMN) module, which leverages the robustness of class-domain representative statistics while dynamically adjusting for mismatches that arise in skipped batches. By primarily basing normalization on stable, representative memory statistics and selectively adapting with recent inference data, IoBMN efficiently corrects for distributional shifts, ensuring both robustness and adaptability in STTA conditions. This approach significantly enhances model stability in sparse adaptation scenarios, as shown in our ablation study in Section [4.](#page-6-0)

Approach. Given a feature map f ∈ R <sup>B</sup>×C×<sup>L</sup>, where B is the batch size, C is the number of channels, and L is the number of spatial locations, the batch-wise statistics µ¯<sup>c</sup> and σ¯ 2 c for the c-th channel are calculated as follows:

$$\bar{\mu}_c = \frac{1}{B \times L} \sum_{b=1}^B \sum_{l=1}^L f_{b,c,l}, \quad \sigma_c^2 = \frac{1}{B \times L} \sum_{b=1}^B \sum_{l=1}^L (f_{b,c,l} - \mu_{b,c}), \quad (2)$$

where µ¯<sup>m</sup> and σ¯ <sup>m</sup> are calculated from the most recent adapted CnDRM samples in the same way with Equation [2,](#page-5-2) using the memory capacity M with m representing the memory. We assume that µ<sup>m</sup> and σ 2 <sup>m</sup> follow the *sampling distribution* of the feature map size L and memory capacity M. The corresponding variances for the memory mean µ<sup>m</sup> and variance σ 2 <sup>m</sup> are calculated as:

$$s_{\mu_m}^2 := \frac{\bar{\sigma}_m^2}{C \times M}, \quad s_{\sigma_m^2}^2 := \frac{2\bar{\sigma}_m^4}{C \times M - 1}. \quad (3)$$

For the normalization process to adapt efficiently to the current inference batch statistics, IoBMN corrects (¯µm, σ¯ 2 <sup>m</sup>) only when µ¯<sup>c</sup> (and σ¯ 2 c ) significantly differ from µ¯<sup>m</sup> (and σ¯ 2 m) through soft shrinkage function:

$$\mu_m^{\text{IoBMN}} = \bar{\mu}_m + S_\lambda(\bar{\mu}_c - \bar{\mu}_m; \alpha s_{\mu_m}), \quad (\sigma_m^{\text{IoBMN}})^2 = \bar{\sigma}_m^2 + S_\lambda(\bar{\sigma}_c^2 - \bar{\sigma}_m^2; \alpha s_{\sigma_m}), \quad (4)$$

**329**

**334**

**354 355 356**

**358 359**

**361**

**364**

**369**

where α ≥ 0 in IoBMN controls the reliance on the normalization layer statistics. A larger α gives more weight to the last adapted memory normalization statistics, whereas a smaller α emphasizes the current inference batch normalization statistics. The soft shrinkage function Sλ(x; λ) is defined as:

$$S_\lambda(x; \lambda) = \begin{cases} x - \lambda & \text{if } x > \lambda, \\ x + \lambda & \text{if } x < -\lambda, \text{ and} \\ 0 & \text{otherwise,} \end{cases}$$

where λ is the threshold, s is a scaling factor, and x is the input. The function allows for proportional adjustments based on the magnitude of the values, where smaller values are adjusted less and larger values more, preserving the critical information inherent in the adapted memory normalization statistics.

Finally, the output of the IoBMN for each feature fb,c,l is computed as:

$$\text{IoBMN}(f_{b,c,l}; \bar{\mu}_m, \bar{\sigma}_m^2, \mu_m^{\text{IoBMN}}, (\sigma_m^{\text{IoBMN}})^2) := \gamma \cdot \frac{f_{b,c,l} - \mu_m^{\text{IoBMN}}}{\sqrt{(\sigma_m^{\text{IoBMN}})^2 + \epsilon}} + \beta, \quad (5)$$

where γ and β are learnable affine parameters of normalization layer, and ϵ is a small constant added for numerical stability. In our experiments, we chose α = 4 to effectively handle various out-ofdistribution scenarios. The parameter s is a hyperparameter that determines the degree of adjustment desired and can be tuned based on specific requirements.

IoBMN utilizes CnDRM's class-domain representative statistics and adjusts them based on the current inferencing batch statistics. This dual-statistic approach allows IoBMN to correct the outdated and skewed distribution of the memory, ensuring alignment with the data distribution at each inference point. By leveraging the statistics of the data used during model update points, IoBMN adapts effectively without significant computational overhead. Additionally, this method mitigates the performance degradation caused by the prolonged intervals between adaptations so that the model remains well-aligned with the evolving data distribution.

# 4 EXPERIMENTS

This section outlines our experimental setup and presents the results obtained under various STTA settings. Refer to Appendix [B](#page-14-1) for further details.

Scenario. We examined how different adaptation rates affect performance to simulate a scenario requiring a certain latency threshold for latency-sensitive applications. We varied the *adaptation rate* to observe its impact on both model accuracy and latency. The main evaluation was run with diverse adaptation rates (0.01, 0.03, 0.05, 0.1, 0.3, and 0.5). We report the average accuracy and standard deviation from three random seeds. Latency measurement was done on our Raspberry Pi 4 [\(Raspberry Pi Foundation, 2019\)](#page-12-6) testbed.

Dataset and Model. We used three standard TTA benchmarks: CIFAR10-C, CIFAR100- C [\(Hendrycks & Dietterich, 2019a\)](#page-10-3) and ImageNet-C [\(Hendrycks & Dietterich, 2019b\)](#page-10-4). These datasets include 15 different types of corruption with five levels of severity, and we used the highest one. CIFAR10-C/CIFAR100-C has 10,000 test data with 10/100 classes, and ImageNet-C has 50,000 test data with 1,000 classes for each corruption. We employed ResNet18 [\(He et al., 2016\)](#page-10-6) as the backbone network, utilizing models pre-trained on CIFAR10 and CIFAR100 [\(Krizhevsky &](#page-11-6) [Hinton, 2009\)](#page-11-6). We also use ResNet50 [\(He et al., 2016\)](#page-10-6) and ViT [\(Dosovitskiy, 2020\)](#page-10-7) pre-trained on ImageNet [\(Deng et al., 2009\)](#page-10-8) from the TorchVision [\(maintainers & contributors, 2016\)](#page-11-7) library.

Baselines. SNAP-TTA is designed to integrate with existing TTA algorithms. Therefore, testing existing *TTA algorithms under different adaptation rates* serves as our baseline (implementation details including hyperparameters are in Appendix [B.1\)](#page-15-0). We selected five SOTA TTA algorithms: (i) Tent [\(Wang et al., 2021\)](#page-12-1) updates only BN affine parameters, (ii) CoTTA [\(Wang et al., 2022\)](#page-12-2) updates the entire model parameters using a teacher-student framework, (iii) EATA [\(Niu et al.,](#page-11-0) [2022\)](#page-11-0), (iv) SAR[\(Niu et al., 2023\)](#page-11-1), and (v) RoTTA[\(Yuan et al., 2023\)](#page-12-3). For efficiency evaluation, we compared our method against BN stats [\(Nado et al., 2020;](#page-11-4) [Schneider et al., 2020\)](#page-12-10).

**381**

**384**

**386**

Table 1: STTA classification accuracy (%) and latency per batch (s) comparing with and without SNAP-TTA on ImageNet-C through Adaptation Rates (AR) (0.3, 0.1, and 0.05).AR is the ratio of the number of backpropagation occurrences to the total, and thus represents the reduction in adaptation latency compared to full adaptation (AR=1). More results on diverse AR (0.5, 0.03 and 0.01) are on Appendix [C.1.](#page-16-0) Bold numbers are the highest accuracy.

| AR Methods | Gau.  | Shot  | Imp.  | Def.  | Gla.  | Mot.  | Zoom  | Snow  | Fro.  | Fog   | Brit. | Cont. | Elas. | Pix.  | JPEG  | Avg.  | Lat.   |
|------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| Source     | 3.00  | 3.70  | 2.64  | 17.90 | 9.74  | 14.72 | 22.45 | 16.60 | 23.06 | 24.00 | 59.11 | 5.37  | 16.50 | 20.88 | 32.63 | 18.15 | 16.60  |
| BN stats   | 14.29 | 15.06 | 14.89 | 13.30 | 13.38 | 23.78 | 35.22 | 31.78 | 30.26 | 44.40 | 62.39 | 15.14 | 40.42 | 45.25 | 36.53 | 29.00 | 17.36  |
| Tent       | 27.03 | 28.98 | 28.64 | 24.66 | 23.63 | 38.70 | 45.77 | 44.82 | 38.06 | 54.59 | 64.61 | 16.84 | 51.64 | 55.54 | 49.38 | 39.53 | 38.33  |
| CoTTA      | 13.12 | 13.98 | 13.94 | 12.44 | 12.18 | 23.74 | 35.22 | 31.78 | 30.26 | 44.40 | 62.40 | 15.13 | 40.42 | 45.26 | 36.53 | 28.72 | 300.23 |
| EATA       | 29.62 | 31.79 | 31.17 | 26.89 | 26.30 | 40.65 | 47.44 | 46.29 | 40.78 | 55.57 | 64.97 | 38.02 | 52.66 | 56.03 | 50.26 | 42.56 | 31.98  |
| SAR        | 17.49 | 22.04 | 21.21 | 11.62 | 12.60 | 39.76 | 44.13 | 45.98 | 29.39 | 55.13 | 63.71 | 17.34 | 52.31 | 56.09 | 49.35 | 35.21 | 78.15  |
| RoTTA      | 20.60 | 22.83 | 19.81 | 10.46 | 10.10 | 21.31 | 31.83 | 39.66 | 32.09 | 46.08 | 62.22 | 20.27 | 42.54 | 47.47 | 40.67 | 31.20 | 87.00  |
| Tent       | 23.63 | 25.18 | 24.80 | 21.81 | 20.97 | 34.11 | 43.60 | 41.44 | 36.98 | 52.66 | 64.21 | 22.74 | 48.96 | 53.46 | 46.80 | 37.42 | 27.34  |
| + SNAP     | 26.60 | 28.21 | 27.94 | 24.37 | 22.39 | 36.45 | 44.36 | 42.64 | 38.54 | 52.91 | 64.26 | 33.47 | 48.58 | 53.90 | 47.41 | 39.47 | 28.84  |
| CoTTA      | 11.74 | 12.74 | 12.68 | 11.77 | 11.62 | 22.64 | 34.97 | 31.05 | 29.81 | 44.24 | 62.12 | 13.73 | 40.31 | 45.19 | 36.71 | 28.09 | 205.22 |
| + SNAP     | 15.26 | 16.00 | 15.83 | 13.81 | 14.13 | 24.84 | 36.46 | 32.58 | 31.73 | 46.04 | 63.52 | 15.69 | 42.18 | 46.74 | 38.00 | 30.19 | 208.10 |
| EATA       | 27.35 | 29.03 | 28.62 | 23.94 | 23.45 | 37.21 | 46.18 | 44.05 | 39.19 | 54.52 | 64.54 | 32.20 | 51.22 | 55.00 | 49.27 | 40.38 | 20.27  |
| + SNAP     | 29.48 | 31.20 | 30.69 | 26.68 | 25.90 | 38.24 | 46.60 | 44.62 | 39.31 | 54.82 | 64.44 | 32.87 | 51.41 | 55.41 | 49.78 | 41.43 | 22.16  |
| SAR        | 28.12 | 29.30 | 29.63 | 22.37 | 23.88 | 39.34 | 45.36 | 45.69 | 36.73 | 54.91 | 64.11 | 10.96 | 52.22 | 55.76 | 49.60 | 39.20 | 36.44  |
| + SNAP     | 32.63 | 34.69 | 34.26 | 28.91 | 27.96 | 43.51 | 47.79 | 48.27 | 42.41 | 56.45 | 64.77 | 32.76 | 53.74 | 57.21 | 51.67 | 43.80 | 38.01  |
| RoTTA      | 16.90 | 17.88 | 17.25 | 12.89 | 12.51 | 23.96 | 35.26 | 36.26 | 32.32 | 47.25 | 63.98 | 17.46 | 42.77 | 48.21 | 39.35 | 30.95 | 59.32  |
| + SNAP     | 18.63 | 19.94 | 19.35 | 14.88 | 14.34 | 25.88 | 36.47 | 37.13 | 33.32 | 47.74 | 63.96 | 19.08 | 42.98 | 48.73 | 40.27 | 32.18 | 60.31  |
| Tent       | 22.00 | 23.51 | 23.07 | 19.38 | 18.86 | 32.15 | 42.29 | 39.70 | 34.33 | 51.62 | 63.70 | 15.79 | 47.74 | 52.35 | 45.54 | 35.47 | 18.01  |
| + SNAP     | 26.21 | 27.85 | 27.50 | 23.62 | 22.73 | 36.01 | 44.11 | 42.19 | 38.15 | 52.95 | 64.57 | 30.23 | 48.56 | 53.71 | 47.09 | 39.03 | 18.76  |
| CoTTA      | 10.97 | 11.92 | 11.98 | 11.45 | 11.38 | 22.39 | 34.96 | 30.88 | 29.89 | 44.09 | 61.96 | 13.08 | 40.20 | 45.27 | 36.71 | 27.81 | 161.98 |
| + SNAP     | 15.13 | 16.03 | 15.91 | 13.86 | 14.02 | 24.90 | 36.51 | 32.56 | 31.81 | 46.02 | 63.60 | 15.69 | 41.94 | 46.78 | 38.03 | 30.19 | 163.24 |
| EATA       | 22.43 | 23.78 | 23.26 | 19.38 | 19.42 | 32.18 | 43.22 | 40.65 | 36.64 | 52.38 | 63.87 | 24.59 | 48.13 | 52.89 | 46.33 | 36.61 | 16.00  |
| + SNAP     | 26.10 | 27.29 | 27.13 | 22.38 | 22.15 | 33.45 | 43.92 | 40.96 | 36.68 | 52.71 | 63.77 | 27.93 | 48.47 | 53.23 | 47.46 | 38.24 | 17.45  |
| SAR        | 26.12 | 27.56 | 26.93 | 22.51 | 23.35 | 36.03 | 44.48 | 43.19 | 37.26 | 53.82 | 64.15 | 19.87 | 50.78 | 54.78 | 48.43 | 38.62 | 21.39  |
| + SNAP     | 30.28 | 31.97 | 31.30 | 26.67 | 26.31 | 39.66 | 46.08 | 45.43 | 40.26 | 54.76 | 64.62 | 36.12 | 51.26 | 55.42 | 49.63 | 41.99 | 23.99  |
| RoTTA      | 14.77 | 15.59 | 15.33 | 13.17 | 13.19 | 23.85 | 35.38 | 32.73 | 30.77 | 45.22 | 63.08 | 15.62 | 41.05 | 46.15 | 37.19 | 29.54 | 45.98  |
| + SNAP     | 15.35 | 16.20 | 16.01 | 13.67 | 13.66 | 24.27 | 35.62 | 33.04 | 31.02 | 45.38 | 62.95 | 15.96 | 41.06 | 46.17 | 37.44 | 29.85 | 47.47  |
| Tent       | 23.77 | 24.65 | 24.44 | 20.54 | 20.27 | 32.73 | 43.57 | 40.82 | 35.92 | 52.78 | 63.82 | 15.95 | 49.33 | 53.46 | 47.19 | 36.62 | 16.93  |
| + SNAP     | 29.12 | 30.46 | 30.30 | 25.77 | 25.22 | 38.21 | 46.14 | 44.29 | 39.95 | 54.65 | 65.47 | 33.81 | 50.83 | 55.59 | 49.21 | 41.27 | 17.55  |
| CoTTA      | 11.03 | 11.91 | 11.75 | 11.03 | 11.20 | 22.30 | 34.98 | 30.87 | 29.78 | 43.99 | 61.87 | 12.92 | 40.26 | 45.23 | 36.63 | 27.72 | 152.94 |
| + SNAP     | 15.22 | 15.97 | 15.93 | 13.91 | 14.05 | 24.87 | 36.48 | 32.60 | 31.65 | 46.09 | 63.59 | 15.67 | 42.00 | 46.71 | 37.96 | 30.18 | 153.34 |
| EATA       | 19.53 | 20.65 | 20.72 | 16.74 | 16.96 | 29.11 | 41.22 | 37.96 | 34.84 | 50.75 | 63.29 | 19.86 | 45.92 | 51.15 | 44.13 | 34.19 | 15.82  |
| + SNAP     | 22.83 | 23.95 | 23.62 | 19.43 | 19.70 | 30.34 | 41.59 | 38.06 | 35.06 | 50.98 | 63.30 | 23.72 | 46.26 | 51.52 | 45.46 | 35.72 | 16.44  |
| SAR        | 23.25 | 24.23 | 23.66 | 19.98 | 20.38 | 33.05 | 43.04 | 40.73 | 36.06 | 52.61 | 64.09 | 20.17 | 49.00 | 53.35 | 46.73 | 36.69 | 19.98  |
| + SNAP     | 27.54 | 29.03 | 28.66 | 24.05 | 23.42 | 36.28 | 44.12 | 42.89 | 38.54 | 53.24 | 64.25 | 31.83 | 48.79 | 54.04 | 47.80 | 39.63 | 20.94  |
| RoTTA      | 14.42 | 15.22 | 15.02 | 13.25 | 13.31 | 23.79 | 35.27 | 32.09 | 30.43 | 44.71 | 62.64 | 15.24 | 40.63 | 45.55 | 36.75 | 29.22 | 43.32  |
| + SNAP     | 14.65 | 15.48 | 15.29 | 13.43 | 13.45 | 23.93 | 35.33 | 32.18 | 30.53 | 44.71 | 62.58 | 15.41 | 40.64 | 45.55 | 36.81 | 29.33 | 44.71  |

Table 2: STTA classification accuracy (%) and latency per batch (s) comparing with and without SNAP-TTA on CIFAR10/100-C at Adaptation Rate 0.1. Numbers in parentheses represent the performance difference of SNAP-TTA compared to full adaptation Bold numbers are the highest accuracy. More results on other adaptation rates are in Appendix [C.2](#page-18-0) and [C.3.](#page-20-0)

| Methods | Gau.  | Shot  | Imp.  | Def.  | Gla.  | Mot.  | Zoom  | Snow  | Fro.  | Fog CIFAR10-C | Brit. | Cont. | Elas. | Pix.  | JPEG  |              | Avg.     |      | Lat.      |
|---------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------------|-------|-------|-------|-------|-------|--------------|----------|------|-----------|
| Tent    | 67.32 | 69.39 | 60.69 | 85.34 | 63.82 | 83.52 | 84.70 | 79.68 | 77.79 | 83.75         | 88.53 | 83.12 | 75.18 | 77.82 | 71.47 | 76.81        | (-3.62)  | 2.80 | (-29.47%) |
| + SNAP  | 70.22 | 71.48 | 63.08 | 87.35 | 65.74 | 85.89 | 86.38 | 81.93 | 80.00 | 85.62         | 90.34 | 87.47 | 76.44 | 79.63 | 72.72 | 78.95        | (1.48)   | 3.08 | (-22.42%) |
| CoTTA   | 59.11 | 60.26 | 56.07 | 72.23 | 56.77 | 73.55 | 72.20 | 68.05 | 66.68 | 72.88         | 77.66 | 65.95 | 65.67 | 64.12 | 65.16 | 66.42        | (-11.58) | 4.92 | (-93.14%) |
| + SNAP  | 71.70 | 73.54 | 66.70 | 85.16 | 66.83 | 84.30 | 84.88 | 81.02 | 80.61 | 84.20         | 89.84 | 81.71 | 76.60 | 79.66 | 75.71 | 78.83        | (+0.83)  | 4.93 | (-93.12%) |
| EATA    | 66.65 | 68.96 | 59.73 | 84.93 | 63.26 | 83.10 | 84.53 | 79.28 | 77.46 | 83.48         | 88.12 | 82.46 | 74.49 | 77.48 | 70.43 | 76.29        | (-5.27)  | 2.52 | (-35.88%) |
| + SNAP  | 69.29 | 70.49 | 61.71 | 87.32 | 65.48 | 85.96 | 86.64 | 81.44 | 79.56 | 85.47         | 90.50 | 86.84 | 76.32 | 79.64 | 72.51 | 78.61        | (-2.95)  | 2.87 | (-26.97%) |
| SAR     | 66.11 | 68.18 | 59.15 | 84.91 | 62.87 | 82.33 | 84.27 | 79.23 | 77.58 | 83.21         | 88.29 | 82.60 | 74.65 | 75.92 | 70.79 | 76.01        | (-3.04)  | 2.85 | (-50.43%) |
| + SNAP  | 67.76 | 70.68 | 60.82 | 86.78 | 64.73 | 85.29 | 86.22 | 80.82 | 79.30 | 84.95         | 91.33 | 86.59 | 75.72 | 78.72 | 71.24 | 78.06(-0.99) |          | 2.98 | (-48.17%) |
| RoTTA   | 63.12 | 64.84 | 56.72 | 84.49 | 62.15 | 82.53 | 83.84 | 78.03 | 76.13 | 82.88         | 87.48 | 81.49 | 73.75 | 76.04 | 68.24 | 74.78        | (-2.22)  | 2.91 | (-50.93%) |
| + SNAP  | 65.35 | 66.99 | 58.09 | 86.77 | 63.63 | 85.47 | 86.01 | 80.54 | 78.38 | 84.99         | 90.00 | 85.99 | 75.67 | 78.14 | 70.09 | 77.07        | (+0.07)  | 2.94 | (-50.42%) |
| Tent    | 43.55 | 44.25 | 37.95 | 62.56 | 41.80 | 59.45 | 62.13 | 53.04 | 51.60 | 56.76         | 64.60 | 61.19 | 51.01 | 56.42 | 46.28 | 52.84        | (-2.92)  | 3.34 | (-27.49%) |
| + SNAP  | 46.51 | 47.68 | 39.92 | 65.39 | 44.14 | 63.29 | 64.53 | 55.20 | 55.55 | 59.71         | 68.05 | 64.90 | 53.91 | 59.28 | 49.58 | 55.84        | (+0.08)  | 3.67 | (-19.17%) |
| CoTTA   | 28.53 | 29.53 | 26.45 | 42.19 | 30.34 | 44.69 | 41.88 | 34.44 | 33.93 | 39.03         | 45.49 | 31.17 | 37.25 | 36.17 | 36.84 | 35.86        | (-13.53) | 4.94 | (-93.40%) |
| + SNAP  | 41.72 | 42.62 | 37.46 | 58.43 | 41.24 | 57.33 | 57.96 | 50.34 | 51.17 | 52.29         | 63.59 | 51.32 | 49.68 | 54.78 | 47.89 | 50.52        | (+1.13)  | 4.95 | (-93.38%) |
| EATA    | 38.41 | 39.03 | 32.29 | 61.07 | 38.45 | 58.21 | 60.62 | 49.59 | 49.19 | 54.23         | 62.88 | 57.39 | 49.00 | 53.01 | 42.05 | 49.70        | (-1.04)  | 3.13 | (-27.17%) |
| + SNAP  | 40.62 | 41.53 | 34.31 | 64.08 | 40.29 | 61.32 | 63.04 | 52.00 | 51.77 | 56.85         | 65.98 | 61.96 | 51.05 | 55.67 | 44.80 | 52.35        | (+1.61)  | 3.51 | (-17.50%) |
| SAR     | 43.92 | 45.28 | 38.64 | 63.36 | 42.58 | 60.36 | 62.78 | 53.39 | 52.23 | 57.54         | 65.41 | 60.88 | 52.07 | 56.80 | 47.16 | 53.49        | (-4.45)  | 2.95 | (-56.16%) |
| + SNAP  | 46.29 | 47.60 | 39.95 | 65.26 | 44.00 | 63.09 | 64.97 | 55.08 | 55.17 | 59.73         | 68.13 | 64.72 | 53.84 | 58.98 | 49.54 | 55.76        | (-2.18)  | 3.09 | (-53.73%) |
| RoTTA   | 36.28 | 37.12 | 31.38 | 61.20 | 38.36 | 58.26 | 60.30 | 49.20 | 48.21 | 53.54         | 62.80 | 56.78 | 49.61 | 52.28 | 41.26 | 49.11        | (-2.44)  | 2.96 | (-55.92%) |
| + SNAP  | 37.83 | 38.42 | 32.38 | 63.73 | 39.72 | 61.32 | 62.58 | 51.38 | 51.18 | 55.61         | 65.70 | 61.39 | 51.36 | 54.51 | 42.85 | 51.33        | (-0.22)  | 2.99 | (-55.41%) |

Overall performance across various adaptation rates. Table [1,](#page-7-0) [2](#page-7-1) and Appedix [C](#page-15-1) summarize the performance comparison of baseline state-of-the-art (SOTA) TTA methods and SNAP-TTA integration across various adaptation rates (0.01 to 0.5) on CIFAR10/100-C and ImageNet-C. These results reveal that while Sparse TTA achieves a substantial reduction in adaptation latency up to 87.5% conventional SOTA algorithms suffer significant accuracy degradation under sparse adaptation settings (Table [3,](#page-8-0) Figure [4\)](#page-8-0). In contrast, SNAP-TTA demonstrates a robust ability to mitigate this performance drop. Leveraging minimal updates with only a few samples, SNAP-TTA consistently outperforms baseline methods and shows competitive accuracy even when compared to fully adapted models. Furthermore, in certain scenarios, SNAP-TTA achieves accuracy gains over the

![](_page_8_Figure_1.jpeg)

Figure 4: Latency and accuracy comparison of original TTA methods and their SNAP-TTA integration on CIFAR100-C. SNAP-TTA significantly enhances the efficiency.

Table 3: Latency reduction and accuracy gaps of SNAP-TTA (adaptation rate 0.1) compared by original TTA, tested on Raspberry Pi 4. Performance averaged over 15 CIFAR10-C corruptions. Numbers in parentheses represent the performance difference of SNAP-TTA compared to full adaptation.

| Methods | Original | Latency per TTA | batch (s) SNAP-TTA | naive | STTA     | Accuracy | (%) SNAP-TTA |
|---------|----------|-----------------|--------------------|-------|----------|----------|--------------|
| Tent    | 3.97     | 2.20            | (-44.0%)           | 76.81 | (-3.62)  | 78.95    | (-1.48)      |
| CoTTA   | 71.68    | 8.96            | (-87.5%)           | 66.42 | (-11.58) | 78.83    | (+0.83)      |
| EATA    | 3.93     | 2.18            | (-44.6%)           | 76.29 | (-5.27)  | 78.61    | (-2.95)      |
| SAR     | 5.75     | 2.30            | (-60.1%)           | 76.01 | (-3.04)  | 78.06    | (-0.99)      |
| RoTTA   | 5.93     | 2.25            | (-62.0%)           | 74.78 | (-2.27)  | 77.07    | (+0.07)      |

original counterparts, highlighting its adaptability and effectiveness. These results underscore the capability of SNAP-TTA to balance efficiency and performance, providing a significant advantage in sparse adaptation scenarios while maintaining or even enhancing classification accuracy. This validates the effectiveness of utilizing class-domain representative samples in the STTA setting.

Furthermore, Figure [5](#page-8-1) shows more computationally complex and latency-intensive methods such as CoTTA tend to have greater performance gain when integrated with SNAP-TTA. This is because methods that update the entire model parameters are more susceptible to the influence of specific adaptation samples, leading to significant performance drops under sparse update conditions, which SNAP-TTA's CnDRM and IoBMN effectively mitigate. In addition, adaptation rates of 0.5 or 0.3, which represent relatively high adaptation frequencies, sometimes can achieves even better performance with SNAP-TTA than the original TTA, despite in the STTA setting. This is likely because the sampling rate was not critically low but rather comparable to that of existing data-efficient methods such as EATA [\(Niu et al., 2022\)](#page-11-0), allowing SNAP-TTA to achieve performance gains similar to various sampling-based TTA methods [\(Niu et al., 2022;](#page-11-0) [2023;](#page-11-1) [Gong et al., 2022;](#page-10-0) [2023\)](#page-10-1) using fewer yet effective samples. Overall, SNAP-TTA significantly reduced the average latency per batch while effectively maintaining accuracy, highlighting its benefits for resource-constrained environments. More details on all other adaptation rates are reported in Appendix [C.](#page-15-1) 0.1

Naïve SNAP-TTA Table 4: Classification accuracy (%) comparison of ablative settings on the STTA (adaptation rate 0.1). Performance averaged over 15 CIFAR10-C corruptions.

| latency AR Methods | Tent 1 (full adaptation) | CoTTA | EATA  | SAR   | RoTTA |
|--------------------|--------------------------|-------|-------|-------|-------|
| Na¨ıve             | 76.81                    | 66.42 | 76.29 | 76.01 | 74.78 |
| Random             | 77.08                    | 65.61 | 76.59 | 76.33 | 75.01 |
| LowEntropy         | 75.66                    | 63.19 | 74.89 | 74.41 | 72.60 |
| CRM                | 77.77                    | 65.71 | 77.18 | 74.36 | 75.27 |
| CnDRM              | 77.46                    | 77.69 | 77.17 | 76.85 | 75.64 |
| CnDRM+EMA          | 78.02                    | 72.19 | 77.05 | 76.84 | 76.18 |
| CnDRM+IoBMN        | 78.95                    | 78.83 | 78.61 | 78.06 | 77.07 |

![](_page_8_Figure_8.jpeg)

65.00 70.00 Accuracy (%)Contribution of individual components of SNAP-TTA. We conducted an ablative evaluation to understand the effects of the individual components of SNAP-TTA (Table [4;](#page-8-1) more results on diverse adaptation rates and datasets are on Appendix [D\)](#page-21-0). CRM denotes prediction-balanced sampling with a confidence threshold (same as the Class-Representative criteria of CnDRM), and CnDRM denotes both Class and Domain Representative sampling (the first component of SNAP-TTA). For inference, the default uses test batch normalization statistics, EMA uses the exponential moving average of the test batch, and IoBMN uses memory samples' statistics corrected to match that of the test batch (the second component of SNAP-TTA).

80.00 85.00 Figure 5: Classification accuracy on CIFAR10-C with varying adaptation rates. SNAP-TTA consistently mitigates accuracy drop across all rates.

Contrary to the hypothesis that low-entropy samples are beneficial for TTA [\(Niu et al., 2022;](#page-11-0) [2023\)](#page-11-1), LowEntropy performed worse than Rand for STTA. This can be attributed to the limited updates of STTA, resulting in poor or longer convergence times due to low entropy minimization loss. CRM, originally used for data-efficient supervised deep learning [\(Choi et al., 2024;](#page-10-2) [Xia et al., 2022\)](#page-12-5), performed better than Rand. However, as CRM on TTA inevitably relies on uncertain pseudo labels instead of the ground truth, its performance remains lower than utilizing domain representative features (CnDRM) (note that TTA is unsupervised domain adaptation rather than training from scratch [\(Xia et al., 2022\)](#page-12-5)). The highest accuracy was achieved when inference was performed us-

**504**

**506**

**509**

**514 515 516**

**518 519**

**524**

**529**

Table 5: Classification accuracy (%) on ImageNet-C through Adaptation Rate 0.1 using ViT-based model. Bold numbers are the highest accuracy.

| Methods    | Gau.  | Shot  | Imp.  | Def.  | Gla.  | Mot.  | Zoom  | Snow  | Fro.  | Fog   | Brit. | Cont. | Elas. | Pix.  | JPEG  | Avg.  |
|------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Tent       | 40.56 | 41.30 | 41.69 | 35.76 | 31.81 | 42.01 | 38.02 | 44.33 | 53.53 | 20.69 | 72.41 | 30.42 | 45.87 | 51.95 | 56.11 | 43.10 |
| + SNAP-TTA | 40.98 | 41.72 | 42.18 | 37.16 | 32.30 | 42.89 | 38.44 | 46.19 | 52.50 | 53.11 | 72.25 | 39.25 | 46.77 | 51.53 | 55.99 | 46.22 |
| EATA       | 20.12 | 21.52 | 21.40 | 20.90 | 23.42 | 15.71 | 18.00 | 16.12 | 28.35 | 22.24 | 35.97 | 11.33 | 19.78 | 20.22 | 19.99 | 21.00 |
| + SNAP-TTA | 40.74 | 43.22 | 43.11 | 40.63 | 44.59 | 51.58 | 50.63 | 54.77 | 58.32 | 61.50 | 73.91 | 33.85 | 60.19 | 63.35 | 63.01 | 52.23 |
| SAR        | 21.45 | 23.02 | 23.17 | 23.67 | 24.64 | 15.98 | 14.62 | 7.70  | 31.49 | 8.94  | 41.33 | 6.82  | 17.35 | 22.39 | 22.49 | 20.34 |
| + SNAP-TTA | 37.59 | 38.27 | 36.78 | 38.58 | 39.99 | 49.00 | 45.77 | 43.96 | 56.61 | 59.96 | 73.02 | 19.69 | 54.30 | 61.16 | 61.85 | 47.77 |

ing IoBMN, which primarily utilizes memory statistics and only shifts slightly to the test batch on demand. These results collectively indicate that utilizing CnDRM and IoBMN of SNAP-TTA enhances performance in a low-latency STTA scenario.

Validation of SNAP-TTA on Vision Transformer (ViT) based Model. To validate the effectiveness of SNAP-TTA on the Vision Transformer (ViT) [\(Dosovitskiy, 2020\)](#page-10-7), we conducted experiments on ImageNet-C with adaptation rate of 0.1. Since ViT uses layer normalization (LN), we adjusted CnDRM and IoBMN to use LN from instances, demonstrating that the core concepts of selecting domain-representative samples and mitigating shift in normalization statistics can be applied effectively to a different normalization type (details in Appendix [F.3\)](#page-27-0). The results in Table [5](#page-9-0) confirm consistent accuracy gains of SNAP-TTA with significant latency decrease, regardless of model and normalization types.

# 5 DISCUSSION AND CONCLUSION

Limitations and future work. Our work could be optimized for more realistic data streams, such as continuous domain adaptation scenarios (Appendix [F.2\)](#page-27-1). For instance, the adaptation rate can be dynamically altered based on the need for adaptation (i.e., the data distribution just changed). Additionally, while SNAP-TTA employed a fixed confidence threshold in CnDRM as a safeguard to filter noisy samples, its adaptability could be improved. Dynamically adjusting the threshold based on data characteristics presents a promising direction for future research to enhance sampling efficiency and overall performance.

Moreover, while we focused on reducing adaptation latency, memory overhead is another concern. We note that SNAP-TTA introduces negligible additional memory overhead, as detailed in the Appendix [E.4,](#page-24-0) where related analysis and tracking information from real-device experiments are provided. Additionally, we demonstrate in the Appendix [E.5](#page-25-0) that SNAP-TTA can be effectively used alongside memory-efficient TTA methods such as MECTA [\(Hong et al., 2023\)](#page-10-9), showcasing its compatibility and practicality. Future works could further explore optimizing SNAP-TTA for both latency and memory.

Conclusion We raised the overlooked issue of latency of TTA methods, which is particularly relevant for applications on resource-constrained edge devices. To this end, we propose SNAP-TTA, a Sparse TTA (STTA) framework that could be applied to existing TTA methods to significantly reduce their latency while maintaining competitive accuracy. For effective performance in an STTA setting, we utilize class-domain representative memory of samples for adaptation. Furthermore, we optimize inference by adapting normalization layers using representative samples to account for domain shifts. Extensive experiments and ablative studies demonstrate SNAP-TTA's effectiveness in latency and adaptation accuracy.

# REPRODUCIBILITY STATEMENT

Details of the experiments, including datasets, scenarios, and hyperparameters for reproducibility, are provided in the Appendix [B.](#page-14-1) Additionally, we share the link (https://anonymous.4open.science/r/SNAPTTA-DD0E) of an anonymous repository containing our source code and instructions to validate the reproducibility.

**554 555 556**

**559**

**561**

**564**

**569**

**579**

**584**

# REFERENCES


[1] Motasem Alfarra, Hani Itani, Alejandro Pardo, Shyma Yaser Alhuwaider, Merey Ramazanova, Juan Camilo Perez, Zhipeng Cai, Matthias Muller, and Bernard Ghanem. Evaluation of test- ¨ time adaptation under computational time constraints. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pp. 976–991. PMLR, 21–27 Jul 2024. URL [https:](https://proceedings.mlr.press/v235/alfarra24a.html) [//proceedings.mlr.press/v235/alfarra24a.html](https://proceedings.mlr.press/v235/alfarra24a.html). Hoyong Choi, Nohyun Ki, and Hye Won Chung. Bws: Best window selection based on sample scores for data pruning across broad ranges. *arXiv preprint arXiv:2406.03057*, 2024. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition*, pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848. Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020. Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization for efficiently improving generalization. In *International Conference on Learning Representations*, 2021. URL <https://openreview.net/forum?id=6Tm1mposlrM>. Taesik Gong, Jongheon Jeong, Taewon Kim, Yewon Kim, Jinwoo Shin, and Sung-Ju Lee. NOTE: Robust continual test-time adaptation against temporal correlation. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), *Advances in Neural Information Processing Systems*, 2022. URL <https://openreview.net/forum?id=E9HNxrCFZPV>. Taesik Gong, Yewon Kim, Taeckyung Lee, Sorn Chottananurak, and Sung-Ju Lee. SoTTA: Robust test-time adaptation on noisy data streams. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2016. Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In *International Conference on Learning Representations*, 2019a. URL <https://openreview.net/forum?id=HJz6tiCqYm>. Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. *arXiv preprint arXiv:1903.12261*, 2019b. Junyuan Hong, Lingjuan Lyu, Jiayu Zhou, and Michael Spranger. Mecta: Memory-economic continual test-time model adaptation. In *International Conference on Learning Representations*, 2023. URL <https://openreview.net/pdf?id=N92hjSf5NNh>. Ziheng Jiang, Chiyuan Zhang, Kunal Talwar, and Michael C Mozer. Characterizing structural regularities of labeled data in overparameterized models. In Marina Meila and Tong Zhang (eds.), *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pp. 5034–5044. PMLR, 18–24 Jul 2021. URL <https://proceedings.mlr.press/v139/jiang21k.html>. Diederick P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In *International Conference on Learning Representations (ICLR)*, 2015. Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In Doina Precup and Yee Whye Teh (eds.), *Proceedings of the 34th International Conference on Machine Learning*, volume 70 of *Proceedings of Machine Learning Research*, pp. 1885–1894. PMLR, 06–11 Aug 2017. URL [https://proceedings.mlr.press/v70/koh17a.](https://proceedings.mlr.press/v70/koh17a.html) [html](https://proceedings.mlr.press/v70/koh17a.html).

[2] **604**

[3] **606**

[4] **614 615**

[5] **617**

[6] **619**

[7] **629**

[8] **634**

[9] **636**

[10] A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. *Master's thesis, Department of Computer Science, University of Toronto*, 2009. Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. *Advances in neural information processing systems*, 31, 2018. Yanghao Li, Naiyan Wang, Jianping Shi, Jiaying Liu, and Xiaodi Hou. Revisiting batch normalization for practical domain adaptation, 2016. URL <https://arxiv.org/abs/1603.04779>. Ji Lin, Wei-Ming Chen, Yujun Lin, john cohn, Chuang Gan, and Song Han. Mcunet: Tiny deep learning on iot devices. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 11711–11722. Curran Associates, Inc., 2020. URL [https://proceedings.neurips.cc/paper\\_files/](https://proceedings.neurips.cc/paper_files/paper/2020/file/86c51678350f656dcc7f490a43946ee5-Paper.pdf) [paper/2020/file/86c51678350f656dcc7f490a43946ee5-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/86c51678350f656dcc7f490a43946ee5-Paper.pdf). Haolan Liu, Zixuan Wang, and Jishen Zhao. Cola: Characterizing and optimizing the tail latency for safe level-4 autonomous vehicle systems. *arXiv preprint arXiv:2305.07147*, 2023. Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In *International Conference on Learning Representations (ICLR)*, 2017. TorchVision maintainers and contributors. Torchvision: Pytorch's computer vision library. [https:](https://github.com/pytorch/vision) [//github.com/pytorch/vision](https://github.com/pytorch/vision), 2016. Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. Coresets for data-efficient training of machine learning models. In Hal Daume III and Aarti Singh (eds.), ´ *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pp. 6950–6960. PMLR, 13–18 Jul 2020. URL [https://proceedings.mlr.](https://proceedings.mlr.press/v119/mirzasoleiman20a.html) [press/v119/mirzasoleiman20a.html](https://proceedings.mlr.press/v119/mirzasoleiman20a.html). Zachary Nado, Shreyas Padhy, D Sculley, Alexander D'Amour, Balaji Lakshminarayanan, and Jasper Snoek. Evaluating prediction-time batch normalization for robustness under covariate shift. *arXiv preprint arXiv:2006.10963*, 2020. Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, and Mingkui Tan. Efficient test-time model adaptation without forgetting. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Proceedings of Machine Learning Research*, pp. 16888–16905. PMLR, 17–23 Jul 2022. URL [https://proceedings.mlr.](https://proceedings.mlr.press/v162/niu22a.html) [press/v162/niu22a.html](https://proceedings.mlr.press/v162/niu22a.html). Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, and Mingkui Tan. Towards stable test-time adaptation in dynamic wild world. In *The Eleventh International Conference on Learning Representations*, 2023. URL [https://openreview.net/forum?](https://openreview.net/forum?id=g2YraF75Tj) [id=g2YraF75Tj](https://openreview.net/forum?id=g2YraF75Tj). NVIDIA Corporation. *NVIDIA Jetson Nano*, 2019. URL [https://developer.nvidia.](https://developer.nvidia.com/embedded/jetson-nano) [com/embedded/jetson-nano](https://developer.nvidia.com/embedded/jetson-nano). Accessed: 2024-11-20. Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. Deep learning on a data diet: Finding important examples early in training. In A. Beygelzimer, Y. Dauphin, P. Liang, and

[11] J. Wortman Vaughan (eds.), *Advances in Neural Information Processing Systems*, 2021. URL <https://openreview.net/forum?id=Uj7pF-D-YvT>. Geoff Pleiss, Tianyi Zhang, Ethan Elenberg, and Kilian Q Weinberger. Identifying mislabeled data using the area under the margin ranking. In H. Larochelle,
  - M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 17044–17056. Curran Associates, Inc., 2020. URL [https://proceedings.neurips.cc/paper\\_files/paper/2020/](https://proceedings.neurips.cc/paper_files/paper/2020/file/c6102b3727b2a7d8b1bb6981147081ef-Paper.pdf)

[file/c6102b3727b2a7d8b1bb6981147081ef-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/c6102b3727b2a7d8b1bb6981147081ef-Paper.pdf).

[13] **654**

[14] **656**

[15] **659**

[16] **661**

[17] **664 665**

[18] **669**

[19] **674**

[20] **684**

[21] **686**

[22] **689 690 691**

[23] Omead Pooladzandi, David Davini, and Baharan Mirzasoleiman. Adaptive second order coresets for data-efficient machine learning. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Proceedings of Machine Learning Research*, pp. 17848–17869. PMLR, 17–23 Jul 2022. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v162/pooladzandi22a.html) [v162/pooladzandi22a.html](https://proceedings.mlr.press/v162/pooladzandi22a.html). Joaquin Quinonero-Candela, Masashi Sugiyama, Anton Schwaighofer, and Neil D Lawrence. ˜ *Dataset shift in machine learning*. Mit Press, 2008. Raspberry Pi Foundation. *Raspberry Pi 4 Model B*, 2019. URL [https://www.raspberrypi.](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) [com/products/raspberry-pi-4-model-b/](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/). Accessed: 2024-11-20. Raspberry Pi Foundation. *Raspberry Pi Zero 2 W*, 2021. URL [https://www.raspberrypi.](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/) [com/products/raspberry-pi-zero-2-w/](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/). Accessed: 2024-11-20. Steffen Schneider, Evgenia Rusak, Luisa Eck, Oliver Bringmann, Wieland Brendel, and Matthias Bethge. Improving robustness against common corruptions by covariate shift adaptation. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 11539–11551. Curran Associates, Inc., 2020. URL [https://proceedings.neurips.cc/paper/2020/file/](https://proceedings.neurips.cc/paper/2020/file/85690f81aadc1749175c187784afc9ee-Paper.pdf) [85690f81aadc1749175c187784afc9ee-Paper.pdf](https://proceedings.neurips.cc/paper/2020/file/85690f81aadc1749175c187784afc9ee-Paper.pdf). Mattia Segu, Alessio Tonioni, and Federico Tombari. Batch normalization embeddings for deep domain generalization. *Pattern Recognition*, 135:109115, 2023. Junha Song, Jungsoo Lee, In So Kweon, and Sungha Choi. Ecotta: Memory-efficient continual test-time adaptation via self-distilled regularization. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 11920–11929, 2023. Ardi Tampuu, Kristjan Roosild, and Ilmar Uduste. The effects of speed and delays on test-time performance of end-to-end self-driving. *Sensors*, 24(6):1963, 2024. Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J. Gordon. An empirical study of example forgetting during deep neural network learning. In *International Conference on Learning Representations*, 2019. URL [https://](https://openreview.net/forum?id=BJlxm30cKm) [openreview.net/forum?id=BJlxm30cKm](https://openreview.net/forum?id=BJlxm30cKm). Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully test-time adaptation by entropy minimization. In *International Conference on Learning Representations*, 2021. URL <https://openreview.net/forum?id=uXl3bZLkr3c>. Qin Wang, Olga Fink, Luc Van Gool, and Dengxin Dai. Continual test-time domain adaptation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 7201–7211, June 2022. Xiaobo Xia, Jiale Liu, Jun Yu, Xu Shen, Bo Han, and Tongliang Liu. Moderate coreset: A universal method of data selection for real-world data-efficient deep learning. In *The Eleventh International Conference on Learning Representations*, 2022. Shuo Yang, Zeke Xie, Hanyu Peng, Min Xu, Mingming Sun, and Ping Li. Dataset pruning: Reducing training data by examining generalization influence. In *The Eleventh International Conference on Learning Representations*, 2023. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=4wZiAXD29TQ) [4wZiAXD29TQ](https://openreview.net/forum?id=4wZiAXD29TQ). Longhui Yuan, Binhui Xie, and Shuang Li. Robust test-time adaptation in dynamic scenarios. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 15922–15932, June 2023. Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In *Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I 13*, pp. 818–833. Springer, 2014.

[24] Marvin Zhang, Sergey Levine, and Chelsea Finn. Memo: Test time robustness via adaptation and augmentation. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), *Advances in Neural Information Processing Systems*, volume 35, pp. 38629–38642. Curran Associates, Inc., 2022. URL [https://proceedings.neurips.cc/paper\\_files/paper/2022/file/](https://proceedings.neurips.cc/paper_files/paper/2022/file/fc28053a08f59fccb48b11f2e31e81c7-Paper-Conference.pdf) [fc28053a08f59fccb48b11f2e31e81c7-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/fc28053a08f59fccb48b11f2e31e81c7-Paper-Conference.pdf).

[25] **759**

[26] **761**

[27] **764**

[28] **766**

[29] **769**

[30] **779 780 781**

[31] **784**

[32] **804 805 806**

[33] **809**
# A RELATED WORK

Test-time adaptation. Test-time adaptation (TTA) aims to improve model performance on Out-of-Distribution (OOD) data by using only the unlabeled test data stream to adapt the model. Test-time normalization [\(Nado et al., 2020;](#page-11-4) [Schneider et al., 2020\)](#page-12-10) adjusts the batch normalization (BN) statistics using test data to improve performance. Other works mainly involve updating the parameters of the model during test-time. Tent [\(Wang et al., 2021\)](#page-12-1) adapts the affine parameters of the BN layers to minimize the entropy of its predictions. EATA [\(Niu et al., 2022\)](#page-11-0) builds upon Tent, sampling reliable and non-redundant samples and utilizing an anti-forgetting regularizer for efficiency. Other works introduce more complex schemes, primarily to improve robustness against more practical test-time scenarios. CoTTA [\(Wang et al., 2022\)](#page-12-2) addresses a continually changing test-time environment by using weight-averaged and augmentation-averaged predictions with stochastic restoring. SAR [\(Niu et al., 2023\)](#page-11-1) filters samples with large and noisy gradients to stabilize the model during wilder test-time scenarios. RoTTA [\(Yuan et al., 2023\)](#page-12-3) targets a practical test-time setting of changing distributions and correlative sampling by introducing a memory bank and a teacher-student model.

Test-time adaptation on edge devices. TTA on edge devices primarily inherit the challenges of on-device learning: limited memory and increased latency from general resource constraints [\(Lin et al., 2020\)](#page-11-8). Several memory-efficient TTA works have been proposed in this regard. MECTA [\(Hong et al., 2023\)](#page-10-9) aims to reduce the memory consumption of gradient-based TTA, proposing an adaptive normalization layer to reduce the intermediate caches for backpropagation. Another work EcoTTA [\(Song et al., 2023\)](#page-12-4) proposes memory-efficient continual TTA by adapting lightweight meta networks instead of the originals to reduce the size of intermediate activations. Despite works to promote memory-efficiency, the latency of TTA, especially on resource-constrained edge devices, has been generally overlooked. While many adaptation-based TTA [\(Wang et al.,](#page-12-1) [2021;](#page-12-1) [Niu et al., 2022;](#page-11-0) [2023;](#page-11-1) [Yuan et al., 2023\)](#page-12-3) update only the affine parameters for general time and memory concerns, they still involve computationally-heavy operations every batch, which can lead to high latency on edge devices. A recent work [\(Alfarra et al., 2024\)](#page-10-5) introduces a more realistic TTA evaluation protocol that penalizes slow TTA methods by providing them with fewer samples for adaptation. We build on from this notion, proposing a sparse TTA setting to reduce the latency of existing TTA methods, but at a minimal cost to performance.

Data-efficient deep learning. Data-efficient deep learning methods enable deep learning models to achieve competitive performance with less data. Among these methods, data selection, or data sampling, involves utilizing a small subset of the training data in an attempt to match that of fulldataset training. A branch of data-selection is score-based selection, which scores each sample based on some predefined metric, such as a sample's influence [\(Koh & Liang, 2017\)](#page-10-10), difficulty [\(Toneva](#page-12-11) [et al., 2019;](#page-12-11) [Paul et al., 2021\)](#page-11-9), prediction confidence [\(Pleiss et al., 2020\)](#page-11-10), or consistency [\(Jiang et al.,](#page-10-11) [2021\)](#page-10-11), and selects samples with scores in a certain range. Another set of data-selection methods involve optimization-based selection, which formulates an optimization problem to find a optimal subset that can best approximate full-dataset training [\(Mirzasoleiman et al., 2020;](#page-11-11) [Yang et al., 2023;](#page-12-12) [Pooladzandi et al., 2022\)](#page-12-13). While these approaches work well in their preconceived settings, they generally suffer performance drop as their settings change, such as a change in sampling ratio. More recent works like the Moderate Coreset [\(Xia et al., 2022\)](#page-12-5) proposes a more robust selection approach by using the distance of a sample to the class center as a score criterion, for an effective representation of the dataset. While our proposed sparse TTA setting is more challenging than the conventional data-efficient setting, as we cannot access ground truths labels nor make assumptions regarding the model, we utilize similar ideas of representative sampling as motivation for our method.

# B EXPERIMENT DETAILS

All experiments presented in this paper were conducted using three random seeds (0, 1, 2), and we report the average accuracies along with their corresponding standard deviations. To ensure efficiency in experimentation, accuracy measurements were obtained using NVIDIA GeForce RTX 3090 GPUs, as the performance differences attributable to the random seed are negligible. Latency measurements were conducted on a Raspberry Pi 4 [\(Raspberry Pi Foundation, 2019\)](#page-12-6), equipped with a Quad-core Cortex-A72 (ARM v8) 64-bit SoC @ 1.8GHz CPU and 4GB RAM.

**814 815**

**817**

**819**

**829**

**834**

**836**

**854**

**856**

#### B.1 BASELINE IMPLEMENTATION DETAILS

In this study, we utilized the official implementations of the baseline methods. To ensure consistency, we adopted the reported best hyperparameters documented in the respective papers or source code repositories as much as possible. Also, we present information about the implementation specifics of the baseline methods and provide a comprehensive overview of our experimental setup, including detailed descriptions of the employed hyperparameters.

We adopt hyperparameters from the original papers or the official code of the baselines for consistency. To assess the generality of SNAP-TTA, the test batch sizes were set to 16 for all baseline methods to ensure a fair comparison. To minimize overhead and maintain consistency with inference batches, we set the size of CnDRM equal to the batch size. TTA is conducted in an online manner, with adaptation or inference performed per batch. When there was a conflict between the implementation of SNAP-TTA and certain components of the existing baseline methods, we prioritized SNAP-TTA's features for fair evaluation at the STTA setting.

For Tent [\(Wang et al., 2021\)](#page-12-1), we update the BN affine parameters using the SGD optimizer [\(Loshchilov & Hutter, 2017\)](#page-11-12) with a learning rate of l = 1e − 3 for CIFAR10/100C and l = 1e − 4 for ImageNet-C. For separate experimentation on the ViT, we used a learning rate of l = 2e − 4. For CoTTA [\(Wang et al., 2022\)](#page-12-2), we update all model parameters using the Adam optimizer [\(Kingma & Ba, 2015\)](#page-10-12) with a learning rate of l = 1e − 4. Furthermore, we set CoTTA's teacher model EMA factor to α = 0.99, the restoration factor to p = 0.1, and the anchor probability to pth = 0.9. For EATA [\(Niu et al., 2022\)](#page-11-0), we use the SGD optimizer with a learning rate of l = 1e − 4. We set the entropy threshold as E<sup>0</sup> = 0.4 × ln |N|, where N is the total number of classes. For SAR [\(Niu et al., 2023\)](#page-11-1), we use SAM [\(Foret et al., 2021\)](#page-10-13) with the base optimizer as SGD with a learning rate of l = 1e − 3. For fair evaluation, we replaced the sample filtering scheme with SNAP-TTA's CnDRM. For RoTTA [\(Yuan et al., 2023\)](#page-12-3), we use the SGD optimizer with a learning rate of l = 1e−3. For fair evaluation, we replaced RoTTA's RBN and CSTU with SNAP-TTA's Cn-DRM and IoBMN. For the teacher-student structure, we set the teacher model's exponential moving average update rate as v = 1e − 3.

Finally, we list the hyperparameters specific to the components of SNAP-TTA. The confidence threshold for CnDRM τconf is set to 0.4 for CIFAR10-C, 0.45 for CIFAR100-C, and 0.5 for ImageNet-C. The entropy threshold for our ablation study τentr is set to log(10)×0.40 for CIFAR10- C and log(100) × 0.40 for CIFAR100-C, as referenced in a previous work using entropy-based filtering [\(Niu et al., 2022\)](#page-11-0). Additionally, the parameters for the soft shrinkage function in IoBMN are fixed with α = 4 for Tent, CoTTA, SAR, RoTTA, and α = 2 for EATA.

# C DETAILED EXPERIMENT RESULTS

In this section, we provide detailed experimental results for the performance comparison of SNAP-TTA across a wide range of adaptation rates. We evaluated the performance on CIFAR10-C, CIFAR100-C, and ImageNet-C datasets with adaptation rates of 0.01, 0.03, 0.05, 0.1, 0.3, and 0.5, and across five state-of-the-art (SOTA) TTA algorithms: Tent, EATA, SAR, CoTTA, and RoTTA. This comprehensive evaluation resulted in a total of 150 combinations (3 datasets, 6 adaptation rates, 5 algorithms).

The results demonstrate that, regardless of the adaptation rate, dataset, or the TTA algorithm, integrating SNAP-TTA consistently outperforms the baseline methods. Specifically, SNAP-TTA achieved the highest accuracy across nearly all of these 150 combinations, effectively demonstrating its robustness in both high and low adaptation settings. For CIFAR10-C and CIFAR100-C, SNAP-TTA showed substantial performance improvements compared to the baseline, even at very low adaptation rates (e.g., 0.01 and 0.05). Similarly, for ImageNet-C, SNAP-TTA maintained superior accuracy across diverse corruption types.

These results highlight that SNAP-TTA effectively balances adaptation and latency, ensuring optimal performance even when the adaptation rate is sparse and regardless of the underlying TTA algorithm. This consistent superiority across all 150 combinations underscores SNAP-TTA's suitability for practical, real-world applications on resource-constrained devices.

**869**

**874**

| AR | Methods   | Gau.           | Shot           | Imp.           | Def.           | Gla.           | Most           | Snow           | Fro.           | Brit.          | Cont.          | Elas.          | Pix.           | JPEG           | Avg. |
|----|-----------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|------|
| 87 | Source    | 3.00<br>±0.00  | 3.00<br>±0.00  | 3.00<br>±0.00  | 3.00<br>±0.00  | 3.00<br>±0.00  | 3.00<br>±0.00  | 3.00<br>±0.00  | 3.00<br>±0.00  | 3.00<br>±0.00  | 3.00<br>±0.00  | 3.00<br>±0.00  | 3.00<br>±0.00  | 3.00<br>±0.00  |      |
|    | BN stats  | 14.29<br>±0.05 | 15.06<br>±0.02 | 14.89<br>±0.08 | 13.30<br>±0.08 | 13.38<br>±0.08 | 35.22<br>±0.06 | 31.78<br>±0.04 | 30.26<br>±0.14 | 44.02<br>±0.11 | 15.14<br>±0.05 | 40.42<br>±0.10 | 36.53<br>±0.04 | 29.07<br>±0.16 |      |
|    | Tent      | 27.03<br>±0.05 | 28.98<br>±0.08 | 28.64<br>±0.29 | 24.66<br>±0.27 | 23.63<br>±0.25 | 45.77<br>±0.10 | 44.82<br>±0.08 | 38.06<br>±0.08 | 54.61<br>±0.08 | 16.84<br>±0.15 | 51.64<br>±0.10 | 55.54<br>±0.15 | 39.53<br>±0.07 |      |
|    | CoTTA     | 13.12<br>±0.08 | 13.98<br>±0.07 | 13.94<br>±0.01 | 12.44<br>±0.10 | 12.34<br>±0.04 | 35.22<br>±0.06 | 31.78<br>±0.06 | 30.26<br>±0.06 | 44.02<br>±0.14 | 15.13<br>±0.03 | 40.42<br>±0.10 | 45.26<br>±0.04 | 28.75<br>±0.16 |      |
|    | EATA      | 29.62<br>±0.02 | 31.17<br>±0.09 | 26.89<br>±0.03 | 26.30<br>±0.06 | 40.65<br>±0.12 | 46.29<br>±0.06 | 40.78<br>±0.09 | 55.57<br>±0.08 | 64.97<br>±0.08 | 38.02<br>±0.08 | 52.66<br>±0.20 | 56.03<br>±0.04 | 42.56<br>±0.16 |      |
|    | SAR       | 17.49<br>±0.40 | 22.04<br>±0.44 | 21.21<br>±0.96 | 12.62<br>±0.72 | 39.76<br>±0.63 | 44.13<br>±0.11 | 45.98<br>±0.23 | 39.39<br>±0.30 | 55.13<br>±0.20 | 63.71<br>±0.01 | 52.31<br>±0.08 | 56.09<br>±0.08 | 49.35<br>±0.43 |      |
|    | RoTTA     | 20.60<br>±0.09 | 22.83<br>±0.24 | 19.81<br>±0.24 | 10.46<br>±0.06 | 10.26<br>±0.27 | 21.31<br>±0.23 | 39.66<br>±0.18 | 32.09<br>±0.18 | 46.08<br>±0.18 | 62.22<br>±0.27 | 20.27<br>±0.49 | 42.54<br>±0.29 | 47.47<br>±0.23 |      |
|    | Tent      | 25.24<br>±0.10 | 26.86<br>±0.27 | 26.35<br>±0.08 | 23.26<br>±0.05 | 22.41<br>±0.09 | 35.99<br>±0.10 | 44.60<br>±0.13 | 37.68<br>±0.17 | 53.60<br>±0.12 | 64.40<br>±0.94 | 21.35<br>±0.12 | 52.32<br>±0.12 | 47.93<br>±0.04 |      |
|    | +SNAP-TTA | 28.05<br>±0.00 | 29.39<br>±0.04 | 25.73<br>±0.15 | 23.89<br>±0.06 | 38.49<br>±0.17 | 44.21<br>±0.03 | 39.57<br>±0.09 | 53.90<br>±0.10 | 64.52<br>±0.09 | 34.39<br>±0.13 | 49.99<br>±0.14 | 54.88<br>±0.07 | 40.72<br>±0.09 |      |
|    | CoTTA     | 11.99<br>±0.13 | 12.86<br>±0.20 | 11.90<br>±0.07 | 11.64<br>±0.02 | 22.92<br>±0.06 | 35.06<br>±0.09 | 31.20<br>±0.06 | 29.97<br>±0.06 | 42.86<br>±0.07 | 14.02<br>±0.09 | 40.39<br>±0.05 | 45.29<br>±0.09 | 38.58<br>±0.12 |      |
|    | +SNAP-TTA | 15.16<br>±0.14 | 15.86<br>±0.02 | 13.98<br>±0.14 | 14.43<br>±0.04 | 24.69<br>±0.09 | 36.51<br>±0.07 | 32.59<br>±0.16 | 31.71<br>±0.06 | 45.98<br>±0.09 | 65.62<br>±0.08 | 15.72<br>±0.08 | 42.05<br>±0.09 | 46.71<br>±0.24 |      |
|    | 0.5       | 28.62<br>±0.10 | 30.24<br>±0.14 | 25.34<br>±0.20 | 24.84<br>±0.04 | 38.94<br>±0.10 | 46.85<br>±0.12 | 45.20<br>±0.12 | 40.53<br>±0.01 | 55.04<br>±0.06 | 34.48<br>±0.01 | 52.06<br>±0.12 | 55.57<br>±0.13 | 41.42          |      |

### C.1 IMAGENET-C

Table 6: STTA classification accuracy (%) comparing with and without SNAP-TTA on ImageNet-C through Adaptation Rates(AR) (0.5, 0.3, and 0.1), including results for full adaptation (AR=1). Bold numbers are the highest accuracy.

|            | AR | Methods   | Gauss          | Short          | Imp.           | Def.           | Gla.           | Mot.           | Zero           | Mot.           | Zero           | Mot.           | Fro.           | For.           | For.           | Brit.          | Cont.          | Pfx.           | Pfx. | Avg. |
|------------|----|-----------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|------|------|
| <b>G2</b>  |    | Tent      | 23.77<br>±0.40 | 26.05<br>±0.43 | 24.44<br>±0.58 | 20.54<br>±0.70 | 20.27<br>±0.62 | 32.73<br>±0.30 | 42.57<br>±0.14 | 40.82<br>±0.15 | 35.92<br>±0.33 | 52.78<br>±0.12 | 62.82<br>±0.02 | 15.95<br>±0.18 | 13.03<br>±0.18 | 53.46<br>±0.09 | 47.19<br>±0.03 | 36.62<br>±0.35 |      |      |
| <b>G3</b>  |    | +SNAP-TTA | 29.12<br>±0.09 | 30.46<br>±0.22 | 30.30<br>±0.48 | 25.77<br>±0.20 | 25.22<br>±0.32 | 38.21<br>±0.43 | 44.29<br>±0.00 | 44.29<br>±0.13 | 39.95<br>±0.07 | 65.47<br>±0.05 | 43.81<br>±0.09 | 58.83<br>±0.10 | 45.22<br>±0.10 | 49.21<br>±0.03 | 41.27<br>±0.03 |                |      |      |
| <b>G4</b>  |    | CoTTA     | 11.03<br>±0.30 | 11.91<br>±0.57 | 11.75<br>±0.33 | 11.03<br>±0.24 | 11.20<br>±0.46 | 34.98<br>±0.18 | 34.98<br>±0.05 | 30.87<br>±0.08 | 29.78<br>±0.01 | 43.99<br>±0.06 | 61.87<br>±0.06 | 12.92<br>±0.36 | 40.26<br>±0.19 | 36.63<br>±0.17 | 27.73<br>±0.07 | 27.72<br>±0.22 |      |      |
| <b>G5</b>  |    | +SNAP-TTA | 15.22<br>±0.08 | 15.97<br>±0.11 | 15.93<br>±0.03 | 13.91<br>±0.06 | 14.05<br>±0.12 | 24.05<br>±0.04 | 36.48<br>±0.00 | 32.60<br>±0.07 | 31.65<br>±0.04 | 46.09<br>±0.03 | 63.59<br>±0.07 | 15.67<br>±0.05 | 42.00<br>±0.03 | 47.96<br>±0.09 | 30.12<br>±0.06 | 30.12<br>±0.06 |      |      |
| <b>G6</b>  |    | EATA      | 19.53<br>±0.31 | 20.72<br>±0.66 | 20.72<br>±0.75 | 16.74<br>±0.41 | 19.96<br>±0.58 | 29.11<br>±0.49 | 23.72<br>±0.27 | 34.84<br>±0.18 | 50.75<br>±0.23 | 50.75<br>±0.21 | 63.29<br>±0.13 | 19.86<br>±0.16 | 45.92<br>±0.17 | 51.15<br>±0.17 | 44.13<br>±0.09 | 44.13<br>±0.04 |      |      |
| <b>G7</b>  |    | +SNAP-TTA | 22.83<br>±0.10 | 23.95<br>±0.34 | 23.62<br>±0.30 | 19.43<br>±0.09 | 19.70<br>±0.19 | 30.56<br>±0.56 | 41.59<br>±0.08 | 38.06<br>±0.11 | 35.06<br>±0.21 | 50.98<br>±0.18 | 63.30<br>±0.13 | 23.72<br>±0.30 | 46.26<br>±0.16 | 51.52<br>±0.18 | 45.46<br>±0.18 | 35.72<br>±0.21 |      |      |
| <b>G8</b>  |    | SAR       | 23.25<br>±0.21 | 24.23<br>±0.34 | 23.66<br>±0.30 | 19.98<br>±0.09 | 20.38<br>±0.16 | 33.05<br>±0.16 | 43.04<br>±0.16 | 40.73<br>±0.02 | 36.06<br>±0.12 | 52.61<br>±0.09 | 20.17<br>±0.07 | 49.00<br>±0.04 | 53.35<br>±0.11 | 46.73<br>±0.11 | 36.69<br>±0.20 | 36.69<br>±0.20 |      |      |
| <b>G9</b>  |    | +SNAP-TTA | 27.54<br>±0.06 | 29.03<br>±0.05 | 28.66<br>±0.04 | 24.05<br>±0.16 | 23.42<br>±0.08 | 36.28<br>±0.12 | 44.12<br>±0.11 | 42.89<br>±0.07 | 38.54<br>±0.07 | 53.24<br>±0.05 | 64.25<br>±0.04 | 31.83<br>±0.24 | 48.79<br>±0.23 | 54.04<br>±0.18 | 47.80<br>±0.18 | 39.63<br>±0.12 |      |      |
| <b>G10</b> |    | RoTTA     | 14.42<br>±0.06 | 15.22<br>±0.05 | 15.02<br>±0.10 | 13.25<br>±0.11 | 13.31<br>±0.07 | 23.79<br>±0.03 | 35.27<br>±0.08 | 32.09<br>±0.05 | 30.43<br>±0.07 | 44.71<br>±0.14 | 62.64<br>±0.14 | 15.24<br>±0.09 | 40.63<br>±0.10 | 45.55<br>±0.17 | 36.75<br>±0.09 | 29.22<br>±0.09 |      |      |
| <b>G11</b> |    | +SNAP-TTA | 14.65<br>±0.06 | 15.48<br>±0.02 | 15.29<br>±0.08 | 13.43<br>±0.09 | 13.45<br>±0.09 | 23.93<br>±0.03 | 35.33<br>±0.06 | 32.18<br>±0.05 |                |                |                |                |                |                |                |                |      |      |

**956**

**959**

**961**

Table 7: STTA classification accuracy (%) comparing with and without SNAP-TTA on ImageNet-C through Adaptation Rates(AR) (0.05, 0.03, and 0.01). Bold numbers are the highest accuracy.

**979**

**989 990 991**

**994**

**1011**

**1014 1015**

**1017**

# C.2 CIFAR10-C

Table 8: STTA classification accuracy (%) comparing with and without SNAP-TTA on CIFAR10-C through Adaptation Rates(AR) (0.5, 0.3, and 0.1), including results for full adaptation (AR=1). Bold numbers are the highest accuracy.

| AR         | Methods | Gaucs | Shot  | Impulse | Def.  | Glass | Mot.  | Zoom  | Snow  | Fro.  | Brit. | Cont. | Elas. | Pix.  | IPEG  | Age   |
|------------|---------|-------|-------|---------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Source     | 22.13   | 0.00  | 22.53 | 0.00    | 25.10 | 0.00  | 25.10 | 0.00  | 25.10 | 0.00  | 25.10 | 0.00  | 25.10 | 0.00  | 25.10 |       |
|            | 0.00    | 0.00  | 0.00  | 0.00    | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  |       |
| BN stats   | 63.72   | 65.67 | 57.14 | 84.99   | 62.72 | 83.86 | 62.46 | 78.98 | 76.95 | 83.32 | 88.46 | 84.60 | 73.96 | 66.71 | 78.59 |       |
|            | 0.04    | 0.12  | 0.25  | 0.31    | 0.23  | 0.48  | 0.30  | 0.30  | 0.08  | 0.17  | 0.16  | 0.17  | 0.18  | 0.02  | 0.42  |       |
| Tent       | 73.66   | 76.18 | 68.04 | 86.61   | 67.12 | 85.73 | 86.24 | 82.34 | 81.56 | 86.02 | 89.99 | 87.16 | 76.40 | 82.95 | 76.45 | 80.45 |
|            | 0.08    | 0.94  | 1.12  | 0.50    | 0.76  | 0.38  | 0.09  | 0.94  | 0.64  | 0.18  | 0.16  | 0.25  | 0.82  | 0.15  | 0.46  | 0.71  |
| 1 CoTTA    | 73.97   | 79.37 | 67.73 | 83.91   | 66.75 | 82.64 | 83.24 | 79.92 | 79.49 | 82.41 | 88.39 | 80.14 | 75.38 | 79.24 | 75.42 | 78.05 |
|            | 0.32    | 0.48  | 0.66  | 0.20    | 0.08  | 0.34  | 0.19  | 0.09  | 0.13  | 0.23  | 0.18  | 0.17  | 0.09  | 0.07  | 0.25  | 0.23  |
| EATA       | 75.82   | 77.61 | 69.63 | 81.14   | 69.41 | 85.96 | 87.08 | 83.42 | 82.28 | 86.58 | 90.04 | 89.26 | 77.62 | 83.35 | 87.71 | 81.51 |
|            | 0.50    | 0.27  | 0.87  | 0.29    | 0.68  | 0.39  | 0.27  | 0.38  | 0.29  | 0.41  | 0.17  | 0.39  | 0.28  | 0.32  | 0.20  | 0.33  |
| SAR        | 73.52   | 74.03 | 65.45 | 85.69   | 65.01 | 84.63 | 85.01 | 81.47 | 80.91 | 84.18 | 88.70 | 86.23 | 74.94 | 81.20 | 74.84 | 79.00 |
|            | 0.13    | 0.46  | 0.81  | 0.37    | 0.35  | 0.53  | 0.34  | 0.37  | 0.72  | 0.09  | 0.12  | 0.16  | 0.03  | 0.28  | 0.69  | 0.52  |
| RoTTA      | 66.54   | 68.60 | 60.27 | 85.73   | 64.84 | 84.68 | 85.01 | 81.57 | 78.02 | 84.13 | 89.00 | 84.91 | 75.06 | 77.96 | 70.12 | 77.03 |
|            | 0.04    | 0.23  | 0.04  | 0.05    | 0.03  | 0.03  | 0.05  | 0.05  | 0.06  | 0.09  | 0.07  | 0.19  | 0.15  | 0.16  | 0.36  | 0.32  |
| Tent       | 73.44   | 75.93 | 67.18 | 86.52   | 67.28 | 85.25 | 86.23 | 82.24 | 80.35 | 85.39 | 89.80 | 87.77 | 77.00 | 82.08 | 75.58 | 80.14 |
|            | 0.06    | 0.44  | 0.18  | 0.17    | 0.17  | 0.49  | 0.42  | 0.77  | 0.14  | 0.20  | 0.28  | 0.27  | 0.05  | 0.06  | 0.60  | 0.55  |
| + SNAP-TTA | 75.17   | 77.66 | 68.78 | 88.25   | 69.18 | 87.11 | 88.19 | 84.21 | 82.72 | 87.34 | 91.63 | 86.30 | 78.76 | 83.43 | 77.28 | 81.74 |
|            | 0.00    | 0.78  | 0.16  | 0.38    | 0.51  | 0.18  | 0.13  | 0.29  | 0.45  | 0.51  | 0.12  | 0.17  | 0.08  | 0.18  | 0.50  | 0.44  |
| CoTTA      | 65.08   | 66.67 |       |         |       |       |       |       |       |       |       |       |       |       |       |       |

|      | AR         | Methods | Gau.  | Shot  | Imp.  | Def.  | Gla.  | Mot.  | Zoom  | Fro.  | Fog   | Brit. | Cont. | Elas. | Pfx.  | JPEG  | Avg   |       |
|------|------------|---------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1020 |            |         |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |       |
| 1030 |            | Tent    | 64.05 | 58.48 | 85.00 | 62.61 | 82.76 | 84.63 | 79.01 | 77.66 | 83.76 | 82.84 | 80.02 | 82.34 | 74.16 | 77.04 | 75.75 |       |
|      |            |         | ±0.55 | ±0.58 | ±0.42 | ±0.60 | ±0.44 | ±0.70 | ±0.55 | ±0.74 | ±0.48 | ±0.56 | ±0.93 | ±0.10 | ±0.60 | ±0.48 | ±0.57 |       |
| 1031 | + SNAP-TTA |         | 67.71 | 69.84 | 59.53 | 87.10 | 64.66 | 85.73 | 86.53 | 80.68 | 78.92 | 85.06 | 90.19 | 86.72 | 76.16 | 88.76 | 70.95 | 77.93 |
|      |            |         | ±0.38 | ±0.82 | ±1.10 | ±0.15 | ±0.25 | ±0.20 | ±0.20 | ±0.23 | ±0.14 | ±0.08 | ±0.31 | ±0.20 | ±0.17 | ±0.42 | ±0.30 | ±0.33 |
| 1032 | CoTTA      |         | 59.27 | 61.18 | 56.33 | 72.22 | 57.37 | 74.27 | 72.61 | 70.03 | 68.68 | 74.82 | 79.72 | 65.57 | 66.92 | 64.13 | 65.25 | 67.22 |
|      |            |         | ±0.66 | ±1.12 | ±0.06 | ±1.43 | ±1.10 | ±1.46 | ±1.11 | ±1.02 | ±0.92 | ±1.09 | ±1.07 | ±1.38 | ±1.14 | ±1.27 | ±0.98 | ±1.03 |
| 1033 | + SNAP-TTA |         | 71.42 | 73.31 | 65.91 | 85.23 | 67.01 | 84.19 | 84.91 | 80.80 | 80.56 | 84.49 | 90.00 | 82.09 | 76.31 | 79.75 | 75.18 | 78.73 |
|      |            |         | ±0.29 | ±0.12 | ±0.13 | ±0.11 | ±0.21 | ±0.20 | ±0.14 | ±0.19 | ±0.34 | ±0.14 | ±0.23 | ±0.35 | ±0.05 | ±0.29 | ±0.21 | ±0.25 |
| 1034 | EATA       |         | 64.68 | 67.01 | 58.07 | 84.90 | 62.56 | 82.64 | 84.57 | 78.77 | 77.16 | 83.09 | 87.80 | 81.62 | 74.05 | 76.99 | 53.71 | 75.58 |
|      |            |         | ±0.31 | ±0.37 | ±0.42 | ±0.54 | ±0.33 | ±0.67 | ±0.61 | ±0.71 | ±0.92 | ±0.44 | ±0.47 | ±0.59 | ±0.28 | ±0.41 | ±0.71 | ±0.54 |
| 1035 | + SNAP-TTA |         | 67.36 | 68.73 | 59.35 | 87.05 | 64.36 | 85.62 | 86.48 | 81.31 | 78.73 | 85.33 | 90.03 | 86.31 | 76.04 | 78.79 | 70.90 | 77.76 |
|      |            |         | ±0.33 | ±0.26 | ±0.37 | ±0.22 | ±0.18 | ±0.18 | ±0.25 | ±0.24 | ±0.22 | ±0.15 | ±0.24 | ±0.07 | ±0.12 | ±0.27 | ±0.38 | ±0.25 |
| 1036 | SAR        |         | 64.79 | 66.32 | 57.58 | 84.66 | 62.46 | 81.42 | 84.13 | 78.87 | 77.20 | 82.62 | 88.10 | 82.12 | 74.04 | 75.38 | 69.13 | 75.25 |
|      |            |         | ±0.13 | ±0.86 | ±0.69 | ±0.72 | ±0.26 | ±1.52 | ±0.34 | ±0.26 | ±0.81 | ±1.24 | ±0.41 | ±0.74 | ±0.05 | ±0.80 | ±0.52 | ±0.62 |
| 1037 | + SNAP-TTA |         | 66.00 | 68.85 | 58.47 | 86.54 | 63.06 | 85.26 | 86.13 | 80.38 | 78.17 | 85.17 | 90.93 | 85.96 | 75.27 | 77.37 | 70.61 | 77.21 |
|      |            |         | ±0.17 | ±0.75 | ±0.42 | ±0.25 | ±0.28 | ±0.09 | ±0.38 | ±0.09 | ±0.27 | ±0.13 | ±0.36 | ±0.20 | ±0.31 | ±0.28 |       |       |

**1071**

Table 9: STTA classification accuracy (%) comparing with and without SNAP-TTA on CIFAR10-C through Adaptation Rates(AR) (0.05, 0.03, and 0.01). Bold numbers are the highest accuracy.

|      | AR | Methods    | Gau.           | Shot           | Imp.           | Def.           | Gla.           | Mot.           | Zoom           | Snow           | Fro.           | Fog            | Brit.          | Cont.          | Elas.          | Pix.           | JPEG           | Avg            |
|------|----|------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| 1092 |    | Source     | 10.26<br>±0.00 | 11.87<br>±0.00 | 6.48<br>±0.00  | 35.16<br>±0.00 | 20.33<br>±0.00 | 44.42<br>±0.00 | 42.13<br>±0.00 | 45.99<br>±0.00 | 34.84<br>±0.00 | 41.12<br>±0.00 | 66.37<br>±0.00 | 19.54<br>±0.00 | 55.99<br>±0.00 | 22.68<br>±0.00 | 45.48<br>±0.00 | 33.15<br>±0.00 |
| 1093 |    | BN stats   | 36.90<br>±0.10 | 37.96<br>±0.24 | 32.13<br>±0.44 | 62.65<br>±0.26 | 39.14<br>±0.19 | 60.05<br>±0.42 | 61.16<br>±0.05 | 50.68<br>±0.13 | 50.38<br>±0.09 | 54.81<br>±0.05 | 64.40<br>±0.05 | 60.33<br>±0.12 | 50.48<br>±0.24 | 53.49<br>±0.12 | 41.98<br>±0.49 | 50.44<br>±0.21 |
| 1095 |    | Tent       | 46.71<br>±0.29 | 48.06<br>±0.47 | 40.98<br>±0.13 | 65.19<br>±0.40 | 44.10<br>±0.41 | 62.78<br>±0.23 | 63.95<br>±0.36 | 55.43<br>±0.49 | 55.46<br>±0.39 | 59.32<br>±0.17 | 67.43<br>±0.12 | 63.83<br>±0.42 | 53.89<br>±0.15 | 59.40<br>±0.15 | 49.91<br>±0.66 | 55.76<br>±0.33 |
| 1096 | 1  | CoTTA      | 42.14<br>±0.34 | 42.92<br>±0.44 | 37.92<br>±0.18 | 55.40<br>±0.12 | 41.01<br>±0.19 | 55.18<br>±0.10 | 55.39<br>±0.08 | 49.46<br>±0.23 | 50.61<br>±0.63 | 50.86<br>±0.13 | 61.35<br>±0.27 | 47.44<br>±0.37 | 48.69<br>±0.18 | 54.38<br>±0.16 | 48.11<br>±0.35 | 49.36<br>±0.37 |
| 1097 |    | EATA       | 38.42<br>±0.41 | 39.96<br>±0.75 | 32.64<br>±0.41 | 62.35<br>±0.41 | 38.73<br>±0.33 | 59.93<br>±0.17 | 61.07<br>±0.36 | 50.50<br>±0.34 | 50.79<br>±0.34 | 55.30<br>±0.12 | 64.38<br>±0.12 | 60.63<br>±0.13 | 49.66<br>±0.32 | 53.63<br>±0.41 | 43.02<br>±0.20 | 50.73<br>±0.33 |
| 1098 |    | SAR        | 50.75<br>±0.44 | 52.00<br>±0.22 | 43.87<br>±0.40 | 65.44<br>±0.39 | 46.30<br>±0.22 | 63.60<br>±0.17 | 64.68<br>±0.09 | 58.41<br>±0.48 | 58.26<br>±0.09 | 61.34<br>±0.40 | 68.03<br>±0.15 | 67.68<br>±0.23 | 54.53<br>±0.25 | 61.52<br>±0.21 | 57.94<br>±0.21 | 57.94<br>±0.21 |
| 1099 |    | RoTTA      | 38.54<br>±0.22 | 39.85<br>±0.24 | 33.73<br>±0.37 | 63.45<br>±0.17 | 40.74<br>±0.32 | 62.03<br>±0.19 | 51.61<br>±0.06 | 51.75<br>±0.09 | 56.20<br>±0.08 | 65.14<br>±0.11 | 61.55<br>±0.10 | 51.22<br>±0.14 | 54.52<br>±0.22 | 42.50<br>±0.22 | 51.25<br>±0.22 | 42.50<br>±0.22 |
| 1100 |    | Tent       | 43.96<br>±0.85 | 45.42<br>±1.34 | 36.57<br>±1.57 | 62.28<br>±0.13 | 36.57<br>±2.97 | 59.96<br>±0.59 | 61.90<br>±0.48 | 53.25<br>±0.72 | 53.14<br>±1.70 | 57.36<br>±0.22 | 65.20<br>±0.20 | 60.14<br>±2.77 | 49.72<br>±0.08 | 57.62<br>±0.61 | 46.83<br>±0.52 | 52.66<br>±0.96 |
| 1101 |    | + SNAP-TTA | 49.06<br>±0.00 | 50.43<br>±0.13 | 41.49<br>±0.80 | 65.55<br>±0.24 | 44.09<br>±0.08 | 63.31<br>±0.53 | 65.62<br>±0.37 | 57.62<br>±0.09 | 56.81<br>±0.31 | 60.75<br>±0.41 | 68.72<br>±0.31 | 67.52<br>±0.64 | 54.08<br>±0.19 | 61.15<br>±0.41 | 51.54<br>±0.11 | 57.18<br>±0.20 |
| 1103 |    | CoTTA      | 34.31<br>±0.09 | 35.16<br>±0.46 | 31.42<br>±0.28 | 47.78<br>±0.45 | 34.99<br>±0.40 | 48.91<br>±0.46 | 47.79<br>±0.46 | 41.27<br>±0.67 | 41.42<br>±0.37 | 43.77<br>±0.57 | 52.16<br>      |                |                |                |                |                |

# C.3 CIFAR100-C

Table 10: STTA classification accuracy (%) comparing with and without SNAP-TTA on CIAFR100- C through Adaptation Rates(AR) (0.5, 0.3, and 0.1), including results for full adaptation (AR=1). Bold numbers are the highest accuracy.

|      | AR | Methods    | Gauss | Shots | Imp.  | Time  | GL2   | Moist | Zoo   | Arm   | Fog   | Brit  | Cont. | Elas. | Pfx.  | Avg   |       |       |
|------|----|------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1137 |    |            | 40.69 | 35.15 | 35.14 | 62.26 | 40.26 | 30.0  | 58.92 | 61.06 | 51.21 | 50.0  | 55.52 | 64.05 | 58.45 | 51.24 | 51.24 |       |
| 1138 |    | Tent       | ±0.35 | ±0.62 | ±0.38 | ±0.52 | ±0.23 | ±0.60 | ±0.43 | ±0.88 | ±0.31 | ±0.33 | ±0.62 | ±0.16 | ±0.80 | ±0.26 | ±0.69 | ±0.54 |
| 1139 |    | + SNAP-TTA | 42.87 | 34.87 | 37.06 | 65.01 | 42.22 | 62.22 | 42.72 | 54.03 | 53.68 | 58.03 | 67.05 | 58.03 | 52.97 | 57.47 | 46.94 | 54.13 |
| 1140 |    |            | ±0.37 | ±0.70 | ±0.08 | ±0.01 | ±0.35 | ±0.31 | ±0.45 | ±0.46 | ±0.39 | ±0.47 | ±0.50 | ±0.10 | ±0.15 | ±0.12 | ±0.13 | ±0.31 |
| 1141 |    | CoTtA      | 26.15 | 26.15 | 25.26 | 29.48 | 28.34 | 41.41 | 37.87 | 32.06 | 30.84 | 35.56 | 41.60 | 28.52 | 34.99 | 33.60 | 34.54 | 33.20 |
| 1142 |    |            | ±0.60 | ±0.32 | ±0.44 | ±0.71 | ±0.74 | ±0.76 | ±1.14 | ±0.85 | ±0.65 | ±1.12 | ±1.36 | ±0.79 | ±0.45 | ±0.82 | ±0.54 | ±0.75 |
| 1143 |    | + SNAP-TTA | 42.02 | 42.70 | 37.67 | 58.30 | 41.57 | 57.54 | 58.02 | 50.55 | 51.31 | 52.34 | 63.63 | 51.25 | 49.76 | 54.44 | 47.98 | 50.60 |
| 1144 |    |            | ±0.21 | ±0.13 | ±0.31 | ±0.26 | ±0.37 | ±0.14 | ±0.18 | ±0.27 | ±0.32 | ±0.17 | ±0.16 | ±0.49 | ±0.18 | ±0.05 | ±0.12 | ±0.25 |
| 1145 |    | EATA       | 38.46 | 39.05 | 33.47 | 61.07 | 38.52 | 58.16 | 39.09 | 49.60 | 49.18 | 54.41 | 63.15 | 57.06 | 49.09 | 52.87 | 42.49 | 49.8  |
| 1146 |    |            | ±0.14 | ±0.58 | ±0.23 | ±0.63 | ±0.29 | ±0.46 | ±0.48 | ±0.55 | ±0.47 | ±0.24 | ±0.43 | ±1.37 | ±0.88 | ±0.42 | ±0.34 | ±0.50 |
| 1147 |    | + SNAP-TTA | 40.49 | 41.64 | 34.37 | 64.28 | 40.38 | 61.52 | 63.17 | 51.66 | 52.12 | 56.50 | 66.03 | 62.01 | 51.76 | 55.66 | 44.83 | 52.45 |
| 1148 |    |            | ±0.21 | ±0.43 | ±0.16 | ±0.20 | ±0.51 | ±0.30 | ±0.18 | ±0.53 | ±0.52 | ±0.21 | ±0.36 | ±0.12 | ±0.12 | ±0.32 | ±0.22 |       |
| 1149 |    | SAR        | 40.28 | 41.62 | 35.35 | 62.84 | 40.37 | 59.51 | 61.68 | 51.29 | 50.66 | 55.60 | 64.43 | 58.49 | 50.90 | 54.82 | 44.64 | 51.50 |
| 1140 |    |            | ±0.07 | ±0.02 | ±0.04 | ±0.26 | ±0.41 | ±0.38 | ±0.28 | ±0.81 | ±0.38 | ±0.40 | ±0.62 | ±0.82 | ±0.64 | ±0.27 | ±0.43 | ±0.43 |
| 1141 |    | + SNAP-TTA | 41.76 | 44.24 | 36.89 | 64.34 | 41.54 | 62.13 | 63.39 | 53.24 | 52.91 | 57.54 | 66.89 | 62.41 | 52.70 | 57.23 | 46.63 | 53.59 |
| 1142 |    |            | ±0.29 | ±0.44 | ±0.21 | ±0.38 | ±0.37 | ±0.15 | ±0.24 | ±0.33 | ±0.02 | ±0.42 | ±0.60 | ±0.50 | ±0.15 | ±0.47 | ±0.57 |       |
| 1143 |    | RoTtA      | 36.38 | 37.38 | 31.78 | 61.44 |       |       |       |       |       |       |       |       |       |       |       |       |

**1171**

**1174 1175**

**1177**

**1179 1180 1181 1182 1183 1184** Tables [12,](#page-22-0) [13,](#page-22-1) [14,](#page-22-2) [15,](#page-22-3) and [16](#page-22-4) summarize the results for different combinations of CnDRM and IoBMN across these adaptation rates. The results indicate that the combination of CnDRM (Class and Domain Representative sampling) and IoBMN (inference using memory statistics corrected to match the test batch) consistently yields the highest accuracy. This trend is observed across all evaluated adaptation rates, suggesting that both components contribute significantly to enhancing adaptation performance.

**1185 1186 1187** Moreover, individual evaluations show that each component has a distinct positive effect, as evidenced by consistently higher accuracy compared to using no adaptation or only a single component. This emphasizes the complementary nature of CnDRM and IoBMN, which together provide

Table 11: STTA classification accuracy (%) comparing with and without SNAP-TTA on CIFAR100- C through Adaptation Rates(AR) (0.05, 0.03, and 0.01). Bold numbers are the highest accuracy.

# D ADDITIONAL RESULTS ON ABLATION STUDY

In this section, we provide additional details on the ablation study to evaluate the contributions of the CnDRM and IoBMN components in SNAP-TTA. Specifically, we measured the average accuracy across 15 corruption types on CIFAR10-C and CIFAR100-C datasets under varying adaptation rates (0.3, 0.1, 0.05) to thoroughly assess the effectiveness of each component.

**1224**

**1227**

**1229**

robust adaptation capabilities for domain-shifted scenarios. These tables provide further insight into the benefits of each configuration and how the synergy of CnDRM and IoBMN results in improved robustness against various corruptions.

Table 12: STTA classification accuracy (%) of ablative settings on the CIFAR10-C, adaptation rate 0.5. Averaged over all 15 corruptions. Bold numbers are the highest accuracy.

| Methods     | Tent  | CoTTA | EATA  | SAR   | RoTTA |
|-------------|-------|-------|-------|-------|-------|
| na¨ıve      | 78.86 | 69.75 | 79.02 | 77.83 | 75.39 |
| Random      | 78.90 | 66.04 | 78.97 | 77.77 | 75.06 |
| LowEntropy  | 78.68 | 63.74 | 78.42 | 76.21 | 72.83 |
| CRM         | 80.32 | 66.50 | 80.14 | 75.78 | 75.49 |
| CnDRM       | 79.62 | 77.68 | 79.63 | 78.22 | 75.85 |
| CnDRM+EMA   | 80.96 | 72.42 | 80.27 | 78.19 | 76.73 |
| CnDRM+IoDMN | 81.23 | 78.75 | 81.30 | 79.77 | 77.41 |

Table 13: STTA classification accuracy (%) of ablative settings on the CIFAR10-C, adaptation rate 0.05. Averaged over all 15 corruptions. Bold numbers are the highest accuracy.

| Methods     | Tent  | CoTTA | EATA  | SAR   | RoTTA |
|-------------|-------|-------|-------|-------|-------|
| na¨ıve      | 75.75 | 67.22 | 75.55 | 75.25 | 74.80 |
| Random      | 75.82 | 65.90 | 75.56 | 75.27 | 74.91 |
| LowEntropy  | 74.07 | 64.08 | 73.73 | 73.58 | 72.83 |
| CRM         | 76.55 | 66.14 | 76.06 | 74.02 | 75.23 |
| CnDRM       | 76.53 | 77.67 | 76.29 | 76.18 | 75.61 |
| CnDRM+EMA   | 76.86 | 71.69 | 75.98 | 75.43 | 75.95 |
| CnDRM+IoDMN | 77.93 | 78.73 | 77.76 | 77.21 | 77.05 |

Table 14: STTA classification accuracy (%) of ablative settings on the CIFAR100-C, adaptation rate 0.3. Averaged over all 15 corruptions. Bold numbers are the highest accuracy.

| Methods     | Tent  | CoTTA | EATA  | SAR   | RoTTA |
|-------------|-------|-------|-------|-------|-------|
| na¨ıve      | 53.36 | 39.11 | 49.97 | 56.65 | 49.84 |
| Random      | 53.00 | 33.49 | 49.24 | 56.06 | 49.00 |
| LowEntropy  | 53.53 | 32.29 | 45.51 | 55.84 | 44.77 |
| CRM         | 54.21 | 32.86 | 47.42 | 56.40 | 46.68 |
| CnDRM       | 55.15 | 50.02 | 51.36 | 57.72 | 50.74 |
| CnDRM+EMA   | 55.39 | 41.34 | 50.11 | 57.68 | 49.88 |
| CnDRM+IoDMN | 57.27 | 50.32 | 52.19 | 58.44 | 51.55 |

Table 15: STTA classification accuracy (%) of ablative settings on the CIFAR100-C, adaptation rate 0.1. Averaged over all 15 corruptions. Bold numbers are the highest accuracy.

| Methods     | Tent  | CoTTA | EATA  | SAR   | RoTTA |
|-------------|-------|-------|-------|-------|-------|
| na¨ıve      | 52.84 | 35.86 | 49.70 | 53.49 | 49.11 |
| Random      | 52.68 | 33.18 | 49.39 | 53.42 | 48.84 |
| LowEntropy  | 51.76 | 32.30 | 46.03 | 52.15 | 45.18 |
| CRM         | 52.43 | 32.54 | 47.68 | 53.12 | 47.01 |
| CnDRM       | 54.46 | 50.06 | 51.41 | 55.24 | 50.47 |
| CnDRM+EMA   | 54.36 | 41.63 | 50.21 | 54.84 | 49.95 |
| CnDRM+IoDMN | 55.84 | 50.52 | 52.35 | 55.76 | 51.33 |

Table 16: STTA classification accuracy (%) of ablative settings on the CIFAR100-C, adaptation rate 0.05. Averaged over all 15 corruptions. Bold numbers are the highest accuracy.

| Methods     | Tent  | CoTTA | EATA  | SAR   | RoTTA |
|-------------|-------|-------|-------|-------|-------|
| na¨ıve      | 51.24 | 33.20 | 49.81 | 51.50 | 49.12 |
| Random      | 51.35 | 33.71 | 49.57 | 51.48 | 48.98 |
| LowEntropy  | 49.79 | 32.36 | 46.65 | 49.51 | 45.41 |
| CRM         | 50.17 | 32.74 | 47.47 | 50.49 | 46.58 |
| CnDRM       | 52.86 | 50.08 | 51.47 | 53.09 | 50.44 |
| CnDRM+EMA   | 52.68 | 41.43 | 50.32 | 52.80 | 50.04 |
| CnDRM+IoDMN | 54.13 | 50.63 | 52.43 | 53.59 | 51.41 |

**1267**

**1281**

**1284**

**1287**

#### E ADDITIONAL ABLATE ANALYSIS

#### E.1 DOMAIN INFLUENCE IN EARLY LAYER REPRESENTATIONS

![](_page_23_Figure_4.jpeg)

Figure 6: PCA embedding of early layer features for one domain from each of the four main CIFAR10-C corruption categories, showing clear sep-

aration between domains. Visualizing early layer feature embeddings using 2D PCA on CIFAR-10C domains reveals distinct domain-specific patterns, highlighting the significant influence of domain information in these representations (Figure [6\)](#page-23-3). Our preliminary experiments further confirm that sparse TTA, using the Wasserstein distance between moving batch normalization statistics and instance-specific statistics derived from early layer hidden features, can significantly improve performance. Selecting instances closer to the target domain distribution center using this distance metric yields better adaptation results, as demonstrated by performance comparisons between the top 20% and bottom 20% of samples (Figure [3\)](#page-4-0). These findings emphasize the crucial role of domain-sensitive early layers in achieving effective adaptation.

In deep learning models, early layers capture low-level features such as textures, edges, and frequency components [\(Zeiler & Fer](#page-12-8)[gus, 2014\)](#page-12-8). These features are inherently domain-specific, making these layers more sensitive to shifts in input data distribution—a critical challenge for tasks requiring domain adaptation and generalization [\(Lee et al., 2018;](#page-11-3) [Segu et al., 2023\)](#page-12-9). This sensitivity arises because early layers encapsulate domain-specific patterns that may not generalize to new distributions. Under the covariate shift assumption [\(Quinonero-Candela et al., 2008\)](#page-12-0), while input distribu- ˜ tions differ between source and target domains, the conditional distribution of labels remains the same. This discrepancy between input distributions makes early layers particularly vulnerable to domain shifts.

#### E.2 ANALYSIS ON CONFIDENCE THRESHOLD ON PSEUDO-LABEL ACCURACY

We analyzed the impact of using a confidence threshold for pseudo-label selection by comparing random sampling with high-confidence sampling across three benchmarks: CIFAR10-C, CIFAR100-C, and ImageNet-C. Table [17](#page-23-4) shows that high-confidence sampling consistently outperformed random sampling, achieving significantly higher pseudo-label accuracy in all datasets. This result demonstrates the effectiveness of selecting high-confidence samples to improve the quality of pseudolabels, thereby enhancing model adaptation under domain shift conditions.

Table 17: Pseudo-label accuracy comparison between random and high-confidence sampling on three benchmakrs: CIFAR10-C, CIFAR100-C, and ImageNet-C. Bold numbers are the highest accuracy.

|                 | CIFAR10-C    | CIFAR100-C   | ImageNet-C   |
|-----------------|--------------|--------------|--------------|
| Random          | 69.91        | 45.30        | 23.90        |
| <b>HighConf</b> | <b>74.80</b> | <b>59.38</b> | <b>59.40</b> |

#### E.3 LATENCY TRACKING OF SNAP-TTA ON DIVERSE EDGE-DEVICES

To evaluate the latency efficiency of SNAP-TTA on resource-constrained edge devices, we measured the adaptation latency across three devices: NVIDIA Jetson Nano [\(NVIDIA Corporation,](#page-11-13) [2019\)](#page-11-13), Raspberry Pi 4 [\(Raspberry Pi Foundation, 2019\)](#page-12-6), and Raspberry Pi Zero 2 W [\(Raspberry Pi](#page-12-14) [Foundation, 2021\)](#page-12-14). These experiments compared the latency of SNAP-TTA with the Original TTA framework, specifically focusing on five state-of-the-art TTA algorithms: Tent [\(Wang et al., 2021\)](#page-12-1), EATA [\(Niu et al., 2022\)](#page-11-0), SAR [\(Niu et al., 2023\)](#page-11-1), RoTTA [\(Yuan et al., 2023\)](#page-12-3), and CoTTA [\(Wang](#page-12-2) [et al., 2022\)](#page-12-2). The experiments were conducted at an adaptation rate of 0.1, demonstrating the effectiveness of SNAP-TTA in reducing adaptation latency while maintaining competitive accuracy.

![](_page_24_Figure_1.jpeg)

**1317**

**1319**

**1321**

**1324**

**1334**

RoTTA SAR EATA Tent Original TTA SNAP-TTA Figure 7: Latency comparison between SNAP-TTA and Original TTA across five state-of-the-art TTA algorithms (Tent, EATA, SAR, RoTTA, CoTTA) on three edge devices: (a) NVIDIA Jetson Nano, (b) Raspberry Pi 4, and (c) Raspberry Pi Zero 2 W. SNAP-TTA demonstrates significant latency reductions while maintaining competitive adaptation performance. The experiments were conducted at an adaptation rate of 0.1.

Average Latency per Batch (s) Figure [7](#page-24-1) illustrates the latency performance for each device. It is evident that SNAP-TTA achieves a significant reduction in adaptation latency compared to the Original TTA framework. Notably, the latency reduction was proportional to the adaptation rate, validating the efficiency of SNAP-TTA in sparse adaptation scenarios. For instance, the latency for CoTTA was reduced by up to 87.5% on the Raspberry Pi 4, emphasizing the practical benefits of SNAP-TTA in latency-sensitive environments. Additionally, similar trends were observed across other devices, including the resourcelimited Raspberry Pi Zero 2 W.

The results confirm that SNAP-TTA not only ensures substantial latency reductions but also adapts effectively to real-world conditions on diverse edge devices, proving its suitability for deployment in latency-sensitive applications.

### E.4 MEMORY OVERHEAD OF SNAP-TTA

The SNAP-TTA framework achieves substantial latency reduction and accuracy improvements with minimal memory overhead, even under resource-constrained scenarios like edge devices. In this section, we present both a theoretical analysis of the memory requirements and empirical results obtained from evaluations on a Raspberry Pi 4[\(Raspberry Pi Foundation, 2019\)](#page-12-6) (CPU-only edge device).

The memory overhead of SNAP-TTA arises from two main components: (1) the memory buffer in Class and Domain Representative Memory (CnDRM) for storing representative samples, including both feature statistics (mean and variance) and the raw image samples, and (2) the statistics required for Inference-only Batch-aware Memory Normalization (IoBMN). For a batch size B, the total theoretical memory overhead can be expressed as: Memory Overhead = B × (Image Size + 2 × Feature Dimension × Bytes per Value)+Feature Dimension×Bytes per Value× 2. The last term accounts for the storage of IoBMN statistics (mean and variance for each feature channel). The image size is calculated based on the dataset resolution and data type.

For ResNet18 on CIFAR10-C, CIFAR10 images have a resolution of 32 × 32 × 3 with each value stored as 1 byte. For a feature dimension of 512 and batch size B = 16, the total overhead is: Image Overhead = 16×(32×32×3×1) = 49, 152 bytes (48 KB), Feature Overhead (CnDRM) = 16 × (512 × 2 × 4) = 65, 536 bytes (64 KB), Feature Overhead (IoBMN) = 512 × 2 × 4 = 4, 096 bytes (4 KB). Thus, the total memory overhead is: Total Overhead = 48 KB + 64 KB + 4 KB = 116 KB.

For ResNet50 on ImageNet-C, ImageNet images have a resolution of 224 × 224 × 3, stored as 1 byte per value. For a feature dimension of 2048 and batch size B = 16, the total overhead is: Image Overhead = 16 × (224 × 224 × 3 × 1) = 12, 044, 928 bytes (11.5 MB), Feature Overhead (CnDRM) = 16 × (2048 × 2 × 4) = 262, 144 bytes (256 KB), Feature Overhead (IoBMN) = 2048 × 2 × 4 = 16, 384 bytes (16 KB). Thus, the total memory overhead is: Total Overhead = 11.5 MB + 256 KB + 16 KB ≈ 11.77 MB.

**1354**

**1371**

**1374**

Table 18 shows the empirical memory usage of SNAP-TTA compared to Original TTA methods (Tent, EATA, CoTTA, SAR, and RoTTA). The results were averaged across three seeds of experiments and represent the memory footprint observed in a CPU-only edge device, Raspberry Pi 4. While minor variations in measurements are expected due to the nature of CPU memory footprint tracking, the results robustly indicate that the actual memory overhead of SNAP-TTA on edge devices is extremely low across all algorithms, ranging from 0.02% to 1.74%. Furthermore, while peak memory usage is either slightly increased or remains comparable to Original TTA methods, the average memory usage of SNAP-TTA is consistently lower. This is because SNAP-TTA performs backpropagation infrequently, which is the most memory-intensive operation in TTA.

Table 18: Comparison of memory usage (Average Memory, Peak Memory, and Memory Overhead) between Original TTA and SNAP-TTA (adaptation rate 0.3) across various methods (Tent, EATA, CoTTA, SAR, and RoTTA) tested on Raspberry Pi 4. Bold numbers are the lowest memory usage.

| Methods | Average Original TTA | Mem (MB) SNAP-TTA | Peak Mem Original TTA | (MB) SNAP-TTA | Mem SNAP | Overhead (MB) Original |
|---------|----------------------|-------------------|-----------------------|---------------|----------|------------------------|
| Tent    | 764.24               | 751.35            | 822.93                | 828.46        | 5.52     | (0.67%)                |
| CoTTA   | 1133.52              | 1099.64           | 1211.21               | 1227.99       | 16.78    | (1.13%)                |
| EATA    | 816.69               | 749.95            | 847.73                | 862.51        | 14.78    | (1.74%)                |
| SAR     | 786.65               | 753.69            | 863.77                | 865.18        | 1.41     | (0.02%)                |
| RoTTA   | 933.23               | 871.64            | 972.23                | 983.94        | 11.71    | (1.20%)                |

These findings demonstrate that SNAP-TTA's memory overhead is negligible compared to its benefits in latency reduction and accuracy improvements. By leveraging a small memory buffer for representative samples and minimizing backpropagation operations, SNAP-TTA not only achieves a lightweight memory profile but also becomes more efficient in terms of average memory usage compared to Original TTA. This lightweight design, combined with its advantages in latency and accuracy, underscores the practicality of SNAP-TTA for deployment in latency-sensitive applications on edge devices.

#### E.5 INTEGRATION OF SNAP-TTA WITH MEMORY-EFFICIENT TTA ALGORITHM: MECTA (H[ONG ET AL](#page-10-9)., [2023\)](#page-10-9)

This section evaluates the integration of SNAP-TTA with MECTA, a memory-efficient TTA algorithm, to demonstrate its applicability for resource-constrained edge devices. The experimental setup follows the evaluation settings presented in the MECTA paper to ensure a fair and consistent comparison. Specifically, we analyze the performance of Tent and EATA, enhanced with MECTA and further integrated with SNAP-TTA, using the ResNet50 model with a batch size of 64 on the ImageNet-C dataset.

Table [19](#page-25-1) presents the classification accuracy and peak memory usage for Tent+MECTA and EATA+MECTA configurations with and without SNAP-TTA. Integrating SNAP-TTA with Tent+MECTA improves accuracy from 35.21% to 39.52%, while reducing peak memory usage by approximately 30% compared to the Tent baseline. Similarly, SNAP-TTA boosts the accuracy of EATA+MECTA from 35.55% to 42.86% while maintaining an efficient memory footprint.

Table 19: Comparison of classification (%) and memory peak (MB) in STTA with an adaptation rate of 0.1. MECTA significantly reduces memory consumption, and SNAP-TTA is applied alongside it to boost the performance of sparse adaptation. The accuracy is the average over 15 corruptions in ImageNet-C. Bold numbers indicate either the lowest memory usage or the highest accuracy.

| Methods            | Accuracy (%) | Max Memory (MB)   |
|--------------------|--------------|-------------------|
| Tent               | 35.21        | 6805.26           |
| +MECTA             | 37.62        | 4620.25 (-32.10%) |
| + MECTA + SNAP-TTA | 39.52        | 4622.12 (-32.08%) |
| EATA               | 35.55        | 6541.02           |
| +MECTA             | 41.41        | 4512.38 (-31.01%) |
| + MECTA + SNAP-TTA | 42.86        | 4535.44 (-30.66%) |

**1447 1448 1449 1450 1451 1452** To address this issue efficiently, our Class and Domain Representative Memory (CnDRM) recalculates the distance of samples only when the shift in the domain centroid exceeds a predefined significance threshold. Specifically, if the change in the domain centroid ∆cdomain surpasses a threshold τ∆, the distances of all samples in memory are updated to reflect the new domain conditions. This threshold-based approach ensures that recalculations occur only when necessary, thereby minimizing computational costs while maintaining the representativeness of the memory.

**1453 1454 1455 1456 1457** In practice, we observed that the performance was not significantly affected as long as the threshold τ<sup>∆</sup> was not set too high, indicating robustness to the choice of threshold. Based on these observations, we set τ<sup>∆</sup> = 0.1 and used this value consistently for all evaluations. By focusing recalculations on significant shifts, this strategy preserves consistency in sample selection, ensuring that both older and newer samples are compared fairly in the context of the current domain characteristics without excessive computational overhead.

Further details are provided in Table [20,](#page-26-0) which evaluates the combination of SNAP-TTA with MECTA across various corruption types and adaptation rates (AR = 0.3, 0.1, and 0.05). These results show that SNAP-TTA consistently outperforms baseline configurations across all adaptation rates and corruption types. This demonstrates the robustness of SNAP-TTA when integrated with MECTA and its suitability for real-world applications.

By adhering to the evaluation settings of the MECTA paper, this study ensures high reliability and comparability of results. The findings confirm that SNAP-TTA is highly compatible with MECTA, significantly improving both accuracy and memory efficiency. This synergy highlights the potential of combining SNAP-TTA and MECTA for deployment in resource-constrained environments such as edge devices.

Table 20: Evaluation of SNAP-TTA with MECTA on ImageNet-C through Adaptation Rates(AR) (0.3, 0.1, and 0.05). Bold numbers are the highest accuracy.

| AR Methods Tent + MECTA + SNAP-TTA 0.3 EATA + MECTA | Gau. 28.20 ±0.30 30.49 ±0.26 32.18 ±0.60 | Shot 30.13 ±0.41 31.98 ±0.14 34.85 ±0.49 | Imp. 29.58 ±0.08 31.66 ±0.21 33.06 ±0.31 | Def. 23.07 ±0.22 26.29 ±0.32 28.80 ±0.22 | Gla. 23.35 ±0.47 26.19 ±0.02 29.18 ±0.18 | Mot. 34.49 ±0.13 38.47 ±0.30 41.02 ±0.26 | Zoom 45.95 ±0.13 47.38 ±0.11 49.24 ±0.08 | Snow 40.97 ±0.15 43.79 ±0.11 47.10 ±0.20 | Fro. 35.68 ±0.41 40.12 ±0.12 41.56 ±0.25 | Fog 55.66 ±0.04 56.38 ±0.05 57.35 ±0.12 | Brit. 66.56 ±0.06 66.81 ±0.07 66.27 ±0.05 | Cont. 14.72 ±0.47 28.87 ±0.28 34.56 ±0.12 | Elas. 53.09 ±0.18 53.53 ±0.09 55.38 ±0.10 | Pix. 57.16 ±0.05 57.61 ±0.10 58.19 ±0.04 | JPEG 50.74 ±0.15 50.86 ±0.08 52.87 ±0.26 | Avg. 39.29 ±0.22 42.03 ±0.15 44.11 ±0.22 |
|-----------------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|-----------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|------------------------------------------|------------------------------------------|------------------------------------------|
|                                                     | 33.67                                    | 35.76                                    | 34.86                                    | 30.35                                    | 30.29                                    | 42.78                                    | 49.55                                    | 47.46                                    | 42.32                                    | 57.50                                   | 66.18                                     | 39.08                                     | 55.38                                     | 58.35                                    | 52.72                                    | 45.08                                    |
| + SNAP-TTA                                          | ±0.19                                    | ±0.24                                    | ±0.10                                    | ±0.11                                    | ±0.04                                    | ±0.06                                    | ±0.10                                    | ±0.10                                    | ±0.05                                    | ±0.15                                   | ±0.06                                     | ±0.81                                     | ±0.16                                     | ±0.12                                    | ±0.02                                    | ±0.15                                    |
| Tent + MECTA                                        | 24.94 ±0.15                              | 26.73 ±0.20                              | 25.63 ±0.07                              | 21.11 ±0.22                              | 21.46 ±0.18                              | 32.11 ±0.02                              | 44.05 ±0.19                              | 38.22 ±0.27                              | 36.36 ±0.09                              | 53.92 ±0.12                             | 66.48 ±0.02                               | 18.50 ±0.45                               | 50.80 ±0.12                               | 55.67 ±0.18                              | 48.33 ±0.11                              | 37.62 ±0.16                              |
| + SNAP-TTA                                          | 27.49 ±0.08                              | 28.90 ±0.14                              | 28.26 ±0.16                              | 23.49 ±0.17                              | 23.76 ±0.12                              | 34.92 ±0.06                              | 45.18 ±0.13                              | 40.21 ±0.09                              | 38.40 ±0.18                              | 53.78 ±0.14                             | 66.54 ±0.03                               | 27.72 ±0.20                               | 51.00 ±0.20                               | 55.48 ±0.13                              | 47.61 ±0.17                              | 39.52 ±0.13                              |
| EATA + MECTA                                        | 29.42 ±0.67                              | 31.72 ±0.30                              | 29.44 ±0.32                              | 24.41 ±0.74                              | 25.48 ±0.45                              | 37.04 ±0.18                              | 47.10 ±0.15                              | 43.60 ±0.19                              | 39.43 ±0.38                              | 55.95 ±0.13                             | 66.42 ±0.14                               | 28.85 ±1.18                               | 53.70 ±0.15                               | 57.34 ±0.15                              | 51.20 ±0.36                              | 41.41 ±0.37                              |
|                                                     | 31.26                                    | 32.71                                    | 32.22                                    | 27.31                                    | 27.61                                    | 38.88                                    | 47.83                                    | 44.52                                    | 40.58                                    | 56.42                                   | 66.24                                     | 35.38                                     | 53.67                                     | 57.39                                    | 50.83                                    | 42.86                                    |
| + SNAP-TTA                                          | ±0.11                                    | ±0.17                                    | ±0.17                                    | ±0.46                                    | ±0.28                                    | ±0.28                                    | ±0.09                                    | ±0.14                                    | ±0.05                                    | ±0.06                                   | ±0.21                                     | ±0.63                                     | ±0.17                                     | ±0.13                                    | ±0.12                                    | ±0.20                                    |
| Tent + MECTA                                        | 21.22 ±0.13                              | 23.19 ±0.22                              | 21.90 ±0.13                              | 18.69 ±0.18                              | 19.39 ±0.20                              | 29.89 ±0.13                              | 42.02 ±0.10                              | 36.53 ±0.22                              | 35.23 ±0.05                              | 51.75 ±0.15                             | 66.23 ±0.04                               | 19.64 ±0.27                               | 48.43 ±0.03                               | 53.54 ±0.13                              | 45.43 ±0.11                              | 35.54 ±0.14                              |
| + SNAP-TTA                                          | 23.93 ±0.27                              | 25.37 ±0.22                              | 24.10 ±0.15                              | 20.42 ±0.18                              | 21.14 ±0.07                              | 31.83 ±0.06                              | 42.68 ±0.04                              | 37.53 ±0.16                              | 36.31 ±0.20                              | 51.42 ±0.17                             | 66.19 ±0.04                               | 23.84 ±0.24                               | 48.62 ±0.05                               | 53.20 ±0.17                              | 44.57 ±0.17                              | 36.74 ±0.15                              |
| EATA + MECTA                                        | 24.97 ±0.42                              | 26.95 ±0.27                              | 21.87 ±3.29                              | 21.19 ±0.90                              | 21.94 ±0.45                              | 33.61 ±0.08                              | 45.11 ±0.11                              | 40.92 ±0.19                              | 37.73 ±0.42                              | 54.64 ±0.10                             | 66.60 ±0.07                               | 23.03 ±0.59                               | 51.87 ±0.35                               | 56.60 ±0.25                              | 49.15 ±0.23                              | 38.41 ±0.51                              |
|                                                     | 28.39                                    | 30.10                                    | 29.45                                    | 24.32                                    | 25.12                                    | 35.54                                    | 46.04                                    | 41.87                                    | 39.16                                    | 55.12                                   | 66.61                                     | 30.34                                     | 52.06                                     | 56.42                                    | 49.11                                    | 40.64                                    |
| + SNAP-TTA                                          | ±0.57                                    | ±0.38                                    | ±0.22                                    | ±0.20                                    | ±0.07                                    | ±0.20                                    | ±0.27                                    | ±0.07                                    | ±0.15                                    | ±0.01                                   | ±0.09                                     | ±0.34                                     | ±0.24                                     | ±0.11                                    | ±0.07                                    | ±0.20                                    |

# F ADDITIONAL DISCUSSIONS

#### F.1 EFFICIENT STRATEGY FOR RE-CALCULATION OF SAMPLE'S DISTANCE

The domain centroid in our framework is updated using a momentum-based approach to effectively capture recent shifts in the target domain. This ensures that the centroid remains adaptive to evolving distributions without being overly influenced by temporary fluctuations. However, during sparse adaptation (SA), where model updates occur at extended intervals, the data distribution can shift substantially between updates. Consequently, distances calculated for older samples may become outdated, leading to inconsistencies when comparing them to more recently added samples that are evaluated based on the updated centroid.

**1465 1466 1467 1468 1469** Instead of employing additional mechanisms like z-score evaluation to detect domain shifts, we rely on the natural adaptability of the centroid to adjust to the incoming data. This simplifies the design and avoids unnecessary overhead while maintaining robustness. As the domain characteristics evolve, the centroid continuously aligns with the new domain without requiring explicit detection of changes or manual intervention.

| 1478 AR Method 1479 Tent 1480 + SNAP-TTA 1481 0.1 CoTTA 1482 | Gau. 24.68 ±0.45 28.71 ±0.66 10.99 ±0.40 | Shot 19.65 ±1.27 30.60 ±1.82 12.21 ±0.04 | Imp. 5.12 ±1.22 22.91 ±2.25 11.54 ±0.30 | Def. 0.63 ±0.05 6.13 ±0.90 11.28 ±0.13 | Gla. 0.43 ±0.02 1.62 ±0.20 11.13 ±0.15 | Mot. 0.40 ±0.04 0.87 ±0.13 22.08 ±0.07 | Zoom 0.44 ±0.06 0.88 ±0.07 34.80 ±0.18 | Snow 0.41 ±0.03 0.64 ±0.08 30.69 ±0.10 | Fro. 0.30 ±0.03 0.64 ±0.06 29.45 ±0.04 | Fog 0.33 ±0.04 0.66 ±0.05 43.87 ±0.19 | Brit. 0.42 ±0.05 0.75 ±0.01 61.92 ±0.09 | Cont. 0.24 ±0.04 0.44 ±0.05 12.76 ±0.16 | Elas. 0.32 ±0.02 0.60 ±0.08 40.03 ±0.13 | Pix. 0.31 ±0.05 0.63 ±0.07 44.99 ±0.14 | JPEG 0.31 ±0.04 0.61 ±0.07 36.43 ±0.16 | Avg. 3.60 ±0.23 6.45 ±0.43 27.61 ±0.15 |
|--------------------------------------------------------------|------------------------------------------|------------------------------------------|-----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|---------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|
|                                                              | 15.19                                    | 15.97                                    | 15.91                                   | 13.94                                  | 14.18                                  | 24.76                                  | 36.50                                  | 32.61                                  | 31.76                                  | 46.14                                 | 63.60                                   | 15.60                                   | 42.17                                   | 46.77                                  | 38.08                                  | 30.21                                  |
| + SNAP-TTA                                                   | ±0.17                                    | ±0.11                                    | ±0.02                                   | ±0.04                                  | ±0.03                                  | ±0.07                                  | ±0.23                                  | ±0.04                                  | ±0.06                                  | ±0.10                                 | ±0.14                                   | ±0.04                                   | ±0.02                                   | ±0.06                                  | ±0.12                                  | ±0.08                                  |
| Tent                                                         | 23.31 ±0.37                              | 27.08 ±1.13                              | 22.71 ±2.50                             | 9.72 ±3.35                             | 4.14 ±3.00                             | 2.03 ±1.53                             | 1.16 ±0.75                             | 0.66 ±0.22                             | 0.45 ±0.12                             | 0.47 ±0.09                            | 0.61 ±0.16                              | 0.33 ±0.09                              | 0.47 ±0.08                              | 0.47 ±0.08                             | 0.46 ±0.07                             | 6.27 ±0.90                             |
| + SNAP-TTA                                                   | 27.10 ±0.23                              | 33.41 ±0.10                              | 31.78 ±0.62                             | 19.85 ±0.79                            | 16.94 ±1.50                            | 14.75 ±2.53                            | 12.46 ±4.27                            | 5.53 ±2.30                             | 2.69 ±1.18                             | 1.47 ±0.49                            | 1.52 ±0.40                              | 0.67 ±0.09                              | 0.88 ±0.10                              | 0.89 ±0.10                             | 0.84 ±0.07                             | 11.39 ±0.98                            |
| CoTTA                                                        | 11.04 ±0.38                              | 12.25 ±0.39                              | 11.73 ±0.42                             | 11.62 ±0.10                            | 11.25 ±0.59                            | 22.05 ±0.13                            | 34.89 ±0.13                            | 30.73 ±0.20                            | 29.50 ±0.17                            | 44.09 ±0.18                           | 61.87 ±0.09                             | 12.87 ±0.18                             | 40.15 ±0.17                             | 45.06 ±0.19                            | 36.53 ±0.14                            | 27.71 ±0.23                            |
|                                                              | 15.20                                    | 15.89                                    | 15.93                                   | 13.81                                  | 14.15                                  | 24.74                                  | 36.68                                  | 32.51                                  | 31.71                                  | 46.11                                 | 63.48                                   | 15.73                                   | 42.20                                   | 46.69                                  | 38.05                                  | 30.19                                  |
| + SNAP-TTA                                                   | ±0.15                                    | ±0.02                                    | ±0.10                                   | ±0.04                                  | ±0.03                                  | ±0.16                                  | ±0.27                                  | ±0.04                                  | ±0.20                                  | ±0.05                                 | ±0.09                                   | ±0.19                                   | ±0.12                                   | ±0.10                                  | ±0.04                                  | ±0.10                                  |

#### F.2 STRATEGY FOR CONTINUOUS DOMAIN SHIFT SETTING

In our proposed framework, the centroid used for selecting domain-representative samples naturally adapts to changes in the domain as new data is encountered. This mechanism inherently ensures that the centroid evolves to reflect the characteristics of the current domain, allowing for effective performance even under continual Test-Time Adaptation (TTA) scenarios, where the domain may gradually or abruptly shift during adaptation.

To validate the effectiveness of SNAP-TTA under continual domain shift scenarios, we conducted experiments across various benchmark datasets with incremental and abrupt domain shifts. Table [21](#page-27-2) summarizes the results, demonstrating that SNAP-TTA maintains strong performance across evolving domains without requiring additional computational overhead for explicit domain shift detection.

Table 21: Performance of SNAP-TTA under continual domain shift scenarios. The table reports the accuracy (%) for different datasets with incremental and abrupt shifts. Bold numbers are the highest accuracy.

These results indicate that SNAP-TTA effectively handles both incremental and abrupt domain shifts, consistently outperforming baseline methods. By leveraging the natural adaptability of the centroid, SNAP-TTA provides a robust solution for continual domain adaptation in real-world scenarios. Notably, SNAP-TTA mitigates catastrophic forgetting not only through its sparse adaptation strategy but also by leveraging domain centroid-based sampling, allowing performance to be sustained longer in continual shift scenarios. Unlike Tent, CoTTA is specifically designed for continual domain shift environments, which highlights its superior performance under such conditions.

Future work could explore augmenting this adaptive mechanism by incorporating techniques like z-score evaluation to enable even more responsive adjustments. For instance, a z-score-based approach could further refine the centroid's responsiveness to subtle, gradual domain shifts by monitoring discrepancies between incoming data statistics and the current centroid. Such enhancements could make the system even more effective at handling continual domain evolution, particularly in scenarios with complex or noisy data streams.

#### F.3 MODIFICATION FOR LAYER NORMALIZATION OF VIT

The main text describes the use of Batch Normalization (BN) statistics for calculating domain centroids and centroid-instance distances, with subsequent adjustment of memory statistics to match the target test batch using the Inference-only Batch-aware Memory Normalization (IoBMN) method. Specifically, these calculations leverage the mean and variance across batches as follows:

$$\bar{\mu}_c = \frac{1}{B \times L} \sum_{b=1}^B \sum_{l=1}^L f_{b,c,l}, \quad \bar{\sigma}_c^2 = \frac{1}{B \times L} \sum_{b=1}^B \sum_{l=1}^L (f_{b,c,l} - \mu_{b,c})^2, \quad (6)$$

**1517**

**1519**

**1521**

**1534**

**1554**

However, modern models like Vision Transformer (ViT) utilize Layer Normalization (LN) instead of BN. Unlike BN, which calculates statistics across the entire batch, LN normalizes each instance independently by using the statistics calculated over individual feature dimensions. Specifically, for a feature vector f<sup>b</sup> belonging to the b-th instance, LN computes:

$$\mu_b = \frac{1}{C} \sum_{c=1}^C f_{b,c}, \quad \sigma_b^2 = \frac{1}{C} \sum_{c=1}^C (f_{b,c} - \mu_b)^2, \quad (7)$$

where C is the number of channels. This difference implies that LN operates without batch-level interactions, focusing solely on within-instance normalization, which makes the method inherently more suitable for handling variable batch sizes, particularly in latency-sensitive applications like those considered in our Test-Time Adaptation (TTA) setting.

Despite the differences between BN and LN, the fundamental mechanism of using feature statistics to capture domain information remains valid. The key domain characteristics in early layer features are preserved in both normalization types, enabling the construction of a domain centroid that reflects the distributional characteristics of the test data. For LN, this centroid can be computed by aggregating across instances instead of across batches:

$$\bar{\mu}_c^{\text{LN}} = \frac{1}{M} \sum_{b=1}^M \mu_b, \quad \bar{\sigma}_c^{\text{2LN}} = \frac{1}{M} \sum_{b=1}^M \sigma_b^2, \quad (8)$$

where M is memory capacity. This modified approach allows the domain centroid to still represent the overall domain-specific characteristics effectively, despite the lack of direct batch-level statistics.

Furthermore, this methodology extends seamlessly to other normalization layers, such as Group Normalization (GN). In GN, the statistics are computed across smaller groups of channels within each instance, but the procedure for aggregating these statistics to form a domain centroid remains the same—by averaging the group-level statistics across instances.

To maintain the core concept of selecting domain-representative samples with minimal modifications, we continue to use the memory of high-confidence domain-representative samples in the Inference-only Batch-aware Memory Normalization (IoBMN) strategy. The adjustment for LN requires: 1. Calculating LN-specific centroids as described in Equation [8.](#page-28-0) 2. Replacing BN statistics with LN statistics in the IoBMN module, thereby aligning the feature normalization during inference with the domain-representative information derived from memory.

The effectiveness of this modification was validated experimentally, as shown in Table [5,](#page-9-0) where ViT models using LN showed improved performance even under sparse TTA conditions. This indicates that, with minimal adjustments, SNAP-TTA remains effective for ViT with LN. The core principle of utilizing domain-representative statistics for aligning test-time feature distributions continues to provide significant benefits, ensuring robust adaptation in shifting domains with limited latency and computational overhead.

#### F.4 IMPACT OF MEMORY SIZE ON SNAP-TTA PERFORMANCE

The memory size of the Class and Domain Representative Memory (CnDRM) in SNAP-TTA has implications for both performance and privacy. Increasing memory size allows storing more samples, which intuitively could improve adaptation. However, such an approach raises privacy concerns and needs additional memory and latency when storing sensitive samples. To evaluate the trade-off, we conducted experiments on ImageNet-C under Gaussian noise corruption, using Tent + SNAP-TTA(adaptation rate 0.3) with a batch size of 16 and varying the memory size.

Table 22: Performance comparison with varying memory sizes on ImageNet-C (Gaussian noise).

| Memory Size | Accuracy (%) |
|-------------|--------------|
| 16 (Base)   | 26.60        |
| 32          | 28.44        |
| 64          | 28.89        |
| 128         | 28.60        |

As shown in Table [22,](#page-28-1) increasing the memory size beyond the base configuration of 16 does not lead to significant performance gains. This observation highlights the efficiency of SNAP-TTA's representative sampling strategy, which prioritizes storing samples based on proximity to class and domain centroids. The saturation in accuracy suggests that a carefully aligned memory size to the batch size is sufficient to balance computational efficiency, performance, and privacy considerations.

**1571**

**1574**

In conclusion, to minimize computational overhead while ensuring robust test-time adaptation, the memory size in SNAP-TTA is designed to align with the batch size. This configuration addresses privacy and memory overhead risks by limiting the number of stored samples without compromising adaptation effectiveness.

#### F.5 EFFECT OF LEARNING RATE ON SPARSE AND FULL ADAPTATION

To investigate the impact of learning rates on the performance of SNAP-TTA and baseline methods, we conducted experiments under sparse adaptation settings. Initially, the same learning rate was applied for each SOTA TTA algorithms across all adaptation rates to ensure fair comparisons (Table [6,](#page-16-1) [7,](#page-17-0) [8,](#page-18-1) [9,](#page-19-0) [10,](#page-20-1)and [11\)](#page-21-1). However, as sparse adaptation inherently limits the number of updates, the updates might be insufficient at lower adaptation rates and explored the effect of increasing the learning rate.

The results, summarized in Table [23,](#page-29-0) reveal that higher learning rates improve the accuracy of both the naive baseline and SNAP-TTA under sparse settings. Notably, while the naive TTA baseline benefits from a higher learning rate, its performance still falls short of that achieved with full adaptation. In contrast, SNAP-TTA surpasses the performance of full adaptation at optimal learning rates, demonstrating its ability to leverage sparse adaptation effectively. At the same time, applying these higher learning rates to full adaptation results in model instability and collapse, underscoring the need to carefully tune learning rates based on adaptation frequency. Therefore, we selected a stable learning rate of 1 × 10−<sup>4</sup> for the evaluations in our work that balances model convergence and performance across all adaptation rates. These findings suggest that SNAP-TTA not only adapts effectively under sparse settings but also maintains robustness under optimized learning rates.

Table 23: Accuracy (%) with varying Learning Rates (LR) on ImageNet-C Gaussian noise adaptation rate 0.3.

| LR         | Tent(Full) | Tent(STTA) | Tent+SNAP | CoTTA(Full) | CoTTA(STTA) | CoTTA+SNAP | EATA(Full) | EATA(STTA) | EATA+SNAP |
|------------|------------|------------|-----------|-------------|-------------|------------|------------|------------|-----------|
| 2 × 10 − 3 | 2.31       | 7.04       | 13.69     | 13.31       | 11.88       | 14.67      | 0.36       | 0.59       | 0.75      |
| 1 × 10 − 3 | 4.54       | 16.13      | 27.63     | 13.18       | 11.86       | 14.68      | 1.31       | 0.95       | 24.35     |
| 5 × 10 − 4 | 10.22      | 24.96      | 29.95     | 13.15       | 11.85       | 15.11      | 21.96      | 20.96      | 27.72     |
| 1 × 10 − 4 | 27.03      | 23.63      | 26.60     | 13.12       | 11.74       | 15.26      | 29.42      | 27.35      | 29.48     |
| 5 × 10 − 5 | 26.34      | 20.94      | 24.87     | 13.34       | 11.92       | 14.85      | 29.37      | 26.07      | 27.9      |

In conclusion, selecting an appropriately high learning rate for sparse adaptation significantly enhances performance while ensuring model stability. This strategy is particularly useful for real-world deployment of SNAP-TTA, where computational efficiency and robust performance are paramount.

# G LICENSE OF ASSETS

Datasets CIFAR10/CIFAR100 (MIT License), CIFAR10-C/CIFAR100-C (Creative Commons Attribution 4.0 International), and ImageNet-C (Apache 2.0).

Codes Torchvision for ResNet18, ResNet50, and VitBase-LN (Apache 2.0), the official repository of CoTTA (MIT License), the official repository of Tent (MIT License), the official repository of EATA (MIT License), the official repository of SAR (BSD 3-Clause License), the official repository of RoTTA (MIT License), and the official repository of MECTA (Sony AI).