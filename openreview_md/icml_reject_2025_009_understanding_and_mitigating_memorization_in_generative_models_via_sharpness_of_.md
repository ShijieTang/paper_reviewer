# Understanding And Mitigating Memorization In Generative Models Via Sharpness Of Probability Landscapes

Dongjae Jeon * 1 **Dueun Kim** * 2 **Albert No** 2

## Abstract

In this paper, we introduce a geometric framework to analyze memorization in diffusion models through the sharpness of the log probability density. We mathematically justify a previously proposed score-difference-based memorization metric by demonstrating its effectiveness in quantifying sharpness. Additionally, we propose a novel memorization metric that captures sharpness at the initial stage of image generation in latent diffusion models, offering early insights into potential memorization. Leveraging this metric, we develop a mitigation strategy that optimizes the initial noise of the generation process using a sharpness-aware regularization term. The code is publicly available at https://github.com/Dongjae0324/
sharpness_memorization_diffusion.

## 1. Introduction

Recent advancements in generative models have significantly improved data generation across various domains, including image synthesis (Rombach et al., 2022), natural language processing (Achiam et al., 2023; Touvron et al., 2023), and molecular design (Alakhdar et al., 2024). Among these, diffusion models (Ho et al., 2020; Song et al., 2021c) have emerged as powerful frameworks, achieving state-ofthe-art results by iteratively refining noisy samples to approximate complex data distributions (Song et al., 2021b; Saharia et al., 2022; Rombach et al., 2022). Despite their successes, diffusion models suffer from memorization, where they replicate training samples instead of generating novel outputs (Carlini et al., 2023; Somepalli et al., 2023b; Webster, 2023). This issue is especially con-
*Equal contribution 1Department of Computer Science, Yonsei University, Seoul, Korea 2Department of Artificial Intelligence, Yonsei University, Seoul, Korea. Correspondence to: Albert No
<albertno@yonsei.ac.kr>.

1 cerning when models are trained on sensitive data, leading to privacy risks (Orrick, 2023; Joseph Saveri, 2023). Addressing memorization is critical for ensuring the responsible deployment of generative models in real-world applications. Previous work has sought to analyze memorization using various approaches, including probability manifold analysis via Local Intrinsic Dimensionality (LID) (Ross et al., 2024; Kamkari et al., 2024), spectral characterizations (Ventura et al., 2024; Stanczuk et al., 2024), and score-based discrepancy measures (Wen et al., 2024). Additionally, attentionbased methods have been used to examine memorization at the feature level (Ren et al., 2024; Chen et al., 2024). In this work, we propose a general sharpness-based framework for understanding memorization in diffusion models. Specifically, we observe that memorization correlates with regions of sharpness in the probability landscape, which can be quantified via the Hessian of the log probability. Large negative eigenvalues of the Hessian indicate sharp, isolated regions in the learned distribution, providing a mathematically grounded explanation of memorization. Furthermore, we show that the trace-based eigenvalue statistics can serve as a robust early-stage indicator of memorization, enabling detection at the initial sampling step of generation. Our framework also provides a justification for score based metric by interpreting it through the lens of sharpness, reinforcing its validity as a memorization detection metric. Building on this, we propose an enhanced sharpness measure with additional Hessian components, improving sensitivity, particularly at the earliest stages of sampling. Beyond detection, we introduce an inference-time mitigation strategy that reduces memorization by selecting initial diffusion noise from regions of lower sharpness. Our method, Sharpness-Aware Initialization for Latent Diffusion (SAIL), utilizes our sharpness metric to identify initializations that avoid trajectories leading to memorization. By simply adjusting the initial noise, SAIL steers the diffusion process toward smoother probability regions, mitigating memorization without requiring retraining. Unlike prompt modifications, which can negatively affect generation quality, SAIL reduces memorization by carefully selecting the initial noise while fully preserving the conditioning inputs.

We validate our approach through experiments on a 2D toy dataset, MNIST, and Stable Diffusion. Our results show that Hessian eigenvalues effectively differentiate memorized from non-memorized samples, and our sharpness measure provides a reliable metric for memorization detection. Additionally, we demonstrate that SAIL mitigates memorization while preserving generation quality, offering a simple yet effective solution for reducing memorization. In summary, our key contributions are:
- We propose a sharpness-based framework for analyzing memorization in diffusion models, examining the patterns of Hessian eigenvalues and their aggregate statistics to characterize memorized samples.

- We provide a theoretical justification for the memorization detection metric introduced by Wen et al. (2024)
through sharpness analysis.

- We introduce a new sharpness measure that enables early-stage memorization detection during the diffusion process.

- We propose SAIL, a simple yet effective mitigation strategy that selects initial noise leading to smoother probability regions, reducing memorization without altering model parameters or prompts.

## 2. Related Works

Understanding and Explaining Memorization. The memorization behavior of diffusion models (DMs) has been extensively studied (Somepalli et al., 2023b; Carlini et al., 2023; Wen et al., 2024), with prior work examining contributing factors such as prompt conditioning (Somepalli et al., 2023b), data duplication (Carlini et al., 2023; Somepalli et al., 2023a), and dataset size or complexity (Gu et al., 2023). Some studies have approached this issue from a geometric standpoint, drawing on the manifold learning conjecture (Fefferman et al., 2016; Pope et al., 2021), where exact memorization is associated with data points lying on a zero-dimensional manifold (Ross et al., 2024; Ventura et al., 2024; Pidstrigach, 2022). This geometric perspective has led to efforts to estimate Local Intrinsic Dimensionality (LID) at the sample level (Stanczuk et al., 2024; Kamkari et al., 2024; Horvat & Pfister, 2024; Wenliang & Moran, 2023; Tempczyk et al., 2022), which has been used to characterize memorization (Ross et al., 2024; Ventura et al., 2024). While our work is inspired by prior studies, it introduces several key distinctions. Unlike approaches that define memorization in terms of overall model behavior (Yoon et al.,
2023; Gu et al., 2023), we focus on sample-specific behavior manifested in the learned probability density. Although our perspective is conceptually aligned with recent geometric interpretations (Ross et al., 2024; Bhattacharjee et al., 2023), our methodology diverges fundamentally by analyzing sharpness in the learned density, without relying on assumptions about an inaccessible ground-truth distribution. In contrast to manifold-based analyses that track variations in individual feature components (Ventura et al., 2024; Achilli et al., 2024), we show that sharpness, treated as an aggregated statistic, can be effectively estimated and used for detecting memorization. Moreover, unlike LID- based methods (Ross et al., 2024) that are restricted to the final denoising step, our approach reveals that memorized samples persistently occupy high-sharpness regions throughout the diffusion process. This allows for earlier detection and targeted intervention, enabling a more proactive and interpretable strategy for mitigating memorization. Detecting and Mitigating Memorization. Detecting and mitigating memorization during the generative process remains a challenging problem. Previous studies have explored various approaches to identify prompts that induce memorization in text-conditional DMs by comparing generated images to training data. For instance, Somepalli et al. (2023a) employed feature-based detectors like SSCD (Pizzi et al., 2022) and DINO (Caron et al., 2021), while Carlini et al. (2023) and Yoon et al. (2023) used calibrated ℓ2 distance in pixel space to quantify memorization. Webster
(2023) developed both white-box and black-box attacks, analyzing edges and noise patterns in generated images.

While these methods provide valuable insights, their computational cost makes real-time detection impractical. To address this limitation, heuristic-based alternatives have been proposed. Wen et al. (2024) introduced a metric based on the magnitude of text-conditional score predictions, leveraging the observation that memorized prompts exhibit stronger text guidance. Similarly, Ren et al. (2024) identified memorization via anomalously high attention scores on specific tokens, while Chen et al. (2024) focused on patterns in end tokens of text embeddings. Since memorization in DMs is often linked to specific text prompts, most mitigation strategies have focused on modifying prompts or adjusting attention mechanisms to reduce their influence (Wen et al., 2024; Ren et al., 2024; Ross et al., 2024). For example, Ross et al. (2024) rephrased prompts using GPT-4 to mitigate memorization. However, these interventions often degrade image quality or compromise user intent by altering model-internal components. In contrast, our approach offers a principled and modelagnostic alternative by optimizing the initial noise input instead of modifying the text prompt or trained model parameters. By selecting initial noise that leads to smoother probability regions, our method mitigates memorization while preserving both user prompts and model fidelity, ensuring minimal impact on generation quality.

## 3. Preliminaries

Score-based Diffusion Models. Diffusion models (DMs) (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2021c) generate images by iteratively refining random noise into samples that approximate the data distribution p0(x0). The process begins with the forward process, where the training data is progressively corrupted by the addition of Gaussian noise. At each timestep t, the conditional distribution of the noisy data is given by:

$$q_{t|0}(\mathbf{x}_{t}|\mathbf{x}_{0})={\mathcal{N}}(\mathbf{x}_{t}|{\sqrt{\alpha_{t}}}\mathbf{x}_{0},(1-\alpha_{t})\mathbf{I}),$$

where xt represents the noisy data at timestep t, and αt decreases monotonically over time in the variance-preserving case, with αT becoming sufficiently small such that the resulting distribution closely resembles pure Gaussian noise:

$$q_{T|0}(\mathbf{x}_{T}|\mathbf{x}_{0})\approx{\mathcal{N}}(\mathbf{0},\mathbf{I}).$$

This process can be equivalently represented as a stochastic differential equation (SDE):

$$d\mathbf{x}_{t}=f(\mathbf{x}_{t},t)d t+g(t)d\mathbf{w}_{t},$$

where wt is a standard Brownian motion.

The reverse process, which reconstructs the data distribution p0(x0) from noise, is then formulated as:

$$d\mathbf{x}_{t}=\left[f(\mathbf{x}_{t},t)-g^{2}(t)\nabla_{\mathbf{x}_{t}}\log p_{t}(\mathbf{x}_{t})\right]d t+g(t)d{\bar{\mathbf{w}}}_{t},$$

where w¯ t denotes a standard Brownian motion in reverse time, and pt(xt) is the marginal distribution at timestep t.

The only unknown term in the reverse process is the score function over timesteps, ∇xtlog pt(xt) := s(xt), which is often parameterized by a neural network with sθ(xt). In many applications the data x0 is often represented with an associated label c (e.g., prompts or class labels). In these scenarios, the additional condition c is incorporated into the model as sθ(xt, c), allowing it to estimate the score of the conditional density ∇xtlog pt(xt|c) := s(xt, c) via classifier free guidance (Ho & Salimans, 2021). Sharpness and Hessian. For a given function f at a point x, the Hessian ∇2xf(x) represents the matrix of secondorder derivatives, encapsulating the local curvature of f around x. The eigenvectors of the Hessian define the principal axes of this curvature, while the corresponding eigenvalues characterize the curvature along these directions. Positive eigenvalues indicate local convexity, negative eigenvalues indicate local concavity, and zero eigenvalues indicate flatness in those directions. The magnitude of an eigenvalue reflects the steepness of the curvature, with larger absolute values indicating steeper changes in f. In this work, we examine the memorization by analyzing the Hessian of log pt(xt), which corresponds to the Jacobian of the score function. We denote it as H(xt) := ∇2xt log pt(xt)
for the unconditional case and H(xt, c) := ∇2xt log pt(xt|c)
for the conditional case. The Hessian estimated by the model is denoted as Hθ(xt) and Hθ(xt, c).

## 4. **Understanding Memorization Via Sharpness**

4.1. **Memorization: Sharpness in Probability Landscape** Sharpness quantifies the concentration of learned log density log p(x) around point x, which can be analyzed through the eigenvalues of its Hessian matrix. Large negative eigenvalues indicate sharp peaks in the distribution, suggesting memorization of specific data points. Conversely, small magnitude or positive eigenvalues characterize broader, smoother regions that facilitate better generalization. Local Intrinsic Dimensionality (LID) (Kamkari et al., 2024) quantifies the effective dimensionality of a point in its local neighborhood, characterizing local sample space geometry. At the final generation step (t ≈ 0), LID serves as a memorization indicator (Ross et al., 2024). Exact Memorization (EM) shows near-zero LID, indicating pure reproduction of training samples, while Partial Memorization (PM) exhibits small but nonzero LID, reflecting limited stylistic variations. In contrast, properly generalized samples demonstrate moderate LID values, indicating more diverse representations. While both sharpness and LID characterize curvature properties of probability density, LID is limited to analyzing sample space at t ≈ 0, where the generated image emerges.

In contrast, we extend memorization detection across all timesteps by leveraging sharpness via Hessian eigenvalues as a more versatile metric, enabling continuous monitoring throughout the generative process rather than relying solely on final output characteristics.

(a) (b)
Exa c t M
e m P
a rt ia l M
e m N
o n M
e m Mem Non-mem 
Figure 2: **Left:** Generated images for memorized (digit "9") and non-memorized (digit "3") samples. **Right:** Eigenvalue distributions for memorized (red) and non-memorized (blue) samples at initial **(top)** and final **(bottom)** sampling steps, revealing more and larger negative eigenvalues in memorized cases. Experimental details in Appendix C. Figure 1(b) demonstrates our approach using a mixture of 2D Gaussians, where sharp peaks represent memorized distributions. From the mid stage of the denoising process, the memorized sample (red) exhibits large negative eigenvalues, indicating highly localized distributions, while the generic sample (blue) maintains near-zero eigenvalues, characterizing broader, smoother regions. Importantly, the memorized sample exhibits sharp characteristics even at intermediate timesteps, making early-stage detection possible.

To validate our approach on real data, we conduct experiments on MNIST by inducing memorization through repeated exposure to a single "9" image while maintaining all
"3" images as a general class (Figure 2). The eigenvalue distributions at t = 1 clearly differentiate memorized from nonmemorized samples: memorized samples show consistently large negative eigenvalues indicating sharp peaks, while non-memorized samples exhibit positive eigenvalues, reflecting locally convex regions that allow sample variations. Notably, these clear distributional differences emerged even at the initial sampling step (t = T − 1), confirming that sharpness-based memorization detection is effective from the very beginning of the generation process.

We further validate our approach on Stable Diffusion (Rombach et al., 2022), analyzing its 16, 384-dimensional latent space. Figure 3 reveals distinct patterns in both the number of non-negative eigenvalues and the magnitude of negative eigenvalues across different memorization categories (EM, PM, and non-memorized) at both initial and final sampling step. These patterns not only align with LID-based analysis at t ≈ 0 but also demonstrate sharpness as a more generalizable memorization measure, capturing distinctive characteristics at generation onset.

## 4.2. Score Norm As A Sharpness Measure

While sharpness serves as a fundamental measure of memorization in generative models, directly computing the full spectrum of Hessian eigenvalues in high-dimensional distributions, such as those in Stable Diffusion, is computationally intractable. A practical alternative is to approximate sharpness using the trace of the Hessian, a single scalar quantity that represents the sum of all eigenvalues, where large negative traces indicate sharp, highly localized regions. A key observation is that the norm of the score function ∥s(x)∥ inherently encodes information about the probability landscape's curvature. In Gaussian distributions, the score norm is directly connected to the Hessian trace, as shown in the following result. (Appendix B.2).

  **Lemma 4.1**.: _For a Gaussian vector $\mathbf{x}\sim\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$,_
$$\mathbb{E}\left[\|s(\mathbf{x})\|^{2}\right]=-\mathrm{tr}(H(\mathbf{x})),$$
$\nabla^{-1}$ : . 
$IS\;III$ . 

where H(x) ≡ −Σ
−1*is the Hessian of the log density.*
This result extends to non-Gaussian distributions under mild regularity assumptions (Appendix B.2). For theoretical clarity and ease of analysis, however, we focus on the Gaussian case. While the distribution xt in diffusion processes is not strictly Gaussian at every timestep, recent studies show that at moderate to high noise levels, corresponding to the early and middle stages of the reverse process—the learned score is predominantly governed by its Gaussian component (Wang & Vastola, 2024). This approximation is further justified in latent diffusion models, where the latent variable zt is explicitly regularized toward a Gaussian prior (Kingma, 2013; Rombach et al., 2022), despite the complexity of the original data distribution. Under this Gaussian assumption at relevant sampling steps, the score norm ∥sθ(xt)∥
2 provides an unbiased estimate of the negative Hessian trace −tr(Hθ(xt)), offering an efficient measure of the sharpness of the probability landscape.

MNIST 
Stable Diffusion v1.4 
Figure 4: Empirical alignment in MNIST and Stable Diffusion between: **(left)** −trHθ(xt, c)and ∥sθ(xt, c)∥
2, and
(right) −trHθ(xt, c)
3and ∥Hθ(xt, c)sθ(xt, c)∥
2.

Figure 4 empirically confirms that this approximation holds reliably across datasets, including MNIST and Stable Diffusion's latent space. Surprisingly, this relationship persists even in the later stages of the diffusion process, suggesting that score norm can serve as a computationally efficient sharpness measure throughout generation. This perspective provides a theoretical foundation for interpreting sharpness in generative models through score norm based statistic, enabling efficient memorization detection and analysis without requiring costly Hessian eigenvalue decompositions.

## 4.3. Wen'S Metric As A Sharpness Measure

Wen et al. (2024) characterized memorization through the norm of difference between conditional and unconditional score functions:

$$\|s_{\theta}^{\Delta}({\bf x}_{t})\|:=\|s_{\theta}({\bf x}_{t},c)-s_{\theta}({\bf x}_{t})\|.$$

This difference vector s
∆ θ
(xt) determines the sampling direction in classifier-free guidance. Their approach is based on the observation that memorized prompts consistently guide generation toward specific images, resulting in larger magnitudes of s
∆ θ
(xt) due to stronger text-driven guidance.

While the theoretical foundations of this heuristic remain to be fully understood, it has proven to be one of the most effective detection metrics thus far. Notably, the structure of ∥s
∆
θ(xt)∥ bears a strong resemblance to the score norm, which we previously identified as a measure of sharpness. This similarity hints at the possibility of interpreting Wen's metric as a sharpness measure, encapsulating the impact of conditioning on the probability distribution's curvature. To rigorously establish this connection, we proceed to analyze the Hessian of the log-density, following the same approach as in the preceding analysis.

Lemma 4.2. For x ∼ N (µ, Σ) and x|c ∼ N (µc, Σc):

$$i o r\,\mathbf{x}\sim{\mathcal{N}}\left({\boldsymbol{\mu}},{\boldsymbol{\Sigma}}\right)$$
$$\begin{array}{l}{{\mathbb{E}_{{\bf x}\sim p({\bf x}|c)}\left[\|s({\bf x},c)-s({\bf x})\|^{2}\right]}}\\ {{\mathrm{~}=\|H({\bf x})(\mu-\mu_{c})\|^{2}+\operatorname{tr}((H({\bf x})-H_{c}({\bf x}))^{2}H_{c}^{-1}({\bf x})),}}\end{array}$$
$${\mathit{where}}\ H(\mathbf{x})\equiv-\mathbf{\Sigma}^{-1}\ {\mathit{and}}\ H_{c}(\mathbf{x})\equiv-\mathbf{\Sigma}_{c}^{-1}.$$

Additionally, when Σ and Σc *commute (i.e.,* ΣΣc = ΣcΣ)
and mean vectors are the same (µ = µc*), this reduces to*

$$\mathbb{E}_{\mathbf{x}\sim p(\mathbf{x}|c)}\left[\|s(\mathbf{x},c)-s(\mathbf{x})\|^{2}\right]=\sum_{i=1}^{d}{\frac{(\lambda_{i}-\lambda_{i,c})^{2}}{\lambda_{i,c}}},$$
$$:|c\sim{\mathcal{N}}\left(\mu_{c},\Sigma_{c}\right):$$

where λi, λi,c are eigenvalues of H(x) and Hc(x).

This result demonstrates that Wen's metric measures sharpness differences through squared eigenvalue differences of the conditional and unconditional Hessian. During early timesteps, when the latent distribution remains close to an isotropic Gaussian, this metric directly captures the extent to which conditioning induces sharpness. At later timesteps, when Σt and Σt,c do not generally commute, the metric can be interpreted through generalized eigenvalues, revealing how conditioning sharpens the learned distribution in similar manner. The details are provided in Appendix A.3. Figure 5: Eigenvalue differences between the conditional and unconditional Hessians. Memorized samples exhibit a significantly larger gap, while non-memorized samples show near-zero differences throughout. At intermediate timesteps (t = 20), the gap remains small but detectable, and at the final stage (t = 1), it widens further. Figure 5 shows the eigenvalue disparities between conditional and unconditional Hessians across timesteps, revealing how conditioning shapes the probability distribution's geometry. For memorized samples, the eigenvalue gap is notably large, showing that conditioning creates a more constrained probability landscape. At intermediate timesteps (t = 20), the differences are subtle but noticeable, indicating early conditioning effects. Near the end (t = 1),
the eigenvalue gap widens substantially, demonstrating conditioning's growing influence on the learned density. In contrast, non-memorized samples show minimal eigenvalue variations throughout, indicating little conditioning influence. These findings support our theoretical framework and confirm Wen's metric effectively measures sharpness.

## 4.4. Upscaling Eigenvalue Statistics Via Hessian

While Wen's metric reveals eigenvalue disparities at intermediate timesteps, identifying and mitigating memorization during the initial generation stage remains challenging. The probability landscape maintains a nearly uniform character since the latent distribution approximates an isotropic Gaussian, making structural sharpness differences subtle. Conventional metrics struggle to capture these fine-grained distributional variations, limiting early-stage applications. To address this limitation, we introduce a curvature-aware scaling that enhances Wen's metric through Hessian-based weighting. By multiplying the Hessian with the score function, we amplify high-curvature directions, rendering sharp

regions more distinguishable within a smooth probability landscape. This approach significantly improves the eigenvalue gap at the earliest generation stage, advancing memorization detection in the diffusion process. The following lemma shows that the Hessian-score product provides an amplified measure of the Hessian trace, thereby increasing sensitivity to distributional sharpness.
Lemma 4.3. For a Gaussian vector x ∼ N (µ, Σ),
$$\mathbb{E}\left[\|H(\mathbf{x})s(\mathbf{x})\|^{2}\right]=-\mathrm{tr}((H(\mathbf{x}))^{3})$$
where H(x) ≡ −Σ *is the Hessian of the log density.* This relationship, empirically verified in Figure 4, demonstrates the curvature-sensitive scaling effect of the Hessian score product. Building on this principle, we propose an enhanced version of Wen's metric that improves early-stage sensitivity through second-order sharpness characterization:

$$\|H_{\theta}^{\Delta}({\bf x}_{t},c)s_{\theta}^{\Delta}({\bf x}_{t},c)\|^{2},$$

where H∆
θ
(xt, c) = Hθ(xt, c) − Hθ(xt), and s
∆ θ
(xt, c) =
sθ(xt, c) − sθ(xt).

To provide intuition, assuming identical means (µ = µc)
and that Σt and Σt,c commute, the expected value of our metric simplifies to:

$$\mathbb{E}_{\mathbf{x}_{t}\sim p_{t}(\mathbf{x}_{t}|c)}\left[\|H_{\theta}^{\Delta}(\mathbf{x}_{t},c)s_{\theta}^{\Delta}(\mathbf{x}_{t},c)\|^{2}\right]=\sum_{i=1}^{d}{\frac{(\lambda_{i}-\lambda_{i,c})^{4}}{\lambda_{i,c}}},$$
$\mathbf{t}_{\uparrow},\mathbf{c}_{\downarrow}$. 
$\lambda_i,\lambda_{i,c}$ are eig

where λi, λi,c are eigenvalues of H(xt) and H(xt, c). Compared to Wen's metric in Lemma 4.2, this refinement substantially improves sensitivity by amplifying the difference in sharpness, thereby enabling more effective detection of memorization at earlier stages.

## 4.5. Detecting Memorization In Stable Diffusion

Experimental Setup. To evaluate our metric, we use 500 memorized prompts identified by Webster (2023) for Stable Diffusion v1.4, and 219 prompts for v2.0. As a complementary set, we include 500 non-memorized prompts sourced from COCO (Lin et al., 2014), Lexica (Lexica, 2024), Tuxemon (HuggingFace, 2024), and GPT-4 (Achiam et al., 2023). Following Wen et al. (2024), we apply the DDIM (Song et al., 2021a) sampler with 50 inference steps. Detection performance is assessed with two standard metrics: the Area Under the Receiver Operating Characteristic Curve (AUC) and the True Positive Rate at 1% False Positive Rate (TPR@1%FPR) with higher values preferable. For comparison, we implement six baseline methods.

Among them, Carlini et al. (2023) analyzed generation density by measuring pixel-wise ℓ2 distances across nonoverlapping image tiles, aiming to detect memorized content based on local similarity patterns. Ren et al. (2024)

SD v1.4 **SD v2.0**

Method Steps n AUC TPR@1%FPR AUC TPR@1%FPR

Tiled ℓ2 (Carlini et al., 2023) 50 4 0.908 0.088 0.792 0.114

16 0.94 0.232 0.907 0.114

LE (Ren et al., 2024) 1

1 0.846 0.116 0.848 0 4 0.839 0.13 0.853 0

16 0.832 0.124 0.851 0

AE (Ren et al., 2024) 50

1 0.606 0 0.809 0 4 0.628 0 0.82 0

16 0.598 0 0.817 0

BE (Chen et al., 2024) 50

1 0.986 0.95 0.983 0.908 4 0.997 0.98 0.99 0.945

16 0.997 0.982 0.99 0.949

1

1 0.976 0.896 0.948 0.739 4 0.992 0.944 0.98 0.876

16 0.99 0.928 0.983 0.881

5

1 0.991 0.932 0.969 0.885 4 0.997 0.978 0.984 0.917

16 **0.998 0.982** 0.987 0.931

50

1 0.983 0.948 0.982 0.904 4 0.996 **0.982** 0.99 **0.949**

16 **0.998** 0.98 **0.991** 0.945

∥H∆

θ(xT )s

∆

θ(xT )∥

2(Ours) 11 0.987 0.908 0.959 0.74

4 0.998 0.982 **0.991** 0.895

Table 1: AUC and TPR@1%FPR across detection strategies and sampling steps for Stable Diffusion (SD) v1.4 and v2.0. Here, n denotes the number of generations per prompt, with results averaged over n. "Steps" indicates the stage along the diffusion sampling path, ranging from step 1 (t = T − 1) to step 50 (t = 1). detected memorized samples by identifying anomalous attention score patterns in text-conditioning during sampling.

Chen et al. (2024) refined Wen et al. (2024)'s metric for partial memorization by incorporating end-token masks that empirically highlight locally memorized regions.

We report detection results at sampling steps 1, 5, and 50, but only include 50-step results for methods requiring full sampling or showing significant performance gains. Additional experimental details are provided in Appendix D.1.

Results. Table 1 demonstrates our metric's strong performance on Stable Diffusion v1.4 and v2.0 using just a single sampling step. By upscaling curvature information via H∆
θ
(xt), we significantly enhance Wen et al. (2024)'s metric. With merely four generations, we achieve an AUC of 0.998 and TPR@1%FPR of 0.982, matching Wen et al. (2024)'s performance using five steps and 16 generations. Similarly, in v2.0, our approach attains an AUC of 0.991 without full-step sampling, underscoring its effectiveness.

| LE (Ren et al., 2024)                                   | 1   |
|---------------------------------------------------------|-----|
| AE (Ren et al., 2024)                                   | 50  |
| BE (Chen et al., 2024)                                  | 50  |
| ∥H∆ θ (xT )s ∆ θ (xT )∥ ∥s ∆ (xt)∥ (Wen et al., 2024) θ |     |

Importantly, our metric can be efficiently computed using Hessian-vector products without explicitly forming the full Hessian matrix. Leveraging automatic differentiation frameworks such as PyTorch, a single Hessian-vector product suffices for detection, incurring minimal overhead.

## 5. Sharpness Aware Memorization Mitigation 5.1. Sharpness Aware Initialization Sampling

Motivation. In Section 4, we observed that memorized samples exhibit a sharp conditional density, pt(xt|c), even at the very beginning of the generation process (i.e., at t = T − 1; note that sampling proceeds in reverse order, starting from t = T). This is substantiated by the strong detection performance of both Wen's metric and our metric at the initial sampling step, which quantifies the sharpness gap between pt(xt|c) and pt(xt).

This phenomenon, linked to the deterministic nature of ODE samplers (a one-to-one mapping between noise and image), implies that initializations from sharper densities remain in sharper regions at each intermediate timestep of the generation process, thereby increasing the likelihood of producing memorized images. In contrast, initializations from smoother regions tend to yield non-memorized images. Thus, we argue that sampling with noise from smoother densities could effectively mitigate memorization. While manually searching for such initializations is a straightforward approach, it becomes infeasible in high-dimensional Gaussian space due to the sheer size and complexity of the search domain. Consequently, we propose to directly optimize the initial noise xT as a more scalable and systematic way to address this challenge. Sharpness Aware Initialization. We propose Sharpness- Aware Initialization for Latent Diffusion (**SAIL**), an inference-time mitigation strategy that optimizes initializations xT by minimizing the sharpness gap at the starting step
(t = T −1). SAIL identifies initial seeds on non-memorized sampling trajectories by selecting xT from smoother regions while maintaining a reasonable density under the isotropic Gaussian prior. The objective function is defined as:

$$\|H_{\theta}^{\Delta}({\bf x}_{T})s_{\theta}^{\Delta}({\bf x}_{T})\|^{2}-\alpha\log p_{G}({\bf x}_{T}),$$

Original Ours Ren et al. Wen et al. RNA RTA
Co lbe rt Bj ör k N
etfl ix So ut h P
ar k 
where pG is the density of an isotropic Gaussian distribution. While ∥H∆
θ
(xT )s
∆ θ
(xT )∥
2can be efficiently computed using Hessian-vector products, the gradient backpropagation required for optimization introduces computational overhead. To overcome the burden, we approximate the term using a Taylor expansion around xT :

$$\|H_{\theta}^{\Delta}({\bf x}_{T})s_{\theta}^{\Delta}({\bf x}_{T})\|^{2}\approx\frac{\left\|s_{\theta}^{\Delta}\left({\bf x}_{T}+\delta s_{\theta}^{\Delta}\left({\bf x}_{T}\right)\right)-s_{\theta}^{\Delta}\left({\bf x}_{T}\right)\right\|^{2}}{\delta^{2}},$$

This leads to the final objective for SAIL:
LSAIL(xT ) := ∥s
∆ θ xT + δs∆
θ
(xT )− s
∆ θ
(xT )∥
2 + α∥xT ∥
2, where α balances the sharpness of the density and the original likelihood. To ensure initializations remain close to the Gaussian distribution, we employ early stopping based on a threshold ℓthres, limiting number of optimization steps.

## 5.2. Mitigating Memorization In Stable Diffusion.

Experimental Setup. To evaluate mitigation strategies, we use the same memorized prompt set employed in the detection experiments described in Section 4.5. However, since verifying mitigation effects requires access to training images, we exclude prompts whose corresponding training samples are unavailable. Further details are in Appendix D.

We employ two key metrics following (Wen et al., 2024; Somepalli et al., 2023a): the SSCD similarity score (Pizzi et al., 2022), which quantifies memorization by comparing model-based features of generated images to their corresponding training data, and the CLIP score (Radford et al., 2021), which evaluates prompt-image alignment. Results are averaged over five generations per prompt. For comparison, we implement four recent mitigation algorithms. Somepalli et al. (2023b) propose Random Token Addition (RTA) and Random Number Addition (RNA), which perturb original prompts to mitigate memorization. Wen et al. (2024) introduce a method that optimizes text embeddings to reduce the influence of memorization-inducing tokens. Ren et al. (2024) propose a strategy that adjusts attention scores of text embeddings for mitigation. For a fair comparison, all methods are evaluated using five distinct hyperparameter settings and optimized with the Adam optimizer at a learning rate of 0.05. For a detailed experimental settings, refer to Appendix D.2. Results. Figure 6 (left) demonstrates that SAIL significantly improves both SSCD and CLIP metrics for Stable Diffusion v1.4 and v2.0. By optimizing the noise initialization xT without altering model components like text embeddings or attention weights, SAIL effectively mitigates memorized content while preserving model behavior and user prompts, ensuring high-quality, non-memorized outputs.

The advantage of SAIL is evident in Figure 6 (right), where it generates images that faithfully preserve key prompt details, such as celebrity names and primary objects. In contrast, methods that modify text-conditional components often reduce the influence of those components during mitigation, leading to degraded alignment with the original prompt and potentially diminishing user utility. Additional qualitative results for algorithms are provided in Appendix E.

## 6. Conclusion

We propose a sharpness-based framework for detecting and mitigating memorization in diffusion models. Our analysis identifies Hessian-based sharpness as a reliable indicator of memorization and introduces an efficient proxy based on the score norm. This perspective also provides a theoretical interpretation of the memorization detection metric proposed by Wen et al. (2024). Building on this foundation, we introduce Sharpness-Aware Initialization for Latent Diffusion (SAIL), an inference-time method that reduces memorization by selecting low-sharpness initial noise. Experiments on synthetic 2D data, MNIST, and Stable Diffusion demonstrate that our approach enables early detection and effective mitigation, all without degrading generation quality.

## Acknowledgement

This work was supported in part by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. RS-2024-00457882, AI Research Hub Project), the Ministry of Science and ICT (MSIT), South Korea, under the Information Technology Research Center (ITRC) Support Program (IITP-2025-RS-2022-00156295), and IITP grant funded by the Korean Government (MSIT) (No. RS-2020- II201361, Artificial Intelligence Graduate School Program (Yonsei University)).

## Impact Statement

Our work aims to advance the understanding and mitigation of memorization in diffusion models, a phenomenon closely tied to potential privacy risks. By proposing a framework to detect and reduce memorization, we seek to enhance the responsible deployment of generative models, especially when they are trained on sensitive data. This approach could contribute positively by lowering the risk of unintentionally revealing private information.

## References

A. Micchelli, C. and Noakes, L. Rao distances. Journal of Multivariate Analysis, 92(1):97–115, 2005.

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I.,
Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Achilli, B., Ventura, E., Silvestri, G., Pham, B., Raya, G.,
Krotov, D., Lucibello, C., and Ambrogioni, L. Losing dimensions: Geometric memorization in generative diffusion, 2024.

Alakhdar, A., Poczos, B., and Washburn, N. Diffusion models in de novo drug design. Journal of Chemical Information and Modeling, 2024.

Arnoldi, W. E. The principle of minimized iterations in the solution of the matrix eigenvalue problem. Quarterly of Applied Mathematics, 1951.

Bhattacharjee, R., Dasgupta, S., and Chaudhuri, K. Datacopying in generative models: a formal framework. In ICML, 2023.

Carlini, N., Hayes, J., Nasr, M., Jagielski, M., Sehwag, V.,
Tramèr, F., Balle, B., Ippolito, D., and Wallace, E. Extracting training data from diffusion models. In USENIX Security, 2023.

Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J.,
Bojanowski, P., and Joulin, A. Emerging properties in self-supervised vision transformers. In *CVPR*, 2021.

Chen, C., Liu, D., Shah, M., and Xu, C. Exploring local memorization in diffusion models via bright ending attention. *arXiv preprint arXiv:2410.21665*, 2024.

Fefferman, C., Mitter, S., and Narayanan, H. Testing the manifold hypothesis. Journal of the American Mathematical Society, 2016.

Gu, X., Du, C., Pang, T., Li, C., Lin, M., and Wang, Y.

On memorization in diffusion models. arXiv preprint arXiv:2310.02664, 2023.

Ho, J. and Salimans, T. Classifier-free diffusion guidance.

In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In *NeurIPS*, 2020.

Horvat, C. and Pfister, J.-P. On gauge freedom, conservativity and intrinsic dimensionality estimation in diffusion models. In *ICLR*, 2024.

HuggingFace. Tuxemon, 2024. URL https:
//huggingface.co/datasets/diffusers/ tuxemon.

Hyvärinen, A. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(24):695–709, 2005.

Joseph Saveri, B. M. Stable diffusion litigation, 2023.

URL https://stablediffusionlitigation. com/.

Kamkari, H., Ross, B. L., Hosseinzadeh, R., Cresswell, J. C., and Loaiza-Ganem, G. A geometric view of data complexity: Efficient local intrinsic dimension estimation with diffusion models. In ICML 2024 Workshop on Structured Probabilistic Inference & Generative Modeling, 2024.

Kingma, D. P. Auto-encoding variational bayes. *arXiv* preprint arXiv:1312.6114, 2013.

Lanczos, C. An iteration method for the solution of the eigenvalue problem of linear differential and integral operators. *J. Res. Natl. Bur. Stand. B*, 1950.

Lexica. Lexica dataset, 2024. URL https:
//huggingface.co/datasets/vera365/
lexica_dataset.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P.,
Ramanan, D., Dollár, P., and Zitnick, C. L. Microsoft coco: Common objects in context. In ECCV, 2014.

Lu, C., Zheng, K., Bao, F., Chen, J., Li, C., and Zhu, J.

Maximum likelihood training for score-based diffusion odes by high order denoising score matching. In *ICML*, 2022.

Meng, C., Song, Y., Li, W., and Ermon, S. Estimating high order gradients of the data distribution by denoising. In NeurIPS, 2021.

Orrick, W. H. Andersen v. Stability AI Ltd.,
2023. URL https://casetext.com/case/ andersen-v-stability-ai-ltd.

Pidstrigach, J. Score-based generative models detect manifolds. In *NeurIPS*, 2022.

Pizzi, E., Roy, S. D., Ravindra, S. N., Goyal, P., and Douze, M. A self-supervised descriptor for image copy detection. In *CVPR*, 2022.

Pope, P., Zhu, C., Abdelkader, A., Goldblum, M., and Goldstein, T. The intrinsic dimension of images and its impact on learning. In *ICLR*, 2021.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G.,
Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In ICML, 2021.

Ren, J., Li, Y., Zeng, S., Xu, H., Lyu, L., Xing, Y., and Tang, J. Unveiling and mitigating memorization in textto-image diffusion models through cross attention. In ECCV, 2024.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.

Ross, B. L., Kamkari, H., Wu, T., Hosseinzadeh, R., Liu, Z., Stein, G., Cresswell, J. C., and Loaiza-Ganem, G. A
geometric framework for understanding memorization in generative models. *arXiv preprint arXiv:2411.00113*,
2024.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022.

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In *ICML*, 2015.

Somepalli, G., Singla, V., Goldblum, M., Geiping, J., and Goldstein, T. Diffusion art or digital forgery? investigating data replication in diffusion models. In *CVPR*, 2023a.

Somepalli, G., Singla, V., Goldblum, M., Geiping, J., and Goldstein, T. Understanding and mitigating copying in diffusion models. In *NeurIPS*, 2023b.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. In *ICLR*, 2021a.

Song, Y., Durkan, C., Murray, I., and Ermon, S. Maximum likelihood training of score-based diffusion models. In NeurIPS, 2021b.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In *ICLR*, 2021c.

Stanczuk, J., Batzolis, G., Deveney, T., and Schönlieb, C.-B.

Diffusion models encode the intrinsic dimension of data manifolds. In *ICML*, 2024.

Tempczyk, P., Michaluk, R., Garncarek, L., Spurek, P.,
Tabor, J., and Golinski, A. Lidl: Local intrinsic dimension estimation using approximate likelihood. In *ICML*, 2022.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.

Ventura, E., Achilli, B., Silvestri, G., Lucibello, C., and Ambrogioni, L. Manifolds, random matrices and spectral gaps: The geometric phases of generative diffusion. *arXiv* preprint arXiv:2410.05898, 2024.

Wang, B. and Vastola, J. The unreasonable effectiveness of gaussian score approximation for diffusion models and its applications. *Transactions on Machine Learning* Research, 2024.

Webster, R. A reproducible extraction of training images from diffusion models. *arXiv preprint arXiv:2305.08694*, 2023.

Wen, Y., Liu, Y., Chen, C., and Lyu, L. Detecting, explaining, and mitigating memorization in diffusion models. In ICLR, 2024.

Wenliang, L. K. and Moran, B. Score-based generative models learn manifold-like structures with constrained mixing. In *NeurIPS Workshop SBM*, 2023.

Yoon, T., Choi, J. Y., Kwon, S., and Ryu, E. K. Diffusion probabilistic models generalize when they fail to memorize. In ICML 2023 Workshop on Structured Probabilistic Inference & Generative Modeling, 2023.

## A. Additional Mathematical Details A.1. Second-Order Score Function

Since the Hessian of interest is simply the Jacobian of the score function, it can be directly computed using automatic differentiation from a trained diffusion model (DM). While a well-trained DM that accurately estimates scores should theoretically yield an accurate Hessian via automatic differentiation, this is not always the case in practice. Therefore, to achieve a more accurate estimation of the Hessian, the model should be parameterized and incorporate a second-order score matching loss that estimates ∇2xt log pt(xt) ≈ ∇xtsθ(xt) := Hθ(xt) as demonstrated by Meng et al. (2021). This can be interpreted as implicit correction of the parametrized score function. To enhance numerical stability in the loss function, we adopt the loss proposed by Lu et al. (2022), an improved version of the loss utilized by Meng et al. (2021). For a fixed t and given trained score function, this loss is defined as:

$$\theta^{*}=\operatorname*{arg\,min}_{\theta}\mathbb{E}_{\mathbf{x}_{0},\epsilon}\left[{\frac{1}{\sigma_{t}^{4}}}\left\|\sigma_{t}^{2}H_{\theta}(\mathbf{x}_{t})+\mathbf{I}-\ell_{1}\ell_{1}^{\top}\right\|_{F}^{2}\right],$$
where $\ell_{1}(\mathbf{\epsilon},\mathbf{x}_{0}):=\sigma_{t}s_{\theta}(\mathbf{x}_{t})+\mathbf{\epsilon},\ \mathbf{x}_{t}=\alpha_{t}\mathbf{x}_{0}+\sigma_{t}\mathbf{\epsilon},\ \mathbf{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$. The proposed objective is 
$${\mathcal{L}}_{D S M}^{(2)}(\theta):=\mathbb{E}_{t,{\mathbf{x}}_{0},\epsilon}\left[\left\|\sigma_{t}^{2}H_{\theta}({\mathbf{x}}_{t})+\mathbf{I}-\ell_{1}\ell_{1}^{\top}\right\|_{F}^{2}\right].$$

To obtain a more accurate Hessian estimate in the Toy experiment, we used L = LDSM(θ) + 0.5L
(2)
DSM(θ), which was simultaneously optimized using a weighted sum format. For Stable Diffusion, no additional training was performed because the original training data were not publicly available, making it difficult to retrain or fine-tune. Nevertheless, as noted in the main text, we still obtained sufficiently good results with the existing pretrained model.

## A.2. Numerical Eigenvalue Algorithm

For high-resolution image data with very large dimensions, such as in Stable Diffusion, calculating the exact Hessian and finding its eigenvalues are computationally complex and mememory inefficient. As an alternative, we employ Arnoldi iteration (Arnoldi, 1951), a numerical algorithm that leverages the efficient computation of Hessian-vector products via torch.autograd.functional.jvp to approximate some leading eigenvalues without forming the Hessian explicitly. In more detail, we can compute the action of the Hessian on a vector v efficiently using automatic differentiation.
Arnoldi iteration is an algorithm derived from the Krylov subspace method that constructs an orthonormal basis Qm = [q1, q2*, . . . ,* qm] of the Krylov subspace Km, and an upper Hessenberg matrix Hm, such that the following relationship
holds:
$$\mathbf{A}\mathbf{Q}_{m}=\mathbf{Q}_{m}\mathbf{H}_{m}+h_{m+1,m}\mathbf{q}_{m+1}\mathbf{e}_{m}^{\top},$$
where em is the m-th canonical basis vector. Since we can compute Aqk without forming A explicitly, using the function
jvp_func(qk), the Arnoldi iteration proceeds as follows. First, we normalize the starting vector b to obtain q1 =b
∥b∥2
.
Then, for each iteration k = 1 to m, we compute:
$$\mathbf{v}=\operatorname{jvp\_func}(\mathbf{q}_{k}),$$
which represents the action of A on qk. We then orthogonalize v against the previous basis vectors q1*, . . . ,* qk, updating h
and v:
$$h_{j,k}={\bf q}_{j}^{\top}{\bf v},\quad{\bf v}={\bf v}-h_{j,k}{\bf q}_{j},\quad\mathrm{for}\;j=1,\ldots,k.$$
After orthogonalization, we compute hk+1,k = ∥v∥2. If hk+1,k is greater than a small threshold ε, we normalize v to obtain
the next basis vector qk+1 =v
hk+1,k
. Otherwise, the iteration terminates. The eigenvalues of Hm(Ritz values) approximate
the m eigenvalues of A. For details on the computational process of Arnoldi iteration, Please refer to the algorithm pesudo code below. The Arnoldi iteration tends to find eigenvalues with larger absolute values first because components associated with these eigenvalues dominate within the Krylov subspace. If the input matrix is symmetric, Arnoldi iteration can be simplified to Lanczos iteration (Lanczos, 1950). However, since the Lanczos iteration is very sensitive to small numerical
errors breaking the symmetry, we use the general version. The computational complexity of the algorithm is O(m2d) with
space complexity O(md), compared to O(d
3) with O(d
2) of exact derivation and eigendecomposition of Hessian. We
calculate all eigenvalues for several samples for clear justification. But with just a few(m ≪ d) iterations, the difference
between memorized samples and non-memorized samples reveals enough.

Algorithm 1 Arnoldi Iteration using Jacobian-Vector Products Require: Starting vector b ∈ R
d, number of iterations m ≤ d, function jvp_func(v) that computes Av, threshold ε Ensure: Orthonormal basis Qm = [q1*, . . . ,* qm],
upper Hessenberg matrix Hm ∈ R
m×m 1: Initialize Q ∈ R
d×(m+1), h ∈ R
(m+1)×m 2: Normalize the starting vector: q1 =b
∥b∥2 3: for k = 1 to m do 4: Compute v ← jvp_func(qk)
5: for j = 1 to k do 6: Compute hj,k ← q
⊤
j v 7: Update v ← v − hj,kqj 8: **end for**
9: Compute hk+1,k ← ∥v∥2 10: if hk+1,k > ε **then**
11: Normalize qk+1 ← v hk+1,k 12: **else** 13: **break** {Terminate iteration} 14: **end if** 15: **end for** 16: Adjust Hm by removing the last row of h 17: **return** Qm = [q1*, . . . ,* qm],
Hm = [hi,j ]i=1,...,m; j=1*,...,m*

## A.3. Generalized Eigenvalue Analysis Of Score Difference

In the main text, we demonstrated that Wen et al. (2024)'s metric can be expressed in terms of Hessian eigenvalue differences. Here, we provide a more detailed derivation, including the non-commuting case, which requires the use of generalized eigenvalues. Consider two Gaussian distributions: the unconditional distribution and the conditional distribution N (µc, Σt,c).

For simplicity, we assume the means are identical (µ = µc) and focus on the effect of covariance differences. Wen's metric approximately measures s(xt, c) − s(xt),
Through direct calculation, the expected squared difference in these scores is

$$\mathbb{E}_{\mathbf{x}_{t}\sim p(\mathbf{x}_{t}\|c)}\left[\left\|\ s(\mathbf{x}_{t},c)\ -\ s(\mathbf{x}_{t})\right\|\right]$$
−1
t,c 2 Σt,ci.
$=\;\text{tr}\bigg[\left(\mathbf{\Sigma}_t^{-1}\;-\;1\right)\bigg]$
N (µ, Σt),
When ΣtΣt,c = Σt,cΣt, this trace simplifies to a sum of squared eigenvalue differences:

$\left[\begin{array}{c}\\ 2t,c\end{array}\right]$. 
$\downarrow$ . 
$$\sum_{i}{\frac{(\lambda_{i}-\lambda_{i,c})^{2}}{\lambda_{i,c}}}.$$
.
However, when Σt and Σt,c do not commute, their respective eigen-decompositions cannot be directly aligned. In this case, we introduce *generalized eigenvalues* λ by solving

$\mathbf{d}$
Σ
−1
t v = λΣ
−1
t,c v.
Intuitively, these λ measure how Σt transforms relative to Σt,c along each direction. Note that we can rewrite the trace term in the expectation as

$$\mathrm{tr}\big{[}(\boldsymbol{\Sigma}_{t}^{-1}-\boldsymbol{\Sigma}_{t,c}^{-1})^{2}\,\boldsymbol{\Sigma}_{t,c}\big{]}=\mathrm{tr}\bigg{[}\big{(}\boldsymbol{\Sigma}_{t,c}^{-1/2}(\boldsymbol{\Sigma}_{t,c}^{1/2}\boldsymbol{\Sigma}_{t}^{-1}\boldsymbol{\Sigma}_{t,c}^{1/2}-\mathbf{I})\boldsymbol{\Sigma}_{t,c}^{-1/2}\big{)}^{2}\boldsymbol{\Sigma}_{t,c}\bigg{]}$$ $$=\mathrm{tr}\bigg{[}\big{(}\boldsymbol{\Sigma}_{t,c}^{1/2}\,\boldsymbol{\Sigma}_{t}^{-1}\,\boldsymbol{\Sigma}_{t,c}^{1/2}\,-\,\mathbf{I})^{2}\,\boldsymbol{\Sigma}_{t,c}^{-1}\bigg{]}$$ $$=\sum_{k=1}^{d}\sum_{j=1}^{d}(\lambda_{k}-1)^{2}\,w_{k,j},$$

where wk,j are weights induced by Σ
$${\mathrm{red~by~}}\Sigma_{t,c}^{-1}.{\mathrm{~The~}}\lambda_{k}$$
t,c . The λks are eigenvalues of Σ
lues of $\Sigma_{t,c}^{1/2}\Sigma_{t}^{-1}\Sigma_{t,c}^{1/2}\,.$ Since
$$\mathbf{\Sigma}_{t,c}^{1/2}\mathbf{\Sigma}_{t}^{-1}\mathbf{\Sigma}_{t,c}^{1/2}\mathbf{y}=\lambda\mathbf{y},$$
$${\mathrm{setting~}}\mathbf{v}=\Sigma_{t,c}^{1/2}\mathbf{y}{\mathrm{~yields}}$$
$${\frac{\mathbf{v}^{\top}\Sigma_{t}^{-1}\mathbf{v}}{\mathbf{v}^{\top}\Sigma_{t,c}^{-1}\mathbf{v}}}=\lambda,$$
When $\lambda<1$, since . 
$$\mathbf{\Sigma}_{t,c}^{1/2}\mathbf{\Sigma}_{t}^{-1}\mathbf{v}=\lambda\mathbf{\Sigma}_{t,c}^{-1/2}\mathbf{v}\implies\mathbf{\Sigma}_{t}^{-1}\mathbf{v}=\lambda\mathbf{\Sigma}_{t,c}^{-1}\mathbf{v}.$$
the unconditional covariance Σt is effectively larger (less sharp) in that eigen-direction, indicating that the conditional distribution is sharper by comparison. Consequently, the difference ∥ s(xt, c) − s(xt)∥ encodes how much sharper (or flatter) the conditional distribution is along each generalized eigenvector. This extends the simpler commuting-case result discussed in the main text, providing a more general interpretation of Wen's metric in terms of non-commuting covariances.

## A.4. Score Difference Norm And Fisher-Rao Equivalence

Here, we show that for small perturbations δΣt, the local geometry prescribed by the Fisher-Rao metric coincides with that implied by the expected squared norm of the score difference. Specifically, let Σt,c = Σt + δΣt with ∥δΣt∥ ≪ 1. By expanding both the Fisher-Rao distance and the expected score-difference norm in powers of δΣt up to second order, we find that their expansions match exactly in this limit. Importantly, this matching of expansions implies that the derivatives of the two measures with respect to Σt also coincide (i.e., as δΣt → 0). In other words, the local (infinitesimal) curvature on the covariance manifold-in other words, the Riemannian structure encoded by the second-order terms-is the same whether we measure distance via Fisher-Rao or via the expected score-difference norm. Consequently, both metrics capture how conditioning sharpens the learned distribution in precisely the same way under small perturbations, thereby confirming that the two approaches share the same local geometry on the Gaussian covariance manifold.

The Fisher-Rao (or affine-invariant) distance (A. Micchelli & Noakes, 2005) between Σt and Σt,c is

$$d_{\mathrm{FR}}(\mathbf{\Sigma}_{t},\mathbf{\Sigma}_{t,c})^{2}\ =\ \left\|\log\biggl(\mathbf{\Sigma}_{t,c}^{-1/2}\,\mathbf{\Sigma}_{t}\,\mathbf{\Sigma}_{t,c}^{-1/2}\biggr)\right\|_{F}^{2}.$$

In particular, we show that for small perturbations in Σt, the expected norm of the score difference coincides with this squared Fisher-Rao distance up to second order. Define a small perturbation on Σt as δΣt, where δ can be arbitrarily small. Let Σt,c = Σt + δΣt, with Σt ≻ 0 and ∥δΣt∥ ≪ 1 so that Σt,c remains positive-definite. Define

$$H^{\Delta}\;:=\;\Sigma_{t}^{-1}\;-\;\Sigma_{t,c}^{-1}.$$
Since $s(\mathbf{x}_t,c)=-\boldsymbol{\Sigma}_{t,c}^{-1}(\mathbf{x}_t-\boldsymbol{\mu})$ and $s(\mathbf{x}_t)=-\boldsymbol{\Sigma}_t^{-1}(\mathbf{x}_t-\boldsymbol{\mu})$, their difference is:
$$s^{\Delta}({\bf x}_{t})\;=\;H^{\Delta}\,({\bf x}_{t}-\mu).$$
Hence,
$$\mathbb{E}_{\mathbf{x}_{t}\sim p_{t}(\mathbf{x}_{t}|c)}\Big[\|s^{\Delta}(\mathbf{x}_{t})\|^{2}\Big]\ =\ \operatorname{tr}\!\left((H^{\Delta})^{2}\,\Sigma_{t,c}\right).$$

14

$${\mathcal{L}}_{t,c}=({\boldsymbol{\Sigma}}_{t}+\delta{\boldsymbol{\Sigma}}$$

Next, expand Σ
−1 t,c = (Σt + δΣt)
−1 using the Neumann series. Up to O(∥δΣt∥
2),

$$\mathbf{\Sigma}_{t,c}^{-1}\;\approx\;\mathbf{\Sigma}_{t}^{-1}-\mathbf{\Sigma}_{t}^{-1}\,\delta\mathbf{\Sigma}_{t}\,\mathbf{\Sigma}_{t}^{-1}\,,$$
$${\mathrm{which~yields}}$$
$$H^{\Delta}\;\approx\;\Sigma_{t}^{-1}\,\delta\Sigma_{t}\,\Sigma_{t}^{-1},\quad(H^{\Delta})^{2}\;\approx\;\Sigma_{t}^{-1}\,\delta\Sigma_{t}\,\Sigma_{t}^{-1}\,\delta\Sigma_{t}\,\Sigma_{t}^{-1}.$$
Then,
$$\mathrm{tr}\Bigl[(H^{\Delta})^{2}\,\Sigma_{t,c}\Bigr]\;\approx\;\mathrm{tr}\Bigl(\Sigma_{t}^{-1}\,\delta\Sigma_{t}\,\Sigma_{t}^{-1}\,\delta\Sigma_{t}\Bigr).$$
$\mathfrak{so}$. 
$$(H^{\Delta})^{2}\,\Sigma_{t,c}\;\approx\;\Sigma_{t}^{-1}\,\delta\Sigma_{t}\,\Sigma_{t}^{-1}\,\delta\Sigma_{t},$$

On the other hand, consider the Fisher-Rao distance:

$$d_{\mathrm{FR}}^{2}(\mathbf{\Sigma}_{t},\mathbf{\Sigma}_{t,c})\;\approx\;\left\|\log\!\left(\mathbf{\Sigma}_{t,c}^{-1/2}\,\mathbf{\Sigma}_{t}\,\mathbf{\Sigma}_{t,c}^{-1/2}\right)\right\|_{F}^{2}.$$

Define A := Σ
$$\stackrel{-1/2}{t,c}\Sigma_{t}\;\Sigma_{t,c}^{-1/2}\;$$
t,c . Since δΣt is small, we can write A ≈ I + X with ∥X∥ ≪ 1. Then,
$$\log(\mathbf{A})\;\approx\;\mathbf{X},\quad\|\log(\mathbf{A})\|_{F}^{2}\;\approx\;\|\mathbf{X}\|_{F}^{2}.$$
$\mathfrak{SN}$. 
$$\mathbf{\omega}\mathbf{\omega}_{t}$$
It can be shown (via expansion in δΣt) that ∥X∥
2F matches tr(Σ
$\Sigma^{-1}\,\delta\Sigma$. 
tδΣt Σ
−1
tδΣt) up to second order, leading to
$$d_{\mathrm{FR}}^{2}(\mathbf{\Sigma}_{t},\mathbf{\Sigma}_{t,c})\;\approx\;\mathrm{tr}\!\left(\mathbf{\Sigma}_{t}^{-1}\,\delta\mathbf{\Sigma}_{t}\,\mathbf{\Sigma}_{t}^{-1}\,\delta\mathbf{\Sigma}_{t}\right).$$
Hence, combining the two expansions shows:

$$\mathbb{E}_{\mathbf{x}_{t}\sim p_{t}\left(\mathbf{x}_{t}\left|c\right.\right)}\left[\left\|s^{\Delta}(\mathbf{x}_{t})\right\|^{2}\right]\quad{\mathrm{and}}\quad d_{\mathrm{FR}}^{2}(\mathbf{\Sigma}_{t},\mathbf{\Sigma}_{t,c})\,.$$

coincide to second order in ∥δΣt∥. Thus, in the small-perturbation limit, the expected value of the squared norm of the score difference encodes the same information as the Fisher-Rao distance, affirming that Wen's metric indeed captures how conditioning sharpens the learned distribution from a Riemannian perspective.

## B. Proofs

B.1. Proof of Lemma 4.1 State. For a Gaussian vector x ∼ N (µ, Σ),

$$\mathbb{E}\big[\|s(\mathbf{x})\|^{2}\big]\ =\ -\operatorname{tr}\bigl(H(\mathbf{x})\bigr),$$

where H(x) ≡ − Σ
−1*is the Hessian of the log-density.*
Proof. A Gaussian log-density has

$$\log p(\mathbf{x})=\mathbf{\nabla}-{\frac{1}{2}}(\mathbf{x}-{\boldsymbol{\mu}})^{\top}{\boldsymbol{\Sigma}}^{-1}(\mathbf{x}-{\boldsymbol{\mu}})+\mathrm{const.},$$
so $H({\bf x})=-\,\Sigma^{-1}$ and $s({\bf x})=-\,\Sigma^{-1}({\bf x}-\mu)$. Then
$$\|s(\mathbf{x})\|^{2}=\ (\mathbf{x}-{\boldsymbol{\mu}})^{\top}{\boldsymbol{\Sigma}}^{-2}(\mathbf{x}-{\boldsymbol{\mu}}).$$
$$-\,\mu)]=\operatorname{tr}(x)$$

Taking expectation, using E[(x − µ)
⊤A (x − µ)] = tr(A Σ)), we get E[∥s(x)∥

$$\mathbf{x})\|^{2}]=\operatorname{tr}(\mathbf{\Sigma}^{-1})=-\operatorname{tr}(H(\mathbf{x})).$$

This result generalizes to non-Gaussian distributions under weak regularity conditions (Hyvärinen, 2005). Although we chose the Gaussian assumption to facilitate theoretical extensions and applications, we will still present the original generalization here.

$$\begin{array}{l}{\square}\end{array}$$

B.2. Generalization of Lemma 4.1 State. For a random vector x ∼ p(x) with regularity conditions E-||s(x)||2< ∞ and lim
∥x∥→∞
p(x)s(x) = 0,

$$\mathbb{E}\left[\|s(\mathbf{x})\|^{2}\right]=-\mathbb{E}\left[\operatorname{tr}(H(\mathbf{x}))\right].$$

Proof. Write si(x) = ∂xilog p(x). Because si p = ∂xi p,

$$\mathbb{E}{\big[}\|s(\mathbf{x})\|^{2}{\big]}=\sum_{i=1}^{d}\int s_{i}(\mathbf{x})\,\partial_{x_{i}}p(\mathbf{x})\,d\mathbf{x}.$$
For each i integrate by parts:
$$\int s_{i}\,\partial_{x_{i}}p=\int\partial_{x_{i}}[p\,s_{i}]\,d\mathbf{x}-\int p\,\partial_{x_{i}}s_{i}\,d\mathbf{x}.$$

The first term is a surface integral over the sphere of radius R; by the assumed boundary condition it vanishes as R → ∞.

Hence Rsi ∂xi p = −Rp ∂xisi. Summing over i gives

$$\mathbb{E}[\|s(\mathbf{x})\|^{2}]=-\int p(\mathbf{x})\,\sum_{i=1}^{d}\partial_{x_{i}}s_{i}(\mathbf{x})\,d\mathbf{x}=-\mathbb{E}[\operatorname{tr}(H(\mathbf{x}))].$$

B.3. Proof of Lemma 4.2 State. For x ∼ N (µ, Σ) and x|c ∼ N (µc, Σc):

$$\mathbb{E}_{\mathbf{x}\sim p(\mathbf{x}|c)}\big[\|s(\mathbf{x},c)-s(\mathbf{x})\|^{2}\big]=\|H(\mathbf{x})(\boldsymbol{\mu}-\boldsymbol{\mu}_{c})\|^{2}+\operatorname{tr}\big[(H(\mathbf{x})-H_{c}(\mathbf{x}))^{2}\,H_{c}^{-1}(\mathbf{x})\big],$$

where H(x) ≡ −Σ
−1and Hc(x) ≡ −Σ
−1 c.

Additionally, if ΣΣc = ΣcΣ and µ = µc*, then*

$$\mathbb{E}_{\mathbf{x}\sim p(\mathbf{x}|c)}\left[\|s(\mathbf{x},c)-s(\mathbf{x})\|^{2}\right]=\sum_{i=1}^{d}{\frac{(\lambda_{i}-\lambda_{i,c})^{2}}{\lambda_{i,c}}},$$

where λi, λi,c are eigenvalues of H(x) and Hc(x).

Proof. Let s(x) = − Σ
−1(x−µ) and s(x, c) = − Σ
−1 c(x−µc) denote the Gaussian score functions for the unconditional and conditional distributions. Then

$$s({\bf x},c)-s({\bf x})\;=\;-\,\Sigma_{c}^{-1}({\bf x}-{\boldsymbol{\mu}}_{c})\;+\;\Sigma^{-1}({\bf x}-{\boldsymbol{\mu}}).$$

Taking the expectation,

Ex∼p(x|c)-∥ − Σ −1 c(x − µc) + Σ −1(x − µ)∥ 2=Ex∼p(x|c)-∥Σ −1 c(x − µc)∥ 2 + Ex∼p(x|c) -∥Σ −1(x − µ)∥ 2 − Ex∼p(x|c)-(x − µc) ⊤Σ −1 c Σ −1(x − µ) − Ex∼p(x|c) -(x − µ) ⊤Σ −1Σ −1 c(x − µc) =tr(Σ −1 c) + tr(Σ −2Σc) + (µc − µ) ⊤Σ −2(µc − µ) − tr(Σ −1 c Σ −1Σc) − tr(Σ −1Σ −1 c Σc) =∥Σ −1(µc − µ)∥ 2 + tr(Σ −1 − Σ −1 c) 2Σc.
16 if µ = µc, and ΣΣc = ΣcΣ so that Σ
−1and Σ
−1 care simultaneously diagonalizable as Σ
−1 = QΛQ⊤ and Σ
−1 c =
QΛcQ⊤, the trace term becomes

$$\text{tr}(\mathbf{\Sigma}^{-1}-\mathbf{\Sigma}_{c}^{-1})^{2}\mathbf{\Sigma}_{c})=\text{tr}(\mathbf{Q}(\mathbf{\Lambda}-\mathbf{\Lambda}_{c})^{2}\mathbf{\Lambda}_{c}^{-1}\mathbf{Q}^{\top})=\text{tr}((\mathbf{\Lambda}-\mathbf{\Lambda}_{c})^{2}\mathbf{\Lambda}_{c}^{-1})$$ $$=\sum_{i=1}^{d}\frac{(\lambda_{i}-\lambda_{i,c})^{2}}{\lambda_{i,c}}.$$
$$=\sum_{i=1}^{d}{\frac{(\lambda_{i}-\lambda_{i,c})^{2}}{\lambda_{i,c}}}.$$
$\square$
B.4. Proof of Lemma 4.3 State. For a Gaussian vector x ∼ N (µ, Σ),

$\square$
E
$$\mathbb{E}\left[\|H(\mathbf{x})s(\mathbf{x})\|^{2}\right]=-\mathrm{tr}((H(\mathbf{x}))^{3})$$
where H(x) ≡ −Σ *is the Hessian of the log density.*

$\mathbf{a}$ $\mathbf{a}$
Proof. As H(x) = − Σ
$$I(\mathbf{x})=-\,\Sigma^{-1}{\mathrm{~and~}}s(\mathbf{x})=-\,\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}),$$
$$\mathbb{E}\left[\|H(\mathbf{x})s(\mathbf{x})\|^{2}\right]=\mathbb{E}\left[(\mathbf{x}-\boldsymbol{\mu})^{\top}\boldsymbol{\Sigma}^{-4}(\mathbf{x}-\boldsymbol{\mu})\right]=\operatorname{tr}(\boldsymbol{\Sigma}^{-3})=-\operatorname{tr}(H(\mathbf{x})^{3}).$$

## C. Details Of The Toy Experiments

This section provides additional details on the 2D and MNIST experiments discussed in Section 4.1. For both experiments, we use the DDPM (Ho et al., 2020) framework with the DDIM (Song et al., 2021a) sampler, employing 500 sampling steps. Additionally, to obtain a more accurate estimate of the Hessian (Jacobian of the score function), we utilize the second-order score matching loss proposed by Lu et al. (2022) during model training. Refer to Appendix A.1 for details.

2D Mixture of Gaussian Experiment. We use a mixture of Gaussians with two modes equidistant from zero but with differing covariance scales. One mode is designed with an extremely small covariance to induce a sharp peak, representing memorization, while the other mode has a larger covariance for the opposite case. The mixture ratio between the two modes is 5:95, with a dataset comprising 3,000 samples in total. Empirically, we observed that only samples from the mode with extremely small covariance exhibited memorization, indicated by extremely small ℓ2 distances between the generated samples and training samples. MNIST Experiment. In the MNIST experiment, we use two digits: "3" for the generalized case and "9" for the memorized case, with 3,000 samples each. Classifier-free guidance (Ho & Salimans, 2021) (CFG) is employed, training the unconditional score function s(xt) with a probability p = 0.2 using all 6,000 samples. For s(xt, c), all samples of digit "3" are used to enable generalization and diversity, while a single sample of digit "9"
(duplicated 100 times) is used to collapse the model's output for this digit into a single conditioned image. Sampling is performed with a guidance scale of 5. As expected, even with CFG, the model generates only a single image for digit "9," while producing diverse outputs for digit "3." In Figure 2, for the non-memorized case, we sample 1,000 images and select the top 500 samples with the largest pairwise ℓ2 distances from training samples to highlight cases clearly deviating from memorization. For the memorized case, as all images collapse into a single image, we sample 500 outputs without comparing ℓ2 distances.

## D. Details Of The Stable Diffusion Experiments

This section describes the experimental setups for the Stable Diffusion experiments presented in Section 4.5 and Section 5. We provide a detailed overview of the configurations, including the specific prompts used and the implementation details of the baseline methods.

Models. We use Stable Diffusion v1.4 and v2.0, the same versions in which memorized prompts were identified by (Wen et al., 2024). For both detection and mitigation experiments, we use the DDIM sampler (Song et al., 2021a) with 50 sampling steps following Wen et al. (2024); Ross et al. (2024).

## Prompt Configuration.

- **Memorized Prompts:** Following recent studies (Wen et al., 2024; Ren et al., 2024; Ross et al., 2024; Chen et al.,
2024), we use memorized prompts identified by Webster (2023) in our experiments. Webster (2023) categorized memorized prompts into three types: 1) *Matching Verbatim (MV)*: Generated images are exact pixel-by-pixel matches with the original paired training image. 2) *Template Verbatim (TV)*: Generated images partially resemble the training image but may differ in attributes like color or style. 3) *Retrieval Verbatim (RV)*: Generated images memorize certain training images but are associated with prompts different from the original captions. The categorization of MV, TV, and RV considers both the memorized portions of generated images and their associations with specific prompt-image pairs. For instance, a prompt generating a pixel-perfect match to a training image is classified as RV, not MV, if the prompt differs from the original training caption. However, in our study, these categories are used to differentiate between images that are exact pixel-level matches and those that replicate specific attributes, such as style or color. For simplicity, we refer exact matches as **Exact Memorization (EM)** and partial matches as **Partial Memorization (PM)**, without considering their caption associations. For detection experiments, we combine prompts from all categories, resulting in a total of 500 memorized prompts for Stable Diffusion v1.4, identical to the prompts used by Wen et al. (2024), and 219 prompts for v2.0. While detection experiments only require a prompt set, mitigation experiments necessitate access to the original training images to evaluate SSCD (Pizzi et al., 2022) scores. Consequently, prompts without accessible training images are excluded, resulting in 454 prompts for v1.4 and 202 prompts for v2.0.

- **Non-memorized Prompts:** To ensure a diverse distribution of non-memorized prompts, we compile a total of 500 prompts drawn from COCO (Lin et al., 2014), Lexica (Lexica, 2024), Tuxemon (HuggingFace, 2024), and GPT-4 (Achiam et al., 2023). Specifically, the GPT-4 prompts are a random subset of those used by (Ren et al., 2024).

## D.1. Memorization Detection

Details for Baseline Methods. We provide details of each baseline detection algorithm.

- Tiled ℓ2 **distance**: Building on the insight that memorized prompts produce similar generations regardless of their initializations, Carlini et al. (2023) propose examining generation density by analyzing multiple generated images for a given prompt using pairwise ℓ2 distances in pixel space. To address false positives from similar backgrounds, Carlini et al. (2023) divide images into non-overlapping 128 × 128 tiles and compute the maximum ℓ2 distance between corresponding tiles. We adopt the identical setting for both Stable Diffusion v1.4 and v2.0. As the detection performance of this metric achieves the best after full sampling steps, we only report the complete **50-step** results in Table 1.

- **(Ren et al., 2024)**: Based on the empirical observation that patterns in attention scores for specific tokens (termed as
"trigger tokens") behaves differently in memorized samples, Ren et al. (2024) introduce the detection score D and layer-specific entropy Elt=T
as primary indicators of memorization.

The first metric D, which we refer to **Average Entropy (AE)** for intuitive notation, is defined as:

$$A E={\frac{1}{T_{D}}}\sum_{t=0}^{T_{D}-1}E_{t}+{\frac{1}{T_{D}}}\sum_{t=0}^{T_{D}-1}|E_{t}^{\mathrm{summary}}-E_{T}^{\mathrm{summary}}|,$$

where Et represents attention entropy, measuring the dispersion of attention scores across different tokens:

$$E_{t}=\sum_{i=1}^{N}-\overline{{{a}}}_{i}\log(\overline{{{a}}}_{i}).$$

In addition, E
summary tis the entropy computed only on the summary tokens, and TD =
T
5 corresponds to the last T5 steps of the reverse diffusion process used for memorization detection.

The second metric, layer-specific entropy Elt=T, which we refer to **Layer Entropy (LE)**, is computed at the first diffusion step and focuses on specific U-Net layers:

$$L E=\sum_{i=1}^{N}-\overline{{{a}}}_{i}^{l}\log(\overline{{{a}}}_{i}^{l}),$$

where a liis the average attention score in layer l. For detection experiments, we follow the implementation and hyperparameter settings of Ren et al. (2024). The detection performance differences between our results in Table 1 and those reported in Ren et al. (2024) can be attributed to different choices of non-memorized prompts. Specifically, our evaluation uses prompts collected from diverse sources, whereas Ren et al. (2024) utilizes GPT-4 generated prompts that share similar characteristics. For comprehensive experimental details, we refer readers to Ren et al. (2024).

- **(Wen et al., 2024)**: Building on the insight that significant text guidance induces memorized samples during sampling, Wen et al. (2024) propose using the magnitude of predicted noise difference between conditional and unconditional noise. It is defined as:

$${\frac{1}{T}}\sum_{t=1}^{T}\|\mathbf{\epsilon}_{\theta}(\mathbf{x}_{t},c)-\mathbf{\epsilon}_{\theta}(\mathbf{x}_{t},\varnothing)\|,$$

where T denotes the number of timesteps, c denotes the specific embedded prompt, and ∅ denotes empty string, equivalent to unconditional case. Recall that diffusion forward process qt|0(xt|x0) = N (
√αtx0,(1 − αt)I) and therefore,

$$\nabla_{\mathbf{x}_{t}}\log p_{t}(\mathbf{x}_{t})=\mathbb{E}_{p_{0}(\mathbf{x}_{0})}\left[\nabla_{\mathbf{x}_{t}}\log q(\mathbf{x}_{t}|\mathbf{x}_{0})\right]\approx\mathbb{E}_{p_{0}(\mathbf{x}_{0})}\left[-{\frac{\epsilon_{\theta}(\mathbf{x}_{t})}{\sqrt{1-\alpha_{t}}}}\right]=-{\frac{\epsilon_{\theta}(\mathbf{x}_{t})}{\sqrt{1-\alpha_{t}}}}=s_{\theta}(\mathbf{x}_{t}).$$

Thus,

$$\|s_{\theta}({\bf x}_{t},c)-s_{\theta}({\bf x}_{t})\|=\frac{1}{\sqrt{1-\alpha_{t}}}\|\epsilon_{\theta}({\bf x}_{t},c)-\epsilon_{\theta}({\bf x}_{t},\emptyset)\|.$$

Consequently, Wen's metric can be defined as the norm of score differences as described in Section 4.3.

- **(Chen et al., 2024)**: Building on the observation that the end token exhibits abnormally high attention scores for memorized prompts, specifically highlighting the memorized region, Chen et al. (2024) leverage this attention score as a mask to amplify the detection of the **Partial Memorization (PM)** cases. We refer this metric as **Bright Ending (BE)** for short.

In detail, Chen et al. (2024) multiply the attention mask m on Wen's metric:

$$B E={\frac{1}{T}}\sum_{t=1}^{T}\left\|\left(\mathbf{\epsilon}_{\theta}(\mathbf{x}_{t},c)-\mathbf{\epsilon}_{\theta}(\mathbf{x}_{t},\emptyset)\right)\circ\mathbf{m}\right\|{\bigg/}\left({\frac{1}{N}}\sum_{i=1}^{N}m_{i}\right),$$
$$\mathrm{on}\ 4.3.$$

where N denotes for the number of elements in the mask m, therefore the result is normalized by the mean of m. We note that the attention mask m is obtainable at the final sampling step (t = 1). Therefore, to utilize BE as a detection metric, the model requires completion of all sampling steps. Consequently, in Table 1, we report experimental results using the complete **50-step** diffusion process. In addition, following the identical setup as Chen et al. (2024), we average attention scores from the first two downsampling layers of U-Net to obtain m for both Stable Diffusion v1.4 and v2.0. For additional details, refer to the original paper of Chen et al. (2024).

## D.2. Memorization Mitigation

Details for Baseline Methods. We provide details for each recent baseline mitigation algorithm. For every mitigation strategy, results are averaged over five generations per memorized prompt. Additionally, each baseline is evaluated using five different hyperparameter settings, which are described in detail below.

- **Random Token Addition (RTA) & Random Number Addition (RNA)**: Somepalli et al. (2023b) propose mitigation strategies that perturb prompts by adding arbitrary tokens or numbers. Following Wen et al. (2024), we insert tokens or numbers in quantities of {1, 2, 4, 6, 8} for both RTA and RNA.

- **(Ren et al., 2024)**: Ren et al. (2024) propose a mitigation strategy that involves masking memorization-inducing tokens and rescaling the attention scores of the beginning token using a hyperparameter C. After token masking, we evaluate the approach by varying C within the range {1.1, 1.2, 1.25, 1.3, 1.5} for both v1.4 and 2.0.

- **(Wen et al., 2024)**: As explained in Appendix D.1, Wen et al. (2024) propose a differentiable metric based on the norm of the difference between the conditional and unconditional scores. Since memorized prompts empirically exhibit a large magnitude for this term, Wen et al. (2024) optimize the text embedding by directly minimizing it.

Wen et al. (2024) introduce ℓ*target*, a hyperparameter for early stopping, to prevent the text embedding from deviating significantly from its original semantic meaning. Following Wen et al. (2024), we investigate ℓ*target* values ranging from 1 to 5 in Stable Diffusion v1.4. However, in v2.0, we found the generated results to be more sensitive. Therefore, for v2.0, we investigate ℓ*target* values in {1, 1.25, 1.5, 1.75, 2}.

## Algorithm 2 Sail Pseudo-Code

Require: Initialization xT ∼ N (0, I), Early stopping threshold ℓ*thres*, Score function s(·), Loss balancing term α, Step size η > 0 Ensure: Set LSAIL ← L0 {where L0 > ℓ*thres*}
1: **while** LSAIL > ℓ*thres* do 2: Compute s
∆
θ(xT ) := sθ(xT , c) − sθ(xT );
3: Normalize s
∆
θ(xT ) with δ and compute s
∆ θ xT + δs∆
θ(xT );
4: Compute SAIL objective:
5: LSAIL(xT ) :=s
∆ θ xT + δ · s
∆
θ(xT )− s
∆
θ(xT )
2 + α∥xT ∥
2; 6: Update initialization: xT ← xT − η ∇xT LSAIL;
7: **end while**
Details for Our Method. Algorithm 2 provides a pseudo-code for SAIL algorithm. While Algorithm 2 shows the case of optimizing a single xT , in practice, it can simultaneously search for several memorization-free candidates by collectively optimizing several initializations in a batch fashion.

To employ SAIL, we need to set α and ℓ*thres*. We set α = 0.05 for Stable Diffusion v1.4 and α = 0.01 for v2.0. In practice, we observe that the generated results are largely insensitive to α, though keeping α sufficiently small helps balance the magnitude of two loss terms effectively. In addition, we investigate ℓthres ∈ {7.6, 7.8, 8.2, 8.6, 9} for v1.4 and
{4, 4.5, 5, 5.5, 6} for v2.0.

As the metric proposed by Wen et al. (2024) also captures sharpness, one may consider replacing the first term of LSAIL(xT ) with ∥sθ(xt, c) − sθ(xt)∥
2. However, we empirically find that this alternative fails to converge and is therefore ineffective for mitigation. This may be due to the higher sensitivity of our proposed metric during the initial phase of generation. Details of prompts in Figure 6. We provide full prompt details with a key prompt detail in **bold**, starting from top image.

- <i>The **Colbert** *Report<i> Gets End Date* - **Björk** *Explains Decision To Pull <i>Vulnicura<i> From Spotify* - **Netflix** *Hits 50 Million Subscribers*
- <em>**South Park***: The Stick of Truth<em> Review (Multi-Platform)*