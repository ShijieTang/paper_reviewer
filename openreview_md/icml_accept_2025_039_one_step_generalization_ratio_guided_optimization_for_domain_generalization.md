# 

Sumin Cho * 1 **Dongwon Kim** * 1 **Kwangsu Kim** 1

## Abstract

Domain Generalization (DG) aims to train models that generalize to unseen target domains but often overfit to domain-specific features, known as undesired correlations. Gradient-based DG methods typically guide gradients in a dominant direction but often inadvertently reinforce spurious correlations. Recent work has employed dropout to regularize overconfident parameters, but has not explicitly adjusted gradient alignment or ensured balanced parameter updates. We propose GENIE (Generalization-ENhancing Iterative Equalizer), a novel optimizer that leverages the One-Step Generalization Ratio (OSGR) to quantify each parameter's contribution to loss reduction and assess gradient alignment. By dynamically equalizing OSGR via a preconditioning factor, GENIE prevents a small subset of parameters from dominating optimization, thereby promoting domaininvariant feature learning. Theoretically, GENIE balances convergence contribution and gradient alignment among parameters, achieving higher OSGR while retaining SGD's convergence rate. Empirically, it outperforms existing optimizers and enhances performance when integrated with various DG and single-DG methods.

## 1. Introduction

Deep neural networks (DNNs) achieve high accuracy when training and test data share a similar distribution. However, in real-world applications, data distributions often shift, causing performance degradation(Muandet et al., 2013). Domain Generalization (DG) addresses this issue by training models to generalize to out-of-distribution data from unseen domains. The main challenge is to prevent overfitting
*Equal contribution 1Department of Computer Science and Engineering, University of Sungkyunkwan, Suwon, Korea.

Correspondence to: Sumin Cho <jsm0707@skku.edu>, Dongwon Kim <kdwaha@skku.edu>, Kwangsu Kim <kim.kwangsu@skku.edu>.

to domain-specific features—known as spurious correlations—while learning invariant features and causal relationships that generalize across diverse domains(Shi et al., 2022; Hemati et al., 2023; Shah et al., 2020a; Ye et al., 2024). Several DG methods have attempted to guide the gradient toward a dominant direction during training (Parascandolo et al., 2021; Shahtalebi et al., 2021; Shi et al., 2022; Rame et al., 2022). However, this dominant direction often itself driven by spurious features, inadvertently reinforcing undesired correlations. This suggests that aligning gradients toward a single dominant direction is insufficient to fully solve the problem, highlighting the need for other perspectives. A recent approach (Michalkiewicz et al., 2023) introduced a parameter-wise dropout mechanism based on Gradient Signal-to-Noise Ratios (GSNR) to suppress overly predictive parameters and reduce their influence on optimization.

While this strategy mitigates parameter updates driven by spurious correlations, it does not adjust the magnitudes of updates based on their individual contributions to general1 ization. This raises the open question of how to design optimizers that explicitly balance parameter updates according to their principled contributions to generalization, thereby mitigating the influence of spurious correlations. Motivated by this perspective, we propose Generalization- ENhancing Iterative Equalizer (GENIE), a novel optimizer for addressing parameter imbalance. Recent work (Liu et al., 2020) introduced the One-Step Generalization Ratio (OSGR) that measures how effectively a single gradient update reduces test loss compared to training loss, providing insight into a model's generalization potential. OSGR reflects the contributions of individual parameters to generalization, based on their convergence speed and degree of gradient alignment. To leverage this insight, GENIE integrates a preconditioning factor that dynamically balances parameter-wise OSGR throughout training. This prevents a small subset of parameters from dominating the optimization, thereby promoting more robust and domain-invariant feature learning.

Our theoretical analysis shows that existing optimizers typically focus on either convergence speed or gradient alignment, often resulting in suboptimal generalization. In contrast, GENIE explicitly balances both, achieving a higher OSGR while maintaining the convergence rate of SGD(Robbins & Monro, 1951) in non-convex settings. We empirically validated GENIE on five standard DG datasets(Li et al., 2017; Fang et al., 2013; Venkateswara et al., 2017; Beery et al., 2018; Peng et al., 2019) where it consistently outperformed established optimizers, even with extended iterations. Furthermore, using our optimizer in existing DG and Single-DG (SDG) algorithms enhances their performance. We summarize our contributions as follows:
- We propose GENIE, a novel optimizer that addresses the overlooked issue of parameter imbalance in DG. It suppresses over-predictive parameters while promoting balanced parameter updates.

- We incorporate OSGR, previously used as a generalization metric, into the optimizer's core principle. This provides an efficient and novel perspective on generalization for addressing DG.

- GENIE is a domain-agnostic optimizer. It is validated across multiple DG benchmarks and SDG tasks, demonstrating its broad applicability and scalability.

## 2. Related Work 2.1. Domain Generalization

Existing DG methods address domain shift through two main strategies: (1) Feature Alignment, which aims to align features across domains to ensure consistent optimization, including methods such as domain-invariant feature learning (Sun & Saenko, 2016; Arjovsky et al., 2019; Krueger et al., 2021), data augmentation (Xu et al., 2020; Yan et al., 2020; Wang et al., 2020), and feature disentanglement (Nam et al., 2021; Mahajan et al., 2021). (2) Gradient Alignment, which focuses on aligning gradients across domains to ensure stable learning dynamics. Representative approaches include minimizing gradient differences (Koyama & Yamaguchi, 2020), increasing gradient inner products (Shi et al., 2022), updating weights only when gradient directions align (Parascandolo et al., 2021; Shahtalebi et al., 2021), and reducing inter-domain gradient variance (Rame et al., 2022). Recently, Sharpness Aware Minima (SAM)(Foret et al., 2021) has improved in-distribution generalization, inspiring the development of optimizers specifically designed for OOD tasks (Zhang et al., 2024; Wang et al., 2023). However, most DG studies overlook imbalanced parameter updates caused by differences in convergence speed or generalization capacity during optimization.

## 2.2. Preconditioning

Preconditioning improves the efficiency of optimization algorithms by incorporating curvature information of the loss function or adjusting the magnitude and direction of parameter updates. It accelerates convergence and enhances stability during training and can be categorized into three main types (Ye, 2024; Amari et al., 2021) (1) Hessian Based Preconditioning: utilizes the inverse or approximations of the Hessian matrix to capture curvature information. (Montavon et al., 2012; Dennis & More´, 1977) (2) Adaptive Learning Rate Based Preconditioning: dynamically adjusts learning rates based on gradient magnitudes, as seen in optimizers like AdaGrad (Duchi et al., 2011), RMSProp (Hinton et al., 2012), and Adam (Kingma, 2014). (3) Normalization- Based Preconditioning: normalizes inputs and activations, as exemplified by Batch Normalization(Ioffe & Szegedy, 2015), to improve the Hessian's condition number and enhance training stability. Previous preconditioning methods aim to optimize speed and stability. The application of preconditioning to improve model generalization remains underexplored.

## 3. Method 3.1. Preliminary

To address the challenge of generalization in unseen target domains, a recent study(Liu et al., 2020) introduced the concept of OSGR R(*Z, n*). OSGR quantifies how well model updates contribute to generalization by measuring the ratio of loss reduction between test D′and training data D after a single optimization step:

$$R(Z,n)=\frac{\mathbb{E}_{D,D^{\prime}\sim\mathcal{Z}^{n}}\Delta L_{D^{\prime}}}{\mathbb{E}_{D\sim\mathcal{Z}^{n}}\Delta L_{D}},$$

ED∼Zn ∆LD, (1)
where ∆LD′ and ∆LD represent the loss changes on test and training data, respectively. OSGR is influenced by two key factors: (1) the contribution of each parameter to loss reduction, characterized by the gradient magnitude, and (2) the alignment of parameter gradients across the data distribution. Higher OSGR indicates better generalization, reflecting consistent and balanced parameter updates. To better understand these dynamics, the following theorem links OSGR to parameter-wise statistics:
Theorem 3.1 (From Paper(Liu et al., 2020)). The relationship between gradient updates and generalization can be expressed as follows:

$$R(Z,n)=1-{\frac{1}{n}}\sum_{j\in J}{\frac{\mathbb{E}_{D\sim\mathcal{Z}^{n}}[g_{j}^{2}]}{\sum_{j^{\prime}\in J}\mathbb{E}_{D\sim\mathcal{Z}^{n}}[g_{j^{\prime}}^{2}]}}\cdot{\frac{1}{r_{j}+{\frac{1}{n}}}},\,\,(2)$$

where J *denotes the set of parameter index,* g 2 j is the squared gradient magnitude, ρ 2 j is the noise variance, and n is the number of samples. Parameters with higher Gradient Signal-to-Noise Ratios (GSNR), defined as rj =
g 2 j ρ 2 j
, yield higher OSGR, contributing more significantly to generalization. A recent study (Michalkiewicz et al., 2023) leveraged GSNR to suppress overly predictive parameters during training, aiming to prioritize robust features and reduce noisy updates. However, this approach overlooks parameter-wise imbalances in OSGR, which limits overall generalization performance.

In this context, we propose a preconditioning-based approach that dynamically balances OSGR across parameters.

By incorporating parameter-specific preconditioning factors, our method ensures that updates are aligned with both gradient magnitude and noise characteristics, preventing overfitting to noisy or well-learned features. This strategy not only enhances generalization but also ensures stable convergence in diverse DG settings.

## 3.2. Proposed Method

Based on Theorem 3.1, Michalkiewicz et al. (2023) introduced a gradient-masking approach that prioritizes updates for parameters with low GSNR, aiming to enhance their contribution to generalization. They argue that boosting updates to low-GSNR parameters can increase the overall GSNR and thus improve the optimization signal-to-gradient ratio (OSGR). Inspired by this perspective, we hypothesize the following relationship: Conjecture Uniformly distributed OSGR across parameters indicate better generalization performance.

This conjecture guides the design of our method. Rather than modifying the dropout ratio across parameters, we introduce a preconditioning term that more accurately adjusts the OSGR. Next, we inject noise into all parameters to encourage exploration toward better optima. Finally, we apply random dropout to stabilize parameter updates and reduce overfitting.

## 3.2.1. Preconditioning

We propose a preconditioning factor pj to ensure balanced contributions of each parameter to the OSGR, thus enhancing generalization. The key idea is to maintain equitable parameter influence on the overall generalization performance throughout the optimization process. We propose the following corollary for this purpose. Corollary 3.2 (Preconditioning and OSGR). If each parameter j applies a preconditioner pj , the OSGR can be expressed as:

$$R^{\prime}(Z,n)=\sum_{j\in J}\frac{p_{j}\mathbb{E}_{D\sim\mathcal{Z}^{n}}[g_{j}^{2}]}{\sum_{j^{\prime}\in J}p_{j^{\prime}}\mathbb{E}_{D\sim\mathcal{Z}^{n}}[g_{j^{\prime}}^{2}]}\cdot\frac{1}{\frac{1}{n\cdot r_{j}}+1},\tag{3}$$

or equivalently:

$$R^{\prime}(Z,n)=1-\frac{1}{n}\sum_{j in J}\frac{p_{j}\mathbb{E}_{D\sim\mathcal{Z}^{n}}[g_{j}^{2}]}{\sum_{j^{\prime}\in J}p_{j^{\prime}}\mathbb{E}_{D\sim\mathcal{Z}^{n}}[g_{j^{\prime}}^{2}]}\cdot\frac{1}{r_{j}+\frac{1}{n}}.\tag{4}$$

From Corollary 3.2, to maintain a balanced influence of parameter j on the overall OSGR, we propose:

$$p_{j}=\frac{1}{\mathbb{E}_{D\sim\mathcal{Z}^{n}}\left[g_{j}^{2}\right]}\left(r_{j}+\frac{1}{n}\right).\tag{5}$$

This leads to the OSGR:

$$R^{\prime}(Z,n)=1-\frac{1}{n}\sum_{j\in J}\frac{1}{\sum_{j^{\prime}\in J}\left(r_{j^{\prime}}+\frac{1}{n}\right)}=1-\frac{1}{n\mathbb{E}_{j\in J}\left(r_{j}+\frac{1}{n}\right)},\tag{6}$$

where Ej∈Jrj +
1 n represents the average GSNR contribution across parameters.Without preconditioning, parameters with large g 2 jbut low GSNR may receive higher weights in the OSGR expression, inflating the subtraction term. Our preconditioning alleviates this issue and improves the OSGR. This dynamic adjustment with preconditioning mitigates parameter-wise imbalances, ensuring that wellgeneralized features are not overwhelmed by noisy or overly dominant parameters.

In implementation, we ignore the 1n term as n is sufficiently large, and clipping variance by *tanh*(
1 σ2 ) for stability. More detailed analysis on influence of variance is described in Section 3.3.3. This preconditioner pj is straightforward to compute and requires only the gradient statistics mt and variance σt, which can be estimated during training. This efficiency makes it suitable for a wide range of DG tasks.

## 3.2.2. Noise Injection

To enhance exploration during optimization, we introduce noise injection, where a noise term scaled by the variance is added to the gradient. Specifically, the noise scale is determined by 1 − tanh( 1 σ2 ), reducing noise for high variance parameters while increasing it for low variance parameters. Motivated by (Mansilla et al., 2021), this injection boosts updates to parameters with low preconditioning value.

## 3.2.3. Random Mask

To further stabilize updates and mitigate overfitting, we apply a *random dropout mask*. This mask, sampled from a Bernoulli distribution, selectively zeroes out gradient components. By applying random masking after the preconditioning step, all parameters are equally considered to ensure robust updates.

## 3.3. Analysis

We provide a comprehensive theoretical analysis of our method from three perspectives. First, we examine generalization through the OSGR, which highlights how our effectively balances OSGR value across parameters. Second, we formalize our approach under the PAC-Bayes framework, showing that our method explicitly minimizes a tighter generalization bound. Finally, we establish that our optimizer retains the convergence rate of standard SGD while enabling more robust generalization. Proofs are provided in Appendix C.

 $\bot$, GENERALIZEDATION Al. 

## 3.3.1. Generalization Analysis With Osgr

We obtain the following corollary regarding the OSGR of these optimizers: Corollary 3.3 (OSGR of Optimizers). The OSGR of our proposed optimizer is:

$$\mathcal{R}_{O u r s}=1-\frac{1}{n\mathbb{E}_{j\in J}\left(r_{j}+\frac{1}{n}\right)},$$
$a.OSCB$ across different. 
$$\mathcal{R}_{O u r s}\geq\mathcal{R}_{S G D}\approx\mathcal{R}_{A d a m}.$$
 , (7)
Comparing the resulting OSGR across different optimizers, we have:
ROurs ≥ RSGD ≈ R*Adam*. (8)
This corollary demonstrates that our proposed preconditioning achieves better generalization by attaining a higher overall OSGR. The following remarks provide further context and analysis: Remark 3.4 (Conceptual Components of Optimizers). The preconditioning applied by common optimizers can be viewed as the element-wise product of two conceptual components:
- **Convergence Term:** controls the effective step size, Algorithm 1 Algorithm for GENIE
Input: Mini-batches {Bt}
T
t=1, Learning Rate α, Total Steps T. Hyperparameters: β ∈ [0, 1], Dropout Probability p Initialize: Parameters θ0, m0 ← 0, v0 ← 0.

for t = 1 to T do Compute Gradient:

$$g_{t}=1$$
$$(\theta_{t},l)$$
$\{\;\;\}$ . 
$$v_{t}\leftarrow\beta$$
$$w_{t-1}+(1-\beta)$$
gt = ∇L(θt; Bt)
Calculate GSNR and Preconditioning:
Update Moving Averages:

mt ← βmt−1+ (1−β)gt, vt ← βvt−1+ (1−β)g
2
t
$$\sigma_{t}^{2}=v_{t}-m_{t}^{2},\quad r_{j}=\operatorname{tanh}(\frac{1}{\sigma_{t}^{2}})m_{t}^{2}$$
$${\hat{g}}_{t}\leftarrow{\frac{m_{t}}{1-\beta^{t}}}\cdot{\frac{1}{v_{t}}}\cdot r_{t}$$
$$\mathbf{Noise}$$
$$\mathbf{\mu}:\mathbf{Injectition}$$
Noise Injection:
$$N o i s e_{t}\leftarrow\xi_{t}\big[1-\operatorname{tanh}(\frac{1}{\sigma_{t}^{2}})\big],\quad\xi_{t}\sim\mathcal{N}(0,\sigma^{2})$$
Random Mask:

$$M_{j}\sim B e r n o u l i(p)$$
$${\hat{g}}_{t}\leftarrow({\hat{g}}_{t}+N o i s e_{t})\odot M$$

Update Parameters:

$$\theta_{t+1}\leftarrow\theta_{t}-\alpha\tilde{g}_{t}$$

end for Output: Final parameters θT +1.

$$\left(7\right)$$

thus contributing to faster convergence. It includes terms such as ED∼Zn [g 2 j
] or ED∼Zn [gj ].

- **Alignment Term:** adjusts gradients toward stable directions. It includes the GSNR term rj .

$$({\boldsymbol{\delta}})$$

Table 1 summarizes the convergence term, alignment term and their resulting OSGR, including SGD, Adam, and our method. Remark 3.5 (Optimizer-Specific Analysis). SGD maintains a baseline OSGR value with no explicit adjustment. Adam introduces a convergence component combined with a partial alignment factor. In contrast, our method effectively integrates both aspects in a balanced manner.

Overall, this analysis highlights how each optimizer's design affects generalization through gradient alignment and

| OPT.        | PRECONDITIONING   | OSGR                   | WEIGHT     |      |      |    |
|-------------|-------------------|------------------------|------------|------|------|----|
| CONVERGENCE | ALIGNMENT         | ED∼Zn [g 2             |            |      |      |    |
| 1           | Wj ·              | 1                      | j ]        |      |      |    |
| SGD         | -                 | -                      | 1 −        | X    | Wj = |    |
| n           | rj + 1            | P j ′ ED∼Zn [g 2 j ′ ] |            |      |      |    |
| n           |                   |                        |            |      |      |    |
| j∈J         | q ED∼Zn [g        |                        |            |      |      |    |
| s           | 1                 | 2 ]                    |            |      |      |    |
| ADAM        | 1                 | X Wj ·                 | 1          | j    |      |    |
| + 1         | Wj =              |                        |            |      |      |    |
| ED∼Zn (gj ) | 1                 | + 1                    | 1          |      |      |    |
| n·rj        | j∈J               | n·rj                   | q ED∼Zn [g |      |      |    |
| P j ′       | j ′ ] 2           |                        |            |      |      |    |
| GENIE       | 1                 | 1                      | 1          | Wj · | 1    | 1  |
| )           | rj + n            | 1 −                    | X          |       | Wj = |    |
| ED∼Zn (g    | Ej∈J  rj +        |                        |            |      |      |    |
| 2           | n                 | 1                      | |J|        |      |      |    |
| j           | j∈J               | n                      |            |      |      |    |

convergence speed.Incorporating both perspectives, Our method leads to a higher OSGR and thus improves generalization performance. Furthermore, we demonstrate that the alignment term in our preconditioning achieves a higher OSGR value than those of existing preconditioning methods. Detailed justifications are provided in the Appendix C.

## 3.3.2. Generalization Analysis With Pac-Bayes Bound

While the previous analysis is based on alignment and convergence dynamics using OSGR, we now adopt a complementary perspective grounded in the PAC-Bayes framework.

We formulate the generalization analysis under a one-step update setting, where the KL divergence between successive parameter distributions reveals the connection between our preconditioning and a tighter generalization bound. Theorem 3.6 (PAC-Bayes Interpretation of Preconditioning). R(θ) is the population risk and L(θ) is empirical risk. Assume that the loss function L(θ) *is bounded in* [0, C]*. For* any λ > 0, with probability at least 1−δ *over the draw of* D, and for any data-dependent distribution p˜ over parameters θ*, the following PAC-Bayes bound holds:*

$$\mathbb{E}_{\theta\sim{\tilde{p}}}[R(\theta)]\leq\underbrace{\mathbb{E}_{\theta\sim{\tilde{p}}}[L(\theta)]}_{T_{1}}+\underbrace{\lambda C^{2}}_{8n}+\underbrace{\mathrm{KL}({\tilde{p}}\|\pi)+\log{\frac{1}{\delta}}}_{\lambda}$$
.
Assume that p˜ = N (θt+1, Σp˜) and π = N (θt, Σπ)*, where* Σp˜ = diag(qj · ρ 2 j) and Σπ = diag(ρ 2 j)*. Let* qj =
E[g]
2 E[g 2 j]
be a variance adaptation factor from SVAG optimizer(Balles &
Hennig, *2018) that minimizes the variance to reduce* max T1 term. (θt+1 = θt − q ⊙ g) Then, minimizing the T2 term via gradient descent yields an update direction:

$$\nabla_{\theta_{t}}\mathrm{KL}({\tilde{p}}\|\pi)=\underbrace{{\frac{1}{\mathbb{E}[g_{j}^{2}]}}\cdot{\frac{\mathbb{E}[g_{j}]^{2}}{\rho_{j}^{2}}}}_{G E N I E}\cdot g_{j,t},$$

which matches the preconditioning rule of our optimizer.

Remark 3.7 (Sharpness and Generalization via KL). This result shows that our method not only improves sharpness—as done in SAM—but also directly enhances generalization by minimizing both terms in the PAC-Bayes bound. Specifically, the variance adaptation factor qj reduces the variability of scaled gradients, thereby tightening the empirical loss term T1 through more stable updates. Simultaneously, the 1 ρ2 term minimizes the KL divergence term T2. This result shows our the generalization property of GENIE comes from correlation with Pac-Bayes theory.

## 3.3.3. Convergence Analysis

This section analyzes the convergence properties of GENIE under non-convex settings. Specifically, we adopt three widely used assumptions in the optimization literature: Assumption 3.8. (Bounded Gradient) There exists a constant G > 0 such that

$$\|\nabla{\mathcal{L}}(\theta_{t})\|\leq G\quad{\mathrm{for~all~}}t.$$
$$({\mathfrak{H}})$$

Assumption 3.9. (L-smooth) The loss function L is L-
smooth, meaning there exists a constant L > 0 such that for all θ1, θ2:

$$\|\nabla{\cal L}(\theta_{1})-\nabla{\cal L}(\theta_{2})\|\leq L\|\theta_{1}-\theta_{2}\|.\tag{10}$$

Assumption 3.10. (Lower bounded variance) The variance of the stochastic gradients have lower bound by a constant 1/Su:

$$\mathbb{E}[||g_{t}-\nabla\mathcal{L}(\theta_{t})||^{2}]\geq1/S_{u},\quad\forall t.\tag{11}$$

Under these assumptions, we establish the following result regarding the convergence rate: Theorem 3.11. Under Assumption 3.8 Assumption 3.9, and Assumption 3.10 the average gradient norm over T iterations can be expressed as:

$$\mathbb{E}[\|\nabla\mathcal{L}(\theta)\|^{2}]\leq O\left(\frac{1}{P_{l}}\left(1+\frac{G\cdot S_{u}^{2}}{2}\right)\frac{1}{\sqrt{\tilde{T}}}\right).\tag{12}$$

where Pl*is lower bound of preconditioning value.*
Remark 3.12 (Convergence Rate and Intuition). Theorem 3.11 shows that the average gradient norm converges at O(T
−1/2), the standard rate for stochastic gradient methods in non-convex optimization. This implies that GENIE retains the fundamental convergence properties of SGD.

Remark 3.13 (Influence of G·Su and Su). The term G·S
2 u represents a trade-off associated with the GSNR. A higher GSNR upper bound( G · Su) indicates a stronger gradient signal, which enhances generalization performance. However, it also acts as a multiplicative factor in the gradient norm, potentially slowing down convergence and thereby creating a trade-off. Furthermore, the variance term(Su) has a significant impact on the bound, further influencing the overall convergence behavior. To address this issue, we regulate the variance term using the *tanh* function, which effectively balances the interplay between generalization and convergence dynamics.

## 4. Experiment

Dataset. We followed the standardized protocols of DomainBed (Gulrajani & Lopez-Paz, 2021), which include dataset splits, hyperparameter searches, and model selection using validation sets. Our approach was evaluated on five DG benchmark datasets: PACS (Li et al., 2017), VLCS (Fang et al., 2013), OfficeHome (Venkateswara et al., 2017), TerraIncognita (Beery et al., 2018), and DomainNet (Peng et al., 2019). Evaluation. In accordance with DomainBed protocols, models were trained for 15,000 iterations on DomainNet and 5,000 iterations on the other datasets. For all DG and SDG experiments, we employed the Training-domain Validation Set approach, partitioning the source domain into training and validation subsets. The optimal model was selected based on validation performance. We followed previous DG methods by constructing 20 train-validation splits, with each split repeated 3 times.

Implementation Details. We used ResNet-50 (He et al.,
2016b) pre-trained on ImageNet (He et al., 2016a) as backbone architectures. Detailed implementation details are presented in Appendix D. The detailed results and corresponding confidence intervals of all experiments are provided in Appendix E.

## 4.1. Comparison Of Optimizers On Dg

Experiment Setup. We examined the impact of various optimization methods on generalization performance under domain shifts using Baseline ERM (Vapnik, 1999). The evaluated methods included: Standard optimizers (SGD (Robbins & Monro, 1951)), Adaptive optimizers (Adam (Kingma, 2014), AdamW (Loshchilov & Hutter, 2019), AdaBelief (Zhuang et al., 2020), AdaHessian (Yao et al., 2021), YOGI (Zaheer et al., 2018)), Sharpness-aware optimizers (SAM (Foret et al., 2021), GAM (Zhang et al., 2023b), FAD (Zhang et al., 2023a)) and our proposed GENIE. Results. As shown in Table 2, our optimizer achieved superior performance across most datasets, surpassing existing methods. GENIE outperformed Adam, the default optimizer in most DG algorithms(Zhang et al., 2023a), by 5.69%. Additionally, it achieved improvements of 6.36% over SGD and 4.37% over SAM. In particular, it achieved remarkable performance on VLCS, which is prone to early convergence and overfitting(Matsuura & Harada, 2020), and on TerraIncognita, a wildlife image dataset with significant challenges such as lighting variations, motion blur, occlusions, and severe class imbalance(Beery et al., 2018). These results suggest that GENIE effectively prevents overfitting and enhances the learning of causal relationships by balancing parameter contributions during training. Optimizers designed for generalization, such as SAM, GAM and FAD, outperform standard optimizers, underscoring the significant role of optimization in generalization. These results emphasize the need for developing optimizers specifically tailored for DG.

Table 2. Comparison of optimizers on DG datasets. Results denoted by * are reproduced from (Zhang et al., 2023a) using the same protocol as our paper. The best results for each dataset are highlighted in bold.

OPT. PACS VLCS OFFICE TERRA DOMAIN AVG.

HOME INC NET

ADAM* 84.2 77.3 67.6 44.4 43.0 63.3 ADAMW* 83.6 77.4 68.8 45.2 43.4 63.7 SGD* 79.9 78.1 68.5 44.9 43.2 62.9 YOGI* 81.2 77.6 68.3 45.4 43.5 63.2 ADABELIEF* 84.6 78.4 68.0 45.2 43.5 63.9

ADAHESSIAN* 84.5 78.6 68.4 44.4 **44.4** 64.1

SAM* 85.3 78.2 68.0 45.7 43.4 64.1 GAM* 86.1 78.5 68.2 45.2 43.8 64.4 FAD* **88.2** 78.9 69.2 45.7 **44.4** 65.3

GENIE 87.8 **80.7 69.7 52.0** 44.1 **66.9**

Experiment Setup. The computational overhead of an optimizer is a critical factor in its practical applicability. To evaluate this, we trained models on the PACS and VLCS
datasets for 5,000, 10,000, and 15,000 iterations, measuring average performance and training time per iteration.

## 4.3. Single Domain Generalization

Results. As reported in Table 3, GENIE consistently outperformed other optimizers, even at 5,000 iterations, while incurring lower computational overhead than SGD and Adam. Additionally, GENIE achieved an average of 1.3× faster training compared to SAM, as SAM's update rule requires two sequential (non-parallelizable) gradient computations per step, which doubles the training time. These results experimentally validate the theoretical convergence analysis in Section 3.3.3, confirming GENIE's ability in computational efficiency and convergence speed.

Table 3. Training time (sec) and average accuracy at different iteration levels.

OPT. ITER. T**RAINING** AVG.

TIME PACS VLCS OFFICE

(/S) HOME

SGD 5000 5,273 69.8 76.7 51.3

10000 10,546 73.9 77 62.5 15000 15,783 75.8 77.7 63.9

ADAM 5000 5,443 84.2 77 63.6

10000 10,934 86.1 77 65.2 15000 16,531 84.5 77 65.2

SAM 5000 5,775 82.4 79.4 69.4

10000 11,500 83.5 80.3 69.6 15000 17,191 84.1 80.4 70

GENIE 5000 4,292 **88.4 81.3 70** (OURS) 10000 8,582 87.1 **81.3** 69.2

15000 12,876 86.9 **81.3** 69.1

## 4.2. Integration With Current Dg Algorithms

Experiment Setup. GENIE is a versatile optimizer that integrates seamlessly with various DG algorithms without requiring changes to the training procedure or model architecture. To validate its compatibility, we combined GENIE with several well-performing DG algorithms—CORAL(Sun & Saenko, 2016) and RSC(Huang et al., 2020) using ResNet50 as the backbone—and compared its performance against other optimization techniques. Results. The performance evaluation results for DG are summarized in Table 5. GENIE consistently outperforms existing optimization methods, demonstrating its robustness and broad applicability. These results validate GENIE's scalability and compatibility with various DG algorithms. Unlike other DG methods, which often require multiple source domains or architecture modifications, GENIE seamlessly integrates with existing training pipelines, providing consistent performance gains without additional complexity. This establishes GENIE as an algorithm-agnostic and highly adaptable optimization framework for DG tasks.

Experiment Setup. We evaluated performance in Single Domain Generalization (SDG), which is more constrained but better reflects real-world applications. The flexibility to operate in SDG without structural modifications is an advantage of our method over certain existing methods that are limited to multi-source settings. In SDG, the model is trained and validated on a single domain and tested on the others, with results averaged across all source domains. We compared GENIE with Adam, SGD, and SAM, and applied it to existing DG methods.

Results. The SDG performance results are presented in Table 4. As in previous DG settings, our optimizer outperformed existing optimizers. When applied to DG methods, conventional optimizers reduced performance, whereas GENIE achieved the highest performance as a standalone model and also improved DG methods when used as an optimizer. These results show that our method enhances DG performance without requiring architectural modifications or multiple source domains, and performs well even as a standalone method.

A**LGORITHM** PACS VLCS OFFICE TERRA AVG.

HOME INC

ADAM 64.3 56.2 50.7 33.5 51.2 SGD 49.5 60.4 45.9 22.8 44.7 SAM 57.7 66.7 **59.2** 26.8 52.6 GENIE (OURS**) 69.5 69.9** 58.6 36.0 **58.5** RSC+ADAM 56.8 51.6 2.1 31.6 35.5 RSC+SGD 22.2 39.8 1.7 17.6 20.3 RSC+GENIE(OURS) 68.2 68.7 54.4 33.2 **56.1**

CORAL+ADAM 64.3 56.2 50.7 33.5 51.2

CORAL+SGD 49.5 60.4 45.9 22.8 44.7 CORAL+GENIE(OURS) 70.9 69.2 56.4 36.7 **58.3**

## 4.4. Model Analysis

Ablation. We conducted an ablation study using the PACS dataset in a DG setting to evaluate the effects of Preconditioning, Noise Injection, and Random Mask (Table 6). The version without all three components corresponds to ERM trained with Adam, while the version incorporating all three represents our proposed GENIE optimizer. Experimental results show that GENIE achieved the highest performance, improving accuracy by 4.9% compared to ERM. Even when using only Preconditioning, performance improved by 3.8%, indicating that a simple preconditioning technique can enhance generalization. Additionally, in the Cartoon and Sketch domains, where objects are placed on a white background, models trained with Noise Injection and Random Mask performed better. Here, we conclude that preconditioning alone is enough for DG, but you can optionally utilize Noise Injection and Random Mask for

A**LGORITHM** PACS VLCS OFFICEHOME TERRAINC AVG. ERM†(VAPNIK, 1999) 85.5 77.5 66.5 46.1 68.9 IRM†(ARJOVSKY ET AL., 2019) 83.5 78.6 64.3 47.6 68.5 GROUPDRO†(SAGAWA ET AL., 2020) 84.4 76.7 66.0 43.2 67.6 I-MIXUP†(XU ET AL., 2020) 84.6 77.4 68.1 47.9 69.5 MLDG†(LI ET AL., 2018A) 84.9 77.2 66.8 47.8 69.2 MMD†(LI ET AL., 2018B) 84.7 77.5 66.4 42.2 67.7 DANN†(GANIN ET AL., 2016) 83.7 78.6 65.9 46.7 68.7 CDANN†(LI ET AL., 2018C) 82.6 77.5 65.7 45.8 67.9 MTL†(BLANCHARD ET AL., 2021) 84.6 77.2 66.4 45.6 68.5 SAGNET†(NAM ET AL., 2021) 86.3 77.8 68.1 48.6 70.2 ARM†(ZHANG ET AL., 2021) 85.1 77.6 64.8 45.5 68.3 VREX†(KRUEGER ET AL., 2021) 84.9 78.3 66.4 46.4 69 MIXSTYLE*(ZHOU ET AL., 2021) 85.2 77.9 60.4 44 66.9 MIRO*(CHA ET AL., 2022) 85.4 78.9 69.5 45.4 69.8 GENIE (OURS) 87.8 80.7 69.7 52.0 **72.6** RSC(HUANG ET AL., 2020)+ADAM* 84.5 77.9 65.7 44.5 68.2 RSC+ADAMW* 83.4 77.5 66.3 45.1 68.1 RSC+SGD* 82.6 78.1 67 43.9 67.9 RSC+GENIE(OURS) 87.3 80.6 68.1 49.5 **71.4** CORAL(SUN & SAENKO, 2016) + ADAM* 86 78.9 68.7 43.7 69.3 CORAL+ADAMW* 86.4 79.5 69.8 45.0 70.2 CORAL+SGD* 85.6 78.2 69.5 45.8 69.8 CORAL+GENIE(OURS) 87.9 80.7 70.6 48.4 **71.9**

additional robustness.

Table 6. Ablation study on the PACS dataset. Results are reported for evaluations on four domains: Art, Cartoon, Photo, and Sketch.

Sensitivity Analysis. GENIE employs two key hyperparameters: the dropout probability P and the coefficient B , which is used to compute the moving average and variance of gradients. To analyze the sensitivity of these hyperparameters, we conducted a grid search while keeping all other training settings fixed. As shown in Figure 2, GENIE consistently outperformed SGD, Adam, and SAM across a wide range of Pand B values, demonstrating strong robustness to hyperparameter variation. Notably, while this experiment involved hyperparameter tuning via grid search, all other experiments followed the DomainBed protocol, using validation performance for hyperparameter selection.

| PRE       | NOISE   | MASK   | PACS   | AVG.   |      |      |      |
|-----------|---------|--------|--------|--------|------|------|------|
| CONDITION | A       | C      | P      | S      |      |      |      |
| X         | X       | X      | 88.0   | 79.7   | 96.7 | 72.7 | 84.2 |
| O         | X       | X      | 89.5   | 82.3   | 98.4 | 79.4 | 87.4 |
| O         | O       | X      | 85.4   | 77.4   | 98.6 | 78.7 | 85.0 |
| O         | X       | O      | 84.6   | 79.9   | 98.3 | 77.4 | 85.1 |
| O         | O       | O      | 89.3   | 84.1   | 98.7 | 81.6 | 88.4 |

OSGR of Network Parameters Over Time. To assess whether our approach enhances the overall OSGR of network parameters during training, we tracked the average OSGR of all parameters throughout the training process. As shown in Figure 4, the OSGR measurements on the VLCS dataset show that GENIE achieves an OSGR closer to 1 than prior optimizers. This means superior generalization performance. These findings align with the theoretical Generalization analysis in Section 3.3.1 , confirming that GENIE ensures more stable and balanced parameter updates during training, which ultimately leads to improved generalization.

Interestingly, while SAM is designed for better generalization performance, it exhibits inferior OSGR values. This suggests that the sharpness-aware regime alone is insufficient for generalization, and that the OSGR regime should also be considered when addressing generalization in DG tasks. This observation is consistent with our PAC-Bayesian analysis in Section 3.3.2, which reveals that inducing balanced OSGR values leads to tighter generalization bounds, reinforcing the role of OSGR as a necessary complement to sharpness-aware optimization. Loss Landscape. We analyzed the convergence paths of SGD, Adam, and GENIE in the loss landscape using the FashionMNIST dataset(Xiao et al., 2017). As shown in Figure 5, each corner represents the local minima of a specific source domain. All optimizers started at (-1,3) and were updated for 30 steps under the same conditions. SGD and Adam follow steep direction and converge quickly. However, fast convergence often causes overfitting to specific source domains in OOD scenarios. Generalizable features are learned later in training(Perez et al. ´ , 2019; Shah et al., 2020b; Nakkiran et al., 2019), so rapid convergence can prevent the model from acquiring them sufficiently. In contrast, as demonstrated in the theoretical analysis in Section 3.3.2, GENIE leads optimization toward flatter minima by effectively reducing sharpness, thereby improving generalization(Foret et al., 2021). Feature Visualization. To examine how the GENIE optimizer operates at the feature level, we performed UMAP visualizations(McInnes et al., 2018) on the PACS dataset, with the Sketch domain held out as the unseen target. Each color represents a different class. The results Figure 3 show that GENIE leads to clear class separation across domains, suggesting effective domain-invariant feature learning.

## 5. Conclusion

We introduce GENIE, an optimizer that leverages OSGR to guide gradients in effective directions, preventing overly predictive parameters from dominating while ensuring all parameters contribute equitably to learning. GENIE achieves a higher OSGR with improved generalization and ensures fast convergence rate comparable to SGD. Empirically, it outperforms state-of-the-art optimizers across five DG benchmarks, demonstrating robust performance under significant domain shifts and limited data. Seamlessly integrating with existing DG and SDG methods, GENIE consistently achieves performance improvements. This work highlights the potential of OSGR as a guiding principle, paving the way for its use in few-shot learning, meta-learning, and other tasks requiring solutions to source-domain overfitting.

## Acknowledgements

This work was supported by Korea Internet & Security Agency(KISA) grant funded by the Korea government(PIPC) (No.RS-2023-00231200, Development of personal video information privacy protection technology capable of AI learning in an autonomous driving environment)

## Impact Statement

Our work proposes GENIE, an optimization method that enhances domain generalization by ensuring stable and balanced updates. GENIE mitigates overfitting, promotes flatter minima, and improves OOD performance, contributing to more robust and generalizable models.

## References

Amari, S.-i., Ba, J., Grosse, R. B., Li, X., Nitanda, A.,
Suzuki, T., Wu, D., and Xu, J. When does preconditioning help or hurt generalization? In International Conference on Learning Representations, 2021.

Arjovsky, M., Bottou, L., Gulrajani, I., and Lopez-
Paz, D. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.

Balles, L. and Hennig, P. Dissecting adam: The sign, magnitude and variance of stochastic gradients. In International Conference on Machine Learning, pp. 404–413. PMLR, 2018.

Beery, S., Van Horn, G., and Perona, P. Recognition in terra incognita. In Proceedings of the European conference on computer vision (ECCV), pp. 456–473, 2018.

Blanchard, G., Deshmukh, A. A., Dogan, U., Lee, G., and Scott, C. Domain generalization by marginal transfer learning. *Journal of machine learning research*, 22(2): 1–55, 2021.

Cha, J., Lee, K., Park, S., and Chun, S. Domain generalization by mutual-information regularization with pretrained models. In Avidan, S., Brostow, G., Cisse, M., ´ Farinella, G. M., and Hassner, T. (eds.), Computer Vision
- ECCV 2022, pp. 440–457, Cham, 2022. Springer Nature Switzerland.

Dennis, Jr, J. E. and More, J. J. Quasi-newton methods, ´
motivation and theory. *SIAM review*, 19(1):46–89, 1977.

Duchi, J., Hazan, E., and Singer, Y. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12(7), 2011.

Fang, C., Xu, Y., and Rockmore, D. N. Unbiased metric learning: On the utilization of multiple datasets and web images for softening bias. In Proceedings of the IEEE International Conference on Computer Vision, pp. 1657– 1664, 2013.

Foret, P., Kleiner, A., Mobahi, H., and Neyshabur, B.

Sharpness-aware minimization for efficiently improving generalization. In *International Conference on Learning* Representations, 2021.

Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., March, M., and Lempitsky, V. Domainadversarial training of neural networks. Journal of machine learning research, 17(59):1–35, 2016.

Gulrajani, I. and Lopez-Paz, D. In search of lost domain generalization. In International Conference on Learning Representations, 2021.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In *Proceedings of the IEEE* Conference on Computer Vision and Pattern Recognition (CVPR), June 2016a.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016b.

Hemati, S., Zhang, G., Estiri, A., and Chen, X. Understanding hessian alignment for domain generalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19004–19014, 2023.

Hinton, G., Srivastava, N., and Swersky, K. Neural networks for machine learning lecture 6a overview of mini-batch gradient descent. *Cited on*, 14(8):2, 2012.

Huang, Z., Wang, H., Xing, E. P., and Huang, D. Selfchallenging improves cross-domain generalization. In Vedaldi, A., Bischof, H., Brox, T., and Frahm, J.-M.

(eds.), *Computer Vision - ECCV 2020*, pp. 124–140, Cham, 2020. Springer International Publishing.

Ioffe, S. and Szegedy, C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *International conference on machine learning*, pp. 448– 456. pmlr, 2015.

Kingma, D. P. Adam: A method for stochastic optimization.

arXiv preprint arXiv:1412.6980, 2014.

Koyama, M. and Yamaguchi, S. When is invariance useful in an out-of-distribution generalization problem? *arXiv* preprint arXiv:2008.01883, 2020.

Krueger, D., Caballero, E., Jacobsen, J.-H., Zhang, A.,
Binas, J., Zhang, D., Priol, R. L., and Courville, A.

Out-of-distribution generalization via risk extrapolation (rex). In Meila, M. and Zhang, T. (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 5815–5826. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/ v139/krueger21a.html.

Li, D., Yang, Y., Song, Y.-Z., and Hospedales, T. M. Deeper, broader and artier domain generalization. In *Proceedings* of the IEEE international conference on computer vision, pp. 5542–5550, 2017.

Li, D., Yang, Y., Song, Y.-Z., and Hospedales, T. Learning to generalize: Meta-learning for domain generalization.

In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018a.

Li, H., Pan, S. J., Wang, S., and Kot, A. C. Domain generalization with adversarial feature learning. In *Proceedings* of the IEEE conference on computer vision and pattern recognition, pp. 5400–5409, 2018b.

Li, Y., Gong, M., Tian, X., Liu, T., and Tao, D. Domain generalization via conditional invariant representations. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018c.

Liu, J., Bai, Y., Jiang, G., Chen, T., and Wang, H.

Understanding why neural networks generalize well through gsnr of parameters. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum? id=HyevIJStwH.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In *International Conference on Learning* Representations, 2019.

Mahajan, D., Tople, S., and Sharma, A. Domain generalization using causal matching. In *International conference* on machine learning, pp. 7313–7324. PMLR, 2021.

Mansilla, L., Echeveste, R., Milone, D. H., and Ferrante, E. Domain generalization via gradient surgery. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 6630–6638, 2021.

Matsuura, T. and Harada, T. Domain generalization using a mixture of multiple latent domains. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 11749–11756, 2020.

McInnes, L., Healy, J., and Melville, J. Umap: Uniform manifold approximation and projection for dimension reduction. *arXiv preprint arXiv:1802.03426*, 2018.

Michalkiewicz, M., Faraki, M., Yu, X., Chandraker, M.,
and Baktashmotlagh, M. Domain generalization guided by gradient signal to noise ratio of parameters. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6177–6188, 2023.

Montavon, G., Orr, G., and Muller, K.-R. ¨ Neural networks:
tricks of the trade, volume 7700. springer, 2012.

Muandet, K., Balduzzi, D., and Scholkopf, B. Domain ¨
generalization via invariant feature representation. In International conference on machine learning, pp. 10–18.

PMLR, 2013.

Nakkiran, P., Kaplun, G., Kalimeris, D., Yang, T., Edelman, B. L., Zhang, F., and Barak, B. Sgd on neural networks learns functions of increasing complexity. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pp. 3496–3506, 2019.

Nam, H., Lee, H., Park, J., Yoon, W., and Yoo, D. Reducing domain gap by reducing style bias. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8690–8699, June 2021.

Parascandolo, G., Neitz, A., Orvieto, A., Gresele, L., and Scholkopf, B. Learning explanations that are hard to vary. ¨ In *International Conference on Learning Representations*, 2021.

Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., and Wang, B. Moment matching for multi-source domain adaptation. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 1406–1415, 2019.

Perez, G. V., Louis, A. A., and Camargo, C. Q. Deep ´
learning generalizes because the parameter-function map is biased towards simple functions. In 7th International Conference on Learning Representations, ICLR 2019, 2019.

Rame, A., Dancette, C., and Cord, M. Fishr: Invariant gradient variances for out-of-distribution generalization. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 18347–18377. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/ v162/rame22a.html.

Robbins, H. and Monro, S. A Stochastic Approximation Method. The Annals of Mathematical Statistics, 22(3):400 - 407, 1951. doi: 10.1214/aoms/ 1177729586. URL https://doi.org/10.1214/
aoms/1177729586.

Sagawa, S., Koh, P. W., Hashimoto, T. B., and Liang, P.

Distributionally robust neural networks. In *International* Conference on Learning Representations, 2020.

Shah, H., Tamuly, K., Raghunathan, A., Jain, P., and Netrapalli, P. The pitfalls of simplicity bias in neural networks. Advances in Neural Information Processing Systems, 33: 9573–9585, 2020a.

Shah, H., Tamuly, K., Raghunathan, A., Jain, P., and Netrapalli, P. The pitfalls of simplicity bias in neural networks. Advances in Neural Information Processing Systems, 33: 9573–9585, 2020b.

Shahtalebi, S., Gagnon-Audet, J.-C., Laleh, T., Faramarzi, M., Ahuja, K., and Rish, I. Sand-mask: An enhanced gradient masking strategy for the discovery of invariances in domain generalization, 2021. URL https: //arxiv.org/abs/2106.02266.

Shi, Y., Seely, J., Torr, P., Siddharth, N., Hannun, A.,
Usunier, N., and Synnaeve, G. Gradient matching for domain generalization. In *International Conference on* Learning Representations, 2022.

Sun, B. and Saenko, K. Deep coral: Correlation alignment for deep domain adaptation. In *Computer Vision–ECCV* 2016 Workshops: Amsterdam, The Netherlands, October 8-10 and 15-16, 2016, Proceedings, Part III 14, pp. 443– 450. Springer, 2016.

Vapnik, V. N. An overview of statistical learning theory.

IEEE transactions on neural networks, 10(5):988–999, 1999.

Venkateswara, H., Eusebio, J., Chakraborty, S., and Panchanathan, S. Deep hashing network for unsupervised domain adaptation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp.

5018–5027, 2017.

Wang, P., Zhang, Z., Lei, Z., and Zhang, L. Sharpnessaware gradient matching for domain generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3769–3778, 2023.

Wang, Y., Li, H., and Kot, A. C. Heterogeneous domain generalization via domain mixup. In *ICASSP 2020-2020* IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 3622–3626. IEEE, 2020.

Xiao, H., Rasul, K., and Vollgraf, R. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, 2017.

Xu, M., Zhang, J., Ni, B., Li, T., Wang, C., Tian, Q., and Zhang, W. Adversarial domain adaptation with domain mixup. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 6502–6509, 2020.

Yan, S., Song, H., Li, N., Zou, L., and Ren, L. Improve unsupervised domain adaptation with mixup training. arXiv preprint arXiv:2001.00677, 2020.

Yao, Z., Gholami, A., Shen, S., Mustafa, M., Keutzer, K.,
and Mahoney, M. Adahessian: An adaptive second order optimizer for machine learning. In *proceedings of the* AAAI conference on artificial intelligence, volume 35, pp. 10665–10673, 2021.

Ye, Q. Preconditioning for accelerated gradient descent optimization and regularization. *arXiv preprint* arXiv:2410.00232, 2024.

Ye, W., Zheng, G., Cao, X., Ma, Y., and Zhang, A. Spurious correlations in machine learning: A survey. arXiv preprint arXiv:2402.12715, 2024.

Zaheer, M., Reddi, S., Sachan, D., Kale, S., and Kumar, S.

Adaptive methods for nonconvex optimization. Advances in neural information processing systems, 31, 2018.

Zhang, M., Marklund, H., Dhawan, N., Gupta, A., Levine, S., and Finn, C. Adaptive risk minimization: learning to adapt to domain shift. In Proceedings of the 35th International Conference on Neural Information Processing Systems, pp. 23664–23678, 2021.

Zhang, R., Fan, Z., Yao, J., Zhang, Y., and Wang, Y. Domaininspired sharpness-aware minimization under domain shifts. In The Twelfth International Conference on Learning Representations, 2024.

Zhang, X., Xu, R., Yu, H., Dong, Y., Tian, P., and Cui, P.

Flatness-aware minimization for domain generalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5189–5202, October 2023a.

Zhang, X., Xu, R., Yu, H., Zou, H., and Cui, P. Gradient norm aware minimization seeks first-order flatness and improves generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20247–20257, 2023b.

Zhou, K., Yang, Y., Qiao, Y., and Xiang, T. Domain generalization with mixstyle. In *International Conference on* Learning Representations, 2021.

Zhuang, J., Tang, T., Ding, Y., Tatikonda, S. C., Dvornek, N.,
Papademetris, X., and Duncan, J. Adabelief optimizer: Adapting stepsizes by the belief in observed gradients. Advances in neural information processing systems, 33: 18795–18806, 2020.

## A. Notation

| Table 7. Final Revised Notation Table   |                                                       |                  |        |
|-----------------------------------------|-------------------------------------------------------|------------------|--------|
| Symbol                                  | Description                                           |                  |        |
| f                                       | neural network                                        |                  |        |
| L(θ)                                    | loss function                                         |                  |        |
| Z                                       | data distribution defined over X × Y                  |                  |        |
| n                                       | number of data samples                                |                  |        |
| D, D′                                   | training/test dataset drawn from Z                    |                  |        |
| θ, θj                                   | model parameters, parameter j                         |                  |        |
| θt,j                                    | parameter of index j at optimization step t           |                  |        |
| gD,j (θ)                                | gradient of parameter j averaged over training set D  |                  |        |
| gt                                      | gradient at step t                                    |                  |        |
| g 2 j                                   | squared gradient for parameter j                      |                  |        |
| ρ 2 j                                   | variance of parameter j's gradient                    |                  |        |
| σ j                                     | variance of gradient averaged over training set       |                  |        |
| 2                                       | 2                                                     |                  |        |
| rj                                      | gradient signal-to-noise ratio (GSNR), rj = g j 2 ρ j |                  |        |
| pj                                      | proposed preconditioning factor for parameter j       |                  |        |
| R(Z, n)                                 | one-step generalization ratio (OSGR)                  |                  |        |
| ξt ∼ N (0, σ2 )                         | Gaussian noise for noise injection                    |                  |        |
| J                                       | set of parameter index                                |                  |        |
| G                                       | bound of gradient l2 norm                             |                  |        |
| 1/Su                                    | lower bound of gradient variance                      |                  |        |
| L                                       | Lipschitz constant                                    |                  |        |
| Pl                                      | lower bound of preconditioning value                  |                  |        |
| Wj                                      | weighting factor showing up in optimizers,            | ED∼Zn (g 2 D,j ) | in SGD |
| P j ′ ED∼Zn (g 2                        | )                                                     |                  |        |
| D′,j                                    |                                                       |                  |        |
| p, π ˜                                  | probability measure of posterior and prior            |                  |        |
| Σp˜, Σπ                                 | covariance matrix of Gaussian distribution            |                  |        |
| ϵ, ϵ′                                   | random error terms in gradients                       |                  |        |
| KL(˜p∥π)                                | KL divergence between distributions                   |                  |        |

## B. Details Of Table 1

We start out from a reinterpretation of the widely-used ADAM optimizer, which maintains moving averages of stochastic gradients and their element-wise square,

$$\tilde{m}_{t}=\beta_{1}\tilde{m}_{t-1}+(1-\beta_{1})g_{t},\quad\hat{m}_{t}=\frac{\tilde{m}_{t}}{1-\beta_{1}^{t+1}},$$
, (13)
$${\bar{v}}_{t}=\beta_{2}{\bar{v}}_{t-1}+(1-\beta_{2})g_{t}^{2},\quad{\hat{v}}_{t}={\frac{{\bar{v}}_{t}}{1-\beta_{2}^{t+1}}},$$
, (14)
with β1, β2 ∈ (0, 1) and updates with learning rate α,

$$\theta_{t+1}=\theta_{t}-\alpha\frac{\hat{m}_{t}}{\sqrt{\hat{v}_{t}}+\varepsilon}$$
√vˆt + ε(15)
$$(13)$$
$$(14)^{\frac{1}{2}}$$
$$(15)^{\frac{1}{2}}$$

with a small constant ε > 0 preventing division by zero. Ignoring ε and assuming |mt,i| > 0 for the moment, we can rewrite the update direction as

$${\frac{m_{t}}{\sqrt{v_{t}}}}=\operatorname{sign}(m_{t}){\sqrt{\frac{v_{t}}{m_{t}^{2}}}}={\underbrace{{\frac{1}{\sqrt{1+{\frac{v_{t}-m_{t}^{2}}{m_{t}^{2}}}}}}}}_{T_{1}}\otimes{\underbrace{\operatorname{sign}(m_{t})}_{T_{2}}}.$$
$$(16)$$
$$(17)$$
$$(18)$$

Here, we divide the preconditioning into two terms. The convergnece term T2 which modulates update size is :

$$\mathrm{sign}(m_{t})={\frac{\mathbb{E}_{D\sim\mathcal{Z}^{n}}\left(g_{j}\right)}{|\mathbb{E}_{D\sim\mathcal{Z}^{n}}\left(g_{j}\right)|}}$$
|ED∼Zn (gj )|(17)
The alignment term T1 which includes GSNR is :

$$\frac{1}{\sqrt{1+\frac{v_{t}-m_{t}^{2}}{m_{t}^{2}}}}=\sqrt{\frac{1}{\frac{1}{n\cdot r_{j}}+1}}\tag{1}$$

The Equation (18) can be justified by definition of GSNR using Equation (19) and Equation (20). The variance of gradient average is:

$$v_{t}-m_{t}^{2}=\frac{\rho_{j}^{2}}{n}\tag{1}$$  $$m_{t}^{2}=\tilde{g}_{j}^{\ 2}\tag{2}$$
and
$$m_{t}^{2}={\widetilde{g_{j}}}^{2}$$

## C. Proof Of Theorems

C.1. Convergence Analysis We provide the detailed derivation and proof for Theorem 3.11. From Assumption 3.9, we start with:

$$L(\theta_{t+1})\leq L(\theta_{t})+\underbrace{\langle\nabla L(\theta_{t}),\theta_{t+1}-\theta_{t}\rangle}_{T_{1}}+\underbrace{\frac{L}{2}\|\theta_{t+1}-\theta_{t}\|^{2}}_{T2},$$
$$(19)$$
$$(20)$$
$$(21)^{\frac{1}{2}}$$
$$(222)$$
$$(23)$$
$$(24)$$
$$(25)$$
where the first term, T1, is given by:
T1 = ⟨∇L(θt), θt+1 − θt⟩. (22)
Using our preconditioning, which we defined as:

$$T_{1}=\langle\nabla L(\theta_{t}),\theta_{t+1}-\theta_{t}\rangle.$$
$$p={\frac{1}{\widetilde{g}^{2}+{\frac{\rho^{2}}{n}}}}\cdot{\frac{\widetilde{g}^{2}}{\rho^{2}}}={\frac{n}{n+{\frac{\rho^{2}}{g^{2}}}}}\cdot{\frac{1}{\rho^{2}}},$$

which satysifies, given Assumption 3.10:
$$\frac{n}{n+\frac{\rho^{2}}{g^{2}}}\leq1,\frac{1}{\rho^{2}}\leq S_{u}\tag{1}$$
ge2 For T2, we have:

$$T_{2}=\frac{L}{2}\lambda^{2}\|p\odot g_{t}\|^{2}\leq\frac{L}{2}\lambda^{2}\|S_{u}\cdot g_{t}\|^{2},$$

and its expectation satisfies, using Assumption 3.8:

$$\mathbb{E}[T_{2}]\leq\frac{L}{2}\lambda^{2}S_{u}^{2}G^{2}.$$
2. (26)
$$(26)^{\frac{1}{2}}$$

14

For T1, we decompose:
ompose:  $$T_{1}=\langle\nabla L(\theta_{t}),\theta_{t+1}-\theta_{t}\rangle=-\lambda_{t}\langle\nabla L(\theta_{t}),p\odot g_{t}\rangle,$$  $$T_{1}\leq\underbrace{-\lambda_{t}P_{i}\langle\nabla L(\theta_{t})\cdot g_{t}\rangle}_{T3}+\underbrace{\lambda_{t}\sum_{j}|[\nabla L(\theta_{t})]_{j}|\cdot\underbrace{|g_{t,j}|}_{\rho_{t,j}^{2}}\cdot1(\operatorname{sign}[[\nabla L(\theta_{t})]_{j}]\neq\operatorname{sign}[g_{t,j}])}_{\mathcal{T}_{4}}.$$
$$(27)$$
$$(28)^{\frac{1}{2}}$$
$$(29)$$
$$(30)$$
$$(31)$$
$$(32)$$
$$(34)$$
. (28)
Now, considering T3, we evaluate its expectation:

$$\mathbb{E}[T_{3}]=-\lambda_{t}P_{l}\mathbb{E}[\langle\nabla L(\theta_{t}),g_{t}\rangle],$$
E[T3] = −λtPlE[⟨∇L(θt), gt⟩], (29)
Plis lower bound of our preconditioning value. Considering T4:

$$\mathbb{E}[T_{4}]=\lambda_{t}\sum_{j}\mathbb{E}\left[|[\nabla L(\theta_{t})]_{j}|\cdot\frac{|g_{t,j}|}{\rho_{t,j}^{2}}\cdot1(\operatorname{sign}[[\nabla L(\theta_{t})]]_{j}\neq\operatorname{sign}[g_{t,j}])\right].$$

Thus, we obtain:

$$\mathbb{E}[T_{4}]=\lambda_{t}\sum_{j}\mathbb{E}\left[|(\nabla L(\theta_{t}))_{j}|\cdot\frac{|g_{t,j}|}{\rho_{t,j}^{2}}\mid P(\text{sign}[|\nabla L(\theta_{t})|_{j}]\neq\text{sign}[g_{t,j})]\right].$$  probability term:  $$\mathbb{E}[\lambda_{t}\mid\nabla L(\theta_{t})\mid\lambda_{t}\mid\lambda_{t}\mid\lambda_{t}].$$
$$P(\mathrm{sign}[\nabla L(\theta_{t})]_{j}]\neq\mathrm{sign}[g_{t,j}])$$

Next, we analyze the probability term:
P(sign[∇L(θt)]j ] ̸= sign[gt,j ]) (32)
and bound it as follows:
$P(\mbox{sign}[\nabla L(\theta_{t})]_{j}\neq\mbox{sign}[g_{t,j}])\leq P(|[\nabla L(\theta_{t})]_{j}-g_{t,j}|\geq|[\nabla L(\theta_{t})]_{j}|)$.  
Using Chebyshev's inequality:

$$P\big{(}\big{|}[\nabla L(\theta_{t})]_{j}-g_{t,j}\big{|}\ \geq\ \big{|}[\nabla L(\theta_{t})]_{j}\big{|}\big{)}\ \ \leq\ \ \frac{\mbox{Var}\big{(}[\nabla L(\theta_{t})]_{j}-g_{t,j}\big{)}}{\big{|}[\nabla L(\theta_{t})]_{j}\big{|}^{2}}\ \ =\ \ \frac{\sigma^{2}}{\big{|}[\nabla L(\theta_{t})]_{j}\big{|}^{2}}\ =\ \ \frac{\rho_{t}^{2}/n}{\big{|}[\nabla L(\theta_{t})]_{j}\big{|}^{2}},$$  the value for $\rho_{t}$ is to make 
. (34)
n is the number of gradient samples. Replacing the n to step T, we bound the expectation:

$$\mathbb{E}[T_{4}]\leq\lambda_{t}\sum_{j}\mathbb{E}\left[|(\nabla L(\theta_{t})|_{j}]\cdot\frac{|g_{t,j}|}{\rho_{t,j}^{2}}\cdot\frac{\rho_{t,j}^{2}/T}{|(\nabla L(\theta_{t})|_{j})^{2}}\right]\leq\lambda_{t}\frac{|J|}{T}\tag{1}$$

|J| is the number of parameters indicated by the size of parameter index set. Now, summing the inequalities until step T:

$$\mathbb{E}[L(\theta_{t+1})]\leq\mathbb{E}[L(\theta_{t})]-\lambda_{t}P_{t}\|\nabla L(\theta_{t})\|^{2}+\lambda_{t}\cdot\frac{|J|}{T}+\frac{L}{2}\lambda_{t}^{2}S_{u}^{2}G^{2}.$$

Rearranging:

$$\mathbb{E}[L(\theta_{t+1})]\leq L(\theta_{0})-\lambda_{t}P_{t}\sum_{t=1}^{T}\|\nabla L(\theta_{t})\|^{2}+T\cdot\lambda_{t}\left({\frac{|J|}{T}}+{\frac{L}{2}}\lambda_{t}S_{u}^{2}G^{2}\right).$$
$$(35)$$
$$(37)$$
This results in:
$$\frac{1}{T}\sum_{t=1}^{T}\left\|\nabla L(\theta_{t})\right\|^{2}\leq\frac{L(\theta_{0})-\mathbb{E}[L(\theta_{t+1})]}{\lambda_{t}P_{l}\cdot T}+\frac{1}{P_{l}}\left(\frac{|J|}{T}+\frac{L}{2}\lambda_{t}S_{u}^{2}G^{2}\right).$$  $$\frac{1}{T}\sum_{t=1}^{T}\left\|\nabla L(\theta_{t})\right\|^{2}\leq\frac{L(\theta_{0})-\mathbb{E}[L(\theta_{*})]}{\lambda_{t}P_{l}\cdot T}+\frac{1}{P_{l}}\left(\frac{|J|}{T}+\frac{L}{2}\lambda_{t}S_{u}^{2}G^{2}\right).$$
(38)  $\begin{array}{l}\text{(39)}\end{array}$ . 
15 Taking T → ∞, the convergence rate is:

$$\mathbb{E}[\|\nabla L(\theta_{t})\|^{2}]\leq\frac{L(\theta_{0})-L(\theta_{*})}{\lambda_{t}P_{l}\cdot T}+\frac{\frac{|J|}{T}+\frac{L}{2}\lambda_{t}S_{u}^{2}G^{2}}{P_{l}}.\tag{1}$$

From the final steps of our derivation, we analyze the convergence rate of the algorithm. Taking λT as:

$$\lambda_{T}=\sqrt{\frac{L(\theta_{0})-L(\theta_{*})}{T\cdot\ell}},$$  $$\mathbb{E}[\|\nabla L(\theta)\|]\leq\frac{1}{P_{1}}\cdot\sqrt{\ell\frac{L(\theta_{0})-L(\theta_{*})}{T}}+\frac{G\cdot S_{2}^{2}}{2\cdot P_{1}}\sqrt{\ell\frac{L(\theta_{0})-L(\theta_{*})}{T}}+\frac{|J|}{P_{1}\cdot T}.$$
we have:
. (42)
 Denoting $\frac{1}{\sqrt{T}}$ as: . 
$$(40)^{3}$$
$$(41)$$
$$(42)$$
$${\frac{1}{\sqrt{\hat{T}}}}={\sqrt{\frac{L(\theta_{0})-L(\theta_{*})\cdot\ell}{T}}},$$
$$(43)$$
$$(44)$$
$$(45)$$
T, (43)
Finally, we rewrite the bound, concluding that:

$$\mathbb{E}[\|\nabla L(\theta)\|^{2}]\leq O\left(\frac{1}{P_{l}}\left(1+\frac{G\cdot S_{u}^{2}}{2}\right)\frac{1}{\sqrt{\tilde{T}}}\right)\quad\blacksquare$$

## C.2. Proof Of Corollary 3.2

We utilized (Liu et al., 2020) for proving Corollary 3.2. In one gradient descent step, the model parameter is updated by ∆θ = θt+1 − θt = −λp ⊙ gD(θ), where λ is the learning rate and p is preconditioning. If λ is small enough, the one-step training and test loss decrease can be approximated by:
Usually, there are some differences between the directions of gD(θ) and gD′ (θ), so statistically ∆L[D] tends to be larger than ∆L[D′], and the generalization gap would increase during training. When λ → 0, in one single training step, the empirical generalization gap increases by ∆L[D] − ∆L[D′]. For simplicity, we denote this quantity as:

$$\nabla:=\Delta L[D]-\Delta L[D^{\prime}]\approx\lambda g_{D}(\theta)\cdot g_{D}(\theta)-\lambda g_{D}(\theta)\cdot g_{D^{\prime}}(\theta),$$

which can be further simplified as:

$$\nabla=\lambda(p\odot\tilde{g}(\theta)+p\odot\epsilon)(\tilde{g}(\theta)+\epsilon^{\prime})-\lambda(p\odot\tilde{g}(\theta)+p\odot\epsilon)(\tilde{g}(\theta)+\epsilon^{\prime}),$$  $$\nabla=\lambda(\tilde{g}(\theta)+\epsilon)(\epsilon-\epsilon^{\prime}).$$
$$(46)$$
$$(47)$$
$$(48)$$

Here, we replaced the random variables by gD(θ) = ˜g(θ) + ϵ and gD′ (θ) = ˜g(θ) + ϵ
′, where ϵ and ϵ
′are random variables with zero mean and variance σ 2(θ). Since E[ϵ
′] = E[ϵ] = 0, ϵ and ϵ
′are independent. The expectation of ∇ is:

$$\mathbb{E}_{D,D^{\prime}\sim\mathbb{Z}^{n}}(\nabla)=\mathbb{E}(\lambda p\odot\epsilon\cdot\epsilon^{\prime})+O(\lambda^{2})=\lambda\sum_{j}p_{j}\cdot\sigma^{2}(\theta_{j})+O(\lambda^{2}),$$

where σ 2(θj ) is the variance of the average gradient of the parameter θj . For simplicity, when it involves a single model parameter θj , we will use only a subscript j instead of the full notation. For example, we use σ 2 j
, rj , and gD,j to denote σ 2(θj ), r(θj ), and gD(θj ), respectively.

$$\Delta L[D]\approx-\Delta\theta\cdot\frac{\partial L[D]}{\partial\theta}+O(\lambda^{2})=\lambda p\odot g_{D}(\theta)\cdot g_{D}(\theta)+O(\lambda^{2}),$$  $$\Delta L[D^{\prime}]\approx-\Delta\theta\cdot\frac{\partial L[D^{\prime}]}{\partial\theta}+O(\lambda^{2})=\lambda p\odot g_{D}(\theta)\cdot g_{D^{\prime}}(\theta)+O(\lambda^{2}).$$
$$(49)$$
$$(50)$$

Expectation Analysis Consider the expectation of ∆L[D] and ∆L[D′] when λ → 0:

$$\mathbb{E}_{D\sim\mathbb{Z}^{n}}(\Delta L[D])\approx\lambda\mathbb{E}_{D\sim\mathbb{Z}^{n}}(p\odot g_{D}(\theta)\cdot g_{D}(\theta))=\lambda\sum_{j}p_{j}\mathbb{E}_{D\sim\mathbb{Z}^{n}}(g_{D,j}^{2}).$$
2D,j ). (51)
$$\mathbb{E}_{D,D^{\prime}\sim\mathcal{Z}^{n}}(\Delta L[D^{\prime}])=\mathbb{E}_{D,D^{\prime}\sim\mathcal{Z}^{n}}(\Delta L[D]-\nabla)\approx\lambda\sum_{j}p_{j}(\mathbb{E}_{D\sim\mathcal{Z}^{n}}(g_{D,j}^{2})-\sigma_{j}^{2}),$$
), (52)
which simplifies further as:
$$\mathbb{E}_{D,D^{\prime}\sim{\mathcal{Z}}^{n}}(\Delta L[D^{\prime}])\approx\lambda\sum_{j}p_{j}(\mathbb{E}_{D\sim{\mathcal{Z}}^{n}}(g_{D,j}^{2})-\rho_{j}^{2}/n).$$
j/n). (53)
Simplification of R(Z, n) Substituting Equation (53) and Equation (51) into R(Z, n), we have:

$$R(\mathcal{Z},n)=1-\frac{\sum_{j}p_{j}\rho_{j}^{2}}{n\sum_{j}p_{j}\mathbb{E}_{D\sim\mathcal{Z}^{n}}(g_{D,j}^{2})}.\tag{1}$$

When rj =
ED∼Zn (gD,j )
2 ρ2 We can rewrite Equation (53) as:

$$(51)$$
$$(52)$$
$$(53)$$
$$(54)$$
$$R(\mathcal{Z},n)=1-\frac{1}{n}\sum_{j}\frac{p_{j}\mathbb{E}_{D\sim\mathcal{Z}^{n}}(g_{D,j}^{2})}{\sum_{j^{\prime}}p_{j}\mathbb{E}_{D\sim\mathcal{Z}^{n}}(g_{D^{\prime},j}^{2})}\cdot\frac{1}{r_{j}+\frac{1}{n}},$$

or equivalently:

$$R(\mathcal{Z},n)=\sum_{j}\frac{p_{j}\mathbb{E}_{D\sim\mathcal{Z}^{n}}(g_{D,j}^{2})}{\sum_{j^{\prime}}p_{j}\mathbb{E}_{D\sim\mathcal{Z}^{n}}(g_{D^{\prime},j}^{2})}\cdot\frac{1}{\left(1+\frac{1}{n-r_{j}}\right)}.\quad\blacksquare$$

C.3. Generalization Analysis C.3.1. PAC BAYES BOUND AND PRECONDITIONING
Theorem C.1 (PAC-Bayes bound). Let D ∼ Zn be a dataset sampled i.i.d. from a data distribution Z.R(θ) is the population risk and L(θ) is empirical risk. Assume that the loss function L(θ) *is bounded in* [0, C] for some constant C > 0. For any λ > 0, with probability at least 1−δ over the draw of D, and for any data-dependent distribution p˜ *over parameters* θ*, the following PAC-Bayes bound holds:*

$$\mathbb{E}_{\theta\sim{\tilde{p}}}[R(\theta)]\leq\underbrace{\mathbb{E}_{\theta\sim{\tilde{p}}}[L(\theta)]}_{T_{1}}+\frac{\lambda C^{2}}{8n}+\underbrace{\frac{\mathrm{KL}({\tilde{p}}\|\pi)+\log{\frac{1}{\delta}}}{\lambda}}_{T_{2}}.$$
$$(55)$$
$$(56)$$
$$(57)$$
$$(58)$$

Here, R(θ) denotes the expected loss over the true data distribution. The SAM algorithm primarily focuses on minimizing the T1 term in Theorem C.1, following the inequality:

$$L_{\mathcal{D}}(\theta)\;\leq\;\mathbb{E}_{\epsilon\sim\mathcal{N}(0,\rho)}\big[L_{\mathcal{D}}(\theta+\epsilon)\big]\;\leq\;\operatorname*{max}_{\|\epsilon\|_{2}\leq\rho}\big[L_{\mathcal{D}}(\theta+\epsilon)\big].$$

In contrast, our method simultaneously minimizes both the T1 and T2 terms. First, we determine a preconditioning vector q to reduce ESEθ∼p˜[L(θ)]. To minimize variance-induced error, we adopt the variance adaptation factor introduced in the SVAG optimizer(Balles & Hennig, 2018), and solve:

$$\mathbb{E}\big[\|q\odot g-\mathbb{E}[g]\|_{2}^{2}\big]=\sum_{j}q_{j}^{2}\mathbb{E}[g_{j}^{2}]-2q_{j}\mathbb{E}[g]^{2}+\mathbb{E}[g]^{2}.$$

17 Minimizing this expression yields the optimal preconditioning:

$$q_{j}={\frac{\mathbb{E}[g]^{2}}{\mathbb{E}[g_{j}^{2}]}}.$$

Next, assume p˜ = N (θt+1, Σp˜) and π = N (θt, Σπ), with both covariances defined as:

$$\Sigma_{\vec{p}}=\mathrm{diag}(q_{1}\cdot\rho_{1}^{2},q_{2}\cdot\rho_{2}^{2},\ldots,q_{|J|}\cdot\rho_{|J|}^{2}),\quad\Sigma_{\pi}=\mathrm{diag}(\rho_{1}^{2},\rho_{2}^{2},\ldots,\rho_{|J|}^{2}),$$

Here, the prior π can be treated as a data-driven prior which is approximated with stochastic gradient descent using all data excluding the current mini-batch. Assuming the variances do not significantly change between steps. Then the KL
divergence can be written as follows:

$$K L({\bar{p}}\|\pi)={\frac{1}{2}}\left[\sum_{i\in J}{\frac{q_{i}\cdot\rho_{i}^{2}}{\rho_{i}^{2}}}+\sum_{i\in J}{\frac{(\theta_{i+1,i}-\theta_{i,i})^{2}}{\rho_{i}^{2}}}-|J|+\sum_{i\in J}\log\left({\frac{\rho_{i}^{2}}{q_{i}\cdot\rho_{i}^{2}}}\right)\right]$$
#(59)
The gradient of this KL term with respect to θt is:

$$[\nabla_{\theta_{t}}K L(\tilde{p}[\pi])]_{j}=-\frac{(\theta_{t+1,j}-\theta_{t,j})}{\rho_{j}^{2}}=\underbrace{\mathbb{E}[g_{j}]^{2}}_{\mathbb{E}[g_{j}^{2}]}\cdot\underbrace{g_{j,t}}_{\rho_{j}^{2}}=\underbrace{\frac{1}{\mathbb{E}[g_{j}^{2}]}}_{\mathrm{GENSHE}}\cdot\underbrace{\mathbb{E}[g_{j,t}]^{2}}_{\mathrm{GENSHE}}\cdot g_{j,t}.$$
$$(59)$$
$$(60)$$

This shows that minimizing the KL divergence in the PAC-Bayes bound via gradient descent naturally leads to the same preconditioning structure. Therefore, our method considers both sharpness (through variance adaptation) and generalization (via KL divergence minimization), whereas SAM focuses solely on sharpness.

It is important to note that our gradient computation is taken with respect to the prior mean θt, rather than the posterior mean θt+1. Due to the asymmetric nature of the forward KL divergence KL(˜p∥π), this choice induces a *mode-covering* behavior rather than mode-seeking. As optimization proceeds, the prior distribution π is iteratively adapted to cover a broader region of the risk landscape, effectively reducing over-concentration around a single mode. This mechanism aligns with the argument in *Risk Extrapolation* (Krueger et al., 2021), where covering a wider set of hypotheses can improve out-of-distribution generalization. Consequently, although the PAC-Bayes bound is originally derived under an i.i.d. assumption, this mode-covering property enables the prior to capture more diverse risk regions, thereby enhancing robustness under distribution shift.

## C.3.2. Osgr Based Analysis

From Section 3.2.1, the OSGR of our method is given by:

$$R_{\mathrm{{ous}}}=1-{\frac{1}{n}}\sum_{j}{\frac{1}{\sum_{j^{\prime}}\left(r_{j^{\prime}}+{\frac{1}{n}}\right)}}=1-{\frac{1}{n{\mathbb{E}}_{j\sim J}\left(r_{j}+{\frac{1}{n}}\right)}}.$$

Similarly, from Theorem 3.1, the OSGR of SGD is given by:

$$(61)$$
$$R_{\mathrm{sgd}}=1-{\frac{1}{n}}\sum_{j}{\frac{\mathbb{E}_{D\sim\mathcal{Z}^{n}}[g_{j}^{2}]}{\sum_{j^{\prime}}\mathbb{E}_{D\sim\mathcal{Z}^{n}}[g_{j^{\prime}}^{2}]}}\cdot{\frac{1}{r_{j}+{\frac{1}{n}}}}.$$
$$(62)$$

Replacing the term Pj ED∼Zn [g 2 j]
Pj′ ED∼Zn [g 2 j′]
to PWj = 1 which represents a weighted average, we rewrite it as:

$$R_{\mathrm{sgd}}=1-{\frac{1}{n}}\sum_{j\in J}W_{j}\left({\frac{1}{r_{j}+{\frac{1}{n}}}}\right).$$
. (63)

$$(63)^{\frac{1}{2}}$$

18 If we assume uniform weight Wj =1 |J|
, by Jensen's inequality:

$$0\leq1-\frac{1}{n}\sum_{j\in J}W_{j}\left(\frac{1}{r_{j}+\frac{1}{n}}\right)\leq1-\frac{1}{n\mathbb{E}_{j\in J}\left(r_{j}+\frac{1}{n}\right)}\leq1.\tag{1}$$
Thus, we conclude:
0 ≤ Rsgd ≤ Rours ≤ 1. - (65)
In the same way, with any preconditioning,

$$(64)$$
$$(65)$$
$$0\leq R_{\mathrm{sgd}}\leq R_{\mathrm{ours}}\leq1.\quad\blacksquare$$
$$R(\mathcal{Z},n)=1-\frac{1}{n}\sum_{j}\frac{p_{j}\mathbb{E}_{D\sim\mathcal{Z}^{n}}(g_{D,j}^{2})}{\sum_{j^{\prime}}p_{j}\mathbb{E}_{D\sim\mathcal{Z}^{n}}(g_{D^{\prime},j}^{2})}\cdot\frac{1}{r_{j}+\frac{1}{n}},$$
$$(66)$$
$$(67)$$

Replacing the term PjpjED∼Zn (g 2 D,j )
Pj
′ pjED∼Zn (g 2 D′,j )
to PWj = 1 with Wj =1 |J|
, which represents a average, we rewrite it as:

$$R_{p e r o n d i t i o n}=1-\frac{1}{n}\sum_{j\in J}W_{j}\left(\frac{1}{r_{j}+\frac{1}{n}}\right).$$

Also by Jensen's inequality, we obtain:
0 ≤ Rprecondition ≤ Rours ≤ 1. - (68)
Consequently, this result establishes that our method achieves the highest OSGR value among preconditioning methods such as Adam, RMSprop, and SVAG(Balles & Hennig, 2018).

However, the assumption of uniform weights Wj =1 |J| can be overly restrictive. In practice, preconditioning methods aim to balance the contribution of parameter updates, which becomes especially important when there exists a strong imbalance in the GSNR distribution. We consider the case where a dominant coordinate exists, denoted as jmax = arg maxj∈J rj , and define the remaining coordinates as J
′ = J \ {jmax}, with j
′max = arg maxj∈J′ rj , such that rj
′max ≪ rjmax.

To demonstrate that our method still yields a higher OSGR under such imbalance, we consider the difference:

$$n(R_{\mathrm{ours}}-R_{\mathrm{procondition}})=\sum_{j\in J}W_{j}\left({\frac{1}{r_{j}+{\frac{1}{n}}}}\right)-{\frac{1}{\mathbb{E}_{j\in J}\left(r_{j}+{\frac{1}{n}}\right)}}.$$

For sufficiently large n, the 1n terms can be neglected:

$$(69)^{\frac{1}{2}}$$
$$n(R_{\mathrm{ours}}-R_{\mathrm{precondition}})\approx\sum_{j\in J}W_{j}\left({\frac{1}{r_{j}}}\right)-{\frac{1}{\mathbb{E}_{j\in J}(r_{j})}}.$$
. (70)
Separating the contribution of the dominant coordinate jmax, we have:

$$n(R_{\mathrm{ours}}-R_{\mathrm{precondine}})=\sum_{j\in J^{\prime}}W_{j}\left({\frac{1}{r_{j}}}\right)+W_{j_{\mathrm{max}}}\left({\frac{1}{r_{j_{\mathrm{max}}}}}\right)-{\frac{1}{\mathbb{E}_{j\in J}(r_{j})}}.$$

Since Ej∈J (rj ) ≥
rjmax |J|
, this difference is lower-bounded by:

$$n(R_{\mathrm{ours}}-R_{\mathrm{precondition}})\geq\sum_{j\in J^{\prime}}W_{j}\left(\frac{1}{r_{j}}\right)+W_{j_{\mathrm{max}}}\left(\frac{1}{r_{j_{\mathrm{min}}}}\right)-\frac{|J|}{r_{j_{\mathrm{max}}}}.$$

Further bounding the terms using rj ≤ rj
′max for all j ∈ J
′, we obtain:

$$\sum_{j\in J^{\prime}}W_{j}\left({\frac{1}{r_{j}}}\right)+W_{j_{\mathrm{nm}}}\left({\frac{1}{r_{j_{\mathrm{nm}}}}}\right)-{\frac{|J|}{r_{j_{\mathrm{nm}}}}}\geq(1-W_{j_{\mathrm{nm}}})\left({\frac{1}{r_{j_{\mathrm{nm}}^{\prime}}}}\right)+W_{j_{\mathrm{nm}}}\left({\frac{1}{r_{j_{\mathrm{nm}}}}}\right)-{\frac{|J|}{r_{j_{\mathrm{nm}}}}}.$$
. (73)
$$(70)$$
$$(71)$$
$$(T2)$$
$$(73)$$

19 Therefore, if the following condition holds:

(1 − Wjmax)rjmax ≥ |J|rj
$$|J|r_{j_{\mathrm{max}}^{\prime}},$$
$\eqref{eq:walpha}$. 
, (74)
then Rours ≥ Rprecondition, and our method guarantees superior OSGR performance compared to other preconditioning strategies. This result highlights the robustness of our formulation, particularly under skewed GSNR distributions.

## D. Implementation Details D.1. Training Details

As introduced in the experimental section, we follow the standard training, hyperparameter search methods, and evaluation protocol proposed by DomainBed (Gulrajani & Lopez-Paz, 2021) to ensure a fair comparison. For each dataset, the models were trained for 15,000 iterations on DomainNet and 5,000 iterations on the other datasets. The search space of hyperparameters is provided in Table 8. All experiments were conducted on an NVIDIA GeForce RTX 4090 under the environment of Python 3.8.10, PyTorch 1.13.1, Torchvision 0.14.1, and CUDA 11.7.

| PARAMETER      | DEFAULT VALUE   | SEARCH DISTRIBUTION   |
|----------------|-----------------|-----------------------|
| BATCH SIZE     | 32              | 2 UNIFORM(3,5.5)      |
| LEARNING RATE  | 0.015           | 10UNIFORM(−3,−1)      |
| RESNET DROPOUT | 0.0             | [0.0, 0.1, 0.5]       |
| WEIGHT DECAY   | 0.0             | 10UNIFORM(−6,−2)      |