# Maxsup: Overcoming Representation Collapse In Label Smoothing

Yuxuan Zhouα,β∗ Heng Liγ,∗† **Zhi-Qi Cheng**γ,ϵ‡ Xudong Yanγ,† **Yifei Dong**γ Mario Fritzβ **Margret Keuper**α,δ α University of Mannheim γ University of Washington ϵ Meta AI
β CISPA Helmholtz Center for Information Security δ Max Planck Institute for Informatics

## Abstract

Label Smoothing (LS) is widely adopted to reduce overconfidence in neural network predictions and improve generalization. Despite these benefits, recent studies reveal two critical issues with LS. First, LS induces overconfidence in misclassified samples. Second, it compacts feature representations into overly tight clusters, diluting intra-class diversity, although the precise cause of this phenomenon remained elusive. In this paper, we analytically decompose the LS-induced loss, exposing two key terms: *(i) a regularization term* that dampens overconfidence only when the prediction is correct, and *(ii) an error-amplification term* that arises under misclassifications. This latter term compels the network to reinforce incorrect predictions with undue certainty, exacerbating representation collapse. To address these shortcomings, we propose Max Suppression (MaxSup), which applies uniform regularization to both correct and incorrect predictions by penalizing the top-1 logit rather than the ground-truth logit. Through extensive feature-space analyses, we show that MaxSup restores intra-class variation and sharpens inter-class boundaries. Experiments on large-scale image classification and multiple downstream tasks confirm that MaxSup is a more robust alternative to LS. 

4

## 1 Introduction

Multi-class classification [19, 26] typically relies on one-hot labels, which implicitly treat different classes as mutually orthogonal. In practice, however, classes often share low-level features [31, 44] or exhibit high-level semantic similarities [3, 24, 42], rendering the one-hot assumption overly restrictive. Such a mismatch can yield over-confident classifiers and ultimately degrade generalization [9]. To moderate overconfidence, Szegedy et al. [34] introduced Label Smoothing (LS), which combines a uniform distribution with the hard ground-truth label, thereby reducing the model's certainty in the primary class. LS has since become prevalent in image recognition [10, 22, 36, 47] and neural machine translation [1, 6], often boosting accuracy and calibration [23]. Yet subsequent work indicates that LS can overly compress features into tight clusters [15, 28, 41], hindering intra-class variability and transferability [5]. In parallel, Zhu et al. [48] found that LS paradoxically fosters overconfidence in misclassified samples, though the precise mechanism behind this remains uncertain. In this paper, we reveal that LS's training objective inherently contains an error amplification term. This term pushes the network to reinforce incorrect predictions with exaggerated certainty, yielding
∗Equal contribution. †Internship at University of Washington. ‡Corresponding author. Assistant Professor, UW Tacoma School of Engineering and Technology.

4https://github.com/ZhouYuxuanYX/Maximum-Suppression-Regularization.

highly confident misclassifications and further compressing feature clusters (Section 3.1, Table 1). Building on Zhu et al. [48], we characterize "overconfidence" in terms of the model's top-1 prediction, rather than through conventional calibration metrics. Through our analysis, we further show that punishing the ground-truth logit during misclassification reduces intra-class variation (Table 2), a phenomenon corroborated by Grad-CAM visualizations (Figure 2). To overcome these shortcomings, we introduce Max Suppression (MaxSup), a method that retains the beneficial regularization effect of LS while eliminating its error amplification. Rather than penalizing the ground-truth logit, MaxSup focuses on the model's top-1 logit, ensuring a consistent regularization signal regardless of whether the current prediction is correct or misclassified. By preserving the ground-truth logit in misclassifications, MaxSup sustains richer intra-class variability and sharpens inter-class boundaries. As visualized in Figure 1, this approach mitigates the feature collapse and attention drift often induced by LS, ultimately leading to more robust representations. Through comprehensive experiments in both image classification (Section 4.2) and semantic segmentation (Section 4.3), we show that MaxSup not only alleviates severe intra-class collapse but also consistently boosts top-1 accuracy and robustly enhances downstream transfer performance (Section 4.1). Our contributions are summarized as follows:
- We perform a logit-level analysis of Label Smoothing, revealing how the error amplification term inflates misclassification confidence and compresses features.

- We propose Max Suppression (MaxSup), removing detrimental error amplification while preserving LS's beneficial regularization. As shown in extensive ablations, MaxSup alleviates intra-class collapse and yields consistent accuracy gains.

- We demonstrate superior performance across tasks and architectures, including ResNet, MobileNetV2, and DeiT-S, where MaxSup significantly boosts accuracy on ImageNet and consistently delivers stronger representations for downstream tasks such as semantic segmentation and robust transfer learning.

## 2 Related Work

We first outline mainstream regularization techniques in deep learning, then survey recent advances in Label Smoothing (LS), and finally clarify how our MaxSup diverges from prior variants.

## 2.1 Regularization

Regularization techniques aim to improve the generalization of deep neural networks by constraining model complexity. Classical methods like ℓ2 [18] and ℓ1 [49] impose direct penalties on large or sparse weights, while Dropout [32] randomly deactivates neurons to discourage over-adaptation. In the realm of loss-based strategies, Label Smoothing (LS) [34] redistributes a fraction of the label probability mass away from the ground-truth class, thereby improving accuracy and calibration [23]. Variants such as Online Label Smoothing (OLS) [45] and Zipf Label Smoothing (Zipf-LS) [21] refine LS by dynamically adjusting the smoothed labels based on a model's evolving predictions. However, they do not fully address the fundamental issue that emerges when the ground-truth logit is not the highest one (see Section 3.1, Table 1). Other loss-based regularizers focus on alternative aspects of the predictive distribution. Confidence Penalty [25] penalizes the model's confidence directly, while Logit Penalty [4] minimizes the global ℓ2-norm of logits, a technique reported to enhance class separation [15]. Despite these benefits, Logit Penalty can inadvertently shrink intra-class variation, thereby hampering transfer learning (see Section 4.1). Unlike the aforementioned methods, MaxSup enforces regularization by penalizing only the top-1 logit zmax rather than the ground-truth logit zgt. In LS-based approaches, suppressing zgt for misclassified samples can worsen errors, whereas MaxSup applies a uniform penalty regardless of whether the model's prediction is correct. Consequently, MaxSup avoids the error amplification effect, retains richer intra-class diversity (see Table 2), and achieves robust transfer performance across diverse datasets and model families (see Table 3).

## 2.2 Studies On Label Smoothing

Label Smoothing has also been studied extensively under knowledge distillation. For instance, Yuan et al. [43] observed that LS can approximate the effect of a teacher–student framework, while Shen et al. [30] investigated its role in such pipelines more systematically. Additionally, Chandrasegaran et al. [2] demonstrated that a low-temperature, LS-trained teacher can notably improve distillation outcomes. Concurrently, Kornblith et al. [15] showed that LS tightens intra-class clusters in the feature space, diminishing transfer performance. From a Neural Collapse perspective [46, 8], LS
nudges the model toward rigid feature clusters, as evidenced by the reduced feature variability measured in Xu and Liu [41]. Our goal is to overcome LS's inherent error amplification effect. Rather than adjusting how the smoothed label distribution is constructed (as in OLS or Zipf-LS),
MaxSup directly penalizes the highest logit zmax. This design ensures consistent regularization even if zgt is not the top logit, thereby avoiding the degradation in performance typical of misclassified samples under LS (see Section 3.2). Moreover, MaxSup integrates seamlessly into standard training pipelines, introducing negligible computational overhead beyond substituting the LS term.

## 3 Max Suppression Regularization (Maxsup)

We first partition the training objective into two components: the standard Cross-Entropy (CE) loss and a regularization term introduced by Label Smoothing (LS). By expressing LS in terms of logits
(Theorem 3.3), we isolate two key factors: a regularization term that controls overconfidence and an error amplification term that enlarges the gap between the ground-truth logit zgt and any higher logits (Theorem 3.4, Equation (5)), ultimately degrading performance. To address these issues, we propose Max Suppression Regularization (MaxSup), which applies the penalty to the largest logit zmax rather than zgt (Equation (8), Section 3.2). This shift delivers consistent regularization for both correct and incorrect predictions, preserves intra-class variation, and bolsters inter-class separability. Consequently, MaxSup mitigates the representation collapse found in LS, attains superior ImageNet1K accuracy (Table 1), and improves transferability (Table 2, Table 3). The following sections elaborate on MaxSup's formulation and integration into the training pipeline.

## 3.1 Revisiting Label Smoothing

Label Smoothing (LS) is a regularization technique designed to reduce overconfidence by softening the target distribution. Rather than assigning probability 1 to the ground-truth class and 0 to all others, LS redistributes a fraction α of the probability uniformly across all classes:
Definition 3.1. For a standard classification task with K classes, Label Smoothing (LS) converts a one-hot label y ∈ R
K into a softened target label s ∈ R
K:

$$s_{k}=(1-\alpha)y_{k}+\frac{\alpha}{K},\tag{1}$$
$$(1)$$

where yk = 1{k=gt} denotes the ground-truth class. The smoothing factor α ∈ [0, 1] reduces the confidence assigned to the ground-truth class and distributes αK
to other classes uniformly, thereby mitigating overfitting, enhancing robustness, and promoting better generalization.

To clarify the effect of LS on model training, we first decompose the Cross-Entropy (CE) loss into a standard CE term and an additional LS-induced regularization term:
Lemma 3.2. *Decomposition of Cross-Entropy Loss with Soft Labels.*
H(s, q) = H(y, q) + LLS, (2)
where
$$L_{L S}\;=\;\alpha\,\biggl(H\Bigl({\frac{1}{K}},{\bf q}\Bigr)\;-\;H({\bf y},{\bf q})\Bigr).$$
. (3)
$$(2)$$
$=\;\;H$ . 
$\square$

$$({\mathfrak{I}})$$

Where, q is the predicted probability vector, H(·) *denotes the Cross-Entropy, and* 1K
is the uniform distribution introduced by LS. This shows that LS adds a regularization term, LLS*, which smooths the* output distribution and helps to reduce overfitting. (See Section A for a formal proof.)
Building on Theorem 3.2, we next explicitly express LLS at the logit level for further analysis.

Theorem 3.3. *Logit-Level Formulation of Label Smoothing Loss.*

$$L_{L S}\;=\;\alpha\Big(z_{g t}\;-\;\frac{1}{K}\sum_{k=1}^{K}z_{k}\Big),$$
, (4)
where zgt is the logit corresponding to the ground-truth class, and 1K
PK
k=1 zk is the average logit.

Thus, LS penalizes the gap between zgt and the average logit, encouraging a more balanced output distribution and reducing overconfidence. (See Section B for the proof.) The behavior of LLS differs depending on whether zgt is already the maximum logit. Specifically, depending on whether the prediction is correct (zgt = zmax) or incorrect (zgt ̸= zmax), we can decompose LLS into two parts:

  **Corollary 3.4**.: _Decomposition of Label Smoothing Loss._  $$L_{LS}\ =\ \underbrace{\frac{\alpha}{K}\sum_{z_{m}<z_{gt}}\left(z_{gt}-z_{m}\right)}_{\text{Regularization}}\ +\ \underbrace{\frac{\alpha}{K}\sum_{z_{n}>z_{gt}}\left(z_{gt}-z_{n}\right)}_{\text{Error amplification}},$$
$$(4)$$
$$(S)$$

$-\;1)$. 
, (5)
where M and N are the numbers of logits below and above zgt*, respectively (*M + N = K − 1). Note that the error amplification term vanishes when zgt = zmax.

1. **Regularization**: Penalizes the gap between zgt *and any smaller logits, thereby moderating* overconfidence.

2. **Error amplification**: Penalizes the gap between zgt and larger logits, inadvertently increasing overconfidence in incorrect predictions.

Although LS aims to combat overfitting by reducing prediction confidence, its error amplification component can be detrimental for misclassified samples, as it widens the gap between the ground-truth logit zgt and the incorrect top logit. Concretely:
1. **Correct Predictions** (zgt = zmax): The error amplification term is zero, and the regularization term effectively reduces overconfidence by shrinking the gap between zgt and any smaller logits.

2. **Incorrect Predictions** (zgt ̸= zmax): LS introduces two potential issues:
- **Error amplification**: Increases the gap between zgt and larger logits, reinforcing overconfidence in incorrect predictions.

- **Inconsistent Regularization**: The regularization term lowers zgt yet does not penalize zmax, which further impairs learning.

These issues with LS on misclassified samples have also been systematically observed in prior work [39]. By precisely disentangling these two components (regularization vs. error amplification), we can design a more targeted and effective solution.

Ablation Study on LS Components. To isolate the effects of each component in LS, we carefully perform a detailed and systematic ablation study on ImageNet-1K using a DeiT-Small model [36] without Mixup or CutMix. As indicated in Table 1, the performance gains from LS stem solely from the regularization term, whereas the error amplification term degrades accuracy. In contrast, our MaxSup omits the error amplification component and leverages only the beneficial regularization, thereby boosting accuracy beyond that of standard LS. Specifically, Table 1 shows that LS's overall improvement can be attributed exclusively to its regularization contribution; the error amplification term consistently reduces accuracy (e.g., to 73.63% or 73.69%). Disabling only the error amplification while retaining the regularization yields a slight but measurable improvement (75.98% vs. 75.91%). By fully removing error amplification and faithfully preserving the helpful aspects of LS, our MaxSup achieves 76.12% accuracy, clearly and consistently outperforming LS. This result underscores that MaxSup directly tackles LS's fundamental shortcoming by maintaining a consistent and meaningful regularization signal—even when the top-1 prediction is incorrect.

Table 1: Ablation on LS components using DeiT-Small on ImageNet-1K (without CutMix or Mixup). "Regularization" denotes penalizing logits smaller than zgt; "error amplification" penalizes logits larger than zgt. MaxSup removes error amplification while retaining regularization.

Method Formulation Accuracy Baseline - 74.21 + Label Smoothing

α K

Pzm<zgt

(zgt − zm)75.91

+

α

K

Pzn>zgt

(zgt − zn)

+ Regularization αM

Pzm<zgt

(zgt − zm) 75.98

+ error amplification αN

Pzn>zgt

(zgt − zn) 73.63

+ error amplification α (zgt − zmax) 73.69

| Method                | Formulation                 | Accuracy   |        |
|-----------------------|-----------------------------|------------|--------|
| Baseline              | -                           | 74.21      |        |
| α P zm<zgt (zgt − zm) | 75.91                       |            |        |
| + Label Smoothing     | K + α P zn>zgt (zgt − zn) K |            |        |
| + Regularization      | α M P zm<zgt (zgt − zm)     | 75.98      |        |
| + error amplification | α N P zn>zgt (zgt − zn)     | 73.63      |        |
| + error amplification | α (zgt − zmax)              | 73.69      |        |
|  zmax −                       | PK                          |             | 76.12  |
| + MaxSup              | α                           | 1          | k=1 zk |
| K                     |                             |            |        |

## 3.2 Max Suppression Regularization

Building on our analysis in Section 3.1, we find that Label Smoothing (LS) not only impacts correctly classified samples but also influences misclassifications in unintended and harmful ways. Specifically, LS suffers from two main limitations: inconsistent regularization and error amplification.

As illustrated in Table 1, LS penalizes the ground-truth logit zgt even in misclassified examples, needlessly widening the gap between zgt and the erroneous top-1 logit. To resolve these critical shortcomings, we propose Max Suppression Regularization (MaxSup), which explicitly penalizes the largest logit zmax rather than zgt. This key design choice ensures uniform regularization across both correct and misclassified samples, effectively eliminating the error-amplification issue in LS (Table 1) and preserving the ground-truth logit's integrity for more stable, robust learning.

## Definition 3.5. **Max Suppression Regularization**

We define the Cross-Entropy loss with MaxSup as follows:

$$\underbrace{H(\mathbf{s},\mathbf{q})}_{\mathrm{CE~with~Soft~Labels}}=\underbrace{H(\mathbf{y},\mathbf{q})}_{\mathrm{CE~with~Hard~Labels}}+\underbrace{L_{M a x}S u p}_{\mathrm{Max~Suppression~Loss}},$$
, (6)
where
$${\cal L}_{M a x S u p}\ =\ \alpha\Big(H\big(\frac{1}{K},{\bf q}\big)\ -\ H({\bf y^{\prime}},{\bf q})\Big),$$
, (7)
$$\operatorname{and}$$
$$y_{k}^{\prime}~=~{\mathds1}_{\left\{\begin{array}{l}{k=\arg\operatorname*{max}({\bf q})}\end{array}\right\}},$$
so that y
′k = 1 identifies the model's top-1 prediction and y
′k = 0 otherwise. Here, H1K
, q encourages a uniform output distribution to mitigate overconfidence, while H(y
′, q) penalizes the current top-1 logit. By shifting the penalty from zgt (the ground-truth logit) to zmax (the highest logit), MaxSup avoids unduly suppressing zgt when the model misclassifies, thus overcoming Label Smoothing's principal shortcoming. Logit-Level Formulation of MaxSup. Building on the logit-level perspective introduced for LS in Section 3.1, we can express L*MaxSup* as:

$$L_{MaxSup}\ =\ \alpha\biggl{(}z_{max}\ -\ \frac{1}{K}\sum_{k=1}^{K}z_{k}\biggr{)},\tag{1}$$
$$(6)$$
$$(T)$$
$$(\mathbf{8})$$

where zmax = maxk{zk} is the largest (top-1) logit, and 1K
PK
k=1 zk is the mean logit. Unlike LS, which penalizes the ground-truth logit zgt and may worsen errors in misclassified samples, MaxSup shifts the highest logit uniformly, thus providing consistent regularization for both correct and incorrect predictions. As shown in Table 1, this approach eliminates LS's error-amplification issue while preserving the intended overconfidence suppression.

Comparison with Label Smoothing. MaxSup fundamentally differs from LS in handling correct and incorrect predictions. When zgt = zmax, both LS and MaxSup similarly reduce overconfidence. However, when zgt ̸= zmax, LS shrinks zgt, widening the gap with the incorrect logit, whereas

MaxSup penalizes zmax, preserving zgt from undue suppression. As illustrated in Figure 2, this helps
the model recover from mistakes more effectively and avoid reinforcing incorrect predictions.
Gradient Analysis. To understand MaxSup's optimization dynamics, we compute its gradients with
respect to each logit zk. Specifically,
Specifically,  $$\frac{\partial L_{MaxSup}}{\partial z_{k}}\ =\ \begin{cases}\alpha\bigg{(}1-\frac{1}{K}\bigg{)},&\text{if}k=\arg\max(\mathbf{q}),\\ -\frac{\alpha}{K},&\text{otherwise}.\end{cases}\tag{9}$$
Thus, the top-1 logit zmax is reduced by α1 −
1 K
, while all other logits slightly increase by α K
.

In misclassified cases, the ground-truth logit zgt is spared from penalization, avoiding the erroramplification issue seen in LS. For completeness, Appendix A provides the full gradient derivation.

While [39] conducted a related gradient analysis of the training loss, it focuses specifically on the setting of selective classification, and examines a posthoc logit normalization technique to mitigate confidence calibration issues. However, this approach addresses only the overconfidence problem of label smoothing (LS), without tackling representation collapse. Moreover, our work presents a logit-level reformulation of LS that provides a deeper theoretical understanding of why LS amplifies errors. Behavior Across Different Samples. MaxSup applies a dynamic penalty based on the model's current predictions. For high-confidence, correctly classified examples, it behaves similarly to LS by reducing overconfidence, effectively mitigating overfitting. In contrast, for misclassified or uncertain samples, MaxSup aggressively suppresses the incorrect top-1 logit, further safeguarding the groundtruth logit zgt. This selective strategy preserves a faithful and reliable representation of the true class while actively discouraging error propagation. As shown in Section 4.2 and Table 5, this promotes more robust decision boundaries and leads to stronger generalization. Theoretical Insights and Practical Benefits. MaxSup provides both theoretical and practical advantages over LS. Whereas LS applies a uniform penalty to the ground-truth logit regardless of correctness, MaxSup penalizes only the most confident logit zmax. This dynamic adjustment robustly prevents error accumulation in misclassifications, ensuring more stable convergence. As a result, MaxSup generalizes better and achieves strong performance on challenging datasets. Moreover, as shown in Section 4.1, MaxSup preserves greater intra-class diversity, substantially improving transfer learning (Table 3) and yielding more interpretable activation maps (Figure 2).

## 4 Experiments

We begin by examining how MaxSup improves feature representations, then evaluate it on large-scale image classification and semantic segmentation tasks. Finally, we visualize class activation maps to illustrate the practical benefits of MaxSup.

## 4.1 Analysis Of Maxsup'S Learning Benefits

Having established how MaxSup addresses Label Smoothing's (LS) principal shortcomings (Section 3.1), we now demonstrate its impact on inter-class separability and intra-class variation—two properties essential for accurate classification and effective transfer learning.

## 4.1.1 Intra-Class Variation And Transferability

As noted in Section 3.1, Label Smoothing (LS) primarily curbs overconfidence for correctly classified samples but inadvertently triggers error amplification in misclassifications. This uneven penalization can overly compress intra-class feature representations. By contrast, Max- Sup uniformly penalizes the top-1 logit, whether the prediction is correct or incorrect, thereby eliminating LS's erroramplification effect and preserving finer distinctions within each class.

| Method        | d¯within ↑   | R 2 ↑   |       |       |
|---------------|--------------|---------|-------|-------|
| Train         | Val          | Train   | Val   |       |
| Baseline      | 0.311        | 0.331   | 0.403 | 0.445 |
| LS            | 0.263        | 0.254   | 0.469 | 0.461 |
| OLS           | 0.271        | 0.282   | 0.594 | 0.571 |
| Zipf-LS       | 0.261        | 0.293   | 0.552 | 0.479 |
| MaxSup        | 0.293        | 0.300   | 0.519 | 0.497 |
| Logit Penalty | 0.284        | 0.314   | 0.645 | 0.602 |

6 Table 2 compares intra-class variation (¯dwithin) and inter-class separability (R2) [15] for ResNet50 trained on ImageNet-1K. Although all investigated regularizers decrease ¯dwithin relative to a baseline, MaxSup yields the smallest reduction, indicating a stronger retention of subtle within-class diversity—widely associated with enhanced generalization and improved transfer performance.

These benefits are further underscored by the linear-probe transfer accuracy on CIFAR-10 (Table 3).

While LS and Logit Penalty each boost ImageNet accuracy, both degrade transfer accuracy, likely by suppressing informative and transferable features. By contrast, MaxSup preserves near-baseline performance, implying that it maintains rich discriminative information crucial for downstream tasks. For extended evaluations on diverse datasets, see Table 12 in the appendix.

## 4.1.2 Connection To Logit Penalty

Table 3: Linear-probe transfer accuracy on CIFAR-10 (higher is better).

| Method          | Acc.   |
|-----------------|--------|
| Baseline        | 0.814  |
| Label Smoothing | 0.746  |
| Logit Penalty   | 0.724  |
| MaxSup          | 0.810  |

As detailed in Section 3, both Label Smoothing (LS) variants and MaxSup impose penalties directly at the logit level, aligning with the perspective that various regularizers influence a model's representational capacity via distinct logit constraints [15]. Within this family of techniques, Logit Penalty and MaxSup both address the maximum logit, yet diverge fundamentally in their specific methods of regularization.

Logit Penalty minimizes the ℓ2-norm of the entire logit vector, inducing a global contraction that can improve class separation but also reduce intra-class diversity, potentially hindering downstream transfer. By contrast, MaxSup focuses exclusively on the top-1 logit, gently nudging it closer to the mean logit. Because only the highest-confidence prediction is penalized, MaxSup avoids the uniform shrinkage observed in Logit Penalty, preserving richer intra-class variation—a property essential for robust transfer. Further insights into this behavior can be found in Section L, where logit-value histograms illustrate how each method affects the logit distribution.

## 4.2 Evaluation On Imagenet Classification

Next, we compare MaxSup to standard Label Smoothing (LS) and various LS extensions on the large-scale ImageNet-1K dataset.

## 4.2.1 Experiment Setup

Model Training Configurations.We evaluate both convolutional (ResNet[10], MobileNetV2 [27]) and transformer (DeiT-Small [36]) architectures on ImageNet [17]. For the **ResNet Series**, we train for 200 epochs using stochastic gradient descent (SGD) with momentum0.9, weight decay of 1 × 10−4, and a batch size of 2048. The initial learning rate is 0.85 and is annealed via a cosine schedule.5 We also test ResNet variants on CIFAR-100 with a conventional setup: an initial learning rate of 0.1 (reduced fivefold at epochs 60, 120, and 160), training for 200 epochs with batch size 128 and weight decay 5 × 10−4. For **DeiT-Small**, we use the official codebase [36], training from scratch without knowledge distillation to isolate MaxSup's contribution. CutMix and Mixup are disabled to ensure the model optimization objective remains unchanged. Hyperparameters for Compared Methods.We compare Max Suppression Regularization against a range of LS extensions, including Zipf Label Smoothing[21] and Online Label Smoothing [45].

Where official implementations exist, we adopt them directly; otherwise, we follow the methodological details provided in each respective paper. Except for any method-specific hyperparameters, all other core training settings remain identical to the baselines. Furthermore, both MaxSup and standard LS employ a linearly increasing α-scheduler for improved training stability (see Section F). This ensures a fair comparison under consistent and reproducible training protocols.

## 4.2.2 Experiment Results

architectures tested, MaxSup consistently delivers the highest top-1 accuracy among label-smoothing approaches. By contrast, OLS [45] and Zipf-LS [21] exhibit less stable gains, suggesting their effectiveness may heavily hinge on specific training protocols. To reproduce OLS and Zipf-LS, we apply the authors' official codebases and hyperparameters but do not replicate their complete training recipes (e.g., OLS trains for 250 epochs with a step-scheduled learning rate of 0.1, and Zipf-LS uses 100 epochs with distinct hyperparameters). Even under these modified settings, MaxSup remains robust, highlighting its effectiveness across a variety of training schedules—unlike the more schedule-sensitive improvements noted for OLS and Zipf-LS.

Method ImageNet CIFAR-100

ResNet-18 ResNet-50 ResNet-101 MobileNetV2 ResNet-18 ResNet-50 ResNet-101 MobileNetV2

Baseline 69.09±0.12 76.41±0.10 75.96±0.18 71.40±0.12 76.16±0.18 78.69±0.16 79.11±0.21 68.06±0.06

Label Smoothing 69.54±0.15 76.91±0.11 77.37±0.15 71.61±0.09 77.05±0.17 78.88±0.13 79.19±0.25 69.65±0.08

Zipf-LS∗69.31±0.12 76.73±0.17 76.91±0.11 71.16±0.15 76.21±0.12 78.75±0.21 79.15±0.18 69.39±0.08 OLS∗69.45±0.15 77.23±0.21 77.71±0.17 71.63±0.11 77.33±0.15 78.79±0.12 79.25±0.15 68.91±0.11

MaxSup **69.96±0.13 77.69±0.07 78.18±0.12 72.08±0.17 77.82±0.15 79.15±0.13 79.41±0.19 69.88±0.07** Logit Penalty 68.48±0.10 76.73±0.10 77.20±0.15 71.13±0.10 76.41±0.15 78.90±0.16 78.89±0.21 69.46±0.08

DeiT Comparison. Table 5 summarizes performance for DeiT-Small on ImageNet across various regularization strategies. Notably, MaxSup attains a top-1 accuracy of 76.49%, surpassing standard Label Smoothing by 0.41%. In contrast, LS variants such as Zipf- LS and OLS offer only minor gains over LS, implying that their heavy reliance on data augmentation may limit their applicability to vision transformers. By outperforming both LS and its variants without additional data manipulations, MaxSup demonstrates robust feature enhancement. These findings underscore MaxSup's adaptability to different architectures and emphasize its utility in scenarios where conventional label-smoothing methods yield limited benefits.

Table 5: DeiT-Small top-1 accuracy (%), reported as mean ± standard deviation. Values in parentheses indicate absolute improvements over the baseline.

Method Mean Std Baseline 74.39 0.19 Label Smoothing 76.08 (+1.69) 0.16 Zipf-LS 75.89 (+1.50) 0.26 OLS 76.16 (+1.77) 0.18 MaxSup 76.49 (+2.10**) 0.12**
Fine-Grained Classification. Beyond largescale benchmarks like ImageNet, we further evaluate MaxSup on two fine-grained visual recognition tasks: CUB-200-2011 [37] and Stanford Cars [16]. These datasets pose unique challenges due to subtle inter-class differences, which often expose the limitations of standard regularization approaches. As shown in Table 6, MaxSup achieves the best performance across both datasets, surpassing LS and its recent variants. This demonstrates that MaxSup encourages the model to learn more discriminative and semantically rich representations that better capture fine-grained attributes, such as textures and part-level details. The consistent improvements on these benchmarks further validate MaxSup's capacity to generalize across different visual domains and its potential to enhance robustness in recognition scenarios where nuanced feature understanding is critical. Long-Tailed Classification. To assess the effectiveness of MaxSup under data imbalance, we performed experiments on the CIFAR-10-LT dataset with imbalance ratios of 50 and 100, following the experimental settings described in [35]. The corresponding results are summarized in Table 7. The evaluation compares three setups: Focal Loss, Focal Loss + LS, and Focal Loss + MaxSup. Across all imbalance ratios and splits (val/test), MaxSup consistently outperforms both the baseline and LS in overall accuracy, which jointly reflects the many-shot, medium-shot, and low-shot (minor class) performance. For example, at an imbalance ratio of 50 on the test split, MaxSup achieves 81.4% accuracy, outperforming Focal Loss (76.8%) by 4.6 percentage points, and LS (80.5%) by Table 6: Classification on CUB and Cars Datasets.

Method CUB[37] Cars[16] Baseline 80.88 90.27 LS 81.96 91.64 OLS 82.33 91.96 Zipf-LS 81.40 90.99 MaxSup **82.53 92.25**

Dataset Split Imbalance Ratio Method Overall Many Medium Low

LT CIFAR-10 val 50

Focal Loss 77.4 76.0 89.7 0.0

Label Smoothing 81.2 81.6 77.0 0.0

MaxSup **82.1** 82.5 78.1 0.0

LT CIFAR-10 test 50

Focal Loss 76.8 75.3 90.4 0.0

Label Smoothing 80.5 81.1 75.4 0.0

MaxSup **81.4** 82.3 73.4 0.0

LT CIFAR-10 val 100

Focal Loss 75.1 71.8 88.3 0.0

Label Smoothing 76.6 80.6 60.7 0.0

MaxSup **77.1** 80.1 65.1 0.0

LT CIFAR-10 test 100

Focal Loss 74.7 71.6 87.2 0.0

Label Smoothing **76.4** 80.8 59.0 0.0

MaxSup **76.4** 79.9 62.4 0.0

0.9 percentage points. These results indicate that MaxSup achieves a better trade-off between many- and medium-shot accuracy. While it does not fully resolve the challenge of imbalanced classification—especially for minority classes—it shows positive effects and offers a promising direction for further extension.

Corrupted Image Classification To evaluate the effectiveness of MaxSup on out-ofdistribution (OOD) settings, we also conducted experiments on CIFAR10-C benchmark [12] shown in Table 8 following settings in [11]. Table 8 reports the performance of MaxSup and Label Smoothing (LS) on this benchmark using ResNet-50 as the backbone. Specifically, LS yields a better NLL (1.5730 vs. 1.8431), implying more confident probabilistic predictions. However, MaxSup achieves a better ECE (0.1479 vs. 0.1741), indicating better calibration of the predicted confidence scores. These results validate that MaxSup remains effective on OOD datasets, achieving performance comparable to LS across all three metrics.

Ablation on the Weight Schedule. We also systematically investigate how different α scheduling strategies impact MaxSup's performance. Empirical results indicate that MaxSup consistently maintains high accuracy across a wide range of schedules, further underscoring its robustness against hyperparameter changes. For additional details and discussions, refer to Section F.

Table 8: Comparison of MaxSup, Label Smoothing (LS), and standard Cross Entropy (CE) on CIFAR-
10-C. Lower is better. Values show mean(std) across three setups.

Metric MaxSup LS CE Error (Corr) 0.362(0.055) 0.359(0.064) **0.354**(0.015) NLL (Corr) 1.770(0.103) **1.476**(0.111) 1.819(0.158) ECE (Corr) **0.145**(0.003) 0.158(0.015) 0.260(0.015)

## 4.3 Evaluation On Semantic Segmentation

We further investigate MaxSup's applicability to downstream tasks by evaluating its performance on semantic segmentation using the widely adopted MMSegmentation framework.6 Specifically, we adopt the Uper-
Net [40] architecture with a DeiT-Small backbone, trained on ADE20K. Models pretrained on ImageNet-1K with either MaxSup or Label Smoothing are then fine-tuned under the same cross-entropy objective (Section 4.2.2).

Table 9 shows that initializing with MaxSup-pretrained weights yields an mIoU of 42.8%, surpassing the 42.4% achieved by Label Smoothing. This improvement indicates that MaxSup fosters more

| Backbone   | Method          | mIoU        |
|------------|-----------------|-------------|
| Baseline   | 42.1            |             |
| DeiT-Small | Label Smoothing | 42.4 (+0.3) |
| MaxSup     | 42.8 (+0.7)     |             |

discriminative feature representations conducive to dense prediction tasks. By more effectively capturing class boundaries and within-class variability, MaxSup promotes stronger segmentation results, underscoring its potential to deliver features that are both transferable and highly robust.

## 4.4 Visualization Via Class Activation Maps

To better understand how MaxSup fundamentally differs from Label Smoothing (LS) in guiding model decisions, we employ Gradientweighted Class Activation Mapping (Grad- CAM) [29], which highlights regions most influential for each prediction. We evaluate DeiT-Small under three training setups: MaxSup (second row), LS (third row), and a baseline with standard cross-entropy (fourth row). As illustrated in Figure 2, MaxSup-trained models more effectively suppress background distractions than LS, which often fixates on unrelated objects—such as poles in "Bird," tubes in "Goldfish," and caps in "House Finch." This behavior reflects LS's error-enhancement mechanism, which can misdirect attention. Moreover, MaxSup retains a wider spectrum of salient features, as exemplified in the Shark" and Monkey" images, where LS-trained models often omit crucial semantic details (e.g., fins, tails, or facial contours). These findings align with our analysis in Section I, clearly demonstrating that MaxSup preserves richer intra-class information. Consequently, MaxSup-trained models produce more accurate and consistent predictions by effectively leveraging fine-grained object cues. Further quantitative Grad-CAM overlay metrics (e.g., precision and recall for target regions) confirm that MaxSup yields more focused and comprehensive activation maps, further underscoring its overall efficacy.

## 5 Conclusion

We examined the shortcomings of Label Smoothing (LS) and introduced Max Suppression Regularization (MaxSup) as a targeted and practical remedy. Our analysis shows that LS can unintentionally heighten overconfidence in misclassified samples by failing to sufficiently penalize incorrect top-1 logits. In contrast, MaxSup uniformly penalizes the highest logit, regardless of prediction correctness, thereby effectively eliminating LS's error amplification. Extensive experiments demonstrate that MaxSup not only improves accuracy but also preserves richer intra-class variation and enforces sharper inter-class boundaries, leading to more nuanced and transferable feature representations and superior transfer performance. Moreover, class activation maps confirm that MaxSup better attends to salient object regions, reducing focus on irrelevant background elements. Limitations. Prior work [23] notes that LS-trained teachers may degrade knowledge distillation [13, 14], and Guo et al. [8] suggests LS accelerates convergence via improved conditioning. Examining MaxSup's potential role in distillation and its overall impact on training dynamics would clarify these underlying effects. Recent studies [33, 7] also show that ℓ2 regularization biases final-layer features toward low-rank solutions, raising interesting questions about whether MaxSup behaves similarly. Impact. In practical applications, MaxSup shows strong promise for systems demanding robust generalization and efficient transfer, and we have not observed any additional adverse effects or trade-offs. By offering researchers and practitioners both a clearer understanding of LS's limitations and a straightforward, computationally light, and easily integrable method to overcome them, MaxSup may help guide the development of more reliable and interpretable deep learning models.

## Acknowledgments

This work was supported by the University of Washington Faculty Startup Fund, the Carwein– Andrews Fellowship, the UW GSFEI Top Scholar Award, the U.S. DOT PacTrans sub-center seed funding program, the DFG Research Unit 5336 - Learning to Sense (L2S), and the ELSA - European Lighthouse on Secure and Safe AI funded by the European Union under grant agreement No. 101070617. Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or European Commission. Neither the European Union nor the European Commission can be held responsible for them. We thank the anonymous reviewers for their helpful comments.

## References

[1] Duarte M Alves, Nuno M Guerreiro, João Alves, José Pombal, Ricardo Rei, José GC de Souza, Pierre Colombo, and André FT Martins. Steering large language models for machine translation with finetuning and in-context learning. *arXiv preprint arXiv:2310.13448*, 2023.

[2] Keshigeyan Chandrasegaran, Ngoc-Trung Tran, Yunqing Zhao, and Ngai-Man Cheung. Revisiting label smoothing and knowledge distillation compatibility: What was missing? In Proceedings of the International Conference on Machine Learning (ICML), pages 2890–2916.

PMLR, 2022.

[3] Shiming Chen, Guosen Xie, Yang Liu, Qinmu Peng, Baigui Sun, Hao Li, Xinge You, and Ling Shao. Hsva: Hierarchical semantic-visual adaptation for zero-shot learning. Advances in Neural Information Processing Systems (NeurIPS), 34:16622–16634, 2021.

[4] Yann Dauphin and Ekin Dogus Cubuk. Deconstructing the regularization of batchnorm. In International Conference on Learning Representations (ICLR), 2021.

[5] Yutong Feng, Jianwen Jiang, Mingqian Tang, Rong Jin, and Yue Gao. Rethinking supervised pre-training for better downstream transferring. *arXiv preprint arXiv:2110.06014*, 2021.

[6] Yingbo Gao, Weiyue Wang, Christian Herold, Zijian Yang, and Hermann Ney. Towards a better understanding of label smoothing in neural machine translation. In Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing, pages 212–223, 2020.

[7] Connall Garrod and Jonathan P Keating. The persistence of neural collapse despite low-rank bias: An analytic perspective through unconstrained features. *arXiv preprint arXiv:2410.23169*, 2024.

[8] Li Guo, Keith Ross, Zifan Zhao, George Andriopoulos, Shuyang Ling, Yufeng Xu, and Zixuan Dong. Cross entropy versus label smoothing: A neural collapse perspective. arXiv preprint arXiv:2402.03979, 2024.

[9] Qiushan Guo, Xinjiang Wang, Yichao Wu, Zhipeng Yu, Ding Liang, Xiaolin Hu, and Ping Luo. Online knowledge distillation via collaborative learning. In *Proceedings of the IEEE/CVF*
Conference on Computer Vision and Pattern Recognition (CVPR), pages 11020–11029, 2020.

[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern* Recognition (CVPR), pages 770–778, 2016.

[11] Markus Heinonen, Ba-Hien Tran, Michael Kampffmeyer, and Maurizio Filippone. Robust classification by coupling data mollification with label smoothing. In *International Conference* on Artificial Intelligence and Statistics, pages 4960–4968. PMLR, 2025.

[12] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In *International Conference on Learning Representations (ICLR)*, 2019.

[13] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network.

arXiv preprint arXiv:1503.02531, 2015.

[14] Xinting Hu, Kaihua Tang, Chunyan Miao, Xian-Sheng Hua, and Hanwang Zhang. Distilling causal effect of data in class-incremental learning. In *Proceedings of the IEEE/CVF Conference* on Computer Vision and Pattern Recognition (CVPR), 2021.

[15] Simon Kornblith, Ting Chen, Honglak Lee, and Mohammad Norouzi. Why do better loss functions lead to less transferable features? Advances in Neural Information Processing Systems
(NeurIPS), 34:28648–28662, 2021.

[16] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for finegrained categorization. In Proceedings of the IEEE international conference on computer vision workshops, pages 554–561, 2013.

[17] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. *Advances in neural information processing systems*, 25, 2012.

[18] Anders Krogh and John Hertz. A simple weight decay can improve generalization. Advances in neural information processing systems, 4, 1991.

[19] Yann LeCun. The mnist database of handwritten digits. *http://yann. lecun. com/exdb/mnist/*,
1998.

[20] Dongkyu Lee, Ka Chun Cheung, and Nevin L Zhang. Adaptive label smoothing with selfknowledge in natural language generation. *arXiv preprint arXiv:2210.13459*, 2022.

[21] Jiajun Liang, Linze Li, Zhaodong Bing, Borui Zhao, Yao Tang, Bo Lin, and Haoqiang Fan.

Efficient one pass self-distillation with zipf's label smoothing. In European Conference on Computer Vision (ECCV), pages 104–119. Springer, 2022.

[22] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In *Proceedings* of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 10012–10022, 2021.

[23] Rafael Müller, Simon Kornblith, and Geoffrey E Hinton. When does label smoothing help?

Advances in Neural Information Processing Systems (NeurIPS), 32, 2019.

[24] Zachary Novack, Julian McAuley, Zachary Chase Lipton, and Saurabh Garg. Chils: Zero-shot image classification with hierarchical label sets. In Proceedings of the International Conference on Machine Learning (ICML), pages 26342–26362. PMLR, 2023.

[25] Gabriel Pereyra, George Tucker, Jan Chorowski, Łukasz Kaiser, and Geoffrey Hinton. Regularizing neural networks by penalizing confident output distributions. arXiv preprint arXiv:1701.06548, 2017.

[26] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. *International journal of computer vision*, 115:211–252, 2015.

[27] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen.

Mobilenetv2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE/CVF*
Conference on Computer Vision and Pattern Recognition (CVPR), pages 4510–4520, 2018.

[28] Mert Bulent Sariyildiz, Yannis Kalantidis, Karteek Alahari, and Diane Larlus. No reason for no supervision: Improved generalization in supervised models. *arXiv preprint arXiv:2206.15369*, 2022.

[29] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradientbased localization. *International Journal of Computer Vision*, 128(2):336–359, October 2019.

ISSN 1573-1405. doi: 10.1007/s11263-019-01228-7. URL http://dx.doi.org/10.1007/
s11263-019-01228-7.

[30] Zhiqiang Shen, Zechun Liu, Dejia Xu, Zitian Chen, Kwang-Ting Cheng, and Marios Savvides.

Is label smoothing truly incompatible with knowledge distillation: An empirical study. arXiv preprint arXiv:2104.00676, 2021.

[31] Carlos N Silla and Alex A Freitas. A survey of hierarchical classification across different application domains. *Data mining and knowledge discovery*, 22:31–72, 2011.

[32] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.

Dropout: A simple way to prevent neural networks from overfitting. In Journal of Machine Learning Research, volume 15, pages 1929–1958, 2014.

[33] Peter Súkeník, Marco Mondelli, and Christoph Lampert. Neural collapse versus low-rank bias:
Is deep neural collapse really optimal? *arXiv preprint arXiv:2405.14468*, 2024.

[34] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2818–2826, 2016.

[35] Kaihua Tang, Jianqiang Huang, and Hanwang Zhang. Long-tailed classification by keeping the good and removing the bad momentum causal effect. Advances in neural information processing systems, 33:1513–1524, 2020.

[36] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In Proceedings of the International Conference on Machine Learning (ICML), pages 10347–10357.

PMLR, 2021.

[37] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011.

[38] Hongxin Wei, Renchunzi Xie, Hao Cheng, Lei Feng, Bo An, and Yixuan Li. Mitigating neural network overconfidence with logit normalization, 2022. URL https://arxiv.org/abs/ 2205.09310.

[39] Guoxuan Xia, Olivier Laurent, Gianni Franchi, and Christos-Savvas Bouganis. Understanding why label smoothing degrades selective classification and how to fix it. arXiv preprint arXiv:2403.14715, 2024.

[40] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In *European Conference on Computer Vision (ECCV)*, pages 418–434, 2018.

[41] Jing Xu and Haoxiong Liu. Quantifying the variability collapse of neural networks. In Proceedings of the International Conference on Machine Learning (ICML), pages 38535–38550.

PMLR, 2023.

[42] Kai Yi, Xiaoqian Shen, Yunhao Gou, and Mohamed Elhoseiny. Exploring hierarchical graph representation for large-scale zero-shot image classification. In *European Conference on* Computer Vision, pages 116–132. Springer, 2022.

[43] Li Yuan, Francis EH Tay, Guilin Li, Tao Wang, and Jiashi Feng. Revisiting knowledge distillation via label smoothing regularization. In *Proceedings of the IEEE/CVF Conference on* Computer Vision and Pattern Recognition (CVPR), pages 3903–3911, 2020.

[44] Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In European Conference on Computer Vision (ECCV), pages 818–833. Springer, 2014.

[45] Chang-Bin Zhang, Peng-Tao Jiang, Qibin Hou, Yunchao Wei, Qi Han, Zhen Li, and Ming-Ming Cheng. Delving deep into label smoothing. *IEEE Transactions on Image Processing*, 30: 5984–5996, 2021.

[46] Jinxin Zhou, Chong You, Xiao Li, Kangning Liu, Sheng Liu, Qing Qu, and Zhihui Zhu. Are all losses created equal: A neural collapse perspective. Advances in Neural Information Processing Systems (NeurIPS), 35:31697–31710, 2022.

[47] Yuxuan Zhou, Wangmeng Xiang, Chao Li, Biao Wang, Xihan Wei, Lei Zhang, Margret Keuper, and Xiansheng Hua. Sp-vit: Learning 2d spatial priors for vision transformers. *arXiv preprint* arXiv:2206.07662, 2022.

[48] Fei Zhu, Zhen Cheng, Xu-Yao Zhang, and Cheng-Lin Liu. Rethinking confidence calibration for failure prediction. In *European Conference on Computer Vision (ECCV)*, pages 518–536. Springer, 2022.

[49] Hui Zou and Trevor Hastie. Regularization and variable selection via the elastic net. *Journal of* the Royal Statistical Society Series B: Statistical Methodology, 67(2):301–320, 2005.

## Neurips Paper Checklist

1. **Claims**
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The paper's abstract and introduction outline the main contributions—including the identification of Label Smoothing (LS) shortcomings and the proposal of Max Suppression (MaxSup)—and these claims align with the theoretical and experimental sections.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes] Justification: A dedicated "Limitations" portion (or equivalent discussion) is provided, acknowledging possible extensions (e.g., knowledge distillation scenarios) and other open questions (e.g., interactions with ℓ2 regularization).

## 3. **Theory Assumptions And Proofs**

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: The paper includes formal statements and proofs in the main text and/or appendix (Lemma/Theorem with proofs in the supplementary material). All assumptions are clearly stated, and references are provided.

## 4. **Experimental Result Reproducibility**

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results? Answer: [Yes] Justification: The main text and appendix provide training pipelines, hyperparameters, datasets, and references to the code. Full details (batch sizes, learning rates, etc.) are included.

5. **Open access to data and code**
Question: Does the paper provide open access to the data and code? Answer: [Yes] Justification: The code is released (anonymized if needed), and the datasets used (ImageNet, CIFAR, etc.) are publicly available under their respective standard licenses.

## 6. **Experimental Setting/Details**

Question: Does the paper specify all the training and test details necessary to understand the results? Answer: [Yes] Justification: Section 4.2.1 and the appendix detail the setup (optimizers, data splits, learning rates, etc.). The authors specify how they selected key hyperparameters.

7. **Experiment statistical significance**
Question: Does the paper report error bars or statistical significance for the experiments? Answer: [Yes] Justification: Tables report "mean ± std" from multiple runs, reflecting the variability due to initialization or training seeds. This is shown in all main experimental tables.

8. **Experiments compute resources**
Question: Does the paper provide sufficient information about compute resources? Answer: [Yes] Justification: The text or appendix indicates GPU usage (e.g., ResNet on cluster GPUs), approximate training duration, and other relevant details. Though high-level, it suffices to gauge feasibility.

## 9. **Code Of Ethics**

Question: Does the research conform with the NeurIPS Code of Ethics? Answer: [Yes]
Justification: The work adheres to standard academic norms, uses publicly available datasets, and presents no known ethical concerns or conflicts with the NeurIPS Code of Ethics.

## 10. **Broader Impacts**

Question: Does the paper discuss both potential positive and negative societal impacts? Answer: [Yes] Justification: The "Impact" statement addresses potential benefits (improved accuracy and transfer, leading to more robust systems) and acknowledges that misuses are minimal given the method's purely algorithmic nature.

## 11. **Safeguards**

Question: Does the paper describe safeguards for high-risk data or models? Answer: [NA] Justification: The paper does not involve high-risk data (e.g., private user info) or high-risk models (e.g., generative LLMs). Standard ImageNet/CIFAR usage and training code are of no particular misuse risk.

## 12. **Licenses For Existing Assets**

Question: Are the creators of assets properly credited, and the licenses mentioned? Answer: [Yes]
Justification: The paper cites and credits publicly available code or datasets (ImageNet, CIFAR, etc.) with references to their original licenses or terms of service.

13. **New assets**
Question: Are newly introduced assets well documented? Answer: [NA] Justification: No new data or special code libraries are introduced beyond the regular code release. The approach modifies existing code for training frameworks but does not constitute a new dataset or model asset.

## 14. **Crowdsourcing And Research With Human Subjects**

Question: Are there human subjects or crowdsourcing experiments, with instructions and compensation described? Answer: [NA] Justification: The work involves no human subjects or crowdsourcing tasks.

15. **Institutional review board (IRB) approvals or equivalent**
Question: Does the paper discuss IRB approvals for human-subjects research? Answer: [NA] Justification: The paper does not involve human subjects; no IRB is necessary.

16. **Declaration of LLM usage**
Question: Does the paper describe usage of LLMs if it is essential to core methods in this research?

Answer: [Yes]
Justification: We used a Large Language Model (LLM) *solely for writing and polishing* the paper's text. The LLM was not involved in designing, conducting, or analyzing the experiments, nor in developing the core algorithmic contributions.

## A Technical Appendices And Supplementary Material

Technical appendices with additional results, figures, graphs and proofs may be submitted with the paper submission before the full submission deadline (see above), or as a separate PDF in the ZIP file below before the supplementary material deadline. There is no page limit for the technical appendices.

## A Proof Of Lemma 3.2

Proof. We aim to demonstrate the validity of Lemma 3.2, which states:

$$H(\mathbf{s},\mathbf{q})=H(\mathbf{y},\mathbf{q})+L_{L S}$$
H(s, q) = H(y, q) + LLS (10)
where LLS = αH1K
, q− H(y, q)
Let us proceed with the proof: We begin by expressing the cross-entropy H(s, q):

$$H(\mathbf{s},\mathbf{q})=-\sum_{k=1}^{K}s_{k}\log q_{k}$$
$$(10)$$
$$(11)$$

$$(12)$$

In the context of label smoothing, sk is defined as:

$$s_{k}=(1-\alpha)y_{k}+{\frac{\alpha}{K}}$$
K(12)
where α is the smoothing parameter, yk is the original label, and K is the number of classes. Substituting this expression for sk into the cross-entropy formula:

$$H(\mathbf{s},\mathbf{q})=-\sum_{k=1}^{K}\left((1-\alpha)y_{k}+{\frac{\alpha}{K}}\right)\log q_{k}$$

Expanding the sum:

$$H(\mathbf{s},\mathbf{q})=-(1-\alpha)\sum_{k=1}^{K}y_{k}\log q_{k}-{\frac{\alpha}{K}}\sum_{k=1}^{K}\log q_{k}$$
$$(13)$$
$$(14)$$
$$(15)$$
$$(16)$$

We recognize that the first term is equivalent to (1 − α)H(y, q), and the second term to αH(
1 K
, q).

Thus:

$$H(\mathbf{s},\mathbf{q})=(1-\alpha)H(\mathbf{y},\mathbf{q})+\alpha H\left({\frac{\mathbf{1}}{K}},\mathbf{q}\right)$$

Rearranging the terms:

$$H(\mathbf{s},\mathbf{q})=H(\mathbf{y},\mathbf{q})+\alpha\left(H\left({\frac{\mathbf{1}}{K}},\mathbf{q}\right)-H(\mathbf{y},\mathbf{q})\right)$$

We can now identify H(y, q) as the original cross-entropy loss and LLS = αH1K
, q− H(y, q)
as the Label Smoothing loss. Therefore, we have demonstrated that:

$$H(\mathbf{s},\mathbf{q})=H(\mathbf{y},\mathbf{q})+L_{L S}$$
H(s, q) = H(y, q) + LLS (17)
with LLS as defined in the lemma. It is noteworthy that the original cross-entropy loss H(y, q) remains unweighted by α in this decomposition, which is consistent with the statement in Lemma 3.2

$$(17)^{\frac{1}{2}}$$

## B Proof Of Theorem 3.3

Proof. We aim to prove the equation:

$$L_{L S}=\alpha(z_{g t}-\frac{1}{K}\sum_{k=1}^{K}z_{k})\tag{1}$$
$$(18)$$
$$(19)$$

Let s be the smoothed label vector and q be the predicted probability vector. We start with the cross-entropy between s and q:

$$H(\mathbf{s},\mathbf{q})=-\sum_{k=1}^{K}s_{k}\log q_{k}\tag{1}$$

With label smoothing, sk = (1 − α)yk +
α K
, where y is the one-hot ground truth vector and α is the smoothing parameter. Substituting this:

$$H(\mathbf{s},\mathbf{q})=-\sum_{k=1}^{K}[(1-\alpha)y_{k}+{\frac{\alpha}{K}}]\log q_{k}$$

Expanding:

$$H(\mathbf{s},\mathbf{q})=-(1-\alpha)\sum_{k=1}^{K}y_{k}\log q_{k}-{\frac{\alpha}{K}}\sum_{k=1}^{K}\log q_{k}$$
$$(20)$$
$$(21)$$

Since y is a one-hot vector, PK
k=1 yk log qk = log qgt, where gt is the index of the ground truth class:

$$H(\mathbf{s},\mathbf{q})=-(1-\alpha)\log q_{g t}-{\frac{\alpha}{K}}\sum_{k=1}^{K}\log q_{k}$$
$$(22)$$
$$(23)$$

Using the softmax function, qk =e zk PK
j=1 e zj, we can express log qk in terms of logits:

$$\log q_{k}=z_{k}-\log(\sum_{j=1}^{K}e^{z_{j}})\tag{1}$$

Substituting this into our expression:

$$H({\bf s},{\bf q})=-\,(1-\alpha)[z_{gt}-\log(\sum_{j=1}^{K}e^{z_{j}})]$$  $$-\,\frac{\alpha}{K}\sum_{k=1}^{K}[z_{k}-\log(\sum_{j=1}^{K}e^{z_{j}})]$$  $$=-\,(1-\alpha)z_{gt}+(1-\alpha)\log(\sum_{j=1}^{K}e^{z_{j}})\tag{24}$$  $$-\,\frac{\alpha}{K}\sum_{k=1}^{K}z_{k}+\alpha\log(\sum_{j=1}^{K}e^{z_{j}})$$  $$=-\,(1-\alpha)z_{gt}-\frac{\alpha}{K}\sum_{k=1}^{K}z_{k}+\log(\sum_{j=1}^{K}e^{z_{j}})$$

19 Rearranging:

$$H({\bf s},{\bf q})=-z_{gt}+\log(\sum_{j=1}^{K}e^{z_{j}})+\alpha[z_{gt}-\frac{1}{K}\sum_{k=1}^{K}z_{k}]\tag{25}$$

We can identify:

 - $H(\mathbf{y},\mathbf{q})=-z_{gt}+\log(\sum_{j=1}^{K}e^{z_j})$ (cross-entropy for one-hot vector $\mathbf{y}$)  - $L_{LS}=\alpha[z_{gt}-\frac{1}{K}\sum_{k=1}^{K}z_k]$
Thus, we have proven:
H(s, q) = H(y, q) + LLS (26)
Due to the broad usage of CutMix and Mixup in the training recipe of modern Neural Networks, we additionally take their impact into account together with Label Smoothing. Now we additionally prove the case **with Cutmix and Mixup**:

$$H(\mathbf{s},\mathbf{q})=H(\mathbf{y},\mathbf{q})+L_{L S}$$
$$L_{L S}^{\prime}=\alpha((\lambda z_{g t1}+(1-\lambda)z_{g t2})-\frac{1}{K}\sum_{k=1}^{K}z_{k})$$
$$s_{k}=(1-\alpha)(\lambda y_{k1}+(1-\lambda)y_{k2})+{\frac{\alpha}{K}}$$
$$(26)$$
$$(27)$$
$$(28)^{\frac{1}{2}}$$

With Cutmix and Mixup, the smoothed label becomes:
sk = (1 − α)(λyk1 + (1 − λ)yk2) + αK(28)
where yk1 and yk2 are one-hot vectors for the two ground truth classes from mixing, and λ is the mixing ratio. Starting with the cross-entropy:

$$H(\mathbf{s},\mathbf{q})=-\sum_{k=1}^{K}s_{k}\log q_{k}$$ $$=-\sum_{k=1}^{K}[(1-\alpha)(\lambda y_{k1}+(1-\lambda)y_{k2})+\frac{\alpha}{K}]\log q_{k}$$ $$=-(1-\alpha)\sum_{k=1}^{K}(\lambda y_{k1}+(1-\lambda)y_{k2})\log q_{k}-\frac{\alpha}{K}\sum_{k=1}^{K}\log q_{k}$$
$$(29)$$
(30)  $\binom{31}{2}$  (31)  . 
Since yk1 and yk2 are one-hot vectors:

$$H(\mathbf{s},\mathbf{q})=-(1-\alpha)(\lambda\log q_{g t1}+(1-\lambda)\log q_{g t2})-{\frac{\alpha}{K}}\sum_{k=1}^{K}\log q_{k}$$
$$(32)$$

where gt1 and gt2 are the indices of the two ground truth classes.

Using qk =e zk PK
j=1 e zj, we express in terms of logits:

$$H({\bf s},{\bf q})=-(1-\alpha)[\lambda(z_{g t1}-\log(\sum_{j=1}^{K}e^{z_{j}}))+(1-\lambda)(z_{g t2}-\log(\sum_{j=1}^{K}e^{z_{j}}))]$$  $$-\frac{\alpha}{K}\sum_{k=1}^{K}[z_{k}-\log(\sum_{j=1}^{K}e^{z_{j}})]$$
(33)  $\binom{34}{2}$  . 
zj))] (33)