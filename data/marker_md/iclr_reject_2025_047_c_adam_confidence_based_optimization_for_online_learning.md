**006**

**009**

**019**

**024**

**029 030**

**032**

**034**

**036**

# CADAM: CONFIDENCE-BASED OPTIMIZATION FOR ONLINE LEARNING

Anonymous authors Paper under double-blind review

### ABSTRACT

Modern recommendation systems frequently employ online learning to dynamically update their models with freshly collected data. The most commonly used optimizer for updating neural networks in these contexts is the Adam optimizer, which integrates momentum (mt) and adaptive learning rate (vt). However, the volatile nature of online learning data, characterized by its frequent distribution shifts and presence of noises, poses significant challenges to Adam's standard optimization process: (1) Adam may use outdated momentum and the average of squared gradients, resulting in slower adaptation to distribution changes, and (2) Adam's performance is adversely affected by data noise. To mitigate these issues, we introduce CAdam, a confidence-based optimization strategy that assesses the consistence between the momentum and the gradient for each parameter dimension before deciding on updates. If momentum and gradient are in sync, CAdam proceeds with parameter updates according to Adam's original formulation; if not, it temporarily withholds updates and monitors potential shifts in data distribution in subsequent iterations. This method allows CAdam to distinguish between the true distributional shifts and mere noise, and adapt more quickly to new data distributions. Our experiments with both synthetic and real-world datasets demonstrate that CAdam surpasses other well-known optimizers, including the original Adam, in efficiency and noise robustness. Furthermore, in large-scale A/B testing within a live recommendation system, CAdam significantly enhances model performance compared to Adam, leading to substantial increases in the system's gross merchandise volume (GMV).

# 1 INTRODUCTION

Modern recommendation systems, such as those used in online advertising platforms, rely on online learning to update real-time models with freshly collected data batches [\(Ko et al., 2022\)](#page-10-0). In online learning, models continuously adapt to users' interests and preferences based on immediate user interactions like clicks or conversions. Unlike traditional offline training—where data is pre-collected and static—online learning deals with streaming data that is often noisy and subject to frequent distribution changes. This streaming nature makes it challenging to effectively denoise and reorganize training samples [\(Su et al., 2024;](#page-11-0) [Zhang et al., 2021\)](#page-11-1).

A widely adopted optimizer in these systems is the Adam optimizer [\(Kingma & Ba, 2015\)](#page-10-1), which combines the strengths of parameter-adaptive methods and momentum-based methods. Adam adjusts learning rates based on the averaged gradient square norm (vt) and incorporates momentum (mt) for faster convergence. Its ability to maintain stable and efficient convergence by dynamically adjusting learning rates based on the first and second moments of gradients has made it a reliable choice for optimizing deep learning models across diverse applications, including image recognition [\(Alexey, 2020\)](#page-9-0), natural language processing [\(Vaswani, 2017\)](#page-11-2), and reinforcement learning [\(Schul](#page-11-3)[man et al., 2017\)](#page-11-3). However, Adam faces significant challenges in online learning environments. Specifically, it treats all incoming data equally, regardless of whether it originates from the original distribution, a new one, or is merely noise. This indiscriminate treatment leads to two key problems:

- 1. Outdated Momentum and Averaged Squared Gradients: When the data distribution shifts—a common occurrence in online systems due to factors such as daily cycles in shopping habits, rapidly changing trends on social media, seasonal changes, promotional events,

**059**

**061**

**064**

**067**

**069 070 071**

**074**

**079**

**089 090 091**

**094**

**104 105 106** and sudden market dynamics—Adam continues to use momentum and averaged squared gradients computed from previous data [\(Lu et al., 2018;](#page-10-2) [Viniski et al., 2021\)](#page-11-4). These outdated statistics can misguide the optimizer, resulting in slower adaptation to the new data distributions.

- 2. Sensitivity to Noise: Online learning data often contains noisy labels [\(Yang et al., 2023\)](#page-11-5). For example, in advertisement systems, users might click ads by mistake (false positives) or ignore ads they are interested in (false negatives) [\(Wang et al., 2021\)](#page-11-6). Sensitivity to such noise can affect convergence speed and may cause parameters to deviate from the correct optimization direction, especially in scenarios where noisy data constitutes a large proportion.

To address these issues inherent in online learning with Adam, we propose Confidence Adaptive Moment Estimation (CAdam), a novel optimization strategy that enhances Adam's robustness and adaptability. CAdam introduces a confidence metric that evaluates whether updating a specific parameter will be beneficial for the system. This metric is calculated by assessing the alignment between the current momentum and the gradient for each parameter dimension.

Specifically, if the momentum and the gradient point in the same direction, indicating consistency in the optimization path, CAdam proceeds with the parameter update following Adam's rule. Otherwise, if they point in opposite directions, CAdam pauses the update for that parameter to observe potential distribution changes in subsequent iterations. This strategy hinges on the idea that persistent opposite gradients suggest a distributional shift, as the momentum (an exponential moving average of past gradients) represents the recent trend. If the opposite gradients do not persist, it it likely to be noise, and the model resumes normal updates, effectively filtering out the noise.

By incorporating this simple, plug-and-play mechanism, CAdam retains the advantages of momentum-based optimization while enhancing robustness to noise and improving adaptability to meaningful distribution changes in online learning scenarios.

Our contribution can be summarized as follows:

- 1. We introduce CAdam, a confidence-based optimization algorithm that improves upon the standard Adam optimizer by addressing its limitations in handling noisy data and adapting to distribution shifts in real-time online learning.
- 2. Through extensive experiments on both synthetic and public datasets, we demonstrate that CAdam consistently outperforms popular optimizers in online recommendation settings.
- 3. We validate the real-world applicability of CAdam by conducting large-scale online A/B tests in a live system, proving its effectiveness in boosting system performance and achieving significant improvements in gross merchandise volume (GMV) worth millions of dollars.

# 2 RELATED WORK

Adam Extensions Adam is one of the most widely used optimizers, and researchers have proposed various modifications to address its limitations. AMSGrad [\(Reddi et al., 2018\)](#page-11-7) addresses Adam's non-convergence issue by introducing a maximum operation in the denominator of the update rule. RAdam [\(Liu et al., 2019\)](#page-10-3) incorporates a rectification term to reduce the variance caused by adaptive learning rates in the early stages of training, effectively combining the benefits of both adaptive and non-adaptive methods. AdamW [\(Loshchilov, 2017\)](#page-10-4) separates weight decay from the gradient update, improving regularization. Yogi [\(Zaheer et al., 2018\)](#page-11-8) modifies the learning rate using a different update rule for the second moment to enhance stability. AdaBelief [\(Zhuang et al.,](#page-12-0) [2020\)](#page-12-0) refines the second-moment estimation by focusing on the deviation of the gradient from its exponential moving average rather than the squared gradient. This allows the step size to adapt based on the "belief" in the current gradient direction, resulting in faster convergence and improved generalization. Our method, CAdam, similarly leverages the consistency between the gradient and momentum for adjustments. However, it preserves the original update structure of Adam and considers the sign (directional consistency) between momentum and gradient, rather than their value deviation, leading to better performance under distribution shifts and in noisy environments.

**114 115**

**117**

**119**

**127**

**129 130**

**134**

**136**

Adapting to Distributional Changes in Online Learning In online learning scenarios, models encounter data streams where the underlying distribution can shift over time, a phenomenon known as concept drift [\(Lu et al., 2018\)](#page-10-2). Adapting to these changes is essential for maintaining model performance. One common strategy is to use sliding windows or forgetting mechanisms [\(Bifet &](#page-9-1) [Gavalda, 2007\)](#page-9-1), which focus updates on the most recent data. Ensemble methods [\(Street & Kim,](#page-11-9) [2001\)](#page-11-9) maintains a collection of models trained on different time segments and combine their predictions to adapt to emerging patterns. Adaptive learning algorithms, such as Online Gradient Descent [\(Zinkevich, 2003\)](#page-12-1), dynamically adjust the learning rate or model parameters based on environmental feedback. Meta-learning approaches [\(Finn et al., 2017\)](#page-10-5) aim to develop models that can quickly adapt to new tasks or distributions with minimal updates. Additionally, [\(Viniski et al., 2021\)](#page-11-4) demonstrated that streaming-based recommender systems outperform batch methods in supermarket data, particularly in handling concept drifts and cold start scenarios.

Robustness to Noisy Data General methods for noise robustness include robust loss functions [\(Ghosh et al., 2017\)](#page-10-6), which modify the objective function to reduce sensitivity to mislabeled or corrupted data; regularization techniques [\(Srivastava et al., 2014\)](#page-11-10), which prevent overfitting by introducing noise during training; and noise-aware algorithms [\(Gutmann & Hyvarinen, 2010\)](#page-10-7), which ¨ explicitly model noise distributions to improve learning. In recommendation systems, enhancing robustness against noisy data is crucial and is typically addressed through two main strategies: *detect and correct* and *detect and remove*. *Detect and correct* methods, such as AutoDenoise [\(Ge](#page-10-8) [et al., 2023\)](#page-10-8) and Dual Training Error-based Correction (DTEC) [\(Panagiotakis et al., 2021\)](#page-10-9), identify noisy inputs and adjust them to improve model accuracy by leveraging mechanisms like validation sets or dual error perspectives. Conversely, *detect and remove* approaches eliminate unreliable data using techniques such as outlier detection with statistical models [\(Xu et al., 2022\)](#page-11-11) or semantic coherence assessments [\(Saia et al., 2016\)](#page-11-12) to cleanse user profiles. While these strategies can effectively enhance recommendation quality, they often require explicit design and customization for specific models or tasks, limiting their general applicability.

# 3 DETAILS OF CADAM OPTIMIZER

Notations We use the following notations for the CAdam optimizer:

- f(θ) ∈ <sup>R</sup>, θ ∈ <sup>R</sup> d : f is the stochastic objective function to minimize, where θ is the parameter vector in R d .
- gt: the gradient at step t, g<sup>t</sup> = ∇θft(θt−1).
- mt: exponential moving average (EMA) of gt, calculated as m<sup>t</sup> = β<sup>1</sup> ·mt−1+ (1−β1)·gt.
- vt: EMA of the squared gradients, given by v<sup>t</sup> = β<sup>2</sup> · vt−<sup>1</sup> + (1 − β2) · g 2 t .
- mˆ <sup>t</sup>, vˆt: bias-corrected estimates of m<sup>t</sup> and vt, respectively, where mˆ <sup>t</sup> = m<sup>t</sup> 1−β and vˆ<sup>t</sup> = vt 1−β t .
- α, ϵ: α is the learning rate, typically set to 10−<sup>3</sup> , and ϵ is a small constant to prevent division by zero, typically set to 10−<sup>8</sup> .
- β1, β2: smoothing parameters, commonly set as β<sup>1</sup> = 0.9, β<sup>2</sup> = 0.999.
- θt: the parameter vector at step t.
- θ0: the initial parameter vector.

Comparison with Adam CAdam (Algorithm [1\)](#page-3-0) and Adam both use the first and second moments of gradients to adapt learning rates. The main difference between CAdam and Adam is that CAdam introduces the alignment between the momentum and the gradient as a confidence metric to address two common problems in real-world online learning: distribution shifts and noise.

In Adam, the update direction is determined by mt, the exponential moving average (EMA) of the gradient gt, and vt, the EMA of the squared gradients g 2 t . This method assumes a relatively stable data distribution, where m<sup>t</sup> serves as a good estimator of the optimal update direction. However, if the data distribution changes, m<sup>t</sup> may no longer point in the correct direction. Adam will continue to update using the outdated m<sup>t</sup> for several iterations until it eventually aligns with the new gradient

**163 164 166 167 169 171 174 175 176 177 178** 1: m<sup>0</sup> ← 0, v<sup>0</sup> ← 0, vˆmax,<sup>0</sup> ← 0, t ← 0, θ<sup>t</sup> = θ<sup>0</sup> 2: while θ<sup>t</sup> not converged do 3: t ← t + 1 4: g<sup>t</sup> ← ∇θft(θt−1) 5: m<sup>t</sup> ← β<sup>1</sup> · mt−<sup>1</sup> + (1 − β1) · g<sup>t</sup> 6: v<sup>t</sup> ← β<sup>2</sup> · vt−<sup>1</sup> + (1 − β2) · g 2 t 7: mˆ <sup>t</sup> ← mt/(1 − β t 1 ) 8: vˆ<sup>t</sup> ← vt/(1 − β t 2 ) 9: if AMSGrad then 10: vˆmax,t ← max(ˆvmax,t−1, vˆt) 11: else 12: vˆmax,t ← vˆ<sup>t</sup> 13: end if 14: mˆ <sup>t</sup> ← max(0, m<sup>t</sup> · sign(gt)) ▷ Element-wise mask out elements where m<sup>t</sup> · g<sup>t</sup> ≤ 0 15: θ<sup>t</sup> ← θt−<sup>1</sup> − α · mˆ <sup>t</sup>/( p vˆmax,t + ϵ) 16: end while 17: return θ<sup>t</sup>

**179 180 181**

**204**

**206**

Algorithm 1 Confidence Adaptive Moment Estimation (CAdam)

direction, leading to poor performance during this adaptation period. Additionally, when encountering noisy examples, Adam blindly updates using mt, which can be problematic as it equivalently increases the learning rate, especially when the proportion of noisy data is high.

In contrast, CAdam dynamically checks the *alignment* between the current gradient g<sup>t</sup> and the momentum m<sup>t</sup> before proceeding with an update. If g<sup>t</sup> and m<sup>t</sup> point in the same direction, indicating that the momentum aligns with the current gradient, CAdam performs the update using mt/ √ vt. However, if g<sup>t</sup> and m<sup>t</sup> point in opposite directions, CAdam pauses the update for that parameter to observe subsequent gradients. This pause allows CAdam to distinguish between a potential distribution shift and noise.

If the reverse gradient signs persist in subsequent steps, it signals a distribution shift, and m<sup>t</sup> will gradually change direction to reflect the new data pattern, while CAdam doesn't update in these iterations, avoiding incorrect updates. Conversely, if the gradient signs realign in the following steps, it indicates that the previous opposite gradient was caused by noise. In this case, CAdam resumes normal updates, effectively filtering out noisy gradients without making unnecessary updates in the process.

In addition, CAdam also has an AMSGrad [\(Reddi et al., 2018\)](#page-11-7) variant as described in [1](#page-3-0) when AMSGrad option is enabled.

Convergence Analysis Given a stream of functions f<sup>t</sup> : <sup>R</sup> <sup>d</sup> → <sup>R</sup>, t = 1, 2, . . . , T, an online learning algorithm chooses θ<sup>t</sup> in each time step t and aims to minimize the T-step regret w.r.t. the optimum, where the regret is defined as

$$R_T := \sum_{t=1}^T f_t(\theta_t) - \sum_{t=1}^T f_t(\theta^*), \quad \theta^* = \operatorname{argmin}_{\theta} \sum_{t=1}^T f_t(\theta). \quad (1)$$

The online learning setting has been widely used to model real-world recommendation scenarios. We show that CAdam has the same O( √ T) regret as Adam/AMSGrad under the same assumptions made in [Reddi et al.](#page-11-7) [\(2018\)](#page-11-7). The detailed proofs can be found in the appendix.

Theorem 1 (Informal). *Under the assumptions introduced in [Reddi et al.](#page-11-7) [\(2018\)](#page-11-7), the CAdam algorithm (with AMSGrad correction) achieves a sublinear regret; that is,*

$$R_T = \mathcal{O}(\sqrt{T}). \quad (2)$$

Remark: We follow the regret analysis in [Reddi et al.](#page-11-7) [\(2018\)](#page-11-7) and adopt the same set of assumptions. In particular, [Reddi et al.](#page-11-7) [\(2018\)](#page-11-7) only considered convex functions and made bounded gradient assumption. Recently, there is a body of work that has provided refined convergence analysis under

**224**

**236 237**

**254**

**256**

**259**

![](_page_4_Figure_1.jpeg)

Figure 1: Trajectory of Adam (top row) and CAdam (bottom row) under different distribution changes: (Left) sudden change, (Middle) linear change, and (Right) sinusoidal change. The first row corresponds to the L1 loss landscapes, while the second row corresponds to the L2 loss landscapes. Adam's X and CAdam's X denote the locations of the optimization trajectories for Adam and CAdam, respectively, while X<sup>∗</sup> represents the location of the optimal solution. CAdam shows superior adaptability to distribution shifts.

nonconvex setting and much weaker assumptions (see e.g., [Alacaoglu et al.](#page-9-2) [\(2020\)](#page-9-2); [Defossez et al.;](#page-9-3) ´ [Zhang et al.](#page-11-13) [\(2022\)](#page-11-13); [Wang et al.](#page-11-14) [\(2024\)](#page-11-14)). We leave the analysis of C-Adam under these more general settings as an interesting future direction.

### 4 EXPERIMENT

In this section, we systematically evaluate the performance of CAdam across various scenarios, starting with synthetic image data, followed by tests on a public advertisement dataset, and concluding with A/B tests in a real-world recommendation system. We first examine CAdam's behaviour under distribution shift, and noisy conditions using the CIFAR-10 dataset[\(Krizhevsky et al., 2009\)](#page-10-10) with the VGG network[\(Simonyan & Zisserman, 2014\)](#page-11-15). Next, we test CAdam against other popular optimizers on the Criteo dataset[\(Jean-Baptiste Tien, 2014\)](#page-10-11), focusing on different models and scenarios. Finally, we conduct A/B tests with millions of users in a real-world recommendation system to validate CAdam's effectiveness in large-scale, production-level environments. The results demonstrate that CAdam consistently outperforms Adam and other optimizers across different tasks, distribution shifts, and noise conditions.

#### 4.1 NUMERICAL EXPERIMENT

Distribution Change To illustrate the different behaviours of Adam and CAdam under distribution shifts, we designed three types of distribution changes for both L1 and L2 loss functions: (1) *Sudden* change, where the minimum shifts abruptly at regular intervals; (2) *Linear* change, where the minimum moves at a constant speed; and (3) *Sinusoidal* change, where the minimum oscillates following a sine function, resulting in variable speed over time.

The loss functions are defined as:

$$L(x, t) = \begin{cases} |x - x^*(t)|, & \text{L1 loss,} \\ (x - x^*(t))^2, & \text{L2 loss,} \end{cases}$$

![](_page_5_Figure_1.jpeg)

**289 290 291**

**294**

**301**

**304**

**306**

**309**

**314 315**

**318 319**

Figure 2: Trajectory of Adam (top row) and CAdam (bottom row) under noisy conditions on four different optimization landscapes: (Left to Right) separable L1 loss, inseparable L1 loss, inseparable L2 loss, and Rosenbrock function. Each column shows the optimization trajectory in the presence of noise, where each dimension's gradient is randomly flipped with a 50% probability. CAdam demonstrates superior robustness, maintaining more stable convergence paths than Adam across all tested functions.

where x ∗ (t) represents the position of the minimum at time t and is defined based on the type of distribution change:

$$x^*(t) = \begin{cases} \left\lfloor \frac{t}{T} \right\rfloor & \text{mod } 2, & \text{sudden change,} \\ \frac{t}{T}, & \text{linear change,} \\ \sin\left(\frac{2\pi t}{T}\right), & \text{sinusoidal change.} \end{cases}$$

The results of these experiments are presented in Figure [1.](#page-4-0) Across different loss functions and distribution changes, CAdam closely follows the trajectory of the minimum point, being less affected by incorrect momentum, exhibiting lower regret and demonstrating its superior ability to adapt to shifting distributions.

Noisy Samples To compare Adam and CAdam in noisy environments, we conducted experiments on four different optimization 2-d landscapes: (1) separable L1 loss, (2) inseparable L1 loss, (3) inseparable L2 loss, and (4) Rosenbrock function. These landscapes are defined as follows:

- 1. Separable L1 Loss: f1(x, y) = |x| + |y|.
- 2. Inseparable L1 Loss: f2(x, y) = |x + y| + |x−y| <sup>10</sup> .
- 3. Inseparable L2 Loss: f3(x, y) = (x + y) <sup>2</sup> + (x−y) 2 <sup>10</sup> .
- 4. Rosenbrock Function: f4(x, y) = (a − x) <sup>2</sup> + b(y − x 2 ) 2 , where a = 1 and b = 100.

To simulate noise in the gradients, we applied a random mask to each dimension of the gradient with a 50% probability using the same random seed across different optimizers. Specifically, the gradient components were multiplied by a uniformly distributed random value from the range [−1, 1] to introduce noise:

$$\nabla_{\text{noisy}}(x, y) = \begin{cases} \nabla f(x, y) \cdot U(-1, 1), & \text{with probability } p = 0.5, \\ \nabla f(x, y), & \text{otherwise,} \end{cases}$$

The results of these experiments are shown in Figure [2.](#page-5-0) For comparison, the results without noise are provided in Figure [5](#page-18-0) in the appendix. The trajectory of CAdam exhibits fewer random perturbations and lower regret, indicating its ability to resist noise interference.

![](_page_6_Figure_1.jpeg)

**354 355 356**

**358 359**

**361**

**364**

**369**

Figure 3: (Left) Performance of CAdam and Adam under different rotation speeds corresponding to sudden distribution shift. CAdam demonstrates superior performance, with a more pronounced advantage over Adam in the presence of rotation. (Right) A detailed view at a 60-degree rotation between steps 1400 to 2300, showing the Alignment Ratio, Accuracy, and Loss. The red dashed lines indicate the rotation points, where the alignment ratio decreases, resulting in fewer parameter updates. This is followed by a gradual recovery in both the alignment ratio and accuracy, and a decline in loss. CAdam's accuracy drop is slower, and its recovery is faster than Adam's, illustrating its enhanced ability to adapt to distribution shifts.

### 4.2 CNN ON IMAGE CLASSIFICATION

We perform experiments using the VGG network on the CIFAR-10 dataset to evaluate the effectiveness of CAdam in handling distribution shifts and noise. We synthesize three experimental conditions: (1) sudden distribution changes, (2) continuous distribution shifts, and (3) added noise to the samples. The hyperparameters for these experiments are provided in Section [B.2.](#page-17-0)

Sudden Distribution Shift To simulate sudden changes in data distribution, we rotate the images by a specific angle at the start of each epoch, relative to the previous epoch, as illustrated in Figure [3.](#page-6-0) CAdam consistently outperforms Adam across varying rotation speeds, with a more significant performance gap compared to the non-rotated condition.

We define the *alignment ratio* as:

$$\text{Alignment Ratio} = \frac{\text{Number of parameters where } m_t \cdot g_t > 0}{\text{Total number of parameters}}$$

A closer inspection in Figure [3](#page-6-0) reveals that, during the rotation (indicated by the red dashed line), the alignment ratio decreases, resulting in fewer parameters being updated, followed by a gradual recovery. Correspondingly, the accuracy declines and subsequently improves, while the loss increases before decreasing. Notably, during these shifts, CAdam's accuracy drops more slowly and recovers faster than Adam's, indicating its superior adaptability to new data distributions.

Continuous Distribution Shifts In contrast to sudden distribution changes, we also tested the scenario where the data distribution changes continuously. Specifically, we simulated this by rotating the data distribution at each iteration by an angle. The results, shown in Figure [4,](#page-7-0) indicate that as the rotation speed increases, the advantage of CAdam over Adam becomes more pronounced.

![](_page_7_Figure_1.jpeg)

Figure 4: (Left) Performance of CAdam and Adam under continuous distribution shifts with different rotation speeds. CAdam demonstrates superior performance, with its advantage becoming more pronounced as the rotation speed increases. (Right) The effect of adding noise to the samples. CAdam exhibits a slower accuracy drop compared to Adam, showcasing its enhanced robustness to noisy data.

Noisy Samples To evaluate the optimizer's robustness to noise, we introduced noise into the dataset by randomly selecting a certain number of batches in each epoch (resampling for each epoch) and replacing the labels of these batches with random values. The results are presented in Figure [4.](#page-7-0) We observed that as the proportion of noisy labels increases, the consistency of CAdam decreases, causing it to update fewer parameters in each iteration. Despite this, both CAdam and Adam experience a performance decline in test set accuracy as noise increases. Nevertheless, CAdam consistently outperforms Adam, maintaining accuracy even with 40% noise, comparable to Adam's performance in a noise-free setting by the end of training.

### 4.3 PUBLIC ADVERTISEMENT DATASET

Experiment Setting To evaluate the effectiveness of the proposed CAdam optimizer, we conducted experiments using various models on the Criteo-x4-001 dataset[\(Jean-Baptiste Tien, 2014\)](#page-10-11). This dataset contains feature values and click feedback for millions of display ads and is commonly used to benchmark algorithms for click-through rate (CTR) prediction[\(Zhu et al., 2021\)](#page-11-16). To simulate a real-world online learning scenario, we trained the models on data up to each timestamp in a single epoch[\(Fukushima et al., 2020\)](#page-10-12). This setup replicates the environment where new data arrives continuously, requiring the model to adapt quickly.

Furthermore, for sparse parameters (e.g., embeddings), we update the optimizer's state only when there is a non-zero gradient for this parameter in the current batch using SparseAdam implementation in Pytorch[\(Paszke et al., 2019\)](#page-10-13). This approach ensures that the optimizer's state reflects the parameters influenced by recent data changes. The hyperparameters are provided in Appendix [B.3.](#page-17-1)

We benchmarked CAdam and other popular optimizers, including SGD, SGDM[\(Qian, 1999\)](#page-10-14), AdaGrad[\(Duchi et al., 2011\)](#page-9-4), AdaDelta[\(Zeiler, 2012\)](#page-11-17), RMSProp, Adam[\(Kingma & Ba, 2015\)](#page-10-1), AMSGrad[\(Reddi et al., 2018\)](#page-11-7), and AdaBelief[\(Zhuang et al., 2020\)](#page-12-0), on various models such as DeepFM(77M)[\(Guo et al., 2017\)](#page-10-15), WideDeep(77M)[\(Cheng et al., 2016\)](#page-9-5), DNN(74M)[\(Covington](#page-9-6) [et al., 2016\)](#page-9-6), PNN(79M)[\(Qu et al., 2016\)](#page-10-16), and DCN(74M)[\(Wang et al., 2017\)](#page-11-18). The performance of these optimizers was evaluated using the Area Under the Curve (AUC) metric.

Table 1: AUC performance of different optimizers on the Criteo dataset across various models. Results are averaged over three seeds with mean and standard deviation (±) reported. CAmsGrad denotes the AMSGrad variant of CAdam, which achieves the highest average performance.

|           |    |    | DeepFM |    |    |       |    |    | DNN   |    |    | PNN   |    |    | DCN   |    | Avg |
|-----------|----|----|--------|----|----|-------|----|----|-------|----|----|-------|----|----|-------|----|-----|
| SGD       | 71 | 90 | ± 006  | 71 | 88 | ± 013 | 68 | 12 | ± 043 | 67 | 61 | ± 318 | 69 | 55 | ± 026 | 69 | 81  |
| SGDM      | 76 | 59 | ± 044  | 76 | 32 | ± 021 | 78 | 80 | ± 014 | 76 | 17 | ± 050 | 77 | 90 | ± 018 | 77 | 16  |
| AdaGrad   | 71 | 77 | ± 032  | 71 | 50 | ± 011 | 68 | 65 | ± 022 | 67 | 49 | ± 027 | 69 | 55 | ± 020 | 69 | 79  |
| AdaDelta  | 71 | 91 | ± 071  | 71 | 64 | ± 005 | 69 | 76 | ± 004 | 67 | 59 | ± 025 | 69 | 76 | ± 024 | 70 | 13  |
| RMSProp   | 71 | 82 | ± 010  | 71 | 54 | ± 021 | 68 | 72 | ± 005 | 67 | 51 | ± 004 | 69 | 60 | ± 007 | 69 | 84  |
| Adam      | 80 | 87 | ± 011  | 80 | 90 | ± 004 | 80 | 89 | ± 003 | 80 | 90 | ± 006 | 81 | 05 | ± 005 | 80 | 92  |
| AdaBelief | 80 | 84 | ± 008  | 80 | 90 | ± 002 | 80 | 88 | ± 011 | 80 | 89 | ± 002 | 81 | 02 | ± 044 | 80 | 91  |
| AdamW     | 80 | 87 | ± 008  | 80 | 90 | ± 010 | 80 | 88 | ± 010 | 80 | 90 | ± 002 | 81 | 00 | ± 047 | 80 | 91  |
| AmsGrad   | 80 | 88 | ± 004  | 80 | 92 | ± 008 | 80 | 91 | ± 001 | 80 | 92 | ± 009 | 81 | 08 | ± 009 | 80 | 94  |
| CAdam     | 80 | 88 | ± 008  | 80 | 93 | ± 004 | 80 | 90 | ± 002 | 80 | 93 | ± 006 | 81 | 06 | ± 009 | 80 | 94  |
| CAmsGrad  | 80 | 90 | ± 006  | 80 | 93 | ± 007 | 80 | 92 | ± 005 | 80 | 94 | ± 009 | 81 | 09 | ± 010 | 80 | 96  |

Main Results The results in Table [1](#page-8-0) show that CAdam and its AMSGrad variants outperform other optimizers across different models. While the AMSGrad variants perform better on certain datasets, they do not consistently outperform standard CAdam. Both versions of CAdam generally achieve higher AUC scores than other optimizers, demonstrating their effectiveness in the online learning setting.

Robustness under Noise To simulate a noisier environment, we introduced noise into the Criteo x4-001 dataset by flipping 1% of the negative training samples to positive. All other settings remained unchanged. The results in Table [2](#page-8-1) show that CAdam consistently outperforms Adam in terms of both AUC and the extent of performance drop. This demonstrates CAdam's robustness in handling noisy data.

Table 2: Results of Adam and CAdam on the Noisy Criteo dataset, averaged over three seeds. "Drop" indicates the decrease in performance compared to training on the original Criteo dataset. CAdam shows a smaller performance drop, highlighting its robustness to noise.

|            | DeepFM       | WideDeep     | DNN          | PNN          | DCN          |
|------------|--------------|--------------|--------------|--------------|--------------|
| Adam       | 80 51 ± 008  | 80 47 ± 006  | 80 48 ± 014  | 80 66 ± 006  | 80 51 ± 010  |
| CAdam      | 80 81 ± 007  | 80 79 ± 006  | 80 78 ± 005  | 80 96 ± 026  | 80 77 ± 007  |
| Adam Drop  | − 0 36 ± 014 | − 0 43 ± 007 | − 0 41 ± 016 | − 0 23 ± 012 | − 0 54 ± 013 |
| CAdam Drop | − 0 08 ± 014 | − 0 14 ± 009 | − 0 12 ± 004 | + 0 04 ± 031 | − 0 28 ± 015 |

### 4.4 EXPERIMENT ON REAL-WORLD RECOMMENDATION SYSTEM

In real-world recommendation scenarios, the differences from the Criteo dataset experiments are quite significant. First, both data volume and model sizes are much larger, with models used in the following experiments ranging from 8.3 billion to 330 billion parameters—100 to 10,000 times larger. Second, as these are online experiments, unlike offline experiments with a fixed dataset, the model's output directly influences user behaviour. To test the effectiveness of CAdam in this setting, we conducted A/B tests on internal models serving millions of users across seven different scenarios (2 pre-ranking, 4 recall, and 1 ranking).

During these online experiments, we used a batch size of B = 4096 The evaluation metric was the Generalized Area Under the Curve (GAUC). Due to limited resources, we compared only Adam and CAdam, running the experiments for 48 hours.

The results, shown in Table [3,](#page-9-7) indicate that CAdam consistently outperformed Adam across all test scenarios, demonstrating its superiority in real-world applications.

**504**

**506**

**509**

**514 515 516**

**518 519**

**524**

**529**

**539**

Table 3: GAUC results for Adam and CAdam across seven internal experiment settings. "Pr" denotes pre-ranking, "Rec" represents recall, and "Rk" indicates ranking. CAdam consistently outperforms Adam, highlighting its effectiveness in real-world recommendation scenarios.

| Metric | Pr     | 1 Pr   | 2 Rec  | 1 Rec  | 2 Rec  | 3 Rec  | 4 Rk   | 1 Average |
|--------|--------|--------|--------|--------|--------|--------|--------|-----------|
| Adam   | 87.41% | 82.89% | 90.18% | 82.41% | 84.57% | 85.39% | 88.52% | 85.34%    |
| CAdam  | 87.61% | 83.28% | 90.43% | 82.61% | 85.06% | 85.49% | 88.74% | 85.64%    |
| Impr.  | 0.20%  | 0.39%  | 0.25%  | 0.20%  | 0.49%  | 0.10%  | 0.22%  | 0.30%     |

## 5 CONCLUSION

In this paper, we addressed the inherent limitations of the Adam optimizer in online learning environments, particularly its sluggish adaptation to distributional shifts and heightened sensitivity to noisy data. To overcome these challenges, we introduced CAdam (Confidence Adaptive Moment Estimation), a novel optimization strategy that enhances Adam by incorporating a confidence-based mechanism. This mechanism evaluates the alignment between momentum and gradients for each parameter dimension, ensuring that updates are performed judiciously. When momentum and gradients are aligned, CAdam updates the parameters following Adam's original formulation; otherwise, it temporarily withholds updates to discern between true distribution shifts and transient noise.

Our extensive experiments across synthetic benchmarks, public advertisement datasets, and largescale real-world recommendation systems consistently demonstrated that CAdam outperforms Adam and other well-established optimizers in both adaptability and robustness. Specifically, CAdam showed superior performance in scenarios with sudden and continuous distribution shifts, as well as in environments with significant noise, achieving higher accuracy and lower regret. Moreover, in live A/B testing within a production recommendation system, CAdam led to substantial improvements in model performance and gross merchandise volume (GMV), underscoring its practical effectiveness.

Future work may explore further refinements of the confidence assessment mechanism, its integration with other optimization frameworks, and its application to a broader range of machine learning models and real-time systems. Ultimately, CAdam represents a promising advancement in the development of more resilient and adaptive optimization algorithms for dynamic learning environments.

# REFERENCES


[1] Ahmet Alacaoglu, Yura Malitsky, Panayotis Mertikopoulos, and Volkan Cevher. A new regret analysis for adam-type algorithms. In *International conference on machine learning*, pp. 202–210. PMLR, 2020. Dosovitskiy Alexey. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv: 2010.11929*, 2020. Albert Bifet and Ricard Gavalda. Learning from time-changing data with adaptive windowing. In *Proceedings of the 2007 SIAM international conference on data mining*, pp. 443–448. SIAM, 2007. Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. Wide & deep learning for recommender systems. In *Proceedings of the 1st workshop on deep learning for recommender systems*, pp. 7–10, 2016. Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for youtube recommendations. In *Proceedings of the 10th ACM conference on recommender systems*, pp. 191–198, 2016. Alexandre Defossez, Leon Bottou, Francis Bach, and Nicolas Usunier. A simple convergence proof ´ of adam and adagrad. *Transactions on Machine Learning Research*. John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. *Journal of machine learning research*, 12(7), 2011.

[2] **554 555 556**

[3] **559**

[4] **561**

[5] **564**

[6] **569**

[7] **579**

[8] **584**

[9] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In *International conference on machine learning*, pp. 1126–1135. PMLR, 2017. Shintaro Fukushima, Atsushi Nitanda, and Kenji Yamanishi. Online robust and adaptive learning from data streams. *arXiv preprint arXiv:2007.12160*, 2020. Yingqiang Ge, Mostafa Rahmani, Athirai Irissappane, Jose Sepulveda, James Caverlee, and Fei Wang. Automated data denoising for recommendation. *arXiv preprint arXiv:2305.07070*, 2023. Aritra Ghosh, Himanshu Kumar, and P Shanti Sastry. Robust loss functions under label noise for deep neural networks. In *Proceedings of the AAAI conference on artificial intelligence*, volume 31, 2017. Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. Deepfm: a factorizationmachine based neural network for ctr prediction. *arXiv preprint arXiv:1703.04247*, 2017. Michael Gutmann and Aapo Hyvarinen. Noise-contrastive estimation: A new estimation principle ¨ for unnormalized statistical models. In *Proceedings of the thirteenth international conference on artificial intelligence and statistics*, pp. 297–304. JMLR Workshop and Conference Proceedings, 2010. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016. Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 4700–4708, 2017. Olivier Chapelle Jean-Baptiste Tien, joycenv. Display advertising challenge, 2014. URL [https:](https://kaggle.com/competitions/criteo-display-ad-challenge) [//kaggle.com/competitions/criteo-display-ad-challenge](https://kaggle.com/competitions/criteo-display-ad-challenge). Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In *International Conference on Learning Representations (ICLR)*, 2015. Hyeyoung Ko, Suyeon Lee, Yoonseo Park, and Anna Choi. A survey of recommendation systems: recommendation models, techniques, and application fields. *Electronics*, 11(1):141, 2022. Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Jiawei Han. On the variance of the adaptive learning rate and beyond. *arXiv preprint arXiv:1908.03265*, 2019. I Loshchilov. Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*, 2017. Jie Lu, Anjin Liu, Fan Dong, Feng Gu, Joao Gama, and Guangquan Zhang. Learning under concept drift: A review. *IEEE transactions on knowledge and data engineering*, 31(12):2346–2363, 2018. Costas Panagiotakis, Harris Papadakis, Antonis Papagrigoriou, and Paraskevi Fragopoulou. Dtec: Dual training error based correction approach for recommender systems. *Software Impacts*, 9: 100111, 2021. Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, highperformance deep learning library. *Advances in neural information processing systems*, 32, 2019. Ning Qian. On the momentum term in gradient descent learning algorithms. *Neural networks*, 12 (1):145–151, 1999. Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, and Jun Wang. Product-based neural networks for user response prediction. In *2016 IEEE 16th international conference on data mining (ICDM)*, pp. 1149–1154. IEEE, 2016.

[10] **604**

[11] **606**

[12] **614 615**

[13] **617**

[14] **619**

[15] **629**

[16] **634**

[17] **636**

[18] Sashank J Reddi, Satyen Kale, and Sanjiv Kumar. On the convergence of adam and beyond. In *International Conference on Learning Representations*, 2018. Roberto Saia, Ludovico Boratto, and Salvatore Carta. A semantic approach to remove incoherent items from a user profile and improve the accuracy of a recommender system. *Journal of Intelligent Information Systems*, 47:111–134, 2016. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017. Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*, 2014. Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. *The journal of machine learning research*, 15(1):1929–1958, 2014. W Nick Street and YongSeog Kim. A streaming ensemble algorithm (sea) for large-scale classification. In *Proceedings of the seventh ACM SIGKDD international conference on Knowledge discovery and data mining*, pp. 377–382, 2001. Rui Su, Husheng Guo, and Wenjian Wang. Elastic online deep learning for dynamic streaming data. *Information Sciences*, pp. 120799, 2024. A Vaswani. Attention is all you need. *Advances in Neural Information Processing Systems*, 2017. Antonio David Viniski, Jean Paul Barddal, Alceu de Souza Britto Jr, Fabr ˆ ´ıcio Enembreck, and Humberto Vinicius Aparecido de Campos. A case study of batch and incremental recommender systems in supermarket data under concept drifts and cold start. *Expert Systems with Applications*, 176:114890, 2021. Bohan Wang, Jingwen Fu, Huishuai Zhang, Nanning Zheng, and Wei Chen. Closing the gap between the upper bound and lower bound of adam's iteration complexity. *Advances in Neural Information Processing Systems*, 36, 2024. Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. Deep & cross network for ad click predictions. In *Proceedings of the ADKDD'17*, pp. 1–7. 2017. Wenjie Wang, Fuli Feng, Xiangnan He, Liqiang Nie, and Tat-Seng Chua. Learning robust recommender from noisy implicit feedback. *arXiv preprint arXiv:2112.01160*, 2021. Yuan-Yuan Xu, Shen-Ming Gu, and Fan Min. Improving recommendation quality through outlier removal. *International Journal of Machine Learning and Cybernetics*, 13(7):1819–1832, 2022. Yifan Yang, Alec Koppel, and Zheng Zhang. A gradient-based approach for online robust deep neural network training with noisy labels. *arXiv preprint arXiv:2306.05046*, 2023. Manzil Zaheer, Sashank Reddi, Devendra Sachan, Satyen Kale, and Sanjiv Kumar. Adaptive methods for nonconvex optimization. *Advances in neural information processing systems*, 31, 2018. Matthew D Zeiler. Adadelta: an adaptive learning rate method. *arXiv preprint arXiv:1212.5701*, 2012. Si-si Zhang, Jian-wei Liu, and Xin Zuo. Adaptive online incremental learning for evolving data streams. *Applied Soft Computing*, 105:107255, 2021. Yushun Zhang, Congliang Chen, Naichen Shi, Ruoyu Sun, and Zhi-Quan Luo. Adam can converge without any modification on update rules. *Advances in neural information processing systems*, 35:28386–28399, 2022. Jieming Zhu, Jinyang Liu, Shuai Yang, Qi Zhang, and Xiuqiang He. Open benchmarking for clickthrough rate prediction. In *Proceedings of the 30th ACM international conference on information & knowledge management*, pp. 2759–2769, 2021.

[19] Jieming Zhu, Quanyu Dai, Liangcai Su, Rong Ma, Jinyang Liu, Guohao Cai, Xi Xiao, and Rui Zhang. Bars: Towards open benchmarking for recommender systems. In *Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pp. 2912–2923, 2022.

[20] Juntang Zhuang, Tommy Tang, Yifan Ding, Sekhar C Tatikonda, Nicha Dvornek, Xenophon Papademetris, and James Duncan. Adabelief optimizer: Adapting stepsizes by the belief in observed gradients. *Advances in neural information processing systems*, 33:18795–18806, 2020.

[21] Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In *Proceedings of the 20th international conference on machine learning (icml-03)*, pp. 928–936, 2003.

[22] **704**

[23] **706**

[24] **709**

[25] **721**

[26] **724**

[27] **729 730**
### A PROOFS OF THEOREM [1](#page-3-1)

Given a stream of objectives f<sup>t</sup> : <sup>R</sup> <sup>d</sup> → <sup>R</sup>, t = 1, 2, . . . , T, online learning aims to minimize the regret w.r.t. the optimum; that is,

$$R_T := \sum_{t=1}^T f_t(x_t) - \sum_{t=1}^T f_t(x^*), \quad x^* = \operatorname{argmin}_x \sum_{t=1}^T f_t(x). \quad (3)$$

Recall that each update in CAdam can be characterized as follows[<sup>1</sup>](#page-13-0)

: m<sup>t</sup> = β1,tmt−<sup>1</sup> + (1 − β1,t)gt, (4)

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \quad (5)$$

$$m_{t,\Xi_t} = \begin{cases} m_{t,i}, & i \in \Xi_t \\ 0, & \text{else} \end{cases}, \quad (6)$$

$$\hat{v}_t = \max(\hat{v}_{t-1}, v_t), \quad (7)$$

$$x_{t+1} = x_t - \alpha_t m_{t,\Xi_t} / \hat{v}_t. \quad (8)$$

where Ξ<sup>t</sup> := {i ∈ [d] : mt,i · gt,i ≥ 0} indicates the set of active entries at step t. For notation clarity, let xt,<sup>Ξ</sup> be the vector of which the entries not belonging to Ξ are masked. Following the AMSGrad [\(Reddi et al., 2018\)](#page-11-7), we are to prove that the sequence of points obtained by CAdam satisfies R<sup>T</sup> /T → 0 as T increases.

We first introduce three standard assumptions:

Assumption 1. *Let* f<sup>t</sup> : <sup>R</sup> <sup>d</sup> → <sup>R</sup>, t = 1, 2, . . . , T *be a sequence of convex and differentiable functions with* ∥∇ft(x)∥<sup>∞</sup> ≤ G<sup>∞</sup> *for all* t ∈ [T]*.*

Assumption 2. *Let* {mt}, {vt} *be the sequences used in CAdam,* <sup>α</sup><sup>t</sup> <sup>=</sup> α/√ t, β1,t = β1λ <sup>t</sup>−<sup>1</sup> < 1, γ = β1/ √ β<sup>2</sup> < 1 *for all* t ∈ [T]*.*

Assumption 3. *The points involved are within a bounded diameter* D∞*; that is, for the optimal point* x <sup>∗</sup> *and any points* x<sup>t</sup> *generated by CAdam, it holds* ∥x<sup>t</sup> − x <sup>∗</sup>∥<sup>∞</sup> ≤ D∞/2*.*

We present several essential lemmas in the following. Given that some of these lemmas have been partially established in prior works [\(Kingma & Ba, 2015;](#page-10-1) [Reddi et al., 2018\)](#page-11-7), we include them here for the sake of completeness.

Lemma 1. *For a convex and differentiable function* f : R <sup>d</sup> → <sup>R</sup>*, we have*

$$f(x) - f(y) \leq \langle \nabla f(x), x - y \rangle. \quad (9)$$

Lemma 2. *Under Assumption [1](#page-13-1) and [2,](#page-13-2) we have*

$$\begin{aligned} \langle g_{t,\Xi_t}, x_{t,\Xi_t} - x_{\Xi_t}^* \rangle &\leq \frac{1}{2\alpha_t(1-\beta_{1,t})} \left( \|V_t^{1/4}(x_t, \Xi_t - x_{\Xi_t}^*)\|^2 - \|V_t^{1/4}(x_{t+1,\Xi_t} - x_{\Xi_t}^*)\|^2 \right) \\ &\quad + \frac{\beta_{1,t}}{2\alpha_t(1-\beta_{1,t})} \|V_t^{1/4}(x_t - x^*)\|^2 \\ &\quad + \frac{\alpha_t}{2(1-\beta_{1,t})} \|V_t^{-1/4} m_t\|^2 + \frac{\alpha_t \beta_{1,t}}{2(1-\beta_{1,t})} \|V_t^{-1/4} m_{t-1}\|^2, \end{aligned} \quad (10)$$

*where* V<sup>t</sup> := diag(ˆvt)*.*

*Proof.* CAdam updates the parameters as follows

$$x_{t+1,\Xi_t} = x_{t,\Xi_t} - \alpha_t m_{t,\Xi_t} / \sqrt{\hat{v}_t} = x_{t,\Xi_t} - \alpha_t V_t^{-1/2} \left( \beta_{1,t} m_{t-1,\Xi_t} + (1 - \beta_{1,t}) g_{t,\Xi_t} \right).$$

Subtracting x ∗ from both sides yields

$$\|V_t^{1/4}(x_{t+1,\Xi_t} - x_{\Xi_t}^*)\|_2^2$$

$$\begin{aligned}
& \|V_t^{1/4}(x_{t+1, \Xi_t} - x_{\Xi_t}^*)\|_2^2 \\
&= \|V_t^{1/4}(x_{t, \Xi_t} - x_{\Xi_t}^*) - \alpha_t V_t^{-1/4} m_{t, \Xi_t}\|_2^2 \\
&= \|V_t^{1/4}(x_{t, \Xi_t} - x_{\Xi_t}^*)\|_2^2 - 2\langle \alpha_t V_t^{-1/4} m_{t, \Xi_t}, V_t^{1/4}(x_{t, \Xi_t} - x_{\Xi_t}^*) \rangle + \|\alpha_t V_t^{-1/4} m_{t, \Xi_t}\|_2^2 \\
&= \|V_t^{1/4}(x_{t, \Xi_t} - x_{\Xi_t}^*)\|_2^2 - 2\alpha_t \langle \beta_{1, t} m_{t-1, \Xi_t} + (1 - \beta_{1, t}) g_{t, \Xi_t}, x_{t, \Xi_t} - x_{\Xi_t}^* \rangle + \|\alpha_t V_t^{-1/4} m_{t, \Xi_t}\|_2^2.
\end{aligned}$$

<sup>1</sup>Note that we omit the bias corrections for clarity purpose. It is not difficult to modify the proofs to obtain a more general one.

**759**

**761**

**764**

**766**

**769**

**779 780 781**

**784**

**804 805 806**

Rearranging the equation gives

$$\begin{aligned} \left\langle g_{t,\Xi_t}, x_{t,\Xi_t} - x_{\Xi_t}^* \right\rangle &= \frac{1}{2\alpha_t(1-\beta_{1,t})} \left( \left\| V_t^{1/4}(x_{t,\Xi_t} - x_{\Xi_t}^*) \right\|_2^2 - \left\| V_t^{1/4}(x_{t+1,\Xi_t} - x_{\Xi_t}^*) \right\|_2^2 \right) \\ &\quad - \frac{\beta_{1,t}}{1-\beta_{1,t}} \left\langle m_{t-1,\Xi_t}, x_{t,\Xi_t} - x_{\Xi_t}^* \right\rangle + \frac{\alpha_t}{2(1-\beta_{1,t})} \left\| V_t^{-1/4} m_{t,\Xi_t} \right\|_2^2. \end{aligned}$$

The results follow from the Cauchy-Schwarz inequality and Young's inequality:

$$\begin{aligned} -\frac{\beta_{1,t}}{1-\beta_{1,t}} \left\langle m_{t-1,\Xi_t}, x_{t,\Xi_t} - x_{\Xi_t}^* \right\rangle &= \frac{\beta_{1,t}}{1-\beta_{1,t}} \left\langle m_{t-1,\Xi_t}, x_{\Xi_t}^* - x_{t,\Xi_t} \right\rangle \\ &= \frac{\beta_{1,t}}{1-\beta_{1,t}} \left\langle \sqrt{\alpha_t} V_t^{-1/4} m_{t-1,\Xi_t}, \frac{1}{\sqrt{\alpha_t}} V_t^{1/4} (x_{\Xi_t}^* - x_{t,\Xi_t}) \right\rangle \\ &\leq \frac{\beta_{1,t}}{1-\beta_{1,t}} \left( \sqrt{\alpha_t} \|V_t^{-1/4} m_{t-1,\Xi_t}\| \cdot \frac{1}{\sqrt{\alpha_t}} \|V_t^{1/4} (x_{\Xi_t}^* - x_{t,\Xi_t})\| \right) \\ &\leq \frac{\beta_{1,t}}{1-\beta_{1,t}} \left( \frac{\alpha_t}{2} \|V_t^{-1/4} m_{t-1,\Xi_t}\|^2 + \frac{1}{2\alpha_t} \|V_t^{1/4} (x_{t,\Xi_t} - x_{\Xi_t}^*)\|^2 \right) \\ &\leq \frac{\beta_{1,t}}{1-\beta_{1,t}} \left( \frac{\alpha_t}{2} \|V_t^{-1/4} m_{t-1}\|^2 + \frac{1}{2\alpha_t} \|V_t^{1/4} (x_t - x^*)\|^2 \right), \end{aligned}$$

and the fact that ∥V −1/4 <sup>t</sup> mt,Ξ<sup>t</sup> 2 <sup>2</sup> ≤ ∥V −1/4 <sup>t</sup> mt∥ 2 2 .

Lemma 3. *Under Assumption [1,](#page-13-1) [2,](#page-13-2) and [3,](#page-13-3) we have*

$$\left\langle g_t, x_t - x^* \right\rangle \leq \left\langle g_{t,\Xi}, x_{t,\Xi} - x_{\Xi}^* \right\rangle + \frac{d\beta_1 \lambda^{t-1} D_{\infty} G_{\infty}}{1 - \beta_1}. \quad (11)$$

*Proof.* If the i-th entry is not updated at step t, i.e., i ∈ [d] \ Ξt, it can be derived that

$$\begin{aligned} & (\beta_{1,t}m_{t-1,i} + (1 - \beta_{1,t})g_{t,i}) \cdot g_{t,i} \leq 0 \\ & \Rightarrow (\beta_{1,t}m_{t-1,i} + (1 - \beta_{1,t})g_{t,i}) \cdot \text{sgn}(g_{t,i}) \leq 0 \\ & \Rightarrow -\beta_{1,t}|m_{t-1,i}| + (1 - \beta_{1,t})|g_{t,i}| \leq 0 \\ & \Rightarrow |g_{t,i}| \leq \frac{\beta_{1,t}}{1 - \beta_{1,t}} |m_{t-1,i}| \\ & \Rightarrow |g_{t,i}| \leq \frac{\beta_{1,t}}{1 - \beta_{1,t}} G_\infty \quad \leftarrow \text{Assumption 1} \\ & \Rightarrow |g_{t,i}| \leq \frac{\beta_1 \lambda^{t-1}}{1 - \beta_1} G_\infty, \quad i \in [d] \setminus \Xi_t. \quad \leftarrow \text{Assumption 2} \end{aligned}$$

With Assumption [3,](#page-13-3) it immediately yields the desired inequality that

$$\begin{aligned} \langle g_t, x_t - x^* \rangle &= \langle g_{t,\Xi}, x_{t,\Xi} - x_{\Xi}^* \rangle + \langle g_{t,[d] \setminus \Xi}, x_{t,[d] \setminus \Xi} - x_{[d] \setminus \Xi}^* \rangle \\ &\leq \langle g_{t,\Xi}, x_{t,\Xi} - x_{\Xi}^* \rangle + \sum_{i=1}^d \frac{\beta_1 \lambda^{t-1} D_{\infty} G_{\infty}}{1 - \beta_1}. \end{aligned}$$

Lemma 4. *Given Assumption [1,](#page-13-1) [2,](#page-13-2) and [3,](#page-13-3) we have*

$$\sum_{t \in [T]} \frac{\beta_{1,t}}{2\alpha_t(1 - \beta_{1,t})} \|V_t^{1/4}(x_t - x^*)\|^2 \leq \frac{dD_\infty^2 G_\infty}{2\alpha(1 - \beta_1)(1 - \lambda)^2}. \quad (12)$$

**814 815**

**817**

**819**

**829**

**834**

**836**

**854**

**856**

*Proof.*

$$\begin{aligned} & \sum_{t \in [T]} \frac{\beta_{1,t}}{2\alpha_t(1 - \beta_{1,t})} \|V_t^{1/4}(x_t - x^*)\|^2 \\ & \leq \frac{1}{2\alpha(1 - \beta_1)} \sum_{t \in [T]} \sqrt{t} \lambda^{t-1} \|V_t^{1/4}(x_t - x^*)\|^2 \\ & \leq \frac{G_\infty}{2\alpha(1 - \beta_1)} \sum_{t \in [T]} \sqrt{t} \lambda^{t-1} \|x_t - x^*\|^2 && \leftarrow \text{Assumption 1} \\ & \leq \frac{dD_\infty^2 G_\infty}{2\alpha(1 - \beta_1)} \sum_{t \in [T]} \sqrt{t} \lambda^{t-1} \\ & \leq \frac{dD_\infty^2 G_\infty}{2\alpha(1 - \beta_1)} \sum_{t \in [T]} \lambda^{t-1} t \\ & \leq \frac{dD_\infty^2 G_\infty}{2\alpha(1 - \beta_1)} \frac{1}{(1 - \lambda)^2}. \end{aligned}$$

Lemma 5 [\(Reddi et al.](#page-11-7) [\(2018\)](#page-11-7) Lemma2). *Under Assumption [2,](#page-13-2) we have*

$$\sum_{t \in [T]} \alpha_t \|V_t^{-1/4} m_t\|^2 \leq \frac{\alpha d G_\infty}{(1-\gamma)(1-\beta_1)\sqrt{1-\beta_2}} \sqrt{T}, \quad (13)$$

*where* γ := β1/ √ β2*.*

We are ready to prove the final results now. Concretely, Theorem [1](#page-3-1) is a straightfoward corollary of the following conclusion.

Theorem 2. *Under the Assumption [1,](#page-13-1) [2,](#page-13-2) and [3,](#page-13-3) the regret is converged with*

$$R_T \leq \frac{dD_\infty^2 G_\infty \sqrt{T}}{2\alpha(1 - \beta_1)} + \frac{d(2\alpha + D_\infty)D_\infty G_\infty}{2\alpha(1 - \beta_1)(1 - \lambda)^2} + \frac{\alpha dG_\infty \sqrt{T}}{(1 - \gamma)(1 - \beta_1)^2 \sqrt{1 - \beta_2}}. \quad (14)$$

*Proof.* Based on Lemma [1,](#page-13-4) Lemma [2,](#page-13-5) and Lemma [3,](#page-14-0) the regret can be firstly bounded by

$$\begin{aligned} R_T &= \sum_{t \in [T]} (f_t(x_t) - f_t(x^*)) \leq \sum_{t \in [T]} \langle g_t, x_t - x^* \rangle \\ &\leq \sum_{t \in [T]} \langle g_t, \Xi_t, x_t, \Xi_t - x_{\Xi_t}^* \rangle + \sum_{t \in [T]} \frac{d\beta_1 \lambda^{t-1} D_\infty G_\infty}{1 - \beta_1} \\ &\leq \underbrace{\sum_{t \in [T]} \frac{1}{2\alpha_t(1 - \beta_{1,t})} \left( \|V_t^{1/4}(x_t, \Xi_t - x_{\Xi_t}^*)\|^2 - \|V_t^{1/4}(x_{t+1, \Xi_t} - x_{\Xi_t}^*)\|^2 \right)}_{\textcircled{1}} \\ &\quad + \underbrace{\sum_{t \in [T]} \frac{\beta_{1,t}}{2\alpha_t(1 - \beta_{1,t})} \|V_t^{1/4}(x_t - x^*)\|^2}_{\textcircled{2}} + \underbrace{\sum_{t \in [T]} \frac{\alpha_t}{2(1 - \beta_{1,t})} \|V_t^{-1/4} m_t\|^2}_{\textcircled{3}} \\ &\quad + \underbrace{\sum_{t \in [T]} \frac{\alpha_t \beta_{1,t}}{2(1 - \beta_{1,t})} \|V_t^{-1/4} m_{t-1}\|^2}_{\textcircled{4}} + \underbrace{\sum_{t \in [T]} \frac{d\beta_1 \lambda^{t-1} D_\infty G_\infty}{1 - \beta_1}}_{\textcircled{5}}. \end{aligned}$$

Let us address each term in turn. For the first term, we are to separately bound each entry and the results follows from the summation. For the i-th entry, let T i <sup>+</sup> = [t : i ∈ Ξt] be a sequence collecting

**869**

**874**

**884**

**886**

**889 890 891**

**904**

**906**

**909**

all steps that x<sup>i</sup> is succesfully updated, and t˜<sup>k</sup> ∈ T <sup>i</sup> <sup>+</sup> be the k-th element of T i <sup>+</sup>. For simplicity, we will omit the superscript without ambiguity.

$$\begin{aligned} \textcircled{1}_i &= \sum_{t=\tilde{t}_1}^{\tilde{t}_{|\tau_{+|}}} \frac{1}{2\alpha_t(1-\beta_{1,t})} \left( (\hat{v}_{t,i}^{1/4}(x_{t,i} - x_i^*))^2 - (\hat{v}_{t,i}^{1/4}(x_{t+1,i} - x_i^*))^2 \right) \\ &\leq \frac{\hat{v}_{\tilde{t}_1,i}^{1/2}(x_{\tilde{t}_1,i} - x_i^*)^2}{2\alpha_{\tilde{t}_1}(1-\beta_1)} + \frac{1}{2} \sum_{t=\tilde{t}_2}^{\tilde{t}_{|\tau_{+|}}} \left[ \frac{\hat{v}_{t,i}^{1/2}(x_{t,i} - x_i^*)^2}{\alpha_t(1-\beta_{1,t})} - \frac{\hat{v}_{t-1,i}^{1/2}(x_{t,i} - x_i^*)^2}{\alpha_{t-1}(1-\beta_{1,t-1})} \right] \\ &= \frac{\hat{v}_{\tilde{t}_1,i}^{1/2}(x_{\tilde{t}_1,i} - x_i^*)^2}{2\alpha_{\tilde{t}_1}(1-\beta_1)} + \frac{1}{2} \sum_{t=\tilde{t}_2}^{\tilde{t}_{|\tau_{+|}}} \left[ \frac{\hat{v}_{t,i}^{1/2}(x_{t,i} - x_i^*)^2}{\alpha_t(1-\beta_{1,t-1})} - \underbrace{\frac{\hat{v}_{t,i}^{1/2}(x_{t,i} - x_i^*)^2}{\alpha_t(1-\beta_{1,t-1})}}_{\leq 0} + \underbrace{\frac{\hat{v}_{t,i}^{1/2}(x_{t,i} - x_i^*)^2}{\alpha_t(1-\beta_{1,t})}}_{\leq 0} \right. \\ &\quad \left. - \frac{\hat{v}_{t-1,i}^{1/2}(x_{t,i} - x_i^*)^2}{\alpha_{t-1}(1-\beta_{1,t-1})} \right] \\ &\leq \frac{\hat{v}_{\tilde{t}_1,i}^{1/2}(x_{\tilde{t}_1,i} - x_i^*)^2}{2\alpha_{\tilde{t}_1}(1-\beta_1)} + \frac{1}{2} \sum_{t=\tilde{t}_2}^{\tilde{t}_{|\tau_{+|}}} \underbrace{\frac{1}{1-\beta_{1,t-1}}}_{\leq 1/(1-\beta_1)} \left[ \underbrace{\frac{\hat{v}_{t,i}^{1/2}(x_{t,i} - x_i^*)^2}{\alpha_t}}_{\geq 0 \text{ by } \hat{v}_{t,i} \geq \hat{v}_{t-1,i}} - \frac{\hat{v}_{t-1,i}^{1/2}(x_{t,i} - x_i^*)^2}{\alpha_{t-1}} \right] \\ &\leq \frac{\hat{v}_{\tilde{t}_1,i}^{1/2}(x_{\tilde{t}_1,i} - x_i^*)^2}{2\alpha_{\tilde{t}_1}(1-\beta_1)} + \frac{D_\infty^2}{2(1-\beta_1)} \sum_{t=\tilde{t}_2}^{\tilde{t}_{|\tau_{+|}}} \left[ \frac{\hat{v}_{t,i}^{1/2}}{\alpha_t} - \frac{\hat{v}_{t-1,i}^{1/2}}{\alpha_{t-1}} \right] \\ &= \frac{\hat{v}_{\tilde{t}_1,i}^{1/2}(x_{\tilde{t}_1,i} - x_i^*)^2}{2\alpha_{\tilde{t}_1}(1-\beta_1)} + \frac{D_\infty^2}{2(1-\beta_1)} \left[ \frac{\hat{v}_{\tilde{t}_{|\tau_{+|},i}}^{1/2}}{\alpha_{\tilde{t}_{|\tau_{+|}}}} - \frac{\hat{v}_{\tilde{t}_1,i}^{1/2}}{\alpha_{\tilde{t}_1}} \right] \\ &\leq \frac{D_\infty^2}{2(1-\beta_1)} \frac{\hat{v}_{\tilde{t}_{|\tau_{+|},i}}^{1/2}}{\alpha_{\tilde{t}_{|\tau_{+|}}}} \leq \frac{D_\infty^2 G_\infty \sqrt{T}}{2\alpha(1-\beta_1)}. \end{aligned}$$

Hence,

$$\mathbb{1} = \sum_{i \in [d]} \mathbb{1}_i \leq \frac{dD_\infty^2 G_\infty \sqrt{T}}{2\alpha(1 - \beta_1)}. \quad (15)$$

$$\mathcal{Q} = \sum_{t \in [T]} \frac{\beta_{1,t}}{2\alpha_t(1 - \beta_{1,t})} \|V_t^{1/4}(x_t - x^*)\|^2 \leq \frac{dD_\infty^2 G_\infty}{2\alpha(1 - \beta_1)(1 - \lambda)^2} \leftarrow \text{Lemma 4.}$$

$$\begin{aligned} \textcircled{3} &= \sum_{t \in [T]} \frac{\alpha_t}{2(1 - \beta_{1,t})} \|V_t^{-1/4} m_t\|^2 \leq \frac{1}{2(1 - \beta_1)} \sum_{t \in [T]} \alpha_t \|V_t^{-1/4} m_t\|^2 \\ &\leq \frac{\alpha d G_\infty \sqrt{T}}{2(1 - \gamma)(1 - \beta_1)^2 \sqrt{1 - \beta_2}}. \qquad \leftarrow \text{Lemma 5} \end{aligned}$$

$$\begin{aligned} \textcircled{4} &= \sum_{t \in [T]} \frac{\alpha_t \beta_{1,t}}{2(1 - \beta_{1,t})} \|V_t^{-1/4} m_{t-1}\|^2 \leq \frac{1}{2(1 - \beta_1)} \sum_{t \in [T]} \alpha_t \|V_{t-1}^{-1/4} m_{t-1}\|^2 \\ &\leq \frac{1}{2(1 - \beta_1)} \sum_{t \in [T]} \alpha_{t-1} \|V_{t-1}^{-1/4} m_{t-1}\|^2 = \frac{1}{2(1 - \beta_1)} \sum_{t \in [T-1]} \alpha_t \|V_t^{-1/4} m_t\|^2 \\ &\leq \frac{\alpha d G_\infty \sqrt{T}}{2(1 - \gamma)(1 - \beta_1)^2 \sqrt{1 - \beta_2}}. \quad \leftarrow \text{Lemma 5} \end{aligned}$$

**924**

**929**

**954**

**956**

**959**

**961**

$$\mathbb{S} = \sum_{t \in [T]} \frac{d\beta_1 \lambda^{t-1} D_\infty G_\infty}{1 - \beta_1} = \frac{d\beta_1 D_\infty G_\infty}{1 - \beta_1} \sum_{t \in [T]} \lambda^{t-1} \leq \frac{dD_\infty G_\infty}{(1 - \beta_1)(1 - \lambda)^2}.$$

Finally, we have

$$R_T \leq \frac{dD_\infty^2 G_\infty \sqrt{T}}{2\alpha(1 - \beta_1)} + \frac{d(2\alpha + D_\infty)D_\infty G_\infty}{2\alpha(1 - \beta_1)(1 - \lambda)^2} + \frac{\alpha dG_\infty \sqrt{T}}{(1 - \gamma)(1 - \beta_1)^2 \sqrt{1 - \beta_2}}.$$

### B HYPERPARAMETERS

#### B.1 NUMERICAL EXPERIMENT

Distribution Shift For the distribution shift experiments, we used the following hyperparameters: a cycle length of 40, a learning rate α = 0.5, exponential decay rates for the first and second moment estimates β<sup>1</sup> = 0.9 and β<sup>2</sup> = 0.999 respectively, and a small constant ϵ = 1 × 10−<sup>8</sup> to prevent division by zero. The number of time steps was set to T = 100.

Noisy Samples For the noisy samples experiments, the hyperparameters were set as follows: a learning rate of 0.1, β<sup>1</sup> = 0.9, β<sup>2</sup> = 0.999, ϵ = 1 × 10−<sup>8</sup> , and a maximum number of iterations T = 1500.

### B.2 CNN ON IMAGE CLASSIFICATION

For the CNN-based image classification experiments on the CIFAR-10 dataset, we used a learning rate of 3 × 10−<sup>4</sup> , β<sup>1</sup> = 0.9, β<sup>2</sup> = 0.999, weight decay of 0.0005, and ϵ = 1 × 10−<sup>8</sup> .

#### B.3 PUBLIC ADVERTISEMENT DATASET

Due to resource limitations, we performed a grid search over the learning rates for each optimizer and model using the following range: {lr default/5, lr default/2, lr default, 2×lr default, 5× lr default}, where lr default is the default learning rate specified in the FuxiCTR library. We reported the best performance for each optimizer based on this search. All other hyperparameters were kept the same as those in the FuxiCTR library [\(Zhu et al., 2021;](#page-11-16) [2022\)](#page-12-2).

### C ADDITIONAL EXPERIMENTS

#### C.1 NUMERICAL EXPERIMENTS

Figure [5](#page-18-0) illustrate how both optimizers perform in a noise-free environment.

### C.2 EXPERIMENT ON RESNET AND DENSENET

We perform experiments on Resnet[\(He et al., 2016\)](#page-10-17) and Densenet[\(Huang et al., 2017\)](#page-10-18) to further illustrate the effectiveness of CAdam.

### C.3 RELATIONSHIP BETWEEN LEARNING RATE, PERFORMANCE, AND ALIGNMENT RATIO

We tested different learning rates on the Criteo x4 001 dataset using the DeepFM model to understand the relationship between the learning rate, performance, and alignment ratio. The results in [4](#page-20-0) show that the performance initially increases with the learning rate but starts to decline as the learning rate continues to rise. Conversely, the consistent ratio R steadily decreases as the learning rate increases.

![](_page_18_Figure_2.jpeg)

Figure 5: Performance of Adam (top row) and CAdam (bottom row) on four different optimization landscapes without noise: (Left to Right) separable L1 loss, inseparable L1 loss, inseparable L2 loss, and Rosenbrock function. This comparison highlights the natural behavior of both optimizers in a noise-free environment.

![](_page_18_Figure_4.jpeg)

Figure 6: Performance of CAdam and Adam under different rotation speeds corresponding to sudden distribution shift. The results for Resnet are shown on the left, while those for Densenet are presented on the right.

![](_page_19_Figure_1.jpeg)

![](_page_19_Figure_3.jpeg)

Figure 7: Performance of CAdam and Adam under different rotation speeds corresponding to continuous distribution shift. The results for Resnet are shown on the left, while those for Densenet are presented on the right.

Figure 8: Performance of CAdam and Adam under noisy data. The results for Resnet are shown on the left, while those for Densenet are presented on the right.

| Learning | Rate AUC | Alignment Ratio ( R ) |
|----------|----------|-----------------------|
| 0.0001   | 80.59%   | 63.02%                |
| 0.0003   | 80.77%   | 59.17%                |
| 0.0005   | 80.80%   | 55.78%                |
| 0.001    | 80.83%   | 46.45%                |
| 0.0015   | 80.83%   | 42.01%                |
| 0.002    | 80.75%   | 42.53%                |
| 0.0025   | 80.66%   | 41.28%                |
| 0.003    | 80.55%   | 37.97%                |
| 0.0035   | 80.46%   | 32.09%                |
| 0.004    | 80.28%   | 32.06%                |

Table 4: Performance Metrics and Alignment Ratio for Different Learning Rates.