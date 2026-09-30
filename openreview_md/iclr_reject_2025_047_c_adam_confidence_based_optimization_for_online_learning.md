000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Modern recommendation systems frequently employ online learning to dynamically update their models with freshly collected data. The most commonly used optimizer for updating neural networks in these contexts is the Adam optimizer, which integrates momentum (mt) and adaptive learning rate (vt). However, the volatile nature of online learning data, characterized by its frequent distribution shifts and presence of noises, poses significant challenges to Adam's standard optimization process: (1) Adam may use outdated momentum and the average of squared gradients, resulting in slower adaptation to distribution changes, and (2) Adam's performance is adversely affected by data noise. To mitigate these issues, we introduce CAdam, a confidence-based optimization strategy that assesses the consistence between the momentum and the gradient for each parameter dimension before deciding on updates. If momentum and gradient are in sync, CAdam proceeds with parameter updates according to Adam's original formulation; if not, it temporarily withholds updates and monitors potential shifts in data distribution in subsequent iterations. This method allows CAdam to distinguish between the true distributional shifts and mere noise, and adapt more quickly to new data distributions. Our experiments with both synthetic and real-world datasets demonstrate that CAdam surpasses other well-known optimizers, including the original Adam, in efficiency and noise robustness. Furthermore, in large-scale A/B testing within a live recommendation system, CAdam significantly enhances model performance compared to Adam, leading to substantial increases in the system's gross merchandise volume (GMV).

## 1 Introduction

Modern recommendation systems, such as those used in online advertising platforms, rely on online learning to update real-time models with freshly collected data batches (Ko et al., 2022). In online learning, models continuously adapt to users' interests and preferences based on immediate user interactions like clicks or conversions. Unlike traditional offline training—where data is pre-collected and static—online learning deals with streaming data that is often noisy and subject to frequent distribution changes. This streaming nature makes it challenging to effectively denoise and reorganize training samples (Su et al., 2024; Zhang et al., 2021). A widely adopted optimizer in these systems is the Adam optimizer (Kingma & Ba, 2015), which combines the strengths of parameter-adaptive methods and momentum-based methods. Adam adjusts learning rates based on the averaged gradient square norm (vt) and incorporates momentum (mt) for faster convergence. Its ability to maintain stable and efficient convergence by dynamically adjusting learning rates based on the first and second moments of gradients has made it a reliable choice for optimizing deep learning models across diverse applications, including image recognition (Alexey, 2020), natural language processing (Vaswani, 2017), and reinforcement learning (Schulman et al., 2017). However, Adam faces significant challenges in online learning environments. Specifically, it treats all incoming data equally, regardless of whether it originates from the original distribution, a new one, or is merely noise. This indiscriminate treatment leads to two key problems:
1. **Outdated Momentum and Averaged Squared Gradients**: When the data distribution shifts—a common occurrence in online systems due to factors such as daily cycles in shopping habits, rapidly changing trends on social media, seasonal changes, promotional events, Anonymous authors Paper under double-blind review

## Abstract

# Cadam: Confidence-Based Optimization For Online Learning

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 and sudden market dynamics—Adam continues to use momentum and averaged squared gradients computed from previous data (Lu et al., 2018; Viniski et al., 2021). These outdated statistics can misguide the optimizer, resulting in slower adaptation to the new data distributions.

2. **Sensitivity to Noise**: Online learning data often contains noisy labels (Yang et al., 2023).

For example, in advertisement systems, users might click ads by mistake (false positives) or ignore ads they are interested in (false negatives) (Wang et al., 2021). Sensitivity to such noise can affect convergence speed and may cause parameters to deviate from the correct optimization direction, especially in scenarios where noisy data constitutes a large proportion.

To address these issues inherent in online learning with Adam, we propose Confidence Adaptive Moment Estimation (CAdam), a novel optimization strategy that enhances Adam's robustness and adaptability. CAdam introduces a confidence metric that evaluates whether updating a specific parameter will be beneficial for the system. This metric is calculated by assessing the alignment between the current momentum and the gradient for each parameter dimension. Specifically, if the momentum and the gradient point in the same direction, indicating consistency in the optimization path, CAdam proceeds with the parameter update following Adam's rule. Otherwise, if they point in opposite directions, CAdam pauses the update for that parameter to observe potential distribution changes in subsequent iterations. This strategy hinges on the idea that persistent opposite gradients suggest a distributional shift, as the momentum (an exponential moving average of past gradients) represents the recent trend. If the opposite gradients do not persist, it it likely to be noise, and the model resumes normal updates, effectively filtering out the noise. By incorporating this simple, plug-and-play mechanism, CAdam retains the advantages of momentum-based optimization while enhancing robustness to noise and improving adaptability to meaningful distribution changes in online learning scenarios. Our contribution can be summarized as follows:
1. We introduce CAdam, a confidence-based optimization algorithm that improves upon the standard Adam optimizer by addressing its limitations in handling noisy data and adapting to distribution shifts in real-time online learning.

2. Through extensive experiments on both synthetic and public datasets, we demonstrate that CAdam consistently outperforms popular optimizers in online recommendation settings.

3. We validate the real-world applicability of CAdam by conducting large-scale online A/B
tests in a live system, proving its effectiveness in boosting system performance and achieving significant improvements in gross merchandise volume (GMV) worth millions of dollars.

## 2 Related Work

Adam Extensions Adam is one of the most widely used optimizers, and researchers have proposed various modifications to address its limitations. AMSGrad (Reddi et al., 2018) addresses Adam's non-convergence issue by introducing a maximum operation in the denominator of the update rule. RAdam (Liu et al., 2019) incorporates a rectification term to reduce the variance caused by adaptive learning rates in the early stages of training, effectively combining the benefits of both adaptive and non-adaptive methods. AdamW (Loshchilov, 2017) separates weight decay from the gradient update, improving regularization. Yogi (Zaheer et al., 2018) modifies the learning rate using a different update rule for the second moment to enhance stability. AdaBelief (Zhuang et al., 2020) refines the second-moment estimation by focusing on the deviation of the gradient from its exponential moving average rather than the squared gradient. This allows the step size to adapt based on the "belief" in the current gradient direction, resulting in faster convergence and improved generalization. Our method, CAdam, similarly leverages the consistency between the gradient and momentum for adjustments. However, it preserves the original update structure of Adam and considers the sign (directional consistency) between momentum and gradient, rather than their value deviation, leading to better performance under distribution shifts and in noisy environments. Adapting to Distributional Changes in Online Learning In online learning scenarios, models encounter data streams where the underlying distribution can shift over time, a phenomenon known as concept drift (Lu et al., 2018). Adapting to these changes is essential for maintaining model performance. One common strategy is to use sliding windows or forgetting mechanisms (Bifet & Gavalda, 2007), which focus updates on the most recent data. Ensemble methods (Street & Kim, 2001) maintains a collection of models trained on different time segments and combine their predictions to adapt to emerging patterns. Adaptive learning algorithms, such as Online Gradient Descent (Zinkevich, 2003), dynamically adjust the learning rate or model parameters based on environmental feedback. Meta-learning approaches (Finn et al., 2017) aim to develop models that can quickly adapt to new tasks or distributions with minimal updates. Additionally, (Viniski et al., 2021) demonstrated that streaming-based recommender systems outperform batch methods in supermarket data, particularly in handling concept drifts and cold start scenarios.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Robustness to Noisy Data General methods for noise robustness include robust loss functions (Ghosh et al., 2017), which modify the objective function to reduce sensitivity to mislabeled or corrupted data; regularization techniques (Srivastava et al., 2014), which prevent overfitting by introducing noise during training; and noise-aware algorithms (Gutmann & Hyvarinen, 2010), which ¨
explicitly model noise distributions to improve learning. In recommendation systems, enhancing robustness against noisy data is crucial and is typically addressed through two main strategies: detect and correct and detect and remove. *Detect and correct* methods, such as AutoDenoise (Ge et al., 2023) and Dual Training Error-based Correction (DTEC) (Panagiotakis et al., 2021), identify noisy inputs and adjust them to improve model accuracy by leveraging mechanisms like validation sets or dual error perspectives. Conversely, *detect and remove* approaches eliminate unreliable data using techniques such as outlier detection with statistical models (Xu et al., 2022) or semantic coherence assessments (Saia et al., 2016) to cleanse user profiles. While these strategies can effectively enhance recommendation quality, they often require explicit design and customization for specific models or tasks, limiting their general applicability.

## 3 Details Of Cadam Optimizer

Notations We use the following notations for the CAdam optimizer:
- f(θ) ∈ R, θ ∈ R
d: f is the stochastic objective function to minimize, where θ is the parameter vector in R
d.

- gt: the gradient at step t, gt = ∇θft(θt−1). - mt: exponential moving average (EMA) of gt, calculated as mt = β1 ·mt−1+ (1−β1)·gt. - vt: EMA of the squared gradients, given by vt = β2 · vt−1 + (1 − β2) · g 2 t.

- mˆ t, vˆt: bias-corrected estimates of mt and vt, respectively, where mˆ t =mt 1−β t 1 and vˆt =
vt 1−β t 2
.

- α, ϵ: α is the learning rate, typically set to 10−3, and ϵ is a small constant to prevent division by zero, typically set to 10−8.

- β1, β2: smoothing parameters, commonly set as β1 = 0.9, β2 = 0.999. - θt: the parameter vector at step t. - θ0: the initial parameter vector.

Comparison with Adam CAdam (Algorithm 1) and Adam both use the first and second moments of gradients to adapt learning rates. The main difference between CAdam and Adam is that CAdam introduces the alignment between the momentum and the gradient as a confidence metric to address two common problems in real-world online learning: distribution shifts and noise.

In Adam, the update direction is determined by mt, the exponential moving average (EMA) of the gradient gt, and vt, the EMA of the squared gradients g 2 t. This method assumes a relatively stable data distribution, where mt serves as a good estimator of the optimal update direction. However, if the data distribution changes, mt may no longer point in the correct direction. Adam will continue to update using the outdated mt for several iterations until it eventually aligns with the new gradient direction, leading to poor performance during this adaptation period. Additionally, when encountering noisy examples, Adam blindly updates using mt, which can be problematic as it equivalently increases the learning rate, especially when the proportion of noisy data is high.

In contrast, CAdam dynamically checks the *alignment* between the current gradient gt and the momentum mt before proceeding with an update. If gt and mt point in the same direction, indicating that the momentum aligns with the current gradient, CAdam performs the update using mt/
√vt.

However, if gt and mt point in opposite directions, CAdam **pauses** the update for that parameter to observe subsequent gradients. This pause allows CAdam to distinguish between a potential distribution shift and noise.

If the reverse gradient signs persist in subsequent steps, it signals a distribution shift, and mt will gradually change direction to reflect the new data pattern, while CAdam doesn't update in these iterations, avoiding incorrect updates. Conversely, if the gradient signs realign in the following steps, it indicates that the previous opposite gradient was caused by noise. In this case, CAdam resumes normal updates, effectively filtering out noisy gradients without making unnecessary updates in the process. In addition, CAdam also has an AMSGrad (Reddi et al., 2018) variant as described in 1 when AMSGrad option is enabled.

Convergence Analysis Given a stream of functions ft : R
d → R, t = 1, 2*, . . . , T*, an online learning algorithm chooses θt in each time step t and aims to minimize the T-step regret w.r.t. the optimum, where the regret is defined as

$$R_{T}:=\sum_{t=1}^{T}f_{t}(\theta_{t})-\sum_{t=1}^{T}f_{t}(\theta^{*}),\quad\theta^{*}=\operatorname*{argmin}_{\theta}\sum_{t=1}^{T}f_{t}(\theta).$$
$$(1)$$
$$(2)$$
ft(θ). (1)
162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Algorithm 1 Confidence Adaptive Moment Estimation (CAdam)
1: m0 ← 0, v0 ← 0, vˆmax,0 ← 0, t ← 0, θt = θ0 2: **while** θt not converged do 3: t ← t + 1 4: gt ← ∇θft(θt−1) 5: mt ← β1 · mt−1 + (1 − β1) · gt 6: vt ← β2 · vt−1 + (1 − β2) · g 2 t 7: mˆ t ← mt/(1 − β t1)
8: vˆt ← vt/(1 − β t2)
9: if AMSGrad **then**
10: vˆmax,t ← max(ˆvmax,t−1, vˆt)
11: **else**
12: vˆmax,t ← vˆt 13: **end if** 14: mˆ t ← max(0, mt · sign(gt)) ▷ Element-wise mask out elements where mt · gt ≤ 0 15: θt ← θt−1 − α · mˆ t/(pvˆmax,t + ϵ)
16: **end while**
17: **return** θt Remark: We follow the regret analysis in Reddi et al. (2018) and adopt the same set of assumptions. In particular, Reddi et al. (2018) only considered convex functions and made bounded gradient assumption. Recently, there is a body of work that has provided refined convergence analysis under

$$R_{T}={\mathcal{O}}({\sqrt{T}}).$$
√T). (2)
The online learning setting has been widely used to model real-world recommendation scenarios. We show that CAdam has the same O(
√T) regret as Adam/AMSGrad under the same assumptions made in Reddi et al. (2018). The detailed proofs can be found in the appendix. Theorem 1 (Informal). Under the assumptions introduced in Reddi et al. (2018), the CAdam algorithm (with AMSGrad correction) achieves a sublinear regret; that is, 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 nonconvex setting and much weaker assumptions (see e.g., Alacaoglu et al. (2020); Defossez et al.; ´ Zhang et al. (2022); Wang et al. (2024)). We leave the analysis of C-Adam under these more general settings as an interesting future direction.

## 4 Experiment

In this section, we systematically evaluate the performance of CAdam across various scenarios, starting with synthetic image data, followed by tests on a public advertisement dataset, and concluding with A/B tests in a real-world recommendation system. We first examine CAdam's behaviour under distribution shift, and noisy conditions using the CIFAR-10 dataset(Krizhevsky et al., 2009) with the VGG network(Simonyan & Zisserman, 2014). Next, we test CAdam against other popular optimizers on the Criteo dataset(Jean-Baptiste Tien, 2014), focusing on different models and scenarios. Finally, we conduct A/B tests with millions of users in a real-world recommendation system to validate CAdam's effectiveness in large-scale, production-level environments. The results demonstrate that CAdam consistently outperforms Adam and other optimizers across different tasks, distribution shifts, and noise conditions.

$$L(x,t)={\begin{cases}|x-x^{*}(t)|,&\text{L1loss,}\\ (x-x^{*}(t))^{2},&\text{L2loss,}\end{cases}}$$

5 Distribution Change To illustrate the different behaviours of Adam and CAdam under distribution shifts, we designed three types of distribution changes for both L1 and L2 loss functions: (1)
Sudden change, where the minimum shifts abruptly at regular intervals; (2) *Linear* change, where the minimum moves at a constant speed; and (3) *Sinusoidal* change, where the minimum oscillates following a sine function, resulting in variable speed over time.

The loss functions are defined as:

## 4.1 Numerical Experiment

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 Figure 2: Trajectory of Adam (top row) and CAdam (bottom row) under noisy conditions on four different optimization landscapes: (Left to Right) separable L1 loss, inseparable L1 loss, inseparable L2 loss, and Rosenbrock function. Each column shows the optimization trajectory in the presence of noise, where each dimension's gradient is randomly flipped with a 50% probability. CAdam demonstrates superior robustness, maintaining more stable convergence paths than Adam across all tested functions. where x
∗(t) represents the position of the minimum at time t and is defined based on the type of distribution change:

$$x^{*}(t)=\begin{cases}\left\lfloor{\frac{t}{T}}\right\rfloor&\text{mod}2,\quad\text{{\it sudden change}},\\ \frac{t}{T},&\text{{\it linear change}},\\ \sin\left({\frac{2\pi t}{T}}\right),&\text{{\it sinusoidal change}}.\end{cases}$$

The results of these experiments are presented in Figure 1. Across different loss functions and distribution changes, CAdam closely follows the trajectory of the minimum point, being less affected by incorrect momentum, exhibiting lower regret and demonstrating its superior ability to adapt to shifting distributions. Noisy Samples To compare Adam and CAdam in noisy environments, we conducted experiments on four different optimization 2-d landscapes: (1) separable L1 loss, (2) inseparable L1 loss, (3) inseparable L2 loss, and (4) Rosenbrock function. These landscapes are defined as follows:
1. Separable L1 Loss: f1(*x, y*) = |x| + |y|. 2. Inseparable L1 Loss: f2(*x, y*) = |x + y| +
|x−y| 10 .

3. Inseparable L2 Loss: f3(x, y*) = (*x + y)
2 +
(x−y)
2 10 .

4. Rosenbrock Function: f4(*x, y*) = (a − x)
2 + b(y − x 2)
2, where a = 1 and b = 100.

The results of these experiments are shown in Figure 2. For comparison, the results without noise are provided in Figure 5 in the appendix. The trajectory of CAdam exhibits fewer random perturbations and lower regret, indicating its ability to resist noise interference. To simulate noise in the gradients, we applied a random mask to each dimension of the gradient with a 50% probability using the same random seed across different optimizers. Specifically, the gradient components were multiplied by a uniformly distributed random value from the range [−1, 1] to introduce noise:

$$\nabla_{\mathrm{noisy}}(x,y)=\begin{cases}\nabla f(x,y)\cdot U(-1,1),&\text{with probability}p=0.5,\\ \nabla f(x,y),&\text{otherwise,}\end{cases}$$

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 Figure 3: **(Left)** Performance of CAdam and Adam under different rotation speeds corresponding to sudden distribution shift. CAdam demonstrates superior performance, with a more pronounced advantage over Adam in the presence of rotation. **(Right)** A detailed view at a 60-degree rotation between steps 1400 to 2300, showing the Alignment Ratio, Accuracy, and Loss. The red dashed lines indicate the rotation points, where the alignment ratio decreases, resulting in fewer parameter updates. This is followed by a gradual recovery in both the alignment ratio and accuracy, and a decline in loss. CAdam's accuracy drop is slower, and its recovery is faster than Adam's, illustrating its enhanced ability to adapt to distribution shifts.

## 4.2 Cnn On Image Classification

We perform experiments using the VGG network on the CIFAR-10 dataset to evaluate the effectiveness of CAdam in handling distribution shifts and noise. We synthesize three experimental conditions: (1) sudden distribution changes, (2) continuous distribution shifts, and (3) added noise to the samples. The hyperparameters for these experiments are provided in Section B.2. Sudden Distribution Shift To simulate sudden changes in data distribution, we rotate the images by a specific angle at the start of each epoch, relative to the previous epoch, as illustrated in Figure 3. CAdam consistently outperforms Adam across varying rotation speeds, with a more significant performance gap compared to the non-rotated condition. We define the *alignment ratio* as:
Alignment Ratio =

## Number Of Parameters Where Mt · Gt > 0 Total Number Of Parameters

A closer inspection in Figure 3 reveals that, during the rotation (indicated by the red dashed line), the alignment ratio decreases, resulting in fewer parameters being updated, followed by a gradual recovery. Correspondingly, the accuracy declines and subsequently improves, while the loss increases before decreasing. Notably, during these shifts, CAdam's accuracy drops more slowly and recovers faster than Adam's, indicating its superior adaptability to new data distributions. Continuous Distribution Shifts In contrast to sudden distribution changes, we also tested the scenario where the data distribution changes continuously. Specifically, we simulated this by rotating the data distribution at each iteration by an angle. The results, shown in Figure 4, indicate that as the rotation speed increases, the advantage of CAdam over Adam becomes more pronounced.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Noisy Samples To evaluate the optimizer's robustness to noise, we introduced noise into the dataset by randomly selecting a certain number of batches in each epoch (resampling for each epoch) and replacing the labels of these batches with random values. The results are presented in Figure 4. We observed that as the proportion of noisy labels increases, the consistency of CAdam decreases, causing it to update fewer parameters in each iteration. Despite this, both CAdam and Adam experience a performance decline in test set accuracy as noise increases. Nevertheless, CAdam consistently outperforms Adam, maintaining accuracy even with 40% noise, comparable to Adam's performance in a noise-free setting by the end of training.

## 4.3 Public Advertisement Dataset

Experiment Setting To evaluate the effectiveness of the proposed CAdam optimizer, we conducted experiments using various models on the Criteo-x4-001 dataset(Jean-Baptiste Tien, 2014). This dataset contains feature values and click feedback for millions of display ads and is commonly used to benchmark algorithms for click-through rate (CTR) prediction(Zhu et al., 2021). To simulate a real-world online learning scenario, we trained the models on data up to each timestamp in a single epoch(Fukushima et al., 2020). This setup replicates the environment where new data arrives continuously, requiring the model to adapt quickly. Furthermore, for sparse parameters (e.g., embeddings), we update the optimizer's state only when there is a non-zero gradient for this parameter in the current batch using SparseAdam implementation in Pytorch(Paszke et al., 2019). This approach ensures that the optimizer's state reflects the parameters influenced by recent data changes. The hyperparameters are provided in Appendix B.3.

We benchmarked CAdam and other popular optimizers, including SGD, SGDM(Qian, 1999),
AdaGrad(Duchi et al., 2011), AdaDelta(Zeiler, 2012), RMSProp, Adam(Kingma & Ba, 2015), AMSGrad(Reddi et al., 2018), and AdaBelief(Zhuang et al., 2020), on various models such as DeepFM(77M)(Guo et al., 2017), WideDeep(77M)(Cheng et al., 2016), DNN(74M)(Covington et al., 2016), PNN(79M)(Qu et al., 2016), and DCN(74M)(Wang et al., 2017). The performance of these optimizers was evaluated using the Area Under the Curve (AUC) metric.

| denotes the AMSGrad variant of CAdam, which achieves the highest average performance. DeepFM WideDeep DNN PNN DCN Avg SGD 71.90±.006 71.88±.013 68.12±.043 67.61±.318 69.55±.026 69.81 SGDM 76.59±.044 76.32±.021 78.80±.014 76.17±.050 77.90±.018 77.16 AdaGrad 71.77±.032 71.50±.011 68.65±.022 67.49±.027 69.55±.020 69.79 AdaDelta 71.91±.071 71.64±.005 69.76±.004 67.59±.025 69.76±.024 70.13 RMSProp 71.82±.010 71.54±.021 68.72±.005 67.51±.004 69.60±.007 69.84 Adam 80.87±.011 80.90±.004 80.89±.003 80.90±.006 81.05±.005 80.92 AdaBelief 80.84±.008 80.90±.002 80.88±.011 80.89±.002 81.02±.044 80.91 AdamW 80.87±.008 80.90±.010 80.88±.010 80.90±.002 81.00±.047 80.91 AmsGrad 80.88±.004 80.92±.008 80.91±.001 80.92±.009 81.08±.009 80.94 CAdam 80.88±.008 80.93±.004 80.90±.002 80.93±.006 81.06±.009 80.94 CAmsGrad 80.90±.006 80.93±.007 80.92±.005 80.94±.009 81.09±.010 80.96   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Main Results The results in Table 1 show that CAdam and its AMSGrad variants outperform other optimizers across different models. While the AMSGrad variants perform better on certain datasets, they do not consistently outperform standard CAdam. Both versions of CAdam generally achieve higher AUC scores than other optimizers, demonstrating their effectiveness in the online learning setting. Robustness under Noise To simulate a noisier environment, we introduced noise into the Criteo x4-001 dataset by flipping 1% of the negative training samples to positive. All other settings remained unchanged. The results in Table 2 show that CAdam consistently outperforms Adam in terms of both AUC and the extent of performance drop. This demonstrates CAdam's robustness in handling noisy data.

| CAdam shows a smaller performance drop, highlighting its robustness to noise. DeepFM WideDeep DNN PNN   | DCN        |            |            |            |            |
|---------------------------------------------------------------------------------------------------------|------------|------------|------------|------------|------------|
| Adam                                                                                                    | 80.51±.008 | 80.47±.006 | 80.48±.014 | 80.66±.006 | 80.51±.010 |
| CAdam                                                                                                   | 80.81±.007 | 80.79±.006 | 80.78±.005 | 80.96±.026 | 80.77±.007 |
| Adam Drop                                                                                               | −0.36±.014 | −0.43±.007 | −0.41±.016 | −0.23±.012 | −0.54±.013 |
| CAdam Drop                                                                                              | −0.08±.014 | −0.14±.009 | −0.12±.004 | +0.04±.031 | −0.28±.015 |

## 4.4 Experiment On Real-World Recommendation System

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 In real-world recommendation scenarios, the differences from the Criteo dataset experiments are quite significant. First, both data volume and model sizes are much larger, with models used in the following experiments ranging from 8.3 billion to 330 billion parameters—100 to 10,000 times larger. Second, as these are online experiments, unlike offline experiments with a fixed dataset, the model's output directly influences user behaviour. To test the effectiveness of CAdam in this setting, we conducted A/B tests on internal models serving millions of users across seven different scenarios (2 pre-ranking, 4 recall, and 1 ranking). During these online experiments, we used a batch size of B = 4096 The evaluation metric was the Generalized Area Under the Curve (GAUC). Due to limited resources, we compared only Adam and CAdam, running the experiments for 48 hours. The results, shown in Table 3, indicate that CAdam consistently outperformed Adam across all test scenarios, demonstrating its superiority in real-world applications.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

## 5 Conclusion

In this paper, we addressed the inherent limitations of the Adam optimizer in online learning environments, particularly its sluggish adaptation to distributional shifts and heightened sensitivity to noisy data. To overcome these challenges, we introduced CAdam (Confidence Adaptive Moment Estimation), a novel optimization strategy that enhances Adam by incorporating a confidence-based mechanism. This mechanism evaluates the alignment between momentum and gradients for each parameter dimension, ensuring that updates are performed judiciously. When momentum and gradients are aligned, CAdam updates the parameters following Adam's original formulation; otherwise, it temporarily withholds updates to discern between true distribution shifts and transient noise. Our extensive experiments across synthetic benchmarks, public advertisement datasets, and largescale real-world recommendation systems consistently demonstrated that CAdam outperforms Adam and other well-established optimizers in both adaptability and robustness. Specifically, CAdam showed superior performance in scenarios with sudden and continuous distribution shifts, as well as in environments with significant noise, achieving higher accuracy and lower regret. Moreover, in live A/B testing within a production recommendation system, CAdam led to substantial improvements in model performance and gross merchandise volume (GMV), underscoring its practical effectiveness. Future work may explore further refinements of the confidence assessment mechanism, its integration with other optimization frameworks, and its application to a broader range of machine learning models and real-time systems. Ultimately, CAdam represents a promising advancement in the development of more resilient and adaptive optimization algorithms for dynamic learning environments.

## References

Ahmet Alacaoglu, Yura Malitsky, Panayotis Mertikopoulos, and Volkan Cevher. A new regret analysis for adam-type algorithms. In *International conference on machine learning*, pp. 202–210. PMLR, 2020.

Dosovitskiy Alexey. An image is worth 16x16 words: Transformers for image recognition at scale.

arXiv preprint arXiv: 2010.11929, 2020.

Albert Bifet and Ricard Gavalda. Learning from time-changing data with adaptive windowing. In Proceedings of the 2007 SIAM international conference on data mining, pp. 443–448. SIAM, 2007.

Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. Wide & deep learning for recommender systems. In *Proceedings of the 1st workshop on deep learning for recommender systems*, pp. 7–10, 2016.

Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for youtube recommendations.

In *Proceedings of the 10th ACM conference on recommender systems*, pp. 191–198, 2016.

Alexandre Defossez, Leon Bottou, Francis Bach, and Nicolas Usunier. A simple convergence proof ´
of adam and adagrad. *Transactions on Machine Learning Research*.

John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. *Journal of machine learning research*, 12(7), 2011.

| forms Adam, highlighting its effectiveness in real-world recommendation scenarios. Metric Pr 1 Pr 2 Rec 1 Rec 2 Rec 3 Rec 4 Rk 1   | Average   |        |        |        |        |        |        |        |
|------------------------------------------------------------------------------------------------------------------------------------|-----------|--------|--------|--------|--------|--------|--------|--------|
| Adam                                                                                                                               | 87.41%    | 82.89% | 90.18% | 82.41% | 84.57% | 85.39% | 88.52% | 85.34% |
| CAdam                                                                                                                              | 87.61%    | 83.28% | 90.43% | 82.61% | 85.06% | 85.49% | 88.74% | 85.64% |
| Impr.                                                                                                                              | 0.20%     | 0.39%  | 0.25%  | 0.20%  | 0.49%  | 0.10%  | 0.22%  | 0.30%  |

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In *International conference on machine learning*, pp. 1126–1135. PMLR, 2017.

Shintaro Fukushima, Atsushi Nitanda, and Kenji Yamanishi. Online robust and adaptive learning from data streams. *arXiv preprint arXiv:2007.12160*, 2020.

Yingqiang Ge, Mostafa Rahmani, Athirai Irissappane, Jose Sepulveda, James Caverlee, and Fei Wang. Automated data denoising for recommendation. *arXiv preprint arXiv:2305.07070*, 2023.

Aritra Ghosh, Himanshu Kumar, and P Shanti Sastry. Robust loss functions under label noise for deep neural networks. In *Proceedings of the AAAI conference on artificial intelligence*, volume 31, 2017.

Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. Deepfm: a factorizationmachine based neural network for ctr prediction. *arXiv preprint arXiv:1703.04247*, 2017.

Michael Gutmann and Aapo Hyvarinen. Noise-contrastive estimation: A new estimation principle ¨
for unnormalized statistical models. In *Proceedings of the thirteenth international conference on* artificial intelligence and statistics, pp. 297–304. JMLR Workshop and Conference Proceedings, 2010.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.

Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In *Proceedings of the IEEE conference on computer vision and pattern* recognition, pp. 4700–4708, 2017.

Olivier Chapelle Jean-Baptiste Tien, joycenv. Display advertising challenge, 2014. URL https:
//kaggle.com/competitions/criteo-display-ad-challenge.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In *International* Conference on Learning Representations (ICLR), 2015.

Hyeyoung Ko, Suyeon Lee, Yoonseo Park, and Anna Choi. A survey of recommendation systems:
recommendation models, techniques, and application fields. *Electronics*, 11(1):141, 2022.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.

2009.

Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Jiawei Han. On the variance of the adaptive learning rate and beyond. *arXiv preprint arXiv:1908.03265*,
2019.

I Loshchilov. Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*, 2017.

Jie Lu, Anjin Liu, Fan Dong, Feng Gu, Joao Gama, and Guangquan Zhang. Learning under concept drift: A review. *IEEE transactions on knowledge and data engineering*, 31(12):2346–2363, 2018.

Costas Panagiotakis, Harris Papadakis, Antonis Papagrigoriou, and Paraskevi Fragopoulou. Dtec:
Dual training error based correction approach for recommender systems. *Software Impacts*, 9: 100111, 2021.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, highperformance deep learning library. *Advances in neural information processing systems*, 32, 2019.

Ning Qian. On the momentum term in gradient descent learning algorithms. *Neural networks*, 12
(1):145–151, 1999.

Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, and Jun Wang. Product-based neural networks for user response prediction. In 2016 IEEE 16th international conference on data mining (ICDM), pp. 1149–1154. IEEE, 2016.

Sashank J Reddi, Satyen Kale, and Sanjiv Kumar. On the convergence of adam and beyond. In International Conference on Learning Representations, 2018.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Roberto Saia, Ludovico Boratto, and Salvatore Carta. A semantic approach to remove incoherent items from a user profile and improve the accuracy of a recommender system. Journal of Intelligent Information Systems, 47:111–134, 2016.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.

Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*, 2014.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.

Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1):1929–1958, 2014.

W Nick Street and YongSeog Kim. A streaming ensemble algorithm (sea) for large-scale classification. In *Proceedings of the seventh ACM SIGKDD international conference on Knowledge* discovery and data mining, pp. 377–382, 2001.

Rui Su, Husheng Guo, and Wenjian Wang. Elastic online deep learning for dynamic streaming data.

Information Sciences, pp. 120799, 2024.

A Vaswani. Attention is all you need. *Advances in Neural Information Processing Systems*, 2017. Antonio David Viniski, Jean Paul Barddal, Alceu de Souza Britto Jr, Fabr ˆ ´ıcio Enembreck, and Humberto Vinicius Aparecido de Campos. A case study of batch and incremental recommender systems in supermarket data under concept drifts and cold start. *Expert Systems with Applications*, 176:114890, 2021.

Bohan Wang, Jingwen Fu, Huishuai Zhang, Nanning Zheng, and Wei Chen. Closing the gap between the upper bound and lower bound of adam's iteration complexity. Advances in Neural Information Processing Systems, 36, 2024.

Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. Deep & cross network for ad click predictions.

In *Proceedings of the ADKDD'17*, pp. 1–7. 2017.

Wenjie Wang, Fuli Feng, Xiangnan He, Liqiang Nie, and Tat-Seng Chua. Learning robust recommender from noisy implicit feedback. *arXiv preprint arXiv:2112.01160*, 2021.

Yuan-Yuan Xu, Shen-Ming Gu, and Fan Min. Improving recommendation quality through outlier removal. *International Journal of Machine Learning and Cybernetics*, 13(7):1819–1832, 2022.

Yifan Yang, Alec Koppel, and Zheng Zhang. A gradient-based approach for online robust deep neural network training with noisy labels. *arXiv preprint arXiv:2306.05046*, 2023.

Yushun Zhang, Congliang Chen, Naichen Shi, Ruoyu Sun, and Zhi-Quan Luo. Adam can converge without any modification on update rules. *Advances in neural information processing systems*, 35:28386–28399, 2022.

Jieming Zhu, Jinyang Liu, Shuai Yang, Qi Zhang, and Xiuqiang He. Open benchmarking for clickthrough rate prediction. In Proceedings of the 30th ACM international conference on information & knowledge management, pp. 2759–2769, 2021.

Manzil Zaheer, Sashank Reddi, Devendra Sachan, Satyen Kale, and Sanjiv Kumar. Adaptive methods for nonconvex optimization. *Advances in neural information processing systems*, 31, 2018.

Matthew D Zeiler. Adadelta: an adaptive learning rate method. *arXiv preprint arXiv:1212.5701*,
2012.

Si-si Zhang, Jian-wei Liu, and Xin Zuo. Adaptive online incremental learning for evolving data streams. *Applied Soft Computing*, 105:107255, 2021.

648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the 20th international conference on machine learning (icml-03), pp. 928–936, 2003.

Jieming Zhu, Quanyu Dai, Liangcai Su, Rong Ma, Jinyang Liu, Guohao Cai, Xi Xiao, and Rui Zhang. Bars: Towards open benchmarking for recommender systems. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 2912–2923, 2022.

Juntang Zhuang, Tommy Tang, Yifan Ding, Sekhar C Tatikonda, Nicha Dvornek, Xenophon Papademetris, and James Duncan. Adabelief optimizer: Adapting stepsizes by the belief in observed gradients. *Advances in neural information processing systems*, 33:18795–18806, 2020.

## A Proofs Of Theorem 1

702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 Proof. CAdam updates the parameters as follows

xt+1,Ξt = xt,Ξt − αtmt,Ξt/pvˆt = xt,Ξt − αtV
$$\sqrt{\hat{v}_{t}}=x_{t,\Xi_{t}}-\alpha_{t}V_{t}^{-1/2}\Big(\beta_{1,t}m_{t-1,\Xi_{t}}+(1-\beta_{1,t})g_{t,\Xi_{t}}\Big).$$
Subtracting x
∗from both sides yields Given a stream of objectives ft : R
d → R, t = 1, 2*, . . . , T*, online learning aims to minimize the regret w.r.t. the optimum; that is,

$$R_{T}:=\sum_{t=1}^{T}f_{t}(x_{t})-\sum_{t=1}^{T}f_{t}(x^{*}),\quad x^{*}=\operatorname*{argmin}_{x}\sum_{t=1}^{T}f_{t}(x).$$
$$({\mathfrak{I}})$$
$$(4)$$
$$({\boldsymbol{S}})$$

$$(T)$$

Recall that each update in CAdam can be characterized as follows1:

mt = β1,tmt−1 + (1 − β1,t)gt, (4)
$\mathbf{E}=\mathbb{R}^{\text{T},\mathbf{P}}$
$\left\{\begin{array}{ll}\mbox{\rm{\small$\pi$}}&\mbox{\rm{\small$\pi$}}\\ \mbox{\rm{\small$\pi$}}&\mbox{\rm{\small$\pi$}}\end{array}\right.$
$$\begin{array}{c}{{\hat{v}_{t}=\operatorname*{max}(\hat{v}_{t-1},v_{t}),}}\\ {{x_{t+1}=x_{t}-\alpha_{t}m_{t,\Xi_{t}}/\hat{v}_{t}.}}\end{array}$$
t, (5)
0, else , (6)
vˆt = max(ˆvt−1, vt), (7)
xt+1 = xt − αtmt,Ξt/vˆt. (8)
where Ξt := {i ∈ [d] : mt,i · gt,i ≥ 0} indicates the set of active entries at step t. For notation clarity, let xt,Ξ be the vector of which the entries not belonging to Ξ are masked. Following the AMSGrad (Reddi et al., 2018), we are to prove that the sequence of points obtained by CAdam satisfies RT /T → 0 as T increases.

We first introduce three standard assumptions:
Assumption 1. Let ft : R
d → R, t = 1, 2, . . . , T be a sequence of convex and differentiable functions with ∥∇ft(x)∥∞ ≤ G∞ *for all* t ∈ [T].

Assumption 2. Let {mt}, {vt} be the sequences used in CAdam, αt = α/√*t, β*1,t = β1λ t−1 <
1, γ = β1/
√β2 < 1 *for all* t ∈ [T].

Assumption 3. The points involved are within a bounded diameter D∞; that is, for the optimal point x
∗ and any points xt generated by CAdam, it holds ∥xt − x
∗∥∞ ≤ D∞/2.

We present several essential lemmas in the following. Given that some of these lemmas have been partially established in prior works (Kingma & Ba, 2015; Reddi et al., 2018), we include them here for the sake of completeness.

Lemma 1. *For a convex and differentiable function* f : R
d → R*, we have* f(x) − f(y) ≤ ⟨∇f(x), x − y⟩. (9)
Lemma 2. *Under Assumption 1 and 2, we have*

$$\left\langle g_{t,\Xi_{t}},x_{t,\Xi_{t}}-x_{\Xi_{t}}^{*}\right\rangle\leq\frac{1}{2\alpha_{t}(1-\beta_{1,t})}\left(\|V_{t}^{1/4}(x_{t,\Xi_{t}}-x_{\Xi_{t}}^{*})\|^{2}-\|V_{t}^{1/4}(x_{t+1,\Xi_{t}}-x_{\Xi_{t}}^{*})\|^{2}\right)$$ $$+\frac{\beta_{1}}{2\alpha_{t}(1-\beta_{1,t})}\|V_{t}^{1/4}(x_{t}-x^{*})\|^{2}$$ $$+\frac{\alpha_{t}}{2(1-\beta_{1,t})}\|V_{t}^{1/4}m_{t}\|^{2}+\frac{\alpha_{t}\beta_{1,t}}{2(1-\beta_{1,t})}\|V_{t}^{-1/4}m_{t-1}\|^{2},$$
$$({\mathfrak{g}})$$
$$x)-f(y)\leq\langle\nabla f(x),x-y\rangle.$$
$J\;\cup\;\emptyset$  . 
$$w e\,h a v e$$
where Vt := diag(ˆvt).
$$(10)$$

=∥V =∥V
1/4
$$-\,x_{\Xi_{t}}^{*})-\alpha_{t}V_{t}^{-1/4}m_{t,\Xi_{t}}\|_{2}^{2}$$
t(xt,Ξt − x
$$:\|V_{t}^{1/4}(x_{t,\Xi_{t}}-x$$
∗ Ξt
)∥
2
⟩ + ∥αtV
−1/4
t mt,Ξt∥
2
2.
1Note that we omit the bias corrections for clarity purpose. It is not difficult to modify the proofs to obtain
a more general one.
$$\Xi_{2}^{2}-2\alpha_{t}\langle\beta_{1,t}m_{t-1,\Xi_{t}}+(1-\beta_{1,t})g_{t,\Xi_{t}},x_{t,\Xi_{t}}-x_{\Xi_{t}}^{*}\rangle+\Xi_{2}^{2}$$
$$=\|V_{t}^{1/4}(x_{t,\Xi_{t}}-x_{\Xi_{t}}^{\star})\|_{2}^{2}-2\langle\alpha_{t}V_{t}^{-1/4}m_{t,\Xi_{t}},V_{t}^{1/4}(x_{t,\Xi_{t}}-x_{\Xi_{t}}^{\star})\rangle+\|\alpha_{t}V_{t}^{-1/4}m_{t,\Xi_{t}}\|_{2}^{2}$$
∥V
1/4 t(xt+1,Ξt − x
∗
Ξt
)∥
22 756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 Rearranging the equation gives

$$\left\langle g_{t,\Xi_{t}},x_{t,\Xi_{t}}-x_{\Xi_{t}}^{\ast}\right\rangle=\frac{1}{2\alpha_{t}(1-\beta_{1,t})}\Big{(}\|V_{t}^{1/4}(x_{t,\Xi_{t}}-x_{\Xi_{t}}^{\ast})\|_{2}^{2}-\|V_{t}^{1/4}(x_{t+1,\Xi_{t}}-x_{\Xi_{t}}^{\ast})\|_{2}^{2}\Big{)}$$ $$-\frac{\beta_{1,t}}{1-\beta_{1,t}}\Big{\langle}m_{t-1,\Xi_{t}},x_{t,\Xi_{t}}-x_{\Xi_{t}}^{\ast}\Big{\rangle}+\frac{\alpha_{t}}{2(1-\beta_{1,t})}\|V_{t}^{-1/4}m_{t,\Xi_{t}}\|_{2}^{2}.$$
The results follow from the Cauchy-Schwarz inequality and Young's inequality: −β1,t 1 − β1,t Dmt−1,Ξt, xt,Ξt − x ∗ Ξt E=β1,t 1 − β1,t Dmt−1,Ξt, x∗Ξt − xt,Ξt E =β1,t 1 − β1,t D√αtV −1/4 t mt−1,Ξt ,1 √αt V 1/4 t(x ∗Ξt − xt,Ξt ) E ≤β1,t 1 − β1,t √αt∥V −1/4 t mt−1,Ξt∥ ·  1 √αt ∥V 1/4 t(x ∗ Ξt − xt,Ξt)∥  ≤β1,t 1 − β1,t αt 2 ∥V −1/4 t mt−1,Ξt ∥ 2 +1 2αt ∥V 1/4 t(xt,Ξt − x ∗Ξt )∥ 2 ≤β1,t 1 − β1,t αt 2 ∥V −1/4 t mt−1∥ 2 +1 2αt ∥V 1/4 t(xt − x ∗)∥ 2, and the fact that ∥V −1/4 t mt,Ξt∥ 22 ≤ ∥V −1/4 t mt∥ 22.

$$(11)$$

Lemma 3. *Under Assumption 1, 2, and 3, we have* Proof. If the i-th entry is not updated at step t, i.e., i ∈ [d] \ Ξt, it can be derived that

$$\left\langle g_{t},x_{t}-x^{*}\right\rangle\leq\left\langle g_{t,\Xi},x_{t,\Xi}-x_{\Xi}^{*}\right\rangle+\frac{d\beta_{1}\lambda^{t-1}D_{\infty}G_{\infty}}{1-\beta_{1}}.$$
. (11)
β1,tmt−1,i + (1 − β1,t)gt,i· gt,i ≤ 0 ⇒β1,tmt−1,i + (1 − β1,t)gt,i· sgn(gt,i) ≤ 0 ⇒ − β1,t|mt−1,i| + (1 − β1,t)|gt,i| ≤ 0 ⇒|gt,i| ≤  β1,t 1 − β1,t |mt−1,i| ⇒|gt,i| ≤ β1,t 1 − β1,t G∞ ← Assumption 1 ⇒|gt,i| ≤  β1λ t−1 1 − β1 G∞, i ∈ [d] \ Ξt. ← Assumption 2
With Assumption 3, it immediately yields the desired inequality that

$$\left\langle g_{t,}x_{t}-x^{*}\right\rangle=\left\langle g_{t,\Xi},x_{t,\Xi}-x_{\Xi}^{*}\right\rangle+\left\langle g_{t,[d]\backslash\Xi},x_{t,[d]\backslash\Xi}-x_{[d]\backslash\Xi}^{*}\right\rangle$$ $$\leq\left\langle g_{t,\Xi},x_{t,\Xi}-x_{\Xi}^{*}\right\rangle+\sum_{i=1}^{d}\frac{\beta_{1}\lambda^{t-1}D_{\infty}G_{\infty}}{1-\beta_{1}}.$$
$$(12)$$
$$\sum_{t\in[T]}\frac{\beta_{1,t}}{2\alpha_{t}(1-\beta_{1,t})}\|V_{t}^{1/4}(x_{t}-x^{*})\|^{2}\leq\frac{d D_{\infty}^{2}G_{\infty}}{2\alpha(1-\beta_{1})(1-\lambda)^{2}}.$$
. (12)
Lemma 4. *Given Assumption 1, 2, and 3, we have* Proof.

810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 where γ := β1/
√β2.

We are ready to prove the final results now. Concretely, Theorem 1 is a straightfoward corollary of the following conclusion. Theorem 2. *Under the Assumption 1, 2, and 3, the regret is converged with*

$$\sum_{t\in[T]}\frac{\beta_{1,t}}{2\alpha_{t}(1-\beta_{1,t})}\|V_{t}^{1/4}(x_{t}-x^{*})\|^{2}$$ $$\leq\frac{1}{2\alpha(1-\beta_{1})}\sum_{t\in[T]}\sqrt{t}\lambda^{t-1}\|V_{t}^{1/4}(x_{t}-x^{*})\|^{2}$$ $$\leq\frac{G_{\infty}}{2\alpha(1-\beta_{1})}\sum_{t\in[T]}\sqrt{t}\lambda^{t-1}\|x_{t}-x^{*}\|^{2}$$ $$\leq\frac{dD_{\infty}^{2}G_{\infty}}{2\alpha(1-\beta_{1})}\sum_{t\in[T]}\sqrt{t}\lambda^{t-1}$$ $$dD_{\infty}^{2}G_{\infty}$$
$$\leftarrow\mathrm{Assumption\1}$$
$\leftarrow$ Assumption 3. 
2 ← Assumption 1
√tλt−1 ← Assumption 3
$$\stackrel{t\in[T]}{\leq}\frac{d D_{\infty}^{2}G_{\infty}}{2\alpha(1-\beta_{1})}\sum_{t\in[T]}\lambda^{t-1}t$$
$$\leq\frac{d D_{\infty}^{2}G_{\infty}}{2\alpha(1-\beta_{1})}\frac{1}{(1-\lambda)^{2}}.$$

Lemma 5 (Reddi et al. (2018) Lemma2). *Under Assumption 2, we have*

$$\sum_{t\in[T]}\alpha_{t}\|V_{t}^{-1/4}m_{t}\|^{2}\leq\frac{\alpha d G_{\infty}}{(1-\gamma)(1-\beta_{1})\sqrt{1-\beta_{2}}}\sqrt{T},$$
√T , (13)

$$(13)$$
$$R_{T}\leq\frac{dD_{\infty}^{2}G_{\infty}\sqrt{T}}{2\alpha(1-\beta_{1})}+\frac{d(2\alpha+D_{\infty})D_{\infty}G_{\infty}}{2\alpha(1-\beta_{1})(1-\lambda)^{2}}+\frac{\alpha dG_{\infty}\sqrt{T}}{(1-\gamma)(1-\beta_{1})^{2}\sqrt{1-\beta_{2}}}.\tag{14}$$

Proof. Based on Lemma 1, Lemma 2, and Lemma 3, the regret can be firstly bounded by

$$R_{T}=\sum_{t\in[T]}\left(f_{t}(x_{t})-f_{t}(x^{*})\right)\leq\sum_{t\in[T]}\left(g_{t},x_{t}-x^{*}\right)$$ $$\leq\sum_{t\in[T]}\left\langle g_{t,\Xi_{t}},x_{t,\Xi_{t}}-x_{\Xi_{t}}^{*}\right\rangle+\sum_{t\in[T]}\frac{d\beta_{1}\lambda^{t-1}D_{\infty}G_{\infty}}{1-\beta_{1}}$$ $$\leq\sum_{t\in[T]}\frac{1}{2\alpha_{t}(1-\beta_{1,t})}\left(\|V_{t}^{1/4}(x_{t,\Xi_{t}}-x_{\Xi_{t}}^{*})\|^{2}-\|V_{t}^{1/4}(x_{t+1,\Xi_{t}}-x_{\Xi_{t}}^{*})\|^{2}\right)$$
| {z } ⃝1 β1,t 2αt(1 − β1,t) ∥V 1/4 t(xt − x ∗)∥ 2 αt 2(1 − β1,t) ∥V −1/4 t mt∥ 2 +X +X t∈[T] t∈[T] | {z } ⃝2 | {z } ⃝3 t∈[T] dβ1λ t−1D∞G∞ 1 − β1 αtβ1,t 2(1 − β1,t) ∥V −1/4 t mt−1∥ 2 + X + X . t∈[T] | {z } ⃝4 | {z } ⃝5
Let us address each term in turn. For the first term, we are to separately bound each entry and the results follows from the summation. For the i-th entry, let T
i+ = [t : i ∈ Ξt] be a sequence collecting 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917

⃝1 i =
t˜
X
|T+|
1
2αt(1 − β1,t)
(ˆv
1/4
t,i (xt,i − x
∗
i))2 − (ˆv
1/4
t,i (xt+1,i − x
∗
i))2
t=t˜1
2αt˜1
(1 − β1)+
1
2
t˜
X
|T+|
≤
vˆ
1/2
t˜1,i (xt˜1,i − x
∗
i
)
2
t=t˜2
hvˆ
1/2
t,i (xt,i − x
∗
i)
2
αt(1 − β1,t)
−
vˆ
1/2
t−1,i(xt,i − x
∗
i)
2
αt−1(1 − β1,t−1)
i
2αt˜1
(1 − β1)
+
1
2
t˜
X
|T+|
=
vˆ
1/2
t˜1,i (xt˜1,i − x
∗
i)
2
t=t˜2
hvˆ
1/2
t,i (xt,i − x
∗
i)
2
αt(1 − β1,t−1)
−
vˆ
1/2
t,i (xt,i − x
∗
i)
2
αt(1 − β1,t−1)
+
vˆ
1/2
t,i (xt,i − x
∗
i)
2
αt(1 − β1,t)
| {z }
≤0
−
vˆ
1/2
t−1,i(xt,i − x
∗
i)
2
αt−1(1 − β1,t−1)
i
2αt˜1
(1 − β1)
+
1
2
t˜
X
|T+|
≤
vˆ
1/2
t˜1,i (xt˜1,i − x
∗
i)
2
1
1 − β1,t−1
| {z }
≤1/(1−β1)
hvˆ
1/2
t,i (xt,i − x
∗
i)
2
αt
−
vˆ
1/2
t−1,i(xt,i − x
∗
i)
2
αt−1
i
t=t˜2
$$\leftarrow{\mathrm{Assumption~}}3$$
| {z }
≥0 by vˆt,i≥vˆt−1,i
2αt˜1
(1 − β1)+
D2∞
2(1 − β1)
t˜
X
|T+|
≤
vˆ
1/2
t˜1,i (xt˜1,i − x
∗
i
)
2
t=t˜2
hvˆ
1/2
t,i
αt−
vˆ
1/2
t−1,i
αt−1
i← Assumption 3
2αt˜1
(1 − β1)
+D2∞
2(1 − β1)
hvˆ
1/2
t˜|T+|,i
αt˜|T+|
−
vˆ
1/2
t˜1,i
αt˜1
i
=
vˆ
1/2
t˜1,i (xt˜1,i − x
∗
i)
2
≤D2∞
2(1 − β1)
vˆ
1/2
t˜|T+|,i
αt˜|T+|
≤
D2∞G∞
√T
2α(1 − β1)
.
Hence,
$$\mathbb{D}=\sum_{i\in[d]}\mathbb{D}_{i}\leq\frac{d D_{\infty}^{2}G_{\infty}\sqrt{T}}{2\alpha(1-\beta_{1})}.$$

$$(15)$$
. (15)
$$\mathcal{Q}=\sum_{t\in[T]}\frac{\beta_{1,t}}{2\alpha_{t}(1-\beta_{1,t})}\|V_{t}^{1/4}(x_{t}-x^{*})\|^{2}\leq\frac{d D_{\infty}^{2}G_{\infty}}{2\alpha(1-\beta_{1})(1-\lambda)^{2}}\qquad\leftarrow\mathrm{Lemma}.$$
$$\begin{split}\mathfrak{D}&=\sum_{t\in[T]}\frac{\alpha_{t}}{2(1-\beta_{1,t})}\|V_{t}^{-1/4}m_{t}\|^{2}\leq\frac{1}{2(1-\beta_{1})}\sum_{t\in[T]}\alpha_{t}\|V_{t}^{-1/4}m_{t}\|^{2}\\ &\leq\frac{\alpha d G_{\infty}\sqrt{T}}{2(1-\gamma)(1-\beta_{1})^{2}\sqrt{1-\beta_{2}}}.\end{split}$$
$$\begin{split}\mathfrak{d}&=\sum_{t\in[T]}\frac{\alpha_{t}\beta_{1,t}}{2(1-\beta_{1,t})}\|V_{t}^{-1/4}m_{t-1}\|^{2}\leq\frac{1}{2(1-\beta_{1})}\sum_{t\in[T]}\alpha_{t}\|V_{t-1}^{-1/4}m_{t-1}\|^{2}\\ &\leq\frac{1}{2(1-\beta_{1})}\sum_{t\in[T]}\alpha_{t-1}\|V_{t-1}^{-1/4}m_{t-1}\|^{2}=\frac{1}{2(1-\beta_{1})}\sum_{t\in[T-1]}\alpha_{t}\|V_{t}^{-1/4}m_{t}\|^{2}\\ &\leq\frac{\alpha\mathcal{d}G_{\infty}\sqrt{T}}{2(1-\gamma)(1-\beta_{1})^{2}\sqrt{1-\beta_{2}}}.\end{split}$$
$\leftarrow$ Lemma 5. 
$$\leftarrow\mathrm{Lemma}\ 5$$
all steps that xiis succesfully updated, and t˜k ∈ T i+ be the k-th element of T
i+. For simplicity, we will omit the superscript without ambiguity. Finally, we have

## B Hyperparameters B.1 Numerical Experiment B.2 Cnn On Image Classification

For the CNN-based image classification experiments on the CIFAR-10 dataset, we used a learning rate of 3 × 10−4, β1 = 0.9, β2 = 0.999, weight decay of 0.0005, and ϵ = 1 × 10−8.

## B.3 Public Advertisement Dataset

918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 Figure 5 illustrate how both optimizers perform in a noise-free environment. C.2 EXPERIMENT ON RESNET AND DENSENET We perform experiments on Resnet(He et al., 2016) and Densenet(Huang et al., 2017) to further illustrate the effectiveness of CAdam. C.3 RELATIONSHIP BETWEEN LEARNING RATE, PERFORMANCE, AND ALIGNMENT RATIO Noisy Samples For the noisy samples experiments, the hyperparameters were set as follows: a learning rate of 0.1, β1 = 0.9, β2 = 0.999, ϵ = 1 × 10−8, and a maximum number of iterations T = 1500. Due to resource limitations, we performed a grid search over the learning rates for each optimizer and model using the following range: {lr default/5, lr default/2, lr default, 2×lr default, 5× lr default}, where lr default is the default learning rate specified in the FuxiCTR library. We reported the best performance for each optimizer based on this search. All other hyperparameters were kept the same as those in the FuxiCTR library (Zhu et al., 2021; 2022).

## C Additional Experiments

C.1 NUMERICAL EXPERIMENTS Distribution Shift For the distribution shift experiments, we used the following hyperparameters: a cycle length of 40, a learning rate α = 0.5, exponential decay rates for the first and second moment estimates β1 = 0.9 and β2 = 0.999 respectively, and a small constant ϵ = 1 × 10−8to prevent division by zero. The number of time steps was set to T = 100.

$$R_{T}\leq\frac{d D_{\infty}^{2}G_{\infty}\sqrt{T}}{2\alpha(1-\beta_{1})}+\frac{d(2\alpha+D_{\infty})D_{\infty}G_{\infty}}{2\alpha(1-\beta_{1})(1-\lambda)^{2}}+\frac{\alpha d G_{\infty}\sqrt{T}}{(1-\gamma)(1-\beta_{1})^{2}\sqrt{1-\beta_{2}}}.$$
$$\Im=\sum_{t\in[T]}{\frac{d\beta_{1}\lambda^{t-1}D_{\infty}G_{\infty}}{1-\beta_{1}}}={\frac{d\beta_{1}D_{\infty}G_{\infty}}{1-\beta_{1}}}\sum_{t\in[T]}\lambda^{t-1}\leq{\frac{d D_{\infty}G_{\infty}}{(1-\beta_{1})(1-\lambda)^{2}}}.$$

We tested different learning rates on the Criteo x4 001 dataset using the DeepFM model to understand the relationship between the learning rate, performance, and alignment ratio. The results in 4 show that the performance initially increases with the learning rate but starts to decline as the learning rate continues to rise. Conversely, the consistent ratio R steadily decreases as the learning rate increases.

972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079