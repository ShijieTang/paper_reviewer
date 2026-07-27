# 

Reyhane Askari-Hemmat * 1 **Mohammad Pezeshki** * 1 **Elvis Dohmatob** 1 2 3 Florian Bordes 1 **Pietro Astolfi** 1 Melissa Hall 1 Jakob Verbeek 1 Michal Drozdzal 1 **Adriana Romero-Soriano** 1 3 4 5

## Abstract

Inspired by the principle of deliberate practice in human learning, we propose Deliberate Practice for Synthetic Data Generation (DP), a novel framework that improves sample efficiency through dynamic synthetic data generation. Prior work has shown that scaling synthetic data is inherently challenging, as naively adding new data leads to diminishing returns. To address this, pruning has been identified as a key mechanism for improving scaling, enabling models to focus on the most informative synthetic samples. Rather than generating a large dataset and pruning it afterward, DP efficiently approximates the direct generation of informative samples. We theoretically show how training on challenging, informative examples improves scaling laws and empirically validate that DP achieves better scaling performance with significantly fewer training samples and iterations. On ImageNet-100, DP generates 3.4× fewer samples and requires six times fewer iterations, while on ImageNet-1k, it generates 8× fewer samples with a 30% reduction in iterations, all while achieving superior performance compared to prior work.

## 1. Introduction

A key principle underlying learning in human is deliberate practice (DP)—progress is made not by repeating what is already known but by continuously engaging with tasks that stretch the limits of one's abilities (Ericsson et al., 1993). For example, when learning to play the guitar, simply practicing songs that one has mastered does little to improve skill. Instead, targeted practice on challenging tasks and
*Equal contribution 1FAIR at Meta - Montreal, Paris, and New York City labs 2Concordia University 3Mila 4McGill University 5Canada CIFAR AI chair. Correspondence to: Reyhane Askari-Hemmat <reyhaneaskari@meta.com>, Mohammad Pezeshki <mpezeshki@meta.com>.

refining learning through feedback, leads to real progress.

This principle highlights that effective learning requires exposure to informative and difficult examples rather than passive repetition. In contrast, most machine learning models are trained on precollected data that remain static throughout training, limiting their ability to dynamically adapt to their own weaknesses. One promising source of data for visual recognition tasks is large-scale pre-trained text-to-image models (Rombach et al., 2022). They provide an essentially infinite source of synthetic training data, presenting an alternative to realworld datasets, which are often expensive or infeasible to curate (Hemmat et al., 2023; Shin et al., 2023; Zhang et al., 2024). With the great promise of text-to-image models, a natural question arises: what is the potential of learning using **only** synthetic data? Empirical studies show that increasing the volume of synthetic training data often leads to diminishing returns, with performance gains following a power law stagnation (Fan et al., 2024; Tian et al., 2024a). Instead, pruning to remove uninformative examples has proven effective in improving the effectiveness of training with real or synthetic data (Sorscher et al., 2022; Kolossov et al., 2024; Feng et al., 2024).

Inspired by human learning principles and recent advances in generative image models, we propose the Deliberate Practice (DP) for Synthetic Data Generation framework. Unlike static approaches that generate all synthetic training data upfront (Fan et al., 2024; Shin et al., 2023; Hemmat et al., 2023), our framework incorporates a dynamic loop between a diffusion model and a downstream learner throughout the training. More concretely, rather than generating an entire dataset at once and irrespective of the learner and then pruning it to remove uninformative samples, we propose DP to efficiently generate data directly from the pruned distribution of informative samples. By leveraging the learner's prediction entropy to guide the generation process, our approach generates only the most challenging and informative training examples. Our framework operates **dynamically**: we begin with an initial set of synthetic data and train a learner until performance on a real validation set plateaus. At this point, the learner's entropy is used to guide the diffusion model to gen1 erate new challenging examples. These examples are added to the training set, and the process repeats, ensuring that the model is continually exposed to increasingly informative data throughout training. This approach aligns with broader goals in machine learning, such as interactive learning environments, continual learning (Kirkpatrick et al., 2017), and active learning (Settles, 2009). By leveraging a dynamic loop, Deliberate Practice reduces inefficiencies from redundant or already learned data, thereby improving the scaling laws of training with synthetic data. Our contributions are summarized as:
- We introduce the Deliberate Practice for Synthetic Data Generation framework, which dynamically adds new data points when the learner's validation accuracy plateaus [Section 3]. Our framework leverages the learner's prediction entropy to generate challenging synthetic data, improving the scaling behavior of synthetic data (Figures 1 and 4).

- We provide a theoretical analysis of the scaling behavior of a simple model trained on selected examples (Section 4). Using random matrix theory, we characterize the test error as a function of data size and the example selection function, showing **improved scaling** when prioritizing hard and informative examples.

- We show that entropy-guided sampling approximates generating from an entropy-pruned distribution (Section 2). We empirically validate that DP can improve the validation accuracy compared to direct pruning while being remarkably **cheaper in compute up to 5**× (Figure 5).

- We demonstrate that DP outperforms prior work on both ImageNet-100 and ImageNet-1k while requiring significantly less data and fewer training iterations. On ImageNet-100, our approach generated 3.4× **less** samples and completed training in only one-sixth of the iterations used in prior work, yet still achieved superior performance. Similarly, on ImageNet-1k, we generated 8× **less samples** and reduced the number of iterations by 30%, while outperforming previous results (Table 1).

- Furthermore, DP exhibits strong performance on outof-distribution (OOD) datasets, even outperforming models trained with real data on ImageNet-R and ImageNet-Sketch, with **improvements of up to 15%**
(Table 1).

## 2. Problem Formulation

Problem Setup. Standard supervised learning relies on a large real labeled training set. Here, however, we assume no real training data is available, and instead, we must rely on a generative model to synthesize training examples. Formally, let Y denote the set of class labels. Our goal is to train a classifier fϕ : *X → Y*, parameterized by ϕ, which maps inputs x ∈ X (*e.g.*, images) to labels y ∈ Y. We are given a predefined label set Y, a fixed (small) validation set Dval = {(xi, yi)}
n i=1 consisting of real data for evaluation, and a generative model gθ capable of sampling synthetic data conditioned on a label, *i.e.*, x ∼ gθ(y). However, no real training data is available, *i.e.*, Dtr = ∅. The objective is to train fϕ using *as few generated examples as possible* while maximizing generalization to real data as measured by performance on Dval. The key challenge is to generate minimal yet effective training data, requiring a principled mechanism to select/generate informative examples. The Need for Informative Examples. Not all synthetic samples contribute equally to learning. Prior work shows that simply increasing the synthetic dataset size leads to diminishing returns, as many generated samples are redundant or too easy (Fan et al., 2024). Instead, training should focus on examples that maximize learning efficiency.

Given a measure of *informativeness* for a synthetic sample x, one approach is to generate a large dataset and prune uninformative examples. Formally, let Dpool = {(xi, yi)}
N i=1 be a large set of N generated samples. We define a pruned dataset as D′:= {(xi, yi) | i ∈ [N], qi = 1}, where qi ∈ {0, 1} is a selection variable determining whether a data point (xi, yi) ∈ Dpool is retained. The subset size is constrained by m =PN
i=1 qi. The quantity N/m is referred to as the over-sampling ratio.

Let P and Q denote the distributions of the original and pruned datasets, respectively. The pruning process operates as an importance sampling scheme:

$$\mathrm{d}Q=\pi\,\mathrm{d}P,$$
$$(1)$$
dQ = π dP, (1)
where π is a normalized weighting function that retains the informative samples. The generate-then-prune approach ensures that only informative examples are kept, it is computationally inefficient, as many generated samples are discarded. This motivates the need to devise mechanisms to directly sample the informative examples. Approximate Sampling of Informative Examples. Suppose that Dpool is generated using a diffusion model with induced probability P. The generative process is governed by a reverse SDE (Song & Ermon, 2019):

$$\mathrm{d}x=\left[v(x,t)-g(t)^{2}\nabla\log p_{t}(x)\right]\mathrm{d}t+g(t)\,\mathrm{d}W(t),$$

(2)
where W(t) is a Wiener process, modeling stochastic noise, v(*x, t*) is a drift term, g(t) is a coefficient controlling the noise level at time t, and ∇ log pt(x) is the score function.

Conventional methods: Expensive one-time selection of challenging samples 13M synthetic data 1.3M synthetic data 1.3M synthetic data with DP
Massive amount of Data Prompt Generator Selector Challenging samples Selection Criterion 
(According to the learner)
Learner 10x less data 50.2 To p1 A
cc ura cy 45.8 Deliberate Practice: Efficient continuous generation of challenging samples 44.1 Prompt Criterion-Guided Generator Challenging samples 
(Dynamic)
Selection Criterion 
(According to the learner)
Learner
Figure 1: (Top): Conventional approaches generate (or collect) a massive static dataset and then select challenging examples in a one-time filtering step based on the learner's selection criterion. This is inefficient, as most generated data is discarded. (**Bottom**): DP **continuously generates only the most challenging examples** based on continuous feedback from the learner, eliminating the need for large-scale data pruning. This iterative process ensures that training focuses on progressively informative examples, improving efficiency and performance. (**Right**): Top-1 validation accuracy on ImageNet-1k with models trained solely on synthetic data. DP (orange) achieves higher accuracy than the 13M synthetic data setup (blue) while using **10× fewer samples**, significantly outperforming the 1.3M baseline (gray). Instead of sampling from P, we aim to sample directly from Q as in Eq. (1). By Girsanov's theorem (Oksendal, 2013), modifying the probability measure from P to Q introduces a correction term in the reverse SDE:
in which ϵ
(t) θ
(xt, y) approximates the conditional score function using a pretrained denoising network (Ho & Salimans, 2022):

$$\epsilon_{\theta}(x_{t},y)\approx(1+\lambda)\bar{\epsilon}_{\theta}(x,y)-\lambda\bar{\epsilon}_{\theta}(x)$$
$$({\mathfrak{H}})$$
$$\mathrm{d}x=\left[v(x,t)-g(t)^{2}(\nabla\log p_{t}(x)+\nabla\log\pi(x,t))\right]\,\mathrm{d}t\tag{3}$$ $$+\ g(t)\,\mathrm{d}W(t).$$

The term ∇ log π(*x, t*) effectively modifies the score function and biases the sampling distribution according to the weighting function π(x, t). This modification allows approximating direct sampling from the pruned distribution Q, eliminating the need to first sample uniformly from P
and later prune the data.

## 2.1. Efficient Entropy-Guided Sampling With Ddim.

We leverage denoising diffusion implicit models (DDIMs) (Song et al., 2020) for efficient sampling. At each step t, the reverse update for generating a conditional sample is:

$x_{t-1}=\sqrt{\xi_{t-1}}\hat{x}_{0,t}+\underbrace{\sqrt{1-\xi_{t-1}-\sigma_{t}^{2}}\cdot\epsilon_{\theta}^{(t)}(x_{t},y)}_{\text{direction pointing to}x_{t}}+\underbrace{\sigma_{t}\epsilon_{t}}_{\text{random noise}}$
where ϵt is random noise and σt and ξt−1 are timedependent coefficients. The term xˆ0,t approximates the final denoised sample:

$$\hat{x}_{0,t}=\frac{x_{t}-\sqrt{1-\xi_{t}}\epsilon_{\theta}^{(t)}(x_{t},y)}{\sqrt{\xi_{t}}},$$
√ξt, (4)
$$(4)$$

where λ is called the classifier-free guidance coefficient which controls the strength of conditional sampling on the label. An efficient way of sampling from a modified diffusion mode as described in Eq. 3 was proposed by Hemmat et al. (2023), where the weighting function is derived from the entropy of the downstream learner, such that,

$$\log\pi\propto H(f_{\phi}(x_{0}))=-\sum_{y\in{\cal Y}}f_{\phi}(y\mid x_{0})\log f_{\phi}(y\mid x_{0}).\tag{6}$$

To compute the entropy as in Eq. 6, we need the denoised sample x0. The term xˆ0,t can be used to cheaply approximate entropy mid-generation. This allows direct sampling of high-entropy examples by modifying the score function:

$$\hat{\epsilon}_{\theta}^{(t)}(x_{t},y)=\epsilon_{\theta}^{(t)}(x_{t},y)+\omega\nabla_{x_{t}}H(f_{\phi}(\hat{x}_{0,t})),\quad\quad(7)$$

where ω controls the contribution of the entropy-guidance.

In (Hemmat et al., 2023), real data is used to pre-train the learner, enabling an accurate estimation of ∇xtH(fϕ(ˆx0,t)).

However, when real data is unavailable, alternative approaches are needed to assess sample informativeness. In the next section, we propose to leverage the learner itself Algorithm 1 Deliberate Practice for Synthetic Data Generation 1: **Input:** Class labels Y, Generative model gθ, Validation set Dval, Initial dataset size N, New data size P, Patience Tmax, Evaluation interval τ .

2: **Output:** Trained classifier fϕ 3: **Initialize:** Generate Dtr0 with N examples from gθ.

Start training fϕ with learning-rate warm-up.

4: Set patience counter T ← 0. 5: **while** training do 6: Update fϕ on a mini-batch drawn uniformly from Dtr k.

7: if (every τ iterations) **then**
8: Evaluate validation accuracy A(fϕ, Dval).

9: Reset T ← 0 if accuracy improves; else increment T ← T + 1.

10: **end if**
11: if T ≥ Tmax **then** 12: Generate P new examples Dnew with feedback:
13: ∇zt log p(xt | y) = ∇zt log pθ(zt) +
ω∇ztH(fϕ(ˆx0,t))
14: Augment training set: Dtr k+1 ← Dtr k ∪ Dnew.

15: Reset T ← 0. 16: **end if** 17: **end while** 18: **Finalize:** Apply learning rate decay.

during training to evaluate entropy and determine the informativeness of generated samples dynamically.

## 3. The Deliberate Practice Framework For Synthetic Data Generation

In this section, we describe our Deliberate Practice framework, in which we efficiently train the learner with synthetic data in absence of any real data. In particular, we move to a setup where we dynamically expand the dataset throughout the training. Our framework is summarized in Algorithm 1. The initial training data. The framework begins by generating an initial set of N synthetic training examples Dtr 0 = {(xi, yi)}
N
i=1 using a pre-trained generative model gθ. For each class yi ∈ Y, the generative model samples images xi ∼ gθ(yi) in a class-conditional manner. The classifier fϕ starts training on this dataset, with a learning-rate warm-up phase. Iterative training and additional data. Training proceeds iteratively with a mechanism to dynamically augment the dataset whenever the classifier's performance stagnates. The process alternates between training the classifier and generating new synthetic examples. Patience mechanism. At regular iteration intervals, τ , the validation accuracy A(fϕ, Dval) is evaluated. If no improvement is observed for Tmax intervals (patience threshold), the framework triggers new data generation. Entropy guided sampling. When the patience mechanism triggers, P new examples Dnew = {(xj , yj )}
P
j=1 are generated. We directly generate samples from the entropy pruned distribution through entropy guided sampling. The entropy is computed based on the current stage of the classifier fϕ.

The ω coefficient controls the effect of entropy-guidance. With ω = 0, we fall back into regular sampling of diffusion models, while ω > 0 results in generations that have a higher entropy under the classifier. Training resumption. The newly generated examples are added to the dataset, Dtrk+1 = Dtr k ∪ Dnew. After augmenting the dataset, training resumes with a constant learning rate until the patience mechanism is triggered again. Minibatches are drawn uniformly from the updated pool, which grows dynamically from size N to N +kP after k iterations of augmentation. This cycle is continued until we reach the cool-down phase where the learning rate is decreased and no more new data is added. See Figure 2 for training dynamics of a classifier training with DP. In Section 4, we provide an intuitive theoretical framework to study the scaling behavior of a simplified DP. In Section 5, we validate the effectiveness of DP in large-scale experiments.

## 4. Training On Informative Examples Improves The Scaling Laws

Before presenting empirical results, we first analyze how selecting informative examples affects the scaling of synthetic data. We study a high-dimensional linear classifier trained with uniform vs. selective sampling and derive an analytic expression for test error using random matrix theory (RMT). Our results show that selecting hard examples improves scaling laws, providing theoretical justification for our approach.

## 4.1. Theoretical Analysis Under An Idealized Setup.

Consider a simple generative model for training data:

$$x\sim{\mathcal{N}}(0,\Sigma),\quad y=\mathrm{sign}(w_{0}^{\top}x),$$
$$({\boldsymbol{\delta}})$$
0 x), (8)
where w0 ∈ R
dis the ground-truth labeling function. This gives a distribution P on R
d × R.

We study the impact of *uniform sampling* versus selective sampling of informative examples on generalization. To formalize this, we assume a pool of n i.i.d. training pairs:

$$X\in\mathbb{R}^{n\times d},\quad Y\in\mathbb{R}^{n}.$$
n. (9)
0 10k 20k 30k 40k 50k Iterations 0 25 50 60 70 0 10k 20k 30k 40k 50k Iterations 1.5 2.0 2.5 3.0 3.5 4.0 4.5 Total Data Size: 130k + 130k + 130k Total Data Size: 130k + 130k Total Data Size: 130k Warm-up phase Cool-down phase Iterations at which new data is added Val ida tio n Ac cura cy Tra ini ng Lo ss Total Data Size: 130k + 130k + 130k Total Data Size: 130k + 130k Total Data Size: 130k Warm-up phase Cool-down phase Iterations at which new data is added
A linear classifier wˆ is trained using the following loss:

$$\hat{w}=\mathop{\rm arg\,min}_{w}\quad\frac{1}{n}\sum_{i=1}^{n}q_{i}\ell(w^{\top}x_{i},y_{i})+\frac{\lambda}{2}\|w\|^{2}.\tag{10}$$

where ℓ(*z, y*) = (z − y)
2/2 is the squared loss, λ > 0 is a regularization parameter, and qi:= q(x
⊤
i ws) is a selection strategy that determines whether an example is included in training based on its projection in a given direction ws ∈ R
d, and an arbitrary measurable binary function q : R → {0, 1}
which encodes the selection strategy. The *selection/pruning ratio* is given by:

$$p=\mathbb{E}[q(x^{\top}w_{s})]{\mathrm{~for~}}x\sim{\mathcal{N}}(0,\Sigma).$$

The resulting classifier has a closed-form solution:

$${\hat{w}}={\frac{1}{n}}R X^{\top}D Y,\quad R:=\left({\frac{1}{n}}X^{\top}D X+\lambda I_{d}\right)^{-1}$$
−1, (12)
where D ∈ R
n×n is a diagonal matrix with Dii = qi.

Our objective is to analyze the asymptotic test error of wˆ:

$$E_{t e s t}({\hat{w}})=\mathbb{P}(\operatorname{sign}(x^{\top}{\hat{w}})\neq y),$$
⊤wˆ) ̸= y), (13)
where (*x, y*) is a test example,

## 4.2. Asymptotic Behavior Of The Test Error.

We leverage random matrix theory (RMT) techniques
(Couillet & Liao, 2022; Liao & Mahoney, 2021; Firdoussi et al., 2024) to characterize the test error in Eq. (13). Our analysis is based on the spectral density of the resolvent matrix R in Eq. (12), allowing us to compute the first two moments of yx⊤wˆ for a test sample x and derive an expression for the test error. For simplicity, we assume an isotropic setup where Σ = Id and defer the general case to Appendix A. We shall work in the following so-called high-dimensional proportionate scaling regime

$$d,n\to\infty,\quad d/n\to\phi,\qquad\qquad(14)$$
$$(11)$$

in which the input-dimension d and the sample size n diverge to infinity at the same rate. The scalar ϕ ∈ (0, ∞) captures the effective dimensionality or over-parametrization rate of the problem.

Key Scalars. WLOG, assume ∥ws∥ = 1. It turns out that the for fixed, pruning, p, the asymptotic test error is fully captured by the following scalars:

$${}^{*}\ \ (12)$$
$$\rho:=w_{s}^{\top}w_{0}/\|w_{0}\|,\,\tau:=\frac{\rho}{\sqrt{1-\rho^{2}}},\,\gamma:=\mathbb{E}[q(G)G^{2}],$$ $$\beta:=2\mathbb{E}[q(G)\varphi(\tau G)],\quad\tilde{\beta}:=2\mathbb{E}[q(G)\Phi(\tau G)G],\tag{15}$$
$$(13)^{\frac{1}{3}}$$

where G ∼ N (0, 1) with pdf φ and cdf Φ. Note that ρ quantifies the alignment between the pruning direction ws and the ground-truth labeler w0, while β and γ capture statistical properties of the pruning strategy q. Spectral functions. The Stieltjes transform m of the limiting spectral density of the resolvent matrix R is shown in

2 9 2 10 Size of Selected Dataset 55 60 65 70 75 80 85 90 Ac cu ra cy Select top 10% Select top 50% Select top 80% Select top 90% Select all
Lemma 3 to be given by the exact formula (with z := −λ)

$$m(z)=\frac{p-\phi-z-\sqrt{(p-\phi-z)^{2}-4\phi z}}{2\phi z},\tag{16}$$

and will play an important role in our theory. The above formula represents a somewhat distorted Marchenko-Pastur law. Indeed, the classical MP (Marcenko & Pastur ˇ , 1967)
corresponds to p → 1 (i.e. no data pruning).

We further define the following auxiliary functions:

$$s(z):=\frac{\gamma}{1+\phi m(z)},\quad\tilde{m}(z):=\frac{1}{s(z)-z},$$ $$r(z):=\omega^{2}\cdot m(z)+\tilde{\omega}^{2}\cdot\tilde{m}(z),\tag{17}$$  with $\omega:=\sqrt{1-\rho^{2}}\beta,\quad\tilde{\omega}:=\rho\tilde{\beta}$.  
Main Result: Test Error Scaling w.r.t Selection Strategy. Theorem 1. In the limit Eq. (14), the classification test error satisfies: E*test*( ˆw) → arccos |m0|/
√ν0/π*, where*

$$m_{0}:=\omega m(-\lambda)+\tilde{\omega}\tilde{m}(-\lambda),$$  $$\nu_{0}:=p\phi m^{\prime}(-\lambda)+r^{\prime}(-\lambda)-\frac{2\phi m^{\prime}(-\lambda)}{1+\phi m(-\lambda)r(-\lambda)}.$$

The scaling behavior of test error is fully determined by the six scalars (*λ, ϕ, p, ρ, γ, β,* β˜). Importantly, the choice of the data point selection strategy i 7→ q(x
⊤
i ws) only influences performance through ρ, γ, β, and β˜.

4.2.1. EXAMPLE: SELECTING INFORMATIVE
 $\text{EXAMPLES}$. 
Consider a selection function of the form qi = q(x
⊤
i ws) for all i, where,

$$q(t):=1[|t|\leq\xi]=\begin{cases}1,&\text{if}|t|\leq\xi,\\ 0,&\text{else,}\end{cases}\tag{18}$$

for some threshold ξ ≥ 0. Such selection strategy selects only the examples near the decision boundary of ws, analogous to using classifier entropy as a selection criterion but simpler to study. Lemma 1 and 2 derive explicit expressions for (*γ, β,* β˜). Figure 3 presents theoretical predictions for test accuracy across different degrees of example selection, showing that selecting hard examples improves scaling laws, reducing the number of training samples needed for the same performance. However, beyond a certain point, excessive pruning degrades performance, as illustrated in Figure 5.

## 4.2.2. Adaptive Selection Strategy.

Data selection relies on a pruning direction ws to select informative/hard examples: i 7→ q(x
⊤
i ws) ∈ {0, 1}, but these examples are ultimately used to train wˆ. If ws and wˆ
are misaligned, what is considered hard by ws may not be hard for wˆ, reducing the effectiveness of selective sampling. In fact, hard examples change over time: an example that was identified hard, might not remain hard are more training is done. To ensure alignment, ws should periodically update to reflect the evolving decision boundary of wˆ. This adaptive selection mechanism motivates the continuous data generation process of DP, as presented in Section 3.

Data selection relies on a pruning direction ws to identify informative or hard examples: i 7→ q(x
⊤
i ws) ∈ {0, 1}.

However, these selected examples are ultimately used to train wˆ, and if ws and wˆ are misaligned, what is considered hard by ws may not be hard for wˆ, reducing the effectiveness of selective sampling. In fact, ws and wˆ deviate from each other the more wˆ is trained on these examples. Moreover, the definition of "hard" changes over time—an example that was initially difficult may become easier as training progresses. To maintain alignment, ws should be periodically updated to reflect the evolving decision boundary of wˆ. This adaptive selection mechanism underpins the continuous data generation process in DP, as presented in Section 3.

## 5. Experiments

For all the experiments, we use the LDM1.5 (Rombach et al., 2022) as the pre-trained text-to-image (T2I) model. We studied four different T2I models and found this model outperforming the rest. For more details see Appendix D.1.

Datasets. We validate our framework on two datasets.

ImageNet-100 (Tian et al., 2020; Sarıyıldız et al., 2023), a subset of ImageNet-1k (Deng et al., 2009), containing 100 classes and 5k validation examples, where the real validation set is used for evaluation and the real training set (126,689 examples) serves as a held-out test set. We also conduct experiment ImageNet-1k, using the 50k validation examples to monitor performance and reserving the real training set (1.3 million examples) as a held-out test set.

## 5.1. Scaling Laws Of Synthetic Data

We train a Vision Transformer (ViT-B) (Dosovitskiy et al., 2021) classifier with synthetic data. We study two scenarios: 1) Static data generation and 2) Deliberate Practice (DP). In all the experiments in this section we have a fixed and controlled setup. We train the models for 100k and 50k iterations for ImageNet-1k and ImageNet-100 respectively. For additional details, see Appendix D.5.

Static data generation. In this setup, all data is generated before training, and the classifier is trained on a fixed dataset.

We experiment with different dataset sizes to see its impact on accuracy. Deliberate Practice data generation. Hyperparameters ω and λ are tuned on ImageNet-100 and found effective for ImageNet-1k as well (see Section D.5 for details). We track validation accuracy throughout training and use it to determine when to generate new data, following a patiencebased criterion. To ensure the model has not over-fitted to the validation set, we also report accuracy on the full real training sets of ImageNet-100 and ImageNet-1k, used as held-out test sets. Figure 4 compares the scaling laws of the **Static** and Deliberate Practice (DP) on ImageNet-100 and ImageNet-1k. On both datasets, we note that DP scales well with dataset size and it consistently outperforms the Static setup, achieving higher validation accuracy at any given dataset size. On ImageNet-100 we observe that DP can reach the best accuracy of the static setup (with 3 million examples) using only 400k examples. This means that DP requires 7.5× less data to reach the same performance. On ImageNet-1k, we observe that DP can outperform the best accuracy of the static setup (with 13 million examples), using only 640k examples. This translates to DP requiring 20× less data to outperform the Static setup. For additional details on the hyper-parameters of these experiments, see Appendix D.5.1. Refer to Figure 13 for a visualization of how the dataset evolves from the start to the end of training.

## 5.2. Comparison With Previous Work

We compare DP with prior works on synthetic data generation for image classification (Sarıyıldız et al., 2023; Fan et al., 2024). Specifically, we evaluate setups that use classnames for prompting and publicly available models for sample generation. Performance is assessed on real ImageNet (held-out) training and validation sets, as well as on ImageNet-V2 (Recht et al., 2019), ImageNet-Sketch (Wang et al., 2019), ImageNet-R (Hendrycks et al., 2021a), and ImageNet-A (Hendrycks et al., 2021b) to measure out-ofdistribution (OOD) generalization. The results in Table 1 show that DP outperforms prior benchmarks on both ImageNet-100 and ImageNet-1k while requiring significantly less data and fewer training iterations. On ImageNet-100, DP generated 4.6 million fewer samples and trained for only one-sixth of the iterations compared to previous works, yet achieved superior performance on the real data. Similarly, on ImageNet-1k, DP reduced sample generation by 56.2 million and cut training iterations by over 30%, while still outperforming previous results. Furthermore, models trained with DP exhibit strong performance on out-of-distribution datasets, even surpassing models trained on real data on ImageNet-R and ImageNet- Sketch, with improvements of up to 15%.

## 5.3. Connection Between Pruning And Dp

In Section 2, we discussed how DP approximates direct sampling from a pruned distribution. Here, we validate this experimentally on ImageNet-100 using two setups:
1. **Oversampling then Pruning:** Generate a large pool and select high-entropy samples.

2. **Direct entropy-guided generation:** Generate only informative samples (a special case of DP with a single step of data addition).

We start with 130k generated samples (regular vanilla sampling), train for 17k iterations, then add a one-time additional 130k samples, increasing the total data size to 260k and training for an additional 33k iterations. In setup 1, we vary the pool size, ranging from no pruning (130k pool) up to an oversampling ratio of 18 (2.4M pool), selecting the top 130k high-entropy samples. In setup 2, we generate exactly 130k entropy-guided samples, varying the entropy-gauidance coefficient. Figure 5 (a, b) shows that both methods improve performance up to a point, after which excessive selection of high-entropy samples leads to degradation—likely due to selecting high-entropy but harmful outliers. This aligns with our theoretical predictions in Figure 5 (c). Regarding computational costs, generating a single image with entropy-guidance on an Nvidia H100 takes 1.82× longer than standard vanilla sampling. However, achieving similar performance through oversampling requires significantly more data, leading to a linear increase in cost.

50k 100k 200k 400k 800k 1.6M 3.2M
Total Data Size (log)
56 58 60 62 64 66 68 0.25M 0.5M 1M 2M 4M 8M 13M
Total Data Size (log)
40 42 44 46 48 50 Top
-1 A
cc ura cy Top
-1 A
cc ura cy DP
Static DP
Static
Table 1: **Comparison with previous work.** DP outperforms other models on both ImageNet-100 and ImageNet-1k while requiring significantly less data and fewer training iterations. Note that DP experiments reported in this table are trained longer than models reported in the previous section and, consistent with other work, use a smaller classifier free guidance scale of λ = 2.

Real IN-100 100k 130k 88.5 - **76.4** 37.1 60.8 **33.5** Syn. Static - (Sarıyıldız et al., 2023) IN-100 13k 130k 63.5 - 62.7 41.8 64.2 13.7 Syn. Static - (Sarıyıldız et al., 2023) IN-100 635k 6.5M 73.3 - 72.3 42.0 59.4 17.1 Syn. DP (ours) IN-100 100k 1.9M **74.3** 75.0 66.3 **52.0 76.6** 25.9 Real IN-1k 200k 1.3M 82.6 - **70.9** 32.5 44.6 **29.4** Syn. Static - (Sarıyıldız et al., 2023) IN-1k 130k 1.3M 42.9 - 43.0 16.6 26.3 3.6 Syn. Static - (Fan et al., 2024) IN-1k 210k 2M 50 - 42.2 27.2 45.7 6.6 Syn. Static - (Fan et al., 2024) IN-1k 315k 64M 54 - 46.0 32.4 52.5 9.4 Syn. DP (ours) IN-1k 200k 6.5M 54.1 54.84 48.5 34.7 56.0 12.3 Syn. DP (ours) IN-1k 200k 9.1M **55.1** 55.73 49.3 **36.0 57.2** 13.4

Task # Iters Data size IN real Val. IN real tr. IN-v2 IN-Sk IN-R IN-A

As a result, DP is 5× more efficient while also providing higher absolute improvements compared to pruning-based selection. See Figure 5 for details and Figure 11 for some visualizations.

## 5.4. The Evolution Of Hard Examples Over Time

"Does the sample hardness change as training progresses?"
To answer this question, Figure 6 tracks the error on examples that were misclassified at the time they were added. As expected, once introduced, the model gradually learns to classify them correctly. However, an interesting trend emerges: even before these examples were added, their error was lower than at the moment of inclusion. This suggests that the notion of hardness is dynamic—what is considered challenging at one point may become easier over time. Conversely, examples that were once easy might later become difficult due to shifts in the learned decision boundaries. This highlights a key limitation of static pruning approaches and underscores the importance of dynamically adapting the selection of informative examples throughout training, as done in Deliberate Practice (DP). See Figure 12 for some visualization of generations through training.

## 6. Related Work

Synthetic data for training neural networks. Synthetic data has become a powerful tool for training machine learning models across various domains. For instance, text-toimage diffusion models have been successfully used for visual representation learning (Astolfi et al., 2023; Li et al., 2025; Tian et al., 2024a;b; Sarıyıldız et al., 2023). However, limitations of synthetic data are highlighted by Fan et al. (2024), emphasizing the importance of generating more challenging and informative examples. Addressing distribution shifts between synthetic and real data, Hemmat et al. (2023) and Yuan et al. (2023) propose synthesizing training data that matches real data distributions or conditioning on real examples to reduce this gap. Expanding small-scale datasets has also been studied, see e.g. Zhang

(a) (b) (c) (d)
5x More tra ini ng Ite ratio ns
et al. (2024). Another related line of work involves using VLMs and LLMs to generate descriptions for augmenting datasets (Dunlap et al., 2023). Synthetic data is increasingly used to train (LLMs). For example, LLaMA3 (Grattafiori et al., 2024) employs AI- generated data for fine-tuning. Similarly, self-play approaches, e.g., Yuan et al. (2024), align with our framework by generating increasingly difficult examples for training. Continual learning and active learning. Our work is also closely related to principles from active learning (Bang et al.,
2024; Evans et al., 2023) and continual learning, which DP coefficient increases prioritize iterative model updates with tailored data. These methods highlight the importance of selecting informative samples based on the model's current state. (Sorscher et al., 2022) showed that pruning static datasets using metrics like margin scores can improve scaling laws by retaining the most informative examples, albeit in a non-adaptive manner. Challenges and risks of synthetic data. The challenges of training models on synthetic data, have gained significant attention. Dohmatob et al. (2024a;b) studied "model collapse", a phenomenon where iterative training on synthetic data degrades performance. They emphasize that data verification mechanisms can mitigate this risk and enable scaling with synthetic data. Similarly, our framework by generating informative examples through a dynamic loop, improves sample efficiency.

## 7. Conclusion

We introduced Deliberate Practice for Synthetic Data Generation, a framework that improves scaling laws by dynamically generating challenging and informative training examples. Unlike traditional methods that rely on static datasets, our approach approximates generating data directly from a pruned distribution, reducing inefficiencies and ensuring models continuously training on informative samples. We provided theoretical insights into the benefits of training on pruned distributions and empirically demonstrated that our method significantly improves performance while requiring fewer training iterations. Our results on ImageNet-100 and ImageNet-1K show that Deliberate Practice achieves superior accuracy with far less data and compute, outperforming previous state-of-the-art. Our work highlights the potential of structured synthetic data generation in advancing efficient and adaptive learning.

## Impact Statement

This work introduces a method for improving the sample efficiency of synthetic data generation through a deliberate practice framework that prioritizes the most informative training examples. By enabling more efficient use of generated data, our approach may reduce the computational and environmental costs associated with training large models, and make data generation more accessible in low-resource settings.

## References

Astolfi, P., Casanova, A., Verbeek, J., Vincent, P., Romero-
Soriano, A., and Drozdzal, M. Instance-conditioned gan data augmentation for representation learning. arXiv preprint arXiv:2303.09677, 2023.

Astolfi, P., Careil, M., Hall, M., Manas, O., Muck- ˜
ley, M., Verbeek, J., Soriano, A. R., and Drozdzal, M. Consistency-diversity-realism pareto fronts of conditional image generative models. arXiv preprint arXiv:2406.10429, 2024.

Bang, J., Ahn, S., and Lee, J.-G. Active prompt learning in vision language models. In CVPR, 2024.

Couillet, R. and Liao, Z. *Random Matrix Methods for* Machine Learning. Cambridge University Press, 2022.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Dohmatob, E., Feng, Y., Subramonian, A., and Kempe, J.

Strong model collapse. *arXiv preprint arXiv:2410.04840*, 2024a.

Dohmatob, E., Feng, Y., Yang, P., Charton, F., and Kempe, J. A tale of tails: Model collapse as a change of scaling laws. *arXiv preprint arXiv:2402.07043*, 2024b.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16×16 words: Transformers for image recognition at scale. In ICLR, 2021.

Dunlap, L., Umino, A., Zhang, H., Yang, J., Gonzalez, J. E., and Darrell, T. Diversify your vision datasets with automatic diffusion-based augmentation. In *NeurIPS*,
2023.

Ericsson, K. A., Krampe, R. T., and Tesch-Romer, C. The ¨
role of deliberate practice in the acquisition of expert performance. *Psychological review*, 100(3):363, 1993.

Evans, T., Pathak, S., Merzic, H., Schwarz, J., Tanno, R.,
and Henaff, O. J. Bad students make great teachers: Active learning accelerates large-scale visual understanding. arXiv preprint, 2312.05328, 2023.

Fan, L., Chen, K., Krishnan, D., Katabi, D., Isola, P., and Tian, Y. Scaling laws of synthetic images for model training... for now. In CVPR, 2024.

Feng, Y., Dohmatob, E., Yang, P., Charton, F., and Kempe, J.

Beyond model collapse: Scaling up with synthesized data requires reinforcement, 2024. URL https://arxiv. org/abs/2406.07515.

Firdoussi, A. E., Seddik, M. E. A., Hayou, S., Alami, R.,
Alzubaidi, A., and Hacid, H. Maximizing the potential of synthetic data: Insights from random matrix theory, 2024.

Grattafiori, A. et al. The llama 3 herd of models, 2024. URL
https://arxiv.org/abs/2407.21783.

Hemmat, R. A., Pezeshki, M., Bordes, F., Drozdzal, M., and Romero-Soriano, A. Feedback-guided data synthesis for imbalanced classification. *arXiv preprint*, 2310.00158, 2023.

Hendrycks, D., Basart, S., Mu, N., Kadavath, S., Wang, F.,
Dorundo, E., Desai, R., Zhu, T., Parajuli, S., Guo, M.,
et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 8340–8349, 2021a.

Hendrycks, D., Zhao, K., Basart, S., Steinhardt, J., and Song, D. Natural adversarial examples. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15262–15271, 2021b.

Ho, J. and Salimans, T. Classifier-free diffusion guidance.

arXiv preprint arXiv:2207.12598, 2022.

Hu, S., Tu, Y., Han, X., He, C., Cui, G., Long, X., Zheng, Z.,
Fang, Y., Huang, Y., Zhao, W., et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. *arXiv preprint*, 2404.06395, 2024.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al. Overcoming catastrophic forgetting in neural networks. *Proceedings of the national* academy of sciences, 114(13):3521–3526, 2017.

Kolossov, G., Montanari, A., and Tandon, P. Towards a statistical theory of data selection under weak supervision.

In *The Twelfth International Conference on Learning* Representations, 2024. URL https://openreview.

net/forum?id=HhfcNgQn6p.

Li, X., Yang, Y., Li, X., Wu, J., Yu, Y., Ghanem, B., and Zhang, M. Genview: Enhancing view quality with pretrained generative model for self-supervised learning. In European Conference on Computer Vision, pp. 306–325. Springer, 2025.

Liao, Z. and Mahoney, M. W. Hessian eigenspectra of more realistic nonlinear models. In Advances in Neural Information Processing Systems, volume 34. Curran Associates, Inc., 2021.

Marcenko, V. A. and Pastur, L. A. Distribution of eigenval- ˇ
ues for some sets of random matrices. *Mathematics of* the USSR-Sbornik, 1(4):457, apr 1967.

Oksendal, B. Stochastic differential equations: an introduction with applications. Springer Science & Business Media, 2013.

Recht, B., Roelofs, R., Schmidt, L., and Shankar, V. Do imagenet classifiers generalize to imagenet? In International conference on machine learning, pp. 5389–5400.

PMLR, 2019.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In *CVPR*, 2022.

Sarıyıldız, M. B., Alahari, K., Larlus, D., and Kalantidis, Y. Fake it till you make it: Learning transferable representations from synthetic imagenet clones. In *CVPR*,
2023.

Settles, B. Active learning literature survey. 2009. Shin, J., Kang, M., and Park, J. Fill-up: Balancing long-tailed data with generative models. *arXiv preprint*, 2306.07200, 2023.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. *arXiv preprint*, 2010.02502, 2020.

Song, Y. and Ermon, S. Generative modeling by estimating gradients of the data distribution. *Advances in neural* information processing systems, 32, 2019.

Sorscher, B., Geirhos, R., Shekhar, S., Ganguli, S., and Morcos, A. Beyond neural scaling laws: beating power law scaling via data pruning. Advances in Neural Information Processing Systems, 35:19523–19536, 2022.

Tian, Y., Krishnan, D., and Isola, P. Contrastive multiview coding. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XI 16, pp. 776–794. Springer, 2020.

Tian, Y., Fan, L., Chen, K., Katabi, D., Krishnan, D., and Isola, P. Learning vision from models rivals learning vision from data. In *CVPR*, 2024a.

Tian, Y., Fan, L., Isola, P., Chang, H., and Krishnan, D.

Stablerep: Synthetic images from text-to-image models make strong visual representation learners. In *NeurIPS*, 2024b.

Wang, H., Ge, S., Lipton, Z., and Xing, E. P. Learning robust global representations by penalizing local predictive power. *Advances in Neural Information Processing* Systems, 32, 2019.

Yuan, H., Chen, Z., Ji, K., and Gu, Q. Self-play fine-tuning of diffusion models for text-to-image generation. arXiv preprint, 2402.10210, 2024.

Yuan, J., Zhang, J., Sun, S., Torr, P., and Zhao, B. Realfake: Effective training data synthesis through distribution matching. *arXiv preprint*, 2310.10402, 2023.

Zhang, Y., Zhou, D., Hooi, B., Wang, K., and Feng, J.

Expanding small-scale datasets with guided imagination.

In *NeurIPS*, 2024.

## A. Further Theoretical Analysis And Proofs A.1. The Unregularized Regime

We now consider our theory in the limit λ → 0
+. Thus, the parameter vector for the classifier is the least-squares estimate for w0, i.e wˆ = ˆwLS = X′†Y
′. Recall the definition of the constants γ, β, β˜, ω, and ω˜ from equations (17). Recall that p ∈ (0, 1] is the proportion of training data left after pruning the original dataset (*X, Y* ) containing n examples. We have the following important corollary to Theorem 1.

Corollary 1. In the (ordered) limit n, d → ∞, d/n → *ϕ, λ* → 0
+, it holds that E*test*( ˆw) → arccos(|a|/
√b)/π*, where the* constants a and b are given as follows: (A) If ϕ < p*, then*

$$a:=(\omega+\tilde{\omega}p/\gamma)/(p-\phi),\qquad b:=\frac{p^{2}\phi+(r_{0}^{\prime}-2\phi r_{0})}{(p-\phi)^{3}},$$  _with $r_{0}:=\beta^{2}+\tilde{\beta}^{2}p/\gamma,\quad r_{0}^{\prime}:=p\cdot\left(\beta^{2}+\tilde{\beta}^{2}\cdot((p-\phi)p/\gamma^{2}+\phi/\gamma)\right).$_
(B) If ϕ > p*, then*

$$a:=(\omega+\tilde{\omega}/c_{1})c_{0},\quad b:=(p\phi-r_{0})c_{0},$$  _with $c_{0}:=1-p/\phi,\quad c_{1}:=1-(p-\gamma)/\phi,\quad r_{0}:=\beta^{2}+\tilde{\beta}^{2}/c_{1}$._
2 + β˜2/c1. (22)
Classifi cation e rror n = 250 n = 600

$$(19)$$
$$(20)$$
$$(21)$$
$$(22)$$

n = 1000 0.0 0.2 0.4 0.6 0.8 1.0 p 0.1 0.2 0.3 0.4 0.5 keep easy keep hard 0.0 0.2 0.4 0.6 0.8 1.0 p 0.0 0.2 0.4 0.6 0.8 1.0 p
(a) Regularization parameter λ = 10−6.

Classifi cation e rror n = 250 n = 600 n = 1000 0.0 0.2 0.4 0.6 0.8 1.0 p 0.1 0.2 0.3 0.4 0.5 keep easy keep hard 0.0 0.2 0.4 0.6 0.8 1.0 p 0.0 0.2 0.4 0.6 0.8 1.0 p
Figure 7: Empirical verification of Theorem 1 **and Corollary** 1. For this experiment, the input dimension is d = 350, and each subplot corresponds to a different value of the original sample size n. The experiment for λ = 10−6is a proxy for the unregularized case λ → 0
+. Solid lines correspond to observed values of the test error Etest( ˆw), while broken lines are the theoretical prediction of Theorem 1 (**bottom row**) and Corollary 1 (**top row**). Notice the excellent match between the experimental results and our theory. Also, observe the multiple-descent patterns, reminiscent of a non-trivial effect of different pruning strategies in different regimes of the pruned training dataset size n0 = np; the vertical line corresponds to an interpolation threshold at p = ϕ, i.e., n0 = d.

The result is empirically verified in Figure 7(a). A.2. Some Important Examples of Pruning Strategies Keep Hard Examples (KH). Consider the case where the pruning strategy is given by qi = qKH(x
⊤
i ws) for all i, where

$$q_{KH}(t):=1[|t|\leq\xi]=\begin{cases}1,&\text{if}|t|\leq\xi,\\ 0,&\text{else,}\end{cases}\tag{1}$$
$$(23)$$

for some ξ ≥ 0. Define α := ξ/∥ws∥. We have explicit formula for the constants β and β˜ appearing in Theorem 1. Viz, Lemma 1. *With* τ := ρ/p1 − ρ 2, ϵ1 := 2Φ(α/p1 − ρ 2) − 1*, and* ϵ2 := 2Φ(τα) − 1*, it holds that*

$$\tilde{\beta}(q_{K H})=2(\rho\varphi(0)\epsilon_{1}-\varphi(\alpha)\epsilon_{2}),\quad\beta(q_{K H})=2\varphi(0)\sqrt{1-\rho^{2}}\cdot\epsilon_{1}.$$
2 · ϵ1. (24)
Example 2: Keep Easy Examples (KE). Here, the pruning strategy is qi = qKE(x
⊤
i ws), where

$$(24)$$

$$q_{K E}(t):=1[|t|>\xi]={\begin{cases}0,\\ 1,\end{cases}}$$
qKE(t) := 1[|t| > ξ] = (0, if |t| ≤ ξ,
if $|t|\leq\xi$,  else.  
Lemma 2. *With* τ := ρ/p1 − ρ
$$\overline{{{-\rho^{2}}}},\,\epsilon_{1}:=2(1-\Phi(\alpha/\sqrt{1-\rho^{2}})),\,\epsilon_{2}:=2\Phi(\tau\alpha)-1,\,i t\,h o l d s\,t h a t.$$
$$\tilde{\beta}(q_{K E})=2(\rho\varphi(0)\epsilon_{1}+\varphi(\alpha)\epsilon_{2}),\quad\beta(q_{K E})=2\varphi(0)\sqrt{1-\rho^{2}}\cdot\epsilon_{1}.$$
2 · ϵ1. (26)
Example 3: Interpolation between Keep Hard and Keep Easy Strategies. Consider the following pruning strategy proposed in (Kolossov et al., 2024)

$$(25)$$

$$(26)$$
$$q(t)\propto\sigma(t)^{\omega}(1-\sigma(t))^{\omega},$$
$$(27)$$

ω(1 − σ(t))ω, (27)
for some tuning parameter ω. Here, σ is the sigmoid function. We can associate q(x
⊤
i ws) with the probability the auxiliary classifier x 7→ *sign*(x
⊤ws) assigns to an example xi. Thus, positive values of ω correspond to keeping examples considered uncertain (i.e hard) by this classifier, while negative values correspond to examples considered easy.

## A.3. Main Ingredients Of Proofs

A.3.1. DETERMINISTIC EQUIVALENT FOR THE RESOLVENT MATRIX R
Definition 1 (Deterministic Equivalents). Given a sequence of random N × N matrices (RN )N , a deterministic equivalent thereof is a sequence of deterministic N × N matrices (RN )N *such that*

$$\operatorname{tr}A_{N}(R_{N}-{\overline{{R}}}_{N})\stackrel{a.s}{\to}0,$$
$$(28)$$

a.s → 0, (28)
for all sequences of N × N matrices (AN )N *with bounded Frobenious norm.* Let Π (resp. Π⊥ = Id − Π) be the projection onto the span (resp. orthogonal complement of the span) of ws. Define the following auxiliary vectors and scalars

$$v=\Sigma^{1/2}w_{s},\quad v_{1}=\frac{v^{\top}w_{s}}{\|w_{s}\|},\quad v_{\perp}=\Pi_{\perp}v.\tag{1}$$

Note that v⊥ is (d − 1)-dimensional and ∥v⊥∥ =p∥v∥
2 − v 2 1.

Henceforth we make the replacement z = −λ < 0, so that the resolvent matrix R now writes

$$R=R(z):=(X^{\top}D X/n-z I_{d})^{-1}.$$
−1. (30)
$$(29)$$
$$(30)^{\frac{1}{2}}$$

13 Let δ(z) be the unique positive solution to the fixed-point equation

$$\begin{array}{l l}{{}}&{{m(z)=d^{-1}\,\mathrm{tr}\,\bar{R}_{b}(z),\quad\delta(z)=n^{-1}\,\mathrm{tr}\,\Sigma\bar{R}_{b}(z),}}\\ {{}}&{{}}\\ {{\bar{R}_{b}(z)=\left(\mathbb{E}_{x\sim\mathcal{N}(0,\Sigma)}\,\left[\frac{q(x^{\top}w_{s})}{1+q(x^{\top}w_{s})\delta(z)}\right]\Sigma-z I_{d}\right)^{-1}.}}\end{array}$$
$$(31)$$

Note that the inner expectation evaluates to

$$(32)$$
$$\mathbb{E}_{x\sim{\mathcal{N}}(0,\Sigma)}\,\left[{\frac{q(x^{\top}w_{s})}{1+q(x^{\top}w_{s})\delta(z)}}\right]={\frac{p}{1+\delta(z)}}=:t(z),$$

and so R¯b(z) = (t(z)Σ − zId)
−1. Observe that R¯b(z)(t(z)Σ − zId) = Id, and so t(z)ΣR¯b(z) = Id + zR¯b(z). We deduce that

$$t(z)\delta(z)=n^{-1}\,\mathrm{tr}\,t(z)\Sigma\bar{R}_{b}(z)=n^{-1}\,\mathrm{tr}(I_{d}+z\bar{R}_{b}(z))=\phi\cdot(1+z m(z))$$

Thus the equations defining m(z) and δ(z) can be rewritten as

$$m(z)=d^{-1}\,\mathrm{tr}(t(z)\Sigma-zI_{d})^{-1},$$ $$t(z)=\frac{p}{1+\delta(z)},$$ $$\phi\cdot(1+zm(z))=t(z)\delta(z)=t(z)\left(\frac{p}{t(z)}-1\right)=p-t(z).$$
$$({\mathfrak{I}}{\mathfrak{I}}{\mathfrak{I}})$$
$$(34)$$
$$(35)$$

Solving for ϕzm(z) in terms of t(z) in the last equation gives

$$\phi z m(z)=\frac{p\delta(z)}{1+\delta(z)}-\phi=p-\phi-\frac{p}{1+\delta(z)}=p-\phi-t(z).$$

Plugging this into the first equation gives the following fixed-point equation for t(z)

$$p-\phi-t(z)=z n^{-1}\operatorname{tr}(t(z)\Sigma-z I_{d})^{-1}.$$
$$(36)$$
$$(37)$$  $$(38)$$
$$(39)$$
−1. (36)
The following result shows that R¯ is a deterministic equivalent for R.

Proposition 1. Recall the function t(z) *as the unique positive solution to the equation* (36)*. Then,*

$$R\simeq\bar{R},\ \mbox{with}\ \bar{R}=\Sigma^{-1/2}(\bar{m}(z)\Pi_{\perp}+\bar{m}(z)\Pi)\Sigma^{-1/2},$$  _where $\bar{m}(z)=\frac{1}{t(z)-z},\quad\bar{m}(z)=\frac{1}{s(z)-z},\quad s(z)=\frac{\gamma}{1+\delta(z)}=(\gamma/p)t(z),$_  $$\gamma:=\mathbb{E}[q(G)G^{2}],\ \mbox{for}\ G\sim\mathcal{N}(0,1).$$

## A.4. Isotropic Case

Consider the special case where the covariance matrix is Σ = Id. It is not hard to see that we must have m¯ (z) ≡ m(z) ≡ δ(z)/ϕ. Let us now compute m(z).

Lemma 3. For every z = −λ < 0, m(z) *is given by formula* (16).

Proof. Indeed, observe that in the isotropic case the equation (36) reduces to p − ϕ − t(z) = ϕz/(t(z) − z), or equivalently

$$0=\phi z+(t(z)-p+\phi)(t(z)-z)=t(z)^{2}-(p-\phi+z)t(z)$$

The discriminant of this quadratic equation evaluates to

$$(p-\phi+z)^{2}-4pz=(p-\phi-z+2z)^{2}-4pz$$ $$=(p-\phi-z)^{2}+4z^{2}+4z(p-\phi-z)-4pz$$ $$=(p-\phi-z)^{2}-4\phi z,$$

and so because z = −λ < 0, the positive solution is

$$t(z)={\frac{p-\phi+z+{\sqrt{(p-\phi-z)^{2}-4\phi z}}}{2}}.$$
$$(40)^{\frac{1}{2}}$$
2. (40)
We deduce that

$$m(z)=\frac{1}{t(z)-z}=\left(\frac{p-\phi-z+\sqrt{(p-\phi-z)^{2}-4\phi z}}{2}\right)^{-1}$$ $$=2\cdot\frac{p-\phi-z-\sqrt{(p-\phi-z)^{2}-4\phi z}}{(p-\phi-z)-((p-\phi-z)^{2}-4\phi z)}$$ $$=\frac{p-\phi-z-\sqrt{(p-\phi-z)^{2}-4\phi z}}{2\phi z},$$

which is precisely the claimed formula given in (16). The following result then follows directly from Proposition 1. Corollary 2. *In the isotropic setting, we have the following deterministic equivalents:*

$\square$
$$\begin{array}{l}{(41)}\\ {(42)}\end{array}$$
$$(43)$$
R ≃ R, ¯ *with* R¯ = m(z)Π⊥ + s(z)Π, (41)
$$\begin{array}{l}{{R\simeq\bar{R},\;\mathrm{with}\;\bar{R}=m(z)\Pi_{\perp}+s(z)\Pi,}}\\ {{R^{2}\simeq m^{\prime}(z)\Pi_{\perp}+\tilde{m}^{\prime}(z)\Pi.}}\end{array}$$
2 ≃ m′(z)Π⊥ + ˜m′(z)Π. (42)
where m˜ (z) := 1/(s(z) − z), s(z) = γ/(1 + ϕm(z)), and γ ≥ 0 *is as given in* (43).

$$\rho=\frac{w_{s}^{\top}w_{0}}{\|w_{s}\|\|w_{0}\|},\ \beta:=\mathbb{E}\left[q(\|w_{s}\|G_{2})|G_{1}|\right],\ \gamma:=\mathbb{E}\left[q(\|w_{s}\|G_{1})G_{1}^{2}\right],\tag{1}$$

## A.5. Test Error Representation ("Scaling Laws")

We are now ready to state our main theoretical results, which is a generalization of Theorem 1. Remark 1. *For simplicity of presentation, all our theoretical results only consider symmetric pruning strategies for which* q(−t) ≡ q(t). This includes the "keep hard" and "keep easy" pruning strategies considered in (Sorscher et al., *2022).* Proposition 2. *Define the following quantities:*

$$m:=\frac{m_{0}}{1+\delta},\quad m_{0}:=\frac{c^{\top}\bar{R}\Sigma w_{0}}{\|\Sigma^{1/2}w_{0}\|}$$  $$\nu:=\frac{\nu_{0}}{(1+\delta)^{2}},\quad\nu_{0}:=\frac{p}{n}\,\mathrm{tr}\,\Sigma\Sigma^{\prime}+c^{\top}\Sigma^{\prime}c-\frac{2c^{\top}\bar{R}c}{1+\delta}\,\mathrm{tr}\,\Sigma\Sigma^{\prime},$$  _with $c:=\mathbb{E}[q_{i}y_{i}x_{i}]=\mathbb{E}_{(x,y)\sim P}[q(x^{\top}w_{s})yx],\quad\Sigma^{\prime}:=\mathbb{E}\,[R\Sigma R]$._
Then, in the limit (14), the test error of wˆ *is given by*

$$E_{t e s t}(\hat{w})\to\frac{1}{\pi}\operatorname{arccos}\left(|m_{0}|/\sqrt{\nu_{0}}\right).\tag{1}$$

## B. Proof Of Proposition 2

The proof follows standard (Couillet & Liao, 2022; Firdoussi et al., 2024) "leave-one-out" techniques which are now standard for analyses based on random matrix theory.

$$(44)$$
(45)  $\binom{46}{45}$  . 
$$(47)$$

## B.1. Main Idea

For a random test point (*x, y*) ∼ P∗, we can write yx⊤wˆ = yz⊤Σ
1/2wˆ = *sign*(z
⊤Σ
1/2w0)z
⊤Σ
1/2w. ˆ
Write Σ
1/2wˆ = αΣ
1/2w0 +r, where r = Σ1/2wˆ−αΣ
1/2w0 and α ≥ 0 is to be determined. Observe that r is perpendicular to Σ
1/2w0 iff r
⊤Σ
1/2w0 = ˆw
⊤Σw0 − α∥Σ
1/2w0∥
2 = 0 iff

$$\alpha=\hat{w}^{\top}w_{0}/\|\Sigma^{1/2}w_{0}\|^{2}.$$
$$(48)$$
$$(49)$$
2. (48)
With this choice of α, one computes

$$y x^{\top}{\hat{w}}=\alpha y z^{\top}\Sigma^{1/2}w_{0}+y z^{\top}r.$$
1/2w0 + yz⊤r. (49)
Because r is perpendicular to Σ
1/2w0, we know that the above is a sum of two independent random variables.

For the first summand in (49), observe that

$$y z^{\top}\Sigma^{1/2}w_{0}=y x^{\top}w_{0}=s i g n(x^{\top}w_{0})x^{\top}w_{0}=|x^{\top}w_{0}|,$$

which has the same distribution as ∥Σ
1/2w0∥|G1| for G1 ∼ N(0, 1). For the second summand, it has the same distribution as distribution ∥r∥G2 where G2 ∼ N (0, 1) and ∥r∥
2 = ∥Σ
1/2wˆ∥
2 − α 2∥Σ
1/2w0∥
2. It follows that if α > 0,

Etest( ˆw) = Px,y(yx⊤wˆ ≤ 0) = P(α∥Σ 1/2w0∥|G1| + ∥r∥G2 ≤ 0) = P(α∥Σ 1/2w0∥|G1| + ∥r∥G2 ≤ 0, G2 < 0) = P(|G2/G1| ≥ η, G2 < 0) with η := α∥Σ 1/2w0∥/∥r∥ = P(G2 < 0)P(|T| ≥ η) with T := G2/G1 ∼ Cauchy(0, 1) = 1 2 · 2P(T ≥ η) = P(T ≥ η) = 1 − ( 1 2 + 1 π arctan η) = 1π (π/2 − arctan η) = 1 π arccos( η p1 + η 2 ) = 1π arccos(α∥Σ 1/2w0∥ ∥Σ1/2wˆ∥). Similarly, if α < 0, we get the same expression with α replaced by −α. Therefore, irrespective of α, we have
$$E_{t e s t}(\hat{w})=\frac{1}{\pi}\operatorname{arccos}(\frac{|\alpha||\Sigma^{1/2}w_{0}||}{||\Sigma^{1/2}\hat{w}||}).$$
∥Σ1/2wˆ∥). (50)
It remains to estimate the random quantities |α| and ∥Σ
1/2wˆ∥, in the asymptotic limit (14).

## B.2. Leave-One-Out Arguments

We start with the Woodbury identity tells us that

$$Rx_{i}=(X^{\top}DX/n+\lambda I_{d})^{-1}x_{i}=(\sum_{j=1}^{n}q_{j}x_{j}x_{j}^{\top}/n+\lambda I_{d})^{-1}x_{i}$$ $$=(R_{-i}^{-1}+q_{i}x_{i}x_{i}^{\top}/n)^{-1}x_{i}=\frac{R_{-i}x_{i}}{1+q_{i}x_{i}^{\top}R_{-i}x_{i}/n},$$
$$(50)$$

where R−i:= (n
−1 Pj̸=i qjxjx
⊤
j + λId)
−1is a version of the resolvent matrix constructed without the ith data point. This
"leave-one-out" trick is well-known in random matrix theory calculations.

On the other hand qix
⊤
i R−ixi/n concentrates around its mean which is

$$\mathbb{E}\left[q_{i}x_{i}^{\top}R_{-i}x_{i}/n\right]=\operatorname{tr}\left(\mathbb{E}[q_{i}x_{i}x_{i}^{\top}]R_{-i}/n\right)=\frac{\alpha}{n}\operatorname{tr}\Sigma R_{-i}\simeq\delta,$$ $$\text{with}\delta:=\frac{p}{n}\operatorname{tr}\Sigma\bar{R},\quad p:=\mathbb{E}[q_{i}].$$

Therefore, we have the following identities holding for every *i, j* ∈ [n] with i ̸= j:

$$Rx_{i}\simeq\frac{R_{-i}x_{i}}{1+\delta},$$ $$R_{-i}\simeq R_{-ij}-\frac{R_{-ij}x_{j}x_{j}^{\top}R_{-ij}}{1+\delta}.\tag{1}$$
$$(51)$$
$$(52)$$

Now, let x be a random test point from class y, independent of training data. For later use, note that

$$yx^{\top}\,\hat{w}=\frac{1}{n}\sum_{i=1}^{n}q_{i}y_{i}yx^{\top}Rx_{i}=\frac{1}{n}\sum_{i=1}^{n}q_{i}y_{i}yx^{\top}Rx_{i}$$ $$=\frac{1}{(1+\delta)n}\sum_{i=1}^{n}q_{i}y_{i}yx^{\top}R_{-i}x_{i}.$$
$$(53)$$

B.3. Asymptotics of ∥Σ
1/2wˆ∥
2 Note that ∥Σ
1/2wˆ∥
2 = Ex,y[(yx⊤wˆ)
2] = E[(x
⊤wˆ)
2]. Squaring (53) gives

$$(x^{\top}{\hat{w}})^{2}={\frac{1}{(1+\delta)^{2}n^{2}}}\sum_{i=1}^{n}q_{i}\cdot(x^{\top}R_{-i}x_{i})^{2}+{\frac{1}{(1+\delta)^{2}n^{2}}}\sum_{i\neq j}q_{i}q_{j}y_{i}y_{j}(x^{\top}R_{-i}x_{i})(x^{\top}R_{-j}x_{j})$$

For the expectation first some, note that

$${\frac{1}{n}}\mathbb{E}\left[q_{i}\cdot(x^{\top}R_{-i}x_{i})^{2}\right]={\frac{1}{n}}\mathbb{E}[q_{i}x^{\top}R_{-i}x_{i}x_{i}^{\top}R_{-i}x]={\frac{1}{n}}\operatorname{tr}\left(\mathbb{E}\left[x x^{\top}\right]\mathbb{E}\left[q_{i}R_{-i}x_{i}x_{i}^{\top}R_{-i}\right]\right)={\frac{p}{n}}\operatorname{tr}\Sigma\Sigma^{\prime},$$

with Σ
′:= E[RΣR]. We deduce that

$$\mathbb{E}\,\frac{1}{(1+\delta)^{2}n^{2}}\sum_{i=1}^{n}q_{i}\cdot(x^{\top}R_{-i}x_{i})^{2}=\frac{1}{(1+\delta)^{2}}\frac{p}{n}\operatorname{tr}\Sigma\mathbb{E}\left[R\Sigma R\right]$$ $$=\frac{p}{(1+\delta)^{2}}\cdot\begin{cases}n^{-1}\operatorname{tr}\mathbb{E}\left[R^{2}\right]\Sigma,&\text{if isotropic,}\\ \operatorname{hard\,life!},&\text{otherwise.}\end{cases}$$

Now, let *i, j* ∈ [n] with i ̸= j. One computes

E [qiqjyiyj · (x ⊤R−ixi)(x ⊤R−jxj )] = 1 1 + δ E -qiqjyiyjx ⊤ i TijΣTjixj, =1 1 + δ (A1 − A2 − A3 + A4), where Tij := R−ij − Sij/n, Sij := R−ijxjx ⊤ j R−ij 1 + δ, A1 := E [qiqjyiyjx ⊤ i R−ijΣR−ijxj ], A2 := 1 (1 + δ)n E [qiqjyiyjx ⊤ i SijΣR−ijxj ], A3 :=1 (1 + δ)n E [qiqjyiyjx ⊤ i R−ijΣSjixj ], A4 := 1 (1 + δ) 2n2 E [qiqjyiyjx ⊤ i SijΣSjixj ]
17 We now compute the terms A1, A2, A3, A4.

$$A_{1}=\mathbb{E}\left[q_{i}q_{j}y_{i}y_{j}x_{i}^{\top}R_{-ij}\Sigma R_{-ij}x_{j}\right]=\mathbb{E}\left[q_{i}q_{j}y_{i}y_{j}x_{i}^{\top}R\Sigma Rx_{j}\right]$$ $$=\operatorname{tr}\left(\mathbb{E}\left[\left(q_{j}y_{j}x_{j}\right)\left(q_{i}y_{i}x_{i}\right)^{\top}\right]\mathbb{E}\left[R\Sigma R\right]\right)=c^{\top}\Sigma^{\prime}c,$$  where $\Sigma^{\prime}:=\mathbb{E}[R\Sigma R]$.  
Similarly, A3 = A2 with

$$A_{2}=\mathbb{E}\left[q_{i}q_{j}y_{i}y_{j}x_{i}^{\top}S_{ij}\Sigma R_{-ij}x_{j}\right]=\frac{1}{(1+\delta)n}\mathbb{E}\left[q_{i}q_{j}y_{i}y_{j}x_{i}^{\top}R_{-ij}x_{j}x_{j}^{\top}R_{-ij}\Sigma R_{-ij}x_{j}\right]$$ $$=\frac{1}{(1+\delta)n}\operatorname{tr}\left(\mathbb{E}\left[q_{i}q_{j}y_{i}y_{j}x_{j}x_{i}^{\top}R_{-ij}x_{j}x_{j}^{\top}|\mathbb{E}\left[R_{-ij}\Sigma R_{-ij}\right]\right)\right].$$

Now, computes

$\mathbb{E}\left[q_{i}y_{i}q_{j}y_{j}x_{i}^{\top}R_{-ij}x_{j}\right]=\mathbb{E}\left[\left(q_{i}y_{i}x_{i}\right)^{\top}R_{-ij}\left(q_{j}y_{j}x_{j}\right)\right]=c^{\top}\mathbb{E}\left[R_{-ij}\right]c\simeq c^{\top}\mathbb{E}\left[R\right]c\simeq c^{\top}Rc,$  $\mathbb{E}\left[R_{-ij}\Sigma R_{-ij}\right]\simeq\mathbb{E}\left[R\Sigma R\right]=:\Sigma^{\prime},$
from which it follows that

$$A_{3}=A_{2}\simeq\frac{c^{\top}\bar{R}c}{1+\delta}\frac{1}{n}\,\mathrm{tr}\,\Sigma\Sigma^{\prime}.$$
$$(54)$$
tr ΣΣ′. (54)
Finally, it is easy to show that A4 = O(1/n) = o(1).

Putting things together, we deduce that

$$\mathbb{E}[\|\Sigma^{1/2}{\hat{w}}\|^{2}]\simeq\nu:={\frac{\nu_{0}}{(1+\delta)^{2}}},$$
, (55)
where ν0 ≥ 0 is as Proposition 2.

B.4. Asymptotics of α Proceeding as in the computation of the asymptotics of ∥Σ
1/2wˆ∥
2above, we can show that

$$\|\Sigma^{1/2}w_{0}\|^{4}\mathbb{E}\alpha^{2}=\mathbb{E}(\hat{w}^{\top}\Sigma w_{0})^{2}=\mathbb{E}\hat{w}^{\top}\Sigma w_{0}w_{0}^{\top}\Sigma\hat{w}\simeq\frac{1}{(1+\delta)^{2}}c^{\top}R\Sigma w_{0}w_{0}^{\top}\Sigma R c\simeq\frac{(c^{\top}\tilde{R}\Sigma w_{0})^{2}}{(1+\delta)^{2}}.$$

On the other hand,

$$\|\Sigma^{1/2}w_{0}\|^{2}\mathbb{E}\alpha=\mathbb{E}\hat{w}^{\top}\Sigma w_{0}\simeq\frac{1}{1+\delta}\mathbb{E}\frac{1}{n}\sum_{i}q_{i}y_{i}x_{i}^{\top}R_{-i}\Sigma w_{0}$$ $$\simeq\frac{1}{1+\delta}\mathbb{E}[q_{i}y_{i}x_{i}^{\top}R_{-i}\Sigma w_{0}]$$ $$=\frac{1}{1+\delta}\mathbb{E}[q_{i}y_{i}x_{i}]^{\top}\mathbb{E}[R_{-i}]\Sigma w_{0}$$ $$\simeq\frac{c^{\top}\bar{R}\Sigma w_{0}}{1+\delta}.$$

$$(55)$$

Thus, the variance of α is vanishing, and we deduce that

$$\alpha\simeq\mathbb{E}\alpha\simeq\frac{c^{\top}\bar{R}\Sigma w_{0}}{(1+\delta)\|\Sigma^{1/2}w_{0}\|^{2}}=:\frac{m_{0}}{(1+\delta)\|\Sigma^{1/2}w_{0}\|}\,,$$
, (56)
where m0 is as given Proposition 2.

$$(56)$$

## B.5. Final Step (Proof Of Proposition 2)

Combining (50), (55), and (56) completes the prove of Proposition 2.

## B.6. An Important Lemma

The following result computes the mean vectors µ and c.

Lemma 4. Let ρ ∈ [−1, 1] *be the cosine of the angle between* w¯s := Σ1/2ws and w¯0 := Σ1/2w0. Let u be the unit-vector in the direction of w¯s and let v be its completion to an orthonormal basis for the span of w¯s and w¯0 (if w¯s and w¯0 are parallel, i.e if ρ = ±1*, we simply set* v = 0).

$$\mu:=\mathbb{E}_{(x,y)\sim P}[y x],\quad c:=\mathbb{E}_{(x,y)\sim P}[q(x^{\top}w_{s})y x]$$
$$u d\,c={\bar{\beta}}u+\beta v,\,w h e r e$$

Then, µ =p2/π · Σw0/∥w0∥Σ, and c = βu˜ + βv*, where*

$\tilde{\beta}=\beta_{1}:=2\mathbb{E}\left[q(\|\bar{w}_{s}\|G)\Phi\left(\tau G\right)G\right],\quad\beta=\beta_{2}:=2\mathbb{E}\left[q(\|\bar{w}_{s}\|G)\varphi(\tau G)\right],\quad\text{with}G\sim\mathcal{N}(0,1).$
In particular, when ρ = ±1 *(i.e pruning along the data generator),*

$$(57)$$
$$(58)$$
$$\sim{\mathcal{N}}(0,1).$$
$$\beta_{1}=\mathbb{E}[q(\|\bar{w}_{s}\|G)|G]],\quad\beta_{2}=0.$$
$$(59)$$
β1 = E[q(∥w¯s∥G)|G|], β2 = 0. (59)
Proof. Observe that by instead considering Σ
−1/2µ, Σ
−1/2c, and defining v := Σ1/2ws and u := Σ1/2w0 when computing µ, and then u = Σ1/2w0 when computing c, we reduce the problem to the isotropic case x ∼ N (0, Id).

So let u = Σ1/2w0, and WLOG, assume u is aligned with the first canonical axis in R
d, i.e u = ∥u∥e1. Write x = (x1, x⊥)
and v = (v1, v⊥), where x⊥ := Pd j=2 xj ej ∈ R
d−1, and v⊥ := Pdj=2 vj ej ∈ R
d−1. It is clear that x
⊤u = ∥u∥x1, and x
⊤v = v1x1 + g, where g = x
⊤⊥v⊥. Furthermore, x1 and g are independent with distributions N (0, 1) and N (0, ∥v⊥∥
2)
respectively. It follows that

$$\Sigma^{-1/2}\mu=\mathbb{E}\left[sign(x^{\top}u)x\right]=\mathbb{E}\left[sign(\|u\|x_{1})x_{1}\right]e_{1}=\mathbb{E}\left[\|x_{1}\|\right]e_{1}$$ $$=\sqrt{\frac{2}{\pi}}e_{1}=\sqrt{\frac{2}{\pi}}\frac{u}{\|u\|}=\sqrt{\frac{2}{\pi}}\frac{\Sigma^{1/2}w_{0}}{\|w_{0}\|_{\Sigma}},$$

from which we deduce the prescribed formula for the vector µ. This proves the first part of the claim.

We now establish the formula c = β1u + β2v. The proof for the formula for µ follows a similar (but simpler) path.

Observe that by instead considering Σ
−1/2c, we reduce the problem to the isotropic case x ∼ N (0, Id). We can explicitly write

$$u=\frac{\bar{w}_{s}}{\|\bar{w}_{s}\|},\quad v=\frac{\Pi^{\perp}\bar{w}_{0}}{\|\Pi^{\perp}\bar{w}_{0}\|},\quad\rho=\frac{\bar{w}_{s}^{\top}\bar{w}_{0}}{\|\bar{w}_{s}\|\|\bar{w}_{0}\|},$$

where Π = uu⊤ and Π⊥ = Id − Π. One can decompose x = G1u + G2v + G⊥ and w¯0 = c1u + c2v + c⊥

$$\begin{array}{l l l}{{}}&{{G_{1}:=x^{\top}u,}}&{{G_{2}:=x^{\top}v,}}&{{G_{\bot}:=P^{\bot}x,}}\\ {{}}&{{}}&{{c_{1}:=w_{0}^{\top}u,}}&{{c_{2}:=x^{\top}v,}}&{{c_{\bot}:=}}&{{P^{\bot}\Sigma^{1/2}w_{0},}}\end{array}$$

where P is the projector onto the span of u and v. Note that G1, G2, and G⊥ forms a set of independent random variables.

Moreover, G1 and G2 have distribution N (0, 1), while G⊥ has distribution N (0, Id−2). We obtain

$$\mathbb{E}[q(x^{\top}w_{s})sign(x^{\top}w_{0})x]=\mathbb{E}\left[q(x^{\top}w_{s})sign(x^{\top}w_{0})x\right]=\mathbb{E}\left[q(x^{\top}w_{s})sign(x^{\top}w_{0})x\right]$$ $$=\mathbb{E}\left[q(\|w_{s}\|G_{1})sign(c_{1}G_{1}+c_{2}G_{2})G_{1}\right]\cdot u$$ $$\quad+\mathbb{E}\left[q(\|w_{s}\|G_{1})sign(c_{1}G_{1}+c_{2}G_{2})G_{2}\right]\cdot v$$ $$\quad+\mathbb{E}\left[q(\|w_{s}\|G_{1})sign(c_{1}G_{1}+c_{2}G_{2})G_{\perp}\right].$$
$$(60)$$
$$(61)$$
$$(62)$$
 $\left(63\right)$  $\left(64\right)$  $\left(65\right)$
Now, due independence, the third term decomposes as

$$\mathbb{E}\left[q(\|w_{s}\|_{\Sigma}\cdot G_{1})s i g n(c_{1}G_{1}+c_{2}G_{2})\right]\cdot\mathbb{E}\left[G_{\perp}\right]=0.$$

We deduce that

$$\mathbb{E}[q(x^{\top}w_{s})s i g n(x^{\top}w_{0})x]=\beta_{1}u+\beta_{2}v,$$

where β1 and β2 are as specified in the lemma and we have used the fact that

$$c_{1}/\|\bar{w}_{0}\|=\rho,\quad c_{2}/\|\bar{w}_{0}\|=\sqrt{1-\rho^{2}}.$$

In particular, if ρ = ±1 (meaning that w0 and ws are parallel), then

$$\beta_{k}=\mathbb{E}\left[s i g n(\pm G_{1})q(\|\bar{w}_{s}\|\cdot G_{1})G_{k}\right]={\begin{cases}\pm\beta,&{\mathrm{if}}\ k=1,\\ 0,&{\mathrm{otherwise}}.\end{cases}}$$
$$\quad(67)$$. 
We now compute the coefficients β1 and β2. Observe that thanks to Lemma 5, one has

$$\begin{array}{c}{{\mathbb{E}[s i g n(G_{3})\mid G_{1}]=\mathbb{E}[s i g n(\rho G_{1}+\sqrt{1-\rho^{2}}G_{2})\mid G_{1}]=2\Phi\left(\tau G_{1}\right)-1,}}\\ {{\mathbb{E}[s i g n(G_{3})G_{2})\mid G_{1}]=\mathbb{E}[s i g n(\rho G_{1}+\sqrt{1-\rho^{2}}G_{2})G_{2}\mid G_{1}]=2\varphi(\tau G_{1}).}}\end{array}$$

Therefore, with r := ∥w¯s∥, we have

$\beta_{1}:=\mathbb{E}[q(rG_{1})sign(G_{3})G_{1}]=2\mathbb{E}\left[q(rG_{1})\Phi\left(\tau G_{1}\right)G_{1}\right]-\mathbb{E}\left[q(rG_{1})G_{1}\right]=2\mathbb{E}\left[q(rG_{1})\Phi\left(\tau G_{1}\right)G_{1}\right],$  $\beta_{2}:=\mathbb{E}[q(rG_{1})sign(G_{3})G_{2}]=2\mathbb{E}\left[q(rG_{1})\varphi(\tau G_{1})\right],$
where we have used the oddness of the function t 7→ tq(rt) in the last equation on the first line.

Lemma 5. Let G ∼ N (0, 1), and let a, b ∈ R *with* a ̸= 0*. Then,*

$$\mathbb{E}[s i n g(a G+b)]=2\Phi(b/|a|)-1,\quad\mathbb{E}[s i n g(a G+b)G]=2\varphi(b/a).$$

Furthermore, it holds that

$$\operatorname*{lim}_{a\to0}\mathbb{E}[s i g n(a G+b)]=s i g n(b),\quad\operatorname*{lim}_{a\to0}\mathbb{E}[s i g n(a G+b)G]=0.$$
E[sign(aG + b)G] = 0. (69)
Proof. Indeed, one computes

$$\mathbb{E}[sign(aG+b)]=\mathbb{P}(aG+b>0)-\mathbb{P}(aG+b<0)=2\mathbb{P}(aG>-b)-1$$ $$=\begin{cases}2\mathbb{P}(G>-b/a)-1=2\Phi(b/a)-1,&\text{if}a>0,\\ 2\mathbb{P}(G<-b/a)-1=2\Phi(-b/a)-1,&\text{if}a<0.\end{cases}$$

We deduce that E[*sign*(aG + b)] = 2Φ(b/|a|) − 1 as claimed. C. Proof of Lemma 1 **and Lemma** 2
"Keep Hard" Examples (Lemma 1). Let b = τ , t =
√1 + b 2 =
√1 + τ 2 = 1/p1 − ρ 2. Using Lemma 4 and standard formulae1for the anti-derivative of the function z 7→ zφ(bz)φ(z)

$$\beta=\beta_{2}=2\mathbb{E}\left[q(rG)\varphi(\tau G)\right]=2\int_{-\alpha}^{\alpha}\varphi(\tau z)\varphi(z)\mathrm{d}z=\frac{2}{i}\varphi(0)\Phi(tz)\bigg{]}_{z=-\alpha}^{\alpha}$$ $$=2\sqrt{1-\rho^{2}}\varphi(0)\left(2\Phi(\alpha/\sqrt{1-\rho^{2}})-1\right)=2\varphi(0)\sqrt{1-\rho^{2}}\epsilon_{2}.$$  see Wikipedia https://en.wikipedia.org/wiki/List_of_integrals_of_Gaussian_fields.  
$$(68)$$
$$(69)$$
$\square$
functions.