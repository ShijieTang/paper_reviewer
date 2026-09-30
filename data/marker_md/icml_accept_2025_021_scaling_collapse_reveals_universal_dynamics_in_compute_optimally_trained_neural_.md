# Scaling Collapse Reveals Universal Dynamics in Compute-Optimally Trained Neural Networks

Shikai Qiu <sup>1</sup> † Lechao Xiao <sup>2</sup> Andrew Gordon Wilson <sup>1</sup> Jeffrey Pennington <sup>2</sup> Atish Agarwala <sup>2</sup>

## Abstract

What scaling limits govern neural network training dynamics when model size and training time grow in tandem? We show that despite the complex interactions between architecture, training algorithms, and data, compute-optimally trained models exhibit a remarkably precise universality. Specifically, loss curves from models of varying sizes collapse onto a single universal curve when training compute and loss are normalized to unity at the end of training. With learning rate decay, the collapse becomes so tight that differences in the normalized curves across models fall below the noise floor of individual loss curves across random seeds, a phenomenon we term *supercollapse*. We observe supercollapse across learning rate schedules, datasets, and architectures, including transformers trained on next-token prediction, and find it breaks down when hyperparameters are scaled suboptimally, providing a precise and practical indicator of good scaling. We explain these phenomena by connecting collapse to the power-law structure in typical neural scaling laws, and analyzing a simple yet surprisingly effective model of SGD noise dynamics that accurately predicts loss curves across various learning rate schedules and quantitatively explains the origin of supercollapse.

# 1. Introduction

As machine learning systems grow in scale, accurate predictive models of their training dynamics become increasingly valuable, both for interpreting costly experiments and for designing robust, efficient training pipelines [\(Wortsman et al.,](#page-11-0)

[2023;](#page-11-0) [Achiam et al.,](#page-9-0) [2023;](#page-9-0) [Xiao,](#page-11-1) [2024\)](#page-11-1). While the complexity of modern architectures, optimizers, and datasets often renders exact, first-principles analyses intractable for any individual model, recent work shows that some key aspects of training are predictable when we focus on their scaling behavior across a family of models. Examples include empirical power-law relations linking optimal final loss, model size, dataset size, and compute budget under computeoptimal training, known as neural scaling laws [\(Hestness](#page-10-0) [et al.,](#page-10-0) [2017;](#page-10-0) [Kaplan et al.,](#page-10-1) [2020;](#page-10-1) [Sharma & Kaplan,](#page-10-2) [2022;](#page-10-2) [Hoffmann et al.,](#page-10-3) [2022\)](#page-10-3), as well as hyperparameter transfer from small to large models based on infinite-width or depth limits of training dynamics [\(Yang et al.,](#page-11-2) [2021;](#page-11-2) [Bordelon](#page-9-1) [et al.,](#page-9-1) [2023;](#page-9-1) [Everett et al.,](#page-9-2) [2024;](#page-9-2) [Bordelon et al.,](#page-9-3) [2024c\)](#page-9-3).

In this work, we show the entire training process follows highly predictable scaling, beyond final losses and optimal hyperparameters. We find that the entire loss curves of compute-optimally trained models exhibit a precise scaling symmetry, collapsing onto a single universal curve across models after a simple normalization. Learning rate decay amplifies this effect dramatically, producing what we call *supercollapse*: collapse so tight that cross-scale differences fall below the noise floor of individual loss curves due to random seeds. Figure [1](#page-1-0) (a-d) summarizes these results.

These findings advance our understanding in two key ways. First, while [Kaplan et al.](#page-10-1) [\(2020,](#page-10-1) Figure 11) found the loss curves roughly follow a sum of power laws, we identify that loss curves follow a universal shape with far greater precision. For typical learning rate schedules, this shape deviates from simple power laws and may not admit any obvious functional form. Second, our work provides compelling empirical evidence for a well-defined joint scaling limit where model size and training time grow together under compute-optimal allocation. This limit contrasts with traditional infinite-width or depth limits that fix training duration [\(Yang & Hu,](#page-11-3) [2021;](#page-11-3) [Bordelon & Pehlevan,](#page-9-4) [2022\)](#page-9-4). While these theories predict initial dynamical consistency, accumulating finite-size effects lead to gradual divergence as training progresses, as demonstrated by [Vyas et al.](#page-11-4) [\(2023\)](#page-11-4). In contrast, the collapse we observe reveals a joint scaling limit that preserves consistency throughout training, precisely the regime relevant for practical large-scale training.

<sup>†</sup>Work done partly during an internship at Google Deep-Mind <sup>1</sup>New York University <sup>2</sup>Google DeepMind. Correspondence to: Shikai Qiu <sq2129@nyu.edu>, Atish Agarwala <thetish@google.com>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

![](_page_1_Figure_1.jpeg)

Figure 1: Scaling collapse of compute-optimal transformer loss curves and its explanation through a model of SGD noise dynamics. (a) Compute-optimal loss curves and fitted scaling law on CIFAR-5M, using a linear learning rate decay schedule. (b) Normalized reducible loss curves collapse onto a single universal curve independent of model size, with both final compute and reducible loss normalized to unity. (c) Collapse deviation ∆ (cross-model variation of normalized loss) falls below per-model noise floor σ (variation of reducible loss across random seeds) for much of training, a phenomenon we term *supercollapse*. (d) Supercollapse occurs during the decay phase of various learning rate schedules, each producing its own universal curve. To explain these phenomena, we show that a simple model of SGD noise dynamics (e) accurately predicts loss curves for different schedules across model scales (Section [3.2\)](#page-5-0) and (f) quantifies how learning rate decay improves the collapse due to the predicted scaling <sup>∆</sup> ∝ p <sup>η</sup>(1 − <sup>τ</sup>ˆ), where <sup>η</sup> is the instantaneous learning rate and τˆ is normalized gradient flow time (Section [3.3\)](#page-7-0). We observe supercollapse in other arhitectures and datasets (Figure [4\)](#page-3-0).

We provide an elementary theoretical analysis that reveals the key mechanisms behind this precise collapse. We first show that for loss curves following typical neural scaling laws, collapse occurs precisely when models are trained for constant multiples of their compute-optimal horizons (Section [3.1\)](#page-4-0). We then analyze a simple theoretical model of the SGD noise dynamics that predicts loss curves under a variety of learning rate schedules remarkably well (Section [3.2\)](#page-5-0), and explains two key observations: why normalized curves retain universal form despite losing their power-law structure, and how learning rate decay suppresses variance to produce supercollapse (Section [3.3\)](#page-7-0).

Beyond theoretical interest, supercollapse provides a practical scaling diagnostic, as we find that deviations from collapse can signal misconfigured scaling choices, such as suboptimal scaling of learning rate and data (Figure [4\)](#page-3-0). Overall, our results suggest supercollapse provides a novel, powerful tool to study scaling. Our code can be found [here.](https://github.com/shikaiqiu/supercollapse.git)

# 2. Empirical Observations

We demonstrate our main empirical findings in this section, independently on multiple tasks and architectures which can be studied even in academic settings.

### 2.1. Experiment Setup

In each task, we train a sequence of models with increasing compute, scaling hyperparameters such as data, initialization, and learning rate with the model. We refer to a sequence of training configurations as a scaling ladder. We provide further experimental details in Appendix [A.](#page-12-0) We focus on width scaling, where hyperparameter transfer is most well-studied, but find scaling transformer depth leads to similar results in Appendix [B,](#page-13-0) suggesting our observations may generalize to more general scaling ladders where width, depth, batch size, weight decay, etc. can be co-scaled.

Transformers Next-Token Prediction. We consider two next-token prediction tasks: 1) CIFAR-5M [\(Nakkiran et al.,](#page-10-4) [2020\)](#page-10-4), a dataset of 6M generated CIFAR-like images, and 2) Lichess, a collection of chess games recorded in algebraic chess notation where the goal is to predict the next move in the game. Our scaling ladder includes models with about 10M to 80M parameters, approximately log-uniformly spaced, by scaling the width (embedding dimensions) from 768 to 2048 and fixing the number of blocks to 3. All models use µP [\(Yang & Hu,](#page-11-3) [2021;](#page-11-3) [Yang et al.,](#page-11-2) [2021\)](#page-11-2) for initialization and learning rates, and are trained with Adam.

MLPs on Power-Law Fourier Features. To investigate

other architectures and training objectives, we train 7-layer MLPs with varying widths from 384 to 2048 on a synthetic regression task. The target function has a power-law Fourier spectrum, designed to elicit the power-law scaling laws observed in natural data. We count each example as 1 token.

### 2.2. Estimating Compute-Optimal Scaling Laws

Let L(t, p, ω) be the test loss after t tokens (proportional to steps) for a model with p parameters trained with random seed ω. We estimate the compute-optimal training horizon in tokens for a p-parameter model as t ⋆ (p) = (p/p0) γ , where γ is the data exponent, by extracting the Pareto frontier of expected loss (estimated using 5 seeds) vs. compute under a constant learning rate schedule, following a procedure similar to Approach 1 in [Hoffmann et al.](#page-10-3) [\(2022\)](#page-10-3), with compute estimated as c = 6tp FLOPs [\(Kaplan et al.,](#page-10-1) [2020\)](#page-10-1). We reuse the same t ⋆ (p) as the training horizon for other learning rate schedules, which prior work suggests is optimal up to a constant factor [\(Pearce & Song,](#page-10-5) [2024\)](#page-10-5). For each task and schedule, we fit the resulting compute-optimal scaling law using the form L<sup>0</sup> + ac−<sup>b</sup> (Figure [1a\)](#page-1-0), for constants <sup>L</sup>0, a, b ≥ <sup>0</sup>. Following [Sharma & Kaplan](#page-10-2) [\(2022\)](#page-10-2) and [Hoffmann et al.](#page-10-3) [\(2022\)](#page-10-3), we refer to L<sup>0</sup> as the estimated irreducible loss. Using the best-fit L0, we define the reducible loss curve L(t, p, ω) = <sup>L</sup>(t, p, ω) − <sup>L</sup>0. We detail the procedure for fitting the compute-optimal training horizon in Appendix [C.](#page-13-1) "Compute-optimal" here primarily refers to the choice of training horizon, not of all hyperparameters.

### 2.3. Scaling Collapse of Compute-Optimal Loss Curves

The loss curves for different model sizes cover varying ranges of compute and loss values, but appear to follow a consistent shape, which motivates us to affinely rescale them to the *normalized loss curve* ℓ given by

$$\ell(x, p, \omega) = \frac{L(xt^*(p), p, \omega) - \hat{L}}{L(t^*(p), p, \omega) - \hat{L}}, \quad x \in [0, 1], \quad (1)$$

![](_page_2_Figure_15.jpeg)

Figure 2: Subtracting irreducible loss leads to the best collapse. Setting Lˆ to values far from L<sup>0</sup> breaks the collapse on CIFAR-5M.

![](_page_2_Figure_1.jpeg)

Figure 3: Collapse with a constant LR schedule. (Left) Estimated mean and 90% confidence interval (shaded) of the normalized loss curves. (Right) ∆ is comparable to σ without LR decay.

for some offset L. ˆ We refer to x as the normalized compute. Note the denominator uses the stochastic final loss value specific to the random seed. We set Lˆ = L<sup>0</sup> to subtract the estimated irreducible loss that bottlenecks the asymptotic performance, leading to <sup>ℓ</sup>(x, p, ω) = <sup>L</sup>(xt<sup>⋆</sup> (p),p,ω) L(t <sup>⋆</sup>(p),p,ω) .

Remarkably, we observe that the family of normalized loss curves is nearly identical across p, revealing equal rates of relative progress (Figure [1b\)](#page-1-0). We say these curves *collapse*, as the phenomenon resembles the ubiquitous scaling collapse found in statistical physics, theoretical biology, and other sciences, where observables from systems of different sizes collapse onto a single curve after appropriate rescaling (see Appendix [D](#page-14-0) for further discussion). We found setting Lˆ = L<sup>0</sup> achieves the best collapse (Figure [2\)](#page-2-0).

### 2.4. Quantifying the Collapse Quality

We quantify the quality of collapse using the *collapse deviation* ∆, defined as:

$$\Delta(x) = \frac{\mathbb{V}_{p,\omega}[\ell(x, p, \omega)]^{1/2}}{\mathbb{E}_{p,\omega}[\ell(x, p, \omega)]}, \quad (2)$$

where Ep,ω and Vp,ω denote the expectation and variance over the random seed and the empirical distribution of model size p in the scaling ladder (approximately log-uniformly distributed). The collapse deviation measures the relative variation of the normalized curves across p. For perspective, we compare it to the per-model (relative) noise floor:

$$\sigma(x, p) = \frac{\mathbb{V}_\omega[\mathcal{L}(xt^*(p), p, \omega)]^{1/2}}{\mathbb{E}_\omega[\mathcal{L}(xt^*(p), p, \omega)]}, \quad (3)$$

which measures the relative fluctuation in the reducible loss curve for each model size p across random seeds.

By the definition of ℓ, ∆(1) = 0 always. As seen in Figure [3,](#page-2-1) for a constant learning rate, ∆(x) quickly rises to a level comparable to σ(x, p), and remains at that level for most x < 1. This observation shows that variations in the normalized curves arise primarily from seed-to-seed fluctuations rather than model-to-model differences, quantitatively demonstrating that the observed collapse is non-trivial.

![](_page_3_Figure_1.jpeg)

Figure 4: Collapse provides a practical indicator of good scaling, as suboptimally scaling key hyperparameters breaks the collapse. With the default setup, we observe supercollapse in MLP regression (b) and transformer trained on chess (e), but even changes that only lead to minor worsening in the scaling law can manifest as significant disruption to the collapse. (Top) Replacing µP with a constant learning rate cross models for MLPs. (Bottom) Increasing the data exponent γ from estimated compute-optimal value 1.02 to 1.2 for Transformers trained on chess. We perform a separate power-law fit to determine the value L<sup>0</sup> for each scaling ladder.

### 2.5. Supercollapse: Consistency Below the Noise Floor

Remarkably, with learning rate decay, we find that the collapse deviation is less than the noise floor for a significant fraction of training; that is, ∆(x) < σ(x, p) for x > <sup>1</sup> − <sup>δ</sup> for some moderate δ as large as 0.5 (Figure [1c\)](#page-1-0). We refer to this stronger form of collapse as *supercollapse* (in contrast to the collapse in Figure [3\)](#page-2-1). Supercollapse appears in the decay phase of all tested learning rate schedules that decay to zero (Figure [1d\)](#page-1-0). All schedules are defined in terms of relative training fraction, i.e., the learning rate is a fixed function of the normalized compute x across model sizes.

Under supercollapse, self-normalized loss curves from different models collapse better than our ability to predict any individual model's loss. Normalizing by the final loss of the particular realization of the stochastic loss curve is key to supercollapse, which reduces variance by exploiting correlations at different times along a single optimization trajectory. We explain this mechanism in detail in Section [3.3.](#page-7-0)

### 2.6. Suboptimal Scaling Breaks Supercollapse

Supercollapse provides a practical method for comparing inherently noisy training loss curves across model scales with precision that exceeds naive noise floor estimates, without the need for expensive multi-seed experiments typically required to obtain equally clean signals. This comparison can provide valuable diagnostic information about scaling where the ability to distinguish small signal from noise is often crucial [\(Xiao,](#page-11-1) [2024\)](#page-11-1), which we now demonstrate.

Model Parameterization. Carefully parameterizing the model, i.e., scaling the initialization, learning rate, and possibly other hyperparameters as model size increases, is crucial for achieving stable and efficient training at scale [\(Yang](#page-11-2) [et al.,](#page-11-2) [2021;](#page-11-2) [Bordelon et al.,](#page-9-1) [2023;](#page-9-1) [2024c;](#page-9-3) [Everett et al.,](#page-9-2) [2024\)](#page-9-2). When models are trained in the wrong parameterization, we expect the loss curves not to collapse due to a lack of consistent training dynamics across scales. Using the MLP setup, we show that replacing µP with a constant learning rate across widths breaks the collapse (Figure [4,](#page-3-0) top row). Remarkably, the normalized loss curves expose inconsistent dynamics even at small scales where the final losses are virtually identical between constant and µP scaling, demonstrating that the collapse is a more sensitive probe of scaling behavior than final performance alone.

Compute-Optimal Data Exponent. For language models, [Kaplan et al.](#page-10-1) [\(2020\)](#page-10-1) showed that compute-optimal training corresponds to training each model to a fixed multiple of its converged loss. If this principle generalizes to our setting, the data exponent γ should match the compute-optimal value for collapse to occur. For example, when γ exceeds the optimal value, larger models will make more rapid relative initial progress but decelerate later as a function of normalized compute, causing their normalized curves to shift downward. We indeed find this shift in Figure [4](#page-3-0) (bottom row). This sensitivity suggests a novel application: rather than fitting power laws to sparse points on the Pareto frontier, one could tune γ to maximize collapse quality, leveraging the full statistical power of entire loss curves.

![](_page_4_Figure_1.jpeg)

Figure 5: Scaling collapse from sum of power-law curves. (a) CIFAR-5M expected loss curves (averaged over 5 seed) without learning rate decay agree well with the sum-of-power-laws fit L(t, p) = L<sup>0</sup> + t <sup>−</sup><sup>µ</sup> + p −ν (constant multipliers not shown), a form commonly observed in natural data. We omit the first 1B tokens to avoid fitting the early time transients. (b) Simulated exact sum-of-power-laws loss curves show scaling collapse precisely when the data exponent γ is the theoretical compute-optimal value γ ⋆ . Small variations of γ around γ ⋆ lead to nearly negligible worsening in the resulting scaling law but dramatically disrupt the collapse.

# 3. Explaining Loss Curve Scaling Collapse

In this section, we investigate theoretical explanations for the scaling collapse of compute-optimal loss curves and supercollapse. Our analysis starts with a simple observation: the numerator of the collapse deviation ∆(x) can be decomposed as:

$$\mathbb{V}_{p,\omega}[\ell(x, p, \omega)] = \mathbb{V}_p\mathbb{E}_\omega[\ell(x, p, \omega)] + \mathbb{E}_p\mathbb{V}_\omega[\ell(x, p, \omega)]. \quad (4)$$

The first term corresponds to the variation between different scales p after averaging over all sources of randomness. We will first show how this term can be small:

- In Section [3.1,](#page-4-0) we prove that for a family of powerlaw neural scaling laws, compute-optimal loss curves indeed collapse after normalization. We show loss curves in our experiments fall into this family when using a constant learning rate schedule.
- In Section [3.2,](#page-5-0) we develop a simple theoretical model that successfully predicts the empirical loss curves under various learning rate schedules and explains why they collapse despite deviating from power laws. Given its effectiveness, we believe this model has value for understanding learning rate schedules more broadly.

We then analyze the second term, which captures the loss variance due to random seeds, averaged across model sizes:

- In Section [3.3,](#page-7-0) we show the same noise model enables us to reason about the noise in the loss curves, and quantitatively predict the variance reduction effect in supercollapse.

Together these findings provide an initial theoretical explanation for supercollapse, and uncover promising directions

for future theoretical work.

### 3.1. Scaling Collapse from Power-Law Scaling

In this section, we consider deterministic models of the loss curves and assume all randomness has been averaged out.

Power-Law Pareto Frontier is Necessary. For a family of differentiable loss curves L(t, p), the compute-optimal loss frontier after subtracting Lˆ must follow a power law for our affine transformation to induce scaling collapse (proof in Appendix [E\)](#page-14-1). The key insight is that collapse requires the transformed loss curves to be related by multiplicative scaling, equivalently translation in log-log space, where the frontier must have constant log-log slope since it remains tangent to shifted versions of the same curve. This motivates choosing Lˆ = L0, which by definition yields the best powerlaw Pareto frontier. However, a sufficient condition for scaling collapse requires an explicit form of L(t, p).

Neural Scaling Laws. Motivated by empirical neural scaling laws in natural data [\(Hestness et al.,](#page-10-0) [2017;](#page-10-0) [Kaplan et al.,](#page-10-1) [2020;](#page-10-1) [Hoffmann et al.,](#page-10-3) [2022\)](#page-10-3), we consider expected loss curves following a sum-of-power-laws scaling of the form

$$L(t, p) = L_0 + t^{-\mu} + p^{-\nu} \quad (5)$$

for constants <sup>L</sup><sup>0</sup> ≥ <sup>0</sup>, µ, ν > <sup>0</sup>, with potential constant multipliers absorbed via an appropriate choice of units. In Figure [5a,](#page-4-1) we show the CIFAR-5M loss curves are well-fit by Equation [\(5\)](#page-4-2) if trained under a constant learning rate schedule (averaged across 5 seeds). We also find decent fits in other datasets in Figure [11.](#page-18-0)

Equivalence by Balance of Power Laws. As before, let t ⋆ (p) denote the training horizon. We will examine conditions under which t ⋆ (p) (a) is compute-optimal, and (b) results in scaling collapse. We assume deterministic loss

curves for now and omit the argument ω. To find computeoptimal t ⋆ (p), we fix c so that t(p) = c/(6p) and minimize the loss L(t(p), p) = <sup>t</sup>(p) <sup>−</sup><sup>µ</sup> +p <sup>−</sup><sup>ν</sup> with respect to p by setting 0 = dL dp = ∂L ∂t dt dp + ∂L ∂p <sup>=</sup> −µt−µ−<sup>1</sup> (−t/p) − νp−ν−<sup>1</sup>

$$\implies \mu t^{-\mu} = \nu p^{-\nu} \quad (6)$$

which yields t ⋆ (p) = r <sup>−</sup>1/µp ν/µ, with r = ν/µ. Under this scaling, the normalized loss curves are:

$$\ell(x, p) = \frac{(xt^*)^{-\mu} + p^{-\nu}}{(t^*)^{-\mu} + p^{-\nu}} = \frac{rx^{-\mu} p^{\nu} + p^{\nu}}{rp^{\nu} + p^{\nu}} = \frac{rx^{-\mu} + 1}{r + 1} \quad (7)$$

.

All p dependencies cancel, leaving the final expression independent of p and giving us an exact collapse. Moreover, it is clear that this is the unique choice for t ⋆ (p) up to a constant multiplier that leads to such cancellation. This agreement is not an accident: compute-optimal scaling requires balancing the derivatives of two power laws, while collapse requires balancing the power laws themselves. For power laws, these two conditions coincide, up to a multiplicative constant.

In Figure [5b,](#page-4-1) we numerically verify the agreement between collapse and compute-optimal scaling. When the data exponent γ deviates from the optimal value ν/µ, we observe a suboptimal scaling law and no collapse. Note that the absence of an irreducible term in ℓ is also necessary. Had we set <sup>L</sup><sup>ˆ</sup> <sup>=</sup> <sup>L</sup><sup>0</sup> <sup>+</sup> <sup>E</sup> for some <sup>E</sup> ̸= 0 in Equation [\(1\)](#page-2-2), we would instead have <sup>ℓ</sup>(x, p) = (xt<sup>⋆</sup> ) <sup>−</sup>µ+p <sup>−</sup>ν+E (t <sup>⋆</sup>)<sup>−</sup>µ+p−ν+E , where no t ⋆ (p) can leave the numerator and denominator homogeneous in p.

In Appendix [F,](#page-16-0) we study the more general form

$$L(t, p) = L_0 + \sum_{i=1}^m a_i t^{-\mu_i} p^{-\nu_i}, \quad (8)$$

which naturally arises in theoretical models of neural scaling laws [\(Paquette et al.,](#page-10-6) [2024b;](#page-10-6) [Bordelon et al.,](#page-9-5) [2024a;](#page-9-5)[b\)](#page-9-6), and show that compute-optimality implies scaling collapse by balancing the two dominant terms, though with m > 2 the collapse is only exact asymptotically.

Together with the close empirical fit in Figure [5a,](#page-4-1) this analysis explains scaling collapse in the constant learning rate setting; however Equation [\(5\)](#page-4-2) fails to fit the empirical loss curves with most learning rate schedules, as varying the learning rate can modulate the loss curve in quite arbitrary ways, clearly shown in Figure [1d.](#page-1-0) Why, then, does the collapse transfer to other schedules?

### 3.2. Universality of Learning Rate Schedules

To understand why scaling collapse is robust across learning rate schedules, we develop a quantitative model for how learning rate schedules affect the loss curves. While an

exact theoretical model seems out of reach for the realistic training setup, we show that a simple model based on quadratic loss analysis proves surprisingly effective. Under this model, we demonstrate that although learning rate schedules deform the loss curves in a schedule-dependent way, the deformation is approximately independent of p. We consider stochastic effects that depend on the random seed ω, but omit ω as an explicit argument for brevity and use bar to denote expectation over ω.

### 3.2.1. A SIMPLE MODEL FOR LR SCHEDULES

Let w(t) and L(w(t)) denote the parameters and loss at step t, we can model the dynamics of full-batch gradient descent under a small learning rate η(t) with a gradient flow dw dt <sup>=</sup> −η(t)∇L(w(t)). To model stochastic effects, a noise term is added to the gradient, leading to the SDE dw dt <sup>=</sup> −η(t) ∇L(w) + Σ<sup>1</sup>/<sup>2</sup> (w)ξ(t) [\(Li et al.,](#page-10-7) [2017;](#page-10-7) [Malladi et al.,](#page-10-8) [2022\)](#page-10-8), where the *mini-batch* gradient noise Σ 1/2 (w)ξ(t) satisfies <sup>E</sup>[ξ(t)ξ(t ′ )] = <sup>δ</sup>(<sup>t</sup> − <sup>t</sup> ′ )I, and we allow its covariance (which depends on batch size) Σ(w) to be a function of the parameters. Prior works have used the SDE model or discrete variants to study learning rate schedules in analytically tractable problems [\(Zhang et al.,](#page-11-5) [2019;](#page-11-5) [d'Ascoli et al.,](#page-9-7) [2022;](#page-9-7) [Wen et al.,](#page-11-6) [2024\)](#page-11-6), but we will show it can make surprisingly accurate predictions in real models. We work in *gradient flow time* <sup>τ</sup> (t) = R <sup>t</sup> 0 η(s)ds, where

$$\frac{dw}{d\tau} = -\left(\nabla L(w) + \Sigma^{1/2}(w)\xi(\tau)\right), \quad (9)$$

and <sup>E</sup>[ξ(τ )ξ(τ ′ )] = <sup>δ</sup>(<sup>t</sup> − <sup>t</sup> ′ )<sup>I</sup> <sup>=</sup> <sup>η</sup>(<sup>τ</sup> )δ(<sup>τ</sup> − <sup>τ</sup> ′ )I. We overload the notation and use η(τ ), w(τ ), and L(τ ) to denote the evolution of these quantities in gradient flow time.

Quadratic Loss. For the moment, let us suppose the loss function is quadratic L(w) = <sup>1</sup> <sup>2</sup>w <sup>⊤</sup>Hw, where we assume the minimum is at the origin without loss of generality. Then ∇L(w) = Hw and standard calculation shows

$$w(\tau) = e^{-H\tau} w(0) - \int_0^\tau ds e^{-H(\tau-s)} \Sigma^{1/2}(w(s)) \xi(s). \quad (10)$$

Letting Σ( ¯ s) = <sup>E</sup>[Σ(w(s))], the expected loss is then

$$\begin{aligned} \bar{L}(\tau) &= \underbrace{\frac{1}{2}\mathbb{E}[\|e^{-H\tau}w(0)\|_H^2]}_{\mathcal{F}(\tau)} \\ &+ \underbrace{\frac{1}{2}\int_0^\tau ds \eta(s) \text{Tr}\left(He^{-2H(\tau-s)}\bar{\Sigma}(s)\right)}_{\mathcal{E}(\tau)}. \end{aligned} \quad (11)$$

The first term F(τ ) is the forcing function, equal to the expected loss curve in the noiseless limit <sup>η</sup><sup>Σ</sup> → <sup>0</sup> and is

![](_page_6_Figure_1.jpeg)

Figure 6: A simple model predicts Transformer loss curves trained across learning rate schedules, model sizes p, and training horizons T on CIFAR-5M. Dashed curves show the predicted loss according to Equation [\(18\)](#page-7-1), with α = 0.21, which closely match with the true curves in solid. Each curve is smoothed with an exponential moving average with half-life equal to 1% of total steps.

independent of the learning rate schedule. The second term E(<sup>τ</sup> ) is the excess loss due to SGD noise, which is a sum of exponential moving averages (up to normalization) of the gradient variance scaled by the learning rate over each eigenmode. Substituting in the specific forms for Σ recovers the convolutional Volterra equation for linear regression analyzed in [Paquette et al.](#page-10-9) [\(2021;](#page-10-9) [2024a\)](#page-10-10), or the noisy quadratic model in [Zhang et al.](#page-11-5) [\(2019\)](#page-11-5) for small learning rates.

If ηΣ¯ varies slowly compared to the timescale of the exponential moving average, we can make the approximation <sup>η</sup>(s)Σ( ¯ <sup>s</sup>) ≈ <sup>η</sup>(<sup>τ</sup> )Σ( ¯ <sup>τ</sup> ) inside the integrand, giving us:

$$\mathcal{E}(\tau) \approx \frac{1}{2} \eta(\tau) \text{Tr} \left( \bar{\Sigma}(\tau) H \int_0^\tau ds e^{-2H(\tau-s)} \right) \quad (12)$$

$$= \frac{1}{4} \eta(\tau) \text{Tr}(\bar{\Sigma}(\tau)(1 - e^{-2H\tau})). \quad (13)$$

For large τ the expected loss is then approximately

$$\bar{L}(\tau) \approx \mathcal{F}(\tau) + \frac{1}{4}\eta(\tau) \text{Tr}(\bar{\Sigma}(\tau)). \quad (14)$$

Given access to Tr Σ( ¯ τ ) , we can derive a prediction for how the loss changes as we change the learning rate schedule without knowing F.

General Case. In Appendix [G,](#page-18-1) we discuss how this analysis can be generalized to more realistic setups. For general loss functions, we show via perturbation theory that, to first order in ηΣ¯, one can make similar approximations to derive Equation [\(14\)](#page-6-0) given an additional assumption that the Hessian is slowly varying, and with the forcing function F(τ ) no longer admitting a quadratic form. We also show in Appendix [G](#page-18-1) that Σ should be the *preconditioned* gradient covariance when using adaptive optimizers. We absorb the layerwise, width-dependent learning rates in µP into the preconditioner, similar to [Noci et al.](#page-10-11) [\(2024\)](#page-10-11), so <sup>η</sup>(t) ∈ [0, 1] reflects only the schedule.

### 3.2.2. PREDICTING LOSS CURVES ACROSS SCHEDULES

We apply this simple model to predict empirical loss curves in the CIFAR-5M experiments. We measure the trace of the preconditioned gradient covariance on a fixed set of 2M tokens (see Appendix [A](#page-12-0) for experiment details).

Let L, η, ¯ Σ¯ be a given reference trajectory and L¯′ = L¯ + δL, η ¯ ′ = η + δη, Σ¯′ = Σ +¯ δΣ¯ be the target trajectory, Equation [\(14\)](#page-6-0) allows us to predict the target loss via

$$\delta\bar{L}(\tau) \approx \frac{1}{4} \text{Tr}[\delta(\eta(\tau)\bar{\Sigma}(\tau))], \quad (15)$$

where δ η(τ )Σ( ¯ τ ) := η ′ (τ )Σ¯′ (<sup>τ</sup> ) − <sup>η</sup>(<sup>τ</sup> )Σ( ¯ <sup>τ</sup> ). We use a constant learning rate for the reference trajectories and various schedules sharing the same peak learning rate for the target. Decomposing δ(ηΣ) = δηΣ ′+ηδΣ, we find the first term is typically 3 to 10 times larger than the second as the learning rate decays, which can be attributed to how learning rate interacts with curvature (Figure [13\)](#page-23-0). In Figure [6,](#page-6-1) we only keep the first term, and predict the target loss as

$$L'(\tau) \approx L(\tau) + \alpha \delta\eta(\tau) \text{Tr}(\Sigma'(\tau)), \quad (16)$$

where α is a shared hyperparameter. We find a single α = 0.21 fits the target loss curves surprisingly well across schedules, model sizes, and training horizons. In Appendix [I,](#page-22-0) we show even better fits for MLPs, though puzzlingly, including the second term can lead to worse fits.

Recent works proposed more complex functional forms for how learning rate schedules affect loss curves, derived primarily from empirical observations [\(Tissue et al.,](#page-10-12) [2024;](#page-10-12) [Luo](#page-10-13) [et al.,](#page-10-13) [2025\)](#page-10-13). The accuracy of our simple model suggests it captures the essential dynamics, and crucially, the correct scaling of the excess loss through Tr(Σ′ ) so that a single α is predictive across model sizes, schedules, and training horizons. Notably, [Luo et al.](#page-10-13) [\(2025\)](#page-10-13) experimented with a similar form to Equation [\(16\)](#page-6-2) but with a constant Σ ′ , which likely explains the reduced effectiveness they observed.

![](_page_7_Figure_1.jpeg)

Figure 7: Universality of gradient noise on CIFAR-5M. Fixing a learning rate schedule, the ratio Tr(Σ)/L is approximately a function of normalized compute alone, independent of model size. We show similar results with MLP regression in Figure [14.](#page-23-1)

### 3.2.3. UNIVERSAL SCALING OF GRADIENT NOISE

For typical loss functions, the gradient covariance can be related to the loss itself. For example, in noiseless highdimensional linear regression with Gaussian features drawn from N (0, K), we have Tr(Σ) ≈ <sup>2</sup>L Tr(K)[\(Paquette et al.,](#page-10-9) [2021\)](#page-10-9), an intuitive result since the gradient scales with both the prediction error and the input. For non-linear regression, K should be taken to be the time-varying Gauss-Newton matrix for a first approximation. In this case, Tr(K) is known to depend strongly with the learning rate [\(Agarwala](#page-9-8) [& Pennington,](#page-9-8) [2024\)](#page-9-8), but we expect weak dependence on model size given our models are trained with µP (see [Noci](#page-10-11) [et al.](#page-10-11) [\(2024\)](#page-10-11) for evidence that curvature statistics depend weakly on model size in µP). Since the schedule is a function of the normalized compute x = t/t<sup>⋆</sup> alone, we hypothesize there exists a schedule-dependent function h(x) such that

$$\text{Tr}(\Sigma(xt^*(p)))/\mathcal{L}(xt^*(p)) \approx h(x), \quad (17)$$

which we verify in the regression (Figure [14\)](#page-23-1) and next-token prediction experiments (Figure [7\)](#page-7-2).

Combining Equation [\(16\)](#page-6-2) and Equation [\(17\)](#page-7-3) and making p-dependence explicit:

$$\bar{\mathcal{L}}'(\tau, p) \approx \bar{\mathcal{L}}(\tau, p)(1 - \alpha h(x)\delta\eta(\tau, p))^{-1}, \quad (18)$$

where x is the normalized compute at gradient flow time τ. We leave to future work an explanation of why this relation appears to hold for cross-entropy loss despite the presence of non-negligible irreducible loss, as this setting is analogous to regression with label noise, where the gradient covariance should scale with the total loss rather than just the reducible component, i.e., Tr(Σ) ≈ <sup>2</sup><sup>L</sup> Tr(K).

Scaling Collapse Across Schedules. Combining our insights so far, we can now understand why collapse happens across schedules. Let ¯ℓ(x, p) and ¯ℓ ′ (x, p) be the expected

normalized loss curves under two schedules S and S ′ . Let y(x) map the normalized compute under S ′ to the normalized compute under S at matching gradient flow time, where y is independent of p for schedules defined in terms of the normalized compute. Let δηˆ(x) = δη(xt<sup>⋆</sup> (p), p) be the difference between the two schedules. Assuming small relative fluctuations (<sup>E</sup>[L(x)/L(y)] ≈ <sup>E</sup>[L(x)]/E[L(y)]), we have:

$$\bar{\ell}'(x, p) \approx \frac{\bar{\mathcal{L}}'(xt^\star(p), p)}{\bar{\mathcal{L}}'(t^\star(p), p)} \quad (19)$$

$$= \frac{\bar{\mathcal{L}}(y(x)t^*(p), p)(1 - \alpha h(x)\delta\hat{\eta}(x))^{-1}}{\bar{\mathcal{L}}(y(1)t^*(p), p)(1 - \alpha h(1)\delta\hat{\eta}(1))^{-1}} \quad (20)$$

$$= \bar{\ell}(y(x), p) \underbrace{\frac{1 - \alpha h(1)\delta\hat{\eta}(1)}{1 - \alpha h(x)\delta\hat{\eta}(x)}}_{\text{independent of } p}, \quad (21)$$

which shows that, in expectation, collapse under one schedule (e.g. constant) implies collapse under any other schedule, provided we take Equation [\(18\)](#page-7-1) to be exact. Since collapse under a constant learning rate can be attributed to the sumof-power-laws scaling law, this result helps explain why we also observe collapse in other schedules.

This analysis also suggests that collapse can serve as a filter for identifying interventions that yield scalable improvements: those that multiplicatively shift the reducible loss curve by the same factor across all model sizes.

### 3.3. Supercollapse as Variance Reduction

Lastly, we turn to understanding the "super" in supercollapse: why does learning rate decay significantly improve the collapse, to the extent that the collapse deviation ∆(x) drops below the per-model noise floor σ(x, p) for a substantial fraction of training? Again, the simple quadratic model provides quantitative insights into this phenomenon.

Recall ∆(x) = <sup>V</sup>p,ω[ℓ(x,p,ω)]<sup>1</sup>/<sup>2</sup> <sup>E</sup>p,ω[ℓ(x,p,ω)] , and the decomposition:

$$\mathbb{V}_{p,\omega}[\ell(x,p)] = \mathbb{E}_p \mathbb{V}_\omega[\ell(x,p)] + \mathbb{V}_p \mathbb{E}_\omega[\ell(x,p)]. \quad (22)$$

The first term measures variance due to the seed alone, averaged over model sizes, while the second term measures variance due to varying the model size, having averaged over the seeds first. Since we observed that variations in the normalized curves primarily arise from seed-to-seed fluctuations rather than model-to-model differences (Section [2.4\)](#page-2-3) under a constant schedule, and switching to other schedules does not significantly increase the model-to-model differences (Section [3.2\)](#page-5-0), we will assume the first term <sup>E</sup>pVω[ℓ(x, p)] dominates, which implies ∆<sup>2</sup> (x) ≈ <sup>E</sup>p∆˜ <sup>2</sup> (x, p), where ∆˜ <sup>2</sup> (x, p) := <sup>V</sup>ω[ℓ(x, p)]/ ¯ℓ 2 (x, p) is the squared *per-model* collapse deviation.

To simplify notation, we temporarily omit p-dependence and write <sup>ℓ</sup> in terms of <sup>t</sup> instead of <sup>x</sup>. Letting L(t) =

![](_page_8_Figure_1.jpeg)

Figure 8: A quantitative explanation of how learning rate decay leads to supercollapse. Across schedules on CIFAR-5M, collapse deviation at normalized gradient flow time τˆ follows the predicted p <sup>η</sup> (1 − <sup>τ</sup>ˆ) scaling, capturing the noise accumulated between that point and end of training. A schedule that decays faster has a smaller <sup>η</sup> and (1 − <sup>τ</sup>ˆ) at a fixed normalized compute or training step.

L¯(t)(1 + <sup>ψ</sup>(t)), where <sup>ψ</sup> is the relative fluctuation, we have <sup>ℓ</sup>(t) = <sup>L</sup>¯(t)(1+ψ(t)) L¯(t <sup>⋆</sup>)(1+ψ(t <sup>⋆</sup>)) ≈ ¯ℓ(t)(1 + <sup>ψ</sup>(t) − <sup>ψ</sup>(<sup>t</sup> ⋆ )), assuming <sup>ψ</sup> ≪ <sup>1</sup>. Therefore,

$$\tilde{\Delta}^2(t) \approx \mathbb{E}[(\psi(t) - \psi(t^*))^2] \quad (23)$$

We see that what controls the relative variance in ℓ(t) is not <sup>ψ</sup>(t) but the difference <sup>ψ</sup>(t) − <sup>ψ</sup>(<sup>t</sup> ⋆ ), which roughly captures the amount of optimization noise accumulated *between* time t and time t ⋆ . Since the optimization noise per step scales with the instantaneous learning rate, decaying the learning rate over time will precisely serve to decrease the variance in ℓ. By contrast, the squared per-model noise floor σ 2 (t) is simply <sup>E</sup>[ψ 2 (t)], which captures the total cumulative optimization noise. Importantly, had we normalized by the expected rather than the empirical final loss in ℓ, ∆( ˜ t) would reduce to σ(t). Normalizing by the stochastic final loss is essential for supercollapse, where it acts as a control-variate [\(Glasserman,](#page-9-9) [2004\)](#page-9-9), leveraging the strong time-correlation of stochastic fluctuations along the optimization trajectory to cancel much of the shared noise and thereby sharply reduce the variance of the collapsed curve.

Quantitatively, we can estimate ∆˜ under the quadratic model in Section [3.2.](#page-5-0) Let <sup>∆</sup>w(<sup>τ</sup> ) and <sup>∆</sup>L(<sup>τ</sup> ) be the fluctuations of the parameters and loss from their means. We have <sup>∆</sup>w(<sup>τ</sup> ) = R <sup>τ</sup> 0 ds e−H(τ−s)Σ 1/2 (s)ξ(s), and <sup>∆</sup>L(<sup>τ</sup> ) = g(τ ) <sup>⊤</sup>∆w(τ ) to first order in ∆w(τ ), where g(τ ) is the expected gradient. Close to the end of training, for τ = τ <sup>⋆</sup>−δτ where τ ⋆ is the final gradient flow time and δτ > 0 is small, direct calculation shows (Appendix [H\)](#page-21-0)

$$\tilde{\Delta}^2(\tau) = \tilde{\mathcal{L}}^{-2}(\tau)g(\tau)^\top \eta(\tau)\bar{\Sigma}(\tau)g(\tau)\delta\tau + O(\delta\tau^2), \quad (24)$$

In linear regression, <sup>Σ</sup> ∝ L and <sup>g</sup> <sup>⊤</sup><sup>g</sup> ∝ L , so we estimate ∆˜ <sup>2</sup> (<sup>τ</sup> ) ∝ <sup>η</sup>(<sup>τ</sup> )δτ to leading order. Since this relation holds for each model size p, we predict ∆<sup>2</sup> (ˆ<sup>τ</sup> ) ≈ <sup>E</sup>p∆˜ <sup>2</sup> (ˆ<sup>τ</sup> ) ∝ <sup>η</sup>(ˆ<sup>τ</sup> )(1−τˆ), where <sup>τ</sup><sup>ˆ</sup> <sup>=</sup> τ /τ <sup>⋆</sup> denotes the normalized gradient flow time. Figure [8](#page-8-0) shows this form fits the observations well, with ∆(ˆτ )/ p √ η(ˆτ ) approximately following the same <sup>1</sup> − <sup>τ</sup><sup>ˆ</sup> scaling across many schedules, quantitatively explaining how learning rate decay leads to supercollapse.

### 4. Discussion

Scale has enabled remarkable progress in machine learning, but a thorough scientific understanding of scaling remains elusive. Key open questions include identifying robust principles that guide general hyperparameter transfer and characterizing scaling limits under realistic scaling ladders. Our discovery of supercollapse provides empirical evidence that a model-size and data joint scaling limit *generically* exists in the compute-optimal regime, and that the scale-invariance of the training dynamics revealed by the collapse can diagnose proper hyperparameter configuration. We believe further investigation of these phenomena holds great potential for advancing the science of scaling.

We see many exciting extensions to this work. Empirically, our small-scale experiments provide a proof-of-concept. While small-scale proxies capture certain behaviors in larger systems [\(Wortsman et al.,](#page-11-0) [2023\)](#page-11-0), validating at larger scales and with practical scaling ladders, where width, depth, batch size, and weight decay are co-scaled [\(McCandlish et al.,](#page-10-14) [2018;](#page-10-14) [Wang & Aitchison,](#page-11-7) [2024;](#page-11-7) [Dey et al.,](#page-9-10) [2025;](#page-9-10) [Bergsma](#page-9-11) [et al.,](#page-9-11) [2025\)](#page-9-11), is important and may yield new insights into optimal scaling and hyperparameter transfer. Scaling collapse beyond the form we studied here can be a general tool to study other scaling relations [\(Tamai et al.,](#page-10-15) [2023\)](#page-10-15).

While we have identified the key ingredients underlying supercollapse—power-law scaling and learning ratedependent noise dynamics—our analysis relies on multiple approximations and takes power-law scaling as given, suggesting deeper theoretical principles may be at work. Taking collapse as a starting point instead may provide an alternative route to understanding scaling laws, analogous to how in physics the renormalization group was developed to provide a unified set of principles explaining both universality and its associated power laws [\(Wilson,](#page-11-8) [1971\)](#page-11-8). Finally, it would be interesting to understand why our simple noise model predicts the impact of learning rate schedules on real models so effectively, compare it with alternative models such as [Schaipp et al.](#page-10-16) [\(2025\)](#page-10-16), and to test its predictive power for optimizing schedules, learning rates, and training horizons.

- Acknowledgements We thank Courtney Paquette and Zixi Chen for helpful comments on an earlier version of this paper. SQ was supported by Google's TPU Research Cloud (TRC) program: <https://sites.research.google/trc/>. Contribution Statement SQ designed and conducted the majority of experiments, led the theory development, and wrote the paper. LX initially observed supercollapse, contributed to theory, experimental design, and writing the paper. AGW advised SQ and edited the paper. JP contributed to theory, experimental design, and writing the paper. AA managed the research project, proved some theorems, guided the theory development, contributed to experimental design, and helped write the paper. Impact Statement This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. References Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023. Agarwala, A. and Fisher, D. S. Adaptive walks on high-dimensional fitness landscapes and seascapes with distance-dependent statistics. *Theoretical population biology*, 130:13–49, 2019. Agarwala, A. and Pennington, J. High dimensional analysis reveals conservative sharpening and a stochastic edge of stability, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2404.19261) [2404.19261](https://arxiv.org/abs/2404.19261). Bergsma, S., Dey, N., Gosal, G., Gray, G., Soboleva, D., and Hestness, J. Power lines: Scaling laws for weight decay and batch size in llm pre-training. *arXiv preprint arXiv:2505.13738*, 2025. Binder, K. Finite size scaling analysis of ising model block distribution functions. *Zeitschrift fur Physik B Condensed ¨ Matter*, 43:119–140, 1981. Bordelon, B. and Pehlevan, C. Self-consistent dynamical field theory of kernel evolution in wide neural networks. *Advances in Neural Information Processing Systems*, 35: 32240–32256, 2022. Bordelon, B., Noci, L., Li, M. B., Hanin, B., and Pehlevan, C. Depthwise hyperparameter transfer in residual networks: Dynamics and scaling limit. *arXiv preprint arXiv:2309.16620*, 2023. Bordelon, B., Atanasov, A., and Pehlevan, C. A dynamical model of neural scaling laws. *arXiv preprint arXiv:2402.01092*, 2024a. Bordelon, B., Atanasov, A., and Pehlevan, C. How feature learning can improve neural scaling laws. *arXiv preprint arXiv:2409.17858*, 2024b. Bordelon, B., Chaudhry, H., and Pehlevan, C. Infinite limits of multi-head transformer dynamics. *Advances in Neural Information Processing Systems*, 37:35824–35878, 2024c. Cohen, J. M., Kaur, S., Li, Y., Kolter, J. Z., and Talwalkar,
  - A. Gradient descent on neural networks typically occurs at the edge of stability. *arXiv preprint arXiv:2103.00065*, 2021. Cohen, J. M., Ghorbani, B., Krishnan, S., Agarwal, N., Medapati, S., Badura, M., Suo, D., Cardoze, D., Nado, Z., Dahl, G. E., et al. Adaptive gradient methods at the edge of stability. *arXiv preprint arXiv:2207.14484*, 2022. Cohen, J. M., Damian, A., Talwalkar, A., Kolter, Z., and Lee,
  - J. D. Understanding optimization in deep learning with central flows. *arXiv preprint arXiv:2410.24206*, 2024. d'Ascoli, S., Refinetti, M., and Biroli, G. Optimal learning rate schedules in high-dimensional non-convex optimization problems. *arXiv preprint arXiv:2202.04509*, 2022. Dey, N., Zhang, B. C., Noci, L., Li, M., Bordelon, B., Bergsma, S., Pehlevan, C., Hanin, B., and Hestness, J. Don't be lazy: Completep enables compute-efficient deep transformers. *arXiv preprint arXiv:2505.01618*, 2025. Diederik P. Kingma, J. B. Adam: A Method for Stochastic Optimization. *International Conference on Learning Representations (ICLR)*, 2015. Everett, K., Xiao, L., Wortsman, M., Alemi, A. A., Novak, R., Liu, P. J., Gur, I., Sohl-Dickstein, J., Kaelbling, L. P., Lee, J., et al. Scaling exponents across parameterizations and optimizers. *arXiv preprint arXiv:2407.05872*, 2024. Fisher, D. S. Asexual evolution waves: fluctuations and universality. *Journal of Statistical Mechanics: Theory and Experiment*, 2013(01):P01011, 2013. Glasserman, P. *Monte Carlo methods in financial engineering*, volume 53. Springer, 2004.

- Hallatschek, O. and Fisher, D. S. Acceleration of evolutionary spread by long-range dispersal. *Proceedings of the National Academy of Sciences*, 111(46):E4911–E4919, 2014. Hendrycks, D. and Gimpel, K. Gaussian Error Linear Units (GELUs). *Preprint arXiv 1606.08415*, 2016. Hestness, J., Narang, S., Ardalani, N., Diamos, G., Jun, H., Kianinejad, H., Patwary, M. M. A., Yang, Y., and Zhou, Y. Deep learning scaling is predictable, empirically. *arXiv preprint arXiv:1712.00409*, 2017. Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A., Welbl, J., Clark, A., et al. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556*, 2022. Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020. Levy, S. F., Blundell, J. R., Venkataram, S., Petrov, D. A., Fisher, D. S., and Sherlock, G. Quantitative evolutionary dynamics using high-resolution lineage tracking. *Nature*, 519(7542):181–186, 2015. Li, Q., Tai, C., et al. Stochastic modified equations and adaptive stochastic gradient algorithms. In *International Conference on Machine Learning*, pp. 2101–2110. PMLR, 2017. Luo, K., Wen, H., Hu, S., Sun, Z., Liu, Z., Sun, M., Lyu, K., and Chen, W. A multi-power law for loss curve prediction across learning rate schedules. *arXiv preprint arXiv:2503.12811*, 2025. Malladi, S., Lyu, K., Panigrahi, A., and Arora, S. On the sdes and scaling rules for adaptive gradient algorithms. *Advances in Neural Information Processing Systems*, 35: 7697–7711, 2022. Marchenko, V. and Pastur, L. A. Distribution of eigenvalues for some sets of random matrices. *Mat. Sb.(NS)*, 72(114): 4, 1967. McCandlish, S., Kaplan, J., Amodei, D., and Team, O. D. An empirical model of large-batch training. *arXiv preprint arXiv:1812.06162*, 2018. McLeish, S., Kirchenbauer, J., Miller, D. Y., Singh, S., Bhatele, A., Goldblum, M., Panda, A., and Goldstein, T. Gemstones: A model suite for multi-faceted scaling laws. *arXiv preprint arXiv:2502.06857*, 2025. Nakkiran, P., Neyshabur, B., and Sedghi, H. The deep bootstrap framework: Good online learners are good offline generalizers. *arXiv preprint arXiv:2010.08127*, 2020. Noci, L., Meterez, A., Hofmann, T., and Orvieto, A. Super consistency of neural network landscapes and learning rate transfer. *Advances in Neural Information Processing Systems*, 37:102696–102743, 2024. Paquette, C., Lee, K., Pedregosa, F., and Paquette, E. Sgd in the large: Average-case analysis, asymptotics, and stepsize criticality. In *Conference on Learning Theory*, pp. 3548–3626. PMLR, 2021. Paquette, C., Paquette, E., Adlam, B., and Pennington, J. Homogenization of sgd in high-dimensions: Exact dynamics and generalization properties. *Mathematical Programming*, pp. 1–90, 2024a. Paquette, E., Paquette, C., Xiao, L., and Pennington, J. 4+ 3 phases of compute-optimal neural scaling laws. *arXiv preprint arXiv:2405.15074*, 2024b. Pearce, M. T., Agarwala, A., and Fisher, D. S. Stabilization of extensive fine-scale diversity by ecologically driven spatiotemporal chaos. *Proceedings of the National Academy of Sciences*, 117(25):14572–14583, 2020. Pearce, T. and Song, J. Reconciling kaplan and chinchilla scaling laws. *arXiv preprint arXiv:2406.12907*, 2024. Rosen, M. J., Davison, M., Bhaya, D., and Fisher, D. S. Fine-scale diversity and extensive recombination in a quasisexual bacterial population occupying a broad niche. *Science*, 348(6238):1019–1023, 2015. Schaipp, F., Hagele, A., Taylor, A., Simsekli, U., and Bach, ¨
  - F. The surprising agreement between convex optimization theory and learning-rate scheduling for large model training. *arXiv preprint arXiv:2501.18965*, 2025. Ser-Giacomi, E., Zinger, L., Malviya, S., De Vargas, C., Karsenti, E., Bowler, C., and De Monte, S. Ubiquitous abundance distribution of non-dominant plankton across the global ocean. *Nature ecology & evolution*, 2(8):1243– 1249, 2018. Sharma, U. and Kaplan, J. Scaling laws from the data manifold dimension. *Journal of Machine Learning Research*, 23(9):1–34, 2022. Tamai, K., Okubo, T., Duy, T. V. T., Natori, N., and Todo,
  - S. Universal scaling laws of absorbing phase transitions in artificial deep neural networks. *arXiv preprint arXiv:2307.02284*, 2023. Tissue, H., Wang, V., and Wang, L. Scaling law with learning rate annealing. *arXiv preprint arXiv:2408.11029*, 2024.

- Venkataram, S., Dunn, B., Li, Y., Agarwala, A., Chang, J., Ebel, E. R., Geiler-Samerotte, K., Herissant, L., Blundell, ´
- J. R., Levy, S. F., et al. Development of a comprehensive genotype-to-fitness map of adaptation-driving mutations in yeast. *Cell*, 166(6):1585–1596, 2016. Vyas, N., Atanasov, A., Bordelon, B., Morwani, D., Sainathan, S., and Pehlevan, C. Feature-learning networks are consistent across widths at realistic scales. *Advances in Neural Information Processing Systems*, 36: 1036–1060, 2023. Wang, X. and Aitchison, L. How to set adamw's weight decay as you scale model and dataset size. *arXiv preprint arXiv:2405.13698*, 2024. Wen, K., Li, Z., Wang, J., Hall, D., Liang, P., and Ma, T. Understanding warmup-stable-decay learning rates: A river valley loss landscape perspective. *arXiv preprint arXiv:2410.05192*, 2024. Wilson, K. G. Renormalization group and critical phenomena. i. renormalization group and the kadanoff scaling picture. *Physical review B*, 4(9):3174, 1971. Wortsman, M., Liu, P. J., Xiao, L., Everett, K., Alemi, A., Adlam, B., Co-Reyes, J. D., Gur, I., Kumar, A., Novak, R., et al. Small-scale proxies for large-scale transformer training instabilities. *arXiv preprint arXiv:2309.14322*, 2023. Xiao, L. Rethinking conventional wisdom in machine learning: From generalization to scaling. *arXiv preprint arXiv:2409.15156*, 2024. Yang, G. and Hu, E. J. Feature Learning in Infinite-Width Neural Networks. *International Conference on Machine Learning (ICML)*, 2021. Yang, G. and Littwin, E. Tensor Programs IVb: Adaptive Optimization in the Infinite-Width Limit. *International Conference on Learning Representations (ICLR)*, 2023. Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., Ryder, N., Pachocki, J., Chen, W., and Gao, J. Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. *Advances in Neural Information Processing Systems (NeurIPS)*, 2021. Zhang, B. and Sennrich, R. Root mean square layer normalization. *Advances in Neural Information Processing Systems*, 32, 2019. Zhang, G., Li, L., Nado, Z., Martens, J., Sachdeva, S., Dahl, G., Shallue, C., and Grosse, R. B. Which algorithmic choices matter at which batch sizes? insights from a noisy quadratic model. *Advances in neural information processing systems*, 32, 2019.

### A. Experiment Details

Transformer Architecture. We use GeLU activations [\(Hendrycks & Gimpel,](#page-10-17) [2016\)](#page-10-17), RMSNorm [\(Zhang & Sennrich,](#page-11-9) [2019\)](#page-11-9), and learned positional embeddings. We untie the embedding matrix from the output head and do not use bias anywhere. The readout layer is always zero-initialized, as suggested by [Yang et al.](#page-11-2) [\(2021\)](#page-11-2) . We denote the embedding dimension with D. We set the intermediate dimension in the feedforward layers to D instead of the usual 4D, which enables us to explore larger widths more efficiently. The head dimension is set to 64.

CIFAR-5M. We use the CIFAR-5M dataset [\(Nakkiran et al.,](#page-10-4) [2020\)](#page-10-4) of 6 million CIFAR-like images. We convert the <sup>32</sup> × <sup>32</sup> × <sup>3</sup> images to greyscale and flatten them into sequences of length <sup>1024</sup>. The model autoregressively predicts the pixel intensities in raster-scan order. The vocabulary is the set of pixel intensities {0, . . . , <sup>255</sup>}. Following <sup>µ</sup>P we parameterize the learning rate for each weight matrix as η = ηbase/D where d is the model dimension, except for the embedding matrix which has η = ηbase. We use a parameter multiplier a on the embedding matrix. We use ηbase = 4 and a = 0.1 as they led to good performance in our early experiments. We initialize the embedding matrix as Wemb ij ∼ N (0, 1), the output head as <sup>W</sup>head = 0, all other non-readout matrices <sup>W</sup> as <sup>W</sup>ij ∼ N (0, <sup>1</sup>/D). These hyperparameters were determined with a small amount of tuning in early experiments. We use a batch size of 256 images. We use a linear warmup for 1000 steps.

For the experiments in Section [3.2.2,](#page-6-3) we used a slightly different setup due to switching to a new codebase in the middle of the research project. We use µP where the base embedding dimension is 128 and base learning rate is 0.01. We initialize the embedding matrix with a standard deviation (std) of 0.1 and multiply its learning rate by 10 relative to the base learning rate. The output projection of the feedforward and attention layers are zero-initialized. All other non-readout matrices are initialized with std 1/ √ D. We use a batch size of 65536 tokens or 64 images. We use a linear warmup for 10M tokens.

Chess. We run our experiments on the Lichess dataset available on Hugging Face at [https://huggingface.co/](https://huggingface.co/datasets/Lichess/standard-chess-games) [datasets/Lichess/standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games). We used character-level tokenization and a context length of 128. We use µP where the base embedding dimension is 128 and base learning rate is 0.01. We initialize the embedding matrix with std 0.1 and multiply its learning rate by 10 relative to the base learning rate. The output projection of the feedforward and attention layers are zero-initialized. All other matrices are initialized with std 1/ √ D. We use a batch size of 65536 tokens. We use a linear warmup for 10M tokens.

MLP Experiments. Our MLP architecture is identical to the transformer with attention layers removed and the token and position embedding layers replaced by a linear layer. We use µP where the base width is 128 and base learning rate is 0.001. The output projection of the feedforward layers are zero-initialized. All non-readout matrices are initialized with std 1/ √ D. We use a batch size of 4096 examples. We do not use warmup.

The target function is defiend as ϕ(x) = P<sup>M</sup> <sup>i</sup>=1 w<sup>i</sup> √ 2 cos 2πk<sup>⊤</sup> <sup>i</sup> x + b<sup>i</sup> , with <sup>x</sup> ∈ <sup>R</sup> 8 , w<sup>i</sup> ∼ N (0, 1), b<sup>i</sup> ∼ π <sup>2</sup> Bernoulli(0.5), k<sup>i</sup> = round(sivi) where s<sup>i</sup> is a scalar sampled from a power law with support [1, ∞) and exponent −2, v<sup>i</sup> is a random unit vector, and round rounds to the nearest point in Z 8 . During training, x is sampled uniformly from [−0.5, <sup>0</sup>.5]<sup>8</sup> , making the Fourier features orthonormal over the data distribution. We suspect the details here are not necessary for generating power-law scaling laws beyond the power-law spectrum.

For Figure [4a,](#page-3-0) the constant learning rate scaling ladder matches the µP learning rate at D = 384 (smallest model).

Measuring Gradient Covariance Trace. As mentioned in Section [3.2.1,](#page-5-1) we use the preconditioned gradient covariance Σ˜ instead of the raw gradient covariance due to the use of Adam (and µP). Σ˜ is defined as P <sup>−</sup>1/<sup>2</sup>ΣP <sup>−</sup>1/<sup>2</sup> where Σ is the raw (mini-batch) gradient covariance, and P is the preconditioner. See an explanation for this definition in Appendix [G.](#page-18-1) Since µP uses layerwise learning rate, the preconditioner is defined as P <sup>−</sup><sup>1</sup> = diag √ η<sup>0</sup> v <sup>2</sup>+ϵ , where η<sup>0</sup> is the vector of per-parameter learning rate (peak learning rate, before applying the schedule), v 2 is the Adam second-moment, and ϵ is the Adam ϵ. As the peak learning rate is absorbed into the preconditioner, any occurrence of the instantaneous learning rate η(t) or η(τ ) in Section [3.2.2](#page-6-3) reflects only the schedule and takes on values in [0, 1]. This definition mirrors what is done in [Noci](#page-10-11) [et al.](#page-10-11) [\(2024\)](#page-10-11); [Cohen et al.](#page-9-12) [\(2024\)](#page-9-12).

Training for More than One Epoch. The CIFAR-5M dataset, tokenized as <sup>32</sup> × <sup>32</sup> greyscale images, has about 5B tokens. Therefore, most models needed to be trained for more than 1 epoch to reach the compute-optimal training horizon. Up to the scales we tested, we did not observe a significant difference between the train and test loss. The chess dataset has about 20B tokens, which also led to data reuse for some models, but did not lead to significant overfitting. As we only processed a subset of the full Lichess dataset, this can be avoided by processing a larger subset if desired.

On Random Seeds. In the CIFAR-5M experiments in Figure [1,](#page-1-0) the random seed controls both the initialization and data ordering. In the other experiments (Transformer on chess and MLP regression), the random seed only controls the initialization while data ordering is held fixed, as is often done in practice. Fixing the data ordering (no shuffling) had the advantage of speeding up data loading. We found that supercollapse occurs regardless of whether seeds affect data ordering. This makes sense: even with fixed ordering, different model sizes process different data due to varying training horizons. More fundamentally, supercollapse should be robust to which training components are randomized, as the variance reduction arises from strong noise correlations along individual trajectories rather than specific noise sources (Section [3.3\)](#page-7-0).

# B. Scaling Collapse Across Transformer Depths

For scaling depth, we additionally apply a branch multiplier of 3/depth on the output of every feedforward and attention layer, as suggested by [Bordelon et al.](#page-9-3) [\(2024c\)](#page-9-3). We find a decent degree of collapse in Figure [9](#page-13-2) when training on chess data. There is a small shift in the normalized curves, though we are unsure if it is simply a finite-size effect.

![](_page_13_Figure_4.jpeg)

Figure 9: Depthwise scaling collapse for transformers trained on chess.

# C. Estimating Compute-Optimal Training Horizon

To estimate the optimal compute for training each model, we perform the following steps in each experiment:

- We trained each model *without* learning rate decay but keeping the initial warmup. We chose a large enough number of steps so that the largest model could reach the compute-optimal loss frontier. We average the loss curves from 5 seeds.
- We numerically computed the compute-loss Pareto frontier to obtain an estimate of c ⋆
  - (p) the optimal compute for each model size p. We use logarithmically spaced points for c and find the p that achieves the best loss given c training FLOPs.
- We fit a power law c ⋆
- (p) = κp1+<sup>γ</sup> where κ and γ are fit parameters. The optimal number of training tokens is then t ⋆
  - (p) = c ⋆ (p)/(6p), which scales as p γ . We remove outliers in this fit by dropping points from the smallest and largest model.

Using a constant learning rate schedule allows us to measure the loss of one model at different token budgets with a single training run, an approach also used in [McLeish et al.](#page-10-18) [\(2025\)](#page-10-18), rather than one training run per token budget as origianlly done in [Hoffmann et al.](#page-10-3) [\(2022\)](#page-10-3). Figure [10](#page-14-2) illustrates this procedure.

![](_page_14_Figure_1.jpeg)

Figure 10: Estimating compute-optimal data exponent in the MLP regression and Transformer CIFAR-5M experiments.

### D. Universality and Scaling Collapse in Other Sciences

The simplest versions of collapse come from statistics and probability, where entire *distributions* of random variables show universal behavior between systems of different types and scales. The most well known is the central limit theorem which predicts a universal Gaussian form for the sums of random variables with appropriately bounded moments (and Levy distributions for heavy tailed distributions). In random matrix theory, there are similar effects when studying the limiting empirical distribution of spectra, most famously the Marcenko-Pastur distribution of Wishart matrices [\(Marchenko & Pastur,](#page-10-19) [1967\)](#page-10-19). In all of these examples, showing universal behavior of a single moment is analogous to showing predictability of the Pareto frontier in our work, while showing the universality of the whole distribution is analogous to our statements about the entire loss curves (where e.g. the CDFs of different problems are converging).

Scaling collapse is ubiquitous in physics as well. There are again distributional collapses like the famed Maxwell Boltzmann distribution first used to describe idealized gasses, but also functional relationships like the universal magnetizationtemperature curves of near-critical Ising lattices of different sizes [\(Binder,](#page-9-13) [1981\)](#page-9-13). In the Ising example, changes to the lattice topology can change the magnetization-temperature curves, similar to how different datasets, architecture, and training algorithms lead to different universal curves in our study. A more general theory unifying and explaining the existence of universality in physics arises from the renormalization group [\(Wilson,](#page-11-8) [1971\)](#page-11-8).

More recently, scaling collapse has been used to describe dynamical systems in biological contexts. Advances in genomics have led to the rise of experimental microbial evolution with rapid timecourse data [\(Levy et al.,](#page-10-20) [2015;](#page-10-20) [Venkataram et al.,](#page-11-10) [2016\)](#page-11-10). Analysis of this data relies on quantitative modeling of evolutionary dynamics. Most of these models show universal scaling dynamics, in situations from rapid evolutions of diverse populations [\(Fisher,](#page-9-14) [2013\)](#page-9-14), populations evolving under changing fitness conditions [\(Agarwala & Fisher,](#page-9-15) [2019\)](#page-9-15), and populations expanding in space [\(Hallatschek & Fisher,](#page-10-21) [2014\)](#page-10-21). In these settings the timecourse of key observables can be described with dynamical curves which can be rescaled to universal forms with transformations depending on population size, mutation frequency, and statistics of the fitness landscape. In ecology, the ubiquity of fine-scale diversity and the seemingly universal, power-law nature of species rank-abundance curves [\(Rosen et al.,](#page-10-22) [2015;](#page-10-22) [Ser-Giacomi et al.,](#page-10-23) [2018\)](#page-10-23) can be explained using dynamical models which themselves show universal scaling behavior over ecosystems of different sizes [\(Pearce et al.,](#page-10-24) [2020\)](#page-10-24).

Scaling collapse has been used to study certain scaling relations in machine learning. [Kaplan et al.](#page-10-1) [\(2020\)](#page-10-1) identified universal scaling of overfitting, showing a collapse of the rescaled excess loss vs rescaled parameter count across dataset sizes. [Tamai](#page-10-15) [et al.](#page-10-15) [\(2023\)](#page-10-15) used scaling collapse to establish universal scaling laws in the forward signal propagation dynamics of MLPs near the order-to-chaos transition.

# E. Power-Law Pareto Frontier is Necessary for Collapse

Recall t ⋆ (p) is the optimal training horizon for model size p, i.e. L(t ⋆ (p), p) = min<sup>t</sup> ,p′ :t ′p′=t <sup>⋆</sup>(p)<sup>p</sup> L(t ′ , p′ ). Let c ⋆ (p) = 6t ⋆ (p)p be the optimal compute for p. In what follows, rather than writing L(t, p), we will find it convenient to express the loss curves in terms of compute and model size. Letting L(c, p) be the loss curves expressed this way, we have the following theorem:

*Write* L(c, p) = <sup>L</sup>(c, p) − <sup>L</sup><sup>ˆ</sup> *for some offset* <sup>L</sup><sup>ˆ</sup> *(e.g.,* <sup>L</sup><sup>ˆ</sup> <sup>=</sup> <sup>L</sup><sup>0</sup> *the irreducible loss). Define the normalized loss curve*

$$\ell(x, p) = \frac{\mathcal{L}(x^\star(p), p)}{\mathcal{L}(c^\star(p), p)}, \quad x \in [0, 1]. \quad (25)$$

*Then,*

*1. Necessity. If* <sup>ℓ</sup> *is independent of* <sup>p</sup> *(collapse), then the Pareto frontier of* {L(c, p)}c,p

$$\mathcal{L}^*(c) := \min_p \mathcal{L}(c, p) = \mathcal{L}(c^*(p), p) \quad (26)$$

*is a power law* L ⋆ (c) = ac−<sup>δ</sup> *for some constants* a, δ*.*

*2. Sufficiency at first order. Conversely, suppose* L ⋆ (c) = ac−<sup>δ</sup> *, then*

$$\frac{d\ell(x, p)}{dx} \Big|_{x=1} = -\delta, \quad (27)$$

*independent of* p*. Hence all curves share the same first-order behavior around* x = 1*, i.e., they collapse to first order around* x = 1*.*

*Proof.* First, we have the following identity for log-derivatives for a general differentiable function u(v)

$$\frac{d \log u}{d \log v} = \frac{v}{u} \frac{du}{dv}. \quad (28)$$

Applying this to our normalized curve from Equation [\(25\)](#page-15-0) and using the chain rule:

$$\frac{d\mathcal{L}(x, p)}{dx} \Big|_{x=1} = \frac{c^*}{\mathcal{L}(c^*, p)} \frac{\partial \mathcal{L}(c, p)}{\partial c} \Big|_{c=c^*} \quad (29)$$

$$= \frac{\partial \log \mathcal{L}(c, p)}{\partial \log c} \bigg|_{c=c^*}, \quad (30)$$

where the second equality follows from Equation [\(28\)](#page-15-1).

Necessity. If ℓ is independent of p (collapse), then dℓ(x,p) dx <sup>x</sup>=1 is the same for all <sup>p</sup>. Set this common value to −<sup>δ</sup>. By Equation [\(30\)](#page-15-2),

$$\frac{\partial \log \mathcal{L}(c, p)}{\partial \log c} \Big|_{c=c^*(p)} = -\delta \quad \text{for every } p. \quad (31)$$

Since (c ⋆ (p), p) lies on the Pareto frontier and L is <sup>C</sup> 1 , the envelope theorem states

$$\frac{d\mathcal{L}^*(c)}{dc} \Big|_{c=c^*(p)} = \frac{\partial \mathcal{L}(c,p)}{\partial c} \Big|_{c=c^*(p)}, \quad (32)$$

i.e. the loss curve is tangent to the Pareto frontier at the compute-optimal point. Applying Equation [\(28\)](#page-15-1) to the frontier L ⋆ (c):

$$\frac{d \log \mathcal{L}^\star(c)}{d \log c} \Big|_{c=c^\star(p)} = \frac{c^\star(p)}{\mathcal{L}^\star(c^\star(p))} \frac{d \mathcal{L}^\star(c)}{dc} \Big|_{c=c^\star(p)} \quad (33)$$

$$= \frac{c^*(p)}{\mathcal{L}(c^*(p), p)} \frac{\partial \mathcal{L}(c, p)}{\partial c} \bigg|_{c=c^*(p)} \quad (34)$$

$$= \frac{\partial \log \mathcal{L}(c, p)}{\partial \log c} \Big|_{c=c^*(p)} \quad (35)$$

$$= -\delta, \quad (36)$$

where we used Equation [\(32\)](#page-15-3) and Equation [\(31\)](#page-15-4). This means the log-log slope of the frontier is constant, which means it is a power law L ⋆ (c) = ac−<sup>δ</sup> .

Sufficiency at first order. Assume L ⋆ (c) = ac−<sup>δ</sup> . For any model size p, the envelope theorem gives the tangency condition at c = c ⋆ (p):

$$\frac{\partial \mathcal{L}(c, p)}{\partial c} \Big|_{c=c^*} = \frac{d\mathcal{L}^*(c)}{dc} \Big|_{c=c^*} = -\delta a(c^*)^{-\delta-1}. \quad (37)$$

Applying Equation [\(30\)](#page-15-2):

$$\begin{aligned} \frac{d\ell(x, p)}{dx} \Big|_{x=1} &= \frac{\partial \log \mathcal{L}(c, p)}{\partial \log c} \Big|_{c=c^*(p)} \\ &= -\delta. \end{aligned} \quad (38) \quad (39)$$

Therefore, all curves collapse to first order around x = 1 as in Equation [\(27\)](#page-15-5).

Remark. A power-law Pareto frontier is not only necessary for full collapse but also already enforces a weaker, first–order form of collapse. Theorem [E.1](#page-14-3) assumes the compute–optimal point lies in the *interior* of each loss curve. This condition can fail for learning rate schedules that reach η = 0 after finitely many steps, because the optimum may then coincide with the boundary of the curve, where the envelope theorem tangency no longer applies. Such schedules are used throughout our experiments and are common in practice. Extension of the proof to handle these boundary-optimal schedules would be interesting.

# F. Collapse for General Sum-of-Power-Laws Loss Curves

Theorem F.1. *Suppose the loss curve is given by*

$$L(t, p) = L_0 + \sum_{i=1}^m a_i t^{-\mu_i} p^{-\nu_i}, \quad a_i > 0, \mu_i, \nu_i \geq 0, \quad (40)$$

*with at least one of* µ<sup>i</sup> , ν<sup>i</sup> *positive for every* i *(else absorb the term into* L0*). Let* t ⋆ (p) = κp<sup>γ</sup> *with* κ > 0, γ > 0 *be the asymptotic compute-optimal training horizon, and define the total exponent* β<sup>i</sup> := µiγ + ν<sup>i</sup> *and* b<sup>i</sup> := aiκ −µ<sup>i</sup> *. Without loss of generality, assume* βi*'s are sorted in non-decreasing order. Then,*

*1. Compute-optimality forces a tie. At least two* βi*'s share the minimum:*

$$\beta_1 = \beta_2 = \dots = \beta_k < \beta_{k+1} \leq \dots \leq \beta_m, \quad k \geq 2. \quad (41)$$

*2. Asymptotic collapse. The normalized loss curve*

$$\ell(x, p) := \frac{L(xt^\star(p), p) - L_0}{L(t^\star(p), p) - L_0}. \quad (42)$$

*is given by*

$$\ell(x, p) = \frac{\sum_{i=1}^k b_i x^{-\mu_i}}{\sum_{i=1}^k b_i} + O(p^{-\epsilon}), \quad \epsilon := \beta_{k+1} - \beta_1 > 0, \quad (43)$$

*independent of* p *up to finite-size error that decays as* O(p −ϵ ). *If* <sup>k</sup> <sup>=</sup> m, ϵ *is taken to be* ∞ *(perfect finite-size collapse).*

*3. Locally fastest decay of finite-size error. Locally,* γ *is the data exponent that achieves the fastest decay of the finite-size error as measured by* <sup>ϵ</sup>*. In particular,* <sup>ϵ</sup> <sup>=</sup> <sup>O</sup>(|δ|) *for any other data exponent* <sup>γ</sup> ′ <sup>=</sup> <sup>γ</sup> <sup>+</sup> <sup>δ</sup> *with* <sup>δ</sup> ̸= 0*, leading to more slowly decaying finite-size error and therefore a worse collapse.*

*4. Compute-optimality up to a constant suffices. Any training horizon that is a constant multiple of* t ⋆ (p) *preserves the collapse, only changing the constants* b<sup>i</sup> *in Equation* [\(43\)](#page-16-1)*.*

*Proof.* Compute-optimality forces a tie. Fix the compute budget c := 6tp and note t(p) = c/(6p) so that dt dp <sup>=</sup> −t/p. With β<sup>i</sup> := µiγ + ν<sup>i</sup> and b<sup>i</sup> := aiκ −µ<sup>i</sup> ,

$$\frac{dL}{dp} = \sum_{i=1}^m a_i \left( \frac{\partial}{\partial p} + \frac{dt}{dp} \frac{\partial}{\partial t} \right) t^{-\mu_i} p^{-\nu_i} \quad (44)$$

$$= \sum_{i=1}^m a_i \left( -\frac{\nu_i}{p} + \frac{t}{p} \frac{\mu_i}{t} \right) t^{-\mu_i} p^{-\nu_i} \quad (45)$$

$$= \frac{1}{p} \sum_{i=1}^m a_i (\mu_i - \nu_i) t^{-\mu_i} p^{-\nu_i} \quad (46)$$

$$= \frac{1}{p} \sum_{i=1}^m b_i (\mu_i - \nu_i) p^{-\beta_i}. \quad (47)$$

If <sup>β</sup><sup>1</sup> < β2, the leading term <sup>b</sup>1(µ<sup>1</sup> −<sup>ν</sup>1)<sup>p</sup> <sup>−</sup>β<sup>1</sup> cannot cancel the rest for asymtotically large p, contradicting dL dp = 0 required by compute-optimality. Hence at least two indices share the minimum exponent, yielding Equation [\(41\)](#page-16-2).[<sup>1</sup>](#page-17-0)

Asymptotic collapse. We compute ℓ(x, p) explicitly. First, evaluate the loss at the optimal horizon:

$$L(t^*(p), p) - L_0 = \sum_{i=1}^m a_i(t^*(p))^{-\mu_i} p^{-\nu_i} = \sum_{i=1}^m a_i(\kappa p^\gamma)^{-\mu_i} p^{-\nu_i} \quad (48)$$

$$= \sum_{i=1}^m a_i \kappa^{-\mu_i} p^{-\mu_i \gamma - \nu_i} = \sum_{i=1}^m b_i p^{-\beta_i}. \quad (49)$$

Since <sup>β</sup><sup>1</sup> <sup>=</sup> <sup>β</sup><sup>2</sup> <sup>=</sup> · · · <sup>=</sup> <sup>β</sup><sup>k</sup> < βk+1 ≤ · · · ≤ <sup>β</sup>m, we can factor out <sup>p</sup> −β<sup>1</sup> :

$$L({}^{\star}(p), p) - L_0 = p^{-\beta_1} \left( \sum_{i=1}^k b_i + \sum_{i=k+1}^m b_i p^{-(\beta_i - \beta_1)} \right) \quad (50)$$

$$= p^{-\beta_1} \left( \sum_{i=1}^k b_i \right) \left( 1 + O\left( p^{-(\beta_{k+1} - \beta_1)} \right) \right). \quad (51)$$

Similarly, for t = xt<sup>⋆</sup> (p):

$$L(xt^*(p), p) - L_0 = \sum_{i=1}^m a_i (xt^*(p))^{-\mu_i} p^{-\nu_i} = \sum_{i=1}^m a_i x^{-\mu_i} (t^*(p))^{-\mu_i} p^{-\nu_i} \quad (52)$$

$$= \sum_{i=1}^m b_i x^{-\mu_i} p^{-\beta_i} = p^{-\beta_1} \left( \sum_{i=1}^k b_i x^{-\mu_i} \right) \left( 1 + O\left( p^{-(\beta_{k+1} - \beta_1)} \right) \right). \quad (53)$$

Taking the ratio gives:

$$\ell(x, p) = \frac{p^{-\beta_1} \left( \sum_{i=1}^k b_i x^{-\mu_i} \right) (1 + O(p^{-(\beta_k+1-\beta_1)}))}{p^{-\beta_1} \left( \sum_{i=1}^k b_i \right) (1 + O(p^{-(\beta_k+1-\beta_1)}))} = \frac{\sum_{i=1}^k b_i x^{-\mu_i}}{\sum_{i=1}^k b_i} + O\left(p^{-(\beta_k+1-\beta_1)}\right). \quad (54)$$

This produces Equation [\(43\)](#page-16-1) with <sup>ϵ</sup> <sup>=</sup> <sup>β</sup>k+1 − <sup>β</sup><sup>1</sup> <sup>&</sup>gt; <sup>0</sup>.

Locally fastest decay of finite-size error. Let γ be the optimal data exponent and perturb it by δ, writing γ ′ = γ + δ. For a small enough |δ| <sup>&</sup>gt; <sup>0</sup>, the previously tied lowest exponents <sup>β</sup>1, . . . , β<sup>k</sup> split into distinct values <sup>β</sup> ′ 1 , . . . , β′ k , which remain

<sup>1</sup>Here we assumed not all <sup>µ</sup><sup>i</sup> − <sup>ν</sup><sup>i</sup> ̸= 0 for <sup>i</sup> = 1, . . . , k, else these terms would not affect dL/dp. If this is not true, then the loss L(t, p) is not interesting because the it would asymptotically be a function of compute c = 6tp alone, independent of how we allocate c between t and p.

the lowest <sup>k</sup> exponents (since <sup>δ</sup> is small), and the gap between the lowest and second-lowest grows as <sup>O</sup>(|δ|), which is strictly smaller than the previous gap <sup>ϵ</sup> <sup>=</sup> <sup>β</sup>k+1 − <sup>β</sup><sup>1</sup> <sup>&</sup>gt; <sup>0</sup> for sufficiently small <sup>δ</sup>. Therefore, locally <sup>γ</sup> maximizes the decay exponent of the finite-size error, i.e., it gives the best collapse locally.

Compute-optimality up to a constant suffices. Replacing t <sup>⋆</sup> by λt<sup>⋆</sup> multiplies each b<sup>i</sup> by λ <sup>−</sup>µ<sup>i</sup> and leaves the rest of the proof unchanged.

### Remarks.

- 1. Compute-optimal data exponent implies asymptotic collapse, but the converse is not necessarily true when <sup>m</sup> ≥ <sup>3</sup>, since there can be multiple choices of γ that lead to balanced dominant power laws, which imply collapse, but only one of them can be compute-optimal.
- 2. In general, when m > 2, asymptotic instead of exact collapse is the best we can hope for. But an asymptotic collapse alone is not that interesting, since under any choice of the data exponent γ only the terms with the lowest β<sup>i</sup> enter the asymptotic normalized loss curve. For example, if L(t, p) = t <sup>−</sup><sup>µ</sup> + p −ν , any γ > ν/µ will cause only the p −ν term to dominate, leading to <sup>ℓ</sup>(x, p) → <sup>1</sup>, whereas any γ < ν/µ will cause <sup>t</sup> <sup>−</sup><sup>µ</sup> to dominate, leading to <sup>ℓ</sup>(x, p) → <sup>x</sup> <sup>−</sup>µ. The latter case is similar to the infinite-width limit in neural networks, where under t = Θ(1), the loss curves become bottlenecked by training time alone and not model size [\(Vyas et al.,](#page-11-4) [2023;](#page-11-4) [Bordelon & Pehlevan,](#page-9-4) [2022;](#page-9-4) [Yang & Littwin,](#page-11-11) [2023\)](#page-11-11). What is interesting about the collapse that happens under compute-optimal training is that γ is tuned (to ν/µ in this example) so that more than one such term exists, so the collapse reflects a balanced scaling of both training time and model size. This fine balance is also why minor perturbations to γ from the optimal value can significantly disrupt the collapse, which is not true if there is only one dominant power law.

![](_page_18_Figure_5.jpeg)

Figure 11: Sum-of-power-laws fit on additional datasets. Both tasks have loss curves that can be approximated by the sum of two power laws when using a constant learning rate schedule. Fitted constant multipliers are not shown in the legend. To not fit to early-time transients, we fit all steps after the first 0.1B tokens for MLPs, and all steps after the first <sup>0</sup>.1× the compute-optimal horizon for chess. The fit for chess is worse than the other datasets.

# G. A Perturbative Model of Learning Rate Schedules

Let w ′ denote the parameter trajectory under the influence of gradient noise. The dynamics of stochastic gradient descent in gradient flow time are given by

$$\frac{dw'}{d\tau} = -\left(\nabla L(w') + \Sigma^{1/2}(w')\xi(\tau)\right), \quad (55)$$

with noise correlation <sup>E</sup>[ξ(τ )ξ(τ ′ ) <sup>⊤</sup>] = <sup>η</sup>(<sup>τ</sup> )δ(<sup>τ</sup> − <sup>τ</sup> ′ ). For convenience, we rewrite this using ξ(τ ) = η 1/2 (τ ) ˜ξ(τ ) so that

$$\frac{dw'}{d\tau} = -\left(\nabla L(w') + \eta^{1/2}(\tau)\Sigma^{1/2}(w')\tilde{\xi}(\tau)\right), \quad (56)$$

Our strategy is to solve w ′ (τ ) as w(τ ) + δw(τ ) where w(τ ) is the deterministic trajectory satisfying dw dτ <sup>=</sup> −∇L(w), up to leading order in the gradient noise scale ηΣ.

Letting δw := w ′ − <sup>w</sup>, <sup>g</sup> := ∇<sup>L</sup> and taking the difference of the two differential equations:

$$\frac{d(\delta w)}{d\tau} = -\left(g(w') - g(w) + \eta^{1/2}(\tau)\Sigma^{1/2}(w')\tilde{\xi}(\tau)\right). \quad (57)$$

At first order,

$$g(w') \approx g(w) + H(w)\delta w \quad (58)$$

where <sup>H</sup>(w) = ∇<sup>2</sup>L(w) is the Hessian.

Our SDE for δw becomes:

$$\frac{d(\delta w)}{d\tau} = -H(w)\delta w - (\eta\Sigma)^{1/2}\tilde{\xi}(\tau) \quad (59)$$

We define the propagator G(τ, s) that satisfies:

$$\frac{dG(\tau, s)}{d\tau} = -H(w(\tau))G(\tau, s) \quad (60)$$

with G(s, s) = I.

For time-dependent H(w(τ )), the propagator is:

$$G(\tau, s) = \mathcal{T} \exp \left( - \int_s^\tau d\lambda H(w(\lambda)) \right) \quad (61)$$

# where T denotes time-ordering.

Assuming the initial perturbation δw(0) = 0, the solution for δw is:

$$\delta w(\tau) = - \int_0^\tau ds \, G(\tau, s) (\eta \Sigma)^{1/2}(s) \tilde{\xi}(s). \quad (62)$$

Now expanding L(w ′ ) = L(w + δw) to second order in δw gives

$$\begin{aligned}\delta L(\tau) &= L(w'(\tau)) - L(w(\tau)) \\ &\approx g(w(\tau))^\top \delta w(\tau) + \frac{1}{2} \delta w(\tau)^\top H(w(\tau)) \delta w(\tau).\end{aligned}\tag{63}$$

Since <sup>E</sup>[ ˜ξ(s)] = 0 and δw is linear in ˜ξ, <sup>E</sup>[δw(τ )] = 0, so

$$\mathbb{E}[g(w(\tau))^\top \delta w(\tau)] = 0. \quad (64)$$

Thus the leading non-vanishing contribution to the expected loss shift comes from the quadratic term.

Using the solution for δw,

$$\delta w(\tau) \delta w(\tau)^\top = \int_0^\tau ds \int_0^\tau du G(\tau, s) (\eta \Sigma)^{1/2}(s) \tilde{\xi}(s) \tilde{\xi}(u)^\top (\eta \Sigma)^{1/2}(u) G(\tau, u)^\top. \quad (65)$$

Taking the expectation with <sup>E</sup>[ ˜ξ(s) ˜ξ(u) <sup>⊤</sup>] = <sup>δ</sup>(<sup>s</sup> − <sup>u</sup>) <sup>I</sup> gives

$$\mathbb{E}[\delta w(\tau) \delta w(\tau)^\top] = \mathbb{E}\left[\int_0^\tau ds G(\tau, s) \eta(s) \Sigma(w'(s)) G(\tau, s)^\top\right]. \quad (66)$$

Substituting this into the quadratic term gives

$$\mathbb{E}[\delta L(\tau)] = \frac{1}{2} \mathbb{E} \left[ \int_0^\tau ds \operatorname{Tr} [H(w(\tau)) G(\tau, s) \eta(s) \Sigma(w'(s)) G(\tau, s)^\top] \right]. \quad (67)$$

Using Tr[ABC] = Tr[CAB],

$$\mathbb{E}[\delta L(\tau)] = \frac{1}{2} \mathbb{E} \left[ \int_0^\tau ds \operatorname{Tr} [G(\tau, s)^\top H(w(\tau)) G(\tau, s) \eta(s) \Sigma(w'(s))] \right]. \quad (68)$$

This equation is the exact leading-order expression for the noise-induced change in expected loss, which is the equivalent of E(<sup>τ</sup> ) in Equation [\(10\)](#page-5-2). The deterministic loss <sup>L</sup>(w(<sup>τ</sup> )) now plays the role of F(<sup>τ</sup> ) in Equation [\(14\)](#page-6-0). Conceptually, the derivation shows that—although the full dynamics are non-linear and the loss is not assumed quadratic—the perturbation generated by small gradient noises behaves in a simple, linear-quadratic fashion: the weight perturbation δw is linear in the injected noise, and the resulting loss shift δL is quadratic in that perturbation.

Consequently the derivation and final formula completely mirrors the familiar quadratic-loss result, the only difference being that the constant Hessian H is now replaced by the time-dependent Hessian H(w(τ )) carried along the deterministic trajectory. In other words, small gradient noise "sees" the network through an instantaneous linearization, so all schedule effects enter through the propagator G(τ, s), the local Hessian, and the noise covariance, exactly as in the linear case.

Relation to Stochastic Asymptotic Expansion. [Li et al.](#page-10-7) [\(2017\)](#page-10-7) used stochastic asymptotic expansion to expand the dynamics in orders of η 1/2 , but they treat the gradient-noise covariance differently. The stochastic asymptotic expansion in [Li et al.](#page-10-7) [\(2017\)](#page-10-7) first expands the diffusion term along the deterministic path, so Σ can be propagated analytically order by order. Our derivation instead keeps the exact Σ(w ′ (s)) inside the leading-order integral, allowing an empirically measured covariance to be inserted without further approximation, at the cost of analytic closure.

Slow-Variation and Late-Time Limit. As in the quadratic loss case, we can simplify the result under an adiabatic approximation where the Hessian, schedule, and noise covariance changes slowly compared to the time-scale set by the instantaneous Hessian. Specifically, if <sup>H</sup>(w(t)) ≈ <sup>H</sup>(w(<sup>τ</sup> )) := <sup>H</sup> over the support of <sup>G</sup>(τ, s) <sup>⊤</sup>H(w(τ )) G(τ, s), then <sup>G</sup>(τ, s) ≈ <sup>e</sup> −H(τ−s) and G(τ, s) <sup>⊤</sup>HG(τ, s) ≈ H e−2H(τ−s) . Assuming the exponential decay is fast compared to the variation of the noise scale <sup>η</sup>Σ, and taking <sup>τ</sup> → ∞, we have

$$\mathbb{E}[\delta L(\tau)] \approx \frac{1}{4} \operatorname{Tr} [\eta(\tau) \bar{\Sigma}(w'(\tau))], \quad (69)$$

which agrees with the expression for E(<sup>τ</sup> ) in Equation [\(14\)](#page-6-0) for the quadratic loss setting.

Adaptive Optimizers. When using adaptive optimizers with a preconditioner P(t), the SDE becomes

$$\frac{dw'}{dt} = -\eta(t)P^{-1}(t) \left( \nabla L(w') + \Sigma^{1/2}(w')\xi(t) \right). \quad (70)$$

The gradient flow time SDE in Equation [\(56\)](#page-18-2) becomes

$$\frac{dw'}{d\tau} = -P^{-1}(\tau) \left( \nabla L(w') + \eta^{1/2}(\tau) \Sigma'^{1/2} \tilde{\xi}(\tau) \right), \quad (71)$$

where we abbreviated Σ(w ′ ) as Σ ′ . If the preconditioner varies slowly, the dynamics can be treated as if there is no preconditioner, but in a transformed coordinate system:

$$\tilde{w}'(t) = P^{1/2}(t) w'(t). \quad (72)$$

Differentiating and neglecting the O(P˙) term gives

$$\frac{d\tilde{w}'}{d\tau} = P^{1/2} \frac{dw'}{d\tau} = -P^{-1/2} \left( \nabla L(w') + \eta^{1/2} \Sigma'^{1/2} \tilde{\xi} \right) \quad (73)$$

$$= -\tilde{g}(\tilde{w}') - \underbrace{(\eta P^{-1} \Sigma')^{1/2} \xi}_{\text{noise}}. \quad (74)$$

The deterministic trajectory in this coordinate system (w˜(t) = P 1/2 (t) w(t)) satisfies

$$\frac{d\tilde{w}}{d\tau} = -\tilde{g}(\tilde{w}). \quad (75)$$

To dervie the SDE for δw˜ := ˜w ′ − w, at first order, we have

$$\tilde{g}(\tilde{w}') - \tilde{g}(\tilde{w}) = P^{-1/2}H\delta w = P^{-1/2}HP^{-1/2}\delta\tilde{w} := \tilde{H}\delta\tilde{w}, \quad (76)$$

where H˜ is the preconditioned Hessian. Therefore,

$$\frac{d\delta\tilde{w}}{d\tau} = -\tilde{H}\delta\tilde{w} - (\eta P^{-1}\Sigma')^{1/2}\xi, \quad (77)$$

It is also easy to show that the preconditioned Hessian indeed governs the leading order perturbation to the expected loss in the transformed coordinates

$$\mathbb{E}[\delta L] \approx \frac{1}{2} \mathbb{E}[\delta \tilde{w}^\top \tilde{H} \delta \tilde{w}]. \quad (78)$$

Given Equation [\(77\)](#page-21-1) and Equation [\(78\)](#page-21-2), all steps in the previous derivation now applies after swapping <sup>H</sup> → <sup>H</sup>˜ and Σ ′ → <sup>P</sup> <sup>−</sup>1/<sup>2</sup>Σ ′P −1/2 := Σ˜′ . The final result for the noise-induced loss perturbation in the slow-variation and late-time limit is

$$\mathbb{E}[\delta L(\tau)] \approx \frac{1}{4} \text{Tr} \left[ \eta(\tau) \bar{\Sigma}(w'(\tau)) \right]. \quad (79)$$

Limitations. It is worth highlighting some limitations of our analysis. First, due to the non-linear nature of neural networks, it is known that gradient flow can fail to model full-batch gradient descent with a finite step size, which exhibits effects such as the Edge of Stability [\(Cohen et al.,](#page-9-16) [2021;](#page-9-16) [2022;](#page-9-17) [2024\)](#page-9-12). Similarly, in the stochastic case, there is a strong coupling between learning rate, batch size, and the Hessian spectrum [\(Agarwala & Pennington,](#page-9-8) [2024\)](#page-9-8). These phenomena suggest that it is unlikely that we can fully model the effect of a time-varying learning rate schedule simply as injecting a schedule-dependent noise component on top of the deterministic trajectory that is itself independent of the schedule, as changing the learning rate also pushes the model into different regions of the parameter space based on curvature (though it is possible that the leading-order perturbation theory can already capture this effect to some extent).

Second, when dealing with adaptive optimizers, we assumed the preconditioner stays the same (as a function of τ ) between the deterministic and stochastic trajectories. For typical optimizers, such as Adam [\(Diederik P. Kingma,](#page-9-18) [2015\)](#page-9-18), this assumption is not correct as P depends on the gradient covariance, which can differ between the two trajectories, i.e., Σ(w(<sup>τ</sup> )) ̸= Σ(<sup>w</sup> ′ (τ )). This can introduce an additional term in the SDE for δw˜, present even at first order, which we did not model.

The empirical effectiveness of our model for predicting the loss curves under different schedules suggests it is nevertheless on the right track, and that there may be other ways to derive similar or improved predictions with more accurate assumptions.

# H. Computing ∆˜ to Leading Order

Let <sup>ψ</sup>(<sup>τ</sup> ) = <sup>L</sup>(τ)−L¯(τ) L¯(τ) and define

$$\tilde{\Delta}^2(\tau) = \mathbb{E}[(\psi(\tau) - \psi(\tau^*))^2] \quad (80)$$

with τ <sup>⋆</sup> <sup>=</sup> <sup>τ</sup> <sup>+</sup> δτ and <sup>0</sup> < δτ ≪ <sup>1</sup>. Because L¯(<sup>τ</sup> ⋆ ) <sup>−</sup><sup>1</sup> <sup>=</sup> L¯(<sup>τ</sup> ) <sup>−</sup><sup>1</sup> + O(δτ ),

$$\tilde{\Delta}^2(\tau) = \bar{\mathcal{L}}^{-2}(\tau) \mathbb{E} \left[ (\Delta \mathcal{L}(\tau) - \Delta \mathcal{L}(\tau^*))^2 \right] + O(\delta \tau^2), \quad (81)$$

where <sup>∆</sup>L(<sup>τ</sup> ) = L(<sup>τ</sup> ) − L¯(<sup>τ</sup> ). For the quadratic model,

$$\Delta\mathcal{L}(\tau) = g(\tau)^\top \Delta w(\tau) \quad (82)$$

to first order in

$$\Delta w(\tau) = \int_0^\tau ds e^{-H(\tau-s)} \Sigma^{1/2}(s) \xi(s) \quad (83)$$

with

$$\mathbb{E}[\xi(s)\xi(s')] = \eta(s)\delta(s-s')I. \quad (84)$$

Splitting the upper limit at τ and expanding g(τ ⋆ ) = g(τ ) + O(δτ ) gives

$$\Delta \mathcal{L}(\tau) - \Delta \mathcal{L}(\tau^*) = g(\tau)^\top \underbrace{\int_\tau^{\tau^*} ds \, e^{-H(\tau^*-s)} \Sigma^{1/2}(s) \xi(s)}_{O(\delta\tau^{1/2})} + O(\delta\tau) \quad (85)$$

where the remainder collects two subleading O(δτ ) contributions:

$$R_1 := g(\tau)^\top \int_0^\tau ds \left[ e^{-H(\tau^*-s)} - e^{-H(\tau-s)} \right] \Sigma^{1/2}(s) \xi(s), \quad (86)$$

$$R_2 := [g(\tau^*) - g(\tau)]^\top \Delta w(\tau) = \dot{g}(\tau)^\top \Delta w(\tau) \delta \tau + O(\delta \tau^2). \quad (87)$$

Using e −H(τ <sup>⋆</sup>−s) = I + O(δτ ), η(s) = η(τ ) + O(δτ ), and Σ( ¯ s) = Σ( ¯ τ ) + O(δτ ) inside the leading-order integral,

$$\mathbb{E} \left[ (\Delta \mathcal{L}(\tau) - \Delta \mathcal{L}(\tau^*))^2 \right] = g(\tau)^\top \int_\tau^{\tau^*} ds \eta(s) e^{-2H(\tau^*-s)} \bar{\Sigma}(s) g(\tau) + O(\delta\tau^2) \quad (88)$$

$$= g(\tau)^\top \eta(\tau) \bar{\Sigma}(\tau) g(\tau) \delta \tau + O(\delta \tau^2). \quad (89)$$

Substituting this into the expression for ∆˜ <sup>2</sup> (τ ) yields the desired result

$$\tilde{\Delta}^2(\tau) = \tilde{\mathcal{L}}^{-2}(\tau) \, g(\tau)^\top \eta(\tau) \tilde{\Sigma}(\tau) g(\tau) \, \delta \tau + O(\delta \tau^2). \quad (90)$$

# I. Additional Results on Learning Rate Schedules

MLP fits. Figure [12](#page-22-1) shows our predictions for MLP loss curves. With a single α = 0.26 (very close to 1/4), we obtain excellent fits across schedules, model sizes, and training horizons.

![](_page_22_Figure_17.jpeg)

Figure 12: A simple model predicts MLP loss curves trained across learning rate schedules, model sizes p, and training horizons T on the synthetictic regression task. Dashed curves show the predicted loss as L ′ (τ ) = L(τ ) + α δη(τ ) Tr(Σ′ (τ )) (Equation [\(18\)](#page-7-1)). α is the only free parameter and is set to 0.26. Each curve is smoothed with an exponential moving average with half life equal to 1% of total steps.

Effect of Dropping η Tr(δΣ). Figure [13](#page-23-0) (top row) shows δη Tr(Σ′ ) is typically 3 to 10 times than η Tr(δΣ) in absolute value in both CIFAR-5M transformer and MLP regression experiments. Since we decay the learning rate to zero (δη is comparable η), this means the gradient covariance does not change much between the constant learning rate and other schedules, which can happen if the Hessian trace does not change much. The fact that this ratio is only moderately large shows that the gradient covariance or the Hessian trace did change considerably due to decaying the learning rate, which is what we expect due to a generically inverse relation between learning rate and Hessian eigenvalues [\(Cohen et al.,](#page-9-16) [2021;](#page-9-16) [Agarwala & Pennington,](#page-9-8) [2024\)](#page-9-8). However, somewhat puzzlingly, we found that including the smaller term η Tr(δΣ) can produce worse fits, particularly for the slow oscillatory schedule (0.5 cos(3πx)), and makes the optimal constant α more schedule-dependent.

![](_page_23_Figure_2.jpeg)

Figure 13: Out of the two terms that make up Tr(δ(ηΣ)), the term η Tr(δΣ) is typically 3 to 10 times smaller than δη Tr(Σ′ ) (top row). Moreover, including it sometimes produces a worse fit and make the optimal α vary more across schedules (bottom row). We determine α for each schedule by matching the prediction with the observation at the end point.

![](_page_23_Figure_4.jpeg)

Figure 14: Universality of gradient noise in MLPs. Fixing a learning rate schedule, the ratio Tr(Σ)/L is approximately a function of normalized compute alone, independent of model size. On this regression task, the estimated irreducible loss is negligible so L ≈ L.