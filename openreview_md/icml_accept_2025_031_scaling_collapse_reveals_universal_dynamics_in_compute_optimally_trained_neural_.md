# Scaling Collapse Reveals Universal Dynamics In Compute-Optimally Trained Neural Networks

Shikai Qiu 1 † Lechao Xiao 2 Andrew Gordon Wilson 1 Jeffrey Pennington 2 **Atish Agarwala** 2

## Abstract

What scaling limits govern neural network training dynamics when model size and training time grow in tandem? We show that despite the complex interactions between architecture, training algorithms, and data, compute-optimally trained models exhibit a remarkably precise universality. Specifically, loss curves from models of varying sizes collapse onto a single universal curve when training compute and loss are normalized to unity at the end of training. With learning rate decay, the collapse becomes so tight that differences in the normalized curves across models fall below the noise floor of individual loss curves across random seeds, a phenomenon we term supercollapse. We observe supercollapse across learning rate schedules, datasets, and architectures, including transformers trained on next-token prediction, and find it breaks down when hyperparameters are scaled suboptimally, providing a precise and practical indicator of good scaling. We explain these phenomena by connecting collapse to the power-law structure in typical neural scaling laws, and analyzing a simple yet surprisingly effective model of SGD noise dynamics that accurately predicts loss curves across various learning rate schedules and quantitatively explains the origin of supercollapse.

## 1. Introduction

As machine learning systems grow in scale, accurate predictive models of their training dynamics become increasingly valuable, both for interpreting costly experiments and for designing robust, efficient training pipelines (Wortsman et al.,
†Work done partly during an internship at Google Deep-
Mind 1New York University 2Google DeepMind. Correspondence to: Shikai Qiu <sq2129@nyu.edu>, Atish Agarwala <thetish@google.com>.

2023; Achiam et al., 2023; Xiao, 2024). While the complexity of modern architectures, optimizers, and datasets often renders exact, first-principles analyses intractable for any individual model, recent work shows that some key aspects of training are predictable when we focus on their scaling behavior across a family of models. Examples include empirical power-law relations linking optimal final loss, model size, dataset size, and compute budget under computeoptimal training, known as neural scaling laws (Hestness et al., 2017; Kaplan et al., 2020; Sharma & Kaplan, 2022; Hoffmann et al., 2022), as well as hyperparameter transfer from small to large models based on infinite-width or depth limits of training dynamics (Yang et al., 2021; Bordelon et al., 2023; Everett et al., 2024; Bordelon et al., 2024c). In this work, we show the entire training process follows highly predictable scaling, beyond final losses and optimal hyperparameters. We find that the entire loss curves of compute-optimally trained models exhibit a precise scaling symmetry, collapsing onto a single universal curve across models after a simple normalization. Learning rate decay amplifies this effect dramatically, producing what we call supercollapse: collapse so tight that cross-scale differences fall below the noise floor of individual loss curves due to random seeds. Figure 1 (a-d) summarizes these results. These findings advance our understanding in two key ways.

First, while Kaplan et al. (2020, Figure 11) found the loss curves roughly follow a sum of power laws, we identify that loss curves follow a universal shape with far greater precision. For typical learning rate schedules, this shape deviates from simple power laws and may not admit any obvious functional form. Second, our work provides compelling empirical evidence for a well-defined joint scaling limit where model size and training time grow together under compute-optimal allocation. This limit contrasts with traditional infinite-width or depth limits that fix training duration (Yang & Hu, 2021; Bordelon & Pehlevan, 2022). While these theories predict initial dynamical consistency, accumulating finite-size effects lead to gradual divergence as training progresses, as demonstrated by Vyas et al. (2023). In contrast, the collapse we observe reveals a joint scaling limit that preserves consistency throughout training, precisely the regime relevant for practical large-scale training.

1

0.0 0.2 0.4 0.6 0.8 1.0 Normalized Compute 10−4 10−3 10−2 10−1 3.16 3.17 3.18 3.19 N
or m al iz ed Loss Width 768 896 1024 1152 1280 1536 1792 2048 L0 + ac−b a = 0.154 b = 0.191 L0 = 3.132 0.0 0.2 0.4 0.6 0.8 1.0 Normalized Compute 1.0 1.2 1.4 1.6 1.8 R
el.

 Va ria ti on Loss Noise floor σ Collapse deviation ∆
103 104 Compute (PetaFLOPs)
(a) Compute-Optimal Scaling Law
(b) Supercollapse
(c) Collapse Below Noise Floor 1.0 1.5 2.01 − x 2 0.5(1 + cos(πx)) 1 − x (20%)
10−2 10−1 100 1 − τ ˆ
10−4 10−3 10−2 10−1 const 1 − x 0.5(1 + cos(πx))
0.5(1 + cos(3πx))
1 − x (20%)
0.5(1 + cos(3πx)) (20%) Observed Prediction 0 2 4 6 8 10 Tokens (Billion)
3.17 3.18 3.19 3.20 Co lla pse Dev.

/

√

η N
or ma liz ed Lo ss Loss 0 0.5 1 1.0 1.5 2.0 1 − x 2 (20%)
0.5(1 + cos(3πx)) (20%)
0.5(1 + cos(3πx))
√
1 − τ ˆ
0 0.5 1 0 0.5 1 Normalized Compute
(d) Collapse in Different Schedules
(e) Predicting Loss Curve per Schedule
(f) Predicting Collapse Quality
We provide an elementary theoretical analysis that reveals the key mechanisms behind this precise collapse. We first show that for loss curves following typical neural scaling laws, collapse occurs precisely when models are trained for constant multiples of their compute-optimal horizons (Section 3.1). We then analyze a simple theoretical model of the SGD noise dynamics that predicts loss curves under a variety of learning rate schedules remarkably well (Section 3.2), and explains two key observations: why normalized curves retain universal form despite losing their power-law structure, and how learning rate decay suppresses variance to produce supercollapse (Section 3.3).

Beyond theoretical interest, supercollapse provides a practical scaling diagnostic, as we find that deviations from collapse can signal misconfigured scaling choices, such as suboptimal scaling of learning rate and data (Figure 4). Overall, our results suggest supercollapse provides a novel, powerful tool to study scaling. Our code can be found here.

## 2. Empirical Observations

We demonstrate our main empirical findings in this section, independently on multiple tasks and architectures which can be studied even in academic settings.

## 2.1. Experiment Setup

In each task, we train a sequence of models with increasing compute, scaling hyperparameters such as data, initialization, and learning rate with the model. We refer to a sequence of training configurations as a scaling ladder. We provide further experimental details in Appendix A. We focus on width scaling, where hyperparameter transfer is most well-studied, but find scaling transformer depth leads to similar results in Appendix B, suggesting our observations may generalize to more general scaling ladders where width, depth, batch size, weight decay, etc. can be co-scaled. Transformers Next-Token Prediction. We consider two next-token prediction tasks: 1) CIFAR-5M (Nakkiran et al., 2020), a dataset of 6M generated CIFAR-like images, and 2) Lichess, a collection of chess games recorded in algebraic chess notation where the goal is to predict the next move in the game. Our scaling ladder includes models with about 10M to 80M parameters, approximately log-uniformly spaced, by scaling the width (embedding dimensions) from 768 to 2048 and fixing the number of blocks to 3. All models use µP (Yang & Hu, 2021; Yang et al., 2021) for initialization and learning rates, and are trained with Adam.

MLPs on Power-Law Fourier Features. To investigate other architectures and training objectives, we train 7-layer MLPs with varying widths from 384 to 2048 on a synthetic regression task. The target function has a power-law Fourier spectrum, designed to elicit the power-law scaling laws observed in natural data. We count each example as 1 token.

## 2.2. Estimating Compute-Optimal Scaling Laws

Let L(*t, p, ω*) be the test loss after t tokens (proportional to steps) for a model with p parameters trained with random seed ω. We estimate the compute-optimal training horizon in tokens for a p-parameter model as t
⋆(p) = (p/p0)
γ, where γ is the data exponent, by extracting the Pareto frontier of expected loss (estimated using 5 seeds) vs. compute under a constant learning rate schedule, following a procedure similar to Approach 1 in Hoffmann et al. (2022),
with compute estimated as c = 6tp FLOPs (Kaplan et al., 2020). We reuse the same t
⋆(p) as the training horizon for other learning rate schedules, which prior work suggests is optimal up to a constant factor (Pearce & Song, 2024). For each task and schedule, we fit the resulting compute-optimal scaling law using the form L0 + ac−b(Figure 1a), for constants L0*, a, b* ≥ 0. Following Sharma & Kaplan (2022) and Hoffmann et al. (2022), we refer to L0 as the estimated irreducible loss. Using the best-fit L0, we define the reducible loss curve L(*t, p, ω*) = L(*t, p, ω*) − L0. We detail the procedure for fitting the compute-optimal training horizon in Appendix C. "Compute-optimal" here primarily refers to the choice of training horizon, not of all hyperparameters.

## 2.3. Scaling Collapse Of Compute-Optimal Loss Curves

The loss curves for different model sizes cover varying ranges of compute and loss values, but appear to follow a consistent shape, which motivates us to affinely rescale them to the *normalized loss curve* ℓ given by

$$\ell(x,p,\omega)=\frac{L(xt^{\star}(p),p,\omega)-\hat{L}}{L(t^{\star}(p),p,\omega)-\hat{L}},\quad x\in[0,1],\tag{1}$$

Nor mali zed L
oss ˆ
L = 0 Nor mali zed L
oss ˆ
L = 3.12 0.0 0.2 0.4 0.6 0.8 1.0 Normalized Compute 1.0 1.2 1.4 1.6 0.0 0.2 0.4 0.6 0.8 1.0 Normalized Compute 1.000 1.002 1.004 1.006 Norma lized Lo ss ˆ L = L0 = 3.132 ˆ
L = 3.14 0.0 0.2 0.4 0.6 0.8 1.0 Normalized Compute 1.00 1.25 1.50 1.75 2.00 2.25 0.0 0.2 0.4 0.6 0.8 1.0 Normalized Compute 1.0 1.2 1.4 1.6 1.8 Norma lize d Loss 0.0 0.2 0.4 0.6 0.8 1.0 Normalized Compute 10−4 10−3 10−2 10−1 Nor mal ized Loss Width 768 896 1024 1152 1280 1536 1792 2048 0.0 0.2 0.4 0.6 0.8 1.0 Normalized Compute 1.0 1.2 1.4 1.6 Rel.

 Vari atio n Noise floor σ Collapse deviation ∆
for some offset L. ˆ We refer to x as the normalized compute.

Note the denominator uses the stochastic final loss value specific to the random seed. We set Lˆ = L0 to subtract the estimated irreducible loss that bottlenecks the asymptotic performance, leading to ℓ(*x, p, ω*) = L(xt⋆(p)*,p,ω*)
L(t
⋆(p)*,p,ω*)
.

Remarkably, we observe that the family of normalized loss curves is nearly identical across p, revealing equal rates of relative progress (Figure 1b). We say these curves *collapse*, as the phenomenon resembles the ubiquitous scaling collapse found in statistical physics, theoretical biology, and other sciences, where observables from systems of different sizes collapse onto a single curve after appropriate rescaling (see Appendix D for further discussion). We found setting Lˆ = L0 achieves the best collapse (Figure 2).

## 2.4. Quantifying The Collapse Quality

We quantify the quality of collapse using the collapse deviation ∆, defined as:

$$\Lambda(x)=\frac{\mathbb{V}_{p,\omega}[\ell(x,p,\omega)]^{1/2}}{\mathbb{E}_{p,\omega}[\ell(x,p,\omega)]}\,,\tag{2}$$

where Ep,ω and Vp,ω denote the expectation and variance over the random seed and the empirical distribution of model size p in the scaling ladder (approximately log-uniformly distributed). The collapse deviation measures the relative variation of the normalized curves across p. For perspective, we compare it to the per-model (relative) noise floor:

$$\sigma(x,p)=\frac{\mathbb{V}_{\omega}[\mathcal{L}(xt^{\star}(p),p,\omega)]^{1/2}}{\mathbb{E}_{\omega}[\mathcal{L}(xt^{\star}(p),p,\omega)]},\tag{3}$$

which measures the relative fluctuation in the reducible loss curve for each model size p across random seeds. By the definition of ℓ, ∆(1) = 0 always. As seen in Figure 3, for a constant learning rate, ∆(x) quickly rises to a level comparable to σ(*x, p*), and remains at that level for most x < 1. This observation shows that variations in the normalized curves arise primarily from seed-to-seed fluctuations rather than model-to-model differences, quantitatively demonstrating that the observed collapse is non-trivial.

0.0 0.2 0.4 0.6 0.8 1.0 Normalized Compute 1.0 1.5 2.0 2.5 3.0 100 101 102 103 Compute (PetaFLOPs)
0.016 0.024 0.032 0.040 0.048 N
or m ali zed L
oss Width 384 512 645 812 1024 1290 1625 2048 N
or m ali zed L
oss Width 384 512 645 812 1024 1290 1625 2048 Const LR µP
0.0 0.2 0.4 0.6 0.8 1.0 Normalized Compute 1.0 1.2 1.4 1.6 Loss
(a) MLP µP vs Constant LR
(b) µP (supercollapse)
(c) Constant LR (no collapse)
102 103 104 Compute (PetaFLOPs)
0.650 0.675 0.700 0.725 0.750 N
or m al iz ed L
oss Width 768 896 1024 1152 1280 1536 1792 2048 0.0 0.2 0.4 0.6 0.8 1.0 Normalized Compute 1.0 1.1 1.2 1.3 1.4 N
or m al iz ed L
oss Width 768 896 1024 1152 1280 1536 1792 2048 Over-trained Optimal 0.0 0.2 0.4 0.6 0.8 1.0 Normalized Compute 1.0 1.1 1.2 1.3 Loss
(d) Transformer Optimal vs Over-trained
(e) Optimal (supercollapse)
(f) Over-trained (no collapse)

## 2.5. Supercollapse: Consistency Below The Noise Floor

Remarkably, with learning rate decay, we find that the collapse deviation is less than the noise floor for a significant fraction of training; that is, ∆(x) < σ(*x, p*) for x > 1 − δ for some moderate δ as large as 0.5 (Figure 1c). We refer to this stronger form of collapse as *supercollapse* (in contrast to the collapse in Figure 3). Supercollapse appears in the decay phase of all tested learning rate schedules that decay to zero (Figure 1d). All schedules are defined in terms of relative training fraction, i.e., the learning rate is a fixed function of the normalized compute x across model sizes. Under supercollapse, self-normalized loss curves from different models collapse better than our ability to predict any individual model's loss. Normalizing by the final loss of the particular realization of the stochastic loss curve is key to supercollapse, which reduces variance by exploiting correlations at different times along a single optimization trajectory. We explain this mechanism in detail in Section 3.3.

## 2.6. Suboptimal Scaling Breaks Supercollapse

Supercollapse provides a practical method for comparing inherently noisy training loss curves across model scales with precision that exceeds naive noise floor estimates, without the need for expensive multi-seed experiments typically required to obtain equally clean signals. This comparison can provide valuable diagnostic information about scaling where the ability to distinguish small signal from noise is often crucial (Xiao, 2024), which we now demonstrate.

Model Parameterization. Carefully parameterizing the model, i.e., scaling the initialization, learning rate, and possibly other hyperparameters as model size increases, is crucial for achieving stable and efficient training at scale (Yang et al., 2021; Bordelon et al., 2023; 2024c; Everett et al., 2024). When models are trained in the wrong parameterization, we expect the loss curves not to collapse due to a lack of consistent training dynamics across scales. Using the MLP setup, we show that replacing µP with a constant learning rate across widths breaks the collapse (Figure 4, top row). Remarkably, the normalized loss curves expose inconsistent dynamics even at small scales where the final losses are virtually identical between constant and µP scaling, demonstrating that the collapse is a more sensitive probe of scaling behavior than final performance alone. Compute-Optimal Data Exponent. For language models, Kaplan et al. (2020) showed that compute-optimal training corresponds to training each model to a fixed multiple of its converged loss. If this principle generalizes to our setting, the data exponent γ should match the compute-optimal value for collapse to occur. For example, when γ exceeds the optimal value, larger models will make more rapid relative initial progress but decelerate later as a function of normalized compute, causing their normalized curves to shift downward. We indeed find this shift in Figure 4 (bottom row). This sensitivity suggests a novel application: rather than fitting power laws to sparse points on the Pareto frontier, one could tune γ to maximize collapse quality, leveraging the full statistical power of entire loss curves.

Norm aliz ed Loss γ = 0.9γ
?

0.00 0.25 0.50 0.75 1.00 1.00 1.25 1.50 1.75 2.00 γ = γ
?

0.00 0.25 0.50 0.75 1.00 1.00 1.25 1.50 1.75 2.00γ = 1.1γ
?

L = t−0.5 + p−0.5 Scaling Law 105 1010 1015 1020 10−4 10−3 10−2 10−1 100 p 101 105 109 0.00 0.25 0.50 0.75 1.00 1.0 1.5 2.0 2.5 3.134 + t−0.41 + p−0.37
(R
2 = 0.999)
Compute-Optimal 102 103 104 Compute (PetaFLOPs)
3.16 3.17 3.18 3.19 3.20 3.21 Loss Lo ss 106 1012 1018 Compute (6tp)
10−9 10−6 10−3 100 L = t−1.0 + t−0.2p−1.0 + p−1.2 Nor mali zed Los s γ γ
?

0.9γ
?

1.1γ
?

0.00 0.25 0.50 0.75 1.00 Normalized Compute 2 4 6 8 0.00 0.25 0.50 0.75 1.00 Normalized Compute 1 2 3 4 0.00 0.25 0.50 0.75 1.00 Normalized Compute 1 2 3 Loss
(a) CIFAR-5M Fit

## 3. Explaining Loss Curve Scaling Collapse

In this section, we investigate theoretical explanations for the scaling collapse of compute-optimal loss curves and supercollapse. Our analysis starts with a simple observation: the numerator of the collapse deviation ∆(x) can be decomposed as:
$\mathbb{V}_{p,\omega}[\ell(x,p,\omega)]=\mathbb{V}_{p}\mathbb{E}_{\omega}[\ell(x,p,\omega)]+\mathbb{E}_{p}\mathbb{V}_{\omega}[\ell(x,p,\omega)]$.  
(4)
The first term corresponds to the variation between different scales p after averaging over all sources of randomness. We will first show how this term can be small:
- In Section 3.1, we prove that for a family of powerlaw neural scaling laws, compute-optimal loss curves indeed collapse after normalization. We show loss curves in our experiments fall into this family when using a constant learning rate schedule.

- In Section 3.2, we develop a simple theoretical model that successfully predicts the empirical loss curves under various learning rate schedules and explains why they collapse despite deviating from power laws. Given its effectiveness, we believe this model has value for understanding learning rate schedules more broadly.

We then analyze the second term, which captures the loss variance due to random seeds, averaged across model sizes:
- In Section 3.3, we show the same noise model enables us to reason about the noise in the loss curves, and quantitatively predict the variance reduction effect in supercollapse.

Together these findings provide an initial theoretical explanation for supercollapse, and uncover promising directions for future theoretical work.

## 3.1. Scaling Collapse From Power-Law Scaling

$$\mathbb{V}_{\omega}[\ell(x,p,\omega)$$

In this section, we consider deterministic models of the loss curves and assume all randomness has been averaged out.

Power-Law Pareto Frontier is Necessary. For a family of differentiable loss curves L(*t, p*), the compute-optimal loss frontier after subtracting Lˆ must follow a power law for our affine transformation to induce scaling collapse (proof in Appendix E). The key insight is that collapse requires the transformed loss curves to be related by multiplicative scaling, equivalently translation in log-log space, where the frontier must have constant log-log slope since it remains tangent to shifted versions of the same curve. This motivates choosing Lˆ = L0, which by definition yields the best powerlaw Pareto frontier. However, a sufficient condition for scaling collapse requires an explicit form of L(*t, p*). Neural Scaling Laws. Motivated by empirical neural scaling laws in natural data (Hestness et al., 2017; Kaplan et al., 2020; Hoffmann et al., 2022), we consider expected loss curves following a sum-of-power-laws scaling of the form

$$L(t,p)=L_{0}+t^{-\mu}+p^{-\nu}$$
$$({\boldsymbol{5}})$$
−ν(5)
for constants L0 ≥ 0*, µ, ν >* 0, with potential constant multipliers absorbed via an appropriate choice of units. In Figure 5a, we show the CIFAR-5M loss curves are well-fit by Equation (5) if trained under a constant learning rate schedule (averaged across 5 seeds). We also find decent fits in other datasets in Figure 11.

Equivalence by Balance of Power Laws. As before, let t
⋆(p) denote the training horizon. We will examine conditions under which t
⋆(p) (a) is compute-optimal, and (b)
results in scaling collapse. We assume deterministic loss curves for now and omit the argument ω. To find computeoptimal t
⋆(p), we fix c so that t(p) = c/(6p) and minimize the loss L(t(p), p) = t(p)
−µ +p
−ν with respect to p by setting 0 = dL
dp 
=
∂L
∂t dt dp 
+
∂L
∂p 
= −µt−µ−1(−t/p) − νp−ν−1

$$\Rightarrow\mu t^{-\mu}=\nu p^{-\nu}$$
=⇒ µt−µ = νp−ν(6)
which yields t
⋆(p) = r
−1/µp ν/µ, with r = *ν/µ.* Under this scaling, the normalized loss curves are:

$$\ell(x,p)={\frac{(x t^{\star})^{-\mu}+p^{-\nu}}{(t^{\star})^{-\mu}+p^{-\nu}}}={\frac{r x^{-\mu}p^{\nu\ell}+p^{\nu\ell}}{r p^{\nu\ell}+p^{\nu\ell}}}={\frac{r x^{-\mu}}{r}}$$

rx−µ + 1
r + 1.
$$\frac{{}^{\mathrm{t}}+1}{{}^{\mathrm{t}}+1}$$

All p dependencies cancel, leaving the final expression independent of p and giving us an exact collapse. Moreover, it is clear that this is the unique choice for t
⋆(p) up to a constant multiplier that leads to such cancellation. This agreement is not an accident: compute-optimal scaling requires balancing the derivatives of two power laws, while collapse requires balancing the power laws themselves. For power laws, these two conditions coincide, up to a multiplicative constant. In Figure 5b, we numerically verify the agreement between collapse and compute-optimal scaling. When the data exponent γ deviates from the optimal value ν/µ, we observe a suboptimal scaling law and no collapse. Note that the absence of an irreducible term in ℓ is also necessary. Had we set Lˆ = L0 + E for some E ̸= 0 in Equation (1), we would instead have ℓ(*x, p*) = (xt⋆)
−µ+p
−ν+E
(t⋆)−µ+p−ν+E
, where no t
⋆(p) can leave the numerator and denominator homogeneous in p. In Appendix F, we study the more general form

$$L(t,p)=L_{0}+\sum_{i=1}^{m}a_{i}t^{-\mu_{i}}p^{-\nu_{i}},$$

which naturally arises in theoretical models of neural scaling laws (Paquette et al., 2024b; Bordelon et al., 2024a;b), and show that compute-optimality implies scaling collapse by balancing the two dominant terms, though with m > 2 the collapse is only exact asymptotically. Together with the close empirical fit in Figure 5a, this analysis explains scaling collapse in the constant learning rate setting; however Equation (5) fails to fit the empirical loss curves with most learning rate schedules, as varying the learning rate can modulate the loss curve in quite arbitrary ways, clearly shown in Figure 1d. Why, then, does the collapse transfer to other schedules?

## 3.2. Universality Of Learning Rate Schedules

To understand why scaling collapse is robust across learning rate schedules, we develop a quantitative model for how learning rate schedules affect the loss curves. While an

$$(6)$$

exact theoretical model seems out of reach for the realistic training setup, we show that a simple model based on quadratic loss analysis proves surprisingly effective. Under this model, we demonstrate that although learning rate schedules deform the loss curves in a schedule-dependent way, the deformation is approximately independent of p. We consider stochastic effects that depend on the random seed ω, but omit ω as an explicit argument for brevity and use bar to denote expectation over ω.

$3.2.1.\text{A}$ 9. 

## 3.2.1. A Simple Model For Lr Schedules

Let w(t) and L(w(t)) denote the parameters and loss at step t, we can model the dynamics of full-batch gradient descent under a small learning rate η(t) with a gradient flow dw dt = −η(t)∇L(w(t)). To model stochastic effects, a noise term is added to the gradient, leading to the SDE dw dt = −η(t)∇L(w) + Σ1/2(w)ξ(t)(Li et al.,
2017; Malladi et al., 2022), where the *mini-batch* gradient noise Σ
1/2(w)ξ(t) satisfies E[ξ(t)ξ(t
′)] = δ(t − t
′)I, and we allow its covariance (which depends on batch size) Σ(w) to be a function of the parameters. Prior works have used the SDE model or discrete variants to study learning rate schedules in analytically tractable problems (Zhang et al., 2019; d'Ascoli et al., 2022; Wen et al., 2024), but we will show it can make surprisingly accurate predictions in real models. We work in *gradient flow time* τ (t) = R t 0 η(s)ds, where

$$\frac{dw}{d\tau}=-\Big{(}\nabla L(w)+\Sigma^{1/2}(w)\xi(\tau)\Big{)},\tag{9}$$

and E[ξ(τ )ξ(τ
′)] = δ(t − t
′)I = η(τ )δ(τ − τ
′)I. We overload the notation and use η(τ ), w(τ ), and L(τ ) to denote the evolution of these quantities in gradient flow time.

$$({\boldsymbol{8}})$$

Quadratic Loss. For the moment, let us suppose the loss function is quadratic L(w) = 12w
⊤Hw, where we assume the minimum is at the origin without loss of generality. Then
∇L(w) = Hw and standard calculation shows

$$w(\tau)=e^{-H\tau}w(0)-\int_{0}^{\tau}ds\,e^{-H(\tau-s)}\Sigma^{1/2}(w(s))\xi(s).\tag{10}$$

Letting Σ( ¯ s) = E[Σ(w(s))], the expected loss is then

$$\bar{L}(\tau)=\underbrace{\frac{1}{2}\mathbb{E}\big{[}\|e^{-H\tau}w(0)\|_{H}^{2}\big{]}}_{\mathscr{F}(\tau)}$$ $$+\underbrace{\frac{1}{2}\int_{0}^{\tau}ds\,\eta(s)\,\mathrm{Tr}\Big{(}He^{-2H(\tau-s)}\bar{\Sigma}(s)\Big{)}}_{\mathscr{E}(\tau)}.\tag{11}$$

The first term F(τ ) is the forcing function, equal to the expected loss curve in the noiseless limit ηΣ → 0 and is

p = 20M, T = 10B
p ={12M, 31M, 79M}, T = 10B
const 1 − x 1 − x (20%)
0.5(1 + cos(3πx)) (20%)
p = 20M, T = {2B, 5B, 10B, 20B}
const 1 − x 1 − x (20%)
0.5(1 + cos(3πx)) (20%)
const 1 − x 0.5(1 + cos(πx))
0.5(1 + cos(3πx))
1 − x (20%)
0.5(1 + cos(3πx)) (20%)
0 2 4 6 8 10 Tokens (Billion)
3.16 3.17 3.18 3.19 3.20 0 2 4 6 8 10 Tokens (Billion)
3.17 3.18 3.19 3.20 3.17 3.18 3.19 Lo ss Loss Loss 0 5 10 15 20 Tokens (Billion)
(a) Vary Schedules
(b) Scale Model Size
(c) Scale Training Horizon
independent of the learning rate schedule. The second term E(τ ) is the excess loss due to SGD noise, which is a sum of exponential moving averages (up to normalization) of the gradient variance scaled by the learning rate over each eigenmode. Substituting in the specific forms for Σ recovers the convolutional Volterra equation for linear regression analyzed in Paquette et al. (2021; 2024a), or the noisy quadratic model in Zhang et al. (2019) for small learning rates.

If ηΣ¯ varies slowly compared to the timescale of the exponential moving average, we can make the approximation η(s)Σ( ¯ s) ≈ η(τ )Σ( ¯ τ ) inside the integrand, giving us:

$$\mathcal{E}(\tau)\approx\frac{1}{2}\eta(\tau)\operatorname{Tr}\biggl{(}\bar{\Sigma}(\tau)H\int_{0}^{\tau}ds\,e^{-2H(\tau-s)}\biggr{)}$$ $$=\frac{1}{4}\eta(\tau)\operatorname{Tr}\bigl{(}\bar{\Sigma}(\tau)\bigl{(}1-e^{-2H\tau}\bigr{)}\bigr{)}.$$
(12)  $\binom{13}{2}$  . 
−2Hτ . (13)
For large τ the expected loss is then approximately

$$\bar{L}(\tau)\approx\mathcal{F}(\tau)+\frac{1}{4}\eta(\tau)\,\mathrm{Tr}\big(\bar{\Sigma}(\tau)\big).$$
η(τ ) TrΣ( ¯ τ ). (14)
Given access to TrΣ( ¯ τ ), we can derive a prediction for how the loss changes as we change the learning rate schedule without knowing F.

General Case. In Appendix G, we discuss how this analysis can be generalized to more realistic setups. For general loss functions, we show via perturbation theory that, to first order in ηΣ¯, one can make similar approximations to derive Equation (14) given an additional assumption that the Hessian is slowly varying, and with the forcing function F(τ ) no longer admitting a quadratic form. We also show in Appendix G that Σ should be the *preconditioned* gradient covariance when using adaptive optimizers. We absorb the layerwise, width-dependent learning rates in µP into the preconditioner, similar to Noci et al. (2024), so η(t) ∈ [0, 1]
reflects only the schedule.

3.2.2. PREDICTING LOSS CURVES ACROSS SCHEDULES
We apply this simple model to predict empirical loss curves in the CIFAR-5M experiments. We measure the trace of the preconditioned gradient covariance on a fixed set of 2M tokens (see Appendix A for experiment details).

Let *L, η,* ¯ Σ¯ be a given reference trajectory and L¯′ = L¯ + δL, η ¯ ′ = η + δη, Σ¯′ = Σ +¯ δΣ¯ be the target trajectory, Equation (14) allows us to predict the target loss via

$$\delta\bar{L}(\tau)\approx\frac{1}{4}\,\mathrm{Tr}\big[\delta\big(\eta(\tau)\bar{\Sigma}(\tau)\big)\big],\tag{15}$$

where δη(τ )Σ( ¯ τ ):= η
′(τ )Σ¯′(τ ) − η(τ )Σ( ¯ τ ). We use a constant learning rate for the reference trajectories and various schedules sharing the same peak learning rate for the target. Decomposing δ(ηΣ) = δηΣ
′+ηδΣ, we find the first term is typically 3 to 10 times larger than the second as the learning rate decays, which can be attributed to how learning rate interacts with curvature (Figure 13). In Figure 6, we only keep the first term, and predict the target loss as

$$L^{\prime}(\tau)\approx L(\tau)+\alpha\,\delta\eta(\tau)\,{\rm Tr}(\Sigma^{\prime}(\tau)),\tag{16}$$
$$(14)$$

where α is a shared hyperparameter. We find a single α = 0.21 fits the target loss curves surprisingly well across schedules, model sizes, and training horizons. In Appendix I, we show even better fits for MLPs, though puzzlingly, including the second term can lead to worse fits. Recent works proposed more complex functional forms for how learning rate schedules affect loss curves, derived primarily from empirical observations (Tissue et al., 2024; Luo et al., 2025). The accuracy of our simple model suggests it captures the essential dynamics, and crucially, the correct scaling of the excess loss through Tr(Σ′) so that a single α is predictive across model sizes, schedules, and training horizons. Notably, Luo et al. (2025) experimented with a similar form to Equation (16) but with a constant Σ
′, which likely explains the reduced effectiveness they observed.

7

1 − x 0.5(1 + cos(3πx))
1 − x (20%)
0 25 50 Tokens (B)
0.5 1.0 1.5 Tr( Σ

)

/

L

12M
20M31M
79M
0 25 50 Tokens (B)
0 25 50 Tokens (B)
0.0 0.5 1.0 Normalized Compute 0.5 1.0 1.5 Tr( Σ

)

/

L

0.0 0.5 1.0 Normalized Compute 0.0 0.5 1.0 Normalized Compute

## 3.2.3. Universal Scaling Of Gradient Noise

For typical loss functions, the gradient covariance can be related to the loss itself. For example, in noiseless highdimensional linear regression with Gaussian features drawn from N (0, K), we have Tr(Σ) ≈ 2L Tr(K)(Paquette et al.,
2021), an intuitive result since the gradient scales with both the prediction error and the input. For non-linear regression, K should be taken to be the time-varying Gauss-Newton matrix for a first approximation. In this case, Tr(K) is known to depend strongly with the learning rate (Agarwala & Pennington, 2024), but we expect weak dependence on model size given our models are trained with µP (see Noci et al. (2024) for evidence that curvature statistics depend weakly on model size in µP). Since the schedule is a function of the normalized compute x = t/t⋆alone, we hypothesize there exists a schedule-dependent function h(x) such that

$$\mathrm{Tr}(\Sigma(x t^{\star}(p)))/{\mathcal{L}}(x t^{\star}(p))\approx h(x),$$

which we verify in the regression (Figure 14) and next-token prediction experiments (Figure 7). Combining Equation (16) and Equation (17) and making p-dependence explicit:

$$\bar{\cal L}^{\prime}(\tau,p)\approx\bar{\cal L}(\tau,p)(1-\alpha h(x)\delta\eta(\tau,p))^{-1},\quad\mathrm{(18)}$$

where x is the normalized compute at gradient flow time τ. We leave to future work an explanation of why this relation appears to hold for cross-entropy loss despite the presence of non-negligible irreducible loss, as this setting is analogous to regression with label noise, where the gradient covariance should scale with the total loss rather than just the reducible component, i.e., Tr(Σ) ≈ 2L Tr(K).

Scaling Collapse Across Schedules. Combining our insights so far, we can now understand why collapse happens across schedules. Let ¯ℓ(*x, p*) and ¯ℓ
′(*x, p*) be the expected normalized loss curves under two schedules S and S
′. Let y(x) map the normalized compute under S
′to the normalized compute under S at matching gradient flow time, where y is independent of p for schedules defined in terms of the normalized compute. Let δηˆ(x) = δη(xt⋆(p), p) be the difference between the two schedules. Assuming small relative fluctuations (E[L(x)/L(y)] ≈ E[L(x)]/E[L(y)]), we have:

$$\bar{\ell}^{\prime}(x,p)\approx\frac{\bar{\ell}^{\prime}(xt^{\star}(p),p)}{\bar{\ell}^{\prime}(t^{\star}(p),p)}\tag{19}$$ $$=\frac{\bar{\ell}(y(x)t^{\star}(p),p)(1-\alpha h(x)\delta\tilde{\eta}(x))^{-1}}{\bar{\ell}(y(1)t^{\star}(p),p)(1-\alpha h(1)\delta\tilde{\eta}(1))^{-1}}$$ (20) $$=\bar{\ell}(y(x),p)\underbrace{\frac{1-\alpha h(1)\delta\tilde{\eta}(1)}{1-\alpha h(x)\delta\tilde{\eta}(x)}}_{\text{independent of$p$}},\tag{21}$$

which shows that, in expectation, collapse under one schedule (e.g. constant) implies collapse under any other schedule, provided we take Equation (18) to be exact. Since collapse under a constant learning rate can be attributed to the sumof-power-laws scaling law, this result helps explain why we also observe collapse in other schedules. This analysis also suggests that collapse can serve as a filter for identifying interventions that yield scalable improvements: those that multiplicatively shift the reducible loss curve by the same factor across all model sizes.

## 3.3. Supercollapse As Variance Reduction

Lastly, we turn to understanding the "super" in supercollapse: why does learning rate decay significantly improve the collapse, to the extent that the collapse deviation ∆(x)
drops below the per-model noise floor σ(*x, p*) for a substantial fraction of training? Again, the simple quadratic model provides quantitative insights into this phenomenon.

$$(17)^{\frac{1}{2}}$$
Recall $\Delta(x)=\frac{\mathbb{V}_{p,\omega}[\ell(x,p,\omega)]^{1/2}}{\mathbb{E}_{p,\omega}[\ell(x,p,\omega)]}$, and the decomposition:  $$\mathbb{V}_{p,\omega}[\ell(x,p)]=\mathbb{E}_{p}\mathbb{V}_{\omega}[\ell(x,p)]+\mathbb{V}_{p}\mathbb{E}_{\omega}[\ell(x,p)].\tag{22}$$

The first term measures variance due to the seed alone, averaged over model sizes, while the second term measures variance due to varying the model size, having averaged over the seeds first. Since we observed that variations in the normalized curves primarily arise from seed-to-seed fluctuations rather than model-to-model differences (Section 2.4) under a constant schedule, and switching to other schedules does not significantly increase the model-to-model differences
(Section 3.2), we will assume the first term EpVω[ℓ(x, p)]
dominates, which implies ∆2(x) ≈ Ep∆˜ 2(*x, p*), where
∆˜ 2(x, p) := Vω[ℓ(*x, p*)]/¯ℓ 2(*x, p*) is the squared *per-model* collapse deviation.

To simplify notation, we temporarily omit p-dependence and write ℓ in terms of t instead of x. Letting L(t) =

0.0 0.2 0.4 0.6 0.8 1.0 Normalized Compute 10−4 10−3 10−2 10−1 1 − x 1 − x 2 0.5(1 + cos(πx))
0.5(1 + cos(3πx))
1 − x (20%)
0.0 0.2 0.4 0.6 0.8 1.0 τ ˆ
10−4 10−3 10−2 10−1 10−2 10−1 100 1 − τ ˆ
10−4 10−3 10−2 10−1 C
oll ap se D
ev
.

/

√

η C
oll ap se D
ev
.

/

√

η C
oll ap se D
ev ia ti on
√
1 − τ ˆ
√
1 − τ ˆ
L¯(t)(1 + ψ(t)), where ψ is the relative fluctuation, we have ℓ(t) = L¯(t)(1+ψ(t))
L¯(t
⋆)(1+ψ(t
⋆)) ≈ 
¯ℓ(t)(1 + ψ(t) − ψ(t
⋆)), assuming ψ ≪ 1. Therefore,

$$\bar{\Lambda}^{2}(t)\approx\mathbb{E}\big[(\psi(t)-\psi(t^{\star}))^{2}\big]\tag{23}$$

We see that what controls the relative variance in ℓ(t) is not ψ(t) but the difference ψ(t) − ψ(t
⋆), which roughly captures the amount of optimization noise accumulated *between* time t and time t
⋆. Since the optimization noise per step scales with the instantaneous learning rate, decaying the learning rate over time will precisely serve to decrease the variance in ℓ. By contrast, the squared per-model noise floor σ 2(t) is simply E[ψ 2(t)], which captures the total cumulative optimization noise. Importantly, had we normalized by the expected rather than the empirical final loss in ℓ,
∆( ˜ t) would reduce to σ(t). Normalizing by the stochastic final loss is essential for supercollapse, where it acts as a control-variate (Glasserman, 2004), leveraging the strong time-correlation of stochastic fluctuations along the optimization trajectory to cancel much of the shared noise and thereby sharply reduce the variance of the collapsed curve.

Quantitatively, we can estimate ∆˜ under the quadratic model in Section 3.2. Let ∆w(τ ) and ∆L(τ ) be the fluctuations of the parameters and loss from their means. We have
∆w(τ ) = R τ 0 ds e−H(τ−s)Σ
1/2(s)ξ(s), and ∆L(τ ) =
g(τ )
⊤∆w(τ ) to first order in ∆w(τ ), where g(τ ) is the expected gradient. Close to the end of training, for τ = τ
⋆−δτ where τ
⋆is the final gradient flow time and *δτ >* 0 is small, direct calculation shows (Appendix H)

$$\bar{\Delta}^{2}(\tau)=\bar{\mathcal{L}}^{-2}(\tau)g(\tau)^{\top}\eta(\tau)\bar{\Sigma}(\tau)g(\tau)\delta\tau+O(\delta\tau^{2}),\tag{24}$$

In linear regression, Σ ∝ L and g
⊤g ∝ L , so we estimate
∆˜ 2(τ ) ∝ η(τ )δτ to leading order. Since this relation holds for each model size p, we predict ∆2(ˆτ ) ≈ Ep∆˜ 2(ˆτ ) ∝
η(ˆτ )(1−τˆ), where τˆ = *τ /τ* 
⋆ denotes the normalized gradient flow time. Figure 8 shows this form fits the observations well, with ∆(ˆτ )/p
√
η(ˆτ ) approximately following the same 1 − τˆ scaling across many schedules, quantitatively explaining how learning rate decay leads to supercollapse.

## 4. Discussion

Scale has enabled remarkable progress in machine learning, but a thorough scientific understanding of scaling remains elusive. Key open questions include identifying robust principles that guide general hyperparameter transfer and characterizing scaling limits under realistic scaling ladders. Our discovery of supercollapse provides empirical evidence that a model-size and data joint scaling limit *generically* exists in the compute-optimal regime, and that the scale-invariance of the training dynamics revealed by the collapse can diagnose proper hyperparameter configuration. We believe further investigation of these phenomena holds great potential for advancing the science of scaling. We see many exciting extensions to this work. Empirically, our small-scale experiments provide a proof-of-concept. While small-scale proxies capture certain behaviors in larger systems (Wortsman et al., 2023), validating at larger scales and with practical scaling ladders, where width, depth, batch size, and weight decay are co-scaled (McCandlish et al.,
2018; Wang & Aitchison, 2024; Dey et al., 2025; Bergsma et al., 2025), is important and may yield new insights into optimal scaling and hyperparameter transfer. Scaling collapse beyond the form we studied here can be a general tool to study other scaling relations (Tamai et al., 2023). While we have identified the key ingredients underlying supercollapse—power-law scaling and learning ratedependent noise dynamics—our analysis relies on multiple approximations and takes power-law scaling as given, suggesting deeper theoretical principles may be at work. Taking collapse as a starting point instead may provide an alternative route to understanding scaling laws, analogous to how in physics the renormalization group was developed to provide a unified set of principles explaining both universality and its associated power laws (Wilson, 1971). Finally, it would be interesting to understand why our simple noise model predicts the impact of learning rate schedules on real models so effectively, compare it with alternative models such as Schaipp et al. (2025), and to test its predictive power for optimizing schedules, learning rates, and training horizons.

## Acknowledgements

We thank Courtney Paquette and Zixi Chen for helpful comments on an earlier version of this paper. SQ was supported by Google's TPU Research Cloud (TRC) program:
https://sites.research.google/trc/.

## Contribution Statement

SQ designed and conducted the majority of experiments, led the theory development, and wrote the paper. LX initially observed supercollapse, contributed to theory, experimental design, and writing the paper. AGW advised SQ and edited the paper. JP contributed to theory, experimental design, and writing the paper. AA managed the research project, proved some theorems, guided the theory development, contributed to experimental design, and helped write the paper.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I.,
Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S.,
Anadkat, S., et al. Gpt-4 technical report. *arXiv preprint* arXiv:2303.08774, 2023.

Agarwala, A. and Fisher, D. S. Adaptive walks on high-dimensional fitness landscapes and seascapes with distance-dependent statistics. Theoretical population biology, 130:13–49, 2019.

Agarwala, A. and Pennington, J. High dimensional analysis reveals conservative sharpening and a stochastic edge of stability, 2024. URL https://arxiv.org/abs/ 2404.19261.

Bergsma, S., Dey, N., Gosal, G., Gray, G., Soboleva, D.,
and Hestness, J. Power lines: Scaling laws for weight decay and batch size in llm pre-training. *arXiv preprint* arXiv:2505.13738, 2025.

Binder, K. Finite size scaling analysis of ising model block distribution functions. *Zeitschrift fur Physik B Condensed* ¨ Matter, 43:119–140, 1981.

Bordelon, B. and Pehlevan, C. Self-consistent dynamical field theory of kernel evolution in wide neural networks.

Advances in Neural Information Processing Systems, 35:
32240–32256, 2022.

Bordelon, B., Noci, L., Li, M. B., Hanin, B., and Pehlevan, C. Depthwise hyperparameter transfer in residual networks: Dynamics and scaling limit. *arXiv preprint* arXiv:2309.16620, 2023.

Bordelon, B., Atanasov, A., and Pehlevan, C. A dynamical model of neural scaling laws. arXiv preprint arXiv:2402.01092, 2024a.

Bordelon, B., Atanasov, A., and Pehlevan, C. How feature learning can improve neural scaling laws. *arXiv preprint* arXiv:2409.17858, 2024b.

Bordelon, B., Chaudhry, H., and Pehlevan, C. Infinite limits of multi-head transformer dynamics. Advances in Neural Information Processing Systems, 37:35824–35878, 2024c.

Cohen, J. M., Kaur, S., Li, Y., Kolter, J. Z., and Talwalkar, A. Gradient descent on neural networks typically occurs at the edge of stability. *arXiv preprint arXiv:2103.00065*, 2021.

Cohen, J. M., Ghorbani, B., Krishnan, S., Agarwal, N.,
Medapati, S., Badura, M., Suo, D., Cardoze, D., Nado, Z., Dahl, G. E., et al. Adaptive gradient methods at the edge of stability. *arXiv preprint arXiv:2207.14484*, 2022.

Cohen, J. M., Damian, A., Talwalkar, A., Kolter, Z., and Lee, J. D. Understanding optimization in deep learning with central flows. *arXiv preprint arXiv:2410.24206*, 2024.

d'Ascoli, S., Refinetti, M., and Biroli, G. Optimal learning rate schedules in high-dimensional non-convex optimization problems. *arXiv preprint arXiv:2202.04509*, 2022.

Dey, N., Zhang, B. C., Noci, L., Li, M., Bordelon, B.,
Bergsma, S., Pehlevan, C., Hanin, B., and Hestness, J. Don't be lazy: Completep enables compute-efficient deep transformers. *arXiv preprint arXiv:2505.01618*, 2025.

Diederik P. Kingma, J. B. Adam: A Method for Stochastic Optimization. International Conference on Learning Representations (ICLR), 2015.

Everett, K., Xiao, L., Wortsman, M., Alemi, A. A., Novak, R., Liu, P. J., Gur, I., Sohl-Dickstein, J., Kaelbling, L. P., Lee, J., et al. Scaling exponents across parameterizations and optimizers. *arXiv preprint arXiv:2407.05872*, 2024.

Fisher, D. S. Asexual evolution waves: fluctuations and universality. Journal of Statistical Mechanics: Theory and Experiment, 2013(01):P01011, 2013.

Glasserman, P. Monte Carlo methods in financial engineering, volume 53. Springer, 2004.

Hallatschek, O. and Fisher, D. S. Acceleration of evolutionary spread by long-range dispersal. *Proceedings of the* National Academy of Sciences, 111(46):E4911–E4919, 2014.

Hendrycks, D. and Gimpel, K. Gaussian Error Linear Units
(GELUs). *Preprint arXiv 1606.08415*, 2016.

Hestness, J., Narang, S., Ardalani, N., Diamos, G., Jun, H.,
Kianinejad, H., Patwary, M. M. A., Yang, Y., and Zhou, Y.

Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E.,
Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A.,
Welbl, J., Clark, A., et al. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556*, 2022.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B.,
Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Levy, S. F., Blundell, J. R., Venkataram, S., Petrov, D. A.,
Fisher, D. S., and Sherlock, G. Quantitative evolutionary dynamics using high-resolution lineage tracking. *Nature*, 519(7542):181–186, 2015.

Li, Q., Tai, C., et al. Stochastic modified equations and adaptive stochastic gradient algorithms. In International Conference on Machine Learning, pp. 2101–2110. PMLR, 2017.

Luo, K., Wen, H., Hu, S., Sun, Z., Liu, Z., Sun, M., Lyu, K., and Chen, W. A multi-power law for loss curve prediction across learning rate schedules. arXiv preprint arXiv:2503.12811, 2025.

Malladi, S., Lyu, K., Panigrahi, A., and Arora, S. On the sdes and scaling rules for adaptive gradient algorithms.

Advances in Neural Information Processing Systems, 35: 7697–7711, 2022.

Marchenko, V. and Pastur, L. A. Distribution of eigenvalues for some sets of random matrices. *Mat. Sb.(NS)*, 72(114): 4, 1967.

McCandlish, S., Kaplan, J., Amodei, D., and Team, O. D.

An empirical model of large-batch training. arXiv preprint arXiv:1812.06162, 2018.

McLeish, S., Kirchenbauer, J., Miller, D. Y., Singh, S.,
Bhatele, A., Goldblum, M., Panda, A., and Goldstein, T. Gemstones: A model suite for multi-faceted scaling laws. arXiv preprint arXiv:2502.06857, 2025.

Nakkiran, P., Neyshabur, B., and Sedghi, H. The deep bootstrap framework: Good online learners are good offline generalizers. *arXiv preprint arXiv:2010.08127*, 2020.

Noci, L., Meterez, A., Hofmann, T., and Orvieto, A. Super consistency of neural network landscapes and learning rate transfer. *Advances in Neural Information Processing* Systems, 37:102696–102743, 2024.

Paquette, C., Lee, K., Pedregosa, F., and Paquette, E. Sgd in the large: Average-case analysis, asymptotics, and stepsize criticality. In *Conference on Learning Theory*,
pp. 3548–3626. PMLR, 2021.

Paquette, C., Paquette, E., Adlam, B., and Pennington, J. Homogenization of sgd in high-dimensions: Exact dynamics and generalization properties. Mathematical Programming, pp. 1–90, 2024a.

Paquette, E., Paquette, C., Xiao, L., and Pennington, J. 4+
3 phases of compute-optimal neural scaling laws. arXiv preprint arXiv:2405.15074, 2024b.

Pearce, M. T., Agarwala, A., and Fisher, D. S. Stabilization of extensive fine-scale diversity by ecologically driven spatiotemporal chaos. *Proceedings of the National* Academy of Sciences, 117(25):14572–14583, 2020.

Pearce, T. and Song, J. Reconciling kaplan and chinchilla scaling laws. *arXiv preprint arXiv:2406.12907*, 2024.

Rosen, M. J., Davison, M., Bhaya, D., and Fisher, D. S.

Fine-scale diversity and extensive recombination in a quasisexual bacterial population occupying a broad niche.

Science, 348(6238):1019–1023, 2015.

Schaipp, F., Hagele, A., Taylor, A., Simsekli, U., and Bach, ¨
F. The surprising agreement between convex optimization theory and learning-rate scheduling for large model training. *arXiv preprint arXiv:2501.18965*, 2025.

Ser-Giacomi, E., Zinger, L., Malviya, S., De Vargas, C.,
Karsenti, E., Bowler, C., and De Monte, S. Ubiquitous abundance distribution of non-dominant plankton across the global ocean. *Nature ecology & evolution*, 2(8):1243– 1249, 2018.

Sharma, U. and Kaplan, J. Scaling laws from the data manifold dimension. *Journal of Machine Learning Research*, 23(9):1–34, 2022.

Tamai, K., Okubo, T., Duy, T. V. T., Natori, N., and Todo, S. Universal scaling laws of absorbing phase transitions in artificial deep neural networks. arXiv preprint arXiv:2307.02284, 2023.

Tissue, H., Wang, V., and Wang, L. Scaling law with learning rate annealing. *arXiv preprint arXiv:2408.11029*,
2024.

Venkataram, S., Dunn, B., Li, Y., Agarwala, A., Chang, J.,
Ebel, E. R., Geiler-Samerotte, K., Herissant, L., Blundell, ´ J. R., Levy, S. F., et al. Development of a comprehensive genotype-to-fitness map of adaptation-driving mutations in yeast. *Cell*, 166(6):1585–1596, 2016.

Vyas, N., Atanasov, A., Bordelon, B., Morwani, D.,
Sainathan, S., and Pehlevan, C. Feature-learning networks are consistent across widths at realistic scales. Advances in Neural Information Processing Systems, 36: 1036–1060, 2023.

Wang, X. and Aitchison, L. How to set adamw's weight decay as you scale model and dataset size. arXiv preprint arXiv:2405.13698, 2024.

Wen, K., Li, Z., Wang, J., Hall, D., Liang, P., and Ma, T.

Understanding warmup-stable-decay learning rates: A
river valley loss landscape perspective. *arXiv preprint* arXiv:2410.05192, 2024.

Wilson, K. G. Renormalization group and critical phenomena. i. renormalization group and the kadanoff scaling picture. *Physical review B*, 4(9):3174, 1971.

Wortsman, M., Liu, P. J., Xiao, L., Everett, K., Alemi, A.,
Adlam, B., Co-Reyes, J. D., Gur, I., Kumar, A., Novak, R., et al. Small-scale proxies for large-scale transformer training instabilities. *arXiv preprint arXiv:2309.14322*, 2023.

Xiao, L. Rethinking conventional wisdom in machine learning: From generalization to scaling. arXiv preprint arXiv:2409.15156, 2024.

Yang, G. and Hu, E. J. Feature Learning in Infinite-Width Neural Networks. International Conference on Machine Learning (ICML), 2021.

Yang, G. and Littwin, E. Tensor Programs IVb: Adaptive Optimization in the Infinite-Width Limit. *International* Conference on Learning Representations (ICLR), 2023.

Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., Ryder, N., Pachocki, J., Chen, W., and Gao, J. Tensor Programs V: Tuning Large Neural Networks via Zero- Shot Hyperparameter Transfer. Advances in Neural Information Processing Systems (NeurIPS), 2021.

Zhang, B. and Sennrich, R. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.

Zhang, G., Li, L., Nado, Z., Martens, J., Sachdeva, S., Dahl, G., Shallue, C., and Grosse, R. B. Which algorithmic choices matter at which batch sizes? insights from a noisy quadratic model. Advances in neural information processing systems, 32, 2019.

## A. Experiment Details

Transformer Architecture. We use GeLU activations (Hendrycks & Gimpel, 2016), RMSNorm (Zhang & Sennrich, 2019), and learned positional embeddings. We untie the embedding matrix from the output head and do not use bias anywhere. The readout layer is always zero-initialized, as suggested by Yang et al. (2021) . We denote the embedding dimension with D. We set the intermediate dimension in the feedforward layers to D instead of the usual 4D, which enables us to explore larger widths more efficiently. The head dimension is set to 64.

CIFAR-5M. We use the CIFAR-5M dataset (Nakkiran et al., 2020) of 6 million CIFAR-like images. We convert the 32 × 32 × 3 images to greyscale and flatten them into sequences of length 1024. The model autoregressively predicts the pixel intensities in raster-scan order. The vocabulary is the set of pixel intensities {0*, . . . ,* 255}. Following µP we parameterize the learning rate for each weight matrix as η = η*base*/D where d is the model dimension, except for the embedding matrix which has η = ηbase. We use a parameter multiplier a on the embedding matrix. We use ηbase = 4 and a = 0.1 as they led to good performance in our early experiments. We initialize the embedding matrix as Wemb ij ∼ N (0, 1),
the output head as Whead = 0, all other non-readout matrices W as Wij ∼ N (0, 1/D). These hyperparameters were determined with a small amount of tuning in early experiments. We use a batch size of 256 images. We use a linear warmup for 1000 steps.

For the experiments in Section 3.2.2, we used a slightly different setup due to switching to a new codebase in the middle of the research project. We use µP where the base embedding dimension is 128 and base learning rate is 0.01. We initialize the embedding matrix with a standard deviation (std) of 0.1 and multiply its learning rate by 10 relative to the base learning rate. The output projection of the feedforward and attention layers are zero-initialized. All other non-readout matrices are initialized with std 1/
√D. We use a batch size of 65536 tokens or 64 images. We use a linear warmup for 10M tokens.

Chess. We run our experiments on the Lichess dataset available on Hugging Face at https://huggingface.co/ datasets/Lichess/standard-chess-games. We used character-level tokenization and a context length of 128. We use µP where the base embedding dimension is 128 and base learning rate is 0.01. We initialize the embedding matrix with std 0.1 and multiply its learning rate by 10 relative to the base learning rate. The output projection of the feedforward and attention layers are zero-initialized. All other matrices are initialized with std 1/
√D. We use a batch size of 65536 tokens. We use a linear warmup for 10M tokens. MLP Experiments. Our MLP architecture is identical to the transformer with attention layers removed and the token and position embedding layers replaced by a linear layer. We use µP where the base width is 128 and base learning rate is 0.001. The output projection of the feedforward layers are zero-initialized. All non-readout matrices are initialized with std 1/
√D.

We use a batch size of 4096 examples. We do not use warmup.

The target function is defiend as ϕ(x) = PM
i=1 wi
√2 cos2πk⊤
i x + bi, with x ∈ R
8, wi ∼ N (0, 1), bi ∼
π 2 Bernoulli(0.5), ki = round(sivi) where siis a scalar sampled from a power law with support [1, ∞) and exponent
−2, viis a random unit vector, and round rounds to the nearest point in Z
8. During training, x is sampled uniformly from [−0.5, 0.5]8, making the Fourier features orthonormal over the data distribution. We suspect the details here are not necessary for generating power-law scaling laws beyond the power-law spectrum.

For Figure 4a, the constant learning rate scaling ladder matches the µP learning rate at D = 384 (smallest model). Measuring Gradient Covariance Trace. As mentioned in Section 3.2.1, we use the preconditioned gradient covariance Σ˜ instead of the raw gradient covariance due to the use of Adam (and µP). Σ˜ is defined as P
−1/2ΣP
−1/2 where Σ is the raw (mini-batch) gradient covariance, and P is the preconditioner. See an explanation for this definition in Appendix G. Since µP uses layerwise learning rate, the preconditioner is defined as P
−1 = diag√
η0 v 2+ϵ
, where η0 is the vector of per-parameter learning rate (peak learning rate, before applying the schedule), v 2is the Adam second-moment, and ϵ is the Adam ϵ. As the peak learning rate is absorbed into the preconditioner, any occurrence of the instantaneous learning rate η(t) or η(τ ) in Section 3.2.2 reflects only the schedule and takes on values in [0, 1]. This definition mirrors what is done in Noci et al. (2024); Cohen et al. (2024).

Training for More than One Epoch. The CIFAR-5M dataset, tokenized as 32 × 32 greyscale images, has about 5B tokens.

Therefore, most models needed to be trained for more than 1 epoch to reach the compute-optimal training horizon. Up to the scales we tested, we did not observe a significant difference between the train and test loss. The chess dataset has about 20B tokens, which also led to data reuse for some models, but did not lead to significant overfitting. As we only processed a subset of the full Lichess dataset, this can be avoided by processing a larger subset if desired. On Random Seeds. In the CIFAR-5M experiments in Figure 1, the random seed controls both the initialization and data ordering. In the other experiments (Transformer on chess and MLP regression), the random seed only controls the initialization while data ordering is held fixed, as is often done in practice. Fixing the data ordering (no shuffling) had the advantage of speeding up data loading. We found that supercollapse occurs regardless of whether seeds affect data ordering. This makes sense: even with fixed ordering, different model sizes process different data due to varying training horizons. More fundamentally, supercollapse should be robust to which training components are randomized, as the variance reduction arises from strong noise correlations along individual trajectories rather than specific noise sources (Section 3.3).

## B. Scaling Collapse Across Transformer Depths

For scaling depth, we additionally apply a branch multiplier of 3/depth on the output of every feedforward and attention layer, as suggested by Bordelon et al. (2024c). We find a decent degree of collapse in Figure 9 when training on chess data. There is a small shift in the normalized curves, though we are unsure if it is simply a finite-size effect.

1.6 L0 + ac−b a = 0.355 b = 0.164 L0 = 0.545 N
or m al ize d Lo ss Depth 9 12 16 20 24 28 36 102 103 104 Compute (PetaFLOPs)
0.63 0.66 0.69 0.72 1.4 Los s 1.2 0.0 0.2 0.4 0.6 0.8 1.0 Normalized Compute 1.0
(a) Scaling Law
(b) Collapse

## C. Estimating Compute-Optimal Training Horizon

To estimate the optimal compute for training each model, we perform the following steps in each experiment:
- We trained each model *without* learning rate decay but keeping the initial warmup. We chose a large enough number of steps so that the largest model could reach the compute-optimal loss frontier. We average the loss curves from 5 seeds.

- We numerically computed the compute-loss Pareto frontier to obtain an estimate of c
⋆(p) - the optimal compute for each model size p. We use logarithmically spaced points for c and find the p that achieves the best loss given c training FLOPs.

- We fit a power law c
⋆(p) = κp1+γ where κ and γ are fit parameters. The optimal number of training tokens is then t
⋆(p) = c
⋆(p)/(6p), which scales as p γ. We remove outliers in this fit by dropping points from the smallest and largest model.

Using a constant learning rate schedule allows us to measure the loss of one model at different token budgets with a single training run, an approach also used in McLeish et al. (2025), rather than one training run per token budget as origianlly done in Hoffmann et al. (2022). Figure 10 illustrates this procedure.

107 Parameters 101 102 103 D = 384 D = 512 D = 645 D = 812 D = 1024 D = 1290 D = 1625 D = 2048 Pareto frontier D = 384 D = 512 D = 640 D = 768 D = 896 D = 1024 D = 1152 D = 1280 D = 1536 D = 1792 D = 2048 Pareto frontier 101 102 103 104 105 Compute (PetaFlops)
3.18 3.21 3.24 3.27 Com put e (P
eta Flo ps)c
∗
= κp 1+γ κ = 533520.12 γ = 0.96 Co mpu te
 (Pe taF
lop s)c
∗
= κp 1+γ κ = 1346251.61 γ = 1.04 107 108 Parameters 102 103 104 10−1 Loss Loss 10−3 10−1 101 103 Compute (PetaFlops)
(a) MLP Pareto Frontier
(b) MLP Data Exponent
(c) CIFAR-5M Pareto Frontier
(d) CIFAR-5M Data Exponent

## D. Universality And Scaling Collapse In Other Sciences

The simplest versions of collapse come from statistics and probability, where entire *distributions* of random variables show universal behavior between systems of different types and scales. The most well known is the central limit theorem which predicts a universal Gaussian form for the sums of random variables with appropriately bounded moments (and Levy distributions for heavy tailed distributions). In random matrix theory, there are similar effects when studying the limiting empirical distribution of spectra, most famously the Marcenko-Pastur distribution of Wishart matrices (Marchenko & Pastur, 1967). In all of these examples, showing universal behavior of a single moment is analogous to showing predictability of the Pareto frontier in our work, while showing the universality of the whole distribution is analogous to our statements about the entire loss curves (where e.g. the CDFs of different problems are converging). Scaling collapse is ubiquitous in physics as well. There are again distributional collapses like the famed Maxwell Boltzmann distribution first used to describe idealized gasses, but also functional relationships like the universal magnetizationtemperature curves of near-critical Ising lattices of different sizes (Binder, 1981). In the Ising example, changes to the lattice topology can change the magnetization-temperature curves, similar to how different datasets, architecture, and training algorithms lead to different universal curves in our study. A more general theory unifying and explaining the existence of universality in physics arises from the renormalization group (Wilson, 1971). More recently, scaling collapse has been used to describe dynamical systems in biological contexts. Advances in genomics have led to the rise of experimental microbial evolution with rapid timecourse data (Levy et al., 2015; Venkataram et al., 2016). Analysis of this data relies on quantitative modeling of evolutionary dynamics. Most of these models show universal scaling dynamics, in situations from rapid evolutions of diverse populations (Fisher, 2013), populations evolving under changing fitness conditions (Agarwala & Fisher, 2019), and populations expanding in space (Hallatschek & Fisher, 2014). In these settings the timecourse of key observables can be described with dynamical curves which can be rescaled to universal forms with transformations depending on population size, mutation frequency, and statistics of the fitness landscape. In ecology, the ubiquity of fine-scale diversity and the seemingly universal, power-law nature of species rank-abundance curves (Rosen et al., 2015; Ser-Giacomi et al., 2018) can be explained using dynamical models which themselves show universal scaling behavior over ecosystems of different sizes (Pearce et al., 2020). Scaling collapse has been used to study certain scaling relations in machine learning. Kaplan et al. (2020) identified universal scaling of overfitting, showing a collapse of the rescaled excess loss vs rescaled parameter count across dataset sizes. Tamai et al. (2023) used scaling collapse to establish universal scaling laws in the forward signal propagation dynamics of MLPs near the order-to-chaos transition.

## E. Power-Law Pareto Frontier Is Necessary For Collapse

Recall t
⋆(p) is the optimal training horizon for model size p, i.e. L(t
⋆(p), p) = mint
′,p′:t
′p′=t⋆(p)p L(t
′, p′). Let c
⋆(p) =
6t
⋆(p)p be the optimal compute for p. In what follows, rather than writing L(t, p), we will find it convenient to express the loss curves in terms of compute and model size. Letting L(c, p) be the loss curves expressed this way, we have the following theorem:
Theorem E.1. Let L(*c, p*) be C
1for all positive c, p *and let* c
⋆(p) *denote the compute-optimal budget for model size* p.

Write L(*c, p*) = L(c, p) − Lˆ for some offset Lˆ (e.g., Lˆ = L0 *the irreducible loss). Define the normalized loss curve*

$$\ell(x,p)=\frac{{\cal L}(xc^{\star}(p),p)}{{\cal L}(c^{\star}(p),p)},\quad x\in[0,1].\tag{1}$$

Then, 1. Necessity. If ℓ is independent of p (collapse), then the Pareto frontier of {L(c, p)}c,p

$${\mathcal{L}}^{\star}(c):=\operatorname*{min}_{p}{\mathcal{L}}(c,p)={\mathcal{L}}(c^{\star}(p),p)$$
⋆(p), p) (26)
is a power law L
⋆(c) = ac−δfor some constants *a, δ*.

2. Sufficiency at first order. *Conversely, suppose* L
⋆(c) = ac−δ*, then*

$$(25)$$
$$(26)$$
$$\left.\frac{d\ell(x,p)}{dx}\right|_{x=1}=-\delta,\tag{1}$$
$$(27)^{\frac{1}{2}}$$

independent of p*. Hence all curves share the same first-order behavior around* x = 1*, i.e., they collapse to first order around* x = 1. Proof. First, we have the following identity for log-derivatives for a general differentiable function u(v)

$$\frac{d\log u}{d\log v}=\frac{v}{u}\frac{du}{dv}.\tag{1}$$

Applying this to our normalized curve from Equation (25) and using the chain rule:

$$\frac{d\ell(x,p)}{dx}\bigg{|}_{x=1}=\frac{c^{\star}}{\mathcal{L}(c^{\star},p)}\frac{\partial\mathcal{L}(c,p)}{\partial c}\bigg{|}_{c=c^{\star}}$$ $$=\frac{\partial\log\mathcal{L}(c,p)}{\partial\log c}\bigg{|}_{c=c^{\star}},$$
$$(28)^{\frac{1}{2}}$$
(29)  $\binom{29}{2}$  (30)  . 
$$(31)$$

where the second equality follows from Equation (28).

Necessity. If ℓ is independent of p (collapse), then dℓ(x,p)
dxx=1 is the same for all p. Set this common value to −δ. By Equation (30),

$$\left.{\frac{\partial\log{\mathcal{L}}(c,p)}{\partial\log{c}}}\right|_{c=e^{*}(p)}=-\delta\quad{\mathrm{for~every~}}p.$$

Since (c
⋆(p), p) lies on the Pareto frontier and L is C
1, the envelope theorem states

$$\left.\frac{d{\cal L}^{\star}(c)}{d c}\right|_{c=c^{\star}(p)}=\left.\frac{\partial{\cal L}(c,p)}{\partial c}\right|_{c=c^{\star}(p)},$$

i.e. the loss curve is tangent to the Pareto frontier at the compute-optimal point. Applying Equation (28) to the frontier L
⋆(c):

$\left.\dfrac{d\log\mathcal{L}^\star(c)}{d\log c}\right|_{c=c^\star(p)}=\dfrac{1}{\mu}$  $$=\frac{1}{\mu}$$ $$=\frac{\theta}{\phi}$$ $$=\frac{1}{\phi}$$
=c
(33) (34) (35)
 $\begin{array}{c|c}c^{\star}(p)&\dfrac{d\mathcal{L}^{\star}(c)}{dc}\Bigg|_{c=c^{\star}(p)}\\ \dfrac{c^{\star}(p)}{\mathcal{L}(c^{\star}(p),p)}\dfrac{\partial\mathcal{L}(c,p)}{\partial c}\Bigg|_{c=c^{\star}(p)}\\ \dfrac{\partial\log\mathcal{L}(c,p)}{\partial\log c}\Bigg|_{c=c^{\star}(p)}\\ -\delta,\end{array}$
=c
$$(32)$$
where we used Equation (32) and Equation (31). This means the log-log slope of the frontier is constant, which means it is a power law L
⋆(c) = ac−δ.

Sufficiency at first order. Assume L
⋆(c) = ac−δ. For any model size p, the envelope theorem gives the tangency condition at c = c
⋆(p):

$$\left.\frac{\partial{\mathcal{L}}(c,p)}{\partial c}\right|_{c=c^{\star}}=\left.\frac{d{\mathcal{L}}^{\star}(c)}{d c}\right|_{c=c^{\star}}=-\delta a(c^{\star})^{-\delta-1}.$$

Applying Equation (30):

$$\frac{d\ell(x,p)}{dx}\bigg{|}_{x=1}=\left.\frac{\partial\log\mathcal{L}(c,p)}{\partial\log c}\right|_{c=c^{*}(p)}$$ $$=-\delta.$$
$$(38)$$
$$(37)$$
$$(39)$$
$$\square$$

Therefore, all curves collapse to first order around x = 1 as in Equation (27). Remark. A power-law Pareto frontier is not only necessary for full collapse but also already enforces a weaker, first–order form of collapse. Theorem E.1 assumes the compute–optimal point lies in the *interior* of each loss curve. This condition can fail for learning rate schedules that reach η = 0 after finitely many steps, because the optimum may then coincide with the boundary of the curve, where the envelope theorem tangency no longer applies. Such schedules are used throughout our experiments and are common in practice. Extension of the proof to handle these boundary-optimal schedules would be interesting.

## F. Collapse For General Sum-Of-Power-Laws Loss Curves

Theorem F.1. *Suppose the loss curve is given by*

$$L(t,p)=L_{0}+\sum_{i=1}^{m}a_{i}t^{-\mu_{i}}p^{-\nu_{i}},\quad a_{i}>0,\,\mu_{i},\nu_{i}\geq0,$$
$$(40)$$
$$(41)$$

with at least one of µi, νi positive for every i (else absorb the term into L0*). Let* t
⋆(p) = κpγ with κ > 0, γ > 0 be the asymptotic compute-optimal training horizon, and define the total exponent βi:= µiγ + νi and bi:= aiκ
−µi. Without loss of generality, assume βi*'s are sorted in non-decreasing order. Then,* 1. Compute-optimality forces a tie. At least two βi*'s share the minimum:*

$$\beta_{1}=\beta_{2}=\cdots=\beta_{k}<\beta_{k+1}\leq\cdots\leq\beta_{m},\quad k\geq2.$$

2. Asymptotic collapse. *The normalized loss curve*

$$\ell(x,p):=\frac{L(xt^{\star}(p),p)-L_{0}}{L(t^{\star}(p),p)-L_{0}}.\tag{1}$$
$$(42)$$

is given by

$$\ell(x,p)=\frac{\sum_{i=1}^{k}b_{i}x^{-\mu_{i}}}{\sum_{i=1}^{k}b_{i}}+O\big{(}p^{-\epsilon}\big{)},\quad\epsilon:=\beta_{k+1}-\beta_{1}>0,$$

independent of p *up to finite-size error that decays as* O(p
−ϵ). If k = m, ϵ is taken to be ∞ *(perfect finite-size collapse).*
3. Locally fastest decay of finite-size error. Locally, γ is the data exponent that achieves the fastest decay of the finite-size error as measured by ϵ. In particular, ϵ = O(|δ|) *for any other data exponent* γ
′ = γ + δ *with* δ ̸= 0, leading to more slowly decaying finite-size error and therefore a worse collapse. 4. Compute-optimality up to a constant suffices. *Any training horizon that is a constant multiple of* t
⋆(p) preserves the collapse, only changing the constants bi*in Equation* (43).

$$(43)$$

Proof. **Compute-optimality forces a tie.** Fix the compute budget c := 6tp and note t(p) = c/(6p) so that dt dp = −t/p.

With βi:= µiγ + νi and bi:= aiκ
−µi,

$$\frac{dL}{dp}=\sum_{i=1}^{m}a_{i}\bigg{(}\frac{\partial}{\partial p}+\frac{dt}{dp}\frac{\partial}{\partial t}\bigg{)}t^{-\mu_{i}}p^{-\nu_{i}}$$ $$=\sum_{i=1}^{m}a_{i}\bigg{(}-\frac{\nu_{i}}{p}+\frac{t}{p}\frac{\mu_{i}}{t}\bigg{)}t^{-\mu_{i}}p^{-\nu_{i}}$$ $$=\frac{1}{p}\sum_{i=1}^{m}a_{i}(\mu_{i}-\nu_{i})t^{-\mu_{i}}p^{-\nu_{i}}$$ $$=\frac{1}{p}\sum_{i=1}^{m}b_{i}(\mu_{i}-\nu_{i})p^{-\beta_{i}}.$$
 (44)  $$\begin{array}{l}~~~~~~~~~~~~~~\end{array}$$ (45)  $$\begin{array}{l}~~~~~~~~~~~~~~\end{array}$$ (46)  $$\begin{array}{l}~~~~~~~~~~~~~~\end{array}$$ (47)  $$\begin{array}{l}~~~~~~~~~~~~~~\end{array}$$ (47)  ... 
If β1 < β2, the leading term b1(µ1 −ν1)p
−β1 cannot cancel the rest for asymtotically large p, contradicting dL
dp = 0 required by compute-optimality. Hence at least two indices share the minimum exponent, yielding Equation (41).1 Asymptotic collapse. We compute ℓ(*x, p*) explicitly. First, evaluate the loss at the optimal horizon:

$$L(t^{*}(p),p)-L_{0}=\sum_{i=1}^{m}a_{i}(t^{*}(p))^{-\mu_{i}}p^{-\nu_{i}}=\sum_{i=1}^{m}a_{i}(\kappa p^{\gamma})^{-\mu_{i}}p^{-\nu_{i}}$$ $$=\sum_{i=1}^{m}a_{i}\kappa^{-\mu_{i}}p^{-\mu_{i}\gamma-\nu_{i}}=\sum_{i=1}^{m}b_{i}p^{-\beta_{i}}.$$

Since β1 = β2 = · · · = βk < βk+1 *≤ · · · ≤* βm, we can factor out p
−β1:

$$L(t^{*}(p),p)-L_{0}=p^{-\beta_{1}}\bigg{(}\sum_{i=1}^{k}b_{i}+\sum_{i=k+1}^{m}b_{i}p^{-(\beta_{i}-\beta_{1})}\bigg{)}$$ $$=p^{-\beta_{1}}\bigg{(}\sum_{i=1}^{k}b_{i}\bigg{)}\bigg{(}1+O\Big{(}p^{-(\beta_{k+1}-\beta_{1})}\Big{)}\bigg{)}.$$
(48)  $$\begin{array}{l}\left(49\right)\end{array}$$ . 
(50)  $\binom{51}{51}$  . 
Similarly, for t = xt⋆(p):

$$L(xt^{\star}(p),p)-L_{0}=\sum_{i=1}^{m}a_{i}(xt^{\star}(p))^{-\mu_{i}}p^{-\nu_{i}}=\sum_{i=1}^{m}a_{i}x^{-\mu_{i}}(t^{\star}(p))^{-\mu_{i}}p^{-\nu_{i}}$$ $$=\sum_{i=1}^{m}b_{i}x^{-\mu_{i}}p^{-\beta_{i}}=p^{-\beta_{i}}\left(\sum_{i=1}^{k}b_{i}x^{-\mu_{i}}\right)\left(1+O\Big{(}p^{-(\beta_{k+1}-\beta_{i})}\Big{)}\right).$$
(52)  $\binom{53}{5}$  . 
Taking the ratio gives:

$$\ell(x,p)=\frac{p^{-\beta_{1}}\left(\sum_{i=1}^{k}b_{i}x^{-\mu_{i}}\right)\left(1+O\left(p^{-\left(\beta_{k+1}-\beta_{1}\right)}\right)\right)}{p^{-\beta_{1}}\left(\sum_{i=1}^{k}b_{i}\right)\left(1+O\left(p^{-\left(\beta_{k+1}-\beta_{1}\right)}\right)\right)}=\frac{\sum_{i=1}^{k}b_{i}x^{-\mu_{i}}}{\sum_{i=1}^{k}b_{i}}+O\left(p^{-\left(\beta_{k+1}-\beta_{1}\right)}\right).\tag{54}$$

This produces Equation (43) with ϵ = βk+1 − β1 > 0.

Locally fastest decay of finite-size error. Let γ be the optimal data exponent and perturb it by δ, writing γ
′ = γ + δ. For a small enough |δ| > 0, the previously tied lowest exponents β1*, . . . , β*k split into distinct values β
′1*, . . . , β*′k, which remain 1Here we assumed not all µi − νi ̸= 0 for i = 1*, . . . , k,* else these terms would not affect *dL/dp.* If this is not true, then the loss L(t, p) is not interesting because the it would asymptotically be a function of compute c *= 6tp* alone, independent of how we allocate c between t and p.

the lowest k exponents (since δ is small), and the gap between the lowest and second-lowest grows as O(|δ|), which is strictly smaller than the previous gap ϵ = βk+1 − β1 > 0 for sufficiently small δ. Therefore, locally γ maximizes the decay exponent of the finite-size error, i.e., it gives the best collapse locally. Compute-optimality up to a constant suffices. Replacing t
⋆ by λt⋆ multiplies each bi by λ
−µi and leaves the rest of the proof unchanged. Remarks.

1. Compute-optimal data exponent implies asymptotic collapse, but the converse is not necessarily true when m ≥ 3, since there can be multiple choices of γ that lead to balanced dominant power laws, which imply collapse, but only one of them can be compute-optimal.

2. In general, when m > 2, asymptotic instead of exact collapse is the best we can hope for. But an asymptotic collapse alone is not that interesting, since under any choice of the data exponent γ only the terms with the lowest βi enter the asymptotic normalized loss curve. For example, if L(t, p) = t
−µ + p
−ν, any *γ > ν/µ* will cause only the p
−νterm to dominate, leading to ℓ(x, p) → 1, whereas any *γ < ν/µ* will cause t
−µ to dominate, leading to ℓ(*x, p*) → x
−µ.

The latter case is similar to the infinite-width limit in neural networks, where under t = Θ(1), the loss curves become bottlenecked by training time alone and not model size (Vyas et al., 2023; Bordelon & Pehlevan, 2022; Yang & Littwin, 2023). What is interesting about the collapse that happens under compute-optimal training is that γ is tuned (to ν/µ in this example) so that more than one such term exists, so the collapse reflects a balanced scaling of both training time and model size. This fine balance is also why minor perturbations to γ from the optimal value can significantly disrupt the collapse, which is not true if there is only one dominant power law.

0.675 0.700 0.725 0.750 0.775 0.800 0.5465 + t−0.28 + p−0.21
(R
2 = 0.998)
Compute-Optimal 0 + t−0.34 + p−0.29
(R
2 = 0.999)
Compute-Optimal 10−1 100 101 102 103 Compute (PetaFLOPs)
0.02 0.04 0.06 0.08 Lo ss Lo ss 102 103 104 Compute (PetaFLOPs)
(a) Transformer on Chess

$$(55)$$

(b) MLP

## G. A Perturbative Model Of Learning Rate Schedules

Let w
′ denote the parameter trajectory under the influence of gradient noise. The dynamics of stochastic gradient descent in gradient flow time are given by

$$\frac{d w^{\prime}}{d\tau}=-\Big(\nabla L(w^{\prime})+\Sigma^{1/2}(w^{\prime})\xi(\tau)\Big),$$

with noise correlation E[ξ(τ )ξ(τ
′)
⊤] = η(τ )δ(τ − τ
′). For convenience, we rewrite this using ξ(τ ) = η 1/2(τ )˜ξ(τ ) so that

$$\frac{d w^{\prime}}{d\tau}=-\Big(\nabla L(w^{\prime})+\eta^{1/2}(\tau)\Sigma^{1/2}(w^{\prime})\bar{\xi}(\tau)\Big),$$
, (56)
where now E[˜ξ(τ )˜ξ(τ
$$\tilde{\xi}(\tau^{\prime})^{\top}]=\delta(\tau-\tau^{\prime}).$$

$$(56)^{\frac{1}{2}}$$

## 

Our strategy is to solve w
′(τ ) as w(τ ) + δw(τ ) where w(τ ) is the deterministic trajectory satisfying dw dτ = −∇L(w), up to leading order in the gradient noise scale ηΣ. Letting δw := w
′ − w, g := ∇L and taking the difference of the two differential equations:

$$\frac{d(\delta w)}{d\tau}=-\Big(g(w^{\prime})-g(w)+\eta^{1/2}(\tau)\Sigma^{1/2}(w^{\prime})\tilde{\xi}(\tau)\Big).$$

At first order,

$$(57)$$
$$g(w^{\prime})\approx g(w)+H(w)\delta w$$
$$(58)$$
$$(59)$$
′) ≈ g(w) + H(w)δw (58)
where H(w) = ∇2L(w) is the Hessian.

Our SDE for δw becomes:

$$\frac{d(\delta w)}{d\tau}=-H(w)\delta w-(\eta\Sigma)^{1/2}\bar{\xi}(\tau)$$
dτ = −H(w)δw − (ηΣ)1/2 ˜ξ(τ ) (59)
We define the propagator G(*τ, s*) that satisfies:

$$\frac{dG(\tau,s)}{d\tau}=-H(w(\tau))G(\tau,s)\tag{1}$$
$$(60)^{\frac{1}{2}}$$

with G(s, s) = I. For time-dependent H(w(τ )), the propagator is:

$$G(\tau,s)={\mathcal{T}}\exp\left(-\int_{s}^{\tau}d\lambda\,H(w(\lambda))\right)$$

where T denotes time-ordering.

Assuming the initial perturbation δw(0) = 0, the solution for δw is:

$$\delta w(\tau)=-\int_{0}^{\tau}d s\,G(\tau,s)(\eta\Sigma)^{1/2}(s)\tilde{\xi}(s).$$
$$(61)$$
$$(62)^{\frac{1}{2}}$$
$$(63)^{\frac{1}{2}}$$

Now expanding L(w
′) = L(w + δw) to second order in δw gives

$$\delta L(\tau)=L(w^{\prime}(\tau))-L(w(\tau))$$ $$\approx g(w(\tau))^{\top}\delta w(\tau)+\frac{1}{2}\,\delta w(\tau)^{\top}H(w(\tau))\,\delta w(\tau).$$

Since E[
˜ξ(s)] = 0 and δw is linear in ˜ξ, E[δw(τ )] = 0, so

$$\mathbb{E}\big[g(w(\tau))^{\top}\delta w(\tau)\big]=0.$$
$$(64)$$
-g(w(τ ))⊤δw(τ )= 0. (64)
Thus the leading non-vanishing contribution to the expected loss shift comes from the quadratic term. Using the solution for δw,

In for $\delta w$,  $$\delta w(\tau)\,\delta w(\tau)^{\top}=\int_{0}^{\tau}ds\int_{0}^{\tau}du\ G(\tau,s)(\eta\Sigma)^{1/2}(s)\tilde{\xi}(s)\tilde{\xi}(u)^{\top}(\eta\Sigma)^{1/2}(u)G(\tau,u)^{\top}.$$  ation with $\mathbb{E}[\tilde{\xi}(s)\tilde{\xi}(u)^{\top}]=\delta(s-u)\,I$ gives
⊤. (65)
Taking the expectation with E[

$$\mathbb{E}\big{[}\delta w(\tau)\,\delta w(\tau)^{\top}\big{]}=\mathbb{E}\bigg{[}\int_{0}^{\tau}ds\,G(\tau,s)\,\eta(s)\Sigma(w^{\prime}(s))\,G(\tau,s)^{\top}\bigg{]}.\tag{1}$$
$$(65)$$
$$(66)$$

20