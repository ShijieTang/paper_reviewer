# On The Closed-Form Of Flow Matching: Generalization Does Not Arise From Target Stochasticity

Quentin Bertrand1∗
, Anne Gagneux2∗
, Mathurin Massias3∗
, Rémi Emonet14∗
1Université Jean Monnet Saint-Étienne, CNRS, Institut d'Optique Graduate School, Inria, Laboratoire Hubert Curien UMR 5516, F-42023 Saint-Étienne, France 2ENS de Lyon, CNRS, Université Claude Bernard Lyon 1, Inria, LIP UMR 5668, 69342 Lyon Cedex 07, France 3Inria, ENS de Lyon, CNRS, Université Claude Bernard Lyon 1, LIP UMR 5668, 69342 Lyon Cedex 07, France 4Institut Universitaire de France Code: https://github.com/generativemodels/closedformfm

## Abstract

Modern deep generative models can now produce high-quality synthetic samples that are often indistinguishable from real training data. A growing body of research aims to understand why recent methods, such as diffusion and flow matching techniques, generalize so effectively. Among the proposed explanations are the inductive biases of deep learning architectures and the stochastic nature of the conditional flow matching loss. In this work, we rule out the noisy nature of the loss as a key factor driving generalization in flow matching. First, we empirically show that in high-dimensional settings, the stochastic and closed-form versions of the flow matching loss yield nearly equivalent losses. Then, using state-of-the-art flow matching models on standard image datasets, we demonstrate that both variants achieve comparable statistical performance, with the surprising observation that using the closed-form can even improve performance.

## 1 Introduction

Recent deep generative models, such as diffusion (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2021) and flow matching models (Lipman et al., 2023; Albergo and Vanden-Eijnden, 2023; Liu et al., 2023), have achieved remarkable success in synthesizing realistic data across a wide range of domains. State-of-the-art diffusion and flow matching methods are now capable of producing multi-modal outputs that are virtually indistinguishable from human-generated content, including images (Stability AI, 2023), audio (Borsos et al., 2023), video (Villegas et al., 2022; Brooks et al., 2024), and text (Gong et al., 2023; Xu et al., 2025). A central question in deep generative modeling concerns the generalization capabilities and underlying mechanisms of these models. Generative models generalization remains a puzzling phenomenon, raising a number of challenging and unresolved questions: whether generative models truly generalize is still the subject of active debate. On one hand, several studies (Carlini et al., 2023; Somepalli et al., 2023b,a; Dar et al., 2023) have shown that large diffusion models are capable of memorizing individual samples from the training set, including licensed photographs, trademarked logos, and sensitive medical data. On the other hand, Kadkhodaie et al. (2024) have empirically demonstrated that while memorization can occur in low-data regimes, diffusion models trained on a *sufficiently large* dataset exhibit clear
∗Equal contribution. Correspondence: quentin.bertrand@inria.fr.

signs of generalization. Taken together, recent work points to a sharp phase transition between memorization and generalization (Yoon et al., 2023; Zhang et al., 2024). Multiple theories have been proposed to explain the puzzling generalization of diffusion and flow matching models. On the one hand, Kadkhodaie et al. (2024); Kamb and Ganguli (2025); Ross et al. (2025) suggested a geometric framework to understand the inductive bias of modern deep convolutional networks on images. On the other hand, Vastola (2025) suggested that generalization is due to the *noisy* nature of the training loss. In this work, we clearly answer the following question:
Does training on noisy/stochastic targets improve flow matching generalization?

If not, what are the main sources of generalization?

## Contributions.

- We challenge the prevailing belief that generalization in flow matching stems from an inherently noisy loss (Section 3.1). This assumption, largely supported by studies in low-dimensional settings, fails to hold in realistic high-dimensional data regimes.

- Instead, we observe that generalization in flow matching emerges precisely when the limitedcapacity neural network fails to approximate the *optimal closed-form velocity field* (Section 3.2).

- We identify two critical time intervals, at early and late times, where neural networks fail to approximate the optimal velocity field (Section 3.3). We show that generalization arises predominantly early along flow matching trajectories, aligning with the transition from the stochastic to the deterministic regime of the flow matching objective.

- Finally, on standard image datasets (CIFAR-10 and CelebA), we show that explicitly regressing against the optimal closed-form velocity field does not impair generalization and can, in some cases, enhance it (Section 4).

The manuscript is organized as follows. Section 2 reviews the fundamentals of conditional flow matching and recalls the closed-form of the "optimal" velocity field. Leveraging the closed-form expression of the flow matching velocity field, Section 3 investigates the key sources of generalization in flow matching. In Section 4, we introduce a learning algorithm based on the closed-form formula. Related work is discussed in detail in Section 5.

## 2 Recalls On Conditional Flow Matching

Let p0 = N (0,Id) be the source distribution2and pdata the data distribution. We are given n data points x
(1)*, . . . , x*(n) ∼ pdata, x
(i) ∈ R
d. The goal of flow matching is to find a velocity field u : R
d × [0, 1] → R
d, such that, if one solves on [0, 1] the ordinary differential equation

$$\begin{cases}x(0)=x_{0}\in\mathbb{R}^{d}\\ i x(t)=u(x(t),t)\end{cases}$$
$$(1)$$
x˙(t) = u(x(t), t)(1)
then the law of x(1) when x0 ∼ p0 is pdata: one says that u *transports* p0 to pdata. For every value of t between 0 and 1, the law of x(t) defines a *probability path*, denoted p(·|t) that progressively transforms p0 to pdata. If one knows the velocity field u, new samples can then be generated by sampling x0 from p0, solving the ordinary differential equation, and using x(1) as the generated point. In conditional flow matching, finding such a velocity field u is achieved in the following way.

(i) First, define a conditioning variable z independent of t, *e.g.,* z = x1 ∼ pdata,
(ii) Then, chose a conditional probability path p(·|z, t), e.g., p(·|z = x1, t) = N (tx1,(1 − t)
2Id).

Through the continuity equation (Lipman et al., 2024, Sec. 3.5), the choice (ii) of the conditional probability path p(·|*z, t*) defines a conditional velocity field u cond(*x, z, t*). With the choices (i)
and (ii), the conditional velocity field writes

$$u^{\mathrm{cond}}(x,z=x_{1},t)={\frac{x_{1}-x}{1-t}}$$
1 − t
. (2)
$$(2)$$
$\star$ . 
The choice (ii) of the conditional probability paths p(·|z = x1, t) fully defines a probability path p(·|t)
(by marginalization against z) and thus defines an *optimal velocity field* u
⋆(through the continuity equation), that transports p0 to pdata (Lipman et al., 2023, Thm. 1)

$$u^{\star}(x,t)=\mathbb{E}_{z|x,t}\;u^{\mathrm{cond}}(x,z,t)\;\;.$$
$$({\mathfrak{I}})$$

$$(4)$$
cond(*x, z, t*) . (3)
Hence, the optimal velocity u
⋆could be approximated by a neural network uθ : R
d × [0, 1] → R
d with parameters θ by minimizing

$${\cal L}_{\rm FM}(\theta)=\mathbb{E}_{t\sim{\cal U}([0,1])}\|u_{\theta}(x_{t},t)-u^{\star}(x_{t},t)\|^{2}\ \.$$

However, u
⋆is usually (believed) intractable, as a remedy, Lipman et al. (2023, Thm. 2) showed that LFM(θ) is equal, up to a constant, to the conditional flow matching loss. With the choices (i) and (ii)
made above, the conditional flow matching loss reads

$$\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{\begin{subarray}{c}x_{0}\sim p_{0}\\ x_{1}\sim\mathcal{U}_{\mathrm{peak}}\\ t\sim\mathcal{U}([0,1])\end{subarray}}\|u_{\theta}(x_{t},t)-\underbrace{u^{\mathrm{cond}}(x_{t},z=x_{1},t)}_{=\frac{x_{1}-x_{t}}{1-t}=x_{1}-x_{0}}\|^{2},$$

where xt := (1 − t)x0 + tx1. The objective LCFM is easy to approximate, since it is easy to sample from p0 = N (0,Id) and U([0, 1]); sampling from pdata is approximated by sampling from pˆdata := 1n Pn i=1 δx(i) . Although it seems natural, replacing pdata by pˆdata in (5) has a very important consequence: it makes the minimizer uˆ
⋆ of LFM available in closed-form, which we recall below.

Proposition 1 (Closed-form Formula of the Optimal Velocity). *When* pdata *is replaced by* pˆdata, with the previous choices (i) and *(ii), the optimal velocity field* uˆ
⋆in (3) *has a closed-form formula:*

$$(S)$$
$$\hat{u}^{\star}(x,t)=\sum_{i=1}^{n}\lambda_{i}(x,t)\frac{x^{(i)}-x}{1-t}\;\;,$$
$$(6)$$
$${\mathrm{with~}}\lambda(x,t)={\mathrm{softmax}}((-\frac{\|}{\|}))$$
∥x−tx(j)∥
2
2(1−t)
2 )j=1*,...,n*) ∈ R
n.
The notation uˆ
⋆emphasizes the velocity field is optimal for the *empirical* probability distribution pˆdata, not the true one pdata. Since u cond(x, z = x
(i), t) ∝ x
(i) − x, the optimal velocity field uˆ
⋆is a weighted average of the n different directions x
(i) − x. Note that the closed-form formula in Equation (6) can be found in various previous works, *e.g.,* Kamb and Ganguli (2025, Eq. 3), Biroli et al. (2024), Gao and Li (2024), Li et al. (2024) or Scarvelis et al. (2025), and can be generalized to other choices of continuous distribution p0 (*e.g.,* the uniform distribution, see Appendix A.1).

From Equation (6), as t → 1, the velocity field uˆ
⋆ diverges at any point x that does not coincide with one of the training samples x
(i), and it points in the direction of the nearest x
(i). This creates a paradox: solving the ordinary differential equation (1) with the velocity field uˆ
⋆can only produce training samples x
(i)(see Gao and Li 2024, Thm. 4.6 for a formal proof). Therefore, in practice, exactly minimizing the conditional flow matching loss would result in uθ = ˆu
⋆, meaning the model memorizes the training data and fails to generalize. This naturally yields the following question:
How can flow matching generalize if the optimal velocity field only generates training samples?

## 3 Investigating The Key Sources Of Generalization

In this section, we investigate the key sources of flow matching generalization using the closed-form formula of its velocity field. First in Section 3.1 we challenge the claim that generalization stems from the stochastic approximation u cond of the optimal velocity field uˆ
⋆. Then, in Section 3.2 we show that generalization arises when uθ fails to approximate the perfect velocity uˆ
⋆. Interestingly, the target velocity estimation particularly fails at two critical time intervals. Section 3.3 shows that one of these critical times is particularly important for generalization.

2 m o o ns t = 0.01 t = 0.21 t = 0.40 t = 0.60 t = 0.79 t = 0.99 0.0 0.5 1.0
−1 0 1 0.0 0.5 1.0 CI
FA
R-
1 0

−1 0 1 −1 0 1 −1 0 1 −1 0 1 −1 0 1
(a) **Non stochasticity of** uˆ
⋆**for high-dimensional real data**.

û
⋆(x, t) =
n
∑
i=1 p (z = x(i)| x, t) ucond (x, t,z = x(i))
û
⋆
û
⋆
x

## 3.1 Target Stochasticity Is Not What You Need

One recent hypothesis is that generalization arises from the fact that the regression target u cond of conditional flow matching is only a stochastic estimate of uˆ
⋆. The fact that the target regression objective only equals the true objective on average is referred to by Vastola (2025) as "generalization through variance". To challenge this assumption, we leverage Proposition 1, which states that the optimal velocity field uˆ
⋆(*x, t*) is a weighted sum of the n values of u cond(*x, t, z* = x
(i)) = x
(i)−x 1−t, for i ∈ [n], and show that, after a *small time value* t, this average is in practice equal to a single value in the expectation (see Figures 1a and 1b).

Comments on Figure 1a. To produce Figure 1a, we sample 256 pairs (x0, x1) from p0 × pˆdata.

For each value of t, we compute the cosine similarity between the optimal velocity field uˆ
⋆((1 −
t)x0 + tx1, t) and the conditional target u cond((1 − t)x0 + tx1, z = x1, t) = x1 − x0. The resulting similarities are aggregated and shown as histograms. The top row displays the results for the two-moons toy dataset (d = 2), and the bottom row displays the results for the CIFAR-10 dataset (Krizhevsky and Hinton 2009, d = 3072); n = 50k for both. As t increases, the histograms become increasingly concentrated around 1, indicating that uˆ
⋆aligns closely with a single conditional vector u cond. From Equation (6), this corresponds to a collapse towards 0 of all but one of the softmax weights λi(xt, t). This time corresponds to the collapse time studied by Biroli et al. (2024)
for diffusion; we discuss the connection in the related works (Section 5). On the two-moons toy dataset, this transition occurs for intermediate-to-large values of t, echoing the observations made in low-dimensional settings by Vastola (2025, Figure 1). In contrast, for high-dimensional real datasets, uˆ
⋆(*x, t*) aligns with a single conditional velocity field x
(i)−x, even at early time steps, suggesting that the non-stochastic regime dominates most of the generative process. This key difference between lowand high-dimensional data suggests that the transition time between the stochastic and non-stochastic regimes is strongly influenced by the dimensionality of the data.

# samples 10 # samples 100
# samples 1000 # samples 2000
# samples 3000 # samples 4000
# samples 5000 # samples 10000 E

xt || u θ
(xt , t) 
−
 
ˆu

?

(xt , t)
|| 2 18 N
ea re st Ne igh bo r D
ist
.

F
I
D
 D
I
N
O
:
 Test 2000 0.5 10 1000 0.0 0 0 0 1 t 10100 1000 2000 3000 4000 5000 10000
# samples 101001000 2000 3000 4000 5000 10000
# samples
Comments on Figure 1c. To further illustrate the strong impact of dimensionality, Figure 1c reports the proportion of samples xt (from a batch of 256) for which the cosine similarity between uˆ
⋆and u cond ∝ x
(i) − x exceeds 0.9, as a function of time t. This analysis is performed across multiple spatial resolutions of the Imagenette dataset (Howard, 2019), obtaining dim × dim images by spatial subsampling. Figure 1c reveals a sharp transition: as the dimensionality increases, the proportion of high-cosine matches rapidly converges to 100%. A practical implication of this behavior is that, for sufficiently large t, if x0 ∼ p0 and x
(i) ∼ pˆdata, then uˆ
⋆((1 − t)x0 + tx(i), t) is approximately proportional to x
(i) − x. Consequently, regressing on x
(i) or on the conditional velocity x1 − x0 becomes effectively equivalent. Section 4 investigates how to learn regressing against optimal velocity field uˆ
⋆, and empirically shows similar results between stochastic and non-stochastic targets.

The regime where flow matching matches stochasticity is mostly concentrated on a very short time interval, for small values of t. We hypothesize that the phenomenon observed here on the optimal velocity field uˆ
⋆ has major implications on the *learned* flow matching model uθ, which we further inspect in the next section.

## 3.2 Failure To Learn The Optimal Velocity Field

This subsection investigates how well the learned velocity field uθ approximates the optimal/ideal velocity field uˆ
⋆, and how the quality of this approximation correlates with generalization. To do so, we propose the following experiment.

Set up of Figure 2. To build Figure 2, we subsampled the CIFAR-10 dataset from 10 to 104samples.

For each size, we trained a flow matching model using a standard 34 million-parameter U-Net (see Appendix D for details). Following Kadkhodaie et al. (2024), the number of parameters of the network uθ remains fixed across dataset sizes. Importantly, the optimal velocity field uˆ
⋆itself depends on the dataset size: as the number of samples increases, the complexity of uˆ
⋆also grows. Thus, we expect the network uθ to accurately approximate the optimal velocity field uˆ
⋆for smaller dataset sizes.

Comments on Figure 2. The leftmost plot shows the average training error

$=\;\;\;\hat{a}^{\star}(x,\,t)|$
E x0∼p0 x1∼pˆdata
∥uθ(xt, t) − uˆ
⋆(xt, t)∥
2, where xt := (1 − t)x0 + tx1 ,
between the learned velocity uθ and the optimal empirical velocity field uˆ
⋆, evaluated across multiple time values t and dataset sizes. With only 10 samples (darkest curve), the network uθ closely approximates uˆ
⋆. As the dataset size increases, the complexity of uˆ
⋆ grows, and the approximation by uθ becomes less accurate. In particular, the approximation fails at two specific time intervals: around

t ≈ 0.15 and near t = 1. The failure near t = 1 is expected, as uˆ
⋆ becomes non-Lipschitz at t = 1.

Interestingly, the early-time failure at t ≈ 0.15 corresponds to the regime where uˆ
⋆and u cond start to correlate (see Figure 1a in Section 3.1). The middle plot of Figure 2 reports the FID-10k, computed on the test set in the DINOv2 embedding space (Oquab et al., 2024), for various dataset sizes. For a small dataset (*e.g.,* \#samples = 10), uθ approximates uˆ
⋆ well but does not generalize - the test FID exceeds 103. As the dataset size increases (1000 ≤ \#samples ≤ 3000), the approximation uθ becomes less accurate. Despite this, the model achieves lower FID scores on the test set but still memorizes the training data. The rightmost plot of Figure 2 illustrates this memorization by showing the average distance between each generated sample and its nearest neighbor in the training set. For larger datasets (\#samples ≥ 3000), this distance increases substantially, indicating that the model generalizes better. Overall, Figure 2 also suggests that the FID metric can be misleading, even when computed on the test set. For example, the model trained with 1000 samples has a low test FID but memorizes training examples.

Figure 2 confirms that generalization arises when the network uθ fails to estimate the optimal velocity field uˆ
⋆, and that this failure occurs at two specific time intervals. In Section 3.3, we investigate which of these two intervals is responsible for driving generalization.

## 3.3 When Does Generalization Arise?

To investigate whether the failure to approximate uˆ
⋆ matters the most at small or large values of t, we carry out the following experiment.

Set up of Figure 3. We first learn a velocity field uθ using standard conditional flow matching (see Appendix D), then we construct a hybrid model: we define a piecewise trajectory where the flow is governed by the optimal velocity field uˆ
⋆for times t ∈ [0, τ ], and by the learned velocity field uθ for times t ∈ [τ, 1], for a given threshold parameter τ ∈ [0, 1]. For the extreme case τ = 1, the full trajectory follows uˆ
⋆, and samples exactly match training data points. Conversely, when τ = 0, the entire trajectory is governed by uθ, yielding novel samples. Intermediate values of τ produce a mixture of both behaviors, which we interpret as reflecting varying degrees of generalization. To assess generalization, we measure the distance of generated samples to the dataset using the LPIPS metric (Zhang et al., 2018), which computes the feature distance between two images via some pretrained classification network. We define the distance of a generated sample x to a dataset D = {x
(1)*, . . . , x*(n)} as dist(x, D) = minx(i)∈D LPIPS(*x, x*(i)). We fix a random batch of 256 pure noise images from p0. Then, for various threshold values τ , we generate 256 images with the hybrid model, always starting from this batch. Finally, we measure the creativity of the hybrid model as the mean of the aforementioned LPIPS distances between the 256 generated samples and the dataset. Comments on Figure 3. The top row displays the LPIPS distances as τ varies, on the CIFAR-10
(left) and CelebA - 64 × 64 (right) datasets. For τ ≤ 0.2, the hybrid model remains as creative as uθ, despite following uˆ
⋆in the first steps. For τ > 0.2, the LPIPS distance starts dropping. On the displayed generated samples (bottom rows), we in fact see that as soon as τ ≥ 0.4, the sample generated by the hybrid model is almost the same as the one obtained with uˆ
⋆(τ = 1). This means the final image is already determined at t = 0.4, and despite the generalization capacity of the learned velocity field uθ, following it only after t ≥ 0.4 is not enough to create a new image: generalization occurs early and cannot fully be explained by the failure to correctly approximate u
⋆ *at large* t.

Although we have shown that the stochastic phase was limited to small values of t in real-data settings, we have not yet definitively ruled it out as the cause of generalization. In the following Section 4, we introduce a learning procedure designed to address this question directly.

## 4 Learning With The Closed-Form Formula

In this section, in order to discard the impact of stochastic target on the generalization, we propose to directly regress against the closed-form formula in Equation (6).

## 4.1 Empirical Flow Matching

Regressing against the closed-form uˆ
⋆, defined in Equation (6), at a point (xt, t) requires computing a weighted sum of the conditional velocity fields over all the n training points x
(i). For a dataset of n samples of size d, and a batch of size |B|, computing the weights of the exact closed-form formula uˆ
⋆(*x, t*) of flow matching requires O(n *× |B| ×* d). These computations are prohibitive since they must be performed for each batch. One natural idea is to estimate the closed-form formula uˆ
⋆
(Equation (6)), by a Monte Carlo approximation (Equation (8)), using M ≤ n samples b
(1)*, . . . , b*(M):

$$\begin{array}{r l}{{{\mathcal L}_{\mathrm{EFM}}(\theta)=\mathbb{E}\qquad}}&{{x_{0}\sim p_{0}\qquad}}&{{\|u_{\theta}(x_{t},t)-\hat{u}_{M}^{\star}(x_{t},t)\|^{2}\ \ ,}}\\ {{}}&{{x_{t}\sim p_{\mathrm{data}}}}\\ {{}}&{{t\sim{\mathcal U}([0,1])}}\\ {{}}&{{b^{(2)},\ldots,b^{(M)}\sim p_{\mathrm{data}}}}\end{array}$$
2, (7)
with xt = (1 − t)x0 + tx1, b
(1) := x1, and

$$\hat{u}_{M}^{*}(x,t)=\sum_{j=1}^{M}\lambda(x,t)\frac{b^{(j)}-x}{1-t}\,\ \ \lambda(x,t)=\mbox{softmax}\left(\left(-\frac{\|x-tx^{(l)}\|^{2}}{2(1-t)^{2}}\right)_{l=1,\ldots,n}\right).\tag{8}$$
$$\left(T\right)$$

$$(9)$$

The formulation in Equation (7) may appear naive at first glance. Still, it hinges on a crucial trick:
the Monte Carlo estimate is computed using a batch that systematically includes the point x1, that generated the current xt. If instead b
(1) were sampled independently from pˆdata, this could introduce a sampling bias (see Ryzhakov et al. 2024, Appendix B, and the corresponding OpenReview comments3 for an in-depth discussion). Proposition 2 shows that the estimate uˆ
⋆
M is unbiased and has lower variance than the standard conditional flow matching target. Proposition 2. *We denote the conditional probability distribution* p(z = x
(i)| x, t) *over* {x
(i)}
n i=1 by pˆdata(z | x, t)*. With no constraints on the learned velocity field* uθ, i) *The minimizer of Equation* (7) writes, for all (*x, t*)

$\{\mathcal{A}_{\mathcal{D}}\subseteq\mathbb{R}\}$
$$\begin{array}{r l}{{\mathbb{E}_{\begin{array}{c}{{b^{(1)}\sim\hat{p}_{\mathrm{data}}(\cdot|x,t)}\end{array}}\left[\hat{u}_{M}^{\star}(x,t)\right]}\\ {{b^{(2)},\ldots,b^{(M)}\sim\hat{p}_{\mathrm{data}}}\end{array}}}\end{array}.$$
M(*x, t*)] . (9)
ii) *In addition, for all* (x, t)*, the minimizer of Equation* (7) *equals the optimal velocity field, i.e.,*

E b
(1)∼pˆdata(·|x,t)
b
(2)*,...,b*(M)∼pˆdata
$$\begin{array}{r l}{{}}&{{}}\\ {{}}&{{}}\\ {{}}&{{}}\\ {{}}&{{}}\end{array}\left[{\hat{u}}_{M}^{\star}(x,t)\right]={\hat{u}}^{\star}(x,t)\;\;.$$
$\overline{\phantom{a}}$  4. 
⋆(*x, t*) . (10)
iii) *The conditional variance of the estimator* uˆ
⋆
M *is smaller than the usual conditional variance:*

$$\begin{array}{l}\mbox{Var}_{b^{(1)}\sim\beta_{\rm class}\cdot(\,|\,x,t)}\ \left[\hat{u}_{M}^{\star}(x,t)\right]\leq\mbox{Var}_{b^{(1)}\sim\beta_{\rm class}\cdot(\,|\,x,t)}\left[u^{\rm cond}(x,b^{(1)},t)\right].\\ \mbox{}_{b^{(2)}},...,b^{(M)}\sim\beta_{\rm class}\end{array}\tag{11}$$

The proof of Proposition 2 is provided in Appendix B.3. The estimator uˆ
⋆M of the optimal field uˆ
⋆is closely related to self-normalized importance sampling (see Appendix B.2 and Owen 2013, Chap. 9.2), as well as to Rao-Blackwellized estimators (Casella and Robert, 1996; Cardoso et al., 2022). As discussed in Ryzhakov et al. (2024), self-normalized importance sampling estimators of uˆ
⋆are generally biased, in the sense that: Eb
(1)*,...,b*(M)∼pˆdata uˆ
⋆M(xt, t) ̸= ˆu
⋆(xt, t) . A key insight is that our estimator includes b
(1) ∼ pˆdata(· | xt, t), which leads to the main result of Proposition 2.

In Section 4.2, we demonstrate that Algorithm 2, designed to solve Equation (7), yields consistent improvements on high-dimensional datasets such as CIFAR-10 and CelebA. Additional details on the unbiasedness of LEFM can be found in the supplementary material (Appendix B). From a computational perspective, despite requiring M additional samples, Algorithm 2 remains significantly more efficient than increasing the batch size by a factor of M: the M samples are merely averaged (with weights), while the backpropagation remains identical to that of Algorithm 1.

| Algorithm 1 Vanilla Flow Matching for k in 1, . . . , niter do t ∼ U([0, 1]) x0 ∼ N (0,Id), x1 ∼ pˆdata, xt = (1 − t)x0 + tx1 x1 − xt cond(xt, t) = u = x1 − x0 1 − t   | Algorithm 2 Empirical Flow Matching param :M // Number of samples in the empirical mean for k in 1, . . . , niter do x0 ∼ N (0,Id), x1 ∼ pˆdata, t ∼ U([0, 1]) xt = (1 − t)x0 + tx1 b (1) = x1 ∀j ∈J2,MK, b (j) ∼ pˆdata // Samples from pˆdata (j)−xt h softmax  − 2 i uˆ ⋆ M(xt, t) = PM b ∥xt−t·b∥ j=1 1−t · 2(1−t) 2 j 2 L(θ) = ∥uθ(xt, t) − uˆ ⋆ M(xt, t)∥ Compute ∇L(θ) and update θ return uθ   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| L(θ) = uθ(xt, t) − u cond(xt, t) 2 Compute ∇L(θ) and update θ                                                                                                           |   |
| return uθ                                                                                                                                                               |   |

## 4.2 Experiments

We now learn with empirical flow matching (EFM, Equation (7) and Algorithm 2) in practical high-dimensional settings. Our goal with this empirical investigation is first to observe if regressing against a more deterministic target leads to performance improvement/degradation. Datasets and Models. We perform experiments on the image datasets CIFAR-10 (Krizhevsky and Hinton, 2009) and CelebA 64 × 64 (Liu et al., 2015). For the experiments, we compare vanilla conditional flow matching (Lipman et al., 2023; Liu et al., 2023; Albergo and Vanden-Eijnden, 2023), optimal transport flow matching (Pooladian et al., 2023; Tong et al., 2024), and the empirical flow matching in Algorithm 2, for multiple numbers of samples M to estimate the empirical mean. Training details are in Appendix D. Metrics. To assess generalization performance, we use the standard Fréchet Inception Distance (Heusel et al., 2017) with Inception-V3 (Szegedy et al., 2016) but we also follow the recommendation of Stein et al. (2023) using the DINOv2 embedding (Oquab et al., 2023), which is known to a more expressive and discriminative embedding, that leads to a less biased evaluation. We also measure the FID between the generated and the train and test sets, rather than only on the training set, as is often done in generative modeling benchmarks. On Figure 2, we also displayed a memorization metric that would detect a pure copy of the training set. Overall, defining and quantifying the generalization ability of generative models is overall a challenging task: train and test FID are known to be imperfect
(Stein et al., 2023; Jiralerspong et al., 2023; Parmar et al., 2022), yet no superior competitor has emerged.

EFM - 256 EFM - 128 CFM OTCFM EFM - 1000 FI
D
 C
IFA
R - 
1 0 Inception - Train 10 Inception - Test 400 DINO - Train 400 DINO - Test 8 2 × 102 103 4 2 × 102 103 6 2 × 102 10 3 200 2 × 10 2 10 3 200 FI
D
 C
el e b A
 - 
6 4 10 10 200 300 102 3 
× 10 2
# epochs 200 300 102 3 
× 102
# epochs 5 102 3 
× 102
# epochs 5 10 2 3 
× 10 2
# epochs
Comments on Figure 4. Figure 4 compares vanilla flow matching, OTCFM, and the empirical flow matching (EFM, Algorithm 2) approaches using various numbers of samples to estimate the empirical mean, M ∈ {128, 256, 1000}. First, we observe that learning with a more deterministic target does not degrade either training or testing performance, across both types of embeddings. On the contrary, we consistently observe modest but steady improvements as stochasticity is reduced. For both CIFAR-10 and CelebA, increasing the number of samples M used to compute the empirical mean—*i.e.,* , making the targets less stochastic—leads to more stable improvements. It is worth noting that Algorithm 2 has a computational complexity of O(M *× |B| ×* d), where |B| is the batch size, M is the number of samples used to estimate the empirical mean, and d is the sample dimension.

In our experiments, choosing M = |B| = 128 yielded a modest time overhead. For empirical flow matching, we experimented with several values beyond M = 1000 (*e.g.,* M = 2000, M = 5000). The results were nearly identical to those obtained with M = 1000, with curves being visually indistinguishable. Therefore, we chose not to report results for M ≥ 1000.

## 5 Related Work

The existing literature related to our study can be roughly divided into three approaches: leveraging the closed-form, studies on the memorization vs generalization, and characterization of the different phases of the generating dynamics.

Leveraging the closed-form. Proposition 1 has been leveraged in several ways. The closest existing work is by Ryzhakov et al. (2024), who propose to regress against uˆ
⋆as we do in Section 4.

Nevertheless, their motivation is that reducing the variance of the velocity field estimation makes learning more accurate: as explained in Section 3.1, we argue this claim rests on misleading 2D- based intuitions (*e.g.,* Figure 1, challenged by Section 3.1). The idea of regressing against a more deterministic target (as Proposition 2 shows) derived from the optimal closed-form velocity field has also been empirically explored for diffusion models (Xu et al., 2023). Scarvelis et al. (2025) bypass training, and suggest using a smoothed version of uˆ
⋆to generate novel samples. In a work specific to images and convolutional neural networks, Kamb and Ganguli (2025) suggested that flow matching indeed ends up learning an optimal velocity, but that instead of memorizing training samples, the velocity memorizes a combination of all possible patches in an image and across the images. They show remarkable agreement between their theory and the trajectories followed by learned vector fields, but their work is limited to convolutional architectures, and was recently extended to a larger class of architectures (Lukoianov et al., 2025). Memorization and reasons for generalization. Kadkhodaie et al. (2024) directly relates the transition from memorization to generalization to the size of the training dataset, and proposes a geometric interpretation. We provide a complementary experiment in Section 3.2, quantifying how much the network fails to estimate the optimal velocity field. Gu et al. (2025) provide a detailed experimental investigation into the potential causes for generalization, primarily based on the characteristics of the dataset and choices for training and model. Vastola (2025) explores different factors of generalization in the case of diffusion, with a special focus on the stochasticity of the target objective in the learning problem. Through a physic-based modeling of the generative dynamics, they study the covariance matrix of the noisy estimation of the exact score. In our work, we believe that we have shown that this claim was not valid for real high-dimensional data. Niedoba et al. (2025) study the poor approximation of the exact score by the learned models: like Kamb and Ganguli (2025),
they suggest that the generalization of the learned models comes from memorization of many patches in the training data. Temporal regimes. Biroli et al. (2024); Sclocchi et al. (2025) provide an analysis of the exact score, the counterpart of the exact velocity field for diffusion. For a multimodal target distribution, the authors identify three phases (we keep the convention that t = 0 is noise and t = 1 is target): for t < t1, all trajectories are indistinguishable; for t1 *< t < t*2, trajectories converging to different modes separate; for *t > t*2, trajectories all point to the training dataset. In the case of Gaussians mixtures target, they highlight the dependency of t2 in the dimension and the number of samples, in O ((log n)/d), meaning that the first phases are observable only if the number of training points is exponential in the dimension. The methodology they adopt to validate the existence of such t2 on real data relies on the stochasticity of the backward generative process, which does not hold in the case of flow matching. Our experiments on *learned* flow matching models allow us to take this theoretical study on memorization and temporal behaviors of generative processes a step further.

## 6 Conclusion, Limitations And Broader Impact

Conclusion. By challenging the assumption that stochasticity in the loss function is a key driver of generalization, our findings help clarify the role of approximation of the exact velocity field in flow matching models. Beyond the different temporal phases in the generation process that we have identified, we expect further results to be obtained by uncovering new properties of the true velocity field. Limitation. Our work is mainly empirical, with a focus on *learned* models, but did not precisely characterize the learned velocity field, in particular, how it behaves outside the trajectories defined by the optimal velocity. Leveraging existing work on the inductive biases of the architectures at hand seems like a promising venue. Another limitation is that we did not investigate the interaction between the architectural inductive bias, and optimization procedures: this is a very challenging, but active area of research (Boursier and Flammarion, 2025; Bonnaire et al., 2025; Favero et al., 2025). Broader impact. We hope that identifying the key factors of generalization will lead to improved training efficiency. However, generative models also raise concerns related to misinformation (notably deepfakes), data privacy, and potential misuse in generating synthetic but realistic content.

## 7 Acknowledgments

The authors thank the Blaise Pascal Center for its computational support, using the SIDUS (Quemener and Corvellec, 2013) solution.

## References

M. S. Albergo and E. Vanden-Eijnden. Building normalizing flows with stochastic interpolants. *ICLR*,
2023.

G. Biroli, T. Bonnaire, V. de Bortoli, and M. Mézard. Dynamical regimes of diffusion models. *Nature* Communications, 15(1):9957, 2024.

T. Bonnaire, R. Urfin, G. Biroli, and M. Mézard. Why diffusion models don't memorize: The role of implicit dynamical regularization in training. *arXiv preprint arXiv:2505.17638*, 2025.

Z. Borsos, R. Marinier, D. Vincent, E. Kharitonov, O. Pietquin, M. Sharifi, D. Roblek, O. Teboul, D. Grangier, M. Tagliasacchi, et al. Audiolm: a language modeling approach to audio generation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023.

E. Boursier and N. Flammarion. Simplicity bias and optimization threshold in two-layer ReLu networks. *ICML*, 2025.

T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman, C. Ng, R. Wang, and A. Ramesh. Video generation models as world simulators. 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.

G. Cardoso, S. Samsonov, A. Thin, E. Moulines, and J. Olsson. Br-snis: bias reduced self-normalized importance sampling. *NeurIPS*, 35:716–729, 2022.

N. Carlini, J. Hayes, M. Nasr, M. Jagielski, V. Sehwag, F. Tramer, B. Balle, D. Ippolito, and E. Wallace.

Extracting training data from diffusion models. In 32nd USENIX Security Symposium (USENIX Security 23), pages 5253–5270, 2023.

G. Casella and C. P. Robert. Rao-blackwellisation of sampling schemes. *Biometrika*, 83(1):81–94, 1996.

S. U. H. Dar, A. Ghanaat, J. Kahmann, I. Ayx, T. Papavassiliu, S. O. Schoenberg, and S. Engelhardt.

Investigating data memorization in 3d latent diffusion models for medical image synthesis. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 56–65. Springer, 2023.

A. Favero, A. Sclocchi, and M. Wyart. Bigger isn't always memorizing: Early stopping overparameterized diffusion models. *arXiv preprint arXiv:2505.16959*, 2025.

A. Gagneux, S. Martin, R. Emonet, Q. Bertrand, and M. Massias. A visual dive into conditional flow matching. In *The Fourth Blogpost Track at ICLR*, 2025.

R. Gao, E. Hoogeboom, J. Heek, V. de Bortoli, K. Murphy, and T. Salimans. Diffusion meets flow matching: Two sides of the same coin. *ICLR Blogpost*, 2025.

W. Gao and M. Li. How do flow matching models memorize and generalize in sample data subspaces?

arXiv preprint arXiv:2410.23594, 2024.

S. Gong, M. Li, J. Feng, Z. Wu, and L. Kong. Diffuseq: Sequence to sequence text generation with diffusion models. *ICLR*, 2023.

X. Gu, C. Du, T. Pang, C. Li, M. Lin, and Y. Wang. On memorization in diffusion models. *TMLR*,
2025.

M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. *NeurIPS*, 30, 2017.

J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. *NeuRIPS*, 2020. J. Howard. Imagenette: A smaller subset of 10 easily classified classes from imagenet, March 2019.

URL https://github.com/fastai/imagenette.

C.-W. Huang, J. H. Lim, and A. C. Courville. A variational perspective on diffusion-based generative models and score matching. *NeurIPS*, 34:22863–22876, 2021.

M. Jiralerspong, J. Bose, I. Gemp, C. Qin, Y. Bachrach, and G. Gidel. Feature likelihood divergence:
evaluating the generalization of generative models using samples. *NeurIPS*, 2023.

Z. Kadkhodaie, F. Guth, E. P. Simoncelli, and S. Mallat. Generalization in diffusion models arises from geometry-adaptive harmonic representations. *ICLR*, 2024.

M. Kamb and S. Ganguli. An analytic theory of creativity in convolutional diffusion models. *ICML*,
2025.

A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. 2009. S. Li, S. Chen, and Q. Li. A good score does not lead to a good generative model. *arXiv preprint* arXiv:2401.04856, 2024.

Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling.

ICLR, 2023.

Y. Lipman, M. Havasi, P. Holderrieth, N. Shaul, M. Le, B. Karrer, R. T. Chen, D. Lopez-Paz, H. Ben-Hamu, and I. Gat. Flow matching guide and code. *arXiv preprint arXiv:2412.06264*, 2024.

X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. ICLR, 2023.

Z. Liu, P. Luo, X. Wang, and X. Tang. Deep learning face attributes in the wild. In *Proceedings of* International Conference on Computer Vision (ICCV), December 2015.

A. Lukoianov, C. Yuan, J. Solomon, and V. Sitzmann. Locality in image diffusion models emerges from data statistics. *NeurIPS*, 2025.

S. Martin, A. Gagneux, P. Hagemann, and G. Steidl. Pnp-flow: Plug-and-play image restoration with flow matching. *ICLR*, 2025.

A. Q. Nichol and P. Dhariwal. Improved denoising diffusion probabilistic models. In *ICML*, pages 8162–8171. PMLR, 2021.

M. Niedoba, B. Zwartsenberg, K. Murphy, and F. Wood. Towards a mechanistic explanation of diffusion model generalization. *ICML*, 2025.

M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. Dinov2: Learning robust visual features without supervision. *arXiv* preprint arXiv:2304.07193, 2023.

M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, R. Howes, P.-Y. Huang, H. Xu, V. Sharma, S.-W. Li, W. Galuba, M. Rabbat, M. Assran, N. Ballas, G. Synnaeve, I. Misra, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski. Dinov2: Learning robust visual features without supervision. *TMLR*, 2024.

A. B. Owen. *Monte Carlo theory, methods and examples*. https://artowen.su.domains/
mc/, 2013.

G. Parmar, R. Zhang, and J.-Y. Zhu. On aliased resizing and surprising subtleties in gan evaluation.

In *CVPR*, 2022.

A.-A. Pooladian, H. Ben-Hamu, C. Domingo-Enrich, B. Amos, Y. Lipman, and R. T. Chen. Multisample flow matching: Straightening flows with minibatch couplings. ICML, 2023.

E. Quemener and M. Corvellec. Sidus—the solution for extreme deduplication of an operating system. *Linux Journal*, 2013(235):3, 2013.

C. P. Robert, G. Casella, and G. Casella. *Monte Carlo statistical methods*, volume 2. Springer, 1999. B. L. Ross, H. Kamkari, T. Wu, R. Hosseinzadeh, Z. Liu, G. Stein, J. C. Cresswell, and G. Loaiza-
Ganem. A geometric framework for understanding memorization in generative models. *ICLR*, 2025.

G. Ryzhakov, S. Pavlova, E. Sevriugov, and I. Oseledets. Explicit flow matching: On the theory of flow matching algorithms with applications. In *ICOMP*, 2024.

C. Scarvelis, H. S. B. de Ocáriz, and J. Solomon. Closed-form diffusion models. *TMLR*, 2025. A. Sclocchi, A. Favero, and M. Wyart. A phase transition in diffusion models reveals the hierarchical nature of data. *PNAS*, 122(1):e2408799121, 2025.

J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In *ICML*, 2015.

G. Somepalli, V. Singla, M. Goldblum, J. Geiping, and T. Goldstein. Understanding and mitigating copying in diffusion models. *NeurIPS*, 36:47783–47803, 2023a.

G. Somepalli, V. Singla, M. Goldblum, J. Geiping, and T. Goldstein. Diffusion art or digital forgery?

investigating data replication in diffusion models. In *Proceedings of the IEEE/CVF conference on* computer vision and pattern recognition, pages 6048–6058, 2023b.

Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. *ICLR*, 2021.

Stability AI. https://stability.ai/stablediffusion, 2023. Accessed: 2023-09-09. G. Stein, J. Cresswell, R. Hosseinzadeh, Y. Sui, B. Ross, V. Villecroze, Z. Liu, A. L. Caterini, E. Taylor, and G. Loaiza-Ganem. Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models. *NeurIPS*, 36:3732–3784, 2023.

C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818–2826, 2016.

A. Tong, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, K. Fatras, G. Wolf, and Y. Bengio.

Improving and generalizing flow-based generative models with minibatch optimal transport. In TMLR, 2024. URL https://openreview.net/forum?id=CD9Snc73AW.

J. J. Vastola. Generalization through variance: how noise shapes inductive biases in diffusion models.

ICLR, 2025.

R. Villegas, M. Babaeizadeh, P.-J. Kindermans, H. Moraldo, H. Zhang, M. T. Saffar, S. Castro, J. Kunze, and D. Erhan. Phenaki: Variable length video generation from open domain textual descriptions. In *ICLR*, 2022.

M. Xu, T. Geffner, K. Kreis, W. Nie, Y. Xu, J. Leskovec, S. Ermon, and A. Vahdat. Energy-based diffusion language models for text generation. *ICLR*, 2025.

Y. Xu, S. Tong, and T. Jaakkola. Stable target field for reduced variance score estimation in diffusion models. *ICLR*, 2023.

T. Yoon, J. Y. Choi, S. Kwon, and E. K. Ryu. Diffusion probabilistic models generalize when they fail to memorize. In ICML 2023 workshop on structured probabilistic inference & generative modeling, 2023.

H. Zhang, J. Zhou, Y. Lu, M. Guo, P. Wang, L. Shen, and Q. Qu. The emergence of reproducibility and consistency in diffusion models. In *ICML*, 2024.

R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep features as a perceptual metric. In *Proceedings of the IEEE conference on computer vision and* pattern recognition, pages 586–595, 2018.

## Neurips Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: **The papers not including the checklist will be desk rejected.** The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit. Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist: - You should answer [Yes] , [No] , or [NA] . - [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.

- Please provide a short (1–2 sentence) justification right after your answer (even for NA). The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper. The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found. IMPORTANT, please: - **Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist"**, - **Keep the checklist subsection headings, questions/answers and guidelines below.** - **Do not modify the questions and only use the provided macros for your answers**.

## 1. **Claims**

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: each claim of the abstract refers to a specific subsection of the paper, that provide empirical evidence of the claim. Guidelines: - The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We do have a specific section for the limitation of our work Guidelines: - The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper. - The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. **Theory Assumptions And Proofs**

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: all results are encapsulated in clearly defined statements, and proofs are provided in appendix.

Guidelines:
- The answer NA means that the paper does not include theoretical results.

- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced. - All assumptions should be clearly stated or referenced in the statement of any theorems. - The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. **Experimental Result Reproducibility**

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## Answer: [Yes]

Justification: We provided as many details as possible in order to reproduce the results, in particular, we refer to the public implementation we used, including the specific (default) parameters used. Guidelines: - The answer NA means that the paper does not include experiments. - If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. **Open Access To Data And Code**

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: Code will be made available along with publication Guidelines: - The answer NA means that paper does not include experiments requiring code. - Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/
guides/CodeSubmissionPolicy) for more details.

- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details.

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. **Experimental Setting/Details**

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: We provide a specific appendix with the experimental details Guidelines: - The answer NA means that the paper does not include experiments. - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. **Experiment Statistical Significance**

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes] Justification: We do not report error bars, however, we do specify the number of samples used for the FID computation and highlight the strong weaknesses of the FID metric. Guidelines: - The answer NA means that the paper does not include experiments. - The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors). - It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. **Experiments Compute Resources**

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: we specified what type of GPU we used Guidelines: - The answer NA means that the paper does not include experiments. - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. **Code Of Ethics**

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS
Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes] Justification: [NA] Guidelines: - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. **Broader Impacts**

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [Yes] Justification: there is a dedicated broader impact section Guidelines: - The answer NA means that there is no societal impact of the work performed. - If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses (e.g.,
disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies
(e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. **Safeguards**

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [No] Justification: We work on standard image datasets Guidelines: - The answer NA means that the paper poses no such risks. - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. **Licenses For Existing Assets**

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: We properly refer the torchcfm and PnPflow codebase. Guidelines:
- The answer NA means that the paper does not use existing assets.

- The authors should cite the original paper that produced the code package or dataset. - The authors should state which version of the asset is used and, if possible, include a URL. - The name of the license (e.g., CC-BY 4.0) should be included for each asset. - For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. **New Assets**

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: [NA] Guidelines: - The answer NA means that the paper does not release new assets. - Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

- The paper should discuss whether and how consent was obtained from people whose asset is used.

- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. **Crowdsourcing And Research With Human Subjects**

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: [NA] Guidelines: - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

15. **Institutional review board (IRB) approvals or equivalent for research with human subjects**
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: [NA] Guidelines: - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. **Declaration Of Llm Usage**

Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [No] Justification: LLMs were only used for grammatical purposes. Guidelines: - The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

- Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.