# On the Closed-Form of Flow Matching: Generalization Does Not Arise from Target Stochasticity

Quentin Bertrand<sup>1</sup><sup>∗</sup> , Anne Gagneux<sup>2</sup><sup>∗</sup> , Mathurin Massias<sup>3</sup><sup>∗</sup> , Rémi Emonet<sup>14</sup><sup>∗</sup>

<sup>1</sup>Université Jean Monnet Saint-Étienne, CNRS, Institut d'Optique Graduate School, Inria, Laboratoire Hubert Curien UMR 5516, F-42023 Saint-Étienne, France

<sup>2</sup>ENS de Lyon, CNRS, Université Claude Bernard Lyon 1, Inria,

LIP UMR 5668, 69342 Lyon Cedex 07, France

3 Inria, ENS de Lyon, CNRS, Université Claude Bernard Lyon 1,

LIP UMR 5668, 69342 Lyon Cedex 07, France

4 Institut Universitaire de France

Code: <https://github.com/generativemodels/closedformfm>

## Abstract

Modern deep generative models can now produce high-quality synthetic samples that are often indistinguishable from real training data. A growing body of research aims to understand why recent methods, such as diffusion and flow matching techniques, generalize so effectively. Among the proposed explanations are the inductive biases of deep learning architectures and the stochastic nature of the conditional flow matching loss. In this work, we rule out the noisy nature of the loss as a key factor driving generalization in flow matching. First, we empirically show that in high-dimensional settings, the stochastic and closed-form versions of the flow matching loss yield nearly equivalent losses. Then, using state-of-the-art flow matching models on standard image datasets, we demonstrate that both variants achieve comparable statistical performance, with the surprising observation that using the closed-form can even improve performance.

## 1 Introduction

Recent deep generative models, such as diffusion [\(Sohl-Dickstein et al.,](#page-12-0) [2015;](#page-12-0) [Ho et al.,](#page-11-0) [2020;](#page-11-0) [Song](#page-12-1) [et al.,](#page-12-1) [2021\)](#page-12-1) and flow matching models [\(Lipman et al.,](#page-11-1) [2023;](#page-11-1) [Albergo and Vanden-Eijnden,](#page-10-0) [2023;](#page-10-0) [Liu et al.,](#page-11-2) [2023\)](#page-11-2), have achieved remarkable success in synthesizing realistic data across a wide range of domains. State-of-the-art diffusion and flow matching methods are now capable of producing multi-modal outputs that are virtually indistinguishable from human-generated content, including images [\(Stability AI,](#page-12-2) [2023\)](#page-12-2), audio [\(Borsos et al.,](#page-10-1) [2023\)](#page-10-1), video [\(Villegas et al.,](#page-12-3) [2022;](#page-12-3) [Brooks et al.,](#page-10-2) [2024\)](#page-10-2), and text [\(Gong et al.,](#page-10-3) [2023;](#page-10-3) [Xu et al.,](#page-12-4) [2025\)](#page-12-4).

A central question in deep generative modeling concerns the generalization capabilities and underlying mechanisms of these models. Generative models generalization remains a puzzling phenomenon, raising a number of challenging and unresolved questions: whether generative models truly generalize is still the subject of active debate. On one hand, several studies [\(Carlini et al.,](#page-10-4) [2023;](#page-10-4) [Somepalli](#page-12-5) [et al.,](#page-12-5) [2023b,](#page-12-5)[a;](#page-12-6) [Dar et al.,](#page-10-5) [2023\)](#page-10-5) have shown that large diffusion models are capable of memorizing individual samples from the training set, including licensed photographs, trademarked logos, and sensitive medical data.

On the other hand, [Kadkhodaie et al.](#page-11-3) [\(2024\)](#page-11-3) have empirically demonstrated that while memorization can occur in low-data regimes, diffusion models trained on a *sufficiently large* dataset exhibit clear

<sup>∗</sup>Equal contribution. Correspondence: quentin.bertrand@inria.fr.

signs of generalization. Taken together, recent work points to a sharp phase transition between memorization and generalization [\(Yoon et al.,](#page-12-7) [2023;](#page-12-7) [Zhang et al.,](#page-12-8) [2024\)](#page-12-8). Multiple theories have been proposed to explain the puzzling generalization of diffusion and flow matching models. On the one hand, [Kadkhodaie et al.](#page-11-3) [\(2024\)](#page-11-3); [Kamb and Ganguli](#page-11-4) [\(2025\)](#page-11-4); [Ross et al.](#page-12-9) [\(2025\)](#page-12-9) suggested a geometric framework to understand the inductive bias of modern deep convolutional networks on images. On the other hand, [Vastola](#page-12-10) [\(2025\)](#page-12-10) suggested that generalization is due to the *noisy* nature of the training loss. In this work, we clearly answer the following question:

*Does training on noisy/stochastic targets improve flow matching generalization? If not, what are the main sources of generalization?*

#### Contributions.

- We challenge the prevailing belief that generalization in flow matching stems from an inherently noisy loss (Section [3.1\)](#page-3-0). This assumption, largely supported by studies in low-dimensional settings, fails to hold in realistic high-dimensional data regimes.
- Instead, we observe that generalization in flow matching emerges precisely when the limitedcapacity neural network fails to approximate the *optimal closed-form velocity field* (Section [3.2\)](#page-4-0).
- We identify two critical time intervals, at early and late times, where *neural networks fail to approximate the optimal velocity field* (Section [3.3\)](#page-5-0). We show that generalization arises predominantly early along flow matching trajectories, aligning with the transition from the stochastic to the deterministic regime of the flow matching objective.
- Finally, on standard image datasets (CIFAR-10 and CelebA), we show that explicitly regressing against the optimal closed-form velocity field does not impair generalization and can, in some cases, enhance it (Section [4\)](#page-6-0).

The manuscript is organized as follows. Section [2](#page-1-0) reviews the fundamentals of conditional flow matching and recalls the closed-form of the "optimal" velocity field. Leveraging the closed-form expression of the flow matching velocity field, Section [3](#page-2-0) investigates the key sources of generalization in flow matching. In Section [4,](#page-6-0) we introduce a learning algorithm based on the closed-form formula. Related work is discussed in detail in Section [5.](#page-8-0)

## 2 Recalls on conditional flow matching

Let <sup>p</sup><sup>0</sup> <sup>=</sup> N (0,Id) be the source distribution[<sup>2</sup>](#page-1-1) and pdata the data distribution. We are given n data points x (1), . . . , x(n) ∼ <sup>p</sup>data, <sup>x</sup> (i) ∈ R d . The goal of flow matching is to find a velocity field u : R <sup>d</sup> × [0, 1] → R d , such that, if one solves on [0, 1] the ordinary differential equation

$$\begin{cases} x(0) = x_0 \in \mathbb{R}^d \\ \dot{x}(t) = u(x(t), t) \end{cases} \quad (1)$$

then the law of <sup>x</sup>(1) when <sup>x</sup><sup>0</sup> ∼ <sup>p</sup><sup>0</sup> is <sup>p</sup>data: one says that <sup>u</sup> *transports* <sup>p</sup><sup>0</sup> to <sup>p</sup>data. For every value of <sup>t</sup> between <sup>0</sup> and <sup>1</sup>, the law of <sup>x</sup>(t) defines a *probability path*, denoted <sup>p</sup>(·|t) that progressively transforms p<sup>0</sup> to pdata. If one knows the velocity field u, new samples can then be generated by sampling x<sup>0</sup> from p0, solving the ordinary differential equation, and using x(1) as the generated point. In conditional flow matching, finding such a velocity field u is achieved in the following way.

- (i) First, define a conditioning variable <sup>z</sup> independent of <sup>t</sup>, *e.g.,* <sup>z</sup> <sup>=</sup> <sup>x</sup><sup>1</sup> ∼ <sup>p</sup>data,
- (ii) Then, chose a conditional probability path <sup>p</sup>(·|z, t), *e.g.,* <sup>p</sup>(·|<sup>z</sup> <sup>=</sup> <sup>x</sup>1, t) = N (tx1,(1 − <sup>t</sup>) 2 Id).

Through the continuity equation [\(Lipman et al.,](#page-11-5) [2024,](#page-11-5) Sec. 3.5), the choice [\(ii\)](#page-1-2) of the conditional probability path <sup>p</sup>(·|z, t) defines a conditional velocity field <sup>u</sup> cond(x, z, t). With the choices [\(i\)](#page-1-3) and [\(ii\),](#page-1-2) the conditional velocity field writes

$$u^{\text{cond}}(x, z = x_1, t) = \frac{x_1 - x}{1 - t} . \quad (2)$$

the choice p<sup>0</sup> = N (0,Id) is made for simplicity; more generic choices are possible and the reader can refer to [Lipman et al.](#page-11-5) [\(2024\)](#page-11-5); [Gagneux et al.](#page-10-6) [\(2025\)](#page-10-6); [Gao et al.](#page-10-7) [\(2025\)](#page-10-7) for deeper introductions to flow matching.

The choice [\(ii\)](#page-1-2) of the conditional probability paths <sup>p</sup>(·|<sup>z</sup> <sup>=</sup> <sup>x</sup>1, t) fully defines a probability path <sup>p</sup>(·|t) (by marginalization against z) and thus defines an *optimal velocity field* u ⋆ (through the continuity equation), that transports p<sup>0</sup> to pdata [\(Lipman et al.,](#page-11-1) [2023,](#page-11-1) Thm. 1)

$$u^*(x, t) = \mathbb{E}_{z|x, t} u^{\text{cond}}(x, z, t) . \quad (3)$$

Hence, the optimal velocity u ⋆ could be approximated by a neural network u<sup>θ</sup> : R <sup>d</sup> × [0, 1] → R d with parameters θ by minimizing

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}([0,1])} \|u_\theta(x_t, t) - u^*(x_t, t)\|^2 \quad (4)$$

However, u ⋆ is usually (believed) intractable, as a remedy, [Lipman et al.](#page-11-1) [\(2023,](#page-11-1) Thm. 2) showed that LFM(θ) is equal, up to a constant, to the conditional flow matching loss. With the choices [\(i\)](#page-1-3) and [\(ii\)](#page-1-2) made above, the conditional flow matching loss reads

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{\substack{x_1 \sim p_{\text{data}} \\ t \sim \mathcal{U}([0,1])}} \left\| u_\theta(x_t, t) - \underbrace{u^{\text{cond}}(x_t, z = x_1, t)}_{= \frac{x_1 - x_t}{1-t} = x_1 - x_0} \right\|^2, \quad (5)$$

where <sup>x</sup><sup>t</sup> := (1 − <sup>t</sup>)x<sup>0</sup> <sup>+</sup> tx1. The objective LCFM is easy to approximate, since it is easy to sample from <sup>p</sup><sup>0</sup> <sup>=</sup> N (0,Id) and U([0, 1]); sampling from <sup>p</sup>data is approximated by sampling from pˆdata := <sup>1</sup> n P<sup>n</sup> <sup>i</sup>=1 δx(i) . Although it seems natural, replacing pdata by pˆdata in [\(5\)](#page-2-1) has a very important consequence: it makes the minimizer uˆ <sup>⋆</sup> of LFM available in closed-form, which we recall below.

Proposition 1 (Closed-form Formula of the Optimal Velocity). *When* pdata *is replaced by* pˆdata*, with the previous choices [\(i\)](#page-1-3) and [\(ii\),](#page-1-2) the optimal velocity field* uˆ ⋆ *in* [\(3\)](#page-2-2) *has a closed-form formula:*

$$\hat{u}^*(x, t) = \sum_{i=1}^n \lambda_i(x, t) \frac{x^{(i)} - x}{1 - t} , \quad (6)$$

*with* <sup>λ</sup>(x, t) = softmax((− ∥x−tx(j)∥ 2 2(1−t) <sup>2</sup> )<sup>j</sup>=1,...,n) ∈ R n.

The notation uˆ ⋆ emphasizes the velocity field is optimal for the *empirical* probability distribution pˆdata, not the true one pdata. Since u cond(x, z = x (i) , t) ∝ <sup>x</sup> (i) − <sup>x</sup>, the optimal velocity field uˆ ⋆ is a weighted average of the n different directions x (i) − <sup>x</sup>. Note that the closed-form formula in Equation [\(6\)](#page-2-3) can be found in various previous works, *e.g.,* [Kamb and Ganguli](#page-11-4) [\(2025,](#page-11-4) Eq. 3), [Biroli](#page-10-8) [et al.](#page-10-8) [\(2024\)](#page-10-8), [Gao and Li](#page-10-9) [\(2024\)](#page-10-9), [Li et al.](#page-11-6) [\(2024\)](#page-11-6) or [Scarvelis et al.](#page-12-11) [\(2025\)](#page-12-11), and can be generalized to other choices of continuous distribution p<sup>0</sup> (*e.g.,* the uniform distribution, see Appendix [A.1\)](#page-20-0).

From Equation [\(6\)](#page-2-3), as <sup>t</sup> → <sup>1</sup>, the velocity field <sup>u</sup><sup>ˆ</sup> <sup>⋆</sup> diverges at any point x that does not coincide with one of the training samples x (i) , and it points in the direction of the nearest x (i) . This creates a paradox: solving the ordinary differential equation [\(1\)](#page-1-4) with the velocity field uˆ ⋆ can only produce training samples x (i) (see [Gao and Li](#page-10-9) [2024,](#page-10-9) Thm. 4.6 for a formal proof). Therefore, in practice, exactly minimizing the conditional flow matching loss would result in u<sup>θ</sup> = ˆu ⋆ , meaning the model memorizes the training data and fails to generalize. This naturally yields the following question:

*How can flow matching generalize if the optimal velocity field only generates training samples?*

## 3 Investigating the key sources of generalization

In this section, we investigate the key sources of flow matching generalization using the closed-form formula of its velocity field. First in Section [3.1](#page-3-0) we challenge the claim that generalization stems from the stochastic approximation u cond of the optimal velocity field uˆ ⋆ . Then, in Section [3.2](#page-4-0) we show that generalization arises when u<sup>θ</sup> fails to approximate the perfect velocity uˆ ⋆ . Interestingly, the target velocity estimation particularly fails at two critical time intervals. Section [3.3](#page-5-0) shows that one of these critical times is particularly important for generalization.

![](_page_3_Figure_0.jpeg)

Figure 1: We challenge the hypothesis that target stochasticity plays a major role in flow matching generalization. In Figure [1a,](#page-3-1) the histograms of the cosine similarities between uˆ ⋆ ((1 − <sup>t</sup>)x<sup>0</sup> <sup>+</sup> tx1, t) and u cond((1 − <sup>t</sup>)x<sup>0</sup> <sup>+</sup> tx1, z <sup>=</sup> <sup>x</sup>1, t) = <sup>x</sup><sup>1</sup> − <sup>x</sup><sup>0</sup> are displayed for various time values <sup>t</sup> and two datasets. *For real, high-dimensional data, non-stochasticity arises very early* (before t = 0.2 for CIFAR-10 with dimension (3, 32, 32)). Figure [1c](#page-3-1) displays the alignment between uˆ ⋆ and u cond over time for varying image dimensions d on Imagenette.

#### 3.1 Target stochasticity is not what you need

One recent hypothesis is that generalization arises from the fact that the regression target u cond of conditional flow matching is only a stochastic estimate of uˆ ⋆ . The fact that the target regression objective only equals the true objective on average is referred to by [Vastola](#page-12-10) [\(2025\)](#page-12-10) as "generalization through variance". To challenge this assumption, we leverage Proposition [1,](#page-2-4) which states that the optimal velocity field uˆ ⋆ (x, t) is a weighted sum of the n values of u cond(x, t, z = x (i) ) = <sup>x</sup> (i)−x 1−t , for <sup>i</sup> ∈ [n], and show that, after a *small time value* <sup>t</sup>, this average is in practice equal to a single value in the expectation (see Figures [1a](#page-3-1) and [1b\)](#page-3-1).

Comments on Figure [1a](#page-3-1). To produce Figure [1a,](#page-3-1) we sample <sup>256</sup> pairs (x0, x1) from <sup>p</sup><sup>0</sup> × <sup>p</sup>ˆdata. For each value of t, we compute the cosine similarity between the optimal velocity field uˆ ⋆ ((1 − t)x<sup>0</sup> + tx1, t) and the conditional target u cond((1 − <sup>t</sup>)x<sup>0</sup> <sup>+</sup> tx1, z <sup>=</sup> <sup>x</sup>1, t) = <sup>x</sup><sup>1</sup> − <sup>x</sup>0. The resulting similarities are aggregated and shown as histograms. The top row displays the results for the two-moons toy dataset (d = 2), and the bottom row displays the results for the CIFAR-10 dataset [\(Krizhevsky and Hinton](#page-11-7) [2009,](#page-11-7) d = 3072); n = 50k for both. As t increases, the histograms become increasingly concentrated around 1, indicating that uˆ ⋆ aligns closely with a single conditional vector u cond. From Equation [\(6\)](#page-2-3), this corresponds to a collapse towards 0 of all but one of the softmax weights λi(xt, t). This time corresponds to the collapse time studied by [Biroli et al.](#page-10-8) [\(2024\)](#page-10-8) for diffusion; we discuss the connection in the related works (Section [5\)](#page-8-0). On the two-moons toy dataset, this transition occurs for intermediate-to-large values of t, echoing the observations made in low-dimensional settings by [Vastola](#page-12-10) [\(2025,](#page-12-10) Figure 1). In contrast, for high-dimensional real datasets, uˆ ⋆ (x, t) aligns with a single conditional velocity field x (i)−<sup>x</sup>, even at early time steps, suggesting that the non-stochastic regime dominates most of the generative process. This key difference between lowand high-dimensional data suggests that the transition time between the stochastic and non-stochastic regimes is strongly influenced by the dimensionality of the data.

![](_page_4_Figure_0.jpeg)

Figure 2: Failure to learn the optimal velocity field, CIFAR-10. *Left*: The leftmost figure represents the average error between the optimal empirical velocity field uˆ ⋆ and the learned velocity u<sup>θ</sup> for multiple values of time t. *Middle*: The middle figure displays the FID-10k computed on the test dataset, using the DINOv2 embedding. *Right*: The rightmost figure displays the average distance between the generated samples and their closest image from the training set – for reference, the horizontal dashed line indicates the mean distance between an image of CIFAR-10 train and its nearest neighbor in the dataset. All the quantities are computed/learned on a varying number of training samples (10 to 10<sup>4</sup> ) of the CIFAR-10 dataset.

Comments on Figure [1c](#page-3-1). To further illustrate the strong impact of dimensionality, Figure [1c](#page-3-1) reports the proportion of samples x<sup>t</sup> (from a batch of 256) for which the cosine similarity between uˆ ⋆ and u cond ∝ <sup>x</sup> (i) − <sup>x</sup> exceeds <sup>0</sup>.9, as a function of time <sup>t</sup>. This analysis is performed across multiple spatial resolutions of the Imagenette dataset [\(Howard,](#page-11-8) [2019\)](#page-11-8), obtaining dim × dim images by spatial subsampling. Figure [1c](#page-3-1) reveals a sharp transition: as the dimensionality increases, the proportion of high-cosine matches rapidly converges to 100%. A practical implication of this behavior is that, for sufficiently large <sup>t</sup>, if <sup>x</sup><sup>0</sup> ∼ <sup>p</sup><sup>0</sup> and <sup>x</sup> (i) ∼ <sup>p</sup>ˆdata, then <sup>u</sup><sup>ˆ</sup> ⋆ ((1 − <sup>t</sup>)x<sup>0</sup> <sup>+</sup> tx(i) , t) is approximately proportional to x (i) − <sup>x</sup>. Consequently, regressing on <sup>x</sup> (i) or on the conditional velocity <sup>x</sup><sup>1</sup> − <sup>x</sup><sup>0</sup> becomes effectively equivalent. Section [4](#page-6-0) investigates how to learn regressing against optimal velocity field uˆ ⋆ , and empirically shows similar results between stochastic and non-stochastic targets.

The regime where flow matching matches stochasticity is mostly concentrated on a very short time interval, for small values of t. We hypothesize that the phenomenon observed here on the optimal velocity field uˆ <sup>⋆</sup> has major implications on the *learned* flow matching model uθ, which we further inspect in the next section.

### 3.2 Failure to learn the optimal velocity field

This subsection investigates how well the learned velocity field u<sup>θ</sup> approximates the optimal/ideal velocity field uˆ ⋆ , and how the quality of this approximation correlates with generalization. To do so, we propose the following experiment.

Set up of Figure [2](#page-4-1). To build Figure [2,](#page-4-1) we subsampled the CIFAR-10 dataset from 10 to 10<sup>4</sup> samples. For each size, we trained a flow matching model using a standard 34 million-parameter U-Net (see Appendix [D](#page-26-0) for details). Following [Kadkhodaie et al.](#page-11-3) [\(2024\)](#page-11-3), the number of parameters of the network u<sup>θ</sup> remains fixed across dataset sizes. Importantly, the optimal velocity field uˆ ⋆ itself depends on the dataset size: as the number of samples increases, the complexity of uˆ ⋆ also grows. Thus, we expect the network u<sup>θ</sup> to accurately approximate the optimal velocity field uˆ ⋆ for smaller dataset sizes.

Comments on Figure [2](#page-4-1). The leftmost plot shows the average training error

$$\mathbb{E}_{\substack{x_0 \sim p_0 \\ x_1 \sim \hat{p}_{\text{data}}}} \|u_\theta(x_t, t) - \hat{u}^*(x_t, t)\|^2, \quad \text{where} \quad x_t := (1-t)x_0 + tx_1,$$

between the learned velocity u<sup>θ</sup> and the optimal empirical velocity field uˆ ⋆ , evaluated across multiple time values t and dataset sizes. With only 10 samples (darkest curve), the network u<sup>θ</sup> closely approximates uˆ ⋆ . As the dataset size increases, the complexity of uˆ <sup>⋆</sup> grows, and the approximation by u<sup>θ</sup> becomes less accurate. In particular, the approximation fails at two specific time intervals: around

![](_page_5_Figure_0.jpeg)

Figure 3: Generalization occurs at small times on CIFAR-10 (left) and CelebA 64 (right). *Top*: Generalization (distance between generated samples and training data) of hybrid models that follow uˆ <sup>⋆</sup> on [0, τ ], then u<sup>θ</sup> on [τ, 1]. The four colored curves correspond to four specific x0, the black dashed curve is the mean distance over the 256 generated images. *Bottom*: visualization of generated images for the four different starting noises and various values of τ (the background color matching the curve in the top figure). *Following* uˆ <sup>⋆</sup> *until* <sup>τ</sup> ≥ <sup>0</sup>.<sup>3</sup> *yields a model that is not able to generalize*.

<sup>t</sup> ≈ <sup>0</sup>.<sup>15</sup> and near <sup>t</sup> = 1. The failure near <sup>t</sup> = 1 is expected, as <sup>u</sup><sup>ˆ</sup> <sup>⋆</sup> becomes non-Lipschitz at t = 1. Interestingly, the early-time failure at <sup>t</sup> ≈ <sup>0</sup>.<sup>15</sup> corresponds to the regime where <sup>u</sup><sup>ˆ</sup> ⋆ and u cond start to correlate (see Figure [1a](#page-3-1) in Section [3.1\)](#page-3-0). The middle plot of Figure [2](#page-4-1) reports the FID-10k, computed on the test set in the DINOv2 embedding space [\(Oquab et al.,](#page-11-9) [2024\)](#page-11-9), for various dataset sizes. For a small dataset (*e.g.,* #samples = 10), u<sup>θ</sup> approximates uˆ <sup>⋆</sup> well but does not generalize – the test FID exceeds 10<sup>3</sup> . As the dataset size increases (<sup>1000</sup> ≤ #samples ≤ <sup>3000</sup>), the approximation <sup>u</sup><sup>θ</sup> becomes less accurate. Despite this, the model achieves lower FID scores on the test set but still memorizes the training data. The rightmost plot of Figure [2](#page-4-1) illustrates this memorization by showing the average distance between each generated sample and its nearest neighbor in the training set. For larger datasets (#samples ≥ <sup>3000</sup>), this distance increases substantially, indicating that the model generalizes better. Overall, Figure [2](#page-4-1) also suggests that the FID metric can be misleading, even when computed on the test set. For example, the model trained with 1000 samples has a low test FID but memorizes training examples.

Figure [2](#page-4-1) confirms that generalization arises when the network u<sup>θ</sup> fails to estimate the optimal velocity field uˆ ⋆ , and that this failure occurs at two specific time intervals. In Section [3.3,](#page-5-0) we investigate which of these two intervals is responsible for driving generalization.

#### 3.3 When does generalization arise?

To investigate whether the failure to approximate uˆ <sup>⋆</sup> matters the most at small or large values of t, we carry out the following experiment.

Set up of Figure [3](#page-5-1). We first learn a velocity field u<sup>θ</sup> using standard conditional flow matching (see Appendix [D\)](#page-26-0), then we construct a hybrid model: we define a piecewise trajectory where the flow is governed by the optimal velocity field uˆ ⋆ for times <sup>t</sup> ∈ [0, τ ], and by the learned velocity field <sup>u</sup><sup>θ</sup> for times <sup>t</sup> ∈ [τ, 1], for a given threshold parameter <sup>τ</sup> ∈ [0, 1]. For the extreme case <sup>τ</sup> = 1, the full trajectory follows uˆ ⋆ , and samples exactly match training data points. Conversely, when τ = 0, the entire trajectory is governed by uθ, yielding novel samples. Intermediate values of τ produce a mixture of both behaviors, which we interpret as reflecting varying degrees of generalization. To assess generalization, we measure the distance of generated samples to the dataset using the LPIPS

metric [\(Zhang et al.,](#page-12-12) [2018\)](#page-12-12), which computes the feature distance between two images via some pretrained classification network. We define the distance of a generated sample <sup>x</sup> to a dataset D <sup>=</sup> {x (1), . . . , x(n)} as dist(x, D) = minx(i)∈D LPIPS(x, x(i) ). We fix a random batch of 256 pure noise images from p0. Then, for various threshold values τ , we generate 256 images with the hybrid model, always starting from this batch. Finally, we measure the creativity of the hybrid model as the mean of the aforementioned LPIPS distances between the 256 generated samples and the dataset.

Comments on Figure [3](#page-5-1). The top row displays the LPIPS distances as τ varies, on the CIFAR-10 (left) and CelebA - <sup>64</sup> × <sup>64</sup> (right) datasets. For <sup>τ</sup> ≤ <sup>0</sup>.2, the hybrid model remains as creative as uθ, despite following uˆ ⋆ in the first steps. For τ > 0.2, the LPIPS distance starts dropping. On the displayed generated samples (bottom rows), we in fact see that as soon as <sup>τ</sup> ≥ <sup>0</sup>.4, the sample generated by the hybrid model is almost the same as the one obtained with uˆ ⋆ (τ = 1). This means *the final image is already determined at* t = 0.4, and despite the generalization capacity of the learned velocity field <sup>u</sup>θ, following it only after <sup>t</sup> ≥ <sup>0</sup>.<sup>4</sup> is not enough to create a new image: *generalization occurs early and cannot fully be explained by the failure to correctly approximate* u <sup>⋆</sup> *at large* t*.*

Although we have shown that the stochastic phase was limited to small values of t in real-data settings, we have not yet definitively ruled it out as the cause of generalization. In the following Section [4,](#page-6-0) we introduce a learning procedure designed to address this question directly.

## 4 Learning with the closed-form formula

In this section, in order to discard the impact of stochastic target on the generalization, we propose to directly regress against the closed-form formula in Equation [\(6\)](#page-2-3).

#### 4.1 Empirical flow matching

Regressing against the closed-form uˆ ⋆ , defined in Equation [\(6\)](#page-2-3), at a point (xt, t) requires computing a weighted sum of the conditional velocity fields over *all* the n training points x (i) . For a dataset of <sup>n</sup> samples of size <sup>d</sup>, and a batch of size |B|, computing the weights of the exact closed-form formula uˆ ⋆ (x, t) of flow matching requires O(<sup>n</sup> × |B| × <sup>d</sup>). These computations are prohibitive since they must be performed for each batch. One natural idea is to estimate the closed-form formula uˆ ⋆ (Equation [\(6\)](#page-2-3)), by a Monte Carlo approximation (Equation [\(8\)](#page-6-1)), using <sup>M</sup> ≤ <sup>n</sup> samples <sup>b</sup> (1), . . . , b(M) :

$$\mathcal{L}_{\text{EFM}}(\theta) = \mathbb{E} \frac{x_0 \sim p_0}{x_1 \sim \hat{p}_{\text{data}}} \frac{\|u_\theta(x_t, t) - \hat{u}_M^*(x_t, t)\|^2}{t \sim \mathcal{U}[0, 1]} b^{(2)}, \dots, b^{(M)} \sim \hat{p}_{\text{data}} \quad (7)$$

with <sup>x</sup><sup>t</sup> = (1 − <sup>t</sup>)x<sup>0</sup> <sup>+</sup> tx1, <sup>b</sup> (1) := x1, and

$$\hat{u}_M^*(x, t) = \sum_{j=1}^M \lambda(x, t) \frac{b^{(j)} - x}{1 - t} \quad , \quad \lambda(x, t) = \text{softmax} \left( \left( -\frac{\|x - tx^{(l)}\|^2}{2(1 - t)^2} \right)_{l=1, \dots, n} \right) \quad . \quad (8)$$

The formulation in Equation [\(7\)](#page-6-2) may appear naive at first glance. Still, it hinges on a crucial trick: the Monte Carlo estimate is computed using a batch that systematically includes the point x1, that generated the current xt. If instead b (1) were sampled independently from pˆdata, this could introduce a sampling bias (see [Ryzhakov et al.](#page-12-13) [2024,](#page-12-13) Appendix [B,](#page-21-0) and the corresponding OpenReview comments[<sup>3</sup>](#page-6-3) for an in-depth discussion). Proposition [2](#page-6-4) shows that the estimate uˆ ⋆ <sup>M</sup> is unbiased and has lower variance than the standard conditional flow matching target.

Proposition 2. *We denote the conditional probability distribution* p(z = x (i) | x, t) *over* {<sup>x</sup> (i)} n i=1 *by* <sup>p</sup>ˆdata(<sup>z</sup> | x, t)*. With no constraints on the learned velocity field* <sup>u</sup>θ*,*

*i) The minimizer of Equation* [\(7\)](#page-6-2) *writes, for all* (x, t)

$$\mathbb{E}_{b^{(1)} \sim \hat{p}_{\text{data}}(\cdot | x, t)} [\hat{u}_M^*(x, t)] \quad (9)$$

$$b^{(2)}, \dots, b^{(M)} \sim \hat{p}_{\text{data}}$$

<sup>3</sup><https://openreview.net/forum?id=XYDMAckWMa>

*ii) In addition, for all* (x, t)*, the minimizer of Equation* [\(7\)](#page-6-2) *equals the optimal velocity field, i.e.,*

$$\mathbb{E}_{b^{(1)} \sim \hat{p}_{\text{data}}(\cdot | x, t)} [\hat{u}_M^*(x, t)] = \hat{u}^*(x, t) \quad (10)$$

*iii) The conditional variance of the estimator* uˆ ⋆ <sup>M</sup> *is smaller than the usual conditional variance:*

$$\text{Var}_{b^{(1)} \sim \hat{p}_{\text{data}}(\cdot|x,t)} [\hat{u}_M^*(x,t)] \leq \text{Var}_{b^{(1)} \sim \hat{p}_{\text{data}}(\cdot|x,t)} [u^{\text{cond}}(x,b^{(1)},t)]. \quad (11)$$

$$b^{(2)}, \dots, b^{(M)} \sim \hat{p}_{\text{data}}$$

The proof of Proposition [2](#page-6-4) is provided in Appendix [B.3.](#page-22-0) The estimator uˆ ⋆ <sup>M</sup> of the optimal field uˆ ⋆ is closely related to self-normalized importance sampling (see Appendix [B.2](#page-21-1) and [Owen](#page-11-10) [2013,](#page-11-10) Chap. 9.2), as well as to Rao-Blackwellized estimators [\(Casella and Robert,](#page-10-10) [1996;](#page-10-10) [Cardoso et al.,](#page-10-11) [2022\)](#page-10-11). As discussed in [Ryzhakov et al.](#page-12-13) [\(2024\)](#page-12-13), self-normalized importance sampling estimators of uˆ ⋆ are generally biased, in the sense that: E<sup>b</sup> (1),...,b(M)∼pˆdata uˆ ⋆ <sup>M</sup>(xt, t) ̸= ˆ<sup>u</sup> ⋆ (xt, t) . A key insight is that our estimator includes b (1) ∼ <sup>p</sup>ˆdata(· | <sup>x</sup>t, t), which leads to the main result of Proposition [2.](#page-6-4) In Section [4.2,](#page-7-0) we demonstrate that Algorithm [2,](#page-7-1) designed to solve Equation [\(7\)](#page-6-2), yields consistent improvements on high-dimensional datasets such as CIFAR-10 and CelebA. Additional details on the unbiasedness of LEFM can be found in the supplementary material (Appendix [B\)](#page-21-0). From a computational perspective, despite requiring M additional samples, Algorithm [2](#page-7-1) remains significantly more efficient than increasing the batch size by a factor of M: the M samples are merely averaged (with weights), while the backpropagation remains identical to that of Algorithm [1.](#page-7-2)

| Algorithm 1 Vanilla Flow Matching   |       |                                                      |
|-------------------------------------|-------|------------------------------------------------------|
| for k in 1 , , n iter do            |       |                                                      |
| t ∼ U ([0 , 1])                     |       |                                                      |
| x 0 ∼ N (0 , Id) , x 1 ∼ p ˆ data , |       |                                                      |
| x t = (1 − t ) x 0 + tx 1           |       |                                                      |
| cond ( x t , t ) = x 1 − x t        |       |                                                      |
| 1 − t                               |       |                                                      |
| = x 1                               | −     | x 0                                                  |
| L ( θ ) =                           |       |                                                      |
| u θ ( x t , t ) − u                 |       |                                                      |
| cond ( x                            | t , t | )                                                    |
| Compute ∇L ( θ ) and update         | θ     |                                                      |
| return u θ                          |       |                                                      |
|                                     |       | Algorithm 2 Empirical Flow Matching                  |
|                                     |       | param : M // Number of samples in the empirical mean |
|                                     |       | for k in 1 , , n iter do                             |
|                                     |       | x 0 ∼ N (0 , Id) , x 1 ∼ p ˆ data , t ∼ U ([0 , 1])  |
|                                     |       | x t = (1 − t ) x 0 + tx 1                            |
|                                     |       | (1) = x 1                                            |
|                                     |       | ∀ j ∈ J 2 ,M K , b                                   |
|                                     |       | ( j ) ∼ p ˆ data // Samples from p ˆ data            |
|                                     |       | u ˆ                                                  |
|                                     |       | M ( x t , t ) = P M                                  |
|                                     |       | j =1                                                 |
|                                     |       | ( j ) − x t                                          |
|                                     |       | 1 − t                                                |
|                                     |       | softmax                                             |
|                                     |       | ∥ x t − t b ∥                                        |
|                                     |       | 2(1 − t )                                            |
|                                     |       | L ( θ ) = ∥ u θ ( x t , t ) − u ˆ                    |
|                                     |       | M ( x t , t ) ∥                                      |
|                                     |       | Compute ∇L ( θ ) and update θ                        |
|                                     |       | return u θ                                           |

#### 4.2 Experiments

We now learn with empirical flow matching (EFM, Equation [\(7\)](#page-6-2) and Algorithm [2\)](#page-7-1) in practical high-dimensional settings. Our goal with this empirical investigation is first to observe if regressing against a more deterministic target leads to performance improvement/degradation.

Datasets and Models. We perform experiments on the image datasets CIFAR-10 [\(Krizhevsky and](#page-11-7) [Hinton,](#page-11-7) [2009\)](#page-11-7) and CelebA <sup>64</sup> × <sup>64</sup> [\(Liu et al.,](#page-11-11) [2015\)](#page-11-11). For the experiments, we compare vanilla conditional flow matching [\(Lipman et al.,](#page-11-1) [2023;](#page-11-1) [Liu et al.,](#page-11-2) [2023;](#page-11-2) [Albergo and Vanden-Eijnden,](#page-10-0) [2023\)](#page-10-0), optimal transport flow matching [\(Pooladian et al.,](#page-11-12) [2023;](#page-11-12) [Tong et al.,](#page-12-14) [2024\)](#page-12-14), and the empirical flow matching in Algorithm [2,](#page-7-1) for multiple numbers of samples M to estimate the empirical mean. Training details are in Appendix [D.](#page-26-0)

Metrics. To assess generalization performance, we use the standard Fréchet Inception Distance [\(Heusel et al.,](#page-10-12) [2017\)](#page-10-12) with Inception-V3 [\(Szegedy et al.,](#page-12-15) [2016\)](#page-12-15) but we also follow the recommendation of [Stein et al.](#page-12-16) [\(2023\)](#page-12-16) using the DINOv2 embedding [\(Oquab et al.,](#page-11-13) [2023\)](#page-11-13), which is known to a more expressive and discriminative embedding, that leads to a less biased evaluation. We also measure the FID between the generated and the train and test sets, rather than only on the training set, as is often done in generative modeling benchmarks. On Figure [2,](#page-4-1) we also displayed a memorization metric that would detect a pure copy of the training set. Overall, defining and quantifying the generalization ability of generative models is overall a challenging task: train and test FID are known to be imperfect [\(Stein et al.,](#page-12-16) [2023;](#page-12-16) [Jiralerspong et al.,](#page-11-14) [2023;](#page-11-14) [Parmar et al.,](#page-11-15) [2022\)](#page-11-15), yet no superior competitor has emerged.

![](_page_8_Figure_0.jpeg)

Figure 4: FID computed on the training set (50k) and the test set (10k) using multiple embeddings, Inception and DINOv2. Regressing against a more deterministic target (EFM - 128, 256, 1000) does not yield performance decreases. On the contrary, the more deterministic the target, the better the performance.

Comments on Figure [4](#page-8-1). Figure [4](#page-8-1) compares vanilla flow matching, OTCFM, and the empirical flow matching (EFM, Algorithm [2\)](#page-7-1) approaches using various numbers of samples to estimate the empirical mean, <sup>M</sup> ∈ {128, <sup>256</sup>, <sup>1000</sup>}. First, we observe that learning with a more deterministic target does not degrade either training or testing performance, across both types of embeddings. On the contrary, we consistently observe modest but steady improvements as stochasticity is reduced. For both CIFAR-10 and CelebA, increasing the number of samples M used to compute the empirical mean—*i.e.,* , making the targets less stochastic—leads to more stable improvements. It is worth noting that Algorithm [<sup>2</sup>](#page-7-1) has a computational complexity of O(<sup>M</sup> × |B| × <sup>d</sup>), where |B| is the batch size, M is the number of samples used to estimate the empirical mean, and d is the sample dimension. In our experiments, choosing <sup>M</sup> <sup>=</sup> |B| = 128 yielded a modest time overhead. For empirical flow matching, we experimented with several values beyond M = 1000 (*e.g.,* M = 2000, M = 5000). The results were nearly identical to those obtained with M = 1000, with curves being visually indistinguishable. Therefore, we chose not to report results for <sup>M</sup> ≥ <sup>1000</sup>.

## 5 Related work

The existing literature related to our study can be roughly divided into three approaches: leveraging the closed-form, studies on the memorization vs generalization, and characterization of the different phases of the generating dynamics.

Leveraging the closed-form. Proposition [1](#page-2-4) has been leveraged in several ways. The closest existing work is by [Ryzhakov et al.](#page-12-13) [\(2024\)](#page-12-13), who propose to regress against uˆ ⋆ as we do in Section [4.](#page-6-0) Nevertheless, their motivation is that reducing the variance of the velocity field estimation makes learning more accurate: as explained in Section [3.1,](#page-3-0) we argue this claim rests on misleading 2Dbased intuitions (*e.g.,* Figure 1, challenged by Section [3.1\)](#page-3-0). The idea of regressing against a more deterministic target (as Proposition [2](#page-6-4) shows) derived from the optimal closed-form velocity field has also been empirically explored for diffusion models [\(Xu et al.,](#page-12-17) [2023\)](#page-12-17). [Scarvelis et al.](#page-12-11) [\(2025\)](#page-12-11) bypass training, and suggest using a smoothed version of uˆ ⋆ to generate novel samples. In a work specific to images and convolutional neural networks, [Kamb and Ganguli](#page-11-4) [\(2025\)](#page-11-4) suggested that flow matching indeed ends up learning an optimal velocity, but that instead of memorizing training samples, the velocity memorizes a combination of all possible patches in an image and across the images. They show remarkable agreement between their theory and the trajectories followed by learned vector fields, but their work is limited to convolutional architectures, and was recently extended to a larger class of architectures [\(Lukoianov et al.,](#page-11-16) [2025\)](#page-11-16).

Memorization and reasons for generalization. [Kadkhodaie et al.](#page-11-3) [\(2024\)](#page-11-3) directly relates the transition from memorization to generalization to the size of the training dataset, and proposes a geometric interpretation. We provide a complementary experiment in Section [3.2,](#page-4-0) quantifying how much the network fails to estimate the optimal velocity field. [Gu et al.](#page-10-13) [\(2025\)](#page-10-13) provide a detailed experimental investigation into the potential causes for generalization, primarily based on the characteristics of the dataset and choices for training and model. [Vastola](#page-12-10) [\(2025\)](#page-12-10) explores different factors of generalization in the case of diffusion, with a special focus on the stochasticity of the target objective in the learning problem. Through a physic-based modeling of the generative dynamics, they study the covariance matrix of the noisy estimation of the exact score. In our work, we believe that we have shown that this claim was not valid for real high-dimensional data. [Niedoba et al.](#page-11-17) [\(2025\)](#page-11-17) study the poor approximation of the exact score by the learned models: like [Kamb and Ganguli](#page-11-4) [\(2025\)](#page-11-4), they suggest that the generalization of the learned models comes from memorization of many patches in the training data.

Temporal regimes. [Biroli et al.](#page-10-8) [\(2024\)](#page-10-8); [Sclocchi et al.](#page-12-18) [\(2025\)](#page-12-18) provide an analysis of the exact score, the counterpart of the exact velocity field for diffusion. For a multimodal target distribution, the authors identify three phases (we keep the convention that t = 0 is noise and t = 1 is target): for t < t1, all trajectories are indistinguishable; for t<sup>1</sup> < t < t2, trajectories converging to different modes separate; for t > t2, trajectories all point to the training dataset. In the case of Gaussians mixtures target, they highlight the dependency of t<sup>2</sup> in the dimension and the number of samples, in O ((log <sup>n</sup>)/d), meaning that the first phases are observable only if the number of training points is exponential in the dimension. The methodology they adopt to validate the existence of such t<sup>2</sup> on real data relies on the stochasticity of the backward generative process, which does not hold in the case of flow matching. Our experiments on *learned* flow matching models allow us to take this theoretical study on memorization and temporal behaviors of generative processes a step further.

## 6 Conclusion, limitations and broader impact

Conclusion. By challenging the assumption that stochasticity in the loss function is a key driver of generalization, our findings help clarify the role of approximation of the exact velocity field in flow matching models. Beyond the different temporal phases in the generation process that we have identified, we expect further results to be obtained by uncovering new properties of the true velocity field.

Limitation. Our work is mainly empirical, with a focus on *learned* models, but did not precisely characterize the learned velocity field, in particular, how it behaves outside the trajectories defined by the optimal velocity. Leveraging existing work on the inductive biases of the architectures at hand seems like a promising venue. Another limitation is that we did not investigate the interaction between the architectural inductive bias, and optimization procedures: this is a very challenging, but active area of research [\(Boursier and Flammarion,](#page-10-14) [2025;](#page-10-14) [Bonnaire et al.,](#page-10-15) [2025;](#page-10-15) [Favero et al.,](#page-10-16) [2025\)](#page-10-16).

Broader impact. We hope that identifying the key factors of generalization will lead to improved training efficiency. However, generative models also raise concerns related to misinformation (notably deepfakes), data privacy, and potential misuse in generating synthetic but realistic content.

## 7 Acknowledgments

The authors thank the Blaise Pascal Center for its computational support, using the SIDUS [\(Quemener](#page-11-18) [and Corvellec,](#page-11-18) [2013\)](#page-11-18) solution.

## References


[1] M. S. Albergo and E. Vanden-Eijnden. Building normalizing flows with stochastic interpolants. *ICLR*, 2023.

[2] G. Biroli, T. Bonnaire, V. de Bortoli, and M. Mézard. Dynamical regimes of diffusion models. *Nature Communications*, 15(1):9957, 2024.

[3] T. Bonnaire, R. Urfin, G. Biroli, and M. Mézard. Why diffusion models don't memorize: The role of implicit dynamical regularization in training. *arXiv preprint arXiv:2505.17638*, 2025.

[4] Z. Borsos, R. Marinier, D. Vincent, E. Kharitonov, O. Pietquin, M. Sharifi, D. Roblek, O. Teboul,
  - D. Grangier, M. Tagliasacchi, et al. Audiolm: a language modeling approach to audio generation. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2023.

[5] E. Boursier and N. Flammarion. Simplicity bias and optimization threshold in two-layer ReLu networks. *ICML*, 2025.

[6] T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman, C. Ng, R. Wang, and A. Ramesh. Video generation models as world simulators. 2024. URL [https://openai.com/research/](https://openai.com/research/video-generation-models-as-world-simulators) [video-generation-models-as-world-simulators](https://openai.com/research/video-generation-models-as-world-simulators).

[7] G. Cardoso, S. Samsonov, A. Thin, E. Moulines, and J. Olsson. Br-snis: bias reduced self-normalized importance sampling. *NeurIPS*, 35:716–729, 2022.

[8] N. Carlini, J. Hayes, M. Nasr, M. Jagielski, V. Sehwag, F. Tramer, B. Balle, D. Ippolito, and E. Wallace. Extracting training data from diffusion models. In *32nd USENIX Security Symposium (USENIX Security 23)*, pages 5253–5270, 2023.

[9] G. Casella and C. P. Robert. Rao-blackwellisation of sampling schemes. *Biometrika*, 83(1):81–94, 1996.

[10] S. U. H. Dar, A. Ghanaat, J. Kahmann, I. Ayx, T. Papavassiliu, S. O. Schoenberg, and S. Engelhardt. Investigating data memorization in 3d latent diffusion models for medical image synthesis. In *International Conference on Medical Image Computing and Computer-Assisted Intervention*, pages 56–65. Springer, 2023.

[11] A. Favero, A. Sclocchi, and M. Wyart. Bigger isn't always memorizing: Early stopping overparameterized diffusion models. *arXiv preprint arXiv:2505.16959*, 2025.

[12] A. Gagneux, S. Martin, R. Emonet, Q. Bertrand, and M. Massias. A visual dive into conditional flow matching. In *The Fourth Blogpost Track at ICLR*, 2025.

[13] R. Gao, E. Hoogeboom, J. Heek, V. de Bortoli, K. Murphy, and T. Salimans. Diffusion meets flow matching: Two sides of the same coin. *ICLR Blogpost*, 2025.

[14] W. Gao and M. Li. How do flow matching models memorize and generalize in sample data subspaces? *arXiv preprint arXiv:2410.23594*, 2024.

[15] S. Gong, M. Li, J. Feng, Z. Wu, and L. Kong. Diffuseq: Sequence to sequence text generation with diffusion models. *ICLR*, 2023.

[16] X. Gu, C. Du, T. Pang, C. Li, M. Lin, and Y. Wang. On memorization in diffusion models. *TMLR*, 2025.

[17] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. *NeurIPS*, 30, 2017.

[18] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. *NeuRIPS*, 2020.

[19] J. Howard. Imagenette: A smaller subset of 10 easily classified classes from imagenet, March 2019. URL <https://github.com/fastai/imagenette>. C.-W. Huang, J. H. Lim, and A. C. Courville. A variational perspective on diffusion-based generative models and score matching. *NeurIPS*, 34:22863–22876, 2021.

[20] M. Jiralerspong, J. Bose, I. Gemp, C. Qin, Y. Bachrach, and G. Gidel. Feature likelihood divergence: evaluating the generalization of generative models using samples. *NeurIPS*, 2023.

[21] Z. Kadkhodaie, F. Guth, E. P. Simoncelli, and S. Mallat. Generalization in diffusion models arises from geometry-adaptive harmonic representations. *ICLR*, 2024.

[22] M. Kamb and S. Ganguli. An analytic theory of creativity in convolutional diffusion models. *ICML*, 2025.

[23] A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. 2009.

[24] S. Li, S. Chen, and Q. Li. A good score does not lead to a good generative model. *arXiv preprint arXiv:2401.04856*, 2024.

[25] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. *ICLR*, 2023.

[26] Y. Lipman, M. Havasi, P. Holderrieth, N. Shaul, M. Le, B. Karrer, R. T. Chen, D. Lopez-Paz,
  - H. Ben-Hamu, and I. Gat. Flow matching guide and code. *arXiv preprint arXiv:2412.06264*, 2024.

[27] X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. *ICLR*, 2023.

[28] Z. Liu, P. Luo, X. Wang, and X. Tang. Deep learning face attributes in the wild. In *Proceedings of International Conference on Computer Vision (ICCV)*, December 2015.

[29] A. Lukoianov, C. Yuan, J. Solomon, and V. Sitzmann. Locality in image diffusion models emerges from data statistics. *NeurIPS*, 2025.

[30] S. Martin, A. Gagneux, P. Hagemann, and G. Steidl. Pnp-flow: Plug-and-play image restoration with flow matching. *ICLR*, 2025.

[31] A. Q. Nichol and P. Dhariwal. Improved denoising diffusion probabilistic models. In *ICML*, pages 8162–8171. PMLR, 2021.

[32] M. Niedoba, B. Zwartsenberg, K. Murphy, and F. Wood. Towards a mechanistic explanation of diffusion model generalization. *ICML*, 2025.

[33] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza,
  - F. Massa, A. El-Nouby, et al. Dinov2: Learning robust visual features without supervision. *arXiv preprint arXiv:2304.07193*, 2023.

[34] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza,
  - F. Massa, A. El-Nouby, R. Howes, P.-Y. Huang, H. Xu, V. Sharma, S.-W. Li, W. Galuba, M. Rabbat,
  - M. Assran, N. Ballas, G. Synnaeve, I. Misra, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and
  - P. Bojanowski. Dinov2: Learning robust visual features without supervision. *TMLR*, 2024.

[35] A. B. Owen. *Monte Carlo theory, methods and examples*. [https://artowen.su.domains/](https://artowen.su.domains/mc/) [mc/](https://artowen.su.domains/mc/), 2013.

[36] G. Parmar, R. Zhang, and J.-Y. Zhu. On aliased resizing and surprising subtleties in gan evaluation. In *CVPR*, 2022. A.-A. Pooladian, H. Ben-Hamu, C. Domingo-Enrich, B. Amos, Y. Lipman, and R. T. Chen. Multisample flow matching: Straightening flows with minibatch couplings. *ICML*, 2023.

[37] E. Quemener and M. Corvellec. Sidus—the solution for extreme deduplication of an operating system. *Linux Journal*, 2013(235):3, 2013.

[38] C. P. Robert, G. Casella, and G. Casella. *Monte Carlo statistical methods*, volume 2. Springer, 1999.

[39] B. L. Ross, H. Kamkari, T. Wu, R. Hosseinzadeh, Z. Liu, G. Stein, J. C. Cresswell, and G. Loaiza-Ganem. A geometric framework for understanding memorization in generative models. *ICLR*, 2025.

[40] G. Ryzhakov, S. Pavlova, E. Sevriugov, and I. Oseledets. Explicit flow matching: On the theory of flow matching algorithms with applications. In *ICOMP*, 2024.

[41] C. Scarvelis, H. S. B. de Ocáriz, and J. Solomon. Closed-form diffusion models. *TMLR*, 2025.

[42] A. Sclocchi, A. Favero, and M. Wyart. A phase transition in diffusion models reveals the hierarchical nature of data. *PNAS*, 122(1):e2408799121, 2025.

[43] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In *ICML*, 2015.

[44] G. Somepalli, V. Singla, M. Goldblum, J. Geiping, and T. Goldstein. Understanding and mitigating copying in diffusion models. *NeurIPS*, 36:47783–47803, 2023a.

[45] G. Somepalli, V. Singla, M. Goldblum, J. Geiping, and T. Goldstein. Diffusion art or digital forgery? investigating data replication in diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 6048–6058, 2023b.

[46] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. *ICLR*, 2021. Stability AI. <https://stability.ai/stablediffusion>, 2023. Accessed: 2023-09-09.

[47] G. Stein, J. Cresswell, R. Hosseinzadeh, Y. Sui, B. Ross, V. Villecroze, Z. Liu, A. L. Caterini,
  - E. Taylor, and G. Loaiza-Ganem. Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models. *NeurIPS*, 36:3732–3784, 2023.

[48] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 2818–2826, 2016.

[49] A. Tong, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, K. Fatras, G. Wolf, and Y. Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. In *TMLR*, 2024. URL <https://openreview.net/forum?id=CD9Snc73AW>.

[50] J. J. Vastola. Generalization through variance: how noise shapes inductive biases in diffusion models. *ICLR*, 2025.

[51] R. Villegas, M. Babaeizadeh, P.-J. Kindermans, H. Moraldo, H. Zhang, M. T. Saffar, S. Castro,
  - J. Kunze, and D. Erhan. Phenaki: Variable length video generation from open domain textual descriptions. In *ICLR*, 2022.

[52] M. Xu, T. Geffner, K. Kreis, W. Nie, Y. Xu, J. Leskovec, S. Ermon, and A. Vahdat. Energy-based diffusion language models for text generation. *ICLR*, 2025.

[53] Y. Xu, S. Tong, and T. Jaakkola. Stable target field for reduced variance score estimation in diffusion models. *ICLR*, 2023.

[54] T. Yoon, J. Y. Choi, S. Kwon, and E. K. Ryu. Diffusion probabilistic models generalize when they fail to memorize. In *ICML 2023 workshop on structured probabilistic inference & generative modeling*, 2023.

[55] H. Zhang, J. Zhou, Y. Lu, M. Guo, P. Wang, L. Shen, and Q. Qu. The emergence of reproducibility and consistency in diffusion models. In *ICML*, 2024.

[56] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep features as a perceptual metric. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 586–595, 2018.
## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1–2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

- Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist",
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers.

#### 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: each claim of the abstract refers to a specific subsection of the paper, that provide empirical evidence of the claim.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

### 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: all results are encapsulated in clearly defined statements, and proofs are provided in appendix.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We provided as many details as possible in order to reproduce the results, in particular, we refer to the public implementation we used, including the specific (default) parameters used.

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
  - (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
  - (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
  - (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
  - (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

#### 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Code will be made available along with publication

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ([https://nips.cc/public/](https://nips.cc/public/guides/CodeSubmissionPolicy) [guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ([https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy) [public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We provide a specific appendix with the experimental details

- The answer NA means that the paper does not include experiments.

- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

### 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We do not report error bars, however, we do specify the number of samples used for the FID computation and highlight the strong weaknesses of the FID metric.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

#### 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: we specified what type of GPU we used

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

#### 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

Answer: [Yes]

Justification: [NA]

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

#### 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: there is a dedicated broader impact section

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

### 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [No]

Justification: We work on standard image datasets

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We properly refer the torchcfm and PnPflow codebase.

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.

- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <paperswithcode.com/datasets> has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

#### 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

#### 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: [NA]

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

#### 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

#### Answer: [No]

Justification: LLMs were only used for grammatical purposes.

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy (<https://neurips.cc/Conferences/2025/LLM>) for what should or should not be described.

## A Proofs of Section [2](#page-1-0)

$$\hat{u}^*(x, t) = \sum_{i=1}^n u^{\text{cond}}(x, z = x^{(i)}, t) \cdot \frac{p(x|z = x^{(i)}, t)}{\sum_{i'=1}^n p(x|z = x^{(i')}, t)} . \quad (12)$$

#### A.1 Proof of Proposition [1](#page-2-4)

*Proof.* • In the case where <sup>z</sup> ∼ <sup>p</sup>ˆdata, conditional probability writes

$$p(z = x^{(i)} | x, t) = \frac{p(x, t, z = x^{(i)})}{p(x, t)} \quad (13)$$

$$= \frac{p(x|t, z = x^{(i)})p(t, z = x^{(i)})}{p(x, t)} \quad (14)$$

$$= \frac{p(x|t, z = x^{(i)})p(t, z = x^{(i)})}{\sum_{i'=1}^n p(x, t, z = x^{(i')})} \quad (15)$$

$$= \frac{p(x|t, z = x^{(i)})p(t) \underbrace{p(z = x^{(i)})}_{1}}{\sum_{i'=1}^n p(x|t, z = x^{(i')})p(t) \underbrace{p(z = x^{(i')})}_1} \quad (16)$$

n

$$= \frac{p(x|t, z = x^{(i)})}{\sum_{i'=1}^n p(x|t, z = x^{(i')})} \quad (17)$$

Pluging Equation [\(17\)](#page-20-1) in Equation [\(3\)](#page-2-2) yields the closed-formed formula for the velocity field:

$$u^*(x, t) = \sum_{i=1}^n u^{\text{cond}}(x, t, z = x^{(i)})p(z = x^{(i)} | x, t) \quad (18)$$

$$= \sum_{i=1}^n u^{\text{cond}}(x, t, z = x^{(i)}) \frac{p(x|t, z = x^{(i)})}{\sum_{i'=1}^n p(x|t, z = x^{(i')})} . \quad (19)$$

which proves Equation [\(12\)](#page-20-2); using that <sup>x</sup>|t, z <sup>=</sup> <sup>x</sup> (i) ∼ N (tx(i) ,(1 − <sup>t</sup>) 2 Id) and u cond(x, t, z = x (i) ) = <sup>x</sup> (i)−x 1−t yields Equation [\(6\)](#page-2-3).

• For the case <sup>z</sup> ∼ <sup>p</sup><sup>0</sup> × <sup>p</sup>ˆdata,

$$\hat{u}^*(x, t) := \int_z u^{\text{cond}}(x, t, z) p(z|x, t) \, dz \quad (20)$$

$$= \int_z u^{\text{cond}}(x, t, z) \frac{p(x, z, t)}{p(x, t)} dz \quad (21)$$

$$= \int_z u^{\text{cond}}(x, t, z) \frac{p(x|z, t)p(z)p(t)}{\int_z p(x|t, z')p(t)p(z') dz'} dz \quad (22)$$

$$= \int_z u^{\text{cond}}(x, t, z) \frac{p(x|z, t)p(z)}{\int_z' p(x|t, z')p(z') dz'} dz \quad (23)$$

Since <sup>z</sup> ∼ <sup>p</sup><sup>0</sup> × <sup>p</sup>ˆdata, the denominator is equal to:

$$\begin{aligned} \int_{z'} p(x|t, z') p(z') dz' &= \frac{1}{n} \int_{x_0} \sum_{i=1}^n \delta_x((1-t)x_0 + tx^{(i)}) \frac{1}{\sqrt{(2\pi)^d}} \exp\left(-\frac{1}{2}x_0^2\right) dx_0 & (24) \\ &= \frac{1}{n} \int_y \sum_{i=1}^n \delta_x(y) \frac{1}{\sqrt{(2\pi)^d}} \exp\left(-\frac{1}{2(1-t)^2} \|y - tx^{(i)}\|^2\right) \frac{1}{(1-t)^d} dy & (y = (1-t)x_0 + tx^{(i)}) \\ & & (25) \end{aligned}$$

$$= \frac{1}{n} \sum_{i=1}^n \frac{1}{\sqrt{(2\pi(1-t)^2)^d}} \exp\left(-\frac{1}{2(1-t)^2} \|x - tx^{(i)}\|^2\right) \quad (26)$$

Likewise, the numerator equals:

$$\begin{aligned} \int_z u^{\text{cond}}(x, t, z) p(x|z, t) p(z) \, dz &= \int_{x_0} \frac{1}{n} \sum_{i=1}^n (x^{(i)} - x_0) \delta_x((1-t)x_0 + tx^{(i)}) \frac{1}{\sqrt{(2\pi)^d}} \exp\left(-\frac{1}{2} \|x_0\|^2\right) dx_0 \\ &= \frac{1}{n} \sum_{i=1}^n \int_y \frac{x^{(i)} - y}{1-t} \delta_x(y) \frac{1}{\sqrt{(2\pi(1-t)^2)^d}} \exp\left(-\frac{1}{2(1-t)^2} \|y - tx^{(i)}\|^2\right) dy \\ &= \sum_{i=1}^n \frac{x^{(i)} - x}{1-t} \frac{1}{\sqrt{(2\pi(1-t)^2)^d}} \exp\left(-\frac{1}{2(1-t)^2} \|x - tx^{(i)}\|^2\right) \end{aligned} \quad (28)$$

$$(29)$$

Taking the ratio of Equations [\(24\)](#page-20-3) and [\(29\)](#page-21-2) concludes the proof.

## B Additional details and comments on empirical flow matching

First, recalls on the optimal velocity (Equation [\(6\)](#page-2-3)) and the empirical flow matching loss (Equations [\(7\)](#page-6-2) and [\(8\)](#page-6-1)) are provided in Appendix [B.1.](#page-21-3) The unbiasedness of the estimator is presented in Appendix [B.2,](#page-21-1) and its proof is in Appendix [B.3.](#page-22-0)

#### B.1 Recalls

The closed-form formula of the "optimal" velocity field is:

$$\hat{u}^*(x, t) = \sum_{l=1}^n \frac{x^{(l)} - x}{1 - t} \cdot \left[ \text{softmax} \left( \left( -\frac{\|x - tx^{(k)}\|^2}{2(1 - t)^2} \right)_{k=1, \dots, n} \right) \right]_l. \quad (6)$$

The proposed loss uses mini-batches of size M (instead of all n training points) to build an estimator uˆ ⋆ <sup>M</sup> of uˆ ⋆ :

$$\mathcal{L}_{\text{EFM}}(\theta) = \mathbb{E} \left[ u_{\theta}(x_t, t) - \hat{u}_M^*(x_t, t) \right]^2, \quad (7)$$

$$\begin{aligned} & x_0 \sim \hat{p}_0 \\ & x_1 \sim \hat{p}_{\text{data}} \\ & x_t = (1-t)x_0 + tx_1 \\ & b^{(1)} := x_1 ; b^{(2)}, \dots, b^{(M)} \sim \hat{p}_{\text{data}} \end{aligned}$$
with
$$\hat{u}_M^*(x_t, t) = \sum_{j=1}^M \frac{b^{(j)} - x_t}{1-t} \cdot \left[ \text{softmax} \left( \left( -\frac{\|x_t - tb^{(k)}\|^2}{2(1-t)^2} \right)_{k=1, \dots, M} \right) \right]_j. \quad (8)$$

Crucially, in Equation [\(7\)](#page-6-2) the sample b (1) depends on x<sup>t</sup> and is reused in the estimate uˆ ⋆ <sup>M</sup>. This important detail yields an unbiased estimator of uˆ ⋆ .

#### B.2 Theoretical properties of the proposed estimator

First, we discuss below the relation between Proposition [2](#page-6-4) and the sampling literature.

Links with importance sampling. The estimator uˆ ⋆ in Equation [\(6\)](#page-2-3) can be seen as a form of *importance sampling* (see [Robert et al.](#page-12-19) [1999,](#page-12-19) Chap. 3 for an in-depth reference). In a nutshell, importance sampling is a way to estimate an expectation when one cannot easily sample from the random variable it depends on. More precisely, in the ideal case <sup>z</sup> ∼ <sup>p</sup>data (as opposed to <sup>z</sup> ∼ <sup>p</sup>ˆdata), the velocity field formula is the following

$$u^*(x_t, t) = \mathbb{E}_{z|x_t, t} [u^{\text{cond}}(x_t, z, t)] \quad (30)$$

$$= \int_z u^{\text{cond}}(x_t, z, t) p(z|x_t, t) dz \quad (31)$$

When <sup>z</sup> ∼ <sup>p</sup>data, it is difficult to sample from <sup>z</sup>|<sup>x</sup>t, t, but the latter equation can be rewritten as

$$u^*(x_t, t) = \int_z u^{\text{cond}}(x_t, z, t) \frac{p(z|x_t, t)}{p(z)} p(z) dz \quad (32)$$

and one can easily sample from <sup>z</sup> ∼ <sup>p</sup>ˆdata using the empirical data distribution <sup>x</sup> (1), . . . , x(n)

$$u^*(x_t, t) \approx \frac{1}{n} \sum_{i=1}^n u^{\text{cond}}(x_t, x^{(i)}, t) \frac{p(z = x^{(i)} | x_t, t)}{p(x^{(i)})} \quad (33)$$

$$= \sum_{i=1}^n u^{\text{cond}}(x_t, x^{(i)}, t) p(z = x^{(i)} | x_t, t) \quad (34)$$

$$:= \hat{u}^*(x_t, t) . \quad (35)$$

#### B.3 Proof of Proposition [2](#page-6-4)

We first recall Appendix [B,](#page-21-0) which we prove in this section.

Proposition 2. *We denote the conditional probability distribution* p(z = x (i) | x, t) *over* {<sup>x</sup> (i)} n i=1 *by* <sup>p</sup>ˆdata(<sup>z</sup> | x, t)*. With no constraints on the learned velocity field* <sup>u</sup>θ*,*

*i) The minimizer of Equation* [\(7\)](#page-6-2) *writes, for all* (x, t)

$$\mathbb{E}_{b^{(1)} \sim \hat{p}_{\text{data}}(\cdot | x, t)} [\hat{u}_M^*(x, t)] \quad (9)$$

$$b^{(2)}, \dots, b^{(M)} \sim \hat{p}_{\text{data}}$$

*ii) In addition, for all* (x, t)*, the minimizer of Equation* [\(7\)](#page-6-2) *equals the optimal velocity field, i.e.,*

$$\mathbb{E}_{b^{(1)} \sim \hat{p}_{\text{data}}(\cdot | x, t)} [\hat{u}_M^*(x, t)] = \hat{u}^*(x, t) \quad (10)$$

$$b^{(2)}, \dots, b^{(M)} \sim \hat{p}_{\text{data}}$$

*iii) The conditional variance of the estimator* uˆ ⋆ <sup>M</sup> *is smaller than the usual conditional variance:*

$$\text{Var}_{b^{(1)} \sim \hat{p}_{\text{data}}(\cdot | x, t)} [\hat{u}_M^*(x, t)] \leq \text{Var}_{b^{(1)} \sim \hat{p}_{\text{data}}(\cdot | x, t)} [u^{\text{cond}}(x, b^{(1)}, t)]. \quad (11)$$

*Proof of Item* [\(i\)\)](#page-6-5)*.* With no constraints on uθ, the empirical flow matching loss writes:

$$\mathbb{E} \left[ t \sim \mathcal{U}([0,1]) \quad \|u_\theta(x_t, t) - \hat{u}_M^*(x_t, t)\|^2 \right], \quad (36)$$

$$\begin{aligned} & x_1 \sim \hat{p}_{\text{data}} \\ & x_t = (1-t)x_0 + tx_1 \\ & b^{(1)} := x_1 ; b^{(2)}, \dots, b^{(M)} \sim \hat{p}_{\text{data}} \end{aligned}$$

$$= \mathbb{E}_{t \sim \mathcal{U}([0,1])} \mathbb{E}_{b^{(1)} \sim \hat{p}_{\text{data}}(\cdot | x_t, t)} \|u_\theta(x_t, t) - \hat{u}_M^*(x_t, t)\|^2 , \quad (37)$$

$$= \mathbb{E}_{t \sim \mathcal{U}([0,1])} \mathbb{E}_{\substack{b^{(1)} := \hat{p}_{\text{data}}(\cdot | x_t, t) \\ b^{(2)} \dots b^{(M)} \sim \hat{p}_{\text{data}}}} \|u_\theta(x_t, t) - \hat{u}_M^*(x_t, t)\|^2 \text{ because } b^{(2)}, \dots, b^{(M)} \perp\!\!\!\perp x_t, t, \quad (38)$$

which is minimized when for all xt, t

$$u_\theta(x_t, t) = \mathbb{E}_{b^{(1)} \sim \hat{p}_{\text{data}}(\cdot | x_t, t)} [\hat{u}_M^*(x_t, t)] \quad . \quad (39)$$

$$b^{(2)}, \dots, b^{(M)} \sim \hat{p}_{\text{data}}$$

*Proof of Item* [\(ii\)\)](#page-7-3)*.* The minimizer for a given (xt, t), removing these elements from the notation for conciseness and abstraction, is a weighted mean:

$$\hat{u}^*(x_t, t) = \hat{u}^* = \sum_{l=1}^n w^{(l)} u^{(l)} , \text{ with } (40)$$

$$w^{(l)} = \hat{p}_{\text{data}}(z = x^{(l)} | t, x_t), \quad \sum_{l=1}^n w^{(l)} = 1 \quad (41)$$

$$u^{(l)} = u^{\text{cond}}(x_t, x^{(l)}, t) \quad (42)$$

We express a mini-batch as an <sup>M</sup>-valued vector of indices, <sup>i</sup> ∈ J1, nK<sup>M</sup>. The mini-batch estimate from Equation [\(7\)](#page-6-2), considering the definition of the softmax, can be expressed as a mini-batch weighted-mean:

$$\hat{u}_M^*(\mathbf{i}) = \frac{\sum_{j=1}^M w(\mathbf{i}_j) u(\mathbf{i}_j)}{\sum_{j=1}^M w(\mathbf{i}_j)} \quad (43)$$

The categorical distribution over J1, nK with probabilities following the weights <sup>w</sup> in [\(41\)](#page-22-1) is denoted Cat(w) and the uniform distribution, *i.e.,* Cat(1/n)), is denoted Unif.

The main result of the following is that, in expectation over the biased-mini-batches, where the first point is drawn according to <sup>w</sup> and the <sup>M</sup> − <sup>1</sup> other points are drawn uniformly, the mini-batch weighted-mean is an unbiased estimate of the w-weighted-mean uˆ ⋆ .

$$\mathbb{E} [\hat{u}_M^*(\mathbf{i})] := \mathbb{E}_{\mathbf{i}_1 \sim \text{Cat}(w)} \mathbb{E}_{i_2, \dots, i_M \sim \text{Unif}} [\hat{u}_M^*(\mathbf{i})] \quad (44)$$

$$= \sum_{i_1=1}^n w^{(i_1)} \mathbb{E}_{i_2, \dots, i_M \sim \text{Unif}} [\hat{u}_M^*(\mathbf{i})] \quad (45)$$

$$= \sum_{i_1=1}^n \mathbb{E}_{i_2, \dots, i_M \sim \text{Unif}} \left[ w^{(i_1)} \hat{u}_M^*(\mathbf{i}) \right] \quad (46)$$

$$= n \sum_{i_1=1}^n \frac{1}{n} \mathbb{E}_{i_2, \dots, i_M \sim \text{Unif}} \left[ w^{(i_1)} \hat{u}_M^*(\mathbf{i}) \right] \quad (47)$$

$$= n \mathbb{E}_{i_1 \sim \text{Unif}} \mathbb{E}_{i_2, \dots, i_M \sim \text{Unif}} \left[ w^{(i_1)} \hat{u}_M^*(\mathbf{i}) \right] \quad (48)$$

$$= n \mathbb{E}_{\mathbf{i}_1, \dots, \mathbf{i}_M \sim \text{Unif}} \left[ w^{(\mathbf{i}_1)} \hat{u}_M^*(\mathbf{i}) \right] \quad (49)$$

The expression in Equation [\(49\)](#page-23-0) is invariant with respect to order of the indices i1, . . . , iM: the indices in expectation in Equation [\(49\)](#page-23-0) can be exchanged, and one thus has

$$\forall k \in [1, M], \mathbb{E}[\hat{u}_M^*(\mathbf{i})] = n \mathbb{E}_{\mathbf{i}_1, \dots, \mathbf{i}_M \sim \text{Unif}} \left[ w^{(\mathbf{i}_k)} \hat{u}_M^*(\mathbf{i}) \right] . \quad (50)$$

$$\frac{1}{M} \sum_{k=1}^M \mathbb{E}\hat{u}_M^*(\mathbf{i}) = \frac{1}{M} \sum_{k=1}^M n \mathbb{E}_{\mathbf{i}_1, \dots, \mathbf{i}_M \sim \text{Unif}} \left[ w^{(\mathbf{i}_k)} \hat{u}_M^*(\mathbf{i}) \right] \quad (51)$$

$$\mathbb{E}\hat{u}_M^*(\mathbf{i}) = \frac{1}{M}n \mathbb{E}_{\mathbf{i}_1, \dots, \mathbf{i}_M \sim \text{Unif}} \left[ \sum_{k=1}^M w^{(\mathbf{i}_k)} \hat{u}_M^*(\mathbf{i}) \right] \quad (52)$$

$$= \frac{1}{M} n \mathbb{E}_{\mathbf{i}_1, \dots, \mathbf{i}_M \sim \text{Unif}} \left[ \sum_{k=1}^M w(\mathbf{i}_k) \frac{\sum_{j=1}^M w(\mathbf{i}_j) u(\mathbf{i}_j)}{\sum_{j=1}^M w(\mathbf{i}_j)} \right] \quad (53)$$

$$= \frac{1}{M} n \mathbb{E}_{\mathbf{i}_1, \dots, \mathbf{i}_M \sim \text{Unif}} \left[ \left( \sum_{k=1}^M w(\mathbf{i}_k) \right) \frac{\sum_{j=1}^M w(\mathbf{i}_j) u(\mathbf{i}_j)}{\left( \sum_{j=1}^M w(\mathbf{i}_j) \right)} \right] \quad (54)$$

$$= \frac{1}{M} n \mathbb{E}_{\mathbf{i}_1, \dots, \mathbf{i}_M \sim \text{Unif}} \left[ \sum_{j=1}^M w^{(\mathbf{i}_j)} u^{(\mathbf{i}_j)} \right] \quad (55)$$

$$= \frac{1}{M} n \sum_{j=1}^M \mathbb{E}_{i_1, \dots, i_M \sim \text{Unif}} \left[ w^{(i_j)} u^{(i_j)} \right] \quad (56)$$

$$= \frac{1}{M} n \sum_{j=1}^M \mathbb{E}_{i_j \sim \text{Unif}} \left[ w^{(i_j)} u^{(i_j)} \right] \quad (57)$$

$$= \frac{1}{M} n M \mathbb{E}_{l \sim \text{Unif}} \left[ w^{(l)} u^{(l)} \right] \quad (58)$$

$$= n \mathbb{E}_{l \sim \text{Unif}} \left[ w^{(l)} u^{(l)} \right] \quad (59)$$

$$= n \sum_{l=1}^n \frac{1}{n} \left[ w^{(l)} u^{(l)} \right] \quad (60)$$

$$\begin{aligned} &= \sum_{l=1}^n \left[ w^{(l)} u^{(l)} \right] \\ &= \hat{u}^* \end{aligned} \tag{61}$$
(62)

*Proof of Item* [\(iii\)\)](#page-7-4)*.* Using the same ideas as for Item [\(ii\)\)](#page-7-3), one has

$$\mathbb{E}_{x^{(1)} \sim \hat{p}_{\text{data}}(\cdot | x_t, t)}; b^{(2)}, \dots, b^{(M)} \sim \hat{p}_{\text{data}} \left[ \hat{u}_M^*(x_t, t)^2 \right] \quad (63)$$

$$= n\mathbb{E}_{\mathbf{i}_1, \dots, \mathbf{i}_M \sim \text{Unif}} \left[ w^{(\mathbf{i}_1)} \hat{u}_M^*(\mathbf{i})^2 \right] \quad (64)$$

$$= n\mathbb{E}_{\mathbf{i}_1, \dots, \mathbf{i}_M \sim \text{Unif}} \left[ w^{(\mathbf{i}_k)} \hat{u}_M^*(\mathbf{i})^2 \right], \forall k \in \llbracket 1, M \rrbracket \quad (65)$$

$$= n \frac{1}{M} \mathbb{E}_{\mathbf{i}_1, \dots, \mathbf{i}_M \sim \text{Unif}} \left[ \sum_{k=1}^M w^{(\mathbf{i}_k)} \hat{u}_M^*(\mathbf{i})^2 \right] \quad (66)$$

$$= n \frac{1}{M} \mathbb{E}_{\mathbf{i}_1, \dots, \mathbf{i}_M \sim \text{Unif}} \left[ \sum_{k=1}^M w^{(\mathbf{i}_k)} \left( \frac{\sum_{j=1}^M w^{(\mathbf{i}_j)} u^{(\mathbf{i}_j)}}{\sum_{j=1}^M w^{(\mathbf{i}_j)}} \right)^2 \right] \quad (67)$$

$$\leq n \frac{1}{M} \mathbb{E}_{\mathbf{i}_1, \dots, \mathbf{i}_M \sim \text{Unif}} \left[ \sum_{k=1}^M w^{(\mathbf{i}_k)} \frac{\sum_{j=1}^M w^{(\mathbf{i}_j)} (u^{(\mathbf{i}_j)})^2}{\sum_{j=1}^M w^{(\mathbf{i}_j)}} \right] \text{ by convexity of } x \mapsto x^2 \quad (68)$$

$$= n \frac{1}{M} \mathbb{E}_{i_1, \dots, i_M \sim \text{Unif}} \left[ \left( \sum_{k=1}^M w(i_k) \right) \frac{\sum_{j=1}^M w(i_j) (w(i_j))^2}{\sum_{j=1}^M w(i_j)} \right] \quad (69)$$

$$= n \frac{1}{M} \mathbb{E}_{i_1, \dots, i_M \sim \text{Unif}} \left[ \sum_{j=1}^M w^{(i_j)} (u^{(i_j)})^2 \right] \quad (70)$$

$$= \mathbb{E}_{\mathbf{i}_1 \sim \text{Unif}} \left[ w^{(\mathbf{i}_1)} (u^{(\mathbf{i}_1)})^2 \right] \quad (71)$$

$$= \mathbb{E}_{l \sim \text{Unif}} \left[ w^{(l)}(u^{(l)})^2 \right] . \quad (72)$$

Hence

$$\mathbb{E}_{x^{(1)} \sim \hat{p}_{\text{data}}(\cdot | x_t, t)} ; b^{(2)}, \dots, b^{(M)} \sim \hat{p}_{\text{data}} \left[ \hat{u}_M^*(x_t, t)^2 \right] - (\hat{u}^*)^2 \leq \mathbb{E}_{l \sim \text{Unif}} \left[ w^{(l)}(u^{(l)})^2 \right] - (\hat{u}^*)^2 , \quad (73)$$

which is exactly

$$\text{Var}_{x^{(1)} \sim \hat{p}_{\text{data}}(\cdot | x_t, t)} ; b^{(2)}, \dots, b^{(M)} \sim \hat{p}_{\text{data}} [\hat{u}_M^*(x_t, t)] \leq \text{Var}_{x^{(1)} \sim \hat{p}_{\text{data}}(\cdot | x_t, t)} \left[ u^{\text{cond}}(x_t, x^{(1)}, t) \right]. \quad (74)$$

## C Additional experiments

We present below the results for the MNIST dataset. The conclusions atre the same as for the CIFAR-10 and CelebA <sup>64</sup> × <sup>64</sup>: regressing against a more deterministic velocity field does not hurt generalization. On the contrary, generalization (*i.e.,* lower test FID) appears earlier during training.

For this experiment, we used the Unet with attention and timestep embedding from torchcfm library, with the Adam optimizer and all the default parameters. We used a pretrained classifier with 99% accuracy on MNIST (90% on FMNIST) as a lower-dimensional embedding of size 128 to compute the FID between the test set and the generated set.

|      | Method     | Ep.    | 1 Ep.  | 2 Ep. | 3 Ep. | 4 Ep. | 5 Ep. 10 | Ep. 15 | Ep. 20 | Ep. 25 |
|------|------------|--------|--------|-------|-------|-------|----------|--------|--------|--------|
| CFM  | (EFM, M=1) | 378.00 | 181.25 | 67.88 | 29.44 | 15.30 | 4.20     | 3.08   | 2.51   | 2.28   |
| EFM, | M=128      | 370.64 | 168.58 | 60.52 | 25.52 | 13.44 | 3.79     | 2.70   | 2.35   | 2.10   |
| EFM, | M=256      | 370.94 | 169.71 | 61.88 | 25.73 | 13.48 | 3.73     | 2.76   | 2.33   | 2.08   |
| EFM, | M=1024     | 369.72 | 168.43 | 60.28 | 24.24 | 12.26 | 3.30     | 2.67   | 2.17   | 1.84   |

Table 1: FID FMNIST. FID scores across training epochs for conditional flow matching and empirical flow matching for multiple values of the number of samples M used to estimate the closed-form uˆ ⋆ .

|      | Method     | FID Ep. | 5 FID Ep. | 10 FID Ep. 50 | FID Ep. 100 | FID Ep. 200 |
|------|------------|---------|-----------|---------------|-------------|-------------|
| CFM  | (EFM, M=1) | 253.56  | 48.67     | 25.36         | 21.35       | 19.67       |
| EFM, | M=128      | 206.27  | 44.08     | 23.39         | 19.63       | 17.72       |
| EFM, | M=256      | 202.62  | 45.06     | 22.16         | 20.08       | 17.74       |
| EFM, | M=512      | 194.66  | 44.19     | 22.10         | 18.93       | 16.85       |

Table 2: FID FMNIST. FID scores across training epochs for conditional flow matching and empirical flow matching for multiple values of the number of samples M used to estimate the closed-form uˆ ⋆ .

## D Experiments details

For all the experiment we used all the same learning hyperparameters, the default ones form [Tong](#page-12-14) [et al.](#page-12-14) [\(2024\)](#page-12-14). The hyperparameter values are summarized in Table [3.](#page-26-1) The details specific to each figure are described in Appendices [D.2](#page-26-2) to [D.5](#page-27-0)

| # Channels | Batch Size | Learning Rate | EMA Decay | Gradient Clipping |
|------------|------------|---------------|-----------|-------------------|
| 128        | 128        | 0.0002        | 0.9999    | 1                 |

#### D.1 Compute time

Given that regressing against an estimate of the closed-form, EFM, seems to improve on CFM, one may wonder what is the additional cost induced by EFN. To alleviate the non-linearity of GPU computing (parallelism may cause some discontinuities in terms of costs), we ran an exhaustive set of timing experiments, varying the batch size and the EFM sample size. To summarize the measurements (numbers are given for an NVIDIA L4 GPU, on CIFAR-10), denoting b the batch size and e the EFM sample size, the cost follows <sup>b</sup> × (4.3ms <sup>+</sup> <sup>e</sup> × <sup>0</sup>.9µs). It can be also be seen as adding ∼ 2% for every 100 EFM samples. Or, for instance with a batch size of 256, 1.1 second will be due to the 256-sample forward/backward, while the additional cost for EFM-1000 will be 230ms (around 17% of the cost) and for EFM-128 under 30ms (under 3%).

#### D.2 Figures [1a](#page-3-1) and [1c](#page-3-1)

For Figure [1a](#page-3-1) no deep learning is involved: the datasets 2-moons and CIFAR-10 are loaded. Then, <sup>256</sup> points from <sup>p</sup><sup>0</sup> × <sup>p</sup>ˆdata are drawn, and one computes the mean of the cosine similarities between uˆ ⋆ ((1 − <sup>t</sup>)x<sup>0</sup> <sup>+</sup> tx1, t) and <sup>u</sup> cond((1 − <sup>t</sup>)x<sup>0</sup> <sup>+</sup> tx1, z <sup>=</sup> <sup>x</sup>1, t) = <sup>x</sup><sup>1</sup> − <sup>x</sup>0, for each value of <sup>t</sup> ∈ {0, <sup>1</sup>/100, <sup>2</sup>/100, . . . , <sup>99</sup>/100}.

No deep learning either is involved in Figure [1c:](#page-3-1) the Imagenette dataset is loaded and spatially subsampled to resolution dim = 8, dim = 16, . . . , dim = 256, *i.e.,* with <sup>d</sup> = " · <sup>8</sup> 2 , d = <sup>3</sup> · <sup>16</sup><sup>2</sup> , . . . , <sup>d</sup> = 3 · <sup>256</sup><sup>2</sup> . Then, as for Figure [1a,](#page-3-1) batches of 256 points from p<sup>0</sup> and pdata are drawn, and one computes the percentage of cosine similarities between uˆ ⋆ ((1 − <sup>t</sup>)x<sup>0</sup> <sup>+</sup> tx1, t) and u cond((1 − <sup>t</sup>)x<sup>0</sup> <sup>+</sup> tx1, z <sup>=</sup> <sup>x</sup>1, t) = <sup>x</sup><sup>1</sup> − <sup>x</sup>0, that are larger than <sup>0</sup>.9, for multiple time values <sup>t</sup>.

#### D.3 Figure [2](#page-4-1)

In Figure [2,](#page-4-1) networks are trained with a vanilla conditional flow matching, with the standard 34 million parameters U-Net for diffusion by [Nichol and Dhariwal](#page-11-19) [\(2021\)](#page-11-19), with default settings from the torchfm codebase [<sup>4</sup>](#page-26-3) [\(Tong et al.,](#page-12-14) [2024\)](#page-12-14). Training uses the CFM loss. For this specific experiment, we removed the usual random flip transform, for uˆ ⋆ to be simpler and easier to estimate by uθ. For each "data" subsampling of the dataset, we trained the model for <sup>5</sup> · <sup>10</sup><sup>4</sup> iterations, with a batch size of 128, *i.e.,* we trained the models for 128 epochs.

<sup>4</sup><https://github.com/atong01/conditional-flow-matching>

### D.4 Figure [3](#page-5-1)

In Figure [3,](#page-5-1) for each dataset (CIFAR-10 and CelebA <sup>64</sup> × <sup>64</sup>), one network is trained using a vanilla conditional flow matching with the default parameters of [Tong et al.](#page-12-14) [\(2024\)](#page-12-14) (the most important ones are recalled in Table [3\)](#page-26-1). Then images are generated first following the closed-form formula of the optimal velocity field uˆ ⋆ from 0 to τ . And then following the velocity field learned with a usual conditional flow matching u<sup>θ</sup> from τ to 1.

#### D.5 Figure [4](#page-8-1)

For experiments involving training on CIFAR-10 (Figures [2](#page-4-1) and [3\)](#page-5-1), we rely on the standard 34 million parameters U-Net for diffusion by [Nichol and Dhariwal](#page-11-19) [\(2021\)](#page-11-19), with default settings from the torchfm codebase [\(Tong et al.,](#page-12-14) [2024\)](#page-12-14). For each algorithm, the networks are trained for 500k iterations with batch size 128, *i.e.,* 1280 epochs.

For CelebA <sup>64</sup> × <sup>64</sup> (Figure [3\)](#page-5-1), we rely on the training script of pnpflow library[<sup>5</sup>](#page-27-1) [\(Martin et al.,](#page-11-20) [2025\)](#page-11-20), which uses a U-Net from [Huang et al.](#page-11-21) [\(2021\)](#page-11-21); [Ho et al.](#page-11-0) [\(2020\)](#page-11-0).

<sup>5</sup><https://github.com/annegnx/PnP-Flow>