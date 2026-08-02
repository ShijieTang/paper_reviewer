# Block Diffusion: Interpolating Between Au- Toregressive And Diffusion Language Models

Marianne Arriola† ∗ Aaron Kerem Gokaslan† Justin T. Chiu‡ **Zhihan Yang**† Zhixuan Qi† Jiaqi Han¶ Subham Sekhar Sahoo† **Volodymyr Kuleshov**†

## Abstract

Diffusion language models offer unique benefits over autoregressive models due to their potential for parallelized generation and controllability, yet they lag in likelihood modeling and are limited to fixed-length generation. In this work, we introduce a class of block diffusion language models that interpolate between discrete denoising diffusion and autoregressive models. Block diffusion overcomes key limitations of both approaches by supporting flexible-length generation and improving inference efficiency with KV caching and parallel token sampling. We propose a recipe for building effective block diffusion models that includes an efficient training algorithm, estimators of gradient variance, and data-driven noise schedules to minimize the variance. Block diffusion sets a new state-of-the-art performance among diffusion models on language modeling benchmarks and enables generation of arbitrary-length sequences. We provide the code1, along with the model weights and blog post on the project page:
https://m-arriola.com/bd3lms

## 1 Introduction

Diffusion models are widely used to generate images (Ho et al., 2020; Dhariwal & Nichol, 2021; Sahoo et al., 2024b) and videos (Ho et al., 2022; Gupta et al., 2023), and are becoming increasingly effective at generating discrete data such as text (Lou et al., 2024; Sahoo et al., 2024a) or biological sequences (Avdeyev et al., 2023; Goel et al., 2024). Compared to autoregressive models, diffusion models have the potential to accelerate generation and improve the controllability of model outputs (Schiff et al., 2024; Nisonoff et al., 2024; Li et al., 2024; Sahoo et al., 2024c). Discrete diffusion models currently face at least three limitations. First, in applications such as chat systems, models must generate output sequences of arbitrary length (e.g., a response to a user's question). However, most recent diffusion architectures only generate fixed-length vectors (Austin et al., 2021; Lou et al., 2024). Second, discrete diffusion uses bidirectional context during generation and therefore cannot reuse previous computations with KV caching, which makes inference less efficient (Israel et al., 2025). Third, the quality of discrete diffusion models, as measured by standard metrics such as perplexity, lags behind autoregressive approaches and further limits their applicability (Gulrajani & Hashimoto, 2024; Sahoo et al., 2024a). This paper makes progress towards addressing these limitations by introducing Block Discrete Denoising Diffusion Language Models (BD3-LMs), which interpolate between discrete diffusion and autoregressive models. Specifically, block diffusion models (also known as semi-autoregressive models) define an autoregressive probability distribution over blocks of discrete random variables (Si et al., 2022; 2023); the conditional probability of a block given previous blocks is specified by a discrete denoising diffusion model (Austin et al., 2021; Sahoo et al., 2024a). Developing effective BD3-LMs involves two challenges. First, efficiently computing the training objective for a block diffusion model is not possible using one standard forward pass of a neural
∗Correspondence to Marianne Arriola: marriola@cs.cornell.edu
†Cornell Tech, NY, USA. ¶Stanford University, CA, USA. ‡ Cohere, NY, USA.

1Code: https://github.com/kuleshov-group/bd3lms 1

High quality Arbitrary-length KV caching Autoregression:
Not Parallelizable There are three categories of the average There are three categories of the average rate There are three categories of the average rate of...

Generation steps Diffusion:
the reusability will continue to the Lower quality Fixed-length No KV caching Parallelizable Repeal the reusability cuts and the law will continue to reduce the Repeal the reusability cuts and prove the law will continue to reduce the deficit.

Block Diffusion (Ours):
On September 17, **we be**
Parallelizable High quality Arbitrary-length KV caching On September 17, 2016, we will be giving the release of On September 17, 2016, we will be giving the beta-release of the to our server **testing ...**
Figure 1: Block diffusion sequentially generates blocks of tokens by performing diffusion within each block and conditioning on previous blocks. By combining strength from autoregressive and diffusion models, block diffusion overcomes the limitations of both approaches by supporting variable-length, higher-quality generation and improving inference efficiency with KV caching and parallel sampling. network and requires developing specialized algorithms. Second, training is hampered by the high variance of the gradients of the diffusion objective, causing BD3-LMs to under-perform autoregression even with a block size of one (when both models should be equivalent). We derive estimators of gradient variance, and demonstrate that it is a key contributor to the gap in perplexity between autoregression and diffusion. We then propose custom noise processes that minimize gradient variance and make progress towards closing the perplexity gap.

We evaluate BD3-LMs on language modeling benchmarks, and demonstrate that they are able to generate sequences of arbitrary length, including lengths that exceed their training context. In addition, BD3-LMs achieve new state-of-the-art perplexities among discrete diffusion models. Compared to alternative semi-autoregressive formulations that perform Gaussian diffusion over embeddings (Han et al., 2022; 2023), our discrete approach features tractable likelihood estimates and yields samples with improved generative perplexity using an order of magnitude fewer generation steps. In summary, our work makes the following contributions:
- We introduce block discrete diffusion language models, which are autoregressive over blocks of tokens; conditionals over each block are based on discrete diffusion. Unlike prior diffusion models, block diffusion supports variable-length generation and KV caching.

- We introduce custom training algorithms for block diffusion models that enable efficiently leveraging the entire batch of tokens provided to the model.

- We identify gradient variance as a limiting factor of the performance of diffusion models, and we propose custom data-driven noise schedules that reduce gradient variance.

- Our results establish a new state-of-the-art perplexity for discrete diffusion and make progress toward closing the gap to autoregressive models.

## 2 Background: Language Modeling Paradigms

Notation We consider scalar discrete random variables with V categories as 'one-hot' column vectors in the space V = {x ∈ {0, 1}
V:Pi xi = 1} ⊂ ∆Vfor the simplex ∆V. Let the V -th category denote a special [MASK] token, where m ∈ V is its one-hot vector. We define x 1:L as a sequence of L tokens, where x ℓ ∈ V for all tokens ℓ ∈ {1*, . . . , L*}, and use V
L to denote the set of all such sequences. Throughout the work, we simplify notation and refer to the token sequence as x and an individual token as x ℓ. Finally, let Cat(·; p) be a categorical distribution with probability p ∈ ∆V.

## 2.1 Autoregressive Models

Consider a sequence of L tokens x =-x 1*, . . . ,* x Ldrawn from the data distribution q(x). Autoregressive (AR) models define a factorized distribution of the form

$$\log p_{\theta}(\mathbf{x})=\sum_{\ell=1}^{L}\log p_{\theta}(\mathbf{x}^{\ell}\mid\mathbf{x}^{<\ell}),$$
$$(1)$$

where each pθ(x ℓ| x
<ℓ) is parameterized directly with a neural network. As a result, AR models may be trained efficiently via next token prediction. However, AR models take L steps to generate L tokens due to the sequential dependencies.

## 2.2 Discrete Denoising Diffusion Probabilistic Models

Diffusion models fit a model pθ(x) to reverse a forward corruption process q (Sohl-Dickstein et al.,
2015; Ho et al., 2020; Sahoo et al., 2024b). This process starts with clean data x and defines latent variables xt =-x 1 t*, . . . ,* x L t for t ∈ [0, 1], which represent progressively noisier versions of x. Given a discretization into T steps, we define s(j*) = (*j − 1)/T and t(j) = j/T. For brevity, we drop j from t(j) and s(j) below; in general, s denotes the time step preceding t.

The D3PM framework (Austin et al., 2021) defines q as a Markov forward process acting independently on each token x ℓ: q(x ℓt | x ℓs
) = Cat(x ℓ t
; Qtx ℓs
) where Qt ∈ R
V ×Vis the diffusion matrix.

The matrix Qt can model various transformations, including masking, random token changes, and related word substitutions.

An ideal diffusion model pθ is the reverse of the process q. The D3PM framework defines pθ as

$$p_{\theta}({\bf x}_{s}\mid{\bf x}_{t})=\prod_{\ell=1}^{L}p_{\theta}({\bf x}_{s}^{\ell}\mid{\bf x}_{t})=\sum_{\bf x}\left[\prod_{\ell=1}^{L}q({\bf x}_{s}^{\ell}\mid{\bf x}_{t}^{\ell},{\bf x}^{\ell})p_{\theta}({\bf x}^{\ell}\mid{\bf x}_{t})\right],\tag{2}$$

where the denoising base model pθ(x ℓ| xt) predicts clean token x ℓ given the noisy sequence xt, and the reverse posterior q(x ℓs | x ℓt
, x) is defined following Austin et al. (2021) in Suppl. B.3.

The diffusion model pθ is trained using variational inference. Let KL[·] denote the Kullback-Leibler

divergence. Then, the Negative ELBG (NELBO) is given by (Soli-Dickstein et al., 2015):  $$\mathcal{L}(\mathbf{x};\theta)=\mathbb{E}_{q}\Bigg{[}-\log p_{\theta}(\mathbf{x}|\mathbf{x}_{t(1)})+\sum_{j=1}^{T}D_{\mathrm{KL}}[q(\mathbf{x}_{t(j)}|\mathbf{x}_{t(j)},\mathbf{x})\|p_{\theta}(\mathbf{x}_{t(j)}|\mathbf{x}_{t(j)})]+D_{\mathrm{KL}}[q(\mathbf{x}_{t(T)}|\mathbf{x})\|p_{\theta}(\mathbf{x}_{t(T)})]\Bigg{]}\tag{3}$$
This formalism extends to continuous time via Markov chain (CTMC) theory and admits score-based generalizations (Song & Ermon, 2019; Lou et al., 2024; Sun et al., 2022). Further simplifications (Sahoo et al., 2024a; Shi et al., 2024; Ou et al., 2025) tighten the ELBO and enhance performance.

## 3 Block Diffusion Language Modeling

We explore a class of Block Discrete Denoising Diffusion Language Models (BD3-LMs) that interpolate between autoregressive and diffusion models by defining an autoregressive distribution over blocks of tokens and performing diffusion within each block. We provide a block diffusion objective for maximum likelihood estimation and efficient training and sampling algorithms. We show that for a block size of one, the diffusion objective suffers from high variance despite being equivalent to the autoregressive likelihood in expectation. We identify high training variance as a limitation of diffusion models and propose data-driven noise schedules that reduce the variance of the gradient updates during training.

## 3.1 Block Diffusion Distributions And Model Architectures

We propose to combine the language modeling paradigms in Sec. 2 by autoregressively modeling blocks of tokens and performing diffusion within each block. We group tokens in x into B blocks of length L
′ with B = L/L′(we assume that B is an integer). We denote each block x
(b−1)L
′:bL′from token at positions (b − 1)L
′to bL′for blocks b ∈ {1*, . . . , B*} as x bfor simplicity. Our likelihood factorizes over blocks as

$$\log p_{\theta}(\mathbf{x})=\sum_{b=1}^{B}\log p_{\theta}(\mathbf{x}^{b}\mid\mathbf{x}^{<b}),$$
$$\quad(4)$$

and each pθ(x b| x
<b) is modeled using discrete diffusion over a block of L
′tokens. Specifically, we define a reverse diffusion process as in (2), but restricted to block b:

$$p_{\theta}({\bf x}_{s}^{b}\mid{\bf x}_{t}^{b},{\bf x}^{<b})=\sum_{{\bf x}^{b}}q({\bf x}_{s}^{b}\mid{\bf x}_{t}^{b},{\bf x}^{b})p_{\theta}({\bf x}^{b}\mid{\bf x}_{t}^{b},{\bf x}^{<b})$$

We obtain a principled learning objective by applying the NELBO in (3) to each term in (4) to obtain

$$-\log p_{\theta}(\mathbf{x})\leq{\mathcal{L}}_{\mathrm{BD}}(\mathbf{x};\theta):=\sum_{b=1}^{B}{\mathcal{L}}(\mathbf{x}^{b},\mathbf{x}^{<b};\theta),$$
$$(5)$$
$$(6)$$
$$\left(7\right)$$

where each L(x b, x
<b; θ) is an instance of (3) applied to log pθ(x b| x
<b). Since the model is conditioned on x
<b, we make the dependence on x
<b, θ explicit in L. We denote the sum of these terms LBD(x; θ) (itself a valid NELBO).

Model Architecture Crucially, we parameterize the B base denoiser models pθ(x b| x b t, x
<b) using a single neural network xθ. The neural network xθ outputs not only the probabilities pθ(x b| x b t, x
<b),
but also computational artifacts for efficient training. This will enable us to compute the loss LBD(x; θ)
in parallel for all B blocks in a memory-efficient manner. Specifically, we parameterize xθ using a transformer (Vaswani et al., 2017) with a block-causal attention mask. The transformer xθ is applied to L tokens, and tokens in block b attend to tokens in blocks 1 to b. When xθ is trained, x b θ
(x bt
, x
<b)
yields L
′ predictions for denoised tokens in block b based on noised x b tand clean x
<b.

In autoregressive generation, it is normal to cache keys and values for previously generated tokens to avoid recomputing them at each step. Similarly, we use Kb, Vbto denote the keys and values at block b, and we define xθ to support these as input and output. The full signature of xθ is

$$\mathbf{x}_{\mathrm{logits}}^{b},\mathbf{K}^{b},\mathbf{V}^{b}\leftarrow\mathbf{x}_{\theta}^{b}(\mathbf{x}_{t}^{b},\mathbf{K}^{1:b-1},\mathbf{V}^{1:b-1}):=\mathbf{x}_{\theta}^{b}(\mathbf{x}_{t}^{b},\mathbf{x}^{<b}),$$

where x b logits are the predictions for the clean x b, and Kb, Vbis the key-value cache in the forward pass of xθ, and K1:b−1, V1:b−1are keys and values cached on a forward pass of xθ over x
<b (hence the inputs x
<b and K1:b−1, V1:b−1are equivalent).

## 3.2 Efficient Training And Sampling Algorithms

Ideally, we wish to compute the loss LBD(x; θ) in one forward pass of xθ. However, observe that denoising x b trequires a forward pass on this noisy input, while denoising the next blocks requires running xθ on the clean version x b. Thus every block has to go through the model at least twice.

Training Based on this observation, we propose a training algorithm with these minimal computational requirements (Alg. 1). Specifically, we precompute keys and values K1:B, V1:B for the full sequence x in a first forward pass (∅, K1:B, V1:B) ← xθ(x). We then compute denoised predictions for all blocks using x b θ
(x bt
, K1:b-1, V1:b-1). Each token passes through xθ twice.

Vectorized Training Naively, we would compute the logits by applying x b θ
(x bt
, K1:b-1, V1:b-1) in a loop B times. We propose a vectorized implementation that computes LBD(x; θ) in one forward pass on the concatenation xnoisy ⊕ x of clean data x with noisy data xnoisy = x 1 t1 *⊕ · · · ⊕* x B
tB
obtained by applying a noise level tb to each block x b. We design an attention mask for xnoisy ⊕ x such that noisy tokens attend to other noisy tokens in their block and to all clean tokens in preceding blocks (see Suppl. B.6). Our method keeps the overhead of training BD3-LMs tractable and combines with pretraining to further reduce costs. Sampling We sample one block at a time, conditioned on previously sampled blocks (Alg 2). We may use any sampling procedure SAMPLE(x bθ
, K1:b-1, V1:b-1) to sample from the conditional distribution pθ(x bs |x b t
, x
<b), where the context conditioning is generated using cross-attention with pre-computed keys and values K1:b−1, V1:b−1. Similar to AR models, caching the keys and values saves computation instead of recalculating them when sampling a new block. Notably, our block diffusion decoding algorithm enables us to sample sequences of arbitrary length, whereas diffusion models are restricted to fixed-length generation. Further, our sampler admits parallel generation within each block, whereas AR samplers are constrained to generate token-by-token.

| Algorithm 1 Block Diffusion Training                                                                                                                                                                                                                                                                                                                            | Algorithm 2 Block Diffusion Sampling   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------|
| Input: datapoint x, # of blocks B, forward noise process qt(·|x), model xθ, loss LBD repeat Sample t1, . . . , tB ∼ U[0, 1] ∀b ∈ {1, ..., B} : x b tb ∼ qtb (·|x b ) ∅, K1:B, V1:B ← xθ(x) ▷ KV cache ∀b: x b logit, ∅, ∅ ← x b θ (x b tb , K1:b-1 , V1:b-1 ) 1 B Let xlogit ← x ⊕ · · · ⊕ x logit logit Take gradient step on ∇θLBD(xlogit; θ) until converged | Input: # blocks B, model xθ, diffusion sampling algorithm SAMPLE x, K, V ← ∅ ▷ output & KV cache for b = 1 to B do x b ← SAMPLE(x b θ , K1:b-1 , V1:b-1 ) ∅, Kb , Vb ← x b θ (x b ) x ← x 1:b−1 ⊕ x b (K, V) ← (K1:b−1 ⊕ Kb , V1:b−1 ⊕ Vb ) end for return x                                        |

## 4 Understanding Likelihood Gaps Between Diffusion & Ar Models 4.1 Masked Bd3-Lms

The most effective diffusion language models leverage a masking noise process (Austin et al., 2021; Lou et al., 2024; Sahoo et al., 2024a), where tokens are gradually replaced with a special mask token. Here, we introduce masked BD3-LMs, a special class of block diffusion models based on the masked diffusion language modeling framework (Sahoo et al., 2024a; Shi et al., 2024; Ou et al., 2025).

More formally, we adopt a per-token noise process q(x ℓt |x ℓ) = Cat(x ℓ t
; αtx ℓ + (1 − αt)m) for tokens ℓ ∈ {1*, . . . , L*} where m is a one-hot encoding of the mask token, and αt ∈ [0, 1] is a strictly decreasing function in t, with α0 = 1 and α1 = 0. We employ the linear schedule where the probability of masking a token at time t is 1 − αt. We adopt the simplified objective from Sahoo et al.

(2024a); Shi et al. (2024); Ou et al. (2025) (the full derivation is provided in Suppl. B.3):

$$-\log p_{\theta}(\mathbf{x})\leq\mathcal{L}_{\text{BD}}(\mathbf{x};\theta):=\sum_{b=1}^{B}\mathbb{E}_{t\sim[0,1]}\mathbb{E}_{q}\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}\log p_{\theta}(\mathbf{x}^{b}|\mathbf{x}_{t}^{b},\mathbf{x}^{<b})\tag{8}$$

where α
′tis the instantaneous rate of change of αt under the continuous-time extension of (3) that takes T → ∞. The NELBO is tight for L
′ = 1 but becomes a looser approximation of the true negative log-likelihood for L
′ → L (see Suppl. B.5).

## 4.2 Case Study: Single Token Generation

Table 1: Test perplexities for singletoken generation (PPL; ↓) across 16B tokens on LM1B.

Our block diffusion parameterization (8) is equivalent in expectation to the autoregressive NLL (1) in the limiting case where L
′ = 1 (see Suppl. B.4). Surprisingly, we find a two point perplexity gap between our block diffusion model for L
′ = 1 and AR when training both models on the LM1B dataset.

Although the objectives are equivalent in expectation, we show that the remaining perplexity gap is a result of high training variance. Whereas AR is trained using the cross-entropy of L tokens, our block diffusion model for L
′ = 1 only computes the cross-entropy for masked tokens x ℓt = m ∀ℓ ∈ {1*, . . . L*}

| PPL (↓)                   |         |
|---------------------------|---------|
| AR                        | 22.88   |
| + random batch size       | 24.37   |
| ′ = 1                     | ≤ 25.56 |
| BD3-LM L + tuned schedule | 22.88   |

50k 100k 150k 200k 250k 3 3.2 3.4 3.6 3.8 4 Model BD3-LM (NELBO) BD3-LM (Tuned schedule) AR AR (random batch size)
Train Negative Log-Likelihood (NLL) for Single Token Generation on LM1B
NL
L

Train steps
so that Et∼U[0,1]q(x ℓt = m|x ℓ) = 0.5. Thus, training on the diffusion objective involves estimating loss gradients with 2x fewer tokens and is responsible for higher training variance compared to AR. To close the likelihood gap, we train a BD3-LM for L
′ = 1 by designing the forward process to fully mask tokens, i.e. q(x ℓ t = m|x ℓ) = 1. Under this schedule, the diffusion objective becomes equivalent to the AR objective (Suppl. B.4). In Table 1, we show that training under the block diffusion objective yields the same perplexity as AR training. Empirically, we see that this reduces the variance of the training loss in Figure 2. We verify that tuning the noise schedule reduces the variance of the objective by measuring Varx,t [LBD(x; θ)] after training on 328M tokens: while training on the NELBO results in a variance of 1.52, training under full masking reduces the variance to 0.11.

## 4.3 Diffusion Gap From High Variance Training

Next, we formally describe the issue of gradient variance in training diffusion models. Given our empirical observations for single-token generation, we propose an estimator for gradient variance that we use to minimize the variance of diffusion model training for L
′ ≥ 1. While the NELBO is invariant to the choice of noise schedule (Suppl. B.3), this invariance does not hold for our Monte Carlo estimator of the loss used during training. As a result, the variance of the estimator and its gradients are dependent on the schedule. First, we express the estimator of the NELBO with a batch size K. We denote a batch of sequences as X =-x
(1), x
(2)*, . . . ,* x
(K), with each x
(k)iid∼ q(x). We obtain the batch NELBO estimator below, where t(*k, b*) is sampled in sequence k and block b:

the dual Markov commutator $\mathcal{L}_{\text{DD}}$, where $\mathcal{L}(\mathbf{x},\theta)$ is sampled in sequence $\mathbf{x}$ and below $\theta$.  $$\mathcal{L}_{\text{BD}}(\mathbf{X};\theta):=l(\mathbf{X};\theta)=\frac{1}{K}\sum_{k=1}^{K}\sum_{b=1}^{B}\frac{\alpha_{t(k,b)}^{t}}{1-\alpha_{t(k,b)}}\log p_{\theta}\left(\mathbf{x}^{(k),b}\mid\mathbf{x}_{t(k,b)}^{(k),b},\mathbf{x}^{(k),\,c,b}\right)\tag{9}$$
The variance of the gradient estimator over M batches for each batch Xm ∀m ∈ {1*, . . . , M*} is:

$$\mathrm{Var}_{\mathbf{X},t}\left[\nabla_{\theta}l(\mathbf{X};\theta)\right]\approx\frac{1}{M-1}\sum_{m=1}^{M}\left\|\nabla_{\theta}l(\mathbf{X}^{m};\theta)-\frac{1}{M}\sum_{m=1}^{M}\nabla_{\theta}l(\mathbf{X}^{m};\theta)\right\|_{2}^{2}\tag{10}$$

## 5 Low-Variance Noise Schedules For Bd3-Lms

5.1 INTUITION: AVOID EXTREME MASK RATES We aim to identify schedules that minimize the variance of the gradient estimator and make training most efficient. In a masked setting, we want to mask random numbers of tokens, so that the model learns to undo varying levels of noise, which is important during sampling. However, if we mask very few tokens, reconstructing them is easy and does not provide useful learning signal. If we mask everything, the optimal reconstruction are the marginals of each token in the data distribution, which is easy to learn, and again is not useful. These extreme masking rates lead to poor high-variance gradients: we want to learn how to clip them via a simple and effective new class of schedules.

## 5.2 Clipped Schedules For Low-Variance Gradients

We propose a class of "clipped" noise schedules that sample mask rates 1 − αt ∼ U[*β, ω*] for 0 ≤ *β, ω* ≤ 1. We argue that from the perspective of deriving Monte Carlo gradient estimates, these schedules are equivalent to a continuous schedule where the mask probability is approximately 0 before the specified range such that 1 − α<β ≈ ϵ and approximately 1 after the specified range 1 − α>ω ≈ 1 − ϵ. Consequently, α
′t is linear within the range: α
′t ≈ 1/(β − ω).

## 5.3 Data-Driven Clipped Schedules Across Block Sizes

As the optimal mask rates may differ depending on the block size L
′, we adaptively learn the schedule during training. While Kingma et al. (2021) perform variance minimization by isolating a variance term using their squared diffusion loss, this strategy is not directly applicable to our variance estimator in Equation 10 since we seek to reduce variance across random batches in addition to random tb.

Instead, we optimize parameters *β, ω* to directly minimize training variance. To limit the computational burden of the optimization, we use the variance of the estimator of the diffusion ELBO as a proxy for the gradient estimator to optimize *β, ω*: minβ,ω VarX,t [L(X; *θ, β, ω*)]. We perform a grid search at regular intervals during training to find the optimal *β, ω* (experimental details in Sec. 6). In Table 2, we show that variance of the diffusion NELBO is correlated with test perplexity. Under a range of "clipped" noise rate distributions, we find that there exists a unique distribution for each block size L
′ ∈ {4, 16, 128} that minimizes both the variance of the NELBO and the test perplexity.

Table 2: Perplexities (PPLs; ↓) and variances of the NELBO VarX,t [LBD(X; θ)] (Var. NELBO; ↓).

Models are trained on LM1B using a linear schedule for 65B tokens, then finetuned for 10B tokens.

U[0, .5] U*[.3, .*8] U[.5, 1] U[0, 1]

L

′PPL Var. NELBO PPL Var. NELBO PPL Var. NELBO PPL Var. NELBO

128 **31.72 1.03** 31.78 1.35 31.92 1.83 31.78 3.80

16 31.27 7.90 **31.19 3.62** 31.29 3.63 31.33 7.39

4 29.23 32.68 29.37 10.39 **29.16 8.28** 29.23 23.65

## 6 Experiments

We evaluate BD3-LMs across standard language modeling benchmarks and demonstrate their ability to generate arbitrary-length sequences unconditionally. We pre-train a base BD3-LM using the maximum block size L
′ = L for 850K gradient steps and fine-tune under varying L
′for 150K gradient steps on the One Billion Words dataset (LM1B; Chelba et al. (2014)) and OpenWebText (OWT; Gokaslan et al. (2019)). Details on training and inference are provided in Suppl C.

To reduce the variance of training on the diffusion NELBO, we adaptively learn the range of masking rates by optimizing parameters *β, ω* as described in Section 5.3. In practice, we do so using a grid search during every validation epoch (after ∼5K gradient

| diffusion value is bolded.                           | PPL (↓)   |
|------------------------------------------------------|-----------|
| Autoregressive Transformer-X Base (Dai et al., 2019) | 23.5      |
| Transformer (Sahoo et al., 2024a)                    | 22.83     |
| Diffusion D3PM (absorb) (Austin et al., 2021)        | ≤ 82.34   |
| SEDD (Lou et al., 2024)                              | ≤ 32.68   |
| MDLM (Sahoo et al., 2024a)                           | ≤ 31.78   |
| Block diffusion (Ours) ′ = 16                        | ≤ 30.60   |
| BD3-LMs L ′ = 8                                      | ≤ 29.83   |
| L ′ = 4                                              | ≤ 28.23   |
| L                                                    |           |

updates) to identify *β, ω*: minβ,ω VarX,t [L(X; *θ, β, ω*)]. During evaluation, we report likelihood under uniformly sampled mask rates (8) as in Austin et al. (2021); Sahoo et al. (2024a).

## 6.1 Likelihood Evaluation

Table 4: Test perplexities (PPL; ↓) on OWT for models trained for 524B tokens. Best diffusion value is bolded.

On LM1B, BD3-LMs outperform all prior diffusion methods in Table 3. Compared to MDLM (Sahoo et al., 2024a), BD3-LMs achieve up to 13% improvement in perplexity. We observe a similar trend on OpenWebText in Table 4. We also evaluate the ability of BD3-LMs to generalize to unseen datasets in a zero-shot setting, following the benchmark from Radford et al. (2019). We evaluate the likelihood of models trained with OWT on datasets Penn Tree Bank (PTB; (Marcus et al., 1993)), Wikitext (Merity et al., 2016), LM1B, Lambada (Paperno et al., 2016), AG News (Zhang et al., 2015), and Scientific Papers (Pubmed and Arxiv subsets; (Cohan et al., 2018)). In Table 5, BD3-
LM achieves the best zero-shot perplexity on Pubmed, surpassing AR, and the best perplexity among diffusion models on Wikitext, LM1B, and AG News.

| PPL (↓)                    |         |
|----------------------------|---------|
| AR (Sahoo et al., 2024a)   | 17.54   |
| SEDD (Lou et al., 2024)    | ≤ 24.10 |
| MDLM (Sahoo et al., 2024a) | ≤ 22.98 |
| BD3-LMs L ′ = 16           | ≤ 22.27 |
| L ′ = 8                    | ≤ 21.68 |
| L ′ = 4                    | ≤ 20.73 |

Table 5: Zero-shot validation perplexities (↓) of models trained for 524B tokens on OWT. All perplexities for diffusion models are upper bounds.

| PTB            | Wikitext   | LM1B   | Lambada   | AG News   | Pubmed   | Arxiv   |       |
|----------------|------------|--------|-----------|-----------|----------|---------|-------|
| AR             | 81.07      | 25.32  | 51.14     | 52.13     | 52.11    | 48.59   | 41.22 |
| SEDD           | 96.33      | 35.98  | 68.14     | 48.93     | 67.82    | 45.39   | 40.03 |
| MDLM           | 90.96      | 33.22  | 64.94     | 48.29     | 62.78    | 43.13   | 37.89 |
| BD3-LM L ′ = 4 | 96.81      | 31.31  | 60.88     | 50.03     | 61.67    | 42.52   | 39.20 |

## 6.2 Sample Quality And Variable-Length Sequence Generation

Table 6: Generation length statistics from sampling 500 documents from models trained on OWT.

| Median            | Max   |      |
|-------------------|-------|------|
| # tokens # tokens |       |      |
| OWT train set     | 717   | 131K |
| AR                | 4008  | 131K |
| SEDD              | 1021  | 1024 |
| BD3-LM L ′ = 16   | 798   | 9982 |

One key drawback of many existing diffusion language models (e.g,. Austin et al. (2021); Lou et al. (2024)) is that they cannot generate full-length sequences that are longer than the length of the output context chosen at training time. The OWT dataset is useful for examining this limitation, as it contains many documents that are longer than the training context length of 1024 tokens. We record generation length statistics of 500 variable-length samples in Table 6. We continue sampling tokens until an end-of-sequence token [EOS] is generated or sample quality significantly degrades (as measured by sample entropy).

BD3-LMs generate sequences up to ≈10× longer than those of SEDD (Lou et al., 2024), which is restricted to the training context size. We also examine the sample quality of BD3-LMs through quantitative and qualitative analyses. In Table 7, we generate sequences of lengths L = 1024, 2048 and measure their generative perplexity under GPT2-Large. To sample L = 2048 tokens from MDLM, we use their block-wise decoding technique (which does not feature block diffusion training as in BD3-LMs).

We also compare to SSD-LM (Han et al., 2022), an alternative block diffusion formulation. Unlike our discrete diffusion framework, SSD-LM uses Gaussian diffusion and does not support likelihood estimation. Further, BD3-LM adopts an efficient sampler from masked diffusion, where the number of generation steps (NFEs) is upper-bounded by L since tokens are never remasked (Sahoo et al., 2024a; Ou et al., 2025). For SSD-LM, we compare sample quality using T = 1K diffusion steps per block, matching their experimental setting (yielding ≥40K NFEs), and T = 25 where NFEs are comparable across methods.

| L = 1024                        | L = 2048   |       |          |      |
|---------------------------------|------------|-------|----------|------|
| Model                           | Gen. PPL   | NFEs  | Gen. PPL | NFEs |
| AR                              | 14.1       | 1K    | 13.2     | 2K   |
| Diffusion SEDD                  | 52.0       | 1K    | -        | -    |
| MDLM                            | 46.8       | 1K    | 41.3     | 2K   |
| Block Diffusion SSD-LM L ′ = 25 | 37.2       | 40K   | 35.3     | 80K  |
| 281.3                           | 1K         | 281.9 | 2K       |      |
| BD3-LMs L ′ = 16                | 33.4       | 1K    | 31.5     | 2K   |
| L ′ = 8                         | 30.4       | 1K    | 28.2     | 2K   |
| L ′ = 4                         | 25.7       | 1K    | 23.6     | 2K   |

Table 7: Generative perplexity (Gen. PPL; ↓) and number of function evaluations (NFEs; ↓) of 300 samples of lengths L = 1024, 2048. All models are trained on OWT. AR, SEDD, MDLM, BD3-LMs use 110M parameters and are trained on 524B tokens, while SSD-LM uses 400M parameters and is pre-trained on 122B tokens. Best diffusion value is bolded. We provide further details in Suppl. C.5. BD3-LMs achieve the best generative perplexities compared to previous diffusion methods. Relative to SSD-LM, our discrete approach yields samples with improved generative perplexity using an order of magnitude fewer generation steps. We also qualitatively examine samples taken from BD3-LM and baselines (AR, MDLM) trained on the OWT dataset; we report samples in Suppl. D. We observe that BD3-LM samples have higher coherence than MDLM samples and approach the quality of AR.

## 6.3 Ablations

We assess the impact of the design choices in our proposed block diffusion recipes, namely 1)
selection of the noise schedule and 2) the efficiency improvement of the proposed training algorithm relative to a naive implementation.

## Selecting Noise Schedules To Reduce Training Variance

Relative to other standard noise schedules (Chang et al., 2022), "clipped" masking achieves the best performance. As heavier masking is effective for the smaller block size L
′ = 4, we compare with logarithmic and square root schedules that also encourage heavy masking. As lighter masking is optimal for L
′ = 16, we compare with square and cosine schedules.

## Efficiency Of Training Algorithm

In the BD3-LM training algorithm (Sec. 3.2), we compute xlogit using two options. We may perform two forward passes through the network (precomputing keys and values for the full sequence x, then computing denoised predictions), or combine these passes by concatenating the two inputs into the same attention kernel. We find that a single forward pass is more efficient as we reduce memory bandwidth bottlenecks by leveraging efficient attention kernels (Dao et al., 2022; Dong et al., 2024), see Suppl. B.7. Instead of paying the cost

Compared to the linear schedule used in Lou et al. (2024); Sahoo et al. (2024a), training under

"clipped" noise schedules is the most effective for reducing the training variance which correlates with

test perplexity. In Table 8, the ideal "clipped" masking rates, which are optimized during training, are specific to the block size and further motivate our optimization.

Table 8: Effect of the noise schedule on likelihood estimation. We finetune BD3-LMs on 3B tokens from LM1B and evaluate on a linear schedule. For clipped schedules, we compare optimal clipping for L

′ = 4, 16.

Noise schedule PPL Var. NELBO L' = 4 Clipped

U[0.45, 0.95] **29.21 6.24** U[0.3, 0.8] 29.38 10.33

Linear U[0, 1] 30.18 23.45 Logarithmic 30.36 23.53 Square root 31.41 26.43 L' = 16 Clipped

U[0*.45,* 0.95] 31.42 3.60 U[0.3, 0.8] **31.12 3.58**

Linear U[0, 1] 31.72 7.62 Square 31.43 13.03 Cosine 31.41 13.00

of two passes through the network, we only pay the cost of a more expensive attention operation. Our vectorized approach has 20-25% speed-up during training relative to performing two forward passes.

## 7 Discussion And Prior Work

Comparison to D3PM Block diffusion builds off D3PM (Austin et al., 2021) and applies it to each autoregressive conditional. We improve over D3PM in three ways: (1) we extend D3PM beyond fixed sequence lengths; (2) we study the perplexity gap of D3PM and AR models, identify gradient variance as a contributor, and design variance-minimizing schedules; (3) we improve over the perplexity of D3PM models. Our work applies to extensions of D3PM (He et al., 2022; Lou et al., 2024) including ones in continuous time (Campbell et al., 2022; Sun et al., 2022). Comparison to MDLM BD3-LMs further make use of the perplexity-enhancing improvements in MDLM (Sahoo et al., 2024a; Shi et al., 2024; Ou et al., 2025). We also build upon MDLM: (1) while Sahoo et al. (2024a) point out that their NELBO is invariant to the noise schedule, we show that the noise schedule has a significant effect on gradient variance; (2) we push the state-of-the-art in perplexity beyond MDLM. Note that our perplexity improvements stem not only from block diffusion, but also from optimized schedules, and could enhance standard MDLM and D3PM models.

Comparison to Gaussian Diffusion Alternatively, one may perform diffusion over continuous embeddings of discrete tokens (Li et al., 2022; Dieleman et al., 2022; Chen et al., 2022). This allows using algorithms for continuous data (Song et al., 2020; Ho & Salimans, 2022), but yields worse perplexity (Graves et al., 2023; Gulrajani & Hashimoto, 2024). Comparison to Semi-Autoregressive Diffusion Han et al. (2022; 2023) introduced a block formulation of Gaussian diffusion. BD3-LMs instead extend Austin et al. (2021), and feature: (1) tractable likelihood estimates for principled evaluation; (2) faster generation, as our number of model calls is bounded by the number of generated tokens, while SSD-LM performs orders of magnitude more calls; (3) improved sample quality. AR-Diffusion (Wu et al., 2023) extends SSD-LM with a left-to-right noise schedule; Chen et al. (2025); Ye et al. (2024) apply to decision traces and videos; Hao et al. (2024); Kong et al. (2025) extend to latent reasoning. PARD (Zhao et al., 2024) applies discrete block diffusion to graphs. In contrast, we (1) interpolate between AR/diffusion performance; (2) support KV caching; (3) perform attention within noised blocks, whereas PARD injects new empty blocks. Autoregressive diffusion models (Hoogeboom et al., 2021b;a) extend any-order AR models (AO- ARMs; Uria et al. (2014)) to support parallel sampling. Zheng et al. (2024) prove equivalence between MDLM and AO-ARM training. Further extensions of ARMs that compete with diffusion include iterative editing (Gu et al., 2019), parallel and speculative decoding (Gu et al., 2017; Santilli et al., 2023; Cai et al., 2024; Gloeckle et al., 2024), consistency training (Kou et al., 2024), guidance (Sanchez et al., 2023), and cross-modal extensions (Liu et al., 2023; Tian et al., 2025). Limitations Training BD3-LMs is more expensive than regular diffusion training. We propose a vectorized algorithm that keeps training speed within <2x of diffusion training speed; in our experiments, we also pre-train with a standard diffusion loss to further reduce the speed gap. Additionally, BD3-LMs generate blocks sequentially, and hence may face the same speed and controllability constraints as AR especially when blocks are small. Their optimal block size is task specific (e.g., larger for greater control). BD3-LMs are subject to inherent limitations of generative models, including hallucinations (Achiam et al., 2023), copyright infringement (Gokaslan et al., 2024), controllability (Schiff et al., 2024; Wang et al., 2023) and harmful outputs (Bai et al., 2022).

## 8 Conclusion

This work explores block diffusion and is motivated by two problems with existing discrete diffusion:
the need to generate arbitrary-length sequences and the perplexity gap to autoregressive models. We introduce BD3-LMs, which represent a block-wise extension of the D3PM framework (Austin et al., 2021), and leverage a specialized training algorithm and custom noise schedules that further improve performance. We observe that in addition to being able to generate long-form documents, these models also improve perplexity, setting a new state-of-the-art among discrete diffusion models. ACKNOWLEDGMENTS AND DISCLOSURE OF FUNDING
This work was partially funded by the National Science Foundation under awards DGE-1922551, CAREER awards 2046760 and 2145577, and by the National Institute of Health under award MIRA R35GM151243. Marianne Arriola is supported by a NSF Graduate Research Fellowship under award DGE-2139899 and a Hopper-Dean/Bowers CIS Deans Excellence Fellowship. We thank Databricks MosaicML for providing access to computational resources.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. *Advances in Neural Information Processing* Systems, 34:17981–17993, 2021.

Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk, and Jian Zhou. Dirichlet diffusion score model for biological sequence generation. In *International Conference on Machine Learning*, pp.

1276–1301. PMLR, 2023.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. *arXiv preprint arXiv:2212.08073*, 2022.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao.

Medusa: Simple llm inference acceleration framework with multiple decoding heads. *arXiv* preprint arXiv:2401.10774, 2024.

Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems, 35:28266–28279, 2022.

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11315–11325, 2022.

Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. One billion word benchmark for measuring progress in statistical language modeling, 2014.

Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann.

Diffusion forcing: Next-token prediction meets full-sequence diffusion. *Advances in Neural* Information Processing Systems, 37:24081–24125, 2025.

Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. *arXiv preprint arXiv:2208.04202*, 2022.

Arman Cohan, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Seokhwan Kim, Walter Chang, and Nazli Goharian. A discourse-aware attention model for abstractive summarization of long documents. *Proceedings of the 2018 Conference of the North American Chapter of the Association* for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), 2018.

doi: 10.18653/v1/n18-2097. URL http://dx.doi.org/10.18653/v1/n18-2097.

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860, 2019.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness. *Advances in Neural Information Processing Systems*, 35:16344–16359, 2022.

Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis, 2021. URL
https://arxiv.org/abs/2105.05233.

Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H
Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, et al. Continuous diffusion for categorical data. *arXiv preprint arXiv:2211.15089*, 2022.

Juechu Dong, Boyuan Feng, Driss Guessous, Yanbo Liang, and Horace He. Flex attention: A
programming model for generating optimized attention kernels. *arXiv preprint arXiv:2412.05496*, 2024.

Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, and Gabriel Synnaeve.

Better & faster large language models via multi-token prediction. *arXiv preprint arXiv:2404.19737*, 2024.

Shrey Goel, Vishrut Thoutam, Edgar Mariano Marroquin, Aaron Gokaslan, Arash Firouzbakht, Sophia Vincoff, Volodymyr Kuleshov, Huong T Kratochvil, and Pranam Chatterjee. Memdlm: De novo membrane protein design with masked discrete diffusion protein language models. *arXiv* preprint arXiv:2410.16735, 2024.

Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. Openwebtext corpus. http:
//Skylion007.github.io/OpenWebTextCorpus, 2019.

Aaron Gokaslan, A Feder Cooper, Jasmine Collins, Landan Seguin, Austin Jacobson, Mihir Patel, Jonathan Frankle, Cory Stephenson, and Volodymyr Kuleshov. Commoncanvas: Open diffusion models trained on creative-commons images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8250–8260, 2024.

Alex Graves, Rupesh Kumar Srivastava, Timothy Atkinson, and Faustino Gomez. Bayesian flow networks. *arXiv preprint arXiv:2308.07037*, 2023.

Jiatao Gu, James Bradbury, Caiming Xiong, Victor OK Li, and Richard Socher. Non-autoregressive neural machine translation. *arXiv preprint arXiv:1711.02281*, 2017.

Jiatao Gu, Changhan Wang, and Junbo Zhao. Levenshtein transformer. Advances in neural information processing systems, 32, 2019.

Ishaan Gulrajani and Tatsunori B Hashimoto. Likelihood-based diffusion language models. *Advances* in Neural Information Processing Systems, 36, 2024.

Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, and Jose Lezama. Photorealistic video generation with diffusion models, 2023. URL https:
//arxiv.org/abs/2312.06662.

Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. Ssd-lm: Semi-autoregressive simplex-based diffusion language model for text generation and modular control. *arXiv preprint arXiv:2210.17432*,
2022.

Xiaochuang Han, Sachin Kumar, Yulia Tsvetkov, and Marjan Ghazvininejad. David helps goliath:
Inference-time collaboration between small specialized and large general diffusion lms. *arXiv* preprint arXiv:2305.14771, 2023.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.

Zhengfu He, Tianxiang Sun, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu. Diffusionbert: Improving generative masked language models with diffusion models. arXiv preprint arXiv:2211.15029, 2022.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. *arXiv preprint arXiv:2207.12598*,
2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J
Fleet. Video diffusion models. *arXiv:2204.03458*, 2022.

Emiel Hoogeboom, Alexey A Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim Salimans. Autoregressive diffusion models. *arXiv preprint arXiv:2110.02037*, 2021a.

Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. Advances in Neural Information Processing Systems, 34:12454–12465, 2021b.

Daniel Israel, Aditya Grover, and Guy Van den Broeck. Enabling autoregressive models to fill in masked tokens. *arXiv preprint arXiv:2502.06901*, 2025.

Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. Advances in neural information processing systems, 34:21696–21707, 2021.

Deqian Kong, Minglu Zhao, Dehong Xu, Bo Pang, Shu Wang, Edouardo Honig, Zhangzhang Si, Chuan Li, Jianwen Xie, Sirui Xie, et al. Scalable language models with posterior inference of latent thought vectors. *arXiv preprint arXiv:2502.01567*, 2025.

Siqi Kou, Lanxiang Hu, Zhezhi He, Zhijie Deng, and Hao Zhang. CLLMs: Consistency large language models. In *Forty-first International Conference on Machine Learning*, 2024. URL https://openreview.net/forum?id=8uzBOVmh8H.

Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Diffusion-lm improves controllable text generation. *Advances in Neural Information Processing Systems*, 35: 4328–4343, 2022.

Xiner Li, Yulai Zhao, Chenyu Wang, Gabriele Scalia, Gokcen Eraslan, Surag Nair, Tommaso Biancalani, Shuiwang Ji, Aviv Regev, Sergey Levine, et al. Derivative-free guidance in continuous and discrete diffusion models with soft value-based decoding. *arXiv preprint arXiv:2408.08252*,
2024.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. *Advances in* neural information processing systems, 36:34892–34916, 2023.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In *Forty-first International Conference on Machine Learning*, 2024. URL https://openreview.net/forum?id=CNicRIVIPA.

Mitch Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. Building a large annotated corpus of english: The penn treebank. *Computational linguistics*, 19(2):313–330, 1993.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016.

Hunter Nisonoff, Junhao Xiong, Stephan Allenspach, and Jennifer Listgarten. Unlocking guidance for discrete state-space diffusion and flow models. *arXiv preprint arXiv:2406.01572*, 2024.

Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. In *The Thirteenth International Conference on Learning Representations*, 2025. URL https://openreview.net/forum?id=sMyXP8Tanm.

Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. The LAMBADA dataset:
Word prediction requiring a broad discourse context. In *Proceedings of the 54th Annual Meeting* of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1525–1534, Berlin, Germany, August 2016. Association for Computational Linguistics. URL http://www. aclweb.org/anthology/P16-1144.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. *OpenAI blog*, 1(8):9, 2019.

Subham Sekhar Sahoo, Marianne Arriola, Aaron Gokaslan, Edgar Mariano Marroquin, Alexander M
Rush, Yair Schiff, Justin T Chiu, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. In *The Thirty-eighth Annual Conference on Neural Information Processing* Systems, 2024a. URL https://openreview.net/forum?id=L4uaAR4ArM.

Subham Sekhar Sahoo, Aaron Gokaslan, Christopher De Sa, and Volodymyr Kuleshov. Diffusion models with learned adaptive noise. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024b. URL https://openreview.net/forum?id=loMa99A4p8.

Subham Sekhar Sahoo, John Xavier Morris, Aaron Gokaslan, Srijeeta Biswas, Vitaly Shmatikov, and Volodymyr Kuleshov. Zero-order diffusion guidance for inverse problems, 2024c. URL https://openreview.net/forum?id=JBgBrnhLLL.

Guillaume Sanchez, Honglu Fan, Alexander Spangher, Elad Levi, Pawan Sasanka Ammanamanchi, and Stella Biderman. Stay on topic with classifier-free guidance. *arXiv preprint arXiv:2306.17806*, 2023.

Andrea Santilli, Silvio Severino, Emilian Postolache, Valentino Maiorca, Michele Mancusi, Riccardo Marin, and Emanuele Rodola. Accelerating transformer inference for translation via parallel decoding. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12336–12355, Toronto, Canada, July 2023. Association for Computational Linguistics. doi:
10.18653/v1/2023.acl-long.689. URL https://aclanthology.org/2023.acl-long.

689.

Yair Schiff, Subham Sekhar Sahoo, Hao Phung, Guanghan Wang, Sam Boshar, Hugo Dalla-torre, Bernardo P de Almeida, Alexander Rush, Thomas Pierrot, and Volodymyr Kuleshov. Simple guidance mechanisms for discrete diffusion models. *arXiv preprint arXiv:2412.10193*, 2024.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. In *The Thirty-eighth Annual Conference on Neural Information* Processing Systems, 2024. URL https://openreview.net/forum?id=xcqSOfHt4g.

Phillip Si, Allan Bishop, and Volodymyr Kuleshov. Autoregressive quantile flows for predictive uncertainty estimation. In *International Conference on Learning Representations*, 2022.

Phillip Si, Zeyi Chen, Subham Sekhar Sahoo, Yair Schiff, and Volodymyr Kuleshov. Semiautoregressive energy flows: exploring likelihood-free training of normalizing flows. In International Conference on Machine Learning, pp. 31732–31753. PMLR, 2023.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In *International conference on machine learning*, pp. 2256–2265. PMLR, 2015.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. *arXiv* preprint arXiv:2010.02502, 2020.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution.

Advances in neural information processing systems, 32, 2019.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. *arXiv preprint arXiv:2104.09864*, 2021.

Haoran Sun, Lijun Yu, Bo Dai, Dale Schuurmans, and Hanjun Dai. Score-based continuous-time discrete diffusion models. *arXiv preprint arXiv:2211.16750*, 2022.

Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling:
Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2025.

Benigno Uria, Iain Murray, and Hugo Larochelle. A deep and tractable density estimator. In International Conference on Machine Learning, pp. 467–475. PMLR, 2014.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing* systems, 30, 2017.

Yingheng Wang, Yair Schiff, Aaron Gokaslan, Weishen Pan, Fei Wang, Christopher De Sa, and Volodymyr Kuleshov. InfoDiffusion: Representation learning using information maximizing diffusion models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of *Proceedings of Machine Learning Research*, pp. 36336–36354. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/wang23ah.html.

Tong Wu, Zhihao Fan, Xiao Liu, Hai-Tao Zheng, Yeyun Gong, yelong shen, Jian Jiao, Juntao Li, zhongyu wei, Jian Guo, Nan Duan, and Weizhu Chen. AR-diffusion: Auto-regressive diffusion model for text generation. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. URL https://openreview.net/forum?id=0EG6qUQ4xE.

Jiacheng Ye, Shansan Gong, Liheng Chen, Lin Zheng, Jiahui Gao, Han Shi, Chuan Wu, Xin Jiang, Zhenguo Li, Wei Bi, et al. Diffusion of thoughts: Chain-of-thought reasoning in diffusion language models. *arXiv preprint arXiv:2402.07754*, 2024.

Xiang Zhang, Junbo Jake Zhao, and Yann LeCun. Character-level convolutional networks for text classification. In *NIPS*, 2015.

Lingxiao Zhao, Xueying Ding, and Leman Akoglu. Pard: Permutation-invariant autoregressive diffusion for graph generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=x4Kk4FxLs3.

Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. *arXiv preprint arXiv:2409.02908*, 2024.

## Contents

1 Introduction 1 2 Background: Language Modeling Paradigms 2 2.1 Autoregressive Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3 2.2 Discrete Denoising Diffusion Probabilistic Models . . . . . . . . . . . . . . . . . 3 3 Block Diffusion Language Modeling 3 3.1 Block Diffusion Distributions and Model Architectures . . . . . . . . . . . . . . . 3

| 3.2   | Efficient Training and Sampling Algorithms    | 4   |
|-------|-----------------------------------------------|-----|

| 4   | Understanding Likelihood Gaps Between Diffusion & AR Models   | 5   |    |
|-----|---------------------------------------------------------------|-----|----|
| 4.1 | Masked BD3-LMs                                                |     | 5  |
| 4.2 | Case Study: Single Token Generation                           |     | 5  |

| 4.3   | Diffusion Gap from High Variance Training    | 6   |
|-------|----------------------------------------------|-----|

| 5   | Low-Variance Noise Schedules for BD3-LMs         | 6   |    |
|-----|--------------------------------------------------|-----|----|
| 5.1 | Intuition: Avoid Extreme Mask Rates              |     | 6  |
| 5.2 | Clipped Schedules for Low-Variance Gradients     |     | 7  |
| 5.3 | Data-Driven Clipped Schedules Across Block Sizes | 7   |    |

| 6   | Experiments                                            | 7   |    |
|-----|--------------------------------------------------------|-----|----|
| 6.1 | Likelihood Evaluation                                  | 8   |    |
| 6.2 | Sample Quality and Variable-Length Sequence Generation |     | 8  |

| 6.3   | Ablations   | 9   |
|-------|-------------|-----|

7 Discussion and Prior Work 10 8 Conclusion 10 A Block Diffusion NELBO 17

B Masked BD3-LMs 17

| B.1   | Forward Process                                               | 18   |    |
|-------|---------------------------------------------------------------|------|----|
| B.2   | Reverse Process                                               | 18   |    |
| B.3   | Simplified NELBO for Masked Diffusion Processes               | 18   |    |
| B.4   | Recovering the NLL from the NELBO for Single Token Generation |      | 19 |
| B.5   | Tightness of the NELBO                                        |      | 20 |

B.6 Specialized Attention Masks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

| B.7   | Optimized Attention Kernel with FlexAttention   | 21   |
|-------|-------------------------------------------------|------|

C Experimental Details 24

| C.1   | Datasets              | 24   |    |
|-------|-----------------------|------|----|
| C.2   | Architecture          | 24   |    |
| C.3   | Training              | 24   |    |
| C.4   | Likelihood Evaluation | 24   |    |
| C.5   | Inference             |      | 25 |

D Samples 26

## A Block Diffusion Nelbo

Below, we provide the Negative ELBO (NELBO) for the block diffusion parameterization. Recall that the sequence x 1:L =-x 1*, . . . ,* x Lis factorized over B blocks, which we refer to as x for simplicity, drawn from the data distribution q(x). Specifically, we will factorize the likelihood over B blocks of length L
′, then perform diffusion in each block over T discretization steps. Let DKL[·] to denote the Kullback-Leibler divergence, *t, s* be shorthand for t(i) = i/T and s(i) = (i − 1)/T ∀i ∈ [1, T]. We derive the NELBO as follows:

− log pθ(x) = −X B b=1 log pθ(x b|x <b) = −X B b=1 log Eq pθ(x b t(1):t(T) |x <b) q(x b t(1):t(T) |xb) b=1 log Eq pθ(x b t(T) |x <b)QT i=1 pθ(x b s(i) |x b t(i) , x <b) QT i=1 q(x bt(i) |x bs(i) ) = −X B ≤X B b=1 −Eq log pθ(x b|x b t= 1T , x <b) | {z } Lrecons + Et∈{ 2 T ,..., T−1 T,1} EqTDKL q(x bs |x b t, x b) ∥ pθ(x bs |x bt, x <b) | {z } Ldiffusion (11) + DKL q(x bt=1|x b) ∥ pθ(x bt=1) | {z } Lprior
$$(11)$$

## B Masked Bd3-Lms

We explore a specific class of block diffusion models that builds upon the masked diffusion language modeling framework. In particular, we focus on masking diffusion processes introduced by Austin et al. (2021) and derive a simplified NELBO under this framework as proposed by Sahoo et al. (2024a); Shi et al. (2024); Ou et al. (2025).

First, we define the diffusion matrix Qt for states i ∈ {1*, . . . , V* }. Consider the noise schedule function αt ∈ [0, 1], which is a strictly decreasing function in t satisfying α0 = 1 and α1 = 0.

Denote the mask index as m = V . The diffusion matrix is defined by Austin et al. (2021) as:

$$[Q_{t}]_{i j}=\left\{\begin{array}{l l}{{1}}&{{\quad\mathrm{if}\;i=j=m}}\\ {{\alpha_{t}}}&{{\quad\mathrm{if}\;i=j\neq m}}\\ {{1-\alpha_{t}}}&{{\mathrm{if}\;j=m,i\neq m}}\end{array}\right.$$
$$(12)$$

The diffusion matrix for the forward marginal Qt|s is:

$$[Q_{t|s}]_{i j}=\left\{\begin{array}{l l}{{1}}&{{\mathrm{if}\ i=j=m}}\\ {{\alpha_{t|s}}}&{{\mathrm{if}\ i=j\neq m}}\\ {{1-\alpha_{t|s}}}&{{\mathrm{if}\ j=m,i\neq m}}\end{array}\right.$$

$$(13)$$

where αt|s = αt/αs.

B.1 FORWARD PROCESS Under the D3PM framework (Austin et al., 2021), the forward noise process applied independently for each token ℓ ∈ {1*, . . . L*} is defined using diffusion matrices Qt ∈ R
V ×Vas

$$q(\mathbf{x}_{t}^{\ell}|\mathbf{x}^{\ell})=\mathrm{{Cat}}\left(\mathbf{x}_{t}^{\ell};\overline{{{Q}}}_{t}\mathbf{x}^{\ell}\right),\quad\mathrm{with}\quad\overline{{{Q}}}_{t(i)}=Q_{t(1)}Q_{t(2)}\ldots Q_{t(i)}$$

## B.2 Reverse Process

Let Qt|s denote the diffusion matrix for the forward marginal. We obtain the reverse posterior q(x ℓs | x ℓ t
, x ℓ) using the diffusion matrices:

$$q(\mathbf{x}_{s}^{\ell}|\mathbf{x}_{t}^{\ell},\mathbf{x}^{\ell})={\frac{q(\mathbf{x}_{t}^{\ell}|\mathbf{x}_{s}^{\ell},\mathbf{x}^{\ell})q(\mathbf{x}_{s}^{\ell}|\mathbf{x}^{\ell})}{q(\mathbf{x}_{t}^{\ell}|\mathbf{x}^{\ell})}}=\mathrm{{Cat}}\left(\mathbf{x}_{s}^{\ell};{\frac{Q_{t i s}\mathbf{x}_{t}^{\ell}\odot Q_{s}^{\top}\mathbf{x}^{\ell}}{(\mathbf{x}_{t}^{\ell})^{\top}Q_{t}^{\top}\mathbf{x}^{\ell}}}\right)$$
$$(14)$$
$$(15)$$

where ⊙ denotes the Hadmard product between two vectors. B.3 SIMPLIFIED NELBO FOR MASKED DIFFUSION PROCESSES Following Sahoo et al. (2024a); Shi et al. (2024); Ou et al. (2025), we simplify the NELBO in the case of masked diffusion processes. Below, we provide the outline of the NELBO derivation; see the full derivation in Sahoo et al. (2024a); Shi et al. (2024); Ou et al. (2025).

We will first focus on simplifying the diffusion loss term Ldiffusion in Eq. 11. We employ the SUBSparameterization proposed in Sahoo et al. (2024b) which simplifies the denoising model pθ for masked diffusion. In particular, we enforce the following constraints on the design of pθ by leveraging the fact that there only exists two possible states in the diffusion process x ℓt ∈ {x ℓ, m} ∀ℓ ∈ {1*, . . . , L*}.

1. **Zero Masking Probabilities**. We set pθ(x ℓ = m|x ℓt) = 0 (as the clean sequence x doesn't contain masks).

2. **Carry-Over Unmasking**. The true posterior for the case where x ℓt ̸= m is q(x ℓs = x ℓt |x ℓt ̸=
m) = 1 (if a token is unmasked in the reverse process, it is never remasked). Thus, we simplify the denoising model by setting pθ(x ℓs = x ℓ t |x ℓt ̸= m) = 1.

As a result, we will only approximate the posterior pθ(x ℓs = x ℓ|x ℓ t = m). Let x b,ℓ denote a token in the ℓ-th position in block b ∈ {1*, . . . , B*}. The diffusion loss term becomes:

Ldiffusion =X B b=1 EtEqT-DKL -q(x b s |x b t , x b)∥pθ(x b s |x b t , x <b) b=1 EtEqT  ℓ=1 DKL hq(x b,ℓ s |x b,ℓ t, x b,ℓ)∥pθ(x b,ℓ s |x bt, x <b) i  L X′ =X B DKL is simply the discrete-time diffusion loss for the block b; hence, from Sahoo et al. (2024a) (Suppl. B.1), we get:
DKL is simply the discrete-time diffusion loss for the block $\theta$,  $=\sum_{b=1}^{B}\mathbb{E}_t\mathbb{E}_qT\left[\sum_{\ell=1}^{L^{\prime}}\frac{\alpha_t-\alpha_s}{1-\alpha_t}\log p_\theta(\mathbf{x}^{b,\ell}\mid\mathbf{x}^{b,\ell}_t,\mathbf{x}^{<b})\right]$  $=\sum_{b=1}^{B}\mathbb{E}_t\mathbb{E}_qT\left[\frac{\alpha_t-\alpha_s}{1-\alpha_t}\log p_\theta(\mathbf{x}^b\mid\mathbf{x}^b_t,\mathbf{x}^{<b})\right]$
(16)
$$(16)$$
Lastly, we obtain a tighter approximation of the likelihood by taking the diffusion steps T → ∞
(Sahoo et al., 2024a), for which T(αt − αs) = α
′t:

$$\mathcal{L}_{\text{diffusion}}=\sum_{b=1}^{B}\mathbb{E}_{t\sim[0,1]}\mathbb{E}_{q}\left[\frac{\alpha_{t}^{t}}{1-\alpha_{t}}\log p_{\theta}(\mathbf{x}^{b}\mid\mathbf{x}_{t}^{b},\mathbf{x}^{<b})\right]\tag{17}$$  For the continuous time case, Sahoo et al. (2024a) (**Suppl. A.2.4**) show the reconstruction loss reduces to 0 as $\mathbf{x}_{(t)}^{b}\sim\lim_{T\rightarrow\infty}\text{Cat}\left(:\mathbf{x}_{t-\perp}^{b}\right)=\text{Cat}(.;\mathbf{x}^{b})$. Using this, we obtain:
$$\operatorname{t}\left(\,.;\mathbf{x}_{t={\frac{1}{T}}}^{b}\,\right)=\operatorname{Cat}(\,.;\mathbf{x}^{b}).$$
$$\mathcal{L}_{\text{recons}}=-\mathbb{E}_{q}\log p_{\theta}(\mathbf{x}^{b}|\mathbf{x}^{b}_{t(1)},\mathbf{x}^{<b})$$ $$=-\log p_{\theta}(\mathbf{x}^{b}|\mathbf{x}^{b}_{t(1)}=\mathbf{x}^{b},\mathbf{x}^{<b})$$ $$=0$$
$$(17)$$
$$(18)$$
$$(19)$$

The prior loss Lprior = DKL q(x bt=1|x b) ∥ pθ(x bt=1)also reduces to 0 because αt=1 = 0 which ensures q(x bt=1|x b) = Cat(.; m) and pθ(x b t=1) = Cat(.; m); see Sahoo et al. (2024a) (Suppl. A.2.4).

Finally, we obtain a simple objective that is a weighted average of cross-entropy terms:

$$\mathcal{L}_{\text{BD}}(\mathbf{x};\theta)=\sum_{b=1}^{B}\mathbb{E}_{t\sim[0,1]}\mathbb{E}_{q}\left[\frac{\alpha_{t}^{b}}{1-\alpha_{t}}\log p_{\theta}(\mathbf{x}^{b}\mid\mathbf{x}_{t}^{b},\mathbf{x}^{c_{b}})\right]\tag{19}$$  The above NELBO is invariant to the choice of noise schedule $\alpha_{t}$; see Sahoo et al. (2024a) (Suppl.  
E.1.1).

## B.4 Recovering The Nll From The Nelbo For Single Token Generation

Consider the block diffuson NELBO for a block size of 1 where L
′ = 1, B = L. The block diffusion NELBO is equivalent to the AR NLL when modeling a single token:

− log p(x) ≤X L b=1 Et∼[0,1]Eq α ′t 1 − αt log pθ(x b| x b t, x <b)  ∵ α ′ t = −1 and αt = 1 − t, = −X L b=1 Et∼[0,1]Eq 1 t log pθ(x b| x b t, x <b)  = −X L b=1 Et∼[0,1] 1 t Eq-log pθ(x b| x b t, x <b) Expanding Eq[.], = −X L b=1 Et∼[0,1] 1 t q(x b t = m|x b) log pθ(x b| x b t = m, x <b) + q(x b t = x b|x b) log pθ(x b| x b t = x b, x <b)
$$(20)$$
(20)
Recall that our denoising model employs the SUBS-parameterization proposed in Sahoo et al. (2024b).

The "carry-over unmasking" property ensures that log pθ(x b| x b t = x b, x
<b) = 0, as an unmasked token is simply copied over from from the input of the denoising model to the output. Hence, (20) reduces to following:

$$-\log p_{\theta}({\bf x})\leq-\sum_{b=1}^{L}\mathbb{E}_{t\sim[0,1]}\frac{1}{t}q({\bf x}_{t}^{b}={\bf m}|{\bf x}^{b})\log p_{\theta}({\bf x}^{b}\mid{\bf x}_{t}^{b}={\bf m},{\bf x}^{<b})$$ $$\therefore q({\bf x}_{t}^{b}={\bf m}|{\bf x}^{b})=t,\,\mbox{we get:}$$ $$=-\sum_{b=1}^{L}\mathbb{E}_{t\sim[0,1]}\log p_{\theta}({\bf x}^{b}\mid{\bf x}_{t}^{b}={\bf m},{\bf x}^{<b})$$
$$=-\sum_{b=1}^{L}\log p_{\theta}(\mathbf{x}^{b}\mid\mathbf{m},\mathbf{x}^{<b})$$
$$(21)$$
<b) (21)
For single-token generation (L
′ = 1) we recover the autoregressive NLL.

B.5 TIGHTNESS OF THE NELBO
For block sizes 1 ≤ K ≤ L, we show that -log p(x) ≤ LK ≤ LK+1. Consider K = 1, where we recover the autoregressive NLL (see Suppl B.4):

$$\mathcal{L}_{1}=\sum_{b=1}^{L}\log\mathbb{E}_{t\sim[0,1]}\mathbb{E}_{q}\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}p_{\theta}(\mathbf{x}^{b}\mid\mathbf{x}_{t}^{b},\mathbf{x}^{<b})$$ $$=-\sum_{b=1}^{L}\log p_{\theta}(\mathbf{x}^{b}\mid\mathbf{m},\mathbf{x}^{<b})\tag{2}$$

Consider the ELBO for block size K = 2:

$${\mathcal{L}}_{2}=\sum_{b=1}^{L/2}\log\mathbb{E}_{t\sim[0,1]}\mathbb{E}_{q}{\frac{\alpha_{t}^{\prime}}{1-\alpha_{t}}}p_{\theta}(\mathbf{x}^{b}\mid\mathbf{x}_{t}^{b},\mathbf{x}^{<b})$$
$${\mathrm{size~}}K=2{\mathrm{:}}$$

We show that L1 ≤ L2, and this holds for all 1 ≤ K ≤ L by induction. Let x b,ℓ correspond to the token in position ℓ ∈ [1, L′] of block b. We derive the below inequality:

b=1 log pθ(x b| m, x <b) = −X L/2 −X L b=1 log Et∼[0,1]Eq1 1 − αt pθ(x b| x b t , x <b) = −X L/2 b=1 log Et∼[0,1]Eq Y 2 1 1 − αt pθ(x b,ℓ | x b t, x <b) i=1 = −X L/2 b=1 logY 2 i=1 Et∼[0,1]Eq1 1 − αt pθ(x b,ℓ | x b t , x <b) ≤ −X L/2 b=1 X 2 i=1 log Et∼[0,1]Eq1 1 − αt pθ(x b,ℓ | x b t, x <b) (24)
$$(22)$$
$$(23)$$
B.6 SPECIALIZED ATTENTION MASKS
We aim to model conditional probabilities pθ(x b| x bt, x
<b) for all blocks b ∈ [1, B] simultaneously by designing an efficient training algorithm with our transformer backbone. However, modeling all B conditonal terms requires processing both the noised sequence x b tand the conditional context x
<b for all b. Rather than calling the denoising network B times, we process both sequences simultaneously by concatenating them xfull ← xt ⊕ x as input to a transformer. We update this sequence xfull of length 2L tokens using a custom attention mask Mfull ∈ {0, 1}
2L×2L for efficient training.

The full attention mask is comprised of four L × L smaller attention masks:

$$\mathcal{M}_{\mathrm{full}}=\begin{bmatrix}\mathcal{M}_{B D}&\mathcal{M}_{O B C}\\ \mathbf{0}&\mathcal{M}_{B C}\end{bmatrix}$$

where MBD and MOBC are used to update the representation of xt and MBC is used to update the representation of x. We define these masks as follows:
- MBD (Block-diagonal mask): Self-attention mask within noised blocks x bt

$$[{\mathcal{M}}_{B D}]_{i j}=\left\{\begin{array}{l l}{{1}}&{{\mathrm{if}\;i,j\;\mathrm{are~in~the~same~block}}}\\ {{0}}&{{\mathrm{otherwise}}}\end{array}\right.$$