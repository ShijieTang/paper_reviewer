# Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

Jaeyeon Kim \* 1 Kulin Shah \* 2 Vasilis Kontonis <sup>2</sup> Sham Kakade <sup>1</sup> Sitan Chen <sup>1</sup>

# Abstract

In recent years, masked diffusion models (MDMs) have emerged as a promising alternative approach for generative modeling over discrete domains. Compared to autoregressive models (ARMs), MDMs trade off complexity at training time with flexibility at inference time. At training time, they must learn to solve an exponentially large number of infilling problems, but at inference time, they can decode tokens in essentially arbitrary order. In this work, we closely examine these two competing effects. On the training front, we theoretically and empirically demonstrate that MDMs indeed train on computationally intractable subproblems compared to their autoregressive counterparts. On the inference front, we show that a suitable strategy for adaptively choosing the token decoding order significantly enhances the capabilities of MDMs, allowing them to sidestep hard subproblems. On logic puzzles like Sudoku, we show that adaptive inference can boost solving accuracy in pretrained MDMs from < 7% to ≈ 90%, even outperforming ARMs with 7× as many parameters and that were explicitly trained via teacher forcing to learn the right order of decoding.

# 1. Introduction

While diffusion models [\(Ho et al.,](#page-9-0) [2020;](#page-9-0) [Song et al.,](#page-10-0) [2021\)](#page-10-0) are now the dominant approach for generative modeling in continuous domains like image, video, and audio, efforts to extend this methodology to discrete domains like text and proteins [\(Austin et al.,](#page-8-0) [2021;](#page-8-0) [Lou et al.,](#page-9-1) [2024;](#page-9-1) [Hoogeboom](#page-9-2) [et al.,](#page-9-2) [2021b\)](#page-9-2) remain nascent. Among numerous proposals, masked diffusion models (MDMs) [\(Lou et al.,](#page-9-1) [2024;](#page-9-1) [Sahoo](#page-10-1) [et al.,](#page-10-1) [2025;](#page-10-1) [Shi et al.,](#page-10-2) [2024\)](#page-10-2) have emerged as a leading

variant, distinguished by a simple and principled objective: to generate samples, learn to reverse a noise process which independently and randomly masks tokens.

In many applications, such as language modeling, masked diffusion models (MDMs) still underperform compared to autoregressive models (ARMs) [\(Nie et al.,](#page-10-3) [2024;](#page-10-3) [Zheng](#page-11-0) [et al.,](#page-11-0) [2024\)](#page-11-0), which instead learn to reverse a noise process that unmasks tokens sequentially from left to right. However, recent studies suggest that MDMs may offer advantages in areas where ARMs fall short, including reasoning [\(Nie et al.,](#page-10-3) [2024;](#page-10-3) [Kitouni et al.,](#page-9-3) [2025\)](#page-9-3), planning [\(Ye et al.,](#page-10-4) [2024\)](#page-10-4), and infilling [\(Gong et al.,](#page-9-4) [2024\)](#page-9-4). This raises a key question: what are the strengths and limitations of MDMs compared to ARMs, and under what conditions can MDMs be scaled to challenge the dominance of ARMs in discrete generative modeling?

To understand these questions, we turn a microscope to two key competing factors when weighing the merits of MDMs over ARMs:

- Complexity at training time: By design, the prediction task that MDMs are trained on is more challenging. Whereas ARMs seek to predict the next token given an unmasked prefix, MDMs seek to predict a token conditioned on a set of unmasked tokens in arbitrary positions.
- Flexibility at inference time: On the other hand, the sampling paths taken by an MDM are less rigid. The order in which tokens are decoded at inference time is random instead of fixed to left-to-right. In fact, even more is possible: MDMs can actually be used to decode in *any order*.

Therefore, we ask:

*Are the benefits of inference flexibility for MDMs enough to outweigh the drawbacks of training complexity?*

In this work, we provide dual perspectives on this question.

(1) Training for the worst. First, we provide theoretical and empirical evidence that the overhead imposed by training complexity quantifiably impacts MDMs' performance.

<sup>\*</sup>Equal contribution <sup>1</sup>Harvard University <sup>2</sup>University of Texas Austin. Correspondence to: Kulin Shah <kulinshah@utexas.edu>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

the corresponding subproblems solved by MDMs are computationally intractable. We then show this imbalance in computational complexity across subproblems persists even in real-world text data (Fig. [2,](#page-3-0) left).

![](_page_1_Diagram_3.jpeg)

(2) Planning for the best. While the above might appear to be bad news for MDMs, in the second part of this paper we answer our guiding question in the affirmative by building upon the observation [\(Zheng et al.,](#page-11-1) [2023\)](#page-11-1) that MDMs which can perfectly solve all masking subproblems can be used to decode in *any* order.

In place of vanilla MDM inference whereby tokens are unmasked in random order, we consider *adaptive* strategies that carefully select which token to unmask next. Our key insight is that this adaptivity makes it possible to *sidestep* the hard subproblems from training (Fig. [1\)](#page-1-0). In fact, we find that even without modifying how MDMs are trained, the resulting models' logits contain enough information to determine the right order in which to unmask.

Our main empirical result is to show that the performance of MDMs pretrained on logic puzzle data dramatically improves when one goes from vanilla to adaptive inference. For example, on Sudoku puzzles, a simple adaptive strategy (Section [4.1\)](#page-6-0) improves the accuracy of MDMs from < 7% to almost 90%. Remarkably, this not only outperforms vanilla ARMs, but even bespoke ARMs trained to learn the right decoding order via supervised teacher forcing [\(Shah](#page-10-5) [et al.,](#page-10-5) [2024;](#page-10-5) [Lehnert et al.,](#page-9-5) [2024\)](#page-9-5) (Table [2\)](#page-6-1).

Organization. In Section [2,](#page-1-1) we provide preliminaries on MDMs and set notation. In Section [3,](#page-2-0) we examine MDM training and demonstrate the imbalance in computational intractability across subproblems. In Section [4,](#page-5-0) we consider adaptive inference in MDMs and investigate its impact on likelihood modeling across various tasks.

# 2. Masked Diffusion Models (MDM)

In this section, we explain the framework of Masked Diffusion Models [\(Shi et al.,](#page-10-2) [2024;](#page-10-2) [Sahoo et al.,](#page-10-1) [2025\)](#page-10-1) and its interpretation as an *order-agnostic learner*. MDMs gradually add noise to the true discrete data and learn the marginal distribution of the induced reverse process. Below, we formulate the forward and reverse processes for MDMs.

Let the distribution pdata on {1, . . . , m} <sup>L</sup> be the data distribution over sequences of length L and with vocabulary {1, . . . , m}. We use 0 to denote the "mask" token.

Forward process. For a given x<sup>0</sup> ∼ pdata and a noise level t ∈ [0, 1], the forward process x<sup>t</sup> ∼ qt|0(· | x0) is a coordinate-independent masking process via qt|0(xt|x0) = Q<sup>L</sup>−<sup>1</sup> <sup>i</sup>=0 qt|0(x i t |x i 0 ), where qt|0(x i t | x i 0 ) = Cat αte<sup>x</sup> 0 +(1−

Figure 1. (Top) MDM training can be seen as learning multiple masked prediction problems, where some are harder to learn, leading to performance imbalance (Section [3\)](#page-2-0). (Bottom) During inference, adaptive MDM can avoid difficult problem instances, improving performance (Section [4\)](#page-5-0).

αt)e<sup>0</sup> and α<sup>t</sup> is the predefined noise schedule satisfying α<sup>0</sup> ≈ 1, α<sup>1</sup> ≈ 0 and e<sup>x</sup> ∈ R <sup>m</sup>+1 denotes a one-hot vector corresponding to the value of token x i 0 . Cat(π) denotes the categorical distribution given by π ∈ ∆<sup>m</sup>. In other words, for each i-th coordinate, x i t is masked to the mask token 0 with probability 1 − α<sup>t</sup> and unchanged otherwise.

Reverse process. The reverse process of the above forward process is denoted using qs|t(xs|xt, x0) and is given by <sup>q</sup>s|t(xs|xt, x0) = Q<sup>L</sup>−<sup>1</sup> <sup>i</sup>=0 qs|t(x i s |xt, x0) for any s < t, where

$$q_{s|t}(x_s^i | x_t, x_0) = \begin{cases} \text{Cat}(\mathbf{e}_{x_t^i}) & x_t^i \neq 0 \\ \text{Cat}\left(\frac{1-\alpha_s}{1-\alpha_t} \mathbf{e}_m + \frac{\alpha_s-\alpha_t}{1-\alpha_t} \mathbf{e}_{x_0}\right) & x_t^i = 0. \end{cases}$$

The reverse transition probability qs|t(x i s |xt, x0) is approximated using gθ(x i s |xt) ≜ qs|t(x i s | xt, x<sup>0</sup> ← pθ(xt, t)) where pθ(xt, t) is a denoising network trained to predict the marginal on x<sup>0</sup> via an ELBO-based loss. To be precise, qs|<sup>t</sup> x i s | xt, x<sup>0</sup> ← pθ(xt, t) indicates the conditional probability where pθ(xt, t) is placed in the position of x<sup>0</sup>

within qs|t(x i s | xt, x0).

$$\mathcal{L}_\theta = \int_0^1 \frac{\alpha'_t}{1 - \alpha_t} \mathbb{E}_{x_0 \sim p_{\text{data}}} \sum_{x_t \sim q_{t|0}(\cdot|x_0)} -\log p_\theta(x_0^i|x_t, t) dt.$$

Here, α ′ <sup>t</sup> = dα<sup>t</sup> dt and δ<sup>x</sup>t,<sup>0</sup> is the indicator function; the summation is computed over coordinates i s.t. x i <sup>t</sup> = 0. In practice, a time-embedding-free architecture for the denoising network, i.e., pθ(xt, t) = pθ(xt), is usually employed as x<sup>t</sup> implicitly contains information about t via the number of masked tokens.

The reverse sampling process starts from the fully masked sentence x<sup>1</sup> = (0, . . . , 0). At a given noise level t ∈ (0, 1], suppose we have a partially masked sequence xt. For predetermined noise level s < t, we sample x<sup>s</sup> ∼ gθ(·|xt). This process is repeated recursively from t = 1 to t = 0.

## 2.1. Reformulating the training and inference of MDMs

In this section, we first discuss vanilla order-agnostic training of MDMs and compare it with "left-to-right" order training of autoregressive models in Section [2.1.1.](#page-2-1) Then, we reformulate vanilla MDM inference in Section [2.1.2](#page-2-2) to set the stage for the upcoming discussion.

#### 2.1.1. ORDER-AGNOSTIC TRAINING OF MDMS

Recent works [\(Zheng et al.,](#page-11-0) [2024;](#page-11-0) [Ou et al.,](#page-10-6) [2024\)](#page-10-6) have observed that the learning problem of MDM is equivalent to a masked language model. Building upon their analysis, we reformulate the loss L<sup>θ</sup> to show that L<sup>θ</sup> is a linear combination of the loss for all possible infilling masks. We first define x0[M] as a masked sequence, obtained from original sequence x<sup>0</sup> where indices in the mask set M (regarded as a subset of [L] ≜ {1, 2, . . . , L}) are replaced with mask token 0.

Proposition 2.1. *Assume* α<sup>0</sup> = 1*,* α<sup>1</sup> = 0 *and denoising network* p<sup>θ</sup> *is time-embedding free. Then* L<sup>θ</sup> ≤ −Ex0∼pdata [log pθ(x0)] *and*

$$\mathcal{L}_\theta = - \sum_{M \subseteq [L], i \in M} \frac{1}{|M|} \frac{1}{\binom{L}{|M|}} \mathbb{E}_{x_0 \sim p_{\text{data}}} [\log p_\theta(x_0^i | x_0[M])], \quad (1)$$

*where* |M| *is the size of the set* M *and* pθ(x<sup>i</sup> | x0[M]) *indicates the conditional probability of the* i*-th coordinate from* pθ(xt)*.*

The proof of the above proposition is given in Appendix [E.](#page-18-0) As the MDM loss is a linear combination of the loss for all possible infilling mask M, the minimizer of the loss L<sup>θ</sup> learns to solve *every* masking problem. In other words, the optimal predictor p<sup>θ</sup> is the posterior marginal of the i-th token, conditioned on x0[M] for all masks M. The training objective of MDM aims to predict x<sup>0</sup> from x0[M] across all

possible masks. Hence, we will refer to the MDM training as *order-agnostic* training.

On the other hand, Autoregressive Models (ARMs) learn to solve a smaller set of infilling problems (L infilling problems in ARMs as opposed to exp(L) infilling problems in MDM) by predicting i th token x <sup>i</sup> given all previous tokens x 0 , . . . , x<sup>i</sup>−<sup>1</sup> . This prediction problem is equivalent to predicting x <sup>i</sup> by masking at positions {i, . . . , L−1}. Therefore, we can write it as

$$\log p_\theta(x_0) = \sum_{i=0}^{L-1} \log p_\theta(x_0^i | x_0[\{i, \dots, L-1\}]). \quad (2)$$

ARMs are trained to predict tokens sequentially from left to right in all sequences. We refer to this as left-to-right training. In general, one can also consider predicting tokens sequentially under some *fixed, known* permutation of the sequence; we refer to this as *order-aware training*.

## 2.1.2. ORDER-AGNOSTIC INFERENCE OF MDMS

The MDM inference can be decomposed into two steps: (a) randomly selecting a set of positions to unmask and (b) assigning token values to each position via the denoising network pθ. More precisely, we can reformulate the reverse process x<sup>s</sup> ∼ gθ(·|xt) as follows.

## Vanilla MDM inference

(a) Sample a set of masked tokens S ⊆ {i | x

i <sup>t</sup> = 0},

<sup>P</sup>(i ∈ S) = <sup>α</sup>s−α<sup>t</sup>

1−α<sup>t</sup> .

(b) For each i ∈ S, sample x

i <sup>s</sup> ∼ pθ(x

i |xt).

Therefore, the inference in MDM is implemented by randomly selecting S and then filling each token value according to the posterior probability pθ(x i s |xt).

# 3. MDMs train on hard problems

In this section, we theoretically and empirically demonstrate that a large portion of masking subproblems pθ(x i 0 | x0[M]) can be difficult to learn. For intuition, consider solving a masked prediction problem pθ(x i | x0[M]) on text data like masking an arbitrary sentence in the middle of a document and predicting the correct word for a specific position in that sentence. It is reasonable that this task should be more complex, even for humans, than left-to-right prediction, and in this section, we place this intuition on a rigorous footing.

In Section [3.1,](#page-3-1) we show several examples of simple, non-pathological distributions for which: (1) the masking problems encountered during order-*aware* training are computationally tractable, yet (2) many of the ones en-

![](_page_3_Figure_1.jpeg)

Figure 2. Left: MDMs train on hard problems (Section [3.2\)](#page-4-0). x-axis and y-axis correspond to log(FLOPs) and − log pθ(x), respectively. MDM (Blue) is worse than ARM (Orange) in likelihood modeling. Most masking problems (Other lines) that MDM is trained on are harder than those encountered by ARM, as indicated by small log-likelihoods. Right: Task error imbalance (Section [3.3\)](#page-5-1). MDM's performance varies across different tasks. For text data (top right), this is indicated by validation loss. For L&O-NAE-SAT (bottom right), MDM performs well on the masking problems for observation positions (light region) but struggles with latent positions (dark region).

countered during order-agnostic training are computationally intractable. In Section [3.2,](#page-4-0) we empirically show that text data also exhibits this gap between the computational complexity of order-aware and order-agnostic training. In Section [3.3,](#page-5-1) we reveal that this discrepancy in computational complexity manifests empirically in performance imbalance across tasks: as predicted by the theory, MDMs trained on data from such distributions exhibits small errors on easy subproblems but suffers from large errors on harder ones.

## 3.1. Benign distributions with hard masking problems

We now describe a simple model of data under which we explore the computational complexity of masking problems.

Definition 3.1. A *latents-and-observations (L&O) distribution* is a data distribution pdata over sequence of length L with alphabet size m (precisely, pdata is over {0, . . . , m} L) is specified by a permutation π over indices {1, 2, . . . , L}, number of latent tokens N, number of observation tokens P such that N + P = L, prior distribution pprior of latent variables over {1, . . . , m} and efficiently learnable *observation functions* O1, . . . , O<sup>P</sup> : {1, . . . , m} <sup>N</sup> → ∆({0, . . . , m}), [1](#page-3-2)

- (Latent tokens) For i = 1, . . . , N, sample x π(i) independently from the prior distribution pprior of the latents.
- (Observation tokens) For j = 1, . . . , P, sample x π(N+j) independently from O<sup>j</sup> (x <sup>π</sup>(1), . . . , x<sup>π</sup>(N) ).

L&O distributions contain two types of tokens: (1) *latent tokens* and (2) *bservation tokens*. Intuitively, latent tokens are

tokens in the sequence, indexed by π(1), π(2), . . . , π(N) that serve as "seeds" that provide randomness in the sequence; the remaining tokens, called observation tokens (indexed by π(N + 1), π(N + 2), . . . , π(N + P)), are determined as (possibly randomized) functions of the latent tokens via O1, . . . , O<sup>P</sup> .

Note that by design, order-aware training, e.g. by permuting the sequence so that π becomes the identity permutation and then performing autoregressive training, is computationally tractable: predicting x <sup>π</sup>(i) given x <sup>π</sup>(1), . . . , x<sup>π</sup>(i−1) is trivial when i ≤ N as the tokens are independent, and computationally tractable when i > N because x <sup>π</sup>(i) only depends on x <sup>π</sup>(1), . . . , x<sup>π</sup>(N) and is efficiently learnable by assumption. In contrast, below we will show examples where if one performs order-agnostic training *a la `* MDMs, one will run into hard masking problems with high probability.

First note that if the observations (O1, . . . , O<sup>P</sup> ) are given by a cryptographic hash function, then the masking problem of predicting (x <sup>π</sup>(1), . . . , x<sup>π</sup>(L) ) given (x <sup>π</sup>(N+1), . . . , x<sup>π</sup>(N+<sup>P</sup> ) ) is computationally intractable by design because it requires inverting the hash function. While this is a well-known folklore observation regarding the role of token ordering in language modeling, it is not entirely satisfying because this construction is worst-case in nature – in real-world data, one rarely trains on sequences given by cryptographic hash functions. Furthermore, it only establishes hardness for a specific masking pattern which need not be encountered in the course of running the reverse process.

We provide several simple instances of L&O distributions that address these issues: instead of leveraging delicate cryptographic constructions, they are *average-case* in nature and furthermore we can establish hardness for *typical* masking

<sup>1</sup>Here *efficiently learnable* is in the standard PAC sense: given polynomially many examples of the form (z, y) where z ∼ π n and y ∼ O<sup>j</sup> (z), there is an efficient algorithm that can w.h.p. learn to approximate O<sup>j</sup> in expectation over π n .

problems encountered along the reverse process.

In all these examples, the hardness results we establish hold even if the algorithm knows all of the parameters of pdata as well as the observation functions O1, . . . , O<sup>P</sup> . Due to space constraints, here we focus on the following example, deferring two others to Apps. [B.1](#page-13-0) and [B.2.](#page-14-0)

Example 3.2 (Sparse predicate observations). *Consider the following class of L&O distributions. Given* arity k ≥ 2*, fix a* predicate *function* g : {1, . . . , m} <sup>k</sup> → {0, 1}*. Consider the set of all ordered subsets of* {1, 2, . . . , N} *of size* k *and set the total number of observation latents* P *equal to the size of this set (hence* P = N!/(N − k)! = N(N − 1)· · ·(N − k + 1)*). To sample a new sequence, we first sample latent tokens* x <sup>π</sup>(1), . . . , x<sup>π</sup>(N) *from the prior distribution* p*prior and an observation latent corresponding to a* k*-sized subset* S *is given by* g({x <sup>π</sup>(i)}i∈S)*. In other words, each observation latent corresponds to a* k*-sized subset* S *of* {1, 2, . . . , N} *and the corresponding observation function* OS(x <sup>π</sup>(1), . . . , x<sup>π</sup>(N) ) *is given by* g({x <sup>π</sup>(i)}i∈S)*.*

Proposition 3.3. *Let* x *be a sample from an L&O distribution* pdata *with sparse predicate observations as defined in Example [3.2,](#page-4-1) with arity* k *and predicate* g *satisfying Assumption [B.11,](#page-16-0) and let* γ *be the probability that* g *is satisfied by a random assignment from* {1, . . . , m} k *. Let* DKS *and* Dcond *be some constants associated with the predicate function* g *(see Definition [B.12\)](#page-16-1). Suppose each token in* x *is independently masked with probability* α*, and* M *is the set of indices for the masked tokens. If* 1 − γ <sup>−</sup><sup>1</sup>DKS/kN<sup>k</sup>−<sup>1</sup> ≤ α ≤ 1 − γ <sup>−</sup><sup>1</sup>Dcond/kN<sup>k</sup>−<sup>1</sup> *, then under the* 1RSB cavity prediction *(see Conjecture [B.13\)](#page-16-2), with probability* Ωk(1) *over the randomness of the masking, no polynomial-time algorithm can solve the resulting subproblem of predicting any of the masked tokens among* x <sup>π</sup>(1), . . . , x<sup>π</sup>(N) *given* x[M]*.*

The complete proof of the proposition is given in Appendix [B.4.](#page-16-3) We also provide a proof outline in Appendix [B.3](#page-14-1) for a comprehensive understanding.

## 3.2. Empirical evidence of hardness via likelihoods

Recent studies [\(Nie et al.,](#page-10-3) [2024;](#page-10-3) [Zheng et al.,](#page-11-0) [2024\)](#page-11-0) have shown that masked diffusion models (MDMs) underperform compared to autoregressive models (ARMs) on natural text data. In this section, we provide evidence that this performance gap is primarily due to the order-agnostic training of MDMs. Since natural text follows a left-to-right token order, we demonstrate that as training deviates from this order, model performance gradually deteriorates.

To understand the importance of the order during the training, we use the following setting: Given a permutation π of indices {0, 1, . . . , L − 1}, define a π*-learner* to be a

likelihood model log pθ(x0) given as follows:

$$\log p_\theta(x_0) = \sum_{i=0}^{L-1} \log p_\theta(x_0^{\pi(i)} | x_0[\pi\{i, \dots, L-1\}]) \quad (3)$$

In other words, the π-learner predicts the token at position π(i) given the clean tokens x π(0) 0 , . . . , x π(i−1) 0 and masked tokens x π(i) 0 , . . . , x π(L−1) . If π is the identity permutation, this reduces to the standard (left-to-right) autoregressive model. Note that the MDM loss encodes a π-learner for every permutation π because the MDM loss [\(1\)](#page-2-3) is equivalent to the average loss of those π-learners over π sampled from Unif(SL):

$$\mathcal{L}_\theta = -\mathbb{E}_{\pi, x_0 \sim p_{\text{data}}} \left[ \sum_{i=0}^{L-1} \log p_\theta \left( x_0^{\pi(i)} \middle| x_0[\pi\{i, \dots, L-1\}] \right) \right],$$

where S<sup>L</sup> denotes the set of all permutations over {0, 1, . . . , L − 1}. The proof of the above equivalence is given in Appendix [E.](#page-18-0) Therefore, by measuring the 'hardness' of each π-learner, we can probe differences in hardness between arbitrary masking problems and left-to-right masking problems.

Experimental setup. We use the Slimpajama dataset [\(Soboleva et al.,](#page-10-7) [2023\)](#page-10-7) to evaluate the performance of training in different orders. To train a π-learner, we employ a transformer with causal attention and use permuted data π(x0) as input. By varying π while maintaining all other training configurations (e.g., model, optimization), we can use the resulting likelihood (computed using Equation [\(3\)](#page-4-2)) as a metric to capture the hardness of subproblems solved by the π-learner.

In our experiments, the sequence length L is approximately 10<sup>3</sup> , so repeating the above for each π is infeasible. Instead, we sample π ∼ Unif(SL) and examine the scaling law of the π-learner's likelihood. We leverage the codebase from [\(Nie et al.,](#page-10-3) [2024\)](#page-10-3), where the baseline scaling laws of MDM and ARM were introduced. Moreover, given that RoPE has an inductive bias towards left-to-right ordering, we employ a learnable positional embedding layer for all experiments to correct this. Consequently, we also re-run the baseline results, where RoPE was employed. To investigate how the distance between π and the identity permutation affects the scaling law, we sample π from other distributions interpolating between Unif(SL) and the point mass at the identical permutation. Further experimental details are provided in Appendix [C.1.](#page-17-0)

Results. As shown in Fig. [2,](#page-3-0) the scaling law for a π-learner with uniformly random π is worse than that of an ARM. This elucidates the inherent hardness of masking problems pθ(x<sup>i</sup> | x0[M]) beyond left-to-right prediction and also explains why MDM, which is trained simultaneously on

all π ∈ <sup>S</sup>L, is worse than ARM in likelihood modeling. Additionally, as π gets closer to the identity permutation, the scaling laws also get closer to ARM (π-learner-closer and π-learner-much-closer in Fig. [2\)](#page-3-0). This also supports the common belief that ARM is a good fit for text data as it inherently follows a *left-to-right* ordering.

That said, it should also be noted that even though MDMs are trained on exponentially more masking problems than ARM (Θ(L2 <sup>L</sup>) versus L), its performance is not significantly worse than π-learners. We attribute this to the *blessing of task diversity*; multi-task training can benefit both the optimization dynamics [\(Kim et al.,](#page-9-6) [2024\)](#page-9-6) and validation performance [\(Tripuraneni et al.,](#page-10-8) [2021;](#page-10-8) [Maurer et al.,](#page-9-7) [2016;](#page-9-7) [Ruder,](#page-10-9) [2017\)](#page-10-9) due to positive transfers across tasks.

## 3.3. Error is imbalanced across masking problems

In previous sections, we have demonstrated that the hardness of different masking problems pθ(x i | x0[M]) can vary significantly, potentially hindering the MDM's learning. In this section, we provide empirical evidence that the MDM's final performance exhibits a similar imbalance across subproblems. Details are provided in App. [C.2.](#page-17-1)

L&O-NAE-SAT. Consider an L&O distribution with π given by the identity permutation and where each observation O<sup>j</sup> is deterministically given by NAE(x<sup>i</sup><sup>1</sup> , x<sup>i</sup><sup>2</sup> , x<sup>i</sup><sup>3</sup> ) ≜ 1 − 1[x<sup>i</sup><sup>1</sup> = x<sup>i</sup><sup>2</sup> = x<sup>i</sup><sup>3</sup> ] for some randomly chosen (prefixed) triples (i1, i2, i3) ∈ [N]. For an MDM trained on this distribution, we measure the error it achieves on each task log pθ(x0|x0[M]) via <sup>E</sup>x<sup>0</sup> log <sup>p</sup>θ(x0|x0[M]) <sup>−</sup> log pdata(x0|x0[M]) 2 , where pdata(x0|x0[M]) denotes the Bayes-optimal predictor. Technically, we do not have access to this, so instead we train another MDM for a much larger number of iterations and use this as a proxy. Fig. [2](#page-3-0) reveals that prediction tasks for latent positions (light region) exhibit larger errors compared to those for observation positions (dark region).

Text. Here we revisit the text experiment from Section [3.2.](#page-4-0) Since we do not have access to the Bayes-optimal predictor, we use the metric <sup>E</sup>x0∼pdata hPL−<sup>1</sup> <sup>i</sup>=0 log p<sup>θ</sup> x π(i) 0 <sup>x</sup>0[π{i, . . . , L <sup>−</sup> <sup>1</sup>}] i. This captures the accumulation of error across subproblems p<sup>θ</sup> x π(i) 0   x0[π{i, . . . , L − 1}] , since pθ(x0|x0[M]) = pdata(x0|x0[M]) minimizes this metric. Fig. [2](#page-3-0) shows a clear gap between different subproblems.

The theoretical and empirical evidence demonstrates that MDMs perform better in estimating pθ(x0|x0[M]) for some subproblems M than for others. We therefore want to avoid encountering hard subproblems M at inference time. In the next section, we show that while vanilla MDM inference can run into such subproblems, simple modifications at the inference stage can effectively circumvent these issues, resulting in dramatic, *training-free* performance improvements.

## 4. MDMs can plan around hard problems

We previously argued that due to the complex nature of masking subproblems, MDM must perform poorly on certain ones pθ(x i |xt). Therefore, during vanilla MDM inference, MDM inevitably encounters such difficult subproblems at Step (b). While this might suggest that we need to fundamentally revisit how MDMs are trained, in this section we show that, surprisingly, simple modifications at the inference stage—*without any further training*—can sidestep these issues and lead to significant performance improvements.

MDM offers multiple sampling paths. The vanilla MDM inference (Algorithm [1\)](#page-2-4) aim to align the intermediate distributions with the forward process, as used in continuous diffusion. However, unlike continuous diffusion, the reverse process of MDM allows multiple valid sampling paths (different orders of unmasking the tokens) that match the starting distribution of the forward process of MDM.

We first show that when we have an ideal MDM that perfectly solves all masking problems, i.e., pθ(x i |x0[M]) = pdata(x i 0 |x0[M]), then using any sampling path (unmasking the tokens in any order) results in the same distribution. Consider the following sampler: For every step, S is a set with one index selected agnostically (without following any distribution). For any clean sample x<sup>0</sup> generated by this sampler, note that pθ(x0) = Q<sup>L</sup>−<sup>1</sup> <sup>i</sup>=0 p<sup>θ</sup> x π(i) 0 x0[π{i, . . . , L − 1}] by chain rule, and this is equal to Q<sup>L</sup>−<sup>1</sup> <sup>i</sup>=0 <sup>p</sup>data x π(i) x0[π{i, . . . , L − 1}] = pdata(x0). Therefore, other choices of S, not necessarily following Algorithm [1,](#page-2-4) still capture the true likelihood.

In practice, unlike this ideal case, MDM does not perform equally well on all subproblems, as shown in Section [3.3.](#page-5-1) Consequently, different sampling paths result in varying likelihood modeling abilities. Motivated by this observation, we consider *adaptive inference for MDMs*:

## Adaptive MDM inference

- (a) Sample a set of masked tokens S = F (θ, xt) ⊆ {i | x i <sup>t</sup> = 0}.
- (b) For each i ∈ S, sample x i <sup>s</sup> ∼ pθ(x i |xt).

Instead of selecting S randomly, adaptive MDM inference leverages an oracle F(θ, xt) to select S strategically to

![](_page_6_Figure_1.jpeg)

Figure 3. Generative Perplexity. We compare the resulting generative perplexity (GenPPL) of adaptive vs. vanilla MDM inference. We employ a pretrained 170M MDM and LLaMA-7B [\(Touvron](#page-10-10) [et al.,](#page-10-10) [2023\)](#page-10-10) as inference and evaluation, respectively. Adaptive MDM inference (Blue) leads to a substantial reduction in generative perplexity, while maintaining the entropy.

avoid hard masking problems. This naturally raises the question of how to design an effective oracle F.

In the following sections, we demonstrate that adaptive MDM inference with careful choices of F enhance MDM's likelihood matching ability. In other words, a pretrained MDM, even if it performs poorly on certain hard subproblems, still contains sufficient information to avoid them when paired with an effective oracle F.

#### 4.1. Effective design of ordering oracle

We introduce two different oracles, Top-K and Top-K probability margin. Intuitively, both strategies are based on the idea that S should be selected based on how "certain" the model is about each position. We caution that these strategies should not be confused with notions like nucleus sampling in ARMs [\(Holtzman et al.,](#page-9-8) [2019\)](#page-9-8); the oracles we describe are for selecting the *position* of the next token to decode, rather than the *value*, and thus are only meaningful in the context of MDMs.

Table 1. L&O-NAE-SAT. Adaptive MDM inference achieves better likelihood matching than vanilla MDM inference. Note that naive guessing leads to 75% accuracy, indicating that vanilla inference performs similarly or worse than naive guessing.

| (    | N, P ) | Vanilla inference | Adaptive inference |
|------|--------|-------------------|--------------------|
| (25  | , 275) | 78.06%            | 93.76%             |
| (30  | , 270) | 75.70%            | 93.54%             |
| (40  | , 260) | 74.60%            | 92.21%             |
| (50  | , 250) | 67.94%            | 90.01%             |
| (100 | , 200) | 62.84%            | 88.91%             |

Table 2. Comparison of accuracy for solving the Sudoku puzzle.

| Method                    | # Param | Accuracy |
|---------------------------|---------|----------|
| ARM (w/o ordering)        | 42M     | 9.73%    |
| ARM (with ordering)       |         | 87.18%   |
| MDM (vanilla)             |         |          |
| MDM (Top- K probability)  |         | 18.51%   |
| MDM (Top- K prob. margin) |         | 89.49%   |

Top-K probability [\(Zheng et al.,](#page-11-1) [2023\)](#page-11-1). Suppose we want to unmask K positions at time step t, i.e., select |S| = K. In the Top-K strategy, the uncertainty of a position is estimated by the maximum probability assigned to any value in the vocabulary. More precisely, the certainty at position i is maxj∈{0,...,m−1} pθ(x <sup>i</sup> = j|xt) and F(θ, xt) = Top K max pθ(x i |xt) .

Top-K strategy is a good proxy for many tasks and works well in practice [\(Zheng et al.,](#page-11-1) [2023;](#page-11-1) [Ye et al.,](#page-10-4) [2024;](#page-10-4) [Wang](#page-10-11) [et al.,](#page-10-11) [2024\)](#page-10-11). However, this approach can often provide misleading estimates of uncertainty. Consider when an MDM is confused between two token values, thus assigning them almost equal but high probabilities. In this case, Top-K strategy may still choose to unmask this position, despite its uncertainty. To mitigate this issue, we propose the following alternative strategy.

Top-K probability margin. In this strategy, the uncertainty of a position is instead estimated using the absolute difference between the two most probable values at position i. More precisely, if j<sup>1</sup> and j<sup>2</sup> are the two most probable values in vocabulary according to pθ(x i |xt) in position i, the certainty in the position is given by |pθ(x <sup>i</sup> = j1|xt) − pθ(x <sup>i</sup> = j2|xt)| and F(θ, xt) = Top K |pθ(x <sup>i</sup> = j1|xt) − pθ(x <sup>i</sup> = j2|xt)| . When multiple values have similar probabilities at a position, Top-K probability margin will provide a better estimate of the uncertainty of a position, and when there is a single best choice of value then Top-K and Top-K probability margin work similarly.

#### 4.2. Adaptive MDM inference

In this section, we experimentally validate that adaptive MDM inference helps MDMs avoid hard subproblems, leading to better likelihood matching. We first show our results on L&O-NAE-SAT and text data, before turning to our primary application to logic puzzles.

L&O-NAE-SAT and text data. For the L&O-NAE-SAT distribution defined in Section [3.3,](#page-5-1) we evaluate the effectiveness of adaptive inference by measuring the accuracy in predicting the observation tokens. Table [1](#page-6-2) in the appendix

reveals a clear improvement over vanilla inference. For the text dataset, we evaluate using the standard metric of *generative perplexity*, by which likelihood is measured by a large language model. We also compute the entropy of the generated samples to ensure both inference strategies exhibit similar levels of diversity. As shown in Fig. [3,](#page-6-3) we observe a substantial decrease in generative perplexity using adaptive inference. We defer further experimental details to Appendix [D.1.](#page-18-1)

Logic puzzles. We consider two different types of logic puzzles: Sudoku and Zebra (Einstein) puzzles. Intuitively, for Sudoku, some empty (masked) cells are significantly easier to predict than others and we want to choose the cells that are easier to predict during the inference. We evaluate the effectiveness of adaptive MDM inference over vanilla MDM inference in selecting such cells.[<sup>2</sup>](#page-7-0)

To measure the performance of an inference method, we use the percentage of correctly solved puzzles. For both puzzles, we use train and test datasets from [\(Shah et al.,](#page-10-5) [2024\)](#page-10-5). For the Sudoku puzzle (Table [2\)](#page-6-1) we observe that adaptive MDM inference, in particular Top-K probability margin, obtains substantially higher accuracy (89.49%) compared to vanilla MDM inference (6.88%). Additionally, Top-K probability margin obtains higher accuracy (89.49%) than Top-K (18.51%). As mentioned in Section [4.1,](#page-6-0) this is because Top-K probability margin more reliably estimates uncertainty when multiple competing values are close in probability at a given position, as is often the case in Sudoku. For the Zebra puzzle, as shown in Table [3,](#page-7-1) we observe a consistent result: Top-K (98.5%) and Top-K probability margin (98.3%) outperform vanilla MDM inference (76.9%).

Table 3. Comparison of accuracy for solving the Zebra puzzle.

| Method                    | # Param | Accuracy |
|---------------------------|---------|----------|
| ARM (w/o ordering)        | 42M     | 80.31 %  |
| ARM (with ordering)       |         | 91.17 %  |
| MDM (vanilla)             |         |          |
|                           |         | 76.9 %   |
| MDM (Top- K probability)  |         | 98.5 %   |
| MDM (Top- K prob. margin) |         | 98.3 %   |

#### 4.3. Eliciting sequence-dependent reasoning paths using adaptive MDM inference in logic puzzles

In this section, we study the effectiveness of adaptive MDM inference in finding the right reasoning/generation order for tasks where every sequence has a different "natural" order. To do so, we will compare the performance of adaptive MDM inference to that of ARM on Sudoku and Zebra puzzles. For these puzzles, the natural order of generation is not only different from left-to-right, but it is also sequencedependent. For such tasks, prior works have shown that ARMs struggle if the information about the order is not provided during the training [\(Shah et al.,](#page-10-5) [2024;](#page-10-5) [Lehnert et al.,](#page-9-5) [2024\)](#page-9-5). Therefore, to obtain a strong baseline, we not only consider an ARM trained without the order information but also consider an ARM trained with the order information for each sequence in the training data. Note that the latter is a much stronger baseline than the former as one can hope to teach the model to figure out the correct order by some form of supervised teacher forcing (as performed in [Shah](#page-10-5) [et al.](#page-10-5) [\(2024\)](#page-10-5); [Lehnert et al.](#page-9-5) [\(2024\)](#page-9-5)), eliminating the issue of finding the right order in an unsupervised manner.

We compare ARMs and MDMs for Sudoku in Table [2](#page-6-1) and Zebra puzzles in Table [3.](#page-7-1) We observe that for both, Top-K probability margin-based adaptive MDM inference not only outperforms the ARM trained without ordering information, but it *even outperforms the ARM trained with ordering information*! This shows that the *unsupervised* way of finding the correct order and solving such logic puzzles using adaptive MDM inference outperforms the *supervised* way of finding the correct order and solving such puzzles using an ARM, and is significantly less computationally intensive.

## 4.4. Adaptive MDM inference on text benchmarks

To examine the effect of different inference strategies on text benchmarks, we adapted LLaDA, the 8B MDM model from [\(Nie et al.,](#page-10-12) [2025\)](#page-10-12). We compare three inference strategies: Vanilla, Top-K probability, and Top-K probability margin . The results are presented in Table [4.](#page-8-1)

We see that both adaptive MDM inference strategies, Top-K probability and Top-K probability margin , consistently outperform vanilla MDM inference. Notably, Top-K probability margin demonstrates a clear advantage over Top-K in challenging tasks like HumanEval-Multiline, HumanEval-Split Line, and Math. This is because Top-k Prob. Margin provides a more reliable estimate of uncertainty when multiple tokens have similar probabilities, a frequent occurrence in these difficult tasks. These results further underscore the potential for developing new, sophisticated adaptive inference strategies for various tasks.

# 4.5. Easy to hard generalization

In the previous section we showed that when the training and inference sequences come from the same distribution, order-agnostic training of MDMs combined with adaptive inference can perform very well on logic puzzles. To evaluate if the model has learned the correct way of solving the puzzles and test the robustness of adaptive inference, we

<sup>2</sup>A prior work [\(Ye et al.,](#page-10-4) [2024\)](#page-10-4) reported that a 6M MDM with Top-K inference achieves 100% accuracy on Sudoku. Given that a 6M MDM with Top-K only achieves 18.51% on our dataset (Table [2\)](#page-6-1), this suggests that the Sudoku dataset in [\(Ye et al.,](#page-10-4) [2024\)](#page-10-4) is significantly easier than ours.

Table 4. Performance of different inference strategies for LLaDa 8B model on coding and math tasks.

| Method       | HumanEval-Single | HumanEval-Multi | HumanEval-Split | Math  | MMLU  | ROCStories |
|--------------|------------------|-----------------|-----------------|-------|-------|------------|
| Vanilla      | 31.8%            | 16.5%           | 14.2%           | 28.5% | 33.2% | 21.23%     |
| Top-k        | 32.9%            | 20.8%           | 18.4%           | 31.3% | 36.5% | 21.10%     |
| Top-k Margin | 33.5%            | 25.4%           | 22.3%           | 34.3% | 35.4% | 21.41%     |

Table 5. Comparison of accuracy for solving the hard Sudokus.

| Method                    | # Param | Accuracy |
|---------------------------|---------|----------|
| ARM (with ordering)       | 42M     | 32.57 %  |
| MDM (random)              |         |          |
|                           |         | 3.62 %   |
| MDM (Top- K probability)  |         | 9.44 %   |
| MDM (Top- K prob. margin) |         | 49.88 %  |

also test the MDMs on harder puzzles than the ones from training, for Sudoku.

We keep the training dataset the same as proposed in [Shah](#page-10-5) [et al.](#page-10-5) [\(2024\)](#page-10-5). [Shah et al.](#page-10-5) [\(2024\)](#page-10-5) created this dataset from [Radcliffe](#page-10-13) [\(2020\)](#page-10-13) by selecting the puzzles that can be solved using 7 fixed strategies and do not require backtrackingbased search. We use the remaining puzzles in [Radcliffe](#page-10-13) [\(2020\)](#page-10-13) as our hard dataset. Hence, these puzzles all use a strategy not seen during training and/or backtracking to obtain the correct solution.

We measure the accuracy of MDMs and ARMs on the hard test set and present the results in Table [5.](#page-8-2) We see that the Top-K probability margin-based adaptive MDM inference strategy (49.88%) again significantly outperforms ARMs trained with order information (32.57%). In particular, although the accuracy drops for both methods due to the more challenging test set, MDMs with adaptive inference appear to be more robust to this distribution shift than ARMs. We believe this is due to the fact that MDMs try to solve a significantly higher number of infilling problems than ARMs (exp(L) compared to L) and therefore are able to extract knowledge about the problem more efficiently than ARMs.

# 5. Conclusion

In this work, we examined the impact of token ordering on training and inference in MDMs. We provided theoretical and experimental evidence that MDMs train on hard masking problems. We also demonstrated that adaptive inference strategies can be used to sidestep these hard problems. For logic puzzles, we find that this leads to dramatic improvements in performance not just over vanilla MDMs, but even over ARMs trained with teacher forcing to learn the right order of decoding.

An important direction for future work is to explore settings beyond logic puzzles where adaptive inference can help MDMs match or surpass ARMs. For these, it may be crucial to go beyond the relatively simple adaptive strategies like Top-K and Top-K probability margin considered here.

Acknowledgements. JK thanks Kiwhan Song for discussions about MDM training. KS and VK are supported by the NSF AI Institute for Foundations of Machine Learning (IFML). KS thanks Nishanth Dikkala for the initial discussions about the project. SC is supported by the Harvard Dean's Competitive Fund for Promising Scholarship and thanks Brice Huang and Sidhanth Mohanty for enlightening discussions about computational-statistical tradeoffs for planted CSPs.

# Impact statement

This paper advances the understanding of discrete diffusion models, contributing to the broader field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

# References


[1] Alaoui, A. E. and Gamarnik, D. Hardness of sampling solutions from the symmetric binary perceptron. *arXiv preprint arXiv:2407.16627*, 2024. Alekhnovich, M. More on average case vs approximation complexity. In *44th Annual IEEE Symposium on Foundations of Computer Science, 2003. Proceedings.*, pp. 298–307. IEEE, 2003. Aubin, B., Perkins, W., and Zdeborova, L. Storage capacity ´ in symmetric binary perceptrons. *Journal of Physics A: Mathematical and Theoretical*, 52(29):294003, 2019. Austin, J., Johnson, D. D., Ho, J., Tarlow, D., and van den Berg, R. Structured denoising diffusion models in discrete state-spaces. *NeruIPS*, 2021. Bormashenko, O. A coupling argument for the random transposition walk. *arXiv preprint arXiv: 1109.3915*, 2011.

[2] Chang, H., Zhang, H., Jiang, L., Liu, C., and Freeman, W. T. Maskgit: Masked generative image transformer. *CVPR*, 2022. Chen, H. and Ying, L. Convergence analysis of discrete diffusion model: Exact implementation through uniformization. *arXiv preprint arXiv: 2402.08095*, 2024. Chen, X., Chi, R. A., Wang, X., and Zhou, D. Premise order matters in reasoning with large language models. *arXiv preprint arXiv:2402.08939*, 2024. Decelle, A., Krzakala, F., Moore, C., and Zdeborova, L. ´ Asymptotic analysis of the stochastic block model for modular networks and its algorithmic applications. *Phys. Rev. E*, 84:066106, Dec 2011. Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pp. 4171–4186, 2019. Gamarnik, D. The overlap gap property: A topological barrier to optimizing over random structures. *Proceedings of the National Academy of Sciences*, 118(41):e2108492118, 2021. Golovneva, O., Allen-Zhu, Z., Weston, J., and Sukhbaatar,

[3] S. Reverse training to nurse the reversal curse. *arXiv preprint arXiv:2403.13799*, 2024. Gong, S., Agarwal, S., Zhang, Y., Ye, J., Zheng, L., Li, M., An, C., Zhao, P., Bi, W., Han, J., et al. Scaling diffusion language models via adaptation from autoregressive models. *arXiv preprint arXiv:2410.17891*, 2024. Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. *Advances in neural information processing systems*, 33:6840–6851, 2020. Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A., Welbl, J., Clark, A., et al. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556*, 2022. Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. The curious case of neural text degeneration. *arXiv preprint arXiv:1904.09751*, 2019. Hoogeboom, E., Gritsenko, A. A., Bastings, J., Poole, B., Berg, R. v. d., and Salimans, T. Autoregressive diffusion models. *arXiv preprint arXiv:2110.02037*, 2021a. Hoogeboom, E., Nielsen, D., Jaini, P., Forre, P., and Welling, ´
  - M. Argmax flows and multinomial diffusion: Learning categorical distributions. *NeurIPS*, 2021b. Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020. Kim, J., Kwon, S., Choi, J. Y., Park, J., Cho, J., Lee, J. D., and Ryu, E. K. Task diversity shortens the icl plateau. *arXiv preprint arXiv:2410.05448*, 2024. Kitouni, O., Nolte, N. S., Williams, A., Rabbat, M., Bouchacourt, D., and Ibrahim, M. The factorization curse: Which tokens you predict underlie the reversal curse and more. *Advances in Neural Information Processing Systems*, 37: 112329–112355, 2025. Krzakala, F. and Zdeborova, L. Hiding quiet solutions in ´ random constraint satisfaction problems. *Physical review letters*, 102(23):238701, 2009. Lehnert, L., Sukhbaatar, S., Su, D., Zheng, Q., McVay, P., Rabbat, M., and Tian, Y. Beyond a\*: Better planning with transformers via search dynamics bootstrapping. 2024. Liao, Y., Jiang, X., and Liu, Q. Probabilistically masked language model capable of autoregressive generation in arbitrary word order. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pp. 263–274. Association for Computational Linguistics, 2020. Liu, A., Broadrick, O., Niepert, M., and Broeck, G. V. d. Discrete copula diffusion. *arXiv preprint arXiv:2410.01949*, 2024a. Liu, S., Mohanty, S., and Raghavendra, P. On statistical inference when fixed points of belief propagation are unstable . In *2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS)*, pp. 395–405. IEEE Computer Society, 2022. Liu, S., Nam, J., Campbell, A., Stark, H., Xu, Y., Jaakkola, ¨ T., and Gomez-Bombarelli, R. Think while you generate: ´ Discrete diffusion with planned denoising. *arXiv preprint arXiv:2410.06264*, 2024b. Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*, 2017. Lou, A., Meng, C., and Ermon, S. Discrete diffusion modeling by estimating the ratios of the data distribution. *ICML*, 2024. Maurer, A., Pontil, M., and Romera-Paredes, B. The benefit of multitask representation learning. *JMLR*, 17(81):1–32, 2016.

[4] Montanari, A. Estimating random variables from random sparse observations. *European Transactions on Telecommunications*, 19(4):385–403, 2008. Nie, S., Zhu, F., Du, C., Pang, T., Liu, Q., Zeng, G., Lin, M., and Li, C. Scaling up masked diffusion models on text. *arXiv preprint arXiv:2410.18514*, 2024. Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., and Li, C. Large language diffusion models. *arXiv preprint arXiv:2502.09992*, 2025. Ou, J., Nie, S., Xue, K., Zhu, F., Sun, J., Li, Z., and Li,

[5] C. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. *arXiv preprint arXiv:2406.03736*, 2024. Papadopoulos, V., Wenger, J., and Hongler, C. Arrows of time for large language models. *arXiv preprint arXiv:2401.17505*, 2024. Peng, F. Z., Bezemek, Z., Patel, S., Yao, S., Rector-Brooks, J., Tong, A., and Chatterjee, P. Path planning for masked diffusion model sampling. *arXiv preprint arXiv:2502.03540*, 2025. Radcliffe, D. G. 3 million sudoku puzzles with ratings, 2020. URL [https://www.kaggle.com/](https://www.kaggle.com/dsv/1495975) [dsv/1495975](https://www.kaggle.com/dsv/1495975). Rector-Brooks, J., Hasan, M., Peng, Z., Quinn, Z., Liu, C., Mittal, S., Dziri, N., Bronstein, M., Bengio, Y., Chatterjee, P., et al. Steering masked discrete diffusion models via discrete denoising posterior prediction. *arXiv preprint arXiv:2410.08134*, 2024. Ruder, S. An overview of multi-task learning in deep neural networks. *arXiv 1706.05098*, 2017. Sahoo, S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J., Rush, A., and Kuleshov, V. Simple and effective masked diffusion language models. *Advances in Neural Information Processing Systems*, 37:130136– 130184, 2025. Schiff, Y., Sahoo, S. S., Phung, H., Wang, G., Boshar, S., Dalla-torre, H., de Almeida, B. P., Rush, A., Pierrot, T., and Kuleshov, V. Simple guidance mechanisms for discrete diffusion models. *arXiv preprint arXiv:2412.10193*, 2024. Shah, K., Dikkala, N., Wang, X., and Panigrahy, R. Causal language modeling can elicit search and reasoning capabilities on logic puzzles. *arXiv preprint arXiv:2409.10502*, 2024. Shi, J., Han, K., Wang, Z., Doucet, A., and Titsias, M. K. Simplified and generalized masked diffusion for discrete data. *NeurIPS*, 2024. Shih, A., Sadigh, D., and Ermon, S. Training and inference on any-order autoregressive models the right way. *NeurIPS*, 2022. Soboleva, D., Al-Khateeb, F., Myers, R., Steeves, J. R., Hestness, J., and Dey, N. Slimpajama: A 627b token cleaned and deduplicated version of redpajama, June 2023. Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. *ICML*, 2015. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. *ICLR*, 2021. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., Bikel, D., Blecher, L., Ferrer, C. C., Chen, M., Cucurull, G., Esiobu, D., Fernandes, J., Fu, J., Fu, W., Fuller, B., Gao, C., Goswami, V., Goyal, N., Hartshorn, A., Hosseini, S., Hou, R., Inan, H., Kardas, M., Kerkez, V., Khabsa, M., Kloumann, I., Korenev, A., Koura, P. S., Lachaux, M.-A., Lavril, T., Lee, J., Liskovich, D., Lu, Y., Mao, Y., Martinet, X., Mihaylov, T., Mishra, P., Molybog, I., Nie, Y., Poulton, A., Reizenstein, J., Rungta, R., Saladi, K., Schelten, A., Silva, R., Smith, E. M., Subramanian, R., Tan, X. E., Tang, B., Taylor, R., Williams, A., Kuan, J. X., Xu, P., Yan, Z., Zarov, I., Zhang, Y., Fan, A., Kambadur, M., Narang, S., Rodriguez, A., Stojnic, R., Edunov, S., and Scialom, T. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv: 2307.09288*, 2023. Tripuraneni, N., Jin, C., and Jordan, M. I. Provable metalearning of linear representations. *ICML*, 2021. Varma, H., Nagaraj, D., and Shanmugam, K. Glauber generative model: Discrete diffusion models via binary classification. *arXiv preprint arXiv: 2405.17035*, 2024. Wang, X., Zheng, Z., Ye, F., Xue, D., Huang, S., and Gu, Q. Diffusion language models are versatile protein learners. *ICML*, 2024. Xu, M., Geffner, T., Kreis, K., Nie, W., Xu, Y., Leskovec, J., Ermon, S., and Vahdat, A. Energy-based diffusion language models for text generation. *arxiv preprint arXiv: 2410.21357*, 2024. Ye, J., Gao, J., Gong, S., Zheng, L., Jiang, X., Li, Z., and Kong, L. Beyond autoregression: Discrete diffusion for complex reasoning and planning. *arXiv preprint arXiv: 2410.14157*, 2024. Zhang, P., Zeng, G., Wang, T., and Lu, W. Tinyllama: An open-source small language model. *arXiv preprint arXiv: 2401.02385*, 2024.

[6] Zheng, K., Chen, Y., Mao, H., Liu, M.-Y., Zhu, J., and Zhang, Q. Masked diffusion models are secretly timeagnostic masked models and exploit inaccurate categorical sampling. *arXiv preprint arXiv:2409.02908*, 2024.

[7] Zheng, L., Yuan, J., Yu, L., and Kong, L. A reparameterized discrete diffusion model for text generation. *arXiv preprint arXiv:2302.05737*, 2023.
# A. Related works

Discrete diffusion models. (Continuous) diffusion models were originally built on continuous-space Markov chains with Gaussian transition kernels [\(Sohl-Dickstein et al.,](#page-10-14) [2015;](#page-10-14) [Ho et al.,](#page-9-0) [2020\)](#page-9-0). This was later extended to continuous time through the theory of stochastic differential equations [\(Song et al.,](#page-10-0) [2021\)](#page-10-0). In a similar vein, discrete diffusion models have emerged from discrete-space Markov chains [\(Hoogeboom et al.,](#page-9-2) [2021b\)](#page-9-2). Specifically, [\(Austin et al.,](#page-8-0) [2021\)](#page-8-0) introduced D3PM with various types of transition matrices. Later, [Lou et al.](#page-9-1) [\(2024\)](#page-9-1) proposed SEDD, incorporating a theoretically and practically robust score-entropy objective. Additionally, [Varma et al.](#page-10-15) [\(2024\)](#page-10-15); [Liu et al.](#page-9-9) [\(2024b\)](#page-9-9) introduced novel modeling strategies that classify tokens in a noisy sequence as either signal (coming from clean data) or noise (arising from the forward process). In particular, [Liu et al.](#page-9-9) [\(2024b\)](#page-9-9) uses this to give a *planner* that adaptively determines which tokens to denoise. While this is similar in spirit to our general discussion about devising adaptive inference strategies, we emphasize that their approach is specific to discrete diffusions for which the forward process *scrambles* the token values, rather than masking them.

Masked diffusion models. Meanwhile, the absorbing transition kernel has gained popularity as a common choice due to its better performance than other kernels. Building on this, [Sahoo et al.](#page-10-1) [\(2025\)](#page-10-1); [Shi et al.](#page-10-2) [\(2024\)](#page-10-2) aligned its framework with continuous diffusion, resulting in a simple and principled training recipe, referring to it as *Masked Diffusion Model*. Subsequent studies have explored various aspects of MDM. [Gong et al.](#page-9-4) [\(2024\)](#page-9-4) efficiently trained MDM via adaptation from autoregressive models, scaling MDM up to 7B parameters. [Zheng et al.](#page-11-0) [\(2024\)](#page-11-0) interpreted MDMs as order-agnostic learners and proposed a first-hitting sampler based on this insight. [Ye et al.](#page-10-4) [\(2024\)](#page-10-4); [Gong et al.](#page-9-4) [\(2024\)](#page-9-4) demonstrated that MDM outperforms autoregressive models in reasoning and planning tasks, emphasizing its impact on downstream applications. [Nie et al.](#page-10-3) [\(2024\)](#page-10-3) examined the scaling laws of MDM, while [Xu et al.](#page-10-16) [\(2024\)](#page-10-16); [Liu et al.](#page-9-10) [\(2024a\)](#page-9-10) identified limitations in capturing coordinate dependencies when the number of sampling steps is small and proposed additional modeling strategies to address this issue. [Schiff et al.](#page-10-17) [\(2024\)](#page-10-17) studied conditional generation using MDM and [Rector-Brooks et al.](#page-10-18) [\(2024\)](#page-10-18) tackled the challenge of controlling generated data distributions through steering methodologies. [Chen & Ying](#page-9-11) [\(2024\)](#page-9-11) provided a theoretical analysis showing that sampling error is small given accurate score function estimation.

Any-order reasoning. Even though language tasks generally have a natural order of "left-to-right" token generation, in many tasks like planning, reasoning, and combinatorial optimization, the natural order of token generation can be quite different from "left-to-right". Even though prominent autoregressive-based language models achieve impressive performance on various tasks, many works [\(Golovneva et al.,](#page-9-12) [2024;](#page-9-12) [Chen et al.,](#page-9-13) [2024;](#page-9-13) [Kitouni et al.,](#page-9-3) [2025\)](#page-9-3) have shown that this performance is tied to the training order of the tasks and therefore can cause brittleness from it. For example, [Chen](#page-9-13) [et al.](#page-9-13) [\(2024\)](#page-9-13) showed that simply permuting the premise order on math tasks causes a performance drop of 30%. The reason behind such brittleness regarding the ordering is the inherent "left-to-right" nature of the autoregressive models. Several works [\(Liao et al.,](#page-9-14) [2020\)](#page-9-14) have tried to address this issue in the autoregressive framework. In particular, [\(Papadopoulos et al.,](#page-10-19) [2024\)](#page-10-19) highlighted the significance of left-to-right ordering in natural language by comparing its likelihood to that of the reverse (right-to-left) ordering.

Recently, discrete diffusion models have emerged as a promising approach for discrete data apart from autoregressive models. Additionally, the order-agnostic training of discrete diffusion models opens up the multiple sampling paths during the inference but it also faces some challenges during the training therefore, they seem a promising approach to elicit any order reasoning. [Zheng et al.](#page-11-1) [\(2023\)](#page-11-1) proposed different ways of implementing an adaptive inference strategy for MDM but a *concrete understanding of why such an adaptive inference strategy is needed is still lacking*. In this work, we explore various aspects of vanilla MDM training and how adaptive MDM inference can mitigate the issues raised by vanilla MDM training and elicit any order reasoning.

We also want to mention the concurrent work by [Peng et al.](#page-10-20) [\(2025\)](#page-10-20) that proposes an alternative adaptive inference strategy by selecting F(θ, xt) based on the BERT model or the denoiser itself. In particular, [Peng et al.](#page-10-20) [\(2025\)](#page-10-20) uses the BERT model or the denoiser to obtain the uncertainty of a token and then uses Top-K to decide the positions to unmask it. In contrast to their work, we disentangle the impact of token ordering on MDM training vs. MDM inference and provide a more complete understanding of the motivations for and benefits of adaptive inference. Additionally, our results indicate drawbacks to using Top-K strategy as opposed to Top-K margin in deciding which tokens to unmask when there are multiple values with high probabilities.

Beyond autoregressive models. Efforts to learn the natural language using non-autoregressive modeling began with BERT [\(Devlin et al.,](#page-9-15) [2019\)](#page-9-15). Non-causal approaches can take advantage of the understanding the text data representation. [\(Chang et al.,](#page-9-16) [2022\)](#page-9-16) adopted a similar approach for learning image representations. Building on these intuitions, [\(Shih](#page-10-21) [et al.,](#page-10-21) [2022;](#page-10-21) [Hoogeboom et al.,](#page-9-17) [2021a\)](#page-9-17) proposed any-order modeling, which allows a model to generate in any desired order. [Shih et al.](#page-10-21) [\(2022\)](#page-10-21) made the same observation that any-order models by default have to solve exponentially more masking problems than autoregressive models. However, whereas our work shows that learning in the face of this challenging task diversity can benefit the model at inference time, their work sought to alleviate complexity at training time by reducing the number of masking problems that need to be solved.

# B. Technical details from Section [3](#page-2-0)

Notations. Throughout this section, we use x i to denote the i-th coordinate of the vector x and z(j) to denote the j-th example. The i-th coordinate of the vector z(j) is denoted by z(j) i .

## B.1. Additional example: sparse parity observations

Example B.1 (Noisy sparse parity observations). *Let* m = 2*,* k ∈ <sup>N</sup>*, and* N<sup>2</sup> log N ≪ P ≤ N<sup>0</sup>.49<sup>k</sup> *. Fix* noise rate η > 0 *as well as strings* z(1), . . . , z(P) *sampled independently and uniformly at random from the set of* k*-sparse strings in* {0, 1} N *. For each* j ∈ [P]*, define* O<sup>j</sup> (x) *to be the distribution which places mass* 1 − η *on* 1 *(resp.* 2*) and mass* η *on* 2 *(resp.* 1*) if* P i x i z(j) i *is odd (resp. even). Note that for* k = O(1)*, each of these observations is efficiently learnable by brute-force.*

Below we show that for a certain range of masking fractions, a constant fraction of the masking problems for the corresponding L&O distributions are computationally hard under the *Sparse Learning Parity with Noise* assumption [\(Alekhnovich,](#page-8-3) [2003\)](#page-8-3). Formally we have:

Proposition B.2. *Let* 0 < α < 1 *be an arbitrary absolute constant, and let* η = 1/poly(N) *be sufficiently large. Let* x *be a sample from a L&O distribution* pdata *with noisy parity observations as defined in Example [B.1.](#page-13-1) Suppose each token is independently masked with probability* α*, and* M *is the set of indices for the masked tokens. If* 1 − 1/N ≤ α ≤ 1 − 1/2N*, then under the Sparse Learning Parity with Noise (SLPN) assumption (see Definition [B.3\)](#page-13-2), with constant probability over* M*, no polynomial-time algorithm can solve the resulting masking problem of predicting any of the masked tokens among* x <sup>π</sup>(1), . . . , x<sup>π</sup>(N) *given* x[M]*.*

We note that it is important for us to take the observations to be *sparse* parities and to leverage the *Sparse* Learning Parity with Noise assumption. If instead we used *dense* parities and invoked the *standard* Learning Parity with Noise (LPN) assumption, we would still get the hardness of masking problems, but the observations themselves would be hard to learn, assuming LPN. This result is based on the following standard hardness assumption:

Definition B.3 (Sparse Learning Parity with Noise). Given input dimension N, noise parameter 0 < η < 1/2, and sample size P, an instance of the *Sparse Learning Parity with Noise (SLPN)* problem is generated as follows:

- Nature samples a random bitstring x from {0, 1} N
- We observe P examples of the form (x(i), y(i)) where x(i) is sampled independently and uniformly at random from k-sparse bitstrings in {0, 1} <sup>N</sup> , and y is given by ϵ<sup>i</sup> + ⟨x(i), x⟩ (mod 2), where ϵ<sup>i</sup> is 1 with probability η and 0 otherwise.

Given the examples {(x(i), y(i))} P <sup>i</sup>=1, the goal is to recover x.

The *SLPN assumption* is that for any P = N(1−ρ)k/<sup>2</sup> for constant 0 < ρ < 1, and any sufficiently large inverse polynomial noise rate η, no poly(N)-time algorithm can recover x with high probability.

*Proof of Proposition [B.2.](#page-13-3)* With probability at least 1 − (1 − 1/N) <sup>N</sup> ≥ Ω(1), all of the variable tokens x π(i) for i ≤ N are masked. Independently, the number of unmasked tokens among the observation tokens O<sup>j</sup> is distributed as Bin(P, 1−α), so by a Chernoff bound, with probability at least 1 − e −Ω(P/N<sup>2</sup> ) = 1 − 1/poly(N) we have that at least P/4N = Ω(N log N) observation tokens are unmasked. The masking problem in this case amounts to an instance of SLPN with input dimension N and sample size in [Ω(N log N), O(N<sup>0</sup>.49<sup>k</sup> )]. Because of the lower bound on the sample size, prediction of x<sup>M</sup> is information-theoretically possible. Because of the upper bound on the sample size, the SLPN assumption makes it computationally hard. As a result, estimating the posterior mean on any entry of x<sup>M</sup> given the unmasked tokens is computationally hard as claimed.

## B.2. Additional example: random slab observations

Example B.4 (Random slab observations). *Let* m = 2 *and* P = γN<sup>2</sup> *for constant* γ > 0*. Fix* slab width β *and vectors* z(1), . . . , z(P) *sampled independently from* N (0, I)*. For each* j ∈ [P]*, define the corresponding observation* O<sup>j</sup> (x) *to be deterministically* 1 *if* |⟨z(j), 2x − 1⟩| ≤ β √ N*, and deterministically* 0 *otherwise.*

In [\(Alaoui & Gamarnik,](#page-8-4) [2024\)](#page-8-4), it was shown that *stable* algorithms (Definition [B.7\)](#page-14-2), which encompass many powerful methods for statistical inference like low-degree polynomial estimators, MCMC, and algorithmic stochastic localization [\(Gamarnik,](#page-9-18) [2021\)](#page-9-18), are unable to sample from the posterior distribution over a random bitstring conditioned on it satisfying |⟨z(j), x⟩| ≤ β √ N for any Θ(N) number of constraints z(1), . . . , z(P ′ ), provided P ′ is not too large that the support of the posterior is empty. This ensemble is the well-studied *symmetric perceptron* [\(Aubin et al.,](#page-8-5) [2019\)](#page-8-5). The following is a direct reinterpretation of the result of [\(Alaoui & Gamarnik,](#page-8-4) [2024\)](#page-8-4):

Proposition B.5. *Let* pdata *be a L&O distribution with random slab observations as defined in Example [B.4,](#page-14-3) with parameter* γ > 0 *and slab width* β > 0*. There exists a constant* c<sup>β</sup> > 0 *such that for any absolute constant* 0 < c < cβ*, if* 1 − cβN/2P ≤ α ≤ 1 − cN/P *and* γ > cβ*, the following holds. Let* p ′ data *denote the distribution given by independently masking every coordinate in* pdata *with probability* α*. Then* any (1 − Ω(1 ˜ / √ N))*-stable algorithm, even one not based on masked diffusion, which takes as input a sample* x ′ *from* p ′ data *and, with probability* 1 − o(1) *outputs a Wassersteinapproximate*[<sup>3</sup>](#page-14-4) *sample from* pdata *conditioned on the unmasked tokens in* x ′ *, must run in super-polynomial time.*

The upshot of this is that any stable, polynomial-time masked diffusion sampler will, with non-negligible probability, encounter a computationally hard masking problem at some point during the reverse process.

For the proof, we first formally define the (planted) symmetric Ising perceptron model:

Definition B.6. Let α, β > 0. The *planted symmetric Ising perceptron* model is defined as follows:

- Nature samples σ uniformly at random from {±1} N
- For each j = 1, . . . , P = ⌊αN⌋, we sample z(j) independently from N (0, I<sup>N</sup> ) conditioned on satisfying |⟨z(j), σ⟩| ≤ β √
  - N.

The goal is to sample from the posterior on σ conditioned on these observations {z(i)} P <sup>i</sup>=1.

Next, we formalize the notion of *stable algorithms*.

Definition B.7. Given a matrix Z ∼ N (0, 1)<sup>⊗</sup><sup>P</sup> <sup>×</sup><sup>N</sup> , define Z<sup>t</sup> = tZ + √ 1 − t <sup>2</sup>Z ′ for independent Z ′ ∼ N (0, 1)<sup>⊗</sup><sup>P</sup> <sup>×</sup><sup>N</sup> . A randomized algorithm A which takes as input Z ∈ R <sup>P</sup> <sup>×</sup><sup>N</sup> and outputs an element of {±1} <sup>N</sup> is said to be t<sup>N</sup> *-stable* if limN→∞ W2(law(A(Z)), law(A(Zt))) = 0.

As discussed at depth in [\(Gamarnik,](#page-9-18) [2021\)](#page-9-18), many algorithms like low-degree polynomial estimators and Langevin dynamics are stable.

Theorem B.8 (Theorem 2.1 in [\(Alaoui & Gamarnik,](#page-8-4) [2024\)](#page-8-4) [4](#page-14-5) ). *For any constant* β > 0*, there exists* c<sup>β</sup> > 0 *such that the following holds for all constants* 0 < α < cβ*. For* t<sup>N</sup> ≤ 1 − Ω(log<sup>2</sup> (n)/n<sup>2</sup> )*, any* t<sup>N</sup> *-stable randomized algorithm* A *which takes as input* Z = (z(1), . . . , z(P)) *and outputs an element of* {±1} <sup>N</sup> *will fail to sample from the posterior on* σ *conditioned on* Z *in the symmetric Ising perceptron model to Wasserstein error* o( √ N)*.*

*Proof of Proposition [B.5.](#page-14-6)* By a union bound, with probability at least 1 − (1 − α)N ≥ 1 − cβN<sup>2</sup>/P ≥ 1 − cβ/γ over a draw x ′ ∼ p ′ data, all of the x π(i) tokens are masked. The number of unmasked tokens in x ′ among the observations O<sup>j</sup> is distributed as Bin(P, 1 − α). By a Chernoff bound, this is in [3cN/4, 3cβN/4] with at least constant probability. The claim then follows immediately from Theorem [B.8](#page-14-7) above.

# B.3. Proof outline of Proposition [3.3](#page-4-3)

To understand the proof idea, we consider the case where all the latent tokens are masked and some of the observation tokens are unmasked. In this case, the prediction task reduces to learning to recover the latent tokens that are consistent

<sup>3</sup>Here the notion of approximation is o(1)-closeness in Wasserstein-2 distance.

<sup>4</sup>Note that while the theorem statement in [\(Alaoui & Gamarnik,](#page-8-4) [2024\)](#page-8-4) refers to the non-planted version of the symmetric binary perceptron, the first step in their proof is to argue that these two models are mutually contiguous in the regime of interest.

![](_page_15_Figure_1.jpeg)

Figure 4. Overlap achieved by belief propagation initialized at ground truth versus random for planted CSP with k = 3, m = 3, and g = NAE, for N = 10000 and varying choices of average degree D. DKS/K can be shown analytically to be 64, consistent with the phase transition depicted. Plot suggests Dcond/K ≈ 50. By Prop. [3.3](#page-4-3) this implies a range of masking fractions at which Ω(1) fraction of masking problems are computationally hard.

with the observations. Intuitively, each observation provides some constraints and the task is to recover an assignment that satisfies the constraints. This is reminiscent of *Constraint Satisfaction Problems* (CSPs). Indeed, to show the hardness result, we use the rich theory developed for *planted* CSPs at the intersection of statistical physics and average-case complexity.

In a planted CSP, there is an unknown randomly sampled vector y of length N and, one is given randomly chosen Boolean constraints which y is promised to satisfy, and the goal is to recover y as best as possible (see Definition [B.9\)](#page-16-4). Prior works have shown the hardness of efficiently learning to solve the planted CSP problem [\(Krzakala & Zdeborova´,](#page-9-19) [2009;](#page-9-19) [Alaoui &](#page-8-4) [Gamarnik,](#page-8-4) [2024\)](#page-8-4). We show the hardness of masking problems in L&O distributions based on these results. Consider the ground truth latent tokens as the random vector y and each observation as a constraint. In this case, the problem of learning to recover the latent tokens from the observation tokens reduces to recovery for the planted CSP.

There are precise predictions for the values of vocabulary size m and the number of observations for which the informationtheoretically best possible overlap and the best overlap achievable by any computationally efficient algorithm are different. We show that these predictions directly translate to predictions about when masking problems become computationally intractable:

As a simple example, let us consider sparse predicate observations with k = 2 and g(x ′ , x′′) = 1[x ′ ̸= x ′′]. These can be formally related to the well-studied problem of *planted* m*-coloring*. In the planted m-coloring, a random graph of average degree D is sampled consistent with an unknown vertex coloring and the goal is to estimate the coloring as well as possible [\(Krzakala & Zdeborova´,](#page-9-19) [2009\)](#page-9-19), as measured by the *overlap* of the output of the algorithm to the ground-truth coloring (see Definition [B.9\)](#page-16-4). As a corollary of our main result, we show that when all the latent tokens x <sup>π</sup>(1), . . . , x<sup>π</sup>(N) are masked and a few unmasked observation tokens provide the information of the form g(x π(i) , x<sup>π</sup>(j) ) = 1[x <sup>π</sup>(i) ̸= x π(j) for i, j ≤ N, then solving the masking problem can be reduced to solving planted coloring.

For planted m-coloring, when m = 5 the thresholds in Proposition [3.3](#page-4-3) are given by DKS/2 = 16 and Dcond/2 ≈ 13.23 [\(Krzakala & Zdeborova´,](#page-9-19) [2009\)](#page-9-19) (the factor of 2 here is simply because the observations correspond to *ordered* subsets of size 2). For general predicates and arities, there is an established recipe for numerically computing DKS and Dcond based on the behavior of the *belief propagation* algorithm (see the discussion in Appendix [B.4\)](#page-16-3). As an example, in Fig. [4,](#page-15-0) we execute this recipe for m = 3, k = 3, and g given by the Not-All-Equal predicate NAE(x ′ , x′′, x′′) = 1−1[x ′ = x ′′ = x ′′′] to obtain thresholds that can be plugged into Proposition [3.3.](#page-4-3)

Additional examples of the hardness. The above setup can also be generalized to capture *Bayesian constraint satisfaction problems* [\(Montanari,](#page-10-22) [2008;](#page-10-22) [Liu et al.,](#page-9-20) [2022\)](#page-9-20), one notable example of which is the stochastic block model [\(Decelle et al.,](#page-9-21) [2011\)](#page-9-21). There are analogous predictions for the onset of hardness of inference, which can likewise be translated to hardness of masking problems for seemingly benign L&O distributions. In Appendix [B.1](#page-13-0) and [B.2,](#page-14-0) we give two more examples of L&O distributions for which order-aware training is tractable yet order-agnostic training of the MDM is computationally

hard.

First, we consider L&O distributions whose observations are sparse, noisy parities in the latents and deduce hardness for order-agnostic training from the Sparse Learning Parity with Noise assumption [\(Alekhnovich,](#page-8-3) [2003\)](#page-8-3). We then consider L&O distributions whose observations are *generalized linear models* in the latents, and deduce hardness for a large class of efficient algorithms from existing results on Lipschitz hardness [\(Alaoui & Gamarnik,](#page-8-4) [2024\)](#page-8-4) for the symmetric binary perceptron [\(Aubin et al.,](#page-8-5) [2019\)](#page-8-5).

#### B.4. Proof of Proposition [3.3:](#page-4-3) sparse predicate observations

Here we formally define the relevant notions needed to formalize our claim about hardness in Proposition [3.3.](#page-4-3)

Definition B.9 (Planted CSPs). Given arity k ∈ N, vocabulary/alphabet size m ∈ N, predicate g : {1, . . . , m} <sup>k</sup> → {0, 1}, latent dimension N, and clause density P/N, the corresponding *planted constraint satisfaction problem* is defined as follows: Nature samples an unknown assignment σ uniformly at random from {1, . . . , m} <sup>N</sup> , and then for each ordered k-tuple S of distinct elements from [N], we observe the *clause* S independently with probability ϕ/N<sup>k</sup>−<sup>1</sup> if g(σ|S) = 1.

To measure the quality of an algorithm for recovering σ given the observations, define the *overlap* between an estimate σˆ and the ground truth σ by d(σ, σˆ) ≜ minπ∈S<sup>N</sup> P i 1[σ<sup>i</sup> = π(ˆσi)] where <sup>S</sup><sup>N</sup> denotes the set of all permutations of {0, 1, . . . , N − 1}. Define the *average degree* to be kP/N, i.e. the expected number of variables that share at least one clause with a given variable.

We begin by defining the central algorithm driving statistical physics predictions about hardness for random constraint satisfaction problems: belief propagation (BP).

Definition B.10 (BP update rules). Belief propagation is an algorithm that iteratively updates a set of *messages* {MS<sup>i</sup>→<sup>S</sup> c [t], MSS→<sup>i</sup> c [t]}, where i, S range over all pairs of variable indices i ∈ [N] and observations S ∋ i. At time t + 1, the messages are computed via

$$\text{MS}_c^{i \rightarrow S}[t+1] \propto \prod_{T: i \in T \neq S} \text{MS}_c^{T \rightarrow i}[t] \quad (4)$$

$$\mathbf{MS}_c^{S \rightarrow i}[t+1] \propto \sum_{\overline{\sigma} \in \{1, \dots, m\}^{S \setminus i}} g(\overline{\sigma} \cup i | c) \prod_{j: i \neq j \in S} \mathbf{MS}_{\overline{\sigma}_j}^{j \rightarrow S}[t], \quad (5)$$

where σ ∪<sup>i</sup> c ∈ {1, . . . , m} <sup>S</sup> assigns c to entry i and σ to the remaining entries.

A set of messages can be used to estimate the marginals of the posterior on σ conditioned on the observations as follows. The marginal on the i-th variable has probability mass function over {1, . . . , m} proportional to { Q <sup>T</sup>:i∈<sup>T</sup> MS<sup>T</sup>→<sup>i</sup> <sup>c</sup> }. Given a set of marginals, a natural way to extract an estimate for σ is to round to the color in {1, . . . , m} at which the probability mass function is largest.

Throughout we will make the following assumption that ensures that the trivial messages MS<sup>i</sup>→<sup>S</sup> <sup>c</sup> = 1/m and MSS→<sup>i</sup> <sup>c</sup> = 1/m are a fixed point, sometimes called the *paramagnetic fixed point*, for the iteration above:

Assumption B.11. The quantity P σ∈{1,...,m}[k]\i g(σ ∪<sup>i</sup> c) is constant across all c ∈ {1, . . . , m} and i ∈ [k].

Definition B.12. Given k, m, g, the *Kesten-Stigum* threshold DKS is defined to be the largest average degree for which BP is locally stable around the paramagnetic fixed point, that is, starting from a small perturbation of the paramagnetic fixed point, it converges to the paramagnetic fixed point. More formally, DKS is the largest average degree at which the Jacobian of the BP operator {MS<sup>i</sup>→<sup>S</sup> [t]} 7→ {MS<sup>i</sup>→<sup>S</sup> [t + 1]} has spectral radius less than 1.

The *condensation* threshold Dcond is defined to be the largest average degree at which the planted CSP ensemble and the following simple *null model* become mutually contiguous and thus statistically indistinguishable as N → ∞. The null model is defined as follows: there is no single unknown assignment, but instead for every ordered subset S of k variables, Nature independently samples an unknown local assignment σ<sup>S</sup> ∈ {1, . . . , m} <sup>S</sup>, and the observation is included with probability ϕ/N<sup>k</sup>−<sup>1</sup> if g(σS) = 1.

For Dcond < kP/N < DKS, there exists some *other* fixed point of the BP operator whose marginals, once rounded to an assignment, achieves strictly higher overlap than does BP with messages initialized randomly. The prediction is that in this regime, no efficient algorithm can achieve optimal recovery [\(Krzakala & Zdeborova´,](#page-9-19) [2009\)](#page-9-19).

Conjecture B.13 (1RSB cavity prediction). *Suppose* k, m, g *satisfy Assumption [B.11,](#page-16-0) and let* DKS *and* Dcond *denote the associated Kesten-Stigum and condensation thresholds for the average degree. Then for all* P *for which* Dcond < kP/N < DKS*, the best overlap achieved by a computationally efficient algorithm for recovering* σ *is strictly less than the best overlap achievable.*

*Proof of Proposition [3.3.](#page-4-3)* At masking fraction α satisfying the bounds in the Proposition, with probability at least α <sup>N</sup> ≥ (1−γ <sup>−</sup><sup>1</sup>DKS/N<sup>k</sup>−<sup>1</sup> ) <sup>N</sup> ≥ Ω(1) we have that all tokens corresponding to latents xπ(i) get masked. Independently of this, the number of unmasked tokens among the observation tokens O<sup>S</sup> is distributed as Bin(N(N −1)· · ·(N −k+ 1), 1−α), so by standard binomial tail bounds, with constant probability (depending on the gap between Dcond and DKS) this lies between γ <sup>−</sup><sup>1</sup>DcondN/k and γ <sup>−</sup><sup>1</sup>DKSN/k. Furthermore, of these unmasked tokens in expectation γ fraction of them correspond to observations for which the associated predicate evaluates to 1. Conditioned on the above events, the masking problem thus reduces exactly to inference for a planted constraint satisfaction problem at average degree Dcond < D < DKS, from which the Proposition follows.

# C. Experimental details in Section [3](#page-2-0)

# C.1. Experimental details in Section [3.2](#page-4-0)

π-learner configurations. We consider two distributions of π that interpolate between Unif (SL) where <sup>S</sup><sup>L</sup> denote the uniform distribution over all permutations of indices {0, 1, . . . , L − 1} and the point mass at the identical distribution: (Closer) and (Much-closer). To construct those distributions, we start from the identity permutation and perform a certain number of random swapping operations. Since Llog(L) number of swaps results in a distribution that is very close to Unif (SL) [\(Bormashenko,](#page-8-6) [2011\)](#page-8-6), we use L/<sup>10</sup> and √ L swaps to construct the (Closer) and (Much-closer) distributions, respectively. For consistency, we repeat this sampling process three times.

Model and training configurations. As explained in Section [3.2,](#page-4-0) to evaluate the scaling law of the π-learner, we can simply adapt the autoregressive training setup (a transformer with causal attention) by modifying the input to π(x0) and using a learnable positional embedding layer instead of RoPE. We borrow the training configurations from [\(Nie et al.,](#page-10-3) [2024\)](#page-10-3), which are also consistent with the TinyLlama [\(Zhang et al.,](#page-10-23) [2024\)](#page-10-23) configurations. In particular, we use AdamW optimizer [\(Loshchilov & Hutter,](#page-9-22) [2017\)](#page-9-22), setting β<sup>1</sup> = 0.9, β<sup>2</sup> = 0.95, and a weight decay of 0.1 and L = 2048. A cosine learning rate schedule is applied, with a maximum learning rate of 4 × 10−<sup>4</sup> and a minimum learning rate of 4 × 10−<sup>5</sup> . We also note that unless otherwise specified, we maintain the same training configuration throughout the paper.

Examining scaling laws. We conduct IsoFLOP analysis [\(Hoffmann et al.,](#page-9-23) [2022\)](#page-9-23). For a given number of FLOPs C, by varying the number of non-embedding parameters of transformers, we set the iteration numbers so that the total number of tokens observed by the model during training equals C/6N, following prior studies [\(Hoffmann et al.,](#page-9-23) [2022;](#page-9-23) [Kaplan et al.,](#page-9-24) [2020\)](#page-9-24). We then select the smallest validation loss and set it as a data point.

## C.2. Experimental details in Section [3.3](#page-5-1)

## C.2.1. EXPERIMENT ON L&O-NAE-SAT DISTRIBUTION

We consider the L&O-NAE-SAT distribution with (N, P) = (20, 280). For each example sequence from L&O-NAE-SAT, we pad the last 212 tokens with an additional token value of 2. We employ a 19M MDM with RoPE and a maximum sequence length of 512. Then, this MDM is trained for 2 × 10<sup>3</sup> iterations. To attain a proxy MDM for the Bayes optimal predictor, we further train it for 5 × 10<sup>4</sup> iterations.

To measure the error across different tasks, we consider the following setup. For each ℓ ∈ [1, N − 1], we randomly mask ℓ tokens in the latent positions and ℓ × (P/N) tokens in the observed positions. Across all masked prediction positions, ℓ(1 + P/N), we measure the error for each position. For certainty, we repeat this process 1000 times. The result in Figure [2](#page-3-0) corresponds to the case when ℓ = 11, and we observe the same tendency for other values of ℓ.

# C.2.2. EXPERIMENT ON TEXT DATA

$$\mathbb{E}_{x_0 \sim p_{\text{data}}} \left[ \sum_{i=0}^{L-1} \log p_\theta \left( x_0^{\pi(i)} \middle| x_0[\pi\{i, \dots, L-1\}] \right) \right].$$

As done in the experiments in Section [3.2,](#page-4-0) we sample πs from three different distributions: Unif(SL), (Closer), the point mass of identical distribution. For each case, we calculate the expectation over 1024 samples of x<sup>0</sup> ∼ pdata.

## D. Experimental details in Section [4](#page-5-0)

## D.1. Experimental details in Section [4.2](#page-6-4)

## D.1.1. EXPERIMENT ON L&O-NAE-SAT DISTRIBUTION

We consider five instances of L&O-NAE-SAT: (N, P) = (25, 275),(30, 270),(40, 260),(50, 250),(100, 200). For each distribution, we train a 19M MDM and measure the accuracy difference between vanilla inference and adaptive inference using Top-K probability margin.

## D.1.2. EXPERIMENT ON TEXT DATA

Top-K probability margin sampler with temperature. To modify our inference for text data modeling, which does not have a determined answer, we found that adding a certain level of temperature to the oracle is useful. This is because Top-K probability margin or Top-K often leads to greedy sampling, which harms the diversity (entropy) of the generated samples. Therefore, we consider a variant of the oracle as follows, incorporating a noise term ϵ:

$$\mathcal{F}(\theta, x_t) = \text{Top } K \left( |p_\theta(x^i = j_1|x_t) - p_\theta(x^i = j_2|x_t)| + \epsilon \right).$$

Note that this approach has also been employed for unconditional sampling [\(Wang et al.,](#page-10-11) [2024;](#page-10-11) [Zheng et al.,](#page-11-1) [2023\)](#page-11-1).

Generative perplexity and entropy. We employ a 1.1B MDM pretrained on text data as a baseline. For each sampling step, we unconditionally generate samples using both vanilla and adaptive inference. Next, we calculate the likelihood using LLama2-7B as a baseline large language model. Moreover, we denote the entropy of a generated sample x as Pp<sup>i</sup> log p<sup>i</sup> , where p<sup>i</sup> = #{x <sup>i</sup> = i}/L.

#### D.2. Experimental details on Sudoku and Zebra puzzles

Dataset. For both Sudoku and Zebra puzzles, we use the dataset provided in [Shah et al.](#page-10-5) [\(2024\)](#page-10-5) to train our model. To evaluate our model on the same difficulty tasks, we use the test dataset proposed in [Shah et al.](#page-10-5) [\(2024\)](#page-10-5). This dataset is created by filtering the puzzles from [\(Radcliffe,](#page-10-13) [2020\)](#page-10-13) that can be solved using a fixed list of 7 strategies. To create a hard dataset to evaluate easy-to-hard generalization, we use the remaining puzzles from [\(Radcliffe,](#page-10-13) [2020\)](#page-10-13) as they either require a new strategy unseen during the training and/or require backtracking. The hard dataset contains around 1M Sudoku puzzles.

Model, training, and inference. For the Sudoku dataset, we use 6M GPT-2 model and for the Zebra dataset, we use 19M model but instead of causal attention, we use complete bidirectional attention. We set the learning rate to 0.001 with batch size 128 to train the model for 300 epochs. For the inference, we use 50 reverse sampling steps using the appropriate strategy. Additionally, we add Gumbel noise with a coefficient of 0.5 to the MDM inference oracle F.

# E. Omitted proofs

*Proof of Proposition [2.1.](#page-2-5)* We first re-state the Proposition 3.1 from [\(Zheng et al.,](#page-11-0) [2024\)](#page-11-0). To clarify, [\(Zheng et al.,](#page-11-0) [2024\)](#page-11-0) generally considers the case beyond the time-embedding denoising network pθ.

Proposition E.1 (Proposition 3.1 of [\(Zheng et al.,](#page-11-0) [2024\)](#page-11-0)). *For clean data* x0*, let* q˜(x(n) | x0) *be the discrete forward process that randomly and uniformly masks* n *tokens of* x0*. Suppose* α<sup>0</sup> = 0 *and* α<sup>1</sup> = 1*. Then the MDM training loss* [\(1\)](#page-2-3) *can be reformulated as*

$$\mathcal{L}_\theta = - \sum_{n=1}^L \mathbb{E}_{x(n) \sim \tilde{q}(\cdot | x_0)} \left[ \frac{1}{n} \sum_{\ell: x(n)^\ell = m} \mathbf{e}_{x_0^\ell} \log p_\theta(x^\ell \mid x(n)) \right]. \quad (6)$$

To obtain an alternative formulation of [\(6\)](#page-18-2), we expand the expectation x(n) ∼ q˜(· | x0). Since there are total L positions of x0, we have the probability assigned for each x(n) equals 1/ L n . Therefore,

$$\begin{aligned}\mathcal{L}_\theta &= - \sum_{n=1}^L \mathbb{E}_{x(n) \sim \tilde{q}(\cdot | x_0)} \left[ \frac{1}{n} \sum_{\ell: x(n)^\ell = m} \mathbf{e}_{x_0^\ell} \log p_\theta(x^\ell \mid x(n)) \right] \\ &= - \sum_{M \in [L], i \in M} \frac{1}{\binom{L}{|M|}} \times \frac{1}{|M|} \mathbf{e}_{x_0^\ell} \log p_\theta(x^\ell \mid x[M]) \\ &= - \sum_{M \in [L], i \in M} \frac{1}{\binom{L}{|M|}} \times \frac{1}{|M|} \log p_\theta(x_0^\ell \mid x[M]) \\ &= - \sum_{M \in [L], i \in M} \frac{1}{L^{\frac{L-1}{|M|-1}}} \log p_\theta(x_0^\ell \mid x[M]).\end{aligned}$$

*Reformulating the MDM loss with* π*-learner s.* In this paragraph, we provide the proof of

$$-\frac{1}{L} \sum_{M \subseteq [L], i \in M} \frac{1}{\binom{L-1}{|M|-1}} \mathbb{E}_{x_0 \sim p_{\text{data}}} [\log p_\theta(x_0^i | x_0[M])] = -\mathbb{E}_{\pi \sim \text{Unif}(\mathbb{S}_L), x_0 \sim p_{\text{data}}} \left[ \sum_{i=0}^{L-1} \log p_\theta \left( x_0^{\pi(i)} \middle| x_0[\pi\{i, \dots, L-1\}] \right) \right].$$

Alternatively, we will demonstrate that

$$-\frac{1}{L} \sum_{M \subseteq [L], i \in M} \frac{1}{\binom{L-1}{|M|-1}} \log p_\theta(x_0^i | x_0[M]) = -\mathbb{E}_{\pi \sim \text{Unif}(\mathbb{S}_L)} \left[ \sum_{i=0}^{L-1} \log p_\theta \left( x_0^{\pi(i)} \middle| x_0[\pi\{i, \dots, L-1\}] \right) \right].$$

holds for every x0. Note that

$$\begin{aligned} & \mathbb{E}_{\pi \sim \text{Unif}(\mathbb{S}_L)} \left[ \sum_{i=0}^{L-1} \log p_\theta \left( x_0^{\pi(i)} \middle| x_0[\pi\{i, \dots, L-1\}] \right) \right] \\ &= \frac{1}{L!} \sum_{\pi \in \mathbb{S}_L} \sum_{j=0}^{L-1} \log p_\theta \left( x_0^{\pi(j)} \middle| x_0[\pi\{j, \dots, L-1\}] \right). \end{aligned}$$

Next, by regarding π{j, . . . , L − 1} = {π(j), . . . , π(L − 1)} = M ⊆ [L] and π(j) = i in the equation [\(1\)](#page-2-3), we count the number of π ∈ S<sup>L</sup> that induces a specific term log pθ(x i 0 |x0[M]). For a given M ∈ [L] and i ∈ M, π must satisfy

$$\pi(j) = i, \quad \{\pi(j), \dots, \pi(L-1)\} = M.$$

The number of π that satisfies above is (L − |M|)! × (|M| − 1)!. Finally, the following calculation concludes the proof.

$$\begin{aligned} & \mathbb{E}_{\pi \sim \text{Unif}(\mathbb{S}_L)} \left[ \sum_{i=0}^{L-1} \log p_\theta \left( x_0^{\pi(i)} \middle| x_0[\pi\{i, \dots, L-1\}] \right) \right] \\ &= \frac{1}{L!} \sum_{\pi \in S_L} \sum_{j=0}^{L-1} \log p_\theta \left( x_0^{\pi(j)} \middle| x_0[\pi\{j, \dots, L-1\}] \right) \\ &= \frac{1}{L!} \sum_{|M| \in [L], i \in M} [\log p_\theta(x_0^i | x_0[M]) \times (L-1 - |M|)! \times (|M|-1)!] \\ &= \frac{1}{L} \sum_{|M| \in [L], i \in M} \frac{1}{\binom{L-1}{|M|-1}} \times \log p_\theta(x_0^i | x_0[M]). \end{aligned}$$