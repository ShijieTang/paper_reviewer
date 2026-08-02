# SELF-IMPROVEMENT IN LANGUAGE MODELS: THE SHARPENING MECHANISM

Audrey Huang<sup>∗</sup> UIUC [audreyh5@illinois.edu](mailto:audreyh5@illinois.edu) Adam Block<sup>∗</sup> Microsoft Research [blockadam@microsoft.com](mailto:blockadam@microsoft.com) Dylan J. Foster<sup>∗</sup> Microsoft Research [dylanfoster@microsoft.com](mailto:dylanfoster@microsoft.com)

Dhruv Rohatgi MIT [drohatgi@mit.edu](mailto:drohatgi@mit.edu) Cyril Zhang Microsoft Research [cyrilzhang@microsoft.com](mailto:cyrilzhang@microsoft.com) Max Simchowitz CMU [msimchow@andrew.cmu.edu](mailto:msimchow@andrew.cmu.edu)

Jordan T. Ash Microsoft Research [ash.jordan@microsoft.com](mailto:ash.jordan@microsoft.com) Akshay Krishnamurthy Microsoft Research [akshaykr@microsoft.com](mailto:akshaykr@microsoft.com)

# ABSTRACT

Recent work in language modeling has raised the possibility of *self-improvement*, where a language models evaluates and refines its own generations to achieve higher performance without external feedback. It is impossible for this self-improvement to create information that is not already in the model, so why should we expect that this will lead to improved capabilities?

We offer a new perspective on the capabilities of self-improvement through a lens we refer to as *sharpening*. Motivated by the observation that language models are often better at verifying response quality than they are at generating correct responses, we formalize self-improvement as using the model itself as a verifier during post-training in order to "sharpen" the model to one placing large mass on high-quality sequences, thereby amortizing the expensive inference-time computation of generating good sequences. We begin by introducing a new statistical framework for sharpening in which the learner aims to sharpen a pre-trained base policy via sample access, and establish fundamental limits. Then, we analyze two natural families of self-improvement algorithms based on SFT and RLHF. We find that (i) the SFT-based approach is minimax optimal whenever the initial model has sufficient coverage, but (ii) the RLHF-based approach can improve over SFT-based self-improvement by leveraging online exploration, bypassing the need for coverage. Finally, we empirically validate the sharpening mechanism via inference-time and amortization experiments. We view these findings as a starting point toward a foundational understanding that can guide the design and evaluation of self-improvement algorithms.

# 1 INTRODUCTION

Contemporary language models are remarkably proficient on a wide range of natural language tasks [\(Brown et al.,](#page-11-0) [2020;](#page-11-0) [Ouyang et al.,](#page-13-0) [2022;](#page-13-0) [Touvron et al.,](#page-14-0) [2023;](#page-14-0) [OpenAI,](#page-13-1) [2023;](#page-13-1) [Google,](#page-12-0) [2023\)](#page-12-0), but inherit shortcomings of the data on which they were trained. A fundamental challenge is to achieve better performance than what is directly induced by the distribution of available, human-generated training data. To this end, recent work [\(Huang et al.,](#page-12-1) [2022;](#page-12-1) [Wang et al.,](#page-15-0) [2022;](#page-15-0) [Bai et al.,](#page-10-0) [2022b;](#page-10-0) [Pang et al.,](#page-13-2) [2023;](#page-13-2) [Yuan et al.,](#page-16-0) [2024\)](#page-16-0) has raised the possibility of "self-improvement," where a model—typically through forms of self-play or self-training in which the model critiques its own generations—learns to improve on its own, without external feedback. This phenomenon is somewhat counterintuitive; at first glance it would seem to disagree with the well-known data-processing inequality [\(Cover,](#page-11-1) [1999\)](#page-11-1), which implies that no form of self-training should be able to create

<sup>∗</sup>Equal contribution.

![](_page_1_Figure_1.jpeg)

Figure 1: Validation of maximum-likelihood sharpening, via Best-of-N (BoN) sampling, at inference time. (a) Percent accuracy improvement over greedy decoding for BoN sharpening with N = 50 on 6 tasks and 7 models, colored by performance. (b) Percent accuracy improvement over greedy for BoN sharpening as a function of N for 7 different models on the MATH dataset. (c) Distribution of sequence-level log probabilities for responses sampled from Phi3.5-Mini (N = 1) on the MATH dataset, conditioned on correctness. Correct completions have noticeably higher likelihood than incorrect completions, demonstrating the utility of inference-time sharpening.

information not already in the model. This motivates the question of why we should expect such supervision-free interventions will lead to stronger reasoning and planning capabilities.

A dominant hypothesis for why improvement without external feedback might be possible is that models contain "hidden knowledge" [\(Hinton et al.,](#page-12-2) [2015\)](#page-12-2) that is difficult to access. Self-improvement, rather than creating knowledge from nothing, is a means of extracting and distilling this knowledge into a more accessible form, and thus is a computational phenomenon rather than a statistical one. While there is a growing body of empirical evidence for this hidden-knowledge hypothesis [\(Furlanello](#page-12-3) [et al.,](#page-12-3) [2018;](#page-12-3) [Gotmare et al.,](#page-12-4) [2019;](#page-12-4) [Dong et al.,](#page-11-2) [2019;](#page-11-2) [Abnar et al.,](#page-10-1) [2020;](#page-10-1) [Allen-Zhu & Li,](#page-10-2) [2020\)](#page-10-2), particularly in the context of self-distillation, a fundamental understanding of self-improvement remains missing. Concretely, where in the model is this hidden knowledge, and when and how can it be extracted?

#### 1.1 OUR PERSPECTIVE: THE SHARPENING MECHANISM

In this paper we posit a source of hidden knowledge, and offer a formal perspective on how to extract it. Our starting point is the widely observed phenomenon that language models are often better at verifying whether responses are correct than they are at generating correct responses [\(Huang et al.,](#page-12-1) [2022;](#page-12-1) [Wang et al.,](#page-15-0) [2022;](#page-15-0) [Bai et al.,](#page-10-0) [2022b;](#page-10-0) [Pang et al.,](#page-13-2) [2023;](#page-13-2) [Yuan et al.,](#page-16-0) [2024\)](#page-16-0). This gap may be explained by the theory of computational complexity, which suggests that generating high-quality responses can be less computationally tractable than verification [\(Cook,](#page-11-3) [1971;](#page-11-3) [Levin,](#page-13-3) [1973;](#page-13-3) [Karp,](#page-13-4) [1972\)](#page-13-4). In autoregressive language modeling, computing the most likely response for a given prompt is NP-hard in the worst case [\(Appendix E\)](#page-30-0), whereas the model's likelihood for a given response can be easily evaluated.

We view self-improvement as any attempt to narrow this gap, i.e., use the model as its own verifier to improve generation and *sharpen* the model toward high-quality responses. Formally, consider a learner with access to a base model πbase : X → ∆(Y) representing a conditional distribution that maps a prompt x ∈ X to a distribution over responses (i.e., πbase(y | x) is the probability that the model generates the response y given the prompt x).[<sup>1</sup>](#page-0-0) We posit that πbase has already been trained in some manner (e.g., through next-token prediction or additional post-training steps such as SFT or RLHF), with the key feature being that πbase is a good verifier, as measured by some *self-reward* function rself(y | x; πbase) measuring model certainty. The self-reward function is derived purely from the base model πbase, without external supervision or feedback. Examples include normalized and/or regularized sequence likelihood [\(Meister et al.,](#page-13-5) [2020\)](#page-13-5), models-as-judges [\(Zheng et al.,](#page-16-1) [2024;](#page-16-1) [Yuan et al.,](#page-16-0) [2024;](#page-16-0) [Wu et al.,](#page-15-1) [2024a;](#page-15-1) [Wang et al.,](#page-15-2) [2024\)](#page-15-2), and model confidence [\(Wang & Zhou,](#page-15-3) [2024\)](#page-15-3).

<sup>1</sup>Our general results are agnostic to the structure of X , Y, and πbase, but an important special case for language modeling is the autoregressive setting where Y = V <sup>H</sup> for a vocabulary space V and sequence length H, and where πbase has the autoregressive structure πbase(y1:<sup>H</sup> | x) = Q<sup>H</sup> <sup>h</sup>=1 πbase,h(y<sup>h</sup> | y1:h−1, x) for y = y1:<sup>H</sup> ∈ Y.

#### Sharpening

We refer to sharpening as any process that tilts πbase toward responses that are more certain in the sense that they enjoy greater self-reward <sup>r</sup>self. That is, a sharpened model <sup>π</sup>b is one that (approximately) maximizes the self-reward:

$$\hat{\pi}(x) \approx \arg \max_{y \in \mathcal{Y}} r_{\text{self}}(y \mid x; \pi_{\text{base}}). \quad (1)$$

An important special case for sharpening is in language/autoregressive modeling. Here, we have Y = V <sup>H</sup> for a vocabulary space V and sequence length H, and πbase has the autoregressive structure <sup>π</sup>base(y1:<sup>H</sup> | <sup>x</sup>) = Q<sup>H</sup> <sup>h</sup>=1 πbase,h(y<sup>h</sup> | y1:h−1, x) for y = y1:<sup>H</sup> ∈ Y. Sharpening in this setting pertains to entire responses, i.e., the optimization over responses in [Eq. \(1\)](#page-2-0) is at the *sequence level*. In contrast, popular decoding strategies such as greedy, low-temperature sampling, and beam search operate at the token-level; nevertheless, they can be viewed as heuristics for *inference-time sharpening*. [<sup>2</sup>](#page-0-0) The combinatorial response space can make sharpening computationally demanding and so, an appealing alternative to inference-time sharpening is *amortization via self-training* [\(Section 2\)](#page-3-0). The latter captures many existing self-training schemes [\(Huang et al.,](#page-12-1) [2022;](#page-12-1) [Wang et al.,](#page-15-0) [2022;](#page-15-0) [Bai et al.,](#page-10-0) [2022b;](#page-10-0) [Pang et al.,](#page-13-2) [2023;](#page-13-2) [Yuan et al.,](#page-16-0) [2024\)](#page-16-0), and is the main focus of this paper; we use the term *sharpening* without further qualification to refer to the latter.

We refer to the sharpening mechanism as the phenomenon where responses from a model with the highest certainty (in the sense of large self-reward rself) exhibit the greatest performance on a task of interest. Though it is unclear a-priori whether there are self-rewards related to task performance, the successes of self-improvement in prior works [\(Huang et al.,](#page-12-1) [2022;](#page-12-1) [Wang et al.,](#page-15-0) [2022;](#page-15-0) [Bai et al.,](#page-10-0) [2022b;](#page-10-0) [Pang et al.,](#page-13-2) [2023;](#page-13-2) [Yuan et al.,](#page-16-0) [2024\)](#page-16-0) give strong positive evidence. These works suggest that, in many settings, models do have hidden knowledge: the model's own self-reward correlates with response quality, but it is computationally challenging to generate high self-rewarding—and thus high quality—responses. It is the role of (algorithmic) sharpening to leverage these verifications to improve the quality of generations, despite computational difficulty.

# 1.2 CONTRIBUTIONS

We initiate the theoretical study of self-improvement via the sharpening mechanism. We disentangle the choice of self-reward from the algorithms used to optimize it, and aim to understand: (i) When and how does self-training achieve sharpening? (ii) What are the fundamental limits for such algorithms?

Algorithms for sharpening [\(Section 2\)](#page-3-0). The starting point for our work is to consider two natural families of self-improvement algorithms based on supervised fine-tuning (SFT) and reinforcement learning (RL/RLHF), respectively, SFT-Sharpening and RLHF-Sharpening. Both algorithms amortize the sharpening objective [\(1\)](#page-2-0) into a dedicated post-training/fine-tuning phase:

- SFT-Sharpening filters responses where the self-reward rself(y | x; πbase) is large and fine-tunes on the resulting dataset, invoking common SFT pipelines [\(Amini et al.,](#page-10-3) [2024;](#page-10-3) [Sessa et al.,](#page-14-1) [2024\)](#page-14-1).
- RLHF-Sharpening directly applies reinforcement learning techniques (e.g., PPO [\(Schulman et al.,](#page-14-2) [2017\)](#page-14-2) or DPO [\(Rafailov et al.,](#page-14-3) [2023\)](#page-14-3)) to optimize the self-reward function rself(y | x; πbase).

In the remainder of the paper, we introduce a theoretical framework to analyze the performance of these algorithms. Our main contributions are as follows.

Maximum-likelihood sharpening objective [\(Section 3.1\)](#page-4-0). As a concrete proposal for one source of hidden knowledge, we focus on self-rewards defined by the model's sequence-level log-probabilities:

$$r_{\text{self}}(y \mid x; \pi_{\text{base}}) := \log \pi_{\text{base}}(y \mid x) \quad (2)$$

This is a stylized self-reward function, which offers perhaps the simplest objective for selfimprovement in the absence of external feedback (i.e., purely supervision-free), yet also connects self-improvement to a rich body of theoretical computer science literature on computational trade-offs for optimization (inference) versus sampling [\(Appendix B\)](#page-25-0). Despite its simplicity, maximum-likelihood sharpening is already sufficient to achieve non-trivial performance gains over

<sup>2</sup>More sophisticated decoding strategies like normalized/regularized sequence likelihood [\(Meister et al.,](#page-13-5) [2020\)](#page-13-5) or chain-of-thought decoding [\(Wang & Zhou,](#page-15-3) [2024\)](#page-15-3) also admit an interpretation as sharpening; see [Appendix B.](#page-25-0)

greedy decoding on a range of reasoning tasks with several language models; [\(Figure 1\)](#page-1-0). We believe it can serve as a starting point toward understanding forms of self-improvement that use more sophisticated self-rewards [\(Huang et al.,](#page-12-1) [2022;](#page-12-1) [Wang et al.,](#page-15-0) [2022;](#page-15-0) [Pang et al.,](#page-13-2) [2023;](#page-13-2) [Yuan et al.,](#page-16-0) [2024\)](#page-16-0).

A statistical framework for sharpening [\(Sections 3.2](#page-5-0) and [3.3\)](#page-6-0). Though the goal of sharpening is computational in nature, we recast self-training according to the maximum-likelihood sharpening objective [Eq. \(2\)](#page-2-1) as a statistical problem where we aim to produce a model approximating [\(1\)](#page-2-0) using a polynomial number of (i) sample prompts x ∼ µ, (ii) sampling queries of the form y ∼ πbase(x), and (iii) likelihood evaluations of the form πbase(y | x). Evaluating the efficiency of the algorithm through the number of such queries, this abstraction offers a natural way to evaluate the performance of self-improvement/sharpening algorithms and establish fundamental limits; we use our framework to prove new lower bounds that highlight the importance of the base model's coverage.

Analysis of sharpening algorithms [\(Section 4\)](#page-6-1). Within our statistical framework for sharpening, we show that SFT-Sharpening and RLHF-Sharpening provably converge to sharpened models, establishing several results: (i) **SFT-Sharpening** is minimax optimal, and learns a sharpened model whenever πbase has sufficient coverage (we also show that a novel variant based on adaptive sampling can sidestep the minimax lower bound); (ii) **RLHF-Sharpening** benefits from on-policy exploration, and can bypass the need for coverage—improving over SFT-Sharpening.

Empirical investigation [\(Appendix A\)](#page-18-0). We explore empirically the extent to which our theoretical framework and methods improve language model performance in a variety of tasks. We consider three choices of self-reward on an extensive list of model-dataset pairs and conclude that sharpening can often improve performance. We then implement one of our algorithms, SFT-Sharpening, on a subset of these model-dataset pairs and observe a significant positive effect on performance. A summary of our inference-time experiments can be found in [Figure 1.](#page-1-0)

### 1.3 RELATED WORK

Our work is most directly related to a growing body of empirical research that studies self-training for language models in a supervision-free setting with no external feedback [\(Huang et al.,](#page-12-1) [2022;](#page-12-1) [Wang et al.,](#page-15-0) [2022;](#page-15-0) [Bai et al.,](#page-10-0) [2022b;](#page-10-0) [Pang et al.,](#page-13-2) [2023;](#page-13-2) [Yuan et al.,](#page-16-0) [2024\)](#page-16-0). The specific algorithms for self-improvement/sharpening we study can be viewed as applications of standard alignment algorithms [\(Amini et al.,](#page-10-3) [2024;](#page-10-3) [Sessa et al.,](#page-14-1) [2024;](#page-14-1) [Christiano et al.,](#page-11-4) [2017;](#page-11-4) [Bai et al.,](#page-10-4) [2022a;](#page-10-4) [Ouyang](#page-13-0) [et al.,](#page-13-0) [2022;](#page-13-0) [Rafailov et al.,](#page-14-3) [2023\)](#page-14-3) with a specific choice of reward function. However, the maximum likelihood sharpening objective [\(2\)](#page-2-1) used for our theoretical results has been relatively unexplored within the alignment and self-improvement literature.

Theoretical understanding of self-training is currently limited. One line of work analyzes the convergence of self-training for classification and regression with the *self-distillation objective*, but is limited to stylized setups such as linear models [\(Mobahi et al.,](#page-13-6) [2020;](#page-13-6) [Frei et al.,](#page-12-5) [2022;](#page-12-5) [Das](#page-11-5) [& Sanghavi,](#page-11-5) [2023;](#page-11-5) [Das et al.,](#page-11-6) [2024;](#page-11-6) [Pareek et al.,](#page-13-7) [2024\)](#page-13-7), feedforward neural networks [\(Allen-Zhu](#page-10-2) [& Li,](#page-10-2) [2020\)](#page-10-2), and a general PAC-style framework [\(Boix-Adsera,](#page-11-7) [2024\)](#page-11-7). To the best of our knowledge, our work is the first to study self-training in a general framework that subsumes language modeling. See [Appendix B](#page-25-0) for a more extensive discussion of related work.

# 2 SHARPENING ALGORITHMS FOR SELF-IMPROVEMENT

This section introduces the two families of self-improvement algorithms for sharpening that we study. Going forward, we omit the dependence of rself on πbase when it is clear from context. We use the notation arg maxπ∈<sup>Π</sup> or arg minπ∈<sup>Π</sup> to denote exact optimization over a user-specified model class Π for theoretical results [\(Agarwal et al.,](#page-10-5) [2019;](#page-10-5) [Foster & Rakhlin,](#page-12-6) [2023\)](#page-12-6); empirically, these operations can be implemented by training a neural network to low loss.

# 2.1 SELF-IMPROVEMENT THROUGH SFT: SFT-Sharpening

SFT-Sharpening filters responses for which the self-reward rself(y | x) is large, and applies standard supervised fine-tuning on the resulting dataset [\(Amini et al.,](#page-10-3) [2024;](#page-10-3) [Sessa et al.,](#page-14-1) [2024;](#page-14-1) [Gui et al.,](#page-12-7) [2024;](#page-12-7) [Pace et al.,](#page-13-8) [2024\)](#page-13-8). This can be viewed as amortizing inference-time sharpening via the effective-but-costly best-of-N sampling approach [\(Brown et al.,](#page-11-8) [2024;](#page-11-8) [Snell et al.,](#page-14-4) [2024;](#page-14-4) [Wu et al.,](#page-15-4) [2024b\)](#page-15-4). Concretely, suppose we have a collection of prompts x1, . . . , xn. For each prompt, we sample N responses yi,1, . . . , yi,N ∼ πbase(· | xi), then compute the best-of-N response y BoN <sup>i</sup> = arg maxj∈[N]{rself(yi,j | xi)}, scoring via the model's self-reward function. We compute the sharpened model via supervised fine-tuning on the best-of-N responses:

$$\hat{\pi}^{\text{BoN}} = \arg \max_{\pi \in \Pi} \sum_{i=1}^n \log \pi(y_i^{\text{BoN}} \mid x_i).$$

This is a simple, flexible self-training scheme, and converges to a sharpened model as n, N → ∞.

#### 2.2 SELF-IMPROVEMENT THROUGH RLHF: RLHF-Sharpening

A drawback of the SFT-Sharpening algorithm is that it may ignore useful information contained in the self-reward function rself(y | x). Fixing a regularization parameter β > 0 throughout, our second class of algorithms solve a KL-regularized reinforcement learning problem in the spirit of RLHF and other alignment methods [\(Christiano et al.,](#page-11-4) [2017;](#page-11-4) [Rafailov et al.,](#page-14-3) [2023\)](#page-14-3). Defining <sup>E</sup>π[·] = <sup>E</sup>x∼µ,y∼π(·|x) [·] and DKL(π ∥ πbase) = <sup>E</sup><sup>π</sup> log <sup>π</sup>(y|x) πbase(y|x) , we choose

$$\hat{\pi} \approx \arg \max_{\pi \in \Pi} \{\mathbb{E}_{\pi}[r_{\text{self}}(y \mid x)] - \beta D_{\text{KL}}(\pi \parallel \pi_{\text{base}})\}. \quad (3)$$

The exact optimizer π ⋆ <sup>β</sup> = arg maxπ∈Π{<sup>E</sup>π[rself(y | x)] − βDKL(π ∥ πbase)} for this objective has the form π ⋆ β (y | x) ∝ πbase(y | x) · exp β −1 rself(y | x) , which converges to the solution to the sharpening objective in [Eq. \(1\)](#page-2-0) as β → 0. Thus, [Eq. \(3\)](#page-4-1) can be seen to encourage sharpening.

There are many choices for what RLHF/alignment algorithm one might use to solve [\(3\).](#page-4-1) For our theoretical results, we implement [Eq. \(3\)](#page-4-1) using an approach inspired by DPO and its reward-based variants [\(Rafailov et al.,](#page-14-3) [2023;](#page-14-3) [Gao et al.,](#page-12-8) [2024\)](#page-12-8). Given a dataset D = {(x, y, y′ )} of n examples sampled via x ∼ µ and y, y′ ∼ πbase(y | x), we consider the algorithm that solves

$$\widehat{\pi} \in \arg \min_{\pi \in \Pi} \sum_{(x,y,y') \in \mathcal{D}} \left( \beta \log \frac{\pi(y | x)}{\pi_{\text{base}}(y | x)} - \beta \log \frac{\pi(y' | x)}{\pi_{\text{base}}(y' | x)} - (r_{\text{self}}(y | x) - r_{\text{self}}(y' | x)) \right)^2. \quad (4)$$

In [Section 4,](#page-6-1) we show that this approach achieves guarantees similar to SFT-Sharpening, while a more sophisticated DPO variant with *online exploration* [\(Xie et al.,](#page-15-5) [2024\)](#page-15-5) provides provable benefits.

# 3 A STATISTICAL FRAMEWORK FOR SHARPENING

This section introduces the theoretical framework within which we will analyze the SFT-Sharpening and RLHF-Sharpening algorithms. We first introduce the maximum-likelihood sharpening objective as a stylized self-reward function, then introduce our statistical framework for sharpening. We write <sup>f</sup> <sup>=</sup> <sup>O</sup>e(g) to denote <sup>f</sup> <sup>=</sup> <sup>O</sup>(<sup>g</sup> · max{1, polylog(g)}) and <sup>a</sup> ≲ <sup>b</sup> as shorthand for <sup>a</sup> <sup>=</sup> <sup>O</sup>(b).

## 3.1 MAXIMUM-LIKELIHOOD SHARPENING

Our theoretical results focus on the maximum-likelihood sharpening objective given by

$$r_{\text{self}}(y \mid x) := \log \pi_{\text{base}}(y \mid x),$$

which we aim to maximize using conditional samples y ∼ πbase(· | x) from the base model. This is a simple and stylized self-reward function, but we will show that it enjoys a rich theory. In particular, we can restate the problem of sharpening with this self-reward through the lens of *amortization*.

*Can we efficiently amortize maximum likelihood inference (optimization) for a conditional distribution* πbase(y | x) *given access to a sampling oracle that can sample* y ∼ πbase(· | x)*?*

The tacit assumption in this framing is that the maximum-likelihood response constitutes a useful form of hidden knowledge. Maximum-likelihood sharpening connects the study of self-improvement to a large body of research in theoretical computer science demonstrating computational reductions between optimization (inference) and sampling (generation) [\(Kirkpatrick et al.,](#page-13-9) [1983;](#page-13-9) [Lovász & Vem](#page-13-10)[pala,](#page-13-10) [2006;](#page-13-10) [Singh & Vishnoi,](#page-14-5) [2014;](#page-14-5) [Ma et al.,](#page-13-11) [2019;](#page-13-11) [Talwar,](#page-14-6) [2019\)](#page-14-6). Our sharpening framework offers a new learning-theoretic perspective by focusing on the problem of amortizing this type of reduction.

$$\mathbf{y}^*(x) := \arg \max_{y \in \mathcal{Y}} \log \pi_{\text{base}}(y \mid x);$$

we interpret y ⋆ (x) ⊂ Y as a set to accommodate non-unique maximizers, and will write y ⋆ (x) to indicate a unique maximizer when it exists (i.e., when y ⋆ (x) = {y ⋆ (x)}).

Definition 3.1 (Sharpened model). *We say that a model* <sup>π</sup>b *is* (ϵ, δ)*-sharpened relative to* <sup>π</sup>base *if*

$$\mathbb{P}_{x \sim \mu}[\widehat{\pi}(\mathbf{y}^*(x) \mid x) \geq 1 - \delta] \geq 1 - \epsilon.$$

That is, an (ϵ, δ)-sharpened model places at least 1 − δ mass on arg-max responses on all but an <sup>ϵ</sup>-fraction of prompts under <sup>µ</sup>. For small <sup>δ</sup> and <sup>ϵ</sup>, we are guaranteed that <sup>π</sup>b is a high-quality generator: sampling from the model will produce an arg-max response with high probability for most prompts.

Maximum-likelihood sharpening for autoregressive models. Though our most general results are agnostic to the structure of X , Y, and πbase, our primary motivation is the autoregressive setting in which Y = V <sup>H</sup> for a *vocabulary space* V and sequence length H, and where πbase has the autoregressive structure <sup>π</sup>base(y1:<sup>H</sup> | <sup>x</sup>) = Q<sup>H</sup> <sup>h</sup>=1 πbase,h(y<sup>h</sup> | y1:h−1, x) for y = y1:<sup>H</sup> ∈ Y. We observe that when the response y = (y1, . . . , yH) ∈ Y = V <sup>H</sup> is a sequence of tokens, the maximum-likelihood sharpening objective [\(2\)](#page-2-1) sharpens toward the *sequence-level* arg-max response:

$$\arg \max_{y_{1:H}} \log \pi_{\text{base}}(y_{1:H} \mid x). \quad (5)$$

Although somewhat stylized, [Eq. \(5\)](#page-4-2) is a non-trivial (in general, computationally intractable; see [Appendix E\)](#page-30-0) solution concept. We view the sequence-level arg-max as a form of hidden knowledge that cannot necessarily be uncovered through naive sampling or greedy decoding.

Role of δ for autoregressive models. As can be verified through simple examples, beam-search and greedy tokenwise decoding do not return an exact (or even approximate) solution to [\(5\)](#page-4-2) in general. There is one notable exception: If the model has already been sharpened to δ < 1/2 and the arg-max sequence is unique, then greedy decoding will succeed.

Proposition 3.1 (Greedy decoding succeeds for sharpened policies). *Let* π = π1:<sup>H</sup> *be an autoregressive model defined over response space* Y = V <sup>H</sup>*. For a given prompt* x ∈ X *, if* y ⋆ (x) = {y ⋆ (x)} *is a singleton and* π(y ⋆ (x) | x) > 1/2*, then the greedy decoding strategy that selects* <sup>y</sup>b<sup>h</sup> = arg max<sup>y</sup>h∈V <sup>π</sup>h(y<sup>h</sup> | <sup>y</sup>b1, . . . , <sup>y</sup>bh−1, x) *guarantees that* <sup>y</sup>b <sup>=</sup> <sup>y</sup> ⋆ (x)*. This result is tight, in the sense that there exist* π *with* π(y ⋆ (x) | x) ≤ 1/2 *for which greedy decoding fails to recover* y ⋆ (x)*.*

This means that if we start from an un-sharpened model, simply sharpening to δ < 1/2 may suffice.

## 3.2 SAMPLE COMPLEXITY FRAMEWORK

Sharpening, as described in [Definition 3.1,](#page-5-1) is a purely computational problem, which makes it difficult to evaluate the optimality of self-improvement algorithms. To address this, we introduce a novel statistical framework for sharpening, inspired by the oracle complexity in optimization [\(Nemirovski](#page-13-12) [et al.,](#page-13-12) [1983;](#page-13-12) [Traub et al.,](#page-15-6) [1988;](#page-15-6) [Raginsky & Rakhlin,](#page-14-7) [2011;](#page-14-7) [Agarwal et al.,](#page-10-6) [2012\)](#page-10-6) and statistical query complexity in computational learning theory [\(Blum et al.,](#page-11-9) [1994;](#page-11-9) [Kearns,](#page-13-13) [1998;](#page-13-13) [Feldman,](#page-11-10) [2012;](#page-11-10) [2017\)](#page-12-9).

Definition 3.2 (Sample-and-evaluate framework). *In the sample-and-evaluate framework, the algorithm designer does not have explicit access to the base model* πbase*. Instead, they access* πbase *only through* sample-and-evaluate queries*: The learner is allowed to sample* n *prompts* x ∼ µ*. For each prompt* x*, they can sample* N *responses* y1, y2, . . . y<sup>N</sup> ∼ πbase(· | x) *and observe the likelihood* πbase(y<sup>i</sup> | x) *for each such response. The efficiency, or* sample complexity*, of the algorithm is measured through the total number of sample-and-evaluate queries* m := n · N*.*

This framework can be seen to capture algorithms like SFT-Sharpening and RLHF-Sharpening (implemented with DPO), which only access the base model πbase through i) sampling responses via y ∼ πbase(· | x) (generation), and ii) evaluating the likelihood πbase(y | x) (verification) for these responses. We view the sample complexity m = n · N as a natural statistical abstraction for the computational complexity of self-improvement (a clear parallel to oracle complexity for optimization algorithms), one which is amenable to information-theoretic lower bounds.[<sup>3</sup>](#page-0-0) We will aim to show that, under appropriate assumptions, SFT-Sharpening and RLHF-Sharpening can learn an (ϵ, δ)-sharpened model with sample complexity

$$m = \text{poly}(\epsilon^{-1}, \delta^{-1}, C_{\text{prob}})$$

<sup>3</sup>Concretely, the sample complexity m = n · N is a lower bound on the running time of any algorithm that operates in the sample-and-evaluate framework.

#### 3.3 FUNDAMENTAL LIMITS

Before diving into our analysis of SFT-Sharpening and RLHF-Sharpening in the sample-andevaluate framework, let us take a brief detour to give a sense for how sample complexity guarantees for sharpening should scale. To this end, we will prove a lower bound or fundamental limit on the sample complexity of any algorithm in the sample-and-evaluate framework.

Intuitively, the performance of any sampling-based sharpening algorithm should depend on well the base model πbase covers the arg-max response y ⋆ (x). To capture this, we use the *coverage coefficient*[<sup>4</sup>](#page-0-0)

$$C_{\text{cov}} = \mathbb{E}_{x \sim \mu} \left[ \frac{1}{\pi_{\text{base}}(\mathbf{y}^*(x) \mid x)} \right], \quad (6)$$

and, for a model π, we define y π (x) = arg maxy∈Y π(y | x) and Ccov(π) = <sup>E</sup>x∼<sup>µ</sup> h π(y<sup>π</sup>(x)|x) i .

Our main lower bound shows that for a worst-case choice of Π, the coverage coefficient serves as a lower bound on the sample complexity of any sharpening algorithm.

Theorem 3.1 (Lower bound for sharpening). *Fix an integer* d ≥ 1 *and parameters* ϵ ∈ (0, 1) *and* C ≥ 1*. There exists a class of models* Π *such that (i)* log |Π| ≍ d(1 + log(Cϵ−<sup>1</sup> ))*, (ii)* supπ∈<sup>Π</sup> Ccov(π) ≲ C*, and (iii)* y π (x) *is a singleton for all* π ∈ Π*,* x ∈ X *. Any sharpening algorithm* <sup>π</sup>b *that achieves* <sup>E</sup>[<sup>P</sup>x∼µ[πb(<sup>y</sup> <sup>π</sup>base (x) | x) > 1/2]] ≥ 1 − ϵ *for all* πbase ∈ Π *must collect a total number of samples* m = n · N *at least*

$$m \gtrsim \frac{C \log |\Pi|}{\epsilon^2 \cdot (1 + \log(C\epsilon^{-1}))}.$$

This result shows that the complexity of any (ϵ, 1/2 − δ)-sharpening algorithm (for δ > 0) in the sample-and-evaluate framework must depend polynomially on the coverage coefficient Ccov, as well as the accuracy ϵ. The lower bound also depends on the expressivity of πbase, as captured by the model class complexity term log|Π|. We will show in the sequel that it is possible to match this lower bound. Note that this result also implies a lower bound for the general sharpening problem (i.e., general rself), since maximum-likelihood sharpening is a special case.

Remark 3.1 (Relaxed notions of sharpening and coverage). *The notion of coverage in [Eq. \(6\)](#page-5-2) is somewhat stringent, since it requires that* πbase *place large mass on* y ⋆ (x) *on average. In [Appendix F,](#page-33-0) we introduce a more general and permissive notion of* approximate sharpening *[\(Definition F.1\)](#page-33-1) which leads to weaker coverage requirements, and use this to give generalized versions of our main results.*

We close this section by noting that numerous recent works—focusing on inference-time computation—show that standard language models exhibit favorable coverage with respect to desirable responses [\(Brown et al.,](#page-11-8) [2024;](#page-11-8) [Snell et al.,](#page-14-4) [2024;](#page-14-4) [Wu et al.,](#page-15-4) [2024b\)](#page-15-4). We replicate these findings in our experimental setup in [Appendix A.](#page-18-0) These works suggest that, despite the exponentially large response space, the coverage coefficient Ccov may be small in standard language modeling tasks.

## 4 ANALYSIS OF SHARPENING ALGORITHMS

Equipped with the sample complexity framework from [Section 3,](#page-4-3) we now prove that the SFT-Sharpening and RLHF-Sharpening families of algorithms provably learn a sharpened model for the maximum likelihood sharpening objective. We treat the model class Π as a fixed, user-specified input. In the tradition of statistical learning theory, our results allow for general classes Π and are agnostic to its structure beyond standard generalization arguments.

#### 4.1 ANALYSIS OF SFT-Sharpening

Recall that when we specialize to the maximum-likelihood sharpening self-reward, the SFT-Sharpening algorithm takes the form <sup>π</sup>b BoN = arg maxπ∈<sup>Π</sup> P<sup>n</sup> <sup>i</sup>=1 log πbase(y BoN i | xi), where y BoN <sup>i</sup> = arg maxj∈[N]{log πbase(yi,j | xi)} for yi,1, . . . , yi,N ∼ πbase(· | xi).

To analyze SFT-Sharpening, we first make a realizability assumption. Let π BoN <sup>N</sup> (x) be the distribution of the random variable y BoN <sup>N</sup> (x) ∼ arg max{log πbase(y<sup>i</sup> | x) | y1, . . . , y<sup>N</sup> ∼ πbase(x)}.

<sup>4</sup>This quantity can be interpreted as a special case of the L1-concentrability coefficient [\(Farahmand et al.,](#page-11-11) [2010;](#page-11-11) [Xie & Jiang,](#page-15-7) [2020;](#page-15-7) [Zanette et al.,](#page-16-2) [2021;](#page-16-2) [Amortila et al.,](#page-10-7) [2024\)](#page-10-7) studied in the theory of offline reinforcement learning.

Assumption 4.1. *The model class* Π *satisfies* π BoN <sup>N</sup> ∈ Π*.*

Our main sample complexity guarantee for SFT-Sharpening is as follows.

Theorem 4.1 (Sample complexity of SFT-Sharpening). *Let* ϵ, δ, ρ ∈ (0, 1) *be given, and suppose we set* n = c · log(|Π|ρ −1 ) δϵ *and* <sup>N</sup><sup>⋆</sup> <sup>=</sup> <sup>c</sup> · Ccov log(2δ −1 ) ϵ *for an appropriate constant* c > 0*. Then with probability at least* <sup>1</sup> <sup>−</sup> <sup>ρ</sup>*,* SFT-Sharpening *produces a model* <sup>π</sup>b *such that that* <sup>P</sup>x∼µ[πb(<sup>y</sup> ⋆ (x) | x) ≤ 1 − δ] ≤ ϵ*, and has total sample complexity*[<sup>5</sup>](#page-0-0)

$$m = O\left(\frac{C_{\text{cov}} \log(|\Pi|\rho^{-1}) \log(\delta^{-1})}{\delta\epsilon^2}\right). \quad (7)$$

This result shows that SFT-Sharpening is minimax optimal in the sample-and-evaluate framework when δ is constant. In particular, the bound in [Eq. \(7\)](#page-7-0) matches the lower bound in [Theorem 3.1](#page-6-2) up to polynomial dependence on δ and logarithmic factors. Whether the 1/δ factor in [Eq. \(7\)](#page-7-0) can be removed is an interesting technical question, but may not be practically consequential because—as discussed in [Section 3.2—](#page-5-0)the regime δ < 1/2 is most meaningful for autoregressive language modeling.

Remark 4.1 (On realizability and coverage). *Realizability assumptions such as [Assumption 4.1](#page-6-3) (which asserts that the class* Π *is powerful enough to model the distribution of the best-of-*N *responses) are standard in learning theory [\(Agarwal et al.,](#page-10-5) [2019;](#page-10-5) [Foster & Rakhlin,](#page-12-6) [2023\)](#page-12-6), though certainly non-trivial (see [Appendix E](#page-30-0) for a natural example where they may not hold). The coverage assumption, while also standard, when combined with the hypothesis that high-likelihood responses are desirable, suggests that* πbase *generates high-quality responses with reasonable probability. In general, doing so may require leveraging non-trivial* serial *computation at inference time via procedures such as Chain-of-Thought [\(Wei et al.,](#page-15-8) [2022\)](#page-15-8). Although recent work shows that such serial computation* cannot *be amortized [\(Li et al.,](#page-13-14) [2024;](#page-13-14) [Malach,](#page-13-15) [2023\)](#page-13-15),* SFT-Sharpening *instead amortizes the* parallel *computation of best-of-*N *sampling, and thus has different representational considerations.*

Benefits of adaptive sampling. SFT-Sharpening is optimal in the sample-and-evaluate framework, but we show in [Appendix D](#page-29-0) that a variant which selects the number of responses adaptively based on the prompt x can bypass this lower bound, improving the ϵ-dependence in [Eq. \(7\)](#page-7-0) from <sup>1</sup> ϵ <sup>2</sup> to <sup>1</sup> ϵ .

Empirical validation. In [Appendix A,](#page-18-0) we empirically investigate the benefits of BoN on a variety of model-dataset pairs. Our results, summarized in [Table 1](#page-24-0) and [Figs. 7](#page-25-1) and [8,](#page-26-0) broadly show that the aforementioned benefits of inference-time sharpening, to an extent, amortized at training time.

#### 4.2 ANALYSIS OF RLHF-Sharpening

We now turn our attention to theoretical guarantees for the RLHF-Sharpening algorithm family, which uses tools from reinforcement learning to optimize the self-reward function. When specialized to maximum-likelihood sharpening, the RL objective used by RLHF-Sharpening takes the form <sup>π</sup>b <sup>≈</sup> arg maxπ∈Π{<sup>E</sup>π[log <sup>π</sup>base(<sup>y</sup> | <sup>x</sup>)] <sup>−</sup> βDKL(<sup>π</sup> ∥ <sup>π</sup>base)} for β > <sup>0</sup>. The exact optimizer π ⋆ <sup>β</sup> = arg maxπ∈Π{<sup>E</sup>π[log πbase(y | x)] − βDKL(π ∥ πbase)} for this objective has the form π ⋆ β (y | x) ∝ π 1+β −1 base (y | x), which converges to a sharpened model (per [Definition 3.1\)](#page-5-1) as β → 0.

The key challenge we encounter in this section is the mismatch between the RL reward log πbase(y | <sup>x</sup>) and the sharpening desideratum <sup>π</sup>b(<sup>y</sup> ⋆ (x) | x). For example, suppose a unique argmax—say, y ⋆ (x)—and second-to-argmax—say, y ′ (x)—are nearly as likely under πbase. Then the RL reward <sup>E</sup>πb[log <sup>π</sup>base(<sup>y</sup> | <sup>x</sup>)] must be optimized to extremely high precision before <sup>π</sup>b can be guaranteed to distinguish the two. To quantify this effect, we introduce a *margin condition*.

Assumption 4.2 (Margin). *For a margin parameter* γmargin > 0*, the base model* πbase *satisfies*

$$\max_{y \in \mathcal{Y}} \pi_{\text{base}}(y \mid x) \geq (1 + \gamma_{\text{margin}}) \cdot \pi_{\text{base}}(y' \mid x) \quad \forall y' \notin \mathbf{y}^*(x), \quad \forall x \in \text{supp}(\mu).$$

SFT-Sharpening does not suffer from the pathology in the example above, because once y ⋆ (x) and y ′ (x) are drawn in a batch of N responses, we have y BoN <sup>i</sup> = y ⋆ (xi) regardless of margin. However, as we shall show in [Section 4.2.2,](#page-9-0) the RLHF-Sharpening algorithm is amenable to online exploration, which may improve dependence on other problem parameters.

<sup>5</sup>We focus on finite classes for simplicity, following a convention in reinforcement learning theory [\(Agarwal](#page-10-5) [et al.,](#page-10-5) [2019;](#page-10-5) [Foster & Rakhlin,](#page-12-6) [2023\)](#page-12-6), but our results extend to infinite classes through standard arguments.

#### 4.2.1 GUARANTEES FOR RLHF-Sharpening WITH DIRECT PREFERENCE OPTIMIZATION

The first of our theoretical results for RLHF-Sharpening takes an offline reinforcement learning approach, whereby we implement [Eq. \(3\)](#page-4-1) using a reward-based variant of Direct Preference Optimization (DPO) [\(Rafailov et al.,](#page-14-3) [2023;](#page-14-3) [Gao et al.,](#page-12-8) [2024\)](#page-12-8). Let Dpref = {(x, y, y′ )} be a dataset of n examples sampled via <sup>x</sup> <sup>∼</sup> <sup>µ</sup>, y, y′ <sup>∼</sup> <sup>π</sup>base(<sup>y</sup> <sup>|</sup> <sup>x</sup>). For a parameter β > <sup>0</sup>, we solve <sup>π</sup>b <sup>∈</sup> arg minπ∈<sup>Π</sup>

$$\sum_{(x,y,y') \in \mathcal{D}_{\text{pref}}} \left( \beta \log \frac{\pi(y \mid x)}{\pi_{\text{base}}(y \mid x)} - \beta \log \frac{\pi(y' \mid x)}{\pi_{\text{base}}(y' \mid x)} - (\log \pi_{\text{base}}(y \mid x) - \log \pi_{\text{base}}(y' \mid x)) \right)^2. \quad (8)$$

Assumptions. Per [Rafailov et al.](#page-14-3) [\(2023\)](#page-14-3), the solution to [Eq. \(8\)](#page-7-1) coincides with that of [Eq. \(2\)](#page-2-1) asymptotically. To provide finite-sample guarantees, we make a number of statistical assumptions. First, we make a natural realizability assumption (e.g., [Zhu et al.](#page-16-3) [\(2023\)](#page-16-3); [Xie et al.](#page-15-5) [\(2024\)](#page-15-5)).

Assumption 4.3 (Realizability). *The model class* Π *satisfies* π ⋆ <sup>β</sup> ∈ Π*.* [6](#page-0-0)

Next, we define two concentrability coefficients for a model π:

$$\mathcal{C}_\pi = \mathbb{E}_\pi \left[ \frac{\pi(y | x)}{\pi_{\text{base}}(y | x)} \right], \quad \text{and} \quad \mathcal{C}_{\pi/\pi';\beta} := \mathbb{E}_\pi \left[ \left( \frac{\pi(y | x)}{\pi'(y | x)} \right)^\beta \right]. \quad (9)$$

The following result shows that both coefficients are bounded for the KL-regularized model π ⋆ β .

Lemma 4.1. *The model* π ⋆ β *satisfies* C<sup>π</sup> ⋆ ≤ Ccov *and* C<sup>π</sup>base/π<sup>⋆</sup> β ;<sup>β</sup> ≤ |Y|*.*

Motivated by this result, we assume the coefficients in [Eq. \(9\)](#page-8-0) are bounded for all π ∈ Π.

Assumption 4.4 (Concentrability). *All* π ∈ Π *satisfy* C<sup>π</sup> ≤ Cconc *for a parameter* Cconc ≥ Ccov*, and* C<sup>π</sup>base/π;<sup>β</sup> ≤ Closs *for a parameter* Closs ≥ |Y|*.*

By [Lemma 4.1,](#page-8-1) this assumption is consistent with [Assumption 4.3](#page-8-2) for reasonable bounds on Cconc and Closs; note that our sample complexity bounds will only incur logarithmic dependence on Closs.

Main result. Our sample complexity guarantee for RLHF-Sharpening (via [Eq. \(8\)\)](#page-7-1) is as follows.

Theorem 4.2. *Let* ϵ, δ, ρ ∈ (0, 1) *be given. Set* β ≲ γmarginδϵ*, and suppose that [Assumptions 4.2](#page-7-2) to [4.4](#page-8-3) hold with parameters* Cconc*,* Closs*, and* γmargin > 0*. For an appropriate choice for* n*, the DPO algorithm [\(Eq. \(8\)\)](#page-7-1) ensures that with probability at least* <sup>1</sup> <sup>−</sup> <sup>ρ</sup>*,* <sup>P</sup>x∼µ[πb(<sup>y</sup> ⋆ (x) | x) ≤ 1 − δ] ≤ ϵ*, and has sample complexity*

$$m = \tilde{O}\left(\frac{C_{\text{conc}} \log^3(C_{\text{loss}}|\Pi|\rho^{-1})}{\gamma_{\text{margin}}^2 \delta^2 \epsilon^2}\right).$$

Compared to the guarantee for SFT-Sharpening, RLHF-Sharpening learns a sharpened model with the same dependence on the accuracy ϵ, but a worse dependence on δ; as we primarily consider δ constant (cf. [Proposition 3.1\)](#page-5-3), we view this as relatively unimportant. We further remark that RLHF-Sharpening uses N = 2 responses per prompt, while SFT-Sharpening uses many (N ≈ Ccov/ϵ) responses but fewer prompts. Other differences include:

- RLHF-Sharpening requires the margin condition in [Assumption 4.2,](#page-7-2) and has sample complexity scaling with γ −1 margin. We believe this dependence is natural for algorithms based on reinforcement learning, as it relates suboptimality with respect to the reward function rself(y | x) = log πbase(y |
  - x) (i.e., <sup>E</sup>x∼<sup>µ</sup>
  - maxy∈Y log <sup>π</sup>base(<sup>y</sup> | <sup>x</sup>) <sup>−</sup> <sup>E</sup>y∼πb(x) [log πbase(y | x)] ≤ ϵ, the objective minimized by reinforcement learning) to approximate sharpening error <sup>P</sup>x∼µ[πb(<sup>y</sup> ⋆
  - (x) | x) ≤ 1 − δ]. However, it is not clear if the precise dependence we pay is necessary.
- RLHF-Sharpening requires a bound on the uniform coverage parameter Cconc, which is generally larger than the parameter Ccov required by SFT-Sharpening. We expect that this assumption can be removed by incorporating pessimism [\(Liu et al.,](#page-13-16) [2024;](#page-13-16) [Huang et al.,](#page-12-10) [2024\)](#page-12-10). Also, RLHF-Sharpening requires a bound on the parameter Closs, which grants control over the (otherwise unbounded) range of the reward function log πbase(y | x). Since the dependence on Closs is only logarithmic, we view this as fairly mild. Overall, the guarantee in [Theorem 4.2](#page-8-4) may be somewhat pessimistic; it would be interesting if the result can be improved to match the sample complexity of SFT-Sharpening.

<sup>6</sup> See [Remark 4.1](#page-7-3) for a discussion of this assumption.

# 4.2.2 BENEFITS OF EXPLORATION

The guarantee in [Theorem 4.2](#page-8-4) scales with the coverage parameter Ccov = <sup>E</sup>[1/πbase(y ⋆ (x)|x)], which in general is unavoidable in the sample-and-evaluate framework via our lower bound, [The](#page-6-2)[orem 3.1.](#page-6-2) Although Ccov is a problem-dependent parameter, in the worst case it can be as large as |Y| (which is exponential in sequence length for autoregressive models). Fortunately, unlike SFT-Sharpening, the RLHF-Sharpening objective [\(3\)](#page-4-1) is amenable to RL algorithms employing active exploration, leading to improved sample complexity when the class Π has additional structure.

Our below guarantees for RLHF-Sharpening replace the assumption of bounded coverage with boundedness of a structural parameter for the model class Π known as the "sequential extrapolation coefficient" (SEC) [\(Xie et al.,](#page-15-9) [2023;](#page-15-9) [2024\)](#page-15-5), which we denote by SEC(Π). The formal definition is deferred to [Appendix J.2.](#page-46-0) Conceptually, SEC(Π) may thought of as a generalization of the eluder dimension [\(Russo & Van Roy,](#page-14-8) [2013;](#page-14-8) [Jin et al.,](#page-13-17) [2021\)](#page-13-17). It can always be bounded by the coverability coefficient of the model class [\(Xie et al.,](#page-15-5) [2024\)](#page-15-5) and can be as large as Cconc in the worst case, so that bounds based on the SEC reflect improvements that are possible in favorable instances.

Beyond boundedness of the SEC, we require a bound on the range of the log-probabilities of πbase.

Assumption 4.5 (Bounded log-probabilities). *For all* π ∈ Π*,* (x, y) ∈ X ×Y*,* log <sup>1</sup> πbase(y|x)  ≤ Rmax*.*

We expect that the dependence on Rmax in our result can be replaced with log(Closs) [\(Assump](#page-8-3)[tion 4.4\)](#page-8-3), but we omit this extension to simplify presentation.

We appeal to (a slight modification of) XPO, an iterative language model alignment algorithm due to [Xie et al.](#page-15-5) [\(2024\)](#page-15-5). XPO is based on the objective in [Eq. \(8\),](#page-7-1) but unlike DPO, incorporates a bonus term to encourage exploration to leverage online interaction. See [Appendix J.2](#page-46-0) for a detailed overview.

Theorem 4.3 (Informal version of [Theorem J.2\)](#page-49-0). *Suppose that [Assumptions 4.2](#page-7-2) and [4.5](#page-9-1) hold with parameters* γmargin, Rmax > 0*, and that [Assumption 4.3](#page-8-2) holds with* β = γmargin/(2 log(2|Y|/δ))*. For any* m ∈ N *and* ρ ∈ (0, 1)*,* XPO *[\(Algorithm 1\)](#page-47-0), when configured appropriately, produces an* (ϵ, δ)*-sharpened model* <sup>π</sup>b <sup>∈</sup> <sup>Π</sup> *with probability at least* <sup>1</sup> <sup>−</sup> <sup>ρ</sup>*, and uses sample complexity*[<sup>7</sup>](#page-0-0)

$$m = \tilde{O}\left(\frac{\text{SEC}(\Pi) \cdot \log(|\Pi|\rho^{-1})}{\gamma_{\text{margin}}^2 \delta^2 \epsilon^2}\right).$$

The takeaway from [Theorem 4.3](#page-9-2) is that there is no dependence on the coverage coefficient for πbase. Instead, the rate depends on the complexity of exploration, as governed by the sequential extrapolation coefficient SEC(Π). We expect similar guarantees can derived for other active exploration algorithms and complexity measures [\(Jiang et al.,](#page-13-18) [2017;](#page-13-18) [Foster et al.,](#page-12-11) [2021;](#page-12-11) [Jin et al.,](#page-13-17) [2021;](#page-13-17) [Xie et al.,](#page-15-9) [2023\)](#page-15-9).

# 5 CONCLUSION

We view our theoretical framework for sharpening as a starting point toward a foundational understanding of self-improvement that can guide the design and evaluation of algorithms. To this end, we raise several directions for future research.

- *Representation learning.* A conceptually appealing feature of our framework is that it is agnostic to the structure of the model under consideration, but an important direction for future work is to study the dynamics of self-improvement for specific models/architectures and understand the representations that these models learn under self-training.
- *Richer forms of self-reward.* Our theoretical results study the dynamics of self-training in a stylized framework where the model uses its own log-probabilities as a self-reward. Empirical research on self-improvement leverages more sophisticated approaches (e.g., specific prompting techniques) [\(Huang et al.,](#page-12-1) [2022;](#page-12-1) [Wang et al.,](#page-15-0) [2022;](#page-15-0) [Bai et al.,](#page-10-0) [2022b;](#page-10-0) [Pang et al.,](#page-13-2) [2023;](#page-13-2) [Yuan et al.,](#page-16-0) [2024\)](#page-16-0) and it is important to understand when and how these forms of self-improvement are beneficial.

<sup>7</sup>Technically, [Algorithm 1](#page-47-0) operates in a slight generalization of the sample-and-evaluate framework [\(Defini](#page-5-4)[tion 3.2\)](#page-5-4), where the algorithm is allowed to query πbase(y | x) for arbitrary x, y. We expect that our lower bound [\(Theorem 3.1\)](#page-6-2) can be extended to this more general framework, in which case [Algorithm 1](#page-47-0) is fundamentally using additional structure of Π (via the SEC) to avoid dependence on Ccov.

# ACKNOWLEDGMENTS

We thank Sivaraman Balakrishnan, Miro Dudík, Susan Dumais, John Langford, Qinghua Liu, and Yuda Song for helpful discussions.

# REFERENCES


[1] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. *arXiv:2404.14219*, 2024. Samira Abnar, Mostafa Dehghani, and Willem Zuidema. Transferring inductive biases through knowledge distillation. *arXiv:2006.00555*, 2020. Alekh Agarwal, Peter L Bartlett, Pradeep Ravikumar, and Martin J Wainwright. Information-theoretic lower bounds on the oracle complexity of stochastic convex optimization. *IEEE Transactions on Information Theory*, 2012. Alekh Agarwal, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. Taming the monster: A fast and simple algorithm for contextual bandits. In *International Conference on Machine Learning*, 2014. Alekh Agarwal, Nan Jiang, Sham M Kakade, and Wen Sun. Reinforcement learning: Theory and algorithms. <https://rltheorybook.github.io/>, 2019. Version: January 31, 2022. Zeyuan Allen-Zhu and Yuanzhi Li. Towards understanding ensemble, knowledge distillation and self-distillation in deep learning. *arXiv:2012.09816*, 2020. Afra Amini, Tim Vieira, and Ryan Cotterell. Variational best-of-n alignment. *arXiv:2407.06057*, 2024. Philip Amortila, Dylan J Foster, and Akshay Krishnamurthy. Scalable online exploration via coverability. In *Forty-first International Conference on Machine Learning*, 2024. Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback. *arXiv:2204.05862*, 2022a. Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. *arXiv:2212.08073*, 2022b. Francisco Barahona. On the computational complexity of ising spin glass models. *Journal of Physics A: Mathematical and General*, 1982. Matthew James Beal. *Variational algorithms for approximate Bayesian inference*. University of London, University College London, 2003. Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, and Yoshua Bengio. Flow network based generative models for non-iterative diverse candidate generation. *Advances in Neural Information Processing Systems*, 2021. Yoav Benjamini and Yosef Hochberg. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 1995. Adam Block, Dylan J Foster, Akshay Krishnamurthy, Max Simchowitz, and Cyril Zhang. Butterfly effects of SGD noise: Error amplification in behavior cloning and autoregression. *arXiv:2310.11428*, 2023.

[2] Avrim Blum, Merrick Furst, Jeffrey Jackson, Michael Kearns, Yishay Mansour, and Steven Rudich. Weakly learning DNF and characterizing statistical query learning using Fourier analysis. In *Symposium on Theory of Computing*, 1994. Enric Boix-Adsera. Towards a theory of model distillation. *arXiv preprint arXiv:2403.09053*, 2024. Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. *arXiv:2407.21787*, 2024. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In *Advances in Neural Information Processing Systems*, 2020. Cristian Bucilua, Rich Caruana, and Alexandru Niculescu-Mizil. Model compression. In ˇ *SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2006. Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models. *arXiv:2401.01335*, 2024. Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. *Advances in Neural Information Processing Systems*, 2017. Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. *arXiv:2110.14168*, 2021. Stephen A Cook. The complexity of theorem-proving procedures. In *Symposium on Theory of Computing*, 1971. Thomas M Cover. *Elements of information theory*. John Wiley & Sons, 1999. Rudrajit Das and Sujay Sanghavi. Understanding self-distillation in the presence of label noise. In *International Conference on Machine Learning*, 2023. Rudrajit Das, Inderjit S Dhillon, Alessandro Epasto, Adel Javanmard, Jieming Mao, Vahab Mirrokni, Sujay Sanghavi, and Peilin Zhong. Retraining with predicted hard labels provably increases model accuracy. *arXiv:2406.11206*, 2024. Jacob Devlin. Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv:1810.04805*, 2018. Bin Dong, Jikai Hou, Yiping Lu, and Zhihua Zhang. Distillation ≈ early stopping? Harvesting dark knowledge utilizing anisotropic information retrieval for overparameterized neural network. *arXiv:1910.01255*, 2019. Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. *arXiv:2407.21783*, 2024. Ronen Eldan, Frederic Koehler, and Ofer Zeitouni. A spectral condition for spectral gap: Fast mixing in high-temperature Ising models. *Probability Theory and Related Fields*, 2022. Amir-massoud Farahmand, Csaba Szepesvári, and Rémi Munos. Error propagation for approximate policy and value iteration. *Advances in Neural Information Processing Systems*, 2010. Vitaly Feldman. A complete characterization of statistical query learning with applications to evolvability. *Journal of Computer and System Sciences*, 2012.

[3] Vitaly Feldman. A general characterization of the statistical query complexity. In *Conference on Learning Theory*, 2017. Dylan J Foster and Alexander Rakhlin. Foundations of reinforcement learning and interactive decision making. *arXiv:2312.16730*, 2023. Dylan J Foster, Sham M Kakade, Jian Qian, and Alexander Rakhlin. The statistical complexity of interactive decision making. *arXiv:2112.13487*, 2021. Spencer Frei, Difan Zou, Zixiang Chen, and Quanquan Gu. Self-training converts weak learners to strong learners in mixture models. In *International Conference on Artificial Intelligence and Statistics*, 2022. Tommaso Furlanello, Zachary Lipton, Michael Tschannen, Laurent Itti, and Anima Anandkumar. Born again neural networks. In *International Conference on Machine Learning*, 2018. Zhaolin Gao, Jonathan D Chang, Wenhao Zhan, Owen Oertell, Gokul Swamy, Kianté Brantley, Thorsten Joachims, J Andrew Bagnell, Jason D Lee, and Wen Sun. REBEL: Reinforcement learning via regressing relative rewards. *arXiv:2404.16767*, 2024. Samuel Gershman and Noah Goodman. Amortized inference in probabilistic reasoning. In *Annual Meeting of the Cognitive Science Society*, 2014. Google. Palm 2 technical report. *arXiv:2305.10403*, 2023. Akhilesh Gotmare, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. A closer look at deep learning heuristics: Learning rate restarts, warmup and distillation. In *International Conference on Learning Representations*, 2019. Yves Grandvalet and Yoshua Bengio. Semi-supervised learning by entropy minimization. *Advances in Neural Information Processing Systems*, 2004. Lin Gui, Cristina Gârbacea, and Victor Veitch. BoNBoN alignment for large language models and the sweetness of best-of-n sampling. *arXiv:2406.00832*, 2024. Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. *arXiv:2009.03300*, 2020. Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. *arXiv:2103.03874*, 2021. Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. *arXiv:1503.02531*, 2015. Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. *arXiv:2106.09685*, 2021. Edward J Hu, Moksh Jain, Eric Elmoznino, Younesse Kaddar, Guillaume Lajoie, Yoshua Bengio, and Nikolay Malkin. Amortizing intractable inference in large language models. *arXiv:2310.04363*, 2023. Audrey Huang, Wenhao Zhan, Tengyang Xie, Jason D Lee, Wen Sun, Akshay Krishnamurthy, and Dylan J Foster. Correcting the mythos of KL-regularization: Direct alignment without overparameterization via Chi-squared Preference Optimization. *arXiv:2407.13399*, 2024. Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve. *arXiv:2210.11610*, 2022. Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. *arXiv:2310.06825*, 2023.

[4] Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire. Contextual decision processes with low Bellman rank are PAC-learnable. In *International Conference on Machine Learning*, 2017. Chi Jin, Qinghua Liu, and Sobhan Miryoosefi. Bellman Eluder dimension: New rich classes of RL problems, and sample-efficient algorithms. *Advances in Neural Information Processing Systems*, 2021. Richard M Karp. *Reducibility among combinatorial problems*. Springer, 1972. Michael Kearns. Efficient noise-tolerant learning from statistical queries. *Journal of the ACM*, 1998. Scott Kirkpatrick, C Daniel Gelatt Jr, and Mario P Vecchi. Optimization by simulated annealing. *Science*, 1983. Leonid Anatolevich Levin. Universal sequential search problems. *Problemy peredachi informatsii*, 1973. Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. Chain of thought empowers transformers to solve inherently serial problems. *arXiv:2402.12875*, 2024. Zhihan Liu, Miao Lu, Shenao Zhang, Boyi Liu, Hongyi Guo, Yingxiang Yang, Jose Blanchet, and Zhaoran Wang. Provably mitigating overoptimization in RLHF: Your SFT loss is implicitly an adversarial regularizer. *arXiv:2405.16436*, 2024. László Lovász and Santosh Vempala. Fast algorithms for logconcave functions: Sampling, rounding, integration and optimization. In *Symposium on Foundations of Computer Science*, 2006. Yi-An Ma, Yuansi Chen, Chi Jin, Nicolas Flammarion, and Michael I Jordan. Sampling can be faster than optimization. *Proceedings of the National Academy of Sciences*, 2019. Eran Malach. Auto-regressive next-token predictors are universal learners. *arXiv:2309.06979*, 2023. Clara Meister, Tim Vieira, and Ryan Cotterell. If beam search is the answer, what was the question? *arXiv:2010.02650*, 2020. Hossein Mobahi, Mehrdad Farajtabar, and Peter Bartlett. Self-distillation amplifies regularization in hilbert space. *Advances in Neural Information Processing Systems*, 2020. Sidharth Mudgal, Jong Lee, Harish Ganapathy, YaGuang Li, Tao Wang, Yanping Huang, Zhifeng Chen, Heng-Tze Cheng, Michael Collins, Trevor Strohman, et al. Controlled decoding from language models. *arXiv:2310.17022*, 2023. Arkadii Nemirovski, David Borisovich Yudin, and Edgar Ronald Dawson. *Problem complexity and method efficiency in optimization*. Wiley, 1983. OpenAI. GPT-4 technical report. *arXiv:2303.08774*, 2023. Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 2022. Alizée Pace, Jonathan Mallinson, Eric Malmi, Sebastian Krause, and Aliaksei Severyn. West-of-n: Synthetic preference generation for improved reward modeling. *arXiv:2401.12086*, 2024. Jing-Cheng Pang, Pengyuan Wang, Kaiyuan Li, Xiong-Hui Chen, Jiacheng Xu, Zongzhang Zhang, and Yang Yu. Language model self-improvement by reinforcement learning contemplation. *arXiv:2305.14483*, 2023. Divyansh Pareek, Simon S Du, and Sewoong Oh. Understanding the gains from repeated selfdistillation. *arXiv:2407.04600*, 2024.

[5] Hieu Pham, Zihang Dai, Qizhe Xie, and Quoc V Le. Meta pseudo labels. In *Conference on Computer Vision and Pattern Recognition*, 2021. Ori Press, Ravid Shwartz-Ziv, Yann LeCun, and Matthias Bethge. The entropy enigma: Success and failure of entropy minimization. *arXiv:2405.05012*, 2024. Yuxiao Qu, Tianjun Zhang, Naman Garg, and Aviral Kumar. Recursive introspection: Teaching language model agents how to self-improve. *arXiv:2407.18219*, 2024. Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems*, 2023. Maxim Raginsky and Alexander Rakhlin. Information-based complexity, feedback and dynamics in convex programming. *IEEE Transactions on Information Theory*, 2011. Mamshad Nayeem Rizve, Kevin Duarte, Yogesh S Rawat, and Mubarak Shah. In defense of pseudolabeling: An uncertainty-aware pseudo-label selection framework for semi-supervised learning. *arXiv:2101.06329*, 2021. Daniel Russo and Benjamin Van Roy. Eluder dimension and the sample complexity of optimistic exploration. In *Advances in Neural Information Processing Systems*, 2013. Abulhair Saparov and He He. Language models are greedy reasoners: A systematic formal analysis of chain-of-thought. In *International Conference on Learning Representations*, 2023. Igal Sason and Sergio Verdú. f-divergence inequalities. *IEEE Transactions on Information Theory*, 2016. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv:1707.06347*, 2017. Pier Giuseppe Sessa, Robert Dadashi, Léonard Hussenot, Johan Ferret, Nino Vieillard, Alexandre Ramé, Bobak Shariari, Sarah Perrin, Abe Friesen, Geoffrey Cideron, et al. Bond: Aligning LLMs with Best-of-N distillation. *arXiv:2407.14622*, 2024. Max Simchowitz, Kevin Jamieson, and Benjamin Recht. The simulator: Understanding adaptive sampling in the moderate-confidence regime. In *Conference on Learning Theory*, 2017. Mohit Singh and Nisheeth K Vishnoi. Entropy, optimization and counting. In *Symposium on Theory of Computing*, 2014. Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling model parameters. *arXiv:2408.03314*, 2024. Yuda Song, Gokul Swamy, Aarti Singh, J Andrew Bagnell, and Wen Sun. Understanding preference fine-tuning through the lens of coverage. *arXiv:2406.01462*, 2024. Kevin Swersky, Yulia Rubanova, David Dohan, and Kevin Murphy. Amortized bayesian optimization over discrete spaces. In *Conference on Uncertainty in Artificial Intelligence*, 2020. Kunal Talwar. Computational separations between sampling and optimization. *Advances in Neural Information Processing Systems*, 32, 2019. Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen

[6] Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. *arXiv:2307.09288*, 2023. Joseph F Traub, Grzegorz W Wasilkowski, and Henryk Wo´zniakowski. *Information-based complexity*. Academic Press Professional, Inc., 1988.

[7] S. A. van de Geer. *Empirical Processes in M-Estimation.* Cambridge University Press, 2000. Ziyu Wan, Xidong Feng, Muning Wen, Stephen Marcus McAleer, Ying Wen, Weinan Zhang, and Jun Wang. Alphazero-like tree-search can guide large language model decoding and training. *International Conference on Machine Learning*, 2024. Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully test-time adaptation by entropy minimization. *arXiv:2006.10726*, 2020. Tianlu Wang, Ilia Kulikov, Olga Golovneva, Ping Yu, Weizhe Yuan, Jane Dwivedi-Yu, Richard Yuanzhe Pang, Maryam Fazel-Zarandi, Jason Weston, and Xian Li. Self-taught evaluators. *arXiv:2408.02666*, 2024. Xuezhi Wang and Denny Zhou. Chain-of-thought reasoning without prompting. *arXiv:2402.10200*, 2024. Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. *arXiv:2212.10560*, 2022. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 2022. Wing Hung Wong and Xiaotong Shen. Probability inequalities for likelihood ratios and convergence rates of sieve mles. *The Annals of Statistics*, 1995. Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, and Sainbayar Sukhbaatar. Meta-rewarding language models: Self-improving alignment with llm-as-a-meta-judge. *arXiv:2407.19594*, 2024a. Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. An empirical analysis of compute-optimal inference for problem-solving with language models. *arXiv:2408.00724*, 2024b. Yue Wu, Zhiqing Sun, Huizhuo Yuan, Kaixuan Ji, Yiming Yang, and Quanquan Gu. Self-play preference optimization for language model alignment. *arXiv:2405.00675*, 2024c. Tengyang Xie and Nan Jiang. Q\* approximation schemes for batch reinforcement learning: A theoretical comparison. In *Conference on Uncertainty in Artificial Intelligence*, 2020. Tengyang Xie, Dylan J Foster, Yu Bai, Nan Jiang, and Sham M Kakade. The role of coverage in online reinforcement learning. In *International Conference on Learning Representations*, 2023. Tengyang Xie, Dylan J Foster, Akshay Krishnamurthy, Corby Rosset, Ahmed Awadallah, and Alexander Rakhlin. Exploratory preference optimization: Harnessing implicit Q\*-approximation for sample-efficient RLHF. *arXiv:2405.21046*, 2024. Wei Xiong, Hanze Dong, Chenlu Ye, Han Zhong, Nan Jiang, and Tong Zhang. Gibbs sampling from human feedback: A provable KL-constrained framework for RLHF. *arXiv:2312.11456*, 2023. Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. *Advances in Neural Information Processing Systems*, 2024. Chenlu Ye, Wei Xiong, Yuheng Zhang, Nan Jiang, and Tong Zhang. A theoretical analysis of Nash learning from human feedback under general KL-regularized preference. *arXiv:2402.07314*, 2024.

[8] Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. Self-rewarding language models. *arXiv:2401.10020*, 2024. Andrea Zanette, Martin J Wainwright, and Emma Brunskill. Provable benefits of actor-critic methods for offline reinforcement learning. *Advances in Neural Information Processing Systems*, 2021. Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. *Advances in Neural Information Processing Systems*, 2022. Tong Zhang. From ϵ-entropy to KL-entropy: Analysis of minimum information complexity density estimation. *The Annals of Statistics*, 2006. Stephen Zhao, Rob Brekelmans, Alireza Makhzani, and Roger Baker Grosse. Probabilistic inference in language models via twisted sequential monte carlo. *International Conference on Machine Learning*, 2024. Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging LLM-as-a-judge with MT-bench and chatbot arena. *Advances in Neural Information Processing Systems*, 2024. Banghua Zhu, Michael Jordan, and Jiantao Jiao. Principled reinforcement learning with human feedback from pairwise or k-wise comparisons. In *International Conference on Machine Learning*, 2023.
# CONTENTS OF APPENDIX

| I  |            | Additional     | Discussion       | and Results                           | 19            |
|----|------------|----------------|------------------|---------------------------------------|---------------|
| A  | Additional |                | Experiments      | and Details                           | 19            |
|    | A.1        | Inference-time |                  | validation experiments                | 19            |
|    | A.2        | Experiments    | with             | other self reward functions           | 23            |
|    | A.3        | Effect of      |                  | SFT-Sharpening                        | 25            |
| B  | Detailed   |                | Discussion       | of Related Work                       | 26            |
| C  |            | Guarantees     | for              | Inference-Time Sharpening             | 29            |
| D  |            | Guarantees     | for              | SFT-Sharpening with Adaptive Sampling | 30            |
| E  |            | Computational  | and              | Representational Challenges in        | Sharpening 31 |
|    | E.1        | Computational  |                  | Challenges                            | 32            |
|    | E.2        |                | Representational | Challenges                            | 32            |
| II | Proofs     |                |                  |                                       | 34            |
| F  |            | Preliminaries  |                  |                                       | 34            |
|    | F.1        | Guarantees     | for              | Approximate Maximizers                | 34            |
|    | F.2        | Technical      | Tools            |                                       | 34            |
| G  | Proofs     | from           | Section          | 3.1                                   | 36            |
| H  | Proofs     | from           | Section          | 3.3                                   | 36            |
| I  | Proofs     | from           | Section          | 4.1 and Appendix D                    | 39            |
| J  | Proofs     | from           | Section          | 4.2                                   | 42            |
|    | J.1        | Proof of       | Theorem          | 4.2                                   | 42            |
|    | J.2        | Proof of       | Theorem          | 4.3 and Theorem J.3                   | 47            |

# Part I

# Additional Discussion and Results

# A ADDITIONAL EXPERIMENTS AND DETAILS

In this section we detail the precise setup required to replicate our empirical results. All of our experiments were run either on 40G NVIDIA A100 GPUs, 192G AMD MI300X GPUs, or through the OpenAI API. We considered the following models. All models, except for gpt-3.5-turbo-instruct, are available on <https://huggingface.co> and we provide HuggingFace model identifiers below.

- 1. Phi models: We experiment with several models from the Phi family of models [\(Abdin et al.,](#page-10-8) [2024\)](#page-10-8), specifically Phi3-Mini ("microsoft/Phi-3-mini-4k-instruct"), Phi3-Small ("microsoft/Phi-3-small-8k-instruct"), Phi3-Medium ("microsoft/Phi-3-medium-4k-instruct"), and Phi3.5-Mini ("microsoft/Phi-3.5-mini-instruct").
- 2. Llama3.2-3B-Instruct ("meta-llama/Llama-3.2-3B-Instruct") [\(Dubey et al.,](#page-11-12) [2024\)](#page-11-12)
- 3. Mistral-7B-Instruct-v0.3 ("mistralai/Mistral-7B-Instruct-v0.3") [\(Jiang et al.,](#page-12-12) [2023\)](#page-12-12)
- 4. gpt-3.5-turbo-instruct [\(Brown et al.,](#page-11-0) [2020\)](#page-11-0): We access this model via the OpenAI API.
- 5. llama2-7b-game24-policy-hf ("OhCherryFire/llama2-7b-game24-policy-hf"): We use the model of [Wan et al.](#page-15-10) [\(2024\)](#page-15-10), which is a Llama-2 model finetuned on the GameOf24 task [\(Yao](#page-15-11) [et al.,](#page-15-11) [2024\)](#page-15-11). We use this model only the GameOf24 task.

We consider the following tasks:

- 1. MATH: We use the above models to generate responses to prompts from the MATH [\(Hendrycks et al.,](#page-12-13) [2021\)](#page-12-13), which consists of more difficult math questions. We consider "all" subsets and take the first 256 examples of the test set where the solution matches the regular expression (\d\*). [8](#page-0-0)
- 2. GSM8k: We use the above models to generate responses to prompts from the GSM-8k dataset [\(Cobbe et al.,](#page-11-13) [2021\)](#page-11-13) where the goal is to generate a correct answer to an elementary school math question. We take the first 256 examples from the test set in the main subset.[<sup>9</sup>](#page-0-0)
- 3. ProntoQA: We use the above models to generate responses to prompts from the ProntoQA dataset [\(Saparov & He,](#page-14-9) [2023\)](#page-14-9), which consists of chain-of-thought-style reasoning questions with boolean answers. We take the first 256 examples from the training set.[<sup>10</sup>](#page-0-0)
- 4. MMLU: We use the above models to generate responses to prompts from three subsets of the MMLU dataset [\(Hendrycks et al.,](#page-12-14) [2020\)](#page-12-14), specifically college\_biology (Bio),college\_physics (Phys), and college\_chemistry (Chem) all of which consist of multiple choice questions[<sup>11</sup>](#page-0-0). We take the first 256 examples of the test set.
- 5. GameOf24: We use only the model of [Wan et al.](#page-15-10) [\(2024\)](#page-15-10) (i.e., llama2-7b-game24-policy-hf), on the GameOf24 task [\(Yao et al.,](#page-15-11) [2024\)](#page-15-11). The prompts are four numbers and the goal is to combine the numbers with standard arithmetic operations to reach the number '24.' Here we use both the train and test splits of the dataset.[<sup>12</sup>](#page-0-0)

# A.1 INFERENCE-TIME VALIDATION EXPERIMENTS

To form the plots in [Figure 1](#page-1-0) and in [Figures 3](#page-20-0) and [4,](#page-21-0) for each (model, task) pair, we sampled N generations per prompt with temperature 1 and returned the best of the N generations according to the maximum-likelihood sharpening self-reward function rself(y | x) = log πbase(y | x); we compare against greedy decoding as a baseline, whose accuracy is displayed in [Figure 2\(d\).](#page-19-0)

<sup>8</sup> <https://huggingface.co/datasets/lighteval/MATH>.

<sup>9</sup> <https://huggingface.co/datasets/openai/gsm8k>.

<sup>10</sup><https://huggingface.co/datasets/longface/prontoqa-train>.

<sup>11</sup><https://huggingface.co/datasets/cais/mmlu>.

<sup>12</sup><https://github.com/princeton-nlp/tree-of-thought-llm/tree/master/src/tot/data/24>

![](_page_19_Figure_9.jpeg)

Figure 2: Performance of alternative decoding schemes beyond BoN. Percent accuracy improvement over greedy decoding for self-improvement with length-normalized log probability (a) and majority voting (b), with both demonstrating efficacy on a range of model-task pairs. (c) Measure of coverage of correct answer, demonstrating that most model-task pairs produce the correct answer most of the time with at least one completion out of 50. (d) Accuracy of greedy decoding baseline on each model-task pair.

**Implementation details.** For all models and datasets except for GameOf24, we used 1-shot prompting to ensure that models conform to the desired output format and to elicit chain of thought reasoning (for GameOf24 we do not provide a demonstration in the prompt). We set the maximum length of decoding to be 512 tokens. We used 10 seeds for all (model, task) pairs with a maximum value of  $N = 50$  in Best-of- $N$  sampling. We simulated  $N$  responses for  $N < 50$  by subsampling the 50 generated samples. For Best-of- $N$  sampling, we always use temperature 1.0. Since greedy decoding is a deterministic strategy, we only use 1 seed for each (model, task) pair. In all experiments, we collect both the responses and their log-likelihoods under the *reference model* (i.e., the original model from which samples were generated).

**Results.** Results for most datasets are presented in Figures 3 and 4. Because we only consider a single model for GameOf24, we separate this task into Figure 5. For all datasets, we visualize both performance—measured as normalized improvement in accuracy over greedy decoding—and log-likelihoods—under  $\pi_{\text{base}}$ —of the selected responses.

In all cases, Best-of- $N$  sampling (using  $r_{\text{self}}(y | x) = \log \pi_{\text{base}}(y | x)$ ) improves over the naïve sampling strategy, wherein we simply sample a single generation with temperature 1.0. In all datasets, we also see improvements over the standard *greedy decoding* strategy, at least for some models. Analogously, for every model, there is at least one dataset for which Best-of- $N$  sampling improves over greedy decoding.

We further explore the relationship between sequence level log probabilities and generation quality in Figure 6, where we plot the empirical distributions of responses sampled with temperature 1 from

![](_page_20_Figure_1.jpeg)

Figure 3: Percent lift in accuracy of inference-time BoN-sharpening over greedy decoding in each task as N is varied. For many task-model pairs, the accuracy improves as N increases, demonstrating the efficacy of maximum likelihood sharpening.

![](_page_21_Figure_1.jpeg)

Figure 4: Effect of N on average sequence level log-probabilities for inference-time BoN-sharpening on various model-task pairs, compared to greedy decoding baseline. As predicted by theory, the likelihood of sequences sampled with BoN-sharpening increases with N.

the base model for a variety of model-dataset pairs, conditioned on whether or not the response is correct. It is clear from the figures that the distribution of log probabilities conditioned on correctness stochastically dominates that conditioned on incorrectness in each case, which provides yet more evidence that log likelihoods represent a reasonable sel-improvement target.

We mention several other observations from the experiments. First, in most cases, performance and log-likelihood saturate at relatively small values of N, typically around 10 or 20. This suggests that significant improvements can be obtained with relatively low computational overhead. Second, in some cases, performance can degrade as N increases. We found that this happens for two reasons: (1) the performance of the reference model is quite low and so rself provides a poor signal (e.g., with Llama3.2-3B-Instruct) and (2) the Best-of-N criteria selects for short responses, which have higher log-likelihood but cannot leverage the computational/representational benefits of chain-ofthought, and thus yield worse performance (e.g., with gpt-3.5-turbo-instruct on GSM8k).

![](_page_22_Figure_3.jpeg)

Figure 5: Effect of inference-time BoN-sharpening on GameOf24 with the finetuned llama2-7b-game24-policy-hf from [Wan et al.](#page-15-10) [\(2024\)](#page-15-10).

#### A.2 EXPERIMENTS WITH OTHER SELF REWARD FUNCTIONS

Although we focus on rself(y | x) = log πbase(y | x) throughout the paper, the sharpening framework is significantly more general. As such, we also ran experiments with other choices for rself, specifically:

- 1. Length-normalized log-likelihood: rself(y | x) = log πbase(y | x)/|y| where |y| is the length, in tokens, of the response.
- 2. Majority (self-consistency): All datasets except GameOf24 have multiple-choice, boolean, or numerical answers. Although we allow responses to contain chain-of-thought tokens, we can extract the answer from each response and use the most-frequently-occuring answer. This can be seen as a sample-based approximation to the following self-reward function: P rself(y | x) = y′ :y′ ans=yans πbase(y ′ | x), where yans are the "answer" tokens in the full response y.

Finally, as a skyline we consider the *coverage* criterion [\(Brown et al.,](#page-11-8) [2024\)](#page-11-8), where we simply check if any of the sampled responses corresponds to the correct answer. This criterion is a skyline and does not fit into the self-improvement framework due to the fact that it uses knowledge of the ground truth (external) task reward function.

Results are displayed in [Figure 2.](#page-19-1) For length-normalized log-likelihood and majority, we see qualitatively similar behavior to (unnormalized) log-likelihood in the sense that inference-time sharpening via these self-reward functions offers improvements over both vanilla (temperature 1.0) sampling and greedy decoding. In both cases, the improvements are generally much larger than those obtained with log-likelihood. Finally, examining the coverage criteria, we see that with N = 50 samples, these models almost always produce a correct answer on these tasks, raising the possibility of other self-reward functions that further improve performance.

![](_page_23_Figure_1.jpeg)

Figure 6: Distribution of sequence-level log-probabilities for responses sampled with temperature 1, conditioned on whether or not the response is correct. We consider four model-dataset pairs: (a) (Phi3.5-Mini, MATH); (b) (Phi3.5-Mini, GSM8k); (c) (Phi3.5-Mini, ProntoQA); (d) (Mistral-7B-Instruct-v0.3, MATH). In all cases except perhaps (c), conditioning on correctness of the response leads to a noticeable increase in log-probabilities, further justifying the use of sequencelevel log-probabilities as a self-reward for self-improvement.

| Model       | Dataset  | % Lift over |    | Greedy | (Accuracy) | Lift over |    | Greedy | (Likelihood) |
|-------------|----------|-------------|----|--------|------------|-----------|----|--------|--------------|
| Phi3.5-Mini | MATH     | 19          | 24 | ± 2    | 41         | 48        | 33 | ± 0    | 17           |
| Phi3.5-Mini | GSM8k    | 1           | 82 | ± 0    | 64         | 1         | 49 | ± 0    | 55           |
| Phi3.5-Mini | ProntoQA | 12          | 46 | ± 1    | 08         | 5         | 64 | ± 0    | 01           |
| Mistral-7B  | MATH     | 8           | 88 | ± 5    | 55         | 5         | 71 | ± 3    | 00           |

Table 1: Experimental results for SFT-Sharpening

| Model                    | Dataset  | Weight Decay | LoRA Rank |
|--------------------------|----------|--------------|-----------|
| Phi3.5-Mini              | MATH     | 0.1          | 16        |
| Phi3.5-Mini              | GSM8k    | 0.5          | 16        |
| Phi3.5-Mini              | ProntoQA | 0.0          | 16        |
| Mistral-7B-Instruct-v0.3 | MATH     | 1.0          | 8         |

Table 2: Hyperparameters for SFT-Sharpening

#### A.3 EFFECT OF SFT-Sharpening

In addition to inference-time experiments demonstrating the validity of the amortization objective considered in our theory, we also demonstrate empirically that amortization can be effected with SFT-Sharpening. Due to the realities of limited computational resources, we choose a strict subset of the model-task pairs considered in [Appendix A.1](#page-18-2) that have particularly promising inference-time BoN performance and apply SFT-Sharpening to amortize the inference time cost of multiple generations.

For each of the chosen model-dataset pairs (cf. [Table 1\)](#page-24-0), we sample N = 50 responses with temperature 1 for each prompt in the dataset and select the most likely (according to the relevant reference model). We then combine these likely responses with the prompts in order to form a training corpus and train a Low Rank Adaptation [\(Hu et al.,](#page-12-15) [2021\)](#page-12-15) to the model, sweeping over LoRA rank, learning rate scheduler, and weight decay in order to return the best optimized model.[<sup>13</sup>](#page-0-0) We report the specific hyperparamters chosen in [Table 2.](#page-24-2) On all models, we used a learning rate of 3 × 10−<sup>4</sup> with linear decay to zero and gradient clamping at 0.1.

Results. In [Table 1](#page-24-0) we report our results for the best model during training of each model-dataset pair, averaged across 3 random seeds, where responses are sampled with temperature 1 from the fine-tuned model. We report both the percent lift in accuracy on the dataset with respect to the greedy generation of the reference model and the increase in average sequence level log likelihood with respect to the same. In all cases, we see improvement on both metrics, demonstrating that some amortization is possible with SFT-Sharpening. In [Figures 7](#page-25-1) and [8,](#page-26-0) we display the evolution throughout training of these same metrics for each of the model-dataset pairs. While Phi3.5-Mini is quite well-behaved on MATH and ProntoQA, there appears to be a fair amount of noise in the training on GSM8k, with the log probability being a significantly less useful proxy for accuracy on this dataset than the others, as has been previously found in [Block et al.](#page-10-9) [\(2023\)](#page-10-9). In the case of Mistral-7B-Instruct-v0.3 on MATH, while we do see some improvement after sufficient training, the optimization suffers an initial substantial drop and then spends ∼ 90% of the gradient steps recovering; we speculate that this is a function of insufficient hyper-parameter tuning of the optimization itself, rather than a fundamental barrier.

Finally, in [Figure 9,](#page-26-1) we investigate the effect that the choice of N has on SFT-Sharpening for Phi3.5-Mini on MATH. In particular, in forming our training set, we choose N ∈ {10, 25, 50} and repeat the procedure described above, averaging our results over three seeds. We find that increasing N leads to a modest increase in the sequence-level log-likelihood and a consequent increment in the accuracy of the fine-tuned model, in accordance with our theory.

<sup>13</sup>In all experiments involving Phi3.5-Mini we use a batch size of 4; unfortunately, due to a known numerical issue with LoRA on Mistral-7B-Instruct-v0.3 involving batch size > 1, we use a batch of 1 in this case. Because of this choice, instead of the 30 epochs we use to train our other models, for Mistral-7B-Instruct-v0.3, we run only 10 epochs.

![](_page_25_Figure_1.jpeg)

Figure 7: Evolution of Phi3.5-Mini under SFT-Sharpening (N = 50) on different datasets, as measured by (i) % lift over Greedy in accuracy; and (ii) difference in average sequence-level logprobability of generated responses under the reference model. The fine-tuned model learns to produce generations with high probability under the reference model, and consequently enjoys an increase in accuracy compared to the base model. However, the model does not fully reach the performance of inference-time BoN sharpening.

# B DETAILED DISCUSSION OF RELATED WORK

In this section, we discuss related work in greater detail, including relevant works not already covered.

![](_page_26_Figure_1.jpeg)

Figure 8: Evolution of Mistral-7B-Instruct-v0.3 under SFT-Sharpening (N = 50) on MATH, as measured by (i) % lift over Greedy in accuracy; and (ii) difference in average sequence-level log-probability of generated responses under the reference model.

![](_page_26_Figure_3.jpeg)

Figure 9: Effect of N on SFT-Sharpening for Phi3.5-Mini on MATH. We report (a) % lift in accuracy over greedy; and (b) lift in sequence-level log-likelihood (averaged over the dataset). In both cases, we see that increasing N leads to greater lift, in accordance with theory.

Self-improvement and self-training. Our work is most directly related to a growing body of empirical research that studies self-improvement/self-training for language models in a supervisionfree setting in which there is no external feedback [\(Huang et al.,](#page-12-1) [2022;](#page-12-1) [Wang et al.,](#page-15-0) [2022;](#page-15-0) [Bai et al.,](#page-10-0) [2022b;](#page-10-0) [Pang et al.,](#page-13-2) [2023\)](#page-13-2), and takes a first step toward providing a theoretical understanding for these methods. There is also a closely related body of research on "LLM-as-a-Judge" techniques, which investigates approaches to designing self-reward functions rself, often based on specific prompting techniques [\(Zheng et al.,](#page-16-1) [2024;](#page-16-1) [Yuan et al.,](#page-16-0) [2024;](#page-16-0) [Wu et al.,](#page-15-1) [2024a;](#page-15-1) [Wang et al.,](#page-15-2) [2024\)](#page-15-2).

A somewhat complementary line of research develops algorithms based on self-training and self-play [\(Zelikman et al.,](#page-16-4) [2022;](#page-16-4) [Chen et al.,](#page-11-14) [2024;](#page-11-14) [Wu et al.,](#page-15-12) [2024c;](#page-15-12) [Qu et al.,](#page-14-10) [2024\)](#page-14-10), but leverages various forms of external feedback (e.g., positive examples for SFT or explicit reward signal). These methods typically outperform feedback-free self-improvement methods [\(Zelikman et al.,](#page-16-4) [2022\)](#page-16-4). However, in many scenarios, obtaining external feedback can be costly or laborious; it may require collecting high-quality labeled/annotated data, rewriting examples in a formal language, etc. Thus, these two approaches are not directly comparable.

We also mention that the self-improvement problem we study is related to a classical line of research on *self-distillation* [\(Bucilua et al.](#page-11-15) ˇ , [2006;](#page-11-15) [Hinton et al.,](#page-12-2) [2015;](#page-12-2) [Devlin,](#page-11-16) [2018;](#page-11-16) [Pham et al.,](#page-14-11) [2021;](#page-14-11) [Rizve](#page-14-12) [et al.,](#page-14-12) [2021\)](#page-14-12), but this specific form of self-training has received limited investigation in the context of language modeling.

Entropy minimization. Sharpening is also closely related to a line of work on *entropy minimization* or *minimum entropy regularization*, where we seek models that have high predictive accuracy and low entropy/uncertainty. This line of work originated in the semi-supervised learning literature [\(Grand](#page-12-16)[valet & Bengio,](#page-12-16) [2004\)](#page-12-16) and was popularized as a test-time adaptation method in computer vision (c.f., [Wang et al.,](#page-15-13) [2020;](#page-15-13) [Press et al.,](#page-14-13) [2024\)](#page-14-13). Maximum-likelihood sharpening, especially via RL, is closely related in that [Eq. \(3\)](#page-4-1) with β → 0 and rself = log πbase maximizes <sup>E</sup>π[log πbase(y | x)] rather than −H(π) = <sup>E</sup>π[log π(y | x)]. (It is important that the latter is optimized continuously with πbase as an initialization, but when this is done it can be seen to sharpen πbase, at least heuristically.) Prior work in this direction is largely empirical, focused on computer vision domains with small output spaces Y, and hence studies statistical benefits of entropy minimization. In contrast, we initiate a theoretical study of sharpening, are primarily motivated by applications to language modeling with exponentially large output spaces, and view sharpening primarily as a computational phenomena. However, it would be interesting to understand whether statistical benefits observed in computer vision translate to the language modeling setting.

Alignment and RLHF. The specific algorithms for self-improvement/sharpening we study can be viewed as special cases of standard alignment algorithms, including classical RLHF methods [\(Christiano et al.,](#page-11-4) [2017;](#page-11-4) [Bai et al.,](#page-10-4) [2022a;](#page-10-4) [Ouyang et al.,](#page-13-0) [2022\)](#page-13-0), direct alignment [\(Rafailov et al.,](#page-14-3) [2023\)](#page-14-3), and (inference-time or training-time) best-of-N methods [\(Amini et al.,](#page-10-3) [2024;](#page-10-3) [Sessa et al.,](#page-14-1) [2024;](#page-14-1) [Gui et al.,](#page-12-7) [2024;](#page-12-7) [Pace et al.,](#page-13-8) [2024\)](#page-13-8). However, the maximum likelihood sharpening objective [\(2\)](#page-2-1) used for our theoretical results has been relatively unexplored within the alignment literature.

Inference-time decoding. Many inference-time decoding strategies such as greedy/low-temperature decoding, beam-search [\(Meister et al.,](#page-13-5) [2020\)](#page-13-5), and chain-of-thought decoding [\(Wang & Zhou,](#page-15-3) [2024\)](#page-15-3) can be viewed as instances of inference-time sharpening for specific choices of the self-reward function rself. More sophisticated inference-time search strategies such tree search and MCTS [\(Yao](#page-15-11) [et al.,](#page-15-11) [2024;](#page-15-11) [Wan et al.,](#page-15-10) [2024;](#page-15-10) [Mudgal et al.,](#page-13-19) [2023;](#page-13-19) [Zhao et al.,](#page-16-5) [2024\)](#page-16-5) are also related, though this line of work frequently makes use of external reward signals or verification, which is somewhat complementary to our work.

Theoretical guarantees for self-training. On the theoretical side, current understanding of selftraining is limited. One line of work, focusing on the *self-distillation* objective [\(Hinton et al.,](#page-12-2) [2015\)](#page-12-2) for binary classification and regression, aims to provide convergence guarantees for self-training in stylized setups such as linear models [\(Mobahi et al.,](#page-13-6) [2020;](#page-13-6) [Das & Sanghavi,](#page-11-5) [2023;](#page-11-5) [Das et al.,](#page-11-6) [2024;](#page-11-6) [Pareek et al.,](#page-13-7) [2024\)](#page-13-7), with [Allen-Zhu & Li](#page-10-2) [\(2020\)](#page-10-2) giving guarantees for feedforward neural networks. Perhaps most closely related to our work is [Frei et al.](#page-12-5) [\(2022\)](#page-12-5), who show that self-training on a model's pseudo-labels can amplify the margin for linear logistic regression. However, to the best of our knowledge, our work is the first to study self-training in a general framework that subsumes language modeling.

Our results for RLHF-Sharpening are related to a body of work that provides sample complexity guarantees for alignment methods [\(Zhu et al.,](#page-16-3) [2023;](#page-16-3) [Xiong et al.,](#page-15-14) [2023;](#page-15-14) [Ye et al.,](#page-15-15) [2024;](#page-15-15) [Huang et al.,](#page-12-10) [2024;](#page-12-10) [Liu et al.,](#page-13-16) [2024;](#page-13-16) [Song et al.,](#page-14-14) [2024;](#page-14-14) [Xie et al.,](#page-15-5) [2024\)](#page-15-5), but our results leverage the structure of the maximum-likelihood sharpening self-reward function rself(y | x) = log πbase(y | x), and provide guarantees for the sharpening objective in [Definition 3.1](#page-5-1) instead of the usual notion of reward suboptimality used in reinforcement learning theory.

Lastly, we mention that our results—particularly our *amortization* perspective on self-improvement are related to work that studies representational advantages afforded by additional inference time [\(Malach,](#page-13-15) [2023;](#page-13-15) [Li et al.,](#page-13-14) [2024\)](#page-13-14). These work focus on truly sequential tasks, while our work focuses on the complementary question of amortizing *parallel* computation. Thus the representational implications are quite different.

Optimization versus sampling. The maximum-likelihood sharpening objective we introduce in [Section 3](#page-4-3) connects the study of *self-improvement* to a large body of research in theoretical computer science on computational tradeoffs (e.g., separations and equivalences) between optimization and sampling [\(Barahona,](#page-10-10) [1982;](#page-10-10) [Kirkpatrick et al.,](#page-13-9) [1983;](#page-13-9) [Lovász & Vempala,](#page-13-10) [2006;](#page-13-10) [Singh & Vishnoi,](#page-14-5) [2014;](#page-14-5) [Ma et al.,](#page-13-11) [2019;](#page-13-11) [Talwar,](#page-14-6) [2019;](#page-14-6) [Eldan et al.,](#page-11-17) [2022\)](#page-11-17). On the one hand, this line of research highlights that there exist natural classes of distributions for which sampling is tractable, yet maximum likelihood optimization is intractable, and vice-versa. On the other hand, various works in this line of research also demonstrate *computational reductions* between optimization and sampling, whereby optimization can be reduced to sampling and vice-versa.

Our setting indeed includes natural model classes where one should not expect there to be a computational reduction from optimization (arg maxy∈Y πbase(y | x)) to sampling (y ∼ πbase(· | x)), and hence inference-time sharpening is computationally intractable [\(Proposition E.1\)](#page-31-2). Of course, coverage assumptions eliminate this intractability. For training-time sharpening (where the goal is to *amortize* across prompts by training a sharpened model, as formulated in [Section 3\)](#page-4-3) the obstacle in natural, concrete model classes is not just computational but in fact *representational* [\(Proposition E.2\)](#page-31-3). Regarding the latter point, we note that while amortized Bayesian inference has received extensive investigation empirically [\(Beal,](#page-10-11) [2003;](#page-10-11) [Gershman & Goodman,](#page-12-17) [2014;](#page-12-17) [Swersky et al.,](#page-14-15) [2020;](#page-14-15) [Bengio](#page-10-12) [et al.,](#page-10-12) [2021;](#page-10-12) [Hu et al.,](#page-12-18) [2023\)](#page-12-18), we are unaware of theoretical guarantees outside of this work.

# C GUARANTEES FOR INFERENCE-TIME SHARPENING

In this section, we give theoretical guarantees for the inference-time best-of-N sampling algorithm for sharpening described in [Section 3.1,](#page-4-0) under the maximum-likelihood sharpening self-reward function

$$r_{\text{self}}(y \mid x; \pi_{\text{base}}) = \log \pi_{\text{base}}(y \mid x).$$

Recall that given a prompt x ∈ X , the inference-time best-of-N sampling algorithm draws N responses <sup>y</sup>1, . . . , y<sup>n</sup> <sup>∼</sup> <sup>π</sup>base(· | <sup>x</sup>), then return the response <sup>y</sup>b = arg max<sup>y</sup><sup>i</sup> log πbase(y<sup>i</sup> | x). We show that this algorithm returns an approximate maximizer for the maximum-likelihood sharpening objective whenever the base policy πbase has sufficient coverage. For a parameter γ ∈ [0, 1) we define

$$\mathbf{y}_\gamma^*(x) := \left\{ y \mid \pi_{\text{base}}(y \mid x) \geq (1 - \gamma) \cdot \max_{y \in \mathcal{Y}} \pi_{\text{base}}(y \mid x) \right\}$$

as the set of (1 − γ)-approximate maximizers for log πbase(y | x) (see [Appendix F.1](#page-33-3) for background on y ⋆ γ (x)).

Proposition C.1. *Let a prompt* x ∈ X *be given. For any* ρ ∈ (0, 1) *and* γ ∈ [0, 1)*, as long as*

$$N \geq \frac{\log(\rho^{-1})}{\pi_{\text{base}}(\mathbf{y}_\gamma^*(x) \mid x)},$$

*inference-time best-of-*<sup>N</sup> *sampling produces a response* <sup>y</sup>b <sup>∈</sup> <sup>y</sup> ⋆ γ (x) *with probability at least* 1 − ρ*.*

Proof of [Proposition C.1.](#page-28-1) Fix a prompt x ∈ X , failure probability ρ ∈ (0, 1), and parameter γ ∈ (0, 1). By definition of the set y ⋆ γ (x), <sup>y</sup>b <sup>∈</sup> <sup>y</sup> ⋆ γ (x) if and only if there exists i ∈ [N] such that y<sup>i</sup> ∈ y ⋆ γ (x). The complement of this event, i.e., that y<sup>i</sup> ∈/ y ⋆ γ (x) for all i ∈ [N], has probability

$$\mathbb{P}(y_i \notin \mathbf{y}_\gamma^\star(x), \forall i \in [N]) = (1 - \pi_{\text{base}}(\mathbf{y}_\gamma^\star(x) \mid x))^N.$$

Rearranging the right-hand side, we have

$$(1 - \pi_{\text{base}}(\mathbf{y}_\gamma^* \mid x))^N = \exp\left(-N \log\left(\frac{1}{1 - \pi_{\text{base}}(\mathbf{y}_\gamma^* \mid x)}\right)\right) \leq \exp(-N \cdot \pi_{\text{base}}(\mathbf{y}_\gamma^* \mid x)),$$

since log(x) ≥ 1 − 1 x for x > <sup>0</sup>, which implies that log 1 1−πbase(y<sup>⋆</sup> |x) ≥ πbase(y ⋆ γ | x). Thus, as long as N ≥ log(ρ −1 ) πbase(y<sup>⋆</sup> |x) , we have

$$\mathbb{P}(y_i \notin \mathbf{y}_\gamma^\star(x), \forall i \in [N]) \leq \exp(-N \cdot \pi_{\text{base}}(\mathbf{y}_\gamma^\star \mid x)) \leq \exp(-\log(\rho^{-1})) = \rho.$$

We conclude that with probability at least 1 − ρ, there exists i ∈ [N] such that y<sup>i</sup> ∈ y ⋆ γ (x), and <sup>y</sup>b <sup>∈</sup> <sup>y</sup> ⋆ γ (x) as a result.

# D GUARANTEES FOR SFT-Sharpening WITH ADAPTIVE SAMPLING

SFT-Sharpening is a simple and natural self-training scheme, and converges to a sharpened policy as n, N → ∞. However, using a fixed response sample size N may be wasteful for prompts where the model is confident. To this end, in this section we introduce and analyze, a variant of SFT-Sharpening based on *adaptive sampling*, which adjusts the number of sampled responses adaptively.

Algorithm. We present the adaptive SFT-Sharpening algorithm only for the special case of the maximum likelihood sharpening self-reward. Let a *stopping parameter* µ > 0 be given. For x<sup>i</sup> ∈ X , and yi,1, yi,<sup>2</sup> . . . ∼ πbase(· | xi), define a stopping time (e.g., [Benjamini & Hochberg](#page-10-13) [\(1995\)](#page-10-13)) via:

$$N_\mu(x_i) := \inf \left\{ k : \frac{1}{\max_{1 \leq j \leq k} \pi_{\text{base}}(y_{i,j} \mid x_i)} \leq \frac{k}{\mu} \right\}. \quad (10)$$

The adaptive SFT-Sharpening algorithm computes adaptively sampled responses y AdaBoN i via

$$y_i^{\text{AdaBoN}} \sim \arg \max \{ \log \pi_{\text{base}}(y_{i,j} \mid x_i) \mid y_{i,1}, \dots, y_{i,N_\mu(x_i)} \},$$

then trains the sharpened model through SFT:

$$\hat{\pi}^{\text{AdaBoN}} = \arg \max_{\pi \in \Pi} \sum_{i=1}^n \log \pi(y_i^{\text{AdaBoN}} \mid x_i).$$

Critically, by using scheme in [Eq. \(10\),](#page-29-1) this algorithm can stop sampling responses for the prompt x<sup>i</sup> if it becomes clear that the confidence is large.

Theoretical guarantee. We now show that adaptive SFT-Sharpening enjoys provable benefits over its non-adaptive counterpart through the dependence on the accuracy parameter ϵ > 0.

Given x ∈ X , and y1, y<sup>2</sup> . . . ∼ πbase(x), let Nµ(x) := inf{k : 1 max1≤i≤<sup>k</sup> <sup>π</sup>base(yi|x) ≤ k/µ}, and define a random variable y AdaBoN(x) ∼ arg max log πbase(y<sup>i</sup> | x) | y1, . . . , y<sup>N</sup><sup>µ</sup> ∼ πbase(x) . Let π AdaBoN µ (x) denote the distribution over y AdaBoN(x). We make the following realizability assumption.

Assumption D.1. *The model class* Π *satisfies* π AdaBoN <sup>µ</sup> ∈ Π*.*

Compared to SFT-Sharpening, we require a somewhat stronger coverage coefficient given by

$$\bar{C}_{\text{cov}} = \mathbb{E}_{x \sim \mu} \left[ \frac{1}{\max_{y \in \mathcal{Y}} \pi_{\text{base}}(y \mid x)} \right].$$

Theorem D.1. *Let* δ, ρ ∈ (0, 1) *be given. Set* µ = ln(2δ −1 )*, and assume [Assumption D.1](#page-29-2) holds. Then with probability at least* 1 − ρ*, the adaptive* SFT-Sharpening *algorithm has*

$$\mathbb{P}_{x \sim \mu}[\widehat{\pi}(\mathbf{y}^*(x) \mid x) \leq 1 - \delta] \lesssim \frac{\log(|\Pi|\rho^{-1})}{\delta n},$$

*and has sample complexity* <sup>E</sup>[m] = n · Ccov log(δ −1 )*. Taking* n ≳ log(|Π|ρ −1 ) δϵ *ensures that with probability at least* <sup>1</sup> <sup>−</sup> <sup>ρ</sup>*,* <sup>P</sup>x∼µ[πb(<sup>y</sup> ⋆ (x) | x) ≤ 1 − δ] ≤ ϵ*, and gives total sample complexity*

$$\mathbb{E}[m] = O\left(\frac{\bar{C}_{\text{cov}} \log(|\Pi|\rho^{-1}) \log(\delta^{-1})}{\delta\epsilon}\right).$$

Compared to the result for SFT-Sharpening in [Theorem 4.1,](#page-7-5) this shows that adaptive SFT-Sharpening achieves sample complexity scaling with <sup>1</sup> ϵ instead of <sup>1</sup> ϵ <sup>2</sup> . We believe the dependence on Ccov for this algorithm is tight, as the adaptive stopping rule used in the algorithm can be overly conservative when |y ⋆ (x)| is large.

A matching lower bound. We now prove a complementary lower bound, which shows that the ϵ-dependence in [Theorem D.1](#page-29-3) is tight. To do so, we consider the following adaptive variant of the sample-and-evaluate framework.

Definition D.1 (Adaptive sample-and-evaluate framework). *In the Adaptive Sample-and-Evaluate framework, the learner is allowed to sample* n *prompts* x ∼ µ*, and sample an arbitrary, adaptively chosen number of samples* y1, y2, · · · ∼ πbase(· | x) *before sampling a new prompt* x ′ ∼ µ*. In this framework we define sample complexity* m *as the total number of pairs* (x, y) *sampled by the algorithm, which is a random variable.*

Our main lower bound is as follows.

Theorem D.2 (Lower bound for sharpening under adaptive sampling). *Fix an integer* d ≥ 1 *and parameters* ϵ ∈ (0, 1) *and* C ≥ 1*. There exists a class of models* Π *such that (i)* log |Π| <sup>≂</sup> d(1 + log(Cϵ−<sup>1</sup> ))*, (ii)* supπ∈<sup>Π</sup> Ccov(π) ≲ C*, and (iii)* y π (x) *is a singleton for all* π ∈ Π*, for which any sharpening algorithm* <sup>π</sup>b *in the adaptive sample-and-evaluate framework that achieves* <sup>E</sup>[<sup>P</sup>x∼µ[πb(<sup>y</sup> <sup>π</sup>base (x) | x) > 1/2]] ≥ 1 − ϵ *for all* πbase ∈ Π *must collect a total number of samples* m = n · N *at least*

$$\mathbb{E}[m] \gtrsim \frac{C \log |\Pi|}{\epsilon \cdot (1 + \log(C\epsilon^{-1}))}.$$

[Theorem D.2](#page-30-1) is a special case of a more general theorem, [Theorem 3.1](#page-35-2)′ , which is stated and proven in [Appendix H.](#page-35-1)

# E COMPUTATIONAL AND REPRESENTATIONAL CHALLENGES IN SHARPENING

In this section, we make several basic observations about the inherent computational and representational challenges of maximum-likelihood sharpening. First, in [Appendix E.1,](#page-31-0) we focus on computational challenges, and show that computing a sharpened response for a given prompt x can be computationally intractable in general, even when sampling y ∼ πbase(· | x) can be performed efficiently. Then, in [Appendix E.2,](#page-31-1) we shift our focus to representational challenges, and show that even if πbase is an autoregressive model, the "sharpened" version of πbase may not be representable as an autoregressive model with the same architecture. These results motivate the statistical assumptions (coverage and realizability) made in our analysis of SFT-Sharpening and RLHF-Sharpening in [Section 4.](#page-6-1)

To make the results in this section precise, we work in perhaps the simplest special case of autoregressive language modelling, where the model class consists of *multi-layer linear softmax models*. Formally, let X be the space of prompts, and let Y := V <sup>H</sup> be the space of responses, where V is the vocabulary space and H is the horizon. For a collection of fixed/known d-dimensional feature mappings ϕ<sup>h</sup> : X × V<sup>h</sup> → <sup>R</sup> d and a norm parameter B, we define the model class Πϕ,B,H as the set of models

$$\pi_\theta(y_{1:H} \mid x) = \prod_{h=1}^H \pi_{\theta_h}(y_h \mid x, y_{1:h-1}) \quad (11)$$

where

$$\pi_\theta(y_h \mid x, y_{1:h-1}) \propto \exp(\langle \phi(x, y_{1:h}), \theta_h \rangle)$$

and θ = (θ1, . . . , θH) ∈ (<sup>R</sup> d ) <sup>H</sup> is any tuple with ∥θh∥<sup>2</sup> ≤ B for all h ∈ [H].

#### E.1 COMPUTATIONAL CHALLENGES

Given query access to ϕ, for any given parameter vector θ and prompt x, *sampling* from a linear softmax model π<sup>θ</sup> [\(Eq. \(11\)\)](#page-30-2) is computationally tractable, since it only requires time poly(H, |V|, d). Similarly, *evaluating* πθ(y1:<sup>H</sup> | x) for given prompt x and response y1:<sup>H</sup> is computationally tractable. However, the following proposition shows that computing the sharpened response arg max<sup>y</sup>1:H∈V<sup>H</sup> πθ(y1:<sup>H</sup> | x) for a given parameter θ and response x is NP-hard. Hence, even inference-time sharpening is computationally intractable in the worst case.

Proposition E.1. *Set* X = {⊥} *and* V = {−1, 1}*. Set* d = d(H) := H + H<sup>2</sup> + H<sup>3</sup> *. Identifying* [d] *with* [H] ⊔ [H] <sup>2</sup> ⊔ [H] 3 *, we define* ϕ<sup>h</sup> : X × V<sup>h</sup> → <sup>R</sup> <sup>d</sup> *by* ϕh(⊥, y1:h)<sup>i</sup> = y<sup>i</sup> *and* ϕh(⊥, y1:h)(i,j) = yiy<sup>j</sup> *and* ϕh(⊥, y1:h)(i,j,k) = yiyjyk*. There is a function* B(H) ≤ poly(H) *such that the following problem is* NP*-hard: given* θ = (θ1, . . . , θH) *with* maxh∈[H]∥θh∥<sup>2</sup> ≤ B(H)*, compute any element of* arg max<sup>y</sup>1:H∈V<sup>H</sup> πθ(y1:<sup>H</sup> | x)*.*

Note that our results in [Section 4](#page-6-1) and [Appendix C](#page-28-0) bypass this hardness through the assumption that the coverage parameter Ccov is bounded.

Proof of [Proposition E.1.](#page-31-2) Fix H and recall that d(H) = H + H<sup>2</sup> + H<sup>3</sup> . We define three collection of basis vectors: {eh}h∈[H] cover the first H coordinates, e(h,h′) h,h′∈[H] <sup>2</sup> cover the next H<sup>2</sup> coordinates, and e(h,h′ ,h′′) h,h′ ,h′′∈[H] <sup>3</sup> cover the last H<sup>3</sup> coordinates. Suppose we define θ1, . . . , θH−<sup>2</sup> = 0, so that πθ(yh|x, y1:h−1) = 1/2 for all 1 ≤ h ≤ H − 2. Define θH−<sup>1</sup> = P 1≤i,j≤H−2 Jij e(i,j,H−1) for a matrix J ∈ <sup>R</sup> (H−2)×(H−2) to be specified later, and define θ<sup>H</sup> = 2 (e(H−1,H) + eH). Then 2 H−2 · πθ(y1:<sup>H</sup> | ⊥) ≤ 1/2 for any y1:<sup>H</sup> with yH−<sup>1</sup> = −1 or y<sup>H</sup> = −1, since this implies that πθ<sup>H</sup> (y<sup>H</sup> | ⊥, y1:H−1) ≤ 1/2. Meanwhile, for any y1:<sup>H</sup> with yH−<sup>1</sup> = y<sup>H</sup> = 1, we have

$$2^{H-2} \cdot \pi_{\theta}(y_{1:H} \mid \perp) = \frac{\exp\left(\sum_{i,j \leq H-2} J_{ij} y_i y_j\right)}{\exp\left(\sum_{i,j \leq H-2} J_{ij} y_i y_j\right) + \exp\left(-\sum_{i,j \leq H-2} J_{ij} y_i y_j\right)} \cdot \frac{\exp(B)}{\exp(B) + \exp(-B)}.$$

Let G be any graph on vertex set [H − 2] and let J = −A(G) where A(G) is the adjacency matrix of G. Then among y1:<sup>H</sup> with yH−<sup>1</sup> = y<sup>H</sup> = 1, 2 H−2 · πθ(y1:<sup>H</sup> | ⊥) is maximized when y1:H−<sup>2</sup> corresponds to a max-cut in G. If G has an odd number of edges, then some max-cut removes strictly more than half of the edges, and for the corresponding sequence y1:<sup>H</sup> we have 2 H−2 · πθ(y1:<sup>H</sup> | ⊥) ≥ (1/2 + Ω(1)) · (1 − exp(−Ω(B))), which is greater than 1/2 when we take B := H and H is sufficiently large. Thus, computing arg max<sup>y</sup>1:H∈V<sup>H</sup> πθ(y1:<sup>H</sup> | ⊥) yields a max-cut of G. It is well-known that computing a max-cut in a graph is NP-hard, and the assumption that G has an odd number of edges is without loss of generality.

#### E.2 REPRESENTATIONAL CHALLENGES

To give provable guarantees for our sharpening algorithms, we required certain *realizability* assumptions, which in particular posited that the model class actually contains a "sharpened" version of πbase [\(Assumptions 4.1](#page-6-3) and [4.3\)](#page-8-2). In the simple example of a *single-layer* linear softmax model classes (corresponding to H = 1 in the above definition), [Assumption 4.3](#page-8-2) is in fact satisfied, and the sharpened model can be obtained by increasing the temperature of πbase. However, multi-layer linear softmax models with H ≫ 1 are more realistic. The following proposition shows that as soon as H ≥ 2, multi-layer linear softmax model classes may not be closed under sharpening. This illustrates a potential drawback of training-time sharpening compared to inference-time sharpening, which requires no realizability assumptions. It also provides a simple example where greedy decoding does not yield a sequence-level arg-max response (since increasing temperature in a multi-layer softmax model class exactly converges to the greedy decoding).

Proposition E.2. *Let* X = {⊥}*,* V = [n]*, and* H = d = 2*. For any* n *sufficiently large, there is a multi-layer linear softmax policy class* Πϕ,B,H *and a policy* πbase ∈ Πϕ,B,H *such that* y ⋆ 1:<sup>H</sup> :=

arg max<sup>y</sup>1:H∈V<sup>H</sup> <sup>π</sup>θ(y1:<sup>H</sup> | ⊥) *is unique, but for all* <sup>B</sup>′ > B *and* <sup>π</sup> ∈ <sup>Π</sup>ϕ,B′ ,H*, it holds that* π(y ⋆ 1:<sup>H</sup> | ⊥) ≤ 1/2*.*

Proof of [Proposition E.2.](#page-31-3) Throughout, we omit the dependence on the prompt ⊥ for notational clarity. Since H = 2, the model class consists of models π<sup>θ</sup> of the form

$$\pi_\theta(a) = \pi_{\theta_1}(y_1)\pi_{\theta_2}(y_2 \mid y_1) = \frac{\exp(\langle \phi_1(y_1), \theta_1 \rangle)}{Z_{\theta_1}} \frac{\exp(\langle \phi_2(y_{1:2}), \theta_2 \rangle)}{Z_{\theta_2}(y_1)} \quad (12)$$

for Zθ<sup>1</sup> := P <sup>y</sup>1∈V exp(⟨ϕ1(y1), θ1⟩) and Zθ<sup>2</sup> (y1) := P <sup>y</sup>2∈V exp(⟨ϕ2(y1:2), θ2⟩).

Define ϕ<sup>1</sup> by:

$$\phi_1(i) = \begin{cases} e_1 & \text{if } i = 1 \\ e_1 & \text{if } i = 2 \\ e_2 & \text{if } i \geq 3 \end{cases}$$

Define ϕ<sup>2</sup> by:

$$\phi_2(i, j) = \begin{cases} e_1 & \text{if } i = 2, j = 1 \\ e_2 & \text{if } i = 2, j \neq 1 \\ 0 & \text{if } i \neq 2 \end{cases}$$

Define πbase := π<sup>θ</sup> <sup>⋆</sup> where θ ⋆ 1 := θ ⋆ 2 := B·e<sup>1</sup> for a parameter B ≥ log(n). Then πbase(1) = πbase(2) and πbase(i) ≤ e <sup>−</sup><sup>B</sup>πbase(2) for all i ∈ {3, . . . , n}. Moreover, πbase(· | i) = Unif([n]) for all i ̸= 2, and πbase(j | 2) ≤ e <sup>−</sup><sup>B</sup>πbase(1 | 2) for all j ̸= 1. Thus,

$$\pi_{\text{base}}(2, 1) = \pi_{\text{base}}(2)\pi_{\text{base}}(1 \mid 2) \geq \frac{1}{2 + (n-2)e^{-B}} \cdot \frac{1}{1 + (n-1)e^{-B}} \geq \Omega(1)$$

whereas πbase(i, j) = O(1/n) for all (i, j) ̸= (2, 1). Thus, (2, 1) is the sequence-level argmax for sufficiently large n. However, for any π<sup>θ</sup> of the form described in [Eq. \(12\),](#page-31-4) we have

$$\pi_\theta(2, 1) \leq \pi_\theta(2) \leq \frac{\pi_\theta(2)}{\pi_\theta(1) + \pi_\theta(2)} = \frac{1}{2}$$

since ϕ(1) = ϕ(2). This means that there is no B′ for which Πϕ,B′ ,H contains an (ϵ, δ)-sharpened policy for πbase for any δ > 1/2.

# Part II Proofs

# F PRELIMINARIES

#### F.1 GUARANTEES FOR APPROXIMATE MAXIMIZERS

Recall that the theoretical guarantees for sharpening algorithms in [Section 4](#page-6-1) provide convergence to the set y ⋆ (x) := arg maxy∈Y πbase(y | x) of (potentially non-unique) maximizers for the maximum-likelihood sharpening self-reward function log πbase(y | x). These guarantees require that the base model πbase places sufficient provability mass on y ⋆ (x), which may not always be realistic. To address this, throughout this appendix we state and prove more general versions of our theoretical results that allow for approximate maximizers, and consequently enjoy weaker coverage assumptions

For a parameter γ ∈ [0, 1) we define

$$\mathbf{y}_\gamma^*(x) := \left\{ y \mid \pi_{\text{base}}(y \mid x) \geq (1 - \gamma) \cdot \max_{y \in \mathcal{Y}} \pi_{\text{base}}(y \mid x) \right\}$$

as the set of (1 − γ)-approximate maximizers for log πbase(y | x). We quantify the quality of a sharpened model as follows.

Definition F.1 (Sharpened model). *We say that a model* <sup>π</sup>b *is* (ϵ, δ, γ)*-sharpened relative to* <sup>π</sup>base *if*

$$\mathbb{P}_{x \sim \mu} [\widehat{\pi}(\mathbf{y}_\gamma^*(x) \mid x) \geq 1 - \delta] \geq 1 - \epsilon.$$

That is, an (ϵ, δ, γ)-sharpened policy places at least 1 − δ mass on (1 − γ)-approximate arg-max responses on all but an ϵ-fraction of prompts under µ.

Lastly, we will make use of the following generalized coverage coefficient

$$C_{\text{cov}, \gamma} = \mathbb{E}_{x \sim \mu} \left[ \frac{1}{\pi_{\text{base}}(\mathbf{y}_\gamma^*(x) \mid x)} \right],$$

which has Ccov,γ ≤ Ccov.

# F.2 TECHNICAL TOOLS

For a pair of probability measures P and Q with a common dominating measure ω, Hellinger distance is defined via

$$D_{\text{H}}^2(\mathbb{P}, \mathbb{Q}) = \int \left( \sqrt{\frac{\text{d}\mathbb{P}}{\text{d}\omega}} - \sqrt{\frac{\text{d}\mathbb{Q}}{\text{d}\omega}} \right)^2 \text{d}\omega.$$

Lemma F.1 (MLE for conditional density estimation (e.g., [Wong & Shen](#page-15-16) [\(1995\)](#page-15-16); [van de Geer](#page-15-17) [\(2000\)](#page-15-17); [Zhang](#page-16-6) [\(2006\)](#page-16-6))). *Consider a conditional density* π ⋆ : X → ∆(Y)*. Let* D = {(x<sup>i</sup> , yi)} n <sup>i</sup>=1 *be a dataset in which* (x<sup>i</sup> , yi) *are drawn i.i.d. as* x<sup>i</sup> ∼ µ ∈ ∆(X ) *and* y<sup>i</sup> ∼ π ⋆ (· | x)*. Suppose we have a finite function class* Π ⊂ (X → ∆(Y)) *such that* π <sup>⋆</sup> ∈ Π*. Define the maximum likelihood estimator*

$$\hat{\pi} := \arg \max_{\pi \in \Pi} \sum_{(x,y) \in \mathcal{D}} \log \pi(y \mid x).$$

*Then with probability at least* 1 − ρ*,*

$$\mathbb{E}_{x \sim \mu} [D_{\text{H}}^2(\hat{\pi}(\cdot \mid x), \pi^*(\cdot \mid x))] \leq \frac{2 \log(|\Pi| \rho^{-1})}{n}.$$

Lemma F.2 (Elliptic potential lemma). *Let* λ, K > 0*, and let* A1, . . . , A<sup>T</sup> ∈ <sup>R</sup> <sup>d</sup>×<sup>d</sup> *be positive semi-definite matrices with* Tr(At) ≤ K *for all* t ∈ [T]*. Fix* Γ<sup>0</sup> = λI<sup>d</sup> *and* Γ<sup>t</sup> = λI<sup>d</sup> + P<sup>t</sup> <sup>i</sup>=1 A<sup>i</sup> *for* t ∈ [T]*. Then*

$$\sum_{t=1}^T \text{Tr}(\Gamma_{t-1}^{-1} A_t) \leq \frac{dK \log \frac{(T+1)K}{\lambda}}{\lambda \log(1 + K/\lambda)}.$$

Proof of [Lemma F.2.](#page-33-5) Fix t ∈ [T]. Since Tr(At) ≤ 1, there is some p<sup>t</sup> ∈ ∆(<sup>R</sup> d ) such that A<sup>t</sup> = <sup>E</sup>a∼p<sup>t</sup> [aa<sup>⊤</sup>] and <sup>P</sup>[∥a∥<sup>2</sup> ≤ 1] = 1. Now observe that

$$\begin{aligned} \log \det(\Gamma_t) &= \log \det(\Gamma_{t-1} + A_t) \\ &= \log \det(\Gamma_{t-1}) + \log \det(I_d + \Gamma_{t-1}^{-1/2} A_t \Gamma_{t-1}^{-1/2}) \\ &= \log \det(\Gamma_{t-1}) + \log \det \left( \mathbb{E}_{a \sim p_t} \left[ I_d + \Gamma_{t-1}^{-1/2} a a^\top \Gamma_{t-1}^{-1/2} \right] \right) \\ &\geq \log \det(\Gamma_{t-1}) + \mathbb{E}_{a \sim p_t} \log \det(I_d + \Gamma_{t-1}^{-1/2} a a^\top \Gamma_{t-1}^{-1/2}) \\ &= \log \det(\Gamma_{t-1}) + \mathbb{E}_{a \sim p_t} \log(1 + a^\top \Gamma_{t-1}^{-1} a). \end{aligned}$$

Now a <sup>⊤</sup>Γ −1 t−1 a ≤ 1/λ with probability 1, where λ = λmin(Γ0). We know that λx log(1 + 1/λ) ≤ log(1 + x) for all x ∈ [0, 1/λ]. Thus,

$$\log \det(\Gamma_t) \geq \log \det(\Gamma_{t-1}) + \lambda \log(1 + 1/\lambda) \mathbb{E}_{a \sim p_t} a^\top \Gamma_{t-1}^{-1} a.$$

Summing over t ∈ [T], we get

$$\log \det(\Gamma_T) \geq \log \det(\Gamma_0) + \lambda \log(1 + 1/\lambda) \sum_{t=1}^T \text{Tr}(\Gamma_{t-1}^{-1} A_t).$$

Finally note that λmax(Γ<sup>T</sup> ) ≤ T + 1 so log det(Γ<sup>T</sup> ) ≤ d log T, whereas log det(Γ0) ≥ d log λ. Thus,

$$\sum_{t=1}^T \text{Tr}(\Gamma_{t-1}^{-1} A_t) \leq \frac{d \log \frac{T+1}{\lambda}}{\lambda \log(1 + 1/\lambda)}$$

as claimed.

Lemma F.3 (Freedman's inequality, e.g. [Agarwal et al.](#page-10-14) [\(2014\)](#page-10-14)). *Let* (Zt) T <sup>t</sup>=1 *be a martingale difference sequence adapted to filtration* (Ft) T −1 <sup>t</sup>=0 *. Suppose that* |Zt| ≤ R *holds almost surely for all* t*. For any* δ ∈ (0, 1) *and* η ∈ (0, 1/R)*, it holds with probability at least* 1 − δ *that*

$$\sum_{t=1}^T Z_t \leq \eta \sum_{t=1}^T \mathbb{E}[Z_t^2 | \mathcal{F}_{t-1}] + \frac{\log(1/\delta)}{\eta}.$$

Corollary F.1. *Let* (Zt) T <sup>t</sup>=1 *be a sequence of random variables adapted to filtration* (Ft) T −1 <sup>t</sup>=0 *. Suppose that* Z<sup>t</sup> ∈ [0, R] *holds almost surely for all* t*. For any* δ ∈ (0, 1)*, it holds with probability at least* 1 − δ *that*

$$\sum_{t=1}^T \mathbb{E}[Z_t | \mathcal{F}_{t-1}] \leq 2 \sum_{t=1}^T Z_t + 4R \log(1/\delta).$$

Proof of [Corollary F.1.](#page-34-0) Observe that for any t ∈ [T],

$$\begin{aligned}\mathbb{E}[(Z_t - \mathbb{E}[Z_t \mid \mathcal{F}_{t-1}])^2 \mid \mathcal{F}_{t-1}] &\leq \mathbb{E}[Z_t^2 \mid \mathcal{F}_{t-1}] \\ &\leq R \cdot \mathbb{E}[Z_t \mid \mathcal{F}_{t-1}].\end{aligned}$$

Applying [Lemma F.3](#page-34-1) to the sequence (E[Z<sup>t</sup> | Ft−1] − Zt) T <sup>t</sup>=1, which is a martingale difference sequence with elements supported almost surely on [−R, R], we get for any η ∈ (0, 1/R) that with probability at least 1 − δ,

$$\begin{aligned} \sum_{t=1}^T (\mathbb{E}[Z_t \mid \mathcal{F}_{t-1}] - Z_t) &\leq \eta \sum_{t=1}^T \mathbb{E}[(Z_t - \mathbb{E}[Z_t \mid \mathcal{F}_{t-1}])^2 \mid \mathcal{F}_{t-1}] + \frac{\log(1/\delta)}{\eta} \\ &\leq \eta R \sum_{t=1}^T \mathbb{E}[Z_t \mid \mathcal{F}_{t-1}] + \frac{\log(1/\delta)}{\eta}. \end{aligned}$$

Set η = 1/(2R). Simplifying gives

$$\sum_{t=1}^T \mathbb{E}[Z_t \mid \mathcal{F}_{t-1}] \leq 2 \sum_{t=1}^T Z_t + 4R \log(1/\delta).$$

# G PROOFS FROM S[ECTION](#page-4-0) 3.1

Proof of [Proposition 3.1.](#page-5-3) We prove the result by induction. Fix x ∈ X , and let y ⋆ 1 , . . . , y<sup>⋆</sup> <sup>H</sup> := y ⋆ (x). Fix <sup>h</sup> <sup>∈</sup> [H], and assume by induction that <sup>y</sup>bh′ <sup>=</sup> <sup>y</sup> ⋆ <sup>h</sup>′ for all h ′ < h. We claim that in this case,

$$\pi_h(y_h^* \mid \widehat{y}_1, \dots, \widehat{y}_{h-1}, x) = \pi_h(y_h^* \mid y_1^*, \dots, y_{h-1}^*, x) > 1/2,$$

which implies that <sup>y</sup>b<sup>h</sup> <sup>=</sup> <sup>y</sup> ⋆ h . To see this, we observe that by Bayes' rule,

$$\begin{aligned}\pi(y_1^*, \dots, y_H^* \mid x) &\leq \pi(y_1^*, \dots, y_h^* \mid x) \\ &= \prod_{h'=1}^h \pi_{h'}(y_{h'}^* \mid y_1^*, \dots, y_{h'-1}^*, x) \leq \pi_h(y_h^* \mid y_1^*, \dots, y_{h-1}^*, x).\end{aligned}$$

If we were to have πh(y ⋆ h | <sup>y</sup>b1, . . . , <sup>y</sup>bh−1, x) = <sup>π</sup>h(<sup>y</sup> ⋆ h | y ⋆ 1 , . . . , y<sup>⋆</sup> h−1 , x) ≤ 1/2, it would contradict the assumption that π(y ⋆ 1 , . . . , y<sup>⋆</sup> <sup>H</sup> | x) > 1/2. This proves the result.

# H PROOFS FROM S[ECTION](#page-6-0) 3.3

Below, we state and prove a generalization of [Theorems 3.1](#page-6-2) and [D.2](#page-30-1) which allows for approximate maximizers in the sense of [Definition F.1,](#page-33-1) as well as a more general coverage coefficient.

To state the result, for a model π, we define

$$\mathbf{y}_\gamma^\pi(x) = \left\{ y \mid \pi(y \mid x) \geq (1 - \gamma) \cdot \max_{y \in \mathcal{Y}} \pi(y \mid x) \right\}.$$

Next, for any integer p ∈ N, we define

$$C_{\text{cov}, \gamma, p}(\pi) = \left( \mathbb{E} \left[ \frac{1}{(\pi(\mathbf{y}_\gamma^\pi(x) \mid x))^p} \right] \right)^{1/p},$$

with the convention that Ccov,γ,p = Ccov,γ,p(πbase). Our most general lower bound, [Theorem 3.1](#page-35-2)′ , holds in the regime where γ = 1/2, and thus the best responsey has bounded margin away from suboptimal responses.

Theorem 3.1′ (Lower bound for sharpening). *Fix integers* d ≥ 1 *and* p ≥ 1 *and parameters* ϵ ∈ (0, 1) *and* C ≥ 1*, and set* γ = 1/2*. There exists a class of models* Π *such that i)* log |Π| ≍ d(1 + log(Cϵ−1/p))*, ii)* supπ∈<sup>Π</sup> Ccov,γ,p(π) ≲ C*, and iii)* y π γ (x) *is a singleton for all* π ∈ Π*, for which any sharpening algorithm* <sup>π</sup>b *that attains* <sup>E</sup> -<sup>P</sup>x∼µ[πb(<sup>y</sup> πbase γ (x)) > 1/2] ≥ 1 − ϵ *for all* πbase ∈ Π *must collect a total number of samples* m = n · N *at least*

$$m \gtrsim \begin{cases} \frac{C \log |\Pi|}{\epsilon^{1+1/p}(1+\log(C\epsilon^{-1/p}))} & \text{sample-and-evaluate oracle,} \\ \frac{C \log |\Pi|}{\epsilon^{1/p}(1+\log(C\epsilon^{-1/p}))} & \text{adaptive sample-and-evaluate oracle.} \end{cases}$$

Proof of [Theorem 3.1](#page-35-2)′ . Let parameters d, p ∈ N and ϵ > 0 be given, and set γ = 1/2. Let M ∈ N and ∆ > 0 be parameters to be chosen later. Let X = {x0, x1, . . . , xd} and Y = {y0, y1, . . . , yM} be arbitrary discrete sets (with |X | = d + 1 and |Y| = M + 1).

Construction of prompt distribution and model class. We use the same construction for the non-adaptive and adaptive lower bounds in the theorem statement. We define the prompt distribution µ via

$$\mu := (1 - \Delta)\delta_{x_0} + \frac{\Delta}{d} \sum_{i=1}^d \delta_{x_i},$$

where δ<sup>x</sup> denotes the Dirac delta distribution on element x.

As the first step toward constructing the model class Π, we introduce a family of distributions (P0, P1, . . . , PM) on Y as follows

$$P_0 = \delta_{y_0}, \quad \forall i \geq 1, \quad P_i = \frac{1}{(1-\gamma)M} \delta_{y_i} + \sum_{j \in [M] \setminus \{i\}} \frac{1}{M} \left(1 - \frac{\gamma}{(M-1)(1-\gamma)}\right) \delta_{y_j}.$$

Next, for or any index I = (j1, j2, . . . , jd) ∈ [M] d , define a model

$$\pi^{\mathcal{I}}(x_i) = \begin{cases} P_0 & i = 0 \\ P_{j_i} & i > 0 \end{cases}.$$

We define the model class as

$$\Pi := \{\pi^{\mathcal{I}} : \mathcal{I} \in [M]^d\},$$

which we note has

$$\log |\Pi| = d \log M.$$

Preliminary technical results. Define

$$\mathbf{y}_\gamma^\mathcal{I}(x) := \{y : \pi^\mathcal{I}(y \mid x) \geq (1 - \gamma) \max_{y \in \mathcal{Y}} \pi^\mathcal{I}(y \mid x)\}.$$

The following property is immediate.

**Lemma H.1.** *Let 
$$\mathcal{I} = (j_1, \dots, j_d) \in [d]^M$$
. Then  $\mathbf{y}_\gamma^{\mathcal{I}}(x_i) = \{y_{j_i}\}$  if  $i > 0$ , and  $\mathbf{y}_\gamma^{\mathcal{I}}(x_0) = \{y_0\}$ .*

In view of this result, we define y I (x) = arg max<sup>y</sup> π I (y | x) as the unique arg-max response for x.

Going forward, let us fix the algorithm under consideration. Let P I [·] denote the law over the dataset used by the algorithm when the true instance is π I (including possible randomness and adaptivity from the algorithm itself), and let E I [·] denote the corresponding expectation. The following lemma is a basic technical result.

Lemma H.2 (Reduction to classification). *Let* <sup>π</sup>b *be the model produced by an algorithm with access to a (adaptive) sample-and-evaluate oracle for* π I *. Suppose that for some* ϵ ≥ 0*,*

$$\mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{E}^{\mathcal{I}} \mathbb{P}_{x \sim \mu}[\widehat{\pi}(\mathbf{y}_{\gamma}^{\mathcal{I}}(x) \mid x) > 1/2] \geq 1 - \epsilon.$$

*Define* <sup>I</sup>b = (b<sup>j</sup>1, . . . ,b<sup>j</sup>d) *via* b<sup>j</sup><sup>i</sup> = arg max<sup>j</sup> <sup>π</sup>b(y<sup>j</sup> <sup>|</sup> <sup>x</sup>i)*, and write* <sup>I</sup> = (<sup>j</sup> ⋆ 1 , . . . , j<sup>⋆</sup> d )*. Then,*

$$\frac{1}{d} \sum_{i=1}^d \mathbb{E}_{\mathcal{I} \sim \text{unif}} \mathbb{E}^{\mathcal{I}} \left[ \mathbb{I}\{\widehat{j}_i \neq j_i^*\} \right] \leq \epsilon / \Delta.$$

Proof of [Lemma H.2.](#page-36-0) As established in [Lemma H.1,](#page-36-1) under instance I, y I γ (xi) = {y<sup>j</sup> i } for any <sup>i</sup> <sup>∈</sup> [d]. Thus, whenever <sup>π</sup>b(<sup>y</sup> I γ (xi)) > 1/2, j ⋆ <sup>i</sup> = arg max<sup>j</sup> <sup>π</sup>b(y<sup>j</sup> <sup>|</sup> <sup>x</sup>i) =: b<sup>j</sup><sup>i</sup> . The result follows by noting that the event {∃i ∈ [d] : x = xi} occurs with probability at least ∆ under x ∼ µ.

Lower bound under sample-and-evaluate oracle. Recall that in the non-adaptive framework, the sample complexity m is fixed. In light of [Lemma H.2,](#page-36-0) it suffices to establishes the following claim.

Lemma H.3. *There exists a universal constant* c > 0 *such that for all* M ≥ 8*, if* m ≤ cdM/∆*, then* EI∼Unif E I h <sup>I</sup>{b<sup>j</sup><sup>i</sup> ̸<sup>=</sup> <sup>j</sup> ⋆ i } i ≥ 1/8 *for all* i*.*

With this, the result follows by selecting ∆ = 16ϵ, with which [Lemma H.2](#page-36-0) implies that any algorithm with EI∼Unif E <sup>I</sup> <sup>P</sup>x∼µ[πb(<sup>y</sup> I γ (x) | x) > 1/2] ≥ 1 − ϵ must have m ≳ dM/∆. To conclude, we choose M ≍ 1 + Cϵ−1/p, which gives m ≍ dM/∆ ≍ dCϵ−(1+1/p) ≍ ϵ −(1+1/p) log Π/ log(1 + Cϵ<sup>1</sup>/p). Finally, we check that with this choice, all π ∈ Π satisfy

$$\begin{aligned} C_{\text{cov}, \gamma, p}(\pi) &= (\mathbb{P}_{x \sim \mu}[x = x_0] + (M(1 - \gamma))^p \mathbb{P}_{x \sim \mu}[x \neq x_0])^{1/p} \\ &= ((1 - \Delta) + (M(1 - \gamma))^p \Delta)^{1/p} \\ &\lesssim ((1 - \Delta) + (8C(1 - \gamma))^p)^{1/1/p} \lesssim C. \end{aligned}$$

Proof of [Lemma H.3.](#page-36-2) Let i ∈ [d] be fixed. Of the m = n · N tuples (x, y, log πbase(y | x)) that are observed by the algorithm, let m<sup>i</sup> denote the (random) number of such examples for which x = x<sup>i</sup> . From Markov's inequality, we have

$$\mathbb{P}[m_i \leq 2\Delta m/d] \geq \frac{1}{2} \quad (13)$$

Going forward, let D = {(x, y, log πbase(y | x))} denote the dataset collected by the algorithm, which has |D| = m. Let E<sup>i</sup> denote the event that, for prompt x = x<sup>i</sup> , (i) there are at least two distinct responses y<sup>j</sup> for which (x<sup>i</sup> , y<sup>j</sup> ) ∈ D / ; and (ii) there are no pairs (x<sup>i</sup> , y) ∈ D for which πbase(y | xi) > 1 <sup>M</sup> . Since E<sup>i</sup> is a measurable function of D, we can write

$$\begin{aligned}\mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{E}^{\mathcal{I}} \left[ \mathbb{I}\{\hat{j}_i \neq j_i^*\} \right] &\geq \mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{E}^{\mathcal{I}} \left[ \mathbb{I}\{\hat{j}_i \neq j_i^*\} \cdot \mathbb{I}\{\mathcal{E}_i\} \right] \\ &= \mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{E}^{\mathcal{I}} \left[ \mathbb{I}\{\mathcal{E}_i\} \mathbb{E}_{\mathcal{I} \sim \mathbb{P}(\mathcal{I}=\cdot|\mathcal{D})} \left[ \mathbb{I}\{\hat{j}_i \neq j_i^*\} \right] \right],\end{aligned}\tag{14}$$

where I ∼ <sup>P</sup>[I = · | D] is sampled from the posterior distribution over I conditioned on the dataset D. Observe that conditioned on E<sup>i</sup> , the posterior distribution over j ⋆ i under I ∼ <sup>P</sup>[I = · | D] is uniform over the set of indices j ∈ [M] for which (x<sup>i</sup> , y<sup>j</sup> ) ∈ D/ , and this set has size at least 2. Hence, <sup>I</sup>{Ei} <sup>E</sup>I∼P[I=·|D] h <sup>I</sup>{b<sup>j</sup><sup>i</sup> ̸<sup>=</sup> <sup>j</sup> ⋆ } i ≥ 1

i

2

, and resuming from [Eq. \(14\),](#page-37-0) we have

$$\begin{aligned}\mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{E}^{\mathcal{I}} \left[ \mathbb{I}\{\hat{j}_i \neq j_i^*\} \right] &\geq \frac{1}{2} \mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{E}^{\mathcal{I}} [\mathbb{I}\{\mathcal{E}_i\}] \geq \frac{1}{2} \mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{P}^{\mathcal{I}} [\mathcal{E}_i \cap \{m_i \leq 2\Delta m/d\}] \\ &\geq \frac{1}{4} \mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{P}^{\mathcal{I}} [\mathcal{E}_i \mid m_i \leq 2\Delta m/d] ,\end{aligned}$$

where the last inequality is from [Eq. \(13\).](#page-36-3) Finally, we can check that under the law P I , the probability of the event Ei—conditioned on the value mi—is at least the probability that (x<sup>i</sup> , y<sup>j</sup> ⋆ ),(x<sup>i</sup> , y<sup>j</sup> ′ ) ∈ D/ for an arbitrary fixed index j ′ ̸= j ⋆ i , which on the event {m<sup>i</sup> ≤ 2∆m/d} is at least

$$\left(1 - \frac{3}{M}\right)^{m_i} \geq \left(1 - \frac{3}{M}\right)^{2\Delta m/d},$$

where we have used that γ = 1/2. The value above is at least <sup>1</sup> <sup>4</sup> whenever m ≤ c · dM/∆ for a sufficiently small absolute constant c > 0. For this value of m, we conclude that EI∼Unif E I h <sup>I</sup>{b<sup>j</sup><sup>i</sup> ̸<sup>=</sup> <sup>j</sup> ⋆ i } i ≥ 1 4 EI∼Unif P I [E<sup>i</sup> | {m<sup>i</sup> ≤ 2∆m/d}] ≥ 1 8 .

Lower bound under adaptive sample-and-evaluate oracle. In the adaptive framework, we let m<sup>i</sup> denote the (potentially random) number of tuples (x, y, log πbase(y | x)) observed by the algorithm in which x = x<sup>i</sup> . Note that unlike the non-adaptive framework, the distribution over m<sup>i</sup> depends on the underlying instance I with which the algorithm interacts.

To begin, from [Lemma H.2](#page-36-0) and Markov's inequality, if <sup>π</sup>b satisfies the guarantee EI∼Unif E <sup>I</sup> <sup>P</sup>x∼µ[πb(<sup>y</sup> I γ (x)) > 1/2] ≥ 1 − ϵ, then there exists a set of indices Sgood ⊂ [d] such that[<sup>14</sup>](#page-0-0)

$$|S_{\text{good}}| \geq \lfloor d/2 \rfloor, \quad \forall i \in S_{\text{good}}, \quad \mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{E}^{\mathcal{I}} \left[ \mathbb{I}\{\hat{j}_i \neq j_i^*\} \right] \leq \frac{2\epsilon}{\Delta}. \quad (15)$$

We now appeal to the following lemma.

Lemma H.4. *As long as* M ≥ 6*, it holds that for all* i ∈ [d]*,*

$$\mathbb{E}_{\mathcal{I} \sim \text{unif}} \mathbb{E}^{\mathcal{I}} \left[ \mathbb{I}\{\hat{j}_i \neq j_i^*\} \right] \geq \frac{1}{4e} \mathbb{E}_{\mathcal{I} \sim \text{unif}} \mathbb{E}^{\mathcal{I}} [\mathbb{I}\{m_i \leq M/3\}].$$

Combining [Lemma H.4](#page-37-1) with [Eq. \(15\),](#page-37-2) it follows that there exist absolute constant c1, c2, c<sup>3</sup> > 0 such that if ∆ = c<sup>1</sup> · ϵ, then for all i ∈ Sgood,

$$\mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{P}^{\mathcal{I}}[m_i \geq c_2 M] \geq c_3.$$

Thus, with this choice for ∆, we have that i ∈ Sgood,

$$\mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{E}^{\mathcal{I}} [m_i] \gtrsim M,$$

<sup>14</sup>We emphasize that the set Sgood is not a random variable, and depends only on the algorithm itself.

and we can lower bound the algorithm's expected sample complexity by summing over i ∈ Sgood:

$$\mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{E}^{\mathcal{I}} [m] \geq \mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{E}^{\mathcal{I}} \left[ \sum_{i \in S_{\text{good}}} m_i \right] \gtrsim |S_{\text{good}}| M \gtrsim dM.$$

The result now follows by tuning M ≍ 1 + Cϵ−1/p as in the proof of the lower bound for non-adaptive sampling, which gives <sup>E</sup>[m] ≳ dM ≍ dCϵ−1/p ≍ ϵ <sup>−</sup>1/p log Π/ log(1 + Cϵ<sup>1</sup>/p) and Ccov,γ,p(π) ≲ C for all π ∈ Π.

Proof of [Lemma H.4.](#page-37-1) Let i ∈ [d] be fixed. Let D = {(x, y, log πbase(y | x))} denote the dataset collected by the algorithm at termination, which has |D| = m. Let E<sup>i</sup> denote the event that, for prompt x = x<sup>i</sup> , (i) there are at least two distinct responses y<sup>j</sup> for which (x<sup>i</sup> , y<sup>j</sup> ) ∈ D/ ; and (ii) there are no pairs (x<sup>i</sup> , y) ∈ D for which πbase(y | xi) > 1 <sup>M</sup> . Since E<sup>i</sup> is a measurable function of D, we can write

$$\begin{aligned}\mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{E}^{\mathcal{I}} \left[ \mathbb{I}\{\hat{j}_i \neq j_i^*\} \right] &\geq \mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{E}^{\mathcal{I}} \left[ \mathbb{I}\{\hat{j}_i \neq j_i^*\} \cdot \mathbb{I}\{\mathcal{E}_i\} \right] \\ &= \mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{E}^{\mathcal{I}} \left[ \mathbb{I}\{\mathcal{E}_i\} \mathbb{E}_{\mathcal{I} \sim \mathbb{P}(\mathcal{I}=\cdot|\mathcal{D})} \left[ \mathbb{I}\{\hat{j}_i \neq j_i^*\} \right] \right],\end{aligned}\tag{16}$$

where I ∼ <sup>P</sup>[I = · | D] is sampled from the posterior distribution over I conditioned on the dataset D. Observe that conditioned on E<sup>i</sup> , the posterior distribution over j ⋆ i under I ∼ <sup>P</sup>[I = · | D] is uniform over the set of indices j ∈ [M] for which (x<sup>i</sup> , y<sup>j</sup> ) ∈ D/ , and this set has size at least 2. Hence, <sup>I</sup>{Ei} <sup>E</sup>I∼P[I=·|D] h <sup>I</sup>{b<sup>j</sup><sup>i</sup> ̸<sup>=</sup> <sup>j</sup> ⋆ i } i ≥ 1 2 , and resuming from [Eq. \(16\),](#page-38-1) we have

$$\begin{aligned}\mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{E}^{\mathcal{I}} \left[ \mathbb{I}\{\hat{j}_i \neq j_i^*\} \right] &\geq \frac{1}{2} \mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{E}^{\mathcal{I}} [\mathbb{I}\{\mathcal{E}_i\}] \\ &\geq \frac{1}{2} \mathbb{E}_{\mathcal{I} \sim \text{Unif}} \mathbb{P}^{\mathcal{I}} [\mathcal{E}_i \cap \{m_i \leq M/3\}] \\ &= \frac{1}{2} \mathbb{E}_{\mathcal{I} \sim \text{Unif}} [\mathbb{P}^{\mathcal{I}} [\mathcal{E}_i \mid m_i \leq M/3] \cdot \mathbb{P}^{\mathcal{I}} [m_i \leq M/3]].\end{aligned}$$

The event E<sup>i</sup> is a superset of the event Ei,j′ that (x<sup>i</sup> , y<sup>j</sup> i ),(x<sup>i</sup> , y<sup>j</sup> ′ ) ∈ D/ for an arbitrary fixed index j ′ ̸= j ⋆ i . Thus,

$$\mathbb{P}^{\mathcal{I}} [\mathcal{E}_i \mid m_i \leq M/3] \geq \mathbb{P}^{\mathcal{I}} [\mathcal{E}_{i,j'} \mid m_i \leq M/3]$$

Moreover, we can realize the law of P I considering an infinite tape, associated to index i, of i.i.d. samples y ∼ πbase(· | xi), and taking the first m<sup>i</sup> elements on this tape to be the samples (x, y, log πbase(y | x)) ∈ D with x = x<sup>i</sup> (see, e.g. [Simchowitz et al.](#page-14-16) [\(2017\)](#page-14-16) for an argument of this form). On the event {m<sup>i</sup> ≤ M/3}, the m<sup>i</sup> samples in (x, y, log πbase(y | x)) ∈ D with x = x<sup>i</sup> are a subset of the first M/3 samples from the index-i tape. Viewed in this way, we can lower bound the probability of Ei,j by the probability of the event E˜ i,j′ that the first M/3 y's on the index-i tape contain neither j ⋆ i , nor the designated index j ′ . As these first M/3 y's are not chosen adaptively, the probability of E˜ i,j′ is at least

$$\left(1 - \frac{3}{M}\right)^{m_i} \geq \left(1 - \frac{3}{M}\right)^{M/3} \geq \frac{1}{2e},$$

as long as M ≥ 6 and γ = 1/2. We conclude that

$$\mathbb{E}_{\mathcal{I} \sim \text{unif}} \mathbb{E}^{\mathcal{I}} \left[ \mathbb{I}\{\hat{j}_i \neq j_i^*\} \right] \geq \frac{1}{4e} \mathbb{E}_{\mathcal{I} \sim \text{unif}} \mathbb{E}^{\mathcal{I}} [\mathbb{I}\{m_i \leq M/3\}].$$

Theorem 4.1′ . *Let* ρ, δ ∈ (0, 1) *be given, and suppose we set* N = N<sup>⋆</sup> log(2δ −1 ) *for a parameter* N<sup>⋆</sup> ∈ <sup>N</sup>*. Then for any* n ∈ <sup>N</sup>*,* SFT-Sharpening *ensures that with probability at least* 1 − ρ*, for any* <sup>γ</sup> <sup>∈</sup> (0, 1)*, the output model* <sup>π</sup>b *satisfies*

$$\mathbb{P}_{x \sim \mu}[\widehat{\pi}(\mathbf{y}_\gamma^*(x) \mid x) \leq 1 - 2\delta] \lesssim \frac{1}{\delta} \cdot \frac{\log(|\Pi|\rho^{-1})}{n} + \frac{C_{\text{cov},\gamma}}{N^*}.$$

*In particular, given* (ϵ, δ, γ)*, by setting* n = C[4](#page-7-5).<sup>1</sup> log|Π| δϵ *and* <sup>N</sup><sup>⋆</sup> <sup>=</sup> <sup>C</sup>[4](#page-7-5).<sup>1</sup> Ccov,γ ϵ *for a sufficiently large absolute constant* C[4](#page-7-5).<sup>1</sup> > 0*, we are guaranteed that*

$$\mathbb{P}_{x \sim \mu} \left[ \widehat{\pi}(\mathbf{y}_\gamma^\star(x) \mid x) \leq 1 - \delta \right] \leq \epsilon.$$

*The total sample complexity is*

$$m = O\left(\frac{C_{\text{cov}, \gamma} \log(|\Pi| \rho^{-1}) \log(\delta^{-1})}{\delta \epsilon^2}\right).$$

Proof of [Theorem 4.1](#page-38-2)′ . Under realizability of π BoN <sup>N</sup> [\(Assumption 4.1\)](#page-6-3), [Lemma F.1](#page-33-6) implies that the output of SFT-Sharpening satisfies, with probability at least 1 − ρ,

$$\mathbb{E}_{x \sim \mu} [D_{\text{H}}^2(\hat{\pi}(\cdot \mid x), \pi_N^{\text{BON}}(\cdot \mid x))] \leq \varepsilon_{\text{stat}}^2 := \frac{2 \log(|\Pi|/\rho)}{n}. \quad (17)$$

Henceforth we condition on the event that [Eq. \(17\)](#page-39-0) holds. Let

$$\mathcal{X}_{\text{good}} := \left\{ x \in \mathcal{X} \mid N^* \geq \frac{1}{\pi_{\text{base}}(\mathbf{y}_\gamma^*(x) \mid x)} \right\}$$

denote the set of prompts for which πbase places sufficiently high mass on y ⋆ γ (x). We can bound

$$\begin{aligned} \mathbb{P}_{x \sim \mu}[\hat{\pi}(\mathbf{y}_\gamma^*(x) \mid x) \leq 1 - \delta] \\ \leq \mathbb{P}_{x \sim \mu}[\hat{\pi}(\mathbf{y}_\gamma^*(x) \mid x) \leq 1 - \delta, x \in \mathcal{X}_{\text{good}}] + \mathbb{P}_{x \sim \mu}[x \notin \mathcal{X}_{\text{good}}]. \end{aligned} \quad (18)$$

To bound the first term in [Eq. \(18\),](#page-39-1) note that if x ∈ Xgood, then π BoN <sup>N</sup> (y ⋆ γ (x) | x) ≥ 1 − δ/2. Indeed, observe that y ∼ π BoN <sup>N</sup> (· | x) ∈/ y ⋆ γ (x) if and only if y1, . . . , y<sup>N</sup> ∼ πbase(x) have y<sup>i</sup> ∈/ y ⋆ γ (x) for all i, which happens with probability (1 − πbase(y ⋆ γ (x) | x))<sup>N</sup> ≤ (1 − 1/N<sup>⋆</sup> ) <sup>N</sup> ≤ δ/2 since x ∈ Xgood. It follows that for any such x, we can lower bound (using the data processing inequality)

$$D_N^2(\widehat{\pi}(\cdot | x), \pi_N^{\text{Bon}}(\cdot | x)) \geq \left( \sqrt{1 - \widehat{\pi}(\mathbf{y}_\gamma^*(x) | x)} - \sqrt{1 - \pi_N^{\text{Bon}}(\mathbf{y}_\gamma^*(x) | x)} \right)^2 \\ \gtrsim \delta \cdot \mathbb{I}\{\widehat{\pi}(\mathbf{y}_\gamma^*(x) | x) \leq 1 - \delta\}. \quad (19)$$

By [Eqs. \(17\)](#page-39-0) and [\(19\),](#page-39-2) it follows that

$$\mathbb{P}_{x \sim \mu} [\widehat{\pi}(\mathbf{y}_\gamma^*(x) \mid x) \leq 1 - \delta, x \in \mathcal{X}_{\text{good}}] \lesssim \frac{\varepsilon_{\text{stat}}^2}{\delta}.$$

For the second term in [Eq. \(18\),](#page-39-1) we bound

$$\begin{aligned}\mathbb{P}_{x \sim \mu}[x \notin \mathcal{X}_{\text{good}}] &= \mathbb{P}_{x \sim \mu}\left[N^* < \frac{1}{\pi_{\text{base}}(\mathbf{y}_\gamma^*(x) \mid x)}\right] \\ &= \mathbb{P}_{x \sim \mu}\left[\frac{1}{N^* \pi_{\text{base}}(\mathbf{y}_\gamma^*(x) \mid x)} > 1\right] \\ &\leq \frac{1}{N^*} \mathbb{E}_{x \sim \mu}\left[\frac{1}{\pi_{\text{base}}(\mathbf{y}_\gamma^*(x) \mid x)}\right] \\ &\leq \frac{C_{\text{cov}, \gamma}}{N^*}\end{aligned}$$

Proof of [Theorem D.1.](#page-29-3) The proof begins similarly to [Theorem 4.1.](#page-7-5) By realizability of π<sup>N</sup><sup>µ</sup> , [Lemma F.1](#page-33-6) implies that the output of SFT-Sharpening satisfies, with probability at least 1 − ρ,

$$\mathbb{E}_{x \sim \mu} [D_{\Pi}^2(\hat{\pi}(\cdot \mid x), \pi_{N_{\mu}}(\cdot \mid x))] \leq \varepsilon_{\text{stat}}^2 := \frac{2 \log(|\Pi|/\rho)}{n}.$$

Condition on the event that this guarantee holds. We invoke the following lemma, proven in the sequel.

Lemma I.1. *Let* P *be a distribution on a discrete space* Y*. Let* y <sup>⋆</sup> = arg maxy∈Y P(y) *and let* P ⋆ := maxy∈Y P(y)*. Let* y1, y2, . . . ∼ P*, and for any stopping time* τ *, define*

$$\hat{y}_\tau \in \arg \max \{P(y) : y \in \{y_1, \dots, y_\tau\}\}.$$

*Next, for a parameter* µ > 0*, define the stopping time*

$$N_\mu := \inf \left\{ k : \frac{1}{\max_{1 \leq i \leq k} P(y_i)} \leq k/\mu \right\}.$$

*Then*

$$\mathbb{E}[N_\mu] \leq \frac{\mu + (1/|\mathbf{y}^*|)}{P^*}.$$

*In addition, for any stopping time* <sup>τ</sup> <sup>≥</sup> <sup>N</sup><sup>µ</sup> *(including* <sup>τ</sup> <sup>=</sup> <sup>N</sup><sup>µ</sup> *itself), we have* <sup>P</sup>[yb<sup>τ</sup> <sup>∈</sup>/ <sup>y</sup> ⋆ ] ≤ e −|y ⋆ |µ*.*

This lemma, with our choice of µ, ensures that *for all* x ∈ X ,

$$\pi_{N_\mu}(\mathbf{y}^\star(x) \mid x) \geq 1 - e^{-\mu} = 1 - \delta/2.$$

Following the reasoning in [Eq. \(19\),](#page-39-2) this implies that

$$D_{\text{H}}^2(\widehat{\pi}(\cdot \mid x), \pi_{N_{\mu}}(\cdot \mid x)) \gtrsim \delta \cdot \mathbb{I}\{\widehat{\pi}(\mathbf{y}^*(x) \mid x) \leq 1 - \delta\},$$

so that

$$\mathbb{P}_{x \sim \mu}[\widehat{\pi}(\mathbf{y}^*(x) \mid x) \leq 1 - \delta] \lesssim \frac{\varepsilon_{\text{stat}}^2}{\delta}$$

as desired.

To bound the expected sample complexity, we observe that

$$\mathbb{E}[m] = n \cdot \mathbb{E}[N_\mu(x)] \stackrel{(i)}{\leq} \mathbb{E} \left[ \frac{1 + \mu}{\pi_{\text{base}}(\mathbf{y}^\star(x) \mid x)} \right] = (1 + \mu) \overline{C}_{\text{cov}},$$

where inequality (i) invokes [Lemma I.1](#page-40-0) once more.

Proof of [Lemma I.1.](#page-40-0) Define N<sup>⋆</sup> := µ/P<sup>⋆</sup> . To bound the tails of Nµ, define

$$\tau = \inf\{k \mid k \geq N^* \text{ and } \mathbf{y}^* \cap \{y_1, \dots, y_k\} \neq \emptyset\}.$$

It follows from the definition that N<sup>µ</sup> ≤ τ , since for any k ≥ N<sup>⋆</sup> , if there exists i ≤ k such that y<sup>i</sup> ∈ y ⋆ , then

$$\frac{1}{P(y_i)} = \frac{1}{P^*} = \frac{N^*}{\mu} \leq \frac{k}{\mu}.$$

Thus, for k ≥ N<sup>⋆</sup> , we can bound

$$\mathbb{P}[N_\mu > k] \leq \mathbb{P}[\tau > k] = \mathbb{P}[\mathcal{Y}^* \cap \{y_1, \dots, y_k\} = \emptyset] \leq (1 - |\mathbf{y}^*|P^*)^k,$$

and consequently

$$\begin{aligned} \mathbb{E}[N_\mu] \leq \mathbb{E}[\tau] &\leq \mathbb{E}[\tau \mathbb{I}\{\tau \leq N^*\}] + \mathbb{E}[\tau \mathbb{I}\{\tau > N^*\}] \\ &\leq N^* + \sum_{k > N^*} (1 - |\mathbf{y}^*|P^*)^k \\ &\leq N^* + \frac{1}{|\mathbf{y}^*|P(\mathbf{y}^*)} = \frac{\mu + 1/|\mathbf{y}^*|}{P(\mathbf{y}^*)}. \end{aligned}$$

To prove correctness, observe that N<sup>µ</sup> ≥ N<sup>⋆</sup> , because for all y ∈ Y, <sup>P</sup> (y) <sup>≥</sup> <sup>N</sup><sup>⋆</sup>/µ. Hence, any stopping time τ ≥ N<sup>µ</sup> also satisfies τ ≥ N<sup>⋆</sup> , and moreover has <sup>y</sup>b<sup>τ</sup> <sup>∈</sup> <sup>y</sup> <sup>⋆</sup> whenever y <sup>⋆</sup> ∩ {y1, y2, . . . , y<sup>τ</sup> } ̸= <sup>∅</sup>. This fails to occur with probability no more than

$$\left(1 - \frac{|\mathbf{y}^*|}{P^*}\right)^{N^*} = \left(1 - \frac{|\mathbf{y}^*|}{P^*}\right)^{\mu/P^*} \leq e^{-|\mathbf{y}^*|\mu}.$$

# J PROOFS FROM S[ECTION](#page-7-4) 4.2

# J.1 PROOF OF T[HEOREM](#page-8-4) 4.2

We state and prove a generalized version of [Theorem 4.2.](#page-8-4) In the assumptions below, we fix a parameter γ ∈ [0, 1); the setting γ = 0 corresponds to [Theorem 4.2.](#page-8-4)

Assumption J.1 (Coverage). *All* π ∈ Π *satisfy* C<sup>π</sup> ≤ Cconc *for a parameter* Cconc ≥ (1−γ) <sup>−</sup><sup>1</sup>Ccov,γ*, and* C<sup>π</sup>base/π;<sup>β</sup> ≤ Closs *for a parameter* Closs ≥ |Y|*.*

By [Lemma 4.1](#page-41-2)′ , [Assumption J.1](#page-41-3) is consistent with the assumption that π ⋆ <sup>β</sup> ∈ Π.

Assumption J.2 (Margin). *For all* x ∈ supp(µ)*, the initial model* πbase *satisfies*

$$\pi_{\text{base}}(\mathbf{y}_\gamma^*(x) \mid x) \geq (1 + \gamma_{\text{margin}}) \cdot \pi_{\text{base}}(y \mid x) \quad \forall y \notin \mathbf{y}_\gamma^*(x)$$

*for a parameter* γmargin > 0*.*

Theorem 4.2′ . *Assume that* π ⋆ <sup>β</sup> ∈ Π *[\(Assumption 4.3\)](#page-8-2), and that [Assumption 4.4](#page-8-3) and [Assumption 4.2](#page-7-2) hold with respect to some* γ ∈ [0, 1)*, with parameters* Cconc*,* Closs*, and* γmargin > 0*. For any* δ, ρ ∈ (0, 1)*, the DPO algorithm in [Eq. \(4\)](#page-4-4) ensures that with probability at least* 1 − ρ*,*

$$\mathbb{P}_{x \sim \mu}[\widehat{\pi}(\mathbf{y}_\gamma^*(x) \mid x) \leq 1 - \delta] \lesssim \frac{1}{\gamma_{\text{margin}} \delta} \cdot \tilde{O}\left(\sqrt{\frac{C_{\text{conc}} \log^3(C_{\text{loss}} |\Pi| \rho^{-1})}{n}} + \beta \log(C_{\text{conc}}) + \gamma\right)$$

*where* <sup>O</sup>e(·) *hides factors logarithmic in* <sup>n</sup> *and* <sup>C</sup>conc *and doubly logarithmic in* <sup>Π</sup>*,* <sup>C</sup>loss*, and* <sup>ρ</sup> −1 *.*

We first state and prove some supporting technical lemmas, then proceed to the proof of [Theorem 4.2](#page-41-4)′ .

#### J.1.1 TECHNICAL LEMMAS

Proof of [Lemma 4.1](#page-41-2)′ . For any fixed x ∈ X , we have

$$\begin{aligned}\mathbb{E}_{y \sim \pi_{\beta}^*(\cdot | x)} \left[ \frac{\pi_{\beta}^*(y | x)}{\pi_{\text{base}}(y | x)} \right] &= \mathbb{E}_{y \sim \pi_{\beta}^*(\cdot | x)} \left[ \frac{\pi_{\text{base}}^{1+\beta^{-1}}(y | x)}{\pi_{\text{base}}(y | x)} \right] \cdot \left( \sum_{y' \in \mathcal{Y}} \pi_{\text{base}}^{1+\beta^{-1}}(y' | x) \right)^{-1} \\ &\leq \max_{y \in \mathcal{Y}} \pi_{\text{base}}^{\beta^{-1}}(y | x) \cdot \left( \sum_{y' \in \mathcal{Y}} \pi_{\text{base}}^{1+\beta^{-1}}(y' | x) \right)^{-1} \\ &\leq (1 - \gamma)^{-1} \pi_{\text{base}}^{\beta^{-1}}(\mathbf{y}_{\gamma}^*(x) | x) \cdot \left( \sum_{y' \in \mathcal{Y}} \pi_{\text{base}}^{1+\beta^{-1}}(y' | x) \right)^{-1} \\ &= (1 - \gamma)^{-1} \frac{\pi_{\text{base}}^{1+\beta^{-1}}(\mathbf{y}_{\gamma}^*(x) | x)}{\pi_{\text{base}}(\mathbf{y}_{\gamma}^*(x) | x)} \cdot \left( \sum_{y' \in \mathcal{Y}} \pi_{\text{base}}^{1+\beta^{-1}}(y' | x) \right)^{-1} \\ &= (1 - \gamma)^{-1} \frac{\sum_{y \in \mathbf{y}_{\gamma}^*(x)} \pi_{\text{base}}^{1+\beta^{-1}}(y | x)}{\pi_{\text{base}}(\mathbf{y}_{\gamma}^*(x) | x)} \cdot \left( \sum_{y' \in \mathcal{Y}} \pi_{\text{base}}^{1+\beta^{-1}}(y' | x) \right)^{-1} \\ &\leq (1 - \gamma)^{-1} \frac{1}{\pi_{\text{base}}(\mathbf{y}_{\gamma}^*(x) | x)}.\end{aligned}$$

It follows that C<sup>π</sup> ⋆ ≤ (1 − γ) <sup>−</sup><sup>1</sup>Ccov,γ as claimed.

For the second result, we have

$$\mathcal{C}_{\pi_{\text{base}}/\pi_{\beta}^*; \beta} = \mathbb{E}_{\pi_{\text{base}}} \left[ \frac{1}{\pi_{\text{base}}(y \mid x)} \cdot \left( \sum_{y' \in \mathcal{Y}} \pi_{\text{base}}^{1+\beta-1}(y' \mid x) \right)^{\beta} \right] \leq \mathbb{E}_{\pi_{\text{base}}} \left[ \frac{1}{\pi_{\text{base}}(y \mid x)} \right] = |\mathcal{Y}|.$$

The next lemmas provide bounds on the tails of the self-rewards used in the algorithm.

Lemma J.1. *Suppose* β ∈ [0, 1]*. For any model* π*, with probability at least* 1 − δ *over the draw of* x ∼ µ*,* y, y′ ∼ πbase(· | x)*, we have that for all* s > 0*,*

$$\mathbb{P} \left[ \left| \beta \log \left( \frac{\pi(y | x)}{\pi_{\text{base}}(y | x)} \right) - \beta \log \left( \frac{\pi(y' | x)}{\pi_{\text{base}}(y' | x)} \right) \right| > \log(2\mathcal{C}_{\pi_{\text{base}}/\pi; \beta} + s) \right] \leq \exp(-s).$$

Proof of [Lemma J.1.](#page-42-0) Define

$$X := \left| \beta \log \left( \frac{\pi(y \mid x)}{\pi_{\text{base}}(y \mid x)} \right) - \beta \log \left( \frac{\pi(y' \mid x)}{\pi_{\text{base}}(y' \mid x)} \right) \right|.$$

By the Chernoff method, we have that with probability at least 1 − δ,

$$\begin{aligned}
& X \leq \log(\mathbb{E}[\exp(X)]) + \log(\delta^{-1}) \\
&= \log \left( \mathbb{E}_{x \sim \mu, y, y' \sim \pi_{\text{base}}(x)} \left[ \exp \left( \left| \beta \log \left( \frac{\pi(y | x)}{\pi_{\text{base}}(y | x)} \right) - \beta \log \left( \frac{\pi(y' | x)}{\pi_{\text{base}}(y' | x)} \right) \right| \right) \right] \right) + \log(\delta^{-1}) \\
&\leq \log \left( \mathbb{E}_{x \sim \mu, y, y' \sim \pi_{\text{base}}(x)} \left[ \exp \left( \beta \log \left( \frac{\pi(y | x)}{\pi_{\text{base}}(y | x)} \right) - \beta \log \left( \frac{\pi(y' | x)}{\pi_{\text{base}}(y' | x)} \right) \right) \right] \right. \\
&\quad \left. + \mathbb{E}_{x \sim \mu, y, y' \sim \pi_{\text{base}}(x)} \left[ \exp \left( \beta \log \left( \frac{\pi(y' | x)}{\pi_{\text{base}}(y' | x)} \right) - \beta \log \left( \frac{\pi(y | x)}{\pi_{\text{base}}(y | x)} \right) \right) \right] \right) + \log(\delta^{-1}) \\
&= \log \left( 2 \mathbb{E}_{x \sim \mu, y, y' \sim \pi_{\text{base}}(x)} \left[ \exp \left( \beta \log \left( \frac{\pi(y | x)}{\pi_{\text{base}}(y | x)} \right) - \beta \log \left( \frac{\pi(y' | x)}{\pi_{\text{base}}(y' | x)} \right) \right) \right] \right) + \log(\delta^{-1}) \\
&= \log \left( \mathbb{E}_{x \sim \mu, y, y' \sim \pi_{\text{base}}(x)} \left[ \left( \frac{\pi(y | x)}{\pi_{\text{base}}(y | x)} \cdot \frac{\pi_{\text{base}}(y' | x)}{\pi(y' | x)} \right)^{\beta} \right] \right) + \log(2\delta^{-1}).
\end{aligned}$$

As long as β ≤ 1, by Jensen's inequality, we can bound

$$\begin{aligned} \mathbb{E}_{x \sim \mu, y, y' \sim \pi_{\text{base}}(x)} & \left[ \left( \frac{\pi(y \mid x)}{\pi_{\text{base}}(y \mid x)} \cdot \frac{\pi_{\text{base}}(y' \mid x)}{\pi(y' \mid x)} \right)^{\beta} \right] \\ & \leq \mathbb{E}_{x \sim \mu, y' \sim \pi_{\text{base}}(x)} \left[ \left( \mathbb{E}_{y \sim \pi_{\text{base}}(x)} \left[ \frac{\pi(y \mid x)}{\pi_{\text{base}}(y \mid x)} \right] \cdot \frac{\pi_{\text{base}}(y' \mid x)}{\pi(y' \mid x)} \right)^{\beta} \right] \\ & = \mathbb{E}_{x \sim \mu, y' \sim \pi_{\text{base}}(x)} \left[ \left( \frac{\pi_{\text{base}}(y' \mid x)}{\pi(y' \mid x)} \right)^{\beta} \right] \\ & = \mathcal{C}_{\pi_{\text{base}}/\pi; \beta}, \end{aligned}$$

which proves the result.

Lemma J.2. *Let* β ∈ [0, 1]*. For all models* π*, we have*

$$\mathbb{E}_{x \sim \mu, y, y' \sim \pi_{\text{base}}(\cdot | x)} \left[ \left| \beta \log \left( \frac{\pi(y | x)}{\pi_{\text{base}}(y | x)} \right) - \beta \log \left( \frac{\pi(y' | x)}{\pi_{\text{base}}(y' | x)} \right) \right|^4 \right] \leq O(\log^4(\mathcal{C}_{\pi_{\text{base}}/\pi; \beta}) + 1).$$

Proof of [Lemma J.2.](#page-43-0) Define

$$X := \left| \beta \log \left( \frac{\pi(y \mid x)}{\pi_{\text{base}}(y \mid x)} \right) - \beta \log \left( \frac{\pi(y' \mid x)}{\pi_{\text{base}}(y' \mid x)} \right) \right|.$$

Set k = log(2C<sup>π</sup>base/π;<sup>β</sup>). We can bound

$$\begin{aligned}\mathbb{E}[X^4] &= \mathbb{E}\left[\int_0^\infty \mathbb{I}\{X^4 > t\} dt\right] \\ &= 4 \mathbb{E}\left[\int_0^\infty \mathbb{I}\{X > t\} t^3 dt\right] \\ &= 4 \int_0^\infty \mathbb{P}[X > t] t^3 dt \\ &\leq k^4 + 4 \int_k^\infty \mathbb{P}[X > t] t^3 dt \\ &\leq k^4 + 4 \int_k^\infty e^{k-t} t^3 dt \\ &= k^4 + 4(k^3 + 3k^2 + 6k + 6) \\ &= O(k^4 + 1),\end{aligned}$$

where the third-to-last line uses [Lemma J.1.](#page-42-0)

# J.1.2 PROOF OF T[HEOREM](#page-41-4) 4.2′

Proof of [Theorem 4.2](#page-41-4)′ . For any model <sup>π</sup> <sup>∈</sup> <sup>Π</sup>, define <sup>J</sup>(π) := <sup>E</sup>π[log <sup>π</sup>base(<sup>y</sup> | <sup>x</sup>)]. Let <sup>π</sup>b <sup>∈</sup> <sup>Π</sup> denote the model returned by the DPO algorithm in [Eq. \(8\).](#page-7-1) Let <sup>E</sup>π,π′ [·] denote shorthand for <sup>E</sup>x∼µ,y∼π(x),y′∼π′(x) [·], and for any r : X × Y → <sup>R</sup> define ∆<sup>r</sup> (x, y, y′ ) := r(x, y) − r(x, y′ ). Define

$$r^*(x, y) := \log \pi_{\text{base}}(y \mid x) = \beta \log \left( \frac{\pi_\beta^*(y \mid x)}{\pi_{\text{base}}(y \mid x)} \right) + Z(x),$$

and let <sup>r</sup>b(x, y) := <sup>β</sup> log <sup>π</sup>b(y|x) πbase(y|x) . By a standard argument [\(Huang et al.,](#page-12-10) [2024\)](#page-12-10), we have

$$\hat{\pi} \in \arg \max_{\pi: \mathcal{X} \rightarrow \Delta(\mathcal{Y})} \mathbb{E}_{\pi} [\hat{r}(x, y)] - \beta D_{\text{KL}}(\pi \parallel \pi_{\text{base}}). \quad (20)$$

Therefore for any comparator model π ⋆ : X → ∆(Y) (not necessarily in the model class Π), we have

$$\begin{aligned}
J(\pi^*) - J(\hat{\pi}) &= \mathbb{E}_{\pi^*}[r^*(x, y)] - \mathbb{E}_{\hat{\pi}}[r^*(x, y)] \\
&= \mathbb{E}_{\pi^*}[\hat{r}(x, y)] - \beta D_{\text{KL}}(\pi^* \parallel \pi_{\text{base}}) - \mathbb{E}_{\hat{\pi}}[\hat{r}(x, y)] + \beta D_{\text{KL}}(\hat{\pi} \parallel \pi_{\text{base}}) \\
&\quad + \mathbb{E}_{\pi^*}[r^*(x, y) - \hat{r}(x, y)] + \beta D_{\text{KL}}(\pi^* \parallel \pi_{\text{base}}) + \mathbb{E}_{\hat{\pi}}[\hat{r}(x, y) - r^*(x, y)] - \beta D_{\text{KL}}(\hat{\pi} \parallel \pi_{\text{base}}) \\
&\leq \mathbb{E}_{\pi^*}[r^*(x, y) - \hat{r}(x, y)] + \beta D_{\text{KL}}(\pi^* \parallel \pi_{\text{base}}) + \mathbb{E}_{\hat{\pi}}[\hat{r}(x, y) - r^*(x, y)] - \beta D_{\text{KL}}(\hat{\pi} \parallel \pi_{\text{base}}) \\
&= \mathbb{E}_{\pi^*, \pi_{\text{base}}} \left[ \Delta^{r^*}(x, y, y') - \Delta^{\hat{r}}(x, y, y') \right] + \mathbb{E}_{\hat{\pi}, \pi_{\text{base}}} \left[ \Delta^{\hat{r}}(x, y, y') - \Delta^{r^*}(x, y, y') \right] \\
&\quad + \beta D_{\text{KL}}(\pi^* \parallel \pi_{\text{base}}) - \beta D_{\text{KL}}(\hat{\pi} \parallel \pi_{\text{base}})
\end{aligned} \tag{21}$$

where the inequality uses [Eq. \(20\).](#page-41-5) To bound the right-hand side above, we will use the following lemma, which is proven in the sequel.

Lemma J.3. *For any model* π *and any* η > 0*, we have that*

$$\begin{aligned} & \mathbb{E}_{\pi, \pi_{\text{base}}} \left[ \left| \Delta^{r^*}(x, y, y') - \Delta^{\widehat{r}}(x, y, y') \right| \right] \\ & \lesssim \mathcal{C}_{\pi}^{1/2} \cdot \left( \mathbb{E}_{\pi_{\text{base}}, \pi_{\text{base}}} \left[ \left| \Delta^{r^*}(x, y, y') - \Delta^{\widehat{r}}(x, y, y') \right|^2 \mathbb{I} \left\{ |\Delta^{r^*}| \leq \eta, |\Delta^{\widehat{r}}| \leq \eta \right\} \right] \right)^{1/2} \\ & \quad + \mathcal{C}_{\pi}^{1/2} (\log(\mathcal{C}_{\pi_{\text{base}}/\widehat{\pi}; \beta}) + \log(\mathcal{C}_{\pi_{\text{base}}/\pi_{\beta}^*; \beta})) \cdot \left( \mathbb{P}_{\pi_{\text{base}}, \pi_{\text{base}}} \left[ |\Delta^{r^*}| > \eta \right] + \mathbb{P}_{\pi_{\text{base}}, \pi_{\text{base}}} \left[ |\Delta^{\widehat{r}}| > \eta \right] \right)^{1/4}. \end{aligned}$$

Using [Lemma J.3](#page-44-0) to bound the first two terms of [Eq. \(21\),](#page-44-1) and using the fact that all π ∈ Π have C<sup>π</sup> ≤ Cconc and C<sup>π</sup>base/π;<sup>β</sup> ≤ Closs, we have that

$$\begin{aligned}
& J(\pi^*) - J(\hat{\pi}) \\
& \lesssim (C_{\pi^*} + C_{\text{conc}})^{1/2} \cdot \left( \mathbb{E}_{\pi_{\text{base}}, \pi_{\text{base}}} \left[ \left| \Delta^{r^*}(x, y, y') - \Delta^{\hat{r}}(x, y, y') \right|^2 \mathbb{I}\{|\Delta^{r^*}| \leq \eta, |\Delta^{\hat{r}}| \leq \eta\} \right] \right)^{1/2} \\
& + (C_{\pi^*} + C_{\text{conc}})^{1/2} \log(C_{\text{loss}}) \cdot \left( \mathbb{P}_{\pi_{\text{base}}, \pi_{\text{base}}} \left[ |\Delta^{r^*}| > \eta \right] + \mathbb{P}_{\pi_{\text{base}}, \pi_{\text{base}}} \left[ |\Delta^{\hat{r}}| > \eta \right] \right)^{1/4} + \beta D_{\text{KL}}(\pi^* \parallel \pi_{\text{base}}).
\end{aligned} \tag{22}$$

Let us overload notation and write ∆<sup>π</sup> (x, y, y′ ) = <sup>β</sup> log π(y|x) πbase(y|x) <sup>−</sup> <sup>β</sup> log π(y |x) πbase(y′ |x) , so that ∆<sup>π</sup>b = ∆<sup>r</sup>b and ∆<sup>π</sup> ⋆ <sup>β</sup> = ∆<sup>r</sup> ⋆ . Since π ⋆ <sup>β</sup> <sup>∈</sup> <sup>Π</sup>, the definition of <sup>π</sup>b in [Eq. \(4\)](#page-4-4) implies that

$$\begin{aligned} \sum_{(x,y,y') \in \mathcal{D}_{\text{pref}}} \left( \Delta^{\widehat{\pi}}(x, y, y') - \Delta^{\pi_{\beta}^*}(x, y, y') \right)^2 &\leq \min_{\pi \in \Pi} \sum_{(x,y,y') \in \mathcal{D}_{\text{pref}}} \left( \Delta^{\pi}(x, y, y') - \Delta^{\pi_{\beta}^*}(x, y, y') \right)^2 \\ &\leq \sum_{(x,y,y') \in \mathcal{D}_{\text{pref}}} \left( \Delta^{\pi_{\beta}^*}(x, y, y') - \Delta^{\pi_{\beta}^*}(x, y, y') \right)^2 \\ &= 0. \end{aligned}$$

Define Bn,ρ := log(2nCloss|Π|ρ −1 ). It is immediate that

$$\sum_{(x,y,y') \in \mathcal{D}_{\text{pref}}} \left( \Delta^{\widehat{\pi}}(x,y,y') - \Delta^{\pi_{\beta}^*}(x,y,y') \right)^2 \mathbb{I} \left\{ |\Delta^{\widehat{\pi}}| \leq B_{n,\rho}, |\Delta^{\pi_{\beta}^*}| \leq B_{n,\rho} \right\} \leq 0.$$

From here, Bernstein's inequality and a union bound implies that with probability at least 1 − ρ,

$$\begin{aligned}\mathbb{E}_{\pi_{\text{base}}, \pi_{\text{base}}} \left[ \left| \Delta^{\widehat{\pi}}(x, y, y') - \Delta^{\pi_{\beta}^*}(x, y, y') \right|^2 \mathbb{I} \left\{ |\Delta^{\widehat{\pi}}| \leq B_{n,\rho}, |\Delta^{\pi_{\beta}^*}| \leq B_{n,\rho} \right\} \right] \\ \lesssim \frac{B_{n,\rho}^2 \log(|\Pi|\rho^{-1})}{n} =: \varepsilon_{\text{stat}}^2.\end{aligned}$$

$$J(\pi^*) - J(\hat{\pi}) \lesssim (\mathcal{C}_{\pi^*} + C_{\text{conc}})^{1/2} \cdot \varepsilon_{\text{stat}} + (\mathcal{C}_{\pi^*} + C_{\text{conc}})^{1/2} \log(C_{\text{loss}}) \cdot \rho^{1/4} + \beta D_{\text{KL}}(\pi^* \parallel \pi_{\text{base}}).$$

Note that the above bound holds for any π ⋆ : X → ∆(Y). We define π <sup>⋆</sup> by

$$\pi^*(y \mid x) := \frac{\pi_{\text{base}}(y \mid x) \mathbb{I}[y \in \mathbf{y}_\gamma^*(x)]}{\pi_{\text{base}}(\mathbf{y}_\gamma^*(x) \mid x)},$$

which can be seen to satisfy Cπ<sup>⋆</sup> ≤ Ccov,γ ≤ Cconc and DKL(π <sup>⋆</sup> ∥ πbase) ≤ log(Cπ<sup>⋆</sup> ) ≤ log(Cconc). With this choice, we can further bound the expression above by

$$J(\pi^\star) - J(\hat{\pi}) \lesssim (C_{\text{conc}})^{1/2} \cdot \varepsilon_{\text{stat}} + (C_{\text{conc}})^{1/2} \log(C_{\text{loss}}) \cdot \rho^{1/4} + \beta \log(C_{\text{conc}})$$

Given a desired failure probability ρ, applying the bound above with ρ ′ := ρ ∧ (εstat/ log(Closs))<sup>4</sup> then gives

$$J(\pi^*) - J(\hat{\pi}) \lesssim (C_{\text{conc}})^{1/2} \cdot \varepsilon_{\text{stat}} + \beta \log(C_{\text{conc}}).$$

Finally, we observe that for our choice of π ⋆ , under the margin condition with parameter γ, we have

$$\begin{aligned} J(\pi^*) - J(\hat{\pi}) &= \mathbb{E}_{x \sim \mu} \mathbb{E}_{y, y' \sim \pi^*, \hat{\pi}} \left[ \log \left( \frac{\pi_{\text{base}}(y \mid x)}{\pi_{\text{base}}(y' \mid x)} \right) \right] \\ &\gtrsim \gamma_{\text{margin}} \cdot \mathbb{E}_{x \sim \mu} \mathbb{E}_{y' \sim \hat{\pi}} [\mathbb{I}\{y' \notin \mathbf{y}_\gamma^*(x)\}] - \gamma \\ &\gtrsim \gamma_{\text{margin}} \delta \cdot \mathbb{E}_{x \sim \mu} [\mathbb{I}\{\hat{\pi}(\mathbf{y}_\gamma^*(x) \mid x) \leq 1 - \delta\}] - \gamma \end{aligned}$$

where the first inequality uses [Assumption J.2](#page-41-6) together with the fact that y ∈ y ⋆ γ (x) with probability 1 over x ∼ µ and y ∼ π ⋆ (· | x). This proves the result.

Proof of [Lemma J.3.](#page-44-0) For any η > 0, we can bound

$$\mathbb{E}_{\pi, \pi_{\text{base}}} \left[ \left| \Delta^{r^*}(x, y, y') - \Delta^{\widehat{r}}(x, y, y') \right| \right] \leq \mathbb{E}_{\pi, \pi_{\text{base}}} \left[ \left| \Delta^{r^*}(x, y, y') - \Delta^{\widehat{r}}(x, y, y') \right| \mathbb{I} \left\{ |\Delta^{r^*}| \leq \eta, |\Delta^{\widehat{r}}| \leq \eta \right\} \right] \\ + \mathbb{E}_{\pi, \pi_{\text{base}}} \left[ \left| \Delta^{r^*}(x, y, y') - \Delta^{\widehat{r}}(x, y, y') \right| \mathbb{I} \left\{ |\Delta^{r^*}| > \eta \vee |\Delta^{\widehat{r}}| > \eta \right\} \right].$$

For the second term above, we can use Cauchy-Schwarz to bound

$$\begin{aligned} & \mathbb{E}_{\pi, \pi_{\text{base}}} \left[ \left| \Delta^{r^*}(x, y, y') - \Delta^{\widehat{r}}(x, y, y') \right| \mathbb{I} \left\{ |\Delta^{r^*}| > \eta \vee |\Delta^{\widehat{r}}| > \eta \right\} \right] \\ & \leq \mathcal{C}_{\pi}^{1/2} \cdot \left( \mathbb{E}_{\pi_{\text{base}}, \pi_{\text{base}}} \left[ \left| \Delta^{r^*}(x, y, y') - \Delta^{\widehat{r}}(x, y, y') \right|^2 \mathbb{I} \left\{ |\Delta^{r^*}| > \eta \vee |\Delta^{\widehat{r}}| > \eta \right\} \right] \right)^{1/2} \\ & \lesssim \mathcal{C}_{\pi}^{1/2} \cdot \left( \mathbb{P}_{\pi_{\text{base}}, \pi_{\text{base}}} \left[ |\Delta^{r^*}| > \eta \right] + \mathbb{P}_{\pi_{\text{base}}, \pi_{\text{base}}} \left[ |\Delta^{\widehat{r}}| > \eta \right] \right)^{1/4} \\ & \quad \cdot \left( \mathbb{E}_{\pi_{\text{base}}, \pi_{\text{base}}} \left[ \left| \Delta^{r^*}(x, y, y') \right|^4 \right] + \mathbb{E}_{\pi_{\text{base}}, \pi_{\text{base}}} \left[ \left| \Delta^{\widehat{r}}(x, y, y') \right|^4 \right] \right)^{1/4} \\ & \lesssim \mathcal{C}_{\pi}^{1/2} \cdot \left( \mathbb{P}_{\pi_{\text{base}}, \pi_{\text{base}}} \left[ |\Delta^{r^*}| > \eta \right] + \mathbb{P}_{\pi_{\text{base}}, \pi_{\text{base}}} \left[ |\Delta^{\widehat{r}}| > \eta \right] \right)^{1/4} \cdot (\log(\mathcal{C}_{\pi_{\text{base}}/\widehat{r}}; \beta) + \log(\mathcal{C}_{\pi_{\text{base}}/\pi_{\beta}^*; \beta})), \end{aligned}$$

where the last inequality follows from [Lemma J.2.](#page-43-0)

Meanwhile, for the first term, for any λ > 0 we can bound

$$\begin{aligned} & \mathbb{E}_{\pi, \pi_{\text{base}}} \left[ \left| \Delta^{r^*}(x, y, y') - \Delta^{\widehat{r}}(x, y, y') \right| \mathbb{1} \left\{ |\Delta^{r^*}| \leq \eta, |\Delta^{\widehat{r}}| \leq \eta \right\} \right] \\ & \leq C_\pi^{1/2} \left( \mathbb{E}_{\pi_{\text{base}}, \pi_{\text{base}}} \left[ \left| \Delta^{r^*}(x, y, y') - \Delta^{\widehat{r}}(x, y, y') \right|^2 \mathbb{1} \left\{ |\Delta^{r^*}| \leq \eta, |\Delta^{\widehat{r}}| \leq \eta \right\} \right] \right)^{1/2}. \end{aligned}$$

# J.2 PROOF OF T[HEOREM](#page-9-2) 4.3 AND T[HEOREM](#page-50-0) J.3

In this section we prove [Theorem 4.3](#page-9-2) as well as [Theorem J.3,](#page-50-0) the application to linear softmax models. For the formal theorem statements, see [Theorem J.2](#page-49-0) and [Theorem J.3](#page-50-0) respectively. The section is organized as follows.

- [Appendix J.2.1](#page-46-1) gives necessary background on KL-regularized policy optimization, as well as the Sequential Extrapolation Coefficient.
- [Appendix J.2.2](#page-47-1) presents a generic guarantee for XPO under a general choice of reward function.
- [Appendix J.2.3](#page-49-1) instantiates the result above with the self-reward function r(x, y) := log πbase(y | x) to prove [Theorem 4.3.](#page-9-2)
- Finally, [Appendix J.2.4](#page-50-1) applies the preceding results to prove [Theorem J.3.](#page-50-0)

# J.2.1 BACKGROUND

To begin, we give background on KL-regularized policy optimization and the Sequential Extrapolation Coefficient.

KL-regularized policy optimization. Let β > 0 be given, and let r : X × Y → [−Rmax, Rmax] be an unknown reward function on prompt/action pairs. Define a value function J<sup>β</sup> over model class Π by:

$$J_{\beta}(\pi) := \mathbb{E}_{\pi}[r(x, y)] - \beta \cdot D_{\text{KL}}(\mathbb{P}^{\pi} \parallel \mathbb{P}^{\pi_{\text{base}}}).$$

We refer to this as a *KL-regularized policy optimization* objective (we use the term "policy" following the reinforcement learning literature; for our setting, policies correspond to models). Given query access to <sup>r</sup>, the goal is to find <sup>π</sup>b <sup>∈</sup> <sup>Π</sup> such that

$$J_{\beta}(\pi_{\beta}^{\star}) - J_{\beta}(\hat{\pi}) \leq \epsilon$$

where π ⋆ β (y | x) ∝ πbase(y | x) exp(β −1 r(x, y)) is the model that maximizes J<sup>β</sup> over all models π : X → ∆(Y).

We make use of the following assumptions, as in [Xie et al.](#page-15-5) [\(2024\)](#page-15-5).

Assumption J.3 (Realizability). *It holds that* π ⋆ <sup>β</sup> ∈ Π*.*

Assumption J.4 (Bounded density ratios). *For all* π ∈ Π*,* (x, y) ∈ X × Y*,* <sup>β</sup> log <sup>π</sup>(y|x) πbase(y|x)  ≤ Vmax*.*

Finally, we require two definitions.

Definition J.1 (Sequential Extrapolation Coefficient for RLHF, [\(Xie et al.,](#page-15-5) [2024\)](#page-15-5)). *For a model class* Π*, reward function* r*, reference model* πbase*, and parameters* T ∈ <sup>N</sup> *and* β, λ > 0*, the Sequential Extrapolation Coefficient is defined as*

$$\begin{aligned} & \text{SEC}(\Pi, r, T, \beta, \lambda; \pi_{\text{base}}) \\ & := \sup_{\pi^{(1)}, \dots, \pi^{(T)} \in \Pi} \left\{ \sum_{t=1}^T \frac{\mathbb{E}^{(t)} \left[ \beta \log \frac{\pi^{(t)}(y|x)}{\pi_{\text{base}}(y|x)} - r(x, y) - \beta \log \frac{\pi^{(t)}(y'|x)}{\pi_{\text{base}}(y'|x)} + r(x, y') \right]^2}{\lambda \vee \sum_{i=1}^{t-1} \mathbb{E}^{(i)} \left[ \left( \beta \log \frac{\pi^{(t)}(y|x)}{\pi_{\text{base}}(y|x)} - r(x, y) - \beta \log \frac{\pi^{(t)}(y'|x)}{\pi_{\text{base}}(y'|x)} + r(x, y') \right)^2 \right]} \right\} \end{aligned}$$

*where* E (t) *denotes expectation over* x ∼ µ*,* y ∼ π (t) (· | x)*, and* y ′ ∼ πbase(· | x)*.*

Definition J.2. *Let* ϵ > 0*. We say that* Ψ ⊆ Π *is a* ϵ*-net for model class* Π *if for every* π ∈ Π *there exists* π ′ ∈ Ψ *such that*

$$\max_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}} \left| \log \frac{\pi(y \mid x)}{\pi'(y \mid x)} \right| \leq \epsilon.$$

Algorithm 1 Reward-based variant of Exploratory Preference Optimization [\(Xie et al.,](#page-15-5) [2024\)](#page-15-5)

input: Base model πbase : X → ∆(Y), reward function r : X × Y → <sup>R</sup>, number of iterations T ∈ N, KL regularization coefficient β > 0, optimism coefficient α > 0.

Initialize: π (1) ← πbase, D(0) ← <sup>∅</sup>.

for iteration t = 1, . . . , T do

Generate sample: (x (t) , y(t) , ye (t) ) via x (t) ∼ µ, y (t) ∼ π (t) (· | x (t) ), <sup>y</sup>e (t) ∼ πbase(· | x (t) ).

Update dataset: D(t) ← D(t−1) ∪ {(x (t) , y(t) , ye (t) )}.

Model optimization with global optimism:

$$\begin{aligned} \pi^{(t+1)} &\leftarrow \arg \min_{\pi \in \Pi} \left\{ \alpha \sum_{(x,y,y') \in \mathcal{D}^{(t)}} \log(\pi(y' | x)) \right. \\ &\quad \left. - \sum_{(x,y,y') \in \mathcal{D}^{(t)}} \left( \beta \log \frac{\pi(y | x)}{\pi_{\text{base}}(y | x)} - \beta \log \frac{\pi(y' | x)}{\pi_{\text{base}}(y' | x)} - (r(x,y) - r(x,y')) \right)^2 \right\}. \end{aligned}$$

return: <sup>π</sup>b <sup>←</sup> arg maxt∈[<sup>T</sup> +1] <sup>J</sup>β(<sup>π</sup>

(t)

). ▷ Can estimate Jβ(π

(t)

) using validation data.

#### J.2.2 GUARANTEES FOR KL-REGULARIZED POLICY OPTIMIZATION WITH XPO

In this section, we give self-contained guarantees for the XPO algorithm [\(Algorithm 1\)](#page-47-0). XPO was introduced in [Xie et al.](#page-15-5) [\(2024\)](#page-15-5) for KL-regularized policy optimization in the related setting where the learner only has indirect access to the reward function r through *preference data* (specifically, pairs of actions labeled via a Bradley-Terry model). Standard offline algorithms for this problem, such as DPO, require bounds on concentrability of the model class (see e.g. [Eq. \(9\)\)](#page-8-0). [Xie et al.](#page-15-5) [\(2024\)](#page-15-5) show that the XPO algorithm avoids this dependence, and instead requires bounded Sequential Extrapolation Coefficient.

[Algorithm 1](#page-47-0) is a variant of the XPO algorithm which is adapted to reward-based feedback (as opposed to preference-based feedback), and [Theorem J.1](#page-47-2) shows that this algorithm enjoys guarantees similar to those of [Xie et al.](#page-15-5) [\(2024\)](#page-15-5) for this setting. Note that this is not an immediate corollary of the results in [Xie et al.](#page-15-5) [\(2024\)](#page-15-5), since the sample complexity in the preference-based setting scales with e O(Rmax) , and for our application to sharpening it is important to avoid this dependence. However, our algorithm and analysis only diverge from [Xie et al.](#page-15-5) [\(2024\)](#page-15-5) in a few places.

Theorem J.1 (Variant of [Xie et al.](#page-15-5) [\(2024,](#page-15-5) Theorem 3.1)). *Suppose that [Assumptions J.3](#page-46-2) and [J.4](#page-46-3) hold. For any* T ∈ <sup>N</sup>*,* ϵdisc, ρ ∈ (0, 1)*, by setting* α := β <sup>R</sup>max+Vmax qlog(2N(Π,ϵdisc)T /ρ) SEC(Π)T *, [Algorithm 1](#page-47-0) produces a model* <sup>π</sup>b <sup>∈</sup> <sup>Π</sup> *such that with probability at least* <sup>1</sup> <sup>−</sup> <sup>ρ</sup>*,*

$$\beta D_{\text{KL}}(\hat{\pi} \parallel \pi_\beta^*) = J_\beta(\pi_\beta^*) - J_\beta(\hat{\pi}) \lesssim (R_{\max} + V_{\max}) \sqrt{\frac{\text{SEC}(\Pi) \log(2\mathcal{N}(\Pi, \epsilon_{\text{disc}}) T/\rho)}{T}} \\ + \beta \epsilon_{\text{disc}} \sqrt{\text{SEC}(\Pi) T}$$

*where* SEC(Π) := SEC(Π, r, T, β, V <sup>2</sup> max; πbase)*.*

Proof of [Theorem J.1.](#page-47-2) For compactness, we abbreviate SEC(Π) := SEC(Π, r, T, β, V <sup>2</sup> max; πbase). From Equation (37) of [Xie et al.](#page-15-5) [\(2024\)](#page-15-5), we have

$$\begin{aligned} & \frac{1}{T} \sum_{t=1}^T J_{\beta}(\pi_{\beta}^{\star}) - J_{\beta}(\pi^{(t)}) \\ & \lesssim \frac{\alpha}{\beta} (R_{\max} + V_{\max})^2 \cdot \text{SEC}(\Pi) + \frac{\beta}{\alpha T} + \frac{V_{\max}}{T} + \frac{1}{T} \sum_{t=2}^T \mathbb{E}_{(x,y) \sim \pi_{\text{base}}} [\beta \log \pi^{(t)}(y \mid x) - \beta \log \pi_{\beta}^{\star}(y \mid x)] \\ & + \frac{\beta}{\alpha (R_{\max} + V_{\max})^2 T} \sum_{t=2}^T \mathbb{E}_{\substack{x \sim \mu \\ y, y' \sim \pi^{(t)} \mid x}} \left[ \left( \beta \log \frac{\pi^{(t)}(y \mid x)}{\pi_{\text{base}}(y \mid x)} - r(x, y) - \beta \log \frac{\pi^{(t)}(y' \mid x)}{\pi_{\text{base}}(y' \mid x)} + r(x, y') \right)^2 \right] \end{aligned}$$

where π (t) := <sup>1</sup> t−1 P i<t π (i) ⊗ πbase denotes the model that, given x ∈ X , samples i ∼ Unif([t − 1]) and then samples y ∼ π (i) (· | x) and y ′ ∼ πbase(· | x). For any 2 ≤ t ≤ T, define L (t) : Π → [0, ∞) by

$$L^{(t)}(\pi) := \mathbb{E}_{(x,y) \sim \pi_{\text{base}}} [\beta \log \pi(y \mid x) - \beta \log \pi_{\beta}^*(y \mid x)] + \frac{\beta}{\alpha(V_{\max} + R_{\max})^2} \frac{\mathbb{E}_{x \sim \mu}}{y, y' \sim \pi^{(t)}|_x} \left[ \left( \beta \log \frac{\pi(y \mid x)}{\pi_{\text{base}}(y \mid x)} - r(x, y) - \beta \log \frac{\pi(y' \mid x)}{\pi_{\text{base}}(y' \mid x)} + r(x, y') \right)^2 \right].$$

Similarly, define

$$\begin{aligned} \hat{L}^{(t)}(\pi) := & \sum_{(x,y,y') \in \mathcal{D}^{(t)}} [\beta \log \pi(y' \mid x) - \beta \log \pi_{\beta}^{\star}(y' \mid x)] \\ & + \frac{\beta}{\alpha(V_{\max} + R_{\max})^2} \sum_{(x,y,y') \in \mathcal{D}^{(t)}} \left[ \left( \beta \log \frac{\pi(y \mid x)}{\pi_{\text{base}}(y \mid x)} - r(x, y) - \beta \log \frac{\pi(y' \mid x)}{\pi_{\text{base}}(y' \mid x)} + r(x, y') \right)^2 \right] \end{aligned}$$

where D(t) is the dataset defined in iteration t of [Algorithm 1.](#page-47-0) By [Assumption J.3](#page-46-2) we have π ⋆ <sup>β</sup> ∈ Π, so infπ∈<sup>Π</sup> <sup>L</sup>b(t) (π) ≤ 0. Moreover by definition, π (t) <sup>∈</sup> arg minπ∈<sup>Π</sup> <sup>L</sup>b(t) .

Let Ψ be an ϵdisc-net over Π, of size N (Π, ϵdisc). Fix any π ∈ Ψ and 2 ≤ t ≤ T, and define increments X<sup>i</sup> := <sup>L</sup>b(i) (π) − <sup>L</sup>b(i−1)(π) for <sup>2</sup> ≤ <sup>i</sup> ≤ <sup>t</sup>, with the notation <sup>L</sup>b(1)(π) := 0 so that <sup>L</sup>b(t) (π) = P<sup>t</sup> <sup>i</sup>=2 X<sup>i</sup> . Let F<sup>i</sup> be the filtration induced by D(i) and define γ<sup>i</sup> := <sup>E</sup>[X<sup>i</sup> | Fi−1]. Observe that (t − 1)L (t) (π) = P<sup>t</sup> <sup>i</sup>=2 γ<sup>i</sup> . For any i, note that we can write X<sup>i</sup> = Y<sup>i</sup> + Z<sup>i</sup> where Y<sup>i</sup> ∈ [−Vmax, Vmax] and Z<sup>i</sup> ∈ [0, β/α]. By [Corollary F.1,](#page-34-0) it holds with probability at least 1 − ρ/(2|Π|T)

$$\sum_{i=2}^t \mathbb{E}[Z_i \mid \mathcal{F}_{i-1}] \lesssim \frac{\beta}{\alpha} \log(2|\Psi|T/\rho) + \sum_{i=2}^t Z_i.$$

By Azuma-Hoeffding, it holds with probability at least 1 − ρ/(2|Π|T) that

$$\sum_{i=2}^t \mathbb{E}[Y_i \mid \mathcal{F}_{i-1}] \lesssim V_{\max} \sqrt{T \log(2|\Psi|T/\rho)} + \sum_{i=2}^t Y_i.$$

Hence, with probability at least 1 − ρ/(|Ψ|T) we have

$$(t-1)L^{(t)}(\pi) \lesssim \frac{\beta}{\alpha} \log(2|\Psi|T/\rho) + V_{\max} \sqrt{T \log(2|\Psi|T/\rho)} + \hat{L}^{(t)}(\pi).$$

With probability at least 1 − ρ this bound holds for all π ∈ Ψ and 2 ≤ t ≤ T. Henceforth condition on this event. Fix any π ∈ Π and 2 ≤ t ≤ T. Since Ψ is an ϵ-net for Π, we see by definition of L (t) that there is some π ′ ∈ Ψ such that

$$|L^{(t)}(\pi) - L^{(t)}(\pi')| \lesssim \beta \epsilon_{\text{disc}} + \frac{\beta}{\alpha(V_{\text{max}} + R_{\text{max}})^2} \cdot \beta \epsilon_{\text{disc}} (V_{\text{max}} + R_{\text{max}}) \leq \beta \epsilon_{\text{disc}} \left( 1 + \frac{\beta}{\alpha(V_{\text{max}} + R_{\text{max}})} \right)$$

and similarly

$$|\widehat{L}^{(t)}(\pi) - \widehat{L}^{(t)}(\pi')| \lesssim (t-1)\beta\epsilon_{\text{disc}} \left( 1 + \frac{\beta}{\alpha(V_{\text{max}} + R_{\text{max}})} \right).$$

It follows that, for all <sup>2</sup> ≤ <sup>t</sup> ≤ <sup>T</sup>, since <sup>L</sup>b(t) (π (t) ) ≤ 0, we get

$$(t-1)L^{(t)}(\pi^{(t)}) \lesssim \frac{\beta}{\alpha} \log(2|\Psi|T/\rho) + V_{\max} \sqrt{T \log(2|\Psi|T/\rho)} + \beta \epsilon_{\text{disc}} T \left( 1 + \frac{\beta}{\alpha(V_{\max} + R_{\max})} \right).$$

Hence,

$$\begin{aligned} & \frac{1}{T} \sum_{t=1}^T J_{\beta}(\pi_{\beta}^{\star}) - J_{\beta}(\pi^{(t)}) \\ & \lesssim \frac{\alpha}{\beta} (R_{\max} + V_{\max})^2 \cdot \text{SEC}(\Pi) + \frac{\beta}{\alpha T} + \frac{V_{\max}}{T} + \frac{1}{T} \sum_{t=2}^T L^{(t)}(\pi^{(t)}) \\ & \lesssim (R_{\max} + V_{\max}) \sqrt{\frac{\text{SEC}(\Pi) \log(2|\Psi|T/\rho)}{T}} + \beta \epsilon_{\text{disc}} \sqrt{\text{SEC}(\Pi)} T \end{aligned}$$

by taking

$$\alpha := \frac{\beta}{R_{\max} + V_{\max}} \sqrt{\frac{\log(2|\Psi|T/\rho)}{\text{SEC}(\Pi)T}}.$$

Since the output <sup>π</sup>b of [Algorithm 1](#page-47-0) satisfies <sup>π</sup>b <sup>∈</sup> arg maxt∈[T] <sup>J</sup>β(<sup>π</sup> (t) ), the claimed bound on Jβ(π ⋆ β ) <sup>−</sup> <sup>J</sup>β(πb) is immediate. Finally, observe that by definition of <sup>π</sup> ⋆ β ,

$$\begin{aligned} J_\beta(\pi_\beta^*) - J_\beta(\hat{\pi}) &= \mathbb{E}_{(x,y) \sim \pi_\beta^*} \left[ r(x, y) - \beta \log \frac{\pi_\beta^*(y \mid x)}{\pi_{\text{base}}(y \mid x)} \right] - \mathbb{E}_{(x,y) \sim \hat{\pi}} \left[ r(x, y) - \beta \log \frac{\hat{\pi}(y \mid x)}{\pi_{\text{base}}(y \mid x)} \right] \\ &= \mathbb{E}_{(x,y) \sim \pi_\beta^*} \left[ r(x, y) - \beta \log \frac{\pi_\beta^*(y \mid x)}{\pi_{\text{base}}(y \mid x)} \right] - \mathbb{E}_{(x,y) \sim \hat{\pi}} \left[ r(x, y) - \beta \log \frac{\pi_\beta^*(y \mid x)}{\pi_{\text{base}}(y \mid x)} \right] \\ &\quad + \mathbb{E}_{(x,y) \sim \hat{\pi}} \left[ \beta \log \frac{\hat{\pi}(y \mid x)}{\pi_\beta^*(y \mid x)} \right] \\ &= \beta \log_{(x,y) \sim \pi_{\text{base}}} [\exp(r(x, y))] - \beta \log_{(x,y) \sim \pi_{\text{base}}} [\exp(r(x, y))] + \beta D_{\text{KL}}(\hat{\pi} \parallel \pi_\beta^*) \\ &= \beta D_{\text{KL}}(\hat{\pi} \parallel \pi_\beta^*). \end{aligned}$$

This completes the proof.

## J.2.3 APPLYING XPO TO MAXIMUM-LIKELIHOOD SHARPENING

We now prove [Theorem J.2,](#page-49-0) the formal statement of [Theorem 4.3,](#page-9-2) which applies XPO to maximum-likelihood sharpening. This result is a straightforward corollary of [Theorem J.1](#page-47-2) with the reward function rself(x, y) := log πbase(y | x), together with the observation that low KLregularized regret implies sharpness (under [Assumption 4.2\)](#page-7-2).

Theorem J.2 (Sharpening via active exploration). *There are absolute constants* c[J](#page-49-0).2, C[J](#page-49-0).<sup>2</sup> > 0 *so that the following holds. Let* ϵ, δ, γmargin, ρ, β ∈ (0, 1) *and* T ∈ <sup>N</sup> *be given. For base model* πbase*, define reward function* r(x, y) := log πbase(y | x)*. Let* Rmax ≥ 1 + maxx,y log <sup>1</sup> πbase(y|x) *. Suppose that* πbase *satisfies [Assumption 4.2](#page-7-2) with parameter* γmargin*, that* β <sup>−</sup><sup>1</sup> ≥ 2γ −1 margin log(2|Y|/δ)*, and that there is* ϵdisc ∈ (0, 1) *so that*

$$T \geq C_{\text{J.2}} \frac{R_{\max}^2 \text{SEC}(\Pi) \log(2\mathcal{N}(\Pi, \epsilon_{\text{disc}}) T/\rho)}{\epsilon^2 \delta^2 \beta^2}$$

*and*

$$\epsilon_{\text{disc}} \leq c_{\text{J.2}} \frac{\epsilon\delta}{\sqrt{\text{SEC}(\Pi)T}}$$

*where* SEC(Π) := SEC(Π, r, T, β, R<sup>2</sup> max; πbase)*. Also suppose that* π ⋆ <sup>β</sup> ∈ Π *where* π ⋆ β (y | x) ∝ π 1+β −1 base (y | x)*.*

*Then applying [Algorithm 1](#page-47-0) with base model* πbase*, reward function* r*, iteration count* T*, regularization* β*, and optimism parameter* α := β <sup>R</sup>max qlog(2N(Π,ϵdisc)T /δ) SEC(Π)T *yields a model* <sup>π</sup>b <sup>∈</sup> <sup>Π</sup> *such that with probability at least* 1 − ρ*,*

$$\mathbb{P}_{x \sim \mu}[\hat{\pi}(\mathbf{y}^*(x) \mid x) < 1 - \delta] \leq \epsilon.$$

*The total sample complexity is*

$$m = \tilde{O}\left(\frac{R_{\max}^2 \text{SEC}(\Pi) \log(\mathcal{N}(\Pi, \epsilon_{\text{disc}})/\rho) \log^2(|\mathcal{Y}|\delta^{-1})}{\gamma_{\text{margin}}^2 \epsilon^2 \delta^2}\right).$$

Proof of [Theorem J.2.](#page-49-0) By definition of r, we have |r(x, y)| ≤ Rmax for all x, y. By assumption, [Assumption J.3](#page-46-2) is satisfied, and by definition of Rmax, [Assumption 4.5](#page-9-1) is satisfied with parameter

Vmax := βRmax ≤ Rmax. It follows from [Theorem J.1](#page-47-2) that with probability at least 1 − ρ, the output <sup>π</sup>b of [Algorithm 1](#page-47-0) satisfies

$$\beta D_{\text{KL}}(\hat{\pi} \parallel \pi_{\beta}^*) \lesssim (R_{\text{max}} + V_{\text{max}}) \sqrt{\frac{\text{SEC}(\Pi) \log(2\mathcal{N}(\Pi, \epsilon_{\text{disc}}) T/\rho)}{T}} \\ + \beta \epsilon_{\text{disc}} \sqrt{\text{SEC}(\Pi) T}.$$

By choice of T and ϵdisc, so long as C[J](#page-49-0).<sup>2</sup> > 0 is chosen to be a sufficiently large constant and <sup>c</sup>[J](#page-49-0).<sup>2</sup> <sup>&</sup>gt; <sup>0</sup> is chosen to be a sufficiently small constant, we have βDKL <sup>π</sup>b ∥ <sup>π</sup> ⋆ β ≤ <sup>12</sup> βϵδ, so by e.g. Equation (16) of [Sason & Verdú](#page-14-17) [\(2016\)](#page-14-17), D<sup>2</sup> H π, π b ⋆ β ≤ ϵδ/(12).

For any x ∈ X and y ′ ∈ Y \ y ⋆ (x), by [Assumption 4.2](#page-7-2) and definition of π ⋆ <sup>β</sup> we have

$$\begin{aligned} \frac{1}{\pi_\beta^*(y' | x)} &\geq \frac{\max_{y \in \mathcal{Y}} \pi_\beta^*(y | x)}{\pi_\beta^*(y' | x)} = \left( \frac{\max_{y \in \mathcal{Y}} \pi_{\text{base}}(y | x)}{\pi_{\text{base}}(y' | x)} \right)^{1+\beta^{-1}} \\ &\geq (1 + \gamma_{\text{margin}})^{1+\beta^{-1}} \geq e^{\gamma_{\text{margin}}/(2\beta)} \geq \frac{2|\mathcal{Y}|}{\delta} \end{aligned}$$

where the final inequality is by the assumption on β in the theorem statement. Therefore

$$\pi_\beta^*(\mathbf{y}^*(x) \mid x) \geq 1 - \sum_{y' \in \mathcal{V} \setminus \mathbf{y}^*(x)} \pi_\beta^*(y' \mid x) \geq 1 - \frac{\delta}{2}.$$

Now for any x, we can lower bound

$$\begin{aligned} D_{\mathfrak{H}}^2(\widehat{\pi}(\cdot \mid x), \pi_{\beta}^*(\cdot \mid x)) &\geq \left( \sqrt{1 - \widehat{\pi}(\mathbf{y}^*(x) \mid x)} - \sqrt{1 - \pi_{\beta}^*(\mathbf{y}^*(x) \mid x)} \right)^2 \\ &\geq \frac{\delta}{12} \cdot \mathbb{I}\{\widehat{\pi}(\mathbf{y}^*(x) \mid x) \leq 1 - \delta\}. \end{aligned}$$

Hence,

$$\begin{aligned}\mathbb{P}_{x\sim\mu}[\widehat{\pi}(\mathbf{y}^*(x) \mid x) < 1 - \delta] &\leq \frac{12}{\delta} \mathbb{E}_{x\sim\mu} D_{\mathfrak{H}}^2(\widehat{\pi}(\cdot \mid x), \pi_{\beta}^*(\cdot \mid x)) \\ &= \frac{12}{\delta} D_{\mathfrak{H}}^2(\widehat{\pi}, \pi_{\beta}^*) \\ &\leq \epsilon.\end{aligned}$$

as claimed.

#### J.2.4 APPLICATION: LINEAR SOFTMAX MODELS

In this section we apply [Theorem 4.3](#page-9-2) to the class of linear softmax models, proving [Theorem J.3.](#page-50-0) This demonstrates that [Algorithm 1](#page-47-0) can achieve an exponential improvement in sample complexity compared to SFT-Sharpening.

Definition J.3 (Linear softmax model). *Let* d ∈ N *be given, and let* ϕ : X ×Y → R <sup>d</sup> *be a feature map with* ∥ϕ(x, y)∥<sup>2</sup> ≤ 1 *for all* x, y*. Let* πzero : X → ∆(Y) *be the uniform model* πzero(y | x) := <sup>1</sup> |Y| *, and let* B ≥ 1*.* [<sup>15</sup>](#page-0-0) *We consider the linear softmax model class* Πϕ,B := {π<sup>θ</sup> : θ ∈ <sup>R</sup> d , ∥θ∥<sup>2</sup> ≤ B} *where* π<sup>θ</sup> : X → ∆(Y) *is defined by*

$$\pi_\theta(y \mid x) \propto \pi_{\text{zero}}(y \mid x) \exp(\langle \phi(x, y), \theta \rangle).$$

Theorem J.3. *Let* ϵ, δ, γmargin, ρ ∈ (0, 1) *be given. Suppose that* πbase = π<sup>θ</sup> <sup>⋆</sup> ∈ Πϕ,B *for some* θ <sup>⋆</sup> ∈ <sup>R</sup> <sup>d</sup> *with* ∥θ <sup>⋆</sup>∥<sup>2</sup> ≤ γmarginB 3 log(2|Y|/δ) *. Also, suppose that* πbase *satisfies [Assumption 4.2](#page-7-2) with parameter* γmargin*. Then [Algorithm 1](#page-47-0) with base model* πbase*, reward function* r(x, y) := log πbase(x, y)*, regularization parameter* β := γmargin/(2 log(2|Y|/δ))*, and optimism parameter* α(T) ∝ β B+log(|Y|) q<sup>d</sup> log(BdT /(ϵδ))+log(T /ρ) dT log(T) *returns an* (ϵ, δ)*-sharpened model with probability at least* 1 − ρ*, and has sample complexity*

$$m = \text{poly}(\epsilon^{-1}, \delta^{-1}, \gamma_{\text{margin}}^{-1}, d, B, \log(|\mathcal{Y}|/\rho)).$$

<sup>15</sup>We use the notation πzero to highlight the fact that πzero = π<sup>θ</sup> for θ = 0.

Before proving the result, we unpack the conditions. [Theorem J.3](#page-50-0) requires the base model πbase to lie in the model class and also satisfy the margin condition [\(Assumption 4.2\)](#page-7-2). For any constant ϵ, δ > 0, the sharpening algorithm then succeeds with sample complexity poly(d, γ−<sup>1</sup> margin, B, log(|Y|)). These conditions are non-vacuous; in fact, there are fairly natural examples for which non-exploratory algorithm such as SFT-Sharpening require sample complexity exp(Ω(d)), whereas all of the above parameters are poly(d). The following is one such example.

Example J.1 (Separation between RLHF-Sharpening and SFT-Sharpening). Set X = {x} and let Y ⊂ R <sup>d</sup> be a 1/4-packing of the unit sphere in <sup>R</sup> <sup>d</sup> of cardinality exp(Θ(d)). Define ϕ : X ×Y → <sup>R</sup> d by ϕ(x, y) := y, and let B = Cd log d for an absolute constant C > 0. Fix any y <sup>⋆</sup> ∈ Y and define πbase := π<sup>θ</sup> <sup>⋆</sup> ∈ Πϕ,B by θ ⋆ := y ⋆ . Then for any y ̸= y ⋆ , we have ⟨y, y<sup>⋆</sup> ⟩ ≤ 1 − Ω(1), so

$$\frac{\pi_{\text{base}}(y^* \mid x)}{\pi_{\text{base}}(y \mid x)} = \exp(\langle y^* - y, y^* \rangle) = \exp(\Omega(1)) = 1 + \Omega(1).$$

Thus, πbase satisfies [Assumption 4.2](#page-7-2) with γmargin = Ω(1). Moreover, ∥θ <sup>⋆</sup>∥<sup>2</sup> = 1 ≤ γmarginB 3 log(2|Y|/δ) for any δ = 1/poly(d), so long as C is a sufficiently large constant. It follows from [Theorem J.3](#page-50-0) that [Algorithm 1](#page-47-0) computes an (ϵ, δ)-sharpened model with sample complexity poly(ϵ −1 , δ−<sup>1</sup> , d). However, since πbase(y ⋆ | x) ≤ πbase(y | x) · exp(2) for all y ∈ Y, it is clear that

$$C_{\text{cov}} = \mathbb{E} \left[ \frac{1}{\pi_{\text{base}}(\mathbf{y}^*(x) \mid x)} \right] = \frac{1}{\pi_{\text{base}}(y^* \mid x)} = \Omega(|\mathcal{Y}|) = \exp(\Omega(d)).$$

Thus, the sample complexity guarantee for SFT-Sharpening in [Theorem 4.1](#page-7-5) will incur *exponential* dependence on d in the sample complexity. It is straightforward to check that this dependence is real for SFT-Sharpening, and not just an artifact of the analysis, since the model that SFT-Sharpening is trying to learn (via MLE) will itself not be sharp in this example, unless exp(Ω(d)) samples are drawn per prompt. ◁

We now proceed to the proof of [Theorem J.3,](#page-50-0) which requires the following bounds on the covering number and the Sequential Extrapolation Coefficient of Πϕ,B.

Lemma J.4. *Let* ϵdisc > 0*. Then* Πϕ,B *has an* ϵdisc*-net of size* (6B/ϵdisc) d *.*

Proof of [Lemma J.4.](#page-51-0) By a standard packing argument, there is a set {θ1, . . . , θ<sup>N</sup> } of size (6B/ϵdisc) d such that for every θ ∈ R <sup>d</sup> with ∥θ∥<sup>2</sup> ≤ B there is some i ∈ [N] with ∥θ<sup>i</sup> − θ∥<sup>2</sup> ≤ ϵdisc/2. Now for any x ∈ X and y ∈ Y,

$$\begin{aligned} \log \frac{\pi_\theta(y | x)}{\pi_{\theta_i}(y | x)} &= \log \frac{\exp(\langle \phi(x, y), \theta \rangle)}{\exp(\langle \phi(x, y), \theta_i \rangle)} + \log \frac{\mathbb{E}_{(x', y') \sim \pi_{zero}} \exp(\langle \phi(x', y'), \theta_i \rangle)}{\mathbb{E}_{(x', y') \sim \pi_{zero}} \exp(\langle \phi(x', y'), \theta \rangle)} \\ &= \langle \phi(x, y), \theta - \theta_i \rangle + \log \frac{\mathbb{E}_{(x', y') \sim \pi_{zero}} [\exp(\langle \phi(x', y'), \theta \rangle) \exp(\langle \phi(x', y'), \theta_i - \theta \rangle)]}{\mathbb{E}_{(x', y') \sim \pi_{zero}} \exp(\langle \phi(x', y'), \theta \rangle)}. \end{aligned}$$

The first term is bounded by ϵdisc/2 in magnitude. In the second term, we have exp(⟨ϕ(x ′ , y′ ), θ<sup>i</sup> − θ⟩) ∈ [exp(−ϵdisc/2), exp(ϵdisc/2)], so the ratio of expectations lies in [exp(−ϵdisc/2), exp(ϵdisc/2)] as well, and so the log-ratio lies in [−ϵdisc/2, ϵdisc/2]. In all, we get log <sup>π</sup>θ(y|x) πθi (y|x)  <sup>≤</sup> <sup>ϵ</sup>disc. Thus, {πθ<sup>1</sup> , . . . , πθ<sup>N</sup> } is an ϵdisc-net for Π.

Lemma J.5. *Let* r : X × Y → [−Rmax, Rmax] *be a reward function and let* T ∈ <sup>N</sup> *and* β > 0*. If* λ ≥ 4β <sup>2</sup>B<sup>2</sup> + R<sup>2</sup> max *then for any* π <sup>⋆</sup> ∈ Πϕ,B*,*

$$\text{SEC}(\Pi_{\phi, B}, r, T, \beta, \lambda; \pi^*) \lesssim d \log(T + 1).$$

Proof of [Lemma J.5.](#page-51-1) Fix π (1), . . . , π(T) ∈ Πϕ,B. By definition, there are some θ (1), . . . , θ(T) ∈ <sup>R</sup> d with ∥θ (t)∥<sup>2</sup> ≤ B and

$$\pi^{(t)}(y \mid x) \propto \pi_{\text{zero}}(y \mid x) \exp(\langle \phi(x, y), \theta^{(t)} \rangle)$$

Define <sup>ϕ</sup>e : X × Y → <sup>R</sup> <sup>d</sup>+1 by <sup>ϕ</sup>e(x, y) := [ϕ(x, y), r(x,y) Rmax ] and define <sup>θ</sup>e(t) := [β(θ (t) − θ ⋆ ), −Rmax]. Then for any t ∈ [T] we have

$$\begin{aligned} & \mathbb{E}^{(t)} \left[ \beta \log \frac{\pi^{(t)}(y|x)}{\pi^{\star}(y|x)} - r(x, y) - \beta \log \frac{\pi^{(t)}(y'|x)}{\pi^{\star}(y'|x)} + r(x, y') \right]^2 \\ & \lambda \vee \sum_{i=1}^{t-1} \mathbb{E}^{(i)} \left[ \left( \beta \log \frac{\pi^{(t)}(y|x)}{\pi^{\star}(y|x)} - r(x, y) - \beta \log \frac{\pi^{(t)}(y'|x)}{\pi^{\star}(y'|x)} + r(x, y') \right)^2 \right] \\ & = \frac{\mathbb{E}^{(t)} \left[ \langle \tilde{\phi}(x, y) - \tilde{\phi}(x, y'), \tilde{\theta}^{(t)} \rangle \right]^2}{\lambda \vee \sum_{i=1}^{t-1} \mathbb{E}^{(i)} \left[ \left( \langle \tilde{\phi}(x, y) - \tilde{\phi}(x, y'), \tilde{\theta}^{(t)} \rangle \right)^2 \right]} \\ & \leq \frac{\langle \tilde{\theta}^{(t)} \rangle^{\top} \Sigma^{(t)} \tilde{\theta}^{(t)}}{\lambda \vee \sum_{i=1}^{t-1} \langle \tilde{\theta}^{(t)} \rangle^{\top} \Sigma^{(t)} \tilde{\theta}^{(t)}} \end{aligned}$$

where for each i ∈ [T] we have defined Σ (i) := E (i) h (ϕe(x, y) − <sup>ϕ</sup>e(x, y′ ))(ϕe(x, y) − <sup>ϕ</sup>e(x, y′ ))<sup>⊤</sup> i . Observe that ∥θe(t)∥ 2 <sup>2</sup> ≤ 4β <sup>2</sup>B<sup>2</sup> + R<sup>2</sup> max ≤ λ by assumption on λ. Therefore,

$$\begin{aligned} \frac{(\tilde{\theta}^{(t)})^\top \Sigma^{(t)} \tilde{\theta}^{(t)}}{\lambda \vee \sum_{i=1}^{t-1} (\tilde{\theta}^{(t)})^\top \Sigma^{(i)} \tilde{\theta}^{(t)}} &\lesssim \frac{(\tilde{\theta}^{(t)})^\top \Sigma^{(t)} \tilde{\theta}^{(t)}}{\lambda + \sum_{i=1}^{t-1} (\tilde{\theta}^{(t)})^\top \Sigma^{(i)} \tilde{\theta}^{(t)}} \\ &\leq \frac{(\tilde{\theta}^{(t)})^\top \Sigma^{(t)} \tilde{\theta}^{(t)}}{(\tilde{\theta}^{(t)})^\top \left( I_d + \sum_{i=1}^{t-1} \Sigma^{(i)} \right) \tilde{\theta}^{(t)}} \\ &\leq \lambda_{\max} \left( \left( I_d + \sum_{i=1}^{t-1} \Sigma^{(i)} \right)^{-1/2} \Sigma^{(t)} \left( I_d + \sum_{i=1}^{t-1} \Sigma^{(i)} \right)^{-1/2} \right) \\ &\leq \text{Tr} \left( \left( I_d + \sum_{i=1}^{t-1} \Sigma^{(i)} \right)^{-1/2} \Sigma^{(t)} \left( I_d + \sum_{i=1}^{t-1} \Sigma^{(i)} \right)^{-1/2} \right) \\ &= \text{Tr} \left( \left( I_d + \sum_{i=1}^{t-1} \Sigma^{(i)} \right)^{-1} \Sigma^{(t)} \right). \end{aligned}$$

Observe that Tr(Σ(t) ) ≤ maxx,y∥ϕe(x, y)∥ 2 <sup>2</sup> ≲ 1. Hence by [Lemma F.2,](#page-33-5) we have

$$\begin{aligned} & \sum_{t=1}^T \frac{\mathbb{E}^{(t)} \left[ \beta \log \frac{\pi^{(t)}(y|x)}{\pi^*(y|x)} - r(x, y) - \beta \log \frac{\pi^{(t)}(y'|x)}{\pi^*(y'|x)} + r(x, y') \right]^2}{\lambda \vee \sum_{i=1}^{t-1} \mathbb{E}^{(i)} \left[ \left( \beta \log \frac{\pi^{(t)}(y|x)}{\pi^*(y|x)} - r(x, y) - \beta \log \frac{\pi^{(t)}(y'|x)}{\pi^*(y'|x)} + r(x, y') \right)^2 \right]} \\ & \lesssim \sum_{t=1}^T \text{Tr} \left( \left( I_d + \sum_{i=1}^{t-1} \Sigma^{(i)} \right)^{-1} \Sigma^{(t)} \right) \\ & \lesssim d \log(T + 1). \end{aligned}$$

Since π (1), . . . , π(T) ∈ Π were arbitrary, this completes the proof.

The proof is now immediate from [Theorem J.2](#page-49-0) and the above lemmas.

Proof of [Theorem J.3.](#page-50-0) By the assumption on θ ⋆ and choice of β, the model π ⋆ β defined by π ⋆ β (y | x) ∝ πbase(y | x) 1+β −1 satisfies π ⋆ <sup>β</sup> = π(1+β−<sup>1</sup>)<sup>θ</sup> <sup>⋆</sup> ∈ Πϕ,B. By [Lemma J.4,](#page-51-0) we have N (Πϕ,B, ϵdisc) ≤ (6B/ϵdisc) d . Take Rmax := p 4β <sup>2</sup>B<sup>2</sup> + (2B + log |Y|) <sup>2</sup>. We know that r(x, y) := log πbase(y | x) satisfies |r(x, y)| ≤ 2B + log |Y| for all x, y. By [Lemma J.5,](#page-51-1) we therefore get that SEC(Πϕ,B, r, T, β, R<sup>2</sup> max; πbase) ≲ d log(T + 1). Substituting these bounds into [Theorem J.2](#page-49-0) yields the claimed result.