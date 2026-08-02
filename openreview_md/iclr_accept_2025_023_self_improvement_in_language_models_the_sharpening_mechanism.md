# Self-Improvement In Language Models: The Sharpening Mechanism

Audrey Huang∗
UIUC
audreyh5@illinois.edu

| Dhruv Rohatgi MIT drohatgi@mit.edu   |
|--------------------------------------|

Adam Block∗
Microsoft Research blockadam@microsoft.com Cyril Zhang Microsoft Research cyrilzhang@microsoft.com Jordan T. Ash Microsoft Research ash.jordan@microsoft.com Dylan J. Foster∗
Microsoft Research dylanfoster@microsoft.com Max Simchowitz CMU
msimchow@andrew.cmu.edu Akshay Krishnamurthy Microsoft Research akshaykr@microsoft.com

## Abstract

Recent work in language modeling has raised the possibility of *self-improvement*, where a language models evaluates and refines its own generations to achieve higher performance without external feedback. It is impossible for this self-improvement to create information that is not already in the model, so why should we expect that this will lead to improved capabilities?

We offer a new perspective on the capabilities of self-improvement through a lens we refer to as *sharpening*. Motivated by the observation that language models are often better at verifying response quality than they are at generating correct responses, we formalize self-improvement as using the model itself as a verifier during post-training in order to "sharpen" the model to one placing large mass on high-quality sequences, thereby amortizing the expensive inference-time computation of generating good sequences. We begin by introducing a new statistical framework for sharpening in which the learner aims to sharpen a pre-trained base policy via sample access, and establish fundamental limits. Then, we analyze two natural families of self-improvement algorithms based on SFT and RLHF. We find that (i) the SFT-based approach is minimax optimal whenever the initial model has sufficient coverage, but (ii) the RLHF-based approach can improve over SFT-based self-improvement by leveraging online exploration, bypassing the need for coverage. Finally, we empirically validate the sharpening mechanism via inference-time and amortization experiments. We view these findings as a starting point toward a foundational understanding that can guide the design and evaluation of self-improvement algorithms.

## 1 Introduction

Contemporary language models are remarkably proficient on a wide range of natural language tasks (Brown et al., 2020; Ouyang et al., 2022; Touvron et al., 2023; OpenAI, 2023; Google, 2023), but inherit shortcomings of the data on which they were trained. A fundamental challenge is to achieve better performance than what is directly induced by the distribution of available, human-generated training data. To this end, recent work (Huang et al., 2022; Wang et al., 2022; Bai et al., 2022b; Pang et al., 2023; Yuan et al., 2024) has raised the possibility of "self-improvement," where a model—typically through forms of self-play or self-training in which the model critiques its own generations—learns to improve on its own, without external feedback. This phenomenon is somewhat counterintuitive; at first glance it would seem to disagree with the well-known data-processing inequality (Cover, 1999), which implies that no form of self-training should be able to create
∗Equal contribution.

1

BoN: % Lift over Greedy Phi3.5 (Mini) on MATH
Correct Incorrect MATH
400 300 200 100 0 Log Probability 0 20 40 60 80 100 120 140

(a)
0 10 20 30 40 50 N
0 20 40 60 80 100 120 140 Phi3 (Mini)
0.7 -2.8 17.0 -4.2 4.5 3.8
% 
Lift ov er Gr eedy Phi3.5 (Mini)
9.9 4.1 3.2 5.9 6.0 -0.9 Phi3 (Small)
-0.2 4.0 -6.8 1.4 0.5 -9.2 Co unt Phi3 (Mini)
Phi3.5 (Mini) Phi3 (Small) Phi3 (Medium)
Mistral-7B
Llama3.2-3B GPT-3.5 Phi3 (Medium)
-2.1 -4.4 1.8 -1.5 3.3 -0.7 Mistral-7B
26.1 10.5 -1.1 11.8 28.6 10.2 Llama3.2-3B
-7.1 -2.6 3.0 -14.9 126.7 6.7 GPT-3.5 3.4 -11.8 -10.4 -0.6 6.7 -7.0 MATH
GSM8K
ProntoQA Bio Phys Chem
(b)
(c)
information not already in the model. This motivates the question of why we should expect such supervision-free interventions will lead to stronger reasoning and planning capabilities. A dominant hypothesis for why improvement without external feedback might be possible is that models contain "hidden knowledge" (Hinton et al., 2015) that is difficult to access. Self-improvement, rather than creating knowledge from nothing, is a means of extracting and distilling this knowledge into a more accessible form, and thus is a computational phenomenon rather than a statistical one.

While there is a growing body of empirical evidence for this hidden-knowledge hypothesis (Furlanello et al., 2018; Gotmare et al., 2019; Dong et al., 2019; Abnar et al., 2020; Allen-Zhu & Li, 2020), particularly in the context of self-distillation, a fundamental understanding of self-improvement remains missing. Concretely, where in the model is this hidden knowledge, and when and how can it be extracted?

## 1.1 Our Perspective: The Sharpening Mechanism

In this paper we posit a source of hidden knowledge, and offer a formal perspective on how to extract it. Our starting point is the widely observed phenomenon that language models are often better at verifying whether responses are correct than they are at generating correct responses (Huang et al., 2022; Wang et al., 2022; Bai et al., 2022b; Pang et al., 2023; Yuan et al., 2024). This gap may be explained by the theory of computational complexity, which suggests that generating high-quality responses can be less computationally tractable than verification (Cook, 1971; Levin, 1973; Karp, 1972). In autoregressive language modeling, computing the most likely response for a given prompt is NP-hard in the worst case (Appendix E), whereas the model's likelihood for a given response can be easily evaluated. We view self-improvement as any attempt to narrow this gap, i.e., use the model as its own verifier to improve generation and *sharpen* the model toward high-quality responses. Formally, consider a learner with access to a base model πbase : X → ∆(Y) representing a conditional distribution that maps a prompt x ∈ X to a distribution over responses (i.e., π*base*(y | x) is the probability that the model generates the response y given the prompt x).1 We posit that πbase has already been trained in some manner (e.g., through next-token prediction or additional post-training steps such as SFT or RLHF), with the key feature being that πbase is a good verifier, as measured by some *self-reward* function rself(y | x; πbase) measuring model certainty. The self-reward function is derived purely from the base model πbase, without external supervision or feedback. Examples include normalized and/or regularized sequence likelihood (Meister et al., 2020), models-as-judges (Zheng et al., 2024; Yuan et al., 2024; Wu et al., 2024a; Wang et al., 2024), and model confidence (Wang & Zhou, 2024).

## Sharpening

We refer to **sharpening** as any process that tilts πbase toward responses that are more certain in the sense that they enjoy greater self-reward rself. That is, a sharpened model πb is one that (approximately) maximizes the self-reward:
πb(x) ≈ arg max y∈Y
r*self*(y | x; πbase). (1)
An important special case for sharpening is in language/autoregressive modeling. Here, we have Y = V
H for a vocabulary space V and sequence length H, and πbase has the autoregressive structure πbase(y1:H | x) = QH
h=1 πbase,h(yh | y1:h−1, x) for y = y1:H ∈ Y. Sharpening in this setting pertains to entire responses, i.e., the optimization over responses in Eq. (1) is at the sequence level. In contrast, popular decoding strategies such as greedy, low-temperature sampling, and beam search operate at the token-level; nevertheless, they can be viewed as heuristics for *inference-time* sharpening.

2 The combinatorial response space can make sharpening computationally demanding and so, an appealing alternative to inference-time sharpening is *amortization via self-training* (Section 2). The latter captures many existing self-training schemes (Huang et al., 2022; Wang et al., 2022; Bai et al., 2022b; Pang et al., 2023; Yuan et al., 2024), and is the main focus of this paper; we use the term *sharpening* without further qualification to refer to the latter. We refer to the **sharpening mechanism** as the phenomenon where responses from a model with the highest certainty (in the sense of large self-reward rself) exhibit the greatest performance on a task of interest. Though it is unclear a-priori whether there are self-rewards related to task performance, the successes of self-improvement in prior works (Huang et al., 2022; Wang et al., 2022; Bai et al., 2022b; Pang et al., 2023; Yuan et al., 2024) give strong positive evidence. These works suggest that, in many settings, models do have hidden knowledge: the model's own self-reward correlates with response quality, but it is computationally challenging to generate high self-rewarding—and thus high quality—responses. It is the role of (algorithmic) sharpening to leverage these verifications to improve the quality of generations, despite computational difficulty.

## 1.2 Contributions

We initiate the theoretical study of self-improvement via the sharpening mechanism. We disentangle the choice of self-reward from the algorithms used to optimize it, and aim to understand: (i) When and how does self-training achieve sharpening? (ii) What are the fundamental limits for such algorithms? Algorithms for sharpening (Section 2). The starting point for our work is to consider two natural families of self-improvement algorithms based on supervised fine-tuning (SFT) and reinforcement learning (RL/RLHF), respectively, SFT-Sharpening and RLHF-Sharpening. Both algorithms amortize the sharpening objective (1) into a dedicated post-training/fine-tuning phase:
- SFT-Sharpening filters responses where the self-reward rself(y | x; πbase) is large and fine-tunes on the resulting dataset, invoking common SFT pipelines (Amini et al., 2024; Sessa et al., 2024).

- RLHF-Sharpening directly applies reinforcement learning techniques (e.g., PPO (Schulman et al.,
2017) or DPO (Rafailov et al., 2023)) to optimize the self-reward function rself(y | x; πbase).

In the remainder of the paper, we introduce a theoretical framework to analyze the performance of these algorithms. Our main contributions are as follows. Maximum-likelihood sharpening objective (Section 3.1). As a concrete proposal for one source of hidden knowledge, we focus on self-rewards defined by the model's sequence-level log-probabilities:
rself(y | x; πbase) := log πbase(y | x) (2)
This is a stylized self-reward function, which offers perhaps the simplest objective for selfimprovement in the absence of external feedback (i.e., purely supervision-free), yet also connects self-improvement to a rich body of theoretical computer science literature on computational trade-offs for optimization (inference) versus sampling (Appendix B). Despite its simplicity, maximum-likelihood sharpening is already sufficient to achieve non-trivial performance gains over 2More sophisticated decoding strategies like normalized/regularized sequence likelihood (Meister et al., 2020)
or chain-of-thought decoding (Wang & Zhou, 2024) also admit an interpretation as sharpening; see Appendix B.

greedy decoding on a range of reasoning tasks with several language models; (Figure 1). We believe it can serve as a starting point toward understanding forms of self-improvement that use more sophisticated self-rewards (Huang et al., 2022; Wang et al., 2022; Pang et al., 2023; Yuan et al., 2024). A statistical framework for sharpening (Sections 3.2 and **3.3).** Though the goal of sharpening is computational in nature, we recast self-training according to the maximum-likelihood sharpening objective Eq. (2) as a **statistical** problem where we aim to produce a model approximating (1) using a polynomial number of (i) sample prompts x ∼ µ, (ii) sampling queries of the form y ∼ πbase(x), and (iii) likelihood evaluations of the form πbase(y | x). Evaluating the efficiency of the algorithm through the number of such queries, this abstraction offers a natural way to evaluate the performance of self-improvement/sharpening algorithms and establish fundamental limits; we use our framework to prove new lower bounds that highlight the importance of the base model's coverage. Analysis of sharpening algorithms (Section 4). Within our statistical framework for sharpening, we show that SFT-Sharpening and RLHF-Sharpening provably converge to sharpened models, establishing several results: (i) SFT-Sharpening **is minimax optimal**, and learns a sharpened model whenever πbase has sufficient coverage (we also show that a novel variant based on adaptive sampling can sidestep the minimax lower bound); (ii) RLHF-Sharpening **benefits from on-policy exploration**, and can bypass the need for coverage—improving over SFT-Sharpening.

Empirical investigation (Appendix A). We explore empirically the extent to which our theoretical framework and methods improve language model performance in a variety of tasks. We consider three choices of self-reward on an extensive list of model-dataset pairs and conclude that sharpening can often improve performance. We then implement one of our algorithms, SFT-Sharpening, on a subset of these model-dataset pairs and observe a significant positive effect on performance. A
summary of our inference-time experiments can be found in Figure 1.

## 1.3 Related Work

Our work is most directly related to a growing body of empirical research that studies self-training for language models in a supervision-free setting with no external feedback (Huang et al., 2022; Wang et al., 2022; Bai et al., 2022b; Pang et al., 2023; Yuan et al., 2024). The specific algorithms for self-improvement/sharpening we study can be viewed as applications of standard alignment algorithms (Amini et al., 2024; Sessa et al., 2024; Christiano et al., 2017; Bai et al., 2022a; Ouyang et al., 2022; Rafailov et al., 2023) with a specific choice of reward function. However, the maximum likelihood sharpening objective (2) used for our theoretical results has been relatively unexplored within the alignment and self-improvement literature. Theoretical understanding of self-training is currently limited. One line of work analyzes the convergence of self-training for classification and regression with the *self-distillation objective*,
but is limited to stylized setups such as linear models (Mobahi et al., 2020; Frei et al., 2022; Das
& Sanghavi, 2023; Das et al., 2024; Pareek et al., 2024), feedforward neural networks (Allen-Zhu
& Li, 2020), and a general PAC-style framework (Boix-Adsera, 2024). To the best of our knowledge, our work is the first to study self-training in a general framework that subsumes language modeling.

See Appendix B for a more extensive discussion of related work.

## 2 Sharpening Algorithms For Self-Improvement

This section introduces the two families of self-improvement algorithms for sharpening that we study.

Going forward, we omit the dependence of rself on πbase when it is clear from context. We use the notation arg maxπ∈Π or arg minπ∈Π to denote exact optimization over a user-specified model class Π for theoretical results (Agarwal et al., 2019; Foster & Rakhlin, 2023); empirically, these operations can be implemented by training a neural network to low loss.

## 2.1 Self-Improvement Through Sft: Sft-Sharpening

SFT-Sharpening filters responses for which the self-reward rself(y | x) is large, and applies standard supervised fine-tuning on the resulting dataset (Amini et al., 2024; Sessa et al., 2024; Gui et al., 2024; Pace et al., 2024). This can be viewed as amortizing inference-time sharpening via the effective-but-costly best-of-N sampling approach (Brown et al., 2024; Snell et al., 2024; Wu et al., 2024b). Concretely, suppose we have a collection of prompts x1*, . . . , x*n. For each prompt, we sample N responses yi,1, . . . , yi,N ∼ πbase(· | xi), then compute the best-of-N response y BoN
i = arg maxj∈[N]{rself(yi,j | xi)}, scoring via the model's self-reward function. We compute the sharpened model via supervised fine-tuning on the best-of-N responses:

$${\hat{\pi}}^{\mathsf{B o N}}=\arg\operatorname*{max}_{\pi\in\Pi}\sum_{i=1}^{n}\log\pi(y_{i}^{\mathsf{B o N}}\mid x_{i}).$$
$$({\mathcal{I}})$$

This is a simple, flexible self-training scheme, and converges to a sharpened model as n, N → ∞. 2.2 SELF-IMPROVEMENT THROUGH RLHF: RLHF-Sharpening A drawback of the SFT-Sharpening algorithm is that it may ignore useful information contained in the self-reward function rself(y | x). Fixing a regularization parameter β > 0 throughout, our second class of algorithms solve a KL-regularized reinforcement learning problem in the spirit of RLHF and other alignment methods (Christiano et al., 2017; Rafailov et al., 2023). Defining Eπ[·] = Ex∼µ,y∼π(·|x)[·] and DKL(π ∥ πbase) = Eπ-log π(y|x)
πbase(y|x)
, we choose

$\pi\approx\arg\max\{\mathbb{E}_{\pi}[r_{\rm self}(y\mid x)]-\beta D_{\rm KL}(\pi\parallel\pi_{\rm base})\}$.  
{Eπ[r*self*(y | x)] − βDKL(π ∥ πbase)}. (3)
The exact optimizer π
⋆
β = arg maxπ∈Π{Eπ[rself(y | x)] − βDKL(π ∥ πbase)} for this objective has the form π
⋆β
(y | x) ∝ π*base*(y | x) · expβ
−1rself(y | x), which converges to the solution to the sharpening objective in Eq. (1) as β → 0. Thus, Eq. (3) can be seen to encourage sharpening. There are many choices for what RLHF/alignment algorithm one might use to solve (3). For our theoretical results, we implement Eq. (3) using an approach inspired by DPO and its reward-based variants (Rafailov et al., 2023; Gao et al., 2024). Given a dataset D = {(*x, y, y*′)} of n examples sampled via x ∼ µ and *y, y*′ ∼ πbase(y | x), we consider the algorithm that solves

$$\widehat{\pi}\in\arg\min_{\pi\in\Pi}\sum_{(x,y,y^{\prime})\in\mathcal{D}}\left(\beta\log\frac{\pi(y\mid x)}{\pi_{\max}(y\mid x)}-\beta\log\frac{\pi(y^{\prime}\mid x)}{\pi_{\max}(y^{\prime}\mid x)}-\left(r_{\mathrm{sat}}(y\mid x)-r_{\mathrm{sat}}(y^{\prime}\mid x)\right)\right)^{2}.\tag{4}$$

In Section 4, we show that this approach achieves guarantees similar to SFT-Sharpening, while a more sophisticated DPO variant with *online exploration* (Xie et al., 2024) provides provable benefits.

## 3 A Statistical Framework For Sharpening

This section introduces the theoretical framework within which we will analyze the SFT-Sharpening and RLHF-Sharpening algorithms. We first introduce the maximum-likelihood sharpening objective as a stylized self-reward function, then introduce our statistical framework for sharpening. We write f = Oe(g) to denote f = O(g · max{1, polylog(g)}) and a ≲ b as shorthand for a = O(b).

3.1 MAXIMUM-LIKELIHOOD SHARPENING Our theoretical results focus on the maximum-likelihood sharpening objective given by rself(y | x) := log π*base*(y | x),
which we aim to maximize using conditional samples y ∼ πbase(· | x) from the base model. This is a simple and stylized self-reward function, but we will show that it enjoys a rich theory. In particular, we can restate the problem of sharpening with this self-reward through the lens of amortization.

Can we efficiently **amortize maximum likelihood inference (optimization)** *for a conditional* distribution πbase(y | x) given access to a **sampling oracle** *that can sample* y ∼ πbase(· | x)?

The tacit assumption in this framing is that the maximum-likelihood response constitutes a useful form of hidden knowledge. Maximum-likelihood sharpening connects the study of self-improvement to a large body of research in theoretical computer science demonstrating computational reductions between optimization (inference) and sampling (generation) (Kirkpatrick et al., 1983; Lovász & Vempala, 2006; Singh & Vishnoi, 2014; Ma et al., 2019; Talwar, 2019). Our sharpening framework offers a new learning-theoretic perspective by focusing on the problem of amortizing this type of reduction. We evaluate the quality of an approximately sharpened model as follows. Let

$$y^{\star}(x):=\operatorname*{arg\,max}_{y\in\mathcal{Y}}\log\pi_{\mathrm{base}}(y\mid x);$$

we interpret y
⋆(x) ⊂ Y as a set to accommodate non-unique maximizers, and will write y
⋆(x) to indicate a unique maximizer when it exists (i.e., when y
⋆(x) = {y
⋆(x)}).

Definition 3.1 (Sharpened model). We say that a model πb is (ϵ, δ)*-sharpened relative to* πbase if Px∼µ[πb(y
⋆(x) | x) ≥ 1 − δ] ≥ 1 − ϵ.

That is, an (ϵ, δ)-sharpened model places at least 1 − δ mass on arg-max responses on all but an ϵ-fraction of prompts under µ. For small δ and ϵ, we are guaranteed that πb is a high-quality generator:
sampling from the model will produce an arg-max response with high probability for most prompts. Maximum-likelihood sharpening **for autoregressive models.** Though our most general results are agnostic to the structure of X , Y, and πbase, our primary motivation is the autoregressive setting in which Y = V
H for a *vocabulary space* V and sequence length H, and where πbase has the autoregressive structure πbase(y1:H | x) = QH
h=1 πbase,h(yh | y1:h−1, x) for y = y1:H ∈ Y.

We observe that when the response y = (y1, . . . , yH) ∈ Y = V
H is a sequence of tokens, the maximum-likelihood sharpening objective (2) sharpens toward the *sequence-level* arg-max response:
arg max y1:H
log πbase(y1:H | x). (5)
Although somewhat stylized, Eq. (5) is a non-trivial (in general, computationally intractable; see Appendix E) solution concept. We view the sequence-level arg-max as a form of hidden knowledge that cannot necessarily be uncovered through naive sampling or greedy decoding. Role of δ **for autoregressive models.** As can be verified through simple examples, beam-search and greedy tokenwise decoding do not return an exact (or even approximate) solution to (5) in general. There is one notable exception: If the model has already been sharpened to δ < 1/2 and the arg-max sequence is unique, then greedy decoding will succeed.

Proposition 3.1 (Greedy decoding succeeds for sharpened policies). Let π = π1:H be an autoregressive model defined over response space Y = V
H. For a given prompt x ∈ X *, if* y
⋆(x) = {y
⋆(x)} *is a singleton and* π(y
⋆(x) | x) > 1/2, then the greedy decoding strategy that selects ybh = arg maxyh∈V πh(yh | yb1, . . . , ybh−1, x) *guarantees that* yb = y
⋆(x). This result is tight, in the sense that there exist π *with* π(y
⋆(x) | x) ≤ 1/2 *for which greedy decoding fails to recover* y
⋆(x).

This means that if we start from an un-sharpened model, simply sharpening to δ < 1/2 may suffice. 3.2 SAMPLE COMPLEXITY FRAMEWORK Sharpening, as described in Definition 3.1, is a purely computational problem, which makes it difficult to evaluate the optimality of self-improvement algorithms. To address this, we introduce a novel statistical framework for sharpening, inspired by the oracle complexity in optimization (Nemirovski et al., 1983; Traub et al., 1988; Raginsky & Rakhlin, 2011; Agarwal et al., 2012) and statistical query complexity in computational learning theory (Blum et al., 1994; Kearns, 1998; Feldman, 2012; 2017). Definition 3.2 (Sample-and-evaluate framework). In the **sample-and-evaluate** framework, the algorithm designer does not have explicit access to the base model πbase*. Instead, they access* πbase only through sample-and-evaluate queries: The learner is allowed to sample n prompts x ∼ µ. For each prompt x, they can sample N responses y1, y2*, . . . y*N ∼ πbase(· | x) *and observe the likelihood* πbase(yi| x) *for each such response. The efficiency, or* sample complexity, of the algorithm is measured through the total number of sample-and-evaluate queries m := n · N.

This framework can be seen to capture algorithms like SFT-Sharpening and RLHF-Sharpening (implemented with DPO), which only access the base model πbase through i) sampling responses via y ∼ πbase(· | x) **(generation)**, and ii) evaluating the likelihood πbase(y | x) **(verification)** for these responses. We view the sample complexity m = n · N as a natural statistical abstraction for the computational complexity of self-improvement (a clear parallel to oracle complexity for optimization algorithms), one which is amenable to information-theoretic lower bounds.3 We will aim to show that, under appropriate assumptions, SFT-Sharpening and RLHF-Sharpening can learn an (ϵ, δ)-sharpened model with sample complexity m = poly(ϵ
−1, δ−1, Cprob)
where Cprob is a potentially problem-dependent constant.

3Concretely, the sample complexity m = n · N is a lower bound on the running time of any algorithm that operates in the sample-and-evaluate framework.

## 3.3 Fundamental Limits

Before diving into our analysis of SFT-Sharpening and RLHF-Sharpening in the sample-andevaluate framework, let us take a brief detour to give a sense for how sample complexity guarantees for sharpening should scale. To this end, we will prove a lower bound or fundamental limit on the sample complexity of any algorithm in the sample-and-evaluate framework. Intuitively, the performance of any sampling-based sharpening algorithm should depend on well the base model πbase covers the arg-max response y
⋆(x). To capture this, we use the *coverage coefficient*4

$C_{\rm cov}=\mathbb{E}_{x\sim\mu}\left[\frac{1}{\pi_{\rm base}(\mathbf{y}^{\star}(x)\mid x)}\right]$,
$$(6)$$
, (6)
and, for a model π, we define y π(x) = arg maxy∈Y π(y | x) and Ccov(π) = Ex∼µ

d $C_{\text{cov}}(\pi)=\mathbb{E}_{x\sim\mu}\left[\frac{1}{\pi(y^{\pi}(x)|x)}\right]$. 
Our main lower bound shows that for a worst-case choice of Π, the coverage coefficient serves as a lower bound on the sample complexity of any sharpening algorithm.

Theorem 3.1 (Lower bound for sharpening). Fix an integer d ≥ 1 *and parameters* ϵ ∈ (0, 1)
and C ≥ 1. There exists a class of models Π *such that (i)* log |Π| ≍ d(1 + log(Cϵ−1))*, (ii)*
supπ∈Π Ccov(π) ≲ C*, and (iii)* y π(x) is a singleton for all π ∈ Π, x ∈ X . Any sharpening algorithm πb *that achieves* E[Px∼µ[πb(y πbase (x) | x) > 1/2]] ≥ 1 − ϵ *for all* πbase ∈ Π *must collect* a total number of samples m = n · N *at least*

$$m\geq{\frac{C\log|\Pi|}{\epsilon^{2}\cdot(1+\log(C\epsilon^{-1}))}}.$$

This result shows that the complexity of any (ϵ, 1/2 − δ)-sharpening algorithm (for δ > 0) in the sample-and-evaluate framework must depend polynomially on the coverage coefficient Ccov, as well as the accuracy ϵ. The lower bound also depends on the expressivity of πbase, as captured by the model class complexity term log|Π|. We will show in the sequel that it is possible to match this lower bound. Note that this result also implies a lower bound for the general sharpening problem (i.e., general rself), since maximum-likelihood sharpening is a special case.

Remark 3.1 (Relaxed notions of sharpening and coverage). The notion of coverage in Eq. (6) is somewhat stringent, since it requires that πbase *place large mass on* y
⋆(x) on average. In *Appendix F,*
we introduce a more general and permissive notion of approximate sharpening (Definition F.1) which leads to weaker coverage requirements, and use this to give generalized versions of our main results. We close this section by noting that numerous recent works—focusing on inference-time computation—show that standard language models exhibit favorable coverage with respect to desirable responses (Brown et al., 2024; Snell et al., 2024; Wu et al., 2024b). We replicate these findings in our experimental setup in Appendix A. These works suggest that, despite the exponentially large response space, the coverage coefficient Ccov may be small in standard language modeling tasks.

## 4 Analysis Of Sharpening Algorithms

Equipped with the sample complexity framework from Section 3, we now prove that the SFT-Sharpening and RLHF-Sharpening families of algorithms provably learn a sharpened model for the maximum likelihood sharpening objective. We treat the model class Π as a fixed, user-specified input. In the tradition of statistical learning theory, our results allow for general classes Π and are agnostic to its structure beyond standard generalization arguments. 4.1 ANALYSIS OF SFT-Sharpening Recall that when we specialize to the maximum-likelihood sharpening self-reward, the SFT-Sharpening algorithm takes the form πb BoN = arg maxπ∈Π
Pn i=1 log πbase(y BoN
i| xi), where y BoN
i = arg maxj∈[N]{log πbase(yi,j | xi)} for yi,1, . . . , yi,N ∼ πbase(· | xi).

Assumption 4.1. The model class Π *satisfies* π BoN
N ∈ Π.

Our main sample complexity guarantee for SFT-Sharpening is as follows. Theorem 4.1 (Sample complexity of SFT-Sharpening). Let ϵ, δ, ρ ∈ (0, 1) be given, and suppose we set n = c ·
log(|Π|ρ
−1)
δϵ and N⋆ = c ·
Ccov log(2δ
−1)
ϵfor an appropriate constant c > 0.

Then with probability at least 1 − ρ, SFT-Sharpening produces a model πb *such that that* Px∼µ[πb(y
⋆(x) | x) ≤ 1 − δ] ≤ ϵ*, and has total sample complexity*5

$$m=O\left(\frac{C_{\mathrm{cov}}\log(|\Pi|\rho^{-1})\log(\delta^{-1})}{\delta\epsilon^{2}}\right).$$

$$\left(7\right)$$
. (7)
This result shows that SFT-Sharpening is minimax optimal in the sample-and-evaluate framework when δ is constant. In particular, the bound in Eq. (7) matches the lower bound in Theorem 3.1 up to polynomial dependence on δ and logarithmic factors. Whether the 1/δ factor in Eq. (7) can be removed is an interesting technical question, but may not be practically consequential because—as discussed in Section 3.2—the regime δ < 1/2 is most meaningful for autoregressive language modeling. Remark 4.1 (On realizability and coverage). Realizability assumptions such as Assumption 4.1
(which asserts that the class Π is powerful enough to model the distribution of the best-of-N responses)
are standard in learning theory (Agarwal et al., 2019; Foster & Rakhlin, 2023), though certainly non-trivial (see Appendix E for a natural example where they may not hold). The coverage assumption, while also standard, when combined with the hypothesis that high-likelihood responses are desirable, suggests that πbase generates high-quality responses with reasonable probability. In general, doing so may require leveraging non-trivial serial *computation at inference time via procedures such as* Chain-of-Thought (Wei et al., *2022). Although recent work shows that such serial computation* cannot be amortized (Li et al., 2024; Malach, *2023),* SFT-Sharpening *instead amortizes the* parallel computation of best-of-N *sampling, and thus has different representational considerations.* Benefits of adaptive sampling. SFT-Sharpening is optimal in the sample-and-evaluate framework, but we show in Appendix D that a variant which selects the number of responses adaptively based on the prompt x can bypass this lower bound, improving the ϵ-dependence in Eq. (7) from 1 ϵ 2 to 1 ϵ
.

Empirical validation. In Appendix A, we empirically investigate the benefits of BoN on a variety of model-dataset pairs. Our results, summarized in Table 1 and Figs. 7 and 8, broadly show that the aforementioned benefits of inference-time sharpening, to an extent, amortized at training time. 4.2 ANALYSIS OF RLHF-Sharpening We now turn our attention to theoretical guarantees for the RLHF-Sharpening algorithm family, which uses tools from reinforcement learning to optimize the self-reward function. When specialized to maximum-likelihood sharpening, the RL objective used by RLHF-Sharpening takes the form πb ≈ arg maxπ∈Π{Eπ[log πbase(y | x)] − βDKL(π ∥ πbase)} for β > 0. The exact optimizer π
⋆
β = arg maxπ∈Π{Eπ[log πbase(y | x)] − βDKL(π ∥ πbase)} for this objective has the form π
⋆
β(y | x) ∝ π 1+β
−1 base (y | x), which converges to a sharpened model (per Definition 3.1) as β → 0.

The key challenge we encounter in this section is the mismatch between the RL reward log πbase(y | x) and the sharpening desideratum πb(y
⋆(x) | x). For example, suppose a unique argmax—say, y
⋆(x)—and second-to-argmax—say, y
′(x)—are nearly as likely under πbase. Then the RL reward Eπb[log πbase(y | x)] must be optimized to extremely high precision before πb can be guaranteed to distinguish the two. To quantify this effect, we introduce a *margin condition*.

Assumption 4.2 (Margin). *For a margin parameter* γmargin > 0*, the base model* πbase *satisfies* max y∈Y
πbase(y | x) ≥ (1 + γmargin) · πbase(y
′| x) ∀y
′ ∈/ y
⋆(x), ∀x ∈ supp(µ).

SFT-Sharpening does not suffer from the pathology in the example above, because once y
⋆(x) and y
′(x) are drawn in a batch of N responses, we have y BoN
i = y
⋆(xi) regardless of margin. However, as we shall show in Section 4.2.2, the RLHF-Sharpening algorithm is amenable to online exploration, which may improve dependence on other problem parameters.

5We focus on finite classes for simplicity, following a convention in reinforcement learning theory (Agarwal et al., 2019; Foster & Rakhlin, 2023), but our results extend to infinite classes through standard arguments.

4.2.1 GUARANTEES FOR RLHF-Sharpening WITH DIRECT PREFERENCE OPTIMIZATION The first of our theoretical results for RLHF-Sharpening takes an offline reinforcement learning approach, whereby we implement Eq. (3) using a reward-based variant of Direct Preference Optimization (DPO) (Rafailov et al., 2023; Gao et al., 2024). Let Dpref = {(*x, y, y*′)} be a dataset of n examples sampled via x ∼ µ, *y, y*′ ∼ πbase(y | x). For a parameter β > 0, we solve πb ∈ arg minπ∈Π

$$\sum_{(x,y,y^{\prime})\in\mathcal{D}_{\mathrm{rest}}}\left(\beta\log{\frac{\pi(y\mid x)}{\pi_{\mathrm{base}}(y\mid x)}}-\beta\log{\frac{\pi(y^{\prime}\mid x)}{\pi_{\mathrm{base}}(y^{\prime}\mid x)}}-\left(\log\pi_{\mathrm{base}}(y\mid x)-\log\pi_{\mathrm{base}}(y^{\prime}\mid x)\right)\right)^{2}.$$

Assumptions. Per Rafailov et al. (2023), the solution to Eq. (8) coincides with that of Eq. (2) asymptotically. To provide finite-sample guarantees, we make a number of statistical assumptions. First, we make a natural realizability assumption (e.g., Zhu et al. (2023); Xie et al. (2024)). Assumption 4.3 (Realizability). The model class Π *satisfies* π
⋆
β ∈ Π.

6 Next, we define two concentrability coefficients for a model π:

$\mathbf{u}=\mathbf{u}\times\mathbf{v}$
$$({\boldsymbol{\delta}})$$
$$\mathcal{C}_{\pi}=\mathbb{E}_{\pi}\Bigg{[}\frac{\pi(y\mid x)}{\pi_{\text{base}}(y\mid x)}\Bigg{]},\quad\text{and}\quad\mathcal{C}_{\pi/\pi^{\prime};\beta}:=\mathbb{E}_{\pi}\Bigg{[}\left(\frac{\pi(y\mid x)}{\pi^{\prime}(y\mid x)}\right)^{\beta}\Bigg{]}.$$  The result shows that both coefficients are bounded for the KL-regularized model 
$$(9)$$
β#. (9)
The following result shows that both coefficients are bounded for the KL-regularized model π
⋆β.

Lemma 4.1. *The model* π
⋆ β satisfies Cπ
⋆ β
≤ Ccov and Cπbase/π⋆β;β *≤ |Y|*.

Motivated by this result, we assume the coefficients in Eq. (9) are bounded for all π ∈ Π. Assumption 4.4 (Concentrability). All π ∈ Π *satisfy* Cπ ≤ Cconc *for a parameter* Cconc ≥ Ccov*, and* Cπbase/π;β ≤ Closs *for a parameter* Closs *≥ |Y|*. By Lemma 4.1, this assumption is consistent with Assumption 4.3 for reasonable bounds on Cconc and Closs; note that our sample complexity bounds will only incur logarithmic dependence on Closs. Main result. Our sample complexity guarantee for RLHF-Sharpening (via Eq. (8)) is as follows.

Theorem 4.2. Let ϵ, δ, ρ ∈ (0, 1) *be given. Set* β ≲ γmarginδϵ, and suppose that Assumptions 4.2 to 4.4 *hold with parameters* Cconc, Closs*, and* γmargin > 0. For an appropriate choice for n, the DPO
algorithm (Eq. (8)) ensures that with probability at least 1 − ρ, Px∼µ[πb(y
⋆(x) | x) ≤ 1 − δ] ≤ ϵ, and has sample complexity

$$m=\tilde{O}\Biggl(\frac{C_{\mathrm{conc}}\log^{3}(C_{\mathrm{loss}}|\Pi|\rho^{-1})}{\gamma_{\mathrm{margin}}^{2}\delta^{2}\epsilon^{2}}\Biggr)$$
!
.
Compared to the guarantee for SFT-Sharpening, RLHF-Sharpening learns a sharpened model with the same dependence on the accuracy ϵ, but a worse dependence on δ; as we primarily consider δ constant (cf. Proposition 3.1), we view this as relatively unimportant. We further remark that RLHF-Sharpening uses N = 2 responses per prompt, while SFT-Sharpening uses many
(N ≈ Ccov/ϵ) responses but fewer prompts. Other differences include:
- RLHF-Sharpening requires the margin condition in Assumption 4.2, and has sample complexity scaling with γ
−1 margin. We believe this dependence is natural for algorithms based on reinforcement learning, as it relates suboptimality with respect to the reward function rself(y | x) = log πbase(y | x) (i.e., Ex∼µ-maxy∈Y log πbase(y | x) − Ey∼πb(x)[log πbase(y | x)]≤ ϵ, the objective minimized by reinforcement learning) to approximate sharpening error Px∼µ[πb(y
⋆(x) | x) ≤ 1 − δ]. However, it is not clear if the precise dependence we pay is necessary.

- RLHF-Sharpening requires a bound on the uniform coverage parameter Cconc, which is generally larger than the parameter Ccov required by SFT-Sharpening. We expect that this assumption can be removed by incorporating pessimism (Liu et al., 2024; Huang et al., 2024). Also, RLHF-Sharpening requires a bound on the parameter Closs, which grants control over the (otherwise unbounded) range of the reward function log πbase(y | x). Since the dependence on Closs is only logarithmic, we view this as fairly mild. Overall, the guarantee in Theorem 4.2 may be somewhat pessimistic; it would be interesting if the result can be improved to match the sample complexity of SFT-Sharpening.

6See Remark 4.1 for a discussion of this assumption.

## 4.2.2 Benefits Of Exploration

The guarantee in Theorem 4.2 scales with the coverage parameter Ccov = E[1/πbase(y
⋆(x)|x)],
which in general is unavoidable in the sample-and-evaluate framework via our lower bound, Theorem 3.1. Although Ccov is a problem-dependent parameter, in the worst case it can be as large as |Y| (which is exponential in sequence length for autoregressive models). Fortunately, unlike SFT-Sharpening, the RLHF-Sharpening objective (3) is amenable to RL algorithms employing active exploration, leading to improved sample complexity when the class Π has additional structure. Our below guarantees for RLHF-Sharpening replace the assumption of bounded coverage with boundedness of a structural parameter for the model class Π known as the "sequential extrapolation coefficient" (SEC) (Xie et al., 2023; 2024), which we denote by SEC(Π). The formal definition is deferred to Appendix J.2. Conceptually, SEC(Π) may thought of as a generalization of the eluder dimension (Russo & Van Roy, 2013; Jin et al., 2021). It can always be bounded by the coverability coefficient of the model class (Xie et al., 2024) and can be as large as Cconc in the worst case, so that bounds based on the SEC reflect improvements that are possible in favorable instances. Beyond boundedness of the SEC, we require a bound on the range of the log-probabilities of πbase.

Assumption 4.5 (Bounded log-probabilities). For all π ∈ Π, (x, y) *∈ X ×Y*,log 1 πbase(y|x)
 ≤ Rmax.

We expect that the dependence on Rmax in our result can be replaced with log(Closs) (Assumption 4.4), but we omit this extension to simplify presentation. We appeal to (a slight modification of) XPO, an iterative language model alignment algorithm due to Xie et al. (2024). XPO is based on the objective in Eq. (8), but unlike DPO, incorporates a bonus term to encourage exploration to leverage **online** interaction. See Appendix J.2 for a detailed overview. Theorem 4.3 (Informal version of Theorem J.2). Suppose that Assumptions 4.2 and 4.5 *hold with* parameters γmargin, Rmax > 0, and that Assumption 4.3 *holds with* β = γmargin/(2 log(2|Y|/δ)).

For any m ∈ N and ρ ∈ (0, 1), XPO *(Algorithm 1), when configured appropriately, produces an*
(ϵ, δ)-sharpened model πb ∈ Π with probability at least 1 − ρ*, and uses sample complexity*7

$$m={\tilde{O}}\left({\frac{{\mathsf{S E C}}(\Pi)\cdot\log(|\Pi|\rho^{-1})}{\gamma_{\mathrm{margin}}^{2}\delta^{2}\epsilon^{2}}}\right)$$
!
.
The takeaway from Theorem 4.3 is that there is no dependence on the coverage coefficient for πbase.

Instead, the rate depends on the complexity of exploration, as governed by the sequential extrapolation coefficient SEC(Π). We expect similar guarantees can derived for other active exploration algorithms and complexity measures (Jiang et al., 2017; Foster et al., 2021; Jin et al., 2021; Xie et al., 2023).

## 5 Conclusion

We view our theoretical framework for sharpening as a starting point toward a foundational understanding of self-improvement that can guide the design and evaluation of algorithms. To this end, we raise several directions for future research. - *Representation learning.* A conceptually appealing feature of our framework is that it is agnostic to the structure of the model under consideration, but an important direction for future work is to study the dynamics of self-improvement for specific models/architectures and understand the representations that these models learn under self-training.

- *Richer forms of self-reward.* Our theoretical results study the dynamics of self-training in a stylized framework where the model uses its own log-probabilities as a self-reward. Empirical research on self-improvement leverages more sophisticated approaches (e.g., specific prompting techniques) (Huang et al., 2022; Wang et al., 2022; Bai et al., 2022b; Pang et al., 2023; Yuan et al., 2024) and it is important to understand when and how these forms of self-improvement are beneficial.

## Acknowledgments

We thank Sivaraman Balakrishnan, Miro Dudík, Susan Dumais, John Langford, Qinghua Liu, and Yuda Song for helpful discussions.

## References

Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. *arXiv:2404.14219*, 2024.

Samira Abnar, Mostafa Dehghani, and Willem Zuidema. Transferring inductive biases through knowledge distillation. *arXiv:2006.00555*, 2020.

Alekh Agarwal, Peter L Bartlett, Pradeep Ravikumar, and Martin J Wainwright. Information-theoretic lower bounds on the oracle complexity of stochastic convex optimization. IEEE Transactions on Information Theory, 2012.

Alekh Agarwal, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. Taming the monster: A fast and simple algorithm for contextual bandits. In International Conference on Machine Learning, 2014.

Alekh Agarwal, Nan Jiang, Sham M Kakade, and Wen Sun. Reinforcement learning: Theory and algorithms. https://rltheorybook.github.io/, 2019. Version: January 31, 2022.

Zeyuan Allen-Zhu and Yuanzhi Li. Towards understanding ensemble, knowledge distillation and self-distillation in deep learning. *arXiv:2012.09816*, 2020.

Afra Amini, Tim Vieira, and Ryan Cotterell. Variational best-of-n alignment. *arXiv:2407.06057*,
2024.

Philip Amortila, Dylan J Foster, and Akshay Krishnamurthy. Scalable online exploration via coverability. In *Forty-first International Conference on Machine Learning*, 2024.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback.

arXiv:2204.05862, 2022a.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. *arXiv:2212.08073*, 2022b.

Francisco Barahona. On the computational complexity of ising spin glass models. Journal of Physics A: Mathematical and General, 1982.

Matthew James Beal. *Variational algorithms for approximate Bayesian inference*. University of London, University College London, 2003.

Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, and Yoshua Bengio. Flow network based generative models for non-iterative diverse candidate generation. Advances in Neural Information Processing Systems, 2021.

Yoav Benjamini and Yosef Hochberg. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 1995.

Adam Block, Dylan J Foster, Akshay Krishnamurthy, Max Simchowitz, and Cyril Zhang. Butterfly effects of SGD noise: Error amplification in behavior cloning and autoregression. *arXiv:2310.11428*, 2023.

Avrim Blum, Merrick Furst, Jeffrey Jackson, Michael Kearns, Yishay Mansour, and Steven Rudich.

Weakly learning DNF and characterizing statistical query learning using Fourier analysis. In Symposium on Theory of Computing, 1994.

Enric Boix-Adsera. Towards a theory of model distillation. *arXiv preprint arXiv:2403.09053*, 2024.

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv:2407.21787, 2024.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, 2020.

Cristian Bucilua, Rich Caruana, and Alexandru Niculescu-Mizil. Model compression. In ˇ *SIGKDD*
International Conference on Knowledge Discovery and Data Mining, 2006.

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models. *arXiv:2401.01335*, 2024.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. *Advances in Neural Information Processing* Systems, 2017.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. *arXiv:2110.14168*, 2021.

Stephen A Cook. The complexity of theorem-proving procedures. In Symposium on Theory of Computing, 1971.

Thomas M Cover. *Elements of information theory*. John Wiley & Sons, 1999. Rudrajit Das and Sujay Sanghavi. Understanding self-distillation in the presence of label noise. In International Conference on Machine Learning, 2023.

Rudrajit Das, Inderjit S Dhillon, Alessandro Epasto, Adel Javanmard, Jieming Mao, Vahab Mirrokni, Sujay Sanghavi, and Peilin Zhong. Retraining with predicted hard labels provably increases model accuracy. *arXiv:2406.11206*, 2024.

Jacob Devlin. Bert: Pre-training of deep bidirectional transformers for language understanding.

arXiv:1810.04805, 2018.

Bin Dong, Jikai Hou, Yiping Lu, and Zhihua Zhang. Distillation ≈ early stopping? Harvesting dark knowledge utilizing anisotropic information retrieval for overparameterized neural network.

arXiv:1910.01255, 2019.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv:2407.21783, 2024.

Ronen Eldan, Frederic Koehler, and Ofer Zeitouni. A spectral condition for spectral gap: Fast mixing in high-temperature Ising models. *Probability Theory and Related Fields*, 2022.

Amir-massoud Farahmand, Csaba Szepesvári, and Rémi Munos. Error propagation for approximate policy and value iteration. *Advances in Neural Information Processing Systems*, 2010.

Vitaly Feldman. A complete characterization of statistical query learning with applications to evolvability. *Journal of Computer and System Sciences*, 2012.

Vitaly Feldman. A general characterization of the statistical query complexity. In Conference on Learning Theory, 2017.

Dylan J Foster and Alexander Rakhlin. Foundations of reinforcement learning and interactive decision making. *arXiv:2312.16730*, 2023.

Dylan J Foster, Sham M Kakade, Jian Qian, and Alexander Rakhlin. The statistical complexity of interactive decision making. *arXiv:2112.13487*, 2021.

Spencer Frei, Difan Zou, Zixiang Chen, and Quanquan Gu. Self-training converts weak learners to strong learners in mixture models. In International Conference on Artificial Intelligence and Statistics, 2022.

Tommaso Furlanello, Zachary Lipton, Michael Tschannen, Laurent Itti, and Anima Anandkumar.

Born again neural networks. In *International Conference on Machine Learning*, 2018.

Zhaolin Gao, Jonathan D Chang, Wenhao Zhan, Owen Oertell, Gokul Swamy, Kianté Brantley, Thorsten Joachims, J Andrew Bagnell, Jason D Lee, and Wen Sun. REBEL: Reinforcement learning via regressing relative rewards. *arXiv:2404.16767*, 2024.

Samuel Gershman and Noah Goodman. Amortized inference in probabilistic reasoning. In Annual Meeting of the Cognitive Science Society, 2014.

Google. Palm 2 technical report. *arXiv:2305.10403*, 2023. Akhilesh Gotmare, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. A closer look at deep learning heuristics: Learning rate restarts, warmup and distillation. In International Conference on Learning Representations, 2019.

Yves Grandvalet and Yoshua Bengio. Semi-supervised learning by entropy minimization. *Advances* in Neural Information Processing Systems, 2004.

Lin Gui, Cristina Gârbacea, and Victor Veitch. BoNBoN alignment for large language models and the sweetness of best-of-n sampling. *arXiv:2406.00832*, 2024.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. *arXiv:2009.03300*, 2020.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv:2103.03874, 2021.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network.

arXiv:1503.02531, 2015.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. *arXiv:2106.09685*, 2021.

Edward J Hu, Moksh Jain, Eric Elmoznino, Younesse Kaddar, Guillaume Lajoie, Yoshua Bengio, and Nikolay Malkin. Amortizing intractable inference in large language models. *arXiv:2310.04363*, 2023.

Audrey Huang, Wenhao Zhan, Tengyang Xie, Jason D Lee, Wen Sun, Akshay Krishnamurthy, and Dylan J Foster. Correcting the mythos of KL-regularization: Direct alignment without overparameterization via Chi-squared Preference Optimization. *arXiv:2407.13399*, 2024.

Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han.

Large language models can self-improve. *arXiv:2210.11610*, 2022.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. *arXiv:2310.06825*, 2023.

Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire. Contextual decision processes with low Bellman rank are PAC-learnable. In *International Conference on* Machine Learning, 2017.

Chi Jin, Qinghua Liu, and Sobhan Miryoosefi. Bellman Eluder dimension: New rich classes of RL
problems, and sample-efficient algorithms. *Advances in Neural Information Processing Systems*, 2021.

Richard M Karp. *Reducibility among combinatorial problems*. Springer, 1972.

Michael Kearns. Efficient noise-tolerant learning from statistical queries. *Journal of the ACM*, 1998. Scott Kirkpatrick, C Daniel Gelatt Jr, and Mario P Vecchi. Optimization by simulated annealing.

Science, 1983.

Leonid Anatolevich Levin. Universal sequential search problems. *Problemy peredachi informatsii*,
1973.

Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. Chain of thought empowers transformers to solve inherently serial problems. *arXiv:2402.12875*, 2024.

Zhihan Liu, Miao Lu, Shenao Zhang, Boyi Liu, Hongyi Guo, Yingxiang Yang, Jose Blanchet, and Zhaoran Wang. Provably mitigating overoptimization in RLHF: Your SFT loss is implicitly an adversarial regularizer. *arXiv:2405.16436*, 2024.

László Lovász and Santosh Vempala. Fast algorithms for logconcave functions: Sampling, rounding, integration and optimization. In *Symposium on Foundations of Computer Science*, 2006.

Yi-An Ma, Yuansi Chen, Chi Jin, Nicolas Flammarion, and Michael I Jordan. Sampling can be faster than optimization. *Proceedings of the National Academy of Sciences*, 2019.

Eran Malach. Auto-regressive next-token predictors are universal learners. *arXiv:2309.06979*, 2023. Clara Meister, Tim Vieira, and Ryan Cotterell. If beam search is the answer, what was the question?

arXiv:2010.02650, 2020.

Hossein Mobahi, Mehrdad Farajtabar, and Peter Bartlett. Self-distillation amplifies regularization in hilbert space. *Advances in Neural Information Processing Systems*, 2020.

Sidharth Mudgal, Jong Lee, Harish Ganapathy, YaGuang Li, Tao Wang, Yanping Huang, Zhifeng Chen, Heng-Tze Cheng, Michael Collins, Trevor Strohman, et al. Controlled decoding from language models. *arXiv:2310.17022*, 2023.

Arkadii Nemirovski, David Borisovich Yudin, and Edgar Ronald Dawson. Problem complexity and method efficiency in optimization. Wiley, 1983.

OpenAI. GPT-4 technical report. *arXiv:2303.08774*, 2023. Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 2022.

Alizée Pace, Jonathan Mallinson, Eric Malmi, Sebastian Krause, and Aliaksei Severyn. West-of-n:
Synthetic preference generation for improved reward modeling. *arXiv:2401.12086*, 2024.

Jing-Cheng Pang, Pengyuan Wang, Kaiyuan Li, Xiong-Hui Chen, Jiacheng Xu, Zongzhang Zhang, and Yang Yu. Language model self-improvement by reinforcement learning contemplation. arXiv:2305.14483, 2023.

Divyansh Pareek, Simon S Du, and Sewoong Oh. Understanding the gains from repeated selfdistillation. *arXiv:2407.04600*, 2024.

Hieu Pham, Zihang Dai, Qizhe Xie, and Quoc V Le. Meta pseudo labels. In Conference on Computer Vision and Pattern Recognition, 2021.

Ori Press, Ravid Shwartz-Ziv, Yann LeCun, and Matthias Bethge. The entropy enigma: Success and failure of entropy minimization. *arXiv:2405.05012*, 2024.

Yuxiao Qu, Tianjun Zhang, Naman Garg, and Aviral Kumar. Recursive introspection: Teaching language model agents how to self-improve. *arXiv:2407.18219*, 2024.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 2023.

Maxim Raginsky and Alexander Rakhlin. Information-based complexity, feedback and dynamics in convex programming. *IEEE Transactions on Information Theory*, 2011.

Mamshad Nayeem Rizve, Kevin Duarte, Yogesh S Rawat, and Mubarak Shah. In defense of pseudolabeling: An uncertainty-aware pseudo-label selection framework for semi-supervised learning. arXiv:2101.06329, 2021.

Daniel Russo and Benjamin Van Roy. Eluder dimension and the sample complexity of optimistic exploration. In *Advances in Neural Information Processing Systems*, 2013.

Abulhair Saparov and He He. Language models are greedy reasoners: A systematic formal analysis of chain-of-thought. In *International Conference on Learning Representations*, 2023.

Igal Sason and Sergio Verdú. f-divergence inequalities. *IEEE Transactions on Information Theory*,
2016.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv:1707.06347*, 2017.

Pier Giuseppe Sessa, Robert Dadashi, Léonard Hussenot, Johan Ferret, Nino Vieillard, Alexandre Ramé, Bobak Shariari, Sarah Perrin, Abe Friesen, Geoffrey Cideron, et al. Bond: Aligning LLMs with Best-of-N distillation. *arXiv:2407.14622*, 2024.

Max Simchowitz, Kevin Jamieson, and Benjamin Recht. The simulator: Understanding adaptive sampling in the moderate-confidence regime. In *Conference on Learning Theory*, 2017.

Mohit Singh and Nisheeth K Vishnoi. Entropy, optimization and counting. In Symposium on Theory of Computing, 2014.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling model parameters. *arXiv:2408.03314*, 2024.

Yuda Song, Gokul Swamy, Aarti Singh, J Andrew Bagnell, and Wen Sun. Understanding preference fine-tuning through the lens of coverage. *arXiv:2406.01462*, 2024.

Kevin Swersky, Yulia Rubanova, David Dohan, and Kevin Murphy. Amortized bayesian optimization over discrete spaces. In *Conference on Uncertainty in Artificial Intelligence*, 2020.

Kunal Talwar. Computational separations between sampling and optimization. Advances in Neural Information Processing Systems, 32, 2019.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv:2307.09288, 2023.

Joseph F Traub, Grzegorz W Wasilkowski, and Henryk Wo´zniakowski. *Information-based complexity*.

Academic Press Professional, Inc., 1988.

S. A. van de Geer. *Empirical Processes in M-Estimation.* Cambridge University Press, 2000. Ziyu Wan, Xidong Feng, Muning Wen, Stephen Marcus McAleer, Ying Wen, Weinan Zhang, and Jun Wang. Alphazero-like tree-search can guide large language model decoding and training. International Conference on Machine Learning, 2024.

Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully test-time adaptation by entropy minimization. *arXiv:2006.10726*, 2020.

Tianlu Wang, Ilia Kulikov, Olga Golovneva, Ping Yu, Weizhe Yuan, Jane Dwivedi-Yu, Richard Yuanzhe Pang, Maryam Fazel-Zarandi, Jason Weston, and Xian Li. Self-taught evaluators. arXiv:2408.02666, 2024.

Xuezhi Wang and Denny Zhou. Chain-of-thought reasoning without prompting. *arXiv:2402.10200*,
2024.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. arXiv:2212.10560, 2022.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 2022.

Wing Hung Wong and Xiaotong Shen. Probability inequalities for likelihood ratios and convergence rates of sieve mles. *The Annals of Statistics*, 1995.

Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, and Sainbayar Sukhbaatar. Meta-rewarding language models: Self-improving alignment with llm-as-a-meta-judge. *arXiv:2407.19594*, 2024a.

Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. An empirical analysis of compute-optimal inference for problem-solving with language models. *arXiv:2408.00724*, 2024b.

Yue Wu, Zhiqing Sun, Huizhuo Yuan, Kaixuan Ji, Yiming Yang, and Quanquan Gu. Self-play preference optimization for language model alignment. *arXiv:2405.00675*, 2024c.

Tengyang Xie and Nan Jiang. Q* approximation schemes for batch reinforcement learning: A
theoretical comparison. In *Conference on Uncertainty in Artificial Intelligence*, 2020.

Tengyang Xie, Dylan J Foster, Yu Bai, Nan Jiang, and Sham M Kakade. The role of coverage in online reinforcement learning. In *International Conference on Learning Representations*, 2023.

Tengyang Xie, Dylan J Foster, Akshay Krishnamurthy, Corby Rosset, Ahmed Awadallah, and Alexander Rakhlin. Exploratory preference optimization: Harnessing implicit Q*-approximation for sample-efficient RLHF. *arXiv:2405.21046*, 2024.

Wei Xiong, Hanze Dong, Chenlu Ye, Han Zhong, Nan Jiang, and Tong Zhang. Gibbs sampling from human feedback: A provable KL-constrained framework for RLHF. *arXiv:2312.11456*, 2023.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan.

Tree of thoughts: Deliberate problem solving with large language models. *Advances in Neural* Information Processing Systems, 2024.

Chenlu Ye, Wei Xiong, Yuheng Zhang, Nan Jiang, and Tong Zhang. A theoretical analysis of Nash learning from human feedback under general KL-regularized preference. *arXiv:2402.07314*, 2024.

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. Self-rewarding language models. *arXiv:2401.10020*, 2024.

Andrea Zanette, Martin J Wainwright, and Emma Brunskill. Provable benefits of actor-critic methods for offline reinforcement learning. *Advances in Neural Information Processing Systems*, 2021.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. *Advances in Neural Information Processing Systems*, 2022.

Tong Zhang. From ϵ-entropy to KL-entropy: Analysis of minimum information complexity density estimation. *The Annals of Statistics*, 2006.

Stephen Zhao, Rob Brekelmans, Alireza Makhzani, and Roger Baker Grosse. Probabilistic inference in language models via twisted sequential monte carlo. *International Conference on Machine* Learning, 2024.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging LLM-as-a-judge with MT-bench and chatbot arena. *Advances in Neural Information Processing Systems*, 2024.

Banghua Zhu, Michael Jordan, and Jiantao Jiao. Principled reinforcement learning with human feedback from pairwise or k-wise comparisons. In *International Conference on Machine Learning*, 2023.

CONTENTS OF APPENDIX

| I   | Additional Discussion and Results                           | 19   |    |
|-----|-------------------------------------------------------------|------|----|
| A   | Additional Experiments and Details                          | 19   |    |
| A.1 | Inference-time validation experiments                       |      | 19 |
| A.2 | Experiments with other self reward functions                |      | 23 |
| A.3 | Effect of SFT-Sharpening                                    |      | 25 |
| B   | Detailed Discussion of Related Work                         | 26   |    |
| C   | Guarantees for Inference-Time Sharpening                    | 29   |    |
| D   | Guarantees for SFT-Sharpening with Adaptive Sampling        | 30   |    |
| E   | Computational and Representational Challenges in Sharpening | 31   |    |
| E.1 | Computational Challenges                                    |      | 32 |
| E.2 | Representational Challenges                                 |      | 32 |
| II  | Proofs                                                      | 34   |    |
| F   | Preliminaries                                               | 34   |    |
| F.1 | Guarantees for Approximate Maximizers                       |      | 34 |
| F.2 | Technical Tools                                             | 34   |    |
| G   | Proofs from Section 3.1                                     | 36   |    |
| H   | Proofs from Section 3.3                                     | 36   |    |
| I   | Proofs from Section 4.1 and Appendix D                      | 39   |    |
| J   | Proofs from Section 4.2                                     | 42   |    |
| J.1 | Proof of Theorem 4.2                                        | 42   |    |
| J.2 | Proof of Theorem 4.3 and Theorem J.3                        | 47   |    |

# Part I Additional Discussion And Results

## A Additional Experiments And Details

In this section we detail the precise setup required to replicate our empirical results. All of our experiments were run either on 40G NVIDIA A100 GPUs, 192G AMD MI300X GPUs, or through the OpenAI API. We considered the following models. All models, except for gpt-3.5-turbo-instruct, are available on https://huggingface.co and we provide HuggingFace model identifiers below.

1. Phi models: We experiment with several models from the Phi family of models (Abdin et al.,
2024), specifically Phi3-Mini ("microsoft/Phi-3-mini-4k-instruct"), Phi3-Small ("microsoft/Phi3-small-8k-instruct"), Phi3-Medium ("microsoft/Phi-3-medium-4k-instruct"), and Phi3.5-Mini ("microsoft/Phi-3.5-mini-instruct").

2. Llama3.2-3B-Instruct ("meta-llama/Llama-3.2-3B-Instruct") (Dubey et al., 2024)
3. Mistral-7B-Instruct-v0.3 ("mistralai/Mistral-7B-Instruct-v0.3") (Jiang et al., 2023)
4. gpt-3.5-turbo-instruct (Brown et al., 2020): We access this model via the OpenAI API. 5. llama2-7b-game24-policy-hf ("OhCherryFire/llama2-7b-game24-policy-hf"): We use the model of Wan et al. (2024), which is a Llama-2 model finetuned on the GameOf24 task (Yao et al., 2024). We use this model only the GameOf24 task.

We consider the following tasks:
1. MATH: We use the above models to generate responses to prompts from the MATH (Hendrycks et al.,
2021), which consists of more difficult math questions. We consider "all" subsets and take the first 256 examples of the test set where the solution matches the regular expression (\d*).

8 2. GSM8k: We use the above models to generate responses to prompts from the GSM-8k dataset
(Cobbe et al., 2021) where the goal is to generate a correct answer to an elementary school math question. We take the first 256 examples from the test set in the main subset.9 3. ProntoQA: We use the above models to generate responses to prompts from the ProntoQA dataset
(Saparov & He, 2023), which consists of chain-of-thought-style reasoning questions with boolean answers. We take the first 256 examples from the training set.10 4. MMLU: We use the above models to generate responses to prompts from three subsets of the MMLU
dataset (Hendrycks et al., 2020), specifically college_biology (Bio),college_physics (Phys),
and college_chemistry (Chem) all of which consist of multiple choice questions11. We take the first 256 examples of the test set.

5. GameOf24: We use only the model of Wan et al. (2024) (i.e., llama2-7b-game24-policy-hf),
on the GameOf24 task (Yao et al., 2024). The prompts are four numbers and the goal is to combine the numbers with standard arithmetic operations to reach the number '24.' Here we use both the train and test splits of the dataset.12

## A.1 Inference-Time Validation Experiments

To form the plots in Figure 1 and in Figures 3 and 4, for each (model, task) pair, we sampled N generations per prompt with temperature 1 and returned the best of the N generations according to the maximum-likelihood sharpening self-reward function rself(y | x) = log πbase(y | x); we compare against greedy decoding as a baseline, whose accuracy is displayed in Figure 2(d).

BoN-Norm: % Lift over Greedy
-0.4
-1.2 Phi3 (Mini)
4.4 7.0 7.2 5.2 5.0 0.7 Phi3.5 (Mini)
0.8 1.5 6.0 4.9 Phi3 (Small)
2.1 7.3
-4.6 3.3 2.4
-11.8 Phi3 (Medium)
0.1
-1.7 0.5 0.2 6.5
-1.3 Mistral-7B
28.6 17.1 2.8 7.6 25.9 4.0 Llama3.2-3B
4.3 1.3 3.6
-12.7 79.3 20.3 GPT-3.5 3.4 1.2
-5.9
-1.8 7.1
-7.9 GMBK
phy MATH
Chom
(a)
Phi3 (Mini)
19.7 5.1 Phi3.5 (Mini)
19.4 8.7 Phi3 (Small)
17.1 1.4 Phi3 (Medium)
16.0 6.1 Mistral-7B
72.2 48.8 Lama3.2-3B
17.5 11.6 GPT-3.5 35.5 18.2 M
Majority: % Lift over Greedy 0.3 7.1 12.5 8.7 1.5 1.2 1.1 6.9
-14.4 7.9
-9.4 2.5 5.0 3.6 13.2 1.3 38.5 19.8 19.5 10.5 9.6 1.5 9.3 59.7 7.4 4.6 7.0 1.0 CMBK
phy?

Chem Pass@50: Accuracy (%)
Phi3 (Mini)
96.6 97.3 98.8 99.2 99.9 98.1 Phi3.5 (Mini)
97.0 96.8 90.9 97.5 98.8 96.7 Phi3 (Small)
97.9 97.3 90.3 98 97.2 96.3 Phi3 (Medium)
98.4 98.3 9.0 9.0 94.7 81.9 Mistral-7B
94.0 9.7 98.3 98.2 98.0 Llama3.2-3B
93.8 95.8 100.0 99.6 9.8 9.2 GPT+3.5 96.0 96.6 95.1 98.5 9.9 9.4 MATH GSMBK Provided py" Chom
(c)
(b)
Greedy: Accuracy (%)
Phi3 (Mini)
6.0 87.1 50.4 80.6 65.7 52.0 50.8 Phi3.5 (Mini)
67.6 84.4 7.1 68.6 55.0 72.3 Phi3 (Small)
79.3 59.4 85.4 74.5 66.0 Phi3 (Medium)
73.4 47.3 60.0 86.7 8.2 70.6 Mistral-7B
23.0 46.9 50.0 61.8 36.3 42.0 Lama3.2-3B
58.2 76.6 47.7 61.8 14.7 30.0 GPT-3.5 55.9 70.3 49.6 68.8 51.0 53.0 MATH
CMBK Prone Bro py chem
(d)
Implementation details. 

For all models and datasets except for GameOf24, we used 1-shot prompting to ensure that models conform to the desired output format and to elicit chain of thought reasoning
(for GameOf24 we do not provide a demonstration in the prompt). We set the maximum length of decoding to be 512 tokens. We used 10 seeds for all (model, task) pairs with a maximum value of N = 50 in Best-of- N sampling. We simulated N responses for N < 50 by subsamplng the 50 generated samples. For Best-of- N sampling, we always use temperature 1.0. Since greedy decoding is a deterministic strategy, we only use 1 seed for each (model, task) pair. In all experiments, we collect both the responses and their log-likelihoods under the reference model (i.e., the original model from which samples were generated).

Results for most datasets are presented in Figures 3 and 4. Because we only consider Results.

a single model for GameOf24, we separate this task into Figure 5 For all datasets, we visualize both performance—measured as normalized improvement in accuracy over greedy decoding—and log-likelihoods—under π base —of the selected responses. In all cases, Best-of- N sampling (using r self ( Y | x ) = log π base ( Y | x )) improves over the naïve sampling strategy, wherein we simply sample a single generation with temperature 1.0. In all datasets, we also see improvements over the standard greedy decoding strategy, at least for some models. Analogously, for every model, there is at least one dataset for which Best-of- N sampling improves over greedy decoding.

We further explore the relationship between sequence level log probabilities and generation quality in Figure 6, where we plot the empirical distributions of responses sampled with temperature 1 from