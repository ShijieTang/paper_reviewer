## How Do Large Language Monkeys Get Their Power (Laws)?

Rylan Schaeffer <sup>1</sup> Joshua Kazdan <sup>2</sup> John Hughes 3 4 Jordan Juravsky <sup>1</sup> Sara Price <sup>4</sup> Aengus Lynch 4 5 Erik Jones <sup>6</sup> Robert Kirk <sup>5</sup> Azalia Mirhoseini <sup>1</sup> Sanmi Koyejo <sup>1</sup>

## Abstract

Recent research across mathematical problem solving, proof assistant programming and multimodal jailbreaking documents a striking finding: when (multimodal) language model tackle a suite of tasks with multiple attempts per task – succeeding if any attempt is correct – then the negative log of the average success rate scales a power law in the number of attempts. In this work, we identify an apparent puzzle: a simple mathematical calculation predicts that on each problem, the failure rate should fall exponentially with the number of attempts. We confirm this prediction empirically, raising a question: from where does aggregate polynomial scaling emerge? We then answer this question by demonstrating per-problem exponential scaling can be made consistent with aggregate polynomial scaling if the distribution of singleattempt success probabilities is heavy tailed such that a small fraction of tasks with extremely low success probabilities collectively warp the aggregate success trend into a power law - even as each problem scales exponentially on its own. We further demonstrate that this distributional perspective explains previously observed deviations from power law scaling, and provides a simple method for forecasting the power law exponent with an order of magnitude lower relative error, or equivalently, ∼<sup>2</sup> − <sup>4</sup> orders of magnitude less inference compute. Overall, our work contributes to a better understanding of how neural language model performance improves with scaling inference compute and the development of scaling-predictable evaluations of (multimodal) language models.

## 1. Introduction

Scaling behaviors of large neural language models have surprised and fascinated engineers, scientists and society alike [\(Hestness et al.,](#page-12-0) [2017;](#page-12-0) [Kaplan et al.,](#page-13-0) [2020;](#page-13-0) [Brown](#page-9-0) [et al.,](#page-9-0) [2020a;](#page-9-0) [Hoffmann et al.,](#page-12-1) [2022;](#page-12-1) [Ganguli et al.,](#page-10-0) [2022;](#page-10-0) [Sorscher et al.,](#page-15-0) [2022;](#page-15-0) [Wei et al.,](#page-21-0) [2022b;](#page-21-0) [Schaeffer et al.,](#page-15-1) [2023;](#page-15-1) [OpenAI et al.,](#page-13-1) [2024\)](#page-13-1), shaping engineering, economic and governmental interests in frontier AI systems [\(Bom](#page-9-1)[masani et al.,](#page-9-1) [2021;](#page-9-1) [Eloundou et al.,](#page-10-1) [2023;](#page-10-1) [Anderljung et al.,](#page-9-2) [2023;](#page-9-2) [Wang et al.,](#page-21-1) [2023;](#page-21-1) [Reuel et al.,](#page-14-0) [2024;](#page-14-0) [Besiroglu et al.,](#page-9-3) [2024a;](#page-9-3) [Maslej et al.,](#page-13-2) [2024\)](#page-13-2). For a more thorough exposition of relevant literature, please see Related Work (Section [6\)](#page-7-0).

One direction of renewed interest is inference-time compute scaling, whereby compute is controllably increased at inference to improve the performance of a model, e.g., [Pachocki](#page-14-1) [et al.](#page-14-1) [\(2024\)](#page-14-1). In this direction, recent research discovered that language model success rates scale predictably with the number of independent attempts made at accomplishing a task. Specifically, in a paper titled, "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling," [Brown et al.](#page-9-4) [\(2024\)](#page-9-4) studied how language model performance changes at mathematical problem solving and coding problems when k independent attempts are sampled per problem. Performance on the i-th problem was measured using the expected (over attempts) success rate [\(Kulal](#page-13-3) [et al.,](#page-13-3) [2019;](#page-13-3) [Chen et al.,](#page-10-2) [2021\)](#page-10-2), defined as:

$$\text{pass}_i \oplus k \stackrel{\text{def}}{=} \left[ \text{II}[\text{Any attempt on } i\text{-th problem succeeds}] \right]. \quad (1)$$

Using the unbiased and numerically stable estimator of [Chen et al.](#page-10-2) [\(2021\)](#page-10-2) (for details, see Appendix [B\)](#page-24-0), [Brown](#page-9-4) [et al.](#page-9-4) [\(2024\)](#page-9-4) found that the negative log averaged-over-Pproblems success rate falls as a power law with the number of independent attempts per problem k:

$$-\log \left( \frac{1}{P} \sum_{i=1}^P \text{pass}_i @k \right) \approx a k^{-b}, \quad (2)$$

for model-specific and benchmark-specific constants a, b > 0 (Fig. [1](#page-1-0) Top). Soon after, on a separate topic of jailbreaking multimodal language models via text, image and audio

<sup>1</sup> Stanford Computer Science <sup>2</sup> Stanford Statistics 3 Speechmatics <sup>4</sup>ML Alignment & Theory Scholars <sup>5</sup>University College London <sup>6</sup>Anthropic. Correspondence to: Rylan Schaeffer <rschaef@cs.stanford.edu>, Sanmi Koyejo <sanmi@cs.stanford.edu>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

![](_page_1_Figure_1.jpeg)

Figure 1: Power Law Scaling in Language Models from Repeat Sampling. Top: [Brown et al.](#page-9-4) [\(2024\)](#page-9-4) found the negative log average pass rate − log(passD@k) at solving mathematical problems scales polynomially (i.e., as a power law) with the number of independent attempts per problem k. Bottom: [Hughes et al.](#page-12-2) [\(2024\)](#page-12-2) similarly found the negative log average attack success rate − log(ASRD@k) when jailbreaking multimodal language models scales polynomially with the number of jailbreak attempts per prompt. Should such power law scaling be expected? From where do large language monkeys obtain their power (laws)?

attacks, independent work by [Hughes et al.](#page-12-2) [\(2024\)](#page-12-2) studied jailbreaking success rates when k independent attempts are made per harmful prompt. Performance was measured using Attack Success Rate (ASR) at k:

$$\text{ASR}_i @k \stackrel{\text{def}}{=} \mathbb{E}_{k \text{ attempts}} \left[ \mathbb{I}[\text{Any attack on } i\text{-th prompt succeeds}] \right]. \quad (3)$$

This "Best-of-N Jailbreaking" attack similarly discovered that the negative log averaged-over-P-prompts attack success rate fell as a power law with the number of jailbreak attempts per prompt k:

$$-\log \left( \frac{1}{P} \sum_{i=1}^P \text{ASR}_i @k \right) \approx ak^{-b}, \quad (4)$$

for model-specific and modality-specific constants a, b > 0 (Fig. [1](#page-1-0) Bottom). For the specific coefficients from both papers, see Appendix. [C.](#page-25-0) As a minor matter of terminology, both papers frame their results in terms of "coverage" – the fraction of problems that can be solved after k attempts per problem – but as [Brown et al.](#page-9-4) [\(2024\)](#page-9-4) pointed out, coverage is equivalent to the average success rate (Appendix [D\)](#page-26-0); we prefer this latter framing as it avoids the binary implication that each problem either is or is not solved after k attempts.

## 2. Should Power Law Scaling Be Expected?

Should we expect large language monkeys to have such power (laws)? That is, should the negative log of the average success rate scale polynomially with the number of independent attempts k? As we now explain mathematically and demonstrate empirically, such polynomial scaling with k is perhaps surprising because, for any single problem, the negative log success rate at k should fall exponentially with k; the intuition is that passi@k is 1 unless *all* attempts fail, and since attempts are independent, the probability that all fail is exponentially unlikely with the number of attempts.

Mathematically, on any given attempt, the model has probability passi@1 of solving the i-th problem. Recalling that passi@k is defined as 1 if *any* of the k attempts succeed, 0 otherwise, by linearity of expectation and by independence of the k attempts, we can rewrite passi@k as:

$$\text{pass}_1 @k = \mathbb{E}_{k \text{ Attempts}} \left[ 1 - \mathbb{I}[\text{All } k \text{ Attempts Fail}] \right] \quad (5)$$

$$\begin{aligned} \text{pass}_i @k &= \mathbb{E}_{k \text{ Attempts}} \left[ 1 - \mathbb{I}[\text{All } k \text{ Attempts Fail}] \right] \\ &= 1 - \prod_{j=1}^k \mathbb{E}_{1 \text{ Attempt}} \left[ \mathbb{I}[j\text{-th Attempt Fails}] \right]. \end{aligned} \quad (5)$$

The probability that the j-th attempt fails is one minus the probability that the j-th attempt succeeds. Since each attempt is i.i.d. with success probability passi@1, we find

$$\text{pass}_i @k = 1 - (1 - \text{pass}_i @1)^k. \quad (7)$$

For large <sup>k</sup>, (1 − passi@1)<sup>k</sup> will be small. Recalling that the Taylor Series expansion of P log(1 + x) for small x is ∞ <sup>i</sup>=1(−1)<sup>i</sup>−<sup>1</sup><sup>x</sup> <sup>i</sup>/i ≈ <sup>x</sup>, we have:

$$\begin{aligned} -\log(\text{pass}_1 @k) &= -\log\left(1 - (1 - \text{pass} @1)^k\right) & (8) \\ &\approx (1 - \text{pass}_1 @1)^k. & (9) \end{aligned}$$

$$\approx (1 - \text{pass}_i @ 1)^k. \quad (9)$$

Thus, *for any single problem*, we should expect the negative log expected (over attempts) success rate to fall *exponentially* with k, not polynomially with k.

To confirm this claim, we plotted the scaling of model performance on each problem – measured either by − log(passi@k) or by − log(ASRi@k) – against the number of independent attempts k. We specifically used [Brown](#page-9-4)

![](_page_2_Figure_1.jpeg)

Figure 2: Schematic: The Origin of Power Laws from Scaling Inference Compute via Repeat Sampling. The − log(passD@k) scales as a power law with the number of attempts per problem <sup>k</sup> (left). This arises from a combination of two factors: (1) for each problem, − log(passi@k) scales exponentially with <sup>k</sup> (center), and (2) the distribution (over problems in the dataset) of single-attempt success rates passi@1 itself has a left power-law tail of small values (right).

[et al.](#page-9-4) [\(2024\)](#page-9-4)'s data of the Pythia language model family [\(Biderman et al.,](#page-9-5) [2023\)](#page-9-5) solving 128 mathematical problems from MATH [Hendrycks et al.](#page-12-3) [\(2021\)](#page-12-3) as well as [Hughes](#page-12-2) [et al.](#page-12-2) [\(2024\)](#page-12-2)'s data from jailbreaking frontier AI systems – Claude, GPT4 [\(OpenAI et al.,](#page-13-1) [2024\)](#page-13-1), Gemini [\(Team et al.,](#page-16-0) [2024a](#page-16-0)[;b\)](#page-19-0) and Llama 3 8B Instruction Tuned (IT) [\(Grattafiori](#page-11-0) [et al.,](#page-11-0) [2024\)](#page-11-0) – on 159 prompts from HarmBench [\(Mazeika](#page-13-4) [et al.,](#page-13-4) [2024\)](#page-13-4). For each individual mathematical problem and jailbreaking prompt, we found the negative log expected (over attempts) success rates fall exponentially with k as expected (Fig. [3\)](#page-3-0), including on Llama 3 8B IT which does not exhibit an aggregate power law (Fig. [1\)](#page-1-0).

## 3. Distribution of Per-Problem Single-Attempt Success Rates Creates Power Law Scaling

How does polynomial scaling of the negative log *average* success rate emerge from exponential scaling of the negative log *per-problem* success rate? The answer to this question *must* lie in the distribution D over benchmark problems of single attempt (i.e., k = 1) success rates because this distribution's density pD(passi@1) links the per-problem scaling behavior to the aggregate scaling behavior via the definition of the aggregate success rate passD@k:

$$\begin{aligned} \text{pass}_{\mathcal{D}} @k &\stackrel{\text{def}}{=} \mathbb{E}_{\text{pass}_i @1 \sim \mathcal{D}} \left[ \text{pass}_i @k(\text{pass}_i @1) \right] \\ &= 1 - \int_0^1 (1 - \text{pass}_i @1)^k p_{\mathcal{D}}(\text{pass}_i @1) \, d\text{pass}_i @1. \end{aligned} \quad (10)$$

Based on a known result that power laws can originate from an appropriately weighted sum of exponential functions (Appendix [E.1\)](#page-27-0), we begin by considering simple distributions for the single-attempt success probabilities and asking which yield power law scaling between − log(passD@k)

and k, as well as what properties of the distributions set the scaling exponent. In Appendices [E.3-](#page-27-1)[E.8,](#page-33-0) we derive that several simple distributions yield power law scaling with different exponents whereas others do not:

- - 
  $$\log \left( \text{pass}_{\text{Uniform}(0, \beta \leq 1)} @k \right) \propto k^{-1}.$$
- -  $\log \left( \text{pass}_{\text{Beta}(\alpha, \beta)} @k \right) \propto k^{-\alpha}.$
- -  $\log \left( \text{pass}_{\text{Kumaraswamy}(\alpha, \beta)} @k \right) \propto k^{-\alpha}.$
- -  $\log \left( \text{pass}_{\text{ContinuousBernoulli}(\lambda < 1/2)} @k \right) \propto k^{-1}.$
- -  $\log \left( \text{pass}_{\text{Reciprocal}(0 < \alpha < \beta < 1)} @k \right) \propto \frac{(1 - \alpha)^k}{k}.$

To test this understanding, we examined whether the data of [Brown et al.](#page-9-4) [\(2024\)](#page-9-4) and [Hughes et al.](#page-12-2) [\(2024\)](#page-12-2) had per-problem single-attempt success rate distributions that matched one of these simple distributions (Fig. [4\)](#page-4-0). We found that the distributions could indeed be well fit by a 3-parameter Kumaraswamy(α, β, a = 0, c) distribution with scale parameter c (Fig. [4,](#page-4-0) black dashed lines); we found the scale parameter was critical to obtain good fits because the standard 2-parameter Kumaraswamy distribution is supported on (0, 1) whereas most single-attempt success distributions have a smaller maximum such as 0.01 or 0.1.

More generally, what are the distributional properties that create such power law scaling and that set the specific power law exponent? As we now show, the negative log average success rate will exhibit power law scaling in k with exponent b if and only if the distribution over problems of single-attempt success probabilities itself behaves like a power law near <sup>0</sup> with exponent <sup>b</sup> − <sup>1</sup>:

Theorem 3.1 (Sufficiency of Power-Law Left Tail in Dis-

![](_page_3_Figure_1.jpeg)

Figure 3: Per-problem performance scales exponentially with the number of attempts per problem k. Top: Pythia language models on 128 problems from MATH, with performance on the <sup>i</sup>-th problem measured as − log(passi@k). Bottom: Frontier AI models on jailbreaking prompts from HarmBench, with performance on the i-th problem measured as − log(ASRi@k). In both settings, on each problem, the negative log *per-problem* success rate falls exponentially with the number of independent attempts k. However, the negative log *average* success rate falls as a power law with k (black).

tribution of Single-Attempt Success Rates). *Let* D *be a probability distribution on* [0, 1] *with PDF* pD(passi@1)*. Suppose there exist constants* b > 0*,* C > 0*,* θ > 0 *and* δ > 0 *such that, for all* 0 < passi@1 < δ*, we have*

$$p_D(\text{pass}_i @ 1) = C \cdot (\text{pass}_i @ 1)^{b-1} + O((\text{pass}_i @ 1)^{b-1+\theta}).$$
Then, for large  $k$ ,

*Then, for large* k*,*

– 
$$\log \left( \text{pass}_{\mathcal{D}} @k \right) \sim C \Gamma(b) k^{-b}$$
.

Theorem 3.2 (Necessity of Power-Law Left Tail in Distribution of Single-Attempt Success Rates). *Let* D *be a*

*distribution over* passi@1 ∈ [0, 1] *with PDF* <sup>p</sup>D(passi@1)*. Suppose there exist constants* b > 0 *and* A > 0 *such that for large* k*,*

− log passD@k *Then, under mild regularity assumptions, the probability density must satisfy*

$$-\log\left(\text{pass}_{\mathcal{D}} @k\right) \sim A k^{-b}.$$

$$p_{\mathcal{D}}(\text{pass}_i @1) \sim \frac{A}{\Gamma(b)} (\text{pass}_i @1)^{b-1} \quad \text{as} \quad \text{pass}_i @1 \rightarrow 0^+.$$

![](_page_4_Figure_2.jpeg)

Figure 4: Single-Attempt Success Rates Distributions Possess Power Law-Like Left Tails. Pythia language models on 128 MATH problems (top) and frontier AI systems on 159 HarmBench prompts (bottom) exhibit distributions (over problems) of passi@1 and ASRi@1 with power law-like tails that are well fit by scaled Beta-Binomial distributions (black dashed lines), which produce aggregate power law scaling. Note that Llama 3 8B Instruction Tuned (IT) does not possess a power law tail, explaining why the model did not exhibit aggregate power law scaling under Best-of-N jailbreaking (Sec. [4\)](#page-5-0).

that whenever − log(passD@k) exhibits power-law decay in k with exponent b, the distribution over problems of single-attempt success rates *must* have "polynomial weight" near passi@1 = 0, i.e. pD(p) = Θ(p b−1 ).

To offer intuition, we know that each problem is being solved by the model (or equivalently, each prompt is jailbreaking the model) exponentially quickly. If one looks across all problems in the benchmark, some have passi@1 so small that they remain unsolved for many, many attempts. Whether these "tiny-passi@1" problems still matter at large k depends on how *many* such problems there are. Polynomial density near 0 "piles up" enough hard problems in just the right way such that even though each of those problems is being solved exponentially quickly, the *aggregate* success rate over problems decreases at only a power-law rate in k. A more succinct mathematical summary is that, for a compound binomial distribution, the lower tail probability controls the upper tail of the marginal survivor function.

![](_page_5_Figure_1.jpeg)

Figure 5: Schematic: Two Estimators of Power Law Parameters for Scaling Inference Compute via Repeat Sampling. (A) Both estimators begin by generating many samples per prompt, then computing the number of successes per prompt. In the standard least squares power law parameter estimator (top), (B) passi@k is estimated for each i-th problem at multiple k values, then (C) averaged over problems and fit with linear regression in log-log space. In the distributional power law parameter estimator (bottom), (D) a distribution D is fit to estimates of passi@1, then (E) the single-attempt success probability distribution is used to simulate passD@k at arbitrary k values for linear regression in log-log space.

## 4. Lack of Distributional Structure Explains Deviations from Power Law Scaling

Notably, previous papers observed that not every model exhibits power law scaling in every setting. To highlight one, [Hughes et al.](#page-12-2) [\(2024\)](#page-12-2) observed that when jailbreaking Meta's Llama 3 8B Instruction Tuned (IT) model [\(Grattafiori et al.,](#page-11-0) [2024\)](#page-11-0), the − log(ASRD@k) fell faster than any power law (Fig. [1\)](#page-1-0), i.e., the ASRD@k rose much more quickly than the other frontier AI systems. Based on our mathematical insights and the empirical per-problem single-attempt attack success rates (Fig. [4\)](#page-4-0), we can understand why: Llama 3 8B IT could be successfully jailbroken on every prompt within the permitted sampling budget and thus had no heavy left tail necessary to create the aggregate power law scaling.

## 5. A New Distributional Estimator for Predicting Power Law Scaling

A natural consequence of this connection between the scaling of − log(passD@k) and the left tail of the distribution pD(passi@1) is that the distribution of single-attempt success rates can be used to predict whether power-law scaling will appear and if so, what the intercept and exponent of the power law will be. To do this, one can fit the distribution

pˆD(passi@1) and then *simulate* how passD@k will scale with k (Fig. [5\)](#page-5-1) using the relationship:

$$\text{pass}_{\mathcal{D}} \widehat{\oplus} k \stackrel{\text{def}}{=} 1 - \int_0^1 (1 - \text{pass}_i @ 1)^k \hat{p}_{\mathcal{D}}(\text{pass}_i @ 1) d\text{pass}_i @ 1. \quad (11)$$

To empirically test this claim, we compared the standard least squares regression estimator (in log-log space) [\(Hoff](#page-12-1)[mann et al.,](#page-12-1) [2022;](#page-12-1) [Caballero et al.,](#page-10-3) [2022;](#page-10-3) [Besiroglu et al.,](#page-9-6) [2024b\)](#page-9-6) against a *distributional estimator*. To motivate our distributional estimator, we first need explain a key obstacle and how the distributional estimator overcomes it. The obstacle is that there are problems or prompts whose single-attempt success probabilities passi@1 lie between (0, 1/Number of Samples) such that, due to finite sampling, we lack the resolution to measure. While we do not know the true single-attempt success probability for the problems that lie in this interval, we *do know how many problems fall into this left tail bucket*, and we can fit a distribution's parameters such that the distribution's probability mass in the interval (0, 1/Number of Samples) matches the empirical fraction of problems in this tail bucket. Thus, our distributional estimator works by first selecting a distribution (e.g., a scaled 3-parameter Beta distribution), discretizing the distribution

![](_page_6_Figure_2.jpeg)

Figure 6: Comparing Estimators of Power Law Exponents. We compare two estimators of the power law exponent b in − log(passD@k) ≈ ak−<sup>b</sup> : (1) the standard least-squares estimator between <sup>k</sup> and − log(passD@k) in log-log space, and (2) the distributional estimator of passi@1 assuming a scaled Kumaraswamy-Binomial distribution. Using all available data to fit both estimators, we find agreement between the least-squares estimate (ordinate) and the distribution-derived estimate (abscissa) for both Pythia models on MATH (left) and for frontier AI systems on HarmBench (right). For an explanation of why the two estimators match more closely for Large Language Monkeys than for Best-of-N Jailbreaking, see Appendix [A.](#page-23-0)

![](_page_6_Figure_4.jpeg)

Figure 7: Comparing Two Estimators of Power Law Exponents via Backtesting. On synthetic data with known ground-truth power law a k−<sup>b</sup> , we compare how well the least squares and the distributional estimator recover the scaling exponent <sup>b</sup> as measured by the relative error | <sup>ˆ</sup><sup>b</sup> − <sup>b</sup>|/b by backtesting: subsampling the number of problems and the number of samples per problem. We find that the distributional estimator obtains significantly better sample efficiency.

according to the sampling resolution 1/Number of Samples and performing maximum likelihood estimation under the discretized distribution's probability mass function.

We tested this distributional estimator in two different ways. First, focusing on Large Language Monkeys, we used all available real data from all problems and all samples per problem to compare the standard least squares regression estimator against the distributional estimator. We found close agreement between the two estimators (Fig. [6\)](#page-6-0), giving us a sense that the two estimators yield reasonably consistent estimates under large sampling budgets.

Second, the distributional estimator also comes with another benefit: it directly provides an estimate of the power law's exponent b in a k−<sup>b</sup> . Estimating the power law's exponent is especially valuable because the exponent dictates how success rates are improving with increasing inference compute. To test how the distributional estimator and least squares estimator compare at recovering the true asymptotic power law exponent, we generated synthetic data so that we would have ground-truth knowledge of the true power law exponent, then backtested how the two scaling estimators compare at recovering the true exponent [\(Alabdulmohsin](#page-9-7) [et al.,](#page-9-7) [2022a;](#page-9-7) [Owen,](#page-14-2) [2024\)](#page-14-2) by subsampling data with fewer

problems and fewer samples per problem. We found that the distributional estimator obtains significantly better sample efficiency, with approximately an order of magnitude lower relative error def <sup>=</sup> | <sup>ˆ</sup><sup>b</sup> − <sup>b</sup>|/b compared with the least squares estimator (Fig. [7\)](#page-6-1), or equivalently, ∼<sup>2</sup> − <sup>4</sup> orders of magnitude less inference-compute. The distributional estimator performs well even under distributional mismatch.

## 6. Related Work

Research into scaling laws of deep neural networks has a rich history spanning theoretical foundations, empirical validations, and diverse applications. The earliest investigations discovered power law scaling in simple machine learning settings [\(Barkai et al.,](#page-9-8) [1993;](#page-9-8) [Mhaskar,](#page-13-5) [1996;](#page-13-5) [Pinkus,](#page-14-3) [1999\)](#page-14-3). However, the modern era of scaling laws began with breakthrough studies in neural language models [\(Hestness et al.,](#page-12-0) [2017;](#page-12-0) [Kaplan et al.,](#page-13-0) [2020;](#page-13-0) [Brown et al.,](#page-10-4) [2020b\)](#page-10-4), catalyzing extensive research across multiple directions. The theoretical understanding of scaling laws has advanced significantly [\(Spigler et al.,](#page-15-2) [2020;](#page-15-2) [Bousquet et al.,](#page-9-9) [2020;](#page-9-9) [Hutter,](#page-12-4) [2021;](#page-12-4) [Sharma & Kaplan,](#page-15-3) [2022;](#page-15-3) [Maloney et al.,](#page-13-6) [2022;](#page-13-6) [Roberts et al.,](#page-14-4) [2022;](#page-14-4) [Bahri et al.,](#page-9-10) [2024;](#page-9-10) [Michaud et al.,](#page-13-7) [2024;](#page-13-7) [Paquette et al.,](#page-14-5) [2024;](#page-14-5) [Atanasov et al.,](#page-9-11) [2024;](#page-9-11) [Bordelon et al.,](#page-9-12) [2024a](#page-9-12)[;b;](#page-9-13) [Lin](#page-13-8) [et al.,](#page-13-8) [2024;](#page-13-8) [Brill,](#page-9-14) [2024\)](#page-9-14), complemented by comprehensive empirical studies [\(Rosenfeld et al.,](#page-14-6) [2020;](#page-14-6) [Henighan et al.,](#page-12-5) [2020;](#page-12-5) [Gordon et al.,](#page-11-1) [2021;](#page-11-1) [Tay et al.,](#page-16-1) [2021;](#page-16-1) [Ghorbani et al.,](#page-11-2) [2021;](#page-11-2) [Tay et al.,](#page-16-2) [2022b;](#page-16-2) [Zhai et al.,](#page-22-0) [2022;](#page-22-0) [Alabdulmohsin](#page-9-15) [et al.,](#page-9-15) [2022b;](#page-9-15) [Dehghani et al.,](#page-10-5) [2023;](#page-10-5) [Bachmann et al.,](#page-9-16) [2023\)](#page-9-16). In the context of language models, researchers have explored scaling behaviors in various aspects: context length [\(Xiong](#page-21-2) [et al.,](#page-21-2) [2023\)](#page-21-2), in-context learning [\(Chan et al.,](#page-10-6) [2022;](#page-10-6) [Agarwal](#page-9-17) [et al.,](#page-9-17) [2024;](#page-9-17) [Arora et al.,](#page-9-18) [2024\)](#page-9-18), vocabulary size [\(Tao et al.,](#page-16-3) [2024\)](#page-16-3), and jailbreaking attempts [\(Anil et al.,](#page-9-19) [2024;](#page-9-19) [Hughes](#page-12-2) [et al.,](#page-12-2) [2024\)](#page-12-2). Studies have also investigated scaling dynamics in fine-tuning [\(Kalajdzievski,](#page-12-6) [2024;](#page-12-6) [Zhang et al.,](#page-22-1) [2024\)](#page-22-1), transfer learning [\(Hernandez et al.,](#page-12-7) [2021\)](#page-12-7), and the impact of repeated data [\(Hernandez et al.,](#page-12-8) [2022;](#page-12-8) [Muennighoff et al.,](#page-13-9) [2023\)](#page-13-9). Architectural considerations have been extensively studied, including network design [\(Tay et al.,](#page-16-4) [2022a;](#page-16-4) [Clark](#page-10-7) [et al.,](#page-10-7) [2022\)](#page-10-7), nested models [\(Kudugunta et al.,](#page-13-10) [2023\)](#page-13-10), pruning strategies [\(Rosenfeld et al.,](#page-14-7) [2021\)](#page-14-7), and precision requirements [\(Dettmers & Zettlemoyer,](#page-10-8) [2023;](#page-10-8) [Kumar et al.,](#page-13-11) [2024;](#page-13-11) [Sun et al.,](#page-16-5) [2025\)](#page-16-5). Research has also addressed multimodal extensions [\(Aghajanyan et al.,](#page-9-20) [2023;](#page-9-20) [Cherti et al.,](#page-10-9) [2023\)](#page-10-9) and inference optimization [\(Sardana et al.,](#page-14-8) [2023;](#page-14-8) [Brown et al.,](#page-9-4) [2024;](#page-9-4) [Snell et al.,](#page-15-4) [2024a;](#page-15-4) [Wu et al.,](#page-21-3) [2024;](#page-21-3) [Chen et al.,](#page-10-10) [2024\)](#page-10-10). The field has expanded to encompass diverse domains including reinforcement learning (both single-agent [\(Jones,](#page-12-9) [2021;](#page-12-9) [Hilton et al.,](#page-12-10) [2023;](#page-12-10) [Neumann & Gros,](#page-13-12) [2024\)](#page-13-12) and multi-agent [\(Neumann & Gros,](#page-13-13) [2022\)](#page-13-13)), graph networks [\(Liu](#page-13-14) [et al.,](#page-13-14) [2024\)](#page-13-14), diffusion models [\(Mei et al.,](#page-13-15) [2024;](#page-13-15) [Liang et al.,](#page-13-16) [2024\)](#page-13-16), and associative memory models [\(Romani et al.,](#page-14-9) [2013;](#page-14-9) [Cabannes et al.,](#page-10-11) [2024;](#page-10-11) [Schaeffer et al.,](#page-15-5) [2024c\)](#page-15-5). Recent work

has explored emerging phenomena such as inverse scaling [\(McKenzie et al.,](#page-13-17) [2024\)](#page-13-17), unique functional forms [\(Caballero](#page-10-3) [et al.,](#page-10-3) [2022\)](#page-10-3), scaling patterns across model families [\(Ruan](#page-14-10) [et al.,](#page-14-10) [2024;](#page-14-10) [Polo et al.,](#page-14-11) [2024\)](#page-14-11), and downstream capabilities [\(Srivastava et al.,](#page-15-6) [2023;](#page-15-6) [Wei et al.,](#page-21-4) [2022a;](#page-21-4) [Hu et al.,](#page-12-11) [2024;](#page-12-11) [Schaeffer et al.,](#page-15-7) [2024b;](#page-15-7) [Snell et al.,](#page-15-8) [2024b;](#page-15-8) [Wu &](#page-21-5) [Lo,](#page-21-5) [2024\)](#page-21-5). Researchers have also investigated critical challenges including data contamination [\(Schaeffer,](#page-15-9) [2023;](#page-15-9) [Jiang](#page-12-12) [et al.,](#page-12-12) [2024;](#page-12-12) [Dominguez-Olmedo et al.,](#page-10-12) [2024\)](#page-10-12), model-data feedback loops [\(Dohmatob et al.,](#page-10-13) [2024;](#page-10-13) [Gerstgrasser et al.,](#page-11-3) [2024;](#page-11-3) [Kazdan et al.,](#page-13-18) [2024\)](#page-13-18), and overtraining effects [\(Gao](#page-11-4) [et al.,](#page-11-4) [2023;](#page-11-4) [Gadre et al.,](#page-10-14) [2024\)](#page-10-14). Additional contributions include studies in sparse autoencoders [\(Gao et al.,](#page-11-5) [2024\)](#page-11-5), biologically-plausible backpropagation [\(Filipovich et al.,](#page-10-15) [2022\)](#page-10-15), and self-supervised learning for vision [\(Schaeffer](#page-15-10) [et al.,](#page-15-10) [2024a\)](#page-15-10). Recent efforts have also focused on reconciling apparent contradictions in scaling behaviors [\(Besiroglu](#page-9-6) [et al.,](#page-9-6) [2024b;](#page-9-6) [Porian et al.,](#page-14-12) [2024\)](#page-14-12).

## 7. Discussion and Future Directions

This work advances our mathematical understanding of how and why language model performance improves with additional inference compute through repeat sampling. By establishing rigorous theoretical foundations for these empirically-observed power laws, our work provides practitioners with principled ways to understand and predict model performance when scaling inference compute. The distributional perspective we develop explains previously puzzling deviations from power law scaling and enables more efficient estimation of scaling parameters.

Two related questions are *why* such distributional structure exists in the single-attempt success rates and whether one should expect such structure to appear in future benchmarks. We conjecture there are at least two reasons: (1) benchmark design, in that benchmarks are intentionally crafted that problems have a spread of difficulty without being too easy or too hard, and (2) selection bias, in that more interesting patterns such as power law scaling are more likely to garner more interest from the research community.

Despite focusing on scaling inference compute, our paper contributes a new hypothesis for an open question in scaling pretraining compute: *why are neural scaling laws power laws?* Just as the scaling behavior of − log(passD@k) only becomes clear for large k, so too might the scaling behavior of pretraining cross entropy with pretraining compute <sup>C</sup>. Specifically, suppose the pretraining cross entropy L as a function of pretraining compute C is a sum of many functions which decay at different rates:

$$\mathcal{L}(C) = \omega\left(\frac{1}{C^\alpha}\right) + \frac{A}{C^\alpha} + o\left(\frac{1}{C^\alpha}\right),$$

where α is the smallest (positive) polynomial exponent and

ω(1/C<sup>α</sup>) represents functions that decay more slowly than any polynomial. Initially, for small C, the dominant term may be unclear, but as pretraining compute is scaled up across <sup>8</sup> − <sup>10</sup> orders of magnitude, the leading order term dominates and an approximate power law emerges:

$$\mathcal{L}(C) \approx \text{const} + \frac{A}{C^\alpha} + 0 \quad \text{as} \quad C \rightarrow \infty.$$

Thus, a power law relationship may only be reasonable for sufficiently large pretraining compute C, which in turn may require excluding the lowest pretraining compute models in order to obtain good predictions, justifying a widespread empirical practice [\(Kaplan et al.,](#page-13-0) [2020\)](#page-13-0). We designate possible functions hiding in ω(1/C<sup>α</sup>) and o(1/C<sup>α</sup>) as *the dark matter of neural scaling laws*.

## Acknowledgments

RS acknowledges support from Stanford Data Science and the OpenAI Superalignment Fast Grant. SK acknowledges support by NSF 2046795 and 2205329, IES R305C240046, ARPA-H, the MacArthur Foundation, Schmidt Sciences, OpenAI, and Stanford HAI.

## Impact Statement

Our findings have important practical implications for the deployment of large language models, as they can help organizations more accurately forecast compute requirements and make informed trade-offs between model size, inference costs, and performance targets. The mathematical framework we develop could also generalize beyond language models to other domains where similar scaling phenomena emerge. While our work is primarily theoretical, we acknowledge that advances in language model capabilities can have broad societal impacts. We hope that better understanding these fundamental scaling behaviors will help the research community develop more efficient and reliable AI systems.

## References


[1] Agarwal, R., Singh, A., Zhang, L. M., Bohnet, B., Rosias, L., Chan, S. C., Zhang, B., Anand, A., Abbas, Z., Nova, A., Co-Reyes, J. D., Chu, E., Behbahani, F., Faust, A., and Larochelle, H. Many-shot in-context learning. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024. URL [https:](https://openreview.net/forum?id=AB6XpMzvqH) [//openreview.net/forum?id=AB6XpMzvqH](https://openreview.net/forum?id=AB6XpMzvqH). Aghajanyan, A., Yu, L., Conneau, A., Hsu, W.-N., Hambardzumyan, K., Zhang, S., Roller, S., Goyal, N., Levy, O., and Zettlemoyer, L. Scaling laws for generative mixed-modal language models. In *International Conference on Machine Learning*, pp. 265–279. PMLR, 2023. Alabdulmohsin, I., Neyshabur, B., and Zhai, X. Revisiting neural scaling laws in language and vision, 2022a. URL <https://arxiv.org/abs/2209.06640>. Alabdulmohsin, I. M., Neyshabur, B., and Zhai, X. Revisiting neural scaling laws in language and vision. *Advances in Neural Information Processing Systems*, 35:22300– 22312, 2022b. Anderljung, M., Barnhart, J., Korinek, A., Leung, J., O'Keefe, C., Whittlestone, J., Avin, S., Brundage, M., Bullock, J., Cass-Beggs, D., Chang, B., Collins, T., Fist, T., Hadfield, G., Hayes, A., Ho, L., Hooker, S., Horvitz, E., Kolt, N., Schuett, J., Shavit, Y., Siddarth, D., Trager, R., and Wolf, K. Frontier ai regulation: Managing emerging risks to public safety, 2023. URL <https://arxiv.org/abs/2307.03718>. Anil, C., DURMUS, E., Rimsky, N., Sharma, M., Benton, J., Kundu, S., Batson, J., Tong, M., Mu, J., Ford,
  - D. J., Mosconi, F., Agrawal, R., Schaeffer, R., Bashkansky, N., Svenningsen, S., Lambert, M., Radhakrishnan, A., Denison, C., Hubinger, E. J., Bai, Y., Bricken, T., Maxwell, T., Schiefer, N., Sully, J., Tamkin, A., Lanham, T., Nguyen, K., Korbak, T., Kaplan, J., Ganguli, D., Bowman, S. R., Perez, E., Grosse, R. B., and Duvenaud,

[2] D. Many-shot jailbreaking. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=cw5mgd71jW) [id=cw5mgd71jW](https://openreview.net/forum?id=cw5mgd71jW). Arora, A., Jurafsky, D., Potts, C., and Goodman, N. D. Bayesian scaling laws for in-context learning, 2024. URL <https://arxiv.org/abs/2410.16531>. Atanasov, A., Zavatone-Veth, J. A., and Pehlevan, C. Scaling and renormalization in high-dimensional regression. *arXiv preprint arXiv:2405.00592*, 2024. Bachmann, G., Anagnostidis, S., and Hofmann, T. Scaling mlps: A tale of inductive bias, 2023. URL [https:](https://arxiv.org/abs/2306.13575) [//arxiv.org/abs/2306.13575](https://arxiv.org/abs/2306.13575). Bahri, Y., Dyer, E., Kaplan, J., Lee, J., and Sharma, U. Explaining neural scaling laws. *Proceedings of the National Academy of Sciences*, 121(27):e2311878121, 2024. Barkai, N., Seung, H. S., and Sompolinsky, H. Scaling laws in learning of classification tasks. *Physical review letters*, 70(20):3167, 1993. Besiroglu, T., Emery-Xu, N., and Thompson, N. Economic impacts of ai-augmented r&d. *Research Policy*, 53(7): 105037, 2024a. Besiroglu, T., Erdil, E., Barnett, M., and You, J. Chinchilla scaling: A replication attempt, 2024b. URL [https:](https://arxiv.org/abs/2404.10102) [//arxiv.org/abs/2404.10102](https://arxiv.org/abs/2404.10102). Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., et al. Pythia: A suite for analyzing large language models across training and scaling. In *International Conference on Machine Learning*, pp. 2397–2430. PMLR, 2023. Bochud, T. and Challet, D. Optimal approximations of power-laws with exponentials, 2006. URL [https://](https://arxiv.org/abs/physics/0605149) [arxiv.org/abs/physics/0605149](https://arxiv.org/abs/physics/0605149). Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, M. S., Bohg, J., Bosselut, A., Brunskill, E., et al. On the opportunities and risks of foundation models. *arXiv preprint arXiv:2108.07258*, 2021. Bordelon, B., Atanasov, A., and Pehlevan, C. A dynamical model of neural scaling laws. *arXiv preprint arXiv:2402.01092*, 2024a. Bordelon, B., Atanasov, A., and Pehlevan, C. How feature learning can improve neural scaling laws. *arXiv preprint arXiv:2409.17858*, 2024b. Bousquet, O., Hanneke, S., Moran, S., van Handel, R., and Yehudayoff, A. A theory of universal learning, 2020. URL <https://arxiv.org/abs/2011.04483>. Brill, A. Neural scaling laws rooted in the data distribution. *arXiv preprint arXiv:2412.07942*, 2024. Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., Ré, C., and Mirhoseini, A. Large language monkeys: Scaling inference compute with repeated sampling, 2024. URL <https://arxiv.org/abs/2407.21787>. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33: 1877–1901, 2020a.

[3] Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners, 2020b. URL [https://](https://arxiv.org/abs/2005.14165) [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165). Caballero, E., Gupta, K., Rish, I., and Krueger, D. Broken neural scaling laws. *arXiv preprint arXiv:2210.14891*, 2022. Cabannes, V., Dohmatob, E., and Bietti, A. Scaling laws for associative memories, 2024. URL [https://arxiv.](https://arxiv.org/abs/2310.02984) [org/abs/2310.02984](https://arxiv.org/abs/2310.02984). Chan, S., Santoro, A., Lampinen, A., Wang, J., Singh, A., Richemond, P., McClelland, J., and Hill, F. Data distributional properties drive emergent in-context learning in transformers. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), *Advances in Neural Information Processing Systems*, volume 35, pp. 18878–18891. Curran Associates, Inc., 2022. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2022/file/77c6ccacfd9962e2307fc64680fc5ace-Paper-Conference.pdf) [cc/paper\\_files/paper/2022/file/](https://proceedings.neurips.cc/paper_files/paper/2022/file/77c6ccacfd9962e2307fc64680fc5ace-Paper-Conference.pdf) [77c6ccacfd9962e2307fc64680fc5ace-Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/77c6ccacfd9962e2307fc64680fc5ace-Paper-Conference.pdf)-Conference. [pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/77c6ccacfd9962e2307fc64680fc5ace-Paper-Conference.pdf). Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto,

[4] H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., Mc-Grew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba, W. Evaluating large language models trained on code, 2021. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2107.03374) [2107.03374](https://arxiv.org/abs/2107.03374). Chen, Y., Pan, X., Li, Y., Ding, B., and Zhou, J. A simple and provable scaling law for the test-time compute of large language models, 2024. URL [https://arxiv.](https://arxiv.org/abs/2411.19477) [org/abs/2411.19477](https://arxiv.org/abs/2411.19477). Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., and Jitsev, J. Reproducible scaling laws for contrastive language-image learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 2818–2829, 2023. Clark, A., de Las Casas, D., Guy, A., Mensch, A., Paganini, M., Hoffmann, J., Damoc, B., Hechtman, B., Cai, T., Borgeaud, S., et al. Unified scaling laws for routed language models. In *International conference on machine learning*, pp. 4057–4086. PMLR, 2022. Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P., Heek, J., Gilmer, J., Steiner, A. P., Caron, M., Geirhos, R., Alabdulmohsin, I., et al. Scaling vision transformers to 22 billion parameters. In *International Conference on Machine Learning*, pp. 7480–7512. PMLR, 2023. Dettmers, T. and Zettlemoyer, L. The case for 4-bit precision: k-bit inference scaling laws. In *International Conference on Machine Learning*, pp. 7750–7774. PMLR, 2023. Dohmatob, E., Feng, Y., Yang, P., Charton, F., and Kempe,
  - J. A tale of tails: Model collapse as a change of scaling laws, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2402.07043) [2402.07043](https://arxiv.org/abs/2402.07043). Dominguez-Olmedo, R., Dorner, F. E., and Hardt, M. Training on the test task confounds evaluation and emergence, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2407.07890) [2407.07890](https://arxiv.org/abs/2407.07890). Elkies, N. D. Is there a way to express an power law decay as a series of exponentials? MathOverflow, 2016. URL <https://mathoverflow.net/q/251661>. URL:https://mathoverflow.net/q/251661 (version: 2016- 10-08). Eloundou, T., Manning, S., Mishkin, P., and Rock, D. Gpts are gpts: An early look at the labor market impact potential of large language models, 2023. URL <https://arxiv.org/abs/2303.10130>. Filipovich, M. J., Cappelli, A., Hesslow, D., and Launay,
  - J. Scaling laws beyond backpropagation, 2022. URL <https://arxiv.org/abs/2210.14593>. Gadre, S. Y., Smyrnis, G., Shankar, V., Gururangan, S., Wortsman, M., Shao, R., Mercat, J., Fang, A., Li, J., Keh, S., et al. Language models scale reliably with over-training and on downstream tasks. *arXiv preprint arXiv:2403.08540*, 2024. Ganguli, D., Hernandez, D., Lovitt, L., Askell, A., Bai, Y., Chen, A., Conerly, T., Dassarma, N., Drain, D., Elhage, N., et al. Predictability and surprise in large generative models. In *2022 ACM Conference on Fairness, Accountability, and Transparency*, pp. 1747–1764, 2022.

[5] Gao, L., Schulman, J., and Hilton, J. Scaling laws for reward model overoptimization. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pp. 10835–10866. PMLR, 23–29 Jul 2023. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v202/gao23h.html) [v202/gao23h.html](https://proceedings.mlr.press/v202/gao23h.html). Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., and Wu, J. Scaling and evaluating sparse autoencoders. *arXiv preprint arXiv:2406.04093*, 2024. Gerstgrasser, M., Schaeffer, R., Dey, A., Rafailov, R., Sleight, H., Hughes, J., Korbak, T., Agrawal, R., Pai, D., Gromov, A., Roberts, D. A., Yang, D., Donoho, D. L., and Koyejo, S. Is model collapse inevitable? breaking the curse of recursion by accumulating real and synthetic data, 2024. URL [https://arxiv.org/abs/2404.](https://arxiv.org/abs/2404.01413) [01413](https://arxiv.org/abs/2404.01413). Ghorbani, B., Firat, O., Freitag, M., Bapna, A., Krikun, M., Garcia, X., Chelba, C., and Cherry, C. Scaling laws for neural machine translation. In *International Conference on Learning Representations*, 2021. Gordon, M. A., Duh, K., and Kaplan, J. Data and parameter scaling laws for neural machine translation. In Moens, M.-F., Huang, X., Specia, L., and Yih, S. W.-t. (eds.), *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pp. 5915–5922, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.478. URL [https://](https://aclanthology.org/2021.emnlp-main.478) [aclanthology.org/2021.emnlp-main.478](https://aclanthology.org/2021.emnlp-main.478). Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., Yang, A., Fan, A., Goyal, A., Hartshorn, A., Yang, A., Mitra, A., Sravankumar, A., Korenev, A., Hinsvark, A., Rao, A., Zhang, A., Rodriguez, A., Gregerson, A., Spataru, A., Roziere, B., Biron, B., Tang, B., Chern, B., Caucheteux, C., Nayak, C., Bi, C., Marra, C., McConnell, C., Keller, C., Touret, C., Wu, C., Wong, C., Ferrer, C. C., Nikolaidis, C., Allonsius, D., Song, D., Pintz, D., Livshits, D., Wyatt, D., Esiobu, D., Choudhary, D., Mahajan, D., Garcia-Olano, D., Perino, D., Hupkes, D., Lakomkin, E., AlBadawy, E., Lobanova, E., Dinan, E., Smith, E. M., Radenovic, F., Guzmán, F., Zhang, F., Synnaeve, G., Lee, G., Anderson, G. L., Thattai, G., Nail, G., Mialon, G., Pang, G., Cucurell, G., Nguyen, H., Korevaar, H., Xu, H., Touvron, H., Zarov, I., Ibarra, I. A., Kloumann, I., Misra, I., Evtimov, I., Zhang, J., Copet, J., Lee, J., Geffert, J., Vranes, J., Park, J., Mahadeokar, J., Shah, J., van der Linde, J., Billock, J., Hong, J., Lee, J., Fu, J., Chi, J., Huang, J., Liu, J., Wang, J., Yu, J., Bitton, J., Spisak, J., Park, J., Rocca, J., Johnstun, J., Saxe, J., Jia, J., Alwala, K. V., Prasad, K., Upasani, K., Plawiak, K., Li, K., Heafield, K., Stone, K., El-Arini, K., Iyer, K., Malik, K., Chiu, K., Bhalla, K., Lakhotia, K., Rantala-Yeary, L., van der Maaten, L., Chen, L., Tan, L., Jenkins, L., Martin, L., Madaan, L., Malo, L., Blecher, L., Landzaat, L., de Oliveira, L., Muzzi, M., Pasupuleti, M., Singh, M., Paluri, M., Kardas, M., Tsimpoukelli, M., Oldham, M., Rita, M., Pavlova, M., Kambadur, M., Lewis, M., Si, M., Singh, M. K., Hassan, M., Goyal, N., Torabi, N., Bashlykov, N., Bogoychev, N., Chatterji, N., Zhang, N., Duchenne, O., Çelebi, O., Alrassy, P., Zhang, P., Li, P., Vasic, P., Weng, P., Bhargava, P., Dubal, P., Krishnan, P., Koura, P. S., Xu, P., He, Q., Dong, Q., Srinivasan, R., Ganapathy, R., Calderer, R., Cabral, R. S., Stojnic, R., Raileanu, R., Maheswari, R., Girdhar, R., Patel, R., Sauvestre, R., Polidoro, R., Sumbaly, R., Taylor, R., Silva, R., Hou, R., Wang, R., Hosseini, S., Chennabasappa, S., Singh, S., Bell, S., Kim, S. S., Edunov, S., Nie, S., Narang, S., Raparthy, S., Shen, S., Wan, S., Bhosale, S., Zhang, S., Vandenhende, S., Batra, S., Whitman, S., Sootla, S., Collot, S., Gururangan, S., Borodinsky, S., Herman, T., Fowler, T., Sheasha, T., Georgiou, T., Scialom, T., Speckbacher, T., Mihaylov, T., Xiao, T., Karn, U., Goswami, V., Gupta, V., Ramanathan, V., Kerkez, V., Gonguet, V., Do, V., Vogeti, V., Albiero, V., Petrovic, V., Chu, W., Xiong, W., Fu, W., Meers, W., Martinet, X., Wang, X., Wang, X., Tan, X. E., Xia, X., Xie, X., Jia, X., Wang, X., Goldschlag, Y., Gaur, Y., Babaei, Y., Wen, Y., Song, Y., Zhang, Y., Li, Y., Mao, Y., Coudert, Z. D., Yan, Z., Chen, Z., Papakipos, Z., Singh, A., Srivastava, A., Jain, A., Kelsey, A., Shajnfeld, A., Gangidi, A., Victoria, A., Goldstand, A., Menon, A., Sharma, A., Boesenberg, A., Baevski, A., Feinstein, A., Kallet, A., Sangani, A., Teo, A., Yunus, A., Lupu, A., Alvarado, A., Caples, A., Gu, A., Ho, A., Poulton, A., Ryan, A., Ramchandani, A., Dong, A., Franco, A., Goyal, A., Saraf, A., Chowdhury, A., Gabriel, A., Bharambe, A., Eisenman, A., Yazdan, A., James, B., Maurer, B., Leonhardi, B., Huang, B., Loyd, B., Paola,
  - B. D., Paranjape, B., Liu, B., Wu, B., Ni, B., Hancock, B., Wasti, B., Spence, B., Stojkovic, B., Gamido, B., Montalvo, B., Parker, C., Burton, C., Mejia, C., Liu, C., Wang, C., Kim, C., Zhou, C., Hu, C., Chu, C.-H., Cai, C., Tindal, C., Feichtenhofer, C., Gao, C., Civin, D., Beaty, D., Kreymer, D., Li, D., Adkins, D., Xu, D., Testuggine, D., David, D., Parikh, D., Liskovich, D., Foss, D., Wang, D., Le, D., Holland, D., Dowling, E., Jamil, E., Montgomery, E., Presani, E., Hahn, E., Wood, E., Le, E.-T., Brinkman, E., Arcaute, E., Dunbar, E., Smothers, E., Sun, F., Kreuk, F., Tian, F., Kokkinos, F., Ozgenel, F., Caggioni, F., Kanayet, F., Seide, F., Florez, G. M., Schwarz, G., Badeer, G., Swee, G., Halpern, G., Herman, G., Sizov, G., Guangyi, Zhang, Lakshminarayanan, G., Inan, H.,

[6] Shojanazeri, H., Zou, H., Wang, H., Zha, H., Habeeb, H., Rudolph, H., Suk, H., Aspegren, H., Goldman, H., Zhan, H., Damlaj, I., Molybog, I., Tufanov, I., Leontiadis, I., Veliche, I.-E., Gat, I., Weissman, J., Geboski, J., Kohli, J., Lam, J., Asher, J., Gaya, J.-B., Marcus, J., Tang, J., Chan, J., Zhen, J., Reizenstein, J., Teboul, J., Zhong, J., Jin, J., Yang, J., Cummings, J., Carvill, J., Shepard, J., McPhie, J., Torres, J., Ginsburg, J., Wang, J., Wu, K., U,

[7] K. H., Saxena, K., Khandelwal, K., Zand, K., Matosich, K., Veeraraghavan, K., Michelena, K., Li, K., Jagadeesh, K., Huang, K., Chawla, K., Huang, K., Chen, L., Garg, L., A, L., Silva, L., Bell, L., Zhang, L., Guo, L., Yu, L., Moshkovich, L., Wehrstedt, L., Khabsa, M., Avalani, M., Bhatt, M., Mankus, M., Hasson, M., Lennie, M., Reso, M., Groshev, M., Naumov, M., Lathi, M., Keneally, M., Liu, M., Seltzer, M. L., Valko, M., Restrepo, M., Patel, M., Vyatskov, M., Samvelyan, M., Clark, M., Macey, M., Wang, M., Hermoso, M. J., Metanat, M., Rastegari, M., Bansal, M., Santhanam, N., Parks, N., White, N., Bawa, N., Singhal, N., Egebo, N., Usunier, N., Mehta, N., Laptev, N. P., Dong, N., Cheng, N., Chernoguz, O., Hart, O., Salpekar, O., Kalinli, O., Kent, P., Parekh, P., Saab, P., Balaji, P., Rittner, P., Bontrager, P., Roux, P., Dollar, P., Zvyagina, P., Ratanchandani, P., Yuvraj, P., Liang, Q., Alao, R., Rodriguez, R., Ayub, R., Murthy, R., Nayani, R., Mitra, R., Parthasarathy, R., Li, R., Hogan, R., Battey, R., Wang, R., Howes, R., Rinott, R., Mehta, S., Siby, S., Bondu, S. J., Datta, S., Chugh, S., Hunt, S., Dhillon, S., Sidorov, S., Pan, S., Mahajan, S., Verma, S., Yamamoto, S., Ramaswamy, S., Lindsay, S., Lindsay, S., Feng, S., Lin, S., Zha, S. C., Patil, S., Shankar, S., Zhang, S., Zhang, S., Wang, S., Agarwal, S., Sajuyigbe, S., Chintala, S., Max, S., Chen, S., Kehoe, S., Satterfield, S., Govindaprasad, S., Gupta, S., Deng, S., Cho, S., Virk, S., Subramanian, S., Choudhury, S., Goldman, S., Remez, T., Glaser, T., Best, T., Koehler, T., Robinson, T., Li, T., Zhang, T., Matthews, T., Chou, T., Shaked, T., Vontimitta, V., Ajayi, V., Montanez, V., Mohan, V., Kumar, V. S., Mangla, V., Ionescu, V., Poenaru, V., Mihailescu, V. T., Ivanov, V., Li, W., Wang, W., Jiang, W., Bouaziz, W., Constable, W., Tang, X., Wu, X., Wang, X., Wu, X., Gao, X., Kleinman, Y., Chen, Y., Hu, Y., Jia, Y., Qi, Y., Li, Y., Zhang, Y., Zhang, Y., Adi, Y., Nam, Y., Yu, Wang, Zhao, Y., Hao, Y., Qian, Y., Li, Y., He, Y., Rait, Z., DeVito, Z., Rosnbrick, Z., Wen, Z., Yang, Z., Zhao, Z., and Ma, Z. The llama 3 herd of models, 2024. URL <https://arxiv.org/abs/2407.21783>. Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*, 2021. Henighan, T., Kaplan, J., Katz, M., Chen, M., Hesse, C., Jackson, J., Jun, H., Brown, T. B., Dhariwal, P., Gray, S., et al. Scaling laws for autoregressive generative modeling. *arXiv preprint arXiv:2010.14701*, 2020. Hernandez, D., Kaplan, J., Henighan, T., and McCandlish,
  - S. Scaling laws for transfer, 2021. URL [https://](https://arxiv.org/abs/2102.01293) [arxiv.org/abs/2102.01293](https://arxiv.org/abs/2102.01293). Hernandez, D., Brown, T., Conerly, T., DasSarma, N., Drain, D., El-Showk, S., Elhage, N., Hatfield-Dodds, Z., Henighan, T., Hume, T., et al. Scaling laws and interpretability of learning from repeated data. *arXiv preprint arXiv:2205.10487*, 2022. Hestness, J., Narang, S., Ardalani, N., Diamos, G., Jun, H., Kianinejad, H., Patwary, M., Ali, M., Yang, Y., and Zhou,
  - Y. Deep learning scaling is predictable, empirically. *arXiv preprint arXiv:1712.00409*, 2017. Hilton, J., Tang, J., and Schulman, J. Scaling laws for single-agent reinforcement learning, 2023. URL [https:](https://arxiv.org/abs/2301.13442) [//arxiv.org/abs/2301.13442](https://arxiv.org/abs/2301.13442). Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks,
  - L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J. W., Vinyals, O., and Sifre, L. Training compute-optimal large language models, 2022. URL [https://arxiv.org/](https://arxiv.org/abs/2203.15556) [abs/2203.15556](https://arxiv.org/abs/2203.15556). Hu, S., Liu, X., Han, X., Zhang, X., He, C., Zhao, W., Lin, Y., Ding, N., Ou, Z., Zeng, G., Liu, Z., and Sun,
  - M. Predicting emergent abilities with infinite resolution evaluation, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2310.03262) [2310.03262](https://arxiv.org/abs/2310.03262). Hughes, J., Price, S., Lynch, A., Schaeffer, R., Barez, F., Koyejo, S., Sleight, H., Jones, E., Perez, E., and Sharma, M. Best-of-n jailbreaking, 2024. URL [https:](https://arxiv.org/abs/2412.03556) [//arxiv.org/abs/2412.03556](https://arxiv.org/abs/2412.03556). Hutter, M. Learning curve theory, 2021. URL [https:](https://arxiv.org/abs/2102.04074) [//arxiv.org/abs/2102.04074](https://arxiv.org/abs/2102.04074). Jiang, M., Liu, K. Z., Zhong, M., Schaeffer, R., Ouyang, S., Han, J., and Koyejo, S. Investigating data contamination for pre-training language models, 2024. URL [https:](https://arxiv.org/abs/2401.06059) [//arxiv.org/abs/2401.06059](https://arxiv.org/abs/2401.06059). Jones, A. L. Scaling scaling laws with board games. *arXiv preprint arXiv:2104.03113*, 2021. Kalajdzievski, D. Scaling laws for forgetting when finetuning large language models, 2024. URL [https://](https://arxiv.org/abs/2401.05605) [arxiv.org/abs/2401.05605](https://arxiv.org/abs/2401.05605).

[8] Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models, 2020. URL [https://arxiv.org/abs/2001.](https://arxiv.org/abs/2001.08361) [08361](https://arxiv.org/abs/2001.08361). Kazdan, J., Schaeffer, R., Dey, A., Gerstgrasser, M., Rafailov, R., Donoho, D. L., and Koyejo, S. Collapse or thrive? perils and promises of synthetic data in a selfgenerating world, 2024. URL [https://arxiv.org/](https://arxiv.org/abs/2410.16713) [abs/2410.16713](https://arxiv.org/abs/2410.16713). Kudugunta, S., Kusupati, A., Dettmers, T., Chen, K., Dhillon, I., Tsvetkov, Y., Hajishirzi, H., Kakade, S., Farhadi, A., Jain, P., et al. Matformer: Nested transformer for elastic inference. *arXiv preprint arXiv:2310.07707*, 2023. Kulal, S., Pasupat, P., Chandra, K., Lee, M., Padon, O., Aiken, A., and Liang, P. S. Spoc: Search-based pseudocode to code. *Advances in Neural Information Processing Systems*, 32, 2019. Kumar, T., Ankner, Z., Spector, B. F., Bordelon, B., Muennighoff, N., Paul, M., Pehlevan, C., Ré, C., and Raghunathan, A. Scaling laws for precision. *arXiv preprint arXiv:2411.04330*, 2024. Liang, Z., He, H., Yang, C., and Dai, B. Scaling laws for diffusion transformers, 2024. URL [https://arxiv.](https://arxiv.org/abs/2410.08184) [org/abs/2410.08184](https://arxiv.org/abs/2410.08184). Lin, L., Wu, J., Kakade, S. M., Bartlett, P. L., and Lee, J. D. Scaling laws in linear regression: Compute, parameters, and data. *arXiv preprint arXiv:2406.08466*, 2024. Liu, J., Mao, H., Chen, Z., Zhao, T., Shah, N., and Tang,

[9] J. Towards neural scaling laws on graphs, 2024. URL <https://arxiv.org/abs/2402.02054>. Maloney, A., Roberts, D. A., and Sully, J. A solvable model of neural scaling laws. *arXiv preprint arXiv:2210.16859*, 2022. Maslej, N., Fattorini, L., Perrault, R., Parli, V., Reuel, A., Brynjolfsson, E., Etchemendy, J., Ligett, K., Lyons, T., Manyika, J., Niebles, J. C., Shoham, Y., Wald, R., and Clark, J. Artificial intelligence index report 2024, 2024. URL <https://arxiv.org/abs/2405.19522>. Mazeika, M., Phan, L., Yin, X., Zou, A., Wang, Z., Mu, N., Sakhaee, E., Li, N., Basart, S., Li, B., Forsyth, D., and Hendrycks, D. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal, 2024. URL [https://arxiv.org/](https://arxiv.org/abs/2402.04249) [abs/2402.04249](https://arxiv.org/abs/2402.04249). McKenzie, I. R., Lyzhov, A., Pieler, M., Parrish, A., Mueller, A., Prabhu, A., McLean, E., Kirtland, A., Ross, A., Liu, A., Gritsevskiy, A., Wurgaft, D., Kauffman, D., Recchia, G., Liu, J., Cavanagh, J., Weiss, M., Huang, S., Droid,
  - T. F., Tseng, T., Korbak, T., Shen, X., Zhang, Y., Zhou, Z., Kim, N., Bowman, S. R., and Perez, E. Inverse scaling: When bigger isn't better, 2024. URL [https://arxiv.](https://arxiv.org/abs/2306.09479) [org/abs/2306.09479](https://arxiv.org/abs/2306.09479). Mei, K., Tu, Z., Delbracio, M., Talebi, H., Patel, V. M., and Milanfar, P. Bigger is not always better: Scaling properties of latent diffusion models, 2024. URL [https:](https://arxiv.org/abs/2404.01367) [//arxiv.org/abs/2404.01367](https://arxiv.org/abs/2404.01367). Mhaskar, H. N. Neural networks for optimal approximation of smooth and analytic functions. *Neural computation*, 8 (1):164–177, 1996. Michaud, E., Liu, Z., Girit, U., and Tegmark, M. The quantization model of neural scaling. *Advances in Neural Information Processing Systems*, 36, 2024. mpmath development team, T. *mpmath: a Python library for arbitrary-precision floating-point arithmetic (version 1.3.0)*, 2023. http://mpmath.org/. Muennighoff, N., Rush, A., Barak, B., Le Scao, T., Tazi, N., Piktus, A., Pyysalo, S., Wolf, T., and Raffel, C. A. Scaling data-constrained language models. *Advances in Neural Information Processing Systems*, 36:50358–50376, 2023. Neumann, O. and Gros, C. Scaling laws for a multiagent reinforcement learning model. *arXiv preprint arXiv:2210.00849*, 2022. Neumann, O. and Gros, C. Alphazero neural scaling and zipf's law: a tale of board games and power laws, 2024. URL <https://arxiv.org/abs/2412.11979>. OpenAI, Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., Avila, R., Babuschkin, I., Balaji, S., Balcom, V., Baltescu, P., Bao, H., Bavarian, M., Belgum, J., Bello, I., Berdine, J., Bernadett-Shapiro, G., Berner, C., Bogdonoff, L., Boiko, O., Boyd, M., Brakman, A.-L., Brockman, G., Brooks, T., Brundage, M., Button, K., Cai, T., Campbell, R., Cann, A., Carey, B., Carlson, C., Carmichael, R., Chan, B., Chang, C., Chantzis, F., Chen, D., Chen, S., Chen, R., Chen, J., Chen, M., Chess, B., Cho, C., Chu, C., Chung, H. W., Cummings, D., Currier, J., Dai, Y., Decareaux, C., Degry, T., Deutsch, N., Deville, D., Dhar, A., Dohan, D., Dowling, S., Dunning, S., Ecoffet, A., Eleti, A., Eloundou, T., Farhi, D., Fedus, L., Felix, N., Fishman, S. P., Forte, J., Fulford, I., Gao, L., Georges, E., Gibson, C., Goel, V., Gogineni, T., Goh, G., Gontijo-Lopes, R., Gordon, J., Grafstein, M., Gray, S., Greene, R., Gross, J., Gu, S. S., Guo, Y., Hallacy, C., Han,

[10] J., Harris, J., He, Y., Heaton, M., Heidecke, J., Hesse, C., Hickey, A., Hickey, W., Hoeschele, P., Houghton, B., Hsu, K., Hu, S., Hu, X., Huizinga, J., Jain, S., Jain, S., Jang, J., Jiang, A., Jiang, R., Jin, H., Jin, D., Jomoto, S., Jonn, B., Jun, H., Kaftan, T., Łukasz Kaiser, Kamali, A., Kanitscheider, I., Keskar, N. S., Khan, T., Kilpatrick, L., Kim, J. W., Kim, C., Kim, Y., Kirchner, J. H., Kiros, J., Knight, M., Kokotajlo, D., Łukasz Kondraciuk, Kondrich, A., Konstantinidis, A., Kosic, K., Krueger, G., Kuo, V., Lampe, M., Lan, I., Lee, T., Leike, J., Leung, J., Levy, D., Li, C. M., Lim, R., Lin, M., Lin, S., Litwin, M., Lopez, T., Lowe, R., Lue, P., Makanju, A., Malfacini, K., Manning, S., Markov, T., Markovski, Y., Martin, B., Mayer, K., Mayne, A., McGrew, B., McKinney, S. M., McLeavey, C., McMillan, P., McNeil, J., Medina, D., Mehta, A., Menick, J., Metz, L., Mishchenko, A., Mishkin, P., Monaco, V., Morikawa, E., Mossing, D., Mu, T., Murati, M., Murk, O., Mély, D., Nair, A., Nakano, R., Nayak, R., Neelakantan, A., Ngo, R., Noh, H., Ouyang, L., O'Keefe, C., Pachocki, J., Paino, A., Palermo, J., Pantuliano, A., Parascandolo, G., Parish, J., Parparita, E., Passos, A., Pavlov, M., Peng, A., Perelman, A., de Avila Belbute Peres, F., Petrov, M., de Oliveira Pinto, H. P., Michael, Pokorny, Pokrass, M., Pong, V. H., Powell, T., Power, A., Power, B., Proehl, E., Puri, R., Radford, A., Rae, J., Ramesh, A., Raymond, C., Real, F., Rimbach, K., Ross, C., Rotsted, B., Roussez, H., Ryder, N., Saltarelli, M., Sanders, T., Santurkar, S., Sastry, G., Schmidt, H., Schnurr, D., Schulman, J., Selsam, D., Sheppard, K., Sherbakov, T., Shieh, J., Shoker, S., Shyam, P., Sidor, S., Sigler, E., Simens, M., Sitkin, J., Slama, K., Sohl, I., Sokolowsky, B., Song, Y., Staudacher, N., Such,

[11] F. P., Summers, N., Sutskever, I., Tang, J., Tezak, N., Thompson, M. B., Tillet, P., Tootoonchian, A., Tseng, E., Tuggle, P., Turley, N., Tworek, J., Uribe, J. F. C., Vallone, A., Vijayvergiya, A., Voss, C., Wainwright, C., Wang,

[12] J. J., Wang, A., Wang, B., Ward, J., Wei, J., Weinmann, C., Welihinda, A., Welinder, P., Weng, J., Weng, L., Wiethoff, M., Willner, D., Winter, C., Wolrich, S., Wong, H., Workman, L., Wu, S., Wu, J., Wu, M., Xiao, K., Xu, T., Yoo, S., Yu, K., Yuan, Q., Zaremba, W., Zellers, R., Zhang, C., Zhang, M., Zhao, S., Zheng, T., Zhuang, J., Zhuk, W., and Zoph, B. Gpt-4 technical report, 2024. URL <https://arxiv.org/abs/2303.08774>. Owen, D. How predictable is language model benchmark performance?, 2024. Pachocki, J., Tworek, J., Fedus, L., Kaiser, L., Chen, M., Sidor, S., and Zaremba, W. Learning to reason with LLMs. Technical report, OpenAI, September 2024. URL [https://openai.com/index/](https://openai.com/index/learning-to-reason-with-llms) [learning-to-reason-with-llms](https://openai.com/index/learning-to-reason-with-llms). Contributors include the o1 Contributions team, Core Contributors, and multiple research and safety teams. Paquette, E., Paquette, C., Xiao, L., and Pennington, J. 4+ 3 phases of compute-optimal neural scaling laws. *arXiv preprint arXiv:2405.15074*, 2024. Pinkus, A. Approximation theory of the mlp model in neural networks. *Acta numerica*, 8:143–195, 1999. Polo, F. M., Somerstep, S., Choshen, L., Sun, Y., and Yurochkin, M. Sloth: scaling laws for llm skills to predict multi-benchmark performance across families, 2024. URL <https://arxiv.org/abs/2412.06540>. Porian, T., Wortsman, M., Jitsev, J., Schmidt, L., and Carmon, Y. Resolving discrepancies in compute-optimal scaling of language models, 2024. URL [https://](https://arxiv.org/abs/2406.19146) [arxiv.org/abs/2406.19146](https://arxiv.org/abs/2406.19146). Reuel, A., Bucknall, B., Casper, S., Fist, T., Soder, L., Aarne, O., Hammond, L., Ibrahim, L., Chan, A., Wills, P., Anderljung, M., Garfinkel, B., Heim, L., Trask, A., Mukobi, G., Schaeffer, R., Baker, M., Hooker, S., Solaiman, I., Luccioni, A. S., Rajkumar, N., Moës, N., Ladish, J., Guha, N., Newman, J., Bengio, Y., South, T., Pentland, A., Koyejo, S., Kochenderfer, M. J., and Trager,
  - R. Open problems in technical ai governance, 2024. URL <https://arxiv.org/abs/2407.14981>. Roberts, D. A., Yaida, S., and Hanin, B. *The principles of deep learning theory*, volume 46. Cambridge University Press Cambridge, MA, USA, 2022. Romani, S., Pinkoviezky, I., Rubin, A., and Tsodyks, M. Scaling laws of associative memory retrieval. *Neural computation*, 25(10):2523–2544, 2013. Rosenfeld, J. S., Rosenfeld, A., Belinkov, Y., and Shavit,
  - N. A constructive prediction of the generalization error across scales. In *International Conference on Learning Representations*, 2020. Rosenfeld, J. S., Frankle, J., Carbin, M., and Shavit,
  - N. On the predictability of pruning across scales. In Meila, M. and Zhang, T. (eds.), *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pp. 9075–9083. PMLR, 18–24 Jul 2021. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v139/rosenfeld21a.html) [v139/rosenfeld21a.html](https://proceedings.mlr.press/v139/rosenfeld21a.html). Ruan, Y., Maddison, C. J., and Hashimoto, T. Observational scaling laws and the predictability of language model performance, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2405.10938) [2405.10938](https://arxiv.org/abs/2405.10938). Sardana, N., Portes, J., Doubov, S., and Frankle, J. Beyond chinchilla-optimal: Accounting for inference in language model scaling laws. In *Forty-first International Conference on Machine Learning*, 2023.

[13] Schaeffer, R. Pretraining on the test set is all you need, 2023. URL <https://arxiv.org/abs/2309.08632>. Schaeffer, R., Miranda, B., and Koyejo, S. Are emergent abilities of large language models a mirage? In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), *Advances in Neural Information Processing Systems*, volume 36, pp. 55565–55581. Curran Associates, Inc., 2023. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2023/file/adc98a266f45005c403b8311ca7e8bd7-Paper-Conference.pdf) [cc/paper\\_files/paper/2023/file/](https://proceedings.neurips.cc/paper_files/paper/2023/file/adc98a266f45005c403b8311ca7e8bd7-Paper-Conference.pdf) [adc98a266f45005c403b8311ca7e8bd7-Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/adc98a266f45005c403b8311ca7e8bd7-Paper-Conference.pdf)-Conference. [pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/adc98a266f45005c403b8311ca7e8bd7-Paper-Conference.pdf). Schaeffer, R., Lecomte, V., Pai, D. B., Carranza, A., Isik, B., Unell, A., Khona, M., Yerxa, T., LeCun, Y., Chung, S., Gromov, A., Shwartz-Ziv, R., and Koyejo, S. Towards an improved understanding and utilization of maximum manifold capacity representations, 2024a. URL [https:](https://arxiv.org/abs/2406.09366) [//arxiv.org/abs/2406.09366](https://arxiv.org/abs/2406.09366). Schaeffer, R., Schoelkopf, H., Miranda, B., Mukobi, G., Madan, V., Ibrahim, A., Bradley, H., Biderman, S., and Koyejo, S. Why has predicting downstream capabilities of frontier ai models with scale remained elusive?, 2024b. URL <https://arxiv.org/abs/2406.04391>. Schaeffer, R., Zahedi, N., Khona, M., Pai, D., Truong, S., Du, Y., Ostrow, M., Chandra, S., Carranza, A., Fiete, I. R., Gromov, A., and Koyejo, S. Bridging associative memory and probabilistic modeling, 2024c. URL [https://](https://arxiv.org/abs/2402.10202) [arxiv.org/abs/2402.10202](https://arxiv.org/abs/2402.10202). Sharma, U. and Kaplan, J. Scaling laws from the data manifold dimension. *Journal of Machine Learning Research*, 23(9):1–34, 2022. Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling llm testtime compute optimally can be more effective than scaling model parameters. *arXiv preprint arXiv:2408.03314*, 2024a. Snell, C., Wallace, E., Klein, D., and Levine, S. Predicting emergent capabilities by finetuning, 2024b. URL [https:](https://arxiv.org/abs/2411.16035) [//arxiv.org/abs/2411.16035](https://arxiv.org/abs/2411.16035). Sorscher, B., Geirhos, R., Shekhar, S., Ganguli, S., and Morcos, A. Beyond neural scaling laws: beating power law scaling via data pruning. *Advances in Neural Information Processing Systems*, 35:19523–19536, 2022. Spigler, S., Geiger, M., and Wyart, M. Asymptotic learning curves of kernel methods: empirical data versus teacher–student paradigm. *Journal of Statistical Mechanics: Theory and Experiment*, 2020(12):124001, December 2020. ISSN 1742-5468. doi: 10.1088/1742-5468/ abc61d. URL [http://dx.doi.org/10.1088/](http://dx.doi.org/10.1088/1742-5468/abc61d) [1742-5468/abc61d](http://dx.doi.org/10.1088/1742-5468/abc61d). Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid, A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A., Garriga-Alonso, A., Kluska, A., Lewkowycz, A., Agarwal, A., Power, A., Ray, A., Warstadt, A., Kocurek, A. W., Safaya, A., Tazarv, A., Xiang, A., Parrish, A., Nie, A., Hussain, A., Askell, A., Dsouza, A., Slone, A., Rahane, A., Iyer, A. S., Andreassen, A., Madotto, A., Santilli, A., Stuhlmüller, A., Dai, A., La, A., Lampinen, A., Zou, A., Jiang, A., Chen, A., Vuong, A., Gupta, A., Gottardi, A., Norelli, A., Venkatesh, A., Gholamidavoodi, A., Tabassum, A., Menezes, A., Kirubarajan, A., Mullokandov, A., Sabharwal, A., Herrick, A., Efrat, A., Erdem, A., Karaka¸s, A., Roberts, B. R., Loe, B. S., Zoph, B., Bojanowski, B., Özyurt, B., Hedayatnia, B., Neyshabur, B., Inden, B., Stein, B., Ekmekci, B., Lin, B. Y., Howald, B., Orinion, B., Diao, C., Dour, C., Stinson, C., Argueta, C., Ramírez,
  - C. F., Singh, C., Rathkopf, C., Meng, C., Baral, C., Wu, C., Callison-Burch, C., Waites, C., Voigt, C., Manning,
  - C. D., Potts, C., Ramirez, C., Rivera, C. E., Siro, C., Raffel, C., Ashcraft, C., Garbacea, C., Sileo, D., Garrette, D., Hendrycks, D., Kilman, D., Roth, D., Freeman, D., Khashabi, D., Levy, D., González, D. M., Perszyk, D., Hernandez, D., Chen, D., Ippolito, D., Gilboa, D., Dohan, D., Drakard, D., Jurgens, D., Datta, D., Ganguli, D., Emelin, D., Kleyko, D., Yuret, D., Chen, D., Tam, D., Hupkes, D., Misra, D., Buzan, D., Mollo, D. C., Yang, D., Lee, D.-H., Schrader, D., Shutova, E., Cubuk,
  - E. D., Segal, E., Hagerman, E., Barnes, E., Donoway, E., Pavlick, E., Rodola, E., Lam, E., Chu, E., Tang, E., Erdem, E., Chang, E., Chi, E. A., Dyer, E., Jerzak, E., Kim, E., Manyasi, E. E., Zheltonozhskii, E., Xia, F., Siar, F., Martínez-Plumed, F., Happé, F., Chollet, F., Rong, F., Mishra, G., Winata, G. I., de Melo, G., Kruszewski, G., Parascandolo, G., Mariani, G., Wang, G., Jaimovitch-López, G., Betz, G., Gur-Ari, G., Galijasevic, H., Kim, H., Rashkin, H., Hajishirzi, H., Mehta, H., Bogar, H., Shevlin, H., Schütze, H., Yakura, H., Zhang, H., Wong, H. M., Ng, I., Noble, I., Jumelet, J., Geissinger, J., Kernion, J., Hilton, J., Lee, J., Fisac, J. F., Simon, J. B., Koppel, J., Zheng, J., Zou, J., Kocon, J., Thompson, J., Wingfield, ´ J., Kaplan, J., Radom, J., Sohl-Dickstein, J., Phang, J., Wei, J., Yosinski, J., Novikova, J., Bosscher, J., Marsh, J., Kim, J., Taal, J., Engel, J., Alabi, J., Xu, J., Song, J., Tang, J., Waweru, J., Burden, J., Miller, J., Balis,
  - J. U., Batchelder, J., Berant, J., Frohberg, J., Rozen, J., Hernandez-Orallo, J., Boudeman, J., Guerr, J., Jones, J., Tenenbaum, J. B., Rule, J. S., Chua, J., Kanclerz, K., Livescu, K., Krauth, K., Gopalakrishnan, K., Ignatyeva, K., Markert, K., Dhole, K. D., Gimpel, K., Omondi, K., Mathewson, K., Chiafullo, K., Shkaruta, K., Shridhar, K., McDonell, K., Richardson, K., Reynolds, L., Gao, L., Zhang, L., Dugan, L., Qin, L., Contreras-Ochando, L., Morency, L.-P., Moschella, L., Lam, L., Noble, L., Schmidt, L., He, L., Colón, L. O., Metz, L., ¸Senel, L. K.,

[14] Bosma, M., Sap, M., ter Hoeve, M., Farooqi, M., Faruqui, M., Mazeika, M., Baturan, M., Marelli, M., Maru, M., Quintana, M. J. R., Tolkiehn, M., Giulianelli, M., Lewis, M., Potthast, M., Leavitt, M. L., Hagen, M., Schubert, M., Baitemirova, M. O., Arnaud, M., McElrath, M., Yee,

[15] M. A., Cohen, M., Gu, M., Ivanitskiy, M., Starritt, M., Strube, M., Sw˛edrowski, M., Bevilacqua, M., Yasunaga, M., Kale, M., Cain, M., Xu, M., Suzgun, M., Walker, M., Tiwari, M., Bansal, M., Aminnaseri, M., Geva, M., Gheini, M., T, M. V., Peng, N., Chi, N. A., Lee, N., Krakover, N. G.-A., Cameron, N., Roberts, N., Doiron, N., Martinez, N., Nangia, N., Deckers, N., Muennighoff, N., Keskar, N. S., Iyer, N. S., Constant, N., Fiedel, N., Wen, N., Zhang, O., Agha, O., Elbaghdadi, O., Levy, O., Evans, O., Casares, P. A. M., Doshi, P., Fung, P., Liang, P. P., Vicol, P., Alipoormolabashi, P., Liao, P., Liang, P., Chang, P., Eckersley, P., Htut, P. M., Hwang, P., Miłkowski, P., Patil, P., Pezeshkpour, P., Oli, P., Mei, Q., Lyu, Q., Chen, Q., Banjade, R., Rudolph, R. E., Gabriel, R., Habacker, R., Risco, R., Millière, R., Garg, R., Barnes, R., Saurous, R. A., Arakawa, R., Raymaekers, R., Frank, R., Sikand, R., Novak, R., Sitelew, R., LeBras, R., Liu, R., Jacobs, R., Zhang, R., Salakhutdinov, R., Chi, R., Lee, R., Stovall, R., Teehan, R., Yang, R., Singh, S., Mohammad,

[16] S. M., Anand, S., Dillavou, S., Shleifer, S., Wiseman, S., Gruetter, S., Bowman, S. R., Schoenholz, S. S., Han, S., Kwatra, S., Rous, S. A., Ghazarian, S., Ghosh, S., Casey, S., Bischoff, S., Gehrmann, S., Schuster, S., Sadeghi, S., Hamdan, S., Zhou, S., Srivastava, S., Shi, S., Singh, S., Asaadi, S., Gu, S. S., Pachchigar, S., Toshniwal, S., Upadhyay, S., Shyamolima, Debnath, Shakeri, S., Thormeyer, S., Melzi, S., Reddy, S., Makini, S. P., Lee, S.-H., Torene, S., Hatwar, S., Dehaene, S., Divic, S., Ermon, S., Biderman, S., Lin, S., Prasad, S., Piantadosi, S. T., Shieber,

[17] S. M., Misherghi, S., Kiritchenko, S., Mishra, S., Linzen, T., Schuster, T., Li, T., Yu, T., Ali, T., Hashimoto, T., Wu, T.-L., Desbordes, T., Rothschild, T., Phan, T., Wang, T., Nkinyili, T., Schick, T., Kornev, T., Tunduny, T., Gerstenberg, T., Chang, T., Neeraj, T., Khot, T., Shultz, T., Shaham, U., Misra, V., Demberg, V., Nyamai, V., Raunak, V., Ramasesh, V., Prabhu, V. U., Padmakumar, V., Srikumar, V., Fedus, W., Saunders, W., Zhang, W., Vossen, W., Ren, X., Tong, X., Zhao, X., Wu, X., Shen, X., Yaghoobzadeh, Y., Lakretz, Y., Song, Y., Bahri, Y., Choi, Y., Yang, Y., Hao, Y., Chen, Y., Belinkov, Y., Hou, Y., Hou, Y., Bai, Y., Seid, Z., Zhao, Z., Wang, Z., Wang, Z. J., Wang, Z., and Wu, Z. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models, 2023. URL <https://arxiv.org/abs/2206.04615>. Sun, X., Li, S., Xie, R., Han, W., Wu, K., Yang, Z., Li, Y., Wang, A., Li, S., Xue, J., Cheng, Y., Tao, Y., Kang, Z., Xu, C., Wang, D., and Jiang, J. Scaling laws for floating point quantization training, 2025. URL [https:](https://arxiv.org/abs/2501.02423) [//arxiv.org/abs/2501.02423](https://arxiv.org/abs/2501.02423). Tao, C., Liu, Q., Dou, L., Muennighoff, N., Wan, Z., Luo, P., Lin, M., and Wong, N. Scaling laws with vocabulary: Larger models deserve larger vocabularies. *arXiv preprint arXiv:2407.13623*, 2024. Tay, Y., Dehghani, M., Rao, J., Fedus, W., Abnar, S., Chung, H. W., Narang, S., Yogatama, D., Vaswani, A., and Metzler, D. Scale efficiently: Insights from pretraining and fine-tuning transformers. *arXiv preprint arXiv:2109.10686*, 2021. Tay, Y., Dehghani, M., Abnar, S., Chung, H. W., Fedus, W., Rao, J., Narang, S., Tran, V. Q., Yogatama, D., and Metzler, D. Scaling laws vs model architectures: How does inductive bias influence scaling? In *The 2023 Conference on Empirical Methods in Natural Language Processing*, 2022a. Tay, Y., Wei, J., Chung, H. W., Tran, V. Q., So, D. R., Shakeri, S., Garcia, X., Zheng, H. S., Rao, J., Chowdhery, A., Zhou, D., Metzler, D., Petrov, S., Houlsby, N., Le, Q. V., and Dehghani, M. Transcending scaling laws with 0.1 URL <https://arxiv.org/abs/2210.11399>. Team, G., Anil, R., Borgeaud, S., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., Millican, K., Silver, D., Johnson, M., Antonoglou, I., Schrittwieser, J., Glaese, A., Chen, J., Pitler, E., Lillicrap, T., Lazaridou, A., Firat, O., Molloy, J., Isard, M., Barham, P. R., Hennigan, T., Lee, B., Viola, F., Reynolds, M., Xu, Y., Doherty, R., Collins, E., Meyer, C., Rutherford, E., Moreira, E., Ayoub, K., Goel, M., Krawczyk, J., Du, C., Chi, E., Cheng, H.-T., Ni, E., Shah, P., Kane, P., Chan, B., Faruqui, M., Severyn, A., Lin, H., Li, Y., Cheng, Y., Ittycheriah, A., Mahdieh, M., Chen, M., Sun, P., Tran, D., Bagri, S., Lakshminarayanan, B., Liu, J., Orban, A., Güra, F., Zhou, H., Song, X., Boffy, A., Ganapathy, H., Zheng, S., Choe, H., Ágoston Weisz, Zhu, T., Lu, Y., Gopal, S., Kahn, J., Kula, M., Pitman, J., Shah, R., Taropa, E., Merey, M. A., Baeuml, M., Chen, Z., Shafey, L. E., Zhang, Y., Sercinoglu, O., Tucker, G., Piqueras, E., Krikun, M., Barr, I., Savinov, N., Danihelka, I., Roelofs, B., White, A., Andreassen, A., von Glehn, T., Yagati, L., Kazemi, M., Gonzalez, L., Khalman, M., Sygnowski, J., Frechette, A., Smith, C., Culp, L., Proleev, L., Luan, Y., Chen, X., Lottes, J., Schucher, N., Lebron, F., Rrustemi, A., Clay, N., Crone, P., Kocisky, T., Zhao, J., Perz, B., Yu, D., Howard, H., Bloniarz, A., Rae, J. W., Lu, H., Sifre, L., Maggioni, M., Alcober, F., Garrette, D., Barnes, M., Thakoor, S., Austin, J., Barth-Maron, G., Wong, W., Joshi, R., Chaabouni, R., Fatiha, D., Ahuja, A., Tomar, G. S., Senter, E., Chadwick, M., Kornakov, I., Attaluri, N., Iturrate, I., Liu, R., Li, Y., Cogan, S.,

[18] Chen, J., Jia, C., Gu, C., Zhang, Q., Grimstad, J., Hartman, A. J., Garcia, X., Pillai, T. S., Devlin, J., Laskin, M., de Las Casas, D., Valter, D., Tao, C., Blanco, L., Badia, A. P., Reitter, D., Chen, M., Brennan, J., Rivera, C., Brin, S., Iqbal, S., Surita, G., Labanowski, J., Rao, A., Winkler, S., Parisotto, E., Gu, Y., Olszewska, K., Addanki, R., Miech, A., Louis, A., Teplyashin, D., Brown, G., Catt, E., Balaguer, J., Xiang, J., Wang, P., Ashwood, Z., Briukhov, A., Webson, A., Ganapathy, S., Sanghavi, S., Kannan, A., Chang, M.-W., Stjerngren, A., Djolonga, J., Sun, Y., Bapna, A., Aitchison, M., Pejman, P., Michalewski, H., Yu, T., Wang, C., Love, J., Ahn, J., Bloxwich, D., Han, K., Humphreys, P., Sellam, T., Bradbury, J., Godbole, V., Samangooei, S., Damoc, B., Kaskasoli, A., Arnold, S. M. R., Vasudevan, V., Agrawal, S., Riesa, J., Lepikhin, D., Tanburn, R., Srinivasan, S., Lim, H., Hodkinson, S., Shyam, P., Ferret, J., Hand, S., Garg, A., Paine, T. L., Li, J., Li, Y., Giang, M., Neitz, A., Abbas, Z., York, S., Reid, M., Cole, E., Chowdhery, A., Das, D., Rogozinska, ´ D., Nikolaev, V., Sprechmann, P., Nado, Z., Zilka, L., Prost, F., He, L., Monteiro, M., Mishra, G., Welty, C., Newlan, J., Jia, D., Allamanis, M., Hu, C. H., de Liedekerke, R., Gilmer, J., Saroufim, C., Rijhwani, S., Hou, S., Shrivastava, D., Baddepudi, A., Goldin, A., Ozturel, A., Cassirer, A., Xu, Y., Sohn, D., Sachan, D., Amplayo, R. K., Swanson, C., Petrova, D., Narayan, S., Guez, A., Brahma, S., Landon, J., Patel, M., Zhao, R., Villela, K., Wang, L., Jia, W., Rahtz, M., Giménez, M., Yeung, L., Keeling, J., Georgiev, P., Mincu, D., Wu, B., Haykal, S., Saputro, R., Vodrahalli, K., Qin, J., Cankara, Z., Sharma, A., Fernando, N., Hawkins, W., Neyshabur, B., Kim, S., Hutter, A., Agrawal, P., Castro-Ros, A., van den Driessche, G., Wang, T., Yang, F., yiin Chang, S., Komarek, P., McIlroy, R., Luciˇ c, M., Zhang, G., Farhan, W., Sharman, ´ M., Natsev, P., Michel, P., Bansal, Y., Qiao, S., Cao, K., Shakeri, S., Butterfield, C., Chung, J., Rubenstein, P. K., Agrawal, S., Mensch, A., Soparkar, K., Lenc, K., Chung, T., Pope, A., Maggiore, L., Kay, J., Jhakra, P., Wang, S., Maynez, J., Phuong, M., Tobin, T., Tacchetti, A., Trebacz, M., Robinson, K., Katariya, Y., Riedel, S., Bailey, P., Xiao, K., Ghelani, N., Aroyo, L., Slone, A., Houlsby, N., Xiong, X., Yang, Z., Gribovskaya, E., Adler, J., Wirth, M., Lee, L., Li, M., Kagohara, T., Pavagadhi, J., Bridgers, S., Bortsova, A., Ghemawat, S., Ahmed, Z., Liu, T., Powell, R., Bolina, V., Iinuma, M., Zablotskaia, P., Besley, J., Chung, D.-W., Dozat, T., Comanescu, R., Si, X., Greer, J., Su, G., Polacek, M., Kaufman, R. L., Tokumine, S., Hu, H., Buchatskaya, E., Miao, Y., Elhawaty, M., Siddhant, A., Tomasev, N., Xing, J., Greer, C., Miller, H., Ashraf, S., Roy, A., Zhang, Z., Ma, A., Filos, A., Besta, M., Blevins, R., Klimenko, T., Yeh, C.-K., Changpinyo, S., Mu, J., Chang, O., Pajarskas, M., Muir, C., Cohen, V., Lan, C. L., Haridasan, K., Marathe, A., Hansen, S., Douglas, S., Samuel, R., Wang, M., Austin, S., Lan, C., Jiang,

[19] J., Chiu, J., Lorenzo, J. A., Sjösund, L. L., Cevey, S., Gleicher, Z., Avrahami, T., Boral, A., Srinivasan, H., Selo, V., May, R., Aisopos, K., Hussenot, L., Soares, L. B., Baumli, K., Chang, M. B., Recasens, A., Caine, B., Pritzel, A., Pavetic, F., Pardo, F., Gergely, A., Frye, J., Ramasesh, V., Horgan, D., Badola, K., Kassner, N., Roy, S., Dyer, E., Campos, V. C., Tomala, A., Tang, Y., Badawy, D. E., White, E., Mustafa, B., Lang, O., Jindal, A., Vikram, S., Gong, Z., Caelles, S., Hemsley, R., Thornton, G., Feng, F., Stokowiec, W., Zheng, C., Thacker, P., Çaglar Ünlü, ˘ Zhang, Z., Saleh, M., Svensson, J., Bileschi, M., Patil, P., Anand, A., Ring, R., Tsihlas, K., Vezer, A., Selvi, M., Shevlane, T., Rodriguez, M., Kwiatkowski, T., Daruki, S., Rong, K., Dafoe, A., FitzGerald, N., Gu-Lemberg, K., Khan, M., Hendricks, L. A., Pellat, M., Feinberg, V., Cobon-Kerr, J., Sainath, T., Rauh, M., Hashemi, S. H., Ives, R., Hasson, Y., Noland, E., Cao, Y., Byrd, N., Hou, L., Wang, Q., Sottiaux, T., Paganini, M., Lespiau, J.-B., Moufarek, A., Hassan, S., Shivakumar, K., van Amersfoort, J., Mandhane, A., Joshi, P., Goyal, A., Tung, M., Brock, A., Sheahan, H., Misra, V., Li, C., Rakicevi ´ c,´ N., Dehghani, M., Liu, F., Mittal, S., Oh, J., Noury, S., Sezener, E., Huot, F., Lamm, M., Cao, N. D., Chen, C., Mudgal, S., Stella, R., Brooks, K., Vasudevan, G., Liu, C., Chain, M., Melinkeri, N., Cohen, A., Wang, V., Seymore, K., Zubkov, S., Goel, R., Yue, S., Krishnakumaran, S., Albert, B., Hurley, N., Sano, M., Mohananey, A., Joughin, J., Filonov, E., Kepa, T., Eldawy, Y., Lim, J., Rishi, R., Badiezadegan, S., Bos, T., Chang, J., Jain, S., Padmanabhan, S. G. S., Puttagunta, S., Krishna, K., Baker, L., Kalb, N., Bedapudi, V., Kurzrok, A., Lei, S., Yu, A., Litvin, O., Zhou, X., Wu, Z., Sobell, S., Siciliano, A., Papir, A., Neale, R., Bragagnolo, J., Toor, T., Chen, T., Anklin, V., Wang, F., Feng, R., Gholami, M., Ling, K., Liu, L., Walter, J., Moghaddam, H., Kishore, A., Adamek, J., Mercado, T., Mallinson, J., Wandekar, S., Cagle, S., Ofek, E., Garrido, G., Lombriser, C., Mukha, M., Sun, B., Mohammad, H. R., Matak, J., Qian, Y., Peswani, V., Janus, P., Yuan, Q., Schelin, L., David, O., Garg, A., He, Y., Duzhyi, O., Älgmyr, A., Lottaz, T., Li, Q., Yadav, V., Xu, L., Chinien, A., Shivanna, R., Chuklin, A., Li, J., Spadine, C., Wolfe, T., Mohamed, K., Das, S., Dai, Z., He, K., von Dincklage, D., Upadhyay, S., Maurya, A., Chi, L., Krause, S., Salama, K., Rabinovitch, P. G., M, P. K. R., Selvan, A., Dektiarev, M., Ghiasi, G., Guven, E., Gupta, H., Liu, B., Sharma, D., Shtacher, I. H., Paul, S., Akerlund, O., Aubet, F.-X., Huang, T., Zhu, C., Zhu, E., Teixeira, E., Fritze, M., Bertolini, F., Marinescu, L.- E., Bölle, M., Paulus, D., Gupta, K., Latkar, T., Chang, M., Sanders, J., Wilson, R., Wu, X., Tan, Y.-X., Thiet, L. N., Doshi, T., Lall, S., Mishra, S., Chen, W., Luong, T., Benjamin, S., Lee, J., Andrejczuk, E., Rabiej, D., Ranjan, V., Styrc, K., Yin, P., Simon, J., Harriott, M. R., Bansal, M., Robsky, A., Bacon, G., Greene, D., Mirylenka, D.,

[20] Zhou, C., Sarvana, O., Goyal, A., Andermatt, S., Siegler, P., Horn, B., Israel, A., Pongetti, F., Chen, C.-W. L., Selvatici, M., Silva, P., Wang, K., Tolins, J., Guu, K., Yogev, R., Cai, X., Agostini, A., Shah, M., Nguyen, H., Donnaile, N. O., Pereira, S., Friso, L., Stambler, A., Kurzrok, A., Kuang, C., Romanikhin, Y., Geller, M., Yan, Z., Jang, K., Lee, C.-C., Fica, W., Malmi, E., Tan, Q., Banica, D., Balle, D., Pham, R., Huang, Y., Avram, D., Shi, H., Singh, J., Hidey, C., Ahuja, N., Saxena, P., Dooley, D., Potharaju, S. P., O'Neill, E., Gokulchandran, A., Foley, R., Zhao, K., Dusenberry, M., Liu, Y., Mehta, P., Kotikalapudi, R., Safranek-Shrader, C., Goodman, A., Kessinger, J., Globen, E., Kolhar, P., Gorgolewski, C., Ibrahim, A., Song, Y., Eichenbaum, A., Brovelli, T., Potluri, S., Lahoti, P., Baetu, C., Ghorbani, A., Chen, C., Crawford, A., Pal, S., Sridhar, M., Gurita, P., Mujika, A., Petrovski, I., Cedoz, P.-L., Li, C., Chen, S., Santo, N. D., Goyal, S., Punjabi, J., Kappaganthu, K., Kwak, C., LV, P., Velury, S., Choudhury, H., Hall, J., Shah, P., Figueira, R., Thomas, M., Lu, M., Zhou, T., Kumar, C., Jurdi, T., Chikkerur, S., Ma, Y., Yu, A., Kwak, S., Ähdel, V., Rajayogam, S., Choma, T., Liu, F., Barua, A., Ji, C., Park, J. H., Hellendoorn, V., Bailey, A., Bilal, T., Zhou, H., Khatir, M., Sutton, C., Rzadkowski, W., Macintosh, F., Shagin, K., Medina, P., Liang, C., Zhou, J., Shah, P., Bi, Y., Dankovics, A., Banga, S., Lehmann, S., Bredesen, M., Lin, Z., Hoffmann, J. E., Lai, J., Chung, R., Yang, K., Balani, N., Bražinskas, A., Sozanschi, A., Hayes, M., Alcalde, H. F., Makarov, P., Chen, W., Stella, A., Snijders, L., Mandl, M., Kärrman, A., Nowak, P., Wu, X., Dyck, A., Vaidyanathan, K., R, R., Mallet, J., Rudominer, M., Johnston, E., Mittal, S., Udathu, A., Christensen, J., Verma, V., Irving, Z., Santucci, A., Elsayed, G., Davoodi, E., Georgiev, M., Tenney, I., Hua, N., Cideron, G., Leurent, E., Alnahlawi, M., Georgescu, I., Wei, N., Zheng, I., Scandinaro, D., Jiang, H., Snoek, J., Sundararajan, M., Wang, X., Ontiveros, Z., Karo, I., Cole, J., Rajashekhar, V., Tumeh, L., Ben-David, E., Jain, R., Uesato, J., Datta, R., Bunyan, O., Wu, S., Zhang, J., Stanczyk, P., Zhang, Y., Steiner, D., Naskar, S., Azzam, M., Johnson, M., Paszke, A., Chiu, C.-C., Elias, J. S., Mohiuddin, A., Muhammad, F., Miao, J., Lee, A., Vieillard, N., Park, J., Zhang, J., Stanway, J., Garmon, D., Karmarkar, A., Dong, Z., Lee, J., Kumar, A., Zhou, L., Evens, J., Isaac, W., Irving, G., Loper, E., Fink, M., Arkatkar, I., Chen, N., Shafran, I., Petrychenko, I., Chen, Z., Jia, J., Levskaya, A., Zhu, Z., Grabowski, P., Mao, Y., Magni, A., Yao, K., Snaider, J., Casagrande, N., Palmer, E., Suganthan, P., Castaño, A., Giannoumis, I., Kim, W., Rybinski, M., Sreevatsa, A., ´ Prendki, J., Soergel, D., Goedeckemeyer, A., Gierke, W., Jafari, M., Gaba, M., Wiesner, J., Wright, D. G., Wei, Y., Vashisht, H., Kulizhskaya, Y., Hoover, J., Le, M., Li, L., Iwuanyanwu, C., Liu, L., Ramirez, K., Khorlin, A., Cui, A., LIN, T., Wu, M., Aguilar, R., Pallo, K., Chakladar,

[21] A., Perng, G., Abellan, E. A., Zhang, M., Dasgupta, I., Kushman, N., Penchev, I., Repina, A., Wu, X., van der Weide, T., Ponnapalli, P., Kaplan, C., Simsa, J., Li, S., Dousse, O., Yang, F., Piper, J., Ie, N., Pasumarthi, R., Lintz, N., Vijayakumar, A., Andor, D., Valenzuela, P., Lui, M., Paduraru, C., Peng, D., Lee, K., Zhang, S., Greene, S., Nguyen, D. D., Kurylowicz, P., Hardin, C., Dixon, L., Janzer, L., Choo, K., Feng, Z., Zhang, B., Singhal, A., Du, D., McKinnon, D., Antropova, N., Bolukbasi, T., Keller, O., Reid, D., Finchelstein, D., Raad, M. A., Crocker, R., Hawkins, P., Dadashi, R., Gaffney, C., Franko, K., Bulanova, A., Leblond, R., Chung, S., Askham, H., Cobo, L. C., Xu, K., Fischer, F., Xu, J., Sorokin, C., Alberti, C., Lin, C.-C., Evans, C., Dimitriev, A., Forbes, H., Banarse, D., Tung, Z., Omernick, M., Bishop, C., Sterneck, R., Jain, R., Xia, J., Amid, E., Piccinno, F., Wang, X., Banzal, P., Mankowitz, D. J., Polozov, A., Krakovna, V., Brown, S., Bateni, M., Duan, D., Firoiu, V., Thotakuri, M., Natan, T., Geist, M., tan Girgin, S., Li, H., Ye, J., Roval, O., Tojo, R., Kwong, M., Lee-Thorp, J., Yew, C., Sinopalnikov, D., Ramos, S., Mellor, J., Sharma, A., Wu, K., Miller, D., Sonnerat, N., Vnukov, D., Greig, R., Beattie, J., Caveness, E., Bai, L., Eisenschlos, J., Korchemniy, A., Tsai, T., Jasarevic, M., Kong, W., Dao, P., Zheng, Z., Liu, F., Yang, F., Zhu, R., Teh, T. H., Sanmiya, J., Gladchenko, E., Trdin, N., Toyama, D., Rosen, E., Tavakkol, S., Xue, L., Elkind, C., Woodman, O., Carpenter, J., Papamakarios, G., Kemp, R., Kafle, S., Grunina, T., Sinha, R., Talbert, A., Wu, D., Owusu-Afriyie, D., Du, C., Thornton, C., Pont-Tuset, J., Narayana, P., Li, J., Fatehi, S., Wieting, J., Ajmeri, O., Uria, B., Ko, Y., Knight, L., Héliou, A., Niu, N., Gu, S., Pang, C., Li, Y., Levine, N., Stolovich, A., Santamaria-Fernandez, R., Goenka, S., Yustalim, W., Strudel, R., Elqursh, A., Deck, C., Lee, H., Li, Z., Levin, K., Hoffmann, R., Holtmann-Rice, D., Bachem, O., Arora, S., Koh, C., Yeganeh, S. H., Põder, S., Tariq, M., Sun, Y., Ionita, L., Seyedhosseini, M., Tafti, P., Liu, Z., Gulati, A., Liu, J., Ye, X., Chrzaszcz, B., Wang, L., Sethi, N., Li, T., Brown, B., Singh, S., Fan, W., Parisi, A., Stanton, J., Koverkathu, V., Choquette-Choo, C. A., Li, Y., Lu, T., Ittycheriah, A., Shroff, P., Varadarajan, M., Bahargam, S., Willoughby, R., Gaddy, D., Desjardins, G., Cornero, M., Robenek, B., Mittal, B., Albrecht, B., Shenoy, A., Moiseev, F., Jacobsson, H., Ghaffarkhah, A., Rivière, M., Walton, A., Crepy, C., Parrish, A., Zhou, Z., Farabet, C., Radebaugh, C., Srinivasan, P., van der Salm, C., Fidjeland, A., Scellato, S., Latorre-Chimoto, E., Klimczak-Plucinska, H., Bridson, D., de Cesare, D., ´ Hudson, T., Mendolicchio, P., Walker, L., Morris, A., Mauger, M., Guseynov, A., Reid, A., Odoom, S., Loher, L., Cotruta, V., Yenugula, M., Grewe, D., Petrushkina, A., Duerig, T., Sanchez, A., Yadlowsky, S., Shen, A., Globerson, A., Webb, L., Dua, S., Li, D., Bhupatiraju, S., Hurt, D., Qureshi, H., Agarwal, A., Shani, T., Eyal,

[22] M., Khare, A., Belle, S. R., Wang, L., Tekur, C., Kale,

[23] M. S., Wei, J., Sang, R., Saeta, B., Liechty, T., Sun, Y., Zhao, Y., Lee, S., Nayak, P., Fritz, D., Vuyyuru, M. R., Aslanides, J., Vyas, N., Wicke, M., Ma, X., Eltyshev, E., Martin, N., Cate, H., Manyika, J., Amiri, K., Kim, Y., Xiong, X., Kang, K., Luisier, F., Tripuraneni, N., Madras, D., Guo, M., Waters, A., Wang, O., Ainslie, J., Baldridge, J., Zhang, H., Pruthi, G., Bauer, J., Yang, F., Mansour, R., Gelman, J., Xu, Y., Polovets, G., Liu, J., Cai, H., Chen, W., Sheng, X., Xue, E., Ozair, S., Angermueller, C., Li, X., Sinha, A., Wang, W., Wiesinger, J., Koukoumidis, E., Tian, Y., Iyer, A., Gurumurthy, M., Goldenson, M., Shah, P., Blake, M., Yu, H., Urbanowicz, A., Palomaki, J., Fernando, C., Durden, K., Mehta, H., Momchev, N., Rahimtoroghi, E., Georgaki, M., Raul, A., Ruder, S., Redshaw, M., Lee, J., Zhou, D., Jalan, K., Li, D., Hechtman, B., Schuh, P., Nasr, M., Milan, K., Mikulik, V., Franco, J., Green, T., Nguyen, N., Kelley, J., Mahendru, A., Hu, A., Howland, J., Vargas, B., Hui, J., Bansal, K., Rao, V., Ghiya, R., Wang, E., Ye, K., Sarr, J. M., Preston, M. M., Elish, M., Li, S., Kaku, A., Gupta, J., Pasupat, I., Juan, D.-C., Someswar, M., M., T., Chen, X., Amini, A., Fabrikant, A., Chu, E., Dong, X., Muthal, A., Buthpitiya, S., Jauhari, S., Hua, N., Khandelwal, U., Hitron, A., Ren, J., Rinaldi, L., Drath, S., Dabush, A., Jiang, N.-J., Godhia, H., Sachs, U., Chen, A., Fan, Y., Taitelbaum, H., Noga, H., Dai, Z., Wang, J., Liang, C., Hamer, J., Ferng, C.-S., Elkind, C., Atias, A., Lee, P., Listík, V., Carlen, M., van de Kerkhof, J., Pikus, M., Zaher, K., Müller, P., Zykova, S., Stefanec, R., Gatsko, V., Hirnschall, C., Sethi, A., Xu, X. F., Ahuja, C., Tsai, B., Stefanoiu, A., Feng, B., Dhandhania, K., Katyal, M., Gupta, A., Parulekar, A., Pitta, D., Zhao, J., Bhatia, V., Bhavnani, Y., Alhadlaq, O., Li, X., Danenberg, P., Tu, D., Pine, A., Filippova, V., Ghosh, A., Limonchik, B., Urala, B., Lanka, C. K., Clive, D., Sun, Y., Li, E., Wu, H., Hongtongsak, K., Li, I., Thakkar, K., Omarov, K., Majmundar, K., Alverson, M., Kucharski, M., Patel, M., Jain, M., Zabelin, M., Pelagatti, P., Kohli, R., Kumar, S., Kim, J., Sankar, S., Shah, V., Ramachandruni, L., Zeng, X., Bariach, B., Weidinger, L., Vu, T., Andreev, A., He, A., Hui, K., Kashem, S., Subramanya, A., Hsiao, S., Hassabis, D., Kavukcuoglu, K., Sadovsky, A., Le, Q., Strohman, T., Wu, Y., Petrov, S., Dean, J., and Vinyals, O. Gemini: A family of highly capable multimodal models, 2024a. URL <https://arxiv.org/abs/2312.11805>. Gulati, A., Tanzer, G., Vincent, D., Pan, Z., Wang, S., Mariooryad, S., Ding, Y., Geng, X., Alcober, F., Frostig, R., Omernick, M., Walker, L., Paduraru, C., Sorokin, C., Tacchetti, A., Gaffney, C., Daruki, S., Sercinoglu, O., Gleicher, Z., Love, J., Voigtlaender, P., Jain, R., Surita, D., Kumar, A., Winkler, S., Caton, J., Brock, A., Dalmia,

[24] Team, G., Georgiev, P., Lei, V. I., Burnell, R., Bai, L., G., Mohamed, K., Blevins, R., Ahn, J., Zhu, T., Kaw-

[25] intiranon, K., Firat, O., Gu, Y., Zhang, Y., Rahtz, M., Faruqui, M., Clay, N., Gilmer, J., Co-Reyes, J., Penchev, I., Zhu, R., Morioka, N., Hui, K., Haridasan, K., Campos, V., Mahdieh, M., Guo, M., Hassan, S., Kilgour, K., Vezer, A., Cheng, H.-T., de Liedekerke, R., Goyal, S., Barham, P., Strouse, D., Noury, S., Adler, J., Sundararajan, M., Vikram, S., Lepikhin, D., Paganini, M., Garcia, X., Yang, F., Valter, D., Trebacz, M., Vodrahalli, K., Asawaroengchai, C., Ring, R., Kalb, N., Soares, L. B., Brahma, S., Steiner, D., Yu, T., Mentzer, F., He, A., Gonzalez, L., Xu, B., Kaufman, R. L., Shafey, L. E., Oh, J., Hennigan, T., van den Driessche, G., Odoom, S., Lucic, M., Roelofs, B., Lall, S., Marathe, A., Chan, B., Ontanon, S., He, L., Teplyashin, D., Lai, J., Crone, P., Damoc, B., Ho, L., Riedel, S., Lenc, K., Yeh, C.-K., Chowdhery, A., Xu, Y., Kazemi, M., Amid, E., Petrushkina, A., Swersky, K., Khodaei, A., Chen, G., Larkin, C., Pinto, M., Yan, G., Badia, A. P., Patil, P., Hansen, S., Orr, D., Arnold, S. M. R., Grimstad, J., Dai, A., Douglas, S., Sinha, R., Yadav, V., Chen, X., Gribovskaya, E., Austin, J., Zhao, J., Patel, K., Komarek, P., Austin, S., Borgeaud, S., Friso, L., Goyal, A., Caine, B., Cao, K., Chung, D.-W., Lamm, M., Barth-Maron, G., Kagohara, T., Olszewska, K., Chen, M., Shivakumar, K., Agarwal, R., Godhia, H., Rajwar, R., Snaider, J., Dotiwalla, X., Liu, Y., Barua, A., Ungureanu, V., Zhang, Y., Batsaikhan, B.-O., Wirth, M., Qin, J., Danihelka, I., Doshi, T., Chadwick, M., Chen, J., Jain, S., Le, Q., Kar, A., Gurumurthy, M., Li, C., Sang, R., Liu, F., Lamprou, L., Munoz, R., Lintz, N., Mehta, H., Howard, H., Reynolds, M., Aroyo, L., Wang, Q., Blanco, L., Cassirer, A., Griffith, J., Das, D., Lee, S., Sygnowski, J., Fisher, Z., Besley, J., Powell, R., Ahmed, Z., Paulus, D., Reitter, D., Borsos, Z., Joshi, R., Pope, A., Hand, S., Selo, V., Jain, V., Sethi, N., Goel, M., Makino, T., May, R., Yang, Z., Schalkwyk, J., Butterfield, C., Hauth, A., Goldin, A., Hawkins, W., Senter, E., Brin, S., Woodman, O., Ritter, M., Noland, E., Giang, M., Bolina, V., Lee, L., Blyth, T., Mackinnon, I., Reid, M., Sarvana, O., Silver, D., Chen, A., Wang, L., Maggiore, L., Chang, O., Attaluri, N., Thornton, G., Chiu, C.-C., Bunyan, O., Levine, N., Chung, T., Eltyshev, E., Si, X., Lillicrap, T., Brady, D., Aggarwal, V., Wu, B., Xu, Y., McIlroy, R., Badola, K., Sandhu, P., Moreira, E., Stokowiec, W., Hemsley, R., Li, D., Tudor, A., Shyam, P., Rahimtoroghi, E., Haykal, S., Sprechmann, P., Zhou, X., Mincu, D., Li, Y., Addanki, R., Krishna, K., Wu, X., Frechette, A., Eyal, M., Dafoe, A., Lacey, D., Whang, J., Avrahami, T., Zhang, Y., Taropa, E., Lin, H., Toyama, D., Rutherford, E., Sano, M., Choe, H., Tomala, A., Safranek-Shrader, C., Kassner, N., Pajarskas, M., Harvey, M., Sechrist, S., Fortunato, M., Lyu, C., Elsayed, G., Kuang, C., Lottes, J., Chu, E., Jia, C., Chen, C.-W., Humphreys, P., Baumli, K., Tao, C., Samuel, R., dos Santos, C. N., Andreassen, A., Rakicevi ´ c, N., Grewe, ´

[26] S., Sheahan, H., Barr, I., Miao, Y., Natsev, P., Devlin, J., Behbahani, F., Prost, F., Sun, Y., Myaskovsky, A., Pillai, T. S., Hurt, D., Lazaridou, A., Xiong, X., Zheng, C., Pardo, F., Li, X., Horgan, D., Stanton, J., Ambar, M., Xia, F., Lince, A., Wang, M., Mustafa, B., Webson, A., Lee, H., Anil, R., Wicke, M., Dozat, T., Sinha, A., Piqueras, E., Dabir, E., Upadhyay, S., Boral, A., Hendricks, L. A., Fry, C., Djolonga, J., Su, Y., Walker, J., Labanowski, J., Huang, R., Misra, V., Chen, J., Skerry-Ryan, R., Singh, A., Rijhwani, S., Yu, D., Castro-Ros, A., Changpinyo, B., Datta, R., Bagri, S., Hrafnkelsson, A. M., Maggioni, M., Zheng, D., Sulsky, Y., Hou, S., Paine, T. L., Yang, A., Riesa, J., Rogozinska, D., Marcus, D., Badawy, D. E., Zhang, Q., Wang, L., Miller, H., Greer, J., Sjos, L. L., Nova, A., Zen, H., Chaabouni, R., Rosca, M., Jiang, J., Chen, C., Liu, R., Sainath, T., Krikun, M., Polozov, A., Lespiau, J.-B., Newlan, J., Cankara, Z., Kwak, S., Xu, Y., Chen, P., Coenen, A., Meyer, C., Tsihlas, K., Ma, A., Gottweis, J., Xing, J., Gu, C., Miao, J., Frank, C., Cankara, Z., Ganapathy, S., Dasgupta, I., Hughes-Fitt, S., Chen, H., Reid, D., Rong, K., Fan, H., van Amersfoort, J., Zhuang, V., Cohen, A., Gu, S. S., Mohananey, A., Ilic, A., Tobin, T., Wieting, J., Bortsova, A., Thacker, P., Wang, E., Caveness, E., Chiu, J., Sezener, E., Kaskasoli, A., Baker, S., Millican, K., Elhawaty, M., Aisopos, K., Lebsack, C., Byrd, N., Dai, H., Jia, W., Wiethoff, M., Davoodi, E., Weston, A., Yagati, L., Ahuja, A., Gao, I., Pundak, G., Zhang, S., Azzam, M., Sim, K. C., Caelles, S., Keeling, J., Sharma, A., Swing, A., Li, Y., Liu, C., Bostock, C. G., Bansal, Y., Nado, Z., Anand, A., Lipschultz, J., Karmarkar, A., Proleev, L., Ittycheriah, A., Yeganeh, S. H., Polovets, G., Faust, A., Sun, J., Rrustemi, A., Li, P., Shivanna, R., Liu, J., Welty, C., Lebron, F., Baddepudi, A., Krause, S., Parisotto, E., Soricut, R., Xu, Z., Bloxwich, D., Johnson, M., Neyshabur, B., Mao-Jones, J., Wang, R., Ramasesh, V., Abbas, Z., Guez, A., Segal, C., Nguyen, D. D., Svensson, J., Hou, L., York, S., Milan, K., Bridgers, S., Gworek, W., Tagliasacchi, M., Lee-Thorp, J., Chang, M., Guseynov, A., Hartman, A. J., Kwong, M., Zhao, R., Kashem, S., Cole, E., Miech, A., Tanburn, R., Phuong, M., Pavetic, F., Cevey, S., Comanescu, R., Ives, R., Yang, S., Du, C., Li, B., Zhang, Z., Iinuma, M., Hu, C. H., Roy, A., Bijwadia, S., Zhu, Z., Martins, D., Saputro, R., Gergely, A., Zheng, S., Jia, D., Antonoglou, I., Sadovsky, A., Gu, S., Bi, Y., Andreev, A., Samangooei, S., Khan, M., Kocisky, T., Filos, A., Kumar, C., Bishop, C., Yu, A., Hodkinson, S., Mittal, S., Shah, P., Moufarek, A., Cheng, Y., Bloniarz, A., Lee, J., Pejman, P., Michel, P., Spencer, S., Feinberg, V., Xiong, X., Savinov, N., Smith, C., Shakeri, S., Tran, D., Chesus, M., Bohnet, B., Tucker, G., von Glehn, T., Muir, C., Mao, Y., Kazawa, H., Slone, A., Soparkar, K., Shrivastava, D., Cobon-Kerr, J., Sharman, M., Pavagadhi, J., Araya, C., Misiunas, K., Ghelani, N., Laskin, M., Barker, D., Li, Q., Briukhov, A., Houlsby, N., Glaese, M., Lakshminarayanan, B., Schucher, N., Tang, Y., Collins, E., Lim, H., Feng, F., Recasens, A., Lai, G., Magni, A., Cao, N. D., Siddhant, A., Ashwood, Z., Orbay, J., Dehghani, M., Brennan, J., He, Y., Xu, K., Gao, Y., Saroufim, C., Molloy, J., Wu, X., Arnold, S., Chang, S., Schrittwieser, J., Buchatskaya, E., Radpour, S., Polacek, M., Giordano, S., Bapna, A., Tokumine, S., Hellendoorn, V., Sottiaux, T., Cogan, S., Severyn, A., Saleh, M., Thakoor, S., Shefey, L., Qiao, S., Gaba, M., yiin Chang, S., Swanson, C., Zhang, B., Lee, B., Rubenstein,
  - P. K., Song, G., Kwiatkowski, T., Koop, A., Kannan, A., Kao, D., Schuh, P., Stjerngren, A., Ghiasi, G., Gibson, G., Vilnis, L., Yuan, Y., Ferreira, F. T., Kamath, A., Klimenko, T., Franko, K., Xiao, K., Bhattacharya, I., Patel, M., Wang, R., Morris, A., Strudel, R., Sharma, V., Choy, P., Hashemi, S. H., Landon, J., Finkelstein, M., Jhakra, P., Frye, J., Barnes, M., Mauger, M., Daun, D., Baatarsukh, K., Tung, M., Farhan, W., Michalewski, H., Viola, F., de Chaumont Quitry, F., Lan, C. L., Hudson, T., Wang, Q., Fischer, F., Zheng, I., White, E., Dragan, A., baptiste Alayrac, J., Ni, E., Pritzel, A., Iwanicki, A., Isard, M., Bulanova, A., Zilka, L., Dyer, E., Sachan, D., Srinivasan, S., Muckenhirn, H., Cai, H., Mandhane, A., Tariq, M., Rae, J. W., Wang, G., Ayoub, K., FitzGerald, N., Zhao, Y., Han, W., Alberti, C., Garrette, D., Krishnakumar, K., Gimenez, M., Levskaya, A., Sohn, D., Matak, J., Iturrate, I., Chang, M. B., Xiang, J., Cao, Y., Ranka, N., Brown, G., Hutter, A., Mirrokni, V., Chen, N., Yao, K., Egyed, Z., Galilee, F., Liechty, T., Kallakuri, P., Palmer, E., Ghemawat, S., Liu, J., Tao, D., Thornton, C., Green, T., Jasarevic, M., Lin, S., Cotruta, V., Tan, Y.-X., Fiedel, N., Yu, H., Chi, E., Neitz, A., Heitkaemper, J., Sinha, A., Zhou, D., Sun, Y., Kaed, C., Hulse, B., Mishra, S., Georgaki, M., Kudugunta, S., Farabet, C., Shafran, I., Vlasic, D., Tsitsulin, A., Ananthanarayanan, R., Carin, A., Su, G., Sun, P., V, S., Carvajal, G., Broder, J., Comsa, I., Repina, A., Wong, W., Chen, W. W., Hawkins, P., Filonov, E., Loher, L., Hirnschall, C., Wang, W., Ye, J., Burns, A., Cate, H., Wright, D. G., Piccinini, F., Zhang, L., Lin, C.-C., Gog, I., Kulizhskaya, Y., Sreevatsa, A., Song, S., Cobo, L. C., Iyer, A., Tekur, C., Garrido, G., Xiao, Z., Kemp, R., Zheng, H. S., Li, H., Agarwal, A., Ngani, C., Goshvadi, K., Santamaria-Fernandez, R., Fica, W., Chen, X., Gorgolewski, C., Sun, S., Garg, R., Ye, X., Eslami, S. M. A., Hua, N., Simon, J., Joshi, P., Kim, Y., Tenney, I., Potluri, S., Thiet, L. N., Yuan, Q., Luisier, F., Chronopoulou, A., Scellato, S., Srinivasan, P., Chen, M., Koverkathu, V., Dalibard, V., Xu, Y., Saeta, B., Anderson, K., Sellam, T., Fernando, N., Huot, F., Jung, J., Varadarajan, M., Quinn, M., Raul, A., Le, M., Habalov, R., Clark, J., Jalan, K., Bullard, K., Singhal, A., Luong, T., Wang, B., Rajayogam, S., Eisenschlos, J., Jia, J., Finchelstein, D., Yakubovich, A., Balle, D., Fink, M., Agarwal, S., Li, J., Dvijotham, D., Pal, S., Kang, K.,

[27] Konzelmann, J., Beattie, J., Dousse, O., Wu, D., Crocker, R., Elkind, C., Jonnalagadda, S. R., Lee, J., Holtmann-Rice, D., Kallarackal, K., Liu, R., Vnukov, D., Vats, N., Invernizzi, L., Jafari, M., Zhou, H., Taylor, L., Prendki, J., Wu, M., Eccles, T., Liu, T., Kopparapu, K., Beaufays, F., Angermueller, C., Marzoca, A., Sarcar, S., Dib, H., Stanway, J., Perbet, F., Trdin, N., Sterneck, R., Khorlin, A., Li, D., Wu, X., Goenka, S., Madras, D., Goldshtein, S., Gierke, W., Zhou, T., Liu, Y., Liang, Y., White, A., Li, Y., Singh, S., Bahargam, S., Epstein, M., Basu, S., Lao, L., Ozturel, A., Crous, C., Zhai, A., Lu, H., Tung, Z., Gaur, N., Walton, A., Dixon, L., Zhang, M., Globerson, A., Uy, G., Bolt, A., Wiles, O., Nasr, M., Shumailov, I., Selvi, M., Piccinno, F., Aguilar, R., McCarthy, S., Khalman, M., Shukla, M., Galic, V., Carpenter, J., Villela, K., Zhang, H., Richardson, H., Martens, J., Bosnjak, M., Belle, S. R., Seibert, J., Alnahlawi, M., McWilliams, B., Singh, S., Louis, A., Ding, W., Popovici, D., Simicich, L., Knight, L., Mehta, P., Gupta, N., Shi, C., Fatehi, S., Mitrovic, J., Grills, A., Pagadora, J., Munkhdalai, T., Petrova, D., Eisenbud, D., Zhang, Z., Yates, D., Mittal, B., Tripuraneni, N., Assael, Y., Brovelli, T., Jain, P., Velimirovic, M., Akbulut, C., Mu, J., Macherey, W., Kumar, R., Xu, J., Qureshi, H., Comanici, G., Wiesner, J., Gong, Z., Ruddock, A., Bauer, M., Felt, N., GP, A., Arnab, A., Zelle, D., Rothfuss, J., Rosgen, B., Shenoy, A., Seybold, B., Li, X., Mudigonda, J., Erdogan, G., Xia, J., Simsa, J., Michi, A., Yao, Y., Yew, C., Kan, S., Caswell, I., Radebaugh, C., Elisseeff, A., Valenzuela, P., McKinney, K., Paterson, K., Cui, A., Latorre-Chimoto, E., Kim, S., Zeng, W., Durden, K., Ponnapalli, P., Sosea, T., Choquette-Choo, C. A., Manyika, J., Robenek, B., Vashisht, H., Pereira, S., Lam, H., Velic, M., Owusu-Afriyie, D., Lee, K., Bolukbasi, T., Parrish, A., Lu, S., Park, J., Venkatraman, B., Talbert, A., Rosique, L., Cheng, Y., Sozanschi, A., Paszke, A., Kumar, P., Austin, J., Li, L., Salama, K., Perz, B., Kim, W., Dukkipati, N., Baryshnikov, A., Kaplanis, C., Sheng, X., Chervonyi, Y., Unlu, C., de Las Casas, D., Askham, H., Tunyasuvunakool, K., Gimeno, F., Poder, S., Kwak, C., Miecnikowski, M., Mirrokni, V., Dimitriev, A., Parisi, A., Liu, D., Tsai, T., Shevlane, T., Kouridi, C., Garmon, D., Goedeckemeyer, A., Brown, A. R., Vijayakumar, A., Elqursh, A., Jazayeri, S., Huang, J., Carthy, S. M., Hoover, J., Kim, L., Kumar, S., Chen, W., Biles, C., Bingham, G., Rosen, E., Wang, L., Tan, Q., Engel, D., Pongetti, F., de Cesare, D., Hwang, D., Yu, L., Pullman, J., Narayanan, S., Levin, K., Gopal, S., Li, M., Aharoni, A., Trinh, T., Lo, J., Casagrande, N., Vij, R., Matthey, L., Ramadhana, B., Matthews, A., Carey, C., Johnson, M., Goranova, K., Shah, R., Ashraf, S., Dasgupta, K., Larsen, R., Wang, Y., Vuyyuru, M. R., Jiang, C., Ijazi, J., Osawa, K., Smith, C., Boppana, R. S., Bilal, T., Koizumi, Y., Xu, Y., Altun, Y., Shabat, N., Bariach, B., Korchemniy, A., Choo, K., Ronneberger, O., Iwuanyanwu, C., Zhao, S., Soergel, D., Hsieh, C.-J., Cai, I., Iqbal, S., Sundermeyer, M., Chen, Z., Bursztein, E., Malaviya, C., Biadsy, F., Shroff, P., Dhillon, I., Latkar, T., Dyer, C., Forbes, H., Nicosia, M., Nikolaev, V., Greene, S., Georgiev, M., Wang, P., Martin, N., Sedghi, H., Zhang, J., Banzal, P., Fritz, D., Rao, V., Wang, X., Zhang, J., Patraucean, V., Du, D., Mordatch, I., Jurin, I., Liu, L., Dubey, A., Mohan, A., Nowakowski, J., Ion, V.-D., Wei, N., Tojo, R., Raad, M. A., Hudson,
  - D. A., Keshava, V., Agrawal, S., Ramirez, K., Wu, Z., Nguyen, H., Liu, J., Sewak, M., Petrini, B., Choi, D., Philips, I., Wang, Z., Bica, I., Garg, A., Wilkiewicz, J., Agrawal, P., Li, X., Guo, D., Xue, E., Shaik, N., Leach, A., Khan, S. M., Wiesinger, J., Jerome, S., Chakladar, A., Wang, A. W., Ornduff, T., Abu, F., Ghaffarkhah, A., Wainwright, M., Cortes, M., Liu, F., Maynez, J., Terzis, A., Samangouei, P., Mansour, R., Kepa, T., Aubet, F.-X., Algymr, A., Banica, D., Weisz, A., Orban, A., Senges, A., Andrejczuk, E., Geller, M., Santo, N. D., Anklin, V., Merey, M. A., Baeuml, M., Strohman, T., Bai, J., Petrov, S., Wu, Y., Hassabis, D., Kavukcuoglu, K., Dean, J., and Vinyals, O. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024b. URL <https://arxiv.org/abs/2403.05530>. Wang, H., Fu, T., Du, Y., Gao, W., Huang, K., Liu, Z., Chandak, P., Liu, S., Van Katwyk, P., Deac, A., et al. Scientific discovery in the age of artificial intelligence. *Nature*, 620(7972):47–60, 2023. Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., Chi, E. H., Hashimoto, T., Vinyals, O., Liang, P., Dean, J., and Fedus, W. Emergent abilities of large language models, 2022a. URL [https:](https://arxiv.org/abs/2206.07682) [//arxiv.org/abs/2206.07682](https://arxiv.org/abs/2206.07682). Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., et al. Emergent abilities of large language models. *arXiv preprint arXiv:2206.07682*, 2022b. Wu, T.-Y. and Lo, P.-Y. U-shaped and inverted-u scaling behind emergent abilities of large language models, 2024. URL <https://arxiv.org/abs/2410.01692>. Wu, Y., Sun, Z., Li, S., Welleck, S., and Yang, Y. Inference scaling laws: An empirical analysis of computeoptimal inference for problem-solving with language models, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2408.00724) [2408.00724](https://arxiv.org/abs/2408.00724). Xiong, W., Liu, J., Molybog, I., Zhang, H., Bhargava, P., Hou, R., Martin, L., Rungta, R., Sankararaman, K. A., Oguz, B., Khabsa, M., Fang, H., Mehdad, Y., Narang, S., Malik, K., Fan, A., Bhosale, S., Edunov, S., Lewis, M., Wang, S., and Ma, H. Effective long-context scaling

[28] of foundation models, 2023. URL [https://arxiv.](https://arxiv.org/abs/2309.16039)

[org/abs/2309.16039](https://arxiv.org/abs/2309.16039). Zhai, X., Kolesnikov, A., Houlsby, N., and Beyer, L. Scaling vision transformers. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 12104–12113, 2022. Zhang, B., Liu, Z., Cherry, C., and Firat, O. When scaling meets llm finetuning: The effect of data, model and finetuning method, 2024. URL [https://arxiv.org/](https://arxiv.org/abs/2402.17193) [abs/2402.17193](https://arxiv.org/abs/2402.17193).
## A. Clarification of How Large Language Monkeys and Best-of-N Jailbreaking Sampled Data

In this manuscript, we used the phrasing of "independent attempts," which is not fully correct. In this appendix section, we clarify why we chose this terminology, what likely impacts we believe this inaccuracy may have had on our results, and how to correct the paper accordingly.

Large Language Monkeys [\(Brown et al.,](#page-9-4) [2024\)](#page-9-4) indeed drew 10, 000 independent attempts per problem, but Best-of-N Jailbreaking [\(Hughes et al.,](#page-12-2) [2024\)](#page-12-2) sampled data slightly different: for each problem, jailbreaking attempts were drawn until either a successful jailbreak was obtained or until a maximum limit of 10, 000 attempts was hit. Samples were also drawn in minibatches of size 60, making the (in)dependence of samples a bit tricky.

We omitted this nuance because it offers a second-order correction to our paper's main story while offering little additional insight. Neither of our theorems and none of our main text figures change. We suspect that this slightly different sampling procedure explains why, in Fig. [6,](#page-6-0) the estimated power law exponents between the least squares power law estimator and the distributional power law estimator deviate more significantly from identity for Best-of-N Jailbreaking than for Large Language Monkeys. A natural way to correct for this is to use a [beta-negative binomial distribution](https://en.wikipedia.org/wiki/Beta_negative_binomial_distribution) rather than a [beta-binomial distribution,](https://en.wikipedia.org/wiki/Beta-binomial_distribution) with an additional correction for the maximum number of attempts. For more information, please see Appendix [H.](#page-44-0)

## B. Estimating Success Rates Using [Chen et al.](#page-10-2) [\(2021\)](#page-10-2)'s Estimator

In this manuscript, we defined passi@k and ASRi@k as:

$$pass_i \otimes k \stackrel{\text{def}}{=} \mathbb{E}_{k \text{ Attempts}} [\text{II}[\text{At least 1 attempt by the model solves the } i\text{-th problem}]]$$
 $\text{ASR}_i \otimes k \stackrel{\text{def}}{=} \mathbb{E}_{k \text{ Attempts}} [\text{II}[\text{At least 1 attempt jailbreaks the model on the } i\text{-th prompt}]]$ 

Throughout this manuscript, to estimate passi@k and ASR@k, we used the unbiased and lower variance estimator introduced by [Chen et al.](#page-10-2) [\(2021\)](#page-10-2): for the <sup>i</sup>-th problem, we sampled <sup>n</sup> ≫ <sup>k</sup> attempts per problem, counted the number of successful attempts c, and then swept k to compute an estimate of passi@k for different k values:

$$\widehat{\text{pass}_i @k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \quad (12)$$

Two comments: Firstly, n as used here has no relationship with the number of problems in the benchmark (Sec. [1\)](#page-0-0), and secondly, our notation differs slightly from that of [Chen et al.](#page-10-2) [\(2021\)](#page-10-2), but the ideas are consistent. A numerically stable Python implementation of the estimator is provided in Fig. [8:](#page-24-1)

**def** estimate\_success\_rate\_at\_k\_per\_problem(n: **int**, c: **int**, k: **int**) -> **float**: """ :param n: number of total attempts on this problem. :param c: number of correct attempts on this problem. :param k: k in pass\_i@\$k\$. """ **if** n - c < k: **return** 1.0 **return** 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

Figure 8: A numerically stable unbiased estimator of passi@k, introduced by [Chen et al.](#page-10-2) [\(2021\)](#page-10-2).

To reiterate a point made by [Chen et al.](#page-10-2) [\(2021\)](#page-10-2), estimating passi@k as <sup>1</sup> − (1 − pass \i@1)<sup>k</sup> is biased (Fig. [9\)](#page-24-2).

![](_page_24_Figure_10.jpeg)

Figure 9: Bias of Estimators of passi@k. Numerical simulations show that estimating passi@k as <sup>1</sup> − (1 − pass \i@1)<sup>k</sup> is biased whereas the estimator of [Chen et al.](#page-10-2) [\(2021\)](#page-10-2) is not. For a mathematical proof of unbiasedness, see the original paper.

## C. Fitting Power Laws to Large Language Monkeys and Best-of-N Jailbreaking

We fit power laws to a subset of data from Large Language Monkeys [\(Brown et al.,](#page-9-4) [2024\)](#page-9-4) and from Best-of-N Jailbreaking [\(Hughes et al.,](#page-12-2) [2024\)](#page-12-2), specifically Pythia language models [\(Biderman et al.,](#page-9-5) [2023\)](#page-9-5) on the MATH benchmark [\(Hendrycks](#page-12-3) [et al.,](#page-12-3) [2021\)](#page-12-3) and frontier AI models – Claude, GPT4 [\(OpenAI et al.,](#page-13-1) [2024\)](#page-13-1), Gemini [\(Team et al.,](#page-16-0) [2024a](#page-16-0)[;b\)](#page-19-0) and Llama 3 [\(Grattafiori et al.,](#page-11-0) [2024\)](#page-11-0) – on the HarmBench jailbreaking benchmark [\(Mazeika et al.,](#page-13-4) [2024\)](#page-13-4). We show the functional forms and the fit parameters in Table [1](#page-25-1) and Table [2](#page-25-1) respectively. To fit the parameters, for Large Language Monkeys, we simply minimized the squared error between the actual and predicted − log(passD@k), and for Best-of-N Jailbreaking, we similarly minimized the squared error between the actual and predicted − log(ASRD@k)).

Note: Llama 3 8B IT does not exhibit power law scaling under Best-of-N Jailbreaking (shown in Fig. [1,](#page-1-0) bottom).

| Model  |      | Benchmark | a     | b     |
|--------|------|-----------|-------|-------|
| Pythia | 70M  | MATH      | 8.026 | 0.194 |
| Pythia | 160M | MATH      | 6.591 | 0.280 |
| Pythia | 410M | MATH      | 5.524 | 0.286 |
| Pythia | 1B   | MATH      | 5.452 | 0.315 |
| Pythia | 2.8B | MATH      | 4.104 | 0.336 |
| Pythia | 6.9B | MATH      | 4.255 | 0.348 |
| Pythia | 12B  | MATH      | 4.113 | 0.370 |

Table 1: Large Language Monkeys [\(Brown et al.,](#page-9-4) [2024\)](#page-9-4) fitted power law parameters on 128 mathematical problems from MATH [\(Hendrycks et al.,](#page-12-3) [2021\)](#page-12-3). Functional Form: − log(passD@k) = a k−<sup>b</sup> .

| Model  |     |        | Modality |       | a b   |
|--------|-----|--------|----------|-------|-------|
| Claude | 3.5 | Opus   | Text     | 2.630 | 0.448 |
| Claude | 3.5 | Sonnet | Text     | 3.436 | 0.312 |
| GPT4o  |     |        | Text     | 3.639 | 0.395 |
| GPT4o  |     | Mini   | Text     | 3.637 | 0.492 |
| Gemini | 1.5 | Flash  | Text     | 6.158 | 0.303 |
| Gemini | 1.5 | Pro    | Text     | 6.296 | 0.256 |
| Llama  | 3   | 8B IT  | Text     |       | – –   |

Table 2: Best-of-N Jailbreaking [\(Hughes et al.,](#page-12-2) [2024\)](#page-12-2) fitted power law parameters on text jailbreak prompts from Harm-Bench [\(Mazeika et al.,](#page-13-4) [2024\)](#page-13-4). Functional Form: − log(ASRD@k) = a k−<sup>b</sup> . Note: Llama 3 8B Instruction Tuned (IT) does not exhibit power law scaling.

## D. Mathematical Equivalence Between Coverage and Average Success Rate

[Brown et al.](#page-9-4) [\(2024\)](#page-9-4) and [Hughes et al.](#page-12-2) [\(2024\)](#page-12-2) phrase their research in terms of "coverage", defined as the fraction of problems that can be solved or the fraction of prompts that can jailbreak a model, but as [Brown et al.](#page-9-4) [\(2024\)](#page-9-4) comment and we here derive, the coverage is mathematically equivalent to the average passi@k (equivalently, ASR@k. due to two simple probabilistic primitives: (1) linearity of expectation, (2) the expectation of an indictor random variable of some event is the probability of said event and (3) the definition of passi@k:

$$\begin{aligned} \mathbb{E} \left[ \text{Coverage} \right] &\stackrel{\text{def}}{=} \mathbb{E}_{\text{Problems}} \left[ \text{Fraction of Problems Solved After } k \text{ Attempts} \right] \\ &= \mathbb{E}_{\text{Problems}} \left[ \mathbb{E}_{\text{Attempts}|\text{Problem}} \left[ \mathbb{I}[\text{Problem Solved After } k \text{ Attempts}] \right] \right] \\ &= \mathbb{E}_{\text{Problems}} \left[ \text{pass}_{\text{problem} \otimes k} \right] \\ &= \text{pass}_{\mathcal{D}} \otimes k \end{aligned}$$

In our work, we prefer phrasing along the lines of "success rate" over "coverage" because success rate avoids coverage's binary implication that each problem/prompt is either "solved" or "not solved".

## E. Aggregate Power Laws from a Probability Distribution over Exponential Functions

## E.1. Preliminaries: Power Laws from Weighted Exponential Functions

A known result is that power laws can emerge from appropriately weighted sums of exponential functions, e.g., [\(Bochud &](#page-9-21) [Challet,](#page-9-21) [2006;](#page-9-21) [Elkies,](#page-10-16) [2016;](#page-10-16) [Bousquet et al.,](#page-9-9) [2020\)](#page-9-9). For a concrete example with a short proof:

$$x^{-r} = \frac{1}{\Gamma(r)} \int_0^\infty p^{r-1} e^{-px} dp, \quad (13)$$

where Γ(r) def = R ∞ 0 s r−1 e <sup>−</sup><sup>s</sup> ds is the [Gamma function.](https://en.wikipedia.org/wiki/Gamma_function) The proof is via u-substitution u def = p x:

$$\frac{1}{\Gamma(r)} \int_0^\infty p^{r-1} e^{-px} dp = \frac{1}{\Gamma(r)} \int_0^\infty (u/x)^{r-1} e^{-u} \frac{du}{x} \quad (14)$$

$$= \frac{1}{\Gamma(r)} x^{-r} \int_0^\infty u^{r-1} e^{-u} du \quad (15)$$

$$= \frac{1}{\Gamma(r)} x^{-r} \Gamma(r) \quad (16)$$

$$= x^{-r} \quad (17)$$

In our particular context, we are interested in the scaling with k of the expected success rate over problems sampled from the benchmark's data distribution:

$$\text{pass}_{\mathcal{D}} @k \stackrel{\text{def}}{=} \mathbb{E}_{\text{pass}_i @1 \sim \mathcal{D}} \left[ \text{pass}_i @k \right] \quad (18)$$

distribution (over problems in a benchmark) of passi@k scores that yields power law scaling with respect to the number of attempts k:

$$-\log \left( \frac{1}{n} \sum_{i=1}^n \text{pass}_i @k \right) \approx ak^{-b}. \quad (19)$$

for constants a, b > 0.

# E.2. Delta Distribution: passi@1 ∼ <sup>δ</sup>(p), p ∈ (0, 1)

To start with a negative result, we will show that not all distributions of the per-problem success probabilities passi@1 yield aggregate power law scaling. Suppose that the model's passi@1 probabilities across the benchmarks' problems are all exactly <sup>p</sup> ∈ (0, 1). For brevity, let <sup>p</sup><sup>i</sup> def = passi@1. Then the aggregate success rate is:

$$\mathbb{E}_{p_i \sim \delta(p)}[\text{pass}_i @k] = 1 - \mathbb{E}_{p_i}[(1 - p_i)^k] \quad (20)$$

$$= 1 - \mathbb{E}_{p_i}[(1 - p_i)^k] \quad (20)$$

$$= \int_0^1 \delta(p) (1 - p_i)^k dp_i \quad (21)$$

$$\begin{aligned} & J_0 \\ &= (1-p)^k. \end{aligned} \tag{22}$$

Recalling that the expansion of log(·) for small <sup>x</sup> is − log(1 − <sup>x</sup>) = <sup>x</sup> <sup>+</sup> <sup>O</sup>(<sup>x</sup> 2 ), in our case, we obtain:

$$-\log \left( 1 - \mathbb{E}_{p_i \sim \delta(p)}[\text{pass@k}] \right) = (1-p)^k + O((1-p)^{2k}) = (1-p)^k + o((1-p)^k). \quad (23)$$

Thus, in the large k regime, we find the negative log aggregate success rate exhibits *exponential* scaling with k as we intuitively expect.

## E.3. Uniform Distribution: passi@1 ∼ Uniform(α, β)

Suppose passi@1 probabilities follow a uniform distribution Uniform(α, β) where <sup>0</sup> ≤ α < β ≤ <sup>1</sup>. The aggregate success rate after k attempts is defined as:

$$\text{pass}_{\text{Uniform}(\alpha, \beta)} @k \stackrel{\text{def}}{=} 1 - \mathbb{E}[(1 - p)^k].$$

If <sup>p</sup> ∼ Uniform(α, β), the expectation of (1 − <sup>p</sup>) k is:

$$\mathbb{E}[(1-p)^k] = \frac{1}{\beta - \alpha} \int_{\alpha}^{\beta} (1-p)^k dp.$$

Evaluating the integral gives:

$$\mathbb{E}[(1-p)^k] = \frac{(1-\alpha)^{k+1} - (1-\beta)^{k+1}}{(\beta-\alpha) \cdot (k+1)}.$$

Thus, the aggregate success rate becomes:

$$\text{passUniform}(\alpha, \beta) @k = 1 - \frac{(1 - \alpha)^{k+1} - (1 - \beta)^{k+1}}{(\beta - \alpha) \cdot (k + 1)}.$$

Case A: α > <sup>0</sup> If α > <sup>0</sup>, then both (1 − <sup>α</sup>) and (1 − <sup>β</sup>) are strictly less than <sup>1</sup>. As <sup>k</sup> → ∞, (1 − <sup>α</sup>) <sup>k</sup>+1 and (1 − <sup>β</sup>) k+1 decay exponentially. Hence:

$$\mathbb{E}[(1-p)^k] \sim \frac{(1-\alpha)^{k+1}}{(\beta-\alpha) \cdot (k+1)},$$

and passUniform(α,β) @k approaches 1 exponentially fast:

$$\text{passUniform}(\alpha, \beta) @k \sim 1 - \frac{(1 - \alpha)^{k+1}}{(\beta - \alpha) \cdot (k + 1)}.$$

Thus, the negative log of the aggregate success rate decays exponentially:

$$-\log\left(\text{pass}_{\text{Uniform}(\alpha, \beta)} @k\right) \sim e^{-\Omega(k)}.$$

Case B: α = 0 When α = 0, the uniform distribution is over [0, β]. In this case:

$$\mathbb{E}[(1-p)^k] = \frac{1}{\beta} \cdot \frac{1 - (1-\beta)^{k+1}}{k+1}.$$

For large <sup>k</sup>, (1 − <sup>β</sup>) <sup>k</sup>+1 becomes exponentially small, and:

$$\mathbb{E}[(1-p)^k] \sim \frac{1}{\beta} \cdot \frac{1}{k+1}.$$

The aggregate success rate is then:

$$\text{passUniform}(0, \beta) @k \sim 1 - \frac{1}{\beta \cdot k}.$$

The negative log exhibits power-law scaling:

$$-\log(\text{passUniform}(0, \beta) @k) \sim \frac{1}{\beta} \cdot \frac{1}{k}.$$

Special Case: Uniform(0, 1) If β = 1, the distribution is uniform on [0, 1]. In this case:

$$\mathbb{E}[(1-p)^k] = \frac{1}{k+1},$$

and the success rate becomes:

$$\text{passUniform}(0,1) @k = 1 - \frac{1}{k+1}.$$

For large k:

$$-\log(\text{pass}_{\text{Uniform}(0,1)} @k) \sim \frac{1}{k}.$$

## E.4. 2-Parameter Beta Distribution: passi@1 ∼ Beta(α, β)

Suppose that the model's passi@1 probabilities across the benchmark problems follow a Beta distribution:

$$\text{pass}_i @1 \sim \text{Beta}(\alpha, \beta)$$

The probability density function of this distribution over the support <sup>x</sup> ∈ (0, 1) is:

$$f(x; \alpha, \beta) \stackrel{\text{def}}{=} \frac{1}{B(\alpha, \beta)} x^{\alpha-1} (1-x)^{\beta-1}, \quad (24)$$

where α > <sup>0</sup>, β > <sup>0</sup> and <sup>B</sup>(·, ·) is the [Beta function.](https://en.wikipedia.org/wiki/Beta_function) For brevity, let <sup>p</sup><sup>i</sup> def = passi@1. Under our assumed Beta distribution:

$$\text{pass}_{\text{Beta}(\alpha, \beta)} @k \stackrel{\text{def}}{=} 1 - \mathbb{E}_{p_i \sim \text{Beta}(\alpha, \beta)} [(1 - p_i)^k] \quad (25)$$

$$= 1 - \int_0^1 \frac{p_i^{\alpha-1} (1-p_i)^{\beta-1}}{B(\alpha, \beta)} (1-p_i)^k dp_i \quad (26)$$

$$= 1 - \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} \frac{\Gamma(\alpha)\Gamma(\beta + k)}{\Gamma(\alpha + \beta + k)} \quad (27)$$

where Γ(·) is again the Gamma function. The Γ(α) terms cancel, and a standard asymptotic result of the gamma function for large k tells us that:

$$\frac{\Gamma(\beta + k)}{\Gamma(\alpha + \beta + k)} \sim k^{-\alpha}, \quad (28)$$

and thus:

$$\frac{\Gamma(\alpha + \beta)}{\Gamma(\beta)} \frac{\Gamma(\beta + k)}{\Gamma(\alpha + \beta + k)} \sim \frac{\Gamma(\alpha + \beta)}{\Gamma(\beta)} k^{-\alpha}. \quad (29)$$

Recalling again that the expansion of log(·) for small <sup>x</sup> is − log(1 − <sup>x</sup>) = <sup>x</sup> <sup>+</sup> <sup>O</sup>(<sup>x</sup> ), in our case, we obtain:

$$-\log \left( \text{pass}_{\mathcal{D}} @k \right) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\beta)} k^{-\alpha} + O(k^{-2\alpha}) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\beta)} k^{-\alpha} + o(k^{-\alpha}). \quad (30)$$

From this final result, we see that under a Beta distribution and in the large k regime, the negative log aggregate success rate exhibits polynomial (power-law) scaling with k for exponent α

## E.5. Kumaraswamy Distribution: passi@1 ∼ Kumaraswamy(α, β)

Next, suppose the model's passi@1 probabilities follow a Kumaraswamy distribution. The probability density function of this distribution over the support <sup>x</sup> ∈ (0, 1) is:

$$f(x; \alpha, \beta) \stackrel{\text{def}}{=} \alpha \beta x^{\alpha-1} (1 - x^\alpha)^{\beta-1} \quad (31)$$

Again for brevity, let p<sup>i</sup> def = passi@1. Under our assumed Kumaraswamy distribution:

$$\text{pass}_{\text{Kumaraswamy}(\alpha, \beta) @k \frac{\text{def}}{1}} 1 - \mathbb{E}_{p_i \sim \text{Kumaraswamy}(\alpha, \beta)} [(1 - p_i)^k] \quad (32)$$

$$= 1 - \int_0^1 (1-p)^k \cdot \alpha \beta p^{\alpha-1} (1-p^\alpha)^{\beta-1} dp. \quad (33)$$

Define the integral

$$I_k \stackrel{\text{def}}{=} \mathbb{E}((1-p)^k) = \int_0^1 (1-x)^k \alpha \beta x^{\alpha-1} (1-x^\alpha)^{\beta-1} dx. \quad (34)$$

We aim to analyze <sup>I</sup><sup>k</sup> for large <sup>k</sup>. Notice that (1−x) k is exponentially small in k unless x is very close to 0. Thus, intuitively, most of the contribution to <sup>I</sup><sup>k</sup> arises from <sup>x</sup> ∈ [0, O(1/k)].

Step 1: Split the integral into two parts. Fix a constant c > 0. Write

$$I_k = \int_0^{c/k} [\cdots] \, dx + \int_{c/k}^1 [\cdots] \, dx \stackrel{\text{def}}{=} I_{k,\text{left}} + I_{k,\text{right}},$$

where [· · · ] indicates the same integrand. In the region <sup>x</sup> ∈ [c/k, 1], we have (1 − <sup>x</sup>) <sup>k</sup> ≤ <sup>e</sup> <sup>−</sup>k x ≤ <sup>e</sup> −c . Hence Ik,right = O e −c . Since c can be made arbitrarily large, Ik,right becomes negligible compared to any polynomial in 1/k.

Step 2: Approximate the integrand in the small-<sup>x</sup> region. On [0, c/k], we use the approximation log(1 − <sup>x</sup>) = −<sup>x</sup> <sup>+</sup> <sup>O</sup>(<sup>x</sup> 2 ). Thus

$$(1-x)^k = \exp(k \log(1-x)) = \exp(-kx + O(kx^2)),$$

$$c^2/k = O(1/k), \text{ and } \exp(\epsilon) = 1 + O(\epsilon), \text{ we get}$$

Since <sup>x</sup> ≤ c/k implies k x<sup>2</sup> ≤ <sup>c</sup> <sup>2</sup>/k = O(1/k), and exp(ϵ) = 1 + O(ϵ), we get

$$(1-x)^k = \exp(-kx) \exp(O(1/k)) = \exp(-kx) \left(1 + O\left(\frac{1}{k}\right)\right).$$

Furthermore, since (1 − <sup>y</sup>) <sup>m</sup> = 1 − my <sup>+</sup> <sup>O</sup>(<sup>y</sup> 2 ), for small x

$$(1 - x^\alpha)^{\beta-1} = 1 - (\beta - 1)x^\alpha + O(x^{2\alpha}) = 1 + O(x^\alpha).$$

In the region <sup>x</sup> ≤ c/k, that error is <sup>O</sup> k −α . Hence, within the small-x region, the integrand

$$(1-x)^k \alpha \beta x^{\alpha-1} (1-x^\alpha)^{\beta-1}$$

can be approximated by

$$\alpha \beta x^{\alpha-1} e^{-kx} + O\left(k^{-\alpha} x^{\alpha-1} e^{-kx}\right).$$

Thus

$$I_{k,\text{left}} = \int_0^{c/k} \alpha \beta x^{\alpha-1} e^{-kx} dx + O\left(k^{-\alpha} \int_0^{c/k} x^{\alpha-1} e^{-kx} dx\right) + O(e^{-c}).$$

Step 3: Substitution u def <sup>=</sup> k x. To handle R c/k 0 x α−1 e <sup>−</sup>k x dx, we substitute u = k x. Then x = u/k, dx = du/k, and the upper limit x = c/k becomes u = c. Hence,

$$\begin{aligned} \int_0^{c/k} x^{\alpha-1} e^{-kx} dx &= \int_0^c \left(\frac{u}{k}\right)^{\alpha-1} e^{-u} \frac{du}{k} \\ &= k^{-\alpha} \int_0^c u^{\alpha-1} e^{-u} du. \end{aligned}$$

As <sup>c</sup> → ∞, R c 0 u α−1 e <sup>−</sup> <sup>u</sup> <sup>d</sup><sup>u</sup> → Γ(α), and for finite <sup>c</sup> the remainder is <sup>O</sup> e − c . Therefore,

$$\int_0^1 x^{\alpha-1} e^{-kx} dx = k^{-\alpha} \Gamma(\alpha) + O(k^{-\alpha} e^{-c}),$$

and absorbing the constant c into big-O notation gives

$$\int_0^1 x^{\alpha-1} e^{-kx} dx = k^{-\alpha} \Gamma(\alpha) + O(k^{-\alpha-\epsilon}) \quad \text{for some } \epsilon > 0.$$

Multiplying by the factor α β, we deduce that

$$I_k = \alpha \beta \Gamma(\alpha) k^{-\alpha} + O(k^{-\alpha-\epsilon}).$$

Step 4: Final conclusion for the success rate. Recall passKumaraswamy(α,β)@k = 1 − <sup>I</sup>k. Hence

pass<sub>Kumaraswamy(
$$\alpha, \beta$$
)</sub>@k = 1 -  $\alpha \beta \Gamma(\alpha) k^{-\alpha} + O(k^{-\alpha-\epsilon})$ .

Since this tends to 1, its negative log is governed by the magnitude of α β Γ(α) k <sup>−</sup><sup>α</sup>. Using the expansion − log(1 − <sup>y</sup>) = y + O(y 2 ) as <sup>y</sup> → <sup>0</sup>, we get

$$-\log \left( \text{pass}_{\text{Kumaraswamy}(\alpha, \beta) @k} \right) = \alpha \beta \Gamma(\alpha) k^{-\alpha} + o(k^{-\alpha}).$$

That is precisely polynomial (power-law) decay in the negative log success rate with exponent α.

E.6. Continuous Bernoulli Distribution: passi@1 ∼ ContinousBernoulli(λ)

Next, suppose the model's passi@1 probabilities follow a Continuous Bernoulli distribution. The probability density function of this distribution over the support <sup>x</sup> ∈ [0, 1] is:

$$f(x; \lambda) \stackrel{\text{def}}{=} C(\lambda) \lambda^x (1 - \lambda)^{1-x} \quad (35)$$

$$C(\lambda) \stackrel{\text{def}}{=} \begin{cases} 2 & \text{if } \lambda = 1/2 \\ \frac{2 \tanh^{-1}(1-2\lambda)}{1-2\lambda} & \text{otherwise} \end{cases} \quad (36)$$

The density can equivalently be rewritten in a more convenient form for our purposes:

$$f(x; \lambda) = C(\lambda)\lambda^x(1-\lambda)(1-\lambda)^{-x} = C(\lambda)(1-\lambda)\left(\frac{\lambda}{1-\lambda}\right)^x \quad (37)$$

Because the individual success probability is low in our data, we shall consider the small λ < 1/2 regime. We follow the same approach as with the Kumaraswamy distribution.

Step 1: Write the aggregate pass rate. The aggregate pass rate is defined as:

$$\text{passContinuousBernoulli}(\lambda) @k = 1 - I_k$$
, where  $I_k \stackrel{\text{def}}{=} \int_0^1 (1-p)^k f(p; \lambda) dp$ .

Substituting the density f(p; λ), we get:

$$I_k = \int_0^1 (1-p)^k C(\lambda) \lambda^p (1-\lambda)^{1-p} dp.$$

Step 2: Simplify using an exponential form. Using the exponential rewriting:

$$\lambda^p (1 - \lambda)^{1-p} = (1 - \lambda) \exp\left(p \log\left(\frac{\lambda}{1-\lambda}\right)\right),$$

the integral becomes:

$$I_k = C(\lambda) (1 - \lambda) \int_0^1 (1 - p)^k \exp\left(p \log\left(\frac{\lambda}{1-\lambda}\right)\right) dp.$$

Step 3: Dominance of the small-<sup>p</sup> region. For large <sup>k</sup>, (1 − <sup>p</sup>) <sup>k</sup> decays exponentially unless p is close to 0. Thus, the main contribution to the integral arises from the region <sup>p</sup> ∈ [0, c/k], where c > <sup>0</sup> is a constant. Decompose the integral:

$$I_k = \int_0^{c/k} [\cdots] dp + \int_{c/k}^1 [\cdots] dp \stackrel{\text{def}}{=} I_{k,\text{left}} + I_{k,\text{right}}.$$

In the region <sup>p</sup> ∈ [c/k, 1], we have (1 − <sup>p</sup>) <sup>k</sup> ≤ <sup>e</sup> <sup>−</sup>kp ≤ <sup>e</sup> −c , making Ik,right = O(e −c ), which is negligible compared to 1/k. Thus, we focus on Ik,left:

$$I_{k,\text{left}} = C(\lambda) (1 - \lambda) \int_0^{c/k} (1 - p)^k \exp\left(p \log\left(\frac{\lambda}{1-\lambda}\right)\right) dp.$$

Step 4: Approximate the integrand. For <sup>p</sup> ∈ [0, c/k], use the same approximations from the Kumaraswamy derivation:

$$(1-p)^k = e^{-kp} (1 + O(p)), \quad \exp\left(p \log\left(\frac{\lambda}{1-\lambda}\right)\right) = 1 + O(p).$$

Thus, the integrand becomes:

$$(1-p)^k \exp\left(p \log\left(\frac{\lambda}{1-\lambda}\right)\right) = e^{-kp} (1 + O(p)).$$

Step 5: Change of variables. Let u def <sup>=</sup> kp, so <sup>p</sup> <sup>=</sup> u/k and dp <sup>=</sup> du/k. The integral becomes:

$$I_{k,\text{left}} = C(\lambda) (1 - \lambda) \int_0^c e^{-u} (1 + O(u/k)) \frac{du}{k}.$$

Split the integral:

$$I_{k,\text{left}} = \frac{C(\lambda)(1-\lambda)}{k} \int_0^c e^{-u} du + O\left(\frac{1}{k^2}\right).$$

As <sup>c</sup> → ∞, R c 0 e <sup>−</sup><sup>u</sup> du → <sup>1</sup>. Thus:

$$I_{k,\text{left}} = \frac{C(\lambda)(1-\lambda)}{k} + O\left(\frac{1}{k^2}\right).$$

Since Ik,right = O(e −c ) is negligible, we have:

$$I_k = \frac{C(\lambda)(1-\lambda)}{k} + O\left(\frac{1}{k^2}\right).$$

Step 7: Final conclusion for the success rate. Recall:

$$\text{passContinuousBernoulli}(\lambda) @k = 1 - I_k.$$

For large k, this implies:

$$\text{passContinuousBernoulli}(\lambda) @k = 1 - \frac{C(\lambda)(1-\lambda)}{k} + O\left(\frac{1}{k^2}\right),$$

$$- y_1 - y_2 + O(y_2^2) \text{ for small } y, \text{ we find:}$$

Using the expansion − log(1 − <sup>y</sup>) = <sup>y</sup> <sup>+</sup> <sup>O</sup>(<sup>y</sup> 2 ) for small y, we find:

$$-\log(\text{pass}_{\text{ContinuousBernoulli}(\lambda) @k}) = C(\lambda) (1 - \lambda) k^{-1} + o(k^{-1}).$$

That is precisely polynomial (power-law) decay in the negative log success rate with exponent −<sup>1</sup>.

As a side comment, recall that tanh−<sup>1</sup> (x) = <sup>1</sup> 2 log 1+x 1−x , the normalizing constant C(λ) can be rewritten as:

$$C(\lambda) = \frac{2}{1-2\lambda} \frac{1}{2} \log \left( \frac{1+(1-2\lambda)}{1-(1-2\lambda)} \right) = \frac{1}{1-2\lambda} \log \left( \frac{1-\lambda}{\lambda} \right). \quad (38)$$

Thus, for small <sup>λ</sup>, note that <sup>C</sup>(λ) ≈ log(1/λ) = − log(λ). For <sup>k</sup> ≪ − log(λ), the <sup>1</sup>/k formula is valid. However, near <sup>k</sup> ≈ − log(λ), the leading term − log(λ)/k becomes of order 1, and for <sup>k</sup> ≫ − log(λ), the success rate is now very close to 1. Consequently, we see that if <sup>λ</sup> is very small, there is a soft cutoff scale around <sup>k</sup> ≈ − log(λ).

## E.7. Any Continuous Distribution with p(passi@1) = c > 0

Suppose that the distribution over passi@1 is continuous and has constant non-zero density near 0:

$$f(0) = c > 0 \quad (39)$$

Because the density is continuous at 0 with f(0) = c > 0, there exist some δ > 0 such that:

$$f(p) = c + O(p) \quad \text{for all } p \in [0, \delta]. \quad (40)$$

Because the small passi@1 region dominates for large k, a similar argument to the Kumaraswamy argument and Continuous Bernoulli argument yields power law scaling with respect to <sup>k</sup> with exponent −<sup>1</sup>:

$$-\log\left(\text{pass}_{\mathcal{D}} @k\right) = ck^{-1} + o(k^{-1}). \quad (41)$$

This result is consistent with the Continuous Bernoulli, where <sup>c</sup> is given by <sup>f</sup>ContinuousBernoulli(λ)(0; <sup>λ</sup>) = <sup>C</sup>(λ)(1 − <sup>λ</sup>) for λ < 1/2. This result reveals that the Continous Bernoulli is just one instance of a larger family: any continuous distribution with non-zero constant density at passi@1 = 0 will exhibit power law scaling with exponent −<sup>1</sup>.

## E.8. Reciprocal Distribution: passi@1 ∼ Reciprocal(a, b)

Next, suppose the model's passi@1 ∼ Reciprocal(a, b) distribution with <sup>0</sup> < a < b < <sup>1</sup>. The probability density function of this distribution over the support <sup>x</sup> ∈ [a, b] is:

$$f(x; a, b) = \frac{1}{(\log(b) - \log(a)) x} \quad (42)$$

As with the other distributions, the aggregate success rate after k attempts is:

$$\text{passReciprocal}(a,b) @k = \mathbb{E}[\text{pass}_1 @k] = 1 - I_k, \quad \text{where} \quad I_k \stackrel{\text{def}}{=} \int_{x=a}^b (1-x)^k \frac{1}{(\log b - \log a) x} dx.$$

We aim to show that <sup>I</sup><sup>k</sup> is on the order of (1−a) k k . The main contribution to the integral arises from the vicinity of x = a, because (1 − <sup>x</sup>) <sup>k</sup> decays rapidly as x grows away from a.

Step 1: Change of variable. Define y def <sup>=</sup> <sup>x</sup> − <sup>a</sup>, so the domain <sup>x</sup> ∈ [a, b] becomes <sup>y</sup> ∈ [0, b − <sup>a</sup>]. Then

$$(1-x)^k = ((1-a) - y)^k,$$

and

$$I_k = \frac{1}{\log(b/a)} \int_{y=0}^{b-a} ((1-a) - y)^k \frac{1}{a+y} dy.$$

Step 2: Expansion near <sup>y</sup> = 0. For small <sup>y</sup>, write (1 − <sup>a</sup>) − <sup>y</sup> = (1 − <sup>a</sup>) 1 − y 1−a ; hence

$$\log((1-a) - y) = \log(1-a) + \log\left(1 - \frac{y}{1-a}\right).$$

Using log(1 − <sup>z</sup>) = −<sup>z</sup> <sup>+</sup> <sup>O</sup>(<sup>z</sup> 2 ) for small z, we get

$$\log((1-a) - y) = \log(1-a) - \frac{y}{1-a} + O\left(\frac{y^2}{(1-a)^2}\right),$$

so

$$(1 - a - y)^k = \exp\left(k \log(1 - a) - k \frac{y}{1-a} + O\left(\frac{ky^2}{(1-a)^2}\right)\right),$$

$$k, \text{ the term } ky^2 = O(1) \text{ remains bounded, so}$$

In particular, for y up to c/k, the term k y<sup>2</sup> = O(1) remains bounded, so

$$(1 - a - y)^k = (1 - a)^k \exp\left(-\frac{ky}{1-a}\right) \left[1 + O\left(\frac{1}{k}\right)\right].$$

Step 3: The integral is dominated by <sup>y</sup> ∈ [0, O( k )]. For large <sup>k</sup>, exp − k y 1−a decays quickly once y exceeds a multiple of <sup>1</sup>−<sup>a</sup> k . Consequently, the integral from <sup>y</sup> <sup>=</sup> <sup>c</sup>0/k to <sup>b</sup> − <sup>a</sup> is exponentially small in <sup>k</sup>. On [0, c0/k], we also have (a + y) <sup>−</sup><sup>1</sup> = <sup>a</sup> + O 1 k . Thus

$$I_k = \frac{1}{\log(b/a)} \int_{y=0}^{c_0/k} (1-a-y)^k \frac{1}{a+y} dy + (\text{exponentially small tail}).$$

Substitute our approximation from Step 2 into the integrand:

$$(1 - a - y)^k \frac{1}{a + y} = (1 - a)^k \exp\left(-\frac{ky}{1-a}\right) \left[\frac{1}{a} + O\left(\frac{1}{k}\right)\right].$$

Step 4: Change variable u = k y 1−a . Then y = (1−a) u k and dy = 1−a k du. The upper limit y = c0/k corresponds to u = c<sup>0</sup> 1−a 1 , so

$$\int_{y=0}^{c_0/k} \exp\left(-\frac{ky}{1-a}\right) dy = \int_{u=0}^{c_0(1-a)} e^{-u} \frac{1-a}{k} du.$$

Letting <sup>c</sup><sup>0</sup> → ∞ only contributes an <sup>e</sup> −c0(1−a) factor to the tail, which vanishes. Hence

$$\int_{y=0}^{\infty} \exp\left(-\frac{ky}{1-a}\right) dy = \frac{1-a}{k} \int_{u=0}^{\infty} e^{-u} du = \frac{1-a}{k}.$$

Putting all factors together,

$$I_k = \frac{1}{\log(b/a)} (1-a)^k \left[ \frac{1}{a} + O\left(\frac{1}{k}\right) \right] \frac{1-a}{k} + (\text{exponentially small in } k).$$

Thus in big-Theta form,

$$I_k = \Theta\left(\frac{(1-a)^k}{k}\right).$$

Conclusion. Since passReciprocal(a,b)@k = 1 − <sup>I</sup>k, we get

$$\text{PassReciprocal}(a,b) @k = 1 - \Theta\left(\frac{(1-a)^k}{k}\right).$$

Moreover, using − log(1 − <sup>y</sup>) = <sup>y</sup> <sup>+</sup> <sup>O</sup>(<sup>y</sup> 2 ) for small y, it follows that

$$-\log\left(\text{passReciprocal}(a,b) @k\right) = \Theta\left(\frac{(1-a)^k}{k}\right).$$

Hence the negative log aggregate success rate converges to 1 *exponentially fast* in k, which is *not* a power law in k.

## Sufficient Condition for Power-Law Scaling in Negative Log of Aggregate Success

Theorem E.1. *Let* D *be a probability distribution on* [0, 1] *with PDF* <sup>f</sup>(p)*. Suppose there exist constants* b > <sup>0</sup>*,* C > <sup>0</sup>*,* θ > 0 *and* δ > 0 *such that, for all* 0 < p < δ*, we have*

$$f(p) = C p^{b-1} + O(p^{b-1+\theta}).$$

*Then, for large* k*,*

$$1 - \text{pass}_{\mathcal{D}} @k = C \Gamma(b) k^{-b} + O(k^{-b-\min(1,\theta)}),$$

*which implies*

$$-\log\left(\text{pass}_{\mathcal{D}} @k\right) = C\Gamma(b) k^{-b} + o(k^{-b}),$$
onstant),

*Equivalently, including the leading constant),*

$$-\log(\text{pass}_{\mathcal{D}} @k) \sim C \Gamma(b) k^{-b}.$$

## *Proof.* Step 1. Decompose the key integral.

Define

$$I_k \stackrel{\text{def}}{=} 1 - \text{pass}_{\mathcal{D}} @k = \int_0^1 (1-p)^k f(p) \, \text{d}p.$$

For a positive constant c > 0, split Ik:

$$I_k = \int_0^{c/k} (1-p)^k f(p) \, dp + \int_{c/k}^1 (1-p)^k f(p) \, dp \stackrel{\text{def}}{=} I_{k,\text{left}} + I_{k,\text{right}}.$$

Right Tail Bound (Ik,right). For <sup>p</sup> ≥ c/k, observe (1 − <sup>p</sup>) <sup>k</sup> ≤ <sup>e</sup> <sup>−</sup>k p ≤ <sup>e</sup> −c . Hence

$$I_{k,\text{right}} = \int_{c/k}^1 (1-p)^k f(p) dp \leq e^{-c} \int_0^1 f(p) dp = e^{-c}.$$

Since c can be made arbitrarily large, e −c can be driven below *any* power of 1/k. Thus Ik,right = o k −α for any α > 0. We may therefore focus on

$$I_{k,\text{left}} = \int_0^{c/k} (1-p)^k f(p) \, dp,$$

knowing that Ik,right is negligible in polynomial-type estimates.

## Step 2. Use the assumed behavior of f(p) near p = 0.

By hypothesis, for p up to some δ > 0,

$$f(p) = C p^{b-1} + O(p^{b-1+\theta}).$$

Choose c/k < δ, so <sup>p</sup> ≤ c/k < δ for <sup>p</sup> in the left integral. Then

$$I_{k,\text{left}} = \int_0^{c/k} (1-p)^k \left[ C p^{b-1} + O(p^{b-1+\theta}) \right] dp.$$

Split it into main term and error term:

$$I_{k,\text{left}} = C \int_0^{c/k} (1-p)^k p^{b-1} dp + \int_0^{c/k} (1-p)^k O(p^{b-1+\theta}) dp.$$

Denote these Tmain and Terr, respectively.

Step 3. Approximate (1 − <sup>p</sup>) For <sup>p</sup> in [0, c/k], expand log(1 − <sup>p</sup>) = −<sup>p</sup> <sup>+</sup> <sup>O</sup>(<sup>p</sup> 2 ). Thus

### <sup>k</sup> by e <sup>−</sup>kp and control the error.

$$(1-p)^k = \exp(k \log(1-p)) = e^{-kp} \exp(O(kp^2)) = e^{-kp} [1 + O(kp^2)].$$

Since <sup>p</sup> ≤ c/k, we get k p<sup>2</sup> ≤ <sup>c</sup> <sup>2</sup>/k, which is bounded for large k. Consequently,

$$(1-p)^k = e^{-kp} + O(kp^2 e^{-kp}).$$

We will use this in both Tmain and Terr.

Step 4. Main term Tmain.

$$T_{\text{main}} = C \int_0^{c/k} (1-p)^k p^{b-1} dp.$$

Substituting (1 − <sup>p</sup>) <sup>k</sup> = e <sup>−</sup>k p + O k p<sup>2</sup> e −k p ,

$$T_{\text{main}} = C \int_0^{c/k} e^{-k p} p^{b-1} dp + C \int_0^{c/k} O(k p^{b+1} e^{-k p}) dp.$$

Call these two integrals T<sup>1</sup> and T2.

T<sup>1</sup> term.

$$T_1 = C \int_0^{c/k} p^{b-1} e^{-kp} dp.$$

Make the substitution u def <sup>=</sup> k p. Then <sup>p</sup> <sup>=</sup> u/k, <sup>d</sup><sup>p</sup> = du/k, and <sup>p</sup> <sup>b</sup>−<sup>1</sup> = k <sup>−</sup>b+1 u b−1 . The upper limit p = c/k becomes u = c. Thus

$$T_1 = C \int_0^c \left(\frac{u}{k}\right)^{b-1} e^{-u} \frac{du}{k} = C k^{-b} \int_0^c u^{b-1} e^{-u} du.$$

As <sup>c</sup> → ∞, R c 0 u b−1 e <sup>−</sup><sup>u</sup> <sup>d</sup><sup>u</sup> → Γ(b). So

$$T_1 = C k^{-b} \left( \Gamma(b) - R_c \right)$$
, where  $|R_c| = O(e^{-c})$ , we conclude

By choosing <sup>c</sup> large after <sup>k</sup> → ∞, we conclude

$$T_1 = C \Gamma(b) k^{-b} + o(k^{-b}).$$

T<sup>2</sup> term.

$$T_2 = C \int_0^{c/k} O(k p^{b+1} e^{-k p}) dp.$$

Inside the integral, k p <sup>b</sup>+1 e <sup>−</sup>k p is the main factor. Substituting u def = k p again,

$$p^{b+1} = \left(\frac{u}{k}\right)^{b+1} = k^{-b-1} u^{b+1}.$$

Hence

$$T_2 = C O(1) \int_0^{c/k} k p^{b+1} e^{-k p} dp = O(k) \int_0^{c/k} p^{b+1} e^{-k p} dp.$$

Substitute u = k p and dp = du/k. Then

$$T_2 = O(k) \int_0^c \left(\frac{u}{k}\right)^{b+1} e^{-u} \frac{du}{k} = O(k) k^{-b-2} \int_0^c u^{b+1} e^{-u} du = O(k^{-b-1}).$$

Thus T<sup>2</sup> is of strictly smaller order than k −b .

$$T_{\text{main}} = C \Gamma(b) k^{-b} + O(k^{-b-1}).$$

## Step 5. Error term Terr.

Recall

$$T_{\text{err}} = \int_0^{c/k} (1-p)^k O(p^{b-1+\theta}) \, dp.$$

Exactly the same substitution (1 − <sup>p</sup>) <sup>k</sup> = e <sup>−</sup>kp + O(k p<sup>2</sup> e <sup>−</sup>k p) plus u = k p shows

$$T_{\text{err}} = O\left(\int_0^{c/k} p^{b-1+\theta} e^{-kp} dp\right) + O\left(\int_0^{c/k} kp^{b+1+\theta} e^{-kp} dp\right).$$

When substituting u = k p, the exponent on p increases by +1 each time if we multiply by k, so each term is of order k −b−θ or smaller. Concretely,

$$\int_0^{c/k} p^{b-1+\theta} e^{-kp} dp = k^{-b-\theta} \int_0^c u^{b-1+\theta} e^{-u} du = O(k^{-b-\theta}),$$

and similarly for the second term, which is even smaller. Hence

$$T_{\text{err}} = O(k^{-b-\theta}).$$

### Step 6. Putting it all together.

Summarize:

$$I_{k,\text{left}} = T_{\text{main}} + T_{\text{err}} = C\Gamma(b)k^{-b} + O(k^{-b-1}) + O(k^{-b-\theta}).$$

Thus

$$\begin{aligned} I_{k,\text{left}} &= C \Gamma(b) k^{-b} + O(k^{-b-\min(1,\theta)}), \\ &= o(k^{-\alpha}) \text{ for any } \alpha, \text{ we obtain} \end{aligned}$$

Recalling the tail piece Ik,right = e <sup>−</sup><sup>c</sup> = o k −α for any α, we obtain

$$I_k = I_{k,\text{left}} + I_{k,\text{right}} = C \Gamma(b) k^{-b} + O(k^{-b-\min(1,\theta)}).$$

Hence

$$1 - \text{pass}_{\mathcal{D}} @k = I_k \quad \sim \quad C \Gamma(b) k^{-b}.$$

Final negative-log argument. Since

$$\text{pass}_{\mathcal{D}} @k = 1 - I_k = 1 - (C \Gamma(b) k^{-b} + O(k^{-b - \min(1, \theta)}))$$
,
1. Then

for large k it is very close to 1. Then

$$-\log\left(\text{pass}_{\mathcal{D}} @k\right) = -\log\left(1 - C \Gamma(b) k^{-b} + \dots\right).$$

Using the expansion − log(1 − <sup>x</sup>) = <sup>x</sup> <sup>+</sup> <sup>O</sup>(<sup>x</sup> 2 ) as <sup>x</sup> → <sup>0</sup>, and here <sup>x</sup> <sup>=</sup> <sup>C</sup> Γ(b) <sup>k</sup> −b , we get

$$-\log\left(\text{pass}_{\mathcal{D}} @k\right) = C \Gamma(b) k^{-b} + o(k^{-b}).$$

In the "∼" notation including the leading coefficient:

$$-\log\left(\text{pass}_{\mathcal{D}}@k\right) \sim C\Gamma(b) k^{-b}.$$

This completes the proof.

## E.9. Necessary Condition for Power Law Scaling from Distribution over passi@1

Theorem E.2. *Let* D *be a probability distribution over* [0, 1] *with a PDF* <sup>f</sup>(p) *satisfying the following regularity near* p = 0*:*

- *No point mass at* <sup>p</sup> = 0*. So* R <sup>1</sup> 0 f(p) dp = 1*, and* f *is a genuine PDF on* (0, 1]*.*
- *Continuity and nonnegative behavior near* p = 0*. There exist* δ > 0 *such that* f *is continuous on* [0, δ] *and has no pathological oscillations or singularities that violate integrability.*

*Define the aggregate success rate at* k *attempts:*

$$\text{pass}_{\mathcal{D}} @k \stackrel{\text{def}}{=} \int_0^1 \left[ 1 - (1-p)^k \right] f(p) \, dp$$

*and relatedly*

$$I_k \stackrel{\text{def}}{=} \int_0^1 (1-p)^k f(p) \, dp = 1 - \text{pass}_D @k.$$

*Assume that there exist constants* A > 0 *and* b > 0 *such that for large* k*:*

$$-\log(\text{pass}_{\mathcal{D}} @k) \sim A k^{-b}$$

*Then*

$$I_k = A k^{-b} + o(k^{-b}),$$

*and under the mild regularity assumptions above,*

$$f(p) \sim \frac{A}{\Gamma(b)} p^{b-1} \quad \text{as } p \rightarrow 0^+.$$

*Proof.* Step 1. Relating <sup>I</sup><sup>k</sup> to − log(passD@k).

By definition,

$$\text{pass}_{\mathcal{D}} @k = 1 - I_k, \quad I_k = \int_0^1 (1-p)^k f(p) \, dp,$$

$$-\log\left(\text{pass}_{\mathcal{D}} @k\right) \sim A k^{-b},$$

Since

we have, for large k,

$$\text{pass}_D @k = \exp(-A k^{-b} (1 + o(1)))$$
. Thus

When <sup>x</sup> is small, exp(−x) = 1 − <sup>x</sup> <sup>+</sup> <sup>O</sup>(<sup>x</sup> 2 ). Thus

$$I_k = 1 - \text{pass}_{\mathcal{D}} @k = A k^{-b} + o(k^{-b}).$$

So

$$I_k \sim A k^{-b}.$$

## Step 2. Restricting to a small interval near p = 0.

Since (1 − <sup>p</sup>) <sup>k</sup> decays exponentially once p is on the order of 1/k or larger, we split:

$$I_k \stackrel{\text{def}}{=} \int_0^1 (1-p)^k f(p) \, dp = \int_0^{c/k} (1-p)^k f(p) \, dp + \int_{c/k}^1 (1-p)^k f(p) \, dp \stackrel{\text{def}}{=} I_{k,\text{left}} + I_{k,\text{right}},$$

for some positive constant <sup>c</sup>. In the region <sup>p</sup> ≥ c/k, we have (1 − <sup>p</sup>) <sup>k</sup> ≤ <sup>e</sup> <sup>−</sup>k p ≤ <sup>e</sup> −c , so

$$I_{k,\text{right}} \leq e^{-c} \int_0^1 f(p) \, dp = e^{-c}.$$

Since c > 0 can be made large, e −c can be driven below any fixed power of 1/k. Hence for the Θ(k −b ) behavior, the main contribution comes from [0, c/k].

Thus

$$I_k = I_{k,\text{left}} + o(k^{-m}) \text{ for every } m > 0$$

Step 3. Change of variables and controlling the ratio of (1 − <sup>p</sup>) k to e −kp .

(a) Ratio to e −kp . For <sup>p</sup> ∈ -0, c k , define the ratio

$$R_k(p) \stackrel{\text{def}}{=} \frac{(1-p)^k}{e^{-k} p}.$$

We will show that <sup>R</sup>k(p) stays close to <sup>1</sup> uniformly in <sup>p</sup> ∈ [0, c/k] for large <sup>k</sup>. Indeed,

$$(1-p)^k = \exp\left[k \log(1-p)\right], \quad \log(1-p) = -p - \frac{p^2}{2} - \frac{p^3}{3} - \dots$$

Hence

$$\log(1-p) + p = -\frac{p^2}{2} - \frac{p^3}{3} - \dots = O(p^2) \quad \text{as } p \rightarrow 0.$$

Multiplying by k, we get

$$k [\log(1-p) + p] = O(k p^2),$$

$$\rightarrow 0 \text{ as } k \rightarrow \infty, \text{ it follows that}$$

Since <sup>0</sup> ≤ <sup>p</sup> ≤ c k implies k p<sup>2</sup> ≤ c k , which → <sup>0</sup> as <sup>k</sup> → ∞, it follows that

$$k \log(1-p) = -kp + O\left(\frac{1}{k}\right).$$

Exponentiating:

$$(1-p)^k = e^{-kp} \exp\left(O\left(\frac{1}{k}\right)\right) = e^{-kp} \left[1 + O\left(\frac{1}{k}\right)\right].$$

Thus

$$R_k(p) = \frac{(1-p)^k}{e^{-k}p} = 1 + O\left(\frac{1}{k}\right),$$

with the O( 1 k ) bound uniform for all <sup>p</sup> ∈ [0, c/k]. In other words, there is some constant M > <sup>0</sup> (independent of <sup>k</sup>) such that

$$|R_k(p) - 1| \leq \frac{M}{k} \quad \text{for all } p \in \left[0, \frac{c}{k}\right].$$

(b) Integral expression using Rk(p). Hence on [0, c/k],

$$(1-p)^k f(p) = e^{-kp} R_k(p) f(p).$$

Thus

$$I_{k, \text{left}} = \int_0^{c/k} e^{-kp} f(p) R_k(p) dp.$$

Define ∆k(p) def <sup>=</sup> <sup>R</sup>k(p) − <sup>1</sup>, which satisfies |<sup>∆</sup>k(p)| ≤ M/k. Then

$$I_{k,\text{left}} = \int_0^{c/k} e^{-kp} f(p) dp + \int_0^{c/k} e^{-kp} f(p) \Delta_k(p) dp. \quad (43)$$

Step 4. Substitution <sup>u</sup> <sup>=</sup> k p and deriving <sup>f</sup>(p) ∼ <sup>p</sup> b−1 .

(a) The leading part. Focus on the first term of equation [43:](#page-39-0)

$$\int_0^{c/k} e^{-kp} f(p) dp.$$

Substitute u def = k p, so p = u k and dp = 1 k du. The upper limit p = c k becomes u = c. Thus

$$\int_0^{c/k} e^{-kp} f(p) dp = \int_0^c e^{-u} f\left(\frac{u}{k}\right) \frac{du}{k}.$$

Hence

$$\int_0^{c/k} e^{-kp} f(p) dp = \frac{1}{k} \int_0^c e^{-u} f\left(\frac{u}{k}\right) du.$$

(b) The error part. The second term in equation [<sup>43</sup>](#page-39-0) has <sup>∆</sup>k(p) = <sup>R</sup>k(p) − <sup>1</sup> satisfying |<sup>∆</sup>k(p)| ≤ <sup>M</sup> k . So

$$\left| \int_0^{c/k} e^{-kp} f(p) \Delta_k(p) dp \right| \leq \frac{M}{k} \int_0^{c/k} e^{-kp} f(p) dp.$$

  But the integral R c/k 0 e <sup>−</sup>k p f(p) dp is precisely the leading part we just considered. Thus the error is bounded by <sup>M</sup> k times a term that will turn out to be Θ(k −b ). Hence the error is subleading if b < 1 is not the case—but even then, we can keep track of it systematically.

Overall, combining both terms, we get

$$I_{k,\text{left}} = \frac{1}{k} \int_0^c e^{-u} f\left(\frac{u}{k}\right) du + O\left(\frac{1}{k} \cdot (\text{leading integral})\right). \quad (44)$$

(c) Matching Θ(k −b ). Since I<sup>k</sup> = Ik,left + Ik,right with Ik,right negligible, we have

$$I_k = \frac{1}{k} \int_0^c e^{-u} f\left(\frac{u}{k}\right) du + (\text{small corrections}).$$

But by hypothesis, <sup>I</sup><sup>k</sup> ∼ α k−<sup>b</sup> . Thus

$$k \cdot I_k = \int_0^c e^{-u} f\left(\frac{u}{k}\right) du + (\text{smaller terms}) \sim \alpha k^{1-b}. \quad (45)$$

Hence the expression

$$\int_0^c e^{-u} f\left(\frac{u}{k}\right) du$$

must be Θ k 1−b for large k. Since <sup>u</sup> k is small for <sup>0</sup> ≤ <sup>u</sup> ≤ <sup>c</sup>, we are effectively sampling <sup>f</sup> near 0. For the integral to produce k 1−b , we deduce

$$f\left(\frac{u}{k}\right) = \Theta\left(\left(\frac{u}{k}\right)^{b-1}\right),$$

i.e. f must behave like p <sup>b</sup>−<sup>1</sup> near p = 0. Rewriting the constant in front, one obtains

$$f\left(\frac{u}{k}\right) = \left(\frac{u}{k}\right)^{b-1} [\text{some positive constant}].$$

Step 5. Conclusion. We have thus shown that over <sup>p</sup> ∈ [0, c/k], one has

$$(1-p)^k = e^{-kp} \left[ 1 + O\left(\frac{1}{k}\right) \right],$$

and upon integrating, the required k −b form for I<sup>k</sup> forces

$$f(p) = \frac{A}{\Gamma(b)} p^{b-1} + o(p^{b-1}), \quad \text{as } p \rightarrow 0^+.$$

This completes the necessity proof.

Remark (Mild Regularity). If <sup>f</sup> had bizarre oscillations or nonintegrable singularities near <sup>0</sup>, the integral R <sup>1</sup> (1−p) <sup>k</sup> f(p) dp might not produce a clean k −b . Typically, we impose monotonicity or at least continuity near p = 0, no atom at p = 0, and f(0) = 0 if b > 1 or f(0) > 0 if b = 1, etc. These assumptions exclude pathological behaviors and guarantee that the local shape of f(p) drives a clean power law.

## F. Maximum Likelihood Estimation of Scaled Beta-Binomial Distribution

To model the distribution of passi@1, we can perform maximum likelihood estimation on a scaled three-parameter Beta-Binomial distribution, which we chose because each attempt on the i-th problem is an i.i.d. Bernoulli random variable with success probability passi@1, and we introduced a scale parameter because the largest passi@1 values were typically 1-2 orders of magnitude less than 1.0 (the maximum of the unscaled beta distribution's support).

In greater detail, as background, the 4-parameter Beta distribution has PDF

$$p_Y(y; \alpha, \beta, a, c) \stackrel{\text{def}}{=} \frac{(y-a)^{\alpha-1}(c-y)^{\beta-1}}{(c-a)^{\alpha+\beta-1} B(\alpha, \beta)}, \quad (46)$$

where B(·, ·) is the [Beta function.](https://en.wikipedia.org/wiki/Beta_function) If the minimum <sup>a</sup> is fixed at <sup>0</sup> and the maximum <sup>c</sup> is constrained to a < c < <sup>1</sup>, then the scaled three parameter Beta distribution simplifies to:

$$f_P(p; \alpha, \beta, a = 0, c) = \frac{p^{\alpha-1}(c-p)^{\beta-1}}{c^{\alpha+\beta-1} \text{B}(\alpha, \beta)}. \quad (47)$$

We want the PMF of a three-parameter Beta-Binomial distribution based on this scaled Beta distribution. For n samples and x successes, the PMF is:

$$P(X = x; \alpha, \beta, c, n) \stackrel{\text{def}}{=} \int_0^c \binom{n}{x} p^x (1-p)^{n-x} f_P(p; \alpha, \beta, a = 0, c) dp \quad (48)$$

$$= \binom{n}{x} \frac{1}{c^{\alpha+\beta-1} B(\alpha, \beta)} \int_0^c p^{x+\alpha-1} (1-p)^{n-x} (c-p)^{\beta-1} dp. \quad (49)$$

Using a change of variable p def = c z, the PMF can be rewritten as

$$P(X = x; \alpha, \beta, c, n) = \binom{n}{x} \frac{c^x}{B(\alpha, \beta)} \int_0^1 z^{x+\alpha-1} (1-z)^{\beta-1} (1-cz)^{n-x} dz \quad (50)$$

$$= \binom{n}{x} \frac{c^x \mathbf{B}(x + \alpha, \beta)}{\mathbf{B}(\alpha, \beta)} {}_2F_1\left(-(n-x), x + \alpha; x + \alpha + \beta; c\right), \quad (51)$$

where <sup>2</sup>F1(·, ·; ·; ·) is the [\(Gauss\) hypergeometric function.](https://en.wikipedia.org/wiki/Hypergeometric_function#Euler_type)

## G. Maximum Likelihood Estimation of Scaled Kumaraswamy-Binomial Distribution

To model the distribution of passi@1, we can perform maximum likelihood estimation on a scaled three-parameter Kumaraswamy-Binomial distribution, which we chose because each attempt on the i-th problem is an i.i.d. Kumaraswamy random variable with success probability passi@1, and we introduced a scale parameter because the largest passi@1 values were typically 1-2 orders of magnitude less than 1.0 (the maximum of the unscaled beta distribution's support).

In greater detail, the scaled three parameter Kumaraswamy distribution simplifies to:

$$f_P(p; \alpha, \beta, a = 0, c) = \frac{\alpha\beta}{c^\alpha} p^{\alpha-1} (1 - (p/c)^\alpha)^{\beta-1}, \quad (52)$$

over the support (0, c). The rescaled Kumaraswamy-Binomial distribution then has PMF:

$$P(X = x; \alpha, \beta, c, n) = \binom{n}{x} \frac{\alpha \beta}{c^\alpha} \int_0^c p^{x+\alpha-1} (1-p)^{n-x} \left(1 - \binom{p}{c}^\alpha\right)^{\beta-1} dp. \quad (53)$$

One can perform a change of variable p def = cz, but simplifying yields sums of hypergeometric functions that add little conceptual clarity and so we resort to numerical integration using Python's [mpmath library](https://mpmath.org/) [\(mpmath development team,](#page-13-19) [2023\)](#page-13-19).

## H. Maximum Likelihood Estimation of Scaled Beta-Negative Binomial Distribution

To model the distribution of passi@1, we can perform maximum likelihood estimation on a scaled three-parameter Beta-Negative Binomial distribution. Recall that the scaled three parameter Beta distribution is:

$$f_P(p; \alpha, \beta, a = 0, c) = \frac{p^{\alpha-1}(c-p)^{\beta-1}}{c^{\alpha+\beta-1} \text{B}(\alpha, \beta)}. \quad (54)$$

We want the PMF of a three-parameter Beta-Negative Binomial distribution based on this scaled Beta distribution. For r desired successes, the PMF that we first draw x failures is:

$$P(X = x; \alpha, \beta, c, r) = \int_0^c \underbrace{\left( \frac{x+r-1}{x} \right)}_{\beta} p^r (1-p)^x \underbrace{\frac{p^{\alpha-1} (c-p)^{\beta-1}}{c^{\alpha+\beta-1} B(\alpha, \beta)}}_{B(\alpha, \beta)} dp \quad (55)$$

$$= \binom{x+r-1}{x} \frac{1}{c^{\alpha+\beta-1} B(\alpha, \beta)} \int_0^c p^{r+\alpha-1} (1-p)^x (c-p)^{\beta-1} dp. \quad (56)$$

Next, substitute <sup>p</sup> <sup>=</sup> c z <sup>=</sup>⇒ dp <sup>=</sup> c dz which rescales the domain [0, c] to [0, 1]. Under this change:

$$\begin{aligned} p^{r+\alpha-1} &= (cz)^{r+\alpha-1} = c^{r+\alpha-1} z^{r+\alpha-1}, \\ (c-p)^{\beta-1} &= (c-cz)^{\beta-1} = (c(1-z))^{\beta-1} = c^{\beta-1} (1-z)^{\beta-1}, \\ (1-p)^x &= (1-cz)^x. \end{aligned}$$

Putting these into the integrand:

$$p^{r+\alpha-1} (1-p)^x (c-p)^{\beta-1} dp = \left( c^{r+\alpha-1} z^{r+\alpha-1} \right) \left( (1-cz)^x \right) \left( c^{\beta-1} (1-z)^{\beta-1} \right) (c dz).$$

Factor out the constants in c:

$$= c^{r+\alpha-1} c^{\beta-1} c^{-z^{r+\alpha-1}} (1-cz)^x (1-z)^{\beta-1} dz.$$

Since c r+α−1 · c β−1 · <sup>c</sup> <sup>=</sup> <sup>c</sup> r+α+β−1 , we get

$$p^{r+\alpha-1} (1-p)^x (c-p)^{\beta-1} dp = c^{r+\alpha+\beta-1} z^{r+\alpha-1} (1-z)^{\beta-1} (1-cz)^x dz.$$

Plugging back into P(X = x; α, β, c, r) and simplifying:

$$P(X = x; \alpha, \beta, c, r) = \binom{x+r-1}{x} \frac{c^r}{B(\alpha, \beta)} \int_0^1 z^{r+\alpha-1} (1-z)^{\beta-1} (1-cz)^x dz. \quad (57)$$

We can re-express this using the [\(Gauss\) hypergeometric function](https://en.wikipedia.org/wiki/Hypergeometric_function#Euler_type) <sup>2</sup>F1(·, ·; ·; ·):

$$P(X = x; \alpha, \beta, c, r) = \binom{x + r - 1}{x} \frac{c^r \mathbf{B}(r + \alpha, \beta)}{\mathbf{B}(\alpha, \beta)} {}_2F_1\left(-x, r + \alpha; r + \alpha + \beta; c\right). \quad (58)$$