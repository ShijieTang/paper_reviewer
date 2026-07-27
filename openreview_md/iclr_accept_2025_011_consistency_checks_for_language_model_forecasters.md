# Consistency Checks For Language Model Forecasters

| Daniel Paleka ∗     | Abhimanyu Pallavi Sudhir *   | Alejandro Alvarez   |
|---------------------|------------------------------|---------------------|
| ETH Zurich          | University of Warwick        | Independent         |
| Adam Shen           | Evan Wang                    | Florian Tramer`     |
| Columbia University | Cornell University           | ETH Zurich          |

## Abstract

Forecasting is a task that is difficult to evaluate: the ground truth can only be known in the future. Recent work showing LLM forecasters rapidly approaching human-level performance begs the question: how can we benchmark and evaluate these forecasters *instantaneously*? Following the consistency check framework, we measure the performance of forecasters in terms of the consistency of their predictions on different logically-related questions. We propose a new, general consistency metric based on *arbitrage*: for example, if a forecasting AI illogically predicts that both the Democratic and Republican parties have 60% probability of winning the 2024 US presidential election, an arbitrageur can trade against the forecaster's predictions and make a profit. We build an automated evaluation system that generates a set of base questions, instantiates consistency checks from these questions, elicits the predictions of the forecaster, and measures the consistency of the predictions. We then build a standard, proper-scoring-rule forecasting benchmark, and show that our (instantaneous) consistency metrics correlate with LLM forecasters' ground truth Brier scores (which are only known in the future). We also release a consistency benchmark that resolves in 2028, providing a longterm evaluation tool for forecasting.

## 1 Introduction

Prediction markets are markets that pay out contingent on an event. For a market such as "$1 if Jeb Bush is elected President in 2028", the price reflects the "market estimate" for the probability of that event. Prediction markets are a promising tool for aggregating information from disparate sources to arrive at the most correct possible belief after taking into account all relevant information (Arrow et al., 2008; Hanson, 2002). Until 2024, LLM forecasters generally performed poorly relative to human forecasters (Zou et al., 2022b; Schoenegger and Park, 2023). However, recent works (Halawi et al., 2024; Schoenegger et al., 2024; Phan et al., 2024) suggest that LLM-based forecasters can rival human forecasts on forecasting websites such as Metaculus, PredictIt, and Manifold Markets. A key question emerges: once LLM forecasters are better than human ones, how can we efficiently evaluate their predictions? In particular, long-term forecasting questions are very important for decision-making (Tetlock et al., 2024; Muehlhauser, Luke, 2019), and finding ground truth for evaluation in such contexts is infeasible by virtue of the questions resolving far in the future. One approach, proposed by Fluri et al. (2023), is that even when we cannot evaluate the correctness of LLM decisions, we can evaluate their *logical consistency*. For example, if an LLM forecaster gives probabilities 0.5 and 0.7 to "Will Trump be elected US president?" and "Will someone other than Trump be elected US president?", this is necessarily inconsistent. Fluri et al. (2023) demonstrated that GPT-4 and GPT-3.5-turbo, when asked one-sentence forecasting questions, were inconsistent on simple logical checks such as negation.

∗Equal contribution. Corresponding: daniel.paleka@inf.ethz.ch. Author contributions in Contributions.

1 Our contributions in this work are as follows: 1) Principled metrics for consistency. In Section 2, we introduce a theoretical framework for measuring consistency violations of binary forecasts, based on two metrics: an *arbitrage metric*, based on market arbitrage, and a *frequentist metric*, based on hypothesis testing. We apply these metrics to 10 different logical consistency rules (see Table 3): NEGATION, PARAPHRASE, CONSEQUENCE, ANDOR, AND, OR, BUT, COND, CONDCOND and EXPEVIDENCE. 2) A consistency evaluation pipeline for binary forecasters. In Section 3, we introduce a consistency evaluation pipeline for LLM forecasters. We create two forecasting datasets with known ground truth resolutions: one scraped from prediction markets, and one synthetically generated from news articles. Both datasets include only events that happen past the training data cutoff of all forecasters we test, and resolve before September 2024. We then generate tuples of forecasting questions satisfying logical consistency rules with associated consistency metrics. 3) Consistency correlates with ground truth forecasting performance. Our consistency metrics are novel performance metrics for forecasters that can be computed right away, no matter the time horizon. Of course, forecasters could also be evaluated using *backtesting*, asking past questions with known ground truth resolutions. Yet, backtesting LLM forecasters can be challenging if we do not have clear information about the models' training data contents. Moreover, there may be new types of questions that we want to evaluate forecasters on, for which we do not have appropriate past results (e.g., questions related to pandemics before 2020). It is thus natural to ask: can consistency metrics tell us anything about future forecasting performance? In Section 4, we show that for all forecasters we test, our consistency metrics correlate positively with forecasting performance (as measured by the Brier score) on both our benchmark datasets. The correlation varies across consistency checks, with some logical checks (e.g., consistency of conditional probabilities) having over R = 0.9 correlation with forecasting performance, while other logical tests provide little signal. We hypothesise that this analysis can extend to smarter forecasters and longer time horizons, to provide instantaneous feedback on forecaster performance. 4) Scaling inference-time compute can improve consistency for some logical checks, but fails to generalize. Since we find that consistency correlates with forecasting performance, it is natural to ask whether we can improve forecasters by making them more consistent. Unfortunately, we find that natural ways of improving consistency tend to overfit to specific consistency checks and do not generalize.

Specifically, we design ArbitrageForecaster: a forecaster that "patches" some base forecaster's output by generating logically related questions and "arbitraging" the base forecaster's forecasts for these related questions against each other. In Section 5 and Appendix G, we show that ArbitrageForecaster improves consistency on checks that we optimize against, but this improvement does not generalize to other held-out consistency checks, nor does it improve the actual forecasting performance. 5) A long-horizon forecasting consistency benchmark. We create a long-horizon benchmark of 3,000 consistency checks for forecasts resolving in 2028. Our benchmark spans questions on various topics for which we will have no ground truth for more than three years, and thus serves as a nice testing ground for advanced LLM forecasters.

We release the full code 1and the datasets 2 used in the paper.

## 2 A Theoretical Framework For Forecasting Consistency

Notation. Let Prop denote the set of forecasting questions we are interested in, Θ denote the set of possible outcomes/resolutions for an individual questions. In this paper, we focus on Prop as a set of binary forecasting questions, so Θ = {⊤, ⊥}. A *Forecaster* is then a map F : Prop → [0, 1].

One special forecaster is the ground truth resolutions θ : Prop → Θ, returning 1 and 0 probability for {⊤, ⊥}, respectively. For conditional questions that can resolve to None, we also have optional resolutions Θ′:= Θ ∪
{*None*} = {⊤, ⊥, None}. We focus on binary questions following Halawi et al. (2024). Our methods can in principle be extended to study consistency between general probability distributions in forecasting, such as the ones discussed in Gooen (2024).

## 2.1 Consistency Checks And Inconsistency Metrics

In line with Fluri et al. (2023), a consistency check is conceptualized as a pair of n-ary relations:
R : Propn *→ {⊤*, ⊥} in question space, S : [0, 1]n *→ {⊤*, ⊥} in forecast space, and a predicate for F such that R(x1*, . . . x*n) =⇒ S(F(x1)*, . . .* F(xn)). In particular, this assertion must be satisfied by all feasible θ, and also any "correct" forecasts generated by a world model that accurately accounts for aleatoric uncertainty. Violation of consistency is measured by some violation metric V : [0, 1]n → R which must satisfy V(F(x1)*, . . .* F(xn)) = 0 ⇐⇒ S(F(x1)*, . . .* F(xn)). For example, intuitively, the "negation" check NEGATION is given by the relation R(x1, x2) := x1 =
¬x2 on questions, and the relation S(F(x1), F(x2)) := F(x1) + F(x2) ≈ 1 on forecasts. The full table of the consistency checks we use is given in Appendix B. Improving upon Fluri et al. (2023), we derive V from R in a principled way, handling all types of logical consistency checks simultaneously. We introduce two new *inconsistency metrics*: the arbitrage metric and the *frequentist metric* for measuring logical inconsistency in probabilistic forecasts.

## 2.1.1 Arbitrage Metric

The arbitrage metric is conceptualized as the minimum profit that an arbitrageur can be guaranteed making bets against the forecaster's predictions. More precisely: suppose that the forecaster's probabilities F(x1)*, . . .* F(xn) were prices offered by a logarithmic market maker 3 with market subsidy parameter $1. If these probabilities are inconsistent, then there are prices p1*, . . . p*n that an arbitrageur could bring to the market such that it is guaranteed to make a profit against the marketmaker, no matter the outcome of each question. We define V(F(x1)*, . . .* F(xn)) as the maximum achievable "minimum profit" that the arbitrageur can guarantee by choosing appropriate p1*, . . . p*n.

We further denote by A(F(x1)*, . . .* F(xn)) the set of prices p1*, . . . p*n that maximize the minimum profit:

$$\left(\arg\max_{p\in[0,1]^{n}}\max_{\omega\in\Omega}\min_{i=1}^{n}\left(\log p_{i}-\log\mathbb{F}(x_{i})\right)\delta_{\omega(i)=\top}+\left(\log\left(1-p_{i}\right)-\log\left(1-\mathbb{F}(x_{i})\right)\right)\delta_{\omega(i)=\bot}\right)\tag{1}$$  where $\delta_{\omega(i)=\top}$ is the $\left(\log p_{i}-\log\mathbb{F}(x_{i})\right)\delta_{\omega(i)=\top}+\left(\log\left(1-p_{i}\right)-\log\left(1-\mathbb{F}(x_{i})\right)\right)\delta_{\omega(i)=\bot}$.  
Here Ω := {ω ∈ Θ′n | R(ω)} is the set of all possible consistent resolutions of this tuple. A
more general version of 1 is given in Appendix D, along with specific worked-out examples of the arbitrage metric for each consistency check, and details on how we compute it; as an example, the arbitrage metric for the Negation Check can be derived exactly (Appendix D.2):

$${\mathcal{V}}(\mathbb{F}(x),\mathbb{F}(\neg x))=-2\log\left({\sqrt{\mathbb{F}(x)(1-\mathbb{F}(\neg x))}}+{\sqrt{(1-\mathbb{F}(x))\mathbb{F}(\neg x)}}\right)$$

To illustrate: V(0.5, 0.6) ≈ 0.01, V(0.5, 0.51) ≈ 10−4. The metric is more sensitive to violations for probabilities very close to 0 or 1, due to the logarithmic market maker. In our evals, for all types of checks, we say that a sampled check does not pass if V ≥ 0.01. We have to pick some hyperparameter as an inconsistency threshold; we set it to correspond to giving 110% probability in total to the events of Republican and Democratic parties winning the US presidential election.

## 2.1.2 Frequentist Metric

We also compute a different, *frequentist* consistency metric. Consider a Monte Carlo forecaster that samples a world model n times, and for any event, returns the fraction of samples in which the event occurs. The frequentist metric is the number of standard deviations a given tuple forecast is off from the mean Monte Carlo forecast, scaled to be independent of n. We say that a consistency violation happened if the number of standard deviations away from the mean of the null is at least as in the (0.5, 0.6) case described in Section 2.1.1. The full description is given in Appendix E.

## 2.1.3 Intuition On Consistency Metrics

Our metrics address two major obstacles with measuring inconsistency: *tolerance to noise* and principled aggregation of inconsistency scores. Tolerance to noise. In the standard Bayesian setting, beliefs are either consistent or not: there either is a Dutch book (a way to bet against the forecaster's beliefs to get infinite profit) or the probabilities are perfectly consistent. In practice, forecasters' beliefs (even on the same question) are never perfectly consistent across runs. If an election model has a presidential candidate at 48% with one random seed and 50% on the other, this is not a reason to discard it as completely flawed. Hence, instead of being a binary measure of consistency, our metrics increase smoothly with inconsistency. Principled aggregation and comparison of inconsistency scores Fluri et al. (2023) developed a set of inconsistency checks, used an ad hoc metric for each check they used, and normalized the scores to [0, 1]. There are two important issues with their approach:
1. The metrics in their work are mostly *linear* and would treat the inconsistencies of (0.5, 0.6)
and (0.89, 0.01) on the NEGATION check as equally bad, which is counterintuitive in many applications.

2. It is unclear how to compare and aggregate scores from different consistency checks.

Our approach ensures that all consistency scores share a common "unit". For example, in the arbitrage metric, to aggregate inconsistencies, we sum up the profit made by an arbitrageur across questions.

## 3 Pipeline Overview

We illustrate the steps in our data collection pipeline below, and provide more details on each individual steps:

Online platforms, news, topicssynthetic
−−−−−→
+scraping
Ptuple
$$\mathrm{\stackrel{\mathrm{tuple}}{\longrightarrow}}\ (P,Q)\ \stackrel{\mathbb{P}}{\rightarrow}\ (p,q)\ \stackrel{\mathcal{V}}{\rightarrow}\ \mathcal{V}(p,q)$$
$\mathbf{r}=\mathbf{r}$. 
instantiation
- (*· · · −→* P) We first prepare datasets of **base questions** in multiple ways:
(a) Scraping questions from online platforms such as Manifold and Metaculus;
(b) A ground-truth resolved dataset synthetically generated from news articles;
(c) Synthetic generation on questions on a list of topics such as Politics, Science, Economics, etc.

For the first two of the above, we also include the *ground truth resolution* for each question. We discuss all of these in more detail in Section 3.1.

- (P −→ (*P, Q*)) The base questions are synthetically **instantiated into tuples** that must satisfy certain consistency checks. For example, every single base question P is instantiated into a tuple (P, ¬P); and pairs of mutually relevant base questions *P, Q* are instantiated into tuples like (P, Q, P ∧ *Q, P* ∨ Q).

- ((*P, Q*)
F
−→ (*p, q*)) The forecaster is separately queried to elicit **forecasts** on each question, resulting in forecast tuples that should, if the forecaster is consistent, satisfy consistency properties. For example, for a size-two tuple where Q = ¬P, it should satisfy p + q = 1.

- ((p, q)
V−→ V(*p, q*)) We score each tuple of forecasts for consistency with both of our violation metrics.

Examples of data at each step of the pipeline are given in Appendix C. The prompts and LLM calls used in each step before forecasting are given in Appendix H.

## 3.1 Generating And Scraping Forecasting Questions

Forecasting question format. Each forecasting question includes a title that states the main question, a body that provides detailed resolution criteria, and a resolution date, along with optional fields such as metadata and creation date.

Real prediction market questions. We scrape questions from two forecasting platforms, Metaculus and Manifold Markets, and only use questions that both resolved and were initially set to resolve between May 1, 2024, and August 15, 2024. This leaves us with over 500 questions, of which 242 pass our verification step (see end of this subsection). An example of a processed question, including its relevant details, is provided in Appendix C.1. Generating forecasting questions from NewsAPI articles. To generate forecasting questions with known resolutions, we use articles sourced from NewsAPI. We focus on articles describing concrete events rather than opinion pieces. To mitigate biases towards positive resolutions (as most questions derived from an article would typically resolve to True), we employ reference class spanning - using an LLM to modify key entities in the questions while keeping the overall thematic structure intact. Each question's ground-truth resolution is verified using the Perplexity API with internet access, yielding ground truth resolution labels with less than a 5% error rate in our testing. We compile a total of 2,621 ground-truth resolved forecasting questions resolving between July 1, 2024, and August 21, 2024. Of these, we use a subset of 1,000 to test the relationship between consistency violation and accuracy. Further details regarding the pipeline can be found in Appendix K. Synthetic question generation. We generate questions by few-shot prompting, we sample six examples of forecasting questions, as style examples, as well as a set of tags (Brazil, NBA...) to diversify the generated questions. We generate question titles, deduplicate them using text-embedding-3-small embeddings from OpenAI, and then for each title we use gpt-4o to create the question body and resolution date. With this method we create 1,000 forecasting questions that resolve either by or in 2028. More details are in Appendix H. Verification and improvement from human feedback. In all of the above steps, we filter generated questions in using gpt-4o to check for properties such as the coherence between the body and title, the clarity and precision of the resolution criteria, and whether the question is about actual world events. Questions failing this step are discarded. To develop this step, we used a feedback form for human reviewers (authors of this paper) to suggest modifications to generated questions.

These suggestions inform refinements to prompts and few-shot examples in our pipeline. An example of the feedback form is provided in Appendix I.

## 3.2 Instantiating Tuples Of Questions For Consistency Checks

The base forecasting questions are subsequently used to synthetically generate tuples of logically related questions. For example, a pair of base questions (P, Q) can be used to generate a 4-tuple (P, Q, P ∧*Q, P* ∨Q) for ANDOR, or a 3-tuple (P, ¬P ∧*Q, P* ∨Q) for BUT (see Appendix B for details). The main question content (titles and bodies) were generated synthetically (using gpt-4o), while the resolution dates and other properties were calculated systematically (e.g. the max of the resolution dates of the base questions). We then conduct two measures to ensure the instantiated tuples are correct and sensible: relevance scoring, and verification that the tuples of questions indeed describe logically related events. Relevance scoring. When combining base questions into tuples, we have to take care to avoid off-distribution questions like "Is SpaceX going to be worth $200B by 2030, given that Sri Lanka's rice production grows 40% by 2040?". For tuples instantiated from more than one base question, we sort 2000 potential base question combinations by their "relevance score", obtained by querying an LLM and asking it to score how relevant the questions are to one another, and choose the top 200 for each consistency check. See Figure 15 for details. Verification. The instantiated tuples of questions are then passed to another LLM call to reject if they do not fit their intended structure; for example, we detect if the resolution criteria of the second question are not truly a negation of the resolution criteria of the first question. Examples of verification prompts are given in Appendix H.

## 3.3 Eliciting Forecasts

We test a range of forecasters based on various LLM models (gpt-4o, gpt-4o-mini, claude-3.5-sonnet, llama-3.1-8B, llama-3.1-70B, llama-3.1-405B, o1-mini and o1-preview) with and without chain-of-thought prompting: see Appendix F for details. We run each of these forecasters on 5000 tuples in total (for each of the 10 checks, we use 200 tuples from scraped questions and 300 from NewsAPI questions), except for o1-preview, which we test on 50 tuples per check only due to cost constraints. We could not test forecasters from Halawi et al. (2024) due to API deprecations; see Section 6.

## 4 Results

We evaluate a range of forecasters on the datasets described above, for both consistency and ground truth Brier score. We note that the Brier score as the standard metric of forecasting accuracy depends both on model capabilities and the training data cutoff: it should not be surprising for a stronger model to have a worse Brier score if its training data cutoff is earlier than for a weaker model. The full list of forecasters is in Appendix F. For all data analysis in this section, we exclude forecasters that have Brier score worse than random guessing (0.25), such as the basic setup with llama-3.1-8B, as it would unfairly advantage our case of "correlating consistency with accuracy".

Average consistency scores correlate strongly with forecasting performance. We can aggregate the consistency scores across all checks for each forecaster by aggregating either the arbitrage or the frequentist violations. We plot the average Brier score against the three aggregate consistency scores in Figure 1.

Correlation: 0.85 aggregated.frequentist.avg_violation vs avg_brier_score (scraped)
Correlation: 0.49 aggregated.arbitrage.avg_violation vs avg_brier_score (newsapi)
CoT-L3-8B
CoT-o1-mini CoT-L3-8B
0.17 0.18 0.19 0.20 0.21 0.22 0.23 avg_brier_score 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.14 0.16 0.18 0.20 0.22 0.24 avg_brier_score 0.030 0.035 0.040 0.045 0.050 0.055 CoT-GPT-4o-mini CoT-GPT-4o-mini CoT-o1-mini avg_viola tion avg_viol ation CoT-L3-70B
Basic-GPT-4o-mini CoT-L3-70B
CoT-L3-405B Basic-L3-70B
Basic-L3-405B
CoT-Sonnet CoT-L3-405B

CoT-o1-preview Basic-GPT-4o-08 Basic-GPT-4o-05 GPT-4o-08 CoT-o1-preview Basic-Sonnet GPT-4o-05 GPT-4o-mini Sonnet CoT-GPT-4o-08 CoT-GPT-4o-08 CoT-Sonnet

Figure 1: Scatter plots showing the relationship between consistency metrics and average Brier scores for different forecasters. Each point represents a forecaster, with the x-axis showing the average Brier score and the y-axis showing the consistency metric . The y-axis values are aggregated across all checks for each forecaster and averaged over the instantiated consistency check tuples. Lower scores are better for both axes.

Bayesian consistency checks are the best proxies for forecasting performance. Figure 2a shows the strong correlation between certain consistency checks from Table 3 and average Brier scores across different forecasters. This relationship suggests that COND, which tracks logical consistency in conditional probability estimates, serves as a proxy for overall forecasting accuracy, *without* knowing how the questions resolved.

Correlation: 0.69 CondCondChecker.frequentist.avg_violation vs avg_brier_score (newsapi)
Correlation: 0.92 CondChecker.arbitrage_scaled.avg_violation vs avg_brier_score (scraped)
0.14 0.16 0.18 0.20 0.22 0.24 avg_brier_score 0.1 0.2 0.3 0.4 0.5 0.6 CoT-GPT-4o-mini CoT-GPT-4o-mini CoT-L3-8B
0.17 0.18 0.19 0.20 0.21 0.22 0.23 avg_brier_score 0.005 0.010 0.015 0.020 GPT-4o-mini Basic-GPT-4o-mini avg
_vio lati on GPT-4o-08 GPT-4o-05 avg
_vio lati on CoT-o1-mini CoT-Sonnet CoT-L3-70B CoT-L3-8B
CoT-L3-405B
CoT-GPT-4o-08 Basic-GPT-4o-08 Basic-GPT-4o-05 CoT-L3-70B
CoT-L3-405B
Basic-L3-70B
Basic-L3-405B
Sonnet CoT-o1-mini CoT-GPT-4o-08 CoT-o1-preview Basic-Sonnet CoT-Sonnet CoT-o1-preview
Certain consistency metrics are not well correlated with forecasting performance. The measured correlations between the consistency checks and Brier scores are given in Table 1. We see that some checks yield higher signal on the ground truth performance than others. Aggregating different consistency metrics seems to improve the correlation. We note that the selection of forecasters we test is quite limited, so we cannot guarantee the trends here are representative of future LLM forecasters. Part of the correlation can be attributed to better models being both more consistent and better forecasters. For comparison, the correlations of the Brier score of our forecasters and the MMLU Hendrycks et al. (2020) (college split) error rate on the best approximation of our forecasters in Appendix F are 0.38 and 0.55 on the NewsAPI and scraped datasets, respectively. We include all data (questions, tuples, forecasts, and scores) in the supplementary material.

| Scraped     | NewsAPI     |           |             |       |
|-------------|-------------|-----------|-------------|-------|
| Arbitrage   | Frequentist | Arbitrage | Frequentist |       |
| NEGATION    | 0.60        | 0.67      | -0.36       | -0.13 |
| PARAPHRASE  | 0.57        | 0.61      | 0.13        | 0.24  |
| CONSEQUENCE | 0.51        | 0.52      | 0.21        | 0.30  |
| ANDOR       | 0.20        | 0.25      | 0.02        | 0.06  |
| AND         | 0.68        | 0.72      | 0.54        | 0.71  |
| OR          | 0.14        | 0.24      | -0.24       | -0.31 |
| BUT         | 0.20        | 0.67      | 0.63        | 0.77  |
| COND        | 0.92        | 0.87      | 0.71        | 0.69  |
| CONDCOND    | 0.78        | 0.71      | 0.75        | 0.69  |
| EXPEVIDENCE | 0.20        | 0.77      | -0.11       | 0.06  |
| Aggregated  | 0.62        | 0.85      | 0.49        | 0.66  |

Even good reasoning models are inconsistent. We give the full set of consistency metrics for OpenAI's o1-mini in Table 2. The Frac column counts the fraction of tuples for which the violation exceeded a certain threshold; see the full exposition of what the thresholds mean in Appendices D and E. The frequentist metric is not directly comparable to the arbitrage metric, but the respective violation counts ("Frac" in the table) are. OpenAI's o1-mini forecaster, despite being one of the best reasoning models so far, violates consistency checks more than the (0.5, 0.6) threshold from Section 2 very often.

| Scraped     | NewsAPI     |           |             |      |      |      |      |      |
|-------------|-------------|-----------|-------------|------|------|------|------|------|
| Arbitrage   | Frequentist | Arbitrage | Frequentist |      |      |      |      |      |
| Check       | Avg         | Frac      | Avg         | Frac | Avg  | Frac | Avg  | Frac |
| NEGATION    | 0.07        | 58%       | 0.26        | 61%  | 0.08 | 52%  | 0.27 | 56%  |
| PARAPHRASE  | 0.07        | 56%       | 0.26        | 61%  | 0.06 | 53%  | 0.24 | 56%  |
| CONSEQUENCE | 0.03        | 27%       | 0.13        | 29%  | 0.03 | 18%  | 0.10 | 19%  |
| ANDOR       | 0.09        | 65%       | 0.34        | 71%  | 0.07 | 57%  | 0.29 | 67%  |
| AND         | 0.02        | 24%       | 0.11        | 27%  | 0.03 | 23%  | 0.11 | 24%  |
| OR          | 0.11        | 48%       | 0.30        | 50%  | 0.05 | 48%  | 0.21 | 50%  |
| BUT         | 0.11        | 60%       | 0.40        | 79%  | 0.11 | 63%  | 0.38 | 80%  |
| COND        | 0.04        | 41%       | 0.22        | 52%  | 0.07 | 66%  | 0.29 | 70%  |
| CONDCOND    | 0.03        | 30%       | 0.19        | 45%  | 0.04 | 54%  | 0.23 | 71%  |
| EXPEVIDENCE | 0.04        | 47%       | 0.27        | 69%  | 0.05 | 45%  | 0.28 | 63%  |
| Aggregated  | 0.06        | −         | 0.25        | −    | 0.06 | −    | 0.24 | −    |

Long-horizon consistency benchmark. The results of the previous section indicate that, even on longer time horizons where it's not possible to have ground truth resolutions, we can still evaluate and compare different forecasters via consistency metrics. We create a dataset of 900 synthetic questions resolving in 2028 and create 3000 tuples in total from this dataset using the method described in Section 3.2, to evaluate the consistency of the forecasters in questions with a longer horizon, where it's not possible to have the ground truth resolutions. Examples of questions and the results for gpt-4o are in Appendix L. We intend this dataset as a working prototype for a continual long-term forecasting benchmark.

## 5 Ar B I T R A G Efo R E C A S T E R: Can We Design A More Consistent Forecaster?

Let (x1*, . . . x*n) be a question tuple for some consistency check R, e.g. (P, ¬P). Given forecasts F(x1)*, ...*F(xn), the arbitrage metric maximization problem in Equation 1 computes the following
(as the argmax and max of the arbitrage respectively):
1. Improved forecasts F
′(x1)*, ...*F
′(xn) which are consistent, i.e. satisfy S; and 2. The profit earned by an arbitrageur who bets these improved forecasts against the original ones - this is the actual metric.

This leads us to wonder: can we use these "improved consistent forecasts" to build a new forecaster which builds on the base forecaster F*, but is more consistent on* R?

We introduce: the ArbitrageForecaster with base F arbitraged on consistency check R,
denoted by ⟨F⟩R, which computes its forecast on a question x as follows:
1. Instantiates a tuple (x1*, . . . x*n) satisfying R;
2. Queries F to obtain F(x1)*, . . .* F(xn);
3. Arbitrages these base forecasts per Eq 1 and returns the arbitraged forecast for x1.

Despite what one might assume, however, an ArbitrageForecaster is not "definitionally" consistent on the check it is arbitraged on, but rather its consistency must be investigated empirically. Suppose, for example, that a forecaster produces forecasts F(P) = 0.5, F(para(P)) = 0.6, F(para(para(P))) = 0.7. Then F
′:= ⟨F⟩PARAPHRASE would produce forecasts F
′(P) ≈ 0.55, F

′(para(P)) ≈ 0.65, which are not consistent.

Appendix G contains a precise definition of ArbitrageForecaster, including the case of sequentially arbitraging on multiple checks ⟨F⟩[R1*,...*Rs], and a theoretical discussion of its consistency properties. In particular, we list strong theoretical reasons to expect consistency gains from *recursive* ArbitrageForecaster setups, i.e. ⟨F⟩
rR := *⟨⟨F⟩*
r−1 R ⟩R, in particular with NEGATION, as well as in a non-recursive ArbitrageForecaster with EXPEVIDENCE. Due to these priorities and the high costs of running recursive ArbitrageForecasters (see Appendix G.1), we limited ourselves to studying only a small number of ArbitrageForecaster setups, with a limited number of checks rather than the whole list; specifically: ⟨g⟩
rN , ⟨g⟩
rP, ⟨g⟩
r [N,P ]
,
⟨g⟩[E]∗s where g :=gpt-4o-mini, *N, P, E* are NEGATION, PARAPHRASE, EXPEVIDENCE respectively, and r and s vary from 0 to 4. The full results of our experiments with these forecasters are reported in Appendix G.2; our key takeaways from these preliminary runs look hopeful:
- In the case of the checks we tested, **arbitraging on a check indeed makes a forecaster**
more consistent on that check, with increasing consistency gains with recursive depth, as shown in Fig 3. Crucially, this also applied when the arbitraging was on more than a single check: ⟨g⟩
r [N,P ]
did well on *both* NEGATION and PARAPHRASE; arbitraging on the next check did not increase inconsistency on the first. We are cautiously optimistic that this may extend to the full list of checks in Table 3.

- This consistency gain was greatest with NEGATION, followed by PARAPHRASE**, and**
lowest with EXPE**VIDENCE**. This finding is in line with our hypothesis in Appendix G that ArbitrageForecaster would be particularly effective on consistency checks which are *symmetric*. and instantiate *deterministically*.

- **We do not observe reliable improvements on ground truth forecasting performance,**
or on consistency checks other than the ones we arbitrage on. I.e. ⟨F⟩R1 does not reliably do better on R2.

## 6 Future Work

We have developed a comprehensive benchmark of *static consistency checks* for LLM forecasters, and demonstrated its correlation with ground truth accuracy, suggesting that our consistency metrics could serve as a proxy for accuracy when we do not have access to ground truth. We envision several directions in which our framework could be extended: Consistency in decision-making. AI systems may be used not only to make forecasts that inform decisions, but also to take decisions directly. Here too, we can have a notion of inconsistency:
for example, *intransitive preferences* 4 - and analogously, an inconsistent decision-maker may be exploited by an arbitrageur. Training for consistency. Modulo consideration of the cost-benefit to safety, our methods could be used train LLMs for consistency, minimizing our violation metrics. This may or may not impact overall forecasting performance and other AI capabilities. One may also imagine an AlphaZero-style set-up, where an LLM F is trained on the outputs of ⟨F⟩
r, i.e. a recursive ArbitrageForecaster wrapped around it. Further experiments with **ArbitrageForecaster**. Most of our experiments with ArbitrageForecaster involved arbitraging on only a *single* check (apart from one experiment with both NEGATION and PARAPHRASE), due to the cost limitations described in G.1. It is easy to imagine how a bad forecaster could still overfit a single check: simply forecasting 50% probability for all questions will pass PARAPHRASE, EXPEVIDENCE and NEGATION - but we expect that being consistent under a variety of checks is difficult without a consistent world model. One approach to using more checks cheaply, particularly in training, may be to *randomly sample* a number of consistency checks for each question.

Dynamic generation of consistency checks. Although we found strong correlations between ground truth accuracy and consistency among existing LLM forecasters, our results with ArbitrageForecaster demonstrate that this isn't necessarily the case: it is possible to do well 4See e.g. Fishburn (1970) and the Von Neumann–Morgenstern utility theorem for an introduction to decision rationality.

NegChecker.default.avg_violation (scraped)
Basic-GPT-4o-mini CF-N1 CF-N2 CF-N3 CF-N4 Forecasters 0.000 0.005 0.010 0.015 0.020 0.025 0.030 0.035 0.040 avg_viola tion ParaphraseChecker.default.avg_violation (scraped)
Basic-GPT-4o-mini CF-P1 CF-P2 CF-P3 CF-P4 Forecasters 0.000 0.002 0.004 0.006 0.008 0.010 0.012 0.014 0.016 avg_ viola tio n NegChecker.default.avg_violation (scraped)
Basic-GPT-4o-mini CF-NP1 CF-NP2 CF-NP3 CF-NP4 Forecasters 0.000 0.005 0.010 0.015 0.020 0.025 0.030 0.035 0.040 avg
_vi ola tio n ParaphraseChecker.default.avg_violation (scraped)
Basic-GPT-4o-mini CF-NP1 CF-NP2 CF-NP3 CF-NP4 Forecasters 0.000 0.002 0.004 0.006 0.008 0.010 0.012 0.014 0.016 avg_
viola tio n

on consistency without improving ground truth. In particular, this means that consistency as a training metric could be "Goodharted" by a learning AI model (Karwowski et al., 2023). One way to prevent this may be via adversarial training: i.e. have an adversarial agent instantiate consistency checks that it believes the agent will perform poorly on. Evaluating RAG-augmented forecasters. We have conducted some preliminary experiments evaluating state-of-the-art forecasters such as Halawi et al. (2024). Unfortunately, we could not reproduce the system from Halawi et al. (2024) at the time of writing, due to deprecations in the Google News API (we could not obtain access to the alternative Newscatcher API). At the time of writing, we are not aware of other publicly-available LLM forecasting systems that are competitive with the results of Halawi et al. (2024) (there exist proprietary systems that may be competitive, such as FutureSearch (2024)). We thus leave the evaluation of better forecasters like Halawi et al. (2024) and Phan et al. (2024) to future work, once such forecasters are more widely available.

## Author Contributions

DP and APS developed consistency checks and the arbitrage and frequentist metrics. DP, AA, APS,
and EW worked on the LLM question to evaluation pipeline. APS thought of and implemented ArbitrageForecaster. VB created the news-derived question dataset. AS and DP created the scraped question dataset. AA and DP created the 2028 synthetic question dataset. DP started and led the project. FT proposed correlating consistency with forecasting accuracy and advised the project. All authors helped with the writing. DP and APS wrote the first draft of the paper.

## Acknowledgements

We thank Danny Halawi for extensive discussions and help with our setup. We thank Brendan Murphy, Ezra Karger, Fred Zhang, and Tatsunori Hashimoto for helpful discussions and feedback on the paper and forecasting in general. We thank Berkeley SPAR for connecting collaborators, and BERI for partially funding the project.

## References

Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mane. Con- ´
crete problems in AI safety. *arXiv preprint arXiv:1606.06565*, 2016.

Kenneth J. Arrow, Robert Forsythe, Michael Gorham, Robert Hahn, Robin Hanson, John O. Ledyard, Saul Levmore, Robert Litan, Paul Milgrom, Forrest D. Nelson, George R. Neumann, Marco Ottaviani, Thomas C. Schelling, Robert J. Shiller, Vernon L. Smith, Erik Snowberg, Cass R. Sunstein, Paul C. Tetlock, Philip E. Tetlock, Hal R. Varian, Justin Wolfers, and Eric Zitzewitz. The Promise of Prediction Markets. *Science*, 320(5878):877–878, May 2008. doi: 10.1126/science.1157679.

Henry Berg and Todd A Proebsting. Hanson's automated market maker. The Journal of Prediction Markets, 3(1):45–59, 2009.

Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering Latent Knowledge in Language Models Without Supervision. In The Eleventh International Conference on Learning Representations, September 2022.

Tsong Y Chen, Shing C Cheung, and Shiu Ming Yiu. Metamorphic testing: a new approach for generating next test cases. Technical report, The Hong Kong University of Science and Technology, 1998.

Maria Christakis, Hasan Ferit Eniser, Jorg Hoffmann, Adish Singla, and Valentin W ¨ ustholz. ¨
Specifying and testing k-safety properties for machine-learning models. arXiv preprint arXiv:2206.06054, 2022.

Yanai Elazar, Nora Kassner, Shauli Ravfogel, Abhilasha Ravichander, Eduard Hovy, Hinrich Schutze, and Yoav Goldberg. Measuring and improving consistency in pretrained language mod- ¨ els. *Transactions of the Association for Computational Linguistics*, 9:1012–1031, 2021.

Peter C. Fishburn. *Utility Theory for Decision Making*. Wiley, January 1970. ISBN 978-0-47126060-8.

Lukas Fluri, Daniel Paleka, and Florian Tramer. Evaluating superhuman models with consistency `
checks. In *2024 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)*,
volume 31, page 194–232. IEEE, April 2023. doi: 10.1109/satml59370.2024.00017. URL
http://dx.doi.org/10.1109/SaTML59370.2024.00017.

FutureSearch. FUTURESEARCH: Manifold markets trading bot, 2024. URL https://
manifold.markets/FUTURESEARCH. Accessed on 26-Sept-2024.

Ozzie Gooen. Scorable Functions: A Format for Algorithmic Forecasting, May 2024. Danny Halawi, Fred Zhang, Chen Yueh-Han, and Jacob Steinhardt. Approaching Human-Level Forecasting with Language Models, February 2024.

Robin Hanson. Logarithmic Market Scoring Rules for Modular Combinatorial Information Aggregation. *The Journal of Prediction Markets*, 1(1):3–15, January 2002. doi: 10.5750/jpm.v1i1.417.

Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. *arXiv preprint arXiv:1903.12261*, 2019.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Elvis Hsieh, Preston Fu, and Jonathan Chen. Reasoning and tools for human-level forecasting. arXiv preprint arXiv:2408.12036, 2024.

Geoffrey Irving, Paul Christiano, and Dario Amodei. AI safety via debate. arXiv preprint arXiv:1805.00899, 2018.

Myeongjun Jang and Thomas Lukasiewicz. Consistency analysis of ChatGPT. *arXiv preprint* arXiv:2303.06273, 2023.

Kaarel, gekaklam, Walter Laurito, Kay Kozaronek, AlexMennen, and June Ku. Searching for a model's concepts by their shape - a theoretical framework, February 2023.

Daniel Kahneman, Ilana Ritov, David Schkade, Steven J Sherman, and Hal R Varian. Economic preferences or attitude expressions?: an analysis of dollar responses to public issues. *Elicitation* of preferences, pages 203–242, 2000.

Ezra Karger, Houtan Bastani, Chen Yueh-Han, Zachary Jacobs, Danny Halawi, Fred Zhang, and Philip E Tetlock. Forecastbench: A dynamic benchmark of ai forecasting capabilities. *arXiv* preprint arXiv:2409.19839, 2024.

Jacek Karwowski, Oliver Hayman, Xingjian Bai, Klaus Kiendlhofer, Charlie Griffin, and Joar Max Viktor Skalse. Goodhart's Law in Reinforcement Learning. In *The Twelfth International* Conference on Learning Representations, October 2023.

Li-Cheng Lan, Huan Zhang, Ti-Rong Wu, Meng-Yu Tsai, I Wu, Cho-Jui Hsieh, et al. Are AlphaZero-like agents robust to adversarial perturbations? *arXiv preprint arXiv:2211.03769*, 2022.

Tao Li, Vivek Gupta, Maitrey Mehta, and Vivek Srikumar. A logic-driven framework for consistency of neural models. *arXiv preprint arXiv:1909.00126*, 2019.

Xiang Lisa Li, Vaishnavi Shrivastava, Siyan Li, Tatsunori Hashimoto, and Percy Liang. Benchmarking and improving generator-validator consistency of language models. arXiv preprint arXiv:2310.01846, 2023.

Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods. *arXiv preprint arXiv:2109.07958*, 2021.

Jason Liu. Instructor: Structured LLM Outputs, May 2024. URL https://github.com/
jxnl/instructor. Version 1.4.1.

Muehlhauser, Luke. How Feasible Is Long-range Forecasting?, October 2019. Long Phan, Adam Khoja, Mantas Mazeika, and Dan Hendrycks. LLMs are superhuman forecasters, 2024. URL https://www.safe.ai/blog/forecasting. Accessed on 26-Sept-2024.

Harsh Raj, Vipul Gupta, Domenic Rosati, and Subhabrata Majumdar. Semantic consistency for assuring reliability of large language models. *arXiv preprint arXiv:2308.09138*, 2023.

Leonard J. Savage. Elicitation of Personal Probabilities and Expectations. Journal of the American Statistical Association, 66(336):783–801, 1971. ISSN 0162-1459. doi: 10.2307/2284229.

Philipp Schoenegger and Peter S. Park. Large Language Model Prediction Capabilities: Evidence from a Real-World Forecasting Tournament, October 2023.

Philipp Schoenegger, Indre Tuminauskaite, Peter S. Park, and Philip E. Tetlock. Wisdom of the Silicon Crowd: LLM Ensemble Prediction Capabilities Rival Human Crowd Accuracy, May 2024.

Arnab Sharma and Heike Wehrheim. Testing monotonicity of machine learning models, 2020. Philip E Tetlock, Christopher Karvetski, Ville A Satopa¨a, and Kevin Chen. Long-range subjective- ¨
probability forecasts of slow-motion variables in world politics: Exploring limits on expert judgment. *Futures & Foresight Science*, 6(1):e157, 2024.

Susan Vineberg. Dutch Book Arguments. In Edward N. Zalta and Uri Nodelman, editors, The Stanford Encyclopedia of Philosophy. Metaphysics Research Lab, Stanford University, fall 2022 edition, 2022.

Qi Yan, Raihan Seraj, Jiawei He, Lili Meng, and Tristan Sylvain. Autocast++: Enhancing world event prediction with zero-shot ranking-based context retrieval. *arXiv preprint arXiv:2310.01880*, 2023.

Andy Zou, Tristan Xiao, Ryan Jia, Joe Kwon, Mantas Mazeika, Richard Li, Dawn Song, Jacob Steinhardt, Owain Evans, and Dan Hendrycks. Forecasting future world events with neural networks. *arXiv preprint arXiv:2206.15474*, 2022a.

Andy Zou, Tristan Xiao, Ryan Jia, Joe Kwon, Mantas Mazeika, Richard Li, Dawn Song, Jacob Steinhardt, Owain Evans, and Dan Hendrycks. Forecasting Future World Events With Neural Networks. *Advances in Neural Information Processing Systems*, 35:27293–27305, December 2022b.

## A Related Work

Metamorphic and consistency checks. Checking logical properties of outputs of programs under semantic-preserving transforms has a long history (Chen et al., 1998). Before Fluri et al. (2023), variants of the consistency check framework were used for simple ML models (Christakis et al., 2022; Sharma and Wehrheim, 2020), vision (Hendrycks and Dietterich, 2019), and chat LLMs (Jang and Lukasiewicz, 2023), among other areas. Li et al. (2019) consider logical consistency checks beyond paraphrasing and negation for simple ML models. Forecasting and large language models. LLMs and forecasting date back to Zou et al. (2022a) and Yan et al. (2023). Recently, strong performance of LLM forecasters on prediction market datasets has been claimed in (Halawi et al., 2024; Tetlock et al., 2024; Hsieh et al., 2024; Phan et al., 2024). Concurrent with our work, Karger et al. (2024) have introduced an automatically updating benchmark for forecasting. Scalable oversight and failures of superhuman AI. The difficulty of evaluating models with superhuman performance in domains without a source of ground truth has long been acknowledged, and falls under the umbrella of *scalable oversight* (Amodei et al., 2016). Forecasting using AI oracles is one such domain. The use of consistency checks for scalable oversight has been studied in the simpler context of superhuman game AIs (Lan et al., 2022; Fluri et al., 2023), and in general question-answering tasks via debate (Irving et al., 2018). Consistency evaluations for LLMs. Even on tasks where the ground truth is in principle knowable, consistency evaluations have long helped in cases where checking consistency is easier than getting the ground truth labels (Elazar et al., 2021; Li et al., 2023). Raj et al. (2023) measure paraphrasing consistency and ground truth accuracy on TruthfulQA (Lin et al., 2021) and find little to no correlation. Some forms of consistency checks have been applied on model internals to discover features related to LLM truthfulness and reliability (Burns et al., 2022; Kaarel et al., 2023).

## B Table Of Consistency Checks

Table 3 includes all the consistency checks tested for in our benchmark. In most of them, we leave the logical relations between forecasting questions R implicit by constructing the sentences directly.

For instance, R(x1, x2) := x1 = ¬x2 is implied by simply writing x1, x2 as P, ¬P. In the rest of the appendix, we use the sentence-based (*P, Q* instead of x1, x2) notation.

| Name                          | Tuple                          | Condition (S)                                        |             |          |    |
|-------------------------------|--------------------------------|------------------------------------------------------|-------------|----------|----|
| NEGATION                      | (P, ¬P)                        | F(P) + F(¬P) = 1                                     |             |          |    |
| PARAPHRASE R(P, Q) := P       | ⇐⇒ Q                           | (P, Q)                                               | F(P) = F(Q) |          |    |
| CONSEQUENCE R(P, Q) := P =⇒ Q | (P, Q)                         | F(P) ≤ F(Q)                                          |             |          |    |
| ANDOR                         | (P, Q, P ∧ Q, P ∨ Q)           | F(P) + F(Q) = F(P ∨ Q) + F(P ∧ Q)                    |             |          |    |
| AND                           | (P, Q, P ∧ Q)                  | max(F(P) + F(Q) − 1, 0) ≤ F(P ∧ Q) ≤ min(F(P), F(Q)) |             |          |    |
| OR                            | (P, Q, P ∨ Q)                  | max(F(P), F(Q))                                      | ≤           | F(P ∨ Q) | ≤  |
| min(1, F(P) + F(Q))           |                                |                                                      |             |          |    |
| BUT                           | (P, ¬P ∧ Q, P ∨ Q)             | F(P ∨ Q) = F(P) + F(¬P ∧ Q)                          |             |          |    |
| COND                          | (P, Q|P, P ∧ Q)                | F(P)F(Q|P) = F(P ∧ Q) F(P)F(Q|P)F(R|P ∧Q) = F(P ∧Q∧  |             |          |    |
| CONDCOND                      | (P, Q|P, R|(P ∧ Q), P ∧ Q ∧ R) | R)                                                   |             |          |    |
| EXPEVIDENCE                   | (P, Q, P|Q, P|¬Q)              | F(P) = F(P|Q)F(Q) + F(P|¬Q)(1 − F(Q))                |             |          |    |

The consistency checks in Table 3 represent core logical relationships between probabilities, but many other forms of consistency checks are possible. Here are two examples that could extend our framework:
- **Comparative checks:** Building on generator-validator checks from Li et al. (2023), we could ask a forecaster to predict both F(P), F(Q), and separately whether P or Q is more likely. The forecaster's probability estimates should match their comparative judgment.

- **Monotonicity checks:** Fluri et al. (2023) propose a variant of CONSEQUENCE for realvalued quantities, where predictions must respect the monotonic ordering of a sequence of future values. This connects to *scope insensitivity* (Kahneman et al., 2000), a cognitive bias where humans fail to scale probability estimates appropriately with the magnitude of outcomes.

We do not include a specific consistency check for Bayesian updates, as conditional probabilities are already covered by COND, CONDCOND, and EXPEVIDENCE.

## C Data Types Used In Our Pipeline C.1 Forecasting Questions

Figure 4 shows the data stored on forecasting questions. Of these, only *title* and *body* are shown to the forecaster.

Forecasting question Data Type

- id: Universally Unique Question Identifier (UUID), auto-generated using a default factory. - **title**: Title of the forecasting question. - **body**: Detailed resolution criteria, background information, etc. - resolution **date**: The date when the question is expected to be resolved. We only consider questions that have a clear date when the resolution should be decided.

- question **type**: Type of the forecasting question; in this paper, only *binary* and conditional-binary. Options not used in this paper include multiple-choice, *interval*, continuous-value, or *opinion*.

- data **source**: Source of the question, either the website from which it was scraped or synthetic.

- created **date**: The date when the question was created, or *null* if not important for the meaning of the question.

- url: URL of the source if the question was scraped, else null.

- **metadata**: Any additional information, e.g., topics, tags, *category*; but also data fields specific to Metaculus, Manifold, etc; the source articles for NewsAPI-generated questions; or instantiation metadata for questions in consistency tuples.

- **resolution**: A boolean indicating whether the question resolves to YES or NO, or *null* if unresolved.
Figure 4: Description of the forecasting question data type.

For instance, a forecasting question from Metaculus, such as the one shown in Figure 5, will be stored in the form depicted in Figure 6 using our method. The original question, which asks whether SpaceX will land people on Mars before 2030, is presented with detailed conditions for resolution, including specific criteria such as the confirmation of the landing by SpaceX and the completion of an extravehicular activity (EVA) on the Martian surface. The data type in Figure 4 is compatible (after appropriate processing) with scraped questions from Metaculus and Manifold, and standardization helps with synthetic question generation and tuple instantiation. We do not include information about human forecasts because we explicitly focus on evaluation without relying on any human-generated probabilities.

Will SpaceX land people on Mars before Resolution Criteria This question will resolve as Yes if a SpaceX-branded mission successfully lands one or more living human beings on the surface of Mars before 2030. The landing itself of the human crew on Mars must occur before January 1, 2030, 00:00 UTC.

At least one person aboard the lander must survive the landing, however it is not necessary for the person to survive long-term or make a return trip to Earth, nor is it necessary for the mission to intend a return or long-term survival.

A "SpaceX-branded" mission is defined to mean that the SpaceX-associated logos on the spacecraft involved (both the boosters and the Mars-bound craft) have a larger surface area than the logos of any other entity
Example forecasting question (scraped)

- id: 07b11b15-6872-4280-a94f-17b6d15a1b8a - **title**: Will SpaceX land people on Mars before 2030? - **body**: This question will resolve as Yes if SpaceX successfully lands at least one human on the surface of Mars on or before December 31, 2030. The landing must be confirmed by SpaceX through an official announcement or live broadcast. The human(s) must be alive upon landing and must perform at least one extravehicular activity (EVA) on the Martian surface, which must be documented and released to the public. In the event of a dispute regarding the success of the mission, the resolution will defer to the judgment of an international space agency such as NASA or ESA. If no landing attempt is made by the specified date, or if all attempts fail to meet the above criteria, the question will resolve as No.

- resolution **date**: 2030-12-31 23:59:59+00:00
- question **type**: binary
- data **source**: metaculus - url: https://www.metaculus.com/questions/349 - **metadata**:
- **topics**:
* id: 184, **slug**: elon-musk, **name**: Elon Musk, **link** id: 27681, num **questions**:
159
* id: 485, **slug**: spacex-reusable-launch-system-development-program, **name**:
SpaceX reusable launch system, **link** id: 27682, num **questions**: 130
* id: 1365, **slug**: spacex, **name**: SpaceX, **link** id: 75197, num **questions**: 112
* id: 564, **slug**: colonization-of-mars, **name**: Colonization of Mars, **link** id:
27683, num **questions**: 70
* id: 1768, **slug**: spacex-mars-transportation-infrastructure, **name**: SpaceX Mars transportation infrastructure, **link** id: 40982, num **questions**: 5
- **resolution**: null
Figure 6: Example of a forecasting question scraped from Metaculus.

By processing this question through our pipeline, we retain all relevant details, such as the resolution date and specific criteria for a binary outcome, while structuring the data in a more standardized format to facilitate further analysis. Additionally, associated metadata, including related topics and links to other questions, is also preserved.

Example forecasting question (synthetic) - id: 4b98368c-6287-47e0-8f9e-5917e2a24a3d - **title**: Will Russia launch a manned mission to the Moon before 2030? - **body**: This question will resolve as Yes if, before January 1, 2030, the Russian Federation successfully launches and completes a manned mission to the Moon, where 'successful' is defined as a mission where astronauts land on the lunar surface and return safely to Earth. The mission must be officially recognized by Roscosmos or another authoritative space agency. In the event of a joint mission involving Russia and other countries, the mission will still resolve as Yes if Russian astronauts are part of the crew that lands on the Moon. If no such mission is launched, or if a mission is launched but does not meet the above criteria, the question will resolve as No. In the case of ambiguity or lack of clear public information by the resolution date, the question will resolve as No unless official statements or evidence are provided by Roscosmos or an equivalent authoritative body that confirm the mission's success as per the defined criteria.

- resolution **date**: 2030-12-31 23:59:59+00:00
- question **type**: binary - data **source**: synthetic - url: null - **metadata**:
- **tags**:
* Russia
- **categories**:
* Space
- **resolution**: null Figure 7: Example of a synthetic forecasting question. All question generations are seeded with the *metadata* field.

As an example, we also show a forecasting question generated synthetically using the source tags "Russia" and "Moon" could ask whether Russia will launch a manned mission to the Moon by 2030. The structure and format of this synthetic question, as illustrated in Figure 7, mirror those of real forecasting questions while maintaining the essential metadata for context.

## C.2 Examples Of Instantiated Tuples

In the following examples, we focus on the question title for clarity. Figure 8 illustrates an instantiated AND tuple, starting from forecasting questions (P and Q) that address distinct events regarding artificial intelligence policy in the U.S. and Canada, together with a conjunction question (P and Q) about their joint occurrence by a specified date. Figure 9 presents an instantiated EXPEVIDENCE tuple, examining the global space industry's revenue potential alongside the political dynamics in the U.S. House of Representatives, including conditional questions that evaluate the influence of one event on another. We note that making the detailed resolution criteria ("body" field) actually correspond to the composite event is not straighforward, and is only in reach of the newest generations of LLMs. A different design option would be to just list the original questions and resolution criteria separately in the "body" field, and then say what the logical operation is. We opt against it for two reasons:
- A separate, unnatural format for composite questions might induce qualitatively different behaviors in LLM forecasters.

- Future works in this framework might not rely on simple logical operations, but rather on an advanced LLM grader that computes "do these forecasts make sense taken together". Our current design allows for an easier extension to this direction.