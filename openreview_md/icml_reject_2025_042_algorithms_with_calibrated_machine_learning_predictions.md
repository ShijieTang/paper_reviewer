# 

Judy Hanwen Shen 1 **Ellen Vitercik** 1 2 **Anders Wikum** 2

## Abstract

The field of *algorithms with predictions* incorporates machine learning advice in the design of online algorithms to improve real-world performance. A central consideration is the extent to which predictions can be trusted—while existing approaches often require users to specify an aggregate trust level, modern machine learning models can provide estimates of prediction-level uncertainty. In this paper, we propose *calibration* as a principled and practical tool to bridge this gap, demonstrating the benefits of calibrated advice through two case studies: the *ski rental* and online job scheduling problems. For ski rental, we design an algorithm that achieves near-optimal prediction-dependent performance and prove that, in high-variance settings, calibrated advice offers more effective guidance than alternative methods for uncertainty quantification. For job scheduling, we demonstrate that using a calibrated predictor leads to significant performance improvements over existing methods. Evaluations on real-world data validate our theoretical findings, highlighting the practical impact of calibration for algorithms with predictions.

## 1. Introduction

In recent years, advances in machine learning (ML) models have inspired researchers to revisit the design of classic online algorithms, incorporating insights from ML-based advice to improve decision-making in real-world environments. This research area, termed algorithms with predictions, seeks to design algorithms that are both robust to worst-case inputs and achieve performance that improves with prediction accuracy (a desideratum termed *consistency*) (Lykouris & Vassilvitskii, 2018). Many learning-augmented 1Department of Computer Science, Stanford University, Stanford, CA, USA 2Department of Management Science & Engineering, Stanford University, Stanford, CA, USA. Correspondence to: Anders Wikum <wikum@stanford.edu>.

algorithms have been developed for online decision-making tasks ranging from rent-or-buy problems like ski rental
(Purohit et al., 2018; Anand et al., 2020; Sun et al., 2024) to sequencing problems like job scheduling (Cho et al., 2022). This framework often produces a family of algorithms indexed by a single parameter intended to reflect the *global* reliability of the ML advice. Extreme settings of this parameter yield algorithms that make decisions as if the predictions are either all perfect or all uninformative (e.g., Mahdian et al., 2007; Lykouris & Vassilvitskii, 2018; Purohit et al., 2018; Rohatgi, 2020; Wei & Zhang, 2020; Antoniadis et al., 2020). In contrast, ML models often produce *local*, prediction-specific uncertainty estimates, exposing a disconnect between theory and practice. For instance, many neural networks provide calibrated probabilities or confidence intervals for each data point. In this paper, we demonstrate that *calibration* can serve as a powerful tool to bridge this gap. An ML predictor is said to be calibrated if the probabilities it assigns to events match their observed frequencies; when the model outputs a high probability, the event is indeed likely, and when it assigns a low probability, the event rarely occurs. Calibrated predictors convey their uncertainty on each prediction, allowing decision-makers to safely rely on the model's advice, and eliminating the need for ad-hoc reliability estimates. Moreover, calibrating an ML model can easily be accomplished using popular methods (e.g. Platt Scaling (Platt et al., 1999) or Histogram Binning (Zadrozny & Elkan, 2001)) that reduce overconfidence (Vasilev & D'yakonov, 2023). Although we are the first to study calibration for algorithms with predictions, Sun et al. (2024) proposed using conformal prediction in this setting—a common tool in uncertainty quantification (Vovk et al., 2005; Shafer & Vovk, 2008). Conformal predictions provide instance-specific confidence intervals that cover the target with high probability. While these approaches are orthogonal, we prove that calibration can offer key advantages over conformal prediction, especially when the predicted quantities have high variance. In extreme cases, conformal intervals can become too wide to be informative: for binary predictions, a conformal approach returns {0, 1} unless the true label is nearly certain to be 0 or 1. In contrast, calibration still conveys information that aids decision-making.

1

## 1.1. Our Contributions

We demonstrate the benefit of using calibrated predictors through two case studies: the ski rental and online job scheduling problems. Theoretically, we develop and give performance guarantees for algorithms that incorporate calibrated predictions. We validate our theoretical findings with strong empirical results on real-world data, highlighting the practical benefits of our approach.

Ski rental. The *ski rental problem* serves as a prototypical example of a broad family of online rent-or-buy problems, where one must choose between an inexpensive, short-term option (renting) and a more costly, long-term option (buying). In this problem, a skier will ski for an unknown number of days and, each day, must decide to either rent skis or pay a one-time cost to buy them. Generalizations of the ski rental problem have informed a broad array of practical applications in networking (Karlin et al., 2001), caching (Karlin et al., 1988), and cloud computing (Khanafer et al., 2013). We design an online algorithm for ski rental that incorporates predictions from a calibrated predictor. We prove that our algorithm achieves optimal expected prediction-level performance for general distributions over instances and calibrated predictors. At a distribution level, its performance degrades smoothly as a function of the mean-squared error and calibration error of the predictor. Moreover, we demonstrate that calibrated predictions can be more informative than the conformal predictions of Sun et al. (2024) when the distribution over instances has high variance that is not explained by features, leading to better performance. Scheduling. We next study online scheduling in a setting where each job has an urgency level, but only a machinelearned estimate of that urgency is available. This framework is motivated by scenarios such as medical diagnostics, where machine-learning tools can flag potentially urgent cases but cannot fully replace human experts. We demonstrate that using a calibrated predictor provides significantly better guarantees than prior work (Cho et al., 2022), which approached this problem by ordering jobs based on the outputs of a binary predictor. We identify that this method implicitly relies on a crude form of calibration that assigns only two distinct values, resulting in many ties that must be broken randomly. In contrast, we prove that a properly calibrated predictor with finer-grained confidence levels provides a more nuanced job ordering, rigorously quantifying the resulting performance gains.

## 1.2. Related Work

(2022) for a survey). Much of the research provides a parameterized family of algorithms with no assumption on the reliability of predictions (e.g., Lykouris & Vassilvitskii, 2018; Purohit et al., 2018; Wei & Zhang, 2020). Subsequent work has studied more practical settings, such as assuming access to ML predictors learned from samples (Anand et al., 2020), with probabilistic correctness guarantees (Gupta et al., 2022), with a known confusion matrix (Cho et al., 2022), or that provide distributional predictions (Dinitz et al., 2024; Angelopoulos et al., 2024; Lin et al., 2022; Diakonikolas et al., 2021). While conceptually related, these papers do not study uncertainty quantification. Recently, Sun et al. (2024) proposed a framework for quantifying prediction-level uncertainty based on conformal prediction. We show that calibration can offer key advantages over conformal prediction in this context, particularly when predicted quantities exhibit high variance. Calibration for decision-making. A recent line of work examines calibration as a tool for downstream decisionmaking. Gopalan et al. (2023) show that a multi-calibrated predictor can be used to optimize any convex, Lipschitz loss function of an action and binary label. Zhao et al. (2021) adapt the required calibration guarantees to specific offline decision-making tasks, while Noarov et al. (2023) extend this algorithmic framework to the online adversarial setting.

Though closely related to our work, these results do not extend to the (often unwieldy) loss functions encountered in competitive analysis.

## 2. Preliminaries

For clarity, we follow the convention that capital letters (e.g., X) denote random variables and lowercase letters denote realizations of random variables (e.g., the event f(X) = v).

Learning-augmented algorithm design. With each algorithmic task, we associate a set I of possible instances, a set X of features for those instances, and a joint distribution D over *X × I*. Given a target function T : *I → Y* that provides information about each instance, we assume access to a predictor f : *X → Z ⊇ Y* that has been trained to predict the target over D. Let R(f) denote the range of f. If A(*v, i*) is the cost incurred by algorithm A with prediction f(X) = v on instance i ∈ I, and OPT(i) is that of the offline optimal solution, the goal is to minimize either the expected competitive ratio (CR)

$$\operatorname*{\mathbb{E}}_{(X,I)\sim{\mathcal{D}}}\left[{\frac{A(f(X),I)}{\mathrm{OPT}(I)}}\right]$$

Algorithms with predictions. There has been significant recent interest in integrating ML advice into the design of online algorithms (see, e.g., Mitzenmacher & Vassilvitskii or the expected additive regret E [A(f(X), I) − OPT(I)],
depending on context. Both measure the performance of A relative to OPT over D. The former is consistent with prior work on training predictors from samples for algorithms with predictions (Anand et al., 2020), while the latter is commonly used to quantify suboptimality in learningaugmented scheduling (Lindermayr & Megow, 2022; Im et al., 2023). When D and f are clear from context, we refer to these quantities as E[CR(A)] and E[R(A)], respectively.

Calibration. An ML model is said to be *calibrated* if its predictions are, on average, correct. Formally, Definition 2.1. A predictor f : *X → Z* with target T : I → Y is calibrated over D if
$$\mathbb{E}_{(X,I)\sim{\mathcal{D}}}[T(I)\mid f(X)]=f(X).$$
When Y = {0, 1}, the equivalent condition Pr[T(I) = 1 | f(X)] = f(X) requires that f(X) is a reliable probabilistic estimate of the event {T(I) = 1}.

A classic result from the literature on probabilistic forecasting states that calibrated predictions are the global minimizers of proper loss functions (DeGroot & Fienberg, 1983).

However, achieving perfect calibration is difficult in practice. As a result, post-hoc calibration methods aim to minimize calibration error, such as the *max calibration error*, which measures the largest deviation from perfect calibration for any prediction. Definition 2.2. The max calibration error of a predictor f : *X → Z* with target T : *I → Y* over D is

$$\operatorname*{max}_{v\in R(f)}|v-\mathbb{E}[T(I)\mid f(X)=v]|\,.$$

Given any black box ML model and sufficient data, these methods yield a new predictor with a desired level of calibration error with high probability.

## 3. Ski Rental

In this section, we analyze calibration as a tool for uncertainty quantification in the classic online ski rental problem. All omitted proofs in this section are in Appendix A.

## 3.1. Setup

Problem. A skier plans to ski for an unknown number of days Z ∈ N and has two options: buy skis at a one-time cost of b ∈ N dollars or rent them for 1 dollar per day. The goal is to determine how many days to rent before buying, minimizing the total cost. If Z = z were known *a priori*,
the optimal policy would rent for b days when *z < b* and buy immediately otherwise, costing min{*z, b*}. Without knowledge of z, competitive ratios of 2 (Karlin et al., 1988)
and e e−1
(Karlin et al., 1994) are tight for deterministic and random strategies, respectively. For convenience, we study a continuous variant of this problem where *Z, b, k* ∈ R≥0 as in prior work (Anand et al., 2020; Sun et al., 2024).

Algorithm 1 Ak∗
input: prediction f(X) = v, max calibration error α if v ≤
4+3α 5**then**
Rent for b days before buying.

else Rent for b q1−v+α v+αdays before buying.

end if Predictions. Let X be a set of skier features, I = R≥0 be the set of possible days skied, and D be an unknown distribution over feature/duration pairs X ×R≥0. Motivated by the form of the optimal offline algorithm, we analyze a calibrated predictor f : X → [0, 1] for the target T(z) =
1{z>b}, indicating if the skier will ski for more than b days.

For (X, Z) ∼ D, a prediction of f(X) ≈ 1 (respectively, f(X) ≈ 0) means *Z > b* (respectively, Z ≤ b) with high certainty.

Learning-augmented ski rental. A deterministic learning-augmented algorithm Ak for ski rental takes as input a prediction f(X) = v and returns a recommendation:
"rent skis for k(v) days before buying." The cost of following this policy when skiing for z days is

$$\mathcal{A}_{k}(v,z)=\begin{cases}k(v)+b&\text{if}z>k(v)\\ z&\text{if}z\leq k(v)\end{cases}.$$
.
We aim to select k : [0, 1] → R+ to minimize E[CR(Ak)].

## 3.2. Ski Rental With Calibrated Predictions

In Algorithm 1, we introduce a deterministic policy for ski rental based on calibrated predictions. To avoid following bad advice, the algorithm defaults to a worst-case strategy of renting for b days unless sufficiently confident that the skier will ski for at least b days. In this second case, the algorithm smoothly interpolates between a strategy that rents for bp(1 − α)/α days and one that rents for bpα/(1 + α)
days, where α ∈ [0, 1] is a bound on local calibration error that hedges against greedily following predictions.

Theorem 3.1. Given a predictor f with mean-squared error η and max calibration error α, Algorithm 1 *achieves* E[CR(Ak∗
)] ≤ 1+ 2α+min E[f(X)] + α, 2
√η + 3α	.

As the predictor becomes more accurate (i.e., both η and α decrease), the algorithm's expected CR approaches 1. The rest of this subsection will build to a proof of Theorem 3.1.

Prediction-level analysis. We begin by upper bounding E[CR(Ak) | f(X) = v]. Let Bv = {f(X) = v} be the event that f predicts v ∈ R(f) and C = {*Z > b*} be the Table 1. Objective values for fixed prediction f(X) = v, z days skied, and renting for k(v) days.

| CONDITION             | OPT(z)   | Ak(v, z)   |
|-----------------------|----------|------------|
| (i) z ≤ min{k(v), b}  | z        | z          |
| (ii) k(v) < z ≤ b     | z        | k(v) + b   |
| (iii) b < z ≤ k(v)    | b        | z          |
| (iv) z > max{k(v), b} | b        | k(v) + b   |

event that the number of days skied is more than b. Then

$$\begin{array}{c}{{\mathbb{R}({\mathcal{A}}_{k})\mid B_{v}]=\mathbb{E}[\mathbb{C}\mathbb{R}({\mathcal{A}}_{k})}}\\ {{+\mathbb{E}[\mathbb{C}\mathbb{R}({\mathcal{A}}_{k})}}\end{array}$$

E[CR(Ak) | Bv] = E[CR(Ak) | Bv, C] · Pr[C | Bv] (1)
+ E[CR(Ak) | Bv, Cc] · Pr[C
c| Bv].

Lemma 3.2 bounds each of the quantities from Equation (1). Lemma 3.2. Given a predictor f *with max calibration error* α*, for all* v ∈ R(f),

$$I.\ \operatorname*{Pr}[C\mid f(X)=v]\leq v+\alpha$$
  ## 2 Pr[C${}^{c}$ | $f(X)=0$] $\leq1-v+\alpha$
3. $\mathbb{E}[\text{CR}(\mathcal{A}_k)\mid B_v,C]\leq1+\frac{k(v)}{b}$
4. $\mathbb{E}[\text{CR}(\mathcal{A}_{k})\mid B_{v},C^{c}]\leq1+\frac{b\cdot1_{\{k(v)<b\}}}{k(v)}$
Proof sketch. (1) and (2) follow from the fact that f predicts 1C with max calibration error α. Under C = {Z ≥ b},
one of conditions (iii) or (iv) from Table 3 hold. In either case, Ak(*v, Z*)/OPT(Z) ≤ 1 + k(v)
b. Under C
c, one of conditions (i) or (ii) hold. CR(Ak) = 1 for (i). For (ii),

$${\frac{\mathcal{A}_{k}(v,Z)}{\mathrm{OPT}(Z)}}\leq{\frac{k(v)+b}{k(v)}}=1+{\frac{b\cdot\mathbb{1}_{\{k(v)<b\}}}{k(v)}}.$$

Applying all four bounds to Equation (1) yields

$$\mathbb{E}[\text{CR}(\mathcal{A}_{k})\mid f(X)=v]\leq\tag{2}$$ $$1+2\alpha+\frac{(v+\alpha)k(v)}{b}+1_{\{k(v)<b\}}\cdot\frac{(1-v+\alpha)b}{k(v)}.$$

The renting strategy k∗(v) from Algorithm 1 is the minimizer of the upper bound in Equation (2).

Theorem 3.3. Given a predictor f with max calibration error α, for any prediction v ∈ R(f), Algorithm 1 *achieves*

$$\mathbb{E}[\text{CR}(\mathcal{A}_{k_{*}})\mid f(X)=v]\leq$$ $$1+2\alpha+\min\bigl{\{}v+\alpha,2\sqrt{(v+\alpha)(1-v+\alpha)}\bigr{\}}.$$

Proof sketch. Given a prediction f(X) = v, Algorithm 1 rents for k∗(v) days where

$$k_{*}(v)={\begin{cases}b&{\mathrm{if~}}0\leq v\leq{\frac{4+3\alpha}{5}}\\ b{\sqrt{\frac{1-v+\alpha}{v+\alpha}}}&{\mathrm{if~}}{\frac{4+3\alpha}{5}}<v\leq1.\end{cases}}$$

Evaluating the right-hand-side of Equation (2) at k∗(v) gives

$$\begin{cases}1+2\alpha+(v+\alpha)&\text{if}0\leq v\leq\frac{4+3\alpha}{5}\\ 1+2\alpha+2{\sqrt{(v+\alpha)(1-v+\alpha)}}&\text{if}\frac{4+3\alpha}{5}<v\leq1.\end{cases}$$
The fact that $v+\alpha\leq2\sqrt{(v+\alpha)(1-v+\alpha)}$ for $v\in[0,\frac{4+3\alpha}{5}]$ and $v+\alpha>2\sqrt{(v+\alpha)(1-v+\alpha)}$ for $v\in(\frac{4+3\alpha}{5},1]$ completes the proof.  
Moreover, no deterministic learning-augmented algorithm for ski rental can outperform Algorithm 1 for general distributions D and calibrated predictors f. The construction is non-trivial, so we refer the reader to the proof in Appendix A.

Theorem 3.4. *For all renting strategies* k : [0, 1] → R+,
predictions v ∈ [0, 1] and ϵ > 0*, there exists a distribution* Dϵv and a calibrated predictor f *such that*

$\mathbb{E}[\text{CR}(\mathcal{A}_{k})\mid f(X)=v]\geq1+\min\left\{v,2\sqrt{v(1-v)}\right\}-\epsilon$.  
Global analysis. In extracting a global bound from the conditional guarantee in Theorem 3.3, we encounter a term
(f(X) + α)(1 − f(X) + α) that is an upper bound on the variance of the conditional distribution 1{Z≥b} | f(X).

Lemma 3.5 relates this quantity to error statistics of f. Lemma 3.5. If f : X → [0, 1] *has mean-squared error* η and max calibration error α*, then*

$$\mathbb{E}[f(X)(1-f(X))]\leq\eta+\alpha.$$

Finally, we prove this section's main theorem.

Proof of Theorem *3.1.* By the tower property of conditional expectation, E[CR(Ak∗)] = E-E[CR(Ak∗) | f(X)]. Applying Theorem 3.3 yields

$$\mathbb{E}[\mathrm{CR}({\mathcal{A}}_{k_{*}})]\leq1+2\alpha$$
$$+\operatorname{E}\biggl[\operatorname*{min}\bigl\{f(X)+\alpha,2{\sqrt{(f(X)+\alpha)(1-f(X)+\alpha)}}\bigr\}\biggr]$$

Recall that E[min(*X, Y* )] ≤ min(E[X], E[Y ]) for random variables p X, Y . Furthermore, the function h(y) =
(y + α)(1 − y + α) is concave over the unit interval, so by Jensen's inequality

$$\mathbb{E}\bigg[\min\bigl\{f(X)+\alpha,2\sqrt{(f(X)+\alpha)(1-f(X)+\alpha)}\bigr\}\bigg]\leq1$$ $$\min\bigl\{\mathbb{E}[f(X)]+\alpha,2\sqrt{\mathbb{E}[(f(X)+\alpha)(1-f(X)+\alpha)]}\bigr\}.$$

Finally, observe that

$(f(X)+\alpha)(1-f(X)+\alpha)\leq f(X)(1-f(X))+2\alpha$.  
We apply Lemma 3.5 to bound E[f(X)(1 − f(X))].

## 3.3. Comparison To Previous Work

Consistency and robustness. It is well known that for λ ∈ (0, 1), any (1 + λ)-consistent algorithm for deterministic ski rental must be at least (1 + 1λ
)-robust (Wei
& Zhang, 2020; Angelopoulos et al., 2020; Gollapudi & Panigrahi, 2019). While Algorithm 1 is subject to this trade-off in the worst case, calibration provides sufficient information to hedge against adversarial inputs in expectation, leading to substantial improvements in average-case performance. Indeed, it can be seen from the bound in Theorem 3.3 that Algorithm 1 is 1-consistent and always satisfies E[CR(Ak∗
)] ≤ 1.8 when advice is calibrated (α = 0). An analysis similar to that of Theorem 15 in Anand et al. (2020)
shows that Algorithm 1 is g(α)-robust, where

$$g(\alpha)={\begin{cases}1+{\sqrt{\frac{1+\alpha}{\alpha}}}&{{\mathrm{if~}}\alpha<1/3}\\ 2&{{\mathrm{if~}}\alpha\geq1/3}\end{cases}}$$

is a decreasing function of α. This is because Algorithm 1 executes a worst-case 2-competitive strategy when α ≥ 1/3 and never buys skis before day b q α 1+α otherwise.

We note that one can run the same algorithm using an artificial upper bound α
′ > α on max calibration error to achieve an improved robustness level g(α
′). As seen from the bounds in Theorem 3.3 and Theorem 3.1, this adjustment will come at the cost of expected performance, highlighting the tradeoff between average and worst-case performance.

Uncertainty quantification. We are not the first to explore uncertainty quantified predictions for ski rental. Sun et al. (2024) take an orthogonal approach based on conformal prediction. Their method, Algorithm 2, assumes access to a probabilistic interval predictor PIPδ : *X → P*([0, 1]).

PIPδ outputs an interval [ℓ, u] = PIPδ(X) containing the true number of days skied Z ∈ [*ℓ, u*] with probability at least 1 − δ. Interval predictions are especially useful when the uncertainty δ and interval width u − ℓ are both small. However, as features become less informative, the width of prediction intervals must increase to maintain the same confidence level. This can result in intervals that are too wide to provide meaningful insight into the true number of days skied. Lemma 3.6 and Theorem 3.7 demonstrate that there are infinite families of distributions for which calibrated predictions are more informative than conformal predictions for ski rental. Lemma 3.6. For all a ∈ [0, 1/2], there exists an infinite family of input distributions for which Algorithm 2 defaults to a worst-case break-even strategy for all interval predictors PIPδ with uncertainty *δ < a*.

Proof sketch. The construction places mass 1 − a on some day z1 ≤
b 2and mass a on z2 ≥ 2b. Any PIPδ with *δ < a* Algorithm 2 (Sun et al., 2024) Optimal ski rental with conformal predictions input: interval prediction [ℓ, u] = PIPδ(X)
if ℓ ≤ *u < b* **then**
Rent for b days else if *b < ℓ* ≤ u **then**
Rent for b · min{pδ/1 − δ, 1} days else if ζ(*δ, ℓ*) ≥ 2 and δ +
u b 
≥ 2 **then**
Rent for b days else if ζ(*δ, ℓ*) ≤ δ +
u b then Rent for ℓ · min{p*bδ/ℓ*(1 − δ), 1} days else Rent for u days end if end if

$\left|\begin{array}{l}{\delta+\frac{(1-\delta)b}{\ell}+}\\ {1+\frac{b}{\ell}}\end{array}\right.$

ζ(*δ, ℓ*) := 

(δ +

(1−δ)b

ℓ + 2qδ(1−δ)b

ℓif δ ∈ [0,ℓ

ℓ+b

)

1 + bℓif δ ∈ [

ℓ

ℓ+b

, 1]

must output an interval [ℓ, u] containing both z1 and z2. Moreover, ζ(*δ, ℓ*) ≥ 2 and δ +
u b 
≥ 2 by construction.

Theorem 3.7. For all a ∈ [0, 1/2], all instantiations A of Algorithm 2 using PIPs with uncertainty δ < a, and all distributions from Lemma 3.6, if f is a predictor with meansquared error η and max calibration error α *satisfying* 2α + 2√η + 3α < a*, then* E[CR(Ak∗
)] < E[CR(A)].

Proof sketch. For the distributions in Lemma 3.6, the number of days skied is greater than b with probability a. Thus, the expected competitive ratio of the break-even strategy is E[CR(A)] = a · 2 + (1 − a)· 1 = 1 + a. The result follows from the bound on E[CR(Ak∗)] given in Theorem 3.1.

## 4. Online Job Scheduling

In this section, we explore the role of calibration in a model for *scheduling with predictions* first proposed by Cho et al. (2022) to direct human review of ML-flagged abnormalities in diagnostic radiology. Omitted proofs from this section can be found in Appendix B.

## 4.1. Setup

Problem. There is a single machine (lab tech) that needs to process n jobs (diagnostic images), each requiring one unit of processing time. Job i has some unknown priority yi ∈ {0, 1} that is independently high (yi = 1) with probability ρ and low (yi = 0) with probability 1 − ρ. Although job priorities are unknown a priori, the priority yiis revealed after completing some fixed fraction θ ∈ (0, 1) of job i. Upon learning yi, a scheduling algorithm can choose to complete job i, or switch to a new job and "store" job i for completion at a later time. The goal is to schedule the n jobs in a way that minimizes the weighted sum of completion times Pn i=1 Ci· ωyi where Ciis the completion time of job i, and ω1 > ω0 > 0 are costs associated with delaying a job of each priority for one unit of time. In hindsight, it is optimal to schedule jobs in decreasing order of priority.

ML predictions. Based on the assumption that the n jobs to be scheduled are iid, let X = X
n 0 be a set of job features, I = {0, 1}
n be the set of possible priorities, and D = Dn 0 be an unknown joint distribution over feature/priority pairs. The prediction task for this problem involves training a predictor f whose target is the true priority of each job T(⃗y) = ⃗y. This amounts to training a 1-dimensional predictor f : X0 → Z that acts on the n jobs independently:
f(X⃗ ) := (f(X⃗1)*, . . . , f*(X⃗n)).

Learning-augmented scheduling. Cho et al. (2022) introduce a threshold-based scheduling rule informed by probabilities pithat job i is high priority based on identifying features (Algorithm 3). Their algorithm switches between two extremes—a *preemptive* policy that starts a new job whenever the current job is revealed to be low priority, and a *non-preemptive* policy that completes any job once it is begun—based on the threshold parameter

$$\beta:={\frac{\theta}{1-\theta}}\cdot{\frac{\omega_{1}}{\omega_{1}-\omega_{0}}}.$$
.
In detail, jobs are opened in decreasing order of pi. Jobs with pi > β are processed preemptively, and the remaining jobs are processed non-preemptively. A learning-augmented algorithm A for job scheduling determines the probabilities pi from ML advice. Cho et al. (2022) assume access to a binary predictor fb : X0 → {0, 1} of job priority and study the case where pi = Pr[Y⃗i = 1 | fb(X⃗i)].

These probabilities can be computed using Bayes' rule, and because fb is binary, this procedure effectively assigns each job one of two probabilities. Although not explicitly discussed by Cho et al. (2022), this amounts to a basic form of post-hoc calibration. In contrast, our results extend to arbitrary calibrated predictors f : X0 → [0, 1]—a more general framework that calls for new mathematical techniquesallowing us to significantly improve upon their results. In this setting, A takes the predictions f(X⃗ ) = ⃗v as input and executes Algorithm 3 with probabilities pi = ⃗vi.

To quantify the optimality gap of A, Cho et al. (2022) note that compared to OPT, Algorithm 3 incurs (1) a cost of θω1 for each *inversion*, or pair of jobs whose true priorities yi are out of order, and (2) a cost of θω0 for each pair of low priority jobs encountered when acting preemptively. When acting non-preemptively, Algorithm 3 incurs (3) a cost of ω1 − ω0 for each inversion. Thus, for fixed predictions

| Algorithm 3 β-threshold rule input: Probabilities {pi} n i=1 that each job is high-priority Define n1 = |{i : pi > β}| Order probabilities p(1) ≥ · · · ≥ p(n) Run jobs j(1), . . . , j(n1) preemptively, in order Complete remaining jobs non-preemptively, in order   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

f(X⃗ ) = ⃗v and true job priorities ⃗y,

$$\begin{array}{l}{{{\cal A}(\vec{v},\vec{y})-\mathrm{OPT}(\vec{y})}}\\ {{=\theta\omega_{1}L(\vec{v},\vec{y})+\theta\omega_{0}M(\vec{v},\vec{y})+(\omega_{1}-\omega_{0})N(\vec{v},\vec{y}),}}\end{array}$$
$$({\mathfrak{I}})$$

where L(*⃗v, ⃗y*), M(⃗v, ⃗y), and N(*⃗v, ⃗y*) count occurrences of
(1), (2), and (3), respectively (see Table 2 for details).

## 4.2. Scheduling With Calibrated Predictions

Calibration and job sequencing. To build intuition for why finer-grained calibrated predictors sequence jobs more accurately, we begin by observing that Algorithm 3 orders jobs with the same probability pi randomly. Given a calibrated predictor f, consider the coarse calibrated predictor

$$f^{\prime}(x)={\begin{cases}\mathbb{E}[f(X)\mid f(X)>\beta]&{\mathrm{if~}}f(x)>\beta\\ \mathbb{E}[f(X)\mid f(X)\leq\beta]&{\mathrm{if~}}f(x)\leq\beta\end{cases}}$$

obtained by averaging the predictions of f above and below the threshold β. Whereas |R(f)| may be large, f
′is only capable of outputting |R(f
′)| = 2 values. As a result, when ordering jobs with features X1*, . . . , X*n according to predictions from f
′, all jobs with f(X) > β will be sequenced before jobs with f(X) ≤ β, but the ordering of jobs within these bins will be random. In contrast, predictions from f provide a more informative ordering of jobs (Figure 1).

Note, however, that f = f
′ when f has no variance in its predictions above or below the threshold β. We demonstrate in Theorem 4.3 that this intuition holds in general: improvements scale with the granularity of predictions.

6 5 4 3 2 1
× × ×
×
× × ×
×
f(X)
f
′(X)
4 5 6 2 3 1 0 β 1

| Table 2. Quantities of interest in learning-augmented scheduling for fixed predictions f(X⃗ ) = ⃗v and job priorities ⃗y.   |                                              |                  |
|--------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|------------------|
| Quantity                                                                                                                 | Description                                  | Relevant setting |
| n1 = |{i : ⃗vi > β}|                                                                                                      | Number of jobs likely to be high priority.   | -                |
| L(⃗v, ⃗y) = Xn1 Xn1 1{⃗y(i)=0∧⃗y(j)=1}                                                                                       | Number of inversions among jobs likely to be | Preemptive       |
| high priority.                                                                                                           |                                              |                  |
| i=1 j=i+1                                                                                                                |                                              |                  |
| M(⃗v, ⃗y) = Xn1 Xn1 1{⃗y(i)=0∧⃗y(j)=0}                                                                                       | Number of low-priority job pairs among jobs  | Preemptive       |
| i=1 j=i+1                                                                                                                | likely to be high priority.                  |                  |
| N(⃗v, ⃗y) = Xn Xn 1{⃗y(i)=0∧⃗y(j)=1} − L(⃗v, ⃗y)                                                                               | Number of inversions among job pairs where   | Non-preemptive   |
| at least one is likely to be low priority.                                                                               |                                              |                  |
| i=1 j=i+1                                                                                                                |                                              |                  |

Performance analysis. Building off of Equation (3), we bound the expected competitive ratio E[CR(A)] by bounding each of E[L(f(X⃗ ), Y⃗ )], E[M(f(X⃗ ), Y⃗ )], and E[N(f(X⃗ ), Y⃗ )]. The dependence on the ordering of predictions from f in these random counts means our analysis heavily involves functions of order statistics. For example, considering the shared summand of L(·) and N(·),

$$\mathbb{E}\left[1_{\{\vec{Y}_{(i)}=0\}}\cdot\mathbb{1}_{\{\vec{Y}_{(j)}=1\}}\mid f(\vec{X})\right]$$ $$=\left(\Pr[\vec{Y}_{(i)}=0\mid f(\vec{X}_{(i)})]\cdot\Pr[\vec{Y}_{(j)}=0\mid f(\vec{X}_{(j)})]\right)$$ $$=(1-f(\vec{X}_{(i)}))f(\vec{X}_{(j)})$$ $$=g(f(\vec{X}_{(i)}),f(\vec{X}_{(j)}))$$

for the function g(*x, y*) = (1 − x)y. Similarly, the analysis for the summand of M(·) yields g(f(X⃗(i)), f(X⃗(j))) for g(*x, y*) = (1 − x)(1 − y). Based on this, our high-level strategy is to relate "ordered" expectations of the form

$$\mathbb{E}\left[\sum_{i=1}^{n}\sum_{j=i+1}^{n}g\big(f({\vec{X}}_{(i)}),f({\vec{X}}_{(j)})\big)\right]$$

to their "unordered" counterparts

$$\mathbb{E}\left[\sum_{i=1}^{n}\sum_{j=i+1}^{n}g{\big(}f({\vec{X}}_{i}),f({\vec{X}}_{j}){\big)},\right]$$

which are simple to compute. Lemma 4.1 shows that the ordered and unordered expectations are, in fact, equivalent when the function g satisfies g(*x, y*) = g(*y, x*).

Lemma 4.1. Let X1, . . . , Xn be iid random variables with order statistics X(1) ≥ · · · ≥ X(n). For any symmetric function g : R × R → R,

$$\sum_{i=1}^{n}\sum_{j=i+1}^{n}g(X_{(i)},X_{(j)})=\sum_{i=1}^{n}\sum_{j=i+1}^{n}g(X_{i},X_{j}).$$

This result is sufficient to compute the expectation of M(·) exactly. For the other counts, the analysis is more technical as g(*x, y*) = (1 − x)y is not symmetric. Lemma 4.2 characterizes the relationship between the ordered and unordered expectations for the function g(*x, y*) = (1 − x)y.

Lemma 4.2. Let X1, . . . , Xn be iid samples from a distribution over the unit interval [0, 1] *with order statistics* X(1) ≥ · · · ≥ X(n)*. Then,*

$$\mathbb{E}\left[\sum_{i=1}^{n}\sum_{j=i+1}^{n}\left(1-X_{(i)}\right)\cdot X_{(j)}\right]\leq$$ $$\mathbb{E}\left[\sum_{i=1}^{n}\sum_{j=i+1}^{n}\left(1-X_{i}\right)\cdot X_{j}\right]-\binom{n}{2}\cdot\operatorname{Var}(X_{1}).$$

Proof sketch. By Lemma 4.1 with g(*x, y*) = xy,

$$\sum_{i=1}^{n}\sum_{j=i+1}^{n}X_{(i)}\cdot X_{(j)}=\sum_{i=1}^{n}\sum_{j=i+1}^{n}X_{i}\cdot X_{j}$$

can be removed from both sides. Then, we apply Lemma 4.1 with g(*x, y*) = min(x, y) to simplify the left-hand-side.

$$\begin{array}{r l}{{\sum_{i=1}^{n}\sum_{j=i+1}^{n}X_{(j)}=\sum_{i=1}^{n}\sum_{j=i+1}^{n}\operatorname*{min}\{X_{(i)},X_{(j)}\}}}\\ {{}}&{{=\sum_{i=1}^{n}\sum_{j=i+1}^{n}\operatorname*{min}\{X_{i},X_{j}\}.}}\end{array}$$

Finally, we show that E[X1 − min{X1, X2}] ≥ Var(X1).

Note that E[X1]−E[min{X1, X2}] = 12 E |X1−X2| since

$$X_{1}-\operatorname*{min}\{X_{1},X_{2}\}={\begin{cases}0&{\mathrm{if~}}X_{1}\leq X_{2}\\ |X_{1}-X_{2}|&{\mathrm{if~}}X_{1}>X_{2}.\end{cases}}$$
Finally, $\mathbb{E}\left|X_{1}-X_{2}\right|\geq\mathbb{E}\left|X_{1}-X_{2}\right|^{2}=2\text{Var}(X_{1})$.  
With careful conditioning to deal with random summation bounds, we apply Lemma 4.2 to bound the expectations of 7 L(·) and N(·), giving this section's main theorem. Of note, Theorem 4.3 says that the expected number of inversions of high and low priority jobs decreases with predictor granularity, measured by κ1 and κ2. For the method from Cho et al. (2022), κ1 = κ2 = 0 and the inequalities hold with equality.

Theorem 4.3. Let f *be calibrated, with* Pr[f(X) > β | Y = 0] = ϵ0, Pr[f(X) ≤ β | Y = 1] = ϵ1,

$\kappa_{1}=\Pr[f(X)>\beta]^{2}\cdot\Var(f(X)\mid f(X)>\beta)$, and $\kappa_{2}=\Pr[f(X)\leq\beta]^{2}\cdot\Var(f(X)\mid f(X)\leq\beta)$.  
Then

1. $\mathbb{E}[L(f(\vec{X}),\vec{Y})]\leq\binom{n}{2}\big{(}\rho(1-\rho)(1+\epsilon_{0})\epsilon_{1}-\kappa_{1}\big{)}$ 2. $\mathbb{E}[M(f(\vec{X}),\vec{Y})]=\binom{n}{2}(1-\rho)^{2}\epsilon_{0}^{2}$ 3. $\mathbb{E}[N(f(\vec{X}),\vec{Y})]\leq\binom{n}{2}\big{(}\rho(1-\rho)\epsilon_{0}(1-\epsilon_{1})-\kappa_{2}\big{)}$
Remark 4.4. A(f(X⃗ ), ·) − OPT(·) = 0 when ϵ0 = ϵ1 = 0, and A inherits the robustness guarantees of Cho et al. (2022) when ϵ0 and ϵ1 are large. An analogous result holds under the weaker assumption that f monotonically calibrated. That is, the empirical frequencies Pr[Y = 1 | f(X)] are non-decreasing in the prediction f(X). This property holds trivially for calibrated predictors, but zero calibration error is not required. In fact, many calibration approaches used in practice (e.g. Platt scaling (Platt et al., 1999) and isotonic regression (Zadrozny & Elkan, 2001)) produce a monotonically calibrated predictor with non-zero calibration error. See Appendix B for details.

## 5. Experiments

We now evaluate our algorithms on two real-world datasets, demonstrating the utility of using calibrated predictions. See Appendix C for additional details about our datasets and model training, as well a broader collection of results for different ML models and parameter settings.1

## 5.1. Ski Rental: Citi Bike Rentals

To model the rent-or-buy scenario in the ski rental problem, we use publicly available Citi Bike usage data.2. This dataset has been used for forecasting (Wang, 2016), system balancing (O'Mahony & Shmoys, 2015), and transportation policy (Lei & Ozbay, 2021), but to the best of our knowledge, this is its first use for ski rental. In this context, a Citi Bike user can choose one of two options: pay by ride duration (rent) or purchase a day pass (buy). If the user plans 1Code and data available here: https://github.com/
heyyjudes/algs-cali-pred 2Monthly usage data is publicly available at https://
citibikenyc.com/system-data.

ALG
Conformal Binary Breakeven Calibrated 4 6 8 10 12 14 16 Breakeven point (minutes)
1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9
[A
L

G/O
P

T]
to ride for longer than the break-even point of b minutes, it is cheaper to buy a day pass than to pay by trip duration.3 We use single-ride durations to approximate the rent vs. buy trade-off for a spectrum of break-even points b. The distribution over ride durations can be seen in Appendix C. We analyze the impact of advice from multiple predictor families, including XGBoost, logistic regression, and small multi-layer perceptrons (MLP). Each predictor has access to available ride features: start time, start location, user age, user gender, user membership, and approximate end station latitude. While these features are not extremely informative, most predictor families are able to achieve AUC
and accuracy above 0.8 for b > 6. Figure 2 summarizes the expected competitive ratios achieved by our method from Algorithm 1 (CALIBRATED) and baselines from previous work when given advice from a small neural network. Baselines include the worst-case optimal deterministic algorithm that rents for b minutes (Karlin et al., 1988) (BREAKEVEN), the black-box binary predictor ski-rental algorithm by Anand et al. (2020) (BINARY), and the PIP algorithm described in Algorithm 2 (Sun et al., 2024) (CONFORMAL). Though each algorithm is aided by predictors from the same family, the actual advice may differ. For example, CONFORMAL assumes access to a regressor that predicts ride duration directly. While performance is distribution-dependent, we see that our calibration-based approach often leads to the most cost-effective rent/buy policy in this scenario.

## 5.2. Scheduling: Sepsis Triage

We use a real-world dataset for sepsis prediction to validate our theory results for scheduling with calibrated predictions.

Sepsis is a life-threatening response to infection that typically appears after hospital admission (Singer et al., 2016).

3The day pass is designed to be more economical for multiple unlocks of a bike (e.g., b ≈ 66 minutes for 1 unlock). However, ride data is anonymous, so we cannot track daily usage.

1/ 0 ratio 0 1 2 3 4 5 6 7 8 Predictor Calibrated Binary 0.1 0.3 0.5
[A
L

G

O

P

T]
2 3 4 5 6 7 8 9
Many works have studied using machine learning to predict the onset of sepsis, as every hour of delayed treatment is associated with a 4-8% increase in mortality (Kumar et al., 2006; Reyna et al., 2020); existing works aim to better predict sepsis to treat high-priority patients earlier. Replicating results from Chicco & Jurman (2020) we train a binary predictor for sepsis onset using logistic regression on a dataset of 110,204 hospital admissions. The base predictor achieves an AUC of 0.86 using age, sex, and septic episodes as features. We then calibrate this predictor using both the naive method from Cho et al. (2022) (BINARY) and more nuanced histogram calibration (Zadrozny & Elkan, 2001) (CALIBRATED). Figure 3 shows the expected competitive ratio (normalized by the number of jobs n = 100) achieved by Algorithm 3 when provided advice from each of these predictors for varying delay costs ω1, ω0 and information barrier θ. We see that the more nuanced predictions consistently result in schedules with smaller delay costs.

## 6. Conclusion

In this paper, we demonstrated that calibration is a powerful tool for algorithms with predictions in settings where performance is measured over a distribution and probabilistic estimates of a binary target enable good decisions. In particular, calibration bridges the gap between traditional theoretical approaches—which treat all predictions as equally reliable—and modern ML methodologies that offer finegrained, instance-specific uncertainty quantification. We focused on the ski rental and online scheduling problems, developing online algorithms that exploit calibration guarantees to achieve strong average-case performance. For both problems, we highlighted settings where our algorithms outperform existing approaches and supported these findings with empirical evidence on real-world datasets. This work exposes a number of directions for future research.

For ski rental, deriving performance guarantees in terms of binary cross entropy and focusing on less rigid calibration measures (e.g. expected calibration error) offer to further close the gap between theory and practice. More broadly, we believe calibration-based approaches offer broad potential for designing online decision-making algorithms beyond these two case studies, particularly in scenarios that require balancing worst-case robustness with reliable per-instance predictions.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## Acknowledgments

This work was supported in part by NSF grant CCF- 2338226, the Simons Foundation Collaboration on the Theory of Algorithmic Fairness, and a National Defense Science & Engineering Graduate (NDSEG) fellowship. We thank Bailey Flanigan for stimulating early discussions that inspired us to pursue this research direction, and Ziv Scully for a technical insight in the proof of Lemma 4.2.

## References

Anand, K., Ge, R., and Panigrahi, D. Customizing ML predictions for online algorithms. In International Conference on Machine Learning (ICML), pp. 303–313, 2020.

Angelopoulos, S., Durr, C., Jin, S., Kamali, S., and Renault, ¨
M. Online computation with untrusted advice. In Innovations in Theoretical Computer Science (ITCS), 2020.

Angelopoulos, S., Bienkowski, M., Durr, C., and Simon, ¨
B. Contract scheduling with distributional and multiple advice. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2024.

Antoniadis, A., Gouleakis, T., Kleer, P., and Kolev, P. Secretary and online matching problems with machine learned advice. In Conference on Neural Information Processing Systems (NeurIPS), 2020.

Chicco, D. and Jurman, G. Survival prediction of patients with sepsis from age, sex, and septic episode number alone. *Scientific reports*, 10(1):17156, 2020.

Cho, W.-H., Henderson, S., and Shmoys, D. Scheduling with predictions. *arXiv preprint arXiv:2212.10433*, 2022.

DeGroot, M. H. and Fienberg, S. E. The comparison and evaluation of forecasters. Journal of the Royal Statistical Society. Series D (The Statistician), 32(1):12–22, 1983.

Diakonikolas, I., Kontonis, V., Tzamos, C., Vakilian, A., and Zarifis, N. Learning online algorithms with distributional advice. In International Conference on Machine Learning (ICML), pp. 2687–2696, 2021.

Lei, Y. and Ozbay, K. A robust analysis of the impacts of the stay-at-home policy on taxi and Citi Bike usage: A case study of Manhattan. *Transport Policy*, 110:487–498, 2021.

Dinitz, M., Im, S., Lavastida, T., Moseley, B., Niaparast, A., and Vassilvitskii, S. Binary search with distributional predictions. In Conference on Neural Information Processing Systems (NeurIPS), 2024.

Lin, H., Luo, T., and Woodruff, D. P. Learning augmented binary search trees. In International Conference on Machine Learning (ICML), 2022.

Lindermayr, A. and Megow, N. Permutation predictions for non-clairvoyant scheduling. In *ACM Symposium on* Parallelism in Algorithms and Architectures (SPAA), pp. 357–368, 2022.

Gollapudi, S. and Panigrahi, D. Online algorithms for rentor-buy with expert advice. In *International Conference* on Machine Learning (ICML), 2019.

Gopalan, P., Hu, L., Kim, M. P., Reingold, O., and Wieder, U. Loss minimization through the lens of outcome indistinguishability. In Innovations in Theoretical Computer Science (ITCS), 2023.

Lykouris, T. and Vassilvitskii, S. Competitive caching with machine learned advice. In International Conference on Machine Learning (ICML), 2018.

Mahdian, M., Nazerzadeh, H., and Saberi, A. Allocating online advertisement space with unreliable estimates. In ACM Conference on Economics and Computation (EC),
pp. 288–294, 2007.

Gupta, A., Panigrahi, D., Subercaseaux, B., and Sun, K.

Augmenting online algorithms with ϵ-accurate predictions. In Conference on Neural Information Processing Systems (NeurIPS), 2022.

Mitzenmacher, M. and Vassilvitskii, S. Algorithms with predictions. *Communications of the ACM*, 65(7):33–35, 2022.

Gupta, C. and Ramdas, A. Distribution-free calibration guarantees for histogram binning without sample splitting. In International Conference on Machine Learning (ICML), 2021.

Noarov, G., Ramalingam, R., Roth, A., and Xie, S. Highdimensional prediction for sequential decision making. In *Conference on Neural Information Processing Systems* (NeurIPS), 2023.

Im, S., Kumar, R., Qaem, M. M., and Purohit, M. Nonclairvoyant scheduling with predictions. In ACM Symposium on Parallelism in Algorithms and Architectures (SPAA), 2023.

O'Mahony, E. and Shmoys, D. Data analysis and optimization for (Citi) bike sharing. In *Proceedings of the AAAI* conference on artificial intelligence, volume 29, 2015.

Karlin, A. R., Manasse, M. S., Rudolph, L., and Sleator, D. D. Competitive snoopy caching. *Algorithmica*, 3(1–4): 79–119, 1988.

Platt, J. et al. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in large margin classifiers*, 10(3):61–74, 1999.

Karlin, A. R., Manasse, M. S., McGeoch, L. A., and Owicki, S. Competitive randomized algorithms for nonuniform problems. *Algorithmica*, 11(6):542–571, 1994.

Purohit, M., Svitkina, Z., and Kumar, R. Improving online algorithms via ML predictions. In Conference on Neural Information Processing Systems (NeurIPS), pp. 9661–
9670, 2018.

Karlin, A. R., Kenyon, C., and Randall, D. Dynamic TCP
acknowledgement and other stories about e/(e-1). In Proceedings of the Annual Symposium on Theory of Computing (STOC), pp. 502–509, 2001.

Reyna, M. A., Josef, C. S., Jeter, R., Shashikumar, S. P.,
Westover, M. B., Nemati, S., Clifford, G. D., and Sharma, A. Early prediction of sepsis from clinical data: the physionet/computing in cardiology challenge 2019. Critical care medicine, 48(2):210–217, 2020.

Khanafer, A., Kodialam, M., and Puttaswamy, K. P. The constrained ski-rental problem and its application to online cloud cost optimization. In Proceedings IEEE INFO- COM, pp. 1492–1500. IEEE, 2013.

Kumar, A., Roberts, D., Wood, K. E., Light, B., Parrillo, J. E., Sharma, S., Suppes, R., Feinstein, D., Zanotti, S., Taiberg, L., et al. Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock. *Critical* care medicine, 34(6):1589–1596, 2006.

Rohatgi, D. Near-optimal bounds for online caching with machine learned advice. In Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), 2020.

Shafer, G. and Vovk, V. A tutorial on conformal prediction.

Journal of Machine Learning Research, 9(3), 2008.

Singer, M., Deutschman, C. S., Seymour, C. W., Shankar-
Hari, M., Annane, D., Bauer, M., Bellomo, R., Bernard, G. R., Chiche, J.-D., Coopersmith, C. M., et al. The third international consensus definitions for sepsis and septic shock (sepsis-3). *Jama*, 315(8):801–810, 2016.

Sun, B., Huang, J., Christianson, N., Hajiesmaili, M.,
Wierman, A., and Boutaba, R. Online algorithms with uncertainty-quantified predictions. In International Conference on Machine Learning (ICML), 2024.

Vasilev, R. and D'yakonov, A. Calibration of neural networks. *arXiv preprint arXiv:2303.10761*, 2023.

Vovk, V., Gammerman, A., and Shafer, G. Algorithmic learning in a random world, volume 29. Springer, 2005.

Wang, W. *Forecasting Bike Rental Demand Using New York* Citi Bike Data. PhD thesis, Technological University Dublin, 2016.

Wei, A. and Zhang, F. Optimal robustness-consistency trade-offs for learning-augmented online algorithms. In Conference on Neural Information Processing Systems (NeurIPS), 2020.

Zadrozny, B. and Elkan, C. Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers. In International Conference on Machine Learning (ICML), pp. 609–616, 2001.

Zhao, S., Kim, M. P., Sahoo, R., Ma, T., and Ermon, S.

Calibrating predictions to decisions: A novel approach to multi-class calibration. In Conference on Neural Information Processing Systems (NeurIPS), 2021.

## A. Ski Rental Proofs

Lemma 3.2. Given a predictor f with max calibration error α, for all v ∈ R(f),
1. Pr[C | f(X) = v] ≤ v + α 2. Pr[C
c| f(X) = v] ≤ 1 − v + α

$$4.\ \operatorname{\mathbb{E}}[\operatorname{\mathsf{CR}}({\mathcal{A}}_{k})\mid B_{v},C]$$
3. E[CR(Ak) | Bv, C] ≤ 1 + k(v)
b
4. E[CR(Ak) | Bv, Cc] ≤ 1 + b·1{k(v)<b}
k(v).
Proof. Recall that Bv = {f(X) = v} is the event that f predicts v ∈ R(f), and C = {*Z > b*} is the event that the true number of days skied is at least b. Because f is a predictor of the indicator function 1C with max calibration error α,

$\Pr[C\mid B_v]=\Pr[Z>b\mid f(X)=v]=v-\alpha_v\leq v+\alpha_v$
and Pr[C
c| Bv] = Pr[Z ≤ b | f(X) = v] = 1 − v + αv ≤ 1 − v + α.

This establishes (1) and (2). In the remainder of the proof we will reference the costs from conditions (i)-(iv) in Table 3.

(3) E[CR(Ak) | Bv, C] ≤ 1 + k(v)
b. Under the event C (*Z > b*), one of conditions (iii) or (iv) must hold. The bound is tight when condition (iv) holds. Under condition (iii), it must be that Z ≤ k(v), so

$${\frac{\mathrm{ALG}(\mathcal{A}_{k},f(X),Z)}{\mathrm{OPT}(Z)}}={\frac{Z}{b}}\leq{\frac{k(v)}{b}}\leq1+{\frac{k(v)}{b}}.$$

(4) E[CR(Ak) | Bv, Cc] ≤ 1 + 1{k(v)<b} ·b k(v)
.

Under the event C
c(Z ≤ b), one of conditions (i) or (ii) hold. The bound is trivial under condition (i). Under condition (ii), because k(v) < Z and Z ≤ b,

$$\mathbf{\Phi}_{v},C^{c}]\leq1$$
$${\frac{\mathrm{ALG}(\mathcal{A}_{k},f(X),Z)}{\mathrm{OPT}(Z)}}={\frac{k(v)+b}{Z}}\leq{\frac{k(v)+b}{k(v)}}=1+\mathbf{1}_{\{k(v)<b\}}\cdot{\frac{b}{k(v)}}.$$

Theorem 3.3. Given a predictor f with max calibration error α, for any prediction v ∈ R(f), Algorithm 1 *achieves*

$$\mathbb{E}[\text{CR}(\mathcal{A}_{k_{*}})\mid f(X)=v]\leq$$ $$1+2\alpha+\min\bigl{\{}v+\alpha,2\sqrt{(v+\alpha)(1-v+\alpha)}\bigr{\}}.$$
$\square$
Proof. Let Bv = {f(X) = v} be the event that f predicts v ∈ R(f), and let C = {*Z > b*} be the event that the true number of days skied is at least b. By the law of total expectation and Lemma 3.2,

$$\mathbb{E}[\text{CR}(\mathcal{A}_{k})\mid B_{v}]=\Pr[C\mid B_{v}]\cdot\mathbb{E}[\text{CR}(\mathcal{A}_{k})\mid C,B_{v}]+\Pr[C^{c}\mid B_{v}]\cdot\mathbb{E}[\text{CR}(\mathcal{A}_{k})\mid C^{c},B_{v}]$$ $$\leq(v+\alpha)\cdot\left(1+\frac{k(v)}{b}\right)+(1-v+\alpha)\cdot\left(1+\mathbb{1}_{\{k(v)<b\}}\cdot\frac{b}{k(v)}\right)$$ $$=1+2\alpha+\frac{(v+\alpha)k(v)}{b}+\mathbb{1}_{\{k(v)<b\}}\cdot\frac{(1-v+\alpha)b}{k(v)}.$$

Finding the number of days to rent skis that minimizes this upper bound on competitive ratio amounts to solving two convex optimization problems - one for the case k(v) < b, and a second for k(v) ≥ b - then taking the minimizing solution.

* Minimize $1+2\alpha+\frac{(v+\alpha)\ell}{b}+\frac{(1-v+\alpha)b}{\ell}$ s.t. $0\leq\ell\leq b$
ℓ(b) Minimize 1 + 2α +
s.t. 0 ≤ ℓ ≤ b s.t. ℓ ≥ b
(b) Minimize $1+2\alpha+\frac{(v+\alpha)\ell}{b}$ s.t. $\ell\geq b$
Note first that (b) has optimal solution ℓ∗ = b. The Lagrangian of (a) is

$${\mathcal{L}}(\ell,\lambda_{1},\lambda_{2})=1+2\alpha+{\frac{(v+\alpha)\ell}{b}}+{\frac{(1-v+\alpha)b}{\ell}}+\lambda_{1}(\ell-b)-\lambda_{2}\ell$$

with KKT optimality conditions

$$\frac{v+\alpha}{b}-\frac{(1-v+\alpha)b}{\ell^{2}}+\lambda_{1}-\lambda_{2}=0$$ $$\ell\leq b$$ $$-\ell\leq0$$ $$\lambda_{1},\lambda_{2}\geq0$$ $$\lambda_{1}(\ell-b)=0$$ $$\lambda_{2}(-\ell)=0.$$

We'll proceed by finding solutions to this system of equations via case analysis.

1. λ2 ̸= 0. Then, ℓ = 0 and λ1 = 0 by complementary slackness. But at least one of the stationarity or dual feasibility constraints are violated, since

$$0>\frac{v+\alpha}{b}-\frac{(1-v+\alpha)b}{\ell^{2}}=\lambda_{2}.$$

2. λ2 = 0 and λ1 ̸= 0. Then, ℓ = b by complementary slackness. Stationarity and dual feasibility are satisfied only when 0 ≤ v ≤ 0.5, since in this casev + α

$${\frac{v+\alpha}{b}}-{\frac{1-v+\alpha}{b}}=-\lambda_{1}\leq0.$$

3. λ2 = 0 and λ1 = 0. Then, the first constraint gives that

$$\ell^{2}={\frac{(1-v+\alpha)b^{2}}{v+\alpha}}.$$

Recall that 0 ≤ ℓ ≤ b, so this constraint is only satisfied when 0.5 ≤ v ≤ 1 and ℓ = b q1−v+α v+α
.

Because ℓ∗ = b is the optimal solution to both (a) and (b) when 0 ≤ v ≤ 0.5, it must be the case that k∗(v) = b if 0 ≤ v ≤ 0.5. When 0.5 < v ≤ 1, the optimal solution to (a) is ℓ∗ = b q1−v+α v+αand the optimal solution to (b) is ℓ∗ = b.

The value of the former is 1 + 2α + 2p(v + α)(1 − v + α), while the value of the latter is 1 + 2α + v + α. Taking the argmin yields

$$k_{*}(v)={\begin{cases}b&{\mathrm{if}}\;0\leq v\leq{\frac{4+3\alpha}{5}}\\ b{\sqrt{\frac{1-v+\alpha}{v+\alpha}}}&{\mathrm{if}}\;{\frac{4+3\alpha}{5}}<v\leq1,\end{cases}}$$

which is exactly Algorithm 1 and achieves a competitive ratio of

$\mathbb{E}[\text{CR}(\mathcal{A}_{k_{*}})\mid f(X)=v]\leq1+2\alpha+\min\bigl{\{}v+\alpha,2\sqrt{(v+\alpha)(1-v+\alpha)}\bigr{\}}$.  
Theorem 3.4. For all renting strategies k : [0, 1] → R+, predictions v ∈ [0, 1] and ϵ > 0*, there exists a distribution* Dϵv and a calibrated predictor f *such that*

$$\mathbb{E}[\mathbf{CR}({\mathcal{A}}_{k})\mid f(X)=v]\geq1+\operatorname*{min}\left\{v,2{\sqrt{v(1-v)}}\right\}-\epsilon.$$

| Table 3. Objective values for fixed prediction f(X) = v, z days skied, and renting for k(v) days. CONDITION OPT(z) ALG(Ak, v, z) (i) z ≤ min{k(v), b} z z (ii) k(v) < z ≤ b z k(v) + b (iii) b < z ≤ k(v) b z (iv) z > max{k(v), b} b k(v) + b   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Proof. Let v ∈ [0, 1] and ϵ > 0. The calibrated predictor f will deterministically output v, while the distribution Dϵv will depend on whether algorithm Ak buys before or after day b.

Case 1: k(v) < b. Define a distribution Dϵv where in a v fraction of the data the true number of days skied is z = b + ϵ
′, and in a 1 − v fraction the number of days skied is z = k(v) + ϵ
′, where ϵ
′is sufficiently small that

$$k(v)+\epsilon^{\prime}\leq b\qquad\mathrm{and}\qquad2{\sqrt{v(1-v)\left(1-{\frac{\epsilon^{\prime}}{b}}\right)}}-{\frac{2v\epsilon^{\prime}}{b+\epsilon^{\prime}}}\geq2{\sqrt{v(1-v)}}-\epsilon.$$

By construction, condition (ii) from Table 3 is satisfied when k(v) *< b < z* = b + ϵ
′ with

$$\mathrm{ALG}({\mathcal{A}}_{k},v,z)/\mathrm{OPT}(z)={\frac{k(v)+b}{b+\epsilon^{\prime}}}=1+{\frac{k(v)-\epsilon^{\prime}}{b+\epsilon^{\prime}}}.$$

Similarly, condition (ii) holds when k(v) < z = k(v) + ϵ

en $k(v)<2=k(v)+\epsilon^{\prime}\leq b$ with 
$$\mathrm{ALG}({\mathcal{A}}_{k},v,z)/\mathrm{OPT}(z)={\frac{k(v)+b}{k(v)+\epsilon^{\prime}}}=1+{\frac{b-\epsilon^{\prime}}{k(v)+\epsilon^{\prime}}}.$$
By the law of total expectation,

$$\mathbb{E}[CR(\mathcal{A}_{k})]=v\cdot\left(1+\frac{k(v)-\epsilon^{\prime}}{b+\epsilon^{\prime}}\right)+(1-v)\cdot\left(1+\frac{b-\epsilon^{\prime}}{k(v)+\epsilon^{\prime}}\right)$$ $$\geq\min_{\ell\geq0}\left\{v\cdot\left(1+\frac{\ell-\epsilon^{\prime}}{b+\epsilon^{\prime}}\right)+(1-v)\cdot\left(1+\frac{b-\epsilon^{\prime}}{\ell+\epsilon^{\prime}}\right)\right\}.$$

Some basic calculus yields ℓ∗ =
q1−v v(b − ϵ
′)(b + ϵ
′) − ϵ
′, and evaluating the lower bound at ℓ
∗ gives

$$\mathbb{E}[CR(\mathcal{A}_{k})]\geq1-\frac{2v\epsilon^{\prime}}{b+\epsilon^{\prime}}+2\sqrt{v(1-v)\left(1-\frac{\epsilon^{\prime}}{b}\right)}$$ $$\geq1+2\sqrt{v(1-v)}-\epsilon.$$

Case 2: k(v) ≥ b.

Define a distribution Dϵv where in a v fraction of the data the true number of days skied is z = k(v) + ϵ, and in a 1 − v fraction the number of days skied is z = b − ϵ. Condition (iv) is satisfied when b ≤ k(v) < z = k(v) + ϵ with ALG(Ak, v, z)/OPT(z) = 1+k(v)
b
. Condition (i) is satisfied when z = b−*ϵ < b* ≤ k(v) with ALG(Ak*, v, z*)/OPT(z) = 1.

By the law of total expectation,

$$\mathbb{E}[CR(\mathcal{A}_{k})]=v\cdot\left(1+\frac{k(v)}{b}\right)+(1-v)\cdot1$$ $$\geq v\cdot2+(1-v)\cdot1$$ $$=1+v.$$

In both cases, f is calibrated with respect to Dϵvsince Pr[*Z > b* | f(X) = v] = v. Moreover, because the cases are exhaustive, at least one of the corresponding lower bounds must hold. It follows immediately that

$$\mathbb{E}[C R({\mathcal{A}}_{k})\mid f(X)=v]\geq1+\operatorname*{min}\left\{v,2{\sqrt{v(1-v)}}\right\}-\epsilon.$$

Lemma 3.5. If f : X → [0, 1] has mean-squared error η and max calibration error α*, then*

$$\lceil\!\!\!\perp\!\!\!\perp$$
$$\mathbb{E}[f(X)(1-f(X))]\leq\eta+c$$

Proof. We have from the law of total expectation that

$$\eta=\min_{(X,\mathbb{E})\sim D}\left[\left(\mathbb{1}_{\{Z>b\}}-f(X)\right)^{2}\right]$$ $$=\sum_{v\in R(I)}\mathbb{E}\left[\left(\mathbb{1}_{\{Z>b\}}-v\right)^{2}\big{|}\ f(X)=v\right]\cdot\Pr\left[f(X)=v\right]$$ $$=\sum_{v\in R(I)}\left(\mathbb{E}\left[\mathbb{1}_{\{Z>b\}}\mid f(X)=v\right]-2v\,\mathbb{E}\left[\mathbb{1}_{\{Z>b\}}\mid f(X)=v\right]+v^{2}\right)\cdot\Pr\left[f(X)=v\right].$$

Applying the definition of the local calibration error αv,

$$\eta=\sum_{v\in R(f)}\left(\mathbb{E}\left[\,\mathbb{1}\,_{\{Z>b\}}\mid f(X)=v\right]-2v\,\mathbb{E}\left[\mathbb{1}\,_{\{Z>b\}}\mid f(X)=v\right]+v^{2}\right)\cdot\Pr\left[f(X)=v\right]$$ $$=\sum_{v\in R(f)}\left((v-\alpha_{v})-2v(v-\alpha_{v})+v^{2}\right)\cdot\Pr\left[f(X)=v\right]$$ $$=\sum_{v\in R(f)}\left(v(1-v)+(2v-1)\alpha_{v}\right)\cdot\Pr\left[f(X)=v\right]$$ $$=\mathbb{E}[f(X)(1-f(X))]+\sum_{v\in R(f)}\left(2v-1\right)\alpha_{v}\cdot\Pr\left[f(X)=v\right].$$
$\square$
The observation that (2v − 1)αv *≥ −|*αv| gives the result. Theorem 3.1. Given a predictor f with mean-squared error η and max calibration error α, Algorithm 1 *achieves* E[CR(Ak∗
)] ≤ 1 + 2α + min E[f(X)] + α, 2
√η + 3α	.

Proof. This result follows from Theorem 3.3, Lemma 3.5, and an application of Jensen's inequality. To begin,

$$\mathbb{E}[\text{CR}(\mathcal{A}_{k_{*}})]=\mathbb{E}\left[\mathbb{E}[\text{CR}(\mathcal{A}_{k_{*}})\mid f(X)]\right]$$ (Tower property) $$\leq\mathbb{E}\left[1+2\alpha+\min\left\{f(X)+\alpha,2\sqrt{(f(X)+\alpha)(1-f(X)+\alpha)}\right\}\right]$$ (Theorem 3.3) $$\leq1+2\alpha+\min\left\{\mathbb{E}\left[f(X)\right]+\alpha,2\,\mathbb{E}\left[\sqrt{(f(X)+\alpha)(1-f(X)+\alpha)}\right]\right\},$$

with the final line following from the fact that E[min(*X, Y* )] ≤ min(E[X], E[Y ]) for random variables *X, Y* . Next, we argue from basic composition rules that the function g(y) = p(y + α)(1 − y + α) is concave for y ∈ [0, 1]. The concavity of g over its domain follows from the facts that (1) the √· function is concave and increasing in its argument and (2)
(y + α)(1 − y + α) is concave. Moreover, g(y) is well-defined for all y ∈ [0, 1]. With concavity established, an application of Jensen's inequality yields

$$\mathbb{E}[\text{CR}(\mathcal{A}_{k},)]\leq1+2\alpha+\min\left\{\mathbb{E}\left[f(X)\right]+\alpha,2\sqrt{\mathbb{E}\left[(f(X)+\alpha)(1-f(X)+\alpha)\right]}\right\}.$$

To finish the proof, we will bound the term within the square root using Lemma 3.5. Notice that

$$(f(X)+\alpha)(1-f(X)+\alpha)=f(X)(1-f(X))+\alpha+\alpha^{2}$$ $$\leq f(X)(1-f(X))+2\alpha.$$
$$({\mathrm{Lemma~}}3.5)$$

Finally,

$$\mathbb{E}[\text{CR}(\mathcal{A}_{k},\cdot)]\leq1+2\alpha+\min\left\{\mathbb{E}\left[f(X)\right]+\alpha,2\sqrt{\mathbb{E}\left[f(X)(1-f(X))\right]+2\alpha}\right\}\,,$$ $$\leq1+2\alpha+\min\left\{\mathbb{E}\left[f(X)\right]+\alpha,2\sqrt{\eta+3\alpha}\right\}.$$
o. (Lemma 3.5)
Lemma 3.6. For all a ∈ [0, 1/2], there exists an infinite family of input distributions for which Algorithm 2 defaults to a worst-case break-even strategy for all interval predictors PIPδ with uncertainty *δ < a*.

Proof. Let a ∈ [0, 1/2] and consider a distribution that, for each unique feature vector x ∈ X , has a true number of days skied that is either z1 ≤
b 2 with probability 1 − a or z2 ≥ 2b with probability a. By construction, any interval prediction PIPδ(X) = [*ℓ, u*] with δ < min{a, 1 − a} = a must satisfy that ℓ ≤ z1 and u ≥ z2. This means b ∈ [*ℓ, u*], so Algorithm 2 makes a determination of which day to buy based on the relative values of ζ(*δ, ℓ*), δ +
u b
, and 2. In particular, the algorithm follows the break-even strategy of buying on day b when ζ(*δ, ℓ*) ≥ 2 and δ +
u b ≥ 2.

It is clear that δ +
u b ≥
u b ≥
z2 b ≥ 2. Next, recall the definition of ζ(*δ, ℓ*).

$$\zeta(\delta,\ell)={\begin{cases}\delta+(1-\delta){\frac{b}{\ell}}+2{\sqrt{\delta(1-\delta)b/\ell}}&{{\mathrm{if~}}\delta\in[0,{\frac{\ell}{b+\ell}})}\\ 1+{\frac{b}{\ell}}&{{\mathrm{if~}}\delta\in[{\frac{\ell}{b+\ell}},1]}\end{cases}}$$

When δ ≥ℓ b+ℓ
, we see that ζ(*δ, ℓ*) = 1 + bℓ ≥ 3. To handle the case where δ < ℓ b+ℓ
, we will show that

$$f(\delta,x)=\delta+(1-\epsilon)$$
$\mathbf{a}$
$$1-\delta)$$
$\left(T\right)$. 
f(δ, x) = δ + (1 − δ)x + 2pδ(1 − δ)x ≥ 2
for all x ≥ 2 and δ ∈ [0, 1/2]. Plugging in x =
b ℓ ≥ 2 and noting that δ < ℓ b+ℓ ≤ 0.5 implies the desired bound. Toward that end, notice that f(*δ, x*) is increasing in x, and so for all x ≥ 2 we have that

$$f(\delta,x)\geq f(\delta,2)=2-\delta+2\sqrt{2\delta(1-\delta)}.$$

All that is left is to show that 2p2δ(1 − δ) ≥ δ. This is straightforward: for δ ∈ [0, 1/2],

$$2{\sqrt{2(1-\delta)}}\geq{\sqrt{1-\delta}}\geq{\sqrt{\delta}},$$

and multiplying through by 
√δ gives the desired inequality. In summary, we've shown that b ∈ [*ℓ, u*], ζ(δ, ℓ) ≥ 2, and δ +
u b ≥ 2 for the family of distributions described above. For this particular case, Algorithm 2 rents for b days.

Theorem 3.7. For all a ∈ [0, 1/2], all instantiations A of Algorithm 2 using PIPs with uncertainty δ < a, and all distributions from Lemma 3.6, if f is a predictor with mean-squared error η and max calibration error α *satisfying* 2α + 2√η + 3α < a*, then* E[CR(Ak∗)] < E[CR(A)].

Proof. Let a ∈ [0, 1/2] and consider any distribution from the infinite family given in Lemma 3.6. In particular, in any of these distributions, the number of days skied is greater than b with probability a. Therefore, the expected competitive ratio of the break-even strategy that rents for b days before buying is

$$\mathbb{E}[\mathbf{CR}({\mathcal{A}})]=a\cdot2+(1-a)\cdot1=1+a.$$

The result follows from the bound on E[CR(Ak∗
)] from Theorem 3.1.

## B. Scheduling Proofs

Lemma 4.1. Let X1, . . . , Xn *be iid random variables with order statistics* X(1) ≥ · · · ≥ X(n)*. For any symmetric function* g : R × R → R, 

$$\sum_{i=1}^{n}\sum_{j=i+1}^{n}g(X_{(i)},X_{(j)})=\sum_{i=1}^{n}\sum_{j=i+1}^{n}g(X_{i},X_{j}).$$

Proof. Beginning with the facts that

$$\sum_{i=1}^{n}\sum_{j=1}^{n}g(X_{(i)},X_{(j)})=\sum_{i=1}^{n}\sum_{j=1}^{n}g(X_{i},X_{j})\quad\text{and}\quad\sum_{i=1}^{n}g(X_{(i)},X_{(i)})=\sum_{i=1}^{n}g(X_{i},X_{i}),$$

it follows from the symmetry of g that

micity of $g$ that $$\sum_{i=1}^n\sum_{j=i+1}^n g(X_{(i)},X_{(j)})=\frac{1}{2}\left(\sum_{i=1}^n\sum_{j=1}^n g(X_{(i)},X_{(j)})-\sum_{i=1}^n g(X_{(i)},X_{(i)})\right)$$ $$=\frac{1}{2}\left(\sum_{i=1}^n\sum_{j=1}^n g(X_i,X_j)-\sum_{i=1}^n g(X_i,X_i)\right)$$ $$=\sum_{i=1}^n\sum_{j=i+1}^n g(X_i,X_j).$$
Lemma 4.2. Let X1, . . . , Xn be iid samples from a distribution over the unit interval [0, 1] *with order statistics* X(1) ≥ · · · ≥ X(n)*. Then,*

$$\mathbb{E}\left[\sum_{i=1}^{n}\sum_{j=i+1}^{n}\left(1-X_{(i)}\right)\cdot X_{(j)}\right]\leq$$ $$\mathbb{E}\left[\sum_{i=1}^{n}\sum_{j=i+1}^{n}\left(1-X_{i}\right)\cdot X_{j}\right]-\binom{n}{2}\cdot\operatorname{Var}(X_{1}).$$

Proof. We'll begin by removing a shared term from both sides of the inequality. Notice that

$$\sum_{i=1}^{n}\sum_{j=i+1}^{n}X_{(i)}X_{(j)}=\sum_{i=1}^{n}\sum_{j=i+1}^{n}X_{i}X_{j}$$

by Lemma 4.1 with g(*x, y*) = xy. So, it is sufficient to show that

$$\mathbb{E}\left[\sum_{i=1}^{n}\sum_{j=i+1}^{n}X_{j}\right]-\mathbb{E}\left[\sum_{i=1}^{n}\sum_{j=i+1}^{n}X_{(j)}\right]\geq{\binom{n}{2}}\operatorname{Var}(X_{1}).$$

By linearity of expectation, the first term on the left-hand side is equal to n2 E[X1]. The random variables in the second term are not identically distributed, however, so a different approach is required. We will use a trick to express the sum in terms of the symmetric function g(*x, y*) = min(x, y), which allows us to remove the dependency on order statistics using Lemma 4.1.

$$\sum_{i=1}^{n}\sum_{j=i+1}^{n}X_{(j)}=\sum_{i=1}^{n}\sum_{j=i+1}^{n}\min\{X_{(i)},X_{(j)}\}(X_{(i)}\geq X_{(j)}\text{since}i\leq j)$$ $$=\sum_{i=1}^{n}\sum_{j=i+1}^{n}\min\{X_{i},X_{j}\}.$$ (Lemma 4.1 with $$g(x,y)=\min\{x,y\}$$ )
Thus, the second term on the RHS is equal to n 2 E[min{X1, X2}]. All that is left is to show that E[X1] − E[min{X1, X2}] ≥ Var(X1).

Toward that end, we can write

$$X_{1}-\operatorname*{min}\{X_{1},X_{2}\}={\begin{cases}0&{\mathrm{if~}}X_{1}\leq X_{2}\\ |X_{1}-X_{2}|&{\mathrm{if~}}X_{1}>X_{2},\end{cases}}$$
so $\mathbb{E}[X_{1}]-\mathbb{E}[\min\{X_{1},X_{2}\}]=\frac{1}{2}\,\mathbb{E}\,|X_{1}-X_{2}|$. Finally, using the fact that $|X_{1}-X_{2}|\in[0,1]$ are iid, we have 
$\frac{1}{2}\,\mathbb{E}\left|X_{1}-X_{2}\right|\geq\frac{1}{2}\,\mathbb{E}[(X_{1}-X_{2})^{2}]$  $$=\frac{1}{2}\,\mathbb{E}\left[X_{1}^{2}-2X_{1}X_{2}+X_{2}^{2}\right]$$ $$=\frac{1}{2}\cdot2\left(\mathbb{E}[X_{1}^{2}]-\mathbb{E}[X_{1}]^{2}\right)$$ $$=\text{Var}(X_{1}).$$
 Let that $ |X_1-X_2|\in[0,1]$ are iid, w. 
Theorem 4.3. Let f *be calibrated, with* Pr[f(X) > β | Y = 0] = ϵ0, Pr[f(X) ≤ β | Y = 1] = ϵ1,

$\kappa_{1}=\Pr[f(X)>\beta]^{2}\cdot\Var(f(X)\mid f(X)>\beta)$, and $\kappa_{2}=\Pr[f(X)\leq\beta]^{2}\cdot\Var(f(X)\mid f(X)\leq\beta)$.  
Then

1. $\mathbb{E}[L(f(\vec{X}),\vec{Y})]\leq\binom{n}{2}\big{(}\rho(1-\rho)(1+\epsilon_{0})\epsilon_{1}-\kappa_{1}\big{)}$ 2. $\mathbb{E}[M(f(\vec{X}),\vec{Y})]=\binom{n}{2}(1-\rho)^{2}\epsilon_{0}^{2}$ 3. $\mathbb{E}[N(f(\vec{X}),\vec{Y})]\leq\binom{n}{2}\big{(}\rho(1-\rho)\epsilon_{0}(1-\epsilon_{1})-\kappa_{2}\big{)}$
Proof. We relax the calibration assumption and only assume that f is monotonically calibrated, a weaker condition that the empirical frequencies Z := Pr[Y = 1 | f(X)] are non-decreasing in the prediction f(X). Given n jobs to schedule with features X⃗ = (X1*, . . . , X*n) and the predictions f(X⃗ ) = (f(X1)*, . . . , f*(Xn)), let n1 = |{i : f(Xi) > β}| be a random variable that counts the number of samples from f with prediction larger than β, and define random variables Zi = Pr[Yi = 1 | f(Xi)] which give empirical frequencies. We'll begin by computing expectations conditioned on n1 before taking an outer expectation.

E[L(f(X⃗ ), Y⃗ ) | n1] = E E[L(f(X⃗ ), Y⃗ ) | f(X⃗ )] | n1 = E  E  Xn1 i=1 Xn1 j=i+1 1{Y⃗(i)=0} · 1{Y⃗(j)=1} | f(X⃗ )   | n1  = E  j=i+1 Pr[Y⃗(i) = 0 | f(X⃗(i))] · Pr[Y⃗(j) = 1 | f(X⃗(j))] | n1  Xn1 i=1 Xn1 = E  j=i+1 (1 − Z(i))Z(j)| n1  Xn1 i=1 Xn1  .
(Tower property)
 (Definition of X)
$$({\mathrm{Tower~property}})$$
$$({\mathrm{Definition~of~}}X)$$
$$({\mathrm{Independence}})$$
Performing the same computation for counts M(·) and N(·) yields

$$\mathbb{E}[M(f({\vec{X}}),{\vec{Y}})\mid n_{1}]=\mathbb{E}\left[\sum_{i=1}^{n_{1}}\sum_{j=i+1}^{n_{1}}(1-Z_{(i)})\cdot(1-Z_{(j)})\mid n_{1}\right].$$

18 and

$$\mathbb{E}[N(f(\vec{X}),\vec{Y})\mid n_{1}]=\mathbb{E}\left[\sum_{i=1}^{n_{1}}\sum_{j=n_{1}+1}^{n}\left(1-Z_{(i)}\right)\cdot Z_{(j)}\mid n_{1}\right]+\mathbb{E}\left[\sum_{i=n_{1}+1}^{n}\sum_{j=i+1}^{n}\left(1-Z_{(i)}\right)\cdot Z_{(j)}\mid n_{1}\right].$$

At this point, we can compute the conditional expectation of M(·) directly. By Lemma 4.1 with g(*x, y*) = (1 − x)(1 − y),

$$\mathbb{E}\left[\sum_{i=1}^{n_{1}}\sum_{j=i+1}^{n_{1}}\left(1-Z_{(i)}\right)\cdot\left(1-Z_{(j)}\right)\mid n_{1}\right]$$ $$=\mathbb{E}\left[\sum_{i=1}^{n_{1}}\sum_{j=i+1}^{n_{1}}\left(1-Z_{i}\right)\cdot\left(1-Z_{j}\right)\mid n_{1}\right]$$ $$=\binom{n_{1}}{2}\cdot\mathbb{E}\left[\Pr[Y=0\mid f(X)]\mid f(X)>\beta\right]^{2}$$ $$=\binom{n_{1}}{2}\cdot\Pr[Y=0\mid f(X)>\beta]^{2}$$ $$=\binom{n_{1}}{2}\cdot\frac{\epsilon_{0}^{2}(1-\rho)^{2}}{\Pr[f(X)>\beta]^{2}}.$$

$$({\mathrm{Lemma}}\,4.1)$$
 (Independence)  $$\text{(Tower property)}$$  $$\text{(Bayes'rule)}$$
 (Lemma 4.1)
2(Tower property)
The same technique cannot be used to evaluate the expectations of L(·) and N(·) because the function g(*x, y*) = (1 − x)y is not symmetric. Instead, we will provide upper bounds on the conditional expectations using Lemma 4.2, then evaluate the unordered results as before. For the conditional expectation of L(·), we have

$$\mathbb{E}\bigg{[}\sum_{i=1}^{n_{1}}\sum_{j=i+1}^{n_{1}}\left(1-Z_{(i)}\right)\cdot Z_{(j)}\mid n_{1}\bigg{]}$$ $$\leq\mathbb{E}\left[\sum_{i=1}^{n_{1}}\sum_{j=i+1}^{n_{1}}\left(1-Z_{i}\right)\cdot Z_{j}\mid n_{1}\right]-\binom{n_{1}}{2}\operatorname{Var}(Z\mid f(X)>\beta)$$ (Lemma 4.2) $$=\binom{n_{1}}{2}\cdot\left(\Pr[Y=0\mid f(X)>\beta]\cdot\Pr[Y=1\mid f(X)>\beta]-\operatorname{Var}(Z\mid f(X)>\beta)\right)$$ $$=\binom{n_{1}}{2}\cdot\left(\frac{\rho(1-\rho)(1-\epsilon_{1})\epsilon_{0}}{\Pr[f(X)>\beta]^{2}}-\operatorname{Var}(Z\mid f(X)>\beta)\right).$$

Similarly for the conditional expectation of the second term of N(·),

$$\mathbb{E}\Bigg{[}\sum_{i=n_{1}+1}^{n}\sum_{j=i+1}^{n}\left(1-Z_{(i)}\right)\cdot Z_{(j)}\mid n_{1}\Bigg{]}$$ $$\leq\mathbb{E}\Bigg{[}\sum_{i=n_{1}+1}^{n}\sum_{j=i+1}^{n}\left(1-Z_{i}\right)\cdot Z_{j}\mid n_{1}\Bigg{]}-\binom{n-n_{1}}{2}\operatorname{Var}(Z\mid f(X)\leq\beta)$$ (Lemma 4.2) $$=\binom{n-n_{1}}{2}\cdot\left(\Pr[Y=0\mid f(X)\leq\beta]\cdot\Pr[Y=1\mid f(X)\leq\beta]-\operatorname{Var}(Z\mid f(X)\leq\beta)\right)$$ $$=\binom{n-n_{1}}{2}\cdot\left(\frac{\rho(1-\rho)(1-\epsilon_{0})\epsilon_{1}}{\Pr[f(X)\leq\beta]^{2}}-\operatorname{Var}(Z\mid f(X)\leq\beta)\right).$$

For the first term of N(·), we simply apply the rearrangement inequality in lieu of Lemma 4.2 for unordering. Note that the sum has the form Pi ai· bi, where ai = (1 − Z(i)) and bi =Pn j=i+1 Z(j). The sequence {ai}
n i=1 is non-decreasing as a result of the monotonic calibration of f, and {bi}
n i=1 is non-increasing. Thus,

$$\mathbb{E}\left[\sum_{i=1}^{n_{1}}\sum_{j=n_{1}+1}^{n}\left(1-Z_{(i)}\right)\cdot Z_{(j)}\mid n_{1}\right]\leq\mathbb{E}\left[\sum_{i=1}^{n_{1}}\sum_{j=n_{1}+1}^{n_{1}}\left(1-Z_{i}\right)\cdot Z_{j}\mid n_{1}\right]$$ $$=n_{1}(n_{1}-n_{1})\cdot\Pr[Y=0\mid f(X)>\beta]\cdot\Pr[Y=1\mid f(X)\leq\beta]$$ $$=n_{1}(n_{1}-n_{1})\cdot\frac{p(1-\rho)\rho_{1}}{\Pr[f(X)>\beta]\cdot\Pr[f(X)\leq\beta]}.$$  Next, we take an outer expectation to remove the dependency on $n_{1}$. Recall that $n_{1}$ follows a Binomial$(n,\Pr[f(X)>\beta])$

distribution, so one can easily verify that
1. $\mathbb{E}[\binom{n_{1}}{2}]=\binom{n}{2}\cdot\Pr[f(X)>\beta]^{2}$ 2. $\mathbb{E}[\binom{n-n_{1}}{2}]=\binom{n}{2}\cdot(1-\Pr[f(X)>\beta])^{2}=\binom{n}{2}\cdot\Pr[f(X)\leq\beta]^{2}$ 3. $\mathbb{E}[n_{1}(n-n_{1})]=2\binom{n}{2}\cdot\Pr[f(X)>\beta]\cdot\Pr[f(X)\leq\beta]$.  
It follows immediately that

$$\mathbb{E}[L(f(\vec{X}),\vec{Y})]=\mathbb{E}[\mathbb{E}[L(f(\vec{X}),\vec{Y})\mid n_{1}]]$$ $$\leq\mathbb{E}\left[\binom{n_{1}}{2}\cdot\left(\frac{\rho(1-\rho)(1-\epsilon_{1})\epsilon_{0}}{\Pr[f(X)>\beta]^{2}}-\text{Var}(Z\mid f(X)>\beta)\right)\right]$$ $$=\binom{n}{2}\cdot\left(\rho(1-\rho)(1-\epsilon_{1})\epsilon_{0}-\kappa_{1}\right)$$
$$\mathbb{E}[M(f(\vec{X}),\vec{Y})]=\mathbb{E}[\mathbb{E}[M(f(\vec{X}),\vec{Y})\mid n_{1}]]$$ $$=\mathbb{E}\left[\binom{n_{1}}{2}\cdot\frac{\epsilon_{0}^{2}(1-\rho)^{2}}{\Pr[f(X)>\beta]^{2}}\right]$$ $$=\binom{n}{2}\cdot(1-\rho)^{2}\epsilon_{0}^{2}$$
$$\mathbb{E}[N(f(\vec{X}),\vec{Y})]=\mathbb{E}[\mathbb{E}[N(f(\vec{X}),\vec{Y})\mid n_{1}]]$$ $$\leq\mathbb{E}\left[\binom{n-n_{1}}{2}\cdot\left(\frac{\rho(1-\rho)(1-\epsilon_{0})\epsilon_{1}}{\Pr[f(X)\leq\beta]^{2}}-\text{Var}(Z\mid f(X)\leq\beta)\right)\right]$$ $$\qquad\qquad+\mathbb{E}\left[n_{1}(n-n_{1})\cdot\frac{\rho(1-\rho)\epsilon_{1}\epsilon_{0}}{\Pr[f(X)>\beta]\cdot\Pr[f(X)\leq\beta]}\right]$$ $$=\binom{n}{2}\cdot\left(\rho(1-\rho)(1-\epsilon_{0})\epsilon_{1}+2\rho(1-\rho)\epsilon_{1}\epsilon_{0}-\kappa_{2}\right)$$ $$=\binom{n}{2}\cdot\left(\rho(1-\rho)(1+\epsilon_{0})\epsilon_{1}-\kappa_{2}\right),$$
$$(Z\mid f(X)>\beta)$$
$$\mathbf{\tau}_{2}:=\operatorname*{Pr}[f(X)\leq\beta]^{2}$$

where κ1 := Pr[f(X) > β]
2· Var(Z | f(X) > β) and κ2 := Pr[f(X) ≤ β]
2· Var(Z | f(X) ≤ β).

The observation that Z = f(X) when f is calibrated gives the result from the main body.

## C. Experimental Details C.1. Ski-Rental: Citibike

Our experiments with CitiBike use ridership duration data from June 2015. Although summer months have slightly longer rides, the overall shape of the distributions is similar across months (i.e. left-skewed distribution). Figure 4 illustrates the

$\square$