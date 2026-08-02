011

014 015 016

018

024

026

034

036

038

# Online Uniform Sampling: Randomized Learning-Augmented Approximation Algorithms with Application to Digital Health

Anonymous Authors<sup>1</sup>

# Abstract

Motivated by applications in digital health, this work studies the novel problem of *online uniform sampling* (OUS), where the goal is to distribute a sampling budget uniformly across *unknown* decision times. In the OUS problem, the algorithm is given a budget b and a time horizon T, and an adversary then chooses a value τ <sup>∗</sup> ∈ [b, T], which is revealed to the algorithm online. At each decision time i ∈ [τ ∗ ], the algorithm must determine a sampling probability that maximizes the budget spent throughout the horizon, respecting budget constraint b, while achieving as uniform a distribution as possible over τ ∗ . We present the first randomized algorithm designed for this problem and subsequently extend it to incorporate learning augmentation. We provide *worst-case* approximation guarantees for both algorithms, and illustrate the utility of the algorithms through both synthetic experiments and a real-world case study involving the HeartSteps mobile application. Our numerical results show strong empirical *average* performance of our proposed randomized algorithms against previously proposed heuristic solutions.

# 1. Introduction

The problem of *online uniform sampling* (OUS) is motivated by applications in digital health, where administering interventions at inappropriate times, such as when users are not at risk,[<sup>1</sup>](#page-0-0) can significantly increase mental burden and hinder engagement with digital interventions [\(Li et al.,](#page-8-0) [2020;](#page-8-0) [Nahum-Shani et al.,](#page-8-1) [2018;](#page-8-1) [Wen et al.,](#page-9-0) [2017;](#page-9-0) [Mc-](#page-8-2)[Connell et al.,](#page-8-2) [2017;](#page-8-2) [Mann & Robinson,](#page-8-3) [2009\)](#page-8-3). Existing studies [\(Heckman et al.,](#page-8-4) [2015;](#page-8-4) [Klasnja et al.,](#page-8-5) [2008;](#page-8-5) [Dim-](#page-8-6) [itrijevic et al.](#page-8-6) ´ , [1972\)](#page-8-6) show excessive digital interventions can heighten user fatigue, suggesting a threshold beyond which intervention effectiveness declines. A strategy rooted in the ecological momentary assessment (EMA) literature and proven effective in mitigating user fatigue *involves allocating a fixed and limited budget for treatments delivered to the patient and delivering them with a uniform distribution across all risk times* (e.g., [Liao et al.](#page-8-7) [2018;](#page-8-7) [Dennis et al.](#page-8-8) [2015;](#page-8-8) [Rathbun et al.](#page-8-9) [2013;](#page-8-9) [Scott et al.](#page-8-10) [2017a](#page-8-10)[;b;](#page-9-1) [Shiffman](#page-9-2) [et al.](#page-9-2) [2008;](#page-9-2) [Stone et al.](#page-9-3) [2007\)](#page-9-3). However, this strategy is challenging because the true number of risk times is unknown, inspiring the OUS problem.

Contributions Our contributions in this paper are two-fold. First, we formulate the common OUS problem in digital health as an online optimization problem and provide randomized algorithms that perform well in practice with *competitive ratio* guarantees. The competitive ratio measures the performance of an online algorithm against an offline clairvoyant benchmark, assuming the unknown parameter is revealed to the clairvoyant in advance. These guarantees are inherently conservative: 1) no online algorithm can achieve the same performance as the clairvoyant in practice (i.e., a competitive ratio of 1 is unattainable in OUS), and 2) they hold across *all* problem instances or sample paths (i.e., they are worst-case guarantees). Consequently, online approximation algorithms may exhibit conservative behavior. To address this, we numerically illustrate the practicality of our algorithm, demonstrating that they outperform naive benchmarks on average.

Second, we extend our algorithm to the practical setting where a confidence interval *containing* the true risk time is provided, potentially through a valid statistical inference procedure. We conduct the competitive ratio analysis for our proposed learning-augmented approximation algorithm, demonstrating its *consistency* in the strong sense—optimal performance is achieved when the confidence interval width is zero—and *robustness*—the learning-augmented algorithm performs no worse than the non-learning augmented counterpart. Our findings indicate that, in almost all tested scenarios, the randomized learning-augmented algorithm outperforms its non-learning augmented counterpart.

<sup>1</sup>[Anonymous Institution, Anonymous City, Anonymous Region,](#page-8-6) [Anonymous Country. Correspondence to: Anonymous Author](#page-8-6) <[anon.email@domain.com](#page-8-6)>.

[Preliminary work. Under review by the International Conference](#page-8-6) [on Machine Learning \(ICML\). Do not distribute.](#page-8-6)

<sup>1</sup>[Risk times are when the patient is susceptible to a negative](#page-8-6) [event, such as smoking relapse.](#page-8-6)

058

071

074

076

078

087 088

090 091

093 094

096

098

100

104

106

108 109 troduce our randomized algorithm without learning augmentation in Section [3.](#page-3-0) This algorithm is segmented into three distinct cases based on the horizon length to budget ratio, with a competitive ratio established for each. In Section [4,](#page-4-0) we develop a learning-augmented algorithm that integrates a prediction interval and provide theoretical justification for its effectiveness. The efficacy of these algorithms is first assessed through synthetic experiments, followed by their application to real-world data in Section [5.](#page-5-0)

## 1.1. Related Work

Online Uniform Sampling Existing methodologies, primarily sourced from the EMA literature, focus on delivering interventions through the form of mobile self-report requests over a fixed time horizon. These approaches are constrained by budget and uniformity considerations to minimize user burden and ensure accurate reflection of user conditions across diverse contexts [\(Dennis et al.,](#page-8-8) [2015;](#page-8-8) [Rathbun et al.,](#page-8-9) [2013;](#page-8-9) [Scott et al.,](#page-8-10) [2017a](#page-8-10)[;b\)](#page-9-1). In this work, we permit intervention only when users are *at risk*, leading to an unknown horizon length. This introduces a significant challenge in balancing the allocation of a limited budget with the need to maintain uniformity in intervention delivery. To address this issue, [Liao et al.](#page-8-7) [\(2018\)](#page-8-7) developed a heuristic algorithm, but its performance depends heavily on the accuracy of the predicted number of risk times. When the prediction is inaccurate, the algorithm lacks theoretical guarantees, highlighting the need for a more robust algorithm design.

Multi-option Ski-rental Problem Our work closely relates to the *multi-option ski-rental* (MOSR) problem [\(Zhang et al.,](#page-9-4) [2011;](#page-9-4) [Shin et al.,](#page-9-5) [2023\)](#page-9-5), where the number of snowy days is unknown. Customers have multiple ski rental options, differing in cost and duration. The goal is to minimize costs while ensuring ski availability on snowy days. [Shin et al.](#page-9-5) [\(2023\)](#page-9-5) introduced a randomized algorithm for MOSR, with a tight e-competitive ratio. A random variable B is introduced as a proxy for the unknown true horizon T. B is initialized to α, following a density function 1/α within [1, e). The algorithm iteratively solves an optimization problem to identify an optimal set of rental options within budget B, maximizing day coverage. Customers sequentially utilize the options until depletion, at which point B is increased by a factor of e, and the process is repeated.

Our work builds upon [Shin et al.](#page-9-5) [\(2023\)](#page-9-5), leveraging the same randomized algorithmic idea. However, our problem setting is *significantly different* from that of MOSR. In particular, instead of having discrete ski-rental options, at each decision time, the algorithm needs to decide on the sampling probability, which is continuous in nature. Further, in our problem, the sum of the sampling probability cannot exceed a predefined budget, while such constraints do not exist in MOSR. Our problem additionally has a uniformity consideration.

Learning-Augmented Online Algorithms Many online algorithms incorporate black-box point predictions on the unknown parameters to improve their worst-case guarantees [\(Purohit et al.,](#page-8-11) [2018;](#page-8-11) [Bamas et al.,](#page-8-12) [2020;](#page-8-12) [Wei & Zhang,](#page-9-6) [2020;](#page-9-6) [Jin & Ma,](#page-8-13) [2022\)](#page-8-13). The confidence of these point estimates is often represented by a single parameter, with a higher value indicating more accurate predictions. When the confidence is low, most work do not guarantee that the learning-augmented algorithm will perform no worse than the non-learning counterpart [\(Bamas et al.,](#page-8-12) [2020\)](#page-8-12). In practice, prediction confidence intervals, rather than point estimates, are often generated using valid statistical inference methods. A wider confidence interval typically indicates less informative predictions [\(Shafer & Vovk,](#page-9-7) [2008\)](#page-9-7). [Im et al.](#page-8-14) [\(2021\)](#page-8-14) consider the setting where the prediction provides a range of values for key parameters in the online knapsack problem. However, their deterministic solution cannot be directly extended to our setting, as the number of risk times in OUS is stochastic. We introduce the first integration of confidence intervals into randomized algorithms for OUS. This integration enables our proposed algorithms to surpass the performance of their non-learning counterpart, even with a wide confidence interval.

# 2. Problem Framework

In the context of digital interventions, we define the OUS problem as presented by [Liao et al.](#page-8-7) [\(2018\)](#page-8-7). Let T denote the total number of decision points within a decision period (e.g., within a day). At any given time t ∈ [1, T] in each decision period, patients encounter binary risk levels[<sup>2</sup>](#page-1-1) (determined by data from wearable devices), indicating whether the patient is likely to experience an adverse event, such as relapse to smoking. The distribution of risk levels is allowed to change *arbitrarily* across decision periods since treatments may influence and reduce subsequent risk.

Let τ <sup>∗</sup> be the *unknown true* number of risk times that a patient experiences in a decision period. Note that τ ∗ is stochastic and is revealed *only* at the end of the horizon T, corresponding to the last decision time in the decision period. We define p<sup>i</sup> ∈ (0, 1) to be the treatment probability at time i ∈ [τ ∗ ]. We preclude the possibility that p<sup>i</sup> = 0 or p<sup>i</sup> = 1 to facilitate after-study inference [\(Boruvka et al.,](#page-8-15) [2018;](#page-8-15) [Zhou et al.,](#page-9-8) [2023;](#page-9-8) [Kallus & Zhou,](#page-8-16) [2022\)](#page-8-16).

The algorithm is provided with a *soft* budget of b, representing the total *expected* number of interventions allowed to be delivered within each decision period. We assume τ <sup>∗</sup> > b as evidenced in practice [\(Liao et al.,](#page-8-7) [2018\)](#page-8-7). At each decision time i, the algorithm decides the intervention probability p<sup>i</sup> . The objectives of the OUS problem [\(Liao et al.,](#page-8-7) [2018;](#page-8-7) [Den-](#page-8-8)

<sup>2</sup>[When multiple risk levels are present, the problem naturally](#page-8-8) [decomposes into independent subproblems for each risk level, see](#page-8-8) [more details in Appendix](#page-8-8) [A.](#page-10-0)

114 115 116

118

124

126

128

131

134

136

138

151

154

158

160

[nis et al.,](#page-8-8) [2015;](#page-8-8) [Rathbun et al.,](#page-8-9) [2013;](#page-8-9) [Scott et al.,](#page-8-10) [2017a;](#page-8-10)[b;](#page-9-1) [Shiffman et al.,](#page-9-2) [2008;](#page-9-2) [Stone et al.,](#page-9-3) [2007\)](#page-9-3) are to 1) assign the intervention probabilities {pi}i∈[<sup>τ</sup> <sup>∗</sup>] as uniform as possible across risk times, and 2) maximize the sum of intervention probabilities across risk times while adhering to the budget constraint b.

Abstractly, in the OUS problem, the algorithm is given a budget b and a time horizon T, and an adversary then chooses a value τ <sup>∗</sup> ∈ [b, T], which is revealed to the algorithm online. At each decision time i ∈ [τ ∗ ], the algorithm must determine a sampling probability that maximizes the budget spent throughout the horizon, respecting the budget constraint b, while achieving as uniform a distribution as possible over τ ∗ .

*[W](#page-8-8)ithout additional information on* τ ∗ , the two objectives compete with each other. A naive solution to fulfill the first objective is to set p<sup>i</sup> = b/T, i ∈ [τ ∗ ], which, however, fails to maximize the sum of intervention probabilities. Conversely, if we set p<sup>i</sup> to be a large constant value, there is a risk of depleting the budget before the end of the horizon, thus failing to achieve the uniformity objective. Therefore, the optimality of the two objectives cannot be simultaneously achieved without additional information on τ ∗ . [Liao](#page-8-7) [et al.](#page-8-7) [\(2018\)](#page-8-7) provided a heuristic algorithm for OUS given a point estimate of τ ∗ . The algorithm's performance is significantly influenced by the accuracy of this forecast. In this work, we introduce randomized algorithms for OUS with robust worst-case guarantees, considering settings both with and without learning augmentation.

#### 2.1. OUS as An Online Optimization Problem

In this section, we formulate OUS as an online optimization problem, where the objective function provides a uniform way of comparing the performance of different approximation algorithms, and the constraint defines the set of feasible solutions.

Specifically, we aim to find a sequence of treatment probability assignments {pi}i∈[<sup>τ</sup> ∗] that achieves the following two objectives:

- 1. Maximizes the sum of treatment probabilities across risk times, subject to the "soft" budget b;
- 2. Penalizes changes in treatment probabilities within each risk level.

Formally, the OUS problem can be expressed using the following optimization problem:

$$\left\{ \max \sum_i^\tau p_i - \frac{1}{\tau^*} \ln \left( \frac{\max_{i \in [\tau^*]} p_i}{\min_{i \in [\tau^*]} p_i} \right) : \mathbb{E} \left[ \sum_{i=1}^\tau p_i \right] \leq b, p_i \in (0, 1), \forall i \in [\tau^*] \right\} \quad (1)$$

where the expectation, E, in the budget constraint is taken over the randomness in the algorithm. This budget constraint is "soft" in the sense that if we have multiple decision periods (which is the case in digital health), we should satisfy the budget constraint in expectation.

*Remark* 2.1*.* Notably, the purpose of formulating the optimization problem is not to solve it optimally, but rather to provide a feasible solution without knowledge of the unknown τ ∗ . Rather than setting uniformity as a constraint, we incorporate it into the design of our approximation algorithms. By including uniformity as a penalty term in the objective function, represented by:

$$\frac{1}{\tau^*} \ln \left( \frac{\max_{i \in [\tau^*]} p_i}{\min_{i \in [\tau^*]} p_i} \right), \quad (2)$$

we can directly compare the overall performance of different online approximation algorithms, including how well they achieve uniformity, by comparing their objective function values.

The choice of the penalty term [\(2\)](#page-2-0) is inspired by the entropy change concept from thermodynamics [\(Smith,](#page-9-9) [1950\)](#page-9-9). This choice is not unique but it has several nice properties: a) it equals to 0 if and only if {pi}i∈[<sup>τ</sup> <sup>∗</sup>] are identical, b) it increases with the maximum difference in {pi}i∈[<sup>τ</sup> ∗] , and c) it tends towards infinity as the value of p<sup>i</sup> approaches to zero, penalizing scenarios where the expected budget is depleted before the horizon ends. We note that one can replace the term 1/τ <sup>∗</sup> in the penalty by a tuning parameter σ, which controls the strength of the penalty, as discussed in Remarks [3.3](#page-4-1) and [4.3.](#page-5-1) [<sup>3</sup>](#page-2-1) Finally, we highlight that KL divergence cannot be used here to impose uniformity (see detailed discussion in Appendix [B\)](#page-10-1).

#### 2.2. Offline Clairvoyant and Competitive Ratio

In the *offline clairvoyant* benchmark, the clairvoyant possesses knowledge of τ ∗ . When provided with this value, the optimal solution to Problem [\(1\)](#page-2-2) is to set p<sup>i</sup> = b/τ <sup>∗</sup> . Consequently, the optimal value of the objective function in Problem [\(1\)](#page-2-2) is OPT(τ ∗ ) = b. Importantly, in practice, no online algorithm can attain OPT(τ ∗ ) as the offline clairvoyant benchmark serves as an upper bound on the best achievable performance for any *online* algorithm without knowledge of τ ∗ . Let SOL be the objective value of Problem [\(1\)](#page-2-2) achieved by a *randomized online* algorithm, we say that

Definition 2.2 (γ-competitive). An algorithm is γ*competitive* if <sup>E</sup>[SOL] ≥ γ · OPT(τ ∗ ).

*Remark* 2.3*.* First, we emphasize that the expectation in

<sup>3</sup> Since the current design of our algorithms does not explicitly account for the form of the penalty term, the penalty [\(2\)](#page-2-0) could also be replaced by any other suitable functions, with performance re-evaluated under the modified objective function.

168

171

174

176

178

194

196 197 198

200

204

206

208

211

214 215 216

Definition [2.2](#page-2-3) is taken *only* over the randomness of the algorithm. Second, we note that if the competitive ratio is provided, it holds in expectation for every feasible τ <sup>∗</sup> ∈ [b, T]. This implies that the competitive ratio serves as a worst-case guarantee: in any OUS instance, as long as the budget b and the maximum horizon length T remain fixed across decision periods, we can expect to meet the budget and achieve the stated competitive ratio, regardless of the specific realization of τ ∗ in each decision period.

The key difficulty in solving Problem [\(1\)](#page-2-2) in the online setting arises due to the unknown nature of τ ∗ . In Section [3,](#page-3-0) we introduce the first approximation algorithm for the OUS problem.

## 2.3. With Learning Augmentation

In the *learning-augmented* setting, we are additionally provided with a prediction confidence interval [L, U], generated by a valid statistical procedure, that contains the unknown *true* τ <sup>∗</sup> with high probability. A wider confidence interval reflects lower prediction quality. For simplicity, we assume τ ∗ lies within the interval, though our results generalize to cases where it is contained with high probability.

To evaluate the performance of the learning-augmented algorithm in the presence of a prediction confidence interval, we extend the standard consistency-robustness analysis from the prior literature [\(Lykouris & Vassilvtiskii,](#page-8-17) [2018;](#page-8-17) [Purohit](#page-8-11) [et al.,](#page-8-11) [2018;](#page-8-11) [Bamas et al.,](#page-8-12) [2020;](#page-8-12) [Shin et al.,](#page-9-5) [2023\)](#page-9-5). Specifically, an algorithm is said to be λ-*consistent* if it achieves <sup>E</sup>[SOL] ≥ λ · OPT(τ ∗ ) when the prediction is perfect, i.e., when L = U, indicating a zero-length interval.[<sup>4</sup>](#page-3-1) This aligns with the standard definition where the prediction is accurate [\(Shin et al.,](#page-9-5) [2023\)](#page-9-5). Conversely, an algorithm is ρ-*robust* if it satisfies <sup>E</sup>[SOL] ≥ ρ · OPT(τ ∗ ) regardless of the width of the prediction interval [L, U], corresponding to the previous definition where the prediction can be arbitrarily inaccurate.

In Section [4,](#page-4-0) we show that our proposed learning-augmented algorithm is 1-consistent, achieving the optimal solution when the interval width is zero. Moreover, the competitive ratio of our learning-augmented algorithm closely matches that of the non-learning augmented counterpart, even when the prediction quality deteriorates. To the best of our knowledge, this is the first work that provide a 1-consistency guarantee on learning-augmented algorithms, after careful engineering of the algorithms.

# 3. Randomized Algorithm

In this section, we introduce our randomized algorithm, Algorithm [1,](#page-3-2) designed for the OUS problem *without* learning

augmentation. This algorithm is inspired by the randomized algorithm proposed by [Shin et al.](#page-9-5) [\(2023\)](#page-9-5) for the MOSR problem. Due to the significant differences in problem setup outlined in Section [1.1,](#page-1-2) the design of our algorithm requires 1) imposing a discrete structure on the sampling probabilities to account for uniformity considerations, making the analysis of the algorithm more tractable, and 2) explicitly addressing the finite horizon length and budget constraint, ensuring that the randomized algorithm does not exceed the budget in expectation.

Algorithm 1 Randomized Online Algorithm

1: Input: T, b 2: Initialize: j = 1, we sample α ∈ [b, be] from a distribution with p.d.f. f(α) = 1/α, and initialize τ˜ = α 3: for i = 1, ..., τ <sup>∗</sup> do 4: We calculate:

$$\text{Int}(\tilde{\tau}) = \begin{cases} \begin{bmatrix} \tilde{\tau} \end{bmatrix} & w.p. & \begin{bmatrix} \tilde{\tau} \end{bmatrix} - \tilde{\tau} \\ \begin{bmatrix} \tilde{\tau} \end{bmatrix} & w.p. & \tilde{\tau} - \begin{bmatrix} \tilde{\tau} \end{bmatrix} \end{cases}$$

5: if T ≤ be then 6: Update τ˜ and set p<sup>i</sup> using Subroutine [1](#page-3-2) 7: else if be < T ≤ be<sup>2</sup> then 8: Update τ˜ and set p<sup>i</sup> using Subroutine [2](#page-4-2) 9: else 10: Update τ, b ˜ and set p<sup>i</sup> using Subroutine [3](#page-4-3) 11: end if 12: Output treatment probability p<sup>i</sup> 13: end for

The proposed algorithm, Algorithm [1,](#page-3-2) provides a feasible solution to Problem [\(1\)](#page-2-2). At its core, our algorithm assigns the sampling probabilities in a monotonically nonincreasing fashion over time. To accommodate varying practical scenarios where the budget-to-horizon ratio differs across applications, we designed specialized approximation algorithms for three possible scenarios: 1) T ≤ be (Subroutine [1\)](#page-3-2), 2) be < T ≤ be<sup>2</sup> (Subroutine [2\)](#page-4-2), and 3) T > be<sup>2</sup> (Subroutine [3\)](#page-4-3).

We maintain a running "guess" of τ ∗ , denoted by τ˜. We initialize τ˜ to be α, where α ∼ [b, b · e] with density 1/α, and e represents the Euler's number. If the current number of risk times i is within our running guess τ˜, then we do not change the current sampling assignment probability. Otherwise, we update τ˜ as τ˜ = ˜τe and update the sampling probability according to Algorithm [1,](#page-3-2) depending on the length of the horizon T relative to b. The random draw τ˜ controls not only the value of the sampling probability but also the duration of each stage. Once the algorithm reaches τ˜, it transitions to the next stage, resulting in a stage-wise constant probability sequence.

<sup>4</sup> Similar to Definition [2.2,](#page-2-3) the expectation is taken over the randomness in the algorithm.

*257*

*264*

*266*

|    | Subroutine |     |    |      | 1 ( i , b , τ ˜ , T , Int(˜ τ ) ) |
|----|------------|-----|----|------|-----------------------------------|
| 1: | if         | i   | >  |      | Int(˜ τ ) then                    |
| 2: |            | τ ˜ | =  |      | ˜ τe                              |
| 3: | end        |     | if |      |                                   |
| 4: | p i        |     | =  |      |                                   |
|    |            |     |    | min( | T ,τ ˜( e − 1))                   |
|    | Subroutine |     |    |      | 2 ( i , b , τ ˜ , Int(˜ τ ) )     |
| 1: | if         | i   | >  |      | Int(˜ τ ) then                    |
| 2: |            | j   | =  | j    | + 1 , τ ˜ = ˜ τe                  |
| 3: | end        |     | if |      |                                   |
| 4: | if         | j   | ≥  | 3    | then                              |
| 5: |            | p   | i  | =    |                                   |
|    |            |     |    |      | τ e ˜                             |
| 6: | else       |     |    |      |                                   |
| 7: |            | p   | i  | =    |                                   |
|    |            |     |    |      | τ ˜( e − 1)                       |
| 8: | end        |     | if |      |                                   |

the sampling probabilities outputted from Algorithm [1](#page-3-2) satisfies the budget constraint in Problem [\(1\)](#page-2-2):

Lemma 3.1. *Let* p A1 i *be the probability returned by Algorithm [1](#page-3-2) at risk time* i ∈ [τ ∗ ]*. This solution always satisfies the budget constraint in expectation, i.e.,* E hP<sup>τ</sup> ∗ <sup>i</sup>=1 p A1 i i ≤ b*, where the expectation is taken over the randomness of the algorithm.*

The proof of Lemma [3.1](#page-4-4) is included in Appendix [C.1.](#page-11-0) Next, by leveraging the monotonically non-increasing nature of the sampling probabilities, the objective in Problem [\(1\)](#page-2-2) simplifies to

$$\max \sum_{i=1}^{\tau^*} p_i - \frac{1}{\tau^*} \ln \left( \frac{p_1}{p_{\tau^*}} \right). \quad (3)$$

Using Equation [\(3\)](#page-4-5), we compute the competitive ratio of Algorithm [1:](#page-3-2)

Theorem 3.2. *Algorithm [1](#page-3-2) is* X (T)*-competitive, where* X *is defined as follows:*

$$\mathcal{X}(T) := \begin{cases} \frac{1}{e} \left( \ln(e-1) + \frac{1}{e-1} \right) & \text{if } T \leq be, \\ \frac{1}{e} & \text{if } be < T \leq be^2, \\ \frac{1}{e} - \frac{1}{e^2} & \text{if } T > be^2. \end{cases}$$

The above competitive ratio is conservative by design: It was derived by taking the worst case over *unknown* τ ∗ and the horizon length T within each case. The proof of Theorem [3.2](#page-4-6) in Appendix [C.2](#page-12-0) outlines the competitive ratio as a function of τ ∗ and T. Additionally, in Section [5,](#page-5-0) we investigate the impact of varying τ <sup>∗</sup> while keeping the horizon length fixed, providing a numerical illustration of how the expected competitive ratio changes. We note that the expected competitive ratio, averaged over the unknown τ ∗ , is much better than our theoretical competitive ratio illustrated above. Based on our theoretical competitive ratio in

| Subroutine <b>3</b> ( <i>i</i> , <i>b</i> , $\tilde{\tau}$ , Int( $\tilde{\tau}$ )) |
|-------------------------------------------------------------------------------------|
| 1: <b>if</b> <i>i</i> > Int( $\tilde{\tau}$ ) <b>then</b>                           |
| 2: <i>j</i> = <i>j</i> + 1, $\tilde{\tau} = \tilde{\tau}e$                          |
| 3: <b>if</b> <i>j</i> ≥ 3 <b>then</b>                                               |
| 4: <i>b</i> = <i>b</i> (1 - $\frac{1}{e}$ )                                         |
| 5: <b>end if</b>                                                                    |
| 6: <b>end if</b>                                                                    |
| 7: <i>p</i> <sub>i</sub> = $\frac{b}{\tilde{\tau}e}$                                |

Theorem [3.2,](#page-4-6) we recommend choosing the horizon length T relative to the budget b to be below be<sup>2</sup> , which aligns with our empirical findings in Section [5](#page-5-0) (see Remark [5.1](#page-6-0) for details).

*Remark* 3.3*.* As stated in Section [2.1,](#page-2-4) the term <sup>1</sup> τ <sup>∗</sup> in the penalty can be replaced by a tunable strength parameter σ. In Section [C.2,](#page-12-0) we show that for T ≤ be<sup>2</sup> , the above results hold over a wide range of σ values, specifically σ ≤ b 2 . However, when T > be<sup>2</sup> , σ should be on the order of <sup>1</sup> τ ∗ , ensuring that the penalty term scales similarly to the budget term in the objective.

*Remark* 3.4*.* Establishing an upper bound on the performance of any randomized algorithm for the OUS problem is challenging due to the non-smooth nature of the objective function and the problem's three different operating regimes. In Appendix [G,](#page-28-0) we derive a loose upper bound of 0.5 for the OUS problem using Yao's lemma [\(Yao,](#page-9-10) [1977\)](#page-9-10) and leave the derivation of a tighter bound for future work.

# 4. Learning-Augmented Algorithm

In this section, we propose a new approximation algorithm, Algorithm [2,](#page-4-2) under the learning-augmented setting, where we are provided with prediction confidence intervals [L, U] for the unknown τ ∗ . Algorithm [2](#page-4-2) builds upon the nonlearning augmented counterpart, Algorithm [1,](#page-3-2) utilizing the given confidence interval for optimization. Similar to Algorithm [1,](#page-3-2) we initialize α ∼ [b, be] with density 1/α, and the current "guess" of τ ∗ is reflected by τ˜ + L.

In Algorithm [2,](#page-4-2) the three scenarios differ from those in Algorithm [1.](#page-3-2) Here, the distinction is based on the relationship between the upper bound of the interval, U, and the budget b. The three scenarios are 1) U ≤ be (Subroutine [4\)](#page-5-2), 2) be < U ≤ be<sup>2</sup> , further divided into 2a) U − L ≤ b(e − 1) (Subroutine [4\)](#page-5-2), and 2b) U − L > b(e − 1) (Subroutine [2\)](#page-4-2), and 3) U > be<sup>2</sup> , further divided into 3a) U − L ≤ b(e + 1) (Subroutine [5\)](#page-5-3), and 3b) U − L > b(e + 1) (Subroutine [6\)](#page-6-1). Similarly, we first demonstrate that Algorithm [2](#page-4-2) produces a feasible solution to Problem [\(1\)](#page-2-2), with the proof provided in Appendix [D.1](#page-17-0) .

Lemma 4.1. *Let* p A2 i *be the probability returned by Algorithm [2](#page-4-2) at risk time* i ∈ [τ ∗ ]*. This solution always satisfies*

278

289 290

294

296

298

300

304

306

308 309

311

314 315 316

318

324

326

328

Algorithm 2 Randomized Online Algorithm With Predic-

tion Confidence Intervals 1: Input: T, b, [L, U]

2: Initialize: j = 1, sample α ∈ [b, be] from a distribution with p.d.f. f(α) = 1/α, and initialize τ˜ = α

3: for i = 1, ..., τ <sup>∗</sup> do 4: We calculate:

$$\text{Int}(\tilde{\tau}) = \begin{cases} \begin{bmatrix} \tilde{\tau} \\ \tilde{\tau} \end{bmatrix} & w.p. & \begin{bmatrix} \tilde{\tau} \end{bmatrix} - \tilde{\tau} \\ \begin{bmatrix} \tilde{\tau} \\ \tilde{\tau} \end{bmatrix} & w.p. & \tilde{\tau} - \begin{bmatrix} \tilde{\tau} \end{bmatrix} \end{cases}$$

5: if U ≤ be then

6: Update τ˜ and set p<sup>i</sup> using Subroutine [4](#page-5-2)

7: else if be < U ≤ be<sup>2</sup>

then

8: if U − L ≤ b(e − 1) then

9: Update τ˜ and set p<sup>i</sup> with Subroutine [4](#page-5-2)

10: else

11: Update τ˜ and set p<sup>i</sup> with Subroutine [2](#page-4-2)

12: end if 13: else

14: if U − L ≤ b(e + 1) then

15: Update τ˜ and set p<sup>i</sup> with Subroutine [5](#page-5-3)

16: else

17: Update τ, b ˜ and set p<sup>i</sup> with Subroutine [6](#page-6-1)

18: end if 19: end if

20: Output sampling probability p<sup>i</sup>

21: end for

*the budget constraint in expectation, i.e.,* E hP<sup>τ</sup> ∗ <sup>i</sup>=1 p A2 i i ≤ b, *where the expectation is taken over the randomness of the algorithm.*

Next, we provide a theoretical guarantee on its performance:

Theorem 4.2. *Algorithm [2](#page-4-2) is* 1*-consistent and* X (U)*-robust, where* X (U) *is defined as follows:*

$$\mathcal{X}(U) := \begin{cases} \ln 2 + \frac{e-1}{e} \ln \frac{e-1}{e} & \text{if } U \leq be, \\ \frac{1}{e} & \text{if } be < U \leq be^2, \\ \frac{1}{2} - \ln(e^2 - e + 1) & \text{if } U > be^2. \end{cases}$$

We first note that Algorithm [2](#page-4-2) is 1-consistent, achieving the performance of the offline clairvoyant when the prediction is perfect. The proof of Theorem [4.2](#page-5-4) in Appendix [D.2](#page-19-0) provides a detailed analysis of the competitive ratio, which depends on the parameters τ ∗ , L, and U. [<sup>5</sup>](#page-5-5) Furthermore, Section [5](#page-5-0) explores the impact of varying the prediction confidence interval width U − L while keeping τ ∗ constant. Our findings reveal that Algorithm [2](#page-4-2) almost always outperforms Algorithm [1.](#page-3-2) Finally, we discuss the design choice

| Subroutine |      |      |       | 4 ( i , b , τ ˜ , L , U , Int(˜ τ ) ) |
|------------|------|------|-------|---------------------------------------|
| 1:         | if i | >    | Int(˜ | τ ) + L then                          |
| 2:         | τ    | ˜ =  | τe    | ˜                                     |
| 3:         | end  | if   |       |                                       |
| 4:         | p i  | =    |       |                                       |
|            |      | min( |       | U,τ ˜+ L )                            |
| Subroutine |      |      |       | 5 ( i , b , τ ˜ , L , U , Int(˜ τ ) ) |
| 1:         | if i | >    | Int(˜ | τ ) + L then                          |
| 2:         | τ    | ˜ =  | τe    | ˜                                     |
| 3:         | end  | if   |       |                                       |
| 4:         | p i  | =    |       |                                       |
|            |      | min( |       | U,τ e ˜ + L )                         |

of T relative to b in the context of prediction intervals in Remark [5.2.](#page-6-2)

*Remark* 4.3*.* Similarly, the term <sup>1</sup> τ <sup>∗</sup> in the penalty can be replaced by a tuning parameter σ. In Section [D.2,](#page-19-0) we show that for U ≤ be<sup>2</sup> , the above results hold for a wide range of σ values, specifically σ ≤ b e . However, when T > be<sup>2</sup> , σ should be of the order <sup>1</sup> τ <sup>∗</sup> to align the penalty term with the budget term in the objective.

# 5. Experiments

In this section, we numerically assess the performance of our proposed algorithms through numerical experiments conducted on both synthetic and real-world datasets.

#### 5.1. Synthetic Experiments

Benchmarks In the setting without learning augmentation, we compare Algorithm [1](#page-3-2) against a conservative benchmark that delivers interventions with a constant probability b/T. In the learning-augmented setting, where a confidence interval [L, U] is provided, we compare Algorithm [2](#page-4-2) against two benchmarks: (1) a benchmark that delivers interventions with a constant probability b/U, and (2) Algorithm [1.](#page-3-2)Due to the limited algorithmic work on OUS (Online Uniformity Scheduling) and the absence of existing algorithms that handle confidence intervals, we do not include additional benchmarks in the synthetic data experiments. However, in the real-world example, we also evaluate the SeqRTS algorithm [\(Liao et al.,](#page-8-7) [2018\)](#page-8-7), which does not account for the prediction uncertainty of τ ∗ . The metric used for the evaluation is the average competitive ratio.

Without Learning Augmentation In this setting, we evaluate the performance of Algorithm [1](#page-3-2) across all three scenarios outlined in Theorem [3.2.](#page-4-6) To do this, we fix the budget at b = 3 and alter the horizon lengths T to align with each scenario. For Scenarios 1 and 2, we set T to the maximum allowable values with b = 3, specifically T = 8 and 22, as illustrated in Figure [1](#page-7-0) (left and middle). For Scenario

<sup>5</sup> In Theorem 4.3, we present the competitive ratios for scenarios 1), 2), and 3) separately, combining the results of the respective subroutines.

334

336

338

351

354

356

358

360 361

364

366

368

371

374

378

#### Subroutine 6 (i, b, τ˜, L, U, Int(˜τ ))

1: if i > Int(˜τ ) + L then 2: j = j + 1 3: if j = 2 then 4: b = b(1 − τ˜+L−b τ˜(e−1)+L ) 5: else 6: b = b(1 − e ) 7: end if 8: τ˜ = ˜τe 9: end if 10: if j = 1 then 11: p<sup>i</sup> = b τ˜(e−1)+L 12: else 13: p<sup>i</sup> = b τ e˜ 14: end if

3, where T can grow asymptotically to infinity, we choose T = 100 for simplicity (Figure [1](#page-7-0) right). To simulate risk occurrences, we randomly choose an integer τ ∗ from the interval [b, T − 1] and then select τ <sup>∗</sup> distinct time points uniformly at random from the T available time steps as risk times.

Figure [1](#page-7-0) displays the average competitive ratio across a range of τ <sup>∗</sup> values. Figure [1a](#page-7-0) indicates that our randomized algorithm consistently outperforms the benchmark by a constant competitive ratio for all values of τ ∗ in Scenario 1. Similarly, Figure [1b](#page-7-0) shows that in Scenario 2, our randomized algorithm increasingly outperforms the benchmark as τ <sup>∗</sup> deviates further from the horizon length T. In Figure [1c,](#page-7-0) as T increases, the average competitive ratio of our algorithm remains constant and consistently outperforms the benchmark.[<sup>6</sup>](#page-6-3) Therefore, we conclude that our algorithm increasingly outperforms the benchmark as T grows to infinity.

*Remark* 5.1 (Design choice of b and T in the absence of prediction confidence intervals)*.* In real-world applications, the intervention budget for each risk level is often fixed. However, a key design consideration is the choice of T, i.e., the granularity of the decision period. As illustrated in Figure [1,](#page-7-0) while Scenario 3 achieves the greatest performance improvement as T approaches infinity, our randomized algorithm attains the highest competitive ratio across all τ ∗ in Scenarios 1 and 2. Thus, in the absence of prediction intervals, we recommend selecting T such that T ≤ be<sup>2</sup> .

With Learning Augmentation In this setting, we evaluate the performance of Algorithms [1](#page-3-2) and [2](#page-4-2) across varying prediction interval widths. As in the non-learningaugmented setting, we fix the budget at b = 3 and examine the performance of our learning-augmented algorithm for T = 8, 22, and 100, covering the three scenarios outlined inAlgorithm [2.](#page-4-2) To compare the performance of our algorithm across various confidence widths, we fix τ <sup>∗</sup> = Int[0.5(T + b)] across all simulations.[<sup>7</sup>](#page-6-4) The confidence intervals are randomly generated based on the given width and must contain τ ∗ .

Figure [2](#page-7-1) plots the average competitive ratio of each algorithm across a range of interval widths. We observe that the naive benchmark (where p<sup>i</sup> = b/U for all i ∈ [τ ∗ ]) outperforms the Algorithm [1](#page-3-2) (which does not have access to the prediction interval) when the confidence interval is narrow. This is not surprising as in this case τ <sup>∗</sup> ≈ U. However, as the prediction interval widens, our Algorithm [1](#page-3-2) outperforms the naive benchmark. In addition, we observe that our learning-augmented algorithm performs no worse than both the naive benchmark and the randomized algorithm. In particular, the advantage of Algorithm [2](#page-4-2) is the largest in Scenario 3.

*Remark* 5.2 (Design choice of b and T in presence of prediction intervals)*.* If we expect the value of τ ∗ to be small, we recommend setting T ≤ be<sup>2</sup> to ensure that the algorithm always operates in Scenario 2, where U ≤ be<sup>2</sup> . If we expect a reasonably large value of τ ∗ , we recommend setting a large value for T > be<sup>2</sup> such that the algorithm operates under Scenario 3, where U can exceed be<sup>2</sup> .

Additional experimental results for small τ ∗ are provided in Appendix [E.1.](#page-26-0) We note that as τ <sup>∗</sup> decreases, the advantage of our algorithm in Scenario 2 increases. We also include competitive ratio figures without the penalization term from Problem [\(1\)](#page-2-2) in Appendix [E.2,](#page-27-0) measuring the fraction of the budget spent by our algorithms.

#### 5.2. Real-World Experiments on HeartSteps

Our research is motivated by the Heartsteps V1 mobile health study, which aimed to increase physical activity among 37 sedentary individuals over a six-week period, with T = 144 decision points per day [\(Klasnja et al.,](#page-8-18) [2019\)](#page-8-18). At each decision time t, a risk variable R<sup>t</sup> is observed, which is binary: R<sup>t</sup> = 1 indicates a sedentary state, identified by recording fewer than 150 steps in the prior 40 minutes, and R<sup>t</sup> = 0 signifies a non-sedentary state. The total number of risk times, τ <sup>∗</sup> = P<sup>T</sup> <sup>t</sup>=1 Rt, is unknown. The primary objective here is to uniformly distribute approximately b = 1.5 interventions across sedentary times each day.

Benchmarks In addition to the naive benchmark b/U, we compare the performance of Algorithms [1](#page-3-2) and [2](#page-4-2) with the SeqRTS algorithm, as proposed by [Liao et al.](#page-8-7) [\(2018\)](#page-8-7). Under SeqRTS, the budget may be exhausted before all available

<sup>6</sup>This is because when b is fixed, the treatment assignment probability is independent of T.

If we allow τ ∗ to change across different simulations, then the difference that we observe in competitive ratio might be due to this change in τ ∗ .

394

396

![](_page_7_Figure_1.jpeg)

Figure 1. Average competitive ratio under non-learning augmented setting with b = 3. The scenarios correspond to T ≤ be, be < T ≤ be<sup>2</sup> , and T > be<sup>2</sup> , respectively.

![](_page_7_Figure_3.jpeg)

Figure 2. Average competitive ratio under learning augmented setting with b = 3. The scenarios correspond to U ≤ be, be < U ≤ be<sup>2</sup> , and U > be<sup>2</sup> , respectively.

![](_page_7_Figure_5.jpeg)

risk times are allocated. In such cases, a minimum probability of 1×10−<sup>6</sup> is assigned to the remaining risk times when evaluating the objective in Problem [\(1\)](#page-2-2). A comprehensive description of the SeqRTS method and additional implementation details are provided in Appendix [F.](#page-28-1) Performance is assessed using the competitive ratio and the average entropy change across user days.

In Figure [3,](#page-7-2) Algorithm [2,](#page-4-2) which incorporates a prediction interval, invariably outperforms the non-learning counterpart, the SeqRTS approach, and the naive benchmark b/U. Moreover, our proposed algorithms exhibit superior uniformity in risk times sampling, evidenced by reduced entropy change compared to both the non-learning algorithm and SeqRTS, as further detailed in Figure [7](#page-28-2) in Appendix [F.](#page-28-1) To better understand the behavior of SeqRTS, we set the minimum probability to 0 in Figure [8](#page-29-0) in Section [F.](#page-28-1) This figure illustrates that SeqRTS could deplete its budget even when the prediction is fairly accurate, highlighting the robustness of our algorithms under adversarial risk level arrivals.

Conclusion and Future Works This paper marks the first attempt to study the online uniform allocation problem within the framework of approximation algorithms. We in-

Figure 3. Average competitive ratio across user days under various prediction interval widths on HeartSteps V1 dataset. The shaded area indicates the ±1.96 standard error bounds across user days.

troduce two novel online algorithms—either incorporating learning augmentation or not—backed by rigorous theoretical guarantees and empirical results. Future works include adapting existing algorithms to scenarios where prediction intervals improve over time.

- 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 specifically highlighted here. References Bamas, E., Maggiori, A., and Svensson, O. The primal-dual method for learning augmented algorithms. *Advances in Neural Information Processing Systems*, 33:20083– 20094, 2020. Boruvka, A., Almirall, D., Witkiewitz, K., and Murphy,
  - S. A. Assessing time-varying causal effect moderation in mobile health. *Journal of the American Statistical Association*, 113(523):1112–1121, 2018. Dennis, M. L., Scott, C. K., Funk, R. R., and Nicholson,
  - L. A pilot study to examine the feasibility and potential effectiveness of using smartphones to provide recovery support for adolescents. *Substance abuse*, 36(4):486–492, 2015. Dimitrijevic, M. R., Faganel, J., Gregori ´ c, M., Nathan, P., ´ and Trontelj, J. Habituation: effects of regular and stochastic stimulation. *Journal of Neurology, Neurosurgery & Psychiatry*, 35(2):234–242, 1972. Heckman, B. W., Mathew, A. R., and Carpenter, M. J. Treatment burden and treatment fatigue as barriers to health. *Current opinion in psychology*, 5:31–36, 2015. Im, S., Kumar, R., Montazer Qaem, M., and Purohit, M. Online knapsack with frequency predictions. *Advances in Neural Information Processing Systems*, 34:2733–2743, 2021. Jin, B. and Ma, W. Online bipartite matching with advice: Tight robustness-consistency tradeoffs for the two-stage model. *Advances in Neural Information Processing Systems*, 35:14555–14567, 2022. Kallus, N. and Zhou, A. Stateful offline contextual policy evaluation and learning. In *International Conference on Artificial Intelligence and Statistics*, pp. 11169–11194. PMLR, 2022. Klasnja, P., Harrison, B. L., LeGrand, L., LaMarca, A., Froehlich, J., and Hudson, S. E. Using wearable sensors and real time inference to understand human recall of routine activities. In *Proceedings of the 10th international conference on Ubiquitous computing*, pp. 154–163, 2008. Klasnja, P., Smith, S., Seewald, N. J., Lee, A., Hall, K., Luers, B., Hekler, E. B., and Murphy, S. A. Efficacy of contextually tailored suggestions for physical activity: a Li, S., Psihogios, A. M., McKelvey, E. R., Ahmed, A., Rabbi, M., and Murphy, S. Microrandomized trials for promoting engagement in mobile health data collection: Adolescent/young adult oral chemotherapy adherence as an example. *Current opinion in systems biology*, 21:1–8, 2020. Liao, P., Dempsey, W., Sarker, H., Hossain, S. M., Al'absi, M., Klasnja, P., and Murphy, S. Just-in-Time but Not Too Much: Determining Treatment Timing in Mobile Health. *Proceedings of the ACM on interactive, mobile, wearable and ubiquitous technologies*, 2(4):179, December 2018. ISSN 2474-9567. doi: 10.1145/3287057. Lykouris, T. and Vassilvtiskii, S. Competitive caching with machine learned advice. In Dy, J. and Krause, A. (eds.), *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pp. 3296–3305. PMLR, 10–15 Jul 2018. Mann, S. and Robinson, A. Boredom in the lecture theatre: An investigation into the contributors, moderators and outcomes of boredom amongst university students. *British Educational Research Journal*, 35(2):243–258, 2009. McConnell, M. V., Shcherbina, A., Pavlovic, A., Homburger,
    - J. R., Goldfeder, R. L., Waggot, D., Cho, M. K., Rosenberger, M. E., Haskell, W. L., Myers, J., et al. Feasibility of obtaining measures of lifestyle from a smartphone app: the myheart counts cardiovascular health study. *JAMA cardiology*, 2(1):67–76, 2017. Nahum-Shani, I., Smith, S. N., Spring, B. J., Collins, L. M., Witkiewitz, K., Tewari, A., and Murphy, S. A. Justin-time adaptive interventions (jitais) in mobile health: key components and design principles for ongoing health behavior support. *Annals of Behavioral Medicine*, pp. 1–17, 2018. Purohit, M., Svitkina, Z., and Kumar, R. Improving online algorithms via ml predictions. *Advances in Neural Information Processing Systems*, 31, 2018. Rathbun, S. L., Song, X., Neustifter, B., and Shiffman, S. Survival analysis with time varying covariates measured at random times by design. *Journal of the Royal Statistical Society Series C: Applied Statistics*, 62(3):419–434, 2013. Scott, C. K., Dennis, M. L., Gustafson, D., and Johnson, K. A pilot study of the feasibility and potential effectiveness of using smartphones to provide recovery support. *Drug and Alcohol Dependence*, 100(171):e185, 2017a.

494

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be micro-randomized optimization trial of heartsteps. *Annals of Behavioral Medicine*, 53(6):573–582, 2019.

- 495 496 497 498 499 500 504 506 508 509 511 514 515 516 518 524 526 528 531 534 536 538 540 541 542 543 544 545 546 547 548 549 Scott, C. K., Dennis, M. L., and Gustafson, D. H. Using smartphones to decrease substance use via selfmonitoring and recovery support: study protocol for a randomized control trial. *Trials*, 18(1):1–11, 2017b. Shafer, G. and Vovk, V. A tutorial on conformal prediction. *Journal of Machine Learning Research*, 9(3), 2008. Shiffman, S., Stone, A. A., and Hufford, M. R. Ecological momentary assessment. *Annu. Rev. Clin. Psychol.*, 4: 1–32, 2008. Shin, Y., Lee, C., Lee, G., and An, H.-C. Improved learningaugmented algorithms for the multi-option ski rental problem via best-possible competitive analysis. *arXiv preprint arXiv:2302.06832*, 2023. Smith, J. M. Introduction to chemical engineering thermodynamics, 1950. Stone, A., Shiffman, S., Atienza, A., and Nebeling, L. *The science of real-time data capture: Self-reports in health research*. Oxford University Press, 2007. Wei, A. and Zhang, F. Optimal robustness-consistency trade-offs for learning-augmented online algorithms. *Advances in Neural Information Processing Systems*, 33: 8042–8053, 2020. Wen, C. K. F., Schneider, S., Stone, A. A., and Spruijt-Metz,
  - D. Compliance with mobile ecological momentary assessment protocols in children and adolescents: a systematic review and meta-analysis. *Journal of medical Internet research*, 19(4):e132, 2017. Yao, A. C.-C. Probabilistic computations: Toward a unified measure of complexity. In *18th Annual Symposium on Foundations of Computer Science (sfcs 1977)*, pp. 222–
  - 227. IEEE Computer Society, 1977. Zhang, G., Poon, C. K., and Xu, Y. The ski-rental problem with multiple discount options. *Information Processing Letters*, 111(18):903–906, 2011. Zhou, Z., Athey, S., and Wager, S. Offline multi-action policy learning: Generalization and optimization. *Operations Research*, 71(1):148–183, 2023.

554

556

558

560

564

566

568

571

574

576

578

594

596

598

#### A. Extension to Multiple Risk Levels

In this section, we discuss the extension of the online uniform risk times sampling problem to multiple risk levels.

At each time t ∈ [1, T], the patient is associated with an ordinal risk level from K possible levels. The higher the risk level, the more likely the patient will experience a negative event, such as a relapse to smoking. As stated previously, the distributions of risk levels are allowed to change *arbitrarily* across decision periods since we anticipate that the treatment will reduce subsequent risk.

Let τ ∗ k be the *unknown true* number of decision times at risk level k ∈ [K] in a decision period, which is revealed at the end of the horizon T. For each risk level k, we define pk,i<sup>k</sup> ∈ (0, 1) to be the treatment probability at time i<sup>k</sup> ∈ [τ ∗ k . The algorithm is provided with a *soft* budget of b<sup>k</sup> for each risk level k, representing the total *expected* number of interventions allowed to be delivered at risk level k within each decision period. As before, we assume τ ∗ <sup>k</sup> > b<sup>k</sup> for technical convenience [\(Liao et al.,](#page-8-7) [2018\)](#page-8-7).

Then at each decision time ik, the algorithm decides the intervention probability pk,i<sup>k</sup> . For each risk level k, the objectives of the online uniform allocation problem are to 1) assign the intervention probabilities {pk,i<sup>k</sup> }<sup>i</sup>k∈[<sup>τ</sup> ∗ k ] as uniform as possible across risk times, and 2) maximize the sum of intervention probabilities across risk times while adhering to the budget constraint bk.

For every risk level k ∈ [K], we define the following optimization problem:

$$\max \sum_{i_k}^{\tau_k^*} p_{k,i_k} - \frac{1}{\tau_k^*} \ln \left( \frac{\max_{i_k \in [\tau_k^*]} p_{k,i_k}}{\min_{i_k \in [\tau_k^*]} p_{k,i_k}} \right)$$

$$\text{s.t. } \mathbb{E} \left[ \sum_{i_k=1}^{\tau_k^*} p_{k,i_k} \right] \leq b_k$$

$$p_{k,i_k} \in (0,1) \quad \forall i \in [\tau_k^*]. \quad (4)$$

Notably, the proposed algorithms offer a feasible solution to the above optimization problem, allowing us to address each risk level independently.

# B. The Penalty Term for Uniformity

We have previously considered statistical distance measures for quantifying the uniformity objective. One important measure is the Kullback-Leibler (KL) divergence. However, this measure is not well defined in our setting since the optimal solution (which is a point mass on b/τ <sup>∗</sup> ) and the solutions given by our proposed algorithms are not defined on the same sample space.

Recall that for two discrete distributions P and Q defined on the same sample space X , the KL divergence is given by

$$D_{KL}(P\|Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)},$$

where P represents the data distribution, i.e., the optimal solution, and Q represents an approximation of P, i.e., the solution given by an algorithm.

Let us consider a toy example where τ <sup>∗</sup> = b(e − 1). In this case, the optimal solution should be p<sup>i</sup> = b <sup>b</sup>(e−1) = 1 e−1 for each risk time i ∈ [τ ∗ ]. The corresponding distribution is a point mass, meaning the sample space X consists of a single element (p<sup>1</sup> = e−1 , · · · , p<sup>τ</sup> <sup>∗</sup> = e−1 ) with probability 1. The solutions given by our proposed algorithms are of the form (p1, · · · , p<sup>τ</sup> <sup>∗</sup> ), but the sample space X is Q<sup>τ</sup> ∗ , where the support of Q is (0, 1).

Clearly, the optimal solution and the solutions given by the proposed algorithms are not defined on the same sample space. Therefore, the KL divergence is not well-defined in this context.

608 *Proof.* We prove that the budget constraint is satisfied in expectation under each subroutine in Algorithm [1.](#page-3-2)

616 617 618 619 620 621 When α falls in the range of [b, η], Algorithm 1 starts with p<sup>i</sup> = b <sup>α</sup>(e−1) with a running length of α, then transitions to p<sup>i</sup> = b T for the second phase with a running length of β − α. Otherwise, it consistently assigns p<sup>i</sup> = b <sup>T</sup> with a running length of β. Therefore, the expected budget consists of two parts: one for α ∈ [b, η], where the term inside represents the used budget or the sum of treatment probabilities, and the other for α ∈ [η, be], where the term inside represents the sum of treatment probabilities:

655 656 657 658 659 When α falls in the range of [b, β], Algorithm 1 starts with p<sup>i</sup> = b <sup>α</sup>(e−1) with a running length of α, transitions to p<sup>i</sup> = αe(e−1) with a running length of αe−α, and then continues with p<sup>i</sup> = e <sup>3</sup> with a running length of βe−αe. Otherwise, Algorithm 1 starts with p<sup>i</sup> = b <sup>α</sup>(e−1) with a running length of α, transitions to p<sup>i</sup> = b αe(e−1) with a running length of βe − α.

## C. Proof for Algorithm 1

#### C.1. Proof of Lemma 3.1: Budget constraint

#### Subroutine 1

Recall that τ ∗ is the true number of risk times. Here, we suppose τ <sup>∗</sup> = βe<sup>j</sup> ∗ for some j <sup>∗</sup> ∈ <sup>Z</sup> <sup>+</sup> and β ∈ [b, be]. Since T ≤ be, we have that j <sup>∗</sup> = 0.

In this analysis, our focus is solely on the worst-case scenario, where both T and τ ∗ are very close to be. Assuming η > b (where η = T e−1 ), if this is not the case, the algorithm uniformly sets p<sup>i</sup> = b T throughout.

$$\begin{aligned}\mathbb{E}[\text{Budget}] &= \int_{\eta}^{be} \frac{b}{T} \beta \frac{1}{\alpha} d\alpha + \int_b^{\eta} \left[ \frac{b}{\alpha(e-1)} \alpha + \frac{b}{T}(\beta - \alpha) \right] \frac{1}{\alpha} d\alpha \\ &= \frac{b\beta}{T} \ln \frac{be}{\eta} + \frac{b}{e-1} \ln \frac{\eta}{b} + \frac{b\beta}{T} \ln \frac{\eta}{b} - \frac{b}{T}(\eta - b) \\ &= \frac{b\beta}{T} + \frac{b}{e-1} \ln \frac{\eta}{b} - \frac{b\eta}{T} + \frac{b^2}{T} \\ &\leq b + \frac{b}{e-1} \ln \frac{T}{b(e-1)} - \frac{b}{e-1} + \frac{b^2}{T} \quad \text{increasing with } \beta \ (\beta = T) \\ &\leq b - \frac{b}{e-1} + \frac{b}{e-1} - \frac{b}{e-1} \ln(e-1) + \frac{b}{e} \quad \text{increasing with } T \ (T = be) \\ &\leq b - \frac{b}{e-1} \ln(e-1) + \frac{b}{e} \\ &\approx b.\end{aligned}$$

#### Subroutine 2

Reiterating our initial assumption, we set τ <sup>∗</sup> = βe<sup>j</sup> ∗ . The condition be < T ≤ be<sup>2</sup> limits j ∗ to either 0 or 1. However, our analysis is particularly concerned with the worst-case scenario, hence we consider only the case where j <sup>∗</sup> = 1.

689 690

694

696

698

700

704

706

708 By combining the above results, we establish Lemma [3.1.](#page-4-4)

709

711

The expected budget is therefore

$$\begin{aligned} \mathbb{E}[\text{Budget}] &= \int_{\beta}^{be} \left[ \frac{b}{\alpha(e-1)} \alpha + \frac{b}{\alpha e(e-1)} (\beta e - \alpha) \right] \frac{1}{\alpha} d\alpha + \int_b^{\beta} \left[ \frac{b}{\alpha(e-1)} \alpha \right. \\ &\quad \left. + \frac{b}{\alpha e(e-1)} (\alpha e - \alpha) + \frac{b}{\alpha e^3} (\beta e - \alpha e) \right] \frac{1}{\alpha} d\alpha \\ &= \frac{b}{e-1} + \frac{b\beta}{e-1} \left( \frac{1}{\beta} - \frac{1}{be} \right) - \frac{b}{e(e-1)} \ln \frac{be}{\beta} + \frac{b}{e} \ln \frac{\beta}{b} + \frac{b\beta}{e^2} \left( \frac{1}{b} - \frac{1}{\beta} \right) - \frac{b}{e^2} \ln \frac{\beta}{b} \\ &= \frac{b}{e-1} + \frac{b}{e-1} - \frac{\beta}{e(e-1)} + \frac{\beta}{e^2} - \frac{b}{e^2} - \frac{b}{e(e-1)} + \left( \frac{b}{e} + \frac{b}{e(e-1)} - \frac{b}{e^2} \right) \ln \frac{\beta}{b} \\ &= \frac{b}{e-1} + \frac{b}{e} - \frac{b}{e^2} - \frac{\beta}{e^2(e-1)} + \left( \frac{b}{e-1} - \frac{b}{e^2} \right) \ln \frac{\beta}{b} \\ &\leq \frac{b}{e-1} + \frac{b}{e} - \frac{b}{e^2} - \frac{be}{e^2(e-1)} + \left( \frac{b}{e-1} - \frac{b}{e^2} \right) \ln \frac{be}{b} \quad \text{increasing with } \beta \ (\beta = be) \\ &= \frac{2b}{e} + \frac{b}{e-1} - \frac{2b}{e^2} \approx b \end{aligned}$$

#### Subroutine 3

Under the assumption of τ <sup>∗</sup> = βe<sup>j</sup> ∗ , and given the condition T > be<sup>2</sup> , it is possible for j ∗ to be 0 or to extend towards infinity. Our focus, however, is confined to the worst-case scenarios, particularly those where j <sup>∗</sup> ≥ 1.

When α falls in the range of [b, β]. Algorithm 1 stops with p<sup>i</sup> = b(1−1/e) j ∗ αej∗+2 with a running length of βe<sup>j</sup> ∗ −αe<sup>j</sup> ∗ . Otherwise, Algorithm 1 stops with p<sup>i</sup> = b(1−1/e) j ∗−<sup>1</sup> αej∗+1 with a running length of βe<sup>j</sup> ∗ − αe<sup>j</sup> <sup>∗</sup>−1 . Therefore, the expected budget is

$$\begin{aligned} \mathbb{E}[\text{Budget}] &= \int_{\beta}^{be} \left[ \frac{b}{\alpha e^j} \alpha + \sum_{i=2}^{j^*} \frac{b(1-1/e)^{j-2}}{\alpha e^j} (\alpha e^{j-1} - \alpha e^{j-2}) + \frac{b(1-1/e)^{j^*-1}}{\alpha e^{j^*+1}} (\beta e^{j^*} - \alpha e^{j^*-1}) \right] \frac{1}{\alpha} d\alpha \\ &\quad + \int_b^{\beta} \left[ \frac{b}{\alpha e^j} \alpha + \sum_{j=2}^{j^*+1} \frac{b(1-1/e)^{j-2}}{\alpha e^j} (\alpha e^{j-1} - \alpha e^{j-2}) + \frac{b(1-1/e)^{j^*}}{\alpha e^{j^*+2}} (\beta e^{j^*} - \alpha e^{j^*}) \right] \frac{1}{\alpha} d\alpha \\ &= \int_{\beta}^{be} \left[ \frac{b}{e} + b(1 - \frac{1}{e} - (1 - \frac{1}{e})^{j^*}) + \frac{b(e-1)^{j^*-1}}{e^{j^*+1}} \frac{\beta e - \alpha}{\alpha} \right] \frac{1}{\alpha} d\alpha \\ &\quad + \int_b^{\beta} \left[ \frac{b}{e} + b(1 - \frac{1}{e} - (1 - \frac{1}{e})^{j^*+1}) + \frac{b(e-1)^{j^*}}{e^{j^*+2}} \frac{\beta - \alpha}{\alpha} \right] \frac{1}{\alpha} d\alpha \\ &= \frac{b}{e} + b(1 - \frac{1}{e}) - b(1 - \frac{1}{e})^{j^*} \ln \frac{be}{\beta} - b(1 - \frac{1}{e})^{j^*+1} \ln \frac{\beta}{b} \\ &\quad + \frac{b(e-1)^{j^*-1}}{e^{j^*+1}} \left[ e - \frac{\beta}{b} + \ln \frac{\beta}{be} \right] + \frac{b(e-1)^{j^*}}{e^{j^*+2}} \left[ \frac{\beta}{b} - 1 + \ln \frac{b}{\beta} \right] \\ &\leq b - b(1 - \frac{1}{e})^{j^*+1} \ln \frac{be}{b} + \frac{b(e-1)^{j^*}}{e^{j^*+2}} \left[ \frac{be}{b} - 1 + \ln \frac{b}{be} \right] \quad \text{increasing with } \beta \ (\beta = be) \\ &= b - b(1 - \frac{1}{e})^{j^*+1} + \frac{b(e-1)^{j^*}(e-2)}{e^{j^*+2}} \leq b \end{aligned}$$

### C.2. Proof of Theorem 3.1: Competitive Ratio

*Proof.* In what follows we derive the competitive ratio under each subroutine.

718

724

726

728

731

734

736

738

751

754

756

758

760

764

766

768

Recall that τ ∗ represents the true number of available risk times at risk level k, we assume τ <sup>∗</sup> = βe<sup>j</sup> ∗ , where j <sup>∗</sup> ∈ <sup>Z</sup> <sup>+</sup> and β ∈ [b, be]. It's evident that when T ≤ be, j <sup>∗</sup> = 0 follows naturally.

Define η = T /(e − 1). Let us first consider the case where η ≤ b, leading to T ≤ b(e − 1). Given that p<sup>i</sup> = b min(T ,τ˜(e−1)) , it follows that the algorithm consistently sets p<sup>i</sup> = b T . Consequently, we have

$$\mathbb{E}[\text{SOL}] = \frac{b}{T}\beta \geq \frac{b}{b(e-1)}\beta \geq \frac{b}{e-1}.$$

Next, let us consider the case where η > b. We focus on two cases: (1) β < η and (2) β ≥ η.

Suppose β < η. When α falls within [b, β], the algorithm initiates with p<sup>i</sup> = b <sup>α</sup>(e−1) with a running length of α, then adjusts to p<sup>i</sup> = b T in the subsequent round with a running length of β − α; when α falls in the range of [β, η], the algorithm initiates with p<sup>i</sup> = b <sup>α</sup>(e−1) with a running length of β and stops on this stage; otherwise, it consistently uses p<sup>i</sup> = b <sup>T</sup> with a running length of β. Therefore, the expected solution is

$$\begin{aligned} \mathbb{E}[\text{SOL}] &= \int_{\eta}^{be} \frac{b}{T} \beta \frac{1}{\alpha} d\alpha + \int_b^{\beta} \left[ \frac{b}{\alpha(e-1)} \alpha + \frac{b}{T} (\beta - \alpha) - \sigma \ln \frac{T}{\alpha(e-1)} \right] \frac{1}{\alpha} d\alpha \\ &\quad + \int_{\beta}^{\eta} \frac{b}{\alpha(e-1)} \beta \frac{1}{\alpha} d\alpha \\ &= \frac{b\beta}{T} \ln \frac{be}{\eta} + \frac{b}{e-1} \ln \frac{\beta}{b} + \frac{b\beta}{T} \ln \frac{\beta}{b} - \frac{b}{T} (\beta - b) + \frac{\sigma}{2} \left( \ln\left(\frac{T}{\beta(e-1)}\right)^2 - \ln\left(\frac{T}{b(e-1)}\right)^2 \right) \\ &\quad + \frac{b\beta}{e-1} \left( \frac{1}{\beta} - \frac{1}{\eta} \right) \\ &\geq \frac{b^2}{T} \ln \frac{be(e-1)}{T} + \frac{b}{e-1} - \frac{b^2}{T} \quad \text{increasing with } \beta \ (\beta = b) \\ &\geq \frac{b}{e} \ln(e-1) + \frac{b}{e(e-1)} \quad \text{decreasing with } T \ (T = be). \end{aligned}$$

Suppose β ≥ η. It follows that the algorithm always proceeds to the second round. When α falls within [b, η], the algorithm initiates with p<sup>i</sup> = b <sup>α</sup>(e−1) with a running length of α, then adjusts to p<sup>i</sup> = b T in the subsequent round with a running length of β − α; otherwise, it consistently uses p<sup>i</sup> = b <sup>T</sup> with a running length of β. Consequently, we have

$$\begin{aligned}\mathbb{E}[\text{SOL}] &= \int_{\eta}^{be} \frac{b}{T} \beta \frac{1}{\alpha} d\alpha + \int_b^{\eta} \left[ \frac{b}{\alpha(e-1)} \alpha + \frac{b}{T} (\beta - \alpha) - \sigma \ln \frac{T}{\alpha(e-1)} \right] \frac{1}{\alpha} d\alpha \\ &= \frac{b\beta}{T} \ln \frac{be}{\eta} + \frac{b}{e-1} \ln \frac{\eta}{b} + \frac{b\beta}{T} \ln \frac{\eta}{b} - \frac{b}{T} (\eta - b) + \frac{\sigma}{2} (\ln(\frac{T}{\eta(e-1)})^2 - \ln(\frac{T}{b(e-1)})^2) \\ &\geq \frac{b}{e-1} \ln \frac{be}{\eta} + \frac{2b}{e-1} \ln \frac{T}{b(e-1)} - \frac{b}{e-1} + \frac{b^2}{T} - \frac{\sigma}{2} \ln(\frac{T}{b(e-1)})^2 \\ &\quad \text{increasing with } \beta \ (\beta = \eta = T/(e-1)) \\ &\geq \frac{2b}{e-1} - \frac{b}{e-1} \ln(e-1) - \frac{b}{e(e-1)} - \frac{\sigma}{2} \ln(\frac{e}{e-1})^2 \quad \text{decreasing with } T \ (T = be).\end{aligned}$$

## Subroutine 2

Recall that for be < T ≤ be<sup>2</sup> , j ∗ is restricted to being either 0 or 1. Below we separately consider these two cases.

Suppose j <sup>∗</sup> = 0. When α falls in the range of [b, β], Algorithm 1 begins with p<sup>i</sup> = b <sup>α</sup>(e−1) with a running length of α, then transitions to p<sup>i</sup> = b αe(e−1) with a running length of β − α. Otherwise, Algorithm 1 begins with p<sup>i</sup> = b <sup>α</sup>(e−1) with a running

774

776

778

794

796

800

804

806

808

length of β and stops. It follows that

$$\begin{aligned}\mathbb{E}[\text{SOL}] &= \int_{\beta}^{be} \frac{b}{\alpha(e-1)} \beta \frac{1}{\alpha} d\alpha + \int_b^{\beta} \left[ \frac{b}{\alpha(e-1)} \alpha + \frac{b}{\alpha e(e-1)} (\beta - \alpha) - \sigma \ln(e) \right] \frac{1}{\alpha} d\alpha \\ &= \frac{b\beta}{e-1} \left( \frac{1}{\beta} - \frac{1}{be} \right) + \frac{b}{e-1} (\ln \beta - \ln b) + \frac{b\beta}{e(e-1)} \left( \frac{1}{b} - \frac{1}{\beta} \right) \\ &\quad - \frac{b}{e(e-1)} (\ln \beta - \ln b) - \sigma \ln \frac{\beta}{b} \\ &= \frac{b}{e} + \frac{b}{e} \ln \frac{\beta}{b} - \sigma \ln \frac{\beta}{b} \\ &\geq \frac{b}{e} \quad \text{increasing with } \beta \ (\beta = b).\end{aligned}$$

Suppose j <sup>∗</sup> = 1. When α falls in the range of [b, β], Algorithm 1 begins with p<sup>i</sup> = b <sup>α</sup>(e−1) with a running length of α, transitions to p<sup>i</sup> = b αe(e−1) with a running length of αe − α, and then continues with p<sup>i</sup> = b e <sup>3</sup> with a running length of βe − αe. Otherwise, Algorithm 1 begins with p<sup>i</sup> = b <sup>α</sup>(e−1) with a running length of α, then transitions to p<sup>i</sup> = b αe(e−1) with a running length of βe − α. Therefore, the expected solution is

$$\begin{aligned}\mathbb{E}[\text{SOL}] &= \int_{\beta}^{be} \left[ \frac{b}{\alpha(e-1)}\alpha + \frac{b}{\alpha e(e-1)}(\beta e - \alpha) - \sigma \ln e \right] \frac{1}{\alpha} d\alpha \\ &\quad + \int_b^{\beta} \left[ \frac{b}{\alpha(e-1)}\alpha + \frac{b}{\alpha e(e-1)}(\alpha e - \alpha) + \frac{b}{\alpha e^3}(\beta e - \alpha e) - \sigma \ln \frac{e^3}{e-1} \right] \frac{1}{\alpha} d\alpha \\ &= \frac{b}{e-1} \ln \frac{be}{\beta} + \frac{b\beta}{e-1} \left( \frac{1}{\beta} - \frac{1}{be} \right) - \frac{b}{e(e-1)} \ln \frac{be}{\beta} - \sigma \ln \frac{be}{\beta} \\ &\quad + \frac{b}{e-1} \ln \frac{\beta}{b} + \frac{b}{e} \ln \frac{\beta}{b} + \frac{b\beta}{e^2} \left( \frac{1}{b} - \frac{1}{\beta} \right) - \frac{b}{e^2} \ln \frac{\beta}{b} - \sigma \ln \frac{e^3}{e-1} \ln \frac{\beta}{b} \\ &= \frac{b}{e} \ln \frac{be}{\beta} + \frac{b}{e-1} - \frac{\beta}{e(e-1)} - \sigma \ln \frac{be}{\beta} \\ &\quad + \frac{b}{e-1} \ln \frac{\beta}{b} + \frac{b}{e} \ln \frac{\beta}{b} + \frac{\beta}{e^2} - \frac{b}{e^2} - \frac{b}{e^2} \ln \frac{\beta}{b} - \sigma \ln \frac{e^3}{e-1} \ln \frac{\beta}{b} \\ &\geq \frac{b}{e} + \frac{b}{e-1} - \frac{b}{e(e-1)} - \sigma \quad \text{increasing with } \beta \ (\beta = b) \\ &= \frac{2b}{e} - \sigma.\end{aligned}$$

#### Subroutine 3

In the scenarios where T > be<sup>2</sup> , we consider two cases: (1) j <sup>∗</sup> ≥ 1 and (2) j <sup>∗</sup> = 0.

Let us first consider the case where j <sup>∗</sup> ≥ 1. If α ≥ β, the algorithm stops at the j <sup>∗</sup> + 1th round by design of the algorithm (αe<sup>j</sup> ∗ ≥ βe<sup>j</sup> ∗ ); on the other hand, if α < β, the algorithm stops at the j <sup>∗</sup> + 2th round (αe<sup>j</sup> <sup>∗</sup>+1 ≥ βe<sup>j</sup> ∗ ). The objective function when α ≥ β is

$$\begin{aligned} \text{SOL}_1 &= \sum_{j=1}^{j^*} \frac{b\left(1 - \frac{1}{e}\right)^{j-2}}{\alpha e^j} \left(\alpha e^{j-1} - \alpha e^{j-2}\right) + \frac{b\left(1 - \frac{1}{e}\right)^{j^*-1}}{\alpha e^{j^*+1}} \left(\beta e^{j^*} - \alpha e^{j^*-1}\right) - \sigma \ln \frac{e^{2j^*-1}}{(e-1)^{j^*-1}} \\ &= \sum_{j=1}^{j^*} \frac{b(e-1)^{j-1}}{e^j} + \frac{b(e-1)^{j^*-1}}{e^{j^*+1}} \frac{\beta e - \alpha}{\alpha} - \sigma \ln \frac{e^{2j^*-1}}{(e-1)^{j^*-1}} \\ &= b\left(1 - \left(1 - \frac{1}{e}\right)^{j^*}\right) + \frac{b(e-1)^{j^*-1}}{e^{j^*+1}} \frac{\beta e - \alpha}{\alpha} - \sigma \ln \frac{e^{2j^*-1}}{(e-1)^{j^*-1}}. \end{aligned}$$

828

831

834

836

838

854

856

858

860

864

866

868

874

876

The objective function when α < β is

$$\begin{aligned} \text{SOL}_2 &= \sum_{j=1}^{j^*+1} \frac{b\left(1 - \frac{1}{e}\right)^{j-2}}{\alpha e^j} \left(\alpha e^{j-1} - \alpha e^{j-2}\right) + \frac{b\left(1 - \frac{1}{e}\right)^{j^*}}{\alpha e^{j^*+2}} \left(\beta e^{j^*} - \alpha e^{j^*}\right) - \sigma \ln \frac{e^{2j^*+1}}{(e-1)^{j^*}} \\ &= \sum_{j=1}^{j^*+1} \frac{b(e-1)^{j-1}}{e^j} + \frac{b(e-1)^{j^*}}{e^{j^*+2}} \frac{\beta - \alpha}{\alpha} - \sigma \ln \frac{e^{2j^*+1}}{(e-1)^{j^*}} \\ &= b\left(1 - \left(1 - \frac{1}{e}\right)^{j^*+1}\right) + \frac{b(e-1)^{j^*}}{e^{j^*+2}} \frac{\beta - \alpha}{\alpha} - \sigma \ln \frac{e^{2j^*+1}}{(e-1)^{j^*}}. \end{aligned}$$

The expected value of our solution is

$$\mathbb{E}[\text{SOL}] = \int_{\beta}^{be} \text{SOL}_1 f(\alpha) d\alpha + \int_b^{\beta} \text{SOL}_2 f(\alpha) d\alpha. \quad (5)$$

Notice that

$$\begin{aligned} \int_{\beta}^{be} \text{SOL}_1 f(\alpha) d\alpha &= b \left( 1 - \left( 1 - \frac{1}{e} \right)^{j^*} \right) \int_{\beta}^{be} \frac{1}{\alpha} d\alpha + \frac{b(e-1)^{j^*-1}}{e^{j^*+1}} \int_{\beta}^{be} \frac{\beta e - \alpha}{\alpha} \frac{1}{\alpha} d\alpha \\ &\quad - \left[ \sigma \ln \frac{e^{2j^*-1}}{(e-1)^{j^*-1}} \right] \int_{\beta}^{be} \frac{1}{\alpha} d\alpha \\ &= b \left( 1 - \left( 1 - \frac{1}{e} \right)^{j^*} \right) \ln \frac{be}{\beta} + \frac{b(e-1)^{j^*-1}}{e^{j^*+1}} \left( e - \frac{\beta}{b} - \ln \frac{be}{\beta} \right) \\ &\quad - \left[ \sigma \ln \frac{e^{2j^*-1}}{(e-1)^{j^*-1}} \right] \ln \frac{be}{\beta}. \end{aligned}$$

and

$$\begin{aligned} \int_b^\beta \text{SOL}_2 f(\alpha) d\alpha &= b \left( 1 - \left( 1 - \frac{1}{e} \right)^{j^*+1} \right) \int_b^\beta \frac{1}{\alpha} d\alpha + \frac{b(e-1)^{j^*}}{e^{j^*+2}} \int_b^\beta \frac{\beta - \alpha}{\alpha} \frac{1}{\alpha} d\alpha \\ &\quad - \left[ \sigma \ln \frac{e^{2j^*+1}}{(e-1)^{j^*}} \right] \int_b^\beta \frac{1}{\alpha} d\alpha \\ &= b \left( 1 - \left( 1 - \frac{1}{e} \right)^{j^*+1} \right) \ln \frac{\beta}{b} + \frac{b(e-1)^{j^*}}{e^{j^*+2}} \left( \frac{\beta}{b} - 1 - \ln \frac{\beta}{b} \right) \\ &\quad - \left[ \sigma \ln \frac{e^{2j^*+1}}{(e-1)^{j^*}} \right] \ln \frac{\beta}{b}. \end{aligned}$$

887 888

890

894

896

898

911

914 915 916

918

924

928

Hence,

$$\begin{aligned} \mathbb{E}[\text{SOL}] &= b \left( 1 - \left( 1 - \frac{1}{e} \right)^{j^*} \right) \ln \frac{be}{\beta} + b \left( 1 - \left( 1 - \frac{1}{e} \right)^{j^*+1} \right) \ln \frac{\beta}{b} \\ &\quad + \frac{b(e-1)^{j^*-1}}{e^{j^*+1}} \left( e - \frac{\beta}{b} - \ln \frac{be}{\beta} \right) + \frac{b(e-1)^{j^*}}{e^{j^*+2}} \left( \frac{\beta}{b} - 1 - \ln \frac{\beta}{b} \right) \\ &\quad - \left[ \sigma \ln \frac{e^{2j^*-1}}{(e-1)^{j^*-1}} \right] \ln \frac{be}{\beta} - \left[ \sigma \ln \frac{e^{2j^*+1}}{(e-1)^{j^*}} \right] \ln \frac{\beta}{b} \\ &= b - b \left( 1 - \frac{1}{e} \right)^{j^*} \left[ \ln \frac{be}{\beta} + \left( 1 - \frac{1}{e} \right) \ln \frac{\beta}{b} \right] \\ &\quad + \frac{b(e-1)^{j^*-1}}{e^{j^*+1}} \left[ e - \frac{\beta}{b} - \ln \frac{be}{\beta} + \frac{e-1}{e} \left( \frac{\beta}{b} - 1 - \ln \frac{\beta}{b} \right) \right] \\ &\quad - \left[ \sigma \ln \frac{e^{2j^*-1}}{(e-1)^{j^*-1}} \right] \ln \frac{be}{\beta} - \left[ \sigma \ln \frac{e^{2j^*+1}}{(e-1)^{j^*}} \right] \ln \frac{\beta}{b} \\ &\geq b - b \left( 1 - \frac{1}{e} \right)^{j^*} + b \left( 1 - \frac{1}{e} \right)^{j^*} \frac{e-2}{e(e-1)} - \sigma \ln \frac{e^{2j^*-1}}{(e-1)^{j^*-1}} \quad \text{increasing with } \beta \ (\beta = b). \end{aligned}$$

Now consider the case where j <sup>∗</sup> = 0. When α falls within [b, β], Algorithm 1 starts with p<sup>i</sup> = b αe with a running length of α, then transitions to p<sup>i</sup> = b αe<sup>2</sup> with a running length of β − α. Otherwise, Algorithm 1 keeps p<sup>i</sup> = b αe for β time points. It follows that

$$\begin{aligned}\mathbb{E}[\text{SOL}] &= \int_{\beta}^{be} \frac{b}{\alpha e} \beta \frac{1}{\alpha} d\alpha + \int_b^{\beta} \left[ \frac{b}{\alpha e} \alpha + \frac{b}{\alpha e^2} (\beta - \alpha) - \sigma \ln e \right] \frac{1}{\alpha} d\alpha \\ &= \frac{b\beta}{e} \left( \frac{1}{\beta} - \frac{1}{be} \right) + \frac{b}{e} \ln \frac{\beta}{b} + \frac{b\beta}{e^2} \left( \frac{1}{b} - \frac{1}{\beta} \right) - \frac{b}{e^2} \ln \frac{\beta}{b} - \sigma \ln \frac{\beta}{b} \\ &= \frac{b}{e} - \frac{\beta}{e^2} + \frac{\beta}{e^2} - \frac{b}{e^2} + \left( \frac{b}{e} - \frac{b}{e^2} \right) \ln \frac{\beta}{b} - \sigma \ln \frac{\beta}{b} \\ &\geq b \left( \frac{1}{e} - \frac{1}{e^2} \right) \quad \text{increasing with } \beta \ (\beta = b).\end{aligned}$$

Tuning parameter selection For Scenario 1) where T ≤ be, the competitive ratio is the

$$\min \left( \frac{1}{e} \left( \ln(e-1) + \frac{1}{e-1} \right), \frac{2}{e-1} - \frac{1}{e-1} \ln(e-1) - \frac{1}{e(e-1)} - \frac{\sigma}{b} (1 - \ln(e-1)) \right).$$

For Scenario 2) where be < T ≤ be<sup>2</sup> , the competitive ratio is

$$\min \left( \frac{1}{e}, \frac{2}{e} - \frac{\sigma}{b} \right).$$

For Scenario 3) where T > be<sup>2</sup> , the competitive ratio is

$$\min \left( \frac{1}{e} - \frac{1}{e^2}, 1 - \left(1 - \frac{1}{e}\right)^{j^*} + \left(1 - \frac{1}{e}\right)^{j^*} \frac{e-2}{e(e-1)} - \frac{\sigma}{b} \ln \frac{e^{2^{j^*-1}}}{(e-1)^{j^*-1}} \right).$$

By restricting the value of σ under each scenario and combining the above results, we establish Theorem [3.2.](#page-4-6) Specifically, when σ = τ <sup>∗</sup> , it can be verified that Theorem [3.2](#page-4-6) holds.

938

954

956

958

971

974

976

978

981 982 983

984 985 Similarly, we assume τ <sup>∗</sup> = L + βe<sup>j</sup> ∗ . Under the condition that δ ≤ b(e + 1), we have j <sup>∗</sup> = 1 or j <sup>∗</sup> = 0. As before, we only consider the worst case j <sup>∗</sup> = 1.

987 988 Let κ = δ e . Note that, when α ∈ [b, κ), Algorithm 2 first sets p<sup>i</sup> = b <sup>L</sup>+αe with a running length of L + α, then transitions to pk,i = b <sup>U</sup> with a running length of L + βe − L − α. However, when α ∈ [κ, be], Algorithm 2 keeps setting p<sup>i</sup> = b <sup>U</sup> with a

# D. Proof for Algorithm 2

#### D.1. Proof of Lemma [4.1:](#page-4-7) Budget constraint

*Proof.* We prove that the budget constraint is satisfied in expectation under each subroutine in Algorithm [2.](#page-4-2)

Subroutine 4 Let us suppose that τ = L + βe<sup>j</sup> ∗ for some j <sup>∗</sup> ∈ <sup>Z</sup> <sup>+</sup> and β ∈ [b, be]. Note that this implicitly implies that τ ≥ L + b, as we only consider the worst case where τ ∗ is large enough. Define δ = U − L. Under the condition U ≤ be or δ ≤ b(e − 1), we have j <sup>∗</sup> = 0.

When δ ≤ b, Algorithm 2 would consistently use p<sup>i</sup> = U , and the budget constraint is satisfied obviously. Now suppose δ > b. When α ∈ [b, β], Algorithm 2 begins by setting p<sup>i</sup> = b <sup>α</sup>+<sup>L</sup> with a running length of L + α and then continues with p<sup>i</sup> = b U for the second round with a running length of L + β − L − α; when α ∈ [β, δ], Algorithm 2 uses p<sup>i</sup> = b <sup>α</sup>+<sup>L</sup> with a running length of L + β and stops; otherwise, the algorithm sets p<sup>i</sup> = U all the time. Therefore, the expected budget is

$$\begin{aligned} \mathbb{E}[\text{Budget}] &= \int_b^\beta \left[ \frac{b}{L+\alpha} (L+\alpha) + \frac{b}{U} (L+\beta-L-\alpha) \right] \frac{1}{\alpha} d\alpha + \int_\beta^\delta \frac{b}{L+\alpha} (L+\beta) \frac{1}{\alpha} d\alpha \\ &\quad + \int_\delta^{be} \frac{b}{U} (L+\beta) \frac{1}{\alpha} d\alpha \\ &= b \ln \frac{\beta}{b} + \frac{b\beta}{U} \ln \frac{\beta}{b} - \frac{b}{U} (\beta-b) + \frac{b(L+\beta)}{L} (\ln \frac{\delta}{\beta} - \ln \frac{L+\delta}{L+\beta}) + \frac{b(L+\beta)}{U} \ln \frac{be}{\delta} \\ &\leq b \ln \frac{U-L}{b} + \frac{b(U-L)}{U} \ln \frac{U-L}{b} - \frac{b}{U} (U-L-b) + \frac{bU}{U} \ln \frac{be}{\delta} \quad \text{increasing with } \beta \ (\beta = U - L) \\ &\leq b \ln(e-1) + \frac{b(b(e-1))}{L+b(e-1)} \ln(e-1) - \frac{b^2}{L+b(e-1)} (e-2) + b \ln \frac{be}{b(e-1)} \\ &\quad \text{increasing with } U \ (U = L + b(e-1)) \\ &\leq b \ln(e-1) + b \frac{e-1}{e} \ln(e-1) - \frac{b}{e} (e-2) + b \ln \frac{e}{e-1} \quad \text{decreasing with } L \ (L = b) \\ &\leq b + b \left( \frac{e-1}{e} \ln(e-1) - \frac{e-2}{e} \right) \\ &\approx b \end{aligned}$$

#### Subroutine 5

990 running length of L + βe. Therefore, the expected budget is

994

996

998

1014

1016

1019

1024

1026

1029

1034

#### 1036 Subroutine 6

1039 Under the assumption of τ <sup>∗</sup> = L + βe<sup>j</sup> ∗ , and given the condition U > be<sup>2</sup> , it is possible for j ∗ to be 0 or to extend to infinity. We only focus on the worst-case scenario, i.e., j <sup>∗</sup> ≥ 1.

1040 1041 1042 1043 1044 When α falls within [b, β], Algorithm 2 stops with p<sup>i</sup> = b 1 − L+α−b L+α(e−1) (1−1/e) ∗ αej∗+2 with a running length of L + βe<sup>j</sup> ∗ − L − αe<sup>j</sup> ∗ . Otherwise, Algorithm 2 stops with p<sup>i</sup> = b 1 − L+α−b L+α(e−1) (1−1/e) ∗−<sup>1</sup> αej∗+1 with a running length of L + βe<sup>j</sup> ∗ − L − αe<sup>j</sup> <sup>∗</sup>−1 ). Therefore, the expected budget is

$$\begin{aligned} \mathbb{E}[\text{Budget}] &= \int_b^\kappa \left[ \frac{b}{L + \alpha e} (L + \alpha) + \frac{b}{U} (L + \beta e - L - \alpha) \right] \frac{1}{\alpha} d\alpha + \int_\kappa^{be} \frac{b}{U} (L + \beta e) \frac{1}{\alpha} d\alpha \\ &= b \left( \ln \frac{\kappa}{b} + \left( \frac{1}{e} - 1 \right) \ln \frac{L + \kappa e}{L + be} \right) + \frac{b}{U} \beta e \ln \frac{\kappa}{b} - \frac{b}{U} (\kappa - b) + \frac{b(\beta e + L)}{U} \ln \frac{be}{\kappa} \\ &\leq b \left( \ln \frac{\kappa}{b} + \left( \frac{1}{e} - 1 \right) \ln \frac{L + \kappa e}{L + be} \right) + \frac{b}{U} (U - L) \ln \frac{\kappa}{b} - \frac{b}{U} (\kappa - b) + b \ln \frac{be}{\kappa} \\ &\quad \text{increasing with } \beta \ (\beta = (U - L)/e) \\ &\leq b + b \left( \frac{1}{e} - 1 \right) \ln \frac{L + b(e + 1)}{L + be} + \frac{b^2}{L + b(e + 1)} (e + 1) \ln \frac{e + 1}{e} - \frac{b^2}{L + b(e + 1)} \left( \frac{e + 1}{e} - 1 \right) \\ &\quad \text{increasing with } U \ (U = L + b(e + 1)) \\ &\leq b + b \left( \frac{1 - e}{e} \ln \frac{e + 2}{e + 1} + \frac{e + 1}{e + 2} \ln \frac{e + 1}{e} - \frac{1}{e(e + 2)} \right) \quad \text{decreasing with } L \ (L = b) \\ &\approx b \end{aligned}$$

$$\begin{aligned}
& \mathbb{E}[\text{Budget}] = \int_{\beta}^{be} \left[ \frac{b}{L + \alpha(e-1)} (L + \alpha) + b \left( 1 - \frac{L + \alpha - b}{L + \alpha(e-1)} \right) \sum_{j=2}^{j^*} \frac{(1 - 1/e)^{j-2}}{\alpha e^j} (\alpha e^{j-1} - \alpha e^{j-2}) \right. \\
& \quad + b \left( 1 - \frac{L + \alpha - b}{L + \alpha(e-1)} \right) \frac{(1 - 1/e)^{j^*-1}}{\alpha e^{j^*+1}} (L + \beta e^{j^*} - L - \alpha e^{j^*-1}) \left. \right] \frac{1}{\alpha} d\alpha \\
& \quad + \int_b^{\beta} \left[ \frac{b}{L + \alpha(e-1)} (L + \alpha) + b \left( 1 - \frac{L + \alpha - b}{L + \alpha(e-1)} \right) \sum_{j=2}^{j^*+1} \frac{(1 - 1/e)^{j-2}}{\alpha e^j} (\alpha e^{j-1} - \alpha e^{j-2}) \right. \\
& \quad + b \left( 1 - \frac{L + \alpha - b}{L + \alpha(e-1)} \right) \frac{(1 - 1/e)^{j^*}}{\alpha e^{j^*+2}} (L + \beta e^{j^*} - L - \alpha e^{j^*}) \left. \right] \frac{1}{\alpha} d\alpha \\
& \leq b \left( \ln \frac{be}{\beta} + \left( \frac{1}{e-1} - 1 \right) \ln \frac{L + be(e-1)}{L + \beta(e-1)} \right) + b(1 - \frac{1}{e} - (1 - \frac{1}{e})^{j^*}) \frac{e-2}{e-1} \ln \frac{L + be(e-1)}{L + \beta(e-1)} \\
& \quad + b(1 - \frac{1}{e} - (1 - \frac{1}{e})^{j^*}) \frac{b}{L} \left( \ln \frac{be}{\beta} - \frac{L + be(e-1)}{L + \beta(e-1)} \right) \\
& \quad + b\beta e^{j^*} \frac{(e-1)^{j^*}}{e^{2j^*}} \frac{1}{L} \left( \ln \frac{be}{\beta} - \ln \frac{L + be(e-1)}{L + \beta(e-1)} \right) - b \frac{(e-1)^{j^*-1}}{e^{j^*+1}} \ln \frac{L + be(e-1)}{L + \beta(e-1)} \\
& \quad + b \left( \ln \frac{\beta}{b} + \left( \frac{1}{e-1} - 1 \right) \ln \frac{L + \beta(e-1)}{L + b(e-1)} \right) + b(1 - \frac{1}{e} - (1 - \frac{1}{e})^{j^*+1}) \frac{e-2}{e-1} \ln \frac{L + \beta(e-1)}{L + b(e-1)} \\
& \quad + b(1 - \frac{1}{e} - (1 - \frac{1}{e})^{j^*}) \frac{b}{L} \left( \ln \frac{\beta}{b} - \frac{L + \beta(e-1)}{L + b(e-1)} \right) \\
& \quad + b\beta e^{j^*} \frac{(e-1)^{j^*+1}}{e^{2j^*+2}} \frac{1}{L} \left( \ln \frac{\beta}{b} - \ln \frac{L + \beta(e-1)}{L + b(e-1)} \right) - b \frac{(e-1)^{j^*}}{e^{j^*+2}} \ln \frac{L + \beta(e-1)}{L + b(e-1)} \\
& \leq b \left( 1 + \left( \frac{1}{e-1} - 1 \right) \ln \frac{L + be(e-1)}{L + b(e-1)} \right) + b(1 - \frac{1}{e} - (1 - \frac{1}{e})^{j^*+1}) \frac{e-2}{e-1} \ln \frac{L + be(e-1)}{L + b(e-1)} \\
& \quad + b(1 - \frac{1}{e} - (1 - \frac{1}{e})^{j^*}) \frac{b}{L} \left( 1 - \frac{L + be(e-1)}{L + b(e-1)} \right) \\
& \quad + b^2 \frac{(e-1)^{j^*+1}}{e^{j^*+1}} \frac{1}{L} \left( 1 - \ln \frac{L + be(e-1)}{L + b(e-1)} \right) - b \frac{(e-1)^{j^*}}{e^{j^*+2}} \ln \frac{L + be(e-1)}{L + b(e-1)} \\
& \text{increasing with } \beta \ (\beta = be) \\
& \leq b \left( 1 + \left( \frac{1}{e-1} - 1 \right) \ln \frac{L + be(e-1)}{L + b(e-1)} \right) + b \frac{e-2}{e} \ln \frac{L + be(e-1)}{L + b(e-1)} \\
& \quad + b(1 - \frac{1}{e}) \frac{b}{L} \left( 1 - \ln \frac{L + be(e-1)}{L + b(e-1)} \right) \\
& \approx b
\end{aligned}$$

1087 Combining the above results with the proof of Subroutine 2, presented in Appendix [C.1,](#page-11-0) establishes Lemma [4.1.](#page-4-7)

#### 1089 1090 D.2. Proof of Theorem [4.2:](#page-5-4) Consistency and Robustness

1091 *Proof.* We begin with the proof of consistency and then proceed to the analysis of robustness.

1092 1093 1094 Consistency Analysis It is straightforward to show that our algorithm is 1- consistent. When the width of the predictive interval is zero, meaning that L = U = τ ∗ , we have

1095 1096 1097

$$\mathbb{E}[\text{SOL}] = \frac{b}{U}\tau^* = b.$$

1104

1106

1109 Suppose τ <sup>∗</sup> < L + b. When α falls within [b, δ], Algorithm 2 assigns p<sup>i</sup> = <sup>L</sup>+<sup>α</sup> with a running length of τ ∗ . Otherwise, Algorithm 2 sets p<sup>i</sup> = b <sup>U</sup> with a running length of τ ∗ . Therefore, the expected solution is

1111

1114

1116

1118 1119

1124

1126

1129

1134

1136

1151

#### Subroutine 4

For cases where δ = U − L ≤ b, the algorithm proceeds with pk,i = b U . Hence, we have

$$\mathbb{E}[\text{SOL}] = \frac{b}{U}\beta \geq \frac{b}{L+b}L \geq \frac{b}{2}.$$

Next, we consider the case where b(e − 1) ≥ δ > b, further divided into τ <sup>∗</sup> < L + b and τ <sup>∗</sup> ≥ L + b.

$$\begin{aligned}\mathbb{E}[\text{SOL}] &= \int_b^\delta \left[ \frac{b}{L+\alpha} \tau^* \right] \frac{1}{\alpha} d\alpha + \int_\delta^{be} \frac{b}{U} \tau^* \frac{1}{\alpha} d\alpha \\ &= \frac{b\tau^*}{L} \left( \ln \frac{\delta}{b} - \ln \frac{L+\delta}{L+b} \right) + \frac{b\tau^*}{U} \ln \frac{be}{\delta} \\ &\geq b \left( \ln \frac{\delta}{b} - \ln \frac{L+\delta}{L+b} \right) + \frac{bL}{U} \ln \frac{be}{\delta} \quad \text{increasing with } \tau^* \ (\tau^* = L) \\ &\geq b \left( \ln(e-1) - \ln \frac{L+b(e-1)}{L+b} \right) + \frac{bL}{L+b(e-1)} \ln \frac{e}{e-1} \\ &\quad \text{decreasing with } U \ (U = L+b(e-1)) \\ &\geq b \left( \ln(e-1) - \ln \frac{e}{2} \right) + b \frac{1}{e} \ln \frac{e}{e-1} \quad \text{increasing with } L \ (L = b) \\ &= b \left( \ln \frac{2(e-1)}{e} + \frac{1}{e} \ln \frac{e}{e-1} \right)\end{aligned}$$

For cases where τ <sup>∗</sup> ≥ L + b, let us suppose τ <sup>∗</sup> = L + βe<sup>j</sup> ∗ where β ∈ [b, be]. Under the condition U ≤ be or δ ≤ b(e − 1), we have j <sup>∗</sup> = 0. Further, since L + β ≤ U, we have β ≤ U − L. When α falls within [b, β], Algorithm 2 starts with p<sup>i</sup> = b <sup>L</sup>+<sup>α</sup> with a running length of L + α, then transitions to p<sup>i</sup> = b <sup>U</sup> with a running length of L + β − L − α; when α falls within [β, δ], Algorithm 2 assigns p<sup>i</sup> = b <sup>L</sup>+<sup>α</sup> with a running length of L + β; otherwise, Algorithm 2 assigns p<sup>i</sup> = b <sup>U</sup> with a running length of L + β. It follows that

$$\begin{aligned} \mathbb{E}[\text{SOL}] &= \int_b^\beta \left[ \frac{b}{L+\alpha}(L+\alpha) + \frac{b}{U}(L+\beta-L-\alpha) - \sigma \ln \frac{U}{L+\alpha} \right] \frac{1}{\alpha} d\alpha \\ &\quad + \int_\beta^\delta \frac{b}{L+\alpha} \frac{1}{\alpha} d\alpha + \int_\delta^{be} \frac{b}{U} \frac{1}{\alpha} d\alpha \\ &= b \ln \frac{\beta}{b} + \frac{b\beta}{U} \ln \frac{\beta}{b} - \frac{b}{U}(\beta-b) - \sigma \ln \frac{e}{2} \ln \frac{\beta}{b} + \frac{b(L+\beta)}{L} \left( \ln \frac{\delta}{\beta} - \ln \frac{L+\delta}{L+\beta} \right) + \frac{b(L+\beta)}{U} \ln \frac{be}{\delta} \\ &\geq \frac{b(L+b)}{L} \left( \ln \frac{U-L}{b} - \ln \frac{U}{L+b} \right) + \frac{b(L+b)}{U} \ln \frac{be}{U-L} \quad \text{increasing with } \beta \ (\beta=b) \\ &\geq \frac{b(L+b)}{L} \left( \ln(e-1) - \ln \frac{L+b(e-1)}{L+b} \right) + \frac{b(L+b)}{L+b(e-1)} \ln \frac{e}{e-1} \\ &\quad \text{decreasing with } U \ (U=L+b(e-1)) \\ &\geq 2b \ln \frac{2(e-1)}{e} + b \frac{2}{e} \ln \frac{e}{e-1} \quad \text{increasing with } L \ (L=b) \\ &= b(2 \ln \frac{2(e-1)}{e} + \frac{2}{e} \ln \frac{e}{e-1}) \end{aligned}$$

#### Subroutine 5

1159 1160 1161

1164

1174 First, suppose τ <sup>∗</sup> < L + b. When α falls in the range [b, κ], Algorithm 2 assigns p<sup>i</sup> = b <sup>L</sup>+αe with a running length τ ∗ . Otherwise, Algorithm 2 assigns p<sup>i</sup> = b <sup>U</sup> with a running length τ ∗ . Therefore, the expected solution is

1176

1194

1196

1199 1200

1204 1206 For situations where τ <sup>∗</sup> ≥ L + b, suppose τ <sup>∗</sup> = L + βe<sup>j</sup> ∗ where β ∈ [b, be]. Below we separately consider two cases: 1) j <sup>∗</sup> ≥ 1 or β ≥ κ, and 2) j <sup>∗</sup> = 0 and β < κ.

1209 Suppose case 1) where j <sup>∗</sup> ≥ 1 or β ≥ κ. When α falls within [b, κ], Algorithm 2 first assigns p<sup>i</sup> = b L+αe for a time length of L + α, and then proceeds with pk,i = b <sup>U</sup> with a running length <sup>L</sup> <sup>+</sup> βe<sup>j</sup> ∗ − L − α. Otherwise, Algorithm 2 assigns

have

$$\mathbb{E}[\text{SOL}] = \frac{b}{U}\tau^* \geq \frac{b}{L+be}L \geq \frac{b}{e+1}.$$

Next, we consider the case where b(e + 1) ≥ δ > be, further divided into τ <sup>∗</sup> < L + b and τ <sup>∗</sup> ≥ L + b.

$$\begin{aligned}\mathbb{E}[\text{SOL}] &= \int_b^\kappa \left[ \frac{b}{L + \alpha e} \tau^* \right] \frac{1}{\alpha} d\alpha + \int_\kappa^{be} \frac{b}{U} \tau^* \frac{1}{\alpha} d\alpha \\ &= \frac{b\tau^*}{L} \left( \ln \frac{\kappa}{b} - \ln \frac{L + \kappa e}{L + be} \right) + \frac{b\tau^*}{U} \ln \frac{be}{\kappa} \\ &\geq b \left( \ln \frac{\kappa}{b} - \ln \frac{L + \kappa e}{L + be} \right) + \frac{bL}{U} \ln \frac{be}{\kappa} \quad \text{increasing with } \tau^* \ (\tau^* = L) \\ &\geq b \left( \ln \frac{e+1}{e} - \ln \frac{L + b(e+1)}{L + be} \right) + \frac{bL}{L + b(e+1)} \ln \frac{e^2}{e+1} \\ &\quad \text{decreasing with } U \ (U = L + b(e+1)) \\ &\geq b \left( \ln \frac{e+1}{e} - \ln \frac{e^2}{e^2-1} \right) + b \frac{e^2 - e - 1}{e^2} \ln \frac{e^2}{e+1} \\ &\quad \text{increasing with } L \ (L = b(e^2 - e - 1)) \\ &= b \left( \ln \frac{e+1}{e} - \ln \frac{e^2}{e^2-1} + \frac{e^2 - e - 1}{e^2} \ln \frac{e^2}{e+1} \right)\end{aligned}$$

1216

1218 1219

1224

1226

1229

1234

1236

1254

1256

1259 1260

1264 Next, consider case 2) where j <sup>∗</sup> = 0 and β < κ. When α falls within [b, β], Algorithm 2 starts with p<sup>i</sup> = b <sup>L</sup>+αe with a running length of L + α, then transits to p<sup>i</sup> = b <sup>U</sup> with a running length of L + β − L − α; when α ∈ [β, κ], Algorithm 2 sets p<sup>i</sup> = b <sup>L</sup>+αe with a running length of L + β; otherwise, Algorithm 2 sets p<sup>i</sup> = b <sup>U</sup> with a running length of L + β. Therefore,

p<sup>i</sup> = b <sup>U</sup> with a running length of <sup>L</sup> <sup>+</sup> βe<sup>j</sup> ∗ . Therefore, the expected solution is

$$\begin{aligned} \mathbb{E}[\text{SOL}] &= \int_b^\kappa \left[ \frac{b}{L + \alpha e} (L + \alpha) + \frac{b}{U} (\beta e^{j^*} + L - L - \alpha) - \sigma \ln \frac{U}{L + \alpha e} \right] \frac{1}{\alpha} d\alpha \\ &\quad + \int_\kappa^{be} \frac{b}{U} (\beta e^{j^*} + L) \frac{1}{\alpha} d\alpha \\ &\geq b \left( \ln \frac{\kappa}{b} + \left( \frac{1}{e} - 1 \right) \ln \frac{L + \kappa e}{L + be} \right) + \frac{b}{U} \beta e^{j^*} \ln \frac{\kappa}{b} - \frac{b}{U} (\kappa - b) - \sigma \ln \frac{e + 2}{e + 1} \ln \frac{\kappa}{b} \\ &\quad + \frac{b(\beta e^{j^*} + L)}{U} \ln \frac{be}{\kappa} \\ &\geq b \left( \ln \frac{\kappa}{b} + \left( \frac{1}{e} - 1 \right) \ln \frac{L + \kappa e}{L + be} \right) + \frac{b}{U} \kappa \ln \frac{\kappa}{b} - \frac{b}{U} (\kappa - b) - \sigma \ln \frac{e + 2}{e + 1} \ln \frac{\kappa}{b} \\ &\quad + \frac{b(\kappa + L)}{U} \ln \frac{be}{\kappa} \quad \text{increasing with } \beta, j^* \quad (\beta = \kappa, j^* = 0) \\ &\geq b \left( \ln \frac{e + 1}{e} - \frac{e - 1}{e} \ln \frac{L + b(e + 1)}{L + be} \right) + \frac{b^2(e + 1)}{L e + be(e + 1)} \ln \frac{e + 1}{e} - \frac{b}{L + b(e + 1)} \left( \frac{b(e + 1)}{e} - b \right) \\ &\quad - \sigma \ln \frac{e + 2}{e + 1} \ln \frac{e + 1}{e} + \frac{b(L + \frac{b(e + 1)}{e})}{L + b(e + 1)} \ln \frac{e^2}{e + 1} \quad \text{decreasing with } U \quad (U = L + b(e + 1)) \\ &\geq b \left( \ln \frac{e + 1}{e} - \frac{e - 1}{e} \ln \frac{e^2}{e^2 - 1} \right) + \frac{b(e + 1)}{e^3} \ln \frac{e + 1}{e} - b \frac{1}{e^2} \frac{1}{e} - \sigma \ln \frac{e + 2}{e + 1} \ln \frac{e + 1}{e} \\ &\quad + b \frac{e^2 - e - 1 + \frac{(e + 1)}{e}}{e^2} \ln \frac{e^2}{e + 1} \quad \text{increasing with } L \quad (L = b(e^2 - e - 1)) \\ &= b \left( \ln \frac{e + 1}{e} - \frac{1}{e^3} - \frac{e - 1}{e} \ln \frac{e^2}{e^2 - 1} + \frac{e^2 - e - 1}{e^2} \ln \frac{e^2}{e + 1} \right) + b \frac{1 + \frac{1}{e}}{e^2} \\ &\quad - \sigma \ln \frac{e + 2}{e + 1} \ln \frac{e + 1}{e} \\ &= b \left( \left( 1 + \frac{1}{e^2} \right) \ln(e + 1) - 1 - \frac{1}{e^2} + \frac{e - 1}{e} \ln(e - 1) \right) - \sigma \ln \frac{e + 2}{e + 1} \ln \frac{e + 1}{e} \end{aligned}$$

1269

1274

1276

1279

1289 1290

1294

1296

1306 1307

1309

1314

1316

we have

$$\begin{aligned} \mathbb{E}[\text{SOL}] &= \int_b^\beta \left[ \frac{b}{L + \alpha e} (L + \alpha) + \frac{b}{U} (\beta + L - L - \alpha) - \sigma \ln \frac{U}{L + \alpha e} \right] \frac{1}{\alpha} d\alpha \\ &\quad + \int_\beta^\kappa \frac{b}{L + \alpha e} (L + \beta) \frac{1}{\alpha} d\alpha + \int_\kappa^{be} \frac{b}{U} (L + \beta) \frac{1}{\alpha} d\alpha \\ &= b \left( \ln \frac{\beta}{b} + \left( \frac{1}{e} - 1 \right) \ln \frac{L + \beta e}{L + be} \right) + \frac{b}{U} \beta \ln \frac{\beta}{b} - \frac{b}{U} (\beta - b) - \sigma \ln \frac{e^2}{e^2 - 1} \ln \frac{\beta}{b} \\ &\quad + \frac{b(L + \beta)}{L} \left( \ln \frac{\kappa}{\beta} - \ln \frac{L + \kappa e}{L + \beta e} \right) + \frac{b(L + \beta)}{U} \ln \frac{be}{\kappa} \\ &\geq \frac{b(L + \beta)}{L} \left( \ln \frac{\kappa}{\beta} - \ln \frac{L + \kappa e}{L + \beta e} \right) + \frac{b(L + \beta)}{U} \ln \frac{be}{\kappa} \quad \text{increasing with } \beta \ (\beta = b) \\ &\geq \frac{b(L + b)}{L} \left( \ln \frac{b(e + 1)}{be} - \ln \frac{L + b(e + 1)}{L + be} \right) + \frac{b(L + b)}{L + b(e + 1)} \ln \frac{e^2}{e + 1} \\ &\quad \text{decreasing with } U \ (U = L + b(e + 1)) \\ &\geq b \frac{e^2 - e}{e^2 - e - 1} \left( \ln \frac{e + 1}{e} - \ln \frac{e^2}{e^2 - 1} \right) + \frac{e^2 - e}{e^2} \ln \frac{e^2}{e + 1} \\ &\quad \text{increasing with } L \ (L = b(e^2 - e - 1)) \\ &\geq b \left( \frac{e^2 - e}{e^2 - e - 1} \ln \frac{(e + 1)^2(e - 1)}{e^3} + \frac{e - 1}{e} \ln \frac{e^2}{e + 1} \right). \end{aligned}$$

#### Subroutine 6

In this scenario, our algorithm initiates with pk,i = b <sup>L</sup>+α(e−1) , subsequently updating τ˜ and b after each iteration. We analyze two cases: one where τ <sup>∗</sup> < L + b, and the other where τ <sup>∗</sup> ≥ L + b.

For the first situation where τ <sup>∗</sup> < L + b, Algorithm 2 consistently sets p<sup>i</sup> = b <sup>L</sup>+α(e−1) . Therefore, we have

$$\begin{aligned}\mathbb{E}[\text{SOL}] &= \int_b^{be} \frac{b}{L + \alpha(e - 1)} \tau^* \frac{1}{\alpha} d\alpha \\ &= \frac{b\tau^*}{L} \left( \ln \frac{be}{b} - \ln \frac{L + be(e - 1)}{L + b(e - 1)} \right) \\ &\geq \tau^* \left( 1 - \ln \frac{e^2 - e + 1}{e} \right) \quad \text{increasing with } L \ (L = b) \\ &\geq b \left( 2 - \ln(e^2 - e + 1) \right)\end{aligned}$$

Next, we consider the case where τ <sup>∗</sup> ≥ L + b. Suppose that τ <sup>∗</sup> = L + βe<sup>j</sup> ∗ where β ∈ [b, be].

When j <sup>∗</sup> ≥ 1, the objective function when α ≥ β is

$$\begin{aligned} \text{SOL}_1 = & \frac{b}{L + \alpha(e-1)}(L + \alpha) + b \left( 1 - \frac{L + \alpha - b}{L + \alpha(e-1)} \right) \sum_{j=2}^{j^*} \frac{(1 - 1/e)^{j-2}}{\alpha e^j} (\alpha e^{j-1} - \alpha e^{j-2}) \\ & + b \left( 1 - \frac{L + \alpha - b}{L + \alpha(e-1)} \right) \frac{(1 - 1/e)^{j^*-1}}{\alpha e^{j^*}} (L + \beta e^{j^*} - L - \alpha e^{j^*-1}) \\ & - \sigma \ln \frac{\alpha e^{2j^*+1}}{(\alpha(e-2) + b)(e-1)^{j^*-1}} \end{aligned}$$

The objective function when α < β is

$$\begin{aligned} \text{SOL}_2 = & \frac{b}{L + \alpha(e-1)}(L + \alpha) + b \left( 1 - \frac{L + \alpha - b}{L + \alpha(e-1)} \right) \sum_{j=2}^{j^*+1} \frac{(1-1/e)^{j-2}}{\alpha e^j} (\alpha e^{j-1} - \alpha e^{j-2}) \\ & + b \left( 1 - \frac{L + \alpha - b}{L + \alpha(e-1)} \right) \frac{(1-1/e)^{j^*}}{\alpha e^{j^*+2}} (L + \beta e^{j^*} - L - \alpha e^{j^*}) \\ & - \sigma \ln \frac{\alpha e^{2j^*+2}}{(\alpha(e-2) + b)(e-1)^{j^*}}. \end{aligned}$$

$$\begin{aligned} 1376 & \mathbb{E}[\text{SOL}] = \int_{\beta}^{be} \text{SOL}_1 \frac{1}{\alpha} d\alpha + \int_b^{\beta} \text{SOL}_2 \frac{1}{\alpha} d\alpha \\ 1377 & \geq b \left( \ln \frac{be}{\beta} + \left( \frac{1}{e-1} - 1 \right) \ln \frac{L + be(e-1)}{L + \beta(e-1)} \right) + b \left( 1 - \frac{1}{e} - \left( 1 - \frac{1}{e} \right)^{j^*} \right) \frac{e-2}{e-1} \ln \frac{L + be(e-1)}{L + \beta(e-1)} \\ 1380 & + b^2 \left( 1 - \frac{1}{e} - \left( 1 - \frac{1}{e} \right)^{j^*} \right) \frac{1}{L} \left( \ln \frac{be}{\beta} - \ln \frac{L + be(e-1)}{L + \beta(e-1)} \right) \\ 1381 & + b^2 \left( 1 - \frac{1}{e} - \left( 1 - \frac{1}{e} \right)^{j^*} \right) \frac{1}{L} \\ 1382 & + b \beta e^{j^*} \frac{(e-1)^{j^*-1}}{e^{2j^*}} \frac{e-2}{L} \left( \ln \frac{be}{\beta} - \ln \frac{L + be(e-1)}{L + \beta(e-1)} \right) - b \frac{(e-1)^{j^*-1}}{e^{j^*+1}} \frac{e-2}{e-1} \ln \frac{L + be(e-1)}{L + \beta(e-1)} \\ 1385 & - \sigma \ln \frac{e^{2j^*+1}}{(e-1)^{j^*+1}} \ln \frac{be}{\beta} \\ 1386 & + b \left( \ln \frac{\beta}{b} + \left( \frac{1}{e-1} - 1 \right) \ln \frac{L + \beta(e-1)}{L + b(e-1)} \right) + b \left( 1 - \frac{1}{e} - \left( 1 - \frac{1}{e} \right)^{j^*+1} \right) \frac{e-2}{e-1} \ln \frac{L + \beta(e-1)}{L + b(e-1)} \\ 1389 & + b^2 \left( 1 - \frac{1}{e} - \left( 1 - \frac{1}{e} \right)^{j^*+1} \right) \frac{1}{L} \left( \ln \frac{\beta}{b} - \ln \frac{L + \beta(e-1)}{L + b(e-1)} \right) \\ 1390 & + b \beta e^{j^*} \frac{(e-1)^{j^*}}{e^{2j^*+2}} \frac{e-2}{L} \left( \ln \frac{\beta}{b} - \ln \frac{L + \beta(e-1)}{L + b(e-1)} \right) - b \frac{(e-1)^{j^*}}{e^{j^*+2}} \frac{e-2}{e-1} \ln \frac{L + \beta(e-1)}{L + b(e-1)} \\ 1391 & - \sigma \ln \frac{e^{2j^*+3}}{(e-1)^{j^*+2}} \ln \frac{\beta}{b} \\ 1392 & \geq b \left( 1 + \left( \frac{1}{e-1} - 1 \right) \ln \frac{L + be(e-1)}{L + b(e-1)} \right) + b \left( 1 - \frac{1}{e} - \left( 1 - \frac{1}{e} \right)^{j^*} \right) \frac{e-2}{e-1} \ln \frac{L + be(e-1)}{L + b(e-1)} \\ 1400 & + b^2 \left( 1 - \frac{1}{e} - \left( 1 - \frac{1}{e} \right)^{j^*} \right) \frac{1}{L} \left( 1 - \ln \frac{L + be(e-1)}{L + b(e-1)} \right) \\ 1401 & + b^2 e^{j^*} \frac{(e-1)^{j^*-1}}{e^{2j^*}} \frac{e-2}{L} \left( 1 - \ln \frac{L + be(e-1)}{L + b(e-1)} \right) - b \frac{(e-1)^{j^*-1}}{e^{j^*+1}} \frac{e-2}{e-1} \ln \frac{L + be(e-1)}{L + b(e-1)} \\ 1402 & - \sigma \ln \frac{e^{2j^*+1}}{(e-1)^{j^*+1}} \quad \text{increasing with } \beta \ (\beta = b) \\ 1403 & \geq b \left( 1 + \left( \frac{1}{e-1} - 1 \right) \ln \frac{b(e^2 - e + 1)}{b + b(e-1)} \right) + b \left( 1 - \frac{1}{e} - \left( 1 - \frac{1}{e} \right)^{j^*} \right) \frac{e-2}{e-1} \ln \frac{b(e^2 - e + 1)}{b + b(e-1)} \\ 1404 & + b \left( 1 - \frac{1}{e} - \left( 1 - \frac{1}{e} \right)^{j^*} \right) \left( 1 - \ln \frac{e^2 - e + 1}{e} \right) \\ 1405 & + b e^{j^*} \frac{(e-1)^{j^*-1}}{e^{2j^*}} (e-2) \left( 1 - \ln \frac{b(e^2 - e + 1)}{b + b(e-1)} \right) - b \frac{(e-1)^{j^*-1}}{e^{j^*+1}} \frac{e-2}{e-1} \ln \frac{e^2 - e + 1}{e} \\ 1406 & - \sigma \ln \frac{e^{2j^*+1}}{(e$$

The expected solution is

1430 1431 1432 L + α, then transitions to p<sup>i</sup> = b 1 − L+α−b L+α(e−1) 1 αe<sup>2</sup> with a running length of L + β − L − α. Otherwise, Algorithm 2 consistently sets p<sup>i</sup> = b <sup>L</sup>+α(e−1) . Therefore, we have

$$\begin{aligned} 1433 \quad & \mathbb{E}[\text{SOL}] = \int_b^\beta \left[ \frac{b}{L + \alpha(e-1)} (L + \alpha) + b \left( 1 - \frac{L + \alpha - b}{L + \alpha(e-1)} \right) \frac{1}{\alpha e^2} (L + \beta - L - \alpha) \right. \\ & - \sigma \ln \frac{\alpha e^2}{\alpha(e-2) + b} \left. \right] \frac{1}{\alpha} d\alpha + \int_\beta^{be} \frac{b}{L + \alpha(e-1)} (L + \beta) \frac{1}{\alpha} d\alpha \\ 1434 \quad & \geq b \left( \ln \frac{\beta}{b} - \frac{e-2}{e-1} \ln \frac{L + \beta(e-1)}{L + b(e-1)} \right) - \frac{b\beta}{e^2} \frac{e-2}{L} \left( \ln \frac{\beta}{b} - \ln \frac{L + \beta(e-1)}{L + b(e-1)} \right) \\ 1441 \quad & - \frac{b}{e^2} \frac{e-2}{e-1} \ln \frac{L + \beta(e-1)}{L + b(e-1)} - \sigma \ln \frac{e^3}{(e-1)^2} \ln \frac{\beta}{b} + \frac{b(L + \beta)}{L} \left( \ln \frac{be}{\beta} - \ln \frac{L + be(e-1)}{L + \beta(e-1)} \right) \\ 1443 \quad & \geq \frac{b(L + b)}{L} \left( 1 - \ln \frac{L + be(e-1)}{L + b(e-1)} \right) \quad \text{increasing with } \beta \ (\beta = b) \\ 1444 \quad & \\ 1445 \quad & \\ 1446 \quad & \\ 1447 \quad & \geq 2b \left( 1 - \ln \frac{b(e^2 - e + 1)}{b + b(e-1)} \right) \quad \text{increasing with } L \ (L = b) \\ 1448 \quad & \\ 1449 \quad & = 2b \left( 2 - \ln(e^2 - e + 1) \right). \end{aligned}$$

1450

1451 1452 Tuning parameter selection For Scenario 1) where U ≤ be, the competitive ratio is

1453 1454 1455

1456 1457 For Scenario 2) where be < U ≤ be<sup>2</sup> , the competitive ratio is

1462 1463 For Scenario 3) where U > be<sup>2</sup> , the competitive ratio is

$$\begin{aligned} & 1464 & \min \left( 2 - \ln(e^2 - e + 1), 1 + \left( \frac{1}{e - 1} - 1 \right) \ln \frac{e^2 - e + 1}{e} + \left( 1 - \frac{1}{e} - (1 - \frac{1}{e})^{j^*} \right) \frac{e - 2}{e - 1} \ln \frac{e^2 - e + 1}{e} \right. \\ & 1465 & \left. + (1 - \frac{1}{e} - (1 - \frac{1}{e})^{j^*}) \left( 1 - \ln \frac{e^2 - e + 1}{e} \right) + \frac{(e - 1)^{j^* - 1}}{e^{j^*}} (e - 2) \left( 1 - \ln \frac{e^2 - e + 1}{e} \right) \right. \\ & 1466 & \left. - \frac{(e - 1)^{j^* - 1}}{e^{j^* + 1}} \frac{e - 2}{e - 1} \ln \frac{e^2 - e + 1}{e} - \frac{\sigma}{b} \ln \frac{e^{2j^* + 1}}{(e - 1)^{j^* + 1}} \right). \end{aligned}$$

1472

1473 1474 1475 By restricting the value of σ under each scenario and combining the above results, we establish Theorem [4.2.](#page-5-4) Specifically, when σ = τ <sup>∗</sup> , it can be verified that Theorem [4.2](#page-5-4) holds.

#### 1476 1477 E. Additional Synthetic Experiments

1478 1479

1480 1481 1482 1483 1484 In this section, we examine the performances of the algorithms under the learning-augmented setting where τ ∗ is small. Specifically, we set the number of risk occurrences τ <sup>∗</sup> = Int[0.2(T + b)] for scenarios with horizon lengths T = 22 and T = 100, and τ <sup>∗</sup> = Int[0.1(T + b)] for scenario T = 100. Figure [4](#page-27-1) presents the average competitive ratio against a range of prediction interval widths.

$$\ln \frac{2(e-1)}{e} + \frac{1}{e} \ln \frac{e}{e-1}.$$

$$\min \left( \frac{1}{e}, \frac{2}{e} - \frac{\sigma}{b} \right).$$

#### E.1. Performance under Small τ ∗

![](_page_27_Figure_2.jpeg)

#### 1499 1500 E.2. Budget Utilization by Each Algorithm

1504

![](_page_27_Figure_6.jpeg)

![](_page_27_Figure_8.jpeg)

Figure 4. Average competitive ratio under learning-augmented setting with b = 3.

To assess the budget utilization by each algorithm, we eliminate the penalty term from the objective in Problem [1.](#page-2-2) Figures [5](#page-27-2) and [6](#page-27-3) display the average competitive ratios in scenarios without and with learning augmentation, respectively. We note that in Figure [5](#page-27-2) (middle), when τ <sup>∗</sup> = 22, the competitive ratio slightly exceeds 1. This is attributed to our algorithm utilizing a slightly higher budget in expectation. We provide detailed insights into this observation in Section 1 of the Supplementary Material, where we demonstrate that the worst-case budget spent is about 1.047bk, slightly surpassing the allocated budget.

Figure 5. Average competitive ratio under non-learning augmented setting with b = 3.

1541 1542 1543 1544 1545 1546 Our research is inspired by the Heartsteps V1 mobile health study, which aims to enhance physical activity among sedentary individuals [\(Klasnja et al.,](#page-8-18) [2019\)](#page-8-18). The study involved 37 participants over a follow-up period of six weeks, gathering detailed data on step counts on a minute-by-minute basis. To ensure the reliability of the step count data, our analysis was restricted to the hours from 9 am to 9 pm, with a decision time frequency set at five-minute intervals [\(Liao et al.,](#page-8-7) [2018\)](#page-8-7). This led to the accumulation of 1585 instances of 12-hour user-days, with T = 144 decision times per day.

1547 1548 1549 At each decision time t,we define the risk variable R<sup>t</sup> with a binary classification: R<sup>t</sup> = 1 indicates a sedentary state, identified by recording fewer than 150 steps in the prior 40 minutes, and R<sup>t</sup> = 0 signifies a non-sedentary state. Additionally, the availability for intervention, It, is contingent on recent messaging activity: if the user has received an anti-sedentary message within the preceding hour, I<sup>t</sup> is set to 0; otherwise, it is set to 1. We want to distribute b = 1.5 interventions over available sedentary times each day.

1554 1556 We implement four algorithms: our randomized and learning-augmented algorithms (Algorithms [1](#page-3-2) and [2,](#page-4-2) respectively), the SeqRTS strategy proposed by [Liao et al.](#page-8-7) [\(2018\)](#page-8-7), and a benchmark method (b/U). Rather than devising a tailored prediction model, we generate prediction intervals by randomly selecting from a range of [2, 144], which contains τ ∗ , with intervals of varying widths. This approach allows us to assess the performance of different algorithms under varying qualities of forecast accuracy.

1559 1560 1561 We adopt the SeqRTS method to include prediction intervals, ensuring a balanced comparison with our algorithms. At the start of each user day, a number is randomly selected from the interval [L, U] to estimate the number of available risk times. Should the budget be exhausted before allocating for all available risk times, a minimum probability of 1 × 10−<sup>6</sup> is assigned to the remaining times. For additional information on the SeqRTS method, readers are referred to [Liao et al.](#page-8-7) [\(2018\)](#page-8-7).

1564

1566 1567

1569

1574

1576

1579

1581 1582 Figure 7. Average entropy change across user days under various prediction interval widths on HeartSteps V1 dataset. The shaded area indicates the ±1.96 standard error bounds across user days.

1583 1584 1585 1586 Figure [8](#page-29-0) shows the average competitive ratio and entropy change across user days, considering the scenario where SeqRTS assigns a minimum probability of 0 to remaining risk times once the budget is depleted. Owing to the Penalization term [2,](#page-2-0) this results in the objective function being negative infinity and the entropy change reaching infinity.

1587

1589 1590 In this section, we derive a loose upper bound for any randomized algorithm for the OUS problem.

1594 *Proof.* We utilize Yao's Lemma [\(Yao,](#page-9-10) [1977\)](#page-9-10), which states that an upper bound can be established by constructing a distribution over problem instances where every deterministic algorithm performs poorly. We construct a randomized

# F. Additional Results on HeartSteps V1 Study

Figure [7](#page-28-2) illustrates the average entropy change across user days. It is evident that SeqRTS exhibits the highest entropy change, suggesting non-uniform distribution behavior. In contrast, our learning-augmented algorithm demonstrates superior uniformity, outperforming the randomized algorithm. The benchmark method records an entropy of zero, attributed to its conservative strategy of assigning a constant probability of b/U.

![](_page_28_Figure_7.jpeg)

# G. Derivation of Lower Bound

![](_page_29_Figure_2.jpeg)

1609 Figure 8. Average competitive ratio and entropy change across user days under various prediction interval widths on the HeartSteps V1 dataset. The shaded area represents the ±1.96 standard error bounds across user days. *Note*: For SeqRTS, a minimum probability of 0 is assigned to the remaining times when the budget is exhausted.

1614 instance I with budget b = 1 and time horizon T = 5, where the true number of risk times τ ∗ takes values in 1, 2, 3, 4, 5 with probabilities π<sup>1</sup> = 0.6, π<sup>2</sup> = 0.15, π<sup>3</sup> = 0.1, π<sup>4</sup> = 0.1, and π<sup>5</sup> = 0.05.

1616 For this instance, the best deterministic algorithm with probabilities (p1, p2, p3, p4, p5) solves:

1618 1619

1624 1626 The optimal solution yields probabilities p<sup>1</sup> = 0.6970, p<sup>2</sup> = 0.1919, p<sup>3</sup> = 0.0606, p<sup>4</sup> = 0.0404, and p<sup>5</sup> = 0.0101, achieving an expected competitive ratio of 0.504. By Yao's Lemma, this implies no randomized algorithm can achieve a competitive ratio exceeding 0.504.

1629 While this bound provides insight, deriving a tight upper bound remains an open challenge. The difficulty stems from several factors:

- 1634 1636
- 1. The analysis requires evaluating every possible deterministic algorithm's expected performance under the budget constraint.
- 2. Each deterministic algorithm is characterized by a sequence of randomization probabilities (p1, . . . , p<sup>T</sup> ), making general analysis without a specific sequence structure intractable.
- 3. The objective function's non-smooth nature and the problem's distinct regimes (finite vs. infinite horizons) further complicate the analysis.

1639 1640 1641 We leave the derivation of a tighter bound as future work, noting that our current bound suggests the potential existence of improved algorithms.

$$\arg \max_{p_1, p_2, p_3, p_4, p_5} \pi_1 p_1 + \pi_2 (p_1 + p_2) + \pi_3 (p_1 + p_2 + p_3) + \pi_4 (p_1 + p_2 + p_3 + p_4) + \pi_5 p_5 - \pi_2 \frac{1}{2} \ln \frac{p_1}{p_2} - \pi_3 \frac{1}{3} \ln \frac{p_1}{p_3} - \pi_4 \frac{1}{4} \ln \frac{p_1}{p_4} - \pi_5 \frac{1}{5} \ln \frac{p_1}{p_5}$$

subject to p<sup>1</sup> + p<sup>2</sup> + p<sup>3</sup> + p<sup>4</sup> + p<sup>5</sup> = b.