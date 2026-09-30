# Online Uniform Sampling: Randomized Learning-Augmented Approximation Algorithms With Application To Digital Health

## Anonymous Authors1 Abstract

Motivated by applications in digital health, this work studies the novel problem of *online uniform* sampling (OUS), where the goal is to distribute a sampling budget uniformly across *unknown* decision times. In the OUS problem, the algorithm is given a budget b and a time horizon T, and an adversary then chooses a value τ
∗ ∈ [*b, T*], which is revealed to the algorithm online. At each decision time i ∈ [τ
∗], the algorithm must determine a sampling probability that maximizes the budget spent throughout the horizon, respecting budget constraint b, while achieving as uniform a distribution as possible over τ
∗. We present the first randomized algorithm designed for this problem and subsequently extend it to incorporate learning augmentation. We provide *worst-case* approximation guarantees for both algorithms, and illustrate the utility of the algorithms through both synthetic experiments and a real-world case study involving the HeartSteps mobile application. Our numerical results show strong empirical *average* performance of our proposed randomized algorithms against previously proposed heuristic solutions.

## 1. Introduction

The problem of *online uniform sampling* (OUS) is motivated by applications in digital health, where administering interventions at inappropriate times, such as when users are not at risk,1can significantly increase mental burden and hinder engagement with digital interventions (Li et al., 2020; Nahum-Shani et al., 2018; Wen et al., 2017; Mc- Connell et al., 2017; Mann & Robinson, 2009). Existing studies (Heckman et al., 2015; Klasnja et al., 2008; Dim1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054 1 itrijevic et al. ´ , 1972) show excessive digital interventions can heighten user fatigue, suggesting a threshold beyond which intervention effectiveness declines. A strategy rooted in the ecological momentary assessment (EMA) literature and proven effective in mitigating user fatigue involves allocating a fixed and limited budget for treatments delivered to the patient and delivering them with a uniform distribution across all risk times (e.g., Liao et al. 2018; Dennis et al. 2015; Rathbun et al. 2013; Scott et al. 2017a;b; Shiffman et al. 2008; Stone et al. 2007). However, this strategy is challenging because the true number of risk times is unknown, inspiring the OUS problem. Contributions Our contributions in this paper are two-fold. First, we formulate the common OUS problem in digital health as an online optimization problem and provide randomized algorithms that perform well in practice with competitive ratio guarantees. The competitive ratio measures the performance of an online algorithm against an offline clairvoyant benchmark, assuming the unknown parameter is revealed to the clairvoyant in advance. These guarantees are inherently conservative: 1) no online algorithm can achieve the same performance as the clairvoyant in practice (i.e., a competitive ratio of 1 is unattainable in OUS), and 2) they hold across all problem instances or sample paths (i.e., they are worst-case guarantees). Consequently, online approximation algorithms may exhibit conservative behavior. To address this, we numerically illustrate the practicality of our algorithm, demonstrating that they outperform naive benchmarks on average. Second, we extend our algorithm to the practical setting where a confidence interval *containing* the true risk time is provided, potentially through a valid statistical inference procedure. We conduct the competitive ratio analysis for our proposed learning-augmented approximation algorithm, demonstrating its *consistency* in the strong sense—optimal performance is achieved when the confidence interval width is zero—and *robustness*—the learning-augmented algorithm performs no worse than the non-learning augmented counterpart. Our findings indicate that, in almost all tested scenarios, the randomized learning-augmented algorithm outperforms its non-learning augmented counterpart. Outline In Section 2, we formalize the OUS problem. We in055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 troduce our randomized algorithm without learning augmentation in Section 3. This algorithm is segmented into three distinct cases based on the horizon length to budget ratio, with a competitive ratio established for each. In Section 4, we develop a learning-augmented algorithm that integrates a prediction interval and provide theoretical justification for its effectiveness. The efficacy of these algorithms is first assessed through synthetic experiments, followed by their application to real-world data in Section 5.

## 1.1. Related Work

Online Uniform Sampling Existing methodologies, primarily sourced from the EMA literature, focus on delivering interventions through the form of mobile self-report requests over a fixed time horizon. These approaches are constrained by budget and uniformity considerations to minimize user burden and ensure accurate reflection of user conditions across diverse contexts (Dennis et al., 2015; Rathbun et al.,
2013; Scott et al., 2017a;b). In this work, we permit intervention only when users are *at risk*, leading to an unknown horizon length. This introduces a significant challenge in balancing the allocation of a limited budget with the need to maintain uniformity in intervention delivery. To address this issue, Liao et al. (2018) developed a heuristic algorithm, but its performance depends heavily on the accuracy of the predicted number of risk times. When the prediction is inaccurate, the algorithm lacks theoretical guarantees, highlighting the need for a more robust algorithm design. Multi-option Ski-rental Problem Our work closely relates to the *multi-option ski-rental* (MOSR) problem (Zhang et al., 2011; Shin et al., 2023), where the number of snowy days is unknown. Customers have multiple ski rental options, differing in cost and duration. The goal is to minimize costs while ensuring ski availability on snowy days. Shin et al. (2023) introduced a randomized algorithm for MOSR, with a tight e-competitive ratio. A random variable B is introduced as a proxy for the unknown true horizon T. B is initialized to α, following a density function 1/α within [1, e). The algorithm iteratively solves an optimization problem to identify an optimal set of rental options within budget B, maximizing day coverage. Customers sequentially utilize the options until depletion, at which point B is increased by a factor of e, and the process is repeated. Our work builds upon Shin et al. (2023), leveraging the same randomized algorithmic idea. However, our problem setting is *significantly different* from that of MOSR. In particular, instead of having discrete ski-rental options, at each decision time, the algorithm needs to decide on the sampling probability, which is continuous in nature. Further, in our problem, the sum of the sampling probability cannot exceed a predefined budget, while such constraints do not exist in MOSR. Our problem additionally has a uniformity consideration.

Learning-Augmented Online Algorithms Many online algorithms incorporate black-box point predictions on the unknown parameters to improve their worst-case guarantees (Purohit et al., 2018; Bamas et al., 2020; Wei & Zhang, 2020; Jin & Ma, 2022). The confidence of these point estimates is often represented by a single parameter, with a higher value indicating more accurate predictions. When the confidence is low, most work do not guarantee that the learning-augmented algorithm will perform no worse than the non-learning counterpart (Bamas et al., 2020). In practice, prediction confidence intervals, rather than point estimates, are often generated using valid statistical inference methods. A wider confidence interval typically indicates less informative predictions (Shafer & Vovk, 2008). Im et al. (2021) consider the setting where the prediction provides a range of values for key parameters in the online knapsack problem. However, their deterministic solution cannot be directly extended to our setting, as the number of risk times in OUS is stochastic. We introduce the first integration of confidence intervals into randomized algorithms for OUS.

This integration enables our proposed algorithms to surpass the performance of their non-learning counterpart, even with a wide confidence interval.

## 2. Problem Framework

In the context of digital interventions, we define the OUS problem as presented by Liao et al. (2018). Let T denote the total number of decision points within a decision period (e.g., within a day). At any given time t ∈ [1, T] in each decision period, patients encounter binary risk levels2(determined by data from wearable devices), indicating whether the patient is likely to experience an adverse event, such as relapse to smoking. The distribution of risk levels is allowed to change *arbitrarily* across decision periods since treatments may influence and reduce subsequent risk. Let τ
∗ be the *unknown true* number of risk times that a patient experiences in a decision period. Note that τ
∗is stochastic and is revealed *only* at the end of the horizon T, corresponding to the last decision time in the decision period. We define pi ∈ (0, 1) to be the treatment probability at time i ∈ [τ
∗]. We preclude the possibility that pi = 0 or pi = 1 to facilitate after-study inference (Boruvka et al.,
2018; Zhou et al., 2023; Kallus & Zhou, 2022).

The algorithm is provided with a *soft* budget of b, representing the total *expected* number of interventions allowed to be delivered within each decision period. We assume τ
∗ > b as evidenced in practice (Liao et al., 2018). At each decision time i, the algorithm decides the intervention probability pi.

The objectives of the OUS problem (Liao et al., 2018; Den2When multiple risk levels are present, the problem naturally decomposes into independent subproblems for each risk level, see more details in Appendix A.

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 nis et al., 2015; Rathbun et al., 2013; Scott et al., 2017a;b; Shiffman et al., 2008; Stone et al., 2007) are to 1) assign the intervention probabilities {pi}i∈[τ
∗] as uniform as possible across risk times, and 2) maximize the sum of intervention probabilities across risk times while adhering to the budget constraint b. Abstractly, in the OUS problem, the algorithm is given a budget b and a time horizon T, and an adversary then chooses a value τ
∗ ∈ [*b, T*], which is revealed to the algorithm online. At each decision time i ∈ [τ
∗], the algorithm must determine a sampling probability that maximizes the budget spent throughout the horizon, respecting the budget constraint b, while achieving as uniform a distribution as possible over τ
∗.

Without additional information on τ
∗, the two objectives compete with each other. A naive solution to fulfill the first objective is to set pi = *b/T, i* ∈ [τ
∗], which, however, fails to maximize the sum of intervention probabilities. Conversely, if we set pito be a large constant value, there is a risk of depleting the budget before the end of the horizon, thus failing to achieve the uniformity objective. Therefore, the optimality of the two objectives cannot be simultaneously achieved without additional information on τ
∗. Liao et al. (2018) provided a heuristic algorithm for OUS given a point estimate of τ
∗. The algorithm's performance is significantly influenced by the accuracy of this forecast. In this work, we introduce randomized algorithms for OUS with robust worst-case guarantees, considering settings both with and without learning augmentation.

## 2.1. Ous As An Online Optimization Problem

In this section, we formulate OUS as an online optimization problem, where the objective function provides a uniform way of comparing the performance of different approximation algorithms, and the constraint defines the set of feasible solutions.

$$\left\{\,\max\sum_{i}^{\tau^{*}}p_{i}-{\frac{1}{\tau^{*}}}\ln\left({\frac{\max_{i\in[\tau^{*}]}p_{i}}{\min_{i\in[\tau^{*}]}p_{i}}}\right):\right.$$ $$\left.\mathbb{E}\left[\sum_{i=1}^{\tau^{*}}p_{i}\right]\leq b,p_{i}\in(0,1),\forall i\in[\tau^{*}].\right\}$$

$$(1)$$

where the expectation, E, in the budget constraint is taken over the randomness in the algorithm. This budget constraint is "soft" in the sense that if we have multiple decision periods (which is the case in digital health), we should satisfy the budget constraint in expectation. Remark 2.1. Notably, the purpose of formulating the optimization problem is not to solve it optimally, but rather to provide a feasible solution without knowledge of the unknown τ
∗. Rather than setting uniformity as a constraint, we incorporate it into the design of our approximation algorithms. By including uniformity as a penalty term in the objective function, represented by:

$$\frac{1}{\tau^{*}}\ln\left(\frac{\max_{i\in[\tau^{*}]}p_{i}}{\min_{i\in[\tau^{*}]}p_{i}}\right),\tag{2}$$

we can directly compare the overall performance of different online approximation algorithms, including how well they achieve uniformity, by comparing their objective function values. The choice of the penalty term (2) is inspired by the entropy change concept from thermodynamics (Smith, 1950). This choice is not unique but it has several nice properties: a)
it equals to 0 if and only if {pi}i∈[τ
∗] are identical, b) it increases with the maximum difference in {pi}i∈[τ
∗], and c) it tends towards infinity as the value of pi approaches to zero, penalizing scenarios where the expected budget is depleted before the horizon ends. We note that one can replace the term 1/τ ∗in the penalty by a tuning parameter σ, which controls the strength of the penalty, as discussed in Remarks 3.3 and 4.3.

3 Finally, we highlight that KL
divergence cannot be used here to impose uniformity (see detailed discussion in Appendix B).

## 2.2. Offline Clairvoyant And Competitive Ratio

In the *offline clairvoyant* benchmark, the clairvoyant possesses knowledge of τ
∗. When provided with this value, the optimal solution to Problem (1) is to set pi = b/τ ∗.

Consequently, the optimal value of the objective function in Problem (1) is OPT(τ
∗) = b. Importantly, in practice, no online algorithm can attain OPT(τ
∗) as the offline clairvoyant benchmark serves as an upper bound on the best achievable performance for any *online* algorithm without knowledge of τ
∗. Let SOL be the objective value of Problem (1) achieved by a *randomized online* algorithm, we say that Definition 2.2 (γ-competitive). An algorithm is γcompetitive if E[SOL] ≥ γ · OPT(τ
∗).

Remark 2.3. First, we emphasize that the expectation in 3Since the current design of our algorithms does not explicitly account for the form of the penalty term, the penalty (2) could also be replaced by any other suitable functions, with performance re-evaluated under the modified objective function.

Specifically, we aim to find a sequence of treatment probability assignments {pi}i∈[τ
∗]that achieves the following two objectives: 1. Maximizes the sum of treatment probabilities across risk times, subject to the "soft" budget b; 2. Penalizes changes in treatment probabilities within each risk level.

Formally, the OUS problem can be expressed using the following optimization problem: Definition 2.2 is taken *only* over the randomness of the algorithm. Second, we note that if the competitive ratio is provided, it holds in expectation for every feasible τ
∗ ∈
[b, T]. This implies that the competitive ratio serves as a worst-case guarantee: in any OUS instance, as long as the budget b and the maximum horizon length T remain fixed across decision periods, we can expect to meet the budget and achieve the stated competitive ratio, regardless of the specific realization of τ
∗in each decision period.

The key difficulty in solving Problem (1) in the online setting arises due to the unknown nature of τ
∗. In Section 3, we introduce the first approximation algorithm for the OUS problem.

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219

## 2.3. With Learning Augmentation

In the *learning-augmented* setting, we are additionally provided with a prediction confidence interval [*L, U*], generated by a valid statistical procedure, that contains the unknown true τ
∗ with high probability. A wider confidence interval reflects lower prediction quality. For simplicity, we assume τ
∗lies within the interval, though our results generalize to cases where it is contained with high probability. To evaluate the performance of the learning-augmented algorithm in the presence of a prediction confidence interval, we extend the standard consistency-robustness analysis from the prior literature (Lykouris & Vassilvtiskii, 2018; Purohit et al., 2018; Bamas et al., 2020; Shin et al., 2023). Specifically, an algorithm is said to be λ-*consistent* if it achieves E[SOL] ≥ λ · OPT(τ
∗) when the prediction is perfect, i.e.,
when L = U, indicating a zero-length interval.4 This aligns with the standard definition where the prediction is accurate (Shin et al., 2023). Conversely, an algorithm is ρ-*robust* if it satisfies E[SOL] ≥ ρ · OPT(τ
∗) regardless of the width of the prediction interval [*L, U*], corresponding to the previous definition where the prediction can be arbitrarily inaccurate. In Section 4, we show that our proposed learning-augmented algorithm is 1-consistent, achieving the optimal solution when the interval width is zero. Moreover, the competitive ratio of our learning-augmented algorithm closely matches that of the non-learning augmented counterpart, even when the prediction quality deteriorates. To the best of our knowledge, this is the first work that provide a 1-consistency guarantee on learning-augmented algorithms, after careful engineering of the algorithms.

## 3. Randomized Algorithm

In this section, we introduce our randomized algorithm, Algorithm 1, designed for the OUS problem *without* learning 4Similar to Definition 2.2, the expectation is taken over the randomness in the algorithm.

augmentation. This algorithm is inspired by the randomized algorithm proposed by Shin et al. (2023) for the MOSR problem. Due to the significant differences in problem setup outlined in Section 1.1, the design of our algorithm requires 1) imposing a discrete structure on the sampling probabilities to account for uniformity considerations, making the analysis of the algorithm more tractable, and 2) explicitly addressing the finite horizon length and budget constraint, ensuring that the randomized algorithm does not exceed the budget in expectation. Algorithm 1 Randomized Online Algorithm 1: **Input:** T, b 2: **Initialize:** j = 1, we sample α ∈ [*b, be*] from a distribution with p.d.f. f(α) = 1/α, and initialize τ˜ = α 3: for i = 1*, ..., τ* ∗ do 4: We calculate:

$$\mathrm{Int}(\tilde{\tau})=\left\{\begin{array}{l l}{{\left\lfloor\tilde{\tau}\right\rfloor}}&{{w.p.}}\\ {{\left\lceil\tilde{\tau}\right\rceil}}&{{w.p.}}\end{array}\right.\left\lceil\tilde{\tau}\right\rceil-\tilde{\tau}$$

5: if T ≤ be **then** 6: Update τ˜ and set pi using **Subroutine** 1 7: **else if** *be < T* ≤ be2**then**
8: Update τ˜ and set pi using **Subroutine** 2 9: **else**
10: Update *τ, b* ˜ and set pi using **Subroutine** 3 11: **end if**
12: Output treatment probability pi 13: **end for**
The proposed algorithm, Algorithm 1, provides a feasible solution to Problem (1). At its core, our algorithm assigns the sampling probabilities in a monotonically nonincreasing fashion over time. To accommodate varying practical scenarios where the budget-to-horizon ratio differs across applications, we designed specialized approximation algorithms for three possible scenarios: 1) T ≤ be
(**Subroutine** 1), 2) *be < T* ≤ be2(**Subroutine** 2), and 3) T > be2(**Subroutine** 3).

We maintain a running "guess" of τ
∗, denoted by τ˜. We initialize τ˜ to be α, where α ∼ [*b, b* · e] with density 1/α, and e represents the Euler's number. If the current number of risk times i is within our running guess τ˜, then we do not change the current sampling assignment probability. Otherwise, we update τ˜ as τ˜ = ˜τe and update the sampling probability according to Algorithm 1, depending on the length of the horizon T relative to b. The random draw τ˜ controls not only the value of the sampling probability but also the duration of each stage. Once the algorithm reaches τ˜, it transitions to the next stage, resulting in a stage-wise constant probability sequence.

We first show the feasibility of our proposed solution, i.e.,

| Subroutine 1 (i, b, τ˜, T, Int(˜τ )) 1: if i > Int(˜τ ) then 2: τ˜ = ˜τe 3: end if 4: pi = b min(T ,τ˜(e−1)) Subroutine 2 (i, b, τ˜, Int(˜τ )) 1: if i > Int(˜τ ) then 2: j = j + 1, τ˜ = ˜τe 3: end if   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

1: if i > Int(˜τ ) **then** 2: j = j + 1, τ˜ = ˜τe 3: **end if** 4: if j ≥ 3 **then**

5: pi =b

τ e˜

6: **else**

7: pi =b

τ˜(e−1)

8: **end if**

the sampling probabilities outputted from Algorithm 1 satisfies the budget constraint in Problem (1): Lemma 3.1. Let p A1 ibe the probability returned by Algorithm 1 *at risk time* i ∈ [τ
∗]. This solution always satisfies the budget constraint in expectation, i.e., E
hPτ
∗
i=1 p A1 ii≤
b, where the expectation is taken over the randomness of the algorithm. The proof of Lemma 3.1 is included in Appendix C.1. Next, by leveraging the monotonically non-increasing nature of the sampling probabilities, the objective in Problem (1) simplifies to

$$\operatorname*{max}\sum_{i=1}^{\tau^{*}}p_{i}-{\frac{1}{\tau^{*}}}\ln\left({\frac{p_{1}}{p_{\tau^{*}}}}\right).$$
. (3)
Using Equation (3), we compute the competitive ratio of Algorithm 1: Theorem 3.2. Algorithm 1 is X (T)-competitive, where X is defined as follows:

$${\mathcal{X}}(T):={\left\{\begin{array}{l l}{{\frac{1}{e}}\left(\ln(e-1)+{\frac{1}{e-1}}\right)}&{{\mathrm{if}}\quad T\leq b e,}\\ {{\frac{1}{e}}}&{{\mathrm{if}}\quad b e<T\leq b e^{2},}\\ {{\frac{1}{e}}-{\frac{1}{e^{2}}}}&{{\mathrm{if}}\quad T>b e^{2}.}\end{array}\right.}$$

The above competitive ratio is conservative by design: It was derived by taking the worst case over *unknown* τ
∗and the horizon length T within each case. The proof of Theorem 3.2 in Appendix C.2 outlines the competitive ratio as a function of τ
∗and T. Additionally, in Section 5, we investigate the impact of varying τ
∗ while keeping the horizon length fixed, providing a numerical illustration of how the expected competitive ratio changes. We note that the expected competitive ratio, averaged over the unknown τ
∗,
is much better than our theoretical competitive ratio illustrated above. Based on our theoretical competitive ratio in 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274

| Subroutine 3 (i, b, τ˜, Int(˜τ )) 1: if i > Int(˜τ ) then 2: j = j + 1, τ˜ = ˜τe 3: if j ≥ 3 then 4: b = b(1 − 1 ) e 5: end if 6: end if 7: pi = b τ e˜   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------|

Theorem 3.2, we recommend choosing the horizon length T relative to the budget b to be below be2, which aligns with our empirical findings in Section 5 (see Remark 5.1 for details).

Remark 3.3. As stated in Section 2.1, the term 1 τ
∗ in the penalty can be replaced by a tunable strength parameter σ.

In Section C.2, we show that for T ≤ be2, the above results hold over a wide range of σ values, specifically σ ≤
b 2
.

However, when *T > be*2, σ should be on the order of 1 τ
∗ ,
ensuring that the penalty term scales similarly to the budget term in the objective. Remark 3.4. Establishing an upper bound on the performance of any randomized algorithm for the OUS problem is challenging due to the non-smooth nature of the objective function and the problem's three different operating regimes. In Appendix G, we derive a loose upper bound of 0.5 for the OUS problem using Yao's lemma (Yao, 1977) and leave the derivation of a tighter bound for future work.

## 4. Learning-Augmented Algorithm

In this section, we propose a new approximation algorithm, Algorithm 2, under the learning-augmented setting, where we are provided with prediction confidence intervals [*L, U*] for the unknown τ
∗. Algorithm 2 builds upon the nonlearning augmented counterpart, Algorithm 1, utilizing the given confidence interval for optimization. Similar to Algorithm 1, we initialize α ∼ [*b, be*] with density 1/α, and the current "guess" of τ
∗is reflected by τ˜ + L.

In Algorithm 2, the three scenarios differ from those in Algorithm 1. Here, the distinction is based on the relationship between the upper bound of the interval, U, and the budget b. The three scenarios are 1) U ≤ be (**Subroutine** 4), 2) be < U ≤ be2, further divided into 2a) U − L ≤ b(e − 1)
(**Subroutine** 4), and 2b) U − *L > b*(e − 1) (**Subroutine** 2),
and 3) *U > be*2, further divided into 3a) U − L ≤ b(e + 1)
(**Subroutine** 5), and 3b) U − *L > b*(e + 1) (**Subroutine** 6). Similarly, we first demonstrate that Algorithm 2 produces a feasible solution to Problem (1), with the proof provided in Appendix D.1 . Lemma 4.1. Let p A2 ibe the probability returned by Algorithm 2 *at risk time* i ∈ [τ
∗]. This solution always satisfies Algorithm 2 Randomized Online Algorithm With Prediction Confidence Intervals 1: **Input:** T, b, [L, U] 2: **Initialize:** j = 1, sample α ∈ [*b, be*] from a distribution with p.d.f. f(α*) = 1*/α, and initialize τ˜ = α 3: for i = 1*, ..., τ* ∗ do 4: We calculate:

$$\mathrm{Int}(\tilde{\tau})=\left\{\begin{array}{l l}{{\left\lfloor\tilde{\tau}\right\rfloor}}&{{w.p.}}\\ {{\left\lceil\tilde{\tau}\right\rceil}}&{{w.p.}}\end{array}\right.\left\lceil\tilde{\tau}\right\rceil-\tilde{\tau}$$

5: if U ≤ be **then** 6: Update τ˜ and set pi using **Subroutine** 4 7: **else if** *be < U* ≤ be2**then**
8: if U − L ≤ b(e − 1) **then**
9: Update τ˜ and set pi with **Subroutine** 4 10: **else**
11: Update τ˜ and set pi with **Subroutine** 2 12: **end if** 13: **else** 14: if U − L ≤ b(e + 1) **then**
15: Update τ˜ and set pi with **Subroutine** 5 16: **else**
17: Update *τ, b* ˜ and set pi with **Subroutine** 6 18: **end if** 19: **end if**
20: Output sampling probability pi 21: **end for**
275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329

| Subroutine 4 (i, b, τ˜, L, U, Int(˜τ ))   |
|-------------------------------------------|

1: if i > Int(˜τ ) + L **then** 2: τ˜ = ˜τe 3: **end if**
4: pi =b min(U,τ˜+L)
Subroutine 5 (i, b, τ˜, L, U, Int(˜τ ))
1: if i > Int(˜τ ) + L **then**
2: τ˜ = ˜τe 3: **end if**
4: pi =b min(*U,τ e*˜ +L)
of T relative to b in the context of prediction intervals in Remark 5.2.

Remark 4.3. Similarly, the term 1 τ
∗ in the penalty can be replaced by a tuning parameter σ. In Section D.2, we show that for U ≤ be2, the above results hold for a wide range of σ values, specifically σ ≤
b e
. However, when *T > be*2, σ should be of the order 1 τ
∗ to align the penalty term with the budget term in the objective.

## 5. Experiments

In this section, we numerically assess the performance of our proposed algorithms through numerical experiments conducted on both synthetic and real-world datasets.

## 5.1. Synthetic Experiments

Benchmarks In the setting without learning augmentation, we compare Algorithm 1 against a conservative benchmark that delivers interventions with a constant probability b/T. In the learning-augmented setting, where a confidence interval [*L, U*] is provided, we compare Algorithm 2 against two benchmarks: (1) a benchmark that delivers interventions with a constant probability b/U, and (2) Algorithm 1.Due to the limited algorithmic work on OUS (Online Uniformity Scheduling) and the absence of existing algorithms that handle confidence intervals, we do not include additional benchmarks in the synthetic data experiments. However, in the real-world example, we also evaluate the SeqRTS algorithm (Liao et al., 2018), which does not account for the prediction uncertainty of τ
∗. The metric used for the evaluation is the average competitive ratio. Without Learning Augmentation In this setting, we evaluate the performance of Algorithm 1 across all three scenarios outlined in Theorem 3.2. To do this, we fix the budget at b = 3 and alter the horizon lengths T to align with each scenario. For Scenarios 1 and 2, we set T to the maximum allowable values with b = 3, specifically T = 8 and 22, as illustrated in Figure 1 (left and middle). For Scenario

$${\mathcal{X}}(U):={\left\{\begin{array}{l l}{\ln2+{\frac{e-1}{e}}\ln{\frac{e-1}{e}}}\\ {{\frac{1}{e}}}\\ {2-\ln(e^{2}-e+1)}\end{array}\right.}$$
eif *be < U* ≤ be2,
2 − e + 1) if *U > be*2.
$$\begin{array}{r l}{i f}&{{}U\leq b e,}\\ {i f}&{{}b e<U\leq b e^{2},}\\ {i f}&{{}U>b e^{2}.}\end{array}$$

We first note that Algorithm 2 is 1-consistent, achieving the performance of the offline clairvoyant when the prediction is perfect. The proof of Theorem 4.2 in Appendix D.2 provides a detailed analysis of the competitive ratio, which depends on the parameters τ
∗, L, and U.

5 Furthermore, Section 5 explores the impact of varying the prediction confidence interval width U − L while keeping τ
∗constant.

Our findings reveal that Algorithm 2 almost always outperforms Algorithm 1. Finally, we discuss the design choice 5In Theorem 4.3, we present the competitive ratios for scenarios 1), 2), and 3) separately, combining the results of the respective subroutines.

the budget constraint in expectation, i.e., E
hPτ
∗
i=1 p A2 i i≤
b, where the expectation is taken over the randomness of the algorithm. Next, we provide a theoretical guarantee on its performance: Theorem 4.2. Algorithm 2 is 1-consistent and X (U)-robust, where X (U) *is defined as follows:*

| Subroutine 6 (i, b, τ˜, L, U, Int(˜τ )) 1: if i > Int(˜τ ) + L then 2: j = j + 1 3: if j = 2 then 4: b = b(1 − τ˜+L−b ) τ˜(e−1)+L 5: else 6: b = b(1 − 1 ) e 7: end if 8: τ˜ = ˜τe 9: end if 10: if j = 1 then 11: pi = b τ˜(e−1)+L 12: else 13: pi = b τ e˜ 14: end if   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 3, where T can grow asymptotically to infinity, we choose T = 100 for simplicity (Figure 1 right). To simulate risk occurrences, we randomly choose an integer τ
∗from the interval [*b, T* − 1] and then select τ
∗ distinct time points uniformly at random from the T available time steps as risk times. Figure 1 displays the average competitive ratio across a range of τ
∗ values. Figure 1a indicates that our randomized algorithm consistently outperforms the benchmark by a constant competitive ratio for all values of τ
∗in Scenario 1. Similarly, Figure 1b shows that in Scenario 2, our randomized algorithm increasingly outperforms the benchmark as τ
∗ deviates further from the horizon length T. In Figure 1c, as T increases, the average competitive ratio of our algorithm remains constant and consistently outperforms the benchmark.6 Therefore, we conclude that our algorithm increasingly outperforms the benchmark as T grows to infinity.

Remark 5.1 (Design choice of b and T **in the absence of**
prediction confidence intervals). In real-world applications, the intervention budget for each risk level is often fixed. However, a key design consideration is the choice of T, i.e., the granularity of the decision period. As illustrated in Figure 1, while Scenario 3 achieves the greatest performance improvement as T approaches infinity, our randomized algorithm attains the highest competitive ratio across all τ
∗in Scenarios 1 and 2. Thus, in the absence of prediction intervals, we recommend selecting T such that T ≤ be2. With Learning Augmentation In this setting, we evaluate the performance of Algorithms 1 and 2 across varying prediction interval widths. As in the non-learningaugmented setting, we fix the budget at b = 3 and ex6This is because when b is fixed, the treatment assignment probability is independent of T.

amine the performance of our learning-augmented algorithm for T = 8, 22, and 100, covering the three scenarios outlined inAlgorithm 2. To compare the performance of our algorithm across various confidence widths, we fix τ
∗ = Int[0.5(T + b)] across all simulations.7 The confidence intervals are randomly generated based on the given width and must contain τ
∗.

Figure 2 plots the average competitive ratio of each algorithm across a range of interval widths. We observe that the naive benchmark (where pi = b/U for all i ∈ [τ
∗]) outperforms the Algorithm 1 (which does not have access to the prediction interval) when the confidence interval is narrow. This is not surprising as in this case τ
∗ ≈ U. However, as the prediction interval widens, our Algorithm 1 outperforms the naive benchmark. In addition, we observe that our learning-augmented algorithm performs no worse than both the naive benchmark and the randomized algorithm. In particular, the advantage of Algorithm 2 is the largest in Scenario 3.

Remark 5.2 (Design choice of b and T **in presence of** prediction intervals). If we expect the value of τ
∗to be small, we recommend setting T ≤ be2to ensure that the algorithm always operates in Scenario 2, where U ≤ be2. If we expect a reasonably large value of τ
∗, we recommend setting a large value for *T > be*2such that the algorithm operates under Scenario 3, where U can exceed be2.

Additional experimental results for small τ
∗are provided in Appendix E.1. We note that as τ
∗ decreases, the advantage of our algorithm in Scenario 2 increases. We also include competitive ratio figures without the penalization term from Problem (1) in Appendix E.2, measuring the fraction of the budget spent by our algorithms.

## 5.2. Real-World Experiments On Heartsteps

Our research is motivated by the Heartsteps V1 mobile health study, which aimed to increase physical activity among 37 sedentary individuals over a six-week period, with T = 144 decision points per day (Klasnja et al., 2019).

At each decision time t, a risk variable Rt is observed, which is binary: Rt = 1 indicates a sedentary state, identified by recording fewer than 150 steps in the prior 40 minutes, and Rt = 0 signifies a non-sedentary state. The total number of risk times, τ
∗ =PT
t=1 Rt, is unknown. The primary objective here is to uniformly distribute approximately b = 1.5 interventions across sedentary times each day. Benchmarks In addition to the naive benchmark b/U, we compare the performance of Algorithms 1 and 2 with the SeqRTS algorithm, as proposed by Liao et al. (2018). Under SeqRTS, the budget may be exhausted before all available 7If we allow τ
∗to change across different simulations, then the difference that we observe in competitive ratio might be due to this change in τ
∗.

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 In Figure 3, Algorithm 2, which incorporates a prediction interval, invariably outperforms the non-learning counterpart, the SeqRTS approach, and the naive benchmark b/U. Moreover, our proposed algorithms exhibit superior uniformity in risk times sampling, evidenced by reduced entropy change compared to both the non-learning algorithm and SeqRTS, as further detailed in Figure 7 in Appendix F. To better understand the behavior of SeqRTS, we set the minimum probability to 0 in Figure 8 in Section F. This figure illustrates that SeqRTS could deplete its budget even when the prediction is fairly accurate, highlighting the robustness of our algorithms under adversarial risk level arrivals. Conclusion and Future Works This paper marks the first attempt to study the online uniform allocation problem within the framework of approximation algorithms. We introduce two novel online algorithms—either incorporating learning augmentation or not—backed by rigorous theoretical guarantees and empirical results. Future works include adapting existing algorithms to scenarios where prediction intervals improve over time.

(a) Scenario 1: T = 8 (b) Scenario 2: T = 22 (c) Scenario 3: T = 100
(a) Scenario 1: T = 8 (b) Scenario 2: T = 22 (c) Scenario 3: T = 100
risk times are allocated. In such cases, a minimum probability of 1×10−6is assigned to the remaining risk times when evaluating the objective in Problem (1). A comprehensive description of the SeqRTS method and additional implementation details are provided in Appendix F. Performance is assessed using the competitive ratio and the average entropy change across user days.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Bamas, E., Maggiori, A., and Svensson, O. The primal-dual method for learning augmented algorithms. Advances in Neural Information Processing Systems, 33:20083– 20094, 2020.

Boruvka, A., Almirall, D., Witkiewitz, K., and Murphy, S. A. Assessing time-varying causal effect moderation in mobile health. *Journal of the American Statistical* Association, 113(523):1112–1121, 2018.

Dennis, M. L., Scott, C. K., Funk, R. R., and Nicholson, L. A pilot study to examine the feasibility and potential effectiveness of using smartphones to provide recovery support for adolescents. *Substance abuse*, 36(4):486–492, 2015.

Dimitrijevic, M. R., Faganel, J., Gregori ´ c, M., Nathan, P., ´
and Trontelj, J. Habituation: effects of regular and stochastic stimulation. Journal of Neurology, Neurosurgery & Psychiatry, 35(2):234–242, 1972.

Heckman, B. W., Mathew, A. R., and Carpenter, M. J. Treatment burden and treatment fatigue as barriers to health.

Current opinion in psychology, 5:31–36, 2015.

Im, S., Kumar, R., Montazer Qaem, M., and Purohit, M.

Online knapsack with frequency predictions. Advances in Neural Information Processing Systems, 34:2733–2743, 2021.

Jin, B. and Ma, W. Online bipartite matching with advice:
Tight robustness-consistency tradeoffs for the two-stage model. Advances in Neural Information Processing Systems, 35:14555–14567, 2022.

Kallus, N. and Zhou, A. Stateful offline contextual policy evaluation and learning. In *International Conference on* Artificial Intelligence and Statistics, pp. 11169–11194. PMLR, 2022.

Klasnja, P., Harrison, B. L., LeGrand, L., LaMarca, A.,
Froehlich, J., and Hudson, S. E. Using wearable sensors and real time inference to understand human recall of routine activities. In Proceedings of the 10th international conference on Ubiquitous computing, pp. 154–163, 2008.

Klasnja, P., Smith, S., Seewald, N. J., Lee, A., Hall, K.,
Luers, B., Hekler, E. B., and Murphy, S. A. Efficacy of contextually tailored suggestions for physical activity: a micro-randomized optimization trial of heartsteps. *Annals* of Behavioral Medicine, 53(6):573–582, 2019.

Li, S., Psihogios, A. M., McKelvey, E. R., Ahmed, A.,
Rabbi, M., and Murphy, S. Microrandomized trials for promoting engagement in mobile health data collection: Adolescent/young adult oral chemotherapy adherence as an example. *Current opinion in systems biology*, 21:1–8, 2020.

Liao, P., Dempsey, W., Sarker, H., Hossain, S. M., Al'absi, M., Klasnja, P., and Murphy, S. Just-in-Time but Not Too Much: Determining Treatment Timing in Mobile Health. Proceedings of the ACM on interactive, mobile, wearable and ubiquitous technologies, 2(4):179, December 2018. ISSN 2474-9567. doi: 10.1145/3287057.

Lykouris, T. and Vassilvtiskii, S. Competitive caching with machine learned advice. In Dy, J. and Krause, A. (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of *Proceedings of Machine* Learning Research, pp. 3296–3305. PMLR, 10–15 Jul 2018.

Mann, S. and Robinson, A. Boredom in the lecture theatre: An investigation into the contributors, moderators and outcomes of boredom amongst university students. British Educational Research Journal, 35(2):243–258, 2009.

McConnell, M. V., Shcherbina, A., Pavlovic, A., Homburger, J. R., Goldfeder, R. L., Waggot, D., Cho, M. K., Rosenberger, M. E., Haskell, W. L., Myers, J., et al. Feasibility of obtaining measures of lifestyle from a smartphone app:
the myheart counts cardiovascular health study. JAMA
cardiology, 2(1):67–76, 2017.

Nahum-Shani, I., Smith, S. N., Spring, B. J., Collins, L. M.,
Witkiewitz, K., Tewari, A., and Murphy, S. A. Justin-time adaptive interventions (jitais) in mobile health: key components and design principles for ongoing health behavior support. *Annals of Behavioral Medicine*, pp.

1–17, 2018.

Purohit, M., Svitkina, Z., and Kumar, R. Improving online algorithms via ml predictions. Advances in Neural Information Processing Systems, 31, 2018.

Rathbun, S. L., Song, X., Neustifter, B., and Shiffman, S.

Survival analysis with time varying covariates measured at random times by design. Journal of the Royal Statistical Society Series C: Applied Statistics, 62(3):419–434, 2013.

Scott, C. K., Dennis, M. L., Gustafson, D., and Johnson, K.

A pilot study of the feasibility and potential effectiveness of using smartphones to provide recovery support. Drug and Alcohol Dependence, 100(171):e185, 2017a.

Scott, C. K., Dennis, M. L., and Gustafson, D. H. Using smartphones to decrease substance use via selfmonitoring and recovery support: study protocol for a randomized control trial. *Trials*, 18(1):1–11, 2017b.

Shafer, G. and Vovk, V. A tutorial on conformal prediction.

Journal of Machine Learning Research, 9(3), 2008.

Shiffman, S., Stone, A. A., and Hufford, M. R. Ecological momentary assessment. *Annu. Rev. Clin. Psychol.*, 4: 1–32, 2008.

Shin, Y., Lee, C., Lee, G., and An, H.-C. Improved learningaugmented algorithms for the multi-option ski rental problem via best-possible competitive analysis. arXiv preprint arXiv:2302.06832, 2023.

Smith, J. M. Introduction to chemical engineering thermodynamics, 1950.

Stone, A., Shiffman, S., Atienza, A., and Nebeling, L. The science of real-time data capture: Self-reports in health research. Oxford University Press, 2007.

Wei, A. and Zhang, F. Optimal robustness-consistency trade-offs for learning-augmented online algorithms. Advances in Neural Information Processing Systems, 33: 8042–8053, 2020.

Wen, C. K. F., Schneider, S., Stone, A. A., and Spruijt-Metz, D. Compliance with mobile ecological momentary assessment protocols in children and adolescents: a systematic review and meta-analysis. Journal of medical Internet research, 19(4):e132, 2017.

Yao, A. C.-C. Probabilistic computations: Toward a unified measure of complexity. In *18th Annual Symposium on* Foundations of Computer Science (sfcs 1977), pp. 222– 227. IEEE Computer Society, 1977.

Zhang, G., Poon, C. K., and Xu, Y. The ski-rental problem with multiple discount options. Information Processing Letters, 111(18):903–906, 2011.

Zhou, Z., Athey, S., and Wager, S. Offline multi-action policy learning: Generalization and optimization. Operations Research, 71(1):148–183, 2023.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549

## A. Extension To Multiple Risk Levels

In this section, we discuss the extension of the online uniform risk times sampling problem to multiple risk levels.

550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 At each time t ∈ [1, T], the patient is associated with an ordinal risk level from K possible levels. The higher the risk level, the more likely the patient will experience a negative event, such as a relapse to smoking. As stated previously, the distributions of risk levels are allowed to change *arbitrarily* across decision periods since we anticipate that the treatment will reduce subsequent risk.

Let τ
∗
kbe the *unknown true* number of decision times at risk level k ∈ [K] in a decision period, which is revealed at the end of the horizon T. For each risk level k, we define pk,ik ∈ (0, 1) to be the treatment probability at time ik ∈ [τ
∗
k
]
. The algorithm is provided with a *soft* budget of bk for each risk level k, representing the total *expected* number of interventions allowed to be delivered at risk level k within each decision period. As before, we assume τ
∗
k > bk for technical convenience (Liao et al., 2018).

Then at each decision time ik, the algorithm decides the intervention probability pk,ik. For each risk level k, the objectives of the online uniform allocation problem are to 1) assign the intervention probabilities {pk,ik}ik∈[τ
∗
k
] as uniform as possible across risk times, and 2) maximize the sum of intervention probabilities across risk times while adhering to the budget constraint bk.

For every risk level k ∈ [K], we define the following optimization problem:

$$\operatorname*{max}\sum_{i_{k}}^{\tau_{k}^{*}}p_{k,i_{k}}-{\frac{1}{\tau_{k}^{*}}}\ln\left({\frac{\operatorname*{max}_{i_{k}\in[\tau_{k}^{*}]}p_{k,i_{k}}}{\operatorname*{min}_{i_{k}\in[\tau_{k}^{*}]}p_{k,i_{k}}}}\right)$$
!
$${\mathrm{s.t.~}}\mathbb{E}\left[\sum_{i_{k}=1}^{\tau_{k}^{*}}p_{k,i_{k}}\right]\leq b_{k}$$
$$p_{k,i_{k}}\in(0,1)\quad\forall i\in[\tau_{k}^{*}].$$
$\eqref{eq:walpha}$. 
]. (4)
Notably, the proposed algorithms offer a feasible solution to the above optimization problem, allowing us to address each risk level independently.

## B. The Penalty Term For Uniformity

We have previously considered statistical distance measures for quantifying the uniformity objective. One important measure is the Kullback-Leibler (KL) divergence. However, this measure is not well defined in our setting since the optimal solution
(which is a point mass on b/τ ∗) and the solutions given by our proposed algorithms are not defined on the same sample space. Recall that for two discrete distributions P and Q defined on the same sample space X , the KL divergence is given by

$$D_{K L}(P\|Q)=\sum_{x\in{\mathcal{X}}}P(x)\log{\frac{P(x)}{Q(x)}},$$

where P represents the data distribution, i.e., the optimal solution, and Q represents an approximation of P, i.e., the solution given by an algorithm. Let us consider a toy example where τ
∗ = b(e − 1). In this case, the optimal solution should be pi =b b(e−1) =1 e−1 for each risk time i ∈ [τ
∗]. The corresponding distribution is a point mass, meaning the sample space X consists of a single element (p1 =1 e−1
, · · · , pτ
∗ =1 e−1
) with probability 1. The solutions given by our proposed algorithms are of the form
(p1, · · · , pτ
∗ ), but the sample space X is Qτ
∗, where the support of Q is (0, 1).

Clearly, the optimal solution and the solutions given by the proposed algorithms are not defined on the same sample space.

Therefore, the KL divergence is not well-defined in this context.

## C. Proof For Algorithm 1

C.1. Proof of Lemma 3.1: Budget constraint Proof. We prove that the budget constraint is satisfied in expectation under each subroutine in Algorithm 1. Subroutine 1 Recall that τ
∗is the true number of risk times. Here, we suppose τ
∗ = βej
∗for some j
∗ ∈ Z
+ and β ∈ [*b, be*]. Since T ≤ be, we have that j
∗ = 0.

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 In this analysis, our focus is solely on the worst-case scenario, where both T and τ
∗are very close to be. Assuming *η > b*
(where η =T
e−1
), if this is not the case, the algorithm uniformly sets pi =
b T
throughout.

When α falls in the range of [*b, η*], Algorithm 1 starts with pi =b α(e−1) with a running length of α, then transitions to pi =
b T
for the second phase with a running length of β − α. Otherwise, it consistently assigns pi =
b T with a running length of β. Therefore, the expected budget consists of two parts: one for α ∈ [*b, η*], where the term inside represents the used budget or the sum of treatment probabilities, and the other for α ∈ [*η, be*], where the term inside represents the sum of treatment probabilities:

## Subroutine 2

Reiterating our initial assumption, we set τ
∗ = βej
∗. The condition *be < T* ≤ be2limits j
∗to either 0 or 1. However, our analysis is particularly concerned with the worst-case scenario, hence we consider only the case where j
∗ = 1.

When α falls in the range of [*b, β*], Algorithm 1 starts with pi =b α(e−1) with a running length of α, transitions to pi =b αe(e−1) with a running length of αe−α, and then continues with pi =
b e 3 with a running length of βe−αe. Otherwise, Algorithm 1 starts with pi =b α(e−1) with a running length of α, transitions to pi =b αe(e−1) with a running length of βe − α.

= bβ T+ b e − 1 ln  η b − bη T + b 2 T ≤ b +b e − 1 ln T b(e − 1)  −b e − 1 + b 2 Tincreasing with β (β = T) ≤ b −b e − 1 +b e − 1 −b e − 1 ln(e − 1) +  b eincreasing with T (T = be) ≤ b −b e − 1 ln(e − 1) +  b e T η e − 1 b T b T
$${\mathfrak{T}}=b.$$
$$\mathbb{E}[\text{Budget}]=\int_{\eta}^{be}\frac{b}{T}\beta\frac{1}{\alpha}d\alpha+\int_{b}^{\eta}\Big{[}\frac{b}{\alpha(e-1)}\alpha+\frac{b}{T}(\beta-\alpha)\Big{]}\frac{1}{\alpha}d\alpha$$ $$=\frac{b\beta}{T}\ln\frac{be}{\eta}+\frac{b}{e-1}\ln\frac{\eta}{b}+\frac{b\beta}{T}\ln\frac{\eta}{b}-\frac{b}{T}(\eta-b)$$
The expected budget is therefore Subroutine 3 Under the assumption of τ
∗ = βej
∗, and given the condition *T > be*2, it is possible for j
∗to be 0 or to extend towards infinity. Our focus, however, is confined to the worst-case scenarios, particularly those where j
∗ ≥ 1.

By combining the above results, we establish Lemma 3.1.

## C.2. Proof Of Theorem 3.1: Competitive Ratio

Proof. In what follows we derive the competitive ratio under each subroutine.

Subroutine 1 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714

E [Budget] = Z be β b α(e − 1)α + b αe(e − 1)(βe − α) 1 α dα + Z β b hb α(e − 1)α +b αe(e − 1)(αe − α) +  b αe3 (βe − αe) i1 α dα =b e − 1 +bβ e − 1 ( 1 β − 1 be  ) −b e(e − 1)  ln  be β + b e ln  β b + bβ e 2 ( 1 b − 1 β ) − b e 2 ln  β b =b e − 1 +b e − 1 −β e(e − 1)  + β e 2 − b e 2 −b e(e − 1)  + (  b e +b e(e − 1)  − b e 2 ) ln  β b =b e − 1 + b e − b e 2 −β e 2(e − 1)  + ( b e − 1 − b e 2 ) ln  β b ≤b e − 1 + b e − b e 2 −be e 2(e − 1)  + ( b e − 1 − b e 2 ) ln  be bincreasing with β (β = be) = 2b e +b e − 1 − 2b e 2 ≈ b
When α falls in the range of [*b, β*]. Algorithm 1 stops with pi =
b(1−1/e)
j
∗
αej∗+2 with a running length of βej
∗−αej
∗. Otherwise, Algorithm 1 stops with pi =
b(1−1/e)
j
∗−1 αej∗+1 with a running length of βej
∗− αej
∗−1. Therefore, the expected budget is

i=2
b(1 − 1/e)
j−2
αej(αej−1 − αej−2) + 
b(1 − 1/e)
j
∗−1
αej∗+1 (βej
∗− αej
∗−1)

b
αe
α +
j
X
∗
E [Budget] = 
Z be
β

1
α
dα

αej
∗+2 (βej
∗− αej
∗)

b
αe
α +
j
X
∗+1
j=2
b(1 − 1/e)
j−2
αej(αej−1 − αej−2) + 
b(1 − 1/e)
j
∗
+
Z β
b

1
α
dα
=
Z be
β
b
e
+ b(1 −
1
e
− (1 −
1
e
)
j
∗) + b(e − 1)j
∗−1
e
j
∗+1
βe − α
α
1
α
dα
+
Z β
b
b
e
+ b(1 −
1
e
− (1 −
1
e
)
j
∗+1) + b(e − 1)j
∗
e
j
∗+2
β − α
α
1
α
dα
=
b
e
+ b(1 −
1
e
) − b(1 −
1
e
)
j
∗ln be
β
− b(1 −
1
e
)
j
∗+1 ln β
b
+
b(e − 1)j
∗−1
e
j
∗+1 
e −
β
b
+ ln 
β
be
+
b(e − 1)j
∗
e
j
∗+2 
β
b
− 1 + ln 
b
β

≤ b − b(1 −
1
e
)
j
∗+1 ln be
b
+
b(e − 1)j
∗
e
j
∗+2 
be
b
− 1 + ln 
b
be
increasing with β (β = be)
= b − b(1 −
1
e
)
j
∗+1 +
b(e − 1)j
∗(e − 2)
e
j
∗+2 ≤ b
Recall that τ
∗represents the true number of available risk times at risk level k, we assume τ
∗ = βej
∗, where j
∗ ∈ Z
+ and β ∈ [*b, be*]. It's evident that when T ≤ be, j
∗ = 0 follows naturally.

Define η = T /(e − 1). Let us first consider the case where η ≤ b, leading to T ≤ b(e − 1). Given that pi =b min(T ,τ˜(e−1)) ,
it follows that the algorithm consistently sets pi =
b T
. Consequently, we have

$$\mathbb{E}[\mathrm{SOL}]={\frac{b}{T}}\beta\geq{\frac{b}{b(e-1)}}\beta\geq{\frac{b}{e-1}}$$
.
Next, let us consider the case where *η > b*. We focus on two cases: (1) *β < η* and (2) β ≥ η.

715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769 Suppose *β < η*. When α falls within [*b, β*], the algorithm initiates with pi =b α(e−1) with a running length of α, then adjusts to pi =
b T
in the subsequent round with a running length of β − α; when α falls in the range of [β, η], the algorithm initiates with pi =b α(e−1) with a running length of β and stops on this stage; otherwise, it consistently uses pi =
b T with a running length of β. Therefore, the expected solution is

E[SOL] =  Z be η b T β 1 α dα + Z β b hb α(e − 1)α + b T (β − α) − σ ln T α(e − 1) i1 α dα + Z η b α(e − 1)β 1 α dα β = bβ T ln  be η +b e − 1 ln  β b + bβ T ln  β b − b T (β − b) +  σ 2 (ln( T β(e − 1)) 2 − ln( T b(e − 1)) 2) +bβ e − 1 ( 1 β − 1 η ) ≥ b 2 T ln  be(e − 1) T+ b e − 1 − b 2 Tincreasing with β (β = b) ≥ b e ln(e − 1) + b e(e − 1) decreasing with T (T = be).
Subroutine 2 Recall that for *be < T* ≤ be2, j
∗is restricted to being either 0 or 1. Below we separately consider these two cases.

Suppose j
∗ = 0. When α falls in the range of [*b, β*], Algorithm 1 begins with pi =b α(e−1) with a running length of α, then transitions to pi =b αe(e−1) with a running length of β − α. Otherwise, Algorithm 1 begins with pi =b α(e−1) with a running

E[SOL] =  Z be η b T β 1 α dα + Z η b hb α(e − 1)α + b T (β − α) − σ ln T α(e − 1) i1 α dα = bβ T ln  be η +b e − 1 ln  η b + bβ T ln  η b − b T (η − b) +  σ 2 (ln( T η(e − 1)) 2 − ln( T b(e − 1)) 2) ≥b e − 1 ln  be η +2b e − 1 ln T b(e − 1)  −b e − 1 + b 2 T − σ 2 ln( T b(e − 1)) 2 increasing with β (β = η = T /(e − 1)) ≥2b e − 1 −b e − 1 ln(e − 1) −b e(e − 1)  − σ 2 ln( e e − 1 ) 2decreasing with T (T = be).
Suppose β ≥ η. It follows that the algorithm always proceeds to the second round. When α falls within [*b, η*], the algorithm initiates with pi =b α(e−1) with a running length of α, then adjusts to pi =
b T
in the subsequent round with a running length of β − α; otherwise, it consistently uses pi =
b T with a running length of β. Consequently, we have length of β and stops. It follows that

≥
increasing with $\beta$ ($\beta=b$).  
b −b e(e − 1)  (ln β − ln b) − σ ln  β b b e + b e ln  β b − σ ln  β b
770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 Suppose j
∗ = 1. When α falls in the range of [*b, β*], Algorithm 1 begins with pi =b α(e−1) with a running length of α, transitions to pi =b αe(e−1) with a running length of αe − α, and then continues with pi =b e3 with a running length of βe − αe. Otherwise, Algorithm 1 begins with pi =b α(e−1) with a running length of α, then transitions to pi =b αe(e−1)
with a running length of βe − α. Therefore, the expected solution is

$$\mathbb{E}[\mathrm{SOL}]=1$$
β b α(e − 1)α + b αe(e − 1)(βe − α) − σ ln e 1 α dα + Z β b b α(e − 1)α + b αe(e − 1)(αe − α) +  b αe3 (βe − αe) − σ ln e 3 e − 1 =b e − 1 ln  be β +bβ e − 1 1 β − 1 be −b e(e − 1)  ln  be β − σ ln  be β +b e − 1 ln  β b + b e ln  β b + bβ e 2 1 b − 1 β − b e 2 ln  β b − σ ln e 3 e − 1 ln  β b = b e ln  be β +b e − 1 −β e(e − 1)  − σ ln  be β +b e − 1 ln  β b + b e ln  β b + β e 2 − b e 2 − b e 2 ln  β b − σ ln e 3 e − 1 ln  β b ≥ b e +b e − 1 −b e(e − 1)  − σ increasing with β (β = b) = 2b e − σ.
Subroutine 3 In the scenarios where *T > be*2, we consider two cases: (1) j
∗ ≥ 1 and (2) j
∗ = 0.

Let us first consider the case where j
∗ ≥ 1. If α ≥ β, the algorithm stops at the j
∗ + 1th round by design of the algorithm
(αej
∗≥ βej
∗); on the other hand, if *α < β*, the algorithm stops at the j
∗ + 2th round (αej
∗+1 ≥ βej
∗). The objective function when α ≥ β is

j=1 b1 − 1 e j−2 αej αej−1 − αej−2+ b1 − 1 e j ∗−1 SOL1 = j X ∗ αej ∗+1  βej ∗− αej ∗−1− σ ln e 2j ∗−1 (e − 1)j ∗−1 = j X ∗ j=1 b(e − 1)j−1 e j+ b(e − 1)j ∗−1 e j ∗+1 βe − α α− σ ln  e 2j ∗−1 (e − 1)j ∗−1 = b   1 − 1 − 1 e j ∗! + b(e − 1)j ∗−1 e j ∗+1 βe − α α− σ ln  e 2j ∗−1 (e − 1)j ∗−1 .
15

$$\mathbb{E}[\text{SOL}]=\int_{\beta}^{b\epsilon}\frac{b}{\alpha(e-1)}\beta\frac{1}{\alpha}d\alpha+\int_{b}^{\beta}\left[\frac{b}{\alpha(e-1)}\alpha+\frac{b}{\alpha e(e-1)}(\beta-\alpha)-\sigma\ln(e)\right]$$ $$=\frac{b\beta}{e-1}\left(\frac{1}{\beta}-\frac{1}{be}\right)+\frac{b}{e-1}\left(\ln\beta-\ln b\right)+\frac{b\beta}{e(e-1)}\left(\frac{1}{b}-\frac{1}{\beta}\right)$$ $$=\frac{b}{\alpha(e-1)}\left(\ln\beta-\ln b\right)-\sigma\ln\frac{\beta}{\alpha}$$
1
α
dα
=
1
α
dα
The objective function when *α < β* is The expected value of our solution is Notice that 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879

b SOL2 f(α)dα = b  1 − 1 − 1 e j ∗+1! Z β b 1 α dα + b(e − 1)j ∗ Z β e j ∗+2  Z β b β − α α 1 α dα − σ ln e 2j ∗+1 (e − 1)j ∗  Z β b 1 α dα = b   1 − 1 − 1 e j ∗+1! ln  β b + b(e − 1)j ∗ e j∗+2  β b − 1 − ln  β b  − σ ln e 2j ∗+1 (e − 1)j ∗ ln β b .
j=1 b1 − 1 e j−2 αej αej−1 − αej−2+ b1 − 1 e j ∗ SOL2 = j X ∗+1 αej∗+2 βej∗ − αej∗− σ ln e 2j ∗+1 (e − 1)j∗ = j X ∗+1 j=1 b(e − 1)j−1 e j+ b(e − 1)j∗ e j ∗+2 β − α α− σ ln  e 2j ∗+1 (e − 1)j ∗ = b   1 − 1 − 1 e j ∗+1! + b(e − 1)j ∗ e j ∗+2 β − α α− σ ln  e 2j ∗+1 (e − 1)j ∗.
$$\mathbb{E}[\mathrm{SOL}]=\int_{\beta}^{b e}\mathrm{SOL}_{1}\,f(\alpha)d\alpha+\int_{b}^{\beta}\mathrm{SOL}_{2}\,f(\alpha)d\alpha.$$
$$({\boldsymbol{5}})$$
SOL2 f(α)dα. (5)
β SOL1 f(α)dα = b   1 − 1 − 1 e j ∗! Z be Z be β 1 α dα + b(e − 1)j ∗−1 e j ∗+1  Z be β βe − α α 1 α dα − σ ln e 2j ∗−1 (e − 1)j ∗−1  Z be β 1 α dα = b   1 − 1 − 1 e j ∗! ln  be β + b(e − 1)j ∗−1 e j ∗+1  e − β b − ln  be β  − σ ln e 2j ∗−1 (e − 1)j ∗−1 ln be β .
16 and Hence, 880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 Tuning parameter selection For Scenario 1) where T ≤ be, the competitive ratio is the

$$\min\left(\frac{1}{e}\left(\ln(e-1)+\frac{1}{e-1}\right),\frac{2}{e-1}-\frac{1}{e-1}\ln(e-1)-\frac{1}{e(e-1)}-\frac{\sigma}{b}(1-\ln(e-1)\right).$$

For Scenario 2) where *be < T* ≤ be2, the competitive ratio is For Scenario 3) where *T > be*2, the competitive ratio is

$$\operatorname*{min}\left({\frac{1}{e}}-{\frac{1}{e^{2}}},1-(1-{\frac{1}{e}})^{j^{*}}+(1-{\frac{1}{e}})^{j^{*}}{\frac{e-2}{e(e-1)}}-{\frac{\sigma}{b}}\ln{\frac{e^{2j^{*}-1}}{(e-1)^{j^{*}-1}}}\right)$$
.
By restricting the value of σ under each scenario and combining the above results, we establish Theorem 3.2. Specifically, when σ =
1 τ∗ , it can be verified that Theorem 3.2 holds.

17

E[SOL] = b  1 − 1 − 1 e j ∗!ln be β + b  1 − 1 − 1 e j ∗+1!ln β b + b(e − 1)j ∗−1 e j ∗+1  e − β b − ln  be β + b(e − 1)j ∗ e j∗+2  β b − 1 − ln  β b  − σ ln e 2j ∗−1 (e − 1)j ∗−1 ln be β − σ ln e 2j ∗+1 (e − 1)j ∗ ln β b = b − b 1 − 1 e j ∗  ln be β + 1 − 1 e ln βb  + b(e − 1)j ∗−1 e j ∗+1  e − β b − ln  be β + e − 1 e β b − 1 − ln  β b  − σ ln e 2j ∗−1 (e − 1)j ∗−1 ln be β − σ ln e 2j ∗+1 (e − 1)j ∗ ln β b ≥ b − b 1 − 1 e j ∗ + b 1 − 1 e j ∗ e − 2 e(e − 1)  − σ ln e 2j ∗−1 (e − 1)j
∗−1increasing with β (β = b).
Now consider the case where j
∗ = 0. When α falls within [*b, β*], Algorithm 1 starts with pi =
b αe with a running length of α, then transitions to pi =b αe2 with a running length of β − α. Otherwise, Algorithm 1 keeps pi =
b αe for β time points. It follows that

E[SOL] = Z be β b αe β 1 α dα + Z β b b αe α +b αe2 (β − α) − σ ln e 1 α dα = bβ e ( 1 β − 1 be  ) +  b e ln  β b + bβ e 2 ( 1 b − 1 β ) − b e 2 ln  β b − σ ln  β b = b e − β e 2 + β e 2 − b e 2 + (  b e − b e 2 ) ln  β b − σ ln  β b ≥ b 1 e − 1 e 2 increasing with β (β = b).
$$\operatorname*{min}\left({\frac{1}{e}},{\frac{2}{e}}-{\frac{\sigma}{b}}\right).$$

## D. Proof For Algorithm 2

D.1. Proof of Lemma **4.1: Budget constraint** Proof. We prove that the budget constraint is satisfied in expectation under each subroutine in Algorithm 2.

Subroutine 4 Let us suppose that τ = L + βej
∗for some j
∗ ∈ Z
+ and β ∈ [*b, be*]. Note that this implicitly implies that τ ≥ L + b, as we only consider the worst case where τ
∗is large enough. Define δ = U − L. Under the condition U ≤ be or δ ≤ b(e − 1), we have j
∗ = 0.

When δ ≤ b, Algorithm 2 would consistently use pi =
b U
, and the budget constraint is satisfied obviously. Now suppose δ > b. When α ∈ [*b, β*], Algorithm 2 begins by setting pi =b α+L with a running length of L + α and then continues with pi =
b U
for the second round with a running length of L + β − L − α; when α ∈ [*β, δ*], Algorithm 2 uses pi =b α+L with a running length of L + β and stops; otherwise, the algorithm sets pi =
b U
all the time. Therefore, the expected budget is 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989

$$\mathbb{E}[\mathrm{Budget}]=\int_{b}^{\beta}\left[\frac{b}{L+\alpha}(L+\alpha)+\frac{b}{U}(L+\beta-L-\alpha)\right]\frac{1}{\alpha}d\alpha+\int_{\beta}^{\delta}\frac{b}{L+\alpha}(L+\beta)\frac{1}{\alpha}d\alpha$$
$$\pm\quad$$
$$\begin{array}{c}\includegraphics[width=140.0pt]{28.45}\end{array}$$  $$\begin{array}{c}\includegraphics[width=140.0pt]{28.45}\end{array}$$  $$\begin{array}{c}\includegraphics[width=140.0pt]{28.45}\end{array}$$  $$\begin{array}{c}\includegraphics[width=140.0pt]{28.45}\end{array}$$  $$\begin{array}{c}\includegraphics[width=140.0pt]{28.45}\end{array}$$  $$\begin{array}{c}\includegraphics[width=140.0pt]{28.45}\end{array}$$  $$\begin{array}{c}\includegraphics[width=140.0pt]{28.45}\end{array}$$  $$\begin{array}{c}\includegraphics[width=140.0pt]{28.45}\end{array}$$  \[\begin{array}{c}\includegraphics[width=140.0pt]{28.  
$$\int_{\delta}^{b e}\frac{b}{U}(L+\beta)\frac{1}{\alpha}d\alpha$$
decreasing with $L\ (L=b)$. 

## Subroutine 5

Similarly, we assume τ
∗ = L + βej
∗. Under the condition that δ ≤ b(e + 1), we have j
∗ = 1 or j
∗ = 0. As before, we only consider the worst case j
∗ = 1.

Let κ =
δ e
. Note that, when α ∈ [b, κ), Algorithm 2 first sets pi =b L+αe with a running length of L + α, then transitions to pk,i =
b U with a running length of L + βe − L − α. However, when α ∈ [*κ, be*], Algorithm 2 keeps setting pi =
b U with a

$$\leq b+b({\frac{e-1}{e}}\ln(e-1)-{\frac{e-2}{e}})$$
$\mathbb{C}$. 
≈ b
$$\leq b\ln(e-1)+b{\frac{e-1}{e}}\ln(e-1)-{\frac{b}{e}}(e-2))+b\ln{\frac{e}{e-1}}$$
e − 1decreasing with L (L = b)
running length of L + βe. Therefore, the expected budget is Subroutine 6 Under the assumption of τ
∗ = L + βej
∗, and given the condition *U > be*2, it is possible for j
∗to be 0 or to extend to infinity. We only focus on the worst-case scenario, i.e., j
∗ ≥ 1.

When α falls within [*b, β*], Algorithm 2 stops with pi = b 1 −L+α−b L+α(e−1)(1−1/e)
j
∗
αej∗+2 with a running length of L +
βej
∗− L − αej
∗. Otherwise, Algorithm 2 stops with pi = b 1 −L+α−b L+α(e−1)(1−1/e)
j
∗−1 αej∗+1 with a running length of L + βej
∗− L − αej
∗−1). Therefore, the expected budget is 990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044

E[Budget] =  Z κ b L + αe (L + α) +  b U (L + βe − L − α) 1 α dα + Z be b U (L + βe) 1 α dα b κ = b ln κ b + (1 e − 1) ln  L + κe L + be  + b U βe ln  κ b − b U (κ − b) +  b(βe + L) Uln  be κ ≤ b ln κ b + (1 e − 1) ln  L + κe L + be  + b U (U − L) ln  κ b − b U (κ − b) + b ln  be κ increasing with β (β = (U − L)/e) ≤ b + b( 1 e − 1) ln  L + b(e + 1) L + be + b 2 L + b(e + 1)(e + 1) ln  e + 1 e− b 2 L + b(e + 1)( e + 1 increasing with U (U = L + b(e + 1))
e− 1)
≤ b + b(
1 − e
eln 
$$\begin{array}{l}{{U\;\left(U=L-1\right)}}\\ {{\frac{e+2}{e+1}+\frac{e-1}{e-1}}}\end{array}$$
$\epsilon$. 
$\ln\dfrac{e}{1}$
e + 1 e + 2
ln 
e + 1
e−
e(e + 2)) decreasing with L (L = b)
≈ b
$\downarrow$ . 
"b
L + α(e − 1)(L + α) + b
1 −L + α − b
L + α(e − 1)
 jX
∗
E[Budget] = Z be
β
j=2
(1 − 1/e)
j−2
αej(αej−1 − αej−2)
+ b
1 −L + α − b
L + α(e − 1)
(1 − 1/e)
j
∗−1
αej
∗+1 (L + βej
∗− L − αej
∗−1)
#1
α
dα
"b
L + α(e − 1)(L + α) + b
1 −L + α − b
L + α(e − 1)
 jX
∗+1
+
Z β
b
j=2
(1 − 1/e)
j−2
αej(αej−1 − αej−2)
+ b
1 −L + α − b
L + α(e − 1)
(1 − 1/e)
j
∗
αej
∗+2 (L + βej
∗− L − αej
∗)
#1
α
dα
≤ b
ln be
β
+ ( 1
e − 1
− 1) ln 
L + be(e − 1)
L + β(e − 1) 
+ b(1 −
1
e
− (1 −
1
e
)
j
∗)
e − 2
e − 1
ln 
L + be(e − 1)
L + β(e − 1)
+ b(1 −
1
e
− (1 −
1
e
)
j
∗)
b
L
ln be
β
−
L + be(e − 1)
L + β(e − 1) 

+ bβej
∗ (e − 1)j
∗
e
2j
∗
1
L
ln be
β
− ln 
L + be(e − 1)
L + β(e − 1) 
− b
(e − 1)j
∗−1
e
j
∗+1 ln 
L + be(e − 1)
L + β(e − 1)
+ b
ln βb
+ ( 1
e − 1
− 1) ln 
L + β(e − 1)
L + b(e − 1) 
+ b(1 −
1
e
− (1 −
1
e
)
j
∗+1)
e − 2
e − 1
ln 
L + β(e − 1)
L + b(e − 1)
+ b(1 −
1
e
− (1 −
1
e
)
j
∗)
b
L
ln βb
−
L + β(e − 1)
L + b(e − 1) 

+ bβej
∗ (e − 1)j
∗+1
e
2j
∗+2
1
L
ln β
b
− ln 
L + β(e − 1)
L + b(e − 1) 
− b
(e − 1)j
∗
e
j
∗+2 ln 
L + β(e − 1)
L + b(e − 1)
≤ b
1 + ( 1
e − 1
− 1) ln 
L + be(e − 1)
L + b(e − 1) 
+ b(1 −
1
e
− (1 −
1
e
)
j
∗+1)
e − 2
e − 1
ln 
L + be(e − 1)
L + b(e − 1)
+ b(1 −
1
e
− (1 −
1
e
)
j
∗)
b
L
1 −
L + be(e − 1)
L + b(e − 1) 

+ b
2(e − 1)j
∗+1
e
j
∗+1
1
L
1 − ln L + be(e − 1)
L + b(e − 1) 
− b
(e − 1)j
∗
e
j
∗+2 ln 
L + be(e − 1)
L + b(e − 1)
increasing with β (β = be)
≤ b
1 + ( 1
e − 1
− 1) ln 
L + be(e − 1)
L + b(e − 1) 
+ b
e − 2
eln 
L + be(e − 1)
L + b(e − 1)
+ b(1 −
1
e
)
b
L
1 − ln L + be(e − 1)
L + b(e − 1) 

≈ b
$$\mathbb{E}[\mathrm{SOL}]={\frac{b}{U}}\tau^{*}=b.$$
Combining the above results with the proof of Subroutine 2, presented in Appendix C.1, establishes Lemma 4.1. D.2. Proof of Theorem **4.2: Consistency and Robustness** Proof. We begin with the proof of consistency and then proceed to the analysis of robustness.

Consistency Analysis It is straightforward to show that our algorithm is 1- consistent. When the width of the predictive interval is zero, meaning that L = U = τ
∗, we have 1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081 1082 1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099 Robustness Analysis Below we show the robustness of our algorithm under each subroutine.