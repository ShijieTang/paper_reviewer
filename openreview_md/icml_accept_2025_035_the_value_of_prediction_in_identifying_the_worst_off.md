# 

Unai Fischer-Abaigar 1 2 **Christoph Kern** 1 2 **Juan Carlos Perdomo** 3

## Abstract

Machine learning is increasingly used in government programs to identify and support the most vulnerable individuals, prioritizing assistance for those at greatest risk over optimizing aggregate outcomes. This paper examines the welfare impacts of prediction in equity-driven contexts, and how they compare to other policy levers, such as expanding bureaucratic capacity. Through mathematical models and a real-world case study on long-term unemployment amongst German residents, we develop a comprehensive understanding of the relative effectiveness of prediction in surfacing the worst-off. Our findings provide clear analytical frameworks and practical, data-driven tools that empower policymakers to make principled decisions when designing these systems.

## 1. Introduction

Faced with pressure to modernize, large bureaucracies are increasingly adopting risk prediction tools to improve efficiency and better serve their populations. Beyond optimizing aggregate outcomes, investments in these programs often aim to address historical inequities and prioritize the needs of the worst-off. For instance, in 2012, Wisconsin launched a risk prediction system to explicitly address deep racial disparities in academic achievement and improve high school graduation rates amongst underserved students. More broadly, such systems are particularly relevant in settings where normative considerations demand prioritizing those at the greatest risk of adverse outcomes, and where well-established downstream interventions can meaningfully benefit these vulnerable individuals.

without reference to the broader social context. The value of a risk predictor is ultimately determined by its impact on bottom-line welfare (e.g., graduation rates) and how these welfare impacts compare to those of other bureaucratic alternatives (Johnson & Zhang, 2022). For example, to understand whether investments in prediction are truly valuable in Wisconsin, we need to assess how much better the risk predictor is at identifying at-risk students relative to existing policies. We also need to understand whether sophisticated prediction systems yield higher graduation rates amongst the underserved than structural investments in teacher training or better facilities. Equity-driven programs are pervasive in applications like social housing, poverty targeting, and unemployment assistance. In these contexts, many government agencies are exploring how algorithmic prediction systems may be an improvement over their current profiling processes (Kortner ¨ & Bonoli, 2023). Yet, due to the absence of an overarching framework that allows the systematic assessment of the relative impacts of different design decisions, efforts to improve predictive accuracy are rarely studied in concert with other policy levers such as expanding screening capacity.

Building on recent work in a budding area of learning in resource allocation contexts, we develop tools to evaluate the design and broader impact of prediction systems that aim to identify the worst-off members of a population. We develop a holistic understanding of the value of statistical prediction in these contexts through theoretical insights into foundational statistical models and a real-world case study on identifying long-term unemployment. Our results establish clear theoretical and empirical criteria characterizing the relative value of core design decisions within these problems. Specifically, we identify when improving prediction provides a higher marginal benefit in helping an institution identify the worst-off. This is compared to alternative strategies, such as keeping prediction accuracy fixed, expanding bureaucratic capacity and screening a larger population. Interestingly, we show that prediction is a first and last-mile effort. The impacts of improving prediction are always outweighed by those of expanded screening capacity, except for when the system explains either none or almost all of the variance in outcomes. While this relationship is moderated by costs, it still largely holds when prediction improvements 1 are more cost-efficient than measures that expand access.

tom 5%, they might have to screen 10% of the population.

These results are counternarrative to current efforts in empirical public policy where agencies focus on incremental improvements within complex prediction systems, starting from the solid baseline performances of their current processes (Desiere et al., 2019; Desiere & Struyven, 2021). Furthermore, implementing more complex profiling systems at scale comes with operational costs (such as staff training and data collection) which need to be contextualized by the cost-benefit ratio of expanding access. Our empirical case study explicates how to systematically assess the relative gains of these design components in a real-world application setting, translating formal insights into critical guidance for designers of these systems. Our results provide theoretically principled and empirically grounded tools for policymakers to make informed decisions when designing prediction systems to identify the worst-off. They also offer a practical framework to help determine how much should be invested in prediction relative to other interventions and how to decide when prediction systems are "good enough" for deployment.

## 1.1. Overview Of Framework And Contributions

Setup. We consider a scenario where a decision-maker seeks to identify worst-off members of a population, as determined by a real-valued welfare metric Y ∈ R, with the goal of prioritizing them for further screening and support.

The population is represented by a distribution D over features X and outcomes Y . The planner aims to identify all individuals whose outcomes Y fall below some threshold t(β), Y ⩽ t(β). Here, β ∈ [0, 1] is a parameter (quantile)
that determines the size of the population that is at risk, Pr[Y ⩽ t(β)] = β. For instance, in poverty prediction, Y is income, and the goal is to identify everyone whose income is below some value. To solve this problem, the social planner has access to data
(X, Y ) ∼ D and builds a screening policy π : *X → {*0, 1}
that determines whether an individual with features x is screened from the broader population to see if they belong to the worst-off group. Learning plays a fundamental role since the optimal policy is to predict each person's expected outcome, f(x) = Yˆ ≈ E [Y | X = x] and screen those in the bottom fraction, πf (x) = 1{f(x) ⩽ t(α)}.

Unpacking this further, α ∈ [0, 1], is a design parameter that determines how many people the planner can screen, Pr[f(x) ⩽ t(α)] = α. The amount of resources α need not be equal to the size of the target population β. For instance, an organization might have normative goal of identifying the poorest 5% of individuals, but only have the resource to screen 1% of the population. Conversely, they might realize that predictions are not perfect, and that to identify the bot-
Given a predictor f, a screening budget of α, and a target parameter β, the value of a prediction system is equal to the fraction of the at-risk population that it identifies, V (α, f; β) = PrD[f(x) ⩽ t(α) | Y ⩽ t(β)],
where again t(α), t(β) are chosen to respect the design constraints. We focus on this notion of value since our driving motivation is to analyze domains like unemployment assistance, or poverty prediction, where there is no harm in the prediction system raising a false positive (π(x) = 1*, Y > t*(β)). By and large, the true value of the system is equal to the extent that it helps an institution efficiently identify the needy amongst a large, diverse population. The focus of our work is to build a holistic understanding of prediction in these contexts by evaluating the relative impacts of different design parameter, such as expanding screening capacity or improving prediction, on this notion of bottom-line value V (*α, f*; β). We develop these insights through theoretical investigations as well as in-depth empirical case study.

Mathematical Results. Following Perdomo (2024), we formalize the relative value of prediction for the worst-off by studying the *prediction-access ratio* or PAR. Intuitively, the PAR measures the relative change in value achieved by optimizing different policy levers.

$\text{PAR}=\dfrac{\text{Marginal Value of Expanding Access}}{\text{Marginal Value of Better Prediction}}$. 
.
We formally define this quantity in Equation 3. While initially developed to specifically study the value of prediction in allocation problems where allocating goods to individuals had heterogeneous effects, here we extend this concept to analyze the value of prediction in a related, but distinct, setting where we aim to identify the worst-off.

Small values of the PAR (i.e. PAR < 1) indicate that small improvements in prediction yield a much larger (relative) impact in the ability to target the worst-off than a small expansion in screening capacity. The opposite is true if the PAR is greater than 1. Calculating this quantity is a fundamental step in deciding which policy lever makes economic sense.
Costs. A full cost-benefit analysis requires combining the
prediction-access ratio with the (marginal) costs of improvements in capacity CAccess and prediction CPred. Once we
factor in costs, it is easy to decide what to focus on. A social planner should expand access whenever
$${\frac{C_{\mathrm{Access}}}{C_{\mathrm{Pred}}}}<\mathrm{PAR}.$$
and invest in better prediction otherwise. The (marginal) costs that competing policy levers carry are inherently context-dependent and will vary across application domains. In many applied settings the cost ratio is comparatively well understood, for example, the salary of an additional caseworker or the cost of a household survey. By presenting PAR separately from costs, we isolate the welfare side of the equation; domain experts can then plug in their own cost estimates to reach a policy decision. In particular, the PAR tells us how much we should be willing to pay for improvements in prediction versus expanding access. We encourage future work to explore scenarios with more complex or less clearly defined cost structures. For instance, many practical applications involve recurring costs, such as ongoing staff salaries or periodic data collection, and fixed costs, such as initial investments in infrastructure or predictive model development. Analyzing how these cost structures affect welfare decisions over time, including amortization of fixed investments or identifying the point at which specific improvements become cost-effective, would significantly enhance our understanding of the relative value of prediction.

To build intuition for the value of prediction in identifying the worst-off, we examine the prediction access ratio in one of the most basic statistical models. The outcomes Y are Gaussian, and the learner has access to a predictor f(x) = Yˆ such that the errors Y − Yˆ are also Gaussian and independent of Yˆ . While extremely simple, the model yields surprisingly precise numerical insights that exactly match up in our real-world case study, where, of course, none of these assumptions hold. In this setting the quality of Yˆ is fully summarized by the coefficient of determination R2 = corr(Y, Yˆ )
2.

Our first result identifies when local improvements in prediction have the highest impact: Theorem 1.1 (Informal, see Theorem 3.2). If α *is at least* a constant, the local improvements in V *with respect to* R2 diverge in two regimes: (1) R2 → 1 and α = β*, or (2)* R2 → 0*. In both cases, the prediction-access ratio satisfies* PAR(α, β) = 0.

Predictions have the highest marginal impact at low and high R2-values, making them a first- and last-mile effort.

Our second result characterizes when the opposite is true.

We prove that whenever screening capacities are severely limited relative to the size of the population one aims to identify α ≪ β , the benefits of increasing α are overwhelming.

Furthermore, it shows that the impacts of improving access are still relatively larger exactly in the regime where most current systems operate: f explains ≈ 20% of the variance and α is equal to, or even slightly larger, than β. Theorem 1.2 (Informal, see Theorem 3.1, Proposition 3.3).

If the predictor f explains an R2fraction of the variance, where R2is at least a constant, then the prediction access ratio is at least Ω(α
−1/(1−R
2))*. Furthermore, if* 0.15 ⩽
R2 ⩽ 0.85 and α ⩽ β or 0.2 ⩽ R2 ⩽ 0.5, β ⩾ 0.15*, and* α ⩽ 0.5 *then the local prediction-access ratio is at least 1.*
Empirical Results. We complement our theoretical discussion by presenting a methodology for policymakers to evaluate the prediction-access ratio in practice. Using a real-world administrative dataset on hundreds of thousands of jobseekers in Germany, we show that our theoretical findings generalize to a more complex, real-world context that closely resembles algorithmic profiling systems widely implemented in many countries. Notably, our results reveal that when considering non-local improvements, expanding screening capacity has an even greater impact compared to enhancing prediction accuracy.

## 1.2. Related Work

Machine learning is increasingly used in the public sector to allocate support by predicting individuals at risk of adverse outcomes (Fischer-Abaigar et al., 2024), with applications spanning a wide range of problem domains (Desiere et al., 2019; Blumenstock, 2016; Perdomo et al., 2023; Chan et al., 2012; Potash et al., 2015; Chouldechova et al., 2018). A large methodological literature draws on decision theory, operations research, economics, and machine learning to learn allocation rules from data (Elmachtoub & Grigas, 2022; Kitagawa & Tetenov, 2018; Manski, 2004; Fernandez-Lor ´ ´ıa & Provost, 2022), with recent work in causal inference focusing on learning treatment policies from observational data (Athey & Wager, 2021; Kallus, 2021). However, many decision-makers rely on separately trained predictive risk scoring-systems to solve "prediction policy problems" (Kleinberg et al., 2015). Recently, this work has been extended using causal inference to train and evaluate these systems (Coston et al., 2023; Guerdan et al., 2023; Boehmer et al., 2024). The widespread use of risk-scoring systems has raised concerns regarding their tradeoffs, pitfalls, and validity (Wang et al., 2024; Coston et al., 2023; Fischer-Abaigar et al., 2024). These concerns include not only questions of empirical performance but also of fairness and equity in how predictive systems shape access to public services (Barocas et al., 2023). Recent work explores alternative design choices—such as employing aggregate rather than individual-level predictions (Shirali et al., 2024), balancing immediate needs with information-gathering (Wilder & Welle, 2024), and introducing randomization (Jain et al., 2024)—to improve downstream outcomes. Perdomo (2024) studies the prediction-access ratio under both linear and probit models, with the latter closely related to our work. While they focus on binary welfare outcomes, we adopt a continuous welfare metric and a distinct policy objective: rather than evaluating changes in overall expected welfare, we measure the fraction of truly worst-off individuals who are identified. This captures a mathematically and conceptually distinct setting frequently encountered in the public sector. For instance, employment agencies often prioritize identifying and assisting individuals in greatest need, rather than optimizing average employment outcomes across all jobseekers. In addition, we introduce a set of empirical tools to analyze these tradeoffs in practice, while the work of Perdomo (2024) is purely theoretical.

## 2. Formal Framework

We start by formally defining our screening problem. Definition 2.1 (Screening Problem). The screening problem seeks to identify a decision rule π : R → {0, 1} that fraction of the worst-off population that is identified while adhering to resource constraints α ∈ (0, 1) that bound the percentage of the population that can be screened by the social planner:

$$\operatorname*{max}_{\pi:{\frac{\mathbb{R}\to\{0,1\}}{2}}}\mathbb{E}\left[\pi({\hat{Y}})=1\mid Y\leqslant F_{Y}^{-1}(\beta)\right]{\mathrm{s.t.}}\,\mathbb{E}\left[\pi({\hat{Y}})\right]\leqslant\alpha$$

The quantile F
−1 Y(β) denotes the welfare cutoff that identifies the worst-off β ∈ (0, 1) fraction of the population.

Given perfect knowledge of the welfare outcomes Yˆ = Y ,
the optimal decision policy is simple: rank individuals based on their outcomes Y and intervene in the bottom α-fraction of the population. In the general case, we have: Proposition 2.2. *The optimal policy* π
∗: R → {0, 1} to solve the screening problem (Definition *2.1) is equal to* π
∗(Yˆi) = 1{s(Yˆi) ⩾ F
−1 s(1 − α)} *where* F
−1 s(1 − α) is the (1 − α)*-quantile of* s(Yˆ ) = Pr[Y ⩽ F
−1 Y(β) | Yˆ ].

Policy Value in Gaussian Setting. For the theoretical investigation, we assume independent, identically distributed errors ε = Y −Yˆ
iid
∼ N (0, γ2) that are independent of Yˆ . In this setting, the screening problem can be solved by ranking individuals in ascending order of their predicted outcomes Yˆ and screening the bottom α-fraction (see Proposition C.1),
achieving the policy value:

$$V(\pi^{*})=\Pr[\hat{Y}\leq F_{\hat{Y}}^{-1}(\alpha)\mid Y\leq F_{Y}^{-1}(\beta)]\tag{1}$$

In addition, we assume welfare outcomes Y ∼ N (*µ, η*2).

Because ε is independent of Yˆ , this implies that Y and Yˆ
follow a bivariate normal distribution. Proposition 2.3. *(Policy Value in Gaussian Setting) Let* Y − Yˆ
iid
∼ N (0, γ2) and Y ∼ N (µ, η2)*, then the value* V (π
∗) *of the optimal screening policy* π
∗*is given by*

$$V(\pi^{*})=V(\alpha,\beta,R^{2})=\frac{\Phi_{2}\big(\Phi^{-1}(\alpha),\Phi^{-1}(\beta);\rho\big)}{\beta}$$

β(2)
where Φ2 (·) *denotes the bivariate standard normal CDF*
with correlation ρ =pη2 − γ2/η and Φ
−1(·) is the quantile function of the standard normal distribution.

In this model, the goodness of the predictions Yˆ are entirely captured by the coefficient of determination R2, which equals the squared correlation ρ 2 between Y and Yˆ .

Our analysis extends to the log-normal distribution log Y ∼
N (*µ, η*2) under a a multiplicative error model Y = Yˆ · u with log u ∼ N (0, γ2). Taking logarithms, leads to log Y = log Yˆ + log u. Since the logarithm is strictly increasing, the ordering of Y and Yˆ is preserved under transformation. This allows us to apply the same framework to the log-transformed variables log Y and log Yˆ . This extension is particularly useful because many welfare outcomes, such as income distributions (Clementi & Gallegati, 2005), can be approximated by a log-normal distribution.

Visualization. For a given screening capacity α and R2 value, we can illustrate the corresponding screening policy by plotting the probability PrYˆ ⩽ F
−1 Yˆ(α) | Y = y	that an individual with welfare outcome Y = y is screened. As shown in Figure 1, lower values of Y correspond to higher probabilities of being screened. We focus on evaluating how effectively the screening policy identifies individuals in the worst-off segment of the population (i.e., on the left side of the β cutoff).

## 3. Theoretical Results

The decision-maker has (at least) two pathways to raise the policy value, which we refer to as *policy levers*:
- **Expanding Access** Increasing the screening threshold from α to α + ∆α. If full screening were possible
(α = 1), the β-fraction would be fully identified, as V (π
∗) = 
Φ2(Φ
−1(α),Φ
−1(β);ρ)
β =
Φ(Φ
−1(β))
β = 1.

- **Improving Predictions** Investing in better predictive models, modeled as increasing R2to R2 +
∆R2 . Perfect predictions (R2 = 1) leads to optimal allocation of available capacities: V (π
∗) =
1 β Φmin(Φ−1(α), Φ
−1(β)).

Figure 1 showcases improvements in access and prediction. Increasing capacity expands the fraction of the population screened, while improving R2shifts probability mass across the β threshold, enhancing targeting accuracy. Following Perdomo (2024), a key quantity of interest is the prediction-access ratio (PAR), which quantifies the relative improvements in policy value from enhancing predictions versus improving access to screening. Specifically, the PAR is defined as:

$$\mathbf{PAR}={\frac{V(\alpha+\Delta_{\alpha},\beta,R^{2})-V(\alpha,\beta,R^{2})}{V(\alpha,\beta,R^{2}+\Delta_{R^{2}})-V(\alpha,\beta,R^{2})}}$$
(3) $\frac{1}{2}$  . 
$$(2)$$

In other words, the PAR can inform a social planner how much more they should be willing to pay for improvements in screening capacity relative to prediction. For example, a PAR > 2 implies that expanding the screening capacity by

−2 0 2 Y
P

(

^
Y

≤

F

−

^
Y

1

(

α

) | Y

=

y

)
β = 0.2 α + ∆α β = 0.2
−2 0 2 Y
P

(

^
Y

≤

F

−

^
Y

1

(

α

) | Y

=

y

)
β = 0.2 R2 + ∆R2 0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00
−2 0 2 Y
P

(

^
Y

≤

F

−

^
Y

1

(

α

) | Y

=

y

)

∆α yields at least twice the increase in policy value compared to investing in improved predictions by ∆R2 . Consequently, the social planner should prioritize investments in screening capacity, provided the costs of doing so are not more than double those of improving predictions.

## 3.1. **Theoretical Bounds For The Prediction-Access Ratio**

In our setting, direct calculation of the PAR is challenging due to the policy value being analytically intractable and the problem featuring strong non-linearities. We derive bounds for specific cases and regimes that we consider particularly insightful, with a focus on marginal local improvements. In our empirical investigation, we find that the main results generalize well to a more complex, real-world setting. What should priorities be if screening is very limited?

Theorem 3.1 (PAR for Small Screening Capacities). For any 0 < R2 < 1, ∆R2 , ∆α > 0 and 0 < β ⩽ 0.5 *there* exists a threshold t(β, R2, ∆R2 ) *such that for any* α+∆α ⩽
t, PAR(α, R2, ∆α, ∆R2 ) *is at least*

$$\frac{\Delta_{\alpha}}{\Delta_{R^{2}}}\sqrt{R^{2}(1-R^{2})}\left(5.1\cdot\alpha\Phi^{-1}\left(1-\alpha\right)\right)^{-\frac{1}{1-R^{2}}+o(1)}$$

where o(1) goes to zero as α *approaches zero.*
Suppose the available screening capacity α + ∆α is very small (α + ∆α ≪ β), and assume there is a baseline level of predictability (i.e., R2is bounded away from 0). Then Theorem 3.1 implies that the PAR can become very large. Specifically, for small α, Φ
−1(1 − α) grows asymptotically like plog (1/α). Consequently, the polynomial growth of α
−1/(1−R
2) drives the PAR to increase rapidly as α decreases. It follows that in the scarce capacity regime, expanding the screening capacity has a far greater impact than improvements in prediction accuracy.

## When Does Prediction Have The Highest Impact?

Theorem 3.2 (Maximally Effective (Local) Prediction Improvements). Let 0 < β < 1 be fixed and 0 *< α <* 1.

Consider the points that maximize the local rate of change in policy value V *with respect to improvements in* R2:

$(\alpha_{*},R_{*}^{2})=\underset{(\alpha,R^{2})\in(0,1)\times(0,1)}{\arg\max}\underset{\Delta\to0}{\lim}\frac{V(\alpha,\beta,R^{2}+\Delta)-V(\alpha,\beta,R^{2})}{\Delta}$
The local improvements in V diverge—and are maximizedin two regimes: (1) R2∗ → 1, α∗ = β, and (2) R2∗ → 0*. For* both regimes, setting ∆R2 = ∆α = ∆, the local predictionaccess ratio satisfies lim∆→0 PAR(*α, β,* ∆) → 0.

According to Theorem 3.2, marginal improvements in prediction are most impactful in two distinct regimes. First, when predictive capacity is very low, even a small initial investment can lead to disproportionately large improvements, provided that a minimal baseline of screening capacity is present. Second, as R2approaches one, further marginal improvements can also have a significant relative impact, specifically around the point where the screening capacity α matches the requirements for screening the entire β-segment of the population. See Figure 2. When are small increases in screening capacity more impactful than improving predictions?

Proposition 3.3 (PAR for Local Improvements). Let R2, β, and α satisfy either R2 ∈ (0.15, 0.85), β ∈ (0.03, 0.5),
and α ⩽ β, or R2 ∈ (0.2, 0.5), β ⩾ 0.15, and α ⩽ 0.5*. If*
∆R2 = ∆α = ∆*, then* lim∆→0 PAR(*α, β,* ∆) ⩾ 1.

We find that the PAR remains above one as long as α ⩽ β and R2is not too extreme. For larger β values (i.e., β ⩾
0.15) the PAR stays above one even for large α provided R2remains in a moderate range. Crucially, this represents the standard parameter regime in which most allocation programs operate, characterized by a moderate baseline of predictions and resource levels comparable to β. Numerical Simulations. We complement our theoretical investigation with numerical simulations of the PAR for different α, β and R2 values (see Figure 2). Consistent with our theoretical results, the PAR becomes large for small screening capacities (α ≪ β) and remains above one for α ⩽ β, provided a small baseline level of predictive performance has been established. The bounds in Proposition 3.3 are conservative, with PAR > 1 observed for a broad range of R2 values. Prediction improvements are particularly impactful when R2is small. Although the PAR falls below one in the high-R2and high-α regime, allocation is nearly perfect, making further improvements a "last mile" effort. Discussion. We found several insights relevant to policymakers aiming to iteratively improve a screening system. First, establishing a baseline level of predictive performance is usually a good starting point. Once this is achieved, expanding the screening capacity becomes the next priority. For very small capacities, Theorem 3.1 tell us that the PAR can increase significantly, making investments in screening capacity highly impactful. Generally, expanding capacity to at least the level where everyone in need could hypothetically be screened (α ⩾
β) is likely cost-efficient. Once both screening capacity and predictive accuracy are high and the allocation system is close to optimal, improvements in prediction become relatively more valuable again for perfecting the system.

However, this regime may rarely be reached in practice.

In Figure 2, we display the PAR for a cost ratio of 1/4. As expected, the regions where investing in R2is more efficient expand, and some of the earlier nonquantitative bounds no longer apply. Nevertheless, the key insights remain consistent: when screening capacities are small, investments in expanding them are very effective, while improvements in R2are more important when predictive accuracy is low.

## 4. Empirically Evaluating The Par

While our theory offers broad intuition when expanding screening capacity or improving predictions is most effective, policymakers need practical tools for their own systems. To support this, we develop a methodology to compute and interpret the prediction-access ratio using empirical data, helping social planners identify the most efficient policy levers for their unique problem context. Policy Value. As before, we define the allocation policy's value as the probability that the worst-off individuals are successfully identified:, i.e. V (α, β) = Pr[Yˆ ⩽ F
−1 Yˆ(α) | Y ⩽ F
−1 Y(β)]. In practice, this can be measured using a recall-like metric, capturing the proportion of truly at-risk individuals screened by the policy.

$$V(\alpha,\beta)\approx{\frac{\sum_{i=1}^{n}1\{\hat{Y}_{i}\leqslant F_{Y,n}^{-1}(\alpha)\}1\{Y_{i}\leqslant F_{Y,n}^{-1}(\beta)\}}{\sum_{i=1}^{n}1\{Y_{i}\leqslant F_{Y,n}^{-1}(\beta)\}}}$$

Increasing Screening Capacity. Given a chosen ∆α the policy improvement can be directly computed V (α +
∆α, β)−V (α, β) by recalculating the empirical policy value at the new threshold. For example, in cash transfer programs
(Blumenstock, 2016), a key question is how many resources α
∗are required to reach a specified fraction p of poor households, i.e. α
∗ = infα∈(0,1){α: V (*α, β*) ⩾ p}.

Improving Predictions. A decision-maker can improve a model's predictions through various pathways:
a) **Data Collection** Collect additional samples and increase the frequency of data collection. Social prediction systems are often vulnerable to distribution shifts over time in dynamic and evolving environments (Fischer-Abaigar et al., 2024; Aiken et al., 2023).

b) **Data Quality** Improve data quality (i.e., reduce errors and missing data) by means such as standardizing data collection processes, implementing centralized data management systems, and offering targeted training programs for staff.

c) **Collect Additional Features** In government, this may involve integrating separate data sources across institutions (Sun & Medaglia, 2019; Wirtz et al., 2019).

d) **Advanced Modeling Techniques** Utilize more sophisticated modeling techniques, which might capture more complex patterns in the data but are often more costly to operationalize.

In resource-constrained settings, planners often focus on incremental improvements rather than rebuilding entire systems. For instance, collecting a small amount of additional data may boost R2 by a few points, uniformly reducing errors. To simulate such minor gains, we scale the model's residuals Yˆ+ = Yˆ + δ(Y − Yˆ ), choosing δ ∈ (0, 1) so that R2increases by a target ∆R2 (see Appendix B.3). This preserves the overall error structure, allowing us to gauge how a
"similar but slightly better" model affects policy outcomes.

This approach can be extended in several ways. For example, residuals could be adjusted for specific subgroups to account for uneven prediction improvements (e.g., targeted data collection for rural or underrepresented populations). Alternatively, planners could retrain models under different conditions—such as sample size, feature set, or architecture—and compare the resulting policy value.

## 5. Case Study: Identifying Long-Term Unemployment In Germany

Public employment services (PES) across the globe make use of profiling approaches to identify jobseekers at risk of long-term unemployment to target preventative measures (Loxha & Morgandi, 2014). Starting from traditional rulebased approaches, many PES either test or already deploy algorithmic profiling to identify jobseekers in need of support (Desiere et al., 2019; Kortner & Bonoli ¨ , 2023). While these profiling tools assist in allocating programs that account for large shares of PES spending - making design choices critical (Kern et al., 2024) - systematic assessments of their relative value compared to other measures for

0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00 0.5 1.0 1.5 2.0 β = 0.2 V(α, β, R2) > 0.9 Pred icti ons
: 
R

2 Pred icti ons
: 
R

2 V(α, β, R2) > 0.9 β = 0.2 0.0 0.1 0.2 0.3 0.4 Screening Capacities: α 0.0 0.1 0.2 0.3 0.4 Screening Capacity: α
(b) 1/4 × PAR
(a) PAR
0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 Screening Capacity: α 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 Screening Capacity: α 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 Screening Capacity: α 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 β values 0.05 (≈ 24 mos.)
0.15 (≈ 12 mos.)
0.25 (≈ 8 mos.)
β values 0.05 (≈ 24 mos.)
0.15 (≈ 12 mos.)
0.25 (≈ 8 mos.)
β values 0.05 (≈ 24 mos.)
0.15 (≈ 12 mos.)
0.25 (≈ 8 mos.)
PAR
PAR
PAR
(a) R
2 = 0
(b) R
2 = 0.15
(c) R
2 = 0.9

## Improving Jobseekers' Outcomes Remain Absent.

We secured access to a dataset1 on German jobseekers derived from German administrative labor market records that cover a large portion of the German labor force. It merges multiple administrative data sources, containing a wide spectrum of individual labor market information - including records on employment histories, received benefits, unemployment periods, participation in job training programs and demographic information. Such administrative records are the primary data source used by PES to build algorithmic profiling models (Bach et al., 2023). Experimental Setup. We train a model to predict how long a newly registered jobseeker remains unemployed, defining the target Y as unemployment duration in days (capped at 24 months). Following Bach et al. (2023), we use a set of covariates capturing demographic information, labor market history, and most recent job details. To ensure full 24-month observations and mimic a realistic deployment scenario, we focus on unemployment spells beginning between 2010 and 1The dataset is provided via a Scientific Use File by the Research Data Centre (FDZ) of the German Federal Employment Agency (BA) at the Institute for Employment Research (IAB) (Schmucker & vom Berge, 2023a;b).

2015, resulting in data on 274,515 different jobseekers and 553,980 unemployment spells. We refer to Appendix B.1 for additional information on the experimental setup and data. Our focus is the β-fraction of jobseekers with the longest expected unemployment durations, representing those most at risk. In Germany, being unemployed for over one year (about 15% of cases in our data; Figure 8) meets the legal definition of long-term unemployment (Bach et al., 2023), but some countries adopt different cutoffs (Desiere et al., 2019).

## 5.1. Results

We train a CatBoost model (see Appendix B.2 for details), achieving an R2 of 0.15 on the test set. This level of predictive power aligns well with what is typically observed in social prediction tasks (Salganik et al., 2020) and similar applied settings (Desiere et al., 2019). How much does the screening capacity need to increase to target a significant fraction of high-risk jobseekers? As expected, larger screening capacities increase both the policy value and the number of high-risk jobseekers screened (see Figure 4(a)). Focusing on the (German) LTU
cutoff (β ≈ 0.15), our policy value aligns well with findings of previous studies2(Bach et al., 2023).

A planner might begin by setting α = β, ensuring that, in theory, enough capacity is provided to screen and support every high-risk jobseeker. A natural question then arises: how much additional capacity ∆α would be required to screen at least a specified percentage of high-risk individuals? This additional capacity represents the overhead that must be invested to account for imperfect predictions. We observe that the ∆α required to ensure at least 75% of high-risk jobseekers are screened remains consistently around 0.25 across different β values. While the policy value increases as α = β rises, the marginal improvements gained from increasing access decrease for higher α, resulting in a somewhat stable ∆α across β. In practice, this means we need to screen 25% more of the population to ensure adequate coverage. What is the impact of improving screening capacity versus prediction errors? We simulate small improvements in the R2 value by uniformly scaling the residuals by a multiplicative factor. To ensure that this approach approximates a realistic pathway of (marginally) improving the model, we train various models at different sample sizes.

We then verify that as R2increases with the amount of training data, the variance of the residuals decreases, while the distribution remains largely unchanged in shape (see Figure 12). We then evaluate the prediction-access ratio for ∆R2 = ∆α = 0.1 in three scenarios : (1) the trained CatBoost model with R2 = 0.15, (2) near-perfect predictions with R2 = 1 − ∆R2 and (3) constant predictions
(R2 = 0), effectively randomizing screening decisions.

We observe a rise in the PAR for small screening capacities α (see Figure 3), consistent with Theorem 3.1. Under random allocation (R2 = 0), the PAR stays below one for α ⩾ 0.1. This result aligns somewhat with Theorem 3.2, where we found that the (local) PAR approaches zero as R2 → 0. Because we consider ∆ = 0.1 (rather than an infinitesimal improvement, see Figure 13 for ∆ = 0.01), the PAR remains large at small α. For the CatBoost model
(R2 = 0.15), capacity improvements stay relatively more effective (i.e., PAR > 1) for larger α, matching Proposition 3.3, where we found that for moderate R2and α ⩽ β, the local PAR remains above one. Meanwhile, near-perfect predictions (R2 = 0.9) make capacity investments highly efficient, causing the PAR to diverge for *α < β*, then drop sharply near α = β because the allocation becomes nearly optimal. When α ⩾ β, the PAR stabilizes at one as numerator and denominator both approach zero. These observations broadly match our theoretical findings, despite the non-local improvements and more complex residual structure. Notably, the theory's focus on local improvements offers a conservative perspective on capacity investments: even under random allocation (R2 = 0), securing a modest screening capacity (5−10%) is often the first priority, while at very high R2, gains in policy value diminish so rapidly once α ⩾ β that the relative advantage of further prediction investments becomes negligible. When do small improvements in prediction error have the largest impact? From theory (Theorem 3.2), we expect local policy value improvements from better predictions to diverge as R2 → 0 and R2 → 1 when α = β. This aligns with our results in Figure 4: for small ∆R2 , the rate of local improvements in V (R2) with respect to R2 diverges. The location of the maximum in α also follows from the theory:
as R2 → 1, the rate only diverges for α = β, while for small R2the maximum is at α ≈ 0.5.

What are the relative benefits and tradeoffs of using a simpler vs more complex prediction model? We compare a shallow 4-depth decision tree with the CatBoost model. As expected, the simpler tree shows a small drop in predictive power (5% decrease in R2) which translates into a 1–8% reduction in policy value (see Figure 15). Compared to a uniform 5% increase in R2achieved by scaling the residuals (see Figure 14), the differences in policy value are only partially similar across α. The CatBoost model does not provide a uniform improvement over the decision tree; for instance, it performs better at distinguishing longer unemployment spells. Despite this performance gap, the simpler model offers potential advantages: it fits on a single sheet of paper, demands minimal computational infrastructure, can be easily explained to frontline case workers and resembles the categorical prioritization rules common in public institutions. (Johnson & Zhang, 2022). Because more complex models incur higher costs, a planner might instead increase screening capacity. Formally, we define

$$\Delta_{\alpha}^{*}=\operatorname*{inf}_{\Delta_{\alpha}\in(0,1-\beta)}\left\{\Delta_{\alpha}:{\frac{V_{\mathrm{m}}}{V_{\mathrm{m}}}}\right\}$$

VTREE(α+∆α,β)−VTREE(α,β)
VCAT(α,β)−VTREE(α,β) ⩾ 1 o the smallest ∆∗α that matches the policy-value gains of the CatBoost model. Empirically, ∆∗α mostly rises with α
(see Figure 15), consistent with our finding that the PAR decreases with α. By framing the difference between models in terms of additional screenings, planners can directly compare the cost of increased capacity to that of deploying a more complex model.

## 6. Conclusion

This paper develops a framework for quantifying the relative value of prediction in identifying the worst-off. We formalize tradeoffs between expanding screening capacity and

0.0 0.1 0.2 0.3 0.4 0.5 Screening Capacity: α 0.0 0.2 0.4 0.6 0.8 1.0 Policy Val ue V

(α, β
)

β values 0.05 (≈ 24 mos.) 0.15 (≈ 12 mos.) 0.25 (≈ 8 mos.)
β values 0.05 0.15 0.25 β values 0.05 0.15 0.25 0.00 0.05 0.10 0.15 0.20 0.25 0.30 Screening Capacity: α 10−3 10−2 10−1 100 101 0.00 0.05 0.10 0.15 0.20 0.25 0.30 Screening Capacity: α 10−1 100 101 102 V (R
2+∆
R

2)−
V (R
2) 
∆

R

2 V (R
2+∆
R

2)−
V (R
2) 
∆

R

2

∆R2 0.1 0.01 0.001
∆R2 0.1 0.001
(b) R
2 = 0
(c) R
2 = 1 − ∆R2
(a) Policy Value Scree nin g Ca pa city Ga p β values 0.05 (≈ 24 mos.)
0.15 (≈ 12 mos.)
0.25 (≈ 8 mos.)
0.0 0.1 0.2 0.3 0.4 0.5 Screening Capacity: α 0.01 0.02 0.03 0.04 0.05 0.06 0.07
improving predictive models, and show through both mathematical analysis and a real-world case study that prediction is not always the most important piece of the puzzle in social allocation systems. Future work could examine more specific application settings and cost structures, including distinctions between fixed and recurring costs, and explore policy levers that improve prediction unevenly, for example, by reducing errors in high-risk subgroups or by increasing robustness to distributional shifts. More broadly, we see a need for clearer theoretical foundations to understand the role of prediction in public-sector allocation, particularly in relation to the institutional and administrative systems in which it is embedded.

## Acknowledgements

This work is supported by the DAAD programme Konrad Zuse Schools of Excellence in Artificial Intelligence, sponsored by the Federal Ministry of Education and Research and by the Volkswagen Foundation, grant "Consequences of Artificial Intelligence for Urban Societies (CAIUS)". Juan Carlos Perdomo is supported by the Center for Research on Computation and Society (CRCS) at Harvard University and by the Alfred P. Sloan Foundation grant G-2020-13941.

We would like to thank the anonymous reviewers for their insightful comments, as well as Frauke Kreuter, Patrick Schenk and Moritz Hardt for their valuable feedback.

## Impact Statement

Our work offers a principled framework for evaluating the relative benefits of using predictive models to target the most vulnerable populations, helping public agencies allocate limited resources more effectively. However, formalizing complex institutional processes inevitably omits important real-world details, risking biases or misalignments if assumptions are not carefully examined. We encourage policymakers and researchers to incorporate fairness, transparency, and accountability measures when implementing these methods, particularly in resource-constrained contexts where small design changes can disproportionately affect marginalized communities.

## References

Aiken, E., Ohlenburg, T., and Blumenstock, J. Moving targets: When does a poverty prediction model need to be updated? In Proceedings of the 6th ACM SIGCAS/SIGCHI Conference on Computing and Sustainable Societies, COMPASS '23, pp. 117, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400701498. doi: 10.1145/
3588001.3609369. URL https://doi.org/10. 1145/3588001.3609369.

Athey, S. and Wager, S. Policy Learning with Observational Data. *Econometrica*, 89(1):133–161, 2021.

Bach, R. L., Kern, C., Mautner, H., and Kreuter, F. The impact of modeling decisions in statistical profiling. Data & Policy, 5:e32, 2023. doi: 10.1017/dap.2023.29.

Barocas, S., Hardt, M., and Narayanan, A. Fairness and Machine Learning: Limitations and Opportunities. MIT
Press, 2023.

Blumenstock, J. E. Fighting Poverty with Data. *Science*,
2016.

Fischer-Abaigar, U., Kern, C., Barda, N., and Kreuter, F.

Bridging the Gap: Towards an Expanded Toolkit for Aidriven Decision-making in the Public Sector. *Government* Information Quarterly, 41(4):101976, 2024. ISSN 0740624X. doi: https://doi.org/10.1016/j.giq.2024.101976. URL https://www.sciencedirect.com/ science/article/pii/S0740624X24000686.

Boehmer, N., Nair, Y., Shah, S., Janson, L., Taneja, A., and Tambe, M. Evaluating the Effectiveness of Index-Based Treatment Allocation. *arXiv preprint arXiv:2402.11771*, 2024.

Chan, C. W., Farias, V. F., Bambos, N., and Escobar, G. J.

Optimizing Intensive Care Unit Discharge Decisions with Patient Readmissions. *Operations Research*, 60(6):1323– 1341, 2012. doi: 10.1287/opre.1120.1105. URL https: //doi.org/10.1287/opre.1120.1105.

Guerdan, L., Coston, A., Holstein, K., and Wu, Z. S. Counterfactual Prediction Under Outcome Measurement Error. In Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency, FAccT '23, pp. 1584–1598, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400701924. doi: 10.1145/3593013.3594101. URL https://doi. org/10.1145/3593013.3594101.

Chouldechova, A., Benavides-Prado, D., Fialko, O., and Vaithianathan, R. A case study of algorithm-assisted decision making in child maltreatment hotline screening decisions. In Conference on Fairness, Accountability and Transparency, pp. 134–148. PMLR, 2018.

Jain, S., Creel, K., and Wilson, A. C. Position: Scarce Resource Allocations That Rely On Machine Learning Should Be Randomized. In Forty-first International Conference on Machine Learning, 2024. URL https: //openreview.net/forum?id=44qxX6Ty6F.

Clementi, F. and Gallegati, M. Pareto's law of income distribution: Evidence for Germany, the United Kingdom, and the United States. *Econophysics of wealth distributions:* Econophys-Kolkata I, pp. 3–14, 2005.

Johnson, R. A. and Zhang, S. What is the Bureaucratic Counterfactual? Categorical versus Algorithmic Prioritization in U.S. Social Policy. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, FAccT '22, pp. 1671–1682, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450393522. doi: 10.1145/
3531146.3533223. URL https://doi.org/10. 1145/3531146.3533223.

Coston, A., Kawakami, A., Zhu, H., Holstein, K., and Heidari, H. A Validity Perspective on Evaluating the Justified Use of Data-driven Decision-making Algorithms.

In 2023 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pp. 690–704, 2023. doi: 10.1109/SaTML54575.2023.00050.

Desiere, S. and Struyven, L. Using Artificial Intelligence to classify Jobseekers: The Accuracy-Equity Tradeoff. *Journal of Social Policy*, 50(2):367–385, April 2021. ISSN 0047-2794, 1469-7823. doi: 10.1017/ S0047279420000203.

Kallus, N. More Efficient Policy Learning via Optimal Retargeting. *Journal of the American Statistical* Association, 116(534):646–658, 2021. doi: 10.1080/ 01621459.2020.1788948. URL https://doi.org/ 10.1080/01621459.2020.1788948.

Desiere, S., Langenbucher, K., and Struyven, L. Statistical Profiling in Public Employment Services: An International Comparison. Technical Report 224, OECD Publishing, 2019. URL https://doi.org/10.1787/ b5e5f16e-en.

Kern, C., Bach, R., Mautner, H., and Kreuter, F. When Small Decisions Have Big Impact: Fairness Implications of Algorithmic Profiling Schemes. ACM Journal on Responsible Computing, 1(4), November 2024. doi: 10.1145/3689485. URL https://doi.org/10. 1145/3689485.

Drezner, Z. and Wesolowsky, G. O. On the Computation of the Bivariate Normal Integral. Journal of Statistical Computation and Simulation, 1990.

Kitagawa, T. and Tetenov, A. Who Should Be Treated? Empirical Welfare Maximization Methods for Treatment Choice. *Econometrica*, 86 (2):591–616, 2018. doi: https://doi.org/10.3982/ ECTA13288. URL https://onlinelibrary.

wiley.com/doi/abs/10.3982/ECTA13288.

Elmachtoub, A. N. and Grigas, P. Smart "predict, then optimize". *Management Science*, 68(1):9–26, 2022.

Fernandez-Lor ´ ´ıa, C. and Provost, F. Causal Decision Making and Causal Effect Estimation Are Not the Same. . . and Why It Matters. *INFORMS Journal on Data Science*, 1(1): 4–16, 2022. doi: 10.1287/ijds.2021.0006. URL https:
//doi.org/10.1287/ijds.2021.0006.

Kleinberg, J., Ludwig, J., Mullainathan, S., and Obermeyer, Z. Prediction Policy Problems. *American Economic* Review, 105(5):491–495, 2015.

Kortner, J. and Bonoli, G. Predictive Algorithms in the ¨
Delivery of Public Employment Services. In Handbook of Labour Market Policy in Advanced Democracies, pp. 387–398. Edward Elgar Publishing, 2023.

Loxha, A. and Morgandi, M. Profiling the unemployed: a review of OECD experiences and implications for emerging economics. *Social protection discussion papers and* notes, (91051), 2014.

Manski, C. F. Statistical Treatment Rules for Heterogeneous Populations. *Econometrica*, 72(4):1221–1246, 2004. doi: https://doi.org/10.1111/j.1468-0262.2004.00530.x. URL https://onlinelibrary.wiley.com/doi/ abs/10.1111/j.1468-0262.2004.00530.x.

Perdomo, J. C. The Relative Value of Prediction in Algorithmic Decision Making. In Proceedings of the 41st International Conference on Machine Learning, ICML'24. JMLR.org, 2024.

Perdomo, J. C., Britton, T., Hardt, M., and Abebe, R. Difficult Lessons on Social Prediction from Wisconsin Public Schools. *arXiv preprint arXiv:2304.06205*, 2023.

Potash, E., Brew, J., Loewi, A., Majumdar, S., Reece, A., Walsh, J., Rozier, E., Jorgenson, E., Mansour, R., and Ghani, R. Predictive Modeling for Public Health: Preventing Childhood Lead Poisoning. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '15, pp. 2039–2047, New York, NY, USA, 2015. Association for Computing Machinery. ISBN 9781450336642.

doi: 10.1145/2783258.2788629. URL https://doi. org/10.1145/2783258.2788629.

Salganik, M. J., Lundberg, I., Kindel, A. T., Ahearn, C. E.,
Al-Ghoneim, K., Almaatouq, A., Altschul, D. M., Brand, J. E., Carnegie, N. B., Compton, R. J., Datta, D., Davidson, T., Filippova, A., Gilroy, C., Goode, B. J., Jahani, E., Kashyap, R., Kirchner, A., McKay, S., Morgan, A. C., Pentland, A., Polimis, K., Raes, L., Rigobon, D. E., Roberts, C. V., Stanescu, D. M., Suhara, Y., Usmani, A., Wang, E. H., Adem, M., Alhajri, A., AlShebli, B., Amin, R., Amos, R. B., Argyle, L. P., Baer-Bositis, L., Buchi, ¨ M., Chung, B.-R., Eggert, W., Faletto, G., Fan, Z., Freese, J., Gadgil, T., Gagne, J., Gao, Y., Halpern-Manners, A., ´ Hashim, S. P., Hausen, S., He, G., Higuera, K., Hogan, B., Horwitz, I. M., Hummel, L. M., Jain, N., Jin, K., Jurgens, D., Kaminski, P., Karapetyan, A., Kim, E. H., Leizman, B., Liu, N., Moser, M., Mack, A. E., Mahajan, M., Man- ¨ dell, N., Marahrens, H., Mercado-Garcia, D., Mocz, V.,
Mueller-Gastell, K., Musse, A., Niu, Q., Nowak, W.,
Omidvar, H., Or, A., Ouyang, K., Pinto, K. M., Porter, E., Porter, K. E., Qian, C., Rauf, T., Sargsyan, A., Schaffner, T., Schnabel, L., Schonfeld, B., Sender, B., Tang, J. D.,
Tsurkov, E., van Loon, A., Varol, O., Wang, X., Wang, Z., Wang, J., Wang, F., Weissman, S., Whitaker, K., Wolters, M. K., Woon, W. L., Wu, J., Wu, C., Yang, K., Yin, J., Zhao, B., Zhu, C., Brooks-Gunn, J., Engelhardt, B. E., Hardt, M., Knox, D., Levy, K., Narayanan, A., Stewart, B. M., Watts, D. J., and McLanahan, S. Measuring the Predictability of Life Outcomes with a Scientific Mass Collaboration. Proceedings of the National Academy of Sciences, 117(15):8398–8403, 2020. doi: 10.1073/pnas. 1915006117. URL https://www.pnas.org/doi/ abs/10.1073/pnas.1915006117.

Schmucker, A. and vom Berge, P. Faktisch anonymisierte Version der Stichprobe der Integrierten Arbeitsmarktbiografien (SIAB-Regionalfile) - Version 7521 v1. Forschungsdatenzentrum der Bundesagentur fur Arbeit ¨ (BA) im Institut fur Arbeitsmarkt- und Berufsforschung ¨ (IAB), 2023a. 10.5164/IAB.SIAB-R7521.de.en.v1.

Schmucker, A. and vom Berge, P. Sample of Integrated Labour Market Biographies Regional File (SIAB-R) 1975
- 2021. FDZ-Datenreport 07/2023 (en), Research Data Centre of the Federal Employment Agency (BA) at the Institute for Employment Research (IAB), Nurnberg, 2023b. ¨ 10.5164/IAB.FDZD.2307.en.v1.

Shirali, A., Abebe, R., and Hardt, M. Allocation Requires Prediction Only if Inequality Is Low. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https://openreview.net/forum? id=WUicA0hOF9.

Sun, T. Q. and Medaglia, R. Mapping the challenges of Artificial Intelligence in the public sector: Evidence from public healthcare. *Government Information Quarterly*, 36(2):368–383, 2019.

Wang, A., Kapoor, S., Barocas, S., and Narayanan, A.

Against Predictive Optimization: On the Legitimacy of Decision-making Algorithms That Optimize Predictive Accuracy. *ACM J. Responsib. Comput.*, 1(1), March 2024.

doi: 10.1145/3636509. URL https://doi.org/10. 1145/3636509.

Wilder, B. and Welle, P. Learning treatment effects while treating those in need. *arXiv preprint arXiv:2407.07596*, 2024.

Wirtz, B. W., Weyerer, J. C., and Geyer, C. Artificial Intelligence and the Public Sector—Applications and Challenges. *International Journal of Public Administration*, 42(7):596–615, 2019.

## A. Theoretical Investigation

0.0 0.1 0.2 0.3 0.4 0.5 β = 0.2 Den sity 0 2 4 6 8 Y
Figure 6. Normal welfare distribution, with vertical lines marking the quantile cutoff (β = 0.2). The shaded region to the left of the vertical line represents the worst-off segment of the population.

0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00 0.5 1.0 1.5 2.0 0.5 1.0 1.5 2.0 Predicti ons: 
R

2 Predicti ons: 
R

2 V(α, β, R2) > 0.9 V(α, β, R2) > 0.9 β = 0.05 β = 0.05 0.0 0.1 0.2 0.3 0.4 Screening Capacities: α 0.0 0.1 0.2 0.3 0.4 Screening Capacity: α
(b) 1/4 × PAR
(a) PAR

## B. Experiments B.1. Experimental Setup And Labor Market Data

The dataset is provided via a Scientific Use File by the Research Data Centre (FDZ) of the German Federal Employment Agency (BA) at the Institute for Employment Research (IAB) (Schmucker & vom Berge, 2023a;b). It is a 2% weakly anonymized random sample of the complete German labor market records from 1975 to 2017 and contains information on 1,827,903 individuals across 62,340,521 observations (Schmucker & vom Berge, 2023b). We follow the same set of covariates and aggregation procedure for individual unemployment spells as described in Bach et al. (2023), incorporating demographic characteristics, labor market histories, and information about the most recent job. This results in 56 numerical variables and 24 categorical variables, which are one-hot encoded for model training. Figure 8 shows a histogram of individual unemployment durations, which we use as the basis for constructing the outcome variables. The distribution is right-skewed, with a concentration on short durations near zero and a long tail. Such a pattern is commonly observed in other welfare-related outcomes, such as health or income metrics. We define as prediction target the duration of the unemployment period in days Y , capped at 24 months3. Differentiating tail values is less important for 3In practice, for a fixed β, the problem can also be framed as a classification task (see Appendix B.5).

decision-making, and capping also allows training across years with varying observation windows.

100000 LTU
75000 Cou nt 50000 25000 0 0 500 1000 1500 Unemployment duration (days)
To avoid the impact of significant labor market reforms in Germany and to ensure full observation of unemployment durations up to 24 months, we restrict our analysis to unemployment episodes that began between 2010 and 2015. We use records from 2010 and 2011 to build the training dataset, records from 2012 for validation, and evaluate test performance on data from 2015 (see Figure 9). We left a gap between the training and test data periods to allow enough time for the outcomes in the training data to have been fully observed at test time, in order to mimic a realistic deployment scenario starting at the beginning of 2015.

Training Validation Test 2010 2011 2012 2013 2014 2015 2016 2017 2018

## B.2. Training Details

We use CatBoost (https://catboost.ai) for model training. The model was trained for a maximum 5,000 iterations with an early stopping criterion (early stopping rounds = 20) based on validation performance. Additionally, we train a shallow Decision Tree (max depth = 4) using the scikit-learn package. All hyperparameters are kept at their default settings unless otherwise specified.

## B.3. Prediction Improvements

To simulate an increase in predictive power by a specified amount ∆R2 , we adjust the model's predictions Yˆ using the residuals Y − Yˆ . Starting with the original predictions Yˆ and true outcomes Y , we define the adjusted predictions as

$$\hat{Y}_{+}=\hat{Y}+\delta(Y-\hat{Y})$$

We can then determine the δ corresponding to an increase of ∆R2 in the model's R2:

$$\delta=1-\sqrt{1-\Delta_{R^{2}}\frac{\sum_{i=1}^{n}(Y_{i}-\bar{Y})^{2}}{\sum_{i=1}^{n}(Y_{i}-\hat{Y}_{i})^{2}}}$$

For a specified δ, the new residuals are

$$Y-\hat{Y}_{+}=(1-\delta)(Y-\hat{Y})$$

Consequently, the variance decreases by a multiplicative factor: Var(Y − Yˆ+) = (1 − δ)
2 Var(Y − Yˆ ).

−20 −15 −10 −5 0 5 10 15 20 Residuals (Y − Yˆ )
0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14
−20 −15 −10 −5 0 5 10 15 20 Residuals (1 − δ)(Y − Yˆ )
0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 σ 2 = 34.81 σ 2 = 30.72 Density Density
(a) ∆R2 = 0
(b) ∆R2 = 0.1 500 1000 1500 2000 2500 5000 7500 10000 Training Set Size (# Samples)
0.000 0.025 0.050 0.075 0.100 0.125 0.150 R2
−20 −15 −10 −5 0 5 10 15 20 Residuals (Y − Yˆ )
0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14
−20 −15 −10 −5 0 5 10 15 20 Residuals (Y − Yˆ )
0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 σ 2 = 37.19 σ 2 = 35.34 Density Density
(b) 10, 000 Training Samples

## B.4. Additional Figures

0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 Screening Capacity: α 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 β values 0.05 (≈ 24 mos.) 0.15 (≈ 12 mos.) 0.25 (≈ 8 mos.) 
PAR
0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 Screening Capacity: α 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 β values 0.05 (≈ 24 mos.) 0.15 (≈ 12 mos.) 0.25 (≈ 8 mos.)
V

(

R

2 + ∆
R

2

)

−

V

(

R

2

)

VC
AT
(α, β
)

−

VT REE
(α, β
)

β values 0.05 (≈ 24 mos.) 0.15 (≈ 12 mos.) 0.25 (≈ 8 mos.)
0.0 0.1 0.2 0.3 0.4 0.5 Screening Capacity: α 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08

## B.5. Binary Classification

Instead of predicting the exact duration of unemployment, the problem can be reframed as a binary classification task. For a fixed β, we can define a binary outcome: Y = 1{Y ⩾ F
−1 Y,n(1 − β)}. This approach more directly encodes the target of interest: identifying individuals who may require further screening or assistance. If the chosen classifier provides estimates of class probabilities pˆ(x), it can be used to formulate a decision policy 1{pˆ(x) ⩾ F
−1 n,pˆ(1 − α)}. However, this forces us to specify β and the resulting decision threshold prior to model training. This requirement reduces flexibility compared to a continuous prediction setup, making classification more appropriate when the model is not intended for use in other tasks and when β remains constant across the deployment context. Additionally, directly converting durations to labels discards information on the precise unemployment durations that could be valuable for the modeling process.

As can be seen in Figure 16, the resulting policy values and true positive counts remain very similar compared to the regression case.

0.0 0.1 0.2 0.3 0.4 0.5 Screening Capacity: α 0.0 0.2 0.4 0.6 0.8 1.0 β values 0.05 (≈ 24 mos.) 0.15 (≈ 12 mos.) 0.25 (≈ 8 mos.)
β values 0.05 (≈ 24 mos.)
0.15 (≈ 12 mos.) 0.25 (≈ 8 mos.)
0.0 0.1 0.2 0.3 0.4 0.5 Screening Capacity: α 0 2500 5000 7500 10000 12500 15000 Po lic y Va lue V

(α, β
)

Tr ue Po sitive Co unt
(a) Policy Value
(b) True Positive Count

## C. Additional Propositions

Proposition C.1. (Optimal Policy with Gaussian Error) If ε = Y −Yˆ ∼ N (0, γ2)*, then the optimal policy* π
∗: R → {0, 1}
to solve the screening problem (Definition *2.1) is equal to:*

$$\pi^{*}(\hat{Y}_{i})=1\{\hat{Y}_{i}\leqslant F_{\hat{Y}}^{-1}(\alpha)\}$$
■

where F
−1 Yˆ(α) is the α-quantile of Yˆ *. The value of the policy is* V (π
∗) = Pr[Yˆ ⩽ F
−1 Yˆ(α) | Y ⩽ F
−1 Y(β)].

Proof. Since Y = Yˆ + ε where ε ∼ N (0, γ2), it follows for the conditional distribution Y | Yˆ ∼ N (*Y , γ* ˆ 2). Since Y | Yˆ
is Gaussian, we can express the conditional probability from Proposition 2.2 in terms of the CDF of the standard normal distribution,

$$\operatorname*{Pr}[Y\leqslant F_{Y}^{-1}(\beta)\mid{\hat{Y}}]=\Phi\left({\frac{F_{Y}^{-1}(\beta)-{\hat{Y}}}{\gamma}}\right)$$

To reproduce the ranking induced by Pr[Y ⩽ F
−1 Y(β) | Yˆ ], individuals can be ranked in ascending order of Yˆ . Thus, we can express the optimal policy (Proposition 2.2) in terms of a ranking of Yˆ ,

$$\pi^{*}(\hat{Y}_{i})=1\{\hat{Y}_{i}\leqslant F_{\hat{Y}}^{-1}(\alpha)\}$$

where F
−1 Yˆ(α) is the α-quantile of Yˆ . The value V (π
∗) that can by achieved by the optimal screening policy π
∗can then be expressed as:

$$V(\pi^{*})=\mathbb{E}\left[\pi^{*}(\hat{Y})=1\mid Y\leqslant F_{Y}^{-1}(\beta)\right]=\mathbb{E}\left[1\{\hat{Y}\leqslant F_{\hat{Y}}^{-1}(\alpha)\}\mid Y\leqslant F_{Y}^{-1}(\beta)\right]$$ $$=\Pr[\hat{Y}\leqslant F_{\hat{Y}}^{-1}(\alpha)\mid Y\leqslant F_{Y}^{-1}(\beta)]$$

## D. Proofs

D.1. Optimal Policy for Screening Problem: Proof of Proposition 2.2 Proof. We rewrite the policy value,

$$\mathbb{E}\left[\pi(\hat{Y}_{i})=1\mid Y\leqslant F_{Y}^{-1}(\beta)\right]=\frac{\mathbb{E}\left[\pi(\hat{Y}_{i})1\{Y\leqslant F_{Y}^{-1}(\beta)\}\right]}{\Pr[Y\leqslant F_{Y}^{-1}(\beta)]}$$ $$=\frac{1}{\beta}\,\mathbb{E}\left[\pi(\hat{Y}_{i})\,\mathbb{E}\left[1\{Y\leqslant F_{Y}^{-1}(\beta)\}\mid\hat{Y}_{i}\right]\right]$$ $$=\frac{1}{\beta}\,\mathbb{E}\left[\pi(\hat{Y}_{i})\Pr[Y\leqslant F_{Y}^{-1}(\beta)\mid\hat{Y}_{i}]\right]$$

To maximize the objective, individuals Yˆi with the largest scores s(Yˆi) = Pr[Y ⩽ F
−1 Y(β) | Yˆi] should be prioritized. Thus, the optimal policy is to intervene (π(Yˆi) = 1) for the top α-fraction of the population ranked by Pr[Y ⩽ F
−1 Y(β) | Yˆ ]. ■
D.2. Optimal Policy Value in Gaussian Setting: Proof of Proposition 2.3 Following Proposition C.1, the value of the optimal screening policy π
∗can then be expressed as:
V (π
∗) = Pr[Yˆ ⩽ F
−1 Yˆ
(α) | Y ⩽ F
−1 Y(β)]
We can rewrite the conditional probability in terms of the joint distribution of Y and Yˆ , and note that PrY ⩽ F
−1 Y(β)	=
FY (F
−1 Y(β*)) =* β,

$$\mathrm{Pr}[\hat{Y}\leqslant F_{\hat{Y}}^{-1}(\alpha)\mid Y\leqslant F_{Y}^{-1}(\beta)]=\frac{1}{\beta}\mathrm{Pr}[\hat{Y}\leqslant F_{\hat{Y}}^{-1}(\alpha),Y\leqslant F_{Y}^{-1}(\beta)]\,$$

We then standardize Y ∼ N (*µ, η*2) and Yˆ ∼ N (*µ, η*2 − γ 2) and make use that for a normal random variable with mean µ and variance σ 2the quantile function is F
−1(p) = µ + σΦ
−1(p).

$$\frac{1}{\beta}\Pr[\tilde{Y}\leqslant F_{\tilde{Y}}^{-1}(\alpha),Y\leqslant F_{Y}^{-1}(\beta)]=\frac{\Pr\bigl{\{}Z_{1}\leqslant\frac{F_{\tilde{Y}}^{-1}(\alpha)-\mu}{\sqrt{\eta^{2}-\gamma^{2}}},Z_{2}\leqslant\frac{F_{Y}^{-1}(\beta)-\mu}{\eta}\bigr{\}}}{\beta}$$ $$=\frac{\Pr\bigl{\{}Z_{1}\leqslant\Phi^{-1}\left(\alpha\right),Z_{2}\leqslant\Phi^{-1}\left(\beta\right)\bigr{\}}}{\beta}$$

Z1 and Z2 are standard Gaussian with Cov(Z1, Z2) = E [Z1Z2] = 1 η
√η2−γ2 Cov(Y , ˆ Yˆ +ε) = Cov(Y , ˆ Yˆ )
η
√η2−γ2 =
√η2−γ2 η. Thus, they are distributed according to a standard bivariate normal distribution with correlation ρ = Cov(Z1, Z2) =
√η2−γ2 η.

Thus,

$$V(\pi^{*})=\mathbb{E}\left[\pi^{*}(\hat{Y})=1\mid Y\leqslant F_{Y}^{-1}(\beta)\right]=\frac{1}{\beta}\Phi_{2}\left(\Phi^{-1}\left(\alpha\right),\Phi^{-1}\left(\beta\right);\rho\right)$$

where

$$\Phi_{2}\left(\Phi^{-1}\left(\alpha\right),\Phi^{-1}\left(\beta\right)\right)=\int_{-\infty}^{\Phi^{-1}\left(\alpha\right)}\int_{-\infty}^{\Phi^{-1}\left(\beta\right)}\phi_{2}\left(z_{1},z_{2};\rho\right)\,\mathrm{d}z_{1}\,\mathrm{d}z_{2}\ .$$

and

$$\phi_{2}\left(z_{1},z_{2}\right)=\frac{1}{2\pi\sqrt{1-\rho^{2}}}e^{-1/2(z_{1}^{2}-2\rho z_{1}z_{2}+z_{2}^{2})/(1-\rho^{2})}$$

D.3. Prediction-Access Ratio for Small Screening Capacities: Proof of Theorem 3.1 Using Taylor's theorem,

$$V(\alpha,\beta,R^{2}+\Delta_{R^{2}})-V(\alpha,\beta,R^{2})=\Delta_{R^{2}}\frac{\partial}{\partial R^{2}}V(\alpha,\beta,R^{2}+p_{R^{2}}\Delta_{R^{2}})$$

where pR2 ∈ (0, 1). We know from Lemma D.3,

$$\frac{\partial}{\partial R^{2}}V(\alpha,\beta,R_{*}^{2})\leqslant\frac{1}{\beta\sqrt{8\pi R_{*}^{2}(1-R_{*}^{2})}}\phi\left(\frac{\Phi^{-1}\left(\alpha\right)-\sqrt{R_{*}^{2}}\Phi^{-1}\left(\beta\right)}{\sqrt{1-R_{*}^{2}}}\right).$$

where R2∗:= R2 + pR2∆R2 . For α < 0.5 and β ⩽ 0.5, we know Φ
−1(α) < 0 and Φ
−1(β) ⩽ 0. It follows, that for any ε1 > 0, 0 < R2∗and 0 < β, there exists a threshold value t1 > 0, such that for all α ⩽ t1, we have

$$\left(1+\varepsilon_{1}\right)\frac{\Phi^{-1}\left(\alpha\right)}{\sqrt{1-R_{*}^{2}}}\leqslant\frac{\Phi^{-1}\left(\alpha\right)-\sqrt{R_{*}^{2}}\Phi^{-1}\left(\beta\right)}{\sqrt{1-R_{*}^{2}}}\leqslant\left(1-\varepsilon_{1}\right)\frac{\Phi^{-1}\left(\alpha\right)}{\sqrt{1-R_{*}^{2}}}$$

If *α < β* we find Φ
−1(α) −pR2∗Φ
−1(β) < 0. Since ϕ(x) ⩽ ϕ(x
′) for x ⩽ x
′ < 0,

$$\frac{1}{\beta\sqrt{8\pi R_{*}^{2}(1-R_{*}^{2})}}\phi\left(\frac{\Phi^{-1}\left(\alpha\right)-\sqrt{R_{*}^{2}}\Phi^{-1}\left(\beta\right)}{\sqrt{1-R_{*}^{2}}}\right)\leqslant\frac{1}{\beta\sqrt{8\pi R_{*}^{2}(1-R_{*}^{2})}}\phi\left(\left(1-\varepsilon_{1}\right)\frac{\Phi^{-1}\left(\alpha\right)}{\sqrt{1-R_{*}^{2}}}\right)$$ $$=\frac{1}{\beta\sqrt{8\pi R_{*}^{2}(1-R_{*}^{2})}}\phi\left(\kappa\Phi^{-1}\left(\alpha\right)\right)$$ $$=\frac{1}{\beta\sqrt{8\pi R_{*}^{2}(1-R_{*}^{2})}}\phi\left(\kappa\Phi^{-1}\left(1-\alpha\right)\right)$$

18 where κ := √
(1−ε1)
1−R2∗
. For any ε2 > 0, there exists a threshold t2 > 0, such that for all α ⩽ t2, we can apply *Lemma B.5.*
from Perdomo (2024) to arrive at the following inequality:

$$\phi\left(\kappa\Phi^{-1}\left(1-\alpha\right)\right)\leqslant\frac{1}{\sqrt{2\pi}}\left((1+\varepsilon_{2})\sqrt{2\pi}\alpha\Phi^{-1}\left(1-\alpha\right)\right)^{\kappa^{2}}$$

Thus,

$$V(\alpha,\beta,R^{2}+\Delta_{R^{2}})-V(\alpha,\beta,R^{2})\leqslant\Delta_{R^{2}}{\frac{1}{\beta4\pi\sqrt{R_{*}^{2}(1-R_{*}^{2})}}}\left((1+\varepsilon_{2})\sqrt{2\pi}\alpha\Phi^{-1}\left(1-\alpha\right)\right)^{\varepsilon^{2}}.$$

We can use Taylor's theorem again and from Lemma D.1 we know that

$$V(\alpha+\Delta_{\alpha},\beta,R^{2})-V(\alpha,\beta,R^{2})=\Delta_{\alpha}\frac{\partial}{\partial\alpha}V(\alpha+p_{\alpha}\Delta_{\alpha},\beta,R^{2})$$ $$=\Delta_{\alpha}\frac{1}{\beta}\Phi\left(\frac{\Phi^{-1}\left(\beta\right)-\sqrt{R^{2}}\Phi^{-1}\left(\alpha+p_{\alpha}\Delta_{\alpha}\right)}{\sqrt{1-R^{2}}}\right)$$

where pα ∈ (0, 1). Since 0 < β and 0 < R2there will always be a small enough α + ∆α such that

$$\Phi^{-1}\left(\beta\right)-\sqrt{R^{2}}\Phi^{-1}\left(\alpha+p_{\alpha}\Delta_{\alpha}\right)\geqslant0$$

Since Φ (x) ⩾ 1/2 for x ⩾ 0, it follows

$$\frac{\Delta_{\alpha}}{2\beta}\leqslant V(\alpha+\Delta_{\alpha},\beta,R^{2})-V(\alpha,\beta,R^{2})$$

It follows for the prediction-access ratio,

$$\frac{\Delta_{\alpha}}{\Delta_{R^{2}}}2\pi\sqrt{R_{*}^{2}(1-R_{*}^{2})}\left((1+\varepsilon_{2})\sqrt{2\pi}\alpha\Phi^{-1}\left(1-\alpha\right)\right)^{-(1-\varepsilon_{1})^{2}\frac{1}{1-R_{*}^{2}}}\leqslant\frac{V(\alpha+\Delta_{\alpha},\beta,R^{2})-V(\alpha,\beta,R^{2})}{V(\alpha,\beta,R^{2}+\Delta_{R^{2}})-V(\alpha,\beta,R^{2})}.$$

For small α, Φ
−1(1 − α) grows asymptotically like plog (1/α). Consequently, the polynomial growth of α
−1/(1−R
2)
drives the PAR to increase rapidly as α decreases. Since 1 1−R2∗
increases with R2∗and R2 ⩽ R2∗, we can lower bound the PAR by inserting R2instead of R2∗:

$$\frac{\Delta_{\alpha}}{\Delta_{R^{2}}}2\pi\sqrt{R^{2}(1-R^{2})}\left((1+\varepsilon_{2})\sqrt{2\pi}\alpha\Phi^{-1}\left(1-\alpha\right)\right)^{-(1-\varepsilon_{1})^{2}\frac{1}{1-R^{2}}}\leqslant\frac{V(\alpha+\Delta_{\alpha},\beta,R^{2})-V(\alpha,\beta,R^{2})}{V(\alpha,\beta,R^{2}+\Delta_{R^{2}})-V(\alpha,\beta,R^{2})}\,.$$

We can simplify the lower-bound by noting that 0 < ε1 and 0 < ε2 can be made arbitrarily small by selecting a sufficiently small threshold t for α + ∆α. Specifically, ε2 < 1 holds for α ⩽ 0.05 (see *Lemma A.6* in Perdomo (2024)).

$$\frac{\Delta_{\alpha}}{\Delta_{R^{2}}}\sqrt{R^{2}(1-R^{2})}\left(5.1\cdot\alpha\Phi^{-1}\left(1-\alpha\right)\right)^{-\frac{1}{1-R^{2}}+o(1)}\leqslant\frac{V(\alpha+\Delta_{\alpha},\beta,R^{2})-V(\alpha,\beta,R^{2})}{V(\alpha,\beta,R^{2}+\Delta_{R^{2}})-V(\alpha,\beta,R^{2})}.$$

D.4. Maximally Effective (Local) Prediction Improvements: Proof of Theorem 3.2 We know from Lemma D.2,

$$\begin{split}\lim_{\Delta\to0}\frac{V(\alpha,\beta,R^{2}+\Delta)-V(\alpha,\beta,R^{2})}{\Delta}&=\frac{\partial}{\partial R^{2}}V(\alpha,\beta,R^{2})\\ &=\frac{1}{2\beta\sqrt{R^{2}}}\phi_{2}\left(\Phi^{-1}\left(\alpha\right),\Phi^{-1}\left(\beta\right);\rho\right)\end{split}$$
We insert ϕ2 (·) and arrive at
(-) and arrive at $$\begin{aligned} \frac{\partial}{\partial R^2}V(\alpha,\beta,R^2)&=\underbrace{\frac{1}{4\pi\beta\sqrt{R^2(1-R^2)}}}_{\hat{\gamma}_1}\\ &\times\underbrace{\exp\left(-\frac{1}{2(1-R^2)}\big(\Phi^{-1}\left(\alpha\right)^2+\Phi^{-1}\left(\beta\right)^2-2\sqrt{R^2}\Phi^{-1}\left(\alpha\right)\Phi^{-1}\left(\beta\right)\big)\right)}_{\hat{\gamma}_2} \end{aligned}$$ $T$ diverges as $R^2\to1$ or $R^2\to0$. 
The prefactor T1 diverges as R2 → 1 or R2 → 0. If R2 → 1, the exponential term will generally suppress the polynomial growth of the prefactor. However for α = β, we find for the exponent

$$-\frac{1}{2{\left(1-R^{2}\right)}}{\left(\alpha\right)}^{2}+\Phi^{-1}\left(\beta\right)^{2}-2\sqrt{R^{2}}\Phi^{-1}\left(\alpha\right)\Phi^{-1}\left(\beta\right)=-\frac{1-\sqrt{R^{2}}}{1-R^{2}}\Phi^{-1}\left(\beta\right)^{2}$$ $$=-\frac{1}{\left(1+\sqrt{R^{2}}\right)}\Phi^{-1}\left(\alpha\right)^{2}$$ $$R_{\frac{\alpha}{2}\to1}^{2\to1}-\frac{1}{2}\Phi^{-1}\left(\beta\right)^{2}$$

which is finite. Therefore, ∂
∂R2 V (*α, β, R*2) becomes unboundedly large if α = β and R2 → 1.

If R2 → 0, the prefector T1 diverges again to +∞. The exponent then simplifies to

$$-\frac{1}{2(1-R^{2})}(\Phi^{-1}\left(\alpha\right)^{2}+\Phi^{-1}\left(\beta\right)^{2}-2\sqrt{R^{2}}\Phi^{-1}\left(\alpha\right)\Phi^{-1}\left(\beta\right))=-\frac{1}{2}(\Phi^{-1}\left(\alpha\right)^{2}+\Phi^{-1}\left(\beta\right)^{2})$$

If α and β are not set arbitrarily small or large ∂
∂R2 V (*α, β, R*2) will diverge. The local PAR (Lemma D.1)

$$\begin{array}{c}{{\operatorname*{lim}_{\Delta\to0}\frac{V(\alpha+\Delta,\beta,R^{2})-V(\alpha,\beta,R^{2})}{V(\alpha,\beta,R^{2}+\Delta)-V(\alpha,\beta,R^{2})}=\frac{\frac{\partial}{\partial\alpha}V(\alpha,\beta,R^{2})}{\frac{\partial}{\partial R^{2}}V(\alpha,\beta,R^{2})}}}\\ {{=\frac{\Phi\left(\frac{\Phi^{-1}(\beta)-\sqrt{R^{2}}\Phi^{-1}(\alpha)}{\sqrt{1-R^{2}}}\right)}{\frac{1}{2\sqrt{R^{2}}}\phi_{2}\left(\Phi^{-1}\left(\alpha\right),\Phi^{-1}\left(\beta\right);\rho\right)}}}\end{array}$$

approaches zero in both regimes. D.5. Prediction-Access Ratio for Local Improvements: Proof of Proposition 3.3 We know

$$\operatorname*{lim}_{\Delta\rightarrow0}{\frac{V(\alpha+\Delta,\beta,R^{2})-V(\alpha,\beta,R^{2})}{V(\alpha,\beta,R^{2}+\Delta)-V(\alpha,\beta,R^{2})}}={\frac{{\frac{\partial}{\partial\alpha}}V(\alpha,\beta,R^{2})}{{\frac{\partial}{\partial R^{2}}}V(\alpha,\beta,R^{2})}}$$

Using Lemma D.1 and Lemma D.3 we find a lower bound for the PAR:

$${\frac{\sqrt{8\pi R^{2}(1-R^{2})}}{\hat{T}_{1}}}\underbrace{{\frac{\Phi\left({\frac{\Phi^{-1}(\beta)-\sqrt{R^{2}}\Phi^{-1}(\alpha)}{\sqrt{1-R^{2}}}}\right)}{\Phi\left({\frac{\Phi^{-1}(\beta)-\sqrt{R^{2}}\Phi^{-1}(\alpha)}{\sqrt{1-R^{2}}}}\right)}}_{\hat{T}_{2}}}\leqslant{\frac{V(\alpha+\Delta_{\alpha},\beta,R^{2})-V(\alpha,\beta,R^{2})}{V(\alpha,\beta,R^{2}+\Delta_{R^{2}})-V(\alpha,\beta,R^{2})}}$$

We then denote z :=
Φ
−1(β)−
√R2Φ
−1(α)
√1−R2and T2 =
Φ(z) ϕ(z)
. We know from Lemma D.4 that Φ(z)
ϕ(z)
increases with z. It follows that we need to find the smallest possible z to find a lower bound for T2. Generally, z decreases with α and increases with β.

We treat both cases separately: