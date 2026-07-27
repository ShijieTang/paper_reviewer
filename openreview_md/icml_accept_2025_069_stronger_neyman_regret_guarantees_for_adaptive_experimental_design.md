# 

Georgy Noarov 1 Riccardo Fogliato 2 Martin Bertran 2 **Aaron Roth** 1 2

## Abstract Have Low Variance.

We study the design of adaptive, sequential experiments for unbiased average treatment effect (ATE)
estimation in the design-based potential outcomes setting. Our goal is to develop adaptive designs offering *sublinear Neyman regret*, meaning their efficiency must approach that of the hindsightoptimal nonadaptive design. Recent work (Dai et al., 2023) introduced ClipOGD, the first method achieving Oe(
√T) expected Neyman regret under mild conditions. In this work, we propose adaptive designs with substantially stronger Neyman regret guarantees. In particular, we modify ClipOGD to obtain anytime Oe(log T) Neyman regret under natural boundedness assumptions. Further, in the setting where experimental units have pretreatment covariates, we introduce and study a class of contextual "multigroup" Neyman regret guarantees: Given any set of possibly overlapping groups based on the covariates, the adaptive design outperforms each group's best non-adaptive designs. In particular, we develop a contextual adaptive design with Oe(
√T) anytime multigroup Neyman regret. We empirically validate the proposed designs through an array of experiments.

## 1. Introduction

Randomized control trials (RCTs) play a central role in a variety of settings where causal effects need to be accurately measured, spanning healthcare and epidemiology, policymaking, the social sciences, econometrics, e-commerce, and beyond. In the classic potential outcomes framework (Neyman, 1923; Rubin, 1974), a central estimand is the average treatment effect (ATE) - the average individual causal effect across experimental units. To obtain precise estimates of the ATE, we generally seek estimators that are unbiased and 1Department of Computer and Information Science, University of Pennsylvania. 2Amazon Web Services. Correspondence to:
Georgy Noarov <gnoarov@seas.upenn.edu>.

1 In many cases, RCTs are run sequentially: Experimental units arrive one by one, and each unit is assigned to treatment or control adaptively, based on previous outcomes or auxiliary information. The data-driven nature and flexibility of these experiments suggest that such adaptive trials can achieve substantial efficiency gains over standard fixed designs, as shown in domains ranging from political science (Offer-Westort et al., 2021; Blackwell et al., 2022) to medicine (Chow & Chang, 2008; Villar et al., 2015; FDA, 2019). However, so far adaptive experiments have received limited attention (Hu & Rosenberger, 2006) and have been rarely used in practice due to concerns that adaptivity could invalidate standard statistical guarantees (van der Laan, 2008). Indeed, classic solutions for improving estimator efficiency in the batch setting, such as Neyman allocation (Neyman, 1992), can be nontrivial to extend to the sequential setting. Recently, a growing body of work (Hahn et al., 2011; Kato et al., 2020; Li & Owen, 2024; Dai et al., 2023; Cook et al., 2023) has made progress on this front by introducing multistage adaptive designs that estimate the ATE via inverseprobability weighting (IPW)-type estimators with adaptively adjusted propensity scores. 1 Our work contributes to this literature by developing novel adaptive sequential designs for IPW-based ATE estimation with efficiency guarantees. Crucially, our methods –unlike most existing work– are developed within the finite-population setting (Wager, 2024), where the ATE is defined as a deterministic function of the observed population rather than a superpopulation parameter. This distinction ensures robustness to treatment effect heterogeneity and temporal data drift, challenges that can undermine conventional superpopulation-based designs. Our Contributions We focus on the design of adaptive RCTs to estimate the ATE as efficiently as the best-inhindsight IPW design from some benchmark class, up to error terms. Specifically, we aim to minimize the Neyman regret (Kato et al., 2020; Dai et al., 2023) - a measure comparing the variance of our adaptive estimator to that of the variance-minimizing nonadaptive Bernoulli trial where units are treated with some fixed probability. Currently, to our knowledge Dai et al. (2023)'s ClipOGD method is the only adaptive design achieving sublinear Neyman regret in the finite-population setting. This method guarantees Oe(
√T) expected regret for any T-unit trial under momentbounded potential outcomes. However, two important questions arise:
I. Can we develop designs with better regret rates? Dai et al. (2023) conjectured that Oe(
√T) is the minimax Neyman rate.

II. Can we develop context-aware designs that use pretreatment covariates to improve efficiency?

In this work, we answer both these questions affirmatively: Contribution I: Exponentially Improved Noncontextual Neyman Regret Bound. We show that, under a natural strengthening of Dai et al. (2023)'s assumptions on the outcomes, we can modify ClipOGD to attain an anytimevalid Neyman regret bound of Oe(log T).

2 To achieve this speedup, we leverage the strong convexity of the Neyman objective under our stricter lower-bounding assumption on the outcomes, which as we show leads to near-logarithmic regret via techniques introduced by (Hazan et al., 2007). Moreover, it can be shown that even under the weaker outcome lower bound assumption of Dai et al. (2023), our adaptive design can be tweaked to have the asymptotic efficiency of (1 + ϵ) V
∗ + Oelog T
T
for any ϵ > 0, where V
∗ denotes the optimal nonadaptive design variance; the interpretation is that any (1 + ϵ)-multiplicative approximation to the optimal variance can be attained at this fast rate. We validate the greater efficiency of our proposed design against that of ClipOGD through a suite of experiments on synthetic and real-world data. Contribution II: Adaptive Designs with Contextual Neyman Regret Guarantees. We next develop a novel adaptive design MGATE (Multi-Group ATE) that leverages pretreatment covariates to improve efficiency relative to the non-contextual setting. In a nutshell, given an arbitrary predefined finite collection G ⊆ 2 X of contextual groups defined by the covariates (e.g., demographics), we propose a no G-multigroup-Neyman-regret adaptive design that obtains sublinear regret simultaneously on all subsequences of 2In fact, a lower bounding construction in the very recent work of Li et al. (2024) shows that the best possible Neyman regret is Ω(1) even in the more relaxed superpopulation setting - and so our method achieves a *best-of-both-worlds* guarantee, up to logarithmic factors.

experimental units corresponding to the groups in G. Critically, we also allow for overlapping groups, i.e., units can simultaneously belong to multiple groups. A key challenge here is to balance the treatment probabilities in a way that balances the efficiency of the ATEs estimates across groups. Our proposed design leverages a variation of the "sleeping experts" approach (Blum & Lykouris, 2020; Acharya et al., 2024) used in the online learning literature (Lee et al., 2022; Deng et al., 2024), that deals with the limited feedback and the fact that the observed objective values do not live in an a-priori bounded range. The method achieves Oe(
√T)
multigroup Neyman regret. We also empirically validate its performance. Our multigroup guarantees can be interpreted through the lens of group ATE (GATE) estimation (Chernozhukov et al., 2017; Semenova & Chernozhukov, 2021; Zimmert & Lechner, 2019). GATE occupies a middle ground between ATE, which measures the average effect over the entire sequence, and CATE (conditional ATE), which measures the ATE conditionally on each covariate vector. Existing works on GATE, however, are mainly focused on learning data-driven disjoint groups to improve overall ATE estimation. In contrast, our objective is to simultaneously ensure efficient GATE inference for any family of arbitrarily overlapping groups. This is related in motivation (though distinct in technique) to the recent work of (Kern et al., 2024) who use
"multiaccuracy" to make CATE inference robust to certain kinds of distribution shift. We expect that such multigroup efficiency guarantees can be broadly useful, and hope future work will study multigroup adaptive designs beyond the sequential finite-population setting that we focus on in this paper. For an additional discussion of related work, including relevant independent work in the superpopulation setting, please see Appendix A. Organization In Section 2, we introduce our general setting and objectives. In Section 3, we focus on the (vanilla) non-contextual setting, and present and analyze our adaptive design ClipOGDSC , which achieves near-logarithmic Neyman regret. We prove the main regret bound in Theorem 3.2 and demonstrate further guarantees on the adaptive design. In Section 4, we introduce the notion of multigroup Neyman regret, and present our multigroup adaptive design MGATE (Algorithm 2), which achieves Oe(
√T) multigroup Neyman regret as shown in Theorem 4.2. Furthermore, in Appendix D we provide a general multigroup design (Algorithm 7) that significantly generalizes MGATE. In Section 5, we compare the empirical performance of our adaptive designs to the Dai et al. (2023) ClipOGD design on an array of real-world and synthetic sequential experimental design tasks.

## 2. Preliminaries

Setting We work in the design-based, sequential variant of the potential outcomes setting (Neyman, 1923; Rubin, 1974; Imbens & Rubin, 2015). A finite number of experimental units in the population arrive one by one at rounds t ∈ N+.

Each unit has two associated fixed potential outcomes, only one of which can be observed: treatment outcome yt(1) ∈ R and control outcome yt(0) ∈ R.

In the basic setting, the observed outcome is the only information the experimenter receives about the units. A richer setting is one where before choosing treatment or control for unit t, the Experimenter is given access to pre-treatment covariate xt ∈ X , where X is a feature space of arbitrary nature (e.g. X may be a finite-dimensional vector space). In this paper, we will study both settings: the noncontextual setting in Section 3 and the contextual one in Section 4. Adaptive Design In a randomized controlled trial (RCT), the experimenter (randomly) decides whether to apply treatment or control to each unit, and observes the corresponding outcome but not the counterfactual. These randomized decisions for all units constitute the experimental design. We study adaptive experimental designs, described as follows.

T**-round Adaptive Design Protocol**
Potential outcomes {(yt(1), yt(0))}t∈[T] are generated upfront (but not shown to Experimenter). Then, sequentially for each unit t = 1 *. . . T*:
1. (*Contextual* setting only) Experimenter observes pre-treatment covariate xt ∈ X .

2. Experimenter sets treatment probability pt.
4. Experimenter observes outcome Yt = yt(Zt).

By contrast, the standard nonadaptive (Bernoulli) trial fixes upfront the same treatment probability pt = p for all units t, and uses it throughout the experiment without any adjustments. Our estimand of interest is the average treatment effect (ATE), which corresponds to the difference between the average outcomes of treatment and control units in the population. We provide the formal definition below. Definition 2.1 (ATE). The *average treatment effect* for potential outcomes {(yt(1), yt(0))}
T
t=1 is:

$$\tau_{T}=\frac{1}{T}\sum_{t=1}^{T}y_{t}(1)-y_{t}(0).$$

A classical estimator of the ATE is the adaptive IPW estimator (Horvitz & Thompson, 1952), which employs inverse probability weighting. We define it next. Definition 2.2 (Adaptive IPW Estimator). The *adaptive* IPW estimator of the ATE τT is:

$$\hat{\tau}_{T}=\frac{1}{T}\sum_{t}Y_{t}\left(\frac{Z_{t}}{p_{t}}-\frac{1-Z_{t}}{1-p_{t}}\right).$$

This estimator is unbiased, meaning that for any outcomes
{(yt(0), yt(1)}
T
t=1 and any adaptive design (pt)
T
t=1 with all pt ∈ (0, 1), we have E[ˆτT ] = τT . Thus, no matter what adaptive design Experimenter employs, the induced adaptive IPW estimator will always be unbiased. However, the estimator's variance will vary based on the design, making some designs more efficient than others.

Objective: Minimize Variance of ATE Estimator Our main goal will be to construct adaptive designs that asymptotically approach the variance of the best-in-hindsight experimental design in some benchmark class. A basic class of designs is that of nonadaptive designs, parameterized by the choice of fixed propensity p ∈ (0, 1). Formally, we measure the *Neyman regret* (Kato et al., 2020; Dai et al., 2023) of any proposed adaptive design as the (time-rescaled) difference between its IPW estimator variance and the variance of same estimator under the most efficient nonadaptive design. To define Neyman regret, note (see Proposition 2.2 of Dai et al. (2023)) that Var[ˆτT ] = PT
t=1 E [ft(pt)] /T2 − kATE,
where ft(p) := yt(1)2/p + yt(0)2/(1 − p) is the variance of the propensity- P
p IPW estimator at unit t, and kATE =
T
t=1(yt(1) − yt(0))2/T2is a design-independent term.

We are now ready to provide the formal definition. Definition 2.3 (Neyman Regret (Kato et al., 2020; Dai et al.,
2023)). The Neyman regret of adaptive design (pt)
T
t=1 on a potential outcomes sequence {(yt(1), yt(0))}
T
t=1 is:3

$$\mathrm{RegVar}_{T}=\operatorname*{max}_{p_{T}^{*}\in(0,1)}\sum_{t=1}^{T}f_{t}(p_{t})-f_{t}(p_{T}^{*}).$$

Thus the variance of the IPW estimator for a design (pt)
T t=1 differs from that of the best nonadaptive design by exactly RegVarT /T2, justifying the Neyman regret definition.

Our goal will be to develop adaptive designs with sublinear expected Neyman regret: E [RegVarT
] = o(T), or equivalently with vanishing average expected Neyman regret:
E
 [RegVarT /T] = o(1). We call any design that satisfies this a no-regret design.

3"Var" stands for variance, as Neyman regret captures the rescaled estimator variance associated with the design.

3. Experimenter flips bias-pt coin to obtain realized treatment decision: Zt ∼ Bernoulli(pt).

## 3. Efficient Non-Contextual Ate Estimation

We now present our first contribution: An adaptive design that achieves Oe(log T) Neyman regret under natural assumptions on the outcomes. We begin by discussing the Oe(
√T)-Neyman regret design ClipOGD of Dai et al.

(2023), and then modifying it to better exploit the strongly convex structure of the Neyman objective. Next, we discuss further guarantees on our method's performance.

## 3.1. Adaptive Design With Logarithmic Neyman Regret

Meta-Design: ClipOGD The first finite-population design that achieves sublinear Neyman regret, ClipOGD, was introduced by Dai et al. (2023). Leveraging the fact that the per-round Neyman objectives ft(p) are convex in p, it performs a modified version of online gradient descent (OGD)
on ft to adaptively modify the treatment probabilities pt. The complicating factor is that the gradients of ft diverge when p is close to 0 or 1: standard OGD analyses typically require explicit or implicit bounds on the gradients of the objective (Hazan et al., 2016), so vanilla projected OGD on the entire interval [0, 1] will not work without modification. ClipOGD solves this problem by clipping the OGD iterates
{pt}t∈N+ to be within a nested family {[δt, 1 − δt]}t∈N+ of subintervals of (0, 1), which gradually expand to cover the whole interval in the infinite time limit (i.e., limt→∞ δt = 0). The expansion is needed to handle cases when p
∗
Tis close to the boundary. In view of this, we let δt = 1/h(t) for all t ∈ N+, where h : N+ → R>0 is some strictly increasing function with limt→∞ h(t) = ∞. We call δt the clipping rate, h the clipping function, and refer to any adaptive design (pt)t∈N+ that satisfies 1/h(t) ≤ pt ≤ 1 − 1/h(t) for all t as h-clipped. Algorithm 1 gives the pseudocode for ClipOGD. Here, ΠS(x) denotes the projection of x onto interval S ⊂ (0, 1). Algorithm 1 ClipOGD (Dai et al., 2023)
Initialize p0 ← 0.5 and g0 ← 0 for units t = 1, 2*, . . .* do Set step size ηt > 0 and clipping rate δt ∈ (0, 0.5) Set treatment probability pt ← Π
[δt,1−δt]
(pt−1 −ηt ·gt−1)
Set treatment decision Zt ∼ Bernoulli(pt) Observe outcome Yt ← yt(Zt) Set gradient estimate: gt ← Y
2 t
−
Zt p 3t
+1−Zt
(1−pt)
3 end for ClipOGD0: A Oe(
√T) **Regret Design** In their paper, Dai et al. (2023) analyzed and provided guarantees for a specific instantiation of ClipOGD, where ηt =p1/T and δt = 0.5·
t
−1/α where α =
√5 log T for all t = 1*, . . . , T*. For clarity, we call this design ClipOGD0. Their main result proves that ClipOGD0 has Oe(
√T) Neyman regret under a moment assumption on the outcomes: 0 < c ≤ (
1 T
PT
t=1 yi(t)
2)
1/2 and (
1 T
PT
t=1 yi(t)
4)
1/4 ≤ C for i ∈ {0, 1} and some c ≤ C. However, the learning rate of ClipOGD0 has several drawbacks. First, it is too conservative, precluding improvement in Neyman regret beyond Oe(
√T). Second, it is horizon-dependent, making it necessary to know (or commit to) T upfront. Finally, it is constant rather than decreasing, so the design probabilities will jump around (rather than gradually converge) during any given run of ClipOGD0.

ClipOGDSC **: Our** Oe(log T) **Regret Design** We now present an adaptive design called ClipOGDSC that addresses these issues: It uses the learning rate ηt ∼ 1/t that, under Assumption 3.1, (1) achieves an exponentially improved Neyman regret bound, (2) is *anytime*, i.e., does not require advance knowledge of the time horizon T, and (3) its propensities converge in L2 to the hindsight-best propensity. Our Neyman regret bound relies on a stricter assumption than the one made by Dai et al. (2023)'s, which we detail below. Assumption 3.1 (Bounds on Potential Outcomes). There exist positive constants *c, C* such that outcomes
{(yt(0), yt(1))}t≥1 satisfy for all time horizons T:

$$\operatorname*{max}_{t\geq1}\{|y_{t}(0)|,|y_{t}(1)|\}\leq C,$$
$$c\leq\operatorname*{min}_{t>1}\,\left(y_{t}(0)^{2}+y_{t}(1)^{2}\right)^{1/2}$$
$$c\leq\operatorname*{min}_{i\in\{0,1\}}\left({\frac{1}{T}}\sum_{t=1}^{T}y_{t}(i)^{2}\right)^{1/2}.$$

Next, let hinv be the inverse function of h, defined via the identity hinv ◦ h = h ◦ hinv = Id. Our main result is the following Neyman regret bound in terms of T, h, and hinv.

Theorem 3.2 (Stronger Neyman Regret Bound). Suppose Assumption 3.1 is satisfied with C, c the corresponding constants. Let h : N+ → R>0 be strictly increasing. Let ClipOGDSC be the adaptive design that instantiates Algorithm 1 *with learning rate* ηt = 1/(2c 2t) *and clipping* rate δt = 1/h(t). Then, ClipOGDSC attains the following anytime-valid Neyman regret bound:

$$\mathbb{E}[\text{RegVar}_{T}]=O\Big{(}\big{(}h(T)\big{)}^{5}\cdot\log(T)+\big{(}h_{\text{inv}}\left(1+C/c\right)\big{)}^{2}\Big{)}\,.\tag{1}$$

Since h *can be chosen to grow arbitrarily slowly, we can* get: E[RegVarT] = Oe(log T).

The proof is contained in Appendix B. It exploits the strong convexity of the Neyman objectives ft enabled by Assumption 3.1 (hence the 'SC' in ClipOGDSC ), by applying the techniques for analyzing strongly convex gradient descent (Hazan et al., 2007; Rakhlin et al., 2012). Compared to the analysis in Dai et al. (2023), we make explicit the dependence of the regret of ClipOGD on the clipping rate. Note that the choice of h is flexible in the sense that any h(t) = o(t 0.2−ε) for any ε > 0 will result in a regret bound that is sublinear in T. From a practical standpoint, however, picking h may be a nontrivial affair, as a slower-growing h will have a faster-growing inverse mapping hinv. While the hinv-dependent term in the regret bound is constant in T, it can still be large in the constants of the problem. Intuitively, if C/c is large, the optimal propensity p
∗
T may be near the boundary and convergence may be slow. We hope future work will further explore the
'well-conditioning' properties of Neyman regret.

## 3.2. Convergence Of Adaptive Treatment Probabilities

We now investigate the trajectory of treatment probabilities
(pt)t≥1 produced by ClipOGDSC. Ideally, these propensities would converge to the optimal probabilities (p
∗T
)T ≥1 as T
grows large. By tweaking the arguments used in establishing our Neyman regret bounds of Theorem 3.2, we can obtain convergence in squared means (and hence in probability). The next claims formalize this result. In particular, we first establish a quantitative bound on the L2 convergence of our propensities to the benchmark ones. (See Appendix B for the derivation.)
Lemma 3.3 (L2-Deviation from Benchmark Design). The deviation of the design probabilities of ClipOGDSC from the best nonadaptive design probabilities is L2*-bounded for all* T as:

$$\mathbb{E}\left[\left(p_{T}-p_{T}^{*}\right)^{2}\right]\leq-\Theta\left(\frac{\mathbb{E}[\text{RegVar}_{T}]}{T}\right)+O\left(\frac{\left(h(T)\right)^{2}\log T}{T}\right)$$

.
This implies the following L2-convergence result, subject to an assumption on the Neyman regret of ClipOGDSC which asks for it to not consistently outperform the optimal nonadaptive design. Corollary 3.4 (L2-Convergence to Benchmark Design). Assume ClipOGDSC has asymptotically nonnegative Neyman regret: lim infT→∞
E[RegVarT ]
T ≥ 0*. Then, its propensities*
(pt)t≥1 will converge to the benchmark nonadaptive propensities (p
∗
T)T ≥1 *in squared means:* E-(pT − p
∗
T)
2→ 0 as T → ∞.

In the special case of sequences of potential outcomes that are (i.i.d.) samples from a superpopulation, the regret nonnegativity holds automatically, implying that our adaptive design will necessarily converge to the best nonadaptive design without further assumptions.

Corollary 3.5 (Convergence in the Superpopulation Setting). Suppose that the outcomes are drawn i.i.d. from a superpopulation: (yt(0), yt(1)) ∼ D for all t ≥ 1 and any fixed distribution D. Then, ClipOGDSC *guarantees that* E

-(pT − p
∗)
2→ 0 *at the rate* Oe(log T /T)*, and thus in* particular that pT → p
∗in probability.

Proof. In the superpopulation setting, any adaptive design will have nonnegative Neyman regret: ft(p) = f(p) =
E[y(1)2]/p + E[y(0)2]/(1 − p) has the same optimum p
∗ =1 + E[(yt(0))2]/ E[(yt(1))2]−1for all units t, so E[RegVarT
] = E
hPT
t=1 
(f(pt) − f(p
∗))i≥ 0.

## 3.3. Valid Cis For The Adaptive Ipw Estimator

We now turn to the issue of endowing the IPW estimator τˆT induced by our adaptive design with asymptotically valid confidence intervals (CIs). In general, the existence and construction of valid CIs for τˆT delicately depends on the choice of the design. However, we will now see that a construction of Dai et al. (2023) lends conservative CIs to all h-clipped adaptive designs with vanishing regret.

To formalize this result, we make a standard assumption: that the outcome sequences are not perfectly anticorrelated. To state it, define "empirical second raw moments" of the two outcome populations as: ST (i)
2:=
1 T
PT
t=1(yt(i))2for i ∈ {0, 1}.

Assumption 3.6 (Correlation of Outcome Populations (Dai et al., 2023)). For a constant cρ > 0 and all T ≥ 1, the running correlation ρT of the sequences {(yt(0), yt(1))}t≥1 satisfies:

 Hint:  $\rho_T\geq-1+c_\rho$, where $\rho_T:=\dfrac{\frac{1}{T}\sum_{t=1}^T y_t(1)y_t(0)}{S_T(1)S_T(0)}$. 
.
Theorem 3.7 (CIs for Clipped Adaptive Designs). Suppose the potential outcomes satisfy Assumption 3.1 and Assumption 3.6. Consider any h*-clipped adaptive design* (pt)t≥1 with vanishing Neyman regret: limT→∞ RegVarT = 0*. Let* VB = 4T
ST (1)ST (0) *be a conservative upper bound on the* hindsight-best nonadaptive variance. Then, letting (Zt)t≥1 be the treatment decisions, the estimator of Dai et al. (2023) given by:

$${\widehat{\mathrm{VB}}}={\frac{4}{T}}{\sqrt{\left({\frac{1}{T}}\sum_{t=1}^{T}(y_{t}(1))^{2}{\frac{Z_{t}}{p_{t}}}\right)\left({\frac{1}{T}}\sum_{t=1}^{T}(y_{t}(0))^{2}{\frac{1-Z_{t}}{1-p_{t}}}\right)}}$$
$\gamma\mu\nu e$
converges to VB *in probability at rate* Op

$$\left({\sqrt{h(T)/T}}\right).$$

Consequently, VBd can be used to construct asymptotically valid Chebyshev-type confidence intervals for the adaptive IPW estimator τˆT under any adaptive design satisfying the above conditions. Specifically, for any confidence level α ∈ (0, 1]:

$$\operatorname*{lim}_{T\to\infty}\operatorname*{Pr}\left[\tau_{T}\in\left[{\hat{\tau}}_{T}\pm\alpha^{-1/2}{\sqrt{\mathrm{{\vec{V B}}}}}\right]\right]\geq1-\alpha.$$

The proof for Theorem 3.7 is outlined in Appendix C.

Oe(
√T) multigroup Neyman regret bound.

## 4. Efficient Multigroup Ate Estimation

The Contextual Setting Section 3 covers non-contextual adaptive designs that only observe outcomes. A contextual adaptive design, however, also observes pre-treatment covariates xt ∈ X at the start of each round, which can help predict potential outcomes (yt(0), yt(1)). We can leverage this extra information to improve treatment assignments and outcome estimation. A Multigroup Formulation We frame the contextual setting in a multigroup way. Before the experiment, we have a finite set of context-defined groups G = {G1, G2*, . . .*},
each G ⊆ X , where X is the feature space. Any covariate vector xt can belong to none, one, or more groups. The group definition is dependent on the specifics of the task, e.g., in a medical application the features xt could represent a patient's health history. Our objective in a multigroup setting, informally, is to design an adaptive scheme that offers ATE estimation efficiency guarantees (such as Neyman regret guarantees) not only on average over the entire sequence of units but also on each subsequence that results from conditioning on units belonging to a group G, simultaneously for all groups G ∈ G.

## 4.1. A New Metric: Multigroup Neyman Regret

We introduce multigroup Neyman regret as a strengthening of (vanilla) Neyman regret. Specifically, given any contextual group collection G, G-multigroup Neyman regret is the maximum Neyman regret that an adaptive design achieves over any group G in the collection. We formalize it next. Definition 4.1 (G-Multigroup Neyman Regret). Given any group collection G ⊆ 2 X , the group-conditional Neyman regret of an adaptive design A on any group G ∈ G is defined as:

$$\mathrm{{RegVar}}_{T}({\mathcal{A}};G)$$
$$\begin{array}{l}{{\mathrm{Neg}\,\mathrm{var}_{T}({\mathcal{A}};G)}}\\ {{\mathrm{:=}\,\mathbb{E}\left[\operatorname*{max}_{p^{*}\in(0,1)}\sum_{t=1}^{T}\mathbb{1}[x_{t}\in G]\left(f_{t}(p_{t})-f_{t}(p^{*})\right)\right].}}\end{array}$$

The G-multigroup Neyman regret of A is then defined as its maximum group-conditional Neyman regret over all groups G ∈ G:

$$\mathrm{{RegVarMG}}_{T}({\mathcal{A}};{\mathcal{G}}):=\operatorname*{max}_{G\in{\mathcal{G}}}\mathrm{{RegVar}}_{T}({\mathcal{A}};G).$$

## 4.2. Achieving Oe( √T) **Multigroup Neyman Regret**

We now present in Algorithm 2 an adaptive design which we call MGATE (for Multi-Group ATE) and achieves the Algorithm 2 A*MGAT E*: Multigroup Adaptive Design Receive clipping function h : N+ → R>0 Receive number of groups d = |G| Set group counts n0 ← 0 d Initialize p1 ← 0.5 · 1 d // At round t, pt = (pt,G)G∈G will contain group propensities Initialize w
′1 ← 1 d, L0 ← 0 d, q0 ← 0 //
Parameters used to update group weights for t = 1, 2*, . . .* do Receive covariate vector xt ∈ X , determine the set of active groups Gt = {G : xt ∈ G, G *∈ G}* Cast Gt as indicator vector at ∈ {0, 1}
d(at,G =
1 ⇐⇒ G ∈ Gt). Set group counts: nt←nt−1 + at Normalize group weights: wt,eff ←
at⊙w
′ t
⟨at,w′t⟩
// Set inactive group weights to 0 Set effective treatment probability: pt,eff ← ⟨wt,eff, pt⟩ // Aggregate group propensities Set treatment decision: Zt ∼ Bernoulli(pt,eff) Receive realized outcome: Yt ← yt(Zt) for active groups G ∈ Gt do
/* Update group propensities using group-specific ClipOGDSC -type update */
Set estimated Neyman gradient as:
get,G ← Y
2 t Zt pt,eff
+1−Zt 1−pt,eff 
 −
Zt p 2 t,G
+1−Zt
(1−pt,G)2

$\;$) . 
Update pt+1,G ← Π
[δt,G,1−δt,G]
(pt,G − ηt,G · get,G),
where ηt,G ← 1 2c 2·nt,G
and δt,G ← 1 h(nt,G)
/* Get losses used to update group weights */
Set estimated Neyman loss as:
ℓet,G ← Y
2 t Zt pt,eff
+1−Zt 1−pt,eff 
  Zt pt,G
+1−Zt 1−pt,G 

end for for inactive groups G *̸∈ G*t do Set pt+1,G ← pt,G and ℓet,G ← 0 // Inactive groups are not updated end for
/* Update group weights: Higher cumulative group losses → larger weights */
Set surrogate loss: ℓt ← at ⊙
ℓet − ⟨ℓet, wt,eff⟩

Set Lt ← Lt−1 + ℓt and qt ← qt−1 + ∥ℓt∥
2 2 Update group weights: w
′
t+1 ← max n0 d, − √
1 qt Lt o end for Additional Notation: We use ⊙ to denote elementwise vector multiplication, and let 1 d, 0 dbe d-dimensional allones and all-zeros vectors. Also note that the update of w
′t+1 takes an *elementwise* maximum of the vectors, and assumes that 0/0 = 0 to account for the corner case qt = 0.

Algorithm Description: Given a collection G of d groups, in each round MGATE reads off the currently active groups Gt ⊆ G, i.e., those groups that contain xt (G ∋ xt), and then proceeds to determine the new treatment probability by aggregating the "best-guess" probabilities for all active groups G ∈ Gt determined based on the past performance of those groups. To do so, MGATE maintains group weights w
′
t,G and group-specific propensities pt,G.

It comes up with a single effective treatment probability:
pt,eff ∼PG∈Gt w
′
t,Gpt,G in each round by reweighing the group specific propensities of the active groups. This effective treatment probability should simultaneously satisfy the interests of all active groups. The treatment decision Zt is then generated according to pt,eff. After the outcome is revealed, MGATE updates all group weights, as well as the propensities of groups that were active.

We can show that MGATE achieves the following multigroup Neyman regret guarantee. We note that MGATE is anytime valid, meaning that just like our noncontextual design ClipOGDSC , it does not require advance knowledge of the time horizon T. Theorem 4.2 (Guarantees for Algorithm 2). Fix any context space X and finite group family G ⊆ 2 X . Suppose4 Assumption 3.1 holds with lower bound constant c > 0*. Then, for* any clipping function h*, the expected multigroup regret of* Algorithm 2 *will be bounded as:*

RegVarMGT (A; G) = O p|G| · (h(T))5· √T .

## 4.3. Technical Overview

The full analysis of Algorithm 2 is contained in Appendix D. It builds on several tools recently developed in the online learning literature, which are formally introduced in Appendix D.1, and we briefly survey them here. The central tool is the sleeping experts algorithmic framework (Blum &
Lykouris, 2020), which has recently been shown to be able to combine the wisdom of multiple sub-learners (or experts) into a meta-algorithm with performance on par with each of the sub-learners. The key difference from typical online aggregation schemes is that each sub-learner is allowed to be inactive (asleep) on some rounds, on which it does not 4By replacing the ClipOGDSC propensity updates in MGATE
with ClipOGD0-style updates, we can straightforwardly obtain a multigroup design which only relies on the assumptions of Dai et al. (2023) while keeping Oe(
√T) multigroup Neyman regret.

This follows from the generality of our multigroup meta-design presented in Appendix D, which can use a wide variety "ClipOGD-
style" updates while still obtaining Oe(
√T) multigroup regret.

give advice to the meta-algorithm. At a high level, to obtain multigroup Neyman regret, we would thus like to use a sleeping experts algorithm to aggregate propensities suggested by |G| = d copies of ClipOGDSC that are respectively active on all groups G ∈ G; the aggregated design would then perform comparably to each copy of ClipOGDSC on its group G. Then, since that copy of ClipOGDSC will have no regret on group G, neither will the aggregated design. Challenges and Solutions Past work on sleeping experts does not fully address the combination of difficulties present in our setting: (1) stochastic (realized outcome) feedback rather than full-information (both outcomes) feedback; (2) the need to perform clipping of the iterates (propensities)
to explicitly restrict them from approaching the feasible set's boundary too fast; and (3) the fact that the gradient feedback magnitude grows unboundedly as T → ∞, even with clipping. While there are a limited number of "sleeping bandits" algorithms in the literature (e.g., see Nguyen & Mehta (2024)) that address the stochastic feedback, they don't naturally extend to cover both of the latter two issues. Therefore, we design from scratch a new sleeping experts algorithm tailored to all of these challenges. It employs *scale-free* updates of the group weights w
′
tso as to control the loss and gradient feedback magnitudes; we achieve this by deploying an instance of the seminal scale-free SOLO FTRL algorithm of Orabona & Pál (2018) and endowing it with sleeping experts regret guarantees via a recent reduction of Orabona (2024). To clip the effective probability magnitudes, our algorithm aggregates over the suggested per-group probabilities via convex combinations rather than via sampling from their mixture. Finally, to ensure that the per-group propensity updates remain valid under stochastic gradient feedback and despite the aggregator using a different propensity than the suggested per-group one, MGATE uses a combination of unbiased first-order (get,G) and zeroth-order (ℓet,G) per-group feedback estimators, which depend on both pt,eff and pt,G.

A Generalized Meta-Design Our analysis in Appendix D generalizes beyond MGATE (Algorithm 2). Indeed, our approach more generally allows the use of any scale-free sleeping experts algorithm to update group weights, and any ClipOGD-style (see Appendix D.3) no-regret adaptive designs to update the groupwise treatment probabilities. Thus, we more generally provide a meta-design that reduces multigroup designs to a broad class of non-contextual, noregret designs. This generalized meta-design is given as Algorithm 7 in Appendix D.4, and Theorem D.6 contains its regret bound, of which Theorem 4.2 above is a corollary.

Treatm e nt p ro b a bi lity σ = 0.1 σ = 1 σ = 10 0.25 0.50 0.75 1000 3000 10000 30000 1000 3000 10000 30000 1000 3000 10000 30000 0.25 0.50 0.75 0.25 0.50 0.75 σ = 0.1 σ = 1 σ = 10 Round N
ey m a n reg ret 1000 3000 10000 30000 1000 3000 10000 30000 1000 3000 10000 30000 30 100 300 0.1 1.0 10.0 0.3 1.0 3.0 10.0 Method CLIPOGD0 CLIPOGDSC

## 5. Experimental Results

We first present the results for the non-contextual setting and then turn to the analysis of the performance for the contextual algorithm. Our code is available at the following link: https://github.com/amazon-science/adaptive-abtester.

## 5.1. Non-Contextual Experiments

Tasks We compare our method ClipOGDSC with ClipOGD0(Dai et al., 2023) on multiple tasks. Below, we show two key datasets (one synthetic and one realworld) used in our experiments, with full details in Appendix E. The first is a synthetic dataset is generated as follows: yt(i)
iid∼ N (µi, σ2) for t = 1, . . . , T and i = 0, 1 with µ0 = 1 and µ1 = 2. We vary σi ∈ R+ to showcase where our method succeeds and where it struggles. The second dataset comes from Egypt's largest microfinance organization (Groh & McKenzie, 2016), covering 2,961 clients. Here, the treatment is a new insurance product, and the outcome is how much individuals invest in machinery. Following Dai et al. (2023), we fill missing values with Gaussian noise and resample each unit five times to increase the population size. We also present experiments on the ASOS Digital Experiments Dataset (Liu et al., 2021), and on question-answering tasks for large language models, including BigBench (Srivastava et al., 2023), in the Appendix.

Experimental Setup In our simulation, each unit is randomly assigned to treatment or control using the treatment probability from our method or ClipOGD0. We repeat this process 10,000 times, generating many different treatmentcontrol paths. We then measure the Neyman regret by averaging the regret across these probabilities obtained at each time step. Hyperparameter Choices Throughout the experiments, we use the following hyperparameters. For our method, we set ηt = 2/t and δt = 1/h(t), where the clipping function is h(t) = exp(log(t + 2))1/4. For ClipOGD0, we follow Dai et al. (2023) with a constant learning rate ηt = 1/
√T
and clipping rate δt = 0.5 · t
−1/
√5 log T.

Results We analyze three synthetic data settings where we vary σ as {0.1, 1, 10}. As σ increases, the ratio C/c also grows, so by Equation (1), we expect slower convergence of our algorithm. We set T = 50,000. Figure 1 shows the Neyman regret across these settings, matching our theoretical expectations: when σ = 0.1, the regret of ClipOGDSC drops to 0 quickly, but for larger σ, the regret remains high and converges later. The regret of ClipOGD0 instead keeps increasing with time. Nonetheless, in line with Corollary 3.4, Figure 1 also shows that our method's adaptively chosen propensities ultimately converge to the Neyman optimal probability in all three cases. By contrast, the propensities of ClipOGD0 only converge when σ = 10, which happens to match the initial probability of 0.5. Next, we turn to examine the results on the microfinance data. Figure 2 illustrates the treatment probabilities and Neyman regret for both algorithms. On average, each design assigns probabilities near the Neyman probability. However, those of ClipOGD0exhibit higher variance compared to ClipOGDSC. This translates into greater Neyman regret in

1000 3000 10000 Round Treat me nt pr obab ility 0.4 0.5 0.6 0.7 0.8 1000 3000 10000 Round Ney ma n reg ret Method CLIPOGD0 CLIPOGDSC
0.00 0.01 0.02 Group = 0 Group = 1 Group = 2 0.000 0.005 0.010 0.015 0.020 Round Neyman re gret 0.00 0.01 0.02 0 5000 10000 15000 0 3000 6000 9000 0 3000 6000 9000 12000 0.01 0.02 Method MGATE CLIPOGDSC CLIPOGD0
later rounds, which never converges to 0. The probabilities assigned by our method, instead, converge to the Neyman probability, yielding vanishing average Neyman regret.

## 5.2. Contextual Experiments

Here we present our contextual results using Algorithm 2 over the previously-described datasets. To standardize the contextual groups in each experiment, we design simple, synthetic post-hoc groups by scoring each sample as st = 1/
1 + yt(0)2 yt(1)2+ϵ
(the optimal Neyman sampling probability for the single sample). Our groups are computed by checking whether sample t belongs to some predetermined quantile of the score function G0 = X , G1 = 1-F
−1(st) ≤
2 3
, G2 = 1-13 ≤ F
−1(st).

We note that these groups are overlapping and informative since G1 is guaranteed to have lower or equal optimal sampling probability than G2.

We stress that these groups are included for illustrative purposes and rely on information that would be unobservable in a real ATE experiment, but nonetheless showcase the potential for high-quality contextual information for multi-group ATE. Figure 3 shows the Neyman regret for ClipOGD0, ClipOGDSC, and MGATE on the microfinance dataset on each group; our MGATE method achieves the lowest group-conditional regret out of all the methods, effectively minimizing the G*-multigroup* Neyman regret, and thereby validating our theoretical results. Additional contextual experiments are provided in the Appendix.

## 6. Conclusion

In this paper, we have studied adaptive designs for unbiased ATE estimation with finite-population guarantees. We introduced a modification of the ClipOGD algorithm that provably yields vanishing Neyman regret, achieving an anytimevalid Oe(log T) Neyman regret, improving upon previous Oe(
√T) guarantees. We also extend our framework to incorporate contextual information by introducing a multigroup formulation. Our proposed multigroup adaptive design ensures Oe(
√T) regret for each predefined group, enabling efficiency improvements for subgroup ATE estimation. Experimental results corroborate these findings. Overall, these results suggest that adaptive experimentation can achieve strong finite-population efficiency guarantees, offering practical advantages for a wide range of applications. Future work could explore extensions to other experimental designs and further reductions in regret rates.

## Acknowledgements

The authors thank Vanessa Murdock for the support throughout this project, and Lorenzo Masoero, Blake Mason, and James McQueen for useful feedback.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Acharya, K., Arunachaleswaran, E. R., Kannan, S., Roth, A., and Ziani, J. Oracle efficient algorithms for groupwise regret. In The Twelfth International Conference on Learning Representations, 2024.

Blackwell, M., Pashley, N. E., and Valentino, D. Batch adaptive designs to improve efficiency in social science experiments, 2022.

Blum, A. and Lykouris, T. Advancing subgroup fairness via sleeping experts. In *Innovations in Theoretical Computer* Science Conference (ITCS), volume 11, 2020.

Chernozhukov, V., Demirer, M., Duflo, E., and Fernández-
Val, I. Fisher-schultz lecture: Generic machine learning inference on heterogenous treatment effects in randomized experiments, with an application to immunization in india. *arXiv preprint arXiv:1712.04802*, 2017.

Chow, S.-C. and Chang, M. Adaptive design methods in clinical trials–a review. *Orphanet journal of rare diseases*,
3:1–13, 2008.

Conneau, A., Rinott, R., Lample, G., Schwenk, H., Stoyanov, V., Williams, A., and Bowman, S. R. Xnli: Evaluating cross-lingual sentence representations. In 2018 Conference on Empirical Methods in Natural Language Processing, EMNLP 2018, pp. 2475–2485. Association for Computational Linguistics, 2018.

Cook, T., Mishler, A., and Ramdas, A. Semiparametric efficient inference in adaptive experiments. In NeurIPS 2023 Workshop on Adaptive Experimental Design and Active Learning in the Real World, 2023.

Dai, J., Gradu, P., and Harshaw, C. Clip-ogd: An experimental design for adaptive neyman allocation in sequential experiments. *Advances in Neural Information Processing* Systems, 36:32235–32269, 2023.

Deng, S., Liu, J., and Hsu, D. J. Group-wise oracle-efficient algorithms for online multi-group learning. Advances in Neural Information Processing Systems, 37:39462– 39500, 2024.

FDA. Adaptive designs for clinical trials of drugs and biologics: Guidance for industry. Technical Report, Center for Drug Evaluation and Research (CDER), Center for Biologics Evaluation and Research (CBER), 2019.

Fogliato, R., Patil, P., Akpinar, N.-J., and Monfort, M.

Precise model benchmarking with only a few observations. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 9563–9575, 2024.

Groh, M. and McKenzie, D. Macroinsurance for microenterprises: A randomized experiment in post-revolution egypt. *Journal of Development Economics*, 118:13–25, 2016.

Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S., and Athey, S. Confidence intervals for policy evaluation in adaptive experiments. Proceedings of the national academy of sciences, 118(15):e2014602118, 2021.

Hahn, J., Hirano, K., and Karlan, D. Adaptive experimental design using the propensity score. Journal of Business & Economic Statistics, 29(1):96–108, 2011.

Harshaw, C., Sävje, F., Spielman, D. A., and and, P. Z.

Balancing covariates in randomized experiments with the gram–schmidt walk design. Journal of the American Statistical Association, 119(548):2934–2946, 2024.

doi: 10.1080/01621459.2023.2285474. URL https:// doi.org/10.1080/01621459.2023.2285474.

Hazan, E., Agarwal, A., and Kale, S. Logarithmic regret algorithms for online convex optimization. Machine Learning, 69(2):169–192, 2007.

Hazan, E. et al. Introduction to online convex optimization.

Foundations and Trends® *in Optimization*, 2(3-4):157– 325, 2016.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E.,
Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A.,
Welbl, J., Clark, A., et al. Training compute-optimal large language models. *Advances in Neural Information* Processing Systems, 2022.

Horvitz, D. G. and Thompson, D. J. A generalization of sampling without replacement from a finite universe. *Journal* of the American statistical Association, 47(260):663–685, 1952.

Hu, F. and Rosenberger, W. F. The theory of responseadaptive randomization in clinical trials. John Wiley & Sons, 2006.

Imbens, G. W. and Rubin, D. B. Causal inference in statistics, social, and biomedical sciences. Cambridge university press, 2015.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C.,
Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Kato, M., Ishihara, T., Honda, J., and Narita, Y. Efficient adaptive experimental design for average treatment effect estimation. *arXiv preprint arXiv:2002.05308*, 2020.

Kern, C., Kim, M. P., and Zhou, A. Multi-accurate cate is robust to unknown covariate shifts. Transactions on Machine Learning Research: TMLR, pp. 1–59, 2024.

Lee, D., Noarov, G., Pai, M., and Roth, A. Online minimax multiobjective optimization: Multicalibeating and other applications. Advances in Neural Information Processing Systems, 35:29051–29063, 2022.

Li, H. H. and Owen, A. B. Double machine learning and design in batch adaptive experiments. Journal of Causal Inference, 12(1):20230068, 2024.

Li, J., Simchi-Levi, D., and Zhao, Y. Optimal adaptive experimental design for estimating treatment effect. arXiv preprint arXiv:2410.05552, 2024.

Liu, C., Cardoso, Â., Couturier, P., and McCoy, E. J.

Datasets for online controlled experiments. *NeurIPS* 2021 Datasets and Benchmarks Track, 2021.

Neopane, O., Ramdas, A., and Singh, A. Logarithmic neyman regret for adaptive estimation of the average treatment effect. *arXiv preprint arXiv:2411.14341*, 2024.

Neopane, O., Ramdas, A., and Singh, A. Optimistic algorithms for adaptive estimation of the average treatment effect. *arXiv preprint arXiv:2502.04673*, 2025.

Semenova, V. and Chernozhukov, V. Debiased machine learning of conditional average treatment effects and other causal functions. *The Econometrics Journal*, 24(2):264– 289, 2021.

Neyman, J. Sur les applications de la théorie des probabilités aux experiences agricoles: Essai des principes. *Roczniki* Nauk Rolniczych, 10(1):1–51, 1923.

Solomon, H. and Zacks, S. Optimal design of sampling from finite populations: A critical review and indication of new research areas. Journal of the American Statistical Association, 65(330):653–677, 1970.

Neyman, J. On the two different aspects of the representative method: the method of stratified sampling and the method of purposive selection. In Breakthroughs in statistics: Methodology and distribution, pp. 123–150. Springer, 1992.

Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A., Abid, A.,
Fisch, A., Brown, A. R., Santoro, A., Gupta, A., Garriga- Alonso, A., et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models.

Transactions on machine learning research, 2023.

Nguyen, Q. M. and Mehta, N. Near-optimal per-action regret bounds for sleeping bandits. In International Conference on Artificial Intelligence and Statistics, pp. 2827– 2835. PMLR, 2024.

Team, G., Riviere, M., Pathak, S., Sessa, P. G., Hardin, C., Bhupatiraju, S., Hussenot, L., Mesnard, T., Shahriari, B., Ramé, A., et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

Offer-Westort, M., Coppock, A., and Green, D. P. Adaptive experimental design: Prospects and applications in political science. *American Journal of Political Science*, 65(4): 826–844, 2021.

van der Laan, M. J. The construction and analysis of adaptive group sequential designs. *U.C. Berkeley Division of* Biostatistics Working Paper Series, 2008.

Orabona, F. Black-box reductions: Sleeping experts.

URL: https://parameterfree.com/2024/05/27/black-boxreductions-sleeping-experts/, 2024. Accessed 2024-1020.

Villar, S. S., Bowden, J., and Wason, J. Multi-armed bandit models for the optimal design of clinical trials: benefits and challenges. *Statistical science: a review journal of* the Institute of Mathematical Statistics, 30(2):199, 2015.

Orabona, F. and Pál, D. Scale-free online learning. Theoretical Computer Science, 716:50–69, 2018.

Wager, S. Causal inference: A statistical learning approach, 2024.

Pal, A., Umapathi, L. K., and Sankarasubbu, M. Medmcqa:
A large-scale multi-subject multi-choice dataset for medical domain question answering. In *Conference on health,* inference, and learning, pp. 248–260. PMLR, 2022.

Wald, A. Sequential tests of statistical hypotheses. In Breakthroughs in statistics: Foundations and basic theory, pp. 256–298. Springer, 1992.

Ponti, E. M., Glavaš, G., Majewska, O., Liu, Q., Vulic,´
I., and Korhonen, A. Xcopa: A multilingual dataset for causal commonsense reasoning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 2362–2376, 2020.

Xu, Y., Trippa, L., Müller, P., and Ji, Y. Subgroup-based adaptive (suba) designs for multi-arm biomarker trials. Statistics in Biosciences, 8:159–180, 2016.

Xu, Z., Zhang, K. W., and Murphy, S. A. The fallacy of minimizing local regret in the sequential task setting. arXiv preprint arXiv:2403.10946, 2024.

Rakhlin, A., Shamir, O., and Sridharan, K. Making gradient descent optimal for strongly convex stochastic optimization. In *Proceedings of the 29th International Conference* on Machine Learning, pp. 1571–1578, 2012.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800, 2019.

Rao, A. and Zhang, P. On distributional discrepancy for experimental design with general assignment probabilities. In The 28th International Conference on Artificial Intelligence and Statistics, 2024.

Zhang, K., Janson, L., and Murphy, S. Inference for batched bandits. Advances in Neural Information Processing Systems, 33:9818–9829, 2020.

Robbins, H. Some aspects of the sequential design of experiments. 1952.

Zhang, K., Janson, L., and Murphy, S. Statistical inference with m-estimators on adaptively collected data. Advances in Neural Information Processing Systems, 34:
7460–7471, 2021.

Rubin, D. B. Estimating causal effects of treatments in randomized and nonrandomized studies. Journal of educational Psychology, 66(5):688, 1974.

Zimmert, M. and Lechner, M. Nonparametric estimation of causal heterogeneity under high-dimensional confounding. *arXiv preprint arXiv:1908.08779*, 2019.

## Organization

The Appendix is organized as follows.

- Appendix A contains a discussion of further related work. - Appendix B contains proofs of our noncontextual method's convergence. - Appendix C discusses confidence interval guarantees for adaptive IPW estimators induced by our design. - Appendix D presents the general multigroup adaptive design framework and proves its efficiency guarantees. - Appendix E describes additional empirical results.

## A. Additional Related Work

Some early work on adaptive designs The efficient adaptive design problem considered in this work (and in (Dai et al., 2023) and other prior works) has its roots in problems studied, or mentioned, in the classical works of Neyman, Wald, and Robbins. The seminal work of Wald (1992) is the historical underpinning of much of the research in sequential experimental design, particularly on the hypothesis testing side. Our work shares the broad motivation of increasing statistical efficiency through adaptivity, though we do not explicitly deal with some of the notions that Wald focused on such as early stopping. As described in (Dai et al., 2023), it was Robbins (1952) who explicitly posed the open problem of constructing efficient adaptive sampling schemes, though he stopped short of formally reasoning about precise benchmarks such as Neyman regret that such designs must optimize; and while within the subsequent decades, adaptive designs have gained traction e.g. in the survey sampling context Solomon & Zacks (1970), but yet again they did not explicitly optimize for efficiency metrics like the ones we study. Improved Neyman Regret in the Superpopulation Setting Independently and concurrently to our work, complementary progress has been made on the problem of (noncontextual) adaptive ATE estimation with low Neyman regret. Neopane et al. (2024) modify ClipOGD to obtain logarithmic Neyman regret guarantees in the substantially more benign superpopulation regime. Neopane et al. (2025), in an environment in which outcomes are generated from some joint distribution with time-stationary means and variances, design a Neyman regret minimization algorithm based on a UCB-like optimistic policy tracking approach, with both theoretical guarantees and strong empirical performance.

Covariate Balancing In relation to our multigroup setting, it is interesting to discuss recent progress on what is known as *covariate balancing* techniques. For instance, studies such as Harshaw et al. (2024), Rao & Zhang (2024), which are conducted in the non-adaptive setting, share with our work the objective of optimizing the estimation variance - and offer a principled way to exploit covariates to do so, by assuming the mapping between covariates and outcomes is linear (and trading off robustness and covariate balance). Thus, a potential future work direction can involve porting the covariate balancing insights over to the adaptive setting, possibly leading to a new perspective on group-aware optimal-variance adaptive designs that could complement our multigroup approach.

## B. Non-Contextual Setting: Proof Of Theorem 3.2 And Of Lemma 3.3

B.1. Neyman Regret Analysis for ClipOGDSC : Proof of Theorem 3.2 We establish Theorem 3.2 via a sequence of claims.

Claim 1 (Optimal Probability Bounds; Lemma C.2 of Dai et al. (2023)). The optimal fixed probability p
∗Tfor any time horizon T satisfies, under Assumption 3.1, the following inequality, defining the constant A = 1 + C/c ≥ 2:

$${\frac{1}{A}}\leq p_{T}^{*}\leq1-{\frac{1}{A}}.$$

Claim 2 (How Quickly Optimal Probability Enters Admissible Region). Under Assumption 3.1, let A = 1 + C/c ≥ 2. Then, for any time horizon T, the optimal probability p
∗
T will satisfy:

$\mathbf{a}$
t ≥ t
∗ =⇒ p
∗
T ∈ [δt, 1 − δt], where t
∗:= hinv(A).

Proof. With 1 in hand, we have that as soon as δt ≤ 1/A, the optimal probability p
∗T
(for any T) is guaranteed to be in the admissible interval [δt, 1 − δt]. This is equivalent to requiring h(t) ≥ A, which by definition of hinv and by the strictly increasing nature of h is equivalent to t ≥ hinv(A).

Claim 3 (Gradient Raw Moment Bounds). Under Assumption 3.1, for every t ≥ 1 we have the following bounds in expectation wrt. the design's randomness:

$$\mathbb{E}[|g_{t}|]\leq2C^{2}h(t)^{2},\quad\mathbb{E}[g_{t}^{2}]\leq2C^{4}h(t)^{5}.$$

Proof. The bounds follow as shown in Lemma C.5 of Dai et al. (2023), by just expanding out the first and second raw absolute moment of the gradient estimator defined above; we will get E[|gt|] ∼ δ
−2 t(yt(1)2 + yt(0)2), and E[g 2 t] ∼
δ
−5 t(yt(1)4 + yt(0)4), so the statement follows from our Assumption 3.1, or from Dai et al. (2023)'s assumption on the boundedness of the second and fourth moments of the two populations.

Claim 4 (Strong Convexity of Objective). For any round t ≥ 1, and for any *p, p*′ ∈ (0, 1), the objective function will satisfy:

$$f_{t}(p)-f_{t}(p^{\prime})\leq f^{\prime}(p)\cdot(p-p^{\prime})-c^{2}(p-p^{\prime})^{2}$$
2.
Proof. To show this, it suffices to establish 2c 2-strong convexity of ft(p) = yt(0)2 p +
yt(1)2 1−p
, and we will do so by verifying that f
′′(p) ≥ 2c 2for all p ∈ (0, 1). Indeed, note that f
′′(p) = 2 yt(0)2 p3 +
yt(1)2
(1−p)
3
≥ 2(yt(0)2 + yt(1)2) ≥ 2c 2since p ∈ (0, 1) and by definition of c in Assumption 3.1.

Claim 5. For any t ≥ 1, any setting of ηt > 0, δt = 1/h(t), and for any point p
∗ ∈ {p
∗
t }t≥1, we have in expectation over the randomness of the design:

$$\mathbb{E}[f_{t}(p_{t})-f_{t}(p^{*})]\leq\left(\frac{1}{2\eta_{t}}-c^{2}\right)\mathbb{E}[(p_{t}-p^{*})^{2}]-\frac{1}{2\eta_{t}}\,\mathbb{E}[(p_{t+1}-p^{*})^{2}]+\eta_{t}\cdot(Ch(t))^{5}$$ $$+2\cdot1[t\leq t^{*}]\cdot\left(\frac{1}{\eta_{t}\cdot h(t)}+(Ch(t))^{2}\right).$$

Proof. By Claim 4 applied to p = pt and p
′ = p
∗, we have ft(pt) − ft(p
∗) ≤ f
′(pt) · (pt − p
∗) − c 2(pt − p
∗)
2. Now, we can bound the first term on the right-hand side as follows. First, start with the inequality: |pt+1 − p
∗*| ≤ |*pt − ηtgt − p
∗| + δt · 1[p
∗ ̸∈ [δt, 1 − δt]], which follows by Lemma C.1 in Dai et al. (2023). By Claim 2, we have that 1[p
∗ ̸∈ [δt, 1 − δt]] = 0 for all t ≥ t
∗, implying that 1[p
∗ ̸∈ [δt, 1 − δt]] ≤ 1[t ≤ t
∗].

Thus, we have |pt+1 − p
∗*| ≤ |*pt − ηtgt − p
∗| + δt · 1[t ≤ t
∗]. Squaring this inequality, we arrive, after rearranging terms and using the triangle inequality, at

$$\left(p_{t+1}-p^{*}\right)^{2}\leq\left(p_{t}-p^{*}\right)^{2}+\eta_{t}^{2}g_{t}^{2}-2\eta_{t}g_{t}(p_{t}-p^{*})+4\cdot1[t\leq t^{*}]\cdot\eta_{t}\cdot\delta_{t}\left(\frac{1}{\eta_{t}}+\frac{|g_{t}|}{2}\right).$$

Rearranging terms once again, we get:

$$2\eta_{t}g_{t}(p_{t}-p^{*})\leq(p_{t}-p^{*})^{2}+\eta_{t}^{2}g_{t}^{2}-(p_{t+1}-p^{*})^{2}+4\cdot1[t\leq t^{*}]\cdot\eta_{t}\cdot\delta_{t}\left(\frac{1}{\eta_{t}}+\frac{|g_{t}|}{2}\right).$$

Dividing this by ηt > 0, we get:

$$2g_{t}(p_{t}-p^{*})\leq\frac{1}{\eta_{t}}\left((p_{t}-p^{*})^{2}-(p_{t+1}-p^{*})^{2}\right)+\eta_{t}g_{t}^{2}+4\cdot1[t\leq t^{*}]\cdot\delta_{t}\left(\frac{1}{\eta_{t}}+\frac{|g_{t}|}{2}\right).$$

Noting that E[gt|Ft] = f
′t(pt) by definition of gt, as well as using the bounds on the expected gradient moments from Claim 3, we can take the expectation of the last inequality to obtain:

$$2f_{t}^{\prime}(p_{t})(p_{t}-p^{*})\leq\frac{1}{\eta_{t}}\left((p_{t}-p^{*})^{2}-\mathbb{E}[(p_{t+1}-p^{*})^{2}|\mathcal{F}_{t}]\right)+\eta_{t}\mathbb{E}[g_{t}^{2}|\mathcal{F}_{t}]+4\cdot1[t\leq t^{*}]\cdot\delta_{t}\left(\frac{1}{\eta_{t}}+\frac{\mathbb{E}[|g_{t}|\mathcal{F}_{t}]}{2}\right)$$

14

$$\leq\frac{1}{\eta_{t}}\left((p_{t}-p^{*})^{2}-\mathbb{E}[(p_{t+1}-p^{*})^{2}|\mathcal{F}_{t}]\right)+\eta_{t}\cdot2C^{4}h(t)^{5}+4\cdot1[t\leq t^{*}]\cdot\delta_{t}\left(\frac{1}{\eta_{t}}+C^{2}h(t)^{2}\right).$$

Returning to the strong convexity-induced inequality above, we thus have:

ft(pt) − ft(p ∗) ≤ f ′(pt) · (pt − p ∗) − c 2(pt − p ∗) 2 ≤1 2ηt (pt − p ∗) 2 − E[(pt+1 − p ∗) 2|Ft]+ ηt · C 4h(t) 5 + 2 · 1[t ≤ t ∗] · δt 1 ηt + C 2h(t) 2 − c 2(pt − p ∗) 2 = 1 2ηt − c 2 (pt − p ∗) 2 −1 2ηt E[(pt+1 − p ∗) 2|Ft] + ηt · C 4h(t) 5 + 2 · 1[t ≤ t ∗] · δt · 1 ηt + C 2h(t) 2.
Now, taking expectation again, now with respect to the randomness up through Ft, we obtain the statement of this claim.

Claim 6 (Convergence Bound). For any time horizon T, and any p
∗ ∈ {p
∗
t }t≥1, we have:

$$\sum_{t=1}^{T}\mathbb{E}[f_{t}(p_{t})-f_{t}(p^{*})]$$ $$\leq-c^{2}(T+1)\,\mathbb{E}[(p_{T+1}-p^{*})^{2}]+\frac{C^{5}}{2c^{2}}h(T)^{5}(\log(T+1)+1)+2C^{2}\left(1+\frac{C}{c}\right)^{2}h_{\rm inv}\left(1+\frac{C}{c}\right)^{2}$$ $$\quad+2c^{2}\left(h_{\rm inv}\left(1+\frac{C}{c}\right)+1\right)^{2}.$$  _in the interval in $\mathbb{C}$ is $\mathbb{C}$ for $\mathbb{A}$ to $\mathbb{C}$. The $\mathbb{A}$ is $\mathbb{A}$-valued 
Proof. Summing the inequality in Claim 5 from t = 1 to t = T, we obtain via telescoping sums:
X
T
t=1
E[ft(pt) − ft(p
∗)]
≤X
T
t=1
1
2ηt
− c
2
E[(pt − p
∗)
2] −X
T
1
2ηt
E[(pt+1 − p
∗)
2] +X
T
t=1
ηt · (Ch(t))5
t=1
+X
T
t=1
2 · 1[t ≤ t
∗] ·
1
ηt · h(t)
+ (Ch(t))2

≤
X
T
t=1
1
2ηt
− c
2E[(pt − p
∗)
2] −
X
T
1
2ηt
E[(pt+1 − p
∗)
2] +X
T
t=1
ηt · (Ch(t))5
t=1
+ 2
tX∗
t=1
1
ηt · h(t)
+ (Ch(t))2

≤X
T
t=1
1
2ηt
− c
2
E[(pt − p
∗)
2] −X
T
1
2ηt
E[(pt+1 − p
∗)
2]
t=1
t=1
ηt + 2t
∗· (Ch(t
∗))2 + 2
tX∗
t=1
+ (Ch(T))5X
T
1
ηt · h(t)
=
1
2η1
− c
2
E[(p1 − p
∗)
2] −1
2ηT +1
E[(pT +1 − p
∗)
2]
t=1
ηt + 2t
∗· (Ch(t
∗))2 + 2
tX∗
+X
T
t=2
1
2ηt
−1
2ηt−1
− c
2E[(pt − p
∗)
2] + (Ch(T))5X
T
1
ηt · h(t)
t=1
15
≤ −c 2(T + 1) E[(pT +1 − p ∗) 2] +  (Ch(T))5 2c 2(log(T + 1) + 1) + 2t ∗· (Ch(t ∗))2 + 4c 2 t X ∗ t h(t) t=1
.
Finally, recalling the definition of t
∗ = hinv(A) = hinv(1 + C/c) and substituting it in, we obtain the desired claim.

Finally, with the result of Claim 6 in hand, we observe that (1) the term −c 2(T + 1) E[(pT +1 − p
∗)
2] is nonpositive and can thus be ignored, (2) the second term on the right hand side is asymptotically O((h(T))2· log T), and (3) the third and fourth terms on the right hand side are constant with respect to T and only a function of the constants *C, c* of the problem. This gives the desired result.

## B.2. Convergence Of Treatment Probabilities Of Clipogdsc : Proof Of Lemma 3.3

We will make use of Claim 6 from the previous subsection. Simply rearranging the terms, we obtain the following bound for the deterministic setting:

$$c^{2}(T+1)\,\mathbb{E}[(p_{T+1}-p_{T}^{*})^{2}]\leq-\sum_{t=1}^{T}\mathbb{E}[f_{t}(p_{t})-f_{t}(p_{T}^{*})]+\frac{C^{2}}{2c^{2}}h(T)^{2}(\log(T+1)+1)+O(1),$$

where the O(1) term hides terms in the bound that do not depend on T. Dividing through by c 2· (T + 1) and reindexing for convenience, we obtain the desired result:

$$\mathbb{E}[(p_{T}-p_{T}^{*})^{2}]\leq-\Theta\left(\frac{\mathbb{E}[\mathrm{Reg}_{T}]}{T}\right)+O\left(\frac{(h(T))^{2}\log T}{T}\right).$$

## C. Confidence Interval Guarantees: Proof Sketch For Theorem 3.7

Remark C.1 (Chebyshev vs. Wald Confidence Intervals). As Dai et al. (2023) point out, it appears that ClipOGD may lead to an asymptotically normal distribution of the IPW estimator. If this were true, that would allow us to get Wald-type confidence intervals for the IPW estimator based on the variance estimator VBd, which would be narrower than Chebyshev-type ones.

Through some simulations, we observed that asymptotically, the z-score of the IPW estimator induced by our adaptive scheme appears to satisfy asymptotic normality. However, below we only prove the validity of Chebyshev-type confidence intervals, and leave Wald-type CIs to be explored in future work. We will convince ourselves that the techniques employed in Dai et al. (2023) for proving the validity of this variance estimator apply to a broad class of adaptive sampling schemes. Dai et al. (2023) state this result for their particular adaptive design but mention that it may apply to other learning rate and clipping rate settings. And indeed, we find that while their approach does depend on the adaptive design having sufficiently slowly decaying clipping rate and vanishing Neyman regret, it is oblivious to hyperparameters such as the learning rate. Moreover, we find that the condition of having asymptotically nonnegative Neyman regret, which Dai et al. (2023) impose on the design, is also not necessary to ensure that the variance estimator VBd is conservatively valid.

For easier tracking of the relevant quantities, recall the notation: ST (i) := q1T
PT
t=1 yt(i)
2 for i ∈ {0, 1}. Following
(Dai et al., 2023), we define the quantities AT (1) = (ST (1))2, AT (0) = (ST (0))2, as well as the quantities A\T (1) =
1 T
Ptyt(1)2 Zt pt
, A\T (0) = 1T
Ptyt(0)2 1−Zt 1−pt that estimate them in an unbiased way. Recalling that the variance of the optimal nonadaptive design (i.e., the variance of the IPW estimator that uses p
∗
Tas its fixed sampling probability on all rounds t = 1 *. . . T*) is2

$${\frac{2}{T}}(1+\rho)S_{T}(1)S_{T}(0)\leq\mathrm{VB}:={\frac{4}{T}}{\sqrt{A_{T}(1)A_{T}(0)}},$$

we can see that VB = d 4 T
qA\T (1)A\T (0) simply aims to approximate the upper bound VB on the optimal fixed-probability sampling scheme's variance. And given that our design has a no-regret guarantee with respect to this benchmark, VBd thus also asymptotically approximates the upper bound on our (and any other such) design's induced IPW estimator variance VT . This is the blueprint of the proof, and we will now briefly revisit the technical steps in Dai et al. (2023) that make this blueprint argument work.

First, Proposition D.1 of Dai et al. (2023) proves that

$$\left|\mathbb{E}[\widehat{A_{T}(1)}\widehat{A_{T}(0)}]-A_{T}(1)A_{T}(0)\right|\leq\frac{C^{4}}{T},$$

which by tracking the proof can be seen to not depend on the sampling scheme. Second, by generalizing the result and steps of Proposition D.2 of Dai et al. (2023), we can bound the variance of the
(normalized version of the) estimator VBd as:

$$\mathrm{Var}(\widehat{A_{T}(1)A_{T}(0)})\leq\frac{C^{8}\cdot h(T)}{T}+\frac{C^{8}\cdot(h(T))^{2}}{T^{2}}\leq\frac{2C^{8}\cdot h(T)}{T}.$$

Thus, applying Chebyshev's inequality to this variance bound and using the preceding in-expectation bound, we conclude that A\T (1)A\T (0) → AT (1)AT (0) in probability at the rate Op((h(T)/T)
1/2).

Now, as in the proof of Theorem 5.1 of Dai et al. (2023), we can observe that a Continuous Mapping Theorem can be applied to this in-probability convergence result to give the implication that qA\T (1)A\T (0) →pAT (1)AT (0) at the same asymptotic rate Op((h(T)/T)
1/2). Indeed, since the target random variable AT (1)AT (0) is bounded below by c 2 by Assumption 3.1, the square root transformation will be Lipschitz on the relevant range (i.e., away from zero). Finally, to establish the validity of the Chebyshev-type confidence intervals given above, it suffices to look at the z-score statistic ζ = √τT −τˆT
Var(ˆτT )
and the estimated z-score statistic ζ
′ =
τ√T −τˆT
VBd and establish that ζ stochastically dominates ζ
′.

Towards this, note as in Dai et al. (2023) that:

$$\zeta^{\prime}=\zeta\cdot\left({\sqrt{\frac{\mathrm{Var}({\hat{\tau}}_{T})}{\mathrm{VB}}}}\cdot{\sqrt{\frac{T\cdot\mathrm{VB}}{T\cdot{\widehat{\mathrm{VB}}}}}}\right)$$
.
First, since the estimator τˆT is induced by a no-regret adaptive design and since VB is an upper bound on the variance of the best fixed SRS scheme (which serves as the benchmark of the design's regret performance), we have that lim supT→∞
Var(ˆτT )
VB ≤ 1. Second, from what we just obtained, T · VBd → T · VB in probability, which in view of T · VBd being lower-bounded by a constant by Assumption 3.1 implies by the Continuous Mapping Theorem that qT·VB
T·VBd converges to 1 in probability. By Slutsky's theorem, this proves the desired stochastic domination and thus implies that the proposed confidence interval construction is asymptotically (conservatively) valid.

## D. Multigroup Adaptive Design: Proofs And Details D.1. Olo Primitives

Our multigroup design will rely on a sequence of reductions, derived with the help of some online learning machinery: a recent reduction of Orabona (2024) and scale-free algorithms by Orabona & Pál (2018). First, we spell out the algorithmic primitives that we will require.

Definition D.1 (OLO algorithm; OLO regret). An *OLO (online linear optimization) algorithm* A over domain V ⊆ R
d, where d ≥ 1 is the dimension of the problem, sequentially receives vectors ℓt ∈ R
d, t = 1, 2*, . . .*. Each ℓt is interpreted as the "gradient", or the "loss", that A suffers at round t.

Each round, before seeing ℓt, algorithm A outputs iterate vt ∈ V as a function of past history. The algorithm's *regret* at any time T is defined as the total loss incurred by its iterates minus the total loss of the best-in-hindsight admissible solution:

$\operatorname{Reg}_{T}(\mathcal{A}):=\max_{v\in V}\operatorname{Reg}_{T}(\mathcal{A};v),\quad\text{where}\operatorname{Reg}_{T}(\mathcal{A};v)=\sum_{t=1}^{T}\langle\ell_{t},v_{t}-v\rangle\text{for}v\in V.$
Definition D.2 (Sleeping Experts algorithm; SE regret). A *sleeping experts (SE) algorithm* A over domain V ⊆ R
d, where d ≥ 1 is the number of "sleeping experts", sequentially receives vectors at ∈ {0, 1}
dand ℓt ∈ R
dat rounds t = 1, 2*, . . .*.

The vector at has the interpretation that at,i ∈ {0, 1} (for each i ∈ [d]) denotes whether expert i is "active" (1) or "inactive" (0) in round t. The vector ℓt has the interpretation that at any round t, for all active experts i (i.e., at,i = 1), expert i's loss is ℓt,i, while for all inactive experts i the loss coordinate ℓt,i is (arbitrarily) equal to 0. Each round, after seeing at but before seeing ℓt (i.e., after seeing which experts are active but before seeing their losses), the algorithm outputs a distribution vt ∈ ∆d as a function of past history, such that vt,i = 0 for all inactive experts (i.e., for all i such that at,i = 0). In words, at each round the algorithm is required to output a distribution vt over the currently active experts only. The algorithm's *Sleeping Experts regret* at any time T is defined as the upper bound, over all experts i ∈ [d], on its performance relative to expert i over those rounds t on which i *was active*:

$$\mathrm{RegSE}_{T}({\mathcal{A}}):=\operatorname*{max}_{i\in[d]}\sum_{t=1}^{T}a_{t,i}\cdot(\langle\ell_{t},v_{t}\rangle-\ell_{t,i}).$$

Scale-Free OLO We will make use of a *scale-free* OLO algorithm (Orabona & Pál, 2018) to design a base algorithm for our multigroup regret algorithm. The property of any such algorithm is that its regret bound does not require the norms of the gradients ℓt to be bounded in [0, 1] for some norm (like standard OLO methods typically require).

Fact 1 (Theorem 1 of (Orabona & Pál, 2018)). Fix any norm ∥·∥ and its dual norm ∥·∥∗
. Then, Algorithm 3 called SOLO
FTRL achieves, for any convex closed set V ⊆ R
d, the following regret bound *to any point* v ∈ V that scales with the magnitude of the losses/gradients:

$$\operatorname{Reg}_{T}(\operatorname{SOLO}\operatorname{FTRL};v)\leq(R(v)+2.75)\sqrt{\sum_{t=1}^{T}\left\|\ell_{t}\right\|_{*}^{2}}+3.5\min\left\{\sqrt{T-1},\operatorname{diam}(V)\right\}\max_{t\in[T]}\left\|\ell_{t}\right\|_{*}.$$
$\mathrm{,2018)}$ . 
.
where diam(V ) = supv1,v2∈V ∥v1 − v2∥, and where SOLO FTRL is parameterized by an arbitrary nonnegative continuous 1-strongly-convex regularizer R : V → R.

Algorithm 3 A*SOLO*: SOLO FTRL (Orabona & Pál, 2018)
Receive domain V ⊆ R
d base regularizer R(w), and norm ∥·∥.

Initialize L0 ← 0 d, q0 ← 0.

for t = 1, 2*, . . .* do Compute new weights wt ← arg min w∈V
{⟨Lt−1, w⟩ + Rt(w)}, where Rt(w) = 
√qt−1 · R(w).

Receive loss vector ℓt.

Set Lt ← Lt−1 + ℓt.

Set qt ← qt−1 + ∥ℓt∥
2
∗.

end for

## D.2. Designing A Scale-Free Sleeping Experts Algorithm

Now, let us instantiate the above Fact 1 appropriately. First, set the norm for the regret bound to be the 2-norm: ∥·∥ =
∥·∥∗ = ∥·∥2
. Second, set the regularizer to be R(v) := ∥v∥
2 2 for v ∈ V , which is 1-convex with respect to the 2-norm.

Third, set the domain of the algorithm to be the non-negative orthant: V = R
d
≥0. We then arrive at the following guarantee.

Corollary D.3 (of Fact 1). *With the nonnegative orthant* V = R
d≥0as domain and the squared L2*-norm as regularizer,*
SOLO FTRL achieves the following scale-free regret bound for all v ∈ V :

$$\mathrm{Reg}_{T}(\mathrm{SOLO\,FTRL};v)\leq\left(\|v\|_{2}^{2}+6.25\right)\operatorname*{max}_{t\in[T]}\|\ell_{t}\|_{2}\,\sqrt{T}.$$

The instantiation of SOLO FTRL for these specific choices is given in Algorithm 4.

We note that the update for wt in Algorithm 4 is the solution to the original argmax problem in Algorithm 3, with the nonnegative orthant as domain and the rescaled L2-norm as regularizer. Algorithm 4 A*SOLO*: Instantiation for scale-free sleeping experts 1: Initialize L0 ← 0 d, q0 ← 0.

2: for t = 1, 2*, . . .* do 3: Set weights wt ← max n0 d, − √
1 qt−1 Lt−1 o(coordinate-wise maximum).

4: Receive loss vector ℓt ∈ R
d≥0
.

5: Set Lt ← Lt−1 + ℓt. 6: Set qt ← qt−1 + ∥ℓt∥
2 2
.

7: **end for**

Scale-Free Sleeping Experts Now, we will turn this just obtained scale-free OLO regret guarantee into a scale-free sleeping experts regret guarantee. We will utilize a recent black-box reduction mechanism of Orabona (2024), which proceeds as follows. Fact 2 (Sleeping Experts to OLO Reduction (Orabona, 2024)). Consider a sleeping experts setting with d experts. Define
any base OLO algorithm A with nonnegative orthant V = R
d
≥0as the domain. Then Algorithm 5, which we refer to as
AOLO→SE, constructs a sequence v1, v2*, . . .* of distributions over active experts that attains the following sleeping experts
regret bound:
RegSET(AOLO→SE) = max
v∈SB(Rd)
RegT
$$\operatorname{\mathbb{g}}_{T}\left({\mathcal{A}}\left(\left\{{\widetilde{\ell}}_{t}\right\}_{t\in\mathbb{I}}\right)\right)$$
t∈[T]
; v
.

Here, SB(R
d) as the collection of the d standard basis (unit) vectors of R
d; and the vectors {ℓet}t∈[T], defined in Algorithm 5, are surrogate loss vectors. Note that these surrogate losses satisfy ℓet
∞
≤ 2 ∥ℓt∥∞ relative to the original losses {ℓt}t∈[T].

Algorithm 5 AOLO→SE: Sleeping Experts to OLO Reduction (Orabona, 2024)
Initialize any base OLO algorithm A with nonnegative orthant V = R
d≥0as domain.

for t = 1, 2*, . . .* do Get unscaled prediction wt ∈ R
d
≥0from A.

Receive indicator vector describing which experts are active: at ∈ {0, 1}
d.

Construct distribution vt ∈ ∆d as: vt,i =
at,iwt,i
⟨at,wt⟩
for i ∈ [d].

Receive loss vector ℓt ∈ R
d.

Construct surrogate loss vector ℓet as ℓet,i = at,i(ℓt,i − ⟨ℓt, vt⟩) for i ∈ [d], and send it to A.

end for To obtain sleeping experts regret bounds scaling with the norm of the losses, we can implement this reduction with the scale-free Algorithm 4 at its base. Formally, we have the following statement.

Theorem D.4 (Scale-Free Sleeping Experts Algorithm). Consider a sleeping experts setting with d experts. Initialize Algorithm 5 using Algorithm 4 (an instance of SOLO FTRL with settings described in Corollary *D.3) as its base OLO* subroutine. Call the resulting sleeping experts algorithm ASOLO SE, with the pseudocode given in Algorithm 6. Then, SOLO
SE obtains the following sleeping experts regret bound on any sequence of losses {ℓt}t∈[T]:
RegSET
A*SOLO SE* {ℓt}t∈[T]
 ≤ 15 max t∈[T]
∥ℓt∥∞
√*dT .*

## D.3. First-Order Neyman Regret Minimization

We now formalize (and generalize) how the ClipOGD design operates. This formalization will define the scope of noncontextual adaptive designs that can be used to estimate group propensities for all groups in our multigroup design.

Definition D.5 (First-order Neyman Regret Minimization). Recall the Neyman objectives: ft(p) = yt(1)2 p +
yt(0)2 1−pfor p ∈ (0, 1), t ≥ 1, where .{(yt(1), yt(0))}t≥1 are the potential outcomes. A first-order Neyman regret minimization algorithm AATE follows the following protocol for sequential ATE estimation: At each round t = 1, 2*, . . .*, AATE decides on a treatment probability pt ∈ (1/h(t), 1 − 1/h(t)), where h : N+ → R>0 is a Algorithm 6 ASOLO SE: Sleeping Experts Algorithm 1: Initialize A*SOLO*, an instance of Algorithm 4.

2: for t = 1, 2*, . . .* do 3: Receive unscaled weights wt ∈ R
d≥0 from A*SOLO*.

4: Receive indicator vector describing which experts are active: at ∈ {0, 1}
d.

5: Set rescaled weights vt ∈ ∆d as: vt,i =
at,iwt,i
⟨at,wt⟩
for i ∈ [d].

6: Receive loss vector ℓt ∈ R
d.

7: Set surrogate loss vector ℓet as ℓet,i = at,i(ℓt,i − ⟨ℓt, vt⟩) for i ∈ [d].

8: Send ℓet to A*SOLO*.

9: **end for**
strictly increasing clipping function. After that, AATE receives *first-order feedback* get from the environment, which is a random variable that satisfies the following properties: (1) It is adapted to the natural filtration {Ft}t≥1 of the process, i.e.,
the distribution of get is determined by all prior history up to and including determining pt; (2) It is an unbiased estimator of f
′
t
(pt), in that E[get|Ft−1] = f
′
t
(pt) = −
yt(1)2 p 2 t
+yt(0)2
(1−pt)
2 .

It is easy to observe that Algorithm 1 conforms to Definition D.5. Algorithm 1 is written as requiring direct access to the selected outcome Yt, but this outcome is only used to compute the unbiased gradient estimator f
′t(pt).

## D.4. Multigroup-Adaptive Design Via Sleeping Experts

We are now ready to present a context-aware algorithm for online ATE estimation. It uses scale-free sleeping experts as derived above, as well first-order Neyman regret minimization algorithms as base learners. The following theorem states its most general guarantees (as well as the specific instantiation that gives MGATE). The proof is presented in the next subsection.

Theorem D.6 (Guarantees for Algorithm 7). *Consider any first-order Neyman regret minimization algorithm* AATE and any scale-free sleeping experts algorithm ASE. Fix any context space X and any finite group family G ⊆ 2 X *. If the base* learners for all G ∈ G are copies of AATE, Algorithm 7's expected multigroup regret will be bounded for all G ∈ G as:
E [RegVarT(A; G)] ≤ E [RegSET(ASE)] + E [RegVarT(AATE(G))] .

Moreover, Algorithm 7 is anytime*, as it does not require advance knowledge of the time horizon* T.

Instantiate Algorithm 7 using h-clipped ClipOGDSC as the base ATE algorithm, for some strictly increasing h, and use ASOLO SE (Algorithm 6) as the scale-free SE algorithm. Then, we obtain the MGATE design (Algorithm 2) that simultaneously offers the following guarantees for all G ∈ G:

$$\mathbb{E}\left[\mathrm{{{\mathrm{RegVar}}}}_{T}({\mathcal{A}};G)\right]=O\left({\sqrt{|{\mathcal{G}}|}}\cdot(h(T))^{5}\cdot{\sqrt{T}}\right).$$

## D.5. Proof Of Theorem D.6

First, note that with the Neyman objective defined, as always, via ft(p) = yt(1)2 p +
yt(0)2 1−pfor p ∈ (0, 1), we have for any group G ∈ G:

$$\mathrm{RegVar}_{T}(\mathcal{A};G)=\sum_{t=1}^{T}\mathbbm{1}\left[x_{t}\in G\right]\left(f_{t}(p_{t,\mathrm{eff}})-f_{t}(p_{T,G}^{*})\right)$$ $$=\sum_{t=1}^{T}\mathbbm{1}\left[x_{t}\in G\right]\left(f_{t}\left(\sum_{G^{\prime}\in\mathcal{G}_{t}}w_{t,G^{\prime}}\cdot p_{t,G^{\prime}}\right)-f_{t}(p_{T,G}^{*})\right)$$ $$\leq\sum_{t=1}^{T}\mathbbm{1}\left[x_{t}\in G\right]\left(\sum_{G^{\prime}\in\mathcal{G}_{t}}w_{t,G^{\prime}}\cdot f_{t}\left(p_{t,G^{\prime}}\right)-f_{t}(p_{T,G}^{*})\right)$$