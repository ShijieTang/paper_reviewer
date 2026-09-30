# R**: Optimal Sub-Gaussianity With** Outlier Robustness And Low Moments Performance

Jasper C.H. Lee 1 **Walter McKelvie** 2 3 4 Maoyuan Song 2 **Paul Valiant** 2

## Abstract

We consider the basic statistical challenge of designing an "all-purpose" mean estimation algorithm that is recommendable across a variety of settings and models. Recent work by Lee & Valiant (2022) introduced the first 1-d mean estimator whose error in the standard finitevariance+i.i.d. setting is optimal even in its constant factors; experimental demonstration of its good performance was shown by Gobet et al. (2022). Yet, unlike for classic (but not necessarily practical) estimators such as median-of-means and trimmed mean, this new algorithm lacked proven robustness guarantees in other settings, including the settings of adversarial data corruption and heavy-tailed distributions with infinite variance. Such robustness is important for practical use cases. This raises a research question: is it possible to have a mean estimator that is robust, without sacrificing provably optimal performance in the standard i.i.d. setting? In this work, we show that Lee and Valiant's estimator is in fact an
"all-purpose" mean estimator by proving:
(A) It is robust to an η-fraction of data corruption, even in the strong contamination model; it has optimal estimation error O(σ
√η) for distributions with variance σ 2.

(B) For distributions with finite z th moment, for z ∈ (1, 2), it has optimal estimation error, matching the lower bounds of Devroye et al.

(2016) up to constants.

We further show (C) that outlier robustness for 1-d mean estimators in fact implies neighborhood optimality, a notion of beyond worst-case and distribution-dependent optimality recently introduced by Dang et al. (2023). Previously, such an 1University of California, Davis 2Purdue University 3Columbia University 4Harvard University. Correspondence to: Jasper C.H. Lee <jasperlee@ucdavis.edu>.

optimality guarantee was only known for medianof-means, but now it holds also for all estimators that are simultaneously *robust* and *sub-Gaussian*, including Lee and Valiant's, resolving a question raised by Dang et al. Lastly, we show (D) the asymptotic normality and efficiency of Lee and Valiant's estimator, as further evidence for its performance across many settings.

## 1. Introduction

The past decade has seen reinvigorated interest in understanding fundamental statistical problems, and in particular, the foundational problem of mean estimation: given n samples from an unknown distribution, and an allowed failure probability δ, what is the most accurate estimate of the distribution mean? The sample mean is the traditional estimate, yet, its sensitivity to extreme values makes it an unreliable estimator in practice. Other classic estimators, for example the well-known median-of-means estimator (Alon et al., 1999; Nemirovsky & Yudin, 1983; Jerrum et al., 1986), also suffer from poor empirical performance. Recent work by Lee & Valiant (2022), sitting in a line of research initiated by the seminal paper of Catoni (2012), gave the first 1-dimensional mean estimator whose estimation error is optimal (i.e. sub-Gaussian error) and tight even in the leading constant. Gobet et al. (2022) further demonstrated its good empirical performance against other competing estimators. Fact 1.1 (Lee & Valiant 2022). Given any distribution D with mean µ *and variance* σ 2*(both unknown), parameters* n, δ > 0, let X be a set of n *independent samples from* D. Then, with probability at least 1 − δ over the sampling process, Estimator 1 on input δ and X *will output an estimate* µˆ *with error at most*

$$|{\hat{\mu}}-\mu|\leq\sigma\cdot\left(({\sqrt{2}}+o(1)){\sqrt{\frac{\log{\frac{1}{\delta}}}{n}}}\right)$$

Here, the o(1) *term tends to* 0 as log 1 δ n, δ→ (0, 0) and, crucially, is independent of D.

1 Estimator 1 Mean Estimator of Lee and Valiant Inputs: n i.i.d. samples {xi}; Confidence parameter δ.

1. Compute an initial estimate κ using any sub-Gaussian, scale-and-translation-equivariant and robust mean estimator (cf. Facts A.18 and A.23). One possible choice is the median-of-means estimate: evenly partition the data into log 1δ groups and let κ be the median of the set of means of the groups; 2. Find the solution α to the monotonic, piecewise-linear equation

$$\sum_{i}\operatorname*{min}(\alpha(x_{i}-\kappa)^{2},1)={\frac{1}{3}}\mathrm{log}\,{\frac{1}{\delta}}$$

3. Output:
µˆ = κ +
1 n Pi
(xi − κ)(1 − min(α(xi − κ)
2, 1))
The constant "√2" in Fact 1.1 is optimal, with matching lower bounds. Essentially, Estimator 1 uses Step 1 to find a preliminary estimate (e.g. using median-of-means, but it can be other choices as well), which is then refined in Steps 2 and 3 into a better estimate—whose accuracy is optimal even in the leading multiplicative constant. Even though the form of the estimator is somewhat complicated—including a function inversion in Step 2—the estimator can in fact be computed easily in quasilinear time via a sorting operation, or even in linear time using a more subtle algorithm.

See (Lee & Valiant, 2022) for more discussion of the design and analysis of Estimator 1. While Fact 1.1 is a strong recommendation for Estimator 1, getting optimal performance in this regime is not the only practical desideratum for an estimator. Practitioners have long observed that some real-world distributions are very heavy-tailed, with variances that are gigantic (thus rendering guarantees like Fact 1.1 effectively useless), but whose lower moments—between the first and second momentsmight be much more bounded. Further, real-world data can be subject to random or even adversarial corruption. Performance guarantees in these more demanding settings are therefore critical for practice. However, while there are relatively straightforward folklore analyses of classic algorithms in these settings—including, notably, median-of-means and trimmed mean—Lee and Valiant's estimator has not been studied in these contexts. The breadth of guarantees shown for these classic estimators might therefore lead researchers and practitioners to choose either of them over Lee and Valiant's estimator, despite the latter being a more accurate estimator in standard finite variance i.i.d. sample settings.

The natural and pressing research question, then, is Is it possible for a mean estimator to satisfy the important robustness guarantees of classic estimators, **without** *losing* the strong and optimal-constant performance that the Lee and Valiant estimator enjoys in the standard setting? In this paper, we show 4 strong positive results demonstrating the robustness and performance of Estimator 1: (A) In the presence of η-fraction of arbitrary adversarial corruption, the error of Estimator 1 degrades by only O(σ
√η)
from the sub-Gaussian error guaranteed in Fact 1.1. This extra error term is optimal under the assumption that the distribution has (finite) variance σ 2.

(B) When samples are drawn from a distribution with finite z th moment, for z ∈ (1, 2), Estimator 1 achieves error matching the lower bound of Devroye et al. (2016) up to constants. (C) Estimator 1 enjoys a fine-grained beyond-worst-case notion of optimality called *neighborhood optimality*, recently introduced by Dang et al. (2023). Previously, only medianof-means was known to be neighborhood optimal, and our result answers the question raised by Dang et al. on whether more modern estimators are also neighborhood optimal. (D) Under the standard finite variance and i.i.d. setting, Lee and Valiant's estimator is asymptotically normal and efficient. We emphasize that these new results hold without any changes at all to either the structure or the parameters of Estimator 1. That is, the estimator does not need to know that it is being expected to perform in these challenging settings. The guarantees we show also smoothly revert to Fact 1.1 as the amount of corruption decreases to 0, or as the z th-moment assumption tends to z → 2.

These results, in aggregate, demonstrate that Lee and Valiant's estimator is in fact "all-purpose", enjoying the same breadth of guarantees as either median-of-means or trimmed mean, while also having what is essentially the smallest possible estimation error (even in the leading constant) in the standard setting. Thus, Lee and Valiant's estimator should be used in practice over existing estimators.

## 2. Our Results At A Glance

We now formally state our technical contributions.

## 2.1. Outlier Robustness

Lee and Valiant's estimator is robust against data corruption from the *strong contamination model*, the most adversarial data corruption model in the literature. Definition 2.1 (Strong Contamination Model). Given a corruption parameter η and a distribution D on the uncorrupted data, an algorithm gets a set of n η-corrupted samples from D as follows. The algorithm specifies n and nature draws n i.i.d. samples from D. Then, an arbitrarily powerful adversary can inspect the n samples from D, and arbitrarily replace ⌈ηn⌉ of them before giving the (new) set of samples to the algorithm, with no indication of which samples were corrupted. In the setting of η-corruption, when we only assume that D has a finite variance σ 2, it is well-known (Diakonikolas &
Kane, 2023) that the minimum estimation error is Ω(σ
√η)
even if there are infinitely many samples. Our analysis of Lee and Valiant's estimator shows that it achieves this estimation error bound, as long as the amount of corruption is O(log 1δ
/n)—where δ, the desired failure probability, is a parameter under the user's control. Put differently, the δ parameter for Estimator 1 can be viewed as "dual use", parameterizing not just the allowed failure probability of the estimator, but also its desired robustness, expressing the maximum tolerated level of corruption. We emphasize that in the following theorem, Estimator 1 does not need to know the precise value of η, the fraction of corruption. It only makes the assumption that η ≤ O(log 1δ
/n), or in other words, Estimator 1 *adapts* to the level of corruption. Theorem 2.2. Given any distribution D with mean µ and variance σ 2, parameters n, δ, η > 0, let X˜ *be a set of* n η*-corrupted samples from* D. Suppose both log 1δ nand δ are bounded by some small universal constant, and suppose η ≤1 24n log 1δ
. Then, with probability at least 1 − δ over the sampling process, Estimator 1 on input δ and X˜ will output an estimate µˆ with error at most

$$|{\hat{\mu}}-\mu|\leq\sigma\cdot\left(({\sqrt{2}}+o(1)){\sqrt{\frac{\log{\frac{1}{\delta}}}{n}}}+222{\sqrt{\eta}}\right)$$

In the above theorem, Estimator 1 gets δ as its input parameter, and fails with probability δ. Accordingly, the estimator error is the sum of the sub-Gaussian error term from Fact 1.1, and a new "robustness" term of O(σ
√η). On the other hand, the standard robustness analysis and guarantee of trimmed mean takes a different form (see Oliveira et al. (2025))—if the desired failure probability is δ
′and the corruption parameter is η, then the number of trimmed samples is chosen as log 1δ = Θ(log 1 δ
′ + ηn). We give an analogous theorem for Estimator 1, where again we preserve the optimal constant of 
√2 in the sub-Gaussian error term. Comparing Theorem 2.2 with Theorem 2.3, the latter asks for a failure probability δ
′ > δ, but correspondingly has a smaller sub-Gaussian error term (since log 1 δ
′ < log 1δ
).

Theorem 2.3. Given any distribution D with mean µ and variance σ 2, parameters n, δ, η > 0, let X˜ be a set of n η-corrupted samples from D.

Suppose both log 1δ nand δ are bounded by some small universal constant, and suppose η ≤1 9n log 1δ
. Let δ
′ *be so that* 1 3 log 1δ =
1 3 log 1 δ
′ + 3ηn*. Then, with probability at least* 1 − δ
′ over the sampling process, Estimator 1 on input δ and X˜ will output an estimate µˆ *with error at most*

$$|\hat{\mu}-\mu|\leq\sigma\cdot\left((\sqrt{2}+o(1))\sqrt{\frac{\log\frac{1}{\delta^{\prime}}}{n}}+(135+o(1))\sqrt{\eta}\right)$$

We emphasize again that, similar to Theorem 2.2, Theorem 2.3 says that Estimator 1 adapts to the value of η without knowing what it is precisely. Namely, Theorem 2.3 simultaneously holds for all values of δ
′even though the algorithm is only given a fixed δ. We sketch the proof of the above two theorems in Section 4 and give full details in Appendix A. While the 
√2 constant in the sub-Gaussian terms is tight, we expect that the constants in the robust error terms (222 and 135) might be significantly improved with a different proof strategy. In Appendix B, we also show that Estimator 1 is robust in the Huber and Total Variation Contamination Models, in addition to the Strong Contamination Model above. Median-of-means also has guarantees analogous to the above, but they are folklore. See the proofs of Theorem 2.2 and Theorem 2.3 in Appendix A.4 and Appendix A.3 respectively for the formal statements and proofs of these folklore guarantees. Trimmed mean also gives analogous robustness guarantees, but current analysis (see Oliveira et al. (2025)) requires precise knowledge of η in order to match the optimal rate. In contrast, although Estimator 1 does require an upper bound on η (from the necessary assumption that η ≤1 24n log 1δ
), the accuracy will gracefully improve if the actual corruption rate is less than the pessimistic upper bound.

We further stress that, while median-of-means and trimmed mean both have relatively straightforward and folklore proofs of robustness, the analysis required for the Lee and Valiant estimator is much more intricate—this is due to the fact that Lee and Valiant's estimator is (a lot) more complicated, in order to yield optimal constants in the i.i.d. setting.

## 2.2. Low Moment Performance

Next, we study the performance of Lee and Valiant's estimator when given data drawn from a distribution that only has finite low moments, specifically, moments between 1 and 2.

Theorem 2.4. Given any distribution D with mean µ and z th moment Mz for some z ∈ (1, 2), let X be a set of n i.i.d. samples from D*. Then, with probability at least* 1 − δ over the randomness of X, Estimator 1 on input δ and X
will output an estimate µˆ *with error at most*

$$|{\hat{\mu}}-\mu|\leq(M_{z})^{\frac{1}{z}}\cdot(1+o(1))\left(c_{z}{\frac{\log{\frac{1}{\delta}}}{n}}\right)^{1-{\frac{1}{z}}}$$

where cz = 2(5.6) 
1 z−1 −1*. Here, the* o(1) *term tends to* 0 as log 1 δ n, δ→ (0, 0), in a manner independent of D and independent of z. The above result matches the lower bounds shown by Devroye et al. (2016) up to constants. As z → 2, the guarantee converges to Fact 1.1. Analogous results (up to constants) were shown for median-of-means in Bubeck et al. (2012), but the multiplicative constant we achieve for Estimator 1, (2(5.6) 
1 z−1 −1) is better than that of median-ofmeans (8
√3(12) 1 z−1 −1) across all values of z ∈ (1, 2].

## 2.3. Neighborhood Optimality

Going beyond the worst-case analysis of Estimator 1, we give a finer-grained analysis of its performance. We provide finite-sample error bounds that *optimally adapt* to the underlying distribution on an instance-by-instance basis, which is far stronger than just having optimal dependence on the variance. Recent work by Dang et al. (2023) gave a first study of the fine-grained optimality of 1-dimensional mean estimators, providing upper and lower bounds that match up to constants. At a high level, they showed that sub-Gaussian error bounds are essentially all one can hope for, for any distribution with a finite mean. More specifically, they define an error rate function ϵn,δ(D) over distributions D,
and prove that i) median-of-means attains this error bound; and ii) for any distribution D, there exists a "reasonable" counterpart distribution D′such that no algorithm can distinguish between the distributions, and thus no estimator can simultaneously get error ≪ ϵn,δ(D) using samples from D,
while also getting error ≪ ϵn,δ(D′) using samples from D′.

We define ϵn,δ(D) below. The combination of lower and upper bounds of i) and ii) is extended into a new optimality notion called *neighborhood optimality* by Dang et al. Intuitively, these are bounds that are optimal in an instanceby-instance basis, because property (ii) shows that no algorithm can get error better than ϵn,δ(D) on samples from distribution D—even an algorithm *customized* specifically for distribution D—without getting unacceptably bad error on a designated nearby distribution D′.

For concreteness, we define ϵn,δ(D) below. Definition 2.5 (Dang et al. 2023). Given a (continuous) distribution D with mean µ and a real number t ∈ [0, 1], define the t*-trimming* operation on D as follows: select a radius r such that the probability mass in [µ − *r, µ* + r]
equals 1 − t; then, return the distribution D **conditioned** on lying in [µ − *r, µ* + r].

Given n and δ, define the trimmed distribution D∗n,δ to be the 0.45 n log 1δ
-trimmed version of D. When δ is implicit, we may denote this as D∗n. Now define the error function ϵn,δ(D) = |µ − µ
∗n| + σ
∗n qlog 1δ n, where µ
∗nand σ
∗nare the mean and standard deviation of D∗nrespectively.

Dang et al. show that median-of-means achieves error O(ϵn,δ(D)), and raises the question of whether more modern estimators such as Lee and Valiant's (Estimator 1) also achieve this error bound and are hence neighborhood optimal. We show a more general result: in fact, every sub-
Gaussian and robust estimator (satisfying a slight variant of Theorem 2.2) achieves this error bound.

Proposition 2.6. Let µˆ be an arbitrary estimator that, when given δ > 0 and a set of n η*-corrupted samples from any* distribution D with mean µ *and variance* σ 2, outputs a mean estimate satisfying

$$|{\hat{\mu}}-\mu|\leq O\left(\sigma\left({\sqrt{\frac{\log{\frac{1}{\delta}}}{n}}}+{\sqrt{\eta}}\right)\right)$$

with probability at least 1 −
δ 2 over the randomness of the
(uncorrupted) samples. Then, the same estimator µˆ, on input n *i.i.d. samples drawn* from a distribution D with finite mean, will output an estimate with error upper bounded by O(ϵn,δ(D)) (as defined in Definition *2.5) with probability at least* 1 − δ. We formally prove Proposition 2.6 in Appendix C. The precondition of Proposition 2.6 is implied by a mild variant of Theorem 2.2 (decreasing the failure probability from δ to δ/2), which holds also for Estimator 1. As a consequence:
Corollary 2.7 (Informal). Estimator 1 *is neighborhood* optimal, in the sense of Dang et al. *(2023).* In Section 5, we state the formal definition of neighborhood optimality and discuss the intuition on Proposition 2.6.

## 2.4. Asymptotic Normality And Efficiency

Lastly, we show that Lee and Valiant's estimator is asymptotically normal and efficient, under the standard finite variance and i.i.d. sample assumption. In particular, we show that, if the δ parameter in Estimator 1 is fixed, and the number of samples n → ∞, then Estimator 1 converges in probability to the sample mean at the appropriate scale. The Central Limit Theorem for the sample mean then implies the asymptotic optimality of Lee and Valiant's estimator as a corollary. Theorem 2.8. Let D be a distribution with mean µ and variance σ 2.

Let µˆ denote Estimator 1 on input parameter δ and n i.i.d. samples from D. Also let X¯n denote the sample mean.

Then, fixing δ and D and taking n → ∞*, we have*

$${\sqrt{n}}{\hat{\mu}}\stackrel{p}{\rightarrow}{\sqrt{n}}{\bar{X}}_{n}$$

that is, |
√nµˆ−
√nX¯n| p→ 0*, that* 
√nµˆ *converges to* √nX¯n in probability. As a corollary, by the Central Limit Theorem, we have

$\mathbf{a}$
$${\sqrt{n}}({\hat{\mu}}-\mu)\stackrel{d}{\to}{\mathcal{N}}\left(0,\sigma^{2}\right)$$
That is, µˆ *is asymptotically normal and efficient.* The above theorem contrasts with the asymptotic behavior of median-of-means, whose error—scaled by 
√n—converges to N (0,(π/2)σ 2) (Minsker, 2023); median-of-means thus has asymptotic variance a π/2-factor larger than desired.

## 3. Related Work

Mean estimation in 1 dimension. Mean estimation, even in 1-dimension, has been studied algorithmically since the 1980s. The classic median-of-means estimator was the first big-O optimal sub-Gaussian mean estimator proposed in the literature, independently invented by different groups of authors (Alon et al., 1999; Nemirovsky & Yudin, 1983; Jerrum et al., 1986). Catoni's influential work (2012) gave the first sub-Gaussian mean estimator that yields the tight multiplicative constant in its error, but under strong assumptions that either the variance is known (to extremely high accuracy) or the distribution kurtosis (normalized 4 th moment) is bounded. Followup work by Devroye et al. (2016) studied
"multiple-δ" estimators, also with sharp error constants, in the same setting. More recently, Lee & Valiant (2022) constructed a sub-Gaussian mean estimator with tight constants, under the bare minimum assumption that the variance exists, and absent any extra knowledge or moment assumptionsthis estimator is the subject of study in the current work. See the survey of Lugosi & Mendelson (2021) on mean estimation results prior to 2019. In low moment settings where the underlying distribution might have infinite variance, Bubeck et al. (2012) studied the performance of median-of-means. Devroye et al. (2016) then showed lower bounds that match up to constants. Our work shows that Lee and Valiant's estimator achieves analogous results as median-of-means in these regimes (Theorem 2.4), with sharper dependence on the z th moment, for every z ∈ (1, 2].

"Beyond worst-case analysis" of 1-d mean estimators is a new research topic of recent interest in the community. In the standard i.i.d. setting, Dang et al. (2023) characterized the optimal distribution-specific error rates up to constants, showing that median-of-means achieve such rates. Our work shows that in fact all estimators that simultaneously achieve (big-O) optimal sub-Gaussian and robust estimation error must also achieve the distribution-specific optimal error rates (Proposition 2.6). In addition to the standard i.i.d. setting, a different line of work has also studied distributionspecific mean estimation error rates for various differential privacy settings (Asi & Duchi, 2020a;b; Huang et al., 2021). Robust mean estimation. Robust statistics, the setting where part of the input data can be corrupted by an adversary, has been an active area of research in the statistics community since the 1960s (Huber, 1992; Tukey, 1960). However, it was only in the past decade that polynomial-time algorithms for these statistical problems were found. See the textbook of Diakonikolas & Kane (2023) for a detailed introduction to these recent advances. Most directly relevant to our present work are results that give simultaneously sub-Gaussian and robust mean estimators, even in arbitrary high dimensions (Diakonikolas et al., 2020; Depersin & Lecue´, 2022; Hopkins et al., 2020). Median-of-means is also known to be such a robust and sub-Gaussian estimator in 1-dimension - this is a folklore result, but see Laforgue et al. (2021) for more details on the analysis in the robust setting. Similarly, trimmed mean also has a folklore analysis for its robustness, and see Oliveira et al. (2025) for more results on the robustness and additional properties of trimmed mean.

## 4. Outlier Robustness

In this section, we outline the proof of Theorem 2.2 (restated below for the reader's convenience), which says that Estimator 1 is robust against adversarial data contamination from the strong contamination model of Definition 2.1. The proof of Theorem 2.3 has analogous structure. Theorem 2.2. Given any distribution D with mean µ and variance σ 2, parameters n, δ, η > 0, let X˜ be a set of n η*-corrupted samples from* D.

Suppose both log 1δ nand δ are bounded by some small universal constant, and suppose η ≤1 24n log 1δ
. Then, with probability at least 1 − δ over the sampling process, Estimator 1 on input δ and X˜ will output an estimate µˆ *with error* at most

$$|{\hat{\mu}}-\mu|\leq\sigma\cdot\left(({\sqrt{2}}+o(1)){\sqrt{\frac{\log{\frac{1}{\delta}}}{n}}}+222{\sqrt{\eta}}\right)$$

The proof strategy is to bound the difference between the estimates returned by Estimator 1 on (corrupted) samples X˜ versus its behavior on *uncorrupted* samples X, fixing the confidence parameter δ.

Changing the input from uncorrupted samples to corrupted samples has two effects on the resulting estimate: 1. The α "influence parameter" (as computed in Step 2 of Estimator 1) may change. However, we show that in a certain sense, when the fraction of corruption η is small compared to log 1δ
/n, this corruption will not change the computed α value by much (Lemma 4.2). We further show that artificially changing the α value by a small amount will not change the mean estimate of Step 3 by much, with high probability, when given *uncorrupted* samples. 2. For a fixed influence parameter α, corrupting the samples from X to X˜ changes the returned mean estimate. However, we show a (high probability) *lower bound* on the value α computed by the algorithm on input X˜ (Lemma 4.3); and this lower bound on α gives us a natural *upper bound* on how much any corrupted input value can affect the final mean estimate.

We state here the two key structural lemmas (and a corresponding prerequisite definition) for the α value computed from corrupted samples X˜.

Definition 4.1. Let X = {xi} be a set of clean samples, and let X˜ = {x˜i} be the corresponding set of η-corrupted samples. Denote by αρ the "influence parameter" computed from the clean samples so as to satisfy a version of Step 2 but with a modified right hand side ( 13 log 1δ + ρn instead of 1 3 log 1δ
):

$$\sum_{i}\operatorname*{min}(\alpha_{\rho}x_{i}^{2},1)={\frac{1}{3}}\mathrm{log}\,{\frac{1}{\delta}}+\rho n\,$$

and denote by α˜ρ the corresponding "influence parameter" computed instead from the *corrupted* samples:

$$\sum_{i}\operatorname*{min}(\bar{\alpha}_{\rho}\bar{x}_{i}^{2},1)=\frac{1}{3}\mathrm{log}\,\frac{1}{\delta}+\rho n$$

Lemma 4.2. Consider an arbitrary set of samples X and a new sample set X η ˜ -corrupted from X. Consider also an arbitrary input parameter δ. Using α˜ to denote the influence parameter of Estimator 1 on inputs (δ, X˜)*, i.e.* α˜0 in Definition *4.1, we have*

$$\alpha_{-2\eta}\leq\bar{\alpha}\leq\alpha_{2\eta}$$

Considering the right hand side of the condition in Step 2 of Estimator 1 as expressing the level of "desired robustness": Lemma 4.2 states that the modified influence parameter from η-corruption is always sandwiched between the uncorrupted influence parameters, but at slightly different levels of desired robustness. We point out that Lemma 4.2 is a deterministic lemma, that always holds, regardless of the sampling over X.

Lemma 4.3. In the setting of Lemma *4.2, suppose both* log 1δ n and δ *are bounded by some small universal constant, and* suppose η ≤1 24n log 1δ
. With probability at least 1 − 4δ/11 over the sampling of n samples X *from a distribution* D
with variance σ 2*, we have* α˜ ≥ 0.0008496η.

These lemmas let us bound α˜ even when given corrupted data, and relate it to the uncorrupted α; these bounds are the crucial tools needed to bound the mean estimation error in both Theorem 2.2 and Theorem 2.3. Lemma 4.3 is used in the proof of Theorem 2.2, and shown inside Proposition A.22 in Appendix A.4. The proof of Theorem 2.3 has an analogous lemma with slightly different parameters. The full analysis of the outlier robustness of Estimator 1 is given in Appendix A.

## 5. Neighborhood Optimality

Neighborhood optimality is a new notion of fine-grained distribution-dependent optimality recently proposed by Dang et al. (2023). While sub-Gaussian bounds are worstcase optimal for the class of finite variance distributions, neighborhood optimality captures the extent to which estimators can beneficially adapt to the non-Gaussianity of the underlying distribution and outperform the sub-Gaussian bound. Before we formally state the definition of neighborhood optimality, let us give some preliminary definitions.

Let P1 be the entire set of all distributions with a finite first moment over R. We say that N is a neighborhood function
(defined over P1) if N maps a distribution D ∈ P1 to a set of distributions N(D) ⊆ P1. Intuitively, the neighborhood N(D) of D is a set of distributions that we expect an estimator to perform similarly well on (and we typically consider neighborhoods where D ∈ N(D) ). Similarly, an error function ϵ maps distributions to non-negative numbers, like the function introduced in Definition 2.5. In the later definitions, we use the notations Nn,δ and ϵn,δ to indicate their dependence on the sample complexity n and failure probability δ.

Given these two notions, we can now define the notion of a neighborhood Pareto bound, as a property that an error function satisfies. Essentially, the definition imposes admissibility/Pareto efficiency structure within the local neighborhood Nn,δ(D) of every distribution D ∈ P1.

Definition 5.1 (Neighborhood Pareto Bounds (Dang et al.,
2023)). Let n be the number of samples and δ be the failure probability. Given a neighborhood function Nn,δ : P1 →
2 P1, we say that the error function ϵn,δ(D) : P1 → R
+
0is a neighborhood Pareto bound for P1 with respect to Nn,δ if for all distributions D ∈ P1, no estimator µˆ taking n i.i.d. samples can simultaneously achieve the following two conditions:
- For all D′ ∈ Nn,δ(D), with probability 1 − δ over the n i.i.d. samples from D′, we have |µˆ − µD′ | ≤ ϵn,δ(D′).

- With probability 1 − δ over the n i.i.d. samples from D,
|µˆ − µD| < ϵn,δ(D).

Neighborhood Pareto ounds essentially play the role of
"lower bounds" in an optimality definition, and the strength of the result depends crucially on the choice of the neighborhood function N under consideration. As a basic observation, the strength of this definition is *monotonic* in the size of the neighborhoods returned by N: if an error function ϵ is a neighborhood Pareto bound for a neighborhood function N, then for any neighborhood function N′ such that N(D) ⊆ N′(D) for every D ∈ P1, ϵ is also a neighborhood Pareto bound for N′. Thus, the smaller each neighborhood is, the stronger the neighborhood Pareto bound is.

Finally, we define neighborhood optimality.

Definition 5.2 ((*κ, τ* )-Neighborhood Optimal Estimators (Dang et al., 2023)). Let κ > 1 be a multiplicative loss factor in estimation error, and τ > 1 be a multiplicative loss factor in sample complexity.

Given the parameters *κ, τ >* 1, sample complexity n, failure probability δ and neighborhood function Nn,δ, a mean estimator µˆ is (*κ, τ* )-neighborhood optimal with respect to Nn,δ if there exists an error function ϵn,δ(D) such that min(ϵn/τ,δ(D), ϵn,δ(D)) is a neighborhood Pareto bound1, and µˆ gives estimation error at most κ · ϵn,δ(D) with probability at least 1 − δ when taking n i.i.d. samples from any distribution D ∈ P1. Dang et al. (2023) showed that the error function ϵn,δ from Definition 2.5 yields a neighborhood Pareto bound 1 κ min(ϵn/τ,δ(D), ϵn,δ(D)) for an appropriate choice of neighborhood function, for some constants *κ, τ >* 1. Their choice of neighborhood function Nn,δ is technical; we state it in Appendix C, and refer the reader to their paper for the justification. Based on this result, they also showed that median-of-means indeed achieves error O(ϵn,δ) from Definition 2.5 and hence is neighborhood optimal by Definition 5.2. Dang et al. (2023) further raised the immediate question of whether other more-modern estimators, such as Lee and Valiant's Estimator 1, can also achieve such estimation error.

and optimally robust to corruption must achieve the error rate from Definition 2.5 (stated formally as Proposition 2.6), and are thus neighborhood optimal as a corollary. Proposition 2.6. Let µˆ *be an arbitrary estimator that, when* given δ > 0 and a set of n η-corrupted samples from any distribution D with mean µ *and variance* σ 2, outputs a mean estimate satisfying

$$|{\hat{\mu}}-\mu|\leq O\left(\sigma\left({\sqrt{\frac{\log{\frac{1}{\delta}}}{n}}}+{\sqrt{\eta}}\right)\right)$$

with probability at least 1 −
δ 2 over the randomness of the
(uncorrupted) samples. Then, the same estimator µˆ, on input n i.i.d. samples drawn from a distribution D with finite mean, will output an estimate with error upper bounded by O(ϵn,δ(D)) *(as defined* in Definition *2.5) with probability at least* 1 − δ. The precondition of Proposition 2.6 is satisfied by a variant of Theorem 2.2—with slightly smaller failure probabilitywhich holds for Estimator 1. Thus the neighborhood optimality of Estimator 1 follows as a corollary of Proposition 2.6. To see the intuition behind Proposition 2.6, recall the definition of the error rate function ϵn,δ(D) from Definition 2.5.

The definition constructs a distribution D∗n,δ from D, by removing the tails of D with probability mass O(log 1δ
/n),
and the error function ϵn,δ(D) = σ
∗n qlog 1δ n + |µ − µ
∗n| is the sub-Gaussian error for distribution D∗n,δ plus the mean difference between D and D∗n,δ. Thus, when given samples from D, one could view them as *corrupted* samples from D∗n,δ where roughly O(log 1 δ
/n) fraction of the samples are corrupted. A sub-Gaussian and robust estimator would thus achieve good error with respect to the mean of D∗n,δ, and by triangle inequality, also with respect to the mean of D. We present proofs of the above statements in Appendix C. For completeness, we also provide a summary of Dang et al. (2023)'s results. We again refer the reader to their paper for a more in-depth discussion on the intricacies of the neighborhood optimality notion.

## 6. Infinite Variance Distributions

In this section, we extend Lee and Valiant's analysis of Estimator 1 to more heavy-tailed distributions. Instead of Fact 1.1, where the performance of the estimator is characterized in terms of the *variance* of the distribution D, we instead ask if we can characterize the performance of the estimator on distributions that may not have a finite variance but instead only have finite z th moment for some 1 < z ≤ 2.

We restate our main theorem for this section, which matches the lower bound of Devroye et al. (2016) up to constants.

as follows:
Theorem 2.4. Given any distribution D with mean µ and z th moment Mz for some z ∈ (1, 2), let X *be a set of* n i.i.d. samples from D*. Then, with probability at least* 1 − δ over the randomness of X, Estimator 1 on input δ and X
will output an estimate µˆ *with error at most*

$$|{\hat{\mu}}-\mu|\leq(M_{z})^{\frac{1}{2}}\cdot(1+o(1))\left(c_{z}{\frac{\log{\frac{1}{\delta}}}{n}}\right)^{1-{\frac{1}{2}}}$$

where cz = 2(5.6) 
$=2(5.6)^{\frac{1}{z-1}-1}.\;\;Here,\;the\;o(1)\;term\;tends\;to\;0$  . 
as log 
1 δ
n, δ→ (0, 0), in a manner independent of D and
independent of z. At a high level, our analysis is a generalization of Lee and Valiant's analysis to the low-moment setting, which allows us to prove a guarantee that gracefully reduces to their main result (Fact 1.1, with the sharp constant of 
√2)
as z → 2. Furthermore, our value of cz is smaller than the corresponding multiplicative constant in the analysis of median-of-means by Bubeck et al. (2012), across all values of z ∈ (1, 2]. Here we give an overview of our analysis. Without loss of generality, from the shift-and-scale equivariance of Estimator 1, we assume the underlying distribution has mean 0 and z th moment Mz = 1. The goal is to prove tailored Chernoff bounds for this estimator to show its concentration. Lee and Valiant's analysis in the finite variance setting provides two useful techniques to address obstacles described in the following subsections.

## 6.1. Estimator 1 **Is A Sum Of Dependent Terms**

Estimator 1 is a sum of *dependent* terms, due to the influence parameter α computed in Step 2 involving all the samples. This makes proving Chernoff bounds tricky, given that moment generating functions multiply only for sums of independent terms. Lee and Valiant's approach is to *reduce* (via a Lipschitz argument) to analyzing the case where the preliminary estimate κ from Step 1 is taken to be exactly equal to the true mean µ = 0, and crucially reformulate Estimator 1 as a "2-parameter ψ-estimator". Definition 6.1 (Lee & Valiant 2022). Consider Estimator 1 but with Step 1 replaced with "κ = 0". The estimator can be equivalently expressed as follows:
1. Input: Failure probability δ, independent samples X =
x1*, . . . , x*n 2. Solve for the (unique) pair (ˆµ, αˆ) satisfying ψµ = 0 and ψα = 0, where the functions ψµ, ψα are defined

$$\psi_{\mu}(X,{\hat{\mu}},{\hat{\alpha}})=\sum_{i=1}^{n}({\hat{\mu}}-x_{i}(1-\operatorname*{min}({\hat{\alpha}}x_{i}^{2},1)))\,;$$  $$\psi_{\alpha}(X,{\hat{\mu}},{\hat{\alpha}})=\sum_{i=1}^{n}\left(\operatorname*{min}({\hat{\alpha}}x_{i}^{2},1)-{\frac{1}{3n}}\mathrm{log}\,{\frac{1}{\delta}}\right)$$

3. Output: µˆ from the previous step.

This reformulation has the advantage that, for any fixed pair
(ˆµ, αˆ), any linear combination of the ψµ and ψα functions is a sum of independent terms. The concentration of Estimator 1 is then reduced to proving Chernoff bounds for these linear combinations. This reformulation and reduction is independent of the finite variance assumption, and therefore also applicable to the low moment setting that our work analyzes.

## 6.2. Proving Chernoff Bounds Over Large Distribution Classes

Even in the finite variance setting, proving a Chernoff bound that applies for all distributions D with mean 0 and variance 1 is daunting, given how large the distribution class is compared to standard concentration bounds. Lee and Valiant showed that the worst-case Chernoff bound (for linear combinations of the ψ-equations from Definition 6.1) can in fact be viewed as a max-min linear programming game. For simplicity, let us illustrate this by sketching the analysis of the Chernoff bound of a hypothetical linear estimator that is a sum of independent terms: f({x1*, . . . , x*n}) =
1 n Pi f(xi), for some fixed function f : R → R. Proving a Chernoff bound is equivalent to upper bounding the moment generating function of the estimator f, and choosing the "Chernoff parameter" t accordingly. Thus, it suffices to upper bound the objective of the following max-min game, where the max player chooses any mean-0 variance-1 distribution D—represented by variables {px} denoting the probability mass at x (ignoring probability formalism issues with non-discrete distributions)—and the min player chooses the Chernoff parameter t. Using the moment generating function as the objective function, we have

 $\begin{array}{ll}\max_{\{p_x\}}&\min_t\sum_x p_x e^{t\cdot f(x)}\\ \text{such that}&\sum_x p_x=1\\ &\sum_x p_x\cdot x=0\\ &\sum_x p_x\cdot x^2=1\\ \text{where}&p_x\geq0\end{array}$ (1)
By using minimax duality and linear programming duality, the game can then be rewritten into a pure minimization program with the same optimum, where the dual variables U, M, V correspond to the 3 respective constraints in the program in (1).

mint min*U,M,V* U + V
such that for all x ∈ R, V x2 + Mx + U ≥ e t·f(x)
(2)
It thus suffices to choose dual variables *U, M, V* and an appropriate Chernoff parameter t to certify an upper bound on the optimum. We modify this approach from Lee & Valiant (2022) by relying on a z th moment bound instead of a variance bound.

The key observation is that the z th moment bound may be expressed as a linear constraint in the above program, replacing Pxpxx 2 = 1 with Pxpx|x| z = 1. The technical challenge from here is to provide a feasible dual solution and choose Chernoff parameter t so as to satisfy the desired bounds, including that the guarantees converge to Fact 1.1 as z → 2. We show our complete proof in Appendix D.

## 7. Asymptotic Normality

In this section, we show that under the standard finite variance assumption, the estimator of Lee and Valiant is asymptotically *normal* and *efficient*. Specifically, we prove that if we fix the input parameter δ and take the number of samples n → ∞, the estimator converges to the sample mean in probability, which by the Central Limit Theorem implies asymptotic normality and efficiency. This result contrasts with median-of-means, which, under the slightly stronger 2+ι moment assumption for any ι > 0, is asymptotically normal yet *inefficient* (Minsker, 2023)— the asymptotic distribution of 
√nµˆMoM is N (µ,(π/2)σ 2)
instead of the desired N (*µ, σ*2).

Theorem 2.8. Let D be a distribution with mean µ and variance σ 2.

Let µˆ denote Estimator 1 on input parameter δ and n i.i.d. samples from D. Also let X¯n denote the sample mean.

Then, fixing δ and D and taking n → ∞*, we have*

$${\sqrt{n}}{\hat{\mu}}\stackrel{p}{\to}{\sqrt{n}}{\bar{X}}_{n}$$

that is, |
√nµˆ−
√nX¯n| p→ 0*, that* 
√nµˆ *converges to* 
√nX¯n in probability.

As a corollary, by the Central Limit Theorem, we have

$${\sqrt{n}}({\hat{\mu}}-\mu)\stackrel{d}{\rightarrow}{\mathcal{N}}\left(0,\sigma^{2}\right)$$

That is, µˆ *is asymptotically normal and efficient.*
The proof is relatively straightforward. The key idea is that Estimator 1 differs from the sample mean by removing Θ(log 1δ
) weighted samples, so we might as well bound the difference by Θ(log 1 δ
) times the maximum sample (and symmetrically the minimum sample), multiplied by a factor of 
√n since that is the scale that the Central Limit Theorem holds at. Under the finite variance assumption, we can use a (slightly refined) Chebyshev's inequality and a standard Chernoff bound to upper bound the magnitude of the maximum sample with high probability. See the complete calculations in Appendix E.

## Acknowledgments

The work of Jasper C.H. Lee was done in part while he was at UW Madison, supported by NSF Medium Award CCF- 2107079. He also thanks Stanislav Minsker for discussions on the asymptotic normality result. The work of Walter McKelvie was partly supported by the National Science Foundation Graduate Research Fellowship Program under Grant No. 2140743. Maoyuan Song and Paul Valiant are partially supported by NSF award CCF-2127806 and by Office of Naval Research award N000142412695.

Maoyuan Song is partially supported by NSF award CCF-
2228814.

## Impact Statement

This work studies a fundamental statistical problem that is broadly applicable to a wide variety of domains. As such, it does not directly raise any societal or ethical concerns that warrant special consideration.

## References

Alon, N., Matias, Y., and Szegedy, M. The space complexity of approximating the frequency moments. J. Comput. Syst. Sci, 58(1):137–147, 1999.

Asi, H. and Duchi, J. C. Near instance-optimality in differential privacy. *arXiv preprint arXiv:2005.10630*, 2020a.

Asi, H. and Duchi, J. C. Instance-optimality in differential privacy via approximate inverse sensitivity mechanisms. Advances in neural information processing systems, 33:
14106–14117, 2020b.

Bubeck, S., Cesa-Bianchi, N., and Lugosi, G. Bandits with heavy tail. *arXiv preprint arXiv:1209.1727*, 2012.

Catoni, O. Challenging the empirical mean and empirical variance: a deviation study. *Ann. I. H. Poincare -PR* ´ , 48 (4):1148–1185, 2012.

Dang, T., Lee, J. C. H., Song, M., and Valiant, P. Optimality in mean estimation: Beyond worst-case, beyond subgaussian, and beyond 1 + α moments. In *Thirty-seventh* Conference on Neural Information Processing Systems, 2023.

Depersin, J. and Lecue, G. Robust sub-gaussian estimation ´
of a mean vector in nearly linear time. The Annals of Statistics, 50(1):511–536, 2022.

Devroye, L., Lerasle, M., Lugosi, G., and Oliveira, R. I. Sub-
Gaussian mean estimators. *Ann. Stat*, 44(6):2695–2725, 2016.

Diakonikolas, I. and Kane, D. M. Algorithmic highdimensional robust statistics. Cambridge university press, 2023.

Diakonikolas, I., Kane, D. M., and Pensia, A. Outlier robust mean estimation with subgaussian rates via stability. Advances in Neural Information Processing Systems, 33: 1830–1840, 2020.

Gobet, E., Lerasle, M., and Metivier, D. Mean estimation ´
for randomized quasi monte carlo method. *Hal preprint* hal-03631879v2, 2022.

Hopkins, S., Li, J., and Zhang, F. Robust and heavy-tailed mean estimation made simple, via regret minimization. Advances in Neural Information Processing Systems, 33: 11902–11912, 2020.

Huang, Z., Liang, Y., and Yi, K. Instance-optimal mean estimation under differential privacy. *Advances in Neural* Information Processing Systems, 34:25993–26004, 2021.

Huber, P. J. Robust estimation of a location parameter. In Breakthroughs in statistics: Methodology and distribution, pp. 492–518. Springer, 1992.

Jerrum, M. R., Valiant, L. G., and Vazirani, V. V. Random generation of combinatorial structures from a uniform distribution. *Theor. Comput. Sci*, 43:169–188, 1986.

Laforgue, P., Staerman, G., and Clemen ´ c¸on, S. Generalization bounds in the presence of outliers: a median-ofmeans study, 2021.

Lee, J. C. H. and Valiant, P. Optimal sub-Gaussian mean estimation in R. In 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS), pp. 672–
683. IEEE, 2022.

Lugosi, G. and Mendelson, S. Robust multivariate mean estimation: The optimality of trimmed mean. *The Annals* of Statistics, 49(1):393 - 410, 2021.

Minsker, S. U-statistics of growing order and sub-gaussian mean estimators with sharp constants. arXiv preprint arXiv:2202.11842, 2023.

Nemirovsky, A. and Yudin, D. Problem Complexity and Method Efficiency in Optimization. Wiley, 1983.

Oliveira, R. I., Orenstein, P., and Rico, Z. F. Finitesample properties of the trimmed mean. *arXiv preprint* arXiv:2501.03694, 2025.

Tukey, J. W. A survey of sampling from contaminated distributions. *Contributions to probability and statistics*, pp. 448–485, 1960.

## A. Remaining Proofs Of Section 4

In Section 4, we discussed the intuition behind our main results, Theorem 2.2 and 2.3. At a high level, our proof strategy for both main theorems uses the triangle inequality to bound estimation error introduced by adversarial corruption as the sum of two parts, one from changing the influence parameter from α to α˜, and one from the adversary arbitrarily corrupting the samples. We provide formal proofs for relevant lemmas and propositions in this section, and restate the main theorems for completeness: Theorem 2.2. Given any distribution D with mean µ *and variance* σ 2, parameters n, δ, η > 0, let X˜ be a set of n η*-corrupted samples from* D.

Suppose both log 1 δ nand δ *are bounded by some small universal constant, and suppose* η ≤1 24n log 1δ
. Then, with probability at least 1 − δ over the sampling process, Estimator 1 on input δ and X˜ will output an estimate µˆ *with error at most*

$$|{\hat{\mu}}-\mu|\leq\sigma\cdot{\Bigg(}({\sqrt{2}}+o(1)){\sqrt{\frac{\log{\frac{1}{\delta}}}{n}}}+222{\sqrt{\eta}}{\Bigg)}$$

Theorem 2.3. Given any distribution D with mean µ *and variance* σ 2, parameters n, δ, η > 0, let X˜ be a set of n η*-corrupted samples from* D.

Suppose both log 1 δ nand δ *are bounded by some small universal constant, and suppose* η ≤1 9n log 1δ
. Let δ
′ *be so that* 1 3 log 1δ =
1 3 log 1 δ
′ + 3ηn*. Then, with probability at least* 1 − δ
′ over the sampling process, Estimator 1 on input δ and X˜
will output an estimate µˆ *with error at most*

$$|{\hat{\mu}}-\mu|\leq\sigma\cdot\left(({\sqrt{2}}+o(1)){\sqrt{\frac{\log{\frac{1}{\delta^{\prime}}}}{n}}}+(135+o(1)){\sqrt{\eta}}\right)$$

Throughout the proofs, we will compare and make use of different values of "α", computed from either corrupted or uncorrupted data, and computed from different choices of parameters in the equation defining α. Here, we give a more general definition of notation (generalizing Definition 4.1) that we will be using.

Definition A.1. Let X = {xi} be a set of clean samples, and let X˜ = {x˜i} be the corresponding set of η-corrupted samples.

Denote by αδ,ρ the "influence parameter" solved from the corresponding condition involving the clean samples, so as to satisfy a version of Step 2 of Estimator 1 but with a modified right hand side ( 13 log 1 δ + ρn instead of 1 3 log 1δ
):

i
$$\sum_{i}\operatorname*{min}(\alpha_{\delta,\rho}x_{i}^{2},1)={\frac{1}{3}}\mathrm{log}\,{\frac{1}{\delta}}+\rho n$$
and denote by α˜δ,ρ the "influence parameter" solved instead from the corresponding condition involving the *corrupted*
samples:
$$\sum_{i}\operatorname*{min}({\tilde{\alpha}}_{\delta,\rho}{\tilde{x}}_{i}^{2},1)={\frac{1}{3}}\mathrm{log}\,{\frac{1}{\delta}}+\rho n$$
i Theorem 2.3 refers to failure probability δ
′, so parts of the analysis will involve the notations αδ
′,η and α˜δ
′,η, for example.

We start with showing the crucial preliminary Lemma A.2, which bounds the value of α˜, allowing us to approximate it using different influence parameters on the *clean* data, with no assumption of the *corrupted* data: Lemma A.2 (Restatement of Lemma 4.2 under the notation of Definition A.1). Consider an arbitrary set of samples X and a new sample set X η ˜ -corrupted from X. Consider also an arbitrary input parameter δ. Using α˜ to denote the influence parameter of Estimator 1 on inputs (δ, X˜), i.e. α˜δ,0 in Definition *A.1, we have*

$$\alpha_{\delta,-2\eta}\leq\bar{\alpha}\leq\alpha_{\delta,2\eta}$$

Proof. Let the clean samples be X = {xi} and the corrupted samples be X˜ = {x˜i}.

To prove the first inequality, suppose for the sake of contradiction that *α < α* ˜ δ,−2η. Then,

$$\sum_{i}\min(\tilde{\alpha}\tilde{x}_{i}^{2},1)$$ $$\leq\sum_{i}\min(\tilde{\alpha}\tilde{x}_{i}^{2},1)+\sum_{i:\,x_{i}\,\,\text{computed}}\min(\tilde{\alpha}\tilde{x}_{i}^{2},1)$$ $$\leq\frac{1}{3}\text{log}\frac{1}{6}-2\eta n+\sum_{i:\,x_{i}\,\,\text{computed}}\min(\tilde{\alpha}\tilde{x}_{i}^{2},1)\text{(since}\tilde{\alpha}<\alpha_{\delta,-2\eta})$$ $$\leq\frac{1}{3}\text{log}\frac{1}{\delta}-\eta n\text{(since the sum has}\eta n\text{elements,each at most1)}$$ $$<\frac{1}{3}\text{log}\frac{1}{\delta}$$
which is a contradiction.
The second inequality follows similarly. Suppose for the sake of contradiction that *α > α* ˜ δ,2η. Then,

$$\sum_{i}\min(\tilde{\alpha}\tilde{x}_{i}^{2},1)$$ $$\geq\sum_{i}\min(\tilde{\alpha}\tilde{x}_{i}^{2},1)-\sum_{i:x_{i}\text{computed}}|\min(\tilde{\alpha}\tilde{x}_{i}^{2},1)-\min(\tilde{\alpha}\tilde{x}_{i}^{2},1)|$$ $$\geq\frac{1}{3}\log\frac{1}{\delta}+2\eta n-\sum_{i:x_{i}\text{computed}}|\min(\tilde{\alpha}\tilde{x}_{i}^{2},1)-\min(\tilde{\alpha}\tilde{x}_{i}^{2},1)|$$ (since $$\tilde{\alpha}>\alpha_{\delta,2\eta}$$ ) $$\geq\frac{1}{3}\log\frac{1}{\delta}+\eta n$$ (since the sum has $$\eta n$$ elements, each at most 1) $$>\frac{1}{3}\log\frac{1}{\delta}$$
which is a contradiction.
We in fact generalize the above lemma slightly for use in the proof of Theorem 2.3, which can be proven from Lemma A.2 by reparameterizing δ.

Corollary A.3. For any set of clean samples X and the corresponding η-corrupted samples X˜, and for any constant c > 2, we have αδ,(c−2)η ≤ α˜*δ,cη* ≤ αδ,(c+2)η.

The proof structure of Theorem 2.2 and 2.3 are essentially identical. We present the proof of Theorem 2.3 first. For the rest of the appendix, we will assume that the uncorrupted data distribution has mean 0 and variance 1 without loss of generality, due to the shift-and-scale equivariance of Estimator 1.

## A.1. Bounding The Error Due To Changing The Influence Parameter

We present the following proposition upper bounding the error incurred on Estimator 1 by using influence parameter α˜ := ˜αδ
′,3η instead of α := αδ
′,0 on the *clean* samples. That is, we compute α˜ on the corrupted samples, but analyze its effect on the clean samples. For the following section, recall our assumption that the underlying distribution has mean 0 and variance 1.

Proposition A.4. *Suppose both* log 1 δ′
nand δ
′ are bounded by some small universal constant. Let α be the influence parameter computed from the **clean** *samples with robustness level* 13 log 1 δ
′*, namely* α := αδ
′,0. Let α˜ be the influence parameter computed from the **corrupted** *samples with robustness level* 13 log 1 δ
′ + 3ηn*, namely,* α˜ := ˜αδ′,3η. Then with probability at least 1 −
6 8 δ
′, the mean estimate using α on the clean samples differs from the mean estimate using α˜ *on the* clean *samples by at most* 125.5
√η*, i.e.,*

$$\left|\sum_{i}x_{i}\operatorname*{min}({\bar{\alpha}}x_{i}^{2},1)-\sum_{i}x_{i}\operatorname*{min}(\alpha x_{i}^{2},1)\right|\leq125.5n{\sqrt{\eta}}$$
√η (3)

$$(3)$$

12 To provide some intuition towards our proof strategy for Proposition A.4, first notice that we can bound the left hand side via Cauchy-Schwarz as

$$\sqrt{\left(\sum_{i}x_{i}^{2}\right)\left(\sum_{i}\left(\operatorname*{min}(\tilde{\alpha}x_{i}^{2},1)-\operatorname*{min}(\alpha x_{i}^{2},1)\right)^{2}\right)}$$

This turns out to be insufficient; we instead bound (3) by defining the set S of indices where min(˜αx2 i, 1) ̸= min(αx2 i, 1),
and restrict the range of both sums in (3) to the range i ∈ S, since doing so only discards zero terms and does not change the sum. Thus we instead have the Cauchy-Schwarz bound

$$\sqrt{\left(\sum_{i\in S}x_{i}^{2}\right)\left(\sum_{i\in S}\left(\operatorname*{min}(\tilde{\alpha}x_{i}^{2},1)-\operatorname*{min}(\alpha x_{i}^{2},1)\right)^{2}\right)}$$

The bound on the second parenthetical makes crucial use of the comparison of α˜ and α provided by Corollary A.3. The first parenthetical is an empirical variance, but the restriction i ∈ S means that |xi| cannot be too large; we thus use Bernstein's inequality to bound this S-truncated empirical second moment, in terms of a lower bound on α, which we prove next. To show our lower bound on α, we first calculate the following straightforward relations between the empirical and population quantiles.

Throughout this section, we denote by Qq(D) the q (true) quantile of D, i.e., P[D ≤ Qq(D)] = q.

Lemma A.5. *Suppose both* log 1 δ′
nand δ
′ *are bounded by some small universal constant. Denote* c1 = 0.277 <
1 3
, and κ := c1( 1 n log 1 δ
′ )*. Let constant* c2 := 102.907. Then the 1 − κ empirical quantile of xiis at most Q1−κ/c2
(D) *with* probability at least 1 −
1 8 δ
′.

Proof. For the 1 − κ empirical quantile of xito be greater than Q1−κ/c2
(D), there has to be more than κn samples greater than Q1−κ/c2
(D). Thus, it suffices to prove that |{i ∈ [n] : xi ≥ Q1−κ/c2
(D)*}| ≥* κn with probability at most 18 δ
′.

Denote Zi:= 1xi≥Q1−κ/c2
(D). Then Z := |{i ∈ [n] : xi ≥ Q1−κ/c2
(D)}| =Pn i=1 Zi. We denote by p =κ c2 the probability that an individual i is in this set; and thus E[Z] = pn. Since each Ziis a coin flip of probability p, we further have that *V ar*[Zi] = p(1 − p).

By multiplicative Chernoff,

$$\mathbb{P}[Z\geq c_{2}pm]\leq\left(\frac{e^{c_{2}-1}}{c_{2}^{c_{2}}}\right)^{pn}$$ $$=\exp\left((c_{2}-1-c_{2}\log c_{2})pm\right)$$ $$=\exp\left(\frac{c_{1}(c_{2}-1-c_{2}\log c_{2})}{c_{2}}\log\frac{1}{\delta}\right)$$ $$\leq\exp\left((-1.01)\log\frac{1}{\delta^{\prime}}\right)\qquad\text{by choice of}c_{1}\text{and}c_{2}$$ $$\leq\exp\left(-\log\frac{8}{\delta^{\prime}}\right)=\frac{1}{8}\delta^{\prime}\qquad\text{since}1.01\text{log}\frac{1}{\delta^{\prime}}\geq\log\frac{8}{\delta^{\prime}}\text{for suff.small}\delta^{\prime}$$

as desired. By symmetry, we have the following corollary as well:
Corollary A.6. The κ empirical quantile of xiis at least Qκ/c2(D) *with probability at least* 1 −
1 8 δ
′.

Lemma A.7. Let Dtrimmed denote a "trimmed" version of D: namely, D conditioned on lying in [Qκ/c2
(D), Q1−κ/c2
(D)].

Then E[D2 trimmed] ≤c 2 2
(c2−2κ)
2 .

$\square$
Proof. Denote by 1*untrimmed*(x) the indicator that returns 1 if x ∈ [Qκ/c2
(D), Q1−κ/c2
(D)] and 0 otherwise. Then observe that D*trimmed* =c2 c2−2κ
(D · 1*untrimmed*). Thus

$$\mathbb{E}[D_{trimmed}^{2}]=\mathbb{E}\left[\left(\frac{c_{2}}{c_{2}-2\kappa}(D\cdot\mathbb{1}_{untrimmed})\right)^{2}\right]$$ $$=\frac{c_{2}^{2}}{(c_{2}-2\kappa)^{2}}\,\mathbb{E}[D^{2}\mathbb{1}_{untrimmed}]$$ $$\leq\frac{c_{2}^{2}}{(c_{2}-2\kappa)^{2}}\,\mathbb{E}[D^{2}]$$ $$=\frac{c_{2}^{2}}{(c_{2}-2\kappa)^{2}}$$

as desired.

Lemma A.8. *Suppose both* log 1 δ′
nand δ
′ are bounded by some small universal constant. Let (xtrimmed)i *denote the sample* xi after trimming, namely: let (xtrimmed)i = xiif xi ∈ [Qκ/c2
(D), Q1−κ/c2
(D)], and (x*trimmed*)i = 0 otherwise. Let constant c3 := 251.099. Then Pi
(x*trimmed*)
2 i ≤
(c3+1)c 2 2
(c2−2κ)
2 n *with probability at least* 1 −
2 8 δ
′.

Proof. P
First notice that if we replace any trimmed xi with a random sample according to D*trimmed*, then the sum i
(x*trimmed*)
2 ican only increase. Thus, to prove the claim, it suffices to bound the sum of n i.i.d. samples from D*trimmed*.

With an abuse of notation, we let {(x*trimmed*)i}i≤n denote a set of such samples.

Also notice that by our choice of c3, we have, crucially, 3c1c 2 3
(6+2c3)c2
≥ 1.01.

We start by bounding Qκ/c2(D) and Q1−κ/c2(D). Since we assume D has mean 0 and variance 1, by Chebyshev's inequality, P-|D| ≥ pc2 κ
≤
κ c2
, which implies that Qκ/c2 ≥ −pc2 κ and Q1−κ/c2 ≤pc2 κ
. As a result, (x*trimmed*)
2 i ≤
c2 κ for all i. Then,

$$\mathrm{Var}[(x_{trimmed})_{i}^{2}]\leq\mathbb{E}\left[((x_{trimmed})_{i}^{2})^{2}\right]$$ $$\leq\frac{c_{2}}{\kappa}\,\mathbb{E}[(x_{trimmed})_{i}^{2}]$$ $$\leq\frac{c_{2}^{3}}{\kappa(c_{2}-2\kappa)^{2}}\qquad\text{by Lemma A.7}$$

Thus, by Bernstein's inequality,

P "X i ((xtrimmed)i) 2 ≥ (c3 + 1)c 2 2n (c2 − 2κ) 2 # = P "1 n X i ((xtrimmed)i) 2 −c 22 (c2 − 2κ) 2 ≥c3c 22 (c2 − 2κ) 2 # ≤ P "1 n X i ((xtrimmed)i) 2 − E[((xtrimmed)i) 2] ≥c3c 22 (c2 − 2κ) 2 # ≤ 2 exp  − c 23nc 4 2 (c2−2κ) 4 2 Var[(xtrimmed) 2 i] + 2c3c 3 2 3κ(c2−2κ) 2   ≤ 2 exp  − c 2 3nc 4 2 (c2−2κ) 4 2c 32 κ(c2−2κ) 2 +2c3c 32 3κ(c2−2κ) 2  

14

$$=2\exp\left(-\frac{3c_{3}^{2}c_{2}n\kappa}{(6+2c_{3})(c_{2}-2\kappa)^{2}}\right)$$ $$\leq\exp\left(-\frac{3c_{3}^{2}n\kappa}{(6+2c_{3})c_{2}}\right)$$ $$=2\exp(-\frac{3c_{1}c_{3}^{2}}{(6+2c_{3})c_{2}}\log\frac{1}{\delta^{\prime}})\qquad\text{by definition of}\kappa$$ $$\leq2\exp(-1.011\log\frac{1}{\delta^{\prime}})\qquad\text{by choice of}c_{3}$$ $$\leq2\exp(-\log\frac{8}{\delta^{\prime}})\qquad\text{since}1.011\log\frac{1}{\delta}\geq\log\frac{8}{\delta^{\prime}}\text{for suff.small}\delta^{\prime}$$ $$=\frac{2}{8}\delta^{\prime}$$

as desired. Combining Lemma A.5, Corollary A.6, and Lemma A.8, we have the following corollary:
Corollary A.9. *Suppose there is a sufficiently small constant that upper bounds* δ
′. Let S<κ denote the set of indices i *s.t.*
x 2 iis not in the top κ *(empirical) quantile. Then* Pi∈S<κ x 2 i ≤
(c3+1)c 2 2
(c2−2κ)
2 n *with probability at least* 1 −
4 8 δ
′.

We are now ready to present a lower bound on α := αδ
′,0, before proving Proposition A.4.

Lemma A.10. *Suppose both* log 1 δ′
nand δ
′ *are bounded by some small universal constant. Then* α := αδ
′,0 ≥
0.000214 1n log 1 δ
′ *with probability at least* 1 −
4 8 δ
′.

Proof. Recall that by definition of α,

$$\frac{1}{3}\log\frac{1}{\delta^{\prime}}=\sum_{i}\min(\alpha x_{i}^{2},1)$$ $$=\sum_{i:x_{i}^{2}\text{not in the top}\kappa\text{quantile}}\min(\alpha x_{i}^{2},1)+\sum_{i:x_{i}^{2}\text{in the top}\kappa\text{quantile}}\min(\alpha x_{i}^{2},1)$$ $$\leq\sum_{i:x_{i}^{2}\text{not in the top}\kappa\text{quantile}}\alpha x_{i}^{2}+\sum_{i:x_{i}^{2}\text{in the top}\kappa\text{quantile}}1$$ $$\leq\sum_{i:x_{i}^{2}\text{not in the top}\kappa\text{quantile}}\alpha x_{i}^{2}+\kappa\kappa n$$ $$=\sum_{i:x_{i}^{2}\text{not in the top}\kappa\text{quantile}}\alpha x_{i}^{2}+c_{1}\log\frac{1}{\delta^{\prime}}$$

Rearranging, this is equivalent to

$$\left({\frac{1}{3}}-c_{1}\right)\log{\frac{1}{\delta^{\prime}}}\leq\sum_{i:x_{i}^{2}{\mathrm{~not~in~the~top~}}\kappa{\mathrm{~quantile~}}}\alpha x_{i}^{2}$$

By Corollary A.9, with probability at least 1 −
4 8 δ
′,

$$\sum_{i:x_{i}^{2}\ \mathrm{not\ in\ the\ top}\ \kappa\ \mathrm{quantile}}x_{i}^{2}\leq{\frac{(c_{3}+1)c_{2}^{2}}{(c_{2}-2\kappa)^{2}}}n\leq{\frac{(c_{3}+1)c_{2}^{2}}{(c_{2}-2)^{2}}}n$$

Which implies that α ≥13 − c1 (c2−2)2
(c3+1)c 22 1 n log 1 δ
′ ≥ 0.000214 1n log 1 δ
′ with probability at least 1 −
4 8 δ
′.

Proof of Proposition *A.4.* First, notice that for all i such that |xi| ≥ q1α ≥
q1 α˜
, the corresponding term in the sum in the left hand side becomes 0. Thus, using the notation P≤ to denote summing over elements |xi| ≤ q1α˜
, the left hand side in the guarantee of Proposition A.4 is equal to

$\sum_{i}\min(\tilde{\alpha}x_{i}^{2},1)-\sum_{i}\min(\alpha x_{i}^{2},1)$
Then, rearranging the sums, we have

$\sum_{i}x_{i}\min(\tilde{\alpha}x_{i}^{2},1)-\sum_{i}x_{i}\min(\alpha x_{i}^{2},1)$  $\leq\sum_{i}\left|x_{i}\left(\min(\tilde{\alpha}x_{i}^{2},1)-\min(\alpha x_{i}^{2},1)\right)\right|$
$$\leq\sqrt{\left(\sum_{\leq}x_{i}^{2}\right)\left(\sum_{\leq}\left(\operatorname*{min}(\bar{\alpha}x_{i}^{2},1)-\operatorname*{min}(\alpha x_{i}^{2},1)\right)^{2}\right)}$$
by Cauchy-Schwarz  $$\newcommand{\vecs}[1]{\overset{\rightharpoonup}{\mathbf{#1}}}$$  $$\newcommand{\vecd}[1]{\overset{-\!-\!\rightharpoonup}{\vphantom{a}\smash{#1}}}$$
for which we can bound the two terms separately.

To bound the first term, since we sum over only those terms where |xi| ≤ q1α for all i, and by Lemma A.10 α ≥
0.000214 nlog 1 δ
′ with probability 1 −
4 8 δ
′, we have that x 2 i ≤
4672n log 1 δ′
for all i. Since X has mean 0 and variance 1, we know

that E[x
2 i
] ≤ E[X2] = 1, and Var[x
2
i i log 1 δ′ i log 1 δ′ P  X ≤ x 2 i ≥ 3150n   = P   1 n X ≤ x 2 i − 1 ≥ 3149   ≤ P   1 n X ≤ x 2 i − E[x 2 i ] ≥ 3149   ≤ 2 exp −31492n 9344 n log 1 δ′ + 9344 · 3149 n log 1 δ′ /3 ! ≤ 2 exp  −1.01log 1 δ ′  ≤ 2 exp  − log 8 δ ′ since 1.01log 1 δ ′ ≥ log  8 δ ′ for suff. small δ ′ = 2 8 δ ′
] ≤ E[x
4
] ≤
4672n
E[x
2
] ≤
4672n
. Thus, by Bernstein's inequality,

In other words, conditioning on Lemma A.10 holding, with probability at least 1 −
2 8 δ
′,P≤ x 2 i ≤ 3150n.

To bound the second term, note that

$$\sum_{\leq}\left(\min(\tilde{\alpha}x_{i}^{2},1)-\min(\alpha x_{i}^{2},1)\right)^{2}\leq\sum_{i}\left(\min(\tilde{\alpha}x_{i}^{2},1)-\min(\alpha x_{i}^{2},1)\right)^{2}.$$
i, 1)2≤X
$$t\leq\sum_{i}\left(\operatorname*{min}(\bar{\alpha}x_{i}^{2},1)-\operatorname*{min}(\alpha x_{i}^{2},1)\right)$$

since α˜ := ˜αδ
′,3η ≥ αδ
′,η ≥ α := αδ
′,0 by Corollary A.3, and "α" is monotonic in the η argument, and thus 0 ≤
min(˜αx2 i, 1) − min(αx2 i, 1) ≤ 1 for all i.

To further upper bound the last quantity, we have

$$\sum_{i}\left(\operatorname*{min}({\hat{\alpha}}x_{i}^{2},1)-\operatorname*{min}(\alpha x_{i}^{2},1)\right)=\sum_{i}\operatorname*{min}({\hat{\alpha}}x_{i}^{2},1)-\sum_{i}\operatorname*{min}(\alpha x_{i}^{2},1)$$

16

$$\begin{array}{l}{{\leq\sum_{i}\operatorname*{min}(\alpha_{\delta^{\prime},5\eta}x_{i}^{2},1)-\sum_{i}\operatorname*{min}(\alpha_{\delta^{\prime},0}x_{i}^{2},1)}}\\ {{={\frac{1}{3}}\mathrm{log}\,{\frac{1}{\delta^{\prime}}}+5\eta n-{\frac{1}{3}}\mathrm{log}\,{\frac{1}{\delta^{\prime}}}}}\\ {{=5\eta n}}\end{array}$$
i, 1) by Corollary A.3 and by definition of α
Finally, summarizing, we have that with probability at least 1 −
6 8 δ
′:

$\sqrt{\left(\sum\limits_{\leq}x_i^2\right)\left(\sum\limits_{\leq}\left(\min(\tilde{\alpha}x_i^2,1)-\min(\alpha x_i^2,1)\right)^2\right)}$ $\leq\sqrt{3150n\cdot5\eta n}$  $=n\sqrt{15750\eta}$  $\leq125.5n\sqrt{\eta}$

by $\mathbb{C}$orolla. 

as desired.

## A.2. Bounding The Error Due To Corrupting The Samples

We now present the following proposition upper bounding the error incurred on Estimator 1 by the arbitrary corruption of the adversary on the clean samples, while *fixing* α˜ := ˜αδ
′,3η as the influence parameter.

We again assume that the uncorrupted distribution has mean 0 and variance 1.

Proposition A.11. *Suppose both* log 1 δ′
nand δ
′ are bounded by some small universal constant. Let α˜ be the influence parameter computed from the **corrupted** *samples with robustness level* 13 log 1 δ
′ + 3ηn*, namely,* α˜ := ˜αδ
′,3η*. Then with* probability at least 1 −
1 8 δ
′, the mean estimate using α˜ *on the* **clean** samples differs from the mean estimate using α˜ *on the* corrupted samples by at most 8.586√η*, i.e.,*

$$\left|\sum_{i}{\bar{x}}_{i}(1-\operatorname*{min}({\bar{\alpha}}{\bar{x}}_{i}^{2},1))-\sum_{i}x_{i}(1-\operatorname*{min}({\bar{\alpha}}x_{i}^{2},1))\right|\leq8.586n{\sqrt{\eta}}$$

To provide some intuition towards our proof strategy for Proposition A.11, consider the adversary's arbitrary corruption, which can be interpreted piece-wise as moving each clean sample that the adversary wishes to corrupt to a new location. Since the influence parameter controls the contribution of each sample to the mean estimate, based on how far from the mean it is, moving a sample too far from the mean or moving the sample too close to the mean will both incur very little error. The question then, is, what is the maximum estimation error the adversary can incur by corrupting a single sample? Fixing the value of α˜ as in the statement of Proposition A.11, we can upper bound the maximum magnitude of the expression 1 n Pi xi(1 − min(˜αx2 i, 1)) by calculus, which is O(1/
√α˜). We will show a lower bound of α˜, specifically that α˜ ≥ Ω(η),
and then conclude that the maximum total error possible by corrupting ηn samples is at most 1n
· ηn · Θ(1/
√η) = Θ(√η),
as desired. We use a similar strategy as in the proof of Proposition A.4, using quantile statistics as well as Corollary A.3 to obtain our desired lower bound on α˜, before proving Proposition A.11 in Appendix A.

Lemma A.12. *Suppose both* log 1 δ′
nand δ
′ *are bounded by some small universal constant. Denote* c1 =
1 3
, and κ :=
c1( 1 n log 1 δ
′ )*. Let* c2 := 55.252. Then the 1 − κ empirical quantile of xiis at most Q1−κ/c2
(D) *with probability at least* 1 −
1 32 δ
′.

Proof. For the 1 − κ empirical quantile of xito be greater than Q1−κ/c2
(D), there has to be more than κn samples greater than Q1−κ/c2
(D). Thus, it suffices to prove that |{i ∈ [n] : xi ≥ Q1−κ/c2
(D)*}| ≥* κn with probability at most 1 32 δ
′.

Denote Zi:= 1xi≥Q1−κ/c2
(D). Then Z := |{i ∈ [n] : xi ≥ Q1−κ/c2
(D)}| =Pn i=1 Zi. Obviously E[Z] = *κn/c*2. Denote p := E[Zi] = κ/c2. Then *V ar*[Zi] = p(1 − p).

By multiplicative Chernoff,

$$\mathbb{P}[Z\geq c_{2}pm]\leq\left(\frac{e^{c_{2}-1}}{c_{2}^{\prime}}\right)^{pn}$$ $$=\exp\left((c_{2}-1-c_{2}\log c_{2})pn\right)$$ $$=\exp\left(\frac{c_{1}(c_{2}-1-c_{2}\log c_{2})}{c_{2}}(\log\frac{1}{\delta^{\prime}})\right)$$ $$\leq\exp\left((-1.01)(\log\frac{1}{\delta^{\prime}})\right)\qquad\text{by choice of}c_{1},c_{2}$$ $$\leq\exp\left(-\log\frac{32}{\delta^{\prime}}\right)=\frac{1}{32}\delta^{\prime}\qquad\text{since}1.01\text{log}\frac{1}{\delta^{\prime}}\geq\log\frac{32}{\delta^{\prime}}\text{for suff.small}\delta^{\prime}$$

as desired.

By symmetry, we have the following corollary as well:
Corollary A.13. The κ empirical quantile of xiis at least Qκ/c2
(D) *with probability at least* 1 −
1 32 δ
′.

Lemma A.14. Let Dtrimmed denote the trimmed version of D conditioned on lying in [Qκ/c2
(D), Q1−κ/c2
(D)]*. Then* E[D2 trimmed] ≤c 2 2
(c2−2κ)
2 .

Proof. Denote by 1*untrimmed*(x) the indicator measure that maps x to 1Qκ/c2
(D)≤x≤Q1−κ/c2
(D), the indicator of whether x is untrimmed. Then observe that D*trimmed* =c2 c2−2κ
(D · 1*untrimmed*). Thus

$$\mathbb{E}[D^{2}_{trimmed}]=\mathbb{E}\left[\left(\frac{c_{2}}{c_{2}-2\kappa}(D\cdot\mathbb{1}_{untrimmed})\right)^{2}\right]$$ $$=\frac{c_{2}^{2}}{(c_{2}-2\kappa)^{2}}\,\mathbb{E}[D^{2}\mathbb{1}_{untrimmed}]$$ $$\leq\frac{c_{2}^{2}}{(c_{2}-2\kappa)^{2}}\,\mathbb{E}[D^{2}]$$ $$=\frac{c_{2}^{2}}{(c_{2}-2\kappa)^{2}}$$
$$\square$$

as desired.

Lemma A.15. *Suppose both* log 1 δ′
nand δ
′ are bounded by some small universal constant. Let (xtrimmed)i denote the sample xi after trimming, such that (xtrimmed)i = xiif xi ∈ [Qκ/c2(D), Q1−κ/c2(D)], and (x*trimmed*)i = 0 *otherwise.*
Denote c3 := 114.532. Then Pi
(x*trimmed*)
2 i ≤
(c3+1)c 2 2
(c2−2κ)
2 n *with probability at least* 1 −
2 32 δ
′.

Proof. P
First notice that if we replace any trimmed xi with a random sample according to D*trimmed*, then the sum i
(x*trimmed*)
2 i can only increase. Thus, to prove the claim, it suffices to bound the sum of n i.i.d. samples from D*trimmed*.

With an abuse of notation, we let {(x*trimmed*)i}i≤n denote a set of such samples.

Also notice that by our choice of c3, we have, crucially, 3c 2 3c1
(6+2c3)c2
≥ 1.01.

We start by bounding Qκ/c2
(D) and Q1−κ/c2
(D). Since we assume D has mean 0 and variance 1, by Chebyshev's inequality, P-|D| ≥ pc2 κ
≤
κ c2
, which implies Qκ/c2 ≥ −pc2 κ and Q1−κ/c2 ≤pc2 κ
. As a result, (x*trimmed*)
2 i ≤
c2 κ for all i. Then,

$$\begin{split}\text{Var}[(x_{trimmed})_{i}^{2}]&\leq\mathbb{E}\left[\left((x_{trimmed})_{i}^{2})^{2}\right)\right]\\ &\leq\frac{c_{2}}{\kappa}\,\mathbb{E}[(x_{trimmed})_{i}^{2}]\end{split}$$

18

$$\leq\frac{c_{2}^{3}}{\kappa(c_{2}-2\kappa)^{2}}$$
by Lemma $\mathbf{A}.14$. 
Thus, by Bernstein's inequality,

P

"X
i
((x*trimmed*)i)
2 ≥
(c3 + 1)c
2
2n
(c2 − 2κ)
2
#
= P
"1
n
X
i
((x*trimmed*)i)
2 −c
22
(c2 − 2κ)
2
≥c3c
22
(c2 − 2κ)
2
#
≤ P
"1
n
X
i
((x*trimmed*)i)
2 − E[((x*trimmed*)i)
2] ≥c3c
22
(c2 − 2κ)
2
#
≤ 2 exp

−
c
23nc
4
2
(c2−2κ)
4
2 Var[(x*trimmed*)
2
i] + 2c3c
3
2
3κ(c2−2κ)
2

≤ 2 exp

−
c
2
3nc
4
2
(c2−2κ)
4
2c
32
κ(c2−2κ)
2 +2c3c
32
3κ(c2−2κ)
2

= 2 exp 
−3c
23
c2nκ
(6 + 2c3)(c2 − 2κ)
2

≤ 2 exp 
−3c
23nκ
(6 + 2c3)c2

= 2 exp(−3c
23c1
(6 + 2c3)c2
(log 
1
δ
′
)) by definition of κ
≤ 2 exp(−1.01(log 
1
δ
′
)) by choice of c3
≤ 2 exp(−(log 
32
δ
′
)) since 1.01log 
1
δ
′
≥ log 
32
δ
′
for suff. small δ
′
=2
32
δ
′
as desired. Combining Lemma A.12, Corollary A.13, and Lemma A.15, we have the following corollary:
Corollary A.16. *Suppose both* log 1 δ′
nand δ
′ are bounded by some small universal constant. Let S<κ *denote the set of* indices i *s.t.* x 2 i is not in the top κ *(empirical) quantile. Then* Pi∈S<κ x 2 i ≤
(c3+1)c 2 2
(c2−2κ)
2 n *with probability at least* 1 −
1 8 δ
′.

Proof of Proposition *A.11.* We start by consider a single sample x˜i which is corrupted such that xi ̸= ˜xi. Fixing all other samples, we solve for the maximum (signed) error, or equivalently mean shift, that the adversary can incur by arbitrarily corrupting this sample xi only. Recall that the estimator (fixing influence parameter α˜ := ˜αδ
′,3η) is defined as µˆ =Pi xi(1 − min(˜αx2 i, 1)). Taking the derivative with respect to xi, restricted to the region in which min(˜αx2 i, 1) = ˜αx2 i, we have

$${\frac{\partial}{\partial x_{i}}}{\hat{\mu}}=1-3{\bar{\alpha}}x_{i}^{2}=0\Rightarrow x=\pm{\frac{1}{\sqrt{3{\hat{\alpha}}}}}$$

Since α˜ ≥ 0, a local maximum occurs at x = √
1 3 ˜α
. The corresponding maximum contribution of this single term is 2 3
√3 ˜α
.

It is easy to verify that this local maximum is in fact the global maximum based on taking the minimization between αx˜
2 i and 1. Symmetrically, the minimal contribution of a single term is −2 3
√3 ˜α
.

Thus, the maximum error the adversary can incur by corrupting a single term is 4 3
√3 ˜α
. The maximum error the adversary can incur by corrupting ηn terms is 4ηn 3
√3 ˜α
.

To upper bound 4ηn 3
√3 ˜α
, we need to lower bound α˜. Towards this, notice that α˜ ≥ αδ
′,η by Corollary A.3, which by definition satisfies:

ηn + 1 3 log 1 δ ′ =X i min(αδ ′,ηx 2 i , 1) ≤X i min(˜αx2 i, 1) by Corollary A.3 ≤X i:x 2 i not in the top κ quantile min(˜αx2 i, 1) + X i:x 2 i in the top κ quantile min(˜αx2 i, 1) ≤X i:x 2 inot in the top κ quantile αx˜ 2 i +X i:x 2 iin the top κ quantile 1 =X i:x 2 i not in the top κ quantile αx˜ 2 i + κn i:x 2 i not in the top κ quantile αx˜ 2 i + 1 3 log 1 δ ′ =X
Rearranging, this is equivalent to
$$\bar{\alpha}x_{i}^{2}$$  e. 
$\eta\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\mu$ = $\
By Corollary A.16, with probability at least 1 −
1 8 δ
′,

$\begin{array}{c}\includegraphics[height=36.135pt]{Fig1}\end{array}$  $\begin{array}{c}\includegraphics[height=36.135pt]{Fig2}\end{array}$  $\begin{array}{c}\includegraphics[height=36.135pt]{Fig3}\end{array}$  $\begin{array}{c}\includegraphics[height=36.135pt]{Fig4}\end{array}$  $\begin{array}{c}\includegraphics[height=36.135pt]{Fig5}\end{array}$  $\begin{array}{c}\includegraphics[height=36.135pt]{Fig6}\end{array}$  $\begin{array}{c}\includegraphics[height=36.135pt]{Fig7}\end{array}$  \(\begin{array}{c}\includegraphics[height=36.135pt]{Fig8}\end{array}\
Thus,

$\tilde{\alpha}x_{i}^{2}\leq\frac{(c_{3}+1)c_{2}^{2}}{(c_{2}-2)^{2}}n\tilde{\alpha}$.  $i$:$x_{i}^{2}$ not in the top $\kappa$ quantile 
which is equivalent to

$$\tilde{\alpha}\geq\frac{(c_{2}-2)^{2}}{(c_{3}+1)c_{2}^{2}}\eta\geq0.00804\eta$$

Thus, we have

$$\left|\sum_{i}\hat{x}_{i}(1-\min(\hat{\alpha}\hat{x}_{i}^{2},1))-\sum_{i}x_{i}(1-\min(\hat{\alpha}x_{i}^{2},1))\right|\leq\frac{4\eta n}{3\sqrt{3\hat{\alpha}}}\leq\frac{4\eta n}{3\sqrt{0.02412\eta}}\leq8.586n\sqrt{\eta}$$

as desired.

## A.3. Proof Of Theorem 2.3

Equipped with Propositions A.4 and A.11, we are ready to formally prove Theorem 2.3.

$$\square$$