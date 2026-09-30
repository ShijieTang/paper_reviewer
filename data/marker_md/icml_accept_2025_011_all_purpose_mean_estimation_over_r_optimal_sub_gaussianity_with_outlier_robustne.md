# All-Purpose Mean Estimation over R: Optimal Sub-Gaussianity with Outlier Robustness and Low Moments Performance

Jasper C.H. Lee <sup>1</sup> Walter McKelvie 2 3 4 Maoyuan Song <sup>2</sup> Paul Valiant <sup>2</sup>

### Abstract

We consider the basic statistical challenge of designing an "all-purpose" mean estimation algorithm that is recommendable across a variety of settings and models. Recent work by [Lee & Valiant](#page-9-0) [\(2022\)](#page-9-0) introduced the first 1-d mean estimator whose error in the standard finitevariance+i.i.d. setting is optimal even in its constant factors; experimental demonstration of its good performance was shown by [Gobet et al.](#page-9-1) [\(2022\)](#page-9-1). Yet, unlike for classic (but not necessarily practical) estimators such as median-of-means and trimmed mean, this new algorithm lacked proven robustness guarantees in other settings, including the settings of adversarial data corruption and heavy-tailed distributions with infinite variance. Such robustness is important for practical use cases. This raises a research question: is it possible to have a mean estimator that is robust, *without* sacrificing provably optimal performance in the standard i.i.d. setting? In this work, we show that Lee and Valiant's estimator is in fact an "all-purpose" mean estimator by proving:

- (A) It is robust to an η-fraction of data corruption, even in the strong contamination model; it has optimal estimation error O(σ √η) for distributions with variance σ 2 .
- (B) For distributions with finite z th moment, for z ∈ (1, 2), it has optimal estimation error, matching the lower bounds of [Devroye et al.](#page-9-2) [\(2016\)](#page-9-2) up to constants.

We further show (C) that outlier robustness for 1-d mean estimators in fact implies neighborhood optimality, a notion of beyond worst-case and distribution-dependent optimality recently introduced by [Dang et al.](#page-8-0) [\(2023\)](#page-8-0). Previously, such an

optimality guarantee was only known for medianof-means, but now it holds also for all estimators that are simultaneously *robust* and *sub-Gaussian*, including Lee and Valiant's, resolving a question raised by Dang et al. Lastly, we show (D) the asymptotic normality and efficiency of Lee and Valiant's estimator, as further evidence for its performance across many settings.

### 1. Introduction

The past decade has seen reinvigorated interest in understanding fundamental statistical problems, and in particular, the foundational problem of mean estimation: given n samples from an unknown distribution, and an allowed failure probability δ, what is the most accurate estimate of the distribution mean? The sample mean is the traditional estimate, yet, its sensitivity to extreme values makes it an unreliable estimator in practice. Other classic estimators, for example the well-known median-of-means estimator [\(Alon et al.,](#page-8-1) [1999;](#page-8-1) [Nemirovsky & Yudin,](#page-9-3) [1983;](#page-9-3) [Jerrum et al.,](#page-9-4) [1986\)](#page-9-4), also suffer from poor empirical performance.

Recent work by [Lee & Valiant](#page-9-0) [\(2022\)](#page-9-0), sitting in a line of research initiated by the seminal paper of [Catoni](#page-8-2) [\(2012\)](#page-8-2), gave the first 1-dimensional mean estimator whose estimation error is optimal (i.e. sub-Gaussian error) and tight even in the leading constant. [Gobet et al.](#page-9-1) [\(2022\)](#page-9-1) further demonstrated its good empirical performance against other competing estimators.

Fact 1.1 [\(Lee & Valiant](#page-9-0) [2022\)](#page-9-0). *Given any distribution* D *with mean* µ *and variance* σ 2 *(both unknown), parameters* n, δ > 0*, let* X *be a set of* n *independent samples from* D*. Then, with probability at least* 1 − δ *over the sampling process, Estimator [1](#page-0-0) on input* δ *and* X *will output an estimate* µˆ *with error at most*

$$|\hat{\mu} - \mu| \leq \sigma \cdot \left( (\sqrt{2} + o(1))\sqrt{\frac{\log \frac{1}{\delta}}{n}} \right)$$

*Here, the* <sup>o</sup>(1) *term tends to* <sup>0</sup> *as* log <sup>1</sup> n , δ → (0, 0) *and, crucially, is independent of* D*.*

<sup>1</sup>University of California, Davis <sup>2</sup> Purdue University <sup>3</sup>Columbia University <sup>4</sup>Harvard University. Correspondence to: Jasper C.H. Lee <jasperlee@ucdavis.edu>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

#### Estimator 1 Mean Estimator of Lee and Valiant

Inputs: n i.i.d. samples {xi}; Confidence parameter δ.

- 1. Compute an initial estimate κ using any sub-Gaussian, scale-and-translation-equivariant and robust mean estimator (cf. Facts [A.18](#page-21-0) and [A.23\)](#page-25-0). One possible choice is the median-of-means estimate: evenly partition the data into log <sup>1</sup> δ groups and let κ be the median of the set of means of the groups;
- 2. Find the solution α to the monotonic, piecewise-linear equation

$$\sum_i \min(\alpha(x_i - \kappa)^2, 1) = \frac{1}{3} \log \frac{1}{\delta}$$

- 3. Output:

$$\hat{\mu} = \kappa + \frac{1}{n} \sum_i (x_i - \kappa)(1 - \min(\alpha(x_i - \kappa)^2, 1))$$

The constant "√ 2" in Fact [1.1](#page-0-0) is optimal, with matching lower bounds. Essentially, Estimator [1](#page-0-0) uses Step 1 to find a preliminary estimate (e.g. using median-of-means, but it can be other choices as well), which is then refined in Steps 2 and 3 into a better estimate—whose accuracy is optimal even in the leading multiplicative constant. Even though the form of the estimator is somewhat complicated—including a function inversion in Step 2—the estimator can in fact be computed easily in quasilinear time via a sorting operation, or even in linear time using a more subtle algorithm. See [\(Lee & Valiant,](#page-9-0) [2022\)](#page-9-0) for more discussion of the design and analysis of Estimator [1.](#page-0-0)

While Fact [1.1](#page-0-0) is a strong recommendation for Estimator [1,](#page-0-0) getting optimal performance in this regime is not the only practical desideratum for an estimator. Practitioners have long observed that some real-world distributions are very heavy-tailed, with variances that are gigantic (thus rendering guarantees like Fact [1.1](#page-0-0) effectively useless), but whose lower moments—between the first and second moments might be much more bounded. Further, real-world data can be subject to random or even adversarial corruption. Performance guarantees in these more demanding settings are therefore critical for practice. However, while there are relatively straightforward folklore analyses of classic algorithms in these settings—including, notably, median-of-means and trimmed mean—Lee and Valiant's estimator has not been studied in these contexts. The breadth of guarantees shown for these classic estimators might therefore lead researchers and practitioners to choose either of them over Lee and Valiant's estimator, despite the latter being a more accurate estimator in standard finite variance i.i.d. sample settings.

The natural and pressing research question, then, is

*Is it possible for a mean estimator to satisfy the important robustness guarantees of classic estimators,* without *losing the strong and optimal-constant performance that the Lee and Valiant estimator enjoys in the standard setting?*

In this paper, we show 4 strong positive results demonstrat-

- ing the robustness and performance of Estimator [1:](#page-0-0)
- (A) In the presence of η-fraction of arbitrary adversarial corruption, the error of Estimator [1](#page-0-0) degrades by only O(σ √η) from the sub-Gaussian error guaranteed in Fact [1.1.](#page-0-0) This extra error term is optimal under the assumption that the distribution has (finite) variance σ 2 .
- (B) When samples are drawn from a distribution with finite z th moment, for z ∈ (1, 2), Estimator [1](#page-0-0) achieves error matching the lower bound of [Devroye et al.](#page-9-2) [\(2016\)](#page-9-2) up to constants.
- (C) Estimator [1](#page-0-0) enjoys a fine-grained beyond-worst-case notion of optimality called *neighborhood optimality*, recently introduced by [Dang et al.](#page-8-0) [\(2023\)](#page-8-0). Previously, only medianof-means was known to be neighborhood optimal, and our result answers the question raised by Dang et al. on whether more modern estimators are also neighborhood optimal.
- (D) Under the standard finite variance and i.i.d. setting, Lee and Valiant's estimator is asymptotically normal and effi-

cient.

We emphasize that these new results hold without any changes at all to either the structure or the parameters of Estimator [1.](#page-0-0) That is, the estimator does not need to know that it is being expected to perform in these challenging settings. The guarantees we show also smoothly revert to Fact [1.1](#page-0-0) as the amount of corruption decreases to 0, or as the z th-moment assumption tends to z → 2.

These results, in aggregate, demonstrate that Lee and Valiant's estimator is in fact "all-purpose", enjoying the same breadth of guarantees as either median-of-means or trimmed mean, while also having what is essentially the smallest possible estimation error (even in the leading constant) in the standard setting. Thus, Lee and Valiant's estimator should be used in practice over existing estimators.

## 2. Our Results at a Glance

We now formally state our technical contributions.

### 2.1. Outlier robustness

Lee and Valiant's estimator is robust against data corruption from the *strong contamination model*, the most adversarial data corruption model in the literature.

Definition 2.1 (Strong Contamination Model). Given a corruption parameter η and a distribution D on the uncorrupted data, an algorithm gets a set of n η-corrupted samples from D as follows. The algorithm specifies n and nature draws

n i.i.d. samples from D. Then, an arbitrarily powerful adversary can inspect the n samples from D, and arbitrarily replace ⌈ηn⌉ of them before giving the (new) set of samples to the algorithm, with no indication of which samples were corrupted.

In the setting of η-corruption, when we only assume that D has a finite variance σ 2 , it is well-known [\(Diakonikolas &](#page-9-5) [Kane,](#page-9-5) [2023\)](#page-9-5) that the minimum estimation error is Ω(σ √η) even if there are infinitely many samples. Our analysis of Lee and Valiant's estimator shows that it achieves this estimation error bound, as long as the amount of corruption is O(log <sup>1</sup> δ /n)—where δ, the desired failure probability, is a parameter under the user's control. Put differently, the δ parameter for Estimator [1](#page-0-0) can be viewed as "dual use", parameterizing not just the allowed failure probability of the estimator, but also its desired robustness, expressing the maximum tolerated level of corruption.

We emphasize that in the following theorem, Estimator [1](#page-0-0) does not need to know the precise value of η, the fraction of corruption. It only makes the assumption that η ≤ O(log <sup>1</sup> δ /n), or in other words, Estimator [1](#page-0-0) *adapts* to the level of corruption.

Theorem 2.2. *Given any distribution* D *with mean* µ *and variance* σ 2 *, parameters* n, δ, η > 0*, let* X˜ *be a set of* n η*-corrupted samples from* D*.*

*Suppose both* log <sup>1</sup> δ n *and* δ *are bounded by some small universal constant, and suppose* η ≤ 1 24n log <sup>1</sup> δ *. Then, with probability at least* 1 − δ *over the sampling process, Estimator [1](#page-0-0) on input* δ *and* X˜ *will output an estimate* µˆ *with error at most*

$$|\hat{\mu} - \mu| \leq \sigma \cdot \left( (\sqrt{2} + o(1)) \sqrt{\frac{\log \frac{1}{\delta}}{n}} + 222\sqrt{\eta} \right)$$

In the above theorem, Estimator [1](#page-0-0) gets δ as its input parameter, and fails with probability δ. Accordingly, the estimator error is the sum of the sub-Gaussian error term from Fact [1.1,](#page-0-0) and a new "robustness" term of O(σ √η). On the other hand, the standard robustness analysis and guarantee of trimmed mean takes a different form (see [Oliveira et al.](#page-9-6) [\(2025\)](#page-9-6))—if the desired failure probability is δ ′ and the corruption parameter is η, then the number of trimmed samples is chosen as log <sup>1</sup> <sup>δ</sup> = Θ(log <sup>1</sup> δ ′ + ηn). We give an analogous theorem for Estimator [1,](#page-0-0) where again we preserve the optimal constant of √ 2 in the sub-Gaussian error term. Comparing Theorem [2.2](#page-2-0) with Theorem [2.3,](#page-2-1) the latter asks for a failure probability δ ′ > δ, but correspondingly has a smaller sub-Gaussian error term (since log <sup>1</sup> δ ′ < log <sup>1</sup> δ ).

Theorem 2.3. *Given any distribution* D *with mean* µ *and variance* σ *, parameters* n, δ, η > 0*, let* X˜ *be a set of* n η*-corrupted samples from* D*.*

*Suppose both* log <sup>1</sup> n *and* δ *are bounded by some small universal constant, and suppose* η ≤ 9n log <sup>1</sup> δ *. Let* δ ′ *be so that* 1 3 log <sup>1</sup> <sup>δ</sup> = 1 3 log <sup>1</sup> δ ′ + 3ηn*. Then, with probability at least* 1 − δ ′ *over the sampling process, Estimator [1](#page-0-0) on input* δ *and* X˜ *will output an estimate* µˆ *with error at most*

$$|\hat{\mu} - \mu| \leq \sigma \cdot \left( (\sqrt{2} + o(1)) \sqrt{\frac{\log \frac{1}{\delta'}}{n}} + (135 + o(1)) \sqrt{\eta} \right)$$

We emphasize again that, similar to Theorem [2.2,](#page-2-0) Theorem [2.3](#page-2-1) says that Estimator [1](#page-0-0) adapts to the value of η without knowing what it is precisely. Namely, Theorem [2.3](#page-2-1) simultaneously holds for all values of δ ′ even though the algorithm is only given a fixed δ.

We sketch the proof of the above two theorems in Section [4](#page-4-0) and give full details in Appendix [A.](#page-10-0) While the √ 2 constant in the sub-Gaussian terms is tight, we expect that the constants in the robust error terms (222 and 135) might be significantly improved with a different proof strategy.

In Appendix [B,](#page-26-0) we also show that Estimator [1](#page-0-0) is robust in the Huber and Total Variation Contamination Models, in addition to the Strong Contamination Model above.

Median-of-means also has guarantees analogous to the above, but they are folklore. See the proofs of Theorem [2.2](#page-2-0) and Theorem [2.3](#page-2-1) in Appendix [A.4](#page-21-1) and Appendix [A.3](#page-19-0) respectively for the formal statements and proofs of these folklore guarantees. Trimmed mean also gives analogous robustness guarantees, but current analysis (see [Oliveira et al.](#page-9-6) [\(2025\)](#page-9-6)) requires precise knowledge of η in order to match the optimal rate. In contrast, although Estimator [1](#page-0-0) does require an upper bound on η (from the necessary assumption that η ≤ 24n log <sup>1</sup> δ ), the accuracy will gracefully improve if the actual corruption rate is less than the pessimistic upper bound.

We further stress that, while median-of-means and trimmed mean both have relatively straightforward and folklore proofs of robustness, the analysis required for the Lee and Valiant estimator is much more intricate—this is due to the fact that Lee and Valiant's estimator is (a lot) more complicated, in order to yield optimal constants in the i.i.d. setting.

#### 2.2. Low moment performance

Next, we study the performance of Lee and Valiant's estimator when given data drawn from a distribution that only has finite low moments, specifically, moments between 1 and 2.

Theorem 2.4. *Given any distribution* D *with mean* µ *and* z *th moment* M<sup>z</sup> *for some* z ∈ (1, 2)*, let* X *be a set of* n *i.i.d. samples from* D*. Then, with probability at least* 1 − δ *over the randomness of* X*, Estimator [1](#page-0-0) on input* δ *and* X

*will output an estimate* µˆ *with error at most*

$$|\hat{\mu} - \mu| \leq (M_z)^{\frac{1}{z}} \cdot (1 + o(1)) \left( c_z \frac{\log \frac{1}{\delta}}{n} \right)^{1 - \frac{1}{z}}$$

*where* <sup>c</sup><sup>z</sup> = 2(5.6) <sup>1</sup> <sup>z</sup>−<sup>1</sup> −1 *. Here, the* o(1) *term tends to* 0 *as* log <sup>1</sup> n , δ → (0, 0)*, in a manner independent of* D *and independent of* z*.*

The above result matches the lower bounds shown by [De](#page-9-2)[vroye et al.](#page-9-2) [\(2016\)](#page-9-2) up to constants. As z → 2, the guarantee converges to Fact [1.1.](#page-0-0) Analogous results (up to constants) were shown for median-of-means in [Bubeck et al.](#page-8-3) [\(2012\)](#page-8-3), but the multiplicative constant we achieve for Estimator [1,](#page-0-0) (2(5.6) <sup>1</sup> <sup>z</sup>−<sup>1</sup> −1 ) is better than that of median-ofmeans (8 √ 3(12) <sup>1</sup> <sup>z</sup>−<sup>1</sup> −1 ) across all values of z ∈ (1, 2].

#### 2.3. Neighborhood optimality

Going beyond the worst-case analysis of Estimator [1,](#page-0-0) we give a finer-grained analysis of its performance. We provide finite-sample error bounds that *optimally adapt* to the underlying distribution on an instance-by-instance basis, which is far stronger than just having optimal dependence on the *variance*.

Recent work by [Dang et al.](#page-8-0) [\(2023\)](#page-8-0) gave a first study of the fine-grained optimality of 1-dimensional mean estimators, providing upper and lower bounds that match up to constants. At a high level, they showed that sub-Gaussian error bounds are essentially all one can hope for, for any distribution with a finite mean. More specifically, they define an error rate function ϵn,δ(D) over distributions D, and prove that i) median-of-means attains this error bound; and ii) for any distribution D, there exists a "reasonable" counterpart distribution D′ such that no algorithm can distinguish between the distributions, and thus no estimator can simultaneously get error ≪ ϵn,δ(D) using samples from D, while also getting error ≪ ϵn,δ(D′ ) using samples from D′ . We define ϵn,δ(D) below. The combination of lower and upper bounds of i) and ii) is extended into a new optimality notion called *neighborhood optimality* by Dang et al.

Intuitively, these are bounds that are optimal in an instanceby-instance basis, because property (ii) shows that *no* algorithm can get error better than ϵn,δ(D) on samples from distribution D—even an algorithm *customized* specifically for distribution D—without getting unacceptably bad error on a designated nearby distribution D′ .

For concreteness, we define ϵn,δ(D) below.

Definition 2.5 [\(Dang et al.](#page-8-0) [2023\)](#page-8-0). Given a (continuous) distribution D with mean µ and a real number t ∈ [0, 1], define the t*-trimming* operation on D as follows: select a radius r such that the probability mass in [µ − r, µ + r]

equals 1 − t; then, return the distribution D conditioned on lying in [µ − r, µ + r].

Given n and δ, define the trimmed distribution D<sup>∗</sup> n,δ to be the <sup>0</sup>.<sup>45</sup> n log <sup>1</sup> δ -trimmed version of D. When δ is implicit, we may denote this as D<sup>∗</sup> n . Now define the error function ϵn,δ(D) = |µ − µ ∗ n | + σ ∗ n q log <sup>1</sup> n , where µ ∗ n and σ ∗ n are the mean and standard deviation of D<sup>∗</sup> n respectively.

Dang et al. show that median-of-means achieves error O(ϵn,δ(D)), and raises the question of whether more modern estimators such as Lee and Valiant's (Estimator [1\)](#page-0-0) also achieve this error bound and are hence neighborhood optimal. We show a more general result: in fact, every sub-Gaussian and robust estimator (satisfying a slight variant of Theorem [2.2\)](#page-2-0) achieves this error bound.

Proposition 2.6. *Let* µˆ *be an arbitrary estimator that, when given* δ > 0 *and a set of* n η*-corrupted samples from any distribution* D *with mean* µ *and variance* σ 2 *, outputs a mean estimate satisfying*

$$|\hat{\mu} - \mu| \leq O\left(\sigma\left(\sqrt{\frac{\log \frac{1}{\delta}}{n}} + \sqrt{\eta}\right)\right)$$

*with probability at least* 1 − δ 2 *over the randomness of the (uncorrupted) samples.*

*Then, the same estimator* µˆ*, on input* n *i.i.d. samples drawn from a distribution* D *with finite mean, will output an estimate with error upper bounded by* O(ϵn,δ(D)) *(as defined in Definition [2.5\)](#page-3-0) with probability at least* 1 − δ*.*

We formally prove Proposition [2.6](#page-3-1) in Appendix [C.](#page-28-0) The precondition of Proposition [2.6](#page-3-1) is implied by a mild variant of Theorem [2.2](#page-2-0) (decreasing the failure probability from δ to δ/2), which holds also for Estimator [1.](#page-0-0) As a consequence:

Corollary 2.7 (Informal). *Estimator [1](#page-0-0) is neighborhood optimal, in the sense of [Dang et al.](#page-8-0) [\(2023\)](#page-8-0).*

In Section [5,](#page-5-0) we state the formal definition of neighborhood optimality and discuss the intuition on Proposition [2.6.](#page-3-1)

#### 2.4. Asymptotic normality and efficiency

Lastly, we show that Lee and Valiant's estimator is asymptotically normal and efficient, under the standard finite variance and i.i.d. sample assumption. In particular, we show that, if the δ parameter in Estimator [1](#page-0-0) is fixed, and the number of samples n → ∞, then Estimator [1](#page-0-0) converges in probability to the sample mean at the appropriate scale. The Central Limit Theorem for the sample mean then implies the asymptotic optimality of Lee and Valiant's estimator as a corollary.

Theorem 2.8. *Let* D *be a distribution with mean* µ *and variance* σ *.*

*Let* µˆ *denote Estimator [1](#page-0-0) on input parameter* δ *and* n *i.i.d. samples from* D*. Also let* X¯ <sup>n</sup> *denote the sample mean. Then, fixing* δ *and* D *and taking* n → ∞*, we have*

$$\sqrt{n}\hat{\mu} \xrightarrow{p} \sqrt{n}\bar{X}_n$$

*that is,* | √ nµˆ− √ nX¯ n| <sup>p</sup><sup>→</sup> <sup>0</sup>*, that* √ nµ<sup>ˆ</sup> *converges to* √ nX¯ n *in probability.*

*As a corollary, by the Central Limit Theorem, we have*

$$\sqrt{n}(\hat{\mu} - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

*That is,* µˆ *is asymptotically normal and efficient.*

The above theorem contrasts with the asymptotic behavior of median-of-means, whose error—scaled by √ n—converges to N (0,(π/2)σ ) [\(Minsker,](#page-9-7) [2023\)](#page-9-7); median-of-means thus has asymptotic variance a π/2-factor larger than desired.

### 3. Related Work

Mean estimation in 1 dimension. Mean estimation, even in 1-dimension, has been studied algorithmically since the 1980s. The classic median-of-means estimator was the first big-O optimal sub-Gaussian mean estimator proposed in the literature, independently invented by different groups of authors [\(Alon et al.,](#page-8-1) [1999;](#page-8-1) [Nemirovsky & Yudin,](#page-9-3) [1983;](#page-9-3) [Jer](#page-9-4)[rum et al.,](#page-9-4) [1986\)](#page-9-4). Catoni's influential work [\(2012\)](#page-8-2) gave the first sub-Gaussian mean estimator that yields the tight multiplicative constant in its error, but under strong assumptions that either the variance is known (to extremely high accuracy) or the distribution kurtosis (normalized 4 th moment) is bounded. Followup work by [Devroye et al.](#page-9-2) [\(2016\)](#page-9-2) studied "multiple-δ" estimators, also with sharp error constants, in the same setting. More recently, [Lee & Valiant](#page-9-0) [\(2022\)](#page-9-0) constructed a sub-Gaussian mean estimator with tight constants, under the bare minimum assumption that the variance exists, and absent any extra knowledge or moment assumptions this estimator is the subject of study in the current work.

See the survey of [Lugosi & Mendelson](#page-9-8) [\(2021\)](#page-9-8) on mean estimation results prior to 2019.

In low moment settings where the underlying distribution might have infinite variance, [Bubeck et al.](#page-8-3) [\(2012\)](#page-8-3) studied the performance of median-of-means. [Devroye et al.](#page-9-2) [\(2016\)](#page-9-2) then showed lower bounds that match up to constants. Our work shows that Lee and Valiant's estimator achieves analogous results as median-of-means in these regimes (Theorem [2.4\)](#page-2-2), with sharper dependence on the z th moment, for every z ∈ (1, 2].

"Beyond worst-case analysis" of 1-d mean estimators is a new research topic of recent interest in the community. In the standard i.i.d. setting, [Dang et al.](#page-8-0) [\(2023\)](#page-8-0) characterized the optimal distribution-specific error rates up to constants, showing that median-of-means achieve such rates. Our work shows that in fact all estimators that simultaneously achieve (big-O) optimal sub-Gaussian and robust estimation error must also achieve the distribution-specific optimal error rates (Proposition [2.6\)](#page-3-1). In addition to the standard i.i.d. setting, a different line of work has also studied distributionspecific mean estimation error rates for various differential privacy settings [\(Asi & Duchi,](#page-8-4) [2020a;](#page-8-4)[b;](#page-8-5) [Huang et al.,](#page-9-9) [2021\)](#page-9-9).

Robust mean estimation. Robust statistics, the setting where part of the input data can be corrupted by an adversary, has been an active area of research in the statistics community since the 1960s [\(Huber,](#page-9-10) [1992;](#page-9-10) [Tukey,](#page-9-11) [1960\)](#page-9-11). However, it was only in the past decade that polynomial-time algorithms for these statistical problems were found. See the textbook of [Diakonikolas & Kane](#page-9-5) [\(2023\)](#page-9-5) for a detailed introduction to these recent advances. Most directly relevant to our present work are results that give simultaneously sub-Gaussian and robust mean estimators, even in arbitrary high dimensions [\(Diakonikolas et al.,](#page-9-12) [2020;](#page-9-12) [Depersin &](#page-9-13) [Lecue´,](#page-9-13) [2022;](#page-9-13) [Hopkins et al.,](#page-9-14) [2020\)](#page-9-14). Median-of-means is also known to be such a robust and sub-Gaussian estimator in 1-dimension — this is a folklore result, but see [Laforgue](#page-9-15) [et al.](#page-9-15) [\(2021\)](#page-9-15) for more details on the analysis in the robust setting. Similarly, trimmed mean also has a folklore analysis for its robustness, and see [Oliveira et al.](#page-9-6) [\(2025\)](#page-9-6) for more results on the robustness and additional properties of trimmed mean.

### 4. Outlier Robustness

In this section, we outline the proof of Theorem [2.2](#page-2-0) (restated below for the reader's convenience), which says that Estimator [1](#page-0-0) is robust against adversarial data contamination from the strong contamination model of Definition [2.1.](#page-1-0) The proof of Theorem [2.3](#page-2-1) has analogous structure.

Theorem 2.2. *Given any distribution* D *with mean* µ *and variance* σ 2 *, parameters* n, δ, η > 0*, let* X˜ *be a set of* n η*-corrupted samples from* D*.*

*Suppose both* log <sup>1</sup> n *and* δ *are bounded by some small universal constant, and suppose* η ≤ 1 24n log <sup>1</sup> δ *. Then, with probability at least* 1 − δ *over the sampling process, Estimator [1](#page-0-0) on input* δ *and* X˜ *will output an estimate* µˆ *with error at most*

$$|\hat{\mu} - \mu| \leq \sigma \cdot \left( (\sqrt{2} + o(1)) \sqrt{\frac{\log \frac{1}{\delta}}{n}} + 222\sqrt{\eta} \right)$$

The proof strategy is to bound the difference between the estimates returned by Estimator [1](#page-0-0) on (corrupted) samples X˜ versus its behavior on *uncorrupted* samples X, fixing the confidence parameter δ.

Changing the input from uncorrupted samples to corrupted samples has two effects on the resulting estimate:

1. The α "influence parameter" (as computed in Step 2 of Estimator [1\)](#page-0-0) may change. However, we show that in a certain sense, when the fraction of corruption η is small compared to log <sup>1</sup> δ /n, this corruption will not change the computed α value by much (Lemma [4.2\)](#page-5-1). We further show that artificially changing the α value by a small amount will not change the mean estimate of Step 3 by much, with high probability, when given *uncorrupted* samples.

2. For a fixed influence parameter α, corrupting the samples from X to X˜ changes the returned mean estimate. However, we show a (high probability) *lower bound* on the value α computed by the algorithm on input X˜ (Lemma [4.3\)](#page-5-2); and this lower bound on α gives us a natural *upper bound* on how much any corrupted input value can affect the final mean estimate.

We state here the two key structural lemmas (and a corresponding prerequisite definition) for the α value computed from corrupted samples X˜.

Definition 4.1. Let X = {xi} be a set of clean samples, and let X˜ = {x˜i} be the corresponding set of η-corrupted samples. Denote by α<sup>ρ</sup> the "influence parameter" computed from the clean samples so as to satisfy a version of Step 2 but with a modified right hand side ( <sup>1</sup> 3 log <sup>1</sup> <sup>δ</sup> + ρn instead of 1 3 log <sup>1</sup> δ ):

$$\sum_i \min(\alpha_\rho x_i^2, 1) = \frac{1}{3} \log \frac{1}{\delta} + \rho n$$

and denote by α˜<sup>ρ</sup> the corresponding "influence parameter" computed instead from the *corrupted* samples:

$$\sum_i \min(\tilde{\alpha}_\rho \tilde{x}_i^2, 1) = \frac{1}{3} \log \frac{1}{\delta} + \rho n$$

Lemma 4.2. *Consider an arbitrary set of samples* X *and a new sample set* X η ˜ *-corrupted from* X*. Consider also an arbitrary input parameter* δ*. Using* α˜ *to denote the influence parameter of Estimator [1](#page-0-0) on inputs* (δ, X˜)*, i.e.* α˜<sup>0</sup> *in Definition [4.1,](#page-5-3) we have*

$$\alpha_{-2\eta} \leq \tilde{\alpha} \leq \alpha_{2\eta}$$

Considering the right hand side of the condition in Step 2 of Estimator [1](#page-0-0) as expressing the level of "desired robustness": Lemma [4.2](#page-5-1) states that the modified influence parameter from η-corruption is always sandwiched between the uncorrupted influence parameters, but at slightly different levels of desired robustness. We point out that Lemma [4.2](#page-5-1) is a deterministic lemma, that always holds, regardless of the sampling over X.

Lemma 4.3. *In the setting of Lemma [4.2,](#page-5-1) suppose both* log <sup>1</sup> δ n *and* δ *are bounded by some small universal constant, and suppose* η ≤ 24n log <sup>1</sup> δ *. With probability at least* 1 − 4δ/11 *over the sampling of* n *samples* X *from a distribution* D *with variance* σ *, we have* α˜ ≥ 0.0008496η*.*

These lemmas let us bound α˜ even when given corrupted data, and relate it to the uncorrupted α; these bounds are the crucial tools needed to bound the mean estimation error in both Theorem [2.2](#page-2-0) and Theorem [2.3.](#page-2-1) Lemma [4.3](#page-5-2) is used in the proof of Theorem [2.2,](#page-2-0) and shown inside Proposition [A.22](#page-24-0) in Appendix [A.4.](#page-21-1) The proof of Theorem [2.3](#page-2-1) has an analogous lemma with slightly different parameters.

The full analysis of the outlier robustness of Estimator [1](#page-0-0) is given in Appendix [A.](#page-10-0)

### 5. Neighborhood Optimality

Neighborhood optimality is a new notion of fine-grained distribution-dependent optimality recently proposed by [Dang et al.](#page-8-0) [\(2023\)](#page-8-0). While sub-Gaussian bounds are worstcase optimal for the class of finite variance distributions, neighborhood optimality captures the extent to which estimators can beneficially adapt to the non-Gaussianity of the underlying distribution and outperform the sub-Gaussian bound.

Before we formally state the definition of neighborhood optimality, let us give some preliminary definitions.

Let P<sup>1</sup> be the entire set of all distributions with a finite first moment over R. We say that N is a neighborhood function (defined over P1) if N maps a distribution D ∈ P<sup>1</sup> to a set of distributions N(D) ⊆ P1. Intuitively, the neighborhood N(D) of D is a set of distributions that we expect an estimator to perform similarly well on (and we typically consider neighborhoods where D ∈ N(D) ). Similarly, an error function ϵ maps distributions to non-negative numbers, like the function introduced in Definition [2.5.](#page-3-0) In the later definitions, we use the notations Nn,δ and ϵn,δ to indicate their dependence on the sample complexity n and failure probability δ.

Given these two notions, we can now define the notion of a *neighborhood Pareto bound*, as a property that an error function satisfies. Essentially, the definition imposes admissibility/Pareto efficiency structure within the local neighborhood Nn,δ(D) of every distribution D ∈ P1.

Definition 5.1 (Neighborhood Pareto Bounds [\(Dang et al.,](#page-8-0) [2023\)](#page-8-0)). Let n be the number of samples and δ be the failure probability. Given a neighborhood function Nn,δ : P<sup>1</sup> → 2 P<sup>1</sup> , we say that the error function ϵn,δ(D) : P<sup>1</sup> → <sup>R</sup> + 0 is a neighborhood Pareto bound for P<sup>1</sup> with respect to Nn,δ if for all distributions D ∈ P1, *no* estimator µˆ taking n i.i.d. samples can simultaneously achieve the following two

conditions:

- For all D′ ∈ Nn,δ(D), with probability 1 − δ over the n i.i.d. samples from D′ , we have |µˆ − µD′ | ≤ ϵn,δ(D′ ).
- With probability 1 − δ over the n i.i.d. samples from D, |µˆ − µD| < ϵn,δ(D).

Neighborhood Pareto ounds essentially play the role of "lower bounds" in an optimality definition, and the strength of the result depends crucially on the choice of the neighborhood function N under consideration. As a basic observation, the strength of this definition is *monotonic* in the size of the neighborhoods returned by N: if an error function ϵ is a neighborhood Pareto bound for a neighborhood function N, then for any neighborhood function N′ such that N(D) ⊆ N′ (D) for every D ∈ P1, ϵ is also a neighborhood Pareto bound for N′ . Thus, the smaller each neighborhood is, the stronger the neighborhood Pareto bound is.

Finally, we define neighborhood optimality.

Definition 5.2 ((κ, τ )-Neighborhood Optimal Estimators [\(Dang et al.,](#page-8-0) [2023\)](#page-8-0)). Let κ > 1 be a multiplicative loss factor in estimation error, and τ > 1 be a multiplicative loss factor in sample complexity.

Given the parameters κ, τ > 1, sample complexity n, failure probability δ and neighborhood function Nn,δ, a mean estimator µˆ is (κ, τ )-neighborhood optimal with respect to Nn,δ if there exists an error function ϵn,δ(D) such that min(ϵn/τ,δ(D), ϵn,δ(D)) is a neighborhood Pareto bound[<sup>1</sup>](#page-6-0) , and µˆ gives estimation error at most κ · ϵn,δ(D) with probability at least 1 − δ when taking n i.i.d. samples from any distribution D ∈ P1.

[Dang et al.](#page-8-0) [\(2023\)](#page-8-0) showed that the error function ϵn,δ from Definition [2.5](#page-3-0) yields a neighborhood Pareto bound <sup>κ</sup> min(ϵn/τ,δ(D), ϵn,δ(D)) for an appropriate choice of neighborhood function, for some constants κ, τ > 1. Their choice of neighborhood function Nn,δ is technical; we state it in Appendix [C,](#page-28-0) and refer the reader to their paper for the justification. Based on this result, they also showed that median-of-means indeed achieves error O(ϵn,δ) from Definition [2.5](#page-3-0) and hence is neighborhood optimal by Definition [5.2.](#page-6-1) [Dang et al.](#page-8-0) [\(2023\)](#page-8-0) further raised the immediate question of whether other more-modern estimators, such as Lee and Valiant's Estimator [1,](#page-0-0) can also achieve such estimation error.

We show a general affirmative answer, that in fact, all estimators that are (up to constants) simultaneously sub-Gaussian

and optimally robust to corruption must achieve the error rate from Definition [2.5](#page-3-0) (stated formally as Proposition [2.6\)](#page-3-1), and are thus neighborhood optimal as a corollary.

Proposition 2.6. *Let* µˆ *be an arbitrary estimator that, when given* δ > 0 *and a set of* n η*-corrupted samples from any distribution* D *with mean* µ *and variance* σ 2 *, outputs a mean estimate satisfying*

$$|\hat{\mu} - \mu| \leq O \left( \sigma \left( \sqrt{\frac{\log \frac{1}{\delta}}{n}} + \sqrt{\eta} \right) \right)$$

*with probability at least* 1 − δ 2 *over the randomness of the (uncorrupted) samples.*

*Then, the same estimator* µˆ*, on input* n *i.i.d. samples drawn from a distribution* D *with finite mean, will output an estimate with error upper bounded by* O(ϵn,δ(D)) *(as defined in Definition [2.5\)](#page-3-0) with probability at least* 1 − δ*.*

The precondition of Proposition [2.6](#page-3-1) is satisfied by a variant of Theorem [2.2—](#page-2-0)with slightly smaller failure probability which holds for Estimator [1.](#page-0-0) Thus the neighborhood optimality of Estimator [1](#page-0-0) follows as a corollary of Proposition [2.6.](#page-3-1)

To see the intuition behind Proposition [2.6,](#page-3-1) recall the definition of the error rate function ϵn,δ(D) from Definition [2.5.](#page-3-0) The definition constructs a distribution D<sup>∗</sup> n,δ from D, by removing the tails of D with probability mass O(log <sup>1</sup> δ /n), and the error function ϵn,δ(D) = σ ∗ n q log <sup>1</sup> δ <sup>n</sup> + |µ − µ ∗ n | is the sub-Gaussian error for distribution D<sup>∗</sup> n,δ plus the mean difference between D and D<sup>∗</sup> n,δ. Thus, when given samples from D, one could view them as *corrupted* samples from D<sup>∗</sup> n,δ where roughly <sup>O</sup>(log <sup>1</sup> δ /n) fraction of the samples are corrupted. A sub-Gaussian and robust estimator would thus achieve good error with respect to the mean of D<sup>∗</sup> n,δ, and by triangle inequality, also with respect to the mean of D.

We present proofs of the above statements in Appendix [C.](#page-28-0) For completeness, we also provide a summary of [Dang](#page-8-0) [et al.](#page-8-0) [\(2023\)](#page-8-0)'s results. We again refer the reader to their paper for a more in-depth discussion on the intricacies of the neighborhood optimality notion.

## 6. Infinite Variance Distributions

In this section, we extend Lee and Valiant's analysis of Estimator [1](#page-0-0) to more heavy-tailed distributions. Instead of Fact [1.1,](#page-0-0) where the performance of the estimator is characterized in terms of the *variance* of the distribution D, we instead ask if we can characterize the performance of the estimator on distributions that may *not* have a finite variance but instead only have finite z th moment for some 1 < z ≤ 2. We restate our main theorem for this section, which matches

<sup>1</sup>While it is intuitive to expect that an error function decreases in n, it might not be true in general. Indeed, the function used by [Dang et al.](#page-8-0) [\(2023\)](#page-8-0) (Definition [2.5\)](#page-3-0) is not necessarily monotonic. This is the reason for the "min" in the neighborhood Pareto bound requirement.

the lower bound of [Devroye et al.](#page-9-2) [\(2016\)](#page-9-2) up to constants.

Theorem 2.4. *Given any distribution* D *with mean* µ *and* z *th moment* M<sup>z</sup> *for some* z ∈ (1, 2)*, let* X *be a set of* n *i.i.d. samples from* D*. Then, with probability at least* 1 − δ *over the randomness of* X*, Estimator [1](#page-0-0) on input* δ *and* X *will output an estimate* µˆ *with error at most*

$$|\hat{\mu} - \mu| \leq (M_z)^{\frac{1}{z}} \cdot (1 + o(1)) \left( c_z \frac{\log \frac{1}{\delta}}{n} \right)^{1 - \frac{1}{z}}$$

*where* <sup>c</sup><sup>z</sup> = 2(5.6) <sup>1</sup> <sup>z</sup>−<sup>1</sup> −1 *. Here, the* o(1) *term tends to* 0 *as* log <sup>1</sup> δ n , δ → (0, 0)*, in a manner independent of* D *and independent of* z*.*

At a high level, our analysis is a generalization of Lee and Valiant's analysis to the low-moment setting, which allows us to prove a guarantee that gracefully reduces to their main result (Fact [1.1,](#page-0-0) with the sharp constant of √ 2) as z → 2. Furthermore, our value of c<sup>z</sup> is smaller than the corresponding multiplicative constant in the analysis of median-of-means by [Bubeck et al.](#page-8-3) [\(2012\)](#page-8-3), across all values of z ∈ (1, 2].

Here we give an overview of our analysis.

Without loss of generality, from the shift-and-scale equivariance of Estimator [1,](#page-0-0) we assume the underlying distribution has mean 0 and z th moment M<sup>z</sup> = 1. The goal is to prove tailored Chernoff bounds for this estimator to show its concentration. Lee and Valiant's analysis in the finite variance setting provides two useful techniques to address obstacles described in the following subsections.

#### 6.1. Estimator [1](#page-0-0) is a sum of dependent terms

Estimator [1](#page-0-0) is a sum of *dependent* terms, due to the influence parameter α computed in Step 2 involving all the samples. This makes proving Chernoff bounds tricky, given that moment generating functions multiply only for sums of *independent* terms. Lee and Valiant's approach is to *reduce* (via a Lipschitz argument) to analyzing the case where the preliminary estimate κ from Step 1 is taken to be exactly equal to the true mean µ = 0, and crucially reformulate Estimator [1](#page-0-0) as a "2-parameter ψ-estimator".

Definition 6.1 [\(Lee & Valiant](#page-9-0) [2022\)](#page-9-0). Consider Estimator [1](#page-0-0) but with Step 1 replaced with "κ = 0". The estimator can be equivalently expressed as follows:

- 1. Input: Failure probability δ, independent samples X = x1, . . . , x<sup>n</sup>
- 2. Solve for the (unique) pair (ˆµ, αˆ) satisfying ψ<sup>µ</sup> = 0 and ψ<sup>α</sup> = 0, where the functions ψµ, ψ<sup>α</sup> are defined

as follows:

$$\psi_\mu(X, \hat{\mu}, \hat{\alpha}) = \sum_{i=1}^n (\hat{\mu} - x_i(1 - \min(\hat{\alpha}x_i^2, 1)));$$

$$\psi_\alpha(X, \hat{\mu}, \hat{\alpha}) = \sum_{i=1}^n \left( \min(\hat{\alpha}x_i^2, 1) - \frac{1}{3n} \log \frac{1}{\delta} \right)$$

#### 3. Output: µˆ from the previous step.

This reformulation has the advantage that, for any fixed pair (ˆµ, αˆ), any linear combination of the ψ<sup>µ</sup> and ψ<sup>α</sup> functions is a sum of independent terms. The concentration of Estimator [1](#page-0-0) is then reduced to proving Chernoff bounds for these linear combinations.

This reformulation and reduction is independent of the finite variance assumption, and therefore also applicable to the low moment setting that our work analyzes.

#### 6.2. Proving Chernoff bounds over large distribution classes

Even in the finite variance setting, proving a Chernoff bound that applies for all distributions D with mean 0 and variance 1 is daunting, given how large the distribution class is compared to standard concentration bounds. Lee and Valiant showed that the worst-case Chernoff bound (for linear combinations of the ψ-equations from Definition [6.1\)](#page-7-0) can in fact be viewed as a max-min linear programming game.

For simplicity, let us illustrate this by sketching the analysis of the Chernoff bound of a hypothetical linear estimator that is a sum of independent terms: f({x1, . . . , xn}) = n P i f(xi), for some fixed function f : <sup>R</sup> → <sup>R</sup>. Proving a Chernoff bound is equivalent to upper bounding the moment generating function of the estimator f, and choosing the "Chernoff parameter" t accordingly. Thus, it suffices to upper bound the objective of the following max-min game, where the max player chooses any mean-0 variance-1 distribution D—represented by variables {px} denoting the probability mass at x (ignoring probability formalism issues with non-discrete distributions)—and the min player chooses the Chernoff parameter t. Using the moment generating function as the objective function, we have

$$\begin{aligned}
 \max_{\{p_x\}} & \min_t \sum_x p_x e^{t \cdot f(x)} \\
 \text{such that} & \sum_x p_x = 1 \\
 & \sum_x p_x \cdot x = 0 \\
 & \sum_x p_x \cdot x^2 = 1 \\
 \text{where} & p_x \geq 0
 \end{aligned} \tag{1}$$

By using minimax duality and linear programming duality, the game can then be rewritten into a pure minimization program with the same optimum, where the dual variables

U, M, V correspond to the 3 respective constraints in the program in [\(1\)](#page-7-1).

$$\begin{aligned}
 \text{min}_t & = \min_{U, M, V} U + V \\
 \text{such that} & \text{for all } x \in \mathbb{R}, Vx^2 + Mx + U \geq e^{t \cdot f}(x)
 \end{aligned} \tag{2}$$

It thus suffices to choose dual variables U, M, V and an appropriate Chernoff parameter t to certify an upper bound on the optimum.

We modify this approach from [Lee & Valiant](#page-9-0) [\(2022\)](#page-9-0) by relying on a z th moment bound instead of a variance bound. The key observation is that the z th moment bound may be expressed as a linear constraint in the above program, replacing P x pxx <sup>2</sup> = 1 with P x px|x| <sup>z</sup> = 1. The technical challenge from here is to provide a feasible dual solution and choose Chernoff parameter t so as to satisfy the desired bounds, including that the guarantees converge to Fact [1.1](#page-0-0) as z → 2. We show our complete proof in Appendix [D.](#page-30-0)

### 7. Asymptotic Normality

In this section, we show that under the standard finite variance assumption, the estimator of Lee and Valiant is asymptotically *normal* and *efficient*. Specifically, we prove that if we fix the input parameter δ and take the number of samples n → ∞, the estimator converges to the sample mean *in probability*, which by the Central Limit Theorem implies asymptotic normality and efficiency.

This result contrasts with median-of-means, which, under the slightly stronger 2+ι moment assumption for any ι > 0, is asymptotically normal yet *inefficient* [\(Minsker,](#page-9-7) [2023\)](#page-9-7) the asymptotic distribution of √ nµˆMoM is N (µ,(π/2)σ 2 ) instead of the desired N (µ, σ<sup>2</sup> ).

Theorem 2.8. *Let* D *be a distribution with mean* µ *and variance* σ *.*

*Let* µˆ *denote Estimator [1](#page-0-0) on input parameter* δ *and* n *i.i.d. samples from* D*. Also let* X¯ <sup>n</sup> *denote the sample mean. Then, fixing* δ *and* D *and taking* n → ∞*, we have*

$$\sqrt{n}\hat{\mu} \xrightarrow{p} \sqrt{n}\bar{X}_n$$

*that is,* | √ nµˆ− √ nX¯ n| <sup>p</sup><sup>→</sup> <sup>0</sup>*, that* √ nµ<sup>ˆ</sup> *converges to* √ nX¯ n *in probability.*

*As a corollary, by the Central Limit Theorem, we have*

$$\sqrt{n}(\hat{\mu} - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

*That is,* µˆ *is asymptotically normal and efficient.*

The proof is relatively straightforward. The key idea is that Estimator [1](#page-0-0) differs from the sample mean by removing Θ(log <sup>1</sup> δ ) weighted samples, so we might as well bound the difference by Θ(log <sup>1</sup> δ ) times the maximum sample (and symmetrically the minimum sample), multiplied by a factor of √ n since that is the scale that the Central Limit Theorem holds at. Under the finite variance assumption, we can use a (slightly refined) Chebyshev's inequality and a standard Chernoff bound to upper bound the magnitude of the maximum sample with high probability. See the complete calculations in Appendix [E.](#page-38-0)

### Acknowledgments

The work of Jasper C.H. Lee was done in part while he was at UW Madison, supported by NSF Medium Award CCF-2107079. He also thanks Stanislav Minsker for discussions on the asymptotic normality result.

The work of Walter McKelvie was partly supported by the National Science Foundation Graduate Research Fellowship Program under Grant No. 2140743.

Maoyuan Song and Paul Valiant are partially supported by NSF award CCF-2127806 and by Office of Naval Research award N000142412695.

Maoyuan Song is partially supported by NSF award CCF-2228814.

## Impact Statement

This work studies a fundamental statistical problem that is broadly applicable to a wide variety of domains. As such, it does not directly raise any societal or ethical concerns that warrant special consideration.

## References


[1] Alon, N., Matias, Y., and Szegedy, M. The space complexity of approximating the frequency moments. *J. Comput. Syst. Sci*, 58(1):137–147, 1999. Asi, H. and Duchi, J. C. Near instance-optimality in differential privacy. *arXiv preprint arXiv:2005.10630*, 2020a. Asi, H. and Duchi, J. C. Instance-optimality in differential privacy via approximate inverse sensitivity mechanisms. *Advances in neural information processing systems*, 33: 14106–14117, 2020b. Bubeck, S., Cesa-Bianchi, N., and Lugosi, G. Bandits with heavy tail. *arXiv preprint arXiv:1209.1727*, 2012. Catoni, O. Challenging the empirical mean and empirical variance: a deviation study. *Ann. I. H. Poincare -PR ´* , 48 (4):1148–1185, 2012. Dang, T., Lee, J. C. H., Song, M., and Valiant, P. Optimality in mean estimation: Beyond worst-case, beyond subgaussian, and beyond 1 + α moments. In *Thirty-seventh*

[2] *Conference on Neural Information Processing Systems*, 2023. Depersin, J. and Lecue, G. Robust sub-gaussian estimation ´ of a mean vector in nearly linear time. *The Annals of Statistics*, 50(1):511–536, 2022. Devroye, L., Lerasle, M., Lugosi, G., and Oliveira, R. I. Sub-Gaussian mean estimators. *Ann. Stat*, 44(6):2695–2725, 2016. Diakonikolas, I. and Kane, D. M. *Algorithmic highdimensional robust statistics*. Cambridge university press, 2023. Diakonikolas, I., Kane, D. M., and Pensia, A. Outlier robust mean estimation with subgaussian rates via stability. *Advances in Neural Information Processing Systems*, 33: 1830–1840, 2020. Gobet, E., Lerasle, M., and Metivier, D. Mean estimation ´ for randomized quasi monte carlo method. *Hal preprint hal-03631879v2*, 2022. Hopkins, S., Li, J., and Zhang, F. Robust and heavy-tailed mean estimation made simple, via regret minimization. *Advances in Neural Information Processing Systems*, 33: 11902–11912, 2020. Huang, Z., Liang, Y., and Yi, K. Instance-optimal mean estimation under differential privacy. *Advances in Neural Information Processing Systems*, 34:25993–26004, 2021. Huber, P. J. Robust estimation of a location parameter. In *Breakthroughs in statistics: Methodology and distribution*, pp. 492–518. Springer, 1992. Jerrum, M. R., Valiant, L. G., and Vazirani, V. V. Random generation of combinatorial structures from a uniform distribution. *Theor. Comput. Sci*, 43:169–188, 1986. Laforgue, P., Staerman, G., and Clemen ´ c¸on, S. Generalization bounds in the presence of outliers: a median-ofmeans study, 2021. Lee, J. C. H. and Valiant, P. Optimal sub-Gaussian mean estimation in R. In *2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS)*, pp. 672–

683. IEEE, 2022. Lugosi, G. and Mendelson, S. Robust multivariate mean estimation: The optimality of trimmed mean. *The Annals of Statistics*, 49(1):393 – 410, 2021. Minsker, S. U-statistics of growing order and sub-gaussian mean estimators with sharp constants. *arXiv preprint arXiv:2202.11842*, 2023. Nemirovsky, A. and Yudin, D. *Problem Complexity and Method Efficiency in Optimization*. Wiley, 1983. Oliveira, R. I., Orenstein, P., and Rico, Z. F. Finitesample properties of the trimmed mean. *arXiv preprint arXiv:2501.03694*, 2025. Tukey, J. W. A survey of sampling from contaminated distributions. *Contributions to probability and statistics*, pp. 448–485, 1960.
### A. Remaining Proofs of Section [4](#page-4-0)

In Section [4,](#page-4-0) we discussed the intuition behind our main results, Theorem [2.2](#page-2-0) and [2.3.](#page-2-1) At a high level, our proof strategy for both main theorems uses the triangle inequality to bound estimation error introduced by adversarial corruption as the sum of two parts, one from changing the influence parameter from α to α˜, and one from the adversary arbitrarily corrupting the samples. We provide formal proofs for relevant lemmas and propositions in this section, and restate the main theorems for completeness:

Theorem 2.2. *Given any distribution* D *with mean* µ *and variance* σ 2 *, parameters* n, δ, η > 0*, let* X˜ *be a set of* n η*-corrupted samples from* D*.*

*Suppose both* log <sup>1</sup> δ n *and* δ *are bounded by some small universal constant, and suppose* η ≤ 24n log <sup>1</sup> δ *. Then, with probability at least* 1 − δ *over the sampling process, Estimator [1](#page-0-0) on input* δ *and* X˜ *will output an estimate* µˆ *with error at most*

$$|\hat{\mu} - \mu| \leq \sigma \cdot \left( (\sqrt{2} + o(1)) \sqrt{\frac{\log \frac{1}{2}}{n}} + 222\sqrt{\eta} \right)$$

Theorem 2.3. *Given any distribution* D *with mean* µ *and variance* σ 2 *, parameters* n, δ, η > 0*, let* X˜ *be a set of* n η*-corrupted samples from* D*.*

*Suppose both* log <sup>1</sup> n *and* δ *are bounded by some small universal constant, and suppose* η ≤ 9n log <sup>1</sup> δ *. Let* δ ′ *be so that* 1 3 log <sup>1</sup> <sup>δ</sup> = 3 log <sup>1</sup> δ ′ + 3ηn*. Then, with probability at least* 1 − δ ′ *over the sampling process, Estimator [1](#page-0-0) on input* δ *and* X˜ *will output an estimate* µˆ *with error at most*

$$|\hat{\mu} - \mu| \leq \sigma \cdot \left( (\sqrt{2} + o(1)) \sqrt{\frac{\log \frac{1}{\delta}}{n}} + (135 + o(1)) \sqrt{\eta} \right)$$

Throughout the proofs, we will compare and make use of different values of "α", computed from either corrupted or uncorrupted data, and computed from different choices of parameters in the equation defining α. Here, we give a more general definition of notation (generalizing Definition [4.1\)](#page-5-3) that we will be using.

Definition A.1. Let X = {xi} be a set of clean samples, and let X˜ = {x˜i} be the corresponding set of η-corrupted samples. Denote by αδ,ρ the "influence parameter" solved from the corresponding condition involving the clean samples, so as to satisfy a version of Step 2 of Estimator [1](#page-0-0) but with a modified right hand side ( <sup>1</sup> 3 log <sup>1</sup> <sup>δ</sup> <sup>+</sup> ρn instead of <sup>1</sup> 3 log <sup>1</sup> δ ):

$$\sum_i \min(\alpha_{\delta,\rho} x_i^2, 1) = \frac{1}{3} \log \frac{1}{\delta} + \rho n$$

and denote by α˜δ,ρ the "influence parameter" solved instead from the corresponding condition involving the *corrupted* samples:

$$\sum_i \min(\tilde{\alpha}_{\delta,\rho}, \tilde{x}_i^2, 1) = \frac{1}{3} \log \frac{1}{\delta} + \rho n$$

Theorem [2.3](#page-2-1) refers to failure probability δ ′ , so parts of the analysis will involve the notations α<sup>δ</sup> ,η and α˜<sup>δ</sup> ,η, for example.

We start with showing the crucial preliminary Lemma [A.2,](#page-10-1) which bounds the value of α˜, allowing us to approximate it using different influence parameters on the *clean* data, with no assumption of the *corrupted* data:

Lemma A.2 (Restatement of Lemma [4.2](#page-5-1) under the notation of Definition [A.1\)](#page-10-2). *Consider an arbitrary set of samples* X *and a new sample set* X η ˜ *-corrupted from* X*. Consider also an arbitrary input parameter* δ*. Using* α˜ *to denote the influence parameter of Estimator [1](#page-0-0) on inputs* (δ, X˜)*, i.e.* α˜δ,<sup>0</sup> *in Definition [A.1,](#page-10-2) we have*

$$\alpha_{\delta,-2\eta} \leq \tilde{\alpha} \leq \alpha_{\delta,2\eta}$$

To prove the first inequality, suppose for the sake of contradiction that α < α ˜ δ,−2η. Then,

$$\begin{aligned} & \sum_i \min(\tilde{\alpha}\tilde{x}_i^2, 1) \\ & \leq \sum_i \min(\tilde{\alpha}x_i^2, 1) + \sum_{i: x_i \text{ corrupted}} \min(\tilde{\alpha}\tilde{x}_i^2, 1) \\ & \leq \frac{1}{3} \log \frac{1}{\delta} - 2\eta n + \sum_{i: x_i \text{ corrupted}} \min(\tilde{\alpha}\tilde{x}_i^2, 1) \quad (\text{since } \tilde{\alpha} < \alpha_{\delta, -2\eta}) \\ & \leq \frac{1}{3} \log \frac{1}{\delta} - \eta n \quad (\text{since the sum has } \eta n \text{ elements, each at most 1}) \\ & < \frac{1}{3} \log \frac{1}{\delta} \end{aligned}$$

which is a contradiction.

The second inequality follows similarly. Suppose for the sake of contradiction that α > α ˜ δ,2η. Then,

$$\begin{aligned} & \sum_i \min(\tilde{\alpha}\tilde{x}_i^2, 1) \\ & \geq \sum_i \min(\tilde{\alpha}x_i^2, 1) - \sum_{i: x_i \text{ corrupted}} |\min(\tilde{\alpha}\tilde{x}_i^2, 1) - \min(\tilde{\alpha}x_i^2, 1)| \\ & \geq \frac{1}{3} \log \frac{1}{\delta} + 2\eta n - \sum_{i: x_i \text{ corrupted}} |\min(\tilde{\alpha}\tilde{x}_i^2, 1) - \min(\tilde{\alpha}x_i^2, 1)| \quad (\text{since } \tilde{\alpha} > \alpha_{\delta, 2\eta}) \\ & \geq \frac{1}{3} \log \frac{1}{\delta} + \eta n \quad (\text{since the sum has } \eta n \text{ elements, each at most 1}) \\ & > \frac{1}{3} \log \frac{1}{\delta} \end{aligned}$$

which is a contradiction.

We in fact generalize the above lemma slightly for use in the proof of Theorem [2.3,](#page-2-1) which can be proven from Lemma [A.2](#page-10-1) by reparameterizing δ.

Corollary A.3. *For any set of clean samples* X *and the corresponding* η*-corrupted samples* X˜*, and for any constant* c > 2*, we have* αδ,(c−2)<sup>η</sup> ≤ α˜δ,cη ≤ αδ,(c+2)η*.*

The proof structure of Theorem [2.2](#page-2-0) and [2.3](#page-2-1) are essentially identical. We present the proof of Theorem [2.3](#page-2-1) first.

For the rest of the appendix, we will assume that the uncorrupted data distribution has mean 0 and variance 1 without loss of generality, due to the shift-and-scale equivariance of Estimator [1.](#page-0-0)

#### A.1. Bounding the Error due to Changing the Influence Parameter

We present the following proposition upper bounding the error incurred on Estimator [1](#page-0-0) by using influence parameter α˜ := ˜α<sup>δ</sup> ′ ,3<sup>η</sup> instead of α := α<sup>δ</sup> ′ ,<sup>0</sup> on the *clean* samples. That is, we compute α˜ on the corrupted samples, but analyze its effect on the clean samples.

For the following section, recall our assumption that the underlying distribution has mean 0 and variance 1.

Proposition A.4. *Suppose both* log <sup>1</sup> δ′ n *and* δ ′ *are bounded by some small universal constant. Let* α *be the influence parameter computed from the* clean *samples with robustness level* <sup>1</sup> 3 log <sup>1</sup> δ *, namely* α := α<sup>δ</sup> ′ ,0*. Let* α˜ *be the influence parameter computed from the* corrupted *samples with robustness level* <sup>1</sup> 3 log <sup>1</sup> δ ′ + 3ηn*, namely,* α˜ := ˜α<sup>δ</sup> ′ ,3η*. Then with probability at least* 1 − 6 8 δ ′ *, the mean estimate using* α *on the clean samples differs from the mean estimate using* α˜ *on the* clean *samples by at most* 125.5 √<sup>η</sup>*, i.e.,*

$$\left| \sum_i x_i \min(\tilde{\alpha}x_i^2, 1) - \sum_i x_i \min(\alpha x_i^2, 1) \right| \leq 125.5n\sqrt{\eta} \quad (3)$$

To provide some intuition towards our proof strategy for Proposition [A.4,](#page-11-0) first notice that we can bound the left hand side via Cauchy-Schwarz as

$$\sqrt{\left(\sum_i x_i^2\right) \left(\sum_i (\min(\tilde{\alpha} x_i^2, 1) - \min(\alpha x_i^2, 1))^2\right)}$$

This turns out to be insufficient; we instead bound [\(3\)](#page-11-1) by defining the set S of indices where min(˜αx<sup>2</sup> i , 1) ̸= min(αx<sup>2</sup> i , 1), and restrict the range of both sums in [\(3\)](#page-11-1) to the range i ∈ S, since doing so only discards zero terms and does not change the sum. Thus we instead have the Cauchy-Schwarz bound

$$\sqrt{\left(\sum_{i \in S} x_i^2\right) \left(\sum_{i \in S} (\min(\tilde{\alpha} x_i^2, 1) - \min(\alpha x_i^2, 1))^2\right)}$$

The bound on the second parenthetical makes crucial use of the comparison of α˜ and α provided by Corollary [A.3.](#page-11-2) The first parenthetical is an empirical variance, but the restriction i ∈ S means that |x<sup>i</sup> | cannot be too large; we thus use Bernstein's inequality to bound this S-truncated empirical second moment, in terms of a lower bound on α, which we prove next.

To show our lower bound on α, we first calculate the following straightforward relations between the empirical and population quantiles.

Throughout this section, we denote by Qq(D) the q (true) quantile of D, i.e., <sup>P</sup>[D ≤ Qq(D)] = q.

Lemma A.5. *Suppose both* log <sup>1</sup> δ′ n *and* δ ′ *are bounded by some small universal constant. Denote* c<sup>1</sup> = 0.277 < 1 3 *, and* κ := c1( 1 n log <sup>1</sup> δ ′ )*. Let constant* c<sup>2</sup> := 102.907*. Then the* 1 − κ *empirical quantile of* x<sup>i</sup> *is at most* Q1−κ/c<sup>2</sup> (D) *with probability at least* 1 − 1 8 δ ′ *.*

*Proof.* For the 1 − κ empirical quantile of x<sup>i</sup> to be greater than Q1−κ/c<sup>2</sup> (D), there has to be more than κn samples greater than Q1−κ/c<sup>2</sup> (D). Thus, it suffices to prove that |{i ∈ [n] : x<sup>i</sup> ≥ Q1−κ/c<sup>2</sup> (D)}| ≥ κn with probability at most <sup>1</sup> 8 δ ′ .

Denote Z<sup>i</sup> := 1<sup>x</sup>i≥Q1−κ/c<sup>2</sup> (D) . Then Z := |{i ∈ [n] : x<sup>i</sup> ≥ Q1−κ/c<sup>2</sup> (D)}| = P<sup>n</sup> <sup>i</sup>=1 Z<sup>i</sup> . We denote by p = κ c<sup>2</sup> the probability that an individual i is in this set; and thus E[Z] = pn. Since each Z<sup>i</sup> is a coin flip of probability p, we further have that V ar[Z<sup>i</sup> ] = p(1 − p).

By multiplicative Chernoff,

$$\begin{aligned}\mathbb{P}[Z \geq c_2 pn] &\leq \left( \frac{e^{c_2-1}}{c_2^2} \right)^{pn} \\ &= \exp((c_2 - 1 - c_2 \log c_2)pn) \\ &= \exp\left(\frac{c_1(c_2 - 1 - c_2 \log c_2)}{c_2} \log \frac{1}{\delta}\right) \\ &\leq \exp\left((-1.01)\log \frac{1}{\delta'}\right) \quad \text{by choice of } c_1 \text{ and } c_2 \\ &\leq \exp\left(-\log \frac{8}{\delta'}\right) = \frac{1}{8}\delta' \quad \text{since } 1.01\log \frac{1}{\delta'} \geq \log \frac{8}{\delta'} \text{ for suff. small } \delta'\end{aligned}$$

as desired.

By symmetry, we have the following corollary as well:

Corollary A.6. *The* κ *empirical quantile of* x<sup>i</sup> *is at least* Qκ/c<sup>2</sup> (D) *with probability at least* 1 − 8 δ ′ *.*

Lemma A.7. *Let* Dtrimmed *denote a "trimmed" version of* D*: namely,* D *conditioned on lying in* [Qκ/c<sup>2</sup> (D), Q1−κ/c<sup>2</sup> (D)]*. Then* <sup>E</sup>[D<sup>2</sup> trimmed] ≤ c 2 2 (c2−2κ) 2 *.*

*Proof.* Denote by <sup>1</sup>untrimmed(x) the indicator that returns 1 if x ∈ [Qκ/c<sup>2</sup> (D), Q1−κ/c<sup>2</sup> (D)] and 0 otherwise. Then observe that Dtrimmed = c<sup>2</sup> c2−2κ (D · <sup>1</sup>untrimmed). Thus

$$\begin{aligned}\mathbb{E}[D_{trimmed}^2] &= \mathbb{E}\left[\left(\frac{c_2}{c_2 - 2\kappa}(D \cdot \mathbb{1}_{untrimmed})\right)^2\right] \\ &= \frac{c_2^2}{(c_2 - 2\kappa)^2} \mathbb{E}[D^2 \mathbb{1}_{untrimmed}] \\ &\leq \frac{c_2^2}{(c_2 - 2\kappa)^2} \mathbb{E}[D^2] \\ &= \frac{c_2^2}{(c_2 - 2\kappa)^2}\end{aligned}$$

as desired.

Lemma A.8. *Suppose both* log <sup>1</sup> δ′ n *and* δ ′ *are bounded by some small universal constant. Let* (xtrimmed)<sup>i</sup> *denote the sample* x<sup>i</sup> *after trimming, namely: let* (xtrimmed)<sup>i</sup> = x<sup>i</sup> *if* x<sup>i</sup> ∈ [Qκ/c<sup>2</sup> (D), Q1−κ/c<sup>2</sup> (D)]*, and* (xtrimmed)<sup>i</sup> = 0 *otherwise. Let constant* c<sup>3</sup> := 251.099*. Then* P i (xtrimmed) 2 <sup>i</sup> ≤ (c3+1)c 2 2 (c2−2κ) <sup>2</sup> n *with probability at least* 1 − 2 8 δ ′ *.*

P *Proof.* First notice that if we replace any trimmed x<sup>i</sup> with a random sample according to Dtrimmed, then the sum i (xtrimmed) 2 i can only increase. Thus, to prove the claim, it suffices to bound the sum of n i.i.d. samples from Dtrimmed. With an abuse of notation, we let {(xtrimmed)i}i≤<sup>n</sup> denote a set of such samples.

Also notice that by our choice of <sup>c</sup>3, we have, crucially, <sup>3</sup>c1<sup>c</sup> 2 3 (6+2c3)c<sup>2</sup> ≥ 1.01.

We start by bounding Qκ/c<sup>2</sup> (D) and Q1−κ/c<sup>2</sup> (D). Since we assume D has mean 0 and variance 1, by Chebyshev's inequality, P -|D| ≥ p<sup>c</sup><sup>2</sup> κ ≤ κ c<sup>2</sup> , which implies that Qκ/c<sup>2</sup> ≥ −p<sup>c</sup><sup>2</sup> κ and Q1−κ/c<sup>2</sup> ≤ p<sup>c</sup><sup>2</sup> κ . As a result, (xtrimmed) 2 <sup>i</sup> ≤ c<sup>2</sup> κ for all i.

Then,

$$\begin{aligned} \text{Var}[(x_{\text{trimmed}})_i^2] &\leq \mathbb{E}[((x_{\text{trimmed}})_i^2)^2] \\ &\leq \frac{c_2}{\kappa} \mathbb{E}[(x_{\text{trimmed}})_i^2] \\ &\leq \frac{c_2^3}{\kappa(c_2 - 2\kappa)^2} \quad \text{by Lemma A.7} \end{aligned}$$

Thus, by Bernstein's inequality,

$$\begin{aligned} & \mathbb{P} \left[ \sum_i ((x_{\text{trimmed}})_i)^2 \geq \frac{(c_3 + 1)c_2^2 n}{(c_2 - 2\kappa)^2} \right] \\ &= \mathbb{P} \left[ \frac{1}{n} \sum_i ((x_{\text{trimmed}})_i)^2 - \frac{c_2^2}{(c_2 - 2\kappa)^2} \geq \frac{c_3 c_2^2}{(c_2 - 2\kappa)^2} \right] \\ &\leq \mathbb{P} \left[ \frac{1}{n} \sum_i ((x_{\text{trimmed}})_i)^2 - \mathbb{E}[((x_{\text{trimmed}})_i)^2] \geq \frac{c_3 c_2^2}{(c_2 - 2\kappa)^2} \right] \\ &\leq 2 \exp \left( - \frac{c_3^2 n \frac{c_2^4}{(c_2 - 2\kappa)^4}}{2 \text{Var}[(x_{\text{trimmed}})_i^2] + \frac{2c_3 c_2^3}{3\kappa(c_2 - 2\kappa)^2}} \right) \\ &\leq 2 \exp \left( - \frac{c_3^2 n \frac{c_2^4}{(c_2 - 2\kappa)^4}}{\frac{2c_3^2}{\kappa(c_2 - 2\kappa)^2} + \frac{2c_3 c_2^3}{3\kappa(c_2 - 2\kappa)^2}} \right) \end{aligned}$$

$$\begin{aligned} &= 2 \exp \left( -\frac{3c_3^2 c_2 n \kappa}{(6 + 2c_3)(c_2 - 2\kappa)^2} \right) \\ &\leq \exp \left( -\frac{3c_3^2 n \kappa}{(6 + 2c_3)c_2} \right) \\ &= 2 \exp \left( -\frac{3c_1 c_3^2}{(6 + 2c_3)c_2} \log \frac{1}{\delta'} \right) \quad \text{by definition of } \kappa \\ &\leq 2 \exp \left( -1.01 \log \frac{1}{\delta} \right) \quad \text{by choice of } c_3 \\ &\leq 2 \exp \left( -\log \frac{8}{\delta'} \right) \quad \text{since } 1.01 \log \frac{1}{\delta} \geq \log \frac{8}{\delta'} \text{ for suff. small } \delta' \\ &= \frac{2}{8} \delta' \end{aligned}$$

as desired.

Combining Lemma [A.5,](#page-12-1) Corollary [A.6,](#page-12-2) and Lemma [A.8,](#page-13-0) we have the following corollary:

Corollary A.9. *Suppose there is a sufficiently small constant that upper bounds* δ ′ *. Let* S<κ *denote the set of indices* i *s.t.* x 2 i *is not in the top* κ *(empirical) quantile. Then* P i∈S<κ x 2 <sup>i</sup> ≤ (c3+1)c 2 (c2−2κ) <sup>2</sup> n *with probability at least* 1 − 4 8 δ ′ *.*

We are now ready to present a lower bound on α := α<sup>δ</sup> ′ ,0, before proving Proposition [A.4.](#page-11-0)

Lemma A.10. *Suppose both* log <sup>1</sup> δ′ n *and* δ ′ *are bounded by some small universal constant. Then* α := α<sup>δ</sup> ′ ,<sup>0</sup> ≥ 0.000214 <sup>1</sup> n log <sup>1</sup> δ ′ *with probability at least* 1 − 4 8 δ ′ *.*

*Proof.* Recall that by definition of α,

$$\begin{aligned} \frac{1}{3} \log \frac{1}{\delta'} &= \sum_i \min(\alpha x_i^2, 1) \\ &= \sum_{\substack{i: x_i^2 \text{ not in the top } \kappa \text{ quantile} \\ i: x_i^2 \text{ in the top } \kappa \text{ quantile}}} \min(\alpha x_i^2, 1) + \sum_{\substack{i: x_i^2 \text{ in the top } \kappa \text{ quantile} \\ i: x_i^2 \text{ in the top } \kappa \text{ quantile}}} \min(\alpha x_i^2, 1) \\ &\leq \sum_{\substack{i: x_i^2 \text{ not in the top } \kappa \text{ quantile} \\ i: x_i^2 \text{ in the top } \kappa \text{ quantile}}} \alpha x_i^2 + \sum_{\substack{i: x_i^2 \text{ in the top } \kappa \text{ quantile} \\ i: x_i^2 \text{ in the top } \kappa \text{ quantile}}} 1 \\ &\leq \sum_{\substack{i: x_i^2 \text{ not in the top } \kappa \text{ quantile} \\ i: x_i^2 \text{ in the top } \kappa \text{ quantile}}} \alpha x_i^2 + \kappa n \\ &= \sum_{\substack{i: x_i^2 \text{ not in the top } \kappa \text{ quantile} \\ i: x_i^2 \text{ in the top } \kappa \text{ quantile}}} \alpha x_i^2 + c_1 \log \frac{1}{\delta'} \end{aligned}$$

Rearranging, this is equivalent to

$$\left(\frac{1}{3} - c_1\right) \log \frac{1}{\delta'} \leq \sum_{i: x_i^2 \text{ not in the top } \kappa \text{ quantile}} \alpha x_i^2$$

By Corollary [A.9,](#page-14-0) with probability at least 1 − 4 8 δ ′ ,

$$\sum_{i: x_i^2 \text{ not in the top } \kappa \text{ quantile}} x_i^2 \leq \frac{(c_3 + 1)c_2^2}{(c_2 - 2\kappa)^2} n \leq \frac{(c_3 + 1)c_2^2}{(c_2 - 2)^2} n$$

Which implies that α ≥ 1 <sup>3</sup> − c<sup>1</sup> (c2−2)<sup>2</sup> (c3+1)c n log <sup>1</sup> δ ′ ≥ 0.000214 <sup>1</sup> n log <sup>1</sup> δ ′ with probability at least 1 − 4 8 δ ′ . *Proof of Proposition [A.4.](#page-11-0)* First, notice that for all i such that |x<sup>i</sup> | ≥ q 1 <sup>α</sup> ≥ q 1 α˜ , the corresponding term in the sum in the left hand side becomes 0. Thus, using the notation P <sup>≤</sup> to denote summing over elements |x<sup>i</sup> | ≤ q 1 α˜ , the left hand side in the guarantee of Proposition [A.4](#page-11-0) is equal to

$$\left| \sum_{\leq} x_i \min(\tilde{\alpha} x_i^2, 1) - \sum_{\leq} x_i \min(\alpha x_i^2, 1) \right|$$

Then, rearranging the sums, we have

$$\begin{aligned} & \left| \sum_{\leq} x_i \min(\tilde{\alpha}x_i^2, 1) - \sum_{\leq} x_i \min(\alpha x_i^2, 1) \right| \\ & \leq \sum_{\leq} |x_i (\min(\tilde{\alpha}x_i^2, 1) - \min(\alpha x_i^2, 1))| \\ & \leq \sqrt{\left( \sum_{\leq} x_i^2 \right) \left( \sum_{\leq} (\min(\tilde{\alpha}x_i^2, 1) - \min(\alpha x_i^2, 1))^2 \right)} \quad \text{by Cauchy-Schwarz} \end{aligned}$$

for which we can bound the two terms separately.

To bound the first term, since we sum over only those terms where |x<sup>i</sup> | ≤ q 1 α for all i, and by Lemma [A.10](#page-14-1) α ≥ 0.000214 n log <sup>1</sup> δ ′ with probability 1 − 4 8 δ ′ , we have that x 2 <sup>i</sup> ≤ 4672n log <sup>1</sup> δ′ for all i. Since X has mean 0 and variance 1, we know that <sup>E</sup>[x 2 i ] ≤ <sup>E</sup>[X<sup>2</sup> ] = 1, and Var[x 2 i ] ≤ <sup>E</sup>[x 4 i ] ≤ 4672n log <sup>1</sup> δ′ <sup>E</sup>[x 2 i ] ≤ 4672n log <sup>1</sup> δ′ . Thus, by Bernstein's inequality,

$$\begin{aligned} \mathbb{P} \left[ \sum_{\leq} x_i^2 \geq 3150n \right] &= \mathbb{P} \left[ \frac{1}{n} \sum_{\leq} x_i^2 - 1 \geq 3149 \right] \\ &\leq \mathbb{P} \left[ \frac{1}{n} \sum_{\leq} x_i^2 - \mathbb{E}[x_i^2] \geq 3149 \right] \\ &\leq 2 \exp \left( -\frac{3149^2 n}{9344 \frac{n}{\log \frac{1}{\delta'}} + 9344 \cdot 3149 \frac{n}{\log \frac{1}{\delta'}} / 3} \right) \\ &\leq 2 \exp \left( -1.01 \log \frac{1}{\delta'} \right) \\ &\leq 2 \exp \left( -\log \frac{8}{\delta'} \right) \quad \text{since } 1.01 \log \frac{1}{\delta'} \geq \log \frac{8}{\delta'} \text{ for suff. small } \delta' \\ &= \frac{2}{8} \delta' \end{aligned}$$

In other words, conditioning on Lemma [A.10](#page-14-1) holding, with probability at least 1 − 2 8 δ ′ , P <sup>≤</sup> x 2 <sup>i</sup> ≤ 3150n.

To bound the second term, note that

$$\sum_{\leq} (\min(\tilde{\alpha}x_i^2, 1) - \min(\alpha x_i^2, 1))^2 \leq \sum_i (\min(\tilde{\alpha}x_i^2, 1) - \min(\alpha x_i^2, 1))^2 \leq \sum_i (\min(\tilde{\alpha}x_i^2, 1) - \min(\alpha x_i^2, 1))^2$$

since α˜ := ˜α<sup>δ</sup> ,3<sup>η</sup> ≥ α<sup>δ</sup> ,η ≥ α := α<sup>δ</sup> ,<sup>0</sup> by Corollary [A.3,](#page-11-2) and "α" is monotonic in the η argument, and thus 0 ≤ min(˜αx<sup>2</sup> i , 1) − min(αx<sup>2</sup> i , 1) ≤ 1 for all i.

To further upper bound the last quantity, we have

$$\sum_i (\min(\tilde{\alpha}x_i^2, 1) - \min(\alpha x_i^2, 1)) = \sum_i \min(\tilde{\alpha}x_i^2, 1) - \sum_i \min(\alpha x_i^2, 1)$$

$$\begin{aligned} &\leq \sum_i \min(\alpha_{\delta',5\eta}x_i^2, 1) - \sum_i \min(\alpha_{\delta',0}x_i^2, 1) && \text{by Corollary A.3 and by definition of } \alpha \\ &= \frac{1}{3} \log \frac{1}{\delta'} + 5\eta n - \frac{1}{3} \log \frac{1}{\delta'} \\ &= 5\eta n \end{aligned}$$

Finally, summarizing, we have that with probability at least 1 − 6 8 δ ′ :

$$\begin{aligned} & \sqrt{\left(\sum_{\leq} x_i^2\right) \left(\sum_{\leq} (\min(\tilde{\alpha}x_i^2, 1) - \min(\alpha x_i^2, 1))^2\right)} \\ & \leq \sqrt{3150n \cdot 5\eta n} \\ & = n\sqrt{15750\eta} \\ & \leq 125.5n\sqrt{\eta} \end{aligned}$$

as desired.

#### A.2. Bounding the Error due to Corrupting the Samples

We now present the following proposition upper bounding the error incurred on Estimator [1](#page-0-0) by the arbitrary corruption of the adversary on the clean samples, while *fixing* α˜ := ˜α<sup>δ</sup> ,3<sup>η</sup> as the influence parameter.

We again assume that the uncorrupted distribution has mean 0 and variance 1.

Proposition A.11. *Suppose both* log <sup>1</sup> δ′ n *and* δ ′ *are bounded by some small universal constant. Let* α˜ *be the influence parameter computed from the* corrupted *samples with robustness level* <sup>1</sup> 3 log <sup>1</sup> δ ′ + 3ηn*, namely,* α˜ := ˜α<sup>δ</sup> ′ ,3η*. Then with probability at least* 1 − 1 8 δ ′ *, the mean estimate using* α˜ *on the* clean *samples differs from the mean estimate using* α˜ *on the* corrupted *samples by at most* <sup>8</sup>.586√<sup>η</sup>*, i.e.,*

$$\left| \sum_i \tilde{x}_i (1 - \min(\tilde{\alpha} \tilde{x}_i^2, 1)) - \sum_i x_i (1 - \min(\tilde{\alpha} x_i^2, 1)) \right| \leq 8.586n\sqrt{\eta}$$

To provide some intuition towards our proof strategy for Proposition [A.11,](#page-16-0) consider the adversary's arbitrary corruption, which can be interpreted piece-wise as moving each clean sample that the adversary wishes to corrupt to a new location. Since the influence parameter controls the contribution of each sample to the mean estimate, based on how far from the mean it is, moving a sample too far from the mean or moving the sample too close to the mean will both incur very little error. The question then, is, what is the maximum estimation error the adversary can incur by corrupting a single sample?

Fixing the value of α˜ as in the statement of Proposition [A.11,](#page-16-0) we can upper bound the maximum magnitude of the expression n P i xi(1 − min(˜αx<sup>2</sup> i , 1)) by calculus, which is O(1/ √ α˜). We will show a lower bound of α˜, specifically that α˜ ≥ Ω(η), and then conclude that the maximum total error possible by corrupting ηn samples is at most <sup>1</sup> n · ηn · Θ(1/ √η) = Θ(√η), as desired.

We use a similar strategy as in the proof of Proposition [A.4,](#page-11-0) using quantile statistics as well as Corollary [A.3](#page-11-2) to obtain our desired lower bound on α˜, before proving Proposition [A.11](#page-16-0) in Appendix [A.](#page-10-0)

Lemma A.12. *Suppose both* log <sup>1</sup> δ′ n *and* δ ′ *are bounded by some small universal constant. Denote* c<sup>1</sup> = 1 3 *, and* κ := c1( n log <sup>1</sup> δ ′ )*. Let* c<sup>2</sup> := 55.252*. Then the* 1 − κ *empirical quantile of* x<sup>i</sup> *is at most* Q1−κ/c<sup>2</sup> (D) *with probability at least* 1 − 1 <sup>32</sup> δ ′ *.*

*Proof.* For the 1 − κ empirical quantile of x<sup>i</sup> to be greater than Q1−κ/c<sup>2</sup> (D), there has to be more than κn samples greater than Q1−κ/c<sup>2</sup> (D). Thus, it suffices to prove that |{i ∈ [n] : x<sup>i</sup> ≥ Q1−κ/c<sup>2</sup> (D)}| ≥ κn with probability at most <sup>1</sup> <sup>32</sup> δ ′ .

Denote Z<sup>i</sup> := 1<sup>x</sup>i≥Q1−κ/c<sup>2</sup> (D) . Then Z := |{i ∈ [n] : x<sup>i</sup> ≥ Q1−κ/c<sup>2</sup> (D)}| = P<sup>n</sup> <sup>i</sup>=1 Z<sup>i</sup> . Obviously E[Z] = κn/c2. Denote p := E[Z<sup>i</sup> ] = κ/c2. Then V ar[Z<sup>i</sup> ] = p(1 − p).

By multiplicative Chernoff,

$$\begin{aligned}
\mathbb{P}[Z \geq c_2 pn] &\leq \left( \frac{e^{c_2-1}}{c_2^2} \right)^{pn} \\
&= \exp((c_2 - 1 - c_2 \log c_2)pn) \\
&= \exp\left(\frac{c_1(c_2 - 1 - c_2 \log c_2)}{c_2}(\log \frac{1}{\delta'})\right) \\
&\leq \exp\left((-1.01)(\log \frac{1}{\delta'})\right) \quad \text{by choice of } c_1, c_2 \\
&\leq \exp\left(-\log \frac{32}{\delta'}\right) = \frac{1}{32} \delta' \quad \text{since } 1.01 \log \frac{1}{\delta'} \geq \log \frac{32}{\delta'} \text{ for suff. small } \delta'
\end{aligned}$$

as desired.

By symmetry, we have the following corollary as well:

Corollary A.13. *The* κ *empirical quantile of* x<sup>i</sup> *is at least* Qκ/c<sup>2</sup> (D) *with probability at least* 1 − <sup>32</sup> δ ′ *.*

Lemma A.14. *Let* Dtrimmed *denote the trimmed version of* D *conditioned on lying in* [Qκ/c<sup>2</sup> (D), Q1−κ/c<sup>2</sup> (D)]*. Then* <sup>E</sup>[D<sup>2</sup> trimmed] ≤ c 2 2 (c2−2κ) 2 *.*

*Proof.* Denote by <sup>1</sup>untrimmed(x) the indicator measure that maps <sup>x</sup> to <sup>1</sup>Qκ/c<sup>2</sup> (D)≤x≤Q1−κ/c<sup>2</sup> (D) , the indicator of whether x is untrimmed. Then observe that Dtrimmed = c<sup>2</sup> c2−2κ (D · <sup>1</sup>untrimmed). Thus

$$\begin{aligned}\mathbb{E}[D_{trimmed}^2] &= \mathbb{E}\left[\left(\frac{c_2}{c_2 - 2\kappa}(D \cdot \mathbb{1}_{untrimmed})\right)^2\right] \\ &= \frac{c_2^2}{(c_2 - 2\kappa)^2} \mathbb{E}[D^2 \mathbb{1}_{untrimmed}] \\ &\leq \frac{c_2^2}{(c_2 - 2\kappa)^2} \mathbb{E}[D^2] \\ &= \frac{c_2^2}{(c_2 - 2\kappa)^2}\end{aligned}$$

as desired.

Lemma A.15. *Suppose both* log <sup>1</sup> δ′ n *and* δ ′ *are bounded by some small universal constant. Let* (xtrimmed)<sup>i</sup> *denote the sample* x<sup>i</sup> *after trimming, such that* (xtrimmed)<sup>i</sup> = x<sup>i</sup> *if* x<sup>i</sup> ∈ [Qκ/c<sup>2</sup> (D), Q1−κ/c<sup>2</sup> (D)]*, and* (xtrimmed)<sup>i</sup> = 0 *otherwise. Denote* c<sup>3</sup> := 114.532*. Then* P i (xtrimmed) 2 <sup>i</sup> ≤ (c3+1)c 2 2 (c2−2κ) <sup>2</sup> n *with probability at least* 1 − <sup>32</sup> δ ′ *.*

P *Proof.* First notice that if we replace any trimmed x<sup>i</sup> with a random sample according to Dtrimmed, then the sum i (xtrimmed) 2 i can only increase. Thus, to prove the claim, it suffices to bound the sum of n i.i.d. samples from Dtrimmed. With an abuse of notation, we let {(xtrimmed)i}i≤<sup>n</sup> denote a set of such samples.

Also notice that by our choice of <sup>c</sup>3, we have, crucially, <sup>3</sup><sup>c</sup> 3 c<sup>1</sup> (6+2c3)c<sup>2</sup> ≥ 1.01.

We start by bounding Qκ/c<sup>2</sup> (D) and Q1−κ/c<sup>2</sup> (D). Since we assume D has mean 0 and variance 1, by Chebyshev's inequality, P -|D| ≥ p<sup>c</sup><sup>2</sup> κ ≤ κ c<sup>2</sup> , which implies Qκ/c<sup>2</sup> ≥ −p<sup>c</sup><sup>2</sup> κ and Q1−κ/c<sup>2</sup> ≤ p<sup>c</sup><sup>2</sup> κ . As a result, (xtrimmed) 2 <sup>i</sup> ≤ c<sup>2</sup> κ for all i.

Then,

$$\begin{aligned} \text{Var}[(x_{\text{trimmed}})_i^2] &\leq \mathbb{E} [((x_{\text{trimmed}})_i^2)^2] \\ &\leq \frac{c_2}{\kappa} \mathbb{E}[(x_{\text{trimmed}})_i^2] \end{aligned}$$

$$\leq \frac{c_2^3}{\kappa(c_2 - 2\kappa)^2} \quad \text{by Lemma A.14}$$

Thus, by Bernstein's inequality,

$$\begin{aligned} & \mathbb{P} \left[ \sum_i ((x_{trimmed})_i)^2 \geq \frac{(c_3 + 1)c_2^2 n}{(c_2 - 2\kappa)^2} \right] \\ &= \mathbb{P} \left[ \frac{1}{n} \sum_i ((x_{trimmed})_i)^2 - \frac{c_2^2}{(c_2 - 2\kappa)^2} \geq \frac{c_3 c_2^2}{(c_2 - 2\kappa)^2} \right] \\ &\leq \mathbb{P} \left[ \frac{1}{n} \sum_i ((x_{trimmed})_i)^2 - \mathbb{E}[((x_{trimmed})_i)^2] \geq \frac{c_3 c_2^2}{(c_2 - 2\kappa)^2} \right] \\ &\leq 2 \exp \left( - \frac{c_3^2 n \frac{c_2^4}{(c_2 - 2\kappa)^4}}{2 \text{Var}[(x_{trimmed})_i^2] + \frac{2c_3 c_3^3}{3\kappa(c_2 - 2\kappa)^2}} \right) \\ &\leq 2 \exp \left( - \frac{c_3^2 n \frac{c_2^4}{(c_2 - 2\kappa)^4}}{\frac{2c_3^2}{\kappa(c_2 - 2\kappa)^2} + \frac{2c_3 c_3^3}{3\kappa(c_2 - 2\kappa)^2}} \right) \\ &= 2 \exp \left( - \frac{3c_3^2 c_2 n \kappa}{(6 + 2c_3)(c_2 - 2\kappa)^2} \right) \\ &\leq 2 \exp \left( - \frac{3c_3^2 n \kappa}{(6 + 2c_3)c_2} \right) \\ &= 2 \exp \left( - \frac{3c_3^2 c_1}{(6 + 2c_3)c_2} (\log \frac{1}{\delta'}) \right) \quad \text{by definition of } \kappa \\ &\leq 2 \exp(-1.01(\log \frac{1}{\delta'})) \quad \text{by choice of } c_3 \\ &\leq 2 \exp(-(\log \frac{32}{\delta'})) \quad \text{since } 1.01 \log \frac{1}{\delta'} \geq \log \frac{32}{\delta'} \text{ for suff. small } \delta' \\ &= \frac{2}{32} \delta' \end{aligned}$$

as desired.

Combining Lemma [A.12,](#page-16-1) Corollary [A.13,](#page-17-1) and Lemma [A.15,](#page-17-2) we have the following corollary:

Corollary A.16. *Suppose both* log <sup>1</sup> δ′ n *and* δ ′ *are bounded by some small universal constant. Let* S<κ *denote the set of indices* i *s.t.* x 2 i *is not in the top* κ *(empirical) quantile. Then* P i∈S<κ x 2 <sup>i</sup> ≤ (c3+1)c 2 (c2−2κ) <sup>2</sup> n *with probability at least* 1 − 8 δ ′ *.*

*Proof of Proposition [A.11.](#page-16-0)* We start by consider a single sample x˜<sup>i</sup> which is corrupted such that x<sup>i</sup> ̸= ˜x<sup>i</sup> . Fixing all other samples, we solve for the maximum (signed) error, or equivalently mean shift, that the adversary can incur by arbitrarily corrupting this sample x<sup>i</sup> only.

Recall that the estimator (fixing influence parameter α˜ := ˜α<sup>δ</sup> ,3η) is defined as µˆ = P i xi(1 − min(˜αx<sup>2</sup> i , 1)). Taking the derivative with respect to x<sup>i</sup> , restricted to the region in which min(˜αx<sup>2</sup> i , 1) = ˜αx<sup>2</sup> i , we have

$$\frac{\partial}{\partial x_i} \hat{\mu} = 1 - 3\tilde{\alpha}x_i^2 = 0 \Rightarrow x = \pm \frac{1}{\sqrt{3\tilde{\alpha}}}$$

Since <sup>α</sup>˜ ≥ <sup>0</sup>, a local maximum occurs at <sup>x</sup> <sup>=</sup> √ 3 ˜α . The corresponding maximum contribution of this single term is <sup>2</sup> 3 √ 3 ˜α . It is easy to verify that this local maximum is in fact the global maximum based on taking the minimization between αx˜ 2 i and 1. Symmetrically, the minimal contribution of a single term is − 2 3 √ 3 ˜α .

Thus, the maximum error the adversary can incur by corrupting a single term is <sup>4</sup> 3 √ 3 ˜α . The maximum error the adversary can incur by corrupting ηn terms is <sup>4</sup>ηn 3 √ 3 ˜α .

To upper bound <sup>4</sup>ηn 3 √ 3 ˜α , we need to lower bound α˜. Towards this, notice that α˜ ≥ α<sup>δ</sup> ,η by Corollary [A.3,](#page-11-2) which by definition satisfies:

$$\begin{aligned} \eta n + \frac{1}{3} \log \frac{1}{\delta'} &= \sum_i \min(\alpha_{\delta', \eta} x_i^2, 1) \\ &\leq \sum_i \min(\tilde{\alpha} x_i^2, 1) \quad \text{by Corollary A.3} \\ &\leq \sum_{\substack{i: x_i^2 \text{ not in the top } \kappa \text{ quantile} \\ i: x_i^2 \text{ in the top } \kappa \text{ quantile}}} \min(\tilde{\alpha} x_i^2, 1) + \sum_{\substack{i: x_i^2 \text{ in the top } \kappa \text{ quantile} \\ i: x_i^2 \text{ in the top } \kappa \text{ quantile}}} \min(\tilde{\alpha} x_i^2, 1) \\ &\leq \sum_{\substack{i: x_i^2 \text{ not in the top } \kappa \text{ quantile} \\ i: x_i^2 \text{ in the top } \kappa \text{ quantile}}} \tilde{\alpha} x_i^2 + \sum_{\substack{i: x_i^2 \text{ in the top } \kappa \text{ quantile} \\ i: x_i^2 \text{ not in the top } \kappa \text{ quantile}}} 1 \\ &= \sum_{\substack{i: x_i^2 \text{ not in the top } \kappa \text{ quantile} \\ i: x_i^2 \text{ in the top } \kappa \text{ quantile}}} \tilde{\alpha} x_i^2 + \kappa n \\ &= \sum_{\substack{i: x_i^2 \text{ not in the top } \kappa \text{ quantile} \\ i: x_i^2 \text{ in the top } \kappa \text{ quantile}}} \tilde{\alpha} x_i^2 + \frac{1}{3} \log \frac{1}{\delta'} \end{aligned}$$

Rearranging, this is equivalent to

$$\eta n \leq \sum_{i: x_i^2 \text{ not in the top } \kappa \text{ quantile}} \tilde{\alpha} x_i^2$$

By Corollary [A.16,](#page-18-0) with probability at least 1 − 1 8 δ ′ ,

$$\begin{aligned} & \sum_i x_i^2 \\ & i: x_i^2 \text{ not in the top } \kappa \text{ quantile} \\ & \leq \frac{(c_3 + 1)c_2^2}{(c_2 - 2\kappa)^2} n \\ & \leq \frac{(c_3 + 1)c_2^2}{(c_2 - 2)^2} n \end{aligned}$$

Thus,

$$\eta n \leq \sum_{i: x_i^2 \text{ not in the top } \kappa \text{ quantile}} \tilde{\alpha} x_i^2 \leq \frac{(c_3 + 1)c_2^2}{(c_2 - 2)^2} n \tilde{\alpha}$$

which is equivalent to

$$\tilde{\alpha} \geq \frac{(c_2 - 2)^2}{(c_3 + 1)c_2^2} \eta \geq 0.00804\eta$$

Thus, we have

$$\left| \sum_i \tilde{x}_i (1 - \min(\tilde{\alpha} \tilde{x}_i^2, 1)) - \sum_i x_i (1 - \min(\tilde{\alpha} x_i^2, 1)) \right| \leq \frac{4\eta n}{3\sqrt{3}\tilde{\alpha}} \leq \frac{4\eta n}{3\sqrt{0.02412}\eta} \leq 8.586n\sqrt{\eta}$$

as desired.

#### A.3. Proof of Theorem [2.3](#page-2-1)

*Proof of Theorem [2.3.](#page-2-1)* We will start by assuming that the underlying distribution has mean 0 and variance 1 (by the shiftand-scale equivariance of Estimator [1\)](#page-0-0), and furthermore, that the initial estimate κ computed in Step 1 is exactly the true mean of 0. At the end of this proof, we will show that the κ = 0 assumption introduces only a negligible amount of mean estimation error.

Fixing the initial estimate κ = 0, then, Estimator [1](#page-0-0) on input the clean samples X and confidence parameter δ ′ computes the influence parameter α := α<sup>δ</sup> ,0, and returns an estimate µˆclean,α := <sup>1</sup> n P i xi(1 − min(αx<sup>2</sup> i , 1)). As a part of their analysis, [Lee & Valiant](#page-9-0) [\(2022\)](#page-9-0) showed that

$$|\hat{\mu}_{clean,\alpha}| \leq (\sqrt{2} + o(1))\sqrt{\frac{\log \frac{1}{\delta'}}{n}}$$

with probability at least 1 − δ ′ . We note that the above guarantee can in fact be slightly strengthened, so that the failure probability becomes δ ′/16, at the expense of a slightly increased o(1) by no more than a constant multiplicative factor.

Now we consider changing the "α" value from α to α˜ computed from the corrupted samples, but applying these α influence parameters on the same clean sample set, and bound the difference in the resulting mean estimates. Specifically, consider α˜ := ˜α<sup>δ</sup> ′ ,3η, which is the influence parameter computed with robustness level <sup>1</sup> 3 log <sup>1</sup> δ ′ + 3ηn from the *corrupted* samples. Let the mean estimate of using α˜ on the clean samples be µˆclean,α˜ := <sup>1</sup> n P i xi(1−min(˜αx<sup>2</sup> i , 1)). Then, by Proposition [A.4,](#page-11-0) with probability at least 1 − 6 8 δ ′ we have

$$|\hat{\mu}_{clean, \bar{\alpha}} - \hat{\mu}_{clean, \alpha}| = \frac{1}{n} \left| \sum_i x_i \min(\tilde{\alpha} x_i^2, 1) - \sum_i x_i \min(\alpha x_i^2, 1) \right| \leq 125.5\sqrt{n}$$

Next we consider the effect of replacing the effect of corruption, but after fixing the influence parameter to be α˜. Specifically, define µˆcorrupt,α˜ to be the mean estimate of the corrupted samples using α˜, namely <sup>1</sup> n P i x˜i(1 − min(˜αx˜ 2 i , 1)). Then, by Proposition [A.11,](#page-16-0) with probability at least 1 − 8 δ ′ , we have

$$|\hat{\mu}_{corrupt, \tilde{\alpha}} - \hat{\mu}_{clean, \tilde{\alpha}}| = \frac{1}{n} \left| \sum_i \tilde{x}_i (1 - \min(\tilde{\alpha} \tilde{x}_i^2, 1)) - \sum_i x_i (1 - \min(\tilde{\alpha} x_i^2, 1)) \right| \leq 8.586 \sqrt{n}$$

By union bound and triangle inequality, summing over all three error terms, we have that with probability at least 1 − 15 <sup>16</sup> δ ′ , µˆcorrupt,α˜ satisfies

$$|\hat{\mu}_{corrupt, \tilde{\alpha}}| \leq (\sqrt{2} + o(1)) \sqrt{\frac{\log \frac{1}{\delta'}}{n}} + 134.086 \sqrt{\eta} \leq (\sqrt{2} + o(1)) \sqrt{\frac{\log \frac{1}{\delta'}}{n}} + 135 \sqrt{\eta}$$

Finally, observe that on a mean-0-variance-1 distribution, and with η-corruption, the only difference between µˆcorrupt,α˜ and the output µˆ of Estimator [1](#page-0-0) (on input as stated in the theorem statement) lies in µˆcorrupt,α˜ assuming that the initial mean estimate κ was the true mean of 0. On the other hand, µˆ (Estimator [1\)](#page-0-0) uses an estimator (for example, median-of-means) to compute κ.

We use the following fact shown in [Lee & Valiant](#page-9-0) [\(2022\)](#page-9-0) about the structural properties of Estimator [1,](#page-0-0) to analyze the effect of the κ assumption:

Fact A.17 [\(Lee & Valiant](#page-9-0) [2022\)](#page-9-0). *Let* X *be a fixed set of samples of size* n*, and let* δ > 0 *be a confidence parameter. Let* µˆ = ˆµ(X, δ) *denote the output of Estimator 1 of LV22. Then Estimator 1 is affine invariant, i.e.:*

$$\hat{\mu}(aX + b, \delta) = a\hat{\mu}(X, \delta) + b$$

*Additionally, let* µˆκ(X, δ) *denote the output of Estimator 1 but where Step 1 is omitted, and the initial estimate* κ *is instead considered as an input. Then:*

$$\left| \frac{\partial \hat{\mu}_\kappa(X, \delta)}{\partial \kappa} \right| = O\left( \sqrt{\frac{\log \frac{1}{\delta}}{n}} \right)$$

We also use the following fact/assumption on the robustness of the initial estimate κ against adversarial corruption: it is a folklore result that median-of-means satisfies this. A proof of Fact [A.18](#page-21-0) is given in Appendix [F](#page-39-0) for completeness, since we are unaware of literature explicitly writing down this proof. As discussed in Estimator [1,](#page-0-0) we are free to use other estimators to compute κ as long as Fact [A.18](#page-21-0) holds analogously.

Fact A.18 (Folklore). *For any distribution* D *with mean* µ *and standard deviation* σ*, let* X˜ *be a set of* n η*-corrupted samples from* D*. The median-of-means estimate* κ *from grouping samples into* O(log <sup>1</sup> δ ′ + ηn) *buckets, on input* X˜*, satisfies*

$$\mathbb{P} \left( |\kappa - \mu| \geq O \left( \sigma \sqrt{\frac{\log \frac{1}{\delta}}{n} + \eta} \right) \right) \leq \frac{1}{16} \delta'$$

By Fact [A.18,](#page-21-0) with probability at least 1 − <sup>16</sup> δ ′ , |κ| ≤ O qlog <sup>1</sup> δ′ <sup>n</sup> + η , and by the Lipschitz bound of Fact [A.17,](#page-20-0) we can bound the (absolute) difference between µˆ and µˆcorrupt,α˜ by

$$|\hat{\mu} - \hat{\mu}_{corrupt, \tilde{\alpha}}| \leq O\left(|\kappa| \sqrt{\frac{\log \frac{1}{\delta'}}{n}}\right) \leq O\left(\sqrt{\left(\frac{\log \frac{1}{\delta'}}{n}\right)^2} + \eta \frac{\log \frac{1}{\delta'}}{n}\right) \leq o\left(\sqrt{\frac{\log \frac{1}{\delta'}}{n}}\right) + o(\sqrt{\eta})$$

Thus, on a mean-0-variance 1 distribution, with probability at least 1 − δ ′ over the clean samples (and with an arbitrary η-corruption), Estimator [1](#page-0-0) outputs

$$|\hat{\mu}| \leq (\sqrt{2} + o(1)) \sqrt{\frac{\log \frac{1}{\delta'}}{n}} + (135 + o(1)) \sqrt{\eta}$$

Combined with the affine invariance of Estimator [1](#page-0-0) as stated in Fact [A.17,](#page-20-0) if the underlying distribution instead had mean µ and variance σ 2 , we have the mean estimation guarantee

$$|\hat{\mu} - \mu| \leq \sigma \cdot \left( (\sqrt{2} + o(1)) \sqrt{\frac{\log \frac{1}{\delta'}}{n}} + (135 + o(1)) \sqrt{\eta} \right)$$

as desired.

We note that the <sup>o</sup>(1)√<sup>η</sup> term in the estimation error is solely incurred by the estimation error from the initial estimate <sup>κ</sup>.

### A.4. Proof of Theorem [2.2](#page-2-0)

In this section, we present the proof of Theorem [2.2.](#page-2-0) While the high level idea of our proof is similar to that of Theorem [2.3,](#page-2-1) there are some important distinctions, most importantly in what we require the failure probability of our component lemmas to be.

Recall that in Theorem [2.3,](#page-2-1) while we gave the parameter δ to Estimator [1,](#page-0-0) we relaxed the failure probability of the estimator to δ ′ ≥ δ. On the other hand, Theorem [2.2](#page-2-0) actually analyzes Estimator [1](#page-0-0) at failure probability δ, which in turn requires the prerequisite lemmas to have failure probability ≤ δ.

Towards that end, notice that given the desired failure probability δ, if we let δ ′ be such that <sup>1</sup> 3 log <sup>1</sup> <sup>δ</sup> = 1 3 log <sup>1</sup> δ ′ + *8*ηn, then δ = δ ′ e <sup>−</sup>24ηn. Thus, the goal of this section is to devise analogs of Proposition [A.4](#page-11-0) and [A.11](#page-16-0) with 1 − δ = 1 − δ ′ e <sup>−</sup>24ηn as failure probability instead. We stress that this reparameterization is purely for analytical purposes; Neither the estimator nor the user know anything about the corruption parameter η or the analytical assumption we use to define δ ′ . Our choice of the constant 8 in front of the ηn term is different from the previous section, and stems from a somewhat arbitrary numerical choice to prevent constants in the theorem from blowing up, while also implicitly posing constraints on η.

We present the counterpart of Proposition [A.4](#page-11-0) first, bounding the error coming from the change in α value on uncorrupted samples, with a smaller failure probability.

Proposition A.19. *Suppose both* log <sup>1</sup> δ n *and* δ *are bounded by some small universal constant. Let* α *be the influence parameter computed from the* clean *samples with robustness level* <sup>1</sup> 3 log <sup>1</sup> δ ′ + 8ηn*, namely,* α := α<sup>δ</sup> ,8η*. Let* α˜ *be the influence parameter computed from the* corrupted *samples with robustness level* <sup>1</sup> 3 log <sup>1</sup> δ ′ + 8ηn*, namely,* α˜ := ˜α<sup>δ</sup> ,8η*. Then with probability at least* 1 − <sup>11</sup> δ ′ e <sup>−</sup>24ηn*, the mean estimate using* α *on the clean samples differs from the mean estimate using* <sup>α</sup>˜ *on the* clean *samples by at most* <sup>195</sup>.065√<sup>η</sup>*, i.e.,*

$$\left| \sum_i x_i \min(\tilde{\alpha} x_i^2, 1) - \sum_i x_i \min(\alpha x_i^2, 1) \right| \leq 195.065 n \sqrt{\eta}$$

We begin proving Proposition [A.19](#page-21-2) by stating an alternative version of Lemma [A.10:](#page-14-1)

Lemma A.20. *Suppose both* log <sup>1</sup> n *and* δ *are bounded by some small universal constant. Then* α := α<sup>δ</sup> ,8<sup>η</sup> ≥ 0.000214 <sup>1</sup> n (log <sup>1</sup> δ ′ + 24ηn) *with probability at least* 1 − 4 <sup>11</sup> δ ′ · e <sup>−</sup>24ηn*.*

The proof is identical to that of Lemma [A.10,](#page-14-1) with the appropriate log <sup>1</sup> δ and δ terms replaced with log <sup>1</sup> δ ′ + 24ηn and δ ′ · e <sup>−</sup>24ηn instead.

Now that we have lower bounded α (with high probability), analogously to the proof of Proposition [A.4,](#page-11-0) we now wish to also lower bound α˜. The proof of this lower bound will deviate from that in Proposition [A.4—](#page-11-0)in Proposition [A.4](#page-11-0) we were analyzing and comparing α˜ := ˜α<sup>δ</sup> ′ ,3<sup>η</sup> and α := α<sup>δ</sup> ′ ,0; here in Proposition [A.19,](#page-21-2) we are comparing α˜ := ˜α<sup>δ</sup> ′ ,8<sup>η</sup> against α := α<sup>δ</sup> ′ ,8η.

For Proposition [A.4,](#page-11-0) the "η-subscripts" between the two "α" values differ by 3η, so we could apply Corollary [A.3](#page-11-2) and the monotonicity of α<sup>δ</sup> ′ ,η in the "η-subscript" to yield α˜ := ˜α<sup>δ</sup> ′ ,3<sup>η</sup> ≥ α<sup>δ</sup> ′ ,η ≥ α := α<sup>δ</sup> ′ ,0. Here, we need a new argument to lower bound α˜ := ˜α<sup>δ</sup> ,8η, shown in the following lemma.

Lemma A.21. *Suppose both* log <sup>1</sup> n *and* δ *are bounded by some small universal constant. Then* α˜ := ˜α<sup>δ</sup> ,8<sup>η</sup> ≥ 0.0000354 <sup>1</sup> n (log <sup>1</sup> δ ′ + 24ηn) *with probability at least* 1 − 4 <sup>11</sup> δ ′ · e <sup>−</sup>24ηn*.*

*Proof.* Notice that α˜ := ˜α<sup>δ</sup> ,8<sup>η</sup> ≥ α<sup>δ</sup> ,6<sup>η</sup> by Corollary [A.3.](#page-11-2) Letting κ := c<sup>1</sup> n (log <sup>1</sup> δ ′ + 24ηn) for some constant c<sup>1</sup> < 1 4 , we have

$$\begin{aligned} \frac{1}{4}(24\eta n + \log \frac{1}{\delta'}) &= 6\eta n + \frac{1}{4}\log \frac{1}{\delta'} \\ &\leq 6\eta n + \frac{1}{3}\log \frac{1}{\delta'} \\ &= \sum_i \min(\alpha_{\delta', 6\eta} x_i^2, 1) \\ &\leq \sum_i \min(\tilde{\alpha} x_i^2, 1) \quad \text{by Corollary A.3 as above} \\ &\leq \sum_{i: x_i^2 \text{ not in the top } \kappa \text{ quantile}} \min(\tilde{\alpha} x_i^2, 1) + \sum_{i: x_i^2 \text{ in the top } \kappa \text{ quantile}} \min(\tilde{\alpha} x_i^2, 1) \\ &\leq \sum_{i: x_i^2 \text{ not in the top } \kappa \text{ quantile}} \tilde{\alpha} x_i^2 + \sum_{i: x_i^2 \text{ in the top } \kappa \text{ quantile}} 1 \\ &= \sum_{i: x_i^2 \text{ not in the top } \kappa \text{ quantile}} \tilde{\alpha} x_i^2 + \kappa n \\ &= \sum_{i: x_i^2 \text{ not in the top } \kappa \text{ quantile}} \tilde{\alpha} x_i^2 + c_1(24\eta n + \log \frac{1}{\delta'}) \end{aligned}$$

This implies that

$$\sum_{i: x_i^2 \text{ not in the top } \kappa \text{ quantile}} \tilde{\alpha} x_i^2 \geq \left(\frac{1}{4} - c_1\right) \left(\log \frac{1}{\delta'} + 24\eta n\right)$$

With a slightly modified version of Corollary [A.16,](#page-18-0) bounding P i:x i not in the top <sup>κ</sup> quantile x i by (c3+1)<sup>c</sup> 2 (c2−2)<sup>2</sup> n with probability at least 1− 4 <sup>11</sup> δ ′ · e <sup>−</sup>24ηn, we can choose c<sup>1</sup> = 0.202, c<sup>2</sup> = 398.432, c<sup>3</sup> = 1328.46, and obtain α˜ ≥ 0.0000354(log <sup>1</sup> δ ′ + 24ηn) as desired.

*Proof of Proposition [A.19.](#page-21-2)* First, notice that for all i such that |x<sup>i</sup> | ≥ max(q α , q 1 α˜ ) = q <sup>1</sup> min(α,α˜) , the corresponding term in the left hand side becomes 0. Thus, using the notation P <sup>≤</sup> to denote summing over elements |x<sup>i</sup> | ≤ q <sup>1</sup> min(α,α˜) , the left hand side in the guarantee of Proposition [A.19](#page-21-2) is equal to

$$\left| \sum_{\leq} x_i \min(\tilde{\alpha} x_i^2, 1) - \sum_{\leq} x_i \min(\alpha x_i^2, 1) \right|$$

Then, rearranging the sums, we have

$$\begin{aligned} & \left| \sum_{\leq} x_i \min(\tilde{\alpha}x_i^2, 1) - \sum_{\leq} x_i \min(\alpha x_i^2, 1) \right| \\ &= \sum_{\leq} |x_i (\min(\tilde{\alpha}x_i^2, 1) - \min(\alpha x_i^2, 1))| \\ &\leq \sqrt{\left( \sum_{\leq} x_i^2 \right) \left( \sum_{\leq} (\min(\tilde{\alpha}x_i^2, 1) - \min(\alpha x_i^2, 1))^2 \right)} \quad \text{by Cauchy-Schwarz} \end{aligned}$$

for which we can bound the two terms separately.

To bound the first term, since we sum over only those terms where |x<sup>i</sup> | ≤ q <sup>1</sup> min(αδ′,8η,α˜) for all i, and by Lemma [A.20](#page-22-0) and [A.21,](#page-22-1) min(α<sup>δ</sup> ,8η, α˜) ≥ 0.0000354 n (log <sup>1</sup> δ ′ + 24ηn) with probability 1− 8 <sup>11</sup> δ ′ · e <sup>−</sup>24ηn, we have that x 2 <sup>i</sup> ≤ 28249n log <sup>1</sup> <sup>δ</sup>′ +24ηn for all i. Since X has mean 0 and variance 1, we know that <sup>E</sup>[x i ] ≤ <sup>E</sup>[X<sup>2</sup> ] = 1, and Var[x i ] ≤ <sup>E</sup>[x 4 i ] ≤ 28249n log <sup>1</sup> <sup>δ</sup>′ +24ηn <sup>E</sup>[x 2 i ] ≤ 28249n log <sup>1</sup> <sup>δ</sup>′ +24ηn . Thus, by Bernstein's inequality,

$$\begin{aligned} \mathbb{P} \left[ \sum_{\leq} x_i^2 \geq 19025n \right] &\leq \mathbb{P} \left[ \frac{1}{n} \sum_{\leq} x_i^2 - 1 \geq 19024 \right] \\ &\leq \mathbb{P} \left[ \frac{1}{n} \sum_{\leq} x_i^2 - \mathbb{E}[x_i^2] \geq 19024 \right] \\ &\leq 2 \exp \left( -\frac{19024^2 n}{56498 \frac{n}{\log \frac{1}{\delta'} + 24\eta n} + 56498 \cdot 19024 \frac{n}{\log \frac{1}{\delta'} + 24\eta n} / 3} \right) \\ &\leq 2 \exp \left( -1.1 \left( \log \frac{1}{\delta'} + 24\eta n \right) \right) \\ &\leq 2 \exp \left( -\left( \log \frac{11}{\delta'} + 24\eta n \right) \right) \text{ for suff. small } \delta \\ &= \frac{2}{11} \delta' \cdot e^{-24\eta n} \end{aligned}$$

To bound the second term, note that depending on the order between α := α<sup>δ</sup> ,8<sup>η</sup> and α˜ := ˜α<sup>δ</sup> ,8η, either min(αx<sup>2</sup> i , 1) − min(˜αx<sup>2</sup> i , 1) ≥ 0 for all i, or min(˜αx<sup>2</sup> i , 1) − min(αx<sup>2</sup> i , 1) ≥ 0 for all i holds. Without loss of generality, we assume that α˜ ≥ α, and that min(˜αx<sup>2</sup> i , 1) − min(αx<sup>2</sup> i , 1) ≥ 0 for all i.

Then, we have

$$\sum_{\leq} (\min(\tilde{\alpha}x_i^2, 1) - \min(\alpha x_i^2, 1))^2 \leq \sum_i (\min(\tilde{\alpha}x_i^2, 1) - \min(\alpha x_i^2, 1))^2 \leq \sum_i (\min(\tilde{\alpha}x_i^2, 1) - \min(\alpha x_i^2, 1))^2$$

since α˜ ≥ α, and that min(˜αx<sup>2</sup> i , 1) − min(αx<sup>2</sup> i , 1) ≥ 0 for all i.

To further upper bound the last quantity, we have

$$\begin{aligned} & \sum_i (\min(\tilde{\alpha}x_i^2, 1) - \min(\alpha x_i^2, 1)) \\ &= \sum_i \min(\tilde{\alpha}x_i^2, 1) - \sum_i \min(\alpha x_i^2, 1) \\ &\leq \sum_i \min(\alpha_{\delta', 10\eta} x_i^2, 1) - \sum_i \min(\alpha_{\delta', 8\eta} x_i^2, 1) \quad \text{by Corollary A.3 and by the definition of } \alpha \\ &= \frac{1}{3} \log \frac{1}{\delta'} + 10\eta n - \frac{1}{3} \log \frac{1}{\delta'} - 8\eta n \\ &= 2\eta n \end{aligned}$$

Finally, summarizing, we have that with high probability at least 1 − <sup>11</sup> δ ′ · e <sup>−</sup>24ηn:

$$\begin{aligned} & \sqrt{\left(\sum_i x_i^2\right) \left(\sum_i (\min(\tilde{\alpha} x_i^2, 1) - \min(\alpha x_i^2, 1))^2\right)} \\ & \leq \sqrt{19025n \cdot 2\eta n} \\ & = n\sqrt{38050\eta} \leq 195.065n\sqrt{\eta} \end{aligned}$$

as desired.

We now present the counterpart of Proposition [A.11,](#page-16-0) bounding the error due to the adversarial corruption with tighter failure probability.

Proposition A.22. *Suppose both* log <sup>1</sup> δ n *and* δ *are bounded by some small universal constant. Let* α˜ *be the influence parameter computed from the* corrupted *samples with robustness level* <sup>1</sup> 3 log <sup>1</sup> δ ′ + 8ηn*, namely,* α˜ := ˜α<sup>δ</sup> ,8η*. Then with probability at least* 1 − 4 <sup>11</sup> δ ′ e <sup>−</sup>24ηn*, the mean estimate using* α˜ *on the* clean *samples differ from the mean estimate using* α˜ *on the* corrupted *samples by at most* <sup>26</sup>.411√<sup>η</sup>*, i.e.,*

$$\left| \sum_i \tilde{x}_i (1 - \min(\tilde{\alpha} \tilde{x}_i^2, 1)) - \sum_i x_i (1 - \min(\tilde{\alpha} x_i^2, 1)) \right| \leq 26.411 n \sqrt{\eta}$$

*Proof.* With arguments identical to that in the proof of Proposition [A.11,](#page-16-0) the maximum error the adversary can incur by corrupting ηn terms is <sup>4</sup>ηn 3 √ 3 ˜α . To arrive at a similar lower bound of α˜ in terms of η, notice that by Lemma [A.21,](#page-22-1) we have that with probability at least 1 − 4 <sup>11</sup> δ ′ · e <sup>−</sup>24ηn,

$$\tilde{\alpha} \geq 0.0000354 \frac{1}{n} \left( \log \frac{1}{\delta'} + 24\eta n \right) \geq 0.0000354 \frac{1}{n} (24\eta n) \geq 0.0008496\eta$$

Thus, we have that

$$\left| \sum_i \tilde{x}_i (1 - \min(\tilde{\alpha} \tilde{x}_i^2, 1)) - \sum_i x_i (1 - \min(\tilde{\alpha} x_i^2, 1)) \right| \leq \frac{4\eta n}{3\sqrt{3}\tilde{\alpha}} \leq \frac{4\eta n}{3\sqrt{3 \cdot 0.0008496\eta}} \leq 26.411n\sqrt{\eta}$$

Equipped with Propositions [A.19](#page-21-2) and [A.22,](#page-24-0) we are ready to formally prove Theorem [2.2.](#page-2-0)

*Proof of Theorem [2.2.](#page-2-0)* We will start by assuming that the underlying distribution has mean 0 and variance 1 (by the shiftand-scale equivariance of Estimator [1\)](#page-0-0), and furthermore, that the initial estimate κ computed in Step 1 is exactly the true mean of 0. At the end of this proof, we will show that the κ = 0 assumption introduces only a negligible amount of mean estimation error.

Fixing the initial estimate κ = 0, then, Estimator [1](#page-0-0) on input the clean samples X and confidence parameter δ computes the influence parameter α := αδ,0, and returns an estimate µˆclean,α := <sup>1</sup> n P i xi(1 − min(αx<sup>2</sup> i , 1)). Note that αδ,<sup>0</sup> = α<sup>δ</sup> ,8η, since <sup>1</sup> 3 log <sup>1</sup> <sup>δ</sup> = 3 log <sup>1</sup> δ ′ + 8ηn by definition of δ ′ .

As a part of their analysis, [Lee & Valiant](#page-9-0) [\(2022\)](#page-9-0) showed that

$$|\hat{\mu}_{clean,\alpha}| \leq (\sqrt{2} + o(1))\sqrt{\frac{\log \frac{1}{\delta}}{n}}$$

with probability at least 1 − δ. We note that the above guarantee can in fact be slightly strengthened, so that the failure probability becomes δ/22, at the expense of a slightly increased o(1) by no more than a constant multiplicative factor.

Now we consider changing the "α" value from α to α˜ computed from the corrupted samples, but applying these α influence parameters on the same clean sample set, and bound the difference in the resulting mean estimates. Specifically, consider α˜ := ˜α<sup>δ</sup> ,8η, which is the influence parameter computed with robustness level <sup>1</sup> 3 log <sup>1</sup> δ ′ + 8ηn from the corrupted samples. Let the mean estimate of using α˜ on the clean samples be µˆclean,α˜ := <sup>1</sup> n P i xi(1−min(˜αx<sup>2</sup> i , 1)). Then, by Proposition [A.19,](#page-21-2) with probability at least 1 − 10 <sup>11</sup> δ ′ · e <sup>−</sup>24ηn = 1 − 10 <sup>11</sup> δ we have

$$|\hat{\mu}_{clean, \bar{\alpha}} - \hat{\mu}_{clean, \alpha}| = \frac{1}{n} \left| \sum_i x_i \min(\tilde{\alpha} x_i^2, 1) - \sum_i x_i \min(\alpha x_i^2, 1) \right| \leq 195.065 \sqrt{n}$$

Next we consider the effect of replacing the effect of corruption, but after fixing the influence parameter to be α˜. Specifically, define µˆcorrupt,α˜ to be the mean estimate of the corrupted samples using α˜, namely <sup>1</sup> n P i x˜i(1 − min(˜αx˜ 2 i , 1)). Then, by Proposition [A.22,](#page-24-0) with probability 1 *conditioned on* Proposition [A.19](#page-21-2) and thus Lemma [A.21](#page-22-1) holding, we have

$$|\hat{\mu}_{corrupt, \tilde{\alpha}} - \hat{\mu}_{clean, \tilde{\alpha}}| = \frac{1}{n} \left| \left| \sum_i \tilde{x}_i (1 - \min(\tilde{\alpha} \tilde{x}_i^2, 1)) - \sum_i x_i (1 - \min(\tilde{\alpha} x_i^2, 1)) \right| \right| \leq 26.411 \sqrt{\eta}$$

By union bound and triangle inequality, summing over all three error terms, we have that with probability at least 1 − 21 <sup>22</sup> δ, µˆcorrupt,α˜ satisfies

$$|\hat{\mu}_{corrupt, \tilde{\alpha}}| \leq (\sqrt{2} + o(1)) \sqrt{\frac{\log \frac{1}{\delta}}{n}} + 221.476 \sqrt{\eta} \leq (\sqrt{2} + o(1)) \sqrt{\frac{\log \frac{1}{\delta}}{n}} + 222 \sqrt{\eta}.$$

Finally, observe that on a mean-0-variance-1 distribution, and with η-corruption, the only difference between µˆcorrupt,α˜ and the output µˆ of Estimator [1](#page-0-0) (on input as stated in the theorem statement) lies in µˆcorrupt,α˜ assuming that the initial mean estimate κ was the true mean of 0. On the other hand, µˆ (Estimator [1\)](#page-0-0) computes κ from data.

We use Fact [A.17](#page-20-0) as well as the following fact/assumption about the robustness of the initial estimate κ against adversarial corruption to analyze the effect of the κ assumption. Fact [A.23](#page-25-0) is known to hold true for median-of-means as a folklore result, which we show in Appendix [F](#page-39-0) for completeness. As mentioned in Estimator [1,](#page-0-0) we can choose to use other estimators to compute the initial estimate κ as long as the estimator satisfies Fact [A.23.](#page-25-0)

Fact A.23 (Folklore). *For any distribution* D *with mean* µ *and standard deviation* σ*, let* X˜ *be a set of* n η*-corrupted samples from* D*. Assuming that* η ≤ 24n log <sup>1</sup> δ *, the median-of-means estimate* κ *from grouping samples into* O(log <sup>1</sup> δ ) *buckets, on input* X˜*, satisfies*

$$\mathbb{P} \left( |\kappa - \mu| \geq O \left( \sigma \sqrt{\frac{\log \frac{1}{\delta}}{n}} \right) \right) \leq \frac{1}{22} \delta$$

By Fact [A.23,](#page-25-0) with probability at least 1 − <sup>22</sup> δ, |κ| ≤ O q log <sup>1</sup> δ n , and by the Lipschitz bound of Fact [A.17,](#page-20-0) we can bound the (absolute) difference between µˆ and µˆcorrupt,α˜ by

$$|\hat{\mu} - \hat{\mu}_{corrupt, \hat{\alpha}}| \leq O\left(|\kappa| \sqrt{\frac{\log \frac{1}{\delta}}{n}}\right) \leq O\left(\sqrt{\left(\frac{\log \frac{1}{\delta}}{n}\right)^2}\right) \leq o\left(\sqrt{\frac{\log \frac{1}{\delta}}{n}}\right)$$

Thus, on a mean-0-variance 1 distribution, with probability at least 1 − δ over the clean samples (and with an arbitrary η-corruption), Estimator [1](#page-0-0) outputs

$$|\hat{\mu}| \leq (\sqrt{2} + o(1))\sqrt{\frac{\log \frac{1}{\delta}}{n}} + 222\sqrt{\eta}$$

Combined with the affine invariance of Estimator [1](#page-0-0) as stated in Fact [A.17,](#page-20-0) if the underlying distribution instead had mean µ and variance σ 2 , we have the mean estimation guarantee

$$|\hat{\mu} - \mu| \leq \sigma \cdot \left( (\sqrt{2} + o(1)) \sqrt{\frac{\log \frac{1}{\delta}}{n}} + 222\sqrt{\eta} \right)$$

as desired.

## B. Robustness of Estimator [1](#page-0-0) in Weaker Models

As corollaries to our main results, stating that Estimator [1](#page-0-0) is robust against adversarially corrupted data, we also show that Estimator [1](#page-0-0) is robust against two (slightly weaker) contamination models, namely Huber contamination and TV contamination. For simplicity, we present and prove direct corollaries of Theorem [2.2](#page-2-0) only. Corollaries of Theorem [2.3](#page-2-1) follow similarly.

Definition B.1 (Huber contamination [\(Huber,](#page-9-10) [1992\)](#page-9-10)). Given a corruption parameter η and a distribution D on the uncorrupted data, we say that a set of n samples is an η-Huber-contaminated sample from D if it is drawn i.i.d. from some distribution (1 − η)D + ηE for an arbitrary distribution E.

The Huber contamination model [\(Huber,](#page-9-10) [1992\)](#page-9-10) can be regarded as being weaker than the strong contamination model, because the corruption is always drawn randomly and obliviously from a fixed distribution E chosen by the adversary; on the other hand, the adversary in the strong contamination model gets to choose the corruptions *adaptively* after inspecting all the samples. We point out however that, due to the random nature of the number of corrupted samples in the Huber model (and TV contamination model later in Definition [B.5\)](#page-27-0), these models are not strictly weaker than strong contamination, despite being "weaker in expectation". Nonetheless, by Chernoff bounds, the number of corrupted samples will concentrate to O(ηn) except with exp(−Ω(ηn)) probability.

Later on, in Appendix [C,](#page-28-0) we aim to show the neighborhood optimality of Estimator [1](#page-0-0) as a corollary of its robustness against Huber contamination. In those results, we aim for a failure probability at most δ when the algorithm is given δ as input, as opposed to a failure probability that is (slightly) larger than δ. Consequently, we write our theorem below (Theorem [B.4\)](#page-27-1) for Huber-contamination robustness with a failure probability δ/2 + exp(−Ω(ηn)), so that it is upper bounded by δ when η is sufficiently large.[<sup>2</sup>](#page-26-1)

We will show Theorem [B.4](#page-27-1) as a corollary of Corollary [B.2,](#page-26-2) which is a variant of the strong-contamination robustness result in Theorem [2.2,](#page-2-0) with failure probability δ/2 instead of δ. Then, recalling that the probability of having too many corruptions is at most exp(−Ω(ηn)), a union bound gives the target failure probability in Theorem [B.4.](#page-27-1)

Corollary B.2. *Given any distribution* D *with mean* µ *and variance* σ 2 *, parameters* n, δ, η > 0*, let* X˜ *be a set of* n η*-corrupted samples from* D*.*

<sup>2</sup>We point out that, somewhat counter-intuitively, the larger η is, and the stronger the corruption is, the less likely that Huber contamination produces many more corrupted samples than adversarial corruption.

*Suppose both* log <sup>1</sup> n *and* δ *are bounded by some small universal constant, and suppose* η ≤ 24n log <sup>1</sup> δ *. Then, with probability at least* 1 − 1 2 δ *over the sampling process, Estimator [1](#page-0-0) on input* δ *and* X˜ *will output an estimate* µˆ *with error at most*

$$|\hat{\mu} - \mu| \leq \sigma \cdot \left( (1 + o(1)) \sqrt{\frac{2 \log \frac{1}{\delta}}{n}} + 222 \sqrt{\eta} \right)$$

The proof of Corollary [B.2](#page-26-2) is almost identical to that of Theorem [2.2.](#page-2-0) The main difference is, instead of using Fact [1.1](#page-0-0) to guarantee the performance of Estimator [1](#page-0-0) on i.i.d. uncorrupted samples, we use a stronger variant which is Fact [B.3](#page-27-2) below. Specifically, Fact [B.3](#page-27-2) inputs the parameter δ to Estimator [1](#page-0-0) but asks for a failure probability of δ/2 instead of δ, at the expense of a slightly larger "o(1)" term. Given the above target parameters, Fact [B.3](#page-27-2) is *not* a black-box corollary of Fact [1.1.](#page-0-0) Nonetheless, it follows directly from the analysis in Lee and Valiant's original paper [\(2022\)](#page-9-0), so we state it without proof below.

Fact B.3 [\(Lee & Valiant](#page-9-0) [2022\)](#page-9-0). *Given any distribution* D *with mean* µ *and variance* σ *, parameters* n, δ > 0*, let* X *be a set of* n *independent samples from* D*. Then, with probability at least* 1 − δ/2 *over the sampling process, Estimator [1](#page-0-0) on input* δ *and* X *will output an estimate* µˆ *with error at most*

$$|\hat{\mu} - \mu| \leq \sigma \cdot \left( (1 + o(1)) \sqrt{\frac{2 \log \frac{1}{\delta}}{n}} \right)$$

*Here, the* <sup>o</sup>(1) *term tends to* <sup>0</sup> *as* log <sup>1</sup> δ n , δ → (0, 0) *and, crucially, is independent of* D*.*

We can now state and prove the robustness of Estimator [1](#page-0-0) against Huber contamination.

Theorem B.4. *Suppose both* log <sup>1</sup> n *and* δ *are bounded by some small universal constant. Given any distribution* D *with mean* µ *and variance* σ *, parameters* n, δ, η > 0*, and a set* X *of* n η*-Huber-contaminated samples from* D*, for some* η ≤ 24en log <sup>1</sup> δ *. Estimator 1 of LV22, when given access to* n, δ*, and* X *only, will, with probability at least* 1 − ( 1 2 δ + exp(−80ηn)) *over the sampling process, yield an estimate* µˆ *with error at most*

$$|\hat{\mu} - \mu| \leq \sigma \cdot \left( (1 + o(1)) \sqrt{\frac{2 \log \frac{1}{\delta}}{n}} + 222 \sqrt{33\eta} \right)$$

*Proof.* Note that sampling from distribution (1 − η)D + ηE is exactly identical to sampling from D with probability 1 − η and from E with η. We bound the probability that such a sampling process samples more than 33ηn samples from E.

Let z<sup>i</sup> denote the indicator variable for the event that sample x<sup>i</sup> is sampled from E. By multiplicative Chernoff:

$$\mathbb{P}\left[\sum_{i=1}^n z_i \geq 33\eta n\right] \leq \left(\frac{e^{33-1}}{33^{33}}\right)^{\eta n} \leq \exp(-80\eta n)$$

Thus with probability at least 1 − exp(−80ηn), the Huber-contaminated sample set has at most 33ηn samples corrupted.

Conditioned on this happening, Corollary [B.2](#page-26-2) applies to the set of contaminated samples X with an adversary capable of arbitrarily corrupting 33ηn samples, with probability at least 1 − 2 δ. Our theorem thus follows from a union bound.

Generalizing from Huber contamination slightly is the TV contamination model, which draws samples from a distribution D′ that is within η TV distance from the genuine underlying distribution D.

Definition B.5 (TV contamination [\(Diakonikolas & Kane,](#page-9-5) [2023\)](#page-9-5)). Given a corruption parameter η and a distribution D on the uncorrupted data, we say that a set of n samples is an η-TV-contaminated sample from D if it is drawn i.i.d. from some distribution D′ such that DTV(D, D′ ) ≤ η.

Theorem B.6. *Suppose both* log <sup>1</sup> n *and* δ *are bounded by some small universal constant. Given any distribution* D *with mean* µ *and variance* σ 2 *, parameters* n, δ, η > 0*, and a set* X *of* n η*-TV-contaminated samples from* D*, for some* η ≤ 24en log <sup>1</sup> δ *. Estimator 1 of LV22, when given access to* n, δ*, and* X *only, will, with probability at least* 1 − ( 1 2 δ + exp(−80ηn)) *over the sampling process, yield an estimate* µˆ *with error at most*

$$|\hat{\mu} - \mu| \leq \sigma \cdot \left( (1 + o(1)) \sqrt{\frac{2 \log \frac{1}{\delta}}{n}} + 222 \sqrt{33\eta} \right)$$

*Proof.* Let D′ be any distribution such that DT V (D, D′ ) ≤ η. There exists a coupling between D and D′ such that for any sample index i, the probability that the sample x<sup>i</sup> from D and the sample x ′ i from D′ differs is at most η.

Let z<sup>i</sup> denote the indicator variable for the event that x<sup>i</sup> differs from x ′ i . By multiplicative chernoff:

$$\mathbb{P}\left[\sum_{i=1}^n z_i \geq 33\eta n\right] \leq \left(\frac{e^{33-1}}{333}\right)^{\eta n} \leq \exp(-80\eta n)$$

Thus with probability at least 1 − exp(−80ηn), the TV-contaminated sample set has at most 33ηn samples corrupted.

Conditioned on this happening, Corollary [B.2](#page-26-2) applies to the set of contaminated samples X with an adversary capable of arbitrarily corrupting 33ηn samples, with probability at least 1 − 2 δ. Our theorem thus follows from a union bound.

## C. Neighborhood Optimality of Estimator [1](#page-0-0)

Neighborhood optimality is a notion of instance-by-instance optimality beyond the worst-case defined in [Dang et al.](#page-8-0) [\(2023\)](#page-8-0). In this section, we show that Estimator [1](#page-0-0) is also asymptotically neighborhood optimal as a corollary to its robustness against TV contamination. We point out that our result matches that of [Dang et al.](#page-8-0) [\(2023\)](#page-8-0), which states that the median-of-means estimator is asymptotically neighborhood optimal, with the same choice of neighborhood structure and asymptotes.

For mean estimation in R, Estimator [1](#page-0-0) achieves the optimal sub-Gaussian rate even up to the constants. Their optimality is only in the worst-case regime, i.e., there *exists* an instance of mean estimation on which no estimator can perform better. A natural question to this observation is: Can we beat the sub-Gaussian rate for some distributions?

[Dang et al.](#page-8-0) [\(2023\)](#page-8-0) presented the following new universal estimation lower bound function:

Definition 2.5 [\(Dang et al.](#page-8-0) [2023\)](#page-8-0). Given a (continuous) distribution D with mean µ and a real number t ∈ [0, 1], define the t*-trimming* operation on D as follows: select a radius r such that the probability mass in [µ − r, µ + r] equals 1 − t; then, return the distribution D conditioned on lying in [µ − r, µ + r].

Given n and δ, define the trimmed distribution D<sup>∗</sup> n,δ to be the <sup>0</sup>.<sup>45</sup> n log <sup>1</sup> δ -trimmed version of D. When δ is implicit, we may denote this as D<sup>∗</sup> n . Now define the error function ϵn,δ(D) = |µ − µ ∗ n | + σ ∗ n q log <sup>1</sup> n , where µ ∗ n and σ ∗ n are the mean and standard deviation of D<sup>∗</sup> n respectively.

The ground truth error function ϵn,δ in Definition [2.5](#page-3-0) applies simultaneously for all distributions with a finite mean, not just for the worst-case distribution, in the sense that for any distribution p we can construct a neighboring distribution q that is indistinguishable with probability at least 1 − δ with n samples, but the mean of p and q are well-separated by O(ϵn,δ(p)) in distance. Thus, no estimator can beat the error function on p and q simultaneously, since any estimator that can will be a distinguisher between p and q.

To formalize this notion of lower bound and optimality, [Dang et al.](#page-8-0) [\(2023\)](#page-8-0) presented the following pair of definitions:

Definition 5.1 (Neighborhood Pareto Bounds [\(Dang et al.,](#page-8-0) [2023\)](#page-8-0)). Let n be the number of samples and δ be the failure probability. Given a neighborhood function Nn,δ : P<sup>1</sup> → 2 P<sup>1</sup> , we say that the error function ϵn,δ(D) : P<sup>1</sup> → <sup>R</sup> + 0 is a neighborhood Pareto bound for P<sup>1</sup> with respect to Nn,δ if for all distributions D ∈ P1, *no* estimator µˆ taking n i.i.d. samples can simultaneously achieve the following two conditions:

- For all D′ ∈ Nn,δ(D), with probability 1 − δ over the n i.i.d. samples from D′ , we have |µˆ − µD′ | ≤ ϵn,δ(D′ ).
- With probability 1 − δ over the n i.i.d. samples from D, |µˆ − µD| < ϵn,δ(D).

Definition 5.2 ((κ, τ )-Neighborhood Optimal Estimators [\(Dang et al.,](#page-8-0) [2023\)](#page-8-0)). Let κ > 1 be a multiplicative loss factor in estimation error, and τ > 1 be a multiplicative loss factor in sample complexity.

Given the parameters κ, τ > 1, sample complexity n, failure probability δ and neighborhood function Nn,δ, a mean estimator µˆ is (κ, τ )-neighborhood optimal with respect to Nn,δ if there exists an error function ϵn,δ(D) such that min(ϵn/τ,δ(D), ϵn,δ(D)) is a neighborhood Pareto bound[<sup>3</sup>](#page-29-0) , and µˆ gives estimation error at most κ · ϵn,δ(D) with probability at least 1 − δ when taking n i.i.d. samples from any distribution D ∈ P1.

Definition [5.1](#page-5-4) enforces that it is impossible to "beat" the neighborhood Pareto bound locally, performing as good over the local neighborhood, while strictly better on the center p. It essentially enforces admissibility over every such local neighborhood, forming a smooth interpolation between instance optimality and admissibility, which are both classical definitions of optimality beyond the worst-case that fails in context of mean estimation in R.

[Dang et al.](#page-8-0) [\(2023\)](#page-8-0) showed that there exists a neighborhood function Nn,δ for which ϵn,δ as defined in Definition [2.5](#page-3-0) is a neighborhood Pareto bound, and for which median-of-means is (κ, 3)-neighborhood optimal for some sufficiently large constant κ. We capture the necessary components of their analysis and state them without proof in the following fact:

Fact C.1. *There exists a neighborhood function* Nn,δ : P<sup>1</sup> → 2 P<sup>1</sup> *for which* ϵn,δ *as defined in Definition [2.5](#page-3-0) is a neighborhood Pareto bound. Any estimator that obtains estimation error* O(ϵn,δ(p)) *for all distributions* p ∈ P<sup>1</sup> *is* (κ, 3)*-neighborhood optimal with respect to* Nn,δ *for some sufficiently large constant* κ*.*

Thus, to show that Estimator [1](#page-0-0) is neighborhood optimal, it suffices to show that it asymptotically matches the performance of the ground truth error bound ϵn,δ. We present a reanalysis of Estimator [1](#page-0-0) using its robustness against TV contamination that proves its neighborhood optimality in such a way.

Theorem C.2. *Suppose both* log <sup>1</sup> δ n *and* δ *are bounded by some small universal constant. Denote by* Nn,δ *the neighborhood function whose existence is implied by Fact [C.1.](#page-29-1) Estimator [1](#page-0-0) is* (κ, 3)*-neighborhood optimal with respect to* Nn,δ *for some sufficiently large constant* κ*.*

*Proof.* Let ϵn,δ be the ground truth error function as defined in Definition [2.5.](#page-3-0) By Fact [C.1,](#page-29-1) it suffices to show that Estimator 1 achieves an error rate of O(ϵn,δ(p)) for all distributions p.

Let p be any distribution with mean µp. Let c be the constant such that <sup>c</sup> n log <sup>2</sup> <sup>δ</sup> = 80n log <sup>1</sup> <sup>δ</sup> ≤ 1 24en log <sup>1</sup> δ . Let p ∗ <sup>n</sup> be the c n log <sup>2</sup> δ -trimmed version of p as defined in Definition [2.5,](#page-3-0) and let µp<sup>∗</sup> n and σp<sup>∗</sup> n be the mean and standard deviation of p ∗ n respectively. Let ϵn,δ be the error function as defined in Definition [2.5.](#page-3-0)

Observe that p ∗ n satisfies that DT V (p ∗ n , p) = <sup>c</sup> n log <sup>2</sup> δ . Thus, by Theorem [B.6](#page-27-3) with D = p ∗ n and η = c n log <sup>2</sup> δ , we have that with probability at least 1 − ( 1 2 δ + exp(−80ηn)) = 1 − δ,

$$\begin{aligned} |\hat{\mu} - \mu_{p_n^*}| &\leq \sigma_{p_n^*} \cdot \left( (1 + o(1)) \sqrt{\frac{2 \log \frac{1}{\delta}}{n}} + 222 \sqrt{\frac{33c \log \frac{2}{\delta}}{n}} \right) \\ &= O \left( \sigma_{p_n^*} \sqrt{\frac{\log \frac{1}{\delta}}{n}} \right) \end{aligned}$$

Thus by the triangle inequality, we have that with probability at least 1 − δ,

$$|\hat{\mu} - \mu_p| \leq |\mu_p - \mu_{p_n^*}| + |\hat{\mu} - \mu_{p_n^*}| \leq |\mu_p - \mu_{p_n^*}| + O\left(\sigma_{p_n^*} \sqrt{\frac{\log \frac{1}{\delta}}{n}}\right) = O(\epsilon_{n,\delta})$$

Combined with Fact [C.1,](#page-29-1) this implies that Estimator [1](#page-0-0) is indeed (κ, 3)-neighborhood optimal with respect to Nn,δ for some sufficiently large constant κ, as desired.

<sup>3</sup>While it is intuitive to expect that an error function decreases in n, it might not be true in general. Indeed, the function used by [Dang](#page-8-0) [et al.](#page-8-0) [\(2023\)](#page-8-0) (Definition [2.5\)](#page-3-0) is not necessarily monotonic. This is the reason for the "min" in the neighborhood Pareto bound requirement.

We point out that the proof of Theorem [B.6](#page-27-3) and Theorem [C.2](#page-29-2) does not depend on the specific characteristics of Estimator [1,](#page-0-0) and instead only relies on Theorem [2.2](#page-2-0) holding. Thus, we can obtain a similar neighborhood optimality result for any mean estimator that is sub-Gaussian and robust against adversarial corruption, and enjoys asymptotically matching bounds as in Theorem [2.2:](#page-2-0)

Corollary C.3. *Any estimator that, when given* δ > 0 *and a set of* n η*-corrupted sample from distribution* D *with mean* µ *and variance* σ 2 *, yields a mean estimate* µˆ *satisfying*

$$|\hat{\mu} - \mu| \leq O \left( \sigma \left( \sqrt{\frac{\log \frac{1}{\delta}}{n}} + \sqrt{\eta} \right) \right)$$

*is* (κ, 3)*-neighborhood optimal with respect to* Nn,δ *for some sufficiently large constant* κ*.*

## D. Remaining Proofs of Section [6](#page-6-2)

In Section [6,](#page-6-2) we outlined the proof strategy of our main theorem for low moment performances of Estimator [1,](#page-0-0) which follows that of [Lee & Valiant](#page-9-0) [\(2022\)](#page-9-0), and provided the statement of the reformulation of Estimator [1](#page-0-0) as a 2-parameter ψ-estimator. In this section, we present the full proof of Theorem [2.4,](#page-2-2) following the structure and organization of [Lee &](#page-9-0) [Valiant](#page-9-0) [\(2022\)](#page-9-0). We present each component lemma along with the intuition behind it, and refer the reader to [Lee & Valiant](#page-9-0) [\(2022\)](#page-9-0) for more detailed motivations and discussions. For completeness, we restate our main theorem:

Theorem 2.4. *Given any distribution* D *with mean* µ *and* z *th moment* M<sup>z</sup> *for some* z ∈ (1, 2)*, let* X *be a set of* n *i.i.d. samples from* D*. Then, with probability at least* 1 − δ *over the randomness of* X*, Estimator [1](#page-0-0) on input* δ *and* X *will output an estimate* µˆ *with error at most*

$$|\hat{\mu} - \mu| \leq (M_z)^{\frac{1}{2}} \cdot (1 + o(1)) \left( c_z \frac{\log \frac{1}{\delta}}{n} \right)^{1 - \frac{1}{2}}$$

*where* <sup>c</sup><sup>z</sup> = 2(5.6) <sup>1</sup> <sup>z</sup>−<sup>1</sup> −1 *. Here, the* <sup>o</sup>(1) *term tends to* <sup>0</sup> *as* log <sup>1</sup> δ n , δ → (0, 0)*, in a manner independent of* D *and independent of* z*.*

From the structural properties of Estimator [1](#page-0-0) stated explicitly in Fact [A.17,](#page-20-0) we make the simplifying assumption that the underlying distribution D has mean 0 and z th moment M<sup>z</sup> = 1, and that the initial estimate κ in Step 1 of Estimator [1](#page-0-0) is replaced by 0. For the rest of the appendix, we will refer to the error bound (czlog <sup>1</sup> δ /n) <sup>1</sup>−1/z as ϵ (omitting the 1 + o(1) factor).

Definition [6.1](#page-7-0) reduced proving Chernoff bounds for Estimator [1,](#page-0-0) which is a sum of *dependent* terms, to proving Chernoff bounds for the sums of independent terms of the 2-parameter ψ-estimator. Thus, it suffices to show that with high probability, the ψ-estimator in Definition [6.1](#page-7-0) returns an estimate µˆ that is close to 0—or equivalently, *every* pair (ˆµ ′ , αˆ ′ ) with µˆ ′ *far* away from 0 must satisfy ψ(X, µˆ ′ , αˆ ′ ) ̸= 0. We turn to analyze the following proposition, capturing this reduction:

Proposition D.1. *There exists a universal constant* c > 0 *such that, for all* 1 < z ≤ 2*, fixing* ϵ ′ = 1 + <sup>c</sup> log log <sup>1</sup> δ log <sup>1</sup> δ ϵ *where* ϵ = cz log <sup>1</sup> n 1− 1 z *and* <sup>c</sup><sup>z</sup> = 2(5.6) <sup>1</sup> <sup>z</sup>−<sup>1</sup> −1 *, we have that for all distributions* D *with mean* 0 *and* z *th moment* 1*, with probability at least* 1 − δ 2 *over the set of samples* X*, for all* µ, ˆ αˆ *where* |µˆ| > ϵ′ *and* α >ˆ 0*, the vector* ψ(X, µ, ˆ αˆ) ̸= 0*.*

We stress that the universal constant c in fact *does not* depend on z.

Towards proving Proposition [D.1,](#page-30-1) we extend [Lee & Valiant](#page-9-0) [\(2022\)](#page-9-0)'s strategy, analyzing the function ψ(X, µ, ˆ αˆ) on a finite bounded mesh of values of µ, ˆ αˆ covering the most delicate range for our analysis, and show via standard ϵ-net arguments that the proposition holds for *any* choice of |µˆ| > ϵ′ , αˆ. Specifically, we show that ψ(X, µ, ˆ αˆ), on the finite mesh, is far from the origin in some direction, and is Lipschitz within the mesh. We then show how to reduce the analysis outside of the mesh to that inside the mesh.

To find a mesh to cover the relevant range of αˆ in the right "scale-invariant" way, we reparameterize αˆ by wˆ := log<sup>2</sup> (1/δ) 3 ˆαn2ϵ <sup>2</sup> , where ϵ in the denominator (as defined in Proposition [D.1\)](#page-30-1) encodes the desired dependence on z. We choose an evenly-spaced mesh over wˆ ∈ [0.05, 555]. This is analogous to the mesh on "vˆ" in [Lee & Valiant](#page-9-0) [\(2022\)](#page-9-0)'s analysis.

The main technical component of our (and [Lee & Valiant](#page-9-0) [\(2022\)](#page-9-0)'s) proof strategy is the proof that for each µ, ˆ wˆ = log<sup>2</sup> (1/δ) 3 ˆαn2ϵ from the finite mesh we analyze, ψ(X, µ, ˆ αˆ) is far away from 0 *in some direction*. We linearize this claim and prove a stronger result, that there exists a specific direction d( ˆw) such that with high probability, ψ(X, µ, ˆ αˆ) is more than <sup>1</sup> log(1/δ) distance away from the origin in direction d. More formally, we prove the following lemma:

Lemma D.2. *Let* D *be an arbitrary distribution with mean* 0 *and* z *th moment* 1 *for some* 1 < z ≤ 2*. There exists a universal constant* c independent of z *where the following is true. Fixing* µˆ = 1 + <sup>c</sup> log log <sup>1</sup> δ log <sup>1</sup> δ ϵ *where* ϵ = cz log <sup>1</sup> δ n 1− 1 *and* <sup>c</sup><sup>z</sup> = 2(5.6) <sup>1</sup> <sup>z</sup>−<sup>1</sup> −1 *, then for all* δ *smaller than some universal constant, and for all* wˆ ∈ [0.05, 555]*, there exists a vector* d( ˆw) := (dµ, dα) *where* d<sup>µ</sup> ≥ 0*, and both* nϵ log <sup>1</sup> |dµ| *and* |dα| *are bounded by a universal constant, such that*

$$x \leftarrow \mathbb{P}^n \left( \mathbf{d}(\hat{w}) \cdot \psi \left( X, \hat{\mu} = \epsilon', \hat{\alpha} = \frac{\log^2(1/\delta)}{3\hat{w}n^2\epsilon^2} \right) > \frac{1}{\log \frac{1}{\delta}} \right) \geq 1 - \frac{\delta}{\log^4 \frac{1}{\delta}}.$$

*Furthermore, for* wˆ = 0.05 *we have* d<sup>µ</sup> = √ <sup>3</sup>.75log <sup>1</sup> δ nϵ *,* d<sup>α</sup> = √ 3*; and for* wˆ = 555 *we have* d<sup>µ</sup> = 0*,* d<sup>α</sup> < 0*.*

From here, Proposition [D.1](#page-30-1) follows from a union bound over pairs of (ˆµ, αˆ) on the finite mesh, the monotonicity of ψ(X, µ, ˆ αˆ) beyond the mesh granted by the boundary conditions specified in Lemma [D.2,](#page-31-0) and the Lipschitzness of ψ(X, µ, ˆ αˆ) within the region covered by the mesh, formalized in the following lemma:

Lemma D.3. *Consider an arbitrary set of* n *samples* X*. Consider the expressions* ψµ(X, µ, ˆ αˆ)*,* ψα(X, αˆ)*, reparameterized in terms of* wˆ = log<sup>2</sup> <sup>1</sup> 3 ˆαn2ϵ <sup>2</sup> *in place of* αˆ*. Suppose the equation* ψα(X, αˆ) = 0 *has a solution in the range* wˆ ∈ [0.05, 555]*. Then the functions* log <sup>1</sup> δ nϵ ψµ(X, µ, ˆ αˆ) *and* ψα(X, αˆ) *are Lipschitz with respect to* wˆ *on the entire interval* wˆ ∈ [0.05, 555]*, with Lipschitz constant* clog <sup>1</sup> δ *for some universal constant* c*.*

With these components, we now prove Proposition [D.1,](#page-30-1) and subsequently formally prove our main theorem, Theorem [2.4,](#page-2-2) before returning to prove Lemma [D.2](#page-31-0) and [D.3](#page-31-1) in Appendix [D.1.](#page-33-0)

*Proof of Proposition [D.1.](#page-30-1)* By symmetry, instead of considering positive and negative µˆ, it suffices to consider the case µ > ϵ ˆ ′ and show that this case succeeds with probability at least 1 − δ 4 .

To prove the claim, we first prove a stronger statement on a restricted domain, that with probability at least 1 − δ 4 over the randomness of the sample set X, for each wˆ ∈ [0.05, 555] there exists a vector d = (dµ, dα) such that d · ψ(X, ϵ′ , αˆ) > 0, with d<sup>µ</sup> > 0 throughout, and , for wˆ = 0.05 we have d<sup>µ</sup> = √ <sup>3</sup>.75log <sup>1</sup> δ nϵ , d<sup>α</sup> = √ 3; and for wˆ = 555 we have d<sup>µ</sup> = 0, d<sup>α</sup> < 0. We will first apply Lemma [D.2](#page-31-0) to each wˆ in a discrete mesh: let M consist of evenly spaced points between 0.05 and 555 with spacing 1/ log<sup>3</sup> <sup>1</sup> δ (thus with Θ(log<sup>3</sup> <sup>1</sup> δ ) many points).

By Lemma [D.2](#page-31-0) and a union bound over these Θ(log<sup>3</sup> <sup>1</sup> δ ) points, we have that with probability at least 1 − δ Θ(log <sup>1</sup> ) (which is at least 1 − δ 4 for δ smaller than some universal constant) over the set of n samples X, for all wˆ ∈ M, there exists a vector d( ˆw) such that d( ˆw) · ψ(X, µˆ = ϵ ′ , αˆ) > 1/log <sup>1</sup> δ , where d further satisfies the desired positivity and boundary conditions, and where both nϵ log <sup>1</sup> δ |dµ| and |dα| are bounded by a universal constant. For the rest of the proof, we will only consider sets of samples X satisfying the above condition.

Now consider an arbitrary wˆ ′ ∈ [0.05, 555] \ M and consider the vector ψ evaluated at αˆ ′ = log<sup>2</sup> <sup>1</sup> δ 3n2ϵ <sup>2</sup>wˆ ′ . We wish to extend the dot product inequality to hold also for wˆ ′ . If ψ<sup>α</sup> ̸= 0 then there is nothing to prove: set d<sup>µ</sup> = 0 and d<sup>α</sup> = sign(ψα); otherwise, <sup>ψ</sup><sup>α</sup> = 0 means we may apply Lemma [D.3](#page-31-1) to conclude that both log <sup>1</sup> δ nϵ ψµ(X, µ, ˆ αˆ ′ ) and ψα(X, µ, ˆ αˆ ′ ) are Lipschitz with respect to wˆ ′ on the interval wˆ ′ ∈ [0.05, 555] with Lipschitz constant clog <sup>1</sup> δ for some universal constant c.

Consider the closest wˆ ∈ M to wˆ ′ , which by definition of M is at most 1/ log<sup>3</sup> <sup>1</sup> δ away. By assumption on X, there exists a vector d such that d · ψ(X, µˆ = ϵ ′ , αˆ) > 1/log <sup>1</sup> δ , with d<sup>µ</sup> > 0 and both nϵ log <sup>1</sup> δ |dµ| and |dα| are bounded by a universal constant. Because of the Lipschitz bounds on ψ, combined with the bounds on the size of d<sup>µ</sup> and dα, we conclude that the Lipschitz constant of the dot product (treating the vector d as fixed) is O(log <sup>1</sup> δ ). Thus, the large positive dot product at wˆ implies at least a positive dot product nearby at wˆ ′ : d · ψ(X, µˆ = ϵ ′ , wˆ ′ ) > log <sup>1</sup> δ − O(log <sup>1</sup> δ ) 1 log<sup>3</sup> <sup>1</sup> δ > 0, for sufficiently small δ as given in the proposition statement.

Having shown the stronger version of the claim for the restriction µˆ = ϵ ′ and wˆ ∈ [0.05, 555] we now extend to the entire domain via three monotonicity arguments. Explicitly, assume the set of samples X satisfies the dot product inequality above with the vector function d( ˆw), where d( ˆw) satisfies the boundary conditions at wˆ = 0.05 and 555 specified in Lemma [D.2.](#page-31-0) From this assumption, we will show that ψ ̸= 0 for *any* positive wˆ = log<sup>2</sup> <sup>1</sup> δ 3n2ϵ <sup>2</sup>αˆ , and for *any* µˆ ≥ ϵ ′ .

First consider w >ˆ 555, still fixing µˆ = ϵ ′ . The function ψ<sup>α</sup> = P<sup>n</sup> i=1(min(ˆαx<sup>2</sup> i , 1) − 1 3n log <sup>1</sup> δ ) is an increasing function of αˆ, and thus a decreasing function of wˆ = log<sup>2</sup> <sup>1</sup> 3n2ϵ <sup>2</sup>αˆ . Since for wˆ = 555, the dot product d · ψ > 0 with d<sup>µ</sup> = 0, d<sup>α</sup> < 0, the dot product will thus remain positive for this same choice of d as we increase wˆ from 555.

Next, for w <ˆ 0.05, again fixing µˆ = ϵ ′ , we analogously show that the dot product of ψ(X, ϵ′ , αˆ) with the fixed vector d(0.05) will increase as we decrease wˆ. The i-th term in the sums defining ψ<sup>µ</sup> or ψ<sup>α</sup> depends on αˆ (and thus wˆ) only in the factor min(ˆαx<sup>i</sup> , 1). Further, there is no dependence unless the first term attains the min, namely |x<sup>i</sup> | ≤ p 1/αˆ, which in turn is upper bounded by √ 0.15 nϵ log <sup>1</sup> because of our assumption that w <ˆ 0.05. Thus, the only i-th terms in the dot product which have αˆ dependency are simply equal to dµαxˆ 3 <sup>i</sup> + dααxˆ 2 <sup>i</sup> = ˆαx<sup>2</sup> i (d<sup>α</sup> + xidµ). By our choice of <sup>d</sup>µ(0.05) = √ <sup>3</sup>.75log <sup>1</sup> δ nϵ and d<sup>α</sup> = √ 3 from Lemma [D.2,](#page-31-0) the expression (d<sup>α</sup> + xidµ) ≥ √ 3 − √ 0.15√ 3.75 is thus always non-negative, and thus the overall dot product cannot decrease as we send wˆ to 0 as desired.

We have thus shown that, for all non-negative αˆ = log<sup>2</sup> <sup>1</sup> δ 3n2ϵ <sup>2</sup>wˆ , there is a vector d with d<sup>µ</sup> ≥ 0 whose dot product with ψ(X, ϵ′ , αˆ) is greater than 0. We complete the proof by noting that the only dependence on µˆ in ψ is that ψ<sup>µ</sup> is (trivially) increasing in µˆ. Since d<sup>µ</sup> ≥ 0, increasing µˆ from ϵ ′ will only increase the dot product, and thus the dot product remains strictly greater than 0, implying that ψ(X, µ, ˆ αˆ) ̸= 0, as desired.

To close out this section, we formally show that Proposition [D.1](#page-30-1) implies our desired main theorem, Theorem [2.4](#page-2-2) for completeness, using the following fact about the performance of the median-of-means estimator for the initial estimate κ:

Fact D.4 [\(Bubeck et al.](#page-8-3) [2012\)](#page-8-3). *For any distribution* D *with mean* µ *and* z *th moment* σz*, the median-of-means estimate* κ *from grouping samples into* O(log <sup>1</sup> δ ) *buckets, on input* n *samples, satisfies*

$$\mathbb{P} \left( |\kappa - \mu| > O \left( (\sigma_z)^{\frac{1}{z}} \left( \frac{\log \frac{1}{\delta}}{n} \right)^{1 - \frac{1}{z}} \right) \right) \leq \frac{1}{2} \delta$$

*Here, the big-O notation hides a universal constant that is crucially also independent of* z*.*

Again, we are free to choose to use other estimators for κ in Step 1 of Estimator [1](#page-0-0) as long as the above fact holds true for the estimator being used.

*Proof of Theorem [2.4.](#page-2-2)* We start by making the simplifying assumption that µ = 0 and M<sup>z</sup> = 1, and reformulate Estimator 1 with Step 1 replaced with κ = 0 as a 2-parameter ψ-estimator, that takes in n independent samples X = x1, · · · , x<sup>n</sup> from D, and solves for the (unique) pair µ, ˆ αˆ satisfying ψ<sup>µ</sup> = 0 and ψ<sup>α</sup> = 0, where the functions are defined as follows:

$$\psi_\mu(X, \hat{\mu}, \hat{\alpha}) = \sum_{i=1}^n (\hat{\mu} - x_i(1 - \min(\hat{\alpha}x_i^2, 1)))$$

$$\psi_\alpha(X, \hat{\mu}, \hat{\alpha}) = \sum_{i=1}^n \left( \min(\hat{\alpha}x_i^2, 1) - \frac{1}{3n} \log \frac{1}{\delta} \right)$$

and outputs µˆ in the solution. We denote by ψ the 2-element vector (ψµ, ψα).

By Proposition [D.1,](#page-30-1) with probability at least 1− δ 2 , any µˆ ′ , αˆ ′ with |µˆ ′ <sup>|</sup> <sup>&</sup>gt; (1 +o(1)) cz log <sup>1</sup> n 1− 1 z and αˆ ′ > 0 satisfies that ψ(X, µˆ ′ , αˆ ′ ) ̸<sup>=</sup> <sup>0</sup>, and thus the solution <sup>µ</sup><sup>ˆ</sup> found by Estimator 1 by solving <sup>ψ</sup> <sup>=</sup> <sup>0</sup> satisfies <sup>|</sup>µˆ| ≤ (1 + <sup>o</sup>(1)) cz log <sup>1</sup> n 1− 1 z . We now remove the simplifying assumptions using the structural properties of Estimator [1.](#page-0-0) Let κ be the initial estimate in Step 1 of Estimator [1,](#page-0-0) say, computed by median-of-means. By Fact [D.4,](#page-32-0) with probability at least 1 − 1 2 δ,

$$|\kappa| \leq O\left(\left(\frac{\log \frac{1}{\delta}}{n}\right)^{1-\frac{1}{z}}\right)$$

and by the Lipschitz bound of Fact [A.17](#page-20-0) with Lipschitz constant O q log <sup>1</sup> δ n , this incurs an error on Estimator 1 of at most

$$O\left(\left(\frac{\log \frac{1}{\delta}}{n}\right)^{1-\frac{1}{z}} \cdot \sqrt{\frac{\log \frac{1}{\delta}}{n}}\right) \leq o\left(\left(\frac{\log \frac{1}{\delta}}{n}\right)^{1-\frac{1}{z}}\right)$$

Since c 1− <sup>1</sup> <sup>z</sup> is lower bounded by a universal constant, the above expression is also o cz log <sup>1</sup> δ n 1− 1 z .

Now consider any distribution with mean µ and z th moment Mz. By the affine invariance of Estimator [1](#page-0-0) of Fact [A.17,](#page-20-0) Estimator [1](#page-0-0) suffers a multiplicative factor of (Mz) 1 <sup>z</sup> in the estimation error.

Thus with probability at least 1 − δ, we have

$$|\hat{\mu}(X) - \mu| \leq (M_z)^{\frac{1}{2}} \cdot (1 + o(1)) \left( c_z \frac{\log \frac{1}{\delta}}{n} \right)^{1 - \frac{1}{2}}$$

as desired.

### D.1. Proof of Lemma [D.2](#page-31-0) and [D.3](#page-31-1)

In this section, we present the proof of Lemma [D.2,](#page-31-0) motivated by the discussion in Section [6.2](#page-7-2) and consequently [Lee &](#page-9-0) [Valiant](#page-9-0) [\(2022\)](#page-9-0), modeling the worst-case Chernoff bound as a max-min linear programming game. We later present the short proof of Lemma [D.3.](#page-31-1)

Lemma D.2. *Let* D *be an arbitrary distribution with mean* 0 *and* z *th moment* 1 *for some* 1 < z ≤ 2*. There exists a universal constant* c independent of z *where the following is true. Fixing* µˆ = 1 + <sup>c</sup> log log <sup>1</sup> δ log <sup>1</sup> δ ϵ *where* ϵ = cz log <sup>1</sup> δ n 1− 1 z *and* <sup>c</sup><sup>z</sup> = 2(5.6) <sup>1</sup> <sup>z</sup>−<sup>1</sup> −1 *, then for all* δ *smaller than some universal constant, and for all* wˆ ∈ [0.05, 555]*, there exists a vector* d( ˆw) := (dµ, dα) *where* d<sup>µ</sup> ≥ 0*, and both* nϵ log <sup>1</sup> |dµ| *and* |dα| *are bounded by a universal constant, such that*

$$x \leftarrow \mathbb{P}^n \left( \mathbf{d}(\hat{w}) \cdot \psi \left( X, \hat{\mu} = \epsilon', \hat{\alpha} = \frac{\log^2(1/\delta)}{3\hat{w}n^2\epsilon^2} \right) > \frac{1}{\log \frac{1}{\delta}} \right) \geq 1 - \frac{\delta}{\log^4 \frac{1}{\delta}}.$$

*Furthermore, for* wˆ = 0.05 *we have* d<sup>µ</sup> = √ <sup>3</sup>.75log <sup>1</sup> δ nϵ *,* d<sup>α</sup> = √ 3*; and for* wˆ = 555 *we have* d<sup>µ</sup> = 0*,* d<sup>α</sup> < 0*.*

*Proof.* We instead bound the contrapositive statement, namely,

$$x {}^{\mathbb{P}} \leftarrow D^n \left( \mathbf{d}(\hat{w}) \cdot \psi \left( X, \hat{\mu} = \epsilon', \hat{\alpha} = \frac{\log^2(1/\delta)}{3\hat{w}n^2\epsilon^2} \right) \leq \frac{1}{\log \frac{1}{\delta}} \right) \leq \frac{\delta}{\log^4 \frac{1}{\delta}}.$$

We start by applying a standard Chernoff bound analysis

$$\begin{aligned} & x \leftarrow D^n \left( \mathbf{d}(\hat{w}) \cdot \psi(X, \hat{\mu}, \hat{\alpha}) \leq \frac{1}{\log \frac{1}{\delta}} \right) \\ & = x \leftarrow D^n \left( \exp(-\mathbf{d}(\hat{w}) \cdot \psi(X, \hat{\mu}, \hat{\alpha})) \geq \exp\left(-\frac{1}{\log \frac{1}{\delta}}\right) \right) \end{aligned}$$

$$\begin{aligned} &\leq 2 \sum_{X \leftarrow D^n} (\exp(-\mathbf{d}(\hat{w}) \cdot \psi(X, \hat{\mu}, \hat{\alpha}))) \quad \text{by Markov's and } \exp\left(-\frac{1}{\log \frac{1}{\delta}}\right) \leq 2 \text{ for suff. small } \delta \\ &= 2 \sum_{x \leftarrow D} (\exp(-\mathbf{d}(\hat{w}) \cdot \psi(x, \hat{\mu}, \hat{\alpha})))^n \quad \text{by independence} \\ &= 2 \left( \exp\left(-d_\mu \hat{\mu} + d_\alpha \frac{1}{3n} \log \frac{1}{\delta}\right) \sum_{x \leftarrow D} (\exp(d_\mu x(1 - \min(\hat{\alpha}x^2, 1)) - d_\alpha \min(\hat{\alpha}x^2, 1))) \right)^n \end{aligned}$$

Motivated by our discussion in Section [6.2,](#page-7-2) we state the following technical claim, Lemma [D.5,](#page-34-0) reminiscent of the constraint in [\(2\)](#page-8-6). We formally prove Lemma [D.5](#page-34-0) in Appendix [D.2.](#page-35-0)

Lemma D.5. *For all* <sup>w</sup><sup>ˆ</sup> ∈ [0.05, 555]*, there exists* a > <sup>0</sup>, b *such that for all* <sup>1</sup> < z ≤ <sup>2</sup> *and* <sup>y</sup> ∈ <sup>R</sup>*, letting* <sup>c</sup><sup>z</sup> = 2(5.6) <sup>1</sup> <sup>z</sup>−<sup>1</sup> −1 *, the following holds true:*

$$ay(1 - \min(y^2, 1)) - b \min(y^2, 1) \leq \log \left( 1 + ay + |y|^z (3\hat{w})^{\frac{z}{2}} c_z^{z-1} \left( -1 + \frac{a}{\sqrt{3\hat{w}}} - \frac{b}{3} \right) \right) \quad (4)$$

*where* a *and* b *are bounded by constants. Further, for* wˆ = 0.05*, the pair* a = 0.75, b = √ 3 *works.*

For wˆ ∈ [0.05, 555), we use Lemma [D.5,](#page-34-0) substituting y = √ αxˆ , to choose d<sup>µ</sup> = √ αxˆ = log <sup>1</sup> δ nϵ√ 3 ˆw a, d<sup>α</sup> = b. In particular, for wˆ = 0.05, we have d<sup>µ</sup> = √ <sup>3</sup>.75log <sup>1</sup> δ nϵ and d<sup>α</sup> = √ 3. Then, the failure probability is bounded by

$$\begin{aligned} & 2 \left( \exp \left( -d_\mu \hat{\mu} + d_\alpha \frac{1}{3n} \log \frac{1}{\delta} \right) \underset{x \leftarrow D}{\mathbb{E}}_{y=x\sqrt{\hat{\alpha}}} \left( 1 + ay + |y|^z (3\hat{w})^{\frac{z}{2}} c_z^{z-1} \left( -1 + \frac{a}{\sqrt{3\hat{w}}} - \frac{b}{3} \right) \right) \right)^n \\ & = 2 \left( \exp \left( -d_\mu \hat{\mu} + d_\alpha \frac{1}{3n} \log \frac{1}{\delta} \right) \left( 1 + \left( \frac{\log \frac{1}{\delta}}{n\epsilon\sqrt{3\hat{w}}} \right)^z (3\hat{w})^{\frac{z}{2}} c_z^{z-1} \left( -1 + \frac{d_\mu \epsilon n}{\log \frac{1}{\delta}} - \frac{d_\alpha}{3} \right) \right) \right)^n \\ & \quad \text{since } D \text{ has mean } 0, z^{\text{th}} \text{ moment } 1 \\ & = 2 \left( \exp \left( -d_\mu \hat{\mu} + d_\alpha \frac{1}{3n} \log \frac{1}{\delta} \right) \left( 1 + \frac{\log \frac{1}{\delta}}{n} \left( -1 + \frac{d_\mu \epsilon n}{\log \frac{1}{\delta}} - \frac{d_v}{3} \right) \right) \right)^n \\ & \leq 2 \left( \exp \left( -d_\mu \hat{\mu} + d_\alpha \frac{1}{3n} \log \frac{1}{\delta} + \frac{\log \frac{1}{\delta}}{n} \left( -1 + \frac{d_\mu \epsilon n}{\log \frac{1}{\delta}} - \frac{d_\alpha}{3} \right) \right) \right)^n \\ & \quad \text{since } 1 + x \leq e^x \text{ for any } x \\ & = 2 \left( \exp \left( -d_\mu \hat{\mu} + d_\alpha \frac{1}{3n} \log \frac{1}{\delta} - \frac{1}{n} \log \frac{1}{\delta} + d_\mu \epsilon - d_\alpha \frac{1}{3n} \log \frac{1}{\delta} \right) \right)^n \\ & = 2 \exp \left( -d_\mu n(\hat{\mu} - \epsilon) - \log \frac{1}{\delta} \right) \\ & = 2 \exp \left( -\frac{a}{\sqrt{3\hat{w}}} c \log \log \frac{1}{\delta} - \log \frac{1}{\delta} \right) \quad \text{by choice of } d_\mu \text{ and } \hat{\mu} \\ & \leq \frac{\delta}{\log^4 \frac{1}{\delta}} \end{aligned}$$

as desired for large enough <sup>c</sup>, since √<sup>a</sup> 3 ˆw is greater than some positive constant.

For wˆ = 555, we use the following technical claim:

Claim D.6. *For all* 1 < z ≤ 2*, for all values of* y*, the following holds true:*

$$4 \min(y^2, 1) \leq \log(1 + 54|y|^2)$$

which is easily verifiable by two subclaims in the form of 4y <sup>2</sup> ≤ log(1 + 54y 2 ) for −1 ≤ y ≤ 1 and 4 ≤ log(1 + 54|y|) for |y| > 1.

We choose d<sup>µ</sup> = 0 and d<sup>α</sup> = −4. Substituting yields

$$\begin{aligned} & 2 \left( \exp \left( -\frac{4}{3n} \log \frac{1}{\delta} \right)_{x \leftarrow D} (\exp (4 \min(\hat{\alpha} x^2, 1))) \right)^n \\ & \leq 2 \delta^{4/3} \mathbb{E}_{x \leftarrow D} (1 + 54 \left( \sqrt{\hat{\alpha}} \right)^z |x|^z)^n \quad \text{by Claim D.6} \\ & = 2 \delta^{4/3} (1 + 54 \left( \frac{\log \frac{1}{\delta}}{n\epsilon\sqrt{3\hat{w}}} \right)^z)^n \quad \text{since } D \text{ has } z^{\text{th}} \text{ moment 1} \\ & \leq 2 \delta^{4/3} \exp \left( n \cdot 54 \left( \frac{\log \frac{1}{\delta}}{n\epsilon\sqrt{3 \cdot 555}} \right)^z \right) \quad \text{since } 1 + x \leq e^x \\ & = 2 \delta^{4/3} \exp \left( n \cdot 54 \left( \frac{1}{2^{1-\frac{1}{z}} 5.6^{\frac{2}{z}-1} \sqrt{1665}} \left( \frac{\log \frac{1}{\delta}}{n} \right)^{\frac{1}{z}} \right)^z \right) \quad \text{by definition of } \epsilon \\ & = 2 \delta^{4/3} \exp \left( \frac{54}{2^{z-15.6^{2-z}} \sqrt{1665^z}} \log \frac{1}{\delta} \right) \\ & \leq 2 \delta^{4/3} \delta^{-0.237} \quad \text{for all values of } 1 < z \leq 2 \\ & \leq 2 \delta^{1.096} \leq \frac{\delta}{\log^4 \frac{1}{\delta}} \quad \text{for suff. small } \delta \end{aligned}$$

as desired.

We have thus proven one of two components necessary to prove Proposition [D.1.](#page-30-1) We restate and prove the remaining component, which is a Lipschitz bound over the region covered by our finite mesh over wˆ ∈ [0.05, 555]:

Lemma D.3. *Consider an arbitrary set of* n *samples* X*. Consider the expressions* ψµ(X, µ, ˆ αˆ)*,* ψα(X, αˆ)*, reparameterized in terms of* wˆ = log<sup>2</sup> <sup>1</sup> δ 3 ˆαn2ϵ <sup>2</sup> *in place of* αˆ*. Suppose the equation* ψα(X, αˆ) = 0 *has a solution in the range* wˆ ∈ [0.05, 555]*. Then the functions* log <sup>1</sup> δ nϵ ψµ(X, µ, ˆ αˆ) *and* ψα(X, αˆ) *are Lipschitz with respect to* wˆ *on the entire interval* wˆ ∈ [0.05, 555]*, with Lipschitz constant* clog <sup>1</sup> δ *for some universal constant* c*.*

*Proof.* Consider the derivative with respect to <sup>w</sup><sup>ˆ</sup> of ψα(X, µ, <sup>ˆ</sup> <sup>α</sup>ˆ) = P<sup>n</sup> <sup>i</sup>=1 min log<sup>2</sup> <sup>1</sup> 3n2ϵ <sup>2</sup>wˆ x i , 1 − 1 3n log <sup>1</sup> δ . The wˆ derivative of min log<sup>2</sup> <sup>1</sup> δ 3n2ϵ <sup>2</sup>wˆ x 2 i , 1 is either − log<sup>2</sup> <sup>1</sup> δ 3n2ϵ <sup>2</sup>wˆ <sup>2</sup> x 2 <sup>i</sup> = − 1 wˆ αxˆ 2 i or 0, depending on which term in the min is the smallest, and in either case has magnitude at most <sup>1</sup> <sup>w</sup><sup>ˆ</sup> min(ˆαx<sup>2</sup> i , 1). Thus, the overall wˆ derivative of ψα(X, µ, ˆ αˆ) has magnitude at most <sup>1</sup> wˆ P <sup>i</sup> min(ˆαx<sup>2</sup> i , 1). Since by assumption P <sup>i</sup> min(ˆαx<sup>2</sup> i , 1) = <sup>1</sup> 3 log <sup>1</sup> δ for some wˆ ∈ [0.05, 555], the derivative with respect to wˆ must be within a constant factor of log <sup>1</sup> δ across the entire range, as desired.

Similarly, consider the derivative with respect to <sup>w</sup><sup>ˆ</sup> of <sup>ψ</sup>µ(X, µ, <sup>ˆ</sup> <sup>α</sup>ˆ) = P<sup>n</sup> <sup>i</sup>=1(ˆµ − <sup>x</sup>i(1 − min(ˆαx<sup>2</sup> i , 1))). The wˆ derivative of (ˆµ − xi(1 − min(ˆαx<sup>2</sup> i , 1))) is either − wˆ αxˆ 3 i or 0, depending on whether x<sup>i</sup> ≤ p 1/αˆ, and thus the magnitude of the entire derivative is bounded by <sup>1</sup> wˆ √ αˆ P <sup>i</sup> min(ˆαx<sup>2</sup> i , 1). Since P <sup>i</sup> min(ˆαx<sup>2</sup> i , 1) is bounded by a constant times log <sup>1</sup> δ , and <sup>1</sup> wˆ √ αˆ is bounded by a constant times √ wˆαˆ = √ 3nϵ log <sup>1</sup> since <sup>w</sup><sup>ˆ</sup> <sup>∈</sup> [0.05, 555], the magnitude of the derivative of log <sup>1</sup> nϵ ψµ(X, µ, ˆ αˆ) is bounded by a constant times log <sup>1</sup> δ , as desired.

#### D.2. Proof of Lemma [D.5](#page-34-0)

In this section, we prove the technical Lemma [D.5.](#page-34-0)

Lemma D.5. *For all* <sup>w</sup><sup>ˆ</sup> ∈ [0.05, 555]*, there exists* a > <sup>0</sup>, b *such that for all* <sup>1</sup> < z ≤ <sup>2</sup> *and* <sup>y</sup> ∈ <sup>R</sup>*, letting* <sup>c</sup><sup>z</sup> = 2(5.6) <sup>1</sup> <sup>z</sup>−<sup>1</sup> −1 *, the following holds true:*

$$ay(1 - \min(y^2, 1)) - b \min(y^2, 1) \leq \log \left( 1 + ay + |y|^z (3\hat{w})^{\frac{z}{2}} c_z^{z-1} \left( -1 + \frac{a}{\sqrt{3\hat{w}}} - \frac{b}{3} \right) \right) \quad (4)$$

*where* a *and* b *are bounded by constants. Further, for* wˆ = 0.05*, the pair* a = 0.75, b = √ 3 *works.*

*Proof.* We remark that Lemma [D.5](#page-34-0) is more nuanced than its counterpart in [Lee & Valiant](#page-9-0) [\(2022\)](#page-9-0), in part due to the introduction of a new variable z in the exponent. However, notice that taking the derivative with respect to z of the right hand side of [\(4\)](#page-34-2), omitting the outer logarithm—which is a monotone function—and omitting terms independent of z, yields

$$\frac{\partial}{\partial z} |y|^z (3\hat{w})^{\frac{z}{2}} c_z^{z-1} = (|y|\sqrt{3\hat{w}})^z \cdot z^{2-1} \cdot 5.6^{2-z} \cdot \log \left( \frac{2|y|\sqrt{3\hat{w}}}{5.6} \right)$$

whose sign is determined solely by the sign of log 2|y| √ 3 ˆw 5.6 , and thus is independent of z. This implies that the right hand side of [\(4\)](#page-34-2) is monotone in z, and thus it suffices to show that [\(4\)](#page-34-2) holds for the boundary cases z → 1 and z = 2. Since

$$\lim_{z \rightarrow 1} c_z^{z-1} = \lim_{z \rightarrow 1} (2(5.6)^{\frac{1}{z-1}-1})^{z-1} = 5.6,$$

we evaluate the limit z → 1 of [\(4\)](#page-34-2) by simply substituting z = 1 into the equation and replacing the term c z−1 <sup>z</sup> with 5.6.

We now choose the values of the parameters a, b. We follow the choices of [Lee & Valiant](#page-9-0) [\(2022\)](#page-9-0) for w < 5.5: for w = 0.05 we set a = 0.75, b = √ 3 as promised in the lemma statement; and for w ∈ (0.05, 5.5] we choose a = − √ 6+√ 6+96 ˆw 2 √ 2 ˆw , i.e., the positive root of the equation √ 2 ˆw(a <sup>2</sup> <sup>−</sup> 12) + √ 6a = 0, and b = 3 − a 2 . Note that these choices set a > 0, as promised in the lemma. For w ≥ 5.5 we differ from [Lee & Valiant](#page-9-0) [\(2022\)](#page-9-0) and instead choose the constants a = 4.2 and b = −3 (independent of wˆ).

Case z = 2 and w ∈ [0.05, 5.5): In this case our lemma is identical to its counterpart in [Lee & Valiant](#page-9-0) [\(2022\)](#page-9-0) and thus needs no proof.

Case z = 2 and w ≥ 5.5: In this case we choose a = 4.2 and b = −3, and thus Equation [4](#page-34-2) simplifies to

$$4.2y(1 - \min(y^2, 1)) + 3 \min(y^2, 1) \leq \log \left( 1 + 4.2y + y^2\sqrt{3\hat{w}} \cdot 2 \cdot 4.2 \right)$$

For y outside the range [−1, 1], the first two terms become the constant 3; and since the polynomial inside the logarithm expression on the right hand side is monotonically increasing in y outside of y ∈ [−1, 1], when wˆ ≥ 5.5, it is sufficient to prove the inequality for y ∈ [−1, 1].

Further, the logarithm term is the only term that depends on wˆ, and the logarithm monotonically increases with wˆ, so thus it is sufficient to prove the inequality for the smallest value of wˆ, namely wˆ = 5.5. We thus show:

$$4.2y(1-y^2) + 3y^2 \leq \log\left(1 + 4.2y + y^2\sqrt{3 \cdot 5 \cdot 5 \cdot 2 \cdot 4.2}\right)$$

To prove this, we note that both sides are smooth for y ∈ [−1, 1]; thus we take derivatives of both sides and set them equal to each other. The derivative of the left hand side is a quadratic polynomial; and the derivative of the right hand side is the multiplicative inverse of a quadratic polynomial. So thus the solutions to this equation are the solutions to a quartic. And it is straightforward to find these four solutions for y with a computer algebra package, and confirm that, at all of these four possible extrema and the points y = −1, y = 1, the desired inequality is true.

Case z = 1 and w ≥ 5.5 Next, we turn to the case of z = 1 and w ≥ 5.5, where we set a = 4.2 and b = −3. Equation [4](#page-34-2) now simplifies to a condition independent of wˆ:

$$4.2y(1 - \min(y^2, 1)) + 3 \min(y^2, 1) \leq \log(1 + 4.2y + |y| \cdot 5.6 \cdot 4.2)$$

As above, the expression inside the logarithm on the right hand side is increasing outside the range y ∈ [−1, 1] and the left hand side is constant outside this range, so it is sufficient to show this inequality for y ∈ [−1, 1] in which case it simplifies to

$$4.2y(1 - y^2) + 3y^2 \leq \log(1 + 4.2y + |y| \cdot 5.6 \cdot 4.2)$$

For negative y, the left hand side is convex and the right hand side is concave, so the left hand side minus the right hand side is convex, and its maximum must be attained at one of its two endpoints, y = −1 or y = 0. Numerically checking both cases confirms the inequality for y ≤ 0. For positive y, as in the previous case, we take derivatives of the left and right hand side and set them equal to each other, leaving us with a cubic equation, which thus has closed form solutions. We thus check that the desired inequality is true at all the positive roots of the cubic, along with the extreme points y = 0, y = 1.

Case z = 1 and w ∈ (0.05, 5.5): In this case we have chosen a = − √ 6+√ 6+96 ˆw 2 √ 2 ˆ<sup>w</sup> —the positive root of the equation √ 2 ˆw(a <sup>2</sup> <sup>−</sup> 12) + √ 6a = 0—and b = 3 − a 2 . Since a is an increasing function of wˆ it is easy to check that, for wˆ in our range (0.05, 5.5) we have a ∈ (0, 3.12]. Because of these relations between a, b, wˆ, we can simplify part of the expression in [\(4\)](#page-34-2) inside the logarithm as

$$\sqrt{3\hat{w}} \left( -1 + \frac{a}{\sqrt{3\hat{w}}} - \frac{b}{3} \right) = \frac{1}{2}a$$

Using this relation, and substituting b = 3 − a 2 into the left hand side of [\(4\)](#page-34-2) simplifies [\(4\)](#page-34-2) to:

$$ay(1 - \min(y^2, 1)) - (3 - \frac{a^2}{2}) \min(y^2, 1) \leq \log(1 + ay + 2.8a|y|)$$

As above, we point out that the left hand side is constant for y outside the range [−1, 1] and the right hand side is monotonic away from this interval, so it suffices to prove the inequality for y ∈ [−1, 1] in which case it simplifies to

$$ay(1 - y^2) - (3 - \frac{a^2}{2})y^2 \leq \log(1 + ay + 2.8a|y|) \quad (5)$$

We first show the y ≥ 0 case. Since a > 0 we have ay ≥ 0. Substituting ay → x above yields the following inequality, which we show for x ≥ 0 and a ∈ (0, 3.12]:

$$x + \frac{x^2}{2} - \frac{1}{a^2}(x^3 + 3x^2) \leq \log(1 + 3.8x)$$

The term x <sup>3</sup> + 3x is nonnegative for x ≥ 0, and thus the left hand side is increasing in a. Thus it suffices to prove the inequality for the maximum value of a = 3.12, and all x ≥ 0.

As above, we prove this by pointing out that both sides are smooth functions for x ≥ 0, so we take their derivatives and set them equal to each other, which yields a cubic equation in x. Our inequality takes its extreme values at either a positive root of the cubic, or at the boundary value x = 0; we thus confirm the inequality in these cases to prove it in general.

We now show the y ≤ 0 case.

We point out that the left hand side of Equation [5,](#page-37-0) ay(1−y )−(3− a 2 )y 2 , is increasing in y at y = 0, convex for sufficiently negative y, and can only transition from convex to concave once. Meanwhile, the right hand side, log (1 − 1.8ay), is decreasing everywhere for y ≤ 0, and concave everywhere.

Let c be the location where the cubic function ay(1 − y 2 ) − (3 − a 2 )y 2 transitions from convex to concave. For those y in the (possibly empty) interval [c, 0], the cubic is concave—since it is also increasing at y = 0, it must be increasing on this entire interval. Recalling that the right hand side of Equation [5](#page-37-0) is decreasing for y ≤ 0, the difference between the left and right hand sides attains its maximum in the interval y ∈ [c, 0] at y = 0, which is just 0, satisfying the inequality.

And for those y < c, where the left hand side is convex, then the difference between the left and right hand sides is convex, and thus its maximum must occur either at the left extreme, y = −1, or the right extreme, y = c; however the difference at y = c we already showed was at most the difference at y = 0, so overall, the maximum difference between the left and right hand sides must occur at either y = −1 or y = 0. As above, the y = 0 case is trivial, and it remains to prove the y = −1 case. For the y = −1 case Equation [5](#page-37-0) becomes:

$$\frac{a^2}{2} - 3 \leq \log(1 + 1.8a)$$

The left hand side is convex and the right hand side is concave; so we prove the inequality by numerically checking both endpoints: a = 0 and a = 3.12.

Case z = 1 and wˆ = 0.05: In this case we choose a = 0.75 and b = √ 3 and Equation [4](#page-34-2) becomes

$$0.75y(1 - \min(y^2, 1)) - \sqrt{3}\min(y^2, 1) \leq \log(1 + 0.75y + 2.086|y|)$$

where <sup>2</sup>.<sup>086</sup> is a lower bound on <sup>15</sup>√ 3 ˆw <sup>−</sup>1 + √<sup>a</sup> 3 ˆw − b 3 

As usual, for y outside [−1, 1] we point out that the left hand side is constant while the right hand side monotonically increases, so it suffices to show the inequality for y ∈ [−1, 1].

For y ∈ [−1, 1], we trivially lower bound the right hand side by log(1 + 0.75y) (dropping the 2.086|y| term), and lower bound this logarithm expression with the quadratic 0.75y − 0.75y 2 . The difference between the left hand side and this polynomial lower bound on the right hand side is thus −.75y <sup>3</sup> + (.75 − √ 3)y 2 , which is negative for y > <sup>1</sup> − √ 4 3 , which is below −1, and hence our inequality holds on the entire interval y ∈ [−1, 1].

Thus we have shown all cases of the desired inequality.

# E. Proof of Theorem [2.8](#page-3-2)

In this section, we present the proof of Theorem [2.8.](#page-3-2) We restate the theorem for completeness:

Theorem 2.8. *Let* D *be a distribution with mean* µ *and variance* σ 2 *.*

*Let* µˆ *denote Estimator [1](#page-0-0) on input parameter* δ *and* n *i.i.d. samples from* D*. Also let* X¯ <sup>n</sup> *denote the sample mean. Then, fixing* δ *and* D *and taking* n → ∞*, we have*

$$\sqrt{n}\hat{\mu} \xrightarrow{p} \sqrt{n}\bar{X}_n$$

*that is,* | √ nµˆ − √ nX¯ n| <sup>p</sup><sup>→</sup> <sup>0</sup>*, that* √ nµ<sup>ˆ</sup> *converges to* √ nX¯ <sup>n</sup> *in probability.*

*As a corollary, by the Central Limit Theorem, we have*

$$\sqrt{n}(\hat{\mu} - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

*That is,* µˆ *is asymptotically normal and efficient.*

*Proof.* Without loss of generality, by the shift-and-scale equivariance of both Lee and Valiant's estimator and the sample mean, we assume µ = 0 for notational simplicity, and that the variance of D is 1.

The second part of the theorem follows from the straightforward reasoning that, since Estimator [1](#page-0-0) converges to the sample mean (as claimed in the first part of the theorem statement), and since the sample mean converges to a Gaussian (from the Central Limit Theorem), then Estimator [1](#page-0-0) also converges to the same Gaussian. Formally, this uses Slutsky's theorem and the fact that convergence in probability implies convergence in distribution. Hence it only remains to show the first part of the theorem, that Lee and Valiant's estimator converges to the sample mean in probability, for fixed δ and as n → ∞.

The claim that √ nµˆ p→ √ nX¯ <sup>n</sup> is equivalent by definition to the statement that, for any fixed ϵ > 0,

$$\lim_{n \rightarrow \infty} \mathbb{P}(\sqrt{n}|\hat{\mu} - \bar{X}_n| > \epsilon) = 0$$

First, recall that µˆ differs from the sample mean by removing a total of <sup>1</sup> 3 log <sup>1</sup> <sup>δ</sup> weighted samples before taking the average. Thus, √ n|µˆ − X¯ <sup>n</sup>| is upper bounded by Θ log <sup>1</sup> δ |xmax|/ √ n where xmax is the largest sample in magnitude (recalling that we assumed µ = 0):

$$\begin{aligned} & \sqrt{n} \left| \kappa + \frac{1}{n} \sum_i (x_i - \kappa)(1 - \min(\alpha(x_i - \kappa)^2, 1)) - \frac{1}{n} \sum_i x_i \right| \\ &= \frac{1}{\sqrt{n}} \left| \sum_i (x_i - \kappa) \min(\alpha(x_i - \kappa)^2, 1) \right| \\ &\leq \frac{2|x_{\max}|}{\sqrt{n}} \sum_i \min(\alpha(x_i - \kappa)^2, 1) \end{aligned}$$

$$= \frac{2}{3} \log \frac{1}{\delta} \frac{|x_{\max}|}{\sqrt{n}}$$

Thus, the event √ n|µˆ − X¯ <sup>n</sup><sup>|</sup> > ϵ implies <sup>|</sup>xmax<sup>|</sup> > ϵ√ n Θ(log <sup>1</sup> δ ), which we now show is an event that occurs with probability → 0 as n → ∞.

We show the following claim, that the probability a *single* sample from D is larger than ϵ √ n Θ(log <sup>1</sup> δ ) is at most o(1/n).

Claim E.1. *Fix any* ϵ > 0 *and* δ ∈ (0, 1)*. Suppose we draw a single sample* X *from a distribution* D *with mean 0 and variance 1. Then,*

$$\mathbb{P}\left(|X| > \epsilon\sqrt{n} / \Theta\left(\log \frac{1}{\delta}\right)\right) = o\left(\frac{1}{n}\right)$$

*Here, the* o(·) *is in the limit where* ϵ*,* δ *and* D *are fixed, and* n → ∞*.*

We prove Claim [E.1](#page-39-1) at the end. Claim [E.1](#page-39-1) implies that the expected number of samples (among the n drawn) with magnitude exceeding ϵ √ n Θ(log <sup>1</sup> δ ) is o(1). Thus, by Markov's inequality, the probability that at least one sample exceeds that threshold is also o(1), showing Theorem [2.8.](#page-3-2)

We finish with the short proof of Claim [E.1.](#page-39-1) Here, for a generic non-negative random variable Y , we use a refined version of Markov's inequality that

$$\mathbb{P}(Y > a) \leq \frac{\mathbb{E}[Y \mathbb{1}[Y \geq a]]}{a}$$

Applying this to X<sup>2</sup> in Claim [E.1,](#page-39-1) we have

$$\mathbb{P}\left(X^2 > \epsilon^2 n / \Theta\left(\log^2 \frac{1}{\delta}\right)\right) \leq \mathbb{E}\left[X^2 \mathbb{1}\left[X^2 \geq \epsilon^2 n / \Theta\left(\log^2 \frac{1}{\delta}\right)\right]\right] \cdot \frac{\Theta\left(\log^2 \frac{1}{\delta}\right)}{\epsilon^2 n}.$$

Since <sup>E</sup>[X<sup>2</sup> ] = 1 and ϵ, δ and D are fixed, we have <sup>E</sup> -X<sup>2</sup><sup>1</sup> -X<sup>2</sup> ≥ ϵ <sup>2</sup>n Θ log<sup>2</sup> <sup>1</sup> δ = o(1) as n → ∞. Thus, the right hand side in the above inequality is o(1/n), showing Claim [E.1.](#page-39-1)

This completes the proof that µˆ converges to X¯ <sup>n</sup> in probability, showing the theorem.

# F. Proofs of the Folklore Robustness of Median-of-Means

For completeness, we provide formal proofs of Facts [A.18](#page-21-0) and [A.23,](#page-25-0) two folklore facts about the robustness of the median-of-means estimator against adversarial corruption, which we use in our proofs of Theorems [2.2](#page-2-0) and [2.3.](#page-2-1)

Fact A.18 (Folklore). *For any distribution* D *with mean* µ *and standard deviation* σ*, let* X˜ *be a set of* n η*-corrupted samples from* D*. The median-of-means estimate* κ *from grouping samples into* O(log <sup>1</sup> δ ′ + ηn) *buckets, on input* X˜*, satisfies*

$$\mathbb{P} \left( |\kappa - \mu| \geq O \left( \sigma \sqrt{\frac{\log \frac{1}{\delta'}}{n} + \eta} \right) \right) \leq \frac{1}{16} \delta'$$

*Proof.* Let k denote the chosen amount of buckets. Choose k = 16(log <sup>1</sup> δ ′ + ηn). Let µ<sup>i</sup> denote the mean of the i-th bucket *before any adversarial corruption*. Then each µ<sup>i</sup> is independent with variance σ q16(log <sup>1</sup> <sup>δ</sup>′ <sup>+</sup>ηn) n . By Chebyshev's inequality, <sup>P</sup>[|µ<sup>i</sup> − µ| ≥ 4σ q16(log <sup>1</sup> <sup>δ</sup>′ <sup>+</sup>ηn) n ] ≤ 1 <sup>16</sup> . Let d<sup>i</sup> denote the indicator variable for the event |µ<sup>i</sup> − µ| ≤ 4σ q16(log <sup>1</sup> <sup>δ</sup>′ <sup>+</sup>ηn) n , then <sup>E</sup>(di) ≥ 15 <sup>16</sup> . By Hoeffding's inequality, <sup>P</sup>[ P i d<sup>i</sup> ≤ 9(log <sup>1</sup> δ ′ + ηn)] ≤ e −4.5(log <sup>1</sup> <sup>δ</sup>′ <sup>+</sup>ηn) ≤ e −(log <sup>1</sup> <sup>δ</sup>′ <sup>+</sup>ηn) , which is at most e − log <sup>16</sup> <sup>δ</sup>′ = <sup>16</sup> δ ′ for large enough n. Note that for |κ−µ| ≥ 4σ q16(log <sup>1</sup> <sup>δ</sup>′ <sup>+</sup>ηn) n , at most k/2 = 8(log <sup>1</sup> δ ′ +ηn) buckets can have |µ<sup>i</sup> − µ| ≤ 4σ q16(log <sup>1</sup> <sup>δ</sup>′ <sup>+</sup>ηn) n . Accounting for the adversarial corruption, which can affect at most ηn buckets, at most 9(log <sup>1</sup> δ ′ + ηn) buckets can have |µ<sup>i</sup> − µ| ≤ 4σ q16(log <sup>1</sup> <sup>δ</sup>′ <sup>+</sup>ηn) n *before adversarial corruption*, which happens with probability at most <sup>1</sup> <sup>16</sup> δ ′ as desired.

Fact A.23 (Folklore). *For any distribution* D *with mean* µ *and standard deviation* σ*, let* X˜ *be a set of* n η*-corrupted samples from* D*. Assuming that* η ≤ 24n log <sup>1</sup> δ *, the median-of-means estimate* κ *from grouping samples into* O(log <sup>1</sup> δ ) *buckets, on input* X˜*, satisfies*

$$\mathbb{P} \left( |\kappa - \mu| \geq O \left( \sigma \sqrt{\frac{\log \frac{1}{\delta}}{n}} \right) \right) \leq \frac{1}{22} \delta$$

*Proof.* Let k denote the chosen amount of buckets. Choose k = 16log <sup>1</sup> δ . Let µ<sup>i</sup> denote the mean of the i-th bucket *before any adversarial corruption*. Then each µ<sup>i</sup> is independent with variance σ q 16log <sup>1</sup> δ n . By Chebyshev's inequality, <sup>P</sup>[|µi−µ| ≥ 4σ q 16log <sup>1</sup> δ n ] ≤ 1 <sup>16</sup> . Let d<sup>i</sup> denote the indicator variable for the event |µi−µ| ≤ 4σ q 16log <sup>1</sup> δ n , then <sup>E</sup>(di) ≥ <sup>16</sup> . By Hoeffding's inequality, <sup>P</sup>[ P i d<sup>i</sup> ≤ 9log <sup>1</sup> δ ] ≤ e −4.5log <sup>1</sup> <sup>δ</sup> , which is at most e − log <sup>22</sup> <sup>δ</sup> = <sup>22</sup> δ for sufficiently small δ. Note that for |κ − µ| ≥ 4σ q 16log <sup>1</sup> δ n , at most k/2 = 8log <sup>1</sup> δ buckets can have |µ<sup>i</sup> − µ| ≤ 4σ q16(log <sup>1</sup> <sup>δ</sup>′ <sup>+</sup>ηn) n . Accounting for the adversarial corruption, which can affect at most ηn ≤ log <sup>1</sup> δ buckets, at most 9log <sup>1</sup> δ buckets can have |µi−µ| ≤ 4σ q 16log <sup>1</sup> δ n *before adversarial corruption*, which happens with probability at most <sup>1</sup> <sup>22</sup> δ as desired.