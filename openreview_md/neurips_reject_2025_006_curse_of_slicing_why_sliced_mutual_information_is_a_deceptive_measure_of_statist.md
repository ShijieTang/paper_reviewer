# Curse Of Slicing: Why Sliced Mutual Information Is A Deceptive Measure Of Statistical Dependence

Anonymous Author(s)
Affilation, Address anon.email@example.org

## Abstract

Sliced Mutual Information (SMI) is widely used as a scalable alternative to mutual information for measuring non-linear statistical dependence. Despite its advantages, such as faster convergence, robustness to high dimensionality, and nullification only under statistical independence, we demonstrate that SMI is highly susceptible to data manipulation and exhibits counterintuitive behavior. Through extensive benchmarking and theoretical analysis, we show that SMI saturates easily, fails to detect increases in statistical dependence (even under linear transformations designed to enhance the extraction of information), prioritizes redundancy over informative content, and in some cases, performs worse than simpler dependence measures like the correlation coefficient.

SMI
max-SMI
optimal-SMI
mutual information perceived position copula actual position our contribution Ability to capture complex statistical dependencies E
stimati on c o mplexity correlation

```
 1
 2
 3
 4
 5
 6
 7
 8
 9
10

```

11 12 13 14 15 16 17 18 19

## 1 Introduction 20

21 22 23 Mutual information (MI) is a fundamental and invariant measure of nonlinear statistical dependence between two random vectors, defined as the Kullback-Leibler divergence between the joint distribution and the product of marginals [1]:
(;  ) = D(ℙ, ‖ ℙ ⊗ ℙ
).

24 25 26 27 28 29 30 Due to several outstanding properties, such as nullification only under statistical independence, invariance to invertible transformations, and ability to capture non-linear dependencies, MI is used extensively for theoretical analysis of overfitting [2], [3], hypothesis testing [4], feature selection [5], [6], [7], representation learning [8], [9], [10], [11], [12], [13], and studying the mechanisms behind generalization in deep neural networks (DNNs) [14], [15], [16], [17].

In practical scenarios, ℙ,
 and ℙ ⊗ ℙ
 are unknown, requiring MI to be estimated from finite samples. Despite all the aforementioned merits, this reliance on empirical estimates leads to the curse Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Do not distribute.

31 32 33 34 35 of dimensionality: the sample complexity of MI grows exponentially with the number of dimensions [18], [19]. A common strategy to mitigate this issue is to use alternative measures of statistical dependence that are more stable in high dimensions. However, such measures usually offer only a fraction of MI capabilities. Therefore, it is crucial to maintain a balance between robustness to the curse of dimensionality and the ability to detect complex dependency structures.

36 37 38 39 40 To strike this balance, popular techniques often retain MI as a backbone statistical measure but employ dimensionality reduction before estimation. While some studies explore sophisticated nonlinear compression methods [17], [20], others favor more scalable linear projection approaches [21], [22], [23], [24], [25]. Among the latter group, the *Sliced Mutual Information* (SMI) [22], [23] stands out, leveraging random projections to cover all directions uniformly:

$${\mathsf{S l}}(X;Y)={\frac{1}{\oint_{{\mathsf{S}}^{d_{x-1}}}\mathrm{d}\theta}}{\frac{1}{\oint_{{\mathsf{S}}^{d_{y-1}}}\mathrm{d}\phi}}\oint_{{\mathsf{S}}^{d_{x-1}}}\oint_{{\mathsf{S}}^{d_{y-1}}}\mathrm{l}(\theta^{\mathsf{T}}X;\phi^{\mathsf{T}}Y)\,\mathrm{d}\theta\,\mathrm{d}\phi.$$
$$(1)$$
 )dd. (1)
41 42 43 44 45 46 47 Uniform slicing allows SMI to maintain some crucial properties of MI (e.g., being zero if and only if and  are independent), while remaining completely free from additional optimization problems (e.g., from finding optimal projections, as in [24], [25]). Combined with fast convergence rates, this has established SMI as a scalable alternative to MI. Consequently, it has been widely adopted for studying DNNs [26], [27], [28], [29], [30], deriving generalization bounds [31], independence testing [32] and auditing differential privacy [33]. It was also proposed to use SMI for feature selection [22] and preventing mode collapse in generative models [23].

48 49 50 51 52 Despite its popularity, the research community has largely overlooked potential shortcomings of SMI. Some studies prematurely attribute their results to underlying phenomena without rigorously investigating whether they stem from artifacts introduced by random projections. Furthermore, existing works fail to comprehensively address issues related to random slicing, focusing primarily on suboptimality of random projections for information preservation [24], [25].

53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 Contribution. In this article, we address this gap by systematically analyzing SMI across diverse settings, demonstrating that it frequently exhibits counterintuitive behavior and fails to accurately capture statistical dependence dynamics. Our key contributions are:
1. **Saturation and Sensitivity Analysis.** Through theoretical analysis and extensive benchmarking, we show that SMI saturates prematurely, even for low-dimensional synthetic problems, and fails to detect significant increases in statistical dependence.

2. **Redundancy Bias.** We refute the prevailing assumption that SMI favors linearly extractable information by constructing an explicit example where introducing such structure increases MI and even linear correlation, but decreases SMI. In fact, we show that SMI prioritizes information redundancy over information content. We argue that this bias can lead to catastrophic failures in some applications, e.g. collapses in representation learning.

3. **Curse of Dimensionality.** We revisit the dynamics of SMI for increasing dimensionality and argue that SMI is, in fact, cursed, with the curse of dimensionality manifesting itself not through sample complexity, but via asymptotic decay to zero in high-dimensional regimes due to diminishing redundancy.

4. **Reestablishing the Trade-off.** Finally, we discuss to which extent the aforementioned problems can be solved by using non-uniform/non-random slicing strategies, and how they affect the tradeoff between scalability and utility of different measures of statistical dependence.

Our paper is structured as follows. In Section 2, we provide the mathematical background that is necessary for our analysis. In Section 3, we discuss previous findings which are related to the research topic of this work. Section 4 consists of our main theoretical results, with the complete proofs being provided in Section B. In Section 5, we employ synthetic benchmarks to show the disconnection between dynamics of MI and SMI. Section 6 illustrates that tasks related to SMI maximization may yield degenerate solutions, contrary to MI maximization. Finally, we discuss our results in Section 7.

## 77 2 Preliminaries

78 79 80 81 82 83 84 Elements of Information Theory. Let (Ω, ℱ, ℙ) be a probability space with sample space Ω, -algebra ℱ, and probability measure ℙ defined on ℱ. Consider random vectors  : Ω → ℝ
 and
 : Ω → ℝ
 with joint distribution ℙ,
 and marginals ℙ and ℙ
, respectively. Wherever it is needed, we assume the relevant Radon-Nikodym derivatives exist. For any probability measure ℚ ≪
ℙ, the Kullback-Leibler (KL) divergence is D(ℚ ‖ ℙ) = ℚ[log d ℚ
d ℙ
], which is non-negative and vanishes if and only if (iff) ℙ = ℚ. The mutual information (MI) between  and  quantifies the divergence between the joint distribution and the product of marginals:

$$\mathbb{I}(X;Y)=\mathbb{E}\log{\frac{\operatorname{d}\mathbb{P}_{X,Y}}{\operatorname{d}\mathbb{P}_{X}\otimes\mathbb{P}_{Y}}}=\operatorname{D}_{\operatorname{KL}}{\big(}\mathbb{P}_{X,Y}\,{\big\|}\,\mathbb{P}_{X}\otimes\mathbb{P}_{Y}{\big)}.$$
$\left(2\right)^3$
85 86 87 88 89 When ℙ admits a probability density function (PDF) () with respect to (w.r.t.) the Lebesgue measure, the differential entropy is defined as () = − [log ()], where log(⋅) denotes the natural logarithm. Likewise, the joint entropy (,  ) is defined via the joint density (,  ),
and conditional entropy is ( |  ) = − [log ( |  )] = − 
[ | log ( |  )]. Under the existence of PDFs, MI satisfies the identities
(;  ) = () − ( |  ) = ( ) − ( | ) = () + ( ) − (,  ). (2)
90 91 92 93 94 In this work, we denote by M the normalized Haar (uniform) probability measure on a compact manifold M, i.e., the unique bi‑invariant measure satisfying M(M) = 1. Hence, to sample uniformly from specific spaces we write W ∼ O(),  ∼ 
−1 , A ∼ St(,), indicating draws from the Haar measures on orthogonal group O() = {Q ∈ ℝ
×: QQ = QQ = I}, the unit sphere 
−1 =
{ ∈ ℝ
: ‖‖2 = 1}, and the Stiefel manifold St(, ) = {Q ∈ ℝ
×: QQ = I}, respectively.

95 96 97 Sliced Mutual Information. To mitigate the curse of dimensionality, one may average MI over all -dimensional projections. The -sliced mutual information (-SMI) [23] between  and  is defined as

$$\mathsf{Sl}_{k}(X;Y)=\int_{\mathrm{St}(k,d_{x})}\int_{\mathrm{St}(k,d_{y})}\mathsf{I}(\Theta^{\mathsf{T}}X;\Phi^{\mathsf{T}}Y)\,\mathrm{d}\mu_{\mathrm{St}(k,d_{x})}(\Theta)\,\mathrm{d}\mu_{\mathrm{St}(k,d_{y})}(\Phi),$$

98 which can be efficiently estimated. Setting  = 1 recovers the standard sliced mutual information (1).

## 99 3 Background

100 101 102 103 104 105 106 107 Merits of SMI are straightforward and have been investigated thoroughly in [22], [23]. We remind the reader of the two most important of them: 1. **Scalability** (i.e., fast convergence in high dimensions), enabled by low-dimensional projections.

2. **Nullification Property** (i.e., (;  ) = 0 iff  and  are independent), which stems from the projections being random and independent.

In contrast, demerits of SMI are not very obvious and not well-covered in the literature. In this section, we recapitulate and analyze previous works which address the shortcomings of SMI. To facilitate the analysis, we divide them into three main categories.

108 109 110 111 112 Suboptimality of random slicing. In [24] and [25], it is argued that a uniform slicing strategy can produce suboptimal projections, impairing SMI's ability to capture dependencies in the presence of noisy or non-informative components. To address this issue, [24] proposed max-sliced MI (mSMI), which selects non-random projectors that maximize the MI between projected representations. This approach is also claimed to improve interpretability and convergence rates.

113 114 115 116 117 However, deterministic slicing may overlook dependencies captured by non-optimal components. To mitigate this, [25] extends the max-sliced approach by optimizing SMI over probability distributions of projectors, with regularization to maintain slice diversity. While the authors emphasize that optimization should occur over joint distributions, their motivation primarily addresses the issue of non-optimal *marginal* distributions of  and  - specifically, the presence of non‑informative 118 119 120 components in  and  . We contend that this represents only a partial understanding of the problem, as many SMI artifacts arise from other factors. Needless to say that optimization over probability distributions is also a heavy burden, which does not align with the slicing philosophy.

121 122 123 124 125 126 127 128 Data Processing Inequality violation. A fundamental property of MI is that it cannot be increased by deterministic processing or, more generally, by Markov kernels. Furthermore, MI is preserved under invertible transformations. This is formalized by the *data processing inequality* (DPI). Theorem 3.1. (Theorem 3.7 in [1]) For a Markov chain  →  → , (;  ) ≥ (;). Additionally, if  = ( ) where  is measurably invertible, then equality holds. In contrast to MI, SMI violates the DPI (see Section 3.2 in [22] for an example). While the intuition behind DPI is clear (raw data already contains full information, and processing can only destroy it), the implications of DPI violation are less straightforward.

129 130 131 132 133 Existing works suggest that SMI's violation of DPI can reflect a preference for linearly extractable features, framing this as a useful property that aligns with the informal understanding of "practically available" (i.e., easily accessible) information [22], [26], [30]. However, this interpretation can be misleading if the factors behind SMI increases are misidentified. Our analysis reveals that this is indeed the case, as SMI exhibits more inherent biases than previously recognized.

134 135 136 137 138 139 140 Asymptotics in high-dimensional regime. Convergence analysis suggests that the sample complexity of SMI estimation is far less sensitive to data dimensionality compared to that of MI. In fact, it has been argued that the estimation error may even decrease with dimensionality in some cases (see Remark 4 in [23]). However, an analysis of SMI itself reveals that this behavior may result from the fact that SMI can decrease as dimensionality grows. Specifically, Theorem 3 in [23] provides an asymptotic expression (as  → ∞) for SMI in the case of jointly normal  and  , which decays hyperbolically with  under some circumstances.

141 142 143 To date, no explanation for this phenomenon has been provided in the literature. We therefore elaborate on this finding by deriving non-asymptotic expressions, along with experimental results for non-Gaussian data, which reveal further nuances behind the decay.

## 144 4 Theoretical Analysis

145 146 147 We start our analysis with considering a simple example, which (a) admits closed-form expression for SMI and (b) is capable of illustrating severe problems of the quantity in question. Lemma 4.1. Consider the following pair of jointly Gaussian -dimensional random vectors:

$$(X,Y)\sim{\mathcal{N}}{\bigg(}0,{\bigg(}{\underset{\rho\mathbf{I}}{\mathbf{I}}}\ \rho\mathbf{I}{\bigg)}{\bigg)},\ \ \rho\in(-1;1).$$

148 In this setup, MI and SMI can be calculated analytically:

$$\mathsf{I}(X;Y)=-\frac{d}{2}\log(1-\rho^{2}),\qquad\mathsf{SI}(X;Y)=\frac{\rho^{2}}{2d}\,_{3}F_{2}\biggl(1,1,\frac{3}{2};\frac{d}{2}+1,2;\rho^{2}\biggr),$$

149 where 32 is the *generalized hypergeometric function*. Additionally, the following limits hold:

$$\begin{array}{l l}{{\operatorname*{lim}_{d\to\infty}\operatorname{l}(X;Y)=+\infty}}&{{\operatorname*{lim}_{d\to\infty}\operatorname*{Sl}(X;Y)=0}}\\ {{}}&{{\operatorname*{lim}_{\rho^{2}\to1}\operatorname{l}(X;Y)=+\infty}}&{{\operatorname*{lim}_{\rho^{2}\to1}\operatorname*{Sl}(X;Y)=\psi(d-1)-\psi\left({\frac{d-1}{2}}\right)-\log2\leq{\frac{3}{d-1}},}}\end{array}$$

150 with  being the *digamma function*.

151 152 153 154 155 Note that while MI correctly captures the growing statistical dependence as  → ∞ (since additional components contribute shared information), SMI drops to zero, exposing a fundamental problem. This issue was briefly noted in [23], but only through providing an asymptotic expression without further discussion. We interpret this behavior as a distinct manifestation of the curse of dimensionality: as  grows, SMI uniformly decays to zero and becomes ineffective for statistical analysis.

0.8 156 157
 = 2 = 3 = 4 = 8 = 16 0 2 4 6 8 10
(;  )/, nats 0 0.2 0.4 0.6 0.8 1

(

; 
 
)/
(
; ) |=
1 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 0.6

(

; 
 
), nat s

 = 2 = 3 = 4 = 8 = 16 0.4 0.2 0 2 4 6 8 10
(;  )/, nats 0
176 177 178 179 180 181 182 The second pair of limits reveals another critical flaw of SMI. When 2 → 1, the - relationship becomes deterministic - a property MI reflects successfully. In stark contrast, SMI remains bounded by a dimension-dependent factor that decays hyperbolically. Furthermore, plotting SMI against MI shows this bound is reached prematurely, demonstrating SMI's **rapid saturation** with increasing dependence (Figure 2). In this saturated regime, SMI becomes effectively insensitive to further growth in shared information. Moreover, this renders estimates of SMI for different dimensionalities fundamentally incomparable, as they are theoretically bounded by factors depending on .

183 184 185 186 187 188 189 These phenomena can not be explained by suboptimality of individual projections. In fact, each individual projection is optimal, as (
;  ) does not depend on  in this particular example.

The proof of Lemma 4.1 suggests that the problem arises from the majority of *pairs* of projectors being suboptimal, yielding near-independent and in the most outcomes, even for  = 2.

Although similar analysis for -SMI is extremely challenging, we argue that the problems in question prevail even when employing -rank projectors.

Proposition 4.2. Under the setup of Lemma 4.1, -SMI has the following integral representation

$$\mathsf{Sl}_{k}(X;Y)=-\frac{1}{2}\int_{[0,1]^{k}}\sum_{i=1}^{k}\log(1-\rho^{2}\lambda_{i})\,p(\boldsymbol{\lambda})\,\mathrm{d}\boldsymbol{\lambda},$$
$$190$$
where $p(\mathbf{\lambda})\propto\prod_{i<j}|\lambda_{j}-\lambda_{i}|\underbrace{\prod_{i=1}^{k}\left(1-\lambda_{i}\right)^{(d-2k-1)/2}}_{\left(1\right)}$.  
(⋆)
191 192 193 194 195 196 197 198 199 Remark. **4.3.** As the dimension  grows, the term (⋆) asymptotically concentrates the eigenvalues near zero, leading to the decay of  to zero.

We argue that the limitations we uncovered can be attributed to a strong bias of SMI toward information redundancy. That is, SMI favors repetition of information across different axes, and suffers from the curse of dimensionality if  and  have high entropy. The following proposition and remark present a simple example to clarify this bias.

Proposition 4.4. Let  and  be , -dimensional random vectors correspondingly, with ,  <
. Let A ∈ ℝ
× and B ∈ ℝ
× be matrices of ranks , . Then (A; B ) = (;  ).

Corollary 4.5. Consider the following pair of jointly Gaussian -dimensional random vectors:

$$(X,Y)\sim{\mathcal{N}}{\bigg(}0,{\bigg(}{\underset{\rho\mathbf{J}}{\mathbf{J}}}\ \rho\mathbf{J}{\bigg)}{\bigg)},\quad\rho\in(-1;1),$$

200 where J =  ⋅ 
 with 
 = (1, …, 1). Then 
(;  ) = (;  ) = −
1 2 log(1 − 
2).

201 202 Remark. **4.6.** Applying  ⋅ 

1 to the random vectors from Lemma 4.1 individually yields the example from Corollary 4.5. Therefore, this linear transform increases SMI despite decreasing MI.

## 203 4.1 Extension To Optimal Slicing

204 205 206 207 208 Although our work primarily focuses on conventional (average) sliced mutual information (SMI), as it is the most widely used variant, we also provide some intuition regarding the limitations of its "optimal" counterparts: max-sliced MI (mSMI) [24] and *optimal-sliced* MI (oSMI) [25]. Since mSMI is a special case of oSMI without regularization constraints, we restrict our discussion to mSMI, though our reasoning extends to oSMI as well. The -mSMI is defined as:

(;  ) = sup Θ∈ St(,) Φ∈ St(,) (Θ; Φ
 ) (3)
209 210 211 212 213 214 215 To highlight the shortcomings of linear compression, we revisit a Gaussian example. The following proposition demonstrates that even in this simple setting, mSMI captures only a subset of dependencies and can exhibit opposite trends to MI. This occurs, for instance, when dependencies become more evenly distributed across components, which again returns us to the redundancy bias.

Proposition 4.7. (Proposition 2 in [24]) Let (,  ) ∼ (, Σ), with marginal covariances Σ, Σ and cross-covariance Σ. Suppose the matrix Σ
−
1 2 Σ Σ
−
1 2 exists, and let {} =1 denote its singular values in descending order, where  = min(, ). Then

$$\mathsf{I}(X;Y)=-\frac{1}{2}\sum_{i=1}^{d}\log(1-\rho_{i}^{2}),\qquad\overline{{{\mathsf{S}}}}\mathsf{I}_{k}(X;Y)=-\frac{1}{2}\sum_{i=1}^{k}\log(1-\rho_{i}^{2}).$$
$$\left({\mathbf{3}}\right)$$

## 216 5 Synthetic Experiments

217 218 219 220 221 222 223 To complement the theoretical analysis from the previous section and address complex, non-Gaussian cases, we conduct an extensive benchmarking of SMI using synthetic tests from [34], based on the works of [35], [36]. This benchmark suite is used to evaluate MI estimators. However, we do not assess whether SMI estimates converge to ground-truth MI values. SMI is a distinct measure of statistical dependance, and should not be viewed as an approximation of MI. Instead, our analysis focuses on the relationship between the two measures: since MI captures the true degree of statistical dependence, opposing trends in MI and SMI reveal problems with the latter quantity.

224 225 226 227 228 For the experiments, we use *correlated normal, correlated uniform, smoothed uniform* and loggamma-exponential distributions, for which the ground-truth value of MI is available. To increase the dimensionality, we use independent components with equally distributed per-component MI. These setups will be referred to as "randomized" and "non-randomized" correpsondingly. For each distribution, we vary both the data dimensionality () and the projection dimensionality ( < ).

229 230 231 232 To estimate MI between projections, we use the KSG estimator [35] with the number of neighbors fixed at 1. For each configuration, we conduct 10 independent runs with different random seeds to compute means and standard deviations. Our experiments use 10 4 samples for (,  ) and 128 samples for (Θ, Φ).

233 234 235 236 237 238 To experimentally verify saturation, we plot SMI against MI normalized by dimensionality  in Figure 3. The plots clearly show that SMI reaches a plateau relatively early for all the featured distributions. The results for the normal distribution also align well with those from Lemma 4.1. We further confirm the saturation of -SMI for  ∈ {2, 3} experimentally in Section C. Finally, we plot the saturated values against  on a log-log scale, demonstrating that the 1/ trend from Lemma 4.1 also holds for non-Gaussian distributions.

## 239 6 Smi For Infomax-Like Tasks

240 241 242 243 244 Since mutual information is interpretable and captures non-linear dependencies, it is widely used as a training objective. Many applications involve maximizing MI (InfoMax) for feature selection [5], [6], [7] and self-supervised representation learning [8], [9], [10], [11], [12], [13]. However, due to the curse of dimensionality, alternative objectives have been proposed, with some works using sliced mutual information maximization for feature extraction [22] and disentanglement in InfoGAN [23].

245 246 247 248 249 250 In this section, we argue that SMI is not a suitable alternative to MI for InfoMax tasks. Since SMI exhibits a strong preference for redundancy, SMI maximization may lead to collapsed (high-redundancy) solutions. We demonstrate this through two experiments. Firstly, we revisit the Gaussian noisy channel to demonstrate that SMI favors linear mappings which decrease robustness to noise. Then, we consider a self-supervised representation learning task and show that using SMI immediately leads to collapsed representations.

251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 0 2 4 6 8 10
(;  )/, nats 0 0.2 0.4 0.6 0.8 1 1.2 0 2 4 6 8 10
(;  )/, nats 0 0.2 0.4 0.6 0.8 1 1.2
 = 2 = 3 = 4 = 6 = 8
 = 2 = 3 = 4 = 6 = 8

(
; 
 )
, nat s

(
; 
 )
, nat s

(a) Correlated Normal
(b) Correlated Uniform 0 2 4 6 8 10
(;  )/, nats 0 0.2 0.4 0.6 0.8 1 1.2 0 2 4 6 8 10
(;  )/, nats 0 0.2 0.4 0.6 0.8 1 1.2
 = 2 = 3 = 4 = 6 = 8
 = 2 = 3 = 4 = 6 = 8

(
; 
 )
, n ats

(
; 
 )
, n ats
(c) Smoothed Uniform
(d) Log-Gamma-Exponential corr. corr. U sm. U LGE
corr. corr. U sm. U LGE
280 281 282 283 284 285 0.1 1 0.1 1
(
; 
 )
, n ats
(
; 
 )
, n ats 10 dimensionality 0.01 10 dimensionality 0.01 286 287 288
(a)  = 1
(b)  = 2 289

## 293 6.1 Gaussian Channel

294 295 296 297 298 Let  be a zero-mean -dimensional random vector, and let  ∼ (0, I) be an independent noise. Additive white noise Gaussian (AWGN) channel is defined as  →  + . Maximization of (;  + ) w.r.t. the distribution of  is a classical information transmission problem, which arises in many fields under the Gaussian noise assumption. Given energy constraints, it admits an analytical solution [37]:

$$\sup_{\mathbb{E}\,X_{i}^{2}=1}\mathbb{I}(X;X+Z)=\frac{d}{2}\log\biggl{(}1+\frac{1}{\sigma^{2}}\biggr{)},\qquad X_{\rm opt}\sim\mathcal{N}(0,\mathbb{I})\tag{4}$$

It is somewhat intuitive that unit covariance matrix allows for more information to be transmitted, as all the components of  are utilized to full extent. However, due to the redundancy bias, SMI prefers less robust distributions. To demonstrate this, we consider two linear normalization mappings which impose energy constraints on a vector  with zero mean and covariance Σ: 1. *Whitening*: Σ
−1/2; 2. *Standardization*: D−1/2, where D = diag(Σ). We conduct numerical experiments for  = 0.1, ′ ∼ A ⋅ U([−1; 1]
5) and ″ ∼ A ⋅ (0,I5),
where A = 10
−2⋅ I +  ⋅ 
 is an ill-conditioned matrix. We employ the same estimators and hyperparameters as in Section 5. The results are presented in Table 1.

Table 1: Results for additive white Gaussian noise channel ( = 0.1), mean and std for 10 runs.

299 300 301 302 303 304 305 306 307 308

| MI      | SMI         | 2-SMI       |             |             |             |             |             |
|---------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| 309     | Σ −1/2      | D−1/2       | Σ −1/2      | D−1/2       | Σ −1/2      | D−1/2       |             |
| 310 311 | 𝑋′         | 7.48 ± 0.01 | 3.04 ± 0.01 | 0.17 ± 0.02 | 1.82 ± 0.04 | 0.96 ± 0.04 | 2.46 ± 0.03 |
| 𝑋″     | 7.49 ± 0.02 | 3.04 ± 0.01 | 0.14 ± 0.02 | 1.83 ± 0.04 | 0.82 ± 0.05 | 2.49 ± 0.05 |             |
| 312     |             |             |             |             |             |             |             |

## 313 6.2 Representation Learning

314 315 316 317 To further demonstrate SMI's sensitivity to information redundancy, we examine its performance in learning compressed representations through mutual information maximization (*Deep InfoMax*) [8]. This approach is known to be equivalent to many popular contrastive self-supervised learning methods [13].

318 319 320 In Deep InfoMax, an encoder network  is trained to maximize a lower bound on (; ()), where represents input data and () its compressed representation. This method is theoretically sound, as maximizing MI ensures the most informative embeddings under the latent space dimensionality 321
(a) MI → max, 2000 epochs. (b) SMI → max, 10 epochs. (c) SMI → max, 2000 epochs.

Figure 5: Visualizations of embeddings from the representation learning experiments, with points colored by class. Note that mutual information maximization (left) produces clustered low-redundancy representations, while SMI maximization results in immediate (after 10 epochs) collapse.

322 323 324 325 326 constraint. For our study, we replace MI with SMI in this framework. This substitution is straightforward since both MI and SMI admit Donsker-Varadhan variational lower bounds [38]:

(;  ) = sup :Ω→ℝ [ℙ,  (,  ) − log(ℙ ⊗ ℙ  (, ))], (;  ) = sup :Ω→ℝ Θ,Φ[ℙ,  (Θ, Φ  , Θ, Φ) − log(ℙ ⊗ ℙ  (Θ,Φ ,Θ,Φ))], (5)
327 328 where  is a critic function, which is also approximated in practice by a neural network. For detailed derivations of these bounds, we refer the reader to [39] (MI) and [22], [23] (SMI).

329 330 331 332 333 334 335 We strictly follow the experimental protocol from [13]. In particular, we use MNIST handwritten digits dataset [40], employ InfoNCE loss [41] to approximate (5), use convolutional network for and fully-connected network for  . Latent space dimensionality is fixed at  = 2 for visualization purposes. Small Gaussian noise is added to the outlet of the encoder to combat representation collapse [13]. More details are provided in Section D. We focus on this simple setup because our objective is to show that SMI produces degenerate results even in elementary tasks, making more complex configurations unnecessary for this demonstration.

336 337 338 Results are presented in Figure 5. As our theory predicts, maximization of SMI immediately leads to collapsed representations, while conventional InfoMax yields embeddings with low or even zero redundancy (components are close to (0,I)). This behavior is consistent across different runs.

## 339 7 Discussion

340 341 342 343 Results. Sliced mutual information (SMI) has been proposed as a scalable alternative to Shannon's mutual information. While SMI enables efficient computation in high-dimensional settings and satisfies the nullification property, our findings reveal critical deficiencies that undermine its reliability for feature extraction and related tasks.

344 345 346 347 348 We demonstrate that SMI saturates rapidly, failing to capture variations in statistical dependence. This makes it difficult to distinguish between intrinsic SMI fluctuations and genuine changes in dependence structure. Furthermore, we invalidate the common hypothesis that SMI favors linear features through a counterexample where even correlation coefficients reflect dependence more faithfully than SMI, which exhibits inverted behavior.

349 350 351 352 In high-dimensional spaces, SMI decays with increasing dimensionality, contrary to MI's monotonic behavior. This is established analytically for Gaussian cases and validated empirically across diverse synthetic experiments. Consequently, SMI variations may reflect redundancy, dependence changes, or high-dimensional artifacts without a principled way to disentangle these factors.

353 354 355 356 Impact. Thanks to fast convergence rates and the absence of additional optimization problems, SMI has been widely applied across various fields of statistics and machine learning. Given our findings, it is therefore crucial to recognize how the inherent biases of SMI affect practical applications.

357 358 359 360 361 362 363 364 365 The works [22] and [23] propose using SMI in a Deep InfoMax setting. However, we demonstrate that maximizing SMI can lead to collapsed solutions due to redundancy bias. Meanwhile, [26], [27], [28], [30] study deep neural networks by measuring SMI between intermediate layers. Yet, as our analysis reveals, changes in SMI do not always reflect true shifts in statistical dependence; they may instead result from differences in layer dimensionality, redundancy in intermediate representations, low sensitivity in saturated regimes, or other factors. Finally, [33] suggests using SMI for independence testing in differential privacy tasks. We contend that this approach poses critical issues, as SMI estimates can become statistically indistinguishable from zero in high-dimensional or lowredundancy settings.

366 367 368 369 370 Limitations. While we support our claims with both theoretical analysis and experimental evidence, we were able to derive analytical expressions for the Gaussian case only. Furthermore, our synthetic tests do not feature complex, highly non-linear distributions (such as structured image data used in [17]). Nevertheless, we demonstrate that our findings are more than sufficient to expose fundamental limitations of SMI, and to support all the claims we made.

## References 371

372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422
[1] Y. Polyanskiy and Y. Wu, *Information Theory: From Coding to Learning*. Cambridge University Press, 2024. [Online]. Available: https://books.google.ru/books?id=CySo0AEACAAJ
[2] A. Asadi, E. Abbe, and S. Verdu, "Chaining Mutual Information and Tightening Generalization Bounds," in *Advances in Neural Information Processing Systems*, S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, Eds., Curran Associates, Inc., 2018, p. . [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2018/file/8d7628dd7a710c8638dbd22d4421 ee46-Paper.pdf
[3] J. Negrea, M. Haghifam, G. K. Dziugaite, A. Khisti, and D. M. Roy, "Information-Theoretic Generalization Bounds for SGLD via Data-Dependent Estimates," in *Advances in Neural Information Processing* Systems, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, Eds., Curran Associates, Inc., 2019, p. . [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2019/ file/05ae14d7ae387b93370d142d82220f1b-Paper.pdf
[4] B. Duong and T. Nguyen, "Conditional Independence Testing via Latent Representation Learning," in 2022 IEEE International Conference on Data Mining (ICDM), Los Alamitos, CA, USA: IEEE Computer Society, Dec. 2022, pp. 121–130. doi: 10.1109/ICDM54844.2022.00022.

[5] S. Yang and J. Gu, "Feature selection based on mutual information and redundancy-synergy coefficient,"
J. Zhejiang Univ. Sci., vol. 5, no. 11, pp. 1382–1391, Nov. 2004.

[6] N. Kwak and C.-H. Choi, "Input feature selection by mutual information based on Parzen window," *IEEE*
Transactions on Pattern Analysis and Machine Intelligence, vol. 24, no. 12, pp. 1667–1671, 2002, doi: 10.1109/TPAMI.2002.1114861.

[7] M. A. Sulaiman and J. Labadin, "Feature selection based on mutual information," in 2015 9th International Conference on IT in Asia (CITA), 2015, pp. 1–6. doi: 10.1109/CITA.2015.7349827.

[8] R. D. Hjelm *et al.*, "Learning deep representations by mutual information estimation and maximization,"
in *International Conference on Learning Representations*, 2019. [Online]. Available: https://openreview. net/forum?id=Bklr3j0cKX
[9] P. Bachman, R. D. Hjelm, and W. Buchwalter, "Learning Representations by Maximizing Mutual Information Across Views," in *Advances in Neural Information Processing Systems*, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, Eds., Curran Associates, Inc., 2019, p. . [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2019/file/ddf354219aac374f1d40b 7e760ee5bb7-Paper.pdf
[10] P. Veličković, W. Fedus, W. L. Hamilton, P. Liò, Y. Bengio, and R. D. Hjelm, "Deep Graph Infomax," in International Conference on Learning Representations, 2019. [Online]. Available: https://openreview. net/forum?id=rklz9iAcKQ
[11] M. Tschannen, J. Djolonga, P. K. Rubenstein, S. Gelly, and M. Lucic, "On Mutual Information Maximization for Representation Learning," in *International Conference on Learning Representations*, 2020. [Online]. Available: https://openreview.net/forum?id=rkxoh24FPH
[12] X. Yu, "Leveraging Superfluous Information in Contrastive Representation Learning." [Online]. Available: https://arxiv.org/abs/2408.10292
[13] I. Butakov, A. Semenenko, A. Tolmachev, A. Gladkov, M. Munkhoeva, and A. Frolov, "Efficient Distribution Matching of Representations via Noise-Injected Deep InfoMax," in *The Thirteenth International* Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id= mAmCdASmJ5
[14] N. Tishby and N. Zaslavsky, "Deep learning and the information bottleneck principle," 2015 IEEE Information Theory Workshop (ITW), pp. 1–5, 2015.

[15] R. Shwartz-Ziv and N. Tishby, "Opening the Black Box of Deep Neural Networks via Information." 2017. [16] Z. Goldfeld *et al.*, "Estimating Information Flow in Deep Neural Networks," in *Proceedings of the 36th* International Conference on Machine Learning, K. Chaudhuri and R. Salakhutdinov, Eds., in Proceedings of Machine Learning Research, vol. 97. PMLR, 2019, pp. 2299–2308. [Online]. Available: https:// proceedings.mlr.press/v97/goldfeld19a.html
[17] I. Butakov, A. Tolmachev, S. Malanchuk, A. Neopryatnaya, A. Frolov, and K. Andreev, "Information Bottleneck Analysis of Deep Neural Networks via Lossy Compression," in *The Twelfth International* 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id= huGECz8dPp
[18] Z. Goldfeld, K. Greenewald, J. Niles-Weed, and Y. Polyanskiy, "Convergence of Smoothed Empirical Measures With Applications to Entropy Estimation," *IEEE Transactions on Information Theory*, vol. 66, no. 7, pp. 4368–4391, 2020, doi: 10.1109/TIT.2020.2975480.

[19] D. McAllester and K. Stratos, "Formal Limitations on the Measurement of Mutual Information," in Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, S. Chiappa and R. Calandra, Eds., in Proceedings of Machine Learning Research, vol. 108. PMLR, 2020, pp. 875–884. [Online]. Available: https://proceedings.mlr.press/v108/mcallester20a.html
[20] G. Gowri, X. Lun, A. M. Klein, and P. Yin, "Approximating mutual information of high-dimensional variables using learned representations," in *The Thirty-eighth Annual Conference on Neural Information* Processing Systems, 2024. [Online]. Available: https://openreview.net/forum?id=HN05DQxyLl
[21] K. H. Greenewald, B. Kingsbury, and Y. Yu, "High-Dimensional Smoothed Entropy Estimation via Dimensionality Reduction," in IEEE International Symposium on Information Theory, ISIT 2023, Taipei, Taiwan, June 25-30, 2023, IEEE, 2023, pp. 2613–2618. doi: 10.1109/ISIT54713.2023.10206641.

[22] Z. Goldfeld and K. Greenewald, "Sliced Mutual Information: A Scalable Measure of Statistical Dependence," in *Advances in Neural Information Processing Systems*, A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, Eds., 2021. [Online]. Available: https://openreview.net/forum?id=27qon5Ut4PSl
[23] Z. Goldfeld, K. Greenewald, T. Nuradha, and G. Reeves, "$k$-Sliced Mutual Information: A Quantitative Study of Scalability with Dimension," in *Advances in Neural Information Processing Systems*, A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, Eds., 2022. [Online]. Available: https://openreview.net/forum?id= L-ceBdl2DPb
[24] D. Tsur, Z. Goldfeld, and K. Greenewald, "Max-Sliced Mutual Information," in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview.net/forum?id= ce9B2x3zQa
[25] A. Fayad and M. Ibrahim, "On Slicing Optimality for Mutual Information," in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview.net/forum?id= JMuKfZx2xU
[26] S. Wongso, R. Ghosh, and M. Motani, "Understanding Deep Neural Networks Using Sliced Mutual Information," in *2022 IEEE International Symposium on Information Theory (ISIT)*, 2022, pp. 133–138. doi: 10.1109/ISIT50566.2022.9834357.

[27] S. Wongso, R. Ghosh, and M. Motani, "Using Sliced Mutual Information to Study Memorization and Generalization in Deep Neural Networks," in Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, F. Ruiz, J. Dy, and J.-W. van de Meent, Eds., in Proceedings of Machine Learning Research, vol. 206. PMLR, 2023, pp. 11608–11629. [Online]. Available: https://proceedings. mlr.press/v206/wongso23a.html
[28] S. Wongso, R. Ghosh, and M. Motani, "Pointwise Sliced Mutual Information for Neural Network Explainability," in *2023 IEEE International Symposium on Information Theory (ISIT)*, 2023, pp. 1776– 1781. doi: 10.1109/ISIT54713.2023.10207010.

[29] J. Dentan, D. Buscaldi, A. Shabou, and S. Vanier, "Predicting and analyzing memorization within finetuned Large Language Models." [Online]. Available: https://arxiv.org/abs/2409.18858
[30] S. Wongso, R. Ghosh, and M. Motani, "Sliced Information Plane for Analysis of Deep Neural Networks,"
Jan. 2025, doi: 10.36227/techrxiv.173833980.08812687/v1.

[31] K. Nadjahi, K. Greenewald, R. B. Gabrielsson, and J. Solomon, "Slicing Mutual Information Generalization Bounds for Neural Networks," in ICML 2023 Workshop Neural Compression: From Information Theory to Applications, 2023. [Online]. Available: https://openreview.net/forum?id=cbLcwK3SZi
[32] Z. Hu, S. Kang, Q. Zeng, K. Huang, and Y. Yang, "InfoNet: Neural Estimation of Mutual Information without Test-Time Optimization," in *Forty-first International Conference on Machine Learning*, 2024. [Online]. Available: https://openreview.net/forum?id=40hCy8n5XH
[33] T. Nuradha and Z. Goldfeld, "Pufferfish Privacy: An Information-Theoretic Study," IEEE Trans. Inf.

Theor., vol. 69, no. 11, pp. 7336–7356, Nov. 2023, doi: 10.1109/TIT.2023.3296288.

[34] I. Butakov *et al.*, "MUTINFO." [Online]. Available: https://github.com/VanessB/mutinfo 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502
[35] A. Kraskov, H. Stögbauer, and P. Grassberger, "Estimating mutual information," *Phys. Rev. E*, vol. 69, no.

6, p. 66138, Jun. 2004, doi: 10.1103/PhysRevE.69.066138.

[36] F. Czyż Pawełand Grabowski, J. Vogt, N. Beerenwinkel, and A. Marx, "Beyond Normal: On the Evaluation of Mutual Information Estimators," in *Advances in Neural Information Processing Systems*, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., Curran Associates, Inc., 2023, pp. 16957–16990. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2023/file/36b80eae 70ff629d667f210e13497edf-Paper-Conference.pdf
[37] T. M. Cover and J. A. Thomas, Elements of Information Theory (Wiley Series in Telecommunications and Signal Processing). USA: Wiley-Interscience, 2006.

[38] M. D. Donsker and S. R. Varadhan, "Asymptotic evaluation of certain markov process expectations for large time. IV," *Communications on Pure and Applied Mathematics*, vol. 36, no. 2, pp. 183–212, Mar. 1983, doi: 10.1002/cpa.3160360204.

[39] M. I. Belghazi *et al.*, "Mutual Information Neural Estimation," in Proceedings of the 35th International Conference on Machine Learning, J. Dy and A. Krause, Eds., in Proceedings of Machine Learning Research, vol. 80. PMLR, 2018, pp. 531–540. [Online]. Available: https://proceedings.mlr.press/v80/ belghazi18a.html
[40] L. Deng, "The mnist database of handwritten digit images for machine learning research," *IEEE Signal* Processing Magazine, vol. 29, no. 6, pp. 141–142, 2012.

[41] A. van den Oord, Y. Li, and O. Vinyals, "Representation Learning with Contrastive Predictive Coding." [Online]. Available: https://arxiv.org/abs/1807.03748
[42] A. Edelman and B. D. Sutton, "The beta-Jacobi matrix model, the CS decomposition, and generalized singular value problems," *Foundations of Computational Mathematics*, vol. 8, no. 2, pp. 259–285, 2008.

[43] A. McBride, "Special functions, by George E. Andrews, Richard Askey and Ranjan Roy. Pp. 664.£ 60.

1999. ISBN 0 521 62321 9 (Cambridge University Press.)," *The Mathematical Gazette*, vol. 83, no. 497, pp. 355–357, 1999.

[44] N. Elezovic, C. Giordano, and J. Pecaric, "The best bounds in Gautschi's inequality," *Math. Inequal. Appl*,
vol. 3, no. 2, pp. 239–252, 2000.

[45] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization." 2017.

## 503 A Supplementary Theory

504 Lemma A.1. (Example 2.4 in [1]) ((, Σ)) =
1 2 log((2)
 det Σ).

505 Corollary A.2. For (,  ) ∼ (, Σ) with non-singular Σ

$$\begin{split}\operatorname{I}(X;Y)&={\frac{1}{2}}\log\operatorname*{det}\Sigma_{X}+{\frac{1}{2}}\log\operatorname*{det}\Sigma_{Y}-{\frac{1}{2}}\log\operatorname*{det}\Sigma_{Y}\\ &=-{\frac{1}{2}}\sum_{i=1}^{d}\log(1-\rho_{i}^{2}),\end{split}$$

506 where Σ, Σ are marginal covariances, Σ is cross-covariance,  = min(, ), and {} =1 are singular values of Σ
−
1 2 Σ Σ
−
1 2

.

Proof of Corollary A.2. Combining Lemma A.1 and (2) yields the first result. Now note that 507 508

$$\mathrm{I}(X;Y)=\mathrm{I}\biggl(\Sigma_{X}^{-\frac{1}{2}}X;\Sigma_{Y}^{-\frac{1}{2}}Y\biggr)=\mathrm{I}\biggl(\mathrm{U}^{\mathrm{T}}\Sigma_{X}^{-\frac{1}{2}}X;\mathrm{V}\Sigma_{Y}^{-\frac{1}{2}}Y\biggr),$$

509

where U diag$(\rho_{i})$V${}^{\sf T}$ is the SVD of $\Sigma_{X}^{-\frac{1}{2}}\Sigma_{XY}\Sigma_{Y}^{-\frac{1}{2}}$. However,
$$\left(\mathbf{U}^{\mathsf{T}}\Sigma_{X}^{-{\frac{1}{2}}}X,\mathbf{V}\Sigma_{Y}^{-{\frac{1}{2}}}Y\right)\sim{\mathcal{N}}\left(\mu^{\prime},\left(\mathbf{\Sigma}_{\mathrm{diag}(\rho_{i})}^{\mathrm{I}}\mathbf{\Sigma}_{\mathrm{I}}^{\mathrm{diag}(\rho_{i})}\right)\right),$$
510 from which we arrive at the second expression. □
Lemma A.3. Let A ∈ ℝ
× be full column-rank matrix and Θ ∼ St(,)
 Then ΘA is full-rank with probability one.

511 512 513 514 515 Proof of Lemma A.3. Performing QR decomposition of A yields ΘA = ΘQR =
d Θ(
I
0
)R. Since A is full-rank, R is invertible and rank ΘA = rank Θ(
I

0
). Therefore,

$$\mathbb{P}\{\Theta^{\mathsf{T}}\mathrm{A\;is\;full-rank}\}=1-\mathbb{P}\bigg\{\Theta^{\mathsf{T}}\bigg(\begin{matrix}\mathrm{I_{m}}\\ 0\end{matrix}\bigg)\;\mathrm{is\;not\;full-rank}\bigg\}=1-0=1.$$

516

$$\lceil\!\!\!\perp\!\!\!\perp$$

517 Lemma A.4. (Theorem 1.5 in [42]) Let W ∼ O()
 and partition

$$\mathbf{W}=\begin{pmatrix}\mathbf{W}_{11}&\mathbf{W}_{12}\\ \mathbf{W}_{21}&\mathbf{W}_{22}\end{pmatrix}.$$

518

with W11
 of size  by . Then the eigenvalues {}

=1
 of W11W
11
 follow the Jacobi ensemble
$$p(\lambda)\propto\prod_{i<j}|\lambda_{i}-\lambda_{j}|^{\beta}\prod_{i=1}^{k}\lambda_{i}^{\frac{\beta}{2}(a+1)-1}(1-\lambda_{i})^{\frac{\beta}{2}(b+1)-1}$$

519 with parameters  = 0,  =  − 2, and  = 1 (over ℝ).

520 521 522 523 524 525 Proof of Lemma A.3. Let A1 ∈ ℝ
× and A2 ∈ ℝ
(−)× be independent matrices with i.i.d. entries from (0, 1). By stacking A1 atop A2 and then performing a block QR decomposition on the resulting Gaussian matrix, the orthogonal invariance of the Gaussian law implies that the two Q‑blocks are independent of the upper‑triangular factor R, with Q1 and Q2 uniformly distributed on O() and St(,  − ), respectively. Finally, computing the SVD of the block rows together with R
yields the generalized singular value decomposition (GSVD) of the pair (A1
, A2
):

$${\binom{\mathrm{A}_{1}}{\mathrm{A}_{2}}}={\binom{\mathrm{Q}_{1}}{\mathrm{Q}_{2}}}\mathrm{R}={\binom{\mathrm{U}_{1}}{\mathrm{U}_{2}}}{\left({\frac{{\tilde{\mathrm{C}}}}{-{\tilde{\mathrm{S}}}}}\right)}{\tilde{\mathrm{V}}}^{\mathrm{T}}\mathrm{R},$$

526 527 528 where U1 ∈ O(), U2 ∈ O( − ), Ṽ ∈ O(), and C = diag(), S = diag() with  ≥ 0,  ≥
0, and 2
 + 
2
 = 1 for all . The diagonal entries of C̃ are known as the generalized singular values of the pair (A1
, A2
).

529 530 531 For a matrix P = diag(1, …, ) with i.i.d.  sampled uniformly from {−1, 1}, we have Q1 =
d W11. Let W11 = UCV be the SVD of W11, then one has

$$\mathrm{U}_{1}\left(\begin{array}{l}{{\tilde{C}}}\\ {{0}}\end{array}\right)\tilde{\mathrm{V}}^{\mathsf{T}}\mathrm{P}\triangleq\mathrm{UCV}^{\mathsf{T}}.$$

532 533 534 535 536 537 538 539 540 541 542 Since U1, Ṽ , and U, V are uniformly distributed and independent of C̃, C, we have C̃ =
dC by the invariance of the Haar measure under orthogonal transformations. On the other hand, the generalized singular values C̃ of a pair (A1, A2) follow the law of the Jacobi ensemble with parameters  =
0,  =  − 2, and  = 1 (Proposition 1.2 in [42]). Therefore, the squared singular values of W11 follow the Jacobi ensemble with the same parameters. □ Corollary A.5. The squared inner product | | 2 between two independent random vectors ,  ∼

−1 follows Beta( 1 2
,
−1 2
). Moreover, the shifted inner product (1 + 
)/2 is symmetrically distributed as Beta(
−1 2
,
−1 2
).

Proof of Corollary A.5. Setting Jacobi parameters  = 1,  = 0,  =  − 2 and  = 1, the density is proportional to 
−1/2(1 − )
(−3)/2 on [0, 1], which matches the Beta( 1 2
,
−1 2
) distribution.

543 544 Next, observe that has a density proportional to (1 − )
−3 2 for  ∈ [−1, 1]. Under the change of variables  ∼ Beta(
−1 2
,
−1 2
).

545
□

## 546 B Complete Proofs

547 548 Proof of Lemma 4.1. One can acquire (;  ) = −

2 log(1 − 
2) from a general expression for MI
of two jointly Gaussian random vectors (see Corollary A.2).

549 550 Recall that (
, 
 ) is also Gaussian with cross-covariance  
. Therefore, by Corollary A.2 we have

$${\mathsf{S l}}(X;Y)=\mathbb{E}[\left|(\theta^{\mathsf{T}}X;\phi^{\mathsf{T}}Y)\,\right|\theta,\varphi]=-\frac{1}{2}\,\mathbb{E}[\log(1-\rho^{2}\,\,|\theta^{\mathsf{T}}\phi|^{2})].$$

551 From Corollary A.5, we note that | |

$|^2\sim\text{Beta}\big(\frac{1}{2},\frac{d-1}{2}\big)$, so . 
$$\mathsf{Sl}(X;Y)=-\frac{1}{2\mathsf{B}\big{(}\frac{1}{2},\frac{d-1}{2}\big{)}}\int_{0}^{1}\log(1-\rho^{2}x)(1-x)^{\frac{d-3}{2}}x^{-\frac{1}{2}}\,\mathrm{d}x$$ $$=\frac{\rho^{2}}{2}\,\frac{\Gamma\big{(}\frac{d}{2}\big{)}}{\Gamma\big{(}\frac{1}{2}\big{)}\Gamma\big{(}\frac{d-1}{2}\big{)}}\int_{0}^{1}x^{\frac{1}{2}}(1-x)^{\frac{d-3}{2}}\,{}_{2}F_{1}(1,1;2;\rho^{2}x)\,\mathrm{d}x,$$
$$(\mathbf{6})$$
552 553 where the last equality follows from the identity log(1 − ) = − 21
(1, 1; 2; ) with hypergeometric function 21
. Appling Euler's integral transform ([43], Eq. (2.2.3)) gives

$$\mathsf{S}|(X;Y)=\frac{\rho^{2}}{2d}\frac{\Gamma(\frac{d}{2}+1)}{\Gamma(\frac{3}{2})\Gamma(\frac{d-2}{2})}\int_{0}^{1}x^{\frac{3}{2}-1}(1-x)^{(\frac{d}{2}+1)-\frac{3}{2}-1}\,_{2}F_{1}\big{(}1,1;2;\rho^{2}x\big{)}\,\mathrm{d}x$$ $$=\frac{\rho^{2}}{2d}\,_{3}F_{2}\bigg{(}1,1,\frac{3}{2};\frac{d}{2}+1,2;\rho^{2}\bigg{)}.$$

554 Here 32 denotes the generalized hypergeometric function.

555 556 Finally, we calculate the limit of (;  ) as 2 → 1 using properties of beta-distribution. Denoting
 = (1 + 
)/2 ∼ Beta(
−1 2
,
−1 2
) (see Corollary A.5), we get

$$\mathsf{S l}(X;Y)=-\log2-\mathbb{E}\log(1-\eta)=-\log2-\mathbb{E}\log\eta=\psi(d-1)-\psi\biggl({\frac{d-1}{2}}\biggr)-\log2,$$

557 where  is the digamma function. Using the bounds on digamma function [44]

$$\log\biggl(x+{\frac{1}{2}}\biggr)-{\frac{1}{x}}\leq\psi(x)\leq\log\bigl(x+e^{\psi(1)}\bigr)-{\frac{1}{x}},$$

558 we derive an upper bound on this expression:

$$\psi(d-1)-\psi\biggl({\frac{d-1}{2}}\biggr)-\log2\leq{\frac{1}{d-1}}+\log\biggl(1+{\frac{1+e^{\psi(1)}}{d}}\biggr)$$

To simplify the bound, one can note that 1 + 
(0) < 2, log(1 + ) <  and 1 <1
−1
.

559 560

$\square$
Proof of Proposition 4.2.

561 562 563 Let QX, QY ∼ St(,)
. Then [QX, QY ] ∼ (0, Σ), where Σ is a 2 × 2 covariance matrix with the following block structure

$$\Sigma=\left(\begin{array}{c c}{{\mathrm{I}_{k}}}&{{\rho\,\mathrm{Q}_{X}^{\mathsf{T}}\mathrm{Q}_{Y}}}\\ {{\rho\,\mathrm{Q}_{Y}^{\mathsf{T}}\mathrm{Q}_{X}}}&{{\mathrm{I}_{k}}}\end{array}\right).$$

564 Using the formula for the determinant of a block matrix Σ yields

$$\mathsf{S l}_{k}(X;Y)=-\frac{1}{2}\,\mathbb{E}[\log\operatorname*{det}(\Sigma)]=-\frac{1}{2}\,\mathbb{E}\Big[\log\operatorname*{det}\Bigl(\mathsf{I}-\rho^{2}\bigl(\mathsf{Q_{X}^{T}Q_{Y}}\bigr)\bigl(\mathsf{Q_{X}^{T}Q_{Y}}\bigr)^{\mathsf{T}}\Bigr)\Big].$$

565 566 567 By the invariance of the Haar measure under left and right multiplication, QXQY =
 W11, where W11 is a  by  left upper block of the matrix W ∼ O()
. According to Lemma A.4, the eigenvalues of W11W
11 follow Jacobi ensemble with parameters  = 0,  =  − 2 and  = 1:

$$p(\lambda)\propto\prod_{i<j}|\lambda_{j}-\lambda_{i}|\prod_{i=1}^{k}\left(1-\lambda_{i}\right)^{\frac{d-2k-1}{2}}.$$
<
=1
568 Thus, we get a general expresion for -SMI

$$\mathrm{{\sf~S l}}_{k}(X;Y)=-\frac{1}{2}\int_{[0,1]^{k}}\sum_{i=1}^{k}\log(1-\rho^{2}\lambda_{i})p(\lambda)\,\mathrm{d}\lambda.$$

569
□
570 571 572 573 Proof of Proposition 4.4. Using Lemma A.3 and 
,  < , we get that ΘA and Φ
B are injective with probability one for independent Θ, Φ distributed uniformly on St(
, ) and St(
, ).

Therefore, according to Theorem 3.1, [(ΘA; Φ
B ) | Θ, Φ] = (;  ) almost sure. As a result,
(A; B ) = (ΘA; Φ
B | Θ, Φ) = (;  ). □
Proof of Proposition 4.7. Direct corollary of Corollary A.2. □
574

## 575 C Additional Experiments

576 577 578 579 In this section, we conduct supplementary experiments to evaluate SMI under a broader range of setups. We begin by assessing -SMI on the same set of benchmarks from Section 5. The results for  = 1, 2, 3 are presented in Figure 3, Figure 6, and Figure 7, respectively. Notably, saturation remains consistent even for  =  − 1 (i.e., when only one component is discarded).

580 581 582 Next, we examine a setup involving randomized distribution parameters, following the methodology of [34]. Among other adjustments, this includes randomizing per-component mutual information (e.g., assigning interactions unevenly in this experiment). In some cases (e.g., the log-gamma-

583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 605 606 0 2 4 6 8 10
(;  )/, nats 0 1 2 3 4 5 0 2 4 6 8 10
(;  )/, nats 0 1 2 3 4 5 2

(

; 
 
), n ats
 = 3 = 4 = 6 = 8 = 16 2

(

; 
 
), n ats
 = 3 = 4 = 6 = 8 = 16
(a) Correlated Normal
(b) Correlated Uniform 0 2 4 6 8 10
(;  )/, nats 0 1 2 3 4 5 0 2 4 6 8 10
(;  )/, nats 0 1 2 3 4 5 2

(

; 
 
), 
nat s

 = 3 = 4 = 6 = 8 = 16 2

(

; 
 
), 
nat s

 = 3 = 4 = 6 = 8 = 16
(c) Smoothed Uniform
610 611 612 613 614 615 616 617 exponential distribution), this increases linear redundancy, as component pairs with higher mutual information also exhibit higher variance in this particular scenario. Our results are displayed in Figure 8. Due to numerical constraints, we do not track (;  )/, instead plotting the results against the total mutual information. While this makes saturation slightly less evident, the general trend of SMI decreasing with  remains observable. We also highlight the log-gamma-exponential distribution (Figure 8d), where SMI is less prone to saturation under parameter randomization due to the reasons mentioned earlier.

## 618 D Implementation Details 619 D.1 Synthetic Experiments

620 621 622 623 624 For the experiments from Section 5 and Section 6.1, we use implementation of Kraskov-Stoegbauer- Grassberger (KSG) [35] mutual information estimator and random slicing from [34]. The number of neighbors is set to NN = 1 for the KSG estimator. For each configuration, we conduct 10 independent runs with different random seeds to compute means and standard deviations. Our experiments use 10 4 samples for (,  ) and 128 samples for (Θ, Φ).

625 626 627 628 629 For the experiments from Section 5, we use independent components with equally distributed percomponent MI. For the supplementary experiments from Figure 8, parameters of each distribution (e.g., covariance matrices) are randomized via the algorithm implemented in [34]. This includes randomization of per-component MI (which is done using a uniform distribution over a ( − 1)- dimensional simplex).

0 2 4 6 8 10
(;  )/, nats 0 2 4 6 0 2 4 6 8 10
(;  )/, nats 0 2 4 6 630 631 632 633 634 635 636 637 3

(
; 
 
), n ats
 = 4 = 6 = 8 = 16 = 32 3

(
; 
 
), n ats
 = 4 = 6 = 8 = 16 = 32 638 639 640
(a) Correlated Normal
(b) Correlated Uniform 641 0 2 4 6 8 10
(;  )/, nats 0 2 4 6 0 2 4 6 8 10
(;  )/, nats 0 2 4 6 642 643 3 (
; 
 )
, nat s

 = 4 = 6 = 8 = 16 = 32 3 (
; 
 )
, nat s

 = 4 = 6 = 8 = 16 = 32 644 645 646 647 648 649 650 651 652
(c) Smoothed Uniform 653
654 655 656 657 658 For the experiments, we used AMD EPYC 7543 CPU, one core per distribution. Each experiment (fixed , varying ) took no longer then 3 days to compute.

## 659

660 661 662 663 664

## D.2 Representation Learning Experiments

For experiments on MNIST dataset, we use a simple ConvNet with three convolutional and two fully connected layers. A three-layer fully-connected perceptron serves as a critic network for the InfoNCE loss. We provide the details in Table 2. We use additive Gaussian noise with  = 0.2 as an input augmentation. Training hyperparameters are as follows: batch size = 512, 2000 epochs, Adam optimizer [45] with learning rate 10
−3.

665 666 667 For the experiments, we used Nvidia A100 GPUs. Each experiment took no longer then 1 day to compute.

Table 2: The NN architectures used to conduct the tests on MNIST images in Section 6.2.

| NN                                                                                                                                                                                                  | Architecture                                         |                                        |                                                                 |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|----------------------------------------|-----------------------------------------------------------------|
| 668 669 670 671 672                                                                                                                                                                                 | ConvNet,                                             | × 1:                                   | Conv2d(1, 32, ks=3), MaxPool2d(2), BatchNorm2d, LeakyReLU(0.01) |
| × 1: Conv2d(32, 64, ks=3), MaxPool2d(2), BatchNorm2d, LeakyReLU(0.01) × 1: Conv2d(64, 128, ks=3), MaxPool2d(2), BatchNorm2d, LeakyReLU(0.01) × 1: Dense(128, 128), LeakyReLU(0.01), Dense(128, dim) |                                                      |                                        |                                                                 |
| 24 × 24 images Critic NN,                                                                                                                                                                           | × 1:                                                 | Dense(dim + dim, 256), LeakyReLU(0.01) |                                                                 |
| pairs of vectors                                                                                                                                                                                    | × 1: Dense(256, 256), LeakyReLU(0.01), Dense(256, 1) |                                        |                                                                 |
| 673 674                                                                                                                                                                                             |                                                      |                                        |                                                                 |

0 2 4 6 8 10
(;  ), nats 0 0.2 0.4 0.6 0.8 0 2 4 6 8 10
(;  ), nats 0 0.2 0.4 0.6 0.8 675 676 677 678 679 680 681
 = 2 = 3 = 4 = 6 = 8
 = 2 = 3 = 4 = 6 = 8

(

; 
 
), n ats

(

; 
 
), n ats 682 683 684 685
(a) Correlated Normal
(b) Correlated Uniform 686 0 2 4 6 8 10
(;  ), nats 0 0.2 0.4 0.6 0.8 0 2 4 6 8 10
(;  ), nats 0 2 4 6 687 688
 = 2 = 3 = 4 = 6 = 8
 = 2 = 3 = 4 = 6 = 8 689 690 691 692 693 694 695

(
; 
 
), nat s

(
; 
 
), nat s 696 697 698

## Neurips Paper Checklist 703

704

## 1. **Claims**

705 706 Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

707 Answer: [YES]
708 709 Justification: We state our claims clearly in the abstract and introduction. The claims are supported by theoretical analysis and various experiments.

710 Guidelines:
711 712 713 714 715 716 717 718 719
- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

720

## 2. **Limitations**

721 Question: Does the paper discuss the limitations of the work performed by the authors?

722 Answer: [YES]
723 Justification: We discuss limitations in Section 7.

724 Guidelines:
725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749
- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper. - The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

750

## 3. **Theory Assumptions And Proofs**

751 752 Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

753 Answer: [YES]
754 755 Justification: We provide comprehensive statements for theorems and lemmas. We also provide complete proofs in Section B.

756 Guidelines:
757
- The answer NA means that the paper does not include theoretical results.

758 759 760 761 762 763 764 765 766
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

- All assumptions should be clearly stated or referenced in the statement of any theorems. - The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

- Theorems and Lemmas that the proof relies upon should be properly referenced.

767

## 4. **Experimental Result Reproducibility**

768 769 770 Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

771 Answer: [YES]
772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 Justification: We provide complete setup descriptions for the experiments in corresponding sections.

- The answer NA means that the paper does not include experiments. - If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

Guidelines: