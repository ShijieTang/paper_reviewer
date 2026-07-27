# Multivariate Conformal Prediction Using Optimal Transport

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054

## Anonymous Authors1 Abstract

Conformal prediction quantifies the uncertainty of machine learning models by constructing sets of plausible outputs instead of relying on a single prediction, which may not exactly match the ground-truth. This is achieved by evaluating all possible output candidates and selecting the most likely ones by ranking their score functions, which measure how well each candidate aligns with the given input, the prediction model, and past observations. Traditionally, this approach has been limited to univariate score functions, as ranking requires a scalar value to order candidates. The challenge lies in extending ranking to multivariate spaces, where no canonical order exists. To address this, we leverage a natural extension of multivariate score ranking based on optimal transport mappings. Our method offers a principled framework for constructing conformal prediction sets in multidimensional settings, preserving distributionfree coverage guarantees with finite data samples.

## 1. Introduction

Conformal prediction (CP) (Gammerman et al., 1998; Vovk et al., 2005; Shafer & Vovk, 2008) has emerged as a simple framework to quantify the prediction uncertainty of machine learning algorithm without relying on distributional assumptions on the data. For a sequence of observed data, and a new input point, Dn = {(x1, y1), ...,(xn, yn)} and xn+1, the objective is to construct a set that contains the unobserved response yn+1 with a specified confidence level 100(1 − α)%. This involves evaluating scores S(*x, y,* yˆ) ∈
R
 such as the prediction error of a model yˆ, for each observation (*x, y*) in Dn and ranking these score values. The conformal prediction set for the new input xn+1 is the collection 1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

1 of all possible responses y whose score S(xn+1*, y,* yˆ) ranks small enough to meet the prescribed confidence threshold, compared to the scores S(xi, yi, yˆ) in the observed data. CP has undergone tremendous developments in recent years,both (Barber et al., 2023; Park et al., 2024; Tibshirani et al., 2019; Guha et al., 2024), that mirror is increased applicability to challenging settings(Straitouri et al., 2023; Lu et al., 2022). To name a few, it has been applied for designing uncertainty sets in active learning (Ho & Wechsler, 2008), anomaly detection (Laxhammar & Falkman, 2015; Bates et al., 2021), few-shot learning (Fisch et al., 2021), time series (Chernozhukov et al., 2018; Xu & Xie, 2021; Chernozhukov et al., 2021; Lin et al., 2022; Zaffran et al., 2022), or to infer the performance guarantee for statistical learning algorithms (Holland, 2020; Cella & Ryan, 2020); and recently to Large Language Models (Kumar et al., 2023; Quach et al., 2023). We refer to the extensive reviews in (Balasubramanian et al., 2014) for other applications to machine learning. By design, CP requires the notion of order, as the inclusion of a candidate response depends on its relative ranking to the scores observed previously. Hence, the classical strategies developed so far largely targets score functions with univariate outputs. This limits their applicability to multivariate responses, as ranking vector-valued scores S(*x, y,* yˆ) ∈ R
d, d ≥ 2 is evidently not as straightforward as ranking numbers. Ordering Vector Distributions using Optimal Transport. In parallel to these developments, and starting with the seminal reference of (Chernozhukov et al., 2017) and more generally the pioneering works of (Hallin et al., 2021; 2022; 2023), multiple references have explored the possibilities offered by optimal transport theory to define a meaningful ranking or ordering in a multidimensional space. Simply put, the analogous of a rank function computed on data can be found in the optimal Brenier map that transports the data measure to a uniform, symmetric, centered measure of reference in R
d. As a result, a simple notion of univariate rank for a vector z ∈ R
dcan be found by evaluating the distance of the image of z according to that optimal map to the origin. This approach ensures that the ordering respects both the geometry i.e spatial arrangement of the data and its distribution: points closer to the center get lower ranks.

Contributions We propose to leverage recent advances in computational optimal transport (Peyre & Cuturi ´ , 2019), using notably differentiable transport map estimators (Pooladian & Niles-Weed, 2021; Cuturi et al., 2019), to leverage the application of such maps in the definition of multivariate score functions. More precisely:
055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109
- **OT-CP**: We extend conformal prediction techniques to multivariate score function by leveraging optimal transport ordering, which offers a principled way to define and compute a higher-dimensional quantile and cumulative distribution function. As a result, we obtain distribution-free uncertainty sets that capture the joint behavior of multivariate predictions that enhance the flexibility and scope of conformal predictions.

- We propose a computational approach to this theoretical ansatz using the entropic map (Pooladian & Niles- Weed, 2021) computed from solutions to the Sinkhorn problem (Cuturi, 2013). We prove that our approach preserves the coverage guarantee while being tractable.

- We showcase the application of **OT-CP** using a recently released benchmark of regression tasks (Dheur et al., 2025).

## 2. Background

Notation We define [n] = {1*, . . . , n*}. We denote the standard uniform measure on [a, b] as U([*a, b*]). For a discrete set of points (zi)i∈[n], the empirical uniform measure is denoted Un =
1 n Pn i=1 δzi.

## 2.1. Univariate Conformal Prediction

For a real valued random variable Z, it is common to construct an interval [*a, b*] within which it is expected to fall as Rα = {z ∈ R : F(z) ∈ [a, b]}
This is based on the probability integral transform that states that the cumulative distribution function F maps variables to uniform distribution i.e. P(F(Z) ∈ [*a, b*]) = U([*a, b*]).

To guarantee a (1 − α) uncertainty region, it suffices to choose a and b such that U([*a, b*]) ≥ 1 − α which implies

$$\mathbb{P}\left(Z\in{\mathcal{R}}_{\alpha}\right)\geq1-\alpha.$$
 (Z ∈ Rα) ≥ 1 − α. (1)
Applying it to the real valued score Z = S(*X, Y* ) of the prediction model yˆ, an uncertainty set for the response of a given a input X can be expressed as

$${\mathcal{R}}_{\alpha}(X)=\left\{y\in{\mathcal{Y}}:F\circ S(X,y)\in[a,b]\right\}.$$

However, this result is typically not directly usable since the ground-truth distribution F is unknown and must be approximated empirically with Fn using a finite sample of data. When the sample size goes to infinity, one expects to recover Equation (1). The following result provides the tool to obtain the finite sample version.

Lemma 2.1. If Z1, . . . , Zn, Z be a sequence of real valued exchangeable random variables, then it holds

$$F_{n}(Z)\sim\mathbb{U}\left\{0,{\frac{1}{n}},{\frac{2}{n}},\ldots,1\right\}$$
$\mathbb{P}(F_{n}(Z)\in[a,b])=\mathbb{U}_{n+1}([a,b])=\frac{\left|nb\right|-\left[na\right]+1}{n+1}$.  
By choosing any *a, b* such that Un+1([*a, b*]) ≥ 1 − α, Lemma 2.1 guarantees a coverage, that is at least equal to the prescribed level of uncertainty

$$\mathbb{P}\left(Z\in{\mathcal{R}}_{\alpha,n}\right)\geq1-\alpha.$$

where, the uncertainty set Rα,n = Rα(Dn) is defined based on observations Dn = {Z1*, . . . , Z*n} and defined as:

$${\cal R}_{\alpha,n}=\left\{z\in\mathbb{R}:F_{n}(z)\in[a,b]\right\}.\tag{3}$$

In short, Equation (3) is an empirical version of Equation (2) based on finite sample data. The striking property is that it preserves the coverage probability (1 − α) and does not depend on the ground-truth distribution of the data.

Given data Dn, a prediction model yˆ and a new input Xn+1, one can build an uncertainty set for the unobserved output Yn+1 by applying it to observed score functions. Proposition 2.2 (Conformal Prediction Coverage). Consider Zi = S(Xi, Yi) for i in [n] and Z = S(Xn+1, Yn+1)
in Lemma *2.1. The conformal prediction set is defined as* Rα,n(Xn+1) = y ∈ Y : Fn ◦ S(Xn+1, y) ∈ [*a, b*]	
and satisfies a finite sample coverage guarantee

-1 $\in\mathcal{R}_{\alpha,n}(X_{n+1})$
P
 (yn+1 ∈ Rα,n(Xn+1)) ≥ 1 − α.

The surprising facts are that the coverage guarantee in Proposition 2.2, holds for the *unknown* ground-truth distribution of the data P, does not require quantifying the estimation error |Fn − F|, and is applicable to any prediction model yˆ
as long as it treats the data exchangeably, e.g., a pre-trained model independent of Dn.

Leveraging the quantile function F
−1 n = Qn, and by setting a = 0 and b = 1 − α, we have the usual description

$${\mathcal{R}}_{\alpha,n}(X_{n+1})=\left\{y\in{\mathcal{Y}}:S(X_{n+1},y)\leq Q_{n}(1-\alpha)\right\}$$
$$(2)^{\frac{1}{2}}$$

namely the set of all possible responses whose score rank is smaller or equal to ⌈(1 − α)(n + 1)⌉ compared to ranking of previously observed scores. For the absolute value difference score function, the CP set corresponds to

$${\mathcal{R}}_{\alpha,n}(X_{n+1})=\left[{\hat{y}}(X_{n+1})\pm Q_{n}(1-\alpha)\right].$$

Center-Outward View Another classical choice is a =
α 2 and b = 1 −
α 2
. In that case, we have the usual confidence set that corresponds to the range of values that captures the central proportion with α/2 of the data lying below Q(α/2) and α/2 lying above Q(1 − α/2). Introducing the center-outward distribution of Z as the function T = 2F − 1 , the probability integral transform T(Z) is uniform in the unit ball [−1, 1]. This ensures a symmetric description of Rα = T
−1(B(0, 1 − α)) around a central point such as the median Q(1/2) = T
−1(0). and the radius of the ball now directly corresponds to the desired confidence level of uncertainty. Similarly, we have the empirical center outward distribution Tn = 2Fn − 1 and the centeroutward view of the conformal prediction set follows as Rα,n(Xn+1) = y ∈ Y : |Tn ◦ S(Xn+1, y)| ≤ 1 − α	.

If Z follows a probability distribution P, then the transformation z 7→ T(z) is mapping the source distribution P to the uniform distribution U over the unit ball. In fact, it can be characterized as essentially the unique monotone increasing function such that T(Z) is uniformly distributed.

## 2.2. Multivariate Conformal Prediction

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 As recalled in (Dheur et al., 2025), several alternative conformal prediction approaches have been proposed to tackle multivariate prediction problems. While many conformal methods exist for univariate prediction, we focus here on those applicable to *multivariate* outputs. Some of these methods can directly operate using a simple predictor (e.g., a conditional mean) of the response y, while some may require stronger assumptions, such as requiring an estimator of the *joint* probability density function between x and y, or access to a generative model that mimics the *conditional* distribution of y given x) (Izbicki et al., 2022; Wang et al.,
2022).

We restrict our attention in this work to approaches that make no such assumption, reflecting our modeling choices for **OT-CP**. M-CP. We will consider the template approach of (Zhou et al., 2024) to use classical CP by aggregating a score function computed on each of the d outputs of the multivariate response. Given a conformity score si (to be defined next) for the i-th dimension, Zhou et al. (2024) define the following aggregation rule:

$$s_{\mathrm{M-CP}}(x,y)=\operatorname*{max}_{i\in[d]}s_{i}(x,y_{i}).$$

As (Dheur et al., 2025), we will use conformalized quantile regression (Romano et al., 2019) to define the score functions above, for each output i ∈ [d], where the conformity score is given by:

$$s_{i}(x,y_{i})=\operatorname*{max}\{{\hat{l}}_{i}(x)-y_{i},y_{i}-{\hat{u}}_{i}(x)\},$$

with ˆli(x) and uˆi(x) representing the lower and upper conditional quantiles of Yi|X = x at levels αl and αu, respectively. In our experiments, we consider equal-tailed prediction intervals, where αl =
α 2
, αu = 1 −
α 2
, and α denotes the miscoverage level. Merge-CP. An alternative approach is simply to use a squared Euclidean aggregation,

$$s(x,y):=\|{\hat{y}}(x)-y\|_{2},$$

where the choice of the norm (e.g., ℓ1, ℓ2, or ℓ∞) depends on the desired sensitivity to errors across tasks. This approach reduces the multidimensional residual to a scalar conformity score, leveraging the natural ordering of the real numbers. This simplification not only makes it straightforward to apply univariate conformal prediction methods, but also avoids the complexities of directly managing vectorvalued scores in conformal prediction. A variant consists of applying a Mahalanobis norm (Johnstone & Cox, 2021) in lieu of the squared Euclidean norm, using the covariance matrix Σ estimated from the training data (Johnstone & Cox, 2021; Katsios & Papadopulos, 2024),

$$s(x,y):=\|\Sigma^{-1/2}({\hat{y}}(x)-y)\|_{2},$$

## 2.3. Kantorovich Ranks

A naive way to define ranks in multiple dimensions might be to measure how far each point is from the origin and then rank them by that distance. This breaks down if the distribution of the data is stretched or skewed in certain directions. To correct for this, Hallin et al. (2021) developed a formal framework of center-outward distributions and quantiles, also called Kantorovich ranks (Chernozhukov et al., 2017), extending the familiar univariate concepts of ranks and quantiles into higher dimensions, building on elements of optimal transport theory. Let µ and ν be source and target probability measures on Ω ⊂ R
d. We consider the optimal transport problem with square Euclidean cost

$$\operatorname*{inf}_{\pi\in\Pi(\mu,\nu)}\int_{\Omega\times\Omega}\left\|\mathbf{x}-\mathbf{y}\right\|^{2}d\pi(\mathbf{x},\mathbf{y}),$$

where Π(*µ, ν*) is the set of all transport plans, i.e. joint distributions π on Ω × Ω whose marginals are µ and ν.

$$(4)^{\frac{1}{2}}$$

Optimal Transport Map One can look for a map T :
Ω → Ω that pushes forward µ to ν and minimizes the average transportation cost

$$T^{\star}\in\operatorname*{arg\,min}_{T_{\#}\mu=\nu}\int_{\Omega}\|x-T(x)\|^{2}\,d\mu(x).\tag{5}$$

Brenier's theorem states that if the source measure µ has a density, there exists a solution to 5 that is the gradient of a convex function ϕ : Ω → R such that T
⋆ = ∇ϕ.

In the one-dimensional case, the cumulative distribution function of a distribution P is the unique increasing function transporting it to the uniform distribution. This monotonicity property generalizes to higher dimensions through the gradient of a convex function ∇ϕ. Thus, one may view the optimal transport map in higher dimensions as a natural analog of the univariate cumulative distribution function both represent the unique monotone way to send one probability distribution onto another. Definition 2.3. The center-outward distribution of a random variable Z ∼ P is defined as the optimal transport map T = ∇ϕ that pushes P forward to the uniform distribution U
 on the unit ball B(0, 1). The rank of Z is defined as Rank(Z) = ∥T(Z)∥, the distance to origin.

Quantile region is an extension of quantiles to multiple dimensions to represent region in the sample space that contains a given proportion of probability mass. The quantile region at probability level (1−τ ) ∈ (0, 1) can be defined as

$${\mathcal{R}}_{\tau}=\{z\in\mathbb{R}^{d}:\|T(z)\|\leq1-\tau\}.$$

By definition of the spherical uniform distribution, we have ∥T(Z)∥ is uniform on (0, 1) which implies

$$\mathbb{P}(Z\in{\mathcal{R}}_{\tau})=1-\tau.$$
P(Z ∈ Rτ ) = 1 − τ. (6)

## 3. Kantorovich Conformal Prediction 3.1. Multi-Output Conformal Prediction

We consider that P is only available through a finite set of samples {Zi}
n+1 i=1 and a grid of U with as many points. We consider first the *discrete* transport map

$$T_{n+1}:\{Z_{i}\}_{i=1}^{n+1}\to\{U_{i}\}_{i=1}^{n+1}$$

which can be obtained by solving the optimal assignment problem, which seeks to minimize the total transport cost between the empirical distributions Pn+1 and Un+1:

$$T_{n+1}\in\operatorname*{arg\,min}_{T\in{\mathcal{T}}}\sum_{i=1}^{n+1}\|Z_{i}-T(Z_{i})\|^{2},$$

where T is the set of bijections mapping the observed sample {Zi}
n+1 i=1 to the grid {Ui}
n+1 i=1 .

Definition 3.1. Let (Z1, . . . , Zn, Zn+1) be a sequence of exchangeable variables in R
dthat follow a common distribution P. The discrete center-outward distribution Tn+1 is the transport map pushing forward Pn+1 to Un+1.

When dealing with empirical distribution with finite samples Z1, . . . , Zn, Zn+1 In this asymptotic regime (Chewi et al.,
2024), the empirical source distribution Pn+1 approximates the ground-truth P and as well as the empirical transport map Tn+1 approximates in sample the exact transport T
⋆.

Following (Hallin et al., 2021) to formalize the discrete spherical uniform distribution and its associated empirical cumulative distribution function, we begin by stating the construction of the discrete spherical uniform distribution involves a uniform grid defined such that the total number of points n = nRnS + no, where no points are at the origin.

- nS unit vectors u1*, . . . ,* unSare uniform on the sphere.

 - $n_R$ radius are regularly spaced as $\left\{\frac{1}{n_R},\frac{2}{n_R},\ldots,1\right\}$. 
The grid discretizes the sphere into layers of concentric shells, with each shell containing nS equally spaced points along directions determined by the unit vectors. The discrete spherical uniform distribution puts equal mass over each points of the grid that is to say no × 1/n mass on the origin and 1/n on the remaining. This ensures isotropic sampling at fixed radius onto [0, 1].

By definition of the target distribution Un+1, it holds

$$\|T_{n+1}(Z_{n+1})\|\sim\mathbb{U}\left\{0,{\frac{1}{n_{R}}},{\frac{2}{n_{R}}},\ldots,1\right\}.$$
$$(6)$$

In order to define an empirical quantile region as Equation (6), we need an extrapolation T¯n+1 of Tn+1 out of the samples (Zi)i∈[n+1]. By definition of such maps

$$\|\bar{T}_{n+1}(Z_{n+1})\|=\|T_{n+1}(Z_{n+1})\|$$

is still uniformly distributed and the empirical quantile region can be defined as

$${\mathcal{R}}_{\alpha,n+1}=\{z\in\mathbb{R}^{d}:\|{\bar{T}}_{n+1}(z)\|\leq1-\alpha\}$$

and expect that P (Z ∈ Rα,n+1) ≈ 1 − α when n is large.

Nevertheless, the core point of conformal prediction methodology is to go beyond asymptotic results or regularity assumptions about the data distribution. This is crucial because we only have access to a finite amount of data, and the ground-truth distribution of the data is unknown in practice. In that case, it is not immediate to have guarantee with respect to the ground-truth distribution such as Equation (7).

## 3.2. Optimal Transport Merging

We introduce the Optimal Transport Merging, a simple procedure that reduces any vector-valued score S(*x, y*) ∈ R
d in a one-dimension score. More precisely, we define the new non-conformity score function of an observation as

$$S_{\mathrm{OT-CP}}(x,y)=\|T^{\star}\circ S(x,y)\|_{2}$$

where T
⋆is the optimal Brenier (1991) map that pushes the distribution of vector-valued scores onto the uniform 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 ball distribution U in the same approach. This approach allows us to exploit the natural ordering of the real line, making it possible to directly apply one-dimensional conformal prediction methods to the sequence of transformed scores Zi = ∥SOT−CP(Xi, Yi)∥2 for i ∈ [n + 1].

In practical implementation, T
⋆can be replaced by any approximation Tˆ that preserves the permutation invariance of the score functions. We introduce the conformal prediction set resulting from the optimal transport merging is

$\Gamma\text{=CP}\left(X_{n+1},\,y\right)$
ROT−CP(Xn+1, α) = Rα(*T, X*n+1)
with respect to a given transport map T
Rα(T) = y : Fn(∥SOT−CP(Xn+1, y)∥2) ≤ 1 − α	.

have a coverage (1 − α), where Fn is empirical (univariate)
cumulative distribution function of the observed scores

$\{\|S_{\rm{or}-CP}(X_{1},Y_{1})\|,\ldots,\|S_{\rm{or}-CP}(X_{n},Y_{n})\|\}$.  
Proposition 2.2 implies P(Yn+1 ∈ ROT−CP(Xn+1)) ≥ 1 − α.

Remark 3.2. Our proposed conformal prediction framework **OT-CP** with optimal transport merging score function generalizes the **Merge-CP** approaches. More specifically, under the additional assumption that we are transporting a source Gaussian (resp. uniform) distribution to a target Gaussian (resp. uniform) distribution, the transport map is linear (Peyre & Cuturi ´ , 2019; Muzellec & Cuturi, 2018)

## 3.3. Coverage Guarantees Under Approximations

When dealing with high-dimensional data or complex distributions, it is essential to find computationally feasible methods to approximate the optimal transport map T
⋆ with a map Tˆ. In practical applications, we will rely on empirical approximations of the Brenier (1991) map using finite samples. Note that this approach may encouter a few statistical roadblocks, as such estimators are significantly hindered by the curse of dimensionality (Chewi et al., 2024). Consequently, one may think that these maps, not serving as reliable approximations, may hurt the performance of our approach. However, the machinery of conformal prediction presented earlier in the background section allows to maintain a coverage level, irrespective of sample size limitations. We defer the presentation of this practical approach to section 3.4 and focus first on coverage guarantees. Coverage of Approximated Quantile Region Let us assume an arbitrary approximation Tˆ of the Brenier
(1991) map and define the corresponding quantile region as 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274

$${\mathcal{R}}({\hat{T}},r)=\{z\in\mathbb{R}^{d}:\|{\hat{T}}(z)\|\leq r\},$$

The coverage in Equation (7) is not automatically maintained since Uˆ := Tˆ\#P may not coincide with U. As a result, the validity of the approximated quantile region may be compromised unless we can control the magnitude of the error
∥Uˆ − U∥, which requires additional regularity assumptions.

In its standard formulation, conformal prediction relies on an empirical setting and does not directly apply to the continuous case. Consequently, it does not provide a solution for calibrating entropic quantile regions, for example. However, a careful inspection of the one-dimensional case reveals that understanding the distribution of the probability integral transform is the key point:

$$\bullet\ \mathbb{U}\left(\left\{0,{\frac{1}{n}},{\frac{1}{2}},\ldots,1\right\}\right)\sim F_{n}(Z)\neq F(Z)\sim\mathbb{U}(0,1)\ .$$

Instead of relying on an analysis of approximation error to quantify the deviation |Fn − F| under certain regularity conditions, conformal prediction fully characterizes the distribution of the probability integral transform and calibrates the radius of the quantile region accordingly. We follow this very simple idea and note that by definition

$$\mathbb{P}({\mathcal{R}}({\hat{T}},r))=1$$

P(R(*T , r* ˆ )) = P(∥Tˆ(z)∥ ≤ r) = Uˆ(B(0, r)).

Instead of relying on Uˆ ≈ U, we define rα(T , ˆ P) = inf{r : Uˆ(B(0, r)) ≥ 1 − α}
that leads to a desired coverage with the approximated transported map . For a radius rˆα = rα(T , ˆ P), it holds

$1-\alpha\}$. 
$$\mathbb{P}\left(Z\in{\mathcal{R}}({\hat{T}},{\hat{r}}_{\alpha})\right)\geq1-\alpha.$$

By extension, a quantile region of the vector-valued score Z = S(*X, Y* ) ∈ R
d of a prediction model yˆ provides an uncertainty set for the response of a given a input X, with prescribed coverage (1 − α) can be expressed as

$${\mathcal{R}}_{\alpha}(X)=\big\{y\in{\mathcal{Y}}:\|T\circ S(X,y)\|\leq1-\alpha\big\}.$$
$$\mathbb{P}(Y\in{\mathcal{R}}_{\alpha}(X))=1-\alpha.$$
$$(7)$$
P(Y ∈ Rα(X)) = 1 − α. (7)
In the following result, we give the finite sample analog of Equation (6), which provides a finite sample guarantee for our optimal transport approach.

Lemma 3.3 (Coverage of Empirical Quantile Region). Let Z1, . . . , Zn, Zn+1 *be a sequence of exchangeable variable* in R
d, then, P(Zn+1 ∈ Rα,n+1) ≥ 1 − α.

Remark that the source probability in Lemma 3.3 is the ground-truth P. Given a transport map Tˆ and applying and the empirical radius rα,n+1 = rα(T , ˆ Pn+1), it holds

$$\mathbb{P}_{n+1}(Z_{n+1}\in{\mathcal{R}}({\hat{T}},r_{\alpha,n+1}))\geq1-\alpha.$$

However, this is *only* an empirical coverage statement:
which does not implies coverage with respect to P unless n → ∞. The following steps show how to obtain finite sample validity. Proof. For simplicity, we will denote the quantile region as Rα,n+1 = R(T , r ˆα,n+1). Then by exchangeability:

$$\mathbb{P}(Z_{n+1}\in\mathcal{R}_{\alpha,n+1})=\frac{1}{n+1}\sum_{i=1}^{n+1}\mathbb{P}_{n+1}(Z_{i}\in\mathcal{R}_{\alpha,n+1})$$ $$=\mathbb{E}\left[\frac{1}{n+1}\sum_{i=1}^{n+1}\mathds{1}\{Z_{i}\in\mathcal{R}_{\alpha,n+1}\}\right]$$ $$=\mathbb{E}\bigg{[}\mathbb{P}_{n+1}(Z_{n+1}\in\mathcal{R}_{\alpha,n+1})\bigg{]}$$ $$\geq1-\alpha.$$

with rˆα,n = inf r ≥ 0 : Uˆ(B(0, r)) ≥ 1 − α	. It satisfies a distribution-free finite sample coverage guarantee

$$\mathbb{P}\left(Y_{n+1}\in\hat{\mathcal{R}}_{\alpha,n+1}(X_{n+1})\right)\geq1-\alpha.\tag{8}$$

275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 Approaches relying on vector-valued probability integral transform, e.g., leveraging Copulas have been explored recently (Messoudi et al., 2021; Park et al., 2024) and concluded that loss of coverage can occur when the estimated copula of the scores deviates from the true copula and thus does not formally guarantee finite-sample validity. To our knowledge, Proposition 3.4 provides the first calibration guarantee for such confidence regions without assumptions on the distribution, for any approximation map Tˆ. Specifically using the discrete spherical uniform grid implies: Proposition 3.5. Given n discrete sample points distributed over a sphere with radius {0, 1 nR
,
2 nR
, . . . , 1} and directions uniformly sampled on the sphere, the smallest radius to obtain a coverage (1 − α) *is determined by*

$r_{\alpha}=\frac{j_{\alpha}}{n_{R}}$ where $j_{\alpha}=\left[\frac{n(1-\alpha)-n_{\alpha}}{n_{S}}\right]$, $n_{S}$ is the number of directions, $n_{R}$ is the number of directions.  
where nS is the number of directions, nR *is the number of* radius, and no *is the number of copies of the origin.*
Remark 3.6. When the discrete transport problem is solved approximately and one obtain Tˆn+1, then choosing rˆα,n+1 = rα(Tˆn+1, Pn+1) ensure finite sample coverage just as Section 3.3. So one can take benefit of numerical efficiency without sacrificing valid coverage.

## 3.4. Implementation With The Entropic Map

We assume access to two sample sets, i.e., one containing residuals µˆn =
1 n Pi δz i , and the second containing the discretized uniform grid on the sphere, νˆm =1m Pj δuj ,
not necessarily assuming a same size, namely n ̸= m. A convenient estimator for the Brenier map T
⋆is the entropic map (Pooladian & Niles-Weed, 2021). Let ε > 0 and write Kij = [exp(−∥z i − u j∥
2/ε)]ij the kernel matrix. One can then define,

$$\mathbf{f^{*},g^{*}=\operatorname*{argmax}\limits_{\mathbf{f}\in\mathbb{R}^{n},\mathbf{g}\in\mathbb{R}^{m}}\langle\mathbf{f},\frac{\mathbf{1}_{n}}{n}\rangle+\langle\mathbf{g},\frac{\mathbf{1}_{m}}{m}\rangle-\varepsilon\langle e^{\frac{\mathbf{f}}{\varepsilon}},Ke^{\frac{\mathbf{g}}{\varepsilon}}\rangle\,.\tag{9}$$
$$\square$$

Problem (9) is an unconstrained concave optimization problem known as the regularized OT problem in dual form (?, Prop. 4.4). Problem (9) can be solved numerically with the Sinkhorn algorithm (Cuturi, 2013). Equipped with these optimal vector, one can define the maps, valid out of sample,

$$\begin{array}{l}{{f_{\varepsilon}(z)=\operatorname*{min}_{\varepsilon}([\|z-u^{j}\|^{2}-{\bf g}_{j}^{*}]_{j})\,,}}\\ {{g_{\varepsilon}(u)=\operatorname*{min}_{\varepsilon}([\|z^{i}-u\|^{2}-{\bf f}_{i}^{*}]_{i})\,,}}\end{array}$$
j]j ), (10)
]i), (11)
where for a vector u or arbitrary size s we define the logsum-exp operator as minε(u) := −ε log( 1 s 1 T
se
−u/ε). Using the Brenier (1991) theorem, linking potential values to optimal map estimation, one obtains an estimator for T
⋆:

$$T_{\varepsilon}(z):=z-\nabla f_{\varepsilon}(z)=\sum_{j=1}^{m}p^{j}(z)u^{j}\,,\tag{12}$$
$$(10)$$ $$(11)$$

where the weights depend on z as:

$$p^{j}(z):=\frac{\exp\left(-\left(\|z-u^{j}\|^{2}-\mathbf{g}_{j}^{\star}\right)/\varepsilon\right)}{\sum_{k=1}^{m}\exp\left(-\left(\|z-u^{k}\|^{2}-\mathbf{g}_{k}^{\star}\right)/\varepsilon\right)}\,.\tag{13}$$

One can obtain, analogously, an estimator for the inverse map (T
⋆)
−1 using the potential gε, as demonstrated in Fig. 5. Using the entropic map estimator requires running the Sinkhorn (1964) algorithm on a n × m cost matrix at train time, and at each evaluation, compute weights in (13) that require computing the distance of any incoming point z to the uniform grid. The complexity is therefore O(nm) when training the map and conformalizing its scores, and then O(m) at each evaluation of a score for a given y.

Sampling on the sphere As mentioned in (Hallin et al.,
2021), it is preferable to sample the uniform measure Ud with diverse samples, and this can be achieved using stratified sampling on radii lengths, but, most importantly, lowdisrepancy samples on the sphere to pick sampling directions. We borrow inspiration from the review provided in This can be directly applied to obtain conformal prediction set for vector-valued non-conformity score functions Zi =
S(Xi, Yi) ∈ R
dfor i in [n + 1] in Lemma 3.3.

Proposition 3.4. *The conformal prediction set is defined as*

$\hat{\mathcal{R}}_{\alpha,n+1}(X_{n+1})=\left\{y\in\mathcal{Y}:\|\hat{T}\circ S(X_{n+1},y)\|\leq\hat{r}_{\alpha,n+1}\right\}$
$${\frac{1}{n+1}}\sum_{i=1}^{n+1}\mathbf{1}\{Z_{i}\in{\mathcal{R}}({\hat{T}},r_{\alpha,n+1})\}\geq1-\alpha$$
10 0 10 1 10 2 10 3 method M-CP Merge-CP Merge-CP (Mah) OT-CP
reg ion size ansur2 (2)bio (2)births1 (2)
blog_
data (2)
calcofi (2)edm (2)enb (2)house (2)taxi (2)jura (3)scpf (3)sf1 (3)sf2 (3)slump (3)
households (4)air (6)atp1d (6)atp7d (6)
(Nguyen et al., 2024) to pick their *Gaussian based mapping* approach (Basu, 2016). This consists in mapping a lowdiscrepancy sequence w1*, . . . , w*L on [0, 1]dto a potentially low-discrepancy sequence θ1*, . . . , θ*L on S
d−1through the mapping θ = Φ−1(w)/∥Φ
−1(w)∥2, where Φ
−1is the inverse CDF of N (0, 1) applied entry-wise.

## 4. Experiments 4.1. Setup And Metrics

330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 We borrow the experimental setting provided by Dheur et al. (2025) and benchmark multivariate conformal methods on a total of 24 tabular datasets. Total data size n in these datasets ranges from 103 to 50,000, with input dimension p ranging from 1 to 348, and output dimension d ranging from 2 to 16. We adopt their approach, which is to rely on a multivariate quantile function forecaster (MQF2, Kan et al.,
2022), a normalizing flow that is able to quantify output uncertainty conditioned on input x. However, in accordance with our stance mentioned in the background section, we will only assume access to the conditional mean (point-wise) estimator for **OT-CP**. As is common in the field, we evaluate the methods using several metrics, including marginal coverage (MC), and mean region size (Size). The latter is using importance sampling, leveraging (when computing test time metrics only), the generative flexibility provided by the MQF2as an invertible flow. See (Dheur et al., 2025) and their code for more details on the experimental setup.

## 4.2. Hyperparameter Choices

We apply default parameters for all three competing methods, **M-CP** and **Merge-CP**, using (or not) the Mahalanobis correction. For **M-CP** using conformalized quantile regression boxes, we follow (Dheur et al., 2025) and leverage the empirical quantiles return by MQF2to compute boxes
(Zhou et al., 2024). OT-CP our implementation requires essentially tuning two important hyperparameters: the entropic regularization ε and the total number of points used to discretize the sphere m, not necessarily equal to the input data sample size. These two parameters describe a fundamental statistical and computational trade-off. On the one hand, it is known that increasing m will mechanically improve the ability of Tε to recover in the limit T
⋆(or at least solve the semidiscrete (Peyre & Cuturi ´ , 2019) problem of mapping n data points to the sphere). However, large m incurs a heavier computational price when running the Sinkhorn algorithm. On the other hand, increasing ε improves on *both* computational and statistical aspects, but deviates further the estimated map from the ground truth T
⋆to target instead a blurred map. We have experimented with these aspects and derive from our experiments that both m and ε should be increased to track increase in dimension. As a sidenote, we do observe that debiasing the outputs of the Sinkhorn algorithm does not result in improved results, which agrees with the findings in (Pooladian et al., 2022).

## 4.3. Results

We present results by differentiating datasets with small dimension d ≤ 6 from datasets with higher dimensionality, that we expect to be more challenging to handle with OT approaches, owing to the curse of dimensionality that might degrade the quality of multivariate quantiles.

Multivariate Conformal Prediction using Optimal Transport
#target points = 4096

#target points = 8192

#target points = 16384

#target points = 32768 scm20d (
16)
10 1 10 1 10 3 10 5 10 7 10 9 10 11 epsilon 0.001 0.010.1 1.0 region si ze ansur2 (
2)
bio (2)
births1 (
2)
blog_data 
(2)
calcofi (
2)
edm (2)enb (2)
house (2)taxi (2)jura (3)scpf (3)sf1 (3)sf2 (3)slump (3)
househol ds (4)air (6)
atp1d (6)atp7d (6)wq (14)
oes10 (1 6)
oes97 (1 6)
scm1d (1 6)
ansur2 (
2)
bio (2)
births1 (
2)
blog_data 
(2)
calcofi (
2)
edm (2)enb (2)
house (2)taxi (2)jura (3)scpf (3)sf1 (3)sf2 (3)slump (3)
househol ds (4)air (6)
atp1d (6)atp7d (6)wq (14)
oes10 (1 6)
oes97 (1 6)
scm1d (1 6)
scm20d (
16)
ansur2 (
2)
bio (2)
births1 (
2)
blog_data 
(2)
calcofi (
2)
edm (2)enb (2)
house (2)taxi (2)jura (3)scpf (3)sf1 (3)sf2 (3)slump (3)
househol ds (4)air (6)
atp1d (6)atp7d (6)wq (14)
oes10 (1 6)
oes97 (1 6)
scm1d (1 6)
scm20d (
16)
ansur2 (
2)
bio (2)
births1 (
2)
blog_data 
(2)
calcofi (
2)
edm (2)enb (2)
house (2)taxi (2)jura (3)scpf (3)sf1 (3)sf2 (3)slump (3)
househol ds (4)air (6)
atp1d (6)atp7d (6)wq (14)
oes10 (1 6)
oes97 (1 6)
scm1d (1 6)
scm20d (
16)
method M-CP Merge-CP Merge-CP (Mah) OT-CP
atp7 d (6)
0 100 200 300 time 
(s)
ansu r2 (2
)

bio (
2)
birth s1 (2
)

blog _dat a (2)
calco fi (2)
edm 
(2)
enb (
2)
hous e (2)
taxi (
2)
jura 
(3)
scpf 
(3)
sf1 (
3)
sf2 (
3)
slum p (3)
house holds
 (4)
air (
6)
atp1 d (6)
385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439

## 5. Conclusion

We have proposed **OT-CP**, a new approach that can leverage a recently proposed formulation for multivariate quantiles that uses optimal transport theory and optimal transport map estimators. We show the theoretical soundness of this approach, but, most importantly, demonstrate its applicability throughout a broad range of tasks compiled by (Dheur et al., 2025). Compared to similar baselines that either leverage a conditional mean regression estimator (**Merge-CP**), or more involved quantile regression estimators (M-CP), OT- CP displays superior performance overall, while incurring, predictably, a higher train / calibration time cost. The challenges brought forward by the estimation of OT maps in high dimensions (Chewi et al., 2024) require being particularly careful when tuning entropic regularization and grid size. However, we show that there exists a reasonable setting for both these parameters that delivers good performance across most tasks.

wq (14)
oes10 (16)oes97 (16)scm1d (16)scm20d (16)
10 2 10 4 10 6 10 8 10 10 method M-CP Merge-CP Merge-CP (Mah) OT-CP
regio n siz e

## Impact Statement References

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. Balasubramanian, V., Ho, S.-S., and Vovk, V. Conformal prediction for reliable machine learning: theory, adaptations and applications. Newnes, 2014.

Barber, R. F., Candes, E. J., Ramdas, A., and Tibshirani, R. J. Conformal prediction beyond exchangeability. The Annals of Statistics, 51(2):816–845, 2023.

Basu, K. Quasi-Monte Carlo Methods in Non-Cubical Spaces. Stanford University, 2016.

Bates, S., Candes, E., Lei, L., Romano, Y., and Sesia, `
M. Testing for outliers with conformal p-values. arXiv preprint arXiv:2104.08279, 2021.

Cella, L. and Ryan, R. Valid distribution-free inferential models for prediction. *arXiv preprint arXiv:2001.09225*, 2020.

Chernozhukov, V., Galichon, A., Hallin, M., and Henry, M.

Monge–Kantorovich depth, quantiles, ranks and signs. The Annals of Statistics, 45(1):223 - 256, 2017. doi:
10.1214/16-AOS1450. URL https://doi.org/10. 1214/16-AOS1450.

Chernozhukov, V., Wuthrich, K., and Zhu, Y. Exact and ro- ¨
bust conformal inference methods for predictive machine learning with dependent data. Conference On Learning Theory, 2018.

Chernozhukov, V., Wuthrich, K., and Zhu, Y. An exact and ¨
robust conformal inference method for counterfactual and synthetic controls. *Journal of the American Statistical* Association, 116(536):1849–1864, 2021.

Cuturi, M. Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in neural information processing systems, pp. 2292–2300, 2013.

Cuturi, M., Teboul, O., and Vert, J.-P. Differentiable ranking and sorting using optimal transport. *Advances in neural* information processing systems, 32, 2019.

Dheur, V., Fontana, M., Estievenart, Y., Desobry, N., and Taieb, S. B. Multi-output conformal regression: A unified comparative study with new conformity scores, 2025. URL https://arxiv.org/abs/2501.10533.

Fisch, A., Schuster, T., Jaakkola, T., and Barzilay, R. Fewshot conformal prediction with auxiliary tasks. ICML, 2021.

Gammerman, A., Vovk, V., and Vapnik, V. Learning by transduction, 1998.

Guha, E., Natarajan, S., Mollenhoff, T., Khan, M. E., ¨
and Ndiaye, E. Conformal prediction via regressionas-classification. *arXiv preprint arXiv:2404.08168*, 2024.

Hallin, M., del Barrio, E., Cuesta-Albertos, J., and Matran, ´
C. Distribution and quantile functions, ranks and signs in dimension d: A measure transportation approach. The Annals of Statistics, 49(2):1139 - 1165, 2021. doi: 10.1214/20-AOS1996. URL https://doi.org/10. 1214/20-AOS1996.

Hallin, M., La Vecchia, D., and Liu, H. Center-outward r-estimation for semiparametric varma models. Journal of the American Statistical Association, 117(538):925–938, 2022.

Hallin, M., Hlubinka, D., and Hudecova,´ S. Efficient fully ˇ
distribution-free center-outward rank tests for multipleoutput regression and manova. Journal of the American Statistical Association, 118(543):1923–1939, 2023.

Ho, S.-S. and Wechsler, H. Query by transduction. IEEE
transactions on pattern analysis and machine intelligence, 2008.

Holland, M. J. Making learning more transparent using conformalized performance prediction. arXiv preprint arXiv:2007.04486, 2020.

Izbicki, R., Shimizu, G., and Stern, R. B. Cd-split and hpd-split: Efficient conformal regions in high dimensions. *Journal of Machine Learning Research*, 23(87):
1–32, 2022.

Johnstone, C. and Cox, B. Conformal uncertainty sets for robust optimization. In Carlsson, L., Luo, Z., Cherubin, G., and An Nguyen, K. (eds.), *Proceedings of the* Tenth Symposium on Conformal and Probabilistic Prediction and Applications, volume 152 of Proceedings of Machine Learning Research, pp. 72–90. PMLR, 08– 10 Sep 2021. URL https://proceedings.mlr.

press/v152/johnstone21a.html.

Kan, K., Aubet, F.-X., Januschowski, T., Park, Y., Benidis, K., Ruthotto, L., and Gasthaus, J. Multivariate quantile 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Chewi, S., Niles-Weed, J., and Rigollet, P. Statistical optimal transport. *arXiv preprint arXiv:2407.18163*, 2024.

Brenier, Y. Polar factorization and monotone rearrangement of vector-valued functions. *Communications on Pure* and Applied Mathematics, 44(4), 1991. doi: 10.1002/cpa. 3160440402. function forecaster. In International Conference on Artificial Intelligence and Statistics, pp. 10603–10621. PMLR, 2022.

Katsios, K. and Papadopulos, H. Multi-label conformal prediction with a mahalanobis distance nonconformity measure. In Vantini, S., Fontana, M., Solari, A., Bostrom, ¨ H., and Carlsson, L. (eds.), Proceedings of the Thirteenth Symposium on Conformal and Probabilistic Prediction with Applications, volume 230 of *Proceedings of Machine* Learning Research, pp. 522–535. PMLR, 09–11 Sep 2024. URL https://proceedings.mlr.press/ v230/katsios24a.html.

Kumar, B., Lu, C., Gupta, G., Palepu, A., Bellamy, D.,
Raskar, R., and Beam, A. Conformal prediction with large language models for multi-choice question answering. arXiv preprint arXiv:2305.18404, 2023.

Laxhammar, R. and Falkman, G. Inductive conformal anomaly detection for sequential detection of anomalous sub-trajectories. *Annals of Mathematics and Artificial* Intelligence, 2015.

Lin, Z., Trivedi, S., and Sun, J. Conformal prediction intervals with temporal dependence. *Transactions of Machine* Learning Research, 2022.

Lu, C., Lemay, A., Chang, K., Hobel, K., and Kalpathy- ¨
Cramer, J. Fair conformal predictors for applications in medical imaging. In *Proceedings of the AAAI Conference* on Artificial Intelligence, volume 36, pp. 12008–12016, 2022.

Messoudi, S., Destercke, S., and Rousseau, S. Copula-based conformal prediction for multi-target regression. Pattern Recognition, 120:108101, 2021.

Muzellec, B. and Cuturi, M. Generalizing point embeddings using the wasserstein space of elliptical distributions. Advances in Neural Information Processing Systems, 31, 2018.

Nguyen, K., Bariletto, N., and Ho, N. Quasi-monte carlo for 3d sliced wasserstein. In The Twelfth International Conference on Learning Representations, 2024.

Park, J. W., Tibshirani, R., and Cho, K. Semiparametric conformal prediction. *arXiv preprint arXiv:2411.02114*,
2024.

Peyre, G. and Cuturi, M. Computational optimal transport. ´
Foundations and Trends® *in Machine Learning*, 11, 2019.

Pooladian, A.-A. and Niles-Weed, J. Entropic estimation of optimal transport maps. *arXiv preprint arXiv:2109.12004*,
2021.

Pooladian, A.-A., Cuturi, M., and Niles-Weed, J. Debiaser beware: Pitfalls of centering regularized transport maps. In *International Conference on Machine Learning*, pp. 17830–17847. PMLR, 2022.

Quach, V., Fisch, A., Schuster, T., Yala, A., Sohn, J. H.,
Jaakkola, T. S., and Barzilay, R. Conformal language modeling. *arXiv preprint arXiv:2306.10193*, 2023.

Romano, Y., Patterson, E., and Candes, E. Conformalized quantile regression. Advances in neural information processing systems, 32, 2019.

Shafer, G. and Vovk, V. A tutorial on conformal prediction.

Journal of Machine Learning Research, 2008.

Sinkhorn, R. A relationship between arbitrary positive matrices and doubly stochastic matrices. *Ann. Math. Statist.*, 35:876–879, 1964.

Straitouri, E., Wang, L., Okati, N., and Rodriguez, M. G.

Improving expert predictions with conformal prediction.

In *International Conference on Machine Learning*, pp.

32633–32653. PMLR, 2023.

Tibshirani, R. J., Foygel Barber, R., Candes, E., and Ramdas, A. Conformal prediction under covariate shift. *Advances* in neural information processing systems, 32, 2019.

Vovk, V., Gammerman, A., and Shafer, G. Algorithmic learning in a random world. Springer, 2005.

Wang, Z., Gao, R., Yin, M., Zhou, M., and Blei, D. M. Probabilistic conformal prediction using conditional random samples. *arXiv preprint arXiv:2206.06584*, 2022.

Xu, C. and Xie, Y. Conformal prediction interval for dynamic time-series. *ICML*, 2021.

Zaffran, M., Feron, O., Goude, Y., Josse, J., and Dieuleveut, ´
A. Adaptive conformal predictions for time series. In International Conference on Machine Learning, pp. 25834– 25866. PMLR, 2022.

Zhou, Y., Lindemann, L., and Sesia, M. Conformalized adaptive forecasting of heterogeneous trajectories. arXiv preprint arXiv:2402.09623, 2024.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604

## A. Appendix

We provide a few additional results related to the experiments proposed in Section 4 Figure 8. Ablation: coverage quality as a function of hyperparameters, with the setting corresponding to Fig.2

#target points = 4096
#target points = 8192
#target points = 16384
#target points = 32768 epsilon 0.001 0.01 0.1 1.0 ansur2 (2) bio (2) births1 (2) 
blog_data (2
) 
calcofi (2) edm (2) enb (2) house (2) taxi (2) jura (3) scpf (3) sf1 (3) sf2 (3) slump (3) 
household s (4) air (6) 
atp1d (6) atp7d (6) wq (14) oes10 (16) oes97 (16) scm1d (16) 
scm20d (16
)

0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 coverage ansur2 (2)bio (2)births1 (2)
blog_data (2
)

calcofi (2)edm (2)enb (2)house (2)taxi (2)jura (3)scpf (3)sf1 (3)sf2 (3)slump (3)
household s (4)
air (6)
atp1d (6)atp7d (6)wq (14)
oes10 (16)oes97 (16)
scm1d (16
)

scm20d (16
)

ansur2 (2)bio (2)births1 (2)
blog_data (2
)

calcofi (2)edm (2)enb (2)house (2)taxi (2)jura (3)scpf (3)sf1 (3)sf2 (3)slump (3)
household s (4)
air (6)
atp1d (6)atp7d (6)wq (14)
oes10 (16)oes97 (16)
scm1d (16
)

scm20d (16
)

ansur2 (2)bio (2)births1 (2)
blog_data (2
)

calcofi (2)edm (2)enb (2)house (2)taxi (2)jura (3)scpf (3)sf1 (3)sf2 (3)slump (3)
household s (4)
air (6)
atp1d (6)atp7d (6)wq (14)
oes10 (16)oes97 (16)
scm1d (16
)

scm20d (16
)

wq (14)
oes10 (16)oes97 (16)scm1d (16)scm20d (16)
0.60 0.65 0.70 0.75 0.80 0.85 0.90 cover age method M-CP Merge-CP Merge-CP (Mah) OT-CP
method M-CP Merge-CP Merge-CP (Mah) OT-CP
wq (14)
oes10 (16)oes97 (16)scm1d (16)scm20d (16)
0.0 2.5 5.0 7.5 10.0 12.5 time (
s)
605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 0.9 co ve ra gemethod M-CP
0.8 Merge-CP Merge-CP (Mah)
0.7 OT-CP
at p7 d 
(6)
0.6 an su r2
 (
2)
bi o (
2)
bi rt hs 1 (
2)
bl og
_d at a (
2)
ca lc ofi
 (
2)
ed m 
(2
)

en b (
2)
ho us e (
2)
ta xi 
(2
)

ju ra
 (
3)
sc pf 
(3
)

sf 1 (
3)
sf 2 (
3)
sl u m p (
3)
ho us eh ol ds
 (
4)
ai r (
6)
at p1 d 
(6)
#target points = 4096
#target points = 8192
#target points = 16384
#target points = 32768 epsilon 0.001 0.01 0.1 1.0 scm20d (1 6)
0 100 200 300 400 time (s)
ansur2 (2)bio (2)births1 (2)
blog_data
 (2)
calcofi (2)edm (2)enb (2)house (2)taxi (2)jura (3)scpf (3)sf1 (3)sf2 (3)slump (3)
household s (4)
air (6)
atp1d (6)atp7d (6)wq (14)
oes10 (16
)

oes97 (16
)

scm1d (1 6)
ansur2 (2)bio (2)births1 (2)
blog_data
 (2)
calcofi (2)edm (2)enb (2)house (2)taxi (2)jura (3)scpf (3)sf1 (3)sf2 (3)slump (3)
household s (4)
air (6)
atp1d (6)atp7d (6)wq (14)
oes10 (16
)

oes97 (16
)

scm1d (1 6)
scm20d (1 6)
ansur2 (2)bio (2)births1 (2)
blog_data
 (2)
calcofi (2)edm (2)enb (2)house (2)taxi (2)jura (3)scpf (3)sf1 (3)sf2 (3)slump (3)
household s (4)
air (6)
atp1d (6)atp7d (6)wq (14)
oes10 (16
)

oes97 (16
)

scm1d (1 6)
scm20d (1 6)
ansur2 (2)bio (2)births1 (2)
blog_data
 (2)
calcofi (2)edm (2)enb (2)house (2)taxi (2)jura (3)scpf (3)sf1 (3)sf2 (3)slump (3)
household s (4)
air (6)
atp1d (6)atp7d (6)wq (14)
oes10 (16
)

oes97 (16
)

scm1d (1 6)
scm20d (1 6)