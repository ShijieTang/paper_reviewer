011

014 015 016

018

024

026

034

036

038

# EBMaC: Empirical Bayes and Matrix Constraints for Label Shift

Anonymous Authors<sup>1</sup>

# Abstract

We estimate the importance weights and their associated confidence set in label shift problems using hierarchical models via the Empirical Bayes and Matrix Constraints (EBMaC) method. Our approach accommodates dispersion beyond what is permitted by the classic multinomial model and produces exact confidence regions in finite samples for confusion matrix and predicted labels. In addition, we describe the dependence structure of the importance weights in matrix constraints. Through a linear programming technique, we are able to compute smaller confidence sets and shorter elementwise confidence intervals for importance weights compared to existing methods, while maintaining the probability guarantee. Applying the results to prediction in the target domain directly yields smaller conformal prediction set and PAC prediction set. Numerical experiments demonstrate the advantages of EBMaC in producing tighter confidence sets for the importance weights both marginally and jointly.

# 1. Introduction

When we simultaneously consider data sets from different sources, problems of distribution shift naturally arise. The most frequently studied distribution shifts are covariate shift and label shift. Here, we focus on label shift, which describes the scenario where the marginal distributions of the labels differ in the source and the target domains, but given the label, the conditional distributions of covariates remain unchanged. The key quantity of interest is importance weights, *i.e.* the ratios of the label proportions between the two domains.

Given a classifier, there are three types of approaches for estimating the importance weights. The first one mainly relies on the linear relationship of the confusion matrix and the predicted label distribution [\(Lipton et al.,](#page-7-0) [2018;](#page-7-0) [Aziz](#page-5-0)[zadenesheli et al.,](#page-5-0) [2019\)](#page-5-0), and is named the confusion matrix method. The classifier is used to produce the confusion matrix in the source domain and to generate the predicted label distribution in the target domain. In forming the confusion matrix, either hard assignments or soft assignments can be implemented [\(Garg et al.,](#page-5-1) [2020\)](#page-5-1). The difference between BBSE [\(Lipton et al.,](#page-7-0) [2018\)](#page-7-0) and RLLS [\(Azizzadenesheli](#page-5-0) [et al.,](#page-5-0) [2019\)](#page-5-0) is that BBSE pioneered the method while RLLS refined it by adding a regularization term on the importance weights to address potential near-singularity issues in the confusion matrix. The second one estimates the importance weights by maximum likelihood estimator (MLE). To this end, [Saerens et al.](#page-7-1) [\(2002\)](#page-7-1) proposed MLLS which finds the MLE by EM algorithm. [Alexandari et al.](#page-5-2) [\(2020\)](#page-5-2) proposed BCTS and demonstrated that further calibrating a classifier on the source domain significantly improves the MLE. The improvement happens because a classifier trained on the source domain may not perfectly represent the true proportions of the labels, even if it achieves high prediction probabilities [\(Guo et al.,](#page-5-3) [2017\)](#page-5-3). Such miscalibration biases the label predicting probability in the source domain and thus the estimated importance weights. The last one solves an estimating equation, formed by the projected score function, and is named ELSA [\(Tian et al.,](#page-7-2) [2023\)](#page-7-2). ELSA has the feature of being robust to an uncalibrated classifier, and it outperforms BCTS in computational efficiency while maintaining competitive accuracy.

In terms of the confidence intervals of the importance weights, most results hold only in the asymptotic sense. In finite samples, BBSE and RLLS rely on expressing the estimators explicitly in terms of confusion matrix and predicted label distribution. On the other hand, [Si et al.](#page-7-3) [\(2023\)](#page-7-3) proposed the Gaussian elimination (GE) method, where they modified each step of the Gaussian elimination procedure when solving the linear system in the confusion matrix method. Nevertheless, these methods do not produce tight confidence sets.

We propose EBMaC (Empirical Bayes and Matrix Constraints) method in the confusion matrix method class. We first construct confidence regions for the confusion matrix and the predicted label distributions using empirical Bayes method in a hierarchical model. It incorporates the overdispersion phenomenon, which is often encountered in

<sup>1</sup>Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

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

108 109 practice. We further take into account by recognizing it as a linear programming problem. This allows us to bypass matrix inversion and to obtain the tightest confidence sets for the importance weights. Furthermore, we demonstrate that applying the resulting confidence set yields the smallest finite sample prediction sets in the target domain. The superiority of EBMaC is rigorously proven in theory and illustrated through extensive numerical experiments.

## 2. Problem Setup

Let x ∈ X = **R**<sup>d</sup> and y ∈ Y = {1, . . . , K} ≡ [K] ⊂ **N**<sup>+</sup> denote the feature and the label, respectively. Let P and Q represent the source and target distributions, respectively. Then, let p(·) denote the pmf/pdf for the source domain and q(·) for the target domain. The letters in the () reflect corresponding random variables. For example, p(y) is the marginal pmf of the labels in the source domain. The label shift setting assumes that p(x | y) = q(x | y) while p(y) ̸= q(y) in general. Suppose we have a classifier g, defined as g : X → Y. Then the K × K confusion matrix C<sup>g</sup> is defined by (Cg)ij ≡ **P**(X,Y )∼<sup>P</sup> {g(X) = i, Y = j}, for i, j ∈ [K], where **P**(X,Y )∼<sup>P</sup> stands for probability in source domain. In the target domain, we define the predicted label proportions, q<sup>g</sup> = (q1, ..., qK) <sup>⊤</sup>. For each k ∈ [K], q<sup>k</sup> = **P**X∼Q<sup>X</sup> {g(X) = k}, where Q<sup>X</sup> is the distribution of X in the target domain. We also use **P** to denote the probability in the combined domain. Let ω = (ω1, ..., ωK) <sup>⊤</sup> be the importance weights, where ω<sup>y</sup> = q(y)/p(y) for y ∈ [K]. We aim at the estimation and inference for ω.

*Remark* 2.1*.* Under label shift, for any classifier g such that <sup>y</sup>b <sup>=</sup> <sup>g</sup>(x), we have <sup>p</sup>(yb| <sup>y</sup>) = <sup>q</sup>(yb| <sup>y</sup>). It is clear that P <sup>y</sup>∈Y <sup>p</sup>(y, y b )ω<sup>y</sup> <sup>=</sup> <sup>q</sup>(yb). When the confusion matrix C<sup>g</sup> is invertible, solving the linear system Cgω = q<sup>g</sup> is a valid method for estimating the importance weights ω. In practice, since the true values of C<sup>g</sup> and q<sup>g</sup> are unknown, we estimate them by the sample versions using the source and target data, as described in [\(Lipton et al.,](#page-7-0) [2018\)](#page-7-0).

# 3. Main Results

EBMaC incorporates multiple classifiers instead of treating a single one [\(Lipton et al.,](#page-7-0) [2018\)](#page-7-0), and adopts the empirical Bayes (EB) approach to estimate model parameters of Cg's and qg's. In addition, EBMaC employs a linear programming method to solve for confidence regions for the importance weights ω, rather than the classic Gaussian elimination method [\(Si et al.,](#page-7-3) [2023\)](#page-7-3). Through these innovations, EBMaC can directly produce the estimation and inference results simultaneously for C<sup>g</sup> <sup>∗</sup> and q<sup>g</sup> <sup>∗</sup> of a chosen classifier g ∗ , which facilitates inference for ω. Together, implementing linear programming in combination with multiple classifiers enables EBMaC to achieve tighter elementwise confidence intervals for ω. As an end result,

given the confidence intervals for C<sup>g</sup> <sup>∗</sup> and q<sup>g</sup> <sup>∗</sup> , EBMaC provides the smallest possible confidence region for ω.

#### 3.1. Estimation and Inference by Empirical Bayes

#### 3.1.1. BAYESIAN MODELING

Let {(xs, ys)} m <sup>s</sup>=1 be the source data and {xm+t} n <sup>t</sup>=1 be the target data. Let G = {g1, ..., gG} be a collection of G classifiers. Given a classifier g ∈ G, we apply it on the set {xs} m <sup>s</sup>=1, resulting in <sup>y</sup>b<sup>s</sup> <sup>=</sup> <sup>g</sup>(xs) for all s = 1, . . . , m. Let Mg,ij = P<sup>m</sup> <sup>s</sup>=1 **<sup>1</sup>**{yb<sup>s</sup> <sup>=</sup> i, y<sup>s</sup> <sup>=</sup> <sup>j</sup>}, then P<sup>K</sup> i=1 P<sup>K</sup> <sup>j</sup>=1 Mg,ij = m. For simplicity, we denote the vectorization of [Mg,ij ] by Mg, *i.e.* M<sup>g</sup> = (Mg,1, . . . , Mg,K<sup>2</sup> ) <sup>⊤</sup>. Similarly, we denote c<sup>g</sup> = vec(Cg), *i.e.* c<sup>g</sup> = (cg,1, . . . , cg,K<sup>2</sup> ) <sup>⊤</sup> ∈ ∆<sup>K</sup><sup>2</sup>−<sup>1</sup> , which is a (K<sup>2</sup> − 1)-dimensional probability simplex. We assume a hierarchical model

$$\mathbf{M}_g \mid \mathbf{c}_g \sim \text{Multinomial}(m, \mathbf{c}_g),$$

$$\mathbf{c}_g \sim \text{Dir}(\boldsymbol{\alpha}_s),$$

where Dir(αs) denotes the Dirichlet distribution, and α<sup>s</sup> = (αs,1, ..., αs,K<sup>2</sup> ) <sup>⊤</sup> is the concentration hyperparameter. Given αs, we assume that cg<sup>1</sup> , ..., cg<sup>G</sup> are independent. Additionally, given a classifier <sup>g</sup>, we write <sup>y</sup>bm+<sup>t</sup> <sup>=</sup> g(xm+t). Let Ng,k = P<sup>n</sup> <sup>t</sup>=1 **<sup>1</sup>**{ybm+<sup>t</sup> <sup>=</sup> <sup>k</sup>}. Note that P<sup>K</sup> <sup>k</sup>=1 Ng,k = n. We assume the hierarchical model for the target domain to be

$$N_g \mid \mathbf{q}_g \quad \sim \quad \text{Multinomial}(n, \mathbf{q}_g),$$

$$\mathbf{q}_g \quad \sim \quad \text{Dir}(\alpha_t).$$

Here, α<sup>t</sup> = (αt,1, ..., αt,K) <sup>⊤</sup> is the hyperparameter for the target data. Similarly, qg<sup>1</sup> , ..., qg<sup>G</sup> are assumed to be independent given αt.

## 3.1.2. ESTIMATION OF HYPERPARAMETERS

Because c<sup>g</sup> is latent, in Appendix [A.2,](#page-8-0) we derive the the marginal distribution of M<sup>g</sup> given α<sup>s</sup> to be

$$f(\mathbf{m}_g; \boldsymbol{\alpha}_s) = \frac{\Gamma(m_0)\Gamma(m+1)}{\Gamma(m_0+m)} \prod_{k=1}^{K^2} \frac{\Gamma(\alpha_{s,k} + m_{g,k})}{\Gamma(\alpha_{s,k})\Gamma(m_{g,k} + 1)},$$

where m<sup>0</sup> = P<sup>K</sup><sup>2</sup> <sup>j</sup>=1 αs,j and Γ(·) is the Gamma function. When we observe (mg<sup>1</sup> , ...,mg<sup>G</sup> ), the log-likelihood is

$$\begin{aligned} & \ell(\alpha_s; \mathbf{m}_{g_1}, \dots, \mathbf{m}_{g_G}) & (1) \\ = & \sum_{i=1}^G \log \left\{ \frac{\Gamma(m_0)\Gamma(m+1)}{\Gamma(m_0+m)} \prod_{k=1}^{K^2} \frac{\Gamma(\alpha_{s,k} + m_{g_i,k})}{\Gamma(\alpha_{s,k})\Gamma(m_{g_i,k} + 1)} \right\} \\ \propto & G\{\log \Gamma(m_0) - \log \Gamma(m_0 + m)\} \\ & + \sum_{i=1}^G \sum_{k=1}^{K^2} \{\log(\alpha_{s,k} + m_{g_i,k}) - \log \Gamma(\alpha_{s,k})\}. \end{aligned}$$

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

164

The partial derivative of with respect to αs,k is

$$\begin{aligned} & \frac{\partial \ell(\alpha_s; m_{g_1}, \dots, m_{g_G})}{\partial \alpha_{s,k}} \\ = & G\{\psi(m_0) - \psi(m_0 + m)\} \\ & + \sum_{i=1}^G \{\psi(\alpha_{s,k} + m_{g_i,k}) - \psi(\alpha_{s,k})\}, \end{aligned}$$

where ψ(x) is the digamma function, defined as ψ(x) = d log Γ(x)/dx. When K is small, we implement numerical optimization in Scipy to minimize the negative of the loglikelihood in [\(1\)](#page-1-0). If K is large, following [Minka](#page-7-4) [\(2000\)](#page-7-4), we find the maximum by fixed-point iteration. In the (t + 1)-th iteration, we set

$$\alpha_{s,k}^{(t+1)} = \alpha_{s,k}^{(t)} \frac{G^{-1} \sum_{i=1}^G \{\psi(\alpha_{s,k}^{(t)} + m_{g_i,k}) - \psi(\alpha_{s,k}^{(t)})\}}{\psi(m_0^{(t)} + m) - \psi(m_0^{(t)})},$$
for  $k = 1, \dots, K^2$ ,

and m (t+1) <sup>0</sup> = P <sup>k</sup> α (t+1) s,k . We use the moment matching estimation to get the initial α (0) <sup>s</sup> , with details in Appendix [A.1.](#page-8-1) Similar procedure is conducted to estimate α<sup>t</sup> based on the target model and data. The only differences are in changing K<sup>2</sup> to K, α<sup>s</sup> to αt, and mg<sup>i</sup> to ng<sup>i</sup> for i ∈ [G]. Let the estimators be <sup>α</sup>b<sup>s</sup> and <sup>α</sup>bt.

## 3.1.3. INFERENCE BASED ON THE POSTERIOR DISTRIBUTION

Given a new classifier g ∗ , we aim at estimating C<sup>g</sup> <sup>∗</sup> and q<sup>g</sup> ∗ , which are simplified as C<sup>∗</sup> and q ∗ . Because Dirichlet distribution is the conjugate prior of multinomial distribution, the posterior distributions of C<sup>∗</sup> and q ∗ are still the Dirichlet distributions with updated parameters <sup>α</sup>e<sup>s</sup> <sup>=</sup> <sup>α</sup>b<sup>s</sup> <sup>+</sup> <sup>m</sup><sup>g</sup> <sup>∗</sup> and <sup>α</sup>e<sup>t</sup> <sup>=</sup> <sup>α</sup>b<sup>t</sup> <sup>+</sup> <sup>n</sup><sup>g</sup> <sup>∗</sup> . Using the mode of posterior distributions, we estimate C<sup>∗</sup> and q <sup>∗</sup> by

$$\begin{aligned}\hat{\mathbf{C}} &= \max(\tilde{\mathbf{A}}_s - 1, 0)/(m_0 + m - K^2), \\ \hat{\mathbf{q}} &= \max(\tilde{\mathbf{\alpha}}_t - 1, 0)/(n_0 + n - K),\end{aligned}$$

where <sup>A</sup>e <sup>s</sup> is a <sup>K</sup> <sup>×</sup> <sup>K</sup> matrix reshaped from <sup>α</sup>es.

Because there is no closed form to build a confidence set for the Dirichlet distribution, we consider each component of C and q marginally. A nice feature is that the marginal distributions of Dirichlet distributions are Beta distributions. Specifically, the marginal posterior distributions are

$$C_{ij} \mid m_{g^*} \sim \text{Beta}(\tilde{A}_{s,ij}, m_0 + m - \tilde{A}_{s,ij}), \quad (2)$$

$$q_k^* \mid n_{g^*} \sim \text{Beta}(\tilde{\alpha}_{t,k}, n_0 + n - \tilde{\alpha}_{t,k}), \quad (3)$$

for i, j, k ∈ [K]. Note that Beta(a, b) has dramatically different shapes depending on a and b, hence we set the confidence intervals differently. For a > 1 and b > 1, it is unimodal, and we set the confidence interval from (δ/2) th to (1 − δ/2)-th quantile. For a ≤ 1 and b > 1, it is monotonically decreasing, and the confidence interval is chosen from 0 to (1 − δ)-th quantile. For a > 1 and b ≤ 1, it is monotonically increasing, and the confidence interval is set from δ-th quantile to 1. We exclude the case of a ≤ 1 and b ≤ 1, which cannot occur. We use [Cij , Cij ] and [q k , q<sup>k</sup> ] to denote the confidence intervals of level (1 − δ) for C ∗ ij and q ∗ k for i, j, k ∈ [K].

## 3.2. Estimation and Inference of Importance Weights

Following [Lipton et al.](#page-7-0) [\(2018\)](#page-7-0), we estimate the importance weights by

$$\hat{\omega} = \max(\hat{\mathbf{C}}^{-1}\hat{\mathbf{q}}, 0),$$

where <sup>C</sup>b <sup>−</sup><sup>1</sup> is the inverse of <sup>C</sup>b. However, we construct confidence sets of ω very differently from the literature [Si](#page-7-3) [et al.](#page-7-3) [\(2023\)](#page-7-3).

Let C = (Cij ), C = (Cij ), q = (q k ), and q = (q<sup>k</sup> ) be the collection of endpoints of confidence intervals for C ∗ ij and q ∗ k . For any matrices A and B of the same size, define A ≤ B if Aij ≤ Bij for all i, j, and similarly define A < B, A > B, and A ≥ B. Let Z ∈ [A, B] if and only if A ≤ Z ≤ B. Let C = [C, C] and Q = [q, q].

Given the relation C∗ω = q ∗ , it is readily obtained that ω = C∗−<sup>1</sup>q ∗ . Based on this explicit expression and the availability of C and Q, [Si et al.](#page-7-3) [\(2023\)](#page-7-3) constructed confidence interval for ω through computing C−<sup>1</sup>q using Gaussian elimination. However, during this process, many relaxations are implemented, which result in inflation of the confidence set. Rather than solving for the explicit solution of ω, we directly impose the linear constraints on ω, which leads to

$$\Omega = \{\omega : \exists \mathbf{C} \in \mathcal{C}, \mathbf{C}\omega \in \mathcal{Q}, \omega > 0\}.$$

Although the definition of Ω is clear, it is hard to implement because to verify ω ∈ Ω, we have to find the particular C such that the requirement is satisfied. An important discovery is the equivalence of Ω and {ω : Cω ≥ q, Cω ≤ q, ω > 0}. We present the result in Theorem [3.1](#page-2-0) and the proof in Appendix [B.1.](#page-9-0)

Theorem 3.1. {ω : ∃ C ∈ C, Cω ∈ Q, ω > 0} = {ω : Cω ≥ q, Cω ≤ q, ω > 0}*.*

Note that if we have used level 1 − δij for confidence interval [Cij , Cij ] and 1 − δ<sup>k</sup> for [q k , q<sup>k</sup> ], then the overall confidence level is 1 − ( P i,j∈[K] δij + P k∈[K] δk). Thus, if we want to reach a reasonable overall confidence level, then the individual confidence levels should be much higher. Although Theorem [3.1](#page-2-0) gives the clear description of the confidence set, it may have irregular shape. In practice,

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

218

people are often interested in more regular shapes such as hyperrectangle. For this purpose, we try to find the bounding hyperrectangle for Ω by solving 2K optimization problems

$$\min_{\boldsymbol{\omega}} \omega_k \text{ subject to } \boldsymbol{\omega} \in \Omega, \quad (4)$$

$$\max_{\omega} \omega_k \text{ subject to } \omega \in \Omega, \quad (5)$$

for k ∈ [K]. Thanks to Theorem 3.1, ω ∈ Ω contains 3K linear constraints, and ω<sup>k</sup> is also a linear function of ω, we can solve the optimization problems using linear programming, by simplex algorithm for example. We denote the resulting bounding hyperrectangle as ΩBH = Q<sup>K</sup> <sup>k</sup>=1[ω<sup>k</sup> , ωk], where ω<sup>k</sup> and ω<sup>k</sup> are the corresponding minimizers and maximizers. We denote the Gaussian elimination-based confidence set of [Si et al.](#page-7-3) [\(2023\)](#page-7-3) by ΩGE with detailed explanation in Appendix [C.](#page-10-0) Then Corollary [3.2](#page-3-0) holds, where the proof is in Appendix [B.2.](#page-9-1)

Corollary 3.2. *When* ΩGE *exists,* Ω ⊂ ΩBH ⊂ ΩGE*.*

## 3.3. Finite Sample Prediction

Benefiting from the confidence set of ω, we propose two ways of constructing a prediction set, conformal prediction [\(Vovk et al.,](#page-7-5) [2005\)](#page-7-5) and probably approximately correct (PAC) prediction [\(Valiant,](#page-7-6) [1984\)](#page-7-6). We provide finite sample guarantee of the prediction set, while in the literature only asymptotic properties can be achieved.

## 3.3.1. CONFORMAL PREDICTION

We consider the split conformal prediction setting, where the source data is divided into calibration set S<sup>1</sup> = {z<sup>i</sup> = (x<sup>i</sup> , yi) : i = 1, ..., m1} and training set S<sup>2</sup> = {(x<sup>i</sup> , yi) : i = m<sup>1</sup> + 1, ..., m}. We assume that the nonconformity score r(x, y) ∈ [0, 1] is trained in S<sup>2</sup> and that the prediction set is then derived from the calibration set S1. Let x<sup>0</sup> be a new covariate in the target domain with the potential label y0. Under label shift assumption, the calibration data set S<sup>1</sup> and z<sup>0</sup> = (x0, y0) satisfy the weighted exchangeability condition in [Tibshirani et al.](#page-7-7) [\(2019\)](#page-7-7), that is,

$$q(z_0, z_1, \dots, z_{m_1}) = p(z_0, z_1, \dots, z_{m_1}) \prod_{i=0}^{m_1} \omega_{y_i},$$

where p(z0, ..., z<sup>m</sup><sup>1</sup> ) = p(zσ(0), ..., zσ(m1)) for any permutation σ : {0, ..., m1} → {0, ..., m1}. Let r<sup>i</sup> = r(x<sup>i</sup> , yi) for i = 1, ..., m1. We can then create the level (1 − α) conformal prediction set FCP(x0; ω), denoted by

$$F_{\text{CP}}(\mathbf{x}_0; \boldsymbol{\omega}) = \{y_0 \in [K] : r(\mathbf{x}_0, y_0) \leq \tau_{\text{CP}}(y_0; \boldsymbol{\omega})\},$$

where τCP(y0; ω) is defined as

$$= Q_{1-\alpha} \left( \sum_{i=1}^{m_1} \delta_{r_i} \frac{\omega_{y_i}}{\sum_{j=1}^{m_1} \omega_{y_j} + \omega_{y_0}} + \delta_1 \frac{\omega_{y_0}}{\sum_{j=1}^{m_1} \omega_{y_j} + \omega_{y_0}} \right),$$

where δ<sup>r</sup> denotes the Dirac measure on r, and Q1−<sup>α</sup> denotes the (1 − α)-th sample quantile. When the true importance weight ω is known, FCP(x0; ω) has a 1 − α coverage rate by Theorem 2 of [Podkopaev & Ramdas](#page-7-8) [\(2021\)](#page-7-8). However, ω is unknown in practice. Given a potential confidence set Ω<sup>0</sup> of ω, we can construct a prediction set by computing

$$\tau_{\text{CP}}(y_0; \boldsymbol{\Omega}_0) = \sup_{\boldsymbol{\omega} \in \boldsymbol{\Omega}_0} \tau_{\text{CP}}(y_0; \boldsymbol{\omega}) \quad (6)$$

and

$$F_{\text{CP}}(\mathbf{x}_0; \boldsymbol{\Omega}_0) = \{y_0 \in [K] : r(\mathbf{x}_0, y_0) \leq \tau_{\text{CP}}(y_0; \boldsymbol{\Omega}_0)\}. \quad (7)$$

Compared to the known ω case, the construction in [\(7\)](#page-3-1) increases the confidence interval. However, we can still control the prediction level at 1 − δ − α, as established in Theorem [3.3.](#page-3-2) See Appendix [B.3](#page-9-2) for the proof.

Theorem 3.3. *If* **P**(ω ∈ Ω0) ≥ 1 − δ*, then*

$$\mathbb{P}_{(\mathbf{X}_0, Y_0) \sim Q}\{Y_0 \in F_{\text{CP}}(\mathbf{X}_0; \boldsymbol{\Omega}_0)\} \geq 1 - \delta - \alpha.$$

Note that Ω<sup>0</sup> can be obtained using the entire source data, while to guarantee the conformal prediction probability, we have to perform data splitting. Additionally, the advantage of ΩBH over ΩGE in Corollary [3.2](#page-3-0) is inherited in the conformal prediction set, in that at the same prediction level, the prediction set based on ΩBH is always smaller than that based on ΩGE. This property and the more general result are summarized in Theorem [3.4.](#page-3-3) The proof is provided in Appendix [B.4.](#page-9-3)

Theorem 3.4. *If* Ω<sup>1</sup> ⊂ Ω2*, then* FCP(x0; Ω1) ⊂ FCP(x0; Ω2) *for all* x0*. In particular,* FCP(x0; ΩBH) ⊂ FCP(x0; ΩGE)*.*

## 3.3.2. PAC PREDICTION

[Si et al.](#page-7-3) [\(2023\)](#page-7-3) constructed PAC prediction set which relies on confidence interval ΩGE. Similar to section 3.3.1, if we replace ΩGE by ΩBH, we can obtain a smaller prediction set with the same PAC guarantee [\(Park et al.,](#page-7-9) [2021\)](#page-7-9).

Theorem 3.5. *If* Ω<sup>1</sup> ⊂ Ω2*, then* FPAC(x0; Ω1) ⊂ FPAC(x0; Ω2) *for all* x0*. In particular,* FPAC(x0; ΩBH) ⊂ FPAC(x0; ΩGE)*.*

See Appendix [D](#page-10-1) for the construction of FPAC(x0; Ω0) and Appendix [B.5](#page-9-4) for the proof of Theorem 3.5.

# 4. Experiments

In this section, we implemented the EBMaC to evaluate its performance on the MNIST [\(LeCun et al.,](#page-7-10) [1998\)](#page-7-10), CIFAR-10, and CIFAR-100 data sets [\(Krizhevsky et al.,](#page-7-11) [2009\)](#page-7-11).

226

228

231

234

236

238

254

256

258

260

264

266

268

271

## 4.1. Classifier Training

For all data sets, we randomly selected 40,000 observations from training data set, combined with the 10,000 testing data to train classifiers, and used the remaining data to perform the analysis. For the MNIST data set, we trained 11 classifiers with different random seeds using the same architectures as in [Azizzadenesheli et al.](#page-5-0) [\(2019\)](#page-5-0). Each model was trained 10 epochs, and the best performer was retained. The final accuracy ranges from 97.25% to 98.07%. For the CIFAR data sets, we trained five classifiers using different architectures as shown in [Table 1.](#page-4-0) Each classifier was trained 200 epochs, and the best performer was retained. In implementing EBMaC, we used the classifier with the lowest testing accuracy as g ∗ , and the remaining classifiers as g1, ..., gG, where G = 10 for MNIST and G = 4 for CIFAR-10 and CIFAR-100.

Table 1. Trained classifiers for CIFAR data sets with different accuracy on corresponding testing data sets.

| Model          |   | CIFAR-10 | CIFAR-100 |
|----------------|---|----------|-----------|
| VGG16          | a | 92.38%   | 71.80%    |
| ResNet18       | b | 93.98%   | 75.47%    |
| MobileNetV2    | c | 93.77%   | 70.23%    |
| PreActResNet18 | d | 93.97%   | 60.15%    |
| RegNetX        | e | 93.93%   | –         |
| GoogLeNet      | f | –        | 75.78%    |

[Simonyan](#page-7-12) [\(2014\)](#page-7-12)

*<sup>b</sup>*[He et al.](#page-5-4) [\(2016a\)](#page-5-4)

*c* [Sandler et al.](#page-7-13) [\(2018\)](#page-7-13)

*<sup>d</sup>*[He et al.](#page-5-5) [\(2016b\)](#page-5-5)

*<sup>e</sup>*[Radosavovic et al.](#page-7-14) [\(2020\)](#page-7-14)

*f* [Szegedy et al.](#page-7-15) [\(2015\)](#page-7-15)

## 4.2. Experimental Design

To generate the data sets that satisfy the label shift assumption, we performed the following construction using Dirichlet shift. In generating the source data, we first generated a random vector v from Dir(α1K). For each k ∈ [K], we randomly draw mv<sup>k</sup> observations, from those with y = k. Here m is the source sample size, and α = 10, 000. We generated the target data in the same way, except that α = 10<sup>p</sup> with p = −3, −2, −1, 0, 1, 2, 3, and we did not retain the labels. We chose eight different m values, ranging from 1000 to 8000, with a step size of 1000. This resulted in 56 different data sets. In each data set, the sizes of the source data and target data are both m. Note that a smaller α leads to less balanced label distribution. This design is applied to MNIST, CIFAR-10 and CIFAR-100.

## 4.3. Performance of EBMaC on Importance Weights

We compared EBMaC to BBSE [\(Lipton et al.,](#page-7-0) [2018\)](#page-7-0), RLLS [\(Azizzadenesheli et al.,](#page-5-0) [2019\)](#page-5-0), and MLLS [\(Azizzadenesheli](#page-5-0) [et al.,](#page-5-0) [2019\)](#page-5-0), using MSE ∥<sup>ω</sup> − <sup>ω</sup>b∥ as a criterion. The implementation code for the existing methods was adapted from [Ye et al.](#page-7-16) [\(2024\)](#page-7-16) [1](#page-4-7) . For CIFAR-100, as shown in the first plot of [Figure 1,](#page-5-6) as the sample size and concentration parameter α increase, the MSE for EBMaC decreases, while the remaining three plots show that the MSE of EBMaC is generally smaller than other methods. The results for MNIST and CIFAR-10 are in Figures [5](#page-12-0) and [6](#page-13-0) in Appendix [E.](#page-11-0)

In our analysis, we found that in some classes in the target data, the variance of predicted label counts sometimes exceeds its mean, which violates the property of the multinomial distribution. However, the hierarchical modeling allows overdispersion by introducing additional hyperparameters, which extends the model flexibility. As shown in Tables [2,](#page-11-1) [3,](#page-12-1) and [4](#page-12-2) in Appendix [E,](#page-11-0) we observe that the CIFAR-100 data set has larger average variance-mean ratios compared to that of MNIST and CIFAR-10.

## 4.4. Performance of EBMaC on Confidence Sets

In obtaining confidence sets for CIFAR-100, we fix the same confidence level for Ω, ΩBH, and ΩGE, and present results in Figure [2.](#page-6-0) In the left panel, the x-axis represents the average length ratios of the GE method to the LP method, computed as K−<sup>1</sup> P<sup>K</sup> <sup>k</sup>=1(lk,GE/lk,BH), where lk,BH = ωk−ω<sup>k</sup> , and lk,GE is defined similarly. Here, GE was implemented using the code from [Si et al.](#page-7-3) [\(2023\)](#page-7-3) [2](#page-4-8) . Further, we perform a one-sided t-test to evaluate whether the log-ratio of the lengths log(lk,GE/lk,BH) is greater than zero across the labels. The resulting − log10(P-value) is shown in the y-axis. The horizontal line is at − log10(0.05), representing the statistical significance. In the right panel, we provide the bar plot of the ratio of the volume of Ω to ΩBH at each α value, presented in percentage. The results for CIFAR-10 and MNIST are similarly presented in Figures [3](#page-6-1) and [4.](#page-6-2)

In Figure [2,](#page-6-0) in the left panel, the vast majority is above the horizontal line, indicating that the improvement of the length ratio is significant in most cases. When comparing the results across three data sets, we find that EBMaC exhibits the best performance on CIFAR-100 in terms of both length ratio and P-value, but shows less improvement on MNIST. Note that all classifiers for MNIST have the best accuracy, while those for CIFAR-100 have the worst. This reflects that worse performance of classifiers generates more improvement of BH over GE. This is because worse classifiers lead to a confusion matrix that is less diagonal

https://github.com/ChangkunYe/MAPLS

<sup>2</sup> https://github.com/averysi224/pac-ps-label-shift

![](_page_5_Figure_1.jpeg)

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

Figure 1. Comparison of label shift estimation methods on CIFAR-100. The first contour plot displays the average MSE of different classifiers in log10 scale for all data sets. The second contour plot shows the log2 ratio of MSE from BBSE to that from EBMaC. The third and fourth contour plots are similar to the second one, but they present the comparison results of RLLS and MLLS to that of EBMaC, respectively.

dominant, which can be handled by BH without any issue but results in much inflation of the confidence set by GE.

From the right panel of Figure [2,](#page-6-0) we can see that in the settings when α is large, the difference between Ω and ΩBH can be very large, reflected in very small volume ratio. This indicates that aiming for a hyperrectangular shape can be costly. In such case, it might be wiser to use Ω to further perform conformal / PAC prediction. The performance in Figures [3](#page-6-1) and [4](#page-6-2) is slightly different in that the ratios of the volumes in the right panels are generally larger. This is because the shapes of Ω resemble more a hyperrectangle for CIFAR-10 and MNIST, due to a more diagonal dominant confusion matrix.

# 5. Discussion

The main innovations of EBMaC are in proposing an empirical Bayesian approach in hierarchical modeling for label shift problems (EB) and in handling the matrix constraints via linear programming (MaC). EB is able to handle overdispersion, while MaC achieves the tightest confidence sets for importance weights. These two components can work separately. For example, we can combine Clopper-Pearson interval (ClP) with MaC to obtain ClPMaC. We can also combine EB with GE to create EBGE. One obvious advantage of EB is in handling overdispersion, while the advantage of MaC is established theoretically. When the collection of classifiers performs poorly, EBMaC showcases the significant improvement of MaC over GE. The nice property of EBMaC naturally leads to a better outcome of downstream analysis, such as the prediction performance described in [3.3.](#page-3-4)

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

# References


[1] Alexandari, A., Kundaje, A., and Shrikumar, A. Maximum likelihood with bias-corrected calibration is hard-to-beat at label shift adaptation. In *International Conference on Machine Learning*, pp. 222–232. PMLR, 2020. Azizzadenesheli, K., Liu, A., Yang, F., and Anandkumar, A. Regularized learning for domain adaptation under label shifts. *arXiv preprint arXiv:1903.09734*, 2019. Garg, S., Wu, Y., Balakrishnan, S., and Lipton, Z. A unified view of label shift estimation. *Advances in Neural Information Processing Systems*, 33:3290–3300, 2020. Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. On calibration of modern neural networks. In *International conference on machine learning*, pp. 1321–1330. PMLR, 2017. He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016a. He, K., Zhang, X., Ren, S., and Sun, J. Identity mappings in deep residual networks. In *Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands,*

[2] 334

[3] 336

[4] 338

[5] 351

[6] 354

[7] 356

[8] 358

360 361

[10] 364

[11] 366

[12] 368

[13] 371

[14] 374

[15] 378

[16] ![](_page_6_Figure_1.jpeg)

[17] Figure 2. (CIFAR-100) Left panel: Comparison of ΩGE and ΩBH. Right panel: Bar plot of ratios volume(Ω)/ volume(ΩBH).

[18] ![](_page_6_Figure_3.jpeg)

[19] Figure 3. (CIFAR-10) Left panel: Comparison of ΩGE and ΩBH. Right panel: Bar plot of ratios volume(Ω)/ volume(ΩBH).

[20] ![](_page_6_Figure_5.jpeg)

[21] Figure 4. (MNIST) Left panel: Comparison of ΩGE and ΩBH. Right panel: Bar plot of ratios volume(Ω)/ volume(ΩBH).

385 386 387 388 389 390 394 396 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 *October 11–14, 2016, Proceedings, Part IV 14*, pp. 630–
  - 645. Springer, 2016b. Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images. 2009. LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradientbased learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, 1998. Lipton, Z., Wang, Y.-X., and Smola, A. Detecting and correcting for label shift with black box predictors. In *International conference on machine learning*, pp. 3122– 3130. PMLR, 2018. Minka, T. Estimating a dirichlet distribution, 2000. Park, S., Dobriban, E., Lee, I., and Bastani, O. PAC prediction sets under covariate shift. *arXiv preprint arXiv:2106.09848*, 2021. Podkopaev, A. and Ramdas, A. Distribution-free uncertainty quantification for classification under label shift. In *Uncertainty in artificial intelligence*, pp. 844–853. PMLR, 2021. Radosavovic, I., Kosaraju, R. P., Girshick, R., He, K., and Dollar, P. Designing network design spaces. In ´ *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 10428–10436, 2020. Saerens, M., Latinne, P., and Decaestecker, C. Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural computation*, 14(1):21–41, 2002. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and Chen, L.-C. Mobilenetv2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 4510–4520, 2018. Si, W., Park, S., Lee, I., Dobriban, E., and Bastani, O. PAC prediction sets under label shift. *arXiv preprint arXiv:2310.12964*, 2023. Simonyan, K. Very deep convolutional networks for largescale image recognition. *arXiv preprint arXiv:1409.1556*, 2014. Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., and Rabinovich,
  - A. Going deeper with convolutions. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 1–9, 2015. Tian, Q., Zhang, X., and Zhao, J. ELSA: Efficient label shift adaptation through the lens of semiparametric models. In *International Conference on Machine Learning*, pp. 34120–34142. PMLR, 2023. Tibshirani, R. J., Foygel Barber, R., Candes, E., and Ramdas,
    - A. Conformal prediction under covariate shift. *Advances in neural information processing systems*, 32, 2019. Valiant, L. G. A theory of the learnable. *Communications of the ACM*, 27(11):1134–1142, 1984. Vovk, V. Conditional validity of inductive conformal predictors. In *Asian conference on machine learning*, pp. 475–490. PMLR, 2012. Vovk, V., Gammerman, A., and Shafer, G. *Algorithmic learning in a random world*, volume 29. Springer, 2005. Ye, C., Tsuchida, R., Petersson, L., and Barnes, N. Label shift estimation for class-imbalance problem: A bayesian approach. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pp. 1073– 1082, 2024.
## A. Dirichlet-Multinomial model

#### A.1. Moment Matching Estimation for Dirichlet-Multinomial Model

We first give the moment matching estimation, which of course requires marginal statistics. Recall that we assume a common α<sup>s</sup> for all g ∈ G in Dirichlet prior. For any k in {1, ..., K<sup>2</sup>}, the marginal expectation and variance of mg,k are

$$\begin{aligned} E(M_{g,k}) &= E\{E(M_{g,k} | \mathbf{c}_g)\} = E(mc_{g,k}) = mE(c_{g,k}) = m\mu_k \\ Var(M_{g,k}) &= E\{Var(M_{g,k} | \mathbf{c}_g)\} + Var\{E(M_{g,k} | \mathbf{c}_g)\} \\ &= E\{mc_{g,k}(1 - c_{g,k})\} + m^2Var(c_{g,k}) \\ &= E(mc_{g,k}) - mE^2(c_{g,k}) - mVar(c_{g,k}) + m^2Var(c_{g,k}) \\ &= m\mu_k(1 - \mu_k)\frac{m_0 + m}{m_0 + 1}, \end{aligned}$$

where µ<sup>k</sup> = αs,k/m<sup>0</sup> and m<sup>0</sup> = P<sup>K</sup><sup>2</sup> <sup>j</sup>=1 αs,j . Next, matching them to the sample mean and the sample variance leads to following equations,

$$m\mu_k = G^{-1} \sum_{g=1}^G m_{g,k} \equiv \overline{m}_k \quad (8)$$

$$m\mu_k(1 - \mu_k) \frac{m_0 + m}{m_0 + 1} = (G - 1)^{-1} \sum_{g=1}^G (m_{g,k} - \overline{m}_k)^2 \equiv \hat{V}_k. \quad (9)$$

From [Equation 8,](#page-8-2) we know µ<sup>k</sup> = mk/m, Rearrange [Equation 9](#page-8-2) and replace µ<sup>k</sup> with mk/m, we have

$$\begin{aligned}\widehat{M}_0 &= \frac{m\overline{m}_k(m - \overline{m}_k) - m\widehat{V}_k}{m\widehat{V}_k - \overline{m}_k(m - \overline{m}_k)} \\ &= (m - 1) \left\{ \frac{m\widehat{V}_k}{\overline{m}_k(m - \overline{m}_k)} - 1 \right\}^{-1} - 1.\end{aligned}$$

Note that, we can have <sup>m</sup>c<sup>0</sup> for each class, so we can average them to have a final <sup>M</sup>c0. Then we substitute it into <sup>α</sup>s,k <sup>=</sup> <sup>m</sup>0mk/m to obtain <sup>α</sup>bs,k, for any <sup>k</sup> ∈ {1, ..., K<sup>2</sup>}.

## A.2. Marginal distribution for Dirichlet-Multinomial model

$$\begin{aligned}
f(\mathbf{m}_g; \boldsymbol{\alpha}_s) &= \int_{\mathbf{c}} f(\mathbf{m}_{g,k} | \mathbf{c}_g) f(\mathbf{c}_g; \boldsymbol{\alpha}_s) d\mathbf{c}_g \\
&= \int_{\mathbf{c}} \frac{\Gamma(m_0)}{\prod_{k=1}^{K^2} \Gamma(\alpha_{s,k})} \prod_{k=1}^{K^2} c_{g,k}^{\alpha_{s,k}-1} \cdot \frac{\Gamma(m+1)}{\prod_{k=1}^{K^2} \Gamma(m_{g,k}+1)} \prod_{k=1}^{K^2} c_{g,k}^{m_{g,k}} d\mathbf{c}_g \\
&= \frac{\Gamma(m_0)}{\prod_{k=1}^{K^2} \Gamma(\alpha_k)} \frac{\Gamma(m+1)}{\prod_{k=1}^{K^2} \Gamma(m_{g,k}+1)} \int_{\mathbf{c}} \prod_{k=1}^{K^2} c_{g,k}^{\alpha_{s,k}-1} \prod_{k=1}^{K^2} c_{g,k}^{m_{g,k}} d\mathbf{c}_g \\
&= \frac{\Gamma(m_0)}{\prod_{k=1}^{K^2} \Gamma(\alpha_{s,k})} \frac{\Gamma(m+1)}{\prod_{k=1}^{K^2} \Gamma(m_{g,k}+1)} \int_{\mathbf{c}} \prod_{k=1}^{K^2} c_{g,k}^{\alpha_{s,k}+m_{g,k}-1} d\mathbf{c}_g \\
&= \frac{\Gamma(m_0)}{\prod_{k=1}^{K^2} \Gamma(\alpha_k)} \frac{\Gamma(m+1)}{\prod_{k=1}^{K^2} \Gamma(m_{g,k}+1)} B(\alpha_{s,1} + m_{g,1}, \dots, \alpha_{s,K^2} + m_{g,K^2}) \\
&= \frac{\Gamma(m_0)\Gamma(m+1)}{\Gamma(m_0 + m)} \prod_{k=1}^{K^2} \frac{\Gamma(\alpha_{s,k} + m_{g,k})}{\Gamma(\alpha_{s,k})\Gamma(m_{g,k}+1)}
\end{aligned}$$

504

506

508 509

511

514 515 516

518

524

526

528

531

534

536

538

## B. Proofs

## B.1. Proof of Theorem [3.1](#page-2-0)

*Proof.* P Suppose that C ∈ C, q ∈ Q, and that ω satisfies Cω = q and ω > 0. The linear equation Cω = q is equivalent to K <sup>j</sup>=1 cijω<sup>j</sup> = q<sup>i</sup> . Since ω<sup>i</sup> > 0, we get that

$$\sum_{j=1}^K \bar{c}_{ij} \omega_j \geq \sum_{j=1}^K c_{ij} \omega_j = q_i \geq q_{\underline{i}},$$

$$\sum_{j=1}^K c_{ij} \omega_j \leq \sum_{j=1}^K c_{ij} \omega_j = q_i \leq \bar{q}_i,$$

which implies that ω ∈ Ω.

Now, we prove the other direction. Suppose that ω ∈ Ω. Then for each i ∈ [K], we can apply the following procedure. If P<sup>K</sup> <sup>j</sup>=1 cijω<sup>j</sup> ≤ q<sup>i</sup> , ∀i ∈ [K], take c ⊤ <sup>i</sup> = (ci1, ..., ciK) and q<sup>i</sup> = P<sup>K</sup> <sup>j</sup>=1 cijω<sup>j</sup> . Otherwise, for l ∈ [K], define

$$q_i(l) = \sum_{j=1}^l c_{ij} \omega_j + \sum_{j=l+1}^K \bar{c}_{ij} \omega_j.$$

Then <sup>q</sup>i(l) is a decreasing function of <sup>l</sup>, and <sup>q</sup>i(K) = P<sup>K</sup> <sup>j</sup>=1 cijω<sup>j</sup> ≤ q<sup>i</sup> by the condition. Thus, we can find l<sup>0</sup> such that qi(l<sup>0</sup> − 1) ≥ q<sup>i</sup> ≥ qi(l0). Let c ⊤ <sup>i</sup> = (ci<sup>1</sup> , ..., ci,l0−<sup>1</sup> , e<sup>c</sup>il<sup>0</sup> , ci,l0+1, ..., ciK) and q<sup>i</sup> = q<sup>i</sup> , where

$$\tilde{c}_{il_0} = \underline{c}_{il_0} + \frac{\bar{q}_i - q_i(l_0)}{\omega_{l_0}}.$$

Then e<sup>c</sup>il<sup>0</sup> <sup>∈</sup> [cil<sup>0</sup> , cil<sup>0</sup> ] and c ⊤ <sup>i</sup> ω = q<sup>i</sup> . Taking c ⊤ i as the i-th row of C and q<sup>i</sup> as the i-th element of q, we get that Cω = q, where C ∈ C and q ∈ Q.

## B.2. Proof of Corollary [3.2](#page-3-0)

*Proof.* The first part, Ω ⊂ ΩBH, is trivial from its definition. For the second part, note that Ω ⊂ ΩGE by Theorem [3.1.](#page-2-0) Since ΩBH is the smallest hyperrectangle that contains Ω, we get ΩBH ⊂ ΩGE.

## B.3. Proof of Theorem [3.3](#page-3-2)

*Proof.* By Theorem 2 of [Podkopaev & Ramdas](#page-7-8) [\(2021\)](#page-7-8), we have **P**(X0,Y0)∼Q{Y<sup>0</sup> ∈ FCP(X0; ω)} ≥ 1 − α. Also, if ω ∈ Ω0, we get FCP(X0; ω) ⊂ FCP(X0; Ω0) by Theorem [3.4.](#page-3-3) Then

$$\begin{aligned} & \mathbb{P}_{(\mathbf{X}_0, Y_0) \sim Q} \{Y_0 \notin F_{\text{CP}}(\mathbf{X}_0; \boldsymbol{\Omega}_0)\} \\ = & \mathbb{P}_{(\mathbf{X}_0, Y_0) \sim Q} \{\boldsymbol{\omega} \in \boldsymbol{\Omega}_0 \text{ and } Y_0 \notin F_{\text{CP}}(\mathbf{X}_0; \boldsymbol{\Omega}_0)\} + \mathbb{P}_{(\mathbf{X}_0, Y_0) \sim Q} \{\boldsymbol{\omega} \notin \boldsymbol{\Omega}_0 \text{ and } Y_0 \notin F_{\text{CP}}(\mathbf{X}_0; \boldsymbol{\Omega}_0)\} \\ \leq & \mathbb{P}_{(\mathbf{X}_0, Y_0) \sim Q} \{Y \notin F_{\text{CP}}(\mathbf{X}_0; \boldsymbol{\omega})\} + \mathbb{P}(\boldsymbol{\omega} \notin \boldsymbol{\Omega}_0) \\ \leq & \alpha + \delta. \end{aligned}$$

## B.4. Proof of Theorem [3.4](#page-3-3)

*Proof.* First, [\(6\)](#page-3-5) implies τCP(y; Ω1) ≤ τCP(y; Ω2) for all y. Then [\(7\)](#page-3-1) gives the result.

## B.5. Proof of Theorem [3.5](#page-3-6)

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

## C. Gaussian Elimination With Intervals

Given that C<sup>∗</sup> ∈ C = [C, C] and q <sup>∗</sup> ∈ Q = [q, q], [Si et al.](#page-7-3) [\(2023\)](#page-7-3) introduce an intuitive way, which they named Gaussian elimination with intervals, of finding Ω that contains ω = C∗−<sup>1</sup>q ∗ . Suppose that cij ≥ 0, q i > 0, and ω<sup>i</sup> > 0 for i, j ∈ [K]. They follow two phases of Gaussian elimination when solving a system of linear equations C∗ω = q ∗ and derive the elementwise interval for ω<sup>i</sup> . First, set c 0 ij = cij , c 0 ij = cij , q 0 i = q i , and q 0 <sup>i</sup> = q<sup>i</sup> . In the first phase (forward elimination), the elementary row operations are applied sequentially for k = 1, ..., K − 1 to delete the (i, k) element in the matrix for i > k by adding the multiple of the k-th row. Then the lower bound c k+1 ij and the upper bound c k+1 ij are derived from the interval [C k , C k ] at the k-th step as

$$\mathcal{C}_{ij}^{k+1} = \begin{cases} 0, & \text{if } i > k, j \leq k, \\ \frac{\mathcal{C}_{ij}^k}{\mathcal{C}_{ij}^k} - \frac{\bar{\mathcal{C}}_{ik}^k \bar{\mathcal{C}}_{kj}^k}{\bar{\mathcal{C}}_{kk}^k}, & \text{if } i, j > k, \\ \frac{\mathcal{C}_{ij}^k}{\mathcal{C}_{ij}^k}, & \text{otherwise.} \end{cases}$$

Simultaneously, q k+1 i and q k+1 i are obtained from the same row operations to be

$$q_{\underline{i}_i}^{k+1} = \begin{cases} \frac{q_i^k}{c_{ik}^k} - \frac{\bar{c}_{ik}^k \bar{q}_k^k}{c_{ik}^k}, & \text{if } i > k, \\ \frac{q_i^k}{c_{ik}^k}, & \text{otherwise.} \end{cases}$$

Then c ∗,k+1 ij and q ∗,k+1 i , which would have been obtained in the forward elimination step solving C∗ω = q ∗ , always lie in [c k+1 ij , c k+1 ij ] and [q k+1 i , q k+1 i ]. In the second phase (back substitution), they compute ω<sup>i</sup> and ω<sup>i</sup> , iteratively for i = K, ..., 1, replacing the truth with intervals as in the first phase.

$$\begin{aligned} \underline{s}_i &= \sum_{j=i+1}^K \underline{c}_{ij}^K \omega_j & \text{and} & \underline{s}_i &= \sum_{j=i+1}^K \underline{c}_{ij}^K \bar{\omega}_j, \\ \underline{\omega}_i &= \frac{\underline{q}_i - \bar{s}_i}{\underline{c}_{ii}^K} & \text{and} & \bar{\omega}_i &= \frac{\bar{q}_i - \underline{s}_i}{\underline{c}_{ii}^K}. \end{aligned}$$

Then <sup>Ω</sup>GE is defined as the <sup>K</sup>-dimensional hyperrectangle Q<sup>K</sup> <sup>i</sup>=1[ω<sup>i</sup> , ω<sup>i</sup> ]. [Si et al.](#page-7-3) [\(2023\)](#page-7-3) provide a theoretical result that their method yields ω ∈ ΩGE if c k ij ≥ 0, c k ii > 0, and q k i ≥ 0 for all i, j, k ∈ [K]. The basic assumption in order to satisfy the condition is that cik ≪ ckk. This is ensured when the classifier g(X) is accurate, that is, when the diagonal terms c ∗ kk in C<sup>∗</sup> dominate non-diagonal terms. If the assumption is violated, we may encounter a possibility that c ∗,k kk ≈ 0, which may lead to c k kk ≤ 0. Then in the forward elimination phase, c k+1 ij for all i, j > k will be −∞, which may make the algorithm impractical. Furthermore, if q i ≤ s<sup>i</sup> or c K ii ≤ 0 for some i, then the back substitution phase would lead to ω<sup>i</sup> ≤ 0 or ω<sup>i</sup> = ∞, which does not provide any information about the interval of ω<sup>i</sup> . In order to deal with the nonpositive bounds, they mention that choosing a wider margin, which would, however, make ΩGE larger than its optimal size.

# D. Details of PAC prediction

Let the calibration set be S<sup>1</sup> = {(x<sup>i</sup> , yi)} m<sup>1</sup> <sup>i</sup>=1 and denote by r(x, y) the nonconformity score trained separately. The PAC prediction set FPAC(x; ω, S1) under label shift [\(Vovk,](#page-7-17) [2012;](#page-7-17) [Park et al.,](#page-7-9) [2021;](#page-7-9) [Si et al.,](#page-7-3) [2023\)](#page-7-3) is defined by

$$\mathbb{P}_{S_1 \sim P^{m_1}} [\mathbb{P}_{(\mathbf{X}_0, Y_0) \sim Q} \{Y_0 \in F_{\text{PAC}}(\mathbf{X}_0; \boldsymbol{\omega}, S_1)\} \geq 1 - \epsilon] \geq 1 - \eta.$$

[Si et al.](#page-7-3) [\(2023\)](#page-7-3) constructed a set that satisfies a modification of PAC guarantee such that

$$\mathbb{P}_{S_1 \sim P^{m_1}, V[\mathbb{P}_{(\mathbf{X}_0, Y_0) \sim Q}\{Y_0 \in F_{\text{PAC}}(\mathbf{X}_0; \boldsymbol{\omega}, S_1, V, b)\}] \geq 1 - \epsilon] \geq 1 - \eta, \quad (10)$$

where V = (V1, ..., V<sup>m</sup><sup>1</sup> ) <sup>⊤</sup> ∼ Unif([0, 1])<sup>m</sup><sup>1</sup> and b = maxk∈[K] ωk. The set FPAC(x; ω, S1, V, b) is in the form of

$$F_{\text{PAC}}(\mathbf{x}; \boldsymbol{\omega}, S_1, V, b) = [y \in [K] : r(\mathbf{x}, y) \leq \tau_{\text{PAC}}\{T(\boldsymbol{\omega}, S_1, V, b)\}],$$

where T(ω, S1, V, b) = {(x<sup>i</sup> , yi) ∈ S<sup>1</sup> : V<sup>i</sup> ≤ ωy<sup>i</sup> /b} is a target sample generated by rejection-sampling from S1. Let m<sup>0</sup> = |T(ω, S1, V, b)|. Here, τPAC{T(ω, S1, V, b)} is chosen to satisfy

$$\begin{aligned} & \sum_{(\mathbf{x}_i, y_i) \in T(\boldsymbol{\omega}, S_1, V, b)} \mathbb{1}\{y_i \notin F_{\text{PAC}}(\mathbf{x}_i; \boldsymbol{\omega}, S_1, V, b)\} \\ &= \sum_{(\mathbf{x}_i, y_i) \in T(\boldsymbol{\omega}, S_1, V, b)} \mathbb{1}[r(\mathbf{x}_i, y_i) > \tau_{\text{PAC}}\{T(\boldsymbol{\omega}, S_1, V, b)\}] \leq k(m_0, \epsilon, \eta), \end{aligned}$$

that is, τPAC{T(ω, S1, V, b)} is the largest value that is less than the k(m0, ϵ, η)-th largest value of {r(x<sup>i</sup> , yi) : (x<sup>i</sup> , yi) ∈ T(ω, S1, V, b)}, where

$$k(m_0, \epsilon, \eta) = \max\{k : F_{\text{Binom}(m_0, \epsilon)}(k) \leq \eta\}.$$

Note that FBinom(n,ϵ)(·) is the CDF of Binom(n, ϵ). If the true importance weight ω is used, then the modified PAC condition [\(10\)](#page-11-4) is satisfied. When the confidence set Ω<sup>0</sup> with **P**(ω ∈ Ω) ≥ 1 − δ is provided, we can define

$$\tau_{\text{PAC}}\{T(\mathbf{\Omega}, S_1, V, b)\} = \sup_{\omega \in \mathbf{\Omega}} \tau_{\text{PAC}}\{T(\omega, S_1, V, b)\} \quad (11)$$

and

$$F_{\text{PAC}}(\mathbf{x}; \boldsymbol{\Omega}, S_1, V, b) = [y \in [K] : r(\mathbf{x}, y) \leq \tau_{\text{PAC}}\{T(\boldsymbol{\Omega}, S_1, V, b)\}]. \quad (12)$$

Then FPAC(x; Ω, S1, V, b) satisfies the modified PAC condition [\(10\)](#page-11-4) with η being η + δ.

Theorem D.1. *Suppose that* **P**(ω ∈ Ω0) ≥ 1 − δ*. Then*

$$\mathbb{P}_{S_1 \sim P^{m_1}, V}[\mathbb{P}_{(\mathbf{X}_0, Y_0) \sim \mathcal{Q}}\{Y_0 \in F_{\text{PAC}}(\mathbf{X}_0; \boldsymbol{\Omega}, S_1, V, b)\} \geq 1 - \epsilon] \geq 1 - \eta - \delta.$$

*Proof.* The proof follows from Theorem 3 of [Park et al.](#page-7-9) [\(2021\)](#page-7-9).

# E. Data Dispersion

Table 2. (MNIST) Average variance-mean ratios for all classes under different sample size and Dirichlet shift combinations.

|                   |      |      | log 10 | ( α ) |      |      |      |
|-------------------|------|------|--------|-------|------|------|------|
| sample size ( m ) | -3   | -2   | -1     | 0     | 1    | 2    | 3    |
| 8000              | 3.62 | 3.17 | 3.10   | 0.43  | 0.32 | 0.31 | 0.32 |
| 7000              | 6.05 | 8.06 | 5.78   | 0.58  | 0.30 | 0.25 | 0.29 |
| 6000              | 8.26 | 6.38 | 3.57   | 0.77  | 0.28 | 0.25 | 0.27 |
| 5000              | 3.77 | 4.73 | 1.44   | 0.38  | 0.22 | 0.23 | 0.26 |
| 4000              | 3.08 | 3.82 | 2.67   | 0.27  | 0.21 | 0.18 | 0.15 |
| 3000              | 4.97 | 3.07 | 1.07   | 0.33  | 0.15 | 0.13 | 0.15 |
| 2000              | 2.69 | 2.08 | 0.94   | 0.42  | 0.14 | 0.12 | 0.11 |
| 1000              | 1.04 | 1.15 | 0.79   | 0.09  | 0.08 | 0.09 | 0.07 |

Table 3. (CIFAR-10) Average variance-mean ratios for all classes under different sample size and Dirichlet shift combinations.

|                   |      |      | log 10 | ( α ) |      |      |      |
|-------------------|------|------|--------|-------|------|------|------|
| sample size ( m ) | -3   | -2   | -1     | 0     | 1    | 2    | 3    |
| 8000              | 3.55 | 2.73 | 1.58   | 0.61  | 0.23 | 0.31 | 0.36 |
| 7000              | 5.70 | 5.72 | 2.33   | 0.63  | 0.23 | 0.18 | 0.17 |
| 6000              | 6.10 | 4.45 | 1.44   | 0.28  | 0.35 | 0.23 | 0.18 |
| 5000              | 3.84 | 3.58 | 0.77   | 0.22  | 0.23 | 0.14 | 0.20 |
| 4000              | 4.15 | 4.02 | 2.43   | 0.27  | 0.15 | 0.28 | 0.26 |
| 3000              | 2.74 | 3.92 | 0.79   | 0.27  | 0.23 | 0.13 | 0.14 |
| 2000              | 1.19 | 2.19 | 0.70   | 0.22  | 0.14 | 0.19 | 0.15 |
| 1000              | 1.06 | 0.85 | 0.83   | 0.16  | 0.16 | 0.21 | 0.13 |

Table 4. (CIFAR100) Average variance-mean ratios for all classes under different sample size and Dirichlet shift combinations.

|                   |       |       | log 10 | ( α ) |      |      |      |
|-------------------|-------|-------|--------|-------|------|------|------|
| sample size ( m ) | -3    | -2    | -1     | 0     | 1    | 2    | 3    |
| 8000              | 25.95 | 12.42 | 3.70   | 1.36  | 0.95 | 0.86 | 0.90 |
| 7000              | 12.82 | 10.68 | 4.31   | 1.14  | 0.81 | 0.78 | 0.82 |
| 6000              | 11.02 | 12.91 | 3.68   | 1.18  | 0.75 | 0.77 | 0.74 |
| 5000              | 5.09  | 16.20 | 2.59   | 1.00  | 0.70 | 0.75 | 0.64 |
| 4000              | 10.08 | 8.03  | 3.07   | 0.93  | 0.77 | 0.57 | 0.65 |
| 3000              | 6.14  | 7.64  | 3.33   | 0.98  | 0.64 | 0.55 | 0.54 |
| 2000              | 2.93  | 2.96  | 2.02   | 0.71  | 0.53 | 0.53 | 0.55 |
| 1000              | 1.15  | 1.68  | 0.87   | 0.48  | 0.46 | 0.55 | 0.46 |

![](_page_12_Figure_6.jpeg)

Figure 5. Comparison of label shift estimation methods on MNIST. The first contour plot displays the average MSE of different classifiers in log10 scale for all data sets. The second contour plot shows the log2 ratio of MSE from BBSE to that from EBMaC. The third and fourth contour plots are similar to the second one, but they present the comparison results of RLLS and MLLS to that of EBMaC, respectively.

![](_page_13_Figure_3.jpeg)

Figure 6. Comparison of label shift estimation methods on CIFAR-10. The first contour plot displays the average MSE of different classifiers in log10 scale for all data sets. The second contour plot shows the log2 ratio of MSE from BBSE to that from EBMaC. The third and fourth contour plots are similar to the second one, but they present the comparison results of RLLS and MLLS to that of EBMaC, respectively.