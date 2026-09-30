# Ebmac: Empirical Bayes And Matrix Constraints For Label Shift

## Anonymous Authors1

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054

## Abstract

We estimate the importance weights and their associated confidence set in label shift problems using hierarchical models via the Empirical Bayes and Matrix Constraints (EBMaC) method. Our approach accommodates dispersion beyond what is permitted by the classic multinomial model and produces exact confidence regions in finite samples for confusion matrix and predicted labels. In addition, we describe the dependence structure of the importance weights in matrix constraints. Through a linear programming technique, we are able to compute smaller confidence sets and shorter elementwise confidence intervals for importance weights compared to existing methods, while maintaining the probability guarantee. Applying the results to prediction in the target domain directly yields smaller conformal prediction set and PAC prediction set. Numerical experiments demonstrate the advantages of EBMaC in producing tighter confidence sets for the importance weights both marginally and jointly.

## 1. Introduction

When we simultaneously consider data sets from different sources, problems of distribution shift naturally arise. The most frequently studied distribution shifts are covariate shift and label shift. Here, we focus on label shift, which describes the scenario where the marginal distributions of the labels differ in the source and the target domains, but given the label, the conditional distributions of covariates remain unchanged. The key quantity of interest is importance weights, *i.e.* the ratios of the label proportions between the two domains. Given a classifier, there are three types of approaches for estimating the importance weights. The first one mainly relies on the linear relationship of the confusion matrix and 1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

1 the predicted label distribution (Lipton et al., 2018; Azizzadenesheli et al., 2019), and is named the confusion matrix method. The classifier is used to produce the confusion matrix in the source domain and to generate the predicted label distribution in the target domain. In forming the confusion matrix, either hard assignments or soft assignments can be implemented (Garg et al., 2020). The difference between BBSE (Lipton et al., 2018) and RLLS (Azizzadenesheli et al., 2019) is that BBSE pioneered the method while RLLS refined it by adding a regularization term on the importance weights to address potential near-singularity issues in the confusion matrix. The second one estimates the importance weights by maximum likelihood estimator (MLE). To this end, Saerens et al. (2002) proposed MLLS which finds the MLE by EM algorithm. Alexandari et al. (2020) proposed BCTS and demonstrated that further calibrating a classifier on the source domain significantly improves the MLE. The improvement happens because a classifier trained on the source domain may not perfectly represent the true proportions of the labels, even if it achieves high prediction probabilities (Guo et al., 2017). Such miscalibration biases the label predicting probability in the source domain and thus the estimated importance weights. The last one solves an estimating equation, formed by the projected score function, and is named ELSA (Tian et al., 2023). ELSA
has the feature of being robust to an uncalibrated classifier, and it outperforms BCTS in computational efficiency while maintaining competitive accuracy. In terms of the confidence intervals of the importance weights, most results hold only in the asymptotic sense. In finite samples, BBSE and RLLS rely on expressing the estimators explicitly in terms of confusion matrix and predicted label distribution. On the other hand, Si et al. (2023) proposed the Gaussian elimination (GE) method, where they modified each step of the Gaussian elimination procedure when solving the linear system in the confusion matrix method. Nevertheless, these methods do not produce tight confidence sets. We propose EBMaC (Empirical Bayes and Matrix Constraints) method in the confusion matrix method class. We first construct confidence regions for the confusion matrix and the predicted label distributions using empirical Bayes method in a hierarchical model. It incorporates the overdispersion phenomenon, which is often encountered in practice. We further take into account by recognizing it as a linear programming problem. This allows us to bypass matrix inversion and to obtain the tightest confidence sets for the importance weights. Furthermore, we demonstrate that applying the resulting confidence set yields the smallest finite sample prediction sets in the target domain. The superiority of EBMaC is rigorously proven in theory and illustrated through extensive numerical experiments.

## 2. Problem Setup

055 056 057 058 059 060 061 062 063

## 064

065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 Let x ∈ X = Rdand y ∈ Y = {1, . . . , K} ≡ [K] ⊂ N+
denote the feature and the label, respectively. Let P and Q represent the source and target distributions, respectively. Then, let p(·) denote the pmf/pdf for the source domain and q(·) for the target domain. The letters in the () reflect corresponding random variables. For example, p(y) is the marginal pmf of the labels in the source domain. The label shift setting assumes that p(x | y) = q(x | y) while p(y) ̸=
q(y) in general. Suppose we have a classifier g, defined as g : *X → Y*. Then the K × K confusion matrix Cg is defined by (Cg)ij ≡ P(X,Y )∼P {g(X) = *i, Y* = j}, for i, j ∈ [K], where P(X,Y )∼P stands for probability in source domain. In the target domain, we define the predicted label proportions, qg = (q1*, ..., q*K)
⊤. For each k ∈ [K], qk =
PX∼QX {g(X) = k}, where QX is the distribution of X in the target domain. We also use P to denote the probability in the combined domain. Let ω = (ω1*, ..., ω*K)
⊤ be the importance weights, where ωy = q(y)/p(y) for y ∈ [K].

We aim at the estimation and inference for ω. Remark 2.1. Under label shift, for any classifier g such that yb = g(x), we have p(yb| y) = q(yb| y). It is clear that Py∈Y p(*y, y* b )ωy = q(yb). When the confusion matrix Cg is invertible, solving the linear system Cgω = qg is a valid method for estimating the importance weights ω. In practice, since the true values of Cg and qg are unknown, we estimate them by the sample versions using the source and target data, as described in (Lipton et al., 2018).

## 3. Main Results

EBMaC incorporates multiple classifiers instead of treating a single one (Lipton et al., 2018), and adopts the empirical Bayes (EB) approach to estimate model parameters of Cg's and qg's. In addition, EBMaC employs a linear programming method to solve for confidence regions for the importance weights ω, rather than the classic Gaussian elimination method (Si et al., 2023). Through these innovations, EBMaC can directly produce the estimation and inference results simultaneously for Cg
∗ and qg
∗ of a chosen classifier g
∗, which facilitates inference for ω. Together, implementing linear programming in combination with multiple classifiers enables EBMaC to achieve tighter elementwise confidence intervals for ω. As an end result, given the confidence intervals for Cg
∗ and qg
∗ , EBMaC
provides the smallest possible confidence region for ω.

## 3.1. Estimation And Inference By Empirical Bayes 3.1.1. Bayesian Modeling

Let {(xs, ys)}
m s=1 be the source data and {xm+t}
n t=1 be the target data. Let G = {g1*, ..., g*G} be a collection of G classifiers. Given a classifier g ∈ G, we apply it on the set {xs}
m s=1, resulting in ybs = g(xs) for all s = 1*, . . . , m*. Let M*g,ij* =Pm s=1 1{ybs = *i, y*s = j},
then PK
i=1 PK
j=1 M*g,ij* = m. For simplicity, we denote the vectorization of [Mg,ij ] by Mg, *i.e.* Mg =
(Mg,1, . . . , Mg,K2 )
⊤. Similarly, we denote cg = vec(Cg),
i.e. cg = (cg,1, . . . , cg,K2 )
⊤ ∈ ∆K2−1, which is a
(K2 − 1)-dimensional probability simplex. We assume a hierarchical model

$$\begin{array}{r c l}{{\mathbf{M}_{g}\,|\,\mathbf{c}_{g}}}&{{\sim}}&{{\mathrm{Multinomial}(m,\mathbf{c}_{g}),}}\\ {{}}&{{\mathbf{c}_{g}}}&{{\sim}}&{{\mathrm{Dir}(\mathbf{\alpha}_{s}),}}\end{array}$$

where Dir(αs) denotes the Dirichlet distribution, and αs = (αs,1, ..., αs,K2 )
⊤ is the concentration hyperparameter. Given αs, we assume that cg1
, ..., cgG are independent.

Additionally, given a classifier g, we write ybm+t =
g(xm+t). Let Ng,k =Pn t=1 1{ybm+t = k}. Note that PK
k=1 Ng,k = n. We assume the hierarchical model for the target domain to be

$$\begin{array}{r c l}{{\mathbf{N}_{g}\,|\,\mathbf{q}_{g}}}&{{\sim}}&{{\mathrm{Multinomial}(n,\mathbf{q}_{g}),}}\\ {{}}&{{}}&{{\mathbf{q}_{g}}}&{{\sim}}&{{\mathrm{Dir}(\mathbf{\alpha}_{t}).}}\end{array}$$

Here, αt = (αt,1, ..., αt,K)
⊤ is the hyperparameter for the target data. Similarly, qg1
, ..., qgG are assumed to be independent given αt.

## 3.1.2. Estimation Of Hyperparameters

Because cg is latent, in Appendix A.2, we derive the the marginal distribution of Mg given αs to be

$$f(\mathbf{m}_{g};\mathbf{\alpha}_{s})={\frac{\Gamma(m_{0})\Gamma(m+1)}{\Gamma(m_{0}+m)}}\prod_{k=1}^{K^{2}}{\frac{\Gamma(\alpha_{s,k}+m_{g,k})}{\Gamma(\alpha_{s,k})\Gamma(m_{g,k}+1)}},$$

where m0 =PK2 j=1 αs,j and Γ(·) is the Gamma function.

When we observe (mg1*, ...,*mgG ), the log-likelihood is

$$\ell(\mathbf{\alpha}_{s};\mathbf{m}_{g_{1}},...,\mathbf{m}_{g_{G}})\tag{1}$$ $$=\sum_{i=1}^{G}\log\left\{\frac{\Gamma(m_{0})\Gamma(m+1)}{\Gamma(m_{0}+m)}\prod_{k=1}^{K^{2}}\frac{\Gamma(\alpha_{s,k}+m_{g_{i},k})}{\Gamma(\alpha_{s,k})\Gamma(m_{g_{i},k}+1)}\right\}$$ $$\propto G\{\log\Gamma(m_{0})-\log\Gamma(m_{0}+m)\}$$ $$+\sum_{i=1}^{G}\sum_{k=1}^{K^{2}}\{\log(\alpha_{s,k}+m_{g_{i},k})-\log\Gamma(\alpha_{s,k})\}.$$

The partial derivative of with respect to αs,k is where ψ(x) is the digamma function, defined as ψ(x) =
d log Γ(x)/dx. When K is small, we implement numerical optimization in Scipy to minimize the negative of the loglikelihood in (1). If K is large, following Minka (2000), we find the maximum by fixed-point iteration. In the (t + 1)-th iteration, we set

## 3.1.3. Inference Based On The Posterior Distribution

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 Given a new classifier g
∗, we aim at estimating Cg
∗ and qg
∗ ,
which are simplified as C∗and q
∗. Because Dirichlet distribution is the conjugate prior of multinomial distribution, the posterior distributions of C∗and q
∗are still the Dirichlet distributions with updated parameters αes = αbs + mg
∗ and αet = αbt + ng
∗ . Using the mode of posterior distributions, we estimate C∗and q
∗ by and m
(t+1)
0 =Pk α
(t+1)
s,k . We use the moment matching estimation to get the initial α
(0)
s , with details in Appendix A.1. Similar procedure is conducted to estimate αt based on the target model and data. The only differences are in changing K2to K, αs to αt, and mgito ngifor i ∈ [G].

Let the estimators be αbs and αbt.

$$\alpha_{s,k}^{(t+1)}=\alpha_{s,k}^{(t)}\frac{G^{-1}\sum_{i=1}^{G}\{\psi(\alpha_{s,k}^{(t)}+m_{g_{i},k})-\psi(\alpha_{s,k}^{(t)})\}}{\psi(m_{0}^{(t)}+m)-\psi(m_{0}^{(t)})},$$ $$\mathrm{for}\;k=1,...,K^{2},$$
$$\begin{array}{r c l}{{\widehat{\mathbf{C}}}}&{{=}}&{{\operatorname*{max}(\widehat{\mathbf{A}}_{s}-1,0)/(m_{0}+m-K^{2}),}}\\ {{\widehat{\mathbf{q}}}}&{{=}}&{{\operatorname*{max}(\widehat{\boldsymbol{\alpha}}_{t}-1,0)/(n_{0}+n-K),}}\end{array}$$

where Ae s is a K × K matrix reshaped from αes.

Because there is no closed form to build a confidence set for the Dirichlet distribution, we consider each component of C and q marginally. A nice feature is that the marginal distributions of Dirichlet distributions are Beta distributions. Specifically, the marginal posterior distributions are

$$\begin{array}{r c l}{{C_{i j}^{*}\mid m_{g^{*}}}}&{{\sim}}&{{\mathrm{Beta}(\tilde{A}_{s,i j},\,m_{0}+m-\tilde{A}_{s,i j}),}}\\ {{}}&{{q_{k}^{*}\mid n_{g^{*}}}}&{{\sim}}&{{\mathrm{Beta}(\tilde{\alpha}_{t,k},\,n_{0}+n-\tilde{\alpha}_{t,k}),}}\end{array}\tag{2}$$

for *i, j, k* ∈ [K]. Note that Beta(a, b) has dramatically different shapes depending on a and b, hence we set the confidence intervals differently. For a > 1 and b > 1, it is unimodal, and we set the confidence interval from (δ/2)- th to (1 − δ/2)-th quantile. For a ≤ 1 and b > 1, it is monotonically decreasing, and the confidence interval is chosen from 0 to (1 − δ)-th quantile. For a > 1 and b ≤ 1, it is monotonically increasing, and the confidence interval is set from δ-th quantile to 1. We exclude the case of a ≤ 1 and b ≤ 1, which cannot occur. We use [Cij , Cij ] and
[qk
, qk] to denote the confidence intervals of level (1 − δ)
for C
∗
ij and q
∗
kfor *i, j, k* ∈ [K].

## 3.2. Estimation And Inference Of Importance Weights

Following Lipton et al. (2018), we estimate the importance weights by

$${\hat{\boldsymbol{\omega}}}=\operatorname*{max}({\hat{\mathbf{C}}}^{-1}{\hat{\mathbf{q}}},0),$$

where Cb −1is the inverse of Cb. However, we construct confidence sets of ω very differently from the literature Si et al. (2023).

Let C = (Cij ), C = (Cij ), q = (qk
), and q = (qk)
be the collection of endpoints of confidence intervals for C
∗
ij and q
∗
k. For any matrices A and B of the same size, define A ≤ B if Aij ≤ Bij for all *i, j*, and similarly define A < B, A > B, and A ≥ B. Let Z ∈ [A, B] if and only if A ≤ Z ≤ B. Let C = [C, C] and Q = [q, q].

Given the relation C∗ω = q
∗, it is readily obtained that ω = C∗−1q
∗. Based on this explicit expression and the availability of C and Q, Si et al. (2023) constructed confidence interval for ω through computing C−1q using Gaussian elimination. However, during this process, many relaxations are implemented, which result in inflation of the confidence set. Rather than solving for the explicit solution of ω, we directly impose the linear constraints on ω, which leads to

$$\Omega=\{\omega:\exists\,\mathbf{C}\in{\mathcal{C}},\mathbf{C}\omega\in{\mathcal{Q}},\,\omega>0\}.$$

Although the definition of Ω is clear, it is hard to implement because to verify ω ∈ Ω, we have to find the particular C such that the requirement is satisfied. An important discovery is the equivalence of Ω and {ω : Cω ≥ q, Cω ≤
q, ω > 0}. We present the result in Theorem 3.1 and the proof in Appendix B.1. Theorem 3.1. {ω : ∃ C ∈ C, Cω ∈ Q, ω > 0} = {ω : Cω ≥ q, Cω ≤ q, ω > 0}.

Note that if we have used level 1 − δij for confidence interval [Cij , Cij ] and 1 − δk for [qk
, qk
], then the overall confidence level is 1 − (Pi,j∈[K]
δij +Pk∈[K]
δk). Thus, if we want to reach a reasonable overall confidence level, then the individual confidence levels should be much higher.

Although Theorem 3.1 gives the clear description of the confidence set, it may have irregular shape. In practice,

$$\frac{\partial\ell(\mathbf{\alpha}_{s};\mathbf{m}_{g_{1}},...,\mathbf{m}_{g_{C}})}{\partial\alpha_{s,k}}$$
$\overline{\phantom{\rule{0.3em}{0ex}}}$
$$G\{\psi(m_{0})-\psi(m_{0}+m)\}$$
$\psi(\alpha_{s,k}+m_{g_{i},k})-\psi(\alpha_{s,k})\}$, $\psi(\alpha_{s,k}+m_{g_{i},k})-\psi(\alpha_{s,k})\}$.  
people are often interested in more regular shapes such as hyperrectangle. For this purpose, we try to find the bounding hyperrectangle for Ω by solving 2K optimization problems

$$\operatorname*{min}_{\omega}\omega_{k}\;\;\mathrm{subject~to}\;\;\omega\in\Omega,$$ $$\operatorname*{max}_{\omega}\omega_{k}\;\;\mathrm{subject~to}\;\;\omega\in\Omega,$$
ωk subject to ω ∈ Ω, (4)
ωωk subject to ω ∈ Ω, (5)
for k ∈ [K]. Thanks to Theorem 3.1, ω ∈ Ω contains 3K
linear constraints, and ωk is also a linear function of ω, we can solve the optimization problems using linear programming, by simplex algorithm for example. We denote the resulting bounding hyperrectangle as ΩBH =QK
k=1[ωk
, ωk],
where ωkand ωk are the corresponding minimizers and maximizers. We denote the Gaussian elimination-based confidence set of Si et al. (2023) by ΩGE with detailed explanation in Appendix C. Then Corollary 3.2 holds, where the proof is in Appendix B.2.

Corollary 3.2. When ΩGE *exists,* Ω ⊂ ΩBH ⊂ ΩGE.

## 3.3. Finite Sample Prediction

Benefiting from the confidence set of ω, we propose two ways of constructing a prediction set, conformal prediction (Vovk et al., 2005) and probably approximately correct (PAC) prediction (Valiant, 1984). We provide finite sample guarantee of the prediction set, while in the literature only asymptotic properties can be achieved.

## 3.3.1. Conformal Prediction

We consider the split conformal prediction setting, where the source data is divided into calibration set S1 = {zi =
(xi, yi) : i = 1*, ..., m*1} and training set S2 = {(xi, yi) :
i = m1 + 1*, ..., m*}. We assume that the nonconformity score r(x, y) ∈ [0, 1] is trained in S2 and that the prediction set is then derived from the calibration set S1. Let x0 be a new covariate in the target domain with the potential label y0. Under label shift assumption, the calibration data set S1 and z0 = (x0, y0) satisfy the weighted exchangeability condition in Tibshirani et al. (2019), that is,

$$q(z_{0},z_{1},...,z_{m_{1}})=p(z_{0},z_{1}...,z_{m_{1}})\prod_{i=0}^{m_{1}}\omega_{y_{i}},$$

where p(z0*, ..., z*m1
) = p(zσ(0)*, ..., z*σ(m1)) for any permutation σ : {0, ..., m1} → {0*, ..., m*1}. Let ri = r(xi, yi) for i = 1*, ..., m*1. We can then create the level (1 − α) conformal prediction set FCP(x0; ω), denoted by FCP(x0; ω) = {y0 ∈ [K] : r(x0, y0) ≤ τCP(y0; ω)}, where τCP(y0; ω) is defined as

$$Q_{1-\alpha}(\sum_{i=1}^{m_{1}}\delta_{r_{i}}\frac{\omega_{y_{i}}}{\sum_{j=1}^{m_{1}}\omega_{y_{j}}+\omega_{y_{0}}}+\delta_{1}\frac{\omega_{y_{0}}}{\sum_{j=1}^{m_{1}}\omega_{y_{j}}+\omega_{y_{0}}})$$

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 τCP(y0; ω)
Compared to the known ω case, the construction in (7) increases the confidence interval. However, we can still control the prediction level at 1 − δ − α, as established in Theorem 3.3. See Appendix B.3 for the proof.

Theorem 3.3. If P(ω ∈ Ω0) ≥ 1 − δ*, then*

$$\mathbb{P}_{(\mathbf{X}_{0},Y_{0})\sim Q}\{Y_{0}\in F_{\mathrm{CP}}(\mathbf{X}_{0};\Omega_{0})\}\geq1-\delta-\alpha.$$

),
where δr denotes the Dirac measure on r, and Q1−α denotes the (1 − α)-th sample quantile. When the true importance weight ω is known, FCP(x0; ω) has a 1 − α coverage rate by Theorem 2 of Podkopaev & Ramdas (2021). However, ω is unknown in practice. Given a potential confidence set Ω0 of ω, we can construct a prediction set by computing Note that Ω0 can be obtained using the entire source data, while to guarantee the conformal prediction probability, we have to perform data splitting. Additionally, the advantage of ΩBH over ΩGE in Corollary 3.2 is inherited in the conformal prediction set, in that at the same prediction level, the prediction set based on ΩBH is always smaller than that based on ΩGE. This property and the more general result are summarized in Theorem 3.4. The proof is provided in Appendix B.4. Theorem 3.4. If Ω1 ⊂ Ω2*, then* FCP(x0; Ω1) ⊂
FCP(x0; Ω2) for all x0*. In particular,* FCP(x0; ΩBH) ⊂
FCP(x0; ΩGE).

## 3.3.2. Pac Prediction

Si et al. (2023) constructed PAC prediction set which relies on confidence interval ΩGE. Similar to section 3.3.1, if we replace ΩGE by ΩBH, we can obtain a smaller prediction set with the same PAC guarantee (Park et al., 2021).

Theorem 3.5. If Ω1 ⊂ Ω2*, then* FPAC(x0; Ω1) ⊂ FPAC(x0; Ω2) for all x0*. In particular,* FPAC(x0; ΩBH) ⊂ FPAC(x0; ΩGE). See Appendix D for the construction of FPAC(x0; Ω0) and Appendix B.5 for the proof of Theorem 3.5.

## 4. Experiments

In this section, we implemented the EBMaC to evaluate its performance on the MNIST (LeCun et al., 1998), CIFAR-10, and CIFAR-100 data sets (Krizhevsky et al., 2009).

$$\tau_{\rm CP}(y_{0};\Omega_{0})=\sup_{\omega\in\Omega_{0}}\tau_{\rm CP}(y_{0};\omega)\tag{6}$$
$\left(5\right)$. 
and FCP(x0; Ω0) = {y0 ∈ [K] : r(x0, y0) ≤ τCP(y0; Ω0)}. (7)

## 4.1. Classifier Training 4.3. Performance Of Ebmac On Importance Weights

For all data sets, we randomly selected 40,000 observations from training data set, combined with the 10,000 testing data to train classifiers, and used the remaining data to perform the analysis. For the MNIST data set, we trained 11 classifiers with different random seeds using the same architectures as in Azizzadenesheli et al. (2019). Each model was trained 10 epochs, and the best performer was retained. The final accuracy ranges from 97.25% to 98.07%. For the CIFAR data sets, we trained five classifiers using different architectures as shown in Table 1. Each classifier was trained 200 epochs, and the best performer was retained. In implementing EBMaC, we used the classifier with the lowest testing accuracy as g
∗, and the remaining classifiers as g1*, ..., g*G, where G = 10 for MNIST and G = 4 for CIFAR-10 and CIFAR-100.

220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274

| Model                                                                                                                              | CIFAR-10   | CIFAR-100   |
|------------------------------------------------------------------------------------------------------------------------------------|------------|-------------|
| VGG16a                                                                                                                             | 92.38%     | 71.80%      |
| ResNet18b                                                                                                                          | 93.98%     | 75.47%      |
| MobileNetV2c                                                                                                                       | 93.77%     | 70.23%      |
| PreActResNet18d                                                                                                                    | 93.97%     | 60.15%      |
| RegNetXe                                                                                                                           | 93.93%     | -           |
| GoogLeNetf                                                                                                                         | -          | 75.78%      |
| a Simonyan (2014) bHe et al. (2016a) c Sandler et al. (2018) dHe et al. (2016b) eRadosavovic et al. (2020) f Szegedy et al. (2015) |            |             |

## 4.2. Experimental Design

To generate the data sets that satisfy the label shift assumption, we performed the following construction using Dirichlet shift. In generating the source data, we first generated a random vector v from Dir(α1K). For each k ∈ [K], we randomly draw mvk observations, from those with y = k.

Here m is the source sample size, and α = 10, 000. We generated the target data in the same way, except that α = 10p with p = −3, −2, −1, 0, 1, 2, 3, and we did not retain the labels. We chose eight different m values, ranging from 1000 to 8000, with a step size of 1000. This resulted in 56 different data sets. In each data set, the sizes of the source data and target data are both m. Note that a smaller α leads to less balanced label distribution. This design is applied to MNIST, CIFAR-10 and CIFAR-100.

We compared EBMaC to BBSE (Lipton et al., 2018), RLLS (Azizzadenesheli et al., 2019), and MLLS (Azizzadenesheli et al., 2019), using MSE ∥ω − ωb∥
2as a criterion. The implementation code for the existing methods was adapted from Ye et al. (2024)
1. For CIFAR-100, as shown in the first plot of Figure 1, as the sample size and concentration parameter α increase, the MSE for EBMaC decreases, while the remaining three plots show that the MSE of EBMaC is generally smaller than other methods. The results for MNIST and CIFAR-10 are in Figures 5 and 6 in Appendix E.

In our analysis, we found that in some classes in the target data, the variance of predicted label counts sometimes exceeds its mean, which violates the property of the multinomial distribution. However, the hierarchical modeling allows overdispersion by introducing additional hyperparameters, which extends the model flexibility. As shown in Tables 2, 3, and 4 in Appendix E, we observe that the CIFAR-100 data set has larger average variance-mean ratios compared to that of MNIST and CIFAR-10.

## 4.4. Performance Of Ebmac On Confidence Sets

In obtaining confidence sets for CIFAR-100, we fix the same confidence level for Ω, ΩBH, and ΩGE, and present results in Figure 2. In the left panel, the x-axis represents the average length ratios of the GE method to the LP method, computed as K−1 PK
k=1(lk,GE/lk,BH), where lk,BH = ωk−ωk, and lk,GE is defined similarly. Here, GE was implemented using the code from Si et al. (2023)
2. Further, we perform a one-sided t-test to evaluate whether the log-ratio of the lengths log(lk,GE/lk,BH) is greater than zero across the labels. The resulting − log10(P-value) is shown in the y-axis. The horizontal line is at − log10(0.05), representing the statistical significance. In the right panel, we provide the bar plot of the ratio of the volume of Ω to ΩBH at each α value, presented in percentage. The results for CIFAR-10 and MNIST are similarly presented in Figures 3 and 4. In Figure 2, in the left panel, the vast majority is above the horizontal line, indicating that the improvement of the length ratio is significant in most cases. When comparing the results across three data sets, we find that EBMaC exhibits the best performance on CIFAR-100 in terms of both length ratio and P-value, but shows less improvement on MNIST. Note that all classifiers for MNIST have the best accuracy, while those for CIFAR-100 have the worst. This reflects that worse performance of classifiers generates more improvement of BH over GE. This is because worse classifiers lead to a confusion matrix that is less diagonal 1https://github.com/ChangkunYe/MAPLS
2https://github.com/averysi224/pac-ps-label-shift 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 dominant, which can be handled by BH without any issue but results in much inflation of the confidence set by GE. From the right panel of Figure 2, we can see that in the settings when α is large, the difference between Ω and ΩBH can be very large, reflected in very small volume ratio.

This indicates that aiming for a hyperrectangular shape can be costly. In such case, it might be wiser to use Ω to further perform conformal / PAC prediction. The performance in Figures 3 and 4 is slightly different in that the ratios of the volumes in the right panels are generally larger. This is because the shapes of Ω resemble more a hyperrectangle for CIFAR-10 and MNIST, due to a more diagonal dominant confusion matrix.

## 5. Discussion

The main innovations of EBMaC are in proposing an empirical Bayesian approach in hierarchical modeling for label shift problems (EB) and in handling the matrix constraints via linear programming (MaC). EB is able to handle overdispersion, while MaC achieves the tightest confidence sets for importance weights. These two components can work separately. For example, we can combine Clopper-Pearson interval (ClP) with MaC to obtain ClPMaC. We can also combine EB with GE to create EBGE. One obvious advantage of EB is in handling overdispersion, while the advantage of MaC is established theoretically. When the collection of classifiers performs poorly, EBMaC showcases the significant improvement of MaC over GE. The nice property of EBMaC naturally leads to a better outcome of downstream analysis, such as the prediction performance described in 3.3.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Alexandari, A., Kundaje, A., and Shrikumar, A. Maximum likelihood with bias-corrected calibration is hard-to-beat at label shift adaptation. In International Conference on Machine Learning, pp. 222–232. PMLR, 2020.

Azizzadenesheli, K., Liu, A., Yang, F., and Anandkumar, A.

Regularized learning for domain adaptation under label shifts. *arXiv preprint arXiv:1903.09734*, 2019.

Garg, S., Wu, Y., Balakrishnan, S., and Lipton, Z. A unified view of label shift estimation. *Advances in Neural* Information Processing Systems, 33:3290–3300, 2020.

Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. On calibration of modern neural networks. In International conference on machine learning, pp. 1321–1330. PMLR,
2017.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In *Proceedings of the IEEE* conference on computer vision and pattern recognition, pp. 770–778, 2016a.

He, K., Zhang, X., Ren, S., and Sun, J. Identity mappings in deep residual networks. In Computer Vision–ECCV 2016:
14th European Conference, Amsterdam, The Netherlands,

-2.2 -1.3 -0.4 0.5 1.4 log10(EBMaC)
-14.1 -7.0 0.0 7.0 14.1 log2(BBSE / EBMaC)
-10.9 -5.4 0.0 5.4 10.9 log2(RLLS / EBMaC)
-4.9 -2.5 0.0 2.5 4.9 log2(MLLS / EBMaC)
3 2 1 0 1 2 3 log10( )
1000 2000 3000 4000 5000 6000 7000 8000 Sam ple Si ze 3 2 1 0 1 2 3 log10( )
3 2 1 0 1 2 3 log10( )
3 2 1 0 1 2 3 log10( )
330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384

1.0 1.5 2.0 2.5 3.0 3.5 4.0 88.8%
43.2%
80.3%
54
.7%
80
.1%
-3 -2 -1 0 1 2 3 log10( )
48.9%
66.3%
-log 1 0(p
-va l)sample size 8000 7000 6000 5000 4000 3000 2000 1000 61.6%
-3.0
-2.0
-1.0 0.0 1.0 2.0 3.0 log10( )
53.9%
60
.8%
70
.0%
67.0%
60.

3%67
.7 %
1.0 1.01 1.02 1.03 1.04 1.05 1.06 1.07 length ratio (GE/BH)
99.5%
74.

1%
96.

8%
88
.9%
70
.0%
1.0 1.0016 1.0032 1.0048 1.0064 1.008 1.0096 1.0112 length ratio (GE/BH)
0.5 1.0 1.5 2.0 2.5 3.0
-3 -2 -1 0 1 2 3 log10( )
-log 10
(p-v al)sample size 8000 7000 6000 5000 74.6%
91.8%
4000 3000 2000 1000 90.9%
-3.0
-2.0
-1.0 0.0 1.0 2.0 3.0 log10( )
82.2%
91
.5%
95
.2%
94.9%
89.

6%94
.7 %
7

1.0 1.04 1.08 1.13 1.17 1.21 1.25 1.29 length ratio (GE/BH)
0 5 10 15 20 25 30 35 40 sample size 8000 7000 6000 5000 4000 3000 2000 1000 78.9%
0.0%
27.7%
0.

000 19 9%
0.0026%
0.0%
0.0%
-lo g1 0( p-va l)
0
.0%
0.0%
0.0%

-3.0
-2.0

-1.0 0.0 1.0 2.0 3.0 log10( )
-3 -2 -1 0 1 2 3 log10( )
0.0%
0.0%
0.

0%
0.

0
%
October 11–14, 2016, Proceedings, Part IV 14, pp. 630– 645. Springer, 2016b.

Tibshirani, R. J., Foygel Barber, R., Candes, E., and Ramdas, A. Conformal prediction under covariate shift. *Advances* in neural information processing systems, 32, 2019.

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images. 2009.

Valiant, L. G. A theory of the learnable. Communications of the ACM, 27(11):1134–1142, 1984.

LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradientbased learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

Vovk, V. Conditional validity of inductive conformal predictors. In *Asian conference on machine learning*, pp. 475–490. PMLR, 2012.

Lipton, Z., Wang, Y.-X., and Smola, A. Detecting and correcting for label shift with black box predictors. In International conference on machine learning, pp. 3122– 3130. PMLR, 2018.

Vovk, V., Gammerman, A., and Shafer, G. Algorithmic learning in a random world, volume 29. Springer, 2005.

Ye, C., Tsuchida, R., Petersson, L., and Barnes, N. Label shift estimation for class-imbalance problem: A bayesian approach. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 1073–
1082, 2024.

Minka, T. Estimating a dirichlet distribution, 2000. Park, S., Dobriban, E., Lee, I., and Bastani, O. PAC
prediction sets under covariate shift. *arXiv preprint* arXiv:2106.09848, 2021.

Podkopaev, A. and Ramdas, A. Distribution-free uncertainty quantification for classification under label shift. In Uncertainty in artificial intelligence, pp. 844–853. PMLR, 2021.

Radosavovic, I., Kosaraju, R. P., Girshick, R., He, K., and Dollar, P. Designing network design spaces. In ´ Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10428–10436, 2020.

Saerens, M., Latinne, P., and Decaestecker, C. Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural computation*, 14(1):21–41, 2002.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and Chen, L.-C. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4510–4520, 2018.

Si, W., Park, S., Lee, I., Dobriban, E., and Bastani, O.

PAC prediction sets under label shift. arXiv preprint arXiv:2310.12964, 2023.

Simonyan, K. Very deep convolutional networks for largescale image recognition. *arXiv preprint arXiv:1409.1556*, 2014.

Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S.,
Anguelov, D., Erhan, D., Vanhoucke, V., and Rabinovich, A. Going deeper with convolutions. In *Proceedings* of the IEEE conference on computer vision and pattern recognition, pp. 1–9, 2015.

Tian, Q., Zhang, X., and Zhao, J. ELSA: Efficient label shift adaptation through the lens of semiparametric models. In *International Conference on Machine Learning*, pp.

34120–34142. PMLR, 2023.

## A. Dirichlet-Multinomial Model

A.1. Moment Matching Estimation for Dirichlet-Multinomial Model We first give the moment matching estimation, which of course requires marginal statistics. Recall that we assume a common αs for all g ∈ G in Dirichlet prior. For any k in {1*, ..., K*2}, the marginal expectation and variance of mg,k are From Equation 8, we know µk = mk/m, Rearrange Equation 9 and replace µk with mk/m, we have 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 A.2. Marginal distribution for Dirichlet-Multinomial model 9

$$=E\{mc_{g,k}(1-c_{g,k})\}+m^{2}Var(c_{g,k})$$ $$=E(mc_{g,k})-mE^{2}(c_{g,k})-mVar(c_{g,k})+m^{2}Var(c_{g,k})$$ $$=m\mu_{k}(1-\mu_{k})\frac{m_{0}+m}{m_{0}+1},$$

where µk = αs,k/m0 and m0 =PK2 j=1 αs,j . Next, matching them to the sample mean and the sample variance leads to following equations,

$$\begin{array}{r c l}{{m\mu_{k}}}&{{=}}&{{G^{-1}\sum_{g=1}^{G}m_{g,k}\equiv\overline{{{m}}}_{k}}}\end{array}$$
$$m\mu_{k}=G^{-1}\sum_{g=1}^{G}m_{g,k}\equiv\overline{m}_{k}$$ $$m\mu_{k}(1-\mu_{k})\frac{m_{0}+m}{m_{0}+1}=(G-1)^{-1}\sum_{g=1}^{G}(m_{g,k}-\overline{m}_{k})^{2}\equiv\widehat{V}_{k}.$$
$$({\mathfrak{s}})$$
$$(9)$$
2 ≡ Vbk. (9)
$$\widehat{M_{0}}=\frac{m\overline{m}_{k}(m-\overline{m}_{k})-m\widehat{V}_{k}}{m\widehat{V}_{k}-\overline{m}_{k}(m-\overline{m}_{k})}$$ $$=(m-1)\left\{\frac{m\widehat{V}_{k}}{\overline{m}_{k}(m-\overline{m}_{k})}-1\right\}^{-1}$$
− 1.
Note that, we can have mc0 for each class, so we can average them to have a final Mc0. Then we substitute it into αs,k = m0mk/m to obtain αbs,k, for any k ∈ {1*, ..., K*2}.

$$\begin{array}{r c l}{{f(\mathbf{m}_{g};\mathbf{\alpha}_{s})}}&{{=}}&{{\int_{\mathcal{C}}f(\mathbf{m}_{g,k}\,|\,\mathbf{c}_{g})f(\mathbf{c}_{g};\mathbf{\alpha}_{s})d\mathbf{c}_{g}}}\end{array}$$
C C Γ(m0) QK2 k=1 Γ(αs,k) Y K2 k=1 c αs,k−1 g,k · Γ(m + 1) QK2 k=1 Γ (mg,k + 1) Y K2 k=1 c mg,k g,k dcg QK2 k=1 Γ (mg,k + 1) Z C Y K2 k=1 c αs,k−1 g,k Y K2 =Γ(m0) QK2 k=1 Γ(αk) Γ(m + 1) k=1 c mg,k g,k dcg
$=\;\frac{1}{2}$
=Z
$$\begin{array}{c c}{{\Gamma(m_{0})}}&{{\Gamma(m+1)}}\\ {{\overline{{{\prod_{k=1}^{K^{2}}\Gamma(\alpha_{s,k})}}}}}&{{\overline{{{\prod_{k=1}^{K^{2}}\Gamma(m_{g,k}+1)}}}}}\end{array}\int_{C}\prod_{k=1}^{K^{2}}c_{g,k}^{\alpha_{s,k}+m_{g,k}-1}\,d\mathbf{c}_{g}$$ $$\begin{array}{c c}{{\Gamma(m_{0})}}&{{\Gamma(m+1)}}\\ {{\overline{{{\prod_{k=1}^{K^{2}}\Gamma(\alpha_{k})}}}}}&{{\overline{{{\prod_{k=1}^{K^{2}}\Gamma(m_{g,k}+1)}}}}}\end{array}B(\alpha_{s,1}+m_{g,1},...,\alpha_{s,K^{2}}+m_{g,K^{2}})$$
$\overline{\overline{\phantom{\rule{0.000pt}{0ex}}}}$
$$\begin{array}{r l}{{}}&{{}{}={\mathrm{~for~}}\Gamma(m_{0})\Gamma(m+1)}\\ {{}}&{{}={\mathrm{~}}{\frac{\Gamma(m_{0})\Gamma(m+1)}{\Gamma(m_{0}+m)}}\prod_{k=1}^{K^{2}}{\frac{\Gamma(\alpha_{s,k}+m_{g,k})}{\Gamma(\alpha_{s,k})\Gamma(m_{g,k}+1)}}}\end{array}$$
$$\begin{array}{r c l}{{E(M_{g,k})}}&{{=}}&{{E\{E(M_{g,k}\,|\,{\bf c}_{g})\}=E(m c_{g,k})=m E(c_{g,k})=m\mu_{k}}}\\ {{V a r(M_{g,k})}}&{{=}}&{{E\{V a r(M_{g,k}\,|\,{\bf c}_{g})\}+V a r\{E(M_{g,k}\,|\,{\bf c}_{g})\}}}\\ {{}}&{{}}&{{}}\end{array}$$
$$\sum_{j=1}^{K}\bar{c}_{i j}\omega_{j}\geq\sum_{j=1}^{K}c_{i j}\omega_{j}=q_{i}\geq q_{i},$$
$$\sum_{j=1}^{K}c_{i j}\omega_{j}\leq\sum_{j=1}^{K}c_{i j}\omega_{j}=q_{i}\leq\overline{{{q}}}_{i},$$
$$q_{i}(l)=\sum_{j=1}^{l}c_{i j}\omega_{j}+\sum_{j=l+1}^{K}\overline{{{c}}}_{i j}\omega_{j}.$$
$$\widetilde{c}_{i l_{0}}=\underline{{{c}}}_{i l_{0}}+\frac{\overline{{{q}}}_{i}-q_{i}(l_{0})}{\omega_{l_{0}}}.$$
.

## B. Proofs B.3. Proof Of Theorem 3.3

B.1. Proof of Theorem 3.1 Proof. P
Suppose that C ∈ C, q ∈ Q, and that ω satisfies Cω = q and ω > 0. The linear equation Cω = q is equivalent to K
j=1 cijωj = qi. Since ωi > 0, we get that which implies that ω ∈ Ω. Now, we prove the other direction. Suppose that ω ∈ Ω. Then for each i ∈ [K], we can apply the following procedure. If PK
j=1 cijωj ≤ qi, ∀i ∈ [K], take c
⊤
i = (ci1*, ...,* ciK) and qi =PK
j=1 cijωj . Otherwise, for l ∈ [K], define Then qi(l) is a decreasing function of l, and qi(K) = PK
j=1 cijωj ≤ qi by the condition. Thus, we can find l0 such that qi(l0 − 1) ≥ qi ≥ qi(l0). Let c
⊤
i = (ci1, ..., ci,l0−1, ecil0, ci,l0+1*, ...,* ciK) and qi = qi, where 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Then ecil0 ∈ [cil0
, cil0] and c
⊤
i ω = qi. Taking c
⊤
ias the i-th row of C and qi as the i-th element of q, we get that Cω = q, where C ∈ C and q ∈ Q. B.2. Proof of Corollary 3.2 Proof. The first part, Ω ⊂ ΩBH, is trivial from its definition. For the second part, note that Ω ⊂ ΩGE by Theorem 3.1. Since ΩBH is the smallest hyperrectangle that contains Ω, we get ΩBH ⊂ ΩGE.

Proof. By Theorem 2 of Podkopaev & Ramdas (2021), we have P(X0,Y0)∼Q{Y0 ∈ FCP(X0; ω)} ≥ 1 − α. Also, if ω ∈ Ω0, we get FCP(X0; ω) ⊂ FCP(X0; Ω0) by Theorem 3.4. Then P(X0,Y0)∼Q{Y0 ̸∈ FCP(X0; Ω0)}
= P(X0,Y0)∼Q{ω ∈ Ω0 and Y0 ̸∈ FCP(X0; Ω0)} + P(X0,Y0)∼Q{ω ̸∈ Ω0 and Y0 ̸∈ FCP(X0; Ω0)} ≤ P(X0,Y0)∼Q{Y ̸∈ FCP(X0; ω)} + P(ω ̸∈ Ω0)
≤ α + δ.

B.4. Proof of Theorem 3.4 Proof. First, (6) implies τCP(y; Ω1) ≤ τCP(y; Ω2) for all y. Then (7) gives the result.

B.5. Proof of Theorem 3.5 Proof. First, (11) implies τPAC{T(Ω1, S1, V, b)} ≤ τPAC{T(Ω2, S1*, V, b*)} for all y. Then (12) gives the result.

$\square$
$\square$
10

## C. Gaussian Elimination With Intervals

Given that C∗ ∈ C = [C, C] and q
∗ ∈ Q = [q, q], Si et al. (2023) introduce an intuitive way, which they named Gaussian elimination with intervals, of finding Ω that contains ω = C∗−1q
∗. Suppose that cij ≥ 0, qi
> 0, and ωi > 0 for *i, j* ∈ [K].

They follow two phases of Gaussian elimination when solving a system of linear equations C∗ω = q
∗and derive the elementwise interval for ωi. First, set c 0 ij = cij , c 0 ij = cij , q 0 i
= qi
, and q 0 i = qi. In the first phase (forward elimination), the elementary row operations are applied sequentially for k = 1*, ..., K* − 1 to delete the (*i, k*) element in the matrix for *i > k* by adding the multiple of the k-th row. Then the lower bound c k+1 ij and the upper bound c k+1 ij are derived from the interval
[C
k, C
k] at the k-th step as

$${\underline{{c}}}_{i j}^{k+1}={\begin{cases}0,\\ {\underline{{c}}}_{i j}^{k}-{\frac{{\overline{{c}}}_{i k}^{k}{\overline{{c}}}_{k j}^{k}}{{\underline{{c}}}_{k k}^{k}}},\\ {\underline{{c}}}_{i j}^{k},\end{cases}}$$
0, if *i > k, j* ≤ k,
ij , otherwise.
$${\mathrm{if~}}i>k,j\leq k,$$ $${\mathrm{if~}}i,j>k,$$ $${\mathrm{otherwise.}}$$
$$\overline{{{c}}}_{i j}^{k+1}=\begin{cases}0,\\ \overline{{{c}}}_{i j}^{k}-\frac{c_{i k}^{k}c_{k j}^{k}}{\overline{{{c}}}_{k k}^{k}},\\ c_{i j}^{k},\end{cases}$$
$${\mathrm{if~}}i>k,j\leq k,$$
0, if *i > k, j* ≤ k,
ij , otherwise.
* [16] A. A. K.  
Simultaneously, q k+1 iand q k+1 iare obtained from the same row operations to be

$$\underline{{{q}}}_{i}^{k+1}=\begin{cases}\underline{{{q}}}_{i}^{k}-\frac{\overline{{{c}}}_{i k}^{k}\overline{{{q}}}_{k}^{k}}{\underline{{{c}}}_{k k}^{k}},&\text{if}i>k,\\ \underline{{{q}}}_{i}^{k},&\text{otherwise.}\end{cases}$$
$$\overline{{{q}}}_{i}^{k+1}=\begin{cases}\overline{{{q}}}_{i}^{k}-\frac{c_{i k}^{k}q_{i}^{k}}{\overline{{{c}}}_{k k}^{k}},&\text{if}i>k,\\ \overline{{{q}}}_{i}^{k},&\text{otherwise}.\end{cases}$$

Then c
∗,k+1 ij and q
∗,k+1 i, which would have been obtained in the forward elimination step solving C∗ω = q
∗, always lie in
[c k+1 ij , c k+1 ij ] and [q k+1 i, q k+1 i]. In the second phase (back substitution), they compute ωiand ωi, iteratively for i = *K, ...,* 1, replacing the truth with intervals as in the first phase.

$\underline{s}_{i}=\sum_{j=i+1}^{K}\underline{c}_{ij}^{K}\underline{\omega}_{j}$ and $\overline{s}_{i}=\sum_{j=i+1}^{K}\overline{c}_{ij}^{K}\overline{\omega}_{j}$, $\underline{\omega}_{i}=\frac{q_{i}-\overline{s}_{i}}{\overline{c}_{ii}^{K}}$ and $\overline{\omega}_{i}=\frac{\overline{q}_{i}-\underline{s}_{i}}{\overline{c}_{ii}^{K}}$.  
550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 Then ΩGE is defined as the K-dimensional hyperrectangle QK
i=1[ωi, ωi]. Si et al. (2023) provide a theoretical result that their method yields ω ∈ ΩGE if c k ij ≥ 0, c k ii > 0, and q k i
≥ 0 for all *i, j, k* ∈ [K]. The basic assumption in order to satisfy the condition is that cik ≪ ckk. This is ensured when the classifier g(X) is accurate, that is, when the diagonal terms c
∗
kk in C∗ dominate non-diagonal terms. If the assumption is violated, we may encounter a possibility that c
∗,k kk ≈ 0, which may lead to c kkk ≤ 0. Then in the forward elimination phase, c k+1 ij for all *i, j > k* will be −∞, which may make the algorithm impractical. Furthermore, if qi
≤ si or c K
ii ≤ 0 for some i, then the back substitution phase would lead to ωi ≤ 0 or ωi = ∞, which does not provide any information about the interval of ωi. In order to deal with the nonpositive bounds, they mention that choosing a wider margin, which would, however, make ΩGE larger than its optimal size.

## D. Details Of Pac Prediction

Let the calibration set be S1 = {(xi, yi)}
m1 i=1 and denote by r(x, y) the nonconformity score trained separately. The PAC
prediction set FPAC(x; ω, S1) under label shift (Vovk, 2012; Park et al., 2021; Si et al., 2023) is defined by

PS1∼P m1 [P(X0,Y0)∼Q{Y0 ∈ FPAC(X0; ω, S1)} ≥ 1 − ϵ] ≥ 1 − η.

$$\omega,S_{1})\}\geq1-$$

11

$$b)\}\geq1-\epsilon]\geq1-\eta,$$
$$(10)^{\frac{1}{2}}$$

Si et al. (2023) constructed a set that satisfies a modification of PAC guarantee such that

PS1∼P m1 ,V [P(X0,Y0)∼Q{Y0 ∈ FPAC(X0; ω, S1, V, b)} ≥ 1 − ϵ] ≥ 1 − η, (10)
where V = (V1*, ..., V*m1)
⊤ ∼ *Unif*([0, 1])m1 and b = maxk∈[K] ωk. The set FPAC(x; ω, S1*, V, b*) is in the form of

$\mathcal{E}_1$
FPAC(x; ω, S1*, V, b*) = [y ∈ [K] : r(x, y) ≤ τPAC{T(ω, S1*, V, b*)}],
where T(ω, S1*, V, b*) = {(xi, yi) ∈ S1 : Vi ≤ ωyi
/b} is a target sample generated by rejection-sampling from S1. Let m0 = |T(ω, S1*, V, b*)|. Here, τPAC{T(ω, S1*, V, b*)} is chosen to satisfy 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659

$$\sum_{(\mathbf{x}_{i},y_{i})\in T(\mathbf{\omega},S_{1},V,b)}\mathbf{1}\{y_{i}\not\in F_{\mathrm{PAC}}(\mathbf{x}_{i};\mathbf{\omega},S_{1},V,b)\}$$ $$=\sum_{(\mathbf{x}_{i},y_{i})\in T(\mathbf{\omega},S_{1},V,b)}\mathbf{1}[r(\mathbf{x}_{i},y_{i})>\tau_{\mathrm{PAC}}\{T(\mathbf{\omega},S_{1},V,b)\}]\leq k(m_{0},\epsilon,\eta),$$

that is, τPAC{T(ω, S1*, V, b*)} is the largest value that is less than the k(m0*, ϵ, η*)-th largest value of {r(xi, yi) : (xi, yi) ∈ T(ω, S1*, V, b*)}, where

$$k(m_{0},\epsilon,\eta)=\operatorname*{max}\{k:F_{\mathrm{Binom}(m_{0},\epsilon)}(k)\leq\eta\}.$$
$$(11)$$

Note that FBinom(n,ϵ)(·) is the CDF of Binom(*n, ϵ*). If the true importance weight ω is used, then the modified PAC condition
(10) is satisfied. When the confidence set Ω0 with P(ω ∈ Ω) ≥ 1 − δ is provided, we can define

$$\tau_{\mathrm{PAC}}\{T(\mathbf{\Omega},S_{1},V,b)\}=\operatorname*{sup}_{\omega\in\mathbf{\Omega}}\tau_{\mathrm{PAC}}\{T(\omega,S_{1},V,b)\}$$
$$(12)$$
$\bigcup\{|\cdot|$. 
τPAC{T(ω, S1*, V, b*)} (11)
and

$$F_{\mathrm{PAC}}(\mathbf{x};\mathbf{\Omega},S_{1},V,b)=[y$$

FPAC(x; Ω, S1*, V, b*) = [y ∈ [K] : r(x, y) ≤ τPAC{T(Ω, S1*, V, b*)}]. (12)
Then FPAC(x; Ω, S1*, V, b*) satisfies the modified PAC condition (10) with η being η + δ.

Theorem D.1. Suppose that P(ω ∈ Ω0) ≥ 1 − δ*. Then*

$\mathbb{P}_{S_{1}\sim P^{m_{1}},V}[\mathbb{P}(\mathbf{x}_{0},Y_{0})\sim_{Q}\{Y_{0}\in F_{\text{PAC}}(\mathbf{X}_{0};\Omega,S_{1},V,b)\}\geq1-\epsilon]\geq1-\eta-\delta$.  
Proof. The proof follows from Theorem 3 of Park et al. (2021).

## E. Data Dispersion

Table 2. (MNIST) Average variance-mean ratios for all classes under different sample size and Dirichlet shift combinations.

log10(α)

sample size (m) -3 -2 -1 0 1 2 3

8000 3.62 3.17 3.10 0.43 0.32 0.31 0.32 7000 6.05 8.06 5.78 0.58 0.30 0.25 0.29 6000 8.26 6.38 3.57 0.77 0.28 0.25 0.27 5000 3.77 4.73 1.44 0.38 0.22 0.23 0.26 4000 3.08 3.82 2.67 0.27 0.21 0.18 0.15 3000 4.97 3.07 1.07 0.33 0.15 0.13 0.15 2000 2.69 2.08 0.94 0.42 0.14 0.12 0.11 1000 1.04 1.15 0.79 0.09 0.08 0.09 0.07

$\square$
Table 3. (CIFAR-10) Average variance-mean ratios for all classes under different sample size and Dirichlet shift combinations.

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696

697

698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714

| log10(α)        |      |      |      |      |      |      |      |
|-----------------|------|------|------|------|------|------|------|
| sample size (m) | -3   | -2   | -1   | 0    | 1    | 2    | 3    |
| 8000            | 3.55 | 2.73 | 1.58 | 0.61 | 0.23 | 0.31 | 0.36 |
| 7000            | 5.70 | 5.72 | 2.33 | 0.63 | 0.23 | 0.18 | 0.17 |
| 6000            | 6.10 | 4.45 | 1.44 | 0.28 | 0.35 | 0.23 | 0.18 |
| 5000            | 3.84 | 3.58 | 0.77 | 0.22 | 0.23 | 0.14 | 0.20 |
| 4000            | 4.15 | 4.02 | 2.43 | 0.27 | 0.15 | 0.28 | 0.26 |
| 3000            | 2.74 | 3.92 | 0.79 | 0.27 | 0.23 | 0.13 | 0.14 |
| 2000            | 1.19 | 2.19 | 0.70 | 0.22 | 0.14 | 0.19 | 0.15 |
| 1000            | 1.06 | 0.85 | 0.83 | 0.16 | 0.16 | 0.21 | 0.13 |

-5.0 -4.3 -3.6 -3.0 -2.3 log10(EBMaC)
-5.0 -2.5 0.0 2.5 5.0 log2(BBSE / EBMaC)
2.0 6.2 10.4 14.6 18.8 log2(RLLS / EBMaC)
-6.6 -3.3 0.0 3.3 6.6 log2(MLLS / EBMaC)
3 2 1 0 1 2 3 log10( )
1000 2000 3000 4000 5000 6000 7000 8000 Sa mple Size 3 2 1 0 1 2 3 log10( )3 2 1 0 1 2 3 log10( )3 2 1 0 1 2 3 log10( )

| log10(α)        |       |       |      |      |      |      |      |
|-----------------|-------|-------|------|------|------|------|------|
| sample size (m) | -3    | -2    | -1   | 0    | 1    | 2    | 3    |
| 8000            | 25.95 | 12.42 | 3.70 | 1.36 | 0.95 | 0.86 | 0.90 |
| 7000            | 12.82 | 10.68 | 4.31 | 1.14 | 0.81 | 0.78 | 0.82 |
| 6000            | 11.02 | 12.91 | 3.68 | 1.18 | 0.75 | 0.77 | 0.74 |
| 5000            | 5.09  | 16.20 | 2.59 | 1.00 | 0.70 | 0.75 | 0.64 |
| 4000            | 10.08 | 8.03  | 3.07 | 0.93 | 0.77 | 0.57 | 0.65 |
| 3000            | 6.14  | 7.64  | 3.33 | 0.98 | 0.64 | 0.55 | 0.54 |
| 2000            | 2.93  | 2.96  | 2.02 | 0.71 | 0.53 | 0.53 | 0.55 |
| 1000            | 1.15  | 1.68  | 0.87 | 0.48 | 0.46 | 0.55 | 0.46 |

715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769

-4.6 -3.9 -3.2 -2.5 -1.8 log10(EBMaC)
-5.5 -2.7 0.0 2.7 5.5 log2(BBSE / EBMaC)
0.7 4.5 8.3 12.2 16.0 log2(RLLS / EBMaC)
-7.0 -3.5 0.0 3.5 7.0 log2(MLLS / EBMaC)
3 2 1 0 1 2 3 log10( )
1000 2000 3000 4000 5000 6000 7000 8000 Sam ple S
ize 3 2 1 0 1 2 3 log10( )3 2 1 0 1 2 3 log10( )3 2 1 0 1 2 3 log10( )