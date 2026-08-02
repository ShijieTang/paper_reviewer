# Inconsistency-Aware Minimization: Improving Generalization With Unlabeled Data

| Anonymous Author(s) Affiliation Address email   |
|-------------------------------------------------|

## Abstract

1 Estimating the generalization gap and devising optimization methods that general2 ize better are crucial for deep learning models, both for theoretical understanding 3 and for practical applications. The ability to leverage unlabeled data for these pur4 poses offers significant advantages in real-world scenarios. This paper introduces 5 a novel generalization measure, termed *local inconsistency*, developed from an 6 information-geometric perspective of the neural network's parameter space. A 7 key feature of local inconsistency is its computability from unlabeled data. We 8 establish its theoretical underpinnings by connecting local inconsistency to the 9 Fisher information matrix and the loss Hessian. Empirically, we demonstrate 10 that local inconsistency correlates with the generalization gap also exhibiting 11 characteristics comparable to *sharpness*. Based on these findings, we propose 12 Inconsistency-Aware Minimization (IAM) that incorporates local inconsistency 13 into the objective. We demonstrate that in standard supervised learning settings, 14 IAM enhances generalization, achieving performance comparable to existing meth15 ods such as Sharpness-Aware Minimization. Furthermore, IAM exhibits efficacy 16 in semi- and self-supervised learning scenarios, where the local inconsistency is 17 computed from the unlabeled data. 19 Estimating the generalization gap and developing optimization techniques to enhance performance 20 on unseen data are pivotal challenges in both the theory and practice of deep learning. There are 21 numerous reports linking the flatness of the loss landscape to generalization [16, 7, 20, 9]. Conse22 quently, numerous minimization methods leveraging sharpness have been proposed, demonstrating 23 improvements in generalization [8, 19, 17, 35] . However, the efficacy of such sharpness-based 24 approaches is not fully understood as recent studies, including Andriushchenko et al. [1], indicate 25 that sharpness alone does not reliably predict the generalization gap. 26 On the other hand, Jiang et al. [14], Johnson and Zhang [15] have investigated alternative gener27 alization measures, such as disagreement and *inconsistency*. Under certain conditions (e.g., zero 28 training error or low randomness of final states), these metrics have been shown to estimate the 29 generalization gap more accurately than ones based on sharpness. From a practical standpoint, it is 30 highly desirable to leverage unlabeled data to compute these measures. However, existing methods 31 for computing disagreement typically require training two separate models under identical conditions 32 (while still being subject to training randomness), and estimating inconsistency often necessitates 33 training multiple models on each of distinct datasets, thereby incurring substantial computational 34 overhead. Such prerequisites hinder the direct minimization of these measures within a single-model 35 training paradigm.

## 18 **1 Introduction**

36 In many real-world applications, unlabeled data is far more abundant and accessible than labeled data. 37 Therefore, methods capable of estimating the generalization gap and subsequently minimizing it using 38 only unlabeled data within a single-model framework (i.e., without resorting to auxiliary models or 39 requiring held-out labels) are crucial for resource-efficient deployment in practical scenarios. 40 Addressing this need, this paper introduces *local inconsistency*, a novel generalization measure derived 41 from an information-geometric perspective of the parameter space. This measure is theoretically 42 grounded through its connection to the Fisher Information Matrix (FIM) and the loss Hessian. 43 Crucially, local inconsistency can be computed using only **unlabeled** data and from a **single** trained 44 model. We demonstrate that local inconsistency exhibits a predictive capability for the generalization 45 gap comparable to sharpness-based measures in some settings. It also shows distinct characteristics 46 where traditional sharpness measures may falter. Building upon local inconsistency, we propose 47 **Inconsistency-Aware Minimization** (IAM), a novel optimization strategy that incorporates local 48 inconsistency into the training objective. In standard supervised learning settings on CIFAR-10 49 and CIFAR-100[18], IAM achieves generalization performance comparable or superior to existing 50 methods like Sharpness-Aware Minimization (SAM) [8]. Furthermore, IAM demonstrates its efficacy 51 in semi- and self-supervised learning, achieving higher test accuracy than standard SGD when 52 integrated with SimCLR [6]. 53 Our contributions can be summarized as follows: 54 - We propose *local inconsistency*, a novel generalization measure rooted in information 55 geometry, designed to capture the sensitivity of model outputs to perturbations in the 56 parameter space. This measure offers significant practical advantages as it can be computed 57 using (i) only unlabeled data and (ii) a single trained model. 58 - We establish the theoretical underpinnings of local inconsistency by linking it to the Fisher 59 Information Matrix (FIM) and the loss Hessian. Furthermore, we discuss its approximate 60 relationship with the inconsistency measure previously proposed by Johnson and Zhang 61 [15].

62 - We develop Inconsistency-Aware Minimization (IAM), a novel optimization framework 63 with two variants (IAM-D and IAM-S), that directly incorporates local inconsistency into 64 the training objective to seek flatter minima in terms of output sensitivity. IAM demonstrates 65 improved generalization performance across diverse learning paradigms, including standard 66 supervised learning and self-supervised learning settings, showcasing its broad applicability.

## 67 **2 Related Work**

68 Understanding and improving generalization in deep neural networks, especially given their large 69 capacity and tendency to overfit [33], remains a central challenge. While networks can memorize 70 random labels [33] and learn simple patterns before noise [2], phenomena like double descent [25]
71 and the inadequacy of uniform convergence theory [24] highlight the need for novel generalization 72 measures beyond loss-based metrics. 73 Traditional measures like VC-dimension often fall short. While spectrally-normalized margin bounds 74 [3] and PAC-Bayes approaches offer insights, no single measure consistently predicts generalization 75 [13]. Recently, disagreement [14] and inconsistency [15] have shown promise, correlating well 76 with the generalization gap, even when computed on unlabeled data. However, their reliance on 77 training multiple models poses practical limitations for direct optimization in a single-model setup, 78 underscoring the need for efficient, label-free, single-model generalization measures. 79 The geometry of the loss landscape, particularly the flatness of minima, has been extensively linked 80 to generalization [16, 20]. However, the utility of sharpness as a sole predictor is debated due to 81 issues like scale invariance [7] and its correlation with training hyperparameters rather than true 82 generalization [1]. Indeed, some studies suggest that output inconsistency and instability can be more 83 reliable predictors than sharpness [15]. Information geometry has inspired reparametrization-invariant 84 sharpness measure [12], but these can be computationally expensive. This context motivates our 85 exploration of "local inconsistency", an alternative geometric measure focusing on output sensitivity 86 within a parameter neighborhood, computable from unlabeled data using a single model. 87 Various regularization techniques, both explicit (e.g., dropout [30], batch normalization [28], Mixup 88 [34]) and implicit (e.g., SGD's bias [10, 29]), aim to improve generalization. Methods like Sharpness89 Aware Minimization (SAM, [8]) and ASAM [19] directly optimize for flat minima and have shown 90 significant improvements. Despite their success, the precise role of sharpness in generalization 91 remains an active area of research [13, 1], further motivating the development of complementary 92 approaches like our proposed IAM.

## 93 **3 Background And Preliminaries**

94 In this section, we briefly review fundamental concepts and notations essential for understanding 95 our proposed metric and its theoretical connections. We focus on probabilistic classification models, 96 information geometry, and aspects of the loss landscape.

## 97 **3.1 Notation And Problem Setup**

98 We consider probabilistic classification models. Let x ∈ X be a data point from the input space 99 X , and y ∈ [C] = {0, 1*, . . . , C* − 1} be the corresponding class label, where C is the total number 100 of classes. The data pair (*x, y*) are assumed to be drawn from an underlying distribution D over X × [C]. A model, parameterized by θ ∈ R
m 101 , outputs a probability distribution over classes for a 102 given input x. This is typically achieved by transforming a logit vector z(*x, θ*) through a softmax function: f(*x, θ*) = softmax(z(*x, θ*)). Thus, f(x, θ) = [p(0|x; θ), p(1|x; θ)*, . . . , p*(C − 1|x; θ)]⊤ 103 .

Given a training dataset Zn = {(xi 104 , yi) : i = 1*, . . . , n*} drawn i.i.d. from D, the model is typically 105 trained by minimizing a loss function. For classification, the empirical Cross-Entropy (CE) loss will be written as L(θ) = 1n Pn i=1 li(θ), where per-sample loss is li(θ) = l(xi, yi; θ) = − log p(yi|xi 106 ; θ).

## 107 **3.2 Fisher Information Matrix (Fim) And Kl Divergence**

108 The Fisher information matrix (FIM), F(θ), for the family of probability density p(x, y; θ) = 109 p(x)p(y|x; θ) parameterized by a parameters θ is defined as F(θ) = Ex∼p(x)
-Ey∼p(y|x;θ)
-∇θl(x, y; θ)∇θl(*x, y*; θ)
⊤
= Ex∼p(x)-∇θz(xi, θ)diag(f(xi, θ)) − f(xi, θ)f(xi, θ)
⊤∇θz(xi, θ)
⊤. (1)
In practice, the expectation Ep(x)
110 is often approximated by an empirical average over the available data (e.g., training data {xi}
n i=1 111 or unlabeled data).

112 The Kullback-Leibler (KL) divergence between the output distributions of a model with parameters θ 113 and a slightly perturbed model θ+δ, f(*x, θ*) and f(*x, θ*+δ), respectively, can be locally approximated 114 using a second-order Taylor expansion with respect to δ as:

$$\mathbb{E}_{x\sim p(x)}\left[\mathrm{KL}\left(f(x,\theta)\|f(x,\theta+\delta)\right)\right]=\frac{1}{2}\delta^{\top}F(\theta)\delta+O(\|\delta\|_{2}^{3}).$$
$$(2)$$
2). (2)

## 115 **3.3 Loss Hessian And Gauss-Newton Approximation**

The geometry of the empirical loss surface L(θ) is described by its Hessian matrix H(θ) = ∇2 116 θL(θ).

117 For the Cross-Entropy (CE) loss, the Hessian can be approximated by the Gauss-Newton (GN) matrix, G(θ). The second derivative of the per-sample CE loss ℓi(θ) with respect to the logits zi = z(xi 118 , θ),
∇2zℓi(θ) = diag(f(xi, θ)) − f(xi, θ)f(xi, θ)
⊤ 119 , depends only on the model's output probabilities f(xi, θ). Consequently, the per-sample GN term, Gi(θ) = ∇θz
⊤
i(∇2zℓi)∇θzi 120 , is equivalent to the FIM contribution in Eq. (1). The empirical GN matrix, G(θ) = 1n Pn 121 i=1 Gi(θ), thus often termed the 122 empirical FIM, provides a positive semi-definite approximation to H(θ) and is frequently used in 123 optimization [22, 26].

## 124 **4 Accessing Generalization Gap Via Local Inconsistency**

125 This section introduces our proposed measure, local inconsistency, designed to capture the gen126 eralization gap. We first define local inconsistency and elucidate its theoretical underpinnings by 127 connecting it to the FIM and the loss Hessian. We then discuss its relationship with inconsistency 128 [15]. Finally, we present empirical results demonstrating the correlation between local inconsistency 129 and the generalization gap, comparing it with other common measures.

## 130 **4.1 Local Inconsistency,** Sρ(Θ)

131 We introduce local inconsistency, Sρ(θ), defined as:
Sρ(θ) = max∥δ∥≤ρ Ex∼p(x)[KL(f(x, θ)∥f(*x, θ* + δ))], (3)
132 which represents the sensitivity of the model's output distribution f(*x, θ*) with respect to the worst 133 perturbations δ, within an Euclidean ball of radius ρ around the parameter θ. Intuitively, a high 134 value of Sρ(θ) indicates that the model's output distribution is highly sensitive to small perturbations 135 in parameter space. This sensitivity suggests potential instability or uncertainty in the model's 136 predictions associated with the vicinity of θ.

137 **Practical Advantages of** Sρ Local inconsistency shares a practical advantage with sharpness138 based measures [16, 8] in that it can be calculated using a **single** trained model. Furthermore, like 139 disagreement [14] and inconsistency [15], our metric can be estimated using only **unlabeled** data.

140 A notable advantage over inconsistency and disagreement estimation is that evaluating Sρ does not 141 require training multiple model instances derived from the same training procedure. This potentially 142 makes Sρ more computationally efficient and practical to compute, especially when model training is 143 resource-intensive.

## 144 **4.2 Connection To Fim And Hessian**

145 The relationship between our metric Sρ and the Fisher Information Matrix (FIM) can be established 146 by leveraging the local quadratic approximation of the KL divergence, as outlined in Section 3. With 147 this quadratic approximation, we can approximate Sρ(w) with the maximum eigenvalue of FIM,
scaled by ρ 2 148 /2:

$$S_{\rho}(\theta)\approx\operatorname*{max}_{\|\delta\|\leq\rho}{\frac{1}{2}}\delta^{\top}F(\theta)\delta={\frac{1}{2}}(\rho v_{\operatorname*{max}})^{\top}F(\theta)(\rho v_{\operatorname*{max}})={\frac{1}{2}}\rho^{2}\lambda_{t}$$
2λmax,
149 where vmax is the eigenvector corresponding to the largest eigenvalue λmax of F(θ). Remarkably, 150 this approximation requires only the model θ and unlabeled data (used to compute the expectation). 151 The Fisher Information Matrix F(θ), to which Sρ(θ) is related via its maximum eigenvalue, also 152 connects to the Hessian of the loss function H(θ). As detailed in Section 3, for Negative Log 153 Likelihood losses such as Cross-Entropy, the Hessian can be approximated by the Gauss-Newton 154 matrix G(θ), equivalent to empirical FIM computed using training data.

Consequently, when calculating Sρ(θ) using the training data, it approximates 12 ρ 2 155 λmax(G(θ)).

156 Given that G(θ) often provides a good approximation to the true loss Hessian near a local minimum, 157 Sρ(θ) therefore offers insights into the maximum curvature of the loss landscape in that vicinity.

## 158 **4.3 Local Inconsistency And Generalization Bound**

159 Beyond its connection to the local geometry of the loss landscape, our proposed local inconsistency 160 measure, Sρ(θ), can also be linked to the generalization ability of the model. Inspired by PAC-
161 Bayesian analyses that connect the geometry of the loss neighborhood to generalization [8], we can 162 sketch a theoretical argument suggesting that controlling Sρ(θ) contributes to a generalization bound.

163 We present an informal theorem that captures this intuition, with the detailed heuristic derivation 164 provided in Appendix C. 165 **Theorem 1 (Informal Generalization Bound with Local Inconsistency)** Under certain assump166 tions regarding the relationship between the worst-case empirical loss increase in a ρ*-neighborhood* 167 and the local inconsistency Sρ(θ) (evaluated on the training set Zn*), the true risk* LD (θ) =
168 E(x,y)∼D [l(x, y; θ)] *can be bounded with high probability as:*
LD (θ) ≲ L(θ) + Sρ(θ) + R(∥θ∥
2/ρ2) (4)
where R : R → R
+ 169 *is a strictly increasing function.*
170 This bound (Eq. (4)) suggests that minimizing a combination of the empirical loss LS (θ) and the 171 local inconsistency Sρ(θ) can lead to a lower upper bound on the true risk. This provides a theoretical 172 motivation for our Inconsistency-Aware Minimization (IAM) framework, which aims to find solutions 173 that are not only accurate on the training data but also exhibit low output sensitivity in the parameter 174 space, as measured by Sρ(θ).

$$(\;||\mathbf{\Phi}||\;\;/\;p\;\;)$$

## 175 **4.4 Relation With Inconsistency In Johnson And Zhang [15]**

176 Local inconsistency exhibits an interesting relationship to the inconsistency in Johnson and Zhang 177 [15] defined as:

$$({\boldsymbol{5}})$$

CP = EZn Eθ,θ′∼ΘP |Zn

$${}_{x)}[\mathrm{KL}(f(x,\theta)\|f(x,\theta^{\prime}))].$$

We consider the conditional inconsistency for a fixed Zn, denoted CP |Zn
178 , without outer expectation.
Then our proposed metric, Sρ(θZn
179 ), is approximately proportional to the conditional inconsistency
CP |Zn
180 :
$$\frac{m}{2C}\mathcal{C}_{P|Z_{n}}\stackrel{<}{\sim}S_{\rho}(\theta_{Z_{n}})\stackrel{<}{\sim}\frac{m}{2}\mathcal{C}_{P|Z_{n}},$$
CP |Zn, (5)
under certain assumptions, such as assuming the parameter posterior ΘP |Zn 181 as a distribution with isotropic covariance and θZn 182 as mean. This connection arises because both metrics are related to the local geometry captured by the FIM at θZn 183 , with Sρ being linked to its maximum eigenvalue and CP |Zn 184 to its trace. Practically, the eigenspectra of the FIM of a neural network are observed 185 to be dominated by a few large eigenvalues (specifically related to the number of classes, C in 186 classification task) while remaining eigenvalues are near zero. This observation indicates that the ratio λmax(F(θ))/Tr(F(θ)) is larger than 1C
187 (C ≪ m). For detailed derivation, please see Appendix B.

## 188 **4.5 Estimating** Sρ(W)

189 Directly computing Sρ(w) requires solving the maximization problem over the high-dimensional 190 parameter perturbation δ. For deep neural networks, finding the exact maximum within the L2-ball 191 of radius ρ is generally intractable. Therefore, we resort to numerical approximation methods. 192 For small perturbations δ, the expected KL divergence can be accurately approximated by a second193 order Taylor expansion involving the Fisher Information Matrix (FIM), F(θ), as Eq. (2) in Section 3 .

Under quadratic approximation, as discussed in Section 4.2, the optimal perturbation δ 194 ∗ = ρvmax, the maximum value is then Sρ(θ) = 12 ρ 2 195 λmax, and the gradient of the approximated KL divergence 196 with respect to δ is F(θ)δ.

197 This connection motivates not an usual Projected Gradient Ascent, that update δk+1 ←
198 Π{δk:∥δk∥≤ρ}(δk + ηF(θ)δk), but an iterative gradient ascent approach that update

$$\delta_{k+1}=\frac{\rho}{\|F(w)\delta_{k}\|}F(w)\delta_{k},\qquad\delta_{0}=\varepsilon\sim\mathcal{N}\left(0,\frac{\sigma^{2}}{m}I_{m}\right),$$

where σ 2 199 is initial noise scale. Iterative gradient ascent is precisely one iteration of the Power Iteration 200 method used to find the dominant eigenvector of F(w).

## Algorithm 1 Estimation Of Sρ(W)

201 1: **Input:** model parameter w ∈ R
m, noise scale σ 2, 2: radius ρ > 0, number of steps K ≥ 1 3: **Initialize** δ0 randomly with N (0, σ 2 m Im)
4: for k = 0 to K − 1 do 5: Compute gk = ∇δEx∼p(x)KL(f(x, θ)∥f(*x, θ*+δ))|δ=δk 6: Update perturbation: δk+1 = ρgk
∥gk∥2 7: **end for**
8: **return** Ex∼p(x)KL(f(x, θ)∥f(*x, θ* + δK))

1 v1 v2 g0 w +
0 w
$$f(x,\theta)\|f(x,\theta+\partial_{K})\}$$
$x\to$. 
$=\underline{-1}$  . 

## 202 **4.5.1 Algorithm For Estimating** Sρ(W)

203 Based on the above, we propose Algorithm 1 to estimate Sρ(w). This algorithm performs K steps of 204 normalized gradient ascent (effectively, Power Iteration under the quadratic approximation) to find an approximate maximizing perturbation δ
∗
205 .

## 206 **4.6 Empirical Results**

10 1 10 0 10 1 S) ( )
0.25 0.30 0.35 0.40 0.45 10 0 10 1 10 2 10 3 10 4 Tr(H)
0.25 0.30 0.35 0.40 0.45 10 1 10 0 10 1 10 2 10 3 max(H)
0.25 0.30 0.35 0.40 0.45 Gene raliz ation Gap Gen eraliza tion Gap Gen erali zatio n Ga p

(a) Sρ. τ = 0.5141
(b) Tr(H). τ *= 0.*5444
(c) λmax(H). τ *= 0.*5175 10 6 10 5 10 4 S ( )
0.05 0.10 0.15 0.20 0.25 0.30 10 2 10 1 10 0 10 1 10 2 10 3 10 4 Tr(H)
0.05 0.10 0.15 0.20 0.25 0.30 10 3 10 2 10 1 10 0 10 1 10 2 10 3 max(H)
0.05 0.10 0.15 0.20 0.25 0.30 gene ralizat ion ga p WD 0.0 0.0001 0.0005 DA False True general ization gap general ization gap
(e) Tr(H). τ = −0.0439
(f) λmax(H). τ = −0.1200
(d) Sρ. τ *= 0.*3658

## 227 **5 Inconsistency-Aware Minimization (Iam): Incorporating Local** 228 **Inconsistency Into The Objective**

229 Our empirical findings suggest that local inconsistency, Sρ(θ) defined in Eq. (3), correlates with the 230 generalization gap. This motivates its use as a regularizer to guide the optimization towards solutions 231 that not only fit the training data, but also exhibit low sensitivity in their output distributions with 232 respect to parameter perturbations. We propose two strategies to incorporate local inconsistency into 233 the training objective. 234 1. **Direct Regularization (IAM-D)**: This approach directly penalizes local inconsistency by adding 235 it to the standard training loss L(θ):
LIAM-D(θ) = L(θ) + βSρ(θ) = L(θ) + β max
∥δ∥2≤ρ EX[KL(f(X, θ)∥f(*X, θ* + δ))], (6)
207 To assess the predictive capability of local inconsistency Sρ for the generalization gap, we conducted 208 experiments on CIFAR-10. We trained two distinct architectures, a 6-layer CNN (6CNN) and a 209 Wide Residual Network (WRN28-2)[32], under various hyperparameter settings (details in Appendix 210 E). Sρ was estimated using a disjoint, unlabeled data set. For comparison, we also computed two 211 common sharpness-based measures: the trace, Tr(H), and the maximum eigenvalue, λmax(H).

212 Figure 1 presents scatter plots of these metrics against the generalization gap, with Kendall's Tau 213 (τ ) reported for each. For the simpler 6CNN model (top row), Sρ (τ = 0.5141) exhibited a 214 positive correlation with the generalization gap, comparable to Tr(H) (τ = 0.5444) and λmax(H)
215 (τ = 0.5175). This suggests that for smaller models, various geometric measures may similarly 216 capture aspects of generalization. However, for the larger WRN28-2 model with data augmentation 217 (bottom row), a more nuanced behavior emerged. As noted by Andriushchenko et al. [1], different 218 training configurations can form distinct solution subgroups. In our WRN28-2 experiments, Tr(H) 219 and λmax(H) showed positive correlations only within such subgroups, but exhibited negative overall 220 correlations globally (τ = −0.0439 and τ = −0.1200, respectively). In stark contrast, our Sρ 221 maintained a positive, albeit reduced, correlation across all settings (τ = 0.3658).

222 This divergence, particularly with larger models and data augmentation, suggests that local incon223 sistency captures information about the generalization gap that is distinct from, or complementary 224 to, traditional Hessian-based sharpness. While the predictive utility of sharpness metrics can be 225 confounded by these subgroup effects, Sρ demonstrates more consistent global predictiveness, hinting 226 at its potential as a more robust generalization indicator in complex training scenarios. 236 where β > 0 is a hyperparameter balancing the trade-off. This objective seeks parameter values θ for 237 which the model outputs are robust across the neighborhood defined by ρ. 238 2. **SAM-like Approach (IAM-S)**: Inspired by SAM [8], this method aims to find parameters θ that 239 reside in a neighborhood of uniformly low loss by minimizing the loss at an adversarially perturbed point θ + δ
∗
240 :
LIAM-S(θ) = L(θ + δ
∗), where δ
∗ = arg max
∥δ∥2≤ρ EX[KL(f(X, θ)∥f(*X, θ* + δ))]. (7)
Here, δ
∗
241 is the perturbation that maximizes the local inconsistency term. Note that the objective 242 minimizes the original loss L at the perturbed point θ + δ:

$\epsilon_{\alpha}\cdot L\left(\theta\right)\to\epsilon$
$$(\mathbf{v}_{J})$$
$\frac{1}{2}$ 2. 
L(θ + δ) ≈ L(θ) + δ
⊤∇θL(θ) + 12 δ
⊤G(θ)δ.

243 Thus, IAM-S implicitly minimizes the principal eigenvalues of G(θ), equivalent to empirical FIM. 244 In the following subsections, we detail the algorithm for IAM-S and provide an analysis of its 245 objective. The algorithm for IAM-D involves a similar inner maximization for Sρ(θ) followed by a 246 standard gradient descent step on LIAM-D(θ).

## 247 **5.1 Algorithm For Iam-D And Iam-S**

248 Optimizing LIAM-S(θ) and LIAM-S(θ) involves a min-max procedure. The inner maximization to find δ
∗(i.e., computing Sρ(θ) and the corresponding δ
∗
249 ) is performed using an Algorithm 1, typically for 250 K = 1 step for efficiency. IAM-D simply add the βSρ(θ) with δK to the L(θ), and then update θ 251 with standard SGD. The outer minimization step of IAM-S updates θ based on the gradient of the 252 loss L(θ + δK) dropping the second-order terms same with SAM: ∇θLIAM-S(θ) ≈ ∇θL(θ)|θ=θ+δK .

253 This two-step process is summarized in Algorithm 2 in Appendix D.

## 254 **5.2 Empirical Evaluation In Supervised Learning**

255 We evaluated the performance of IAM against SGD and SAM in image classification tasks. WRN16256 8[32] served as the baseline model, trained on CIFAR-10, 100 with basic augmentations. Optimal 257 hyperparameters for IAM-D were found to be β = 1.0, ρ = 0.1 for CIFAR-10, and β = 10.0, ρ = 0.1 258 for CIFAR-100, and for IAM-S were ρ = 0.1, 0.5 in CIFAR-10 and CIFAR-100 respectively. Table 259 1 summarizes the test error rates. Both IAM-D (Direct Regularization) and IAM-S (SAM-like 260 Approach) variants not only reduce test error compared to SGD but also achieve performance 261 comparable to SAM. Notably, on CIFAR-100, IAM-S outperforms SAM by a margin of 0.75%, 262 demonstrating its effectiveness in more complex datasets.

Table 1: Test Error (mean ± stderr) of IAM, SAM, and SGD on WRN-16-8 trained with CIFAR-10,

CIFAR-100

SGD SAM IAM-D IAM-S

CIFAR-10 3.95 ±0.048 3.31±0.010 **3.28**±0.060 3.30±0.042

CIFAR-100 19.17±0.192 17.63±0.119 17.16±0.028 **16.88**±0.021

263 Figure 2 illustrates the evolution of local inconsistency Sρ(θ) and test accuracy during training for 264 SGD and IAM-D. IAM-D effectively suppresses the increase in Sρ(θ) and mitigates overfitting, 265 particularly evident after learning rate decay points where test accuracy for SGD can degrade. Both 266 on CIFAR-10, 100 (Figure 2), IAM-D maintains Sρ(θ) below SGD. Although second LR decay 267 temporarily reduces inconsistency for both, SGD's inconsistency quickly rebounds, unlike the stable 268 behavior of IAM-D. These observations suggest that minimizing local inconsistency helps confine the 269 model to parameter regions with smoother output distributions, correlating with the generalization 270 improvements shown in Table 1.

## 271 **5.3 Iam For Learning With Limited Or No Explicit Labels**

272 A key advantage of local inconsistency is its computability from unlabeled data, making IAM well273 suited for scenarios with limited or no explicit supervision. We demonstrate this in semi-supervised 274 and self-supervised learning settings. Detailed settings are listed in Appendix E.

70 75 80 85 90 95 100 30 40 50 60 70 80 90 100 0 50 100 150 200 Train Step 0.02 0.04 0.06 0.08 0.10 0.12 SGD Inconsistency IAM Inconsistency SGD Accuracy IAM Accuracy 0 50 100 150 200 Train Step 0.5 1.0 1.5 2.0 Tes t A
ccuracy Incons iste ncy Test A
ccuracy Incons istenc y

(a) CIFAR-10
(b) CIFAR-100
275 **Semi-Supervised Learning.** In many practical scenarios, labeled data is scarce while unlabeled 276 data is abundant. We simulated semi-supervised learning on CIFAR-10 and CIFAR-100 by masking 277 80% to 99% of training labels. IAM-D was configured to optimize a joint objective: cross-entropy 278 loss on the labeled subset plus the local inconsistency penalty computed over the entire mini-batch 279 (both labeled and unlabeled examples). In contrast, SGD and SAM utilized only the labeled examples. 280 As summarized in Table 2, IAM-D consistently outperforms both baselines across most missing-label 281 ratios, confirming that leveraging unlabeled data via the local inconsistency term enhances robustness 282 when supervision is sparse.

| Dataset   | Model         | Label Rate    |               |               |
|-----------|---------------|---------------|---------------|---------------|
| 1%        | 5%            | 10%           | 20%           |               |
| SGD       | 56.54 ± 0.159 | 28.64 ± 1.648 | 21.93 ± 0.234 | 17.42 ± 0.430 |
| SAM       | 55.83 ± 0.728 | 28.45 ± 0.119 | 20.68 ± 0.102 | 14.00 ± 0.075 |
| IAM-D     | 52.78 ± 0.497 | 25.66 ± 0.723 | 19.44 ± 0.354 | 14.24 ± 0.142 |
| SGD       | 89.35 ± 0.098 | 72.65 ± 0.519 | 60.86 ± 0.204 | 50.01 ± 0.299 |
| SAM       | 89.31 ± 0.156 | 72.02 ± 0.337 | 58.04 ± 0.594 | 45.64 ± 0.085 |
| IAM-D     | 88.36 ± 0.292 | 69.99 ± 0.649 | 57.60 ± 0.251 | 45.05 ± 0.956 |

283 **Self-Supervised Learning (SSL).** The label284 agnostic nature of IAM makes it directly applicable 285 to SSL objectives. We integrated IAM-D into the 286 SimCLR framework [6], training a ResNet-18[11]
287 encoder on CIFAR-10. Performance was evaluated 288 via linear probing. The local inconsistency term for 289 IAM-D was computed using the model's projection290 head outputs. Figure 3 shows that SimCLR trained 291 with IAM-D (SimCLR-IAM) achieves higher test ac292 curacy on the downstream linear classification task 293 compared to vanilla SimCLR (SimCLR-SGD). Fur294 thermore, SimCLR-IAM tends to converge faster in 295 terms of test error and also minimizes the SimCLR 296 training loss more rapidly, despite the additional lo297 cal inconsistency regularization. This suggests that 298 controlling local inconsistency is beneficial even 299 when no explicit labels are available during repre300 sentation learning.

0 50 100 150 200 Epoch 40 50 60 70 80 6.0 6.5 7.0 Test A
ccura cy (%
)

IAM (accuracy) SGD (accuracy)
IAM (loss)
SGD (loss)
Los s

## 301 **5.4 Iam As Implicit Output Entropy Regularization**

302 Improving generalization in deep neural networks often involves modulating the entropy of model 303 output distributions. Techniques such as Label Smoothing (LS) [31, 23, 4] and Entropy Regularization 304 (ER) [27, 5, 34] mitigate the common issue of overconfidence of cross-entropy loss minimization, by 305 specifically targeting this output entropy.

306 Our Inconsistency-Aware Minimization (IAM) framework regularizes local inconsistency, Sρ(θ). To 307 understand its regularizing effect, we can leverage the identity KL(P∥Q) = −H(P) + CE(P, Q), 308 where H(P) is the entropy of P and CE(*P, Q*) is cross entropy of P and Q. Since the expectation is 309 linear and H(f(x; θ)) does not depend on δ, we can rewrite Sρ as:

$$S_{\rho}(\theta)=-\mathbb{E}_{x\sim p(x)}[H(f(x,\theta))]+\operatorname*{max}_{\|\delta\|_{2}\leq\rho}\mathbb{E}_{x\sim p(x)}[\mathrm{CE}(f(x,\theta),f(x,\theta+\delta))].$$
$$({\boldsymbol{\delta}})$$

310 Minimizing Sρ(θ) as defined in Eq. (8) thus involves two concurrent objectives. First, maximizing 311 the expected output entropy EX[H(f(*x, θ*))]. This discourages overconfident (low-entropy) pre312 dictions. Second, Minimizing the worst-case expected cross-entropy between the original model's 313 output f(*x, θ*) and the perturbed model's output f(*x, θ* + δ), which promotes stability of the output 314 distribution under parameter perturbations. This dual objective resonates with the goals of established 315 regularization techniques like LS and ER. LS effectively increases the entropy of the target distribu316 tion, thereby preventing the model from becoming overly confident in its predictions for the training 317 labels. ER directly penalizes the low-entropy output distributions of the model. Formally, LS can 318 be interpreted as minimizing a KL divergence to a smoothed target (e.g., KL(qLS(y|x)∥p(y|x; θ))),
319 while ER often involves minimizing KL(p(y|x; θ)∥u), while u is a uniform distribution, both encour320 aging the model to avoid excessive output certainty.

321 The first term in Eq. (8) (−EX[H(f(x, θ))]) directly aligns with the aim of ER to penalize low322 entropy outputs. The second term, by minimizing the "distance" (via CE) to the output of a perturbed 323 model, enforces a form of local distributional stability. If highly confident predictions are less stable 324 in parameter perturbation, then minimizing Sρ(θ) would implicitly penalize such an overconfidence. 325 Therefore, minimizing local inconsistency Sρ(θ) acts as a regularization strategy that not only 326 promotes prediction stability against minor parameter variations but also indirectly encourages higher 327 output entropy, akin to mechanisms in LS and ER. This multifaceted regularization is anticipated to 328 yield improved generalization by encouraging the model to learn more robust and less overconfident 329 representations, making them less susceptible to training data idiosyncrasies or parameter instabilities 330 [34, 8].

## 331 **6 Conclusion**

332 In this work, we introduced "local inconsistency," a novel information-geometric generalization 333 measure computable from a single model using only unlabeled data. We theoretically linked it to the 334 Fisher Information Matrix (FIM) and the loss Hessian. Empirically, local inconsistency correlates 335 with the generalization gap and exhibits distinct characteristics from traditional sharpness-based 336 metrics. 337 Based on this, we proposed Inconsistency-Aware Minimization (IAM), an optimization framework 338 that directly incorporates local inconsistency into the training objective. IAM enhances generalization 339 in supervised learning, matching or exceeding that of Sharpness-Aware Minimization (SAM). Cru340 cially, IAM proves effective in semi- and self-supervised learning by leveraging unlabeled data for 341 local inconsistency computation, improving performance in label-scarce settings. We also elucidated 342 IAM's mechanism as an implicit regularizer of model output entropy. 343 These findings offer a practical and theoretically-grounded approach to improving model gener344 alization, particularly valuable in real-world applications where labeled data is limited. Future 345 research could focus on a more rigorous establishment of the theoretical relationship between local 346 inconsistency and generalization bounds (such as the informal bound presented in Theorem |1) and 347 on exploring the scalability and applicability of IAM to a wider array of model architectures and 348 large-scale datasets.

## 349 **References**

350 [1] M. Andriushchenko, F. Croce, M. Müller, M. Hein, and N. Flammarion. A modern look at the 351 relationship between sharpness and generalization, 2023. URL https://arxiv.org/abs/
352 2302.07011. 353 [2] D. Arpit, S. Jastrz˛ebski, N. Ballas, D. Krueger, E. Bengio, M. S. Kanwal, T. Maharaj, A. Fischer, 354 A. Courville, Y. Bengio, and S. Lacoste-Julien. A closer look at memorization in deep networks, 355 2017. URL https://arxiv.org/abs/1706.05394. 356 [3] P. L. Bartlett, D. J. Foster, and M. J. Telgarsky. Spectrally-normalized margin bounds for neural 357 networks. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and 358 R. Garnett, editors, *Advances in Neural Information Processing Systems*, volume 30. Curran 359 Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper_files/paper/ 360 2017/file/b22b257ad0519d4500539da3c8bcf4dd-Paper.pdf. 361 [4] K. Chandrasegaran, N.-T. Tran, Y. Zhao, and N.-M. Cheung. Revisiting label smoothing and 362 knowledge distillation compatibility: What was missing?, 2022. URL https://arxiv.org/ 363 abs/2206.14532. 364 [5] P. Chaudhari, A. Choromanska, S. Soatto, Y. LeCun, C. Baldassi, C. Borgs, J. Chayes, L. Sagun, 365 and R. Zecchina. Entropy-sgd: Biasing gradient descent into wide valleys, 2017. URL 366 https://arxiv.org/abs/1611.01838. 367 [6] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton. A simple framework for contrastive learning 368 of visual representations, 2020. URL https://arxiv.org/abs/2002.05709. 369 [7] L. Dinh, R. Pascanu, S. Bengio, and Y. Bengio. Sharp minima can generalize for deep nets. In 370 D. Precup and Y. W. Teh, editors, *Proceedings of the 34th International Conference on Machine* 371 *Learning*, volume 70 of *Proceedings of Machine Learning Research*, pages 1019–1028. PMLR, 372 06–11 Aug 2017. URL https://proceedings.mlr.press/v70/dinh17b.html. 373 [8] P. Foret, A. Kleiner, H. Mobahi, and B. Neyshabur. Sharpness-aware minimization for efficiently 374 improving generalization, 2021. URL https://arxiv.org/abs/2010.01412. 375 [9] T. Garipov, P. Izmailov, D. Podoprikhin, D. P. Vetrov, and A. G. Wilson. Loss surfaces, mode con376 nectivity, and fast ensembling of dnns. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, 377 N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Sys378 *tems*, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/
379 paper_files/paper/2018/file/be3087e74e9100d4bc4c6268cdbe8456-Paper.pdf. 380 [10] M. Hardt, B. Recht, and Y. Singer. Train faster, generalize better: Stability of stochastic 381 gradient descent. In M. F. Balcan and K. Q. Weinberger, editors, *Proceedings of The 33rd* 382 *International Conference on Machine Learning*, volume 48 of *Proceedings of Machine Learning* 383 *Research*, pages 1225–1234, New York, New York, USA, 20–22 Jun 2016. PMLR. URL
384 https://proceedings.mlr.press/v48/hardt16.html.

385 [11] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition, 2015. URL
386 https://arxiv.org/abs/1512.03385.

387 [12] C. Jang, S. Lee, F. Park, and Y.-K. Noh. A reparametrization-invariant sharpness measure based 388 on information geometry. *Advances in neural information processing systems*, 35:27893–27905, 389 2022. 390 [13] Y. Jiang, B. Neyshabur, H. Mobahi, D. Krishnan, and S. Bengio. Fantastic generalization 391 measures and where to find them, 2019. URL https://arxiv.org/abs/1912.02178.

392 [14] Y. Jiang, V. Nagarajan, C. Baek, and J. Z. Kolter. Assessing generalization of sgd via disagree393 ment, 2022. URL https://arxiv.org/abs/2106.13799.

394 [15] R. Johnson and T. Zhang. Inconsistency, instability, and generalization gap of deep neural 395 network training, 2023. URL https://arxiv.org/abs/2306.00169.

396 [16] N. S. Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, and P. T. P. Tang. On large-batch 397 training for deep learning: Generalization gap and sharp minima, 2017. URL https://arxiv.

398 org/abs/1609.04836. 399 [17] M. Kim, D. Li, S. X. Hu, and T. M. Hospedales. Fisher sam: Information geometry and 400 sharpness aware minimisation, 2022. URL https://arxiv.org/abs/2206.04920.

401 [18] A. Krizhevsky. Learning multiple layers of features from tiny images. Technical 402 report, University of Toronto, 2009. URL https://www.cs.toronto.edu/~kriz/ 403 learning-features-2009-TR.pdf.

404 [19] J. Kwon, J. Kim, H. Park, and I. K. Choi. Asam: Adaptive sharpness-aware minimization for 405 scale-invariant learning of deep neural networks. In M. Meila and T. Zhang, editors, *Proceedings* 406 *of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of* 407 *Machine Learning Research*, pages 5905–5914. PMLR, 18–24 Jul 2021. URL https:// 408 proceedings.mlr.press/v139/kwon21b.html.

409 [20] H. Li, Z. Xu, G. Taylor, C. Studer, and T. Goldstein. Visualizing the loss landscape of neural 410 nets. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, 411 editors, *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, 412 Inc., 2018. URL https://proceedings.neurips.cc/paper_files/paper/2018/file/ 413 a41b3bb3e6b050b6c9067c67f663b915-Paper.pdf.

414 [21] S. Mandt, M. D. Hoffman, and D. M. Blei. Stochastic gradient descent as approximate bayesian 415 inference, 2018. URL https://arxiv.org/abs/1704.04289. 416 [22] J. Martens. New insights and perspectives on the natural gradient method. *Journal of Machine* 417 *Learning Research*, 21(146):1–76, 2020. URL http://jmlr.org/papers/v21/17-678. 418 html. 419 [23] R. Müller, S. Kornblith, and G. Hinton. When does label smoothing help?, 2020. URL 420 https://arxiv.org/abs/1906.02629. 421 [24] V. Nagarajan and J. Z. Kolter. Uniform convergence may be unable to explain generalization 422 in deep learning. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and 423 R. Garnett, editors, *Advances in Neural Information Processing Systems*, volume 32. Curran 424 Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper_files/paper/ 425 2019/file/05e97c207235d63ceb1db43c60db7bbb-Paper.pdf. 426 [25] P. Nakkiran, G. Kaplun, Y. Bansal, T. Yang, B. Barak, and I. Sutskever. Deep double descent: 427 where bigger models and more data hurt*. *Journal of Statistical Mechanics: Theory and* 428 *Experiment*, 2021(12):124003, dec 2021. doi: 10.1088/1742-5468/ac3a74. URL https: 429 //dx.doi.org/10.1088/1742-5468/ac3a74. 430 [26] R. Pascanu and Y. Bengio. Revisiting natural gradient for deep networks, 2014. URL https: 431 //arxiv.org/abs/1301.3584. 432 [27] G. Pereyra, G. Tucker, J. Chorowski, Łukasz Kaiser, and G. Hinton. Regularizing neural 433 networks by penalizing confident output distributions, 2017. URL https://arxiv.org/abs/
434 1701.06548. 435 [28] S. Santurkar, D. Tsipras, A. Ilyas, and A. Madry. How does batch normalization help optimiza436 tion? In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, 437 editors, *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, 438 Inc., 2018. URL https://proceedings.neurips.cc/paper_files/paper/2018/file/
439 905056c1ac1dad141560467e0a99e1cf-Paper.pdf. 440 [29] D. Soudry, E. Hoffer, M. S. Nacson, S. Gunasekar, and N. Srebro. The implicit bias of gradient 441 descent on separable data. *Journal of Machine Learning Research*, 19(70):1–57, 2018. URL
442 http://jmlr.org/papers/v19/18-188.html.

443 [30] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: a simple 444 way to prevent neural networks from overfitting. *J. Mach. Learn. Res.*, 15(1):1929–1958, Jan. 445 2014. ISSN 1532-4435. 446 [31] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architec447 ture for computer vision, 2015. URL https://arxiv.org/abs/1512.00567.

448 [32] S. Zagoruyko and N. Komodakis. Wide residual networks, 2017. URL https://arxiv.org/ 449 abs/1605.07146. 450 [33] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals. Understanding deep learning requires 451 rethinking generalization, 2017. URL https://arxiv.org/abs/1611.03530. 452 [34] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz. mixup: Beyond empirical risk mini453 mization, 2018. URL https://arxiv.org/abs/1710.09412. 454 [35] J. Zhuang, B. Gong, L. Yuan, Y. Cui, H. Adam, N. Dvornek, S. Tatikonda, J. Duncan, and 455 T. Liu. Surrogate gap minimization improves sharpness-aware training, 2022. URL https: 456 //arxiv.org/abs/2203.08065.

## 457 **A Theoretical Analysis: Generalization Bound With Local Inconsistency**

458 In this section, we provide a theoretical sketch to connect our proposed local inconsistency measure, 459 Sρ(θ), to the generalization error of a model. Our goal is to show that minimizing Sρ(θ), along with 460 the empirical loss, can lead to a tighter upper bound on the true risk, thereby offering a theoretical 461 motivation for our Inconsistency-Aware Minimization (IAM) approach We adapt insights from PAC- 462 Bayesian theory, particularly drawing from the analysis of Sharpness-Aware Minimization (SAM) 463 [8].

Let LD (θ) = E(X,Y )∼D [l(f(X, θ), Y )] be the true risk and LS (θ) = 1n Pn i=1 l(f(Xi 464 , θ), Yi) be the 465 empirical risk on a training set S of size n. Foret et al. [8] provide a PAC-Bayesian generalization 466 bound (informally stated, see their Appendix A for the full theorem [Thm 1, 8]) which, with high 467 probability 1 − ξ over the draw of S, states:

$$L_{\mathcal{B}}(\theta)\leq\operatorname*{max}_{\|\epsilon\|_{2}\leq\rho}L_{\mathcal{S}}(\theta+\epsilon)+\mathcal{R}(\theta,\rho,n,m,\xi)$$

where R(*θ, ρ, n, m, ξ*) is a complexity term that depends on the norm of the parameters ∥θ∥
22, the perturbation radius ρ, the number of training samples n, the number of parameters m, and the confidence ξ. Specifically,

$${\mathcal{R}}(\theta,\rho,n,m,\xi)={\sqrt{\frac{m\log\left(1+{\frac{\|\theta\|_{2}^{2}}{\rho^{2}}}\left(1+{\sqrt{\frac{\log n}{m}}}\right)^{2}\right)+4\log{\frac{n}{\xi}}+{\tilde{O}}(1)}}}.$$
$$(9)$$
$$(10)$$

$$(11)$$
$$(12)$$

468 The first term on the right-hand side of Eq. (9) can be rewritten as:

$$\operatorname*{max}_{\|\epsilon\|_{2}\leq\rho}L_{\mathcal{S}}(\theta+\epsilon)=L_{\mathcal{S}}(\theta)+\left[\operatorname*{max}_{\|\epsilon\|_{2}\leq\rho}\left(L_{\mathcal{S}}(\theta+\epsilon)-L_{\mathcal{S}}(\theta)\right)\right].$$
(LS (θ + ϵ) − LS (θ)). (10)
469 The term in the square brackets, let's call it *SHARP*ρ(θ), represents the sharpness of the loss 470 landscape at θ within a ρ-neighborhood, as considered by SAM. Our aim is to relate this sharpness 471 term to our proposed local inconsistency measure Sρ(θ).

472 Recall our definition of local inconsistency (calculated with respect to the empirical distribution over 473 the training set S for this analysis):

$$S_{\rho}(\theta)=\operatorname*{max}_{\|\delta\|_{2}\leq\rho}\mathbb{E}_{X\sim\mathcal{S}}[\mathrm{KL}(f(X,\theta)\|f(X,\theta+\delta))].$$

474 For the cross-entropy loss l(f(X, θ), Y ) = − log f(X, θ)Y , under certain conditions, particularly 475 near a good minimizer where the model's predictions f(*X, θ*) are close to the true label distribution 476 (or if we consider f(X, θ) as a reference distribution), the change in loss due to parameter perturbation 477 can be related to the KL divergence of the output distributions. Specifically, for a single sample 478 (*X, Y* ), a second-order Taylor expansion of the loss l(f(X, θ + δ), Y ) around δ = 0 gives:

$${\cal L}_{S}(\theta+\delta)-{\cal L}_{S}(\theta)=\nabla_{\theta}{\cal L}_{S}(\theta)^{\top}\delta+\frac{1}{2}\delta^{\top}\nabla_{\theta}^{2}{\cal L}_{S}(\theta)\delta+O(\|\delta\|^{3}).$$
3). (12)
479 If θ is a point where ∇θLS (θ) ≈ 0 (i.e., near a minimum of the empirical risk), the first-order term in 480 the empirical average LS (θ + δ) − LS (θ) becomes small. Furthermore, as discussed in Section 4.2
(and your Background section), for cross-entropy loss, the Hessian ∇2
481 θLS (θ) can be approximated
482 by the empirical Fisher Information Matrix (or Gauss-Newton matrix) FS (θ). Also, we know that
EX∼S [KL(f(X, θ)∥f(*X, θ* + δ))] ≈
1 2
δ
⊤ 483 FS (θ)δ for small δ.
484 **Assumption (Approximate Equivalence of Loss Increase and Output KL-Divergence):** We 485 heuristically assume that, for well-trained models near a local minimum, the worst-case increase in 486 empirical loss due to parameter perturbation is approximately proportional to the local inconsistency
487 (our Sρ(θ) defined on the empirical data S):
$$\operatorname*{max}_{\|\epsilon\|_{2}\leq\rho}\left(L_{S}(\theta+\epsilon)-L_{S}(\theta)\right)\approx c\cdot S_{\rho}(\theta)$$
(LS (θ + ϵ) − LS (θ)) ≈ c · Sρ(θ) (13)
488 for some positive constant c, which may depend on factors like the temperature of a Gibbs distribution 489 if one were to formally link the loss to KL divergence from a Bayesian perspective (e.g., c = 1 if the

$$(13)$$

490 loss essentially measures KL divergence to empirical targets). This assumption relies on the idea that 491 changes in model output distributions (measured by KL divergence) are primary drivers of changes 492 in the CE loss, especially for worst-case perturbations. While this is an approximation, it captures the 493 intuition that models whose outputs are highly sensitive to parameter changes (high Sρ(θ)) are likely 494 to experience larger increases in loss under such perturbations. 495 Substituting Eq. (13) into Eq. (10), and then into Eq. (9), we obtain the following generalization 496 bound:
LD (θ) ≲ LS (θ) + c · Sρ(θ) + R(*θ, ρ, n, m, ξ*) (14)
497 where ≲ indicates that the inequality relies on the approximation in Eq. (13). 498 **Interpretation and Implications.** The bound in Eq. (14) suggests that the true risk LD (θ) is 499 upper-bounded by the sum of the empirical risk LS (θ), our local inconsistency measure Sρ(θ) (scaled 500 by a constant c), and the PAC-Bayesian complexity term R. This provides a theoretical rationale for 501 our IAM procedure: by minimizing an objective that includes both the empirical loss and the local 502 inconsistency Sρ(θ) (as in IAM-Direct, or implicitly by IAM-S seeking regions where Sρ(θ) allows LS (θ + δ
∗
503 ) to be low), we are effectively attempting to minimize this upper bound on the true risk.

504 A smaller Sρ(θ) contributes to a tighter bound, potentially leading to better generalization.

505 The constant c and the tightness of the approximation in Eq. (13) warrant further investigation.

506 However, this sketch provides a plausible pathway to connect Sρ(θ) with generalization guarantees 507 by leveraging existing PAC-Bayesian frameworks that deal with loss landscape geometry. Rigorously 508 establishing the relationship in Eq. (13) or deriving a similar bound with Sρ(θ) appearing more 509 directly through the KL term in a PAC-Bayes analysis (perhaps by defining a posterior whose "spread" 510 is related to Sρ(θ)) are important directions for future theoretical work.

## 511 **B Relation Between Our Metric And Inconsistency**

512 This section outlines an approximate derivation relating the model output inconsistency CP , as defined 513 by Johnson and Zhang [15], to the local sensitivity metric Sρ(w) defined previously. we will show 514 simple demonstrations that these two metrics are related primarily through the Fisher Information 515 Matrix (FIM), under specific assumptions like isotropic covariance. Then will show results with 516 anisotropic covariance.

## 517 **Definitions**

518 - **Inconsistency (**CP ): Measures the average difference (in terms of KL divergence) between 519 the outputs of models generated by a stochastic training procedure P applied to the same 520 training data Zn. The average is taken over draws of the training data Zn and pairs of models (Θ, Θ′) drawn from the conditional distribution ΘP |Zn 521 .

CP = EZn EΘ,Θ′∼ΘP |Zn EX[KL(f(Θ, X)∥f(Θ′, X))]
Here, ΘP |Zn 522 denotes the distribution over parameters resulting from applying procedure P
523 to dataset Zn. 524 - **Local Sensitivity (**Sρ(w)): Measures the expected maximum change in the model's output 525 distribution within a ρ-radius ball around a specific parameter vector w. For consistency 526 with the derivation below, we use the form where the expectation is inside the maximization.

$$S_{\rho}(w)=\operatorname*{max}_{\|\delta\|_{2}\leq\rho}\mathbb{E}_{X}[\operatorname{KL}(f(X,w+\delta)\|f(X,w))]$$

Here, δ ∈ R
d 527 is a perturbation to the parameters w.

529 1. **Isotropic Covariance Posterior Assumption**: For a given training set Zn, the conditional parameter distribution ΘP |Zn 530 can be approximated by an isotropic distribution centered at a specific parameter vector wZn derived from Zn: E[ΘP |Zn] = wZn, Cov[ΘP |Zn] = s 2 531 Id, where s 2 532 is a small variance. This approximation is motivated by studies interpreting 528 **Assumptions** The following derivation relies on several key assumptions: 533 Stochastic Gradient Descent (SGD) as a form of approximate Bayesian inference, where the 534 distribution of parameters after training can resemble a Gaussian centered near a mode of a 535 posterior distribution related to the loss function [21]. 536 2. **Validity of Second-Order KL Approximation**: The KL divergence between outputs of 537 models with slightly different parameters can be accurately approximated by a quadratic 538 form involving the Fisher Information Matrix (FIM). This relies on the parameter difference being small, implying s 539 2 must be small.

3. **Effective FIM Constancy in Expectation**: The variations of the FIM F(Θ′
540 ) for Θ′ ∼
N (wZn, s2Id) around F(wZn 541 ) are assumed to average out sufficiently within the expectation required to calculate CP |Zn. This allows the approximation CP |Zn ≈ s 2Tr(F(wZn 542 )).

543 **Approximation of** CP We first consider the conditional inconsistency for a fixed Zn, denoted CP |Zn, by removing the outer expectation EZn 544 :

$${\bf\Phi}={\bf\Phi},{\bf\Theta}^{\prime}$$

CP |Zn = EΘ,Θ′∼ΘP |Zn EX[KL(f(Θ, X)∥f(Θ′, X))]
Applying the isotropic covariance posterior assumption, Θ = wZn + δ and Θ′ = wZn + δ
′
545 , where δ, δ′are independent perturbations (E[δ] = E[δ
′] = 0, Cov[δ] = Cov[δ
′] = s 2 546 Id).

CP |Zn ≈ Eδ,δ′EX[KL(f(wZn + δ, X)∥f(wZn + δ
′, X))]
547 Using the second-order Taylor expansion for KL divergence taking the expectation over X, valid for small ∥δ − δ
′∥ (i.e., small s 2 548 ):

$$\mathbb{E}_{X}[\mathrm{KL}(f(w_{Z_{n}}+\delta,X)\|f(w_{Z_{n}}+\delta^{\prime},X))]=\frac{1}{2}(\delta-\delta^{\prime})^{T}F(w_{Z_{n}}+\delta^{\prime})(\delta-\delta^{\prime})+O(\|\delta\|^{3})$$

Let u = Θ − Θ′ = δ − δ
′. Since *δ, δ*′are independent, u ∼ N (0, 2s 2 549 Id). Substituting this into the expression for CP |Zn 550 :

$$\mathcal{C}_{P|Z_{n}}=\mathbb{E}_{u}\left[\frac{1}{2}u^{T}F(\Theta^{\prime})u\right]+O(\|\delta\|^{3})$$ $$=\mathbb{E}_{u}\left[\frac{1}{2}u^{T}F(w_{Z_{n}})u\right]+O(\|\delta\|^{3})\quad\text{(FIM Constancy in Expectation Assumption)}$$ $$=\frac{1}{2}\text{Tr}(\text{Cov}(u)F(w_{Z_{n}}))+\frac{1}{2}\mathbb{E}[u]^{T}F(w_{Z_{n}})\mathbb{E}[u]+O(\|\delta\|^{3})$$ $$=\frac{1}{2}\text{Tr}(2s^{2}\text{I}_{d}F(w_{Z_{n}}))+O+O(\|\delta\|^{3})\quad(\mathbb{E}[u]=0)$$ $$\approx s^{2}\text{Tr}(F(w_{Z_{n}}))$$  Thus, the conditional inconsistency for a fixed $Z_{n}$ is approximately proportional to the trace of the 
FIM evaluated at $w_{Z_{n}}$ :  $$\mathcal{C}_{P|Z_{n}}\approx s^{2}\mathrm{Tr}(F(w_{Z_{n}}))\tag{15}$$  The overall inconsistency $\mathcal{C}_{P}$ is the expectation of this quantity over $Z_{n}$: $\mathcal{C}_{P}\approx\mathbb{E}_{Z_{n}}[s^{2}\mathrm{Tr}(F(w_{Z_{n}}))]$.  
Approximation of Sρ(wZn 554 ) Applying the same second-order KL approximation to the definition of Sρ(wZn 555 ):

$$S_{\rho}(w_{Z_{n}})=\operatorname*{max}_{\|\delta\|_{2}\leq\rho}{\frac{1}{2}}\delta^{\top}F(w_{Z_{n}})\delta+O(\|\delta\|^{3})$$

The maximum value of the quadratic form δ 556 T Aδ for a positive semi-definite matrix A subject to 557 ∥δ∥2 ≤ ρ is achieved when δ is aligned with the eigenvector corresponding to the largest eigenvalue 558 (λmax(A)) and has norm ρ. Thus:

$$S_{\rho}(w_{Z_{n}})=\frac{1}{2}\rho^{2}\lambda_{\operatorname*{max}}(F(w_{Z_{n}}))$$
2λmax(F(wZn)) (16)
559 This shows that the local sensitivity Sρ is approximately proportional to the largest eigenvalue of the 560 FIM.

$$(16)$$

Connecting CP |Zn and Sρ(wZn 561 ) For a d × d positive semi-definite matrix A, the relationship between its trace and largest eigenvalue is given by 1 562 dTr(A) ≤ λmax(A) ≤ Tr(A). Applying this to the FIM F(wZn 563 ):

$${\frac{1}{d}}\mathrm{Tr}(F(w_{Z_{n}}))\leq\lambda_{\operatorname*{max}}(F(w_{Z_{n}}))\leq\mathrm{Tr}(F(w_{Z_{n}}))$$

Substituting this into the approximation for Sρ(wZn 564 ) from Eq. (16):

$${\frac{\rho^{2}}{2d}}\mathrm{Tr}(F(w_{Z_{n}}))\leq S_{\rho}(w_{Z_{n}})\leq{\frac{\rho^{2}}{2}}\mathrm{Tr}(F(w_{Z_{n}}))$$

Let's assume a plausible connection, for instance, s 2 = ρ 2 565 /d. Substituting this into the approximation for CP |Znfrom Eq. (15), we get CP |Zn ≈
ρ 2 d Tr(F(wZn 566 )). Combining this with the bounds for Sρ(wZn 567 ):

$$\frac{1}{2}\left(\frac{\rho^{2}}{d}\mathrm{Tr}(F(w_{Z_{n}}))\right)\leq S_{\rho}(w_{Z_{n}})\leq\frac{d}{2}\left(\frac{\rho^{2}}{d}\mathrm{Tr}(F(w_{Z_{n}}))\right)$$

568 This leads to the final approximate relationship between the conditional inconsistency (for a fixed Zn) and the local sensitivity (at the corresponding wZn 569 ):

$$\frac{1}{2}{\mathcal{C}}_{P|Z_{n}}\leq S_{\rho}(w_{Z_{n}})\leq\frac{d}{2}{\mathcal{C}}_{P|Z_{n}}$$
$$(17)$$
CP |Zn(17)
This result suggests that, under the stated assumptions, the conditional inconsistency CP |Zn 570 and the local sensitivity Sρ(wZn 571 ) are approximately proportional, with the proportionality factor potentially 572 depending on the parameter dimension d.

 #### anisotropic covariance  Let $\text{Cov}[\Theta_{P|Z_n}]=s^2\Sigma$, where $s^2=\frac{e^2}{d}.$  Starting from $\mathcal{C}_{P|Z_n}=\frac{1}{2}\text{Tr}(\Sigma F(w_{Z_n}))$,
$$\begin{array}{c}{{\lambda_{m i n}(\Sigma)\mathrm{Tr}(F)\leq\mathrm{Tr}(\Sigma F)\leq\lambda_{\operatorname*{max}}(\Sigma)\mathrm{Tr}(F)}}\\ {{\lambda_{m i n}(\Sigma)\lambda_{\operatorname*{max}}(F)\leq\mathrm{Tr}(\Sigma F)\leq\lambda_{\operatorname*{max}}(\Sigma)d\lambda_{\operatorname*{max}}(F)}}\\ {{\frac{\rho^{2}}{\lambda_{\operatorname*{max}}(\Sigma)}\mathrm{Tr}(\Sigma F)\leq\frac{\rho^{2}}{2}\lambda_{\operatorname*{max}}(F)\leq\frac{\rho^{2}}{2\lambda_{m i n}(\Sigma)}\mathrm{Tr}(\Sigma F)}}\\ {{\frac{1}{\lambda_{\operatorname*{max}}(\Sigma)}\mathcal{C}_{P|Z_{n}}\leq S_{\rho}(w_{Z_{n}})\leq\frac{d}{\lambda_{m i n}(\Sigma)}\mathcal{C}_{P|Z_{n}}}}\end{array}$$
 #### anisotropic covariance  $\mathbf{L}=\frac{1}{2}\text{Tr}(\boldsymbol{\Sigma}F(w_{Z_n})),$  ×
$$\lambda_{\mathrm{m}}$$
$$\frac{\rho^{2}}{2d\lambda_{\mathrm{max}}}$$
573 **Practical Considerations: Eigenvalue Spectrum of Neural Networks** In practice, for deep 574 learning models, the FIM often exhibits a sparse eigenvalue spectrum: many eigenvalues are close to 575 zero, and only a few are significantly large. In such cases:
- The trace Tr(F) = Pλi 576 is dominated by the sum of the few large eigenvalues.

- The ratio λmax(F)/Tr(F) might be closer to 1/m′
577 than 1/d, where m′ ≪ d is the "effective 578 rank" or number of dominant eigenvalues.

579 This implies that the bounds relating λmax(F) and Tr(F) might be tighter than the general 1/d and 1 factors suggest. Consequently, the relationship between CP |Zn 580 (related to trace) and Sρ (related to max eigenvalue) could be closer to direct proportionality than Eq. (5) indicates, especially if s 2 581 is appropriately related to ρ 2 582 . 583 **Summary and Limitations** This analysis provides a heuristic argument suggesting a connection between conditional inconsistency CP |Zn and local sensitivity Sρ(wZn 584 ). Under assumptions of a Gaussian posterior, small variance s 2 585 , validity of second-order KL approximations, local FIM
constancy, and a specific link between s 2and ρ 2(e.g., s 2 = ρ 2/d), we find that Sρ(wZn 586 ) is approximately proportional to CP |Zn 587 , potentially up to a factor related to dimension d. This connection 588 is mediated by the trace and the maximum eigenvalue of the Fisher Information Matrix. The practical 589 observation of sparse FIM eigenvalues might strengthen this relationship.

## 590 **C Decision Boundary Of Neural Networks And Principal Eigenspace Of Fim**

591 To intuitively analysis the role of δ1 in training of neural network, we conducted experiments using 592 3-layer fully-connected neural network on two-dimensional synthetic data. the data is generated 593 from a mixture of three Gaussian distributions, a setup analogous to that employed by [12] in their 594 investigation of the characteristic of the FIM eigensubspace. Their work demonstrated that perturbing 595 parameters along the principal eigenvectors of the FIM can lead to significant modifications in the decision boundary, such as increasing or decreasing the margins of specific classes.

1 Model Model +

Model 1 Model Model +

Model 1 Model Model +

Model 1 Model Model +

Model 0 0 0 0 1 0 1 1 1 0 1 1 1 0 1 1 1 0 1 1
(d) Decision boundary perturbed by ε Figure 4: A synthetic classification example. the black, blue, orange lines correspond to decision boundaries of the NN with trained parameter values, and parameter values perturbed by δ1. Each plot use different noise.

596 597 Our investigation focuses on whether δ1, despite being derived from only a single gradient step 598 (as described in Algorithm 1) and thus influenced by an initial random noise vector ε, still induces 599 substantial changes in the neural network's decision boundary. Figure 4 visualizes these effects. 600 The black lines in each subfigures depict the original decision boundary obtained with the trained 601 parameters w. Figure 4 (a-c) show the perturbed decision boundaries (blue and orange lines) when 602 distinct ±δ1 with ρ = 0.5 is added to w. Each of these δ1 vectors was computed using a different 603 random initialization noise vector, denoted as ε1, ε2, and ε3, respectively. For a direct comparison of 604 the pertubation's nature, Figure 4(d) illustrates the decision boundary perturbed by directly adding 605 the random noise vector ε to w. This vector ε is sampled from same distribution as initial vectors 606 (e.g.ε1) and, is scaled to ∥ε∥2 = ρ same with δ1. As observed in Figure 4 (d), direct perturbation with 607 such an arbitrary random noise vector does not meaning fully alter the decision boundary, even when 608 its norm is equivalent to that of the δ1. This is sharply opposed with the significant changes induced 609 by δ1 perturbations shown in Figures 4 (a-c), underscoring that the direction derived by Algorithm 610 1, even in a single step, is substantially more influential than arbitrary noise of the same magnitude.

611 This result intuitively suggest that the perturbation δ1 with single gradient step still meaningful and 612 aligning with principle eigen vectors of FIM.

613 To investigate the alignment between the single-step perturbation vector δ1 and principle eigenspace 614 of FIM, we explicitly calculate the FIM and its top three eigenvector v1, v2, and v3, corresponding 615 to largesst eigenvalues λ1 > λ2 > λ3. The perturbation δ1, results from one normalized gradient 616 ascent step applied to the KL divergence objective, starting from an initial random noise ε. In terms 617 of power iteration algorithm, the δ1 after first iteration without normalization, is sum of eigenvector of FIM weighted by λiαi 618 .

Formally, let the initial random noise ε be expressed in the eigenbasis of F(w) as ε =Pm i αivi 619 .

ε ∼ N (0, σ2Im), then the coefficient αi are i.i.d. as N (0, σ2 620 ) since {vi}form an orthonormal basis.

$$F(\theta)\varepsilon=\sum_{i}^{m}\lambda_{i}v_{i}v_{i}^{\top}\sum_{i}^{m}$$
αivi
=X
$$\sum^{m}\lambda_{i}\alpha_{i}v_{i}$$
i
So cosine similarity between δ1 and viis λiαi. And ∥P3 i δ T
1vi∥
∥δ1∥
621 , which indicates how much the δ1 is in principle eigen space, {u|u = av1 + bv2 + cv3*, abc* ∈ [0, 1]} of FIM, is ∥P3 i αiλivi∥
∥δ∥
622 .

absolute Cosine similarity with eigenvectors v_max v_2 v_3 0.0 0.2 0.4 0.6 0.8 1.0 0 1 2 3 4 5 6 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 0 5 10 15 20 25 30 Den sity Den sity
(b) alignment of δ1, ∥P3 iδ T
1 vi∥/∥δ1∥
(a) cosine similarity between δ1 and vi, i ∈ {1, 2, 3}
Figure 5: A synthetic classification example. δ1 are align with top three eigen Vector of FIM sampling from 10000 gaussian noises ε 623 Figure 5 presents empirical results from this analysis. Figure 5 (a) shows histograms of the absolute 624 cosine similarities between δ1 (generated from 10,000 different ε samples) and each of the top 625 three eigenvectors v1, v2, and v3. We observe that δ1 tends to have a higher cosine similarity with 626 v1 (corresponding to the largest eigenvalue λ1) compared to v2, and v3. Furthermore, Figure 5 627 (b) displays the distribution of the squared norm of the projection of δ1 onto the top-3 eigenspace.

628 The values are predominantly close to 1, indicating that δ vectors derived from different initial 629 noise samples are largely confined to this principal subspace. These results empirically support the 630 theoretical expectation that the single-step perturbation δ1 is predominantly aligned with the principal 631 eigenspace of the FIM.

## 632 **D Iam Algorithms**

Algorithm 2 Inconsistency-Aware Minimization (SAM-like variant: IAM-S)
1: **Input:** Initial model parameters θ 0; Learning rate η; neighborhood size ρ; training set Zn; Batch size b; Number of steps K for Algorithm 1.

2: **while** not converged do 3: Sample batch {(xi, yi)}
b i=1.

4: Compute δK from Algorithm 1 using current θ, ρ, and data {xi}
b i=1.

5: Compute gradient g = ∇θL(θ)|θ+δK
6: Update parameters: θ ← θ − ηg. 7: **end while** 8: **Return** optimized parameters θ.

## 633 **E Experimental Details**

634 **Practical Considerations in estimating** Sρ(θ) 635 - **Computational Efficiency:** Calculating the FIM explicitly and performing eigenvalue decomposition is computationally expensive (O(m2 636 ) or worse, where m is the number of 637 parameters). Algorithm 1 avoids this by requiring only K gradient computations (forward 638 and backward passes) per estimation, making its computational cost approximately O(mK),
639 which is significantly more feasible for large networks. 649 **Infrastructure** Experiments are implemented in PyTorch2.7 and executed on NVIDIAA40 and L4 650 GPUs.

651 **E.1 Image classification**
652 Each reported metric is the mean± standard error computed over minimum test error from three 653 independent runs. 654 **Dataset.** We evaluate on the **CIFAR-10** (50,000 training, 10,000 test images) and **CIFAR-100**
655 (50,000 training, 10,000 test images). All images are resized to 32 × 32 and preprocessed with 656 - *RandomCrop*(32, padding= 4), 657 - *RandomHorizontalFlip*(p = 0.5), and 658 - *Normalization* using the official mean and standard deviation. 659 No additional augmentation such as Cutout or Mixup is applied. 660 **Optimization.** Models are trained for **200 epochs** with mini-batch size 128. We use SGD with momentum 0.9, weight decay 5 × 10−4 661 as an optimizer, and a multistep learning rate schedule that 662 decays the initial rate 0.1 by 0.2 at epochs 60, 120, and 160. 663 **Hyperparameters.** The inconsistency weight β and neighborhood radius ρ are selected from 664 β ∈ {0.1, 1.0, 5.0, 10.0, 20.0} and ρ ∈ {0.01, 0.05, 0.1, 0.5, 1.0} via grid search on the validation 665 split using 10% of the training dataset. The best pairs are (1.0, 0.1) for CIFAR-10 and (10.0, 0.1) for 666 CIFAR-100. For IAM-S, 0.1 and 0.5 were selected ρ value for CIFAR-10, 100 respectively. 667 **Loss function.** Cross-entropy with label smoothing (α = 0.1) is used for all methods. 668 **E.2 Semi-supervised learning** 669 In semi-supervised learning experiment, we shared most of the settings with image classification. 670 Each reported metric computed over minimum test error from three independent runs. 671 **Optimization.** Models are trained for **100 epochs** without learning rate scheduling. 672 **Hyperparameters.** We used β = 1.0 and ρ = 0.1 for both CIFAR-10 and CIFAR-100. SAM is 673 also trained with ρ = 0.1. 674 **E.3 Self-supervised learning** 675 Each reported metric is the mean **test accuracy** obtained from three independent runs. 676 **Dataset.** We use the **CIFAR-10** benchmark. All images are resized to 32×32 and augmented with 677 the SimCLR[6] pipeline: 678 - *RandomResizedCrop*(32, scale=(0.4, 1.0)), 679 - *RandomHorizontalFlip*(p = 0.5), 640 - **Number of Steps (K):** Empirical studies on neural network Hessians and FIMs suggest 641 that the eigenspectrum is often dominated by a huge largest eigenvalues. Thus, the Power 642 Iteration method can converge quickly to the dominant eigenvector. In practice, using a 643 small number of steps, often just K = 3, is found to be sufficient to get a reasonable estimate 644 of the maximizing direction. This makes the computation highly efficient.

645 - **Averaging for reduce Variance from initialization:** The estimate of Sρ(w) obtained from 646 Algorithm 1 depends on the random initialization δ0 with just K = 1. To obtain a more 647 stable estimate, we compute the metric multiple times (e.g., 10 times) with different random initializations for δ0 and report the average value: Eδ0 648 [Estimate from Alg 1]. 680 - *ColorJitter*(0.4, 0.4, 0.2, 0.1) with probability 0.8, 681 - *RandomGrayscale*(p=0.2), and 682 - *Normalization* using the official mean and standard deviation. 683 **Encoder&Projection Head.** We adopt a **ResNet-18** backbone with the first convolution modified 684 to 3×3 layer with stride = 1 and the max-pool removed. The projector is a two-layer MLP (hidden 685 size 512, output size128) with ReLU activation. 686 **Optimization.** Models are trained for **200 epochs** with mini-batch size **1024**. We use SGD
(momentum 0.9, weight decay 1×10−4 687 ) and a cosine-annealing learning-rate schedule starting at 1.0 688 after a 10-epoch warm-up. 689 **Contrastive Loss.** The NT-Xent loss is computed with temperature τ=0.5. 690 **IAM Hyperparameters.** We set the inconsistency weight β=1.0, neighborhood radius ρ=0.1, and 691 noise-scale 3.0 (Gaussian initialization). The local inconsistency is computed between projection 692 head outputs with temperature τ=0.5. 693 **Stability Heuristics.** Is identical to image classification setting. 694 **Linear Evaluation.** After every 5 epochs (and at the final epoch), a frozen encoder is evaluated via 695 a linear probe trained for 20 epochs with AdamW optimizer on the full training set (batch size 1024). 696 The reported metric is the probe's test accuracy.