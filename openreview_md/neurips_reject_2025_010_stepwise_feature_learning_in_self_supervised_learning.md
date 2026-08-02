# Stepwise Feature Learning In Self-Supervised Learning

Anonymous Author(s)

Affiliation

Address

email

## Abstract 13 **1 Introduction**

1 Recent advances in self-supervised learning (SSL) have shown remarkable progress 2 in representation learning. However, SSL models often exhibit shortcut learning 3 phenomenon, where they exploit dataset-specific biases rather than learning gen4 eralizable features, sometimes leading to severe over-optimization on particular 5 datasets. We present a theoretical framework that analyzes this shortcut learning 6 phenomenon through the lens of *extent bias* and *amplitude bias*. By investigating 7 the relations among extent bias, amplitude bias, and learning priorities in SSL, 8 we demonstrate that learning dynamics is fundamentally governed by the dimen9 sional properties and amplitude of features rather than their semantic importance.

10 Our analysis reveals how the eigenvalues of the feature cross-correlation matrix 11 influence which features are learned earlier, providing insights into why models 12 preferentially learn shortcut features over more generalizable features. 14 While deep neural networks have shown remarkable success in various learning tasks, recent studies 15 have revealed a concerning trend: models often exploit unexpected learning behavior, particularly 16 shortcut learning, which tends to take easier but potentially less reliable paths to solve general tasks 17 [13]. For example, in image classification tasks, models tend to learn earlier larger background 18 features than smaller foreground objects [17], potentially leading them to classify cows based on 19 whether they appear on grass rather than learning actual cow features, or identify camels primarily by 20 detecting desert backgrounds [5]. This phenomenon is prevalent even in SSL [11, 22, 29, 10]. 21 While previous research has shown that neural networks are vulnerable to spurious correlations in 22 data [1], several other contributing factors to shortcut learning have been identified. Hermann et al. 23 [17] find shortcuts emerging from color, size, and background. Rahaman et al. [25], Tancik et al. 24 [27] find spectral bias that low-frequency features are learned faster than high-frequency features. 25 While significant progress has been achieved, current theoretical frameworks provide insufficient 26 explanations for why models consistently induce shortcuts. 27 Recent studies have demonstrated that SSL models with small weight initialization exhibit stepwise 28 learning dynamics, where features are learned sequentially based on the corresponding eigenvalues 29 of the feature cross-correlation matrix [26]. Building on this insight, we analyze the eigenvalue 30 and eigenvector structure of the feature cross-correlation matrix. This approach provides a novel 31 theoretical framework for understanding why certain features, regardless of their semantic importance, 32 are consistently learned earlier in the training process. Our investigation focuses particularly on how 33 dimensional properties influence learning priority, potentially explaining some observed shortcut 34 learning phenomena beyond traditional spurious correlations. 35 The contributions of our work are as follows:
Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Do not distribute.

36 - We establish theoretical connections between shortcut learning phenomenon, stepwise 37 learning, and eigenvalue-eigenvector of feature cross-correlation matrix on SSL. 38 - We extend theoretical research on shortcut learning from supervised learning to SSL.

39 - We characterize *extent bias*, a tendency to prioritize features based on their dimensional 40 extent or spatial coverage rather than their semantic importance.

41 - We analyze how amplitude and frequency determine which features are learned earlier 42 in SSL, and characterize *amplitude bias*, a tendency to prioritize features based on their 43 amplitude rather than their semantic importance.

## 44 **2 Related Works**

45 **Self-supervised learning** SimCLR [7] established a foundational contrastive learning framework 46 but required large batch sizes to generate sufficient negative pairs for preventing representational 47 collapse. This limitation prompted research into non-contrastive approaches, leading to innovations 48 like SimSiam [8] and BYOL [14]. Further research introduced methods focusing on different training 49 objectives: VICReg [4] introduced variance-invariance-covariance regularization, while Barlow 50 Twins [31] employed cross-correlation matrix to prevent collapse. DINO [6] advanced the field by 51 introducing self-distillation with no labels. The success of DINO v2 [23] sparked interest in Joint 52 Embedding Predictive Architectures (JEPA) [2], with recent work by Littwin et al. [20] revealing 53 JEPA's tendency to prioritize learning "related" features over "frequently" occurring ones. 54 **Learning dynamics** Following the introduction of Neural Tangent Kernel (NTK) [18], researchers 55 have discovered important connections between eigenvalue dynamics and learning behavior, including 56 spectral bias phenomena [27, 15]. This theoretical framework has enabled deeper analysis of loss 57 function trajectories and saddle point behaviors [19, 24]. Notably, Simon et al. [26] demonstrated 58 that these saddle-to-saddle dynamics appear not only in supervised learning but also extend to SSL 59 settings.

60 **Shortcut learning** Shortcut learning was first identified in Geirhos et al. [13], describing how 61 neural networks take easier but incorrect paths to solve tasks. This phenomenon appears in various 62 ways: Geirhos et al. [12], Baker et al. [3], Hermann and Lampinen [16] showed that CNNs rely 63 on object texture rather than object shape, Wu et al. [30] demonstrated that even a single pixel can 64 mislead model's decisions, and Hermann et al. [17] revealed that CNNs preferentially learn salient 65 but potentially irrelevant features like scale and background elements. These shortcuts can arise 66 from dataset properties, particularly through spurious correlations [1] and implicit biases. Our work 67 specifically examines how dataset correlations contribute to shortcut learning.

## 68 **3 Background (Stepwise Nature Of Ssl [26])**

69 In this section, following Simon et al. [26], we analyze the stepwise learning dynamics of SSL systems 70 through the lens of toy Barlow Twins models [31]. We first introduce the loss function and gradient 71 flow dynamics, then derive the connection between cross-correlation matrix and feature learning. 72 Finally, we examine how the eigendecomposition of feature cross-correlation matrix connects to the 73 theoretical foundation for our analysis of extent bias, amplitude bias.

Given training data {x
(i) ∈ R
m 74 : i = 1, 2, · · · , n}, the training loss of toy Barlow twins is defined as L = ||C −Id||2F
, C ≡1 2n Pn i=1(W x(i))(W x′(i))
⊤ + (W x′(i))(W x(i))
⊤ 75 , where *||·||*F is Frobenius norm, W ∈ R
d×m is learnable parameters, and C ∈ R
d×d 76 is cross-correlation matrix of W x and W x′for another view x
′
77 from x. Using the feature cross-correlation matrix

$$\Gamma\equiv\frac{1}{2n}\sum_{i=1}^{n}(x^{(i)}x^{\prime(i)\top}+x^{\prime(i)}x^{(i)\top})\in\mathbb{R}^{m\times m},$$
m×m, (1)
we have L = ||WΓW⊤ − Id||2F
and C = WΓW⊤ 78 . The eigendecomposition of the feature cross-correlation matrix is Γ = VΓΛΓV
⊤ 79 Γ with ΛΓ = diag(γ1, · · · , γm) and VΓ = [v1 *· · ·* vm] ∈
R

m×m 80 ,where γ1 ≥ γ2 *≥ · · · ≥* γm are eigenvalues of Γ and vi's are the corresponding eigenvectors 81 for γi's.

$$(1)$$

82 Using (3), we can express the gradient flow as follows:

$${\frac{d W}{d t}}=-\nabla_{W}{\mathcal{L}}=-4(W\Gamma W^{\top}-I_{d})W\Gamma.$$
$$(2)$$
dt = −∇W L = −4(WΓW⊤ − Id)WΓ. (2)
83 To analyze eigenvector dynamics of weights, we assume weight initialization is aligned. 84 **Assumption 3.1** (Aligned Initialization Simon et al. [26]). At the initialization, we assume that the 85 right-singular vectors of W(0) are aligned with the top d eigenvectors of Γ, i.e., the singular value decomposition is W(0) = US0V
(≤d)⊤
Γfor a orthogonal matrix U ∈ R
d×d 86 , the top-d eigenvector matrix V
(≤d)
Γ = [v1 *· · ·* vd] ∈ R
m×d 87 , and a diagonal matrix S0 = diag(s1(0), · · · , sd(0)) with a 88 small initialization sj (0) > 0. 89 Under Assumption 3.1, the solution W(t) for the gradient flow (2) can be expressed as follows
[26, Proposition 4.1]: W(t) = US(t)V
(≤d)⊤
Γ
90 for S(t) = diag(s1(t), · · · , sd(t)), where the singular 91 values of W(t) evolve as

$$s_{j}(t)={\frac{e^{4\gamma_{j}t}}{\sqrt{{s_{j}}^{-2}(0)+(e^{8\gamma_{j}t}-1)\gamma_{j}}}}$$

which has a limit of γ
−1/2 j 92 as t → ∞ and nearly sigmoidal

$$s_{j}^{2}(t)\approx\frac{1}{\gamma_{j}+s_{j}^{-2}(0)e^{-8\gamma_{j}t}}=:\vec{s}_{j}^{2}(t).$$
(t). (3)
Solving s˜
2 j
$\langle t\rangle=\frac{1}{2}s_j^2(\infty)$ at its end. 
$\mathfrak{so}$. 

93 (∞) at its critical time t = τj , we have

$$(3)$$
ne $\tau=\tau_j$, we have 
$$\tau_{j}=-\frac{\log\left(s_{j}^{2}(0)\gamma_{j}\right)}{8\gamma_{j}}$$

$$(4)$$
8γj(4)
around which sj (t) (or s˜j (t)) passes 12 γ
−1/2 j 94 and rapidly increases from near zero to near the saturation γ
−1/2 j 95 .

96 In this paper, we focus on the property that the eigenvector feature vj corresponding to a larger γj 97 leads to an earlier critical point τj from (4).

## 98 **4 Extent Bias**

99 In computer vision tasks, backgrounds typically span larger regions while foreground objects occupy 100 more concentrated areas. Recent work by Hermann et al. [17] reveals that CNNs preferentially 101 learn these background features over object-specific details, creating a specific form of spurious 102 correlation between backgrounds and class labels. For example, cows are often classified based on 103 grass backgrounds rather than their distinctive features, and camels are identified through desert scenes 104 [5]. This phenomenon points to a underlying learning mechanism we term *extent bias*, a fundamental 105 tendency of neural networks to prioritize features based on their dimensional extent or spatial coverage 106 rather than their semantic importance. The connection between extent bias and learning dynamics 107 implies the need for understanding a more fundamental mechanism beyond traditional spurious 108 correlations. While spurious correlations emerge from dataset-specific relationships, the bias toward 109 learning background features is inherent in the learning dynamics of neural networks themselves. 110 Through our analysis of SSL systems, we demonstrate that this bias for background features emerges 111 naturally from how models learn earlier features with higher extent bias, independent of their semantic 112 relevance or predictive power.

113 In this section, we investigate how different feature properties influence learning priorities in SSL.

114 Through extent bias analysis, we demonstrate how features with larger dimensional coverage are 115 learned before those with smaller coverage, regardless of their semantic importance. 116 We construct a theoretical framework that identifies dimensional effects in feature learning. By 117 analyzing how SSL models process features of varying extent bias, we can directly observe how 118 extent bias influences learning priority and connects to the background-foreground learning dynamics 119 observed in practice.

0 1000 2000 3000 4000 Step 0.0 0.5 1.0 0 1000 2000 3000 4000 Step 0.0 0.5 1.0 0 1000 2000 3000 4000 Step 0 1 2 Eig env alue W

ei 2 Loss FA(el)
FA(es)
l s
Figure 1: **Effects of extent bias on learning dynamics in SSL.** (Left) Stepwise learning curves of Barlow Twins. There are two (d = 2) learning steps shown with two black dashed vertical lines (also shown in the other two panels) which indicate the time steps t1 and t2 with t1 : t2 ≈
1 γl
:
1 γs
=
1 ml
:1 ms
. The predicted loss (dashed green) of L =Pd j=1(λ˜j (t) − 1)2 =Pd j=1(˜s 2 j(t)γj − 1)2 using (3) match the empirical result (solid green). (Center) Evolution of eigenvalues λj 's of C during training. At the beginning, the first eigenvalue λ1 (blue) increases to 1 and then later the second λ2 (red) follows. We also compare them with the predicted evolution λ˜j (t) (dashed lines). (Right)
Evolution of the feature alignment ||W e||2 for e = el (blue) and e = es (red). It shows very similar behaviors with the eigenvalues λ˜
1/2 j(dashed lines). See Theorem 4.5. We use ml = 9, ms = 1. See Appendix A.1 for more detailed settings.

## 120 **4.1 Settings**

We first consider the following base input xbase = [bl1
⊤ml
, bs1
⊤ms
]
⊤ ∈ R
m, where bl, bs i.i.d.

121 ∼ B(p =
122 0.5) follow the Bernoulli distribution and take the value ±1 with the equal probability, ml and ms 123 indicate the size of larger part and smaller part, respectively, i.e., ml > ms and ml + ms = m, and 124 1k is the k-dimensional all-one vector. From now on, we will use the subscript l and s for the indices 125 with respect to the *larger*-part and *smaller*-part features, respectively.

Then, to obtain the positive pair (*x, x*′
126 ), we introduce the following data augmentation x = xbase +
ε and x
′ = xbase + ε
′, with the noise *ε, ε*′i.i.d. ∼ N (0m, a2 127 Im) for some a > 0.

## 128 **4.2 Learning Dynamics On Extent Bias**

129 In this subsection, we discuss the relationship between γj and L, focusing on which features are 130 learned earlier. From Section 4.1, we can simplify the feature cross-correlation matrix Γ by analyzing 131 the expected values of the augmented features. Based on the definition in (1), we have:

$$\Gamma={\frac{1}{2n}}\sum_{i=1}^{n}(x^{(i)}x^{\prime(i)\top}+x^{\prime(i)}x^{(i)\top})=\mathbb{E}[x_{\mathrm{base}}x_{\mathrm{base}}^{\top}].$$

$$({\boldsymbol{5}})$$
base]. (5)
141 We defer the proof to Appendix B.1.

138 The eigendecomposition of Γ is given by the following proposition:

139 **Theorem 4.1.** For the correlation matrix in (5), we have the eigenvalue matrix ΛΓ *and eigenvector* 140 *matrix* VΓ:
ΛΓ = diag ([ml, ms, 0m−2]), V (≤2)
$$\tau_{\Gamma}^{(\leq2)}=[e l/$$
√ml es/
√ms] .

142 We hypothesize that features with larger dimensions are learned faster, regardless of their predictive 143 power or potential to cause shortcuts. This is particularly relevant in vision tasks where such features

132 To identify which features drive the loss as stepwise phenomena, we consider basis vectors that
133 disentangle individual features. Specifically, we define basis vectors el and es where each vector has
134 ones only in the dimensions corresponding to its respective feature:
$e_{l}=[\mathbf{1}_{m_{l}}^{\top},\mathbf{0}_{m_{s}}^{\top}]^{\top},e_{s}=[\mathbf{0}_{m_{l}}^{\top},\mathbf{1}_{m_{s}}^{\top}]^{\top}\in\mathbb{R}^{m}$.  FA($e$) = $\|We\|_{2}$ for $e=e_{l},e_{s}$.  
FA(e) = ∥W e∥2 for e = el, es. (6)
135 By measuring the feature alignment between these basis vectors and the weight matrix through
136 FA(e) = ∥W e∥2, we can identify which features are being learned at each stage of the training
137 process.
$\left(6\right)$. 
$\mathbf{v}$

## 163 4.3 Cross-Correlation Eigenvalue Λ **And Loss Relationship**

164 In this subsection, we analyze the relationship between the eigenvalues λj of cross-correlation matrix 165 C.

166 **Theorem 4.2.** Under Assumption 3.1, the eigenvalues λj *of feature cross-correlation matrix* C =
WΓW⊤, using the approximation sj ≈ s˜j *in (3), are approximated as* λj = s 2 jγj ≈ s˜
2 jγj =: λ˜ 167 j 168 *which have*

λ˜j (τj ) = 12 and λ˜′i(τj ) = 2γj if i = j,
≈ 0 if i ̸= j(7)
at τj = −log(s 2 j
(0)γj )/8γj *in (4). For the Barlow Twins loss* L = ∥C − Id∥
2F
169 *, we have* L = 
Pd j=1(λj − 1)2 and −
dL
dt (τj ) ≈ λ˜′j 170 (τj ) = 2γj .

171 We defer the proof to Appendix B.3. 172 Figure 6 in Appendix C shows the relationship between cross-correlation eigenvalue λ differentiated with respect to t and loss derivatives dL
dt 173 . The close alignment between the loss derivative and λ 174 derivative curves demonstrates that the decrease in loss is directly driven by λ, with larger ml features 175 learned, and smaller ms features learned later. The curves' relative magnitudes show an approximate ml 176 : ms ratio, which matches our theoretical predictions.

## 177 **4.4 Weight Singular Value Evolution**

178 To verify the dynamics of weight singular values sj , we propose the following theorem:
179 **Theorem 4.3.** Using the approximation (3), the singular values of the weight matrix W *satisfy* s˜j (τj ) = 1/p2γj and s˜
′ j
(τj ) = p2γj 180 *at the critical point* t = τj .

181 We defer the proof to Appendix B.4. 182 Figure 7 in Appendix C shows two key aspects of singular value dynamics during training. First, the singular values sj evolve to their theoretical limits 1/
√γj and 1/
√ 
183 γs, as predicted by our 184 analysis. Second, the derivatives of these singular values exhibit peaks at their respective critical points, with magnitudes that follow the predicted 
√2γl:
√
185 2γs ratio. These results provide strong 186 empirical validation of our theoretical framework, demonstrating that both the convergence values 187 and learning priority on different features are governed by their corresponding eigenvalues in the 188 feature cross-correlation matrix Γ. 144 might correspond to larger pixel regions. We experiment using a simple toy model to validate our 145 theoretical analysis of dimensional influence on feature learning. In our experimental setup, we used 146 two distinct features with different dimensional coverage (ml = 9 and ms = 1), allowing us to 147 clearly observe the learning dynamics.

148 As shown in Figure 1, the results demonstrate three key phenomena:
149 Figure 1 (Left) shows loss trajectory (green line) exhibits two distinct stepwise phenomena, marked 150 by black vertical lines. These stepwise decreases precisely align with the abrupt increase in the 151 eigenvalue observed in Figure 1 (Center), confirming our theoretical prediction that eigenvalue 152 dynamics drives the learning process. 153 Figure 1 (Center) shows a clear stepwise pattern in which two distinct eigenvalues of Γ increase 154 sequentially. This sequential increase directly corresponds to the learning priority of feature, with the 155 higher-dimensional feature (ml = 9) being learned first.

156 Figure 1 (Right) shows that, feature alignment measurements ||W e||2 from (6) provide direct evidence 157 of the learning order: the alignment with e1 (blue line, corresponding to the larger feature dimension) 158 increases during the first loss decrease, while e2 alignment (red line) follows during the second phase.

159 This learning pattern strongly supports our hypothesis that dimensional coverage determines how 160 early the features learned. 161 This result suggests that the spatial extent of features, rather than their semantic content, plays a 162 crucial role in determining learning priority.

$$(7)$$

## 189 **4.5 Aligned Initialization And Subspace Alignment**

190 To justify our alignment initialization assumption in Assumption 3.1, we first define the following 191 subspace alignment metric: 192 **Definition 4.4** (Subspace Alignment). We define subspace alignment of two subspaces Im(A) and 193 Im(B):
SA(*A, B*) = ||A
⊤B||2F /d, where Im(A) = {Av ∈ R
m : v ∈ R
d}, A = [a1 · · · ad], B = [b1 *· · ·* bd] ∈ R
m×dand ai, bi ∈ R
m 194 195 are unit vectors.

$\mathbf{\hat{}}$ ||$\mathbf{A}$ | $\mathbf{B}$
196 Note that 0 ≤ SA(*A, B*) ≤ 1 and it attains SA(*A, B*) = 0 when Im(A) ⊥ Im(B), and SA(*A, B*) = 1 197 when Im(A) = Im(B). Figure 10 (Top) in Appendix D empirically validates Assumption 3.1 using 198 the subspace alignment metric. The model becomes aligned rapidly in the early stages of training, 199 satisfying the assumption.

## 200 **4.6 Orthogonal Feature Learning**

201 Our analysis shows that features are learned as orthogonal to each other, where each feature is acquired 202 independently without interference from others. This orthogonal learning pattern is particularly 203 evident in the evolution of the model's weight matrix singular vectors. To formalize this observation, 204 we analyze how the left singular vectors of the weight matrix align with the feature vectors during 205 training. 206 **Theorem 4.5.** Under Assumption 3.1, the left singular vectors u of W(t) learn features orthogonally:

$|F/\theta|$
ProjU(≤2) (W el) := (u
⊤
l W el, u⊤
s W el) = (pλl, 0),
ProjU(≤2) (W es) := (u
⊤
l W es, u⊤
s W es) = (0,pλs),
where ul, us *are the corresponding left singular vectors for the singular values* sl 207 , ss.

208 Figure 11 shows orthogonal learning pattern that features are learned independently and sequentially, 209 supporting our theoretical analysis of stepwise learning dynamics.

## 210 **4.7 Non-Linear Multi Layer Network**

211 Nonlinearity exhibits distinct learning dynamics compared to linearity. Therefore, we aim to investi212 gate whether extent biass also exists in multilayer perceptrons (MLPs). We experiment with a 3-layer 213 network, using leakyReLU as the activation function, for understanding non-linear feature learning 214 dynamics. Our non-linear network experiments demonstrate that extent bias persists beyond linear 215 models. As shown in Figure 14 in Appendix G, the non-linear network exhibits remarkably similar 216 stepwise learning patterns to those observed in linear models Figure 1. Key similarities include: simi217 lar eigenvalue evolution patterns, consistent stepwise loss reduction phases. These results suggest that 218 extent bias is a fundamental learning phenomenon that transcends network architecture complexity, 219 rather than being merely an artifact of linear models.

## 220 **4.8 Practical Study On Colored-Mnist Dataset**

221 We conducted experiments using a Colored-MNIST dataset, where we adjusted the ratio of digits 222 pixels relative to the total image pixels. We tested three different ratios: 0.05, 0.10, and 0.15. In this 223 dataset, we set the correlation between background and label to 70% for both training and test sets, 224 making it difficult for a model that predicts solely based on background to achieve accuracy higher 225 than 70%. According to our hypothesis, since backgrounds have larger extent bias than objects, the 226 test set accuracy would rapidly increase from an initial 10% (random choosing) to 70% (as the model 227 learns background features), then plateau for a period, before slowly rising to 100% (as it learns 228 object features). We also hypothesized that this plateau period would decrease as the ratio of label 229 pixels increases in the images, with shorter plateaus observed in the 0.15 ratio condition compared to 230 0.05.

231 Figure 2 supports our hypothesis. Across all pixel ratio conditions (0.05, 0.10, 0.15), test accuracy 232 exhibited a consistent pattern: a rapid increase from initial 10% to 70%, followed by a plateau period,

100 Test A
cc urac y bg correlation Object ratio : 0.05 Object ratio : 0.10 Object ratio : 0.15 50 0 10 20 30 40 50 60 Epoch

## 240 **5 Amplitude Bias** 248 **5.1 Settings**

249 To analyze how frequency and amplitude bias affect learning dynamics, we consider input data xbase ∈ R
m 250 composed of two sinusoidal components with different frequencies:
xbase[t] = chbh sin(fht) + clbl sin(flt), (8)

## 258 **5.2 Learning Dynamics On Amplitude Bias**

Similar to Section 4.2, we consider basis vectors eh and el 259 that isolate individual features: eh =
260 ch sin(fht) and el = cl sin(flt), where 0 ≤ t ≤ m. Note that these two are orthogonal since fh =
2π m k and fl =
2π m k
′ with k ̸= k
′
261 . Similar to Theorem 4.1, the cross-correlation matrix Γ for the 262 data generated from (8) can be expressed as follows: 263 **Theorem 5.1.** Under (8), the correlation matrix Γ has 241 In regression tasks, the phenomenon of spectral bias has been observed, wherein low-frequency 242 components are learned more rapidly than high-frequency components during the training process. 243 Conversely, in classification tasks, a phenomenon known as frequency shortcut [28] has been observed, 244 wherein the model preferentially learns the distinctive Fourier components of the input during the 245 training process. While these studies have primarily focused on supervised learning, we extend this 246 investigation to the SSL, seeking to understand whether similar learning dynamics persist within SSL 247 frameworks.

where fh =
2π m k and fl =
2π m k
′represent different frequencies for some integers k and k
′, bh, bl i.i.d.

251 ∼
B(p = 0.5) follow the Bernoulli distribution and take the value ±1. Suppose fh < fl 252 to examine 253 the learning dynamics between low and high frequency components. The coefficients ch and cl 254 control the amplitude of each sinusoidal component, allowing us to investigate how magnitudes affect learning earlier. The Bernoulli variables bh and bl 255 introduce phase reversal in the signal. The time 256 vector t spans the input dimension m. We use the same augmentation with (4.1) to generate positive pairs (*x, x*′
257 ) by adding Gaussian noise.

ΛΓ = *diag* -c 2 hm/2, c2 l m/2, 0m−2 , V (≤2)

$$\left]\right),V_{\Gamma}^{(\leq2)}=\left[e_{h}\ e_{l}\right].$$

233 and then a gradual ascent to 100%. Notably, as the object pixel ratio increased, the duration of the 234 plateau phase decreased. The loss function continued to decrease even when accuracy remained 235 stagnant at 70%. This suggests a extent bias where larger objects are prioritized during the learning 236 process. The pattern reflects how the model initially achieves 70% accuracy by relying on background 237 features, which statistically occupy larger regions, before progressively learning object features. 238 Furthermore, this indicates that larger extents occupy greater eigenvalues, implying a reduction in the 239 critical point τj .

0 100 200 300 400 500 Step 0.0 0.5 1.0 0 100 200 300 400 500 Step 0.0 0.5 1.0 0 100 200 300 400 500 Step 0 1 2 Eigen valu e W

ei 2 FA(el) FA(eh)
Loss l h
264 We defer the proof to Appendix B.2.

From (9), we observe that eigenvalues are proportional to the squares of the coefficients c 2hand c 2 l 265 . 266 This implies that the learning dynamics are more strongly influenced by the amplitude rather than the 267 underlying frequency. 268 To validate our theoretical analysis of amplitude bias effect on learning dynamics, we conduct experiments using input data defined in (8). Especially, we set ch > cl 269 . This configuration shown in 270 Figure 4 in Appendix A, allows us to examine how high-amplitude ch sin(fht) and low-amplitude 271 cl sin(flt) affects feature amplitude bias. More details about the experiment are in Appendix A.3.

272 Our analysis reveals two dominant eigenvalues. The large eigenvalue corresponds to the high273 amplitude feature, and small eigenvalue corresponds to the low-amplitude component. The eigen274 vectors of Γ are shown in Figure 5 , Appendix A. The first eigenvector, which corresponds to the 275 largest eigenvalue, captures the dominant high-amplitude oscillation. The second eigenvector, which 276 matches next-largest eigenvalue, captures the low-amplitude oscillation. Other eigenvectors are noise, 277 corresponding to eigenvalues that are almost 0.

## 278 5.3 Cross-Correlation Eigenvalue Λ **And Loss Relationship**

279 We analyze how the eigenvalues λ relate to the loss dynamics. The relationship follows similar 280 patterns to those observed in Section 4.3, but with coefficients ch and cl rather than ml and ms.

281 Figure 8 in Appendix C shows the close relationship between the derivatives of cross-correlation eigenvalues dλh dt ,
dλl dt and dL
dt 282 . The peaks in these derivatives occur at the critical points with magnitudes proportional to the corresponding coefficients γh : γl = c 2h
: c 2 l 283 (see (9)). This shows our 284 theoretical predictions Theorem 4.2 matches empirical result.

## 285 **5.4 Weight Singular Value Evolution** 295 **5.5 Aligned Initialization And Subspace Alignment**

286 We now analyze how the singular values of the weight matrix evolve during training. Similarly to the 287 extent bias case, we expect the singular values sj to converge to theoretical limits determined by the 288 feature coefficients.

289 Figure 9 in Appendix C shows the evolution of singular values sh and sl of weight matrix W (Left)
and their derivatives (Right). The singular values converge to their theoretical limits 1/
√ 
290 γj predicted by Theorem 4.3, where γj = c 2 j m 2 291 . At the critical points τj , the derivatives achieve their maximum values of p 292 2γj , showing that rates of feature learning are proportional to the coefficients. These 293 results confirm that the feature coefficients, rather than their frequencies, govern both the convergence 294 values and rates of feature learning. 296 To validate Assumption 3.1 about alignment between the weight matrix singular vectors and eigen297 vectors of Γ, we measure the subspace alignment metric as defined in the extent case Definition 4.4.

298 Figure 10 (Bottom) in Appendix D empirically validates our assumption through subspace alignment 299 measurements. As discussed in Section 4.5, the model achieves alignment rapidly in the early stages 300 of training, even with small random initializations.

## 301 **5.6 Orthogonal Feature Learning**

302 Similar to the extent case, we investigate how the weight matrix learns different frequency components 303 orthogonally as shown in Theorem 4.5. The orthogonal learning pattern reveals how frequency features 304 are acquired independently despite their different spectral characteristics. 305 Figure 12 in Appendix E shows the trajectories of weight matrix in terms of their alignments with frequency components eh and el 306 . The blue trajectory shows the first learning phase where 307 u1 aligns with the high-amplitude feature (ch sin(fht)), followed by the red trajectory showing u2 308 aligning with the low-amplitude feature (cl sin(flt)). This sequential, orthogonal learning pattern 309 demonstrates that feature learning is primarily determined by coefficient magnitudes rather than 310 frequency characteristics, supporting our analysis in Theorem 4.5.

## 311 **5.7 Non-Linear Multi Layer Network**

312 Same as Section 4.7 in Appendix G, we conduct experiments with a 3-layer network using leakyReLU 313 activations to analyze how amplitude coefficients affect learning dynamics in non-linear settings. 314 Figure 15 in Appendix G demonstrates amplitude bias effects in non-linear networks is similar 315 to linear networks on Figure 3. These results confirm that amplitude bias persists in non-linear 316 architectures, suggesting amplitude magnitude remains a primary determinant of feature learning 317 priority regardless of network complexity.

## 318 **5.8 Discussion**

319 Figure 13 in Appendix F shows that a learning process is driven primarily by feature coefficient 320 magnitude rather than frequency characteristics. The key observation is that the first learned features 321 are those with large coefficients, independent of their spectral properties. This finding parallels 322 frequency shortcut [28] in classification tasks, but reveals a different underlying mechanism. While 323 frequency shortcut suggests models preferentially learn distinctive Fourier components, our results 324 demonstrate that amplitude magnitude—not frequency characteristics—primarily determines feature 325 learning priority.

## 326 **6 Conclusion**

327 In this work, we establish a theoretical connection between eigendecomposition of the feature cross328 correlation matrix, shortcut learning, and stepwise learning behavior in SSL. We provide insights 329 into how dimensional feature properties influence the learning process in SSL frameworks. This 330 work not only explains observed shortcut learning phenomena but also offers a theoretical lens for 331 understanding and potentially mitigating such learning biases. This theoretical framework lays the 332 groundwork for developing more robust SSL algorithms. Future work should focus on leveraging 333 these insights to design mechanisms that encourage learning of generalizable features despite their 334 potentially lower extent bias or amplitude bias.

## 335 **References**

336 [1] M. Arjovsky, L. Bottou, I. Gulrajani, and D. Lopez-Paz. Invariant risk minimization. *arXiv* 337 *preprint arXiv:1907.02893*, 2019.

338 [2] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Bal339 las. Self-supervised learning from images with a joint-embedding predictive architecture. In 340 *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 341 15619–15629, 2023. 342 [3] N. Baker, H. Lu, G. Erlikhman, and P. J. Kellman. Deep convolutional networks do not classify 343 based on global object shape. *PLoS computational biology*, 14(12):e1006613, 2018.

344 [4] A. Bardes, J. Ponce, and Y. LeCun. Vicreg: Variance-invariance-covariance regularization for 345 self-supervised learning. *arXiv preprint arXiv:2105.04906*, 2021. 346 [5] S. Beery, G. Van Horn, and P. Perona. Recognition in terra incognita. In *Proceedings of the* 347 *European Conference on Computer Vision (ECCV)*, September 2018. 348 [6] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin. Emerging 349 properties in self-supervised vision transformers. In *Proceedings of the IEEE/CVF international* 350 *conference on computer vision*, pages 9650–9660, 2021. 351 [7] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton. A simple framework for contrastive learning 352 of visual representations. In *International conference on machine learning*, pages 1597–1607. 353 PMLR, 2020.

354 [8] X. Chen and K. He. Exploring simple siamese representation learning. In *Proceedings of the* 355 *IEEE/CVF conference on computer vision and pattern recognition*, pages 15750–15758, 2021.

356 [9] L. Deng. The mnist database of handwritten digit images for machine learning research [best of 357 the web]. *IEEE signal processing magazine*, 29(6):141–142, 2012. 358 [10] C. Doersch, A. Gupta, and A. A. Efros. Unsupervised visual representation learning by context 359 prediction. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 360 December 2015. 361 [11] C. Doersch, A. Gupta, and A. A. Efros. Unsupervised visual representation learning by context 362 prediction. In *Proceedings of the IEEE international conference on computer vision*, pages 363 1422–1430, 2015. 364 [12] R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann, and W. Brendel. Imagenet365 trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. 366 *arXiv preprint arXiv:1811.12231*, 2018. 367 [13] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann.

368 Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11):665–673, 2020.

369 [14] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch, 370 B. Avila Pires, Z. Guo, M. Gheshlaghi Azar, et al. Bootstrap your own latent-a new ap371 proach to self-supervised learning. *Advances in neural information processing systems*, 33: 372 21271–21284, 2020. 373 [15] M. S. Halvagal, A. Laborieux, and F. Zenke. Implicit variance regularization in non-contrastive 374 ssl. *arXiv preprint arXiv:2212.04858*, 2022. 375 [16] K. Hermann and A. Lampinen. What shapes feature representations? exploring datasets, 376 architectures, and training. *Advances in Neural Information Processing Systems*, 33:9995– 377 10006, 2020. 378 [17] K. L. Hermann, H. Mobahi, T. Fel, and M. C. Mozer. On the foundations of shortcut learning. 379 *arXiv preprint arXiv:2310.16228*, 2023. 380 [18] A. Jacot, F. Gabriel, and C. Hongler. Neural tangent kernel: Convergence and generalization in 381 neural networks. *Advances in neural information processing systems*, 31, 2018. 382 [19] A. Jacot, F. Ged, B. ¸Sim¸sek, C. Hongler, and F. Gabriel. Saddle-to-saddle dynamics in deep linear 383 networks: Small initialization training, symmetry, and sparsity. *arXiv preprint arXiv:2106.15933*, 384 2021. 385 [20] E. Littwin, O. Saremi, M. Advani, V. Thilak, P. Nakkiran, C. Huang, and J. Susskind. How jepa 386 avoids noisy features: The implicit bias of deep linear self distillation networks. *arXiv preprint* 387 *arXiv:2407.03475*, 2024. 388 [21] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. *arXiv preprint* 389 *arXiv:1711.05101*, 2017.

390 [22] M. Noroozi, H. Pirsiavash, and P. Favaro. Representation learning by learning to count. In 391 *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, Oct 2017. 392 [23] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, 393 F. Massa, A. El-Nouby, et al. Dinov2: Learning robust visual features without supervision. 394 *arXiv preprint arXiv:2304.07193*, 2023. 395 [24] S. Pesme and N. Flammarion. Saddle-to-saddle dynamics in diagonal linear networks. *Advances* 396 *in Neural Information Processing Systems*, 36:7475–7505, 2023. 397 [25] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, Y. Bengio, and A. Courville. 398 On the spectral bias of neural networks. In *International conference on machine learning*, pages 399 5301–5310. PMLR, 2019. 400 [26] J. B. Simon, M. Knutins, L. Ziyin, D. Geisz, A. J. Fetterman, and J. Albrecht. On the stepwise 401 nature of self-supervised learning. In *International Conference on Machine Learning*, pages 402 31852–31876. PMLR, 2023. 403 [27] M. Tancik, P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ra404 mamoorthi, J. Barron, and R. Ng. Fourier features let networks learn high frequency functions in 405 low dimensional domains. *Advances in neural information processing systems*, 33:7537–7547, 406 2020. 407 [28] S. Wang, R. Veldhuis, C. Brune, and N. Strisciuglio. What do neural networks learn in image 408 classification? a frequency shortcut perspective. In *Proceedings of the IEEE/CVF International* 409 *Conference on Computer Vision*, pages 1433–1442, 2023.

410 [29] D. Wei, J. J. Lim, A. Zisserman, and W. T. Freeman. Learning and using the arrow of time.

411 In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 412 June 2018. 413 [30] S. Wu, S. Chen, C. Xie, and X. Huang. One-pixel shortcut: on the learning preference of deep 414 neural networks. *arXiv preprint arXiv:2205.12141*, 2022. 415 [31] J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Deny. Barlow twins: Self-supervised learning via 416 redundancy reduction. In *International conference on machine learning*, pages 12310–12320. 417 PMLR, 2021.

## 418 **A Experimental Details** 419 **A.1 Extent Bias Experiment**

420 For the extent bias experiment shown in Section 4.1, we train the model using 400 epochs. The 421 augmentation noise parameter a was set to 0.01. We use a dataset size of n = 1000 samples with feature dimension m = 10. We also use learning rate η = 6 · 10−4and scaling factor 5 · 10−1 422 .

## 423 **A.2 Colored Mnist Experiment**

424 For the Colored MNIST shown in Section 4.8, we train the model using default augmentation 425 (RandomResizedCrop, RandomHorizontalFlip, RandomColorJitter, RandomGrayscale, Random426 GaussianBlur, RandomSolarization) with augmentated image size 42×42. We use background colors 427 as [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 123, 123], [123, 428 0, 123], [123, 123, 0], [123, 0, 0]][digit]. We trained ResNet18 with 60 epochs, AdamW [21] with learning rate η = 4 × 10−6 429 .

## 430 **A.3 Amplitude Experiment**

431 For the amplitude experiment shown in Section 5.1, we train the model using 500 epochs. The 432 augmentation noise parameter a is set to 0.1. We use a dataset size of n = 1000 samples with feature frequency fh = 2 2π 24 , fl = 32 2π 24 . We also use learning rate η = 5 · 10−5, scaling factor 3 · 10−3 433 and m = 96.

0 20 40 60 80 t
¡1 0 1 x

Figure 4: **Input data** x = x*base* + ϵ. xbase[t] = bhch sin(fht) + blcl sin(flt), where ch = 1, cl = 0.5, fh =
2π m 32, fl =
2π m 8, m = 96.

0 20 40 60 80 v1 index
¡0.1 0.0 0.1 0 20 40 60 80 v3 index 0.0 0.2 0 20 40 60 80 v2 index
¡0.1 0.0 0.1
Figure 5: The eigenvectors vi's of Γ for i = 1, 2, 3 **(from Left to Right).** (Left) The first eigenvector that correspondent to the largest eigenvalue indicates the (high frequency) feature with a high amplitude ch sin (fht), (Center) the second the (low frequency) feature with a low amplitude feature cl sin (flt), (Right) the third (and beyond) noise, where cl < ch.

434

## 435 **B Proofs** 436 **B.1 Proof Of Theorem 4.1**

437 Through matrix analysis, we can express:

$$\Gamma=\mathbb{E}[x_{\mathrm{base}}x_{\mathrm{base}}^{\top}]=\begin{bmatrix}\mathbf{1}_{m_{l}\times m_{l}}&\mathbf{0}_{m_{s}\times m_{l}}\\ \mathbf{0}_{m_{l}\times m_{s}}&\mathbf{1}_{m_{s}\times m_{s}}\end{bmatrix},$$

438 which has two eigenvectors el/∥el∥ and es/∥es∥ correspond to nonzero eigenvalues. We get the 439 eigenvalues ml and ms from the following equation:
det(Γ − λI) = det(1ml×ml − λIml×ml) det(1ms×ms − λIms×ms) = 0.

440 Finally, we can get the eigendecomposition Γ = VΓΛΓVΓ where

$$\begin{array}{c}{{\Lambda_{\Gamma}=\mathrm{diag}\left([m_{l},m_{s},\ {\bf0}_{m-2}]\right),}}\\ {{V_{\Gamma}^{(\leq d)}=\left[\frac{1}{\sqrt{m_{l}}}e_{l}\ \frac{1}{\sqrt{m_{s}}}e_{s}\ \right].}}\end{array}$$

## 441 **B.2 Proof Of Theorem 5.1**

442 The cross-correlation matrix Γ for this input can be expressed using (5):
Γ = E[x*base*x
⊤
base]
= E[c 2hb 2hsin(fht) sin(fht)
⊤ + c 2 lb 2hsin(flt) sin(flt)
⊤ + chclbhbl sin(fht) sin(flt)
⊤ + chclbhbl sin(flt) sin(fht)
⊤]
= c 2 hsin(fht) sin(fht)
⊤ + c 2 lsin(flt) sin(flt)
⊤.

443 Using the orthogonality between sin(fht) and sin(flt) (fh ̸= fl), where t ∈ N,

$\mathrm{erf}$ 4. 
Γ = c 2hsin(fht) sin(fht)
⊤ + c 2 lsin(flt) sin(flt)
⊤,
Γ sin(fht) = c 2h ||sin(fht)||2sin(fht),
Γ sin(flt) = c 2 l ||sin(flt)||2sin(flt).

444 We find eigenvector and eigenvalue as:

and eigenvalue as:  $$\Lambda_{\Gamma}=\mbox{diag}\left(\left[c_{h}^{2}||\sin(f_{h}t)||^{2},c_{l}^{2}||\sin(f_{l}t)||^{2},{\bf0}_{m-2}\right]\right),$$ $$V_{\Gamma}^{(\leq2)}=\left[e_{h}\ e_{l}\right]^{\top}.$$
With f =
2π 445 m k for some integer k, we have

$$||\sin(fx)||^{2}=\int_{0}^{m}\sin^{2}(fx)dx=\int_{0}^{m}\frac{1-\cos(2fx)}{2}dx$$ $$=\frac{1}{2}\left[x-\frac{\sin(2fx)}{2}\right]_{0}^{m}=\frac{m}{2}-\frac{\sin(2fm)}{4}=\frac{m}{2}.$$

446 Finally, we have

$$\begin{array}{c}{{\Lambda_{\Gamma}=\mathrm{diag}\left(\left[c_{h}^{2}\frac{m}{2},c_{l}^{2}\frac{m}{2},{\bf0}_{m-2}\right]\right),}}\\ {{V_{\Gamma}^{(\leq2)}=\left[e_{h}\;e_{l}\right].}}\end{array}$$

447 **B.3 Proof of Theorem 4.2** 448 We have

$$\bar{\lambda}_{j}(t)=\tilde{s}_{j}^{2}(t)\gamma_{j}=(1+\lambda_{j}(0)^{-1}e^{-8\gamma_{j}t})^{-1},$$

and thus if we plug in τj = − log(λj (0))/8γj , i.e., exp(−8γj τj ) = λj (0), then we have λ˜ 449 j (τj ) =
(1 + 1)−1 =
1 2
. The derivative λ˜′j 450 (t) at t = τj is given as follows:

$$\tilde{\lambda}_{j}^{\prime}(t)=-(1+\lambda_{j}(0)^{-1}e^{-8\gamma_{j}t})^{-2}(-8\gamma_{j}\lambda_{j}(0)^{-1}e^{-8\gamma_{j}t})$$ $$=-\tilde{\lambda}_{j}^{2}(t)(-8\gamma_{j}\lambda_{j}(0)^{-1}e^{-8\gamma_{j}t})$$ $$\tilde{\lambda}_{j}^{\prime}(\tau_{j})=-\tilde{\lambda}_{j}^{2}(\tau_{j})(-8\gamma_{j}\lambda_{j}^{-1}(0)\lambda_{j}(0))$$ $$=2\gamma_{j}.$$

451 Using the equations

$$C=\sum_{j=1}^{d}\lambda_{j}u_{j}u_{j}^{\top}\mathrm{~and~}C^{2}=\sum_{j=1}^{d}\lambda_{j}^{2}u_{j}u_{j}^{\top},$$

452 we get the loss

$$\mathcal{L}=||C-I||_{F}^{2}=\operatorname{Tr}((C-I)(C-I))=\operatorname{Tr}(C^{2})-2\operatorname{Tr}(C)+d$$ $$=\sum_{j=1}^{d}\lambda_{j}^{2}-2\sum_{j=1}^{d}\lambda_{j}+d=\sum_{j=1}^{d}(\lambda_{j}-1)^{2}.$$  The following equation:
453 Thus, we get the following equation:

$$\begin{array}{l}{{\frac{d\mathcal{L}}{d t}(\tau_{j})=\sum_{i=1}^{d}2(\lambda_{i}(\tau_{j})-1)\lambda_{i}^{\prime}(\tau_{j})}}\\ {{\approx\sum_{i=1}^{d}2(\bar{\lambda}_{i}(\tau_{j})-1)\bar{\lambda}_{i}^{\prime}(\tau_{j})}}\\ {{\approx2(\bar{\lambda}_{j}(\tau_{j})-1)\bar{\lambda}_{j}^{\prime}(\tau_{j})}}\\ {{=-\bar{\lambda}_{j}^{\prime}(\tau_{j})=-2\gamma_{j}.}}\end{array}$$

## 454 **B.4 Proof Of Theorem 4.3**

455 First, we have

$$\begin{array}{c}{{\tilde{s}_{j}(t)=(\gamma_{j}+s_{j}^{-2}(0)\exp(-8\gamma_{j}t))^{-1/2},}}\\ {{\tilde{s}_{j}(\tau_{j})=(\gamma_{j}+s_{j}^{-2}(0)\lambda_{j}(0))^{-1/2}}}\\ {{=(2\gamma_{j})^{-1/2}.}}\end{array}$$

456 and its derivative is given as follows:

$$\tilde{s}_{j}^{\prime}(t)=-\frac{1}{2}(\gamma_{j}+s_{j}^{-2}(0)\exp(-8\gamma_{j}t))^{-3/2}(-8\gamma_{j}s_{j}^{-2}(0)\exp(-8\gamma_{j}t)),$$ $$\tilde{s}_{j}^{\prime}(\tau_{j})=-\frac{1}{2}(\gamma_{j}+s_{j}^{-2}(0)\lambda_{j}(0))^{-3/2}(-8\gamma_{j}s_{j}^{-2}(0)\lambda_{j}(0))$$ $$=-\frac{1}{2}(2\gamma_{j})^{-3/2}(-8\gamma_{j}^{2})$$ $$=(2\gamma_{j})^{1/2}.$$

## 457 **C Derivatives**

20 2 l d l/dt d s/dt d /dt 10 2 s 0 500 1000 1500 2000 2500 3000 3500 4000 Step 0 0 500 1000 1500 2000 2500 3000 3500 4000 Step 0.0 0.6 1.2 0 500 1000 1500 2000 2500 3000 3500 4000 Step 0.0 2.5 5.0 2 l sl ss dsl dt dss dt 1/ l 1/ s sj 2 s d sj d t 120 2 h d l/dt d h/dt d /dt 60 2 l 0 0 50 100 150 200 250 Step
¡60 0 100 200 300 400 500 Step 0.0 0.2 0.4 0 100 200 300 400 500 Step 0 6 12 2 h 2 l d sj d t dsl dt dsh dt 1/ h 1/ l sj sl sh

## 458 **D Subspace Alignment**

0 500 1000 1500 2000 2500 3000 3500 4000 Step 0.00 0.25 0.50 0.75 1.00 Subspace Alignment 1.0 Subspace Alignment 0.5 0 100 200 300 400 500 Step 0.0

## 459 **E Orthogonal Feature Learning**

ProjU
( 2)(Wel)
ProjU
( 2)(Wes)
0.0 0.2 0.4 0.6 0.8 1.0 u 2 0500 1000 1500 2000 2500 3000 3500 4000 Step 0.0 0.2 0.4 0.6 0.8 1.0 u1 ProjU
( 2)(Wel)
ProjU
( 2)(Weh)
0.0 0.2 0.4 0.6 0.8 1.0 u 2 0 100 200 300 400 500 Step 0.0 0.2 0.4 0.6 0.8 1.0 u1

## 460 **F Right Singular Vectors Of** W

0 20 40 60 80 v1 index
¡0.2 0.0 0.2 0 20 40 60 80 v1 index
¡0.1 0.0 0.1 0 20 40 60 80 v1 index
¡0.1 0.0 0.1 0 20 40 60 80 v2 index
¡0.2 0.0 0.2 0 20 40 60 80 v2 index
¡0.2 0.0 0.2 0 20 40 60 80 v2 index
¡0.1 0.0 0.1

## 461 **G Non-Linear Experiments**

0 50 100 150 200 250 300 350 400 Step 0.0 0.5 1.0 1.5 2.0 0 50 100 150 200 250 300 350 400 Step 0.00 0.25 0.50 0.75 1.00 Eigenvaluel s Loss 0 50 100 150 200 250 300 350 400 Step 0.0 0.5 1.0 1.5 2.0 0 50 100 150 200 250 300 350 400 Step 0.00 0.25 0.50 0.75 1.00 Eigenval ue h l Loss