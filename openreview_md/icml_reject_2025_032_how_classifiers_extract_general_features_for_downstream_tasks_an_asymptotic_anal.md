000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054

## 1. Introduction

1

# How Classifiers Extract General Features For Downstream Tasks: An Asymptotic Analysis In Two-Layer Models

## Anonymous Authors1 Abstract

Neural networks learn effective feature representations through intermediate layers, enabling feature transfer without additional training for new tasks. However, the conditions for successful feature transfer remain underexplored. In this paper, we investigate feature transfer in classifier-trained networks, focusing on clustering in unseen distributions. In binary classification, we find that higher similarity between training and unseen distributions improves Cohesion and Separability, while Separability further requires unseen data to be assigned to different training classes. In multiclass classification, our analysis shows that the feature extractor maps input point based on their similarity to training classes, i.e. that unrelated training classes to input have negligible impact on feature extraction. We validate our theoretical findings in synthetic dataset and demonstrate practical applicability utilizing ResNet and variations of CAR, CUB, SOP, ISC, and ImageNet datasets.

Neural networks have the remarkable ability to adapt to specific tasks, learning representations through penultimate layers. Training these intermediate layers is crucial for neural network generalization (Damian et al., 2022). Also, these layers can extract semantically meaningful and transferable features from new data, enabling feature transfer for new tasks (Yosinski et al., 2014; Kornblith et al., 2019). A wide range of techniques, from open set clustering (Roth et al., 2020; Huang et al., 2024) to vision-language models (Li et al., 2023) and language models (Brown et al., 2020; Kojima et al., 2023), leverage feature transfer for downstream tasks. However, the specific conditions where features can be effectively transferred remain underexplored.

1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

Among various applications, classification based visual open-set clustering (Musgrave et al., 2020) serves as a fundamental benchmark for evaluating whether a feature extractor can generalize to unseen data. Typically, this task involves classifier training on one set of classes and then testing it on disjoint classes to assess whether the extracted features form cohesive and separable class-wise clusters on unseen data (Wang et al., 2018; Seidenschwarz et al., 2021; Deng et al., 2022). Given this context, we aim to investigate feature clustering with the following research questions:
Can we **capture the presences of feature learning** in classification and **identify the conditions where** features cluster effectively on new distributions?

To address this question, we analyze a two-layer nonlinear network network trained with a single large gradient descent step on a mean-squared classification loss in the proportional regime (in section 2). The proportional regime intuitively represents a scenario where the network width and the size of the dataset are of similar scales, aligning with common practices in model scaling (Ba et al., 2022), and they are known to effectively capture the phenomena occurring during the actual training process, as demonstrated in studies such as Mei & Montanari (2020); Moniri et al. (2024). We capture that the dominant part of the trained feature is composed of random initialization and *spikes* (Def. 3.4) associated with the training classes (section 3). Leveraging dominant features, we identify conditions for effective clustering on new distributions (section 4). In a binary classification setting, we assess the intra-class cohesion and inter-class *separability* of trained features in a numerical-analytical manner representing the clustering population risks (Def. 4.3) (Clemen ´ c¸con, 2011; Papa et al., 2015; Li & Liu, 2021) and goals for clustering performance (Liu et al., 2017). As a result, *Cohesion* increases as the *train-unseen similarity* (in Def. 4.1) grows larger. Meanwhile, for *Separability*, if classes classes are assigned (Notes 4.2, E.1) to different training classes, Separability increases as the *train-unseen similarity* grows larger; otherwise, it decreases, as illustrated in Figure 1.

Figure 1: Mapping data from the input space (left) to the learned feature space (right). Training classes are shown as balls, and unseen classes as dashed lines (a, b, p, n). Cohesion: Strong *cohesion* occurs for *a, p, n*, which have high similarity to the training classes compared to b. Separability of *a, n*: a and n, *assigned* to different training class, demonstrate high Separability. Separability of a, p: a and p, assigned to the same training class, exhibit low Separability.

055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 In the multi-class classification setting, we analyze the spikes of features and find that *spikes* map new inputs based on a linear combination of randomly initialized classifier heads' weight with coefficients that represent the similarity of the training classes. Therefore, the more *spikes* aligned with the input data the greater their contribution to feature extraction, enhancing the expressiveness of the features.

In the experiments, we empirically observe train-unseen similarity, cohesion, *Separability*, and *recall@1* under our theoretical assumptions in synthetic datasets. As a result, we confirm that the theoretical interpretation aligns with the actual findings (subsection 5.2). Additionaly, we explore practical metric learning settings and find evidence supporting the validity of our analysis results in a practical setup
(subsection 5.4). In most cases, we observe that clustering performance is higher when the unseen classes share the same sementic domain as the training classes. Moreover, adding semantically relevant training classes improves performance, whereas adding unrelated training classes does not lead to performance improvement. Our contributions are summarized into following: - We analyze the classifier feature, providing insights into how feature extractors operate:
- Higher *train-unseen similarity* increases *cohesion*.

- Higher *train-unseen similarity* increases separability between data *assigned* to different classes but reduces it otherwise.

- Expressiveness of feature improves with an increased number of *spikes* non-orthogonal to input.

- We generalize the distribution assumption of prior works and present novel proof techniques for classifier analysis.

- The theoretical results are validated through diverse experiments, including synthetic and real-world datasets.

## 1.1. Related Works

Metric Learning and Open Set Clustering Metric learning is proposed to cluster visually similar unseen classes using classification or triplet loss (Movshovitz-Attias et al., 2017; Zhai & Wu, 2019; Boudiaf et al., 2021). Several recent approaches have focused on increasing the number of classes in the training data to improve clustering. One approach adds virtual classes (Chen et al., 2018; Qian et al., 2020; Gu et al., 2021). Another approach suggested leveraging a larger number of classes induced from Schuhmann et al. (2021) to achieve state-of-the-art performance (An et al., 2023). This aligns with our analysis, which suggests that performance improves as the number of relevant classes in clustering increases. Neural Collapse (NC) and Unconstrained Layer-Peeled Model (ULPM) Recent studies have introduced the concept of Neural Collapse (Papyan et al., 2020) to explain the emergence of intra-class features and feature-weight alignment in trained neural networks. Several studies propose the ULPM to understand training dynamics of NC treating features and weights as unconstrained free variables (Fang et al., 2021; Zhu et al., 2021; Ji et al., 2022; Tirer & Bruna, 2022). However, ULPM, unlike the two layer network model we use, assumes the free variable features, which limits analyzability about input distribution and, consequently, prevents studying feature transferability. Feature Learning in Two-Layer Networks Many works (Louart et al., 2017; Goldt et al., 2020; Hu & Lu, 2022) study the Conjugate Kernel (CK), which enables the analysis of the structure of the first layer in two-layer networks. Ba et al. (2022); Moniri et al. (2024); Ba et al. (2023) argue that feature learning aids in reducing the population risk when evaluated on distributions same to the training data. Unlike these studies, we claim that the CK feature learning model not only explains this generalization but also enables the analysis of features from non-identical distributions, facilitating a deeper understanding of feature transfer. Additional related works are provided in Appendix A.

## 2. Problem Statement

Notations Let ∥·∥ be L
2 or the operator norm. Let ⊙ be the Hadamard product. Let A◦k be the Hadamard power.

Let *C, c >* 0 and κ ∈ R be constants that may change from line to line. Define [d] ≜ {1, 2, · · · , d}. For *o, O,* Θ
notations we follow Moniri et al. (2024) Training Data We define data for one vs. one classification with \#cls classes. The number of problem \#P ≜
\#cls(\#cls−1)
2. Let \#cls be the number of training classes, and let C1, · · · , C\#cls represent the class-conditional distributions of the training data. Define the training dataset as D = (*X, Y* ), where X ∈ R
n×d, Y ∈ [\#cls]
n, X = ({x ∼ C1} × m ∪ · · · ∪ {x ∼ C\#cls } × m), where
\#clsm = n and m is the number of instances per class. Let D˜ = (X, ˜ Y˜ ) an i.i.d. copy of D.

Network Structure We consider two-layer networks. The initial weight of the first layer, W0 ∈ R
d×N, is initialized as W0[i] ∼ *Unif*(S
d−1) for i ∈ [d]. We denote W obtained via a single step of gradient descent. The initial weights of the second layer, aij ∈ R
N for i, j ∈ [\#cls] s.t. *i < j*, are initialized as aij ∼ N(0, 1 N I). We define the initialized feature as F0(x) ≜ σ(W⊤
0 x) and the one-step trained feature as F(x) ≜ σ(W⊤x). The network output is defined as the following \#P -dimensional vector: F(x)
⊤aij |ij .

Proportional Regime We consider the two-layer neural networks in the proportional regime. n, d, and N are sample size, data and feature dimension, respectively. We perform our analysis under d/n, N/n → c as n, d, N → ∞. Optimization Problem Denote the set of all network parameters as θ = {W, a12, · · · , a\#P −1,\#P
}. Let Xij be a matrix in R
2m×d, where the first m rows contain samples x ∼ ci and the last m rows contain samples x ∼ cj . Let y ≜ [1, 1, . . . , 1, −1*, . . . ,* −1]⊤ ∈ R
2m be a vector consisting of m ones followed by m negative ones. To classify the given data, we use the Mean Squared Error,

$$L(x,y;\theta)=\frac{1}{2n}\sum_{i<j}^{c}\|y-\sigma(X_{i j}W)a_{i j}\|^{2}.\qquad(1)$$

The weight update formula for the first layer is given by W = W0 + G, where G ≜ −
∂L
∂w =Pi<j Gij , s.t.

$$G_{i j}=-\frac{1}{n}\Biggl[X_{i j}^{T}[(\sigma(X_{i j}W)a_{i j}-y)a_{i j}^{T}\odot\sigma^{\prime}(X_{i j}W)]\Biggr].\tag{2}$$

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 Now, we introduce the assumptions for theoretical analysis. Assumption 2.1 (Activation Function). Let σ(x) be an element-wise activation s.t. σ, σ′, σ′′ is bounded by λσ almost surely. It admits a Hermite decomposition i.e.

σ(z) = P∞
k=0 ckHk(z), where ck =
1 k!

E[σ(z)Hk(z)] for standard gaussian z. We assume c0 = 0, c1 > 0 and c 2 kk! ≤ Ck−3/2−w, for constants *C, w >* 0. For example, Shifted ReLU max(x, 0) − √
1 2π satisfies this condition.

Assumption 2.2 (Training Data). Let the class-conditional training data distributions Ci be non-centered Sub- Gaussians (Vershynin, 2018; Cao et al., 2021; Cole & Lu, 2024). This distribution family is suitable for classification, including distributions with limited support that are separable. It is an extension of the Gaussian assumption of Ba et al. (2022).

## 3. Feature Decomposition

This section analyzes the learning dynamics during a single gradient descent step. First, we demonstrate that the gradient with respect to the W0 exhibits an almost Rank-
\#P property within the proportional regime. Subsequently, we prove that the learned features can be predominantly expressed as Rank-\#P components, establishing the dominant components for subsequent analyses. Gradient Decomposition We decompose the gradient (equation 2) using Hermite decomposition, which allows us to extract the essential rank-one matrix structure for each ij-th classification problem. Note that σ
′ = c1 + σ
′
⊥.

$$G_{ij}=\frac{c_{1}}{n}X_{ij}^{T}ya_{ij}^{T}+\frac{1}{n}X_{ij}^{T}ya_{ij}^{T}\odot\sigma^{\prime}_{\perp}(X_{ij}W_{0})$$ $$-\frac{1}{n}X_{ij}^{T}\sigma(X_{ij}W_{0})(a_{ij}a_{ij}^{T})\odot\sigma^{\prime}(X_{ij}W_{0})\tag{3}$$ $$\triangleq\mathbb{A}_{ij}+\mathbb{B}_{ij}+\mathbb{C}_{ij}.$$

We derive the norm bound for the terms Aij , Bij , and Cij in Lemma I.1. Using these bounds, we establish the following Theorem 3.1. For the proof, please refer to Appendix I
Theorem 3.1 (Approximation of Gradient). Under the assumptions in section 2, and when n satisfies 12 > κlog2
√ n n
,
the following holds w.p. 1 − C(ne
−c log2 n + e
−cn):

$$\|G-\sum_{i<j}\mathbb{A}_{i j}\|\leq\kappa{\frac{\log^{2}\mathbf{n}}{\mathbf{n}}}.$$
(4)  $\frac{1}{2}$ .............................. 
n. (4)
Feature Decomposition Now we utilize Pi<j Aij to decompose the feature extractor. We decompose the one-step trained feature function F(x) = σ((W0 + G)
⊤x), which serves as a key step in deriving our main analysis. For the proof, please refer to Appendix J. Definition 3.2 (Data-Label Covariance). Data-Label Covariance for Xij is defined as βij =
1 nX⊤
ij y ∈ R
d.

Theorem 3.3 (Decomposition of Trained Features). Under the assumptions in *section 2, let* F0 = σ(XW˜0),
L ≜ log n, F
L
0 =PL
k=1 ckHk(XW˜0), and *spike*L = 
PL
k=1 c k1ck(X˜ Pi<j βija T
ij )
ok*. With probability* 1 − o(1),

$$F=F_{0}^{L}+s p i k e_{L}+\Delta.$$
$$({\boldsymbol{5}})$$
0 + *spike*L + ∆. (5)
Moreover, ∥spikeL∥ *is greater than* 
√n, ∥F
L
0 ∥ = Θ(√n),
and ∥∆∥ = o(
√n).

Based on these results, we analyze the feature representation using the approximation FL, which dominates the residual term ∥∆∥ = o(
√n) with probability 1 − o(1).

(a) Cohesion (b) Separability Figure 2: Numerical Observation of Cohesion and Separability. Plot of *Cohesion* and Heatmap of *Separability* calculated by adjusting β
⊤µ1 and β
⊤µ2.

Definition 3.4 (Dominant Feature FL = F
L
0 + spikeL).

$$F_{L}(x)\triangleq\sum_{k=1}^{L}c_{k}[H_{k}(\tilde{X}W_{0})+c_{1}^{k}(\sum_{i<j}(\beta_{ij}^{\top}x)a_{ij}^{T})^{c_{k}}].\tag{6}$$

## 4. Feature Analysis

Using the feature decomposition conducted so far, the next section analyzes clustering risk and explores the conditions for effective clustering of unseen data.

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 In this section, we analyze clustering risks. We show **train** (β)-**unseen** (µ) similarity governs the the clustering population risk i.e. *Cohesion* and *Separability* of FL from Definition 3.4 under condition 4.4. We derive *cohesion* and *separability* of FL for two "unseen" class-conditional distributions. Definition 4.1 (*Train-Unseen Similarity*). Given Train Data- Label Covariance β in Definition 3.2 and mean of Unseen distribution µ, *Train-Unseen Similarity* is defined as β
⊤µ.

Note 4.2 (Explanation of *assignment* and β
⊤µ). βij represents the normal vector of the linear decision boundary, i.e. the direction determining class i vs. j based on the sign of its inner product with data. Therefore, the sign of β
⊤µ indicates the class *assignment* of unseen data with µ.

Definition 4.3 (*Cohesion* and *Separability*). We define the clustering risks based on similarity between feature vectors using inner products. Cohesion measures the expected similarity between i.i.d. features of the same class over network parameters θ and data *x, x*′ ∼ c1, i.e.

$$\mathbb{E}_{\theta}[\mathbb{E}_{x\sim e_{1}}F(x)^{T}\mathbb{E}_{x^{\prime}\sim e_{1}}F(x^{\prime})].$$

Separability measures the expected dissimilarity between independent features of different classes over θ, x ∼ c1 and x
′ ∼ c2 i.e.

$$-\mathbb{E}_{\theta}[\mathbb{E}_{x\sim e_{1}}F(x)^{T}\mathbb{E}_{x^{\prime}\sim e_{2}}F(x^{\prime})].$$

Condition 4.4. We fix n, d, N large enough. Under assumptions 2.1, 2.2, let ci = N(µi, Id) for i ∈ [2]
be the class conditional distributions. Define ρ
(1)
k,k′ >
0, ρ
(2)
k,k′ (cos(µ1, µ2)), ρ
(3)
k,k′,r > 0, ρ
(4)
k,k′*,r,r*′ > 0 as functions of *N, d*. Note that ρ
(2)
k,k′ increases as cos(µ1, µ2)
grows. Exact definitions are in Def. K.1. The Shifted ReLU, as stated in Assumption 2.1, is used as the activation.

Proposition 4.5 (Cohesion). Following condition *4.4, the* Cohesion of FL for ci, i ∈ [2] *is given by:*

$$\sum_{\begin{subarray}{c}\text{$\sum_{1,k^{\prime}=1}^{L}$}}^{L}c_{k}c_{k^{\prime}}\left[\begin{matrix}\rho_{k,k^{\prime}}^{(1)}\|\mu\|^{k+k^{\prime}}\\ +2\sum_{r^{\prime}=0}^{k^{\prime}}\rho_{k,k^{\prime},r^{\prime}}^{(3)}\|\mu^{T}\beta|^{k^{\prime}-r^{\prime}}\|\beta\|^{r^{\prime}}\|\mu\|^{k}\\ +\sum_{r,r^{\prime}=(0,0)}^{(k,k^{\prime})}\rho_{k,k^{\prime},r,r^{\prime}}^{(4)}\|\mu^{T}\beta|^{k+k^{\prime}-r-r^{\prime}}\|\beta\|^{r+r^{\prime}}\end{matrix}\right]\tag{1}$$
$\eqref{eq:walpha}$. 
Proposition 4.6 (Separability). Following condition *4.4, the* Separability of FL for c1, c2 *is given by:*

k=1,k′=1 ckck′   ρ (2) k,k′ (cos(µ1, µ2))∥µ1∥ k∥µ2∥ k ′ +Pk r=0 ρ (3) k,k′,r|µ T 1 β| k−r∥β∥ r ′∥µ2∥ k ′ −X L +Pk ′ r ′=0 ρ (3) k,k′,r′ |µ T 2 β| k ′−r ′∥β∥ r ′∥µ1∥ k + (k,k′  P ) r,r′=(0,0) ρ (4) k,k′,r,r′ (µ T 1 β) k−r(µ T 2 β) k ′−r ′∥β∥ r+r ′. (8)
$$({\mathfrak{s}})$$
The proofs of Propositions 4.5 and 4.6 are provided in Appendix K. We numerically analyze the results of propositions 4.5 and 4.6 to investigate *Cohesion* and Separability further. For this numerical observations, we set
∥µ1∥ = ∥µ2∥ = ∥β∥ = 1, µ1 = −µ2 ∈ R
320000 and L = log10 n. We calculate equation 7 and equation 8 by adjusting µ T
1 β and µ T
2 β, as shown in Figure 2, which demonstrates the *Cohesion* and Separability of FL. *Cohesion* increases when the |µ T β| increases. *Separability* increases when µ T
1 β and µ T2 β grow with opposite signs and decreases when they grow with the same sign. Moreover, we observe that this phenomenon is governed by the last term of equation 7, 8 (related to ρ
(4)) , as shown by separately computing this term and the others numerically in Appendix B. Additionally, under the theoretical setup, we observe that our hypothesis tends to hold over a wider range as n increases (please refer to Appendix B). The analytical results in equation 7 and equation 7 can be explained as follows. With ρ
(4) > 0, the last term inside the bracket of *Cohesion* in equation 7 increases in value as Train-Unseen Similarity grows. The last term of Separability is influenced by (µ T
1 β)
k−r(µ T
2 β)
k
′−r
′. Provided that k − r and k
′ − r
′are odd, this term implies that if the Train-
Unseen Similarities have opposite signs and increase, then this term improves; otherwise, if the signs are the same and increase, *Separability* decreases. According to the analysis 4.1. Clustering Risk Analysis in binary classification

x1 x2 β1 β2 β3 a1 β⊤
1 x1 =β⊤
1 x2 β⊤ β 2 x1
⊤
2 x2 a2 β⊤ β 3 x1
⊤
3 x2 a3 β⊤
1 x1 =β⊤
1 x2 a4 β4
in Appendix H, the first coefficient c1 of Shifted ReLU is a large positive value, and subsequent Hermite coefficients approach zero while oscillating around it. Thus, we hypothesize that the positive part is likely to dominate Pckck′ ,
but further work is needed to confirm this.

Thoery FL Two-Layer Network F
220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 In this section, based on the previous feature decomposition and extend it to examine the impact of a multi-class classifier's spike structure on unseen data clustering. We examine the spike structure in FL = F
L
0 + spikeL and its influence on feature mapping. This examination allows us to explore the impact of the training data's structure β on the feature generation of unseen data. The spike structure inside the Hadamard power involves the linear combination coefficient β
⊤
ijx and the random initialized classifier head aij (equation 3.4). Thus, the feature extraction is closely linked to the inner product between βij and the input point x. If the direction of x is not orthogonal to βij , then spike of βij involve feature extraction.

Figure 5: Comparison of log average slope between Theory and Two-layer Networks. Midpoint (β1) Interpolation (β2) Extrapolation (β3) Orthogonal (β4). The intersection implies learning intersecting β. Conversely, when x is orthogonal to βij , the impact of spike βij is eliminated. To validate this, we define following four spikes, given test input x1, x2 ∈ S
d−1(
√d),
β1 =
x1+x2 2, β2 =
x1+3x2 4, β3 =
−x1+5x2 4and β4, a random vector orthogonal to x1, x2. Then, the magnitudes are adjusted to 
√d. By definition, β1, β4 cannot contribute to feature extraction because they are Midpoint or Orthogonal, while β2 and β3 can distinguish the two inputs. For illustration see Figure 3.

Now, we demonstrate this explanation using the approximated features FL and the two layer neural network F with the four disjoint sub-classification problem 1 defined as follows: We generated four classification problems by creating Gaussian training data with means βi and −βi, and a covariance of 0.1I for n, d, N = 211, enabling the networks to learn βi as their *spike*. F is trained by this data and FL
is calculated by its definition. We observed the feature distance between F(x1), F(x2) and between FL(x1), FL(x2)
for 4k combinations of βiin this problem by varying the angle between x1, x2. Please refer to Figure 4 and 21 for results. It can be observed that the feature from β1 and β4 hardly captures variations in the angle of test input x1, x2 within the data space. In contrast, the feature from β2 and β3 is highly sensitive to such variations, suggesting that it effectively preserves the structural changes in the input data. Both FL(x1) and F(x1) exhibit the same trends, which supports the validity of our feature approximation. To aggregate these combinatorial results, we measure the log of the average slope, which indicates that features with sensitive changes tend to have larger values, as shown in Figure 5.

As a result in Figure 5, we observe that when multiple βs are used in training, features are more sensitive to changes 1Instead of studying all combinations for 8 classes classification, we simplify the task by grouping four pairs, performing only four combinations of classifications.

275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329

Train Data 1, 2, 3 Eval 1, 2
in distance within the data space. Meanwhile, the Midpoint β1 and Orthogonal spike β4 seem ineffective for feature extraction, even when learned alongside other spikes. Experiments show that learning representations with unrelated classes limits expressiveness, while related classes enhance the model's ability to capture fine-grained features of unseen data. This trend is consistently observed in real-world datasets in Expr V, VI at subsection 5.4. Additionally, to clarify the effect of the spikes, we compute F
L
0and spikeL
separately as shown in Figure 22. The results show that the spikeL created by β1 and β4 embeds x1 and x2 as the same feature. Therefore, it confirms that the distinction between x1 and x2 created by the model trained with β1 and β4 is due to the random feature F
L
0.

## 5. Experiments

Remark 5.1. *recall@1* ≜ Exi,yi1yi=ˆyi,1-NN . yˆi,1-NN is class of the closest feature to xi. This is a feasible measure for evaluating whether new classes form clusters. In this section, we conduct seven experimental setups to validate our theoretical results. First, in Experiments I, II and III, we utilize a synthetic dataset to confirm that, as discussed in subsection 4.1, Cohesion, *Separability* are determined by the *Train-unseen similarity*. Second, to demonstrate how our theoretical explanations can provide intuition in practical settings, we conduct Experiments IV, V, VI, and VII. For this purpose, we analyze the open-set clustering problem using fine-grained real image datasets.

## 5.1. Setup For Theory Vaildation: Expr. I, Ii, Iii

We use three types of different non-centered Sub-Gaussian distributions as training datasets that are symmetric about the origin. For the evaluation, we introduce two distribution i.e. Eval 1, Eval 2 with translation parameter e and rotation parameter R ∈ R ⊆ SO(n) to control the train-unseen similarity β
⊤µ. e.g. as e increases from 0 towards 1, β
⊤µ increases, and as R approaches the identity matrix I, β
⊤µ increases. For illustration of the data, see Figure 6. For detail, refer to subsection D.1. We follow the condition described in section 2 and subsection 4.1.

Now we explain Expr. I, II, III. For each experiment, we utilize all datasets 1, 2, 3, with distinct Eval data usage. Expr. I uses two Eval 1 data with translation parameter e1 ∈ [−0.9, 0.9] and e2 = −e1, so they are *assigned* to opposite training classes (say pos-neg). Experiments II and III are based on two Eval 2 data distributions, each parameterized by a small-angle random rotation matrix R ∈ R. In Experiment II, considering the case where the datasets are *assigned* to opposite classes, the first distribution uses R and the second distribution is origin symmetry of the first distribution. In Experiment III, considering the situation where the datasets are *assigned* to the same class (say pospos), the first distribution uses R and the second uses R⊤ to slightly rotate given means.

## 5.2. Results Of Theory Vaildation: Expr. I, Ii, Iii

In this experiment, we examine the relationships between the *train-unseen similarity*( i.e. β
⊤µ), Cohesion, Separability that we discussed in subsection 4.1 and *Recall@1* to evaluate performance using practical measures. All test data are generated symmetrically, so for simplicity in visualization, we report the measurement for a single class. For Expr I, we present a summary of the results in Figure 8. We observe that for large values of |β
⊤µ|, strong *Cohesion* and Separability occur across all datasets. For Expr II and III, in accordance with the *Separability* structure observed in subsection 4.1, when the signs of β
⊤µ1, β⊤µ2 are opposite
(Expr II), we observed an increase in *Separability*, whereas in the other case (Expr III), we observed a decrease Figure 7. For *recall@1*, we observed a similar trend as *Separability*. These results correspond to our theoretical findings. For individual graphs, refer to Appendix D.

## 5.3. **Setup For Practical Vaildation: Expr. Iv, V, Vi, Vii**

We designed experiments to examine whether these insights are also applicable to clustering performance in image datasets and practical neural networks. In these scenarios, we utilize *train-unseen similarity* to conceptualize semantic similarity between training and unseen classes (Expr. IV). The number of non-orthogonal *spikes* is interpretable as the number of semantically similar or dissimilar training classes

330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384
(d) Cohesion: Expr II (e) Separability: Expr II (f) Recall@1(IP): Data 1, , Expr II
(a) Cohesion: Expr III (b) Separability: Expr III (c) Recall@1(IP): Expr III
Figure 7: Data 1 evaluated in the Eval 2 setup. Upper row: In Expr II, all metrics increase as |β
⊤µ| increases. Lower row:
In Expr III, where two test classes are *assigned* to a single train class, recall@1 and Separability tend to decrease as |β
⊤µ| increases. This aligns with our predictions. The red line represents the values after one step training. Tje blue line represents the values from initialization. (Expr. V, VI). Additionally, we validate whether removing the duplicatively *assigned* unseen classes improve clustering risk compared to random removal, as suggested by the results of *Separability* (Expr. VII). For this investigation, we used the benchmark datasets CAR(Vehicle) (Krause et al., 2013), CUB(Bird) (Wah et al., 2011), SOP(Product) (Song et al., 2015), and ISC (Clothing) (Liu et al., 2016), referred to as *Domain*. Additionally, we utilized ImageNet subsets corresponding to the domains Vehicle, Bird, Product, and Clothing, denoted as I(V), I(B), I(P), and I(C), referred to as *sub In1k* for extra classes. Also, we performed experiments on the whole classes ImageNet by sampling 100 instances per class (say subsampled whole In1k). Details are in Appendix N. The objective function and most experimental configurations followed the approach outlined in Zhai & Wu (2019), which is a seminal baseline. We use ResNet18 and ResNet50 (He et al., 2015). In addition to the randomly initialized networks in the main text, we conducted experiments with pre-trained networks common in feature learning, and results are included in Appendix E. The two setups exhibited similar trends.

ResNet18 (init) ResNet50(init)
For **Expr. IV**, we trained with each *Domain* dataset (CAR, CUB, SOP, and ISC train datasets) and Domain+*sub In1k* dataset (CAR+I(V), CUB+I(B), SOP+I(P), and ISC+I(C)), and then measured how each model well cluster on all of the test datasets (CAR, CUB, SOP, ISC test datasets). As shown in Figure 9, we verify whether clustering the test dataset related to the train classes is more effective than clustering unrelated data, analogous to result in subsection 4.1.

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 In **Expr. V**, we measured the clustering performance for corresponding test datasets after learning the Domain, Domain+*sub In1k*, and Domain+*subsampled whole In1k*. We find that adding classes from the entire ImageNet dataset during training, rather than including only related classes, does not significantly improve clustering (Figure 10, 32). In **Expr. VI**, experiments are conducted by dividing the Domain datasets into four steps to observe the impact of increasing the number of related classes on *recall@1* performance (Figure 11). From Step 0 to Step 3, 25%, 50%, 75%,
and 100% of the *Domain* dataset classes are sequentially added for training. The added classes are randomly selected, and each experiment is repeated three times. For the number of classes, refer to Table 6. Furthermore, we observed that some results of Expr. V align with those of Expr. VI, as discussed in detail in subsection E.1.

For **Expr. VII**, in evaluation, removing duplicatively assigned of unseen classes resulted in a 1.73 ± 2.87% improvement in recall@1 compared to random removal of same amount of unseen classes, with max improve: 13.65%, min decrease: -3.28%, a success rate: 79% and p = 9.40×10−7.

This suggest that duplicate *assignments* hinder clustering, which aligns with our theory. Details are in subsection E.2.

## 6. Conclusion

In this study, we explored the feature learning dynamics of a two-layer classifier in the proportional regime to uncover the mechanisms underlying feature transferability. Specifically, we analyzed the conditions where the learned features of unseen classes form cohesive and separable cluster. Our theoretical analysis extends the Conjugate Kernel framework to classification tasks. As a result, our numerical-analytical theory demonstrated that feature *cohesion* increases with greater similarity between training and unseen data, while feature *separability* is influenced not only by similarity but also by avoiding duplicate class *assignments* in binary classification. Additionally, we showed that only when the spikes are non-orthogonal to the input, do they get involved in feature extraction. In addition to validation on synthetic datasets, we observed that our theory offers valuable insights even when applied to real-world datasets. Our empirical findings suggest that clustering performance improves when the test data share the same semantic domain as the training data. Furthermore, adding semantically relevant classes to the training set leads to performance gains, whereas introducing unrelated classes has little effect. Contrary to existing research that focuses on performance improvement through large-scale learning on broad domains (Brown et al., 2020; An et al., 2023), our study provides evidence that only certain relevant knowledge, closely related to the domain, influences feature transfer. This evidence mirrors classical problems in the field of artificial intelligence, such as the frame problem and the installation problem. Specifically, AI agents do not require all available knowledge to solve a given problem; only specific, detailed knowledge is necessary. Dennett (1984) states about this as follows: "People in AI ... take the shortcut of installing all that an agent has to know to solve a problem. This may, of course, be a dangerous shortcut." We hope that our study may remind the AI community of the longstanding principle that it may not be the scale of the data that matters. We have also discussed the limitations and future research directions related to the Hermite expansion approximation and general results for cohesion and *separability* in Appendix F.

## References

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 This paper presents work aimed at advancing the field of Machine Learning. In this research, we analyze the potential for clustering performance improvement through the classification training of a large number of highly granular classes. Such an approach may lead to a reduction in the level of personal data masking required for fine-grained data differentiation, which could trigger new ethical discussions regarding privacy protection. Additionally, to effectively implement this approach, there may be a tendency to collect more data, which can have significant implications for the scale and scope of data collection, as well as data management practices. An, X., Deng, J., Yang, K., Li, J., Feng, Z., Guo, J., Yang, J., and Liu, T. Unicom: Universal and compact representation learning for image retrieval, 2023. URL
https://arxiv.org/abs/2304.05884.

Ba, J., Erdogdu, M. A., Suzuki, T., Wang, Z., Wu, D.,
and Yang, G. High-dimensional asymptotics of feature learning: How one gradient step improves the representation, 2022. URL https://arxiv.org/abs/2205. 01445.

Ba, J., Erdogdu, M. A., Suzuki, T., Wang, Z., and Wu, D.

Learning in the presence of low-dimensional structure: A spiked random matrix perspective. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S.

(eds.), Advances in Neural Information Processing Systems, volume 36, pp. 17420–17449. Curran Associates, Inc., 2023.

Bai, Z. and Silverstein, J. W. Spectral Analysis of Large Dimensional Random Matrices. Springer New York, 2010. ISBN 9781441906618. doi: 10.1007/ 978-1-4419-0661-8. URL http://dx.doi.org/ 10.1007/978-1-4419-0661-8.

Bellet, A. and Habrard, A. Robustness and generalization for metric learning. *Neurocomputing*, 151:259–267, March 2015. ISSN 0925-2312. doi: 10.1016/j.neucom.2014. 09.044. URL http://dx.doi.org/10.1016/j.

neucom.2014.09.044.

Bienstman, P. Mathematics for photonics. Course Syllabus, September 2023. URL https:
//studiekiezer.ugent.be/studiefiche/
en/E002640/current. Course size: 4.0 credits, Study time: 120 hours. Offered in English and Dutch.

Boudiaf, M., Rony, J., Ziko, I. M., Granger, E., Pedersoli, M., Piantanida, P., and Ayed, I. B. A unifying mutual information view of metric learning: cross-entropy vs.

## Impact Statement

pairwise losses, 2021. URL https://arxiv.org/ abs/2003.08983.

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners, 2020. URL https:// arxiv.org/abs/2005.14165.

Cao, Y., Gu, Q., and Belkin, M. Risk bounds for over-parameterized maximum margin classification on sub-gaussian mixtures. In Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 8407–8418. Curran Associates, Inc.,
2021. URL https://proceedings.neurips.

cc/paper_files/paper/2021/file/ 46e0eae7d5217c79c3ef6b4c212b8c6f-Paper.

pdf.

Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, J., ´
Bojanowski, P., and Joulin, A. Emerging properties in self-supervised vision transformers, 2021. URL https: //arxiv.org/abs/2104.14294.

Chang, Y., Hu, C., and Turk, M. Manifold of facial expression. In 2003 IEEE International SOI Conference. Proceedings (Cat. No.03CH37443), pp. 28–35, 2003. doi: 10.1109/AMFG.2003.1240820.

Chen, B., Deng, W., and Shen, H. Virtual class enhanced discriminative embedding learning, 2018. URL https: //arxiv.org/abs/1811.12611.

Chopra, S., Hadsell, R., and LeCun, Y. Learning a similarity metric discriminatively, with application to face verification. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), volume 1, pp. 539–546 vol. 1, 2005. doi: 10.1109/CVPR.2005.202.

Clemen ´ c¸con, S. On u-processes and clustering performance. In Shawe-Taylor, J., Zemel, R., Bartlett, P., Pereira, F., and Weinberger, K. (eds.),
Advances in Neural Information Processing Systems, volume 24. Curran Associates, Inc., 2011. URL https://proceedings.neurips.

cc/paper_files/paper/2011/file/ a1d0c6e83f027327d8461063f4ac58a6-Paper.

pdf.

Cole, F. and Lu, Y. Score-based generative models break the curse of dimensionality in learning a family of sub-gaussian distributions. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=wG12xUSqrI.

Gu, G., Ko, B., and Kim, H.-G. Proxy synthesis: Learning with synthetic classes for deep metric learning, 2021. URL https://arxiv.org/abs/2103.15454.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Han, X. Y., Papyan, V., and Donoho, D. L. Neural collapse under mse loss: Proximity to and dynamics on the central path, 2022. URL https://arxiv.org/abs/ 2106.02073.

Damian, A., Lee, J. D., and Soltanolkotabi, M. Neural networks can learn representations with gradient descent, 2022. URL https://arxiv.org/abs/ 2206.15144.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition, 2015. URL https:// arxiv.org/abs/1512.03385.

Deng, J., Guo, J., Yang, J., Xue, N., Kotsia, I., and Zafeiriou, S. Arcface: Additive angular margin loss for deep face recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10):5962–5979, October 2022. ISSN 1939-3539. doi: 10.1109/tpami.2021. 3087709. URL http://dx.doi.org/10.1109/
TPAMI.2021.3087709.

Hu, H. and Lu, Y. M. Universality laws for high-dimensional learning with random features, 2022. URL https:// arxiv.org/abs/2009.07669.

Huai, M., Xue, H., Miao, C., Yao, L., Su, L., Chen, C.,
and Zhang, A. Deep metric learning: The generalization analysis and an adaptive algorithm. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI-19, pp. 2535–2541. International Joint Conferences on Artificial Intelligence Organization, 7 2019. doi: 10.24963/ijcai.2019/352. URL https: //doi.org/10.24963/ijcai.2019/352.

Dennett, D. Cognitive wheels: The frame problem of ai. 01 1984.

El-Nouby, A., Neverova, N., Laptev, I., and Jegou, H. Train- ´
ing vision transformers for image retrieval, 2021. URL
https://arxiv.org/abs/2102.05644.

Ermolov, A., Mirvakhabova, L., Khrulkov, V., Sebe, N., and Oseledets, I. Hyperbolic vision transformers: Combining improvements in metric learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7409–7419, 2022.

Huang, H., Nie, Z., Wang, Z., and Shang, Z. Cross-modal and uni-modal soft-label alignment for image-text retrieval, 03 2024. URL https://arxiv.org/pdf/ 2403.05261.pdf.

Hui, L., Belkin, M., and Nakkiran, P. Limitations of neural collapse for understanding generalization in deep learning, 2022. URL https://arxiv.org/abs/2202. 08384.

Fan, Z. and Wang, Z. Spectra of the conjugate kernel and neural tangent kernel for linear-width neural networks, 2020. URL https://arxiv.org/abs/ 2005.11879.

Isserlis, L. On a formula for the product-moment coefficient of any order of a normal frequency distribution in any number of variables. *Biometrika*, 12(1/2): 134–139, 1918. ISSN 00063444, 14643510. URL http://www.jstor.org/stable/2331932.

Fang, C., He, H., Long, Q., and Su, W. J. Exploring deep neural networks via layer-peeled model: Minority collapse in imbalanced training. Proceedings of the National Academy of Sciences, 118(43), October 2021. ISSN 1091-6490. doi: 10.1073/pnas.2103091118. URL http: //dx.doi.org/10.1073/pnas.2103091118.

Ji, W., Lu, Y., Zhang, Y., Deng, Z., and Su, W. J. An unconstrained layer-peeled perspective on neural collapse, 2022.

URL https://arxiv.org/abs/2110.02796.

Galanti, T., Gyorgy, A., and Hutter, M. On the role of ¨
neural collapse in transfer learning, 2022. URL https: //arxiv.org/abs/2112.15121.

Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners, 2023.

URL https://arxiv.org/abs/2205.11916.

Goldt, S., Loureiro, B., Reeves, G., Krzakala, F., Mezard, ´
M., and Zdeborova, L. The gaussian equivalence of gen- ´ erative models for learning with shallow neural networks. Proceedings of the 2nd Mathematical and Scientific Machine Learning Conference, PMLR 145:426-471 (2021),
06 2020. URL https://arxiv.org/pdf/2006. 14709.pdf.

Kornblith, S., Shlens, J., and Le, Q. V. Do better imagenet models transfer better?, 2019. URL https://arxiv. org/abs/1805.08974.

Krause, J., Stark, M., Deng, J., and Fei-Fei, L. 3d object representations for fine-grained categorization. In Proceedings of the IEEE International Conference on Computer Vision (ICCV) Workshops, June 2013.

Goodfellow, I., Bengio, Y., Courville, A., and Bengio, Y.

Deep learning, volume 1. MIT Press, 2016.

Kuchibhotla, A. K. and Chakrabortty, A. Moving beyond sub-gaussianity in high-dimensional statistics: applications in covariance estimation and linear regression. *Information and Inference: A Journal of the IMA*, 11(4):1389–1456, June 2022. ISSN 2049-8772. doi: 10.1093/imaiai/iaac012. URL http://dx.doi.org/ 10.1093/imaiai/iaac012.

Moniri, B., Lee, D., Hassani, H., and Dobriban, E. A
theory of non-linear feature learning with one gradient step in two-layer neural networks, 2024. URL https: //openreview.net/forum?id=MY8SBpUece.

550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 Movshovitz-Attias, Y., Toshev, A., Leung, T. K., Ioffe, S.,
and Singh, S. No fuss distance metric learning using proxies, 2017. URL https://arxiv.org/abs/1703. 07464.

Lee, K.-C., Ho, J., Yang, M.-H., and Kriegman, D. Videobased face recognition using probabilistic appearance manifolds. In 2003 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2003. Proceedings., volume 1, pp. I–I, 2003. doi: 10.1109/CVPR. 2003.1211369.

Musgrave, K., Belongie, S., and Lim, S.-N. A metric learning reality check, 2020. URL https://arxiv.org/ abs/2003.08505.

O'Donnell, R. Analysis of boolean functions, 2021. URL
https://arxiv.org/abs/2105.10386.

Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023. URL https://arxiv.org/abs/2301.12597.

Papa, G., Clemen ´ c¸on, S., and Bellet, A. Sgd algorithms based on incomplete u-statistics: Large-scale minimization of empirical risk. In Cortes, C., Lawrence, N., Lee, D., Sugiyama, M., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015. URL https://proceedings.neurips.

cc/paper_files/paper/2015/file/
67e103b0761e60683e83c559be18d40c-Paper.

pdf.

Li, S. and Liu, Y. Sharper generalization bounds for clustering. In Meila, M. and Zhang, T. (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 6392–6402. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/ v139/li21k.html.

Liaw, C., Mehrabian, A., Plan, Y., and Vershynin, R. A
simple tool for bounding the deviation of random matrices on geometric sets, 2016. URL https://arxiv.org/ abs/1603.00897.

Papyan, V., Han, X. Y., and Donoho, D. L. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40):24652–24663, September 2020. ISSN
1091-6490. doi: 10.1073/pnas.2015509117. URL http: //dx.doi.org/10.1073/pnas.2015509117.

Liu, W., Wen, Y., Yu, Z., and Yang, M. Large-margin softmax loss for convolutional neural networks, 2017. URL https://arxiv.org/abs/1612.02295.

Qian, Q., Shang, L., Sun, B., Hu, J., Li, H., and Jin, R. Softtriple loss: Deep metric learning without triplet sampling, 2020. URL https://arxiv.org/abs/ 1909.05235.

Liu, W., Wen, Y., Yu, Z., Li, M., Raj, B., and Song, L.

Sphereface: Deep hypersphere embedding for face recognition, 2018. URL https://arxiv.org/abs/ 1704.08063.

Roth, K., Milbich, T., Sinha, S., Gupta, P., Ommer, B., and Cohen, J. P. Revisiting training strategies and generalization performance in deep metric learning. In International Conference on Machine Learning, pp. 8242–8252.

PMLR, 2020.

Liu, Z., Luo, P., Qiu, S., Wang, X., and Tang, X. Deepfashion: Powering robust clothes recognition and retrieval with rich annotations. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

Schuhmann, C., Vencu, R., Beaumont, R., Kaczmarczyk, R., Mullis, C., Katta, A., Coombes, T., Jitsev, J., and Komatsuzaki, A. Laion-400m: Open dataset of clipfiltered 400 million image-text pairs, 2021. URL https:
//arxiv.org/abs/2111.02114.

Louart, C., Liao, Z., and Couillet, R. A random matrix approach to neural networks, 2017. URL https:// arxiv.org/abs/1702.05419.

Mei, S. and Montanari, A. The generalization error of random features regression: Precise asymptotics and double descent curve, 2020. URL https://arxiv.org/
abs/1908.05355.

Seidenschwarz, J., Elezi, I., and Leal-Taixe, L. Learning ´
intra-batch connections for deep metric learning, 2021.

URL https://arxiv.org/abs/2102.07753.

Sohoni, N., Dunnmon, J., Angus, G., Gu, A., and Re, C. ´
No subclass left behind: Fine-grained robustness in coarse-grained classification problems. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 19339–19352. Curran Associates, Inc., 2020. URL https://proceedings.neurips. cc/paper_files/paper/2020/file/ e0688d13958a19e087e123148555e4b4-Paper. pdf.

Song, H. O., Xiang, Y., Jegelka, S., and Savarese, S.

Deep metric learning via lifted structured feature embedding, 2015. URL https://arxiv.org/abs/ 1511.06452.

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 Szego, G. ˝ *Orthogonal Polynomials*. American Math.

Soc: Colloquium publ. American Mathematical Society, 1975. ISBN 9780821810231. URL https://books. google.co.kr/books?id=ZOhmnsXlcY0C.

Talwalkar, A., Kumar, S., and Rowley, H. Large-scale manifold learning. In 2008 IEEE Conference on Computer Vision and Pattern Recognition, pp. 1–8, 2008. doi: 10.1109/CVPR.2008.4587670.

Tirer, T. and Bruna, J. Extended unconstrained features model for exploring deep neural collapse, 2022. URL
https://arxiv.org/abs/2202.08087.

Vershynin, R. Introduction to the non-asymptotic analysis of random matrices. Chapter 5 of: Compressed Sensing, Theory and Applications. Edited by Y. Eldar and G. Kutyniok. Cambridge University Press, 2012, 11 2010. URL https://arxiv.org/pdf/1011.3027.pdf.

Vershynin, R. High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2018.

Vignat, C. A generalized isserlis theorem for location mixtures of gaussian random vectors, 07 2011. URL
https://arxiv.org/pdf/1107.2309.pdf.

Wah, C., Branson, S., Welinder, P., Perona, P., and Belongie, S. The caltech-ucsd birds-200-2011 dataset. 2011.

Wang, H., Wang, Y., Zhou, Z., Ji, X., Gong, D., Zhou, J.,
Li, Z., and Liu, W. Cosface: Large margin cosine loss for deep face recognition, 2018. URL https://arxiv. org/abs/1801.09414.

Yang, Y., Steinhardt, J., and Hu, W. Are neurons actually collapsed? on the fine-grained structure in neural representations, 2023. URL https://arxiv.org/abs/
2306.17105.

Yosinski, J., Clune, J., Bengio, Y., and Lipson, H. How transferable are features in deep neural networks?, 2014. URL https://arxiv.org/abs/1411.1792.

Zavatone-Veth, J. A., Yang, S., Rubinfien, J. A., and Pehlevan, C. Neural networks learn to magnify areas near decision boundaries, 2023.

Zhai, A. and Wu, H.-Y. Classification is a strong baseline for deep metric learning, 2019. URL https://arxiv. org/abs/1811.12649.

Zhou, J., Li, X., Ding, T., You, C., Qu, Q., and Zhu, Z. On the optimization landscape of neural collapse under mse loss: Global optimality with unconstrained features, 2022.

URL https://arxiv.org/abs/2203.01238.

Zhou, J., Wang, P., and Zhou, D.-X. Generalization analysis with deep relu networks for metric and similarity learning, 2024. URL https://arxiv.org/abs/2405. 06415.

Zhu, Z., Ding, T., Zhou, J., Li, X., You, C., Sulam, J., and Qu, Q. A geometric analysis of neural collapse with unconstrained features, 2021. URL https://arxiv. org/abs/2105.02375.

## A. Additional Related Works

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 Feature Transferability in Deep Metric Learning The explanation for how Deep Metric Learning learns transferable features towards unseen data remains insufficient. Chopra et al. (2005) suggested that CNNs' robustness to geometric distortions enables the creation of generalizable features. This explanation has been replaced in transformer-based research by the idea that, without the inductive biases of CNNs, transformers are less constrained and thus capable of extracting generalizable features (El-Nouby et al., 2021; Caron et al., 2021). Additionally, following the manifold hypothesis (Chang et al., 2003; Lee et al., 2003; Talwalkar et al., 2008; Goodfellow et al., 2016), Liu et al. (2018); Ermolov et al. (2022)
explained that normalized softmax for metric learning works well because hyperspherical/hyperbolic feature space and the data lies on a manifold. However, these studies do not provide a detailed analysis of how features are learned and transferred through classification.

Neural Collapse (NC) and Features learned by Classifiers There exist studies exploring Neural Collapse (NC) and features learned by classifiers that cannot be explained under the free variable assumption. Hui et al. (2022) argue that NC does not manifest on test data. Sohoni et al. (2020); Yang et al. (2023) claim that even on training data, NC is not fully realized, with critical fine-grained structures concealed. Notably, Yang et al. (2023) utilized a two-layer network to analyze training data features. Regarding NC on novel data, Galanti et al. (2022) statistically analyze NC in transfer learning, suggesting that NC generalizes not only to new samples within training classes but also to unseen classes with empirical observations. However, their analysis is constrained by focusing on general function spaces rather than specific neural network architectures. MSE for Classification Utilizing MSE in classification is as well-established as using softmax-cross entropy, especially in theoretical analyses of classification problems (Han et al., 2022; Zhou et al., 2022). Generalization Bound for Metric Learning Research on the generalization bounds of metric learning related to the U-process we use is also ongoing (Bellet & Habrard, 2015; Huai et al., 2019; Zhou et al., 2024). However, these studies do not analyze the exact feature learning structure.

## B. Empirical Insights Into High-Dimensional Asymptotics

In asymptotic analysis, n, d, N → ∞ is crucial for observe result. Please see Figure 12, Figure 13 for the cohesion and Separability in R
2000, R
20000, R
320000. As the dimension increases, the range where cohesion and Separability align with our expectations expands. For component analysis, please see Figure 14, Figure 15, Figure 16 , Figure 17, Figure 18, Figure 19

## C. Additional Observation Of Multi Classes Feature Analysis

See Figure 21 for multi-directional training result. For F
L
0, and spikeL term depiced in Figure 22, Figure 23.

## D. Additional Results Of Two-Classes Experiments

D.1. Additional setup for Experiment I, II, III
We set d = n = N = 211 and use Shifted ReLU. We repeat each experiment with 3 different initializations of the neural network parameters.

Training Datasets (Data 1) two uniform distributions over a radius-√d ball, (Data 2) two multi-dimensional element-wise truncated Gaussian distributions, and (Data 3) two uniform distributions over a radius-√d sphere, symmetric about the origin 2. The two means of training class are denoted as v and −v, respectively. For Data 1, 3 v ≜ 2r · u, with u ∼ Unif S
d−1.

For Data 2, one class has support on [1, ∞) across all dimensions, while the other class has support on (−∞, −1].

Evaluation Datasets Eval 1, 2 use the projected Gaussian distribution, which is projected onto the mean direction of one training data v, as defined in equation 9. For Eval 1, we translate mean of projected Gaussian distribution with e, and 2The Sub-Gaussian property is proven for Data 1 and 3 in Vershynin (2018), and for Data 2 in Lemma L.1.

715

716 717 718 7 719 Cohesion 720 721 722 3 723 2.

e 10 724
-10
-S
s 725 726 727 728 eo 729 Cohesion
 
730 731 200 732
--
73
-So es

Let al s 100 734 735 736 737 2000 738 10000 Cion 739 ee 4000 740 2000 741 742 o
-400
-200 o 18 40 743 B
74 745 less 746 1.2 747 1.0 748 ol of 18.

749 750 0.2 751 a 752
-500 a re 500 1000 753 754 75 le10 1.2 756 1.0 757 an Construction
 1966 758 759 760 0.2 761 o 10000
-1000  -5000 a is 500 762 763 23 2 Cohesion
 1999 19 18 17.

10 e
-10
-
600 500 5, 400 300 Cahe 200 100
.

–50 e ls 100
−100 2000 15000 00

1015:470 5000 o 88
-400
-200 o 40 B T 
6000 50000 40000 3000 Cahes 2000 10000 o
-1000
-500 o 500 1000 le13 1,2 1.0 Cohesion
 Cohesion
 Cohesion 0.4 a2 o, o
-10000
-5000 5000 10000 a

Let a 423 422
:421 420 Cohes 419 418 417 es 10
-S
–10 1000 900 800 Cohesion 700 600 so 400 100
-100
- SO
o s 15000 12500 S 10000 Cohes 7500 5000 2500 o 400 40
-200 o 8 P
7000 6000 50000 us, 40000 e 2000 2000 10000
-1000
-500 e

Le Te 500 1000 1s9 2.0 g 1.5 e 1.0 as 00 a
Top socc 10000
764 765 766 767 768 769

Submission and Formatting Instructions for ICML 2025 10 10 10.0 770 7.6e+00 6.3e+00 6.3e+00 1.7e+02 7.5 71 5.0e+00 1.6e+02 5.5.0e+00 5 5 5.0 772 3.8e+00 1.6e+02 3.8e+00 2.5 2.5e+00 773 1.6e+02 2.5e+00 β l 2 h 1.3e+00 0 0.0 1.6e+02 o 1.3e+00 774 β B
0.0e+00 1.6e+02 0000000
-2.5 775
-1.3e+00 1.6e+02
-1.3e+00
-5
-5
−5.0
-2.5e+00 1.6e+02 776
-2.5e+00
-3.8e+00
-7.5 1.6e+02
-3.8e+00 777
----
-5.0e+00 1.5e+02
-10
-10
-5 o CONCRETE
s 10
–5 o 10 778 10 β β β 779 780 781 100 100 2.6e+02 100 7.6e+02 5.0e+02 782 1.3e+02 75 75 75 6.3e+02 3.8e+02 00 5.0e+02 783 59 50 50 2.5e+02
-1.3e+02 3.8e+02 784 25 25 25 1.3e+02
-2.6e+02 2.5e+02 l H2 l f 4 42 00 0 0 o 785
-3.8e+02 1.3e+02 p p p
-1.3e+02
-5.1c+02
–25
–25
-25 0.0e+00 786
-2.5e+02 66.4e+02
-1.3e+02
−50
–50
–50 787
-3,8c+02
-7.6e+02
-2.5e+02
−75
−75
-5.0e+02
−75 78
-8.9e+02
-3,8c+02
-100 6.3e+02
-100
-100 789
−50 o 50 100
−50 o 50 100
−50 0 50 100 β β β 790 791 792 8.5e+03 00 1.3e+04 400 400 400
-2.5e+04 4.2e+03 793 9.5e+03
-5.0e+04 00 6.3e+03 794 200 200 200 7.5e+04
-4.2e+03 3.2e+03
-1.0e+05
-8.5e+03 795 f f h h 2 55 0.0e+00 o
-1.2e+05 o 0
-1.3e+04 p t t
. B
-3.2e+03 796 1.1.5e+05
-1.7e+04
-6.3e+03 797
-200
–200
-200
-1.8e+05
-2.1e+04
-9.5e+03
-2.0e+05
-2.6e+04 798
-1.3e+04
-400
-2.2e+05
-400
-3.0e+04
-400 79 1.6e+04
-400 -200 200 400
-400 -200 o 200 400
-400 -200 0 200 400 0 800 ββ B T H1 β 801 802 1000 0000000 1000 00000 1000 5.0e+04 803
-3.8e+05
-1.1c+06 3.8c+04 804
-7.7e+05
-2.2e+06 2.5e+04 500 500 500
-1.2e+06
-3.3e+06 1.3e+04 805
-1.5e+06 00000
-4.4e+06 B T H2
  
  
806 o
-1.9e+06 o
-5.5e+06
-1.3e+04 p p 807
-2,3e+06
-2.5e+04
-6.6e+06
-2.7e+06
-7.7e+06
-3.8e+04 808
−500 5–500
−500
-3.1e+06
-8.8e+06
-5.0e+04 809
-3.5e+06
-9.9e+06
-6.3e+04
-1000
-1000 -500
-1000 -500 810
−500 o 500 1000 500 1000 500 1000 0 o ββ B T 
β 81 812 00000 000000 0000000 813 10000 10000 10000
-3,8c+09 1-1.4e+14 3.3e+10 814
-7.5e+09
-2.8e+14
+6666 5000 5000 5000 815
-1.1e+10
-4.3e+14
-1.0e+11
-1.5e+10
-5.7e+14
-1.3e+11 816 β 
  
H2 o
-1.9e+10
-7.1e+14 0
-1.7e+11 817 8 8
-2.3e+10
-8.6e+14
-2.0e+11 818
-2.6e+10
-2.3e+11 1.0e+15
−5000
−5000
−5000
-3.0e+10
-1.1e+15
-2.7e+11 819
-3.4e+10
-3.0e+11
-1.3e+15 820
-10000-5000
-10000-5000
-10000-5000 5000 10000 o 5000 10000 o 5000 10000 821 β Γ μι ββ β 822 823 824
15

825 600 826 600 827 600 500 500 828 400 400 Cohesion
 300
 
on Chesion 829 40 Chesis 300 830 200 200 831 200 100 832 100 833 o o o 100 100
-100
−50 o 50 100
-100
−50 0 50
-100
-50 o 50 834 β T µ1 β T µ1 β T μι 835 836 837 17.4 838 2.3 418.0 17.3 839 17.2 417.5 840 Cohesion
 2 Cohesion
17.0
17.

Cohesion

Cohesion 841 842 16.9 416.5 843 2.0 16.8 84 416.0 16.7 845 100 100
-100
-50 50
-50 50 o
-100 o
-100
−50 o 50 100 846 β T μι β β T µ1 847 848
849

851 852 853 2000 15000 854 8000 12500 15000 855 6000 10000 856 Cohesion Con Cohesion 10000 Cohesi 7500 857 40000 858 5000 5000 2000 859 2500 860 0 o 0 400 200 400 200 400
-400
-200 200
-400
-200 0
−400
-200 o 861 o β β T μι β 862 863 864 425.0 865 17.0 5 866 422.5 867 420.0 16.5 Cohesion Cohesion
 16.0 Cohesion
 Cohesion
 Caris o 868 4 869 415.0 870 3 412.5 871 15.5 872 410.0 2 200 400 200 400
-400
–200 873
-400
-200 o o
-200 o 200 400
-400 β T μι β T µ1 β 874 875
850 876 877 878 879 for Eval 2, we Rotate mean of projected Gaussian distribution with R ∈ R and fixed e. We generate 300 distinct rotation matrices R using the process in Appendix O. The projected gaussian distribution is sampled as follows,

z −
$$\cdot\,\frac{z^{\top}\nu\nu}{\|\nu\|^{4}}+\nu,\quad\mathrm{where}\quad z\sim\mathcal{H}(0,c I).$$
$$(9)$$
$\widehat{\varepsilon}^{\prime}$
$\neg$
+ ν, where z ∼ N(0*, cI*). (9)
For Eval 1, ν ≜ ev, c = 1 and for Eval 2, ν ≜ Rev, c = 10−1 with e = 0.01 for Data 2 experiment and e = 0.008 for Data 1 and 3 experiments, R ∈ SO(d).

## D.2. Comprehensive Results Of All Experiments

The overall experimental results for *Cohesion* and *Separability* are shown in Figure 24. The results for Eval 1 experimental settings are presented in linear scale in Figure 25 and in logarithmic scale in Figure 26. Additionally, as presented in Figure 7, experiments for Eval 2 settings on Data 2 and 3 are shown in linear scale in Figure 27, with results for *Cohesion*, Separability, and Recall@1 (IP). Furthermore, results for Recall@1 (cos) are presented in linear scale in Figure 28. All observed results align with the theoretical predictions.

## E. Additional Results Of Real-World Dataset Experiments

Figure 29 summarizes the experimental results and the purpose of the experiment. Expr. IV is in Figure 30, 31, 1. Expr. V is e in Figure 32, Table 2. Expr. VI is in Figure 33. Expr. VII is in Figure 34, 35, 36, 37, Table 3, and 4.

## E.1. Relation Between Expr. V And Vi

880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934

## F. Limitations And Future Work

While our study provides valuable insights into feature learning and transferability, several important directions remain for future research. First, while the Hermite approximation aided our feature analysis, it posed numerical challenges due to the discrepancy between polynomials and nonlinear neural networks. Specifically, the need for extremely high-dimensional approximations Figure 2 and the lack of precise scaling alignment between the approximation and the neural networks in

## E.2. Expr. Vii: Removing Duplicately Assigned **Eval Classes**

In **Expr. VII**, as suggested by the theoretical results on *Separability*, we validated whether eliminating duplicate in the assignments improves performance. To clarify, we will provide an example of duplicate *assignment* at Note E.1. Note E.1 (Example of duplicate *assignment*). For two train classes C
(*train*)
1, C
(*train*)
2and two test classes c
(*test*)
1, c
(test)
2, if most instances of c
(*test*)
1and c
(*test*)
2are classified as C
(*train*)
1, both test classes are assigned to C
(*train*)
1, resulting in duplication. Conversely, if c
(test)
1is classified as C
(*train*)
2and c
(*test*)
2as C
(*train*)
1, they are assigned without duplication.

To validate, we introudce treatment and control groups. For treatment group, we eliminate duplicate in the textitassignments for the train classes, i.e. , for each unseen class, the most frequently classified training class is aggregated, and the classes are randomly removed to ensure that the selected training classes become unique (2). For the control group, we performed random selection of the same number of classes of treatment group (1). These two groups are evaluated using *recall@1*. This process was repeated five times, and the average was reported. The experimental results are presented in 34, 35, 36, 37, Table 3, and 4. A total of 64 experiments are conducted, of which 51 demonstrated performance improvements: the estimated success rate is 79%. There is a 1.73%± 2.87% average improvement in recall@1, with a maximum improvement of 13.65%, a minimum decrease of -3.28%. These findings suggest that the duplicate reduction treatment group outperforms the randomly removed group with a binomial test p-value of 9.40 × 10−7.

On the other hand, certain results from Expr. V align with those from Expr. VI. As shown in Table 5, for datasets such as CAR and CUB, the number of additional classes introduced by the *sub In1k* dataset is significantly larger compared to SOP. For these data, inclusion of the additional *sub In1k* dataset contributes to improved *recall@1* performance when trained using a Random Initialized Network. Meanwhile, the performance of the pre-trained network is not significantly affected by the additional dataset. We attribute this to the fact that the pre-trained model is additionally re-trained using the same ImageNet dataset *sub In1k*. These findings suggest that further research on the behavior of pre-trained networks is necessary. Algorithm 1 Random Sampling

| Input: Number if unseen classes u, number of classes |L| Output: Sampled class set Srandom Set Srandom ← random.sample({0, 1, . . . , u − 1}, |L|) return Srandom   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Algorithm 2 Duplicated *assignment* reduction sampling Input: Model f, unseen data loader D, number of train classes Ctrain, number of unseen classes Cunseen Output: Sampled class set Snondup Initialize counter matrix counter ← 0 Cunseen×Ctrain for (img, label) in D do pred ← f(img) *Predicted class indices* Update counter: counter[label, pred] += 1 end for top1 index ← argsort(counter, dim = 1, descending = True)[*...,* 0]
unique label ← unique(top1 index)
Initialize Snondup ← ∅ for each label ℓ in unique label do Iℓ ← {i | top1 index[i] = ℓ} *Indices corresponding to label* ℓ isample ← random.sample(Iℓ, 1) *Select one random index* Snondup ← Snondup ∪ {isample}
end for return Snondup finite dimensions Figure 4.

935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 These limitations highlight the need for alternative approximation techniques or analytical approaches. Second, the relationship between semantic similarity and train-unseen similarity requires further theoretical exploration. Third, an important direction for future research is expanding the concepts of cohesion and Separability to multi-class softmax classification problems, incorporating normalization and temperature scaling to better align with practical settings or Neural Collapse research. Finally, recently Zavatone-Veth et al. (2023) suggest neural networks tend to compress the feature space around training data while expanding the regions between decision boundaries. We consider this phenomenon appears closely related to the train-unseen similarity-driven cohesion and Separability observed in our study. Investigating this connection through the lens of Riemannian geometry could yield novel insights into the fundamental structure of learned representations.

990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044
1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055

1056 1057 100 100 100 1058 2.6e+02 5.0e+02 5.0e+02 1059 1.3c+02 75 75 75 3.8e+02 3.8e+02 00000000 1060 s so 50 2.5e+02 2.5e+02
-1.3e+02 1061 25 25 1.3e+02 25 1.3e+02 1062 2-2.6e+02 B T H2 0.0e+00 ls t 000000 a a
-3.8e+02 10 1063 g g
+1.3e+02 1.1.3c+02 5.5.1e+02
-25
-25
−25 1064
-2.5e+02 2-2.5e+02
-6.4e+02 1065
-50
-50
–50
-3.8e+02
-7.6e+02 3-3.8e+02 1066
-75
-75 05.0c+02
−75
-5.0e+02 8.9e+02 1067
-----
-6,3e+02
-100
-100 100
− 50 0 50 100
-50 o 50 100
−50 o SO
100 1068 BTH1 8 T H 1 8 T H1 1069 1070 100 100 1071 1.6e+00 1.0e+00 400 1.6e+02 1072 75 75 1.6e+00 1.0e+00 1.6e+02 1073 50 50 1.6e+02 1.6e+00 1.0e+00 200 1074 25 25 1.6e+00 1.0e+00 1.6e+02 1075 15 β 4 μ 2 1.6e+02 1.6e+00 o o 0 1.0e+00 1076
. B
1.6e+02 1.5e+00 99 1077
-25
-25 20–200 1.6e+02 1.5e+00 9.9e-01 1078
-50
-50 1.6e+02 9.9e-01 1.5e+00 1079
−75
−75 1.6e+02 1.5e+00 9.8e-01
−400 1080
-100 1.1.6e+02 9.9.8e-01 1.5e+00
-100 1081 400
-200 o 200 400
-50
−50 o 50 100 100 o 50 100 PT HL
8 T H1 β T μ 1 1082 1083
1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099