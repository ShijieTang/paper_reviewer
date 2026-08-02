011

014 015 016

018

024

026

034

036

038

# How Classifiers Extract General Features for Downstream Tasks: An Asymptotic Analysis in Two-Layer Models

Anonymous Authors<sup>1</sup>

# Abstract

Neural networks learn effective feature representations through intermediate layers, enabling feature transfer without additional training for new tasks. However, the conditions for successful feature transfer remain underexplored. In this paper, we investigate feature transfer in classifier-trained networks, focusing on clustering in unseen distributions. In binary classification, we find that higher similarity between training and unseen distributions improves Cohesion and Separability, while Separability further requires unseen data to be assigned to different training classes. In multiclass classification, our analysis shows that the feature extractor maps input point based on their similarity to training classes, i.e. that unrelated training classes to input have negligible impact on feature extraction. We validate our theoretical findings in synthetic dataset and demonstrate practical applicability utilizing ResNet and variations of CAR, CUB, SOP, ISC, and ImageNet datasets.

# 1. Introduction

Neural networks have the remarkable ability to adapt to specific tasks, learning representations through penultimate layers. Training these intermediate layers is crucial for neural network generalization [\(Damian et al.,](#page-9-0) [2022\)](#page-9-0). Also, these layers can extract semantically meaningful and transferable features from new data, enabling feature transfer for new tasks [\(Yosinski et al.,](#page-11-0) [2014;](#page-11-0) [Kornblith et al.,](#page-9-1) [2019\)](#page-9-1). A wide range of techniques, from open set clustering [\(Roth et al.,](#page-10-0) [2020;](#page-10-0) [Huang et al.,](#page-9-2) [2024\)](#page-9-2) to vision-language models [\(Li](#page-10-1) [et al.,](#page-10-1) [2023\)](#page-10-1) and language models [\(Brown et al.,](#page-8-0) [2020;](#page-8-0) [Ko](#page-9-3)[jima et al.,](#page-9-3) [2023\)](#page-9-3), leverage feature transfer for downstream tasks. However, the specific conditions where features can be effectively transferred remain underexplored.

Among various applications, classification based visual open-set clustering [\(Musgrave et al.,](#page-10-2) [2020\)](#page-10-2) serves as a fundamental benchmark for evaluating whether a feature extractor can generalize to unseen data. Typically, this task involves classifier training on one set of classes and then testing it on disjoint classes to assess whether the extracted features form cohesive and separable class-wise clusters on unseen data [\(Wang et al.,](#page-11-1) [2018;](#page-11-1) [Seidenschwarz et al.,](#page-10-3) [2021;](#page-10-3) [Deng](#page-9-4) [et al.,](#page-9-4) [2022\)](#page-9-4). Given this context, we aim to investigate feature clustering with the following research questions:

Can we capture the presences of feature learning in classification and identify the conditions where features cluster effectively on new distributions?

To address this question, we analyze a two-layer nonlinear network network trained with a single large gradient descent step on a mean-squared classification loss in the *proportional regime* (in [section 2\)](#page-1-0). The proportional regime intuitively represents a scenario where the network width and the size of the dataset are of similar scales, aligning with common practices in model scaling [\(Ba et al.,](#page-8-1) [2022\)](#page-8-1), and they are known to effectively capture the phenomena occurring during the actual training process, as demonstrated in studies such as [Mei & Montanari](#page-10-4) [\(2020\)](#page-10-4); [Moniri et al.](#page-10-5) [\(2024\)](#page-10-5). We capture that the dominant part of the trained feature is composed of random initialization and *spikes* (Def. [3.4\)](#page-2-0) associated with the training classes [\(section 3\)](#page-2-1). Leveraging dominant features, we identify conditions for effective clustering on new distributions [\(section 4\)](#page-3-0).

In a binary classification setting, we assess the intra-class *cohesion* and inter-class *separability* of trained features in a numerical-analytical manner representing the clustering population risks (Def. [4.3\)](#page-3-1) [\(Clemen](#page-8-2) ´ c¸con, [2011;](#page-8-2) [Papa](#page-10-6) [et al.,](#page-10-6) [2015;](#page-10-6) [Li & Liu,](#page-10-7) [2021\)](#page-10-7) and goals for clustering performance [\(Liu et al.,](#page-10-8) [2017\)](#page-10-8). As a result, *Cohesion* increases as the *train-unseen similarity* (in Def. [4.1\)](#page-3-2) grows larger. Meanwhile, for *Separability*, if classes classes are *assigned* (Notes [4.2,](#page-3-3) [E.1\)](#page-16-0) to different training classes, *Separability* increases as the *train-unseen similarity* grows larger; otherwise, it decreases, as illustrated in [Figure 1.](#page-1-1)

<sup>1</sup>Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

Preliminary work. Under review by the International Conference on Machine Learning (ICML). Do not distribute.

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

108 109

![](_page_1_Diagram_1.jpeg)

Figure 1: Mapping data from the input space (left) to the learned feature space (right). Training classes are shown as balls, and unseen classes as dashed lines (a, b, p, n). *Cohesion*: Strong *cohesion* occurs for a, p, n, which have high similarity to the training classes compared to b. *Separability of* a, n: a and n, *assigned* to different training class, demonstrate high Separability. *Separability of* a, p: a and p, *assigned* to the same training class, exhibit low Separability.

In the multi-class classification setting, we analyze the *spikes* of features and find that *spikes* map new inputs based on a linear combination of randomly initialized classifier heads' weight with coefficients that represent the similarity of the training classes. Therefore, the more *spikes* aligned with the input data the greater their contribution to feature extraction, enhancing the expressiveness of the features.

In the experiments, we empirically observe *train-unseen similarity*, *cohesion*, *Separability*, and *recall@1* under our theoretical assumptions in synthetic datasets. As a result, we confirm that the theoretical interpretation aligns with the actual findings [\(subsection 5.2\)](#page-5-0). Additionaly, we explore practical metric learning settings and find evidence supporting the validity of our analysis results in a practical setup [\(subsection 5.4\)](#page-6-0). In most cases, we observe that clustering performance is higher when the unseen classes share the same sementic domain as the training classes. Moreover, adding semantically relevant training classes improves performance, whereas adding unrelated training classes does not lead to performance improvement.

Our contributions are summarized into following:

- We analyze the classifier feature, providing insights into how feature extractors operate:
  - Higher *train-unseen similarity* increases *cohesion*.
  - Higher *train-unseen similarity* increases *separability* between data *assigned* to different classes but reduces it otherwise.
  - Expressiveness of feature improves with an increased number of *spikes* non-orthogonal to input.
- We generalize the distribution assumption of prior works and present novel proof techniques for classifier analysis.
- The theoretical results are validated through diverse experiments, including synthetic and real-world datasets.

#### 1.1. Related Works

Metric Learning and Open Set Clustering Metric learning is proposed to cluster visually similar unseen classes using classification or triplet loss [\(Movshovitz-Attias et al.,](#page-10-9) [2017;](#page-10-9) [Zhai & Wu,](#page-11-2) [2019;](#page-11-2) [Boudiaf et al.,](#page-8-3) [2021\)](#page-8-3). Several recent approaches have focused on increasing the number of classes in the training data to improve clustering. One approach adds virtual classes [\(Chen et al.,](#page-8-4) [2018;](#page-8-4) [Qian et al.,](#page-10-10) [2020;](#page-10-10) [Gu et al.,](#page-9-5) [2021\)](#page-9-5). Another approach suggested leveraging a larger number of classes induced from [Schuhmann](#page-10-11) [et al.](#page-10-11) [\(2021\)](#page-10-11) to achieve state-of-the-art performance [\(An](#page-8-5) [et al.,](#page-8-5) [2023\)](#page-8-5). This aligns with our analysis, which suggests that performance improves as the number of relevant classes in clustering increases.

Neural Collapse (NC) and Unconstrained Layer-Peeled Model (ULPM) Recent studies have introduced the concept of Neural Collapse [\(Papyan et al.,](#page-10-12) [2020\)](#page-10-12) to explain the emergence of intra-class features and feature-weight alignment in trained neural networks. Several studies propose the ULPM to understand training dynamics of NC treating features and weights as unconstrained free variables [\(Fang et al.,](#page-9-6) [2021;](#page-9-6) [Zhu et al.,](#page-11-3) [2021;](#page-11-3) [Ji et al.,](#page-9-7) [2022;](#page-9-7) [Tirer](#page-11-4) [& Bruna,](#page-11-4) [2022\)](#page-11-4). However, ULPM, unlike the two layer network model we use, assumes the free variable features, which limits analyzability about input distribution and, consequently, prevents studying feature transferability.

Feature Learning in Two-Layer Networks Many works [\(Louart et al.,](#page-10-13) [2017;](#page-10-13) [Goldt et al.,](#page-9-8) [2020;](#page-9-8) [Hu & Lu,](#page-9-9) [2022\)](#page-9-9) study the Conjugate Kernel (CK), which enables the analysis of the structure of the first layer in two-layer networks. [Ba et al.](#page-8-1) [\(2022\)](#page-8-1); [Moniri et al.](#page-10-5) [\(2024\)](#page-10-5); [Ba et al.](#page-8-6) [\(2023\)](#page-8-6) argue that feature learning aids in reducing the population risk when evaluated on distributions same to the training data. Unlike these studies, we claim that the CK feature learning model not only explains this generalization but also enables the analysis of features from non-identical distributions, facilitating a deeper understanding of feature transfer.

Additional related works are provided in [Appendix A.](#page-12-0)

# 2. Problem Statement

Notations Let ∥·∥ be L <sup>2</sup> or the operator norm. Let ⊙ be the Hadamard product. Let A◦<sup>k</sup> be the Hadamard power. Let C, c > 0 and κ ∈ R be constants that may change from line to line. Define [d] ≜ {1, 2, · · · , d}. For o, O, Θ notations we follow [Moniri et al.](#page-10-5) [\(2024\)](#page-10-5)

Training Data We define data for one vs. one classification with #cls classes. The number of problem #<sup>P</sup> ≜ #cls(#cls−1) 2 . Let #cls be the number of training classes, and let C1, · · · , C#cls represent the class-conditional distri-

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

butions of the training data. Define the training dataset as D = (X, Y ), where X ∈ <sup>R</sup> <sup>n</sup>×<sup>d</sup>, Y ∈ [#cls] n, X = ({x ∼ C1} × m ∪ · · · ∪ {x ∼ C#cls } × m), where #clsm = n and m is the number of instances per class. Let D˜ = (X, ˜ Y˜ ) an i.i.d. copy of D.

Network Structure We consider two-layer networks. The initial weight of the first layer, W<sup>0</sup> ∈ <sup>R</sup> <sup>d</sup>×<sup>N</sup>, is initialized as W0[i] ∼ Unif(<sup>S</sup> d−1 ) for i ∈ [d]. We denote W obtained via a single step of gradient descent. The initial weights of the second layer, aij ∈ <sup>R</sup> <sup>N</sup> for i, j ∈ [#cls] s.t. i < j, are initialized as aij ∼ N(0, 1 <sup>N</sup> I). We define the initialized feature as F0(x) ≜ σ(W<sup>⊤</sup> <sup>0</sup> x) and the one-step trained feature as F(x) ≜ σ(W⊤x). The network output is defined as the following #<sup>P</sup> -dimensional vector: F(x) <sup>⊤</sup>aij |ij .

Proportional Regime We consider the two-layer neural networks in the proportional regime. n, d, and N are sample size, data and feature dimension, respectively. We perform our analysis under d/n, N/n → c as n, d, N → ∞.

Optimization Problem Denote the set of all network parameters as θ = {W, a12, · · · , a#<sup>P</sup> <sup>−</sup>1,#<sup>P</sup> }. Let Xij be a matrix in R 2m×d , where the first m rows contain samples x ∼ c<sup>i</sup> and the last m rows contain samples x ∼ c<sup>j</sup> . Let y ≜ [1, 1, . . . , 1, −1, . . . , −1]<sup>⊤</sup> ∈ <sup>R</sup> <sup>2</sup><sup>m</sup> be a vector consisting of m ones followed by m negative ones. To classify the given data, we use the Mean Squared Error,

$$L(x, y; \theta) = \frac{1}{2n} \sum_{i < j}^c \|y - \sigma(X_{ij}W)a_{ij}\|^2. \quad (1)$$

The weight update formula for the first layer is given by W = W<sup>0</sup> + G, where G ≜ − ∂L ∂w = P i<j Gij , s.t.

$$G_{ij} = -\frac{1}{n} \left[ X_{ij}^T [(\sigma(X_{ij} W) a_{ij} - y) a_{ij}^T \odot \sigma'(X_{ij} W)] \right]. \quad (2)$$

Now, we introduce the assumptions for theoretical analysis.

Assumption 2.1 (Activation Function). Let σ(x) be an element-wise activation s.t. σ, σ′ , σ′′ is bounded by λ<sup>σ</sup> almost surely. It admits a Hermite decomposition i.e. <sup>σ</sup>(z) = P<sup>∞</sup> <sup>k</sup>=0 ckHk(z), where c<sup>k</sup> = k! <sup>E</sup>[σ(z)Hk(z)] for standard gaussian z. We assume c<sup>0</sup> = 0, c<sup>1</sup> > 0 and c k k! ≤ Ck−3/2−<sup>w</sup>, for constants C, w > 0. For example, Shifted ReLU max(x, 0) − √ 2π satisfies this condition.

Assumption 2.2 (Training Data). Let the class-conditional training data distributions C<sup>i</sup> be non-centered Sub-Gaussians [\(Vershynin,](#page-11-5) [2018;](#page-11-5) [Cao et al.,](#page-8-7) [2021;](#page-8-7) [Cole & Lu,](#page-8-8) [2024\)](#page-8-8). This distribution family is suitable for classification, including distributions with limited support that are separable. It is an extension of the Gaussian assumption of [Ba](#page-8-1) [et al.](#page-8-1) [\(2022\)](#page-8-1).

### 3. Feature Decomposition

This section analyzes the learning dynamics during a single gradient descent step. First, we demonstrate that the gradient with respect to the W<sup>0</sup> exhibits an almost Rank- #<sup>P</sup> property within the proportional regime. Subsequently, we prove that the learned features can be predominantly expressed as Rank-#<sup>P</sup> components, establishing the dominant components for subsequent analyses.

Gradient Decomposition We decompose the gradient (equation [2\)](#page-2-2) using Hermite decomposition, which allows us to extract the essential rank-one matrix structure for each ij-th classification problem. Note that σ ′ = c<sup>1</sup> + σ ′ ⊥.

$$\begin{aligned} G_{ij} &= \frac{c_1}{n} X_{ij}^T y a_{ij}^T + \frac{1}{n} X_{ij}^T y a_{ij}^T \odot \sigma'_{\perp}(X_{ij} W_0) \\ &\quad - \frac{1}{n} X_{ij}^T \sigma(X_{ij} W_0) (a_{ij} a_{ij}^T) \odot \sigma'(X_{ij} W_0) \\ &\triangleq \mathbb{A}_{ij} + \mathbb{B}_{ij} + \mathbb{C}_{ij}. \end{aligned} \quad (3)$$

We derive the norm bound for the terms Aij , <sup>B</sup>ij , and Cij in Lemma [I.1.](#page-40-0) Using these bounds, we establish the following Theorem [3.1.](#page-2-3) For the proof, please refer to [Appendix I](#page-40-1)

Theorem 3.1 (Approximation of Gradient). *Under the assumptions in [section 2,](#page-1-0) and when* n *satisfies* <sup>1</sup> <sup>2</sup> > κlog<sup>2</sup> √ <sup>n</sup> n *, the following holds w.p.* 1 − C(ne <sup>−</sup><sup>c</sup> log<sup>2</sup> <sup>n</sup> + e <sup>−</sup>c<sup>n</sup>)*:*

$$\|G - \sum_{i < j} \mathbb{A}_{ij}\| \leq \kappa \frac{\log^2 \mathbf{n}}{\mathbf{n}}. \quad (4)$$

Feature Decomposition Now we utilize P i<j <sup>A</sup>ij to decompose the feature extractor. We decompose the one-step trained feature function F(x) = σ((W<sup>0</sup> + G) <sup>⊤</sup>x), which serves as a key step in deriving our main analysis. For the proof, please refer to [Appendix J.](#page-43-0)

Definition 3.2 (Data-Label Covariance). Data-Label Covariance for Xij is defined as βij = nX<sup>⊤</sup> ij y ∈ <sup>R</sup> d.

Theorem 3.3 (Decomposition of Trained Features). *Under the assumptions in [section 2,](#page-1-0) let* F<sup>0</sup> = σ(XW˜ <sup>0</sup>)*,* L ≜ log n*,* F L <sup>0</sup> = P<sup>L</sup> <sup>k</sup>=1 <sup>c</sup>kHk(XW˜ <sup>0</sup>)*, and spike*<sup>L</sup> = P<sup>L</sup> <sup>k</sup>=1 c k 1 ck(X˜ P i<j βija T ij ) ok*. With probability* 1 − o(1)*,*

$$F = F_0^L + spike_L + \Delta. \quad (5)$$

*Moreover,* <sup>∥</sup>*spike*L<sup>∥</sup> *is greater than* √ n*,* ∥F L <sup>0</sup> <sup>∥</sup> = Θ(√ n)*, and* ∥∆∥ = o( √ n)*.*

Based on these results, we analyze the feature representation using the approximation FL, which dominates the residual term ∥∆∥ = o( √ n) with probability 1 − o(1).

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

![](_page_3_Figure_1.jpeg)

Figure 2: Numerical Observation of Cohesion and Separability. Plot of *Cohesion* and Heatmap of *Separability* calculated by adjusting β <sup>⊤</sup>µ<sup>1</sup> and β <sup>⊤</sup>µ2.

Definition 3.4 (Dominant Feature F<sup>L</sup> = F L <sup>0</sup> + spikeL).

$$F_L(x) \triangleq \sum_{k=1}^L c_k [H_k(\tilde{X}W_0) + c_1^k (\sum_{i$$

Using the feature decomposition conducted so far, the next section analyzes clustering risk and explores the conditions for effective clustering of unseen data.

## 4. Feature Analysis

### 4.1. Clustering Risk Analysis in binary classification

In this section, we analyze clustering risks. We show train (β)-unseen (µ) similarity governs the the clustering population risk i.e. *Cohesion* and *Separability* of F<sup>L</sup> from Definition [3.4](#page-2-0) under condition [4.4.](#page-3-4) We derive *cohesion* and *separability* of F<sup>L</sup> for two "unseen" class-conditional distributions.

Definition 4.1 (*Train-Unseen Similarity*). Given Train Data-Label Covariance β in Definition [3.2](#page-2-4) and mean of Unseen distribution µ, *Train-Unseen Similarity* is defined as β <sup>⊤</sup>µ.

*Note* 4.2 (Explanation of *assignment* and β <sup>⊤</sup>µ)*.* βij represents the normal vector of the linear decision boundary, i.e. the direction determining class i vs. j based on the sign of its inner product with data. Therefore, the sign of β <sup>⊤</sup>µ indicates the class *assignment* of unseen data with µ.

Definition 4.3 (*Cohesion* and *Separability*). We define the clustering risks based on similarity between feature vectors using inner products.

*Cohesion* measures the expected similarity between i.i.d. features of the same class over network parameters θ and data x, x′ ∼ c1, i.e.

$$\mathbb{E}_\theta [\mathbb{E}_{x \sim e_1} F(x)^T \mathbb{E}_{x' \sim e_1} F(x')].$$

*Separability* measures the expected dissimilarity between independent features of different classes over θ, x ∼ c<sup>1</sup> and x ′ ∼ c<sup>2</sup> i.e.

$$-\mathbb{E}_\theta [\mathbb{E}_{x \sim c_1} F(x)^T \mathbb{E}_{x' \sim c_2} F(x')].$$

*Condition* 4.4*.* We fix n, d, N large enough. Under assumptions [2.1,](#page-2-5) [2.2,](#page-2-6) let c<sup>i</sup> = N(µ<sup>i</sup> , Id) for i ∈ [2] be the class conditional distributions. Define ρ (1) k,k′ > 0, ρ (2) k,k′ (cos(µ1, µ2)), ρ (3) k,k′ ,r > 0, ρ (4) k,k′ ,r,r′ > 0 as functions of N, d. Note that ρ (2) k,k′ increases as cos(µ1, µ2) grows. Exact definitions are in Def. [K.1.](#page-49-0) The Shifted ReLU, as stated in Assumption [2.1,](#page-2-5) is used as the activation.

Proposition 4.5 (Cohesion). *Following condition [4.4,](#page-3-4) the Cohesion of* F<sup>L</sup> *for* c<sup>i</sup> *,* i ∈ [2] *is given by:*

$$\sum_{k=1, k'=1}^L c_k c_{k'} \begin{bmatrix} \rho_{k, k'}^{(1)} \|\mu\|^{k+k'} \\ +2 \sum_{r'=0}^{k'} \rho_{k, k', r''}^{(3)} |\mu^T \beta|^{k'-r'} \|\beta\|^{r'} \|\mu\|^k \\ + \sum_{r, r'=(0,0)}^{k'} \rho_{k, k', r, r'}^{(4)} |\mu^T \beta|^{k+k'-r-r'} \|\beta\|^{r+r'}. \end{bmatrix} \quad (7)$$

Proposition 4.6 (Separability). *Following condition [4.4,](#page-3-4) the Separability of* F<sup>L</sup> *for* c1, c<sup>2</sup> *is given by:*

$$-\sum_{k=1, k'=1}^L c_k c_{k'} \left[ \rho_{k, k'}^{(2)}(\cos(\mu_1, \mu_2)) \|\mu_1\|^k \|\mu_2\|^{k'} + \sum_{r=0}^k \rho_{k, k', r}^{(3)} |\mu_1^T \beta|^{k-r} \|\beta\|^{r'} \|\mu_2\|^{k'} + \sum_{r'=0}^{k'} \rho_{k, k', r'}^{(3)} |\mu_2^T \beta|^{k'-r'} \|\beta\|^{r'} \|\mu_1\|^k + \sum_{r, r'=(0,0)}^{(k, k')} \rho_{k, k', r, r'}^{(4)} (\mu_1^T \beta)^{k-r} (\mu_2^T \beta)^{k'-r'} \|\beta\|^{r+r'} \right] \quad (8)$$

The proofs of Propositions [4.5](#page-3-5) and [4.6](#page-3-6) are provided in [Ap](#page-49-1)[pendix K.](#page-49-1) We numerically analyze the results of propositions [4.5](#page-3-5) and [4.6](#page-3-6) to investigate *Cohesion* and *Separability* further. For this numerical observations, we set ∥µ1∥ = ∥µ2∥ = ∥β∥ = 1, µ<sup>1</sup> = −µ<sup>2</sup> ∈ <sup>R</sup> <sup>320000</sup> and L = log<sup>10</sup> n. We calculate equation [7](#page-3-7) and equation [8](#page-3-8) by adjusting µ T <sup>1</sup> β and µ T <sup>2</sup> β, as shown in [Figure 2,](#page-3-9) which demonstrates the *Cohesion* and *Separability* of FL. *Cohesion* increases when the |µ <sup>T</sup> β| increases. *Separability* increases when µ T <sup>1</sup> β and µ T <sup>2</sup> β grow with opposite signs and decreases when they grow with the same sign. Moreover, we observe that this phenomenon is governed by the last term of equation [7,](#page-3-7) [8](#page-3-8) (related to ρ (4)) , as shown by separately computing this term and the others numerically in [Appendix B.](#page-12-1) Additionally, under the theoretical setup, we observe that our hypothesis tends to hold over a wider range as n increases (please refer to [Appendix B\)](#page-12-1).

The analytical results in equation [7](#page-3-7) and equation [7](#page-3-7) can be explained as follows. With ρ (4) > 0, the last term inside the bracket of *Cohesion* in equation [7](#page-3-7) increases in value as *Train-Unseen Similarity* grows. The last term of *Separability* is influenced by (µ T <sup>1</sup> β) k−r (µ T <sup>2</sup> β) k ′−r . Provided that k − r and k ′ − r ′ are odd, this term implies that if the *Train-Unseen Similarities* have opposite signs and increase, then this term improves; otherwise, if the signs are the same and increase, *Separability* decreases. According to the analysis

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

274

![](_page_4_Figure_1.jpeg)

Figure 3: As shown in equation [6,](#page-3-10) after one step of training with spike β1, β2, β3, β4, the inner product between input x<sup>i</sup> and β<sup>i</sup> acts as the coefficient in the linear combination of ai , forming the *spikes* structure of the feature.

in [Appendix H,](#page-39-0) the first coefficient c<sup>1</sup> of Shifted ReLU is a large positive value, and subsequent Hermite coefficients approach zero while oscillating around it. Thus, we hypothesize that the positive part is likely to dominate Pckck′ , but further work is needed to confirm this.

### 4.2. Spike Component Analysis

![](_page_4_Figure_8.jpeg)

Figure 4: When trained along the directions β<sup>2</sup> and β3, we observe significant changes in the feature space distance as x<sup>1</sup> and x<sup>2</sup> vary, compared to β1, β4.

In this section, based on the previous feature decomposition and extend it to examine the impact of a multi-class classifier's spike structure on unseen data clustering. We examine the spike structure in F<sup>L</sup> = F L <sup>0</sup> + spike<sup>L</sup> and its influence on feature mapping. This examination allows us to explore the impact of the training data's structure β on the feature generation of unseen data. The spike structure inside the Hadamard power involves the linear combination coefficient β ⊤ ijx and the random initialized classifier head aij (equation [3.4\)](#page-2-0). Thus, the feature extraction is closely linked to the inner product between βij and the input point x. If the direction of x is not orthogonal to βij , then spike of βij involve feature extraction.

![](_page_4_Figure_2.jpeg)

Figure 5: Comparison of log average slope between Theory and Two-layer Networks. Midpoint (β1) Interpolation (β2) Extrapolation (β3) Orthogonal (β4). The intersection implies learning intersecting β.

Conversely, when x is orthogonal to βij , the impact of spike βij is eliminated. To validate this, we define following four spikes, given test input x1, x<sup>2</sup> ∈ <sup>S</sup> d−1 ( √ d), β<sup>1</sup> = x1+x<sup>2</sup> 2 , β<sup>2</sup> = x1+3x<sup>2</sup> 4 , β<sup>3</sup> = −x1+5x<sup>2</sup> 4 and β4, a random vector orthogonal to x1, x2. Then, the magnitudes are adjusted to √ d. By definition, β1, β<sup>4</sup> cannot contribute to feature extraction because they are Midpoint or Orthogonal, while β<sup>2</sup> and β<sup>3</sup> can distinguish the two inputs. For illustration see [Figure 3.](#page-4-0)

Now, we demonstrate this explanation using the approximated features F<sup>L</sup> and the two layer neural network F with the four disjoint sub-classification problem [<sup>1</sup>](#page-4-1) defined as follows: We generated four classification problems by creating Gaussian training data with means β<sup>i</sup> and −β<sup>i</sup> , and a covariance of 0.1I for n, d, N = 2<sup>11</sup>, enabling the networks to learn β<sup>i</sup> as their *spike*. F is trained by this data and F<sup>L</sup> is calculated by its definition. We observed the feature distance between F(x1), F(x2) and between FL(x1), FL(x2) for 4 k combinations of β<sup>i</sup> in this problem by varying the angle between x1, x2. Please refer to [Figure 4](#page-4-2) and [21](#page-22-0) for results. It can be observed that the feature from β<sup>1</sup> and β<sup>4</sup> hardly captures variations in the angle of test input x1, x<sup>2</sup> within the data space. In contrast, the feature from β<sup>2</sup> and β<sup>3</sup> is highly sensitive to such variations, suggesting that it effectively preserves the structural changes in the input data. Both FL(x1) and F(x1) exhibit the same trends, which supports the validity of our feature approximation. To aggregate these combinatorial results, we measure the log of the average slope, which indicates that features with sensitive changes tend to have larger values, as shown in [Figure 5.](#page-4-3)

As a result in [Figure 5,](#page-4-3) we observe that when multiple βs are used in training, features are more sensitive to changes

Instead of studying all combinations for 8 classes classification, we simplify the task by grouping four pairs, performing only four combinations of classifications.

278

289 290

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

![](_page_5_Diagram_1.jpeg)

Train Data 1, 2, 3 Eval 1, 2

Figure 6: Examples of training datasets (Data 1, 2, 3) and evaluation data Eval 1, 2.

in distance within the data space. Meanwhile, the Midpoint β<sup>1</sup> and Orthogonal spike β<sup>4</sup> seem ineffective for feature extraction, even when learned alongside other spikes. Experiments show that learning representations with unrelated classes limits expressiveness, while related classes enhance the model's ability to capture fine-grained features of unseen data. This trend is consistently observed in real-world datasets in Expr V, VI at [subsection 5.4.](#page-6-0) Additionally, to clarify the effect of the spikes, we compute F L 0 and spike<sup>L</sup> separately as shown in [Figure 22.](#page-23-0) The results show that the spike<sup>L</sup> created by β<sup>1</sup> and β<sup>4</sup> embeds x<sup>1</sup> and x<sup>2</sup> as the same feature. Therefore, it confirms that the distinction between x<sup>1</sup> and x<sup>2</sup> created by the model trained with β<sup>1</sup> and β<sup>4</sup> is due to the random feature F L 0 .

# 5. Experiments

*Remark* 5.1*. recall@1* ≜ <sup>E</sup>xi,yi1yi=ˆyi,1-NN . yˆi,1-NN is class of the closest feature to x<sup>i</sup> . This is a feasible measure for evaluating whether new classes form clusters.

In this section, we conduct seven experimental setups to validate our theoretical results. First, in Experiments I, II and III, we utilize a synthetic dataset to confirm that, as discussed in [subsection 4.1,](#page-3-11) *Cohesion*, *Separability* are determined by the *Train-unseen similarity*. Second, to demonstrate how our theoretical explanations can provide intuition in practical settings, we conduct Experiments IV, V, VI, and VII. For this purpose, we analyze the open-set clustering problem using fine-grained real image datasets.

## 5.1. Setup for Theory Vaildation: Expr. I, II, III

We use three types of different non-centered Sub-Gaussian distributions as training datasets that are symmetric about the origin. For the evaluation, we introduce two distribution i.e. Eval 1, Eval 2 with translation parameter e and rotation parameter R ∈ R ⊆ SO(n) to control the *train-unseen similarity* β <sup>⊤</sup>µ. e.g. as e increases from 0 towards 1, β <sup>⊤</sup>µ increases, and as R approaches the identity matrix I, β <sup>⊤</sup>µ increases. For illustration of the data, see [Figure 6.](#page-5-1) For detail, refer to [subsection D.1.](#page-12-2) We follow the condition described in [section 2](#page-1-0) and [subsection 4.1.](#page-3-11)

Now we explain Expr. I, II, III. For each experiment, we utilize all datasets 1, 2, 3, with distinct Eval data usage. Expr. I uses two Eval 1 data with translation parameter e<sup>1</sup> ∈ [−0.9, 0.9] and e<sup>2</sup> = −e1, so they are *assigned* to opposite training classes (say pos-neg). Experiments II and III are based on two Eval 2 data distributions, each parameterized by a small-angle random rotation matrix R ∈ R. In Experiment II, considering the case where the datasets are *assigned* to opposite classes, the first distribution uses R and the second distribution is origin symmetry of the first distribution. In Experiment III, considering the situation where the datasets are *assigned* to the same class (say pospos), the first distribution uses R and the second uses R<sup>⊤</sup> to slightly rotate given means.

## 5.2. Results of Theory Vaildation: Expr. I, II, III

In this experiment, we examine the relationships between the *train-unseen similarity*( i.e. β <sup>⊤</sup>µ), *Cohesion*, *Separability* that we discussed in [subsection 4.1](#page-3-11) and *Recall@1* to evaluate performance using practical measures. All test data are generated symmetrically, so for simplicity in visualization, we report the measurement for a single class. For Expr I, we present a summary of the results in [Figure 8.](#page-6-1) We observe that for large values of |β <sup>⊤</sup>µ|, strong *Cohesion* and *Separability* occur across all datasets. For Expr II and III, in accordance with the *Separability* structure observed in [subsection 4.1,](#page-3-11) when the signs of β <sup>⊤</sup>µ1, β⊤µ<sup>2</sup> are opposite (Expr II), we observed an increase in *Separability*, whereas in the other case (Expr III), we observed a decrease [Figure 7.](#page-6-2) For *recall@1*, we observed a similar trend as *Separability*. These results correspond to our theoretical findings. For individual graphs, refer to [Appendix D.](#page-12-3)

## 5.3. Setup for Practical Vaildation: Expr. IV, V, VI, VII

We designed experiments to examine whether these insights are also applicable to clustering performance in image datasets and practical neural networks. In these scenarios, we utilize *train-unseen similarity* to conceptualize semantic similarity between training and unseen classes (Expr. IV). The number of non-orthogonal *spikes* is interpretable as the number of semantically similar or dissimilar training classes

334

336

338

351

354

356

358

360 361

364

366

368

371

374

378

![](_page_6_Figure_1.jpeg)

Figure 7: Data 1 evaluated in the Eval 2 setup. Upper row: In Expr II, all metrics increase as |β <sup>⊤</sup>µ| increases. Lower row: In Expr III, where two test classes are *assigned* to a single train class, recall@1 and Separability tend to decrease as |β <sup>⊤</sup>µ| increases. This aligns with our predictions. The red line represents the values after one step training. Tje blue line represents the values from initialization.

Figure 8: Summary of Expr. I. D<sup>i</sup> denotes Data i and C, S denote *Cohesion* and *Separability*. Dark and large points indicate low |β <sup>⊤</sup>µ| values, while the opposite indicates high values. All measurements increase with respect to |β <sup>⊤</sup>µ|. We scaled using the absolute value at the 85th percentile.

(Expr. V, VI). Additionally, we validate whether removing the duplicatively *assigned* unseen classes improve clustering risk compared to random removal, as suggested by the results of *Separability* (Expr. VII).

For this investigation, we used the benchmark datasets CAR(Vehicle) [\(Krause et al.,](#page-9-10) [2013\)](#page-9-10), CUB(Bird) [\(Wah et al.,](#page-11-6) [2011\)](#page-11-6), SOP(Product) [\(Song et al.,](#page-11-7) [2015\)](#page-11-7), and ISC (Clothing) [\(Liu et al.,](#page-10-14) [2016\)](#page-10-14), referred to as *Domain*. Additionally, we utilized ImageNet subsets corresponding to the domains Vehicle, Bird, Product, and Clothing, denoted as I(V), I(B), I(P), and I(C), referred to as *sub In1k* for extra classes. Also,

we performed experiments on the whole classes ImageNet by sampling 100 instances per class (say *subsampled whole In1k*). Details are in [Appendix N.](#page-57-0) The objective function and most experimental configurations followed the approach outlined in [Zhai & Wu](#page-11-2) [\(2019\)](#page-11-2), which is a seminal baseline. We use ResNet18 and ResNet50 [\(He et al.,](#page-9-11) [2015\)](#page-9-11). In addition to the randomly initialized networks in the main text, we conducted experiments with pre-trained networks common in feature learning, and results are included in [Appendix E.](#page-16-1) The two setups exhibited similar trends.

![](_page_6_Figure_4.jpeg)

## 5.4. Results of Practical Vaildation: Expr. IV, V, VI, VII

![](_page_6_Figure_7.jpeg)

Figure 9: Expr. IV, *recall@1* measurements. Most cases show the highest performance when the domain of the Train and Test corresponds.

394

396

![](_page_7_Figure_9.jpeg)

For Expr. IV, we trained with each *Domain* dataset (CAR, CUB, SOP, and ISC train datasets) and *Domain*+*sub In1k* dataset (CAR+I(V), CUB+I(B), SOP+I(P), and ISC+I(C)), and then measured how each model well cluster on all of the test datasets (CAR, CUB, SOP, ISC test datasets). As shown in [Figure 9,](#page-6-3) we verify whether clustering the test dataset related to the train classes is more effective than clustering unrelated data, analogous to result in [subsection 4.1.](#page-3-11)

Figure 10: Expr V in ResNet50(init). The pink , red , and blue bars represent *Domain*, *Domain*+*sub In1k*, *Domain*+*subsampled whole In1k*, respectively.

In Expr. V, we measured the clustering performance for corresponding test datasets after learning the *Domain*, *Domain*+*sub In1k*, and *Domain*+*subsampled whole In1k*. We find that adding classes from the entire ImageNet dataset during training, rather than including only related classes, does not significantly improve clustering [\(Figure 10,](#page-7-0) [32\)](#page-30-0).

Figure 11: Expr VI, Recall@1 values for the CAR, CUB, SOP, and ISC datasets are shown with dashed lines for ResNet18 and solid lines for ResNet50.

In Expr. VI, experiments are conducted by dividing the *Domain* datasets into four steps to observe the impact of increasing the number of related classes on *recall@1* performance [\(Figure 11\)](#page-7-1). From Step 0 to Step 3, 25%, 50%, 75%, and 100% of the *Domain* dataset classes are sequentially added for training. The added classes are randomly selected, and each experiment is repeated three times. For the number of classes, refer to [Table 6.](#page-57-1) Furthermore, we observed that some results of Expr. V align with those of Expr. VI, as discussed in detail in [subsection E.1.](#page-16-2)

For Expr. VII, in evaluation, removing duplicatively *assigned* of unseen classes resulted in a 1.73 ± 2.87% improvement in recall@1 compared to random removal of same amount of unseen classes, with max improve: 13.65%, min decrease: -3.28%, a success rate: 79% and p = 9.40×10−<sup>7</sup> . This suggest that duplicate *assignments* hinder clustering, which aligns with our theory. Details are in [subsection E.2.](#page-16-3)

# 6. Conclusion

![](_page_7_Figure_4.jpeg)

In this study, we explored the feature learning dynamics of a two-layer classifier in the proportional regime to uncover the mechanisms underlying feature transferability. Specifically, we analyzed the conditions where the learned features of unseen classes form cohesive and separable cluster. Our theoretical analysis extends the Conjugate Kernel framework to classification tasks. As a result, our numerical-analytical theory demonstrated that feature *cohesion* increases with greater similarity between training and unseen data, while feature *separability* is influenced not only by similarity but also by avoiding duplicate class *assignments* in binary classification. Additionally, we showed that only when the *spikes* are non-orthogonal to the input, do they get involved in feature extraction. In addition to validation on synthetic datasets, we observed that our theory offers valuable insights even when applied to real-world datasets.

Our empirical findings suggest that clustering performance improves when the test data share the same semantic domain as the training data. Furthermore, adding semantically relevant classes to the training set leads to performance gains, whereas introducing unrelated classes has little effect. Contrary to existing research that focuses on performance improvement through large-scale learning on broad domains [\(Brown et al.,](#page-8-0) [2020;](#page-8-0) [An et al.,](#page-8-5) [2023\)](#page-8-5), our study provides evidence that only certain relevant knowledge, closely related to the domain, influences feature transfer. This evidence mirrors classical problems in the field of artificial intelligence, such as the frame problem and the installation problem. Specifically, AI agents do not require all available knowledge to solve a given problem; only specific, detailed knowledge is necessary. [Dennett](#page-9-12) [\(1984\)](#page-9-12) states about this as follows: "People in AI ... take the shortcut of installing all that an agent has to know to solve a problem. This may, of course, be a dangerous shortcut." We hope that our study may remind the AI community of the longstanding principle that it may not be the scale of the data that matters. We have also discussed the limitations and future research directions related to the Hermite expansion approximation and general results for *cohesion* and *separability* in [Appendix F.](#page-16-4)

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 This paper presents work aimed at advancing the field of Machine Learning. In this research, we analyze the potential for clustering performance improvement through the classification training of a large number of highly granular classes. Such an approach may lead to a reduction in the level of personal data masking required for fine-grained data differentiation, which could trigger new ethical discussions regarding privacy protection. Additionally, to effectively implement this approach, there may be a tendency to collect more data, which can have significant implications for the scale and scope of data collection, as well as data management practices. References An, X., Deng, J., Yang, K., Li, J., Feng, Z., Guo, J., Yang, J., and Liu, T. Unicom: Universal and compact representation learning for image retrieval, 2023. URL <https://arxiv.org/abs/2304.05884>. Ba, J., Erdogdu, M. A., Suzuki, T., Wang, Z., Wu, D., and Yang, G. High-dimensional asymptotics of feature learning: How one gradient step improves the representation, 2022. URL [https://arxiv.org/abs/2205.](https://arxiv.org/abs/2205.01445) [01445](https://arxiv.org/abs/2205.01445). Ba, J., Erdogdu, M. A., Suzuki, T., Wang, Z., and Wu, D. Learning in the presence of low-dimensional structure: A spiked random matrix perspective. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), *Advances in Neural Information Processing Systems*, volume 36, pp. 17420–17449. Curran Associates, Inc., 2023. Bai, Z. and Silverstein, J. W. *Spectral Analysis of Large Dimensional Random Matrices*. Springer New York, 2010. ISBN 9781441906618. doi: 10.1007/ 978-1-4419-0661-8. URL [http://dx.doi.org/](http://dx.doi.org/10.1007/978-1-4419-0661-8) [10.1007/978-1-4419-0661-8](http://dx.doi.org/10.1007/978-1-4419-0661-8). Bellet, A. and Habrard, A. Robustness and generalization for metric learning. *Neurocomputing*, 151:259–267, March 2015. ISSN 0925-2312. doi: 10.1016/j.neucom.2014. 09.044. URL [http://dx.doi.org/10.1016/j.](http://dx.doi.org/10.1016/j.neucom.2014.09.044) [neucom.2014.09.044](http://dx.doi.org/10.1016/j.neucom.2014.09.044). Bienstman, P. Mathematics for photonics. Course Syllabus, September 2023. URL [https:](https://studiekiezer.ugent.be/studiefiche/en/E002640/current) [//studiekiezer.ugent.be/studiefiche/](https://studiekiezer.ugent.be/studiefiche/en/E002640/current) [en/E002640/current](https://studiekiezer.ugent.be/studiefiche/en/E002640/current). Course size: 4.0 credits, Study time: 120 hours. Offered in English and Dutch. Boudiaf, M., Rony, J., Ziko, I. M., Granger, E., Pedersoli, M., Piantanida, P., and Ayed, I. B. A unifying mutual information view of metric learning: cross-entropy vs. pairwise losses, 2021. URL [https://arxiv.org/](https://arxiv.org/abs/2003.08983) [abs/2003.08983](https://arxiv.org/abs/2003.08983). Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners, 2020. URL [https://](https://arxiv.org/abs/2005.14165) [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165). Cao, Y., Gu, Q., and Belkin, M. Risk bounds for over-parameterized maximum margin classification on sub-gaussian mixtures. In Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), *Advances in Neural Information Processing Systems*, volume 34, pp. 8407–8418. Curran Associates, Inc., 2021. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2021/file/46e0eae7d5217c79c3ef6b4c212b8c6f-Paper.pdf) [cc/paper\\_files/paper/2021/file/](https://proceedings.neurips.cc/paper_files/paper/2021/file/46e0eae7d5217c79c3ef6b4c212b8c6f-Paper.pdf) [46e0eae7d5217c79c3ef6b4c212b8c6f-Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/46e0eae7d5217c79c3ef6b4c212b8c6f-Paper.pdf). [pdf](https://proceedings.neurips.cc/paper_files/paper/2021/file/46e0eae7d5217c79c3ef6b4c212b8c6f-Paper.pdf). Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, J., ´ Bojanowski, P., and Joulin, A. Emerging properties in self-supervised vision transformers, 2021. URL [https:](https://arxiv.org/abs/2104.14294) [//arxiv.org/abs/2104.14294](https://arxiv.org/abs/2104.14294). Chang, Y., Hu, C., and Turk, M. Manifold of facial expression. In *2003 IEEE International SOI Conference. Proceedings (Cat. No.03CH37443)*, pp. 28–35, 2003. doi: 10.1109/AMFG.2003.1240820. Chen, B., Deng, W., and Shen, H. Virtual class enhanced discriminative embedding learning, 2018. URL [https:](https://arxiv.org/abs/1811.12611) [//arxiv.org/abs/1811.12611](https://arxiv.org/abs/1811.12611). Chopra, S., Hadsell, R., and LeCun, Y. Learning a similarity metric discriminatively, with application to face verification. In *2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05)*, volume 1, pp. 539–546 vol. 1, 2005. doi: 10.1109/CVPR.2005.202. Clemen ´ c¸con, S. On u-processes and clustering performance. In Shawe-Taylor, J., Zemel, R., Bartlett, P., Pereira, F., and Weinberger, K. (eds.), *Advances in Neural Information Processing Systems*, volume 24. Curran Associates, Inc., 2011. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2011/file/a1d0c6e83f027327d8461063f4ac58a6-Paper.pdf) [cc/paper\\_files/paper/2011/file/](https://proceedings.neurips.cc/paper_files/paper/2011/file/a1d0c6e83f027327d8461063f4ac58a6-Paper.pdf) [a1d0c6e83f027327d8461063f4ac58a6-Paper](https://proceedings.neurips.cc/paper_files/paper/2011/file/a1d0c6e83f027327d8461063f4ac58a6-Paper.pdf). [pdf](https://proceedings.neurips.cc/paper_files/paper/2011/file/a1d0c6e83f027327d8461063f4ac58a6-Paper.pdf). Cole, F. and Lu, Y. Score-based generative models break the curse of dimensionality in learning a fam-

494

## Impact Statement

- 495 496 497 498 499 500 504 506 508 509 511 514 515 516 518 524 526 528 531 534 536 538 540 541 542 543 544 545 546 547 548 549 ily of sub-gaussian distributions. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=wG12xUSqrI) [id=wG12xUSqrI](https://openreview.net/forum?id=wG12xUSqrI). Damian, A., Lee, J. D., and Soltanolkotabi, M. Neural networks can learn representations with gradient descent, 2022. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2206.15144) [2206.15144](https://arxiv.org/abs/2206.15144). Deng, J., Guo, J., Yang, J., Xue, N., Kotsia, I., and Zafeiriou,
  - S. Arcface: Additive angular margin loss for deep face recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(10):5962–5979, October 2022. ISSN 1939-3539. doi: 10.1109/tpami.2021. 3087709. URL [http://dx.doi.org/10.1109/](http://dx.doi.org/10.1109/TPAMI.2021.3087709) [TPAMI.2021.3087709](http://dx.doi.org/10.1109/TPAMI.2021.3087709). Dennett, D. Cognitive wheels: The frame problem of ai. 01 1984. El-Nouby, A., Neverova, N., Laptev, I., and Jegou, H. Train- ´ ing vision transformers for image retrieval, 2021. URL <https://arxiv.org/abs/2102.05644>. Ermolov, A., Mirvakhabova, L., Khrulkov, V., Sebe, N., and Oseledets, I. Hyperbolic vision transformers: Combining improvements in metric learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 7409–7419, 2022. Fan, Z. and Wang, Z. Spectra of the conjugate kernel and neural tangent kernel for linear-width neural networks, 2020. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2005.11879) [2005.11879](https://arxiv.org/abs/2005.11879). Fang, C., He, H., Long, Q., and Su, W. J. Exploring deep neural networks via layer-peeled model: Minority collapse in imbalanced training. *Proceedings of the National Academy of Sciences*, 118(43), October 2021. ISSN 1091-6490. doi: 10.1073/pnas.2103091118. URL [http:](http://dx.doi.org/10.1073/pnas.2103091118) [//dx.doi.org/10.1073/pnas.2103091118](http://dx.doi.org/10.1073/pnas.2103091118). Galanti, T., Gyorgy, A., and Hutter, M. On the role of ¨ neural collapse in transfer learning, 2022. URL [https:](https://arxiv.org/abs/2112.15121) [//arxiv.org/abs/2112.15121](https://arxiv.org/abs/2112.15121). Goldt, S., Loureiro, B., Reeves, G., Krzakala, F., Mezard, ´ M., and Zdeborova, L. The gaussian equivalence of gen- ´ erative models for learning with shallow neural networks. *Proceedings of the 2nd Mathematical and Scientific Machine Learning Conference, PMLR 145:426-471 (2021)*, 06 2020. URL [https://arxiv.org/pdf/2006.](https://arxiv.org/pdf/2006.14709.pdf) [14709.pdf](https://arxiv.org/pdf/2006.14709.pdf). Goodfellow, I., Bengio, Y., Courville, A., and Bengio, Y. *Deep learning*, volume 1. MIT Press, 2016. Gu, G., Ko, B., and Kim, H.-G. Proxy synthesis: Learning with synthetic classes for deep metric learning, 2021. URL <https://arxiv.org/abs/2103.15454>. Han, X. Y., Papyan, V., and Donoho, D. L. Neural collapse under mse loss: Proximity to and dynamics on the central path, 2022. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2106.02073) [2106.02073](https://arxiv.org/abs/2106.02073). He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition, 2015. URL [https://](https://arxiv.org/abs/1512.03385) [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385). Hu, H. and Lu, Y. M. Universality laws for high-dimensional learning with random features, 2022. URL [https://](https://arxiv.org/abs/2009.07669) [arxiv.org/abs/2009.07669](https://arxiv.org/abs/2009.07669). Huai, M., Xue, H., Miao, C., Yao, L., Su, L., Chen, C., and Zhang, A. Deep metric learning: The generalization analysis and an adaptive algorithm. In *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI-19*, pp. 2535–2541. International Joint Conferences on Artificial Intelligence Organization, 7 2019. doi: 10.24963/ijcai.2019/352. URL [https:](https://doi.org/10.24963/ijcai.2019/352) [//doi.org/10.24963/ijcai.2019/352](https://doi.org/10.24963/ijcai.2019/352). Huang, H., Nie, Z., Wang, Z., and Shang, Z. Cross-modal and uni-modal soft-label alignment for image-text retrieval, 03 2024. URL [https://arxiv.org/pdf/](https://arxiv.org/pdf/2403.05261.pdf) [2403.05261.pdf](https://arxiv.org/pdf/2403.05261.pdf). Hui, L., Belkin, M., and Nakkiran, P. Limitations of neural collapse for understanding generalization in deep learning, 2022. URL [https://arxiv.org/abs/2202.](https://arxiv.org/abs/2202.08384) [08384](https://arxiv.org/abs/2202.08384). Isserlis, L. On a formula for the product-moment coefficient of any order of a normal frequency distribution in any number of variables. *Biometrika*, 12(1/2): 134–139, 1918. ISSN 00063444, 14643510. URL <http://www.jstor.org/stable/2331932>. Ji, W., Lu, Y., Zhang, Y., Deng, Z., and Su, W. J. An unconstrained layer-peeled perspective on neural collapse, 2022. URL <https://arxiv.org/abs/2110.02796>. Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa,
    - Y. Large language models are zero-shot reasoners, 2023. URL <https://arxiv.org/abs/2205.11916>. Kornblith, S., Shlens, J., and Le, Q. V. Do better imagenet models transfer better?, 2019. URL [https://arxiv.](https://arxiv.org/abs/1805.08974) [org/abs/1805.08974](https://arxiv.org/abs/1805.08974). Krause, J., Stark, M., Deng, J., and Fei-Fei, L. 3d object representations for fine-grained categorization. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV) Workshops*, June 2013.

- 551 554 556 558 560 564 566 568 571 574 576 578 580 581 582 583 584 585 586 587 588 589 590 594 596 598 600 601 602 603 604 Kuchibhotla, A. K. and Chakrabortty, A. Moving beyond sub-gaussianity in high-dimensional statistics: applications in covariance estimation and linear regression. *Information and Inference: A Journal of the IMA*, 11(4):1389–1456, June 2022. ISSN 2049-8772. doi: 10.1093/imaiai/iaac012. URL [http://dx.doi.org/](http://dx.doi.org/10.1093/imaiai/iaac012) [10.1093/imaiai/iaac012](http://dx.doi.org/10.1093/imaiai/iaac012). Lee, K.-C., Ho, J., Yang, M.-H., and Kriegman, D. Videobased face recognition using probabilistic appearance manifolds. In *2003 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2003. Proceedings.*, volume 1, pp. I–I, 2003. doi: 10.1109/CVPR. 2003.1211369. Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023. URL <https://arxiv.org/abs/2301.12597>. Li, S. and Liu, Y. Sharper generalization bounds for clustering. In Meila, M. and Zhang, T. (eds.), *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pp. 6392–6402. PMLR, 18–24 Jul 2021. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v139/li21k.html) [v139/li21k.html](https://proceedings.mlr.press/v139/li21k.html). Liaw, C., Mehrabian, A., Plan, Y., and Vershynin, R. A simple tool for bounding the deviation of random matrices on geometric sets, 2016. URL [https://arxiv.org/](https://arxiv.org/abs/1603.00897) [abs/1603.00897](https://arxiv.org/abs/1603.00897). Liu, W., Wen, Y., Yu, Z., and Yang, M. Large-margin softmax loss for convolutional neural networks, 2017. URL <https://arxiv.org/abs/1612.02295>. Liu, W., Wen, Y., Yu, Z., Li, M., Raj, B., and Song, L. Sphereface: Deep hypersphere embedding for face recognition, 2018. URL [https://arxiv.org/abs/](https://arxiv.org/abs/1704.08063) [1704.08063](https://arxiv.org/abs/1704.08063). Liu, Z., Luo, P., Qiu, S., Wang, X., and Tang, X. Deepfashion: Powering robust clothes recognition and retrieval with rich annotations. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2016. Louart, C., Liao, Z., and Couillet, R. A random matrix approach to neural networks, 2017. URL [https://](https://arxiv.org/abs/1702.05419) [arxiv.org/abs/1702.05419](https://arxiv.org/abs/1702.05419). Mei, S. and Montanari, A. The generalization error of random features regression: Precise asymptotics and double descent curve, 2020. URL [https://arxiv.org/](https://arxiv.org/abs/1908.05355) [abs/1908.05355](https://arxiv.org/abs/1908.05355). Moniri, B., Lee, D., Hassani, H., and Dobriban, E. A theory of non-linear feature learning with one gradient step in two-layer neural networks, 2024. URL [https:](https://openreview.net/forum?id=MY8SBpUece) [//openreview.net/forum?id=MY8SBpUece](https://openreview.net/forum?id=MY8SBpUece). Movshovitz-Attias, Y., Toshev, A., Leung, T. K., Ioffe, S., and Singh, S. No fuss distance metric learning using proxies, 2017. URL [https://arxiv.org/abs/1703.](https://arxiv.org/abs/1703.07464) [07464](https://arxiv.org/abs/1703.07464). Musgrave, K., Belongie, S., and Lim, S.-N. A metric learning reality check, 2020. URL [https://arxiv.org/](https://arxiv.org/abs/2003.08505) [abs/2003.08505](https://arxiv.org/abs/2003.08505). O'Donnell, R. Analysis of boolean functions, 2021. URL <https://arxiv.org/abs/2105.10386>. Papa, G., Clemen ´ c¸on, S., and Bellet, A. Sgd algorithms based on incomplete u-statistics: Large-scale minimization of empirical risk. In Cortes, C., Lawrence, N., Lee, D., Sugiyama, M., and Garnett,
  - R. (eds.), *Advances in Neural Information Processing Systems*, volume 28. Curran Associates, Inc., 2015. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2015/file/67e103b0761e60683e83c559be18d40c-Paper.pdf) [cc/paper\\_files/paper/2015/file/](https://proceedings.neurips.cc/paper_files/paper/2015/file/67e103b0761e60683e83c559be18d40c-Paper.pdf) [67e103b0761e60683e83c559be18d40c-Paper](https://proceedings.neurips.cc/paper_files/paper/2015/file/67e103b0761e60683e83c559be18d40c-Paper.pdf). [pdf](https://proceedings.neurips.cc/paper_files/paper/2015/file/67e103b0761e60683e83c559be18d40c-Paper.pdf). Papyan, V., Han, X. Y., and Donoho, D. L. Prevalence of neural collapse during the terminal phase of deep learning training. *Proceedings of the National Academy of Sciences*, 117(40):24652–24663, September 2020. ISSN 1091-6490. doi: 10.1073/pnas.2015509117. URL [http:](http://dx.doi.org/10.1073/pnas.2015509117) [//dx.doi.org/10.1073/pnas.2015509117](http://dx.doi.org/10.1073/pnas.2015509117). Qian, Q., Shang, L., Sun, B., Hu, J., Li, H., and Jin,
  - R. Softtriple loss: Deep metric learning without triplet sampling, 2020. URL [https://arxiv.org/abs/](https://arxiv.org/abs/1909.05235) [1909.05235](https://arxiv.org/abs/1909.05235). Roth, K., Milbich, T., Sinha, S., Gupta, P., Ommer, B., and Cohen, J. P. Revisiting training strategies and generalization performance in deep metric learning. In *International Conference on Machine Learning*, pp. 8242–8252. PMLR, 2020. Schuhmann, C., Vencu, R., Beaumont, R., Kaczmarczyk, R., Mullis, C., Katta, A., Coombes, T., Jitsev, J., and Komatsuzaki, A. Laion-400m: Open dataset of clipfiltered 400 million image-text pairs, 2021. URL [https:](https://arxiv.org/abs/2111.02114) [//arxiv.org/abs/2111.02114](https://arxiv.org/abs/2111.02114). Seidenschwarz, J., Elezi, I., and Leal-Taixe, L. Learning ´ intra-batch connections for deep metric learning, 2021. URL <https://arxiv.org/abs/2102.07753>.

- 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 Sohoni, N., Dunnmon, J., Angus, G., Gu, A., and Re, C. ´ No subclass left behind: Fine-grained robustness in coarse-grained classification problems. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 19339–19352. Curran Associates, Inc., 2020. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2020/file/e0688d13958a19e087e123148555e4b4-Paper.pdf) [cc/paper\\_files/paper/2020/file/](https://proceedings.neurips.cc/paper_files/paper/2020/file/e0688d13958a19e087e123148555e4b4-Paper.pdf) [e0688d13958a19e087e123148555e4b4-Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/e0688d13958a19e087e123148555e4b4-Paper.pdf). [pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/e0688d13958a19e087e123148555e4b4-Paper.pdf). Song, H. O., Xiang, Y., Jegelka, S., and Savarese, S. Deep metric learning via lifted structured feature embedding, 2015. URL [https://arxiv.org/abs/](https://arxiv.org/abs/1511.06452) [1511.06452](https://arxiv.org/abs/1511.06452). Szego, G. ˝ *Orthogonal Polynomials*. American Math. Soc: Colloquium publ. American Mathematical Society, 1975. ISBN 9780821810231. URL [https://books.](https://books.google.co.kr/books?id=ZOhmnsXlcY0C) [google.co.kr/books?id=ZOhmnsXlcY0C](https://books.google.co.kr/books?id=ZOhmnsXlcY0C). Talwalkar, A., Kumar, S., and Rowley, H. Large-scale manifold learning. In *2008 IEEE Conference on Computer Vision and Pattern Recognition*, pp. 1–8, 2008. doi: 10.1109/CVPR.2008.4587670. Tirer, T. and Bruna, J. Extended unconstrained features model for exploring deep neural collapse, 2022. URL <https://arxiv.org/abs/2202.08087>. Vershynin, R. Introduction to the non-asymptotic analysis of random matrices. *Chapter 5 of: Compressed Sensing, Theory and Applications. Edited by Y. Eldar and G. Kutyniok. Cambridge University Press, 2012*, 11 2010. URL <https://arxiv.org/pdf/1011.3027.pdf>. Vershynin, R. *High-Dimensional Probability: An Introduction with Applications in Data Science*. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2018. Vignat, C. A generalized isserlis theorem for location mixtures of gaussian random vectors, 07 2011. URL <https://arxiv.org/pdf/1107.2309.pdf>. Wah, C., Branson, S., Welinder, P., Perona, P., and Belongie,
  - S. The caltech-ucsd birds-200-2011 dataset. 2011. Wang, H., Wang, Y., Zhou, Z., Ji, X., Gong, D., Zhou, J., Li, Z., and Liu, W. Cosface: Large margin cosine loss for deep face recognition, 2018. URL [https://arxiv.](https://arxiv.org/abs/1801.09414) [org/abs/1801.09414](https://arxiv.org/abs/1801.09414). Yang, Y., Steinhardt, J., and Hu, W. Are neurons actually collapsed? on the fine-grained structure in neural representations, 2023. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2306.17105) [2306.17105](https://arxiv.org/abs/2306.17105). Yosinski, J., Clune, J., Bengio, Y., and Lipson, H. How transferable are features in deep neural networks?, 2014. URL <https://arxiv.org/abs/1411.1792>. Zavatone-Veth, J. A., Yang, S., Rubinfien, J. A., and Pehlevan, C. Neural networks learn to magnify areas near decision boundaries, 2023. Zhai, A. and Wu, H.-Y. Classification is a strong baseline for deep metric learning, 2019. URL [https://arxiv.](https://arxiv.org/abs/1811.12649) [org/abs/1811.12649](https://arxiv.org/abs/1811.12649). Zhou, J., Li, X., Ding, T., You, C., Qu, Q., and Zhu, Z. On the optimization landscape of neural collapse under mse loss: Global optimality with unconstrained features, 2022. URL <https://arxiv.org/abs/2203.01238>. Zhou, J., Wang, P., and Zhou, D.-X. Generalization analysis with deep relu networks for metric and similarity learning, 2024. URL [https://arxiv.org/abs/2405.](https://arxiv.org/abs/2405.06415) [06415](https://arxiv.org/abs/2405.06415). Zhu, Z., Ding, T., Zhou, J., Li, X., You, C., Sulam, J., and Qu, Q. A geometric analysis of neural collapse with unconstrained features, 2021. URL [https://arxiv.](https://arxiv.org/abs/2105.02375) [org/abs/2105.02375](https://arxiv.org/abs/2105.02375).

689 690

694

696

698

700

704

706

708 709

711

# A. Additional Related Works

Feature Transferability in Deep Metric Learning The explanation for how Deep Metric Learning learns transferable features towards unseen data remains insufficient. [Chopra et al.](#page-8-9) [\(2005\)](#page-8-9) suggested that CNNs' robustness to geometric distortions enables the creation of generalizable features. This explanation has been replaced in transformer-based research by the idea that, without the inductive biases of CNNs, transformers are less constrained and thus capable of extracting generalizable features [\(El-Nouby et al.,](#page-9-13) [2021;](#page-9-13) [Caron et al.,](#page-8-10) [2021\)](#page-8-10). Additionally, following the manifold hypothesis [\(Chang](#page-8-11) [et al.,](#page-8-11) [2003;](#page-8-11) [Lee et al.,](#page-10-15) [2003;](#page-10-15) [Talwalkar et al.,](#page-11-8) [2008;](#page-11-8) [Goodfellow et al.,](#page-9-14) [2016\)](#page-9-14), [Liu et al.](#page-10-16) [\(2018\)](#page-10-16); [Ermolov et al.](#page-9-15) [\(2022\)](#page-9-15) explained that normalized softmax for metric learning works well because hyperspherical/hyperbolic feature space and the data lies on a manifold. However, these studies do not provide a detailed analysis of how features are learned and transferred through classification.

Neural Collapse (NC) and Features learned by Classifiers There exist studies exploring Neural Collapse (NC) and features learned by classifiers that cannot be explained under the free variable assumption. [Hui et al.](#page-9-16) [\(2022\)](#page-9-16) argue that NC does not manifest on test data. [Sohoni et al.](#page-11-9) [\(2020\)](#page-11-9); [Yang et al.](#page-11-10) [\(2023\)](#page-11-10) claim that even on training data, NC is not fully realized, with critical fine-grained structures concealed. Notably, [Yang et al.](#page-11-10) [\(2023\)](#page-11-10) utilized a two-layer network to analyze training data features. Regarding NC on novel data, [Galanti et al.](#page-9-17) [\(2022\)](#page-9-17) statistically analyze NC in transfer learning, suggesting that NC generalizes not only to new samples within training classes but also to unseen classes with empirical observations. However, their analysis is constrained by focusing on general function spaces rather than specific neural network architectures.

MSE for Classification Utilizing MSE in classification is as well-established as using softmax-cross entropy, especially in theoretical analyses of classification problems [\(Han et al.,](#page-9-18) [2022;](#page-9-18) [Zhou et al.,](#page-11-11) [2022\)](#page-11-11).

Generalization Bound for Metric Learning Research on the generalization bounds of metric learning related to the U-process we use is also ongoing [\(Bellet & Habrard,](#page-8-12) [2015;](#page-8-12) [Huai et al.,](#page-9-19) [2019;](#page-9-19) [Zhou et al.,](#page-11-12) [2024\)](#page-11-12). However, these studies do not analyze the exact feature learning structure.

# B. Empirical Insights into High-Dimensional Asymptotics

In asymptotic analysis, n, d, N → ∞ is crucial for observe result. Please see [Figure 12,](#page-13-0) [Figure 13](#page-14-0) for the cohesion and Separability in R 2000 , R 20000 , R <sup>320000</sup>. As the dimension increases, the range where cohesion and Separability align with our expectations expands.

For component analysis, please see [Figure 14,](#page-15-0) [Figure 15,](#page-15-1) [Figure 16](#page-18-0) , [Figure 17,](#page-19-0) [Figure 18,](#page-20-0) [Figure 19](#page-21-0)

# C. Additional Observation of Multi Classes Feature Analysis

See [Figure 21](#page-22-0) for multi-directional training result. For F L , and spike<sup>L</sup> term depiced in [Figure 22,](#page-23-0) [Figure 23.](#page-23-1)

# D. Additional Results of two-classes Experiments

# D.1. Additional setup for Experiment I, II, III

We set d = n = N = 2<sup>11</sup> and use Shifted ReLU. We repeat each experiment with 3 different initializations of the neural network parameters.

Training Datasets (Data 1) two uniform distributions over a radius-√ d ball, (Data 2) two multi-dimensional element-wise truncated Gaussian distributions, and (Data 3) two uniform distributions over a radius-√ d sphere, symmetric about the origin . The two means of training class are denoted as v and −v, respectively. For Data 1, 3 v ≜ 2r · u, with u ∼ Unif d−1 . For Data 2, one class has support on [1, ∞) across all dimensions, while the other class has support on (−∞, −1].

Evaluation Datasets Eval 1, 2 use the projected Gaussian distribution, which is projected onto the mean direction of one training data v, as defined in equation [9.](#page-16-5) For Eval 1, we translate mean of projected Gaussian distribution with e, and

<sup>2</sup>The Sub-Gaussian property is proven for Data 1 and 3 in [Vershynin](#page-11-5) [\(2018\)](#page-11-5), and for Data 2 in Lemma [L.1.](#page-51-0)

![](_page_13_Figure_1.jpeg)

Figure 12: Cohesion in R , R , R (left to right), with the computed range expanding from top to bottom.

![](_page_14_Figure_2.jpeg)

Figure 13: Separability in R , R , R (left to right), with the computed range expanding from top to bottom.

![](_page_15_Figure_1.jpeg)

![](_page_15_Figure_3.jpeg)

Figure 14: Component analysis of Cohesion in R , R , R (left to right) in range [−100, 100], top: the dominant last component, bottom: sum of the other terms.

Figure 15: Component analysis of Cohesion in R , R , R (left to right) in range [−500, 500], top: the dominant last component, bottom: sum of the other terms.

887 888

890

894

896

898

911

914 915 916

918

924

928

934

for Eval 2, we Rotate mean of projected Gaussian distribution with R ∈ R and fixed e. We generate 300 distinct rotation matrices R using the process in [Appendix O.](#page-58-0) The projected gaussian distribution is sampled as follows,

$$z - \frac{z^\top \nu \nu}{\|\nu\|^4} + \nu, \quad \text{where} \quad z \sim \mathcal{N}(0, cI). \quad (9)$$

For Eval 1, ν ≜ ev, c = 1 and for Eval 2, ν ≜ Rev, c = 10−<sup>1</sup> with e = 0.01 for Data 2 experiment and e = 0.008 for Data 1 and 3 experiments, R ∈ SO(d).

#### D.2. Comprehensive Results of All Experiments

The overall experimental results for *Cohesion* and *Separability* are shown in [Figure 24.](#page-24-0) The results for Eval 1 experimental settings are presented in linear scale in [Figure 25](#page-25-0) and in logarithmic scale in [Figure 26.](#page-26-0) Additionally, as presented in [Figure 7,](#page-6-2) experiments for Eval 2 settings on Data 2 and 3 are shown in linear scale in [Figure 27,](#page-27-0) with results for *Cohesion*, *Separability*, and Recall@1 (IP). Furthermore, results for Recall@1 (cos) are presented in linear scale in [Figure 28.](#page-28-0) All observed results align with the theoretical predictions.

# E. Additional Results of Real-world dataset Experiments

[Figure 29](#page-28-1) summarizes the experimental results and the purpose of the experiment. Expr. IV is in Figure [30,](#page-28-2) [31,](#page-29-0) [1.](#page-36-0) Expr. V is e in [Figure 32,](#page-30-0) [Table 2.](#page-36-0) Expr. VI is in [Figure 33.](#page-31-0) Expr. VII is in Figure [34,](#page-32-0) [35,](#page-33-0) [36,](#page-34-0) [37,](#page-35-0) Table [3,](#page-37-0) and [4.](#page-38-0)

## E.1. Relation between Expr. V and VI

On the other hand, certain results from Expr. V align with those from Expr. VI. As shown in [Table 5,](#page-57-1) for datasets such as CAR and CUB, the number of additional classes introduced by the *sub In1k* dataset is significantly larger compared to SOP. For these data, inclusion of the additional *sub In1k* dataset contributes to improved *recall@1* performance when trained using a Random Initialized Network. Meanwhile, the performance of the pre-trained network is not significantly affected by the additional dataset. We attribute this to the fact that the pre-trained model is additionally re-trained using the same ImageNet dataset *sub In1k*. These findings suggest that further research on the behavior of pre-trained networks is necessary.

### E.2. Expr. VII: Removing Duplicately *Assigned* Eval Classes

In Expr. VII, as suggested by the theoretical results on *Separability*, we validated whether eliminating duplicate in the *assignments* improves performance. To clarify, we will provide an example of duplicate *assignment* at Note [E.1.](#page-16-0)

*Note* E.1 (Example of duplicate *assignment*)*.* For two train classes C (train) 1 , C (train) 2 and two test classes c (test) 1 , c (test) 2 , if most instances of c (test) 1 and c (test) 2 are classified as C (train) 1 , both test classes are assigned to C (train) 1 , resulting in duplication. Conversely, if c (test) 1 is classified as C (train) and c (test) 2 as C (train) 1 , they are assigned without duplication.

To validate, we introudce treatment and control groups. For treatment group, we eliminate duplicate in the textitassignments for the train classes, i.e. , for each unseen class, the most frequently classified training class is aggregated, and the classes are randomly removed to ensure that the selected training classes become unique [\(2\)](#page-17-0). For the control group, we performed random selection of the same number of classes of treatment group [\(1\)](#page-17-1). These two groups are evaluated using *recall@1*. This process was repeated five times, and the average was reported. The experimental results are presented in [34,](#page-32-0) [35,](#page-33-0) [36,](#page-34-0) [37,](#page-35-0) Table [3,](#page-37-0) and [4.](#page-38-0) A total of 64 experiments are conducted, of which 51 demonstrated performance improvements: the estimated success rate is 79%. There is a 1.73%± 2.87% average improvement in recall@1, with a maximum improvement of 13.65%, a minimum decrease of -3.28%. These findings suggest that the duplicate reduction treatment group outperforms the randomly removed group with a binomial test p-value of 9.40 × 10−<sup>7</sup> .

# F. Limitations and Future Work

While our study provides valuable insights into feature learning and transferability, several important directions remain for future research. First, while the Hermite approximation aided our feature analysis, it posed numerical challenges due to the discrepancy between polynomials and nonlinear neural networks. Specifically, the need for extremely high-dimensional approximations [Figure 2](#page-3-9) and the lack of precise scaling alignment between the approximation and the neural networks in

938

954

956

958

971

974

976

978

Algorithm 1 Random Sampling

Input: Number if unseen classes u, number of classes |L| Output: Sampled class set Srandom Set Srandom ← random.sample({0, 1, . . . , u − 1}, |L|) return Srandom

Algorithm 2 Duplicated *assignment* reduction sampling

Input: Model f, unseen data loader D, number of train classes Ctrain, number of unseen classes Cunseen

Output: Sampled class set Snondup Initialize counter matrix counter ← 0

Cunseen×Ctrain

for (img, label) in D do

pred ← f(img) *Predicted class indices*

Update counter: counter[label, pred] += 1

end for

top1 index ← argsort(counter, dim = 1, descending = True)[..., 0]

unique label ← unique(top1 index)

Initialize Snondup ← ∅

for each label ℓ in unique label do

I<sup>ℓ</sup> ← {i | top1 index[i] = ℓ} *Indices corresponding to label* ℓ isample ← random.sample(Iℓ, 1) *Select one random index*

Snondup ← Snondup ∪ {isample}

end for return Snondup

finite dimensions [Figure 4.](#page-4-2)

These limitations highlight the need for alternative approximation techniques or analytical approaches. Second, the relationship between semantic similarity and train-unseen similarity requires further theoretical exploration. Third, an important direction for future research is expanding the concepts of cohesion and Separability to multi-class softmax classification problems, incorporating normalization and temperature scaling to better align with practical settings or Neural Collapse research. Finally, recently [Zavatone-Veth et al.](#page-11-13) [\(2023\)](#page-11-13) suggest neural networks tend to compress the feature space around training data while expanding the regions between decision boundaries. We consider this phenomenon appears closely related to the train-unseen similarity-driven cohesion and Separability observed in our study. Investigating this connection through the lens of Riemannian geometry could yield novel insights into the fundamental structure of learned representations.

![](_page_18_Figure_1.jpeg)

Figure 16: Component analysis of Cohesion in R , R , R (left to right) in range [−1000, 1000], top: the dominant last component, bottom: sum of the other terms.

![](_page_19_Figure_1.jpeg)

 dominant last component, bottom: sum of the other terms.

![](_page_20_Figure_1.jpeg)

Figure 18: Component analysis of Separability in R , R , R (left to right) in range [−500, 500], top: the dominant last component, bottom: sum of the other terms.

![](_page_21_Figure_1.jpeg)

Figure 19: Component analysis of Separability in R , R , R (left to right) in range [−1000, 1000], top: the dominant last component, bottom: sum of the other terms.

![](_page_22_Figure_1.jpeg)

Figure 21: Extra results of [subsection 4.2](#page-4-4) experiments for multiple β<sup>i</sup> direction

![](_page_23_Figure_2.jpeg)

Figure 23: Comparison of log average slope between F L , and spikeL. Midpoint (β1) Interpolation (β2) Extrapolation (β3) Orthogonal (β4) F L is not influenced by spikes and generates random features in all cases. spike<sup>L</sup> is influenced only by spikes, so when using only the β<sup>1</sup> or β<sup>4</sup> spikes, the two features are always mapped to the same position.

Figure 22: Extra results of [subsection 4.2](#page-4-4) experiments for seperate term F L , and spikeL.

![](_page_23_Figure_4.jpeg)

![](_page_24_Figure_1.jpeg)

Figure 24: Summary of the synthetic data experiments: The large and dark circles represent low *train-unseen similarity*, while the small and light circles indicate high *train-unseen similarity*. The datasets D1, D2, and D<sup>3</sup> correspond to synthetic Data 1, 2, and 3, respectively. C denotes *Cohesion*, and S denotes *Separability*. In panels (a) and (b), the two unseen classes are *assigned* to different training classes (i.e., a positive-negative), and as the *train-unseen similarity* increases, both *Separability* and *Cohesion* increase accordingly. In contrast, in panel (c), the two unseen classes are *assigned* to the same training class (i.e., a positive-positive), leading to a decrease in *Separability*. These observations are consistent with our theoretical predictions. We scaled all measurement using the absolute value at the 85th percentile.

![](_page_25_Figure_1.jpeg)

 Figure 25: Expr. I: translation(e) variation case (linear scale). is after one step training. is from initialization. As the *train-unseen similarity* increases, both cohesion and Separability become larger due to pos-neg setup.

![](_page_26_Figure_1.jpeg)

Figure 26: Expr. I: translation(e) variation (log scale). is after one step training. is from initialization. As the *train-unseen similarity* increases, both cohesion and Separability become larger due to pos-neg setup.

![](_page_27_Figure_1.jpeg)

Figure 27: Expr. II, Expr. III: rotation(R) variation (linear scale). is after one step training. is from initialization. Expr. II is pos-neg. Expr. III is pos-pos.

![](_page_28_Figure_2.jpeg)

![](_page_28_Figure_4.jpeg)

Figure 28: Recall@1 with cosine similarity of Expr. II, Expr. III: rotation(R) variation (linear scale). is after one step training. is from initialization. Expr. II is pos-neg. Expr. III is pos-pos.

Figure 29: Expr. IV: High clustering performance with same train-unseen domain. Expr. V: Extra unrelated training classes do not affect *recall@1* performance. Expr. VI: Extra related training classes improve *recall@1* performance. Expr. VII: Removing duplicately *assigned* eval classes improves performance over random removal.

![](_page_28_Figure_6.jpeg)

Figure 30: Expr. IV on ResNet18 with *Domain* datasets (CAR, CUB, SOP, ISC)

ResNet18 (init) ResNet50 (init)

ResNet18 (pre) ResNet50 (pre) Figure 31: Expr. IV on ResNet18, ResNet50 with Domain + In(S) e.g. CAR+I(V), CUB+I(B), SOP+I(P), ISC+I(C)

![](_page_30_Figure_3.jpeg)

Figure 32: Expr. V, additional results, it is represented as follows Domain Domain + Related Subset of In1k Domain + Whole In1k subsampled Adding unrelated classes for training does not significantly affect the performance.

![](_page_31_Figure_2.jpeg)

Figure 33: Expr VI, it is represented as follows: ResNet18 , ResNet50 , Dataset car, cub, sop, isc. As the steps increased and related classes were added, performance generally improved consistently.

![](_page_32_Figure_2.jpeg)

![](_page_32_Picture_3.jpeg)

Figure 34: Expr. VII, ResNet18 (Init), depending on the experimental setup, there are three cases: removing redundancy, randomly selecting the same number of classes as those with redundancy removed, and using all test classes. For dataset we use, we denote as 'Train data(Test data)'. 'In' denote using whole classes of subsampled ImageNet.

![](_page_33_Picture_2.jpeg)

![](_page_33_Figure_3.jpeg)

Figure 35: Expr. VII, ResNet18 (Pre), depending on the experimental setup, there are three cases: removing redundancy, randomly selecting the same number of classes as those with redundancy removed, and using all test classes. For dataset we use, we denote as 'Train data(Test data)'. 'In' denote using whole classes of subsampled ImageNet.

![](_page_34_Figure_2.jpeg)

![](_page_34_Picture_3.jpeg)

Figure 36: Expr. VII, ResNet50 (Init), depending on the experimental setup, there are three cases: removing redundancy, randomly selecting the same number of classes as those with redundancy removed, and using all test classes. For dataset we use, we denote as 'Train data(Test data)'. 'In' denote using whole classes of subsampled ImageNet.

![](_page_35_Picture_2.jpeg)

![](_page_35_Figure_3.jpeg)

Figure 37: Expr. VII, ResNet50 (Pre), depending on the experimental setup, there are three cases: removing redundancy, randomly selecting the same number of classes as those with redundancy removed, and using all test classes. For dataset we use, we denote as 'Train data(Test data)'. 'In' denote using whole classes of subsampled ImageNet.

1983 1984

1986 1987

1989 1990 1991

1994

1996 1997

2014

2016

2018 2019

2024

2026

2029

Table 1: Table results for Expr. IV

ResNet18 (Randomly Initialized)

|          | CAR    | CUB    | SOP    | ISC    |
|----------|--------|--------|--------|--------|
| CAR+I(V) | 0.3922 | 0.0847 | 0.3126 | 0.2079 |
| CAR      | 0.2383 | 0.0685 | 0.2766 | 0.1994 |
| I(V)     | 0.1117 | 0.0618 | 0.2610 | 0.1793 |
| CUB+I(B) | 0.1456 | 0.1205 | 0.3117 | 0.2067 |
| CUB      | 0.1432 | 0.1089 | 0.3179 | 0.1998 |
| I(B)     | 0.0973 | 0.0640 | 0.2658 | 0.1703 |
| SOP+I(P) | 0.1753 | 0.0748 | 0.3720 | 0.3304 |
| SOP      | 0.1754 | 0.0876 | 0.3790 | 0.3306 |
| I(P)     | 0.1405 | 0.0586 | 0.3129 | 0.2327 |
| ISC+I(C) | 0.1409 | 0.0613 | 0.3295 | 0.4870 |
| ISC      | 0.1328 | 0.0685 | 0.3338 | 0.4887 |
| I(C)     | 0.0908 | 0.0471 | 0.2485 | 0.1823 |

ResNet50 (Randomly Initialized)

|          | CAR    | CUB    | SOP    | ISC    |
|----------|--------|--------|--------|--------|
| CAR+I(V) | 0.3280 | 0.0879 | 0.3226 | 0.2000 |
| CAR      | 0.2067 | 0.0495 | 0.2611 | 0.1583 |
| I(V)     | 0.1048 | 0.0459 | 0.2670 | 0.1410 |
| CUB+I(B) | 0.0755 | 0.0527 | 0.2303 | 0.1414 |
| CUB      | 0.0626 | 0.0393 | 0.1950 | 0.1081 |
| I(B)     | 0.0456 | 0.0358 | 0.1954 | 0.1074 |
| SOP+I(P) | 0.1662 | 0.0829 | 0.3812 | 0.2934 |
| SOP      | 0.1725 | 0.0743 | 0.3750 | 0.2754 |
| I(P)     | 0.0940 | 0.0422 | 0.2716 | 0.1697 |
| ISC+I(C) | 0.1090 | 0.0550 | 0.3001 | 0.5318 |
| ISC      | 0.1022 | 0.0503 | 0.2699 | 0.4581 |
| I(C)     | 0.0625 | 0.0412 | 0.2294 | 0.1446 |

ResNet18 (ImageNet 1K Pretrained)

|          | CAR    | CUB    | SOP    | ISC    |
|----------|--------|--------|--------|--------|
| CAR+I(V) | 0.8610 | 0.1131 | 0.4104 | 0.2133 |
| CAR      | 0.8680 | 0.1008 | 0.3966 | 0.1931 |
| I(V)     | 0.4210 | 0.1698 | 0.4618 | 0.2507 |
| CUB+I(B) | 0.3474 | 0.5289 | 0.4745 | 0.2171 |
| CUB      | 0.3476 | 0.5366 | 0.4872 | 0.2527 |
| I(B)     | 0.3771 | 0.3400 | 0.5062 | 0.2278 |
| SOP+I(P) | 0.4073 | 0.1565 | 0.4775 | 0.2827 |
| SOP      | 0.3802 | 0.1499 | 0.4827 | 0.3261 |
| I(P)     | 0.4003 | 0.2076 | 0.4838 | 0.2569 |
| ISC+I(C) | 0.2420 | 0.0976 | 0.4616 | 0.7098 |
| ISC      | 0.2130 | 0.0847 | 0.4550 | 0.7115 |
| I(C)     | 0.3738 | 0.2227 | 0.4994 | 0.2457 |

ResNet50 (ImageNet 1K Pretrained)

|          | CAR    | CUB    | SOP    | ISC    |
|----------|--------|--------|--------|--------|
| CAR+I(V) | 0.9081 | 0.1268 | 0.4192 | 0.1805 |
| CAR      | 0.9078 | 0.1020 | 0.3945 | 0.1673 |
| I(V)     | 0.4013 | 0.1648 | 0.4815 | 0.2330 |
| CUB+I(B) | 0.2831 | 0.5657 | 0.4580 | 0.1895 |
| CUB      | 0.3075 | 0.5778 | 0.4794 | 0.2203 |
| I(B)     | 0.3212 | 0.3337 | 0.4781 | 0.1846 |
| SOP+I(P) | 0.4662 | 0.2264 | 0.6367 | 0.3702 |
| SOP      | 0.4666 | 0.2200 | 0.6276 | 0.3700 |
| I(P)     | 0.3547 | 0.2208 | 0.4602 | 0.2337 |
| ISC+I(C) | 0.2301 | 0.1207 | 0.5376 | 0.8718 |
| ISC      | 0.2230 | 0.1274 | 0.5390 | 0.8710 |
| I(C)     | 0.3655 | 0.2311 | 0.5167 | 0.2413 |

Table 2: Table results of performance for Expr. V.

ResNet18 (Randomly Initialized)

|          | CAR    | CUB    | SOP    | ISC    |
|----------|--------|--------|--------|--------|
| D        | 0.2383 | 0.1089 | 0.3790 | 0.4887 |
| D+I(Sub) | 0.3922 | 0.1205 | 0.3720 | 0.4870 |
| D+I      | 0.3074 | 0.1404 | 0.3591 | 0.4532 |

ResNet50 (Randomly Initialized)

|          | CAR    | CUB    | SOP    | ISC    |
|----------|--------|--------|--------|--------|
| D        | 0.2067 | 0.0393 | 0.3750 | 0.4581 |
| D+I(Sub) | 0.3280 | 0.0527 | 0.3812 | 0.5318 |
| D+I      | 0.3276 | 0.0968 | 0.3726 | 0.4992 |

ResNet18 (ImageNet 1K Pretrained)

|          | CAR    | CUB    | SOP    | ISC    |
|----------|--------|--------|--------|--------|
| D        | 0.8680 | 0.5366 | 0.4827 | 0.7115 |
| D+I(Sub) | 0.8610 | 0.5289 | 0.4775 | 0.7098 |
| D+I      | 0.7604 | 0.5357 | 0.4766 | 0.6897 |

ResNet50 (ImageNet 1K Pretrained)

|          | CAR    | CUB    | SOP    | ISC    |
|----------|--------|--------|--------|--------|
| D        | 0.9078 | 0.5778 | 0.6276 | 0.8710 |
| D+I(Sub) | 0.9081 | 0.5657 | 0.6367 | 0.8718 |
| D+I      | 0.7603 | 0.4689 | 0.6360 | 0.8481 |

Table 3: Expr. VII from (Randomly Initialized)

ResNet18 (Randomly Initialized)

| Test         | Train       | Treatment | Random | ∆     | Total |
|--------------|-------------|-----------|--------|-------|-------|
| CAR Test     |             |           |        |       |       |
|              | CAR         | 35.26     | 34.49  | 0.77  | 23.83 |
|              | I(V)        | 29.02     | 26.96  | 2.06  | 11.17 |
|              | CAR+I(V)    | 49.27     | 46.73  | 2.54  | 39.22 |
|              | In          | 39.51     | 36.48  | 3.03  | 25.70 |
| CUB Test     |             |           |        |       |       |
|              | CUB         | 20.01     | 18.49  | 1.52  | 10.89 |
|              | I(B)        | 16.36     | 14.75  | 1.61  | 6.40  |
|              | CUB+I(B)    | 19.58     | 18.39  | 1.19  | 12.05 |
|              | In          | 32.16     | 28.74  | 3.42  | 21.49 |
| ISC Test     |             |           |        |       |       |
|              | ISC         | 60.64     | 59.45  | 1.19  | 48.87 |
|              | I(C)        | 60.93     | 57.78  | 3.15  | 18.23 |
|              | ISC+I(C)    | 59.59     | 59.11  | 0.48  | 48.70 |
|              | In          | 45.01     | 46.92  | -1.91 | 24.75 |
| SOP Test     |             |           |        |       |       |
|              | SOP         | 43.58     | 42.96  | 0.62  | 37.90 |
|              | I(P)        | 49.76     | 48.45  | 1.31  | 31.29 |
|              | SOP+I(P)    | 42.57     | 42.12  | 0.45  | 37.20 |
|              | In          | 51.84     | 54.03  | -2.19 | 38.82 |
| Average      | Improvement |           |        | 1.20  |       |
| Success Rate |             |           |        | 0.875 |       |

ResNet50 (Randomly Initialized)

| Test         | Train       | Treatment | Random | ∆      | Total |
|--------------|-------------|-----------|--------|--------|-------|
| CAR Test     |             |           |        |        |       |
|              | CAR         | 30.45     | 29.95  | 0.50   | 20.67 |
|              | I(V)        | 24.49     | 22.16  | 2.33   | 10.48 |
|              | CAR+I(V)    | 42.25     | 42.67  | -0.42  | 32.80 |
|              | In(CAR)     | 51.69     | 42.39  | 9.30   | 30.06 |
| CAR Test     |             |           |        |        |       |
|              | CUB         | 13.24     | 15.84  | -2.60  | 3.93  |
|              | I(B)        | 21.20     | 16.65  | 4.55   | 3.58  |
|              | CUB+I(B)    | 16.30     | 13.66  | 2.64   | 5.27  |
|              | In          | 48.10     | 39.59  | 8.51   | 28.06 |
| CAR Test     |             |           |        |        |       |
|              | ISC         | 60.63     | 59.41  | 1.22   | 45.81 |
|              | I(C)        | 53.67     | 51.22  | 2.45   | 14.46 |
|              | ISC+I(C)    | 67.46     | 66.88  | 0.58   | 53.18 |
|              | In          | 44.60     | 44.85  | -0.25  | 22.85 |
| CAR Test     |             |           |        |        |       |
|              | SOP         | 44.02     | 43.34  | 0.68   | 37.50 |
|              | I(P)        | 44.93     | 45.22  | -0.29  | 27.16 |
|              | SOP+I(P)    | 43.51     | 43.70  | -0.19  | 38.12 |
|              | In          | 59.49     | 59.47  | 0.02   | 42.93 |
| Average      | Improvement |           |        | 1.81   |       |
| Success Rate |             |           |        | 0.6875 |       |

2099 2100

2104

2106

2109

2111

2114

2116

2119

2124

2126

2129

2134

2136

Table 4: Expr. VII (ImageNet 1K Pretrained)

ResNet18 (ImageNet 1K Pretrained)

| Test         | Train       | Treatment | Random | ∆      | Total |
|--------------|-------------|-----------|--------|--------|-------|
| CAR Test     |             |           |        |        |       |
|              | CAR         | 90.90     | 90.33  | 0.57   | 86.80 |
|              | I(V)        | 64.51     | 65.03  | -0.52  | 42.10 |
|              | CAR+I(V)    | 90.06     | 88.79  | 1.27   | 86.10 |
|              | In(CAR)     | 71.77     | 73.08  | -1.31  | 26.00 |
| CAR Test     |             |           |        |        |       |
|              | CUB         | 66.19     | 63.12  | 3.07   | 53.66 |
|              | I(B)        | 48.67     | 46.90  | 1.77   | 34.00 |
|              | CUB+I(B)    | 64.48     | 63.89  | 0.59   | 52.89 |
|              | In          | 44.95     | 39.30  | 5.65   | 30.32 |
| CAR Test     |             |           |        |        |       |
|              | ISC         | 78.81     | 77.15  | 1.66   | 71.15 |
|              | I(C)        | 70.48     | 66.47  | 4.01   | 24.57 |
|              | ISC+I(C)    | 78.58     | 77.35  | 1.23   | 70.98 |
|              | In          | 32.65     | 35.78  | -3.13  | 13.85 |
| CAR Test     |             |           |        |        |       |
|              | SOP         | 52.45     | 51.81  | 0.64   | 48.27 |
|              | I(P)        | 66.72     | 66.81  | -0.09  | 48.38 |
|              | SOP+I(P)    | 51.34     | 51.01  | 0.33   | 47.75 |
|              | In          | 46.31     | 46.95  | -0.64  | 30.66 |
| Average      | Improvement |           |        | 0.94   |       |
| Success Rate |             |           |        | 0.6875 |       |

ResNet50 (ImageNet 1K Pretrained)

| Test         | Train       | Treatment | Random | ∆      | Total |
|--------------|-------------|-----------|--------|--------|-------|
| CAR Test     |             |           |        |        |       |
|              | CAR         | 93.78     | 93.57  | 0.21   | 90.77 |
|              | I(V)        | 70.12     | 63.34  | 6.78   | 40.13 |
|              | CAR+I(V)    | 94.45     | 93.34  | 1.11   | 90.81 |
|              | In(CAR)     | 84.20     | 77.43  | 6.77   | 32.51 |
| CAR Test     |             |           |        |        |       |
|              | CUB         | 71.44     | 68.51  | 2.93   | 57.78 |
|              | I(B)        | 47.78     | 46.19  | 1.59   | 33.37 |
|              | CUB+I(B)    | 70.63     | 67.15  | 3.48   | 56.56 |
|              | In          | 75.96     | 62.32  | 13.64  | 35.53 |
| CAR Test     |             |           |        |        |       |
|              | ISC         | 91.35     | 90.49  | 0.86   | 87.10 |
|              | I(C)        | 68.62     | 71.90  | -3.28  | 24.13 |
|              | ISC+I(C)    | 91.60     | 90.59  | 1.01   | 87.18 |
|              | In          | 39.54     | 35.39  | 4.15   | 8.68  |
| CAR Test     |             |           |        |        |       |
|              | SOP         | 68.40     | 68.07  | 0.33   | 62.75 |
|              | I(P)        | 66.24     | 64.09  | 2.15   | 64.02 |
|              | SOP+I(P)    | 68.83     | 68.40  | 0.43   | 63.66 |
|              | In          | 59.94     | 54.78  | 5.16   | 28.87 |
| Average      | Improvement |           |        | 2.96   |       |
| Success Rate |             |           |        | 0.9375 |       |

2154

2156

2164

2166

2169

2174

2176

2194

2196

# G. Additional Notations

The operator diag(·) creates a matrix with the elements of the input vector placed along the diagonal. Let 1condition be 1 if the condition is true and 0 otherwise. Let m! be factorials of m. Let n!! be double factorial. We define (−1)!! = 0!! = 1. For oP, OP, Θ<sup>P</sup> notations we follow [Moniri et al.](#page-10-5) [\(2024\)](#page-10-5) ∥·∥<sup>F</sup> is the Frobenius norm. ∥·∥<sup>∞</sup> is the infinity norm. ∥·∥<sup>ψ</sup><sup>2</sup> is orlicz-2 norm e (i) Standard basis vector with 1 at position i. ⌊n/2⌋ denotes the floor of n/2. Γ(z) is the Gamma function.

Additional information of Hermite Polynomials We employ the probabilist's Hermite polynomials [\(Szego˝,](#page-11-14) [1975;](#page-11-14) [Bienstman,](#page-8-13) [2023;](#page-8-13) [Moniri et al.,](#page-10-5) [2024\)](#page-10-5). We denote Hk(x) as k-th Hermite polynomial.

The n-th Hermite polynomials, Hn(·), are defined by the recurrence relation: Hn+1(x) = xHn(x)−nHn−1(x), for n ≥ 1, with the initial conditions H0(x) = 1, H1(x) = x. Using this recurrence, we have H2(x) = x <sup>2</sup> − 1, H3(x) = x <sup>3</sup> − 3x, · · · .

Hermite polynomials can be represented as the following explicit form:

$$H_n(x) = (-1)^n e^{\frac{x^2}{2}} \frac{d^n}{dx^n} e^{-\frac{x^2}{2}}.$$

for n ∈ <sup>N</sup>0. Lastly, there are another expression:

$$H_n(x) = n! \sum_{m=0}^{\lfloor \frac{n}{2} \rfloor} \frac{(-1)^m}{m!(n-2m)!} \frac{x^{n-2m}}{2^m}$$

The probabilist's Hermite polynomials form an orthogonal set with respect to the standard normal weight function ϕ(x) = √ 1 2π e − <sup>x</sup> <sup>2</sup> on the interval (−∞, ∞). Their orthogonality condition is given by:

$$\int_{-\infty}^{\infty} H_m(x) H_n(x) \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}} dx = n! \mathbf{1}_{m=n}.$$

# H. hermite coef of shifted ReLU

One of the activation function that satisfy our condition [2.1](#page-2-5) is shifted ReLU,

$$\sigma(x) = \max(0, x) - \frac{1}{\sqrt{2\pi}}.$$

This allow hermite decomposition with coefficient is calculated as

$$c_n = \frac{1}{n!} \mathbb{E}_z[\sigma(z) H_n(z)].$$

Then for the zero-th coefficient is calculated as

$$\begin{aligned} c_0 &= \mathbb{E}_z[\sigma(z) \times 1] = \mathbb{E}_z[\max(0, x)] - \frac{1}{\sqrt{2\pi}} \\ &= \int_0^\infty x \phi(x) dx - \frac{1}{\sqrt{2\pi}} = 0 \end{aligned} \quad (10)$$

By the way, if <sup>n</sup> ̸= 0, <sup>E</sup>[ √ 1 2π × <sup>H</sup>n] = √ 2π <sup>E</sup>[1 × <sup>H</sup>n] = √ 2π <sup>E</sup>[H<sup>0</sup> × Hn] = 0 by orthogonality. Thus, shift is only effects on n = 0.

The coefficient c<sup>n</sup> of the expansion of Shifted-ReLU is defined as:

$$c_n = \begin{cases} 0, & \text{if } n = 0, \\ \sum_{m=0}^{\lfloor n/2 \rfloor} \frac{(-1)^m \cdot 2^{\frac{n-2m}{2} - m} \cdot \Gamma(\frac{n-2m+2}{2})}{m! \cdot (n-2m)! \cdot \sqrt{2\pi}}, & \text{otherwise.} \end{cases} \quad (11)$$

2204

2206

2209

2214

2216

2218 2219

2224

2226

2229

2234

2236

![](_page_40_Figure_1.jpeg)

Figure 38: Hermite Coefficient of Shifted ReLU

# I. Proof of [Theorem 3.1](#page-2-3)

In this section, we follow the proof structure of [Ba et al.](#page-8-1) [\(2022\)](#page-8-1) to decompose gradient in our classification learning setting. Unlike their assumption of centered Gaussian training data, we consider non-centered Sub-Gaussian data distributions. In this process, we apply a novel approach involving the concentration of the operator norm on a random matrix. Also, since our framework is not in a teacher-student setting, we use class labels instead of a teacher function.

We will omit the subscript ij since it does not cause any confusion untill equation [35.](#page-43-1) The following statements hold for ∀ij. For the aforementinoed A, B, and C, we obtain bounds for each operator norm as follows

## Lemma I.1.

$$\begin{aligned} \mathbb{P}\left(\|\mathbb{A}\| \leq C\left(\frac{1}{\sqrt{\mathbf{N}}} - C\frac{\sqrt{\mathbf{d}}}{\sqrt{\mathbf{n}\mathbf{N}}}\right)\right) &\leq 2(e^{-c\mathbf{N}} + e^{-c\mathbf{n}}) \\ \mathbb{P}\left(\|\mathbb{B}\| \geq \frac{C}{\mathbf{n}\sqrt{\mathbf{N}\mathbf{d}}}(\sqrt{\mathbf{n}} + \sqrt{\mathbf{d}})(\sqrt{\mathbf{n}} + \sqrt{\mathbf{N}})\log \mathbf{N}\right) &\leq C(e^{-c\mathbf{N}} + e^{-c\mathbf{d}} + \mathbf{N}e^{-c\log^2 \mathbf{n}} + e^{-(\sqrt{\mathbf{n}}+\sqrt{\mathbf{d}})^2}) \quad (12) \\ \mathbb{P}\left(\|\mathbb{C}\| \geq \frac{C}{\sqrt{\mathbf{n}\mathbf{N}}}(2\sqrt{\mathbf{d}} + \sqrt{\mathbf{n}})\log \mathbf{n}\log \mathbf{N}\right) &\leq 2(\mathbf{n}e^{-c\mathbf{d}} + \mathbf{n}e^{-c\log^2 \mathbf{n}} + \mathbf{N}e^{-c\log^2 \mathbf{n}}). \end{aligned}$$

*Proof of Lemma [I.1](#page-40-0) (*A*).* We obtain

$$\mathbb{A} = \frac{c_1}{n\sqrt{N}} X^\top y a^\top. \quad (13)$$

Then, we can find an explicit notation of the norm as

$$\|\mathbb{A}\| = \frac{c_1}{n\sqrt{N}} \|X^\top y a^\top\| = \frac{c_1}{n\sqrt{N}} \|X^\top y\|_2 \|a\|_2 = \frac{c_1}{n\sqrt{N}} (y^\top X X^\top y)^{1/2} \|a\|_2 \quad (14)$$

∥a∥<sup>2</sup> study By definition, a ∼ N(0, <sup>N</sup> ), so √ Nα[i] is a sub-Gaussian. Use Thm 3.3.1 in [Vershynin](#page-11-5) [\(2018\)](#page-11-5),

$$\mathbb{P}\left(\left|\|\sqrt{\mathbf{N}}\alpha\| - \sqrt{\mathbf{N}}\right| \geq t\right) \leq 2e^{-ct^2} \quad \text{let } t = \sqrt{\mathbf{N}} \quad (15)$$

$$\mathbb{P}(\|\alpha\|_2 \leq 1) \leq 2e^{-c\mathbf{N}}$$

 y <sup>⊤</sup>XX⊤y <sup>1</sup>/<sup>2</sup> study Note that the U, V matrices resulting from the SVD belong to the O-group, so there is no length transformation.

$$\begin{aligned} y^\top X X^\top y &= \|X^\top y\|_2^2 = \|U\Sigma V^\top y\|_2^2 = \|\Sigma V^\top y_1\| \\ &= \sum_i \sigma_i^2 |V^\top y_i|^2 \geq \sigma_{\min}^2 \sum_i |V^\top y_i|^2 = \sigma_{\min}^2 \|y\|_2^2 = \mathbf{n} \sigma_{\min}^2 \end{aligned} \quad (16)$$

2259 2260

2264

2266

2269

2274

2276

2279

2281 2282

2289 2290

2294

2296

2299 2300

2304

2306

$$2,300$$
     $\mathbb{P}(\mathcal{A}_B)$  **study**    We choose  $t = C\sqrt{\frac{d}{N}}$ ,  $B = C\sqrt{\frac{d}{N}}$ .

We get y <sup>⊤</sup>XX⊤y <sup>1</sup>/<sup>2</sup> ≥ √ nσmin. σmin is singular value of X which is a anistropic sub-Gaussian matrix. With the result of Remark 1.2 in [Liaw et al.](#page-10-17) [\(2016\)](#page-10-17),

$$\mathbb{P}\sigma_{\min} \leq (\sqrt{\mathbf{n}} - c\sqrt{\mathbf{d}}) \leq e^{-\mathbf{n}}. \quad (17)$$

Therefore, <sup>P</sup>(∥A∥ ≤ <sup>C</sup>( √ N − C √ √ d nN )) <sup>≤</sup> 2(<sup>e</sup> <sup>−</sup>c<sup>N</sup> + e <sup>−</sup>c<sup>n</sup>).

*Fact* I.2 (from [Ba et al.](#page-8-1) [\(2022\)](#page-8-1))*.* For m ∈ R <sup>m</sup>, n ∈ <sup>R</sup> <sup>n</sup>, M ∈ <sup>R</sup> <sup>m</sup>×<sup>n</sup>,

$$\begin{aligned} mn^\top \odot M &= \text{diag}(m)M\text{diag}(n) \\ \|mn^\top \odot M\| &\leq \|\text{diag}(m)\| \|M\| \|\text{diag}(n)\| = \|m\|_\infty \|M\| \|n\|_\infty \end{aligned} \quad (18)$$

Lemma I.3. *For Sub-Gaussian R.V.* a*,*

$$\mathbb{P}(\|a\|_\infty \leq t/\sqrt{N}) \geq 1 - 2Ne^{-ct^2}$$

*Proof.* We use the Hoeffding inequality such that

$$\mathbb{P}(\|a\|_\infty \geq \frac{t}{\sqrt{\mathbf{N}}}) = \mathbb{P}\left(\max_i |a_i| \geq \frac{t}{\sqrt{\mathbf{N}}}\right) \leq \mathbb{P}\left(\bigcup_i \{|a_i| \geq \frac{t}{\sqrt{\mathbf{N}}}\}\right) \leq \sum_i \mathbb{P}\left(|a_i| \geq \frac{t}{\sqrt{\mathbf{N}}}\right) \quad (19)$$

$$\stackrel{\text{i.i.d.}}{=} \mathbf{N}\mathbb{P}\left(|a_i| \geq \frac{t}{\sqrt{\mathbf{N}}}\right) = \mathbb{P}(|\sqrt{\mathbf{N}}a_i| \geq t) \leq 2\mathbf{N} \exp(-ct^2)$$

*Fact* I.4*.* Let a sub-Gaussian random variable v s.t. ∥v∥<sup>ψ</sup><sup>2</sup> ≤ k, and bounded function σ, then σ(v) is Sub-Gaussian, i.e. ∥σ(v)∥<sup>ψ</sup><sup>2</sup> ≤ ∥λ∥<sup>ψ</sup><sup>2</sup> < ∞.

*Proof of Lemma [I.1](#page-40-0) (*B*).*

$$\mathbb{B} = \frac{1}{\mathbf{n}\sqrt{N}} X^\top y a^\top \odot \sigma'_\perp(XW_0) \quad (20)$$

$$\begin{aligned} \|\mathbb{B}\| &\leq \frac{1}{\mathbf{n}\sqrt{\mathbf{N}}} \|X\| \|ya^\top \odot \sigma'_\perp(XW_0)\| \\ &\leq \frac{1}{\mathbf{n}\sqrt{\mathbf{N}}} \|X\| \|ya^\top \odot \sigma'_\perp(XW_0)\| \\ &\leq \frac{1}{\mathbf{n}\sqrt{\mathbf{N}}} \|X\| \|y\|_\infty \|\sigma'_\perp(XW_0)\| \|a\|_\infty \\ &= \frac{1}{\mathbf{n}\sqrt{\mathbf{N}}} \|X\| \|\sigma'_\perp(XW_0)\| \|a\|_\infty \end{aligned} \tag{21}$$

∥σ ′ <sup>⊥</sup>(XW0)∥ study Use the result of D.4 in [Fan & Wang](#page-9-20) [\(2020\)](#page-9-20), which is hold for orthogonal columns. X is sampled from continuous support distribution c1, c2. The first vector is linearly independent with probability 1 due to the continuous support of its distribution. For the second vector, which is drawn independently, the probability that it lies in the span of the first vector is 0, as it also has a continuous density. This reasoning extends to n vectors, implying that, with high probability, they are orthogonal or nearly orthogonal because no vector falls into the span of the others. Thus, ∀B > 0 following is hold.

$$\mathbb{P}(\{\|\sigma'_\perp\| \geq C(\sqrt{\mathbf{n}} + \sqrt{\mathbf{N}})\lambda_\sigma B\}, \mathcal{A}_B) \leq 2e^{-c\mathbf{N}}$$

$$\mathcal{A}_B = \{\{\|W_0\| \leq B\} \cup \{\sum_{i=1}^N (\|W_{0,i}\|^2 - 1)^2 \leq B^2\}\}. \quad (22)$$

Therefore,

$$\mathbb{P}(\|\sigma'_\perp\| \geq C(\sqrt{\mathbf{n}} + \sqrt{\mathbf{N}})\lambda_\sigma B) \leq 2e^{-\mathbf{c}\mathbf{N}} + \mathbb{P}(\mathcal{A}_B^c) \quad (23)$$

2316

2318 2319

2324

2326

2329

2334

2336

2354

2356

2359 2360 2361

case of ∥W0,i∥ ≤ B By Lemma [L.3,](#page-52-0)

$$\mathbb{P}(\|\sqrt{\mathbf{N}}W_0\| \geq 2\sqrt{\mathbf{N}} + \sqrt{\mathbf{d}}) \leq 2e^{-c\mathbf{N}} \Rightarrow \mathbb{P}(\|W_0\| \geq C\sqrt{\frac{\mathbf{d}}{\mathbf{N}}}) \leq 2e^{-c\mathbf{N}} \quad (24)$$

Therefore, ∥W0∥ ≤ B at least w.p. 1 − 2e −cN

case of P<sup>N</sup> <sup>i</sup>=1(∥W0,i∥ <sup>2</sup> − 1)<sup>2</sup> ≤ B<sup>2</sup> By definition, ∥W0,i∥ <sup>2</sup> = 1, so 0 ≤ B<sup>2</sup> , trivialy.

We know <sup>P</sup>(A<sup>c</sup> <sup>B</sup>) ≤ 2e <sup>−</sup>cN .

$$\mathbb{P}(\|\sigma'_{\perp}\| \geq C(\sqrt{\mathbf{n}} + \sqrt{\mathbf{N}})\sqrt{\frac{\mathbf{d}}{\mathbf{N}}}) \leq 2e^{-c\mathbf{N}} \quad (25)$$

Use Lemma [I.3,](#page-41-0) and [L.3,](#page-52-0)

$$\|\sigma'_\perp\| \leq C \left( \sqrt{\frac{\mathbf{nN}}{\mathbf{d}}} + \sqrt{\frac{\mathbf{N}^2}{\mathbf{d}}} \right) \quad \text{w.p. } 1 - C(e^{-c\mathbf{N}} + e^{-c\mathbf{d}}) \quad (26)$$

$$\|a\|_\infty \leq \frac{t}{\sqrt{N}} \quad \text{w.p. } 1 - 2Ne^{-ct^2} \quad (27)$$

$$\|X\| \leq \sqrt{\mathbf{n}} + \sqrt{\mathbf{d}} + t' \qquad \text{w.p. } 1 - 2e^{-ct'^2}. \qquad (28)$$

In summary, we get

$$\|\mathbb{B}\| \leq \frac{C}{\mathbf{n}\sqrt{\mathbf{N}}}(\sqrt{\mathbf{n}} + \sqrt{\mathbf{d}} + t') \left( \sqrt{\frac{\mathbf{n}\mathbf{N}}{\mathbf{d}}} + \sqrt{\frac{\mathbf{N}^2}{\mathbf{d}}} \right) \frac{t}{\sqrt{\mathbf{N}}}$$
let  $t = \log \mathbf{n}$ ,  $t' = \sqrt{\mathbf{n}} + \sqrt{\mathbf{d}}$  (29)
$$\mathbb{P}(\|\mathbb{B}\| \geq \frac{C}{\mathbf{n}\sqrt{\mathbf{N}}})(\sqrt{\mathbf{n}} + \sqrt{\mathbf{d}})(\sqrt{\mathbf{n}} + \sqrt{\mathbf{N}}) \log N \leq C(e^{-\mathbf{c}\mathbf{N}} + e^{-\mathbf{c}\mathbf{d}} + \mathbf{N}e^{-\mathbf{c}\log^2 \mathbf{n}} + e^{-(\sqrt{\mathbf{n}}+\sqrt{\mathbf{d}})^2}).$$

This compelete the proof.

*Proof of Lemma [I.1](#page-40-0) (*C*).* We know that σ ′ is bounded, so ∥σ ′∥<sup>F</sup> ≤ λ<sup>σ</sup> √ nN

$$\mathbb{C} = -\frac{1}{\mathbf{nN}} X^\top \sigma(XW_0)(aa^\top) \odot \sigma'(XW_0), \quad (30)$$

ans we can bound the norm as follows

$$\begin{aligned} \|\mathbb{C}\| &\leq \frac{1}{\mathbf{nN}} \|X\| \|\sigma a a^\top \odot \sigma'\| \\ &\leq \frac{1}{\mathbf{nN}} \|X\| \|\sigma a\|_\infty \|a\|_\infty \|\sigma'\|_F \\ &\leq \frac{\lambda_\sigma}{\sqrt{\mathbf{nN}}} \|X\| \|\sigma a\|_\infty \|a\|_\infty \end{aligned} \quad (31)$$

Control of ∥σa∥<sup>∞</sup> Let t = √ d. Given X s.t. <sup>P</sup>( X<sup>i</sup> − √ d  ≥ √ d) ≤ 2e −ct<sup>2</sup> , consider one element σ X<sup>⊤</sup> <sup>j</sup> W<sup>0</sup> a = P<sup>N</sup> i aiσ X<sup>⊤</sup> <sup>j</sup> W0[i] .

We know a<sup>i</sup> , √ nW0,i is an independent centered sub-Gaussian, and use Fact [I.4,](#page-41-1) then σ <sup>X</sup><sup>⊤</sup> √j N √ NW<sup>0</sup> a is sub-exponential and mean is zero, since ∥aσ(x ⊤ <sup>j</sup> W0,i)∥<sup>ψ</sup><sup>1</sup> ≤ ∥a∥<sup>ψ</sup><sup>2</sup> ∥σ(x ⊤ <sup>j</sup> W0,i)∥<sup>ψ</sup><sup>2</sup> < ∞. Apply the Bernstein inequality for the subexponential,

$$\mathbb{P}(|\sigma(X_j^\top a)| \geq \log \mathbf{n} \text{ given } \{|X_j - \sqrt{\mathbf{d}}| \geq \sqrt{\mathbf{d}}\}) \leq 2e^{-c \log^2 \mathbf{n}}. \quad (32)$$

2369

2374

2376

2379

2389 2390

2394

2396

2399 2400

2401 Sum up for ∀ij,

2408 2409 Lemma J.1. *The following facts will be used in subsequent proofs. Remark* βij ≜ 1 nX<sup>T</sup> ijy *in [Theorem 3.2.](#page-2-4)*

By Lemma [I.3](#page-41-0) <sup>P</sup>(∥a∥<sup>∞</sup> <sup>≤</sup> t/√ N) ≥ 1 − 2Ne −ct<sup>2</sup> , and Lemma [L.3](#page-52-0) with t = √ d

$$\mathbb{P}\left(\|\mathbb{C}\| \geq \frac{C}{\sqrt{\mathbf{n}\mathbf{N}}}(2\sqrt{\mathbf{d}} + \sqrt{\mathbf{n}}) \log \mathbf{n} \log \mathbf{N}\right) \leq 2(\mathbf{n}e^{-\mathbf{c}\mathbf{d}} + ne^{-\mathbf{c}\log^2 \mathbf{n}} + \mathbf{N}e^{-\mathbf{c}\log^2 \mathbf{n}}). \quad (33)$$

*Remark* I.5*.* In the proportional regime, as n, d, N → ∞, these quantities can be interchanged to a constant. Thus, Lemma [I.1](#page-40-0) is reformulated as follows

$$\begin{aligned}\mathbb{P}(\|\mathbf{A}\| \leq \kappa/\sqrt{\mathbf{n}}) &\leq Ce^{-\mathbf{c}\mathbf{n}} \\ \mathbb{P}\left(\|\mathbb{B}\| \geq \frac{C\log \mathbf{N}}{\mathbf{n}}\right) &\leq C(e^{-\mathbf{c}\mathbf{n}} + \mathbf{n}e^{-c\log^2 \mathbf{n}}) \\ \mathbb{P}\left(\|\mathbb{C}\| \geq \frac{C\log^2 \mathbf{N}}{\mathbf{n}}\right) &\leq C(\mathbf{n}e^{-\mathbf{c}\mathbf{n}} + \mathbf{n}e^{-c\log^2 \mathbf{n}})\end{aligned}\tag{34}$$

Also, for gradient, we have

$$\|G\| = \|\mathbb{A} + \mathbb{B} + \mathbb{C}\| \leq \|\mathbb{A}\| + \|\mathbb{B}\| + \|\mathbb{C}\| = O_{\mathbb{P}}\left(\frac{1}{\sqrt{\mathbf{n}}} + \frac{\log \mathbf{n}}{\mathbf{n}} + \frac{\log^2 \mathbf{n}}{\mathbf{n}}\right) = O_{\mathbb{P}}\left(\frac{1}{\sqrt{\mathbf{n}}}\right) \quad (35)$$

Now we denote subscript ij for summary.

*Proof of Theorem [3.1.](#page-2-3)* Using ∥Gij − <sup>A</sup>ij∥ = ∥<sup>B</sup>ij + <sup>C</sup>ij∥ ≤ ∥<sup>B</sup>ij∥ + ∥Cij∥ and Lemma [I.5](#page-43-2)

$$\mathbb{P}\left(\left\|G_{ij} - \mathbb{A}_{ij}\right\| \geq C \frac{\log^2 \mathbf{n}}{\mathbf{n}}\right) \leq \mathbb{P}\left(\left\|G_{ij} - \mathbb{A}_{ij}\right\| \geq C \left(\frac{\log n}{n} + \frac{\log^2 \mathbf{n}}{\mathbf{n}}\right)\right) \leq C ne^{-c \log^2 \mathbf{n}}. \quad (36)$$

Therefore, almost surely, in the proportional limit,

$$\|G_{ij} - \mathbb{A}_{ij}\| \leq C \frac{\log^2 \mathbf{n}}{\mathbf{n}} = \frac{\kappa}{\sqrt{\mathbf{n}}} \frac{C}{\kappa} \frac{\log^2 \mathbf{n}}{\sqrt{\mathbf{n}}} \leq \|\mathbb{A}_{ij}\| \frac{C}{\kappa} \frac{\log^2 \mathbf{n}}{\sqrt{\mathbf{n}}} \leq \kappa' \frac{\log^2 \mathbf{n}}{\sqrt{\mathbf{n}}} (\|G_{ij}\| + \|G_{ij} - \mathbb{A}_{ij}\|). \quad (37)$$

We get (1 − κ ′ log<sup>2</sup> √ <sup>n</sup> n )||Gij − <sup>A</sup>|| ≤ κ ′ log<sup>2</sup> √ <sup>n</sup> n ||Gij ||. For large enough n for 1 − κ ′ log<sup>2</sup> √ <sup>n</sup> <sup>n</sup> ≥ 2 ,

$$\|G_{ij} - \mathbb{A}_{ij}\| \leq \kappa' \frac{\log^2 \mathbf{n}}{\sqrt{\mathbf{n}}} \|G_{ij}\| \leq C \frac{\log^2 \mathbf{n}}{\mathbf{n}}$$

$$\|G - \sum_{i < j} \mathbb{A}_{ij}\| = \|\sum_{i < j} G_{ij} - \mathbb{A}_{ij}\| \leq \sum_{i < j} \|G_{ij} - \mathbb{A}_{ij}\| \leq C \frac{\log^2 \mathbf{n}}{\mathbf{n}}$$

# J. Proof of [Theorem 3.3](#page-2-7)

A. 
$$\|X_{ij}\| = O_{\mathbb{P}}(\sqrt{\mathbf{n}})$$
,  $\|y\| = O_{\mathbb{P}}(\sqrt{\mathbf{n}})$ ,  $\|\beta_{ij}\| = O_{\mathbb{P}}(1)$ 

*B.* ||Xijβijaij || = ||Xβij ||2||aij ||<sup>2</sup> = OP(

√ n)

*C.* ∥W0∥ = OP(1)*,* ||W|| = ||W<sup>0</sup> + G|| ≤ ||W0|| + ||G|| = OP(1)

*D.* ||XijG|| = OP(

√ n)

$$E. \quad M_a \triangleq \|a\|_\infty = \max_{1 \leq i \leq N} |a_i| \leq \frac{C \log^{1/2} \mathbf{n}}{\sqrt{\mathbf{n}}} w \cdot p \quad 1 - 2ne^{-c \log \mathbf{n}}$$

- 2420 2421 2422 2423 2424
- *F.* M<sup>b</sup> <sup>≜</sup> ∥Xβ∥<sup>∞</sup> = max1≤i≤<sup>n</sup> | < X˜[i], β > | ≤ C log<sup>1</sup>/<sup>2</sup> n*, w.p.* 1 − 2ne −c log n
- *G.* M<sup>W</sup><sup>0</sup> <sup>≜</sup> supk≥<sup>1</sup> ||(W0W<sup>⊤</sup> 0 ) ◦k || ≤ C *w.p.* 1 − o(1)
- *H.* ||A◦<sup>k</sup> || ≤ ||A||<sup>k</sup>

2425 2426 2427 2428 2429 *Proof.* It is evident from Lemma [L.3,](#page-52-0) equation [15](#page-40-3) in the proportional regime, that A, B, C, and D hold. To proof E, F, and G, we employ proof techniques adapted from [Moniri et al.](#page-10-5) [\(2024\)](#page-10-5). For E, by Lemma [I.3,](#page-41-0) with t = log <sup>2</sup> n, M<sup>a</sup> ≤ C log 1 √ 2 n n , w.p. 1 − o(1).

$$\begin{aligned} 2430 \quad \text{For F,} \quad & \mathbb{P}(C|x^T \beta| \geq t) = \mathbb{P}(C|x^T \beta - Ex^T \beta + Ex^T \beta| \geq t) \\ 2431 & \\ 2432 & \leq \mathbb{P}(C|x^T \beta - Ex^T \beta| \geq t - C|Ex^T \beta|) \leq 2 \exp(-ct^2). \\ 2433 & \end{aligned} \tag{38}$$

2434 Then, <sup>P</sup>(|x <sup>T</sup> β| ≥ t) ≤ 2 exp(−c(t − Ex<sup>T</sup> β) 2 ) ≤ 2 exp(−ct<sup>2</sup> ).

2435 2436

2437 For G, refer [Moniri et al.](#page-10-5) [\(2024\)](#page-10-5). For H, refer [Bai & Silverstein](#page-8-14) [\(2010\)](#page-8-14) Corollary A.21.

2470 2471 Now we will show

2472 2473 2474 For F,

Therefore, M<sup>b</sup> ≤ C log <sup>2</sup> n, w.p. 1 − o(1) with t = log 1 <sup>2</sup> n.

Corollary J.2 (Corollary of Theorem [3.1\)](#page-2-3). *By Lemma [J.1,](#page-43-3) we have w.p.* 1 − o(1)*,*

$$\|\tilde{X}G - c_1\tilde{X} \sum_{i < j} \beta_{ij} a_{ij}^T\| = O\left(\frac{\log^2 \mathbf{n}}{\mathbf{n}} \cdot \sqrt{\mathbf{n}}\right) = O\left(\frac{\log^2 \mathbf{n}}{\sqrt{\mathbf{n}}}\right) \quad (39)$$

*Remark* J.3*.* W<sup>1</sup> = W<sup>0</sup> + G, so XW˜ <sup>1</sup> = XW˜ <sup>0</sup> + XG˜ . X˜ is i.i.d. copy of training data X

We generalize Corollary [J.2](#page-44-0) i.e. monomial approximation of data-gradient product in polynomial form as Lemma [J.4](#page-44-1) .

Lemma J.4 (Polynomial Approximation of Data-Gradient Product). *For any* k ∈ N*, sufficiently large* n*, and w.p. 1 - o(1),*

$$\|(\tilde{X}G)^{ok} - c_1^k(\tilde{X} \sum_{i < j} \beta_{ij} a_{ij}^T)^{ok}\| = O(\mathbf{n}^{-\frac{k}{2}} \log^{2k} \mathbf{n}) \quad (40)$$

*Proof of Lemma [J.4.](#page-44-1)* k = 1 is trivial Corollary [J.2.](#page-44-0) We follow [Moniri et al.](#page-10-5) [\(2024\)](#page-10-5) for k ≥ 2. We need to show ∃C > 0, w.p. 1-o(1)

$$\|(\tilde{X}G)^{ok} - c_1^k(\tilde{X} \sum_{i < j} \beta_{ij} a_{ij}^T)^{ok}\| \leq C \mathbf{n}^{-\frac{k}{2} \log^{2k} \mathbf{n}} \quad (41)$$

$$\begin{aligned} (\tilde{X}G)^{ok} &= (\tilde{X}G - c_1\tilde{X} \sum_{i$$

Thus,

$$\begin{aligned} & (\tilde{X}G)^{ok} - c_1^k \left( \tilde{X} \sum_{i < j} \beta_{ij} a_{ij}^T \right)^{ok} \\ &= \sum_{j=1}^k \binom{k}{j} (\tilde{X}G - c_1 \tilde{X} \sum_{i < j} \beta_{ij} a_{ij}^T)^{oj} \odot c_1^{k-j} \left( \sum_{i < j} (\tilde{X} \beta_{ij} a_{ij}^T)^{o(k-j)} \right) \end{aligned} \quad (43)$$

$$||(\tilde{X}G - c_1\tilde{X} \sum_{i < j} \beta_{ij} a_{ij}^T)^{oj} \odot c_1^{k-j} (\sum_{i < j} (\tilde{X} \beta_{ij} a_{ij}^T))^{o(k-j)}|| = O_{\mathbb{P}}(\log^{k+j} \mathbf{n} \cdot \mathbf{n}^{-\frac{1}{2}k}).$$

2504

2506

2509

2514

2516

2518 2519

2524 Firstly, we proof µ = 0 case. For centered Sub Gaussian vector g, let z = g <sup>⊤</sup>u, z′ = g <sup>⊤</sup>v, ρ-correlated. s.t. ||u||<sup>2</sup> = ||v||<sup>2</sup> = 1, u <sup>T</sup> v = ρ, then by equation [47](#page-45-3)

2526

$$\begin{aligned}
& \|(\tilde{X}G - c_1\tilde{X} \sum_{i < j} \beta_{ij} a_{ij}^T)^{oj} \odot c_1^{k-j} (\sum_{i < j} (\tilde{X} \beta_{ij} a_{ij}^T))^{o(k-j)}\| \\
& \leq C \|(\tilde{X}G - c_1\tilde{X} \sum_{i < j} \beta_{ij} a^T)^{oj} \odot (\tilde{X} \beta a^T)^{o(k-j)}\| \\
& \leq C \|\text{diag}(\tilde{X}\beta)^{ok-j}\|_{op} \|(\tilde{X}G^T - c_1\tilde{X} \sum_{i < j} \beta_{ij} a^T)^{oj}\|_{op} \|\text{diag}(a)^{ok-j}\| \\
& \leq C(M_a M_b)^{k-j} \|(\tilde{X}G - c_1\tilde{X} \sum_{i < j} \beta_{ij} a^T)^{oj}\|^j \\
& \leq C(n^{-\frac{1}{2}(k-j)} \log^{k-j} \mathbf{n}) \log^{2j} \mathbf{n} \cdot \mathbf{n}^{-\frac{1}{2}j} \\
& = O_{\mathbb{P}}(\mathbf{n}^{-\frac{1}{2}k} \log^{k+j} \mathbf{n})
\end{aligned} \tag{44}$$

Therefore,

$$\|(\tilde{X}G)^{ok} - c_1^k(\tilde{X} \sum_{i < j} \beta_{ij} a_{ij}^T)^{ok}\| = O_{\mathbb{P}}(\mathbf{n}^{-\frac{k}{2}} \log^{2k} \mathbf{n}) \quad (45)$$

Lemma J.5. *Following condition in [section 2,](#page-1-0) Assume event* Ω = supk≥<sup>1</sup> ||(W0W<sup>T</sup> 0 ) ok||op ≤ C *occur, following statement holds.*

$$\|H_j(\tilde{X}W_0)\|_{op} = O_{\mathbb{P}}(\sqrt{n} \log^{\frac{3}{2}} n \sqrt{j!})$$

Lemma J.6. *Given random matrix* A*, Following statement holds,*

$$\mathbb{P}(\|A\|_{op} \geq t) \leq \mathbb{P}(\|\frac{1}{n}AA^T - EAA^T\|_{op} \geq \frac{t^2}{n} - \|EAA^T\|_{op})$$

*Proof of Lemma [J.6.](#page-45-0)*

$$\begin{aligned}\mathbb{P}(\|A\|_{op} \geq t) &= \mathbb{P}(\|A\|_{op}^2 \geq t^2) = \mathbb{P}(\|\frac{1}{n}AA^T\|_{op} \geq \frac{t^2}{n}) \\ &= \mathbb{P}(\|\frac{1}{n}AA^T - EAA^T + EAA^T\|_{op} \geq \frac{t^2}{n}) \\ &\leq \mathbb{P}(\|\frac{1}{n}AA^T - EAA^T\|_{op} + \|EAA^T\|_{op} \geq \frac{t^2}{n}) \\ &= \mathbb{P}(\|\frac{1}{n}AA^T - E(AA^T)\|_{op} \geq \frac{t^2}{n} - E\|AA^T\|_{op})\end{aligned}\tag{46}$$

Lemma J.7. *Following condition of Lemma [J.5,](#page-45-1)*

$$E\|H_j(\tilde{X}W_0)H_j(\tilde{X}W_0)^\top\|_{op} \leq Cj!$$

*Proof of Lemma [J.7.](#page-45-2)* For non-centered Sub Gaussian random variable X with mean µ,

$$\begin{aligned} E(e^{(X-\mu)t}) &\leq e^{\frac{k^2}{2}t^2} \\ Ee^{Xt} &\leq e^{\frac{k^2}{2}t^2+\mu t} \end{aligned} \tag{47}$$

$$\begin{aligned}\mathbb{E} \exp(sz + tz') &\leq \exp\left(\frac{k^2}{2} \|u\|^2 s^2 + k^2 < \vec{u}, \vec{v} > st + \frac{k^2}{2} \|v\|^2 t^2\right) \\ &\leq \exp\left(\frac{k^2}{2} (s^2 + 2\rho st + t^2)\right)\end{aligned}$$

2536

2539

2540 2541 2542 For µ ̸= 0 case, considering non-centered Sub Gaussian Random vector g with mean µ and centered Sub Gaussian Random vector ξ s.t. g = ξ + µ. We use proof techniques similar to those in [Theorem M.11.](#page-55-0)

2543 2544 Denote ν = min(j, k). Considering u <sup>⊤</sup>g, v⊤g,

2554

2556

2558 2559

2560 *Proof of Lemma [J.5.](#page-45-1)* Let A = H<sup>j</sup> (XW˜ <sup>0</sup>), then

2564

2566

2569

2574

2576 Let M = E max<sup>i</sup> ||H<sup>j</sup> (W0x˜i)||<sup>2</sup> and δ = C q M log n N . Moreover, we note that ||x˜i||<sup>2</sup><sup>j</sup> N is sub-weibull random variable and bound of [\(Kuchibhotla & Chakrabortty,](#page-10-18) [2022\)](#page-10-18) proposition A.6 can be applied.

2579 Use property of ||x˜i||<sup>2</sup><sup>j</sup> N , W<sup>0</sup> and hermite polynomials, we have

2581 2582

Dividing by exp( <sup>k</sup> (s <sup>2</sup> + t 2 )), then

$$E[\exp(sz - \frac{k^2}{2}s^2) \exp(tz' - \frac{k^2}{2}t^2)] \leq \exp(\rho st) = \sum_{j=0}^{\infty} \frac{\rho^j}{j!} s^j t^j$$

Using proof techniques similar to those in Lemma [M.1,](#page-52-1) one can acquire

$$EH_j(u^T g)H_k(v^T g) \leq j!\rho^j \mathbf{1}_{\mathbf{j}=k} \quad (48)$$

$$\begin{aligned} & \mathbb{E}[H_j(u^T \mu + u^T \xi) H_k(v^T \mu + v^T \xi)] \\ &= \mathbb{E}\left[\left\{\sum_{i=0}^j \binom{j}{i} (u^T \mu)^i H_{j-i}(u^T \xi)\right\} \cdot \left\{\sum_{h=0}^k \binom{k}{h} (v^T \mu)^h H_{k-h}(v^T \xi)\right\}\right] \\ &= \mathbb{E}\left[\sum_{q=0}^{\nu} \binom{\nu}{q} (u^T \mu)^{j-q} (v^T \mu)^{k-q} H_q(u^T \xi) H_q(v^T \xi)\right] \text{ by equation 48} \\ &\leq \sum_{q=0}^{\nu} \binom{\nu}{q} (u^T \mu)^{j-q} (v^T \mu)^{k-q} \cdot \nu! \rho^{\nu} \\ &\leq C \min(j, k)! \end{aligned} \quad (49)$$

$$\begin{aligned} \mathbb{P}(\|A\|_{op} \geq t) &\leq \mathbb{P}\left(\left\|\frac{1}{n}AA^T - EAA^T\|_{op} \geq \frac{t^2}{n} - \|EAA^T\|_{op}\right\|\right) \quad (\text{by Lemma J.6}) \\ &\leq \frac{1}{\frac{t^2}{n} - \|EAA^T\|_{op}} E\left[\left\|\frac{1}{n}AA^T - EAA^T\|_{op}\right\|\right] \quad (\text{by Markov's inequality}) \\ &\leq \left[\frac{t^2}{n} - E[\|AA^T\|_{op}]\right]^{-1} \delta \max\left(\sqrt{\|EAA^T\|_{op}}, \delta\right) \quad (\text{by Theorem 5.48 in Vershynin (2010)}) \\ &\leq \left[\frac{t^2}{n} - E[\|AA^T\|_{op}]\right]^{-1} \delta \max\left(\sqrt{E[\|AA^T\|_{op}]}, \delta\right) \quad (\text{by Jensen's inequality}). \end{aligned}$$

$$M \leq c_j E \max_i \|(W_0 \tilde{x}_i)^{\circ j}\|_2^2 \leq c_j E \max_i \|x\|^{2j} \leq c_j N(\log n)^{\frac{1}{2}}.$$

$$\begin{aligned}
& \left[ \frac{t^2}{n} - E[\|AA^T\|_{op}] \right]^{-1} \delta \max \left( \sqrt{E[\|AA^T\|_{op}]}, \delta \right) \\
& \leq \left[ \frac{t^2}{n} - E[\|AA^T\|_{op}]^{-1} C \log n \max(\sqrt{E[\|AA^T\|_{op}]}, \log n) \right. \\
& = [E[\|AA^T\|_{op}(Q_n - 1)]^{-1} C \log n \max(\sqrt{E[\|AA^T\|_{op}]}, \log n) \\
& \leq C \frac{\log n \max(\sqrt{E[\|AA^T\|_{op}]}, \log n)}{E[\|AA^T\|_{op} Q_n}
\end{aligned} \tag{50}$$
2595

2596

2614

2616

2618 2619

2624

2626

2629 Finally, we proof [Theorem 3.3.](#page-2-7)

2634

2636

above, we can continue expanding the expression as follows:

Choosing Q<sup>n</sup> = log<sup>3</sup> n, and using Lemma [J.7,](#page-45-2) we conclude the proof.

*Fact* J.8*.* For any vector u, v and any matrix A, B

**A.** 
$$\|uv^T\|_{op} = \|u\|_2 \|v\|_2$$

B. ||u||<sup>∞</sup> ≤ ||u||<sup>2</sup> ≤

√ n||u||<sup>∞</sup>

C. ||u ◦k

|| ≤ ||u||<sup>k</sup>

D. ||u ◦k ||<sup>2</sup> ≤ √ n||u ◦k ||<sup>∞</sup> ≤ √

n maxi(|u

k i <sup>|</sup>) = √

n(max<sup>i</sup>

|u<sup>i</sup> |) <sup>k</sup> = √ n||u||<sup>k</sup> ∞

E. Schur product theorem

$$\|A \circ B\|_{op} = \sup_{\|x\|=1} \text{tr}(A^T \text{diag}(x) B \text{diag}(x)) \leq \|A\|_{op} \cdot \|B\|_{op}$$

Next, let L = O(log n).

Denote <sup>σ</sup>L(z) = P<sup>L</sup> <sup>k</sup>=1 ckHk(z), F <sup>L</sup> = σL(XW˜ ) and F L <sup>0</sup> <sup>=</sup> <sup>σ</sup>L(XW˜ <sup>0</sup>).

Then, F = F <sup>L</sup> + (σ − σL)(XW˜ ).

Using Lemma [J.5,](#page-45-1) w in assumption [2.1,](#page-2-5) w.p. 1 − o(1)

$$\begin{aligned} & ||E[(\sigma - \sigma_L)(W_0 X)(\sigma - \sigma_L)(W_0 X)^T]|| \\ & \leq C \sum_{k=L+1}^{\infty} k! c_k^2 \leq C \sum_{k=L+1}^{\infty} k^{-3-w} \leq C \int_L^{\infty} k^{-\frac{3}{2}-w} dk \leq C L^{-2-w}. \end{aligned} \quad (51)$$

Therefore, following same proof technique as Lemma [J.5,](#page-45-1) [J.6,](#page-45-0) [J.7,](#page-45-2)

$$||(\sigma - \sigma_L)(\tilde{X}W_0)||_{op} = o_{\mathbb{P}}(\sqrt{n \log^3 n} \cdot L^{-2-w}) = o_{\mathbb{P}}(\sqrt{n}) \quad (52)$$

Also, because ||W||op = O(1),

$$||(\sigma - \sigma_L)(\tilde{X}W)||_{op} = o(\sqrt{n \log^3 n} \cdot L^{-2-w}) = o_{\mathbb{P}}(\sqrt{n}) \quad (53)$$

*Proof of [Theorem 3.3.](#page-2-7)* We write F <sup>L</sup> + F L <sup>0</sup> = F <sup>L</sup> + F L 0 , then F <sup>L</sup> = F L <sup>0</sup> + P<sup>L</sup> <sup>k</sup>=1 <sup>c</sup>k(Hk(XW˜ ) <sup>−</sup> <sup>H</sup>k(XW˜ <sup>0</sup>)). We have to study Hk(XW˜ ) − Hk(XW˜ <sup>0</sup>) term.

$$\begin{aligned}
& H_k(\tilde{X}W) - H_k(\tilde{X}W_0) \\
&= H_k(\tilde{X}W_0^T + \tilde{X}G^T) - H_k(\tilde{X}W_0) \\
&= (\tilde{X}G)^{ok} + \sum_{j=1}^{k-1} \binom{k}{j} H_{k-j}(\tilde{X}W_0) \circ (\tilde{X}G)^{oj}
\end{aligned} \tag{54}$$

2654

2656

2665 2666 ∥F L <sup>0</sup> <sup>∥</sup> = Θ(√ n) by [Moniri et al.](#page-10-5) [\(2024\)](#page-10-5).

2667

2668 2669 For ∆1, ∆2, ∆3, it is derived as follows

2674

2676

2679

2689 2690

Thus,

$$\begin{aligned} F^L &= F_0^L + \sum_{k=1}^L c_k (\tilde{X}G)^{ok} + \sum_{k=1}^L \sum_{j=1}^{k-1} c_k \binom{k}{j} H_{k-j}(XW_0) \circ (\tilde{X}G)^{\circ j} \\ &= F_0^L + \sum_{k=1}^L c_1^k c_k (\tilde{X} \sum_{i < j} \beta_{ij} a_{ij}^T)^{ok} \\ \Delta_1 & \left[ - \sum_{k=1}^L c_1^k c_k (\tilde{X} \sum_{i < j} \beta_{ij} a_{ij}^T)^{ok} \right. \\ & \left. + \sum_{k=1}^L c_k (\tilde{X}G)^{ok} \right] \quad (55) \\ \Delta_2 & \left[ + \sum_{k=1}^L \sum_{j=1}^{k-1} c_k \binom{k}{j} H_{k-j}(\tilde{X}W_0) \circ (\tilde{X}G)^{\circ j} \right. \\ & \left. - \sum_{k=1}^L \sum_{j=1}^{k-1} c_1^j c_k \binom{k}{j} H_{k-j}(\tilde{X}W_0) \circ [\tilde{X} \sum_{i < j} \beta_{ij} a_{ij}^T]^{\circ j} \right] \\ \Delta_3 & \left[ + \sum_{k=1}^L \sum_{j=1}^{k-1} c_1^j c_k \binom{k}{j} H_{k-j}(\tilde{X}W_0) \circ [\tilde{X} \sum_{i < j} \beta_{ij} a_{ij}^T]^{\circ j} \right] \end{aligned}$$

P<sup>L</sup> <sup>k</sup>=1 c k 1 ck(X˜ P i<j βija T ij ) ok<sup>∥</sup> is bigger than √ n.

$$\begin{aligned} \|\Delta_1\| &\leq \sum_{k=1}^L c_k \|(\tilde{X}G)^{ok} - c_1^k (\tilde{X} \sum_{i$$

$$\begin{aligned} \|\Delta_2\| &\leq \sum_{k=1}^L \sum_{j=1}^{k-1} c_k \binom{k}{j} \|H_{k-j}(\tilde{X}W_0^T) \circ [(\tilde{X}G^T)^{\circ j} - c_1^j[\tilde{X} \sum_{i$$

2699 2700

2704

2706

2709

2714

2716

2718 2719

2724

2726

2729

2734

2736

$$\begin{aligned}
\|\Delta_3\| &\leq C \sum_{k=1}^L \sum_{j=1}^{k-1} \|H_{k-j}(\tilde{X}W_0) \circ [\tilde{X} \sum_{i < j} \beta_{ij} a_{ij}^T]^{\circ j}\| \\
&\leq C \sum_{k=1}^L \sum_{j=1}^{k-1} \|\text{diag}(\tilde{X}\beta)^{\circ j}\| \|H_{k-j}(\tilde{X}W_0)\| \|\text{diag}(a)^{\circ j}\| \\
&\leq C \sum_{k=1}^L \sum_{j=1}^{k-1} (M_a M_b)^j \|H_{k-j}(\tilde{X}W_0)\| \\
&\leq C \sum_{k=1}^L \sum_{j=1}^{k-1} n^{-\frac{1}{2}j} \log^j n \sqrt{n \log^{\frac{3}{2}}} = O(\log^{\frac{5}{2}} n)
\end{aligned} \tag{58}$$

Therefore, we conclude the proof.

# K. Proof of Clustering Risk Analysis in two-classes case

Definition K.1. Given N, d, let

$$\begin{aligned}
S_{d,k}^{(1)} &= \mathbb{E}_{w \sim Unif(\mathbb{S}^{d-1})}[(w^T e_1)^k] \in \mathbb{R}_+ \\
S_{d,k,k'}^{(2)} &= E_w[(w^T \hat{\mu}_1)^k (w^T \hat{\mu}_2)^{k'}] \\
\rho_{k,k'}^{(1)} &= NS_{d,k+k'}^{(1)} \mathbf{1}_{k+k'} \text{ is even} \in \mathbb{R}_+ \\
\rho_{k,k'}^{(2)}(\cos(\mu_1, \mu_2)) &= NS_{d,k,k'}^{(2)} \mathbf{1}_{k+k'} \text{ is even} \in \mathbb{R}_+ \\
\rho_{k,k',r}^{(3)} &= \frac{c_1^k S_{d,k'}^{(1)}}{N^{\frac{k}{2}-1}} \binom{k}{r} (r-1)!!(k-1)!! \mathbf{1}_{k,k',r} \text{ is even} \in \mathbb{R}_+ \\
\rho_{k,k',r,r'}^{(4)} &= \frac{2c_1^{k+k'} S_{d,k}^{(1)}}{N^{\frac{k'}{2}-1}} \binom{k'}{r'} (r'-1)!!(k'-1)!! \mathbf{1}_{k,k',r'} \text{ is even} \in \mathbb{R}_+
\end{aligned} \tag{59}$$

For S (2) d,k,k′ , it depends on cos(µ1, µ2). As cos(µ1, µ2) increases, S (2) d,k,k′ grows, while it decreases as cos(µ1, µ2) decreases. e.g. when µ<sup>1</sup> = µ2, S (2) d,k,k′ = S (1) d,k+k′ , and when µ<sup>1</sup> = −µ<sup>2</sup> = −S (1) d,k+k′ .

Lemma K.2. *Let* Cd,k ≜ <sup>E</sup>ω[(ω <sup>⊤</sup>e1) k ] s.t. ω ∼ *Unif*(S d−1 )*, then*

$$\mathbb{E}_{\omega}[(\omega^{\top} \mu)^k] = \|\mu\|^k S_{d,k}^{(1)} \mathbf{1}_k \text{ is even} \quad (60)$$

*Proof of [K.2.](#page-49-2)* The uniform distribution on the sphere is origin-symmetric. Therefore, when k is odd, Expectation is zero. In the other case, also use isotropic property of uniform sphere,

$$E_\omega[(\omega^\top \mu)^k] = \|\mu\|^k E_\omega[(\omega^\top e_1)^k] = \|\mu\|^k S_{d,k}^{(1)}$$

In the proof below, we utilize the results of Corollary [M.12,](#page-56-0) Corollary [M.13,](#page-56-1) and Lemma [K.2.](#page-49-2)

Lemma K.3. *Given vector* a ∈ R <sup>N</sup> β ∈ <sup>R</sup> <sup>d</sup> *and Gaussian Random vector* x ∼ N(µ, I)*. Let* b = x <sup>⊤</sup>β ∼ N(µ <sup>⊤</sup>β, ∥β∥ 2 )*, then*

$$\mathbb{E}_x(x^\top \beta a^\top)^{\circ k} = \sum_{r=0}^k \binom{k}{r} (\mu^\top \beta)^{k-r} \|\beta\|^r (r-1)! \mathbf{1}_{\mathbf{1}_r \text{ is even}} a^{\circ k\top} \quad (61)$$

$$\mathbb{E}_a a^{\circ k} = \frac{(k-1)!! \mathbb{1}_{k \text{ is even}}}{N^{\frac{k}{2}}} \mathbb{1}$$

$$\mathbb{E}_a a^{ok'} a^{ok'} = \frac{(k + k' - 1)!! 1_{k+k'} \text{ is even}}{N^{\frac{k+k'}{2}-1}} \quad (63)$$

2756

2759 2760

2764

2766

2769

2774 Calculate coh<sup>1</sup>

2776

2779

2789 2790

2794

2796

2799 2800

*Proof.* This follows directly from Corollary [M.12.](#page-56-0)

*Proof of Proposition [4.5.](#page-3-5)* Let *cohesion* of initialized feature as

$$coh_0 = \mathbb{E}_{W_0}[\mathbb{E}_{x \sim c_1} F_0^L(x)^T \mathbb{E}_{x' \sim c_1} F_0^L(x')] \quad (64)$$

Let *cohesion* of feature after training as

$$coh_1 = \mathbb{E}_{W_{0,a}}[\mathbb{E}_{x \sim c_1} F_L(x)^T \mathbb{E}_{x' \sim c_1} F_L(x')] \quad (65)$$

Calculate coh<sup>0</sup> By Lemma [K.2,](#page-49-2)

$$\begin{aligned}
coh_0 &= \mathbb{E}_{W_0} [\mathbb{E}_{x \sim c_1} [\sum_{k=1}^L c_k H_k(W_0^T x)]^T \mathbb{E}_{x' \sim c_1} [\sum_{k'=1}^L c_{k'} H_{k'}(W_0^T x)]] \\
&= \sum_{k=1, k'=1}^L c_k c_{k'} \mathbb{E}_{W_0} [\sum_{q=1}^N (W_0[q]^T \mu_1)^{k+k'}] \\
&= N \sum_{k=1, k'=1}^L c_k c_{k'} (\|\mu_1\|^{k+k'} S_{d, k+k'}^{(1)}) \mathbf{1}_{(k+k') \text{ even}} \\
&= \sum_{k=1, k'=1}^L c_k c_{k'} \rho_{k, k'}^{(1)} \|\mu\|^{k+k'}
\end{aligned} \tag{66}$$

$$\begin{aligned}
coh_1 &= \mathbb{E}_{W_0, a} [\mathbb{E}_{x \sim c_1} [\sum_{k=1}^L (c_k H_k(W_0^T x) + c_k c_1^k (x^T \beta a)^{ok'})^T \mathbb{E}_{x' \sim c_1} [\sum_{k'=1}^L (c_{k'} H_{k'}(W_0^T x) + c_1^k (x^T \beta a)^{ok'}]]] \\
&= \mathbb{E}_{W_0, a} [\sum_{k, k'=1}^L c_k c_{k'} [\mathbb{E}_x H_k(W_0^T x)^T \mathbb{E}_{x'} H_{k'}(W_0^T x') \\
&\quad + 2 \mathbb{E}_x H_k(W_0^T x)^T \mathbb{E}_{x'} c_1^{k'} (x^T \beta a)^{ok'} + c_1^{k+k'} \mathbb{E}_x (x^T \beta a)^{ok'} \mathbb{E}_{x'} (x'^T \beta a)^{ok'}]] \\
&= coh_0 + 2 \sum_{k, k'=1}^L c_k c_{k'} c_1^{k'} \mathbb{E}_{W_0} \mathbb{E}_x H_k(W_0^T x)^T \mathbb{E}_a \mathbb{E}_{x'} (x'^\top \beta a)^{ok'} \\
&\quad + \sum_{k, k'=1}^L c_k c_{k'} c_1^{k+k'} \mathbb{E}_a [\mathbb{E}_x (x^\top \beta a)^{ok'} \mathbb{E}_{x'} (x'^\top \beta a)^{ok}] \\
&= coh_0 + 2N \sum_{k, k'=1}^L c_k c_{k'} c_1^{k'} (\|\mu_1\|^k S_{d,k}^{(1)}) \left( \frac{1}{N^{\frac{k}{2}}} \sum_{r'=0}^{k'} \binom{k'}{r'} (\mu_1^T \beta)^{k'-r'} \|\beta\|^{r'} (r' - 1) !! (k' - 1) !! \mathbf{1}_{k,k'}, r' \text{ is even} \right. \\
&\quad \left. + \sum_{k, k'=1}^L \frac{c_k c_{k'} c_1^{k+k'}}{N^{\frac{k+k'}{2}}} \sum_{r=0}^k \sum_{r'=0}^{k'} \binom{k}{r} \binom{k'}{r'} (\mu_1^T \beta)^{k+k'-r-r'} \|\beta\|^{r+r'} (r - 1) !! (r' - 1) !! \mathbf{1}_{k+k'}, r, r' \text{ is even} \right)
\end{aligned}$$

*Proof of Proposition [4.6.](#page-3-6)* Let *separability* of initialized feature as

$$sep_0 = -\mathbb{E}_{W_0}[\mathbb{E}_{x \sim c_1} F_0^L(x)^T \mathbb{E}_{x' \sim c_2} F_0^L(x')] \quad (67)$$

Let *separability* of feature after training as

$$sep_1 = -\mathbb{E}_{W_{0,a}}[\mathbb{E}_{x \sim C_1} F_L(x)^T \mathbb{E}_{x' \sim C_2} F_L(x')] \quad (68)$$

Calculate sep<sup>0</sup> By Lemma [K.2,](#page-49-2)

$$\begin{aligned}
sep_0 &= - \sum_{k=1, k'=1}^L c_k c_{k'} \mathbb{E}_{W_0} \left[ \sum_{q=1}^N (W_0[q]^T \mu_1)^k (W_0[q]^T \mu_2)^{k'} \right] \\
&= -N \sum_{k=1, k'=1}^L c_k c_{k'} \mathbb{E}_{w \sim Unif(\mathbb{S}^{d-1})} [(w^T \mu_1)^k (w^T \mu_2)^{k'}] \\
&= -N \sum_{k=1, k'=1}^L c_k c_{k'} \|\mu_1\|^k \|\mu_2\|^{k'} E_w [(w^T \hat{\mu}_1)^k (w^T \hat{\mu}_2)^{k'}] \\
&= -N \sum_{k=1, k'=1}^L c_k c_{k'} \|\mu_1\|^k \|\mu_2\|^{k'} S_{d, k, k'}^{(2)} \mathbf{1}_{k+k'} \text{ is even} \\
&= - \sum_{k=1, k'=1}^L c_k c_{k'} \|\mu_1\|^k \|\mu_2\|^{k'} \rho_{k, k'}^{(1)}
\end{aligned} \tag{69}$$

Calculate sep<sup>1</sup>

$$\begin{aligned}
sep_1 = & - \sum_{k,k'=1}^L c_k c_{k'} \mathbb{E}_{W_0,a} \left[ \mathbb{E}_{x \sim c_1} H_k(W_0^T x)^T \mathbb{E}_{x' \sim c_2} H_{k'}(W_0^T x') \right. \\
& + \mathbb{E}_{x \sim c_1} H_k(W_0^T x)^T \mathbb{E}_{x' \sim c_2} c_1^{k'} (x'^T \beta a)^{ok'} \\
& + \mathbb{E}_{x \sim c_1} c_1^k (x^T \beta a)^{ok'} \mathbb{E}_{x' \sim c_2} H_{k'}(W_0^T x) \\
& \left. + c_1^{k+k'} \mathbb{E}_{x \sim c_1} (x^T \beta a)^{ok'} \mathbb{E}_{x' \sim c_2} (x'^T \beta a)^{ok'} \right] \\
= & sep_0 - \sum_{k,k'=1}^L c_k c_{k'} \\
& \left[ c_1^{k'} (\|\mu_1\|^k S_{d,k}^{(1)}) \frac{1}{N^{\frac{k'}{2}-1}} \sum_{r'=0}^{k'} \binom{k'}{r'} (\mu_2^T \beta)^{k'-r'} \|\beta\|^{r'} (r'-1)!! (k'-1)!! \mathbf{1}_{k,k',r'} \text{ is even} \right. \\
& + c_1^k (\|\mu_2\|^{k'} S_{d,k'}^{(1)}) \frac{1}{N^{\frac{k}{2}-1}} \sum_{r=0}^k \binom{k}{r} (\mu_1^T \beta)^{k-r} \|\beta\|^r (r-1)!! (k-1)!! \mathbf{1}_{k,r,k'} \text{ is even} \\
& + c_1^{k+k'} \sum_{r=0}^k \sum_{r'=0}^k \binom{k}{r} \binom{k'}{r'} (\mu_1^T \beta)^{k-r} (\mu_2^T \beta)^{k'-r'} \|\beta\|^{r+r'} (r-1)!! (r'-1)!! \\
& \left. \frac{1}{N^{\frac{k+k'}{2}-1}} (k+k'-1)!! \mathbf{1}_{k+k',r,r'} \text{ is even} \right]
\end{aligned}$$

# L. Additional Lemmas of Sub-Gaussian Distribution

For more detailed explanation and well known results of Sub-Gaussian we used, please refer to [Vershynin](#page-11-15) [\(2010;](#page-11-15) [2018\)](#page-11-5). We show below that the truncated Gaussian distribution, utilized in our synthetic data experiments, is a sub-Gaussian distribution.

Lemma L.1. *Truncated Gaussian distribution which have support on* (a, b) s.t. a, b ∈ (−∞, ∞) *is Sub-Gaussian.*

*Proof.* Denote N(a,b)(0, σ<sup>2</sup> ) is Truncated Gaussian distribution which have support on (a, b) s.t. a, b ∈ (−∞, ∞). support (N(a,b)(0, σ<sup>2</sup> )) ⊂ <sup>R</sup> d . Therefore, <sup>P</sup>(|X| ≥ t) s.t. X ∼ N(a,b)(0, σ<sup>2</sup> ) have same tail behavior with Gaussian and Gaussian is Sub-Gaussian.

## L.1. Generalization of centered Sub-Gaussian results toward non-centered

We verify below that the results on centered sub-Gaussian distributions from [Vershynin](#page-11-5) [\(2018\)](#page-11-5) can be extended to noncentered sub-Gaussian distributions.

Lemma L.2. *Sum of non-centered Sub-Gaussian random variable is Sub-Gaussian.*

*Proof.* If the Orlicz 2 norm is bounded ||X||<sup>ψ</sup><sup>2</sup> < ∞, then X is Sub-Gaussian. Also, ||EX||<sup>ψ</sup><sup>2</sup> ≤ C||X||<sup>ψ</sup><sup>2</sup> , and Sum of centered Sub-Gaussian random variable is Sub-Gaussian. We show ||PX<sup>i</sup> ||<sup>ψ</sup><sup>2</sup> < ∞, s.t. X is non-centered Sub-Gaussian.

$$\begin{aligned} \| \sum X_i \|_{\psi_2} &\leq \| \sum (X_i - \mathbb{E}X_i) \|_{\psi_2} + \| \sum \mathbb{E}X_i \|_{\psi_2} \\ &\leq \| \sum (X_i - \mathbb{E}X_i) \|_{\psi_2} + \sum \| \mathbb{E}X_i \|_{\psi_2} \\ &\leq \| \sum (X_i - \mathbb{E}X_i) \|_{\psi_2} + C \sum \| X_i \|_{\psi_2} < \infty \end{aligned} \quad (70)$$

Lemma L.3. *(Operator norm bound for non-centered Sub-Gaussian matrix, generalization of 4.4.5 in [Vershynin](#page-11-5) [\(2018\)](#page-11-5)) let* A ∈ R <sup>m</sup>×n*,* A[i][j] *is independent, non-centered Sub-Gaussian.* ∀t > 0*,*

$$\begin{aligned}\|A\| &\leq CK(\sqrt{m} + \sqrt{n} + t) \text{ w.p. } 1 - \exp(-t^2) \\ \text{Alternatively, } \|A\| &\leq CK(\sqrt{m + n} + t) \text{ w.p. } 1 - \exp(-t^2)\end{aligned}\tag{71}$$

K = maxi,j ||A[i][j]||<sup>ψ</sup><sup>2</sup>

Lemma L.4. *(Expectation of operator norm for non-centered Sub-Gaussian matrix generalization of 4.4.6 in [Vershynin](#page-11-5) [\(2018\)](#page-11-5))*

$$\begin{aligned}\mathbb{E}\|A\| &\leq CK(\sqrt{m} + \sqrt{n}) \\ \text{Alternatively, } \mathbb{E}\|A\| &\leq CK(\sqrt{m + n}), \quad \text{and, } \mathbb{E}\|A\|^2 \leq C(m + n)\end{aligned}\tag{72}$$

*Proof of Lemma [L.3](#page-52-0) and Lemma [L.4.](#page-52-2)* Based on the result of Lemma [L.2,](#page-51-1) one can follow the same proof process of [Vershynin](#page-11-5) [\(2018\)](#page-11-5)

# M. Additional Results of Expectation of Hermite Polynomials

The non standard gaussian expectation of the product of two Hermite polynomials is computed as follows. It is an generalization of results of standard Gaussian distributions in [O'Donnell](#page-10-19) [\(2021\)](#page-10-19); [Moniri et al.](#page-10-5) [\(2024\)](#page-10-5) into a generalized multivariate Gaussian. We start with previously known facts, and derive our generalized results. These findings provide a useful analysis tool for Hermite polynomials, and may offer a foundation for broader applications in future works involving nonlinear activations decomposable into Hermite polynomials under the assumption of a multivariate Gaussian distribution.

## M.1. Expectation of a product of two Hermite polynomials

Here is the result of the expectation of the product of two Hermite polynomials, utilizing the orthogonality of Hermite polynomials.

Lemma M.1 (Orthogonality of Hermite polynomials from Lemma C.1 [Moniri et al.](#page-10-5) [\(2024\)](#page-10-5)). *See also derivation in Chapter 11.2 [O'Donnell](#page-10-19) [\(2021\)](#page-10-19).*

*Let* (Z1, Z2) *be jointly Gaussian with* <sup>E</sup>[Z1] = <sup>E</sup>[Z2] = 0*,* <sup>E</sup>[Z 2 1 ] = <sup>E</sup>[Z 2 2 ] = 1*, and* <sup>E</sup>[Z1Z2] = ρ*. Then for any* k1, k<sup>2</sup> ∈ {0, 1, · · · , }

$$\mathbb{E}[H_{k_1}(Z_1)H_{k_2}(Z_2)] = k_1!\rho^{k_1}\mathbf{1}_{k_1=k_2}$$

*In the other form, for* d ∈ <sup>N</sup>*,* Z ∼ N(0, Id)*,* a, b ∈ <sup>S</sup> d−1

*,*

$$\mathbb{E}[H_{k_1}(Z^\top a)H_{k_2}(Z^\top b)] = k_1!(a^\top b)^{k_1} \mathbf{1}_{k_1=k_2}$$

*Fact* M.2*.* Let W ∈ R <sup>d</sup>×<sup>N</sup> s.t. ∀i W[i] ∈ <sup>S</sup> d−1 . For Z ∼ N(0, I),

$$\mathbb{E}_{Z \sim n(0,1)}[H_j(W^\top Z) H_k(W^\top Z)^\top] = k!(W^\top W)^{\circ j} \mathbf{1}_{\mathbf{j}=\mathbf{k}} \quad (73)$$

$$\mathbb{E}_{Z \sim n(0,1)}[H_j(W^\top Z)^\top H_k(W^\top Z)] = k! \sum \|W[i]\|^{2j} \mathbf{1}_{\mathbf{j}=\mathbf{k}} = k! N \mathbf{1}_{\mathbf{j}=\mathbf{k}} \quad (74)$$

2918 2919

2924

2926

2929

2934 Additionally,

2936

2954

2956

2966 2967 *Proof of Theorem [M.5.](#page-53-1)* (Generalize Chapter 11.2 [O'Donnell](#page-10-19) [\(2021\)](#page-10-19)'s derivation to non unit variance)

*Proof.* We apply H<sup>j</sup> element-wise. By Lemma [M.1,](#page-52-1) we can acquire the above result.

The following remark presents a modified condition of Lemma [M.1](#page-52-1) for the case where a, b /∈ S d−1 in Lemma [M.1.](#page-52-1) In this case, the variances of Z <sup>⊤</sup>a and Z <sup>⊤</sup>b are not equal to 1, and the covariance may exceed the bounds [−1, 1]. Under this condition, we will compute the expectation of the product of two Hermite polynomials as in Lemma [M.1.](#page-52-1)

*Remark* M.3 (the modified condition of Lemma [M.1\)](#page-52-1)*.* For d ∈ N, u, v ∈ R d , Z ∼ N(0, Id),

Z<sup>1</sup> = ⟨u, Z⟩ ∼ N(0, ||u||<sup>2</sup> 2 ), Z<sup>2</sup> = ⟨v, Z⟩ ∼ N(0, ||v||<sup>2</sup> 2 ).

Then, Z1, Z<sup>2</sup> is ρ =≜ ⟨ u ||u|| , v ||v||⟩ - correlated

$$\begin{aligned} corr(Z_1, Z_2) &= \frac{\mathbb{E}[Z_1 Z_2]}{\sqrt{V(Z_1)} \sqrt{V(Z_2)}} = \frac{\mathbb{E}_Z \langle u, Z \rangle \langle v, Z \rangle}{\|u\| \|v\|} \\ &= \frac{\mathbb{E}_g \sum_i \sum_j u_i v_j Z_i Z_j}{\|u\| \|v\|} = \frac{\sum_i \sum_j u_i v_j \mathbb{E}_Z [Z_i Z_j]}{\|u\| \|v\|} \\ &= \frac{\langle u, v \rangle}{\|u\| \|v\|} \end{aligned} \tag{75}$$

$$\begin{pmatrix} Z_1 \\ Z_2 \end{pmatrix} \sim n \left( \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} \|u\|^2 & \langle u, v \rangle \\ \langle v, u \rangle & \|v\|^2 \end{pmatrix} \right) \quad (76)$$

We first introduce Isserlis' theorem, which is essential for the proof. This theorem allows the expectation of the product of centered Gaussian random variables to be expressed as a product of covariances, making the computation feasible.

Theorem M.4 (Isserlis' Theorem [\(Isserlis,](#page-9-21) [1918;](#page-9-21) [Vignat,](#page-11-16) [2011\)](#page-11-16)). *Let* X = (X1, · · · , Xd) *Gaussian random vector* s.t. <sup>E</sup>[X] = 0 *, and let* A = {α1, · · · , α<sup>N</sup> } *be set of integers* s.t. 1 ≤ α<sup>i</sup> ≤ d*,* ∀i*. Denote* X<sup>A</sup> = Q <sup>α</sup>i∈<sup>A</sup> X<sup>α</sup><sup>i</sup> *, and* X<sup>∅</sup> = 1*. Let* Q (A) *denote partitions of* A *into disjoint pairs and* σ ∈ Q (A) *is pair.*

$$\mathbb{E}[X_A] = \sum_{\sigma \in \prod(A)} \prod_{(i,j) \in \sigma} \mathbb{E}[X_{\alpha_i} X_{\alpha_j}] \mathbf{1}_{\text{d is even}}. \quad (77)$$

Now, we generalize the assumptions from the previous works so that Lemma [M.1](#page-52-1) holds for arbitrary vectors as Remark [M.3.](#page-53-0) This could allow the weights of the networks to become analyzable when they go beyond the assumption of lying on the unit spheres.

Theorem M.5 (Generalization of Lemma [M.1](#page-52-1) for centered Gaussian distribution). *For* d ∈ N*,* u, v ∈ R d *,* g ∼ N(0, Id)*,* ⟨u, g⟩ ∼ N(0, ||u||<sup>2</sup> 2 )*,* ⟨v, g⟩ ∼ N(0, ||v||<sup>2</sup> 2 )*.*

$$\begin{aligned} & \mathbb{E}_g[H_j(u^\top g)H_k(v^\top g)] \\ &= \frac{j!\langle u, v \rangle^j}{||u||^2||v||^2} \mathbf{1}_{j=k} - \frac{(||u||^2 - 1)(||v||^2 - 1)}{||u||^2||v||^2} \mathbb{E}_g[(v^\top g)^k(u^\top g)^j] \\ &+ \frac{(||v||^2 - 1)}{||v||^2} \mathbb{E}_g[H_j(u^\top g)(v^\top g)^k] + \frac{(||u||^2 - 1)}{||u||^2} \mathbb{E}_g[H_k(v^\top g)(u^\top g)^j] \end{aligned} \tag{78}$$

*Remark* M.6*.* The same results can be derived as in Lemma [M.1](#page-52-1) when the variance is 1 in Thm. [M.5.](#page-53-1)

*2973 2974*

*2979*

*2984*

*2989 2990*

*2994*

*2996*

*2999 3000*

*3001 3002 3003* By using the fact that exp(⟨u, v⟩st) = <sup>E</sup>g[exp(su⊤g − 1 2 s 2 ||u||<sup>2</sup> ) exp(tv⊤g − t 2 ||v||<sup>2</sup> )], we can eliminate the different orders of s t by a Taylor expansion and equating all monomials of the resulting polynomials.

*3014*

*3016*

*3019*

First, we study about <sup>E</sup>g∼N(0,σ<sup>2</sup>) [e tg] in order to analysis non unit variance case.

$$\begin{aligned}\mathbb{E}_{g \sim n(0, \sigma^2)}[e^{tg}] &= \frac{1}{\sqrt{2\pi}\sigma} \int e^{tg} e^{-\frac{g^2}{2\sigma^2}} dg \\ &= \frac{1}{\sqrt{2\pi}\sigma} e^{\frac{1}{2}t^2} \int \exp\left(-\frac{(g - \sigma^2 t)^2}{2\sigma^2}\right) \text{ complete square} \\ &= e^{\frac{1}{2}t^2}\end{aligned}\tag{79}$$

<sup>E</sup>Z,Z′ [exp(sZ + tZ′ )] study

Studying <sup>E</sup>Z,Z′ [exp(sZ + tZ′ )], we can derive what we need to show.

$$\begin{aligned} \mathbb{E}_{Z,Z'}[\exp(sZ + tZ')] &= \mathbb{E}_{g \sim n(0,1)}[\exp(s\langle u, g \rangle) + \exp(t\langle v, g \rangle)] \\ &= \prod_i \mathbb{E}_{g \sim n(0,1)}[\exp((su_i + tv_i)g_i)] && \text{Use equation 79} \\ &= \prod_i \exp(\frac{1}{2}(su_i + tv_i)^2) = \prod_i \exp(\frac{1}{2}s^2\|u\|^2 + \langle u, v \rangle st + \frac{1}{2}t^2\|v\|^2) \end{aligned} \tag{80}$$

Therefore,

$$\exp(\langle u, v \rangle st) = \mathbb{E}_g[\exp(su^\top g - \frac{1}{2}s^2\|u\|^2) \exp(tv^\top g - \frac{1}{2}t^2\|v\|^2)].$$

*Fact* M.7*.* One can verify below propositions with simple calculations.

Let P<sup>j</sup> (z) + z <sup>j</sup> = H<sup>j</sup> (z), C<sup>u</sup> = ||u||<sup>2</sup> − 1, a > 0.

Let f(s) = exp(sz − 1 2 s 2 ), ¯f(s) = exp(sz − 2 as<sup>2</sup> ), then

A. By Taylor expansion, exp(⟨u, v⟩st) = P<sup>∞</sup> j=0 j! ⟨u, v⟩ j s j t j .

B. By Taylor expansion, ¯f(s) = P<sup>∞</sup> j=0 j! ¯f (n) (0)s j

C. ¯f (n) (0) = Hn(z) + CuPn(z)

$$\begin{aligned} j!\langle u, v \rangle^j \mathbf{1}_{j=\mathbf{k}} &= \mathbb{E}_g \left[ (H_j(u^\top g) + P_j(u^\top g)C_u)(H_j(v^\top g) + P_j(v^\top g)C_v) \right] \\ &= \mathbb{E}_g \left[ (H_j(u^\top g) + (H_j(u^\top g) - (u^\top g)^j)C_u)(H_j(v^\top g) + (H_j(v^\top g) - (v^\top g)^j)C_v) \right] \\ &= \|u\|^2 \|v\|^2 \mathbb{E}_g [H_j(u^\top g) H_j(v^\top g)] + (\|u\|^2 - 1)(\|v\|^2 - 1) \mathbb{E}_g [(v^\top g)^j (u^\top g)^j] \\ &\quad - \|u\|^2 (\|v\|^2 - 1) \mathbb{E}_g [H_j(u^\top g) (v^\top g)^j] - \|v\|^2 (\|u\|^2 - 1) \mathbb{E}_g [H_j(v^\top g) (u^\top g)^j] \end{aligned} \tag{81}$$

Therefore,

$$\begin{aligned} & \mathbb{E}_g[H_j(u^\top g) H_j(v^\top g)] \\ &= \frac{j!\langle u, v \rangle^j}{\|u\|^2 \|v\|^2} \mathbf{1}_{j=k} - \frac{(\|u\|^2 - 1)(\|v\|^2 - 1)}{\|u\|^2 \|v\|^2} \mathbb{E}_g[(v^\top g)^j (u^\top g)^j] \\ &+ \frac{(\|v\|^2 - 1)}{\|v\|^2} \mathbb{E}_g[H_j(u^\top g)(v^\top g)^j] + \frac{(\|u\|^2 - 1)}{\|u\|^2} \mathbb{E}_g[H_j(v^\top g)(u^\top g)^j] \end{aligned} \tag{82}$$

Note that the result of Lemma [M.8](#page-54-1) can be applied for concrete calculation, and conclude the proof.

Lemma M.8. *For* d ∈ N*,* u, v ∈ R d *,* g ∼ N(0, Id)*,* Z¯ <sup>1</sup> = ⟨u, g⟩*,* Z¯ <sup>2</sup> = ⟨v, g⟩*.*

$$\begin{pmatrix} \bar{Z}_1 \\ \bar{Z}_2 \end{pmatrix} \sim n \left( \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} \|u\|^2 & \langle u, v \rangle \\ \langle v, u \rangle & \|v\|^2 \end{pmatrix} \right) \quad (83)$$

$$\begin{aligned}
 3029 \quad & \mathbb{E}_{\bar{Z}_1, \bar{Z}_2}[H_j(\bar{Z}_1)\bar{Z}_2^k] = j! \sum_{m=0}^{\lfloor \frac{1}{2} \rfloor} \frac{(-1)^m}{m!(j-2m)!2^m} \sum_{\sigma \in \prod(\{\bar{Z}_1\} \times j-2m) \cup \{\bar{Z}_2\} \times k\}} \prod_{(p,q) \in \sigma} \mathbb{E}[X_{\alpha_p} X_{\alpha_q}] \mathbf{1}_{j+k-2m \text{ is even}} \\
 3030 & \\
 3031 \quad & \mathbb{E}_{\bar{Z}_1, \bar{Z}_2}[\bar{Z}_1^j \bar{Z}_2^k] = \sum_{\sigma \in \prod(\{\bar{Z}_1\} \times j) \cup \{\bar{Z}_2\} \times k\}} \prod_{(p,q) \in \sigma} \mathbb{E}[X_{\alpha_p} X_{\alpha_q}] \mathbf{1}_{j+k \text{ is even}} \\
 3033 & \\
 & 
 \end{aligned}
 \tag{84}$$

3034

3036

3039 3040

3041 3042 3043 Therefore, we need to figure out <sup>E</sup>Z¯1,Z¯<sup>2</sup> [Z¯<sup>p</sup> 1Z¯<sup>q</sup> ]. We know Z¯ <sup>1</sup>,Z¯ <sup>2</sup> is mean zero Gaussian, so we can apply Thm. [M.4](#page-53-2) with A = {{Z¯ <sup>1</sup>} × p} ∪ {{Z¯ <sup>2</sup>} × q}}, <sup>E</sup>[Z¯<sup>p</sup> 1Z¯<sup>q</sup> 2 ] = P σ∈ Q(A) Q (τ,υ)∈<sup>σ</sup> <sup>E</sup>[X<sup>α</sup><sup>τ</sup> X<sup>α</sup><sup>υ</sup> ].1p+q is even

3044

3045 3046 Corollary M.9 (Corollary of Lemma [M.8\)](#page-54-1). *Remark* Z<sup>1</sup> ∼ N(0, ∥u∥ 2 ) *For the case* k = 0*,*

3047 3048

3049 *Proof.*

$$\begin{aligned}
& \mathbb{E}_{\bar{z}_1, \bar{z}_2}[\bar{Z}_1^j \bar{Z}_2^k] = \mathbb{E}_{\bar{z}_1}[\bar{Z}_1^j] = \sum_{\sigma \in \Pi(\{\bar{z}_1\} \times j)} \prod_{(p,q) \in \sigma} \mathbb{E}[X_{\alpha_p} X_{\alpha_q}] \mathbf{1}_{\mathbf{j} \text{ is even}} \\
&= \sum_{\sigma \in \Pi(\{\bar{z}_1\} \times j)} \prod_{(p,q) \in \sigma} \|u\|^2 \mathbf{1}_{\mathbf{j} \text{ is even}} = \sum_{\sigma \in \Pi(\{\bar{z}_1\} \times j)} \|u\|^j \mathbf{1}_{\mathbf{j} \text{ is even}} = (j-1)!! \|u\|^j \mathbf{1}_{\mathbf{j} \text{ is even}}
\end{aligned} \tag{87}$$

3056

3059 3060 We will change Theorem [M.5](#page-53-1) and Lemma [M.8](#page-54-1) to adopt a generalized Gaussian assumption with a mean of zero.

3061 3062 3063 Lemma M.10 (Taylor expansion of Hermite polynomials from Lemma C.2 [Moniri et al.](#page-10-5) [\(2024\)](#page-10-5)). *For any* k1, k<sup>2</sup> ∈ {0, 1, · · · , } *and* x, y ∈ <sup>R</sup>*,*

3064 3065

3066 3067 3068 Theorem M.11 (Generalization of Thm. [M.5](#page-53-1) for any Gaussian distribution). *For* d ∈ N*,* u, v ∈ R d *,* ξ ∼ N(0, 1)*,* g ∼ N(µ, Σ)*,* Z<sup>1</sup> = ⟨u, g⟩ ∼ N(µ <sup>⊤</sup>u, u⊤Σu)*,* Z<sup>2</sup> = ⟨v, g⟩ ∼ N(µ <sup>⊤</sup>v, v⊤Σv)*.*

$$\begin{aligned}
& \mathbb{E}_g[H_j(Z_1)H_k(Z_2)] \\
& = \sum_{\alpha=0}^j \sum_{\beta=0}^k \binom{j}{\alpha} \binom{k}{\beta} (u^\top \mu)^\alpha (v^\top \mu)^\beta \\
& \times \left[ \frac{(j-\alpha)!(u^\top \Sigma v)^{j-\alpha}}{u^\top \Sigma u v^\top \Sigma v} \mathbf{1}_{j-\alpha=k-\beta} - \frac{(u^\top \Sigma u - 1)(v^\top \Sigma v - 1)}{u^\top \Sigma u v^\top \Sigma v} \mathbb{E}_g[(\sqrt{u^\top \Sigma u} \xi)^{j-\alpha} (\sqrt{v^\top \Sigma v} \xi)^{k-\beta}] \right. \\
& \left. + \frac{(v^\top \Sigma v - 1)}{v^\top \Sigma v} \mathbb{E}_g[H_{j-\alpha}(\sqrt{u^\top \Sigma u} \xi)(\sqrt{v^\top \Sigma v} \xi)^{k-\beta}] + \frac{(u^\top \Sigma u - 1)}{u^\top \Sigma u} \mathbb{E}_g[(\sqrt{u^\top \Sigma u} \xi)^{j-\alpha} H_{k-\beta}(\sqrt{v^\top \Sigma v} \xi)] \right]
\end{aligned} \tag{89}$$

X<sup>α</sup><sup>i</sup> *is defined at Thm. [M.4](#page-53-2)*

*Proof.* By explicit formula of Hermite polynomials

$$\mathbb{E}_{\bar{Z}_1, \bar{Z}_2}[H_j(\bar{Z}_1)(\bar{Z}_2)^k] = j! \sum_{m=0}^{\lfloor \frac{j}{2} \rfloor} \frac{(-1)^m}{m!(j-2m)!2^m} \mathbb{E}_{\bar{Z}_1, \bar{Z}_2}[\bar{Z}_1^{j-2m} \bar{Z}_2^k] \quad (85)$$

$$\mathbb{E}_{\bar{Z}_1}[\bar{Z}_1^j] = \|u\|^j (j-1)!! \mathbf{1}_{j \text{ is even}} \quad (86)$$

### M.2. Expectation of a product of two Hermite polynomials—Generalization toward non-centered Gaussian

$$H_k(x + y) = \sum_{j=0}^k \binom{k}{j} x^j H_{k-j}(y). \quad (88)$$

*3089 3090*

*3094 3096* Use same proof technique Thm. [M.5,](#page-53-1) with √ √<sup>u</sup>⊤Σuξ v⊤Σvξ ∼ N 0 0 , u <sup>⊤</sup>Σu u⊤Σv v <sup>⊤</sup>Σu v⊤Σv !

*3099 3100*

*3104* In summary,

*3106*

*3109*

*3114*

*3116*

*3119*

*3124*

*3126*

*3129*

*Proof of Theorem [M.11.](#page-55-0)* By reparametrization i.e. Z<sup>1</sup> = √ u⊤Σuξ + u <sup>⊤</sup>µ, Z<sup>2</sup> = √ v⊤Σvξ + v <sup>⊤</sup>µ, and Lemma [M.10,](#page-55-1)

$$H_j(\sqrt{u^\top \Sigma} u \xi + u^\top \mu) = \sum_{\alpha=0}^j \binom{j}{\alpha} (u^\top \mu)^\alpha H_{j-\alpha}(\sqrt{\mu^\top \Sigma} u \xi). \quad (90)$$

$$\begin{aligned} \mathbb{E}_g[H_j(u^\top g)H_k(v^\top g)] &= \mathbb{E}_\xi[H_j(\sqrt{u^\top \Sigma u}\xi + u^\top \mu)H_k(\sqrt{v^\top \Sigma v}\xi + v^\top \mu)] \\ &= \mathbb{E}_\xi\left[\sum_{\alpha=0}^j \binom{j}{\alpha} (u^\top \mu)^\alpha H_{j-\alpha}(\sqrt{\mu^\top \Sigma u}\xi)\right] \left[\sum_{\beta=0}^k \binom{k}{\beta} (v^\top \mu)^\beta H_{k-\beta}(\sqrt{\mu^\top \Sigma v}\xi)\right] \\ &= \sum_{\alpha=0}^j \sum_{\beta=0}^k \binom{j}{\alpha} \binom{k}{\beta} (u^\top \mu)^\alpha (v^\top \mu)^\beta \mathbb{E}_\xi[H_{j-\alpha}(\sqrt{\mu^\top \Sigma u}\xi)H_{k-\beta}(\sqrt{\mu^\top \Sigma v}\xi)] \end{aligned} \quad (91)$$

$$\begin{aligned} & \mathbb{E}_\xi [H_{j-\alpha}(\sqrt{u^\top \Sigma u} \xi) H_{k-\beta}(\sqrt{v^\top \Sigma v} \xi)] \\ &= \frac{(j-\alpha)! u^\top \Sigma v)^{j-\alpha}}{u^\top \Sigma u v^\top \Sigma v} \mathbf{1}_{j-\alpha=k-\beta} - \frac{(u^\top \Sigma u - 1)(v^\top \Sigma v - 1)}{u^\top \Sigma u v^\top \Sigma v} \mathbb{E}_g [(\sqrt{u^\top \Sigma u} \xi)^{j-\alpha} (\sqrt{v^\top \Sigma v} \xi)^{k-\beta}] \\ &+ \frac{(v^\top \Sigma v - 1)}{v^\top \Sigma v} \mathbb{E}_g [H_{j-\alpha}(\sqrt{u^\top \Sigma u} \xi) (\sqrt{v^\top \Sigma v} \xi)^{k-\beta}] + \frac{(u^\top \Sigma u - 1)}{u^\top \Sigma u} \mathbb{E}_g [(\sqrt{u^\top \Sigma u} \xi)^{j-\alpha} H_{k-\beta}(\sqrt{v^\top \Sigma v} \xi)] \end{aligned} \quad (92)$$

$$\begin{aligned} & \mathbb{E}_g[H_j(u^\top g) H_k(v^\top g)] \\ &= \sum_{\alpha=0}^j \sum_{\beta=0}^k \binom{j}{\alpha} \binom{k}{\beta} (u^\top \mu)^\alpha (v^\top \mu)^\beta \\ &\times \left[ \frac{(j-\alpha)! (u^\top \Sigma u)^{j-\alpha}}{u^\top \Sigma u v^\top \Sigma v} \mathbf{1}_{j-\alpha=k-\beta} - \frac{(u^\top \Sigma u - 1)(v^\top \Sigma v - 1)}{u^\top \Sigma u v^\top \Sigma v} \mathbb{E}_\xi[(\sqrt{u^\top \Sigma u} \xi)^{j-\alpha} (\sqrt{v^\top \Sigma v} \xi)^{k-\beta}] \right. \\ &\left. + \frac{(v^\top \Sigma v - 1)}{v^\top \Sigma v} \mathbb{E}_\xi[H_{j-\alpha}(\sqrt{u^\top \Sigma u} \xi)(\sqrt{v^\top \Sigma v} \xi)^{k-\beta}] + \frac{(u^\top \Sigma u - 1)}{u^\top \Sigma u} \mathbb{E}_\xi[(\sqrt{u^\top \Sigma u} \xi)^{j-\alpha} H_{k-\beta}(\sqrt{v^\top \Sigma v} \xi)] \right] \end{aligned} \quad (93)$$

The following Corollary which calculates the Expectation of the Power of a Gaussian Random Variable can be derived using the binomial expansion with the reparametrization technique and Corollary [M.9.](#page-55-2) It corresponds to the case k = 0 in Lemma [M.8.](#page-54-1)

Corollary M.12 (Corollary of Lemma [M.8\)](#page-54-1). *Given* ω ∈ R d *, let Gaussian Random Variable* Z ∼ N(µ <sup>⊤</sup>ω, ∥ω∥ 2 )*, then*

$$\begin{aligned}\mathbb{E}_Z(Z)^k &= \sum_{t=0}^k \binom{k}{t} (\mu^\top \omega)^{k-t} \mathbb{E}_{\bar{Z} \sim \mathcal{N}(0, \|\omega\|^2)}[\bar{Z}^t] \\ &= \sum_{t=0}^k \binom{k}{t} (\mu^\top \omega)^{k-t} (t-1)! \cdot \|\omega\|^t \mathbf{1}_{\text{t is even}}.\end{aligned}\tag{94}$$

The following corollary, which computes the Gaussian expectation of Hermite polynomials, is derived from the explicit form of Hermite polynomials and Corollary [M.9.](#page-55-2) It corresponds to the case k = 0 in [Theorem M.11.](#page-55-0)

3154

3156

3159 3160 3161

3164 3165 3166

3169

3174

3176

3179

Corollary M.13. *Given* ω ∈ S d−1 *, let Gaussian Random Variable* Z ∼ N(µ <sup>⊤</sup>ω, 1)*, then*

$$\begin{aligned}\mathbb{E}_x[H_k(\omega^\top x)] &= \mathbb{E}_{\xi \sim n(0,1)}[H_k(\omega^\top \mu + \xi)] \\ &= \sum_{j=0}^k \binom{k}{j} (\omega^\top \mu)^{\circ j} E[H_k(\xi) H_0(\xi)] = (\omega^\top \mu)^k\end{aligned}\tag{95}$$

# N. Information of ImageNet subset used in Experiments

Table 5: Configuration of Expr. V

|          | Vehicle | Bird | Product | Clothing |
|----------|---------|------|---------|----------|
| D        | 98      | 100  | 11316   | 3985     |
| D+I(Sub) | 138     | 159  | 11568   | 4031     |
| D+I      | 1098    | 1100 | 12316   | 4985     |

Table 6: Configuration of Expr. VI

|          | Step 0 | Step 1 | Step 2 | Step 3 |
|----------|--------|--------|--------|--------|
| Vehicle  | 25     | 50     | 75     | 98     |
| Bird     | 25     | 50     | 75     | 100    |
| Product  | 2829   | 5658   | 8487   | 11316  |
| Clothing | 996    | 1992   | 2989   | 3985   |

In this section, we present the criteria used to select classes for constructing the ImageNet subsets. We manually verified the label information to select the classes. The ImageNet subsets corresponding to the base fine-grained datasets were constructed as follows: I(V), I(B), I(P), and I(C), representing the Vehicle, Bird, Product, and Clothing subsets, respectively. These subsets consist of 59, 40, 353, and 46 classes, respectively. To balance the number of samples per class with those in the base fine-grained datasets, we extracted 82, 58, 5, and 6 samples per class for I(V), I(B), I(P), and I(C), respectively.

## N.1. I(V): The Vehicle classes chosen in ImageNet

Total 40 classes.

ambulance, cab, convertible, fire engine, forklift, freight car, garbage truck, go-kart, golfcart, half track, harvester, horse cart, jeep, jinrikisha, limousine, minibus, minivan, Model T, moped, motor scooter, mountain bike, moving van, oxcart, passenger car, pickup, police van, racer, recreational vehicle, school bus, snowmobile, snowplow, sports car, streetcar, tank, tow truck, tractor, trailer truck, tricycle, trolleybus, unicycle

## N.2. I(B): The bird classes chosen in ImageNet

Total 59 classes.

cock, hen, ostrich, brambling, goldfinch, house finch, junco, indigo bunting, robin, bulbul, jay, magpie, chickadee, water ouzel, bald eagle, vulture, great grey owl, black grouse, ptarmigan, ruffed grouse, prairie chicken, peacock, quail, partridge, African grey, macaw, sulphur-crested cockatoo, lorikeet, coucal, bee eater, hornbill, hummingbird, jacamar, toucan, drake, red-breasted merganser, goose, black swan, tusker, white stork, black stork, spoonbill, flamingo, little blue heron, American egret, bittern, crane, limpkin, European gallinule, American coot, bustard, ruddy turnstone, red-backed sandpiper, redshank, dowitcher, oystercatcher, pelican, king penguin, albatross

# N.3. I(P): The Product classes chosen in ImageNet

Total 353 classes.

abacus, accordion, acoustic guitar, altar, analog clock, apiary, ashcan, assault rifle, backpack, balance beam, balloon, ballpoint, Band Aid, banjo, barbell, barber chair, barometer, barrel, barrow, baseball, basketball, bassinet, bassoon, bathing cap, bath towel, bathtub, beach wagon, beacon, beaker, bearskin, beer bottle, beer glass, bell cote, bib, bicycle-built-for-two, binder, binoculars, bobsled, bolo tie, bonnet, bookcase, bottlecap, bow tie, brass, breakwater, broom, bucket, buckle, bulletproof vest, caldron, candle, cannon, canoe, can opener, car mirror, carousel, carpenter's kit, carton, car wheel, cash machine, cassette, cassette player, CD player, cello, cellular telephone, chain, chain saw, chest, chiffonier, chime, china cabinet, cleaver, clog, cocktail shaker, coffee mug, coffeepot, coil, combination lock, computer keyboard, confectionery, corkscrew, cornet, cradle, crash helmet, crate, crib, Crock Pot, croquet ball, crutch, dam, desk, desktop computer, dial telephone, digital clock, digital watch, dining table, dishrag, dishwasher, disk brake, dogsled, doormat, drum, drumstick, dumbbell, Dutch oven, electric fan, electric guitar, electric locomotive, envelope, espresso maker, face powder, feather boa, 3190 3194 3196 3199 3200 3204 3206 3209 file, fire screen, flagpole, flute, folding chair, football helmet, fountain pen, four-poster, French horn, frying pan, gasmask, gas pump, goblet, golf ball, gondola, gong, grand piano, grille, guillotine, hair slide, hair spray, hammer, hamper, hand blower, hand-held computer, handkerchief, hard disc, harmonica, harp, hatchet, holster, honeycomb, hook, horizontal bar, hourglass, iPod, iron, jack-o'-lantern, jigsaw puzzle, joystick, knot, ladle, lampshade, laptop, lawn mower, lens cap, letter opener, lighter, lipstick, lotion, loudspeaker, loupe, magnetic compass, mailbox, maraca, marimba, matchstick, maypole, measuring cup, medicine chest, microphone, microwave, milk can, mixing bowl, modem, monitor, mountain tent, mousetrap, muzzle, nail, neck brace, necklace, nipple, notebook, oboe, ocarina, odometer, oil filter, organ, oscilloscope, oxygen mask, packet, paddle, paddlewheel, padlock, paintbrush, paper towel, parachute, parallel bars, park bench, parking meter, pay-phone, pedestal, pencil box, pencil sharpener, perfume, Petri dish, photocopier, pick, picket fence, piggy bank, pill bottle, pillow, ping-pong ball, plastic bag, plate rack, plow, plunger, Polaroid camera, pole, pool table, pop bottle, pot, potter's wheel, power drill, prayer rug, printer, prison, projectile, projector, puck, punching bag, purse, quill, quilt, racket, radiator, radio, radio telescope, rain barrel, reel, reflex camera, refrigerator, remote control, revolver, rifle, rocking chair, rotisserie, rubber eraser, rugby ball, rule, safe, safety pin, saltshaker, sax, scabbard, scale, scoreboard, screen, screw, screwdriver, seat belt, sewing machine, shield, shopping basket, shopping cart, shovel, shower cap, shower curtain, ski, sleeping bag, sliding door, slot, snorkel, soap dispenser, soccer ball, sock, solar dish, soup bowl, space bar, space heater, spatula, spider web, spindle, spotlight, steel drum, stethoscope, stole, stopwatch, stove, strainer, stretcher, studio couch, sunscreen, swab, switch, syringe, table lamp, tape player, teapot, teddy, television, tennis ball, theater curtain, thimble, thresher, throne, tile roof, toaster, tobacco shop, toilet seat, torch, totem pole, tray, tripod, trombone, tub, turnstile, typewriter keyboard, umbrella, vacuum, vase, vault, velvet, vending machine, violin, volleyball, waffle iron, wall clock, wallet, wardrobe, washbasin, washer, water bottle, water jug, water tower, whiskey jug, whistle, window screen, window shade, wine bottle, wing, wok, wooden spoon, comic book, crossword puzzle, street sign, traffic light, book jacket, menu, plate

3214 Total 46 classes.

3216 abaya, academic gown, apron, bikini, brassiere, breastplate, cardigan, chain mail, Christmas stocking, cloak, cowboy boot, cowboy hat, cuirass, diaper, fur coat, gown, hoopskirt, jean, jersey, kimono, knee pad, lab coat, Loafer, mailbag, mask, military uniform, miniskirt, mitten, overskirt, pajama, poncho, running shoe, sandal, sarong, ski mask, sombrero, suit, sunglass, sunglasses, sweatshirt, swimming trunks, trench coat, vestment, wig, Windsor tie, wool

3219

3224 To generate a set of rotation matrices with diverse magnitudes of rotation, we constructed an algorithm that samples k = 300 random matrices, each formed by adding i.i.d. Gaussian noise matrix of varying variance to the identity matrix I. The process ensures the generation of rotation matrices with varying extents of rotation, from slight to more substantial deviations from the identity matrix.

3226 The rotation matrices are generated as follows:

- 3229
- 1. A matrix is initialized as I + ϵ · M, where M is a i.i.d. standard random Gaussian matrix.
- 2. Using the QR decomposition, we orthogonalize this matrix to ensure it forms a valid rotation matrix.
- 3. Finally, if the determinant of the resulting matrix is negative, we flip the sign of the first column to maintain a determinant of +1, ensuring it is a valid rotation.

3234

3236 In summary, this method provides a collection of matrices that progressively deviate from I, allowing us to observe and sample rotations of increasing magnitude. Please refer Algorithm [3](#page-59-0)

# N.4. I(C): The Clothing classes chosen in ImageNet

# O. Rotation Matrix Generation Process of *Setup 2*

3254

3256

3258 3259 3260

3264

3266

3269

3274

3276

3279

3289 3290

3294

3296

Algorithm 3 Gaussian-Sampled Random Rotation Matrix Generation

Input: Number of dimensions n, number of matrices k Output: Stack of random rotation matrices Initialize empty list Q Set ϵ ← 0.5 for i ← 0 to k − 1 do if i mod k <sup>16</sup> = 0 and i ̸= 0 then ϵ ← ϵ × 0.22360679775 end if Generate random matrix M: M ∼ N(0, 1)<sup>n</sup>×<sup>n</sup> Compute perturbed matrix: A ← I<sup>n</sup> + ϵ × M Compute QR decomposition: Q, R ← QR(A) if det(Q) < 0 then Flip first column of Q: Q[:, 0] ← −Q[:, 0] end if Add Q to Q end for return Q