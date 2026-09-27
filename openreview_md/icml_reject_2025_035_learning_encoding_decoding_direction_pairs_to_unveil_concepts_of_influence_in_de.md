# Learning Encoding - Decoding Direction Pairs To Unveil Concepts Of Influence In Deep Vision Networks

## Anonymous Authors1 Abstract

Latent space directions have played a key role in understanding, debugging, and improving deep learning models, since concepts are encoded in directions of the feature space as superpositions. The encoding direction of a concept maps a latent factor to a feature component, while the decoding direction retrieves it. These encoding-decoding direction pairs unlock significant potential to open the black-box nature of deep networks. Decoding directions help attribute meaning to latent codes, while encoding directions help assess the influence of the concept on the predictions, and both directions may assist in unlearning irrelevant concepts. Compared to previous autoencoder and dictionary learning approaches, we offer a different perspective in learning these direction pairs. We base the decoding direction on unsupervised interpretable basis learning and introduce signal vectors to estimate encoding directions. Meanwhile, we empirically prove that the uncertainty region of the model is informative and can be used to effectively reveal meaningful and influential concepts that impact model predictions. Tests on synthetic data show the approach's efficacy in recovering the underlying encoding-decoding direction pairs in a controlled setting, while experiments on state-of-the art deep image classifiers show notable improvements, or competitive performance, in interpretability and influence, compared to previous unsupervised and even supervised direction learning approaches.

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054

## 1. Introduction

1 2014; Alain & Bengio, 2018; Zhou et al., 2018; Kim et al.,
2018; Elhage et al., 2022; Nanda et al., 2023). The latent factor of a concept constitutes a scalar, signifying the presence of the concept within an image patch. When this scalar is multiplied by the concept's embedding, also referred to as the encoding or *signal* direction, this direction maps this factor to a component in the patch's representation. In contrast, a *filter* can be used to extract this latent factor from the representation using the inner product, designating it as a *decoding* direction. The decoding direction of a concept enables understanding representations, attributing meaning to latent codes (Zhou et al., 2018; Kim et al., 2018), while the encoding direction allows for assessing its influence on the network's predictions (Fel et al., 2023b; Pahde et al., 2024), and both directions may be used in compelling the network to unlearn concepts irrelevant to the prediction task (Anders et al., 2022; Pahde et al., 2023; Dreyer et al., 2024). Most previous approaches (Zhou et al., 2018; Kim et al., 2018; Zhang et al., 2021; Fel et al., 2023b; Doumanoglou et al., 2023; Pahde et al., 2024; Doumanoglou et al., 2024) usually focus on identifying either the decoding or the encoding directions in isolation, limiting their applicability to specific appropriate tasks. Moreover, many of them do not explicitly make this distinction and consider using the concepts' decoding directions in use cases where the encoding direction is a better fit. This has recently been pinpointed in the context of concept influence assessment and model correction in (Pahde et al., 2024). In this work, we learn the concept encoding-decoding direction pairs, jointly, in an unsupervised manner. Unlike recent advances in sparse autoencoders and dictionary learning (Bricken et al., 2023; Lim et al., 2024; Cunningham et al., 2024), which focus on sparsity within feature space units, we emphasize sparsity in the semantic space of concepts. We model the decoding directions using the principles of the recently introduced, unsupervised interpretable basis learning (Doumanoglou et al., 2023). This modeling provides an explicit rule for concept detection by learning the decoding direction together with an additional threshold to ascertain the presence of a concept. We term this rule as a concept detector due to its ability to detect the presence of a concept.

Furthermore, we introduce *signal vectors* as estimators of a concept's encoding direction, confirming their precision in synthetic settings and their impact in real world contexts. Finally, we show that the uncertainty region of a network, that is, the subspace where the network's predictions are uncertain, is informative, and when aligned with the uncertainty region of the concept detectors, can significantly guide the search towards more meaningful concepts that notably impact the network's predictions. Experiments in a controlled setting show the efficacy of the proposed approach in identifying the correct concept encoding-decoding direction pairs, when prior work fails, while experiments on deep vision networks demonstrate the superiority of our method, or competitive performance on par with previous unsupervised and supervised direction learning approaches, in interpretability and influence metrics.

## 2. Related Work

We categorize related work of direction learning into supervised and unsupervised approaches. In each category, we go through previous methods for learning the concept directions, describing the limitations, differences, and similarities with the approach proposed here. Supervised Concept Direction Learning Typical approaches (Zhou et al., 2018; Kim et al., 2018) to interpretable (concept) direction learning use a linear classifier with annotations of a concept dataset. This classifier distinguishes representations of samples with the concept from those without, with interpretable directions as the filter weights and the learned bias as a classification threshold. Known as Concept Activation Vectors (CAVs), these directions resemble the concept decoding directions of Section 1, although they are not exact due to distractor-noise in feature components (Haufe et al., 2014; Kindermans et al., 2017; Pahde et al., 2024). While CAVs could be considered as estimators of concept decoding directions, the recently introduced Pattern-CAVs (Pahde et al., 2024), estimate a concept's encoding direction by the difference in (positive and negative) cluster means. Unsupervised Concept Direction Learning Matrix decomposition methods help identify concept encoding directions without annotations, yet with some limitations. For instance, Principal Component Analysis (PCA) (Graziani et al., 2023)
is limited by orthogonality and cannot represent concepts that do not affect variance (Fel et al., 2023a). Likewise, Non-negative Matrix Factorization (NMF) (Zhang et al., 2021; Fel et al., 2023b) assumes positive components and lacks bias, limiting expressivity. While PCA's transpose matrix estimates decoding directions, NMF lacks a simple equivalent. Besides that, for NMF the concept classification rule requires an optimization problem to be solved for every test sample, making the approach computationally more ex055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 pensive than a calculation of an inner product. Our method overcomes these limitations. In Dictionary Learning (Bricken et al., 2023; Yun et al., 2023) and Sparse Autoencoders (Sharkey et al., 2022; Cunningham et al., 2024), the goal is to learn decoding-encoding directions by reconstructing representations after decomposing them into latent factors and enforcing sparsity in units of latent variables. Moreover, those factors are constrained to be non-negative. Unlike these methods, our approach allows negative latent factors since it enforces sparsity in the semantic space (a soft-binary vector space) and additionally considers the use of the directions by the model. Beyond that, our method uses a different principle for identifying direction pairs, independent of feature reconstruction. In contrast to earlier techniques, the method in (Doumanoglou et al., 2023; 2024) learns filter directions of linear classifiers, similar to the supervised methods in (Zhou et al., 2018; Kim et al., 2018). These classifiers convert representations into a soft-binary concept space, guided by a sparsity objective. We ground our approach on this model and additionally enhance it by removing orthogonality constraints, feature space standardization, and adding loss terms to a) sustain or improve interpretability of the identified concepts and b) reduce impact of distractor-noise on filter weights. Although (Doumanoglou et al., 2024) proposed a technique to exploit the utilization of the directions by the network in direction search, our subspace alignment approach shows a notable relative improvement over this previous approach (up to 22.56% in the interpretability metrics), in 3 of 4 cases. Finally, these methods did not consider estimators for the concepts' encoding directions as we do. More details on this comparison can be found in Section H.

## 3. Background

The latent factor of a concept is a scalar linked to the concept's presence, embedded in the feature space via multiplication with its encoding direction, also called the concept's signal direction. For this reason, we also refer to this latent factor as the **signal value**. Features are considered as superpositions of signals and noisy directional components called **distractors**. In the proposed approach, a **filter** is a decoding direction that, through the inner product with a feature representation, extracts the signal's value. Below we provide a more formal explanation of these terms and provide details essential to understand our contributions.

## 3.1. Preliminaries

Let X ∈ R
H×W×D denote the representation of an image in an intermediate layer of a convolutional neural network with spatial dimensions *H, W* ∈ N
+ and feature space dimensionality D ∈ N
+. Let also xp ∈ R
D denote an element of this representation at the spatial location p =
(w, h), w ∈ {0, 1, ..., W − 1}, h ∈ {0, 1*, ..., H* − 1}.

## 3.2. Signals, Distractors, Filters, Concept-Detectors, Pattern-Cavs

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 In encoding a single concept i, (Kindermans et al., 2017; Pahde et al., 2024) propose a binary model for the data generation process of feature representations: xp = αpsi +
βpd, si, d ∈ R
D, αp, βp ∈ R. Here, siis the signal direction indicating if xp is part of concept i. The key information is in the signal value αp. Larger αp suggests greater confidence that xp belongs to concept i. d is the distractor direction, modeling noise, or information unrelated to the concept. βp follows a Gaussian distribution N (*µ, σ*2) independent of whether xp belongs to concept i. As per (Kindermans et al., 2017), the value of the signal αp can be extracted via a *regression filter* wi:
zp,i = wT
i xp = αpwT
i si + βpwT
i d, if we choose wi:
wi ⊥ d, and wT
i si = 1. Since stronger values of αp indicate more confidence in concept presence, when combined with a threshold bi ∈ R which may be learned from data, this regression filter can be turned into a **concept detector**:
yp,i = σ(zp,i − bi), with σ denoting the sigmoid function.

With access to the signal's value, (Haufe et al., 2014; Kindermans et al., 2017) offer a formula to estimate the concept's signal direction:

$${\hat{\mathbf{s}}}_{i}={\frac{\operatorname{cov}[\mathbf{x_{p}},z_{\mathbf{p},i}]}{\sigma_{z_{p,i}^{2}}}}$$

$$(1)$$

where, σ 2zp,i denotes the variance of the signal values in the dataset. This signal estimator relies on signal values, but when trying to explain the latent space, these data are not available. We only have access to xp, while si and d are latent variables of the underlying process. For this reason, based on (Haufe et al., 2014; Kindermans et al., 2017), (Pahde et al., 2024) proposed Pattern-CAVs as concept signal estimators that don't need signal values but rely on labeled data, i.e. concept's positive and negative samples. Their method is based on (1), approximating the signal value with binary labels, i.e., zp,i ∈ {0, 1}, and simplifies to the difference between the means of the concept's positive and negative samples.

## 3.3. Unsupervised Interpretable Direction Learning

Recent research (Doumanoglou et al., 2023) introduced an unsupervised method to identify concepts from the structure of the feature space. Motivated by the directional encoding of concepts, the method partitions the latent space into linear regions, each represented by a hyperplane and a normal vector, forming clusters. Features from an unlabeled *concept* dataset, possibly the network's training set, are assigned to these clusters. It learns W and b of a feature-to-cluster membership function, a mapping to the semantic space, with yp = σ(WT xp − b) ∈ [0, 1]I,W ∈ R
D×I, b ∈ R
I,
and I as the cluster count. By softly assigning features to a small number of clusters, interpretability is improved. This is grounded in the idea that an image patch generally holds only a few semantic labels from a larger set, reflecting sparsity in the semantic space. Sparsity in the assignments is achieved using two loss terms: the first is *Sparsity Loss* (L
s), and the second is *Maximum Activation Loss* (L
ma),
which ensures binary cluster membership.:

$$\begin{array}{l}{{\cal L}^{s}=\mathbb{E}_{\mathbf{p}}\big{[}{\cal L}_{\mathbf{p}}^{s}\big{]},\quad{\cal L}^{ma}=-\mathbb{E}_{\mathbf{p}}\big{[}\mathbf{q}_{p}^{T}\log_{2}(\mathbf{y}_{\mathbf{p}})\big{]},}\\ {{\cal L}_{\mathbf{p}}^{s}={\cal H}(\mathbf{q}_{\mathbf{p}}),\quad\mathbf{q}_{\mathbf{p}}=\frac{\mathbf{y}_{\mathbf{p}}}{||\mathbf{y}_{\mathbf{p}}||_{1}}}\end{array}\tag{2}$$

with H denoting entropy. The columns of W and elements of b (e.g., wi, bi) form a linear classifier or concept detector yp,i = σ(wT
i xp − bi). This method also optimizes linear separability by minimizing the inverse of the classification margin Mi =1 ||wi||2
(*Maximum Margin Loss* - L
mm) and penalizes clusters with few assignments using the *Inactive* Classifier Loss - L
ic (Doumanoglou et al., 2024) (More details in Section A). Despite the potential misalignment with human intuition, the sparse nature of transformed representations facilitates concept definition or identification.

## 3.4. Direction Labeling

In (Doumanoglou et al., 2023; 2024), classifier filters form an orthogonal feature space basis, with vectors aligned to cluster regions. Although annotations are not needed for basis learning, interpretability evaluation uses Network Dissection (Bau et al., 2017), a method assigning semantic labels to each vector based on classifier performance with annotated concepts. Despite possible biases against unsupervised learning, we adopt and expand this protocol to evaluate the interpretability of our concept detectors.

## 3.5. Concept Influence Testing

Given an image's intermediate representation for class k and a concept's direction i in latent space, RCAV (Pfau et al., 2020) measures concept sensitivity for the class by perturbing the representation towards the concept's direction with strengh α, and subsequently comparing the network's output probability for class k before and after the perturbation.

A total dataset score in the range [−1, 1] follows, where zero means inconsistent use of the concept by the model, while extremes indicate strong positive or negative concept contributions to predict class k. A statistical test compares concept sensitivity against sensitivity towards random directions to ensure significance. We refer to directions of significant influence in cases where the directions meets the criteria of this statistical significance test.

## 4. Method

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 For a specific network layer, our method receives as input the feature representations of images sourced from a *concept dataset*. The aim of our approach is to learn encoding-decoding direction pairs that correspond to meaningful concepts with influence on the network's predictions. The method is unsupervised, and therefore, the identified concepts may not align with human intuition. However, they reflect clear directional clusters in the feature space of the network. Thus, this approach has the potential to reveal erroneous strategies exploited by the model to make predictions (Section I). In an attempt to make these clusters as meaningful as possible, we optimize for a sparsity property of interpretability in the feature-to-cluster assignments.

We extend the binary signal-distractor data model (Section 3.2) to multiple concepts (Section 4.1) and learn concept detectors {W, b} using the objectives of Section 3.3. In the new data model, we remove the constraints of (Doumanoglou et al., 2023) on filter orthogonality and feature space standardization, allowing flexible representation clustering. These prior constraints though, acted as regularizers to prevent degenerate solutions; thus, we address their removal with additional loss terms discussed in Section 4.2 that sustain or even improve the interpretability of the clustering. We additionally estimate concept signal directions using learnable **signal vectors** sˆi (Section 4.3).

We also propose Uncertainty Region Alignment (Section 4.4), a loss that significantly improves cluster quality. In summary, concept detectors and signal vectors are learned together in an end-to-end process, influenced by the losses of Sections 3.3, 4.2, 4.3 and 4.4. A summary of the interconnections between components of the method is provided in Fig. 1.

## 4.1. Multi-Concept Signal-Distractor Data Model

We introduce an extended signal-distractor model for the latent space, encoding multiple concepts. Each spatial element xp is a linear combination of latent concept signals S ∈ R
D×Iand distractors D ∈ R
D×F , F ≤ D − I

$$x_{p}=S\alpha_{p}+D\beta_{p}$$
$$({\mathfrak{I}})$$
xp = Sαp + Dβp (3)
with αp ∈ R
Iand βp ∈ R
F . S is a matrix of I ∈ N
+,
D-dimensional, unit-norm concept signal directions and D a matrix denoting a basis for distractor components. Each signal direction encodes the presence of a distinct concept. We apply the same assumptions for individual signal values αp,i (the i-th element of αp) and distractor coefficients βp,f as in Section 3.2. Finally, we further assume that only a limited number of semantic concepts are assigned to xp, among many possible semantic labels.

## 4.2. Interpretability Losses To Recover Implicit Regularizations

We propose **Self-Weighted Reduction** (RSW ) as a loss aggregation method to optimize upper bounds. Consider a set of un-reduced loss values Lk, k ∈ N. The Self-Weighted Reduction is:

$${\cal R}_{S W}(\{{\cal L}_{k}\})=\frac{\sum_{k}{\cal L}_{k}^{\nu+1}}{\sum_{k}{\cal L}_{k}^{\nu}}\qquad\qquad(4)$$

which is equal to the weighted average of elements in {Lk}
with each element being weighted by L
νk
, ν > 1, ν ∈ R
+
a sharpening factor. This loss may be seen as a softdifferentiable version of the max operation, since the largest value in the set of {Lk}, is weighted with the largest weight.

Excessively Active Classifier Loss (L
eac) This loss penalizes excessively large clusters to prevent trivial solutions where all inputs belong to one cluster. It relies on a hyperparameter ρ, similar to sparse autoencoders (Ng et al., 2011), which sets a proportional bound on cluster size. The unreduced formula is below, with γ > 1, γ ∈ R
+ as a sharpening factor and 1 − ρ normalizing the loss in range [0, 1]:

$${\mathcal{L}}_{i}^{e a c}={\frac{1}{1-\rho}}\mathrm{ReLU}(\mathbb{E}_{\mathbf{p}}[y_{\mathbf{p},i}^{\gamma}]-\rho)$$
$$({\mathfrak{H}})$$
$\neg$
$${\mathcal{L}}^{e a c}$$

The final reduced loss, is using RSW : L
eac =
RSW ({Leac i})
Sparsity Bound Loss (L
sb) With this loss term we minimize the upper bound of the un-reduced L
s, among pixel locations, using RSW . In more detail, if L
sp(2) denotes the Sparsity Loss for pixel p, the Sparsity Bound Loss (L
sb) is defined as L
sb = RSW ({Lsp})

## 4.3. Signal Vectors As Concept Signal Estimators

Considering the new data model outlined in Section 4.1, the assumptions of (1) for estimating a concept's signal direction may not be valid. Specifically, there may be an anti-correlated relationship between the variables αp,i and αp,j , i ̸= j due to the fact that concept labels are sparsely assigned to each xp, indicating mutual exclusivity in concept label attributions. Nevertheless, as detailed in the Section B, we can effectively apply (1) by only using positive samples of the concept (p : yp,i > 0.5) when calculating variance and covariance, rather than both positive and negative samples as previously recommended. We call the signal estimator for concept i, obtained under these conditions, the signal vector sˆi. However, we still require access to signal values. As explained in Section 3.2, estimating signal values can be attributed to the concept detectors' filters. They can serve as signal value estimators if the weight vector wiis orthogonal to all sj where j ̸= i, as well as the distractor subspace D. Thus, we employ the folllowing Filter-Signal Vector Orthogonality Loss to learn the directions:

$${\mathcal{L}}^{f s o}={\sqrt{\mathbb{E}_{i,j}\left[(1-\delta_{i,j}){\bar{\mathbf{w}}}_{i}^{T}{\bar{\mathbf{s}}}_{j})^{2}\right]}}$$
2(6)
with δi,j the kronecker delta and w¯, s¯ denoting the L2normalized filter weights and signal vectors. To achieve accurate signal value extraction, wi should additionally be orthogonal to the distractor basis; however, we do not explicitly estimate the distractors. Instead, we use the Uncertainty Region Alignment loss from Section 4.4 to ensure alignment of the directions with utilization by the network.

## 4.4. Uncertainty Region Alignment To Improve Interpretability And Influence

The presence or absence of a concept in a representation can offer neutral, supportive, or opposing evidence against the prediction of a class. Since the concept-class pair association is unknown when learning concept directions, a straightforward strategy to perform concept arithmetic on the features in order to find their utility by the network lacks ground-truth information on how concepts affect class predictions. To overcome this difficulty, we can make a simple but more elegant hypothesis that uncertain network predic220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 tions occur when the representation has ambiguous concept information. We propose improving the direction search by aligning the uncertainty regions of the network and the concept detectors. The *uncertainty region of the network* is the subspace where its predictions are most uncertain, and the *uncertainty region of the concept detectors* is the subspace where their decision hyperplanes intersect. Figure 2 illustrates the concept of Uncertainty Region Alignment.

We first manipulate spatial features xp towards the direction −dxp to arrive in x
′
p = xp − dxp. Based on our estimates of wi, bi, and sˆi, we select the direction dxp so that the shifted x
′
plies at the intersection of the concept detectors' decision hyperplanes. Then, we ensure the network's predictions for these features are highly uncertain, effectively aligning both uncertainty regions.

## Unconstrained And Constrained Uncertainty Region Losses (L Uur, L Cur)

We define two types of Uncertainty Region Loss: i) unconstrained L
uur and ii) constrained L
cur. Each loss uses a different feature manipulation strategy dxp but both share the same final formula

$${\mathcal{L}}^{u u r}={\mathcal{L}}^{c u r}=-\mathbb{E}_{\mathbf{X}^{\prime}}\big[{\mathcal{H}}(f^{+}(\mathbf{X}^{\prime})\big]\qquad\qquad(7)$$

(with H denoting entropy and f
+ denoting the part of the network after the layer of study providing output class probabilities). In (7), X′ denotes a manipulated image representation, with every xp shifted in the direction −dxp.

i) Unconstrained Uncertainty Region Manipulation Suppose that we know the concept decoding directions wi and the classification thresholds bi, we can bring all xp to the concept detectors' uncertainty region by manipulating each xp in the direction −dxp with the following formula:

$$(6)$$
$\mathbf{w}_{i}^{T}\mathbf{x}_{p}^{\prime}-b_{i}=0\Rightarrow\mathbf{w}_{i}^{T}(\mathbf{x_{p}}-\mathbf{dx_{p}})-b_{i}=0$, $\forall i$, $\mathbf{W}^{T}(\mathbf{x_{p}}-\mathbf{dx_{p}})-\mathbf{b}=\mathbf{0}\Rightarrow\mathbf{dx_{p}}=(\mathbf{W}^{T})^{+}(\mathbf{W}^{T}\mathbf{x_{p}}-\mathbf{b})$
with A+ denoting the pseudo-inverse of A.

ii) Constrained Uncertainty Region Manipulation The previous unrestricted approach to the manipulation of features might lead to datapoints falling outside the concept encoding manifold of the network, causing an unfaithful alignment. To address this, we suggest restricting feature manipulation to occur within the span of the signal vectors, i.e. dxp = Svˆ , v ∈ R
I, and thus:

$W^{T}(\mathbf{x_{p}}-\mathbf{dx_{p}})-\mathbf{b}=\mathbf{0}\Rightarrow W^{T}(\mathbf{x_{p}}-\hat{\mathbf{S}}\mathbf{v})-\mathbf{b}=\mathbf{0}\Rightarrow$  $W^{T}\hat{\mathbf{S}}\mathbf{v}=W^{T}\mathbf{x_{p}}-\mathbf{b}\Rightarrow\mathbf{v}=(W^{T}\hat{\mathbf{S}})^{+}(W^{T}\mathbf{x_{p}}-\mathbf{b})\Rightarrow$  $\mathbf{dx_{p}}=\hat{\mathbf{S}}\mathbf{v}=\hat{\mathbf{S}}(W^{T}\hat{\mathbf{S}})^{+}(W^{T}\mathbf{x_{p}}-\mathbf{b})$
where Sˆ represents a matrix whose i-th column is equal to the estimated signal vector sˆi.

275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 Table 1. Evaluating the performance of the concept detectors in classifying pixel representations in the experiment on synthetic data. The metric is Intersection over Union (IoU). Rows correspond to concept detectors and columns to ground-truth concept classes.

Concept
\#0 \#1 \#2

| Concept   |      |      |    |      |
|-----------|------|------|----|------|
| #0        | #1   | #2   |    |      |
| #0        | 0    | 0.96 | 0  |      |
| ctor #1   | 0.92 | 0    | 0  |      |
| ete       | #2   | 0    | 0  | 0.96 |
| D         |      |      |    |      |

## 5. Experiments 5.1. Experiment On Synthetic Data

In this section, we test our method on synthetic data, demonstrating that it reliably identifies the key elements of the data generation process detailed in Section 4.1, under challenging conditions for conventional techniques.

We generate features xp according to (3) by setting the embedding space dimensionality to D = 16, the number of distinct concepts to I = 3, the size of the distractor basis to F = 2 and randomly create unit-norm vectors to construct the matrices S and D. The latent signal values and distractor coefficients follow the uniform distribution: αp,i ∼ U(0.0, 2.5) if p is not part of concept i and αp,i ∼ U(2.5, 5.0), otherwise, while βp,f ∼ U(0, 5.0) is independent of the pixel concept label. We introduce a bias of 10 across all dimensions of the representations to maintain them in the positive quartile, similar to the impact of a ReLU layer. The image representations are considered with two spatial elements p1, p2, where W = 2 and H = 1. Each pixel representation corresponds to a single concept. Let c(p) ∈ {0, 1, 2} represent the concept label of p, and k ∈ {*a, b, c*} denote an image class. We construct image representations as follows: for k = a, c(p1) = 0 and c(p2) = 1; for k = b, c(p1) = 0 and c(p2) = 2; and for k = c, c(p1) = 1 and c(p2) = 2. We generate a balanced dataset with each class being represented by 1000 images. The network we use is composed of just two layers (corresponding to the top part of a potentially larger convolutional network). The first is an average-pooling layer, and the second is a linear layer with K = 3 output classes. After training, the network attains 99% accuracy on a test set, randomly generated based on the previous principles. More details of this experiment are in Section E.

Decoding directions: We first assess the ability of the concept detectors to identify the specified concepts. Table 1 shows Intersection over Union scores for each detector against actual concept classes. Zero values indicate complete purity and no mixing of the concepts, and all scores are above 0.92, showing success. We also examine how well the learned filters extract signal values from representations. Our method identifies signal values as a deviation from the dataset's average since distractor directions aren't directly estimated (See also Section C). The Root Mean Squared Error (RMSE) between these extracted values and the ground truth, after subtracting the mean signal value, is noted as 0.26. Encoding directions: We also examine the cosine similarity between signal vectors and true concept encoding directions, comparing them with a Sparse Autoencoder (Bricken et al., 2023) at varying sparsity levels and Pattern-CAVs learned using ground-truth labels. Fig. 3 shows that signal vectors closely estimate the concept's encoding directions, unlike Pattern-CAVs, which falter due to the different assumptions made about the data. The sparse autoencoder fails due to the fact that the reconstruction goal objective can be met with any basis of the data manifold. The ground-truth encoding directions of this example (Table 6) contain both positive and negative components and the relationship among the encoding directions (and the distractors) is in general not orthogonal. In theory and without the need for practical experiments, this example cannot be addressed by NMF, K-Means, or PCA. NMF would produce a signal basis of non-negative components akin to the cluster centers of K-Means, which would point toward the positive quartile where the centroids reside. Moreover, the nonorthogonal nature of the ground truth encoding directions implies that the PCA's solution space is insufficient.

Table 2. Ablation study wrt interpretability losses (top part) and uncertainty region alignment losses (bottom part).

ResNet18 / Places365

I L

uur L

sb L

eac L

cur L

f so S

1 S

2 SDC SCDP

| uncertainty region alignment losses (bottom part). ResNet18 / Places365 sb L eac L cur L f so S 1 S 2   | SDC   | SCDP   |    |    |       |       |     |      |
|---------------------------------------------------------------------------------------------------------|-------|--------|----|----|-------|-------|-----|------|
| ✓                                                                                                       | ✗     | ✗      | ✗  | ✗  | 59.06 | 25.91 | 377 | 2487 |
| ✓                                                                                                       | ✓     | ✗      | ✗  | ✗  | 49.02 | 35.23 | 354 | 2480 |
| ✓                                                                                                       | ✓     | ✓      | ✗  | ✗  | 54.55 | 37.38 | 359 | 2118 |
| ✗                                                                                                       | ✓     | ✓      | ✓  | ✓  | 57.34 | 38.36 | 376 | 3271 |
| ✓                                                                                                       | ✓     | ✓      | ✗  | ✗  | 50.49 | 34.76 | 283 | 1451 |
| ✗                                                                                                       | ✓     | ✓      | ✓  | ✗  | 50.94 | 36.01 | 335 | 2930 |
| ✗                                                                                                       | ✓     | ✓      | ✓  | ✓  | 52.63 | 34.86 | 360 | 2956 |

330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384

$$\mathcal{S}^{1}=\int_{0}^{1}\sum_{i=0}^{I-1}\mathbb{1}_{x\geq\xi}\big{(}\phi_{i}(c_{i}^{*},\mathcal{K}_{v})\big{)}d\xi\tag{8}$$  $$\mathcal{S}^{2}=\int_{0}^{1}|\{c_{i}^{*}\,|\,\exists\,i:\phi_{i}(c_{i}^{*},\mathcal{K}_{v})\geq\xi\}|d\xi\tag{9}$$

The first metric S
1counts concept detectors with an IoU
performance that exceeds a score threshold ξ. The second

## 5.2. Experiment On Deep Image Classifiers

We evaluate the method's components through practical experiments on the final convolutional layer of a ResNet18 (He et al., 2016) trained on Places365 (Zhou et al., 2017). For comparison with earlier unsupervised approaches, we also experimented with a ResNet50 trained on Moments in Time (Monfort et al., 2019). Unless stated otherwise, our Encoding-Decoding Direction Pairs (EDDP) uses a weighted combination of all losses from Sections 3.3 and 4.2, along with L
f so and L
cur from Sections 4.3 and 4.4.

The hyperparameter I is set to I = 500, with other method parameters detailed in the Section F. Evaluation of the Decoding Directions: Interpretability Our method's effectiveness in identifying meaningful concepts is assessed using a quantitative approach measuring the interpretability of the directional clustering obtained by the learned concept detectors. We follow the protocol in (Doumanoglou et al., 2023). We utilize the Broden (Bau et al., 2017) dataset for ResNet18, and Broden Action (Ramakrishnan et al., 2019) for ResNet50 to learn and label directions (Section 3.4). The datasets feature dense pixel annotations; Broden includes 1197 concepts across 63K images in 5 concept categories (object, part, material, texture, color), while Broden Action incorporates an additional action category with 210 labels and 23K more images. We employ two metrics from (Doumanoglou et al., 2023).

Specifically, let ϕi(c, K) the Intersection Over Union for concept detector i in identifying concept c within the dataset K. Define c
⋆
i = argmaxcϕi(c, Kt), indicating the concept label detected best by concept detector i within the training subset of the dataset (Kt). With Kv as the validation subset, our interpretability scores S
1and S
2are:
metric S
2 uses the cardinality of the set |.| to count the unique concept labels detected by the concept detectors with IoU above ξ. Both metrics become threshold-agnostic, by integrating on all ξ ∈ [0, 1]. Qualitative segmentations using the learned concept detectors appear in Section G. Evaluation of the Encoding Directions: Influence We assess the ability of our method to identify influential concepts to model predictions using RCAV (Pfau et al., 2020) (Section 3.5). For sensitivity scores, we spatially replicate signal vectors or Pattern-CAVs. Direction significance is tested with RCAV's label permutation test to generate random directions, with the significance threshold set to 0.05 and Bonferroni correction. Two metrics summarize the results: Significant Direction Count (SDC) and Significant Class-Direction Pairs (SCDP). SDC is the count of signal vectors that significantly influence at least one model class, while SCDP tallies class-direction pairs where vectors significantly affect the class. In ablation studies excluding L
cur, we report influence metrics for signal vectors estimated **post learning the directions** with the conditions discussed in Section 4.3. Network explanations using the learned encoding directions and RCAV are in Section F.2.

Ablation Study: Interpretability Losses The top part of Table 2 presents the outcome of an ablation study focusing on the interpretability loss terms introduced in Section 4.2. We start with only L
uur and progressively incorporate L
sb and L
eac, observing a steady and notable improvement in S
2, which is more challenging to optimize compared to S
1, while still keeping up with performance in terms of S
1. Eventually, retaining the interpretability terms and transitioning to the use of the proposed signal vectors and L
cur we see further improvement in interpretability, and a significant increase in the SCDP influence metric. Ablation Study: Uncertainty Region Alignment and Signal Vectors The lower section of Table 2 presents the metric scores of an ablation study centered on uncertainty region alignment and signal vectors. By moving from L
uur to L
cur and subsequently incorporating L
f so, we observe a steady enhancement across all influence metrics. The table highlights the substantial effect of employing signal vectors
(used in L
cur) to detect influential directions. Although the signal vectors derived from the combination of L
cur and L
f so prove to be more influential than the others, this comes at a minor reduction in interpretability. This trade-off between interpretability and influence is aligned with the observations in (Kim et al., 2018; Pfau et al., 2020), where non-interpretable random directions can significantly affect a model's predictions. Interpretability Comparison with Unsupervised Approaches Previous unsupervised approaches (Zhang et al., 2021; Graziani et al., 2023; Fel et al., 2023b) lack a clear classification rule for concept detection, impeding quan385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 Table 5. Interpretability comparison with prior work on unsupervised basis learning. Works considered: UIBE (Doumanoglou et al., 2023), CBE (Doumanoglou et al., 2024) and CBE with CNN Classifier Loss replaced with the proposed L
uur (CBE /w L
uur).

titative evaluation, and relying primarily on human assessment of interpretability. Our work overcomes these limitations, enabling a more effective quantitative evaluation. We compare in interpretability terms with the work of (Doumanoglou et al., 2023; 2024). Since we base our method on them, the meaningful aspect to compare is the contribution of our L
uur (and not L
cur since those works did not consider concept encoding directions). We use the exact setup of (Doumanoglou et al., 2024) (i.e. without lifting the orthogonality of the directions, the feature standardization, or considering signal vectors) and replace their CNN Classifier Loss with our L
uur. The experimental results are provided in Table 5. Our Uncertainty Region Alignment loss significantly improves the interpretability metrics in 3 of 4 scenarios (by up to +22.56% relative improvement) while remaining competent in the remaining case. This justifies that our alignment of uncertainty regions is more effective than the technique proposed in (Doumanoglou et al., 2024).

| ResNet18 / Places365   |                |                 |                  |
|------------------------|----------------|-----------------|------------------|
| UIBE                   | CBE            | CBE /w L uur    |                  |
| 1                      | 60.93 (+0.0%)  | 69.43 (+13.95%) | 67.3 (+10.45%)   |
| 2                      | 28.39 (+0.0%)  | 31.53 (+11.06%) | 32.16 (+13.28%)  |
| ResNet50 / MiT         |                |                 |                  |
| UIBE                   | CBE            | CBE /w L uur    |                  |
| 1                      | 124.73 (+0.0%) | 131.73 (+5.61%) | 158.76 (+27.28%) |
| 2                      | 18.47 (+0.0%)  | 26.94 (+45.86%) | 33.02 (+78.78%)  |

Interpretability Comparison with a Supervised Approach We compare the concept classifiers learned using our method against those learned through a supervised approach, with a focus on interpretability (Table 3). We calculate averaged binary classification metrics across detectors. We consider three variants of the proposed method: a) independent learning of directions (the exact outcome of our method), b) combining directions with a shared label (post initial leaning) using a linear layer for classification (this layer classifies representations as positive if any detector with the same label does), and c) considering the learned Table 4. Influence Comparison against Pattern-CAVs.

Table 3. Comparison of concept-detectors (CDs)' performance in pixel classification and image segmentation tasks. Comparing between: a) individual CDs, b) combined CDs (*Linear-OR*) c) individual CDs with their thresholds learned with supervision, and d) IBD: a set of classifiers learned in a supervised way.

mPrecision mRecall mAP mF1Score S

1 S

2 mIoU

IBD (Zhou et al., 2018) 0.84 0.6 0.77 0.69 53.32 53.32 0.20

EDDP Individual (ours) 0.81 0.24 0.53 0.33 57.33 38.35 0.11

EDDP Linear-OR (ours) 0.73 0.4 0.59 0.45 30.56 30.56 0.11

EDDP Individual /w sup thresholds (ours) 0.62 0.49 0.52 0.53 N/A N/A N/A

directions but optimizing the classification threshold in a supervised manner to enhance the F1 Score. This last approach assesses direction quality independent of bias. Results show that individual classifiers from the proposed method achieve high precision, comparable to supervised ones, but suffer from low recall due to their sparsity-driven objectives. Combining classifiers with the same label improves recall, while supervised optimization of the bias further enhances F1 Scores by reducing sparsity. Influence Comparison with a Supervised Approach We compare network sensitivity to Pattern-CAVs and signal vectors. Let j ∈ {0, 1*, ..., N*l − 1} index concept detectors with the same concept label l, and S
lj,k represent the RCAV
sensitivity score of class k relative to the signal vector of the j-th concept detector for label l. Similarly, S
lP,k is the sensitivity score of class k relative to the Pattern-CAV for the same label. Pattern-CAVs use ground-truth pixel-level labels. Network Dissection can assign identical labels to multiple detectors, making direct comparisons with Pattern- CAVs challenging. Inspired by RCAV, we regard signal vectors as *noise vectors* and assess Pattern-CAV sensitivity for a label against them. We define a metric S
3 which when above 0.5 indicates Pattern-CAVs have more network influence than signal vectors at significance level θ = 0.05 (Bonferroni correction applies):

| RCAV α   | 0.5   | 2.0   | 5.0   |
|----------|-------|-------|-------|
| 3        | 0.37  | 0.37  | 0.38  |
| S        |       |       |       |

$$\begin{array}{c}{{\mathcal{S}^{3}=\mathbb{E}_{l,k}\Big[\mathbb{1}\left(p_{l,k}<\frac{\theta}{N_{l}}\right)\Big]}}\\ {{\ }}\\ {{p_{l,k}=\frac{1}{N_{l}}\sum_{j}\mathbb{1}\left(|S_{j,k}^{l}|\geq|S_{P,k}^{l}|\right)}}\end{array}$$

Metrics for different RCAV values α are in Table 4. The S
3 scores are below 0.5, indicating that Pattern-CAVs are less influential than signal vectors on network predictions.

## 6. Conclusion

We introduced an innovative unsupervised technique to uncover pairs of latent space encoding-decoding directions that align with interpretable and influential concepts. This research offers a new perspective on the unsupervised identification of concept directions, unlike previous methods based on feature reconstruction or decomposition, paving the way for additional exploration.

## Impact Statement References

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. Alain, G. and Bengio, Y. Understanding intermediate layers using linear classifier probes. *arXiv:1610.01644 [cs, stat]*, November 2018. arXiv: 1610.01644.

Anders, C. J., Weber, L., Neumann, D., Samek, W., Muller, ¨
K.-R., and Lapuschkin, S. Finding and removing clever hans: Using explanation methods to debug and improve deep models. *Information Fusion*, 77:261–295, 2022. ISSN 1566-2535.

Bau, D., Zhou, B., Khosla, A., Oliva, A., and Torralba, A. Network dissection: Quantifying interpretability of deep visual representations. *arXiv:1704.05796 [cs]*, April 2017. arXiv: 1704.05796.

Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A.,
Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., and Olah, C. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023.

Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L. Sparse autoencoders find highly interpretable features in language models. International Conference on Learning Representations (ICLR), October 2024.

Doumanoglou, A., Asteriadis, S., and Zarpalas, D. Unsupervised interpretable basis extraction for concept–based visual explanations. IEEE Transactions on Artificial Intelligence, 2023.

Doumanoglou, A., Zarpalas, D., and Driessens, K. Concept basis extraction for latent space interpretation of image classifiers. *VISIGRAPP. Proceedings*, 3:417–424, 2024. ISSN 2184-4321.

Dreyer, M., Pahde, F., Anders, C. J., Samek, W., and Lapuschkin, S. From hope to safety: Unlearning biases of deep models via gradient penalization in latent space. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 21046–21054, 2024.

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., et al. Toy models of superposition. *arXiv* preprint arXiv:2209.10652, 2022.

Fel, T., Boutin, V., Bethune, L., Cadene, R., Moayeri, M., ´
Andeol, L., Chalvidal, M., and Serre, T. A holistic ap- ´ proach to unifying automatic concept extraction and concept importance estimation. In Advances in Neural Information Processing Systems, volume 36, pp. 54805–54818. Curran Associates, Inc., 2023a.

Fel, T., Picard, A., Bethune, L., Boissin, T., Vigouroux, D., Colin, J., Cadenc, R., and Serre, T. Craft: Concept ´ recursive activation factorization for explainability. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2711–2721, Vancouver, BC, Canada, June 2023b. IEEE. ISBN 979-8-3503-01298. doi: 10.1109/CVPR52729.2023.00266.

Graziani, M., Nguyen, A.-P., and Mahony, L. O. Concept discovery and dataset exploration with singular value decomposition. 2023.

Haufe, S., Meinecke, F., Gorgen, K., D ¨ ahne, S., Haynes, ¨
J.-D., Blankertz, B., and Bießmann, F. On the interpretation of weight vectors of linear models in multivariate neuroimaging. *NeuroImage*, 87:96–110, February 2014. ISSN 1053-8119.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J.,
Viegas, F., and Sayres, R. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). June 2018. arXiv: 1711.11279.

Kindermans, P.-J., Schutt, K. T., Alber, M., M ¨ uller, K.-R., ¨
Erhan, D., Kim, B., and Dahne, S. Learning how to ¨ explain neural networks: Patternnet and patternattribution. *arXiv:1705.05598 [cs, stat]*, October 2017. arXiv:
1705.05598.

Kingma, D. P. Adam: A method for stochastic optimization.

arXiv preprint arXiv:1412.6980, 2014.

Lim, H., Choi, J., Choo, J., and Schneider, S. Sparse autoencoders reveal selective remapping of visual concepts during adaptation. (arXiv:2412.05276), December 2024. doi: 10.48550/arXiv.2412.05276. arXiv:2412.05276 [cs].

Monfort, M., Andonian, A., Zhou, B., Ramakrishnan, K.,
Bargal, S. A., Yan, T., Brown, L., Fan, Q., Gutfruend, D., Vondrick, C., et al. Moments in time dataset: one million videos for event understanding. IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–8, 2019.

ISSN 0162-8828.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Nanda, N., Lee, A., and Wattenberg, M. Emergent linear representations in world models of self-supervised sequence models. In Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, BlackboxNLP@EMNLP 2023, Singapore, December 7, 2023, pp. 16–30. Association for Computational Linguistics, 2023.

Ng, A. et al. Sparse autoencoder. *CS294A Lecture notes*, 72
(2011):1–19, 2011.

Pahde, F., Dreyer, M., Samek, W., and Lapuschkin, S. Reveal to revise: An explainable ai life cycle for iterative bias correction of deep models. In Medical Image Computing and Computer Assisted Intervention - MICCAI 2023, Lecture Notes in Computer Science, pp. 596–606, Cham, 2023. Springer Nature Switzerland. ISBN 978-3031-43895-0.

Pahde, F., Dreyer, M., Weber, L., Weckbecker, M., Anders, C. J., Wiegand, T., Samek, W., and Lapuschkin, S. Navigating neural space: Revisiting concept activation vectors to overcome directional divergence, 2024.

Pfau, J., Young, A. T., Wei, J., Wei, M. L., and Keiser, M. J. Robust semantic interpretability: Revisiting concept activation vectors. In *Fifth Annual Workshop on Human* Interpretability in Machine Learning (WHI), ICML 2020, 2020, 2020.

Ramakrishnan, K., Monfort, M., McNamara, B. A., Lascelles, A., Gutfreund, D., Feris, R. S., and Oliva, A. Identifying interpretable action concepts in deep networks. In CVPR Workshops, pp. 12–15, 2019.

Sharkey, L., Braun, D., and Millidge, B. Taking features out of superposition with sparse autoencoders, 2022.

Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I. J., and Fergus, R. Intriguing properties of neural networks. In 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014.

Yun, Z., Chen, Y., Olshausen, B. A., and LeCun, Y. Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors. (arXiv:2103.15949), April 2023. doi: 10.48550/arXiv.2103.15949. arXiv:2103.15949 [cs].

Zhang, R., Madumal, P., Miller, T., Ehinger, K. A., and Rubinstein, B. I. Invertible concept-based explanations for cnn models with non-negative concept activation vectors. In *Proceedings of the AAAI Conference on Artificial* Intelligence, volume 35, pp. 11682–11690, 2021.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., and Torralba, A. Places: A 10 million image database for scene recognition. IEEE transactions on pattern analysis and machine intelligence, 40(6):1452–1464, 2017.

Zhou, B., Sun, Y., Bau, D., and Torralba, A. Interpretable basis decomposition for visual explanation. In European Conference on Computer Vision (ECCV), pp. 119–134, 2018.

## A. Unsupervised Interpretable Basis Extraction And Concept-Basis Extraction Losses

Sparsity Loss (L
s) (Doumanoglou et al., 2023)
550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604

## B. Signal Direction Estimation

$${\mathcal{L}}_{\mathbf{p}}^{s}=-\sum_{i}q_{\mathbf{p},i}\mathrm{log}_{2}q_{\mathbf{p},i},\quad q_{\mathbf{p},i}={\frac{y_{\mathbf{p},i}}{\sum_{i}y_{\mathbf{p},i}}}$$
yp,i(10)
and the aggregated sparsity loss L
s:

$$(10)$$
$${\mathcal{L}}^{s}=\mathbb{E}_{\mathbf{p}}\big[{\mathcal{L}}_{\mathbf{p}}^{s}\big]$$
$$(111)$$
$$(12)^{\frac{1}{2}}$$
(11)
Maximum Activation Loss (L
ma) (Doumanoglou et al., 2023)
With the complement of this loss the pixel classifications are enforced to become binary:

$${\mathcal{L}}^{m a}=\mathbb{E}_{\mathbf{p}}\bigg[-\sum_{i}q_{\mathbf{p},i}\mathrm{log}_{2}y_{\mathbf{p},i}\bigg]$$
i
yp,ii(12)
Inactive Classifier Loss (L
ic) (Doumanoglou et al., 2024)
This loss ensures that each classifier in the set, classifies positively at least ν ∈ [0, 1] percent of pixels in the concept dataset.

$${\mathcal{L}}^{i c}=\mathbb{E}_{i}\Big[{\frac{1}{\nu}}\mathrm{ReLU}\big(\nu-\mathbb{E}_{\mathbf{p}}[y_{\mathbf{p},i}^{\gamma}]\big)\Big]$$
p,i]i(13)
with ν =
τ I
, γ > 1, γ ∈ R
+ denoting a sharpening factor and τ ∈ [0, 1] denoting a percent of pixels in the dataset to be evenly distributed among the I classifiers in the set. Maximum Margin Loss (L
mm) In the original formulation of (Doumanoglou et al., 2023), the Maximum Margin Loss was defined as L
mm =1M with M being a single parameter for the whole set of classifiers since the optimization was performed in the standardized space with shared parameters for the margins M and biases b. In this work, we removed the standardized space constraints and instead, we have a margin parameter Mi for each classifier in the set. Thus, we modify the Maximum Margin loss to become:

$${\mathcal{L}}^{m m}={\frac{1}{I}}\sum_{i}{\frac{1}{M_{i}}}$$
$$(13)^{\frac{1}{2}}$$

$$(14)$$

Based on the observation that the number of semantic labels that may be attributed to an image's patch, are only a fraction of the set of possible semantic labels, this loss enforces sparsity across the classification results yp,i for each spatial representation xp. In particular, the sparsity loss for pixel p is defined as:
In the formulation of signal-distractor data-model defined in (Kindermans et al., 2017) and detailed in Section 3.2, (1) is an accurate estimator of a concept's encoding direction whenever the non-signal (here distractor) components in the data contain information independent of whether the representation xp belongs to concept i. The formula exploits the fact that cov[zp, d] should be 0. An important role in this discussion has the threshold bithat delimits the positive samples of a concept. When considering the encoding of multiple concepts, like the data model that we proposed in Section 4.1, it is reasonable to make the assumption of (Doumanoglou et al., 2023) that concept label attributions are mutually-exclusive
(e.g. when an image patch corresponds to concept *tree* it is not car or sky or ...). Thus, the signal values ap,i, ap,j in xp may not be independent but anti-correlated since for a pixel p of concept i ap,i > bi and ap,j < bj , possibly violating the assumptions of (1) regarding independence. Whether the violation is significant or not may depend on the relationship between bi and the mean of ap,i as well as whether we consider a balanced dataset. This assumption is not violated though, if we consider only the reference samples that belong to the concept. In that case, among that subset of the data, the signal values ap,i and ap,j are now independent by assumption, as we now removed the biases bi, bj due to sub-sampling. This allows us to still consider (1) as a signal estimator, even under the extended data model of multiple concepts, provided that we subsample the data based on their concept label.

## C. Extracting Signal Values With The Filters Of Concept Detectors

Starting from the data model of Section 4.1 for the encoding of multiple concepts in the representation, we have:

$$x_{p}=S\alpha_{p}+D\beta_{p}$$

In the experiment on Synthetic Data (Section 5.1) we introduced an additional constant bias of 10 to bring all the feature representation in the positive quartile. This changed the above data model to:
The latter is an estimator of ap,i with respect to the mean Ep[ap,i] and with an error term depending on distractors, irrespective of constant bias.

## D. Direction Learning Process

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 We learn the directions of the proposed method in a a four-step process: a) we first learn the parameters wi, bi following (Doumanoglou et al., 2024), replacing the *CNN Classifier Loss* with proposed L
uur; b) we then continue optimizing wi, bi, removing the orthogonality and standardization constraints while incorporating the additional losses from Section 4.2; c)
next, we learn the signal vectors using the filters of the learned classifiers as regressors in (1) to initialize {Sˆ}; and d) finally, we jointly optimize sˆi, wi, bi, using all previous losses, replacing L
uur with L
cur and adding L
f so from Sections 4.3 and 4.4.

$$z_{\mathbf{p}}=\mathbf{w}_{i}^{T}\mathbf{x_{p}}=\mathbf{w}_{i}^{T}\mathbf{S}\alpha_{\mathbf{p}}+\mathbf{w}_{i}^{T}\mathbf{D}\beta_{\mathbf{p}}$$ $${\frac{z_{\mathbf{p}}}{\mathbf{w}_{i}^{T}\mathbf{s}_{i}}}=a_{\mathbf{p},i}+{\frac{\mathbf{w}_{i}^{T}\mathbf{D}\beta_{\mathbf{p}}}{\mathbf{w}_{i}^{T}\mathbf{s}_{i}}}$$

$$(15)$$

Thus, we can estimate the signal value, by the inner product between wi and xp and divide by wT
i sˆi.

$$x_{p}=S\alpha_{p}+D\beta_{p}+\mu$$

with µ ∈ R
D denoting the constant bias, equal to an element-wise repetition of 10. In this case, when extracting the signal value, the estimation becomes:

$$z_{\mathbf{p}}=\mathbf{w}_{i}^{T}\mathbf{x_{p}}=\mathbf{w}_{i}^{T}\mathbf{S}\mathbf{\alpha_{p}}+\mathbf{w}_{i}^{T}\mathbf{D}\mathbf{\beta_{p}}+\mathbf{w}_{i}^{T}\mathbf{\mu}$$ $$\frac{z_{\mathbf{p}}}{\mathbf{w}_{i}^{T}\mathbf{s}_{i}}=a_{\mathbf{p},i}+\frac{\mathbf{w}_{i}^{T}\mathbf{D}\mathbf{\beta_{p}}}{\mathbf{w}_{i}^{T}\mathbf{s}_{i}}+\frac{\mathbf{w}_{i}^{T}\mathbf{\mu}}{\mathbf{w}_{i}\mathbf{s}_{i}}$$
$$(16)$$

We see that this extra bias that we used, introduces an additional constant error term when estimating the signal values. In a real-world scenario when this µ is unknown, we can use the following estimator aˆp,i which depends on the average of features xp:

aˆp,i = wT i xp wT isi − wT i  Ep[xp] wT isi = aˆp,i = wT i xp wT isi − wT i siEp[ap,i] wT isi − wT i DEp[βp] wT isi aˆp,i = ap,i − Ep[ap,i] +  wT i D wT isi (βp − Ep[βp])

$$(17)$$
As discussed in Section 4.3, signal values, which are required to estimate the encoding direction of a concept, are extracted using the filter weights of the concept detectors. Yet, as we discussed in that Section, in order for this to happen, the filter weights wi need to be orthogonal to sj and D. Since we do not explicitly estimate distractors in this work, there maybe an innevitable error when extracting the value of the signal (we say maybe, because this might also be mitigated by the Uncertainty Region Alignment losses). Here we study on the order of this error.

Table 6. Left: Data matrices S and D for the experiment on synthetic data. Right: Cosine similarities for every pair of vectors in S, D, i.e.: C

T C, C = [S|D].

S

0.5396 0.5914 0.8122 0.4415 0.5833 0.2983

-0.1283 -0.1745 0.4681 -0.0093 -0.4899 0.17

0.59 -0.193 -0.0005

-0.3718 -0.0051 0.0229

0.1051 0.0467 -0.0524 0.0003 -0.0103 0.0056 0.0063 0.0037 -0.0021

-0.0009 0.0004 0.001 -0.0004 -0.0002 0.0002

0.0024 -0.0001 -0.0006 0.0036 0.0001 -0.0013

-0.0007 0.0 0.0002

0.0002 -0.0 -0.0

-0.0001 -0.0 -0.0

| D       |         |                     |        |        |
|---------|---------|---------------------|--------|--------|
| 0.7693  | 0.7527  |                     |        |        |
| -0.0396 | -0.6147 |                     |        |        |
| -0.6216 | 0.1661  |                     |        |        |
| 0.1416  | -0.1293 |                     |        |        |
| -0.0123 | 0.1065  |                     |        |        |
| -0.007  | -0.0042 |                     |        |        |
| 0.0022  | 0.0003  |                     |        |        |
| 0.0014  | 0.0007  |                     |        |        |
| -0.0004 | -0.0001 |                     |        |        |
| 0.0001  | 0.0001  |                     |        |        |
| -0.0    | 0.0     |                     |        |        |
| 0.0     | 0.0     |                     |        |        |
| 0.0     | 0.0     |                     |        |        |
| -0.0    | 0.0     |                     |        |        |
| 0.0     | -0.0    |                     |        |        |
| 0.0     | 0.0     | Cosine-Similarities |        |        |
| 1.0     | 0.4965  | 0.4939              | 0.4716 | 0.179  |
| 0.4965  | 1.0     | 0.4868              | 0.4735 | 0.1004 |
| 0.4939  | 0.4868  | 1.0                 | 0.3458 | 0.4836 |
| 0.4716  | 0.4735  | 0.3458              | 1.0    | 0.4805 |
| 0.179   | 0.1004  | 0.4836              | 0.4805 | 1.0    |

## E. Details For The Experiment On Synthetic Data

We train the network using cross-entropy loss and the Adam (Kingma, 2014) optimizer, with learning rate 0.005 and batch size 1024 for 2000 epochs. In principle, we follow the process defined in Section D, but due to the simplicity of the example, we omit step (b) and proceed directly from (a) to (c). We formulate the optimization of steps (a) and (d) using the Augmented Lagrangian Loss, essentially converting the problem to a constraint optimization one. This greatly stabilizes learning and avoids local optima. For step (a) we solve the constrained optimization problem of minimizing λ sL
s + λ uurL
uur with L
ma < 0.8, L
mm < 8 and L
ic < 0.1. For step (d) we minimize λ sL
s + λ sbL
sb + λ curL
cur with L
ma < 0.8, L
mm < 8, L
ic < 0.1, L
eac < 0.1, L
f so < 0.1. The learning rate we use for both steps is 0.00005 and the number of epochs are set to 20000. The loss weights λ are the same as in Table 7. The sharpening factor γ of L
ic is set to γ = 1.1, τ = 1. and the ρ of L
eac is set to ρ = 1.5/3.

The specific values of matrices S and D used in this experiment, and the cosine similarities between every pair of vectors are provided in Table 6.

## F. Details For The Experiment On Deep Image Classifier

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714

| Table 7. Loss weights used for the experiment on Resnet18/Places365. s λ sb λma λmm λ ic λ uur λ cur λ eac λ f so λ step (a) 2.6 - 2.8 0.6 5.0 0.25 - - - step (b) 0.85 2.6 2.8 0.6 15.0 0.25 - 15.0 - step (d) 0.85 2.6 2.8 0.6 15.0 - 0.25 15.0 1.0   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

For step (a) direction learning lasts 800 epochs using an initial learning rate of 0.001 for a reference batch size of 4096 (which, in all steps, we scale based on the available GPU memory). We reduce the learning rate on plateau, by a factor of 0.5 with patience and cooldown set to 10 epochs. Step (b) lasts for 2000 epochs with initial learning rate of 0.0001 for the same reference batch size. We also reduce the learning rate on plateau by a factor of 0.5 but with patience and cooldown set to 50 epochs. For both steps (a) and (b) we use τ = 0.9 for L
ic. For L
eac we use ρ = 12*τ /I*, chosen to roughly match the maximum number of pixels in any of the Broden classes. Step (d) lasts for another 2000 epochs with initial learning rate of 0.0005 with the τ hyper-parameter of L
ic set to 0.2 and ρ = 70*τ /I*. The rest of the parameters remain intact with respect to step (b). For RSW we use ν = 4.0. When using L
uur and L
cur, we observed better results when manipulating features with a stochastic magnitude in the direction dxp, i.e. shifting representations as x
′p = xp − κdxp with κ a random number in [0.5, 0.9]. Table 7, summarizes the loss weights that we used for steps (a), (b) and (d). In practice, we separate filter directions from their magnitude 1/Mi and learn them independently as suggested in (Doumanoglou et al., 2024). For enforcing ||wi||2 = 1 (i.e. unit norm filter vectors) we use parametrization on the unit hyper-sphere.

When learning the supervised classifiers to compared against, for each concept, we construct a dataset comprised of negative samples that are up-to 20 times more than the number of positive samples, as a means to mitigate the great imbalance. The supervised concept classifiers are learned for the labels assigned to our concept detectors at the Direction Labeling phase (Section 3.4). For RCAV's perturbation hyper-parameter, we use α = 5. For direction significance testing, we use RCAV's label permutation test. To construct random noise signal vectors, we (a) construct a dataset of feature-label pairs based on the decision rule of each one of the concept detectors. To deal with great class imbalance, we construct a pool of negative samples that is at most 20 times more than the positive ones (b) we construct N noisy versions of that dataset by label permutation (c) we learn a noise-classifier to distinguish features based on the permuted labels, and (d) we concurrently, estimate a noise-signal vector using (1) and the conditions described in Section 4.3. To learn each one of the noise signal vectors and before permuting the labels, we construct a balanced dataset of at most 5000 samples, picked randomly from the pool. We train the noise classifiers using Adam for 100 epochs and a learning rate 0.01. By using noise signal vectors as RCAV's noisy directions, and with the number of those vectors per classifier set to N = 100, we subsequently calculate RCAV's p-values. We apply Bonferroni correction to all p-values, by diving the significance threshold 0.05 with the number of concept detectors I and the number of model classes (K = 365).

## F.1. Detailed Interpretability Comparison Against The Supervised Approach For The Experiment On Deep Image Classifier

Figure 4 plots a histogram of classification metric differences between the *Linear-OR* set of classifiers and the classifiers learned in a supervised way. The differences are based on the labels, effectively taking the difference of metrics that regard two classifiers (the first from the *Linear-OR* set and the second from (Zhou et al., 2018)) with the same concept name. Figures 5, 6, 7 depict concrete binary classification metrics for some of the concept detectors in the *Linear-OR* set of classifiers, comparing them with concept classifiers learned with supervision.

715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824

825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879
Figure 6. Interpretability Comparison. Exact Precision/Recall/F1Scores for specific concepts in Broden: comparison between the *linear-or* set of classifiers learned with the proposed method (EDDP, I = 500) and classifiers learned in a supervised way (IBD: (Zhou et al.,
2018)).

880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934

## F.2. Detailed Influence Metrics And Diagrams For The Experiment On The Deep Image Classifier

935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 Table 8 provides more summarizing influence statistics regarding the signal vectors learned with the proposed method.

In that table, SCi,j denotes RCAV's sensitivity score in the direction of the i-th signal vector, for the network's class j.

Figures 8,9,10,11 depict concrete examples of how each concept's signal direction impacts the Resnet18's class predictions. Concepts appearing more than once. correspond to different directions that have been attributed the same label by Network Dissection. Seemingly irrelevant concepts with positive influence may have three possible explanations: a) the network has some sensitivity to those concepts (as it's top1 accuracy is 56.51%) b) their impact might be low, since RCAV only considers the sign of the class prediction difference before and after the manipulation, regardless of its magnitude, (thus those concepts may influence the prediction class positively, but by only a small amount) and c) their label may be misleading as the respective concept detectors do not reliably predict the concept (i.e. they exhibit a low IoU score).

Table 8. This table summarizes statistics of the RCAV's sensitivity score matrix SC for the set of directions learned with L
uur or L
cur + L
fso, I = 500, RCAV α = 5.0. All entries in the sensitivity score matrix SC are masked for significance before computing the

| statistics. Sensitivity scores were obtained using signal vectors calculated using (1). Metric Formula   | L                     |        |    |
|----------------------------------------------------------------------------------------------------------|-----------------------|--------|----|
| Significant Direction Count                                                                              | 359                   | 376.0  |    |
| Significant Class-Direction Pairs                                                                        | 2118                  | 3271.0 |    |
| Directions /w Positive Influence                                                                         | P                     |        |    |
| Directions /w Negative Influence                                                                         | P                     |        |    |
| Positively Impactful Directions Per Class                                                                | 1 K                   |        |    |
| Negatively Impactful Directions Per Class                                                                | 1 K                   |        |    |
| Minimum # of Positively Influencing Classes Across Directions                                            | mini P j 1x>0(SCi,j ) | 0      | 0  |
| Maximum # of Positively Influencing Classes Across Directions                                            | maxi P j 1x>0(SCi,j ) | 16     | 13 |
| Minimum # of Negatively Influencing Classes Across Directions                                            | mini P j 1x<0(SCi,j ) | 0      | 0  |
| Maximum # of Negatively Influencing Classes Across Directions                                            | maxi P j 1x<0(SCi,j ) | 27     | 46 |
| # of Classes /w at Least One Positively Impactful Direction                                              | P j 1x>1 P i          |        |    |
| # of Classes /w at Least One Negatively Impactful Direction                                              | P j 1x>1 P i          |        |    |

990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 Figure 8. Concept Influence Diagram for Resnet18 trained on Places365. The depicted concepts have sensitivity scores above 0.99 in absolute terms. (We use RCAV to quantify the sensitivity, and re-scale the score to [−1, 1]) Positive influencing and negative influencing concepts are provided. The number of concepts have been limited to 10. When concepts appear more than once, they correspond to different signal directions (as labeling the classifiers with NetDissect may assign the same concept name to more than one directions.) Figure 9. Concept Influence Diagram for Resnet18 trained on Places365. The depicted concepts have sensitivity scores above 0.99 in absolute terms. (We use RCAV to quantify the sensitivity, and re-scale the score to [−1, 1]) Positive influencing and negative influencing concepts are provided. The number of concepts have been limited to 10. When concepts appear more than once, they correspond to different signal directions (as labeling the classifiers with NetDissect may assign the same concept name to more than one directions.)
1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081 1082 1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099