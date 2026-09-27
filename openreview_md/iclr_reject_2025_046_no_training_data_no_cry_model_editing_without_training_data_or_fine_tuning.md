# No Training Data, No Cry: Model Editing With- Out Training Data Or Finetuning

Anonymous authors Paper under double-blind review

## Abstract

Model Editing(ME)–such as classwise unlearning and structured pruning–is a nascent field that deals with identifying editable components that, when modified, significantly change the model's behaviour, typically requiring fine-tuning to regain performance. The challenge of model editing increases when dealing with multibranch networks(e.g. ResNets) in the data-free regime, where the training data and the loss function are not available. Identifying editable components is more difficult in multi-branch networks due to the coupling of individual components across layers through skip connections. This paper addresses these issues through the following contributions. First, we hypothesize that in a well-trained model, there exists a small set of channels, which we call HiFi channels, whose input contributions strongly correlate with the output feature map of that layer. Finding such subsets can be naturally posed as an expected reconstruction error problem. To solve this, we provide an efficient heuristic called RowSum. Second, to understand how to regain accuracy after editing, we prove, for the first time, an upper bound on the loss function post-editing in terms of the change in the stored BatchNorm(BN) statistics. With this result, we derive BNFix, a simple algorithm to restore accuracy by updating the BN statistics using distributional access to the data distribution.

With these insights, we propose retraining free algorithms for structured pruning and classwise unlearning, CoBRA-P and CoBRA-U, that identify HiFi components and retains(structured pruning) or discards(classwise unlearning) them. CoBRA-P achieves at least 50% larger reduction in FLOPS and at least 10% larger reduction in parameters for similar drop in accuracy in the training free regime. In the training regime, for ImageNet, it achieves 60% larger parameter reduction. CoBRA-U achieves, on average, a 94% reduction in forget-class accuracy with a minimal drop in remaining class accuracy.1

## 1 Introduction

The improved performance of deep learning models on various tasks (Krizhevsky et al., 2012; Ioffe & Szegedy, 2015; He et al., 2016) has increased their adoption. However, such models may not always be suitable for direct use in various applications. For instance, a pre-trained classification model might not run on an edge device without compressing it using a technique such as pruning (Prakash et al., 2019). We use the term *Model Editing* to refer to such modifications. This work focuses on two model editing tasks - pruning and classwise unlearning for vision tasks. Pruning (LeCun et al., 1989; Hoefler et al., 2021) is one of the methods to improve latencies and memory requirements of models during inference. Pruning involves discarding "unimportant" components of a model, such as weights, neurons, or channels. This work focuses on structured pruning (Luo et al., 2017; Wang et al., 2020b; Shen et al., 2022) that discards entire channels in Convolution Neural Networks (CNNs) as opposed to unstructured pruning (LeCun et al., 1989; Han et al., 2015; Tanaka et al., 2020) that discards weights individually. Classwise unlearning (Jia et al.,
2023) refers to the task where the goal is to unlearn training data points of an entire class while maintaining the predictive performance on remaining classes. Classwise unlearning can be efficiently performed using pruning (Jia et al., 2023). Editing tasks such as pruning and classwise unlearning require an understanding of the components 1 000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 of a model - such as weights, neurons, or convolutional filters that contribute significantly to its prediction (Räuker et al., 2023). This becomes more challenging when dealing with modern neural networks that consist of skip connections (He et al., 2016; Huang et al., 2017) that couple elements between layers (Liu et al., 2021a; Fang et al., 2023). However, this is not generally addressed in relevant works (Jia et al., 2023; Ding et al., 2021; Joo et al., 2021; Luo et al., 2017). Editing algorithms often take a toll on the original task performance and thus rely on retraining to alleviate this (Luo et al., 2017; Wang et al., 2020b; Jia et al., 2023). However, retraining requires significant computational resources and access to the loss function & training set pertaining to the original task. It is not uncommon for the relevant training set & loss function to be unavailable due to privacy or commercial concerns (Yin et al., 2020), making retraining more challenging. Most existing works either assume access to data and finetune models (Jia et al., 2023; Wang et al., 2020b; Shen et al., 2022) or assume the absence of training data and do not finetune (Narshana et al., 2022; Murti et al., 2022; Tanaka et al., 2020). However, the gap between the accuracy of data-free and data-driven methods is significant (Hoefler et al., 2021). Thus, it is important to bridge this gap. For model editing, this work, similar to Murti et al. (2022), assumes access to samples with similar distributional properties to that of the training set. For instance, to construct a cat-dog classifier, a training set could be a large collection of images of cats and dogs taken from a private image repository, while samples available via distributional access could be the photos of cats and dogs taken from a personal device. We use this distributional information to study CNNs with Batchnorm layers (Ioffe & Szegedy, 2015). Batch Normalization, a popular deep learning technique developed to decrease training time, is used in many successful architectures like ResNets (He et al., 2016), VGGs (Simonyan & Zisserman, 2015), and MobileNet (Howard, 2017). Existing theoretical analysis of Batch Normalization has focused on understanding its effect during training (Santurkar et al., 2018); however, to the best of our knowledge, there has been little insight into its effect on the loss function during inference upon model perturbation. Towards addressing the challenges presented above, the following are our contributions: 1. It is important for model editing to understand what components of a well-trained model are necessary for predictions. To address this, we propose the notion of High-Fidelity(HiFi) components, components of the network that contribute significantly to the output of the corresponding layer. Using this notion, we hypothesize that in each layer of a well-trained model, the set of HiFi components are responsible for the model's performance, which we empirically validate in Section 7. Thus, the problem of model editing boils down to identifying HiFi components.

2. Towards identifying HiFi components in a layer for model editing without access to training data or the loss function, We use correlation as the measure of similarity between the distribution of the input channel's contribution to the output and the distribution of the output. In Section 4, we show that this choice of similarity naturally connects HiFi components to those with low expected reconstruction error, a popular saliency measure in pruning. However, this problem is NP-Hard, and the use of a heuristic called RowSum is required to solve this problem. This enables the identification of editable components using distributional access.

3. Typically, editing causes a degradation in the model's performance. To understand the impact of BatchNorm parameters on this degradation, we derive a connection between the learned parameters of BatchNorm layers and the loss function. We show that the loss function can be upper bounded by a quadratic function of the learned parameters of the BatchNorm layer. We state this formally in Theorem 1. Based on our analysis, we propose Algorithm 2, called BNFix, an algorithm requiring only distributional access to modify the stored statistics in a BatchNorm layer to reduce performance degradation due to model editing. We observe an interesting phenomenon, which we call **BN Recall**, when applying BNFix as a replacement for retraining using remaining class examples - applying BNFix on a model whose forget accuracy has significantly fallen using only remain class samples causes the forget class accuracy to increase significantly.

4. In addition to identifying HiFi components and BNFix, we use fidelity compensation - where we improve the fidelity of the feature maps via weight rescaling - to design the CoBRA family of editing algorithms and analyze this improvement in Theorem 2. CoBRA(Correlation-based editing with Batchnorm Re-Adjustment) is an editing scheme that identifies HiFi components in each layer of a network to either retain(CoBRA-P) or discard(CoBRA-U), and recovers model performance by BNFix and weight compensation. Our experiments show that CoBRA-P achieves at least 50% larger reduction in FLOPS and at least 10% larger reduction in parameters for similar 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

## 2 Preliminaries

Notation Let a ∈ R
n denote an n-dimensional vector whose i th element is ai, and B ∈ R
n×m a matrix with n rows and m columns whose i th row is bi ∈ R
m. For p ∈ N, let [p] = {1*, . . . , p*}.

For matrices, A, B ∈ R
n×m, we define ⟨A, B⟩ = Tr(A⊤B) and frobenius ∥A∥
2F = ⟨A, A⟩.

For tensors A, B ∈ R
C×K×K, we define ⟨A, B⟩ =PC
i=1⟨Ai, Bi⟩, where Ai, Bi ∈ R
K×K and ||A||2 = ⟨A, A⟩. For a vector v, diag(v) is a diagonal matrix whose i th entry is vi. Topp(v) denotes a function that which returns the indices of the elements in the top p th-percentile of v.

Neural Network Preliminaries Let fθ be a neural network with parameters θ with L layers.

Consider data drawn from a distribution PD, we use X as a random variable drawn from this distribution. We use Lθ(x) as the loss function evaluated with parameters θ on a point x and parameters are trained to minimize the expected loss over the distribution. The parameters are grouped into structural units, such as convolutional filters in CNNs, and are stacked in layers. We refer to such structures as *components* of the network. The structures and the operations performed on the input by these structures form the architecture of the network. 2D Convolution Let the l th layer of a network be a 2D convolution layer with c lin input channels and c l out output channels whose weights are Wl ∈ R
c l out×c l in×k×k, where k is the kernel size. Let the input to the convolution layer be Φ
l(x) ∈ R
c l in×h l−1×w l−1, and the output Y
l(x) ∈ R
c l out×h l×w l, where h l−1, hland w l−1, wlrepresent the heights and widths of the input and output respectively.

The c th output channel, Y
l cis then,

$$\mathbf{Y}_{c}^{l}(\mathbf{x})=\sum_{i=1}^{c_{in}^{l}}\mathbf{\Phi}_{i}^{l}(\mathbf{x})*\mathbf{W}_{ci}^{l}=\sum_{i=1}^{c_{in}^{l}}\mathbf{A}_{ci}^{l}(\mathbf{x})\tag{1}$$
$$(1)$$
$$(2)$$

Batch Normalization during inference Let the l th layer of a neural network be a a BatchNorm layer with dimension m whose input is y l(x) ∈ R
m, parameterized by two stored statistics, mean µ ∈ R
m and standard deviation σ ∈ R
m, and two learned parameters, shift β ∈ R
m and scale γ ∈ R
m. The c th output of the layer during inference, v l(x) ∈ R
m, is given by

$$\mathbf{v}^{l}(\mathbf{x})=\mathbf{G}\mathbf{z}^{l}(\mathbf{x})+\mathbf{\beta}\;\;\mathrm{where}\;\;z_{c}^{l}(\mathbf{x})={\frac{y_{c}^{l}(\mathbf{x})-\mu_{c}}{\sigma_{c}}}$$

σc(2)
where G = diag(γ). The stored statistics are meant to estimate the mean and standard deviation of y l(X) from the *training data*. Additional details are in Appendix C.

drop in accuracy in the training free regime. In the training regime, for ImageNet, it achieves 60% larger parameter reduction. CoBRA-U achieves, on average, a 94% reduction in forget-class accuracy with a minimal drop in remain class accuracy.

where ∗ denotes the convolution operation. We say Alci(x) ∈ R
h l×w lis the *input contribution* of input channel i to output channel c; this is illustrated in Figure 1a.

## 3 The Problem Of Editing Well-Trained Models Without Training Data

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Model editing refers to techniques that selectively change the model parameters to modify its statistical behaviour (Jia et al., 2023; Santurkar et al., 2021; Shah et al., 2024), motivated by issues such as privacy and GDPR regulations (Bourtoule et al., 2021; Nguyen et al., 2022). Editing encompasses a wide variety of tasks, including debiasing (Jain et al., 2022), selective unlearning (Golatkar et al., 2020), network scrubbing (Kurmanji et al., 2024), and lifelong learning (Sahoo et al., 2024; Golkar et al., 2019). Recently, *component attribution* - that is, identifying components responsible for predictions - has gained traction for model unlearning (Shah et al., 2024; Wang et al., 2022; Kodge et al.). However, it is challenging to use model editing without the loss function and training data (Shah et al., 2024), as well as for analyzing models with complex interconnections (Narshana et al., 2022; Liu et al., 2021a). Extensive related work is cited in Appendix A. In this section we formalize the problem of Model Editing via pruning.

What is Model Editing? Consider the model fθ0, and let Di, i ∈ [M] be conditional data
distributions, such as classes. Our goal is to *edit* the model by removing entire components. That is,
given the weights of the well-trained model θ0, we edit θ0 to θE = θ0 − θ
⋆, where θ
⋆ ∈ SB := {θ ∈
R

d: count(fθ0−θ) = B} ⊂ R
d by editing the parameters of at most Ctotal − B *components, where*
Ctotal *is the total number of components in the network (i.e., convolutional filters) by solving*
$$\theta^{\star}=\operatorname*{arg\,min}_{\theta\in S_{B}}\;\sum_{i}\mathbb{E}_{X\sim{\mathcal{D}}_{i}}\left[\alpha_{i}\left({\mathcal{L}}_{\theta_{0}-\theta}(X)-{\mathcal{L}}_{\theta}(X)\right)\right],$$
[αi (Lθ0−θ(X) − Lθ(X))] , (Edit)
$$(\mathbb{E}\mathrm{dil})$$
where αi ∈ R are multipliers to weight tasks, depending on whether we want the model to increase the loss or decrease it on the corresponding distribution Di. While a variety of tasks can be classified as model editing (Shah et al., 2024); in this work, we address the problems of **structured pruning** and **classwise unlearning.** Structured Pruning, in the setting of equation Edit, is when M = 1, and α1 = 1. Thus, we write θ
⋆ = arg min θ∈SB
EX∼D [(Lθ0−θ(X) − Lθ(X))] , (Prune)
Classwise unlearning involves removing the model's ability to make accurate predictions on a chosen class, called the **forget class** with distribution Df , while maintaining the statistical performance on the remaining classes - called the **remain classes**, with distribution Dr. In the setting of equation Edit, we have M = 2, D1 = Df , D2 = Dr, α1 = −1 and α2 = κ > 0. Solving this problem ensures that the loss on Df increases, while the loss on Dr decreases, with κ penalizing the extent to which EX∼Df
[Lθ0−θ(X)] is allowed to increase. We write this as θ
⋆ = arg min θ∈SB
EX∼Dr[κ (Lθ0−θ(X) − Lθ(X))] − EX∼Df[Lθ0−θ(X) − Lθ(X)] . (Forget)
Challenges in editing models without the training data or loss function? Unlike works such as Jia et al. (2023) and the references therein, fine-tuning or retraining the model is not possible in this setting. Thus to effectively edit the behavior of a network, it is necessary to identify the components that are responsible for making predictions. These can be characterized as components which when modified, significantly change the behaviour of the network. The key challenge is thus: Problem Statement: Solve equation Prune or equation Forget **without access to original**
training data and loss function which was used to obtain θ0.

It is well known that pruning or perturbing a large number of components significantly affects statistical performance (Hoefler et al., 2021). Thus, it is necessary to identify a *small subset of* editable components; components which are **editable** can be removed to aid an editing task. In the case of pruning, components that have no effect on the performance of the model are editable, whereas for model unlearning, components required only for the prediction of the forget class are editable. We use this insight to pursue the stated problem and develop algorithms to address it.

## 4 Indentifying Editable Components Through Hifi Components

As stated in the previous section, editing well-trained models without access to the training data or loss function requires identifying components that have a disproportionate impact on the models's 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 Figure 2: Comparision of the fidelity scores of two different layers

## 4.2 High Fidelity Components And The Fidelity Score

Suppose Y
l+1(X) is the feature map generated by layer l + 1, and suppose A
l+1 i(X) is the ith input contribution, as defined in equation 1. We say the i-th component in layer l is a **high-fidelity (HiFi)**
component if the distribution of the input contribution A
l+1 i(X), D
l+1 iin layer l + 1 is similar to the distribution of Y
l+1(X), Dl+1. HiFi components are those with input contributions that can reconstruct the aggregate feature map2. To capture this, we analyze the dissimilarity between the distributions of Yˆ l+1(X) = Y
l+1(X)−EX
-Y
l+1(X)and Aˆl+1 i(X) = A
l+1 i(X)−EX
-A
l+1 i(X).

We define FS(i), a *Fidelity score* that measures the similarity between an input contribution and the aggregate feature map, below.

$$\mathsf{FS}(i)=\mathsf{DIS}(\hat{\mathcal{D}}^{l+1},\hat{\mathcal{D}}_{i}^{l+1})=\left(\frac{\mathbb{E}_{X}\left[\|\hat{Y}^{l+1}(X)-\beta_{i}\hat{A}_{i}^{l+1}(X)\|^{2}\right]}{\mathbb{E}_{X}\left[\|\hat{Y}^{l+1}(X)\|^{2}\right]}\right)^{\frac{1}{2}}$$
$$(3)$$
$-2$
where $\beta_{i}=\mathbb{E}_{X}\left[\langle\hat{Y}^{l+1}(X),\hat{A}_{i}^{l+1}(X)\rangle\right]\mathbb{E}_{X}\left[\|\hat{A}_{i}^{l+1}(X)\|\right]^{\top}$
i−2

## 4.1 Which Features Are Distributionally Similar To The Output Features?

We provide the empirical observation that in many layers of deep networks, there are only a few filters for which the input contribution distribution is similar to that of the output distribution. In Figure 2, we show the relative reconstruction error after removing filters from a selection of layers of a ResNet50 trained on CIFAR10 - we use the expected reconstruction error as a measure of distributional similarity. We see that in well-trained models, a small subset of filters - between 5% and up to 30% of the number of filters in the layer - generate input contributions that are distributionally similar to the aggregate feature maps. This observation motivates us to edit models by identifying those components whose input contributions are distributionally similar to the feature maps. We call such components **High Fidelity (HiFi) components**, which we define in the sequel.

In the above definition, the smaller the value of DIS(Dˆl+1, Dˆl+1 i) (or higher the value of βi) is, better the reconstructability of Y in the mean-square sense. Furthermore, note that we can apply equation 3 on a channel-by-channel basis by considering the distributions of a single output feature map in a layer; we add an the additional subscript c to indicate that the feature map (and the input contribution) are generated by the cth component in the layer. In well-trained models we often observe that a small number of components have relatively lower DIS scores than the rest. Identifying such components is key to understanding the statistical behavior of model outputs, and hence will be the most critical insight for the subsequent development of our algorithms.

2This motivates the name HiFi components: Components whose sum can accurately reconstruct the output with Hi-Fidelity predictive performance. In this section, we propose the notion of High-Fidelity (HiFi) components, and hypothesize that HiFi components are what govern a model's predictive performance. We empirically validate our hypothesis and provide a template for model editing algorithms derived from it.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 highlights the relation between FS(i) and βi - if ∥Aˆl+1 i(X)∥ is roughly equivalent for all i, then FS(i)
is low when βiis large. Thus, a heuristic for identifying HiFi components is finding components for which βiis large. Moreoever, note that βi can be written as the sum of the elements of the row of a matrix, motivating the naming of the heuristic **RowSum**. Specifically, βiEX
h∥Aˆl+1 i(X)∥
2i=
Pj Qij , where Qij = EX
h⟨Aˆl+1 i, Aˆl+1 j⟩
i. We examine this in greater detail in Appendix E, along with an examination of the reconstruction error after the BatchNorm layers. Based on our empirical observations that a small subset of components in well-trained models generate input contributions that are distributionally similar to the feature maps, we now state the main hypothesis of our work. We validate our hypothesis empirically in Section 7.

Hypothesis 1. *Suppose we have a well-trained model with parameters* W = (W1, · · · , WL). We hypothesize that the HiFi components of this model contribute most to the predictions of the model, and those components that are not high fidelity can be discarded without affecting the performance of the model. Using HiFi Components for Model Editing Hypothesis 1 states that only the HiFi components - a small subset of the components in a layer - are responsible for the model's predictions. Thus, it facilitates model editing as the distributional similarity between input contributions and aggregate feature maps, as measured using equation 3, can be used as a surrogate for the impact of removing that component on the loss function. Thus, leveraging this hypothesis, we can either *prune* the HiFi components to increase the loss (for instance, for classwise unlearning tasks), or *retain* them to ensure the loss remains low (for instance, for structured pruning). We provide a generic algorithmic recipe for model editing using HiFi components, specialized for the tasks of classwise unlearning and structured pruning in Algorithm 1; these are discussed in greater detail in Section 6.

Algorithm 1: Model Editing by Identifying HiFi Channels Input: Model fθ0 with layer indices [L], layerwise budgets {Bl}l∈[L], data distributions D1, Df , Dr Output: Edited model fθE
for l ∈ [L] do Compute FS(i) using equation 3 on D, Df ,
Dr Determine which components to edit Recover accuracy on D

## 5 Bnfix: An Alternative To Retraining By Resetting Bn Statistics

In this section, we analyze BatchNorm1D in single branch networks during inference and how the change in distribution due to editing affects the relationship between the loss and BatchNorm parameters. Using this, we derive an algorithm to correct stored statistics after editing. This update has been previously employed in pruning literature (Frantar et al., 2022), but to the best of our knowledge, this is the first work to provide theoretical basis to the update in a distributional setting.

Analysis of BatchNorm at Inference BatchNorm at inference shifts the distribution of the intermediate representation at the output of a layer to have mean β and standard deviation γ. These are parameters of the model which are minimize a loss function L as described in Section 2 .We use the following fact to analyze the loss in terms of the intermediate representation at the output of a layer.

Fact 1 (**Stochastic Mean Value theorem).** Let f *be a twice differentiable real valued function from* R

dto R and Hf (x) *be the Hessian at any* x ∈ R
d*. For any point* c ∈ R
d *and a multivariate random* variable X ∈ R
d with finite second order moments, there exists a random variate t ∈ (0, 1) *such that*

$$f(\mathbf{X}+\mathbf{c})=f(\mathbf{c})+\nabla f(\mathbf{c})^{\top}\mathbf{X}+{\frac{1}{2}}\mathbf{X}^{\top}\mathbf{H}_{f}(\mathbf{c}+\mathrm{t}\mathbf{X})\mathbf{X}$$

For a proof, see Corollary 2 in Yang & Zhou (2021). Though for the case discussed here the above fact suffices, but one could potentially use similar facts which can be deduced from other techniques such as Delta Method (Benichou & Gail, 1989) or obtain a result on the expectation such as (Massey The Role of βi **and RowSum:** βiis a variant of the Tensor correlation between the input contribution A
l+1 iand the feature map Y
l+1. Furthermore, we can show that

$$\text{DIS}(\hat{\mathcal{D}}^{l+1},\hat{\mathcal{D}}^{l+1}_{i})^{2}=\mathbb{E}_{X}\left[\|\hat{Y}^{l+1}(X)\|^{2}-\beta_{i}^{2}\|\hat{A}^{l+1}_{i}(X)\|^{2}\right]/\mathbb{E}_{X}\left[\|\hat{Y}^{l+1}(X)\|^{2}\right]\tag{4}$$

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377
& Whitt, 1993). Using the optimality of the learned parameters of BatchNorm and Fact 1, we make some assumptions on the first and second-order derivatives on the loss for a well trained model in terms of the learned parameters of BatchNorm layers.

A.1 For a well-trained model ∇L(β) = 0, the gradient with respect to the shift parameter is zero.

A.2 For a fixed G and any β, we can bound the eigenvalues of the hessian with a constant K
for all inputs. Formally, ∥HL(GZ + β)∥2 ≤ K for all random variables Z ∈ R
dsuch that E(Z
2 i) = 1, E(Zi) = 0 for all β ∈ R
d. Here, the norm is the spectral norm of a matrix.

The assumptions A.1 and A.2 capture the model's "well-trained-ness" on the objective function L and follow from the first and second-order necessary conditions of optimality. We note that A.1 would not hold if the input distribution to the network was different from that of the training distribution. The constant K captures the smoothness of the loss function with respect to the parameter β and subsumes the effect of the rest of the network, which may contain more linear and non linear layers. Equipped with these assumptions about well-trained models, we derive a bound on the average loss over the learned distribution in terms of the learned parameters of the BatchNorm layer. With the observation that E[V (X)] = β, the term L(E[V (X)]) = L(β) reflects the loss of the averaged intermediate representation. Lemma 1 (Loss of a well trained model expressed with BatchNorm). *Consider a model that satisfies* assumptions 5. We can express an upper bound on the expected loss during inference in terms of the statistics of the output of the BatchNorm layer V (X) ∈ R
m.

$$|\mathbb{E}[{\mathcal{L}}(V(X))]-{\mathcal{L}}(\beta)|\leq{\frac{K}{2}}||\gamma||^{2}$$
||γ||2(5)
proof sketch. We prove this with fact 1 and using the statistics of V (X). A full proof of Lemma 1 can be found in D.1. How Editing affects BatchNorm We now study how editing affects the statistics of the output of the batch norm layer and the loss. Using lemma 1, we analyse the effect on the objective L due to the change in the intermediate distribution to state Theorem 1. It shows that the loss is upper bounded by a quadratic function of the difference of the mean of the distribution and ratio of the variances. This allows us to qualitatively measure the effect of the shift in distribution on the loss function. Theorem 1. *Let the* l thlayer of a network be a BatchNorm layer as described in 2 with stored data statistics µc and σ 2 c*. Editing components of preceeding layers causes a change in the distribution* of the intermediate representation to some Y
(p)(X)*, with modified moments* µ
(p) and (σ
(p))
2. The output of BatchNorm after editing is then, V(p) = GZ(p) + β *where* Z
(p) =
Y
(p)
c(X)−µc σc*. Then,*

$$({\boldsymbol{5}})$$

Algorithm 2: BNFix Input :Batch Norm Layer l with m channels, dataset D = {Xi}
N i=1 for c ∈ [m] do µ l c ← 1N
PN
b=1 Y
l c(Xb);
σ 2(l)
c ←PN
b=1
(Y l c
(Xb
)−µlc
)
2 N−1;

$$|\mathbb{E}[\mathcal{L}(\mathbf{V}^{(p)}(X))]-\mathcal{L}(\beta)|\leq\frac{K}{2}\left(\sum_{i=1}^{d}\gamma_{i}^{2}\left(\left(\frac{\sigma_{i}^{(p)}}{\sigma_{i}}\right)^{2}+\left(\frac{\mu_{i}^{(p)}-\mu_{i}}{\sigma_{i}}\right)^{2}\right)\right)$$
$$(6)$$
 (6)
proof sketch. We prove this result using the properties of normalization and apply Lemma 1. The full proof of this theorem can be found in D.1. Based on Theorem 1, we observe that updating stored statistics to represent the new moments of the intermediate representations after editing, i.e., setting µi = µ
(p)
iand σi = σ
(p)
i, restores the upper bound on the loss function to Lemma 1. However, the bound suggests that only channels for which the coefficient of γ 2 iin equation 6 is greater than 1 should be updated to decrease the upper bound.

We study this in Appendix B.9 and emperically show that updating the statistics of all channels leads to larger accuracy recovery in the case of pruning. Algorithm 2 shows the update procedure for the stored statistics of a single batch norm layer. This gradient-free procedure does not require training samples and can be implimented using a small number of samples obtained from distributional access. In Appendix B.2, we display the effectiveness of the algorithm on a simple synthetic task.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

## 6 Model Editing Through Correlation Structure Of Complex Interconnections

A key challenge in applying the HiFi hypothesis 4 is identify HiFi components across groups of interconnected layers in complex networks. We propose Algorithm 3 to identify HiFi components over all layers in a DFC to extend the HiFi hypothesis to networks with complex interconnections. Computational cost of Algorithm 3. Let N be the number of data points used to estimate the saliency and Ml be the complexity of computing the input contribution at layer l for a single sample in a DFC with m layers. The complexity to compute the set of HiFi channels for an output channel of a layer is, t lsal = O(NMlC
l ind l). To select the HiFi components for the DFC, the top p elements for each layer and output channel in the DFC are collected, this costs O(Pm l=1 C
lout(C
lin log C
lin + t lsal)). We compare this with the BGSC algorithm (Narshana et al., 2022) which has a quadratic dependence on the number of layers in the network, as opposed to the proposed work which is linear in the number of layers.

Algorithm 3: Compute HiFi channels over Coupled channels Input: Model, keepRatio p, Samples D = {Xi}
N i=1 Output: HiFi channels Function ComputeHiFiSet(Coupled Channels CC,
p, D):
for layer l ∈ CC do for o ∈ C
l out do Compute R
l o according to Equation RowSum; H
l o ← Topp
(R
l o
);
return Sl,o∈[Clout]
H
l o; Fidelity Compensation by Weight Rescaling In order to improve the model's performance *without* fine-tuning, we propose a distributional approach to modifying the weights to regain accuracy, by modifying the weights of layer l+ 1 after pruning layer l (similarly, we can modify the weights of feed out layers after pruning the feed-in layers of a DFC). Unlike prior work which modifies the weights of entire filters with a single parameter (Xie et al., 2021; Halabi et al., 2022), our result modifies the weights of individual convolutional kernels, thereby granting a more fine-grained approach to weight compensation. First, we define the reconstruction error as follows.

$$\mathsf{R E}_{c}^{l+1}(v)=\mathbb{E}[\|Y_{c}^{l+1}(X)-\sum_{i\in[C_{i n}]}v_{i}\Phi_{i}^{l}(X)W_{c i}^{l+1}\|^{2}]$$
$$(7)$$

2] (7)
where v ∈ R
Cin . With this definition, we state the solution to the post-pruning fidelity compensation problem, and the reconstruction error improvement in Theorem 2. Theorem 2. Let s l ∈ {0, 1}
Cin = [1K; 0Cin−K], where 1K is a vector of K ones, and 0Cin−K *is a* vector of Cin − K *zeros; we ignore the subscripts for brevity in the sequel. Define* δc ∈ R
Cin *such* that δci = 0 *when* si = 0*. We solve* Wˆ l+1 ci = ˆδ l+1 ci Wl+1 ci *, where* δ l+1 ci = [ˆδ l+1 ci ; 0Cin−K] *that satisfies*

$$\delta_{c}^{i+1}=\operatorname*{arg\,min}_{\delta_{c}\in\mathbb{R}^{K}}\operatorname{\mathsf{RE}}_{c}^{I}([\delta,0])=P_{c}^{-1}p_{c}\;\;a n d\;\frac{\operatorname{\mathsf{RE}}_{c}^{I}(s^{i})-\operatorname{\mathsf{RE}}_{c}^{I}(\overline{{{\delta}}}_{C_{i n}}^{i+1})}{\operatorname{\mathsf{RE}}_{c}^{I}(s^{i})}\leq1-\frac{\|1-\overline{{{\delta}}}_{c}^{i+1}\|^{2}}{\kappa(Q_{c}^{i+1})(C_{i n}-K)}\;.$$
$\left(8\right)$. 
where δ
⋆ c is a vector containing the optimal values of δci, Q*c,ij* = E-(Wl+1 cj )
⊤Φj (X)
⊤Φi(X)Wl+1 ci ,
Pc,ij = Qc,ij and pc,i = E-(Y
l+1 c)
⊤Φ
li
(X)Wl+1 ci when si, sj = 1, and κ(Qc,ij ) *denotes the* condition number of Q*c,ij* .

Based on the RowSum heuristic, fidelity compensation scheme 6, and BNFix 5, following the recipe of 1, we develop, **CoBRA**(Correlation based editing with Batchnorm Re-Adjustment), a model editing framework for pruning and classwise forgetting. We provide the key components of our proposed pruning and unlearning algorithm. Detailed algorithms are presented in Appendix B.11.

CoBRA-P. Compute: Compute HiFi channels using Algorithm 3 using distributional samples. Determine: Retain HiFi components **Recover:** Compute weight compensation according to equation 8 and perform BNFix using distributional samples. CoBRA-U. Compute: Compute HiFi channels using Algorithm 3 using distributional samples from the *forget class*. **Determine:** Discard HiFi components **Recover:** Compute weight compensation according to equation 8 and perform BNFix using distributional samples of the *remain* class.

## 7 Experiments

In this section, we present experimental validation of our method on pruning and class unlearning tasks for CNNs with complex interconnections like ResNets to answer the following questions. (Q1) **HiFi Hypothesis.** Is it true that there is a small set of High-fidelity channels in a well-trained network?

(Q2) **Effectiveness of CoBRA-P.** Does CoBRA-P result in better accuracy-sparsity tradeoff compared to other data-free algorithms?

(Q3) **BNFix replace retraining.** How does BNFix fare against fine-tuning using synthetic samples when pruning models?

(Q4) **CoBRA-U for unlearning.** Is classwise unlearning, as posed by Jia et al. (2023), feasible without fine-tuning? If yes, how does CoBRA-U fare against their method?

(Q5) **Total Recall of BN.** What role do batch norm statistics play in class forgetting, and how can BNFix help in recovering accuracy?

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 Datasets and architectures. We perform experiments on models including ResNet50/101 and VGG19 trained on CIFAR10/100 and ImageNet datasets. Distributional Access. As a proxy for distributional access to data in CIFAR10/100 experiments, we use samples that are synthetically generated using image generation models. Details of synthetically generated samples are available in Appendix B.1. For ImageNet experiments, we use the test split, which contains 100,000 images without labels to identify HiFi channels. Note that test split, as suggested by the name, is not used to evaluate the performance of ImageNet models. For pruning experiments on ImageNet, we perform full retraining instead of BNFix. CoBRA Hyperparameters. We discuss the hyperparameters used for CoBRA-P/U in Appendix B.10 Validating HiFi hypothesis. To answer (Q1), we compute the reconstruction error described in equation 3 for 3 different untrained and trained models on CIFAR10 using 1000 samples from the CIFAR10 validation set. We present these sorted values averaged across different trained and untrained models for every layer in Appendix B.12. We make several observations based on these results. First, for most layers, there is a diversity of scores in trained models compared to untrained models, where the scores of all the channels in untrained models are concentrated around a single value. Second, in trained models, there is a small subset of channels, typically less than 10%, which have fidelity scores less than 1. Thus, this validates the HiFi hypothesis, answering (Q1)

## 7.1 Pruning Experiments: Exploring (Q2) And (Q3) 7.2 Forgetting Experiments: Exploring (Q4) And (Q5)

Metrics. We report the forget and retain accuracy averaged across 10 classes of the CIFAR10 dataset. Additional details. Experiments with VGG-19 architecture are present in Appendix B.7 where we Baselines. To compare the performance of CoBRA-P against other data-free methods, we compare against **DFPC** (Narshana et al., 2022), a state-of-the-art data-free structured pruning algorithm for networks with complex interconnections. To gauge the efficacy of BNFix against retraining with distributional access, we compare against L2-norm-based structured pruning, which computes grouped saliencies for a coupled channel based on the L2 norm of the weights of its filters. We *train* the model obtained with L2 norm-based structured pruning *using the synthetic set* for comparison.

To the best of our knowledge, these are the only baselines addressing structured pruning of coupled channels in the data free regime. Training details. Details of pre-trained networks and post-training are given in Appendix B.4. Results of Pruning Experiments. Table 1 presents the results of pruning experiments on ResNet-50. We observe that for a similar drop in accuracy in the training-free regime, we gain **at least** 50% larger reduction in FLOPS and at least 10% larger reduction in parameters. In the training regime, we observe that for similar drop in accuracy, CoBRA-P obtains 60% fewer parameters. To answer (Q2), we find that CoBRA-P, for most cases, results in better accuracy-vs-sparsity tradeoff when compared to other data-free algorithms. To answer (Q3), BNFix is able to outperform fine-tuning in some cases using synthetic samples. While in a few cases, it does not, it still leads to a reasonably good performance when compared to no-finetuning.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

Dataset Algorithm Acc.(%) RF RP

Unpruned 94.99 1x 1x DFPC (Narshana et al., 2022) 90.25 1.46x 2.07x

L2 15.91 4.07x 4.71x L2 w/ ST 90.12 4.07x 4.71x

CoBRA-P(n) 92.64 1.74x 1.64x CoBRA-P **91.02 4.07x 5.36x**

| CIFAR10 CIFAR100 ImageNet   |
|-----------------------------|

Unpruned 78.85 1x 1x DFPC 70.31 1.27x 1.22x

L2 16.77 1.93x 1.40x L2 w/ ST **73.83** 1.93x **1.40x**

CoBRA-P(n) 72.96 1.40x 1.10x CoBRA-P 70.93 **1.93x** 1.38x Unpruned 76.1 1x 1x ThiNet (Luo et al., 2017) 71.6 3.46x 2.95x GReg-2 (Wang et al., 2021) 73.9 3.02x 2.31x OTO (Chen et al., 2021) 74.7 2.86x 2.81x DFPC **73.8** 3.46x 2.65x CoBRA-P 73.25 **3.60x 4.46x**

Table 2: Class forgetting on CIFAR10 with ResNet-50. CoBRA- U(p) indicates the hyperparameter for Algorithm 3. For p = 0.003, we only prune the last 12 convolution layers and for the last 30 convolution layers for p = 0.2. FA=Forget Accuracy, RA = Remain Accuracy, PR=Parameters removed Table 1: Experiments of CoBRA-P on CIFAR10, CIFAR100 and ImageNet compared with baselines for ResNet-50. ST=Synthetic Training, training using synthetic samples. CoBRA(n) is the CoBRA algorithm without using BNFix or Weight compensation. RF=relative FLOP reduction, RP=relative parameter reduction

## Make Similar Observations.

Results of Class-Unlearning Experiments. We report the results of our algorithm in Table 2 and Table 5. To answer (Q4), we observe that it is possible to perform unlearning even without finetuning to retain performance on the forgotten class. However, we also make the observation that it is possible to recover the accuracy of a forgotten class by updating the batch norm statistics by using *only* samples from the remaining class. We call this phenomenon the **BN Recall**. Thus, answering (Q5), it is necessary to modify the stored statistics in BN layers to truly forget class information.

| Algorithm                | FA(%)   | RA.    | PR    |
|--------------------------|---------|--------|-------|
| -                        | 94.99   | 94.99  | -     |
| Jia et al. (2023)        | 5.54    | 99.11  | -     |
| CoBRA-U(0.003)(no BNFix) | 4.22    | 91.131 | 1.0M  |
| CoBRA-U(0.003)           | 90.61   | 90.629 | 1.0M  |
| CoBRA-U(0.2)             | 20.90   | 78.786 | 3.63M |

## 7.3 Discussion Of Empirical Results.

In this section, we empirically answer questions (Q1) to (Q5). With (Q1), we show that the for each layer of a network, there exists a small set of High-Fidelity channels that contribute to the performance of the network. To answer (Q2), we conclude that CoBRA-P, for most cases, leads to a better sparsity vs. accuracy tradeoff against baseline data-free algorithms by at least 50% larger reduction in FLOPs. We also find, to answer (Q3), that BNFix sometimes results in better performance as compared to fine-tuning when using synthetic samples. However, BNFix is always better than no-finetuning. With reference to (Q4), we find that it is possible to perform unlearning even without finetuning to retain performance on the forgotten class. In trying to answer (Q5), we observe that when only remain class samples are for BNFix, it causes a significant increase in forget class performance.

## 8 Conclusion

In this paper, we study model editing in the setting where both training data and loss functions are not available, a setting not studied before. Our main contributions are algorithms devised through correlation analysis of Hifi-components- introduced for the first time here- for both Pruning Complex networks and Class Forgetting. We highlight the importance of BatchNorm statistics, which when updated, yields predictions which can be as good as those obtained from a retrained network. We provide both empirical evidence as well as a formal explaination. The results obtained here, specially those related to identifying Hi-fi components, can open doors to new research avenues useful for understanding Deep Networks. One direction for future work is to use different measures of similarity between distributions, including moment matching, Wasserstein distances, and other divergences. Limitations: The techniques proposed in this work are effective when the number of classes is less than the width of the network. This may be especially true for unlearning tasks, which implicitly requires that each class is learned by disjoint set of filters.

## References

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Jacques Benichou and Mitchell H. Gail. A delta method for implicitly defined random variables. The American Statistician, 43(1):41–44, 1989. ISSN 00031305. URL http://www.jstor.org/
stable/2685169.

Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, and John Guttag. What is the state of neural network pruning? *Proceedings of machine learning and systems*, 2:129–146, 2020.

Lucas Bourtoule, Varun Chandrasekaran, Christopher A Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. Machine unlearning. In *2021 IEEE Symposium* on Security and Privacy (SP), pp. 141–159. IEEE, 2021.

Tianyi Chen, Bo Ji, Tianyu Ding, Biyi Fang, Guanyi Wang, Zhihui Zhu, Luming Liang, Yixin Shi, Sheng Yi, and Xiao Tu. Only train once: A one-shot neural network training and pruning framework, 2021. URL https://arxiv.org/abs/2107.07467.

Xiaohan Ding, Tianxiang Hao, Jianchao Tan, Ji Liu, Jungong Han, Yuchen Guo, and Guiguang Ding.

Resrep: Lossless cnn pruning via decoupling remembering and forgetting. In *Proceedings of the* IEEE/CVF International Conference on Computer Vision, pp. 4510–4520, 2021.

Ronen Eldan and Mark Russinovich. Who's harry potter? approximate unlearning in llms. arXiv preprint arXiv:2310.02238, 2023.

Gongfan Fang, Xinyin Ma, Mingli Song, Michael Bi Mi, and Xinchao Wang. Depgraph: Towards any structural pruning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and* Pattern Recognition, pp. 16091–16101, 2023.

Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In *International Conference on Learning Representations*, 2018.

Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, and Michael Carbin. Pruning neural networks at initialization: Why are we missing the mark?, 2021.

Elias Frantar, Sidak Pal Singh, and Dan Alistarh. Optimal brain compression: A framework for accurate post-training quantization and pruning, 2022. URL https://openreview.net/ forum?id=ksVGCOlOEba.

Rohit Gandikota, Joanna Materzynska, Jaden Fiotto-Kaufman, and David Bau. Erasing concepts from diffusion models. In *Proceedings of the IEEE/CVF International Conference on Computer* Vision, pp. 2426–2436, 2023.

Aditya Golatkar, Alessandro Achille, and Stefano Soatto. Eternal sunshine of the spotless net:
Selective forgetting in deep networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9304–9312, 2020.

Siavash Golkar, Michael Kagan, and Kyunghyun Cho. Continual learning via neural pruning. arXiv preprint arXiv:1903.04476, 2019.

Sven Gowal, Sylvestre-Alvise Rebuffi, Olivia Wiles, Florian Stimberg, Dan Calian, and Timothy Mann. Improving robustness using generated data. In Proceedings of the 35th International Conference on Neural Information Processing Systems, NIPS '21, Red Hook, NY, USA, 2024.

Curran Associates Inc. ISBN 9781713845393.

Laura Graves, Vineel Nagisetty, and Vijay Ganesh. Amnesiac machine learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 11516–11524, 2021.

Varun Gupta, Christopher Jung, Seth Neel, Aaron Roth, Saeed Sharifi-Malvajerdi, and Chris Waites.

Adaptive machine unlearning. *Advances in Neural Information Processing Systems*, 34:16319–
16330, 2021.

Marwa El Halabi, Suraj Srinivas, and Simon Lacoste-Julien. Data-efficient structured pruning via submodular optimization. In *Advances in Neural Information Processing Systems*, 2022.

Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. *arXiv preprint arXiv:1510.00149*, 2015.

Babak Hassibi and David Stork. Second order derivatives for network pruning: Optimal brain surgeon.

Advances in neural information processing systems, 5, 1992.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper_files/paper/ 2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf.

Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. *Journal of* Machine Learning Research, 22(241):1–124, 2021.

Andrew G Howard. Mobilenets: Efficient convolutional neural networks for mobile vision applications. *arXiv preprint arXiv:1704.04861*, 2017.

Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4700–4708, 2017.

Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *International conference on machine learning*, pp. 448–456.

pmlr, 2015.

Zachary Izzo, Mary Anne Smart, Kamalika Chaudhuri, and James Zou. Approximate data deletion from machine learning models. In *International Conference on Artificial Intelligence and Statistics*, pp. 2008–2016. PMLR, 2021.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Yann LeCun, John Denker, and Sara Solla. Optimal brain damage. Advances in neural information processing systems, 2, 1989. URL https://proceedings.neurips.cc/paper% 5Ffiles/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper. pdf.

Saachi Jain, Hannah Lawrence, Ankur Moitra, and Aleksander Madry. Distilling model failures as directions in latent space. *arXiv preprint arXiv:2206.14754*, 2022.

Jinghan Jia, Jiancheng Liu, Parikshit Ram, Yuguang Yao, Gaowen Liu, Yang Liu, Pranay Sharma, and Sijia Liu. Model sparsity can simplify machine unlearning. In *Thirty-seventh Conference on* Neural Information Processing Systems, 2023. URL https://openreview.net/forum? id=0jZH883i34.

Donggyu Joo, Eojindl Yi, Sunghyun Baek, and Junmo Kim. Linearly replaceable filters for deep network channel pruning. In *The 34th AAAI Conference on Artificial Intelligence,(AAAI)*, 2021.

Sangamesh Kodge, Gobinda Saha, and Kaushik Roy. Deep unlearning: Fast and efficient gradient-free class forgetting. *Transactions on Machine Learning Research*.

Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, and Eleni Triantafillou. Towards unbounded machine unlearning. *Advances in Neural Information Processing Systems*, 36, 2024.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In F. Pereira, C.J. Burges, L. Bottou, and K.Q. Weinberger (eds.), *Advances in Neural Information Processing Systems*, volume 25. Curran Associates, Inc., 2012. URL https://proceedings.neurips.cc/paper_files/paper/ 2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf.

Bailin Li, Bowen Wu, Jiang Su, and Guangrun Wang. Eagleeye: Fast sub-net evaluation for efficient neural network pruning. In *European conference on computer vision*, pp. 639–654. Springer, 2020.

Guihong Li, Hsiang Hsu, Radu Marculescu, et al. Machine unlearning for image-to-image generative models. *arXiv preprint arXiv:2402.00351*, 2024.

Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. In *International Conference on Learning Representations*, 2017. URL https:
//openreview.net/forum?id=rJqFGTslg.

648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 Mingbao Lin, Rongrong Ji, Yan Wang, Yichen Zhang, Baochang Zhang, Yonghong Tian, and Ling Shao. Hrank: Filter pruning using high-rank feature map. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1529–1538, 2020.

Xiaofeng Lin, Seungbae Kim, and Jungseock Joo. Fairgrape: Fairness-aware gradient pruning method for face attribute classification. In *European Conference on Computer Vision*, pp. 414–432. Springer, 2022.

Liyang Liu, Shilong Zhang, Zhanghui Kuang, Aojun Zhou, Jing-Hao Xue, Xinjiang Wang, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Group fisher pruning for practical network compression. In *International Conference on Machine Learning*, pp. 7021–7032. PMLR, 2021a.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows, 2021b. URL https://arxiv.org/abs/2103.14030.

Jian-Hao Luo, Jianxin Wu, and Weiyao Lin. Thinet: A filter level pruning method for deep neural network compression. In *Proceedings of the IEEE international conference on computer vision*, pp. 5058–5066, 2017.

William A. Massey and Ward Whitt. A probabilistic generalization of taylor's theorem. Statistics & Probability Letters, 16(1):51–54, 1993. ISSN 0167-7152. doi: https://doi.org/10.

1016/0167-7152(93)90122-Y. URL https://www.sciencedirect.com/science/
article/pii/016771529390122Y.

P Molchanov, S Tyree, T Karras, T Aila, and J Kautz. Pruning convolutional neural networks for resource efficient inference. In 5th International Conference on Learning Representations, ICLR 2017-Conference Track Proceedings, 2019a.

Pavlo Molchanov, Arun Mallya, Stephen Tyree, Iuri Frosio, and Jan Kautz. Importance estimation for neural network pruning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11264–11272, 2019b.

Chaitanya Murti, Tanay Narshana, and Chiranjib Bhattacharyya. Tvsprune-pruning nondiscriminative filters via total variation separability of intermediate representations without fine tuning. In *The Eleventh International Conference on Learning Representations*, 2022.

Preetum Nakkiran, Behnam Neyshabur, and Hanie Sedghi. The deep bootstrap framework: Good online learners are good offline generalizers, 2021. URL https://arxiv.org/abs/2010. 08127.

Tanay Narshana, Chaitanya Murti, and Chiranjib Bhattacharyya. Dfpc: Data flow driven pruning of coupled channels without data. In *The Eleventh International Conference on Learning* Representations, 2022.

Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, and Quoc Viet Hung Nguyen. A survey of machine unlearning. *arXiv preprint arXiv:2209.02299*, 2022.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper_files/paper/
2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf.

Prafull Prakash, Chaitanya Murti, Saketha Nath, and Chiranjib Bhattacharyya. Optimizing dnn architectures for high speed autonomous navigation in gps denied environments on edge devices. In *Pacific Rim International Conference on Artificial Intelligence*, pp. 468–481. Springer, 2019.

Tilman Räuker, Anson Ho, Stephen Casper, and Dylan Hadfield-Menell. Toward transparent ai: A
survey on interpreting the inner structures of deep neural networks. In 2023 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pp. 464–483. IEEE, 2023.

Sabyasachi Sahoo, Mostafa Elaraby, Jonas Ngnawe, Yann Pequignot, Frédéric Precioso, and Christian Gagné. Layerwise early stopping for test time adaptation. *arXiv preprint arXiv:2404.03784*, 2024.

Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry. How does batch normalization help optimization? *Advances in neural information processing systems*, 31, 2018.

702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 Shibani Santurkar, Dimitris Tsipras, Mahalaxmi Elango, David Bau, Antonio Torralba, and Aleksander Madry. Editing a classifier by rewriting its prediction rules. *Advances in Neural Information* Processing Systems, 34:23359–23373, 2021.

Ayush Sekhari, Jayadev Acharya, Gautam Kamath, and Ananda Theertha Suresh. Remember what you want to forget: Algorithms for machine unlearning. Advances in Neural Information Processing Systems, 34:18075–18086, 2021.

Juwon Seo, Sung-Hoon Lee, Tae-Young Lee, Seungjun Moon, and Gyeong-Moon Park. Generative unlearning for any identity. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9151–9161, 2024.

Harshay Shah, Andrew Ilyas, and Aleksander Madry. Decomposing and editing predictions by modeling model computation. *arXiv preprint arXiv:2404.11534*, 2024.

Maying Shen, Hongxu Yin, Pavlo Molchanov, Lei Mao, Jianna Liu, and Jose M Alvarez. Structural pruning via latency-saliency knapsack. *Advances in Neural Information Processing Systems*, 35:
12894–12908, 2022.

Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In *International Conference on Learning Representations*, 2015. URL http:
//arxiv.org/abs/1409.1556.

Nimit Sohoni, Jared Dunnmon, Geoffrey Angus, Albert Gu, and Christopher Ré. No subclass left behind: Fine-grained robustness in coarse-grained classification problems. Advances in Neural Information Processing Systems, 33:19339–19352, 2020.

Hidenori Tanaka, Daniel Kunin, Daniel L Yamins, and Surya Ganguli. Pruning neural networks without any data by iteratively conserving synaptic flow. Advances in Neural Information Processing Systems, 33:6377–6389, 2020.

Anvith Thudi, Gabriel Deza, Varun Chandrasekaran, and Nicolas Papernot. Unrolling sgd: Understanding factors influencing machine unlearning. In *2022 IEEE 7th European Symposium on* Security and Privacy (EuroS&P), pp. 303–319. IEEE, 2022.

Huan Wang, Can Qin, Yulun Zhang, and Yun Fu. Neural pruning via growing regularization. arXiv preprint arXiv:2012.09243, 2020b.

Chaoqi Wang, Guodong Zhang, and Roger Grosse. Picking winning tickets before training by preserving gradient flow. *arXiv preprint arXiv:2002.07376*, 2020a.

Huan Wang, Can Qin, Yulun Zhang, and Yun Fu. Neural pruning via growing regularization. In International Conference on Learning Representations, 2021. URL https://openreview. net/forum?id=o966_Is_nPA.

Junxiao Wang, Song Guo, Xin Xie, and Heng Qi. Federated unlearning via class-discriminative pruning. In *Proceedings of the ACM Web Conference 2022*, pp. 622–632, 2022.

Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. A comprehensive survey of continual learning:
Theory, method and application. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2024.

Zhenyi Wang, Enneng Yang, Li Shen, and Heng Huang. A comprehensive survey of forgetting in deep learning beyond continual learning. *arXiv preprint arXiv:2307.09218*, 2023.

Alexander Warnecke, Lukas Pirch, Christian Wressnegger, and Konrad Rieck. Machine unlearning of features and labels. *arXiv preprint arXiv:2108.11577*, 2021.

Zhouyang Xie, Yan Fu, Shengzhao Tian, Junlin Zhou, and Duanbing Chen. Pruning with compensation: Efficient channel pruning for deep convolutional neural networks, 2021. URL https://arxiv.org/abs/2108.13728.

Tianyun Yang, Juan Cao, and Chang Xu. Pruning for robust concept erasing in diffusion models.

arXiv preprint arXiv:2405.16534, 2024.

Yifan Yang and Xiaoyu Zhou. A note on taylor's expansion and mean value theorem with respect to a random variable, 2021. URL https://arxiv.org/abs/2102.10429.

756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 Hongxu Yin, Pavlo Molchanov, Jose M Alvarez, Zhizhong Li, Arun Mallya, Derek Hoiem, Niraj K
Jha, and Jan Kautz. Dreaming to distill: Data-free knowledge transfer via deepinversion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8715–8724, 2020.

Ruichi Yu, Ang Li, Chun-Fu Chen, Jui-Hsin Lai, Vlad I Morariu, Xintong Han, Mingfei Gao, Ching-Yung Lin, and Larry S Davis. Nisp: Pruning networks using neuron importance score propagation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 9194–9203, 2018.

Shixing Yu, Zhewei Yao, Amir Gholami, Zhen Dong, Sehoon Kim, Michael W Mahoney, and Kurt Keutzer. Hessian-aware pruning and optimal neural implant. In Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision, pp. 3880–3891, 2022.

# Appendix

This appendix is organised as follows:
1. Appendix A contains details of related work 2. Appendix B contains additional experimental details 3. Appendix C contains details about BatchNorm 4. Appendix D contains derivations and proofs not presented in the main body.

## A Related Work A.1.1 Editing Classifiers

810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 In this subsection, we discuss **model editing**, which refers to techniques by which model parameters are perturbed in order to change or influence the statistical performance of the model. A variety of tasks fall under this umbrella, including pruning, model unlearning Shah et al. (2024), debiasing Santurkar et al. (2021), and continual learning Sahoo et al. (2024). Interpreting and editing classifier models is an active area of research, motivated by problems such as subclass stratification (wherein subgroups within classes of a dataset can exhibit significantly different statistical performance)Sohoni et al. (2020) and debiasing Santurkar et al. (2021); Jain et al.

(2022); Shah et al. (2024). The methods proposed in the latter works are of particular interest. In Jain et al. (2022), CLIP embeddings are used to find "failure directions" between samples upon which the model succeeds and those on which the model fails using an SVM; these "directions" are then used to design a variety of interventions in the weight space. In Santurkar et al. (2021), classifier prediction rules are edited by using learned rank-1 updates on a subset of layers of a DNN. Most pertinently, in Shah et al. (2024), an exhaustive approach to component attribution is used, and a variety of tasks including classwise unlearning, debiasing, editing individual predictions, and improving subpopulation robustness. In the sequel, we discuss other methods that show that model unlearning can also be achieved via model editing.

## A.1.2 Editing Other Models

Model editing, while of interest to classifier models, has gained more interest in generative modeling. For instance, component editing and pruning have been successfully applied to model editing tasks in GANsLi et al. (2024); Seo et al. (2024) and diffusion models Yang et al. (2024), particularly for unlearning tasks.

## A.2 Machine Unlearning

In this subsection, we provide a detailed literature survey on machine unlearning, both with and without model editing. Machine unlearning assumes that a model f(·) is given, trained on a dataset D. The dataset is then partitioned into Dr (i.e. the retain or *remember* set) and Df (the *forget* set). The goal of machine unlearning is to minimize the accuracy on Df while maintaining the accuracy on Dr.

## A.2.1 Machine Unlearning Without Model Editing

Machine unlearning has gained importance in recent years owing to data privacy and security concerns Bourtoule et al. (2021); Nguyen et al. (2022). A wide variety of works exist to address this problem. Several works aim to forget data points, even in the adaptive setting, while maintaining the accuracy of the model, such as Sekhari et al. (2021); Gupta et al. (2021); Izzo et al. (2021); Golatkar et al. (2020). The work in Sekhari et al. (2021) also provides bounds on the number of samples that a model can be allowed to forget before accuracy degradation. Machine unlearning is also a significant

## A.1 Model Editing

864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 area of research in the space of large language models, as noted in Kurmanji et al. (2024); Eldan & Russinovich (2023), and generative models Gandikota et al. (2023). Another aspect of machine unlearning is selective forgetting, wherein classes, groups, or sets of samples are forgotten from the network, as described in Wang et al. (2023) and the references therein. This connects machine unlearning to the continual learning setting as well, as described in Wang et al. (2024) and the references cited there. There are a variety of approaches to selective or classwise forgetting, many of which require retraining or fine-tuning on subsets of the data. Fine-tuning, which includes methods such as Golatkar et al. (2020); Warnecke et al. (2021), requires retraining the model on Dr, assuming that after sufficient iterations, the accuracy on Df would be degraded. Other works, such as Graves et al. (2021); Thudi et al. (2022) use gradient *ascent* on the loss function with Df ,
thereby destroying the accuracy of the model on Df .

## A.2.2 Machine Unlearning With Model Editing

Recent works have demonstrated the promise of model unlearning by *editing* models. In Jia et al.

(2023); Sahoo et al. (2024), tools for unstructured pruning are leveraged to analyze machine unlearning on sparse models, and the impact of model sparsity on such tasks. More recently, works such as Shah et al. (2024); Kodge et al.; Wang et al. (2022) directly uses structured pruning for model unlearning, by identifying components responsible for classwise predictions and removing them.

## A.3 Structured Pruning

Structured pruning is a popular technique for improving real-world performance of models - in terms of metrics such as inference time, power consumption, and throughput - without requiring additional specialized hardware or software Hoefler et al. (2021); Blalock et al. (2020). Unlike unstructured pruning (see Frankle & Carbin (2018); Frankle et al. (2021) and the references therein for a more detailed discussion), wherein individual weights are removed, structured pruning directly reduces the number of matrix-matrix multiplications, thereby improving performance Hoefler et al. (2021). Early work on structured pruning involved pruning neurons in feedforward networks, such as LeCun et al. (1989); Hassibi & Stork (1992). More recent work typically utilizes derivatives of the loss function, such as Molchanov et al. (2019a;b); Shen et al. (2022); Li et al. (2020), which use gradients, or Hessian Liu et al. (2021a); Yu et al. (2022); Wang et al. (2020a). More recently, Lin et al. (2022) proposes estimating class-conditional gradient based saliency scores for identifying filters responsible for class-wise or group-wise predictions, with a view toward fair pruning.

## A.3.1 Structured Pruning In The Data-Free Regime

The space of pruning without access to the training data or loss function remains an under-researched area. There are a variety of methods that do not use training data to generate saliency scores for filters, such as Yu et al. (2018), which uses an L1 reconstruction error bound, Lin et al. (2020) which uses the rank of feature maps, ,Li et al. (2017) which uses weight norms, and Joo et al. (2021) which uses linear combinations of filters to replace redundant filters. These methods do not directly apply them to pruning in the data-free regime. In this work, we assume access to the training data distributions, with which we derive derivative-free meausres of importance of filters based on correlations between the input contributions they generate.

## B Additional Experiments B.1 Synthetic Datasets B.1.1 Cifar5M

For experiments with the CIFAR10 dataset, we use CIFAR5M, a dataset containing 6 million synthetic CIFAR-10-like images sampled from a Diffusion model and labelled by a Big-Transfer model(Nakkiran et al., 2021), which we randomly sample 10,000 samples from each of the 10 classes to create our dataset. This dataset has an FID(Heusel et al., 2017) of 15.95 with respect to the CIFAR10 training set. This dataset is obtained from https://github.com/preetum/cifar5m. ImageNet post training: For ImageNet, we use off-the-shelf pretrained models from Torchvision(Paszke et al., 2019). We train the model for 3 epochs after each iteration of CoBRA-P with learning rates of 0.1, 0.01, 0.001. After the pruning ends, we finally prune the network for 200 with a

| Dataset   | Model Architecture   | Original   | σ       | +Noising   | +BNFix   |       |       |       |
|-----------|----------------------|------------|---------|------------|----------|-------|-------|-------|
| Loss      | Acc.(%)              | Loss       | Acc.(%) | Loss       | Acc.(%)  |       |       |       |
| 0.010     | 2.2                  | 32.31      | 0.5     | 87.16      |          |       |       |       |
| ResNet-50 | 0.21                 | 94.99      | 0.012   | 4.96       | 10.67    | 1.12  | 72.91 |       |
| 0.014     | 20.49                | 9.89       | 1.87    | 37.07      |          |       |       |       |
| CIFAR-10  | 0.010                | 6.04       | 18.75   | 0.5        | 86.33    |       |       |       |
| VGG19     | 0.31                 | 93.50      | 0.012   | 15.11      | 11.62    | 1.23  | 59.52 |       |
| 0.014     | 69.69                | 10.05      | 2.01    | 26.20      |          |       |       |       |
| 0.010     | 3.00                 | 30.31      | 1.61    | 64.06      |          |       |       |       |
| ResNet-50 | 0.9                  | 78.85      | 0.012   | 4.52       | 2.84     | 2.42  | 51.14 |       |
| 0.014     | 5.31                 | 0.97       | 3.36    | 31.35      |          |       |       |       |
| CIFAR-100 | 0.010                | 1.62       | 62.74   | 1.55       | 66.02    |       |       |       |
| VGG19     | 1.46                 | 72.02      | 0.012   | 2.27       | 48.94    | 1.62  | 62.71 |       |
| 0.014     | 3.75                 | 13.58      | 1.80    | 58.21      |          |       |       |       |
| ImageNet  | ResNet-50            | 0.96       | 76.15   | 0.010      | 4.38     | 20.56 | 1.73  | 63.63 |

Table 3: Effect of BNFix on noising a network. σ represents the variance of the noise added to the network

## B.1.2 Cifar100-Ddpm

For experiements with the CIFAR100 dataset, we use CIFAR100-DDPM(Gowal et al., 2024), which we randomly downsample to contain 1,000 samples from each of the 100 classes. This dataset has an FID of 4.74 with respect to the CIFAR100 training set. We randomly sample 1,000 samples from each of the 100 classes to create our dataset. This dataset is obtained from https://github.com/google-deepmind/deepmind-research/tree/ master/adversarial_robustness/iclrw2021doing.

## B.2 Batchnorm Noising

918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 To illustrate the effect of BNFix, we will first consider an artificial editing task we call model noising. Although not a practical procedure, it serves to illustrate the effect of BNFix. The model is "edited" by adding gaussian noise to all of the learned parameters of the network. We add a zero mean random value to every learned parameter(including biases) of the model and apply BNFix for 5 iterations over the synthetic set. Table 3 showcases the performance of the model before and after noising in terms of the accuracy of the model and the value of the crossentropy loss averaged over the test set.

Noising causes a dramatic fall in accuracy and increase in loss but BNFix is able to recover from around 10% to 60% of the validation accuracy across models and datasets.

## B.3 Effect Of Number Of Samples For Bnfix

To understand the number of samples required for BNFix, we use random pruning to prune a ResNet50 model trained on the CIFAR10 dataset to achieve 2x FLOP reduction. We then apply BNFix using the synthetic set. In Figure 3, we showcase the effect of the size of the synthetic set use and show a 95% confidence interval over 4 runs with different random subsets. We see that after around 1500 samples the gains due to adding additional samples diminish.

## B.4 Training Procedure

Pretraining procedure: For CIFAR10 and CIFAR100, we train models using SGD Optimizer with a momentum factor of 0.9 and weight decay of 5 × 10−4for 200 epochs using Cosine Annealing step sizes with an initial learning rate of 0.1. batch size of 512. We use the SGD Optimizer with a momentum factor of 0.9 and weight decay of 1 × 10−4and Cosine Annealed step sizes with an initial learning rate of 0.1.

## B.5 Bnfix And Pruning

972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 We use pruning algorithms like L1, L2, and Random pruning on CIFAR10 trained ResNet-50 models to obtain models with 3x FLOP reduction. We then apply BNFix with 5000 synthetic samples for 20 iterations. Figure 4 shows the effectiveness of BNFix on these models, recovering upto 65% validation accuracy for this model.

L2 **Post training procedure:** For the synthetic training experiments mentioned in Section 7, we first prune the model using L2 norm as the grouped saliency to a similar sparsity as CoBRA-P. We then train the model using 50000 samples from the synthetic dataset for 100 epochs with a batch size of 128 using SGD optimizer with momentum factor of 0.9 with inital learning rate of 0.01 and a MultiStepLR learning rate scheduler with milestones at 60 and 80 epochs.

## B.6 Additional Pruning Experiments

1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 Table 4: Experiments of CoBRA-P on CIFAR100 compared with baselines. RF=Reduction in FLOPs. RP=Reduction in Parameters. ST=Synthetic training, training using synthetic samples.

## B.7 Additional Forgetting Experiments

We report additional experiments on class unlearning on different architectures. For VGG-19 networks, we remove the HiFi channels for the forget class of the last 12 convolution layers.

| Model                    | Algorithm      | Forget Acc.(%)   | Remain Acc.   | params. removed   |
|--------------------------|----------------|------------------|---------------|-------------------|
| -                        | 93.50          | 93.50            | -             |                   |
| CoBRA-U(0.001)(no BNFix) | 0.86           | 77.85            | 0.79M         |                   |
| VGG19                    | CoBRA-U(0.001) | 45.87            | 91.31         | 0.79M             |
| CoBRA-U(p=0.2)           | 5.63           | 84.34            | 3.18M         |                   |

Table 5: Class forgetting on CIFAR10 for VGG19. CoBRA-U(p) indicates the hyperparameter for Algorithm 3.

## B.8 Class Unlearning For Vision Transformers

In this subsection, we describe how CoBRA-U can be applied to Vision Transformers to perform gradient free class unlearning without training data or access to the loss function. We focus on the SwinTransformer(Liu et al., 2021b) architecture and prune linear layers in the network. We use the distributional measure described in 4 to measure the importance of weights in linear layers of the network for the forget class. We use this measure in the form of an unstructured saliency to prune the weights of linear layers which include the WQ, WK, WV and MLP layers in the network. For sequence models like transformers, we compute the expectation described in equation 3 over all elements in the sequence. We report class forgetting results on the SwinTransformer(Liu et al., 2021b) architecture trained on CIFAR-10. We train the model on the CIFAR10 dataset for 300 epochs from scratch3to achieve a validation accuracy of 92.31%. We apply CoBRA-U on the linear layers in a vision transformer.

Unpruned 72.02 1x 1x

DFPC 70.10 1.26x 1.50x

L2 56.46 1.50x 2.40x

L2 w/ ST **72.42** 1.50x **2.40x**

CoBRA-P 70.26 **1.51x** 2.31x

ResNet-101

Unpruned 95.09 1x 1x

DFPC 89.80 1.53x 1.84x

L2 w/ ST 90.49 4.20 **5.29x**

CoBRA-P **91.20 4.21x** 4.79x

VGG19

Unpruned 93.50 1x 1x

DFPC 90.25 1.46x 2.07x

L2 w/ ST 89.23 2.39x **9.19x**

CoBRA-P **91.80 2.39x** 5.52x

Dataset Model Algorithm Acc.(%) RF RP

| CIFAR-100   | VGG19 ResNet-101   |
|-------------|--------------------|
| CIFAR10     | VGG19              |