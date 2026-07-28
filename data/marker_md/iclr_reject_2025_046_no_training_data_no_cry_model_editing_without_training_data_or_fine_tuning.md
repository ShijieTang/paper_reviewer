**006**

**009**

**019**

**024**

**029 030**

**032**

**034**

**036**

# NO TRAINING DATA, NO CRY: MODEL EDITING WITH-OUT TRAINING DATA OR FINETUNING

Anonymous authors Paper under double-blind review

# ABSTRACT

Model Editing(ME)–such as classwise unlearning and structured pruning–is a nascent field that deals with identifying editable components that, when modified, significantly change the model's behaviour, typically requiring fine-tuning to regain performance. The challenge of model editing increases when dealing with multibranch networks(e.g. ResNets) in the data-free regime, where the training data and the loss function are not available. Identifying editable components is more difficult in multi-branch networks due to the coupling of individual components across layers through skip connections. This paper addresses these issues through the following contributions. First, we hypothesize that in a well-trained model, there exists a small set of channels, which we call HiFi channels, whose input contributions strongly correlate with the output feature map of that layer. Finding such subsets can be naturally posed as an expected reconstruction error problem. To solve this, we provide an efficient heuristic called RowSum. Second, to understand how to regain accuracy after editing, we prove, for the first time, an upper bound on the loss function post-editing in terms of the change in the stored BatchNorm(BN) statistics. With this result, we derive BNFix, a simple algorithm to restore accuracy by updating the BN statistics using distributional access to the data distribution. With these insights, we propose retraining free algorithms for structured pruning and classwise unlearning, CoBRA-P and CoBRA-U, that identify HiFi components and retains(structured pruning) or discards(classwise unlearning) them. CoBRA-P achieves at least 50% larger reduction in FLOPS and at least 10% larger reduction in parameters for similar drop in accuracy in the training free regime. In the training regime, for ImageNet, it achieves 60% larger parameter reduction. CoBRA-U achieves, on average, a 94% reduction in forget-class accuracy with a minimal drop in remaining class accuracy.[<sup>1</sup>](#page-0-0)

# 1 INTRODUCTION

The improved performance of deep learning models on various tasks [\(Krizhevsky et al., 2012;](#page-11-0) [Ioffe &](#page-11-1) [Szegedy, 2015;](#page-11-1) [He et al., 2016\)](#page-11-2) has increased their adoption. However, such models may not always be suitable for direct use in various applications. For instance, a pre-trained classification model might not run on an edge device without compressing it using a technique such as pruning [\(Prakash](#page-13-0) [et al., 2019\)](#page-13-0). We use the term *Model Editing* to refer to such modifications.

This work focuses on two model editing tasks - pruning and classwise unlearning for vision tasks. Pruning [\(LeCun et al., 1989;](#page-11-3) [Hoefler et al., 2021\)](#page-11-4) is one of the methods to improve latencies and memory requirements of models during inference. Pruning involves discarding "unimportant" components of a model, such as weights, neurons, or channels. This work focuses on structured pruning [\(Luo et al., 2017;](#page-12-0) [Wang et al., 2020b;](#page-13-1) [Shen et al., 2022\)](#page-13-2) that discards entire channels in Convolution Neural Networks (CNNs) as opposed to unstructured pruning [\(LeCun et al., 1989;](#page-11-3) [Han](#page-11-5) [et al., 2015;](#page-11-5) [Tanaka et al., 2020\)](#page-13-3) that discards weights individually. Classwise unlearning [\(Jia et al.,](#page-11-6) [2023\)](#page-11-6) refers to the task where the goal is to unlearn training data points of an entire class while maintaining the predictive performance on remaining classes. Classwise unlearning can be efficiently performed using pruning [\(Jia et al., 2023\)](#page-11-6).

Editing tasks such as pruning and classwise unlearning require an understanding of the components

<sup>1</sup>The code is available at <https://anonymous.4open.science/r/cobra-197B>

**059**

**061**

**064**

**067**

**069 070 071**

**074**

**079**

**089 090 091**

**094**

**104 105 106** of a model - such as weights, neurons, or convolutional filters that contribute significantly to its prediction [\(Räuker et al., 2023\)](#page-13-4). This becomes more challenging when dealing with modern neural networks that consist of skip connections [\(He et al., 2016;](#page-11-2) [Huang et al., 2017\)](#page-11-7) that couple elements between layers [\(Liu et al., 2021a;](#page-12-1) [Fang et al., 2023\)](#page-10-0). However, this is not generally addressed in relevant works [\(Jia et al., 2023;](#page-11-6) [Ding et al., 2021;](#page-10-1) [Joo et al., 2021;](#page-11-8) [Luo et al., 2017\)](#page-12-0). Editing algorithms often take a toll on the original task performance and thus rely on retraining to alleviate this [\(Luo et al., 2017;](#page-12-0) [Wang et al., 2020b;](#page-13-1) [Jia et al., 2023\)](#page-11-6).

However, retraining requires significant computational resources and access to the loss function & training set pertaining to the original task. It is not uncommon for the relevant training set & loss function to be unavailable due to privacy or commercial concerns [\(Yin et al., 2020\)](#page-14-0), making retraining more challenging. Most existing works either assume access to data and finetune models [\(Jia et al.,](#page-11-6) [2023;](#page-11-6) [Wang et al., 2020b;](#page-13-1) [Shen et al., 2022\)](#page-13-2) or assume the absence of training data and do not finetune [\(Narshana et al., 2022;](#page-12-2) [Murti et al., 2022;](#page-12-3) [Tanaka et al., 2020\)](#page-13-3). However, the gap between the accuracy of data-free and data-driven methods is significant [\(Hoefler et al., 2021\)](#page-11-4). Thus, it is important to bridge this gap.

For model editing, this work, similar to [Murti et al.](#page-12-3) [\(2022\)](#page-12-3), assumes access to samples with similar distributional properties to that of the training set. For instance, to construct a cat-dog classifier, a training set could be a large collection of images of cats and dogs taken from a private image repository, while samples available via distributional access could be the photos of cats and dogs taken from a personal device. We use this distributional information to study CNNs with Batchnorm layers [\(Ioffe & Szegedy, 2015\)](#page-11-1). Batch Normalization, a popular deep learning technique developed to decrease training time, is used in many successful architectures like ResNets [\(He et al., 2016\)](#page-11-2), VGGs [\(Simonyan & Zisserman, 2015\)](#page-13-5), and MobileNet [\(Howard, 2017\)](#page-11-9). Existing theoretical analysis of Batch Normalization has focused on understanding its effect during training [\(Santurkar et al.,](#page-13-6) [2018\)](#page-13-6); however, to the best of our knowledge, there has been little insight into its effect on the loss function during inference upon model perturbation. Towards addressing the challenges presented above, the following are our contributions:

- 1. It is important for model editing to understand what components of a well-trained model are necessary for predictions. To address this, we propose the notion of High-Fidelity(HiFi) components, components of the network that contribute significantly to the output of the corresponding layer. Using this notion, we hypothesize that in each layer of a well-trained model, the set of HiFi components are responsible for the model's performance, which we empirically validate in Section [7.](#page-8-0) Thus, the problem of model editing boils down to identifying HiFi components.
- 2. Towards identifying HiFi components in a layer for model editing without access to training data or the loss function, We use correlation as the measure of similarity between the distribution of the input channel's contribution to the output and the distribution of the output. In Section [4,](#page-3-0) we show that this choice of similarity naturally connects HiFi components to those with low expected reconstruction error, a popular saliency measure in pruning. However, this problem is NP-Hard, and the use of a heuristic called RowSum is required to solve this problem. This enables the identification of editable components using distributional access.
- 3. Typically, editing causes a degradation in the model's performance. To understand the impact of BatchNorm parameters on this degradation, we derive a connection between the learned parameters of BatchNorm layers and the loss function. We show that the loss function can be upper bounded by a quadratic function of the learned parameters of the BatchNorm layer. We state this formally in Theorem [1.](#page-6-0) Based on our analysis, we propose Algorithm [2,](#page-6-1) called BNFix, an algorithm requiring only distributional access to modify the stored statistics in a BatchNorm layer to reduce performance degradation due to model editing. We observe an interesting phenomenon, which we call BN Recall, when applying BNFix as a replacement for retraining using remaining class examples - applying BNFix on a model whose forget accuracy has significantly fallen using only remain class samples causes the forget class accuracy to increase significantly.
- 4. In addition to identifying HiFi components and BNFix, we use fidelity compensation where we improve the fidelity of the feature maps via weight rescaling - to design the CoBRA family of editing algorithms and analyze this improvement in Theorem [2.](#page-7-0) CoBRA(Correlation-based editing with Batchnorm Re-Adjustment) is an editing scheme that identifies HiFi components in each layer of a network to either retain(CoBRA-P) or discard(CoBRA-U), and recovers model performance by BNFix and weight compensation. Our experiments show that CoBRA-P achieves at least 50% larger reduction in FLOPS and at least 10% larger reduction in parameters for similar

**114 115**

**117**

**119**

**127**

**129 130**

**134**

**136**

![](_page_2_Diagram_1.jpeg)

Figure 1: Left Image: Each channel of the input features generates an *input contribution*, which are then summed to obtain the feature map. Right Image: After editing a layer, feature distributions of subsequent layers are changed; adjusting BN stats helps address this.

drop in accuracy in the training free regime. In the training regime, for ImageNet, it achieves 60% larger parameter reduction. CoBRA-U achieves, on average, a 94% reduction in forget-class accuracy with a minimal drop in remain class accuracy.

# 2 PRELIMINARIES

Notation Let a ∈ R <sup>n</sup> denote an n-dimensional vector whose i th element is a<sup>i</sup> , and B ∈ R n×m a matrix with n rows and m columns whose i th row is b<sup>i</sup> ∈ <sup>R</sup> <sup>m</sup>. For p ∈ <sup>N</sup>, let [p] = {1, . . . , p}. For matrices, A, B ∈ R <sup>n</sup>×<sup>m</sup>, we define ⟨A, B⟩ = Tr(A⊤B) and frobenius ∥A∥ 2 <sup>F</sup> = ⟨A, A⟩. For tensors **A**, **B** ∈ R <sup>C</sup>×K×<sup>K</sup>, we define ⟨**A**, **B**⟩ = P<sup>C</sup> <sup>i</sup>=1⟨A<sup>i</sup> , Bi⟩, where A<sup>i</sup> , B<sup>i</sup> ∈ <sup>R</sup> <sup>K</sup>×<sup>K</sup> and ||**A**||<sup>2</sup> = ⟨**A**, **A**⟩. For a vector v, diag(v) is a diagonal matrix whose i th entry is v<sup>i</sup> . Top<sup>p</sup> (v) denotes a function that which returns the indices of the elements in the top p th-percentile of v.

Neural Network Preliminaries Let f<sup>θ</sup> be a neural network with parameters θ with L layers. Consider data drawn from a distribution <sup>P</sup>D, we use X as a random variable drawn from this distribution. We use Lθ(x) as the loss function evaluated with parameters θ on a point x and parameters are trained to minimize the expected loss over the distribution. The parameters are grouped into structural units, such as convolutional filters in CNNs, and are stacked in layers. We refer to such structures as *components* of the network. The structures and the operations performed on the input by these structures form the architecture of the network.

2D Convolution Let the l th layer of a network be a 2D convolution layer with c l in input channels and c l out output channels whose weights are **W**<sup>l</sup> ∈ <sup>R</sup> c l out×c l in×k×k , where k is the kernel size. Let the input to the convolution layer be Φ l (x) ∈ <sup>R</sup> c in×h <sup>l</sup>−<sup>1</sup>×w l−1 , and the output **Y** l (x) ∈ <sup>R</sup> c l out×h <sup>l</sup>×w l , where h l−1 , h<sup>l</sup> and w l−1 , w<sup>l</sup> represent the heights and widths of the input and output respectively. The c th output channel, Y c is then,

$$\mathbf{Y}_c^l(\mathbf{x}) = \sum_{i=1}^{c_{in}^l} \Phi_i^l(\mathbf{x}) * \mathbf{W}_{ci}^l = \sum_{i=1}^{c_{in}^l} \mathbf{A}_{ci}^l(\mathbf{x}) \quad (1)$$

where ∗ denotes the convolution operation. We say A<sup>l</sup> ci(x) ∈ <sup>R</sup> h <sup>l</sup>×w is the *input contribution* of input channel i to output channel c; this is illustrated in Figure [1a.](#page-2-0)

Batch Normalization during inference Let the l th layer of a neural network be a a BatchNorm layer with dimension m whose input is y l (x) ∈ <sup>R</sup> <sup>m</sup>, parameterized by two stored statistics, mean µ ∈ R <sup>m</sup> and standard deviation σ ∈ <sup>R</sup> <sup>m</sup>, and two learned parameters, shift β ∈ <sup>R</sup> <sup>m</sup> and scale γ ∈ R <sup>m</sup>. The c th output of the layer during inference, v l (x) ∈ <sup>R</sup> <sup>m</sup>, is given by

$$v^l(x) = Gz^l(x) + \beta \quad \text{where} \quad z_c^l(x) = \frac{y_c^l(x) - \mu_c}{\sigma_c} \quad (2)$$

where G = diag(γ). The stored statistics are meant to estimate the mean and standard deviation of y l (X) from the *training data*. Additional details are in Appendix [C.](#page-21-0)

**166 167**

**169**

**171**

**204**

**206**

# 3 THE PROBLEM OF EDITING WELL-TRAINED MODELS WITHOUT TRAINING DATA

Model editing refers to techniques that selectively change the model parameters to modify its statistical behaviour [\(Jia et al., 2023;](#page-11-6) [Santurkar et al., 2021;](#page-13-7) [Shah et al., 2024\)](#page-13-8), motivated by issues such as privacy and GDPR regulations [\(Bourtoule et al., 2021;](#page-10-2) [Nguyen et al., 2022\)](#page-12-4). Editing encompasses a wide variety of tasks, including debiasing [\(Jain et al., 2022\)](#page-11-10), selective unlearning [\(Golatkar et al.,](#page-10-3) [2020\)](#page-10-3), network scrubbing [\(Kurmanji et al., 2024\)](#page-11-11), and lifelong learning [\(Sahoo et al., 2024;](#page-13-9) [Golkar](#page-10-4) [et al., 2019\)](#page-10-4). Recently, *component attribution* - that is, identifying components responsible for predictions - has gained traction for model unlearning [\(Shah et al., 2024;](#page-13-8) [Wang et al., 2022;](#page-14-1) [Kodge](#page-11-12) [et al.\)](#page-11-12). However, it is challenging to use model editing without the loss function and training data [\(Shah et al., 2024\)](#page-13-8), as well as for analyzing models with complex interconnections [\(Narshana](#page-12-2) [et al., 2022;](#page-12-2) [Liu et al., 2021a\)](#page-12-1). Extensive related work is cited in Appendix [A.](#page-15-0) In this section we formalize the problem of Model Editing via pruning.

What is Model Editing? Consider the model f<sup>θ</sup><sup>0</sup> , and let D<sup>i</sup> , i ∈ [M] be conditional data distributions, such as classes. Our goal is to *edit* the model by removing entire components. That is, given the weights of the well-trained model θ0, we edit θ<sup>0</sup> to θ<sup>E</sup> = θ<sup>0</sup> − θ ⋆ , where θ <sup>⋆</sup> ∈ S<sup>B</sup> := {θ ∈ R d : count(f<sup>θ</sup>0−θ) = B} ⊂ <sup>R</sup> <sup>d</sup> by *editing the parameters of at most* Ctotal − B *components, where* Ctotal *is the total number of components in the network (i.e., convolutional filters) by solving*

$$\theta^* = \arg \min_{\theta \in S_B} \sum_i \mathbb{E}_{X \sim \mathcal{D}_i} [\alpha_i (\mathcal{L}_{\theta_0 - \theta}(X) - \mathcal{L}_\theta(X))], \quad (\text{Edit})$$

where α<sup>i</sup> ∈ <sup>R</sup> are multipliers to weight tasks, depending on whether we want the model to increase the loss or decrease it on the corresponding distribution D<sup>i</sup> . While a variety of tasks can be classified as model editing [\(Shah et al., 2024\)](#page-13-8); in this work, we address the problems of structured pruning and classwise unlearning.

Structured Pruning, in the setting of equation [Edit](#page-3-1), is when M = 1, and α<sup>1</sup> = 1. Thus, we write

$$\theta^* = \arg \min_{\theta \in S_B} \mathbb{E}_{X \sim \mathcal{D}} [(\mathcal{L}_{\theta_0 - \theta}(X) - \mathcal{L}_\theta(X))], \quad (\text{Prune})$$

Classwise unlearning involves removing the model's ability to make accurate predictions on a chosen class, called the forget class with distribution D<sup>f</sup> , while maintaining the statistical performance on the remaining classes - called the remain classes, with distribution Dr. In the setting of equation [Edit](#page-3-1), we have M = 2, D<sup>1</sup> = D<sup>f</sup> , D<sup>2</sup> = Dr, α<sup>1</sup> = −1 and α<sup>2</sup> = κ > 0. Solving this problem ensures that the loss on D<sup>f</sup> increases, while the loss on D<sup>r</sup> decreases, with κ penalizing the extent to which <sup>E</sup>X∼D<sup>f</sup> [L<sup>θ</sup>0−θ(X)] is allowed to increase. We write this as

$$\theta^* = \arg \min_{\theta \in S_B} \mathbb{E}_{X \sim \mathcal{D}_r} [\kappa(\mathcal{L}_{\theta_0 - \theta}(X) - \mathcal{L}_\theta(X))] - \mathbb{E}_{X \sim \mathcal{D}_f} [\mathcal{L}_{\theta_0 - \theta}(X) - \mathcal{L}_\theta(X)]. \quad (\text{Forget})$$

Challenges in editing models without the training data or loss function? Unlike works such as [Jia](#page-11-6) [et al.](#page-11-6) [\(2023\)](#page-11-6) and the references therein, fine-tuning or retraining the model is not possible in this setting. Thus to effectively edit the behavior of a network, it is necessary to identify the components that are responsible for making predictions. These can be characterized as components which when modified, significantly change the behaviour of the network. The key challenge is thus:

Problem Statement: Solve equation **[Prune](#page-3-2)** or equation **[Forget](#page-3-3)** without access to original training data and loss function which was used to obtain θ0.

It is well known that pruning or perturbing a large number of components significantly affects statistical performance [\(Hoefler et al., 2021\)](#page-11-4). Thus, it is necessary to identify a *small subset of editable components*; components which are editable can be removed to aid an editing task. In the case of pruning, components that have no effect on the performance of the model are editable, whereas for model unlearning, components required only for the prediction of the forget class are editable. We use this insight to pursue the stated problem and develop algorithms to address it.

# 4 INDENTIFYING EDITABLE COMPONENTS THROUGH HIFI COMPONENTS

As stated in the previous section, editing well-trained models without access to the training data or loss function requires identifying components that have a disproportionate impact on the models's

**224**

**236 237**

**254**

**256**

**259**

predictive performance. In this section, we propose the notion of High-Fidelity (HiFi) components, and hypothesize that HiFi components are what govern a model's predictive performance. We empirically validate our hypothesis and provide a template for model editing algorithms derived from it.

#### 4.1 WHICH FEATURES ARE DISTRIBUTIONALLY SIMILAR TO THE OUTPUT FEATURES?

We provide the empirical observation that in many layers of deep networks, there are only a few filters for which the input contribution distribution is similar to that of the output distribution. In Figure [2,](#page-4-0) we show the relative reconstruction error after removing filters from a selection of layers of a ResNet50 trained on CIFAR10 - we use the expected reconstruction error as a measure of distributional similarity. We see that in well-trained models, a small subset of filters - between 5% and up to 30% of the number of filters in the layer - generate input contributions that are distributionally similar to the aggregate feature maps. This observation motivates us to edit models *by identifying those components whose input contributions are distributionally similar to the feature maps.* We call such components High Fidelity (HiFi) components, which we define in the sequel.

![](_page_4_Figure_4.jpeg)

![](_page_4_Figure_5.jpeg)

(a) Reconstruction error for layer 1, block 0, conv 2.

(b) Reconstruction error for layer 4, block 2, conv 2

Figure 2: Comparision of the fidelity scores of two different layers

### 4.2 HIGH FIDELITY COMPONENTS AND THE FIDELITY SCORE

Suppose Y <sup>l</sup>+1(X) is the feature map generated by layer l + 1, and suppose A l+1 i (X) is the ith input contribution, as defined in equation [1.](#page-2-1) We say the i-th component in layer l is a high-fidelity (HiFi) component if the distribution of the input contribution A l+1 i (X), D l+1 i in layer l + 1 is similar to the distribution of Y <sup>l</sup>+1(X), D<sup>l</sup>+1. HiFi components are those with input contributions that can reconstruct the aggregate feature map[<sup>2</sup>](#page-4-1) . To capture this, we analyze the dissimilarity between the distributions of Yˆ <sup>l</sup>+1(X) = Y <sup>l</sup>+1(X)−E<sup>X</sup> -Y <sup>l</sup>+1(X) and Aˆl+1 i (X) = A l+1 i (X)−E<sup>X</sup> -A l+1 i (X) . We define FS(i), a *Fidelity score* that measures the similarity between an input contribution and the aggregate feature map, below.

$$\text{FS}(i) = \text{DIS}(\hat{\mathcal{D}}^{l+1}, \hat{\mathcal{D}}_i^{l+1}) = \left( \frac{\mathbb{E}_X \left[ \|\hat{Y}^{l+1}(X) - \beta_i \hat{A}_i^{l+1}(X)\|^2 \right]}{\mathbb{E}_X \left[ \|\hat{Y}^{l+1}(X)\|^2 \right]} \right)^{\frac{1}{2}} \quad (3)$$

where 
$$\beta_i = \mathbb{E}_X \left[ \langle \hat{Y}^{l+1}(X), \hat{A}_i^{l+1}(X) \rangle \right] \mathbb{E}_X \left[ \|\hat{A}_i^{l+1}(X)\| \right]^{-2}$$

In the above definition, the smaller the value of DIS(Dˆl+1 , Dˆl+1 i ) (or higher the value of βi) is, better the reconstructability of Y in the mean-square sense. Furthermore, note that we can apply equation [3](#page-4-2) on a channel-by-channel basis by considering the distributions of a single output feature map in a layer; we add an the additional subscript c to indicate that the feature map (and the input contribution) are generated by the cth component in the layer. In well-trained models we often observe that a small number of components have relatively lower DIS scores than the rest. Identifying such components is key to understanding the statistical behavior of model outputs, and hence will be the most critical insight for the subsequent development of our algorithms.

<sup>2</sup>This motivates the name HiFi components: Components whose sum can accurately reconstruct the output with Hi-Fidelity

**289 290 291**

**294**

**301**

**304**

**306**

**309**

**314 315**

**318 319**

The Role of β<sup>i</sup> and RowSum: β<sup>i</sup> is a variant of the Tensor correlation between the input contribution A l+1 i and the feature map Y <sup>l</sup>+1. Furthermore, we can show that

$$\text{DIS}(\hat{\mathcal{D}}^{l+1}, \hat{\mathcal{D}}_i^{l+1})^2 = \mathbb{E}_X \left[ \|\hat{Y}^{l+1}(X)\|^2 - \beta_i^2 \|\hat{A}_i^{l+1}(X)\|^2 \right] / \mathbb{E}_X \left[ \|\hat{Y}^{l+1}(X)\|^2 \right] \quad (4)$$

highlights the relation between FS(i) and β<sup>i</sup> - if ∥Aˆl+1 i (X)∥ is roughly equivalent for all i, then FS(i) is low when β<sup>i</sup> is large. Thus, a heuristic for identifying HiFi components is finding components for which β<sup>i</sup> is large. Moreoever, note that β<sup>i</sup> can be written as the sum of the elements of the row of a matrix, motivating the naming of the heuristic RowSum. Specifically, βiE<sup>X</sup> h ∥Aˆl+1 i (X)∥ i = P <sup>j</sup> Qij , where Qij = <sup>E</sup><sup>X</sup> h ⟨Aˆl+1 i , Aˆl+1 j ⟩ i . We examine this in greater detail in Appendix [E,](#page-25-0) along with an examination of the reconstruction error after the BatchNorm layers. Based on our empirical observations that a small subset of components in well-trained models generate input contributions that are distributionally similar to the feature maps, we now state the main hypothesis of our work. We validate our hypothesis empirically in Section [7.](#page-8-1)

Hypothesis 1. *Suppose we have a well-trained model with parameters* W = (W1, · · · , WL)*. We hypothesize that the HiFi components of this model contribute most to the predictions of the model, and those components that are not high fidelity can be discarded without affecting the performance of the model.*

Algorithm 1: Model Editing by Identifying HiFi Channels Input: Model fθ<sup>0</sup> with layer indices [L], layerwise budgets {Bl}l∈[L] , data distributions D1, D<sup>f</sup> , D<sup>r</sup> Output: Edited model fθE for l ∈ [L] do Compute FS(i) using equation [3](#page-4-2) on D, D<sup>f</sup> , D<sup>r</sup> Determine which components to edit Recover accuracy on D sis 1 states that only the HiFi components - a small subset of the components in a layer - are responsible for the model's predictions. Thus, it facilitates model editing as the distributional similarity between input contributions and aggregate feature maps, as measured using equation [3,](#page-4-2) can be used as a surrogate for the impact of removing that component on the loss function. Thus, leveraging this hypothesis, we can either *prune* the HiFi components to increase the loss (for instance, for classwise unlearning tasks), or *retain* them to ensure the loss remains low (for instance, for structured pruning). We provide a generic algorithmic recipe for model editing using HiFi components, specialized for the tasks of classwise unlearning and structured pruning in Algorithm [1;](#page-5-0) these are discussed in greater detail in Section [6.](#page-7-1)

### Using HiFi Components for Model Editing Hypothe-

# 5 BNFIX: AN ALTERNATIVE TO RETRAINING BY RESETTING BN STATISTICS

In this section, we analyze BatchNorm1D in single branch networks during inference and how the change in distribution due to editing affects the relationship between the loss and BatchNorm parameters. Using this, we derive an algorithm to correct stored statistics after editing. This update has been previously employed in pruning literature [\(Frantar et al., 2022\)](#page-10-5), but to the best of our knowledge, this is the first work to provide theoretical basis to the update in a distributional setting.

Analysis of BatchNorm at Inference BatchNorm at inference shifts the distribution of the intermediate representation at the output of a layer to have mean β and standard deviation γ. These are parameters of the model which are minimize a loss function L as described in Section [2](#page-2-2) .We use the following fact to analyze the loss in terms of the intermediate representation at the output of a layer.

Fact 1 (Stochastic Mean Value theorem). *Let* f *be a twice differentiable real valued function from* R d *to* <sup>R</sup> *and* H<sup>f</sup> (x) *be the Hessian at any* x ∈ <sup>R</sup> d *. For any point* c ∈ R <sup>d</sup> *and a multivariate random variable* X ∈ R <sup>d</sup> *with finite second order moments, there exists a random variate* t ∈ (0, 1) *such that*

$$f(\mathbf{X} + \mathbf{c}) = f(\mathbf{c}) + \nabla f(\mathbf{c})^\top \mathbf{X} + \frac{1}{2} \mathbf{X}^\top \mathbf{H}_f(\mathbf{c} + \mathbf{t}\mathbf{X}) \mathbf{X}$$

For a proof, see Corollary 2 in [Yang & Zhou](#page-14-2) [\(2021\)](#page-14-2). Though for the case discussed here the above fact suffices, but one could potentially use similar facts which can be deduced from other techniques such as Delta Method [\(Benichou & Gail, 1989\)](#page-10-6) or obtain a result on the expectation such as [\(Massey](#page-12-5)

**329**

**334**

**354 355 356**

**358 359**

**361**

**364**

**369**

[& Whitt, 1993\)](#page-12-5). Using the optimality of the learned parameters of BatchNorm and Fact [1,](#page-5-1) we make some assumptions on the first and second-order derivatives on the loss for a well trained model in terms of the learned parameters of BatchNorm layers.

A.1 For a well-trained model ∇L(β) = 0, the gradient with respect to the shift parameter is zero. A.2 For a fixed G and any β, we can bound the eigenvalues of the hessian with a constant K for all inputs. Formally, ∥HL(GZ + β)∥<sup>2</sup> ≤ K for all random variables Z ∈ <sup>R</sup> d such that E(Z 2 i ) = 1, E(Zi) = 0 for all β ∈ <sup>R</sup> d . Here, the norm is the spectral norm of a matrix.

The assumptions [A.1](#page-5-1) and [A.2](#page-5-1) capture the model's "well-trained-ness" on the objective function L and follow from the first and second-order necessary conditions of optimality. We note that [A.1](#page-5-1) would not hold if the input distribution to the network was different from that of the training distribution. The constant K captures the smoothness of the loss function with respect to the parameter β and subsumes the effect of the rest of the network, which may contain more linear and non linear layers. Equipped with these assumptions about well-trained models, we derive a bound on the average loss over the learned distribution in terms of the learned parameters of the BatchNorm layer. With the observation that <sup>E</sup>[V (X)] = β, the term L(E[V (X)]) = L(β) reflects the loss of the averaged intermediate representation.

Lemma 1 (Loss of a well trained model expressed with BatchNorm). *Consider a model that satisfies assumptions [5.](#page-5-1) We can express an upper bound on the expected loss during inference in terms of the statistics of the output of the BatchNorm layer* V (X) ∈ <sup>R</sup> m*.*

$$|\mathbb{E}[\mathcal{L}(V(X))] - \mathcal{L}(\beta)| \leq \frac{K}{2} \|\gamma\|^2 \quad (5)$$

*proof sketch.* We prove this with fact [1](#page-5-1) and using the statistics of V (X). A full proof of Lemma [1](#page-6-2) can be found in [D.1.](#page-23-0)

Algorithm 2: BNFix

Input :Batch Norm Layer l with m channels, dataset D = {Xi}

N i=1

for c ∈ [m] do µ l <sup>c</sup> ← <sup>1</sup> N P<sup>N</sup> <sup>b</sup>=1 Y l c (Xb);

> σ 2(l) <sup>c</sup> ← P<sup>N</sup> b=1 (<sup>Y</sup> <sup>l</sup> c (Xb )−µ<sup>l</sup> c N−1

How Editing affects BatchNorm We now study how editing affects the statistics of the output of the batch norm layer and the loss. Using lemma [1,](#page-6-2) we analyse the effect on the objective L due to the change in the intermediate distribution to state Theorem [1.](#page-6-0) It shows that the loss is upper bounded by a quadratic function of the difference of the mean of the distribution and ratio of the variances. This allows us to qualitatively measure the effect of the shift in distribution on the loss function.

Theorem 1. *Let the* l th*layer of a network be a BatchNorm layer as described in [2](#page-2-3) with stored data statistics* µ<sup>c</sup> *and* σ 2 c *. Editing components of preceeding layers causes a change in the distribution of the intermediate representation to some* Y (p) (X)*, with modified moments* µ (p) *and* (σ (p) ) 2 *. The output of BatchNorm after editing is then,* V(p) = GZ(p) + β *where* Z (p) = Y (p) c (X)−µ<sup>c</sup> σ<sup>c</sup> *. Then,*

$$|\mathbb{E}[\mathcal{L}(\mathbf{V}^{(p)}(X))] - \mathcal{L}(\beta)| \leq \frac{K}{2} \left( \sum_{i=1}^d \gamma_i^2 \left( \left( \frac{\sigma_i^{(p)}}{\sigma_i} \right)^2 + \left( \frac{\mu_i^{(p)} - \mu_i}{\sigma_i} \right)^2 \right) \right) \quad (6)$$

*proof sketch.* We prove this result using the properties of normalization and apply Lemma [1.](#page-6-2) The full proof of this theorem can be found in [D.1.](#page-23-0)

Based on Theorem [1,](#page-6-0) we observe that updating stored statistics to represent the new moments of the intermediate representations after editing, i.e., setting µ<sup>i</sup> = µ (p) i and σ<sup>i</sup> = σ (p) i , restores the upper bound on the loss function to Lemma [1.](#page-6-2) However, the bound suggests that only channels for which the coefficient of γ 2 i in equation [6](#page-6-3) is greater than 1 should be updated to decrease the upper bound. We study this in Appendix [B.9](#page-20-0) and emperically show that updating the statistics of all channels leads to larger accuracy recovery in the case of pruning. Algorithm [2](#page-6-4) shows the update procedure for the stored statistics of a single batch norm layer. This gradient-free procedure does not require training samples and can be implimented using a small number of samples obtained from distributional access. In Appendix [B.2,](#page-17-0) we display the effectiveness of the algorithm on a simple synthetic task.

**381**

**384**

**386**

# 6 MODEL EDITING THROUGH CORRELATION STRUCTURE OF COMPLEX INTERCONNECTIONS

Algorithm 3: Compute HiFi channels over Coupled channels

Input: Model, keepRatio p, Samples D = {Xi}

N i=1

Output: HiFi channels

Function ComputeHiFiSet(*Coupled Channels* CC*,*

p*,* D): for layer l ∈ CC do for *o* ∈ C out do Compute R o according to Equation RowSum; H l <sup>o</sup> ← Top<sup>p</sup> (R l return S l,o∈[C<sup>l</sup> out] *H* o Computational cost of Algorithm [3.](#page-7-2) Let N be the number of data points used to estimate the saliency and M<sup>l</sup> be the complexity of computing the input contribution at layer l for a single sample in a DFC with m layers. The complexity to compute the set of HiFi channels for an output channel of a layer is, t l sal = O(NM<sup>l</sup>C l ind l ). To select the HiFi components for the DFC, the top p elements for each layer and output channel in the DFC are collected, this costs O( P<sup>m</sup> <sup>l</sup>=1 C l out(C l in log C l in + t l sal)). We compare this with the BGSC algorithm [\(Narshana et al., 2022\)](#page-12-2) which has a quadratic dependence on the number of layers in the network, as opposed to the proposed work which is linear in the number of layers.

A key challenge in applying the HiFi hypothesis [4](#page-3-0) is identify HiFi components across groups of interconnected layers in complex networks. We propose Algorithm [3](#page-7-2) to identify HiFi components over all layers in a DFC to extend the HiFi hypothesis to networks with complex interconnections.

Fidelity Compensation by Weight Rescaling In order to improve the model's performance *without* fine-tuning, we propose a distributional approach to modifying the weights to regain accuracy, by modifying the weights of layer l+ 1 after pruning layer l (similarly, we can modify the weights of *feed out layers* after pruning the feed-in layers of a DFC). Unlike prior work which modifies the weights of entire filters with a single parameter [\(Xie et al., 2021;](#page-14-3) [Halabi et al., 2022\)](#page-10-7), our result modifies the weights of individual convolutional kernels, thereby granting a more fine-grained approach to weight compensation. First, we define the reconstruction error as follows.

$$\text{RE}_c^{l+1}(v) = \mathbb{E}[\|Y_c^{l+1}(X) - \sum_{i \in [C_{in}]} v_i \Phi_i^l(X) W_{ci}^{l+1}\|^2] \quad (7)$$

where v ∈ R <sup>C</sup>in . With this definition, we state the solution to the post-pruning fidelity compensation problem, and the reconstruction error improvement in Theorem [2.](#page-7-0)

Theorem 2. *Let* s <sup>l</sup> ∈ {0, 1} <sup>C</sup>in = [1K; 0Cin−K]*, where* 1<sup>K</sup> *is a vector of* K *ones, and* 0Cin−<sup>K</sup> *is a vector of* Cin − K *zeros; we ignore the subscripts for brevity in the sequel. Define* δ<sup>c</sup> ∈ <sup>R</sup> <sup>C</sup>in *such that* δci = 0 *when* s<sup>i</sup> = 0*. We solve* Wˆ <sup>l</sup>+1 ci <sup>=</sup> <sup>ˆ</sup><sup>δ</sup> l+1 ci <sup>W</sup><sup>l</sup>+1 ci *, where* δ l+1 ci = [ˆ<sup>δ</sup> l+1 ci ; 0Cin−K] *that satisfies*

$$\hat{\delta}_c^{l+1} = \arg \min_{\delta_c \in \mathbb{R}^K} \text{RE}_c^l([\delta, 0]) = P_c^{-1} p_c \quad \text{and} \quad \frac{\text{RE}_c^l(s^l) - \text{RE}_c^l(\bar{\delta}_{C_{in}}^{l+1})}{\text{RE}_c^l(s^l)} \leq 1 - \frac{\|1 - \bar{\delta}_{C_i}^{l+1}\|^2}{\kappa(Q_c^{l+1})(C_{in} - K)} \quad (8)$$

*where* δ ⋆ c *is a vector containing the optimal values of* δci*,* Qc,ij = <sup>E</sup> -(W<sup>l</sup>+1 cj ) <sup>⊤</sup>Φ<sup>j</sup> (X) <sup>⊤</sup>Φi(X)W<sup>l</sup>+1 ci *,* Pc,ij = Qc,ij *and* pc,i = <sup>E</sup> -(Y l+1 c ) <sup>⊤</sup>Φ l i (X)W<sup>l</sup>+1 ci *when* s<sup>i</sup> , s<sup>j</sup> = 1*, and* κ(Qc,ij ) *denotes the condition number of* Qc,ij *.*

Based on the RowSum heuristic, fidelity compensation scheme [6,](#page-7-3) and BNFix [5,](#page-5-2) following the recipe of [1,](#page-5-0) we develop, CoBRA(Correlation based editing with Batchnorm Re-Adjustment), a model editing framework for pruning and classwise forgetting. We provide the key components of our proposed pruning and unlearning algorithm. Detailed algorithms are presented in Appendix [B.11.](#page-20-1)

CoBRA-P. Compute: Compute HiFi channels using Algorithm [3](#page-7-2) using distributional samples. Determine: Retain HiFi components Recover: Compute weight compensation according to equation [8](#page-7-4) and perform BNFix using distributional samples.

CoBRA-U. Compute: Compute HiFi channels using Algorithm [3](#page-7-2) using distributional samples from the *forget class*. Determine: Discard HiFi components Recover: Compute weight compensation according to equation [8](#page-7-4) and perform BNFix using distributional samples of the *remain* class.

# 7 EXPERIMENTS

In this section, we present experimental validation of our method on pruning and class unlearning tasks for CNNs with complex interconnections like ResNets to answer the following questions.

(Q1) HiFi Hypothesis. Is it true that there is a small set of High-fidelity channels in a well-trained network? (Q2) Effectiveness of CoBRA-P. Does CoBRA-P result in better accuracy-sparsity tradeoff compared to other data-free algorithms? (Q3) BNFix replace retraining. How does BNFix fare against fine-tuning using synthetic samples when pruning models? (Q4) CoBRA-U for unlearning. Is classwise unlearning, as posed by [Jia et al.](#page-11-6) [\(2023\)](#page-11-6), feasible without fine-tuning? If yes, how does CoBRA-U fare against their method? (Q5) Total Recall of BN. What role do batch norm statistics play in class forgetting, and how can BNFix help in recovering accuracy?

Datasets and architectures. We perform experiments on models including ResNet50/101 and VGG19 trained on CIFAR10/100 and ImageNet datasets.

Distributional Access. As a proxy for distributional access to data in CIFAR10/100 experiments, we use samples that are synthetically generated using image generation models. Details of synthetically generated samples are available in Appendix [B.1.](#page-16-0) For ImageNet experiments, we use the test split, which contains 100,000 images without labels to identify HiFi channels. Note that test split, as suggested by the name, is not used to evaluate the performance of ImageNet models. For pruning experiments on ImageNet, we perform full retraining instead of BNFix.

CoBRA Hyperparameters. We discuss the hyperparameters used for CoBRA-P/U in Appendix [B.10](#page-20-2) Validating HiFi hypothesis. To answer [\(Q1\),](#page-8-2) we compute the reconstruction error described in equation [3](#page-4-2) for 3 different untrained and trained models on CIFAR10 using 1000 samples from the CIFAR10 validation set. We present these sorted values averaged across different trained and untrained models for every layer in Appendix [B.12.](#page-21-1) We make several observations based on these results. First, for most layers, there is a diversity of scores in trained models compared to untrained models, where the scores of all the channels in untrained models are concentrated around a single value. Second, in trained models, there is a small subset of channels, typically less than 10%, which have fidelity scores less than 1. Thus, this validates the HiFi hypothesis, answering [\(Q1\)](#page-8-2)

### 7.1 PRUNING EXPERIMENTS: EXPLORING [\(Q2\)](#page-8-3) AND [\(Q3\)](#page-8-4)

Baselines. To compare the performance of CoBRA-P against other data-free methods, we compare against DFPC [\(Narshana et al., 2022\)](#page-12-2), a state-of-the-art data-free structured pruning algorithm for networks with complex interconnections. To gauge the efficacy of BNFix against retraining with distributional access, we compare against L2-norm-based structured pruning, which computes grouped saliencies for a coupled channel based on the L<sup>2</sup> norm of the weights of its filters. We *train* the model obtained with L<sup>2</sup> norm-based structured pruning *using the synthetic set* for comparison. To the best of our knowledge, these are the only baselines addressing structured pruning of coupled channels in the data free regime.

Training details. Details of pre-trained networks and post-training are given in Appendix [B.4.](#page-17-1)

Results of Pruning Experiments. Table [1](#page-9-0) presents the results of pruning experiments on ResNet-50. We observe that for a similar drop in accuracy in the training-free regime, we gain at least 50% larger reduction in FLOPS and at least 10% larger reduction in parameters. In the training regime, we observe that for similar drop in accuracy, CoBRA-P obtains 60% fewer parameters. To answer [\(Q2\),](#page-8-3) we find that CoBRA-P, for most cases, results in better accuracy-vs-sparsity tradeoff when compared to other data-free algorithms. To answer [\(Q3\),](#page-8-4) BNFix is able to outperform fine-tuning in some cases using synthetic samples. While in a few cases, it does not, it still leads to a reasonably good performance when compared to no-finetuning.

# 7.2 FORGETTING EXPERIMENTS: EXPLORING [\(Q4\)](#page-8-5) AND [\(Q5\)](#page-8-1)

Metrics. We report the forget and retain accuracy averaged across 10 classes of the CIFAR10 dataset. Additional details. Experiments with VGG-19 architecture are present in Appendix [B.7](#page-19-0) where we

**504**

**506**

**509**

**514 515 516**

**518 519**

**524**

**529**

| Dataset Algorithm            | Acc.(%) | RF    | RP    |
|------------------------------|---------|-------|-------|
| Unpruned                     | 94.99   | 1x    | 1x    |
| DFPC (Narshana et al., 2022) | 90.25   | 1.46x | 2.07x |
| L 2                          | 15.91   | 4.07x | 4.71x |
| L 2 w/ ST                    | 90.12   | 4.07x | 4.71x |
| CoBRA-P(n)                   | 92.64   | 1.74x | 1.64x |
| CoBRA-P                      | 91.02   | 4.07x | 5.36x |
| Unpruned                     | 78.85   | 1x    | 1x    |
| DFPC                         | 70.31   | 1.27x | 1.22x |
| L 2                          | 16.77   | 1.93x | 1.40x |
| L 2 w/ ST                    | 73.83   | 1.93x | 1.40x |
| CoBRA-P(n)                   | 72.96   | 1.40x | 1.10x |
| CoBRA-P                      | 70.93   | 1.93x | 1.38x |
| Unpruned                     | 76.1    | 1x    | 1x    |
| ThiNet (Luo et al., 2017)    | 71.6    | 3.46x | 2.95x |
| GReg-2 (Wang et al., 2021)   | 73.9    | 3.02x | 2.31x |
| OTO (Chen et al., 2021)      | 74.7    | 2.86x | 2.81x |
| DFPC                         | 73.8    | 3.46x | 2.65x |
| CoBRA-P                      | 73.25   | 3.60x | 4.46x |

Table 1: Experiments of CoBRA-P on CIFAR10, CIFAR100 and ImageNet compared with baselines for ResNet-50. ST=Synthetic Training, training using synthetic samples. CoBRA(n) is the CoBRA algorithm without using BNFix or Weight compensation. RF=relative FLOP reduction, RP=relative parameter reduction

| Algorithm                | FA(%) | RA.    | PR    |
|--------------------------|-------|--------|-------|
|                          | 94.99 | 94.99  |       |
| Jia et al. (2023)        | 5.54  | 99.11  |       |
| CoBRA-U(0.003)(no BNFix) | 4.22  | 91.131 | 1.0M  |
| CoBRA-U(0.003)           | 90.61 | 90.629 | 1.0M  |
| CoBRA-U(0.2)             | 20.90 | 78.786 | 3.63M |

Table 2: Class forgetting on CIFAR10 with ResNet-50. CoBRA-U(p) indicates the hyperparameter for Algorithm [3.](#page-7-2) For p = 0.003, we only prune the last 12 convolution layers and for the last 30 convolution layers for p = 0.2. FA=Forget Accuracy, RA = Remain Accuracy, PR=Parameters removed

#### make similar observations.

Results of Class-Unlearning Experiments. We report the results of our algorithm in Table [2](#page-9-0) and Table [5.](#page-19-1) To answer [\(Q4\),](#page-8-5) we observe that it is possible to perform unlearning even without finetuning to retain performance on the forgotten class. However, we also make the observation that it is possible to recover the accuracy of a forgotten class by updating the batch norm statistics by using *only samples from the remaining class*. We call this phenomenon the BN Recall. Thus, answering [\(Q5\),](#page-8-1) it is necessary to modify the stored statistics in BN layers to truly forget class information.

### 7.3 DISCUSSION OF EMPIRICAL RESULTS.

In this section, we empirically answer questions [\(Q1\)](#page-8-2) to [\(Q5\).](#page-8-1) With [\(Q1\),](#page-8-2) we show that the for each layer of a network, there exists a small set of High-Fidelity channels that contribute to the performance of the network. To answer [\(Q2\),](#page-8-3) we conclude that CoBRA-P, for most cases, leads to a better sparsity vs. accuracy tradeoff against baseline data-free algorithms by at least 50% larger reduction in FLOPs. We also find, to answer [\(Q3\),](#page-8-4) that BNFix sometimes results in better performance as compared to fine-tuning when using synthetic samples. However, BNFix is always better than no-finetuning. With reference to [\(Q4\),](#page-8-5) we find that it is possible to perform unlearning even without finetuning to retain performance on the forgotten class. In trying to answer [\(Q5\),](#page-8-1) we observe that when only remain class samples are for BNFix, it causes a significant increase in forget class performance.

# 8 CONCLUSION

In this paper, we study model editing in the setting where both training data and loss functions are not available, a setting not studied before. Our main contributions are algorithms devised through correlation analysis of Hifi-components- introduced for the first time here- for both Pruning Complex networks and Class Forgetting. We highlight the importance of BatchNorm statistics, which when updated, yields predictions which can be as good as those obtained from a retrained network. We provide both empirical evidence as well as a formal explaination. The results obtained here, specially those related to identifying Hi-fi components, can open doors to new research avenues useful for understanding Deep Networks. One direction for future work is to use different measures of similarity between distributions, including moment matching, Wasserstein distances, and other divergences.

Limitations: The techniques proposed in this work are effective when the number of classes is less than the width of the network. This may be especially true for unlearning tasks, which implicitly requires that each class is learned by disjoint set of filters.

**554 555 556**

**559**

**561**

**564**

**569**

**579**

**584**

# REFERENCES


[1] Jacques Benichou and Mitchell H. Gail. A delta method for implicitly defined random variables. *The American Statistician*, 43(1):41–44, 1989. ISSN 00031305. URL [http://www.jstor.org/](http://www.jstor.org/stable/2685169) [stable/2685169](http://www.jstor.org/stable/2685169). Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, and John Guttag. What is the state of neural network pruning? *Proceedings of machine learning and systems*, 2:129–146, 2020. Lucas Bourtoule, Varun Chandrasekaran, Christopher A Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. Machine unlearning. In *2021 IEEE Symposium on Security and Privacy (SP)*, pp. 141–159. IEEE, 2021. Tianyi Chen, Bo Ji, Tianyu Ding, Biyi Fang, Guanyi Wang, Zhihui Zhu, Luming Liang, Yixin Shi, Sheng Yi, and Xiao Tu. Only train once: A one-shot neural network training and pruning framework, 2021. URL <https://arxiv.org/abs/2107.07467>. Xiaohan Ding, Tianxiang Hao, Jianchao Tan, Ji Liu, Jungong Han, Yuchen Guo, and Guiguang Ding. Resrep: Lossless cnn pruning via decoupling remembering and forgetting. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 4510–4520, 2021. Ronen Eldan and Mark Russinovich. Who's harry potter? approximate unlearning in llms. *arXiv preprint arXiv:2310.02238*, 2023. Gongfan Fang, Xinyin Ma, Mingli Song, Michael Bi Mi, and Xinchao Wang. Depgraph: Towards any structural pruning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 16091–16101, 2023. Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In *International Conference on Learning Representations*, 2018. Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, and Michael Carbin. Pruning neural networks at initialization: Why are we missing the mark?, 2021. Elias Frantar, Sidak Pal Singh, and Dan Alistarh. Optimal brain compression: A framework for accurate post-training quantization and pruning, 2022. URL [https://openreview.net/](https://openreview.net/forum?id=ksVGCOlOEba) [forum?id=ksVGCOlOEba](https://openreview.net/forum?id=ksVGCOlOEba). Rohit Gandikota, Joanna Materzynska, Jaden Fiotto-Kaufman, and David Bau. Erasing concepts from diffusion models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 2426–2436, 2023. Aditya Golatkar, Alessandro Achille, and Stefano Soatto. Eternal sunshine of the spotless net: Selective forgetting in deep networks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 9304–9312, 2020. Siavash Golkar, Michael Kagan, and Kyunghyun Cho. Continual learning via neural pruning. *arXiv preprint arXiv:1903.04476*, 2019. Sven Gowal, Sylvestre-Alvise Rebuffi, Olivia Wiles, Florian Stimberg, Dan Calian, and Timothy Mann. Improving robustness using generated data. In *Proceedings of the 35th International Conference on Neural Information Processing Systems*, NIPS '21, Red Hook, NY, USA, 2024. Curran Associates Inc. ISBN 9781713845393. Laura Graves, Vineel Nagisetty, and Vijay Ganesh. Amnesiac machine learning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 35, pp. 11516–11524, 2021. Varun Gupta, Christopher Jung, Seth Neel, Aaron Roth, Saeed Sharifi-Malvajerdi, and Chris Waites. Adaptive machine unlearning. *Advances in Neural Information Processing Systems*, 34:16319– 16330, 2021. Marwa El Halabi, Suraj Srinivas, and Simon Lacoste-Julien. Data-efficient structured pruning via

[2] **604**

[3] **606**

[4] **614 615**

[5] **617**

[6] **619**

[7] **629**

[8] **634**

[9] **636**

[10] Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. *arXiv preprint arXiv:1510.00149*, 2015. Babak Hassibi and David Stork. Second order derivatives for network pruning: Optimal brain surgeon. *Advances in neural information processing systems*, 5, 1992. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016. Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In

[11] I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 30. Curran Associates, Inc., 2017. URL [https://proceedings.neurips.cc/paper\\_files/paper/](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf) [2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf). Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. *Journal of Machine Learning Research*, 22(241):1–124, 2021. Andrew G Howard. Mobilenets: Efficient convolutional neural networks for mobile vision applications. *arXiv preprint arXiv:1704.04861*, 2017. Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 4700–4708, 2017. Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *International conference on machine learning*, pp. 448–456. pmlr, 2015. Zachary Izzo, Mary Anne Smart, Kamalika Chaudhuri, and James Zou. Approximate data deletion from machine learning models. In *International Conference on Artificial Intelligence and Statistics*, pp. 2008–2016. PMLR, 2021. Saachi Jain, Hannah Lawrence, Ankur Moitra, and Aleksander Madry. Distilling model failures as directions in latent space. *arXiv preprint arXiv:2206.14754*, 2022. Jinghan Jia, Jiancheng Liu, Parikshit Ram, Yuguang Yao, Gaowen Liu, Yang Liu, Pranay Sharma, and Sijia Liu. Model sparsity can simplify machine unlearning. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. URL [https://openreview.net/forum?](https://openreview.net/forum?id=0jZH883i34) [id=0jZH883i34](https://openreview.net/forum?id=0jZH883i34). Donggyu Joo, Eojindl Yi, Sunghyun Baek, and Junmo Kim. Linearly replaceable filters for deep network channel pruning. In *The 34th AAAI Conference on Artificial Intelligence,(AAAI)*, 2021. Sangamesh Kodge, Gobinda Saha, and Kaushik Roy. Deep unlearning: Fast and efficient gradient-free class forgetting. *Transactions on Machine Learning Research*. Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In F. Pereira, C.J. Burges, L. Bottou, and K.Q. Weinberger (eds.), *Advances in Neural Information Processing Systems*, volume 25. Curran Associates, Inc., 2012. URL [https://proceedings.neurips.cc/paper\\_files/paper/](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) [2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf). Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, and Eleni Triantafillou. Towards unbounded machine unlearning. *Advances in Neural Information Processing Systems*, 36, 2024. Yann LeCun, John Denker, and Sara Solla. Optimal brain damage. *Advances in neural information processing systems*, 2, 1989. URL [https://proceedings.neurips.cc/paper%](https://proceedings.neurips.cc/paper%5Ffiles/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf) [5Ffiles/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.](https://proceedings.neurips.cc/paper%5Ffiles/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf) [pdf](https://proceedings.neurips.cc/paper%5Ffiles/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf).

[12] **648 649 654 656 659 661 664 665 669 674 678 679 680 681 682 684 686 689 690 691 694 695 696 697 698 699 700** Bailin Li, Bowen Wu, Jiang Su, and Guangrun Wang. Eagleeye: Fast sub-net evaluation for efficient neural network pruning. In *European conference on computer vision*, pp. 639–654. Springer, 2020. Guihong Li, Hsiang Hsu, Radu Marculescu, et al. Machine unlearning for image-to-image generative models. *arXiv preprint arXiv:2402.00351*, 2024. Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. In *International Conference on Learning Representations*, 2017. URL [https:](https://openreview.net/forum?id=rJqFGTslg) [//openreview.net/forum?id=rJqFGTslg](https://openreview.net/forum?id=rJqFGTslg). Mingbao Lin, Rongrong Ji, Yan Wang, Yichen Zhang, Baochang Zhang, Yonghong Tian, and Ling Shao. Hrank: Filter pruning using high-rank feature map. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 1529–1538, 2020. Xiaofeng Lin, Seungbae Kim, and Jungseock Joo. Fairgrape: Fairness-aware gradient pruning method for face attribute classification. In *European Conference on Computer Vision*, pp. 414–432. Springer, 2022. Liyang Liu, Shilong Zhang, Zhanghui Kuang, Aojun Zhou, Jing-Hao Xue, Xinjiang Wang, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Group fisher pruning for practical network compression. In *International Conference on Machine Learning*, pp. 7021–7032. PMLR, 2021a. Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows, 2021b. URL <https://arxiv.org/abs/2103.14030>. Jian-Hao Luo, Jianxin Wu, and Weiyao Lin. Thinet: A filter level pruning method for deep neural network compression. In *Proceedings of the IEEE international conference on computer vision*, pp. 5058–5066, 2017. William A. Massey and Ward Whitt. A probabilistic generalization of taylor's theorem. *Statistics & Probability Letters*, 16(1):51–54, 1993. ISSN 0167-7152. doi: https://doi.org/10. 1016/0167-7152(93)90122-Y. URL [https://www.sciencedirect.com/science/](https://www.sciencedirect.com/science/article/pii/016771529390122Y) [article/pii/016771529390122Y](https://www.sciencedirect.com/science/article/pii/016771529390122Y). P Molchanov, S Tyree, T Karras, T Aila, and J Kautz. Pruning convolutional neural networks for resource efficient inference. In *5th International Conference on Learning Representations, ICLR 2017-Conference Track Proceedings*, 2019a. Pavlo Molchanov, Arun Mallya, Stephen Tyree, Iuri Frosio, and Jan Kautz. Importance estimation for neural network pruning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 11264–11272, 2019b. Chaitanya Murti, Tanay Narshana, and Chiranjib Bhattacharyya. Tvsprune-pruning nondiscriminative filters via total variation separability of intermediate representations without fine tuning. In *The Eleventh International Conference on Learning Representations*, 2022. Preetum Nakkiran, Behnam Neyshabur, and Hanie Sedghi. The deep bootstrap framework: Good online learners are good offline generalizers, 2021. URL [https://arxiv.org/abs/2010.](https://arxiv.org/abs/2010.08127) [08127](https://arxiv.org/abs/2010.08127). Tanay Narshana, Chaitanya Murti, and Chiranjib Bhattacharyya. Dfpc: Data flow driven pruning of coupled channels without data. In *The Eleventh International Conference on Learning Representations*, 2022. Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, and Quoc Viet Hung Nguyen. A survey of machine unlearning. *arXiv preprint arXiv:2209.02299*, 2022.

[13] **704**

[14] **706**

[15] **709**

[16] **721**

[17] **724**

[18] **729 730**

[19] **754**

[20] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and

[21] R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019. URL [https://proceedings.neurips.cc/paper\\_files/paper/](https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf) [2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf). Prafull Prakash, Chaitanya Murti, Saketha Nath, and Chiranjib Bhattacharyya. Optimizing dnn architectures for high speed autonomous navigation in gps denied environments on edge devices. In *Pacific Rim International Conference on Artificial Intelligence*, pp. 468–481. Springer, 2019. Tilman Räuker, Anson Ho, Stephen Casper, and Dylan Hadfield-Menell. Toward transparent ai: A survey on interpreting the inner structures of deep neural networks. In *2023 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)*, pp. 464–483. IEEE, 2023. Sabyasachi Sahoo, Mostafa Elaraby, Jonas Ngnawe, Yann Pequignot, Frédéric Precioso, and Christian Gagné. Layerwise early stopping for test time adaptation. *arXiv preprint arXiv:2404.03784*, 2024. Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry. How does batch normalization help optimization? *Advances in neural information processing systems*, 31, 2018. Shibani Santurkar, Dimitris Tsipras, Mahalaxmi Elango, David Bau, Antonio Torralba, and Aleksander Madry. Editing a classifier by rewriting its prediction rules. *Advances in Neural Information Processing Systems*, 34:23359–23373, 2021. Ayush Sekhari, Jayadev Acharya, Gautam Kamath, and Ananda Theertha Suresh. Remember what you want to forget: Algorithms for machine unlearning. *Advances in Neural Information Processing Systems*, 34:18075–18086, 2021. Juwon Seo, Sung-Hoon Lee, Tae-Young Lee, Seungjun Moon, and Gyeong-Moon Park. Generative unlearning for any identity. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 9151–9161, 2024. Harshay Shah, Andrew Ilyas, and Aleksander Madry. Decomposing and editing predictions by modeling model computation. *arXiv preprint arXiv:2404.11534*, 2024. Maying Shen, Hongxu Yin, Pavlo Molchanov, Lei Mao, Jianna Liu, and Jose M Alvarez. Structural pruning via latency-saliency knapsack. *Advances in Neural Information Processing Systems*, 35: 12894–12908, 2022. Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In *International Conference on Learning Representations*, 2015. URL [http:](http://arxiv.org/abs/1409.1556) [//arxiv.org/abs/1409.1556](http://arxiv.org/abs/1409.1556). Nimit Sohoni, Jared Dunnmon, Geoffrey Angus, Albert Gu, and Christopher Ré. No subclass left behind: Fine-grained robustness in coarse-grained classification problems. *Advances in Neural Information Processing Systems*, 33:19339–19352, 2020. Hidenori Tanaka, Daniel Kunin, Daniel L Yamins, and Surya Ganguli. Pruning neural networks without any data by iteratively conserving synaptic flow. *Advances in Neural Information Processing Systems*, 33:6377–6389, 2020. Anvith Thudi, Gabriel Deza, Varun Chandrasekaran, and Nicolas Papernot. Unrolling sgd: Understanding factors influencing machine unlearning. In *2022 IEEE 7th European Symposium on Security and Privacy (EuroS&P)*, pp. 303–319. IEEE, 2022. Chaoqi Wang, Guodong Zhang, and Roger Grosse. Picking winning tickets before training by preserving gradient flow. *arXiv preprint arXiv:2002.07376*, 2020a. Huan Wang, Can Qin, Yulun Zhang, and Yun Fu. Neural pruning via growing regularization. *arXiv preprint arXiv:2012.09243*, 2020b.

[22] **756 757 759 761 764 766 769 771 772 773 774 779 780 781 784 786 787 788 789 790** Huan Wang, Can Qin, Yulun Zhang, and Yun Fu. Neural pruning via growing regularization. In *International Conference on Learning Representations*, 2021. URL [https://openreview.](https://openreview.net/forum?id=o966_Is_nPA) [net/forum?id=o966\\_Is\\_nPA](https://openreview.net/forum?id=o966_Is_nPA). Junxiao Wang, Song Guo, Xin Xie, and Heng Qi. Federated unlearning via class-discriminative pruning. In *Proceedings of the ACM Web Conference 2022*, pp. 622–632, 2022. Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. A comprehensive survey of continual learning: Theory, method and application. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2024. Zhenyi Wang, Enneng Yang, Li Shen, and Heng Huang. A comprehensive survey of forgetting in deep learning beyond continual learning. *arXiv preprint arXiv:2307.09218*, 2023. Alexander Warnecke, Lukas Pirch, Christian Wressnegger, and Konrad Rieck. Machine unlearning of features and labels. *arXiv preprint arXiv:2108.11577*, 2021. Zhouyang Xie, Yan Fu, Shengzhao Tian, Junlin Zhou, and Duanbing Chen. Pruning with compensation: Efficient channel pruning for deep convolutional neural networks, 2021. URL <https://arxiv.org/abs/2108.13728>. Tianyun Yang, Juan Cao, and Chang Xu. Pruning for robust concept erasing in diffusion models. *arXiv preprint arXiv:2405.16534*, 2024. Yifan Yang and Xiaoyu Zhou. A note on taylor's expansion and mean value theorem with respect to a random variable, 2021. URL <https://arxiv.org/abs/2102.10429>. Hongxu Yin, Pavlo Molchanov, Jose M Alvarez, Zhizhong Li, Arun Mallya, Derek Hoiem, Niraj K Jha, and Jan Kautz. Dreaming to distill: Data-free knowledge transfer via deepinversion. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 8715–8724, 2020. Ruichi Yu, Ang Li, Chun-Fu Chen, Jui-Hsin Lai, Vlad I Morariu, Xintong Han, Mingfei Gao, Ching-Yung Lin, and Larry S Davis. Nisp: Pruning networks using neuron importance score propagation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 9194–9203, 2018. Shixing Yu, Zhewei Yao, Amir Gholami, Zhen Dong, Sehoon Kim, Michael W Mahoney, and Kurt Keutzer. Hessian-aware pruning and optimal neural implant. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pp. 3880–3891, 2022.

[23] **791**

[24] **804 805 806**

[25] **814 815**

[26] **817**

[27] **819**

[28] **829**

[29] **834**

[30] **836**

[31] **854**

[32] **856**
# APPENDIX

This appendix is organised as follows:

- 1. Appendix [A](#page-15-0) contains details of related work
- 2. Appendix [B](#page-16-1) contains additional experimental details
- 3. Appendix [C](#page-21-0) contains details about BatchNorm
- 4. Appendix [D](#page-23-1) contains derivations and proofs not presented in the main body.

### A RELATED WORK

## A.1 MODEL EDITING

In this subsection, we discuss model editing, which refers to techniques by which model parameters are perturbed in order to change or influence the statistical performance of the model. A variety of tasks fall under this umbrella, including pruning, model unlearning [Shah et al.](#page-13-8) [\(2024\)](#page-13-8), debiasing [Santurkar et al.](#page-13-7) [\(2021\)](#page-13-7), and continual learning [Sahoo et al.](#page-13-9) [\(2024\)](#page-13-9).

### A.1.1 EDITING CLASSIFIERS

Interpreting and editing classifier models is an active area of research, motivated by problems such as subclass stratification (wherein subgroups within classes of a dataset can exhibit significantly different statistical performance[\)Sohoni et al.](#page-13-10) [\(2020\)](#page-13-10) and debiasing [Santurkar et al.](#page-13-7) [\(2021\)](#page-13-7); [Jain et al.](#page-11-10) [\(2022\)](#page-11-10); [Shah et al.](#page-13-8) [\(2024\)](#page-13-8). The methods proposed in the latter works are of particular interest. In [Jain et al.](#page-11-10) [\(2022\)](#page-11-10), CLIP embeddings are used to find "failure directions" between samples upon which the model succeeds and those on which the model fails using an SVM; these "directions" are then used to design a variety of interventions in the weight space. In [Santurkar et al.](#page-13-7) [\(2021\)](#page-13-7), classifier prediction rules are edited by using learned rank-1 updates on a subset of layers of a DNN. Most pertinently, in [Shah et al.](#page-13-8) [\(2024\)](#page-13-8), an exhaustive approach to component attribution is used, and a variety of tasks including classwise unlearning, debiasing, editing individual predictions, and improving subpopulation robustness.

In the sequel, we discuss other methods that show that model unlearning can also be achieved via model editing.

# A.1.2 EDITING OTHER MODELS

Model editing, while of interest to classifier models, has gained more interest in generative modeling. For instance, component editing and pruning have been successfully applied to model editing tasks in GAN[sLi et al.](#page-12-6) [\(2024\)](#page-12-6); [Seo et al.](#page-13-11) [\(2024\)](#page-13-11) and diffusion models [Yang et al.](#page-14-5) [\(2024\)](#page-14-5), particularly for unlearning tasks.

### A.2 MACHINE UNLEARNING

In this subsection, we provide a detailed literature survey on machine unlearning, both with and without model editing. Machine unlearning assumes that a model f(·) is given, trained on a dataset D. The dataset is then partitioned into D<sup>r</sup> (i.e. the *retain* or *remember* set) and D<sup>f</sup> (the *forget* set). The goal of machine unlearning is to minimize the accuracy on D<sup>f</sup> while maintaining the accuracy on Dr.

# A.2.1 MACHINE UNLEARNING WITHOUT MODEL EDITING

Machine unlearning has gained importance in recent years owing to data privacy and security concerns [Bourtoule et al.](#page-10-2) [\(2021\)](#page-10-2); [Nguyen et al.](#page-12-4) [\(2022\)](#page-12-4). A wide variety of works exist to address this problem. Several works aim to forget data points, even in the adaptive setting, while maintaining the accuracy of the model, such as [Sekhari et al.](#page-13-12) [\(2021\)](#page-13-12); [Gupta et al.](#page-10-9) [\(2021\)](#page-10-9); [Izzo et al.](#page-11-13) [\(2021\)](#page-11-13); [Golatkar et al.](#page-10-3) [\(2020\)](#page-10-3). The work in [Sekhari et al.](#page-13-12) [\(2021\)](#page-13-12) also provides bounds on the number of samples that a model can be allowed to forget before accuracy degradation. Machine unlearning is also a significant

**869**

**874**

**884**

**886**

**889 890 891**

**904**

**906**

**909**

**917**

area of research in the space of large language models, as noted in [Kurmanji et al.](#page-11-11) [\(2024\)](#page-11-11); [Eldan &](#page-10-10) [Russinovich](#page-10-10) [\(2023\)](#page-10-10), and generative models [Gandikota et al.](#page-10-11) [\(2023\)](#page-10-11).

Another aspect of machine unlearning is selective forgetting, wherein classes, groups, or sets of samples are forgotten from the network, as described in [Wang et al.](#page-14-6) [\(2023\)](#page-14-6) and the references therein. This connects machine unlearning to the continual learning setting as well, as described in [Wang](#page-14-7) [et al.](#page-14-7) [\(2024\)](#page-14-7) and the references cited there. There are a variety of approaches to selective or classwise forgetting, many of which require retraining or fine-tuning on subsets of the data. Fine-tuning, which includes methods such as [Golatkar et al.](#page-10-3) [\(2020\)](#page-10-3); [Warnecke et al.](#page-14-8) [\(2021\)](#page-14-8), requires retraining the model on Dr, assuming that after sufficient iterations, the accuracy on D<sup>f</sup> would be degraded. Other works, such as [Graves et al.](#page-10-12) [\(2021\)](#page-10-12); [Thudi et al.](#page-13-13) [\(2022\)](#page-13-13) use gradient *ascent* on the loss function with D<sup>f</sup> , thereby destroying the accuracy of the model on D<sup>f</sup> .

### A.2.2 MACHINE UNLEARNING WITH MODEL EDITING

Recent works have demonstrated the promise of model unlearning by *editing* models. In [Jia et al.](#page-11-6) [\(2023\)](#page-11-6); [Sahoo et al.](#page-13-9) [\(2024\)](#page-13-9), tools for unstructured pruning are leveraged to analyze machine unlearning on sparse models, and the impact of model sparsity on such tasks. More recently, works such as [Shah](#page-13-8) [et al.](#page-13-8) [\(2024\)](#page-13-8); [Kodge et al.;](#page-11-12) [Wang et al.](#page-14-1) [\(2022\)](#page-14-1) directly uses structured pruning for model unlearning, by identifying components responsible for classwise predictions and removing them.

### A.3 STRUCTURED PRUNING

Structured pruning is a popular technique for improving real-world performance of models - in terms of metrics such as inference time, power consumption, and throughput - without requiring additional specialized hardware or software [Hoefler et al.](#page-11-4) [\(2021\)](#page-11-4); [Blalock et al.](#page-10-13) [\(2020\)](#page-10-13). Unlike unstructured pruning (see [Frankle & Carbin](#page-10-14) [\(2018\)](#page-10-14); [Frankle et al.](#page-10-15) [\(2021\)](#page-10-15) and the references therein for a more detailed discussion), wherein individual weights are removed, structured pruning directly reduces the number of matrix-matrix multiplications, thereby improving performance [Hoefler et al.](#page-11-4) [\(2021\)](#page-11-4). Early work on structured pruning involved pruning neurons in feedforward networks, such as [LeCun](#page-11-3) [et al.](#page-11-3) [\(1989\)](#page-11-3); [Hassibi & Stork](#page-11-14) [\(1992\)](#page-11-14). More recent work typically utilizes derivatives of the loss function, such as [Molchanov et al.](#page-12-7) [\(2019a](#page-12-7)[;b\)](#page-12-8); [Shen et al.](#page-13-2) [\(2022\)](#page-13-2); [Li et al.](#page-12-9) [\(2020\)](#page-12-9), which use gradients, or Hessian [Liu et al.](#page-12-1) [\(2021a\)](#page-12-1); [Yu et al.](#page-14-9) [\(2022\)](#page-14-9); [Wang et al.](#page-13-14) [\(2020a\)](#page-13-14). More recently, [Lin et al.](#page-12-10) [\(2022\)](#page-12-10) proposes estimating class-conditional gradient based saliency scores for identifying filters responsible for class-wise or group-wise predictions, with a view toward fair pruning.

### A.3.1 STRUCTURED PRUNING IN THE DATA-FREE REGIME

The space of pruning without access to the training data or loss function remains an under-researched area. There are a variety of methods that do not use training data to generate saliency scores for filters, such as [Yu et al.](#page-14-10) [\(2018\)](#page-14-10), which uses an L1 reconstruction error bound, [Lin et al.](#page-12-11) [\(2020\)](#page-12-11) which uses the rank of feature maps, [,Li et al.](#page-12-12) [\(2017\)](#page-12-12) which uses weight norms, and [Joo et al.](#page-11-8) [\(2021\)](#page-11-8) which uses linear combinations of filters to replace redundant filters. These methods do not directly apply them to pruning in the data-free regime. In this work, we assume access to the training data distributions, with which we derive derivative-free meausres of importance of filters based on correlations between the input contributions they generate.

# B ADDITIONAL EXPERIMENTS

# B.1 SYNTHETIC DATASETS

# B.1.1 CIFAR5M

For experiments with the CIFAR10 dataset, we use CIFAR5M, a dataset containing 6 million synthetic CIFAR-10-like images sampled from a Diffusion model and labelled by a Big-Transfer model[\(Nakkiran et al., 2021\)](#page-12-13), which we randomly sample 10,000 samples from each of the 10 classes to create our dataset. This dataset has an FID[\(Heusel et al., 2017\)](#page-11-15) of 15.95 with respect to the CIFAR10 training set. This dataset is obtained from <https://github.com/preetum/cifar5m>.

**924**

**929**

**954**

**956**

**959**

**961**

| Dataset   | Model Architecture | Loss | Original Acc.(%) | σ     | Loss  | +Noising Acc.(%) | Loss | +BNFix Acc.(%) |
|-----------|--------------------|------|------------------|-------|-------|------------------|------|----------------|
|           | ResNet-50          | 0.21 | 94.99            |       |       |                  |      |                |
|           |                    |      |                  | 0.010 | 2.2   | 32.31            | 0.5  | 87.16          |
|           |                    |      |                  | 0.012 | 4.96  | 10.67            | 1.12 | 72.91          |
| CIFAR-10  |                    |      |                  | 0.014 | 20.49 | 9.89             | 1.87 | 37.07          |
|           | VGG19              | 0.31 | 93.50            |       |       |                  |      |                |
|           |                    |      |                  | 0.010 | 6.04  | 18.75            | 0.5  | 86.33          |
|           |                    |      |                  | 0.012 | 15.11 | 11.62            | 1.23 | 59.52          |
|           |                    |      |                  | 0.014 | 69.69 | 10.05            | 2.01 | 26.20          |
|           | ResNet-50          | 0.9  | 78.85            |       |       |                  |      |                |
|           |                    |      |                  | 0.010 | 3.00  | 30.31            | 1.61 | 64.06          |
|           |                    |      |                  | 0.012 | 4.52  | 2.84             | 2.42 | 51.14          |
| CIFAR-100 |                    |      |                  | 0.014 | 5.31  | 0.97             | 3.36 | 31.35          |
|           | VGG19              | 1.46 | 72.02            |       |       |                  |      |                |
|           |                    |      |                  | 0.010 | 1.62  | 62.74            | 1.55 | 66.02          |
|           |                    |      |                  | 0.012 | 2.27  | 48.94            | 1.62 | 62.71          |
|           |                    |      |                  | 0.014 | 3.75  | 13.58            | 1.80 | 58.21          |
| ImageNet  | ResNet-50          | 0.96 | 76.15            | 0.010 | 4.38  | 20.56            | 1.73 | 63.63          |

Table 3: Effect of BNFix on noising a network. σ represents the variance of the noise added to the network

### B.1.2 CIFAR100-DDPM

For experiements with the CIFAR100 dataset, we use CIFAR100-DDPM[\(Gowal et al., 2024\)](#page-10-16), which we randomly downsample to contain 1,000 samples from each of the 100 classes. This dataset has an FID of 4.74 with respect to the CIFAR100 training set. We randomly sample 1,000 samples from each of the 100 classes to create our dataset. This dataset is obtained from [https://github.com/google-deepmind/deepmind-research/tree/](https://github.com/google-deepmind/deepmind-research/tree/master/adversarial_robustness/iclrw2021doing) [master/adversarial\\_robustness/iclrw2021doing](https://github.com/google-deepmind/deepmind-research/tree/master/adversarial_robustness/iclrw2021doing).

### B.2 BATCHNORM NOISING

To illustrate the effect of BNFix, we will first consider an artificial editing task we call model noising. Although not a practical procedure, it serves to illustrate the effect of BNFix. The model is "edited" by adding gaussian noise to all of the learned parameters of the network. We add a zero mean random value to every learned parameter(including biases) of the model and apply BNFix for 5 iterations over the synthetic set. Table [3](#page-17-2) showcases the performance of the model before and after noising in terms of the accuracy of the model and the value of the crossentropy loss averaged over the test set. Noising causes a dramatic fall in accuracy and increase in loss but BNFix is able to recover from around 10% to 60% of the validation accuracy across models and datasets.

### B.3 EFFECT OF NUMBER OF SAMPLES FOR BNFIX

To understand the number of samples required for BNFix, we use random pruning to prune a ResNet50 model trained on the CIFAR10 dataset to achieve 2x FLOP reduction. We then apply BNFix using the synthetic set. In Figure [3,](#page-18-0) we showcase the effect of the size of the synthetic set use and show a 95% confidence interval over 4 runs with different random subsets. We see that after around 1500 samples the gains due to adding additional samples diminish.

### B.4 TRAINING PROCEDURE

Pretraining procedure: For CIFAR10 and CIFAR100, we train models using SGD Optimizer with a momentum factor of 0.9 and weight decay of 5 × 10−<sup>4</sup> for 200 epochs using Cosine Annealing step sizes with an initial learning rate of 0.1.

ImageNet post training: For ImageNet, we use off-the-shelf pretrained models from Torchvision[\(Paszke et al., 2019\)](#page-13-15). We train the model for 3 epochs after each iteration of CoBRA-P with learning rates of 0.1, 0.01, 0.001. After the pruning ends, we finally prune the network for 200 with a

![](_page_18_Figure_1.jpeg)

Figure 3: BNFix applied to a ResNet-50 model trained on CIFAR10 pruned using random channel pruning to achieve 2x FLOP sparsity.

![](_page_18_Figure_3.jpeg)

Figure 4: BNFix applied to a ResNet-50 model trained on CIFAR10 pruned using different pruning strategies to achieve 3x FLOP sparsity. For random pruning, we display the mean and 95% confidence interval computed over 4 runs.

batch size of 512. We use the SGD Optimizer with a momentum factor of 0.9 and weight decay of × 10−<sup>4</sup> and Cosine Annealed step sizes with an initial learning rate of 0.1.

L<sup>2</sup> Post training procedure: For the synthetic training experiments mentioned in Section [7,](#page-8-0) we first prune the model using L<sup>2</sup> norm as the grouped saliency to a similar sparsity as CoBRA-P. We then train the model using 50000 samples from the synthetic dataset for 100 epochs with a batch size of 128 using SGD optimizer with momentum factor of 0.9 with inital learning rate of 0.01 and a MultiStepLR learning rate scheduler with milestones at 60 and 80 epochs.

# B.5 BNFIX AND PRUNING

We use pruning algorithms like L1, L2, and Random pruning on CIFAR10 trained ResNet-50 models to obtain models with 3x FLOP reduction. We then apply BNFix with 5000 synthetic samples for 20 iterations. Figure [4](#page-18-1) shows the effectiveness of BNFix on these models, recovering upto 65% validation accuracy for this model.

**1029**

**1034**

**1054**

**1056**

**1071**

# B.6 ADDITIONAL PRUNING EXPERIMENTS

| Dataset Model Algorithm | Acc.(%) | RF    | RP    |
|-------------------------|---------|-------|-------|
| CIFAR-100 VGG19         |         |       |       |
| Unpruned                | 72.02   | 1x    | 1x    |
| DFPC                    | 70.10   | 1.26x | 1.50x |
| L 2                     | 56.46   | 1.50x | 2.40x |
| L 2 w/ ST               | 72.42   | 1.50x | 2.40x |
| CoBRA-P                 | 70.26   | 1.51x | 2.31x |
| Unpruned                | 95.09   | 1x    | 1x    |
| DFPC                    | 89.80   | 1.53x | 1.84x |
| L 2 w/ ST               | 90.49   | 4.20  | 5.29x |
| CoBRA-P                 | 91.20   | 4.21x | 4.79x |
| Unpruned                | 93.50   | 1x    | 1x    |
| DFPC                    | 90.25   | 1.46x | 2.07x |
| L 2 w/ ST               | 89.23   | 2.39x | 9.19x |
| CoBRA-P                 | 91.80   | 2.39x | 5.52x |

Table 4: Experiments of CoBRA-P on CIFAR100 compared with baselines. RF=Reduction in FLOPs. RP=Reduction in Parameters. ST=Synthetic training, training using synthetic samples.

### B.7 ADDITIONAL FORGETTING EXPERIMENTS

We report additional experiments on class unlearning on different architectures. For VGG-19 networks, we remove the HiFi channels for the forget class of the last 12 convolution layers.

| Model Algorithm                | Forget Acc.(%) | Remain Acc. | params. removed |
|--------------------------------|----------------|-------------|-----------------|
|                                | 93.50          | 93.50       |                 |
| CoBRA-U(0.001)(no BNFix) VGG19 | 0.86           | 77.85       | 0.79M           |
| CoBRA-U(0.001)                 | 45.87          | 91.31       | 0.79M           |
| CoBRA-U(p=0.2)                 | 5.63           | 84.34       | 3.18M           |

Table 5: Class forgetting on CIFAR10 for VGG19. CoBRA-U(p) indicates the hyperparameter for Algorithm [3.](#page-7-2)

### B.8 CLASS UNLEARNING FOR VISION TRANSFORMERS

In this subsection, we describe how CoBRA-U can be applied to Vision Transformers to perform gradient free class unlearning without training data or access to the loss function. We focus on the SwinTransformer[\(Liu et al., 2021b\)](#page-12-14) architecture and prune linear layers in the network. We use the distributional measure described in [4](#page-3-0) to measure the importance of weights in linear layers of the network for the forget class. We use this measure in the form of an unstructured saliency to prune the weights of linear layers which include the WQ, WK, W<sup>V</sup> and MLP layers in the network. For sequence models like transformers, we compute the expectation described in equation [3](#page-4-2) over all elements in the sequence.

We report class forgetting results on the SwinTransformer[\(Liu et al., 2021b\)](#page-12-14) architecture trained on CIFAR-10. We train the model on the CIFAR10 dataset for 300 epochs from scratch[<sup>3</sup>](#page-19-2) to achieve a validation accuracy of 92.31%. We apply CoBRA-U on the linear layers in a vision transformer.

<sup>3</sup><https://github.com/jordandeklerk/SwinViT>

**1099**

**1104**

**1106**

**1109**

**1119**

| Class   | Forget Acc. | Remain Acc. |
|---------|-------------|-------------|
| Best    | 7.80%       | 48.29%      |
| Average | 40.52%      | 60.50%      |
| Worst   | 90.40%      | 90.74%      |

Table 6: Training free class forgetting on CIFAR10 for SwinTransformer using CoBRA-U. Metrics are reported for the best, worst, and average over all 10 classes.

### B.9 VARIANTS OF BNFIX

Based on our analysis in Theorem [1,](#page-6-0) we develop two additional algorithms. Algorithm [4](#page-20-3) is a variant of BNFix where the stored statistics of only channels whose coefficients in equation [6](#page-6-3) are greater than or equal to one are updated. This ensures that only large terms of the bound are reduced by the update.

Algorithm 4: BNFix-Scale

Input :Batch Norm Layer l with m channels, dataset D = {Xi} N i=1

for c ∈ [m] do (p)

µ

<sup>c</sup> ← <sup>1</sup> N P<sup>N</sup> <sup>b</sup>=1 Y l c (Xb);

σ 2 (p) <sup>c</sup> ← P<sup>N</sup> b=1 (Y c (Xb)−µ (p) c )

N−1

;

a<sup>c</sup> = (σ (p) i

<sup>2</sup>+(µi−µ

(p) i ) )

σ

;

if a<sup>c</sup> ≥ 1 then µ l <sup>c</sup> ← <sup>1</sup> N P<sup>N</sup> <sup>b</sup>=1 Y l c (Xb);

σ 2 <sup>c</sup> ← P<sup>N</sup> b=1 (Y l c (Xb)−µ l c ) 2

N−1

;

We compare the performance of these variants as a substitution for retraining for CoBRA-P for a VGG model pretrained on CIFAR10.

### B.10 HYPERPARAMETERS FOR EXPERIMENTS

We randomly sample 2000 data points from distributional access for computing HiFi channels and for BNFix. For CoBRA-P, we typically set p = 0.05. We perform BNFix for 10 epochs as a substitution for retraining. We use 2000 samples from a synthetic dataset for BNFix. For ImageNet, we use 20000 samples from the unlabelled imagenet test set.

### B.11 DETAILED COBRA ALGORITHMS

In this section we provide details of CoBRA-P and CoBRA-U.

| Algorithm   | Acc.  | RF    | RP    |
|-------------|-------|-------|-------|
|             | 93.50 | 1x    | 1x    |
| No BNFix    | 63.75 | 2.42x | 4.72x |
| BNFix-Scale | 89.10 | 1.97x | 4.18x |
| BNFix       | 92.47 | 1.91x | 3.98x |

**1139 1140 1141 1142 1143 1144 1145 1146 1147 1148 1149 1150 1151** Input: Model, keepRatio p, Samples D = {(X<sup>i</sup> , yi)} N i=1 Output: Edited model Function Prune(*Model,* p*,* D): // Find the set of all coupled channels DFCs ← FindCoupledChannels(Model); for CC ∈ *DFCs* do HiFiChannels← ComputeHiFiSet(*CC,p,D*); EditableChannels← [C CC in ]\HiFiChannels; for l ∈*CC* do for *i* ∈*EditableChannels* do // Prune the input channel in each layer of the DFC PruneInputChannel(l, i); Compute δ (l)<sup>⋆</sup> based on equation [8](#page-7-4) using D; for *i* ∈*HiFiChannels* do InputChannel(l, i) ← δ (l)⋆ · InputChannel(l, i); // Run Algorithm [2](#page-6-4) for all BatchNorm layers in the model for lbn ∈ *FindBNLayers(Model)* do BNFix(lbn, D); Result: Model

**1154**

**1159 1161 1162 1163 1164 1165 1166 1167 1168 1169 1171** Input: Model, keepRatio p, Forget Samples D<sup>f</sup> , Remain samples D<sup>r</sup> Output: Edited model Function Unlearn(*Model,* p*,* D<sup>f</sup> *,* Dr): // Find the set of all coupled channels DFCs ← FindCoupledChannels(Model); for CC ∈ *DFCs* do for l ∈*CC* do HiFiChannels← ComputeHiFiSet(*l,p,Df* ); EditableChannels←HiFiChannels; for *i* ∈*EditableChannels* do // Prune the input channel in each layer of the DFC PruneInputChannel(l, i); Compute δ (l)<sup>⋆</sup> based on equation [8](#page-7-4) using Dr; for *i* ∈*HiFiChannels* do InputChannel(l, i) ← δ (l)⋆ · InputChannel(l, i); // Run Algorithm [2](#page-6-4) for all BatchNorm layers in the model for lbn ∈ *FindBNLayers(Model)* do BNFix(lbn, Dr); Result: Model

**1174 1175**

**1177**

**1183 1184 1185 1186 1187** For multi-channel data like images, BatchNorm is modified "to satisfy the convolution property"[\(Ioffe](#page-11-1) [& Szegedy, 2015\)](#page-11-1). Let Φ l (x) denote the input to the l th layer of a neural network with L layers on input x. Let the l th layer be a Convolution layer with m ouput channels, for a single multi channel sample x, let Y l (x) ∈ <sup>R</sup> <sup>m</sup>×<sup>d</sup> be the flattened representation of the output which is computed according to equation [1.](#page-2-1) The output of the BatchNorm layer(called BatchNorm2D in the multi-

Algorithm 5: CoBRA-P

Algorithm 6: CoBRA-U

### B.12 VALIDATING HIFI HYPOTHESIS

In this section, we present the plots for the fidelity score computed as per [7.](#page-8-1)

# C ADDITIONAL DETAILS ABOUT BATCHNORM

# C.1 BATCHNORM2D

![](_page_22_Figure_1.jpeg)

 Figure 5: Comparison of distributional similarity between input contributions and output feature map.

**1267**

**1281**

**1284**

**1287**

channel case), V (X) = BN<sup>l</sup>+1 γ,β (X) ∈ <sup>R</sup> m×d , is given by

$$V_c(X) = \gamma_c Z_c(X) + \beta_c \mathbf{1} \quad \text{where} \quad Z_c(X) = \frac{Y_c^l(X) - \mu_c \mathbf{1}}{\sigma_c} \quad \forall c \in [m] \quad (9)$$

Where 1 is a vector of ones, µ<sup>c</sup> = N P<sup>N</sup> i=1 Y c (xi) <sup>⊤</sup>1 <sup>d</sup> ≈ <sup>E</sup>X[ Y c (X) <sup>⊤</sup>1 d ] and σ 2 <sup>c</sup> ≈ VarX( Y c (X) <sup>⊤</sup>1 d ) are stored data statistics and γ ∈ R <sup>m</sup> and β ∈ <sup>R</sup> <sup>m</sup> are the learned scale and shift parameters that determine the first two moments of the output of the layer, i.e., for the random variable Vˆ (X) = 1 <sup>⊤</sup>Vc(X)/d, the moments are <sup>E</sup>X[Vˆ (X)] = β<sup>c</sup> and VarX(Vˆ (X)) = γ 2 c .

# C.2 BATCHNORM DURING TRAINING

We describe the behavior of BatchNorm1D during training, which is similar to that of BatchNorm2D. For an input batch of size B, let the output of the linear layer, the l th layer in the network, be ϕ l (X) ∈ R<sup>B</sup>×<sup>d</sup> . Then, the output is given by,

$$Y_c(x_i) = \gamma_c Z_c(x_i) + \beta_c \quad \text{where} \quad Z_c(x_i) = \frac{\phi^l(x_i) - \frac{1}{B} \sum_{b=1}^B \phi_c^l(x_b)}{\sqrt{\frac{1}{B} \sum_{j=1}^B \left( \phi_c^l(x_j) - \frac{1}{B} \sum_{b=1}^B \phi_c^l(x_b) \right)^2}} \quad (10)$$

The estimate over the whole training set from equation [2](#page-2-4) is now replaced with batch estimates. Observe that Z are being normalized and, in an average sense, represent zero mean unit variance random variables. To compute the stored statistics to use during inference, at every forward pass during training, a running estimate of the mean and variance are stored in the layer. This running estimate is used in equation [2.](#page-2-4)

# D PROOFS

In this section, we provide proofs for the main theoretical results proposed in this work. Specifically, we propos

### D.1 PROOF OF LEMMA [1](#page-6-2)

Lemma 1 (Loss of a well trained model expressed with BatchNorm). *Consider a model that satisfies assumptions [5.](#page-5-1) We can express an upper bound on the expected loss during inference in terms of the statistics of the output of the BatchNorm layer* V (X) ∈ <sup>R</sup> m*.*

$$|\mathbb{E}[\mathcal{L}(V(X))] - \mathcal{L}(\beta)| \leq \frac{K}{2} \|\gamma\|^2 \quad (5)$$

*Proof.* From fact [1,](#page-5-1)

$$L(Y(X)) = L(\beta + GZ) = L(\beta) + \nabla L(\beta)^\top GZ + \frac{1}{2} Z(X)^\top GH(\beta, GZ(X)) GZ(X)$$

The proof follows from the assumptions on the Hessian. Comments: |E(X)| ≤ E(|X|).

# D.2 PROOF OF THEOREM [1](#page-6-0)

Theorem 1. *Let the* l th*layer of a network be a BatchNorm layer as described in [2](#page-2-3) with stored data statistics* µ<sup>c</sup> *and* σ c *. Editing components of preceeding layers causes a change in the distribution of the intermediate representation to some* Y (p) (X)*, with modified moments* µ (p) *and* (σ (p) ) 2 *. The output of BatchNorm after editing is then,* V(p) = GZ(p) + β *where* Z (p) = Y (p) c (X)−µ<sup>c</sup> σ<sup>c</sup> *. Then,*

$$|\mathbb{E}[\mathcal{L}(\mathbf{V}^{(p)}(X))] - \mathcal{L}(\beta)| \leq \frac{K}{2} \left( \sum_{i=1}^d \gamma_i^2 \left( \left( \frac{\sigma_i^{(p)}}{\sigma_i} \right)^2 + \left( \frac{\mu_i^{(p)} - \mu_i}{\sigma_i} \right)^2 \right) \right) \quad (6)$$

**1317**

**1319**

**1321**

**1324**

**1334**

*Proof.* Let V(p) be the output at the batchnorm layer for the edited model with E(V(p) ) = µ (p) , E((V (p) i ) 2 ) = µ (p) i 2 + σ (p) i 2 , ∀i ∈ [d].

Define U<sup>i</sup> = V (p) <sup>i</sup> −µ<sup>i</sup> σ<sup>i</sup> . Consider <sup>V</sup>d(p) <sup>=</sup> GU <sup>+</sup> <sup>β</sup>

$$\begin{aligned} |\mathbb{E}(L(\widehat{\mathbf{V}}^{(p)})) - L(\beta)| &\leq \nabla L(\beta)^\top U + \mathbb{E} \frac{1}{2} \|H(\beta, GU)\|_2 \|GU\|_2^2 \\ &\leq \sum_{i=1}^d g_i \gamma_i \frac{\mu_i^{(p)} - \mu_i}{\sigma_i} + \frac{K}{2} \left( \sum_{i=1}^d \gamma_i^2 \left( \left( \frac{\sigma_i^{(p)}}{\sigma_i} \right)^2 + \left( \frac{(\mu_i^{(p)} - \mu_i)}{\sigma_i} \right)^2 \right) \right) \\ & \end{aligned} \quad (11)$$

where g<sup>i</sup> = ∇iL. Applying assumption [5](#page-5-1) finishes the proof.

### D.3 PROOF OF THEOREM [2](#page-7-0)

Theorem 2. *Let* s <sup>l</sup> ∈ {0, 1} <sup>C</sup>in = [1K; 0Cin−K]*, where* 1<sup>K</sup> *is a vector of* K *ones, and* 0Cin−<sup>K</sup> *is a vector of* Cin − K *zeros; we ignore the subscripts for brevity in the sequel. Define* δ<sup>c</sup> ∈ <sup>R</sup> <sup>C</sup>in *such that* δci = 0 *when* s<sup>i</sup> = 0*. We solve* Wˆ <sup>l</sup>+1 ci <sup>=</sup> <sup>ˆ</sup><sup>δ</sup> l+1 ci <sup>W</sup><sup>l</sup>+1 ci *, where* δ l+1 ci = [ˆ<sup>δ</sup> l+1 ci ; 0Cin−K] *that satisfies*

$$\hat{\delta}_c^{l+1} = \arg \min_{\delta_c \in \mathbb{R}^K} \text{RE}_c^l([\delta, 0]) = P_c^{-1} p_c \quad \text{and} \quad \frac{\text{RE}_c^l(s^l) - \text{RE}_c^l(\bar{\delta}_{C_{in}}^{l+1})}{\text{RE}_c^l(s^l)} \leq 1 - \frac{\|1 - \bar{\delta}_{ci}^{l+1}\|^2}{\kappa(Q_c^{l+1})(C_{in} - K)} \quad (8)$$

*where* δ ⋆ c *is a vector containing the optimal values of* δci*,* Qc,ij = <sup>E</sup> -(W<sup>l</sup>+1 cj ) <sup>⊤</sup>Φ<sup>j</sup> (X) <sup>⊤</sup>Φi(X)W<sup>l</sup>+1 ci *,* Pc,ij = Qc,ij *and* pc,i = <sup>E</sup> -(Y l+1 c ) <sup>⊤</sup>Φ l i (X)W<sup>l</sup>+1 ci *when* s<sup>i</sup> , s<sup>j</sup> = 1*, and* κ(Qc,ij ) *denotes the condition number of* Qc,ij *.*

*Proof.* First, note that

$$\begin{aligned} \mathbf{RE}_c^{l+1}([\delta; \mathbf{0}_{C_{in}-K}]) &= \mathbb{E}[\|Y_c^{l+1}(X) - \sum_{i:s_i=0} \delta_i \Phi_i^l(X) W_{ci}^{l+1}\|^2] \\ &= \mathbb{E}[\|\sum_i \delta_i \Phi_i^l(X) W_{ci}^{l+1} - \sum_{i:s_i=0} \delta_i \Phi_i^l(X) W_{ci}^{l+1}\|^2] \\ &= (1 - [\delta; \mathbf{0}_{C_{in}-K}])^\top Q(1 - [\delta; \mathbf{0}_{C_{in}-K}]). \end{aligned}$$

We can rewrite this as

$$\arg \min_{\delta} \text{RE}_c^{l+1}([\delta; \mathbf{0}_{C_{in}-K}]) = \arg \min_{\delta} \delta^\top P_c \delta - 2p_c^\top \delta = P_c^{-1} p_c.$$

To measure the error, note that

$$\text{RE}_c^{l+1}(s^l) = (1 - s^l)^\top Q(1 - s^l).$$

Thus, we have

$$\begin{aligned} \frac{\mathbf{RE}_c^l(s^l) - \mathbf{RE}_c^l(\bar{\mathbf{d}}_c^{l+1})}{\mathbf{RE}_c^l(s^l)} &= 1 - \frac{\mathbf{RE}_c^l(\bar{\mathbf{d}}_{Cin}^{l+1})}{\mathbf{RE}_c^l(s^l)} = 1 - \frac{(1 - \bar{\mathbf{d}}_c^{l+1})^\top \mathbf{Q}(1 - \bar{\mathbf{d}}_c^{l+1})}{(1 - s^l)^\top \mathbf{Q}(1 - s^l)} \\ &\leq 1 - \frac{\lambda_{\min}(\mathbf{Q})\|1 - \bar{\mathbf{d}}_c^{l+1}\|^2}{\lambda_{\max}(\mathbf{Q})(C_{in} - K)} = 1 - \frac{\|1 - \bar{\mathbf{d}}_c^{l+1}\|^2}{\kappa(\mathbf{Q})(C_{in} - K)}. \end{aligned}$$

**1354**

**1371**

**1374**

# E RECONSTRUCTION ERROR AND HIFI COMPONENTS

Reconstruction Error and the Fidelity Score A useful way to identify HiFi components is by measuring the expected reconstruction error, as fidelity implies reconstructability. We define reconstion error as follows. First, let s ∈ {0, 1} <sup>C</sup>in satisfy s<sup>i</sup> = 1 if i ∈ keep and 0 otherwise, and let Yˆ <sup>l</sup> (X) = P i siA<sup>l</sup> i (X). The expected *local reconstruction error* at layer l between Y (X) and Yˆ l (X), as well the reconstruction error of a single output channel, are

$$\begin{aligned} \mathbf{RE}^l(s) &= \mathbb{E}_X \left[ \|Y^l(X) - \hat{Y}^l(X; s)\|^2 \right] \\ \mathbf{RE}^l_c(s) &= \mathbb{E}_X \left[ \|Y^l_c(X) - \hat{Y}^l_c(X; s)\|^2 \right] \quad \text{and} \quad \mathbf{RE}^l(s) = \sum_c \mathbf{RE}^l_c(s). \end{aligned} \quad (13)$$

Next, we consider the reconstruction error after a 1D BatchnNorm layer, assuming that the stored statistics are not updated after editing.

$$\text{RE}_c^{l+1}(s) = \frac{1}{2}\mathbb{E}_X \left[ \left( V_c(X) - \hat{V}_c(X; s) \right)^2 \right] = \frac{\gamma_c^2}{2} \left( 1 + \frac{\hat{\sigma}_c^2}{\sigma_c^2} - \frac{(\hat{\mu}_c - \mu_c)^2}{\sigma_c^2} - 2 \frac{\hat{\sigma}}{\sigma_c} \rho_c(s) \right) \quad (14)$$

where Vc(X), Vˆ <sup>c</sup>(X), µc, σc, µˆc, and σˆ<sup>c</sup> are defined as in equation [2](#page-2-4) .

When we adjust the batchnorm statistics" the reconstruction error is given by,

$$\text{RE}_c^{l+1}(s) = \gamma_c^2(1 - \rho_c(s)) \quad (\text{RE-BN})$$

This adjustment shows that the reconstruction error is a function of the *correlation* between the input contributions and the output. This motivates us to define FS(i) using the centered distributions. In Section [5,](#page-5-2) we rigorously analyze the effect of adjusting the BatchNorm statistics on the loss of the network.

Deriving BNFix From Reconstruction Error Consider the output of the BatchNorm layer before and after pruning where the stored statistics are changed after pruning. Let V (X) = BNγ,β(Y (X), µ, σ) and Vˆ (X; s, µ′ , σ′ ) = BNγ,β(Yˆ (X), µ′ , σ′ ), where <sup>E</sup>[Yc(X)] = µc, Var(Yc(X)) = σ 2 c , <sup>E</sup>[Yˆ <sup>c</sup>(X; s)] = ˆµ<sup>c</sup> and Var(Yˆ <sup>c</sup>(X; s)) = ˆσ<sup>c</sup> 2 . The reconstruction error for output channel c is given by,

$$\text{RE}_c^{l+1}(s) = \frac{1}{2}\mathbb{E}_X \left[ \left( V_c(X) - \hat{V}_c(X; s, \mu', \sigma') \right)^2 \right] = \frac{\gamma_c^2}{2} \left( 1 + \frac{\hat{\sigma}_c^2}{\sigma_c'^2} - \frac{(\hat{\mu}_c - \mu'_c)^2}{\sigma_c'^2} - 2 \frac{\hat{\sigma}}{\sigma_c'} \rho_c(s) \right) \quad (15)$$
where  $\rho_c(s) = \frac{\text{Cov}(Y_c(X), \hat{Y}_c(X; s))}{\sigma_c \sqrt{\text{Var}(\hat{Y}_c(X; s))}}$ . When  $\mu' = \hat{\mu}$  and  $\sigma' = \hat{\sigma}$ , the reconstruction error is given by,
$$\text{RE}_c^{l+1}(s) = \gamma_c^2 (1 - \rho_c(s)) \quad (\text{RE-BN})$$

Reconstruction Error for Structured Pruning Solving equation [Prune](#page-3-2) without access to the training data or loss function can now be formulated as *minimizing the reconstruction error between the edited feature map and the original*. Thus, we formulate the problem of structured pruning with a fixed budget as follows. For a layer with Cin filters, and a sparsity budget of B filters, we write

$$s^* = \arg \min_{s \in \{0,1\}^{C_{in}}} (1-s)^T Q_c(1-s) \quad \text{s.t.} \quad \sum_i s_i \leq B \quad (16)$$

where Q<sup>c</sup> is a symmetric matrix with elements Qcij = W<sup>⊤</sup> ic <sup>E</sup><sup>X</sup> Φi(X) <sup>⊤</sup>Φ<sup>j</sup> (X) Wjc.

The optimization problem in [16](#page-25-1) is a binary optimization problem and thus NP-Hard , and can be reduced to a graph problem like maximum independent set by considering that Q represents the adjacency matrix of a graph. Solving equation [16](#page-25-1) provides an independent set of size B. Based on this observation, we use a simple heuristic, analogous to the degree of each vertex, to find solutions to equation [16.](#page-25-1) Consider minimizing the reconstruction error for a single output channel, say c. With

our graph analogy, we remove vertices with the lowest degree, corresponding to channels with the lowest row sums of the matrix Qc. The row sum corresponding to each channel, Rci is

$$R_{ci} = \sum_j Q_{cij} = \sum_j W_{ic}^\top \mathbb{E}_X [\Phi_i(X)^\top \Phi_j(X)] W_{jc} = W_{ic}^\top \mathbb{E}_X [\Phi_i(X)^\top Y_c(X)] \quad (\text{RowSum})$$

Based on equation [RE-BN](#page-25-2), we normalize the input contribution and the output to zero mean random variables due to the presence of BatchNorm layers. This algorithm is stated formally in Algorithm [1.](#page-5-3)

Reconstruction Error for Classwise Unlearning We aim to use the HiFi hypothesis to *unlearn* a class from a well-trained model, by removing a small, fixed number of filters responsible for predictions of that class. We formulate the problem of classwise unlearning via model editing using the Reconstruction Error to guide the edit. Our goal is to maximize the reconstruction error on the forget class drawn from distribution D<sup>f</sup> while minimizing the reconstruction error on the remaining classes, which we denote by Dr. Similar to equation [16,](#page-25-1) we write

$$\arg \max_{s \in \{0,1\}^{C_{in}}} (1-s)^T (Q_c^f - \alpha Q_c^r) (1-s) \quad \text{s.t.} \quad \sum_i s_i \geq B \quad (\text{Forget})$$

where Q<sup>f</sup> c is a symmetric matrix with elements Q f cij <sup>=</sup> <sup>W</sup><sup>⊤</sup> ic <sup>E</sup>X∼D<sup>f</sup> -Φi(X) <sup>⊤</sup>Φ<sup>j</sup> (X) Wjc, Q<sup>r</sup> c Q<sup>r</sup> cij <sup>=</sup> <sup>W</sup><sup>⊤</sup> ic <sup>E</sup>X∼D<sup>r</sup> -Φi(X) <sup>⊤</sup>Φ<sup>j</sup> (X) Wjc, and α is a hyperparameter that penalizes the reconstruction error on Dr. Our experiments show that typically, setting α = 0 suffices, particularly for wide networks such as ResNet50 and VGG19 trained on CIFAR10.