# CROSS-ENTROPY IS ALL YOU NEED TO INVERT THE DATA GENERATING PROCESS

Patrik Reizinger∗<sup>1</sup> , Alice Bizeul∗1,<sup>2</sup> , Attila Juhos∗<sup>1</sup> ,

Julia E. Vogt<sup>2</sup> , Randall Balestriero<sup>3</sup> , Wieland Brendel<sup>1</sup> , David Klindt<sup>4</sup>

{patrik.reizinger, attila.juhos, wieland.brendel}@tuebingen.mpg.de {alice.bizeul, julia.vogt}@inf.ethz.ch, rbalestr@brown.edu, klindt@cshl.edu

# ABSTRACT

Supervised learning has become a cornerstone of modern machine learning, yet a comprehensive theory explaining its effectiveness remains elusive. Empirical phenomena, such as neural analogy-making and the linear representation hypothesis, suggest that supervised models can learn interpretable factors of variation in a linear fashion. Recent advances in self-supervised learning, particularly nonlinear Independent Component Analysis, have shown that these methods can recover latent structures by inverting the data generating process. We extend these identifiability results to parametric instance discrimination, then show how insights transfer to the ubiquitous setting of supervised learning with cross-entropy minimization. We prove that even in standard classification tasks, models learn representations of ground-truth factors of variation up to a linear transformation under a certain DGP. We corroborate our theoretical contribution with a series of empirical studies. First, using simulated data matching our theoretical assumptions, we demonstrate successful disentanglement of latent factors. Second, we show that on DisLib, a widely-used disentanglement benchmark, simple classification tasks recover latent structures up to linear transformations. Finally, we reveal that models trained on ImageNet encode representations that permit linear decoding of proxy factors of variation. Together, our theoretical findings and experiments offer a compelling explanation for recent observations of linear representations, such as superposition in neural networks. This work takes a significant step toward a cohesive theory that accounts for the unreasonable effectiveness of supervised learning.

# 1 INTRODUCTION

Representation learning is a central task in machine learning, underpinning the success of extracting and encoding meaningful information from data [\(Bengio et al., 2013\)](#page-10-0). Among the various paradigms, supervised learning—particularly classification tasks using cross-entropy minimization—has become the dominant method in deep learning [\(Krizhevsky et al., 2012\)](#page-12-0). Despite its simplicity, this form of supervised learning has led to several intriguing and widely-observed phenomena, including: *neural analogy making* [\(Mikolov et al., 2013\)](#page-13-0), where models seemingly map between related concepts; the *linear representation hypothesis* [\(Park et al., 2023\)](#page-13-1), which posits that interpretable features can be linearly decoded from neural representations; recent work on *superposition* in neural networks [\(Elhage et al., 2022\)](#page-11-0), showing evidence that interpretable features are linearly represented in neural activations [\(Templeton et al., 2024\)](#page-14-0); and the success of *transfer learning* [\(Donahue et al., 2014\)](#page-10-1), where a linear readout can be trained on top of learned representations to solve new tasks. These phenomena suggest that deep learning models encode various features in a manner that allows for linear decoding. Yet, a comprehensive theory that explains why these properties emerge in deep learning models has remained elusive [\(Arora et al., 2016;](#page-10-2) [Park et al., 2023\)](#page-13-1).

We address this gap by building on the theory of [Independent Component Analysis \(ICA\),](#page-30-0) which studies the conditions under which latent variables in probabilistic models can be uniquely identified [\(Comon, 1994;](#page-10-3) [Hyvarinen et al., 2001\)](#page-12-1). Recently, ICA has been extended to nonlinear models [\(Hyvarinen et al., 2023\)](#page-12-2), providing a theoretical foundation for recovering latent variables in a broad ¨ class of machine learning tasks [\(Hyvarinen & Morioka, 2016;](#page-12-3) [Hyvarinen et al., 2019;](#page-12-4) [Gresele et al.,](#page-11-1) [2019;](#page-11-1) [Khemakhem et al., 2020a;](#page-12-5) [Klindt et al., 2021;](#page-12-6) [Khemakhem et al., 2020b;](#page-12-7) [Locatello et al., 2020;](#page-13-2)

<sup>∗</sup> Joint first authorship; <sup>1</sup>Max Planck Institute for Intelligent Systems, Tubingen AI Center, ELLIS Institute, ¨ Tubingen, Germany; ¨ <sup>2</sup>Department of Computer Science, ETH Zurich and ETH AI Center, ETH Z ¨ urich, Z ¨ urich, ¨ Switzerland; <sup>3</sup>Department of Computer Science, Brown University, Rhode Island, USA; <sup>4</sup>Cold Spring Harbor Laboratory, Cold Spring Harbor, New York, USA;

![](_page_1_Figure_1.jpeg)

Figure 1: DIET [\(Ibrahim et al., 2024\)](#page-12-8) learns identifiable features: given N samples and a d−dimensional latent representation, DIET learns a linear (N × d)−dimensional classification head Won top of a nonlinear encoder f through an instance discrimination objective [\(1\)](#page-3-0). For unit-normalized f(xn), DIET maps samples and their augmentations close to the cluster vector v<sup>c</sup> corresponding to the class—as if sampled from a [von Mises-Fisher \(vMF\)](#page-30-1) distribution, centered around vc. For duplicate samples, i.e., matching class labels, the corresponding rows of W will be the same, as shown for x<sup>1</sup> and x<sup>i</sup> with w<sup>1</sup> = w<sup>i</sup> .

[Morioka et al., 2021;](#page-13-3) Halv ¨ [a et al., 2021;](#page-12-9) [Morioka & Hyvarinen, 2023\)](#page-13-4). Most of these advances have ¨ focused on [self-supervised learning \(SSL\)](#page-30-2) [\(Hyvarinen & Morioka, 2016;](#page-12-3) [Hyvarinen et al., 2019;](#page-12-4) [Zimmermann et al., 2021;](#page-14-1) [von Kugelgen et al., 2021;](#page-14-2) [Rusak et al., 2024\)](#page-14-3), i.e., when neural networks ¨ are trained by solving a surrogate (classification) task to learn from unlabeled data—the exceptions that study supervised learning, though either in the multitask setting, or with a single task with additional assumptions, include [\(Ahuja et al., 2022;](#page-10-4) [Lachapelle et al., 2023;](#page-13-5) [Fumero et al., 2023\)](#page-11-2). However, we seek to understand whether similar identifiability guarantees can explain under what conditions cross-entropy-based supervised learning, i.e., when the labels for the classification task are provided in the dataset, recovers interpretable and transferable representations.

Our journey starts with a recent development in SSL: nonlinear ICA has been shown to provide identifiability guarantees in contrastive learning, where models invert the [data generating process \(DGP\)](#page-30-3) and recover latent variables up to linear transformations [\(Hyvarinen et al., 2019;](#page-12-4) [Zimmermann et al.,](#page-14-1) [2021\)](#page-14-1). Building on this insight, we first extend nonlinear ICA to a simple form of SSL—i.e., [para](#page-30-4)[metric instance discrimination \(PID\)](#page-30-4) [\(Dosovitskiy et al., 2014\)](#page-11-3)—through the DIET method [\(Ibrahim](#page-12-8) [et al., 2024\)](#page-12-8), which streamlines the auxiliary task into an instance-discrimination paradigm. We model the [DGP](#page-30-3) in a new, cluster-centric way, and show that DIET's learned representation is linearly related to the ground-truth representation.

From this foundation, we take the crucial step of extending the theoretical framework to the more common paradigm of supervised learning. Specifically, we show that models can recover groundtruth latent variables up to a linear transformation even in standard classification tasks using the cross-entropy loss, which is the most prevalent setting in modern machine learning. By doing so, we aim to explain why deep learning, particularly supervised classification, is so effective in learning interpretable and transferable representations, offering a unifying framework to explain phenomena such as linear representations and neural analogy-making. Thus, our theoretical insights offer a potential explanation for the extraordinary success of supervised deep learning across a wide variety of tasks. Our contributions are

- We propose a cluster-centric [DGP](#page-30-3) as a model for the [parametric instance discrimination](#page-30-4) method of [Ibrahim et al.](#page-12-8) [\(2024\)](#page-12-8) and prove the DGP's linear identifiability (Thm. [1\)](#page-4-0);
- We use our insight to extend the identifiability guarantee to standard cross-entropy-based supervised classification under the a cluster-centric [DGP](#page-30-3) (Thm. [2\)](#page-4-1);
- We provide a "genealogy" of cross-entropy-based classification methods to connect our identifiability results in instance discrimination and supervised classification to auxiliary-variable nonlinear [Independent Component Analysis \(ICA\)](#page-30-0) [\(Hyvarinen et al., 2019\)](#page-12-4) and [self-supervised learning](#page-30-2) [\(SSL\)](#page-30-2) (§ [3.4\)](#page-5-0) [\(Zimmermann et al., 2021\)](#page-14-1);
- We corroborate our findings in synthetic experiments matching our cluster-centric [DGP,](#page-30-3) the DisLib disentanglement benchmark [\(Locatello et al., 2019\)](#page-13-6), and real-world ImageNet-X data [\(Idrissi et al.,](#page-12-10) [2022\)](#page-12-10), showing that the cross-entropy loss, irrespective of the meaningfulness of labels, can lead to linear identifiability of the features (§ [4\)](#page-6-0).

# 2 BACKGROUND

Empirical evidence of a linear latent representation. The *linear representation hypothesis* [\(Park](#page-13-1) [et al., 2023\)](#page-13-1) has lately received a lot of attention. A weak version of this hypothesis could mean that there are directions in neural activation space that correspond to interpretable features. In the

case of *neural analogy making*, [Mikolov et al.](#page-13-0) [\(2013\)](#page-13-0) showed that there exist directions in word embeddings that are interpretable and preserved across input pairs. As an example for encoder f, producing latent variables z, the direction z = f(man)−f(woman) seems to correspond to gender and can be added to other words such as f(king) + z ≈ f(queen). Several datasets, such as the Google Analogy Dataset (GA) [\(Mikolov, 2013\)](#page-13-7) and BATS [\(Drozd et al., 2016\)](#page-11-4), have been developed to evaluate neural analogy-making. These were, for instance, evaluated in [\(Dufter & Schutze, 2019\)](#page-11-5). ¨ Theoretical explanations of linear representations have been proposed for word embeddings by [Arora](#page-10-2) [et al.](#page-10-2) [\(2016\)](#page-10-2) and [Allen & Hospedales](#page-10-5) [\(2019\)](#page-10-5). Both approaches take a statistical learning theory perspective and focus on characterizing the pointwise mutual information. They do not consider cross-entropy-based classification; and, thus, do not make a connection to supervised classification, as we do in Thm. [2.](#page-4-1) [Park et al.](#page-13-1) [\(2023\)](#page-13-1) provide a framework to specify what exactly is meant by the linear representation hypothesis. They also provide a strong, causal hypothesis where finding that a feature is linearly represented does not imply that an intervention on that linear subspace will causally remove the feature from the model output. [Engels et al.](#page-11-6) [\(2024\)](#page-11-6) point out that some latent representations are not linear. This makes intuitive sense if we consider that some latent features, such as the pose of an object, have a non-Euclidean topology that will have to be embedded on a curved manifold in a linear subspace of the latent representation [\(Higgins et al., 2018;](#page-11-7) [Pfau et al., 2020;](#page-13-8) [Keurti et al.,](#page-12-11) [2023\)](#page-12-11). For instance, the quadrature pair of sines and cosines representing rotations in a 2D subspace in [\(Klindt et al., 2021,](#page-12-6) Fig. 15) depends on the object symmetries [\(Bouchacourt et al., 2021\)](#page-10-6). [Roeder](#page-13-9) [et al.](#page-13-9) [\(2020\)](#page-13-9) prove that different models trained with a discriminative objective converge to learning the same latent representation. Importantly, their claim is about the linear relationship between *any two learned* representations, and not the learned and the ground-truth one, as is usually the case in identifiability theory [\(Hyvarinen et al., 2001\)](#page-12-1). They also show this empirically for pairs of models trained on different datasets. Their results are corroborated even with widely varying training factors by [Moschella et al.](#page-13-10) [\(2023\)](#page-13-10). These findings are also supported by recent large scale empirical studies in the converging representations of vision models [\(Chen & Bonner, 2023\)](#page-10-7). This could also explain the recently proposed *platonic representation hypothesis* [\(Huh et al., 2024\)](#page-12-12) about the convergence of representations, the improved disentanglement across model families [\(Du & Xiang, 2021\)](#page-11-8), and the better identifiability of biological mechanisms [\(Genkin & Engel, 2020\)](#page-11-9). However, these insights from the literature fail to connect the linearity of learned representations to the identifiability of the assumed ground-truth [DGP—](#page-30-3)this is the gap our contribution aims to address.

Identifiable weakly-/self-supervised learning and ICA. [Independent Component Analysis \(ICA\)](#page-30-0) theory studies the conditions under which latent variables in probabilistic models can be uniquely identified [\(Comon, 1994;](#page-10-3) [Hyvarinen et al., 2001\)](#page-12-1). *Identifiability* means that the learned representation, at the global optimum of the training loss, relates to the *ground-truth* representation (i.e., the groundtruth latent variables underlying the data) via a "simple" transformation, such as permutations or elementwise invertible transformations—this is different to investigations relating two instances of learned representations, such as in [Roeder et al.](#page-13-9) [\(2020\)](#page-13-9); [Moschella et al.](#page-13-10) [\(2023\)](#page-13-10); [Zhang et al.](#page-14-4) [\(2023\)](#page-14-4). Recently, ICA has been extended to nonlinear models [\(Hyvarinen et al., 2023\)](#page-12-2), providing a theoretical ¨ foundation for recovering latent variables in a broad class of learning tasks [\(Hyvarinen & Morioka,](#page-12-3) [2016;](#page-12-3) [Hyvarinen et al., 2019;](#page-12-4) [Gresele et al., 2019;](#page-11-1) [Khemakhem et al., 2020a;](#page-12-5) [Klindt et al., 2021;](#page-12-6) [Khemakhem et al., 2020b;](#page-12-7) [Locatello et al., 2020;](#page-13-2) [Morioka et al., 2021;](#page-13-3) Halv ¨ [a et al., 2021;](#page-12-9) [Morioka](#page-13-4) ¨ [& Hyvarinen, 2023\)](#page-13-4). Most of these advances have focused on [SSL,](#page-30-2) [\(Hyvarinen & Morioka, 2016;](#page-12-3) [Hyvarinen et al., 2019;](#page-12-4) [Zimmermann et al., 2021;](#page-14-1) [von Kugelgen et al., 2021;](#page-14-2) [Rusak et al., 2024\)](#page-14-3). ¨

# 3 THEORY

This section presents our main theoretical contribution. We start with our motivation to understand [self-supervised learning \(SSL\)](#page-30-2) with the help of the simplified DIET [\(Ibrahim et al., 2024\)](#page-12-8) algorithmic pipeline. For this, we propose a cluster-centric [data generating process \(DGP\)](#page-30-3) that can model semantic classes (§ [3.1\)](#page-3-1). Then we state our main result in § [3.2](#page-4-2) and discuss an intuition behind the identifiability of the representation learned by DIET. We conclude by investigating how DIET fits into the vast literature of (identifiable) [SSL](#page-30-2) and auxiliary-variable [Independent Component Analysis](#page-30-0) [\(ICA\)](#page-30-0) methods (§ [3.4\)](#page-5-0). This leads to a significant result for proving the identifiability of the latents learned via supervised classification under the DIET [DGP](#page-30-3) (§ [3.3\)](#page-4-3). We provide the technical details for [Generalized Contrastive Learning \(GCL\)](#page-30-5) [\(Hyvarinen et al., 2019\)](#page-12-4) in Appx. [B.1](#page-23-0) and InfoNCE [\(Chen](#page-10-8) [et al., 2020;](#page-10-8) [Zimmermann et al., 2021\)](#page-14-1) in Appx. [B.3.](#page-25-0)

Motivation. Despite significant theoretical progress [\(Zimmermann et al., 2021;](#page-14-1) [von Kugelgen et al.,](#page-14-2) ¨ [2021;](#page-14-2) [Rusak et al., 2024\)](#page-14-3), it remains elusive why [SSL](#page-30-2) methods work well in practice. [Rusak et al.](#page-14-3) [\(2024\)](#page-14-3) highlighted two remaining gaps between theory and practice: 1) practitioners often discard

the encoder's last few layers (termed the projector) for better performance, despite identifiability guarantees not reflecting this fact; and 2) the data is presumably clustered, not reflected in the common assumption of a uniform marginal. Despite a similar terminology in auxiliary-variable nonlinear [ICA](#page-30-0) algorithms, such as [Time-Contrastive Learning \(TCL\)](#page-30-6) [\(Hyvarinen & Morioka, 2016\)](#page-12-3) or [GCL](#page-30-5) [\(Hyvarinen et al., 2019\)](#page-12-4), it is unclear how such methods relate to [SSL](#page-30-2) at large. Interestingly, the identifiability proofs for nonlinear [ICA](#page-30-0) partition the model into a separate encoder and a regression function [\(Hyvarinen & Morioka, 2016;](#page-12-3) [Hyvarinen et al., 2019\)](#page-12-4) and prove identifiability for the latent variables after the encoder, but before the regression function. This aligns with the practice of discarding the projector in [SSL](#page-30-2) [\(Bordes et al., 2023\)](#page-10-9), though identifiability results do not reflect this fact [\(Zimmermann et al., 2021;](#page-14-1) [von Kugelgen et al., 2021;](#page-14-2) [Rusak et al., 2024\)](#page-14-3). These observations ¨ served as our motivation to investigate

*How can we extend the identifiability guarantees to more realistic self-supervised classification scenarios, and can we apply these insights to improve our understanding of supervised learning?*

Results overview. We aim to advance our theoretical understanding of [SSL,](#page-30-2) for this, we use the recently proposed DIET [\(Ibrahim et al., 2024\)](#page-12-8) (detailed in § [3.1\)](#page-3-1), which, beyond its simplicity, promises the strongest and most realistic results, based on similarities to [GCL](#page-30-5) [\(Hyvarinen et al., 2019\)](#page-12-4). Namely, DIET uses a separate encoder and classification head, and solves an auxiliary classification task akin to [GCL—](#page-30-5)furthermore, its loss correlates with downstream performance, a non-obvious and welcome fact [\(Rusak et al., 2024\)](#page-14-3). This provides the hope to resolve the two above points by modeling the cluster structure of the data and proving identifiability for the representation used for downstream tasks (Thm. [1\)](#page-4-0). Subsequently, we leverage the insights from our identifiability theory and the DIET pipeline's similarity to *supervised* classification to show how the latter is a special case of DIET, where the sample indices correspond to the semantic class labels (Thm. [2\)](#page-4-1).

#### 3.1 SETUP

DIET [\(Ibrahim et al., 2024\)](#page-12-8). DIET solves an instance classification problem, where each sample x in the training dataset of size N has a unique instance label i. Augmentations do not affect this label. We have a composite model W ◦ f, where the backbone f produces d-dimensional representations, and a linear, bias-free classification head W ∈ R <sup>N</sup>×<sup>d</sup> maps these representations to a logit vector equal in size to the cardinality of the training dataset. If the parameter vector corresponding to logit i is denoted as w<sup>i</sup> , then W effectively computes similarity scores (scalar products) between the wi's and embeddings f(x). DIET trains this architecture to predict the correct instance label using multinomial regression (with f,W and temperature β as learnable variables), i.e., it solves a [parametric instance discrimination \(PID\)](#page-30-4) task [\(Dosovitskiy et al., 2014;](#page-11-3) [Wu et al., 2018\)](#page-14-5):

$$\mathcal{L}_{\text{PID}}(\mathbf{f}, \mathbf{W}, \beta) = \mathbb{E}_{(\mathbf{x}, i)} \left[ -\ln \frac{e^{\beta \langle \mathbf{w}_i, \mathbf{f}(\mathbf{x}) \rangle}}{\sum_j e^{\beta \langle \mathbf{w}_j, \mathbf{f}(\mathbf{x}) \rangle}} \right]. \quad (1)$$

An important fact is that [\(1\)](#page-3-0) is the cross-entropy loss with instance labels, which we will leverage to connect instance discrimination to supervised classification.

The proposed cluster-centric [data generating process \(DGP\).](#page-30-3) To prove the identifiability of the latent variables, we need to formally define a [latent variable model \(LVM\)](#page-30-7) for the [data generating](#page-30-3) [process \(DGP\).](#page-30-3) We take a cluster-centric approach, representing semantic classes by cluster vectors, similar to proxy-based metric learning [\(Kirchhof et al., 2022\)](#page-12-13). Then, we model the samples of a class with a [von Mises-Fisher \(vMF\)](#page-30-1) distribution (intuitively, this is an isotropic multivariate Normal distribution that is restricted to the unit hypershere), centered around the class's cluster vector. This conditional distribution jointly models intra-class sample selection and *augmentations* of samples, together called *intra-class variances*. In contrast to conventional [SSL](#page-30-2) methods such as InfoNCE [\(Zimmermann et al., 2021\)](#page-14-1), this conceptually separates global and local structure in the latent space: 1) the cluster-vectors describe the global structure of the latent space; and 2) the clustercentric conditional in [\(2\)](#page-4-4) describes the local structure. This cluster-centric conditional embodies that data augmentations are selected such that they ought not to change the sample's semantic class. Our conditional does not mean that each sample pair transforms into each other via augmentations *with high probability*. It does mean that—since we assume a [latent variable model \(LVM\)](#page-30-7) on the hypersphere; i.e., all semantic concepts (color, position, etc.) correspond to a continuous latent variable—the latent manifold is connected, or equivalently, that the augmentation graph is connected, which is an assumption used in [\(Wang et al., 2022;](#page-14-6) [Balestriero & LeCun, 2022;](#page-10-10) [HaoChen et al., 2022\)](#page-11-10). We provide an overview of our assumptions, and defer additional details to Assums. [1C](#page-15-0) in Appx. [A:](#page-15-1)

- *(i) There is a finite set of semantic classes* C *, represented by a set of unit-norm* d*-dimensional cluster-vectors* {vc|c ∈ C } ⊆ <sup>S</sup> d−1 *. The system* {vc} *is sufficiently large and spread out.*
- *(ii) Any instance label* i *belongs to exactly one class* c = C(i)*.*
- *(iii) The latent variable* z ∈ S <sup>d</sup>−<sup>1</sup> *of our data sample with instance label* i *is drawn from a [vMF](#page-30-1) distribution with concentration parameter* κ *around the cluster vector* v<sup>c</sup> *of class* c = C(i)*:*

$$z \sim p(z|c) \propto e^{\kappa\langle v_c, z \rangle}. \quad (2)$$

- *(iv) Sample* x *is generated by passing latent* z *through an injective generator function:* x = g(z)*.*

3.2 MAIN RESULT: DIET IDENTIFIES BOTH LATENT VARIABLES AND CLUSTER VECTORS Under Assums. [1,](#page-3-2) we prove the identifiability of both the latent representations z and the cluster vectors, vc, in all four combinations of unit-normalized (i.e., when the latent space is the hypersphere, commonly used, e.g., in InfoNCE [\(Chen et al., 2020\)](#page-10-8)); and non-normalized (as in the original DIET paper [\(Ibrahim et al., 2024\)](#page-12-8)) learned embeddings, z˜, and weight vectors, w<sup>i</sup> . We state a concise version of our result and defer the full treatment and the proof to Thm. [1C](#page-16-0) in Appx. [A:](#page-15-1)

Theorem 1 (Identifiability of latent variables drawn from vMF around cluster vectors. *Simplified.*). *Let* (f,W, β) *globally minimize the DIET objective* [\(1\)](#page-3-0) *under the following additional constraints:*

*C3. the embeddings* f(x) *are unnormalized, while the* wi*'s are unit-normalized. Then* w<sup>i</sup> *identifies the cluster vector* vC(i) *up to an orthogonal linear transformation* O*:* w<sup>i</sup> = OvC(i) *, for any* i*. Furthermore, the inferred latent variables* z˜ = f(x) *identify the ground-truth latent variables* z *up to a scaled orthogonal transformation with the same* O*:* z = κ <sup>β</sup>Oz˜*. C4. neither the embeddings* f(x) *nor the* wi*'s are unit-normalized. Then* w<sup>i</sup> *identifies the cluster vectors* v<sup>c</sup> *up to an affine linear transformation. Furthermore, the inferred latent variables* z˜ *identify the ground-truth latent variables* z *up to a linear transformation.*

*In all cases, the weight vectors belonging to samples of the same class are equal, i.e., for any* i, j*,* C(i) = C(j) *implies* w<sup>i</sup> = w<sup>j</sup> *.*

Intuition. DIET assigns a different (instance) label and a unique weight vector w<sup>i</sup> to each training sample. The cross-entropy objective is optimized if the trained neural network can distinguish between the samples. Thus, the learned representation z˜ = f(x) should capture enough information to distinguish different samples, even from the same class. However, the weight vectors wi's cannot be sensitive to the intra-class sample variance or the sample's instance label i (because the conditional distribution over latent variables is identical for all samples of the same class). This leads to the weight vectors taking the values of the cluster vectors. As cluster vectors only capture some statistics of the conditional [\(1\)](#page-3-0), feature recovery is more fine-grained than cluster identifiability. The interaction between the two is dictated by the cross-entropy loss, which is minimized if the representation z˜ is most similar to its own assigned weight vector w<sup>i</sup> . Fig. [1](#page-1-0) provides a visualization conveying the intuition behind Thm. [1.](#page-4-0)

#### 3.3 SUPERVISED CLASSIFICATION

This section relates our cluster-centric [DGP](#page-30-3) to *supervised* classification. To see how supervised machine learning is a special case of self-supervised approaches, consider that the sample index (i.e., the target of the cross-entropy loss) can be defined *arbitrarily* (as long as Assums. [1](#page-3-2) are still satisfied). This means that many labelings are possible, including the one used for supervised classification. This, *in hindsight* obvious insight has important consequences: it can explain the success of supervised cross-entropy-based classification. Namely, supervised learning performs non-linear [ICA](#page-30-0) under our proposed DGP (Assums. [1\)](#page-3-2). We demonstrate this in §§ [4.1](#page-7-0) and [4.3.](#page-9-0) We state a concise version of our result and defer the full treatment to Appx. [A:](#page-15-1)

Theorem 2 (Identifiability of latent variables drawn from a vMF around class vectors). *Let Assum. [3](#page-20-0) hold, and suppose that a continuous encoder* f : R <sup>D</sup> → <sup>R</sup> d *, a linear classifier* W *with rows* {w<sup>⊤</sup> c | c ∈ C }*, and* β > 0 *globally minimize the cross-entropy objective:*

$$\mathcal{L}_{\text{supervised}}(\mathbf{f}, \mathbf{W}, \beta) = \mathbb{E}_{(\mathbf{x}, C)} \left[ -\ln \frac{e^{\beta \langle \mathbf{w}_C, \mathbf{f}(\mathbf{x}) \rangle}}{\sum_{c' \in \mathcal{C}} e^{\beta \langle \mathbf{w}_{c'}, \mathbf{f}(\mathbf{x}) \rangle}} \right].$$

![](_page_5_Diagram_1.jpeg)

Figure 2: The simplified genealogy of cross-entropy-based classification methods (cf. Tab. [1](#page-5-1) for details): The labeled arrows express how to go from general to special methods. (a) The most general auxiliary-variable [ICA](#page-30-0) framework, [Generalized Contrastive Learning \(GCL\)](#page-30-5) [\(Hyvarinen et al., 2019\)](#page-12-4), yields [Time-Contrastive Learning \(TCL\)](#page-30-6) [\(Hyvarinen & Morioka, 2016\)](#page-12-3) as the special case when the latent conditional is assumed to come from an exponential family (of order one) with a scalar auxiliary variable; (b) [TCL](#page-30-6) relates to non-unit-normalized DIET by further restricting the latent conditional to a [vMF](#page-30-1) distribution; (c) if the neural network used in InfoNCE is partitioned into a linear classifier head and a backbone, the marginal is assumed to be a [vMF](#page-30-1) instead of uniform, we get the unit-normalized version of DIET; (d) if the labeling function in DIET is assumed to assign the semantic class labels to the samples, we get classic supervised training

Intuition: In the context of DIET, the cross-entropy objective encourages the learned representations to align with the cluster vectors corresponding to each class. The identifiability of the latent variables is ensured by the fact that the cluster structure reflects the underlying data distribution, modeled as a vMF distribution. This leads to a representation that captures the latent structure up to an orthogonal transformation. *Given the same underlying structure as in DIET, supervised learning can be viewed as a special case of instance discrimination, where the instance labels are replaced by class labels.* The cross-entropy objective, when applied to classification tasks and assuming our DGP from Assums. [1,](#page-3-2) similarly encourages representations to align with class vectors. As a result, the latent variables are recovered up to a linear transformation, providing a theoretical explanation for the success of supervised classification in learning linearly decodable representations.

### 3.4 THE GENEALOGY OF IDENTIFIABLE CLASSIFICATION WITH CROSS-ENTROPY

Our main result in Thm. [1,](#page-4-0) and its corollary for supervised classification (Thm. [2\)](#page-4-1) suggest the following surprising conclusion to invert the proposed [DGP](#page-30-3) (Assums. [1\)](#page-3-2):

*Solving an (almost) arbitrary classification task by optimizing the cross-entropy objective is sufficient to invert the [DGP](#page-30-3) and identify the ground-truth representation up to a linear transformation.* To show how solving a cross-entropy-based classification task is a key component to invert the [DGP](#page-30-3) and to achieve linear identifiability, we provide a unified treatment of auxiliary-variable ICA (i.e., weakly supervised or self-supervised classification) and supervised classification methods. We call this a *genealogy* to allude to the fact that these methods can be seen as special cases, descending from each other (cf. Fig. [2](#page-5-2) and Tab. [1](#page-5-1) for an overview, and Appx. [B](#page-23-1) for details).

Table 1: Comparison of the components of different cross-entropy-based classification methods: u denotes a (possibly) vector-valued auxiliary variable, t is the scalar Fam stands for exponential family, ⊥<sup>u</sup> for conditionally independent sources given the auxiliary variable, W is the classifier head, f the encoder, whereas N/A stands for no assumption Property GCL TCL InfoNCE DIET Supervised Latent space R <sup>d</sup> R <sup>d</sup> [S](#page-30-8) <sup>d</sup>−<sup>1</sup> R d /[S](#page-30-8) <sup>d</sup>−<sup>1</sup> R d Network W ◦ f W ◦ f f W ◦ f W ◦ f Aux.info u t i i c Conditional ⊥ <sup>u</sup> ExpFam [vMF](#page-30-1) [vMF](#page-30-1) [vMF](#page-30-1) Marginal N/A N/A uniform uniform uniform From [GCL](#page-30-5) to [TCL](#page-30-6) (Fig. [2a](#page-5-2): arbitrary scalar labels and exponential family latent variables). The most general framework we consider is [Generalized Contrastive Learning](#page-30-5) [\(GCL\)](#page-30-5) [\(Hyvarinen et al., 2019\)](#page-12-4), i.e., auxiliary-variable nonlinear [ICA.](#page-30-0) [GCL](#page-30-5) works with conditionally independent latent variables in Euclidean space given (possibly vector-valued) auxiliary information u. It aims to classify different values of u by distinguishing (x, u) from (x, u ∗ ), where u ∗ is an arbitrary value of the auxiliary variable. At the Bayes optimum of the cross-entropy loss, [GCL](#page-30-5) provides identifiability of the latent variables after the encoder f, but before the classifier head W, up to elementwise invertible transformations. When the latent variables are distributed

time step, i the sample index, and c the semantic class; Exp-

| Property     | GCL   | TCL    | InfoNCE | DIET    | Supervised |
|--------------|-------|--------|---------|---------|------------|
| Latent space | R     |        |         |         |            |
|              | d     | R      |         |         |            |
|              |       | d      | S       |         |            |
|              |       |        | d − 1   | R       |            |
|              |       |        |         | / S     |            |
|              |       |        |         | d − 1   | R          |
| Network      | W ◦ f | W ◦ f  | f       | W ◦ f   | W ◦ f      |
| Aux.info     | u     | t      | i       | i       | c          |
| Conditional  | ⊥     |        |         |         |            |
|              | u     | ExpFam | vMF     | vMF     | vMF        |
| Marginal     | N/A   | N/A    | uniform | uniform | uniform    |

Table 2: Identifiability results for [parametric instance discrimination \(PID\)](#page-30-4) in numerical simulations: Mean <sup>±</sup> standard deviation across 5 random seeds. Settings that match and violate our theoretical assumptions are denoted as ✓ and ✗, respectively. We report the [R](#page-30-9)<sup>2</sup> score for linear maps z˜ → z and w<sup>i</sup> → v<sup>c</sup> with normalized (subscript o) and not normalized (subscript a) w<sup>i</sup> . For normalized w<sup>i</sup> , we verify that the z˜ → z maps are orthogonal by reporting the [Mean Absolute Error](#page-30-10) [\(MAE\)](#page-30-10) between their singular values and those of an orthogonal transformation.

|    | N | d  | C    |         | p ( z | v | c )   | M.    |    | z ˜ | →   | R z | 2 ( ↑ o w | ) i | →   | v c |   | z ˜ | w → i | MAE z | o w | ( ↓ i | ) → | v c |    | z ˜ | →   | R z | 2 ( ↑ a w | ) i | → w | i v c |
|----|---|----|------|---------|-------|---|-------|-------|----|-----|-----|-----|-----------|-----|-----|-----|---|-----|-------|-------|-----|-------|-----|-----|----|-----|-----|-----|-----------|-----|-----|-------|
| 10 | 3 | 5  | 100  | vMF     | ( κ   | = | 10)   | ✓     | 98 | 6   | ± 0 | 01  | 99        | 9   | ± 0 | 00  | 0 | 01  | ± 0   | 00    | 0   | 00    | ± 0 | 00  | 99 | 0   | ± 0 | 00  | 99        | 9   | ± 0 | 00    |
| 10 | 5 | 5  | 100  | vMF     | ( κ   | = | 10)   | ✓     | 98 | 2   | ± 0 | 01  | 99        | 5   | ± 0 | 00  | 0 | 00  | ± 0   | 00    | 0   | 00    | ± 0 | 00  | 99 | 7   | ± 0 | 00  | 99        | 8   | ± 0 | 00    |
| 10 | 3 | 5  | 100  | vMF     | ( κ   | = | 10)   | ✓     | 98 | 6   | ± 0 | 01  | 99        | 9   | ± 0 | 00  | 0 | 01  | ± 0   | 00    | 0   | 00    | ± 0 | 00  | 99 | 0   | ± 0 | 00  | 99        | 9   | ± 0 | 00    |
| 10 | 3 | 10 | 100  | vMF     | ( κ   | = | 10)   | ✓     | 92 | 5   | ± 0 | 01  | 99        | 6   | ± 0 | 00  | 0 | 01  | ± 0   | 00    | 0   | 00    | ± 0 | 00  | 93 | 0   | ± 0 | 03  | 99        | 6   | ± 0 | 00    |
| 10 | 3 | 20 | 100  | vMF     | ( κ   | = | 10)   | ✓     | 70 | 8   | ± 0 | 02  | 97        | 1   | ± 0 | 01  | 0 | 03  | ± 0   | 00    | 0   | 00    | ± 0 | 00  | 81 | 9   | ± 0 | 01  | 99        | 7   | ± 0 | 00    |
| 10 | 3 | 5  | 10   | vMF     | ( κ   | = | 10)   | ✓     | 88 | 6   | ± 0 | 05  | 85        | 7   | ± 0 | 15  | 0 | 02  | ± 0   | 00    | 0   | 00    | ± 0 | 00  | 90 | 0   | ± 0 | 05  | 99        | 0   | ± 0 | 03    |
| 10 | 3 | 5  | 100  | vMF     | ( κ   | = | 10)   | ✓     | 98 | 6   | ± 0 | 01  | 99        | 9   | ± 0 | 01  | 0 | 01  | ± 0   | 00    | 0   | 00    | ± 0 | 00  | 99 | 0   | ± 0 | 00  | 99        | 9   | ± 0 | 00    |
| 10 | 3 | 5  | 1000 | vMF     | ( κ   | = | 10)   | ✓     | 99 | 3   | ± 0 | 00  | 99        | 9   | ± 0 | 00  | 0 | 00  | ± 0   | 00    | 0   | 00    | ± 0 | 00  | 99 | 2   | ± 0 | 00  | 99        | 9   | ± 0 | 00    |
| 10 | 3 | 5  | 100  | vMF     | ( κ   | = | 5)    | ✓     | 98 | 6   | ± 0 | 01  | 99        | 9   | ± 0 | 01  | 0 | 01  | ± 0   | 00    | 0   | 00    | ± 0 | 00  | 99 | 0   | ± 0 | 00  | 99        | 8   | ± 0 | 00    |
| 10 | 3 | 5  | 100  | vMF     | ( κ   | = | 10)   | ✓     | 99 | 0   | ± 0 | 00  | 99        | 9   | ± 0 | 00  | 0 | 00  | ± 0   | 00    | 0   | 00    | ± 0 | 00  | 99 | 1   | ± 0 | 00  | 99        | 9   | ± 0 | 00    |
| 10 | 3 | 5  | 100  | vMF     | ( κ   | = | 50)   | ✓     | 45 | 0   | ± 0 | 06  | 49        | 7   | ± 0 | 06  | 0 | 30  | ± 0   | 00    | 0   | 00    | ± 0 | 00  | 72 | 5   | ± 0 | 03  | 75        | 5   | ± 0 | 00    |
| 10 | 3 | 5  | 100  | vMF     | ( κ   | = | 10)   | ✓     | 98 | 6   | ± 0 | 01  | 99        | 9   | ± 0 | 01  | 0 | 01  | ± 0   | 00    | 0   | 00    | ± 0 | 00  | 99 | 0   | ± 0 | 00  | 99        | 9   | ± 0 | 00    |
| 10 | 3 | 5  | 100  | Laplace | (     | b | = 1 0 | ) ✗   | 85 | 2   | ± 0 | 01  | 99        | 7   | ± 0 | 01  | 0 | 01  | ± 0   | 00    | 0   | 00    | ± 0 | 00  | 85 | 4   | ± 0 | 00  | 99        | 5   | ± 0 | 00    |
| 10 | 3 | 5  | 100  | Normal  | ( σ   |   |       |       |    |     |     |     |           |     |     |     |   |     |       |       |     |       |     |     |    |     |     |     |           |     |     |       |
|    |   |    |      |         |       | 2 | = 1   | 0 ) ✗ | 98 | 7   | ± 0 | 00  | 99        | 8   | ± 0 | 00  | 0 | 01  | ± 0   | 00    | 0   | 00    | ± 0 | 00  | 98 | 6   | ± 0 | 00  | 99        | 6   | ± 0 | 00    |

according to an exponential family distribution and the auxiliary variable is a scalar (e.g., time), then we get the more specialized method, named [Time-Contrastive Learning \(TCL\)](#page-30-6) [\(Hyvarinen &](#page-12-3) [Morioka, 2016\)](#page-12-3). If the order of the exponential family is one, identifiability holds only up to a linear transformation, otherwise, up to elementwise invertible transformations.

From [TCL](#page-30-6) to DIET (Fig. [2b](#page-5-2): sample index as u and [vMF](#page-30-1) latent variables). Using our clustercentric [DGP](#page-30-3) (Assums. [1\)](#page-3-2), and assuming an even more special latent distribution (i.e., a [vMF\)](#page-30-1), we get the identifiability guarantee for DIET, i.e., our main result in Thm. [1.](#page-4-0) The auxiliary variable is a scalar for our result, too; however, instead of time, it is the (arbitrary) sample index.

From InfoNCE to DIET (Fig. [2c](#page-5-2): a compositional model W ◦ f and unit-normalized latent variables). Importantly, our main result also encompasses unit-normalized representations, the conventional choice in (identifiable) [SSL](#page-30-2) such as InfoNCE (cf. Appx. [B.3](#page-25-0) for details on InfoNCE) this is why we illustrate both InfoNCE and [TCL](#page-30-6) as being the "parents" of DIET in Fig. [2.](#page-5-2) Thus, Thm. [1](#page-4-0) is more general in terms of latent spaces than nonlinear [ICA,](#page-30-0) and it proves identifiability for the latent variables that are used post-training, as opposed to the proofs for InfoNCE in [\(Zimmermann](#page-14-1) [et al., 2021;](#page-14-1) [Rusak et al., 2024\)](#page-14-3), where practitioners discard the last few layers.

From DIET to supervised classification (Fig. [2d](#page-5-2): semantic class labels). When the labeling function assigns the semantic class labels, and not arbitrary indices, then our identifiability result still holds, yielding the case of supervised learning (Thm. [2\)](#page-4-1).

# 4 EMPIRICAL RESULTS

In § [4.1,](#page-7-0) we empirically verify the claims made in Thm. [1](#page-4-0) and Thm. [2](#page-4-1) in the synthetic setting. We generate data samples according to Assums. [1:](#page-3-2) ground-truth latent variables are sampled around cluster centroids v<sup>c</sup> following a [vMF](#page-30-1) distribution. Data augmentations, which share the same instance label i, are sampled from the same [vMF](#page-30-1) distribution around vc. In § [4.2,](#page-7-1) we describe our results on the DisLib disentanglement benchmark [\(Locatello et al., 2019\)](#page-13-6), and § [4.3](#page-9-0) includes our experiments on ImageNet-X [\(Idrissi et al., 2022\)](#page-12-10). We made our code publicly available on GitHub[<sup>1</sup>](#page-6-1) .

<sup>1</sup><https://github.com/klindtlab/csi>

#### 4.1 SYNTHETIC DATA

Setup. We consider N latent samples of dimensionality d generated from the conditional [vMF](#page-30-1) z ∼ p(z|vc), sampled around a set of |C | class vectors vc, which are uniformly distributed across the unit hyper-sphere [S](#page-30-8) d−1 . We use an invertible multi-layer perceptron (MLP) to map ground-truth latent variables to data samples. We train a classification head W =[w<sup>⊤</sup> i N <sup>i</sup>=1] and an MLP encoder that maps samples to representations z˜ ∈ R <sup>d</sup> using the DIET objective [\(1\)](#page-3-0). While to verify Thm. [1](#page-4-0) case C4., we do not normalize W, we do unit-normalize the weight vectors to validate Thm. [1](#page-4-0) case C3. We verify our theoretical claims by measuring the predictability of the ground-truth z from z˜ and v<sup>c</sup> from w<sup>i</sup> using the [R](#page-30-9)<sup>2</sup> score on a held-out dataset [\(Wright, 1921\)](#page-14-7). For identifiability up to orthogonal linear transformations, we train linear mappings with no intercept, assess the [R](#page-30-9)<sup>2</sup> score and verify that the singular values of this transformation converge to 1, while for identifiability up to affine linear transformations, we simply assess the [R](#page-30-9)<sup>2</sup> of a linear predictor with intercept.

d |C | p(z|vc) M. R<sup>2</sup> : z˜→z 5 100 vMF(κ= 10) ✓ 99.8<sup>±</sup>0.<sup>00</sup> 10 100 vMF(κ= 10) ✓ 97.2<sup>±</sup>0.<sup>01</sup> 20 100 vMF(κ= 10) ✓ 82.1<sup>±</sup>0.<sup>02</sup> 5 10 vMF(κ= 10) ✓ 97.5<sup>±</sup>0.<sup>03</sup> 5 100 vMF(κ= 10) ✓ 99.8<sup>±</sup>0.<sup>00</sup> 5 1000 vMF(κ= 10) ✓ 99.8<sup>±</sup>0.<sup>00</sup> 5 10000 vMF(κ= 10) ✓ 99.8<sup>±</sup>0.<sup>00</sup> 5 100 vMF(κ= 5) ✓ 99.7<sup>±</sup>0.<sup>00</sup> 5 100 vMF(κ= 10) ✓ 99.7<sup>±</sup>0.<sup>00</sup> 5 100 vMF(κ= 50) ✓ 65.5<sup>±</sup>0.<sup>09</sup> 5 100 vMF(κ= 10) ✓ 99.8<sup>±</sup>0.<sup>00</sup> 5 100 Laplace (b= 1.0) ✗ 85.4<sup>±</sup>0.<sup>01</sup> 5 100 Normal (σ <sup>2</sup> = 1.0) ✗ 99.6<sup>±</sup>0.<sup>00</sup> Table 3: Identifiability results for supervised learning in numerical simulations: Mean <sup>±</sup> standard deviation across 5 random seeds. Settings that match and violate our theoretical assumptions are denoted as ✓ and ✗, respectively. We report the [R](#page-30-9)<sup>2</sup> score for linear mappings z˜ → z, and not normalized w<sup>i</sup> . We used N = 10<sup>3</sup> samples Results for DIET. In Tab. [2,](#page-6-2) we report the [R](#page-30-9)<sup>2</sup> scores for the recovery of the cluster vectors v<sup>c</sup> from W's rows and of the ground-truth latent variables z from the learned latent variables z˜. For DIET's [PID](#page-30-4) task, we also consider cases with row-normalized W. We observe scores close to 100% (≥ 98%), even with many clusters (≥ 10<sup>3</sup> ) and samples (∼ 10<sup>5</sup> ). High latent dimensionality (> 10) does impact the recovery of ground-truth latent variables—such scalability problems are a common artifact in [SSL](#page-30-2) [\(Zim](#page-14-1)[mermann et al., 2021;](#page-14-1) [Rusak et al., 2024\)](#page-14-3). For a higher concentration of samples around v<sup>c</sup> (i.e., κ= 50) as well as a lower number of clusters (i.e., |C | = 10), the [R](#page-30-9)<sup>2</sup> score decreases, which is also a common phenomenon, and is possibly explained by too strong augmentation overlap [\(Wang et al., 2022;](#page-14-6) [Rusak et al., 2024\)](#page-14-3). For a low number of clusters, high κ and a fixed number of training samples, the concentration of samples in regions surrounding centroids, vc, increases, a setting, refered to as "overly overlapping augmentations", known to be suboptimal and leading to a drop in downstream performance [\(Wang et al., 2022\)](#page-14-6). Our results also suggest that even under model misspecification (last two rows in Tab. [2](#page-6-2) with non[-vMF](#page-30-1) distributions), identifiability still holds. For unit-normalized Wrows, the [MAE](#page-30-10) is lower, confirming the orthogonality of the map wi→vc. We additionally ablate over batch size, concentration, and conditional in Appx. [D.](#page-27-0)

| d 5 10 20 5 5 5 5 5 5 5 5 | C   100 100 100 10 100 1000 10000 100 100 100 100 | p ( z   v c ) vMF ( κ = 10) vMF ( κ = 10) vMF ( κ = 10) vMF ( κ = 10) vMF ( κ = 10) vMF ( κ = 10) vMF ( κ = 10) vMF ( κ = 5) vMF ( κ = 10) vMF ( κ = 50) vMF ( κ = 10) | M. ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ R 2 : z ˜ → z 99 8 ± 0 00 97 2 ± 0 01 82 1 ± 0 02 97 5 ± 0 03 99 8 ± 0 00 99 8 ± 0 00 99 8 ± 0 00 99 7 ± 0 00 99 7 ± 0 00 65 5 ± 0 09 99 8 ± 0 00 |
|---------------------------|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 5                         | 100                                               | Laplace ( b = 1 0 )                                                                                                                                                    | ✗ 85 4 ± 0 01                                                                                                                                                              |
| 5                         | 100                                               | Normal ( σ                                                                                                                                                             |                                                                                                                                                                            |
|                           |                                                   | 2 = 1 0 )                                                                                                                                                              | ✗ 99 6 ± 0 00                                                                                                                                                              |

Results for Supervised Classification. In Tab. [3,](#page-7-2) where the semantic class labels were used instead of the sample index, we only report the [R](#page-30-9)<sup>2</sup> score for the recovery of the ground-truth latent variables z from the learned latent variables z˜. In all but one setting, we observe higher [R](#page-30-9)<sup>2</sup> from representations learned with class labels rather than instance indices. This suggests that even a coarser classification task may suffice to learn linearly identifiable representations of the underlying latent variables.

#### 4.2 DISLIB

Setup. Next, we evaluate our methods on the DisLib disentanglement benchmark [\(Locatello et al.,](#page-13-6) [2019\)](#page-13-6), which provides a controlled setting for testing disentanglement and latent variable recovery. It includes the vision datasets dSprites, Shapes 3D, MPI 3D, Cars 3D, and smallNORB. We train both a three-layer MLP with 512 latent dimensions and BatchNorm (which helped with trainability) and a CNN (ResNet18) also with 512 latent dimensions. We only consider latent variables with Euclidean topology, as non-Euclidean, e.g., periodic latent variables such as orientation, are problematic to learn and are potentially mapped to a nonlinear manifold [\(Higgins et al., 2018;](#page-11-7) [Pfau et al., 2020;](#page-13-8) [Keurti et al., 2023;](#page-12-11) [Engels et al., 2024\)](#page-11-6). We evaluate the recovery of latent variables by computing the Pearson correlation between ground-truth and predicted factors. We detail our setup in Appx. [C.2.](#page-26-0)

Table 4: Identifiability in DisLib datasets [\(Locatello et al., 2019\)](#page-13-6): We train different models to predict the categorical variable in each setting: (x): as a baseline, from the inputs; (f MLP(x)): from a three-layer MLP; and (f CNN(x)): from a CNN (ResNet18). All continuous latent variables can be decoded from the learned representations, corroborated by the Pearson correlation—reported with mean <sup>±</sup> standard deviation across 3 random seeds. Including the category is informative to see how well the underlying training classification task was solved.

|           | Model   | Latent     | x |    |     |    | f |    | MLP ( | x )  | f | CNN |     | ( x | )  |
|-----------|---------|------------|---|----|-----|----|---|----|-------|------|---|-----|-----|-----|----|
| dSprites  |         | category   | 0 | 26 | ± 0 | 00 | 0 | 94 | ± 0   | 01   | 1 | 00  | ±   | 0   | 00 |
| dSprites  |         | scale      | 0 | 62 | ± 0 | 00 | 0 | 98 | ±     | 0 00 | 0 | 92  | ± 0 | 05  |    |
| dSprites  |         | posX       | 0 | 92 | ± 0 | 00 | 0 | 97 | ± 0   | 00   | 0 | 99  | ±   | 0   | 00 |
| dSprites  |         | posY       | 0 | 92 | ± 0 | 00 | 0 | 97 | ± 0   | 00   | 0 | 99  | ±   | 0   | 00 |
| Shapes    | 3D      | category   | 0 | 42 | ± 0 | 00 | 1 | 00 | ± 0   | 00   | 1 | 00  | ±   | 0   | 00 |
| Shapes    | 3D      | objSize    | 0 | 21 | ± 0 | 00 | 0 | 89 | ± 0   | 01   | 0 | 99  | ±   | 0   | 00 |
| Shapes    | 3D      | objAzimuth | 0 | 04 | ± 0 | 00 | 0 | 85 | ± 0   | 02   | 0 | 93  | ±   | 0   | 01 |
| MPI       | 3D      | category   | 0 | 03 | ± 0 | 00 | 0 | 71 | ± 0   | 01   | 0 | 97  | ±   | 0   | 00 |
| MPI       | 3D      | posX       | 0 | 28 | ± 0 | 00 | 0 | 76 | ± 0   | 01   | 0 | 90  | ±   | 0   | 01 |
| MPI       | 3D      | posY       | 0 | 46 | ± 0 | 00 | 0 | 76 | ± 0   | 01   | 0 | 84  | ±   | 0   | 01 |
| MPI       | 3D real | category   | 0 | 19 | ± 0 | 00 | 0 | 88 | ± 0   | 01   | 0 | 98  | ±   | 0   | 00 |
| MPI       | 3D real | posX       | 0 | 14 | ± 0 | 00 | 0 | 74 | ± 0   | 01   | 0 | 83  | ±   | 0   | 01 |
| MPI       | 3D real | posY       | 0 | 44 | ± 0 | 00 | 0 | 54 | ± 0   | 01   | 0 | 71  | ±   | 0   | 02 |
| Cars      | 3D      | category   | 0 | 05 | ± 0 | 00 | 0 | 63 | ± 0   | 11   | 0 | 77  | ±   | 0   | 02 |
| Cars      | 3D      | elevation  | 0 | 15 | ± 0 | 00 | 0 | 87 | ±     | 0 03 | 0 | 78  | ± 0 | 02  |    |
| smallNORB |         | category   | 0 | 22 | ± 0 | 00 | 0 | 94 | ± 0   | 01   | 1 | 00  | ±   | 0   | 00 |
| smallNORB |         | elevation  | 0 | 15 | ± 0 | 00 | 0 | 83 | ±     | 0 01 | 0 | 79  | ± 0 | 01  |    |

![](_page_8_Figure_3.jpeg)

Figure 3: Approximate identifiability on ImageNet-X against a random (shuffled) baseline: Using ImageNet-X [\(Idrissi et al., 2022\)](#page-12-10), we test how well linear decoders are able to predict each latent from the second-to-last layer of different models, i.e., when the classification head is discarded. We train a linear classifier on the features, and plot the accuracy of predicting different latent variables. As baselines, we also try decoding from the raw input and from the randomly initialized model representations. Error-bars indicate standard error of the mean (SEM) across 10 seeds of balanced resampling. Asterisks indicate significant p-values (against a null hypothesis of 0.5 chance level accuracy) at an κ = 0.05/85 multiple comparison (Bonferroni) adjusted significance level.

Results. The models trained using cross-entropy were able to recover latent variables such as object position, scale, and orientation with high accuracy. As shown in Tab. [4,](#page-8-0) the Pearson correlation is generally highest when predicting the latent variables from the CNN's representation, which we attribute to the CNN's suitable inductive bias for images. In few cases, such as the position in dSprites, this can be done with fairly high accuracy even on the input data. Nevertheless, in all settings the nonlinear function estimated by the model is necessary to linearly identify the correct latent variables.

### 4.3 REAL DATA: IMAGENET-X

Setup. Finally, we test the generalizability of our theoretical insights on real-world data using ImageNet-X [\(Idrissi et al., 2022\)](#page-12-10). The latent variables are binary proxies, defined by human annotators [\(Idrissi et al., 2022\)](#page-12-10). We evaluate how well linear decoders can predict latent variables from pretrained model representations. We use two architectures, a ResNet50 and a Vit-b-16 both trained on standard supervised classification using a cross-entropy loss on the full ImageNet dataset [\(Deng et al., 2009\)](#page-10-11). As baselines, we also decode from the inputs and the randomly initialized models. After balanced sub-sampling, over 10 random seeds, we report accuracies. We use t-tests against a chance level of 50% with a Bonferroni adjusted significance level of κ = 0.05 17·5 . Detail are in Appx. [C.3.](#page-26-1)

Results. Fig. [3](#page-8-1) shows that even in complex, high-dimensional data, latents can be linearly decoded from representations learned via supervised learning, in most cases significantly above chance level. Some factors (e.g., *darker* and *brighter*) are linearly decodable even from untrained models or input space. Unsurprisingly, decoding *class* (binarized ImageNet labels, every index < 500 is set to 0 and every index ≥500 is set to 1) works well for the trained models. ResNet50 has slightly higher decoding performance, possibly due to the larger latent space (d= 2048, compared to d= 768 in ViT). While texture information may be expected [\(Geirhos et al., 2018\)](#page-11-11), the presence of shape information suggests that shortcut learning may be mitigated even after standard training [\(Geirhos et al., 2020\)](#page-11-12).

# 5 DISCUSSION

Limitations. One limitation of our work is that we mainly focus on synthetic and controlled datasets. While the results on ImageNet-X [\(Idrissi et al., 2022\)](#page-12-10) are promising, they only provide some supporting evidence for our theory on real data. The factors in ImageNet-X are likely not the true latent variables of the data generating process, still, the linear identifiablity results on these proxy latent variables support our theoretical results. Further experiments on other large-scale datasets would support the generality of our findings. However, this would require the availability of such datasets with full latent variable annotations. Although our cluster-centric modeling of the [data](#page-30-3) [generating process](#page-30-3) allows capturing the inherent structure of the data, our assumption about the latent variables' geometric properties (such as being drawn from a vMF distribution on a hypersphere), may not hold in all real-world settings. For instance, the pose of an object in a scene is, arguably, an independent component/subspace corresponding to a point on SO(3), which has a distinct topology from our assumed latent variables on a hypersphere. Moreover, the assumption that a data sample and its augmented version are conditionally independent given their semantic class could be relaxed in future work, since it may be misaligned with realistic scenarios [\(Wang et al., 2022\)](#page-14-6). Despite these simplifications, our experimental results also suggest that our assumptions can be relaxed, as linear identifiability seems to hold even when some of the assumptions are violated (cf. Tab. [5\)](#page-27-1). In Appx. [D,](#page-27-0) we demonstrate the remarkable robustness of latent identifiability (Fig. [6\)](#page-30-11), the interaction between batch size, latent dimensionality, concentration, and latent conditional.

Implications for Deep Learning. Our results indicate that deep learning models trained using cross-entropy and assuming a certain DGP recover the underlying latent variables up to linear transformations. As our identifiability proof for [parametric instance discrimination](#page-30-4) illustrates with DIET, this statement also holds when the classification task is standard supervised learning. Our analysis on the key role of cross-entropy-based classification provides a theoretical foundation for phenomena such as neural analogy-making, transfer learning, and linear decoding of features.

Conclusion. We extend the identifiability results of the auxiliary-variable nonlinear [Independent](#page-30-0) [Component Analysis \(ICA\)](#page-30-0) literature to [parametric instance discrimination](#page-30-4) with a cluster-centric [data generating process.](#page-30-3) Our modeling choice can capture the clustered structure of the data, accommodates non-normalized (as in [ICA\)](#page-30-0) and unit-normalized (as in InfoNCE) representations (Thm. [1\)](#page-4-0). Furthermore, our identifiability result holds for the latent representation used post-training, i.e., for the latent variables before the classification head. Our results offer new insights into the success of deep learning, particularly in supervised classification tasks, which we show is a special case of the DIET [parametric instance discrimination](#page-30-4) algorithm, where the instance labels equal the semantic class labels (Thm. [2\)](#page-4-1). By linking self-supervised learning—via nonlinear ICA and DIET—to supervised classification for a specific DGP, we provide a theoretical framework that explains why simple classification tasks recover interpretable and transferable representations.

Future Work. Future research could extend these insights to connections between nonlinear ICA and other forms of supervised learning and testing the scalability of our theoretical results to larger models and datasets. To assess our theory's predictions beyond proxy labels [\(Idrissi et al., 2022\)](#page-12-10), we need real world image datasets with full specification of the latent variables, e.g., in rendered scenes.

### ACKNOWLEDGMENTS

The authors thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting Patrik Reizinger and Attila Juhos. Patrik Reizinger acknowledges his membership in the European Laboratory for Learning and Intelligent Systems (ELLIS) PhD program. This work was supported by the German Federal Ministry of Education and Research (BMBF): Tubingen AI Center, ¨ FKZ: 01IS18039A. Wieland Brendel acknowledges financial support via an Emmy Noether Grant funded by the German Research Foundation (DFG) under grant no. BR 6382/1-1 and via the Open Philantropy Foundation funded by the Good Ventures Foundation. Wieland Brendel is a member of the Machine Learning Cluster of Excellence, EXC number 2064/1 – Project number 390727645. This research utilized compute resources at the Tubingen Machine Learning Cloud, DFG FKZ INST ¨ 37/1057-1 FUGG. Alice Bizeul's work is supported by an ETH AI Center Doctoral fellowship.

# REFERENCES


[1] Kartik Ahuja, Divyat Mahajan, Vasilis Syrgkanis, and Ioannis Mitliagkas. Towards efficient representation identification in supervised learning. In *Proceedings of the First Conference on Causal Learning and Reasoning*, pp. 19–43. PMLR, June 2022. URL [https://proceedings.mlr.](https://proceedings.mlr.press/v177/ahuja22a.html) [press/v177/ahuja22a.html](https://proceedings.mlr.press/v177/ahuja22a.html). ISSN: 2640-3498. [2](#page-1-1) Carl Allen and Timothy Hospedales. Analogies explained: Towards understanding word embeddings. In *International Conference on Machine Learning*, pp. 223–231. PMLR, 2019. [3](#page-2-0) Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski. A Latent Variable Model Approach to PMI-based Word Embeddings. *Transactions of the Association for Computational Linguistics*, 4:385–399, July 2016. ISSN 2307-387X. doi: 10.1162/tacl a 00106. URL [https:](https://doi.org/10.1162/tacl_a_00106) [//doi.org/10.1162/tacl\\_a\\_00106](https://doi.org/10.1162/tacl_a_00106). [1,](#page-0-0) [3](#page-2-0) Randall Balestriero and Yann LeCun. Contrastive and Non-Contrastive Self-Supervised Learning Recover Global and Local Spectral Embedding Methods, June 2022. URL [http://arxiv.](http://arxiv.org/abs/2205.11508) [org/abs/2205.11508](http://arxiv.org/abs/2205.11508). arXiv:2205.11508 [cs, math, stat]. [4](#page-3-3) Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. *IEEE transactions on pattern analysis and machine intelligence*, 35(8):1798–1828, 2013. [1](#page-0-0) Florian Bordes, Randall Balestriero, Quentin Garrido, Adrien Bardes, and Pascal Vincent. Guillotine Regularization: Why removing layers is needed to improve generalization in Self-Supervised Learning. *Transactions on Machine Learning Research*, May 2023. ISSN 2835-8856. URL <https://openreview.net/forum?id=ZgXfXSz51n&s=09>. [4](#page-3-3) Diane Bouchacourt, Mark Ibrahim, and Stephane Deny. Addressing the topological defects of ´ disentanglement via distributed operators. *arXiv preprint arXiv:2102.05623*, 2021. [3](#page-2-0) Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A Simple Framework for Contrastive Learning of Visual Representations. *arXiv:2002.05709 [cs, stat]*, June 2020. URL <http://arxiv.org/abs/2002.05709>. arXiv: 2002.05709. [3,](#page-2-0) [5](#page-4-5) Zirui Chen and Michael Bonner. Canonical dimensions of neural visual representation. *Journal of Vision*, 23(9):4937–4937, 2023. [3](#page-2-0) Pierre Comon. Independent component analysis, a new concept? *Signal processing*, 36(3):287–314, 1994. [1,](#page-0-0) [3](#page-2-0) Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE conference on computer vision and pattern recognition*, pp. 248–255. Ieee, 2009. [10,](#page-9-1) [28](#page-27-2) Jeff Donahue, Yangqing Jia, Oriol Vinyals, Judy Hoffman, Ning Zhang, Eric Tzeng, and Trevor Darrell. Decaf: A deep convolutional activation feature for generic visual recognition. In *International conference on machine learning*, pp. 647–655. PMLR, 2014. [1](#page-0-0)

[2] Alexey Dosovitskiy, Jost Tobias Springenberg, Martin Riedmiller, and Thomas Brox. Discriminative unsupervised feature learning with convolutional neural networks. *Advances in neural information processing systems*, 27, 2014. [2,](#page-1-1) [4](#page-3-3) Aleksandr Drozd, Anna Gladkova, and Satoshi Matsuoka. Word embeddings, analogies, and machine learning: Beyond king-man+ woman= queen. In *Proceedings of coling 2016, the 26th international conference on computational linguistics: Technical papers*, pp. 3519–3530, 2016. [3](#page-2-0) Kang Du and Yu Xiang. Causal Inference from Slowly Varying Nonstationary Processes. *arXiv:2012.13025 [cs, math, stat]*, September 2021. URL [http://arxiv.org/abs/2012.](http://arxiv.org/abs/2012.13025) [13025](http://arxiv.org/abs/2012.13025). arXiv: 2012.13025. [3](#page-2-0) Philipp Dufter and Hinrich Schutze. Analytical methods for interpretable ultradense word embeddings. ¨ *arXiv preprint arXiv:1904.08654*, 2019. [3](#page-2-0) Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy Models of Superposition, September 2022. URL <http://arxiv.org/abs/2209.10652>. arXiv:2209.10652 [cs]. [1](#page-0-0) Joshua Engels, Isaac Liao, Eric J Michaud, Wes Gurnee, and Max Tegmark. Not all language model features are linear. *arXiv preprint arXiv:2405.14860*, 2024. [3,](#page-2-0) [8,](#page-7-3) [27](#page-26-2) Benoˆıt Frenay and Michel Verleysen. Classification in the presence of label noise: a survey. ´ *IEEE transactions on neural networks and learning systems*, 25(5):845–869, 2013. [30](#page-29-0) Marco Fumero, Florian Wenzel, Luca Zancato, Alessandro Achille, Emanuele Rodola, Stefano Soatto, ` Bernhard Scholkopf, and Francesco Locatello. Leveraging sparse and shared feature activations ¨ for disentangled representation learning, April 2023. URL [http://arxiv.org/abs/2304.](http://arxiv.org/abs/2304.07939) [07939](http://arxiv.org/abs/2304.07939). arXiv:2304.07939 [cs]. [2](#page-1-1) Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A Wichmann, and Wieland Brendel. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. *arXiv preprint arXiv:1811.12231*, 2018. [10](#page-9-1) Robert Geirhos, Jorn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias ¨ Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11):665–673, November 2020. ISSN 2522-5839. doi: 10.1038/s42256-020-00257-z. URL <https://www.nature.com/articles/s42256-020-00257-z>. Number: 11 Publisher: Nature Publishing Group. [10](#page-9-1) Mikhail Genkin and Tatiana A Engel. Moving beyond generalization to accurate interpretation of flexible models. *Nature machine intelligence*, 2(11):674–683, 2020. [3](#page-2-0) Luigi Gresele, Paul K. Rubenstein, Arash Mehrjou, Francesco Locatello, and Bernhard Scholkopf. ¨ The Incomplete Rosetta Stone Problem: Identifiability Results for Multi-View Nonlinear ICA. *arXiv:1905.06642 [cs, stat]*, August 2019. URL <http://arxiv.org/abs/1905.06642>. arXiv: 1905.06642. [1,](#page-0-0) [3](#page-2-0) Jeff Z. HaoChen, Colin Wei, Ananya Kumar, and Tengyu Ma. Beyond Separability: Analyzing the Linear Transferability of Contrastive Representations to Related Subpopulations, May 2022. URL <http://arxiv.org/abs/2204.02683>. arXiv:2204.02683 [cs]. [4](#page-3-3) Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016. [27](#page-26-2) Irina Higgins, David Amos, David Pfau, Sebastien Racaniere, Loic Matthey, Danilo Rezende, and Alexander Lerchner. Towards a Definition of Disentangled Representations. *arXiv:1812.02230 [cs, stat]*, December 2018. URL <http://arxiv.org/abs/1812.02230>. arXiv: 1812.02230. [3,](#page-2-0) [8,](#page-7-3) [27](#page-26-2)

[3] Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. The platonic representation hypothesis. *arXiv preprint arXiv:2405.07987*, 2024. [3](#page-2-0) Aapo Hyvarinen and Hiroshi Morioka. Unsupervised Feature Extraction by Time-Contrastive Learning and Nonlinear ICA. *arXiv:1605.06336 [cs, stat]*, May 2016. URL [http://arxiv.](http://arxiv.org/abs/1605.06336) [org/abs/1605.06336](http://arxiv.org/abs/1605.06336). arXiv: 1605.06336. [1,](#page-0-0) [2,](#page-1-1) [3,](#page-2-0) [4,](#page-3-3) [6,](#page-5-3) [7](#page-6-3) Aapo Hyvarinen, Juha Karhunen, and Erkki Oja. *Independent component analysis*. J. Wiley, New York, 2001. ISBN 978-0-471-40540-5. [1,](#page-0-0) [3](#page-2-0) Aapo Hyvarinen, Hiroaki Sasaki, and Richard E. Turner. Nonlinear ICA Using Auxiliary Variables and Generalized Contrastive Learning. *arXiv:1805.08651 [cs, stat]*, February 2019. URL [http:](http://arxiv.org/abs/1805.08651) [//arxiv.org/abs/1805.08651](http://arxiv.org/abs/1805.08651). arXiv: 1805.08651. [1,](#page-0-0) [2,](#page-1-1) [3,](#page-2-0) [4,](#page-3-3) [6,](#page-5-3) [24,](#page-23-2) [25,](#page-24-0) [26](#page-25-1) Aapo Hyvarinen, Ilyes Khemakhem, and Ricardo Monti. Identifiability of latent-variable and ¨ structural-equation models: from linear to nonlinear, February 2023. URL [http://arxiv.](http://arxiv.org/abs/2302.02672) [org/abs/2302.02672](http://arxiv.org/abs/2302.02672). arXiv:2302.02672 [cs, stat]. [1,](#page-0-0) [3](#page-2-0) Hermanni Halv ¨ a, Sylvain Le Corff, Luc Leh ¨ ericy, Jonathan So, Yongjie Zhu, Elisabeth Gassiat, and ´ Aapo Hyvarinen. Disentangling Identifiable Features from Noisy Data with Structured Nonlinear ICA. *arXiv:2106.09620 [cs, stat]*, June 2021. URL <http://arxiv.org/abs/2106.09620>. arXiv: 2106.09620. [2,](#page-1-1) [3](#page-2-0) Mark Ibrahim, David Klindt, and Randall Balestriero. Occam's Razor for Self Supervised Learning: What is Sufficient to Learn Good Representations?, June 2024. URL [http://arxiv.org/](http://arxiv.org/abs/2406.10743) [abs/2406.10743](http://arxiv.org/abs/2406.10743). arXiv:2406.10743 [cs]. [2,](#page-1-1) [3,](#page-2-0) [4,](#page-3-3) [5](#page-4-5) Badr Youbi Idrissi, Diane Bouchacourt, Randall Balestriero, Ivan Evtimov, Caner Hazirbas, Nicolas Ballas, Pascal Vincent, Michal Drozdzal, David Lopez-Paz, and Mark Ibrahim. Imagenet-x: Understanding model mistakes with factor of variation annotations, 2022. URL [https://](https://arxiv.org/abs/2211.01866) [arxiv.org/abs/2211.01866](https://arxiv.org/abs/2211.01866). [2,](#page-1-1) [7,](#page-6-3) [9,](#page-8-2) [10,](#page-9-1) [27](#page-26-2) Li Jing, Pascal Vincent, Yann LeCun, and Yuandong Tian. Understanding Dimensional Collapse in Contrastive Self-supervised Learning, April 2022. URL [http://arxiv.org/abs/2110.](http://arxiv.org/abs/2110.09348) [09348](http://arxiv.org/abs/2110.09348). Number: arXiv:2110.09348 arXiv:2110.09348 [cs]. [29](#page-28-0) Hamza Keurti, Patrik Reizinger, Bernhard Scholkopf, and Wieland Brendel. Desiderata for Represen- ¨ tation Learning from Identifiability, Disentanglement, and Group-Structuredness. June 2023. URL <https://openreview.net/forum?id=r6C86JjuiW>. [3,](#page-2-0) [8,](#page-7-3) [27](#page-26-2) Ilyes Khemakhem, Diederik Kingma, Ricardo Monti, and Aapo Hyvarinen. Variational Autoencoders and Nonlinear ICA: A Unifying Framework. In *International Conference on Artificial Intelligence and Statistics*, pp. 2207–2217. PMLR, June 2020a. URL [http://proceedings.](http://proceedings.mlr.press/v108/khemakhem20a.html) [mlr.press/v108/khemakhem20a.html](http://proceedings.mlr.press/v108/khemakhem20a.html). ISSN: 2640-3498. [1,](#page-0-0) [3](#page-2-0) Ilyes Khemakhem, Ricardo Pio Monti, Diederik P. Kingma, and Aapo Hyvarinen. ICE-BeeM: Identi- ¨ fiable Conditional Energy-Based Deep Models Based on Nonlinear ICA. *arXiv:2002.11537 [cs, stat]*, October 2020b. URL <http://arxiv.org/abs/2002.11537>. arXiv: 2002.11537. [1,](#page-0-0) [3](#page-2-0) Michael Kirchhof, Karsten Roth, Zeynep Akata, and Enkelejda Kasneci. A Non-isotropic Probabilistic Take on Proxy-based Deep Metric Learning, July 2022. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2207.03784) [2207.03784](http://arxiv.org/abs/2207.03784). arXiv:2207.03784 [cs, stat]. [4](#page-3-3) David Klindt, Lukas Schott, Yash Sharma, Ivan Ustyuzhaninov, Wieland Brendel, Matthias Bethge, and Dylan Paiton. Towards Nonlinear Disentanglement in Natural Data with Temporal Sparse Coding. *arXiv:2007.10930 [cs, stat]*, March 2021. URL [http://arxiv.org/abs/2007.](http://arxiv.org/abs/2007.10930) [10930](http://arxiv.org/abs/2007.10930). arXiv: 2007.10930. [1,](#page-0-0) [3](#page-2-0) Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. *Advances in neural information processing systems*, 25, 2012. [1](#page-0-0)

[4] Sebastien Lachapelle, Tristan Deleu, Divyat Mahajan, Ioannis Mitliagkas, Yoshua Bengio, Simon Lacoste-Julien, and Quentin Bertrand. Synergies between Disentanglement and Sparsity: Generalization and Identifiability in Multi-Task Learning. In *Proceedings of the 40th International Conference on Machine Learning*, pp. 18171–18206. PMLR, July 2023. URL [https:](https://proceedings.mlr.press/v202/lachapelle23a.html) [//proceedings.mlr.press/v202/lachapelle23a.html](https://proceedings.mlr.press/v202/lachapelle23a.html). ISSN: 2640-3498. [2](#page-1-1) Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Scholkopf, ¨ and Olivier Bachem. Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations. In *International Conference on Machine Learning*, pp. 4114–4124. PMLR, May 2019. URL [http://proceedings.mlr.press/v97/locatello19a.](http://proceedings.mlr.press/v97/locatello19a.html) [html](http://proceedings.mlr.press/v97/locatello19a.html). ISSN: 2640-3498. [2,](#page-1-1) [7,](#page-6-3) [8,](#page-7-3) [9,](#page-8-2) [27](#page-26-2) Francesco Locatello, Ben Poole, Gunnar Ratsch, Bernhard Sch ¨ olkopf, Olivier Bachem, and Michael ¨ Tschannen. Weakly-Supervised Disentanglement Without Compromises. *arXiv:2002.02886 [cs, stat]*, October 2020. URL <http://arxiv.org/abs/2002.02886>. arXiv: 2002.02886. [1,](#page-0-0) [3](#page-2-0) Tomas Mikolov. Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*, 2013. [3](#page-2-0) Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed Representations of Words and Phrases and their Compositionality. In *Advances in Neural Information Processing Systems*, volume 26. Curran Associates, Inc., 2013. URL [https://papers.nips.cc/paper\\_files/paper/2013/hash/](https://papers.nips.cc/paper_files/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html) [9aa42b31882ec039965f3c4923ce901b-Abstract.html](https://papers.nips.cc/paper_files/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html). [1,](#page-0-0) [3](#page-2-0) Hiroshi Morioka and Aapo Hyvarinen. Connectivity-contrastive learning: Combining causal discovery and representation learning for multimodal data. In *Proceedings of The 26th International Conference on Artificial Intelligence and Statistics*, pp. 3399–3426. PMLR, April 2023. URL <https://proceedings.mlr.press/v206/morioka23a.html>. ISSN: 2640-3498. [2,](#page-1-1) [3](#page-2-0) Hiroshi Morioka, Hermanni Halv ¨ a, and Aapo Hyv ¨ arinen. Independent Innovation Analysis for ¨ Nonlinear Vector Autoregressive Process. *arXiv:2006.10944 [cs, stat]*, February 2021. URL <https://arxiv.org/abs/2006.10944>. arXiv: 2006.10944. [2,](#page-1-1) [3](#page-2-0) Luca Moschella, Valentino Maiorca, Marco Fumero, Antonio Norelli, Francesco Locatello, and Emanuele Rodola. Relative representations enable zero-shot latent space communication, March ` 2023. URL <http://arxiv.org/abs/2209.15430>. arXiv:2209.15430 [cs]. [3](#page-2-0) David F Nettleton, Albert Orriols-Puig, and Albert Fornells. A study of the effect of different types of noise on the precision of supervised learning techniques. *Artificial intelligence review*, 33:275–306, 2010. [30](#page-29-0) Kiho Park, Yo Joong Choe, and Victor Veitch. The Linear Representation Hypothesis and the Geometry of Large Language Models, November 2023. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2311.03658) [2311.03658](http://arxiv.org/abs/2311.03658). arXiv:2311.03658 [cs, stat]. [1,](#page-0-0) [2,](#page-1-1) [3](#page-2-0) David Pfau, Irina Higgins, Alex Botev, and Sebastien Racani ´ ere. Disentangling by Subspace Diffusion. ` In *Advances in Neural Information Processing Systems*, volume 33, pp. 17403–17415. Curran Associates, Inc., 2020. URL [https://proceedings.neurips.cc/paper/2020/hash/](https://proceedings.neurips.cc/paper/2020/hash/c9f029a6a1b20a8408f372351b321dd8-Abstract.html) [c9f029a6a1b20a8408f372351b321dd8-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/c9f029a6a1b20a8408f372351b321dd8-Abstract.html). [3,](#page-2-0) [8,](#page-7-3) [27](#page-26-2) Geoffrey Roeder, Luke Metz, and Diederik P. Kingma. On Linear Identifiability of Learned Representations. *arXiv:2007.00810 [cs, stat]*, July 2020. URL [http://arxiv.org/abs/2007.](http://arxiv.org/abs/2007.00810) [00810](http://arxiv.org/abs/2007.00810). arXiv: 2007.00810. [3](#page-2-0) D Rolnick. Deep learning is robust to massive label noise. *arXiv preprint arXiv:1705.10694*, 2017. [31](#page-30-12) Karsten Roth, Mark Ibrahim, Zeynep Akata, Pascal Vincent, and Diane Bouchacourt. Disentanglement of correlated factors via hausdorff factorized support. *arXiv preprint arXiv:2210.07347*, 2022. [27](#page-26-2)

[5] Evgenia Rusak, Patrik Reizinger, Attila Juhos, Oliver Bringmann, Roland S. Zimmermann, and Wieland Brendel. InfoNCE: Identifying the Gap Between Theory and Practice, June 2024. URL <http://arxiv.org/abs/2407.00143>. arXiv:2407.00143 [cs, stat]. [2,](#page-1-1) [3,](#page-2-0) [4,](#page-3-3) [7,](#page-6-3) [8,](#page-7-3) [29](#page-28-0) James B. Simon, Maksis Knutins, Liu Ziyin, Daniel Geisz, Abraham J. Fetterman, and Joshua Albrecht. On the Stepwise Nature of Self-Supervised Learning, May 2023. URL [http://](http://arxiv.org/abs/2303.15438) [arxiv.org/abs/2303.15438](http://arxiv.org/abs/2303.15438). arXiv:2303.15438 [cs]. [29](#page-28-0) Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, and et al. Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet, 2024. URL [https:](https://transformer-circuits.pub/2024/scaling-monosemanticity) [//transformer-circuits.pub/2024/scaling-monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity). [1](#page-0-0) Julius von Kugelgen, Yash Sharma, Luigi Gresele, Wieland Brendel, Bernhard Sch ¨ olkopf, Michel ¨ Besserve, and Francesco Locatello. Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style, June 2021. URL <http://arxiv.org/abs/2106.04619>. arXiv: 2106.04619. [2,](#page-1-1) [3,](#page-2-0) [4,](#page-3-3) [29](#page-28-0) Yifei Wang, Qi Zhang, Yisen Wang, Jiansheng Yang, and Zhouchen Lin. Chaos is a Ladder: A New Theoretical Understanding of Contrastive Learning via Augmentation Overlap, May 2022. URL <http://arxiv.org/abs/2203.13457>. arXiv:2203.13457 [cs, stat]. [4,](#page-3-3) [8,](#page-7-3) [10](#page-9-1) Wikipedia. Gibbs' inequality, 2024a. URL [https://en.wikipedia.org/w/index.php?](https://en.wikipedia.org/w/index.php?title=Gibbs%27_inequality&oldid=1231436245) [title=Gibbs%27\\_inequality&oldid=1231436245](https://en.wikipedia.org/w/index.php?title=Gibbs%27_inequality&oldid=1231436245). Online; accessed 10-September-2024. [18](#page-17-0) Wikipedia. Tietze extension theorem, 2024b. URL [https://en.wikipedia.org/w/index.](https://en.wikipedia.org/w/index.php?title=Tietze_extension_theorem&oldid=1237682676) [php?title=Tietze\\_extension\\_theorem&oldid=1237682676](https://en.wikipedia.org/w/index.php?title=Tietze_extension_theorem&oldid=1237682676). Online; accessed 10-September-2024. [17](#page-16-1) Sewall Wright. Correlation and causation. *Journal of Agricultural Research*, (7), 1921. [8](#page-7-3) Zhirong Wu, Yuanjun Xiong, Stella X. Yu, and Dahua Lin. Unsupervised Feature Learning via Non-Parametric Instance Discrimination. pp. 3733–3742, 2018. URL [https://openaccess.thecvf.com/content\\_cvpr\\_2018/html/Wu\\_](https://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.html) [Unsupervised\\_Feature\\_Learning\\_CVPR\\_2018\\_paper.html](https://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.html). [4](#page-3-3) Qi Zhang, Yifei Wang, and Yisen Wang. Identifiable Contrastive Learning with Automatic Feature Importance Discovery, October 2023. URL <http://arxiv.org/abs/2310.18904>. arXiv:2310.18904 [cs]. [3](#page-2-0) Roland S. Zimmermann, Yash Sharma, Steffen Schneider, Matthias Bethge, and Wieland Brendel. Contrastive Learning Inverts the Data Generating Process. *arXiv:2102.08850 [cs]*, February 2021. URL <http://arxiv.org/abs/2102.08850>. arXiv: 2102.08850. [2,](#page-1-1) [3,](#page-2-0) [4,](#page-3-3) [7,](#page-6-3) [8,](#page-7-3) [29](#page-28-0)
# A IDENTIFIABILITY OF LATENTS DRAWN FROM A VMF AROUND CLUSTER VECTORS

This section contains the formal statement and proof of our main theoretical result. Appx. [A.1](#page-15-2) contains the relevant definition of affine generator systems. Appx. [A.2](#page-15-3) contains the assumptions and the proof for all four combinations of unit-normalized and non-normalized features/cluster vectors for [parametric instance discrimination.](#page-30-4) Appx. [A.3](#page-20-1) discusses a special case, supervised classification.

#### A.1 AFFINE GENERATOR SYSTEMS

Definition 1 (Affine Generator System). *A system of vectors* {v<sup>c</sup> ∈ <sup>R</sup> d |c ∈ C } *is called an* affine generator system *if any vector in* R d *is an affine linear combination of the vectors in the system. Put into symbols: for any* v ∈ R d *there exist coefficients* α<sup>c</sup> ∈ <sup>R</sup>*, such that*

$$\mathbf{v} = \sum_{c \in \mathcal{C}} \alpha_c \mathbf{v}_c \quad \text{and} \quad \sum_{c \in \mathcal{C}} \alpha_c = 1. \quad (3)$$

Lemma 1 (Properties of affine generator systems). *The following hold for any affine generator system* {v<sup>c</sup> ∈ <sup>R</sup> d |c ∈ C }*:*

- *1. for any* a ∈ C *the system* {v<sup>c</sup> − va|c ∈ C } *is now a generator system of* <sup>R</sup> d *;*
- *2. the invertible linear image of an affine generator system is also an affine generator system.*

#### A.2 IDENTIFIABILITY OF [PARAMETRIC INSTANCE DISCRIMINATION](#page-30-4)

Assumptions [1C](#page-3-2) (DGP with vMF samples around cluster vectors). *Assume the following [DGP:](#page-30-3)*

- *(i) There exists a finite set of classes* C *, represented by a set of unit-norm* d*-dimensional clustervectors* {vc|c ∈ C } ⊆ <sup>S</sup> d−1 *such that they form an affine generator system of* R d *.*
- *(ii) There is a finite set of instance labels* I *and a well-defined, surjective* class function C : I → C *(every label belongs to exactly one class and every class is in use).*
- *(iii) A data sample* x *belongs to class* C = C(I) *and is labeled with a uniformly-chosen instance label, i.e.,* I ∈ Uni(I )*.*
- *(iv) The latent* z ∈ S <sup>d</sup>−<sup>1</sup> *of our data sample with label* I *is drawn from a vMF distribution around the cluster vector* v<sup>C</sup> *, where* C = C(I)*:*

$$z \sim p(z|C) \propto e^{\kappa \langle v_C, z \rangle}. \quad (4)$$

- *(v) The data sample* x *is generated by passing the latent* z *through a continuous and injective generator function* g :S <sup>d</sup>−<sup>1</sup>→<sup>R</sup> <sup>D</sup>*, i.e.,* x = g(z)*.*

Assume that, using the DIET objective [\(6\)](#page-15-4), we train a continuous encoder f : R <sup>D</sup> → <sup>R</sup> <sup>d</sup> on x and a linear classification head W on top of f. The rows of W are w<sup>⊤</sup> i |i ∈ I . In other words, W computes similarities (scalar products) between its rows and the embeddings:

$$W : \mathbf{f}(\mathbf{x}) \mapsto [\langle \mathbf{w}_i, \mathbf{f}(\mathbf{x}) \rangle |_{i \in \mathcal{I}}]. \quad (5)$$

In DIET, we optimize the following objective among all possible continuous encoders f, linear classifiers W, and β > 0:

$$\mathcal{L}(\mathbf{f}, \mathbf{W}, \beta) = \mathbb{E}_{(\mathbf{x}, I)} \left[ -\ln \frac{e^{\beta \langle \mathbf{w}_I, \mathbf{f}(\mathbf{x}) \rangle}}{\sum_{j \in \mathcal{J}} e^{\beta \langle \mathbf{w}_j, \mathbf{f}(\mathbf{x}) \rangle}} \right] \quad (6)$$

In the special case where the embeddings f(x) are unnormalized, but the parameter vectors w<sup>i</sup> are unit-normalized, the identifiability proof will solicit another, technical assumption:

Assumption 2 (Diverse data). *The system* {vc|c ∈ C } *is said to be diverse enough, if the following* |C | × 2d *matrix has full column rank of* 2d*:*

$$\begin{pmatrix} \dots & \dots & \dots \\ (\mathbf{v}_c \odot \mathbf{v}_c)^\top & \mathbf{v}_c^\top & \dots \\ \dots & \dots & \dots \end{pmatrix}, \quad (7)$$

*where* [x ⊙ y]<sup>i</sup> = xiy<sup>i</sup> *is the elementwise- or Hadamard product.*

*As long as* |C | ≥ 2d*, this property holds almost surely w.r.t. the Lebesgue-measure of* <sup>S</sup> <sup>d</sup>−<sup>1</sup> *or any continuous probability distribution of* v<sup>c</sup> ∈ <sup>S</sup> d−1 *.*

Theorem [1C](#page-4-0) (Identifiability of latents drawn from a vMF around cluster vectors). *Let* (f,W, β) *globally minimize the DIET objective* [\(6\)](#page-15-4) *under Assums. [1C](#page-15-0) and the following additional constraints:*

- *C1. both the embeddings* f(x) *and* wi*'s are unit-normalized. Then:*
  - *(a)* h = f g *is orthogonal linear, i.e., the latents are identified up to an orthogonal linear transformation;*
  - *(b)* w<sup>i</sup> = h(vC(i)) *for any* i ∈ I *, i.e.,* wi*'s identify the cluster-vectors* v<sup>c</sup> *up to the same orthogonal linear transformation;*
- *(c)* β = κ*, the temperature of the vMF distribution is also identified. C2. the embeddings* f(x) *are unit-normalized, the* wi*'s are unnormalized. Then:*
  - *(a)* h = f g *is orthogonal linear;*
- *(b)* w<sup>i</sup> = κ β h(vC(i)) + ψ *for any* i ∈ I *, where* ψ *is a constant vector independent of* i*. C3. the embeddings* f(x) *are unnormalized, while the* wi*'s are unit-normalized. If the system* {vc|c} *is diverse enough in the sense of Assum. [2,](#page-15-5) then:*
  - *(a)* w<sup>i</sup> = OvC(i) *, for any* i ∈ I *, where* O *is orthogonal linear;*
- *(b)* h = f ◦ g = κ <sup>β</sup><sup>O</sup> *with the same orthogonal linear transformation, but scaled with* <sup>κ</sup> β *. C4. neither the embeddings* f(x) *nor the rows of* W *are unit-normalized. Then:*
  - *(a)* h = f g *is linear;*
  - *(b)* w<sup>i</sup> *identifies* vC(i) *up to an affine linear transformation.*

*Furthermore, in all cases, the row vectors that belong to samples of the same class are equal, i.e., for any* i, j ∈ I *,* C(i) = C(j) *implies* w<sup>i</sup> = w<sup>j</sup> *.*

Remark. *In cases [C2](#page-16-2) and [C4,](#page-16-3) the cluster vectors are unnormalized and, therefore, can absorb the temperature parameter* β*. Thus* β *can be set to* 1 *without loss of generality. In case [C3,](#page-16-4) it is* f *that can absorb* β*.*

*Proof.* Step 1: Deriving an equation characterizing the global optimizers of the objective.

Rewriting the objective in terms of latents: we plug the expression x = g(z)into the optimization objective [\(6\)](#page-15-4) to express the dependence in terms of the latents z:

$$\mathcal{L}(\mathbf{f}, \mathbf{W}, \beta) = \mathbb{E}_{(\mathbf{z}, I)} \left[ -\ln \frac{e^{\beta(\mathbf{w}_I, \mathbf{f} \circ \mathbf{g}(\mathbf{z}))}}{\sum_{j \in \mathcal{J}} e^{\beta(\mathbf{w}_j, \mathbf{f} \circ \mathbf{g}(\mathbf{z}))}} \right] = \mathcal{L}_{\mathbf{z}}(\mathbf{f} \circ \mathbf{g}, \mathbf{W}, \beta), \quad (8)$$

where the optimization is still over f (and not h = f ◦ g).

We note that the generator g is, by assumption, continuously invertible on the *compact* set S d−1 . Therefore, its image g(S d−1 ) is compact, too, and its inverse g −1 is also continuous. By Tietze's extension theorem [\(Wikipedia, 2024b\)](#page-14-8), g −1 can be continuously extended to a function F : R <sup>D</sup> → d−1 . Therefore, any continuous function h : S <sup>d</sup>−<sup>1</sup> → <sup>R</sup> d can take the role of f ◦ g by substituting f = h ◦ F continuous, since now f ◦ g = h ◦ (F ◦ g) = h ◦ id<sup>S</sup> <sup>d</sup>−<sup>1</sup> = h.

Hence, minimizing Lz(f ◦ g,W, β) (and by extension L(f,W, β)) for continuous f equates to minimizing Lz(h,W, β) for continuous h:

$$\mathcal{L}_z(\mathbf{h}, \mathbf{W}, \beta) = \mathbb{E}_{(z, I)} \left[ -\ln \frac{e^{\beta \langle \mathbf{w}_I, \mathbf{h}(z) \rangle}}{\sum_{j \in \mathcal{J}} e^{\beta \langle \mathbf{w}_j, \mathbf{h}(z) \rangle}} \right]. \quad (9)$$

Expressing the condition for global optimality of the objective: We rewrite the objective [\(9\)](#page-16-5) by 1) using the indicator variable δI=<sup>i</sup> of the event {I = i} and 2) applying the law of total expectation:

$$\mathcal{L}_{\mathbf{z}}(\mathbf{h}, \mathbf{W}, \boldsymbol{\beta}) = \mathbb{E}_{(\mathbf{z}, I)} \left[ - \sum_{i \in \mathcal{I}} \delta_{I=i} \ln \frac{e^{\beta \langle \mathbf{w}_i, \mathbf{h}(\mathbf{z}) \rangle}}{\sum_{j \in \mathcal{J}} e^{\beta \langle \mathbf{w}_j, \mathbf{h}(\mathbf{z}) \rangle}} \right] \quad (10)$$

$$= \mathbb{E}_z \left[ \mathbb{E}_z \left[ - \sum_{i \in \mathcal{I}} \delta_{I=i} \ln \frac{e^{\beta \langle w_i, \mathbf{h}(z) \rangle}}{\sum_{j \in \mathcal{I}} e^{\beta \langle w_j, \mathbf{h}(z) \rangle}} \middle| \mathbf{z} \right] \right]. \quad (11)$$

Using the properties that E A f(B) B = E -A B f(B) and that <sup>E</sup>[δI=<sup>i</sup> ] = <sup>P</sup>(I = i), we conclude that:

$$\mathcal{L}_z(\mathbf{h}, \mathbf{W}, \beta) = \mathbb{E}_z \left[ - \sum_{i \in \mathcal{I}} \mathbb{E}_I \left[ \delta_{I=i} \ln \frac{e^{\beta \langle \mathbf{w}_i, \mathbf{h}(\mathbf{z}) \rangle}}{\sum_{j \in \mathcal{I}} e^{\beta \langle \mathbf{w}_j, \mathbf{h}(\mathbf{z}) \rangle}} \middle| \mathbf{z} \right] \right] \quad (12)$$

$$= \mathbb{E}_{\mathbf{z}} \left[ - \sum_{i \in \mathcal{I}} \mathbb{E}_I \left[ \delta_{I=i} | \mathbf{z} \right] \ln \frac{e^{\beta \langle \mathbf{w}_i, \mathbf{h}(\mathbf{z}) \rangle}}{\sum_{j \in \mathcal{I}} e^{\beta \langle \mathbf{w}_j, \mathbf{h}(\mathbf{z}) \rangle}} \right] \quad (13)$$

$$= \mathbb{E}_{\mathbf{z}} \left[ - \sum_{i \in \mathcal{I}} \mathbb{P}(I = i | \mathbf{z}) \ln \frac{e^{\beta \langle \mathbf{w}_i, \mathbf{h}(\mathbf{z}) \rangle}}{\sum_{j \in \mathcal{I}} e^{\beta \langle \mathbf{w}_j, \mathbf{h}(\mathbf{z}) \rangle}} \right]. \quad (14)$$

By Gibbs' inequality [\(Wikipedia, 2024a\)](#page-14-9), the cross-entropy inside the expectation is globally minimized if and only if

$$\frac{e^{\beta\langle w_i, h(z) \rangle}}{\sum_{j \in \mathcal{J}} e^{\beta\langle w_j, h(z) \rangle}} = \mathbb{P}(I = i | z), \quad \text{for any } i \in \mathcal{J}. \quad (15)$$

Moreover, the entire expectation is globally minimized if and only if the above equality [\(15\)](#page-17-1) holds almost everywhere for z ∈ S d−1 .

Using that instance label I is uniformly distributed, or <sup>P</sup>(I = j) = <sup>P</sup>(I = i), the likelihood of the sample being in class i can be expressed via Bayes' theorem as:

$$\mathbb{P}(I = i | \mathbf{z}) = \frac{p(\mathbf{z}|I = i)\mathbb{P}(I = i)}{\sum_{j \in \mathcal{J}} p(\mathbf{z}|I = j)\mathbb{P}(I = j)} = \frac{p(\mathbf{z}|I = i)}{\sum_{j \in \mathcal{J}} p(\mathbf{z}|I = j)}. \quad (16)$$

Substituting [\(16\)](#page-17-2) into [\(15\)](#page-17-1) yields that for any i ∈ I and almost everywhere w.r.t. z ∈ <sup>S</sup> d−1

$$\text{yields that for any } i \in \mathcal{I} \text{ and almost everywhere w.r.t. } \mathbf{z} \in \mathbb{S}^{d-1}: \frac{e^{\beta\langle \mathbf{w}_i, \mathbf{h}(\mathbf{z}) \rangle}}{\sum_{j \in \mathcal{I}} e^{\beta\langle \mathbf{w}_j, \mathbf{h}(\mathbf{z}) \rangle}} = \frac{p(\mathbf{z} | I = i)}{\sum_{j \in \mathcal{I}} p(\mathbf{z} | I = j)}. \quad (17)$$

We now divide the equation [\(17\)](#page-17-3) for the probability of a sample having label i with that of having label k and take the logarithm. This yields that Lz(h,W, β) is globally minimized if and only if

$$\beta \langle w_i - w_k, \mathbf{h}(z) \rangle = \ln \frac{p(z|I = i)}{p(z|I = k)} \quad (18)$$

holds for any i, k ∈ I and almost everywhere w.r.t. z ∈ <sup>S</sup> d−1 .

Plugging in the vMF distribution: Plugging the assumed conditional distribution from [\(4\)](#page-15-6) into [\(18\)](#page-17-4) yields the equivalent expression:

$$\beta \langle w_i - w_k, h(z) \rangle = \kappa \langle v_{\mathcal{C}(i)} - v_{\mathcal{C}(k)}, z \rangle, \quad (19)$$

which holds for any i, k ∈ I and almost everywhere w.r.t. z ∈ <sup>S</sup> d−1 . Since h is continuous, the equation holds almost everywhere w.r.t. z if and only if it holds for all z ∈ S d−1 .

Observe that if h = id|<sup>S</sup> <sup>d</sup>−<sup>1</sup> , w<sup>i</sup> = vC(i) for any i ∈ I , and β = κ, then the equation is satisfied. Thus, we can conclude that the global minimum of the cross-entropy loss is achieved.

# Step 2: Solving the equation for h,W and proving identifiability.

We now find all solutions to prove the identifiability of the latent variables and that of the cluster vectors. Denote w˜ <sup>i</sup> = β <sup>κ</sup>w<sup>i</sup> to simplify the above equation to:

$$\langle \tilde{w}_i - \tilde{w}_k, \mathbf{h}(\mathbf{z}) \rangle = \langle \mathbf{v}_{\mathcal{C}(i)} - \mathbf{v}_{\mathcal{C}(k)}, \mathbf{z} \rangle. \quad (20)$$

h is injective and has full-dimensional image: We prove that h is injective. Assume that h(z1) = h(z2) for some z1, z<sup>2</sup> ∈ <sup>S</sup> d−1 . Plugging z<sup>1</sup> and z<sup>2</sup> into [\(20\)](#page-17-5) and subtracting the two equations yields:

$$0 = \langle \tilde{w}_i - \tilde{w}_k, \mathbf{h}(z_1) - \mathbf{h}(z_2) \rangle = \langle \mathbf{v}_{\mathcal{C}(i)} - \mathbf{v}_{\mathcal{C}(k)}, z_1 - z_2 \rangle, \quad (21)$$

for any i, k. However, as the cluster vectors {vc|c} form an affine generator system, the vectors {vC(i) − vC(k) |i, k} form a generator system of <sup>R</sup> d (see Lem. [1\)](#page-15-7). Therefore, ⟨y, z<sup>1</sup> − z2⟩ = 0, for any y ∈ R d , which holds if and only if z<sup>1</sup> = z2. Hence, h is injective.

By the Borsuk-Ulam theorem, for any continuous map from S d−1 to a space of dimensionality at most d−1 there exists some pair of antipodal points that are mapped to the same point. Consequently, no such function can be injective at the same time. Since h : S <sup>d</sup>−<sup>1</sup> → <sup>R</sup> d is injective, the linear span of its image must be R d .

Collapse of wi's: We prove that w˜ <sup>i</sup> = w˜ <sup>k</sup> if C(i) = C(k), i.e., samples from the same cluster will have equal rows of W associated with them.

Assume that C(i) = C(k) and substitute them into [\(20\)](#page-17-5):

$$\langle \tilde{w}_i - \tilde{w}_k, \mathbf{h}(z) \rangle = 0 \quad \text{for any } z \in \mathbb{S}^{d-1}. \quad (22)$$

However, we have just seen that the linear span of the image of h is R d , which implies that w˜ <sup>i</sup> = w˜ <sup>k</sup>. We may abuse our notation by setting w˜ <sup>c</sup> = w˜ <sup>i</sup> if C(i) = c, which yields a new form for [\(20\)](#page-17-5):

$$\langle \tilde{w}_a - \tilde{w}_b, h(z) \rangle = \langle v_a - v_b, z \rangle, \quad (23)$$

for any a, b ∈ C and any z ∈ <sup>S</sup> d−1 .

Linear transformation from v<sup>a</sup> − v<sup>b</sup> to w˜ <sup>a</sup> − w˜ <sup>b</sup>: We now prove the existence of a linear map A on R d such that A(v<sup>a</sup> − vb) = w˜ <sup>a</sup> − w˜ <sup>b</sup> for any a, b ∈ C . For this, we prove that the following mapping is well-defined:

$$\mathcal{A} : \sum_{a,b \in \mathcal{C}} \lambda_{ab}(\mathbf{v}_a - \mathbf{v}_b) \mapsto \sum_{a,b \in \mathcal{C}} \lambda_{ab}(\tilde{\mathbf{w}}_a - \tilde{\mathbf{w}}_b). \quad (24)$$

Since the system {v<sup>a</sup> − vb|a, b} is not necessarily linearly independent, we have to prove that the mapping is independent of the choice of the linear combination. More precisely if for some coefficients λab, λ′ ab

$$\sum_{a,b \in \mathcal{C}} \lambda_{ab}(\mathbf{v}_a - \mathbf{v}_b) = \sum_{a,b \in \mathcal{C}} \lambda'_{ab}(\mathbf{v}_a - \mathbf{v}_b) \quad (25)$$

holds, then it should be implied that

$$\sum_{a,b \in \mathcal{C}} \lambda_{ab}(\tilde{w}_a - \tilde{w}_b) = \sum_{a,b \in \mathcal{C}} \lambda'_{ab}(\tilde{w}_a - \tilde{w}_b). \quad (26)$$

Assume that [\(25\)](#page-18-0) holds. Then, the difference of the two sides is:

$$0 = \sum_{a,b \in \mathcal{C}} (\lambda_{ab} - \lambda'_{ab})(\mathbf{v}_a - \mathbf{v}_b). \quad (27)$$

Taking the scalar product with an arbitrary z ∈ S d−1 and using the linearity of the scalar product gives us:

$$0 = \left\langle \sum_{a,b \in \mathcal{E}} (\lambda_{ab} - \lambda'_{ab})(\mathbf{v}_a - \mathbf{v}_b), \mathbf{z} \right\rangle = \sum_{a,b \in \mathcal{E}} (\lambda_{ab} - \lambda'_{ab})\langle \mathbf{v}_a - \mathbf{v}_b, \mathbf{z} \rangle. \quad (28)$$

Now using [\(23\)](#page-18-1) yields:

$$0 = \sum_{a,b \in \mathcal{E}} (\lambda_{ab} - \lambda'_{ab}) \langle \tilde{w}_a - \tilde{w}_b, \mathbf{h}(z) \rangle = \left\langle \sum_{a,b \in \mathcal{E}} (\lambda_{ab} - \lambda'_{ab})(\tilde{w}_a - \tilde{w}_b), \mathbf{h}(z) \right\rangle. \quad (29)$$

However, the linear span of the image of h is R d , which implies that

$$\sum_{a,b \in \mathcal{C}} (\lambda_{ab} - \lambda'_{ab})(\tilde{w}_a - \tilde{w}_b) = 0, \quad (30)$$

equivalent to [\(26\)](#page-18-2). Therefore, the mapping is well-defined and the linearity of A follows.

h is linear: Equation [\(23\)](#page-18-1) becomes:

$$\langle \mathcal{A}(\mathbf{v}_a - \mathbf{v}_b), \mathbf{h}(z) \rangle = \langle \mathbf{v}_a - \mathbf{v}_b, z \rangle, \quad (31)$$

for any a, b ∈ C and any z ∈ <sup>S</sup> d−1 . Nevertheless, {v<sup>a</sup> − vb|a, b ∈ C } is a generator system of <sup>R</sup> d , and, hence, [\(31\)](#page-18-3) is equivalent to

$$\langle Ay, h(z) \rangle = \langle y, z \rangle, \quad \text{for any } y \in \mathbb{R}^d \text{ and any } z \in \mathbb{S}^{d-1}. \quad (32)$$

This is further equivalent to

$$\langle y, \mathcal{A}^\top h(z) \rangle = \langle y, z \rangle. \quad (33)$$

Proving Thm. [1C](#page-16-0) case [C4:](#page-16-3) We have shown that h is linear. Furthermore, from [\(31\)](#page-18-3) it follows, by fixing b and defining ψ = Av<sup>b</sup> − wb, that

$$\tilde{w}_a = \mathcal{A}v_a + \psi, \quad \text{for any } a \in \mathcal{C}, \quad (34)$$

which proves case [C4](#page-16-3) of Thm. [1C.](#page-16-0)

Proving Thm. [1C](#page-16-0) case [C2:](#page-16-2) As a special case of the previous one, now we assume that h(z) is unit-normalized and maps S d−1 to S d−1 . That amounts to h = (A<sup>⊤</sup>) <sup>−</sup><sup>1</sup> being linear, normpreserving, and therefore orthogonal. Consequently A is also orthogonal, h = A and [\(34\)](#page-19-0) simplifies to <sup>β</sup> <sup>κ</sup>w<sup>a</sup> = w˜ <sup>a</sup> = Av<sup>a</sup> + ψ = h(va) + ψ, which proves [C2](#page-16-2) of Thm. [1C.](#page-16-0)

Proving Thm. [1C](#page-16-0) case [C1:](#page-16-6) We now assume that both h and wi's are unit-normalized. Consequently, h = A is orthogonal linear and w<sup>a</sup> = κ <sup>β</sup>Av<sup>a</sup> + ψ.

Therefore, on one hand, the wa's lie on a d-dimensional hypersphere of radius <sup>κ</sup> β and center ψ. On the other hand, by definition, wa's also lie on the unit hypersphere S d−1 .

Since the system {wa|a ∈ C } is the bijective affine linear image of the affine generator system {va|a ∈ C }, {wa|a ∈ C } is also an affine generator system (Lem. [1\)](#page-15-7). Consequently, there could be at most one hypersphere in R <sup>d</sup> which contains all the wa's. Hence <sup>κ</sup> <sup>β</sup> = 1, ψ = 0, and w<sup>a</sup> = h(va), which proves [C1](#page-16-6) of Thm. [1C.](#page-16-0)

Proving Thm. [1C](#page-16-0) case [C3:](#page-16-4) Finally, we assume that wi's are unit-normalized. As this is a special case of Thm. [1C](#page-16-0) [C4,](#page-16-3) we know that there exists a constant vector ψ such that:

$$w_a = \frac{\kappa}{\beta} \mathcal{A}v_a + \psi, \quad (35)$$

for any a ∈ C . We are going to prove that O = κ <sup>β</sup>A is orthogonal and ψ = 0.

Let O = U <sup>⊤</sup>ΣV be the singular value decomposition (SVD) of O. Premultiplying with U yields:

$$\mathcal{U}w_a = \Sigma \mathcal{V}v_a + \mathcal{U}\psi. \quad (36)$$

As orthogonal transformations U and V keep their arguments unit-normalized and {Vv<sup>a</sup> − Vvb} is still an affine generator system (Lem. [1\)](#page-15-7), we may assume without the loss of generality that

$$w_a = \Sigma v_a + \psi, \quad (37)$$

for any a ∈ C , where all va's and wa's are unit-normalized.

Let us assume that ψ ̸= 0. In that case both sides of [\(37\)](#page-19-1) can be scaled such that the offset ψ has unit norm. In this case wa's are no longer on the unit hypersphere, but they instead have a mutual norm r. Assuming that the diagonal elements of Σ are σ = (σ1, . . . , σd), this is equivalent to:

$$r^2 = \|\Sigma \mathbf{v}_a + \psi\|^2 = \|\Sigma \mathbf{v}_a\|^2 + 2\langle \Sigma \mathbf{v}_a, \psi \rangle + \|\psi\|^2 \quad (38)$$

$$= \langle \mathbf{v}_a \odot \mathbf{v}_a, \boldsymbol{\sigma} \odot \boldsymbol{\sigma} \rangle + \langle \mathbf{v}_a, 2\boldsymbol{\sigma} \odot \boldsymbol{\psi} \rangle + 1, \quad (39)$$

where [x ⊙ y]<sup>i</sup> = xiy<sup>i</sup> is the elementwise product. Eq. [\(39\)](#page-19-2) is equivalent to the following:

$$(v_a \odot v_a)^\top (\boldsymbol{\sigma} \odot \boldsymbol{\sigma}) + v_a^\top (2\boldsymbol{\sigma} \odot \boldsymbol{\psi}) - r^2 = -1. \quad (40)$$

Collecting the equations for all a ∈ C yields:

$$\mathcal{D} \begin{pmatrix} \boldsymbol{\sigma} \odot \boldsymbol{\sigma} \\ 2\boldsymbol{\sigma} \odot \boldsymbol{\psi} \\ r^2 \end{pmatrix} = -\mathbf{1}_{|\mathcal{C}|}, \quad (41)$$

where D is the following |C | × (2d + 1) matrix:

$$\mathcal{D} = \begin{pmatrix} \dots & \dots & \dots \\ (\mathbf{v}_a \odot \mathbf{v}_a)^\top & \mathbf{v}_a^\top & -1 \\ \dots & \dots & \dots \end{pmatrix}. \quad (42)$$

By Assum. [2,](#page-15-8) the left |C | × 2d submatrix of D has full rank of 2d. Consequently, the solution space to the more general, linear equation Dt = −1|C<sup>|</sup> , t ∈ R d , has a dimensionality of at most 1. By the unit-normality of va, we have (v<sup>a</sup> ⊙ va) <sup>⊤</sup>1<sup>d</sup> = 1. From this, the solutions are exactly the following:

$$t = \begin{pmatrix} \gamma \cdot \mathbf{1}_d \\ \mathbf{0}_d \\ \gamma + 1 \end{pmatrix}, \quad \text{where } \gamma \in \mathbb{R}. \quad (43)$$

Therefore, for any solution of [\(41\)](#page-19-3) there exists γ such that:

$$\sigma \odot \sigma = \gamma \cdot 1_d \quad (44)$$

$$\sigma \odot \psi = \mathbf{0}_d. \quad (45)$$

However, as the original transformation A was invertible, all singular values σ<sup>i</sup> are strictly positive and, thus, it follows that ψ = 0. This is a technical contradiction to our initial assumption that ψ ̸= 0. Thus, it follows that ψ = 0.

Therefore, [\(37\)](#page-19-1) becomes:

$$w_a = \sum v_a, \quad (46)$$

where all va's and wa's are unit-normalized. Following the same derivation yields:

$$1 = \|\Sigma \mathbf{v}_a\|^2 = (\mathbf{v}_a \odot \mathbf{v}_a)^\top (\boldsymbol{\sigma} \odot \boldsymbol{\sigma}), \quad (47)$$

or, after collecting the equations for all a ∈ C :

$$\mathcal{B}(\boldsymbol{\sigma} \odot \boldsymbol{\sigma}) = \mathbf{1}_{|\mathcal{C}|}, \quad (48)$$

where B is the |C | × d matrix

$$\mathcal{B} = \begin{pmatrix} & & & & & & & \\ & & \ddots & & & & & \\ & & & \ddots & & & & \\ & & & & \ddots & & & \\ & & & & & \ddots & & \\ & & & & & & \ddots & \\ & & & & & & & \ddots \end{pmatrix}. \quad (49)$$

By Assum. [2,](#page-15-8) B has full rank, thus, there is at most one solution to the equation Bt = 1|C<sup>|</sup> . Due to the unit-normality of va's, this solution is exactly t = 1d. However, as the singular values σ<sup>i</sup> are all positive, the only solution to σ ⊙ σ = 1<sup>d</sup> is σ = 1d. Equivalently, O = κ <sup>β</sup>A is orthogonal.

Furthermore, 
$$h = (\mathcal{A}^\top)^{-1} = (\frac{\beta}{\kappa} \mathcal{O}^\top)^{-1} = \frac{\kappa}{\beta} \mathcal{O}$$
. □

A.3 IDENTIFIABILITY OF SUPERVISED CLASSIFICATION

Assumption 3 (DGP with vMF samples around cluster vectors). *Assume the following [DGP:](#page-30-3)*

- *(i) There exists a finite set of classes* C *, represented by a set of unit-norm* d*-dimensional clustervectors* {vc|c ∈ C } ⊆ <sup>S</sup> d−1 *such that they form an affine generator system of* R d *.*
- *(ii) A data sample* x *belongs to a* uniformly *chosen class* C ∈ Uni(C )*.*
- *(iii) The latent* z ∈ S <sup>d</sup>−<sup>1</sup> *of our data sample* x *with label* C *is drawn from a vMF distribution around the cluster vector* v<sup>C</sup> *:*

$$z \sim p(z|C) \propto e^{\kappa \langle v_C, z \rangle}. \quad (50)$$

- *(iv) The data sample* x *is generated by passing the latent* z *through a continuous and injective generator function* g :S <sup>d</sup>−<sup>1</sup>→<sup>R</sup> <sup>D</sup>*, i.e.,* x = g(z)*.*

We would like to point out that the assumption of the class label C being uniform restricts the scope the following theorem as it cannot account for imbalanced class labels. This shortcoming did not affect Thm. [1,](#page-4-0) as the uniform distribution over instance labels is a natural choice in practical scenarios with finite datasets.

Theorem [2C](#page-4-1) (Identifiability of latent variables drawn from a vMF around class vectors). *Let Assums. [1C](#page-15-0) hold and suppose that a continuous encoder* f : R <sup>D</sup> → <sup>R</sup> d *, a linear classifier* W *with rows* {w<sup>⊤</sup> c | c ∈ C }*, and* β > 0 *globally minimize the cross-entropy objective:*

$$\mathcal{L}(\mathbf{f}, \mathbf{W}, \boldsymbol{\beta}) = \mathbb{E}_{(\mathbf{x}, C)} \left[ -\ln \frac{e^{\beta \langle \mathbf{w}_C, \mathbf{f}(\mathbf{x}) \rangle}}{\sum_{c' \in \mathcal{C}} e^{\beta \langle \mathbf{w}_{c'}, \mathbf{f}(\mathbf{x}) \rangle}} \right].$$

#### *Proof.*

Step 1: Rewriting the Objective in Terms of h. We begin by expressing the loss function in terms of the latent variable z. Recall that x = g(z) and h = f ◦ g. Substituting into the loss function:

$$\mathcal{L}(\mathbf{f}, \mathbf{W}, \beta) = \mathbb{E}_{(\mathbf{z}, \mathbf{c})} \left[ -\ln \frac{e^{\beta \langle \mathbf{w}_{\mathbf{c}}, \mathbf{h}(\mathbf{z}) \rangle}}{\sum_{\mathbf{c}' \in \mathcal{C}} e^{\beta \langle \mathbf{w}_{\mathbf{c}'}, \mathbf{h}(\mathbf{z}) \rangle}} \right]. \quad (51)$$

Since g is continuous and injective on the compact set S d−1 , its inverse g −1 exists and is continuous on g(S d−1 ). By Tietze's extension theorem, we can extend g −1 to a continuous function g −1 ext : R <sup>D</sup> → <sup>S</sup> d−1 . Therefore, any continuous function h : S <sup>d</sup>−<sup>1</sup> → <sup>R</sup> d corresponds to a continuous encoder f = h ◦ g −1 ext , satisfying f(x) = h(z).

Step 2: Optimality Condition of the Cross-Entropy Loss. At the global minimum of the crossentropy loss, the predicted class probabilities match the true conditional probabilities almost everywhere. That is, for all z ∈ S d−1 and all c ∈ C :

$$\frac{e^{\beta\langle \mathbf{w}_c, \mathbf{h}(\mathbf{z}) \rangle}}{\sum_{c' \in \mathcal{C}} e^{\beta\langle \mathbf{w}_{c'}, \mathbf{h}(\mathbf{z}) \rangle}} = \mathbb{P}(C = c \mid \mathbf{z}). \quad (52)$$

Step 3: Expressing the True Conditional Probabilities. Using Bayes' theorem and the fact that classes are uniformly distributed (<sup>P</sup>(C = c) is constant[<sup>2</sup>](#page-21-0) ), we have:

$$\mathbb{P}(C = c \mid \mathbf{z}) = \frac{p(\mathbf{z} \mid C = c)}{\sum_{c' \in \mathcal{C}} p(\mathbf{z} \mid C = c')}. \quad (53)$$

Given that, by assumption, the latent z follows a von Mises-Fisher (vMF) distribution centered at vc:

$$p(\mathbf{z} \mid C = c) \propto e^{\kappa(\mathbf{v}_c, \mathbf{z})}. \quad (54)$$

Substituting into the conditional probability:

$$\mathbb{P}(C = c \mid \mathbf{z}) = \frac{e^{\kappa(\mathbf{v}_c, \mathbf{z})}}{\sum_{c' \in \mathcal{C}} e^{\kappa(\mathbf{v}_{c'}, \mathbf{z})}}. \quad (55)$$

Step 4: Equating Predicted and True Probabilities. Setting the predicted probabilities equal to the true probabilities, we obtain:

$$\frac{e^{\beta\langle \mathbf{w}_c, \mathbf{h}(\mathbf{z}) \rangle}}{\sum_{c' \in \mathcal{C}} e^{\beta\langle \mathbf{w}_{c'}, \mathbf{h}(\mathbf{z}) \rangle}} = \frac{e^{\kappa\langle \mathbf{v}_c, \mathbf{z} \rangle}}{\sum_{c' \in \mathcal{C}} e^{\kappa\langle \mathbf{v}_{c'}, \mathbf{z} \rangle}}. \quad (56)$$

Dividing the expressions for classes c and c ′ , we eliminate the denominators:

$$\frac{e^{\beta\langle \mathbf{v}_c, \mathbf{h}(\mathbf{z}) \rangle}}{e^{\beta\langle \mathbf{v}_{c'}, \mathbf{h}(\mathbf{z}) \rangle}} = \frac{e^{\beta\langle \mathbf{v}_c, \mathbf{z} \rangle}}{e^{\beta\langle \mathbf{v}_{c'}, \mathbf{z} \rangle}}. \quad (57)$$

Taking the logarithm of both sides:

$$\beta \left( \langle w_c, h(z) \rangle - \langle w_{c'}, h(z) \rangle \right) = \kappa \left( \langle v_c, z \rangle - \langle v_{c'}, z \rangle \right). \quad (58)$$

Simplifying:

$$\beta\langle w_c - w_{c'}, h(z) \rangle = \kappa\langle v_c - v_{c'}, z \rangle. \quad (59)$$

<sup>2</sup>We acknowledge that this assumption does not hold in many realistic scenarios, where the data distribution is unbalanced between the classes

Step 5: Defining Scaled Parameters. Let us define:

$$\tilde{w}_c = \frac{\beta}{\kappa} \mathbf{w}_c. \quad (60)$$

Then the key equation becomes:

$$\langle \tilde{w}_c - \tilde{w}_{c'}, \mathbf{h}(\mathbf{z}) \rangle = \langle \mathbf{v}_c - \mathbf{v}_{c'}, \mathbf{z} \rangle, \quad \forall c, c' \in \mathcal{C}. \quad (61)$$

Step 6: Establishing a Linear Relationship. Define the difference vectors:

$$\delta_{\tilde{\mathbf{v}}_{cc'}} = \tilde{w}_c - \tilde{w}_{c'}, \quad \delta_{\mathbf{v}_{cc'}} = \mathbf{v}_c - \mathbf{v}_{c'}. \quad (62)$$

Our key equation is now:

$$\langle \delta_{\tilde{w}_{cc'}}, \mathbf{h}(z) \rangle = \langle \delta_{\mathbf{v}_{cc'}}, z \rangle, \quad \forall c, c' \in \mathcal{C}. \quad (63)$$

Since the set {δ<sup>v</sup>cc′ | c, c′ ∈ C } spans <sup>R</sup> d (due to the affine generator system property), we can interpret this equation as stating that the inner products between h(z) and δw˜cc′ correspond to the inner products between z and δ<sup>v</sup>cc′ .

Step 7: Proving Injectivity and Full Rank of h. Suppose there exist z1, z<sup>2</sup> ∈ <sup>S</sup> d−1 such that h(z1) = h(z2). Then, for all c, c′ ∈ C :

$$\langle \delta_{v_{cc'}}, z_1 - z_2 \rangle = \langle \delta_{\tilde{w}_{cc'}}, \mathbf{h}(z_1) - \mathbf{h}(z_2) \rangle = 0. \quad (64)$$

Since {δ<sup>v</sup>cc′ } spans <sup>R</sup> d , it follows that z<sup>1</sup> − z<sup>2</sup> = 0, i.e., z<sup>1</sup> = z2. Therefore, h is injective.

By the Borsuk-Ulam theorem, an injective continuous map from S d−1 to R d with d ′ < d cannot exist. Thus, the image of h must be full-dimensional in R d .

Step 8: Defining a Linear Map A. We aim to find a linear map A : R <sup>d</sup> → <sup>R</sup> d such that:

$$\delta_{\tilde{w}_{cc'}} = \mathcal{A}^\top \delta_{\mathbf{v}_{cc'}}, \quad \forall c, c' \in \mathcal{C}. \quad (65)$$

This is well-defined because any linear dependency among the δ<sup>v</sup>cc′ translates to the same linear dependency among the δw˜cc′ , as shown below.

Suppose there are scalars {λcc′} such that:

$$\sum_{c,c'} \lambda_{cc'} \delta_{\mathbf{v}_{cc'}} = \mathbf{0}. \quad (66)$$

Then, using the key equation [\(63\)](#page-22-0):

$$\sum_{c,c'} \lambda_{cc'} \langle \delta_{\tilde{w}_{cc'}}, \mathbf{h}(\mathbf{z}) \rangle = \sum_{c,c'} \lambda_{cc'} \langle \delta_{\mathbf{v}_{cc'}}, \mathbf{z} \rangle = \left\langle \sum_{c,c'} \lambda_{cc'} \delta_{\mathbf{v}_{cc'}}, \mathbf{z} \right\rangle = 0. \quad (67)$$

Since h is injective and its image spans R d , the only way for this to hold for all h(z) is if:

$$\sum_{c,c'} \lambda_{cc'} \delta_{\tilde{w}_{cc'}} = \mathbf{0}. \quad (68)$$

Step 9: Concluding that h is Linear. Using the linear map A<sup>⊤</sup>, the key equation becomes:

$$\langle \mathcal{A}^\top \delta_{v_{cc'}}, h(z) \rangle = \langle \delta_{v_{cc'}}, z \rangle. \quad (69)$$

This implies:

$$\langle \delta_{v_{cc'}}, \mathcal{Ah}(z) - z \rangle = 0, \quad \forall c, c' \in \mathcal{C}. \quad (70)$$

Since {δ<sup>v</sup>cc′ } spans <sup>R</sup> d , it follows that:

$$\mathcal{A}h(z) = z, \quad \forall z \in \mathbb{S}^{d-1}. \quad (71)$$

Therefore, h is the inverse of A restricted to S d−1 , and since A is linear and invertible (due to the injectivity of h), it follows that h is linear:

$$h(z) = \mathcal{A}^{-1}z. \quad (72)$$

This completes the proof that h is linear.

Step 10: Conclusion. Under the given assumptions, we have shown that h = f ◦ g must be a linear function. This means that the latent variables z are identifiable up to a linear transformation determined by A<sup>−</sup><sup>1</sup> .

# B THE GENEALOGY OF CROSS-ENTROPY–BASED CLASSIFICATION METHODS

This section provides the necessary background on auxiliary-variable [ICA](#page-30-0) and discusses the connection between ICA and DIET, and InfoNCE and DIET.

#### B.1 AUXILIARY-VARIABE NONLINEAR ICA: GENERALIZED CONTRASTIVE LEARNING (GCL)

In this section, we discuss the most general auxiliary-variable nonlinear ICA, termed Generalized Contrastive Learning (GCL) [\(Hyvarinen et al., 2019\)](#page-12-4). GCL uses a conditionally factorizing source distribution (given auxiliary variable u): log p(s|u) is a sum of components qi(s<sup>i</sup> , u):

$$\log p(\mathbf{s}|u) = \sum_i q_i(s_i, u) \quad (73)$$

For this generalized model, [Hyvarinen et al.](#page-12-4) [\(2019\)](#page-12-4) define the following variability condition:

Assumption 4 (Assumption of Variability). *For any* y ∈ R <sup>n</sup> *(used as a drop-in replacement for the sources* s*), there exist* 2n + 1 *values for the auxiliary variable* u*, denoted by* u<sup>j</sup> , j = 0 . . . 2n *such that the* 2n *vectors in* R <sup>2</sup><sup>n</sup> *given by*

$$(\mathbf{w}(\mathbf{y}, \mathbf{u}_1) - \mathbf{w}(\mathbf{y}, \mathbf{u}_0)), (\mathbf{w}(\mathbf{y}, \mathbf{u}_2) - \mathbf{w}(\mathbf{y}, \mathbf{u}_0)), \dots, (\mathbf{w}(\mathbf{y}, \mathbf{u}_{2n}) - \mathbf{w}(\mathbf{y}, \mathbf{u}_0))$$

*with*

$$\mathbf{w}(\mathbf{y}, \mathbf{u}) = \left( \frac{\partial q_1(y_1, \mathbf{u})}{\partial y_1}, \dots, \frac{\partial q_n(y_n, \mathbf{u})}{\partial y_n}, \frac{\partial^2 q_1(y_1, \mathbf{u})}{\partial y_1^2}, \dots, \frac{\partial^2 q_n(y_n, \mathbf{u})}{\partial y_n^2} \right)$$

*are linearly independent.*

Assum. [4](#page-23-3) constrains the components of the first- and second derivatives of the functions constituting the sources' conditional log-density, given the auxiliary variable u. As the authors write: *"[Assum. [4\]](#page-23-3) is basically saying that the auxiliary variable must have a sufficiently strong and diverse effect on the distributions of the independent components."*

We state the required assumptions for the identifiability of [GCL,](#page-30-5) adapted from [\(Hyvarinen et al.,](#page-12-4) [2019,](#page-12-4) Thm. 1):

Assumption 5 (Auxiliary-variable ICA with conditionally independent sources (GCL)). *We assume the following for latent factors* z*, observations* x*, generative model* g*, encoder* f*(parametrized by a neural network), linear map* W*with* (f,W) *solving a multinomial regression problem:*

- *1. The observations are generated with a diffeomorphism* g : x = g(z), *where* dim x = dim z = d
- *2. The source components* z<sup>i</sup> *are conditionally independent, given a fully observed,* m−*dimensional [random variable \(RV\)](#page-30-13)* u, *i.e.,*

$$\log p(\mathbf{z}|\mathbf{u}) = \sum_i q_i(z_i, \mathbf{u}), \quad (74)$$

- *3. The conditional log-pdf* q<sup>i</sup> *is sufficiently smooth as a function of* z<sup>i</sup> *for any fixed* u
- *4. Assum. [6](#page-24-1) holds*
- *5. the multinomial regression function*

$$r(\mathbf{x}, \mathbf{u}) = \sum_i^n \psi_i(f_i(\mathbf{x}), \mathbf{u}), \quad (75)$$

*discriminating* (x, u) *vs* (x, u ∗ ) *has universal approximation capability, both for* ψ<sup>i</sup> *and a diffeomorphic* f = (f1, . . . , fn) *(parametrized by a neural network)*

When Assum. [5](#page-23-4) holds, [Hyvarinen et al.](#page-12-4) [\(2019\)](#page-12-4) showed identifiability up to component-wise invertible transformations.

For the special case when the conditional distribution comes from the exponential family (in the case of our chosen [vMF](#page-30-1) conditional, the distribution has order one), Assum. [5](#page-23-4) turns into a simpler form (Assum. [6\)](#page-24-1).

B.2 PARAMETRIC INSTANCE DISCRIMINATION (DIET) AND TIME-C[ONTRASTIVE](#page-30-6) LEARNING [\(TCL\)](#page-30-6)

Arbitrary labels: time and sample index As [Hyvarinen et al.](#page-12-4) [\(2019\)](#page-12-4) note in [\(Hyvarinen et al.,](#page-12-4) [2019,](#page-12-4) 5.4), u can stand for many types of additional information. [TCL](#page-30-6) uses the time index, which is assumed to be a [RV.](#page-30-13) Importantly, an arbitrarily defined class label, such as in DIET, can serve the same purpose. In this case, we denote the auxiliary variable u = c

Adapting the assumptions between [TCL](#page-30-6) and DIET. The only reason we cannot apply [\(Hyvarinen](#page-12-4) [et al., 2019,](#page-12-4) Thm. 1) is that our exponential family has order one, violating Assum. [4.](#page-23-3) This fact, however, shows our theory's consistency as we cannot go beyond identifiability up to linear (orthogonal or affine) transformation.

To fit our theory into the [ICA](#page-30-0) family of methods, we note that modeling the [DGP](#page-30-3) in DIET with a cluster-centric approach, we naturally fit most of the ICA assumptions. To compare our Assums. [1C](#page-15-0) to all the assumptions used for [\(Hyvarinen et al., 2019,](#page-12-4) Thm. 3) (cf. Assum. [5\)](#page-23-4), we note that the [vMF](#page-30-1) distribution belongs to the exponential family, and that requiring that the cluster vectors form an affine generator system (cf. Appx. [A.1](#page-15-2) for a definition and properties) satisfies the special case of the general sufficient variability Assum. [4](#page-23-3) condition:

Assumption 6 (Sufficient variability). *Define the modulation parameter matrix* L ∈ R (E−1)×dk *for* d−*dimensional exponential family distributions of order* k *with rows as:*

$$[\mathbf{L}]_j = (\boldsymbol{\theta}^j - \boldsymbol{\theta}^0)^T \quad (76)$$

$$\boldsymbol{\theta}^j = [\theta_{11}^j, \dots, \theta_{dk}^j]. \quad (77)$$

*Then, sufficient variability means that* rank(L) = dk, *i.e., the modulation parameter matrix has full column rank.*

To see how a [vMF](#page-30-1) fulfills Assum. [6,](#page-24-1) consider that the log-pdf qi(z<sup>i</sup> , c) comes from a conditional exponential family, i.e.:

$$q_i(z_i, c) = \sum_{j=1}^k [\tilde{q}_{ij}(z_i) \theta_{ij}(c)] - \log N_i(c) + \log Q_i(z_i), \quad (78)$$

$$= \kappa \langle \mathbf{v}_c, \mathbf{z} \rangle + \log C_d(\kappa) \quad (79)$$

where k is the order of the exponential family, N<sup>i</sup> is the normalizing constant, Q<sup>i</sup> the base measure, q˜i is the sufficient statistics, and the modulation parameters θ<sup>i</sup> := θi(c) depend on c. In our clustercentric [vMF](#page-30-1) conditional in [\(2\)](#page-4-4), k = 1 (i.e., we can drop the j index) and θi(c) = vc. This corresponds to [\(79\)](#page-24-2) above, where Cd(κ) is the concentration- and dimension-dependent normalization constant.

As our [DGP](#page-30-3) assumes that the cluster vectors form an affine generator system, and in the above Eq. [\(78\)](#page-24-3) the cluster vectors take the role of θij (c), we can prove that our DGP fulfils Assum. [6.](#page-24-1)

Lemma 2 (The cluster-based DIET DGP is sufficiently variable). *Assuming that the cluster-vectors form an affine generator system (Assums. [1C\)](#page-15-0), then the modulation parameter matrix* L *(defined in Assum. [6\)](#page-24-1) formed by the cluster vectors* v<sup>c</sup> − v<sup>a</sup> *has full column rank.*

*Proof.* First we need to show that the cluster vectors v<sup>c</sup> have the same role as θij (c). The derivative of the log-pdf of the [vMF](#page-30-1) distribution in [\(79\)](#page-24-2) w.r.t. z is the exponent in the DIET conditional (we can differentiate for non-normalized z, which is the case for auxiliary-bvariable ICA).

$$\frac{\partial}{\partial \mathbf{z}} [\kappa \langle \mathbf{v}_c, \mathbf{z} \rangle + \log C_d(\kappa)] = \kappa \mathbf{v}_c \quad (80)$$

Then, we need d + 1 cluster vectors to use one as a pivot to calculate L as defined in Assum. [6.](#page-24-1) By Lem. [1,](#page-15-7) this new set of vectors (i.e., offset by va, expressed as v<sup>c</sup> −va) also forms a generator system of R d , which implies that L has rank d, concluding the proof.

To apply [\(Hyvarinen et al., 2019,](#page-12-4) Thm. 3) to recover the identifiability result of [TCL,](#page-30-6) we need to show that our setting can solve the regression problem defined in Assum. [5.](#page-23-4) What we will show, w.l.o.g., is that our regression function akin to [\(Hyvarinen et al., 2019,](#page-12-4) (11)) does not have an auxiliary-variable dependent constant.

Lemma 3 (Regression function). *The regression function in [\(Hyvarinen et al., 2019,](#page-12-4) Thm. 3), which solves the multiclass classification problem, consists of three items: 1) a scalar product of vectorvalued functions of either* z *or* c*, and scalar-valued functions of 2)* z *and 3)* c*. Our DGP and neural network pipeline used for learning can also match this regression function, by choosing a pivot (zero) value for the* c*-dependent scalar function. This is without loss of generality.*

*Proof.* In Thm. [1,](#page-4-0) the identifiability of the cluster vectors is up to an affine transformation, where the bias is denoted by ψ. Calculating the scalar product of the learned cluster vector with the learned latents yields two terms:

- 1. a scalar product term between z− and c−dependent vectors; and
- 2. a ψ · h(z) term, which depends only on z

Comparing to [\(Hyvarinen et al., 2019,](#page-12-4) Eq. (11)), we see that a c−dependent scalar function is missing. Following the common practice in multinomial regression, we can, w.l.o.g., arbitrarily choose the pivot value of the c−dependent scalar function to be 0, thus we do not need that term. This yields the following expression for the regression function:

$$r(\mathbf{x}, c) = \mathbf{h}(\mathbf{x})^\top \tilde{\mathbf{w}}_c = \mathbf{h}(\mathbf{x})^\top (\mathcal{A}\mathbf{v}_c + \boldsymbol{\psi}) = \mathbf{h}(\mathbf{x})^\top \mathcal{A}\mathbf{v}_c + \mathbf{h}(\mathbf{x})^\top \boldsymbol{\psi} \quad (81)$$

$$= \mathbf{h}(\mathbf{x})^\top \mathcal{A} \mathbf{v}_c + a(\mathbf{x}), \quad (82)$$

where h is linear, so the first term depends on both x, c, the second term a(x) only on x, and we can choose (as usual practice in MLR), w.l.o.g., b(c) = 0, which concludes the proof.

#### B.3 THE RELATIONSHIP BETWEEN INFONCE AND DIET

Last, we show how DIET relates to InfoNCE, where we reframe InfoNCE in form of instance disrimination. InfoNCE optimizes the cross entropy between the true conditional of the underlying [DGP](#page-30-3) (a [vMF](#page-30-1) distribution) and the approximate conditional parametrized by an encoder network. This cross entropy can be formulated as a loss for an N−class classification problem, where N is the dataset size:

$$\mathcal{L} = \sum_{i=1}^B CE(q(\mathbf{x}_i^+), \mathbf{e}_i) \quad \text{s.t.} \quad q_k(\mathbf{x}_i^+) = \frac{\exp(\mathbf{f}(\mathbf{x}_i^+)^\top \mathbf{f}(\mathbf{x}_k))}{\sum_{b=1\dots B} \exp(\mathbf{f}(\mathbf{x}_i^+)^\top \mathbf{f}(\mathbf{x}_b))}, \quad (83)$$

where e<sup>i</sup> is the i th unit vector, encoding the class label in a one-hot fashion, and x + i denotes the positive pairs. Note that the last part is simply the standard softmax σ(.) over the innner product (f(x + i ) <sup>⊤</sup>f(xk)). To go from InfoNCE to DIET, we need to make the following modifications:

- 1. Sum over the whole dataset N, not just the batch B.

- 2. Replace the encoding of the anchor sample f(xk) with a learnable linear projection W, i.e., setting q(x + i ) = σ(W f(x + i ))

A remaining difference to the original DIET formulation is that InfoNCE assumes unit-normalized features. However, our theory (cf. Thm. [1C\)](#page-16-0) can accommodate unit-normalized vectors, so this is not a problem.

Let (xn, x + n ) be positive pair for sample n and let there be N samples in total. The InfoNCE loss is equivalent to a multi-class N−pair loss of the form:

$$\mathcal{L} = \sum_{i=1}^B CE(q(\mathbf{x}_i^+), \mathbf{e}_i) \quad \text{s.t.} \quad q_k(\mathbf{x}_i^+) = \frac{\exp(\mathbf{f}(\mathbf{x}_i^+)^\top \mathbf{f}(\mathbf{x}_k))}{\sum_{b=1\dots B} \exp(\mathbf{f}(\mathbf{x}_i^+)^\top \mathbf{f}(\mathbf{x}_b))}. \quad (84)$$

Now instead of having mini-batches of size B, we take the loss over the whole dataset:

$$\mathcal{L} = \sum_{i=1}^N CE(q(\mathbf{x}_i^+), \mathbf{e}_i) \quad \text{s.t.} \quad q_k(\mathbf{x}_i^+) = \frac{\exp(\mathbf{f}(\mathbf{x}_i^+)^\top \mathbf{f}(\mathbf{x}_k))}{\sum_{b=1\dots N} \exp(\mathbf{f}(\mathbf{x}_i^+)^\top \mathbf{f}(\mathbf{x}_b))}. \quad (85)$$

Next, replace f(xk) with a learnt and normalized weight vector wk:

$$\mathcal{L} = \sum_{i=1}^N CE(q(\mathbf{x}_i^+), \mathbf{e}_i) \quad \text{s.t.} \quad q_k(\mathbf{x}_i^+) = \frac{\exp(\mathbf{f}(\mathbf{x}_i^+)^\top \mathbf{w}_k)}{\sum_{b=1\dots N} \exp(\mathbf{f}(\mathbf{x}_i^+)^\top \mathbf{w}_b)}. \quad (86)$$

Note that the last part is simply the standard softmax σ(.) over a linear projection:

$$\mathcal{L} = \sum_{i=1}^N CE(q(\mathbf{x}_i^+), \mathbf{e}_i) \quad \text{s.t.} \quad q(\mathbf{x}_i^+) = \sigma(\mathbf{W}\mathbf{f}(\mathbf{x}_i^+)) \quad (87)$$

where W is the projection matrix for which the k th row corresponds to wk. Since i in this case corresponds to the sample index in the dataset, we recovered DIET up to normalization, and so W is simply the linear classifier.

# C ADDITIONAL EXPERIMENTAL DETAILS

#### C.1 SYNTHETIC DATA

The code is based on <https://brendel-group.github.io/cl-ica/>.

#### C.2 DISLIB

We evaluate our methods on the DisLib disentanglement benchmark [\(Locatello et al., 2019\)](#page-13-6), which provides a controlled setting for testing disentanglement and latent variable recovery. We used the version of the DisLib [\(Locatello et al., 2019\)](#page-13-6) dataset based on the GitHub repository from [\(Roth](#page-13-11) [et al., 2022\)](#page-13-11)[<sup>3</sup>](#page-26-3) . It includes the vision datasets dSprites, Shapes 3D, MPI 3D, Cars 3D, and smallNORB. Using Pytorch, we train both a three-layer MLP with 512 latent dimensions and BatchNorm (which helped with trainability) and a CNN (ResNet18) also with 512 latent dimensions [\(He et al., 2016\)](#page-11-13). We only consider latent variables with Euclidean topology, as non-Euclidean, e.g., periodic latent variables such as orientation, are problematic to learn and are potentially mapped to a nonlinear manifold [\(Higgins et al., 2018;](#page-11-7) [Pfau et al., 2020;](#page-13-8) [Keurti et al., 2023;](#page-12-11) [Engels et al., 2024\)](#page-11-6). We evaluate the recovery of latent variables by computing the Pearson correlation between ground-truth and predicted factors. Both models were trained for 100 epochs, with the Adam optimizer, a learning rate of 0.001 and a batch size of 4096.

# C.3 REAL DATA: IMAGENET-X

Finally, we test the generalizability of our theoretical insights on real-world data using ImageNet-X [\(Idrissi et al., 2022\)](#page-12-10). The latent variables are proxies, defined by human annotators [\(Idrissi et al.,](#page-12-10) [2022\)](#page-12-10). They are binary labels, representing the deviation of a certain latent variable on a given sample from the mode of that latent variable. We evaluate how well linear decoders can predict

<sup>3</sup><https://github.com/facebookresearch/disentangling-correlated-factors>

latent variables from pretrained model representations. We use two architectures, a ResNet50 (latent dimension d = 2048) and a Vit-b-16 (latent dimension d = 768) both trained on standard supervised classification using a cross-entropy loss on the full ImageNet dataset [\(Deng et al., 2009\)](#page-10-11). Moreover, to get a baseline decoding performance from inputs (like in the DisLib experiments), we also fix a random linear projection from the full 224 · 224 · 3 = 150, 528 ImageNet input dimensionality down to 2048 the ResNet50 latent dimensionality. This is purely for computational reasons and can be justified based on the Johnson–Lindenstrauss lemma[<sup>4</sup>](#page-27-3) .

We randomly split the data into 70% training and 30% testing data. For some latent variables, the label distribution was heavily imbalanced with less than 1% positive examples. To compensate class imbalance, for each latent variable, we resampled both training and testing data to achieve an even distribution. We repeat this and all following analysis averaged over 10 random seeds. Using the LogisticRegression module from sklearn[<sup>5</sup>](#page-27-4) , we fit a linear decoder to predict the latent variable. Finally, we compute p-values based on one sample t-tests against a null hypothesis of chance level (50%) accuracy with a multi-comparison Bonferroni adjusted significance level of α = 0.05 <sup>17</sup>·<sup>5</sup> < 0.0006 (17 factors and 5 models).

# D ADDITIONAL EXPERIMENTAL RESULTS

Ablating the choice of the cluster vectors. In Tab. [5,](#page-27-1) we present additional ablation studies exploring the effect of varying the distribution of the cluster vectors v<sup>c</sup> on the unit hyper-sphere. We do not observe any significant impact on the R<sup>2</sup> scores of more concentrated cluster centroids vc.

Table 5: Identifiability in the synthetic setup. Mean <sup>±</sup> standard deviation across 5 random seeds. Settings that match our theoretical assumptions are ✓. We report the R<sup>2</sup> score for linear mappings, z˜ → z and w<sup>i</sup> → v<sup>c</sup> for cases with normalized (o) and unormalized (a) w<sup>i</sup> . For unormalized w<sup>i</sup> , we verify that mappings z˜ → z are orthogonal by reporting the mean absolute error between their singular values and those of an orthogonal transformation.

|    | N | d | C   | p ( v   | c ) | p ( z | v c ) | M.   | z ˜ | →   | R z | 2 ( ↑ o w | ) i | →   | v c |   | w z ˜ | i → | cases MAE z | o w | ( ↓ i | ) → | v c |    | z ˜ | →   | R z | 2 ( ↑ a w | ) i | → w | i v c |
|----|---|---|-----|---------|-----|-------|-------|------|-----|-----|-----|-----------|-----|-----|-----|---|-------|-----|-------------|-----|-------|-----|-----|----|-----|-----|-----|-----------|-----|-----|-------|
| 10 | 3 | 5 | 100 | Uniform | vMF | ( κ   | = 10) | ✓ 98 | 6   | ± 0 | 01  | 99        | 9   | ± 0 | 01  | 0 | 01    | ± 0 | 00          | 0   | 00    | ± 0 | 00  | 99 | 0   | ± 0 | 00  | 99        | 9   | ± 0 | 00    |
| 10 | 3 | 5 | 100 | Laplace | vMF | ( κ   | = 10) | ✓ 98 | 7   | ± 0 | 00  | 99        | 5   | ± 0 | 00  | 0 | 01    | ± 0 | 00          | 0   | 00    | ± 0 | 00  | 99 | 1   | ± 0 | 00  | 99        | 8   | ± 0 | 00    |
| 10 | 3 | 5 | 100 | Normal  | vMF | ( κ   | = 10) | ✓ 98 | 2   | ± 0 | 01  | 99        | 2   | ± 0 | 01  | 0 | 01    | ± 0 | 00          | 0   | 00    | ± 0 | 00  | 99 | 2   | ± 0 | 00  | 99        | 8   | ± 0 | 00    |

Quantifying the violation of the assumption on the conditional with a generalized normal. Tabs. [2](#page-6-2) and [3](#page-7-2) show that using a Laplace conditional instead of a [vMF](#page-30-1) or normal distribution leads to substantially lower [R](#page-30-9)<sup>2</sup> scores, though one might argue that the Laplace distribution is not that different (according to some intuitive notion) from the [vMF](#page-30-1) or normal distributions. To understand why using a Laplace conditional leads to such a poor performance, we ran synthetic experiments with a generalized normal conditional with scale α (this is conceptually similar to our concentration parameter κ) and shape β:

$$\mathbf{z} \sim p(\mathbf{z}|C) \propto e^{\alpha \|\mathbf{v}_C - \mathbf{z}\|_\beta^\beta}, \text{ where } \|\mathbf{x}\|_\beta^\beta = \sum_{i=1}^d |x_i|^\beta. \quad (88)$$

Importantly, β = 1 gives a Laplace, whereas β = 2 gives a normal distribution. Thus, the generalized normal can be thought of as "interpolating" between these two distributions, providing the perfect testbed to determine when performance starts to break down. We show the [R](#page-30-9)<sup>2</sup> scores for both recovering z (Fig. [4](#page-28-1) Left) and v<sup>c</sup> (Fig. [4](#page-28-1) Middle) across multiple scale (α) and shape (β) values, averaged over 5 seeds. We also report the average representation norm across multiple scale (α) and

<sup>4</sup>[https://en.wikipedia.org/wiki/JohnsonLindenstrauss\\_lemma](https://en.wikipedia.org/wiki/Johnson–Lindenstrauss_lemma)

<sup>5</sup>[https://scikit-learn.org/1.5/modules/generated/sklearn.linear\\_model.](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html) [LogisticRegression.html](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html)

![](_page_28_Figure_1.jpeg)

Figure 4: Quantifying the assumption violation of a Laplace conditional: Tabs. [2](#page-6-2) and [3](#page-7-2) show that using a Laplace conditional leads to substantially lower [R](#page-30-9)<sup>2</sup> scores. Numbers are averages across 5 seeds. (Left and Middle:) using a generalized normal distribution to "interpolate" between a normal (β = 2) and a Laplace (β = 1) distribution for different scale values (denoted as α, which is conceptually akin to our concentration κ) and show the [R](#page-30-9)<sup>2</sup> score for recovering z (Left) and v<sup>c</sup> (Middle). (Right:) The average norm of the representation for the one-dimensional case for different β values. As β approaches 1, the average norm increases, indicating a larger spread

shape (β) values (calculated in the one-dimensional case and averaged over 5 seeds) and the crucial effect of having a fat tail (Fig. [5](#page-29-1) Right) with a truncated Laplace distribution. Our results indicate that:

- 1. v<sup>c</sup> is easier to recover than z: the numbers are higher in Fig. [4\(](#page-28-1)Middle) than in Fig. [4\(](#page-28-1)Left)
- 2. More concentrated conditionals degrade identifiability for all shapes: [R](#page-30-9)<sup>2</sup> scores decrease as α increases in both Fig. [4\(](#page-28-1)Left, Middle)
- 3. The average representation norm increases with increasing scale and decreasing shape: as the conditional approaches the Laplace distribution β → 1, the samples have a larger norm, i.e., they are further away from the unit hypersphere, potentially leading to insufficient overlap Fig. [4\(](#page-28-1)Right)
- 4. Fat tails worsen identifiability performance: Allow more and more of the tail of a Laplace distribution to be included in the support (truncated symmetrically between −1, 1 and −3, 3 shows a strong anti-correlation with the [R](#page-30-9)<sup>2</sup> score for multiple scales (α).

Loss saturation: the role of batch size and latent dimensionality. Our results in Tabs. [2](#page-6-2) and [3](#page-7-2) show that increasing latent dimensionality leads to substantially lower [R](#page-30-9)<sup>2</sup> scores—in line with the findings in many prior works on the identifiability of SSL methods [\(Zimmermann et al., 2021;](#page-14-1) [von](#page-14-2) [Kugelgen et al., 2021;](#page-14-2) [Rusak et al., 2024\)](#page-14-3). In general, the issue of extracting large dimensional ¨ representations from practical, real-world datasets is an open question [\(Simon et al., 2023;](#page-14-10) [Jing et al.,](#page-12-14) [2022\)](#page-12-14). We investigate the interaction between batch size and concentration in Fig. [5\(](#page-29-1)Left).

With increasing concentrations, intra-cluster samples become more indistinguishable. This means that achieving close to optimum instance discrimination loss is easy with relatively coarse-grain features. The intuition is that most SSL objectives (including DIET) saturate, i.e.. get close to optimum as the underlying pretext problem (in our cases the instance discrimination) is nearly solved—a very good example is learning only the content features (but not the style ones) in [von Kugelgen et al.](#page-14-2) [\(2021\)](#page-14-2). ¨ The result is a population gradient with a very low norm. As the empirical loss is calculated based on a finite batch size, the variance of the gradient overtakes the norm, and the training effectively stalls. Additionally, higher concentrations result in less overlap between classes, which can have a detrimental effect on source recovery. However, the signal-to-noise ratio improves with a larger batch size, as the increasing [R](#page-30-9)<sup>2</sup> scores show in Fig. [5\(](#page-29-1)Left) in the columns from left to right. The results

![](_page_29_Figure_1.jpeg)

Figure 5: The role of batch size, number of clusters, and fat tails for identifiability: (Left:) Increasing batch size improves [R](#page-30-9)<sup>2</sup> scores, counteracting the detrimental effect of more concentrated (higher κ) conditionals; (Middle:) More clusters improve [R](#page-30-9)<sup>2</sup> scores, counteracting the detrimental effect higher dimensional representations (10 clusters for 10 and 20 dimensions violate Assum. [2;](#page-15-8) thus, the low [R](#page-30-9)<sup>2</sup> score); (Right:) The Laplace distribution leads to low [R](#page-30-9)<sup>2</sup> scores due to its fat tails. Experiments with a truncated Laplace conditional (where the support is restricted to [−Truncation; Truncation]) shows that the closer the truncated Laplace distribution is to the Laplace distribution (i.e., with increasing Truncation), [R](#page-30-9)<sup>2</sup> scores decrease for all tested scales α. Averages and error bars are reported across 5 seeds

suggest that this issue can, at least theoretically, be mitigated—note that identifiability results hold in the infinite-sample regime, so requiring a larger batch size does not contradict our results.

When the latent dimensionality increases, the saturating loss implies that relatively low-dimensional features are sufficient to achieve this near-optimum loss. This phenomenon can also be seen in Fig. [5\(](#page-29-1)Middle), where for each column (i.e., a fixed number of clusters), the [R](#page-30-9)<sup>2</sup> score deteriorates with increasing latent dimensionality.

It is important to mention that none of these contradict our theory, which holds for the global optimizer of the DIET population loss. In practice, the additional challenges of estimation error (due to finite sample size and finite batch size) and algorithmic error (using GD-based methods to solve a likely non-convex problem) may impose adverse effects on the evaluation.

Diversity: the role of the number of clusters. To investigate the role of Assum. [2,](#page-15-8) we investigate how the number of clusters affects the [R](#page-30-9)<sup>2</sup> score. Though the number of clusters is intrinsic to the dataset, thus, it cannot be chosen arbitrarily, knowing its effect on performance can inform practitioners about potential failure cases. We ablated the number of clusters for different latent dimensionalities (Fig. [5\(](#page-29-1)Middle)). For a given dimensionality, the [R](#page-30-9)<sup>2</sup> score improves with more clusters, although there seems to be a sweet spot where a further increase in the number of clusters only marginally improves the [R](#page-30-9)<sup>2</sup> score. We also see that our requirement of d + 1 clusters (affine generator systems of R <sup>d</sup> have at least this many members) is essential for good performance. This is reflected in the extremely poor [R](#page-30-9)<sup>2</sup> scores for n ∈ {10, 20}, when the number of clusters is only 10 (first column from the left in Fig. [5\(](#page-29-1)Middle)).

Robustness to label noise. In this section, we evaluate the robustness of our method under increasing label noise. For DIET (Fig. [6](#page-30-11) Left) we perturbed the instance label for each sample with a probability equal to the label noise ratio (x-axis in the figure). The perturbed labels were drawn uniformly from the set of all instance labels. For the supervised case (Fig. [6](#page-30-11) Right), the cluster label was perturbed instead. In both cases, the y-axis represents the identifiability score ([R](#page-30-9)<sup>2</sup> score).

We believe this setup reflects realistic scenarios, where label noise is equally likely to affect any data point as commonly assumed in the literature [\(Nettleton et al., 2010;](#page-13-12) [Frenay & Verleysen, 2013\)](#page-11-14). ´

Despite this increasingly challenging setup, the results demonstrate remarkable robustness to label noise. Up to an 80% label noise ratio, latent recovery shows minimal degradation, and the cluster recovery performance of DIET remains perfect—though both metric substantially decrease for a larger (100%) ratio. We attribute this robustness to the symmetry of label noise across all instances and labels. While increased uncertainty does reduce the accuracy of individual label predictions, the

![](_page_30_Figure_1.jpeg)

Figure 6: The robustness of DIET and supervised classification to label noise: The x−axis shows the proportion of instances with perturbed labels; the y−axis the [R](#page-30-9)<sup>2</sup> score of learning the ground truth latents (and the cluster vectors on the left). Perturbed labels are uniformly resampled from the whole instance label (left) or cluster label (right) sets, respectively. (Left:) DIET perfectly recovers the cluster vectors v<sup>c</sup> up to 80% label noise, and shows only a small degradation for the latents z; (Right:) Supervised classification robustly recovers the latents up to 80% label noise with only a small degradation. Averages and error bars are reported across 5 seeds.

optimal logit values predicted by the encoder shouldn't change under symmetrical label noise. The stability of deep learning models to label noise has also been shown by [Rolnick](#page-13-13) [\(2017\)](#page-13-13).

# E ACRONYMS

DGP data generating process GCL Generalized Contrastive Learning ICA Independent Component Analysis LVM latent variable model MAE Mean Absolute Error PID parametric instance discrimination RV random variable SSL self-supervised learning TCL Time-Contrastive Learning vMF von Mises-Fisher

# F NOMENCLATURE

R<sup>2</sup>

coefficient of determination S hypersphere