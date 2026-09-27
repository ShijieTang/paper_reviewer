# Cross-Entropy Is All You Need To Invert The Data Generating Process

Patrik Reizinger∗1, Alice Bizeul∗1,2**, Attila Juhos**∗1, Julia E. Vogt2, Randall Balestriero3, Wieland Brendel1, David Klindt4 {patrik.reizinger, attila.juhos, wieland.brendel}@tuebingen.mpg.de {alice.bizeul, julia.vogt}@inf.ethz.ch, rbalestr@brown.edu, klindt@cshl.edu

## Abstract

Supervised learning has become a cornerstone of modern machine learning, yet a comprehensive theory explaining its effectiveness remains elusive. Empirical phenomena, such as neural analogy-making and the linear representation hypothesis, suggest that supervised models can learn interpretable factors of variation in a linear fashion. Recent advances in self-supervised learning, particularly nonlinear Independent Component Analysis, have shown that these methods can recover latent structures by inverting the data generating process. We extend these identifiability results to parametric instance discrimination, then show how insights transfer to the ubiquitous setting of supervised learning with cross-entropy minimization.

We prove that even in standard classification tasks, models learn representations of ground-truth factors of variation up to a linear transformation under a certain DGP. We corroborate our theoretical contribution with a series of empirical studies. First, using simulated data matching our theoretical assumptions, we demonstrate successful disentanglement of latent factors. Second, we show that on DisLib, a widely-used disentanglement benchmark, simple classification tasks recover latent structures up to linear transformations. Finally, we reveal that models trained on ImageNet encode representations that permit linear decoding of proxy factors of variation. Together, our theoretical findings and experiments offer a compelling explanation for recent observations of linear representations, such as superposition in neural networks. This work takes a significant step toward a cohesive theory that accounts for the unreasonable effectiveness of supervised learning.

## 1 Introduction

Representation learning is a central task in machine learning, underpinning the success of extracting and encoding meaningful information from data (Bengio et al., 2013). Among the various paradigms, supervised learning—particularly classification tasks using cross-entropy minimization—has become the dominant method in deep learning (Krizhevsky et al., 2012). Despite its simplicity, this form of supervised learning has led to several intriguing and widely-observed phenomena, including: neural analogy making (Mikolov et al., 2013), where models seemingly map between related concepts; the *linear representation hypothesis* (Park et al., 2023), which posits that interpretable features can be linearly decoded from neural representations; recent work on *superposition* in neural networks (Elhage et al., 2022), showing evidence that interpretable features are linearly represented in neural activations (Templeton et al., 2024); and the success of *transfer learning* (Donahue et al., 2014),
where a linear readout can be trained on top of learned representations to solve new tasks. These phenomena suggest that deep learning models encode various features in a manner that allows for linear decoding. Yet, a comprehensive theory that explains why these properties emerge in deep learning models has remained elusive (Arora et al., 2016; Park et al., 2023). We address this gap by building on the theory of Independent Component Analysis (ICA), which studies the conditions under which latent variables in probabilistic models can be uniquely identified (Comon, 1994; Hyvarinen et al., 2001). Recently, ICA has been extended to nonlinear models (Hyvarinen et al., 2023), providing a theoretical foundation for recovering latent variables in a broad ¨ class of machine learning tasks (Hyvarinen & Morioka, 2016; Hyvarinen et al., 2019; Gresele et al., 2019; Khemakhem et al., 2020a; Klindt et al., 2021; Khemakhem et al., 2020b; Locatello et al., 2020;
∗Joint first authorship; 1Max Planck Institute for Intelligent Systems, Tubingen AI Center, ELLIS Institute, ¨
Tubingen, Germany; ¨
2Department of Computer Science, ETH Zurich and ETH AI Center, ETH Z ¨ urich, Z ¨ urich, ¨
Switzerland; 3Department of Computer Science, Brown University, Rhode Island, USA; 4Cold Spring Harbor Laboratory, Cold Spring Harbor, New York, USA;
1 Morioka et al., 2021; Halv ¨ a et al., 2021; Morioka & Hyvarinen, 2023). Most of these advances have ¨ focused on self-supervised learning (SSL) (Hyvarinen & Morioka, 2016; Hyvarinen et al., 2019; Zimmermann et al., 2021; von Kugelgen et al., 2021; Rusak et al., 2024), i.e., when neural networks ¨ are trained by solving a surrogate (classification) task to learn from unlabeled data—the exceptions that study supervised learning, though either in the multitask setting, or with a single task with additional assumptions, include (Ahuja et al., 2022; Lachapelle et al., 2023; Fumero et al., 2023). However, we seek to understand whether similar identifiability guarantees can explain under what conditions cross-entropy-based supervised learning, i.e., when the labels for the classification task are provided in the dataset, recovers interpretable and transferable representations. Our journey starts with a recent development in SSL: nonlinear ICA has been shown to provide identifiability guarantees in contrastive learning, where models invert the data generating process (DGP) and recover latent variables up to linear transformations (Hyvarinen et al., 2019; Zimmermann et al., 2021). Building on this insight, we first extend nonlinear ICA to a simple form of SSL—i.e., parametric instance discrimination (PID) (Dosovitskiy et al., 2014)—through the DIET method (Ibrahim et al., 2024), which streamlines the auxiliary task into an instance-discrimination paradigm. We model the DGP in a new, cluster-centric way, and show that DIET's learned representation is linearly related to the ground-truth representation. From this foundation, we take the crucial step of extending the theoretical framework to the more common paradigm of supervised learning. Specifically, we show that models can recover groundtruth latent variables up to a linear transformation even in standard classification tasks using the cross-entropy loss, which is the most prevalent setting in modern machine learning. By doing so, we aim to explain why deep learning, particularly supervised classification, is so effective in learning interpretable and transferable representations, offering a unifying framework to explain phenomena such as linear representations and neural analogy-making. Thus, our theoretical insights offer a potential explanation for the extraordinary success of supervised deep learning across a wide variety of tasks. Our **contributions** are
- We propose a cluster-centric DGP as a model for the parametric instance discrimination method of Ibrahim et al. (2024) and prove the DGP's linear identifiability (Thm. 1);
- We use our insight to extend the identifiability guarantee to standard cross-entropy-based supervised classification under the a cluster-centric DGP (Thm. 2);
- We provide a "genealogy" of cross-entropy-based classification methods to connect our identifiability results in instance discrimination and supervised classification to auxiliary-variable nonlinear Independent Component Analysis (ICA) (Hyvarinen et al., 2019) and self-supervised learning (SSL) (§ 3.4) (Zimmermann et al., 2021);
- We corroborate our findings in synthetic experiments matching our cluster-centric DGP, the DisLib disentanglement benchmark (Locatello et al., 2019), and real-world ImageNet-X data (Idrissi et al., 2022), showing that the cross-entropy loss, irrespective of the meaningfulness of labels, can lead to linear identifiability of the features (§ 4).

## 2 Background

Empirical evidence of a linear latent representation. The *linear representation hypothesis* (Park et al., 2023) has lately received a lot of attention. A weak version of this hypothesis could mean that there are directions in neural activation space that correspond to interpretable features. In the case of *neural analogy making*, Mikolov et al. (2013) showed that there exist directions in word embeddings that are interpretable and preserved across input pairs. As an example for encoder f, producing latent variables z, the direction z = f(man)−f(*woman*) seems to correspond to gender and can be added to other words such as f(*king*) + z ≈ f(*queen*). Several datasets, such as the Google Analogy Dataset (GA) (Mikolov, 2013) and BATS (Drozd et al., 2016), have been developed to evaluate neural analogy-making. These were, for instance, evaluated in (Dufter & Schutze, 2019). ¨ Theoretical explanations of linear representations have been proposed for word embeddings by Arora et al. (2016) and Allen & Hospedales (2019). Both approaches take a statistical learning theory perspective and focus on characterizing the pointwise mutual information. They do not consider cross-entropy-based classification; and, thus, do not make a connection to supervised classification, as we do in Thm. 2. Park et al. (2023) provide a framework to specify what exactly is meant by the linear representation hypothesis. They also provide a strong, causal hypothesis where finding that a feature is linearly represented does not imply that an intervention on that linear subspace will causally remove the feature from the model output. Engels et al. (2024) point out that some latent representations are not linear. This makes intuitive sense if we consider that some latent features, such as the pose of an object, have a non-Euclidean topology that will have to be embedded on a curved manifold in a linear subspace of the latent representation (Higgins et al., 2018; Pfau et al., 2020; Keurti et al., 2023). For instance, the quadrature pair of sines and cosines representing rotations in a 2D subspace in (Klindt et al., 2021, Fig. 15) depends on the object symmetries (Bouchacourt et al., 2021). Roeder et al. (2020) prove that different models trained with a discriminative objective converge to learning the same latent representation. Importantly, their claim is about the linear relationship between any two learned representations, and not the learned and the ground-truth one, as is usually the case in identifiability theory (Hyvarinen et al., 2001). They also show this empirically for pairs of models trained on different datasets. Their results are corroborated even with widely varying training factors by Moschella et al. (2023). These findings are also supported by recent large scale empirical studies in the converging representations of vision models (Chen & Bonner, 2023). This could also explain the recently proposed *platonic representation hypothesis* (Huh et al., 2024) about the convergence of representations, the improved disentanglement across model families (Du & Xiang, 2021), and the better identifiability of biological mechanisms (Genkin & Engel, 2020). However, these insights from the literature fail to connect the linearity of learned representations to the identifiability of the assumed ground-truth DGP—this is the gap our contribution aims to address. Identifiable weakly-/self-supervised learning and ICA. Independent Component Analysis (ICA) theory studies the conditions under which latent variables in probabilistic models can be uniquely identified (Comon, 1994; Hyvarinen et al., 2001). *Identifiability* means that the learned representation, at the global optimum of the training loss, relates to the *ground-truth* representation (i.e., the groundtruth latent variables underlying the data) via a "simple" transformation, such as permutations or elementwise invertible transformations—this is different to investigations relating two instances of learned representations, such as in Roeder et al. (2020); Moschella et al. (2023); Zhang et al. (2023). Recently, ICA has been extended to nonlinear models (Hyvarinen et al., 2023), providing a theoretical ¨ foundation for recovering latent variables in a broad class of learning tasks (Hyvarinen & Morioka, 2016; Hyvarinen et al., 2019; Gresele et al., 2019; Khemakhem et al., 2020a; Klindt et al., 2021; Khemakhem et al., 2020b; Locatello et al., 2020; Morioka et al., 2021; Halv ¨ a et al., 2021; Morioka ¨
& Hyvarinen, 2023). Most of these advances have focused on SSL, (Hyvarinen & Morioka, 2016; Hyvarinen et al., 2019; Zimmermann et al., 2021; von Kugelgen et al., 2021; Rusak et al., 2024). ¨

## 3 Theory

This section presents our main theoretical contribution. We start with our motivation to understand self-supervised learning (SSL) with the help of the simplified DIET (Ibrahim et al., 2024) algorithmic pipeline. For this, we propose a cluster-centric data generating process (DGP) that can model semantic classes (§ 3.1). Then we state our main result in § 3.2 and discuss an intuition behind the identifiability of the representation learned by DIET. We conclude by investigating how DIET fits into the vast literature of (identifiable) SSL and auxiliary-variable Independent Component Analysis (ICA) methods (§ 3.4). This leads to a significant result for proving the identifiability of the latents learned via supervised classification under the DIET DGP (§ 3.3). We provide the technical details for Generalized Contrastive Learning (GCL) (Hyvarinen et al., 2019) in Appx. B.1 and InfoNCE (Chen et al., 2020; Zimmermann et al., 2021) in Appx. B.3. Motivation. Despite significant theoretical progress (Zimmermann et al., 2021; von Kugelgen et al., ¨ 2021; Rusak et al., 2024), it remains elusive why SSL methods work well in practice. Rusak et al. (2024) highlighted two remaining gaps between theory and practice: 1) practitioners often discard the encoder's last few layers (termed the projector) for better performance, despite identifiability guarantees not reflecting this fact; and 2) the data is presumably clustered, not reflected in the common assumption of a uniform marginal. Despite a similar terminology in auxiliary-variable nonlinear ICA algorithms, such as Time-Contrastive Learning (TCL) (Hyvarinen & Morioka, 2016) or GCL (Hyvarinen et al., 2019), it is unclear how such methods relate to SSL at large. Interestingly, the identifiability proofs for nonlinear ICA partition the model into a separate encoder and a regression function (Hyvarinen & Morioka, 2016; Hyvarinen et al., 2019) and prove identifiability for the latent variables after the encoder, but before the regression function. This aligns with the practice of discarding the projector in SSL (Bordes et al., 2023), though identifiability results do not reflect this fact (Zimmermann et al., 2021; von Kugelgen et al., 2021; Rusak et al., 2024). These observations ¨ served as our motivation to investigate How can we extend the identifiability guarantees to more realistic self-supervised classification scenarios, and can we apply these insights to improve our understanding of supervised learning?

Results overview. We aim to advance our theoretical understanding of SSL, for this, we use the recently proposed DIET (Ibrahim et al., 2024) (detailed in § 3.1), which, beyond its simplicity, promises the strongest and most realistic results, based on similarities to GCL (Hyvarinen et al., 2019). Namely, DIET uses a separate encoder and classification head, and solves an auxiliary classification task akin to GCL—furthermore, its loss correlates with downstream performance, a non-obvious and welcome fact (Rusak et al., 2024). This provides the hope to resolve the two above points by modeling the cluster structure of the data and proving identifiability for the representation used for downstream tasks (Thm. 1). Subsequently, we leverage the insights from our identifiability theory and the DIET pipeline's similarity to *supervised* classification to show how the latter is a special case of DIET, where the sample indices correspond to the semantic class labels (Thm. 2).

3.1 SETUP DIET (Ibrahim et al., 2024). DIET solves an instance classification problem, where each sample x in the training dataset of size N has a unique instance label i. Augmentations do not affect this label. We have a composite model W ◦ f, where the backbone f produces d-dimensional representations, and a linear, bias-free classification head W ∈ R
N×d maps these representations to a logit vector equal in size to the cardinality of the training dataset. If the parameter vector corresponding to logit i is denoted as wi, then W effectively computes similarity scores (scalar products) between the wi's and embeddings f(x). DIET trains this architecture to predict the correct instance label using multinomial regression (with f,W and temperature β as learnable variables), i.e., it solves a parametric instance discrimination (PID) task (Dosovitskiy et al., 2014; Wu et al., 2018):

$${\mathcal{L}}_{\mathrm{PID}}(\mathbf{f},\mathbf{W},\beta)=\mathbb{E}_{(\mathbf{x},i)}\bigg[-\ln\frac{e^{\beta\langle\mathbf{w}_{i},\mathbf{f}(\mathbf{x})\rangle}}{\sum_{j}e^{\beta\langle\mathbf{w}_{j},\mathbf{f}(\mathbf{x})\rangle}}\bigg].$$

$$(1)$$
. (1)
An important fact is that (1) is the cross-entropy loss with instance labels, which we will leverage to connect instance discrimination to supervised classification. The proposed cluster-centric **data generating process (DGP).** To prove the identifiability of the latent variables, we need to formally define a latent variable model (LVM) for the data generating process (DGP). We take a cluster-centric approach, representing semantic classes by cluster vectors, similar to proxy-based metric learning (Kirchhof et al., 2022). Then, we model the samples of a class with a von Mises-Fisher (vMF) distribution (intuitively, this is an isotropic multivariate Normal distribution that is restricted to the unit hypershere), centered around the class's cluster vector. This conditional distribution jointly models intra-class sample selection and *augmentations* of samples, together called *intra-class variances*. In contrast to conventional SSL methods such as InfoNCE (Zimmermann et al., 2021), this conceptually separates global and local structure in the latent space: 1) the cluster-vectors describe the global structure of the latent space; and 2) the clustercentric conditional in (2) describes the local structure. This cluster-centric conditional embodies that data augmentations are selected such that they ought not to change the sample's semantic class. Our conditional does not mean that each sample pair transforms into each other via augmentations with high probability. It does mean that—since we assume a latent variable model (LVM) on the hypersphere; i.e., all semantic concepts (color, position, etc.) correspond to a continuous latent variable—the latent manifold is connected, or equivalently, that the augmentation graph is connected, which is an assumption used in (Wang et al., 2022; Balestriero & LeCun, 2022; HaoChen et al., 2022). We provide an overview of our assumptions, and defer additional details to Assums. 1C in Appx. A: Assumptions 1 (DGP with vMF samples around cluster vectors. *Simplified.*).

(i) There is a finite set of semantic classes C , represented by a set of unit-norm d*-dimensional* cluster-vectors {vc|c ∈ C } ⊆ S
d−1. The system {vc} is sufficiently large and spread out.

(ii) Any instance label i *belongs to exactly one class* c = C(i).

(iii) *The latent variable* z ∈ S
d−1 of our data sample with instance label i is drawn from a vMF
distribution with concentration parameter κ around the cluster vector vc *of class* c = C(i):

$$z\sim p(z|c)\propto e^{\kappa\langle\mathbf{v}_{c},\mathbf{z}\rangle}.$$
$\eqref{eq:walpha}$. 
κ⟨vc,z⟩. (2)
(iv) Sample x is generated by passing latent z *through an injective generator function:* x = g(z).

3.2 MAIN RESULT: DIET IDENTIFIES BOTH LATENT VARIABLES AND CLUSTER VECTORS Under Assums. 1, we prove the identifiability of both the latent representations z and the cluster vectors, vc, in all four combinations of unit-normalized (i.e., when the latent space is the hypersphere, commonly used, e.g., in InfoNCE (Chen et al., 2020)); and non-normalized (as in the original DIET
paper (Ibrahim et al., 2024)) learned embeddings, z˜, and weight vectors, wi. We state a concise version of our result and defer the full treatment and the proof to Thm. 1C in Appx. A:
Theorem 1 (Identifiability of latent variables drawn from vMF around cluster vectors. Simplified.). Let (f,W, β) *globally minimize the DIET objective* (1) *under the following additional constraints:* C3. the embeddings f(x) are unnormalized, while the wi's are unit-normalized. Then wiidentifies the cluster vector vC(i) up to an orthogonal linear transformation O: wi = OvC(i), for any i.

Furthermore, the inferred latent variables z˜ = f(x) *identify the ground-truth latent variables* z up to a scaled orthogonal transformation with the same O: z =
κ βOz˜.

C4. neither the embeddings f(x) nor the wi's are unit-normalized. Then wi*identifies the cluster* vectors vc *up to an affine linear transformation. Furthermore, the inferred latent variables* z˜
identify the ground-truth latent variables z *up to a linear transformation.*
In all cases, the weight vectors belonging to samples of the same class are equal, i.e., for any *i, j*,
C(i) = C(j) *implies* wi = wj .

Intuition. DIET assigns a different (instance) label and a unique weight vector wito each training sample. The cross-entropy objective is optimized if the trained neural network can distinguish between the samples. Thus, the learned representation z˜ = f(x) should capture enough information to distinguish different samples, even from the same class. However, the weight vectors wi's cannot be sensitive to the intra-class sample variance or the sample's instance label i (because the conditional distribution over latent variables is identical for all samples of the same class). This leads to the weight vectors taking the values of the cluster vectors. As cluster vectors only capture some statistics of the conditional (1), feature recovery is more fine-grained than cluster identifiability. The interaction between the two is dictated by the cross-entropy loss, which is minimized if the representation z˜ is most similar to its own assigned weight vector wi. Fig. 1 provides a visualization conveying the intuition behind Thm. 1.

## 3.3 Supervised Classification

This section relates our cluster-centric DGP to *supervised* classification. To see how supervised machine learning is a special case of self-supervised approaches, consider that the sample index (i.e., the target of the cross-entropy loss) can be defined *arbitrarily* (as long as Assums. 1 are still satisfied). This means that many labelings are possible, including the one used for supervised classification. This, in hindsight obvious insight has important consequences: it can explain the success of supervised cross-entropy-based classification. Namely, supervised learning performs non-linear ICA under our proposed DGP (Assums. 1). We demonstrate this in §§ 4.1 and 4.3. We state a concise version of our result and defer the full treatment to Appx. A: Theorem 2 (Identifiability of latent variables drawn from a vMF around class vectors). Let Assum. 3 hold, and suppose that a continuous encoder f : R
D → R
d, a linear classifier W *with rows*
{w⊤
c| c ∈ C }, and β > 0 *globally minimize the cross-entropy objective:*

$$\mathcal{L}_{\mathrm{supervised}}(\mathbf{f},\mathbf{W},\beta)=\mathbb{E}_{(\mathbf{x},C)}\left[-\ln\frac{e^{\beta\langle\mathbf{w}_{C},\mathbf{f}(\mathbf{x})\rangle}}{\sum_{c^{\prime}\in\mathcal{C}}e^{\beta\langle\mathbf{w}_{c^{\prime}},\mathbf{f}(\mathbf{x})\rangle}}\right],$$
.
Then, the composition h = f ◦ g *is a linear map from* S
d−1to R
d.

Supervised
(a) (b)
(d)
GCL TCL DIET
(c)
InfoNCE
Figure 2: **The simplified genealogy of cross-entropy-based classification methods** (cf. Tab. 1 for details): The labeled arrows express how to go from general to special methods. (a) The most general auxiliary-variable ICA framework, Generalized Contrastive Learning (GCL) (Hyvarinen et al., 2019),
yields Time-Contrastive Learning (TCL) (Hyvarinen & Morioka, 2016) as the special case when the latent conditional is assumed to come from an exponential family (of order one) with a scalar auxiliary variable; (b) TCL relates to non-unit-normalized DIET by further restricting the latent conditional to a vMF distribution; (c) if the neural network used in InfoNCE is partitioned into a linear classifier head and a backbone, the marginal is assumed to be a vMF instead of uniform, we get the unit-normalized version of DIET; (d) if the labeling function in DIET is assumed to assign the semantic class labels to the samples, we get classic supervised training Intuition: In the context of DIET, the cross-entropy objective encourages the learned representations to align with the cluster vectors corresponding to each class. The identifiability of the latent variables is ensured by the fact that the cluster structure reflects the underlying data distribution, modeled as a vMF distribution. This leads to a representation that captures the latent structure up to an orthogonal transformation. *Given the same underlying structure as in DIET, supervised learning can* be viewed as a special case of instance discrimination, where the instance labels are replaced by class labels. The cross-entropy objective, when applied to classification tasks and assuming our DGP from Assums. 1, similarly encourages representations to align with class vectors. As a result, the latent variables are recovered up to a linear transformation, providing a theoretical explanation for the success of supervised classification in learning linearly decodable representations. 3.4 THE GENEALOGY OF IDENTIFIABLE CLASSIFICATION WITH CROSS-ENTROPY
Our main result in Thm. 1, and its corollary for supervised classification (Thm. 2) suggest the following surprising conclusion to invert the proposed DGP (Assums. 1):
Solving an (almost) arbitrary classification task by optimizing the cross-entropy objective is sufficient to invert the DGP *and identify the ground-truth representation up to a linear transformation.*
To show how solving a cross-entropy-based classification task is a key component to invert the DGP and to achieve linear identifiability, we provide a unified treatment of auxiliary-variable ICA (i.e., weakly supervised or self-supervised classification) and supervised classification methods. We call this a *genealogy* to allude to the fact that these methods can be seen as special cases, descending from each other (cf. Fig. 2 and Tab. 1 for an overview, and Appx. B for details).

## From Gcl To Tcl (Fig. 2A: Ar-

bitrary scalar labels and exponential family latent variables). The most general framework we consider is Generalized Contrastive Learning (GCL) (Hyvarinen et al., 2019), i.e., auxiliary-variable nonlinear ICA. GCL works with conditionally independent latent variables in Euclidean space given (possibly vector-valued) auxiliary information u. It aims to classify different values of u by distinguishing (x, u) from (x, u
∗), where u
∗is an arbitrary value of the auxiliary variable. At the Bayes optimum of the cross-entropy loss, GCL provides identifiability of the latent variables after the encoder f, but before the classifier head W, up to elementwise invertible transformations. When the latent variables are distributed Table 1: **Comparison of the components of different** cross-entropy-based classification methods: u denotes a (possibly) vector-valued auxiliary variable, t is the scalar time step, i the sample index, and c the semantic class; Exp-
Fam stands for exponential family, ⊥u for conditionally independent sources given the auxiliary variable, W is the classifier head, f the encoder, whereas N/A stands for no assumption Property GCL TCL InfoNCE DIET Supervised

```
Latent space R
                d R
                         
                         d S
                                 d−1 R
                                         
                                         d/S
                                             d−1 R
                                                       
                                                       d
Network W ◦ f W ◦ f f W ◦ f W ◦ f
Aux.info u t i i c
Conditional ⊥
                u ExpFam vMF vMF vMF
Marginal N/A N/A uniform uniform uniform

```

Table 2: Identifiability results for parametric instance discrimination (PID) **in numerical** simulations: Mean ± standard deviation across 5 random seeds. Settings that match and violate our theoretical assumptions are denoted as ✓ and ✗, respectively. We report the R2score for linear maps z˜ → z and wi → vc with normalized (subscript o) and not normalized (subscript a) wi. For normalized wi, we verify that the z˜ → z maps are orthogonal by reporting the Mean Absolute Error
(MAE) between their singular values and those of an orthogonal transformation.

| normalized wi   | unnormalized wi   |                        |                                           |                                           |                                         |                     |        |         |        |         |
|-----------------|-------------------|------------------------|-------------------------------------------|-------------------------------------------|-----------------------------------------|---------------------|--------|---------|--------|---------|
| R2 o (↑)        | MAEo(↓)           | R2 a (↑)               |                                           |                                           |                                         |                     |        |         |        |         |
| N               | d                 | |C |                   | p(z|vc)                                   | M.                                        | z˜ → z                                  | wi → vc             | z˜ → z | wi → vc | z˜ → z | wi → vc |
| 103             | 5                 | 100                    | vMF(κ= 10)                                | ✓ 98.6±0.01 99.9±0.00 0.01±0.00 0.00±0.00 | 99.0±0.00 99.9±0.00                     |                     |        |         |        |         |
| 105             | 5                 | 100                    | vMF(κ= 10)                                | ✓ 98.2±0.01 99.5±0.00 0.00±0.00 0.00±0.00 | 99.7±0.00 99.8±0.00                     |                     |        |         |        |         |
| 103             | 5                 | 100                    | vMF(κ= 10)                                | ✓ 98.6±0.01 99.9±0.00 0.01±0.00 0.00±0.00 | 99.0±0.00 99.9±0.00                     |                     |        |         |        |         |
| 103 10          | 100               | vMF(κ= 10)             | ✓ 92.5±0.01 99.6±0.00 0.01±0.00 0.00±0.00 | 93.0±0.03 99.6±0.00                       |                                         |                     |        |         |        |         |
| 103 20          | 100               | vMF(κ= 10)             | ✓ 70.8±0.02 97.1±0.01 0.03±0.00 0.00±0.00 | 81.9±0.01 99.7±0.00                       |                                         |                     |        |         |        |         |
| 103             | 5                 | 10                     | vMF(κ= 10)                                | ✓ 88.6±0.05 85.7±0.15 0.02±0.00 0.00±0.00 | 90.0±0.05 99.0±0.03                     |                     |        |         |        |         |
| 103             | 5                 | 100                    | vMF(κ= 10)                                | ✓ 98.6±0.01 99.9±0.01 0.01±0.00 0.00±0.00 | 99.0±0.00 99.9±0.00                     |                     |        |         |        |         |
| 103             | 5                 | 1000                   | vMF(κ= 10)                                | ✓ 99.3±0.00 99.9±0.00 0.00±0.00 0.00±0.00 | 99.2±0.00 99.9±0.00                     |                     |        |         |        |         |
| 103             | 5                 | 100                    | vMF(κ= 5)                                 | ✓ 98.6±0.01 99.9±0.01 0.01±0.00 0.00±0.00 | 99.0±0.00 99.8±0.00                     |                     |        |         |        |         |
| 103             | 5                 | 100                    | vMF(κ= 10)                                | ✓ 99.0±0.00 99.9±0.00 0.00±0.00 0.00±0.00 | 99.1±0.00 99.9±0.00                     |                     |        |         |        |         |
| 103             | 5                 | 100                    | vMF(κ= 50)                                | ✓ 45.0±0.06 49.7±0.06 0.30±0.00 0.00±0.00 | 72.5±0.03 75.5±0.00                     |                     |        |         |        |         |
| 103             | 5                 | 100                    | vMF(κ= 10)                                | ✓ 98.6±0.01 99.9±0.01 0.01±0.00 0.00±0.00 | 99.0±0.00 99.9±0.00                     |                     |        |         |        |         |
| 103             | 5                 | 100                    | Laplace (b= 1.0)                          | ✗                                         | 85.2±0.01 99.7±0.01 0.01±0.00 0.00±0.00 | 85.4±0.00 99.5±0.00 |        |         |        |         |
| 103             | 5                 | 100 Normal (σ 2 = 1.0) | ✗                                         | 98.7±0.00 99.8±0.00 0.01±0.00 0.00±0.00   | 98.6±0.00 99.6±0.00                     |                     |        |         |        |         |

according to an exponential family distribution and the auxiliary variable is a scalar (e.g., time), then we get the more specialized method, named Time-Contrastive Learning (TCL) (Hyvarinen & Morioka, 2016). If the order of the exponential family is one, identifiability holds only up to a linear transformation, otherwise, up to elementwise invertible transformations. From TCL to DIET (Fig. 2b: sample index as u and vMF **latent variables).** Using our clustercentric DGP (Assums. 1), and assuming an even more special latent distribution (i.e., a vMF), we get the identifiability guarantee for DIET, i.e., our main result in Thm. 1. The auxiliary variable is a scalar for our result, too; however, instead of time, it is the (arbitrary) sample index.

From InfoNCE to DIET (Fig. 2c: a compositional model W ◦ f **and unit-normalized latent** variables). Importantly, our main result also encompasses unit-normalized representations, the conventional choice in (identifiable) SSL such as InfoNCE (cf. Appx. B.3 for details on InfoNCE)—
this is why we illustrate both InfoNCE and TCL as being the "parents" of DIET in Fig. 2. Thus, Thm. 1 is more general in terms of latent spaces than nonlinear ICA, and it proves identifiability for the latent variables that are used post-training, as opposed to the proofs for InfoNCE in (Zimmermann et al., 2021; Rusak et al., 2024), where practitioners discard the last few layers. From DIET to supervised classification (Fig. 2d: semantic class labels). When the labeling function assigns the semantic class labels, and not arbitrary indices, then our identifiability result still holds, yielding the case of supervised learning (Thm. 2).

## 4 Empirical Results

In § 4.1, we empirically verify the claims made in Thm. 1 and Thm. 2 in the synthetic setting. We generate data samples according to Assums. 1: ground-truth latent variables are sampled around cluster centroids vc following a vMF distribution. Data augmentations, which share the same instance label i, are sampled from the same vMF distribution around vc. In § 4.2, we describe our results on the DisLib disentanglement benchmark (Locatello et al., 2019), and § 4.3 includes our experiments on ImageNet-X (Idrissi et al., 2022). We made our code publicly available on GitHub1.

## 4.1 Synthetic Data

Setup. We consider N latent samples of dimensionality d generated from the conditional vMF z ∼ p(z|vc), sampled around a set of |C | class vectors vc, which are uniformly distributed across the unit hyper-sphere S
d−1. We use an invertible multi-layer perceptron (MLP) to map ground-truth latent variables to data samples. We train a classification head W =[w⊤
i | N
i=1] and an MLP encoder that maps samples to representations z˜ ∈ R
d using the DIET objective (1). While to verify Thm. 1 case C4., we do not normalize W, we do unit-normalize the weight vectors to validate Thm. 1 case C3. We verify our theoretical claims by measuring the predictability of the ground-truth z from z˜
and vc from wi using the R2score on a held-out dataset (Wright, 1921). For identifiability up to orthogonal linear transformations, we train linear mappings with no intercept, assess the R2score and verify that the singular values of this transformation converge to 1, while for identifiability up to affine linear transformations, we simply assess the R2 of a linear predictor with intercept.

Results for DIET. In Tab. 2, we report the R2 scores for the recovery of the cluster vectors vc from W's rows and of the ground-truth latent variables z from the learned latent variables z˜. For DIET's PID task, we also consider cases with row-normalized W. We observe scores close to 100% (≥ 98%), even with many clusters (≥ 103) and samples (∼ 105). High latent dimensionality (> 10) does impact the recovery of ground-truth latent variables—such scalability problems are a common artifact in SSL (Zimmermann et al., 2021; Rusak et al., 2024). For a higher concentration of samples around vc
(i.e., κ= 50) as well as a lower number of clusters (i.e., |C | = 10), the R2score decreases, which is also a common phenomenon, and is possibly explained by too strong augmentation overlap (Wang et al., 2022; Rusak et al., 2024). For a low number of clusters, high κ and a fixed number of training samples, the concentration of samples in regions surrounding centroids, vc, increases, a setting, refered to as "overly overlapping augmentations", known to be suboptimal and leading to a drop in downstream performance (Wang et al., 2022). Our results also suggest that even under model misspecification (last two rows in Tab. 2 with non-vMF distributions), identifiability still holds. For unit-normalized Wrows, the MAE is lower, confirming the orthogonality of the map wi→vc. We additionally ablate over batch size, concentration, and conditional in Appx. D. Results for Supervised Classification. In Tab. 3, where the semantic class labels were used instead of the sample index, we only report the R2score for the recovery of the ground-truth latent variables z from the learned latent variables z˜. In all but one setting, we observe higher R2from representations learned with class labels rather than instance indices. This suggests that even a coarser classification task may suffice to learn linearly identifiable representations of the underlying latent variables.

## 4.2 Dislib

Setup. Next, we evaluate our methods on the DisLib disentanglement benchmark (Locatello et al., 2019), which provides a controlled setting for testing disentanglement and latent variable recovery. It includes the vision datasets dSprites, Shapes 3D, MPI 3D, Cars 3D, and smallNORB. We train both a three-layer MLP with 512 latent dimensions and BatchNorm (which helped with trainability) and a CNN (ResNet18) also with 512 latent dimensions. We only consider latent variables with Euclidean topology, as non-Euclidean, e.g., periodic latent variables such as orientation, are problematic to learn and are potentially mapped to a nonlinear manifold (Higgins et al., 2018; Pfau et al., 2020; Keurti et al., 2023; Engels et al., 2024). We evaluate the recovery of latent variables by computing the Pearson correlation between ground-truth and predicted factors. We detail our setup in Appx. C.2.

d |C | p(z|vc) M. R2: z˜→z 5 100 vMF(κ= 10) ✓ 99.8±0.00 10 100 vMF(κ= 10) ✓ 97.2±0.01 20 100 vMF(κ= 10) ✓ 82.1±0.02 5 10 vMF(κ= 10) ✓ 97.5±0.03 5 100 vMF(κ= 10) ✓ 99.8±0.00 5 1000 vMF(κ= 10) ✓ 99.8±0.00 5 10000 vMF(κ= 10) ✓ 99.8±0.00 5 100 vMF(κ= 5) ✓ 99.7±0.00 5 100 vMF(κ= 10) ✓ 99.7±0.00 5 100 vMF(κ= 50) ✓ 65.5±0.09 5 100 vMF(κ= 10) ✓ 99.8±0.00 5 100 Laplace (b= 1.0) ✗ 85.4±0.01 5 100 Normal (σ 2 = 1.0) ✗ 99.6±0.00 Table 3: **Identifiability results for supervised** learning in numerical simulations: Mean ± standard deviation across 5 random seeds. Settings that match and violate our theoretical assumptions are denoted as ✓ and ✗, respectively. We report the R2score for linear mappings z˜ → z, and not normalized wi. We used N = 103samples Table 4: **Identifiability in DisLib datasets (Locatello et al., 2019):** We train different models to predict the categorical variable in each setting: (x): as a baseline, from the inputs; (f MLP(x)): from a three-layer MLP; and (f CNN(x)): from a CNN (ResNet18). All continuous latent variables can be decoded from the learned representations, corroborated by the Pearson correlation—reported with mean ± standard deviation across 3 random seeds. Including the category is informative to see how well the underlying training classification task was solved.

Model Latent x f MLP(x) fCNN(x)

dSprites category 0.26±0.00 0.94±0.01 1.00±0.00 dSprites scale 0.62±0.00 0.98±0.00 0.92±0.05 dSprites posX 0.92±0.00 0.97±0.00 0.99±0.00 dSprites posY 0.92±0.00 0.97±0.00 0.99±0.00 Shapes 3D category 0.42±0.00 1.00±0.00 1.00±0.00 Shapes 3D objSize 0.21±0.00 0.89±0.01 0.99±0.00 Shapes 3D objAzimuth 0.04±0.00 0.85±0.02 0.93±0.01 MPI 3D category 0.03±0.00 0.71±0.01 0.97±0.00 MPI 3D posX 0.28±0.00 0.76±0.01 0.90±0.01 MPI 3D posY 0.46±0.00 0.76±0.01 0.84±0.01 MPI 3D real category 0.19±0.00 0.88±0.01 0.98±0.00 MPI 3D real posX 0.14±0.00 0.74±0.01 0.83±0.01 MPI 3D real posY 0.44±0.00 0.54±0.01 0.71±0.02 Cars 3D category 0.05±0.00 0.63±0.11 0.77±0.02 Cars 3D elevation 0.15±0.00 0.87±0.03 0.78±0.02

smallNORB category 0.22±0.00 0.94±0.01 1.00±0.00

smallNORB elevation 0.15±0.00 0.83±0.01 0.79±0.01

object blockingpose color larger smaller style shape darker class 0.4 0.6 0.8 1.0
* *** *
* **
*
chance level Input Vit-b-16 (init) ResNet50 (init)
Vit-b-16 (trained) ResNet50 (trained)
*
** **
**
*
Ac cu ra cy
*
*
*
*
* * *
*
* *
background brighter pattern partial viewsubcategory multiple objectstexture person blocking 0.4 0.6 0.8 1.0
*****
*
chance level *
Input Vit-b-16 (init) ResNet50 (init)
Vit-b-16 (trained) ResNet50 (trained)
* *
* *
*
**
** ** *
*
Accu ra cy
Results. The models trained using cross-entropy were able to recover latent variables such as object position, scale, and orientation with high accuracy. As shown in Tab. 4, the Pearson correlation is generally highest when predicting the latent variables from the CNN's representation, which we attribute to the CNN's suitable inductive bias for images. In few cases, such as the position in dSprites, this can be done with fairly high accuracy even on the input data. Nevertheless, in all settings the nonlinear function estimated by the model is necessary to linearly identify the correct latent variables.

## 4.3 Real Data: Imagenet-X

Setup. Finally, we test the generalizability of our theoretical insights on real-world data using ImageNet-X (Idrissi et al., 2022). The latent variables are binary proxies, defined by human annotators (Idrissi et al., 2022). We evaluate how well linear decoders can predict latent variables from pretrained model representations. We use two architectures, a ResNet50 and a Vit-b-16 both trained on standard supervised classification using a cross-entropy loss on the full ImageNet dataset (Deng et al., 2009). As baselines, we also decode from the inputs and the randomly initialized models. After balanced sub-sampling, over 10 random seeds, we report accuracies. We use t-tests against a chance level of 50% with a Bonferroni adjusted significance level of κ =
0.05 17·5
. Detail are in Appx. C.3.

Results. Fig. 3 shows that even in complex, high-dimensional data, latents can be linearly decoded from representations learned via supervised learning, in most cases significantly above chance level. Some factors (e.g., *darker* and *brighter*) are linearly decodable even from untrained models or input space. Unsurprisingly, decoding *class* (binarized ImageNet labels, every index < 500 is set to 0 and every index ≥500 is set to 1) works well for the trained models. ResNet50 has slightly higher decoding performance, possibly due to the larger latent space (d= 2048, compared to d= 768 in ViT). While texture information may be expected (Geirhos et al., 2018), the presence of shape information suggests that shortcut learning may be mitigated even after standard training (Geirhos et al., 2020).

## 5 Discussion

Limitations. One limitation of our work is that we mainly focus on synthetic and controlled datasets. While the results on ImageNet-X (Idrissi et al., 2022) are promising, they only provide some supporting evidence for our theory on real data. The factors in ImageNet-X are likely not the true latent variables of the data generating process, still, the linear identifiablity results on these proxy latent variables support our theoretical results. Further experiments on other large-scale datasets would support the generality of our findings. However, this would require the availability of such datasets with full latent variable annotations. Although our cluster-centric modeling of the data generating process allows capturing the inherent structure of the data, our assumption about the latent variables' geometric properties (such as being drawn from a vMF distribution on a hypersphere),
may not hold in all real-world settings. For instance, the pose of an object in a scene is, arguably, an independent component/subspace corresponding to a point on SO(3), which has a distinct topology from our assumed latent variables on a hypersphere. Moreover, the assumption that a data sample and its augmented version are conditionally independent given their semantic class could be relaxed in future work, since it may be misaligned with realistic scenarios (Wang et al., 2022). Despite these simplifications, our experimental results also suggest that our assumptions can be relaxed, as linear identifiability seems to hold even when some of the assumptions are violated (cf. Tab. 5). In Appx. D, we demonstrate the remarkable robustness of latent identifiability (Fig. 6), the interaction between batch size, latent dimensionality, concentration, and latent conditional. Implications for Deep Learning. Our results indicate that deep learning models trained using cross-entropy and assuming a certain DGP recover the underlying latent variables up to linear transformations. As our identifiability proof for parametric instance discrimination illustrates with DIET, this statement also holds when the classification task is standard supervised learning. Our analysis on the key role of cross-entropy-based classification provides a theoretical foundation for phenomena such as neural analogy-making, transfer learning, and linear decoding of features. Conclusion. We extend the identifiability results of the auxiliary-variable nonlinear Independent Component Analysis (ICA) literature to parametric instance discrimination with a cluster-centric data generating process. Our modeling choice can capture the clustered structure of the data, accommodates non-normalized (as in ICA) and unit-normalized (as in InfoNCE) representations (Thm. 1). Furthermore, our identifiability result holds for the latent representation used post-training, i.e., for the latent variables before the classification head. Our results offer new insights into the success of deep learning, particularly in supervised classification tasks, which we show is a special case of the DIET parametric instance discrimination algorithm, where the instance labels equal the semantic class labels (Thm. 2). By linking self-supervised learning—via nonlinear ICA and DIET—to supervised classification for a specific DGP, we provide a theoretical framework that explains why simple classification tasks recover interpretable and transferable representations. Future Work. Future research could extend these insights to connections between nonlinear ICA and other forms of supervised learning and testing the scalability of our theoretical results to larger models and datasets. To assess our theory's predictions beyond proxy labels (Idrissi et al., 2022), we need real world image datasets with full specification of the latent variables, e.g., in rendered scenes.

## Acknowledgments

The authors thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting Patrik Reizinger and Attila Juhos. Patrik Reizinger acknowledges his membership in the European Laboratory for Learning and Intelligent Systems (ELLIS) PhD program. This work was supported by the German Federal Ministry of Education and Research (BMBF): Tubingen AI Center, ¨ FKZ: 01IS18039A. Wieland Brendel acknowledges financial support via an Emmy Noether Grant funded by the German Research Foundation (DFG) under grant no. BR 6382/1-1 and via the Open Philantropy Foundation funded by the Good Ventures Foundation. Wieland Brendel is a member of the Machine Learning Cluster of Excellence, EXC number 2064/1 - Project number 390727645. This research utilized compute resources at the Tubingen Machine Learning Cloud, DFG FKZ INST ¨ 37/1057-1 FUGG. Alice Bizeul's work is supported by an ETH AI Center Doctoral fellowship.

## References

Kartik Ahuja, Divyat Mahajan, Vasilis Syrgkanis, and Ioannis Mitliagkas. Towards efficient representation identification in supervised learning. In Proceedings of the First Conference on Causal Learning and Reasoning, pp. 19–43. PMLR, June 2022. URL https://proceedings.mlr. press/v177/ahuja22a.html. ISSN: 2640-3498. 2 Carl Allen and Timothy Hospedales. Analogies explained: Towards understanding word embeddings.

In *International Conference on Machine Learning*, pp. 223–231. PMLR, 2019. 3 Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski. A Latent Variable Model Approach to PMI-based Word Embeddings. *Transactions of the Association for Computational* Linguistics, 4:385–399, July 2016. ISSN 2307-387X. doi: 10.1162/tacl a 00106. URL https:
//doi.org/10.1162/tacl_a_00106. 1, 3 Randall Balestriero and Yann LeCun. Contrastive and Non-Contrastive Self-Supervised Learning Recover Global and Local Spectral Embedding Methods, June 2022. URL http://arxiv.

org/abs/2205.11508. arXiv:2205.11508 [cs, math, stat]. 4 Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. *IEEE transactions on pattern analysis and machine intelligence*, 35(8):1798–1828, 2013. 1 Florian Bordes, Randall Balestriero, Quentin Garrido, Adrien Bardes, and Pascal Vincent. Guillotine Regularization: Why removing layers is needed to improve generalization in Self-Supervised Learning. *Transactions on Machine Learning Research*, May 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=ZgXfXSz51n&s=09. 4 Diane Bouchacourt, Mark Ibrahim, and Stephane Deny. Addressing the topological defects of ´
disentanglement via distributed operators. *arXiv preprint arXiv:2102.05623*, 2021. 3 Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A Simple Framework for Contrastive Learning of Visual Representations. *arXiv:2002.05709 [cs, stat]*, June 2020. URL
http://arxiv.org/abs/2002.05709. arXiv: 2002.05709. 3, 5 Zirui Chen and Michael Bonner. Canonical dimensions of neural visual representation. Journal of Vision, 23(9):4937–4937, 2023. 3 Pierre Comon. Independent component analysis, a new concept? *Signal processing*, 36(3):287–314, 1994. 1, 3 Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE conference on computer vision and pattern recognition*, pp. 248–255. Ieee, 2009. 10, 28 Jeff Donahue, Yangqing Jia, Oriol Vinyals, Judy Hoffman, Ning Zhang, Eric Tzeng, and Trevor Darrell. Decaf: A deep convolutional activation feature for generic visual recognition. In International conference on machine learning, pp. 647–655. PMLR, 2014. 1 Alexey Dosovitskiy, Jost Tobias Springenberg, Martin Riedmiller, and Thomas Brox. Discriminative unsupervised feature learning with convolutional neural networks. *Advances in neural information* processing systems, 27, 2014. 2, 4 Aleksandr Drozd, Anna Gladkova, and Satoshi Matsuoka. Word embeddings, analogies, and machine learning: Beyond king-man+ woman= queen. In *Proceedings of coling 2016, the 26th international* conference on computational linguistics: Technical papers, pp. 3519–3530, 2016. 3 Kang Du and Yu Xiang. Causal Inference from Slowly Varying Nonstationary Processes.

arXiv:2012.13025 [cs, math, stat], September 2021. URL http://arxiv.org/abs/2012.

13025. arXiv: 2012.13025. 3 Philipp Dufter and Hinrich Schutze. Analytical methods for interpretable ultradense word embeddings. ¨
arXiv preprint arXiv:1904.08654, 2019. 3 Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy Models of Superposition, September 2022. URL http://arxiv.org/abs/2209.10652. arXiv:2209.10652
[cs]. 1 Joshua Engels, Isaac Liao, Eric J Michaud, Wes Gurnee, and Max Tegmark. Not all language model features are linear. *arXiv preprint arXiv:2405.14860*, 2024. 3, 8, 27 Benoˆıt Frenay and Michel Verleysen. Classification in the presence of label noise: a survey. ´ IEEE
transactions on neural networks and learning systems, 25(5):845–869, 2013. 30 Marco Fumero, Florian Wenzel, Luca Zancato, Alessandro Achille, Emanuele Rodola, Stefano Soatto, `
Bernhard Scholkopf, and Francesco Locatello. Leveraging sparse and shared feature activations ¨
for disentangled representation learning, April 2023. URL http://arxiv.org/abs/2304. 07939. arXiv:2304.07939 [cs]. 2 Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A Wichmann, and Wieland Brendel. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. *arXiv preprint arXiv:1811.12231*, 2018. 10 Robert Geirhos, Jorn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias ¨
Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, November 2020. ISSN 2522-5839. doi: 10.1038/s42256-020-00257-z. URL https://www.nature.com/articles/s42256-020-00257-z. Number: 11 Publisher: Nature Publishing Group. 10 Mikhail Genkin and Tatiana A Engel. Moving beyond generalization to accurate interpretation of flexible models. *Nature machine intelligence*, 2(11):674–683, 2020. 3 Luigi Gresele, Paul K. Rubenstein, Arash Mehrjou, Francesco Locatello, and Bernhard Scholkopf. ¨
The Incomplete Rosetta Stone Problem: Identifiability Results for Multi-View Nonlinear ICA. arXiv:1905.06642 [cs, stat], August 2019. URL http://arxiv.org/abs/1905.06642. arXiv: 1905.06642. 1, 3 Jeff Z. HaoChen, Colin Wei, Ananya Kumar, and Tengyu Ma. Beyond Separability: Analyzing the Linear Transferability of Contrastive Representations to Related Subpopulations, May 2022. URL http://arxiv.org/abs/2204.02683. arXiv:2204.02683 [cs]. 4 Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*,
pp. 770–778, 2016. 27 Irina Higgins, David Amos, David Pfau, Sebastien Racaniere, Loic Matthey, Danilo Rezende, and Alexander Lerchner. Towards a Definition of Disentangled Representations. arXiv:1812.02230 [cs, stat], December 2018. URL http://arxiv.org/abs/1812.02230. arXiv: 1812.02230. 3, 8, 27 Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. The platonic representation hypothesis. *arXiv preprint arXiv:2405.07987*, 2024. 3 Aapo Hyvarinen and Hiroshi Morioka. Unsupervised Feature Extraction by Time-Contrastive Learning and Nonlinear ICA. *arXiv:1605.06336 [cs, stat]*, May 2016. URL http://arxiv. org/abs/1605.06336. arXiv: 1605.06336. 1, 2, 3, 4, 6, 7 Aapo Hyvarinen, Juha Karhunen, and Erkki Oja. *Independent component analysis*. J. Wiley, New York, 2001. ISBN 978-0-471-40540-5. 1, 3 Aapo Hyvarinen, Hiroaki Sasaki, and Richard E. Turner. Nonlinear ICA Using Auxiliary Variables and Generalized Contrastive Learning. *arXiv:1805.08651 [cs, stat]*, February 2019. URL http:
//arxiv.org/abs/1805.08651. arXiv: 1805.08651. 1, 2, 3, 4, 6, 24, 25, 26 Aapo Hyvarinen, Ilyes Khemakhem, and Ricardo Monti. Identifiability of latent-variable and ¨
structural-equation models: from linear to nonlinear, February 2023. URL http://arxiv. org/abs/2302.02672. arXiv:2302.02672 [cs, stat]. 1, 3 Hermanni Halv ¨ a, Sylvain Le Corff, Luc Leh ¨ ericy, Jonathan So, Yongjie Zhu, Elisabeth Gassiat, and ´
Aapo Hyvarinen. Disentangling Identifiable Features from Noisy Data with Structured Nonlinear ICA. *arXiv:2106.09620 [cs, stat]*, June 2021. URL http://arxiv.org/abs/2106.09620. arXiv: 2106.09620. 2, 3 Mark Ibrahim, David Klindt, and Randall Balestriero. Occam's Razor for Self Supervised Learning:
What is Sufficient to Learn Good Representations?, June 2024. URL http://arxiv.org/
abs/2406.10743. arXiv:2406.10743 [cs]. 2, 3, 4, 5 Badr Youbi Idrissi, Diane Bouchacourt, Randall Balestriero, Ivan Evtimov, Caner Hazirbas, Nicolas Ballas, Pascal Vincent, Michal Drozdzal, David Lopez-Paz, and Mark Ibrahim. Imagenet-x: Understanding model mistakes with factor of variation annotations, 2022. URL https:// arxiv.org/abs/2211.01866. 2, 7, 9, 10, 27 Li Jing, Pascal Vincent, Yann LeCun, and Yuandong Tian. Understanding Dimensional Collapse in Contrastive Self-supervised Learning, April 2022. URL http://arxiv.org/abs/2110. 09348. Number: arXiv:2110.09348 arXiv:2110.09348 [cs]. 29 Hamza Keurti, Patrik Reizinger, Bernhard Scholkopf, and Wieland Brendel. Desiderata for Represen- ¨
tation Learning from Identifiability, Disentanglement, and Group-Structuredness. June 2023. URL https://openreview.net/forum?id=r6C86JjuiW. 3, 8, 27 Ilyes Khemakhem, Diederik Kingma, Ricardo Monti, and Aapo Hyvarinen. Variational Autoencoders and Nonlinear ICA: A Unifying Framework. In International Conference on Artificial Intelligence and Statistics, pp. 2207–2217. PMLR, June 2020a. URL http://proceedings. mlr.press/v108/khemakhem20a.html. ISSN: 2640-3498. 1, 3 Ilyes Khemakhem, Ricardo Pio Monti, Diederik P. Kingma, and Aapo Hyvarinen. ICE-BeeM: Identi- ¨
fiable Conditional Energy-Based Deep Models Based on Nonlinear ICA. *arXiv:2002.11537 [cs,* stat], October 2020b. URL http://arxiv.org/abs/2002.11537. arXiv: 2002.11537. 1, 3 Michael Kirchhof, Karsten Roth, Zeynep Akata, and Enkelejda Kasneci. A Non-isotropic Probabilistic Take on Proxy-based Deep Metric Learning, July 2022. URL http://arxiv.org/abs/ 2207.03784. arXiv:2207.03784 [cs, stat]. 4 David Klindt, Lukas Schott, Yash Sharma, Ivan Ustyuzhaninov, Wieland Brendel, Matthias Bethge, and Dylan Paiton. Towards Nonlinear Disentanglement in Natural Data with Temporal Sparse Coding. *arXiv:2007.10930 [cs, stat]*, March 2021. URL http://arxiv.org/abs/2007. 10930. arXiv: 2007.10930. 1, 3 Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. *Advances in neural information processing systems*, 25, 2012. 1 Sebastien Lachapelle, Tristan Deleu, Divyat Mahajan, Ioannis Mitliagkas, Yoshua Bengio, Simon Lacoste-Julien, and Quentin Bertrand. Synergies between Disentanglement and Sparsity: Generalization and Identifiability in Multi-Task Learning. In Proceedings of the 40th International Conference on Machine Learning, pp. 18171–18206. PMLR, July 2023. URL https:
//proceedings.mlr.press/v202/lachapelle23a.html. ISSN: 2640-3498. 2 Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Scholkopf, ¨
and Olivier Bachem. Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations. In *International Conference on Machine Learning*, pp. 4114–4124. PMLR, May 2019. URL http://proceedings.mlr.press/v97/locatello19a. html. ISSN: 2640-3498. 2, 7, 8, 9, 27 Francesco Locatello, Ben Poole, Gunnar Ratsch, Bernhard Sch ¨ olkopf, Olivier Bachem, and Michael ¨
Tschannen. Weakly-Supervised Disentanglement Without Compromises. arXiv:2002.02886 [cs, stat], October 2020. URL http://arxiv.org/abs/2002.02886. arXiv: 2002.02886. 1, 3 Tomas Mikolov. Efficient estimation of word representations in vector space. *arXiv preprint* arXiv:1301.3781, 2013. 3 Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed Representations of Words and Phrases and their Compositionality. In Advances in Neural Information Processing Systems, volume 26. Curran Associates, Inc., 2013. URL https://papers.nips.cc/paper_files/paper/2013/hash/ 9aa42b31882ec039965f3c4923ce901b-Abstract.html. 1, 3 Hiroshi Morioka and Aapo Hyvarinen. Connectivity-contrastive learning: Combining causal discovery and representation learning for multimodal data. In Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, pp. 3399–3426. PMLR, April 2023. URL https://proceedings.mlr.press/v206/morioka23a.html. ISSN: 2640-3498. 2, 3 Hiroshi Morioka, Hermanni Halv ¨ a, and Aapo Hyv ¨ arinen. Independent Innovation Analysis for ¨
Nonlinear Vector Autoregressive Process. *arXiv:2006.10944 [cs, stat]*, February 2021. URL https://arxiv.org/abs/2006.10944. arXiv: 2006.10944. 2, 3 Luca Moschella, Valentino Maiorca, Marco Fumero, Antonio Norelli, Francesco Locatello, and Emanuele Rodola. Relative representations enable zero-shot latent space communication, March ` 2023. URL http://arxiv.org/abs/2209.15430. arXiv:2209.15430 [cs]. 3 David F Nettleton, Albert Orriols-Puig, and Albert Fornells. A study of the effect of different types of noise on the precision of supervised learning techniques. *Artificial intelligence review*, 33:275–306, 2010. 30 Kiho Park, Yo Joong Choe, and Victor Veitch. The Linear Representation Hypothesis and the Geometry of Large Language Models, November 2023. URL http://arxiv.org/abs/ 2311.03658. arXiv:2311.03658 [cs, stat]. 1, 2, 3 David Pfau, Irina Higgins, Alex Botev, and Sebastien Racani ´ ere. Disentangling by Subspace Diffusion. `
In *Advances in Neural Information Processing Systems*, volume 33, pp. 17403–17415. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ c9f029a6a1b20a8408f372351b321dd8-Abstract.html. 3, 8, 27 Geoffrey Roeder, Luke Metz, and Diederik P. Kingma. On Linear Identifiability of Learned Representations. *arXiv:2007.00810 [cs, stat]*, July 2020. URL http://arxiv.org/abs/2007. 00810. arXiv: 2007.00810. 3 D Rolnick. Deep learning is robust to massive label noise. *arXiv preprint arXiv:1705.10694*, 2017.

31 Karsten Roth, Mark Ibrahim, Zeynep Akata, Pascal Vincent, and Diane Bouchacourt. Disentanglement of correlated factors via hausdorff factorized support. *arXiv preprint arXiv:2210.07347*, 2022. 27 Evgenia Rusak, Patrik Reizinger, Attila Juhos, Oliver Bringmann, Roland S. Zimmermann, and Wieland Brendel. InfoNCE: Identifying the Gap Between Theory and Practice, June 2024. URL
http://arxiv.org/abs/2407.00143. arXiv:2407.00143 [cs, stat]. 2, 3, 4, 7, 8, 29 James B. Simon, Maksis Knutins, Liu Ziyin, Daniel Geisz, Abraham J. Fetterman, and Joshua Albrecht. On the Stepwise Nature of Self-Supervised Learning, May 2023. URL http:// arxiv.org/abs/2303.15438. arXiv:2303.15438 [cs]. 29 Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, and et al. Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet, 2024. URL https:
//transformer-circuits.pub/2024/scaling-monosemanticity. 1 Julius von Kugelgen, Yash Sharma, Luigi Gresele, Wieland Brendel, Bernhard Sch ¨ olkopf, Michel ¨
Besserve, and Francesco Locatello. Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style, June 2021. URL http://arxiv.org/abs/2106.04619.

arXiv: 2106.04619. 2, 3, 4, 29 Yifei Wang, Qi Zhang, Yisen Wang, Jiansheng Yang, and Zhouchen Lin. Chaos is a Ladder: A New Theoretical Understanding of Contrastive Learning via Augmentation Overlap, May 2022. URL http://arxiv.org/abs/2203.13457. arXiv:2203.13457 [cs, stat]. 4, 8, 10 Wikipedia. Gibbs' inequality, 2024a. URL https://en.wikipedia.org/w/index.php?

title=Gibbs%27_inequality&oldid=1231436245. Online; accessed 10-September2024. 18 Wikipedia. Tietze extension theorem, 2024b. URL https://en.wikipedia.org/w/index.

php?title=Tietze_extension_theorem&oldid=1237682676. Online; accessed 10-September-2024. 17 Sewall Wright. Correlation and causation. *Journal of Agricultural Research*, (7), 1921. 8 Zhirong Wu, Yuanjun Xiong, Stella X. Yu, and Dahua Lin. Unsupervised Feature Learning via Non-Parametric Instance Discrimination. pp. 3733–3742, 2018.

URL https://openaccess.thecvf.com/content_cvpr_2018/html/Wu_ Unsupervised_Feature_Learning_CVPR_2018_paper.html. 4 Qi Zhang, Yifei Wang, and Yisen Wang. Identifiable Contrastive Learning with Automatic Feature Importance Discovery, October 2023. URL http://arxiv.org/abs/2310.18904.

arXiv:2310.18904 [cs]. 3 Roland S. Zimmermann, Yash Sharma, Steffen Schneider, Matthias Bethge, and Wieland Brendel.

Contrastive Learning Inverts the Data Generating Process. *arXiv:2102.08850 [cs]*, February 2021.

URL http://arxiv.org/abs/2102.08850. arXiv: 2102.08850. 2, 3, 4, 7, 8, 29

## A Identifiability Of Latents Drawn From A Vmf Around Cluster Vectors

This section contains the formal statement and proof of our main theoretical result. Appx. A.1 contains the relevant definition of affine generator systems. Appx. A.2 contains the assumptions and the proof for all four combinations of unit-normalized and non-normalized features/cluster vectors for parametric instance discrimination. Appx. A.3 discusses a special case, supervised classification.

## A.1 Affine Generator Systems

Definition 1 (Affine Generator System). *A system of vectors* {vc ∈ R
d|c ∈ C } *is called an* affine generator system *if any vector in* R
d*is an affine linear combination of the vectors in the system. Put* into symbols: for any v ∈ R
dthere exist coefficients αc ∈ R*, such that*

$\mathbf{v}=\sum_{c\in\mathscr{C}}\alpha_{c}\mathbf{v}_{c}$ and $\sum_{c\in\mathscr{C}}\alpha_{c}=1$.  
$$(3)$$
Lemma 1 (Properties of affine generator systems). The following hold for any affine generator system {vc ∈ R
d|c ∈ C }:
1. for any a ∈ C the system {vc − va|c ∈ C } *is now a generator system of* R
d; 2. *the invertible linear image of an affine generator system is also an affine generator system.* A.2 IDENTIFIABILITY OF PARAMETRIC INSTANCE DISCRIMINATION Assumptions 1C (DGP with vMF samples around cluster vectors). Assume the following DGP:
(i) There exists a finite set of classes C , represented by a set of unit-norm d-dimensional clustervectors {vc|c ∈ C } ⊆ S
d−1*such that they form an affine generator system of* R
d.

(ii) There is a finite set of instance labels I *and a well-defined, surjective* class function C : I →
C (every label belongs to exactly one class and every class is in use).

(iii) A data sample x belongs to class C = C(I) and is labeled with a uniformly-chosen instance label, i.e., I ∈ Uni(I ).

(iv) *The latent* z ∈ S
d−1 of our data sample with label I *is drawn from a vMF distribution around* the cluster vector vC *, where* C = C(I):
z ∼ p(z|C) ∝ e κ⟨vC ,z⟩. (4)
(v) The data sample x is generated by passing the latent z *through a continuous and injective* generator function g :S
d−1→R
D*, i.e.,* x = g(z).

Assume that, using the DIET objective (6), we train a continuous encoder f : R
D → R
d on x and a linear classification head W on top of f. The rows of W are w⊤
i |i ∈ I	. In other words, W
computes similarities (scalar products) between its rows and the embeddings:
W : f(x) 7→-⟨wi, f(x)⟩ | i∈I. (5)
In DIET, we optimize the following objective among all possible continuous encoders f, linear classifiers W, and β > 0:

$${\mathcal{L}}(\mathbf{f},\mathbf{W},\beta)=\mathbb{E}_{(\mathbf{x},I)}\left[-\ln\frac{e^{\beta\langle\mathbf{w}_{I},\mathbf{f}(\mathbf{x})\rangle}}{\sum_{j\in{\mathcal{I}}}e^{\beta\langle\mathbf{w}_{j},\mathbf{f}(\mathbf{x})\rangle}}\right]\tag{1}$$

In the special case where the embeddings f(x) are unnormalized, but the parameter vectors wi are unit-normalized, the identifiability proof will solicit another, technical assumption:
Assumption 2 (Diverse data). The system {vc|c ∈ C } *is said to be diverse enough, if the following* |C | × 2d *matrix has full column rank of* 2d:

$$\left(\begin{array}{l l l}{{\cdots\cdots\cdots}}&{{\cdots\cdots\cdots}}\\ {{(\mathbf{v}_{c}\odot\mathbf{v}_{c})^{\top}}}&{{}}&{{\mathbf{v}_{c}^{\top}}}\\ {{\cdots\cdots\cdots\cdots}}&{{\cdots\cdots\cdots\cdots}}\end{array}\right)\,,$$
$$(4)$$
$$({\mathfrak{H}})$$
$$(6)$$

$$(7)$$
 , (7)
where [x ⊙ y]i = xiyiis the elementwise- or Hadamard product.

As long as |C | ≥ 2d*, this property holds almost surely w.r.t. the Lebesgue-measure of* S
d−1 or any continuous probability distribution of vc ∈ S
d−1.

Theorem 1C (Identifiability of latents drawn from a vMF around cluster vectors). Let (f,W, β) globally minimize the DIET objective (6) *under Assums. 1C and the following additional* constraints: C1. both the embeddings f(x) and wi's are unit-normalized. Then:
(a) h = f ◦ g is orthogonal linear, i.e., the latents are identified up to an orthogonal linear transformation;
(b) wi = h(vC(i)) for any i ∈ I , i.e., wi's identify the cluster-vectors vc up to the same orthogonal linear transformation;
(c) β = κ, the temperature of the vMF distribution is also identified.

C2. the embeddings f(x) are unit-normalized, the wi's are unnormalized. Then:
(a) h = f ◦ g *is orthogonal linear;*
(b) wi =
κ β h(vC(i)) + ψ for any i ∈ I , where ψ *is a constant vector independent of* i.

C3. the embeddings f(x) are unnormalized, while the wi*'s are unit-normalized. If the system*
{vc|c} **is diverse enough in the sense of** Assum. 2, then:
(a) wi = OvC(i), for any i ∈ I , where O is orthogonal linear;
(b) h = f ◦ g =
κ βO *with the same orthogonal linear transformation, but scaled with* κβ
.

C4. neither the embeddings f(x) nor the rows of W are unit-normalized. Then:
(a) h = f ◦ g *is linear;*
(b) wiidentifies vC(i) up to an affine linear transformation.

Furthermore, in all cases, the row vectors that belong to samples of the same class are equal, i.e., for any *i, j* ∈ I , C(i) = C(j) *implies* wi = wj .

Remark. In cases C2 and C4, the cluster vectors are unnormalized and, therefore, can absorb the temperature parameter β. Thus β can be set to 1 without loss of generality. In case C3, it is f that can absorb β. Proof. **Step 1: Deriving an equation characterizing the global optimizers of the objective.** Rewriting the objective in terms of latents: we plug the expression x = g(z)into the optimization objective (6) to express the dependence in terms of the latents z:

$$\mathcal{L}(\mathbf{f},\mathbf{W},\beta)=\mathbb{E}_{(\mathbf{z},I)}\left[-\ln\frac{e^{\beta(\mathbf{w}_{I}\cdot\mathbf{f}\circ\mathbf{g}(\mathbf{z}))}}{\sum_{j\in\mathscr{S}}e^{\beta(\mathbf{w}_{j}\cdot\mathbf{f}\circ\mathbf{g}(\mathbf{z}))}}\right]=\mathcal{L}_{\mathbf{z}}(\mathbf{f}\circ\mathbf{g},\mathbf{W},\beta),\tag{8}$$

where the optimization is still over f (and not h = f ◦ g).

We note that the generator g is, by assumption, continuously invertible on the *compact* set S
d−1.

Therefore, its image g(S
d−1) is compact, too, and its inverse g
−1is also continuous. By Tietze's extension theorem (Wikipedia, 2024b), g
−1can be continuously extended to a function F : R
D →
S

d−1. Therefore, any continuous function h : S
d−1 → R
dcan take the role of f ◦ g by substituting f = h ◦ F continuous, since now f ◦ g = h ◦ (F ◦ g) = h ◦ idS

d−1 = h.

Hence, minimizing Lz(f ◦ g,W, β) (and by extension L(f,W, β)) for continuous f equates to minimizing Lz(h,W, β) for continuous h:

$$\mathcal{L}_{x}(\mathbf{h},\mathbf{W},\beta)=\mathbb{E}_{(\mathbf{z},I)}\left[-\ln\frac{e^{\beta\langle\mathbf{w}_{I},\mathbf{h}(\mathbf{z})\rangle}}{\sum_{j\in\mathscr{I}}e^{\beta\langle\mathbf{w}_{j},\mathbf{h}(\mathbf{z})\rangle}}\right].\tag{9}$$

Expressing the condition for global optimality of the objective: We rewrite the objective (9) by 1) using the indicator variable δI=i of the event {I = i} and 2) applying the law of total expectation:

$$\mathcal{L}_{\mathbf{z}}(\mathbf{h},\mathbf{W},\beta)=\mathbb{E}_{(\mathbf{z},\,I)}\left[-\sum_{i\in\mathscr{I}}\delta_{I=i}\ln\frac{e^{\beta\langle\mathbf{w}_{i},\mathbf{h}(\mathbf{z})\rangle}}{\sum_{j\in\mathscr{I}}e^{\beta\langle\mathbf{w}_{j},\mathbf{h}(\mathbf{z})\rangle}}\right]\tag{10}$$ $$=\mathbb{E}_{\mathbf{z}}\left[\mathbb{E}_{I}\left[-\sum_{i\in\mathscr{I}}\delta_{I=i}\ln\frac{e^{\beta\langle\mathbf{w}_{i},\mathbf{h}(\mathbf{z})\rangle}}{\sum_{j\in\mathscr{I}}e^{\beta\langle\mathbf{w}_{j},\mathbf{h}(\mathbf{z})\rangle}}\left|\ \mathbf{z}\right.\right]\right].\tag{11}$$

Using the properties that E-A f(B)B= E-ABf(B) and that E[δI=i] = P(I = i), we conclude

that: Lz(h,W, β) = Ez "−X i∈I EI δI=iln e β⟨wi,h(z)⟩ Pj∈I e β⟨wj ,h(z)⟩  z #(12) = Ez − X i∈I EI hδI=i z iln e β⟨wi,h(z)⟩ Pj∈I e β⟨wj ,h(z)⟩ (13) = Ez −X i∈I P(I = i|z) ln e β⟨wi,h(z)⟩ Pj∈I e β⟨wj ,h(z)⟩ . (14) By Gibbs' inequality (Wikipedia, 2024a), the cross-entropy inside the expectation is globally mini-
$$(13)$$
$$(14)$$

$$(15)$$
$$(16)$$
$$(17)$$
$$(18)$$
mized if and only if
= P(I = i|z), for any i ∈ I . (15)
$e^{\beta\langle\mathbf{w}_{i},\mathbf{h}(\mathbf{z})\rangle}$  $\sum_{j\in\mathscr{I}}e^{\beta\langle\mathbf{w}_{j},\mathbf{h}(\mathbf{z})\rangle}=\mathbb{P}(I=i|\mathbf{z}),\quad\text{for any}i\in\mathscr{I}.$  expectation is globally minimized if and only if the above 
Moreover, the entire expectation is globally minimized if and only if the above equality (15) holds
almost everywhere for z ∈ S
d−1.
Using that instance label I is uniformly distributed, or P(I = j) = P(I = i), the likelihood of the
sample being in class i can be expressed via Bayes' theorem as: P(I = i|z) = p(z|I = i)P(I = i) Pj∈I p(z|I = j)P(I = j) =p(z|I = i) Pj∈I p(z|I = j) Substituting (16) into (15) yields that for any i ∈ I and almost everywhere w.r.t. z ∈ S d−1: e β⟨wi,h(z)⟩ Pj∈I e β⟨wj ,h(z)⟩ =p(z|I = i) Pj∈I p(z|I = j) . (17) We now divide the equation (17) for the probability of a sample having label i with that of having
. (16)
label k and take the logarithm. This yields that Lz(h,W, β) is globally minimized if and only if

$$\beta\langle\mathbf{w}_{i}-\mathbf{w}_{k},\mathbf{h}(\mathbf{z})\rangle=\ln\frac{p(\mathbf{z}|I=i)}{p(\mathbf{z}|I=k)}$$  holds for any $i,k\in\mathscr{I}$ and almost everywhere w.r.t. $\mathbf{z}\in\mathbb{S}^{d-1}$.  
p(z|I = k)(18)
Plugging in the vMF distribution: Plugging the assumed conditional distribution from (4) into (18) yields the equivalent expression:
β⟨wi − wk, h(z)⟩ = κ⟨vC(i) − vC(k), z⟩, (19)
which holds for any *i, k* ∈ I and almost everywhere w.r.t. z ∈ S
d−1. Since h is continuous, the equation holds almost everywhere w.r.t. z if and only if it holds for all z ∈ S
d−1.

Observe that if h = id|S
d−1 , wi = vC(i)for any i ∈ I , and β = κ, then the equation is satisfied.

Thus, we can conclude that the global minimum of the cross-entropy loss is achieved. Step 2: Solving the equation for h,W **and proving identifiability.** We now find all solutions to prove the identifiability of the latent variables and that of the cluster vectors. Denote w˜ i =
β κwito simplify the above equation to:
⟨w˜ i − w˜ k, h(z)⟩ = ⟨vC(i) − vC(k), z⟩. (20)
h **is injective and has full-dimensional image:** We prove that h is injective. Assume that h(z1) = h(z2) for some z1, z2 ∈ S
d−1. Plugging z1 and z2 into (20) and subtracting the two equations yields:
0 = ⟨w˜ i − w˜ k, h(z1) − h(z2)⟩ = ⟨vC(i) − vC(k), z1 − z2⟩, (21)
for any i, k. However, as the cluster vectors {vc|c} form an affine generator system, the vectors
{vC(i) − vC(k)|*i, k*} form a generator system of R
d(see Lem. 1). Therefore, ⟨y, z1 − z2⟩ = 0, for any y ∈ R
d, which holds if and only if z1 = z2. Hence, h is injective.

By the Borsuk-Ulam theorem, for any continuous map from S
d−1to a space of dimensionality at most d−1 there exists some pair of antipodal points that are mapped to the same point. Consequently, no such function can be injective at the same time. Since h : S
d−1 → R
dis injective, the linear span of its image must be R
d.

$${\bf\Phi}_{(1)}-{\mathbf{v}}_{C(k)},{\mathbf{z}}\rangle.$$

Collapse of wi's: We prove that w˜ i = w˜ k if C(i) = C(k), i.e., samples from the same cluster will
have equal rows of W associated with them.
Assume that C(i) = C(k) and substitute them into (20):
$$\langle\bar{\mathbf{w}}_{i}-\bar{\mathbf{w}}_{k},\mathbf{h}(\mathbf{z})\rangle=0\quad\text{for any}\mathbf{z}\in\mathbb{S}^{d-1}.$$
d−1. (22)
However, we have just seen that the linear span of the image of h is R
d, which implies that w˜ i = w˜ k.
We may abuse our notation by setting w˜ c = w˜ iif C(i) = c, which yields a new form for (20):
for any *a, b* ∈ C and any z ∈ S
d−1.
 Using $\pmb{w}_c=\pmb{w}_i$ if $\mathbb{C}(i)=c$, which yields a new form:  $\langle\pmb{w}_a-\pmb{w}_b,\pmb{h}(\pmb{z})\rangle=\langle\pmb{v}_a-\pmb{v}_b,\pmb{z}\rangle$,  $-1$. 
Linear transformation from va − vb to w˜ a − w˜ b: We now prove the existence of a linear map A on R
dsuch that A(va − vb) = w˜ a − w˜ b for any *a, b* ∈ C . For this, we prove that the following mapping is well-defined:

$$(22)$$
$$\mathcal{A}:\sum_{a,b\in\mathcal{C}}\lambda_{a b}(\mathbf{v}_{a}-\mathbf{v}_{b})\mapsto\sum_{a,b\in\mathcal{C}}\lambda_{a b}({\tilde{\mathbf{w}}}_{a}-{\tilde{\mathbf{w}}}_{b}).$$
$$(24)$$

Since the system {va − vb|*a, b*} is not necessarily linearly independent, we have to prove that the mapping is independent of the choice of the linear combination. More precisely if for some coefficients λab, λ′ab

$$\sum_{a,b\in\mathcal{C}}\lambda_{a b}(\mathbf{v}_{a}-\mathbf{v}_{b})=\sum_{a,b\in\mathcal{C}}\lambda_{a b}^{\prime}(\mathbf{v}_{a}-\mathbf{v}_{b})$$

holds, then it should be implied that

$$\sum_{a,b\in\mathcal{C}}\lambda_{a b}(\tilde{\mathbf{w}}_{a}-\tilde{\mathbf{w}}_{b})=\sum_{a,b\in\mathcal{C}}\lambda_{a b}^{\prime}(\tilde{\mathbf{w}}_{a}-\tilde{\mathbf{w}}_{b}).$$
ab(w˜ a − w˜ b). (26)
Assume that (25) holds. Then, the difference of the two sides is:  $$0=\sum_{a,b\in\mathscr{C}}(\lambda_{ab}-\lambda_{ab}^{\prime})(\mathbf{v}_{a}-\mathbf{v}_{b}).$$
$$(25)$$
$$(26)$$
$$(27)$$

$$(28)$$
Taking the scalar product with an arbitrary $\mathbf{z}\in\mathbb{S}^{d-1}$ and using the linearity of the scalar product gives us:  $$0=\langle\sum_{a,b\in\mathscr{C}}(\lambda_{ab}-\lambda^{\prime}_{ab})(\mathbf{v}_{a}-\mathbf{v}_{b}),\mathbf{z}\rangle=\sum_{a,b\in\mathscr{C}}(\lambda_{ab}-\lambda^{\prime}_{ab})\langle\mathbf{v}_{a}-\mathbf{v}_{b},\mathbf{z}\rangle.\tag{28}$$

Now using (23) yields:

$$0=\sum_{a,b\in\mathcal{C}}(\lambda_{a b}-\lambda_{a b}^{\prime})\langle\hat{\mathbf{w}}_{a}-\hat{\mathbf{w}}_{b},\mathbf{h}(\mathbf{z})\rangle=\langle\sum_{a,b\in\mathcal{C}}(\lambda_{a b}-\lambda_{a b}^{\prime})(\hat{\mathbf{w}}_{a}-\hat{\mathbf{w}}_{b}),\mathbf{h}(\mathbf{z})\rangle.$$

However, the linear span of the image of h is R
d, which implies that

$$(29)$$
$$\sum_{a,b\in{\mathcal{C}}}(\lambda_{a b}-\lambda_{a b}^{\prime})({\bar{\mathbf{w}}}_{a}-{\bar{\mathbf{w}}}_{b})=0,$$
$$(30)$$
′ab)(w˜ a − w˜ b) = 0, (30)
equivalent to (26). Therefore, the mapping is well-defined and the linearity of A follows. h **is linear:** Equation (23) becomes:

$$\langle{\cal A}(\mathbf{v}_{a}-\mathbf{v}_{b}),\mathbf{h}(\mathbf{z})\rangle=\langle\mathbf{v}_{a}-\mathbf{v}_{b},\mathbf{z}\rangle,\tag{31}$$  $\mathbb{S}^{d-1}$. Nevertheless, $\{\mathbf{v}_{a}-\mathbf{v}_{b}|a,b\in\mathscr{C}\}$ is a generator system of $\mathbb{R}^{d}$.  
for any *a, b* ∈ C and any z ∈ S
d,
and, hence, (31) is equivalent to
d−1. (32)
$\langle\mathcal{A}\boldsymbol{y},\boldsymbol{h}(\boldsymbol{z})\rangle=\langle\boldsymbol{y},\boldsymbol{z}\rangle,\quad\text{for any}\boldsymbol{y}\in\mathbb{R}^d\text{and any}\boldsymbol{z}\in\mathbb{S}^{d-1}.$  valent to 
This is further equivalent to
⊤h(z)⟩ = ⟨y, z⟩. (33)
Since y is arbitrary, we conclude that A⊤h(z) = z for any z ∈ S
d−1. Therefore A is an invertible
transformation and h = (A⊤)
−1is linear.
$\langle\mathbf{y},\mathcal{A}^\top\mathbf{h}(\mathbf{z})\rangle=\langle\mathbf{y},\mathbf{z}\rangle$.  $\mathbf{t},\mathcal{A}^\top\mathbf{h}(\mathbf{z})=\mathbf{z}$ for any $\mathbf{z}\in\mathbb{S}^{d-1}$.  
$$(31)$$
$$(32)$$
$\eqref{eq:walpha}$. 
Proving Thm. 1C case C4: We have shown that h is linear. Furthermore, from (31) it follows, by fixing b and defining ψ = Avb − wb, that

$${\tilde{\mathbf{w}}}_{a}={\mathcal{A}}\mathbf{v}_{a}+\mathbf{\psi},\quad{\mathrm{for~any~}}a\in{\mathcal{C}},$$
$$(34)$$

w˜ a = Ava + ψ, for any a ∈ C , (34)
which proves case C4 of Thm. 1C. Proving Thm. 1C case C2: As a special case of the previous one, now we assume that h(z)
is unit-normalized and maps S
d−1to S
d−1. That amounts to h = (A⊤)
−1 being linear, normpreserving, and therefore orthogonal. Consequently A is also orthogonal, h = A and (34) simplifies to βκwa = w˜ a = Ava + ψ = h(va) + ψ, which proves C2 of Thm. 1C.

Proving Thm. 1C case C1: We now assume that both h and wi's are unit-normalized. Consequently, h = A is orthogonal linear and wa =
κ βAva + ψ.

Therefore, on one hand, the wa's lie on a d-dimensional hypersphere of radius κ β and center ψ. On the other hand, by definition, wa's also lie on the unit hypersphere S
d−1.

Since the system {wa|a ∈ C } is the bijective affine linear image of the affine generator system {va|a ∈ C }, {wa|a ∈ C } is also an affine generator system (Lem. 1). Consequently, there could be at most one hypersphere in R
d which contains all the wa's. Hence κβ = 1, ψ = 0, and wa = h(va),
which proves C1 of Thm. 1C.

Proving Thm. 1C case C3: Finally, we assume that wi's are unit-normalized. As this is a special case of Thm. 1C C4, we know that there exists a constant vector ψ such that:

$$\mathbf{w}_{a}={\frac{\kappa}{\beta}}\mathcal{A}\mathbf{v}_{a}+\mathbf{\psi},$$
$$(35)$$

for any a ∈ C . We are going to prove that O =
κ βA is orthogonal and ψ = 0.

Let O = U
⊤ΣV be the singular value decomposition (SVD) of O. Premultiplying with U yields:

$${\mathcal{U}}\mathbf{w}_{a}=\Sigma{\mathcal{V}}\mathbf{v}_{a}+{\mathcal{U}}\psi.$$
Uwa = ΣVva + Uψ. (36)
As orthogonal transformations U and V keep their arguments unit-normalized and {Vva − Vvb} is still an affine generator system (Lem. 1), we may assume without the loss of generality that

$\mathbf{v}$
$$\mathbf{w}_{a}=\Sigma\mathbf{v}_{a}+\psi,$$
$$(37)$$

$$\mathrm{normalized.}$$
wa = Σva + ψ, (37)
for any a ∈ C , where all va's and wa's are unit-normalized. Let us assume that ψ ̸= 0. In that case both sides of (37) can be scaled such that the offset ψ has unit norm. In this case wa's are no longer on the unit hypersphere, but they instead have a mutual norm r. Assuming that the diagonal elements of Σ are σ = (σ1*, . . . , σ*d), this is equivalent to:

$$r^{2}=\left\|\Sigma\mathbf{v}_{a}+\mathbf{\psi}\right\|^{2}=\left\|\Sigma\mathbf{v}_{a}\right\|^{2}+2\langle\Sigma\mathbf{v}_{a},\mathbf{\psi}\rangle+\left\|\mathbf{\psi}\right\|^{2}$$ $$=\left\langle\mathbf{v}_{a}\odot\mathbf{v}_{a},\mathbf{\sigma}\odot\mathbf{\sigma}\right\rangle+\left\langle\mathbf{v}_{a},2\mathbf{\sigma}\odot\mathbf{\psi}\right\rangle+1,$$

where [x ⊙ y]i = xiyiis the elementwise product. Eq. (39) is equivalent to the following:

$$(\mathbf{v}_{a}\odot\mathbf{v}_{a})^{\top}(\mathbf{\sigma}\odot\mathbf{\sigma})+\mathbf{v}_{a}^{\top}(2\mathbf{\sigma}\odot\mathbf{\psi})-r^{2}=-1.$$
2 = −1. (40)
Collecting the equations for all a ∈ C yields:

$${\cal D}\left(\begin{array}{l}{{\sigma\stackrel{{(\circ)}}{{\longrightarrow}}\sigma}}\\ {{2\sigma\stackrel{{(\circ)}}{{\longrightarrow}}\psi}}\\ {{r^{2}}}\end{array}\right)=-{\bf1}_{|\mathscr{C}|},$$
$$(38)$$

where D is the following |C | × (2d + 1) matrix:

$${\mathcal{D}}=\left(\begin{array}{l l l l}{{\cdots\cdots\cdots}}&{{\cdots\cdots\cdots}}&{{\cdots}}\\ {{(\mathbf{v}_{a}\odot\mathbf{v}_{a})^{\top}}}&{{\mathbf{v}_{a}^{\top}}}&{{-1}}\\ {{\cdots\cdots\cdots}}&{{\cdots\cdots\cdots}}&{{\cdots\cdots}}\end{array}\right)\,.$$
$$(40)^{\frac{1}{2}}$$
$$(41)$$

$$(42)$$
 . (42)