---

# Hide & Seek: Transformer Symmetries Obscure Sharpness & Riemannian Geometry Finds It

---

Marvin F. da Silva<sup>1,2</sup> Felix Dangel<sup>2</sup> Sageev Oore<sup>1,2</sup>

## Abstract

The concept of sharpness has been successfully applied to traditional architectures like MLPs and CNNs to predict their generalization. For transformers, however, recent work reported weak correlation between flatness and generalization. We argue that existing sharpness measures fail for transformers, because they have much richer symmetries in their attention mechanism that induce directions in parameter space along which the network or its loss remain identical. We posit that sharpness must account fully for these symmetries, and thus we redefine it on a quotient manifold that results from quotienting out the transformer symmetries, thereby removing their ambiguities. Leveraging tools from Riemannian geometry, we propose a fully general notion of sharpness in terms of a geodesic ball on the symmetry-corrected quotient manifold. In practice, we need to approximate the geodesics. Doing so up to first order yields existing adaptive sharpness measures, and we demonstrate that including higher-order terms is crucial to recover correlation with generalization. We present results on diagonal nets with synthetic data and show that our geodesic sharpness reveals strong correlation with generalization for real-world transformers on both text and image classification tasks.

## 1. Introduction

Predicting generalization of neural nets (NNs)—the discrepancy between training and test set performance—remains an open challenge. Generalization-predictive metrics are valuable though: they enable explicit regularization of training to enhance generalization (Foret et al., 2021), and provide

broader theoretical insights into generalization itself.

There is a long history of hypotheses linking sharpness to generalization, but evidence has been conflicting (Hochreiter & Schmidhuber, 1994; Andriuschenko et al., 2023). Generalization has been speculated as correlating with flatness, but recent evidence has indicated that, in the case of transformers, it has little to no correlation whatsoever. Measures of sharpness have varied widely, ranging from trace of the Hessian to worst-case loss within a local neighborhood, with adaptive and relative variations proposed to address specific challenges (Kwon et al., 2021; Petzka et al., 2021).

We suspect that some of the confusion stems from the specificity of the problem these measures have attempted to address: the issue of parameter rescaling. In contrast, we argue that rescaling (Dinh et al., 2017) is merely a special case of a broader, more fundamental obstacle to measuring sharpness accurately: the presence of full and continuous parameter symmetries. Addressing this challenge is crucial to ensure that we are studying the right quantity when investigating the relationship between sharpness and generalization.

Beyond discrete permutation symmetries, neural nets naturally exhibit continuous symmetries in their parameter space. These symmetries are intrinsic, data-independent properties that emerge from standard architectural components. For example: normalization layers (Ioffe & Szegedy, 2015; Ba et al., 2016; Wu & He, 2018) induce scale invariance on the pre-normalization weights (Salimans & Kingma, 2016); homogeneous activation functions like ReLU introduce re-scaling symmetries between pre- and post-activation weights (Dinh et al., 2017); some normalization layers and softmax impose translation symmetries in the preceding layer’s biases (Kunin et al., 2021). As a result, arguably almost any NN, along with its corresponding loss, exhibit symmetries and can therefore represent the *same* function using *different* parameter values (Figure 1a).

Adaptive flatness (Kwon et al., 2021) accounts for some symmetries, both element- and filter-wise re-scaling, but fails to capture the attention mechanism’s *full* symmetry, represented by  $GL(h)$  (re-scaling by invertible  $h \times h$  matrices, where  $h$  is the hidden dimension), as we will discuss later. Aiming to break the cycle between discovery of a

<sup>1</sup>Faculty of Computer Science, Dalhousie University, Halifax, Canada<sup>2</sup>Vector Institute for Artificial Intelligence, Toronto, Canada. Correspondence to: Marvin F. da Silva <marvinf.silva@dal.ca>.

*Proceedings of the 4<sup>th</sup> International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

![](_page_1_Figure_1.jpeg)

Figure 1: Quantities from the Riemannian quotient manifold respect the loss landscape's symmetry; Euclidean quantities do not. We illustrate this here for a synthetic least squares regression task with a two-layer NN, where x 7→ θ2θ1x with scalar parameters θ ∈ <sup>R</sup> 2 and input x ∈ <sup>R</sup> (i.e. each layer is a linear function). The NN is re-scale invariant, i.e. has GL(1) symmetry: For any α ∈ <sup>R</sup> \ {0}, the parameters (θ ′ 1 , θ′ 2 ) = (α −1 θ1, αθ2) represent the same function. [\(a\)](#page-1-0) The loss function inherits this symmetry and has hyperbolic level sets. [\(b\)](#page-1-0) The Euclidean gradient norm does not share the loss function's geometry and changes throughout an orbit where the NN function remains constant. [\(c\)](#page-1-0) The Riemannian gradient norm follows the loss function's symmetry and remains constant throughout an orbit, i.e., it does not suffer from ambiguities for two points in parameter space that represent the same NN function.

specific symmetry and techniques to deal with it, we ask:

*Can we provide a one-size-fits-many recipe for developing symmetry-invariant quantities for a wider range of continuous symmetries?*

Here, we positively answer this question by proposing a principled approach to eliminate ambiguities stemming from symmetry. Essentially, this boils down to using the geometry that correctly captures symmetry-imposed parameter equivalences. We apply concepts from Riemannian geometry to work on the Riemannian quotient manifold implied by a symmetry group [\(Boumal,](#page-8-3) [2023,](#page-8-3) §9). We thus identify objects on the quotient manifold—like the Riemannian metric and gradient—and show how to translate them back to the Euclidean space. Our contributions are the following:

- (a) We introduce the application of Riemannian geometry [\(Boumal,](#page-8-3) [2023\)](#page-8-3) to the study of NN parameter space symmetries by using geometry from the quotient manifold induced by a symmetry as a general recipe to remove symmetry-induced ambiguities in parameter space. We do so by translating concepts like gradients from the quotient manifold back to the original space through *horizontal lifts*.
- (b) Based on [\(a\),](#page-1-1) we propose and analyze *geodesic sharpness*, a novel adaptive sharpness measure: By Taylorexpanding our refined geometry, we show that (i) symmetries introduce curvature into the parameter space, which (ii) results in previous adaptive sharpness measures when ignored. Geodesic sharpness differs from traditional sharpness measures in two key aspects: (i) the norm of the perturbation parameter is redefined to reflect the underlying geometry; (ii) perturbations

follow geodesic paths in the quotient manifold rather than straight lines in the ambient space.

- (c) For diagonal nets, we analytically solve *geodesic sharpness* and find a strong correlation with generalization. Then, we apply our approach to the unstudied and higher-dimensional GL(h) symmetry in the attention mechanism. On both large vision transformers and language models, we empirically find stronger correlation than any previously seen (that we are aware of) between our geodesic sharpness and generalization.

# 2. Related Work

Symmetry versus reparameterization: [Kristiadi et al.](#page-9-8) [\(2023\)](#page-9-8) pointed out how to fix ambiguities stemming from reparameterization, i.e. a change of variables to a *new* parameter space: Invariance under reparameterization follows by correctly transforming the (often implicitly treated) Riemannian metric into the new coordinates. Our work focuses on invariance of the parameter space M under a symmetry group G with action ψ : G × M → M, (g, θ) 7→ ψ(g, θ) that operates on a *single* parameter space.

Symmetry teleportation: Another ways to use symmetryimplied ambiguity is to view it as a degree of freedom and develop adaptation heuristics to improve algorithms which are not symmetry-agnostic [\(Zhao et al.,](#page-9-9) [2022a\)](#page-9-9).

Geometric constraints & NN dynamics: Previous studies analyze how parameter space symmetries impose geometric constraints on derivatives and introduce conserved quantities during training [\(Kunin et al.,](#page-9-7) [2021\)](#page-9-7). Our approach

differs by systematically removing symmetry-induced ambiguity through quotienting out the the symmetry group.

We generalize earlier post-hoc solutions for simpler symmetries (e.g., GL(1)) to more complex, higher-dimensional symmetries such as GL(h), common in neural network attention mechanisms. Unlike [Kunin et al.](#page-9-7) [\(2021\)](#page-9-7), who consider geometry in augmented spaces for simpler symmetries, we directly use the quotient space geometry. Objects are then 'lifted' back into the original space, yielding symmetrycorrected quantities. This method provides a principled framework capable of handling high-dimensional symmetries, leading to a more effective dimensionality reduction.

Quotient manifolds in deep neural networks: [Ranga](#page-9-10)[mani et al.](#page-9-10) [\(2019\)](#page-9-10) introduce a quotient manifold construction for re-scaling symmetries and then use the Riemannian spectral norm as a measure of worst-case flatness. This differs from our approach in several ways:

- (a) Our approach is more general and contains both the GL(h) symmetry of transformers, and the original rescaling/scaling symmetry of CNNs/MLPs, rendering it applicable to a wider range of modern architectures.
- (b) Our experimental setup is more challenging: we test on large-scale models (large transformers vs CNNs) and large-scale datasets (ImageNet vs CIFAR-10). Sharpness measures that account for re-scaling/scaling symmetries (e.g. adaptive sharpness) work quite well on CIFAR-10 with CNNs, and tends to break down on datasets like ImageNet with transformers.
- (c) Conceptually, [Rangamani et al.](#page-9-10) [\(2019\)](#page-9-10) defines worstcase sharpness on the usual norm-ball, appropriately generalized to the Riemannian setting. We propose instead that the ball should be the one traced out by geodesics, to better respect the underlying geometry.
- (d) Performance-wise, our approach is cheaper as it does not use the Hessian and only uses symmetry-corrected gradients (see [Dagreou et al.](#page-8-4) ´ [\(2024\)](#page-8-4) for an in-depth cost comparison of computing Hessians vs gradients).

Relative sharpness: Another promising approach to sharpness was proposed by [Petzka et al.](#page-9-3) [\(2021\)](#page-9-3), where the generalization gap is shown to admit a decomposition into a representativeness term and a feature robustness term. Focusing on the feature robustness term, they introduce relative sharpness, which is invariant to a layer- and neuron-wise re-scaling, and performs better than traditional sharpness measures [\(Adilova et al.,](#page-8-5) [2023;](#page-8-5) [Walter et al.,](#page-9-11) [2025\)](#page-9-11).

#### 3. Preliminary Definitions, Notation & Math

Generalization measures: We consider a neural net f<sup>θ</sup> with parameters θ ∈ <sup>R</sup> d that is trained on a data set <sup>D</sup>train using a loss function ℓ by minimizing the empirical risk

$$L_{\mathbb{D}_{\text{train}}}(\boldsymbol{\theta}) := \frac{1}{|\mathbb{D}_{\text{train}}|} \sum_{(\boldsymbol{x}, \boldsymbol{y}) \in \mathbb{D}_{\text{train}}} \ell(f_{\boldsymbol{\theta}}(\boldsymbol{x}), \boldsymbol{y}).$$

Our goal is to compute a quantity on the training data that is predictive of the network's generalization, i.e. performance on a held-out data set.

Sharpness: A popular way to predict generalization is via sharpness—i.e., how much the loss changes when perturbing the weights in a small neighbourhood—like average (Savg) or worst-case sharpness (Smax) [\(Keskar et al.,](#page-9-12) [2017\)](#page-9-12)

$$S_{\text{avg}} = \mathbb{E}_{\mathbb{S}} [L_{\mathbb{S}}(\boldsymbol{\theta} + \boldsymbol{\delta}) - L_{\mathbb{S}}(\boldsymbol{\theta})], \quad \boldsymbol{\delta} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\rho}^2 \mathbf{I}), \\ S_{\text{max}} = \mathbb{E}_{\mathbb{S}} \left[ \max_{\|\boldsymbol{\delta}\| \leq \rho} (L_{\mathbb{S}}(\boldsymbol{\theta} + \boldsymbol{\delta}) - L_{\mathbb{S}}(\boldsymbol{\theta})) \right],$$

with batches <sup>S</sup> ∼ <sup>D</sup>train of size |S| <sup>=</sup> <sup>m</sup>, neighbourhood size ρ, and perturbation δ. Near critical points, they closely relate to the Hessian H (and thus parameter space curvature): Savg ∝ Tr(H), and Smax ∝ <sup>λ</sup>max(H).

Adaptive sharpness: Hessian-based sharpness measures can be made to assume arbitrary values by rescaling parameters, even though the NN function stays the same. To fix this inconsistency, [Kwon et al.](#page-9-2) [\(2021\)](#page-9-2) proposed adaptive sharpness (invariant under special symmetries), and [An](#page-8-0)[driushchenko et al.](#page-8-0) [\(2023\)](#page-8-0) use adaptive notions of sharpness that are invariant to element-wise scaling,

$$S_{\max}^{\text{ad}}(\mathbf{w}, \mathbf{c}) = \mathbb{E}_{\mathbf{S}} \left[ \max_{\|\boldsymbol{\delta} \odot \mathbf{c}\|_2 \leq \rho} L_{\mathbf{S}}(\boldsymbol{\theta} + \boldsymbol{\delta}) - L_{\mathbf{S}}(\boldsymbol{\theta}) \right], \quad (1)$$

with scaling vector c (usually set to |θ|, [Kwon et al.,](#page-9-2) [2021\)](#page-9-2).

The problem: Adaptive sharpness only considers the symmetry induced by element-wise re-scaling. But symmetries of transformers go beyond the invariance that adaptive sharpness captures. Maybe unsurprisingly, [Andriushchenko et al.](#page-8-0) [\(2023\)](#page-8-0) find inconsistent trends for adaptive sharpness in transformers, with sharpness failing to correlate with generalisation, versus other architectures. We hypothesize this is related to adaptive sharpness not accounting for the full symmetry in transformers. In this paper, we address this. The central question is: *If adaptive sharpness is the fix for a special symmetry, can we provide a more general solution for the symmetries of transformers, to fix the above inconsistency?*

#### 3.1. Symmetries in Neural Networks

Here, we give a brief overview and make the notion of NN symmetries more concrete, focusing on those studied by [Kunin et al.](#page-9-7) [\(2021\)](#page-9-7). Those symmetries lead to rather small effective dimensionality reduction as they are often of GL(1) or GL<sup>+</sup>(1), but they can still impact the network behaviour. Let θ denote the parameters of a neural net, 1<sup>A</sup> a binary mask, and 1¬A its complement such that their sum is a vector of ones, 1<sup>A</sup> +1¬A = 1. Let θ<sup>A</sup> := θ ⊙1<sup>A</sup> with ⊙ the element-wise product. Further, let A1,<sup>2</sup> be two disjoint subsets, A<sup>1</sup> ∩ A<sup>2</sup> <sup>=</sup> ∅ with masks <sup>1</sup>A<sup>1</sup> , 1A<sup>2</sup> . Then we have the following common symmetries, characterized by their symmetry group G, such that for any g ∈ G the parameters ψ(g, θ) and θ represent the same function:

- Translation: ψ(α, θ) = 1<sup>A</sup> ⊙ α + θ with α ∈ <sup>R</sup> h
- Scaling: ψ(α, θ) = αθ<sup>A</sup> + θ¬A with α ∈ <sup>R</sup>><sup>0</sup>
- Re-scaling: <sup>ψ</sup>(α, <sup>θ</sup>) = <sup>α</sup>θA<sup>1</sup> <sup>+</sup> <sup>1</sup>/αθA<sup>2</sup> <sup>+</sup> <sup>θ</sup>¬(A1∨A2) with α ∈ <sup>R</sup>><sup>0</sup>

Their associated groups are G = <sup>R</sup> h , GL<sup>+</sup>(1), GL<sup>+</sup>(1). In practice, there may be multiple symmetries acting onto disjoint parameter sub-spaces. Note that re-scaling is essentially the symmetry that adaptive sharpness corrects for.

#### 3.2. Rescale Symmetry of Transformers

Transformers exhibit a higher-dimensional symmetry than the previous examples; we formalize the treatment of this symmetry in the following canonical form.

Definition 3.1 (Functional GL-symmetric building block). Consider a function f(G, H) on <sup>R</sup> <sup>m</sup>×<sup>h</sup> × <sup>R</sup> n×h that consumes two matrices G ∈ <sup>R</sup> n×h , H ∈ R <sup>m</sup>×<sup>h</sup> but only uses the product GH<sup>⊤</sup>, i.e. f(G, H) = g(GH<sup>⊤</sup>) for some g over <sup>R</sup> <sup>m</sup>×n. f is symmetric under the *general linear group*

$$\text{GL}(h) := \{ \mathbf{A} \in \mathbb{R}^{h \times h} \mid \mathbf{A} \text{ invertible} \}$$

with dim(GL(h)) = h 2 and action

$$\psi(\mathbf{A}, (\mathbf{G}, \mathbf{H})) = (\mathbf{G}\mathbf{A}^{-1}, \mathbf{H}\mathbf{A}^\top). \quad (3)$$

In other words, we can insert and then absorb the identity A−<sup>1</sup>A into G, H to obtain equivalent parameters GA−<sup>1</sup> , HA<sup>⊤</sup> that represent the same function.

Example [A.2](#page-10-0) illustrates GL symmetry for a shallow linear net. Indeed, many popular NN building blocks feature this form, most prominently the attention mechanism in transformers [\(Vaswani et al.,](#page-9-13) [2017\)](#page-9-13). We give the attention symmetry in Example [A.1,](#page-10-1) and we provide the symmetry for low-rank adapters [\(Hu et al.,](#page-9-14) [2022\)](#page-9-14) in Example [A.3.](#page-10-2) These examples are NN building blocks that introduce GL symmetries into a loss function and can all be treated through the

canonical form in Definition [3.1.](#page-3-0) In contrast to the symmetries from Section [3.1,](#page-3-1) they lead to more drastic dimensionality reduction. Consider for example a single self-attention layer where <sup>h</sup> <sup>=</sup> <sup>h</sup><sup>v</sup> <sup>=</sup> <sup>h</sup>k. The number of trainable parameters is 4h 2 and the two GL(d) symmetries reduce the effective dimension to 4h <sup>2</sup>−2 dim(GL(h)) = 2h , i.e. they render *half* the parameter space redundant. We hypothesize that the impact of a low-dimensional symmetry on objects like the Euclidean Hessian's trace [\(Dinh et al.,](#page-8-1) [2017\)](#page-8-1) may be amplified for such higher-dimensional symmetries.

#### 3.3. Mathematical Concepts for Riemannian Geometry

We now outline required properties of manifolds for the full development of our approach. We list essential concepts here, and provide definitions and a brief review Appendix [B.](#page-11-0) For further information, see for instance [Lee](#page-9-15) [\(2003\)](#page-9-15). Figure [2](#page-4-0) illustrates the main concepts we will require.

Ambient embedding space: We assume that the manifold of possible parameters is embedded in a linear Euclidean space E ≃ <sup>R</sup> <sup>d</sup> with d the number of parameters. We can think of E as the *ambient space*. For instance, for a loss function ℓ : E → <sup>R</sup>, θ 7→ ℓ(θ) , we can use ML libraries to evaluate its value, as well as its Euclidean gradient

$$\text{grad}_{\boldsymbol{\theta}} \bar{\bar{\ell}} = \left( \frac{\partial \bar{\bar{\ell}}(\boldsymbol{\theta})}{\partial \theta_i} \right)_{i=1, \dots, d} \in \mathbb{R}^d.$$

Because the geometry of E is flat, i.e. uses the standard metric ⟨θ1, θ2⟩ := θ ⊤ <sup>1</sup> <sup>θ</sup>2, this object consists of partial derivatives. However, the Riemannian generalization will add correction terms. In what follows we consider only the restriction of objects like ℓ to the parameter manifold.

Definition 3.2. We take M to be the manifold of network parameters, and consider it a sub-manifold embedded into E, the computational space of matrices on which all our numerical calculations are done. We call M the *total space*. On the total space we have a loss function ℓ : M → <sup>R</sup>.

Our goal is to calculate derivatives/geometric quantities after removing the NN's symmetries. The symmetry relation induces natural equivalence classes, which we write [θ], and explain in Appendix [B.1.](#page-11-1) We let M = M/ ∼ represent the *quotient* of the original parameter space manifold by the equivalence relation ∼ associated with the symmetry (Appendix [B.2\)](#page-11-2). We also require *tangent vectors*; these are straightforward on the total space M, but the tangent space of the quotient manifold, M, requires more machinery: *vertical* and *horizontal spaces*, and corresponding *lift*s. These concepts are all defined in Appendix [B.3.](#page-11-3)

Once we endow M with a smooth inner product over its tangent vectors, we obtain a *Riemannian manifold* (defined in Appendix [B.4\)](#page-11-4). This construction lets us analyze differ-

![](_page_4_Picture_19.jpeg)

Figure 2: **Illustrative sketch relating total and quotient space and their tangent spaces.** A tangent vector at a point in total space,  $\bar{\xi}_{\bar{x}} \in T_{\bar{x}}\overline{\mathcal{M}}$  can be decomposed into a horizontal component  $\bar{\xi}_{\bar{x}}^{\mathcal{H}}$  and a vertical component  $\bar{\xi}_{\bar{x}}^{\mathcal{V}}$ . The vertical component points along the direction where the quotient space  $x = [\bar{x}]$  remains unaffected. The horizontal component points along the direction that changes the equivalence class. We can use  $\bar{\xi}_{\bar{x}}^{\mathcal{H}}$  as a representation of the tangent vector  $\xi_x \in T_x\mathcal{M}$  on the quotient space. The component  $\bar{\xi}_{\bar{x}}^{\mathcal{H}}$  represents the *horizontal lift* of  $\xi_x$ .

ential objects that live on quotient manifolds, in the ambient space in a natural way. Furthermore, this allows us to define the horizontal space as the orthogonal complement of the vertical space (Appendix B.4), and to define a *Riemannian gradient* (Appendix B.5). Most properties from the Euclidean case still hold for the Riemannian gradient, but of particular interest to us is the fact that the direction  $\text{grad}f(x)$  is still the steepest-ascent direction of  $f$  at a point  $x$ .

We additionally make use of *geodesic curves*. Intuitively, geodesic curves can either be seen as curves of minimal distance between two points on a manifold  $\overline{\mathcal{M}}$ , or equivalently, as curves through a given point with some initial velocity, and whose acceleration is zero—a generalization of Euclidean straight lines. See Appendix B.6 for details.

Putting it all together, this gives us a *recipe* for computing quantities invariant to a given symmetry relation: (i) find a Riemannian metric compatible with this symmetry; (ii) determine the vertical space for the symmetry relation; (iii) use the metric to find the orthogonal complement of this vertical space, i.e. the projector into the horizontal space; (iv) find the horizontal geodesics. Combined, these steps allow us to do calculations in the quotient manifold along the proper paths (given by geodesics).

#### 4. Geodesic Sharpness

We posit that adaptive sharpness measures should take into account the geometry of the quotient parameter manifold that arises after removing symmetries from the parameter space. We base our sharpness measure on the notion of a *geodesic ball*: the set of points that can be reached by geodesics, starting at a point  $p$  and whose initial velocity has a norm smaller than  $\rho$ , after one time unit. In  $\mathbb{R}^d$  this

is just the usual definition of a ball, since the geodesics are straight lines. If  $\xi \in H_{\bar{x}}\overline{\mathcal{M}}$  is a horizontal vector, and  $\bar{\gamma}(t)$  is a geodesic starting at  $\theta$  and with initial velocity  $\xi$ :

$$S_{\max}^{\rho}(\mathbf{w}) = \mathbb{E}_{\mathbb{S}} \left[ \max_{\|\xi\|_{\bar{\gamma}(0)} \leq \rho} L_{\mathbb{S}}(\bar{\gamma}\xi(1)) - L_{\mathbb{S}}(\bar{\gamma}\xi(0)) \right]. \quad (4)$$

If the initial velocity,  $\xi$ , is a horizontal vector, then the velocity of the geodesic,  $\bar{\gamma}\xi$ , will stay horizontal. The choice of  $t = 1$  in  $\bar{\gamma}\xi(1)$  is not as arbitrary as it first seems (do Carmo, 1992): since for a positive  $a$ ,  $\bar{\gamma}\xi(at) = \bar{\gamma}_a\xi(t)$ , positions reached with arbitrary  $t$  can be reached by instead fixing  $t = 1$  and manipulating the initial velocity's norm via  $\rho$ .

When we do not have an analytical solution for the geodesic, we can use the approximation:

$$\bar{\gamma}_{\xi}^i(t) = \bar{\gamma}_{\xi}^i(0) + \bar{\xi}^i t - \frac{1}{2}\Gamma_{kl}^i \bar{\xi}^k \bar{\xi}^l t^2 + \mathcal{O}(\bar{\xi}^3), \quad (5)$$

where  $\bar{\xi} = (\bar{\xi}^i)$  is the initial (horizontal) velocity, and  $\Gamma_{kl}^i$  are the Christoffel symbols. We show that geodesic sharpness reduces to adaptive sharpness measures in Appendix F, under appropriate metric choices and by taking a first-order approximation to the geodesics, that is, ignoring the terms corresponding to the curvature,  $\Gamma_{kl}^i$ .

#### 5. Geodesic Sharpness in Practice

We now apply geodesic sharpness to concrete examples. A fully worked out scalar toy model is in Appendix D.

Following previous works by Dziugaite et al. (2020); Kwon et al. (2021); Andriushchenko et al. (2023), we use the Kendall rank correlation coefficient (Kendall, 1938) to assess the correlation between generalization and sharpness

in the empirical validations of our approach:

$$\tau(\mathbf{t}, \mathbf{s}) = \frac{2}{M(M-1)} \sum_{i < j} \text{sign}(t_i - t_j) \text{sign}(s_i - s_j),$$

where t and s are the vectors of observed variables between which we are measuring correlation.

Although the criterion of symmetry compatibility restricts the class of suitable metrics, these are not necessarily unique. As long as it is symmetry-compatible, we have no reason to prefer one metric over another, except for practical aspects like numerical cost and stability. We will present results on two symmetry-compatible metrics that are simple, yet nontrivial, and often used in the related literature on Riemannian optimization on fixed-rank matrix spaces [\(Luo et al.,](#page-9-17) [2023\)](#page-9-17).

#### 5.1. Diagonal Networks

We start by studying *diagonal linear nets*, one of the simplest non-trivial neural networks [\(Pesme et al.](#page-9-18) [\(2021\)](#page-9-18), [Wood](#page-9-19)[worth et al.](#page-9-19) [\(2020\)](#page-9-19)). These have two parameters, u, v, and predict a label, y, given an input, x, via y = x <sup>⊤</sup>(u ⊙ v). We consider linear regression with labels y ∈ <sup>R</sup> <sup>n</sup>, a data matrix X ∈ <sup>R</sup> n×d , and take as our loss L(u, v) = ∥X(u ⊙ v) − y∥ 2 2 . Our parameter manifold M is <sup>R</sup> <sup>d</sup> × <sup>R</sup> d .

The nets are symmetric under element-wise rescaling: (u, v) 7→ (αu, α−<sup>1</sup>v), leaves β= u ⊙ v and L invariant.

Metric: At a point (u, v) ∈ M, for two tangent vectors η = (ηu, ηv), <sup>ν</sup> = (νu, <sup>ν</sup>v) ∈ <sup>T</sup>(u,v)M, we use the following two symmetry-compatible metrics:

$$\langle \eta, \nu \rangle^{\text{inv}} := \sum_{i=1}^d \frac{\eta_u^i \nu_u^i}{(u^i)^2} + \frac{\eta_v^i \nu_v^i}{(v^i)^2}, \quad (6)$$

$$\langle \eta, \nu \rangle^{\text{mix}} := \sum_{i=1}^d \eta_u^i \nu_u^i (v^i)^2 + \eta_v^i \nu_v^i (u^i)^2. \quad (7)$$

Horizontal space: Both have the same horizontal space

$$\mathcal{H}_{(u,v)}\overline{\mathcal{M}} = \left\{ (\eta_u, \eta_v) \in T_{(u,v)}\mathcal{M} \mid \frac{\eta_u^i}{u^i} = \frac{\eta_v^i}{v^i} \quad \forall i \right\}.$$

Geodesics: With b<sup>i</sup> := η u <sup>u</sup><sup>i</sup> = η v v<sup>i</sup> , the geodesics are

$$\begin{aligned} \gamma_{\text{inv}}(t)^i &= (\mathbf{u}_0^i \exp(\mathbf{b}_i t), \mathbf{v}_0^i \exp(\mathbf{b}_i t)) , \\ \gamma_{\text{mix}}(t)^i &= \left( \mathbf{u}_0^i \sqrt{1+2\mathbf{b}_i t}, \mathbf{v}_0^i \sqrt{1+2\mathbf{b}_i t} \right) , \end{aligned}$$

with starting points u i 0 and v i 0 , i.e. the trained parameters.

![](_page_5_Figure_1.jpeg)

Figure 3: Adaptive vs. geodesic sharpness on diagonal nets. The generalization gap is the test loss (remember all models are trained to 10−<sup>5</sup> training loss). The correlation coefficient's magnitude is larger for geodesic sharpness.

The minimum norm least squares predictor is β<sup>∗</sup> := (X⊤X) <sup>−</sup><sup>1</sup>X⊤y = X⊤y. Using Equation [\(4\)](#page-4-1) (details in Appendix [E\)](#page-16-0), we get (to first and to second order)

$$S_{\max; \text{inv}}^{\rho}(\mathbf{u}, \mathbf{v}) = 4\rho \|\boldsymbol{\beta}_0 \odot (\boldsymbol{\beta}_0 - \boldsymbol{\beta}_*)\|_2 + 4\rho^2 \max [(\beta_0^i)^2] , \quad (8)$$

which depends on ρ and the difference between the learned, and the optimal minimum norm, predictor. Eq. [8](#page-5-0) is the square of adaptive sharpness (when the residual ∥β0⊙(β0− β∗)∥<sup>2</sup> is small) if very carefully chosen hyperparameters were used (by contrast, this result naturally appears using our geodesic approach). For the second metric, we have

$$S_{\max; \text{mix}}^\rho(\mathbf{u}, \mathbf{v}) = \rho \|\boldsymbol{\beta}_0 - \boldsymbol{\beta}_*\|_2.$$

### 5.1.1. EMPIRICAL VALIDATION

Experimental setup: We follow [Andriushchenko et al.](#page-8-0) [\(2023\)](#page-8-0), generate a randomly distributed data matrix X, a random ground-truth vector β ∗ that is 90% sparse, and train 50 diagonal networks to 10−<sup>5</sup> training loss on a regression task.

We focus on the more practically relevant case of overparametrization (d > n). One downside of this is that the theoretical expressions derived in the previous section, while a useful sanity check, no longer hold (since overparameterization breaks the assumption X⊤X = Id=200). To obtain our geometric sharpness, we directly solve Equation [\(4\)](#page-4-1).

Results: All three notions of sharpness are able to predict generalization (Figure [3\)](#page-5-1). Geodesic sharpness, although closely related for diagonal nets to adaptive worst-case sharpness, does slightly better. This applies to both metrics studied, and they perform roughly the same. See Section [7](#page-7-0) for comments about the sign of the correlation.

### 5.2. Attention Layers

R (n+m)h and we restrict weights to have full column rank:

Assumption 5.1. The rank of G, H corresponds to their number of columns, rank(G) = rank(H) = h.

This implies h ≤ n, m, which is usually satisfied in (multihead) attention layers (Example [A.1\)](#page-10-1) for the default choices of <sup>d</sup>v, dk. While the weights of multi-head attention tend to have high column rank [\(Yu & Wu,](#page-9-20) [2023\)](#page-9-20), they are not guaranteed to be full column rank. To account for this, we introduce a small relaxation parameter, ϵ, to the Gram matrices s.t. G⊤G → G⊤G + ϵIh. Empirically, we observe that as long as ϵ is sufficiently small, it does not affect our results (Appendix [H.2\)](#page-20-0). Therefore, we restrict both G, H to the set of fixed-rank matrices, M ← <sup>R</sup> n×h <sup>h</sup> × <sup>R</sup> m×h <sup>h</sup> where R n×h k := B ∈ R n×h | rank(B) = k . We can represent a point x¯ ∈ M by a matrix tuple (G, H) ∈ <sup>R</sup> n×h <sup>h</sup> ×<sup>R</sup> m×h h . Its tangent space Tx¯M is

$$T_{\bar{x}}\overline{\mathcal{M}} = \{\bar{\eta} = (\bar{\eta}_G, \bar{\eta}_H) \in \mathbb{R}^{n \times h} \times \mathbb{R}^{m \times h}\} ,$$

Metric: We endow <sup>M</sup> with the two metrics ⟨·, ·⟩inv,mix x¯ : Tx¯M × Tx¯M → <sup>R</sup> (proof they are valid in Appendix [I.1\)](#page-20-1):

$$\langle \bar{\eta}, \bar{\zeta} \rangle_{\bar{x}}^{\text{inv}} := \text{Tr} \left( (\mathbf{G}^\top \mathbf{G})^{-1} \bar{\eta}_G^\top \bar{\zeta}_G + (\mathbf{H}^\top \mathbf{H})^{-1} \bar{\eta}_H^\top \bar{\zeta}_H \right), \quad (9)$$

$$\langle \bar{\eta}, \bar{\zeta} \rangle_{\bar{x}}^{\text{mix}} := \text{Tr} \left( (H^\top H) \bar{\eta}_G^\top \bar{\zeta}_G + (G^\top G) \bar{\eta}_H^\top \bar{\zeta}_H \right). \quad (10)$$

They differ from the Euclidean metric that simply flattens and concatenates the matrix tuples into vectors and takes their dot product, ⟨η, ζ⟩ = Tr η ⊤ <sup>G</sup>ζ<sup>G</sup> + η ⊤ <sup>H</sup>ζ<sup>H</sup> . Importantly, they are invariant under symmetries of the attention mechanism, and thus define valid metrics on the quotient manifold [\(Absil et al.,](#page-8-8) [2008\)](#page-8-8).

Horizontal space: For ⟨·, ·⟩inv, mix <sup>x</sup>¯ and ¯ξ<sup>G</sup>,<sup>H</sup> ∈ <sup>R</sup> <sup>m</sup>×<sup>r</sup> we have (for a proof, see for example [Luo et al.](#page-9-17) [\(2023\)](#page-9-17))

$$\begin{aligned}\mathcal{H}_{\bar{x}}^{\text{inv}} \overline{\mathcal{M}} &= \{(\bar{\xi}_G, \bar{\xi}_H) \mid \bar{\xi}_G^\top G H^\top H = G^\top G H^\top \xi_H^\top\}, \\ \mathcal{H}_{\bar{x}}^{\text{mix}} \overline{\mathcal{M}} &= \{(\bar{\xi}_G, \bar{\xi}_H) \mid G^\top \bar{\xi}_G H^\top H = G^\top G \xi_H^\top H\}.\end{aligned}$$

Projection onto horizontal space: Given ξ ∈ TxM in the total tangent space, the horizontal space is

$$\mathcal{H}_{\bar{\mathbf{x}}}^{\text{inv}, \text{mix}} \overline{\mathcal{M}} = \{ (\bar{\xi}_G + \mathbf{G} \boldsymbol{\Lambda}^{\text{inv}, \text{mix}}, \bar{\xi}_H - \mathbf{H}(\boldsymbol{\Lambda}^{\text{inv}, \text{mix}})^\top) \}$$

where <sup>Λ</sup>inv solves the Sylvester equation A<sup>Λ</sup> <sup>+</sup> <sup>Λ</sup>A<sup>⊤</sup> <sup>=</sup> B, with A = G⊤GH⊤H, B = G⊤GH<sup>⊤</sup> ¯ξ<sup>H</sup> − ¯ξ ⊤ <sup>G</sup>GH⊤H, whereas <sup>Λ</sup>mix has an explicit form: <sup>Λ</sup>mix <sup>=</sup> <sup>1</sup>/<sup>2</sup> ¯ξ ⊤ <sup>H</sup>H(H⊤H) <sup>−</sup><sup>1</sup> − (G⊤G) <sup>−</sup><sup>1</sup>G<sup>⊤</sup> ¯ξ<sup>G</sup> .

Geodesics: We are unaware of analytical solutions for the geodesics of either (Eq. [9](#page-6-0) and Eq. [10\)](#page-6-1), so we approximate

![](_page_6_Figure_1.jpeg)

Figure 4: Adaptive vs. geodesic sharpness on ImageNet ViTs. We use 72 trained models from [Wortsman et al.](#page-9-21) [\(2022\)](#page-9-21), and measure their generalization gap as the difference between test and train error. The correlation coefficient's magnitude is larger for geodesic sharpness.

them with Eq. [5.](#page-4-2) For horizontal tangent vectors ( ¯ξG, ¯ξH), we have for ⟨·, ·⟩inv x¯

$$\begin{aligned} (\Gamma_{kl}^i)^{\text{inv}} \bar{\xi}_G^k \bar{\xi}_G^l &= -\bar{\xi}_G (G^\top G)^{-1} [\bar{\xi}_G^\top G + G^\top \bar{\xi}_G] \\ &\quad + G (G^\top G)^{-1} \bar{\xi}_G^\top \bar{\xi}_G \end{aligned} \quad (11)$$

(similar for the <sup>H</sup> components). For ⟨·, ·⟩mix x¯ , the geodesic equations are coupled and the G components are

$$\begin{aligned} [(\Gamma_{kl}^i)^{\text{mix}} \bar{\xi}^k \bar{\xi}^l]_G &= \bar{\xi}_G [\bar{\xi}_H^\top H + H^\top \bar{\xi}_H] (H^\top H)^{-1} \\ &\quad - G (\bar{\xi}_H^\top \bar{\xi}_H) (H^\top H)^{-1} \end{aligned} \quad (12)$$

(the H components are similar, proof in Appendix [I.2\)](#page-21-0).

#### 5.3. Transformers

Transformers have a mix of attention layers and layers with more restricted symmetries for which adaptive sharpness is more appropriate. We present in Appendix [C.1](#page-12-2) how we treat each layer of transformers. We introduce relaxations In Appendix [C.2](#page-13-0) we present Algorithm [1,](#page-13-1) which we use to solve for geodesic sharpness.

### 5.3.1. EMPIRICAL VALIDATION: VISION TRANSFORMERS

Experimental setup: We follow [Andriushchenko et al.](#page-8-0) [\(2023\)](#page-8-0), and look at models obtained from fine-tuning CLIP on ImageNet-1k [\(Radford et al.,](#page-9-22) [2021\)](#page-9-22). Specifically, we use the trained classifiers after fine-tuning a CLIP ViT-B/32 on ImageNet with randomly selected hyperparameters from [\(Wortsman et al.,](#page-9-21) [2022\)](#page-9-21). We compute adaptive worst-case, and our geodesic, sharpness on the same 2048 data points from the ImageNet training set, divided into batches of 256, by calculating sharpness on each batch separately, then averaging the results. The generalization gap is the difference between test and training error.

Results: Figure [4](#page-6-2) shows our results. We find a strong correlation between geodesic sharpness and the generalization gap on ImageNet. This correlation is stronger than

![](_page_7_Figure_1.jpeg)

Figure 5: Adaptive vs. geodesic sharpness on MNLI language models. We use 35 trained models from [McCoy et al.](#page-9-23) [\(2020\)](#page-9-23), and show the generalization gap on the MNLI dev matched set [\(Williams et al.,](#page-9-24) [2018\)](#page-9-24). Geodesic sharpness shows the largest correlation.

that observed with adaptive sharpness and is consistently negative, implying that the geodesically sharpest models studied on ImageNet are those that generalize best–contrary to what might have been expected, but consistent with the correlation from the diagonal networks.

#### 5.3.2. EMPIRICAL VALIDATION: LANGUAGE MODELS

Experimental Setup: We also consider BERT models that were fine-tuned on MNLI [\(Williams et al.,](#page-9-24) [2018\)](#page-9-24) by [Mc-](#page-9-23)[Coy et al.](#page-9-23) [\(2020\)](#page-9-23) . We compute adaptive worst-case, and our geodesic, sharpness on the same 1024 data points from the MNLI training set, with batches of 128 points, by calculating then averaging sharpness on each batch.

Results: Figure [5](#page-7-1) shows our results. We find a consistent correlation between geodesic sharpness and the generalization gap on MNLI for both metrics, while adaptive sharpness (τ = 0.06) cannot find any correlation. The correlation is positive, i.e. geodesically flatter models generalize better.

# 6. Additional Experiments

#### 6.1. Comparison With Relative Sharpness

Relative sharpness [\(Petzka et al.,](#page-9-3) [2021\)](#page-9-3) is a promising sharpness measure that has proven useful in regularizing transformer training, outperforming other approaches [\(Adilova](#page-8-5) [et al.,](#page-8-5) [2023\)](#page-8-5). We compare it with our geodesic sharpness in the language model setting from Section [5.3.2;](#page-7-2) see Figure [6.](#page-7-3)

### 6.2. Verification of Reparametrization Invariance

Mathematically, geodesic sharpness is invariant to symmetry transformations of the form of Equation [\(3\)](#page-3-2). Here, we verify empirically that our practical version that can be computed efficiently numerically is close to invariant.

Experimental setup: We take a single batch and language model from Section [5.3.2,](#page-7-2) and compute geodesic sharpness

![](_page_7_Figure_2.jpeg)

Figure 6: Extension of Figure [5](#page-7-1) to relative sharpness. We find that relative flatness [\(Petzka et al.,](#page-9-3) [2021\)](#page-9-3) fails to find a significant correlation, compared to our geodesic sharpness.

for various points on an orbit that represent the same function. Specifically, we reparametrize using A = aG, where G is a random standard Gaussian matrix (which is almost always invertible and sampled once in each run), and control the scale a. We sample one G for each attention head. We compare this with adaptive sharpness.

Results: Figure [7](#page-7-4) visualizes the sharpness ratio before and after reparameterization. The colors represent different values of the scale factor, which goes from 10−<sup>2</sup> to 10<sup>2</sup> . Our numerically computed geodesic sharpness remains constant.

![](_page_7_Figure_9.jpeg)

Figure 7: Variation of adaptive vs. geodesic sharpness within an orbit where the neural net function remains unchanged. We show the ratios between the original sharpness and the sharpness obtained after applying a symmetry transformation. Geodesic sharpness stays constant, whereas adaptive sharpness assumes several different values.

# 7. Remarks, Limitations & Future Work

Discovering correlation: Adaptive sharpness, as discussed thoroughly by [Andriushchenko et al.](#page-8-0) [\(2023\)](#page-8-0), is unable to reveal a correlation between sharpness and generalization for transformers. Our geodesic sharpness consistently recovers strong correlation on transformers, and strengthens the correlation in the case of diagonal networks.

Metric choice: Our results are robust w.r.t. the choice of metric, as long as *it captures the parameter symmetry*. The mixed metric yields slightly better results on BERT, perhaps owing to its more stable numerics (e.g. possible inversion of nearly singular matrices is side-stepped). Additionally, the mixed metric avoids calling expensive Sylvester equation solvers and has a simple horizontal space projection.

**Sign of the correlation:** One of our surprising results is that the sign of the correlation between geodesic sharpness and generalization varies depending on the setting and is at times negative, somewhat at odds with the common view that sharpness always *positively* correlates with generalization (i.e., flatter models generalize better). This artifact is not inherent to our proposed metrics. E.g., adaptive sharpness anti-correlates with generalization in our diagonal network setting, but was previously found to positively correlate with generalization on other tasks (Kwon et al., 2021).

Our geodesic sharpness improves over adaptive sharpness in the following sense: Where adaptive sharpness finds no correlation, our metrics do find a signed correlation, and where adaptive sharpness finds signed correlation, our metrics find a stronger similarly-signed correlation. That is, we for the first time observe *consistent correlations* within-task for transformers, opening questions for further study.

**Limitations:** While our *geodesic sharpness* is more general than previous measures, there remain symmetries for which taking the quotient may be computationally expensive or intractable. Still, we think that accounting for some symmetry is better than none, and even under computational constraints it could be useful as a diagnostic “probe”.

Our new measures detect previously undetected correlation with generalization. In the process, however, we also discovered that the sign of the correlation, while consistent across metrics and models, can vary across tasks. Until this new variability is understood, this limits the utility of geodesic sharpness, e.g. for regularizing transformer training.

**Future work:** Our work is concerned with accounting for parameter space symmetries that are data-independent. This opens up the question: what is the role of data and how can it be integrated into our framework? A more complete understanding of the interplay between data and parameter symmetries might help explain when geodesic sharpness correlates or anti-correlates with generalization.

## 8. Conclusion

In this paper, we revisited the limitations of traditional sharpness measures attempting to predict generalization for transformers, highlighting how traditional sharpness measures fail to properly account for the rich  $GL(h)$  symmetries present in transformers. Addressing this, we introduced geodesic sharpness, a measure defined on the Riemannian quotient manifold obtained by quotienting out transformer symmetries. This framework provides a principled, symmetry-aware measure of sharpness and contains prior adaptive sharpness metrics as first-order approximations.

Through experiments on diagonal networks, vision trans-

formers (ImageNet), and language models (MNLI), we demonstrated that properly accounting for the transformer symmetries restores the correlation between sharpness and generalization. Interestingly, our findings indicate that the sign of the correlation between sharpness and generalization can vary across tasks, suggesting deeper underlying relationships involving data distribution and model structure. This work lays the groundwork for further exploration of these interactions and motivates future research into geometry-informed optimization strategies tailored to transformers.

## Impact Statement

This paper presents work whose goal is to advance the study of deep learning. There are potential indirect societal consequences of our work, none which we feel must be specifically highlighted here.

## Acknowledgements

We would like to express our sincere gratitude to Agustinus Kristiadi and Rob Brekelmans for early feedback on the manuscript. Resources used in preparing this research were provided, in part, by NSERC, the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute.

## References


[1] Absil, P.-A., Mahony, R., and Sepulchre, R. *Optimization algorithms on matrix manifolds*. 2008.

[2] Adilova, L., Abourayya, A., Li, J., Dada, A., Petzka, H., Egger, J., Kleesiek, J., and Kamp, M. Fam: Relative flatness aware minimization, 2023.

[3] Andriushchenko, M., Croce, F., Müller, M., Hein, M., and Flammarion, N. A modern look at the relationship between sharpness and generalization. 2023.

[4] Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization. 2016.

[5] Boumal, N. *An introduction to optimization on smooth manifolds*. Cambridge University Press, 2023.

[6] Croce, F. and Hein, M. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks, 2020.

[7] Dagréou, M., Ablin, P., Vaiter, S., and Moreau, T. How to compute hessian-vector products? In *ICLR Blogposts 2024*, 2024.

[8] Dinh, L., Pascanu, R., Bengio, S., and Bengio, Y. Sharp minima can generalize for deep nets, 2017.

[9] do Carmo, M. *Riemannian Geometry*. Mathematics (Boston, Mass.). Birkhäuser, 1992. ISBN 9783764334901.

[10] Dziugaite, G. K., Drouin, A., Neal, B., Rajkumar, N., Caballero, E., Wang, L., Mitliagkas, I., and Roy, D. M. In search of robust

[11] measures of generalization. In *Advances in Neural Information Processing Systems*, volume 33, pp. 11723–11733. Curran Associates, Inc., 2020. Foret, P., Kleiner, A., Mobahi, H., and Neyshabur, B. Sharpnessaware minimization for efficiently improving generalization. In *International Conference on Learning Representations (ICLR)*, 2021. Hochreiter, S. and Schmidhuber, J. Simplifying neural nets by discovering flat minima. In *Advances in Neural Information Processing Systems (NIPS)*, 1994. Hu, E. J., yelong shen, Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations (ICLR)*, 2022. Ioffe, S. and Szegedy, C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *International Conference on Machine Learning (ICML)*, 2015. Kendall, M. G. A new measure of rank correlation. *Biometrika*, 30(1-2):81–93, 1938. Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., and Tang, P. T. P. On large-batch training for deep learning: Generalization gap and sharp minima, 2017. Kirrinnis, P. Fast algorithms for the sylvester equation. *Theoretical Computer Science*, 259(1):623–638, 2001. ISSN 0304-3975. Kristiadi, A., Dangel, F., and Hennig, P. The geometry of neural nets' parameter spaces under reparametrization. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2023. Kunin, D., Sagastuy-Brena, J., Ganguli, S., Yamins, D. L., and Tanaka, H. Symmetry, conservation laws, and learning dynamics in neural networks. In *International Conference on Learning Representations (ICLR)*, 2021. Kwon, J., Kim, J., Park, H., and Choi, I. K. Asam: Adaptive sharpness-aware minimization for scale-invariant learning of deep neural networks. In *International Conference on Machine Learning (ICML)*, 2021. Lee, J. *Introduction to Smooth Manifolds*. Graduate Texts in Mathematics. Springer, 2003. ISBN 9780387954486. Luo, Y., Li, X., and Zhang, A. R. On geometric connections of embedded and quotient geometries in riemannian fixed-rank matrix optimization, 2023. McCoy, R. T., Min, J., and Linzen, T. Berts of a feather do not generalize together: Large variability in generalization across models with similar test set performance, 2020. Pesme, S., Pillaud-Vivien, L., and Flammarion, N. Implicit bias of sgd for diagonal linear networks: a provable benefit of stochasticity, 2021. Petzka, H., Kamp, M., Adilova, L., Sminchisescu, C., and Boley,

[12] M. Relative flatness and generalization. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2021. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. *CoRR*, abs/2103.00020, 2021. Rangamani, A., Nguyen, N. H., Kumar, A., Phan, D., Chin, S. H., and Tran, T. D. A Scale Invariant Flatness Measure for Deep Network Minima, February 2019. Salimans, T. and Kingma, D. P. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. *Advances in neural information processing systems (NeurIPS)*, 29, 2016. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I. Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2017. Walter, N. P., Adilova, L., Vreeken, J., and Kamp, M. The uncanny valley: Exploring adversarial robustness from a flatness perspective, 2025. Williams, A., Nangia, N., and Bowman, S. A broad-coverage challenge corpus for sentence understanding through inference. In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*, pp. 1112–1122, 2018. Woodworth, B., Gunasekar, S., Lee, J. D., Moroshko, E., Savarese, P., Golan, I., Soudry, D., and Srebro, N. Kernel and rich regimes in overparametrized models, 2020. Wortsman, M., Ilharco, G., Yitzhak Gadre, S., Roelofs, R., Gontijo-Lopes, R., Morcos, A. S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., and Schmidt, L. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. *arXiv e-prints*, 2022. Wu, Y. and He, K. Group normalization, 2018. Yen, J.-N., Si, S., Meng, Z., Yu, F., Surya Duvvuri, S., Dhillon,
  - I. S., Hsieh, C.-J., and Kumar, S. LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization. *arXiv e-prints*, 2024. Yu, H. and Wu, J. Compressing transformers: Features are lowrank, but weights are not! *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(9):11007–11015, Jun. 2023. Zhao, B., Dehmamy, N., Walters, R., and Yu, R. Symmetry teleportation for accelerated optimization. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2022a. Zhao, Y., Zhang, H., and Hu, X. Penalizing gradient norm for efficiently improving generalization in deep learning. In *ICML*, 2022b.
# Hide & Seek: Transformer Symmetries Obscure Sharpness & Riemannian Geometry Finds It (Supplemental Material)

We provide in Table 1 a summary of correlation coefficients between sharpness and generalization for our experiments.

| Setting           | Adaptive sharpness | Rank correlation coefficient ⟨· , ·⟩ inv geodesic sharpness | τ ⟨· , ·⟩ mix geodesic sharpness () |
|-------------------|--------------------|-------------------------------------------------------------|-------------------------------------|
| Diagonal networks | -0.68              | -0.83                                                       | -0.86                               |
| ImageNet          | -0.41              | -0.71                                                       | -0.7                                |
| MNLI              | 0.06               | 0.28                                                        | 0.38                                |

Table 1: Summary of the correlation between sharpness measures and generalization. We boldface the best performing metric

In the sections that follow, we provide additional details to supplement the main text.

# A. Additional Examples of GL symmetries Symmetries in Neural Networks

Example A.1 (Self-attention [\(Vaswani et al.,](#page-9-13) [2017\)](#page-9-13)). Given a sequence X ∈ <sup>R</sup> <sup>t</sup>×<sup>d</sup> with t tokens and model dimension d, self-attention (SA) uses four matrices <sup>W</sup>q,W<sup>k</sup> ∈ <sup>R</sup> d×dk ,Wv,W<sup>⊤</sup> o ∈ <sup>R</sup> d×dv (usually, <sup>d</sup> <sup>=</sup> <sup>d</sup><sup>v</sup> <sup>=</sup> <sup>d</sup>k) to produce a new <sup>t</sup> × <sup>d</sup> sequence

$$\begin{aligned} & \text{SA}(\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o) \\ &= \text{softmax} \left( \frac{\mathbf{X} \mathbf{W}_q \mathbf{W}_k^\top \mathbf{X}^\top}{\sqrt{d_k}} \right) \mathbf{X} \mathbf{W}_v \mathbf{W}_o. \end{aligned} \quad (13)$$

This block contains two GL symmetries: one of dimension <sup>d</sup><sup>k</sup> between the key and query projection weights, G, <sup>H</sup> ← <sup>W</sup>q,Wk, and one of dimension <sup>d</sup><sup>v</sup> between the value and out projection weights, G, <sup>H</sup> ← <sup>W</sup>v,W<sup>⊤</sup> o . Similar to Eq. [14,](#page-10-3) we can account for biases in the key, query, and value projections by appending them to their weight,

$$G, H \leftarrow \begin{pmatrix} W_k \\ b_k^\top \end{pmatrix}, \begin{pmatrix} W_q \\ b_q \end{pmatrix}^\top, \quad G, H \leftarrow \begin{pmatrix} W_v \\ b_v \end{pmatrix}, W_o^\top.$$

Commonly, H attention heads {W<sup>i</sup> q ,W<sup>i</sup> k ,W<sup>i</sup> <sup>v</sup>,i,W<sup>i</sup> o } H <sup>i</sup>=1 independently process <sup>X</sup> and concatenate their results into the final output (usually <sup>d</sup><sup>k</sup> <sup>=</sup> <sup>d</sup><sup>v</sup> <sup>=</sup> <sup>d</sup>/H). This introduces <sup>2</sup><sup>H</sup> GL symmetries. Everything also applies to general attention where, instead of X, independent data is fed as keys, queries, and values to Eq. [13.](#page-10-4)

Example A.2 (Shallow linear net). Consider a two-layer linear net NN(W2,W1) = W2W1x with weight matrices W<sup>1</sup> ∈ R h×din ,W<sup>2</sup> ∈ <sup>R</sup> dout×h and some input x ∈ <sup>R</sup> <sup>d</sup>in . This net has GL symmetry with correspondence G, H ← W2,W<sup>1</sup> <sup>⊤</sup> to Definition [3.1.](#page-3-0) With first-layer bias, we have

$$\mathbf{W}_2(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) = \mathbf{W}_2\left(\mathbf{W}_1 - \mathbf{b}_1\right) \begin{pmatrix} \mathbf{x} \\ 1 \end{pmatrix}, \quad (14)$$

corresponding to G, H ← W2, W<sup>1</sup> b<sup>1</sup> ⊤ .

Example A.3 (Low-rank adapters (LoRA, [Hu et al.](#page-9-14) [\(2022\)](#page-9-14))). Fine-tuning tasks with large language models add a trainable low-rank perturbation L ∈ <sup>R</sup> d1×h , R ∈ R d2×h to the pre-trained weight W ∈ <sup>R</sup> d1×d<sup>2</sup> ,

$$\text{LoRA}(\mathbf{W}) = \mathbf{W} + \mathbf{L}\mathbf{R}^\top, \quad (15)$$

introducing a GL(h) symmetry where G, H ← L, R. [Yen et al.](#page-9-25) [\(2024\)](#page-9-25) propose an invariant way to train the parameters L, R and show that doing so improves the result obtained via LoRA.

# B. Concepts and Review for Riemannian Geometry

Recall that M is the total space: the manifold of parameters of our network. Also, on the total space we have a loss function ℓ : M → <sup>R</sup>. Useful resources are [Lee](#page-9-15) [\(2003\)](#page-9-15), [Absil et al.](#page-8-8) [\(2008\)](#page-8-8), and [Boumal](#page-8-3) [\(2023\)](#page-8-3).

#### B.1. Orbit of x

A symmetry relation naturally defines an equivalence relation: two points x, y ∈ M are equivalent under the symmetry, if they can be mapped onto each other by the action,

$$x \sim y \iff \exists g \in \mathcal{G} : y = \psi(g, x). \quad (16)$$

In other words, if we let orbit(x) :− {ψ(g, x) | g ∈ G} be all points on the total space that are reachable from x through the action of G, all points in an orbit are equivalent. Instead of orbit(x), we will write

$$[x] := \{y \in \overline{M} \mid y \sim x\} \quad (17)$$

for the symmetry-induced equivalence class [x] of x ∈ M.

Let's further assume that ℓ is symmetric under G, i.e. for any x ∈ M and all g ∈ G, ℓ(x) = ℓ(ψ(g, x)).

### B.2. Quotient M and Natural Projection

If we take the quotient of the original parameter space manifold M, by the equivalence relation, ∼, induced by the symmetries of our neural architecture, we get a quotient M = M/ ∼. Under certain conditions, M is a quotient manifold. The mapping between a point in total space to its equivalence class is called the natural projection:

Definition B.1. Let π : M → M/ ∼, be defined by x 7→ x. π is called the natural, or canonical projection. We use π(x) to denote x viewed as a point of M :− M/ ∼.

### B.3. Tangent Space, Vertical and Horizontal Spaces

Tangent vectors on the total space M, embedded in a vector space E can be viewed as tangent vectors to E, but the tangent space of the quotient manifold, M is not as straightforward. First, note that any element ¯ξ ∈ Tx¯M that satisfies Dπ(¯x)[¯ξ] = ξ (where D is the differential) is a candidate for a representation of ξ ∈ TxM. These aren't unique, and as we wish to work without any numerical ambiguity we introduce the notions of the vertical and horizontal spaces:

Definition B.2. For a quotient manifold M = M/ ∼, the vertical space at x¯ ∈ M is the subspace Vx¯ = Tx¯F = ker Dπ(x) where F = {y¯ ∈ M : ¯y ∼ x¯} is the fiber of x¯. The complement of Vx¯ is the horizontal space at x¯: Tx¯M = Vx¯ ⊕ Hx¯.

Definition B.3. There is only one element ¯ξx¯ that belongs to Hx¯ and satisfies Dπ(¯x)[¯ξx¯] = ξ. This unique vector is called the *horizontal lift* of of ξ at x¯. We denote the operator that affects the procedure by liftx¯(·) When the ambient space, E is a subset of <sup>R</sup> n×p , the horizontal space can also be seen as such a subset, providing a convenient matrix representation of *a priori* abstract tangent vectors of M.

### B.4. Riemannian Manifold

We give our total space M a smooth inner product over its tangent vectors to give a Riemannian manifold.

Definition B.4. A Riemannian manifold is a pair (M, g), where M is a smooth manifold and g is a Riemannian metric, defined as the inner product on the tangent space TxM for each point x ∈ M, gx(·, ·) : TxM × TxM → <sup>R</sup>. We also use the notation ⟨·, ·⟩<sup>x</sup> to denote the inner product.

Note that this definition is not as arcane as it may appear since any smooth manifold admits a Riemannian metric, and we can consider the space of parameters of most neural architectures as constituting a smooth manifold, admitting at least a simple, Euclidean, metric.

The horizontal space can now be defined as the *orthogonal* complement of the vertical space: Hx¯ = (Vx¯) <sup>⊥</sup> = {u ∈ Tx¯M : ⟨u, v⟩<sup>x</sup> = 0 for all v ∈ Vx¯}. Additionally, letting g¯x¯ denote the metric on M, if for every x ∈ M and every ξx, ζ<sup>x</sup> in TxM, g¯x¯( ¯ξx¯, ¯ζx¯) does not depend on x¯ ∈ π −1 (x) then, gx(ξx, ζx) = ¯gx¯( ¯ξx¯, ¯ζx¯) defines a valid metric on the quotient manifold M.

### B.5. Riemannian Gradient

Definition B.5. If ¯f is a smooth scalar field on a Riemannian manifold M, then the *gradient* of ¯f at x¯, grad ¯f(¯x) is the unique element of Tx¯M such that

$$\langle \text{grad } \bar{f}(\bar{x}), \bar{\xi} \rangle_{\bar{x}} = D \bar{f}(\bar{x})[\bar{\xi}], \forall \bar{\xi} \in T_{\bar{x}} \overline{\mathcal{M}}$$

If ¯f is a function on M, that induces a function f on a quotient manifold M of M, then we can express the horizontal lift of grad f at x¯ as

$$\text{lift}_{\bar{x}}(\text{grad } f) = \text{grad } \bar{f}(\bar{x}).$$

#### B.6. Geodesic Curves

### Definition B.6.

- (a) Geodesic curves, γ¯, are the curves of minimal distance between two points on a manifold M. The distance along the geodesic is called the *geodesic distance*. If M is a Riemannian quotient manifold of M, with canonical projection π, and γ¯ is a geodesic on M, then γ = π ◦ γ¯ is a geodesic curve on M.
- (b) Alternatively, geodesics, γ¯(t) = 0 can be defined as curves from a given point p ∈ M, (i.e., γ¯(0) = p), with initial *velocity*, γ¯˙(0) = ¯ξ ∈ Tp¯M, such that their *acceleration* is zero (a generalization of Euclidean straight lines). This characterization provides us with the following equation in local coordinates for the geodesic:

$$\frac{d^2\gamma^\lambda}{dt^2} + \Gamma_{\mu\nu} \frac{d\gamma^\mu}{dt} \frac{d\gamma^\nu}{dt} = 0$$

where Γ λ µν are the Christoffel symbols, <sup>Γ</sup> λ µν = 1 2 g λσ ∂gσµ ∂x<sup>ν</sup> + ∂gσν ∂x<sup>µ</sup> − ∂gµν ∂x<sup>σ</sup> . Additionally, the geodesics can also be derived as the curves that are minima of the energy functional

$$S(\gamma) = \int_a^b g_{\gamma(t)}(\gamma(t), \gamma(t)) dt$$

This second perspective will prove useful for the geodesics of the attention layers.

If the initial velocity tangent vector, ξ, is horizontal then, ∀t, γ¯˙(t) ∈ <sup>H</sup>γ¯(t) , that is, if the velocity vector starts out as horizontal, then it will stay horizontal. We call these geodesics, *horizontal geodesics*. The curve γ = π ◦ γ¯ is a geodesic of the quotient manifold M, with the same length as γ¯. This also holds the other way, i.e., a geodesic in the quotient manifold can be lifted to a horizontal geodesic in the total space.

# C. Geodesic sharpness: practical concerns

### C.1. Transformers

Transformers, introduced by [Vaswani et al.](#page-9-13) [\(2017\)](#page-9-13), consist of multiheaded self-attention and feedforward layers, both wrapped in residual connections and layer normalizations. Visual transformers, in addition, tend to have convolutional layers.

Mathematically, focusing for the moment on the multi-headed attention blocks,

MultiHead(
$$Q$$
,  $K$ ,  $V$ ) = [head<sub>1</sub>, ..., head <sub>$h$</sub> ]  $W^o$   
 where   head <sub>$i$</sub>  = Attention( $QW_i^Q$ ,  $KW_i^K$ ,  $VW_i^V$ )

where Attention(Q, K, V ) = softmax QK<sup>T</sup> √ d<sup>k</sup> V . From this we can ascertain the following symmetries:

$$\begin{aligned} 1) \quad & (W_i^Q, W_i^K) \rightarrow (W_i^Q G^{-1}, W_i^K G^T), \forall G \in \mathbf{GL}_n(d_{\text{head}}) \\ 2) \quad & (W_i^V, W_i^o) \rightarrow (W_i^V G^{-1}, W_i^o G^T), \forall G \in \mathbf{GL}_n(d_{\text{head}}) \end{aligned}$$

where W<sup>o</sup> i are the columns of W<sup>o</sup> that are relevant for the matrix multiplication with each W<sup>V</sup> i , taking into consideration the head concatenation procedure.

In the full transformer model when solving for geodesic sharpness, for each layer, we apply Eq. [5](#page-4-2) to each (W Q i , W <sup>K</sup> i ) and (W<sup>V</sup> i , W<sup>o</sup> i ), using Eq. [11.](#page-6-3) This results in horizontal vectors ( ¯ξ Q i , ¯ξ K i ) and ( ¯ξ V i , ¯ξ o i ). For the non-attention parameters, w, (belonging to fully connected layers, convolutional layers and layer norm), we keep to the recipe of adaptive sharpness, so that ||¯ξw|| <sup>=</sup> || ¯ξ<sup>w</sup> ⊙ |w| −1 ||2. The norm of the full update vector, ¯ξ = concat( ¯ξ Q i , ¯ξ K i , ¯ξ V i , ¯ξ o i , ¯ξw), where a sum over all parameters of the network is implicit, is ||¯ξ||<sup>2</sup> = P ||( ¯ξ Q i , ¯ξ K i )||<sup>2</sup> + ||( ¯ξ V i , ¯ξ o i )||<sup>2</sup> + ||¯ξw||<sup>2</sup> .

### C.2. Algorithm

Following the lead of [Andriushchenko et al.](#page-8-0) [\(2023\)](#page-8-0), we use Auto-PDG, proposed in [Croce & Hein](#page-8-9) [\(2020\)](#page-8-9), but now optimizing the horizontal vector ¯ξ instead of the input. In Algorithm [1,](#page-13-1) ℓ is the loss over the batch we are optimizing over, S is the feasible set of horizontal vectors, ¯ξ, with norm smaller than ρ, and P<sup>S</sup> is the projection onto this set. Γ are the Christoffel symbols for the parameters. η and W are fixed hyperparameters, which we keep as in [Andriushchenko et al.](#page-8-0) [\(2023\)](#page-8-0), and the two conditions in Line [20](#page-13-1) can be found in [Croce & Hein](#page-8-9) [\(2020\)](#page-8-9). The only differences to the algorithm employed to calculate adaptive sharpness are in lines [3,](#page-13-1) [8,](#page-13-1) [10,](#page-13-1) and [12.](#page-13-1) For the metric ⟨·, ·⟩mix the only differences are in the Christoffel symbols and in the Riemannian gradient (∇Gℓ → ∇Gℓ H<sup>T</sup>H <sup>−</sup><sup>1</sup> )

Algorithm 1 Auto-PGD

1: Input: objective function ℓ, perturbation set S, ¯ξ (0), initial weights w(0) , η, <sup>N</sup>iter, <sup>W</sup> <sup>=</sup> {w0, . . . , wn} 2: Output: ¯ξmax, <sup>ℓ</sup>max 3: v (1) ← w(0) + ¯ξ (0) − 2 Γ¯ξ (0) ¯ξ (0) ▷ Perturb weights according to Eq. [5](#page-4-2) 4: ¯ξ (1) ← P<sup>S</sup> ¯ξ (0) <sup>+</sup> η∇ξ¯ℓ(<sup>v</sup> (1)) 5: ℓmax ← max{ℓ(w(0)), ℓ(v (1))} 6: ¯ξmax ← ¯<sup>ξ</sup> (0) if <sup>ℓ</sup>max ≡ <sup>ℓ</sup>(w(0)) else ¯ξmax ← ¯<sup>ξ</sup> (1) 7: for k = 1 to Niter − 1 do 8: v (k+1) ← w(0) + ¯ξ (k) − 1 2 Γ¯ξ (k) ¯ξ (k) ▷ Perturb weights according to Eq. [5](#page-4-2) 9: if w(0) is an attention weight then 10: g ← ∇ξ¯ℓ(v (k+1))w(0),T w(0) ▷ Make attention gradients Riemannian 11: else 12: g ← ∇ξ¯ℓ(v (k+1)) ⊙ (w(0)) <sup>2</sup> ▷ Make the other gradients Riemannian 13: end if 14: z (k+1) ← P<sup>S</sup> ¯ξ (k) + ηg) 15: ¯ξ (k+1) ← P<sup>S</sup> ¯ξ (k) + α(z (k+1) − ¯ξ (k) ) + (1 − α)(¯ξ (k) − ¯ξ (k−1)) 16: if ℓ(v (k+1)) > ℓmax then 17: ¯ξmax ← ¯<sup>ξ</sup> (k+1) and ℓmax ← ℓ(<sup>v</sup> (k+1)) 18: end if 19: if k ∈ W then 20: if Condition 1 or Condition 2 then 21: η ← η/<sup>2</sup> and <sup>w</sup>(k+1) ← <sup>w</sup>max 22: end if 23: end if 24: end for

#### C.3. Complexity

Geodesic sharpness is slightly more expensive than adaptive sharpness in the following sense: Our approach consists of three steps: 1) perturbing the weights according to Eq. [5,](#page-4-2) 2) optimizing the perturbations with gradient descent, and 3) projecting them onto the feasible set, i.e. horizontal vectors within the geodesic ball with a small enough norm.

Steps 1) and 2) are also present in adaptive sharpness. Step 1) in our approach is slightly more expensive because we need to evaluate the quadratic form that involves the Christoffel symbols (given by Eq. [11](#page-6-3) and Eq. [12\)](#page-6-4); this step introduces <sup>n</sup>params weight matrix multiplications, but these are quite efficient. Making the gradients Riemannian, costs another <sup>n</sup>params weight matrix multiplications. Neither of these bottleneck our approach. For ⟨·, ·⟩inv, Step 3) requires solving a Sylvester equation to project the direction of the updated geodesic back onto the horizontal space. This solve is cubic in h [\(Kirrinnis,](#page-9-26) [2001\)](#page-9-26), but <sup>h</sup> is usually small (e.g. <sup>h</sup> = 64 in the ImageNet and BERT experiments). For ⟨·, ·⟩mix, only efficient matrix multiplications are required.

On practical transformers, we expect the bottleneck to be the forward and backward propagations, just like in adaptive sharpness.

# D. Geodesic sharpness: Scalar Toy model

To make our approach explicit, we illustrate it on a NN with two scalar parameters G and H, square loss, and a single (scalar) training point (x, y). We use ⟨·, ·⟩inv throughout. For this example, everything is analytically tractable. We also contrast our sharpness measure with previously proposed ones to highlight its invariance.

Since we require full column-rank, our parameter space is M = <sup>R</sup><sup>∗</sup> × <sup>R</sup><sup>∗</sup> with <sup>R</sup><sup>∗</sup> = <sup>R</sup> \ {0}.

Metric: At a point (G, H) ∈ M, for two tangent vectors η = (ηG, ηH), <sup>ν</sup> = (νG, νH) ∈ <sup>T</sup>(G,H)M, we have

$$\langle \boldsymbol{\eta}, \boldsymbol{\nu} \rangle^{\text{inv}} = \frac{\eta_G \nu_G}{G^2} + \frac{\eta_H \nu_H}{H^2} = \eta^\top \underbrace{\begin{pmatrix} \frac{1}{G^2} & 0 \\ 0 & \frac{1}{H^2} \end{pmatrix}}_{g_{kl}} \nu \quad (18)$$

We denote the inverse metric by g kl = G<sup>2</sup> 0 0 H<sup>2</sup>

Horizontal space: <sup>H</sup>(G,H) <sup>=</sup> {(ηG, ηH) ∈ <sup>T</sup>(G,H)M | <sup>η</sup><sup>G</sup> <sup>G</sup> = η<sup>H</sup> H }

Geodesics: To compute the geodesics on the quotient space, we need the Christoffel symbols Γ i km.

Using a coordinate system (p , p<sup>2</sup> ) = (G, H), we have the following equation for a geodesic γ(t) = (γG(t), γH(t)), with initial conditions <sup>γ</sup>(0) = (G0, H0) and <sup>γ</sup>˙(0) = (η<sup>G</sup><sup>0</sup> , η<sup>H</sup><sup>0</sup> )

$$\frac{d^2\gamma_G}{dt^2} + \Gamma_{11}^1 \left( \frac{d\gamma_G}{dt} \right)^2 = 0$$

and similarly for H with Γ 2 <sup>22</sup> instead of <sup>Γ</sup> 1 11.

The Christoffel symbols can be found using the metric, g, and its inverse. Using the Einstein notation and denoting the inverse of g by the use of upper indices:

$$\Gamma^i{}_{kl} = \frac{1}{2}g^{im} \left( \frac{\partial g_{mk}}{\partial x^l} + \frac{\partial g_{ml}}{\partial x^k} - \frac{\partial g_{kl}}{\partial x^m} \right)$$

Then

$$\Gamma^1_{11} = \frac{1}{2} g^{1m} \left( \frac{\partial g_{m1}}{\partial p^1} + \frac{\partial g_{m1}}{\partial p^1} - \frac{\partial g_{kl}}{\partial p^m} \right) = -\frac{1}{G}$$

$$\Gamma^2_{22} = -\frac{1}{H}$$

All other Christoffel symbols are 0. Our geodesic equations then become (we omit the derivation for H, which is identical but with G ↔ H)

$$\frac{d^2\gamma_G}{dt^2} - \frac{1}{\gamma_G} \left( \frac{d\gamma_G}{dt} \right)^2 = 0$$

This ODE has the (unique) solution γG(t) = A<sup>G</sup> exp(bGt). Taking into account the initial conditions, A<sup>G</sup> = G0, A<sup>H</sup> = H<sup>0</sup> and due to the definition of the horizontal space, b<sup>G</sup> = η<sup>G</sup> G<sup>0</sup> and b<sup>H</sup> = η<sup>H</sup> H0 , this becomes

$$\gamma(t) = \left( G_0 \exp\left(\frac{\eta_G}{G_0}t\right), H_0 \exp\left(\frac{\eta_H}{H_0}t\right) \right)$$

One important detail to note is that these geodesics are not complete, that is, not all two points can be connected by a geodesic. Points with different signs cannot be connected, which makes sense since we excluded the origin from the acceptable parameters and in 1D we need to cross it to connect points with differing signs. All points that lie in the same quadrant as (G0, H0) can be connected through a geodesic.

Putting it all together

$$S_{\max}^\rho((G_0, H_0)) = \left[ \max_{||b|| \leq \rho} x^2 G_0^2 H_0^2 (\exp(4b) - 1) - 2yx G_0 H_0 (\exp(2b) - 1) \right], \quad (19)$$

Letting y<sup>0</sup> = G0H0x, this becomes:

$$S_{\max}^\rho((G_0, H_0)) = \left[ \max_{||b|| \leq \rho} y_0^2(\exp(4b) - 1) - 2yy_0(\exp(2b) - 1) \right], \quad (20)$$

Since η<sup>H</sup> is completely determined by η<sup>G</sup> we can ignore the maximization over it.

Since in practice we'll take ρ ≪ 1, we Taylor expand to get

$$S_{\max}^\rho \approx 4\rho|y_0||y - y_0|$$

This presents an issue when the residual, |y − y0|, is zero, so we can also expand to second order, to get, when |y − y0| ≈ 0

$$S_{\max}^\rho \propto \rho^2 |y_0| |y - 2y_0| = 2\rho^2 y_0^2$$

This is, up to constants, just ||G ⊙ H||<sup>2</sup> 2 . This is also invariant to GL<sup>1</sup> transformations, as expected.

Very close to the minimum we only capture (second-order in ρ) properties of the network, a bit further away from it we capture a (first-order in ρ) mix of data and network properties.

Comparison with more traditional measures: The local average and worst case Euclidean sharpness (at a minimum) are

$$S_{\text{avg}} = \text{Tr} \nabla^2 L_S = G^2 + H^2$$

$$S_{\text{max}} = \lambda_{\text{max}}(\nabla^2 L_S) = G^2 + H^2$$

Adaptive sharpness is defined as

$$S_{\text{avg}}^\rho(w, c) = \mathbb{E}_{S \sim \mathbb{P}_m} [L_S(w + \delta) - L_S(w)], \quad \delta \sim \mathcal{N}(0, \rho^2 \text{diag}(c^2))$$

$$S_{\text{max}}^\rho(w, c) = \mathbb{E}_{S \sim \mathbb{P}_m} \left[ \max_{\|\delta \odot c^{-1}\|_p \leq \rho} L_S(w + \delta) - L_S(w) \right],$$

By picking c very carefully one can get

$$S_{\text{avg}}^\rho(w, c) = |GH|$$

$$S_{\text{max}}^\rho(w, c) = |GH|$$

Geodesic flatness with more data points: How does the geodesic flatness look like with more data points?

$$L_S(G, H) = \frac{1}{n} \sum_{i=1}^n (GHx_i - y_i)^2$$

which leads to (defining y 0 <sup>i</sup> <sup>=</sup> GHxi):

$$S_{\max}^\rho = \max_b \frac{1}{n} \sum_{i=1}^n \left[ (y_i^0)^2 \left( \exp\left(\frac{b}{|b|} 2\sqrt{2}\rho\right) - 1 \right) - 2yy_i^0 \left( \exp\left(\frac{b}{|b|} \sqrt{2}\rho\right) - 1 \right) \right] \quad (21)$$

Taylor expanding (in ρ) once more, we see that

$$S_{\max}^{\rho} \approx \max_b \frac{1}{n} \sum_{i=1}^n \left[ 2\sqrt{2}\rho \frac{b}{|b|} y_i^0 (y_i^0 - y) + 2\rho^2 (y_i^0)^2 \right] \quad (22)$$

Which <sup>b</sup> maximizes Eq. [22,](#page-16-1) depends on the sign of P<sup>n</sup> <sup>i</sup>=1 y 0 i (y 0 <sup>i</sup> − y) : b < 0 if the sum is negative, the reverse if the opposite is true.

#### D.1. Traditional flatness

In Figure [8](#page-17-0) we extend Figure [1](#page-1-0) to include the trace of the Hessian, both Euclidean and Riemannian. The trace of the network Hessian is a quantity that can be used to quantify flatness. We plot, for the scalar toy model, the level sets of: a) the loss function; b) the Euclidean and Riemannian gradient; c) the traces of the Euclidean and Riemannian network Hessian. Several features of the plots are important to note: a) the Riemannian version of the gradient and Hessian have the same level set geometry as the loss function; b) both the Riemannian gradient norm and the trace of the Riemannian Hessian have smaller values throughout than their Euclidean equivalents; c) the trace of the Riemannian Hessian actually reaches 0 when at the local minimum, whereas the Euclidean Hessian actually attains its highest value there; d) the Euclidean trace of the Hessian cannot distinguish between a minimum and a maximum whereas the Riemannian trace can actually do so. Even for simple flatness measures, correcting for the quotient geometry can provide a much clearer picture.

# E. Geodesic sharpness: Diagonal networks in full generality

#### E.1. Metric [\(6\)](#page-5-2)

Metric: At a point (u, <sup>v</sup>) ∈ M, for two tangent vectors η = (ηu, ηv), <sup>ν</sup> = (νu, <sup>ν</sup>v) ∈ <sup>T</sup>(u,v)M, we have

$$\langle \eta, \nu \rangle^{\text{inv}} = \sum_{i=1}^d \frac{\eta_u^i \nu_u^i}{(u^i)^2} + \frac{\eta_v^i \nu_v^i}{(v^i)^2} \quad (23)$$

Horizontal space: <sup>H</sup>(u,v) <sup>=</sup> {(ηu, ηv) <sup>∈</sup> <sup>T</sup>(u,v)M | <sup>η</sup> u <sup>u</sup><sup>i</sup> = η <sup>v</sup><sup>i</sup> ∀i ∈ {1, . . . , d}}

Geodesics: We define b <sup>i</sup> = η u <sup>u</sup><sup>i</sup> = η v <sup>v</sup><sup>i</sup> <sup>∀</sup><sup>i</sup> ∈ {1, . . . , d}, so that

$$\gamma(t)^i = (\mathbf{u}(t), \mathbf{v}(t)) = (\mathbf{u}_0^i \exp(\mathbf{b}_i t), \mathbf{v}_0^i \exp(\mathbf{b}_i t)) \forall i \in \{1, \dots, d\} \quad (24)$$

where u i 0 and v i 0 are the initial positions for our parameters, i.e., the parameters that the network actually learned.

Geodesic sharpness: We assume that in what follows X<sup>T</sup> X = Idd, and we denote β<sup>0</sup> = u<sup>0</sup> ⊙ v0, γ<sup>t</sup> = exp 2b 1 t , . . . exp 2B<sup>d</sup> t , β<sup>t</sup> = (u<sup>t</sup> ⊙ <sup>v</sup>t) = β<sup>0</sup> ⊙ γt, β<sup>∗</sup> <sup>=</sup> <sup>X</sup><sup>T</sup> y. Note that β<sup>∗</sup> is just the optimal least squares predictor when X<sup>T</sup> X = Id. With this notation

$$S_{\max} = \max_{\|\mathbf{b}\| \leq \rho} \sum_i^d [(\beta_0^i)^2 (\gamma_t \odot \gamma_t - 1)] - 2(\beta_0 \odot \gamma_t - 1)^T \beta_* \quad (25)$$

![](_page_17_Figure_1.jpeg)

Figure 8: Quantities from the Riemannian quotient manifold respect the loss landscape's symmetry; Euclidean quantities do not. We use a synthetic least squares regression task with a two-layer NN x 7→ θ2θ1x with scalar parameters θ<sup>i</sup> ∈ <sup>R</sup> and input x ∈ <sup>R</sup>. The NN is re-scale invariant, i.e. has GL(1) symmetry: For any α ∈ <sup>R</sup> \ {0}, the parameters (θ ′ , θ′ ) = (α −1 θ1, αθ2) represent the same function. [\(a\)](#page-1-0) The loss function inherits this symmetry and has hyperbolic level sets. [\(b,c\)](#page-17-0) The Euclidean gradient norm does not share the loss function's geometry and changes throughout an orbit where the NN function remains constant. [\(d,e\)](#page-17-0) The Riemannian gradient norm and Hessian trace follow the loss function's symmetry and remain constant throughout an orbit, i.e. they do not suffer from ambiguities for two points in parameter space that represent the same NN function. All quantities were normalized to [0; 1] and we fixed six points in parameter space and computed the level sets running through them to illustrate the geometry.

At a first glance, this expression does not seem to have a simple interpretation, but we Taylor expand it to second order in B (since ρ is supposed to be small):

$$S_{\max} \approx \max_{\|b\| \leq \rho} 4b^T r + 4b^T D_{\beta_0, \beta_*} b \quad (26)$$

where r = {β i 0 (β i <sup>0</sup> − β i ∗ ), i = 1, . . . , d}, r ′ = {(β i <sup>0</sup> − β i ∗ ), i = 1, . . . , d} and <sup>D</sup>β0,β<sup>∗</sup> <sup>=</sup> diag(<sup>β</sup> i 0 (2β i <sup>0</sup> − β i ∗ )) = diag(β i 0 (β i <sup>0</sup> + (r ′ ) i )). We separate the analysis of Eq[.26](#page-17-1) into three cases:

case a): r ̸= 0 and first order suffices Eq[.26](#page-17-1) becomes

$$S_{\max} = \max_{\|\mathbf{b}\| \leq \rho} 4\mathbf{b}^T \mathbf{r}$$

with solution Smax = 4ρ||r||. This is essentially the gradient norm– a useful quantity for understanding generalization [\(Zhao](#page-9-27) [et al.,](#page-9-27) [2022b\)](#page-9-27).

case b): r = 0 Here we necessarily have to consider the second order terms, so that Eq[.26](#page-17-1) becomes

$$S_{\max} = \max_{||\mathbf{b}|| \leq \rho} 4\mathbf{b}^T \mathbf{D}_{\beta_0, \beta_*} \mathbf{b}$$

This has the well known solution of Smax <sup>=</sup> ρ <sup>2</sup>λmax(Dβ0,β<sup>∗</sup> ) = ρ <sup>2</sup> max((β i 0 ) 2 ). This is just ||β||<sup>2</sup> <sup>∞</sup>, which is the square of what we would get by using adaptive sharpness, Eq[.39,](#page-19-1) with a very carefully chosen hyper-parameter c. This is a quantity that is useful when our ground-truth, β ∗ is dense.

case c): r ̸= 0 and we need both first and second order terms In this case, Eq[.26](#page-17-1) needs to be considered in full, and we solve the maximization problem using Lagrange multipliers. The Lagrangian will be

$$\mathcal{L} = -4\mathbf{b}^T \mathbf{r} - 4\mathbf{b}^T \mathbf{D}_{\beta_0, \beta_*} \mathbf{b} + \lambda(\mathbf{b}^T \mathbf{b} - \rho^2)$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = -4\mathbf{r} - 8\mathbf{D}_{\beta_0, \beta_*} \mathbf{b} + 2\lambda \mathbf{b} = 0 \quad (27)$$

$$\lambda(\mathbf{b}^T \mathbf{B} - \rho^2) = 0 \quad (28)$$

$$\lambda \geq 0 \quad (29)$$

If the constraint is not active, then λ = 0 and

$$b_* = -\frac{1}{2}D_{\beta_0, \beta_*}^{-1} r$$

In practice, unless ρ is large the constraint will always be active, in which case [27](#page-18-0) becomes

$$\begin{aligned} -4\mathbf{r} - 8\mathbf{D}_{\beta_0, \beta_*} \mathbf{b} + 2\lambda(\mathbf{B}) &= 0 \\ (\mathbf{b}^T \mathbf{b} - \rho^2) &= 0 \\ \lambda &\geq 0 \end{aligned}$$

this then becomes

$$B_* = 2(\lambda I - 4D_{\beta_0, \beta_*})^{-1} \mathbf{r}$$

$$4 \sum_i^d \frac{(\mathbf{r}^i)^2}{(\lambda - 4(\beta_0^i(\beta_0^i + \mathbf{r}'))^2)} = \rho^2$$

$$\lambda \geq 0$$

### E.2. Metric [\(7\)](#page-5-3)

We follow the same approach as in the previous section. The main difference will be in the form of the geodesics: u(t) ⊙ v(t) = (u<sup>0</sup> ⊙ v0) ⊙ (1 + 2bt), where b <sup>i</sup> = η u <sup>u</sup><sup>i</sup> = η v<sup>i</sup> , as in the previous section. This essentially treats the two-layer neural network as if it were a single layer, with predictor β = u ⊙ v, that it then perturbs linearly to determine sharpness. For ⟨·, ·⟩mix, and denoting by <sup>D</sup><sup>β</sup> <sup>=</sup> diag(<sup>β</sup> i 0 ):

$$S_{max} = \max_{||\eta||^{\max} \leq \rho} 4 [\mathbf{b}^T(\boldsymbol{\beta}_0 - \boldsymbol{\beta}_*) + \mathbf{b}^T \mathbf{D}_{\boldsymbol{\beta}}^2 \mathbf{b}] \quad (30)$$

We also have that

$$(||\eta||^{\text{mix}})^2 = [\dots + (\mathbf{v}^i)^2(\eta_{\mathbf{v}}^i)^2 + (\mathbf{u}^i)^2(\eta_{\mathbf{v}}^i)^2 + \dots] \quad (31)$$

$$= \left[ \dots + (\mathbf{v}^i)^2(\mathbf{u}^i)^2 \left( \frac{(\eta_{\mathbf{u}}^i)^2}{(\mathbf{u}^i)^2} + \frac{(\eta_{\mathbf{v}}^i)^2}{(\mathbf{v}^i)^2} \right) + \dots \right] \quad (32)$$

$$= [\dots + 2(\mathbf{v}^i)^2(\mathbf{u}^i)^2(\mathbf{b}^i)^2 + \dots] = \|2\mathbf{D}_{\mathbf{b}_0}\mathbf{b}\|_2 \quad (33)$$

Substituting <sup>2</sup>Dβ<sup>0</sup> <sup>b</sup> <sup>=</sup> δ, Eq. [30](#page-18-1) becomes

$$S_{max} = \max_{\|\delta\| \leq \rho} [\delta^T (\beta_0 - \beta_*) + \delta^T \delta] \quad (34)$$

with the solution (up to constants)

$$S_{max} = \rho \|\boldsymbol{\beta}_0 - \boldsymbol{\beta}_*\|_2 \quad (35)$$

# F. Geodesic Sharpness: GL<sup>1</sup> symmetry and Adaptive Sharpness

What happens if instead of a general GL<sup>n</sup> symmetry, we factor out a GL<sup>1</sup> re-scaling symmetry? That is, we identify, element-wise, (¯x, y¯) ∼ (¯x ′y¯ ′ ) if ∃α ∈ <sup>R</sup><sup>∗</sup> = <sup>R</sup> \ {0} s.t. x¯ = αx¯ ′ and y¯ = α <sup>−</sup><sup>1</sup>y¯.

This is the symmetry present in diagonal networks, and so we utilize the metric given by Eq. [6,](#page-5-2) reproduced below for convenience of the reader:

$$g \left[ (\eta_{\mathbf{u}}, \eta_{\mathbf{v}}), (\nu_{\mathbf{u}}, \nu_{\mathbf{v}}) \right] = \sum_{i=1}^d \frac{\eta_{\mathbf{u}}^i \nu_{\mathbf{u}}^i}{(\mathbf{u}^i)^2} + \frac{\eta_{\mathbf{v}}^i \nu_{\mathbf{v}}^i}{(\mathbf{v}^i)^2} \quad (36)$$

Note that this metric is equivalent to the following metric:

$$g \left[ (\eta_{\mathbf{u}}, \eta_{\mathbf{v}}), (\nu_{\mathbf{u}}, \nu_{\mathbf{v}}) \right] = g \left[ (\eta_{\mathbf{u}}/|\mathbf{u}|, \eta_{\mathbf{v}}/|\mathbf{v}|), (\nu_{\mathbf{u}}/|\mathbf{u}|, \nu_{\mathbf{v}}/|\mathbf{v}|) \right]_{\text{euc}} \quad (37)$$

where geuc is the usual Euclidean metric and the division is taken to be element-wise. Denoting the concatenation of all tangent vectors by ξ, the concatenation of all parameters by w, we have ||ξ|| = ||ξ/|w|||2.

In this situation Eq. [4](#page-4-1) becomes (γ denotes our geodesics as usual)

$$S_{\max}^\rho(w, c) = \mathbb{E}_{\mathbb{S} \sim \mathbb{D}} \left[ \max_{\| |\xi| / |\mathbf{w}| \|_2 \leq \rho} L_S(\bar{\gamma}_\xi(1)) - L_S(\bar{\gamma}_\xi(0)) \right], \quad (38)$$

If we then ignore the corrections induced by the geometry of the metric on the geodesics, i.e., take <sup>γ</sup>¯ξ¯(1) = ¯γξ¯(0)+¯<sup>ξ</sup> <sup>=</sup> <sup>w</sup>+¯ξ, then we get

$$S_{\max}^\rho(w, c) = \mathbb{E}_{\mathbb{S} \sim \mathbb{D}} \left[ \max_{\|\xi / \|w\|\|_2 \leq \rho} L_S(\mathbf{w} + \xi) - L_S(\mathbf{w}) \right] \quad (39)$$

which is exactly the formula for adaptive sharpness.

# G. Geodesic Sharpness: Ablations

In this appendix we conduct ablation studies on geodesic sharpness (Equation [\(4\)](#page-4-1)). There are two main components to our recipe that differ from adaptive sharpness: a) the norm ||¯ξ||; b) the weight update formula, which instead of the usual w<sup>i</sup> = w<sup>i</sup> + ¯ξ takes into account the curvature induced by the parameter space symmetries w<sup>i</sup> = w<sup>i</sup> + ¯ξ <sup>i</sup> − 1 2 Γ i kl ¯ξ k ¯ξ l . Below we turn off these components one by one and re-compute the resulting sharpness on MNLI using the BERT models described in Section [5.3.2.](#page-7-2)

Metric [\(9\)](#page-6-0): In Figure [9](#page-20-2) we show the results for our ablation studies using metric [\(9\)](#page-6-0). The norm component is much more impactful than the second-order weight corrections. Turning off the second-order weight corrections results in a small performance drop only.

Metric [\(10\)](#page-6-1): In Figure [10](#page-20-3) we show the results for our ablation studies using metric [\(10\)](#page-6-1). The norm component is still much more impactful than the second-order weight corrections for this metric, but now the second-order weight corrections are essential, and without them sharpness loses a considerable amount of predictive power.

# H. Geodesic Sharpness: Ranks and Relaxation

# H.1. Ranks: how natural is Assumption 5.1?

In general, in non-linear networks there is a tendency towards low-rank representations, which might make Assumption 5.1 seem excessive and counter to realistic situations. However, while the learned WQW<sup>T</sup> <sup>K</sup> tend to be low-rank, <sup>W</sup><sup>Q</sup> and <sup>W</sup><sup>K</sup> (on which Assumption 5.1 ought to apply) themselves are usually high/full (column) rank [\(Yu & Wu,](#page-9-20) [2023\)](#page-9-20).

![](_page_20_Figure_1.jpeg)

Figure 9: The generalization gap on the MNLI dev matched set [\(Williams et al.,](#page-9-24) [2018\)](#page-9-24) vs. worst-case adaptive sharpness with metric [\(9\)](#page-6-0) is shown for 35 models from [\(McCoy et al.,](#page-9-23) [2020\)](#page-9-23). On the left we plot the results when we turn off the corrected norm, and on the middle when we turn off the second-order weight corrections. Right are the results with no ablations.

![](_page_20_Figure_3.jpeg)

Figure 10: The generalization gap on the MNLI dev matched set [\(Williams et al.,](#page-9-24) [2018\)](#page-9-24) vs. worst-case adaptive sharpness with metric [\(10\)](#page-6-1) is shown for 35 models from [\(McCoy et al.,](#page-9-23) [2020\)](#page-9-23). On the left we plot the results when we turn off the corrected norm, and on the middle when we turn off the second-order weight corrections. On the right are the results with no ablations.

#### H.2. Relaxation

Due to the definition of metric [9,](#page-6-0) we need to invert matrices of the type of W<sup>T</sup> <sup>Q</sup>WQ. When these are not full-rank, numerical stability can suffer. Due to floating-point precision rounding errors, in practice W<sup>T</sup> <sup>Q</sup>W<sup>Q</sup> is always invertible, but sometimes the inverted matrices have huge singular values. To combat this, we introduce a relaxation parameter, so that W<sup>T</sup> <sup>Q</sup>W<sup>Q</sup> → W<sup>T</sup> <sup>Q</sup>W<sup>Q</sup> <sup>+</sup> ϵIh, which dampens the resulting singular values. Although we cannot take it to be exactly zero, as long as it is small enough, numerical stability is improved and the results remain roughly the same. We study the effects of varying this parameter on our results empirically below (Figure [11\)](#page-21-1), using the same setup as in Section [5.3.2.](#page-7-2) The results are not significantly affected by the variation of this parameter.

# I. Additional Derivations and Proofs

### I.1. Proof that Eq. [9](#page-6-0) defines a valid Riemannian metric

Eq. [9](#page-6-0) defines a valid metric on the total space M if it is smooth, and for each point (G, ¯ H¯ ) ∈ M it defines a valid inner product on the tangent space <sup>T</sup>(G, ¯ <sup>H</sup>¯ )M. That it is smooth is obvious, so we show that ⟨η, ¯ ¯ζ⟩(G, ¯ <sup>H</sup>¯ ) <sup>=</sup> Tr (G⊤G) <sup>−</sup><sup>1</sup>η¯ ⊤ G ¯ζ<sup>G</sup> + (H⊤H) <sup>−</sup><sup>1</sup>η¯ ⊤ H ¯ζH defines a valid inner product:

- (i) *Symmetry* ⟨η, ¯ ¯ζ⟩ = ⟨ ¯ζ, η¯⟩: omitting the H term as it is identical, ⟨η, ¯ ¯ζ⟩ = Tr (G⊤G) <sup>−</sup><sup>1</sup>η¯ ⊤ G ¯ζG = Tr (G⊤G) <sup>−</sup><sup>1</sup> ¯ζ ⊤ <sup>G</sup>η¯<sup>G</sup> = ⟨ ¯ζ, η¯⟩ ;

![](_page_21_Figure_1.jpeg)

Figure 11: The generalization gap on the MNLI dev matched set [\(Williams et al.,](#page-9-24) [2018\)](#page-9-24) vs. worst-case adaptive sharpness (left) and geodesic sharpness (⟨·, ·⟩inv), is shown for 35 models from [\(McCoy et al.,](#page-9-23) [2020\)](#page-9-23). Only the relaxation parameter differs between plots. The results stay broadly the same.

- (ii) *Bilinearity* ⟨aη¯ + b ¯ζ, λ¯⟩ = a⟨η, ¯ λ¯⟩ + b⟨ ¯ζ, λ¯⟩ = ⟨λ, a ¯ η¯ + b ¯ζ⟩: follows by linearity of the trace;
- (iii) *Positive Definiteness* ⟨η, ¯ η¯⟩ ≥ 0: using assumption 5.1, G<sup>T</sup> G is invertible and is positive-definite; this means that (G<sup>T</sup> G) −1 is also positive-definite, and so ⟨η, ¯ η¯⟩ ≥ 0, with equality only when η¯ = 0.

The proof that Equation [\(10\)](#page-6-1) defines a valid metric is analogous.

# I.2. Derivation of the geodesic corrections for attention

We apply the Euler-Lagrange formalism to the energy functional to derive the geodesic equation on the attention quotient manifold, and hence Γ i kl ¯ξ k G ¯ξ <sup>G</sup>, remembering that geodesics, in local coordinates, obey the equation <sup>d</sup> 2γ dt<sup>2</sup> + Γ<sup>i</sup> kl dγ<sup>k</sup> dt dγ<sup>l</sup> dt = 0. Starting from

$$E(\gamma) = \int_0^1 \mathcal{L} \, dt = \int_0^1 \langle \dot{\gamma}(t), \dot{\gamma}(t) \rangle_{\gamma(t)} dt \quad (40)$$

$$= \int_0^1 [\text{Tr}(\gamma_{\mathbf{G}}(t)^T \gamma_{\mathbf{G}}(t)) \dot{\gamma}_{\mathbf{G}}(t)^T \dot{\gamma}_{\mathbf{G}}(t) + \text{Tr}(\gamma_{\mathbf{H}}(t)^T \gamma_{\mathbf{H}}(t)) \dot{\gamma}_{\mathbf{H}}(t)^T \dot{\gamma}_{\mathbf{H}}(t)] dt \quad (41)$$

,

The Euler-Lagrange equation, for the G part only, reads

$$\frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{G}} \right) - \frac{\partial \mathcal{L}}{\partial G} = 0 \quad (42)$$

We have

$$\frac{\partial \mathcal{L}}{\partial G} = -2G (G^T G)^{-1} (\dot{G}^T \dot{G}) (G^T G)^{-1} \quad (43)$$

$$\frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{G}} \right) = 2\ddot{G} (G^T G)^{-1} - 2\dot{G} (G^T G)^{-1} \left( \dot{G}^T G + G^T \dot{G} \right) (G^T G)^{-1} \quad (44)$$

So that Eq. [42](#page-22-0) becomes:

$$\vec{G} - \vec{G} (G^T G)^{-1} (\vec{G}^T G + G^T \vec{G}) + G (G^T G)^{-1} (\vec{G}^T \vec{G}) = 0 \quad (45)$$

From which we read

$$\Gamma_{kl}\bar{\xi}_G^k\bar{\xi}_G^l = \left[ -\bar{\xi} (\mathbf{G}^T \mathbf{G})^{-1} (\bar{\xi}^T \mathbf{G} + \mathbf{G}^T \bar{\xi}) + \mathbf{G} (\mathbf{G}^T \mathbf{G})^{-1} (\bar{\xi}^T \bar{\xi}) \right]^t \quad (46)$$

The same reasoning is used to deduce Eq. [12.](#page-6-4)

### I.3. Metrics related by scaling and constants

If g is a metric and gscaled <sup>=</sup> Cg <sup>+</sup> <sup>D</sup>, then from Eq. [40](#page-22-1) and Eq. [42](#page-22-0) we see that the geodesics induced by gscaled are the same as those induced by g. The geodesic sharpness induced by gscaled is

$$\begin{aligned} S_{\max}^\rho(w) &= \mathbb{E}_{\mathbb{S} \sim \mathbb{D}} \left[ \max_{\|\xi\| | \gamma_{\text{scaled}} \leq \rho} L_S(\bar{\gamma}_{\xi;\text{scaled}}(1)) - L_S(\bar{\gamma}_{\xi;\text{scaled}}(0)) \right] \\ &= \mathbb{E}_{\mathbb{S} \sim \mathbb{D}} \left[ \max_{C | | \xi | | \gamma + D \leq \rho} L_S(\bar{\gamma}_{\xi}(1)) - L_S(\bar{\gamma}_{\xi}(0)) \right], \\ &= \mathbb{E}_{\mathbb{S} \sim \mathbb{D}} \left[ \max_{\|\xi\| | \gamma \leq \rho'} L_S(\bar{\gamma}_{\xi}(1)) - L_S(\bar{\gamma}_{\xi}(0)) \right], \end{aligned}$$

So they are the same up to some re-definition of the hyperparameter ρ.