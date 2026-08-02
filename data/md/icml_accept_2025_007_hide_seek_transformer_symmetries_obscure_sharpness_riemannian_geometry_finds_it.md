# Hide & Seek: Transformer Symmetries Obscure Sharpness & Riemannian Geometry Finds It

Marvin F. da Silva 1 2 Felix Dangel 2 **Sageev Oore** 1 2

## Abstract

The concept of sharpness has been successfully applied to traditional architectures like MLPs and CNNs to predict their generalization. For transformers, however, recent work reported weak correlation between flatness and generalization. We argue that existing sharpness measures fail for transformers, because they have much richer symmetries in their attention mechanism that induce directions in parameter space along which the network or its loss remain identical. We posit that sharpness must account fully for these symmetries, and thus we redefine it on a quotient manifold that results from quotienting out the transformer symmetries, thereby removing their ambiguities. Leveraging tools from Riemannian geometry, we propose a fully general notion of sharpness in terms of a geodesic ball on the symmetrycorrected quotient manifold. In practice, we need to approximate the geodesics. Doing so up to first order yields existing adaptive sharpness measures, and we demonstrate that including higher-order terms is crucial to recover correlation with generalization. We present results on diagonal nets with synthetic data and show that our geodesic sharpness reveals strong correlation with generalization for real-world transformers on both text and image classification tasks.

## 1. Introduction

Predicting generalization of neural nets (NNs)—the discrepancy between training and test set performance—remains an open challenge. Generalization-predictive metrics are valuable though: they enable explicit regularization of training to enhance generalization (Foret et al., 2021), and provide 1Faculty of Computer Science, Dalhousie University, Halifax, Canada 2Vector Insitute for Artificial Intelligence, Toronto, Canada.

Correspondence to: Marvin F. da Silva <marvinf.silva@dal.ca>.

broader theoretical insights into generalization itself.

There is a long history of hypotheses linking sharpness to generalization, but evidence has been conflicting (Hochreiter & Schmidhuber, 1994; Andriushchenko et al., 2023). Generalization has been speculated as correlating with flatness, but recent evidence has indicated that, in the case of transformers, it has little to no correlation whatsoever. Measures of sharpness have varied widely, ranging from trace of the Hessian to worst-case loss within a local neighborhood, with adaptive and relative variations proposed to address specific challenges (Kwon et al., 2021; Petzka et al., 2021). We suspect that some of the confusion stems from the specificity of the problem these measures have attempted to address: the issue of parameter rescaling. In contrast, we argue that rescaling (Dinh et al., 2017) is merely a special case of a broader, more fundamental obstacle to measuring sharpness accurately: the presence of full and continuous parameter symmetries. Addressing this challenge is crucial to ensure that we are studying the right quantity when investigating the relationship between sharpness and generalization.

Beyond discrete permutation symmetries, neural nets naturally exhibit continuous symmetries in their parameter space. These symmetries are intrinsic, data-independent properties that emerge from standard architectural components. For example: normalization layers (Ioffe & Szegedy, 2015; Ba et al., 2016; Wu & He, 2018) induce scale invariance on the pre-normalization weights (Salimans & Kingma, 2016); homogeneous activation functions like ReLU introduce re-scaling symmetries between pre- and post-activation weights (Dinh et al., 2017); some normalization layers and softmax impose translation symmetries in the preceding layer's biases (Kunin et al., 2021). As a result, arguably almost any NN, along with its corresponding loss, exhibit symmetries and can therefore represent the *same* function using *different* parameter values (Figure 1a). Adaptive flatness (Kwon et al., 2021) accounts for some symmetries, both element- and filter-wise re-scaling, but fails to capture the attention mechanism's *full* symmetry, represented by GL(h) (re-scaling by invertible h × h matrices, where h is the hidden dimension), as we will discuss later. Aiming to break the cycle between discovery of a 1

(a) Loss 0.0 0.2 0.4 0.6 0.8 1.0
(b) Euclidean gradient norm (c) Riemannian gradient norm
−1.0 −0.5 0.0 0.5 1.0 θ1
−1.00
−0.75 −0.50 −0.25 0.00 0.25 0.50 0.75 1.00 θ2
Figure 1: **Quantities from the Riemannian quotient manifold respect the loss landscape's symmetry; Euclidean**
quantities do not. We illustrate this here for a synthetic least squares regression task with a two-layer NN, where x 7→ θ2θ1x with scalar parameters θ ∈ R
2and input x ∈ R (i.e. each layer is a linear function). The NN is re-scale invariant, i.e. has GL(1) symmetry: For any α ∈ R \ {0}, the parameters (θ
′1, θ′2) = (α
−1θ1*, αθ*2) represent the same function. (a) The loss function inherits this symmetry and has hyperbolic level sets. (b) The Euclidean gradient norm does not share the loss function's geometry and changes throughout an orbit where the NN function remains constant. (c) The Riemannian gradient norm follows the loss function's symmetry and remains constant throughout an orbit, i.e., it does not suffer from ambiguities for two points in parameter space that represent the same NN function. specific symmetry and techniques to deal with it, we ask: Can we provide a one-size-fits-many recipe for developing symmetry-invariant quantities for a wider range of continuous symmetries? Here, we positively answer this question by proposing a principled approach to eliminate ambiguities stemming from symmetry. Essentially, this boils down to using the geometry that correctly captures symmetry-imposed parameter equivalences. We apply concepts from Riemannian geometry to work on the Riemannian quotient manifold implied by a symmetry group (Boumal, 2023, §9). We thus identify objects on the quotient manifold—like the Riemannian metric and gradient—and show how to translate them back to the Euclidean space. Our contributions are the following:
(a) We introduce the application of Riemannian geometry (Boumal, 2023) to the study of NN parameter space symmetries by using geometry from the quotient manifold induced by a symmetry as a general recipe to remove symmetry-induced ambiguities in parameter space. We do so by translating concepts like gradients from the quotient manifold back to the original space through *horizontal lifts*.

(b) Based on (a), we propose and analyze geodesic sharpness, a novel adaptive sharpness measure: By Taylorexpanding our refined geometry, we show that (i) symmetries introduce curvature into the parameter space, which (ii) results in previous adaptive sharpness measures when ignored. Geodesic sharpness differs from traditional sharpness measures in two key aspects: (i)
the norm of the perturbation parameter is redefined to reflect the underlying geometry; (ii) perturbations follow geodesic paths in the quotient manifold rather than straight lines in the ambient space.

(c) For diagonal nets, we analytically solve geodesic sharpness and find a strong correlation with generalization. Then, we apply our approach to the unstudied and higher-dimensional GL(h) symmetry in the attention mechanism. On both large vision transformers and language models, we empirically find stronger correlation than any previously seen (that we are aware of) between our geodesic sharpness and generalization.

## 2. Related Work

Symmetry versus reparameterization: Kristiadi et al. (2023) pointed out how to fix ambiguities stemming from reparameterization, i.e. a change of variables to a new parameter space: Invariance under reparameterization follows by correctly transforming the (often implicitly treated) Riemannian metric into the new coordinates. Our work focuses on invariance of the parameter space M under a symmetry group G with action ψ : G × M → M, (g, θ) 7→ ψ(g, θ) that operates on a *single* parameter space. Symmetry teleportation: Another ways to use symmetryimplied ambiguity is to view it as a degree of freedom and develop adaptation heuristics to improve algorithms which are not symmetry-agnostic (Zhao et al., 2022a). Geometric constraints & NN dynamics: Previous studies analyze how parameter space symmetries impose geometric constraints on derivatives and introduce conserved quantities during training (Kunin et al., 2021). Our approach differs by systematically removing symmetry-induced ambiguity through quotienting out the the symmetry group.

## 3. Preliminary Definitions, Notation & Math

We generalize earlier post-hoc solutions for simpler symmetries (e.g., GL(1)) to more complex, higher-dimensional symmetries such as GL(h), common in neural network attention mechanisms. Unlike Kunin et al. (2021), who consider geometry in augmented spaces for simpler symmetries, we directly use the quotient space geometry. Objects are then 'lifted' back into the original space, yielding symmetrycorrected quantities. This method provides a principled framework capable of handling high-dimensional symmetries, leading to a more effective dimensionality reduction.

Quotient manifolds in deep neural networks: Rangamani et al. (2019) introduce a quotient manifold construction for re-scaling symmetries and then use the Riemannian spectral norm as a measure of worst-case flatness. This differs from our approach in several ways:
(a) Our approach is more general and contains both the GL(h) symmetry of transformers, and the original rescaling/scaling symmetry of CNNs/MLPs, rendering it applicable to a wider range of modern architectures.

(b) Our experimental setup is more challenging: we test on large-scale models (large transformers vs CNNs) and large-scale datasets (ImageNet vs CIFAR-10). Sharpness measures that account for re-scaling/scaling symmetries (e.g. adaptive sharpness) work quite well on CIFAR-10 with CNNs, and tends to break down on datasets like ImageNet with transformers.

(c) Conceptually, Rangamani et al. (2019) defines worstcase sharpness on the usual norm-ball, appropriately generalized to the Riemannian setting. We propose instead that the ball should be the one traced out by geodesics, to better respect the underlying geometry.

(d) Performance-wise, our approach is cheaper as it does not use the Hessian and only uses symmetry-corrected gradients (see Dagreou et al. ´ (2024) for an in-depth cost comparison of computing Hessians vs gradients).

Relative sharpness: Another promising approach to sharpness was proposed by Petzka et al. (2021), where the generalization gap is shown to admit a decomposition into a representativeness term and a feature robustness term. Focusing on the feature robustness term, they introduce relative sharpness, which is invariant to a layer- and neuron-wise re-scaling, and performs better than traditional sharpness measures (Adilova et al., 2023; Walter et al., 2025).

Generalization measures: We consider a neural net fθ with parameters θ ∈ R
dthat is trained on a data set Dtrain using a loss function ℓ by minimizing the empirical risk

$$L_{\mathbb{D}_{\mathrm{train}}}(\theta):={\frac{1}{|\mathbb{D}_{\mathrm{train}}|}}\sum_{(\mathbf{x},\mathbf{y})\in\mathbb{D}_{\mathrm{train}}}\ell(f_{\theta}(\mathbf{x}),\mathbf{y})\,.$$

Our goal is to compute a quantity on the training data that is predictive of the network's generalization, i.e. performance on a held-out data set. Sharpness: A popular way to predict generalization is via sharpness—i.e., how much the loss changes when perturbing the weights in a small neighbourhood—like average
(Savg) or worst-case sharpness (Smax) (Keskar et al., 2017)

$$\begin{array}{l}{{S_{\mathrm{avg}}=\mathbb{E}_{\mathbb{S}}\left[L_{\mathbb{S}}(\theta+\delta)-L_{\mathbb{S}}(\theta)\right],\quad\delta\sim\mathcal{N}(\mathbf{0},\rho^{2}\mathbf{I})\,,}}\\ {{S_{\mathrm{max}}=\mathbb{E}_{\mathbb{S}}\left[\operatorname*{max}_{\|\delta\|_{2}\leq\rho}\left(L_{\mathbb{S}}(\theta+\delta)-L_{\mathbb{S}}(\theta)\right)\right]\,,}}\end{array}$$

with batches S ∼ Dtrain of size |S| = m, neighbourhood size ρ, and perturbation δ. Near critical points, they closely relate to the Hessian H (and thus parameter space curvature):
Savg ∝ Tr(H), and Smax ∝ λmax(H).

Adaptive sharpness: Hessian-based sharpness measures can be made to assume arbitrary values by rescaling parameters, even though the NN function stays the same. To fix this inconsistency, Kwon et al. (2021) proposed adaptive sharpness (invariant under special symmetries), and Andriushchenko et al. (2023) use adaptive notions of sharpness that are invariant to element-wise scaling,

$$S_{\max}^{\rm ad}(\mathbf{w},\mathbf{c})=\mathbb{E}_{\mathbb{S}}\left[\max_{\|\mathbf{\delta}\circ\mathbf{c}\|_{2}\leq\rho}L_{\mathbb{S}}(\mathbf{\theta}+\mathbf{\delta})-L_{\mathbb{S}}(\mathbf{\theta})\right],\tag{1}$$

with scaling vector c (usually set to |θ|, Kwon et al., 2021). The problem: Adaptive sharpness only considers the symmetry induced by element-wise re-scaling. But symmetries of transformers go beyond the invariance that adaptive sharpness captures. Maybe unsurprisingly, Andriushchenko et al. (2023) find inconsistent trends for adaptive sharpness in transformers, with sharpness failing to correlate with generalisation, versus other architectures. We hypothesize this is related to adaptive sharpness not accounting for the full symmetry in transformers. In this paper, we address this. The central question is: If adaptive sharpness is the fix for a special symmetry, can we provide a more general solution for the symmetries of transformers, to fix the above inconsistency?

## 3.1. Symmetries In Neural Networks

Here, we give a brief overview and make the notion of NN symmetries more concrete, focusing on those studied by Kunin et al. (2021). Those symmetries lead to rather small effective dimensionality reduction as they are often of GL(1) or GL+(1), but they can still impact the network behaviour. Let θ denote the parameters of a neural net, 1A a binary mask, and 1¬A its complement such that their sum is a vector of ones, 1A +1¬A = 1. Let θA := θ ⊙1A with ⊙ the element-wise product. Further, let A1,2 be two disjoint subsets, A1 ∩ A2 = ∅ with masks 1A1
, 1A2. Then we have the following common symmetries, characterized by their symmetry group G, such that for any g ∈ G the parameters ψ(g, θ) and θ represent the same function:
- **Translation:** ψ(α, θ) = 1A ⊙ α + θ with α ∈ R
h
- **Scaling:** ψ(α, θ) = αθA + θ¬A with α ∈ R>0

 - **Re-scaling:** $\psi(\alpha,\theta)=\alpha\theta_{\mathcal{A}_1}+\sfrac{1}{2}/\alpha\theta_{\mathcal{A}_2}+\theta_{\neg(\mathcal{A}_1\lor\mathcal{A}_2)}$ with $\alpha\in\mathbb{R}_{\geq0}$                
Their associated groups are G = R
h, GL+(1), GL+(1). In practice, there may be multiple symmetries acting onto disjoint parameter sub-spaces. Note that re-scaling is essentially the symmetry that adaptive sharpness corrects for.

## 3.2. Rescale Symmetry Of Transformers

Transformers exhibit a higher-dimensional symmetry than the previous examples; we formalize the treatment of this symmetry in the following canonical form. Definition 3.1 (Functional GL-symmetric building block).

Consider a function f(G, H) on R
m×h × R
n×hthat consumes two matrices G ∈ R
n×h, H ∈ R
m×h but only uses the product GH⊤, i.e. f(G, H) = g(GH⊤) for some g over R
m×n. f is symmetric under the *general linear group*

GL($h$) = {$A\in\mathbb{R}^{h\times h}$ | $A$ invertible}  
with dim(GL(h*)) =* h 2and action

$$\psi(A,(G,H))=(G A^{-1},H A^{\top}).$$

In other words, we can insert and then absorb the identity A−1A into G, H to obtain equivalent parameters GA−1, HA⊤ that represent the same function.

Example A.2 illustrates GL symmetry for a shallow linear net. Indeed, many popular NN building blocks feature this form, most prominently the attention mechanism in transformers (Vaswani et al., 2017). We give the attention symmetry in Example A.1, and we provide the symmetry for low-rank adapters (Hu et al., 2022) in Example A.3. These examples are NN building blocks that introduce GL symmetries into a loss function and can all be treated through the canonical form in Definition 3.1. In contrast to the symmetries from Section 3.1, they lead to more drastic dimensionality reduction. Consider for example a single self-attention layer where h = hv = hk. The number of trainable parameters is 4h 2and the two GL(d) symmetries reduce the effective dimension to 4h 2−2 dim(GL(h)) = 2h 2, i.e. they render *half* the parameter space redundant. We hypothesize that the impact of a low-dimensional symmetry on objects like the Euclidean Hessian's trace (Dinh et al., 2017) may be amplified for such higher-dimensional symmetries.

## 3.3. Mathematical Concepts For Riemannian Geometry

We now outline required properties of manifolds for the full development of our approach. We list essential concepts here, and provide definitions and a brief review Appendix B.

For further information, see for instance Lee (2003). Figure 2 illustrates the main concepts we will require. Ambient embedding space: We assume that the manifold of possible parameters is embedded in a linear Euclidean space E ≃ R
d with d the number of parameters. We can think of E as the *ambient space*. For instance, for a loss function ℓ : E → R, θ 7→ ℓ(θ) , we can use ML libraries to evaluate its value, as well as its Euclidean gradient

$$\operatorname{grad}_{\boldsymbol{\theta}}{\overline{{\boldsymbol{\ell}}}}=\left({\frac{\partial{\overline{{{\boldsymbol{\ell}}}}}({\boldsymbol{\theta}})}{\partial\theta_{i}}}\right)_{i=1,\dots,d}\in\mathbb{R}^{d}\,.$$

Because the geometry of E is flat, i.e. uses the standard metric ⟨θ1, θ2⟩ := θ
⊤
1 θ2, this object consists of partial derivatives. However, the Riemannian generalization will add correction terms. In what follows we consider only the restriction of objects like ℓ to the parameter manifold.

Definition 3.2. We take M to be the manifold of network parameters, and consider it a sub-manifold embedded into E, the computational space of matrices on which all our numerical calculations are done. We call M the *total space*.

On the total space we have a loss function ℓ : M → R.

$$({\mathfrak{I}})$$

Our goal is to calculate derivatives/geometric quantities after removing the NN's symmetries. The symmetry relation induces natural equivalence classes, which we write [θ], and explain in Appendix B.1. We let M = M/ ∼ represent the *quotient* of the original parameter space manifold by the equivalence relation ∼ associated with the symmetry (Appendix B.2). We also require *tangent vectors*; these are straightforward on the total space M, but the tangent space of the quotient manifold, M, requires more machinery: vertical and *horizontal spaces*, and corresponding *lift*s. These concepts are all defined in Appendix B.3.

Once we endow M with a smooth inner product over its tangent vectors, we obtain a *Riemannian manifold* (defined in Appendix B.4). This construction lets us analyze differ-

x *= [¯x] =* x′
E
M = M/G
E Ambient embedding space M Total space M Quotient space G Symmetry group x, ¯ y¯ Points on the total space x, y Points on the quotient space
¯ξx¯ Tangent vector in the tangent space at point x¯, Tx¯M
ξx Tangent vector in the tangent space at point x, TxM
¯ξ V
x¯ Vertical component of ¯ξx¯ in the vertical space Vx¯M
¯ξ H
x¯ Horizontal component of ¯ξx¯ in the horizontal space Hx¯M ≃ TxM, horizontal lift of ξx ξx y M
x¯′
x¯
¯ξx¯
¯ξ V
x¯
¯ξ H x¯
y¯
ential objects that live on quotient manifolds, in the ambient space in a natural way. Furthermore, this allows us to define the horizontal space as the orthogonal complement of the vertical space (Appendix B.4), and to define a Riemannian gradient (Appendix B.5). Most properties from the Euclidean case still hold for the Riemannian gradient, but of particular interest to us is the fact that the direction gradf(x) is still the steepest-ascent direction of f at a point x. We additionally make use of *geodesic curves*. Intuitively, geodesic curves can either be seen as curves of minimal distance between two points on a manifold M, or equivalently, as curves through a given point with some initial velocity, and whose acceleration is zero—a generalization of Euclidean straight lines. See Appendix B.6 for details. Putting it all together, this gives us a *recipe* for computing quantities invariant to a given symmetry relation: (i) find a Riemannian metric compatible with this symmetry; (ii) determine the vertical space for the symmetry relation; (iii) use the metric to find the orthogonal complement of this vertical space, i.e. the projector into the horizontal space; (iv) find the horizontal geodesics. Combined, these steps allow us to do calculations in the quotient manifold along the proper paths (given by geodesics).

## 4. Geodesic Sharpness

We posit that adaptive sharpness measures should take into account the geometry of the quotient parameter manifold that arises after removing symmetries from the parameter space. We base our sharpness measure on the notion of a *geodesic ball*: the set of points that can be reached by geodesics, starting at a point p and whose initial velocity has a norm smaller than ρ, after one time unit. In R
dthis is just the usual definition of a ball, since the geodesics are straight lines. If ¯ξ ∈ Hx¯M is a horizontal vector, and γ¯(t) is a geodesic starting at θ and with initial velocity ¯ξ:

$$S_{\max}^{\rho}(\mathbf{w})=\mathbb{E}_{\mathbb{S}}\left[\max_{\|\tilde{\mathbf{\xi}}\|_{\mathbf{\gamma}(0)}\leq\rho}L_{\mathbb{S}}(\tilde{\mathbf{\gamma}}\mathbf{\xi}(1))-L_{\mathbb{S}}(\tilde{\mathbf{\gamma}}\mathbf{\xi}(0))\right].\tag{4}$$

If the initial velocity, ¯ξ, is a horizontal vector, then the velocity of the geodesic, γ¯˙ξ¯, will stay horizontal. The choice of t = 1 in γ¯ξ¯(1) is not as arbitrary as it first seems (do Carmo, 1992): since for a positive a, γ¯ξ¯(at) = γ¯aξ¯(t), positions reached with arbitrary t can be reached by instead fixing t = 1 and manipulating the initial velocity's norm via ρ. When we do not have an analytical solution for the geodesic, we can use the approximation:

$$\bar{\gamma}_{\bar{\xi}}^{i}(t)=\bar{\gamma}_{\bar{\xi}}^{i}(0)+\bar{\xi}^{i}t-\frac{1}{2}\Gamma_{\;\;k l}^{i}\bar{\xi}^{k}\bar{\xi}^{l}t^{2}+\mathcal{O}(\bar{\xi}^{3})\,,\quad\,$$

where ¯ξ = ( ¯ξ i) is the initial (horizontal) velocity, and Γ
ikl are the Christoffel symbols. We show that geodesic sharpness reduces to adaptive sharpness measures in Appendix F, under appropriate metric choices and by taking a first-order approximation to the geodesics, that is, ignoring the terms corresponding to the curvature, Γ
i kl.

## 5. Geodesic Sharpness In Practice

We now apply geodesic sharpness to concrete examples. A fully worked out scalar toy model is in Appendix D. Following previous works by Dziugaite et al. (2020); Kwon et al. (2021); Andriushchenko et al. (2023), we use the Kendall rank correlation coefficient (Kendall, 1938) to assess the correlation between generalization and sharpness in the empirical validations of our approach:

$$\tau(\mathbf{t},\mathbf{s})={\frac{2}{M(M-1)}}\sum_{i<j}\operatorname{sign}(t_{i}-t_{j})\operatorname{sign}(s_{i}-s_{j})\,,$$

where t and s are the vectors of observed variables between which we are measuring correlation. Although the criterion of symmetry compatibility restricts the class of suitable metrics, these are not necessarily unique. As long as it is symmetry-compatible, we have no reason to prefer one metric over another, except for practical aspects like numerical cost and stability. We will present results on two symmetry-compatible metrics that are simple, yet nontrivial, and often used in the related literature on Riemannian optimization on fixed-rank matrix spaces (Luo et al., 2023).

## 5.1. Diagonal Networks

We start by studying *diagonal linear nets*, one of the simplest non-trivial neural networks (Pesme et al. (2021), Woodworth et al. (2020)). These have two parameters, u, v, and predict a label, y, given an input, x, via y = x
⊤(u ⊙ v).

We consider linear regression with labels y ∈ R
n, a data matrix X ∈ R
n×d, and take as our loss L(u, v) =
∥X(u ⊙ v) − y∥
22. Our parameter manifold M is R
d × R
d.

The nets are symmetric under element-wise rescaling:
(u, v) 7→ (αu, α−1v), leaves β= u ⊙ v and L invariant.

Metric: At a point (u, v) ∈ M, for two tangent vectors η = (ηu, ηv), ν = (νu, νv) ∈ T(u,v)M, we use the following two symmetry-compatible metrics:

$$\langle\eta,\nu\rangle^{\rm inv}:=\sum_{i=1}^{d}\frac{\eta_{\mathbf{u}}^{i}\nu_{\mathbf{u}}^{i}}{(\mathbf{u}^{i})^{2}}+\frac{\eta_{\mathbf{v}}^{i}\nu_{\mathbf{v}}^{i}}{(\mathbf{v}^{i})^{2}}\,,\tag{6}$$  $$\langle\eta,\nu\rangle^{\rm mix}:=\sum_{i=1}^{d}\eta_{\mathbf{u}}^{i}\nu_{\mathbf{u}}^{i}(\mathbf{v}^{i})^{2}+\eta_{\mathbf{v}}^{i}\nu_{\mathbf{v}}^{i}(\mathbf{u}^{i})^{2}\,.\tag{7}$$

Horizontal space: Both have the same horizontal space

$${\mathcal{H}}_{(u,v)}{\overline{{{\mathcal{M}}}}}=\left\{(\eta_{u},\eta_{v})\in T_{(u,v)}{\mathcal{M}}\mid{\frac{\eta_{u}^{i}}{u^{i}}}={\frac{\eta_{v}^{i}}{v^{i}}}\;\;\forall i\right\}\;.$$

```
Geodesics: With bi:=
                      η
                       i
                       u
                      ui =
                           η
                            i
                            v
                           vi, the geodesics are

```

$$\begin{array}{l}{{\gamma_{\mathrm{inv}}(t)^{i}=\left(\mathbf{u}_{0}^{i}\exp(\mathbf{b}_{i}t),\mathbf{v}_{0}^{i}\exp(\mathbf{b}_{i}t)\right)\,,}}\\ {{\gamma_{\mathrm{mix}}(t)^{i}=\left(\mathbf{u}_{0}^{i}{\sqrt{1+2\mathbf{b}_{i}t}},\mathbf{v}_{0}^{i}{\sqrt{1+2\mathbf{b}_{i}t}}\right)\,,}}\end{array}$$

with starting points u i0and v i0, i.e. the trained parameters.

Geodesic sharpness: Assume that X⊤X = Id (Andriushchenko et al., 2023), and denote β0 = u0 ⊙ v0.

Gene raliza tion Ga p τ =-0.69 τ =-0.86 τ =-0.83 2.0 2.2 Adaptive 0.0 0.5 1.0 1.5 2 4 Geodesic (inv)
2.5 3.0 3.5 Geodesic (mix)
The minimum norm least squares predictor is β∗ :=
(X⊤X)
−1X⊤y = X⊤y. Using Equation (4) (details in Appendix E), we get (to first and to second order)

$$S^{\rho}_{\rm max;\;inv}(\mathbf{u},\mathbf{v})=4\rho||\mathbf{\beta}_{0}\odot(\mathbf{\beta}_{0}-\mathbf{\beta}_{*})||_{2}\tag{8}$$ $$+4\rho^{2}\max\left[(\beta_{0}^{i})^{2}\right]\,,$$

which depends on ρ and the difference between the learned, and the optimal minimum norm, predictor. Eq. 8 is the square of adaptive sharpness (when the residual ∥β0⊙(β0− β∗)∥2 is small) if very carefully chosen hyperparameters were used (by contrast, this result naturally appears using our geodesic approach). For the second metric, we have

$$S_{\mathrm{max};\;\mathrm{mix}}^{\rho}(\mathbf{u},\mathbf{v})=\rho\|\beta_{0}-\beta_{*}\|_{2}\;.$$

## 5.1.1. Empirical Validation

Experimental setup: We follow Andriushchenko et al. (2023), generate a randomly distributed data matrix X, a random ground-truth vector β
∗that is 90% sparse, and train 50 diagonal networks to 10−5training loss on a regression task. We focus on the more practically relevant case of overparametrization (*d > n*). One downside of this is that the theoretical expressions derived in the previous section, while a useful sanity check, no longer hold (since overparameterization breaks the assumption X⊤X = Id=200). To obtain our geometric sharpness, we directly solve Equation (4). Results: All three notions of sharpness are able to predict generalization (Figure 3). Geodesic sharpness, although closely related for diagonal nets to adaptive worst-case sharpness, does slightly better. This applies to both metrics studied, and they perform roughly the same. See Section 7 for comments about the sign of the correlation.

## 5.2. Attention Layers

Next, we look at the symmetric functional block from Definition 3.1. Our computation space is E := R
n×h×R
m×h ≃

```
R
 
 (n+m)hand we restrict weights to have full column rank:

```

Assumption 5.1. The rank of G, H corresponds to their number of columns, rank(G) = rank(H) = h. This implies h ≤ *n, m*, which is usually satisfied in (multihead) attention layers (Example A.1) for the default choices of dv, dk. While the weights of multi-head attention tend to have high column rank (Yu & Wu, 2023), they are not guaranteed to be full column rank. To account for this, we introduce a small relaxation parameter, ϵ, to the Gram matrices s.t. G⊤G → G⊤G + ϵIh. Empirically, we observe that as long as ϵ is sufficiently small, it does not affect our results (Appendix H.2). Therefore, we restrict both G, H to the set of fixed-rank matrices, M ← R
n×h h × R
m×h h where R 
n×h k:=B ∈ R
n×h| rank(B) = k	. We can represent a point x¯ ∈ M by a matrix tuple (G, H) ∈ R
n×h h ×R
m×h h.

Its tangent space Tx¯M is

$$\mathrm{T}_{\bar{x}}\overline{{{\mathcal{M}}}}=\left\{\bar{\eta}=(\bar{\eta}_{G},\bar{\eta}_{H})\in\mathbb{R}^{n\times h}\times\mathbb{R}^{m\times h}\right\}\,,$$

Metric: We endow M with the two metrics ⟨·, ·⟩inv,mix x¯:
Tx¯M × Tx¯M → R (proof they are valid in Appendix I.1):

⟨η¯,¯ζ⟩ inv x¯:= Tr  (G ⊤G) −1η¯ ⊤G ¯ζG + (H⊤H) −1η¯ ⊤H ¯ζH , (9) ⟨η¯,¯ζ⟩ mix x¯:= Tr  (H⊤H)η¯ ⊤G ¯ζG + (G ⊤G)η¯ ⊤H ¯ζH . (10)
They differ from the Euclidean metric that simply flattens and concatenates the matrix tuples into vectors and takes their dot product, ⟨η, ζ⟩ = Tr η
⊤GζG + η
⊤HζH. Importantly, they are invariant under symmetries of the attention mechanism, and thus define valid metrics on the quotient manifold (Absil et al., 2008).

Horizontal space: For ⟨·, ·⟩inv, mix x¯ and ¯ξG,H ∈ R
m×r we have (for a proof, see for example Luo et al. (2023))

$$\begin{array}{l}{{{\mathcal{H}}_{\bar{x}}^{\mathrm{inv}}\overline{{{\mathcal{M}}}}=\left\{\left(\bar{\xi}_{G},\bar{\xi}_{H}\right)\mid\bar{\xi}_{G}^{\top}G H^{\top}H=G^{\top}G H^{\top}\xi_{H}^{\top}\right\},}}\\ {{{\mathcal{H}}_{\bar{x}}^{\mathrm{mix}}\overline{{{\mathcal{M}}}}=\left\{\left(\bar{\xi}_{G},\bar{\xi}_{H}\right)\mid G^{\top}\bar{\xi}_{G}H^{\top}H=G^{T}G\xi_{H}^{\top}H\right\}.}}\end{array}$$

Projection onto horizontal space: Given ξ ∈ TxM in the total tangent space, the horizontal space is

$${\cal H}_{\overline{{{x}}}}^{\mathrm{inv,\,mix}}{\overline{{{\cal M}}}}=\left\{(\bar{\xi}_{G}+G\Lambda^{\mathrm{inv,\,mix}},\bar{\xi}_{H}-H(\Lambda^{\mathrm{inv,\,mix}})^{\top})\right\},$$

where Λinv solves the Sylvester equation AΛ + ΛA⊤ =
B, with A = G⊤GH⊤H, B = G⊤GH⊤ ¯ξH −
¯ξ
⊤GGH⊤H, whereas Λmix has an explicit form: Λmix =
1/2¯ξ
⊤HH(H⊤H)
−1 − (G⊤G)
−1G⊤ ¯ξG.

Geodesics: We are unaware of analytical solutions for the geodesics of either (Eq. 9 and Eq. 10), so we approximate

Gene raliza tion Ga p τ =-0.41 τ =-0.71 τ =-0.70 5 10 Adaptive 0.05 0.10 0.15 0.20 10 20 Geodesic (inv)
2 4 Geodesic (mix)
them with Eq. 5. For horizontal tangent vectors (
¯ξG,
¯ξH),
we have for ⟨·, ·⟩inv x¯

$$(\Gamma^{i}_{kl})^{\rm inv}\bar{\xi}^{k}_{G}\bar{\xi}^{l}_{G}=-\ \bar{\xi}_{G}(G^{\top}G)^{-1}\left[\bar{\xi}^{\top}_{G}G+G^{\top}\bar{\xi}_{G}\right]\tag{11}$$ $$+\ G(G^{\top}G)^{-1}\bar{\xi}^{\top}_{G}\bar{\xi}_{G}$$

(similar for the H components). For ⟨·, ·⟩mix x¯, the geodesic equations are coupled and the G components are

$$\begin{split}\left(\Gamma^{i}_{kl}\right)^{\text{mix}}\bar{\xi}^{k}\bar{\xi}^{l}\right]_{G}=&\bar{\xi}_{G}\left[\bar{\xi}^{\top}_{H}H+H^{\top}\bar{\xi}_{H}\right]\left(H^{\top}H\right)^{-1}\\ &-G(\bar{\xi}^{T}_{H}\bar{\xi}_{H})(H^{\top}H)^{-1}\end{split}\tag{12}$$

(the H components are similar, proof in Appendix I.2).

## 5.3. Transformers

Transformers have a mix of attention layers and layers with more restricted symmetries for which adaptive sharpness is more appropriate. We present in Appendix C.1 how we treat each layer of transformers. We introduce relaxations In Appendix C.2 we present Algorithm 1, which we use to solve for geodesic sharpness.

5.3.1. EMPIRICAL VALIDATION: VISION TRANSFORMERS
Experimental setup: We follow Andriushchenko et al.

(2023), and look at models obtained from fine-tuning CLIP on ImageNet-1k (Radford et al., 2021). Specifically, we use the trained classifiers after fine-tuning a CLIP ViT-B/32 on ImageNet with randomly selected hyperparameters from (Wortsman et al., 2022). We compute adaptive worst-case, and our geodesic, sharpness on the same 2048 data points from the ImageNet training set, divided into batches of 256, by calculating sharpness on each batch separately, then averaging the results. The generalization gap is the difference between test and training error.

Results: Figure 4 shows our results. We find a strong correlation between geodesic sharpness and the generalization gap on ImageNet. This correlation is stronger than

Gene raliza tion Ga p τ =0.06 G
eneraliz ati on Gap τ =0.06 τ =0.28 τ =0.38 τ =0.38 0.020 0.022 0.024 Adaptive 0.102 0.104 0.106 0.108 0.020 0.022 0.024 Adaptive 0.102 0.104 0.106 0.108 6.5 7.0 7.5 Geodesic (inv)
5.00 5.25 5.50 Geodesic (mix)
5.00 5.25 5.50 Geodesic (mix)
that observed with adaptive sharpness and is consistently negative, implying that the geodesically sharpest models studied on ImageNet are those that generalize best–contrary to what might have been expected, but consistent with the correlation from the diagonal networks.

## 5.3.2. Empirical Validation: Language Models

Experimental Setup: We also consider BERT models that were fine-tuned on MNLI (Williams et al., 2018) by Mc-
Coy et al. (2020) . We compute adaptive worst-case, and our geodesic, sharpness on the same 1024 data points from the MNLI training set, with batches of 128 points, by calculating then averaging sharpness on each batch.

Results: Figure 5 shows our results. We find a consistent correlation between geodesic sharpness and the generalization gap on MNLI for both metrics, while adaptive sharpness (τ = 0.06) cannot find any correlation. The correlation is positive, i.e. geodesically flatter models generalize better.

## 6. Additional Experiments 6.1. Comparison With Relative Sharpness

Relative sharpness (Petzka et al., 2021) is a promising sharpness measure that has proven useful in regularizing transformer training, outperforming other approaches (Adilova et al., 2023). We compare it with our geodesic sharpness in the language model setting from Section 5.3.2; see Figure 6.

## 6.2. Verification Of Reparametrization Invariance

Mathematically, geodesic sharpness is invariant to symmetry transformations of the form of Equation (3). Here, we verify empirically that our practical version that can be computed efficiently numerically is close to invariant. Experimental setup: We take a single batch and language model from Section 5.3.2, and compute geodesic sharpness

4 6 Relative
τ =-0.09 Figure 6: **Extension of Figure** 5 to relative sharpness. We find that relative flatness (Petzka et al., 2021) fails to find a significant correlation, compared to our geodesic sharpness.

for various points on an orbit that represent the same function. Specifically, we reparametrize using A = aG, where G is a random standard Gaussian matrix (which is almost always invertible and sampled once in each run), and control the scale a. We sample one G for each attention head. We compare this with adaptive sharpness. Results: Figure 7 visualizes the sharpness ratio before and after reparameterization. The colors represent different values of the scale factor, which goes from 10−2to 102. Our numerically computed geodesic sharpness remains constant.

10−2 10−1 100 101 102 Scale factor (a)
1.0 1.2 1.4 1.6 Sha rpn ess r atioAdapative Geodesic

## 7. Remarks, Limitations & Future Work

Discovering correlation: Adaptive sharpness, as discussed thoroughly by Andriushchenko et al. (2023), is unable to reveal a correlation between sharpness and generalization for transformers. Our geodesic sharpness consistently recovers strong correlation on transformers, and strengthens the correlation in the case of diagonal networks. Metric choice: Our results are robust w.r.t. the choice of metric, as long as *it captures the parameter symmetry*. The mixed metric yields slightly better results on BERT, perhaps owing to its more stable numerics (e.g. possible inversion of nearly singular matrices is side-stepped). Additionally, the mixed metric avoids calling expensive Sylvester equation solvers and has a simple horizontal space projection.

Sign of the correlation: One of our surprising results is that the sign of the correlation between geodesic sharpness and generalization varies depending on the setting and is at times negative, somewhat at odds with the common view that sharpness always *positively* correlates with generalization (i.e., flatter models generalize better). This artifact is not inherent to our proposed metrics. E.g., adaptive sharpness anti-correlates with generalization in our diagonal network setting, but was previously found to positively correlate with generalization on other tasks (Kwon et al., 2021). Our geodesic sharpness improves over adaptive sharpness in the following sense: Where adaptive sharpness finds no correlation, our metrics do find a signed correlation, and where adaptive sharpness finds signed correlation, our metrics find a stronger similarly-signed correlation. That is, we for the first time observe *consistent correlations* withintask for transformers, opening questions for further study. Limitations: While our *geodesic sharpness* is more general than previous measures, there remain symmetries for which taking the quotient may be computationally expensive or intractable. Still, we think that accounting for some symmetry is better than none, and even under computational constraints it could be useful as a diagnostic "probe". Our new measures detect previously undetected correlation with generalization. In the process, however, we also discovered that the sign of the correlation, while consistent across metrics and models, can vary across tasks. Until this new variability is understood, this limits the utility of geodesic sharpness, e.g. for regularizing transformer training. Future work: Our work is concerned with accounting for parameter space symmetries that are data-independent. This opens up the question: what is the role of data and how can it be integrated into our framework? A more complete understanding of the interplay between data and parameter symmetries might help explain when geodesic sharpness correlates or anti-correlates with generalization.

## 8. Conclusion

In this paper, we revisited the limitations of traditional sharpness measures attempting to predict generalization for transformers, highlighting how traditional sharpness measures fail to properly account for the rich GL(h) symmetries present in transformers. Addressing this, we introduced geodesic sharpness, a measure defined on the Riemannian quotient manifold obtained by quotienting out transformer symmetries. This framework provides a principled, symmetry-aware measure of sharpness and contains prior adaptive sharpness metrics as first-order approximations. Through experiments on diagonal networks, vision transformers (ImageNet), and language models (MNLI), we demonstrated that properly accounting for the transformer symmetries restores the correlation between sharpness and generalization. Interestingly, our findings indicate that the sign of the correlation between sharpness and generalization can vary across tasks, suggesting deeper underlying relationships involving data distribution and model structure. This work lays the groundwork for further exploration of these interactions and motivates future research into geometryinformed optimization strategies tailored to transformers.

## Impact Statement

This paper presents work whose goal is to advance the study of deep learning. There are potential indirect societal consequences of our work, none which we feel must be specifically highlighted here.

## Acknowledgements

We would like to express our sincere gratitude to Agustinus Kristiadi and Rob Brekelmans for early feedback on the manuscript. Resources used in preparing this research were provided, in part, by NSERC, the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute.

## References

Absil, P.-A., Mahony, R., and Sepulchre, R. Optimization algorithms on matrix manifolds. 2008.

Adilova, L., Abourayya, A., Li, J., Dada, A., Petzka, H., Egger, J., Kleesiek, J., and Kamp, M. Fam: Relative flatness aware minimization, 2023.

Andriushchenko, M., Croce, F., Muller, M., Hein, M., and Flam- ¨
marion, N. A modern look at the relationship between sharpness and generalization. 2023.

Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization.

2016.

Boumal, N. *An introduction to optimization on smooth manifolds*.

Cambridge University Press, 2023.

Croce, F. and Hein, M. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks, 2020.

Dagreou, M., Ablin, P., Vaiter, S., and Moreau, T. How to compute ´
hessian-vector products? In *ICLR Blogposts 2024*, 2024.

Dinh, L., Pascanu, R., Bengio, S., and Bengio, Y. Sharp minima can generalize for deep nets, 2017.

do Carmo, M. *Riemannian Geometry*. Mathematics (Boston, Mass.). Birkhauser, 1992. ISBN 9783764334901. ¨
Dziugaite, G. K., Drouin, A., Neal, B., Rajkumar, N., Caballero, E., Wang, L., Mitliagkas, I., and Roy, D. M. In search of robust measures of generalization. In Advances in Neural Information Processing Systems, volume 33, pp. 11723–11733. Curran Associates, Inc., 2020.

Foret, P., Kleiner, A., Mobahi, H., and Neyshabur, B. Sharpnessaware minimization for efficiently improving generalization. In International Conference on Learning Representations (ICLR), 2021.

Hochreiter, S. and Schmidhuber, J. Simplifying neural nets by discovering flat minima. In Advances in Neural Information Processing Systems (NIPS), 1994.

Hu, E. J., yelong shen, Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S.,
Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022.

Ioffe, S. and Szegedy, C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning (ICML), 2015.

Kendall, M. G. A new measure of rank correlation. *Biometrika*,
30(1-2):81–93, 1938.

Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., and Tang, P. T. P. On large-batch training for deep learning: Generalization gap and sharp minima, 2017.

Kirrinnis, P. Fast algorithms for the sylvester equation. Theoretical Computer Science, 259(1):623–638, 2001. ISSN 0304-3975.

Kristiadi, A., Dangel, F., and Hennig, P. The geometry of neural nets' parameter spaces under reparametrization. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Kunin, D., Sagastuy-Brena, J., Ganguli, S., Yamins, D. L., and Tanaka, H. Symmetry, conservation laws, and learning dynamics in neural networks. In *International Conference on Learning* Representations (ICLR), 2021.

Kwon, J., Kim, J., Park, H., and Choi, I. K. Asam: Adaptive sharpness-aware minimization for scale-invariant learning of deep neural networks. In *International Conference on Machine* Learning (ICML), 2021.

Lee, J. *Introduction to Smooth Manifolds*. Graduate Texts in Mathematics. Springer, 2003. ISBN 9780387954486.

Luo, Y., Li, X., and Zhang, A. R. On geometric connections of embedded and quotient geometries in riemannian fixed-rank matrix optimization, 2023.

McCoy, R. T., Min, J., and Linzen, T. Berts of a feather do not generalize together: Large variability in generalization across models with similar test set performance, 2020.

Pesme, S., Pillaud-Vivien, L., and Flammarion, N. Implicit bias of sgd for diagonal linear networks: a provable benefit of stochasticity, 2021.

Petzka, H., Kamp, M., Adilova, L., Sminchisescu, C., and Boley, M. Relative flatness and generalization. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. CoRR, abs/2103.00020, 2021.

Rangamani, A., Nguyen, N. H., Kumar, A., Phan, D., Chin, S. H.,
and Tran, T. D. A Scale Invariant Flatness Measure for Deep Network Minima, February 2019.

Salimans, T. and Kingma, D. P. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. Advances in neural information processing systems (NeurIPS), 29, 2016.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
Gomez, A. N., Kaiser, L. u., and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

Walter, N. P., Adilova, L., Vreeken, J., and Kamp, M. The uncanny valley: Exploring adversarial robustness from a flatness perspective, 2025.

Williams, A., Nangia, N., and Bowman, S. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 1112–1122, 2018.

Woodworth, B., Gunasekar, S., Lee, J. D., Moroshko, E., Savarese, P., Golan, I., Soudry, D., and Srebro, N. Kernel and rich regimes in overparametrized models, 2020.

Wortsman, M., Ilharco, G., Yitzhak Gadre, S., Roelofs, R., Gontijo-
Lopes, R., Morcos, A. S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., and Schmidt, L. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. *arXiv e-prints*, 2022.

Wu, Y. and He, K. Group normalization, 2018. Yen, J.-N., Si, S., Meng, Z., Yu, F., Surya Duvvuri, S., Dhillon, I. S., Hsieh, C.-J., and Kumar, S. LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization. arXiv e-prints, 2024.

Yu, H. and Wu, J. Compressing transformers: Features are lowrank, but weights are not! Proceedings of the AAAI Conference on Artificial Intelligence, 37(9):11007–11015, Jun. 2023.

Zhao, B., Dehmamy, N., Walters, R., and Yu, R. Symmetry teleportation for accelerated optimization. In *Advances in Neural* Information Processing Systems (NeurIPS), 2022a.

Zhao, Y., Zhang, H., and Hu, X. Penalizing gradient norm for efficiently improving generalization in deep learning. In *ICML*, 2022b.

# Hide & Seek: Transformer Symmetries Obscure Sharpness & Riemannian Geometry Finds It (Supplemental Material)

We provide in Table 1 a summary of correlation coefficients between sharpness and generalization for our experiments.

| Rank correlation coefficient τ   |                    |                                |                                  |
|----------------------------------|--------------------|--------------------------------|----------------------------------|
| Setting                          | Adaptive sharpness | ⟨·, ·⟩inv - geodesic sharpness | ⟨·, ·⟩mix- geodesic sharpness () |
| Diagonal networks                | -0.68              | -0.83                          | -0.86                            |
| ImageNet                         | -0.41              | -0.71                          | -0.7                             |
| MNLI                             | 0.06               | 0.28                           | 0.38                             |

Table 1: Summary of the correlation between sharpness measures and generalization. We boldface the best performing metric In the sections that follow, we provide additional details to supplement the main text.

## A. Additional Examples Of Gl Symmetries Symmetries In Neural Networks

Example A.1 (Self-attention (Vaswani et al., 2017)). Given a sequence X ∈ R
t×d with t tokens and model dimension d, self-attention (SA) uses four matrices Wq,Wk ∈ R
d×dk,Wv,W⊤
o ∈ R
d×dv(usually, d = dv = dk) to produce a new t × d sequence

$$\text{SA}(\mathbf{W}_{\text{q}},\mathbf{W}_{\text{k}},\mathbf{W}_{\text{v}},\mathbf{W}_{\text{o}})$$ $$=\text{softmax}\left(\frac{\mathbf{X}\mathbf{W}_{\text{q}}\mathbf{W}_{\text{k}}^{\top}\mathbf{X}^{\top}}{\sqrt{d_{\text{k}}}}\right)\mathbf{X}\mathbf{W}_{\text{v}}\mathbf{W}_{\text{o}}\,.$$  In a row of diagonal elements, the above column is the weighted 
$$(13)$$

This block contains two GL symmetries: one of dimension dk between the key and query projection weights, G, H ←
Wq,Wk, and one of dimension dv between the value and out projection weights, G, H ← Wv,W⊤
o. Similar to Eq. 14, we can account for biases in the key, query, and value projections by appending them to their weight,

$${\mathbf{G}},{\mathbf{H}}\leftarrow\left({\frac{W_{\mathrm{k}}}{{\mathbf{b}}_{\mathrm{k}}^{\top}}}\right),\left({\frac{W_{\mathrm{q}}}{{\mathbf{b}}_{\mathrm{q}}}}\right)^{\top},\quad{\mathbf{G}},{\mathbf{H}}\leftarrow\left({\frac{W_{\mathrm{v}}}{{\mathbf{b}}_{\mathrm{v}}}}\right),{\mathbf{W}}_{\mathrm{o}}^{\top}\,.$$

Commonly, H attention heads {Wi q,Wi k,Wiv,i,Wi o }
H
i=1 independently process X and concatenate their results into the final output (usually dk = dv = d/H). This introduces 2H GL symmetries. Everything also applies to general attention where, instead of X, independent data is fed as keys, queries, and values to Eq. 13. Example A.2 (Shallow linear net). Consider a two-layer linear net NN(W2,W1) = W2W1x with weight matrices W1 ∈ R

h×din,W2 ∈ R
dout×hand some input x ∈ R
din . This net has GL symmetry with correspondence G, H ← W2,W1
⊤ to Definition 3.1. With first-layer bias, we have

$$W_{2}(W_{1}x+b_{1})=W_{2}\left(W_{1}\quad b_{1}\right)\left(\begin{array}{l}{{x}}\\ {{1}}\end{array}\right)\,,$$
, (14)
corresponding to G, H ← W2,W1 b1
⊤.

Example A.3 (Low-rank adapters (LoRA, Hu et al. (2022))). Fine-tuning tasks with large language models add a trainable low-rank perturbation L ∈ R
d1×h, R ∈ R
d2×hto the pre-trained weight W ∈ R
d1×d2,

$$\operatorname{LoRA}(\mathbf{W})=\mathbf{W}+\mathbf{L}\mathbf{R}^{\top}\,,$$
LoRA(W) = W + LR⊤ , (15)

$$(14)$$
$$(15)^{\frac{1}{2}}$$

introducing a GL(h) symmetry where G, H ← L, R. Yen et al. (2024) propose an invariant way to train the parameters L, R and show that doing so improves the result obtained via LoRA.

## B. Concepts And Review For Riemannian Geometry

Recall that M is the total space: the manifold of parameters of our network. Also, on the total space we have a loss function ℓ : M → R. Useful resources are Lee (2003), Absil et al. (2008), and Boumal (2023).

## B.1. Orbit Of X

A symmetry relation naturally defines an equivalence relation: two points *x, y* ∈ M are equivalent under the symmetry, if they can be mapped onto each other by the action,

$$x\sim y\quad\Leftrightarrow\quad\exists g\in{\mathcal{G}}:y=\psi(g,x)\,.$$
x ∼ y ⇔ ∃g ∈ G : y = ψ(*g, x*). (16)
In other words, if we let orbit(x) :− {ψ(g, x) | g *∈ G}* be all points on the total space that are reachable from x through the action of G, all points in an orbit are equivalent. Instead of orbit(x), we will write

$$(16)^{\frac{1}{2}}$$
$$[x]:=\{y\in{\overline{{\mathcal{M}}}}\mid y\sim x\}$$
$$(17)$$
[x] := {y ∈ M | y ∼ x} (17)
for the symmetry-induced equivalence class [x] of x ∈ M. Let's further assume that ℓ is symmetric under G, i.e. for any x ∈ M and all g ∈ G, ℓ(x) = ℓ(ψ(*g, x*)).

## B.2. Quotient M **And Natural Projection**

If we take the quotient of the original parameter space manifold M, by the equivalence relation, ∼, induced by the symmetries of our neural architecture, we get a quotient M = M/ ∼. Under certain conditions, M is a quotient manifold. The mapping between a point in total space to its equivalence class is called the natural projection: Definition B.1. Let π : M → M/ ∼, be defined by x 7→ x. π is called the natural, or canonical projection. We use π(x) to denote x viewed as a point of M :− M/ ∼.

## B.3. Tangent Space, Vertical And Horizontal Spaces

Tangent vectors on the total space M, embedded in a vector space E can be viewed as tangent vectors to E, but the tangent space of the quotient manifold, M is not as straightforward. First, note that any element ¯ξ ∈ Tx¯M that satisfies Dπ(¯x)[¯ξ] = ξ (where D is the differential) is a candidate for a representation of ξ ∈ TxM. These aren't unique, and as we wish to work without any numerical ambiguity we introduce the notions of the vertical and horizontal spaces:
Definition B.2. For a quotient manifold M = M/ ∼, the vertical space at x¯ ∈ M is the subspace Vx¯ = Tx¯F = ker Dπ(x) where F = {y¯ ∈ M : ¯y ∼ x¯} is the fiber of x¯. The complement of Vx¯ is the horizontal space at x¯: Tx¯M = Vx¯ ⊕ Hx¯.

Definition B.3. There is only one element ¯ξx¯ that belongs to Hx¯ and satisfies Dπ(¯x)[¯ξx¯] = ξ. This unique vector is called the *horizontal lift* of of ξ at x¯. We denote the operator that affects the procedure by liftx¯(·) When the ambient space, E is a subset of R
n×p, the horizontal space can also be seen as such a subset, providing a convenient matrix representation of a priori abstract tangent vectors of M.

## B.4. Riemannian Manifold

We give our total space M a smooth inner product over its tangent vectors to give a Riemannian manifold.

Definition B.4. A Riemannian manifold is a pair (M, g), where M is a smooth manifold and g is a Riemannian metric, defined as the inner product on the tangent space TxM for each point x ∈ M, gx(·, ·) : TxM × TxM → R. We also use the notation ⟨·, ·⟩x to denote the inner product.

Note that this definition is not as arcane as it may appear since any smooth manifold admits a Riemannian metric, and we can consider the space of parameters of most neural architectures as constituting a smooth manifold, admitting at least a simple, Euclidean, metric.

The horizontal space can now be defined as the *orthogonal* complement of the vertical space: Hx¯ = (Vx¯)
⊥ = {u ∈ Tx¯M :
⟨*u, v*⟩x = 0 for all v ∈ Vx¯}. Additionally, letting g¯x¯ denote the metric on M, if for every x ∈ M and every ξx, ζx in TxM, g¯x¯(
¯ξx¯,
¯ζx¯) does not depend on x¯ ∈ π
−1(x) then, gx(ξx, ζx) = ¯gx¯(
¯ξx¯,
¯ζx¯) defines a valid metric on the quotient manifold M.

## B.5. Riemannian Gradient

Definition B.5. If ¯f is a smooth scalar field on a Riemannian manifold M, then the *gradient* of ¯f at x¯, grad ¯f(¯x) is the unique element of Tx¯M such that
⟨grad ¯f(¯x),
¯ξ⟩x¯ = D ¯f(¯x)[¯ξ], ∀¯ξ ∈ Tx¯M
If ¯f is a function on M, that induces a function f on a quotient manifold M of M, then we can express the horizontal lift of grad f at x¯ as liftx¯(grad f) = grad ¯f(¯x).

B.6. Geodesic Curves Definition B.6.

(a) Geodesic curves, γ¯, are the curves of minimal distance between two points on a manifold M. The distance along the geodesic is called the *geodesic distance*. If M is a Riemannian quotient manifold of M, with canonical projection π, and γ¯ is a geodesic on M, then γ = π ◦ γ¯ is a geodesic curve on M.

(b) Alternatively, geodesics, γ¯(t) = 0 can be defined as curves from a given point p ∈ M, (i.e., γ¯(0) = p), with initial velocity, γ¯˙(0) = ¯ξ ∈ Tp¯M, such that their *acceleration* is zero (a generalization of Euclidean straight lines). This characterization provides us with the following equation in local coordinates for the geodesic:

$$\frac{d^{2}\gamma^{\lambda}}{d t^{2}}+\Gamma_{\mu\nu}^{\lambda}\frac{d\gamma^{\mu}}{d t}\frac{d\gamma^{\nu}}{d t}=0$$
$\mathbf{I}J\left(x\boldsymbol{E}\right)$. 
where Γ
λ µν are the Christoffel symbols, Γ
λ µν =
1 2 g λσ ∂gσµ
∂xν +
∂gσν
∂xµ −
∂gµν
∂xσ
. Additionally, the geodesics can also be derived as the curves that are minima of the energy functional

$$S(\gamma)=\int_{a}^{b}g_{\gamma(t)}(\dot{\gamma(t)},\dot{\gamma(t)})d t$$

This second perspective will prove useful for the geodesics of the attention layers.

If the initial velocity tangent vector, ξ, is horizontal then, ∀t, γ¯˙(t) ∈ Hγ¯(t), that is, if the velocity vector starts out as horizontal, then it will stay horizontal. We call these geodesics, *horizontal geodesics*. The curve γ = π ◦ γ¯ is a geodesic of the quotient manifold M, with the same length as γ¯. This also holds the other way, i.e., a geodesic in the quotient manifold can be lifted to a horizontal geodesic in the total space.

## C. Geodesic Sharpness: Practical Concerns C.1. Transformers

Transformers, introduced by Vaswani et al. (2017), consist of multiheaded self-attention and feedforward layers, both wrapped in residual connections and layer normalizations. Visual transformers, in addition, tend to have convolutional layers. Mathematically, focusing for the moment on the multi-headed attention blocks,

MultiHead$(Q,K,V)=\left[\text{head}_{1},\ldots,\text{head}_{h}\right]W^{o}$  where $\text{head}_{i}=\text{Attention}\big{(}QW_{i}^{Q},KW_{i}^{K},VW_{i}^{V}\big{)}$

where Attention(*Q, K, V* ) = softmax QKT
√dk V . From this we can ascertain the following symmetries:
1) (W
Q
i, W K
i) → (W
Q
i G
−1, W K
i G
T) , ∀G ∈ GLn(dhead)
2) (WV
i
, Wo i
) → (WV
i G
−1, Wo i G
T) , ∀G ∈ GLn(dhead)
where Wo i are the columns of Wothat are relevant for the matrix multiplication with each WV
i
, taking into consideration the head concatenation procedure. In the full transformer model when solving for geodesic sharpness, for each layer, we apply Eq. 5 to each (W
Q
i
, W K
i) and
(WV
i
, Wo i
), using Eq. 11. This results in horizontal vectors (¯ξ Q
i
,¯ξ K
i
) and (¯ξ V
i
,¯ξ o i
). For the non-attention parameters, w,
(belonging to fully connected layers, convolutional layers and layer norm), we keep to the recipe of adaptive sharpness, so that ||¯ξw|| = || ¯ξw ⊙ |w|
−1||2. The norm of the full update vector, ¯ξ = concat(¯ξ Q i
,¯ξ K i
,¯ξ V i
,¯ξ o i
,¯ξw), where a sum over all parameters of the network is implicit, is ||¯ξ||2 =P||(¯ξ Q i
,¯ξ K i
)||2 + ||(¯ξ V i
,¯ξ o i
)||2 + ||¯ξw||2.

## C.2. Algorithm

Following the lead of Andriushchenko et al. (2023), we use Auto-PDG, proposed in Croce & Hein (2020), but now optimizing the horizontal vector ¯ξ instead of the input. In Algorithm 1, ℓ is the loss over the batch we are optimizing over, S is the feasible set of horizontal vectors, ¯ξ, with norm smaller than ρ, and PS is the projection onto this set. Γ are the Christoffel symbols for the parameters. η and W are fixed hyperparameters, which we keep as in Andriushchenko et al.

(2023), and the two conditions in Line 20 can be found in Croce & Hein (2020). The only differences to the algorithm employed to calculate adaptive sharpness are in lines 3, 8, 10, and 12. For the metric ⟨·, ·⟩mix the only differences are in the Christoffel symbols and in the Riemannian gradient (∇Gℓ → ∇GℓHTH−1)

## Algorithm 1 Auto-Pgd

1: **Input:** objective function ℓ, perturbation set S,¯ξ
(0), initial weights w(0), η, Niter, W = {w0*, . . . , w*n}
2: **Output:** ¯ξmax, ℓmax 3: v
(1) ← w(0) + ¯ξ
(0) −
1 2 Γ¯ξ
(0) ¯ξ
(0) ▷ Perturb weights according to Eq. 5 4: ¯ξ
(1) ← PS¯ξ
(0) + η∇ξ¯ℓ(v
(1))
5: ℓmax ← max{ℓ(w(0)), ℓ(v
(1))}
6: ¯ξmax ← ¯ξ
(0) if ℓmax ≡ ℓ(w(0)) **else** ¯ξmax ← ¯ξ
(1)
7: for k = 1 to Niter − 1 do 8: v
(k+1) ← w(0) + ¯ξ
(k) −
1 2 Γ¯ξ
(k) ¯ξ
(k) ▷ Perturb weights according to Eq. 5 9: if w(0) is an attention weight **then**
10: g ← ∇ξ¯ℓ(v
(k+1))w(0),T w(0) ▷ Make attention gradients Riemannian 11: **else**
12: g ← ∇ξ¯ℓ(v
(k+1)) ⊙ (w(0))
2 ▷ Make the other gradients Riemannian 13: **end if** 14: z
(k+1) ← PS
¯ξ
(k) + ηg)
15: ¯ξ
(k+1) ← PS¯ξ
(k) + α(z
(k+1) − ¯ξ
(k)) + (1 − α)(¯ξ
(k) − ¯ξ
(k−1))
16: if ℓ(v
(k+1)) > ℓmax **then**
17: ¯ξmax ← ¯ξ
(k+1) and ℓmax ← ℓ(v
(k+1))
18: **end if** 19: if k ∈ W **then** 20: if Condition 1 or Condition 2 **then**
21: η ← η/2 and w(k+1) ← wmax 22: **end if** 23: **end if** 24: **end for**

## C.3. Complexity

Geodesic sharpness is slightly more expensive than adaptive sharpness in the following sense: Our approach consists of three steps: 1) perturbing the weights according to Eq. 5, 2) optimizing the perturbations with gradient descent, and 3) projecting them onto the feasible set, i.e. horizontal vectors within the geodesic ball with a small enough norm. Steps 1) and 2) are also present in adaptive sharpness. Step 1) in our approach is slightly more expensive because we need to evaluate the quadratic form that involves the Christoffel symbols (given by Eq. 11 and Eq. 12); this step introduces nparams weight matrix multiplications, but these are quite efficient. Making the gradients Riemannian, costs another nparams weight matrix multiplications. Neither of these bottleneck our approach. For ⟨·, ·⟩inv, Step 3) requires solving a Sylvester equation to project the direction of the updated geodesic back onto the horizontal space. This solve is cubic in h (Kirrinnis, 2001), but h is usually small (e.g. h = 64 in the ImageNet and BERT experiments). For ⟨·, ·⟩mix, only efficient matrix multiplications are required.

On practical transformers, we expect the bottleneck to be the forward and backward propagations, just like in adaptive sharpness.

## D. Geodesic Sharpness: Scalar Toy Model

To make our approach explicit, we illustrate it on a NN with two scalar parameters G and H, square loss, and a single
(scalar) training point (*x, y*). We use ⟨·, ·⟩inv throughout. For this example, everything is analytically tractable. We also contrast our sharpness measure with previously proposed ones to highlight its invariance.

Since we require full column-rank, our parameter space is M = R∗ × R∗ with R∗ = R \ {0}.

Metric: At a point (G, H) ∈ M, for two tangent vectors η = (ηG, ηH), ν = (νG, νH) ∈ T(G,H)M, we have

$$\langle\eta,\nu\rangle^{\mathrm{inv}}=\frac{\eta_{G}\nu_{G}}{G^{2}}+\frac{\eta_{H}\nu_{H}}{H^{2}}=\eta^{\top}\underbrace{\left(\begin{array}{l l}{{\frac{1}{G^{2}}}}&{0}\\ {0}&{{\frac{1}{H^{2}}}}\end{array}\right)}_{g_{k l}}\nu$$
$$(18)$$

We denote the inverse metric by g
$$y\;g^{k l}=\begin{pmatrix}G^{2}&0\\ 0&H^{2}\end{pmatrix}$$

$$(G,H)=\{(\eta_{G},\eta_{H})\}$$

Horizontal space: H(G,H) = {(ηG, ηH) ∈ T(G,H)M | ηG

$$\begin{array}{l l l}{{\tilde{\ }(G,H){\mathcal{M}}}}&{{|}}&{{\frac{\eta_{G}}{G}=\frac{\eta_{H}}{H}\}}\end{array}$$

Geodesics: To compute the geodesics on the quotient space, we need the Christoffel symbols Γ
i km.

Using a coordinate system (p 1, p2) = (*G, H*), we have the following equation for a geodesic γ(t) = (γG(t), γH(t)), with initial conditions γ(0) = (G0, H0) and γ˙(0) = (ηG0
, ηH0
)

$$\frac{d^{2}\gamma_{G}}{d t^{2}}+\Gamma_{11}^{1}\left(\frac{d\gamma_{G}}{d t}\right)^{2}=0$$

and similarly for H with Γ
222 instead of Γ
111.

The Christoffel symbols can be found using the metric, g, and its inverse. Using the Einstein notation and denoting the inverse of g by the use of upper indices:

$$\Gamma^{i}{}_{k l}=\frac{1}{2}g^{i m}\left(\frac{\partial g_{m k}}{\partial x^{l}}+\frac{\partial g_{m l}}{\partial x^{k}}-\frac{\partial g_{k l}}{\partial x^{m}}\right)$$

Then

 $ \Gamma^1{}_{11}=\frac{1}{2}g^{1m}\left(\frac{\partial g_{m1}}{\partial p^1}+\frac{\partial g_{m1}}{\partial p^1}-\frac{\partial g_{kl}}{\partial p^m}\right)=-\frac{1}{G}$  $ \Gamma^2{}_{22}=-\frac{1}{H}$
H
All other Christoffel symbols are 0. Our geodesic equations then become (we omit the derivation for H, which is identical but with G ↔ H)

$$\frac{d^{2}\gamma_{G}}{d t^{2}}-\frac{1}{\gamma_{G}}\left(\frac{d\gamma_{G}}{d t}\right)^{2}=0$$

This ODE has the (unique) solution γG(t) = AG exp(bGt). Taking into account the initial conditions, AG = G0, AH = H0 and due to the definition of the horizontal space, bG =
ηG G0 and bH =
ηH H0
, this becomes

$$\gamma(t)=\left(G_{0}\exp\!\left({\frac{\eta_{G}}{G_{0}}}t\right),H_{0}\exp\!\left({\frac{\eta_{H}}{H_{0}}}t\right)\right)$$

One important detail to note is that these geodesics are not complete, that is, not all two points can be connected by a geodesic. Points with different signs cannot be connected, which makes sense since we excluded the origin from the acceptable parameters and in 1D we need to cross it to connect points with differing signs. All points that lie in the same quadrant as (G0, H0) can be connected through a geodesic.

Putting it all together

$$S_{\max}^{o}((G_{0},H_{0}))=\left[\max_{|b|\leq\rho}x^{2}G_{0}^{2}H_{0}^{2}(\exp(4b)-1)-2yxG_{0}H_{0}(\exp(2b)-1)\right],\tag{19}$$  which is 
Letting y0 = G0H0x, this becomes:

$$S_{\mathrm{max}}^{\rho}((G_{0},H_{0}))=\left[\operatorname*{max}_{||b||\leq\rho}y_{0}^{2}(\exp(4b)-1)-2y y_{0}(\exp(2b)-1)\right],$$

Since ηH is completely determined by ηG we can ignore the maximization over it.

Since in practice we'll take ρ ≪ 1, we Taylor expand to get

$$(20)$$
$$S_{\mathrm{max}}^{\rho}\approx4\rho|y_{0}||y-y_{0}|$$

This presents an issue when the residual, |y − y0|, is zero, so we can also expand to second order, to get, when |y − y0| ≈ 0

$$S_{\mathrm{max}}^{\rho}\propto\rho^{2}|y_{0}||y-2y_{0}|=2\rho^{2}y_{0}^{2}$$

This is, up to constants, just ||G ⊙ H||22
. This is also invariant to GL1 transformations, as expected.

Very close to the minimum we only capture (second-order in ρ) properties of the network, a bit further away from it we capture a (first-order in ρ) mix of data and network properties. Comparison with more traditional measures: The local average and worst case Euclidean sharpness (at a minimum) are

$$\begin{array}{l}{{S_{\mathrm{avg}}=\mathrm{Tr}\,\nabla^{2}L_{S}=G^{2}+H^{2}}}\\ {{S_{\mathrm{max}}=\lambda_{\mathrm{max}}(\nabla^{2}L_{S})=G^{2}+H^{2}}}\end{array}$$

Adaptive sharpness is defined as

$$\begin{array}{l}{{S_{\mathrm{avg}}^{\rho}(w,c)=\mathbb{E}_{S\sim\mathbb{P}_{m}}\left[L_{S}(w+\delta)-L_{S}(w)\right],\quad\delta\sim\mathcal{N}(0,\rho^{2}\mathrm{diag}(c^{2}))}}\\ {{S_{\mathrm{max}}^{\rho}(w,c)=\mathbb{E}_{S\sim\mathbb{P}_{m}}\left[\operatorname*{max}_{\|\delta\odot c^{-1}\|_{p}\leq\rho}L_{S}(w+\delta)-L_{S}(w)\right],}}\end{array}$$

By picking c very carefully one can get

$$S_{\mathrm{avg}}^{p}(w,c)=|G H|$$

By contrast, in our approach there is no need for careful hyperparameter choices Geodesic flatness with more data points: How does the geodesic flatness look like with more data points?

$$L_{S}(G,H)={\frac{1}{n}}\sum_{i=1}^{n}(G H x_{i}-y_{i})^{2}$$
$$(21)$$

which leads to (defining y 0 i = GHxi):

which leads to extending $g_{1}=0$ or $\sqrt{10}$,  $$S^{\prime}_{\rm max}=\max_{b}\frac{1}{n}\sum_{i=1}^{n}\left[(y_{i}^{0})^{2}\left(\exp\biggl{(}\frac{b}{|b|}2\sqrt{2}\rho\biggr{)}-1\right)-2yy_{i}^{0}\left(\exp\biggl{(}\frac{b}{|b|}\sqrt{2}\rho\biggr{)}-1\right)\right]$$  Taylor expanding (in $\rho$) once more, we see that 
 (21)
$$S^{\rho}_{\rm max}\approx\max_{b}\frac{1}{n}\sum_{i=1}^{n}\left[2\sqrt{2}\rho\frac{b}{|b|}y_{i}^{0}(y_{i}^{0}-y)+2\rho^{2}(y_{i}^{0})^{2}\right]\tag{22}$$
Which b maximizes Eq. 22, depends on the sign of Pn i=1 -y 0 i
(y 0 i − y): b < 0 if the sum is negative, the reverse if the opposite is true.

## D.1. Traditional Flatness

In Figure 8 we extend Figure 1 to include the trace of the Hessian, both Euclidean and Riemannian. The trace of the network Hessian is a quantity that can be used to quantify flatness. We plot, for the scalar toy model, the level sets of: a) the loss function; b) the Euclidean and Riemannian gradient; c) the traces of the Euclidean and Riemannian network Hessian. Several features of the plots are important to note: a) the Riemannian version of the gradient and Hessian have the same level set geometry as the loss function; b) both the Riemannian gradient norm and the trace of the Riemannian Hessian have smaller values throughout than their Euclidean equivalents; c) the trace of the Riemannian Hessian actually reaches 0 when at the local minimum, whereas the Euclidean Hessian actually attains its highest value there; d) the Euclidean trace of the Hessian cannot distinguish between a minimum and a maximum whereas the Riemannian trace can actually do so. Even for simple flatness measures, correcting for the quotient geometry can provide a much clearer picture.

## E. Geodesic Sharpness: Diagonal Networks In Full Generality E.1. Metric (6)

Metric: At a point (u, v) ∈ M, for two tangent vectors η = (ηu, ηv), ν = (νu, νv) ∈ T(u,v)M, we have

$$\langle\eta,\nu\rangle^{\rm inv}=\sum_{i=1}^{d}\frac{\eta_{\mathbf{u}}^{i}\nu_{\mathbf{u}}^{i}}{(\mathbf{u}^{i})^{2}}+\frac{\eta_{\mathbf{v}}^{i}\nu_{\mathbf{v}}^{i}}{(\mathbf{v}^{i})^{2}}\tag{10}$$
$$(23)$$
$$(24)$$
Horizontal space: H(u,v) = {(ηu, ηv) ∈ T(u,v)M | η
$$\forall i\in\{1,\ldots,d\}$$
$$T_{(\mathbf{u},\mathbf{v})}{\mathcal{M}}\;\mid\;{\frac{\eta_{\mathbf{u}}^{i}}{\mathbf{u}^{i}}}={\frac{\eta_{\mathbf{v}}^{i}}{\mathbf{v}^{i}}}$$
Geodesics: We define b
i =
η
i
u
ui =
$\mathbf{v}=\frac{\eta_{\mathbf{v}}^{i}}{\mathbf{v}^{i}}\forall i\in\{1,\ldots,d\}$, so that 
$\mathbf{\gamma}(t)^{i}=(\mathbf{u}(t),\mathbf{v}(t))=\left(\mathbf{u}_{0}^{i}\exp(\mathbf{b}_{i}t),\mathbf{v}_{0}^{i}\exp(\mathbf{b}_{i}t)\right)\forall i\in\{1,\ldots,d\}$
where u i0 and v i0 are the initial positions for our parameters, i.e., the parameters that the network actually learned.

Geodesic sharpness: We assume that in what follows XT X = Idd, and we denote β0 = u0 ⊙ v0, γt = 
exp2b 1t*, . . .* exp2Bdt , βt = (ut ⊙ vt) = β0 ⊙ γt, β∗ = XT y. Note that β∗ is just the optimal least squares predictor when XT X = Id. With this notation

$$S_{\max}=\max_{||\mathbf{b}||\leq\rho}\sum_{i}^{d}\left[(\mathbf{\beta}_{0}^{i})^{2}(\mathbf{\gamma}_{t}\odot\mathbf{\gamma}_{t}-1)\right]-2(\mathbf{\beta}_{0}\odot\mathbf{\gamma}_{t}-1)^{T}\mathbf{\beta}_{*}\tag{1}$$
$$(25)$$

Euclidean
(b) Gradient norm Riemannian
(d) Gradient norm
(a) Loss
−1.0 −0.5 0.0 0.5 1.0 θ1
−1.00 −0.75 −0.50
−0.25 0.00 0.25 0.50 0.75 1.00 0.0 0.2 0.4 0.6 0.8 1.0 θ2
(c) Hessian trace

$$(26)$$

(e) Hessian trace
At a first glance, this expression does not seem to have a simple interpretation, but we Taylor expand it to second order in B (since ρ is supposed to be small):
$$S_{\operatorname*{max}}\approx\operatorname*{max}_{||b||\leq\rho}4b^{T}r+4b^{T}D_{\beta_{0},\beta_{*}}b$$
T Dβ0,β∗ b (26)
where r = {β i 0(β i 0 − β i∗), i = 1*, . . . , d*}, r
′ = {(β i 0 − β i∗), i = 1*, . . . , d*} and Dβ0,β∗ = *diag*(β i 0(2β i 0 − β i∗)) =
diag(β i 0(β i 0 + (r
′)
i)). We separate the analysis of Eq.26 into three cases:

$\mathbf{a}\cdot\mathbf{m}\mathbf{a}\times\mathbf{m}$
case a): r ̸= 0 **and first order suffices** Eq.26 becomes

$${\mathrm{becomes}}$$
$$S_{\operatorname*{max}}=\operatorname*{max}_{||\mathbf{b}||\leq\rho}4\mathbf{b}^{T}\mathbf{r}$$

with solution Smax = 4ρ||r||. This is essentially the gradient norm– a useful quantity for understanding generalization (Zhao et al., 2022b). case b): r = 0 Here we necessarily have to consider the second order terms, so that Eq.26 becomes

$$S_{\operatorname*{max}}=\operatorname*{max}_{||{\boldsymbol{b}}||\leq\rho}4{\boldsymbol{b}}^{T}{\boldsymbol{D}}_{\beta_{0},\beta_{*}}{\boldsymbol{b}}$$

This has the well known solution of Smax = ρ 2λmax(Dβ0,β∗
) = ρ 2 max((β i0
)
2). This is just ||β||2∞, which is the square of what we would get by using adaptive sharpness, Eq.39, with a very carefully chosen hyper-parameter c. This is a quantity that is useful when our ground-truth, β
∗is dense.

case c): r ̸= 0 **and we need both first and second order terms** In this case, Eq.26 needs to be considered in full, and we solve the maximization problem using Lagrange multipliers. The Lagrangian will be

$${\mathcal{L}}=-4\mathbf{b}^{T}\mathbf{r}-4\mathbf{b}^{T}\mathbf{D}_{\beta_{0},\beta_{*}}\mathbf{b}+\lambda(\mathbf{b}^{T}\mathbf{b}-\rho^{2})$$

The KKT conditions then are

∂L ∂b = −4r − 8Dβ0,β∗ b + 2λb = 0 (27) λ(b T B − ρ 2) = 0 (28) λ ≥ 0 (29)
If the constraint is not active, then λ = 0 and

$$(27)$$
$$\begin{array}{c}{{(28)}}\\ {{(29)}}\end{array}$$
$$b_{*}=-\frac{1}{2}D_{\beta_{0},\beta_{*}}^{-1}r$$

In practice, unless ρ is large the constraint will always be active, in which case 27 becomes

$$\begin{array}{l}{{-4\mathbf{r}-8\mathbf{D}_{\beta_{0},\beta_{*}}\mathbf{b}+2\lambda(\mathbf{B})=0}}\\ {{(\mathbf{b}^{T}\mathbf{b}-\rho^{2})=0}}\\ {{\lambda\geq0}}\end{array}$$

this then becomes

$$\begin{array}{l}{{B_{\ast}=2\left(\lambda I-4D_{\beta_{0},\beta_{\ast}}\right)^{-1}r}}\\ {{4\sum_{i}^{d}\frac{(r^{i})^{2}}{\left(\lambda-4(\beta_{0}^{i}(\beta_{0}^{i}+r^{\prime})\right)^{2}}=\rho^{2}}}\\ {{\lambda\geq0}}\end{array}$$

$$(30)^{\frac{1}{2}}$$

## E.2. Metric (7)

We follow the same approach as in the previous section. The main difference will be in the form of the geodesics:
u(t) ⊙ v(t) = (u0 ⊙ v0) ⊙ (1 + 2bt), where b i =
η i u ui =
η i v vi, as in the previous section. This essentially treats the two-layer neural network as if it were a single layer, with predictor β = u ⊙ v, that it then perturbs linearly to determine sharpness.

For ⟨·, ·⟩mix, and denoting by Dβ = *diag*(β i 0):

$$S_{m a x}=\operatorname*{max}_{||\eta||^{m a}\leq\rho}4\left[{\boldsymbol{b}}^{T}({\boldsymbol{\beta}}_{0}-{\boldsymbol{\beta}}_{*})+{\boldsymbol{b}}^{T}{\boldsymbol{D}}_{\boldsymbol{\beta}}^{2}{\boldsymbol{b}}\right]$$
T D2βb(30)
We also have that

$$(||\eta||^{\min})^{2}=\left[\ldots+(\mathbf{v}^{i})^{2}(\eta^{i}_{\mathbf{u}})^{2}+(\mathbf{u}^{i})^{2}(\eta^{i}_{\mathbf{v}})^{2}+\ldots\right]$$ $$=\left[\ldots+(\mathbf{v}^{i})^{2}(\mathbf{u}^{i})^{2}\left(\frac{(\eta^{i}_{\mathbf{u}})^{2}}{(\mathbf{u}^{i})^{2}}+\frac{(\eta^{i}_{\mathbf{v}})^{2}}{(\mathbf{v}^{i})^{2}}\right)+\ldots\right]$$ $$=\left[\ldots+2(\mathbf{v}^{i})^{2}(\mathbf{u}^{i})^{2}(\mathbf{b}^{i})^{2}+\ldots\right]=||2\mathbf{D}_{\beta_{0}}\mathbf{b}||_{2}$$

Substituting 2Dβ0 b = δ, Eq. 30 becomes

$$S_{m a x}=\operatorname*{max}_{||\delta||\leq\rho}\left[\delta^{T}(\beta_{0}-\beta_{*})+\delta^{T}\delta\right]$$
Tδ(34)
with the solution (up to constants)

$$S_{m a x}=\rho||\mathbf{\beta}_{0}-\mathbf{\beta}_{*}||_{2}$$
Smax = ρ||β0 − β∗||2 (35)
$$(31)$$
$$(32)$$
$$(33)$$
$$(34)$$
$$(35)$$

19

## F. Geodesic Sharpness: Gl1 **Symmetry And Adaptive Sharpness**

What happens if instead of a general GLn symmetry, we factor out a GL1 re-scaling symmetry? That is, we identify, element-wise, (¯x, y¯) ∼ (¯x
′y¯
′) if ∃α ∈ R∗ = R \ {0} s.t. x¯ = αx¯
′and y¯ = α
−1y¯.

This is the symmetry present in diagonal networks, and so we utilize the metric given by Eq. 6, reproduced below for convenience of the reader:

$$g\left[\left(\eta_{\mathbf{u}},\eta_{\mathbf{v}}\right),\left(\nu_{\mathbf{u}},\nu_{\mathbf{v}}\right)\right]=\sum_{i=1}^{d}\frac{\eta_{\mathbf{u}}^{i}\nu^{i}_{\mathbf{u}}}{(\mathbf{u}^{i})^{2}}+\frac{\eta_{\mathbf{v}}^{i}\nu^{i}_{\mathbf{v}}}{(\mathbf{v}^{i})^{2}}\tag{1}$$
$$(36)$$
$$(37)$$
$$(38)$$
$$(39)$$

Note that this metric is equivalent to the following metric:

$$g\left[\left(\eta_{\mathbf{u}},\eta_{\mathbf{v}}\right),\left(\nu_{\mathbf{u}},\nu_{\mathbf{v}}\right)\right]=g\left[\left(\eta_{\mathbf{u}}/|\mathbf{u}|,\eta_{\mathbf{v}}/|\mathbf{v}|\right),\left(\nu_{\mathbf{u}}/|\mathbf{u}|,\nu_{\mathbf{v}}/|\mathbf{v}|\right)\right]_{\mathrm{euc}}$$
g [(ηu, ηv),(νu, νv)] = g [(ηu/|u|, ηv/|v|),(νu/|u|, νv/|v|)]euc (37)
where geuc is the usual Euclidean metric and the division is taken to be element-wise. Denoting the concatenation of all tangent vectors by ξ, the concatenation of all parameters by w, we have ||ξ|| = ||ξ/|w|||2.

In this situation Eq. 4 becomes (γ denotes our geodesics as usual)

$$S_{\mathrm{max}}^{\rho}(w,c)=\mathbb{E}_{\xi\sim\mathbb{D}}\left[\operatorname*{max}_{||\xi/\|w\|_{2}\leq\rho}L_{S}(\bar{\gamma}_{\xi}(1))-L_{S}(\bar{\gamma}_{\xi}(0))\right],$$

If we then ignore the corrections induced by the geometry of the metric on the geodesics, i.e., take γ¯ξ¯(1) = ¯γξ¯(0)+¯ξ = w+¯ξ, then we get

$$S_{\operatorname*{max}}^{\rho}(w,c)=\mathbb{E}_{S\sim\mathbb{D}}\left[\operatorname*{max}_{||\xi/|w||_{2}\leq\rho}L_{S}(\mathbf{w}+\xi)-L_{S}(\mathbf{w})\right]$$

(39)
which is exactly the formula for adaptive sharpness.

## G. Geodesic Sharpness: Ablations

In this appendix we conduct ablation studies on geodesic sharpness (Equation (4)). There are two main components to our recipe that differ from adaptive sharpness: a) the norm ||¯ξ||; b) the weight update formula, which instead of the usual wi = wi + ¯ξ takes into account the curvature induced by the parameter space symmetries wi = wi + ¯ξ i −
1 2 Γ
i kl
¯ξ k ¯ξ l.

Below we turn off these components one by one and re-compute the resulting sharpness on MNLI using the BERT models described in Section 5.3.2. Metric (9): In Figure 9 we show the results for our ablation studies using metric (9). The norm component is much more impactful than the second-order weight corrections. Turning off the second-order weight corrections results in a small performance drop only. Metric (10): In Figure 10 we show the results for our ablation studies using metric (10). The norm component is still much more impactful than the second-order weight corrections for this metric, but now the second-order weight corrections are essential, and without them sharpness loses a considerable amount of predictive power.

## H. Geodesic Sharpness: Ranks And Relaxation H.1. Ranks: How Natural Is Assumption 5.1?

In general, in non-linear networks there is a tendency towards low-rank representations, which might make Assumption 5.1 seem excessive and counter to realistic situations. However, while the learned WQWTK tend to be low-rank, WQ and WK
(on which Assumption 5.1 ought to apply) themselves are usually high/full (column) rank (Yu & Wu, 2023).