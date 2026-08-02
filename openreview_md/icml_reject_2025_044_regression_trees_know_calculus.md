# Regression Trees Know Calculus

## Abstract

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054

## 1. Introduction

Tree-based methods, such as regression trees, are a workhorse of the contemporary data scientist. Their ease of use, computational efficiency and predictive capability without the need for extensive feature engineering makes them popular with practitioners. The most widely used version of regression trees approximate with greedily constructed, piecewise-constant functions than can handle data which exhibit discontinuities or divergent behavior in various parts of the feature-space. Perhaps because of their capability to tackle pathological problems, it seems that some of their 1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

1 Anonymous Authors1 Regression trees have emerged as a preeminent tool for solving real-world regression problems due to their ability to deal with nonlinearities, interaction effects and sharp discontinuities. In this article, we rather study regression trees applied to well-behaved, differentiable functions, and determine the relationship between node parameters and the local gradient of the function being approximated. We find a simple estimate of the gradient which can be efficiently computed using quantities exposed by popular tree learning libraries. This allows tools developed in the context of differentiable algorithms, like neural nets and Gaussian processes, to be deployed to tree-based models. To demonstrate this, we study measures of model sensitivity defined in terms of integrals of gradients and demonstrate how to compute them for regression trees using the proposed gradient estimates. Quantitative and qualitative numerical experiments reveal the capability of gradients estimated by regression trees to improve predictive analysis, solve tasks in uncertainty quantification, and provide interpretation of model behavior.

0.00 0.25 0.50 0.75 1.00 0.0 0.2 0.4 0.6 0.8 1.0Function 0.00 0.25 0.50 0.75 1.00 0.0 0.2 0.4 0.6 0.8 1.0Depth 3 estimate 0.00 0.25 0.50 0.75 1.00 0.0 0.2 0.4 0.6 0.8 1.0Depth 10 estimate 0.00 0.25 0.50 0.75 1.00 0.0 0.2 0.4 0.6 0.8 1.0Depth 15 estimate 0.00 0.25 0.50 0.75 1.00 0.0 0.2 0.4 0.6 0.8 1.0Depth 6 estimate 0.00 0.25 0.50 0.75 1.00 0.0 0.2 0.4 0.6 0.8 1.0True Vector Field
Figure 1. **Illustration of Gradient Estimates**. The top left gives a target function, and the bottom right gives its vector field. Shown in between are estimates of the gradient extracted from a regression tree fit to data from the function converging to the true vector field.

properties in approximating well-behaved functions may have gone unnoticed.

In this article, we will study the approximation of a continuously differentiable function f on the unit cube in dimension P with a piecewise constant regression tree. In particular, we will investigate a means of approximating ∇f using only information contained in the tree structure computable in a single pass through the tree. We find a simple and easily computable quantity analogous to a finite difference and, via the tree's structure, can use it to efficiently form estimates of integro-differential quantities. Previously, gradient estimation in regression trees has been studied in the context where the leaves have differentiable models, and the gradients of these models are used to estimate the gradient (e.g. Chaudhuri et al. (1995); Loh (2011)). However, the constant-leaf tree remains prevalent in practice, and the purpose of this article is rather to examine the *implicit* gradient estimation that occurs within these constant-leaf trees where, formally, the gradient of the tree is almost-everywhere zero. With a gradient estimator in hand, we unlock for tree-based models the stable of existing gradient-based methods for variable interpretation and dimension reduction developed in other areas. Among many possibilities, we will study in particular the Active Subspace Method (Constantine, 2015),
a global dimension reduction technique from the Uncer-

Depth = 1 Depth = 2 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 2
∇f(x) ≈
∆1
∆2
=
1 0.8−0.2 1−0 0.6−1.0 1−0
=
0.6
−0.4 x x
tainty Quantification literature, and the Integrated Gradient Method (Sundararajan et al., 2017), a local model interpretation technique from the neural network literature.

055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 1. We develop a simple algorithm to extract gradient and integrated gradient information from regression trees.

2. We show how to use this to port gradient-based interpretability techniques from other fields to benefit interpretability of regression trees.

3. We find that in certain circumstances gradient-enabled regression trees can produce better estimates of active subspaces than existing UQ methods.

We begin in Section 2 by discussing pertinent background on regression trees and gradient-based interpretability. Our main methodological contributions are given in Section 3 and developed theoretically in Section 4. Subsequently, we illustrate these procedure on numerical examples in Section 5 before offering conclusions and future research directions in Section 6.

## 2. Background

While small decision trees are easily interpretable, they become significantly less so as they grow in size or if they are aggregated via Random Forests (Breiman, 2001; 2002) or gradient boosting (Friedman, 2001). Consequently, developing methods for increasing the interpretability of trees is of great importance (Louppe, 2014).

## 2.1. **Variable Interpretation And Selection In Regression** Trees

Several methods for variable importance were suggested alongside some of the original tree-based methods (Breiman, 2017; 2001; 2002), the two most salient for our study being the Mean Decrease in Impurity (called MDI or TreeWeight (Kazemitabar et al., 2017)) and the Feature Permutation method. MDI assigns importance to variables in proportion to the mean reduction in cost when splitting along a given variable, an intuitively reasonable approach. However, MDI as originally proposed was sensitive to properties of the feature variables as well as to the depth within the tree that a split occurred, leading to "debiased" variants (Sandri & Zuccolotto, 2008; Li et al., 2019). Kazemitabar et al. (2017); Klusowski & Tian (2020) studied theoretical properties of a simplified version of this metric using only the first split
("stumps"). On the other hand Feature Permutation is based on measuring decrease in performance when shuffling a Active Subspaces provide for linear dimension reduction, which is already commonly used in the setting of regressions trees, such as when using random projection or PCA for rotated trees (Breiman, 2001; Rodriguez et al., 2006). In contrast to these methods, active subspaces consists of supervised linear dimension reduction which takes the relationship between features and response into account. Already, supervised linear dimension reduction has been proposed for use with tree based methods where, e.g. a kernel method is used to perform the dimension reduction which is then applied to a tree-based method (Shan et al., 2015). But here, we show how to actually use the tree itself to perform a linear sensitivity analysis, rather than relying on a helper model to do this. This is essential if the analyst is interested in model-interpretation (as contrasted with *data*-interpretation (Chen et al., 2020)), and to the best of our knowledge the application of the Active Subspace method, enabled by our novel gradient estimates, is the first such linear sensitivity metric for trees. And while we've so far discussed what active subspaces can do for regression trees, we don't think this new relationship will be one-sided. Our numerical experiments show that regression trees can serve as scalable estimators of the active subspace, favorable to existing methods in certain circumstances. We hope this can highlight the potential for regression trees in the gradient-based UQ space. We view the major contributions of our article as follows:
110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 given feature, and though the method does have some desirable properties (Ishwaran, 2007), it has come under criticism for undesirable behavior in the context of correlated predictors (Hooker et al., 2021). Empirical studies have found issues in both of these methods (Strobl et al., 2007; 2008). Against this backdrop of negative results on the behavior of these initially suggested methods came contributions in general-purpose machine learning interpretability methods. SHAP values (Lundberg & Lee, 2017) have been proposed as a general-purpose tool for understanding machine learning models. However, they would prove especially popular for understanding tree-based methods where efficient algorithms have been proposed for estimating these otherwise combinatorially difficult quantities (Lundberg et al., 2019; Karczmarz et al., 2022).

## 2.2. Gradient-Based Model Interpretation

The gradient of a function tells us how its output is related to its input over short distances, and is a natural candidate to help explain the behavior of a model. Hechtlinger (2016)
suggested to simply look at the gradient of a model evaluated at a particular observation to gain local understanding. Another approach is the Integrated Gradient (Sundararajan et al., 2017) introduced in the context of neural networks, which involves involves privileging some setting of input features which is called the *reference point*, which we'll denote x
∗. Subsequently, in order to explain a prediction at a given point x, we integrate the gradient along the path from x
∗to x and multiply elementwise against that difference; that is, denoting the output of the neural network as f:

$$I G(\mathbf{x}):=(\mathbf{x}-\mathbf{x}^{*})\odot\int_{\alpha=0}^{1}\nabla f(\alpha\mathbf{x}+(1-\alpha)\mathbf{x}^{*})d\alpha\,.\,\,(1)$$
$\square$
This theoretically appealing approach has also seen practical success in applications including medicine (Sayres et al., 2019) and chemistry (McCloskey et al., 2019). One could also integrate the gradient over a larger part of the space to perform a more global sensitivity analysis. This is the idea behind the Active Subspace Method (Constantine, 2015). In our context, this involves defining the Active Subspace Matrix as:
where µ is a measure. Eigenanalysis on Cfµreveals linear combinations of features which are important, similar to PCA. Various techniques have been developed to estimate active subspaces in the presence only of input-output data. A Polynomial Ridge Approximation (PRA) procedure (Hokanson & Constantine, 2018) is available, as is a closed form estimate using Gaussian processes (GP; Wycoff et al. (2021)), and also an approach based on neural net-

$$\mathbf{C}_{\mu}^{f}:=\int_{[0,1]^{P}}\nabla f(\mathbf{x})\nabla f(\mathbf{x})^{\top}d\mu(\mathbf{x})\,,\qquad(2)$$

Depth = 1 Depth = 2 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0i = 1 i = 2 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 i = 3 i = 6 i = 5 x x i = 4
works called the Deep Active Subspace Method (Tripathy & Bilionis, 2019; Edeling, 2023, DASM) has been proposed.

## 3. Integro-Differentiation Of Regression Trees

In this section we propose an easy to compute estimator of derivatives and integrals of derivatives for regression trees. While the underlying idea in this article is simple, discussing it rigorously requires a fair amount of notation; Figure 3 illustrates our notation on simple regression tree. Say we fit a regression tree of depth K to data (xn, yn = f(xn) + ϵn)
with xn ∈ [0, 1]P , yn ∈ R for n ∈ {1*, . . . , N*}, and ϵn mean zero with finite variance. Purely for simplicity, we will assume that the tree has leaf nodes only at depth K, i.e. there is no path through the tree that ends before depth K. Associate with every node from all depths an integer index i such that the root node is labeled 0 and if node i is deeper in the tree than node j, then *i > j*. That is, the indices at depth 1 are given by {1, 2}, at depth 2 by {3, 4, 5, 6}, and generally at depth k the indices are given by Nk = {2 k−1 + 1*, . . . ,* 2 k}. We denote the variable along which the i th node splits by σi ∈ {1*, . . . , P*} and the threshold at which that split occurs as τi.

## 3.1. Differentiation

The difference between the mean response µ l ion the left side of node i's split (containing points in that node where xσi ≤ τi) and the mean response µ r ion the right side (where xσi > τi) contains information about the σ th icomponent of the gradient in that area. By dividing this difference proportionally to the size of the node along dimension σi, we form a quantity similar to a finite difference. Let [l ip
, uip
]
denote the extent of node i along dimension p for p ∈
{1*, . . . , P*}. Then define the following quantity:
We have used the notation l i = [l i1
, . . . , liP
] to denote the vector of lower bounds of the extent of node i and similar for u i, and by [l i, u i] we denote the set of points lying within the bounds of the i th node. The ratio ∂f˜
∂x˜ σi
i contains information about the partial derivative of f with respect to xσi(which is, again, the variable that node i splits along) inside of [l i, u i]. The idea that ∂f˜
∂x˜ σi
i
≈
∂f(x)
∂xσi
i for any x in node i only makes sense if the gradient does not vary much within it. We expect this to be the case only for very small nodes, or in other words, very deep trees, estimating functions with gradients which vary smoothly; we make this precise in the next section. At a particular node, we can only form estimates of a single partial derivative. However, if our tree is sufficiently deep, we can combine estimates of partial derivatives from nodes at multiple depths to form an estimate of the entire gradient.

Recall that Nk = {2 k−1 + 1*, . . . ,* 2 k} contains the index of all nodes of depth k and let ρi denote the index of the parent of node i (which will be a member of Nk−1). Then Algorithm 1 shows how to aggregate the partial derivative estimates into a gradient estimate. Algorithm 1 Tree-Based Gradient Estimation

Gi ← 0 ∈ R
P for all i.
for k ∈ {1*, . . . , K*} do
for i ∈ Nk do
Gi ← Gρi {Get parent's estimate}
Gi[σi] ←2(µ
r
i −µ
l
i)
ui,σi−l*i,σi*
{Update along split direction}
end for
end for 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 We thus have associated each node i with an estimator of the gradient, though if we have not split along a variable p in the tree upstream of i, the estimate of that component will simply be 0. This is actually somewhat reasonable, as the tree will split along variables with small gradient components less frequently. Of course, nothing would stop us from comparing values of means that are adjacent in the input domain but not adjacent in the tree structure (for example, nodes 3 and 6 of Figure 3). However, the advantage of considering only nodes adjacent in the tree is that this Tree-Based Gradient Estimator (TBGE) can be computed efficiently merely by traversing the tree. Furthermore, if we needed only an estimate of a gradient at a particular point x, we would need only to traverse the regression tree in the standard manner, visiting only nodes upstream of the leaf nod x. To be explicit, let B(x) be the index of the node of the greatest depth which contains x. We use as our estimate of ∇f(x) the TBGE
associated that node; that is, ∇˜ f(x) = GB(x).

We can see that once each feature has been iterated over once, Gi will be in some sense an estimator of the gradient within node i. Furthermore, in the large data limit, the analysis of Appendix A.2 reveals that this will occur within (P − 1) log2 maxp |
∂f(x)
∂xp| minp |
∂f(x)
∂xp | splits when approximating linear functions. This suggests that depth should grow linearly with dimension, or least with the number of nonzero coefficients in the gradient.

## 3.2. Integration

We next develop estimators for quantities of the form:

$${\mathcal{I}}(f)=\int h(\nabla f({\bf x}))d\mu({\bf x})\,.\tag{4}$$

Here, h is some integrable R
P → H function and µ is a measure on RP . The Integrated Gradient at point z with reference point z
∗ may be written in this form by choosing h(a) = (z − z
∗) ⊙ a, H = RP , and µ as the degenerate uniform measure on the line segment between z and z
∗.

Similarly, Active Subspace can also be seen to belong to this class by setting h(a) = aa⊤, H = R
P ×P and arbitrary µ.

We will consider two classes of estimators for such quantities. The first is a Monte Carlo Estimator (MCE) which is appropriate whenever µ is a probability measure which it is possible to sample from. The second is a Partition-Based Estimator (PBE) which is suitable whenever we can compute µ([a, b]) for arbitrary a, b. Start with the MCE. We fix some Monte Carlo sample size M and form the standard Monte Carlo approximation, then plug in the TBGE where the gradient appears:

$${\mathcal{I}}(f)\approx{\frac{1}{M}}\sum_{{\bf x}_{m}\sim\mu}h(\nabla f({\bf x}_{m}))\approx{\frac{1}{M}}\sum_{{\bf x}_{m}\sim\mu}h({\bf G}^{{\bf x}_{m}})\,.\,\,\,(5)$$

We denote this estimator by IˆMC (f). In particular, we can form a Tree-Based Integrated Gradient (TBIG) as follows by sampling random uniform variables um on [0, 1]:

$$\hat{G}({\bf x})=({\bf x}-{\bf x}^{*})\odot\frac{1}{M}\sum_{m=1}^{M}{\bf G}^{B(u_{m}{\bf x}+(1-u_{m}){\bf x}^{*})}.\tag{6}$$

Next, we consider the PBE. This involves simply computing a weighted average of the action of h on the TBGE on each node at the penultimate depth K − 1, weighted by the measure of that node. For simplicity, we assume that µ([0, 1]P ) = 1. In notation, we define the estimator as

$$\hat{\mathcal{I}}_{PE}(f)=\sum_{i\in\mathcal{N}_{K-1}}h(\mathbf{G}^{i})\mu([\mathbf{i}^{i},\mathbf{u}^{i}])\,.\tag{7}$$
$$\frac{\tilde{\partial}f}{\partial x_{\sigma_{i}}}\bigg{|}_{i}:=\frac{2(\mu_{i}^{r}-\mu_{i}^{l})}{u_{\sigma_{i}}^{i}-l_{\sigma_{i}}^{i}}\approx\frac{\partial f}{\partial x_{\sigma_{i}}}(\mathbf{x})\;\forall\mathbf{x}\in[1^{i},\mathbf{u}^{i}]\,.\tag{3}$$
$$({\mathfrak{s}})$$

We can use this to form Tree-Based Active Subspace (TBAS) for f as follows:

$$\hat{C}_{\mu}^{f}=\sum_{i\in{\mathcal{N}}_{K-1}}\mathbf{G}^{i}\mathbf{G}^{i\top}\mu([\mathbf{I}^{i},\mathbf{u}^{i}])\,.$$
i]). (8)
Which estimator is preferred in practice will depend on which properties of µ are computationally tractable. If both sampling and computing measure of rectangles is simple for µ, both will be available, and in this circumstance, the PBE will be favorable as it does not have Monte Carlo error.

## 4. Asymptotic Analysis

We'll now make rigorous the intuition developed in the previous section with a basic asymptotic analysis of the TBGE under the assumption that the mean value in a node converges at the usual square root rate, which is the case under typical regularity conditions.

Theorem 4.1. Let S(N) denote the least number of splits in the tree along any variable as a function of the sample size. Under Assumption *A.1 of the Appendix, we have that:*

$${\frac{\bar{\partial}f(\mathbf{x})}{\bar{\partial}x_{\sigma_{i}}}}={\frac{\partial f(\mathbf{x})}{\partial x_{\sigma_{i}}}}+O_{P}\!\left({\sqrt{\frac{S(N)}{N}}}+2^{-{\frac{S(N)}{P}}}\right).$$

Proof. We give only an outline of the proof, with the details given in the Appendix. Expanding f around x can be used to show that

$${\frac{\bar{\partial}f(\mathbf{x})}{\bar{\partial}x_{p}}}={\frac{\partial f(\mathbf{x})}{\partial x_{\sigma_{i}}}}+O_{P}\!\left({\frac{1}{\sqrt{\eta(N)}}}\right)+o(\mathrm{diam}([\mathbf{l},\mathbf{u}])\,,$$

where η(N) is the least number of observations in any node.

Then, noting that η(N) ≤N
S(N)
and measuring the decrease in node width with depth yields the desired result. This shows us the trade-off between decreasing the variance of the mean estimate in each cell and narrowing the sizes of the cells. Taking S(N) = P log N yields a convergence rate of pP N −1 log N.

In the remainder of this section, we will establish consistency of estimates for integro-differential quantities, which will require only consistency of the TBGE. We see that this occurs whenever S(N) grows to ∞ sublinearly such that there are eventually arbitrarily many splits, each with arbitrarily many observations as the sample size increases. We begin with the MCE; the remaining proofs are in the Appendix. Theorem 4.2. Under Assumptions A.1 and A.2 of the appendix, the Partition-Based estimator converges to the true Figure 4. **Integrated Gradient for Trees.** Each pair of panels gives a training example from MNIST. The second pair in the image superimposes the IG values onto the example. Redder means more strongly suggesting correct class membership.

integro-differential quantity as the sample size diverges. That is,

$$\operatorname*{lim}_{N\to\infty}{\hat{\mathcal{I}}}_{P B}(f)={\mathcal{I}}(f)\,.$$

where I(f) is the integro-differential quantity given in Equation 4. Theorem A.1 of the Appendix gives a similar result for the MCE.

We also explicitly state there the implications of these results for Active Subspace and Integrated Gradient estimation, namely the consistency of the estimators Cˆfµand ˆIG
proposed in Section 3. We have thus established consistency of gradient-based model interpretation quantities for regression trees. Next, we will empirically study their behavior in finite samples via a battery of numerical experiments.

## 5. Numerical Experiments

We now study how the proposed gradient estimator might be profitably exploited in practice. We begin with a qualitative study showing how a tree-based integrated gradient (TBIG) can facilitate local model interpretation. Then come three quantitative studies, first investigating the potential of a Tree-Based Active Subspace (TBAS) to improve prediction accuracy of a downstream tree via a rotation of the space. Subsequently, we evaluate the capacity of regression trees to estimate the Active Subspace in low and high dimension. Finally, we end with another qualitative study demonstrating how a TBAS can provide data visualization.

## 5.1. Integrated Gradient For Tree-Based Methods

We now consider random forest classifier fit the to two subsets of the MNIST dataset, one contrasting zeros against 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329

TBAS **0.635 0.47 0.578 0.688 0.194 0.856 0.128 0.511** Id **0.655** 0.537 0.595 0.773 0.35 0.963 0.226 **0.514** PCA **0.655** 0.543 **0.591** 0.773 0.349 0.964 0.226 **0.513** Rand 0.656 0.524 0.593 0.752 0.349 0.95 0.226 **0.513**

Regression Tree (Depth 8)

TBAS **0.402 0.35 0.429 0.521 0.078 0.586 0.094 0.392** Id **0.405** 0.403 **0.432 0.522** 0.121 0.862 **0.103 0.395** PCA **0.407 0.395 0.423 0.524** 0.122 0.872 **0.107 0.392** Rand **0.411 0.391 0.435** 0.55 0.124 0.836 **0.109 0.397**

Random Forest (Depth 4)

TBAS **0.609 0.406 0.559 0.602 0.161 0.802 0.123 0.479** Id 0.65 0.462 **0.574** 0.659 0.344 0.954 0.173 **0.485** PCA 0.649 0.462 **0.567** 0.654 0.344 0.954 0.174 **0.486** Rand 0.646 0.458 **0.57** 0.66 0.33 0.938 0.173 **0.486** Dataset: bike concrete gas grid keggu kin40k obesity supercond

Regression Trees Know Calculus Regression Tree (Depth 4)
Table 1. **Predictive Impact of Various Transformations on Selected Datasets**. Numbers give 100-fold RMSE; bold indicates confidence interval overlaps with the lowest confidence interval.

eights and the other fours against sevens. While classification was not the focus of this article, we present some preliminary numerical experiments in Appendix C. We use the estimator ˆIG of Equation 6 with M = 500 random points along a given path. Figure 4 gives the results of this analysis. We see that when comparing eights against zeros, the intersection point of the eight is most important, while the left and right sides of the zero are. This makes sense as they are the parts of each digit which are distinct. By contrast, the shared upper and lower arcs are not highlighted. In the four versus seven case, the location of the horizontal bar seems to be most influential in determining one versus the other. This suggests that the model is comparing the relative position of the bar to the rest of the digit in making its classifications.

## 5.2. Active-Subspace Rotated Trees

This section evaluates the ability of TBAS to improve the accuracy of a downstream predictive analysis. Given some mapping L, we refer to postmultiplication of the feature matrix X to form a new feature matrix Z := XL as a rotation. In standard tree-based methods, non-diagonal rotations can have an impact on predictive performance because the axes along which splits are made have been changed. In order to quantitatively assess the utility of TBAS, we compare the prediction error of regression trees and random forests on eight benchmark datasets fit on data augmented by a rotation of the space. We compare TBAS rotations to the those made by drawing orthogonal directions uniformly over the Grassmannian (Rand; similar to Breiman (2001)) as well as those formed from a Principal Components Analysis (PCA; similar to Rodriguez et al. (2006)) and no rotation at all (denoted Id). Table 1 presents the results of this analysis. TBAS does at least as well as the other methods, and sometimes offers significant improvement (such as on the Kin40k and Keggu datasets).

## 5.3. Computationally Efficient Active Subspace Estimation In Low Dimension

In this section we demonstrate the capability of TBAS to offer extremely fast, gradient-free estimation of the active subspace in low dimension when significant data are available. We compare to the three popular methods for gradient-free active subspace estimation introduced in Section 2.2, based on a Gaussian Process (GP), Ridge Polynomial (PRA), and Neural Network (DASM). Each of these methods was tasked to estimate a one dimensional active subspace in dimensions 2, 3 and 4, using a logarithmically spaced grid of sample sizes ranging from 10 to 10,000 on a toy algebraic function. We found that the GP and PRA methods worked efficiently when sample sizes were small, but computation time grew quickly, such that we were only able to run these methods for samples of size 150 or less. Figure 5 shows the active subspace error against the execution time. We see that the tree-based estimator forms the majority of the Pareto front in all three cases.

## 5.4. Estimating Sparse Active Subspaces In High Dimension

330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 Imposing sparsity in the coefficients of linear dimension reduction can improve interpretability (Zou et al., 2006). Because of the manner that tree-based methods produce gradient estimates, we conjecture that TBAS has built-in inductive bias to favor entry-wise active subspaces. Our analysis in Appendix A.2 revealed that in the large sample limit, the tree-based gradient estimator will behave as though it was in a lower dimensional space, with dimension given by the cardinality of the gradient. In this section, we will investigate this property by comparing TBAS against the DASM in estimating a sparse active subspace in dimensions 10, 50 and 100. We will use the same setup as the previous section, except that the true subspace has only three nonzero coordinates. Because of the sample size that is required to estimate an active subspace in high dimension, it is computationally infeasible to deploy the GP or PRA active subspace estimation methods. Figure 6 shows the results of this analysis. The TBAS estimator is able to achieve much better accuracy with a significantly smaller cost,

## 5.5. Dimension Reduction With The Active Subspace

We now turn to the qualitative benefits of performing an active subspace analysis using TBAS. We used the NHEFS dataset of biochemistry tape and mortality data which were used by Lundberg et al. (2019). This consisted of a dataset of 14,407 observations and 90 complete variables. We fit a TBAS to this dataset using a regression tree of depth 15, requiring at least 10 samples per leaf. We subsequently performed an eigendecomposition of the estimated active subspace matrix. Like Lundberg et al. (2019), we found that age was the most important variable and it mapped cleanly onto the first active subspace dimension. However, we found that the next two dimensions consisted of many

 P=2 10 2 10 0 10 2 Execution Time (s)
0.2 0.4 0.6 0.8 1.0 P=3 0.6 0.7 0.8 0.9 1.0 1.1 1.2 P=4 10 2 10 0 10 2 Execution Time (s)
0.0 0.2 0.4 0.6 E
rr or 
(
R
a di a ns
)

DASM GP PRA Tree 10 2 10 0 10 2 Execution Time (s)
variables (see Appendix B.3). Figure 7 shows there result of this analysis, which reveals that after Age there appears to be a gap in the spectrum of the active subspace matrix (Constantine, 2015), indicating the presence of an active subspace of dimension two (left panel). The middle panel shows a projection of the data, which breaks into two clusters along the second principal component, while the right panel shows a projection of randomly sampled points to visualize the predictive surface of the function. The right panel reveals the predictive surface has a fairly simple, "S-
shaped" form over these two dimension. It would not be possible to detect this by considering only main effects or two factor interactions.

## 6. Discussion

Summary: We proposed a simple method for estimating gradients in sufficiently deep regression trees on large datasets. Subsequently, we demonstrate how these estimates could be used to calculate active subspaces and integrated gradients. We found significant improvement in predictive performance could be achieved by using the tree-based active subspace to rotate the space, and also found that these estimates executed quicker than existing gradient-free active subspace estimators and had useful inductive bias towards detecting coordinate-sparse subspaces. Also, we were not able to run the PRA and GP based active subspace estimators on larger datasets due to computational constraints. Finally, we used these gradient estimates to reveal the simple global structure of a tree-based model fit to a complex dataset as well as the mechanism by which a random forest conducted MNIST classification.

Limitations: When it comes to limitations of our experiments, when comparing our active subspace estimates to the DASM, we used a particular neural network architecture and optimizer. It is possible that a different configuration 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 would have performed better. Additionally, when it comes to limitations of the method, our analysis suggests that it would take a very deep regression tree to estimate gradients in high dimension if the gradient is dense in all of its entries. Conclusions: We think this work offers two high level conclusions. First, that trees may have a place in the field of Uncertainty Quantification, which has more commonly used differentiable but slow to estimate surrogates such as Gaussian processes and neural networks. Secondly, we hope that this work can also enable improve cross-pollination between developments in interpretability for neural networks and for regression trees by creating analogs for gradient-based neural network techniques.

We also would like to comment on how TBAS fit into the existing literature on interpretability in trees. It is somewhat distinct from the axiomatic approach to interpretability proffered by SHAP (Lundberg & Lee, 2017) and Integrated Gradients (Sundararajan et al., 2017). While an axiomatic approach can be useful, its universality should not be exaggerated. Hancox-Li & Kumar (2021) write about how it is unlikely that any variable selection method could satisfy all possible demands, no matter how attractive its axiomatic foundation. Since the active subspace is parameterized by a measure, it actually consists of an entire family of analyses, with different choices of emphasis over the input space possibly leading to different conclusions.

Future Work: We are excited about the future work opened up by the here-proposed gradient estimates. First, while we have conducted some preliminary analyses in Appendix C, more work is needed to understand the properties of these estimates in classification problems with categorical inputs and missing data. We are also interested in the possibility of estimating higher order derivatives from tree structure. Furthermore, extending the class of functions approximated to nondifferentiable functions is of interest: does the quantity proposed here converge in such a case, and to what? But we are most excited about the potential to bring in other gradient-based techniques to tree-based methods. For instance, physics-informed machine learning (e.g. Raissi et al. (2019; 2017)) has been making waves in the fluid dynamics community (Cai et al., 2021) among other applications, where they allow the analyst to use information about function derivatives to improve predictions of differentiable models such as neural networks or Gaussian processes. Our gradient estimator opens up the possibility of deploying this technique in the context of regression trees.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

Error P=10 Error P=50 Error P=100 log Se con ds Time P=10 log Se con ds Time P=50 log Se con ds Time P=100 10 2 10 3 10 4 N
2 0 2 10 2 10 3 10 4 N
2 0 2 10 2 10 3 10 4 N
2 0 2 10 2 10 3 10 4 N
0 1 Rad ia ns 10 2 10 3 10 4 N
0 1 Rad ia ns 10 2 10 3 10 4 N
0 1 Rad ia ns DASM
Tree

## References

Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., and Zhang, Q. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/google/jax.

Breiman, L. Random forests. *Machine learning*, 45:5–32, 2001.

Breiman, L. Manual on setting up, using, and understanding random forests v3. 1. Statistics Department University of California Berkeley, CA, USA, 1(58):3–42, 2002.

Breiman, L. *Classification and regression trees*. Routledge, 2017.

Cai, S., Mao, Z., Wang, Z., Yin, M., and Karniadakis, G. E.

Physics-informed neural networks (pinns) for fluid mechanics: A review. *Acta Mechanica Sinica*, 37(12):1727– 1738, 2021.

Chaudhuri, P., Lo, W.-D., Loh, W.-Y., and Yang, C.-C. Generalized regression trees. *Statistica Sinica*, pp. 641–666, 1995.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Chen, H., Janizek, J. D., Lundberg, S., and Lee, S.-I.

True to the model or true to the data? *arXiv preprint* arXiv:2006.16234, 2020.

Constantine, P. G. Active subspaces: Emerging ideas for dimension reduction in parameter studies. SIAM, 2015.

Edeling, W. On the deep active-subspace method.

SIAM/ASA Journal on Uncertainty Quantification, 11(1): 62–90, 2023.

Friedman, J. H. Greedy function approximation: a gradient boosting machine. *Annals of statistics*, pp. 1189–1232, 2001.

Hancox-Li, L. and Kumar, I. E. Epistemic values in feature importance methods: Lessons from feminist epistemology. In proceedings of the 2021 ACM conference on fairness, accountability, and transparency, pp. 817–826, 2021.

Hechtlinger, Y. Interpretation of prediction models using the input gradient. *arXiv preprint arXiv:1611.07634*, 2016.

Hokanson, J. M. and Constantine, P. G. Data-driven polynomial ridge approximation using variable projection. SIAM
Journal on Scientific Computing, 40(3):A1566–A1589, 2018.

Hooker, G., Mentch, L., and Zhou, S. Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance. *Statistics and Computing*, 31:1–16, 2021.

Ishwaran, H. Variable importance in binary regression trees and forests. 2007.

Karczmarz, A., Michalak, T., Mukherjee, A., Sankowski, P.,
and Wygocki, P. Improved feature importance computation for tree models based on the banzhaf value. In Uncertainty in Artificial Intelligence, pp. 969–979. PMLR, 2022.

Kazemitabar, J., Amini, A., Bloniarz, A., and Talwalkar, A. S. Variable importance using decision trees. *Advances* in neural information processing systems, 30, 2017.

Klusowski, J. M. and Tian, P. M. Nonparametric variable screening with optimal decision stumps. arXiv preprint arXiv:2011.02683, 2020.

Li, X., Wang, Y., Basu, S., Kumbier, K., and Yu, B. A debiased mdi feature importance measure for random forests. Advances in Neural Information Processing Systems, 32, 2019.

Loh, W.-Y. Classification and regression trees. Wiley interdisciplinary reviews: data mining and knowledge discovery, 1(1):14–23, 2011.

Louppe, G. Understanding random forests: From theory to practice. *arXiv preprint arXiv:1407.7502*, 2014.

Lundberg, S. M. and Lee, S.-I. A unified approach to interpreting model predictions. Advances in neural information processing systems, 30, 2017.

Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., and Lee, S.-I. Explainable ai for trees: From local explanations to global understanding. arXiv preprint arXiv:1905.04610, 2019.

McCloskey, K., Taly, A., Monti, F., Brenner, M. P., and Colwell, L. J. Using attribution to decode binding mechanism in neural network models for chemistry. *Proceedings of* the National Academy of Sciences, 116(24):11624–11629, 2019.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V.,
Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.

Raissi, M., Perdikaris, P., and Karniadakis, G. E. Machine learning of linear differential equations using gaussian processes. *Journal of Computational Physics*, 348:683–
693, 2017.

Raissi, M., Wang, Z., Triantafyllou, M. S., and Karniadakis, G. E. Deep learning of vortex-induced vibrations. *Journal* of Fluid Mechanics, 861:119–137, 2019.

Rodriguez, J. J., Kuncheva, L. I., and Alonso, C. J. Rotation forest: A new classifier ensemble method. IEEE transactions on pattern analysis and machine intelligence, 28 (10):1619–1630, 2006.

Sandri, M. and Zuccolotto, P. A bias correction algorithm for the gini variable importance measure in classification trees. *Journal of Computational and Graphical Statistics*, 17(3):611–628, 2008.

Sayres, R., Taly, A., Rahimy, E., Blumer, K., Coz, D., Hammel, N., Krause, J., Narayanaswamy, A., Rastegar, Z., Wu, D., et al. Using a deep learning algorithm and integrated gradients explanation to assist grading for diabetic retinopathy. *Ophthalmology*, 126(4):552–564, 2019.

Shan, H., Zhang, J., and Kruger, U. Learning linear representation of space partitioning trees based on unsupervised kernel dimension reduction. *IEEE Transactions on* Cybernetics, 46(12):3427–3438, 2015.

Strobl, C., Boulesteix, A.-L., Zeileis, A., and Hothorn, T.

Bias in random forest variable importance measures: Illustrations, sources and a solution. *BMC bioinformatics*, 8:1–21, 2007.

Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T., and Zeileis, A. Conditional variable importance for random forests. *BMC bioinformatics*, 9:1–11, 2008.

Sundararajan, M., Taly, A., and Yan, Q. Axiomatic attribution for deep networks. In International conference on machine learning, pp. 3319–3328. PMLR, 2017.

Tripathy, R. and Bilionis, I. Deep active subspaces: A
scalable method for high-dimensional uncertainty propagation. In *International Design Engineering Technical* Conferences and Computers and Information in Engineering Conference, volume 59179, pp. V001T02A074.

American Society of Mechanical Engineers, 2019.

Wycoff, N., Binois, M., and Wild, S. M. Sequential learning of active subspaces. *Journal of Computational and* Graphical Statistics, 30(4):1224–1237, 2021.

Zou, H., Hastie, T., and Tibshirani, R. Sparse principal component analysis. Journal of computational and graphical statistics, 15(2):265–286, 2006.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549

## A. Proofs

We begin by stating our required assumptions. Assumption 1 will be required for Theorem A, establishing the convergence rate of our gradient estimator. The other results, regarding consistency of integro-differential estimators, will require Assumption 2 as well.

Assumption 1. Let yn = f(xn) + ϵn*. We assume that:*
1. The input points xn *are sampled independently from a continuous distribution on* [0, 1]P .

2. The estimate of the mean of f(x) over some interval [a, b] converges at usual square root rate; this is achieved for example if the error terms ϵ1, . . . , ϵN *are such that* E[ϵn] = 0; V[ϵn] ≤ ∞ and ϵn1 ⊥⊥ ϵn2 ∀n1, n2 ∈ {1*, . . . , N*}.

3. We alternate between which variable is split along at each depth. 4. f *is continuously differentiable.*
Items 1 and 2 are necessary for the mean estimates in leaf nodes to converge to the mean value of f on that node, while 3 and 4 are necessary to make a series of f within a node accurate. Assumption 2. Let S(N) *denote the depth as a function of sample size.*
2. limN→∞
S(N)
N = 0.

3. h *is measurable and bounded.*
550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 Assumptions 1 and 2 ensure that we accumulate enough data in each leaf node, while still accumulating sufficiently small leaf nodes, to achieve gradient consistency.

We are now prepared to prove our results.

Theorem 4.1. Let S(N) denote the least number of splits in the tree along any variable as a function of the sample size. Under Assumption *A, we have that:* Proof. If x ∈ [l, u], expand f around u (or any other point in [l, u]):
where diam(A) is the diameter of the set A and is given by maxx1,x2∈A ∥x1 − x2∥. We denote the minimal number of points in any node by η(N). Then:

$${\frac{\ddot{\partial}f(\mathbf{x})}{\dot{\partial}x_{p}}}={\frac{\partial f(\mathbf{x})}{\partial x_{\sigma_{i}}}}+O_{P}\left({\frac{1}{\sqrt{\eta(N)}}}\right)+o(\mathrm{diam}([\mathbf{l},\mathbf{u}]).$$
+ o(diam([l, u]). (11)
$$\begin{array}{r l}{f(\mathbf{x})=f(\mathbf{u})+\nabla f(\mathbf{u})^{\top}(\mathbf{x}-\mathbf{u})+o(\|\mathbf{x}-\mathbf{u}\|)}\\ {=f(\mathbf{u})+\nabla f(\mathbf{u})^{\top}(\mathbf{x}-\mathbf{u})+o(\operatorname{diam}([\mathbf{l},\mathbf{u}]))\,,}\end{array}$$
⊤(x − u) + o(∥x − u∥) (9)
⊤(x − u) + o(diam([l, u])), (10)
Denote by S(N) the depth of the tree as a function of the sample size. If we are making cuts at midpoints in each node, then diam = max1≤p≤P1 2 Sp, where Sp is the number of splits along variable p. It's clear that reducing the diameter is most speedily achieved by alternating which variable is split along such that diam = 2−⌊ SP
⌋ ≤ C2
− S
P for some C.

We have that η(N) ≤N
S(N) with equality if the points are evenly distributed. So OP (
q 1 η(N)
) = OP (
qS(N)
N) and o(diam([l, u]) = o(2− SP ), which taken together with 11 yield the desired result.

$$({\mathfrak{g}})$$
$$(10)^{\frac{1}{2}}$$
$$(11)$$

1. limN→∞ S(N) = ∞.

$${\frac{\bar{\partial}f(\mathbf{x})}{\bar{\partial}x_{\sigma_{i}}}}={\frac{\partial f(\mathbf{x})}{\partial x_{\sigma_{i}}}}+O_{P}\!\left({\sqrt{\frac{S(N)}{N}}}+2^{-{\frac{S(N)}{P}}}\right).$$

Theorem 4.2. Under Assumptions A and A, the Partition-Based estimator converges to the true integro-differential quantity as the observation sample size diverges. That is,

$$\operatorname*{lim}_{N\to\infty}{\hat{\mathcal{I}}}_{P B}(f)=\operatorname*{lim}_{N\to\infty}\sum_{i\in{\mathcal{N}}_{K-1}}h(\mathbf{G}^{i})\mu([\mathbf{i}^{i},\mathbf{u}^{i}])\,.=\int h(\nabla f(\mathbf{x}))d\mu(\mathbf{x})={\mathcal{I}}(f)\,.$$

Proof. Since the function sequence g k(x) := GBk(x) converges pointwise to ∇f(x) by Proposition 1, we need only establish a function H(x) which dominates g k(x) and apply the Dominated Convergence Theorem.

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 To this end, denote by C
r, Clthe child nodes of node i and examine its gradient estimator's pth entry, given by:

$$\frac{2(\mu_{l}^{r}-\mu_{l}^{i})}{u_{p}^{i}-l_{p}^{i}}\stackrel{N\rightarrow\infty}{\longrightarrow}\frac{2\left(\frac{1}{|\mathcal{N}_{C r}|}\int_{\mathcal{N}_{C r}}f(\mathbf{x})d\mathbf{x}-\frac{1}{|\mathcal{N}_{C r}|}\int_{\mathcal{N}_{C l}}f(\mathbf{x})d\mathbf{x}\right)}{u_{p}^{i}-l_{p}^{i}}\,.$$
$$(12)$$
$$(13)$$
$$(14)$$
$$(15)$$

The magnitude of this difference in averages is bounded by the magnitude of the difference of extremes:

$$|\frac{1}{|{\cal N}_{C_{1}^{+}}|}\int_{{\cal N}_{C_{1}^{+}}}f({\bf x})d{\bf x}-\frac{1}{|{\cal N}_{C_{1}^{+}}|}\int_{{\cal N}_{C_{1}^{+}}}f({\bf x})d{\bf x}|\leq\operatorname*{max}_{({\bf x}_{1},{\bf x}_{2})\in{\cal N}_{C_{1}^{+}}\times{\cal N}_{C_{1}^{+}}}|f({\bf x}_{1})-f({\bf x}_{2})|\,.$$

But since f is continuously differentiable, it is also Lipschitz continuous (call the constant L), and we have that:

$$\operatorname*{max}_{(\mathbf{x}_{1},\mathbf{x}_{2})\in\mathcal{N}_{C_{\mathrm{T}}^{\mathrm{T}}}\times\mathcal{N}_{C_{\mathrm{I}}^{\mathrm{T}}}}|f(\mathbf{x}_{1})-f(\mathbf{x}_{2})|\leq L\|\mathbf{x}_{1}-\mathbf{x}_{2}\|_{2}\leq PL\|\mathbf{x}_{1}-\mathbf{x}_{2}\|_{\infty}\,.$$

Therefore:

$$|\frac{2(v_{C_{i}^{*}}-v_{C_{i}^{*}})}{u_{p}^{i}-l_{p}^{i}}|\leq|\frac{2(P L\|{\bf x}_{1}-{\bf x}_{2}\|_{\infty})}{u_{p}^{i}-l_{p}^{i}}|\leq2P L\,.\tag{1}$$

Hence the constant 2P L bounds g k(x), and thence 4P
2L
2 bounds the outer product function. Since the integral is over the unit hypercube, the constant function is integrable and we can apply the Dominated Convergence Theorem to yield the desired result. Theorem A.1. Under Assumptions A and A, the Monte Carlo estimator converges to the true integro-differential quantity as the Monte Carlo sample size and the observation sample size diverge. That is,

$$\operatorname*{lim}_{N,M\to\infty}\hat{\mathcal{I}}_{M C}(f)=\operatorname*{lim}_{N,M\to\infty}{\frac{1}{M}}\sum_{\mathbf{x}_{m}\sim\mu}h(\mathbf{G}^{\mathbf{x}_{m}})=\int h(\nabla f(\mathbf{x}))d\mu(\mathbf{x})={\mathcal{I}}(f)\,.$$

Proof. This follows from the fact that and an application of Theorem 4.2.

$$\lim_{N\to\infty}\lim_{M\to\infty}\frac{1}{M}\sum_{\mathbf{x}_{m}\sim\mu}\mathbf{G}^{B^{K(N)-1}(\mathbf{x}_{m})}\mathbf{G}^{B^{K(N)-1}(\mathbf{x}_{m})\top}=$$ $$=\lim_{N\to\infty}\int_{\mathbf{x}\in[0,1]^{P}}\mathbf{G}^{B^{K(N)-1}(\mathbf{x})}\mathbf{G}^{B^{K(N)-1}(\mathbf{x})\top}d\mu=\lim_{N\to\infty}\sum_{i\in\mathcal{D}_{k}}\mathbf{G}_{i}\mathbf{G}_{i}^{\top}\mu(\mathcal{N}_{i})$$
$$\square$$
$$(16)$$
$$(17)$$

A.1. Implications for Active Subspaces and Integrated Gradients We conclude this section by explicitly stating the implications of these results for Tree-based Active Subspace (TBAS) estimation. Corollary A.2. Let K(N) denote the depth of the regression tree as a function of N. Under Assumptions A and *A, we have* that Proof. Follows from Theorem 4.2. As well as for Tree-Based Integrated Gradient (TBIG) estimation. Corollary A.3. Let K(N) denote the depth of the regression tree as a function of N. Under Assumptions A and A, we have that, assuming that un *are iid uniform on [0,1]:* Proof. Follows from Theorem A.1.

## A.2. Thresholding Behavior And Iteration

For any fixed s, the minimum with respect to t occurs at ui,s+li,s 2, yielding the profile problem:

$$\operatorname*{argmin}_{s\in\{1,\ldots,P\}}2\sum_{r\not=s}a_{p}^{2}(u_{p}^{i}-l_{p}^{i})^{2}+{\frac{a_{p}^{2}}{2}}(u_{s}^{i}-l_{s}^{i})^{2}\,.$$

p̸=s
a
2p
2. (19)
660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714

p̸=ρ
$$\sum_{p\neq a}\bigl\lfloor\log_{2}{\frac{|a_{p}|}{|a_{p^{\wedge}}|}}\bigr\rfloor\leq(P-1)\log_{2}{\frac{|a_{p_{1}}|}{|a_{p^{\wedge}}|}}\,.$$
Subsequently, in the second phase, the algorithm will subdivide boxes by proceeding alternatively through the variables one by one *ad infinitum*. If one or more ap is zero, the analysis above can be applied on the nonzero parameters. This reduces the number of iterations until the first phase concludes and the second starts below ∥a∥0 log2maxp |ap| min{p:ap̸=0} |ap|
.

$$\operatorname*{lim}_{N\to\infty}\sum_{i\in{\mathcal{D}}_{K(N)-1}}\mathbf{G}_{i}\mathbf{G}_{i}^{\top}\mu({\mathcal{N}}_{i})$$ $$=\int_{[0,1]^{P}}\nabla f(\mathbf{x})\nabla f(\mathbf{x})d\mu(\mathbf{x})\,.$$
$$\lim_{k,N,M\to\infty}({\bf x}-{\bf x}^{*})\odot\frac{1}{M}\sum_{m=1}^{M}{\bf G}^{B^{k}}\big{(}u_{m}{\bf x}+(1-u_{m}){\bf x}^{*}\big{)}$$ $$=({\bf x}-{\bf x}^{*})\odot\int_{\alpha=0}^{1}\nabla f(\alpha{\bf x}+(1-\alpha){\bf x}^{*})d{\bf x}\,.$$
$\square$
$\square$
$$(18)$$
$$(19)^{\frac{1}{2}}$$

In this section we study large sample iteration behavior of a greedily estimated regression tree on linear functions f(x) =

a
⊤x + b. As the sample size diverges, we have that V[a
⊤x; [l, u]] =
Pp
a
2 p
(u
i
p−l
i p
)
2
12 such that that optimization may be
rewritten as:
$$\operatorname*{argmin}_{a\in\{1,\ldots,n\}}2\sum a_{p}^{2}(u_{p}^{i}-l_{p}^{i})^{2}+a_{s}^{2}\Bigl((u_{p}^{i}-t)^{2}+(t-l_{p}^{i})^{2}\Bigr).$$
s∈{1,...,P }, t∈[l
ip
,uip
]
p̸=s
2. (18)
$$(20)$$
By comparing any two costs, we can see that the s that will be chosen is that such that the quantity |as|(u is − l is) is maximized. Such is the behavior for a specific node at a specific depth. Let's now reflect on what this implies for the iteration. Initially, up−lp = 1 for all p, such that the variable with the largest coefficient p will be chosen for the first split; let's call that variable p1 := argmaxp|ap|. Subsequently, if p1 is more than twice the size of the second largest coefficient p2 := argmaxp̸=p1 |ap|, the second split in each of the two leaves will occur along variable p1 as well. Otherwise, it will occur along variable p2. Under our large sample assumptions, all the nodes at a given depth will behave uniformly, splitting alternatively along variables until the terms |ap|(u ip − l ip) are within a multiple of two of one another. Denote the index of the minimum coefficient as p
∧ := argminp |ap| and assume for now that ap∧ is nonzero. Using the notation ⌊c⌋ to denote the integer part of c, the number of iterations in this first stage is thus given by

## B. Details Of Numerical Experiments

This section gives additional details and discussion of the numerical results presented in Section 5. All tree-based models are estimated using Scikit-Learn (Pedregosa et al., 2011).

## B.1. Rotation Prediction Study Additional Details

In this study, when performing a rotation, we used only the 
√P many PCA components or Active Subspace dimensions, and appended these to the original design matrix as new variables. We measure prediction error using 100-fold cross validation.

The table below gives the parameters of the datasets used for the prediction study of Section 5.2.

| Name      | N      | P   | URL                                                           |
|-----------|--------|-----|---------------------------------------------------------------|
| concrete  | 1,030  | 9   | https://archive.ics.uci.edu/dataset/165/                      |
| kin40k    | 40,000 | 9   | https://github.com/alshedivat/keras-gp/kgp/datasets/kin40k.py |
| keggu     | 65,554 | 28  | https://www.genome.jp/kegg/pathway.html                       |
| bike      | 17,379 | 13  | https://archive.ics.uci.edu/dataset/560/                      |
| obesity   | 2,111  | 24  | https://archive.ics.uci.edu/dataset/544/                      |
| gas       | 36,733 | 12  | https://archive.ics.uci.edu/dataset/224                       |
| grid      | 10,000 | 13  | https://archive.ics.uci.edu/dataset/471/                      |
| supercond | 21,263 | 82  | https://archive.ics.uci.edu/dataset/464/                      |

We also provide boxplots of Cross Validation errors in Figure 8. Running this study took about five hours on a 40 core Ubuntu machine with 128 GB of RAM.

## B.2. Active Subspace Estimation Study Additional Details

In Section 5.3, We randomly sampled a unit vector a from the uniform distribution over directions and then sampled input points uniformly at random on the unit cube. We subsequently evaluated the function f(x) = cos(6π(a
⊤(x − 0.5)) which was treated as the noiseless observed response y. We compared the estimates with using the angle each made with a. This experiment was repeated 20 times.

We used the implementation of PRA provided with the pypi package PSDR1. For GP-based active subspace estimation, we used the CRAN package activegp. We used all default settings for the PRA method as well as for the GP method. For the DASM, we used a neural network with an additional layer of width 512 subsequent to the active subspace layer, and used gradient descent with a step size of 10−3 on the Mean Squared Error cost function. This neural network was implemented in JAX (Bradbury et al., 2018). In addition to the quantitative advantages enjoyed by the tree-based method of active subspace estimation, we would also like to note that like the GP-based method, and unlike the PRA and DASM, it provides an estimate of the entire active subspace matrix, rather than simply a basis for the active subspace. This is important for two reasons. First, with the active subspace matrix in hand, we can create analogs of PCA scree plots to determine what dimension of the active subspace is most desirable, or to get some idea of how much information is being lost in, say, a two dimensional visualization. And secondly, it allows us to decide on an active subspace dimension *after* having seen the data rather than before, without requiring the estimation procedure to be re-run. Running this study took about eight hours on a 40 core Ubuntu machine with 128 GB of RAM.

## B.3. Nhefs Data Analysis Details

This table presents the first 3 eigenvectors of the mortality data analysis, restricting to the top 9 variables with highest coefficients. The first eigenvector captures almost entirely the age variable, which (Lundberg et al., 2019) also found to be most important. We see that the second eigenvector is evenly distributed across sex and one of the urine variables. The third eigenvector is dominated by the urineDark variable.

715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769

| Eigenvector   | age   | urineDark   | sex   | urineNeg   | SGOT   | hemoglobin   | urineAlb   | total   | physical   |
|---------------|-------|-------------|-------|------------|--------|--------------|------------|---------|------------|
| 1             | -1.00 | -0.00       | 0.01  | 0.01       | -0.00  | -0.00        | 0.00       | -0.00   | -0.00      |
| 2             | 0.01  | -0.15       | 0.56  | 0.53       | -0.28  | -0.30        | 0.30       | -0.16   | 0.09       |
| 3             | -0.00 | -0.96       | -0.03 | -0.11      | 0.16   | 0.07         | 0.02       | 0.04    | 0.07       |

1https://psdr.readthedocs.io/en/latest/
770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824

1.0 Tree Depth 4 1.0 concrete TBAS Id PCA Rand 0.5 0.6 0.7 gas grid TBAS Id PCA Rand 0.6 0.8 TBAS Id PCA Rand 0.5 kin40k obesity TBAS Id PCA Rand 0.5 0.6 supercond 0.2 TBAS Id PCA Rand 0.8 TBAS Id PCA Rand 0.0 Tree Depth 8 TBAS Id PCA Rand 0.5 1.0 concrete gas TBAS Id PCA Rand 0.5 0.6 0.7 grid TBAS Id PCA Rand 0.4 0.5 kin40k obesity supercond TBAS Id PCA Rand 0.3 0.4 0.5 0.2 TBAS Id PCA Rand 0.6 0.8 TBAS Id PCA Rand 0.0 Random Forest Depth 4 TBAS Id PCA Rand 0.25 0.50 0.75 concrete TBAS Id PCA Rand 0.5 0.6 0.7 gas grid TBAS Id PCA Rand 0.5 0.6 0.7 TBAS Id PCA Rand 0.8 1.0 kin40k TBAS Id PCA Rand 0.1 0.2 0.3 obesity supercond TBAS Id PCA Rand 0.4 0.5 bike TBAS Id PCA Rand 0.5 0.6 0.7 TBAS Id PCA Rand 0.2 0.3 0.4 keggu TBAS Id PCA Rand 0.6 0.8 bike keggu 0.4 TBAS Id PCA Rand 0.2 TBAS Id PCA Rand 0.4 0.5 bike keggu TBAS Id PCA Rand 0.1 0.2
When producing the right panel of Figure 7, we used the prediction after accounting for the effect of age in order to demonstrate the change in the predictive surface over the second and third eigenvalues.

## C. Numerical Experiments On Classification Trees

We repeat the experiments of Section 5.2, but now by replacing each regression problem with a classification problem by assessing whether a given observation falls above or below the median observation. The results are given in Figure 9. Intriguingly, the results are significantly less promising for the TBAS method, despite the fact that by construction, there is structure in the data that TBAS could possibly exploit. This indicates that there may be special considerations to be taken in deploying this methodology to classification problems.

825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934

Tree Depth 4 TBAS Id PCA Rand 0.6 0.8 bike 1.0 concrete TBAS Id PCA Rand 0.5 0.6 0.7 gas grid TBAS Id PCA Rand 0.6 0.8 TBAS Id PCA Rand 0.5 keggu kin40k obesity TBAS Id PCA Rand 0.5 0.6 supercond 0.4 1.0 0.2 TBAS Id PCA Rand 0.2 TBAS Id PCA Rand 0.8 TBAS Id PCA Rand 0.0 Tree Depth 8 TBAS Id PCA Rand 0.4 0.5 bike TBAS Id PCA Rand 0.5 1.0 concrete gas TBAS Id PCA Rand 0.5 0.6 0.7 grid TBAS Id PCA Rand 0.4 0.5 keggu kin40k obesity supercond TBAS Id PCA Rand 0.3 0.4 0.5 TBAS Id PCA Rand 0.1 0.2 0.2 TBAS Id PCA Rand 0.6 0.8 TBAS Id PCA Rand 0.0 Random Forest Depth 4 bike TBAS Id PCA Rand 0.25 0.50 0.75 concrete TBAS Id PCA Rand 0.5 0.6 0.7 gas grid TBAS Id PCA Rand 0.5 0.6 0.7 TBAS Id PCA Rand 0.5 0.6 0.7 TBAS Id PCA Rand 0.2 0.3 0.4 keggu TBAS Id PCA Rand 0.8 1.0 kin40k TBAS Id PCA Rand 0.1 0.2 0.3 obesity supercond TBAS Id PCA Rand 0.4 0.5