# Stochastic Smoothed Primal-Dual Algorithms for Nonconvex Optimization with Linear Inequality Constraints

Ruichuan Huang <sup>1</sup> Jiawei Zhang \* 2 3 Ahmet Alacaoglu \* 1

## Abstract

We propose smoothed primal-dual algorithms for solving stochastic nonconvex optimization problems with linear *inequality* constraints. Our algorithms are single-loop and only require a single (or two) samples of stochastic gradients at each iteration. A defining feature of our algorithm is that it is based on an inexact gradient descent framework for the Moreau envelope, where the gradient of the Moreau envelope is estimated using one step of a stochastic primal-dual (linearized) augmented Lagrangian algorithm. To handle inequality constraints and stochasticity, we combine the recently established global error bounds in constrained optimization with a Moreau envelope-based analysis of stochastic proximal algorithms. We establish the optimal (in their respective cases) O(ε −4 ) and O(ε −3 ) sample complexity guarantees for our algorithms and provide extensions to stochastic linear constraints. Unlike existing methods, iterations of our algorithms are free of subproblems, large batch sizes or increasing penalty parameters in their iterations and they use dual variable updates to ensure feasibility.

## 1. Introduction

We focus on the problem template

$$\min_{\mathbf{x} \in X} f(\mathbf{x}) \text{ subject to } A\mathbf{x} = \mathbf{b}, \quad (1)$$

where f : R <sup>n</sup> → <sup>R</sup> is L<sup>f</sup> -smooth, the set X ⊆ <sup>R</sup> <sup>n</sup> is polyhedral, and easy to project. In particular, let X be given as X = {x: Hx ≤ h} for some matrix H and vector h. Taking H = I, for example, gives this template the ability to model *linear inequality* constraints.

In particular, when we have the problem

$$\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x}) \text{ subject to } A\mathbf{x} \leq \mathbf{b}, \quad (2)$$

we introduce a slack variable t = Ax−b so that Ax−t = b and our optimization variable becomes x t . Then, we can equivalently write the problem in the template [\(1\)](#page-0-0) by using the constraint t ≤ 0, where the set X = { x t : x ∈ R <sup>n</sup>, t ≤ 0} is easy to project. As such, we focus on [\(1\)](#page-0-0) and our results directly apply to solving [\(2\)](#page-0-1) by using this standard slack variable reformulation.

The assumption of X being easy-to-project is without loss of generality. Indeed, when X is not easy to project, we can add a slack variable for Hx ≤ h similar to the above paragraph, to have a linear equality constrained problem with projectable constraints (cf. [\(1\)](#page-0-0)). We refer to [\(Li et al.,](#page-9-0) [2021,](#page-9-0) Remark 6), for the classical conversion of an ε-stationarity point of the problem with the slack variable to the original inequality constrained problem. Throughout, we assume that we have access to an unbiased oracle F(x) such that

$$\mathbb{E}[F(\mathbf{x})] = \nabla f(\mathbf{x}), \text{ and } \mathbb{E}\|F(\mathbf{x}) - \nabla f(\mathbf{x})\|^2 \leq \sigma^2. \quad (3)$$

A common setting is when f(x) = <sup>E</sup>ξ∼Ξ[f(x, ξ)] where Ξ is an unknown distribution that we can draw i.i.d. samples from. In this case, it is common to set F(x) = ∇f(x, ξ) where <sup>E</sup>[∇f(x, ξ)] = ∇f(x). This will be our main focus.

Inclusion of the set X in [\(1\)](#page-0-0) increases the modeling power of [\(1\)](#page-0-0) significantly, while causing difficulties in the analysis. Many problems fit this template, including constrained and distributed optimization, nonnegative matrix factorization, sparse subspace estimation and collaborative learning, see for example[<sup>1</sup>](#page-0-2) [\(Zhang et al.,](#page-10-0) [2022;](#page-10-0) [Hong,](#page-9-1) [2016\)](#page-9-1). Moreover, reformulations of nonconvex problems are also common by using linear inequality constraints [\(Zhang et al.,](#page-10-0) [2022\)](#page-10-0).

Algorithm development for [\(1\)](#page-0-0) and related templates with global complexity guarantees, have been active in the last couple of years [\(Alacaoglu & Wright,](#page-9-2) [2024;](#page-9-2) [Zhang & Luo,](#page-10-1) [2020;](#page-10-1) [Zhang et al.,](#page-10-2) [2020;](#page-10-2) [Lu et al.,](#page-10-3) [2024;](#page-10-3) [Li et al.,](#page-9-0) [2021;](#page-9-0) [Lin et al.,](#page-10-4) [2022;](#page-10-4) [Yan & Xu,](#page-10-5) [2022;](#page-10-5) [Li et al.,](#page-9-3) [2024;](#page-9-3) [Boob](#page-9-4)

<sup>\*</sup>Co-last authors <sup>1</sup>University of British Columbia <sup>2</sup>University of Wisconsin–Madison <sup>3</sup>MIT. Correspondence to: Jiawei Zhang <jzhang2924@wisc.edu>, Ahmet Alacaoglu <alacaoglu@math.ubc.ca>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

<sup>1</sup>[Details of some applications are given in Sec. 6 of our ex](#page-9-4)tended version: [https://arxiv.org/abs/2504.07607](#page-9-4)

[et al.,](#page-9-4) [2023;](#page-9-4) [Hong,](#page-9-1) [2016\)](#page-9-1), mainly due to the new applications of functionally constrained nonconvex optimization problems in the context of neural network training [\(Katz-](#page-9-5)[Samuels et al.,](#page-9-5) [2022;](#page-9-5) [Dener et al.,](#page-9-6) [2020\)](#page-9-6). In these applications with problems involving nonconvex functional constraints, stochastic augmented Lagrangian methods (ALM) have found widespread use, whereas their behavior for even *linearly constrained* nonconvex optimization of the form [\(1\)](#page-0-0) remain poorly understood. Our focus is to improve our understanding of stochastic ALM in the context of nonconvex optimization, by focusing on the fundamental template [\(1\)](#page-0-0).

Compared to the setting of convex f, where the global complexity analysis is mature for ALM and its stochastic version [\(Yan & Xu,](#page-10-5) [2022\)](#page-10-5), nonconvexity of f poses significant difficulties in the analysis of ALM. Many works in the literature focus on penalty based algorithms (which will be formally introduced later in this section) that do not perform dual updates (or perform *negligible* dual updates that we clarify later) [\(Lu et al.,](#page-10-3) [2024;](#page-10-3) [Li et al.,](#page-9-0) [2021;](#page-9-0) [Lin et al.,](#page-10-4) [2022\)](#page-10-4), rather than primal-dual algorithms such as ALM. However, in practice, dual updates are known to be essential for accelerating convergence. Penalty methods are known to be unstable since increasing penalty parameter causes Lipschitz constant of the subproblems to increase and can lead to numerical issues. These differences in behavior between penalty and augmented Lagrangian methods are well-known, see, for example, the classical books [\(Bertsekas,](#page-9-7) [2014,](#page-9-7) Sec. 2.2.5) [\(Nocedal & Wright,](#page-10-6) [1999,](#page-10-6) Sec. 17.5).

For problem [\(1\)](#page-0-0) with access to full gradients of f and the full matrix A, optimal complexity with primal-dual methods are obtained in the work of [Zhang & Luo](#page-10-7) [\(2022\)](#page-10-7). When one has access to stochastic gradients of f and the matrix A, a recent work by [Alacaoglu & Wright](#page-9-2) [\(2024\)](#page-9-2) showed optimal complexity guarantees under expected smoothness (see Assumption [5.2\)](#page-7-0), for the special case of [\(1\)](#page-0-0) when X = R n. However, this latter restriction significantly reduces the generality of the template. For example, modeling standard quadratic programming requires X to be a half-space, which was not supported in the analysis of [Alacaoglu & Wright](#page-9-2) [\(2024\)](#page-9-2). Our goal is to go beyond these results by handling both the case when X ̸= R <sup>n</sup> as well as the case when we do not have access to the matrix A but only to an unbiased estimate of A, by keeping optimal complexity guarantees. A more detailed comparison of complexity guarantees will be made in Section [6](#page-7-1) and a summary is provided in Table [1.](#page-3-0)

Lagrangian, penalty and augmented Lagrangian. The standard approach to tackle [\(1\)](#page-0-0) is to design algorithms operating on the Lagrangian, augmented Lagrangian or penalty functions. In particular, the Lagrangian function is given as

$$L(\mathbf{x}, \mathbf{y}) = f(\mathbf{x}) + \langle A\mathbf{x} - \mathbf{b}, \mathbf{y} \rangle,$$

with the dual variables y, whereas the penalty function (or

more precisely the quadratic penalty (QP)) has the form of

$$\text{Pen}_\rho(\mathbf{x}) = f(\mathbf{x}) + \frac{\rho}{2} \|A\mathbf{x} - \mathbf{b}\|^2.$$

It is common for algorithms based on the penalty function to require ρ → ∞ for convergence [\(Bertsekas,](#page-9-7) [2014,](#page-9-7) Sec. 2.2.5). One major disadvantage of this strategy is that ρ getting larger makes the subproblem of minimizing the penalty function more and more ill-conditioned (cf. [\(4\)](#page-1-0)).

An influential idea was the introduction of the augmented Lagrangian (AL) function which combined the idea of the Lagrangian and penalty formulations [\(Hestenes,](#page-9-8) [1969;](#page-9-8) [Pow](#page-10-8)[ell,](#page-10-8) [1969\)](#page-10-8). In particular, the AL function is defined as

$$L_\rho(\mathbf{x}, \mathbf{y}) = f(\mathbf{x}) + \langle A\mathbf{x} - \mathbf{b}, \mathbf{y} \rangle + \frac{\rho}{2} \|A\mathbf{x} - \mathbf{b}\|^2.$$

Augmented Lagrangian methods in the classical literature were favoured because they did not require ρ to grow arbitrarily large. In fact, many instances of ALM converge to the optimal solution with fixed ρ since the incorporation of the dual variable updates aids in satisfying feasibility [\(Bertsekas,](#page-9-7) [2014,](#page-9-7) Prop. 2.4, Prop. 2.6).

Primal vs primal-dual algorithms. The algorithms based on the penalty function are generally referred to as *penalty algorithms* and are easier to analyze in different settings since they are primal-only algorithms, meaning that they only perform updates on primal variable x where approximate feasibility is ensured by ρ → ∞. In particular, a classical penalty method iterates for k = 1, 2, . . . as

$$\mathbf{x}_{k+1} \approx \arg \min_{\mathbf{x} \in X} f(\mathbf{x}) + \frac{\rho_k}{2} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2, \quad (4)$$

The algorithms based on the AL function are generally more difficult to analyze due to the additional dynamics coming from the dual updates which are critical to ensure that the approximate feasibility is attained with constant ρ. An ALM iteration proceeds for k = 1, 2, . . . by updating

$$\mathbf{x}_{k+1} \approx \arg \min_{\mathbf{x} \in X} f(\mathbf{x}) + \langle \mathbf{y}_k, \mathbf{Ax} - \mathbf{b} \rangle + \frac{\rho}{2} \|\mathbf{Ax} - \mathbf{b}\|^2,$$

$$\mathbf{y}_{k+1} = \mathbf{y}_k + \sigma(\mathbf{Ax}_{k+1} - \mathbf{b}).$$

For penalty methods and ALM, different strategies exist to generate xk+1 that approximately minimize the penalty or augmented Lagrangian functions by either iterating multiple steps of gradient descent (GD), known as *inexact* algorithms, or applying one step of GD, known as *linearized* algorithms [\(Ouyang et al.,](#page-10-9) [2015\)](#page-10-9).

In view of the earlier discussion, when f is nonconvex, most of the literature focuses on either analyzing penalty methods, or analyzing ALM with *negligible* dual updates and increasing penalty parameters ρ, due to the inherent

difficulty in analyzing the dual variable and its effect in convergence. In particular, as also highlighted in [\(Alacaoglu](#page-9-2) [& Wright,](#page-9-2) [2024\)](#page-9-2), many of the recent analysis of ALM is of the form of a *perturbed penalty analysis*, meaning that the feasibility is driven by increasing penalty parameters, and the dual updates are designed so that they do not deteriorate the estimates too much. Because of this, the dual step sizes are selected to be small to ensure boundedness of the dual variable (or controlling the growth of the dual variable). We refer to such updates as *negligible* dual updates since the analyses do not harness the benefit of such updates in ensuring feasibility. Feasibility is driven by large penalty parameters. Some representative examples are [\(Lu et al.,](#page-10-3) [2024\)](#page-10-3), [\(Li et al.,](#page-9-0) [2021\)](#page-9-0), [\(Lin et al.,](#page-10-4) [2022\)](#page-10-4), [\(Li et al.,](#page-9-3) [2024\)](#page-9-3).

This is the case even in the deterministic setting and the only method that we are aware that can handle true ALM with fixed penalty parameters and non-negligible dual updates are due to [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7) that uses a linearized *proximal* AL function with a dynamic adjustment on the proximal center, which will be clarified in Section [2](#page-3-1) since it will form the basis of our algorithmic development.

#### 1.1. Contributions

In this paper, we propose a stochastic smoothed linearized ALM for solving [\(1\)](#page-0-0) that only uses a single sample of stochastic gradient at every iteration. This algorithm also works with a constant penalty parameter and incorporates non-negligible dual updates for feasibility where the dual step sizes have the same order as the primal step sizes. We show that this method has its iteration complexity and sample complexity guarantees in the order of O(ε −4 ). Such a sample complexity result is optimal even in the unconstrained nonconvex case under our assumptions (see Assumption [1.1\)](#page-3-2) [\(Arjevani et al.,](#page-9-9) [2023\)](#page-9-9). In contrast, the prior results with optimal complexity required large penalty parameters, no dual updates and further assumptions [\(Lu et al.,](#page-10-3) [2024\)](#page-10-3). We then prove that this complexity can be improved to O(ε −3 ) with variance reduction when an additional expected smoothness assumption is made (see Assumption [5.2\)](#page-7-0). Under this stronger assumption, this is the optimal complexity even without constraints [\(Arjevani et al.,](#page-9-9) [2023\)](#page-9-9).

We consider extensions of this framework when we have linear constraints that hold in expectation, that is, when the constraints are given as <sup>E</sup>ξ[Aξx − bξ] = 0, with the same complexity guarantees. To our knowledge, this is the first algorithm achieving the optimal O(ε −4 ) benchmark sample complexity for nonconvex optimization with stochastic constraints using one sample per iteration, going beyond the best-known O(ε −5 ) complexity that is achieved for a more general problem that does not capture the structure of linear constraints [\(Li et al.,](#page-9-3) [2024;](#page-9-3) [Alacaoglu & Wright,](#page-9-2) [2024\)](#page-9-2).

A more detailed comparison with the related works is given

in Section [6.](#page-7-1) A summary is given in Table [1.](#page-3-0)

#### 1.2. Preliminaries

We denote the indicator function of a convex closed set X as IX(z) = 0 if x ∈ X and IX(x) = ∞ if x ̸∈ X. The notation ∂f for a convex, closed function denotes the subdifferential set and ∂IX(x) is the normal cone of X at x, by definition. For matrix A, ∥A∥ denotes its operator norm.

Given closed and convex X, projection onto X is given as

$$\text{proj}_X(\mathbf{x}) = \arg \min_{\mathbf{v} \in X} \|\mathbf{x} - \mathbf{v}\|^2.$$

Similarly, we define the proximal operator of f as

$$\text{prox}_f(\mathbf{x}) = \arg \min_{\mathbf{v}} f(\mathbf{v}) + \frac{1}{2} \|\mathbf{v} - \mathbf{x}\|^2.$$

We say that f is L-smooth when its gradient is L-Lipschitz:

$$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\|.$$

We say that f is ρ-weakly convex when f + ρ 2 ∥·∥<sup>2</sup> is convex. An L-smooth function is automatically L-weakly convex. Moreau envelope of the weakly convex f is defined as

$$\varphi_\lambda(\mathbf{z}) = \min_{\mathbf{v}} f(\mathbf{v}) + \frac{1}{2\lambda} \|\mathbf{v} - \mathbf{z}\|^2,$$

which can be interpreted as a notion of *smoothing*. Moreau envelope has many useful properties such as being smooth when f is nonsmooth and weakly convex, when λ is selected accordingly. Moreover, stationary points of f and the Moreau envelope coincide [\(Drusvyatskiy & Paquette,](#page-9-10) [2019,](#page-9-10) Lemma 4.3). The gradient of the Moreau envelope can be computed as

$$\lambda^{-1}(\mathbf{x} - \text{prox}_{\lambda\varphi}(\mathbf{x})).$$

Stationary points. A succinct way of characterizing a stationary point of [\(1\)](#page-0-0) is the following: x ⋆ is a stationary point if there exists y ⋆ such that the following hold:

$$0 \in \nabla f(\mathbf{x}^*) + A^\top \mathbf{y}^* + \partial I_X(\mathbf{x}^*) \quad \text{and} \quad 0 = A\mathbf{x}^* - \mathbf{b}.$$

One may, for example, refer to [\(Rockafellar,](#page-10-10) [2000\)](#page-10-10). Accordingly, we say that (x, y) is ε-stationary if

$$\begin{aligned}\|\mathbf{Ax} - \mathbf{b}\| &\leq \varepsilon \text{ and} \\ \|\mathbf{v}\| &\leq \varepsilon \text{ where } \mathbf{v} \in \nabla f(\mathbf{x}) + A^\top \mathbf{y} + \partial I_X(\mathbf{x}) \quad (5)\end{aligned}$$

which is a common notion used in related works, for example [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7).

We also use the following related notion of near-stationarity, as used in [\(Davis & Drusvyatskiy,](#page-9-11) [2019\)](#page-9-11). We say that x is ε-near stationary if it satisfies

$$\|\nabla\Psi(\mathbf{x})\| \leq \varepsilon, \quad (6)$$

where Ψ(x) is the Moreau envelope of the objective function f(x) +IX(x) +I{v:Av=b}(x) in [\(1\)](#page-0-0), see also [\(7\)](#page-3-3). We refer to [\(Davis & Drusvyatskiy,](#page-9-11) [2019\)](#page-9-11) for the precise notion of near stationarity.

| Reference Constraint Oracle Complexity     | Loops | Method    |
|--------------------------------------------|-------|-----------|
| (Alacaoglu & Wright, 2024) A x = b         |       |           |
| Eq. (3) and                                |       |           |
| Asmp. 5.2 O e ( ε                          |       |           |
| − 3                                        |       |           |
| )                                          | 1     | ALM       |
| (Alacaoglu & Wright, 2024)                 |       |           |
| E [ c ( x , ζ )] = 0 ,                     |       |           |
| and x ∈ X †                                |       |           |
| Eq. (3) and                                |       |           |
| Asmp. 5.2 O e ( ε                          |       |           |
| − 5                                        |       |           |
| )                                          | 1     | Penalty   |
| (Lu et al., 2024)                          |       |           |
| c ( x ) = 0 ,                              |       |           |
| and x ∈ X †                                |       |           |
| Eq. (3) and                                |       |           |
| Asmp. 5.2 O ( ε                            |       |           |
| − 3                                        |       |           |
| )                                          | 1     | Penalty   |
| (Li et al., 2024)                          |       |           |
| E [ c ( x , ζ )] = 0 ,                     |       |           |
| and x ∈ X †                                |       |           |
| Eq. (3) and                                |       |           |
| Asmp. 5.2 O ( ε                            |       |           |
| − 5                                        |       |           |
| )                                          | 2     | Penalty ∗ |
| This work A x = b ,                        |       |           |
| and x ∈ X is a polyhedral Eq. (3) O ( ε    |       |           |
| − 4                                        |       |           |
| )                                          | 1     | ALM       |
| This work E ζ [ A ( ζ ) x − b ( ζ )] = 0 , |       |           |
| and x ∈ X is a polyhedral Eq. (3) O ( ε    |       |           |
| − 4                                        |       |           |
| )                                          | 1     | ALM       |
| This work A x = b ,                        |       |           |
| and x ∈ X is a polyhedral                  |       |           |
| Eq. (3) and                                |       |           |
| Asmp. 5.2 O ( ε                            |       |           |
| − 3                                        |       |           |
| )                                          | 1     | ALM       |

Table 1. Comparison of methods. <sup>∗</sup>This method is referred to as a penalty method because the penalty parameter is taken to infinity to ensure feasibility and dual updates do not contribute in achieving feasibility. †The set X is assumed to have an efficient projection.

#### 1.3. Assumptions

We next state the assumptions that will be used throughout. These assumptions are standard and to our knowledge, the weakest, in the literature for both deterministic and stochastic nonconvex problems with linear constraints [\(Zhang &](#page-10-7) [Luo,](#page-10-7) [2022;](#page-10-7) [Alacaoglu & Wright,](#page-9-2) [2024\)](#page-9-2). A more detailed comparison of assumptions will be made in Section [6.](#page-7-1)

Assumption 1.1. For the problem [\(1\)](#page-0-0), the following holds:

- 1. The function f is L<sup>f</sup> -smooth and lower bounded over the feasible set: f(x) ≥ f > −∞ for any x ∈ X and Ax = b.
- 2. The set X admits an efficient projection and is polyhedral. That is, it has the form X = {x: Hx ≤ h} for some H, h.
- 3. We have access to stochastic gradients satisfying [\(3\)](#page-0-3).

## 2. Algorithm

We introduce Algorithm [1](#page-5-0) in this section. To gain a deeper understanding of the algorithm, we will go over two different ways of interpreting it.

Interpretation 1: Linearized proximal ALM. Algorithm [1](#page-5-0) incorporates a single-step SGD approximation of the proximal AL function. This strategy is also known as the linearized proximal ALM. In particular, the first step of the algorithm approximates the proximal AL function[<sup>2</sup>](#page-3-4) , that is,

$$\mathbf{x}_{t+1} \approx \arg \min_{\mathbf{x} \in X} L_\rho(\mathbf{x}, \mathbf{y}_{t+1}) + \frac{\mu}{2} \|\mathbf{x} - \mathbf{z}_t\|^2,$$

by a single step of projected SGD, followed by a dual variable update and updating the proximal center zt, which

takes average of z<sup>t</sup> and xt, resulting in the terminology *smoothed* that we use for the algorithm.

Interpretation 2: Inexact GD on the Moreau envelope.[<sup>3</sup>](#page-3-5) Algorithm [1](#page-5-0) can also be interpreted as an inexact gradient descent step on the Moreau envelope of the function in [\(1\)](#page-0-0). In particular, this Moreau envelope is given as

$$\Psi(\mathbf{z}_t) = \min_{\mathbf{x} \in X, \mathbf{Ax}=\mathbf{b}} \left\{ f(\mathbf{x}) + \frac{\mu}{2} \|\mathbf{x} - \mathbf{z}_t\|^2 \right\}. \quad (7)$$

By observing that minimizing the Moreau envelope helps on obtaining a near-stationary point in view of [\(6\)](#page-2-0) (cf. [\(Davis](#page-9-11) [& Drusvyatskiy,](#page-9-11) [2019\)](#page-9-11)), inexact gradient update on this function requires the computation of

$$\arg \min_{\mathbf{x} \in X, \mathbf{Ax}=\mathbf{b}} \left\{ f(\mathbf{x}) + \frac{\mu}{2} \|\mathbf{x} - \mathbf{z}_t\|^2 \right\},$$

which is a nontrivial optimization subproblem. However, it is easier than [\(1\)](#page-0-0) because the regularization provides us a *strongly convex objective* in the subproblem (given that λ is larger than L<sup>f</sup> ). As a result, we can approximate the solution of this problem by applying one iteration of ALM since this problem is a strongly convex optimization problem over linear constraints. We show that just one step of stochastic ALM is sufficient at every iteration by using a stochastic gradient computed with a single sample and one dual update, followed by the update of the proximal center zt.

On the surface, this algorithm strongly resembles that of [Zhang & Luo](#page-10-7) [\(2022\)](#page-10-7), from which we draw many ideas. However, in addition to using stochastic gradients, there is another subtle change, on the update of zt+1. Unlike [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7), we update zt+1 by using x<sup>t</sup> to be

<sup>2</sup>Note that this is also a classical function [\(Rockafellar,](#page-10-11) [1976\)](#page-10-11).

<sup>3</sup>Let us note that [Hu et al.](#page-9-12) [\(2024\)](#page-9-12) used a similar idea in a different context.

able to continue the analysis with the bounded variance assumption on G (cf. Algorithm [1\)](#page-5-0) instead of boundedness assumption on G, since the latter would require bounded domains. Thanks to this small change in this section, we handle the case with unbounded primal and dual domains.

## 3. Convergence Analysis

In this section, we first provide the main complexity results, then introduce the main analysis tools and a proof sketch.

#### 3.1. Main Theorem

In view of the two stationarity notions given in Section [1.2,](#page-2-1) we start with the result showing that Algorithm [1](#page-5-0) outputs a point at which the norm of the gradient of Moreau envelope is small, in expectation.

For the result, we state the algorithmic parameters. To avoid clutter, we write the orders of the parameters by highlighting their dependences on the problem parameters. The explicit forms of the parameters are given in [\(25\)](#page-11-0), in App. [A.](#page-11-1)

$$\tau \asymp \frac{1}{\sqrt{T}}, \quad \eta \asymp \frac{1}{\sqrt{T}}, \quad \beta \asymp \frac{1}{\sqrt{T}}, \quad (8)$$

$$\mu \asymp L_f, \quad \lambda \asymp L_f + \mu(\|A\|^2 + 1).$$

We are now ready to state the first main result.

Theorem 3.1. *Let Assumption [1.1](#page-3-2) hold and run Alg. [1](#page-5-0) with parameters from* [\(8\)](#page-4-0)*. We have that* <sup>E</sup>∥∇Ψ(z<sup>t</sup> <sup>∗</sup> )∥ ≤ ε *where* t ∗ *is selected uniformly at random from* {0, . . . , T −1} *with* T = Ω(ε −4 )*. The stochastic oracle complexity is* O(ε −4 )*.*

In particular, the above result gives us an ε-near stationary point in view of [\(Davis & Drusvyatskiy,](#page-9-11) [2019\)](#page-9-11). To get an ε-stationary point, we perform a post-processing procedure to obtain the following output from the result of Alg. [1:](#page-5-0)

$$\hat{\mathbf{x}} = \text{proj}_X(\mathbf{x}_{t^*} - \tau\hat{G}(\mathbf{x}_{t^*}, \mathbf{y}_{t^*+1}, \mathbf{z}_{t^*})), \quad (9)$$

with τ ≤ L<sup>K</sup> where L<sup>K</sup> is the Lipschitz constant of Lρ(·, y, z) + <sup>λ</sup> 2 ∥ · −x∥ 2 (cf. [\(25\)](#page-11-0)) and

$$\hat{G}(\mathbf{x}_{t^*}, \mathbf{y}_{t^*+1}, \mathbf{z}_{t^*}) = \frac{1}{B} \sum_{i=1}^B G(\mathbf{x}_{t^*}, \mathbf{y}_{t^*+1}, \mathbf{z}_{t^*}, \xi_i)$$

for ξ<sup>i</sup> i.i.d. and B = Θ(ε −2 ). This is the only place where we use a large batch size and Algorithm [1](#page-5-0) only runs with a single sample at every iteration. This post processing step is only done once and does not affect the overall complexity. The details are given in Appendix [A.3.](#page-20-0)

Corollary 3.2. *Let Assumption [1.1](#page-3-2) hold. From the output of Algorithm [1,](#page-5-0) we can obtain* xˆ *which is an* ε*-stationary point. The complexity of the whole procedure is* O(ε −4 )*.*

#### 3.2. Analysis Tools

In our analysis, Moreau envelope of two functions is critical. The first was the Moreau envelope of the composite objective in [\(1\)](#page-0-0), defined in [\(7\)](#page-3-3). We next define the Moreau envelope on the proximal AL which is the main function to analyze projected SGD, cf. [\(Davis & Drusvyatskiy,](#page-9-11) [2019\)](#page-9-11)

$$\varphi_{1/\lambda}(\mathbf{x}, \mathbf{y}, \mathbf{z}) = \min_{\mathbf{u} \in X} \left\{ L_\rho(\mathbf{u}, \mathbf{y}) + \frac{\mu}{2} \|\mathbf{u} - \mathbf{z}\|^2 + \frac{\lambda}{2} \|\mathbf{u} - \mathbf{x}\|^2 \right\}. \quad (10)$$

Another important quantity that has a significant role in the analysis is the proximal point

$$\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) = \arg \min_{\mathbf{u} \in X} L_\rho(\mathbf{u}, \mathbf{y}) + \frac{\mu}{2} \|\mathbf{u} - \mathbf{z}\|^2 + \frac{\lambda}{2} \|\mathbf{u} - \mathbf{x}\|^2. \quad (11)$$

With this, we trivially have

$$\begin{aligned} \varphi_{1/\lambda}(\mathbf{x}, \mathbf{y}, \mathbf{z}) = L_\rho(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}), \mathbf{y}) \\ + \frac{\mu}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{z}\|^2 + \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{x}\|^2. \end{aligned}$$

This is the main point of departure from [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7) where the proximal AL function is used in the analysis, in the potential function. This is because [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7) used a projected *full* GD step on the proximal AL function for which, a descent inequality follows directly. In our case, because we apply a projected SGD step, to be able to handle updates with single-sample stochastic gradients, we need to use the Moreau envelope of the proximal AL function in our potential. This analysis of projected SGD was pioneered in [\(Davis & Drusvyatskiy,](#page-9-11) [2019\)](#page-9-11).

The first result is a descent result on the Moreau envelope.

Lemma 3.3 (cf. Lemma [A.5\)](#page-13-0). *Under Assumption [1.1,](#page-3-2) for the* xt+1 *update given in Algorithm [1,](#page-5-0) we have*

$$\begin{aligned} &16\mathbb{E} \left[ \varphi_{1/\lambda}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \right] \\ &\leq 16\mathbb{E} \left[ \varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \right] \\ &\quad - \tau \lambda^2 \mathbb{E} \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 + 8\lambda\tau^2\sigma^2 \\ &\quad + 2 \left( 4\lambda\tau\mu + 16\lambda\tau^2\mu^2 + \tau\lambda^2\mu^2/\gamma_s^2 \right) \mathbb{E} \|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2, \end{aligned}$$

*where* γ<sup>s</sup> = 2µ + ρ∥A∥*.*

This follows mostly from [\(Davis & Drusvyatskiy,](#page-9-11) [2019\)](#page-9-11) and handles the transition from x<sup>t</sup> to xt+1 in our analysis. One additional error term we have here is ∥zt+1 − zt∥ 2 , due to the change in the proximal center zt, a term that was not involved in the analysis of [\(Davis & Drusvyatskiy,](#page-9-11) [2019\)](#page-9-11).

Next, we incorporate the dynamics of the updates on the dual variable y<sup>t</sup> and the proximal center zt. These results use some ideas from [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7) with additional insights. This is because [Zhang & Luo](#page-10-7) [\(2022\)](#page-10-7) use

Algorithm 1 Stochastic smoothed and linearized ALM

Initialize: x<sup>0</sup> = z<sup>0</sup> ∈ X, y<sup>0</sup> ∈ <sup>R</sup> <sup>m</sup> and ρ ≥ 0.

for t = 0 to T − 1 do

yt+1 = y<sup>t</sup> + η(Ax<sup>t</sup> − b)

Sample ξ<sup>t</sup> ∈ Ξ i.i.d. and let G(xt, yt+1, zt, ξt) = ∇f(xt, ξt) + A⊤yt+1 + ρA<sup>⊤</sup>(Ax<sup>t</sup> − b) + µ(x<sup>t</sup> − zt).

xt+1 = projX(x<sup>t</sup> − τG(xt, yt+1, zt, ξt))

zt+1 = z<sup>t</sup> + β(x<sup>t</sup> − zt)

Lρ(x, y) + <sup>λ</sup> 2 ∥x − z∥ 2 in their potential, so their analysis only characterizes the change in y and z in this function. Our analysis however, needs to characterize this change in the Moreau envelope of this function. This requires further estimations using the properties of the Moreau envelope, and the proximal point u ∗ (x, y, z) (see e.g. Lem. [A.6\)](#page-14-0).

Lemma 3.4. *(cf. Lemma [A.6\)](#page-14-0) Under Assumption [1.1,](#page-3-2) for the iterates of Alg. [1,](#page-5-0) we have*

$$\begin{aligned} & 2\mathbb{E} [\varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1})] \\ & \leq 2\mathbb{E} [\varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t)] \\ & \quad - 2\mathbb{E}\langle \mathbf{y}_{t+1} - \mathbf{y}_t, \mathbf{A}\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \mathbf{b} \rangle \\ & \quad - \mu\mathbb{E}\langle \mathbf{z}_t - \mathbf{z}_{t+1}, 2\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{z}_{t+1} - \mathbf{z}_t \rangle. \end{aligned}$$

It is easy to notice that combining the last two lemmas will give us a bound on the change of φ1/λ from t to t+1. On the other hand, the inner products appearing on the right-hand side of the last bound will require an intricate analysis after combining with the terms coming from other components in the potential function, introduced next. One aim, is to make sure we get enough slack to be able to cancel error terms coming from ∥zt+1 −zt∥ 2 in the previous lemma and further errors that will arise as we handle the inner products.

## 3.3. Proof Sketch

### 3.3.1. ONE ITERATION INEQUALITY ON THE POTENTIAL

As alluded to earlier, we introduce the potential function we work with, which incorporates the Moreau envelopes defined in [\(10\)](#page-4-1) and [\(7\)](#page-3-3):

$$V_t = \varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - 2d(\mathbf{y}_t, \mathbf{z}_t) + 2\Psi(\mathbf{z}_t),$$

where we used the new notation

$$d(\mathbf{y}, \mathbf{z}) = \min_{\mathbf{x} \in X} L_\rho(\mathbf{x}, \mathbf{y}) + \frac{\mu}{2} \|\mathbf{x} - \mathbf{z}\|^2. \quad (12)$$

There are two main changes compared to the analysis of [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7). The first is that the *primal descent* portion of our analysis investigates the behavior of the Moreau envelope of the proximal AL function (given in [\(10\)](#page-4-1)) whereas the analysis of [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7) analyzes the proximal AL function (given in [\(19\)](#page-7-2)) directly.

The reason for this departure is the well-known difficulty while analyzing SGD for constrained problems with single sample of stochastic gradients. Hence, it is not clear if it is possible to show a useful inequality with the proximal AL function in the constrained case. In particular, until the work of [\(Davis & Drusvyatskiy,](#page-9-11) [2019\)](#page-9-11), convergence analyses of projected SGD required large batches.

In addition to combining the bounds from the previous section on the change of φ1/λ, we have to characterize the change in d(y, z) and Ψ(z), for which we can use the following estimations, which only use the definition of yt+1 and hence have the same proof as the previous work.

Lemma 3.5. *[\(Zhang & Luo,](#page-10-1) [2020,](#page-10-1) Lemma 3.2, Lemma 3.3) For* d(y, z) *and* Ψ(z) *defined in* [\(7\)](#page-3-3) *and* [\(12\)](#page-5-1)*, we have*

$$\begin{aligned} & 2d(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - 2d(\mathbf{y}_t, \mathbf{z}_t) \\ & \geq 2\eta \langle \mathbf{Ax}_t - \mathbf{b}, \mathbf{Ax}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b} \rangle \\ & \quad + \mu \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_{t+1} + \mathbf{z}_t - 2\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle, \end{aligned}$$

*and*

$$\begin{aligned} \Psi(\mathbf{z}_{t+1}) - \Psi(\mathbf{z}_t) &\leq \mu \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_t - \bar{\mathbf{x}}^*(\mathbf{z}_t) \rangle \\ &\quad + \frac{\mu}{2\sigma_4} \|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2, \end{aligned}$$

*where* σ<sup>4</sup> = µ−L<sup>f</sup> µ *and*

$$\mathbf{x}^*(\mathbf{y}, \mathbf{z}) = \arg \min_{\mathbf{x} \in X} L_\rho(\mathbf{x}, \mathbf{y}) + \frac{\mu}{2} \|\mathbf{x} - \mathbf{z}\|^2, \quad (13)$$

$$\bar{\mathbf{x}}^*(\mathbf{z}) = \arg \min_{\mathbf{x} \in X, \mathbf{A}\mathbf{x} = \mathbf{b}} f(\mathbf{x}) + \frac{\mu}{2} \|\mathbf{x} - \mathbf{z}\|^2. \quad (14)$$

We continue with the main inequality on the potential function with one iteration of Alg. [1.](#page-5-0) The proof of this lemma is rather intricate and requires a careful combination of the inner products coming from the previous lemmas, and uses the particular update of the proximal center zt+1 as well as parameter selections. Recall that u ∗ (x, y, z) and x ∗ (y, z) appearing in the lemma are defined in [\(11\)](#page-4-2) and [\(13\)](#page-5-2).

Lemma 3.6 (cf. Lemma [A.9\)](#page-16-0). *With Assumption [1.1](#page-3-2) and parameters in* [\(8\)](#page-4-0) *(see* [\(25\)](#page-11-0)*), we have for Alg. [1](#page-5-0) that*

$$\begin{aligned}\mathbb{E}V_t - \mathbb{E}V_{t+1} &\geq c_\beta \mathbb{E}\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 - \lambda \tau^2 \sigma^2/2 \\ &\quad + c_\eta \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 \\ &\quad + c_\eta \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2, \quad (15)\end{aligned}$$

*where* c<sup>τ</sup> = Θ(1/ √ T)*,* c<sup>η</sup> = Θ(1/ √ T)*,* c<sup>β</sup> = Θ(1/ √ T) *with their precise definitions given in Lemma [A.9.](#page-16-0)*

One novelty in our analysis is to show that this potential function is still lower bounded and decreases, in expectation, up to an error term depends on τ 2 and the variance. To integrate this change into the framework of [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7) under reasonable assumptions on the stochastic oracle as mentioned earlier in Section [2,](#page-3-1) we also slightly changed the definition of zt+1 in the algorithm, due to technical reasons. In particular, in our case, we lose the control over ∥xt+1 − xt∥ 2 (since we do not assume bounded domains in this section), whereas the deterministic analysis of [\(Zhang](#page-10-7) [& Luo,](#page-10-7) [2022\)](#page-10-7) have a natural control over such terms.

The other change is the error coming from the variance of stochastic gradients. This causes the complexity to deteriorate compared to the deterministic case, which is an effect common with algorithms based on SGD. In particular, with a correctly selected step size, we obtain a sample complexity with the same-order as SGD, which is optimal even for unconstrained nonconvex problems [\(Arjevani et al.,](#page-9-9) [2023\)](#page-9-9).

#### 3.3.2. COMPLEXITY ANALYSIS

After Lemma [3.6,](#page-5-3) it is straightforward to obtain

$$\begin{aligned}\mathbb{E}\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 &\leq \varepsilon^2, \\ \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 &\leq \varepsilon^2, \\ \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 &\leq \varepsilon^2,\end{aligned}$$

when T = Θ(ε −4 ). Then, by tedious but straightforward calculations, we can directly get the bound on the norm of the gradient of the Moreau envelope, ∇Ψ(zt), obtaining near-stationarity. The details appear in Appendix [A.2.](#page-19-0)

A couple more steps let us go from this result to an εstationary point. The idea is simple: since we know that small ∥∇Ψ(zt)∥ means that we are near a stationary point, we can perform just *one* more iteration of SGD with batch size ≈ ε −2 to get an ε-stationary point, without changing the worst-case complexity. The details are in App. [A.3.](#page-20-0)

## 4. Extension to Random Linear Constraints

We turn to the case when constraints are sampled, that is, we do not have access to the full matrix A, or vector b but only unbiased samples of them. This is a suitable setting, when, for example, we have a large matrix A. In particular, we have A = <sup>E</sup>ζ∼<sup>P</sup> [A<sup>ζ</sup> ], b = <sup>E</sup>ζ∼<sup>P</sup> [b<sup>ζ</sup> ] and use A<sup>ζ</sup> , b<sup>ζ</sup> in the algorithm. We rewrite the template for convenience, as

$$\min_{\mathbf{x} \in X} f(\mathbf{x}) \text{ subject to } \mathbb{E}_{\zeta \sim P}[A_{\zeta} \mathbf{x} - \mathbf{b}_{\zeta}] = 0. \quad (16)$$

In this case, to get an unbiased stochastic gradient for proximal AL, we need to sample two i.i.d. samples of ζ:

$$G(\mathbf{x}, \mathbf{y}, \mathbf{z}, \xi) = \nabla f(\mathbf{x}, \xi) + A_{\zeta^1}^\top \mathbf{y} + \rho A_{\zeta^1}^\top (A_{\zeta^2} \mathbf{x} - \mathbf{b}_{\zeta^2}) + \mu(\mathbf{x} - \mathbf{z}). \quad (17)$$

An immediate issue here is that the variance of stochastic gradients of the proximal AL function scales linearly with x and y. Hence, assuming bounded variance would require assuming bounded dual variables, which is a strong assumption that is not satisfied in practice. To go around this difficulty, we have two adjustments, *(i)* we assume a constraint qualification (CQ) and compactness of X and *(ii)* we include a safeguarding procedure in the algorithm to monitor when the dual variable gets too large. Under these two modifications, we obtain the same complexity guarantees as our previous setting with deterministic constraints.

Assumption 4.1. For problem [\(16\)](#page-6-0), the following holds:

- 1. The feasible set {x : x ∈ X, Ax = b} is bounded.
- 2. The origin is in the relative interior of the set {Ax − b: x ∈ X}.
- 3. A has full row-rank.

In addition to the assumptions in the earlier setting, we require a Slater's condition as well as compact domains to ensure boundedness of the dual variable. Slater's condition is a classical CQ, see e.g., [\(Bertsekas et al.,](#page-9-13) [2003,](#page-9-13) Sec. 5.3.1). *Remark* 4.2*.* The choice of M<sup>y</sup> is given next, which admittedly can be difficult in practice. Let M<sup>V</sup> = maxx,z∈X{K(x, 0, z) − 2d(0, z) + 2Ψ(z)}, M = maxx,z∈X{|f(x)|+ µ 2 ∥x−z∥ <sup>2</sup>+ ρ 2 ∥Ax−b∥ <sup>2</sup>}, where K is defined in [\(19\)](#page-7-2) and M<sup>Ψ</sup> is a uniform lower bound of Ψ(zt), e.g., f. According to Assumption [4.1,](#page-6-1) there exists r > 0 such that for any direction d ∈ Range(A), we can find x ∈ X satisfying ∥Ax−b∥ = r and Ax−b has the same direction as d. Then, we choose M<sup>y</sup> as M<sup>y</sup> > M<sup>V</sup> −MΨ+2M r .

In this setting, we only state our theorem for nearstationarity. The ε-stationarity would follow in the same way as the previous section by a post-processing step.

Theorem 4.3. *Let Assumptions [1.1](#page-3-2) and [4.1](#page-6-1) hold and run Alg. [2](#page-7-3) with parameters from* [\(8\)](#page-4-0)*. We have that* <sup>E</sup>∥∇Ψ(z<sup>t</sup> <sup>∗</sup> )∥ ≤ ε *where* t ∗ *is randomly selected from* {0, . . . , T − 1} *with* T = Ω(ε −4 )*. The stochastic oracle complexity is* O(ε −4 )*.*

As mentioned earlier, the optimal sample complexity for nonconvex optimization with Lipschitz ∇f is O(ε −4 ) [\(Arje](#page-9-9)[vani et al.,](#page-9-9) [2023\)](#page-9-9). Our result matches this complexity while handling linear constraints with random sampling.

## 5. Extension with Variance Reduction

We now integrate the STORM variance reduction technique from [\(Cutkosky & Orabona,](#page-9-14) [2019\)](#page-9-14) into our framework to solve [\(1\)](#page-0-0) (See [arXiv:2504.07607](https://arxiv.org/abs/2504.07607) for extension to stochastic constraints). We obtain Alg. [3,](#page-7-4) which improves the iteration and oracle complexity from O(ε −4 ) to O(ε −3 ) under a stronger assumption on the oracle, compared to Sec. [3.](#page-4-3) This not only leads to an improved rate, but also to a simpler analysis that does not rely on the Moreau envelope φ1/λ.

Algorithm 2 Stochastic smoothed and linearized ALM for stochastic constraints with dual safeguarding

Input and Initialization: M<sup>y</sup> > M<sup>V</sup> −MΨ+2M r (check Remark [4.2\)](#page-6-2), x<sup>0</sup> = z<sup>0</sup> ∈ X, y<sup>0</sup> ∈ <sup>R</sup> <sup>m</sup>, ρ ≥ 0.

for t = 0 to T − 1 do

yt+1 = y<sup>t</sup> + η(A<sup>ζ</sup>tx<sup>t</sup> − bζ<sup>t</sup> ) where ζ<sup>t</sup> ∼ P is generated i.i.d.

if ∥yt+1∥ ≥ M<sup>y</sup> then

yt+1 = 0 Sample ξ<sup>t</sup> ∈ Ξ i.i.d. and generate <sup>E</sup>ξ<sup>t</sup> [G(xt, yt+1, zt, ξt)] = ∇xLρ(xt, yt+1) + µ(x<sup>t</sup> − zt) as in [\(17\)](#page-6-3)

xt+1 = projX(x<sup>t</sup> − τG(xt, yt+1, zt, ξt))

zt+1 = z<sup>t</sup> + β(x<sup>t</sup> − zt)

Algorithm 3 Stochastic smoothed and linearized ALM with STORM

Initialize: x<sup>0</sup> = z<sup>0</sup> ∈ X, y<sup>0</sup> ∈ <sup>R</sup> <sup>m</sup>, ∇b <sup>f</sup><sup>0</sup> <sup>=</sup> 1 N P<sup>N</sup> <sup>i</sup>=1 ∇f(x0, ζi), N = T 1/3 and ρ ≥ 0

for t = 0 to T − 1 do

yt+1 = y<sup>t</sup> + η(Ax<sup>t</sup> − b)

<sup>G</sup>(xt, <sup>y</sup>t+1, <sup>z</sup>t) = ∇b <sup>f</sup><sup>t</sup> <sup>+</sup> <sup>A</sup>⊤yt+1 <sup>+</sup> ρA<sup>⊤</sup>(Ax<sup>t</sup> − <sup>b</sup>) + <sup>µ</sup>(x<sup>t</sup> − <sup>z</sup>t)

xt+1 = projX(x<sup>t</sup> − τG(xt, yt+1, zt))

zt+1 = z<sup>t</sup> + β(x<sup>t</sup> − zt)

Sample <sup>ξ</sup>t+1 ∼ <sup>Ξ</sup> i.i.d. and set ∇b <sup>f</sup>t+1 <sup>=</sup> ∇f(xt+1, ξt+1) + (1 − <sup>α</sup>)(∇b <sup>f</sup><sup>t</sup> − ∇f(xt, ξt+1))

Alg. [3](#page-7-4) and Alg. [1](#page-5-0) mainly differ in the update of stochastic gradient estimate ∇b <sup>f</sup>t. If <sup>α</sup> = 0, Alg. [<sup>3</sup>](#page-7-4) trivially reduces to Alg. [1.](#page-5-0) We next see that a particular choice of α gives better complexity under Assumption [5.2](#page-7-0) (which is stronger than the oracle access and smoothness in Assumption [1.1\)](#page-3-2).

*Remark* 5.1*.* We only use a minibatch in the initialization, which does not affect the overall complexity. The minibatch size is N = T 1/3 , which is small compared to the total number of iterations T. Iterations of our algorithm only require 2 stochastic gradients, ∇f(xt, ξt+1) and ∇f(xt+1, ξt+1).

For the analysis of Alg. [3,](#page-7-4) we introduce Assumption [5.2,](#page-7-0) used, e.g., in [\(Arjevani et al.,](#page-9-9) [2023\)](#page-9-9). In particular, [Arjevani](#page-9-9) [et al.](#page-9-9) [\(2023\)](#page-9-9) showed that the oracle complexity O(ε −3 ) is tight under Assumption [5.2](#page-7-0) even with no constraints.

Assumption 5.2. We have access to a stochastic gradient of f satisfying [\(3\)](#page-0-3). For a given ξ ∼ Ξ, we can query ∇f(x, ξ) and ∇f(y, ξ) for different points x, y. Moreover, we have <sup>E</sup>ξ∼Ξ∥∇f(x, ξ) − ∇f(y, ξ)∥ <sup>2</sup> ≤ L 2 <sup>0</sup>∥x − y∥ 2 .

We introduce the potential V¯ <sup>t</sup> differing from Sec. [3](#page-4-3) and [4.](#page-6-4) This is similar to [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7), except the last term which controls the error from the variance. Define

$$\begin{aligned} \bar{V}_t &= K(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - 2d(\mathbf{y}_t, \mathbf{z}_t) + 2\Psi(\mathbf{z}_t) \\ &+ \frac{1}{48(L_0^2 + L_f^2)\tau} \|\hat{\nabla} f_t - \nabla f(\mathbf{x}_t)\|^2, \end{aligned} \quad (18)$$

where

$$K(\mathbf{x}, \mathbf{y}, \mathbf{z}) = L_\rho(\mathbf{x}, \mathbf{y}) + \frac{\mu}{2} \|\mathbf{x} - \mathbf{z}\|^2. \quad (19)$$

One-step evolution of Vˆ <sup>t</sup> that we analyze next is a key step in the analysis. Compared to [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7), we have the extra error due to using ∇b <sup>f</sup><sup>t</sup> instead of the full gradient. Lemma 5.3 (cf. Lemma [C.4\)](#page-31-0). *Under Assumptions [1.1](#page-3-2) and [5.2,](#page-7-0) with parameters*

$$\begin{aligned} \mu &= \max\{2, 4L_f\}, \quad \tau = T^{-3/2}, \\ \eta &= \Theta(\tau), \quad \beta = \Theta(\tau), \quad \alpha = \Theta(\tau^2), \end{aligned} \tag{20}$$

*(for detailed parameters, see* [\(82\)](#page-31-1)*) we have*

$$\begin{aligned}\mathbb{E}\bar{V}_t - \mathbb{E}\bar{V}_{t+1} &\geq \frac{2\mu}{\beta} \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 + \frac{1}{2\tau} \mathbb{E}\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 \\ &\quad + 2\eta \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - b\|^2 \\ &\quad + \tau \mathbb{E}\|\widehat{\nabla} f_t - \nabla f(\mathbf{x}_t)\|^2 - O(\sigma^2 \tau^3). \quad (21)\end{aligned}$$

Note that, on a high level, the main difference between Lemma [5.3](#page-7-5) and Lemma [3.6](#page-5-3) is that the order of τ in the error term is different. In Lemma [5.3,](#page-7-5) the order of τ is O(τ 3 ), while in Lemma [3.6,](#page-5-3) the order of τ is O(τ 2 ), which contribute to a faster convergence rate in for Alg. [3.](#page-7-4)

Theorem 5.4. *Let Assumptions [1.1](#page-3-2) and [5.2](#page-7-0) hold. We have that* <sup>E</sup>∥∇Ψ(z<sup>t</sup> <sup>∗</sup> )∥ ≤ ε*, where* t ∗ *is selected uniformly at random from* {0, . . . , T − 1} *with* T = Ω(ε −3 )*. The complexity of the whole procedure is* O(ε −3 )*.*

## 6. Related Works

We now compare the complexity results for obtaining an ε-stationary point, in view of Section [1.2.](#page-2-1)

Deterministic objective and deterministic constraints. The setting when objective f in [\(1\)](#page-0-0) is deterministic is the most well-studied with many results in the classical literature [\(Bertsekas,](#page-9-7) [2014\)](#page-9-7). Recent work characterized the global oracle complexity of Lagrangian-based methods or ALM. With nonlinear and nonconvex constraints, many of the existing works analyzing AL-based algorithms rely on strong CQs and boundedness assumptions and use large penalty parameters to ensure feasibility [\(Li et al.,](#page-9-0) [2021;](#page-9-0) [Lin et al.,](#page-10-4) [2022;](#page-10-4) [Kong et al.,](#page-9-15) [2019;](#page-9-15) [Kong & Monteiro,](#page-9-16) [2023;](#page-9-16) [Kong](#page-9-17) [et al.,](#page-9-17) [2023\)](#page-9-17). The existing frameworks so far fail to capture the importance of dual variable updates, which are, in fact, the main reason behind the ability to use constant penalty parameters while ensuring convergence, see e.g., [\(Bertsekas,](#page-9-7) [2014,](#page-9-7) Sec. 2.2.5). Recent works mentioned above obtained the complexity bound O(ε −3 ) for general nonlinear constraints with no specialization for linear constraints. When specialized to convex functional constraints, the best-known complexity for these methods is O(ε −2.5 ) [\(Lin et al.,](#page-10-4) [2022\)](#page-10-4).

When the constraints are linear, such as [\(1\)](#page-0-0) with X = R n, [Hong](#page-9-1) [\(2016\)](#page-9-1) analyzed ALM with constant penalty parameters and non-negligible dual updates to get optimal complexity O(ε −2 ). The case of X ̸= <sup>R</sup> <sup>n</sup> turned out to be significantly more challenging with many works focusing on variants of ALM with large penalty parameters (depending on the inverse of the final accuracy) to ensure near-feasibility and *negligible* dual updates that do not help with feasibility [\(Kong & Monteiro,](#page-9-16) [2023;](#page-9-16) [Kong et al.,](#page-9-17) [2023\)](#page-9-17) and obtained the suboptimal complexity <sup>O</sup>e(<sup>ε</sup> −2.5 ). The exceptions are the works [\(Zhang & Luo,](#page-10-1) [2020;](#page-10-1) [2022\)](#page-10-7) that showed, for the case X polyhedral, near-optimal complexity O(ε −2 ) with a constant penalty parameter and dual steps with constant step sizes, with no constraint qualification. The key step was the global error bound that our work also relied on.

Stochastic objective and deterministic constraints. One important step in generalizing the template to tasks arising in ML was to consider stochastic objectives where we access unbiased estimates. With general nonlinear constraints and Lipschitzness of ∇f, the optimal sample complexity is O(ε −4 ), obtained with double loop algorithms [\(Curtis et al.,](#page-9-18) [2024;](#page-9-18) [Boob et al.,](#page-9-4) [2023;](#page-9-4) [Ma et al.,](#page-10-12) [2020\)](#page-10-12). These works require strong assumptions on the boundedness of the primal domain as well as constraint qualifications, which are often not necessary with linear constraints.

Another set of results concerns stochastic optimization with deterministic nonlinear constraints with penalty-based algorithms. These works require large penalty parameters to ensure near-feasibility rather than dual updates [\(Lu et al.,](#page-10-3) [2024;](#page-10-3) [Alacaoglu & Wright,](#page-9-2) [2024\)](#page-9-2). They assume expected Lipschitzness as Assumption [5.2,](#page-7-0) which is stronger than Lipschitzness of ∇f. Since these works focus on nonlinear functional constraints, the analysis requires boundedness assumptions as well as constraint qualifications, unlike our results in Section [3](#page-4-3) for deterministic linear constraints.

[Alacaoglu & Wright](#page-9-2) [\(2024\)](#page-9-2) considered ALM with a constant penalty parameter and non-negligible dual updates and obtained the complexity O(ε −3 ) for linear *equality* con-

straints under Assumption [5.2.](#page-7-0) This work only covered the case X = R <sup>n</sup> and left open the question of handling the case of general X, see [\(Alacaoglu & Wright,](#page-9-2) [2024,](#page-9-2) Sec. 5). We resolve a special case of this question when X is polyhedral (covering many applications), allowing our analysis to cover linear inequality constraints. [Alacaoglu & Wright](#page-9-2) [\(2024\)](#page-9-2) used variance reduction for ∇f, which meant that they required Assumption [5.2,](#page-7-0) stronger than Assumption [1.1.](#page-3-2) In Sec. [5,](#page-6-5) we get the same complexity as this paper while allowing a polyhedral X to cover linear inequality constraints, which cannot be handled by [Alacaoglu & Wright](#page-9-2) [\(2024\)](#page-9-2).

Moreover, we also get the complexity O(ε −4 ) under Assumption [1.1.](#page-3-2) This is optimal under Assumption [1.1](#page-3-2) and we refer to [\(Arjevani et al.,](#page-9-9) [2023\)](#page-9-9) for further details on the lower bounds. In contrast, the work in [\(Alacaoglu & Wright,](#page-9-2) [2024\)](#page-9-2) does not have guarantees without Assumption [5.2.](#page-7-0)

In addition, though [\(Lu et al.,](#page-10-3) [2024\)](#page-10-3) considers the more general problem with nonconvex functional constraints, they make strong assumptions which are not easy to verify. It is not clear if their assumptions would hold with a general polyhedral constraint we have (see e.g., their Assumption 1(iv) and Eq. (7)). When the constraints are deterministic, we do not have any bounded domain assumption (our Sec. [3\)](#page-4-3) whereas the assumptions of [\(Lu et al.,](#page-10-3) [2024\)](#page-10-3) are rather difficult to be satisfied without a bounded primal domain.

[Lu et al.](#page-10-3) [\(2024\)](#page-10-3) analyzes a QP-based method, whereas we analyze an ALM-variant. ALM is known to be more stable and desirable in practice, but significantly more difficult to analyze, which is because the penalty parameter is fixed in ALM and it increases to infinity for QP. Our ALM algorithm could be extended to stochastic constraints, while [\(Lu et al.,](#page-10-3) [2024\)](#page-10-3) only handles deterministic constraints. [Alacaoglu &](#page-9-2) [Wright](#page-9-2) [\(2024\)](#page-9-2) highlights the importance of analyzing ALM compared to QP methods in their Sections 1 and 6.

Stochastic objective and stochastic constraints. This is the most general class, where the existing results come with many assumptions that are not always easy to interpret, similar to the case of stochastic objective and deterministic constraints described above [\(Li et al.,](#page-9-3) [2024;](#page-9-3) [Alacaoglu](#page-9-2) [& Wright,](#page-9-2) [2024\)](#page-9-2). The best-known complexity O(ε −5 ) is obtained by using Assumption [5.2,](#page-7-0) with an inexact, doubleloop, ALM in [\(Li et al.,](#page-9-3) [2024\)](#page-9-3) and by a single-loop QP algorithm in [\(Alacaoglu & Wright,](#page-9-2) [2024\)](#page-9-2). These results concerning ALM need to use large penalty parameters, which renders them essentially as QP-methods since the dual updates do not contribute to the analysis for ensuring the feasibility. Other approaches for solving this sub-case also require double-loop algorithms and stronger assumptions since they focus on a generic nonconvex constraint [\(Boob](#page-9-4) [et al.,](#page-9-4) [2023;](#page-9-4) [Ma et al.,](#page-10-12) [2020\)](#page-10-12), obtaining O(ε −6 ) without expected Lipschitzness. Hence, in this sub-case, none of these results harness the structure of linear constraints.

Acknowledgements Jiawei Zhang is supported by the startup fund from the Department of Computer Sciences at the University of Wisconsin–Madison and the MIT Postdoctoral Fellowship for Engineering Excellence. Ahmet Alacaoglu acknowledges the support of the Natural Sciences and Engineering Research Council of Canada (NSERC), [funding reference number RGPIN-2025-06634]. Impact Statement *This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.* References Alacaoglu, A. and Wright, S. J. Complexity of single loop algorithms for nonlinear programming with stochastic objective and constraints. In *International Conference on Artificial Intelligence and Statistics*, pp. 4627–4635. PMLR, 2024. Arjevani, Y., Carmon, Y., Duchi, J. C., Foster, D. J., Srebro, N., and Woodworth, B. Lower bounds for non-convex stochastic optimization. *Mathematical Programming*, 199 (1):165–214, 2023. Bertsekas, D. *Constrained optimization and Lagrange multiplier methods*. Academic press, 2014. Bertsekas, D., Nedic, A., and Ozdaglar, A. *Convex analysis and optimization*, volume 1. Athena Scientific, 2003. Boob, D., Deng, Q., and Lan, G. Stochastic first-order methods for convex and nonconvex functional constrained optimization. *Mathematical Programming*, 197(1):215– 279, 2023. Curtis, F. E., O'Neill, M. J., and Robinson, D. P. Worstcase complexity of an sqp method for nonlinear equality constrained stochastic optimization. *Mathematical Programming*, 205(1):431–483, 2024. Cutkosky, A. and Orabona, F. Momentum-based variance reduction in non-convex sgd. *Advances in neural information processing systems*, 32, 2019. Davis, D. and Drusvyatskiy, D. Stochastic model-based minimization of weakly convex functions. *SIAM Journal on Optimization*, 29(1):207–239, 2019. Dener, A., Miller, M. A., Churchill, R. M., Munson, T., and Chang, C.-S. Training neural networks under physical constraints using a stochastic augmented lagrangian approach. *arXiv preprint arXiv:2009.07330*, 2020. Drusvyatskiy, D. and Paquette, C. Efficiency of minimizing compositions of convex functions and smooth maps. *Mathematical Programming*, 178:503–558, 2019. Hestenes, M. R. Multiplier and gradient methods. *Journal of optimization theory and applications*, 4(5):303–320, 1969. Hiriart-Urruty, J.-B. and Lemarechal, C. *Convex Analysis and Minimization Algorithms II: Advanced Theory and Bundle Methods*, volume 306. Springer Berlin Heidelberg, Berlin, Heidelberg, 1st 1993.;1; edition, 1993. ISBN 0072-7830. Hong, M. Decomposing linearly constrained nonconvex problems by a proximal primal dual approach: Algorithms, convergence, and applications. *arXiv preprint arXiv:1604.00543*, 2016. Hu, Q., Qi, Q., Lu, Z., and Yang, T. Single-loop stochastic algorithms for difference of max-structured weakly convex functions. In *Advances in Neural Information Processing Systems*, volume 37, pp. 56738–56765, 2024. Katz-Samuels, J., Nakhleh, J. B., Nowak, R., and Li, Y. Training OOD detectors in their natural habitats. In *ICML*, 2022. Kong, W. and Monteiro, R. D. An accelerated inexact dampened augmented lagrangian method for linearlyconstrained nonconvex composite optimization problems. *Computational Optimization and Applications*, 85(2):509– 545, 2023. Kong, W., Melo, J. G., and Monteiro, R. D. Complexity of a quadratic penalty accelerated inexact proximal point method for solving linearly constrained nonconvex composite programs. *SIAM Journal on Optimization*, 29(4): 2566–2593, 2019. Kong, W., Melo, J. G., and Monteiro, R. D. Iteration complexity of an inner accelerated inexact proximal augmented lagrangian method based on the classical lagrangian function. *SIAM Journal on Optimization*, 33(1): 181–210, 2023. Lan, G. *First-order and stochastic optimization methods for machine learning*. Springer, 2020. Li, Z., Chen, P.-Y., Liu, S., Lu, S., and Xu, Y. Rate-improved inexact augmented lagrangian method for constrained nonconvex optimization. In *International Conference on Artificial Intelligence and Statistics*, pp. 2170–2178. PMLR, 2021. Li, Z., Chen, P.-Y., Liu, S., Lu, S., and Xu, Y. Stochastic inexact augmented lagrangian method for nonconvex expectation constrained optimization. *Computational Optimization and Applications*, 87(1):117–147, 2024.

Lin, Q., Ma, R., and Xu, Y. Complexity of an inexact proximal-point penalty method for constrained smooth non-convex optimization. *Computational optimization and applications*, 82(1):175–224, 2022. Lu, Z., Mei, S., and Xiao, Y. Variance-reduced first-order methods for deterministically constrained stochastic nonconvex optimization with strong convergence guarantees. *arXiv preprint arXiv:2409.09906*, 2024. Ma, R., Lin, Q., and Yang, T. Quadratically regularized subgradient methods for weakly convex optimization with weakly convex constraints. In *International Conference on Machine Learning*, pp. 6554–6564. PMLR, 2020. Nocedal, J. and Wright, S. J. *Numerical optimization*. Springer, 1999. Ouyang, Y., Chen, Y., Lan, G., and Pasiliao Jr, E. An accelerated linearized alternating direction method of multipliers. *SIAM Journal on Imaging Sciences*, 8(1): 644–681, 2015. Planiden, C. and Wang, X. Strongly convex functions, moreau envelopes, and the generic nature of convex functions with strong minimizers. *SIAM Journal on Optimization*, 26(2):1341–1364, 2016. Powell, M. J. A method for nonlinear constraints in minimization problems. *Optimization*, pp. 283–298, 1969. Rockafellar, R. T. Augmented lagrangians and applications of the proximal point algorithm in convex programming. *Mathematics of operations research*, 1(2):97–116, 1976. Rockafellar, R. T. Extended nonlinear programming. *Nonlinear optimization and related topics*, pp. 381–399, 2000. Yan, Y. and Xu, Y. Adaptive primal-dual stochastic gradient method for expectation-constrained convex stochastic programs. *Mathematical Programming Computation*, 14 (2):319–363, 2022. Zhang, J. and Luo, Z.-Q. A proximal alternating direction method of multiplier for linearly constrained nonconvex minimization. *SIAM Journal on Optimization*, 30(3): 2272–2302, 2020. Zhang, J. and Luo, Z.-Q. A global dual error bound and its application to the analysis of linearly constrained nonconvex optimization. *SIAM Journal on Optimization*, 32 (3):2319–2346, 2022. doi: 10.1137/20M135474X. URL <https://doi.org/10.1137/20M135474X>. Zhang, J., Xiao, P., Sun, R., and Luo, Z. A singleloop smoothed gradient descent-ascent algorithm for nonconvex-concave min-max problems. *Advances in neural information processing systems*, 33:7377–7389, 2020. Zhang, J., Ge, S., Chang, T.-H., and Luo, Z.-Q. Decentralized non-convex learning with linearly coupled constraints: Algorithm designs and application to vertical learning problem. *IEEE Transactions on Signal Processing*, 70:3312–3327, 2022.

## Notation.

Let us note that we define by <sup>E</sup><sup>t</sup> the expectation conditioned on all the randomness up to and including xt.

## A. Proofs for Section [3](#page-4-3)

In the proofs, let us recall

$$\begin{aligned} K(\mathbf{x}, \mathbf{y}, \mathbf{z}) &= L_\rho(\mathbf{x}, \mathbf{y}) + \frac{\mu}{2} \|\mathbf{x} - \mathbf{z}\|^2 \\ &= f(\mathbf{x}) + \langle A\mathbf{x} - \mathbf{b}, \mathbf{y} \rangle + \frac{\rho}{2} \|A\mathbf{x} - \mathbf{b}\|^2 + \frac{\mu}{2} \|\mathbf{x} - \mathbf{z}\|^2. \end{aligned} \quad (22)$$

With this notation, we have the following, equivalent to [\(11\)](#page-4-2):

$$\begin{aligned} \mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) &= \arg \min_{\mathbf{u} \in X} \left\{ K(\mathbf{u}, \mathbf{y}, \mathbf{z}) + \frac{\lambda}{2} \|\mathbf{u} - \mathbf{x}\|^2 \right\} \\ &= \arg \min_{\mathbf{u} \in X} \left\{ L_\rho(\mathbf{u}, \mathbf{y}, \mathbf{z}) + \frac{\mu}{2} \|\mathbf{u} - \mathbf{z}\|^2 + \frac{\lambda}{2} \|\mathbf{u} - \mathbf{x}\|^2 \right\}. \end{aligned} \quad (23)$$

We also recall [\(10\)](#page-4-1).

$$\begin{aligned}\varphi_{1/\lambda}(\mathbf{x}, \mathbf{y}, \mathbf{z}) &= \min_{\mathbf{u} \in X} \left\{ L_\rho(\mathbf{u}, \mathbf{y}) + \frac{\mu}{2} \|\mathbf{u} - \mathbf{z}\|^2 + \frac{\lambda}{2} \|\mathbf{u} - \mathbf{x}\|^2 \right\} \\ &= \min_{\mathbf{u} \in X} \left\{ K(\mathbf{u}, \mathbf{y}, \mathbf{z}) + \frac{\lambda}{2} \|\mathbf{u} - \mathbf{x}\|^2 \right\}.\end{aligned}\tag{24}$$

We also introduce here some parameters that are used throughout, for convenience.

$$\begin{aligned} \mu &= \max\{2, 4L_f\}, \\ L_K &= L_f + \rho\|A\| + \mu, \\ \lambda &= 2L_K, \\ \sigma_4 &= \frac{\mu - L_f}{\mu}, \\ \tau &= \frac{1}{6\lambda^2\sqrt{T}}, \\ \eta &= \min \left\{ \frac{2\mu + \rho\|A\|}{4\|A\|^4}, \frac{\tau}{200\|A\|^2}, \frac{\tau(2\mu + \rho\|A\|^2)}{20\|A\|^2} \right\}, \\ \beta &= \min \left\{ \frac{\tau}{100}, \frac{1}{50\lambda}, \frac{\eta}{36\mu\bar{\sigma}^2} \right\}, \\ \gamma_s &= 2\mu + \rho\|A\|, \gamma = \frac{(\mu - L_f)\lambda}{\mu - L_f + \lambda}, \gamma_K = \mu - L_f. \end{aligned} \tag{25}$$

We also mention the following basic facts that are used in the sequel.

Fact A.1. *For* x ∈ X*, we have that* x 7→ K(x, y, z) *is strongly convex with modulus* γ<sup>K</sup> := µ−L<sup>f</sup> *, and* x 7→ ∇xK(x, y, z) *is* L<sup>K</sup> := (L<sup>f</sup> + ρ∥A∥ <sup>2</sup> + µ)*-Lipschitz continuous.*

*For* u ∈ X*,* u 7→ K(u, y, z) + <sup>λ</sup> ∥x − u∥ 2 *is strongly convex with modulus* γ<sup>s</sup> = µ − L<sup>f</sup> + λ*, and* u ∗ (x, y, z) = arg minu∈<sup>X</sup> K(u, y, z) + <sup>λ</sup> 2 ∥x − u∥ 2 *.*

Lemma A.2. *[\(Planiden & Wang,](#page-10-13) [2016,](#page-10-13) Lemma 2.19) Let* r > 0*. The function f is r-strongly convex if and only if* f1(x) = min<sup>u</sup> f(u) + <sup>1</sup> 2 ∥x − u∥ *is* <sup>r</sup> <sup>r</sup>+1 *-strongly convex.*

Lemma A.3. *The function* x 7→ φ1/λ(x, y, z) *is* γ = (µ−L<sup>f</sup> )λ µ−L<sup>f</sup> +λ *-strongly convex.*

*Proof.* By definition, we have

$$\varphi_{1/\lambda}(\mathbf{x}, \mathbf{y}, \mathbf{z}) = \min_{\mathbf{u}} K(\mathbf{u}, \mathbf{y}, \mathbf{z}) + I_X(\mathbf{u}) + \frac{\lambda}{2} \|\mathbf{x} - \mathbf{u}\|^2 = \lambda \min_{\mathbf{u}} \frac{K(\mathbf{u}, \mathbf{y}, \mathbf{z}) + I_X(\mathbf{u})}{\lambda} + \frac{1}{2} \|\mathbf{x} - \mathbf{u}\|^2.$$

Recall that γ<sup>K</sup> = µ − L<sup>f</sup> . Then, since K(x, y, z)/λ is <sup>γ</sup><sup>K</sup> λ -strongly convex, we have min<sup>u</sup> K(u,y,z)+IX(u) <sup>λ</sup> + 1 2 ∥x − u∥ 2 is <sup>γ</sup>K/λ <sup>γ</sup>K/λ+1 -strongly convex, by Lemma [A.2.](#page-11-2) Hence, <sup>φ</sup>1/λ(x, <sup>y</sup>, <sup>z</sup>) is strongly convex with modulus <sup>γ</sup><sup>K</sup> <sup>γ</sup>K/λ+1 = λγ<sup>K</sup> λ+γ<sup>K</sup> = (µ−L<sup>f</sup> )λ µ−L<sup>f</sup> +λ . ■

#### A.1. Proofs for Lemma [3.6](#page-5-3)

In the next lemma, the first part is using the idea of [Davis & Drusvyatskiy](#page-9-11) [\(2019\)](#page-9-11) to analyze the algorithm under the bounded variance assumption instead of the restrictive bounded stochastic gradient assumption. The second part of the lemma also follows a similar idea as this work, with the exception of the dependence on the changing center point zt. This introduces additional issues, since the stochastic gradient in the update of xt+1 depends on z<sup>t</sup> whereas the proximal point u ∗ (xt, yt+1, zt+1) (that characterizes the iteration below) depends on zt+1. Our analysis below estimates this additional error and shows it to be in the order of ∥zt+1 − zt∥ 2 , which will be handled later.

Lemma A.4. *Suppose that Assumption [1.1](#page-3-2) holds, for the proximal point* u ∗ (xt, yt+1, zt+1)*, defined as* [\(11\)](#page-4-2) *we have the characterization*

$$\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) = \text{proj}_X(\tau\lambda \mathbf{x}_t + (1 - \tau\lambda)\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \tau\nabla_{\mathbf{x}}K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1})). \quad (26)$$

(xt, yt+1, zt+1) = projX(τλx<sup>t</sup> + (1 − τλ)u *Moreover, for the sequence* xt+1 *calculated as Algorithm [1,](#page-5-0) with* λ = 2L<sup>K</sup> *and* τ ≤ 1 6λ *, where* L<sup>K</sup> = L<sup>f</sup> + ρ∥A∥ <sup>2</sup> + µ*, we have*

$$\mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_{t+1}\|^2 \leq \left(1 - \frac{\tau\lambda}{4}\right) \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_t\|^2 + (\tau\mu + 2\tau^2\mu^2)\mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 + \tau^2\sigma^2.$$

*Proof.* From the definition of u ∗ (xt, yt+1, zt+1) in [\(11\)](#page-4-2) (see also [\(23\)](#page-11-3)), we have

$$\lambda(\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1})) \in \nabla_{\mathbf{x}} K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + \partial I_X(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1})).$$

Multiplying both sides by the step size τ , adding u ∗ (xt, yt+1, zt+1) to both sides, and rearranging give

$$\begin{aligned} & \tau \lambda \mathbf{x}_t - \tau \nabla_{\mathbf{x}} K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + (1 - \tau \lambda) \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \\ & \in \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + \tau \partial I_X(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1})). \end{aligned}$$

Since (I + τ ∂IX) <sup>−</sup><sup>1</sup> = prox<sup>I</sup><sup>X</sup> = proj<sup>X</sup> due to ∂I<sup>X</sup> being a cone and proximal operator of a normal cone being the projection to the set, we have the first assertion.

We next establish the second assertion. Using the just established identity [\(26\)](#page-12-0), the update rule of xt+1 in Algorithm [1,](#page-5-0) and nonexpansiveness of the projection, we derive

$$\begin{aligned} & \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_{t+1}\|^2 \\ & \leq \|\tau\lambda\mathbf{x}_t + (1 - \tau\lambda)\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \tau\nabla_{\mathbf{x}}K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - [\mathbf{x}_t - \tau G(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t, \xi_t)]\|^2. \end{aligned}$$

We add and subtract ∇xK(xt, yt+1, zt) inside the squared norm on the right-hand side, expand and take conditional expectation to obtain

$$\begin{aligned} \mathbb{E}_t \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_{t+1}\|^2 \\ = \|(1 - \tau\lambda)(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_t) - \tau\nabla_{\mathbf{x}}K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + \tau\nabla_{\mathbf{x}}K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \\ + \tau^2\mathbb{E}_t \|G(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t, \xi_t) - \nabla_{\mathbf{x}}K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2, \end{aligned} \quad (27)$$

where the cross term disappeared because

$$\mathbb{E}_t[G(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t, \xi_t)] = \nabla_{\mathbf{x}} K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)$$

and xt, yt+1, zt+1, u ∗ (xt, yt+1, zt+1) are deterministic under the conditioning since zt+1 defined in Algorithm [1](#page-5-0) only depends on x<sup>t</sup> (that is, zt+1 is independent of ξt).

The second term on the right-hand side of [\(27\)](#page-12-1) is trivially bounded by the oracle assumptions, that is,

$$\mathbb{E}_{\mathbf{x}} \|G(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t, \xi_t) - \nabla_{\mathbf{x}} K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \leq \sigma^2. \quad (28)$$

For the first term on the right-hand side of [\(27\)](#page-12-1), we further estimate as

$$\begin{aligned} & \|(1 - \tau\lambda)(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_t) - \tau\nabla_{\mathbf{x}}K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + \tau\nabla_{\mathbf{x}}K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \\ &= (1 - \tau\lambda)^2 \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_t\|^2 \\ &\quad + 2\tau(1 - \tau\lambda)\langle \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_t, \nabla_{\mathbf{x}}K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \nabla_{\mathbf{x}}K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle \\ &\quad + \tau^2 \|\nabla_{\mathbf{x}}K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \nabla_{\mathbf{x}}K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1})\|^2. \end{aligned} \quad (29)$$

Next, we turn to estimating

$$\begin{aligned} & \|\nabla_{\mathbf{x}} K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \nabla_{\mathbf{x}} K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1})\| \\ & \leq \|\nabla_{\mathbf{x}} K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \nabla_{\mathbf{x}} K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1})\| \\ & \quad + \|\nabla_{\mathbf{x}} K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \nabla_{\mathbf{x}} K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1})\|. \end{aligned} \quad (30)$$

Note that, by definition, we have

$$\nabla_{\mathbf{x}} K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \nabla_{\mathbf{x}} K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) = \mu(\mathbf{z}_{t+1} - \mathbf{z}_t).$$

Using this and the LK-Lipschitzness of ∇xK(·, yt+1, zt+1) as per Fact [A.1,](#page-11-4) in [\(30\)](#page-13-1), we obtain

$$\|\nabla_{\mathbf{x}}K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \nabla_{\mathbf{x}}K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1})\| \leq \mu\|\mathbf{z}_{t+1} - \mathbf{z}_t\| + L_K\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_t\|.$$

We plug this bound into the second term on the right-hand side of [\(29\)](#page-13-2) after using Cauchy-Schwarz inequality, and then, we use Young's inequality to get

$$\begin{aligned} & 2\tau(1 - \tau\lambda)\langle \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_t, \nabla_{\mathbf{x}}K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \nabla_{\mathbf{x}}K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle \\ & \leq 2\tau(1 - \tau\lambda)\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_t\|(\mu\|\mathbf{z}_{t+1} - \mathbf{z}_t\| + LK\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_t\|) \\ & \leq \tau(1 - \tau\lambda)(2LK + \mu)\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_t\|^2 + \tau(1 - \tau\lambda)\mu\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2. \end{aligned}$$

Using the last two inequalities in [\(29\)](#page-13-2), along with Young's inequality, we obtain

$$\begin{aligned} & \|(1 - \tau\lambda)(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_t) - \tau\nabla_{\mathbf{x}}K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + \tau\nabla_{\mathbf{x}}K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \\ & \leq [(1 - \tau\lambda)^2 + \tau(1 - \tau\lambda)(2L_K + \mu) + 2\tau^2L_K^2]\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_t\|^2 \\ & \quad + (\tau(1 - \tau\lambda)\mu + 2\tau^2\mu^2)\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2. \end{aligned} \quad (31)$$

We estimate the coefficient of the first term. First, note that 1 − τλ ≤ 1. As a result, we have

$$\begin{aligned} (1 - \tau\lambda)^2 + \tau(1 - \tau\lambda)(2L_K + \mu) + 2\tau^2L_K^2 &\leq 1 - 2\tau\lambda + \tau^2\lambda^2 + \tau(2L_K + \mu) + 2\tau^2L_K^2 \\ &\leq 1 - 2\tau\lambda + \frac{1}{6}\tau\lambda + \tau\lambda + \frac{1}{2}\tau\lambda + \frac{1}{12}\tau\lambda \\ &= 1 - \frac{\tau\lambda}{4}, \end{aligned}$$

where in second inequality, we use τλ ≤ 6 ,L<sup>K</sup> = λ and τµ ≤ τL<sup>K</sup> = 1 2 τλ.

Finally, since τ (1 − τλ)µ + 2τ 2µ <sup>2</sup> ≤ τµ + 2τ 2µ 2 , the proof is completed after taking full expectation of [\(27\)](#page-12-1) and plugging in [\(28\)](#page-12-2) and [\(31\)](#page-13-3). ■

Lemma A.5 (cf. Lemma [3.3\)](#page-4-4). *Let Assumption [1.1](#page-3-2) hold. Then, if* λ = 2L<sup>K</sup> *and* τ ≤ 6λ *, we have for the iterates of Algorithm [1](#page-5-0) that*

$$\mathbb{E}\varphi_{1/\lambda}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \leq \mathbb{E}\varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \frac{\tau\lambda^2}{16}\mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 + \left( \frac{\lambda\tau\mu}{2} + \lambda\tau^2\mu^2 + \frac{\tau\lambda^2\mu^2}{8\gamma_s^2} \right) \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 + \frac{\lambda\tau^2\sigma^2}{2}, \quad (32)$$

*Proof.* By the definition of φ1/λ from [\(24\)](#page-11-5) and u ∗ (x, yt+1, zt+1) from [\(23\)](#page-11-3), we have

$$\begin{aligned}\mathbb{E}\varphi_{1/\lambda}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) &\leq \mathbb{E}K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + \frac{\lambda}{2}\mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_{t+1}\|^2 \\ &\leq \mathbb{E}K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + \left(\frac{\lambda}{2} - \frac{\tau\lambda^2}{8}\right)\mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_t\|^2 \\ &\quad + \left(\frac{\lambda\tau\mu}{2} + \lambda\tau^2\mu^2\right)\mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 + \frac{\lambda\tau^2\sigma^2}{2} \\ &= \mathbb{E}\varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \frac{\tau\lambda^2}{8}\mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_t\|^2 \\ &\quad + \left(\frac{\lambda\tau\mu}{2} + \lambda\tau^2\mu^2\right)\mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 + \frac{\lambda\tau^2\sigma^2}{2}.\end{aligned}\tag{33}$$

We next bound the second term on the right-hand side by using Young's inequality as

$$\begin{aligned} \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{x}_t\|^2 &\geq \frac{1}{2} \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 - \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \\ &\geq \frac{1}{2} \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 - \frac{\mu^2}{\gamma_s^2} \|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2, \end{aligned} \quad (34)$$

where the last line used [\(61\)](#page-22-0).

We substitute the last inequality into [\(33\)](#page-14-1) to conclude. ■

Since the previous result only allowed us to connect φ1/λ(xt+1, yt+1, zt+1) to φ1/λ(xt, yt+1, zt+1), we now need to analyze the effect of changing yt+1 and zt+1 in φ1/λ. The main idea of this lemma is similar to [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7), where the difference lies in the fact that our potential involves the Moreau envelope of K(x, y, z) whereas the potential of [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7) involves K(x, y, z). Hence this work considers the change of the arguments in the function K instead of φ1/λ. Therefore, our proof uses the properties of the Moreau envelope which was not needed in [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7).

Lemma A.6. *(cf. Lemma [3.4\)](#page-5-4) Suppose that Assumption [1.1](#page-3-2) holds, for* φ1/λ *defined in* [\(10\)](#page-4-1)*, we have for the iterates of Algorithm [1](#page-5-0) that*

$$\begin{aligned}\varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) &\geq \langle \mathbf{y}_t - \mathbf{y}_{t+1}, \mathbf{A}\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \mathbf{b} \rangle \\ &\quad + \frac{\gamma_s}{2} \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2, \\ \varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) &\geq \frac{\mu}{2} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, 2\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{z}_{t+1} - \mathbf{z}_t \rangle \\ &\quad + \frac{\gamma_s}{2} \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2,\end{aligned}$$

*where* γ<sup>s</sup> = 2µ + ρ∥A∥*.*

*Proof.* We first consider the change in y argument of φ1/λ. By using the definition of φ1/λ in [\(24\)](#page-11-5), we have

$$\begin{aligned}\varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) &= K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t), \mathbf{y}_t, \mathbf{z}_t) + \frac{\lambda}{2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t)\|^2 \\ &\quad - K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{y}_{t+1}, \mathbf{z}_t) - \frac{\lambda}{2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \\ &= K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t), \mathbf{y}_t, \mathbf{z}_t) - K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t), \mathbf{y}_{t+1}, \mathbf{z}_t) \\ &\quad + K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t), \mathbf{y}_{t+1}, \mathbf{z}_t) + \frac{\lambda}{2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t)\|^2 \\ &\quad - K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{y}_{t+1}, \mathbf{z}_t) - \frac{\lambda}{2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2, \quad (35)\end{aligned}$$

From the definition of K in [\(22\)](#page-11-6), it trivially follows that

$$K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t), \mathbf{y}_t, \mathbf{z}_t) - K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t), \mathbf{y}_{t+1}, \mathbf{z}_t) = \langle \mathbf{y}_t - \mathbf{y}_{t+1}, A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \mathbf{b} \rangle.$$

Next, we use the property that K(·, yt+1, zt) + <sup>λ</sup> ∥ · −xt∥ 2 is γs-strongly convex with minimizer u ∗ (xt, yt+1, zt) (see Fact [A.1](#page-11-4) and [\(23\)](#page-11-3)) to obtain

$$\begin{aligned} K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t), \mathbf{y}_{t+1}, \mathbf{z}_t) + \frac{\lambda}{2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t)\|^2 - K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{y}_{t+1}, \mathbf{z}_t) - \frac{\lambda}{2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \\ \geq \frac{\gamma_s}{2} \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2. \end{aligned}$$

Combining the last two estimates in [\(35\)](#page-14-2) gives the first assertion.

Next, we analyze the effect of changing the z component in φ1/λ. Similar to the proof of the first assertion, we start with the definition of φ1/λ and then add and subtract K(u ∗ (xt, yt+1, zt+1) to obtain

$$\begin{aligned} & \varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \\ &= K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{y}_{t+1}, \mathbf{z}_t) + \frac{\lambda}{2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \\ &\quad - K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \frac{\lambda}{2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1})\|^2 \\ &= K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{y}_{t+1}, \mathbf{z}_t) - K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \\ &\quad + K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + \frac{\lambda}{2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \\ &\quad - K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \frac{\lambda}{2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1})\|^2. \end{aligned} \quad (36)$$

First, by definition, of K, it trivially follows that

$$K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{y}_{t+1}, \mathbf{z}_t) - K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) = \frac{\mu}{2} \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{z}_t\|^2 \\ - \frac{\mu}{2} \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{z}_{t+1}\|^2.$$

For the remaining terms on the right-hand side, we again use that K(·, yt+1, zt+1) + <sup>λ</sup> 2 ∥ · −xt∥ 2 is γs-strongly convex with minimizer u ∗ (xt, yt+1, zt+1) to deduce

$$\begin{aligned} & K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + \frac{\lambda}{2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \\ & - K(\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}), \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \frac{\lambda}{2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1})\|^2 \\ & \geq \frac{\gamma_s}{2} \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2. \end{aligned}$$

Plugging in the last two estimates in [\(36\)](#page-15-0) gives the second assertion. ■

Corollary A.7. *Suppose that Assumption [1.1](#page-3-2) holds, for* φ1/λ *defined in* [\(10\)](#page-4-1)*, if* λ = 2L<sup>K</sup> *and* τ ≤ 6λ *, we have that*

$$\begin{aligned}\mathbb{E}\varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \mathbb{E}\varphi_{1/\lambda}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) &\geq \frac{\tau\lambda^2}{16}\mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 \\ &\quad - \left( \frac{\lambda\tau\mu}{2} + \lambda\tau^2\mu^2 + \frac{\tau\lambda^2\mu^2}{8\gamma_s^2} \right) \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 - \frac{\lambda\tau^2\sigma^2}{2} \\ &\quad - \eta\mathbb{E}\langle A\mathbf{x}_t - \mathbf{b}, A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \mathbf{b} \rangle \\ &\quad + \frac{\mu}{2}\mathbb{E}\langle \mathbf{z}_{t+1} - \mathbf{z}_t, 2\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{z}_{t+1} - \mathbf{z}_t \rangle,\end{aligned}$$

*Proof.* We sum up the results in Lemma [A.5](#page-13-0) and Lemma [A.6,](#page-14-0) plug in the definition of yt+1 and discard two nonnegative terms on the right-hand side to get the result. ■

Next, we analyze the rest of the terms appearing in the potential function. This lemma is only using the definition of d(y, z) and Ψ(z) and is equivalent to [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7) and hence we omit its proof. Notably, these bounds are agnostic to the algorithm used to generate the sequences. Note that the only difference is that in the result below, we do not use the definition of yt+1 whereas the proof in [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7) uses this definition. The rest of the estimations are precisely the same.

Lemma A.8. *[\(Zhang & Luo,](#page-10-1) [2020,](#page-10-1) Lemma 3.2, Lemma 3.3) For the functions* d(y, z) *and* Ψ(z) *defined in* [\(12\)](#page-5-1) *and* [\(7\)](#page-3-3)*,we have*

$$\begin{aligned} d(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - d(\mathbf{y}_t, \mathbf{z}_t) &\geq \langle \mathbf{Ax}_t - \mathbf{b}, \mathbf{Ax}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b} \rangle + \frac{\mu}{2} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_{t+1} + \mathbf{z}_t - 2\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle, \\ \Psi(\mathbf{z}_{t+1}) - \Psi(\mathbf{z}_t) &\leq \mu \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_t - \bar{\mathbf{x}}^*(\mathbf{z}_t) \rangle + \frac{\mu}{2\sigma_4} \|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2, \end{aligned}$$

*where* σ<sup>4</sup> *is defined in* [\(25\)](#page-11-0)*.*

In the next lemma, we will join the previous lemmas and characterize the change in the potential function.

Lemma A.9 (cf. Lemma [3.6\)](#page-5-3). *Let Assumption [1.1](#page-3-2) hold. By using the parameters* [\(25\)](#page-11-0) *in Algorithm [1,](#page-5-0) we obtain*

$$\mathbb{E}V_t - \mathbb{E}V_{t+1} \geq c_\beta \mathbb{E}\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 + c_\tau \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 + c_\eta \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 - \frac{1}{2}\lambda\tau^2\sigma^2, \quad (37)$$

where 
$$c_\beta = \frac{\mu}{50\beta}$$
,  $c_\tau = \frac{7\tau\lambda^2}{400}$ ,  $c_\eta = \frac{\eta}{4}$ .

*Proof.* Combining Corollary [A.7](#page-15-1) and Lemma [A.8,](#page-16-1) we obtain

$$\begin{aligned} \mathbb{E}[V_t - V_{t+1}] &= \mathbb{E} [\varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \varphi_{1/\lambda}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + 2d(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - 2d(\mathbf{y}_t, \mathbf{z}_t) + 2\Psi(\mathbf{z}_t) - 2\Psi(\mathbf{z}_{t+1})] \\ &\geq \frac{\tau\lambda^2}{16} \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 - \left( \frac{\tau\mu}{2} + \lambda\tau^2\mu^2 + \frac{\tau\lambda^2\mu^2}{8\gamma_s^2} \right) \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 - \frac{\lambda\tau^2\sigma^2}{2} \\ &\quad - \eta\mathbb{E}\langle A\mathbf{x}_t - \mathbf{b}, A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \mathbf{b} \rangle + \frac{\mu}{2}\mathbb{E}\langle \mathbf{z}_{t+1} - \mathbf{z}_t, 2\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{z}_t - \mathbf{z}_{t+1} \rangle \\ &\quad + 2\eta\mathbb{E}\langle A\mathbf{x}_t - \mathbf{b}, A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b} \rangle + \mu\mathbb{E}\langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_{t+1} + \mathbf{z}_t - 2\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle \\ &\quad - 2\mu\mathbb{E}\langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_t - \bar{\mathbf{x}}^*(\mathbf{z}_t) \rangle - \frac{\mu}{\sigma_4} \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2. \end{aligned} \quad (38)$$

We next manipulate the terms on the right-hand side. First, by adding and subtracting Ax<sup>t</sup> on the second argument of the first inner product on the right-hand side, we get

$$-\eta \langle A\mathbf{x}_t - \mathbf{b}, A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \mathbf{b} \rangle = -\eta \|A\mathbf{x}_t - \mathbf{b}\|^2 - \eta \langle A\mathbf{x}_t - \mathbf{b}, A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - A\mathbf{x}_t \rangle.$$

Consequently, we use this estimate and rewrite the third inner product on the right-hand side of [\(38\)](#page-16-2) with quadratics to have

$$\begin{aligned} & -\eta \langle A\mathbf{x}_t - \mathbf{b}, A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \mathbf{b} \rangle + 2\eta \langle A\mathbf{x}_t - \mathbf{b}, A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b} \rangle \\ & = -\eta \|A\mathbf{x}_t - A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 + \eta \|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 - \eta \langle A\mathbf{x}_t - \mathbf{b}, A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - A\mathbf{x}_t \rangle. \end{aligned}$$

Second, adding and subtracting 2x<sup>t</sup> in the second argument of the second inner product on the right-hand side of [\(38\)](#page-16-2) gives

$$\frac{\mu}{2} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, 2\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{z}_t - \mathbf{z}_{t+1} \rangle = \frac{\mu}{2} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, 2\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - 2\mathbf{x}_t \rangle + \frac{\mu}{2} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, 2\mathbf{x}_t - \mathbf{z}_t - \mathbf{z}_{t+1} \rangle.$$

For the right-hand side of this term, note that zt+1 = z<sup>t</sup> + β(x<sup>t</sup> − zt) ⇐⇒ 2x<sup>t</sup> − 2z<sup>t</sup> = 2 β (zt+1 − zt) and hence

$$\begin{aligned} \frac{\mu}{2} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, 2\mathbf{x}_t - \mathbf{z}_t - \mathbf{z}_{t+1} \rangle &= \frac{\mu}{2} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, 2\mathbf{x}_t - 2\mathbf{z}_t \rangle + \frac{\mu}{2} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_t - \mathbf{z}_{t+1} \rangle \\ &= \frac{\mu}{2} \left( \frac{2}{\beta} - 1 \right) \|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 \geq \frac{\mu}{2\beta} \|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2, \end{aligned}$$

where the last inequality is due to β ≤ 1.

Next, for the remaining inner products in [\(38\)](#page-16-2), we have

$$\begin{aligned} & \mu \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_{t+1} + \mathbf{z}_t - 2\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle - 2\mu \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_t - \bar{\mathbf{x}}^*(\mathbf{z}_t) \rangle \\ & = \mu \|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 + 2\mu \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \bar{\mathbf{x}}^*(\mathbf{z}_t) - \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle. \end{aligned} \quad (39)$$

We can use Cauchy-Schwarz, triangle and Young's inequalities on the second term here to get

$$\begin{aligned} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \bar{\mathbf{x}}^*(\mathbf{z}_t) - \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle &\geq -\|\mathbf{z}_{t+1} - \mathbf{z}_t\|(\|\bar{\mathbf{x}}^*(\mathbf{z}_t) - \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\| + \|\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1})\|) \\ &\geq -\left(\frac{1}{2\zeta} + \frac{1}{\sigma_4}\right) \|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 - \frac{\zeta}{2} \|\bar{\mathbf{x}}^*(\mathbf{z}_t) - \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\|^2, \end{aligned}$$

where the last step also used [\(63\)](#page-22-1). Consequently, plugging in this estimate to [\(39\)](#page-17-0), we obtain

$$\begin{aligned} & \mu \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_{t+1} + \mathbf{z}_t - 2\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle - 2\mu \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_t - \bar{\mathbf{x}}^*(\mathbf{z}_t) \rangle \\ & \geq \left( \mu - \frac{\mu}{\zeta} - \frac{2\mu}{\sigma_4} \right) \|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 - \mu \zeta \|\bar{\mathbf{x}}^*(\mathbf{z}_t) - \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\|^2. \end{aligned}$$

After combining these estimates in [\(38\)](#page-16-2), we get

$$\begin{aligned} & \mathbb{E}[V_t] - \mathbb{E}[V_{t+1}] \\ & \geq \frac{\tau\lambda^2}{16} \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 - \left( \frac{1}{2} \lambda \tau \mu + \lambda \tau^2 \mu^2 + \frac{\tau\lambda^2 \mu^2}{8\gamma_s^2} + \frac{\mu}{\zeta} + \frac{3\mu}{\sigma_4} - \mu - \frac{\mu}{2\beta} \right) \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 - \frac{1}{2} \lambda \tau^2 \sigma^2 \\ & \quad - \eta \mathbb{E}\langle A\mathbf{x}_t - \mathbf{b}, A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \mathbf{Ax}_t \rangle - \eta \mathbb{E}\|A\mathbf{x}_t - A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 + \eta \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 \\ & \quad - \mu \zeta \mathbb{E}\|\bar{\mathbf{x}}^*(\mathbf{z}_t) - \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 + \mu \mathbb{E}\langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t \rangle. \end{aligned} \quad (40)$$

We will now operate on some of terms from the right-hand side of [\(40\)](#page-17-1), by using Lemma [A.11](#page-23-0) and [A.12.](#page-24-0) First, we have by Cauchy-Schwarz and Young's inequalities that

$$\begin{aligned}& -\eta \langle A\mathbf{x}_t - \mathbf{b}, A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - A\mathbf{x}_t \rangle \\& \geq -\frac{\eta}{4} \|A\mathbf{x}_t - \mathbf{b}\|^2 - \eta \|A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - A\mathbf{x}_t\|^2 \\& \geq -\frac{\eta}{4} \|A\mathbf{x}_t - \mathbf{b}\|^2 - 2\eta \|A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 - 2\eta \|A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - A\mathbf{x}_t\|^2.\end{aligned}$$

Next, by using the Lipschitzness of u ∗ (xt, ·, zt) from [\(60\)](#page-22-2), we have

$$\begin{aligned} \|A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 &\leq \|A\|^2 \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \\ &\leq \frac{\|A\|^4}{\gamma_s^2} \|\mathbf{y}_t - \mathbf{y}_{t+1}\|^2 \\ &= \frac{\|A\|^4 \eta^2}{\gamma_s^2} \|A\mathbf{x}_t - \mathbf{b}\|^2, \end{aligned}$$

where the last step also used the definition of yt+1. Using this estimation along with [\(66\)](#page-24-1) gives

$$\begin{aligned} & -\eta \langle A\mathbf{x}_t - \mathbf{b}, A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - A\mathbf{x}_t \rangle \\ & \geq - \left( \frac{\eta}{4} + \frac{2\|A\|^4 \eta^3}{\gamma_s^2} \right) \|A\mathbf{x}_t - \mathbf{b}\|^2 - 2\eta \|A\|^2 \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 \\ & \geq - \left( \frac{\eta \|A\|^2 \lambda^2}{2\gamma^2} + \frac{4\|A\|^6 \eta^3 \lambda^2}{\gamma^2 \gamma_s^2} + 2\eta \|A\|^2 \right) \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 \\ & \quad - \left( \frac{\eta}{2} + \frac{4\|A\|^4 \eta^3}{\gamma_s^2} \right) \|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2. \end{aligned}$$

We next have by Young's inequality that for any θ > 0:

$$\mu \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t \rangle \geq -\frac{\mu}{4\theta} \|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 - \theta \mu \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2.$$

The inequality derived in [\(65\)](#page-24-2) directly implies

$$-\eta \|A\mathbf{x}_t - A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \geq -\frac{\eta \|A\|^2 \lambda^2}{\gamma^2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2.$$

The key global error bound given in Lemma [A.12](#page-24-0) originally proved in [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7) results in

$$-6\mu\beta\|\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \bar{\mathbf{x}}^*(\mathbf{z}_t)\|^2 \geq -6\mu\beta\bar{\sigma}^2\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2.$$

Combining these estimates in [\(40\)](#page-17-1) leads to

$$\begin{aligned} \mathbb{E}[V_t] - \mathbb{E}[V_{t+1}] &\geq - \left( \frac{1}{2} \lambda \tau \mu + \lambda \tau^2 \mu^2 + \frac{\tau \lambda^2 \mu^2}{8 \gamma_s^2} + \frac{\mu}{\zeta} + \frac{3\mu}{\sigma_4} - \mu - \frac{\mu}{2\beta} + \frac{\mu}{4\theta} \right) \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 - \frac{1}{2} \lambda \tau^2 \sigma^2 \\ &+ \left( \frac{\tau \lambda^2}{16} - \frac{3\|A\|^2 \lambda^2 \eta}{2 \gamma^2} - \frac{4\|A\|^6 \eta^3 \lambda^2}{\gamma_s^2 \gamma^2} - 2\eta \|A\|^2 - \mu \theta \right) \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 \\ &+ \left( \frac{\eta}{2} - \frac{4\|A\|^4 \eta^3}{\gamma_s^2} - 6\mu\beta\bar{\sigma}^2 \right) \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2. \end{aligned} \quad (41)$$

We now estimate the coefficients inside the parantheses, with straightforward but tedious calculations which follow from the parameter settings.

First, we estimate the coefficient of <sup>E</sup>∥z<sup>t</sup> − zt+1∥ 2 in [\(41\)](#page-18-0): Let µ ≥ 4L<sup>f</sup> , we have σ<sup>4</sup> ≥ 1 2 because σ<sup>4</sup> = µ−L<sup>f</sup> µ . Then letting ζ = 6β, β < <sup>1</sup> <sup>30</sup> , we have

$$\mu - \frac{3\mu}{\sigma_4} \geq -5\mu \geq -\frac{\mu}{6\beta}, \quad \frac{\mu}{\zeta} = \frac{\mu}{6\beta}.$$

Therefore, we have that

$$\frac{\mu}{2\beta} + \mu - \frac{3\mu}{\sigma_4} - \frac{\mu}{\zeta} \geq \left( \frac{1}{2} - \frac{1}{6} - \frac{1}{6} \right) \frac{\mu}{\beta} \geq \frac{\mu}{6\beta}. \quad (42)$$

Hence, we estimate:

$$\text{coefficient of } \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\| \geq -\frac{1}{2}\lambda\tau\mu - \lambda\tau^2\mu^2 - \frac{\tau\lambda^2\mu^2}{8\gamma_s^2} + \frac{\mu}{6\beta} - \frac{\mu}{8\beta}.$$

Let η = η <sup>2</sup>∥A∥<sup>2</sup> , θ = 2β, η′ <sup>≤</sup> 1 <sup>40</sup> , and µ = max{2, 4L<sup>f</sup> }, λ = 2L<sup>K</sup> = 2(L<sup>f</sup> +ρ∥A∥+µ), τ ≤ <sup>10</sup>λ<sup>2</sup> , and γ<sup>s</sup> = µ−L<sup>f</sup> +γ from Fact [A.1.](#page-11-4) We have −λτµ ≥ − <sup>µ</sup> <sup>10</sup> and −2λτ <sup>2</sup><sup>µ</sup> <sup>2</sup> ≥ − <sup>µ</sup> <sup>100</sup> , then

$$\text{coefficient of } \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\| \geq \frac{\mu}{24\beta} - \frac{\mu}{20} - \frac{\mu}{100} - \tau\lambda^2 \frac{\mu^2}{(\mu - L_f + \lambda)^2}.$$

By β ≤ 1/30, we have <sup>1</sup> <sup>24</sup><sup>β</sup> − 1 <sup>20</sup> − 1 <sup>100</sup> ≥ 1 30β . In addition, using τλ<sup>2</sup> <sup>µ</sup> (µ−L<sup>f</sup> +λ) <sup>2</sup> ≤ τλ<sup>2</sup> ≤ 1 <sup>10</sup> , we fanally obtain:

$$\text{coefficient of } \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\| \geq \frac{\mu}{30\beta} - \frac{1}{10} \stackrel{\mu \geq 2}{\geq} \frac{\mu}{50\beta}. \quad (43)$$

Then we estimate the coefficient of <sup>E</sup>∥u ∗ (xt, yt+1, zt) − xt∥ 2 in [\(41\)](#page-18-0).

From above assumptions, we can easily get γ = (µ−L<sup>f</sup> )λ <sup>µ</sup>−L<sup>f</sup> <sup>+</sup><sup>λ</sup> ≥ because λ ≥ µ ≥ 2. Moreover, we assume η ′ ≤ τ <sup>40</sup> , η ′ <sup>µ</sup>−L<sup>f</sup> <sup>+</sup><sup>λ</sup> ≤ τ <sup>10</sup> , β ≤ τ <sup>40</sup> First, by our new notations, we have

$$\text{coefficient of } \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 = \frac{\tau\lambda^2}{16} - \frac{3\eta'\lambda^2}{4\gamma^2} - \frac{\eta'^3\lambda^2}{2\gamma^2\gamma_s^2} - \eta' - 2\mu\zeta$$

By γ ≥ 1 2 and the definition of γs, we have − 3η ′λ <sup>4</sup>γ<sup>2</sup> ≥ −3η ′λ 2 , − η ′3λ 2γ<sup>2</sup>γ<sup>2</sup> s ≥ − <sup>η</sup> ′3λ (µ−L<sup>f</sup> +λ) <sup>2</sup> , Then

$$\text{coefficient of } \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 \geq \frac{\tau\lambda^2}{16} - 3\eta'\lambda^2 - \frac{2\eta'^3\lambda^2}{(\mu - L_f + \lambda)^2} - \eta' - 2\mu\beta.$$

With 2 ≤ µ ≤ λ, η

′ ≤ τ <sup>100</sup> ,

η <sup>µ</sup>−L<sup>f</sup> <sup>+</sup><sup>λ</sup> ≤

τ <sup>10</sup> , β ≤ τ

<sup>200</sup> , we can obtain −3η

′λ

<sup>2</sup> ≥ −<sup>3</sup>τλ<sup>2</sup>

<sup>400</sup> , −

2η ′3λ (µ−L<sup>f</sup> +λ)

<sup>2</sup> ≥ −<sup>λ</sup> 2 τ 2 <sup>400</sup> ≥ −<sup>λ</sup>

2 τ <sup>400</sup> ,

−η

′ ≥ − <sup>τ</sup>

<sup>100</sup> ≥ −τλ<sup>2</sup>

<sup>100</sup> , <sup>−</sup>2µβ ≥ −τµ

50

µ≤λ ≥ −τλ

<sup>50</sup> ≥ −τλ<sup>2</sup>

<sup>100</sup> . Hence,

$$\text{coefficient of } \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 \geq \frac{\tau\lambda^2}{16} - \frac{3\tau\lambda^2}{100} - \frac{\tau\lambda^2}{400} - \frac{\tau\lambda^2}{400} - \frac{\tau\lambda^2}{100} = \frac{7\tau\lambda^2}{400}. \quad (44)$$

Last, we estimate the coefficient of <sup>E</sup>∥Ax ∗ (yt+1, zt) − b∥ 2 in [\(41\)](#page-18-0). By 6µβσ¯ <sup>2</sup> ≤ η 6 and the definition η ′ , γs, we have

$$-\frac{4\|A\|^2\eta^3}{\gamma_s^2} = -\frac{\eta'^2\eta}{(\mu-L_f+\lambda)^2} \frac{\frac{\eta'}{\mu-L_f+\lambda} \leq \frac{\tau}{10}}{\geq} -\frac{\eta\tau^2}{100} \geq -\frac{\eta}{100} \text{ and } -6\mu\beta\bar{\sigma}^2 \geq -\frac{\eta}{6}. \text{ Hence, we have}$$

$$\text{coefficient of } \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 \geq \frac{\eta}{2} - \frac{\eta}{100} - \frac{\eta}{6} \geq \frac{\eta}{4}. \quad (45)$$

Plugging [\(43\)](#page-18-1), [\(44\)](#page-19-1) and [\(45\)](#page-19-2) to [\(41\)](#page-18-0), we finish the proof. ■

## A.2. Proof of Theorem [3.1](#page-4-5)

*Proof of Theorem [3.1.](#page-4-5)* We start from the result in Lemma [A.9.](#page-16-0) First, it follows from the definition of zt+1 that

$$\|\mathbf{z}_t - \mathbf{z}_{t+1}\| = \beta \|\mathbf{x}_t - \mathbf{z}_t\|.$$

So, we rewrite [\(37\)](#page-16-3), as:

$$\mathbb{E}V_t - \mathbb{E}V_{t+1} \geq \beta^2 c_\beta \mathbb{E}\|\mathbf{x}_t - \mathbf{z}_t\|^2 + c_\tau \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 + c_\eta \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 - \frac{1}{2}\lambda\tau^2\sigma^2. \quad (46)$$

For t > 0, we have V<sup>t</sup> ≥ f, which is proven in Lemma [A.13.](#page-24-3) It then follows that

$$\sum_{t=0}^{T-1} (\mathbb{E}V_t - \mathbb{E}V_{t+1}) = V_0 - \mathbb{E}V_T \leq V_0 - \underline{f}. \quad (47)$$

Then, summing up [\(46\)](#page-19-3), using [\(47\)](#page-19-4), and the fact that c<sup>τ</sup> = Θ(τ ), c<sup>η</sup> = Θ(τ ), β 2 c<sup>β</sup> = Θ(τ ) from [\(25\)](#page-11-0), we have

$$V_0 - \underline{f} + \frac{1}{2}T\lambda\tau^2\sigma^2 \geq \sum_{t=1}^T C_0\tau \left[\mathbb{E}\|\mathbf{x}_t - \mathbf{z}_t\|^2 + \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 + \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2\right],$$

for some explicit constant C0.

Dividing both sides by T, rearranging and using the definition τ = 1 6λ<sup>2</sup> √ T gives

$$\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}\|\mathbf{x}_t - \mathbf{z}_t\|^2 + \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 + \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 \leq \frac{1}{C_0\sqrt{T}} \left( 6\lambda(V_0 - \underline{f}) + \frac{\sigma^2}{12} \right). \quad (48)$$

$$\nabla \Psi(\mathbf{z}_t) = \mu(\mathbf{z}_t - \bar{\mathbf{x}}^*(\mathbf{z}_t)),$$

by Danskin's theorem, we deduce for any t

$$\begin{aligned} \frac{1}{\mu^2} \|\nabla \Psi(\mathbf{z}_t)\| &= \|\mathbf{z}_t - \bar{\mathbf{x}}^*(\mathbf{z}_t)\| \\ &\leq \|\mathbf{z}_t - \mathbf{x}^*(\mathbf{y}_{t+1}, z_t)\| + \|\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \bar{\mathbf{x}}^*(\mathbf{z}_t)\| \\ &\leq \|\mathbf{z}_t - \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\| + \bar{\sigma} \|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_s) - \mathbf{b}\| \\ &\leq \|\mathbf{z}_t - \mathbf{x}_t\| + \|\mathbf{x}_t - \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\| + \bar{\sigma} \|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\| \\ &\leq \|\mathbf{z}_t - \mathbf{x}_t\| + \frac{\lambda}{\gamma} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\| + \bar{\sigma} \|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|, \end{aligned}$$

where the first inequality is by triangle inequality, the second by [\(A.12\)](#page-24-0), the third by triangle inequality and the fourth by [\(58\)](#page-22-3).

Next, we take square of both sides, take expectation, use Young's inequality, sum for all t = 0, 1, . . . , T − 1, divide by T and use [\(48\)](#page-19-5) to derive

$$\begin{aligned} \frac{1}{\mu^2} \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E} \|\nabla \Psi(\mathbf{z}_t)\|^2 &\leq \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E} \left[ 3 \|\mathbf{z}_t - \mathbf{x}_t\|^2 + \frac{3\lambda^2}{\gamma^2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 + 3\bar{\sigma}^2 \|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 \right] \\ &= O\left(\frac{1}{\sqrt{T}}\right). \end{aligned}$$

The result then follows since t ∗ is selected uniformly at random from {0, 1, 2, . . . , T − 1}. ■

## A.3. Proof of Corollary [3.2](#page-4-6)

*Proof of Corollary [3.2.](#page-4-6)* From the definition of xˆ, we have

$$0 \in \hat{G}(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) + \frac{2}{\tau}(\hat{\mathbf{x}} - \mathbf{x}_t) + \partial I_X(\hat{\mathbf{x}}).$$

Let us set

$$\mathbf{v} = \nabla_{\mathbf{x}} K(\hat{\mathbf{x}}, \mathbf{y}_{t+1}, \mathbf{z}_t) - \hat{G}(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \frac{2}{\tau}(\hat{\mathbf{x}} - \mathbf{x}_t) - \rho A^T(A\hat{\mathbf{x}} - \mathbf{b}) - \mu(\hat{\mathbf{x}} - \mathbf{z}_t). \quad (49)$$

Combining with the optimality condition, we have

$$\begin{aligned} \mathbf{v} &\in \nabla_{\mathbf{x}} K(\hat{\mathbf{x}}, \mathbf{y}_{t+1}, \mathbf{z}_t) - \rho A^T(A\hat{\mathbf{x}} - \mathbf{b}) - \mu(\hat{\mathbf{x}} - \mathbf{z}_t) + \partial I_X(\hat{\mathbf{x}}) \\ &= \nabla f(\hat{\mathbf{x}}) + A^T \mathbf{y}_{t+1} + \partial I_X(\hat{\mathbf{x}}). \end{aligned}$$

Hence, we need to estimate <sup>E</sup>∥Axˆ − b∥ and <sup>E</sup>∥v∥.

For the mini-batch gradient in the post-processing step, we have

$$\mathbb{E}\|\hat{G}(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \nabla K(\mathbf{x}, \mathbf{y}, \mathbf{z})\|^2 \leq \frac{\sigma^2}{B}. \quad (50)$$

which is a standard calculation, see for example, [\(Lan,](#page-9-19) [2020,](#page-9-19) Section 5.2.3). Since B = Θ(ε −2 ), this gives us

$$\mathbb{E}\|\hat{G}(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \nabla K(\mathbf{x}, \mathbf{y}, \mathbf{z})\|^2 \leq \varepsilon^2. \quad (51)$$

First, let us note that the purpose of xˆ is to estimate u ∗ (xt, yt+1, zt), where

$$\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) = \arg \min_{\mathbf{u} \in X} \{l(\mathbf{u}) := K(\mathbf{u}, \mathbf{y}_{t+1}, \mathbf{z}_t) + \frac{\lambda}{2} \|\mathbf{x}_t - \mathbf{u}\|^2\}.$$

Note that the gradient of this objective is

$$\nabla l(\mathbf{u}) = \nabla_{\mathbf{x}} K(\mathbf{x}, \mathbf{y}_{t+1}, \mathbf{z}_t) + \lambda(\mathbf{x} - \mathbf{x}_t).$$

As a result, we have ∇l(xt) = ∇xK(xt, yt+1, zt).

Let us also denote

$$\mathbf{x}_t^* = \text{proj}_X(\mathbf{x}_t - \tau \nabla l(\mathbf{x}_t)).$$

That is, x ∗ t is the output of doing a full-gradient step on xt. Of course, this is not tractable in our setting, but we only use this as a theoretical tool.

Since this is a GD step on the objective l which is LK-smooth and convex with optimizer u ∗ (xt, yt+1, zt), the standard analysis for GD gives

$$\|\mathbf{x}_t^* - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \leq \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2, \quad (52)$$

as long as τ ≤ 1 L<sup>K</sup> .

Next, by the definitions of x ∗ t and xˆ, along with nonexpansiveness of the projection, we have

$$\begin{aligned} \mathbb{E}\|\mathbf{x}_t^* - \hat{\mathbf{x}}\|^2 &\leq \mathbb{E}\tau^2 \|\hat{G}(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \nabla_{\mathbf{x}} K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \\ &\leq \tau^2 \varepsilon^2, \end{aligned} \quad (53)$$

where the second inequality used [\(51\)](#page-20-1).

In view of [\(49\)](#page-20-2), we estimate ∥v∥ as

$$\|\mathbf{v}\| \leq \|\nabla_x K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \hat{G}(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\| + L_K \|\mathbf{x}_t - \hat{\mathbf{x}}\| + \frac{2}{\tau} \|\hat{\mathbf{x}} - \mathbf{x}_t\| + \rho \|A\| \|A\hat{\mathbf{x}} - \mathbf{b}\| + \mu \|\hat{\mathbf{x}} - \mathbf{z}_t\|.$$

On this, multiple applications of triangle inequality give

$$\begin{aligned} \|\hat{\mathbf{x}} - \mathbf{x}_t\| &\leq \|\hat{\mathbf{x}} - \mathbf{x}_t^*\| + \|\mathbf{x}_t^* - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\| + \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\| \\ &\leq \|\hat{\mathbf{x}} - \mathbf{x}_t^*\| + 2\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{x}_t) - \mathbf{x}_t\|, \end{aligned} \quad (54)$$

where the second line is due to [\(52\)](#page-21-0).

For the feasibility, we have by triangle inequality that

$$\|\hat{\mathbf{x}} - \mathbf{z}_t\| \leq \|\hat{\mathbf{x}} - \mathbf{x}_t\| + \|\mathbf{x}_t - \mathbf{z}_t\|. \quad (55)$$

As a result, we have that

$$\begin{aligned} \|\mathbf{v}\| &= O(\|\hat{\mathbf{x}} - \mathbf{x}_t^*\| + \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\| + \|A\hat{\mathbf{x}} - \mathbf{b}\| + \|\mathbf{x}_t - \mathbf{z}_t\| \\ &\quad + \|\nabla_{\mathbf{x}} K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \hat{G}(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|). \end{aligned} \quad (56)$$

For the feasibility, we have

$$\begin{aligned} \|A\hat{\mathbf{x}} - \mathbf{b}\| &\leq \|A\hat{\mathbf{x}} - A\mathbf{x}_t\| + \|A\mathbf{x}_t - \mathbf{b}\| \\ &\leq \|A\| \|\hat{\mathbf{x}} - \mathbf{x}_t\| + \|A\mathbf{x}_t - \mathbf{b}\|. \end{aligned}$$

Now, by invoking the above inequality for t = t ∗ , taking expectation, using Young's inequality, [\(54\)](#page-21-1), [\(53\)](#page-21-2) and [\(48\)](#page-19-5) along with [\(66\)](#page-24-1), we get that

$$\mathbb{E}\|A\hat{\mathbf{x}} - \mathbf{b}\|^2 \leq \varepsilon^2, \quad (57)$$

since T = Ω(ε −4 ).

Finally, using t = t ∗ , taking square and then expectation of [\(56\)](#page-21-3), using Young's inequality and then combining [\(57\)](#page-21-4), [\(53\)](#page-21-2), [\(51\)](#page-20-1) and [\(48\)](#page-19-5) gives the result since T = Ω(ε −4 ). ■

#### A.4. Auxiliary Results

Lemma A.10. *Under Assumption [1.1,](#page-3-2) for any* x, z, z ′ ∈ X*, we have*

$$\frac{\lambda}{\gamma} \|\mathbf{x} - \mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z})\| \geq \|\mathbf{x} - \mathbf{x}^*(\mathbf{y}, \mathbf{z})\|, \quad (58)$$

$$\|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{x}\| \leq \|\mathbf{x} - \mathbf{x}^*(\mathbf{y}, \mathbf{z})\|, \quad (59)$$

$$\|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{u}^*(\mathbf{x}, \mathbf{y}', \mathbf{z})\| \leq \frac{\|A\|}{\gamma_s} \|\mathbf{y} - \mathbf{y}'\|, \quad (60)$$

$$\|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}')\| \leq \frac{\mu}{\gamma_s} \|\mathbf{z} - \mathbf{z}'\|, \quad (61)$$

$$\|\mathbf{z}' - \mathbf{z}\| \geq \frac{\mu - L_f}{\mu} \|\mathbf{x}^*(\mathbf{y}, \mathbf{z}') - \mathbf{x}^*(\mathbf{y}, \mathbf{z})\|, \quad (62)$$

$$\|\mathbf{y}' - \mathbf{y}\| \geq \frac{\gamma_K}{\|A\|} \|\mathbf{x}^*(\mathbf{y}', \mathbf{z}) - \mathbf{x}^*(\mathbf{y}, \mathbf{z})\|, \quad (63)$$

$$\|\bar{\mathbf{x}}^*(\mathbf{z}) - \bar{\mathbf{x}}^*(\mathbf{z}')\| \leq \frac{\mu}{\mu - L_f} \|\mathbf{z} - \mathbf{z}'\|, \quad (64)$$

*where* γ = (µ−L<sup>f</sup> )λ µ−L<sup>f</sup> +λ , γ<sup>s</sup> = µ − L<sup>f</sup> + λ, γ<sup>K</sup> = µ − L<sup>f</sup> *.*

where 
$$\gamma = \frac{(\mu - L_f)\lambda}{\mu - L_f + \lambda}$$
,  $\gamma_s = \mu - L_f + \lambda$ ,  $\gamma_K = \mu - L_f$ .

*Proof.* The proofs for [\(62\)](#page-22-4), [\(63\)](#page-22-1), and [\(64\)](#page-22-5) appear in [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7), so we omit these proofs.

We first prove [\(58\)](#page-22-3). Let us note that x ∗ (y, z) minimizes φ1/λ, see for example [\(Hiriart-Urruty & Lemarechal,](#page-9-20) [1993,](#page-9-20) Theorem XV4.1.7). As a result, we have ∇xφ1/λ(x ∗ (y, z), y, z) = 0. From Lemma [A.3,](#page-11-7) we have that φ1/λ(·, y, z) is γ = (µ−L<sup>f</sup> )λ µ−L<sup>f</sup> +λ -strongly convex.

Then, by strong convexity, we have

$$\begin{aligned} \langle \nabla_{\mathbf{x}} \varphi_{1/\lambda}(\mathbf{x}^*(\mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) - \nabla_{\mathbf{x}} \varphi_{1/\lambda}(\mathbf{x}, \mathbf{y}, \mathbf{z}), \mathbf{x}^*(\mathbf{y}, \mathbf{z}) - \mathbf{x} \rangle &\geq \gamma \|\mathbf{x} - \mathbf{x}^*(\mathbf{y}, \mathbf{z})\|^2 \\ \iff \|\nabla_{\mathbf{x}} \varphi_{1/\lambda}(\mathbf{x}, \mathbf{y}, \mathbf{z})\| &\geq \gamma \|\mathbf{x} - \mathbf{x}^*(\mathbf{y}, \mathbf{z})\|, \end{aligned}$$

where the inclusion used ∇xφ1/λ(x ∗ (y, z), y, z) = 0 established in the previous paragraph as well as Cauchy-Schwarz inequality. Then, using ∇xφ1/λ(x, y, z) = λ(x − u ∗ (x, y, z)), we obtain [\(58\)](#page-22-3).

From definition of u ∗ (x, y, z), we have,

$$K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) + \frac{\lambda}{2} \|\mathbf{x} - \mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z})\|^2 \leq K(\mathbf{x}^*(\mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) \frac{\lambda}{2} \|\mathbf{x} - \mathbf{x}^*(\mathbf{y}, \mathbf{z})\|^2,$$

where we also remark that x ∗ (y, z) ∈ X. Combining with K(x ∗ (y, z), y, z) ≤ K(u ∗ (x, y, z), y, z), which follows from the definition of x ∗ (y, z) we have [\(59\)](#page-22-6).

The proofs of the other two assertions will use a similar idea to [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7), but there will be differences in the estimations since this previous work did not use the function φ1/λ.

For [\(60\)](#page-22-2), we proceed by using the definition of φ1/λ and adding and subtracting K(u ∗ (x, y ′ , z), y, z) to get

$$\begin{aligned} & K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) + \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{x}\|^2 - K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}', \mathbf{z}), \mathbf{y}', \mathbf{z}) - \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}', \mathbf{z}) - \mathbf{x}\|^2 \\ &= K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) + \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{x}\|^2 \\ &\quad - K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}', \mathbf{z}), \mathbf{y}, \mathbf{z}) - \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}', \mathbf{z}) - \mathbf{x}\|^2 \\ &\quad + K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}', \mathbf{z}), \mathbf{y}, \mathbf{z}) - K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}', \mathbf{z}), \mathbf{y}', \mathbf{z}) \\ &\leq \frac{-\gamma_s}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{u}^*(\mathbf{x}, \mathbf{y}', \mathbf{z})\|^2 + \langle \mathbf{y} - \mathbf{y}', A\mathbf{u}^*(\mathbf{x}, \mathbf{y}', \mathbf{z}) - \mathbf{b} \rangle, \end{aligned}$$

where last step uses u 7→ K(u, y, z) + <sup>λ</sup> 2 ∥u − x∥ <sup>2</sup> being γs-strongly convex (cf. Fact [A.1\)](#page-11-4) with minimizer u ∗ (x, y, z), as well as the definition of K.

We then argue similarly, this time adding and subtracting K(u ∗ (x, y, z), y ′ , z):

$$\begin{aligned} & K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) + \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{x}\|^2 - K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}', \mathbf{z}), \mathbf{y}', \mathbf{z}) - \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}', \mathbf{z}) - \mathbf{x}\|^2 \\ &= K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}), \mathbf{y}', \mathbf{z}) + \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{x}\|^2 \\ &\quad - K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}', \mathbf{z}), \mathbf{y}', \mathbf{z}) - \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}', \mathbf{z}) - \mathbf{x}\|^2 \\ &\quad - K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}), \mathbf{y}', \mathbf{z}) + K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) \\ &\geq \frac{\gamma_s}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{u}^*(\mathbf{x}, \mathbf{y}', \mathbf{z})\|^2 + \langle \mathbf{y} - \mathbf{y}', A\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{b} \rangle. \end{aligned}$$

where last step uses that u 7→ K(u, y ′ , z) + <sup>λ</sup> 2 ∥u − x∥ 2 is γs-strongly convex (cf. Fact [A.1\)](#page-11-4) with minimizer u ∗ (x, y ′ , z) and the definition of K.

Combining the last two estimates give

$$\langle \mathbf{y} - \mathbf{y}', A\mathbf{u}^*(\mathbf{x}, \mathbf{y}', \mathbf{z}) - A\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) \rangle \geq \gamma_s \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{u}^*(\mathbf{x}, \mathbf{y}', \mathbf{z})\|^2.$$

Using Cauchy-Schwarz inequality and the definition of operator norm gives [\(60\)](#page-22-2).

The proof of [\(61\)](#page-22-0) is similar to the proof of [\(60\)](#page-22-2), just completed. In particular, by adding and subtracting K(u ∗ (x, y, z), y, z ′ ), we have

$$\begin{aligned} K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) + \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{x}\|^2 - K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}'), \mathbf{y}, \mathbf{z}') + \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}') - \mathbf{x}\|^2 \\ = K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) + \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{x}\|^2 - K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}'), \mathbf{y}, \mathbf{z}) - \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}') - \mathbf{x}\|^2 \\ - K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}'), \mathbf{y}, \mathbf{z}') + K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}'), \mathbf{y}, \mathbf{z}) \\ \leq -\frac{\gamma s}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}')\|^2 + \frac{\mu}{2} (\|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}') - \mathbf{z}\|^2 - \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}') - \mathbf{z}'\|^2), \end{aligned}$$

where we used that u 7→ K(u, y, z) + <sup>λ</sup> 2 ∥u − x∥ 2 is γs-strongly convex with minimizer u ∗ (x, y, z) and the definition of K.

Finally, we add and subtract K(u ∗ (x, y, z ′ ), y, z) to get

$$\begin{aligned} & K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) + \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{x}\|^2 - K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}'), \mathbf{y}, \mathbf{z}') - \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}') - \mathbf{x}\|^2 \\ & = K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}') + \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{x}\|^2 - K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}'), \mathbf{y}, \mathbf{z}) - \frac{\lambda}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}') - \mathbf{x}\|^2 \\ & + K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) - K(\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}') \\ & \geq \frac{\gamma s}{2} \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}')\|^2 + \frac{\mu}{2} (\|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{z}\|^2 - \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{z}'\|^2), \end{aligned}$$

where we used that u 7→ K(u, y, z ′ ) + <sup>λ</sup> 2 ∥u − x∥ 2 is γs-strongly convex with minimizer u ∗ (x, y, z ′ ) and the definition of K.

Combining the last two inequalities give

$$\mu\langle \mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}') - \mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}), \mathbf{z}' - \mathbf{z} \rangle \geq \gamma_s \|\mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \mathbf{u}^*(\mathbf{x}, \mathbf{y}, \mathbf{z}')\|^2.$$

Lemma A.11. *Under Assumption [1.1,](#page-3-2) for* xt, yt+1, z<sup>t</sup> *generated by Algorithm [1,](#page-5-0) we have*

$$\|\mathbf{Ax}_t - \mathbf{Ax}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \leq \frac{\|\mathbf{A}\|^2 \lambda^2}{\gamma^2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2, \quad (65)$$

$$\|A\mathbf{x}_t - \mathbf{b}\|^2 \leq \frac{2\|A\|^2\lambda^2}{\gamma^2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 + 2\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2, \quad (66)$$

$$\|A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - A\mathbf{x}_t\|^2 \leq \frac{2\|A\|^4}{\gamma_s^2} \|\mathbf{y}_t - \mathbf{y}_{t+1}\|^2 + 2\|A\|^2 \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2, \quad (67)$$

*where* γ, γ<sup>s</sup> *are defined in* [\(25\)](#page-11-0)*.*

*Proof.* The assertion in [\(65\)](#page-24-2) follows directly from [\(58\)](#page-22-3) since

$$\|A\mathbf{x}_t - A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \leq \|A\|^2 \|\mathbf{x}_t - \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 \leq \frac{\|A\|^2 \lambda^2}{\gamma^2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2.$$

Combining the first assertion with Young's inequality gives the second assertion, since

$$\begin{aligned} \|A\mathbf{x}_t - \mathbf{b}\|^2 &\leq 2\|A\mathbf{x}_t - A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 + 2\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 \\ &\leq \frac{2\|A\|^2\lambda^2}{\gamma^2} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 + 2\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2. \end{aligned}$$

Young's inequality and [\(60\)](#page-22-2) gives the third assertion

$$\begin{aligned} \|A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - A\mathbf{x}_t\|^2 &\leq 2\|A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 + 2\|A\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - A\mathbf{x}_t\|^2 \\ &\leq \frac{2\|A\|^4}{\gamma_s^2} \|\mathbf{y}_t - \mathbf{y}_{t+1}\|^2 + 2\|A\|^2 \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2. \end{aligned}$$

The proof is completed. ■

The following important lemma is known as the global error bound in [\(Zhang & Luo,](#page-10-7) [2022\)](#page-10-7). This global result holds in its entirety in our case, so we only state it here and refer to where it appeared originally for the precise definition of the constant σ¯ which depends on Hoffman constant of certain linear systems.

Lemma A.12. *[\(Zhang & Luo,](#page-10-7) [2022,](#page-10-7) Lemma 3.2) If* µ > L<sup>f</sup> *, then we have*

$$\|\mathbf{x}^*(\mathbf{y}, \mathbf{z}) - \bar{\mathbf{x}}^*(\mathbf{z})\| \leq \bar{\sigma} \|A\mathbf{x}^*(\mathbf{y}, \mathbf{z}) - \mathbf{b}\| \quad \text{for any } \mathbf{y}, \mathbf{z},$$

*where* σ >¯ 0 *depends only on the constants* C<sup>1</sup> = (L<sup>f</sup> + ρ∥A∥ <sup>2</sup> + µ)*,* C<sup>2</sup> = −L<sup>f</sup> + µ*, and the matrices* A, H *and is always finite.*

Lemma A.13. *If* (x, z) ∈ X × X*, we have* φ1/λ(x, y, z) − 2d(y, z) + 2Ψ(z) ≥ f*.*

*Proof.* Because x ∗ (y, z) minimizes φ1/λ(·, y, z) (see for example [\(Hiriart-Urruty & Lemarechal,](#page-9-20) [1993,](#page-9-20) Theorem XV4.1.7)), we have

$$\varphi_{1/\lambda}(\mathbf{x}, \mathbf{y}, \mathbf{z}) \geq \varphi_{1/\lambda}(\mathbf{x}^*(\mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) = K(\mathbf{x}^*(\mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}).$$

We can then deduce

$$\begin{aligned}\varphi_{1/\lambda}(\mathbf{x}, \mathbf{y}, \mathbf{z}) - 2d(\mathbf{y}, \mathbf{z}) + 2\Psi(z) &\geq K(\mathbf{x}^*(\mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) - 2d(\mathbf{y}, \mathbf{z}) + 2\Psi(\mathbf{z}) \\ &= d(\mathbf{y}, \mathbf{z}) - 2d(\mathbf{y}, \mathbf{z}) + 2\Psi(\mathbf{z}) \\ &= \Psi(\mathbf{z}) + \Psi(\mathbf{z}) - d(\mathbf{y}, \mathbf{z}) \\ &\geq \Psi(\mathbf{z}) \\ &\geq \underline{f}\end{aligned}$$

The second inequality in the above chain comes from definition, that is, denoting x ∗ <sup>µ</sup> = arg minx∈X,Ax=b{f(x) + <sup>µ</sup> 2 ∥x − z∥ <sup>2</sup>} in view of [\(7\)](#page-3-3), we have

$$d(\mathbf{y}, \mathbf{z}) = \min_{\mathbf{x} \in X} K(\mathbf{x}, \mathbf{y}, \mathbf{z}) \leq K(\mathbf{x}_\mu^*, \mathbf{y}, \mathbf{z}) = f(\mathbf{x}_\mu^*) + \frac{\mu}{2} \|\mathbf{x}_\mu^* - \mathbf{z}\|^2 = \Psi(\mathbf{z}),$$

where the first inequality also uses x ∗ <sup>µ</sup> ∈ X, which is by definition. ■

## B. Proofs for Section [4](#page-6-4)

Notation. In this section, we have ∥∇f(x, ξt) − ∇f(xt)∥ <sup>2</sup> ≤ σ 2 and <sup>E</sup>∥A<sup>ζ</sup>tx<sup>t</sup> − bζ<sup>t</sup> <sup>2</sup> ≤ L, then we denote the boundedness of variance as <sup>E</sup>∥G(xt, yt, zt, ξt) − ∇xK(xt, yt, zt)∥ <sup>2</sup> ≤ σ 2 2 , where the boundedness is proved in [B.2.](#page-26-0)

We start with some helper lemmas before proving Theorem [4.3.](#page-6-6)

Lemma B.1. *Let Assumption [4.1](#page-6-1) hold. With the update rule of* yt+1 = y<sup>t</sup> + η(A<sup>ζ</sup>tx<sup>t</sup> − bζ<sup>t</sup> )*, where* <sup>E</sup>ζ<sup>t</sup> [A<sup>ζ</sup>tx<sup>t</sup> − bζ<sup>t</sup> ] = Ax<sup>t</sup> − b*, we have*

$$\begin{aligned}\mathbb{E}d(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbb{E}d(\mathbf{y}_t, \mathbf{z}_t) &\geq \eta \mathbb{E}\langle (A\mathbf{x}_t - \mathbf{b}), A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b} \rangle - \frac{\eta^2}{32} \mathbb{E}\|A\mathbf{x}_t - \mathbf{b}\|^2 - \left( \frac{1}{2} + \frac{17\|A\|^4}{2\gamma_K} \right) \eta^2 L^2 \\ &\quad + \frac{\mu}{2} \mathbb{E}\langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_{t+1} + \mathbf{z}_t - 2\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle, \\ \mathbb{E}\Psi(\mathbf{z}_{t+1}) - \mathbb{E}\Psi(\mathbf{z}_t) &\leq \mu \mathbb{E}\langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_t - \bar{\mathbf{x}}^*(\mathbf{z}_t) \rangle + \frac{\mu}{2\sigma_4} \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2,\end{aligned}\tag{68}$$

*where* γK, σ<sup>4</sup> *are introduceed in [A.10,](#page-22-7) and by Assumption [4.1,](#page-6-1) we have* <sup>E</sup>∥A<sup>ζ</sup>tx<sup>t</sup> − bζ<sup>t</sup> ∥ <sup>2</sup> ≤ L *for some finite* L*.*

*Proof.* It is easy to derive, for example as [\(Zhang & Luo,](#page-10-1) [2020,](#page-10-1) Lemma 3.2), that

$$d(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - d(\mathbf{y}_t, \mathbf{z}_t) \geq \langle \mathbf{y}_{t+1} - \mathbf{y}_t, A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b} \rangle + \frac{\mu}{2} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_{t+1} + \mathbf{z}_t - 2\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle.$$

Hence, by using the update rule of yt+1, we get

$$\begin{aligned}
d(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - d(\mathbf{y}_t, \mathbf{z}_t) &\geq \langle \mathbf{y}_{t+1} - \mathbf{y}_t, \mathbf{Ax}^*(\mathbf{y}_t, \mathbf{z}_t) - \mathbf{b} \rangle + \langle \mathbf{y}_{t+1} - \mathbf{y}_t, \mathbf{Ax}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{Ax}^*(\mathbf{y}_t, \mathbf{z}_t) \rangle \\
&\quad + \frac{\mu}{2} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_{t+1} + \mathbf{z}_t - 2\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle \\
&\geq \langle \mathbf{y}_{t+1} - \mathbf{y}_t, \mathbf{Ax}^*(\mathbf{y}_t, \mathbf{z}_t) - \mathbf{b} \rangle - \frac{1}{2} \|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 - \frac{1}{2} \|\mathbf{Ax}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{Ax}^*(\mathbf{y}_t, \mathbf{z}_t)\|^2 \\
&\quad + \frac{\mu}{2} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_{t+1} + \mathbf{z}_t - 2\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle \\
&\geq \langle \eta(A_{\zeta_t} \mathbf{x}_t - \mathbf{b}_{\zeta_t}), \mathbf{Ax}^*(\mathbf{y}_t, \mathbf{z}_t) - \mathbf{b} \rangle - \frac{1}{2} \eta^2 L^2 - \frac{\|A\|^4}{2\gamma_K^2} \|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\
&\quad + \frac{\mu}{2} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_{t+1} + \mathbf{z}_t - 2\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle,
\end{aligned}$$

where we use Cauchy-Schwarz inequality in the second step, and the last inequality comes from the bound of <sup>E</sup>∥A<sup>ζ</sup>tx<sup>t</sup> − bζ<sup>t</sup> also [\(63\)](#page-22-1).

After taking expectation and using tower property along with yt, z<sup>t</sup> being deterministic under the conditioning, we have

$$\begin{aligned}\mathbb{E}d(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - \mathbb{E}d(\mathbf{y}_t, \mathbf{z}_t) &\geq \mathbb{E}\langle \eta(\mathbf{Ax}_t - \mathbf{b}), \mathbf{Ax}^*(\mathbf{y}_t, \mathbf{z}_t) - \mathbf{b} \rangle - \frac{1}{2}\eta^2 L^2 - \frac{\|\mathbf{A}\|^4}{2\gamma_K^2} \mathbb{E}\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\ &\quad + \frac{\mu}{2} \mathbb{E}\langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_{t+1} + \mathbf{z}_t - 2\mathbf{x}(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle.\end{aligned}$$

Then we estimate as

$$\begin{aligned} & \mathbb{E}\langle \eta(A\mathbf{x}_t - \mathbf{b}), A\mathbf{x}^*(\mathbf{y}_t, \mathbf{z}_t) - \mathbf{b} \rangle - \frac{1}{2}\eta^2 L^2 - \frac{\|A\|^4}{2\gamma_K^2} \mathbb{E}\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\ & \geq \eta \mathbb{E}\langle (A\mathbf{x}_t - \mathbf{b}), A\mathbf{x}^*(\mathbf{y}_t, \mathbf{z}_t) - \mathbf{b} \rangle - \left( \frac{1}{2} + \frac{\|A\|^4}{2\gamma_K^2} \right) \eta^2 L^2 \\ & = \eta \mathbb{E}[\langle (A\mathbf{x}_t - \mathbf{b}), A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b} \rangle + \eta \langle (A\mathbf{x}_t - \mathbf{b}), -A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) + A\mathbf{x}^*(\mathbf{y}_t, \mathbf{z}_t) \rangle] - \left( \frac{1}{2} + \frac{\|A\|^4}{2\gamma_K^2} \right) \eta^2 L^2 \\ & \geq \eta \mathbb{E}[\langle (A\mathbf{x}_t - \mathbf{b}), A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b} \rangle - 8\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - A\mathbf{x}^*(\mathbf{y}_t, \mathbf{z}_t)\|^2] - \frac{\eta^2}{32} \|A\mathbf{x}_t - \mathbf{b}\|^2 - \left( \frac{1}{2} + \frac{\|A\|^4}{2\gamma_K^2} \right) \eta^2 L^2 \\ & \geq \eta \mathbb{E}\langle (A\mathbf{x}_t - \mathbf{b}), A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b} \rangle - \frac{\eta^2}{32} \mathbb{E}\|A\mathbf{x}_t - \mathbf{b}\|^2 - \frac{1}{2}\eta^2 L^2 - \frac{17\|A\|^4}{2\gamma_K^2} \eta^2 L^2, \end{aligned}$$

where the first inequality comes from <sup>E</sup>[∥A<sup>ζ</sup>tx<sup>t</sup> − bζ<sup>t</sup> ∥ 2 ] ≤ L and the second inequality comes from ⟨a, b⟩ ≤ <sup>1</sup> <sup>32</sup> ∥a∥ <sup>2</sup> + 8∥b∥ 2 (∀a, b). And in last inequality we use [\(63\)](#page-22-1) again.

The estimation of <sup>E</sup>Ψ(zt+1) − <sup>E</sup>Ψ(zt) is the same as Lemma [A.8.](#page-16-1) Because the randomness of ζ<sup>t</sup> in the stochastic dual update does not change the recursion in <sup>E</sup>Ψ(zt+1) − <sup>E</sup>Ψ(zt), where zt, zt+1 only depend on the randomness before ζt. Hence we omit the proof here.

This completes the proof. ■

Lemma B.2. *Let Assumption [1.1](#page-3-2) and [4.1](#page-6-1) hold. By using the parameters* [\(25\)](#page-11-0) *in Algorithm [2,](#page-7-3) then in the iteration* t + 1*, if the dual update runs as* yt+1 = y<sup>t</sup> + η(A<sup>ζ</sup>tx<sup>t</sup> − bζ<sup>t</sup> )*, we obtain*

$$\begin{aligned} \mathbb{E}V_t - \mathbb{E}V_{t+1} &\geq \tilde{c}_\beta \mathbb{E}\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 + \tilde{c}_\eta \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 + \tilde{c}_\eta \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 \\ &\quad - \lambda\tau^2\sigma_2^2 - \left(1 + \frac{17\|A\|^4}{\gamma_K^2}\right) \eta^2 L^2, \end{aligned} \quad (69)$$

*where* e<sup>c</sup><sup>β</sup> <sup>=</sup> µ 50β *,* e<sup>c</sup><sup>τ</sup> <sup>=</sup> 6τλ<sup>2</sup> <sup>400</sup> *,* e<sup>c</sup><sup>η</sup> <sup>=</sup> η 8 *and* <sup>E</sup>∥G(xt, yt, zt, ξt) − ∇xK(xt, yt, zt)∥ <sup>2</sup> ≤ σ 2 2 *.*

*Proof.* First, we show <sup>E</sup>∥G(xt, yt, zt, ξt) − ∇xK(xt, yt, zt)∥ 2 is bounded.

Recall that in Equation [\(17\)](#page-6-3) we have

$$G(\mathbf{x}, \mathbf{y}, \mathbf{z}, \xi) = \nabla f(\mathbf{x}, \xi) + A_{\xi^1}^\top \mathbf{y} + \rho A_{\xi^1}^\top (A_{\xi^2} \mathbf{x} - \mathbf{b}_{\xi^2}) + \mu(\mathbf{x} - \mathbf{z}). \quad (70)$$

We estimate by using Young's inequalities

$$\begin{aligned} & \mathbb{E}\|G(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t, \xi_t) - \nabla_{\mathbf{x}}K(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t)\|^2 \\ & \leq 2\mathbb{E}\|G(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t, \xi_t) - G(\mathbf{x}_t, 0, \mathbf{z}_t, \xi_t)\|^2 + 2\mathbb{E}\|G(\mathbf{x}_t, 0, \mathbf{z}_t, \xi_t) - \nabla_{\mathbf{x}}K(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t)\|^2 \\ & \leq 2\mathbb{E}L_G\|\mathbf{y}_t\|^2 + 2\mathbb{E}\|G(\mathbf{x}_t, 0, \mathbf{z}_t, \xi_t) - \nabla_{\mathbf{x}}K(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t)\|^2 \\ & \leq 2\mathbb{E}L_G\|\mathbf{y}_t\|^2 + 4\mathbb{E}\|G(\mathbf{x}_t, 0, \mathbf{z}_t, \xi_t) - \nabla_{\mathbf{x}}K(\mathbf{x}_t, 0, \mathbf{z}_t)\|^2 + 4\mathbb{E}\|\nabla_{\mathbf{x}}K(\mathbf{x}_t, 0, \mathbf{z}_t) - \nabla_{\mathbf{x}}K(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t)\|^2 \\ & \leq 2L_G M_y^2 + 4\mathbb{E}\|G(\mathbf{x}_t, 0, \mathbf{z}_t, \xi_t) - \nabla_{\mathbf{x}}K(\mathbf{x}_t, 0, \mathbf{z}_t)\|^2 + 4\|A\|^2\|\mathbf{y}_t\|^2, \end{aligned}$$

where in second inequality we use L<sup>G</sup> is the Lipschitz constant of G with respect to variable y, then in third inequality we use M<sup>y</sup> as the upper bound of ∥yt∥.

Because xt, yt, z<sup>t</sup> are all bounded, <sup>E</sup>∥G(xt, yt, zt, ξt) − ∇xK(xt, yt, zt)∥ is bounded, we denote the upper bound as σ 2 . Note that Corollary [A.7](#page-15-1) still holds for xt, yt, zt, xt+1, yt+1, zt+1, but the variance σ is changed to σ<sup>2</sup> (since this corollary

and the lemmas used in its proof do not use the particular form of yt+1). Then combining with Lemma [B.1,](#page-25-0) we have

$$\begin{aligned} \mathbb{E}[V_t - V_{t+1}] &= \mathbb{E} \left[ \varphi_{1/\lambda}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \varphi_{1/\lambda}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + 2d(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - 2d(\mathbf{y}_t, \mathbf{z}_t) + 2\Psi(\mathbf{z}_t) - 2\Psi(\mathbf{z}_{t+1}) \right] \\ &\geq \frac{\tau\lambda^2}{16} \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 - \left( \frac{\lambda\tau\mu}{2} + \lambda\tau^2\mu^2 + \frac{\tau\lambda^2\mu^2}{8\gamma_s^2} \right) \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 - \frac{\lambda\tau^2\sigma_2^2}{2} \\ &\quad - \eta\mathbb{E}\langle \mathbf{Ax}_t - \mathbf{b}, \mathbf{Au}^*(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \mathbf{b} \rangle + \frac{\mu}{2}\mathbb{E}\langle \mathbf{z}_{t+1} - \mathbf{z}_t, 2\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{z}_t - \mathbf{z}_{t+1} \rangle \\ &\quad + 2\eta\mathbb{E}\langle \mathbf{Ax}_t - \mathbf{b}, \mathbf{Ax}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b} \rangle + \mu\mathbb{E}\langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_{t+1} + \mathbf{z}_t - 2\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle \\ &\quad - 2\mu\mathbb{E}\langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_t - \bar{\mathbf{x}}^*(\mathbf{z}_t) \rangle - \frac{\mu}{\sigma_4} \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 - \frac{\eta^2}{32} \mathbb{E}\|\mathbf{Ax}_t - \mathbf{b}\|^2 - \left( \frac{1}{2} + \frac{17\|\mathbf{A}\|^4}{2\gamma_K^2} \right) \eta^2 L^2, \end{aligned}$$

where − η <sup>32</sup><sup>E</sup>∥Ax<sup>t</sup> − b∥ <sup>2</sup> − 1 <sup>2</sup> + 17∥A∥ 4 2γ 2 K η <sup>2</sup>L 2 is the difference comparing to the deterministic linear constraints result in Lemma [A.9.](#page-16-0) We then estimate like Lemma [A.9](#page-16-0) to have

$$\begin{aligned} \mathbb{E}V_t - \mathbb{E}V_{t+1} &\geq c_{\beta}\mathbb{E}\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 + c_{\tau}\mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 + c_{\eta}\mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 - \frac{1}{2}\lambda\tau^2\sigma_2^2 \\ &\quad - \frac{\tau^2}{16}\mathbb{E}\|A\mathbf{x}_t - \mathbf{b}\|^2 - \left(1 + \frac{17\|A\|^4}{\gamma_K^2}\right)\eta^2L^2, \end{aligned} \quad (71)$$

where c<sup>β</sup> = µ 50β , c<sup>τ</sup> = 7τλ<sup>2</sup> <sup>400</sup> , c<sup>η</sup> = η 4 .

where 
$$c_\beta = \frac{\mu}{50\beta}$$
,  $c_\tau = \frac{7\tau\lambda^2}{400}$ ,  $c_\eta = \frac{\eta}{4}$ .

We also have by Young's inequality and Lemma [A.11](#page-23-0) that

$$-\frac{\eta^2}{16}\mathbb{E}\|A\mathbf{x}_t - \mathbf{b}\|^2 \geq -\frac{\|A\|^2\lambda^2\eta^2}{8\gamma^2}\mathbb{E}\|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 - \frac{\eta^2}{8}\mathbb{E}\|A\mathbf{x}(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2.$$

By the parameter choices, we have <sup>7</sup>τλ<sup>2</sup> <sup>400</sup> − ∥A∥ <sup>2</sup>λ <sup>2</sup>η <sup>8</sup>γ<sup>2</sup> ≥ 6τλ<sup>2</sup> <sup>400</sup> and <sup>η</sup> <sup>4</sup> − η <sup>8</sup> ≥ η 8 . Using these estimations in [\(71\)](#page-27-0) gives the proof. ■

Proposition B.3. *Under Assumption [4.1,](#page-6-1)* ∥yt∥ ≤ Ψ(zt)−d(yt,zt)+2<sup>M</sup> r *, where* M = maxx,z∈X{|f(x)| + µ 2 ∥x − z∥ <sup>2</sup> + ρ 2 ∥Ax − b∥ <sup>2</sup>} *and* r > 0 *is defined as* ∥Axˆ − b∥ = r *where* xˆ *is in the relative interior of the constraints. The existence of this is guaranteed by our assumption.*

*Proof.* Given <sup>x</sup>e ∈ <sup>X</sup>, we have

$$\begin{aligned}
\Psi(\mathbf{z}_t) - d(\mathbf{y}_t, \mathbf{z}_t) &\geq f(\bar{\mathbf{x}}^*(\mathbf{z}_t)) + \frac{\mu}{2} \|\bar{\mathbf{x}}^*(\mathbf{z}_t) - \mathbf{z}_t\|^2 - K(\bar{\mathbf{x}}, \mathbf{y}_t, \mathbf{z}_t) \\
&\geq f(\bar{\mathbf{x}}^*(\mathbf{z}_t)) + \frac{\mu}{2} \|\bar{\mathbf{x}}^*(\mathbf{z}_t) - \mathbf{z}_t\|^2 - [f(\bar{\mathbf{x}}) + \langle \mathbf{y}_t, A\bar{\mathbf{x}} \rangle + \frac{\rho}{2} \|A\bar{\mathbf{x}} - \mathbf{b}\|^2 + \frac{\mu}{2} \|\bar{\mathbf{x}} - \mathbf{z}_t\|^2] \\
&= \left[ f(\bar{\mathbf{x}}^*(\mathbf{z}_t)) + \frac{\mu}{2} \|\bar{\mathbf{x}}^*(\mathbf{z}_t) - \mathbf{z}_t\|^2 - f(\bar{\mathbf{x}}) - \frac{\mu}{2} \|\bar{\mathbf{x}} - \mathbf{z}_t\|^2 \right] - \langle \mathbf{y}_t, A\bar{\mathbf{x}} - \mathbf{b} \rangle - \frac{\rho}{2} \|A\bar{\mathbf{x}} - \mathbf{b}\|^2 \\
&= \left[ f(\bar{\mathbf{x}}^*(\mathbf{z}_t)) + \frac{\mu}{2} \|\bar{\mathbf{x}}^*(\mathbf{z}_t) - \mathbf{z}_t\|^2 - f(\bar{\mathbf{x}}) - \frac{\mu}{2} \|\bar{\mathbf{x}} - \mathbf{z}_t\|^2 - \frac{\rho}{2} \|A\bar{\mathbf{x}} - \mathbf{b}\|^2 \right] - \langle \mathbf{y}_t, A\bar{\mathbf{x}} - \mathbf{b} \rangle \\
&\geq -2M - \langle \mathbf{y}_t, A\bar{\mathbf{x}} - \mathbf{b} \rangle,
\end{aligned}$$

where the first inequality comes from the definition of Ψ(zt) and

$$d(\mathbf{y}_t, \mathbf{z}_t) = \min_{\mathbf{x} \in X} K(\mathbf{x}, \mathbf{y}, \mathbf{z}).$$

Finally, in the last inequality, we let

$$M = \max_{(\mathbf{x}, \mathbf{z}) \in X \times X} \{ |f(\mathbf{x})| + \frac{\mu}{2} \|\mathbf{x} - \mathbf{z}\|^2 + \frac{\rho}{2} \|A\mathbf{x} - \mathbf{b}\|^2 \}.$$

According to Assumption [4.1\(](#page-6-1)2), there exists a positive r > 0 such that for any direction d ∈ Range(A), we can find a x ∈ X satisfying ∥Ax − b∥ = r and Ax − b has the same direction as d. Because y<sup>t</sup> ∈ Range(A) (by assumption [4.1\(](#page-6-1)3), Range(A) = <sup>R</sup> <sup>m</sup>) we can choose <sup>x</sup>e such that <sup>A</sup>xe <sup>−</sup> <sup>b</sup> is of the same direction as <sup>−</sup>y<sup>t</sup> and <sup>∥</sup>Axe <sup>−</sup> <sup>b</sup><sup>∥</sup> <sup>=</sup> <sup>r</sup>. Then we obtain

$$\Psi(\mathbf{z}_t) - d(\mathbf{y}_t, \mathbf{z}_t) \geq -2M + r\|\mathbf{y}_t\| \implies \|\mathbf{y}_t\| \leq \frac{\Psi(\mathbf{z}_t) - d(\mathbf{y}_t, \mathbf{z}_t) + 2M}{r}, \forall t \in \{0, 1, \dots, T\}.$$

This concludes the proof. ■

## Proof of Theorem [4.3](#page-6-6)

*Proof of Theorem [4.3.](#page-6-6)* First, let M<sup>V</sup> = maxx,z∈X{K(x, 0, z) − 2d(0, z) + 2Ψ(z)} and M<sup>y</sup> > M<sup>V</sup> −MΨ+2M <sup>r</sup> where M<sup>Ψ</sup> is a uniform lower bound of Ψ(zt), for example, f.

Here, We denote the x, y, z generated by Algorithm [2](#page-7-3) at iteration t as xt, yt, z<sup>t</sup> and the output of iteration t + 1 as xt+1, yt+1, zt+1.

If ∥y<sup>t</sup> + η(A<sup>ζ</sup>tx<sup>t</sup> − bζ<sup>t</sup> )∥ ≤ My, then

$$\begin{aligned} & \mathbb{E} [V(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - V(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_{t+1})] \\ & \geq \tilde{c}_\beta \mathbb{E} \|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 + \tilde{c}_\tau \mathbb{E} \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 + \tilde{c}_\eta \mathbb{E} \|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 \\ & \quad - \frac{1}{2} \lambda \tau^2 \sigma_2^2 - \left(1 + \frac{17 \|A\|^4}{\gamma_K^2}\right) \eta^2 L^2 \\ & = \tilde{c}_\beta \mathbb{E} \|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 + \tilde{c}_\tau \mathbb{E} \|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t + \eta(A_{\zeta_t} \mathbf{x}_t - \mathbf{b}_{\zeta_t}), \mathbf{z}_t) - \mathbf{x}_t\|^2 \\ & \quad + \tilde{c}_\eta \mathbb{E} \|A\mathbf{x}^*(\mathbf{y}_t + \eta(A_{\zeta_t} \mathbf{x}_t - \mathbf{b}_{\zeta_t}), \mathbf{z}_t) - \mathbf{b}\|^2 - \frac{1}{2} \lambda \tau^2 \sigma_2^2 - \left(1 + \frac{17 \|A\|^4}{\gamma_K^2}\right) \eta^2 L^2, \end{aligned} \quad (72)$$

where the first inequality use Lemma [B.2](#page-26-0) and the equality comes from the update of yt+1 when ∥yt+η(A<sup>ζ</sup>txt−bζ<sup>t</sup> )∥ ≤ My.

If ∥y<sup>t</sup> + η(A<sup>ζ</sup>tx<sup>t</sup> − bζ<sup>t</sup> )∥ > My, we have yt+1 = 0. Let us use yˆt+1, xˆt+1, zˆt+1 denote the iteration generated with yˆt+1 = y<sup>t</sup> + η(A<sup>ζ</sup>tx<sup>t</sup> − bζ<sup>t</sup> ). Then

$$\begin{aligned}
K(\hat{\mathbf{x}}_{t+1}, \hat{\mathbf{y}}_{t+1}, \hat{\mathbf{z}}_{t+1}) - 2d(\hat{\mathbf{y}}_{t+1}, \hat{\mathbf{z}}_{t+1}) + 2\Psi(\hat{\mathbf{z}}_{t+1}) &\geq \Psi(\hat{\mathbf{z}}_{t+1}) - d(\hat{\mathbf{y}}_{t+1}, \hat{\mathbf{z}}_{t+1}) + \Psi(\hat{\mathbf{z}}_{t+1}) \\
&\geq r\|\hat{\mathbf{y}}_{t+1}\| - 2M + M_\Psi \\
&\geq rM_y - 2M + M_\Psi \\
&\geq M_V \\
&= \max_{\mathbf{x}, \mathbf{z} \in X} \{K(\mathbf{x}, 0, \mathbf{z}) - 2d(0, \mathbf{z}) + 2\Psi(\mathbf{z})\} \\
&\geq K(\mathbf{x}_{t+1}, 0, \mathbf{z}_{t+1}) - 2d(0, \mathbf{z}_{t+1}) + 2\Psi(\mathbf{z}_{t+1}) \\
&= K(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - 2d(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + 2\Psi(\mathbf{z}_{t+1}),
\end{aligned}$$

where the first step used d(yˆt+1, zˆt+1) ≤ K(xˆt+1, yˆt+1, zˆt+1) and the second line uses Prop. [B.3](#page-27-1) and Ψ(zˆt+1) ≥ MΨ. Hence we have

$$\begin{aligned} & \mathbb{E}V(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - \mathbb{E}V(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \\ &= \mathbb{E}[K(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - 2d(\mathbf{y}_t, \mathbf{z}_t) + 2\Psi(\mathbf{z}_t)] - \mathbb{E}[K(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) - 2d(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + 2\Psi(\mathbf{z}_{t+1})] \\ &\geq \mathbb{E}[K(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - 2d(\mathbf{y}_t, \mathbf{z}_t) + 2\Psi(\mathbf{z}_t)] - \mathbb{E}[K(\hat{\mathbf{x}}_{t+1}, \hat{\mathbf{y}}_{t+1}, \hat{\mathbf{z}}_{t+1}) - 2d(\hat{\mathbf{y}}_{t+1}, \hat{\mathbf{z}}_{t+1}) + 2\Psi(\hat{\mathbf{z}}_{t+1})] \\ &\geq \tilde{c}_\beta \mathbb{E}\|\hat{\mathbf{z}}_{t+1} - \mathbf{z}_t\|^2 + \tilde{c}_\tau \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \hat{\mathbf{y}}_{t+1}, \mathbf{z}_t) - \mathbf{x}_t\|^2 + \tilde{c}_\eta \mathbb{E}\|A\mathbf{x}^*(\hat{\mathbf{y}}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 \\ &\quad - \frac{1}{2}\lambda\tau^2\sigma_2^2 - \left(1 + \frac{17\|A\|^4}{\gamma_K^2}\right)\eta^2L^2 \\ &= \tilde{c}_\beta \mathbb{E}\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 + \tilde{c}_\tau \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t + \eta(A_{\zeta_t}\mathbf{x}_t - \mathbf{b}_{\zeta_t}), \mathbf{z}_t) - \mathbf{x}_t\|^2 + \tilde{c}_\eta \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_t + \eta(A_{\zeta_t}\mathbf{x}_t - \mathbf{b}_{\zeta_t}), \mathbf{z}_t) - \mathbf{b}\|^2 \\ &\quad - \frac{1}{2}\lambda\tau^2\sigma_2^2 - \left(1 + \frac{17\|A\|^4}{\gamma_K^2}\right)\eta^2L^2, \end{aligned} \tag{73}$$

where in last inequality, we use Lemma [B.2,](#page-26-0) and in the last equality we use the fact that zˆt+1 = z<sup>t</sup> + β(x<sup>t</sup> − zt) = zt+1, yˆt+1 = y<sup>t</sup> + η(A<sup>ζ</sup>tx<sup>t</sup> − bζ<sup>t</sup> ).

Combining [\(72\)](#page-28-0) and [\(73\)](#page-28-1), we have that

$$\begin{aligned} \mathbb{E}[V_t - V_{t+1}] \\ &\geq \tilde{c}_{\beta} \mathbb{E}\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 + \tilde{c}_{\tau} \mathbb{E}\|\mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t + \eta(A_{\zeta_t} \mathbf{x}_t - \mathbf{b}_{\zeta_t}), \mathbf{z}_t) - \mathbf{x}_t\|^2 + \tilde{c}_{\eta} \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_t + \eta(A_{\zeta_t} \mathbf{x}_t - \mathbf{b}_{\zeta_t}), \mathbf{z}_t) - \mathbf{b}\|^2 \\ &\quad - \frac{1}{2} \lambda \tau^2 \sigma_2^2 - \left(1 + \frac{17\|A\|^4}{\gamma_K^2}\right) \eta^2 L^2, \end{aligned}$$

holds for both ∥y<sup>t</sup> + η(A<sup>ζ</sup>tx<sup>t</sup> − bζ<sup>t</sup> )∥ ≤ M<sup>y</sup> and ∥y<sup>t</sup> + η(A<sup>ζ</sup>tx<sup>t</sup> − bζ<sup>t</sup> )∥ > My, which means it holds for xt+1, yt+1, zt+1 generated by Algorithm [2.](#page-7-3) Then we can telescope as before and the convergence result follows.

We also now sketch the argument for the complexity. We have for the gradient of the Moreau envelope that

$$\begin{aligned} \frac{1}{\mu^2} \|\nabla \Psi(\mathbf{z}_t)\| &= \|\mathbf{z}_t - \bar{\mathbf{x}}^*(\mathbf{z}_t)\| \\ &\leq \|\mathbf{z}_t - \mathbf{x}^*(\mathbf{y}_t + \eta(A_{\zeta_t} \mathbf{x}_t - \mathbf{b}_{\zeta_t}), \mathbf{z}_t)\| + \|\mathbf{x}^*(\mathbf{y}_t + \eta(A_{\zeta_t} \mathbf{x}_t - \mathbf{b}_{\zeta_t}), \mathbf{z}_t) - \bar{\mathbf{x}}^*(\mathbf{z}_t)\| \\ &\leq \|\mathbf{z}_t - \mathbf{x}^*(\mathbf{y}_t + \eta(A_{\zeta_t} \mathbf{x}_t - \mathbf{b}_{\zeta_t}), \mathbf{z}_t)\| + \bar{\sigma} \|A\mathbf{x}^*(\mathbf{y}_t + \eta(A_{\zeta_t} \mathbf{x}_t - \mathbf{b}_{\zeta_t}), \mathbf{z}_s) - \mathbf{b}\| \\ &\leq \|\mathbf{z}_t - \mathbf{x}_t\| + \|\mathbf{x}_t - \mathbf{x}^*(\mathbf{y}_t + \eta(A_{\zeta_t} \mathbf{x}_t - \mathbf{b}_{\zeta_t}), \mathbf{z}_t)\| + \bar{\sigma} \|A\mathbf{x}^*(\mathbf{y}_t + \eta(A_{\zeta_t} \mathbf{x}_t - \mathbf{b}_{\zeta_t}), \mathbf{z}_t) - \mathbf{b}\| \\ &\leq \|\mathbf{z}_t - \mathbf{x}_t\| + \frac{\lambda}{\gamma} \|\mathbf{x}_t - \mathbf{u}^*(\mathbf{x}_t, \mathbf{y}_t + \eta(A_{\zeta_t} \mathbf{x}_t - \mathbf{b}_{\zeta_t}), \mathbf{z}_t)\| + \bar{\sigma} \|A\mathbf{x}^*(\mathbf{y}_t + \eta(A_{\zeta_t} \mathbf{x}_t - \mathbf{b}_{\zeta_t}), \mathbf{z}_t) - \mathbf{b}\|, \end{aligned}$$

where the second line is by triangle inequality, the second inequality is by Lemma [A.12,](#page-24-0) and the fourth line is by triangle inequality and the last estimation is by [\(58\)](#page-22-3).

The rest of the proof for the complexity result proceeds the same as Appendix [A.2](#page-19-0) up to simple changes in the constants, and hence is omitted. ■

## C. Proofs for Section [5](#page-6-5)

Notation. Let us note that we define by <sup>E</sup>ξ<sup>t</sup> the expectation conditioned on all the randomness before ξt.

## C.1. Proofs for Theorem [5.3](#page-7-5)

First, with the idea of the STORM estimator of [Cutkosky & Orabona](#page-9-14) [\(2019\)](#page-9-14), we have the following lemma to control the variance of the stochastic gradient.

Lemma C.1. *(from [\(Cutkosky & Orabona,](#page-9-14) [2019\)](#page-9-14)) Let Assumption [5.2](#page-7-0) hold. We have the estimation of the variance as:*

$$\mathbb{E}\|\widehat{\nabla} f_{t+1} - \nabla f(\mathbf{x}_{t+1})\|^2 \leq (1 - \alpha)^2 \mathbb{E}\|\widehat{\nabla} f_t - \nabla f(\mathbf{x}_t)\|^2 + 3(L_0^2 + L_f^2)\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 3\alpha^2\sigma^2.$$

*Proof.* By the definition of ∇b <sup>f</sup>t+1 in Alg. [3,](#page-7-4) we have

$$\begin{aligned} & \widehat{\nabla} f_{t+1} - \nabla f(\mathbf{x}_{t+1}) \\ &= \nabla f(\mathbf{x}_{t+1}, \xi_{t+1}) + (1 - \alpha)(\widehat{\nabla} f_t - \nabla f(\mathbf{x}_t, \xi_{t+1})) - \nabla f(\mathbf{x}_{t+1}) \\ &= \nabla f(\mathbf{x}_{t+1}, \xi_{t+1}) + (1 - \alpha)(\widehat{\nabla} f_t - \nabla f(\mathbf{x}_t)) + (1 - \alpha)(\nabla f(\mathbf{x}_t) - \nabla f(\mathbf{x}_t, \xi_{t+1})) - \nabla f(\mathbf{x}_{t+1}) \\ &= (1 - \alpha)(\widehat{\nabla} f_t - \nabla f(\mathbf{x}_t)) + (1 - \alpha)(\nabla f(\mathbf{x}_t) - \nabla f(\mathbf{x}_t, \xi_{t+1})) + \nabla f(\mathbf{x}_{t+1}, \xi_{t+1}) - \nabla f(\mathbf{x}_{t+1}), \end{aligned} \quad (74)$$

where in the second equality, we added and subtracted (1 − α)∇f(xt).

Then, we compute the squared norm of [\(74\)](#page-29-0) and expand to get

$$\begin{aligned} & \| \widehat{\nabla} f_{t+1} - \nabla f(\mathbf{x}_{t+1}) \|^2 \\ &= (1 - \alpha)^2 \| \widehat{\nabla} f_t - \nabla f(\mathbf{x}_t) \|^2 + \| (1 - \alpha)(\nabla f(\mathbf{x}_t) - \nabla f(\mathbf{x}_t, \xi_{t+1})) + \nabla f(\mathbf{x}_{t+1}, \xi_{t+1}) - \nabla f(\mathbf{x}_{t+1}) \|^2 \\ &\quad + 2(1 - \alpha)(\widehat{\nabla} f_t - \nabla f(\mathbf{x}_t), (1 - \alpha)(\nabla f(\mathbf{x}_t) - \nabla f(\mathbf{x}_t, \xi_{t+1})) + \nabla f(\mathbf{x}_{t+1}, \xi_{t+1}) - \nabla f(\mathbf{x}_{t+1})). \end{aligned}$$

Next, we take expectation with respect to the randomness of ξt+1 to obtain

$$\begin{aligned} \mathbb{E}_{\xi_{t+1}} \| |\nabla f_{t+1} - \nabla f(\mathbf{x}_{t+1})| \|^2 &= (1 - \alpha)^2 \mathbb{E}_{\xi_{t+1}} \| |\nabla f_t - \nabla f(\mathbf{x}_t)| \|^2 \\ &\quad + \mathbb{E}_{\xi_{t+1}} \| (1 - \alpha) (\nabla f(\mathbf{x}_t) - \nabla f(\mathbf{x}_t, \xi_{t+1})) + \nabla f(\mathbf{x}_{t+1}, \xi_{t+1}) - \nabla f(\mathbf{x}_{t+1}) \|^2, \quad (75) \end{aligned}$$

which is due to ∇b <sup>f</sup><sup>t</sup> − ∇f(xt) being independent of <sup>ξ</sup>t+1, as well as

$$\mathbb{E}_{\xi_{t+1}}[\nabla f(\mathbf{x}_t) - \nabla f(\mathbf{x}_t, \xi_{t+1})] = 0, \quad \mathbb{E}_{\xi_{t+1}}[\nabla f(\mathbf{x}_{t+1}, \xi_{t+1}) - \nabla f(\mathbf{x}_{t+1})] = 0.$$

Finally, we estimate the last term on the right-hand side of [\(75\)](#page-30-0):

$$\begin{aligned} & \mathbb{E}_{\xi_{t+1}} \| (1 - \alpha)(\nabla f(\mathbf{x}_t) - \nabla f(\mathbf{x}_t, \xi_{t+1})) + \nabla f(\mathbf{x}_{t+1}, \xi_{t+1}) - \nabla f(\mathbf{x}_{t+1}) \|^2 \\ &= \mathbb{E}_{\xi_{t+1}} \| \nabla f(\mathbf{x}_{t+1}, \xi_{t+1}) - \nabla f(\mathbf{x}_t, \xi_{t+1}) + \nabla f(\mathbf{x}_t) - \nabla f(\mathbf{x}_{t+1}) + \alpha(f(\mathbf{x}_t, \xi_{t+1}) - \nabla f(\mathbf{x}_t)) \|^2 \\ &\leq 3\mathbb{E}_{\xi_{t+1}} [\| \nabla f(\mathbf{x}_{t+1}, \xi_{t+1}) - \nabla f(\mathbf{x}_t, \xi_{t+1}) \|^2 + \| \nabla f(\mathbf{x}_t) - \nabla f(\mathbf{x}_{t+1}) \|^2 + \| \alpha(\nabla f(\mathbf{x}_t, \xi_{t+1}) - \nabla f(\mathbf{x}_t)) \|^2] \\ &\leq 3L_0^2 \| \mathbf{x}_{t+1} - \mathbf{x}_t \|^2 + 3L_f^2 \| \mathbf{x}_t - \mathbf{x}_{t+1} \|^2 + 3\alpha^2 \sigma^2, \end{aligned}$$

where in the first equality, we rearrange the terms, and in the first inequality, we use Young's inequality. In the second inequality, we use Assumption [5.2,](#page-7-0) L<sup>f</sup> -smoothness of f(x) and <sup>E</sup>ξ∥∇f(x, ξ) − ∇f(x)∥ <sup>2</sup> ≤ σ 2 . We use this estimation in [\(75\)](#page-30-0) and take total expectation to get the result. ■

Let us recall from [\(18\)](#page-7-6) that

$$\bar{V}_t = K(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - 2d(\mathbf{y}_t, \mathbf{z}_t) + 2\Psi(\mathbf{z}_t) + \frac{1}{48(L_0^2 + L_f^2)\tau} \|\hat{\nabla} f_t - \nabla f(\mathbf{x}_t)\|^2, \quad (76)$$

where (as [\(22\)](#page-11-6))

$$K(\mathbf{x}, \mathbf{y}, \mathbf{z}) = L_\rho(\mathbf{x}, \mathbf{y}) + \frac{\mu}{2} \|\mathbf{x} - \mathbf{z}\|^2 \quad (77)$$

and x 7→ ∇K(x, y, z) is LK-Lipschitz with L<sup>K</sup> = L<sup>f</sup> + ρ∥A∥ + µ (see also Fact [A.1\)](#page-11-4).

We already have the descent-type lemma of d(yt, zt) and Ψ(zt) in Lemma [A.8,](#page-16-1) and only need to show the descent-type lemma of K(xt, yt, zt). We write K(xt, yt, zt) − K(xt+1, yt+1, zt+1) as:

$$[K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - K(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_t)] + [K(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)] + [K(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_t) - K(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_{t+1})]$$

and lower bound each term separately in the following lemmas.

Lemma C.2. *Let Assumption [1.1](#page-3-2) hold. For the iterates generated by Algorithm [3,](#page-7-4) we have*

$$K(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_t) - K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) \leq \frac{\tau}{2} \|\nabla f(\mathbf{x}_t) - \widehat{\nabla} f_t\|^2 - \left( \frac{1}{2\tau} - \frac{L_K}{2} \right) \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2.$$

*Proof.* We have, by smoothness of K(·, yt+1, zt):

$$K(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_t) \leq K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) + \langle \nabla_{\mathbf{x}} K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{x}_{t+1} - \mathbf{x}_t \rangle + \frac{L_K}{2} \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2. \quad (78)$$

We estimate the inner product here as

$$\begin{aligned} \langle \nabla_{\mathbf{x}} K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{x}_{t+1} - \mathbf{x}_t \rangle &= \langle G(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{x}_{t+1} - \mathbf{x}_t \rangle \\ &\quad + \langle \nabla_{\mathbf{x}} K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - G(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{x}_{t+1} - \mathbf{x}_t \rangle. \end{aligned} \quad (79)$$

We first have, in view of Alg. [3](#page-7-4) that

$$\nabla_{\mathbf{x}} K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) - G(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) = \nabla f(\mathbf{x}_t) - \hat{\nabla} f_t.$$

The definition of xt+1 in Alg. [3](#page-7-4) gives

$$\langle \mathbf{x}_{t+1} - \mathbf{x}_t + \tau G(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{x}_t - \mathbf{x}_{t+1} \rangle \geq 0 \iff \langle G(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{x}_{t+1} - \mathbf{x}_t \rangle \leq -\frac{1}{\tau} \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2. \quad (80)$$

Using ⟨∇xK(xt, yt+1, zt) − G(xt, yt+1, zt), xt+1 − xt⟩ ≤ <sup>τ</sup> 2 ∥∇f(xt) − ∇b <sup>f</sup>t∥ <sup>2</sup> + 1 2τ ∥xt+1 − xt∥ 2 along with [\(80\)](#page-30-1) in [\(79\)](#page-30-2), we have

$$\langle \nabla_{\mathbf{x}} K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t), \mathbf{x}_{t+1} - \mathbf{x}_t \rangle \leq \frac{\tau}{2} \|\nabla f(\mathbf{x}_t) - \hat{\nabla} f_t\|^2 - \frac{1}{2\tau} \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2.$$

Then the result follows after substituting the last estimate in [\(78\)](#page-30-3). ■

Lemma C.3. *Let Assumption [1.1](#page-3-2) hold. For the iterates generated by Algorithm [3,](#page-7-4) we have*

$$K(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - K(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \geq -\eta \|A\mathbf{x}_t - \mathbf{b}\|^2 + \left(\frac{\mu}{\beta} - \frac{3\mu}{4}\right) \|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 - \frac{\tau}{2} \|\nabla f(\mathbf{x}_t) - \hat{\nabla} f_t\|^2 + \left(\frac{1}{2\tau} - \frac{L_K}{2} - \mu\right) \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2. \quad (81)$$

*Proof.* First, from the definition of K in [\(22\)](#page-11-6), we have

$$K(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t) = -\eta \|A\mathbf{x}_t - \mathbf{b}\|^2.$$

Moreover, it follows that

$$\begin{aligned}
& K(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_t) - K(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \\
&= \frac{\mu}{2} (\|\mathbf{x}_{t+1} - \mathbf{z}_t\|^2 - \|\mathbf{x}_{t+1} - \mathbf{z}_{t+1}\|^2) \\
&= \frac{\mu}{2} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, 2\mathbf{x}_{t+1} - \mathbf{z}_t - \mathbf{z}_{t+1} \rangle \\
&= \frac{\mu}{2} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, 2\mathbf{x}_{t+1} - 2\mathbf{x}_t + 2\mathbf{x}_t - 2\mathbf{z}_t + \mathbf{z}_t - \mathbf{z}_{t+1} \rangle \\
&= \frac{\mu}{2} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, 2\mathbf{x}_{t+1} - 2\mathbf{x}_t \rangle + \frac{\mu}{2} \langle \mathbf{z}_{t+1} - \mathbf{z}_t, 2\mathbf{x}_t - 2\mathbf{z}_t \rangle - \frac{\mu}{2} \|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 \\
&\geq -\frac{\mu}{4} \|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 - \mu \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + \frac{\mu}{\beta} \|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 - \frac{\mu}{2} \|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2,
\end{aligned}$$

where the first equality comes from the definition of K. In the last inequality, we use ⟨a, b⟩ ≥ −<sup>1</sup> 4 ∥a∥ <sup>2</sup> − ∥b∥ 2 and x<sup>t</sup> − z<sup>t</sup> = zt+1−z<sup>t</sup> β by the definition of zt+1 in Algorithm [3.](#page-7-4)

Fanally combining the above two results with Lemma [C.2](#page-30-4) and combining like-terms yields the claim. ■

We next follow with a detailed restatement of Lemma [5.3](#page-7-5) and its proof.

Lemma C.4 (cf. Lemma [5.3\)](#page-7-5). *Under Assumption [1.1](#page-3-2) and Assumption [5.2,](#page-7-0) with the parameters chosen as:*

$$\begin{aligned} \mu &= \max\{2, 4L_f\}, \quad \tau \leq \min \left\{ \frac{1}{8L_K + 16\mu}, \frac{1}{\sqrt{48(L_0^2 + L_f^2)}} \right\} \\ \eta &= \min \left\{ \frac{(\mu - L_f)^2 \tau}{8\|A\|^2}, \frac{2\mu + \rho\|A\|}{4\|A\|^4}, \frac{\tau}{200\|A\|^2}, \frac{\tau(2\mu + \rho\|A\|^2)}{20\|A\|^2} \right\}, \\ \beta &= \min \left\{ \frac{\tau}{100}, \frac{1}{50}, \frac{\eta}{36\mu\bar{\sigma}^2} \right\}, \\ \alpha &= 48(L_0^2 + L_f^2)\tau^2, \end{aligned} \tag{82}$$

*where* L<sup>K</sup> = L<sup>f</sup> + ρ∥A∥ + µ*,* σ¯ *is defined in Lemma [A.12,](#page-24-0) we have*

$$\begin{aligned} \mathbb{E}\bar{V}_t - \mathbb{E}\bar{V}_{t+1} &\geq \frac{\mu}{2\beta} \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 + \frac{1}{8\tau} \mathbb{E}\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 + \frac{\eta}{2} \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - b\|^2 + \frac{\tau}{4} \mathbb{E}\|\hat{\nabla} f_t - \nabla f(\mathbf{x}_t)\|^2 \\ &\quad - 144(L_0^2 + L_f^2)\sigma^2\tau^3. \end{aligned} \quad (83)$$

*Proof.* We denote

$$V_t = K(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - 2d(\mathbf{y}_t, \mathbf{z}_t) + 2\Psi(\mathbf{z}_t). \quad (84)$$

Joining [\(81\)](#page-31-2) with Lemma [A.8](#page-16-1) (since this lemma only uses the update rules of yt+1, zt+1 that is common in Alg. [1](#page-5-0) and Alg. [3\)](#page-7-4), we have

$$\begin{aligned} \mathbb{E}V_t - \mathbb{E}V_{t+1} &\geq -\eta\mathbb{E}\|A\mathbf{x}_t - \mathbf{b}\|^2 + \left(\frac{\mu}{\beta} - \frac{3\mu}{4}\right)\mathbb{E}\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 \\ &\quad - \frac{\tau}{2}\mathbb{E}\|\nabla f(\mathbf{x}_t) - \hat{\nabla} f_t\|^2 + \left(\frac{1}{2\tau} - \frac{L_K}{2} - \mu\right)\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 \\ &\quad + 2\eta\mathbb{E}\langle A\mathbf{x}_t - \mathbf{b}, A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b} \rangle + \mu\mathbb{E}\langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_{t+1} + \mathbf{z}_t - 2\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle \\ &\quad - 2\mu\mathbb{E}\langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_t - \bar{\mathbf{x}}^*(\mathbf{z}_t) \rangle - \frac{\mu}{\sigma_4}\mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2. \end{aligned} \quad (85)$$

First, let us combine the first and fifth terms on the right-hand side to obtain

$$-\eta\|A\mathbf{x}_t - \mathbf{b}\|^2 + 2\eta\langle A\mathbf{x}_t - \mathbf{b}, A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b} \rangle = -\eta\|A\mathbf{x}_t - A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 + \eta\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2. \quad (86)$$

Next, we combine the sixth and seventh terms on the right-hand side of [\(85\)](#page-32-0) to get

$$\begin{aligned} & \mu \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_{t+1} + \mathbf{z}_t - 2\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle - 2\mu \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_t - \bar{\mathbf{x}}^*(\mathbf{z}_t) \rangle \\ & = \mu \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_{t+1} - \mathbf{z}_t - 2\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + 2\bar{\mathbf{x}}^*(\mathbf{z}_t) \rangle \\ & = \mu \|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 + 2\mu \langle \mathbf{z}_{t+1} - \mathbf{z}_t, -\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + \bar{\mathbf{x}}^*(\mathbf{z}_t) \rangle. \end{aligned} \quad (87)$$

We now single out the inner product in the last equality and estimate it by adding and subtracting x ∗ (yt+1, zt) in the second argument of the inner product:

$$\begin{aligned} & 2\mu\langle \mathbf{z}_{t+1} - \mathbf{z}_t, -\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + \bar{\mathbf{x}}^*(\mathbf{z}_t) \rangle \\ &= 2\mu\langle \mathbf{z}_{t+1} - \mathbf{z}_t, -\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) + \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) \rangle + 2\mu\langle \mathbf{z}_{t+1} - \mathbf{z}_t, -\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) + \bar{\mathbf{x}}^*(\mathbf{z}_t) \rangle \\ &\geq -\mu\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 - \mu\|\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1})\|^2 - \frac{\mu}{\zeta}\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 - \mu\zeta\|\bar{\mathbf{x}}^*(\mathbf{z}_t) - \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\|, \quad (88) \end{aligned}$$

for any ζ, where we used Young's inequality twice. Then, we plug this into [\(87\)](#page-32-1) to obtain

$$\begin{aligned} & \mu \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_{t+1} + \mathbf{z}_t - 2\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_{t+1}) \rangle - 2\mu \langle \mathbf{z}_{t+1} - \mathbf{z}_t, \mathbf{z}_t - \bar{\mathbf{x}}^*(\mathbf{z}_t) \rangle \\ & \geq -\frac{\mu}{\sigma_4^2} \|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 - \frac{\mu}{\zeta} \|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 - \mu \zeta^* (\bar{\mathbf{x}}^*(\mathbf{z}_t) - \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)) \|^2, \end{aligned} \quad (89)$$

where we use [\(62\)](#page-22-4) to bound the second term on the right-hand side of [\(88\)](#page-32-2), with σ<sup>4</sup> being as [\(25\)](#page-11-0).

Then we use [\(86\)](#page-32-3) and [\(89\)](#page-32-4) in [\(85\)](#page-32-0) to obtain

$$\begin{aligned} \mathbb{E}V_t - \mathbb{E}V_{t+1} &\geq \left( \frac{\mu}{\beta} - \frac{3\mu}{4} \right) \mathbb{E}\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 - \frac{\tau}{2} \mathbb{E}\|\nabla f(\mathbf{x}_t) - \widehat{\nabla} f_t\|^2 + \left( \frac{1}{2\tau} - \frac{L_K}{2} - \mu \right) \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 \\ &\quad - \eta \mathbb{E}\|A\mathbf{x}_t - A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 + \eta \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 \\ &\quad - \frac{\mu}{\sigma_4^2} \mathbb{E}\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 - \frac{\mu}{\zeta} \mathbb{E}\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 - \mu\zeta \|\bar{\mathbf{x}}^*(\mathbf{z}_t) - \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\|^2 - \frac{\mu}{\sigma_4} \mathbb{E}\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 \\ &\geq \left( \frac{\mu}{\beta} - \frac{3\mu}{4} - \frac{\mu}{\sigma_4^2} - \frac{\mu}{\zeta} - \frac{\mu}{\sigma_4} \right) \mathbb{E}\|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 \\ &\quad - \frac{\tau}{2} \mathbb{E}\|\nabla f(\mathbf{x}_t) - \widehat{\nabla} f_t\|^2 - \frac{2\eta \|A\|^2}{(\mu - L_f)^2} \mathbb{E}\|\nabla f(\mathbf{x}_t) - \widehat{\nabla} f_t\|^2 \\ &\quad + \left( \frac{1}{2\tau} - \frac{L_K}{2} - \mu - \eta \|A\|^2 \frac{2}{\tau(\mu - L_f)^2} \right) \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 \\ &\quad + (\eta - \mu\zeta\bar{\sigma}^2) \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2, \end{aligned} \tag{90}$$

where in the last inequality, we use Lem. [C.6](#page-34-0) and Lem. [A.12,](#page-24-0) then combine the like-terms.

Then we need to estimate the coefficients of each terms in the above inequality. Let us recall from [\(25\)](#page-11-0) that σ<sup>4</sup> = µ−L<sup>f</sup> <sup>µ</sup> > and let ζ = 6β.

We now estimate the coefficient of <sup>E</sup>∥z<sup>t</sup> − zt+1∥ 2 in [\(90\)](#page-32-5). First, by σ<sup>4</sup> > 1 2 , we have <sup>µ</sup> σ 4 ≤ 4µ and <sup>µ</sup> σ<sup>4</sup> ≤ 2µ. By also using ζ = 6β, we have:

The coefficient of 
$$\mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 \geq \frac{\mu}{\beta} - \frac{3\mu}{4} - 4\mu - \frac{\mu}{6\beta} - 2\mu$$
.

Using β ≤ 1/50, we obtain ( 3 <sup>4</sup> + 4 + 2)µ ≤ µ 5β , then we estimate:

The coefficient of 
$$\mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 \geq \frac{\mu}{\beta} - \frac{\mu}{5\beta} - \frac{\mu}{6\beta} \geq \frac{\mu}{2\beta}$$
.

We move on to estimating the coefficient of <sup>E</sup>∥x<sup>t</sup> − xt+1∥ 2 in [\(90\)](#page-32-5). With η ≤ (µ−L<sup>f</sup> ) 2 τ <sup>8</sup>∥A∥<sup>2</sup> , we have 2η∥A∥ 2 1 τ <sup>2</sup>(µ−L<sup>f</sup> ) <sup>2</sup> ≤ 1 4τ , we have:

The coefficient of 
$$\mathbb{E}\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 \geq \frac{1}{4\tau} - \frac{L_K}{2} - \mu$$
.

Last, we work on the coefficient of <sup>E</sup>∥Ax ∗ (yt+1, zt)−b∥ 2 in [\(90\)](#page-32-5). Because ζ = 6β, it follows that η −µζσ¯ <sup>2</sup> = η −6µβσ¯ 2 . With β ≤ η 36µσ¯ <sup>2</sup> , we have 6µβσ¯ <sup>2</sup> ≤ η 6 , then we estimate:

the coefficient of 
$$\mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 \geq \eta - \frac{\eta}{6} \geq \frac{\eta}{2}$$
.

Next, we estimate the coefficient of <sup>E</sup>∥∇f(xt) − ∇b <sup>f</sup>t∥ 2 . With η ≤ (µ−L<sup>f</sup> ) 2 τ <sup>8</sup>∥A∥<sup>2</sup> , we have − τ <sup>2</sup> − 2η∥A∥ (µ−L<sup>f</sup> ) <sup>2</sup> ≥ −<sup>3</sup> 4 τ . Finally, we have

$$\begin{aligned} &\mathbb{E}V_t - \mathbb{E}V_{t+1} \\ &\geq \frac{\mu}{2\beta} \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 + \left( \frac{1}{4\tau} - \frac{L_K}{2} - \mu \right) \mathbb{E}\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 + \frac{\eta}{2} \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 - \frac{3\tau}{4} \mathbb{E}\|\nabla f(\mathbf{x}_t) - \widehat{\nabla} f_t\|^2 \\ &= \frac{\mu}{2\beta} \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 + \left( \frac{1}{4\tau} - \frac{L_K}{2} - \mu \right) \mathbb{E}\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 + \frac{\eta}{2} \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 \\ &\quad + \frac{\tau}{4} \mathbb{E}\|\nabla f(\mathbf{x}_t) - \widehat{\nabla} f_t\|^2 - \tau \mathbb{E}\|\nabla f(\mathbf{x}_t) - \widehat{\nabla} f_t\|^2. \end{aligned}$$

Then recalling Lemma [C.1](#page-29-1) and assuming 0 < α ≤ 1, we have

$$\mathbb{E}\|\widehat{\nabla} f_{t+1} - \nabla f(\mathbf{x}_{t+1})\|^2 \leq (1 - \alpha)\mathbb{E}\|\widehat{\nabla} f_t - \nabla f(\mathbf{x}_t)\|^2 + 3(L_0^2 + L_f^2)\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 3\alpha^2\sigma^2. \quad (91)$$

We multiply [\(91\)](#page-33-0) by <sup>τ</sup> α , rearrange, and plug into [\(91\)](#page-33-0), to get

$$\begin{aligned} \mathbb{E}V_t - \mathbb{E}V_{t+1} &\geq \frac{\mu}{2\beta} \mathbb{E}\|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 + \left( \frac{1}{4\tau} - \frac{L_K}{2} - \mu \right) \mathbb{E}\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 \\ &\quad + \frac{\eta}{2} \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 + \frac{\tau}{4} \mathbb{E}\|\nabla f(\mathbf{x}_t) - \hat{\nabla} f_t\|^2 \\ &\quad + \frac{\tau}{\alpha} \mathbb{E}\|\hat{\nabla} f_{t+1} - \nabla f(\mathbf{x}_{t+1})\|^2 - \frac{\tau}{\alpha} \mathbb{E}\|\hat{\nabla} f_t - \nabla f(\mathbf{x}_t)\|^2 \\ &\quad - \frac{3(L_0^2 + L_f^2)\tau}{\alpha} \mathbb{E}\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 - 3\alpha\sigma^2\tau. \end{aligned} \quad (92)$$

Because α = 48(L 2 <sup>0</sup> + L 2 f )τ 2 and <sup>τ</sup> <sup>≤</sup> min 8LK+16µ , √ 1 48(L<sup>2</sup> 0+L<sup>2</sup> f ) , we obtain

$$\frac{L_K}{2} + \mu \leq \frac{1}{16\tau}, \quad \frac{3(L_0^2 + L_f^2)\tau}{\alpha} = \frac{1}{16\tau}.$$

Hence, we have

$$\begin{aligned} \mathbb{E} V_t - \mathbb{E} V_{t+1} &\geq \frac{\mu}{2\beta} \mathbb{E} \|\mathbf{z}_t - \mathbf{z}_{t+1}\|^2 + \frac{1}{8\tau} \mathbb{E} \|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 + \frac{\eta}{2} \mathbb{E} \|\mathbf{A}\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 + \frac{\tau}{4} \mathbb{E} \|\nabla f(\mathbf{x}_t) - \hat{\nabla} f_t\|^2 \\ &\quad + \frac{1}{48(L_0^2 + L_f^2)\tau} \mathbb{E} \|\hat{\nabla} f_{t+1} - \nabla f(\mathbf{x}_{t+1})\|^2 - \frac{1}{48(L_0^2 + L_f^2)\tau} \mathbb{E} \|\hat{\nabla} f_t - \nabla f(\mathbf{x}_t)\|^2 - 144(L_0^2 + L_f^2)\sigma^2\tau^3. \end{aligned}$$

Finally, we move <sup>1</sup> 48(L<sup>2</sup> 0+L<sup>2</sup> f )τ <sup>E</sup>∥∇b <sup>f</sup>t+1 − ∇f(xt+1)∥ <sup>2</sup> − 48(L<sup>2</sup> 0+L<sup>2</sup> f )τ <sup>E</sup>∥∇b <sup>f</sup><sup>t</sup> − ∇f(xt)∥ 2 to the left-hand side of the above inequality and use the definition of V¯ <sup>t</sup> in [\(18\)](#page-7-6) to get the desired result. ■

## C.2. Proofs for Theorem [5.4](#page-7-7)

First, we need two lemmas for the error bound that helps us analyze the sample complexity that we include for being self-contained.

Lemma C.5. *[\(Zhang & Luo,](#page-10-1) [2020,](#page-10-1) Lemma 3.10) Under Assumption [1.1,](#page-3-2) we have*

$$\|\mathbf{x} - \text{proj}_X(\mathbf{x} - \tau \nabla K(\mathbf{x}, \mathbf{y}, \mathbf{z}))\| \geq \frac{\tau(\mu - L_f)}{2} \|\mathbf{x} - \mathbf{x}^*(\mathbf{y}, \mathbf{z})\|,$$

*where* K(x, y, z) = Lρ(x, y) + <sup>µ</sup> 2 ∥x − z∥ 2 *, and* x ∗ (y, z) = arg minx∈<sup>X</sup> K(x, y, z)*.*

*Proof.* First, we denote that xˆ = x − projX(x − τ∇K(x, y, z)), then by the definition of x ∗ (y, z), we have

$$\langle \mathbf{x} - \hat{\mathbf{x}} - \mathbf{x}^*(\mathbf{y}, \mathbf{z}), \tau \nabla K(\mathbf{x}^*(\mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) \rangle \geq 0,$$

where we use the fact that x − xˆ ∈ X.

Then by the definition of projection (that is, z¯ = projX(z) ⇐⇒ ⟨z¯ − z, t − z¯⟩ ≥ 0 ∀t ∈ X), the definition of xˆ, and x ∗ (y, z) ∈ X, we have

$$\begin{aligned} \langle \mathbf{x}^*(\mathbf{y}, \mathbf{z}) - \text{proj}_X(\mathbf{x} - \tau \nabla K(\mathbf{x}, \mathbf{y}, \mathbf{z})), \mathbf{x} - \tau \nabla K(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \text{proj}_X(\mathbf{x} - \tau \nabla K(\mathbf{x}, \mathbf{y}, \mathbf{z})) \rangle \\ = \langle \mathbf{x}^*(\mathbf{y}, \mathbf{z}) - (\mathbf{x} - \hat{\mathbf{x}}), -\tau \nabla K(\mathbf{x}, \mathbf{y}, \mathbf{z}) + \hat{\mathbf{x}} \rangle \leq 0. \end{aligned}$$

Combining above two inequalities and rearranging terms, we have

$$\begin{aligned} &\langle \mathbf{x} - \mathbf{x}^*(\mathbf{y}, \mathbf{z}), \tau \nabla K(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \tau \nabla K(\mathbf{x}^*(\mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) \rangle \\ &\leq \langle \hat{\mathbf{x}}, \tau \nabla K(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \tau \nabla K(\mathbf{x}^*(\mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) + \mathbf{x} - \mathbf{x}^*(\mathbf{y}, \mathbf{z}) \rangle - \|\hat{\mathbf{x}}\|^2 \\ &\leq \|\hat{\mathbf{x}}\| \|\tau \nabla K(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \tau \nabla K(\mathbf{x}^*(\mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) + \mathbf{x} - \mathbf{x}^*(\mathbf{y}, \mathbf{z})\| \\ &\leq \|\hat{\mathbf{x}}\| (\tau L_K + 1) \|\mathbf{x} - \mathbf{x}^*(\mathbf{y}, \mathbf{z})\| \\ &\leq 2\|\hat{\mathbf{x}}\| \|\mathbf{x} - \mathbf{x}^*(\mathbf{y}, \mathbf{z})\|, \end{aligned}$$

where in the second inequality we use the Cauchy-Schwarz inequality and in the last inequality we use the Lipschitz continuity of ∇K with respect to x.

By K(x, y, z) being (µ − L<sup>f</sup> )-strongly convex with respect to x (see Fact [A.1\)](#page-11-4), we have

$$\langle \mathbf{x} - \mathbf{x}^*(\mathbf{y}, \mathbf{z}), \tau \nabla K(\mathbf{x}, \mathbf{y}, \mathbf{z}) - \tau \nabla K(\mathbf{x}^*(\mathbf{y}, \mathbf{z}), \mathbf{y}, \mathbf{z}) \rangle \geq \tau(\mu - L_f) \|\mathbf{x} - \mathbf{x}^*(\mathbf{y}, \mathbf{z})\|^2.$$

Then, the desired result follows by combining the above two inequalities and using the definition of xˆ = x − projX(x − τ∇K(x, y, z)). ■

With the next lemma, we proceed to prove that ∥x<sup>t</sup> − x ∗ (yt+1, zt)∥ is bounded by a combination of ∥x<sup>t</sup> − xt+1∥ and ∥∇b <sup>f</sup><sup>t</sup> − ∇f(xt)∥.

Lemma C.6. *Under Assumption [1.1,](#page-3-2) for the iterates generated by Algorithm [3](#page-7-4) we have*

$$\|\mathbf{x}_t - \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\| \leq \frac{2}{\tau(\mu - L_f)} \|\mathbf{x}_t - \mathbf{x}_{t+1}\| + \frac{2}{(\mu - L_f)} \|\widehat{\nabla} f_t - \nabla f(\mathbf{x}_t)\|.$$

*Proof.* Taking x, y, z as xt, yt+1, z<sup>t</sup> in Lemma [C.5,](#page-34-1) we have

$$\begin{aligned} \|\mathbf{x}_t - \mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t)\| &\leq \frac{2}{\tau(\mu - L_f)} \|\mathbf{x}_t - \text{proj}_X(\mathbf{x}_t - \tau \nabla K(\mathbf{x}, \mathbf{y}_{t+1}, \mathbf{z}_t))\| \\ &\leq \frac{2}{\tau(\mu - L_f)} \|\mathbf{x}_t - \text{proj}_X(\mathbf{x}_t - \tau G(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t))\| \\ &\quad + \frac{2}{\tau(\mu - L_f)} \|\text{proj}_X(\mathbf{x}_t - \tau \nabla K(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t)) - \text{proj}_X(\mathbf{x}_t - \tau G(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{z}_t))\| \\ &\leq \frac{2}{\tau(\mu - L_f)} \|\mathbf{x}_t - \mathbf{x}_{t+1}\| + \frac{2}{(\mu - L_f)} \|\hat{\nabla} f_t - \nabla f(\mathbf{x}_t)\|, \end{aligned}$$

where the second inequality comes form triangle inequality and the last inequality comes from the fact that proj<sup>X</sup> is nonexpansive and ∇K(xt, <sup>y</sup>t+1, <sup>z</sup>t) − <sup>G</sup>(xt, <sup>y</sup>t+1, <sup>z</sup>t) = ∇f(xt) − ∇b <sup>f</sup>t. ■

We now continue with the proof of Theorem [5.4.](#page-7-7)

*Proof of Theorem [5.4.](#page-7-7)* Because zt+1 − z<sup>t</sup> = β(x<sup>t</sup> − zt), µβ <sup>2</sup> = Θ(<sup>τ</sup> ) and <sup>η</sup> <sup>2</sup> = Θ(τ ) in view of Lemma [C.4,](#page-31-0) hence there exists a constant C such that we get from [\(83\)](#page-31-3):

$$\begin{aligned} \mathbb{E}\bar{V}_t - \mathbb{E}\bar{V}_{t+1} &\geq C\tau\{\mathbb{E}\|\mathbf{x}_t - \mathbf{z}_t\|^2 + \mathbb{E}\|\tau^{-1}(\mathbf{x}_t - \mathbf{x}_{t+1})\|^2 + \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 + \mathbb{E}\|\nabla f(\mathbf{x}_t) - \hat{\nabla} f\|^2\} \\ &\quad - 144(L_0^2 + L_f^2) \sigma^2 \tau^3. \end{aligned} \quad (93)$$

Then, summing up [\(93\)](#page-35-0) over t = 0, 1, . . . , T − 1, we have

$$\begin{aligned} \bar{V}_0 - \mathbb{E}\bar{V}_T &\geq \sum_{t=0}^{T-1} C\tau\{\mathbb{E}\|\mathbf{x}_t - \mathbf{z}_t\|^2 + \mathbb{E}\|\tau^{-1}(\mathbf{x}_t - \mathbf{x}_{t+1})\|^2 + \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 + \mathbb{E}\|\nabla f(\mathbf{x}_t) - \hat{\nabla} f_t\|^2\} \\ &\quad - 144 (L_0^2 + L_f^2) \sigma^2 \tau^3 T. \end{aligned} \quad (94)$$

From the definition, we have K(x, y, z) ≥ d(y, z) (since d(y, z) = minx∈<sup>X</sup> K(x, y, z)) and Ψ(z) ≥ d(y, z) (see also Lemma [A.13\)](#page-24-3), then

$$V_t = K(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - 2d(\mathbf{y}_t, \mathbf{z}_t) + 2\Psi(\mathbf{z}_t) \geq \Psi(\mathbf{z}_t) \geq \underline{f}.$$

Consequently, we have

$$\bar{V}_t = K(\mathbf{x}_t, \mathbf{y}_t, \mathbf{z}_t) - 2d(\mathbf{y}_t, \mathbf{z}_t) + 2\Psi(\mathbf{z}_t) + \frac{1}{48(L_0^2 + L_f^2)\tau} \mathbb{E}\|\hat{\nabla} f_t - \nabla f(\mathbf{x}_t)\|^2 \geq \underline{f}. \quad (95)$$

Let τ = T −1/3 and use mini-batch in the initial step where we will have <sup>E</sup>∥∇b <sup>f</sup><sup>0</sup> − ∇f(x0)∥ <sup>2</sup> ≤ T <sup>−</sup>1/<sup>3</sup>σ 2 (by the definition of ∇b <sup>f</sup><sup>0</sup> and a standard computation), then

$$\begin{aligned} \bar{V}_0 &= K(\mathbf{x}_0, \mathbf{y}_0, \mathbf{z}_0) - 2d(\mathbf{y}_0, \mathbf{z}_0) + 2\Psi(\mathbf{z}_0) + \frac{1}{48(L_0^2 + L_f^2)\tau} \mathbb{E}\|\hat{\nabla} f_0 - \nabla f(\mathbf{x}_0)\|^2 \\ &\leq K(\mathbf{x}_0, \mathbf{y}_0, \mathbf{z}_0) - 2d(\mathbf{y}_0, \mathbf{z}_0) + 2\Psi(\mathbf{z}_0) + \frac{\sigma^2}{48(L_0^2 + L_f^2)}, \end{aligned} \quad (96)$$

where the right-hand is proportional to a constant independent of T, we denote it as C0.

Combining [\(94\)](#page-35-1) with [\(95\)](#page-35-2) and [\(96\)](#page-35-3), we have

$$\begin{aligned} & \frac{1}{T} \sum_{t=0}^{T-1} C\{\mathbb{E}\|\mathbf{x}_t - \mathbf{z}_t\|^2 + \mathbb{E}\|\tau^{-1}(\mathbf{x}_t - \mathbf{x}_{t+1})\|^2 + \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{t+1}, \mathbf{z}_t) - \mathbf{b}\|^2 + \mathbb{E}\|\nabla f(\mathbf{x}_t) - \hat{\nabla} f\|^2\} \\ & \leq T^{-2/3} (C_0 - \underline{f} + 144(L_0^2 + L_f^2)\sigma^2). \end{aligned} \quad (97)$$

Then, for index s selected uniformly at random from {0, 1, ..., T − 1}, we have

$$\begin{aligned}\mathbb{E}\|\mathbf{x}_s - \mathbf{z}_s\|^2 &= O(T^{-2/3}), \quad \mathbb{E}\|\tau^{-1}(\mathbf{x}_s - \mathbf{x}_{s+1})\|^2 = O(T^{-2/3}), \\ \mathbb{E}\|A\mathbf{x}^*(\mathbf{y}_{s+1}, \mathbf{z}_s) - \mathbf{b}\|^2 &= O(T^{-2/3}), \quad \mathbb{E}\|\nabla f(\mathbf{x}_t) - \hat{\nabla} f_t\|^2 = O(T^{-2/3}).\end{aligned}\tag{98}$$

According to Algorithm [3,](#page-7-4) we have

$$\mathbf{x}_{s+1} = \arg \min_{\mathbf{x}} \left\{ \left( G(\mathbf{x}_s, \mathbf{y}_{s+1}, \mathbf{z}_s), \mathbf{x} - \mathbf{x}^s \right) + \frac{1}{\tau} \|\mathbf{x} - \mathbf{x}_s\|^2 + \partial I_X(\mathbf{x}) \right\}.$$

By the definition of xs+1, we have

$$0 \in G(\mathbf{x}_s, \mathbf{y}_{s+1}, \mathbf{z}_s) + \frac{2}{\tau}(\mathbf{x}_{s+1} - \mathbf{x}_s) + \partial I_X(\mathbf{x}_{s+1}). \quad (99)$$

We now set

$$\mathbf{v} = \nabla_{\mathbf{x}} K(\mathbf{x}_{s+1}, \mathbf{y}_{s+1}, \mathbf{z}_s) - G(\mathbf{x}_s, \mathbf{y}_{s+1}, \mathbf{z}_s) - \frac{2}{\tau}(\mathbf{x}_{s+1} - \mathbf{x}_s) - \rho A^\top (A\mathbf{x}_{s+1} - \mathbf{b}) - \mu(\mathbf{x}_{s+1} - \mathbf{z}_s).$$

Now, by using the definition of K(x, y, z) from [\(19\)](#page-7-2) and [\(99\)](#page-36-0), we obtain (cf. [\(5\)](#page-2-2))

$$\mathbf{v} \in \nabla f(\mathbf{x}_{s+1}) + A^\top \mathbf{y}_{s+1} + \partial I_X(\mathbf{x}_{s+1})$$

We now derive the guarantees on the feasibility and the norm of v. First, by triangle inequality, we have

$$\begin{aligned} \|A\mathbf{x}_{s+1} - \mathbf{b}\| &\leq \|A\mathbf{x}^*(\mathbf{y}_{s+1}, \mathbf{z}_s) - \mathbf{b}\| + \|A\mathbf{x}_{s+1} - A\mathbf{x}_s\| + \|A(\mathbf{x}_s - \mathbf{x}^*(\mathbf{y}_{s+1}, \mathbf{z}_s))\| \\ &\leq \|A\mathbf{x}^*(\mathbf{y}_{s+1}, \mathbf{z}_s) - \mathbf{b}\| + \|A\|\|\mathbf{x}_{s+1} - \mathbf{x}_s\| + \frac{2\|A\|}{\tau(\mu - L_f)}\|\mathbf{x}_s - \mathbf{x}_{s+1}\| + \frac{2\|A\|}{\mu - L_f}\|\widehat{\nabla} f_s - \nabla f(\mathbf{x}_s)\| \\ &= O(T^{-1/3}), \end{aligned} \tag{100}$$

where in the second inequality, we use Lemma [C.6](#page-34-0) and the last estimate uses [\(98\)](#page-36-1).

Then, we have by triangle inequality that

$$\begin{aligned} \|\mathbf{v}\| &\leq \|\nabla_{\mathbf{x}} K(\mathbf{x}_{s+1}, \mathbf{y}_{s+1}, \mathbf{z}_s) - \nabla_{\mathbf{x}} K(\mathbf{x}_s, \mathbf{y}_{s+1}, \mathbf{z}_s)\| + \|\nabla_{\mathbf{x}} K(\mathbf{x}_s, \mathbf{y}_{s+1}, \mathbf{z}_s) - G(\mathbf{x}_s, \mathbf{y}_{s+1}, \mathbf{z}_s)\| \\ &\quad + \frac{2}{\tau} \|\mathbf{x}_{s+1} - \mathbf{x}_s\| + \rho \|A\| \|A\mathbf{x}_{s+1} - \mathbf{b}\| + \mu \|\mathbf{x}_{s+1} - \mathbf{z}_s\| \\ &\leq \left( L_K + \frac{2}{\tau} \right) \|\mathbf{x}_{s+1} - \mathbf{x}_s\| + \|\nabla f(\mathbf{x}_s) - \hat{\nabla} f_s\| + \rho \|A\| \|A\mathbf{x}_{s+1} - \mathbf{b}\| + \mu (\|\mathbf{x}_s - \mathbf{z}_s\| + \|\mathbf{x}_{s+1} - \mathbf{x}_s\|) \\ &= O(T^{-1/3}), \end{aligned}$$

where in first inequality, we introduce a term ∇xK(xs, ys+1, zs) and then use triangle inequality. The second inequality used Lipschitzness of K, the definition of G, and the triangle inequality. The last step uses [\(98\)](#page-36-1) and [\(100\)](#page-36-2) and ρ = O(1) since it is chosen arbitrarily in Alg. [3.](#page-7-4) ■