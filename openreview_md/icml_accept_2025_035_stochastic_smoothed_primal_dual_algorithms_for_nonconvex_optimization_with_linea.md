# Stochastic Smoothed Primal-Dual Algorithms For Nonconvex Optimization With Linear Inequality Constraints

Ruichuan Huang 1 **Jiawei Zhang** * 2 3 **Ahmet Alacaoglu** * 1

## Abstract

We propose smoothed primal-dual algorithms for solving stochastic nonconvex optimization problems with linear *inequality* constraints. Our algorithms are single-loop and only require a single (or two) samples of stochastic gradients at each iteration. A defining feature of our algorithm is that it is based on an inexact gradient descent framework for the Moreau envelope, where the gradient of the Moreau envelope is estimated using one step of a stochastic primal-dual (linearized) augmented Lagrangian algorithm. To handle inequality constraints and stochasticity, we combine the recently established global error bounds in constrained optimization with a Moreau envelope-based analysis of stochastic proximal algorithms. We establish the optimal (in their respective cases) O(ε
−4) and O(ε
−3) sample complexity guarantees for our algorithms and provide extensions to stochastic linear constraints. Unlike existing methods, iterations of our algorithms are free of subproblems, large batch sizes or increasing penalty parameters in their iterations and they use dual variable updates to ensure feasibility.

## 1. Introduction

We focus on the problem template

$${\mathbf{}}\mathbf{\tau}=\mathbf{b},$$
$\widetilde{\mathbf{x}}\widetilde{\in X}$ 4. 
min x∈X
f(x) subject to Ax = b, (1)
where f : R
n → R is Lf -smooth, the set X ⊆ R
n is polyhedral, and easy to project. In particular, let X be given as X = {x: Hx ≤ h} for some matrix H and vector h. Taking H = I, for example, gives this template the ability to model *linear inequality* constraints.

*Co-last authors 1University of British Columbia 2University of Wisconsin–Madison 3MIT. Correspondence to: Jiawei Zhang <jzhang2924@wisc.edu>, Ahmet Alacaoglu <alacaoglu@math.ubc.ca>.

1 In particular, when we have the problem

$$\operatorname*{min}_{\mathbf{x}\in\mathbb{R}^{n}}f(\mathbf{x})\leq$$
$$(2)^{\frac{1}{2}}$$
$$\leq{\bf b},$$
$=\;\frac{1}{2}$ . 
f(x) subject to Ax ≤ b, (2)
we introduce a slack variable t = Ax−b so that Ax−t = b and our optimization variable becomes x t
. Then, we can equivalently write the problem in the template (1) by using the constraint t ≤ 0, where the set X = {x t
: x ∈
R 
n, t ≤ 0} is easy to project. As such, we focus on (1)
and our results directly apply to solving (2) by using this standard slack variable reformulation. The assumption of X being easy-to-project is without loss of generality. Indeed, when X is not easy to project, we can add a slack variable for Hx ≤ h similar to the above paragraph, to have a linear equality constrained problem with projectable constraints (cf. (1)). We refer to (Li et al., 2021, Remark 6), for the classical conversion of an ε-stationarity point of the problem with the slack variable to the original inequality constrained problem. Throughout, we assume that we have access to an unbiased oracle F(x) such that

$$\mathbb{E}[F(\mathbf{x})]=\nabla f(\mathbf{x}),\text{and}\mathbb{E}\|F(\mathbf{x})-\nabla f(\mathbf{x})\|^{2}\leq\sigma^{2}.\tag{3}$$

A common setting is when f(x) = Eξ∼Ξ[f(x, ξ)] where Ξ
is an unknown distribution that we can draw i.i.d. samples from. In this case, it is common to set F(x) = ∇f(x, ξ)
where E[∇f(x, ξ)] = ∇f(x). This will be our main focus.

$$(1)$$

Inclusion of the set X in (1) increases the modeling power of (1) significantly, while causing difficulties in the analysis. Many problems fit this template, including constrained and distributed optimization, nonnegative matrix factorization, sparse subspace estimation and collaborative learning, see for example1(Zhang et al., 2022; Hong, 2016). Moreover, reformulations of nonconvex problems are also common by using linear inequality constraints (Zhang et al., 2022).

Algorithm development for (1) and related templates with global complexity guarantees, have been active in the last couple of years (Alacaoglu & Wright, 2024; Zhang & Luo, 2020; Zhang et al., 2020; Lu et al., 2024; Li et al., 2021; Lin et al., 2022; Yan & Xu, 2022; Li et al., 2024; Boob 1Details of some applications are given in Sec. 6 of our extended version: https://arxiv.org/abs/2504.07607 more precisely the quadratic penalty (QP)) has the form of et al., 2023; Hong, 2016), mainly due to the new applications of functionally constrained nonconvex optimization problems in the context of neural network training (Katz- Samuels et al., 2022; Dener et al., 2020). In these applications with problems involving nonconvex functional constraints, stochastic augmented Lagrangian methods (ALM) have found widespread use, whereas their behavior for even linearly constrained nonconvex optimization of the form (1) remain poorly understood. Our focus is to improve our understanding of stochastic ALM in the context of nonconvex optimization, by focusing on the fundamental template (1). Compared to the setting of convex f, where the global complexity analysis is mature for ALM and its stochastic version (Yan & Xu, 2022), nonconvexity of f poses significant difficulties in the analysis of ALM. Many works in the literature focus on penalty based algorithms (which will be formally introduced later in this section) that do not perform dual updates (or perform *negligible* dual updates that we clarify later) (Lu et al., 2024; Li et al., 2021; Lin et al., 2022), rather than primal-dual algorithms such as ALM. However, in practice, dual updates are known to be essential for accelerating convergence. Penalty methods are known to be unstable since increasing penalty parameter causes Lipschitz constant of the subproblems to increase and can lead to numerical issues. These differences in behavior between penalty and augmented Lagrangian methods are well-known, see, for example, the classical books (Bertsekas, 2014, Sec. 2.2.5) (Nocedal & Wright, 1999, Sec. 17.5). For problem (1) with access to full gradients of f and the full matrix A, optimal complexity with primal-dual methods are obtained in the work of Zhang & Luo (2022). When one has access to stochastic gradients of f and the matrix A, a recent work by Alacaoglu & Wright (2024) showed optimal complexity guarantees under expected smoothness (see Assumption 5.2), for the special case of (1) when X = R
n.

However, this latter restriction significantly reduces the generality of the template. For example, modeling standard quadratic programming requires X to be a half-space, which was not supported in the analysis of Alacaoglu & Wright (2024). Our goal is to go beyond these results by handling both the case when X ̸= R
n as well as the case when we do not have access to the matrix A but only to an unbiased estimate of A, by keeping optimal complexity guarantees.

A more detailed comparison of complexity guarantees will be made in Section 6 and a summary is provided in Table 1. Lagrangian, penalty and augmented Lagrangian. The standard approach to tackle (1) is to design algorithms operating on the Lagrangian, augmented Lagrangian or penalty functions. In particular, the Lagrangian function is given as

$$L(\mathbf{x},\mathbf{y})=f(\mathbf{x})+\langle A\mathbf{x}-\mathbf{b},\mathbf{y}\rangle,$$

with the dual variables y, whereas the penalty function (or

$$\mathrm{Pen}_{\rho}(\mathbf{x})=f(\mathbf{x})+{\frac{\rho}{2}}\|A\mathbf{x}-\mathbf{b}\|^{2}.$$

It is common for algorithms based on the penalty function to require ρ → ∞ for convergence (Bertsekas, 2014, Sec.

2.2.5). One major disadvantage of this strategy is that ρ getting larger makes the subproblem of minimizing the penalty function more and more ill-conditioned (cf. (4)). An influential idea was the introduction of the augmented Lagrangian (AL) function which combined the idea of the Lagrangian and penalty formulations (Hestenes, 1969; Powell, 1969). In particular, the AL function is defined as

$$L_{\rho}(\mathbf{x},\mathbf{y})=f(\mathbf{x})+\langle A\mathbf{x}-\mathbf{b},\mathbf{y}\rangle+{\frac{\rho}{2}}\|A\mathbf{x}-\mathbf{b}\|^{2}.$$

Augmented Lagrangian methods in the classical literature were favoured because they did not require ρ to grow arbitrarily large. In fact, many instances of ALM converge to the optimal solution with fixed ρ since the incorporation of the dual variable updates aids in satisfying feasibility (Bertsekas, 2014, Prop. 2.4, Prop. 2.6). Primal vs primal-dual algorithms. The algorithms based on the penalty function are generally referred to as penalty algorithms and are easier to analyze in different settings since they are primal-only algorithms, meaning that they only perform updates on primal variable x where approximate feasibility is ensured by ρ → ∞. In particular, a classical penalty method iterates for k = 1, 2*, . . .* as

$$\mathbf{x}_{k+1}\approx\arg\min_{\mathbf{x}\in X}f(\mathbf{x})+\frac{\rho_{k}}{2}\|A\mathbf{x}-\mathbf{b}\|^{2},\tag{4}$$  Select $\rho_{k+1}>\rho_{k}$.  
The algorithms based on the AL function are generally more difficult to analyze due to the additional dynamics coming from the dual updates which are critical to ensure that the approximate feasibility is attained with constant ρ. An ALM iteration proceeds for k = 1, 2*, . . .* by updating

$$\begin{array}{l}{{\mathbf{x}_{k+1}\approx\operatorname*{arg\,min}_{\mathbf{x}\in X}f(\mathbf{x})+\langle\mathbf{y}_{k},A\mathbf{x}-\mathbf{b}\rangle+{\frac{\rho}{2}}\|A\mathbf{x}-\mathbf{b}\|^{2},}}\\ {{\mathbf{y}_{k+1}=\mathbf{y}_{k}+\sigma(A\mathbf{x}_{k+1}-\mathbf{b}).}}\end{array}$$

For penalty methods and ALM, different strategies exist to generate xk+1 that approximately minimize the penalty or augmented Lagrangian functions by either iterating multiple steps of gradient descent (GD), known as *inexact* algorithms, or applying one step of GD, known as *linearized* algorithms (Ouyang et al., 2015).

In view of the earlier discussion, when f is nonconvex, most of the literature focuses on either analyzing penalty methods, or analyzing ALM with *negligible* dual updates and increasing penalty parameters ρ, due to the inherent difficulty in analyzing the dual variable and its effect in convergence. In particular, as also highlighted in (Alacaoglu & Wright, 2024), many of the recent analysis of ALM is of the form of a *perturbed penalty analysis*, meaning that the feasibility is driven by increasing penalty parameters, and the dual updates are designed so that they do not deteriorate the estimates too much. Because of this, the dual step sizes are selected to be small to ensure boundedness of the dual variable (or controlling the growth of the dual variable). We refer to such updates as *negligible* dual updates since the analyses do not harness the benefit of such updates in ensuring feasibility. Feasibility is driven by large penalty parameters. Some representative examples are (Lu et al., 2024), (Li et al., 2021), (Lin et al., 2022), (Li et al., 2024). This is the case even in the deterministic setting and the only method that we are aware that can handle true ALM with fixed penalty parameters and non-negligible dual updates are due to (Zhang & Luo, 2022) that uses a linearized *proximal* AL function with a dynamic adjustment on the proximal center, which will be clarified in Section 2 since it will form the basis of our algorithmic development.

## 1.1. Contributions

In this paper, we propose a stochastic smoothed linearized ALM for solving (1) that only uses a single sample of stochastic gradient at every iteration. This algorithm also works with a constant penalty parameter and incorporates non-negligible dual updates for feasibility where the dual step sizes have the same order as the primal step sizes. We show that this method has its iteration complexity and sample complexity guarantees in the order of O(ε
−4). Such a sample complexity result is optimal even in the unconstrained nonconvex case under our assumptions (see Assumption 1.1) (Arjevani et al., 2023). In contrast, the prior results with optimal complexity required large penalty parameters, no dual updates and further assumptions (Lu et al., 2024). We then prove that this complexity can be improved to O(ε
−3) with variance reduction when an additional expected smoothness assumption is made (see Assumption 5.2). Under this stronger assumption, this is the optimal complexity even without constraints (Arjevani et al., 2023).

We consider extensions of this framework when we have linear constraints that hold in expectation, that is, when the constraints are given as Eξ[Aξx − bξ] = 0, with the same complexity guarantees. To our knowledge, this is the first algorithm achieving the optimal O(ε
−4) benchmark sample complexity for nonconvex optimization with stochastic constraints using one sample per iteration, going beyond the best-known O(ε
−5) complexity that is achieved for a more general problem that does not capture the structure of linear constraints (Li et al., 2024; Alacaoglu & Wright, 2024). A more detailed comparison with the related works is given in Section 6. A summary is given in Table 1.

## 1.2. Preliminaries

We denote the indicator function of a convex closed set X as IX(z) = 0 if x ∈ X and IX(x) = ∞ if x ̸∈ X.

The notation ∂f for a convex, closed function denotes the subdifferential set and ∂IX(x) is the normal cone of X at x, by definition. For matrix A, ∥A∥ denotes its operator norm. Given closed and convex X, projection onto X is given as

$$\operatorname{proj}_{X}(\mathbf{x})=\arg\min_{\mathbf{v}\in X}\|\mathbf{x}-\mathbf{v}\|.$$
$\tau$ as. 
2.
Similarly, we define the proximal operator of f as

$$\operatorname{prox}_{f}(\mathbf{x})=\operatorname*{arg\,min}_{\mathbf{v}}f(\mathbf{v})+{\frac{1}{2}}\|\mathbf{v}-\mathbf{x}\|^{2}.$$

We say that f is L-smooth when its gradient is L-Lipschitz:

$$|\nabla f(\mathbf{x})-\nabla f(\mathbf{y})|$$
$$\|\nabla f(\mathbf{x})-\nabla f(\mathbf{y})\|\leq L\|\mathbf{x}-\mathbf{y}\|.$$

We say that f is ρ-weakly convex when f +
ρ 2
∥·∥2is convex.

An L-smooth function is automatically L-weakly convex. Moreau envelope of the weakly convex f is defined as

$$\varphi_{\lambda}(\mathbf{z})=\operatorname*{min}_{\mathbf{v}}f(\mathbf{v})+{\frac{1}{2\lambda}}\|\mathbf{v}-\mathbf{z}\|^{2},$$

which can be interpreted as a notion of *smoothing*. Moreau envelope has many useful properties such as being smooth when f is nonsmooth and weakly convex, when λ is selected accordingly. Moreover, stationary points of f and the Moreau envelope coincide (Drusvyatskiy & Paquette, 2019, Lemma 4.3). The gradient of the Moreau envelope can be computed as λ
−1(x − proxλφ(x)).

Stationary points. A succinct way of characterizing a stationary point of (1) is the following: x
⋆is a stationary point if there exists y
⋆such that the following hold:
0 ∈ ∇f(x
⋆) + A
⊤y
⋆ + ∂IX(x
⋆) and 0 = Ax
⋆ − b.

$$\varepsilon{\mathrm{~and~}}$$
One may, for example, refer to (Rockafellar, 2000). Accordingly, we say that (x, y) is ε**-stationary** if
∥Ax − b∥ ≤ ε and
(5)  $\|\mathbf{v}\|\leq\varepsilon$ where $\mathbf{v}\in\nabla f(\mathbf{x})+A^{\top}\mathbf{y}+\partial I_{X}(\mathbf{x})$
which is a common notion used in related works, for example (Zhang & Luo, 2022). We also use the following related notion of near-stationarity, as used in (Davis & Drusvyatskiy, 2019). We say that x is ε**-near stationary** if it satisfies
∥∇Ψ(x)∥ ≤ ε, (6)

$$(6)$$
$$\leq\varepsilon,$$

where Ψ(x) is the Moreau envelope of the objective function f(x) +IX(x) +I{v:Av=b}(x) in (1), see also (7). We refer to (Davis & Drusvyatskiy, 2019) for the precise notion of near stationarity.

| Reference                  | Constraint                        | Oracle                | Complexity   | Loops   | Method   |
|----------------------------|-----------------------------------|-----------------------|--------------|---------|----------|
| (Alacaoglu & Wright, 2024) | Ax = b                            | Eq. (3) and Asmp. 5.2 | Oe(ε         |         |          |
| E[c(x, ζ)] = 0,            | Eq. (3) and                       |                       |              |         |          |
| (Alacaoglu & Wright, 2024) | Asmp. 5.2                         | Oe(ε                  |              |         |          |
| and x ∈ X†                 |                                   |                       |              |         |          |
| (Lu et al., 2024)          | c(x) = 0,                         | Eq. (3) and Asmp. 5.2 | O(ε          |         |          |
| and x ∈ X† E[c(x, ζ)] = 0, | Eq. (3) and                       |                       |              |         |          |
| (Li et al., 2024)          | Asmp. 5.2                         | O(ε                   |              |         |          |
| and x ∈ X†                 |                                   |                       |              |         |          |
| This work                  | Ax = b, and x ∈ X is a polyhedral | Eq. (3)               | O(ε          |         |          |
| Eζ [A(ζ)x − b(ζ)] = 0,     |                                   |                       |              |         |          |
| This work                  | and x ∈ X is a polyhedral         | Eq. (3)               | O(ε          |         |          |
| This work                  | Ax = b,                           | Eq. (3) and Asmp. 5.2 | O(ε          |         |          |
| and x ∈ X is a polyhedral  |                                   |                       |              |         |          |

Table 1. Comparison of methods. ∗This method is referred to as a penalty method because the penalty parameter is taken to infinity to ensure feasibility and dual updates do not contribute in achieving feasibility. †The set X is assumed to have an efficient projection.

## 1.3. Assumptions

We next state the assumptions that will be used throughout. These assumptions are standard and to our knowledge, the weakest, in the literature for both deterministic and stochastic nonconvex problems with linear constraints (Zhang & Luo, 2022; Alacaoglu & Wright, 2024). A more detailed comparison of assumptions will be made in Section 6. Assumption 1.1. For the problem (1), the following holds:
1. The function f is Lf -smooth and lower bounded over the feasible set: f(x) ≥ f > −∞ for any x ∈ X and Ax = b.

2. The set X admits an efficient projection and is polyhedral. That is, it has the form X = {x: Hx ≤ h} for some *H, h*.

3. We have access to stochastic gradients satisfying (3).

## 2. Algorithm

We introduce Algorithm 1 in this section. To gain a deeper understanding of the algorithm, we will go over two different ways of interpreting it. Interpretation 1: Linearized proximal ALM. Algorithm 1 incorporates a single-step SGD approximation of the proximal AL function. This strategy is also known as the linearized proximal ALM. In particular, the first step of the algorithm approximates the proximal AL function2, that is,

$$\mathbf{x}_{t+1}\approx\arg\operatorname*{min}_{\mathbf{x}\in X}L_{\rho}(\mathbf{x},\mathbf{y}_{t+1})+{\frac{\mu}{2}}\|\mathbf{x}-\mathbf{z}_{t}\|^{2},$$

by a single step of projected SGD, followed by a dual variable update and updating the proximal center zt, which 2Note that this is also a classical function (Rockafellar, 1976).

takes average of zt and xt, resulting in the terminology smoothed that we use for the algorithm. Interpretation 2: Inexact GD on the Moreau envelope.3 Algorithm 1 can also be interpreted as an inexact gradient descent step on the Moreau envelope of the function in (1). In particular, this Moreau envelope is given as

$$\Psi(\mathbf{z}_{t})=\min_{\mathbf{x}\in X,A\mathbf{x}=\mathbf{b}}\left\{f(\mathbf{x})+\frac{\mu}{2}\|\mathbf{x}-\mathbf{z}_{t}\|^{2}\right\}.\tag{7}$$

By observing that minimizing the Moreau envelope helps on obtaining a near-stationary point in view of (6) (cf. (Davis & Drusvyatskiy, 2019)), inexact gradient update on this function requires the computation of

$$\operatorname*{arg\,min}_{\mathbf{x}\in X,A\mathbf{x}=\mathbf{b}}\left\{f(\mathbf{x})+{\frac{\mu}{2}}\|\mathbf{x}-\mathbf{z}_{t}\|^{2}\right\},$$

which is a nontrivial optimization subproblem. However, it is easier than (1) because the regularization provides us a strongly convex objective in the subproblem (given that λ is larger than Lf ). As a result, we can approximate the solution of this problem by applying one iteration of ALM since this problem is a strongly convex optimization problem over linear constraints. We show that just one step of stochastic ALM is sufficient at every iteration by using a stochastic gradient computed with a single sample and one dual update, followed by the update of the proximal center zt. On the surface, this algorithm strongly resembles that of Zhang & Luo (2022), from which we draw many ideas.

However, in addition to using stochastic gradients, there is another subtle change, on the update of zt+1. Unlike (Zhang & Luo, 2022), we update zt+1 by using xt to be 3Let us note that Hu et al. (2024) used a similar idea in a different context.

able to continue the analysis with the bounded variance assumption on G (cf. Algorithm 1) instead of boundedness assumption on G, since the latter would require bounded domains. Thanks to this small change in this section, we handle the case with unbounded primal and dual domains.

## 3. Convergence Analysis

In this section, we first provide the main complexity results, then introduce the main analysis tools and a proof sketch.

## 3.1. Main Theorem

In view of the two stationarity notions given in Section 1.2, we start with the result showing that Algorithm 1 outputs a point at which the norm of the gradient of Moreau envelope is small, in expectation. For the result, we state the algorithmic parameters. To avoid clutter, we write the orders of the parameters by highlighting their dependences on the problem parameters. The explicit forms of the parameters are given in (25), in App. A.

$$\begin{array}{l l}{{\tau\asymp{\frac{1}{\sqrt{T}}},}}&{{\eta\asymp{\frac{1}{\sqrt{T}}},}}&{{\beta\asymp{\frac{1}{\sqrt{T}}},}}\\ {{\mu\asymp L_{f},}}&{{\lambda\asymp L_{f}+\mu(\left\|A\right\|^{2}+1).}}\end{array}$$
$$(8)$$

We are now ready to state the first main result. Theorem 3.1. Let Assumption 1.1 hold and run Alg. 1 with parameters from (8)*. We have that* E∥∇Ψ(zt
∗ )∥ ≤ ε *where* t
∗is selected uniformly at random from {0, . . . , T −1} *with* T = Ω(ε
−4)*. The stochastic oracle complexity is* O(ε
−4).

In particular, the above result gives us an ε-near stationary point in view of (Davis & Drusvyatskiy, 2019). To get an ε-stationary point, we perform a post-processing procedure to obtain the following output from the result of Alg. 1:

$$\hat{\bf x}={\rm proj}_{X}({\bf x}_{t^{*}}-\tau\hat{G}({\bf x}_{t^{*}},{\bf y}_{t^{*}+1},{\bf z}_{t^{*}})),\tag{9}$$
 with $\tau\leq\frac{1}{L_K}$ where $L_K$ is   $L_\tau(-\mathbf{v},\mathbf{r})+\frac{\lambda}{\|}\|_{-\mathbf{v}\|^2}(\text{cf.}\mathcal{O})$. 
where LK is the Lipschitz constant of Lρ(·, y, z) + λ 2
∥ · −x∥
2(cf. (25)) and

$${\hat{G}}(\mathbf{x}_{t^{*}},\mathbf{y}_{t^{*}+1},\mathbf{z}_{t^{*}})={\frac{1}{B}}\sum_{i=1}^{B}G(\mathbf{x}_{t^{*}},\mathbf{y}_{t^{*}+1},\mathbf{z}_{t^{*}},\xi_{i})$$

for ξii.i.d. and B = Θ(ε
−2). This is the only place where we use a large batch size and Algorithm 1 only runs with a single sample at every iteration. This post processing step is only done once and does not affect the overall complexity. The details are given in Appendix A.3.

Corollary 3.2. Let Assumption 1.1 hold. From the output of Algorithm 1, we can obtain xˆ which is an ε-stationary point. The complexity of the whole procedure is O(ε
−4).

## 3.2. Analysis Tools

In our analysis, Moreau envelope of two functions is critical. The first was the Moreau envelope of the composite objective in (1), defined in (7). We next define the Moreau envelope on the proximal AL which is the main function to analyze projected SGD, cf. (Davis & Drusvyatskiy, 2019)

$$\varphi_{1/\lambda}(\mathbf{x},\mathbf{y},\mathbf{z})=\min_{\mathbf{u}\in X}\bigg{\{}L_{\rho}(\mathbf{u},\mathbf{y})+\frac{\mu}{2}\|\mathbf{u}-\mathbf{z}\|^{2}\\ +\frac{\lambda}{2}\|\mathbf{u}-\mathbf{x}\|^{2}\bigg{\}}.\tag{10}$$

Another important quantity that has a significant role in the analysis is the proximal point

$$\mathbf{u}^{*}(\mathbf{x},\mathbf{y},\mathbf{z})=\operatorname*{arg\,min}_{\mathbf{u}\in X}L_{\rho}(\mathbf{u},\mathbf{y})+\frac{\mu}{2}\|\mathbf{u}-\mathbf{z}\|^{2}\tag{11}$$ $$+\frac{\lambda}{2}\|\mathbf{u}-\mathbf{x}\|^{2}.$$

With this, we trivially have

φ1/λ(x, y, z) = Lρ(u
$$\begin{array}{l}{{\partial_{+}{\bf y},{\bf z})=L_{\rho}({\bf u}^{*}({\bf x},{\bf y},{\bf z}),{\bf y})}}\\ {{\quad+\frac{\mu}{2}\|{\bf u}^{*}({\bf x},{\bf y},{\bf z})-{\bf z}\|^{2}+\frac{\lambda}{2}\|{\bf u}^{*}({\bf x},{\bf y},{\bf z})-{\bf x}\|^{2}.}}\end{array}$$

This is the main point of departure from (Zhang & Luo, 2022) where the proximal AL function is used in the analysis, in the potential function. This is because (Zhang & Luo, 2022) used a projected *full* GD step on the proximal AL
function for which, a descent inequality follows directly. In our case, because we apply a projected SGD step, to be able to handle updates with single-sample stochastic gradients, we need to use the Moreau envelope of the proximal AL
function in our potential. This analysis of projected SGD was pioneered in (Davis & Drusvyatskiy, 2019).

The first result is a descent result on the Moreau envelope.
Lemma 3.3 (cf. Lemma A.5). Under Assumption 1.1, for
the xt+1 update given in Algorithm *1, we have*
$$16\mathbb{E}\left[\varphi_{1/\lambda}\big{(}\mathbf{x}_{t+1},\mathbf{y}_{t+1},\mathbf{z}_{t+1}\big{)}\right]$$ $$\leq16\mathbb{E}\left[\varphi_{1/\lambda}\big{(}\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t+1}\big{)}\right]$$ $$\quad-\tau\lambda^{2}\mathbb{E}\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{x}_{t}\|^{2}+8\lambda\tau^{2}\sigma^{2}$$ $$\quad+2\left(4\lambda\tau\mu+16\lambda\tau^{2}\mu^{2}+\tau\lambda^{2}\mu^{2}/\gamma_{s}^{2}\right)\mathbb{E}\|\mathbf{z}_{t}-\mathbf{z}_{t+1}\|^{2},$$  _where $\gamma_{s}=2\mu+\rho\|A\|$._
This follows mostly from (Davis & Drusvyatskiy, 2019) and handles the transition from xt to xt+1 in our analysis. One additional error term we have here is ∥zt+1 − zt∥
2, due to the change in the proximal center zt, a term that was not involved in the analysis of (Davis & Drusvyatskiy, 2019).

Next, we incorporate the dynamics of the updates on the dual variable yt and the proximal center zt. These results use some ideas from (Zhang & Luo, 2022) with additional insights. This is because Zhang & Luo (2022) use Algorithm 1 Stochastic smoothed and linearized ALM
Initialize: x0 = z0 ∈ X, y0 ∈ R
m and ρ ≥ 0.

for t = 0 to T − 1 do yt+1 = yt + η(Axt − b)
Sample ξt ∈ Ξ i.i.d. and let G(xt, yt+1, zt, ξt) = ∇f(xt, ξt) + A⊤yt+1 + ρA⊤(Axt − b) + µ(xt − zt). xt+1 = projX(xt − τG(xt, yt+1, zt, ξt))
zt+1 = zt + β(xt − zt)
Lρ(x, y) + λ 2
∥x − z∥
2in their potential, so their analysis only characterizes the change in y and z in this function. Our analysis however, needs to characterize this change in the Moreau envelope of this function. This requires further estimations using the properties of the Moreau envelope, and the proximal point u
∗(x, y, z) (see e.g. Lem. A.6).

Lemma 3.4. (cf. Lemma A.6) Under Assumption 1.1, for the iterates of Alg. *1, we have*

$$2\mathbb{E}\left[\varphi_{1/\lambda}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t+1})\right]$$ $$\leq2\mathbb{E}\left[\varphi_{1/\lambda}(\mathbf{x}_{t},\mathbf{y}_{t},\mathbf{z}_{t})\right]$$ $$\quad-2\mathbb{E}\langle\mathbf{y}_{t+1}-\mathbf{y}_{t},A\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t},\mathbf{z}_{t})-\mathbf{b}\rangle$$ $$\quad-\mu\mathbb{E}\langle\mathbf{z}_{t}-\mathbf{z}_{t+1},2\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{z}_{t+1}-\mathbf{z}_{t}\rangle.$$

It is easy to notice that combining the last two lemmas will give us a bound on the change of φ1/λ from t to t+1. On the other hand, the inner products appearing on the right-hand side of the last bound will require an intricate analysis after combining with the terms coming from other components in the potential function, introduced next. One aim, is to make sure we get enough slack to be able to cancel error terms coming from ∥zt+1 −zt∥
2in the previous lemma and further errors that will arise as we handle the inner products.

## 3.3. Proof Sketch

3.3.1. ONE ITERATION INEQUALITY ON THE POTENTIAL
As alluded to earlier, we introduce the potential function we work with, which incorporates the Moreau envelopes defined in (10) and (7):

$V_{t}=\varphi_{1/\lambda}(\mathbf{x}_{t},\mathbf{y}_{t},\mathbf{z}_{t})-2d(\mathbf{y}_{t},\mathbf{z}_{t})+2\Psi$
where we used the new notation

$$d({\bf y},{\bf z})=\min_{{\bf x}\in X}L_{\rho}({\bf x},{\bf y})+\frac{\mu}{2}\|{\bf x}-{\bf z}\|^{2}.\tag{12}$$

There are two main changes compared to the analysis of
(Zhang & Luo, 2022). The first is that the *primal descent* portion of our analysis investigates the behavior of the Moreau envelope of the proximal AL function (given in (10)) whereas the analysis of (Zhang & Luo, 2022) analyzes the proximal AL function (given in (19)) directly.

The reason for this departure is the well-known difficulty while analyzing SGD for constrained problems with single sample of stochastic gradients. Hence, it is not clear if it is possible to show a useful inequality with the proximal AL function in the constrained case. In particular, until the work of (Davis & Drusvyatskiy, 2019), convergence analyses of projected SGD required large batches. In addition to combining the bounds from the previous section on the change of φ1/λ, we have to characterize the change in d(y, z) and Ψ(z), for which we can use the following estimations, which only use the definition of yt+1 and hence have the same proof as the previous work. Lemma 3.5. (Zhang & Luo, 2020, Lemma 3.2, Lemma 3.3) For d(y, z) and Ψ(z) *defined in* (7) and (12)*, we have*

$$2d(\mathbf{y}_{t+1},\mathbf{z}_{t+1})-2d(\mathbf{y}_{t},\mathbf{z}_{t})$$ $$\geq2\eta\langle A\mathbf{x}_{t}-\mathbf{b},A\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{b}\rangle$$ $$\quad+\mu\langle\mathbf{z}_{t+1}-\mathbf{z}_{t},\mathbf{z}_{t+1}+\mathbf{z}_{t}-2\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t+1})\rangle,$$

and

$\Psi(\mathbf{z}_{t+1})-\Psi(\mathbf{z}_{t})\leq\mu\langle\mathbf{z}_{t+1}-\mathbf{z}_{t},\mathbf{z}_{t}-\bar{\mathbf{x}}^{*}(\mathbf{z}_{t})\rangle$  $$+\frac{\mu}{2\sigma_{4}}\left\|\mathbf{z}_{t}-\mathbf{z}_{t+1}\right\|^{2},$$
* [16] A. A. K.  
$$\mathbf{x}^{*}(\mathbf{y},\mathbf{z})=\operatorname*{arg\,min}_{\mathbf{x}\in X}L_{\rho}(\mathbf{x},\mathbf{y})+\frac{\mu}{2}\|\mathbf{x}-\mathbf{z}\|^{2},\tag{13}$$ $$\mathbf{\bar{x}}^{*}(\mathbf{z})=\operatorname*{arg\,min}_{\mathbf{x}\in X,A\mathbf{x}=\mathbf{b}}f(\mathbf{x})+\frac{\mu}{2}\|\mathbf{x}-\mathbf{z}\|^{2}.\tag{14}$$
We continue with the main inequality on the potential function with one iteration of Alg. 1. The proof of this lemma is rather intricate and requires a careful combination of the inner products coming from the previous lemmas, and uses the particular update of the proximal center zt+1 as well as parameter selections. Recall that u
∗(x, y, z) and x
∗(y, z)
appearing in the lemma are defined in (11) and (13). Lemma 3.6 (cf. Lemma A.9). With Assumption 1.1 and parameters in (8) *(see* (25)), we have for Alg. 1 *that*

$$\mathbb{E}V_{t}-\mathbb{E}V_{t+1}\geq c_{\beta}\mathbb{E}\|\mathbf{z}_{t+1}-\mathbf{z}_{t}\|^{2}-\lambda\tau^{2}\sigma^{2}/2$$ $$+c_{\tau}\mathbb{E}\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{x}_{t}\|^{2}$$ $$+c_{\eta}\mathbb{E}\|A\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{b}\|^{2},\tag{15}$$

where cτ = Θ(1/
√T), cη = Θ(1/
√T), cβ = Θ(1/
√T)
with their precise definitions given in Lemma A.9.

One novelty in our analysis is to show that this potential function is still lower bounded and decreases, in expectation, up to an error term depends on τ 2and the variance. To integrate this change into the framework of (Zhang & Luo, 2022) under reasonable assumptions on the stochastic oracle as mentioned earlier in Section 2, we also slightly changed the definition of zt+1 in the algorithm, due to technical reasons. In particular, in our case, we lose the control over
∥xt+1 − xt∥
2(since we do not assume bounded domains in this section), whereas the deterministic analysis of (Zhang & Luo, 2022) have a natural control over such terms. The other change is the error coming from the variance of stochastic gradients. This causes the complexity to deteriorate compared to the deterministic case, which is an effect common with algorithms based on SGD. In particular, with a correctly selected step size, we obtain a sample complexity with the same-order as SGD, which is optimal even for unconstrained nonconvex problems (Arjevani et al., 2023).

## 3.3.2. Complexity Analysis

After Lemma 3.6, it is straightforward to obtain

$$\begin{array}{c}{{\mathbb{E}\|\mathbf{z}_{t+1}-\mathbf{z}_{t}\|^{2}\leq\varepsilon^{2},}}\\ {{\mathbb{E}\|A\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{b}\|^{2}\leq\varepsilon^{2},}}\\ {{\mathbb{E}\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{x}_{t}\|^{2}\leq\varepsilon^{2},}}\end{array}$$

when T *= Θ(*ε
−4). Then, by tedious but straightforward calculations, we can directly get the bound on the norm of the gradient of the Moreau envelope, ∇Ψ(zt), obtaining near-stationarity. The details appear in Appendix A.2. A couple more steps let us go from this result to an εstationary point. The idea is simple: since we know that small ∥∇Ψ(zt)∥ means that we are near a stationary point, we can perform just one more iteration of SGD with batch size ≈ ε
−2to get an ε-stationary point, without changing the worst-case complexity. The details are in App. A.3.

## 4. Extension To Random Linear Constraints

We turn to the case when constraints are sampled, that is, we do not have access to the full matrix A, or vector b but only unbiased samples of them. This is a suitable setting, when, for example, we have a large matrix A. In particular, we have A = Eζ∼P [Aζ ], b = Eζ∼P [bζ ] and use Aζ , bζ in the algorithm. We rewrite the template for convenience, as

$$\operatorname*{min}_{\mathbf{x}\in X}f(\mathbf{x}){\mathrm{~subject~to~}}\mathbb{E}_{\zeta\sim P}[A_{\zeta}\mathbf{x}-\mathbf{b}_{\zeta}]=0.$$

In this case, to get an unbiased stochastic gradient for proximal AL, we need to sample two i.i.d. samples of ζ:

$$\begin{array}{c}{{G({\bf x},{\bf y},{\bf z},\xi)=\nabla f({\bf x},\xi)}}\\ {{\quad+A_{\zeta^{1}}^{\top}{\bf y}+\rho A_{\zeta^{1}}^{\top}(A_{\zeta^{2}}{\bf x}-{\bf b}_{\zeta^{2}})+\mu({\bf x}-{\bf z}).}}\end{array}$$
$$(17)^{\frac{1}{2}}$$

7 An immediate issue here is that the variance of stochastic gradients of the proximal AL function scales linearly with x and y. Hence, assuming bounded variance would require assuming bounded dual variables, which is a strong assumption that is not satisfied in practice. To go around this difficulty, we have two adjustments, (i) we assume a constraint qualification (CQ) and compactness of X and *(ii)* we include a safeguarding procedure in the algorithm to monitor when the dual variable gets too large. Under these two modifications, we obtain the same complexity guarantees as our previous setting with deterministic constraints. Assumption 4.1. For problem (16), the following holds:
1. The feasible set {x : x ∈ *X, A*x = b} is bounded. 2. The origin is in the relative interior of the set {Ax −
b: x ∈ X}.

3. A has full row-rank.

In addition to the assumptions in the earlier setting, we require a Slater's condition as well as compact domains to ensure boundedness of the dual variable. Slater's condition is a classical CQ, see e.g., (Bertsekas et al., 2003, Sec. 5.3.1).

Remark 4.2. The choice of My is given next, which admittedly can be difficult in practice. Let MV = maxx,z∈X{K(x, 0, z) − 2d(0, z) + 2Ψ(z)}, M = maxx,z∈X{|f(x)|+
µ 2
∥x−z∥
2+
ρ 2
∥Ax−b∥
2}, where K is defined in (19) and MΨ is a uniform lower bound of Ψ(zt),
e.g., f. According to Assumption 4.1, there exists r > 0 such that for any direction d ∈ Range(A), we can find x ∈ X satisfying ∥Ax−b∥ = r and Ax−b has the same direction as d. Then, we choose My as My >
MV −MΨ+2M
r.

In this setting, we only state our theorem for nearstationarity. The ε-stationarity would follow in the same way as the previous section by a post-processing step. Theorem 4.3. Let Assumptions 1.1 and 4.1 *hold and run Alg.*
2 *with parameters from* (8)*. We have that* E∥∇Ψ(zt
∗ )∥ ≤ ε where t
∗is randomly selected from {0, . . . , T − 1} *with* T = Ω(ε
−4)*. The stochastic oracle complexity is* O(ε
−4).

As mentioned earlier, the optimal sample complexity for nonconvex optimization with Lipschitz ∇f is O(ε
−4) (Arjevani et al., 2023). Our result matches this complexity while handling linear constraints with random sampling.

## 5. Extension With Variance Reduction

$$(16)^{\frac{1}{2}}$$

We now integrate the STORM variance reduction technique from (Cutkosky & Orabona, 2019) into our framework to solve (1) (See arXiv:2504.07607 for extension to stochastic constraints). We obtain Alg. 3, which improves the iteration and oracle complexity from O(ε
−4) to O(ε
−3) under a stronger assumption on the oracle, compared to Sec. 3. This not only leads to an improved rate, but also to a simpler analysis that does not rely on the Moreau envelope φ1/λ.

Algorithm 2 Stochastic smoothed and linearized ALM for stochastic constraints with dual safeguarding Input and Initialization: My >
MV −MΨ+2M
r(check Remark 4.2), x0 = z0 ∈ X, y0 ∈ R
m, ρ ≥ 0.

for t = 0 to T − 1 do yt+1 = yt + η(Aζtxt − bζt
) where ζt ∼ P is generated i.i.d.

if ∥yt+1∥ ≥ My **then**
yt+1 = 0 Sample ξt ∈ Ξ i.i.d. and generate Eξt[G(xt, yt+1, zt, ξt)] = ∇xLρ(xt, yt+1) + µ(xt − zt) as in (17)
xt+1 = projX(xt − τG(xt, yt+1, zt, ξt))
zt+1 = zt + β(xt − zt)
Algorithm 3 Stochastic smoothed and linearized ALM with STORM
Initialize: x0 = z0 ∈ X, y0 ∈ R
m, ∇b f0 =
1 N
PN
i=1 ∇f(x0, ζi), N = T
1/3and ρ ≥ 0 for t = 0 to T − 1 do yt+1 = yt + η(Axt − b)
G(xt, yt+1, zt) = ∇b ft + A⊤yt+1 + ρA⊤(Axt − b) + µ(xt − zt)
xt+1 = projX(xt − τG(xt, yt+1, zt))
zt+1 = zt + β(xt − zt)
Sample ξt+1 ∼ Ξ i.i.d. and set ∇b ft+1 = ∇f(xt+1, ξt+1) + (1 − α)(∇b ft − ∇f(xt, ξt+1))
Alg. 3 and Alg. 1 mainly differ in the update of stochastic gradient estimate ∇b ft. If α = 0, Alg. 3 trivially reduces to Alg. 1. We next see that a particular choice of α gives better complexity under Assumption 5.2 (which is stronger than the oracle access and smoothness in Assumption 1.1). Remark 5.1. We only use a minibatch in the initialization, which does not affect the overall complexity. The minibatch size is N = T
1/3, which is small compared to the total number of iterations T. Iterations of our algorithm only require 2 stochastic gradients, ∇f(xt, ξt+1) and ∇f(xt+1, ξt+1).

For the analysis of Alg. 3, we introduce Assumption 5.2, used, e.g., in (Arjevani et al., 2023). In particular, Arjevani et al. (2023) showed that the oracle complexity O(ε
−3) is tight under Assumption 5.2 even with no constraints. Assumption 5.2. We have access to a stochastic gradient of f satisfying (3). For a given ξ ∼ Ξ, we can query ∇f(x, ξ) and ∇f(y, ξ) for different points x, y. Moreover, we have Eξ∼Ξ∥∇f(x, ξ) − ∇f(y, ξ)∥
2 ≤ L
20∥x − y∥
2.

We introduce the potential V¯t differing from Sec. 3 and 4.

This is similar to (Zhang & Luo, 2022), except the last term which controls the error from the variance. Define

$$\begin{array}{c}{{\bar{V}_{t}=K({\bf x}_{t},{\bf y}_{t},{\bf z}_{t})-2d({\bf y}_{t},{\bf z}_{t})+2\Psi({\bf z}_{t})}}\\ {{\qquad+\frac{1}{48(L_{0}^{2}+L_{f}^{2})\tau}\|\hat{\nabla}f_{t}-\nabla f({\bf x}_{t})\|^{2},}}\end{array}$$
2,(18)
where

$$K(\mathbf{x},\mathbf{y},\mathbf{z})=L_{\rho}(\mathbf{x},\mathbf{y})+{\frac{\mu}{2}}\|\mathbf{x}-\mathbf{z}\|^{2}.$$
2. (19)
One-step evolution of Vˆt that we analyze next is a key step in the analysis. Compared to (Zhang & Luo, 2022), we have the extra error due to using ∇b ft instead of the full gradient.

Lemma 5.3 (cf. Lemma C.4). Under Assumptions 1.1 and 5.2, with parameters

$$\begin{array}{c}{{\mu=\operatorname*{max}\{2\}}}\\ {{\eta=\Theta(\tau),}}\end{array}$$
µ = max{2, 4Lf }, τ = T
$$\tau_{f}\},\quad\tau=T^{-3/2},$$
$\mathbf{a}\cdot\mathbf{a}=\mathbf{a}$
η = Θ(τ ), β = Θ(τ ), α = Θ(τ
$$\alpha=\Theta(\tau^{2}),$$
$$(20)$$

(for detailed parameters, see (82)*) we have*

$$\mathbb{E}\bar{V}_{t}-\mathbb{E}\bar{V}_{t+1}\geq\frac{2\mu}{\beta}\mathbb{E}\|\mathbf{z}_{t}-\mathbf{z}_{t+1}\|^{2}+\frac{1}{2\tau}\mathbb{E}\|\mathbf{x}_{t}-\mathbf{x}_{t+1}\|^{2}$$ $$+2\eta\mathbb{E}\|A\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t})-b\|^{2}$$ $$+\tau\mathbb{E}\|\widehat{\nabla}f_{t}-\nabla f(\mathbf{x}_{t})\|^{2}-O(\sigma^{2}\tau^{3}).\tag{21}$$

Note that, on a high level, the main difference between Lemma 5.3 and Lemma 3.6 is that the order of τ in the error term is different. In Lemma 5.3, the order of τ is O(τ 3), while in Lemma 3.6, the order of τ is O(τ 2), which contribute to a faster convergence rate in for Alg. 3. Theorem 5.4. Let Assumptions 1.1 and 5.2 hold. We have that E∥∇Ψ(zt
∗ )∥ ≤ ε*, where* t
∗is selected uniformly at random from {0, . . . , T − 1} *with* T = Ω(ε
−3). The complexity of the whole procedure is O(ε
−3).

$$(18)$$

## 6. Related Works

$$(19)$$

We now compare the complexity results for obtaining an ε-stationary point, in view of Section 1.2.

Deterministic objective and deterministic constraints. The setting when objective f in (1) is deterministic is the most well-studied with many results in the classical literature (Bertsekas, 2014). Recent work characterized the global oracle complexity of Lagrangian-based methods or ALM.

With nonlinear and nonconvex constraints, many of the existing works analyzing AL-based algorithms rely on strong CQs and boundedness assumptions and use large penalty parameters to ensure feasibility (Li et al., 2021; Lin et al., 2022; Kong et al., 2019; Kong & Monteiro, 2023; Kong et al., 2023). The existing frameworks so far fail to capture the importance of dual variable updates, which are, in fact, the main reason behind the ability to use constant penalty parameters while ensuring convergence, see e.g., (Bertsekas, 2014, Sec. 2.2.5). Recent works mentioned above obtained the complexity bound O(ε
−3) for general nonlinear constraints with no specialization for linear constraints. When specialized to convex functional constraints, the best-known complexity for these methods is O(ε
−2.5) (Lin et al., 2022).

When the constraints are linear, such as (1) with X = R
n, Hong (2016) analyzed ALM with constant penalty parameters and non-negligible dual updates to get optimal complexity O(ε
−2). The case of X ̸= R
n turned out to be significantly more challenging with many works focusing on variants of ALM with large penalty parameters (depending on the inverse of the final accuracy) to ensure near-feasibility and *negligible* dual updates that do not help with feasibility (Kong & Monteiro, 2023; Kong et al., 2023) and obtained the suboptimal complexity Oe(ε
−2.5). The exceptions are the works (Zhang & Luo, 2020; 2022) that showed, for the case X polyhedral, near-optimal complexity O(ε
−2) with a constant penalty parameter and dual steps with constant step sizes, with no constraint qualification. The key step was the global error bound that our work also relied on. Stochastic objective and deterministic constraints. One important step in generalizing the template to tasks arising in ML was to consider stochastic objectives where we access unbiased estimates. With general nonlinear constraints and Lipschitzness of ∇f, the optimal sample complexity is O(ε
−4), obtained with double loop algorithms (Curtis et al.,
2024; Boob et al., 2023; Ma et al., 2020). These works require strong assumptions on the boundedness of the primal domain as well as constraint qualifications, which are often not necessary with linear constraints.

Another set of results concerns stochastic optimization with deterministic nonlinear constraints with penalty-based algorithms. These works require large penalty parameters to ensure near-feasibility rather than dual updates (Lu et al., 2024; Alacaoglu & Wright, 2024). They assume expected Lipschitzness as Assumption 5.2, which is stronger than Lipschitzness of ∇f. Since these works focus on nonlinear functional constraints, the analysis requires boundedness assumptions as well as constraint qualifications, unlike our results in Section 3 for deterministic linear constraints. Alacaoglu & Wright (2024) considered ALM with a constant penalty parameter and non-negligible dual updates and obtained the complexity O(ε
−3) for linear *equality* constraints under Assumption 5.2. This work only covered the case X = R
n and left open the question of handling the case of general X, see (Alacaoglu & Wright, 2024, Sec. 5). We resolve a special case of this question when X is polyhedral (covering many applications), allowing our analysis to cover linear inequality constraints. Alacaoglu & Wright (2024) used variance reduction for ∇f, which meant that they required Assumption 5.2, stronger than Assumption 1.1. In Sec. 5, we get the same complexity as this paper while allowing a polyhedral X to cover linear inequality constraints, which cannot be handled by Alacaoglu & Wright (2024). Moreover, we also get the complexity O(ε
−4) under Assumption 1.1. This is optimal under Assumption 1.1 and we refer to (Arjevani et al., 2023) for further details on the lower bounds. In contrast, the work in (Alacaoglu & Wright, 2024) does not have guarantees without Assumption 5.2. In addition, though (Lu et al., 2024) considers the more general problem with nonconvex functional constraints, they make strong assumptions which are not easy to verify. It is not clear if their assumptions would hold with a general polyhedral constraint we have (see e.g., their Assumption 1(iv) and Eq. (7)). When the constraints are deterministic, we do not have any bounded domain assumption (our Sec. 3) whereas the assumptions of (Lu et al., 2024) are rather difficult to be satisfied without a bounded primal domain. Lu et al. (2024) analyzes a QP-based method, whereas we analyze an ALM-variant. ALM is known to be more stable and desirable in practice, but significantly more difficult to analyze, which is because the penalty parameter is fixed in ALM and it increases to infinity for QP. Our ALM algorithm could be extended to stochastic constraints, while (Lu et al., 2024) only handles deterministic constraints. Alacaoglu & Wright (2024) highlights the importance of analyzing ALM compared to QP methods in their Sections 1 and 6.

Stochastic objective and stochastic constraints. This is the most general class, where the existing results come with many assumptions that are not always easy to interpret, similar to the case of stochastic objective and deterministic constraints described above (Li et al., 2024; Alacaoglu
& Wright, 2024). The best-known complexity O(ε
−5) is obtained by using Assumption 5.2, with an inexact, doubleloop, ALM in (Li et al., 2024) and by a single-loop QP algorithm in (Alacaoglu & Wright, 2024). These results concerning ALM need to use large penalty parameters, which renders them essentially as QP-methods since the dual updates do not contribute to the analysis for ensuring the feasibility. Other approaches for solving this sub-case also require double-loop algorithms and stronger assumptions since they focus on a generic nonconvex constraint (Boob et al., 2023; Ma et al., 2020), obtaining O(ε
−6) without expected Lipschitzness. Hence, in this sub-case, none of these results harness the structure of linear constraints.

## Acknowledgements

Jiawei Zhang is supported by the startup fund from the Department of Computer Sciences at the University of Wisconsin–Madison and the MIT Postdoctoral Fellowship for Engineering Excellence.

Ahmet Alacaoglu acknowledges the support of the Natural Sciences and Engineering Research Council of Canada (NSERC), [funding reference number RGPIN-2025-06634].

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Alacaoglu, A. and Wright, S. J. Complexity of single loop algorithms for nonlinear programming with stochastic objective and constraints. In International Conference on Artificial Intelligence and Statistics, pp. 4627–4635.

PMLR, 2024.

Arjevani, Y., Carmon, Y., Duchi, J. C., Foster, D. J., Srebro, N., and Woodworth, B. Lower bounds for non-convex stochastic optimization. *Mathematical Programming*, 199 (1):165–214, 2023.

Bertsekas, D. Constrained optimization and Lagrange multiplier methods. Academic press, 2014.

Bertsekas, D., Nedic, A., and Ozdaglar, A. *Convex analysis* and optimization, volume 1. Athena Scientific, 2003.

Boob, D., Deng, Q., and Lan, G. Stochastic first-order methods for convex and nonconvex functional constrained optimization. *Mathematical Programming*, 197(1):215– 279, 2023.

Curtis, F. E., O'Neill, M. J., and Robinson, D. P. Worstcase complexity of an sqp method for nonlinear equality constrained stochastic optimization. Mathematical Programming, 205(1):431–483, 2024.

Cutkosky, A. and Orabona, F. Momentum-based variance reduction in non-convex sgd. Advances in neural information processing systems, 32, 2019.

Davis, D. and Drusvyatskiy, D. Stochastic model-based minimization of weakly convex functions. *SIAM Journal* on Optimization, 29(1):207–239, 2019.

Dener, A., Miller, M. A., Churchill, R. M., Munson, T.,
and Chang, C.-S. Training neural networks under physical constraints using a stochastic augmented lagrangian approach. *arXiv preprint arXiv:2009.07330*, 2020.

Drusvyatskiy, D. and Paquette, C. Efficiency of minimizing compositions of convex functions and smooth maps. Mathematical Programming, 178:503–558, 2019.

Hestenes, M. R. Multiplier and gradient methods. Journal of optimization theory and applications, 4(5):303–320, 1969.

Hiriart-Urruty, J.-B. and Lemarechal, C. Convex Analysis and Minimization Algorithms II: Advanced Theory and Bundle Methods, volume 306. Springer Berlin Heidelberg, Berlin, Heidelberg, 1st 1993.;1; edition, 1993. ISBN 0072-7830.

Hong, M. Decomposing linearly constrained nonconvex problems by a proximal primal dual approach: Algorithms, convergence, and applications. arXiv preprint arXiv:1604.00543, 2016.

Hu, Q., Qi, Q., Lu, Z., and Yang, T. Single-loop stochastic algorithms for difference of max-structured weakly convex functions. In Advances in Neural Information Processing Systems, volume 37, pp. 56738–56765, 2024.

Katz-Samuels, J., Nakhleh, J. B., Nowak, R., and Li, Y.

Training OOD detectors in their natural habitats. In *ICML*, 2022.

Kong, W. and Monteiro, R. D. An accelerated inexact dampened augmented lagrangian method for linearlyconstrained nonconvex composite optimization problems. Computational Optimization and Applications, 85(2):509– 545, 2023.

Kong, W., Melo, J. G., and Monteiro, R. D. Complexity of a quadratic penalty accelerated inexact proximal point method for solving linearly constrained nonconvex composite programs. *SIAM Journal on Optimization*, 29(4):
2566–2593, 2019.

Kong, W., Melo, J. G., and Monteiro, R. D. Iteration complexity of an inner accelerated inexact proximal augmented lagrangian method based on the classical lagrangian function. *SIAM Journal on Optimization*, 33(1): 181–210, 2023.

Lan, G. First-order and stochastic optimization methods for machine learning. Springer, 2020.

Li, Z., Chen, P.-Y., Liu, S., Lu, S., and Xu, Y. Rate-improved inexact augmented lagrangian method for constrained nonconvex optimization. In International Conference on Artificial Intelligence and Statistics, pp. 2170–2178.

PMLR, 2021.

Li, Z., Chen, P.-Y., Liu, S., Lu, S., and Xu, Y. Stochastic inexact augmented lagrangian method for nonconvex expectation constrained optimization. *Computational* Optimization and Applications, 87(1):117–147, 2024.

Lin, Q., Ma, R., and Xu, Y. Complexity of an inexact proximal-point penalty method for constrained smooth non-convex optimization. Computational optimization and applications, 82(1):175–224, 2022.

Lu, Z., Mei, S., and Xiao, Y. Variance-reduced first-order methods for deterministically constrained stochastic nonconvex optimization with strong convergence guarantees.

arXiv preprint arXiv:2409.09906, 2024.

Ma, R., Lin, Q., and Yang, T. Quadratically regularized subgradient methods for weakly convex optimization with weakly convex constraints. In *International Conference* on Machine Learning, pp. 6554–6564. PMLR, 2020.

Nocedal, J. and Wright, S. J. *Numerical optimization*.

Springer, 1999.

Ouyang, Y., Chen, Y., Lan, G., and Pasiliao Jr, E. An accelerated linearized alternating direction method of multipliers. *SIAM Journal on Imaging Sciences*, 8(1): 644–681, 2015.

Planiden, C. and Wang, X. Strongly convex functions, moreau envelopes, and the generic nature of convex functions with strong minimizers. SIAM Journal on Optimization, 26(2):1341–1364, 2016.

Powell, M. J. A method for nonlinear constraints in minimization problems. *Optimization*, pp. 283–298, 1969.

Rockafellar, R. T. Augmented lagrangians and applications of the proximal point algorithm in convex programming.

Mathematics of operations research, 1(2):97–116, 1976.

Rockafellar, R. T. Extended nonlinear programming. Nonlinear optimization and related topics, pp. 381–399, 2000.

Yan, Y. and Xu, Y. Adaptive primal-dual stochastic gradient method for expectation-constrained convex stochastic programs. *Mathematical Programming Computation*, 14 (2):319–363, 2022.

Zhang, J. and Luo, Z.-Q. A proximal alternating direction method of multiplier for linearly constrained nonconvex minimization. *SIAM Journal on Optimization*, 30(3): 2272–2302, 2020.

Zhang, J. and Luo, Z.-Q. A global dual error bound and its application to the analysis of linearly constrained nonconvex optimization. *SIAM Journal on Optimization*, 32 (3):2319–2346, 2022. doi: 10.1137/20M135474X. URL https://doi.org/10.1137/20M135474X.

Zhang, J., Xiao, P., Sun, R., and Luo, Z. A singleloop smoothed gradient descent-ascent algorithm for nonconvex-concave min-max problems. Advances in neural information processing systems, 33:7377–7389, 2020.

Zhang, J., Ge, S., Chang, T.-H., and Luo, Z.-Q. Decentralized non-convex learning with linearly coupled constraints: Algorithm designs and application to vertical learning problem. IEEE Transactions on Signal Processing, 70:3312–3327, 2022.

## Notation.

Let us note that we define by Et the expectation conditioned on all the randomness up to and including xt.

## A. Proofs For Section 3

In the proofs, let us recall

$$K(\mathbf{x},\mathbf{y},\mathbf{z})=L_{\rho}(\mathbf{x},\mathbf{y})+\frac{\mu}{2}\|\mathbf{x}-\mathbf{z}\|^{2}$$ $$=f(\mathbf{x})+\langle A\mathbf{x}-\mathbf{b},\mathbf{y}\rangle+\frac{\rho}{2}\|A\mathbf{x}-\mathbf{b}\|^{2}+\frac{\mu}{2}\|\mathbf{x}-\mathbf{z}\|^{2}.$$  In the following, equivalent to (11):
$$(22)$$

With this notation, we have the following, equivalent to (11):

$$\mathbf{u}^{*}(\mathbf{x},\mathbf{y},\mathbf{z})=\operatorname*{arg\,min}_{\mathbf{u}\in X}\left\{K(\mathbf{u},\mathbf{y},\mathbf{z})+\frac{\lambda}{2}\|\mathbf{u}-\mathbf{x}\|^{2}\right\}$$ $$=\operatorname*{arg\,min}_{\mathbf{u}\in X}\left\{L_{\rho}(\mathbf{u},\mathbf{y},\mathbf{z})+\frac{\mu}{2}\|\mathbf{u}-\mathbf{z}\|^{2}+\frac{\lambda}{2}\|\mathbf{u}-\mathbf{x}\|^{2}\right\}.$$
$$(23)$$
We also recall (10).  $$\varphi_{1/\lambda}(\mathbf{x},\mathbf{y},\mathbf{z})=\min_{\mathbf{u}\in\mathbb{X}}\bigg{\{}L_{\rho}(\mathbf{u},\mathbf{y})+\frac{\mu}{2}\|\mathbf{u}-\mathbf{z}\|^{2}+\frac{\lambda}{2}\|\mathbf{u}-\mathbf{x}\|^{2}\bigg{\}}$$ $$=\min_{\mathbf{u}\in\mathbb{X}}\bigg{\{}K(\mathbf{u},\mathbf{y})+\frac{\lambda}{2}\|\mathbf{u}-\mathbf{x}\|^{2}\bigg{\}}.$$  We also introduce here some parameters that are used throughout, for convenience.  
$$(24)$$
Parameters that are used throughout, for convenience:  $\begin{array}{c}\mu=\max\{2,4L_f\},\\ L_K=L_f+\rho||A||+\mu,\\ \lambda=2L_K,\\ \sigma_4=\dfrac{\mu-L_f}{\mu},\\ \tau=\dfrac{1}{6\lambda^2\sqrt{T}},\\ \eta=\min\left\{\dfrac{2\mu+\rho||A||}{4||A||^4},\dfrac{\tau}{200||A||^2},\dfrac{\tau(2\mu+\rho||A||^2)}{20||A||^2}\right\},\\ \beta=\min\left\{\dfrac{\tau}{100},\dfrac{1}{50\lambda},\dfrac{\eta}{36\mu\sigma^2}\right\},\\ \gamma_s=2\mu+\rho||A||,\gamma=\dfrac{(\mu-L_f)\lambda}{\mu-L_f+\lambda},\gamma_K=\mu-L_f.\end{array}$  In the following we can take a non-zero value. 
$$(25)$$

We also mention the following basic facts that are used in the sequel.

Fact A.1. For x ∈ X, we have that x 7→ K(x, y, z) is strongly convex with modulus γK := µ−Lf , and x *7→ ∇*xK(x, y, z) is LK := (Lf + ρ∥A∥
2 + µ)*-Lipschitz continuous.*
For u ∈ X, u 7→ K(u, y, z) + λ2
∥x − u∥
2is strongly convex with modulus γs = µ − Lf + λ*, and* u
∗(x, y, z) =
arg minu∈X K(u, y, z) + λ 2
∥x − u∥
2.

Lemma A.2. (Planiden & Wang, 2016, Lemma 2.19) Let r > 0*. The function f is r-strongly convex if and only if* f1(x) = minu f(u) + 12
∥x − u∥
2is r r+1 *-strongly convex.*
Lemma A.3. *The function* x 7→ φ1/λ(x, y, z) is γ =
(µ−Lf )λ µ−Lf +λ
-strongly convex.

Proof. By definition, we have

$$\varphi_{1/\lambda}({\bf x},{\bf y},{\bf z})=\min_{\bf u}K({\bf u},{\bf y},{\bf z})+I_{X}({\bf u})+\frac{\lambda}{2}\|{\bf x}-{\bf u}\|^{2}=\lambda\min_{\bf u}\frac{K({\bf u},{\bf y},{\bf z})+I_{X}({\bf u})}{\lambda}+\frac{1}{2}\|{\bf x}-{\bf u}\|^{2}.$$

12 Recall that γK = µ − Lf . Then, since K(x, y, z)/λ is γK
λ
-strongly convex, we have minu K(u,y,z)+IX(u)
λ +
1 2
∥x − u∥
2 is γK/λ γK/λ+1 -strongly convex, by Lemma A.2. Hence, φ1/λ(x, y, z) is strongly convex with modulus γK
γK/λ+1 =λγK
λ+γK
=
(µ−Lf )λ µ−Lf +λ
. ■

## A.1. Proofs For Lemma 3.6

In the next lemma, the first part is using the idea of Davis & Drusvyatskiy (2019) to analyze the algorithm under the bounded variance assumption instead of the restrictive bounded stochastic gradient assumption. The second part of the lemma also follows a similar idea as this work, with the exception of the dependence on the changing center point zt. This introduces additional issues, since the stochastic gradient in the update of xt+1 depends on zt whereas the proximal point u
∗(xt, yt+1, zt+1) (that characterizes the iteration below) depends on zt+1. Our analysis below estimates this additional error and shows it to be in the order of ∥zt+1 − zt∥
2, which will be handled later.

Lemma A.4. Suppose that Assumption 1.1 *holds, for the proximal point* u
∗(xt, yt+1, zt+1)*, defined as* (11) we have the characterization u
∗(xt, yt+1, zt+1) = projX(τλxt + (1 − τλ)u
∗(xt, yt+1, zt+1) − τ∇xK(u
∗(xt, yt+1, zt+1), yt+1, zt+1)). (26)
Moreover, for the sequence xt+1 calculated as Algorithm *1, with* λ = 2LK and τ ≤
1 6λ
, where LK = Lf + ρ∥A∥
2 + µ, we have

$$\mathbb{E}[\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t+1})-\mathbf{x}_{t+1}\|^{2}\leq\left(1-\frac{r\lambda}{4}\right)\mathbb{E}\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t+1})-\mathbf{x}_{t}\|^{2}+(r\mu+2r^{2}\mu^{2})\mathbb{E}\|\mathbf{z}_{t}-\mathbf{z}_{t+1}\|^{2}+r^{2}\sigma^{2}.$$

Proof. From the definition of u
∗(xt, yt+1, zt+1) in (11) (see also (23)), we have λ(xt − u
∗(xt, yt+1, zt+1)) ∈ ∇xK(u
∗(xt, yt+1, zt+1), yt+1, zt+1) + ∂IX(u
∗(xt, yt+1, zt+1)).

Multiplying both sides by the step size τ , adding u
∗(xt, yt+1, zt+1) to both sides, and rearranging give τλxt − τ∇xK(u
∗(xt, yt+1, zt+1), yt+1, zt+1) + (1 − τλ)u
∗(xt, yt+1, zt+1)
∈ u
∗(xt, yt+1, zt+1) + *τ ∂I*X(u
∗(xt, yt+1, zt+1)).

Since (I + *τ ∂I*X)
−1 = proxIX = projX due to ∂IX being a cone and proximal operator of a normal cone being the projection to the set, we have the first assertion.

We next establish the second assertion. Using the just established identity (26), the update rule of xt+1 in Algorithm 1, and nonexpansiveness of the projection, we derive
∥u
∗(xt, yt+1, zt+1) − xt+1∥
2
≤ ∥τλxt + (1 − τλ)u
∗(xt, yt+1, zt+1) − τ∇xK(u
∗(xt, yt+1, zt+1), yt+1, zt+1) − [xt − τG(xt, yt+1, zt, ξt)]∥
2.

We add and subtract ∇xK(xt, yt+1, zt) inside the squared norm on the right-hand side, expand and take conditional expectation to obtain

Et∥u ∗(xt, yt+1, zt+1) − xt+1∥ 2 = ∥(1 − τλ)(u ∗(xt, yt+1, zt+1) − xt) − τ∇xK(u ∗(xt, yt+1, zt+1), yt+1, zt+1) + τ∇xK(xt, yt+1, zt)∥ 2 + τ 2Et∥G(xt, yt+1, zt, ξt) − ∇xK(xt, yt+1, zt)∥ 2, (27)
where the cross term disappeared because

$$\mathbf{Z}_{t+1})-\left[\mathbf{x}_{t}-\tau G(\mathbf{x}_{t},$$

Et[G(xt, yt+1, zt, ξt)] = ∇xK(xt, yt+1, zt)
and xt, yt+1, zt+1, u
∗(xt, yt+1, zt+1) are deterministic under the conditioning since zt+1 defined in Algorithm 1 only depends on xt (that is, zt+1 is independent of ξt).

The second term on the right-hand side of (27) is trivially bounded by the oracle assumptions, that is,

$$\mathbb{E}_{t}\|G(\mathbf{x}_{t})$$

Et∥G(xt, yt+1, zt, ξt) − ∇xK(xt, yt+1, zt)∥
$$\mathbf{\dot{\Gamma}}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})\|^{2}\leq\sigma^{2}.$$
2. (28)
, $\xi_t$)]. 
$$(27)^{\frac{1}{2}}$$
$$(28)^{\frac{1}{2}}$$
For the first term on the right-hand side of (27), we further estimate as

∥(1 − τλ)(u ∗(xt, yt+1, zt+1) − xt) − τ∇xK(u ∗(xt, yt+1, zt+1), yt+1, zt+1) + τ∇xK(xt, yt+1, zt)∥ 2 = (1 − τλ) 2∥u ∗(xt, yt+1, zt+1) − xt∥ 2 + 2τ (1 − τλ)⟨u ∗(xt, yt+1, zt+1) − xt, ∇xK(xt, yt+1, zt) − ∇xK(u ∗(xt, yt+1, zt+1), yt+1, zt+1)⟩ + τ 2∥∇xK(xt, yt+1, zt) − ∇xK(u ∗(xt, yt+1, zt+1), yt+1, zt+1)∥ 2. (29)
Next, we turn to estimating

$$\|\nabla_{\mathbf{x}}K(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\nabla_{\mathbf{x}}K(\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t+1}),\mathbf{y}_{t+1},\mathbf{z}_{t+1})\|$$ $$\leq\|\nabla_{\mathbf{x}}K(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\nabla_{\mathbf{x}}K(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t+1})\|$$ $$\quad+\|\nabla_{\mathbf{x}}K(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t+1})-\nabla_{\mathbf{x}}K(\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t+1}),\mathbf{y}_{t+1},\mathbf{z}_{t+1})\|.$$
∗(xt, yt+1, zt+1), yt+1, zt+1)∥. (30)
Note that, by definition, we have

$$(29)$$
$$(30)$$
$$\nabla_{\mathbf{x}}K_{\mathbb{I}}$$

∇xK(xt, yt+1, zt) − ∇xK(xt, yt+1, zt+1) = µ(zt+1 − zt).

Using this and the LK-Lipschitzness of ∇xK(·, yt+1, zt+1) as per Fact A.1, in (30), we obtain
∥∇xK(xt, yt+1, zt) − ∇xK(u
∗(xt, yt+1, zt+1), yt+1, zt+1)∥ ≤ µ∥zt+1 − zt∥ + LK∥u
∗(xt, yt+1, zt+1) − xt∥.

We plug this bound into the second term on the right-hand side of (29) after using Cauchy-Schwarz inequality, and then, we use Young's inequality to get

2τ (1 − τλ)⟨u ∗(xt, yt+1, zt+1) − xt, ∇xK(xt, yt+1, zt) − ∇xK(u ∗(xt, yt+1, zt+1), yt+1, zt+1)⟩ ≤ 2τ (1 − τλ)∥u ∗(xt, yt+1, zt+1) − xt∥(µ∥zt+1 − zt∥ + LK∥u ∗(xt, yt+1, zt+1) − xt∥) ≤ τ (1 − τλ)(2LK + µ)∥u ∗(xt, yt+1, zt+1) − xt∥ 2 + τ (1 − τλ)µ∥zt+1 − zt∥ 2.
Using the last two inequalities in (29), along with Young's inequality, we obtain

∥(1 − τλ)(u ∗(xt, yt+1, zt+1) − xt) − τ∇xK(u ∗(xt, yt+1, zt+1), yt+1, zt+1) + τ∇xK(xt, yt+1, zt)∥ 2 ≤ [(1 − τλ) 2 + τ (1 − τλ)(2LK + µ) + 2τ 2L 2 K]∥u ∗(xt, yt+1, zt+1) − xt∥ 2 + (τ (1 − τλ)µ + 2τ 2µ 2)∥zt+1 − zt∥ 2. (31)
We estimate the coefficient of the first term. First, note that 1 − τλ ≤ 1. As a result, we have

$$(1-\tau\lambda)^{2}+\tau(1-\tau\lambda)(2L_{K}+\mu)+2\tau^{2}L_{K}^{2}\leq1-2\tau\lambda+\tau^{2}\lambda^{2}+\tau(2L_{K}+\mu)+2\tau^{2}L_{K}^{2}$$ $$\leq1-2\tau\lambda+\frac{1}{6}\tau\lambda+\tau\lambda+\frac{1}{2}\tau\lambda+\frac{1}{12}\tau\lambda$$ $$=1-\frac{\tau\lambda}{4},$$
$$(31)$$

where in second inequality, we use τλ ≤
1 6
,LK =
1 2 λ and τµ ≤ τLK =
1 2 τλ.

Finally, since τ (1 − τλ)µ + 2τ 2µ 2 ≤ τµ + 2τ 2µ 2, the proof is completed after taking full expectation of (27) and plugging in (28) and (31). ■
Lemma A.5 (cf. Lemma 3.3). Let Assumption 1.1 *hold. Then, if* λ = 2LK and τ ≤1 6λ
, we have for the iterates of Algorithm 1 *that*

$$\mathbb{E}\varphi_{1/\lambda}(\mathbf{x}_{t+1},\mathbf{y}_{t+1},\mathbf{z}_{t+1})\leq\mathbb{E}\varphi_{1/\lambda}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t+1})-\frac{\tau\lambda^{2}}{16}\mathbb{E}\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{x}_{t}\|^{2}$$ $$+\left(\frac{\lambda\tau\mu}{2}+\lambda\tau^{2}\mu^{2}+\frac{\tau\lambda^{2}\mu^{2}}{8\gamma_{s}^{2}}\right)\mathbb{E}\|\mathbf{z}_{t}-\mathbf{z}_{t+1}\|^{2}+\frac{\lambda\tau^{2}\sigma^{2}}{2},$$  $\mathbb{E}\|\mathbf{u}^{*}\|$
$$(32)$$
2, (32)
where γs = 2µ + ρ∥A∥. Proof. By the definition of φ1/λ from (24) and u
∗(x, yt+1, zt+1) from (23), we have

Eφ1/λ(xt+1, yt+1, zt+1) ≤ EK(u ∗(xt, yt+1, zt+1), yt+1, zt+1) + λ2 E∥u ∗(xt, yt+1, zt+1) − xt+1∥ 2 ≤ EK(u ∗(xt, yt+1, zt+1), yt+1, zt+1) + λ 2 − τλ2 8 E∥u ∗(xt, yt+1, zt+1) − xt∥ 2 + λτµ 2+ λτ 2µ 2 E∥zt − zt+1∥ 2 + λτ 2σ 2 2 = Eφ1/λ(xt, yt+1, zt+1) − τλ2 8 E∥u ∗(xt, yt+1, zt+1) − xt∥ 2 + λτµ 2+ λτ 2µ 2 E∥zt − zt+1∥ 2 + λτ 2σ 2 2. (33)
We next bound the second term on the right-hand side by using Young's inequality as

$$\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t+1})-\mathbf{x}_{t}\|^{2}\geq\frac{1}{2}\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{x}_{t}\|^{2}-\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t+1})-\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t}\|^{2})$$ $$\geq\frac{1}{2}\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{x}_{t}\|^{2}-\frac{\mu^{2}}{\gamma_{\pi}^{2}}\|\mathbf{z}_{t}-\mathbf{z}_{t+1}\|^{2},$$
$$(33)$$

where the last line used (61).

We substitute the last inequality into (33) to conclude. ■
Since the previous result only allowed us to connect φ1/λ(xt+1, yt+1, zt+1) to φ1/λ(xt, yt+1, zt+1), we now need to analyze the effect of changing yt+1 and zt+1 in φ1/λ. The main idea of this lemma is similar to (Zhang & Luo, 2022),
where the difference lies in the fact that our potential involves the Moreau envelope of K(x, y, z) whereas the potential of (Zhang & Luo, 2022) involves K(x, y, z). Hence this work considers the change of the arguments in the function K instead of φ1/λ. Therefore, our proof uses the properties of the Moreau envelope which was not needed in (Zhang & Luo, 2022). Lemma A.6. (cf. Lemma 3.4) Suppose that Assumption 1.1 holds, for φ1/λ *defined in* (10), we have for the iterates of Algorithm 1 *that*

φ1/λ(xt, yt, zt) − φ1/λ(xt, yt+1, zt) ≥ ⟨yt − yt+1, Au ∗(xt, yt, zt) − b⟩ + γs 2 ∥u ∗(xt, yt, zt) − u ∗(xt, yt+1, zt)∥ 2, φ1/λ(xt, yt+1, zt) − φ1/λ(xt, yt+1, zt+1) ≥ µ 2 ⟨zt+1 − zt, 2u ∗(xt, yt+1, zt) − zt+1 − zt⟩ + γs 2 ∥u ∗(xt, yt+1, zt+1) − u ∗(xt, yt+1, zt)∥ 2,
$$(34)$$
where γs = 2µ + ρ∥A∥.

Proof. We first consider the change in y argument of φ1/λ. By using the definition of φ1/λ in (24), we have

φ1/λ(xt, yt, zt) − φ1/λ(xt, yt+1, zt) = K(u ∗(xt, yt, zt), yt, zt) + λ2 ∥xt − u ∗(xt, yt, zt)∥ 2 − K(u ∗(xt, yt+1, zt), yt+1, zt) − λ 2 ∥xt − u ∗(xt, yt+1, zt)∥ 2 = K(u ∗(xt, yt, zt), yt, zt) − K(u ∗(xt, yt, zt), yt+1, zt) + K(u ∗(xt, yt, zt), yt+1, zt) + λ2 ∥xt − u ∗(xt, yt, zt)∥ 2 − K(u ∗(xt, yt+1, zt), yt+1, zt) − λ 2 ∥xt − u ∗(xt, yt+1, zt)∥ 2, (35)
where the second equality adds and subtracts K(u
∗(xt, yt, zt), yt+1, zt).

From the definition of K in (22), it trivially follows that

$$K(\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t})$$

∗(xt, yt, zt), yt, zt) − K(u
∗(xt, yt, zt), yt+1, zt) = ⟨yt − yt+1, Au
∗(xt, yt, zt) − b⟩.

Next, we use the property that K(·, yt+1, zt) + λ 2
∥ · −xt∥
2is γs-strongly convex with minimizer u
∗(xt, yt+1, zt) (see Fact A.1 and (23)) to obtain

K(u ∗(xt, yt, zt), yt+1, zt) + λ 2 ∥xt − u ∗(xt, yt, zt)∥ 2 − K(u ∗(xt, yt+1, zt), yt+1, zt) − λ 2 ∥xt − u ∗(xt, yt+1, zt)∥ 2 ≥ γs 2 ∥u ∗(xt, yt, zt) − u ∗(xt, yt+1, zt)∥ 2.
Combining the last two estimates in (35) gives the first assertion.

Next, we analyze the effect of changing the z component in φ1/λ. Similar to the proof of the first assertion, we start with the definition of φ1/λ and then add and subtract K(u
∗(xt, yt+1, zt+1) to obtain

φ1/λ(xt, yt+1, zt) − φ1/λ(xt, yt+1, zt+1) = K(u ∗(xt, yt+1, zt), yt+1, zt) + λ2 ∥xt − u ∗(xt, yt+1, zt)∥ 2 − K(u ∗(xt, yt+1, zt+1), yt+1, zt+1) − λ 2 ∥xt − u ∗(xt, yt+1, zt+1)∥ 2 = K(u ∗(xt, yt+1, zt), yt+1, zt) − K(u ∗(xt, yt+1, zt), yt+1, zt+1) + K(u ∗(xt, yt+1, zt), yt+1, zt+1) + λ2 ∥xt − u ∗(xt, yt+1, zt)∥ 2 − K(u ∗(xt, yt+1, zt+1), yt+1, zt+1) − λ 2 ∥xt − u ∗(xt, yt+1, zt+1)∥
(36)  $\frac{1}{2}$
First, by definition, of K, it trivially follows that

$$K(\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t}),\mathbf{y}_{t+1},\mathbf{z}_{t})-K(\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t}),\mathbf{y}_{t+1},\mathbf{z}_{t+1})=\frac{\mu}{2}\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{z}_{t}\|^{2}\\ -\frac{\mu}{2}\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{z}_{t+1}\|^{2}.$$

For the remaining terms on the right-hand side, we again use that K(·, yt+1, zt+1) + λ2
∥ · −xt∥
2is γs-strongly convex with minimizer u
∗(xt, yt+1, zt+1) to deduce

K(u ∗(xt, yt+1, zt), yt+1, zt+1) + λ2 ∥xt − u ∗(xt, yt+1, zt)∥ 2 − K(u ∗(xt, yt+1, zt+1), yt+1, zt+1) − λ 2 ∥xt − u ∗(xt, yt+1, zt+1)∥ 2 ≥ γs 2 ∥u ∗(xt, yt+1, zt+1) − u ∗(xt, yt+1, zt)∥ 2.
Plugging in the last two estimates in (36) gives the second assertion. ■

Corollary A.7. Suppose that Assumption 1.1 holds, for φ1/λ *defined in* (10), if λ = 2LK and τ ≤
1
6λ Eφ1/λ(xt, yt, zt) − Eφ1/λ(xt+1, yt+1, zt+1) ≥ τλ2 16 E∥u ∗(xt, yt+1, zt) − xt∥ 2 − λτµ 2+ λτ 2µ 2 + τλ2µ 2 8γ 2 s E∥zt − zt+1∥ 2 − λτ 2σ 2 2 − ηE⟨Axt − b, Au ∗(xt, yt, zt) − b⟩ + µ 2 E⟨zt+1 − zt, 2u ∗(xt, yt+1, zt) − zt+1 − zt⟩,
, we have that where γs = 2µ + ρ∥A∥. Proof. We sum up the results in Lemma A.5 and Lemma A.6, plug in the definition of yt+1 and discard two nonnegative terms on the right-hand side to get the result. ■
Next, we analyze the rest of the terms appearing in the potential function. This lemma is only using the definition of d(y, z) and Ψ(z) and is equivalent to (Zhang & Luo, 2022) and hence we omit its proof. Notably, these bounds are agnostic to the algorithm used to generate the sequences. Note that the only difference is that in the result below, we do not use the definition of yt+1 whereas the proof in (Zhang & Luo, 2022) uses this definition. The rest of the estimations are precisely the same. Lemma A.8. (Zhang & Luo, 2020, Lemma 3.2, Lemma 3.3) For the functions d(y, z) and Ψ(z) *defined in* (12) and (7),we have

$$d(\mathbf{y}_{t+1},\mathbf{z}_{t+1})-d(\mathbf{y}_{t},\mathbf{z}_{t})\geq\eta_{j}(A\mathbf{x}_{t}-\mathbf{b},A\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{b})+\frac{\mu}{2}\langle\mathbf{z}_{t+1}-\mathbf{z}_{t},\mathbf{z}_{t+1}+\mathbf{z}_{t}-2\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t+1})\rangle,$$ $$\Psi(\mathbf{z}_{t+1})-\Psi(\mathbf{z}_{t})\leq\mu(\mathbf{z}_{t+1}-\mathbf{z}_{t},\mathbf{z}_{t}-\mathbf{x}^{*}(\mathbf{z}_{t}))+\frac{\mu}{2\sigma_{t}}\|\mathbf{z}_{t}-\mathbf{z}_{t+1}\|^{2},$$

where σ4 *is defined in* (25).

In the next lemma, we will join the previous lemmas and characterize the change in the potential function. Lemma A.9 (cf. Lemma 3.6). Let Assumption 1.1 *hold. By using the parameters* (25) in Algorithm *1, we obtain*

$$\mathbb{E}V_{t}-\mathbb{E}V_{t+1}\geq c_{p}\mathbb{E}\|\mathbf{z}_{t+1}-\mathbf{z}_{t}\|^{2}+c_{r}\mathbb{E}\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{x}_{t}\|^{2}+c_{v}\mathbb{E}\|A\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{b}\|^{2}-\frac{1}{2}\lambda\tau^{2}\sigma^{2},\tag{37}$$
where cβ =µ
50β
, cτ =
7τλ2
$${\frac{2}{0}},\,c_{\eta}={\frac{\eta}{4}}.$$

Proof. Combining Corollary A.7 and Lemma A.8, we obtain

E[Vt − Vt+1] = E-φ1/λ(xt, yt, zt) − φ1/λ(xt+1, yt+1, zt+1) + 2d(yt+1, zt+1) − 2d(yt, zt) + 2Ψ(zt) − 2Ψ(zt+1) ≥ τλ2 16 E∥u ∗(xt, yt+1, zt) − xt∥ 2 − λτµ 2+ λτ 2µ 2 + τλ2µ 2 8γ 2s E∥zt − zt+1∥ 2 − λτ 2σ 2 2 − ηE⟨Axt − b, Au ∗(xt, yt, zt) − b⟩ + µ 2 E⟨zt+1 − zt, 2u ∗(xt, yt+1, zt) − zt − zt+1⟩ + 2ηE⟨Axt − b, Ax ∗(yt+1, zt) − b⟩ + µE⟨zt+1 − zt, zt+1 + zt − 2x ∗(yt+1, zt+1)⟩ − 2µE⟨zt+1 − zt, zt − x¯ ∗(zt)⟩ − µ σ4 E∥zt − zt+1∥ 2. (38)
We next manipulate the terms on the right-hand side. First, by adding and subtracting Axt on the second argument of the first inner product on the right-hand side, we get

$$-\eta(A\mathbf{x}_{t}-\mathbf{b},A\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t},\mathbf{z}_{t})-\mathbf{b})$$

∗(xt, yt, zt) − b⟩ = −η∥Axt − b∥
2 − η⟨Axt − b, Au
∗(xt, yt, zt) − Axt⟩.

Consequently, we use this estimate and rewrite the third inner product on the right-hand side of (38) with quadratics to have

$$-\eta(A{\bf x}_{t}-{\bf b},A{\bf u}^{*}({\bf x}_{t},{\bf y}_{t},{\bf z}_{t})-{\bf b})+2\eta(A{\bf x}_{t}-{\bf b},A{\bf x}^{*}({\bf y}_{t+1},{\bf z}_{t})-{\bf b})$$ $$=-\eta\|A{\bf x}_{t}-A{\bf x}^{*}({\bf y}_{t+1},{\bf z}_{t})\|^{2}+\eta\|A{\bf x}^{*}({\bf y}_{t+1},{\bf z}_{t})-{\bf b}\|^{2}-\eta(A{\bf x}_{t}-{\bf b},A{\bf u}^{*}({\bf x}_{t},{\bf y}_{t},{\bf z}_{t})-A{\bf x}_{t}).$$

Second, adding and subtracting 2xt in the second argument of the second inner product on the right-hand side of (38) gives

µ 2 ⟨zt+1 − zt, 2u ∗(xt, yt+1, zt) − zt − zt+1⟩ = µ 2 ⟨zt+1 − zt, 2u ∗(xt, yt+1, zt) − 2xt⟩ + µ 2 ⟨zt+1 − zt, 2xt − zt − zt+1⟩. For the right-hand side of this term, note that zt+1 = zt + β(xt − zt) ⇐⇒ 2xt − 2zt = 2 β (zt+1 − zt) and hence µ 2 ⟨zt+1 − zt, 2xt − zt − zt+1⟩ = µ 2 ⟨zt+1 − zt, 2xt − 2zt⟩ + µ 2 ⟨zt+1 − zt, zt − zt+1⟩ = µ 2 2 β − 1 ∥zt − zt+1∥ 2 ≥ µ 2β ∥zt − zt+1∥ 2,
where the last inequality is due to β ≤ 1. Next, for the remaining inner products in (38), we have

$$\begin{array}{l}{{\mu(\mathbf{z}_{t+1}-\mathbf{z}_{t},\mathbf{z}_{t+1}+\mathbf{z}_{t}-2\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t+1}))-2\mu(\mathbf{z}_{t+1}-\mathbf{z}_{t},\mathbf{z}_{t}-\tilde{\mathbf{x}}^{*}(\mathbf{z}_{t}))}}\\ {{=\mu\|\mathbf{z}_{t+1}-\mathbf{z}_{t}\|^{2}+2\mu(\mathbf{z}_{t+1}-\mathbf{z}_{t},\tilde{\mathbf{x}}^{*}(\mathbf{z}_{t})-\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t+1})).}}\end{array}$$

We can use Cauchy-Schwarz, triangle and Young's inequalities on the second term here to get

$$\langle\mathbf{z}_{t+1}-\mathbf{z}_{t},\mathbf{x}^{*}(\mathbf{z}_{t})-\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t+1})\rangle\geq-\|\mathbf{z}_{t+1}-\mathbf{z}_{t}\|(\|\mathbf{x}^{*}(\mathbf{z}_{t})-\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t})\|+\|\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t+1})\|)$$ $$\geq-\left(\frac{1}{2\zeta}+\frac{1}{\sigma_{d}}\right)\|\mathbf{z}_{t+1}-\mathbf{z}_{t}\|^{2}-\frac{\zeta}{2}\|\mathbf{x}^{*}(\mathbf{z}_{t})-\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t})\|^{2},$$
$$(39)$$

where the last step also used (63). Consequently, plugging in this estimate to (39), we obtain

$$\mu\langle\mathbf{z}_{t+1}-\mathbf{z}_{t},\mathbf{z}_{t+1}+\mathbf{z}_{t}-2\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t+1})\rangle-2\mu\langle\mathbf{z}_{t+1}-\mathbf{z}_{t},\mathbf{z}_{t}-\bar{\mathbf{x}}^{*}(\mathbf{z}_{t})\rangle$$ $$\geq\left(\mu-\frac{\mu}{\zeta}-\frac{2\mu}{\sigma_{4}}\right)\|\mathbf{z}_{t+1}-\mathbf{z}_{t}\|^{2}-\mu\zeta\|\bar{\mathbf{x}}^{*}(\mathbf{z}_{t})-\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t})\|^{2}.$$

After combining these estimates in (38), we get

E[Vt] − E[Vt+1] ≥ τλ2 16 E∥u ∗(xt, yt+1, zt) − xt∥ 2 − 1 2 λτµ + λτ 2µ 2 + τλ2µ 2 8γ 2 s + µ ζ + 3µ σ4 − µ − µ 2β E∥zt − zt+1∥ 2 − 1 2 λτ 2σ 2 − ηE⟨Axt − b, Au ∗(xt, yt, zt) − Axt⟩ − ηE∥Axt − Ax ∗(yt+1, zt)∥ 2 + ηE∥Ax ∗(yt+1, zt) − b∥ 2 − µζE∥x¯ ∗(zt) − x ∗(yt+1, zt)∥ 2 + µE⟨zt+1 − zt, u ∗(xt, yt+1, zt) − xt⟩. (40)
We will now operate on some of terms from the right-hand side of (40), by using Lemma A.11 and A.12. First, we have by Cauchy-Schwarz and Young's inequalities that

− η⟨Axt − b, Au ∗(xt, yt, zt) − Axt⟩ ≥ − η 4 ∥Axt − b∥ 2 − η∥Au ∗(xt, yt, zt) − Axt∥ 2 ≥ − η 4 ∥Axt − b∥ 2 − 2η∥Au ∗(xt, yt, zt) − Au ∗(xt, yt+1, zt)∥ 2 − 2η∥Au ∗(xt, yt+1, zt) − Axt∥ 2. Next, by using the Lipschitzness of u ∗(xt, ·, zt) from (60), we have
$$\|A\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t},\mathbf{z}_{t})-A\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})\|^{2}\leq\|A\|^{2}\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t},\mathbf{z}_{t})-\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})\|^{2}$$ $$\leq\frac{\|A\|^{2}}{\gamma_{s}^{2}}\|\mathbf{y}_{t}-\mathbf{y}_{t+1}\|^{2}$$ $$=\frac{\|A\|^{4}\eta^{2}}{\gamma_{s}^{2}}\|A\mathbf{x}_{t}-\mathbf{b}\|^{2},$$
where the last step also used the definition of yt+1. Using this estimation along with (66) gives

− η⟨Axt − b, Au ∗(xt, yt, zt) − Axt⟩ ≥ − η4 + 2∥A∥ 4η 3 γ 2 s ∥Axt − b∥ 2 − 2η∥A∥ 2∥u ∗(xt, yt+1, zt) − xt∥ 2 ≥ − η∥A∥ 2λ 2 2γ 2+ 4∥A∥ 6η 3λ 2 γ 2γ 2 s + 2η∥A∥ 2∥u ∗(xt, yt+1, zt) − xt∥ 2 − η 2 + 4∥A∥ 4η 3 γ 2 s ∥Ax ∗(yt+1, zt) − b∥ 2.
We next have by Young's inequality that for any θ > 0:

$$\mu({\bf z}_{t+1}-{\bf z}_{t},{\bf u}^{*}({\bf x}_{t},{\bf y}_{t+1},{\bf z}_{t})-{\bf x}_{t})\geq-\frac{\mu}{4\theta}\|{\bf z}_{t+1}-{\bf z}_{t}\|^{2}-\theta\mu\|{\bf u}^{*}({\bf x}_{t},{\bf y}_{t+1},{\bf z}_{t})-{\bf x}_{t}\|^{2}.$$

The inequality derived in (65) directly implies

$$-\eta\|A{\bf x}_{t}-A{\bf x}^{*}({\bf y}_{t+1},{\bf z}_{t})\|^{2}\geq-\frac{\eta\|A\|^{2}\lambda^{2}}{\gamma^{2}}\|{\bf x}_{t}-{\bf u}^{*}({\bf x}_{t},{\bf y}_{t+1},{\bf z}_{t})\|^{2}.$$

The key global error bound given in Lemma A.12 originally proved in (Zhang & Luo, 2022) results in

$$\mathbf{\hat{\tau}}\geq-6\mu\beta$$

−6µβ∥x
∗(yt+1, zt) − x¯
∗(zt)∥
2 ≥ −6µβσ¯
2∥Ax
∗(yt+1, zt) − b∥
2.

Combining these estimates in (40) leads to

E[Vt] − E[Vt+1] ≥ − 12 λτµ + λτ 2µ 2 + τλ2µ 2 8γ 2 s + µ ζ + 3µ σ4 − µ − µ 2β + µ 4θ E∥zt − zt+1∥ 2 − 1 2 λτ 2σ 2 + τλ2 16− 3∥A∥ 2λ 2η 2γ 2− 4∥A∥ 6η 3λ 2 γ 2 s γ 2− 2η∥A∥ 2 − µθE∥u ∗(xt, yt+1, zt) − xt∥ 2 + η 2 − 4∥A∥ 4η 3 γ 2s − 6µβσ¯ 2 E∥Ax ∗(yt+1, zt) − b∥ 2. (41)
We now estimate the coefficients inside the parantheses, with straightforward but tedious calculations which follow from the parameter settings.

First, we estimate the coefficient of E∥zt − zt+1∥
2in (41): Let µ ≥ 4Lf , we have σ4 ≥
1 2 because σ4 =
µ−Lf µ. Then letting ζ = 6*β, β <* 1 30 , we have

$$\mu-\frac{3\mu}{\sigma_{4}}\geq-5\mu\geq-\frac{\mu}{6\beta},\quad\frac{\mu}{\zeta}=\frac{\mu}{6\beta}.$$
Therefore, we have that
$${\frac{\mu}{2\beta}}+\mu-{\frac{3\mu}{\sigma_{4}}}-{\frac{\mu}{\zeta}}\geq\left({\frac{1}{2}}-{\frac{1}{6}}-{\frac{1}{6}}\right){\frac{\mu}{\beta}}\geq{\frac{\mu}{6\beta}}.$$
(42)  $$\begin{array}{l}\mbox{(42)}\end{array}$$ . 
Hence, we estimate:

coefficient of $\mathbb{E}\|\mathbf{z}_{t}-\mathbf{z}_{t+1}\|\geq-\frac{1}{2}\lambda\tau\mu-\lambda\tau^{2}\mu^{2}-\frac{\tau\lambda^{2}\mu^{2}}{8\gamma_{s}^{2}}+\frac{\mu}{6\beta}-\frac{\mu}{8\beta}$.  
Let η =η
′
2∥A∥2 , θ = 2*β, η*′ ≤
1 40 , and µ = max{2, 4Lf }, λ = 2LK = 2(Lf +ρ∥A∥+µ), τ ≤1 10λ2 , and γs = µ−Lf +γ from Fact A.1. We have −λτµ ≥ − µ 10 and −2λτ 2µ 2 ≥ − µ 100 , then

$${\mathrm{coefficient~of~}}\mathbb{E}\|\mathbf{z}_{t}-\mathbf{z}_{t+1}\|\geq{\frac{\mu}{24\beta}}-{\frac{\mu}{20}}-{\frac{\mu}{100}}-\tau\lambda^{2}{\frac{\mu^{2}}{(\mu-L_{f}+\lambda)^{2}}}.$$
By $\beta\leq1/30$, we have $\frac{1}{24\beta}-\frac{1}{20}-\frac{1}{100}\geq\frac{1}{30\beta}$. In addition, using $\tau\lambda^{2}\frac{\mu^{2}}{(\mu-L_{f}+\lambda)^{2}}\leq\tau\lambda^{2}$,  $$\text{coefficient of}\mathbb{E}\|\mathbf{z}_{t}-\mathbf{z}_{t+1}\|\geq\frac{\mu}{30\beta}-\frac{1}{10}\stackrel{{\mu\geq2}}{{\geq}}\frac{\mu}{50\beta}.$$
1
10 , we fanally obtain:
. (43)
Then we estimate the coefficient of E∥u
$$\mathbb{E}\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{x}_{t}\|^{2}\,{\hat{\mathbf{r}}}$$
2in (41).

From above assumptions, we can easily get γ =
(µ−Lf )λ µ−Lf +λ ≥
1 2 because λ ≥ µ ≥ 2. Moreover, we assume η
′ ≤
τ 40 ,η
′
µ−Lf +λ ≤
τ 10 , β ≤
τ 40 First, by our new notations, we have

coefficient of $\mathbb{E}\|\mathbf{u}^{*}\left(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t}\right)-\mathbf{x}_{t}\|^{2}=\frac{\tau\lambda^{2}}{16}-\frac{3\eta^{\prime}\lambda^{2}}{4\gamma^{2}}-\frac{\eta^{\prime3}\lambda^{2}}{2\gamma^{2}\gamma_{s}^{2}}-\eta^{\prime}-2\mu\beta$
$$(43)$$
19

By $\gamma\geq\frac{1}{2}$ and the definition of $\gamma_{\mu}$, we have $-\frac{3\eta^{\prime}\lambda^{2}}{4\gamma^{2}}\geq-3\eta^{\prime}\lambda^{2}$, $-\frac{\eta^{\prime2}\lambda^{2}}{2\gamma^{2}\gamma^{2}}\geq-\frac{\eta^{\prime3}\lambda^{2}}{(\mu-L_{f}+\lambda)^{2}}$, Then  $$\text{coefficient of}\mathbb{E}\|\mathbf{u}^{\star}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{x}_{t}\|^{2}\geq\frac{\tau\lambda^{2}}{16}-3\eta^{\prime}\lambda^{2}-\frac{2\eta^{\prime3}\lambda^{2}}{(\mu-L_{f}+\lambda)^{2}}-\eta^{\prime}-2\mu\beta.$$
With 2 ≤ µ ≤ λ, η ′ ≤τ 100 ,η ′ µ−Lf +λ ≤ τ 10 , β ≤τ 200 , we can obtain −3η ′λ 2 ≥ −3τλ2 400 , −2η ′3λ 2 (µ−Lf +λ) −η ′ ≥ − τ 100 ≥ −τλ2 100 , −2µβ ≥ −τµ 50 µ≤λ ≥ −τλ 50 ≥ −τλ2 100 . Hence, coefficient of E∥u ∗(xt, yt+1, zt) − xt∥ 2 ≥ τλ2 16 − 3τλ2 100− τλ2 400 − τλ2 400 − τλ2 100 = 7τλ2 400
2 ≥ −λ
2τ
2
400 ≥ −λ
2τ
400 ,
. (44)
$${\frac{\lambda^{2}}{00}}-{\frac{\tau\lambda^{2}}{100}}={\frac{7\tau\lambda^{2}}{400}}.$$
Last, we estimate the coefficient of E∥Ax ∗(yt+1, zt) − b∥ 2in (41). By 6µβσ¯ 2 ≤ η 6 and the definition η ′, γs, we have η ′ µ−Lf +λ ≤ τ 10 ≥ −ητ2 100 ≥ − η 100 and −6µβσ¯ 2 ≥ −η6 . Hence, we have − 4∥A∥ 2η 3 γ2s= −η ′2η (µ−Lf +λ) 2 coefficient of E∥Ax ∗(yt+1, zt) − b∥ 2 ≥ η 2 −η 100 − η 6 ≥ η 4 . (45)
Plugging (43), (44) and (45) to (41), we finish the proof. ■
A.2. Proof of Theorem 3.1 Proof of Theorem *3.1.* We start from the result in Lemma A.9. First, it follows from the definition of zt+1 that

$$(44)$$
$$(45)$$
$$\|\mathbf{z}_{t}-\mathbf{z}_{t+1}\|=\beta\|\mathbf{x}_{t}-\mathbf{z}_{t}\|.$$
$$(47)$$

So, we rewrite (37), as:

EVt − EVt+1 ≥ β
$$\mathbb{E}V_{t+1}\geq\beta^{2}c_{\beta}\mathbb{E}\|\mathbf{x}_{t}-\mathbf{z}_{t}\|^{2}+c_{r}\mathbb{E}\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{x}_{t})\|^{2}+c_{\eta}\mathbb{E}\|A\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{b}\|^{2}-\frac{1}{2}\lambda\tau^{2}\sigma^{2}.$$
2. (46)
For t > 0, we have Vt ≥ f, which is proven in Lemma A.13. It then follows that

$$\sum_{t=0}^{T-1}(\mathbb{E}V_{t}-\mathbb{E}V_{t+1})=V_{0}-\mathbb{E}V_{T}\leq V_{0}-\underline{f}.\tag{1}$$

Then, summing up (46), using (47), and the fact that cτ = Θ(τ ), cη = Θ(τ ), β 2cβ = Θ(τ ) from (25), we have

$$V_{0}-\underline{f}+\frac{1}{2}T\lambda\tau^{2}\sigma^{2}\geq\sum_{t=1}^{T}C_{0}\tau\left[\mathbb{E}\|\mathbf{x}_{t}-\mathbf{z}_{t}\|^{2}+\mathbb{E}\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{x}_{t}\|^{2}+\mathbb{E}\|A\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{b}\|^{2}\right],$$  multi-quark $C$
for some explicit constant C0. Dividing both sides by T, rearranging and using the definition τ =1 6λ2
√T
gives

$$\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}\|\mathbf{x}_{t}-\mathbf{z}_{t}\|^{2}+\mathbb{E}\|\mathbf{u}^{*}(\mathbf{x}_{t},\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{x}_{t}\|^{2}+\mathbb{E}\|A\mathbf{x}^{*}(\mathbf{y}_{t+1},\mathbf{z}_{t})-\mathbf{b}\|^{2}\leq\frac{1}{C_{0}\sqrt{T}}\left(6\lambda(V_{0}-\underline{f})+\frac{\sigma^{2}}{12}\right).\tag{48}$$

Since we have
∇Ψ(zt) = µ(zt − x¯
∗(zt)),