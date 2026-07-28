# Stochastic Regret Guarantees for Online Zeroth- and First-Order Bilevel Optimization

Anonymous Authors<sup>1</sup>

### Abstract

Online bilevel optimization (OBO) is a powerful framework for machine learning problems where both outer and inner objectives evolve over time, requiring dynamic updates. Current OBO approaches rely on deterministic *window-smoothed* regret minimization, which may not accurately reflect system performance when functions change rapidly. In this work, we introduce a novel search direction and show that both first- and zerothorder (ZO) stochastic OBO algorithms leveraging this direction achieve sublinear stochastic bilevel regret without window smoothing. Beyond these guarantees, our framework enhances efficiency by: (i) reducing oracle dependence in hypergradient estimation, (ii) updating inner and outer variables alongside the linear system solution, and (iii) employing ZO-based estimation of Hessians, Jacobians, and gradients. Experiments on online parametric loss tuning and black-box adversarial attacks validate our approach.

# 1. Introduction

Bilevel optimization (BO) minimizes an outer objective dependent on an inner problem's solution. Originating in game theory [\(Stackelberg,](#page-10-0) [1952\)](#page-10-0) and formalized in mathematical optimization [\(Bracken & McGill,](#page-8-0) [1973\)](#page-8-0), BO finds applications in operations research, engineering, economics [\(Dempe,](#page-8-1) [2002\)](#page-8-1), and image processing [\(Crock](#page-8-2)[ett et al.,](#page-8-2) [2022\)](#page-8-2). Recently, BO has gained traction in machine learning, including hyperparameter optimization [\(Franceschi et al.,](#page-8-3) [2018\)](#page-8-3), meta-learning [\(Finn et al.,](#page-8-4) [2017\)](#page-8-4), reinforcement learning [\(Stadie et al.,](#page-10-1) [2020\)](#page-10-1), and neural architecture search [\(Liu et al.,](#page-9-0) [2018a\)](#page-9-0).

In the *offline setting*, BO solves the following problem:

$$\begin{aligned} \mathbf{x}^* &\in \operatorname{argmin}_{\mathbf{x} \in \mathbb{R}^{d_1}} f(\mathbf{x}, \mathbf{y}^*(\mathbf{x})) \\ \text{subj. to } \mathbf{y}^*(\mathbf{x}) &= \operatorname{argmin}_{\mathbf{y} \in \mathbb{R}^{d_2}} g(\mathbf{x}, \mathbf{y}), \end{aligned} \quad (\text{BO})$$

where f and g are the outer and inner objectives, and x and y are their respective optimization variables.

OBO [\(Tarzanagh et al.,](#page-10-2) [2024\)](#page-10-2) addresses dynamic scenarios where objectives evolve over time, requiring the agent to update the outer decision in response to the optimal inner decision. Similar to online single-level optimization (OSO) [\(Zinkevich,](#page-10-3) [2003\)](#page-10-3), OBO involves iterative decisionmaking without prior knowledge of outcomes [\(Tarzanagh](#page-10-2) [et al.,](#page-10-2) [2024;](#page-10-2) [Lin et al.,](#page-9-1) [2024;](#page-9-1) [Bohne et al.,](#page-8-5) [2024\)](#page-8-5). Let T be the total number of rounds. Define x<sup>t</sup> ∈ X ⊂ <sup>R</sup> <sup>d</sup><sup>1</sup> as the decision variable and f<sup>t</sup> : X × <sup>R</sup> <sup>d</sup><sup>2</sup> → <sup>R</sup> as the outer function. Similarly, define y<sup>t</sup> ∈ <sup>R</sup> <sup>d</sup><sup>2</sup> and g<sup>t</sup> : X × <sup>R</sup> <sup>d</sup><sup>2</sup> → <sup>R</sup> for the inner problem, where y ∗ t (x) = argminy∈Rd<sup>2</sup> gt(x, y). OBO can be seen as a *single-player* problem, where the player selects x<sup>t</sup> without knowing y ∗ t (x), using y<sup>t</sup> as an estimate based on gt. Alternatively, it can be framed as a *two-player* game [\(Stackelberg,](#page-10-0) [1952\)](#page-10-0), where the leader (xt) competes with the follower (yt), who selects y ∗ t (x) based on limited knowledge of gt; see Section [2.](#page-1-0) This framework includes online and adversarial variants of [\(BO\)](#page-0-0), such as online actor-critic algorithms [\(Zhou et al.,](#page-10-4) [2020\)](#page-10-4), online meta-learning [\(Finn et al.,](#page-8-6) [2019\)](#page-8-6), and online hyperparameter optimization [\(Lin et al.,](#page-9-1) [2024\)](#page-9-1). The inner and outer functions may be time-varying, adversarial, unavailable *a priori*, and require *nonstationary* optimization.

#### 1.1. Our Contributions

This paper addresses stochastic OBO, introducing novel first<sup>1</sup> - and zeroth-order methods to minimize stochastic bilevel regret. Key contributions are summarized below.

• Stochastic regret minimization without windowsmoothing. Existing OBO methods [\(Tarzanagh et al.,](#page-10-2) [2024;](#page-10-2) [Lin et al.,](#page-9-1) [2024;](#page-9-1) [Huang et al.,](#page-9-2) [2023;](#page-9-2) [Bohne et al.,](#page-8-5) [2024\)](#page-8-5) rely on deterministic *window-smoothed* regret minimization, which may not accurately reflect system performance when functions change rapidly. We address these limitations by introducing a novel search direction (Section [3\)](#page-2-0) and proving that both first-order and ZO methods achieve sublinear *stochastic bilevel regret without window-smoothing (*w = 1*)*; see Theorems [3.6](#page-4-0) and [4.2](#page-5-0) and Table [1.](#page-1-1)

• OBO with function value oracle feedback. In large-scale

<sup>1</sup>Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

<sup>1</sup> First-order refers to the setting where only partial gradients of the leader objective f<sup>t</sup> are accessible, while second-order information is still required for the follower objective gt; refer to Section [3.](#page-2-0)

058

071

074

076

078

087 088

090 091

093 094

096

098

100

104

106

108 109

| OBO     | Window Size     | System            | Stochastic | Const.      | Only Func. | Local               |
|---------|-----------------|-------------------|------------|-------------|------------|---------------------|
| Method  | in Regret ( w ) | Iters.            | Regret     | Regret Min. | Feedback   | Regret Bound        |
| OAGD    | o ( T )         | N.A. (Exact)      | ✗          | ✗           | ✗          |                     |
|         |                 |                   |            |             |            | w + H 1 ,T + H 2 ,T |
| SOBOW   | o ( T )         | O ( κ g log κ g ) | ✗          | ✗           | ✗          |                     |
|         |                 |                   |            |             |            | w + V T + H 2 ,T    |
| SOBBO   | o ( T )         | O ( κ g log κ g ) | ✓          | ✓           | ✗          |                     |
|         |                 |                   |            |             |            | 2 + V T + H 2 ,T    |
| SOGD    | 1               | 1                 | ✓          | ✓           | ✗          | T                   |
|         |                 |                   |            |             |            | 3 ( σ               |
|         |                 |                   |            |             |            | 2 + ∆ T ) + T       |
|         |                 |                   |            |             |            | 3 Ψ T               |
| ZO-SOGD | 1               | 1                 | ✓          | ✓           | ✓          | ( d 1 + d 2 )       |
|         |                 |                   |            |             |            | 4 T                 |
|         |                 |                   |            |             |            | 3 (ˆ σ              |
|         |                 |                   |            |             |            | 2 + ∆ˆ              |
|         |                 |                   |            |             |            | T ) + ( d 1 + d 2 ) |
|         |                 |                   |            |             |            | 2 T                 |
|         |                 |                   |            |             |            | 3 Ψˆ                |

Table 1. Comparison of OBO algorithms based on regret window size (w), system solver iterations, stochastic regret, constrained regret minimization, function feedback settings, and local regret bounds. Here, κ<sup>g</sup> denotes the condition number of gt, while V<sup>T</sup> , Hp,T , ∆<sup>T</sup> , Ψ<sup>T</sup> , ∆ˆ <sup>T</sup> , and Ψˆ <sup>T</sup> are defined in [\(10\)](#page-3-0), [\(13\)](#page-3-1), and [\(25\)](#page-5-1), respectively. The compared algorithms include OAGD [\(Tarzanagh et al.,](#page-10-2) [2024\)](#page-10-2), SOBOW [\(Lin et al.,](#page-9-1) [2024\)](#page-9-1), and SOBBO [\(Bohne et al.,](#page-8-5) [2024\)](#page-8-5).

and black-box settings [\(Chen et al.,](#page-8-7) [2017;](#page-8-7) [Nesterov,](#page-9-3) [2005\)](#page-9-3), first- and second-order information is often unavailable or costly. Constructing accurate (hyper)-gradient estimators using only function value oracles is particularly challenging due to BO's nested structure. Existing methods rely on gradient, Hessian, and Jacobian oracles, limiting scalability [\(Franceschi et al.,](#page-8-8) [2017;](#page-8-8) [Ghadimi & Wang,](#page-8-9) [2018\)](#page-8-9). We propose Algorithm [2,](#page-5-2) which estimates Hessians, Jacobians, and gradients using function value oracles, achieving sublinear local regret (Theorem [4.2\)](#page-5-0).

• OBO with one subproblem solver iteration. A major challenge in BO is solving implicit systems to approximate the hypergradient [\(Ji et al.,](#page-9-4) [2021;](#page-9-4) [Chen et al.,](#page-8-10) [2021\)](#page-8-10). While efficient offline BO methods exist [\(Ji et al.,](#page-9-4) [2021;](#page-9-4) [Dagreou](#page-8-11) ´ [et al.,](#page-8-11) [2022\)](#page-8-11), extending them to OBO is difficult due to timevarying objectives. SOBOW [\(Lin et al.,](#page-9-1) [2024\)](#page-9-1) partially addresses this using a conjugate gradient (CG) algorithm with increasing iterations (Table [1\)](#page-1-1). We improve upon SOBOW by introducing Algorithms [1](#page-2-1) and [2,](#page-5-2) which require only a *single* subproblem solver iteration.

# 2. Preliminaries

Notation. R <sup>d</sup> denotes the d-dimensional real space, with R d <sup>+</sup> and <sup>R</sup> d ++ as its positive and negative orthants. Vectors are bold lower-case letters (e.g., x, y), with ⟨x, y⟩ for inner product and ∥·∥ for Euclidean norm. A gradient is ∇x, with ∇<sup>2</sup> xy = ∇x∇y. A function is L-smooth if its gradient is L-Lipschitz. The Euclidean projection onto a convex set X is Π<sup>X</sup> (z) = argminx∈X (1/2)∥x−z∥ . The set {1, . . . , T} is denoted by [T], and <sup>E</sup>[·] represents expectation. Lastly, O(·) hides problem-independent constants.

Stochastic OBO Setting. Let T be the total rounds [\(Tarzanagh et al.,](#page-10-2) [2024\)](#page-10-2). Define x<sup>t</sup> ∈ X ⊂ <sup>R</sup> <sup>d</sup><sup>1</sup> as the decision variable and f<sup>t</sup> : X × <sup>R</sup> <sup>d</sup><sup>2</sup> as the outer objective. The inner decision variable and objective are y<sup>t</sup> ∈ <sup>R</sup> <sup>d</sup><sup>2</sup> and g<sup>t</sup> : X × <sup>R</sup> d<sup>2</sup> , where the optimal inner decision is:

$$\mathbf{y}_t^*(\mathbf{x}) \in \operatorname{argmin}_{\mathbf{y} \in \mathbb{R}^{d_2}} \left\{ g_t(\mathbf{x}, \mathbf{y}) := \mathbb{E}_{\zeta_t \sim \mathcal{D}_g} [g_t(\mathbf{x}, \mathbf{y}; \zeta_t)] \right\}. \quad (1)$$

Further, we have

$$f_t(\mathbf{x}, \mathbf{y}_t^*(\mathbf{x})) := \mathbb{E}_{\xi_t \sim \mathcal{D}_f} [f_t(\mathbf{x}, \mathbf{y}_t^*(\mathbf{x}); \xi_t)].$$

Here, (D<sup>f</sup> , Dg) are data distributions. Note that our setting is stochastic, and only noisy evaluations of the function, gradient, and Hessian are accessible.

Unlike OSO, where true losses are revealed immediately, in OBO, the outer function ft(x, y ∗ t (x)) is unavailable for updating xt. Moreover, ft(x, y ∗ t (x)) is typically non-convex in x, making standard regret definitions from online convex optimization [\(Hazan,](#page-9-5) [2016b\)](#page-9-5) inapplicable.

Given a sequence {α<sup>t</sup> ∈ <sup>R</sup>++} T <sup>t</sup>=1, we define the following notion of *bilevel local regret*:

$$\text{BL-Reg}_T := \sum_{t=1}^T \mathbb{E} \left[ \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \right], \quad (2a)$$

with

$$\begin{aligned} & \mathcal{P}_{\mathcal{X}, \alpha_t} (\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))) \\ &= \frac{1}{\alpha_t} \left( \mathbf{x}_t - \Pi_{\mathcal{X}} [\mathbf{x}_t - \alpha_t \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))] \right). \end{aligned} \quad (2b)$$

The local regret [\(2\)](#page-1-2) compares the leader's decision x<sup>t</sup> to the stationary points x ∗ t satisfying P<sup>X</sup> ,α<sup>t</sup> (x ∗ t ; ∇ft(x ∗ t , y ∗ t (x ∗ t ))) = 0. This can also be viewed as dynamic local regret, as the baseline corresponds to a stationary point of the leader's objective ft.

Previous work on (nonconvex) OBO examined unconstrained local regret using window-smoothed objectives: Ft,w(x, y) = (1/w) P<sup>w</sup>−<sup>1</sup> <sup>i</sup>=0 ft−i(x, y). For w = 1 and X = R d<sup>1</sup> , this reduces to [\(2\)](#page-1-2). [Tarzanagh et al.](#page-10-2) [\(2024\)](#page-10-2); [Lin et al.](#page-9-1) [\(2024\)](#page-9-1) showed that w = o(T) ensures sublinear regret under slow variations in {Ft,w} T <sup>t</sup>=1, while rapid changes can lead to deviations. However, smoothing may misrepresent regret (Figure [1\)](#page-2-2). This paper introduces a new projection-based local regret notion [\(2\)](#page-1-2) without smoothing, and establishes sublinear regret for constrained OBO.

Online Gradient Descent (OGD). One of the most widely used algorithms for online (single-level) optimization is

114 115 116

118

124

126

128

131

134

136

138

151

154

158

160

164

Figure 1. Smoothly and rapidly changing f<sup>t</sup> in OBO with gt(xt, yt) = (y<sup>t</sup> − cos(xt))<sup>2</sup> , a<sup>t</sup> = 1 + 0.5 sin(t), b<sup>t</sup> = 1 + sin(0.5t), and c<sup>t</sup> = 10bt.

OGD [\(Zinkevich,](#page-10-3) [2003\)](#page-10-3). The procedure for OGD is as follows: For each t ∈ [T], the algorithm selects x<sup>t</sup> ∈ X , observes the function f<sup>t</sup> : X ⊂ <sup>R</sup> <sup>d</sup> → <sup>R</sup>, and updates according to

$$\mathbf{x}_{t+1} = \Pi_{\mathcal{X}}(\mathbf{x}_t - \alpha_t \nabla f_t(\mathbf{x}_t)), \quad \alpha_t > 0. \quad (\text{OGD})$$

In the following, we adapt [OGD](#page-2-3) to OBO and introduce a novel framework that requires limited feedback and can utilize ZO updates within a single-loop structure.

# 3. Stochastic OBO with Access to First and Second Order Oracles

To adapt [OGD](#page-2-3) to OBO, [Tarzanagh et al.](#page-10-2) [\(2024\)](#page-10-2); [Lin et al.](#page-9-1) [\(2024\)](#page-9-1); [Bohne et al.](#page-8-5) [\(2024\)](#page-8-5) developed a variant alternating between inner and outer [OGD,](#page-2-3) achieving sublinear bilevel regret bounds. We introduce a new search direction that enables sublinear bilevel regret without window smoothing.

To compute the hypergradient ∇ft(x, y ∗ t (x)) where y ∗ t (x) is defined in [\(1\)](#page-1-3), since ∇ygt(x, y ∗ t (x)) = 0, using the implicit function theorem, yields

$$\begin{aligned}\nabla f_t(\mathbf{x}, \mathbf{y}_t^*(\mathbf{x})) &= \nabla_{\mathbf{x}} f_t(\mathbf{x}, \mathbf{y}_t^*(\mathbf{x})) \\ &\quad + \nabla \mathbf{y}_t^*(\mathbf{x}) \nabla_{\mathbf{y}} f_t(\mathbf{x}, \mathbf{y}_t^*(\mathbf{x})),\end{aligned}\quad (3)$$

where ∇y ∗ t (x)∇<sup>2</sup> y g<sup>t</sup> (x, y ∗ t (x)) + ∇<sup>2</sup> xyg<sup>t</sup> (x, y ∗ t (x)) = 0.

As the exact y ∗ t (x) is not available, we estimate the hypergradient of f<sup>t</sup> at (x, y) by

$$\tilde{\nabla} f_t(\mathbf{x}, \mathbf{y}) := \nabla_{\mathbf{x}} f_t(\mathbf{x}, \mathbf{y}) + \nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}, \mathbf{y}) \mathbf{v}_t^*(\mathbf{x}), \quad (4a)$$

where

$$\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}, \mathbf{y}) \mathbf{v}_t^*(\mathbf{x}) + \nabla_{\mathbf{y}} f_t(\mathbf{x}, \mathbf{y}) = 0. \quad (4b)$$

An accurate solution of [\(4b\)](#page-2-4) is crucial for tight regret bounds. [Tarzanagh et al.](#page-10-2) [\(2024\)](#page-10-2) assumes an exact solution, which is restrictive in large-scale settings. To address this, [Lin](#page-9-1) [et al.](#page-9-1) [\(2024\)](#page-9-1) proposed an efficient OBO algorithm with window averaging, using CG methods to solve [\(4b\)](#page-2-4), which

Algorithm 1 SOGD

![](_page_2_Figure_2.jpeg)

Require: (x1, y1, v1) ∈ X × <sup>R</sup> <sup>d</sup><sup>2</sup> × <sup>R</sup> d<sup>2</sup> ; T ∈ N; p ∈ <sup>R</sup>++; stepsizes {(αt, βt, δt) ∈ <sup>R</sup> 3 ++} T <sup>t</sup>=1; parameters {(γt, λt, ηt)} T <sup>t</sup>=1 ∈ (0, 1); z<sup>t</sup> := (xt, yt).

For t = 1 to T do:

S1. Draw samples B<sup>t</sup> and B¯ <sup>t</sup> with batch sizes b and ¯b. Get search directions d y t , d v t , and d x t :

$$\mathbf{d}_t^{\text{yy}}(\mathbf{z}_t; \bar{\mathcal{B}}_t) = \nabla_{\mathbf{y}} g_t(\mathbf{z}_t; \bar{\mathcal{B}}_t), \quad (7a)$$

d y <sup>t</sup> = d yy t

(zt; B¯t) + (1 − γt)(d

y <sup>t</sup>−<sup>1</sup> − d yy t

(zt−1; B¯t)),

$$\begin{aligned} \mathbf{d}_t^{\text{v}}(\mathbf{z}_t; \mathcal{B}_t) &= \nabla_{\mathbf{y}} f_t(\mathbf{z}_t; \mathcal{B}_t) + \nabla_{\mathbf{y}}^2 g_t(\mathbf{z}_t; \bar{\mathcal{B}}_t) \mathbf{v}_t, \quad (7b) \\ \mathbf{d}_t^{\text{v}} &= \mathbf{d}_t^{\text{v}\text{v}}(\mathbf{z}_t; \mathcal{B}_t) + (1 - \lambda_t)(\mathbf{d}_{t-1}^{\text{v}} - \mathbf{d}_t^{\text{v}\text{v}}(\mathbf{z}_{t-1}; \mathcal{B}_t)), \end{aligned}$$

$$\begin{aligned} \mathbf{d}_{\mathbf{x}}^{\mathbf{x}}(\mathbf{z}_t; \mathcal{B}_t) &= \nabla_{\mathbf{x}} f_t(\mathbf{z}_t; \mathcal{B}_t) + \nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{z}_t; \bar{\mathcal{B}}_t) \mathbf{v}_t, \quad (7c) \\ \mathbf{d}_{\mathbf{x}}^{\mathbf{x}} &= \mathbf{d}_{\mathbf{x}}^{\mathbf{x}\mathbf{x}}(\mathbf{z}_t; \mathcal{B}_t) + (1 - \eta_t)(\mathbf{d}_{\mathbf{x}-1}^{\mathbf{x}} - \mathbf{d}_{\mathbf{x}}^{\mathbf{x}\mathbf{x}}(\mathbf{z}_{t-1}; \mathcal{B}_t)). \end{aligned}$$

S2. Update inner, system, and outer solutions:

$$\begin{aligned} \mathbf{y}_{t+1} &= \mathbf{y}_t - \beta_t \mathbf{d}_t^\mathbf{y}, \quad \mathbf{v}_{t+1} = \Pi_{\mathcal{Z}_p}[\mathbf{v}_t - \delta_t \mathbf{d}_t^\mathbf{y}], \\ \mathbf{x}_{t+1} &= \Pi_{\mathcal{X}}[\mathbf{x}_t - \alpha_t \mathbf{d}_t^\mathbf{x}]. \end{aligned}$$

is equivalent to:

$$\min_{\mathbf{v}_t \in \mathbb{R}^{d_2}} (1/2) \|\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}, \mathbf{y}) \mathbf{v}_t + \nabla_{\mathbf{y}} f_t(\mathbf{x}, \mathbf{y})\|^2. \quad (5)$$

New Search Direction for OBO. Next, we introduce a novel search direction that enables both first- and ZO stochastic OBO algorithms to achieve sublinear bilevel regret without smoothing. We first state the following lemma:

Lemma 3.1. *Let* w = t *and* W = 1/η *in the window-smoothed gradient* ∇ˆ Ft,ν(xt, yt; Bt) = (1/W) P<sup>w</sup>−<sup>1</sup> <sup>i</sup>=0 ν <sup>i</sup>∇ˆ ft−i(xt−<sup>i</sup> , yt−<sup>i</sup> ; Bt−i), *where* B<sup>t</sup> := {ξ1, . . . , ξb} *is drawn i.i.d. from* D<sup>f</sup> *. Then,*

$$\hat{\nabla} F_{t,\nu}(\mathbf{x}_t, \mathbf{y}_t; \mathcal{B}_t) = \sum_{j=1}^t \eta(1-\eta)^{t-j} \hat{\nabla} f_j(\mathbf{x}_j, \mathbf{y}_j; \mathcal{B}_j).$$

*Furthermore, we have* ∇ˆ Ft,ν(xt, yt; Bt) = dˆ<sup>x</sup> <sup>t</sup> *with* dˆx <sup>t</sup> <sup>=</sup> <sup>η</sup>∇<sup>ˆ</sup> <sup>f</sup>t(xt, <sup>y</sup>t; Bt) + (1 − <sup>η</sup>)dˆ<sup>x</sup> t−1 , *and* dˆ <sup>1</sup> = (1/W)∇ˆ f1(x1, y1; B1) *for all* t ≥ 2*.*

As shown in Lemma [3.1,](#page-2-5) for a specific choice of w and W, the time-smoothed gradient forms a recursive momentumtype search direction. However, achieving sublinear regret in stochastic OBO requires large-window smoothing (w = o(T)). To address this, we propose the following search direction:

$$\begin{aligned} \mathbf{d}_t^{\times} &= \eta \nabla f_t(\mathbf{x}_t, \mathbf{y}_t; \mathcal{B}_t) + (1 - \eta) \mathbf{d}_{t-1}^{\times} & (6a) \\ + (1 - \eta) (\nabla f_t(\mathbf{x}_t, \mathbf{y}_t; \mathcal{B}_t) - \nabla f_t(\mathbf{x}_{t-1}, \mathbf{y}_{t-1}; \mathcal{B}_t)). & (6b) \end{aligned}$$

This direction is used for updating x, with similar updates for y and v, as discussed below and detailed in Algorithm [1.](#page-2-1)

168

171

174

176

178

194

196 197 198

200

204

206

208

211

214 215 216

218

The quadratic optimization formulation of [\(4b\)](#page-2-4) in [\(5\)](#page-2-6) leads to single-loop frameworks such as [Dagreou et al.](#page-8-11) ´ [\(2022\)](#page-8-11). Inspired by this, we present Simultaneous Online Gradient Descent (SOGD) for constrained OBO, outlined in Algorithm [1.](#page-2-1) SOGD evolves the follower's decision (inner) variable, the linear system solution, and the leader's decision (outer) variable simultaneously at each step for given batches B := {ξ1, . . . , ξb} and B¯ := {ζ1, . . . , ζ¯b}, which are drawn i.i.d. from unknown distributions D<sup>f</sup> and D<sup>g</sup> with batch sizes b and ¯b. Computing directions in [S1.](#page-2-7) of Algorithm [1](#page-2-1) does not require ∇<sup>2</sup> y g<sup>t</sup> (xt, yt) and ∇<sup>2</sup> xyg<sup>t</sup> (xt, yt), only their product with a vector, at the same cost as computing a gradient. Technically, it utilizes an auxiliary variable v<sup>t</sup> and concurrently updates yt, vt, and x<sup>t</sup> at each local iteration t. Moreover, [S2.](#page-5-3) of Algorithm [1](#page-2-1) introduces an auxiliary projection ΠZ<sup>p</sup> on the ball Z<sup>p</sup> defined as follows:

$$\Pi_{\mathcal{Z}_p}(\mathbf{v}) := \min \left\{ 1, \frac{p}{\|\mathbf{v}\|} \right\} \mathbf{v}, \quad (8)$$

where Z<sup>p</sup> := {v ∈ <sup>R</sup> d<sup>2</sup> | ∥v∥ ≤ p}.

Unlike OAGD [\(Tarzanagh et al.,](#page-10-2) [2024\)](#page-10-2), which updates x and y in separate loops, SOGD updates both simultaneously. Compared to SOBOW [\(Lin et al.,](#page-9-1) [2024\)](#page-9-1), which uses multiple CG updates, our method employs a single OGD to update the inner solution, linear system, and outer variable. Assumption 3.2. gt(x, y) is twice continuously differentiable and µg-strongly convex in y for all x ∈ X , t ∈ [T].

Assumption 3.3. Let z = [x; y] and z ′ = [x ′ ; y ′ ], where x, x ′ ∈ X and y, y ′ ∈ <sup>R</sup> d<sup>2</sup> . For any z, z ′ , and t ∈ [T]:

B1. ∃ ℓf,<sup>0</sup> ∈ <sup>R</sup><sup>+</sup> s.t. ∥ft(z; ξ) − ft(z ′ ; ξ)∥ ≤ ℓf,0∥z − z ′ ∥; B2. ∃ ℓf,<sup>1</sup> ∈ <sup>R</sup><sup>+</sup> s.t. ∥∇ft(z; ξ) − ∇ft(z ′ ; ξ)∥ ≤ ℓf,1∥z − z ′ ∥; B3. ∃ ℓg,<sup>1</sup> ∈ <sup>R</sup>+s.t.∥∇gt(z; ζ) − ∇gt(z ′ ; ζ)∥ ≤ ℓg,1∥z − z ′ ∥; B4. ∃ ℓg,<sup>2</sup> ∈ <sup>R</sup><sup>+</sup> s.t.∥∇<sup>2</sup> gt(z; ζ)−∇<sup>2</sup> gt(z ′ ; ζ)∥ ≤ ℓg,2∥z−z ′ ∥.

Assumption 3.4. For any t ∈ [T], |ft(x, y ∗ t (x))| ≤ M for some finite constant M ∈ <sup>R</sup>++ and any x ∈ X .

Assumption 3.5. There exist constants σg<sup>y</sup> , σgyy , σgxy , σf<sup>y</sup> , σf<sup>y</sup> such that, for all z = [x, y]:

C1. <sup>E</sup>∥∇ygt(z; ζ) − ∇ygt(z)∥ <sup>2</sup> ≤ σ 2 g<sup>y</sup> , C2. <sup>E</sup>∥∇<sup>2</sup> y gt(z; ζ) − ∇<sup>2</sup> y gt(z)∥ <sup>2</sup> ≤ σ 2 gyy , C3. <sup>E</sup>∥∇<sup>2</sup> xygt(z; ζ) − ∇<sup>2</sup> xygt(z)∥ <sup>2</sup> ≤ σ 2 gxy , C4. <sup>E</sup>∥∇yft(z; ξ) − ∇yft(z)∥ <sup>2</sup> ≤ σ 2 f<sup>y</sup> , C5. <sup>E</sup>∥∇xft(z; ξ) − ∇xft(z)∥ <sup>2</sup> ≤ σ 2 f<sup>x</sup> .

Throughout this paper, we define

$$\sigma^2 := \sigma_{gy}^2 + \sigma_{gyy}^2 + \sigma_{fy}^2 + \sigma_{gxy}^2 + \sigma_{fx}^2. \quad (9)$$

Assumptions [3.2](#page-3-2) and [3.3](#page-3-3) are widely used in both BO [\(Chen](#page-8-10) [et al.,](#page-8-10) [2021;](#page-8-10) [Ji et al.,](#page-9-4) [2021\)](#page-9-4) and OBO [\(Tarzanagh et al.,](#page-10-2) [2024\)](#page-10-2), and many bilevel machine learning problems satisfy it [\(Franceschi et al.,](#page-8-3) [2018\)](#page-8-3). Further, Assumption [3.4](#page-3-4) is widely used in the study of non-convex online optimization [\(Hazan et al.,](#page-9-6) [2017;](#page-9-6) [Lin et al.,](#page-9-1) [2024\)](#page-9-1). Assumption [3.5](#page-3-5)

assumes that we have access to an unbiased stochastic gradient, Hessian and Jacobian with bounded variance, which is standard in the literature [\(Chen et al.,](#page-8-10) [2021\)](#page-8-10).

Achieving sublinear dynamic regret is generally impossible due to arbitrary fluctuations in time-varying functions [\(Bes](#page-8-12)[bes et al.,](#page-8-12) [2015\)](#page-8-12). Existing analyses [\(Tarzanagh et al.,](#page-10-2) [2024;](#page-10-2) [Lin et al.,](#page-9-1) [2024\)](#page-9-1) bound regret by imposing regularity constraints on the comparator sequence. To achieve sublinear regret, we introduce the following regularities:

• Path-length (of order p) and function variation: [Tarzanagh et al.](#page-10-2) [\(2024\)](#page-10-2) defines the following metrics for bilevel sequences:

$$H_{p,T} := \sum_{t=2}^T \sup_{\mathbf{x} \in \mathcal{X}} \|\mathbf{y}_{t-1}^*(\mathbf{x}) - \mathbf{y}_t^*(\mathbf{x})\|^p, \quad (10)$$

$$V_T := \sum_{t=2}^T \sup_{\mathbf{x} \in \mathcal{X}} |f_{t-1}(\mathbf{x}, \mathbf{y}_{t-1}^*(\mathbf{x})) - f_t(\mathbf{x}, \mathbf{y}_t^*(\mathbf{x}))|.$$

Path-length Hp,T measures changes in the follower's costs, while V<sup>T</sup> captures the smoothness of the leader's objective. We use path-length for the follower and function variation for the leader, as the follower's objective is strongly convex (see Assumption [3.2\)](#page-3-2), while the leader's is nonconvex.

• Inner and Outer Gradient Variations: Another regularity is the sequential difference between the individual gradients of the upper-level loss function:

$$D_{\mathbf{x},T} := \sum_{t=2}^T \sup_{\mathbf{x},\mathbf{y}} \|\nabla_{\mathbf{x}} f_{t-1}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{x}} f_t(\mathbf{x}, \mathbf{y})\|^2, \quad (11)$$

$$D_{\mathbf{y},T} := \sum_{t=2}^T \sup_{\mathbf{x},\mathbf{y}} \|\nabla_{\mathbf{y}} f_{t-1}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{y}} f_t(\mathbf{x}, \mathbf{y})\|^2.$$

As in [Huang et al.;](#page-9-7) [Hallak et al.](#page-9-8) [\(2021\)](#page-9-8), Dx,T and Dy,T measure the gradient drift of f<sup>t</sup> relative to ft−<sup>1</sup> for x and y, respectively. We further define deviations in the gradient, Hessian, and Jacobian of the lower-level objective as:

$$G_{\mathbf{y}, T} := \sum_{t=2}^T \|\nabla_{\mathbf{y}} g_{t-1}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2,$$

$$G_{\mathbf{y}, \mathbf{y}, T} := \sum_{t=2}^T \|\nabla_{\mathbf{y}}^2 g_{t-1}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2, \quad (12)$$

$$G_{\mathbf{x}, \mathbf{y}, T} := \sum_{t=2}^T \|\nabla_{\mathbf{x}, \mathbf{y}}^2 g_{t-1}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}, \mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2.$$

We introduce the following notations for simplicity:

$$\Delta_T := E_1 + V_T, \quad \Psi_T := H_{2,T} + G_T + D_T, \quad (13)$$

where (V<sup>T</sup> , Hp,T ) are defined in [\(10\)](#page-3-0), and

$$\begin{aligned} E_1 &:= \|\mathbf{y}_1 - \mathbf{y}_1^*(\mathbf{x}_1)\|^2 + \|\mathbf{v}_1 - \mathbf{v}_1^*(\mathbf{x}_1)\|^2, \\ G_T &:= G_{\mathbf{y},T} + G_{\mathbf{y}\mathbf{y},T} + G_{\mathbf{x}\mathbf{y},T}, \\ D_T &:= D_{\mathbf{x},T} + D_{\mathbf{y},T}. \end{aligned} \tag{14}$$

226

228

231

234

236

238

254

256

258

260

264

266

268

271

274

By accounting for both D<sup>T</sup> and G<sup>T</sup> , we can represent the variations in the environments of OBO.

Theorem 3.6. *Let* {(ft, gt)} T <sup>t</sup>=1 *be the sequence of functions presented to Algorithm [1,](#page-2-1) satisfying Assumptions [3.2-](#page-3-2) [3.5.](#page-3-5) For all* t ∈ [T]*, let*

$$\alpha_t = \frac{1}{(c+t)^{1/3}}, \quad \beta_t = c_\beta \alpha_t, \quad \delta_t = c_\delta \alpha_t, \quad b = \bar{b} = 1,$$

$$\gamma_{t+1} = c_\gamma \alpha_t^2, \quad \eta_{t+1} = c_\eta \alpha_t^2, \quad \lambda_{t+1} = c_\lambda \alpha_t^2. \quad (15)$$

*Here,* c*,* cβ*,* cδ*,* cγ*,* cη*, and* c<sup>λ</sup> *are specified in* [\(104\)](#page-34-0)*. Algorithm [1](#page-2-1) guarantees:*

$$\text{BL-Reg}_T \leq \mathcal{O} \left( T^{1/3}(\sigma^2 + \Delta_T) + T^{2/3}\Psi_T \right), \quad (16)$$

*where* σ *and* (∆<sup>T</sup> , Ψ<sup>T</sup> ) *are defined in* [\(9\)](#page-3-6) *and* [\(13\)](#page-3-1)*.*

Theorem [3.6](#page-4-0) bounds the regret of Algorithm [1](#page-2-1) without window-smoothing, based on the regularities in [\(14\)](#page-3-7). We note that the average dynamic regret BL-Reg<sup>T</sup> /T ≤ O(T −2/3 (σ <sup>2</sup> + ∆<sup>T</sup> ) + T <sup>−</sup>1/<sup>3</sup>Ψ<sup>T</sup> ) remains sublinear under suitable conditions on ∆<sup>T</sup> and Ψ<sup>T</sup> .

*Remark* 3.7 (Stochastic Regret Guarantee for OBO and OSO with w = 1)*.* The additional terms in [\(6b\)](#page-2-8) improve the average regret dependence on variance, achieving a T <sup>−</sup>2/<sup>3</sup>σ <sup>2</sup> bound, better than the T <sup>−</sup>1/<sup>2</sup>σ <sup>2</sup> bound for stochastic OBO [\(Bohne et al.,](#page-8-5) [2024\)](#page-8-5). This also provides the first regret bound without window-smoothing, unlike [\(Bohne](#page-8-5) [et al.,](#page-8-5) [2024;](#page-8-5) [Tarzanagh et al.,](#page-10-2) [2024;](#page-10-2) [Lin et al.,](#page-9-1) [2024;](#page-9-1) [Huang](#page-9-2) [et al.,](#page-9-2) [2023\)](#page-9-2). For OSO, our approach improves the T <sup>−</sup>1/<sup>2</sup>σ 2 dependence from [\(Hallak et al.,](#page-9-8) [2021\)](#page-9-8).

# 4. OBO with Zeroth Order Oracles

Black-box optimization arises in machine learning when explicit gradients are unavailable [\(Chen et al.,](#page-8-7) [2017\)](#page-8-7). We study ZO-type OBO algorithms with limited access to the leader's and follower's objective values. Let s ∈ R <sup>d</sup><sup>1</sup> and r ∈ R <sup>d</sup><sup>2</sup> be vectors uniformly generated from the unit balls B<sup>1</sup> and B2, respectively. Given positive smoothing parameters ρ = (ρs, ρr), we use the Gaussian smoothing function [\(Nesterov & Spokoiny,](#page-9-9) [2017\)](#page-9-9) to define the OBO objectives:

$$f_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x})) = \mathbb{E}_{(\mathbf{s}, \mathbf{r})} [f_t(\mathbf{x} + \rho_s \mathbf{s}, \hat{\mathbf{y}}_t^*(\mathbf{x}) + \rho_{\mathbf{r}} \mathbf{r}; \xi)], \quad (17)$$

where

$$\begin{aligned} \hat{\mathbf{y}}_t^*(\mathbf{x}) &\in \operatorname{argmin}_{\mathbf{y} \in \mathbb{R}^{d_2}} \{g_{t,\rho}(\mathbf{x}, \mathbf{y})\} \\ &:= \mathbb{E}_{(\mathbf{s}, \mathbf{r})} [g_t(\mathbf{x} + \rho_s \mathbf{s}, \mathbf{y} + \rho_r \mathbf{r}; \zeta)] \end{aligned} \quad (18)$$

Using [\(17\)](#page-4-1), we provide methodology to approximate each term in [\(7\)](#page-2-9) using ZO oracles. Specifically, following [Shamir](#page-10-5) [\(2017\)](#page-10-5), we estimate the gradient of a function h : R <sup>d</sup> → <sup>R</sup>, querying at x − λs and x + λs, yielding an estimator (d/2λ) (h(x + λs) − h(x − λs)) s. Using this strategy, the finite-difference estimation of ∇gt,ρ(x, y), denoted as ∇ˆ gt(x, y), is constructed for given smoothing

parameters ρ = (ρs, ρr), and batches B := {ξ1, . . . , ξb} and B¯ := {ζ1, . . . , ζ¯b}, drawn i.i.d. from D<sup>f</sup> and Dg, as:

$$\hat{\nabla}_{\mathbf{y}} g_t(\mathbf{x}, \mathbf{y}; \bar{\mathcal{B}}) := \frac{d_2}{2\bar{b}\rho_{\mathbf{r}}} \sum_{i=1}^{\bar{b}} (g_t(\mathbf{x}, \mathbf{y} + \rho_{\mathbf{r}} \mathbf{r}_i; \zeta_i) - g_t(\mathbf{x}, \mathbf{y} - \rho_{\mathbf{r}} \mathbf{r}_i; \zeta_i)) \mathbf{r}_i, \quad (19a)$$

$$\begin{aligned} \hat{\nabla}_{\mathbf{x}} g_t(\mathbf{x}, \mathbf{y}; \bar{\mathcal{B}}) &:= \frac{d_1}{2\bar{b}\rho_{\mathbf{s}}} \sum_{i=1}^{\bar{b}} (g_t(\mathbf{x} + \rho_{\mathbf{s}} \mathbf{s}_i, \mathbf{y}; \zeta_i) \\ &\quad - g_t(\mathbf{x} - \rho_{\mathbf{s}} \mathbf{s}_i, \mathbf{y}; \zeta_i)) \mathbf{s}_i. \end{aligned} \quad (19b)$$

Similarly, we estimate ∇yft,ρ(x, y; B) and ∇xft,ρ(x, y; B), respectively, by

$$\begin{aligned} \hat{\nabla}_{\mathbf{y}} f_t(\mathbf{x}, \mathbf{y}; \mathcal{B}) &:= \frac{d_2}{2b\rho_{\mathbf{r}}} \sum_{i=1}^b (f_t(\mathbf{x}, \mathbf{y} + \rho_{\mathbf{r}}\mathbf{r}_i; \xi_i) \\ &\quad - f_t(\mathbf{x}, \mathbf{y} - \rho_{\mathbf{r}}\mathbf{r}_i; \xi_i)) \mathbf{r}_i, \end{aligned} \quad (20a)$$

$$\nabla_{\mathbf{x}} f_t(\mathbf{x}, \mathbf{y}; \mathcal{B}) := \frac{d_1}{2b\rho_s} \sum_{i=1}^b (f_t(\mathbf{x} + \rho_s \mathbf{s}_i, \mathbf{y}; \xi_i) - f_t(\mathbf{x} - \rho_s \mathbf{s}_i, \mathbf{y}; \xi_i)) \mathbf{s}_i. \quad (20b)$$

Further, given a smoothing parameter ρ<sup>v</sup> > 0, we can approximate the Hessian-vector product ∇<sup>2</sup> y gt,ρ(x, y)v and the Jacobian-vector product ∇<sup>2</sup> xygt,ρ(x, y)v as the finite difference between two gradients, respectively, as

$$\hat{\nabla}_y^2 g_t(\mathbf{x}, \mathbf{y}; \bar{\mathbf{B}}) := \frac{1}{2\bar{b}\rho_{\mathbf{v}}} \sum_{i=1}^{\bar{b}} (\hat{\nabla}_y g_t(\mathbf{x}, \mathbf{y} + \rho_{\mathbf{v}}\mathbf{v}; \zeta_i) - \hat{\nabla}_y g_t(\mathbf{x}, \mathbf{y} - \rho_{\mathbf{v}}\mathbf{v}; \zeta_i)), \quad (21a)$$

$$\hat{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}, \mathbf{y}; \bar{B}) := \frac{1}{2\bar{b}\rho_{\mathbf{v}}} \sum_{i=1}^{\bar{b}} (\hat{\nabla}_{\mathbf{x}} g_t(\mathbf{x}, \mathbf{y} + \rho_{\mathbf{v}} \mathbf{v}; \zeta_i) - \hat{\nabla}_{\mathbf{x}} g_t(\mathbf{x}, \mathbf{y} - \rho_{\mathbf{v}} \mathbf{v}; \zeta_i)). \quad (21b)$$

Using [\(19\)](#page-4-2)–[\(21\)](#page-4-3), the first-order terms in [\(7\)](#page-2-9) are approximated as dˆ<sup>y</sup> t , dˆ<sup>v</sup> t , and dˆ<sup>x</sup> t in [\(22\)](#page-5-4). The approximations in [\(21a\)](#page-4-4) and [\(21b\)](#page-4-5) introduce errors in the hypergradient, which must be controlled. [\(21\)](#page-4-3) depends on the dimension of y, as in ZO optimization [\(Nesterov & Spokoiny,](#page-9-9) [2017;](#page-9-9) [Shamir,](#page-10-5) [2017\)](#page-10-5). The projection ΠZ<sup>p</sup> in [\(8\)](#page-3-8) bounds v, controlling variance in v and x updates for convergence.

Assumption 4.1. There exist constants σˆg<sup>y</sup> , σˆg<sup>x</sup> , σˆf<sup>y</sup> , σˆf<sup>x</sup> such that, for all z = [x, y], the following holds:

$$\begin{aligned} \text{D1.} & \mathbb{E}\|\hat{\nabla}_y g_t(\mathbf{z}; \zeta) - \nabla_y g_{t,\rho}(\mathbf{z})\|^2 \leq \hat{\sigma}_{g_y}^2, \\ \text{D2.} & \mathbb{E}\|\hat{\nabla}_x g_t(\mathbf{z}; \zeta) - \nabla_x g_{t,\rho}(\mathbf{z})\|^2 \leq \hat{\sigma}_{g_x}^2, \\ \text{D3.} & \mathbb{E}\|\hat{\nabla}_y f_t(\mathbf{z}; \xi) - \nabla_y f_{t,\rho}(\mathbf{z})\|^2 \leq \hat{\sigma}_{f_y}^2, \\ \text{D4.} & \mathbb{E}\|\hat{\nabla}_x f_t(\mathbf{z}; \xi) - \nabla_x f_{t,\rho}(\mathbf{z})\|^2 \leq \hat{\sigma}_{f_x}^2. \end{aligned}$$

Assumption [4.1](#page-4-6) is analogous to the upper bound on the variance of stochastic partial gradients discussed in [Luo](#page-9-10) [et al.](#page-9-10) [\(2020\)](#page-9-10); [Wang et al.](#page-10-6) [\(2020\)](#page-10-6). We simplify the notation by introducing the following shorthand.

$$\hat{\sigma}^2 := \hat{\sigma}_{g_y}^2 + \hat{\sigma}_{g_x}^2 + \hat{\sigma}_{f_y}^2 + \hat{\sigma}_{f_x}^2. \quad (23)$$

Next, we establish a regret bound for ZO-SOGD. Similar to

278

289 290

294

296

298

300

304

306

308 309

311

314 315 316

318

324

326

328

Algorithm 2 ZO-SOGD

Require: In addition to parameters in SOGD, choose ρv, ρr, ρs, ∈ <sup>R</sup>++.

For t = 1 to T do:

S1. Draw samples B<sup>t</sup> and B¯ <sup>t</sup> with batch sizes b and ¯b. Using [\(19\)](#page-4-2)–[\(21\)](#page-4-3), get ZO search directions dˆ<sup>y</sup> t , dˆ<sup>v</sup> t , dˆ<sup>x</sup> t :

$$\mathbf{d}_t^y(\mathbf{z}_t; \bar{\mathcal{B}}_t) = \hat{\nabla}_y g_t(\mathbf{z}_t; \bar{\mathcal{B}}_t), \quad (22a)$$

$$\hat{\mathbf{d}}_t^{\mathbf{y}} = \mathbf{d}_t^{\mathbf{y}}(\mathbf{z}_t; \bar{\mathcal{B}}_t) + (1 - \gamma_t)(\hat{\mathbf{d}}_{t-1}^{\mathbf{y}} - \mathbf{d}_t^{\mathbf{y}}(\mathbf{z}_{t-1}; \bar{\mathcal{B}}_t)),$$

$$\mathbf{d}_t^{\mathbf{vv}}(\mathbf{z}_t; \mathcal{B}_t) = \hat{\nabla}_{\mathbf{y}} f_t(\mathbf{z}_t; \mathcal{B}_t) + \hat{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t; \bar{\mathcal{B}}_t), \quad (22b)$$

$$\hat{\mathbf{d}}_t^{\mathbf{v}} = \mathbf{d}_t^{\mathbf{v}\mathbf{v}}(\mathbf{z}_t; \mathcal{B}_t) + (1 - \lambda_t)(\hat{\mathbf{d}}_{t-1}^{\mathbf{v}} - \mathbf{d}_t^{\mathbf{v}\mathbf{v}}(\mathbf{z}_{t-1}; \mathcal{B}_t)),$$

$$\mathbf{d}_t^{\mathbf{xy}}(\mathbf{z}_t; \mathcal{B}_t) = \hat{\nabla}_{\mathbf{x}} f_t(\mathbf{z}_t; \mathcal{B}_t) + \hat{\nabla}_{\mathbf{xy}}^2 g_t(\mathbf{z}_t; \bar{\mathcal{B}}_t), \quad (22c)$$

$$\hat{\mathbf{d}}_t^x = \mathbf{d}_t^{xy}(\mathbf{z}_t; \mathcal{B}_t) + (1 - \eta_t)(\hat{\mathbf{d}}_{t-1}^x - \mathbf{d}_t^{xy}(\mathbf{z}_{t-1}; \mathcal{B}_t)),$$

S2. Update inner, system, and outer solutions:

$$\begin{aligned} \mathbf{y}_{t+1} &= \mathbf{y}_t - \boldsymbol{\beta}_t \hat{\mathbf{d}}_t^\mathbf{y}, \quad \mathbf{v}_{t+1} = \Pi_{\mathcal{Z}_p} [\mathbf{v}_t - \boldsymbol{\delta}_t \hat{\mathbf{d}}_t^\mathbf{y}], \\ \mathbf{x}_{t+1} &= \Pi_{\mathcal{X}} [\mathbf{x}_t - \alpha_t \hat{\mathbf{d}}_t^\mathbf{x}]. \end{aligned}$$

previous results, we introduce regularity conditions for the smoothed functions in [\(17\)](#page-4-1) and [\(18\)](#page-4-7).

Inner and Outer Perturbed Gradient Variations: We define the gradient variations at the perturbed point as follows:

$$G_{\mathbf{v},T} := \sum_{t=2}^T (\chi_{1t} + \chi_{2t}), \quad G_{\mathbf{x},T} := \sum_{t=2}^T (\chi_{3t} + \chi_{4t}). \quad (24)$$

where z + t := (xt−1, yt−1+ρvvt−1), z − t := (xt−1, yt−1− ρvvt−1), and

$$\begin{aligned}\chi_{1t} &:= \|\nabla_{\mathbf{y}} g_t(\mathbf{z}_t^+) - \nabla_{\mathbf{y}} g_{t-1}(\mathbf{z}_t^+)\|^2, \\ \chi_{2t} &:= \|\nabla_{\mathbf{y}} g_t(\mathbf{z}_t^-) - \nabla_{\mathbf{y}} g_{t-1}(\mathbf{z}_t^-)\|^2, \\ \chi_{3t} &:= \|\nabla_{\mathbf{x}} g_t(\mathbf{z}_t^+) - \nabla_{\mathbf{x}} g_{t-1}(\mathbf{z}_t^+)\|^2, \\ \chi_{4t} &:= \|\nabla_{\mathbf{x}} g_t(\mathbf{z}_t^-) - \nabla_{\mathbf{x}} g_{t-1}(\mathbf{z}_t^-)\|^2.\end{aligned}$$

Further, for simplicity of notation, we define

$$\begin{aligned}\hat{\Delta}_T &:= E_1 + V_T + D_T + G_{\mathbf{y},T}, \\ \hat{\Psi}_T &:= H_{2,T} + G_{\mathbf{v},T} + G_{\mathbf{x},T},\end{aligned}\tag{25}$$

where (V<sup>T</sup> , Hp,T ) and (E1, D<sup>T</sup> ) are defined in [\(10\)](#page-3-0), and [\(14\)](#page-3-7), repectively. Moreover, Gy,T and (Gv,T , Gx,T ), are defined in [\(12\)](#page-3-9) and [\(24\)](#page-5-5), respectively.

Theorem 4.2. *Let* {(ft, gt)} T <sup>t</sup>=1 *be the sequence of functions presented to Algorithm [2,](#page-5-2) satisfying Assumptions [3.2-](#page-3-2) [3.4](#page-3-4) and [4.1.](#page-4-6) For all* t ∈ [T]*, let*

$$\begin{aligned} \alpha_t &= \frac{1}{(d_1 + d_2)^{3/4}(c + t)^{1/3}}, & \beta_t &= c_\beta \alpha_t, & \delta_t &= c_\delta \alpha_t, \\ \gamma_{t+1} &= c_\gamma \alpha_t, & \eta_{t+1} &= c_\eta \alpha_t, & \lambda_{t+1} &= c_\lambda \alpha_t, \\ \rho_v^2 &= c_v \alpha_t, & \rho_r^2 &= \frac{1}{d_2^2 T}, & \rho_s^2 &= \frac{1}{d_1^2 T}, \\ b &= \frac{T^{1/3}}{(d_1 + d_2)^{3/2}}, & \bar{b} &= \frac{T^{2/3}}{(d_1 + d_2)^{3/4}}, \end{aligned} \quad (26)$$

*where* c*,* cβ*,* cδ*,* cγ*,* cη*,* cv*, and* c<sup>λ</sup> *are specified in* [\(232\)](#page-72-0)*. Let* p = ℓf,0/µ<sup>g</sup> *for the set* Z<sup>p</sup> *defined in* [\(8\)](#page-3-8)*. Then, Algorithm [2](#page-5-2) guarantees:*

$$\text{BL-Reg}_T \leq \mathcal{O} \left( (d_1 + d_2)^{3/4} T^{1/3} \left( \hat{\sigma}^2 + \hat{\Delta}_T \right) + (d_1 + d_2)^{3/2} T^{2/3} \hat{\Psi}_T \right).$$

*where* σˆ <sup>2</sup> *and* (∆ˆ <sup>T</sup> , Ψˆ <sup>T</sup> ) *are defined in* [\(23\)](#page-4-8) *and* [\(25\)](#page-5-1)*.*

Theorem [4.2](#page-5-0) bounds the regret of Algorithm [2](#page-5-2) without window-smoothing, based on the regularities in [\(25\)](#page-5-1). We note that the average dynamic regret BL-Reg<sup>T</sup> /T ≤ O((d1+d2) <sup>3</sup>/<sup>4</sup>T −2/3 σˆ <sup>2</sup> + ∆ˆ T +(d1+d2) <sup>3</sup>/<sup>2</sup>T <sup>−</sup>1/3Ψˆ T ) remains sublinear under suitable conditions on ∆ˆ <sup>T</sup> and Ψˆ T . *Remark* 4.3 (Regret Guarantee for Zeroth Order OBO)*.* Theorem [4.2](#page-5-0) provides the first regret guarantee for OBO with access only to noisy function evaluations of the leader and follower. The dimensional dependence O(d<sup>1</sup> + d2) in Theorem [4.2](#page-5-0) aligns with optimal results for simpler offline min-max problems [\(Huang et al.,](#page-9-11) [2022\)](#page-9-11). The bound also depends on the sample sizes b, ¯b and smoothing parameters ρv, ρr, ρ<sup>s</sup> at each iteration.

*Remark* 4.4 (Improved Regret for OSO)*.* Our dynamic regret for single-level non-stationary optimization is O((d<sup>1</sup> + d2) <sup>3</sup>/<sup>4</sup>T −2/3 (ˆσ <sup>2</sup> + E<sup>1</sup> + V<sup>T</sup> + D<sup>T</sup> )), improving the result in [Roy et al.](#page-10-7) [\(2022\)](#page-10-7), which is O(T <sup>−</sup>1/<sup>2</sup>σ 2 √ d). [Roy](#page-10-7) [et al.](#page-10-7) [\(2022\)](#page-10-7) proposed a zeroth-order stochastic gradient descent algorithm for unconstrained, non-convex, timevarying objective functions, achieving a regret bound of O(T <sup>−</sup>1/<sup>2</sup>σ 2√ dW<sup>T</sup> ) using a two-point gradient estimator, where W<sup>T</sup> bounds the nonstationarity. Additionally, [Guan](#page-9-12) [et al.](#page-9-12) [\(2023a\)](#page-9-12) showed that the local regret for standard online stochastic gradient descent with the standard two-point gradient estimator [\(Agarwal et al.,](#page-8-13) [2010\)](#page-8-13) is O(T <sup>−</sup>1/<sup>2</sup>d √ V<sup>T</sup> ).

# 5. Experimental Results

In this section, we provide experimental results on bilevel optimization-based black-box attacks on deep neural networks and parametric loss tuning for imbalanced data.

# 5.1. Bilevel Optimization-Based Black-Box Attacks

Deep neural network classifiers are vulnerable to adversarial examples—images subtly modified to mislead the classifier. These examples can deceive classifiers even without knowledge of the model, as seen in black-box adversarial attacks (BBAA) [\(Chen et al.,](#page-8-7) [2017;](#page-8-7) [Liu et al.,](#page-9-13) [2018b;](#page-9-13) [Chen et al.,](#page-8-14)

334

336

338

351

354

356

358

360 361

364

366

368

371

374

378

![](_page_6_Figure_1.jpeg)

Figure 2. Performance comparison (mean±std) of optimizers including ZO-O-GD, ZO-O-Adam, ZO-O-SignSGD, ZO-O-ConservSGD, ZO-SOGD, and ZO-SOGD (Adam) on online adversarial attack for MNIST data across five runs.

![](_page_6_Figure_3.jpeg)

Figure 3. Performance comparison (mean±std) on imbalanced loss tuning with distribution shift for MNIST data across five runs between OGD [\(Zinkevich,](#page-10-3) [2003\)](#page-10-3), OAGD [\(Tarzanagh et al.,](#page-10-2) [2024\)](#page-10-2), SOBOW [\(Lin et al.,](#page-9-1) [2024\)](#page-9-1), and our SOGD.

[2019\)](#page-8-14).

We first review the ZO single-level optimization for BBAA [\(Chen et al.,](#page-8-7) [2017\)](#page-8-7). Let (a, b) denote a legitimate image a ∈ R <sup>d</sup> with true label b ∈ {1, 2, . . . , J}, where J is the total number of classes. Define a ′ = a+y as an adversarial example, with y as the adversarial perturbation. Let Y := [−5, 5]<sup>d</sup> ⊂ <sup>R</sup> d , and ℓ : R <sup>d</sup> → <sup>R</sup> denote the black-box attack loss. The goal of BBAA [\(Chen et al.,](#page-8-7) [2017\)](#page-8-7) is to design y for images {ai} m <sup>i</sup>=1 by solving:

$$\min_{\mathbf{y} \in \mathcal{Y}} \frac{1}{m} \sum_{i=1}^m \ell(\mathbf{a}_i + \mathbf{y}) + \lambda \|\mathbf{y}\|^2. \quad (27)$$

Here, λ > 0 is a hyperparameter balancing attack loss minimization and ℓ<sup>2</sup> regularization.

To adapt [\(27\)](#page-6-0) to our OBO, consider OBO for supervised learning: at each timestep t, new samples (at, bt) ∈ D<sup>t</sup> := {Dval t , Dtr <sup>t</sup> } are received, where a<sup>t</sup> ∈ <sup>R</sup> d<sup>2</sup> is the feature vector (image) and b<sup>t</sup> ∈ <sup>R</sup> is the corresponding target. Note that the correct decision can change abruptly. We consider an S-stage scenario where (x ∗ s , y ∗ s (x ∗ s )) represents the best decisions for the s-th stage, for all s ∈ [S].

$$\mathbf{x}_s^* \in \operatorname{argmin}_{\mathbf{x} \in \mathcal{X}} \sum_{t=1}^T f(\mathbf{y}_s^*(\mathbf{x}); \mathcal{D}_t^{\text{val}})$$
s.t.  $\mathbf{y}_s^*(\mathbf{x}) \in \operatorname{argmin}_{\mathbf{y} \in \mathcal{Y}} \sum_{t=1}^{T_s} g(\mathbf{x}, \mathbf{y}; \mathcal{D}_t^{\text{tr}})$ , (28)

where

$$g(\mathbf{x}_t, \mathbf{y}_t; \mathcal{D}_t^{\text{tr}}) = \frac{1}{|\mathcal{D}_t^{\text{tr}}|} \sum_{i \in \mathcal{D}_t^{\text{tr}}} \ell(\mathbf{a}_t^{(i)} + \mathbf{y}_t) \\ + \frac{1}{2} \sum_{\iota = 1}^p e^{[\mathbf{x}_t]_{\iota}} [\mathbf{y}_t]_{\iota}^2, \quad (29a)$$

and

$$f(\mathbf{y}_t(\mathbf{x}_t); \mathcal{D}_t^{\text{val}}) = \frac{1}{|\mathcal{D}_t^{\text{val}}|} \sum_{i \in \mathcal{D}_t^{\text{val}}} \ell(\mathbf{a}_t^{(i)} + \mathbf{y}_t). \quad (29b)$$

Here, {a (i) <sup>t</sup> }i∈Dtr and {a (i) <sup>t</sup> }i∈Dval are batches of training and validation samples at timestep t; a (i) t is the ith sample in that batch; and [xt]<sup>ι</sup> and [yt]<sup>ι</sup> denote the ιth component of x<sup>t</sup> and yt, respectively.

We normalize the pixel values to Y. For an untargeted attack, the loss in [\(29\)](#page-6-1) is ℓ(a ′ t ) = max{Z(a ′ t )<sup>b</sup><sup>t</sup> − maxj̸=b<sup>t</sup> Z(a ′ t )<sup>j</sup> , −κ}, where Z(a ′ t )<sup>j</sup> is the prediction score for class j given input a ′ <sup>t</sup> = a<sup>t</sup> + yt, and κ > 0 controls the confidence gap. In our experiments, we set κ = 0.

Eq. [\(28\)](#page-6-2) introduces the first OBO formulation of BBAA. Using a vector x ∈ R d <sup>+</sup> for hyperparameters instead of λ ∈ <sup>R</sup>++ in [\(27\)](#page-6-0) enables finer control over model components, enhancing performance for complex models and heterogeneous data [\(Lorraine et al.,](#page-9-14) [2020\)](#page-9-14). For a fair comparison with single-level BBAA, we replace λ with a fixed

394

396

vector multiplied by each component of y in [\(27\)](#page-6-0).

We compare our ZO-SOGD and ZO-SOGD (Adam) with the following competing methods in the online setting:

ZO-O-GD: A single-level method that updates y<sup>t</sup> with a fixed x at each timestep using ZO gradient descent [\(Nes](#page-9-9)[terov & Spokoiny,](#page-9-9) [2017\)](#page-9-9).

ZO-O-Adam: A single-level method that updates y<sup>t</sup> with a fixed x at each timestep using ZO Adam [\(Kingma & Ba,](#page-9-15) [2014;](#page-9-15) [Chen et al.,](#page-8-14) [2019\)](#page-8-14).

ZO-O-SignSGD: A single-level method that updates y<sup>t</sup> with a fixed x at each timestep using ZO SignSGD [\(Bern](#page-8-15)[stein et al.,](#page-8-15) [2018\)](#page-8-15).

ZO-O-ConservSGD: A single-level method that updates y<sup>t</sup> with a fixed x at each timestep using ZO Conservative SGD [\(Cutkosky & Boahen,](#page-8-16) [2019\)](#page-8-16).

Note that ZO-SOGD (Adam) is a variant of our algorithm with an adaptive stepsize, similar to that of [\(Kingma & Ba,](#page-9-15) [2014\)](#page-9-15).

We evaluated the proposed algorithms based on runtime, test accuracy on perturbed samples, and the infinity norm of yt. Figure [2](#page-6-3) compares the methods. The left panel shows that ZO-SOGD has similar runtime to single-level baselines, despite outer-level optimization on x. The middle panel shows that all methods' accuracy decreases as the adversarial attack y strengthens, with ZO-SOGD outperforming ZO-O-GD and ZO-O-ConservGD, and ZO-SOGD (Adam) outperforming ZO-O-Adam and all baselines. The right panel shows that the increasing infinity norm of y<sup>t</sup> over time for all methods, which reduces accuracy. However, the perturbations remain unnoticeable with a max y<sup>t</sup> no larger than 4, demonstrating that ZO-SOGD achieves effective attacks with better performance than other methods.

# 5.2. Parametric Loss Tuning for Imbalanced Data

Imbalanced datasets are common in modern machine learning, causing challenges in generalization and fairness due to underrepresented classes and sensitive attributes. Deep NNs often overfit, seeming accurate and fair during training but performing poorly during testing. A common solution is designing a parametric training loss that balances accuracy and fairness while preventing overfitting [\(Li et al.,](#page-9-16) [2021\)](#page-9-16).

We consider an optimization problem similar to [\(28\)](#page-6-2). For a new sample (at, bt), the follower and leader incur a parametric and balanced cross-entropy loss, respectively:

$$g(\mathbf{x}_t, \mathbf{y}_t; \mathcal{D}_t^{\text{tr}}) = -\log \frac{e^{\gamma_t(\mathbf{y}_t(\mathbf{x}_t))_{b_t}}}{\sum_{j=1}^J e^{\gamma_j(\mathbf{y}_t(\mathbf{x}_t))_{j+\Delta_j}}}, \quad \text{and}$$

$$f(\mathbf{y}_t(\mathbf{x}_t); \mathcal{D}_t^{\text{val}}) = -u_{b_t} \log \frac{e^{[\mathbf{y}_t(\mathbf{x}_t)]_{b_t}}}{\sum_{j=1}^J e^{[\mathbf{y}_t(\mathbf{x}_t)]_j}}. \quad (30)$$

Here, x<sup>t</sup> := (∆<sup>j</sup> , γ<sup>j</sup> ) J <sup>j</sup>=1 represents the logits adjustments, with j indexing the J classes, and u<sup>j</sup> is the reciprocal of the proportion of samples from the j-th class to the total

number of samples [\(Li et al.,](#page-9-16) [2021\)](#page-9-16).

To clarify the notation in [\(30\)](#page-7-0): yt(xt) denotes the follower y<sup>t</sup> conditioned on the leader xt, while [yt(at)]<sup>b</sup><sup>t</sup> represents the predicted logit for class b<sup>t</sup> on sample at. The backbone model for y<sup>t</sup> is a 4-layer CNN, leading to a nonconvex bilevel objective.

We compare SOGD with the following methods:

OAGD [\(Tarzanagh et al.,](#page-10-2) [2024\)](#page-10-2): A state-of-the-art static online bilevel gradient descent method using the Neumann series for hypergradient approximation.

SOBOW [\(Lin et al.,](#page-9-1) [2024\)](#page-9-1): A dynamic online bilevel gradient descent method using conjugate gradients (CG) for hypergradient approximation.

We conducted experiments on the MNIST [\(LeCun et al.,](#page-9-17) [2010\)](#page-9-17). We used a batch size of 64 per timestep. We evaluated cumulative runtime, balanced accuracy, and test accuracy, where balanced accuracy is the class-specific average accuracy:

$$\frac{1}{J} \sum_{j=1}^J \mathbb{P}_{\mathbf{a}_t \sim \mathcal{D}_j} [\operatorname{argmax}_i([\mathbf{y}_t(\mathbf{a}_t)]_i) = j],$$

with D<sup>j</sup> denoting the distribution over samples of class j [\(Li et al.,](#page-9-16) [2021\)](#page-9-16). Learning rates were tuned as β<sup>t</sup> = δ<sup>t</sup> = β ∈ {0.001, 0.005, 0.01, 0.05, 0.1} and α<sup>t</sup> = α ∈ {0.0001, 0.0005, 0.001, 0.005, 0.01} for all t ∈ [T]. The parameters γt, λt, η<sup>t</sup> were tuned as γ<sup>t</sup> = λ<sup>t</sup> = η<sup>t</sup> = γ ∈ {0.9, 0.99, 0.999}. The Neumann series iterations in OAGD and CG iterations in SOBOW were set to 5.

We evaluated performance over 400 timesteps in four 100 timestep phases, transitioning from an imbalanced (0.4 i ) to a balanced (0.8 i ) distribution for each class (i = 0, 1, . . . , 9). Figure [3](#page-6-4) (left) shows SOBOW's longer runtime due to CG complexity, while SOGD is the fastest with simultaneous updates. Figures [3](#page-6-4) (middle, right) show accuracy gains as balance increases, with SOGD achieving competitive accuracy.

# 6. Conclusion

We introduced a novel online bilevel optimization (OBO) framework that overcomes the limitations of existing algorithms, which often rely on extensive oracle information and incur high computational costs. Our approach uses limited feedback and zeroth-order updates for efficient hypergradient estimation and simultaneous updates of decision variables, achieving sublinear bilevel regret without window smoothing. Experiments on online parametric loss tuning and black-box adversarial attacks confirm its effectiveness.

# Impact Statements

This paper develops methods to advance online learning. While our work has societal implications, none require specific emphasis here.

- 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 References Agarwal, A., Dekel, O., and Xiao, L. Optimal algorithms for online convex optimization with multi-point bandit feedback. In *Colt*, pp. 28–40. Citeseer, 2010. Agarwal, N., Gonen, A., and Hazan, E. Learning in nonconvex games with an optimization oracle. In *Conference on Learning Theory*, pp. 18–29. PMLR, 2019. Aghasi, A. and Ghadimi, S. Fully zeroth-order bilevel programming via gaussian smoothing. *arXiv preprint arXiv:2404.00158*, 2024. Allen-Zhu, Z. and Li, Y. Neon2: Finding local minima via first-order oracles. *Advances in Neural Information Processing Systems*, 31, 2018. Bach, F. and Perchet, V. Highly-smooth zero-th order online optimization. In *Conference on Learning Theory*, pp. 257–283. PMLR, 2016. Bernstein, J., Wang, Y.-X., Azizzadenesheli, K., and Anandkumar, A. Signsgd: Compressed optimisation for nonconvex problems. In *International Conference on Machine Learning*, pp. 560–569. PMLR, 2018. Besbes, O., Gur, Y., and Zeevi, A. Non-stationary stochastic optimization. *Operations research*, 63(5):1227–1244, 2015. Bohne, J., Rosenberg, D., Kazantsev, G., and Polak, P. Online nonconvex bilevel optimization with bregman divergences. *arXiv preprint arXiv:2409.10470*, 2024. Bracken, J. and McGill, J. T. Mathematical programs with optimization problems in the constraints. *Operations Research*, 21(1):37–44, 1973. Bubeck, S., Stoltz, G., Szepesvari, C., and Munos, R. Online ´ optimization in x-armed bandits. *Advances in Neural Information Processing Systems*, 21, 2008. Chen, P.-Y., Zhang, H., Sharma, Y., Yi, J., and Hsieh, C.-
  - J. Zoo: Zeroth order optimization based black-box attacks to deep neural networks without training substitute models. In *Proceedings of the 10th ACM workshop on artificial intelligence and security*, pp. 15–26, 2017. Chen, T., Sun, Y., and Yin, W. Closing the gap: Tighter analysis of alternating stochastic gradient methods for bilevel problems. *Advances in Neural Information Processing Systems*, 34, 2021. Chen, X., Liu, S., Xu, K., Li, X., Lin, X., Hong, M., and Cox, D. Zo-adamm: Zeroth-order adaptive momentum method for black-box optimization. *Advances in neural information processing systems*, 32, 2019. Crockett, C., Fessler, J. A., et al. Bilevel methods for image reconstruction. *Foundations and Trends® in Signal Processing*, 15(2-3):121–289, 2022. Cutkosky, A. and Boahen, K. Anytime online-to-batch conversions and the conservative algorithm. *Advances in Neural Information Processing Systems*, 32, 2019. Dagreou, M., Ablin, P., Vaiter, S., and Moreau, T. A frame- ´ work for bilevel optimization that enables stochastic and global variance reduction algorithms. *arXiv preprint arXiv:2201.13409*, 2022. Dempe, S. *Foundations of bilevel programming*. Springer Science & Business Media, 2002. Duchi, J. C., Jordan, M. I., Wainwright, M. J., and Wibisono,
    - A. Optimal rates for zero-order convex optimization: The power of two function evaluations. *IEEE Transactions on Information Theory*, 61(5):2788–2806, 2015. Finn, C., Abbeel, P., and Levine, S. Model-agnostic metalearning for fast adaptation of deep networks. In *International Conference on Machine Learning*, pp. 1126–1135. PMLR, 2017. Finn, C., Rajeswaran, A., Kakade, S., and Levine, S. Online meta-learning. In *International Conference on Machine Learning*, pp. 1920–1930. PMLR, 2019. Flaxman, A. D., Kalai, A. T., and McMahan, H. B. Online convex optimization in the bandit setting: gradient descent without a gradient. *arXiv preprint cs/0408007*, 2004. Franceschi, L., Donini, M., Frasconi, P., and Pontil, M. Forward and reverse gradient-based hyperparameter optimization. In *International Conference on Machine Learning*, pp. 1165–1173. PMLR, 2017. Franceschi, L., Frasconi, P., Salzo, S., Grazzi, R., and Pontil,
    - M. Bilevel programming for hyperparameter optimization and meta-learning. In *International Conference on Machine Learning*, pp. 1568–1577. PMLR, 2018. Gao, X., Li, X., and Zhang, S. Online learning with nonconvex losses and non-stationary regret. In *International Conference on Artificial Intelligence and Statistics*, pp. 235–243. PMLR, 2018. Ghadimi, S. and Lan, G. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. *SIAM journal on optimization*, 23(4):2341–2368, 2013. Ghadimi, S. and Wang, M. Approximation methods for bilevel programming. *arXiv preprint arXiv:1802.02246*, 2018. Ghadimi, S., Lan, G., and Zhang, H. Mini-batch stochastic approximation methods for nonconvex stochastic composite optimization. *Mathematical Programming*, 155 (1-2):267–305, 2016. Goel, G., Lin, Y., Sun, H., and Wierman, A. Beyond online balanced descent: An optimal algorithm for smoothed online optimization. *Advances in Neural Information Processing Systems*, 32, 2019.

- 495 496 497 498 499 500 504 506 508 509 511 514 515 516 518 524 526 528 531 534 536 538 540 541 542 543 544 545 546 547 548 Guan, Z., Zhou, Y., and Liang, Y. On the hardness of online nonconvex optimization with single oracle feedback. In *The Twelfth International Conference on Learning Representations*, 2023a. Guan, Z., Zhou, Y., and Liang, Y. Online nonconvex optimization with limited instantaneous oracle feedback. In *The Thirty Sixth Annual Conference on Learning Theory*, pp. 3328–3355. PMLR, 2023b. Hallak, N., Mertikopoulos, P., and Cevher, V. Regret minimization in stochastic non-convex learning via a proximal-gradient approach. In *International Conference on Machine Learning*, pp. 4008–4017. PMLR, 2021. Hansen, P., Jaumard, B., and Savard, G. New branch-andbound rules for linear bilevel programming. *SIAM Journal on scientific and Statistical Computing*, 13(5):1194– 1217, 1992. Hazan, E. Introduction to online convex optimization. *Foundations and Trends® in Optimization*, 2(3-4):157–325, 2016a. URL [http://ocobook.cs.princeton.](http://ocobook.cs.princeton.edu/OCObook.pdf) [edu/OCObook.pdf](http://ocobook.cs.princeton.edu/OCObook.pdf). Hazan, E. Introduction to online convex optimization. *Foundations and Trends in Optimization*, 2(3-4):157–325, 2016b. Hazan, E., Agarwal, A., and Kale, S. Logarithmic regret algorithms for online convex optimization. *Machine Learning*, 69(2):169–192, 2007. Hazan, E., Singh, K., and Zhang, C. Efficient regret minimization in non-convex games. In *International Conference on Machine Learning*, pp. 1433–1441. PMLR, 2017. Heliou, A., Martin, M., Mertikopoulos, P., and Rahier, T. ´ Online non-convex optimization with imperfect feedback. *Advances in Neural Information Processing Systems*, 33: 17224–17235, 2020. Heliou, A., Martin, M., Mertikopoulos, P., and Rahier, ´
  - T. Zeroth-order non-convex learning via hierarchical dual averaging. In *International Conference on Machine Learning*, pp. 4192–4202. PMLR, 2021. Huang, F., Gao, S., Pei, J., and Huang, H. Accelerated zeroth-order and first-order momentum methods from mini to minimax optimization. *Journal of Machine Learning Research*, 23(36):1–70, 2022. Huang, Y., Cheng, Y., Liang, Y., and Huang, L. Online minmax problems with non-convexity and non-stationarity. *Transactions on Machine Learning Research*. Huang, Y., Cheng, Y., Liang, Y., and Huang, L. Online minmax problems with non-convexity and non-stationarity. *Transactions on Machine Learning Research*, 2023. Ji, K., Wang, Z., Zhou, Y., and Liang, Y. Improved zerothorder variance reduced algorithms and analysis for nonconvex optimization. In *International conference on machine learning*, pp. 3100–3109. PMLR, 2019. Ji, K., Yang, J., and Liang, Y. Bilevel optimization: Convergence analysis and enhanced design. In *International Conference on Machine Learning*, pp. 4882–4892. PMLR, 2021. Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In *International Conference on Learning Representations*, 2014. Kleinberg, R., Slivkins, A., and Upfal, E. Multi-armed bandits in metric spaces. In *Proceedings of the fortieth annual ACM symposium on Theory of computing*, pp. 681–690, 2008. Krichene, W., Balandat, M., Tomlin, C., and Bayen, A. The hedge algorithm on a continuum. In *International Conference on Machine Learning*, pp. 824–832. PMLR, 2015. LeCun, Y., Cortes, C., and Burges, C. Mnist handwritten digit database. *ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist*, 2, 2010. Li, M., Zhang, X., Thrampoulidis, C., Chen, J., and Oymak,
    - S. Autobalance: Optimized loss functions for imbalanced data. *Advances in Neural Information Processing Systems*, 34:3163–3177, 2021. Lin, S., Sow, D., Ji, K., Liang, Y., and Shroff, N. Nonconvex bilevel optimization with time-varying objective functions. *Advances in Neural Information Processing Systems*, 36, 2024. Liu, H., Simonyan, K., and Yang, Y. Darts: Differentiable architecture search. *arXiv preprint arXiv:1806.09055*, 2018a. Liu, S., Chen, J., Chen, P.-Y., and Hero, A. Zeroth-order online alternating direction method of multipliers: Convergence analysis and applications. In *International Conference on Artificial Intelligence and Statistics*, pp. 288–297. PMLR, 2018b. Lorraine, J., Vicol, P., and Duvenaud, D. Optimizing millions of hyperparameters by implicit differentiation. In *International conference on artificial intelligence and statistics*, pp. 1540–1552. PMLR, 2020. Luo, L., Ye, H., Huang, Z., and Zhang, T. Stochastic recursive gradient descent ascent for stochastic nonconvexstrongly-concave minimax problems. *Advances in Neural Information Processing Systems*, 33:20566–20577, 2020. Lv, Y., Hu, T., Wang, G., and Wan, Z. A penalty function method based on kuhn–tucker condition for solving linear bilevel programming. *Applied Mathematics and Computation*, 188(1):808–813, 2007. Nesterov, Y. Smooth minimization of non-smooth functions. *Mathematical programming*, 103:127–152, 2005. Nesterov, Y. and Spokoiny, V. Random gradient-free minimization of convex functions. *Foundations of Computational Mathematics*, 17(2):527–566, 2017.

- 551 554 556 558 560 564 566 568 571 574 576 578 580 581 582 583 584 585 586 587 588 589 590 594 596 598 600 601 602 603 604 Roy, A., Balasubramanian, K., Ghadimi, S., and Mohapatra,
  - P. Stochastic zeroth-order optimization under nonstationarity and nonconvexity. *Journal of Machine Learning Research*, 23(64):1–47, 2022. Shalev-Shwartz, S. et al. Online learning and online convex optimization. *Foundations and trends in Machine Learning*, 4(2):107–194, 2011. Shamir, O. An optimal algorithm for bandit and zero-order convex optimization with two-point feedback. *Journal of Machine Learning Research*, 18(52):1–11, 2017. Sow, D., Ji, K., and Liang, Y. On the convergence theory for hessian-free bilevel algorithms. *Advances in Neural Information Processing Systems*, 35:4136–4149, 2022. Stackelberg, H. v. Theory of the market economy. *Oxford University Press*, 1952. Stadie, B., Zhang, L., and Ba, J. Learning intrinsic rewards as a bi-level optimization problem. In *Conference on Uncertainty in Artificial Intelligence*, pp. 111–120. PMLR, 2020. Suggala, A. S. and Netrapalli, P. Online non-convex learning: Following the perturbed leader is optimal. In *Algorithmic Learning Theory*, pp. 845–861. PMLR, 2020. Tarzanagh, D. A., Nazari, P., Hou, B., Shen, L., and Balzano,
  - L. Online bilevel optimization: Regret analysis of online alternating gradient methods. In *International Conference on Artificial Intelligence and Statistics*, pp. 2854–2862. PMLR, 2024. Wang, Z., Balasubramanian, K., Ma, S., and Razaviyayn,
  - M. Zeroth-order algorithms for nonconvex minimax problems with improved complexities. *arXiv preprint arXiv:2001.07819*, 2020. Zhang, Y., Zhou, Y., Ji, K., and Zavlanos, M. M. Boosting one-point derivative-free online optimization via residual feedback. *arXiv preprint arXiv:2010.07378*, 2020. Zhou, W., Li, Y., Yang, Y., Wang, H., and Hospedales,
  - T. Online meta-critic learning for off-policy actor-critic methods. *Advances in Neural Information Processing Systems*, 33:17662–17673, 2020. Zinkevich, M. Online convex programming and generalized infinitesimal gradient ascent. In *Proceedings of the 20th international conference on machine learning (icml-03)*, pp. 928–936, 2003.

*641* We first provide several useful lemmas for the main proofs.

#### A. Related Work

BO was introduced in game theory by [\(Stackelberg,](#page-10-0) [1952\)](#page-10-0) and modeled mathematically in [\(Bracken & McGill,](#page-8-0) [1973\)](#page-8-0). Initial works [\(Hansen et al.,](#page-9-18) [1992;](#page-9-18) [Lv et al.,](#page-9-19) [2007\)](#page-9-19) reduced it to single-level optimization. Recently, gradient-based approaches have gained popularity for their simplicity and efficacy [\(Franceschi et al.,](#page-8-8) [2017;](#page-8-8) [Ghadimi & Wang,](#page-8-9) [2018;](#page-8-9) [Ji et al.,](#page-9-4) [2021;](#page-9-4) [Chen et al.,](#page-8-10) [2021\)](#page-8-10), though they assume offline objectives.

OBO was initiated by [Tarzanagh et al.](#page-10-2) [\(2024\)](#page-10-2), proposing the OAGD method with regret bounds. [\(Huang et al.,](#page-9-2) [2023\)](#page-9-2) developed algorithms for online minimax optimization, special cases of OBO with local regret guarantees. [\(Lin et al.,](#page-9-1) [2024\)](#page-9-1) introduced SOBOW, a single-loop optimizer using window-smoothed functions and multiple CGs for nonconvexstrongly-convex cases. Unlike these works, we propose using *projected gradient* as a more general performance measure for constrained objectives, focusing on the original functions and their regret; See Table [1](#page-1-1) for a comparison.

Single-Level Regret Minimization. Single-level online optimization predominantly focuses on convex problems, either with static or dynamic convex regret minimization [\(Zinkevich,](#page-10-3) [2003;](#page-10-3) [Hazan,](#page-9-20) [2016a;](#page-9-20) [Shalev-Shwartz et al.,](#page-10-8) [2011\)](#page-10-8). Nonconvex online optimization [\(Hazan et al.,](#page-9-6) [2017;](#page-9-6) [Guan et al.,](#page-9-21) [2023b;](#page-9-21)[a\)](#page-9-12) poses greater challenges than its convex counterparts [\(Shalev-Shwartz et al.,](#page-10-8) [2011;](#page-10-8) [Zinkevich,](#page-10-3) [2003;](#page-10-3) [Hazan et al.,](#page-9-22) [2007;](#page-9-22) [Besbes et al.,](#page-8-12) [2015\)](#page-8-12). Notable contributions in this field include adversarial multi-armed bandit algorithms [\(Bubeck et al.,](#page-8-17) [2008;](#page-8-17) [Heliou et al.](#page-9-23) ´ , [2020;](#page-9-23) [2021;](#page-9-24) [Krichene et al.,](#page-9-25) [2015\)](#page-9-25) and the Follow-the-Perturbed-Leader approach [\(Agarwal et al.,](#page-8-18) [2019;](#page-8-18) [Kleinberg et al.,](#page-9-26) [2008;](#page-9-26) [Suggala & Netrapalli,](#page-10-9) [2020\)](#page-10-9). Hazan et al. [\(Hazan et al.,](#page-9-6) [2017\)](#page-9-6) introduced window-smoothed local regret for gradient averaging in non-convex models, which Hallak et al. [\(Hallak et al.,](#page-9-8) [2021\)](#page-9-8) extended to non-smooth, non-convex problems. Inspired by their work, we employ local regret for Online Bandit Optimization (OBO) without window-smoothing.

Zeroth-Order Optimization. Single-Level ZO Optimization has been widely studied in both offline [\(Ghadimi & Lan,](#page-8-19) [2013;](#page-8-19) [Duchi et al.,](#page-8-20) [2015;](#page-8-20) [Agarwal et al.,](#page-8-13) [2010;](#page-8-13) [Nesterov & Spokoiny,](#page-9-9) [2017\)](#page-9-9) and online settings [\(Liu et al.,](#page-9-13) [2018b;](#page-9-13) [Guan](#page-9-12) [et al.,](#page-9-12) [2023a;](#page-9-12)[b;](#page-9-21) [Zhang et al.,](#page-10-10) [2020;](#page-10-10) [Bach & Perchet,](#page-8-21) [2016\)](#page-8-21). We next review closely related work. Liu et al. [\(Liu et al.,](#page-9-13) [2018b\)](#page-9-13) proposed ZOO-ADMM, a gradient-free online optimization algorithm utilizing ADMM. Guan et al. [\(Guan et al.,](#page-9-21) [2023b\)](#page-9-21) studied online non-convex optimization with limited oracle feedback. Research on online non-convex optimization with bandit feedback includes work by Heliou et al. [\(Heliou et al.](#page-9-23) ´ , [2020\)](#page-9-23), which established bounds on global static and dynamic regret using dual averaging, further refined in [\(Heliou et al.](#page-9-24) ´ , [2021\)](#page-9-24). Gao et al. [\(Gao et al.,](#page-8-22) [2018\)](#page-8-22) extended these ideas to ZO algorithms. Flaxman et al. [\(Flaxman et al.,](#page-8-23) [2004\)](#page-8-23) provided algorithms for bandit online optimization of convex functions using ZO gradient approximation. Our work closely relates to [\(Sow et al.,](#page-10-11) [2022\)](#page-10-11), which proposes a Hessian-free method approximating the Jacobian matrix using a ZO method based on finite differences of gradients. In contrast, our method uses function oracles to approximate both the Hessian and gradients and is derivative-free. We also point out the recent work [\(Aghasi & Ghadimi,](#page-8-24) [2024\)](#page-8-24) on ZO stochastic algorithms for solving bilevel problems when neither the upper/lower objective values nor their unbiased gradient estimates are available. Their approach, limited to the *offline* setting, does not include numerical results, thus leaving its practical efficiency unclear.

# B. Additional Preliminaries and Notations

### B.1. Preliminary Lemmas

Definition B.1 (Projected gradient [\(Ghadimi et al.,](#page-8-25) [2016\)](#page-8-25)). Let X ⊂ R <sup>d</sup><sup>1</sup> be a closed convex set. Then, the projected gradient for any α<sup>t</sup> > 0 and p ∈ <sup>R</sup> d<sup>1</sup> is defined as

$$\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}; \mathbf{p}) := \frac{1}{\alpha_t} (\mathbf{x} - \mathbf{x}^+), \quad (31a)$$

where

$$\mathbf{x}^+ = \Pi_{\mathcal{X}}(\mathbf{x} - \alpha_t \mathbf{p}), \quad (31b)$$

and Π<sup>X</sup> [·] denotes the orthogonal projection operator onto set X .

Lemma B.2. [Goel et al.](#page-8-26) [\(2019,](#page-8-26) Lemma 13) *If* f : X → <sup>R</sup> *is a* µ<sup>f</sup> *-strongly convex function with respect to some norm* ∥ · ∥*, and* x ∗ *is the minimizer of* f *(i.e.* x <sup>∗</sup> = arg minx∈X f(x)*), then we have* ∀ x ∈ X *,*

$$\frac{\mu_f}{2} \|\mathbf{x} - \mathbf{x}^*\|^2 \leq f(\mathbf{x}) - f(\mathbf{x}^*) \leq \frac{1}{2\mu_f} \|\nabla f(\mathbf{x})\|^2.$$

689 690

694

696

698

700

704

706

708 709

711

*gradient at any given point* x ∈ R d *in terms of the objective sub optimality at* x*, as follows:*

$$\frac{1}{2L}\|\nabla f(\mathbf{x})\|^2 \leq f(\mathbf{x}) - f(\mathbf{x}^*) \leq \frac{L}{2}\|\mathbf{x} - \mathbf{x}^*\|^2. \quad (32)$$

Lemma B.4. *For any set of vectors* {xi} m <sup>i</sup>=1 *with* x<sup>i</sup> ∈ <sup>R</sup> d *, we have*

$$\left\| \sum_{i=1}^m \mathbf{x}_i \right\|^2 \leq m \sum_{i=1}^m \|\mathbf{x}_i\|^2.$$

Lemma B.5. *For any* x, y ∈ R d *, the following holds for any* c > 0*:*

$$\|\mathbf{x} + \mathbf{y}\|^2 \leq (1+c)\|\mathbf{x}\|^2 + \left(1 + \frac{1}{c}\right)\|\mathbf{y}\|^2, \quad \text{and} \quad (33)$$

$$\|\mathbf{x} - \mathbf{y}\|^2 \geq (1-c) \|\mathbf{x} - \mathbf{z}\|^2 + \left(1 - \frac{1}{c}\right) \|\mathbf{z} - \mathbf{y}\|^2. \quad (34)$$

We provide a set of auxiliary lemmas that will be used in establishing the proofs for the main theorems.

Lemma B.6. [Ghadimi et al.](#page-8-25) [\(2016,](#page-8-25) Proposition 1) *Let* P<sup>X</sup> ,α<sup>t</sup> (x; p) *be defined in Definition [B.1.](#page-11-0) Then, for any* p<sup>1</sup> *and* p<sup>2</sup> *in* R d *, we*

$$\|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}; \mathbf{p}_1) - \mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}; \mathbf{p}_2)\| \leq \|\mathbf{p}_1 - \mathbf{p}_2\| .$$

Lemma B.7. [Hazan et al.](#page-9-6) [\(2017,](#page-9-6) Proposition 2.4) *Let* P<sup>X</sup> ,α<sup>t</sup> (x; p) *be the projected gradient as per Definition [B.1.](#page-11-0) For any* x, p1, p<sup>2</sup> ∈ <sup>R</sup> <sup>d</sup> *and* α<sup>t</sup> > 0 *it holds that*

$$\|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}; \mathbf{p}_1 + \mathbf{p}_2)\| \leq \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}; \mathbf{p}_1)\| + \|\mathbf{p}_2\|.$$

Lemma B.8. *Let* P<sup>X</sup> ,α<sup>t</sup> (x; p) *be as given in Definition [B.1.](#page-11-0) Then, for any* p ∈ <sup>R</sup> <sup>d</sup> *and* α<sup>t</sup> > 0*, we have*

$$\langle \mathbf{p}, \mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}; \mathbf{p}) \rangle \geq \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}; \mathbf{p})\|^2.$$

*Proof.* By the definition of x <sup>+</sup>, the optimality condition of [\(31b\)](#page-11-1) is

$$\left\langle \mathbf{p} + \frac{1}{\alpha_t}(\mathbf{x}^+ - \mathbf{x}), \mathbf{z} - \mathbf{x}^+ \right\rangle \geq 0, \quad \forall \mathbf{z} \in \mathcal{X}.$$

Letting z = x, we obtain

$$\langle \mathbf{p}, \mathbf{x} - \mathbf{x}^+ \rangle \geq \frac{1}{\alpha_t} \langle \mathbf{x} - \mathbf{x}^+, \mathbf{x} - \mathbf{x}^+ \rangle,$$

which can be rearranged to

$$\begin{aligned} \langle \mathbf{p}, \mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}; \mathbf{p}) \rangle &= \frac{1}{\alpha_t} \langle \mathbf{p}, \mathbf{x} - \mathbf{x}^+ \rangle \geq \frac{1}{\alpha_t^2} \langle \mathbf{x} - \mathbf{x}^+, \mathbf{x} - \mathbf{x}^+ \rangle \\ &= \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}; \mathbf{p})\|^2. \end{aligned}$$

## B.2. Examples

Theorem [3.6](#page-4-0) achieves sublinear bilevel regret when the variations V<sup>T</sup> and H2,T are both o(T). Below, we provide some examples of online optimization in both single-level and bilevel settings to illustrate when this occurs.

*Example* B.9*.* Consider function ft(x) = ∥Atx − bt∥ 2 , where A<sup>t</sup> = [1, 0; 0, 1 + <sup>1</sup> t ], x = b<sup>t</sup> = (1, 1). Then, V<sup>T</sup> :=

718

724

726

728

731

734

736

738

751

754

756

758

760

764

766

P<sup>T</sup> <sup>t</sup>=2 max<sup>x</sup> |ft(x) − ft−1(x)| = P<sup>T</sup> <sup>t</sup>=2 | 1 t 2 − t−1 2 |. By a <sup>2</sup> − b <sup>2</sup> = (a − b)(a + b), we have

$$\begin{aligned} V_T &= \sum_{t=2}^T \left| \left( \frac{1}{t} - \frac{1}{t-1} \right) - \left( \frac{1}{t} + \frac{1}{t-1} \right) \right| \\ &= \sum_{t=2}^T \left| \left( \frac{t-1-t}{t(t-1)} \right) - \left( \frac{1}{t} + \frac{1}{t-1} \right) \right| \\ &= \sum_{t=2}^T \left| \left( -\frac{1}{t(t-1)} \right) - \left( \frac{1}{t} + \frac{1}{t-1} \right) \right| \\ &= \sum_{t=2}^T \left| \frac{1}{t(t-1)} \right| \left| \frac{t-1+t}{t(t-1)} \right| \\ &= \sum_{t=2}^T \left| \frac{2}{t(t-1)^2} \right|. \end{aligned}$$

Then, V<sup>T</sup> ≤ P<sup>T</sup> t=2 t <sup>3</sup> ≈ R T 2 t <sup>3</sup> dt = <sup>4</sup> − <sup>T</sup> <sup>2</sup> . As T → ∞, V<sup>T</sup> becomes bounded and approaches a constant value, indicating that V<sup>T</sup> grows slower than T itself.

*Example* B.10*.* Let ft(x) = (− T , 0, 0, 0) if t is even, and ft(x) = (0, − 1 T , 0, 0) if t is odd. Then, V<sup>T</sup> = P<sup>T</sup> <sup>t</sup>=2 max<sup>x</sup> |ft(x) − ft−1(x)| = O(1).

*Example* B.11*.* Let x ∈ X = [−1, 1] ⊂ <sup>R</sup>, y ∈ <sup>R</sup>, and consider a sequence of quadratic cost functions

$$f_t(x, y) = \frac{1}{2} \left( x + 2a_t^{(1)} \right)^2 + \frac{1}{2} \left( y - a_t^{(2)} \right)^2,$$

$$g_t(x, y) = \frac{1}{2} y^2 - \left( x - a_t^{(2)} \right) y,$$

where a (1) <sup>t</sup> = 1/t and a (2) <sup>t</sup> = 1/ √ t for all t ∈ [T].

We have

$$y_t^*(x) = x - a_t^{(2)}.$$

We have

$$\begin{aligned}
& f_t(x, y_t^*(x)) - f_{t-1}(x, y_{t-1}^*(x)) \\
&= \frac{1}{2} \left[ \left( x + 2a_t^{(1)} \right)^2 - \left( x + 2a_{t-1}^{(1)} \right)^2 \right] + \frac{1}{2} \left[ \left( y_t^*(x) - a_t^{(2)} \right)^2 - \left( y_{t-1}^*(x) - a_{t-1}^{(2)} \right)^2 \right] \\
&= \frac{1}{2} \left[ \left( x^2 + 4xa_t^{(1)} + 4(a_t^{(1)})^2 \right) - \left( x^2 + 4xa_{t-1}^{(1)} + 4(a_{t-1}^{(1)})^2 \right) \right] \\
&+ \frac{1}{2} \left[ \left( (x - a_t^{(2)})^2 - 2(x - a_t^{(2)})a_t^{(2)} + (a_t^{(2)})^2 \right) - \left( (x - a_{t-1}^{(2)})^2 - 2(x - a_{t-1}^{(2)})a_{t-1}^{(2)} + (a_{t-1}^{(2)})^2 \right) \right] \\
&= 2x \left( a_t^{(1)} - a_{t-1}^{(1)} - a_t^{(2)} + a_{t-1}^{(2)} \right) + 2 \left( (a_t^{(1)})^2 - (a_{t-1}^{(1)})^2 + (a_t^{(2)})^2 - (a_{t-1}^{(2)})^2 \right).
\end{aligned}$$

Taking the maximum over x and using x ∈ [−1, 1] :

$$\sup_x |f_t(x, y_t^*(x)) - f_{t-1}(x, y_{t-1}^*(x))| = 2 \left| a_t^{(1)} - a_{t-1}^{(1)} \right| + 2 \left| -a_t^{(2)} + a_{t-1}^{(2)} \right| + 2 \left| (a_t^{(1)})^2 - (a_{t-1}^{(1)})^2 \right| + 2 \left| (a_t^{(2)})^2 - (a_{t-1}^{(2)})^2 \right|.$$

774

776

778

794

796

800

804

806

808

Since a (1) <sup>t</sup> = 1/t and a (2) <sup>t</sup> = 1/ √ t for all t ∈ [T], then we have

$$|a_t^{(1)} - a_{t-1}| \approx \frac{1}{t^2}, \quad |a_t^{(2)} - a_{t-1}| \approx \frac{1}{2t^{3/2}},$$

$$|(a_t^{(1)})^2 - (a_{t-1}^{(1)})^2| \approx \frac{1}{t^3}, \quad |(a_t^{(2)})^2 - (a_{t-1}^{(2)})^2| \approx \frac{1}{t^2}.$$

Then, we get

$$V_T := \sum_{t=2}^T \sup_x |f_t(x, y_t^*(x)) - f_{t-1}(x, y_{t-1}^*(x))| = \sum_{t=2}^T \left( \frac{2}{t^2} + \frac{1}{2t^{3/2}} + \frac{1}{t^3} \right).$$

The series P<sup>T</sup> <sup>t</sup>=2 2 t <sup>2</sup> + 1 2t <sup>3</sup>/<sup>2</sup> + 1 t converges, implying V<sup>T</sup> = O(1). Moreover, we have

$$\begin{aligned} H_{2,T} &= \sum_{t=2}^T \sup_x \|y_t^*(x) - y_{t-1}^*(x)\|^2 = \sum_{t=2}^T \sup_x \|x - a_t^{(2)} - x + a_{t-1}^{(2)}\|^2 \\ &= \sum_{t=2}^T | -a_t^{(2)} + a_{t-1}^{(2)} |^2 = \sum_{t=2}^T |a_t^{(2)} - a_{t-1}^{(2)}|^2 \approx \sum_{t=2}^T \frac{1}{4t^3}, \end{aligned}$$

which implies H2,T = O(1).

To achieve V<sup>T</sup> = o(T), the changes in the cost functions ft(x, y ∗ t (x)) and y ∗ t (x) should decay to zero faster than O(1/t). For example, if the coefficients in the functions change as O(1/t<sup>a</sup> ) with a > 1, then the cumulative sum over T will be o(T). When ft(x, y ∗ t (x)) and y ∗ t (x) decay as O(1/ √ t), then the total variation grows at most as O( √ T).

# C. Proof of Regret Bounds for Simultaneous Online Gradient Descent (SOGD)

Proof Roadmap. We introduce Lemma [C.2,](#page-16-0) which quantifies the error between the approximated direction of the momentum-based gradient estimator, d y t , and the true direction, ∇ygt(xt, yt), at each iteration. To bound the error of the lower-level variable, we provide Lemma [C.4,](#page-18-0) which captures the gap ∥yt+1 − y ∗ t (xt)∥ 2 and incorporates the error introduced in Lemma [C.2.](#page-16-0) Moreover, we provide Lemma [C.5,](#page-20-0) which quantifies the error between the approximated direction of the momentum-based gradient estimator, d v t , and the true direction, ∇<sup>2</sup> y g<sup>t</sup> (zt) v<sup>t</sup> + ∇yft(zt), at each iteration. To bound the error of the system solution, we provide Lemma [C.8,](#page-24-0) which captures the gap ∥vt+1 − v ∗ t (xt)∥ 2 and incorporates the error introduced in Lemma [C.5.](#page-20-0) Moreover, we provide Lemma [C.9,](#page-26-0) which quantifies the error between the approximated direction of the momentum-based hypergradient estimator, d x t , and the true direction, ∇xft(zt) + ∇<sup>2</sup> xyg<sup>t</sup> (zt) vt, at each iteration. We also present Lemma [C.11,](#page-28-0) which provides an upper bound for the projection mapping and relates to the three errors discussed in Lemmas [C.4,](#page-18-0) [C.8,](#page-24-0) and [C.9.](#page-26-0) Finally, by combining these lemmas and appropriately setting the parameters, we achieve the desired result.

# C.1. Proof of Lemma [3.1](#page-2-5)

*Proof.* SOBOW [\(Lin et al.,](#page-9-1) [2024\)](#page-9-1) has estimated the hypergradient as the weighted average of previous ones over a sliding window of size w for a given B<sup>t</sup> := {ξ1, . . . , ξb} drawn i.i.d. from the distribution D<sup>f</sup> , as follows:

$$\hat{\nabla} F_{t,\nu}(\mathbf{x}_t, \mathbf{y}_t; \mathcal{B}_t) = \frac{1}{W} \sum_{i=0}^{w-1} \nu^i \hat{\nabla} f_{t-i}(\mathbf{x}_{t-i}, \mathbf{y}_{t-i}; \mathcal{B}_{t-i}),$$

with W = P<sup>w</sup>−<sup>1</sup> <sup>i</sup>=0 ν i , ν ∈ (0, 1). Let ν = 1 − η for η ∈ (0, 1).

Then, the above equality is equivalent to

$$\hat{\nabla} F_{t,\nu}(\mathbf{x}_t, \mathbf{y}_t; \mathcal{B}_t) = \frac{1}{W} \sum_{j=t-w+1}^t (1-\eta)^{t-j} \hat{\nabla} f_j(\mathbf{x}_j, \mathbf{y}_j; \mathcal{B}_j), \quad (35)$$

828

831

834

836

838

854

856

858

860

864

866

868

874

876

Let dˆ<sup>x</sup> t := ∇ˆ Ft,ν(xt, yt; Bt). Then [\(35\)](#page-14-0) is equivalent to

$$\hat{\mathbf{d}}_t^x = \frac{1}{W} \hat{\nabla} f_t(\mathbf{x}_t, \mathbf{y}_t; \mathcal{B}_t) + (1 - \eta) \hat{\mathbf{d}}_{t-1}^x - \frac{(1 - \eta)^w}{W} \hat{\nabla} f_{t-w}(\mathbf{x}_{t-w}, \mathbf{y}_{t-w}; \mathcal{B}_{t-w}), \quad (36)$$

with fi(·) = 0 for all i ≤ 0.

If w = t and W = 1 η , then, we have

$$\hat{\mathbf{d}}_t^x = \eta \hat{\nabla} f_t(\mathbf{x}_t, \mathbf{y}_t; \mathcal{B}_t) + (1 - \eta) \hat{\mathbf{d}}_{t-1}^x.$$

### C.2. Bounds on the Inner Decision Variable

We first provide a lemma that characterizes the Lipschitz continuity of approximate gradients, inner, and system solutions.

Lemma C.1. *Under Assumptions [3.2](#page-3-2) and [3.3,](#page-3-3) for all* x, x ′ ∈ X *, and the search directions* {d x t } T <sup>t</sup>=1 *and* {d v t } T <sup>t</sup>=1 *generated by Algorithm [1,](#page-2-1) we have*

$$\|\mathbf{d}_t^{\mathbf{x}} - \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))\|^2 \leq M_f^2 \left( \|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 + \|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2 \right), \quad (37a)$$

$$\|\mathbf{d}_t^Y\|^2 \leq M_{\mathbf{v}}^2 \left( \|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 + \|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2 \right), \quad (37b)$$

$$\|\nabla f_t(\mathbf{x}, \mathbf{y}^*(\mathbf{x})) - \nabla f_t(\mathbf{x}', \mathbf{y}^*(\mathbf{x}'))\| \leq L_f \|\mathbf{x} - \mathbf{x}'\|, \quad (37c)$$

$$\|\mathbf{y}_t^*(\mathbf{x}) - \mathbf{y}_t^*(\mathbf{x}')\| \leq L_{\mathbf{y}} \|\mathbf{x} - \mathbf{x}'\|, \quad (37d)$$

$$\|\mathbf{v}_t^*(\mathbf{x}) - \mathbf{v}_t^*(\mathbf{x}')\| \leq L_{\mathbf{v}} \|\mathbf{x} - \mathbf{x}'\|, \quad (37e)$$

*where* M<sup>f</sup> *,* Mv*, and* (Ly, Lv, L<sup>f</sup> ) *are defined in* [\(40\)](#page-15-0)*,* [\(41\)](#page-16-1)*, and* [\(42\)](#page-16-2)*, respectively.*

*Proof.* We first show [\(37a\)](#page-15-1).

Using Assumptions [3.2](#page-3-2) and [3.3,](#page-3-3) we have ∇<sup>2</sup> y g<sup>t</sup> (xt, y ∗ t (xt)) ⪰ µg, and

$$\|\mathbf{v}_t^*(\mathbf{x}_t)\| = \|(\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))^{-1} \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))\| \leq \frac{\ell_{f,0}}{\mu_g}. \quad (38)$$

Observe that

$$\begin{aligned}
\|\mathbf{d}_t^x - \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))\| &\leq \|\nabla_{\mathbf{x}} f_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}} f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))\| \\
&\quad + \|\mathbf{v}_t \nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) - \mathbf{v}_t^*(\mathbf{x}_t) \nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))\| \\
&\leq \|\nabla_{\mathbf{x}} f_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}} f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))\| \\
&\quad + \|\nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t)\| \|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\| \\
&\quad + \|\mathbf{v}_t^*(\mathbf{x}_t)\| \|\nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))\| \\
&\leq \left( \ell_{f,1} + \frac{\ell_{g,2} \ell_{f,0}}{\mu_g} \right) \|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\| + \ell_{g,1} \|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\| \\
&\leq M_f^2 (\|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\| + \|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|), \tag{39}
\end{aligned}$$

where

$$M_f := \sqrt{2} \max \left\{ \ell_{f,1} + \frac{\ell_{g,2} \ell_{f,0}}{\mu_g}, \ell_{g,1} \right\}, \quad (40)$$

the third inequality is by Assumption [3.3,](#page-3-3) and the last inequality follows from [\(38\)](#page-15-2).

887 888

890

894

896

898

911

914 915 916

918

924

928

Since d v t ∗ := ∇yft(xt, y ∗ t (xt)) + ∇<sup>2</sup> y g<sup>t</sup> (xt, y ∗ t (xt)) v ∗ t (xt) = 0, we have

$$\begin{aligned} \|\mathbf{d}_t^{\mathbf{y}}\| &= \|\mathbf{d}_t^{\mathbf{y}} - \mathbf{d}_t^{\mathbf{y}*}\| \\ &= \|\mathbf{v}_t \nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) + \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t) \\ &\quad - (\mathbf{v}_t^*(\mathbf{x}_t) \nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) + \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\| \\ &\leq \|(\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))) \mathbf{v}_t^*(\mathbf{x}_t)\| \\ &\quad + \|\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) (\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t))\| \\ &\quad + \|\nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))\|. \end{aligned}$$

Then, from Assumption [3.3](#page-3-3) and [\(38\)](#page-15-2), we have

$$\begin{aligned} \|\mathbf{d}_t^{\mathbf{y}}\| &\leq \ell_{g,2}\|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\| \|\mathbf{v}_t^*(\mathbf{x}_t)\| + \ell_{g,1}\|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\| + \ell_{f,1}\|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\| \\ &\leq \left( \frac{\ell_{g,2}\ell_{f,0}}{\mu_g} + \ell_{f,1} \right) \|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\| + \ell_{g,1}\|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\| \\ &\leq M_{\mathbf{v}}(\|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\| + \|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|), \end{aligned}$$

where

$$M_{\mathbf{v}} := \sqrt{2} \max \left\{ \frac{\ell_{g,2} \ell_{f,0}}{\mu_g} + \ell_{f,1}, \ell_{g,1} \right\}. \quad (41)$$

The proofs of Eqs. [\(37c\)](#page-15-4)-[\(37e\)](#page-15-5) follow from [Tarzanagh et al.](#page-10-2) [\(2024,](#page-10-2) Lemma 17) by setting

$$L_{\mathbf{y}} := \frac{\ell_{g,1}}{\mu_g},$$

$$L_{\mathbf{v}} := \ell_{f,1} + \frac{\ell_{g,1}\ell_{f,1}}{\mu_g} + \frac{\ell_{f,0}}{\mu_g} \left( \ell_{g,2} + \frac{\ell_{g,1}\ell_{g,2}}{\mu_g} \right), \quad (42)$$

$$L_f := \ell_{f,1} + \frac{\ell_{g,1}(\ell_{f,1} + M_f)}{\mu_g} + \frac{\ell_{f,0}}{\mu_g} \left( \ell_{g,2} + \frac{\ell_{g,1}\ell_{g,2}}{\mu_g} \right),$$

where the other constants are defined in Assumption [3.3.](#page-3-3)

Lemma C.2. *Suppose Assumptions [3.5,](#page-3-5)* [B3.](#page-3-10) *and* [C1.](#page-3-11) *hold. Let* {(xt, yt, vt)} T <sup>t</sup>=1 *be generated according to Algorithm [1.](#page-2-1) For* e g <sup>t</sup> *defined as*

$$e_t^g := \mathbf{d}_t^y - \nabla_y g_t(\mathbf{x}_t, \mathbf{y}_t), \quad (43)$$

*we have:*

$$\begin{aligned} \mathbb{E}\|e_{t+1}^g\|^2 &\leq (1 - \gamma_{t+1})^2(1 + 48\ell_{g,1}^2\beta_t^2)\mathbb{E}\|e_t^g\|^2 + 2\gamma_{t+1}^2\frac{\sigma_y^2}{b} + 24(1 - \gamma_{t+1})^2\ell_{g,1}^2\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 \\ &\quad + 6(1 - \gamma_{t+1})^2\mathbb{E}\|\nabla_{\mathbf{y}}g_t(\mathbf{z}_{t+1}) - \nabla_{\mathbf{y}}g_{t+1}(\mathbf{z}_{t+1})\|^2 \\ &\quad + 48(1 - \gamma_{t+1})^2\ell_{g,1}^2\beta_t^2\mathbb{E}\|\nabla_{\mathbf{y}}g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2. \end{aligned} \tag{44}$$

*Proof.* From Algorithm [1,](#page-2-1) we have

$$\mathbf{d}_{t+1}^y = \nabla_y g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) + (1 - \gamma_{t+1})(\mathbf{d}_t^y - \nabla_y g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1})).$$

Then, we have

$$\begin{aligned}\mathbb{E}\|e_{t+1}^g\|^2 &= \mathbb{E}\|\mathbf{d}_{t+1}^y - \nabla_y g_{t+1}(\mathbf{z}_{t+1})\|^2 \\ &= \mathbb{E}\|\nabla_y g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) + (1 - \gamma_{t+1})(\mathbf{d}_t^y - \nabla_y g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1})) - \nabla_y g_{t+1}(\mathbf{z}_{t+1})\|^2 \\ &= \mathbb{E}\|(1 - \gamma_{t+1})e_t^g + (\nabla_y g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \nabla_y g_{t+1}(\mathbf{z}_{t+1})) \\ &\quad - (1 - \gamma_{t+1})(\nabla_y g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1})) - \nabla_y g_t(\mathbf{z}_t)\|^2,\end{aligned}$$

938

954

956

958

971

974

976

978

which implies that

$$\begin{aligned} \mathbb{E}\|e_{t+1}^g\|^2 &= (1 - \gamma_{t+1})^2 \mathbb{E}\|e_t^g\|^2 + \mathbb{E}\|(\nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_{t+1})) \\ &\quad - (1 - \gamma_{t+1}) (\nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1})) - \nabla_{\mathbf{y}} g_t(\mathbf{z}_t)\|^2 \\ &\leq (1 - \gamma_{t+1})^2 \mathbb{E}\|e_t^g\|^2 + 2\gamma_{t+1}^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_{t+1})\|^2 \\ &\quad + 2(1 - \gamma_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) \\ &\quad - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1}) + \nabla_{\mathbf{y}} g_t(\mathbf{z}_t)\|^2 \\ &\leq (1 - \gamma_{t+1})^2 \mathbb{E}\|e_t^g\|^2 + 2\gamma_{t+1}^2 \frac{\sigma_{\mathbf{y}}^2}{\bar{b}} \\ &\quad + 2(1 - \gamma_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) \\ &\quad - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1}) + \nabla_{\mathbf{y}} g_t(\mathbf{z}_t)\|^2, \end{aligned}$$

where the second inequality follows from Cauchy–Schwartz inequality and Assumption [3.5.](#page-3-5) Moreover, from Cauchy–Schwartz inequality, we have

$$\begin{aligned} \mathbb{E}\|e_{t+1}^g\|^2 &\leq (1 - \gamma_{t+1})^2 \mathbb{E}\|e_t^g\|^2 + 2\gamma_{t+1}^2 \frac{\sigma_y^2}{b} \\ &\quad + 6(1 - \gamma_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{z}_t) - \nabla_{\mathbf{y}} g_t(\mathbf{z}_{t+1})\|^2 \\ &\quad + 6(1 - \gamma_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{z}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_{t+1})\|^2 \\ &\quad + 6(1 - \gamma_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_{t+1}; \bar{B}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_t; \bar{B}_{t+1})\|^2. \end{aligned}$$

From Assumption [B3.,](#page-3-10) we have

$$\begin{aligned} & \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_{t+1}) - \nabla_{\mathbf{y}} g_t(\mathbf{x}_t)\|^2 \\ & \leq 2\mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} g_t(\mathbf{x}_{t+1}, \mathbf{y}_t)\|^2 + 2\mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_{t+1}, \mathbf{y}_t) - \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & \leq 2\ell_{g,1}^2 \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\ell_{g,1}^2 \mathbb{E}\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\ & = 2\ell_{g,1}^2 \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\ell_{g,1}^2 \beta_t^2 \mathbb{E}\|\mathbf{d}_Y^t\|^2, \end{aligned}$$

and

$$\begin{aligned} & \mathbb{E}\|\nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1})\|^2 \\ & \leq 2\mathbb{E}\|\nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_t; \bar{\mathcal{B}}_{t+1})\|^2 \\ & + 2\mathbb{E}\|\nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_t; \bar{\mathcal{B}}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t; \bar{\mathcal{B}}_{t+1})\|^2 \\ & \leq 2\ell_{g,1}^2 \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\ell_{g,1}^2 \mathbb{E}\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\ & = 2\ell_{g,1}^2 \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\ell_{g,1}^2 \beta_t^2 \mathbb{E}\|\mathbf{d}_t^{\mathbf{y}}\|^2. \end{aligned}$$

From the two inequalities above, we have

$$\begin{aligned}\mathbb{E}\|e_{t+1}^g\|^2 &\leq (1 - \gamma_{t+1})^2 \mathbb{E}\|e_t^g\|^2 + 2\gamma_{t+1}^2 \frac{\sigma_y^2}{b} \\ &\quad + 6(1 - \gamma_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{z}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_{t+1})\|^2 \\ &\quad + 24(1 - \gamma_{t+1})^2 \ell_{g,1}^2 (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + \beta_t^2 \mathbb{E}\|\mathbf{d}_t^y\|^2).\end{aligned}$$

*994*

*996*

*1000 1001 1002*

*1003 1004 1005* Lemma C.3. *Suppose Assumptions [3.2,](#page-3-2) and* [B3.](#page-3-10) *hold. Then, for the sequence* {(xt, yt)} T <sup>t</sup>=1 *generated by Algorithm [1,](#page-2-1) we have*

*1014 where* e g <sup>t</sup> *defined in* [\(43\)](#page-16-3) *and* a > 0 *is a constant.*

*1019*

*1024* We have

*1026*

*1029*

*1034*

*1036*

*1039 1040*

*1041* To simplify the notation in the analysis, we introduce the definitions

*1042 1043 1044* Since e g t := d y <sup>t</sup> − ∇ygt(xt, yt), we have

$$\begin{aligned} \mathbb{E}\|e_{t+1}^g\|^2 &\leq (1 - \gamma_{t+1})^2 \mathbb{E}\|e_t^g\|^2 + 2\gamma_{t+1}^2 \frac{\sigma_y^2}{b} + 24(1 - \gamma_{t+1})^2 \ell_{g,1}^2 \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 \\ &\quad + 6(1 - \gamma_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{z}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_{t+1})\|^2 \\ &\quad + 48(1 - \gamma_{t+1})^2 \ell_{g,1}^2 \beta_t^2 \mathbb{E}\|e_t^g\|^2 + 48(1 - \gamma_{t+1})^2 \ell_{g,1}^2 \beta_t^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ &\leq (1 - \gamma_{t+1})^2 (1 + 48\ell_{g,1}^2 \beta_t^2) \mathbb{E}\|e_t^g\|^2 + 2\gamma_{t+1}^2 \frac{\sigma_y^2}{b} + 24(1 - \gamma_{t+1})^2 \ell_{g,1}^2 \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 \\ &\quad + 6(1 - \gamma_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{z}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{z}_{t+1})\|^2 \\ &\quad + 48(1 - \gamma_{t+1})^2 \ell_{g,1}^2 \beta_t^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2. \end{aligned}$$

$$\begin{aligned} \mathbb{E} [\|\mathbf{y}_{t+1} - \mathbf{y}_t^*(\mathbf{x}_t)\|^2] &\leq (1+a) \left( 1 - 2\beta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \right) \mathbb{E} [\|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\|^2] \\ &\quad + \left( -(1+a) \left( \frac{2\beta_t}{\mu_g + \ell_{g,1}} - \beta_t^2 \right) \right) \mathbb{E} [\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2] \\ &\quad + (1 + \frac{1}{a}) \beta_t^2 \mathbb{E} [\|e_t^g\|^2] , \end{aligned}$$

*Proof.* From Lemma [B.5,](#page-12-0) we have

$$\begin{aligned} \mathbb{E} [\|\mathbf{y}_{t+1} - \mathbf{y}_t^*(\mathbf{x}_t)\|^2] &= \mathbb{E} [\|\mathbf{y}_t - \beta_t \mathbf{d}_t^Y - \mathbf{y}_t^*(\mathbf{x}_t)\|^2] \\ &\leq (1 + a) \mathbb{E} [\|\mathbf{y}_t - \beta_t \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2] \\ &\quad + (1 + \frac{1}{a}) \beta_t^2 \mathbb{E} [\|\mathbf{d}_t^Y - \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2] . \end{aligned} \quad (45)$$

Next, we will bound the first term on the RHS of [\(45\)](#page-18-1).

$$\begin{aligned} \mathbb{E} [\|\mathbf{y}_t - \beta_t \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2] &= \mathbb{E} [\|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\|^2] + \beta_t^2 \mathbb{E} [\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2] \\ &\quad - 2\beta_t \mathbb{E} [\langle \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t), \mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t) \rangle] \\ &\leq \left( 1 - 2\beta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \right) \mathbb{E} [\|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\|^2] \\ &\quad - \left( \frac{2\beta_t}{\mu_g + \ell_{g,1}} - \beta_t^2 \right) \mathbb{E} [\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2], \end{aligned} \tag{46}$$

where the inequality results from the strong convexity of g<sup>t</sup> by Assumption [3.2,](#page-3-2) which implies

$$\langle \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t), \mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t) \rangle \geq \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 + \frac{1}{\mu_g + \ell_{g,1}} \|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2.$$

Substituting [\(46\)](#page-18-2) into [\(45\)](#page-18-1), gives the desired result.

$$\theta_t^{\mathbf{y}} := \|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\|^2, \quad \text{and} \quad \theta_t^{\mathbf{v}} := \|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2. \quad (47)$$

1045 1046 Lemma C.4. *Suppose Assumptions [3.2,](#page-3-2) and* [B2.](#page-3-12)*,* [B3.](#page-3-10) *hold. Let* θ y <sup>t</sup> *be defined as in* [\(47\)](#page-18-3)*. Then, for the sequence* {(xt, yt)} T <sup>t</sup>=1 *generated by Algorithm [1,](#page-2-1) the following bound is guaranteed:*

1047 1048 1049

1054

1056 *where* Lµ<sup>g</sup> = µgℓg,<sup>1</sup> µg+ℓg,<sup>1</sup> *,* L<sup>y</sup> = ℓg,<sup>1</sup> µ<sup>g</sup> *is defined as in* [\(42\)](#page-16-2)*;* H2,T *is defined in* [\(10\)](#page-3-0)*. Moreover,* e g t *is defined in* [\(43\)](#page-16-3)*.*

1059

1060 1061 *Proof.* From Lemma [B.5,](#page-12-0) we have for any c >´ 0

1067 1068 From Lemma [C.3,](#page-18-4) we have for any a > 0

1069

1074

1076

1079

1089 1090 Choose c´ = βtLµg /2 1−βtLµg and a = βtLµg 1−2βtLµg . Then, the following equations and inequalities are satisfied.

$$\begin{aligned} & \sum_{t=1}^T (\mathbb{E}[\theta_{t+1}^{\mathbf{y}}] - \mathbb{E}[\theta_t^{\mathbf{y}}]) \\ & \leq -\frac{L_{\mu_g}}{2} \sum_{t=1}^T \beta_t \mathbb{E}[\theta_t^{\mathbf{y}}] + \frac{2}{L_{\mu_g}} \sum_{t=1}^T \beta_t \mathbb{E}[\|e_t^g\|^2] + \frac{4L_{\mathbf{y}}^2}{L_{\mu_g}} \sum_{t=1}^T \frac{1}{\beta_t} \mathbb{E}\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 \\ & + \frac{4}{L_{\mu_g}} \sum_{t=2}^T \frac{1}{\beta_t} \sup_{\mathbf{x} \in \mathcal{X}} \mathbb{E}\|\mathbf{y}_{t-1}^*(\mathbf{x}) - \mathbf{y}_t^*(\mathbf{x})\|^2 + \sum_{t=1}^T \left( -\frac{2\beta_t}{\mu_g + \ell_{g,1}} + \beta_t^2 \right) \mathbb{E}[\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2], \end{aligned} \quad (48)$$

$$\begin{aligned} \mathbb{E} \left[ \|\mathbf{y}_{t+1} - \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1})\|^2 \right] &= \mathbb{E} \left[ \|\mathbf{y}_{t+1} - \mathbf{y}_t^*(\mathbf{x}_t) + \mathbf{y}_t^*(\mathbf{x}_t) - \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1})\|^2 \right] \\ &\leq (1 + \epsilon) \mathbb{E} \left[ \|\mathbf{y}_{t+1} - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 \right] \\ &\quad + \left( 1 + \frac{1}{\epsilon} \right) \mathbb{E} \left[ \|\mathbf{y}_{t+1}^*(\mathbf{x}_{t+1}) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 \right]. \end{aligned} \quad (49)$$

$$\begin{aligned} \mathbb{E} [\|\mathbf{y}_{t+1} - \mathbf{y}_t^*(\mathbf{x}_t)\|^2] &\leq (1+a) \left( 1 - 2\beta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \right) \mathbb{E} [\|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\|^2] \\ &\quad + \left( -(1+a) \left( \frac{2\beta_t}{\mu_g + \ell_{g,1}} - \beta_t^2 \right) \right) \mathbb{E} [\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2] \\ &\quad + \left( 1 + \frac{1}{a} \right) \beta_t^2 \mathbb{E} [\|e_t^g\|^2]. \end{aligned} \quad (50)$$

Substituting [\(50\)](#page-19-0) into [\(49\)](#page-19-1), we get

$$\begin{aligned} & \mathbb{E} \left[ \|\mathbf{y}_{t+1} - \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1})\|^2 \right] \\ & \leq (1 + \hat{\epsilon})(1 + a) \left( 1 - 2\beta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \right) \mathbb{E} \left[ \|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 \right] \\ & + \left( -(1 + \hat{\epsilon})(1 + a) \left( \frac{2\beta_t}{\mu_g + \ell_{g,1}} - \beta_t^2 \right) \right) \mathbb{E} \left[ \|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \right] \\ & + (1 + \hat{\epsilon})(1 + \frac{1}{a})\beta_t^2 \mathbb{E} \left[ \|e_t^g\|^2 \right] \\ & + \left( 1 + \frac{1}{\tilde{\epsilon}} \right) \mathbb{E} \left[ \|\mathbf{y}_{t+1}^*(\mathbf{x}_{t+1}) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 \right]. \end{aligned} \tag{51}$$

$$\begin{aligned} (1 + \hat{\epsilon})(1 + a) (1 - 2\beta_t L_{\mu_g}) &= 1 - \frac{\beta_t L_{\mu_g}}{2}, \\ (1 + a) (1 - 2\beta_t L_{\mu_g}) &= 1 - \beta_t L_{\mu_g}, \\ (1 + \hat{\epsilon}) (1 - \beta_t L_{\mu_g}) &= 1 - \frac{\beta_t L_{\mu_g}}{2}, \\ 1 + \frac{1}{a} &\leq \frac{1}{\beta_t L_{\mu_g}}, \quad 1 + \frac{1}{\hat{\epsilon}} \leq \frac{2}{\beta_t L_{\mu_g}}, \end{aligned} \tag{52}$$

1104

1106

1109

1111

1114

1116

1118 1119

1124

1126

1129

1134

1136

1139

1140 1141 where

1147 1148 Let u = [x; y; v]. Then, we have

1149

1151

where Lµ<sup>g</sup> = µgℓg,<sup>1</sup> µg+ℓg,<sup>1</sup> . Based on [\(51\)](#page-19-2) and [\(52\)](#page-19-3), we get

$$\begin{aligned} & \mathbb{E} [\|\mathbf{y}_{t+1} - \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1})\|^2] - \mathbb{E} [\|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\|^2] \\ & \leq -\frac{\beta_t L_{\mu_g}}{2} \mathbb{E} [\|\mathbf{y}_t - \mathbf{y}_t^*(\mathbf{x}_t)\|^2] + \left( -\left( \frac{2\beta_t}{\mu_g + \ell_{g,1}} - \beta_t^2 \right) \right) \mathbb{E} [\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2] \\ & + \frac{2}{\beta_t L_{\mu_g}} \beta_t^2 \mathbb{E} [\|e_t^g\|^2] + \frac{2}{\beta_t L_{\mu_g}} \mathbb{E} [\|\mathbf{y}_{t+1}^*(\mathbf{x}_{t+1}) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2]. \end{aligned} \quad (53)$$

Next, we upper-bound the last term of the above inequality.

$$\begin{aligned} & \mathbb{E} \left[ \|\mathbf{y}_{t+1}^*(\mathbf{x}_{t+1}) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 \right] \\ & \leq 2 \left( \mathbb{E} \left[ \|\mathbf{y}_{t+1}^*(\mathbf{x}_{t+1}) - \mathbf{y}_{t+1}^*(\mathbf{x}_t)\|^2 \right] + \mathbb{E} \left[ \|\mathbf{y}_{t+1}^*(\mathbf{x}_t) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 \right] \right) \\ & \leq 2 \left( L_{\mathbf{y}}^2 \mathbb{E} \left[ \|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 + \|\mathbf{y}_{t+1}^*(\mathbf{x}_t) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 \right] \right), \end{aligned} \tag{54}$$

where the second inequality is by Lemma [D.2.](#page-40-0)

Substituting [\(54\)](#page-20-1) into [\(53\)](#page-20-2) and summing over t ∈ [T], give the desired result.

# C.3. Bounds on the Linear System Solution

Lemma C.5. *Suppose Assumptions* [B2.](#page-3-12)*,* [B3.](#page-3-10)*,* [B4.](#page-3-13)*,* [C2.](#page-3-14) *and* [C4.](#page-3-15) *hold. Let* {(xt, yt, vt)} T <sup>t</sup>=1 *be generated according to Algorithm [1.](#page-2-1) For* e v <sup>t</sup>+1 *defined as*

$$e_t^{\vee} := \mathbf{d}_t^{\vee} - \nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t), \quad \text{where} \quad \nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t) := \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t) \mathbf{v}_t + \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t). \quad (55)$$

*we have:*

$$\begin{aligned} \mathbb{E}\|e_{t+1}^{\mathbf{y}}\|^2 &\leq (1 - \lambda_{t+1})^2(1 + 72\ell_{g,1}^2\delta_t^2)\mathbb{E}\|e_t^{\mathbf{y}}\|^2 + 4\lambda_{t+1}^2(\frac{\sigma_{gyy}^2}{b}p^2 + \frac{\sigma_{fy}^2}{b}) \\ &\quad + 12p^2(1 - \lambda_{t+1})^2\mathbb{E}\|\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &\quad + 12(1 - \lambda_{t+1})^2\mathbb{E}\|\nabla_{\mathbf{y}} f_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &\quad + 72(1 - \lambda_{t+1})^2(\ell_{g,2}^2 p^2 + \ell_{f,1}^2)(\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\beta_t^2 \mathbb{E}\|e_t^g\|^2 + 2\beta_t^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\ &\quad + 72(1 - \lambda_{t+1})^2\ell_{g,1}^4\delta_t^2\theta_t^{\mathbf{y}}, \end{aligned} \tag{56}$$

*for all* t ∈ [T] *and* θ v t *is defined in* [\(47\)](#page-18-3)*.*

*Proof.* Note that

$$e_{t+1}^{\mathbf{v}} := \mathbf{d}_{t+1}^{\mathbf{v}} - \nabla P_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{v}_{t+1}),$$

$$\nabla P_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{v}_{t+1}) := \nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) \mathbf{v}_{t+1} + \nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}).$$

From Algorithm [1,](#page-2-1) we have

$$\mathbf{d}_{t+1}^{\mathbf{v}} = \mathbf{d}_{t+1}^{\mathbf{v}\mathbf{v}}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \mathcal{B}_{t+1}) + (1 - \lambda_{t+1})(\mathbf{d}_t^{\mathbf{v}} - \mathbf{d}_{t+1}^{\mathbf{v}\mathbf{v}}(\mathbf{x}_t, \mathbf{y}_t; \mathcal{B}_{t+1})).$$

$$\begin{aligned}\mathbb{E}\|e_{t+1}^{\mathbf{v}}\|^2 &= \mathbb{E}\|\mathbf{d}_{t+1}^{\mathbf{v}} - \nabla P_{t+1}(\mathbf{u}_{t+1})\|^2 \\ &= \mathbb{E}\|\nabla P_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) + (1 - \lambda_{t+1})(\mathbf{d}_t^{\mathbf{v}} - \nabla P_{t+1}(\mathbf{u}_t; \mathcal{B}_{t+1})) - \nabla P_{t+1}(\mathbf{u}_{t+1})\|^2 \\ &= \mathbb{E}\|(1 - \lambda_{t+1})e_t^{\mathbf{v}} + \nabla P_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) - \nabla P_{t+1}(\mathbf{u}_{t+1}) \\ &\quad - (1 - \lambda_{t+1})(\nabla P_{t+1}(\mathbf{u}_t; \mathcal{B}_{t+1}) - \nabla P_t(\mathbf{u}_t))\|^2,\end{aligned}$$

$$\begin{aligned}
& 1156 \\
& 1157 \quad \mathbb{E}\|e_{t+1}^{\mathbf{y}}\|^2 \\
& 1158 \quad = (1 - \lambda_{t+1})^2 \mathbb{E}\|e_t^{\mathbf{y}}\|^2 + \mathbb{E}\|\lambda_{t+1} (\nabla P_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) - \nabla P_{t+1}(\mathbf{u}_{t+1})) \\
& 1159 \quad - (1 - \lambda_{t+1}) (\nabla P_{t+1}(\mathbf{u}_t; \mathcal{B}_{t+1}) - \nabla P_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) + \nabla P_{t+1}(\mathbf{u}_{t+1}) - \nabla P_t(\mathbf{u}_t)) \|^2 \\
& 1160 \quad \leq (1 - \lambda_{t+1})^2 \mathbb{E}\|e_t^{\mathbf{y}}\|^2 + 2\lambda_{t+1}^2 \mathbb{E}\|\nabla P_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) - \nabla P_{t+1}(\mathbf{u}_{t+1})\|^2 \\
& 1161 \quad + 2(1 - \lambda_{t+1})^2 \mathbb{E}\|\nabla P_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) - \nabla P_{t+1}(\mathbf{u}_{t+1}) - \nabla P_{t+1}(\mathbf{u}_t; \mathcal{B}_{t+1}) + \nabla P_t(\mathbf{u}_t)\|^2,
\end{aligned}$$

$$\begin{aligned} 1166 \quad & \mathbb{E}\|\nabla P_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) - \nabla P_{t+1}(\mathbf{u}_{t+1})\|^2 \\ 1167 \quad & = \mathbb{E}\|(\nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})) \mathbf{v}_{t+1} \\ 1168 \quad & + \nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \mathcal{B}_{t+1}) - \nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ 1169 \quad & \leq 2\mathbb{E}\|(\nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})) \mathbf{v}_{t+1}\|^2 \\ 1170 \quad & \leq 2\mathbb{E}\|\nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \mathcal{B}_{t+1}) - \nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ 1171 \quad & + 2\mathbb{E}\|\nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \mathcal{B}_{t+1}) - \nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ 1172 \quad & \leq 2(\frac{\sigma^2}{b} p^2 + \frac{\sigma^2}{b}), \\ 1173 \quad & \leq 2(\frac{\sigma^2}{b} p^2 + \frac{\sigma^2}{b}), \\ 1174 \quad & \\ 1175 \quad & \end{aligned}$$

1176

$$\begin{aligned} & 1187 & \mathbb{E}\|\nabla P_t(\mathbf{u}_{t+1}) - \nabla P_t(\mathbf{u}_t)\|^2 \\ & 1188 & \leq 3\mathbb{E}\|\nabla P_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{v}_{t+1}) - \nabla P_t(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{v}_{t+1})\|^2 \\ & 1189 & + 3\mathbb{E}\|\nabla P_t(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{v}_{t+1}) - \nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_{t+1})\|^2 \\ & 1190 & + 3\mathbb{E}\|\nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_{t+1}) - \nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2 \\ & 1191 & + 3\mathbb{E}\|\nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_{t+1}) - \nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2 \\ & 1192 & \leq 3\mathbb{E}\|(\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_{t+1}))\mathbf{v}_{t+1} + \nabla_{\mathbf{y}} f_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_{t+1})\|^2 \\ & 1193 & + 3\mathbb{E}\|(\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t))\mathbf{v}_{t+1} + \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & 1194 & + 3\mathbb{E}\|(\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t))\mathbf{v}_{t+1} + \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & 1195 & + 3\mathbb{E}\|\nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_{t+1}) - \nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2 \\ & 1196 & \leq 6(\ell_{g,2}^2 \mathbb{E}\|\mathbf{v}_{t+1}\|^2 + \ell_{f,1}^2) (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + \mathbb{E}\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2) + 3\ell_{g,1}^2 \mathbb{E}\|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 \\ & 1197 & \leq 6(\ell_{g,2}^2 p^2 + \ell_{f,1}^2) (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + \beta_t^2 \mathbb{E}\|\mathbf{d}_t^Y\|^2) + 3\ell_{g,1}^2 \delta_t^2 \mathbb{E}\|\mathbf{d}_t^Y\|^2 \\ & 1198 & \leq 6(\ell_{g,2}^2 p^2 + \ell_{f,1}^2) (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + \beta_t^2 \mathbb{E}\|\mathbf{e}_t^g\|^2 + 2\beta_t^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\ & 1199 & \leq 6(\ell_{g,2}^2 p^2 + \ell_{f,1}^2) (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\beta_t^2 \mathbb{E}\|\mathbf{e}_t^g\|^2 + 2\beta_t^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\ & 1200 & + 6\ell_{g,1}^2 \delta_t^2 (\mathbb{E}\|\mathbf{e}_t^Y\|^2 + \mathbb{E}\|\nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2) \\ & 1201 & \leq 6(\ell_{g,2}^2 p^2 + \ell_{f,1}^2) (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\beta_t^2 \mathbb{E}\|\mathbf{e}_t^g\|^2 + 2\beta_t^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\ & 1202 & \leq 6(\ell_{g,2}^2 p^2 + \ell_{f,1}^2) (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\beta_t^2 \mathbb{E}\|\mathbf{e}_t^g\|^2 + 2\beta_t^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\ & 1203 & + 6\ell_{g,1}^2 \delta_t^2 (\mathbb{E}\|\mathbf{e}_t^Y\|^2 + \ell_{g,1}^2 \mathbb{E}\|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_$$

1206

which implies that

where the inequality follows from Cauchy–Schwartz inequality.

For the first term, from Assumptions [C2.](#page-3-14) and [C4.,](#page-3-15) we have

where the last inequality follows from [\(8\)](#page-3-8).

Then, from the above inequality and ∥a + b + c∥ <sup>2</sup> ≤ 3(∥a∥ <sup>2</sup> + ∥b∥ <sup>2</sup> + ∥c∥ ), we have

$$\begin{aligned} \mathbb{E}\|e_{t+1}^{\mathbf{v}}\|^2 &\leq (1 - \lambda_{t+1})^2 \mathbb{E}\|e_t^{\mathbf{v}}\|^2 + 4\lambda_{t+1}^2 \left(\frac{\sigma_{gy}^2}{b} p^2 + \frac{\sigma_{fy}^2}{b}\right) \\ &\quad + 6(1 - \lambda_{t+1})^2 \mathbb{E}\|\nabla P_t(\mathbf{u}_t) - \nabla P_t(\mathbf{u}_{t+1})\|^2 \\ &\quad + 6(1 - \lambda_{t+1})^2 \mathbb{E}\|\nabla P_t(\mathbf{u}_{t+1}) - \nabla P_{t+1}(\mathbf{u}_{t+1})\|^2 \\ &\quad + 6(1 - \lambda_{t+1})^2 \mathbb{E}\|\nabla P_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) - \nabla P_{t+1}(\mathbf{u}_t; \mathcal{B}_{t+1})\|^2. \end{aligned} \quad (57)$$

Moreover, from ∥a + b + c∥ <sup>2</sup> ≤ 3(∥a∥ <sup>2</sup> + ∥b∥ <sup>2</sup> + ∥c∥ 2 ), we have

where the third inequality follows from Assumptions [B2.,](#page-3-12) [B3.](#page-3-10) and [B4.;](#page-3-13) the last inequality follows from [\(62\)](#page-23-0).

1216

1218 1219

1224

1226

1229

1234

1236

1254

1256

1259 1260

Similarly, we have

$$\begin{aligned} & \mathbb{E} \|\nabla P_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) - \nabla P_{t+1}(\mathbf{u}_t; \mathcal{B}_{t+1})\|^2 \\ & \leq 6(\ell_{g,2}^2 p^2 + \ell_{j,1}^2) (\mathbb{E} \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\beta_t^2 \mathbb{E} \|e_t^i\|^2 + 2\beta_t^2 \mathbb{E} \|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\ & + 6\ell_{g,1}^2 \delta_t^2 (\mathbb{E} \|e_Y\|^2 + \ell_{g,1}^2 \mathbb{E} \|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2). \end{aligned} \quad (59)$$

Substituting [\(59\)](#page-22-0) and [\(58\)](#page-21-0) into [\(57\)](#page-21-1), we have

$$\begin{aligned} \mathbb{E}\|e_{t+1}^{\mathbf{v}}\|^2 &\leq (1 - \lambda_{t+1})^2(1 + 72\ell_{g,1}^2\delta_t^2)\mathbb{E}\|e_{\mathbf{v}}^{\mathbf{v}}\|^2 + 4\lambda_{t+1}^2\left(\frac{\sigma_{g\mathbf{y}\mathbf{y}}^2}{b}p^2 + \frac{\sigma_{\mathbf{f}\mathbf{y}}^2}{b}\right) \\ &\quad + 6(1 - \lambda_{t+1})^2\mathbb{E}\|\nabla P_t(\mathbf{u}_{t+1}) - \nabla P_{t+1}(\mathbf{u}_{t+1})\|^2 \\ &\quad + 72(1 - \lambda_{t+1})^2(\ell_{g,2}^2p^2 + \ell_{f,1}^2)(\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\beta_t^2\mathbb{E}\|e_t^g\|^2 + 2\beta_t^2\mathbb{E}\|\nabla_{\mathbf{y}}g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\ &\quad + 72(1 - \lambda_{t+1})^2\ell_{g,1}^4\delta_t^2\mathbb{E}\|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2. \end{aligned}$$

From ∥a + b∥ <sup>2</sup> ≤ 2∥a∥ <sup>2</sup> + 2∥b∥ 2 and [\(8\)](#page-3-8), we have

$$\begin{aligned} \mathbb{E}\|\nabla P_t(\mathbf{u}_{t+1}) - \nabla P_{t+1}(\mathbf{u}_{t+1})\|^2 &= \mathbb{E}\|\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) \mathbf{v}_{t+1} - \nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) \mathbf{v}_{t+1} \\ &\quad + \nabla_{\mathbf{y}} f_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &\leq 2\mathbb{E}\|(\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})) \mathbf{v}_{t+1}\|^2 \\ &\quad + 2\mathbb{E}\|\nabla_{\mathbf{y}} f_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &\leq 2\mathbb{E}\|\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 p^2 \\ &\quad + 2\mathbb{E}\|\nabla_{\mathbf{y}} f_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2. \end{aligned}$$

This completes the proof.

As demonstrated in Lemma [C.5,](#page-20-0) the gradient estimation error e v <sup>t</sup>+1 for the linear system consists of four key components: (1) an iteratively refined error term (1 − λt+1) 2 (1 + 72ℓ 2 g,1 δ 2 t )E∥e v t ∥ 2 , which depends on the stepsize δt; (2) the error arising from the variation in the Hessian of the lower-level objectiv; (3) the error resulting from the variation in the gradient of the upper-level objective, and (4) an approximation error term of order O(δ 2 t θ v t ) associated with solving the linear system.

Lemma C.6. *Suppose Assumptions [3.2](#page-3-2) and [3.3](#page-3-3) hold. Then, for the sequence* {(xt, yt, vt)} T <sup>t</sup>=1 *generated by Algorithm [1,](#page-2-1) we have*

$$\begin{aligned} \mathbb{E}\|\mathbf{v}_{t+1} - \mathbf{v}_t^*(\mathbf{x}_t)\|^2 &\leq (1 + \epsilon) \left( 1 - 2\delta_t \frac{(\ell_{g,1} + \ell_{g,1}^3)\mu_g}{\mu_g + \ell_{g,1}} + \delta_t^2 \ell_{g,1}^2 \right) \mathbb{E}\|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2 \\ &\quad + (1 + \frac{1}{\epsilon})\delta_t^2 \mathbb{E}\|e_t^{\mathbf{v}}\|^2, \end{aligned}$$

*where* e v <sup>t</sup> *defined in* [\(55\)](#page-20-3) *and for any* c >´ 0*.*

*Proof.* From the update rules in Algorithm [1,](#page-2-1) we have the following:

$$\begin{aligned} \mathbb{E}\|\mathbf{v}_{t+1} - \mathbf{v}_t^*(\mathbf{x}_t)\|^2 &= \mathbb{E}\|\mathbf{v}_t - \delta_t \mathbf{d}_t^\mathbf{v} - \mathbf{v}_t^*(\mathbf{x}_t)\|^2 \\ &\leq (1 + \hat{c})\mathbb{E}\|\mathbf{v}_t - \delta_t \nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t) - \mathbf{v}_t^*(\mathbf{x}_t)\|^2 \\ &\quad + (1 + \frac{1}{\hat{c}})\delta_t^2 \mathbb{E}\|\mathbf{d}_t^\mathbf{v} - \nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2, \end{aligned} \quad (60)$$

1269

1274

1276

1279

1289 1290

1294

1296

1306 1307

1309

1314

1316

For the first term of the above eq. [\(60\)](#page-22-1), we have

$$\begin{aligned} & \mathbb{E}\|\mathbf{v}_t - \delta_t \nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t) - \mathbf{v}_t^*(\mathbf{x}_t)\|^2 \\ &= \mathbb{E}\|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2 - 2\delta_t \mathbb{E}\langle \mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t), \nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t) \rangle + \delta_t^2 \mathbb{E}\|\nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2 \\ &\leq \left(1 - 2\delta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}}\right) \mathbb{E}\|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2 - (2\delta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} - \delta_t^2) \mathbb{E}\|\nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2 \\ &\leq \left(1 - 2\delta_t \frac{(\ell_{g,1} + \ell_{g,1}^3)\mu_g}{\mu_g + \ell_{g,1}} + \delta_t^2 \ell_{g,1}^2\right) \mathbb{E}\|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2, \end{aligned} \tag{61}$$

where the first inequality follows from the strong convexity of P<sup>t</sup> function (in eq. [\(4b\)](#page-2-4)) that

$$\mathbb{E}\langle \mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t), \nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t) \rangle \geq \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \mathbb{E}\|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2 + \frac{1}{\mu_g + \ell_{g,1}} \mathbb{E}\|\nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2.$$

The second inequality is derived from the following inequality.

$$\begin{aligned} \mathbb{E}\|\nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2 &= \mathbb{E}\|\nabla_{\mathbf{y}}^2 P_t(\mathbf{x}_t, \mathbf{y}_t) \mathbf{v}_t + \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ &= \mathbb{E}\|\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) (\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t))\|^2 \leq \ell_{g,1}^2 \mathbb{E}\|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2. \end{aligned} \quad (62)$$

Combining [\(60\)](#page-22-1) and [\(61\)](#page-23-1), we get the desired result.

Lemma C.7. *Suppose Assumptions [3.2](#page-3-2) and [3.3](#page-3-3) hold. Then, we have*

$$\|\mathbf{v}_t^*(\mathbf{x}_t) - \mathbf{v}_{t+1}^*(\mathbf{x}_{t+1})\|^2 \leq 2 \frac{\nu^2}{\mu_g^2} \left( \|\mathbf{y}_{t+1}^*(\mathbf{x}_{t+1}) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 + \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 \right),$$

*where* ν := ℓf,<sup>1</sup> + ℓg,2ℓf,<sup>0</sup> µ<sup>g</sup> *, and* v ∗ t (x) *is a solution of Subproblem* [\(4b\)](#page-2-4)*.*

*Proof.* Based on [\(4b\)](#page-2-4), we have that

$$\begin{aligned} & \left\| \mathbf{v}_t^*(\mathbf{x}_t) - \mathbf{v}_{t+1}^*(\mathbf{x}_{t+1}) \right\|^2 \\ &= \left\| (\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))^{-1} \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) \right. \\ & \quad \left. - (\nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1})))^{-1} \nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1})) \right\|^2 \\ & \leq 2 \left\| \left( (\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))^{-1} - (\nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1})))^{-1} \right) \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) \right\|^2 \\ & + 2 \left\| (\nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1})))^{-1} (\nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - \nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1}))) \right\|^2. \end{aligned} \quad (63a)$$

In the following steps, we bound the terms [\(63a\)](#page-23-2) and [\(63b\)](#page-23-3), respectively.

For [\(63a\)](#page-23-2), we have:

$$\begin{aligned} & \left\| (\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))^{-1} - (\nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1})))^{-1} \right\|^2 \\ &= \left\| (\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))^{-1} (\nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1}))) \right. \\ &\quad - \nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) (\nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1})))^{-1} \left\|^2 \right. \\ &\leq \frac{1}{\mu_g^2} \left\| \nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - \nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1})) \right\|^2 \\ &\leq \frac{\ell_{g,2}}{\mu_g^2} \left\| (\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - (\mathbf{x}_{t+1}, \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1})) \right\|^2 \\ &\leq \frac{\ell_{g,2}}{\mu_g^2} \left( \left\| \mathbf{y}_t^*(\mathbf{x}_t) - \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1}) \right\|^2 + \left\| \mathbf{x}_t - \mathbf{x}_{t+1} \right\|^2 \right), \end{aligned} \tag{64}$$

1326

1329

1334

1336

1339

1340 By raising both sides of the above inequality to the power 2 and using (a + b) <sup>2</sup> ≤ 2a <sup>2</sup> + 2b 2 , we complete the proof.

1354

1356

1369

where the first equality holds since for any invertible matrix A and B we have ∥A−<sup>1</sup> − B−<sup>1</sup>∥ = ∥A−<sup>1</sup> (B − A)B−<sup>1</sup>∥, and the second inequality is obtained from Assumption [3.3.](#page-3-3)

Thus, from [\(64\)](#page-23-4) and Assumption [3.3,](#page-3-3) we get

$$(63a) \leq \frac{\ell_{f,0}\ell_{g,2}}{\mu_g^2} \left( \|\mathbf{y}_t^*(\mathbf{x}_t) - \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1})\|^2 + \|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 \right). \quad (65)$$

For [\(63b\)](#page-23-3), we have

$$\begin{aligned}
 (63b) &\leq \frac{1}{\mu_g} \|\nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - \nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1}))\|^2 \\
 &\leq \frac{\ell_{f,1}}{\mu_g} \|(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - (\mathbf{x}_{t+1}, \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1}))\|^2 \\
 &\leq \frac{\ell_{f,1}}{\mu_g} (\|\mathbf{y}_{t+1}^*(\mathbf{x}_{t+1}) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 + \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2). \tag{66}
 \end{aligned}$$

Combining [\(65\)](#page-24-1) and [\(66\)](#page-24-2), we have

$$\|\mathbf{v}_t^*(\mathbf{x}_t) - \mathbf{v}_{t+1}^*(\mathbf{x}_{t+1})\|^2 \leq \frac{1}{\mu_g} \left( \frac{\ell_{f,0}\ell_{g,2}}{\mu_g} + \ell_{f,1} \right) \left( \|\mathbf{y}_{t+1}^*(\mathbf{x}_{t+1}) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 + \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 \right).$$

Lemma C.8. *Suppose Assumptions [3.2](#page-3-2) and [3.3](#page-3-3) hold. Let* θ v <sup>t</sup> *be defined in* [\(47\)](#page-18-3)*. Then, for any positive choice of step sizes as*

$$\delta_t \leq \frac{L_{\mu_g}}{\ell_{g,1}^2}, \quad \text{where} \quad L_{\mu_g} = \frac{(\ell_{g,1} + \ell_{g,1}^3)\mu_g}{(\mu_g + \ell_{g,1})},$$

*for all* t ∈ [T]*, the sequence* {vt} T <sup>t</sup>=1 *generated by Algorithm [1](#page-2-1) satisfy*

$$\begin{aligned} \sum_{t=1}^T (\mathbb{E}[\theta_{t+1}^{\mathbf{v}}] - \mathbb{E}[\theta_t^{\mathbf{v}}]) &\leq -\frac{\delta_t L_{\mu_g}}{4} \sum_{t=1}^T \mathbb{E}[\theta_t^{\mathbf{v}}] + \frac{4}{L_{\mu_g}} \delta_t \sum_{t=1}^T \mathbb{E}\|e_t^{\mathbf{v}}\|^2 \\ &\quad + \frac{16\nu^2}{L_{\mu_g}\mu_g^2\delta_t} \sum_{t=1}^T \mathbb{E}\|\mathbf{y}_{t+1}^*(\mathbf{x}_t) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 \\ &\quad + \frac{8\nu^2}{L_{\mu_g}\mu_g^2\delta_t} (1 + 2L_{\mathbf{y}}^2) \sum_{t=1}^T \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2, \end{aligned} \tag{67}$$

*where* e v t *is defined in* [\(55\)](#page-20-3)*.*

*Proof.* By Lemma [B.5,](#page-12-0) for any a > 0, we have

$$\begin{aligned}\mathbb{E} \left\| \mathbf{v}_{t+1} - \mathbf{v}_{t+1}^*(\mathbf{x}_{t+1}) \right\|^2 &= \mathbb{E} \left\| \mathbf{v}_{t+1} - \mathbf{v}_t^*(\mathbf{x}_t) + \mathbf{v}_t^*(\mathbf{x}_t) - \mathbf{v}_{t+1}^*(\mathbf{x}_{t+1}) \right\|^2 \\ &\leq (1+a) \mathbb{E} \left\| \mathbf{v}_{t+1} - \mathbf{v}_t^*(\mathbf{x}_t) \right\|^2 \\ &\quad + \left( 1 + \frac{1}{a} \right) \mathbb{E} \left\| \mathbf{v}_{t+1}^*(\mathbf{x}_{t+1}) - \mathbf{v}_t^*(\mathbf{x}_t) \right\|^2.\end{aligned}\tag{68}$$

From Lemma [C.6,](#page-22-2) we have for any c >´ 0:

$$\begin{aligned} \mathbb{E}\|\mathbf{v}_{t+1} - \mathbf{v}_t^*(\mathbf{x}_t)\|^2 &\leq (1 + \epsilon) \left( 1 - 2\delta_t \frac{(\ell_{g,1} + \ell_{g,1}^3)\mu_g}{\mu_g + \ell_{g,1}} + \delta_t^2 \ell_{g,1}^2 \right) \mathbb{E}\|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2 \\ &\quad + (1 + \frac{1}{\epsilon})\delta_t^2 \mathbb{E}\|e_t^Y\|^2. \end{aligned} \quad (69)$$

1379

1389 1390

1394

1396

1427 1428 1429 where Lµ<sup>g</sup> := (ℓg,1+ℓ g,1 )µ<sup>g</sup> µg+ℓg,<sup>1</sup> .

Substituting [\(69\)](#page-24-3) into [\(68\)](#page-24-4), we get

$$\begin{aligned} \mathbb{E} \|\mathbf{v}_{t+1} - \mathbf{v}_{t+1}^*(\mathbf{x}_{t+1})\|^2 &\leq (1+a)(1+\hat{c}) \left( 1 - 2\delta_t \frac{(\ell_{g,1} + \ell_{g,1}^3)\mu_g}{\mu_g + \ell_{g,1}} + \delta_t^2 \ell_{g,1}^2 \right) \mathbb{E} \|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2 \\ &\quad + (1+a) \left(1 + \frac{1}{\hat{c}}\right) \delta_t^2 \mathbb{E} \|\mathbf{e}_t^Y\|^2 \\ &\quad + \left(1 + \frac{1}{a}\right) \mathbb{E} \|\mathbf{v}_{t+1}^*(\mathbf{x}_{t+1}) - \mathbf{v}_t^*(\mathbf{x}_t)\|^2. \end{aligned} \quad (70)$$

In the following, we provide a bound for the third term on the right-hand side of [\(70\)](#page-25-0). To this end, we have from Lemma [C.7:](#page-23-5)

$$\begin{aligned} \mathbb{E} \left\| \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1}) - \mathbf{y}_t^*(\mathbf{x}_t) \right\|^2 &\leq 2 \frac{\nu^2}{\mu_g^2} \left( \mathbb{E} \left\| \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1}) - \mathbf{y}_t^*(\mathbf{x}_t) \right\|^2 + \mathbb{E} \left\| \mathbf{x}_{t+1} - \mathbf{x}_t \right\|^2 \right) \\ &\leq 2 \frac{\nu^2}{\mu_g^2} \left( 2 \mathbb{E} \left\| \mathbf{y}_{t+1}^*(\mathbf{x}_{t+1}) - \mathbf{y}_{t+1}^*(\mathbf{x}_t) \right\|^2 \right. \\ &\quad \left. + 2 \mathbb{E} \left\| \mathbf{y}_{t+1}^*(\mathbf{x}_t) - \mathbf{y}_t^*(\mathbf{x}_t) \right\|^2 + \mathbb{E} \left\| \mathbf{x}_{t+1} - \mathbf{x}_t \right\|^2 \right) \\ &\leq 2 \frac{\nu^2}{\mu_g^2} \left( (1 + 2L_{\mathbf{y}}^2) \mathbb{E} \left\| \mathbf{x}_{t+1} - \mathbf{x}_t \right\|^2 + 2 \mathbb{E} \left\| \mathbf{y}_{t+1}^*(\mathbf{x}_t) - \mathbf{y}_t^*(\mathbf{x}_t) \right\|^2 \right), \end{aligned}$$

where the last inequality follows from Lemma [C.1.](#page-15-6)

Combining this result with [\(70\)](#page-25-0) gives

$$\begin{aligned} \mathbb{E} \left\| \mathbf{v}_{t+1} - \mathbf{v}_{t+1}^*(\mathbf{x}_{t+1}) \right\|^2 &\leq (1+a)(1+\hat{c}) \left( 1 - 2\delta_t \frac{(\ell_{g,1} + \ell_{g,1}^3)\mu_g}{\mu_g + \ell_{g,1}} + \delta_t^2 \ell_{g,1}^2 \right) \mathbb{E} \|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2 \\ &\quad + (1+a)(1 + \frac{1}{\hat{c}})\delta_t^2 \mathbb{E} \|e_t^{\mathbf{y}}\|^2 + 4 \left( 1 + \frac{1}{a} \right) \frac{\nu^2}{\mu_g^2} \mathbb{E} \|\mathbf{y}_{t+1}^*(\mathbf{x}_t) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 \\ &\quad + 2 \left( 1 + \frac{1}{a} \right) \frac{\nu^2}{\mu_g^2} (1 + 2L_{\mathbf{y}}^2) \mathbb{E} \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2. \end{aligned} \quad (71)$$

Let Lµ<sup>g</sup> := (ℓg,1+ℓ g,1 )µ<sup>g</sup> µg+ℓg,<sup>1</sup> , then we have

$$1 - 2\delta_t \frac{(\ell_{g,1} + \ell_{g,1}^3)\mu_g}{\mu_g + \ell_{g,1}} + \delta_t^2 \ell_{g,1}^2 = 1 - 2\delta_t L_{\mu_g} + \delta_t^2 \ell_{g,1}^2 \leq 1 - \delta_t L_{\mu_g}, \quad (72)$$

where the last inequality follows from δ<sup>t</sup> ≤ Lµg ℓ .

g,1 Choose a = δtLµg /4 1− δtLµg and c´ = δtLµg /2 1−δtLµg . Then, from [\(72\)](#page-25-1), we have

$$\begin{aligned} (1+a)(1+\hat{c}) \left( 1 - 2\delta_t \frac{(\ell_{g,1} + \ell_{g,1}^3)\mu_g}{\mu_g + \ell_{g,1}} + \delta_t^2 \ell_{g,1}^2 \right) \\ \leq (1+a)(1+\hat{c}) (1 - \delta_t L_{\mu_g}) = 1 - \frac{\delta_t L_{\mu_g}}{4}, \\ (1+a) \left( 1 + \frac{1}{\hat{c}} \right) \leq \frac{4}{\delta_t L_{\mu_g}}, \\ 1 + \frac{1}{\hat{c}} \leq \frac{2}{\delta_t L_{\mu_g}}, \quad 1 + \frac{1}{a} \leq \frac{4}{\delta_t L_{\mu_g}}, \end{aligned} \tag{73}$$

#### 1441 1442 C.4. Bounds on the Outer Objective and its Projected Gradient

1447 1448 *we have:*

1469 where

1475 1476 where d xx <sup>t</sup>+1 (xt+1, yt+1; Bt+1) = ∇xft+1(xt+1, yt+1; Bt+1) + ∇<sup>2</sup> xygt+1 (xt+1, yt+1; Bt+1) vt+1.

Thus, from [\(71\)](#page-25-2) and [\(73\)](#page-25-3) we have

$$\begin{aligned}\mathbb{E} \left\| \mathbf{v}_{t+1} - \mathbf{v}_{t+1}^*(\mathbf{x}_{t+1}) \right\|^2 &\leq \left( 1 - \frac{\delta_t L_{\mu_g}}{4} \right) \mathbb{E} \left\| \mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t) \right\|^2 \\ &\quad + \frac{4}{L_{\mu_g}} \delta_t \mathbb{E} \|e_t^{\mathbf{y}}\|^2 + \frac{16\nu^2}{L_{\mu_g} \mu_g^2 \delta_t} \mathbb{E} \left\| \mathbf{y}_{t+1}^*(\mathbf{x}_t) - \mathbf{y}_t^*(\mathbf{x}_t) \right\|^2 \\ &\quad + \frac{8\nu^2}{L_{\mu_g} \mu_g^2 \delta_t} (1 + 2L_{\mathbf{y}}^2) \mathbb{E} \left\| \mathbf{x}_{t+1} - \mathbf{x}_t \right\|^2.\end{aligned}$$

Rearranging the terms and summing from t = 1 to T, gives the desired result.

Lemma C.9. *Suppose Assumptions* [B2.](#page-3-12)*,* [B3.](#page-3-10)*,* [C3.](#page-3-16) *and* [C5.](#page-3-17) *hold. Let* {(xt, yt, vt)} T <sup>t</sup>=1 *be generated according to Algorithm [1.](#page-2-1) For* e f <sup>t</sup> *defined as*

$$e_t^f := \mathbf{d}_t^x - \tilde{\mathbf{d}}_t(\mathbf{z}_t), \quad \text{where} \quad \tilde{\mathbf{d}}_t(\mathbf{z}_t) = \nabla_{\mathbf{x}} f_t(\mathbf{z}_t) + \nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{z}_t) \mathbf{v}_t, \quad (74)$$

$$\begin{aligned} \mathbb{E}\|e_{t+1}^f\|^2 &\leq (1 - \eta_{t+1})^2 \mathbb{E}\|e_t^f\|^2 + 4\eta_{t+1}^2 \left(\frac{\sigma_{g_{\mathbf{xy}}}^2}{b} p^2 + \frac{\sigma_{f_{\mathbf{x}}}^2}{b}\right) \\ &\quad + 12p^2(1 - \eta_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{xy}}^2 g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{xy}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &\quad + 12(1 - \eta_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{x}} f_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{x}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &\quad + 72(1 - \eta_{t+1})^2 (\ell_{g,2}^2 p^2 + \ell_{f,1}^2) (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\beta_t^2 \mathbb{E}\|e_t^g\|^2 + 2\beta_t^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\ &\quad + 72\ell_{g,1}^2 (1 - \eta_{t+1})^2 \delta_t^2 \mathbb{E}\|e_t^{\mathbf{y}}\|^2 + 72(1 - \eta_{t+1})^2 \ell_{g,1}^4 \delta_t^2 \mathbb{E}[\theta_t^{\mathbf{y}}], \end{aligned} \tag{75}$$

*for all* t ∈ [T]*, and* θ v <sup>t</sup> *are defined in* [\(47\)](#page-18-3)*.*

*Proof.* Note that

$$e_{t+1}^f = \mathbf{d}_{t+1}^x - \tilde{\mathbf{d}}_{t+1} (\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{v}_{t+1}),$$

$$\tilde{\mathbf{d}}_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{v}_{t+1}) = \nabla_{\mathbf{x}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) + \nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) \mathbf{v}_{t+1}. \quad (76)$$

From Algorithm [1,](#page-2-1) we have

$$\mathbf{d}_{t+1}^x = \mathbf{d}_{t+1}^{xx}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \mathcal{B}_{t+1}) + (1 - \eta_{t+1})(\mathbf{d}_{t+1}^x - \mathbf{d}_{t+1}^{xx}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \mathcal{B}_{t+1})),$$

Let u = [x; y; v]. Then, we have

$$\begin{aligned}\mathbb{E}\|e_{t+1}^f\|^2 &= \mathbb{E}\|\mathbf{d}_{t+1}^x - \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1})\|^2 \\ &= \mathbb{E}\|\tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) + (1 - \eta_{t+1})(\mathbf{d}_t^x - \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_t; \mathcal{B}_{t+1})) - \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1})\|^2 \\ &= \mathbb{E}\|(1 - \eta_{t+1})e_t^f + \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) - \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1}) \\ &\quad - (1 - \eta_{t+1})(\tilde{\mathbf{d}}_{t+1}(\mathbf{u}_t; \mathcal{B}_{t+1}) - \tilde{\mathbf{d}}_t(\mathbf{u}_t))\|^2,\end{aligned}$$

$$\begin{aligned} &1486 \\ &1487 \quad \mathbb{E}\|e_{t+1}^f\|^2 \\ &1488 \quad = (1 - \eta_{t+1})^2 \mathbb{E}\|e_t^f\|^2 + \mathbb{E}\|\eta_{t+1}(\tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) - \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1})) \\ &1489 \quad - (1 - \eta_{t+1})(\tilde{\mathbf{d}}_{t+1}(\mathbf{u}_t; \mathcal{B}_{t+1}) - \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) + \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1}) - \tilde{\mathbf{d}}_t(\mathbf{u}_t))\|^2 \\ &1490 \quad \leq (1 - \eta_{t+1})^2 \mathbb{E}\|e_t^f\|^2 + 2\eta_{t+1}^2 \mathbb{E}\|\tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) - \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1})\|^2 \\ &1492 \quad + 2(1 - \eta_{t+1})^2 \mathbb{E}\|\tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) - \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1}) - \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_t; \mathcal{B}_{t+1}) + \tilde{\mathbf{d}}_t(\mathbf{u}_t)\|^2, & (77) \\ &1493 \end{aligned}$$

1504

1506

1509

1513 1514 1515

1516 Moreover, from ∥a + b + c∥ <sup>2</sup> ≤ 3(∥a∥ <sup>2</sup> + ∥b∥ <sup>2</sup> + ∥c∥ 2 ), we have

$$\begin{aligned}
& 1518 & \mathbb{E}\|\tilde{\mathbf{d}}_t(\mathbf{u}_{t+1}) - \tilde{\mathbf{d}}_t(\mathbf{u}_t)\|^2 \\
& 1519 & \leq 3\mathbb{E}\|\tilde{\mathbf{d}}_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}, \mathbf{v}_{t+1}) - \tilde{\mathbf{d}}_t(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{v}_{t+1})\|^2 \\
& 1520 & + 3\mathbb{E}\|\tilde{\mathbf{d}}_t(\mathbf{x}_t, \mathbf{y}_{t+1}, \mathbf{v}_{t+1}) - \tilde{\mathbf{d}}_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_{t+1})\|^2 \\
& 1521 & + 3\mathbb{E}\|\tilde{\mathbf{d}}_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_{t+1}) - \tilde{\mathbf{d}}_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2 \\
& 1522 & + 3\mathbb{E}\|\tilde{\mathbf{d}}_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_{t+1}) - \tilde{\mathbf{d}}_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2 \\
& 1523 & \\
& 1524 & \stackrel{(i)}{\leq} 3\mathbb{E}\|(\nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_{t+1}))\mathbf{v}_{t+1} + \nabla_{\mathbf{x}} f_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{x}} f_t(\mathbf{x}_t, \mathbf{y}_{t+1})\|^2 \\
& 1525 & + 3\mathbb{E}\|(\nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_{t+1}) - \nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t))\mathbf{v}_{t+1} + \nabla_{\mathbf{x}} f_t(\mathbf{x}_t, \mathbf{y}_{t+1}) - \nabla_{\mathbf{x}} f_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\
& 1526 & + 3\mathbb{E}\|(\nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_{t+1}) - \nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t))\mathbf{v}_{t+1} + \nabla_{\mathbf{x}} f_t(\mathbf{x}_t, \mathbf{y}_{t+1}) - \nabla_{\mathbf{x}} f_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\
& 1527 & + 3\mathbb{E}\|\tilde{\mathbf{d}}_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_{t+1}) - \tilde{\mathbf{d}}_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2 \\
& 1528 & \\
& 1529 & \stackrel{(ii)}{\leq} 6(\ell_{g,2}^2 \mathbb{E}\|\mathbf{v}_{t+1}\|^2 + \ell_{f,1}^2) (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + \mathbb{E}\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2) + 3\ell_{g,1}^2 \mathbb{E}\|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 \\
& 1530 & \\
& 1531 & \stackrel{(iii)}{\leq} 6(\ell_{g,2}^2 p^2 + \ell_{f,1}^2) (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + \beta_t^2 \mathbb{E}\|\mathbf{d}_t^Y\|^2) + 3\ell_{g,1}^2 \delta_t^2 \mathbb{E}\|\mathbf{d}_t^Y\|^2 \\
& 1532 & \\
& 1533 & \stackrel{(iv)}{\leq} 6(\ell_{g,2}^2 p^2 + \ell_{f,1}^2) (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\beta_t^2 \mathbb{E}\|e_t^g\|^2 + 2\beta_t^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\
& 1534 & + 6\ell_{g,1}^2 \delta_t^2 (\mathbb{E}\|e_t^Y\|^2 + \mathbb{E}\|\nabla P_t(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2) \\
& 1535 & \\
& 1536 & \stackrel{(vi)}{\leq} 6(\ell_{g,2}^2 p^2 + \ell_{f,1}^2) (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\beta_t^2 \mathbb{E}\|e_t^g\|^2 + 2\beta_t^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\
& 1537 & \\
& 1538 & + 6\ell_{g,1}^2 \delta_t^2 (\mathbb{E}\|e_t^Y\|^2 + \ell_{g,1}^2 \mathbb{E}\|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2), \tag{79}
\end{aligned}$$

which implies that

where the inequality follows from ∥a + b∥ <sup>2</sup> ≤ 2∥a∥ <sup>2</sup> + 2∥b∥ .

Let us bound the second term in the right-hand side of [\(77\)](#page-27-0). Based on [\(76\)](#page-26-1), we have

$$\begin{aligned} & \mathbb{E}\|\tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) - \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1})\|^2 \\ &= \mathbb{E}\| (\nabla_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})) \mathbf{v}_{t+1} \\ &+ \nabla_{\mathbf{x}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \mathcal{B}_{t+1}) - \nabla_{\mathbf{x}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &\leq 2\mathbb{E}\| (\nabla_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})) \mathbf{v}_{t+1}\|^2 \\ &+ 2\mathbb{E}\|\nabla_{\mathbf{x}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \mathcal{B}_{t+1}) - \nabla_{\mathbf{x}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &\leq 2(\frac{\sigma_{\mathbf{g}\mathbf{x}\mathbf{y}}^2}{b} p^2 + \frac{\sigma_{\mathbf{f}\mathbf{x}}^2}{b}), \end{aligned}$$

where the first inequality is by and ∥a + b∥ <sup>2</sup> ≤ 2∥a∥ <sup>2</sup> + 2∥b∥ ; the second inequality follows from Assumptions [C3.,](#page-3-16) [C5.](#page-3-17) and [\(8\)](#page-3-8).

Substituting the above inequality into [\(77\)](#page-27-0) and using ∥a + b + c∥ <sup>2</sup> ≤ 3(∥a∥ <sup>2</sup> + ∥b∥ <sup>2</sup> + ∥c∥ 2 ), we obtain

$$\begin{aligned}\mathbb{E}\|e_{t+1}^f\|^2 &\leq (1 - \eta_{t+1})^2 \mathbb{E}\|e_t^f\|^2 + 4\lambda_{t+1}^2 \left(\frac{\sigma_{g \times \mathbf{x}}^2}{b} p^2 + \frac{\sigma_{f \times \mathbf{x}}^2}{b}\right) \\ &\quad + 6(1 - \eta_{t+1})^2 \mathbb{E}\|\tilde{\mathbf{d}}_t(\mathbf{u}_t) - \tilde{\mathbf{d}}_t(\mathbf{u}_{t+1})\|^2 \\ &\quad + 6(1 - \eta_{t+1})^2 \mathbb{E}\|\tilde{\mathbf{d}}_t(\mathbf{u}_{t+1}) - \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1})\|^2 \\ &\quad + 6(1 - \eta_{t+1})^2 \mathbb{E}\|\tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1}; \mathcal{B}_{t+1}) - \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_t; \mathcal{B}_{t+1})\|^2.\end{aligned}\tag{78}$$

*1547 1548* Substituting [\(80\)](#page-28-1) and [\(79\)](#page-27-1) into [\(78\)](#page-27-2), we have

*1549*

*1554*

*1556*

*1559 1560*

*1564*

*1569*

*1574*

*1576*

*1579*

*1584*

*1589 1590 1591*

where the (i) follows from [\(76\)](#page-26-1); (ii) follows from Assumptions [B2.,](#page-3-12) [B3.](#page-3-10) and [B4.;](#page-3-13) (iii) follows from [\(8\)](#page-3-8); (iv) follows from [\(43\)](#page-16-3) and [\(55\)](#page-20-3); (vi) follows from [\(62\)](#page-23-0).

Similarly, we have

$$\begin{aligned} & \mathbb{E}\|\tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1}) - \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_t; \mathcal{B}_{t+1})\|^2 \\ & \leq 6(\ell_{g,2}^2 p^2 + \ell_{f,1}^2) (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\beta_t^2 \mathbb{E}\|e_t^g\|^2 + 2\beta_t^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\ & + 6\ell_{g,1}^2 \delta_t^2 (\mathbb{E}\|e_t^{\mathbf{y}}\|^2 + \ell_{g,1}^2 \mathbb{E}\|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2). \end{aligned} \quad (80)$$

$$\begin{aligned} \mathbb{E}\|e_{t+1}^f\|^2 &\leq (1 - \eta_{t+1})^2 \mathbb{E}\|e_t^f\|^2 + 4\eta_{t+1}^2 \left(\frac{\sigma_{yy}^2}{b} p^2 + \frac{\sigma_{fy}^2}{b}\right) \\ &\quad + 6(1 - \eta_{t+1})^2 \mathbb{E}\|\tilde{\mathbf{d}}_t(\mathbf{u}_{t+1}) - \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1})\|^2 \\ &\quad + 72(1 - \eta_{t+1})^2 \ell_{g,2}^2 p^2 + \ell_{f,1}^2 (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\beta_t^2 \mathbb{E}\|e_t^g\|^2 + 2\beta_t^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\ &\quad + 72\ell_{g,1}^2 (1 - \eta_{t+1})^2 \delta_t^2 \mathbb{E}\|e_t^{\mathbf{y}}\|^2 + 72(1 - \eta_{t+1})^2 \ell_{g,1}^4 \delta_t^2 \mathbb{E}\|\mathbf{v}_t - \mathbf{v}_t^*(\mathbf{x}_t)\|^2. \end{aligned}$$

From ∥a + b∥ <sup>2</sup> ≤ 2∥a∥ <sup>2</sup> + 2∥b∥ 2 and [\(8\)](#page-3-8), we have

$$\begin{aligned} \mathbb{E}\|\tilde{\mathbf{d}}_t(\mathbf{u}_{t+1}) - \tilde{\mathbf{d}}_{t+1}(\mathbf{u}_{t+1})\|^2 &= \mathbb{E}\|\nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) \mathbf{v}_{t+1} - \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) \mathbf{v}_{t+1} \\ &\quad + \nabla_{\mathbf{x}} f_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{x}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &\leq 2\mathbb{E}\|(\nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})) \mathbf{v}_{t+1}\|^2 \\ &\quad + 2\mathbb{E}\|\nabla_{\mathbf{x}} f_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{x}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &\leq 2\mathbb{E}\|\nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 p^2 \\ &\quad + 2\mathbb{E}\|\nabla_{\mathbf{x}} f_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{x}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2. \end{aligned}$$

This completes the proof.

As demonstrated in Lemma [C.9,](#page-26-0) the hypergradient estimator error e f <sup>t</sup>+1 comprises five key components: (1) the term (1 − ηt+1) <sup>2</sup>E∥e f t ∥ 2 , representing the per-iteration improvement achieved by the momentum-based update; (2) the error arising from the variation in the Jacobian of the lower-level objectiv; (3) the error caused by the variation in the gradient of the upper-level objective ; (4) the error term O(2β 2 <sup>t</sup> <sup>E</sup>∥e g t ∥ <sup>2</sup>+2β 2 <sup>t</sup> <sup>E</sup>∥∇ygt(xt, yt)∥ 2 ), which is due to solving the lower-level problem; and (5) the error term O(δ 2 <sup>t</sup> <sup>E</sup>∥e v t ∥ <sup>2</sup> + 72(1 − ηt+1) 2 ℓ 4 g,1 δ 2 t θ v t ), which is introduced by the one-step momentum update in solving the linear system problem.

Lemma C.10. *Let Assumption [3.4](#page-3-4) holds. Then, for the sequence of functions* {ft} T <sup>t</sup>=1*, we have*

$$\sum_{t=1}^T (f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - f_t(\mathbf{x}_{t+1}, \mathbf{y}_t^*(\mathbf{x}_{t+1}))) \leq 2M + V_T,$$

*where* M *is defined in Assumption [3.4;](#page-3-4)* V<sup>T</sup> *is defined in* [\(10\)](#page-3-0)*.*

*Proof.* Note that, we have

$$\begin{aligned} & \sum_{t=1}^T (f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - f_t(\mathbf{x}_{t+1}, \mathbf{y}_t^*(\mathbf{x}_{t+1}))) \\ &= f_1(\mathbf{x}_1, \mathbf{y}_1^*(\mathbf{x}_1)) - f_T(\mathbf{x}_{T+1}, \mathbf{y}_T^*(\mathbf{x}_{T+1})) \\ &+ \sum_{t=2}^T (f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - f_{t-1}(\mathbf{x}_t, \mathbf{y}_{t-1}^*(\mathbf{x}_t))) \\ &\leq 2M + V_T, \end{aligned}$$

1596 Lemma C.11. *Let* {ft} T <sup>t</sup>=1 *denote the sequence of functions presented to Algorithm [1,](#page-2-1) satisfying Assumptions [3.2,](#page-3-2) [3.3](#page-3-3) and [3.4.](#page-3-4) Let* P<sup>X</sup> ,α<sup>t</sup> *be defined as in Definition [B.1.](#page-11-0) For any positive step size* α<sup>t</sup> *such that* α<sup>t</sup> ≤ L<sup>f</sup> *for all* t ∈ [T]*, Algorithm [1](#page-2-1) ensures the following bound:*

1607 1608 *Here,* θ y <sup>t</sup> *and* θ v <sup>t</sup> *are defined in* [\(47\)](#page-18-3)*;* V<sup>T</sup> *is defined in* [\(10\)](#page-3-0)*,* M *is given in Assumption [3.4;](#page-3-4) and* M<sup>f</sup> *is defined in* [\(40\)](#page-15-0)*.*

1609

1614

1616

1618 1619

1624

1626

1629

1634

1636

1639 where e f t := d x <sup>t</sup> − <sup>d</sup>˜ <sup>t</sup> (zt). This implies that

$$\begin{aligned} 1645 \quad & f_t(\mathbf{x}_{t+1}, \mathbf{y}_t^*(\mathbf{x}_{t+1})) - f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) \\ 1646 & \leq \frac{(L_f \alpha_t^2 - \alpha_t)}{2} \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \mathbf{d}_t^{\mathbf{x}})\|^2 + 2\alpha_t \left\| e_t^f \right\|^2 + M_f^2 (\theta_t^{\mathbf{y}} + \theta_t^{\mathbf{y}}) \alpha_t, \\ 1647 & \\ 1648 & \\ 1649 & \end{aligned}$$

$$\begin{aligned} & \sum_{t=1}^T (\alpha_t - L_f \alpha_t^2) \mathbb{E} \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \\ & \leq 8M + 4V_T + 2M_f^2 \sum_{t=1}^T (2\alpha_t - L_f \alpha_t^2) (\mathbb{E}[\theta_t^\mathbf{y}] + \mathbb{E}[\theta_t^\mathbf{y}]) \\ & + 2 \sum_{t=1}^T (2\alpha_t - L_f \alpha_t^2) \mathbb{E} \left\| e_t^f \right\|^2. \end{aligned} \quad (81)$$

*Proof.* It follows from Lemma [C.1](#page-15-6) that

$$\begin{aligned} & f_t(\mathbf{x}_{t+1}, \mathbf{y}_t^*(\mathbf{x}_{t+1})) - f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) \\ & \leq \langle \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)), \mathbf{x}_{t+1} - \mathbf{x}_t \rangle + \frac{L_f}{2} \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 \\ & = -\alpha_t \langle \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)), \mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \mathbf{d}_t^\mathbf{x}) \rangle + \frac{L_f \alpha_t^2}{2} \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \mathbf{d}_t^\mathbf{x})\|^2. \end{aligned} \quad (82)$$

For the first term on the right hand side of [\(82\)](#page-29-0), we have that

$$\begin{aligned} & - \langle \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)), \mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \mathbf{d}_t^x) \rangle \\ & = - \langle \mathbf{d}_t^x, \mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \mathbf{d}_t^x) \rangle - \langle \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - \mathbf{d}_t^x, \mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \mathbf{d}_t^x) \rangle \\ & \leq -\frac{1}{2} \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \mathbf{d}_t^x)\|^2 + \frac{1}{2} \|\mathbf{d}_t^x - \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))\|^2, \end{aligned}$$

where the inequality follows from Lemma [B.8.](#page-12-1)

Let d˜ <sup>t</sup> (zt) = ∇xft(zt) + ∇<sup>2</sup> xyg<sup>t</sup> (zt) vt. Then, from Lemma [C.1,](#page-15-6) we have

$$\begin{aligned} \|\mathbf{d}_t^x - \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))\|^2 &= \left\| \mathbf{d}_t^x - \tilde{\mathbf{d}}_t(\mathbf{z}_t) + \tilde{\mathbf{d}}_t(\mathbf{z}_t) - \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) \right\|^2 \\ &\leq 2 \left\| \mathbf{d}_t^x - \tilde{\mathbf{d}}_t(\mathbf{z}_t) \right\|^2 + 2 \left\| \tilde{\mathbf{d}}_t(\mathbf{z}_t) - \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) \right\|^2 \\ &\leq 2 \left\| e_t^f \right\|^2 + 2 \left\| \tilde{\mathbf{d}}_t(\mathbf{z}_t) - \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) \right\|^2 \\ &\leq 2 \left\| e_t^f \right\|^2 + M_f^2 (\theta_t^y + \theta_t^y), \end{aligned} \tag{83}$$

$$\begin{aligned} & -\langle \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)), \mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \mathbf{d}_t^x) \rangle \\ & \leq -\frac{1}{2} \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \mathbf{d}_t^x)\|^2 + 2 \left\| e_t^f \right\|^2 + M_f^2 \left( \mathbf{y}_t^y + \theta_t^y \right), \end{aligned} \quad (84)$$

Plugging the bound [\(84\)](#page-29-1) into [\(82\)](#page-29-0), we have that

1656 In addition, we have

1663 1664 1665 where the second inequaliy follows from non-expansiveness of the projection operator and the last inequality follows from [\(83\)](#page-29-2).

1674

1676

1679

1681 where the second inequality is due to Lemma [C.10.](#page-28-2)

1689 1690 *Proof.* From the update rule of Algorithm [1,](#page-2-1) we have

1694

1696

1699 1700

1704 where the first inequality is by (a+b) <sup>2</sup> ≤ 2a <sup>2</sup> + 2b 2 ; the second inequality follows from non-expansiveness of the projection operator; and the last inequality follows from Eq. [\(37a\)](#page-15-1) in Lemma [C.1.](#page-15-6)

which can be rearranged into

$$\begin{aligned} & (\alpha_t - L_f \alpha_t^2) \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \mathbf{d}^\mathbf{x})\|^2 \\ & \leq 2f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - f_t(\mathbf{x}_{t+1}, \mathbf{y}_t^*(\mathbf{x}_{t+1})) + 4\alpha_t \left\| e_t^f \right\|^2 + 2M_f^2 (\theta_t^\mathbf{y} + \theta_t^\mathbf{v}) \alpha_t. \end{aligned} \quad (85)$$

$$\begin{aligned} & \|\mathcal{P}_{\mathcal{X},\alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \\ & \leq 2 \|\mathcal{P}_{\mathcal{X},\alpha_t}(\mathbf{x}_t; \mathbf{d}_t^\mathbf{x}) - \mathcal{P}_{\mathcal{X},\alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 + 2 \|\mathcal{P}_{\mathcal{X},\alpha_t}(\mathbf{x}_t; \mathbf{d}_t^\mathbf{x})\|^2 \\ & \leq 2 \|\mathbf{d}_t^\mathbf{x} - \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))\|^2 + 2 \|\mathcal{P}_{\mathcal{X},\alpha_t}(\mathbf{x}_t; \mathbf{d}_t^\mathbf{x})\|^2 \\ & \leq 4 \left\| e_t^f \right\|^2 + 4M_f^2 (\theta_t^\mathbf{y} + \theta_t^\mathbf{x}) + 4 \|\mathcal{P}_{\mathcal{X},\alpha_t}(\mathbf{x}_t; \mathbf{d}_t^\mathbf{x})\|^2, \end{aligned} \quad (86)$$

Combining [\(85\)](#page-30-0) and [\(86\)](#page-30-1), we have

$$\begin{aligned} & \sum_{t=1}^T (\alpha_t - L_f \alpha_t^2) \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \\ & \leq 4 \sum_{t=1}^T (f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - f_t(\mathbf{x}_{t+1}, \mathbf{y}_t^*(\mathbf{x}_{t+1}))) \\ & + 2M_f^2 \sum_{t=1}^T (2\alpha_t - L_f \alpha_t^2) (\theta_t^\mathbf{y} + \theta_t^\mathbf{y}) + 2 \sum_{t=1}^T (2\alpha_t - L_f \alpha_t^2) \left\| e_t^f \right\|^2 \\ & \leq 8M + 4V_T \\ & + 2M_f^2 \sum_{t=1}^T (2\alpha_t - L_f \alpha_t^2) (\theta_t^\mathbf{y} + \theta_t^\mathbf{y}) + 2 \sum_{t=1}^T (2\alpha_t - L_f \alpha_t^2) \left\| e_t^f \right\|^2, \end{aligned}$$

Lemma C.12. *Let Assumptions [3.3](#page-3-3) and [3.4](#page-3-4) hold. Let* {xt} T <sup>t</sup>=1 *be generated according to Algorithm [1.](#page-2-1) Then, we have*

$$\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 \leq 2\alpha_t^2 \left( \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 + M_f^2(\theta_t^{\mathbf{y}} + \theta_t^{\mathbf{y}}) \right),$$

*where* θ y <sup>t</sup> *and* θ v <sup>t</sup> *are defined in* [\(47\)](#page-18-3)*.*

$$\begin{aligned} \|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 &= \alpha_t^2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \mathbf{d}_t^{\mathbf{x}})\|^2 \\ &\leq 2\alpha_t^2 \left( \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \right. \\ &\quad \left. + \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \mathbf{d}_t^{\mathbf{x}}) - \mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \right) \\ &\leq 2\alpha_t^2 \left( \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \right. \\ &\quad \left. + \|\mathbf{d}_t^{\mathbf{x}} - \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))\|^2 \right) \\ &\leq 2\alpha_t^2 \left( \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 + M_f^2 (\mathbf{y}_t^{\mathbf{y}} + \theta_t^{\mathbf{y}}) \right), \end{aligned} \tag{87}$$

1709

1714

1716

1719

1724

1726

1729

1734

1736

1754

1756

#### C.5. Proof of Theorem [3.6](#page-4-0)

*Proof.* Bounding <sup>E</sup>∥e f t ∥ 2 in [\(75\)](#page-26-2) . From [\(75\)](#page-26-2), we have

$$\begin{aligned} \frac{\mathbb{E}\|e_{t+1}^f\|^2}{\alpha_t} - \frac{\mathbb{E}\|e_t^f\|^2}{\alpha_{t-1}} &\leq \left( \frac{(1 - \eta_{t+1})^2}{\alpha_t} - \frac{1}{\alpha_{t-1}} \right) \mathbb{E}\|e_t^f\|^2 + \frac{4\eta_{t+1}^2}{\alpha_t} \left( \frac{\sigma_{g_{\mathbf{x}\mathbf{y}}}^2}{\bar{b}} p^2 + \frac{\sigma_{f_{\mathbf{x}}}^2}{b} \right) \\ &+ \frac{12p^2}{\alpha_t} (1 - \eta_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{x}\mathbf{y}} g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &+ \frac{12}{\alpha_t} (1 - \eta_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{x}} f_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{x}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &+ \frac{72}{\alpha_t} (1 - \eta_{t+1})^2 (\ell_{g,2}^2 p^2 + \ell_{f,1}^2) (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\beta_t^2 \mathbb{E}\|e_t^g\|^2 + 2\beta_t^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\ &+ \frac{72}{\alpha_t} \ell_{g,1}^2 (1 - \eta_{t+1})^2 \delta_t^2 \mathbb{E}\|e_t^{\mathbf{y}}\|^2 + \frac{72}{\alpha_t} (1 - \eta_{t+1})^2 \ell_{g,1}^4 \delta_t^2 \mathbb{E}[\theta_t^{\mathbf{y}}]. \end{aligned} \quad (88)$$

With respect to the coefficient of the first term on the right-hand side of equation [\(88\)](#page-31-0), it is important to note that we have:

$$\frac{(1 - \eta_{t+1})^2}{\alpha_t} - \frac{1}{\alpha_{t-1}} \leq \frac{1}{\alpha_t} - \frac{\eta_{t+1}}{\alpha_t} - \frac{1}{\alpha_{t-1}}. \quad (89)$$

Using the definition of α<sup>t</sup> in [\(15\)](#page-4-9), we have

$$\begin{aligned} \frac{1}{\alpha_t} - \frac{1}{\alpha_{t-1}} &= (c+t)^{1/3} - (c+t-1)^{1/3} \stackrel{(i)}{\leq} \frac{1}{3(c+t-1)^{2/3}} \stackrel{(ii)}{\leq} \frac{1}{3(\frac{c}{2}+t)^{2/3}} \\ &= \frac{2^{2/3}}{3(c+2t)^{2/3}} \stackrel{(iii)}{\leq} \frac{2^{2/3}}{3(c+t)^{2/3}} \stackrel{(iv)}{\leq} \frac{2^{2/3}}{3} \alpha_t^{(vi)} \stackrel{(vi)}{\leq} \frac{\alpha_t}{6L_f}, \end{aligned} \quad (90)$$

where the (i) follows from (a + b) <sup>1</sup>/<sup>3</sup> − a <sup>1</sup>/<sup>3</sup> ≤ b/(3a 2/3 ); (ii) follows from c ≥ 2 in [\(104\)](#page-34-0); (iii) follows from [\(15\)](#page-4-9); (iv) follows from α<sup>t</sup> ≤ 1/4L<sup>f</sup> in [\(104\)](#page-34-0).

Substituting [\(90\)](#page-31-1) into [\(89\)](#page-31-2) and using δ<sup>t</sup> = cδα<sup>t</sup> and ηt+1 = cηα 2 t , we have

$$\frac{(1 - \eta_{t+1})^2}{\alpha_t} - \frac{1}{\alpha_{t-1}} \leq \frac{\alpha_t}{6L_f} - \frac{\eta_{t+1}}{\alpha_t} = \frac{\alpha_t}{6L_f} - c_\eta \alpha_t \leq -5\Omega \alpha_t, \quad (91)$$

where the inequalities follow from c<sup>η</sup> = 1 6L<sup>f</sup> + 5Ω with Ω in [\(103\)](#page-34-1).

Then, substituting [\(91\)](#page-31-3) into [\(88\)](#page-31-0) yields

$$\begin{aligned} \frac{1}{\Omega} \mathbb{E} \left( \frac{\|e_{t+1}^f\|^2}{\alpha_t} - \frac{\|e_t^f\|^2}{\alpha_{t-1}} \right) &\leq -5\alpha_t \mathbb{E} \|e_t^f\|^2 + \frac{4\eta_{t+1}^2}{\Omega \alpha_t} \left( \frac{\sigma_{g \times y}^2}{b} p^2 + \frac{\sigma_{f \times x}^2}{b} \right) \\ &+ \frac{12p^2}{\Omega \alpha_t} (1 - \eta_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{x}\mathbf{y}} g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &+ \frac{12}{\Omega \alpha_t} (1 - \eta_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{x}} f_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{x}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &+ \frac{72}{\Omega \alpha_t} (1 - \eta_{t+1})^2 (\ell_{g,2}^2 p^2 + \ell_{f,1}^2) (\mathbb{E} \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\beta_t^2 \mathbb{E} \|e_t^g\|^2 + 2\beta_t^2 \mathbb{E} \|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\ &+ \frac{72}{\Omega \alpha_t} \ell_{g,1}^2 (1 - \eta_{t+1})^2 \delta_t^2 \mathbb{E} \|e_t^y\|^2 + \frac{72}{\Omega \alpha_t} (1 - \eta_{t+1})^2 \ell_{g,1}^4 \delta_t^2 \mathbb{E} [\theta_t^y]. \end{aligned} \quad (92)$$

Bounding <sup>E</sup>∥e g t ∥ 2 in [\(44\)](#page-16-4) .

From [\(44\)](#page-16-4), we have

1764

1766

1769

1774

1776

1779

1790

1794

1796

$$\begin{aligned} \frac{\mathbb{E}\|e_{t+1}^g\|^2}{\alpha_t} - \frac{\mathbb{E}\|e_t^g\|^2}{\alpha_{t-1}} &\leq \left( \frac{1}{\alpha_t} (1 - \gamma_{t+1})^2 (1 + 48\ell_{g,1}^2\beta_t^2) - \frac{1}{\alpha_{t-1}} \right) \mathbb{E}\|e_t^g\|^2 \\ &\quad + 2 \frac{\gamma_{t+1}^2}{\alpha_t} \frac{\sigma_{g_y}^2}{b} + \frac{24}{\alpha_t} (1 - \gamma_{t+1})^2 \ell_{g,1}^2 \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 \\ &\quad + \frac{6}{\alpha_t} (1 - \gamma_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &\quad + 48(1 - \gamma_{t+1})^2 \ell_{g,1}^2 \frac{\beta_t^2}{\alpha_t} \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2. \end{aligned} \quad (93)$$

Let us examine the coefficient of the first term on the right-hand side of Eq. [\(93\)](#page-32-0). Specifically, for γt+1 = cγα 2 t and β<sup>t</sup> = cβαt, we have:

$$\begin{aligned} \frac{1}{\alpha_t}(1 - \gamma_{t+1})^2(1 + 48\ell_{g,1}^2\beta_t^2) - \frac{1}{\alpha_{t-1}} &\leq \frac{1}{\alpha_t}(1 - \gamma_{t+1})(1 + 48\ell_{g,1}^2\beta_t^2) - \frac{1}{\alpha_{t-1}} \\ &= \frac{1}{\alpha_t} - \frac{1}{\alpha_{t-1}} - \frac{\gamma_{t+1}}{\alpha_t} + \frac{1 - \gamma_{t+1}}{\alpha_t} 48\ell_{g,1}^2\beta_t^2 \\ &= \frac{1}{\alpha_t} - \frac{1}{\alpha_{t-1}} - c_\gamma \alpha_t + \left( \frac{1}{\alpha_t} - c_\gamma \alpha_t \right) 48\ell_{g,1}^2 c_\beta^2 \alpha_t^2 \\ &\leq \frac{\alpha_t}{6L_f} + 48\ell_{g,1}^2 c_\beta^2 \alpha_t - c_\gamma \alpha_t, \end{aligned} \tag{94}$$

where the last inequality follows from [\(90\)](#page-31-1).

Recalling Φ from [\(103\)](#page-34-1) that we selected, we obtain

$$c_\gamma = \frac{1}{6L_f} + 48\ell_{g,1}^2 c_\beta^2 + \hbar\Phi, \quad \text{where} \quad \hbar := 25 \frac{M_f^2}{L_{\mu_g}^2},$$

which, when combined with Eq. [\(94\)](#page-32-1), results in

$$\frac{1}{\alpha_t}(1 - \gamma_{t+1})^2(1 + 48\ell_{g,1}^2\beta_t^2) - \frac{1}{\alpha_{t-1}} \leq -\hbar\Phi\alpha_t. \quad (95)$$

Substituting eq. [\(95\)](#page-32-2) into eq. [\(93\)](#page-32-0) yields

$$\begin{aligned} \frac{1}{\Phi} \left( \frac{\mathbb{E}\|e_{t+1}^g\|^2}{\alpha_t} - \frac{\mathbb{E}\|e_t^g\|^2}{\alpha_{t-1}} \right) &\leq -\hbar\alpha_t \mathbb{E}\|e_t^g\|^2 \\ &\quad + 2 \frac{\gamma_{t+1}^2}{\Phi\alpha_t} \frac{\sigma_{gy}^2}{b} + \frac{24}{\Phi\alpha_t} (1 - \gamma_{t+1})^2 \ell_{g,1}^2 \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 \\ &\quad + \frac{6}{\Phi\alpha_t} (1 - \gamma_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &\quad + 48(1 - \gamma_{t+1})^2 \ell_{g,1}^2 \frac{\beta_t^2}{\Phi\alpha_t} \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2. \end{aligned} \quad (96)$$

Bounding <sup>E</sup>∥e v t ∥ 2 in [\(56\)](#page-20-4) .

$$\begin{aligned} 1816 & \frac{\mathbb{E}\|e_{t+1}^y\|^2}{\alpha_t} - \frac{\mathbb{E}\|e_t^y\|^2}{\alpha_{t-1}} \leq \left( \frac{1}{\alpha_t} (1 - \lambda_{t+1})^2 (1 + 72\ell_{g,1}^2 \delta_t^2) - \frac{1}{\alpha_{t-1}} \right) \mathbb{E}\|e_t^y\|^2 \\ 1818 & + 4 \frac{\lambda_{t+1}^2}{\alpha_t} \left( \frac{\sigma_{gyy}^2}{b} p^2 + \frac{\sigma_{fy}^2}{b} \right) + \frac{12p^2}{\alpha_t} (1 - \lambda_{t+1})^2 \mathbb{E}\|\nabla_y^2 g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_y^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ 1820 & + \frac{12}{\alpha_t} (1 - \lambda_{t+1})^2 \mathbb{E}\|\nabla_y f_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_y f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ 1821 & + \frac{72}{\alpha_t} (1 - \lambda_{t+1})^2 (\ell_{g,2}^2 p^2 + \ell_{f,1}^2) (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\beta_t^2 \mathbb{E}\|e_t^g\|^2 + 2\beta_t^2 \mathbb{E}\|\nabla_y g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\ 1824 & + \frac{72}{\alpha_t} (1 - \lambda_{t+1})^2 (\ell_{g,2}^2 p^2 + \ell_{f,1}^2) (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\beta_t^2 \mathbb{E}\|e_t^g\|^2 + 2\beta_t^2 \mathbb{E}\|\nabla_y g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\ 1825 & + \frac{72}{\alpha_t} (1 - \lambda_{t+1})^2 \ell_{g,1}^4 \delta_t^2 \mathbb{E}[\theta_t^y]. \\ 1827 & \end{aligned} \tag{97}$$

From [\(56\)](#page-20-4), we get

Let us examine the coefficient of the first term on the right-hand side of equation [\(97\)](#page-33-0). Specifically, for λt+1 = cλα 2 t and δ<sup>t</sup> = cδαt, we have:

$$\begin{aligned} \frac{1}{\alpha_t}(1 - \lambda_{t+1})^2(1 + 72\ell_{g,1}^2\delta_t^2) - \frac{1}{\alpha_{t-1}} &\leq \frac{1}{\alpha_t}(1 - \lambda_{t+1})(1 + 72\ell_{g,1}^2\delta_t^2) - \frac{1}{\alpha_{t-1}} \\ &= \frac{1}{\alpha_t} - \frac{1}{\alpha_{t-1}} - \frac{\lambda_{t+1}}{\alpha_t} + \frac{1 - \lambda_{t+1}}{\alpha_t} 72\ell_{g,1}^2\delta_t^2 \\ &= \frac{1}{\alpha_t} - \frac{1}{\alpha_{t-1}} - c_\lambda \alpha_t + \left( \frac{1}{\alpha_t} - c_\lambda \alpha_t \right) 72\ell_{g,1}^2 c_\delta^2 \alpha_t^2 \\ &\leq \frac{\alpha_t}{6L_f} + 72\ell_{g,1}^2 c_\delta^2 \alpha_t - c_\lambda \alpha_t, \end{aligned} \tag{98}$$

where the last inequality follows from [\(90\)](#page-31-1).

Recalling Ψ from [\(103\)](#page-34-1) that we selected, we obtain

$$c_\lambda = \frac{1}{6L_f} + 72\ell_{g,1}^2 c_\delta^2 + j\Psi, \quad \text{where} \quad j = 90 \frac{M_f^2}{L_{\mu_g}^2},$$

which, when combined with Eq. [\(98\)](#page-33-1), results in

$$\frac{1}{\alpha_t} (1 - \lambda_{t+1})^2 (1 + 7 2 \ell_{g,1}^2 \delta_t^2) - \frac{1}{\alpha_{t-1}} \leq - \gamma \Psi \alpha_t. \quad (99)$$

Substituting eq. [\(99\)](#page-33-2) into eq. [\(97\)](#page-33-0) yields

$$\begin{aligned} \frac{1}{\Psi} \left( \frac{\mathbb{E}\|e_{t+1}^{\mathbf{y}}\|^2}{\alpha_t} - \frac{\mathbb{E}\|e_t^{\mathbf{y}}\|^2}{\alpha_{t-1}} \right) &\leq -\mathcal{J}\alpha_t \mathbb{E}\|e_t^{\mathbf{y}}\|^2 \\ &+ 4 \frac{\lambda_{t+1}^2}{\Psi \alpha_t} \left( \frac{\sigma_{gy}^2}{b} p^2 + \frac{\sigma_{fy}^2}{b} \right) + \frac{12p^2}{\Psi \alpha_t} (1 - \lambda_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &+ \frac{12}{\Psi \alpha_t} (1 - \lambda_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} f_t(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\|^2 \\ &+ \frac{72}{\Psi \alpha_t} (1 - \lambda_{t+1})^2 (\ell_{g,2}^2 p^2 + \ell_{f,1}^2) (\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 2\beta_t^2 \mathbb{E}\|e_t^g\|^2 + 2\beta_t^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2) \\ &+ \frac{72}{\Psi \alpha_t} (1 - \lambda_{t+1})^2 \ell_{g,1}^4 \delta_t^2 \mathbb{E}[\theta_t^{\mathbf{y}}]. \end{aligned} \tag{100}$$

Combining the outcomes . We recall from Lemma [C.12](#page-30-2) that we have

$$\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 \leq 2\alpha_t^2 \left( \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 + M_f^2(\theta_t^y + \theta_t^v) \right). \quad (101)$$

1877 1878 Here

$$\begin{aligned} 1879 \\ 1880 \\ 1881 \\ 1882 \\ 1883 \\ 1884 \\ 1885 \\ 1886 \\ 1887 \end{aligned} \quad \Gamma = \frac{11M_f^2}{L_{\mu_g c_\beta}}, \quad \Upsilon = \frac{22M_f^2}{L_{\mu_g c_\delta}}, \quad \Phi = 480\ell_{g,1}^2, \quad (103)$$

$$\Psi = \max \left\{ 144(\ell_{g,2}^2 p^2 + \ell_{f,1}^2) \left( 10 + \frac{L_{\mu_g}^2 c_\beta^2}{11M_f^2} \right), \frac{288\ell_{g,1}^4}{M_f^2 c_\delta^2} \right\},$$

$$\Omega = \max \left\{ 144(\ell_{g,2}^2 p^2 + \ell_{f,1}^2) \left( 10 + \frac{L_{\mu_g}^2 c_\beta^2}{11M_f^2} \right), \frac{288\ell_{g,1}^4}{M_f^2 c_\delta^2} \right\}.$$

1887 1888 Here, from [\(15\)](#page-4-9), we have

1903 1904 Using [\(100\)](#page-33-3), [\(96\)](#page-32-3), [\(92\)](#page-31-4), [\(81\)](#page-29-3), [\(67\)](#page-24-5), and [\(48\)](#page-19-4), along with [\(101\)](#page-33-4) and the fact that α<sup>t</sup> decreases with respect to t, we obtain:

1914

1916

$$\begin{aligned} & \frac{1}{18} + \frac{\sigma_y^2}{b} \frac{2}{\Phi} \sum_{t=1}^T \frac{\gamma_{t+1}^2}{\alpha_t} + \frac{4}{\Psi} \left( \frac{\sigma_{yy}^2}{b} p^2 + \frac{\sigma_{fy}^2}{b} \right) \sum_{t=1}^T \frac{\lambda_{t+1}^2}{\alpha_t} + \frac{4}{\Omega} \left( \frac{\sigma_{xy}^2}{b} p^2 + \frac{\sigma_{fx}^2}{b} \right) \sum_{t=1}^T \frac{\eta_{t+1}^2}{\alpha_t} \\ & (105d) \end{aligned}$$

Let

$$\begin{aligned} \Lambda &:= \Gamma \sum_{t=1}^T \left( \mathbb{E}[\theta_{t+1}^\mathbf{y}] - \mathbb{E}[\theta_t^\mathbf{y}] \right) + \Upsilon \sum_{t=1}^T \left( \mathbb{E}[\theta_{t+1}^\mathbf{y}] - \mathbb{E}[\theta_t^\mathbf{y}] \right) + \frac{1}{\Phi} \sum_{t=1}^T \left( \frac{\mathbb{E}\|e_{t+1}^g\|^2}{\alpha_t} - \frac{\mathbb{E}\|e_t^g\|^2}{\alpha_{t-1}} \right) \\ &+ \frac{1}{\Psi} \sum_{t=1}^T \left( \frac{\mathbb{E}\|e_{t+1}^\mathbf{y}\|^2}{\alpha_t} - \frac{\mathbb{E}\|e_t^\mathbf{y}\|^2}{\alpha_{t-1}} \right) + \frac{1}{\Omega} \sum_{t=1}^T \left( \frac{\mathbb{E}\|e_{t+1}^f\|^2}{\alpha_t} - \frac{\mathbb{E}\|e_t^f\|^2}{\alpha_{t-1}} \right). \end{aligned} \quad (102)$$

$$\begin{aligned}
 c &\geq \max \{4L_f, c_\beta(\mu_g + \ell_{g,1}), 2\}, \\
 c_\beta &= \sqrt{\frac{880}{L_{\mu_g}^2} \frac{L_y^2 M_f^2}{L_{\mu_g}^2}}, \\
 c_\delta &= \sqrt{\frac{3520}{L_{\mu_g}^2 \mu_g^2} \frac{\nu^2 M_f^2}{(1 + 2L_y^2)}}, \\
 c_\gamma &= \frac{2}{3L_f} + 48\ell_{g,1}^2 c_\beta^2 + \hbar \Phi, \quad \text{where} \quad \hbar := 25 \frac{M_f^2}{L_{\mu_g}^2}, \\
 c_\eta &= \frac{2}{3L_f} + 5\Omega, \\
 c_\lambda &= \frac{2}{3L_f} + 72\ell_{g,1}^2 c_\delta^2 + \jmath \Psi, \quad \text{where} \quad \jmath = 90 \frac{M_f^2}{L_{\mu_g}^2}.
 \end{aligned} \tag{104}$$

$$\begin{aligned} & \sum_{t=1}^T A(\alpha_t, \beta_t, \delta_t) \mathbb{E} \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 + \Lambda \\ & \leq 8M + 4V_T + \sum_{t=1}^T B(\alpha_t, \beta_t, \delta_t) \mathbb{E}[\theta_t^Y] + \sum_{t=1}^T C(\alpha_t, \beta_t) \mathbb{E}[\theta_t^Y] \end{aligned} \quad (105a)$$

$$+ \sum_{t=1}^T D(\alpha_t) \mathbb{E}\|e_t^f\|^2 + \sum_{t=1}^T F(\beta_t, \delta_t) \mathbb{E}\|e_t^g\|^2 + \sum_{t=1}^T I(\alpha_t, \beta_t, \delta_t) \mathbb{E}\|e_t^v\|^2 \quad (105b)$$

$$+ \sum_{t=1}^T L(\beta_t) \mathbb{E} \|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 + \sum_{t=2}^T N(\beta_t, \delta_t) \sup_{\mathbf{x} \in \mathcal{X}} \|\mathbf{y}_{t-1}^*(\mathbf{x}) - \mathbf{y}_t^*(\mathbf{x})\|^2 \quad (105c)$$

$$+ \frac{\sigma_y^2}{b} \frac{2}{\Phi} \sum_{t=1}^T \frac{\gamma_{t+1}^2}{\alpha_t} + \frac{4}{\Psi} \left( \frac{\sigma_{yyy}^2}{b} p^2 + \frac{\sigma_{fy}^2}{b} \right) \sum_{t=1}^T \frac{\lambda_{t+1}^2}{\alpha_t} + \frac{4}{\Omega} \left( \frac{\sigma_{yxy}^2}{b} p^2 + \frac{\sigma_{fx}^2}{b} \right) \sum_{t=1}^T \frac{\eta_{t+1}^2}{\alpha_t} \quad (105d)$$

$$\begin{aligned} & + \frac{6}{\Phi \alpha_T} G_{y,T} + \frac{12p^2}{\Omega \alpha_T} G_{xy,T} + \frac{12p^2}{\Psi \alpha_T} G_{yy,T} + \frac{12\ell_{f,1}^2}{\Psi \alpha_T} D_{y,T} + \frac{12\ell_{f,1}^2}{\Omega \alpha_T} D_{x,T}. \\ & (105e) \end{aligned}$$

1929

1934

1936

1947 1948 Moreover, we have

1949

1954

1956

1963 1964 which together with β<sup>t</sup> = cβαt, δ<sup>t</sup> = cδαt, we have

1974

1976 where the first inequality follows from Γ = <sup>11</sup>M<sup>2</sup> f Lµg c<sup>β</sup> and Υ = <sup>22</sup>M<sup>2</sup> f Lµg cδ in [\(103\)](#page-34-1); the last inequality follows from c<sup>β</sup> ≥

[\(12\)](#page-3-9). Let

$$\begin{aligned} E(\beta_t, \delta_t) &:= \frac{4L_{\mathbf{y}}^2}{L_{\mu_g}\beta_t} \Gamma + \frac{8\nu^2}{L_{\mu_g}\mu_g^2\delta_t} (1 + 2L_{\mathbf{y}}^2) \Upsilon + 72(1 - \eta_{t+1})^2 (\ell_{g,2}^2 p^2 + \ell_{f,1}^2) \frac{1}{\Omega\alpha_t} \\ &\quad + 24(1 - \gamma_{t+1})^2 \ell_{g,1}^2 \frac{1}{\Phi\alpha_t} + 72(1 - \lambda_{t+1})^2 (\ell_{g,2}^2 p^2 + \ell_{f,1}^2) \frac{1}{\Psi\alpha_t}, \\ A(\alpha_t, \beta_t, \delta_t) &:= \alpha_t - (L_f + 2E(\beta_t, \delta_t)) \alpha_t^2, \\ B(\alpha_t, \beta_t, \delta_t) &:= -\frac{L_{\mu_g} \Upsilon}{4} \delta_t + 4M_f^2 \alpha_t - 2M_f^2 L_f \alpha_t^2 + 2M_f^2 E(\beta_t, \delta_t) \alpha_t^2 \\ &\quad + 72(1 - \lambda_{t+1})^2 \ell_{g,1}^4 \delta_t^2 \frac{1}{\Psi\alpha_t} + 72(1 - \eta_{t+1})^2 \ell_{g,1}^4 \delta_t^2 \frac{1}{\Omega\alpha_t}, \\ C(\alpha_t, \beta_t) &:= -\frac{L_{\mu_g} \Gamma}{2} \beta_t + 4M_f^2 \alpha_t - 2L_f M_f^2 \alpha_t^2 + 2M_f^2 E(\beta_t, \delta_t) \alpha_t^2, \\ D(\alpha_t) &:= 2(2\alpha_t - L_f \alpha_t^2) - 5\alpha_t, \\ F(\alpha_t, \beta_t, \delta_t) &:= \frac{2\Gamma}{L_{\mu_g}} \beta_t - \hbar\alpha_t + 72(1 - \lambda_{t+1})^2 (\ell_{g,2}^2 p^2 + \ell_{f,1}^2) 2 \frac{\beta_t^2}{\Psi\alpha_t} \\ &\quad + 72(1 - \eta_{t+1})^2 (\ell_{g,2}^2 p^2 + \ell_{f,1}^2) 2 \frac{\beta_t^2}{\Omega\alpha_t}, \\ I(\alpha_t, \beta_t, \delta_t) &:= \frac{4\Upsilon}{L_{\mu_g}} \delta_t - j\alpha_t + 72\ell_{g,1}^2 (1 - \eta_{t+1})^2 \frac{\delta_t^2}{\Omega\alpha_t}. \end{aligned} \tag{106}$$

$$\begin{aligned} L(\beta_t) &:= -\frac{2\Gamma}{\mu_g + \ell_{g,1}} \beta_t + \Gamma \beta_t^2 + 48(1 - \gamma_{t+1})^2 \ell_{g,1}^2 \frac{\beta_t^2}{\Phi \alpha_t} \\ &\quad + 72(1 - \lambda_{t+1})^2 (\ell_{g,2}^2 p^2 + \ell_{f,1}^2) 2 \frac{\beta_t^2}{\Psi \alpha_t} + 72(1 - \eta_{t+1})^2 (\ell_{g,2}^2 p^2 + \ell_{f,1}^2) 2 \frac{\beta_t^2}{\Omega \alpha_t}, \\ N(\beta_t, \delta_t) &:= \frac{4}{L_{\mu_g} \beta_t} \Gamma + \frac{16\nu^2}{L_{\mu_g} \mu_g^2 \delta_t} \Upsilon. \end{aligned} \quad (107)$$

Note that, we have

$$\begin{aligned} E(\beta_t, \delta_t) &= \frac{4L_{\mathbf{y}}^2}{L_{\mu_g}\beta_t} \Gamma + \frac{8\nu^2}{L_{\mu_g}\mu_g^2\delta_t}(1 + 2L_{\mathbf{y}}^2)\Upsilon + 72(1 - \eta_{t+1})^2(\ell_{g,2}^2p^2 + \ell_{f,1}^2)\frac{1}{\Omega\alpha_t} \\ &\quad + 24(1 - \gamma_{t+1})^2\ell_{g,1}^2\frac{1}{\Phi\alpha_t} + 72(1 - \lambda_{t+1})^2(\ell_{g,2}^2p^2 + \ell_{f,1}^2)\frac{1}{\Psi\alpha_t}, \end{aligned}$$

$$\begin{aligned} \alpha_t^2 E(\beta_t, \delta_t) &= \frac{44L_{\mathbf{y}}^2}{L_{\mu_g}} \Gamma\left(\frac{\alpha_t^2}{\beta_t} + \frac{8\nu^2}{L_{\mu_g}\mu_g^2}(1 + 2L_{\mathbf{y}}^2)\Upsilon \frac{\alpha_t^2}{\delta_t} + 72(1 - \eta_{t+1})^2(\ell_{g,2}^2 p^2 + \ell_{f,1}^2)\right) \frac{\alpha_t}{\Omega} \\ &\quad + 24(1 - \gamma_{t+1})^2 \ell_{g,1}^2 \frac{\alpha_t}{\Phi} + 72(1 - \lambda_{t+1})^2 (\ell_{g,2}^2 p^2 + \ell_{f,1}^2) \frac{\alpha_t}{\Psi} \\ &\leq \frac{44L_{\mathbf{y}}^2}{L_{\mu_g}^2} M_f^2 \frac{\alpha_t}{c_\beta^2} + \frac{176\nu^2}{L_{\mu_g}^2\mu_g^2} (1 + 2L_{\mathbf{y}}^2) M_f^2 \frac{\alpha_t}{c_\delta^2} \\ &\quad + 24\ell_{g,1}^2 \frac{\alpha_t}{\Phi} + 72(\ell_{g,2}^2 p^2 + \ell_{f,1}^2) \left(\frac{1}{\Omega} + \frac{1}{\Psi}\right) \alpha_t \\ &\leq \frac{\alpha_t}{4}, \end{aligned} \tag{108}$$

1981 Moreover, we have

1983 1984

1986 1987

1989

1990 1991 From [\(106\)](#page-35-0), we have

1994

1996 1997

2003 2004 2005 2006 2007 where the first inequality follows from <sup>β</sup><sup>t</sup> <sup>=</sup> <sup>c</sup>βαt, <sup>δ</sup><sup>t</sup> <sup>=</sup> <sup>c</sup>δαt, and [\(108\)](#page-35-1); the second inequality is by Υ = <sup>22</sup>M<sup>2</sup> f Lµg cδ , and Ψ, Ω ≥ 288ℓ 4 g,1 M<sup>2</sup> f c 2 δ in [\(103\)](#page-34-1); the last inequality follows from in [\(103\)](#page-34-1). Moreover, from [\(106\)](#page-35-0), and β<sup>t</sup> = cβαt, we have

2008 2009

2014

2016

2018 2019

2024

2026

2029

r <sup>880</sup> <sup>L</sup><sup>2</sup> yM<sup>2</sup> L<sup>2</sup> µg , c<sup>δ</sup> ≥ r <sup>3520</sup> <sup>ν</sup>2M<sup>2</sup> L<sup>2</sup> µg µ<sup>2</sup> g (1 + 2L<sup>2</sup> y ), in [\(104\)](#page-34-0) and Φ ≥ 480ℓ 2 g,1 , and Ω, Ψ ≥ 1440(ℓ 2 g,2p <sup>2</sup> + ℓ 2 f,1 ) in [\(103\)](#page-34-1).

$$\begin{aligned} A(\alpha_t, \beta_t, \delta_t) &:= \alpha_t - L_f \alpha_t^2 - 2E(\beta_t, \delta_t) \alpha_t^2 \\ &\geq \alpha_t - L_f \alpha_t^2 - \frac{\alpha_t}{2} \\ &\geq \frac{\alpha_t}{4}, \end{aligned} \tag{109}$$

where the last inequality is by α<sup>t</sup> ≤ 1/4L<sup>f</sup> in [\(104\)](#page-34-0).

Bounding [\(105a\)](#page-34-2) .

$$\begin{aligned}
B(\alpha_t, \beta_t, \delta_t) &= -\frac{L_{\mu_g}\Upsilon}{4}\delta_t + 4M_f^2\alpha_t - 2M_f^2L_f\alpha_t^2 + 2M_f^2E(\beta_t, \delta_t)\alpha_t^2 \\
&\quad + 72(1 - \lambda_{t+1})^2\ell_{g,1}^4\delta_t^2\frac{1}{\Psi\alpha_t} + 72(1 - \eta_{t+1})^2\ell_{g,1}^4\delta_t^2\frac{1}{\Omega\alpha_t} \\
&\leq -\frac{L_{\mu_g}\Upsilon}{4}\delta_t + 4M_f^2\alpha_t - 2M_f^2L_f\alpha_t^2 + \frac{M_f^2}{2}\alpha_t + 72\ell_{g,1}^4\left(\frac{1}{\Psi} + \frac{1}{\Omega}\right)\frac{\delta_t^2}{\alpha_t} \\
&= \left(-\frac{L_{\mu_g}\Upsilon}{4}\delta_t + \frac{9}{2}M_f^2 + 72\ell_{g,1}^4\left(\frac{1}{\Psi} + \frac{1}{\Omega}\right)c_\delta^2\right)\alpha_t \\
&\leq -\frac{1}{2}M_f^2\alpha_t, \tag{110}
\end{aligned}$$

$$\begin{aligned} C(\alpha_t, \beta_t) &= -\frac{L_{\mu_g}\Gamma}{2}\beta_t + 4M_f^2\alpha_t - 2L_fM_f^2\alpha_t^2 + 2M_f^2E(\beta_t, \delta_t)\alpha_t^2 \\ &\leq -\frac{L_{\mu_g}}{2}\Gamma c_\beta \alpha_t + \frac{9}{2}M_f^2\alpha_t \\ &= -M_f^2\alpha_t, \end{aligned} \tag{111}$$

where the first inequality follows from [\(108\)](#page-35-1); the last equality follows from Γ = <sup>11</sup>M<sup>2</sup> f Lµg c<sup>β</sup> in [\(103\)](#page-34-1).

Thus, from [\(110\)](#page-36-0) and [\(111\)](#page-36-1), we get

$$(105a) \leq \mathcal{O}(V_T). \quad (112)$$

Bounding [\(105b\)](#page-34-3) .

From [\(106\)](#page-35-0), we also obtain

$$D(\alpha_t) = 4\alpha_t - 2L_f\alpha_t^2 - 5\alpha_t \leq 0.$$

2054

2056

2064 2065 2066 where the second inequality follows from Υ = <sup>22</sup>M<sup>2</sup> f Lµg cδ and Ω ≥ 72ℓ 2 g,1L 2 µg M<sup>2</sup> f c δ ; the last equality follows from <sup>ȷ</sup> = 90 <sup>M</sup><sup>2</sup> f L<sup>2</sup> µg .

2067 Thus, we get

2068 2069

2074

2076

2079

2081 2082 2083

2084 2086 2087 where the second inequality is by Φ ≥ 192ℓ 2 g,1 (µg+ℓg,1) Γ cβ, and Ω, Ψ ≥ 576(ℓ 2 g,2p <sup>2</sup> + ℓ 2 f,1 ) (µg+ℓg,1) Γ c<sup>β</sup> in [\(103\)](#page-34-1); the last inequality follows from α<sup>t</sup> ≤ cβ(µg+ℓg,1) in [\(104\)](#page-34-0).

From <sup>β</sup><sup>t</sup> <sup>=</sup> <sup>c</sup>βαt, Γ = <sup>11</sup>M<sup>2</sup> Lµg c<sup>β</sup> and [\(106\)](#page-35-0), we obtain

$$\begin{aligned} F(\alpha_t, \beta_t, \delta_t) &= \frac{2\Gamma}{L_{\mu_g}} \beta_t - \hbar \alpha_t + 144(1 - \lambda_{t+1})^2 (\ell_{g,2}^2 p^2 + \ell_{f,1}^2) \frac{\beta_t^2}{\Psi \alpha_t} \\ &\quad + 144(1 - \eta_{t+1})^2 (\ell_{g,2}^2 p^2 + \ell_{f,1}^2) \frac{\beta_t^2}{\Omega \alpha_t} \\ &\leq \frac{22M_f^2}{L_{\mu_g}^2} \alpha_t - \hbar \alpha_t + 144(\ell_{g,2}^2 p^2 + \ell_{f,1}^2) \left( \frac{1}{\Psi} + \frac{1}{\Omega} \right) c_\beta^2 \alpha_t \\ &\leq 24 \frac{M_f^2}{L_{\mu_g}^2} \alpha_t - \hbar \alpha_t \\ &= -\frac{M_f^2}{L_{\mu_g}^2} \alpha_t, \end{aligned}$$

where the second inequality follows from Ω, Ψ ≥ 144(ℓ 2 g,2p <sup>2</sup> + ℓ 2 f,1 ) L µg c β M<sup>2</sup> in [\(103\)](#page-34-1); and the last equality is by <sup>ℏ</sup> := 25 <sup>M</sup><sup>2</sup> f L<sup>2</sup> µg . From δ<sup>t</sup> = cδαt, we obtain

$$\begin{aligned} I(\alpha_t, \beta_t, \delta_t) &= \frac{4\Upsilon}{L_{\mu_g}} \delta_t - j\alpha_t + 72\ell_{g,1}^2(1 - \eta_{t+1})^2 \frac{\delta_t^2}{\Omega\alpha_t} \\ &\leq \frac{4\Upsilon}{L_{\mu_g}} c_\delta \alpha_t - j\alpha_t + 72\ell_{g,1}^2 \frac{c_\delta^2 \alpha_t}{\Omega} \\ &\leq \frac{89M_f^2}{L_{\mu_g}^2} \alpha_t - j\alpha_t \\ &= -\frac{M_f^2}{L_{\mu_g}^2} \alpha_t, \end{aligned}$$

(105b) 
$$\leq 0$$
. (113)

#### Bounding [\(105c\)](#page-34-4) .

From β<sup>t</sup> = cβα<sup>t</sup> and [\(107\)](#page-35-2), we have

$$\begin{aligned}
L(\beta_t) &= -\frac{2\Gamma\beta_t}{\mu_g + \ell_{g,1}} + \Gamma\beta_t^2 + 48(1 - \gamma_{t+1})^2\ell_{g,1}^2 \frac{\beta_t^2}{\Phi\alpha_t} \\
&\quad + 72(1 - \lambda_{t+1})^2(\ell_{g,2}^2p^2 + \ell_{f,1}^2)2\frac{\beta_t^2}{\Psi\alpha_t} + 72(1 - \eta_{t+1})^2(\ell_{g,2}^2p^2 + \ell_{f,2}^2)2\frac{\beta_t^2}{\Omega\alpha_t} \\
&\leq -\frac{2\Gamma c_\beta\alpha_t}{\mu_g + \ell_{g,1}} + \Gamma c_\beta^2\alpha_t^2 + 48\ell_{g,1}^2c_\beta^2\frac{\alpha_t}{\Phi} + 144(\ell_{g,2}^2p^2 + \ell_{f,1}^2)\left(\frac{1}{\Psi} + \frac{1}{\Omega}\right)c_\beta^2\alpha_t \\
&\leq -\frac{2\Gamma c_\beta\alpha_t}{\mu_g + \ell_{g,1}} + \Gamma c_\beta^2\alpha_t^2 + \frac{3\Gamma c_\beta\alpha_t}{4(\mu_g + \ell_{g,1})} \\
&\leq -\frac{\Gamma c_\beta\alpha_t}{4(\mu_g + \ell_{g,1})},
\end{aligned}$$

2097 Thus, we get

2099 2100

2104

2106

2109

2111

2114

2116

2119

2124

2126

2129

2134

2136

From β<sup>t</sup> = cβαt, δ<sup>t</sup> = cδα<sup>t</sup> and [\(107\)](#page-35-2), we obtain

$$\begin{aligned} N(\beta_t, \delta_t) &= \frac{4}{L_{\mu_g}\beta_t}\Gamma + \frac{16\nu^2}{L_{\mu_g}\mu_g^2\delta_t}\Upsilon \\ &= \frac{4}{L_{\mu_g}c_\beta\alpha_t}\Gamma + \frac{16\nu^2}{L_{\mu_g}\mu_g^2c_\delta\alpha_t}\Upsilon. \end{aligned}$$

$$\begin{aligned} (105c) &= \sum_{t=1}^T L(\beta_t) \mathbb{E} \|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 + \sum_{t=2}^T N(\beta_t, \delta_t) \sup_{\mathbf{x} \in \mathcal{X}} \|\mathbf{y}_{t-1}^*(\mathbf{x}) - \mathbf{y}_t^*(\mathbf{x})\|^2 \\ &\leq \mathcal{O}\left(\frac{H_{2,T}}{\alpha_T}\right). \end{aligned} \quad (114)$$

Bounding [\(105d\)](#page-34-5) .

From ηt+1 = cηα 2 t , γt+1 = cγα 2 t , λt+1 = cλα 2 t , we obtain

$$\begin{aligned} (105d) &= \frac{\sigma_{gy}^2}{b} \frac{2}{\Phi} \sum_{t=1}^T \frac{\gamma_{t+1}^2}{\alpha_t} + \frac{4}{\Psi} \left( \frac{\sigma_{gyy}^2}{b} p^2 + \frac{\sigma_{fy}^2}{b} \right) \sum_{t=1}^T \frac{\lambda_{t+1}^2}{\alpha_t} + \frac{4}{\Omega} \left( \frac{\sigma_{gxy}^2}{b} p^2 + \frac{\sigma_{fx}^2}{b} \right) \sum_{t=1}^T \frac{\eta_{t+1}^2}{\alpha_t} \\ &\leq \mathcal{O} \left( \left( \frac{\sigma_{gy}^2}{b} + \frac{\sigma_{gyy}^2}{b} + \frac{\sigma_{fy}^2}{b} + \frac{\sigma_{gxy}^2}{b} + \frac{\sigma_{fx}^2}{b} \right) \sum_{t=1}^T \alpha_t^3 \right). \end{aligned} \quad (115)$$

Bounding [\(105e\)](#page-34-6) .

We have

$$\begin{aligned} (105e) &= \frac{6}{\Phi_{\alpha_T}} G_{\mathbf{y},T} + \frac{12p^2}{\Omega_{\alpha_T}} G_{\mathbf{x}\mathbf{y},T} + \frac{12p^2}{\Psi_{\alpha_T}} G_{\mathbf{y}\mathbf{y},T} + \frac{12\ell_{f,1}^2}{\Psi_{\alpha_T}} D_{\mathbf{y},T} + \frac{12\ell_{f,1}^2}{\Omega_{\alpha_T}} D_{\mathbf{x},T} \\ &\leq \mathcal{O} \left( \frac{1}{\alpha_T} (G_{\mathbf{y},T} + G_{\mathbf{x}\mathbf{y},T} + G_{\mathbf{y}\mathbf{y},T} + D_{\mathbf{y},T} + D_{\mathbf{x},T}) \right). \end{aligned} \quad (116)$$

Let

$$\begin{aligned} G_T &:= G_{\mathbf{y},T} + G_{\mathbf{x}\mathbf{y},T} + G_{\mathbf{y}\mathbf{x},T}, \\ D_T &:= D_{\mathbf{y},T} + D_{\mathbf{x},T}, \\ \sigma^2 &:= \sigma_{g_{\mathbf{y}}}^2 + \sigma_{g_{\mathbf{y}\mathbf{y}}}^2 + \sigma_{f_{\mathbf{y}}}^2 + \sigma_{g_{\mathbf{x}\mathbf{y}}}^2 + \sigma_{f_{\mathbf{x}}}^2, \\ b &= \bar{b} = 1. \end{aligned}$$

By inequalities [\(109\)](#page-36-2), [\(112\)](#page-36-3), [\(113\)](#page-37-0), [\(114\)](#page-38-0), [\(115\)](#page-38-1), [\(116\)](#page-38-2), we have

$$\begin{aligned} & \sum_{t=1}^T \frac{\alpha_t}{2} \mathbb{E} \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 + \Lambda \\ & \leq \mathcal{O} \left( V_T + \frac{H_{2,T}}{\alpha_T} + \frac{\sigma^2}{b} \sum_{t=1}^T \alpha_t^3 + \frac{G_T}{\alpha_T} + \frac{D_T}{\alpha_T} \right). \end{aligned} \quad (117)$$

2154

2156 Using [\(118\)](#page-39-0), we get

2164

2166

2169

2174

2176

2194

2196

From the definition of Λ in [\(102\)](#page-34-7), we have

$$\begin{aligned} -\Lambda &= \Gamma \sum_{t=1}^T (\mathbb{E}[\theta_t^\mathbf{y}] - \mathbb{E}[\theta_{t+1}^\mathbf{y}]) + \Upsilon \sum_{t=1}^T (\mathbb{E}[\theta_t^\mathbf{y}] - \mathbb{E}[\theta_{t+1}^\mathbf{y}]) + \frac{1}{\Phi} \sum_{t=1}^T \left( \frac{\mathbb{E}\|e_t^g\|^2}{\alpha_{t-1}} - \frac{\mathbb{E}\|e_{t+1}^g\|^2}{\alpha_t} \right) \\ &\quad + \frac{1}{\Psi} \sum_{t=1}^T \left( \frac{\mathbb{E}\|e_t^\mathbf{y}\|^2}{\alpha_{t-1}} - \frac{\mathbb{E}\|e_{t+1}^\mathbf{y}\|^2}{\alpha_t} \right) + \frac{1}{\Omega} \sum_{t=1}^T \left( \frac{\mathbb{E}\|e_t^f\|^2}{\alpha_{t-1}} - \frac{\mathbb{E}\|e_{t+1}^f\|^2}{\alpha_t} \right) \\ &\leq \Gamma \theta_1^\mathbf{y} + \Upsilon \theta_1^\mathbf{y} + \frac{\sigma_{gy}^2}{\Phi \alpha_0} + \frac{\sigma_{gyy}^2 + \sigma_{fy}^2}{\Psi \alpha_0} + \frac{\sigma_{gxy}^2 + \sigma_{fx}^2}{\Omega \alpha_0}. \end{aligned} \tag{118}$$

$$\begin{aligned} & \sum_{t=1}^T \frac{\alpha_t}{2} \mathbb{E} \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \\ & \leq \mathcal{O} \left( V_T + \frac{H_{2,T}}{\alpha_T} + \frac{\sigma^2}{b} \sum_{t=1}^T \alpha_t^3 + \frac{G_T}{\alpha_T} + \frac{D_T}{\alpha_T} - \Lambda \right) \\ & \leq \mathcal{O} \left( V_T + \theta_1^\mathbf{y} + \theta_1^\mathbf{y} + \frac{\sigma^2}{b} \sum_{t=1}^T \alpha_t^3 + \frac{H_{2,T}}{\alpha_T} + \frac{G_T}{\alpha_T} + \frac{D_T}{\alpha_T} + \frac{\sigma^2}{\alpha_0} \right). \end{aligned}$$

Since α<sup>t</sup> = 1/(c + t) 1/3 , we get

$$\sum_{t=1}^T \alpha_t^3 = \sum_{t=1}^T \frac{1}{c+t} \leq \sum_{t=1}^T \frac{1}{1+t} \leq \log(T+1),$$

which, combined with the fact that α<sup>t</sup> decreases with respect to t and by multiplying both sides by 2/α<sup>T</sup> , results in Thus, we have

$$\begin{aligned} \text{BL-Reg}_T &= \sum_{t=1}^T \mathbb{E} \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \\ &\leq \mathcal{O}\left(\frac{1}{\alpha_T} (V_T + \|\mathbf{y}_1 - \mathbf{y}_1^*(\mathbf{x}_1)\|^2 + \|\mathbf{v}_1 - \mathbf{v}_1^*(\mathbf{x}_1)\|^2 + \sigma^2 \log(T+1) + \frac{\sigma^2}{\alpha_0})\right. \\ &\quad \left. + \frac{1}{\alpha_T^2} (H_{2,T} + G_T + D_T)\right). \end{aligned}$$

This completes the proof.

# D. Proof of Regret Bounds for Zeroth Order SOGD (ZO-SOGD)

Proof Roadmap. We provide Lemma [D.9,](#page-46-0) which quantifies the error between the approximated direction of the momentumbased gradient estimator, dˆ<sup>y</sup> t and the true direction, ∇ygt,ρ(xt, yt), at each iteration. Lemma [D.11](#page-49-0) assesses the convergence of the iterative solutions {yt} T <sup>t</sup>=1, specifically the gap <sup>E</sup> -∥yt+1 − yˆ ∗ t (xt)∥ 2 , while accounting for the error introduced in Lemma [D.9.](#page-46-0) To establish Lemma [D.15,](#page-56-0) which quantifies the error between the approximated direction of the momentumbased gradient estimator, dˆ<sup>v</sup> t , and the true direction, ∇yft,ρ(xt, yt) + ∇<sup>2</sup> y gt,<sup>ρ</sup> (xt, yt) vt, we need to present Lemma [D.13.](#page-51-0) This lemma quantifies the error between dˆ<sup>v</sup> t and ∇yft,ρ(xt, yt) + <sup>1</sup> 2ρ<sup>v</sup> (∇ygt,ρ(xt, y<sup>t</sup> + ρvvt) − ∇ygt,ρ(xt, y<sup>t</sup> − ρvvt)). Then, Lemma [D.17](#page-59-0) captures the error of the system solution of Problem [\(17\)](#page-4-1), i.e., gap E -∥vt+1 −vˆ ∗ t (xt)∥ 2 , based on these errors. To establish Lemma [D.21,](#page-66-0) which quantifies the error between the approximated direction of the momentum-based hypergradient estimator, dˆ<sup>x</sup> t , and the true direction, ∇xft,ρ(xt, yt)+∇<sup>2</sup> xygt,<sup>ρ</sup> (xt, yt) vt, it is necessary to introduce Lemma [D.19.](#page-61-0) This lemma quantifies the error between dˆ<sup>x</sup> t and ∇xft,ρ(xt, yt) + <sup>1</sup> 2ρ<sup>v</sup> (∇xgt,ρ(xt, y<sup>t</sup> + ρvvt) − ∇xgt,ρ(xt, y<sup>t</sup> − ρvvt)). Then, Lemma [D.22](#page-67-0) bounds the projection mapping based on these errors. By combining these lemmas and setting parameters, we achieve the desired result.

2204

2206

2209

2214

2216

2218 2219

2224

2226

2229

2234

2236 Lemma D.1. *[Allen-Zhu & Li](#page-8-27) [\(2018,](#page-8-27) Lemma A.1.) Suppose Assumption* [B4.](#page-3-13) *holds. Then, for any* x, v ∈ X *, we have:*

2239 2240 2241 Lemma D.2. *Suppose that Assumptions [3.2](#page-3-2) and [3.3](#page-3-3) hold for all* x, x ′ ∈ X *, and* t ∈ [T]*, and that* d x t,<sup>ρ</sup> *and* d v t,<sup>ρ</sup> *are defined in* [\(120\)](#page-40-1)*. Then, we have*

2249 *Here,* vˆ ∗ t (x) *and* ft,ρ*,* yˆ ∗ t (x) *are defined in* [\(119b\)](#page-40-2) *and* [\(17\)](#page-4-1)*, respectively. Moreover, the constants* M<sup>f</sup> *,* Mv*, and* (Ly, Lv, L<sup>f</sup> ) *are defined as in* [\(40\)](#page-15-0)*,* [\(41\)](#page-16-1)*, and* [\(42\)](#page-16-2)*, respectively.*

#### D.1. Auxiliary Lemmas for Proof of Theorem [4.2](#page-5-0)

To solve the online bilevel problem in [\(17\)](#page-4-1), we need to obtain the hyper-gradient of ft,<sup>ρ</sup> in [\(17\)](#page-4-1) at (x, y) as

$$\begin{aligned}\nabla f_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x})) &:= \nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x})) + \nabla_{\mathbf{x}_y^2} g_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x})) \mathbf{v}_t^*(\mathbf{x}), & \text{where} \\ \mathbf{v}_t^*(\mathbf{x}) &:= -[\nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x}))]^{-1} \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x})).\end{aligned}$$

Obtaining yˆ ∗ t (x) in closed-form is usually a challenging task, so it is natural to use the following gradient surrogate. At any (x, y), define:

$$\tilde{\nabla} f_{t,\rho}(\mathbf{x}, \mathbf{y}) := \nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{x}, \mathbf{y}) + \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y}) \hat{\mathbf{v}}_t^*(\mathbf{x}), \quad \text{where} \quad (119a)$$

$$\hat{v}_t^*(\mathbf{x}) := -[\nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y})]^{-1} \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}, \mathbf{y}). \quad (119b)$$

To do so, we also introduce d y t,ρ , d v t,ρ and d x t,ρ as follows:

$$d_{t,\rho}^y(\mathbf{x}, \mathbf{y}) = \nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}, \mathbf{y}), \quad (120a)$$

$$\mathbf{d}_{t,\rho}^{\mathbf{v}}(\mathbf{x}, \mathbf{y}, \mathbf{v}) = \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}, \mathbf{y}) + \nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y}) \mathbf{v}, \quad (120b)$$

$$\mathbf{d}_{t,\rho}^x(\mathbf{x}, \mathbf{y}, \mathbf{v}) = \nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{x}, \mathbf{y}) + \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y}) \mathbf{v}. \quad (120c)$$

To approximate these directions, we use [\(19\)](#page-4-2)-[\(21\)](#page-4-3). It can be shown that ∇ˆ <sup>y</sup>ft(x, y; ξ) and ∇ˆ <sup>x</sup>ft(x, y; ξ) are unbiased estimators of the true gradients ∇yft,ρ(x, y) and ∇xft,ρ(x, y) with respect to y and x [\(Flaxman et al.,](#page-8-23) [2004\)](#page-8-23), respectively, i.e.,

$$\mathbb{E}_{\mathbf{r}} \left[ \hat{\nabla}_{\mathbf{y}} f_t(\mathbf{x}, \mathbf{y}; \xi) \right] = \nabla_{\mathbf{y}} f_{t, \rho}(\mathbf{x}, \mathbf{y}), \quad \mathbb{E}_{\mathbf{z}} \left[ \hat{\nabla}_{\mathbf{x}} f_t(\mathbf{x}, \mathbf{y}; \xi) \right] = \nabla_{\mathbf{x}} f_{t, \rho}(\mathbf{x}, \mathbf{y}),$$

and,

$$\mathbb{E}_{\mathbf{r}} \left[ \hat{\nabla}_{\mathbf{y}} g_t(\mathbf{x}, \mathbf{y}; \zeta) \right] = \nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}, \mathbf{y}), \quad \mathbb{E}_{\mathbf{z}} \left[ \hat{\nabla}_{\mathbf{x}} g_t(\mathbf{x}, \mathbf{y}; \zeta) \right] = \nabla_{\mathbf{x}} g_{t,\rho}(\mathbf{x}, \mathbf{y}). \quad (121)$$

Similarly,

$$\mathbb{E}_{\mathbf{r}} \left[ \hat{\nabla}_{\mathbf{y}} f_t(\mathbf{x}, \mathbf{y}; \mathcal{B}) \right] = \nabla_{\mathbf{y}} f_{t, \rho}(\mathbf{x}, \mathbf{y}), \quad \mathbb{E}_{\mathbf{z}} \left[ \hat{\nabla}_{\mathbf{x}} f_t(\mathbf{x}, \mathbf{y}; \mathcal{B}) \right] = \nabla_{\mathbf{x}} f_{t, \rho}(\mathbf{x}, \mathbf{y}),$$

and,

$$\mathbb{E}_{\mathbf{r}} \left[ \hat{\nabla}_{\mathbf{y}} g_t(\mathbf{x}, \mathbf{y}; \bar{\mathbf{B}}) \right] = \nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}, \mathbf{y}), \quad \mathbb{E}_{\mathbf{z}} \left[ \hat{\nabla}_{\mathbf{x}} g_t(\mathbf{x}, \mathbf{y}; \bar{\mathbf{B}}) \right] = \nabla_{\mathbf{x}} g_{t,\rho}(\mathbf{x}, \mathbf{y}). \quad (122)$$

$$\|\nabla g_t(\mathbf{x} + \mathbf{v}, \mathbf{y} + \mathbf{v}) - \nabla g_t(\mathbf{x}, \mathbf{y}) - \nabla^2 g_t(\mathbf{x}, \mathbf{y})\mathbf{v}\| \leq \ell_{g,2} \|\mathbf{v}\|^2.$$

$$\|\mathbf{d}_{t,\rho}^* - \nabla f_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x}))\|^2 \leq M_f \left( \|\mathbf{y} - \hat{\mathbf{y}}_t^*(\mathbf{x})\|^2 + \|\mathbf{v} - \hat{\mathbf{v}}_t^*(\mathbf{x})\|^2 \right), \quad (123a)$$

$$\|\mathbf{d}_{t,\mathbf{y}}^{\mathbf{r}}\|^2 \leq M_{\mathbf{v}}^2 \left( \|\mathbf{y} - \hat{\mathbf{y}}_t^*(\mathbf{x})\|^2 + \|\mathbf{v} - \hat{\mathbf{v}}_t^*(\mathbf{x})\|^2 \right), \quad (123b)$$

$$\|\nabla f_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x})) - \nabla f_{t,\rho}(\mathbf{x}', \hat{\mathbf{y}}_t^*(\mathbf{x}'))\| \leq L_f \|\mathbf{x} - \mathbf{x}'\|, \quad (123c)$$

$$\|\hat{\mathbf{y}}_t^*(\mathbf{x}) - \hat{\mathbf{y}}_t^*(\mathbf{x}')\| \leq L_y \|\mathbf{x} - \mathbf{x}'\|, \quad (123d)$$

$$\|\hat{\mathbf{y}}_t(\mathbf{x}) - \hat{\mathbf{y}}_t(\mathbf{x}')\| \leq L_{\mathbf{y}} \|\mathbf{x} - \mathbf{x}'\|, \quad (1250)$$

$$\|\hat{\mathbf{v}}_t^*(\mathbf{x}) - \hat{\mathbf{v}}_t^*(\mathbf{x}')\| \leq L_{\mathbf{v}} \|\mathbf{x} - \mathbf{x}'\|. \quad (123e)$$

2259 2260

2264

2266

2269

2274

2276

2279

2281 2282

2289 2290

2294

2296

2299 2300

2304

2306

Using Assumptions [3.2](#page-3-2) and [B1.,](#page-3-18) we have ∇<sup>2</sup> y gt,<sup>ρ</sup> (x, yˆ ∗ t (x)) ⪰ µg, and

$$\|\hat{\mathbf{v}}_t^*(\mathbf{x})\| = \|(\nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x})))^{-1} \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x}))\| \leq \frac{\ell_{f,0}}{\mu_g}. \quad (124)$$

Observe that we have

$$\begin{aligned}
\|\mathbf{d}_{t,\rho}^x - \nabla f_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x}))\| &\leq \|\nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x}))\| \\
&\quad + \|\mathbf{v} \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y}) - \hat{\mathbf{v}}_t^*(\mathbf{x}) \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x}))\| \\
&\leq \|\nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x}))\| \\
&\quad + \|\nabla_{\mathbf{x}\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y})\| \|\mathbf{v} - \hat{\mathbf{v}}_t^*(\mathbf{x})\| \\
&\quad + \|\hat{\mathbf{v}}_t^*(\mathbf{x})\| \|\nabla_{\mathbf{x}\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x}))\| \\
&\leq \left( \ell_{f,1} + \frac{\ell_{g,2} \ell_{f,0}}{\mu_g} \right) \|\mathbf{y} - \hat{\mathbf{y}}_t^*(\mathbf{x})\| + \ell_{g,1} \|\mathbf{v} - \hat{\mathbf{v}}_t^*(\mathbf{x})\| \\
&\leq M_f^2 (\|\mathbf{y} - \hat{\mathbf{y}}_t^*(\mathbf{x})\| + \|\mathbf{v} - \hat{\mathbf{v}}_t^*(\mathbf{x})\|), \tag{125}
\end{aligned}$$

where M<sup>f</sup> is defined as in [\(40\)](#page-15-0); the third inequality is by Assumption [3.3](#page-3-3) and the last inequality is by Eq. [\(124\)](#page-41-0).

We now show Eq. [\(123b\)](#page-40-4).

Since d v∗ t,ρ := ∇yft,ρ(x, yˆ ∗ t (x)) + ∇<sup>2</sup> y gt,<sup>ρ</sup> (x, yˆ ∗ t (x)) vˆ ∗ t (x) = 0, we have

$$\begin{aligned} \|\mathbf{d}_{t,\rho}^{\mathbf{y}}\| &= \|\mathbf{d}_{t,\rho}^{\mathbf{y}*} - \mathbf{d}_{t,\rho}^{\mathbf{y}}\| \\ &= \|\mathbf{v}_{\mathbf{y}} \nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y}) + \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}, \mathbf{y}) \\ &\quad - (\hat{\mathbf{v}}_t^*(\mathbf{x}) \nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x})) + \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x})))\| \\ &\leq \|(\nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x}))) \hat{\mathbf{v}}_t^*(\mathbf{x})\| \\ &\quad + \|\nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y}) (\mathbf{v} - \hat{\mathbf{v}}_t^*(\mathbf{x}))\| \\ &\quad + \|\nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x}))\|. \end{aligned}$$

Then, from Assumption [3.3](#page-3-3) and Eq. [\(124\)](#page-41-0), we have

$$\begin{aligned} \|\mathbf{d}_{t,\rho}^{\mathbf{y}}\| &\leq \ell_{g,2}\|\mathbf{y} - \hat{\mathbf{y}}_t^*(\mathbf{x})\| \|\hat{\mathbf{v}}_t^*(\mathbf{x})\| + \ell_{g,1}\|\mathbf{v} - \hat{\mathbf{v}}_t^*(\mathbf{x})\| + \ell_{f,1}\|\mathbf{y} - \hat{\mathbf{y}}_t^*(\mathbf{x})\| \\ &\leq \left( \frac{\ell_{g,2}\ell_{f,0}}{\mu_g} + \ell_{f,1} \right) \|\mathbf{y} - \hat{\mathbf{y}}_t^*(\mathbf{x})\| + \ell_{g,1}\|\mathbf{v} - \hat{\mathbf{v}}_t^*(\mathbf{x})\| \\ &\leq M_{\mathbf{v}} (\|\mathbf{y} - \hat{\mathbf{y}}_t^*(\mathbf{x})\| + \|\mathbf{v} - \hat{\mathbf{v}}_t^*(\mathbf{x})\|), \end{aligned}$$

where M<sup>v</sup> is defined as in [\(41\)](#page-16-1).

The proofs of Eqs. [\(123c\)](#page-40-5)-[\(123e\)](#page-40-6) follow from [Tarzanagh et al.](#page-10-2) [\(2024,](#page-10-2) Lemma 17) by setting (Ly, Lv, L<sup>f</sup> ) as in [\(42\)](#page-16-2).

# D.2. Perturbation Bounds for OBO Objectives and Their Smoothing Variants

Lemma D.3. *Given* ρ = (ρs, ρr) *as positive smoothing parameters, let* gt,ρ(x, y) *and* ft,ρ(x, y) *be the functions defined by* [\(17\)](#page-4-1)*.*

*(a) Suppose Assumption* [B3.](#page-3-10) *holds. Then, we have*

$$|g_{t,\rho}(\mathbf{x}, \mathbf{y}) - g_t(\mathbf{x}, \mathbf{y})| \leq \frac{\ell_{g,1}(\rho_s^2 + \rho_r^2)}{2}. \quad (126)$$

*(b) Suppose Assumption* [B2.](#page-3-12) *holds. Then, we have*

$$|f_{t,\rho}(\mathbf{x}, \mathbf{y}) - f_t(\mathbf{x}, \mathbf{y})| \leq \frac{\ell_{f,1}(\rho_s^2 + \rho_r^2)}{2}. \quad (127)$$

2316

2318 Thus, we get

2319

2324

2326

2329

2334

2336

2354

2356

2359 2360 2361

and R d<sup>2</sup> , respectively. Then, we have

$$\begin{aligned} & |g_{t,\rho}(\mathbf{x}, \mathbf{y}) - g_t(\mathbf{x}, \mathbf{y})| \\ &= \left| \frac{1}{\mathcal{V}(d_1)\mathcal{V}(d_2)} \int_{B_1} \int_{B_2} (g_t(\mathbf{x} + \rho_s \mathbf{s}, \mathbf{y} + \rho_r \mathbf{r}) - g_t(\mathbf{x}, \mathbf{y})) \, ds d\mathbf{r} \right| \\ &= \left| \frac{1}{\mathcal{V}(d_1)\mathcal{V}(d_2)} \int_{B_1} \int_{B_2} (g_t(\mathbf{x} + \rho_s \mathbf{s}, \mathbf{y} + \rho_r \mathbf{r}) - g_t(\mathbf{x}, \mathbf{y}) - \langle \nabla g_t(\mathbf{x}, \mathbf{y}), (\rho_s \mathbf{s}, \rho_r \mathbf{r}) \rangle) \, ds d\mathbf{r} \right|. \end{aligned}$$

$$\begin{aligned}
& |g_{t,\rho}(\mathbf{x}, \mathbf{y}) - g_t(\mathbf{x}, \mathbf{y})| \\
& \leq \int_{B_1} \int_{B_2} |g_t(\mathbf{x} + \rho_s \mathbf{s}, \mathbf{y} + \rho_r \mathbf{r}) - g_t(\mathbf{x}, \mathbf{y}) - \langle \nabla g_t(\mathbf{x}, \mathbf{y}), (\rho_s \mathbf{s}, \rho_r \mathbf{r}) \rangle| \, ds dr \\
& \leq \int_{B_1} \int_{B_2} \frac{\ell_{g,1}}{2} (\rho_s^2 \|\mathbf{s}\|^2 + \rho_r^2 \|\mathbf{r}\|^2) \, ds dr \\
& = \frac{\ell_{g,1} \rho_s^2}{2} \int_{B_1} \|\mathbf{s}\|^2 d\mathbf{s} + \frac{\ell_{g,1} \rho_r^2}{2} \int_{B_2} \|\mathbf{r}\|^2 d\mathbf{r} \\
& = \frac{\ell_{g,1} \rho_s^2}{2} \frac{d_1}{d_1 + 2} + \frac{\ell_{g,1} \rho_r^2}{2} \frac{d_2}{d_2 + 2} \\
& \leq \frac{\ell_{g,1} (\rho_s^2 + \rho_r^2)}{2},
\end{aligned}$$

where the last equality follows since <sup>1</sup> V(d) R s∈B ∥s∥ <sup>p</sup>ds = d d+p . The proof of part (b) follows using similar arguments.

Lemma D.4. *Given* ρ = (ρs, ρr) *as positive smoothing parameters, let* gt,ρ(x, y) *and* ft,ρ(x, y) *be the functions defined by* [\(17\)](#page-4-1)*.*

*(a) Suppose Assumption* [B3.](#page-3-10) *holds. Then, we have*

$$\|\nabla g_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla g_t(\mathbf{x}, \mathbf{y})\| \leq \frac{\ell_{g,1}(\rho_{\mathbf{s}}d_1 + \rho_{\mathbf{r}}d_2)}{2}. \quad (128)$$

*(b) Suppose Assumption* [B2.](#page-3-12) *holds. Then, we have*

$$\|\nabla f_t(\mathbf{x}, \mathbf{y}) - \nabla f_{t,\rho}(\mathbf{x}, \mathbf{y})\| \leq \frac{\ell_{f,1}(\rho_{\mathbf{s}}d_1 + \rho_{\mathbf{r}}d_2)}{2}. \quad (129)$$

2369

2374

2376

2379

2389 2390

2394

2396

*Proof.* Let S(d1) be the surface area of the unit sphere in <sup>R</sup> d<sup>1</sup> . Moreover, let U<sup>B</sup><sup>1</sup> be the unit sphere.

$$\begin{aligned}
& \|\nabla_{\mathbf{x}} g_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{x}} g_t(\mathbf{x}, \mathbf{y})\| \\
&= \left\| \frac{1}{S(d_1)} \left( \frac{d_1}{\rho_s} \int_{U_{B_1}} g_t(\mathbf{x} + \rho_s \mathbf{s}, \mathbf{y}) \mathrm{d}\mathbf{s} \right) - \nabla_{\mathbf{x}} g_t(\mathbf{x}, \mathbf{y}) \right\| \\
&= \left\| \frac{1}{S(d_1)} \left( \frac{d_1}{\rho_s} \int_{U_{B_1}} g_t(\mathbf{x} + \rho_s \mathbf{s}, \mathbf{y}) \mathrm{d}\mathbf{s} - \int_{U_{B_1}} \frac{d_1}{\rho_s} g_t(\mathbf{x}, \mathbf{y}) \mathrm{d}\mathbf{s} \right. \right. \\
&\quad \left. \left. - \int_{U_{B_1}} \frac{d_1}{\rho_s} \langle \nabla_{\mathbf{x}} g_t(\mathbf{x}, \mathbf{y}), \rho_s \mathbf{s} \rangle \mathrm{d}\mathbf{s} \right) \right\| \\
&\leq \frac{d_1}{S(d_1)\rho_s} \int_{U_{B_1}} |g_t(\mathbf{x}_t + \rho_s \mathbf{s}, \mathbf{y}) - g_t(\mathbf{x}, \mathbf{y}) - \langle \nabla_{\mathbf{x}} g_t(\mathbf{x}, \mathbf{y}), \rho_s \mathbf{s} \rangle| \|\mathbf{s}\| \, \mathrm{d}\mathbf{s} \\
&\leq \frac{d_1}{S(d_1)\rho_s} \cdot \frac{\ell_{g,1} \rho_s^2}{2} \int_{U_{B_1}} \|\mathbf{s}\|^3 \, \mathrm{d}\mathbf{s} \\
&= \frac{\rho_s d_1 \ell_{g,1}}{2}, \tag{130}
\end{aligned}$$

where the second equality follows from R UB<sup>1</sup> ss⊤ds = S(d1) d<sup>1</sup> I.

Similarly, let S(d2) be the surface area of the unit sphere in <sup>R</sup> d<sup>2</sup> . Moreover, let U<sup>B</sup><sup>2</sup> be the unit sphere.

$$\begin{aligned}
& \|\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{y}} g_t(\mathbf{x}, \mathbf{y})\| \\
&= \left\| \frac{1}{S(d_2)} \left( \frac{d_2}{\rho_{\mathbf{r}}} \int_{U_{B_2}} g_t(\mathbf{x}, \mathbf{y} + \rho_{\mathbf{r}}\mathbf{r}) \mathbf{r} d\mathbf{r} \right) - \nabla_{\mathbf{y}} g_t(\mathbf{x}, \mathbf{y}) \right\| \\
&= \left\| \frac{1}{S(d_2)} \left( \frac{d_2}{\rho_{\mathbf{r}}} \int_{U_{B_2}} g_t(\mathbf{x}, \mathbf{y} + \rho_{\mathbf{r}}\mathbf{r}) \mathbf{r} d\mathbf{r} - \int_{U_{B_2}} \frac{d_2}{\rho_{\mathbf{r}}} g_t(\mathbf{x}, \mathbf{y}) \mathbf{r} d\mathbf{r} \right. \right. \\
&\quad \left. \left. - \int_{U_{B_2}} \frac{d_2}{\rho_{\mathbf{r}}} \langle \nabla_{\mathbf{y}} g_t(\mathbf{x}, \mathbf{y}), \rho_{\mathbf{r}}\mathbf{r} \rangle \mathbf{r} d\mathbf{r} \right) \right\| \\
&\leq \frac{d_2}{S(d_2)\rho_{\mathbf{r}}} \int_{U_{B_2}} |g_t(\mathbf{x}_t, \mathbf{y} + \rho_{\mathbf{r}}\mathbf{r}) - g_t(\mathbf{x}, \mathbf{y}) - \langle \nabla_{\mathbf{y}} g_t(\mathbf{x}, \mathbf{y}), \rho_{\mathbf{r}}\mathbf{r} \rangle| \|\mathbf{r}\| d\mathbf{r} \\
&\leq \frac{d_2}{S(d_2)\rho_{\mathbf{r}}} \cdot \frac{\ell_{g,1}\rho_{\mathbf{r}}^2}{2} \int_{U_{B_2}} \|\mathbf{r}\|^3 d\mathbf{r} \\
&= \frac{\rho_{\mathbf{r}} d_2 \ell_{g,1}}{2}, \tag{131}
\end{aligned}$$

where the second equality follows from R UB<sup>2</sup> rr⊤dr = S(d2) d<sup>2</sup> I.

Thus, we get

$$\begin{aligned} & \|\nabla g_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla g_t(\mathbf{x}, \mathbf{y})\| \\ & \leq \|\nabla_{\mathbf{x}} g_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{x}} g_t(\mathbf{x}, \mathbf{y})\| + \|\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{y}} g_t(\mathbf{x}, \mathbf{y})\| \\ & \leq \frac{\rho_s d_1 \ell_{g,1}}{2} + \frac{\rho_r d_2 \ell_{g,1}}{2}. \end{aligned}$$

Finally, by a similar argument as in Part (a), we obtain

$$\|\nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{x}} f_t(\mathbf{x}, \mathbf{y})\| \leq \frac{\rho_{\mathbf{s}} d_1 \ell_{f,1}}{2}, \quad (132)$$

and

$$\|\nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{y}} f_t(\mathbf{x}, \mathbf{y})\| \leq \frac{\rho_r d_2 \ell_{f,1}}{2}, \quad (133)$$

which implies

$$\|\nabla f_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla f_t(\mathbf{x}, \mathbf{y})\| \leq \frac{(\rho_s d_1 + \rho_r d_2)\ell_{f,1}}{2}.$$

Lemma D.5. *Suppose Assumption* [B4.](#page-3-13) *holds. Given* ρ = (ρs, ρr) *as positive smoothing parameters, let* gt,ρ(x, y) *be the function defined in* [\(17\)](#page-4-1)*. Then, we have*

$$\|\nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{y}}^2 g_t(\mathbf{x}, \mathbf{y})\|^2 \leq \frac{d_2^2 \ell_{g,2}^2}{4} \rho_{\mathbf{r}}^2.$$

*Proof.* Similary, let S(d2) be the surface area of the unit sphere in <sup>R</sup> d<sup>2</sup> . Moreover, let U<sup>B</sup><sup>2</sup> be the unit sphere.

$$\begin{aligned}
& \left\| \nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{y}}^2 g_t(\mathbf{x}, \mathbf{y}) \right\| \\
&= \left\| \frac{1}{S(d_2)} \left( \frac{d_2}{\rho_{\mathbf{r}}} \int_{U_{B_2}} \nabla_{\mathbf{y}} g_t(\mathbf{x}, \mathbf{y} + \rho_{\mathbf{r}} \mathbf{r}) \mathbf{r} d\mathbf{r} \right) - \nabla_{\mathbf{y}}^2 g_t(\mathbf{x}, \mathbf{y}) \right\| \\
&= \left\| \frac{1}{S(d_2)} \left( \frac{d_2}{\rho_{\mathbf{r}}} \int_{U_{B_2}} \nabla_{\mathbf{y}} g_t(\mathbf{x}, \mathbf{y} + \rho_{\mathbf{r}} \mathbf{r}) \mathbf{r} d\mathbf{r} - \int_{U_{B_2}} \frac{d_2}{\rho_{\mathbf{r}}} \nabla_{\mathbf{y}} g_t(\mathbf{x}, \mathbf{y}) \mathbf{r} d\mathbf{r} \right. \right. \\
&\quad \left. \left. - \int_{U_{B_2}} \frac{d_2}{\rho_{\mathbf{r}}} \langle \nabla_{\mathbf{y}}^2 g_t(\mathbf{x}, \mathbf{y}), \rho_{\mathbf{r}} \mathbf{r} \rangle \mathbf{r} d\mathbf{r} \right) \right\| \\
&\leq \frac{d_2}{S(d_2) \rho_{\mathbf{r}}} \int_{U_{B_2}} |\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y} + \rho_{\mathbf{r}} \mathbf{r}) - \nabla_{\mathbf{y}} g_t(\mathbf{x}, \mathbf{y}) - \langle \nabla_{\mathbf{y}}^2 g_t(\mathbf{x}, \mathbf{y}), \rho_{\mathbf{r}} \mathbf{r} \rangle| \|\mathbf{r}\| d\mathbf{r} \\
&\leq \frac{d_2}{S(d_2) \rho_{\mathbf{r}}} \cdot \frac{\ell_{g,2} \rho_{\mathbf{r}}^2}{2} \int_{U_{B_2}} \|\mathbf{r}\|^3 d\mathbf{r} \\
&= \frac{\rho_{\mathbf{r}} d_2 \ell_{g,2}}{2},
\end{aligned}$$

where the second equality follows from R UB<sup>2</sup> rr⊤dr = S(d2) d<sup>2</sup> I.

Lemma D.6. *Suppose Assumption* [B3.](#page-3-10) *holds. Let* ∇ˆ <sup>y</sup>gt(x, y; B¯) *and* ∇ˆ <sup>x</sup>gt(x, y; B¯) *be defined as in* [\(19a\)](#page-4-10) *and* [\(19b\)](#page-4-11)*, respectively. Then, for any* (x, y) ∈ <sup>R</sup> <sup>d</sup><sup>1</sup> × <sup>R</sup> <sup>d</sup><sup>2</sup> *and* ρr, ρ<sup>s</sup> ≥ 0*, we have*

$$\mathbb{E}_{\mathbf{r}} \left[ \left\| \hat{\nabla}_{\mathbf{y}} g_t(\mathbf{x}, \mathbf{y}; \bar{\mathbf{B}}) - \hat{\nabla}_{\mathbf{y}} g_t(\mathbf{x}, \hat{\mathbf{y}}; \bar{\mathbf{B}}) \right\|^2 \right] \leq 3d_2 \ell_{g,1}^2 \|\mathbf{y} - \hat{\mathbf{y}}\|^2 + \frac{3\ell_{g,1}^2 d_2^2 \rho_{\mathbf{r}}^2}{2} \quad \forall \hat{\mathbf{y}} \in \mathbb{R}^{d_2}, \quad (134a)$$

$$\mathbb{E}_{\mathbf{z}} \left[ \left\| \hat{\nabla}_{\mathbf{x}} g_t(\mathbf{x}, \mathbf{y}; \bar{\mathcal{B}}) - \hat{\nabla}_{\mathbf{x}} g_t(\mathbf{x}, \mathbf{y}; \bar{\mathcal{B}}) \right\|^2 \right] \leq 3d_1 \ell_{g,1}^2 \|\mathbf{x} - \mathbf{x}\|^2 + \frac{3\ell_{g,1}^2 d_1^2 \rho_s^2}{2} \quad \forall \mathbf{x} \in \mathbb{R}^{d_1}. \quad (134b)$$

*Proof.* The proof is similar to that of Lemma 5 in [\(Ji et al.,](#page-9-27) [2019\)](#page-9-27).

Lemma D.7. *Suppose Assumptions [3.2](#page-3-2) and* [B3.](#page-3-10) *hold. Let* ρ = (ρs, ρr) *be positive smoothing parameters. Let* y ∗ t (x) *and* yˆ ∗ t (x) *be defined in* [\(1\)](#page-1-3) *and* [\(18\)](#page-4-7)*, respectively. Then, we have*

$$\mathbb{E} \left[ \|\hat{\mathbf{y}}_t^*(\mathbf{x}) - \mathbf{y}_t^*(\mathbf{x})\|^2 \right] \leq \frac{\ell_{g,1}(\rho_s^2 + \rho_r^2)}{\mu_g}. \quad (135)$$

2487 2488 Since by Assumption [3.2,](#page-3-2) gt,<sup>ρ</sup> (x, y) is µg-strongly convex in terms of y. Then, from Lemma [B.2,](#page-11-2) we have

2504

2506

2509

2514

2516

2518 2519

2524

2526

terms of y. Then, by Lemma [B.2,](#page-11-2) we get

$$\|\mathbf{y} - \mathbf{y}_t^*(\mathbf{x})\|^2 \leq \frac{2}{\mu_g} (g_t(\mathbf{x}, \mathbf{y}) - g_t(\mathbf{x}, \mathbf{y}_t^*(\mathbf{x}))).$$

By setting y = yˆ ∗ t (x), we have

$$\|\hat{\mathbf{y}}_t^*(\mathbf{x}) - \mathbf{y}_t^*(\mathbf{x})\|^2 \leq \frac{2}{\mu_g} (g_t(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x})) - g_t(\mathbf{x}, \mathbf{y}_t^*(\mathbf{x}))). \quad (136)$$

Similarly, from [\(18\)](#page-4-7), we have

$$\hat{\mathbf{y}}_t^*(\mathbf{x}) \in \arg \min_{\mathbf{y} \in \mathbb{R}^{d_2}} \{g_{t,\rho}(\mathbf{x}, \mathbf{y}) = \mathbb{E}_{(\mathbf{z}, \mathbf{r})} [g_t(\mathbf{x} + \rho_s \mathbf{s}, \mathbf{y} + \rho_r \mathbf{r}; \zeta)]\}.$$

$$\|\mathbf{y} - \hat{\mathbf{y}}_t^*(\mathbf{x})\|^2 \leq \frac{2}{\mu_g} (g_{t,\rho}(\mathbf{x}, \mathbf{y}) - g_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x}))).$$

By setting y = y ∗ t (x), we have

$$\|\mathbf{y}_t^*(\mathbf{x}) - \hat{\mathbf{y}}_t^*(\mathbf{x})\|^2 \leq \frac{2}{\mu_g} (g_{t,\rho}(\mathbf{x}, \mathbf{y}_t^*(\mathbf{x})) - g_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x}))). \quad (137)$$

Summing up [\(136\)](#page-45-0) and [\(137\)](#page-45-1), we get

$$\begin{aligned} \|\mathbf{y}_t^*(\mathbf{x}) - \hat{\mathbf{y}}_t^*(\mathbf{x})\|^2 &\leq \frac{1}{\mu_g} (g_{t,\rho}(\mathbf{x}, \mathbf{y}_t^*(\mathbf{x})) - g_t(\mathbf{x}, \mathbf{y}_t^*(\mathbf{x}))) \\ &+ \frac{1}{\mu_g} (g_t(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x})) - g_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x}))), \end{aligned}$$

which implies

$$\begin{aligned} \|\mathbf{y}_t^*(\mathbf{x}) - \hat{\mathbf{y}}_t^*(\mathbf{x})\|^2 &\leq \frac{1}{\mu_g} |g_{t,\rho}(\mathbf{x}, \mathbf{y}_t^*(\mathbf{x})) - g_t(\mathbf{x}, \mathbf{y}_t^*(\mathbf{x}))| \\ &\quad + \frac{1}{\mu_g} |g_t(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x})) - g_{t,\rho}(\mathbf{x}, \hat{\mathbf{y}}_t^*(\mathbf{x}))| \\ &\leq \frac{\ell_{g,1}(\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2)}{\mu_g}, \end{aligned}$$

where the last inequality is by Eq. [\(126\)](#page-41-1).

Lemma D.8. *Suppose Assumptions [3.2](#page-3-2) and [3.3](#page-3-3) hold. Let* v ∗ t (x) *and* vˆ ∗ t (x) *be defined in* [\(4b\)](#page-2-4) *and* [\(119b\)](#page-40-2)*, respectively. Then, we have*

$$\mathbb{E} \left[ \|\hat{\mathbf{v}}_t^*(\mathbf{x}) - \mathbf{v}_t^*(\mathbf{x})\|^2 \right] \leq \frac{d_2^2}{2\mu_g^4} (\ell_{f,1}^2 \mu_g^2 + \ell_{g,2}^2 \ell_{f,0}^2) \rho_{\mathbf{r}}. \quad (138)$$

*Proof.* From [\(4b\)](#page-2-4) and [\(119b\)](#page-40-2), we have

$$\begin{aligned} & \|\hat{\mathbf{v}}_t^*(\mathbf{x}) - \mathbf{v}_t^*(\mathbf{x})\|^2 \\ &= \|[\nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y})]^{-1} \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}, \mathbf{y}) - [\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}, \mathbf{y})]^{-1} \nabla_{\mathbf{y}} f_t(\mathbf{x}, \mathbf{y})\|^2 \\ &\leq 2 \|[\nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y})]^{-1} \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}, \mathbf{y}) - [\nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y})]^{-1} \nabla_{\mathbf{y}} f_t(\mathbf{x}, \mathbf{y})\|^2 \\ &+ 2 \|[\nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y})]^{-1} \nabla_{\mathbf{y}} f_t(\mathbf{x}, \mathbf{y}) - [\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}, \mathbf{y})]^{-1} \nabla_{\mathbf{y}} f_t(\mathbf{x}, \mathbf{y})\|^2. \end{aligned} \tag{139a}$$
(139b)

2536

2543 which implies

2554

2556

2558 2559 2560

2564

2566

2569

2574

2576

2579 *Proof.* From the definition of dˆ<sup>y</sup> <sup>t</sup>+1 in Algorithm [2,](#page-5-2) we have

Next, we separately bound each of the above terms, [\(139a\)](#page-45-2) and [\(139b\)](#page-45-3).

$$\begin{aligned}
 (139a) &\leq 2 \left\| [\nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y})]^{-1} \right\|^2 \left\| \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{y}} f_t(\mathbf{x}, \mathbf{y}) \right\|^2 \\
 &\leq \frac{2}{\mu_g^2} \left\| \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{y}} f_t(\mathbf{x}, \mathbf{y}) \right\|^2 \\
 &\leq \frac{2}{\mu_g^2} \frac{\rho_{\Gamma}^2 d_2^2 \ell_{f,1}^2}{4}, \tag{140}
 \end{aligned}$$

where the second inequality holds due to the Assumption [3.2,](#page-3-2) the third inequality is by [\(133\)](#page-44-0).

To bound [\(139b\)](#page-45-3), note that for any invertible matrices A<sup>1</sup> and A2, we have

$$\|\mathbf{A}_2^{-1} - \mathbf{A}_1^{-1}\| = \|\mathbf{A}_1^{-1}(\mathbf{A}_1 - \mathbf{A}_2)\mathbf{A}_2^{-1}\| \leq \|\mathbf{A}_1^{-1}\| \|\mathbf{A}_2^{-1}\| \|\mathbf{A}_1 - \mathbf{A}_2\|,$$

$$\begin{aligned}
 (139b) &\leq 2 \left\| [\nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y})]^{-1} - [\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}, \mathbf{y})]^{-1} \right\|^2 \left\| \nabla_{\mathbf{y}} f_t(\mathbf{x}, \mathbf{y}) \right\|^2 \\
 &\leq 2 \left\| [\nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y})]^{-1} \right\|^2 \left\| [\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}, \mathbf{y})]^{-1} \right\|^2 \\
 &\left\| [\nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y}) - \nabla_{\mathbf{y}}^2 g_t(\mathbf{x}, \mathbf{y})] \right\|^2 \left\| \nabla_{\mathbf{y}} f_t(\mathbf{x}, \mathbf{y}) \right\|^2 \\
 &\leq \frac{2}{\mu_g^4} \frac{\rho_{\mathbf{r}}^2 d_2^2 \ell_{g,2}^2}{4} \ell_{f,0}^2,
 \end{aligned} \tag{141}$$

where the last inequality follows from Lemma [D.5.](#page-44-1)

Using [\(139\)](#page-45-4)– [\(140\)](#page-46-1), we obtain the desired result.

# D.3. Bounds on the Zeroth-Order Inner Solution

Lemma D.9. *Suppose that Assumptions* [B3.](#page-3-10) *and* [D1.](#page-4-12) *hold. Consider the sequence* {(xt, yt, vt)} T <sup>t</sup>=1 *generated by Algorithm [2,](#page-5-2) and define*

$$e_t^{g_\rho} := \nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \hat{\mathbf{d}}_t^{\mathbf{y}}. \quad (142)$$

*Then, we have*

$$\begin{aligned} \mathbb{E}\|e_{t+1}^{g\rho}\|^2 &\leq (1 - \gamma_{t+1})^2 \mathbb{E}\|e_t^{g\rho}\|^2 + 12(1 - \gamma_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_{t-1}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ &\quad + 9d_2^2 \ell_{g,1}^2 (1 - \gamma_{t+1})^2 \rho_{\mathbf{r}}^2 + 24d_2 \ell_{g,1}^2 (1 - \gamma_{t+1})^2 \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 \\ &\quad + 24d_2 \ell_{g,1}^2 (1 - \gamma_{t+1})^2 \mathbb{E}\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 + 2 \frac{\hat{\sigma}_{gy}^2}{b} \gamma_{t+1}^2. \end{aligned} \quad (143)$$

$$\begin{aligned}\hat{\mathbf{d}}_{t+1}^{\mathbf{y}} - \hat{\mathbf{d}}_t^{\mathbf{y}} &= -\gamma_{t+1}\hat{\mathbf{d}}_t^{\mathbf{y}} + \gamma_{t+1}\hat{\nabla}_{\mathbf{y}}g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1}) \\ &\quad + (1 - \gamma_{t+1}) \left( \hat{\nabla}_{\mathbf{y}}g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \hat{\nabla}_{\mathbf{y}}g_{t+1}(\mathbf{x}_t, \mathbf{y}_t; \bar{\mathcal{B}}_{t+1}) \right).\end{aligned}$$

2585 Then, we have

$$\begin{aligned}
& 2586 \\
& 2587 & \mathbb{E} \|\nabla_{\mathbf{y}} g_{t+1,\rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \hat{\mathbf{a}}_{t+1}^{\mathbf{y}}\|^2 \\
& 2588 & = \mathbb{E} \|\nabla_{\mathbf{y}} g_{t+1,\rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \hat{\mathbf{a}}_t^{\mathbf{y}} - (\hat{\mathbf{a}}_{t+1}^{\mathbf{y}} - \hat{\mathbf{a}}_t^{\mathbf{y}})\|^2 \\
& 2589 & = \mathbb{E} \|\nabla_{\mathbf{y}} g_{t+1,\rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \hat{\mathbf{a}}_t^{\mathbf{y}} + \gamma_{t+1} \hat{\mathbf{a}}_t^{\mathbf{y}} - \gamma_{t+1} \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1}) \\
& 2590 & - (1 - \gamma_{t+1}) \left( \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t; \bar{\mathcal{B}}_{t+1}) \right) \|^2 \\
& 2591 & = \mathbb{E} \|(1 - \gamma_{t+1})(\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \hat{\mathbf{a}}_t^{\mathbf{y}}) \\
& 2592 & + \gamma_{t+1}(\nabla_{\mathbf{y}} g_{t+1,\rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1})) \\
& 2593 & + (1 - \gamma_{t+1})(\nabla_{\mathbf{y}} g_{t+1,\rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) \\
& 2594 & + (1 - \gamma_{t+1})(\nabla_{\mathbf{y}} g_{t+1,\rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1,\rho}(\mathbf{x}_t, \mathbf{y}_t)) \\
& 2595 & + (1 - \gamma_{t+1})(\nabla_{\mathbf{y}} g_{t+1,\rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) \\
& 2596 & + \nabla_{\mathbf{y}} g_{t+1,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} g_{t+1,\rho}(\mathbf{x}_t, \mathbf{y}_t) \\
& 2597 & - \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1}) + \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t; \bar{\mathcal{B}}_{t+1}) \|^2. \\
& 2598 & \\
& 2599
\end{aligned}$$

2600 From [\(122\)](#page-40-7), we have

2606 2607 then, we have

$$\begin{aligned}
& 2608 \\
& 2609 \quad \mathbb{E} \|\nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \hat{\mathbf{a}}_{t+1}^{\mathbf{y}}\|^2 \\
& 2610 \quad = (1 - \gamma_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{y}} g_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t) - \hat{\mathbf{a}}_t^{\mathbf{y}}\|^2 \\
& 2611 \quad + \mathbb{E} \|\gamma_{t+1}(\nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1})) \\
& 2612 \quad + (1 - \gamma_{t+1}) (\nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} g_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t) + \nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_t, \mathbf{y}_t) \\
& 2613 \quad - \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1}) + \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t; \bar{\mathcal{B}}_{t+1}) \Big) \|^2 \\
& 2614 \quad - \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1}) \Big) \|^2 \\
& 2615 \quad \leq (1 - \gamma_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{y}} g_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t) - \hat{\mathbf{a}}_t^{\mathbf{y}}\|^2 \\
& 2616 \quad + 2(1 - \gamma_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} g_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t) + \nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_t, \mathbf{y}_t) \\
& 2617 \quad - \nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_t, \mathbf{y}_t) - \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1}) + \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t; \bar{\mathcal{B}}_{t+1}) \|^2 \\
& 2618 \quad + 2\gamma_{t+1}^2 \mathbb{E} \|\nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1}) \|^2, \\
& 2620
\end{aligned}$$

$$\begin{aligned}
& \mathbb{E} \|\nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \hat{\mathbf{a}}_{t+1}^{\mathbf{y}}\|^2 \\
& \leq (1 - \gamma_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{y}} g_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t) - \hat{\mathbf{a}}_t^{\mathbf{y}}\|^2 \\
& + 4(1 - \gamma_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} g_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\
& + 4(1 - \gamma_{t+1})^2 \mathbb{E} \|\hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathbf{B}}_{t+1}) - \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t; \bar{\mathbf{B}}_{t+1})\|^2 + 2\gamma_{t+1}^2 \frac{\hat{\sigma}_{g_{\mathbf{y}}}^2}{b} \\
& \leq (1 - \gamma_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{y}} g_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t) - \hat{\mathbf{a}}_t^{\mathbf{y}}\|^2 \\
& + 4(1 - \gamma_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} g_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\
& + 12(1 - \gamma_{t+1})^2 d_2 \ell_{g, 1}^2 \mathbb{E} \|(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - (\mathbf{x}_t, \mathbf{y}_t)\|^2 \\
& + 3(1 - \gamma_{t+1})^2 \ell_{g, 1}^2 d_2^2 \rho_{\mathbf{r}}^2 + 2\gamma_{t+1}^2 \frac{\hat{\sigma}_{g_{\mathbf{y}}}^2}{b},
\end{aligned}$$

$$\begin{aligned}\mathbb{E} \left[ \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1}) \right] &= \nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}), \\ \mathbb{E} \left[ \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t; \bar{\mathcal{B}}_{t+1}) \right] \\ &= \nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_t, \mathbf{y}_t),\end{aligned}$$

where the second inequality holds by Cauchy-Schwarz inequality.

Then, from <sup>E</sup>∥a − <sup>E</sup>[a]∥ <sup>2</sup> = <sup>E</sup>∥a∥ <sup>2</sup> − ∥E[a]∥ 2 and Assumption [4.1,](#page-4-6) we have

$$\begin{aligned} & 2641 \\ & 2642 & \mathbb{E}\|\nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} g_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & 2643 \leq 3\mathbb{E}\|\nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & 2644 + 3\mathbb{E}\|\nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & 2645 + 3\mathbb{E}\|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} g_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & 2646 \\ & 2647 \leq 3\mathbb{E}\|\nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 + \frac{3\rho_{\mathbf{r}}^2 d_2^2 \ell_{g,1}^2}{2} \\ & 2648 \end{aligned}$$

2649 Finally, we get

$$\begin{aligned} & \mathbb{E}\|\nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \hat{\mathbf{a}}_{t+1}^{\mathbf{y}}\|^2 \leq (1 - \gamma_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t) - \hat{\mathbf{a}}_t^{\mathbf{y}}\|^2 \\ & + 12(1 - \gamma_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 + 6(1 - \gamma_{t+1})^2 \rho_{\mathbf{r}}^2 d_2^2 \ell_{g, 1}^2 \\ & + 12(1 - \gamma_{t+1})^2 d_2 \ell_{g, 1}^2 \mathbb{E}\|(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - (\mathbf{x}_t, \mathbf{y}_t)\|^2 + 3(1 - \gamma_{t+1})^2 \ell_{g, 1}^2 d_2^2 \rho_{\mathbf{r}}^2 + 2\gamma_{t+1}^2 \frac{\partial_{\mathbf{y}}^2}{\theta}. \end{aligned}$$

2656

2659 Lemma D.10. *Suppose Assumptions [3.2](#page-3-2) and* [B3.](#page-3-10) *hold. Then, for the sequence* {(xt, yt)} T <sup>t</sup>=1 *generated by Algorithm [2,](#page-5-2) we have*

2674

2676

2679

2689 2690

From Eq. [\(131\)](#page-43-0), we have

$$\begin{aligned} \mathbb{E} [\|\mathbf{y}_{t+1} - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2] &\leq (1+a) \left( 1 - 2\beta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \right) \mathbb{E} [\|\mathbf{y}_t - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2] \\ &\quad + \left( -(1+a) \left( \frac{2\beta_t}{\mu_g + \ell_{g,1}} - \beta_t^2 \right) \right) \mathbb{E} [\|\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2] \\ &\quad + (1 + \frac{1}{a}) \beta_t^2 \mathbb{E} [\|e_t^{g\rho}\|^2], \end{aligned}$$

*where* a > 0 *is a constant,* e g<sup>ρ</sup> t *is defined in* [\(142\)](#page-46-2)*, and* yˆ ∗ t (xt) *is defined in* [\(18\)](#page-4-7)*.*

*Proof.* From Lemma [B.5,](#page-12-0) we have

$$\begin{aligned}\mathbb{E} \left[ \|\mathbf{y}_{t+1} - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] &= \mathbb{E} \left[ \|\mathbf{y}_t - \beta_t \hat{\mathbf{a}}_t^{\mathbf{y}} - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] \\ &\leq (1+a)\mathbb{E} \left[ \|\mathbf{y}_t - \beta_t \nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] \\ &\quad + (1+\frac{1}{a})\beta_t^2 \mathbb{E} \left[ \|\hat{\mathbf{a}}_t^{\mathbf{y}} - \nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \right].\end{aligned}\tag{144}$$

Next, we will separately bound the first term on the RHS of the above inequality.

We have

$$\begin{aligned} \mathbb{E} \left[ \|\mathbf{y}_t - \beta_t \nabla_{\mathbf{y}} g_{t,\boldsymbol{\rho}}(\mathbf{x}_t, \mathbf{y}_t) - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] &= \mathbb{E} \left[ \|\mathbf{y}_t - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] + \beta_t^2 \mathbb{E} \left[ \|\nabla_{\mathbf{y}} g_{t,\boldsymbol{\rho}}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \right] \\ &\quad - 2\beta_t \mathbb{E} \left[ \langle \nabla_{\mathbf{y}} g_{t,\boldsymbol{\rho}}(\mathbf{x}_t, \mathbf{y}_t), \mathbf{y}_t - \hat{\mathbf{y}}_t^*(\mathbf{x}_t) \rangle \right] \\ &\leq \left( 1 - 2\beta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \right) \mathbb{E} \left[ \|\mathbf{y}_t - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] \\ &\quad - \left( \frac{2\beta_t}{\mu_g + \ell_{g,1}} - \beta_t^2 \right) \mathbb{E} \left[ \|\nabla_{\mathbf{y}} g_{t,\boldsymbol{\rho}}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \right], \end{aligned} \tag{145}$$

where the inequality results from the strong convexity of gt,<sup>ρ</sup> by Assumption [3.2,](#page-3-2) which implies

$$\langle \nabla_{\mathbf{y}} g_{t, \boldsymbol{\rho}}(\mathbf{x}_t, \mathbf{y}_t), \mathbf{y}_t - \hat{\mathbf{y}}_t^*(\mathbf{x}_t) \rangle \geq \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \|\mathbf{y}_t - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 + \frac{1}{\mu_g + \ell_{g,1}} \|\nabla_{\mathbf{y}} g_{t, \boldsymbol{\rho}}(\mathbf{x}_t, \mathbf{y}_t)\|^2.$$

Substituting [\(145\)](#page-48-0) into [\(144\)](#page-48-1), gives the desired result.

2699 2700

2704

2706

2709

2714

2716

2718 2719

2724

2726

2729

2734

2736

For notational brevity in the analysis, we define

$$\hat{\theta}_t^y := \|\mathbf{y}_t - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2, \quad \hat{\theta}_t^y := \|\mathbf{v}_t - \hat{\mathbf{v}}_t^*(\mathbf{x}_t)\|^2, \quad (146)$$

where yˆ ∗ t (x) and vˆ ∗ t (x) are defined in [\(18\)](#page-4-7) and [\(119b\)](#page-40-2), respectively.

Lemma D.11. *Suppose Assumptions [3.2,](#page-3-2)* [B2.](#page-3-12) *and* [B3.](#page-3-10) *hold. Let* ˆθ y <sup>t</sup> *be defined in* [\(146\)](#page-49-1)*. Then, for the sequence* {(xt, yt)} T t=1 *generated by Algorithm [2](#page-5-2) guarantees the following bound:*

$$\begin{aligned} & \sum_{t=1}^T \left( \mathbb{E}[\hat{\theta}_{t+1}^{\mathbf{y}}] - \mathbb{E}[\hat{\theta}_t^{\mathbf{y}}] \right) \\ & \leq \left( -\frac{L_{\mu_g}}{2} \sum_{t=1}^T \mathbb{E}[\hat{\theta}_t^{\mathbf{y}}] + \frac{2}{L_{\mu_g}} \sum_{t=1}^T \mathbb{E} [\|e_t^{g_\rho}\|^2] \right) \beta_t + \frac{4L_{\mathbf{y}}^2}{L_{\mu_g}} \sum_{t=1}^T \mathbb{E}\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 \frac{1}{\beta_t} \\ & + \sum_{t=1}^T \left( \frac{24\ell_{g,1}}{L_{\mu_g}\mu_g} (\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2) + \frac{12}{L_{\mu_g}} \sup_{\mathbf{x} \in \mathcal{X}} \|\mathbf{y}_{t-1}^*(\mathbf{x}) - \mathbf{y}_t^*(\mathbf{x})\|^2 \right) \frac{1}{\beta_t} \\ & + \sum_{t=1}^T \left( -\frac{2\beta_t}{\mu_g + \ell_{g,1}} + \beta_t^2 \right) \mathbb{E} [\|\nabla_{\mathbf{y}} g_t, \rho(\mathbf{x}_t, \mathbf{y}_t)\|^2], \end{aligned} \tag{147}$$

*where* Lµ<sup>g</sup> = µgℓg,<sup>1</sup> µg+ℓg,<sup>1</sup> *, and* L<sup>y</sup> = ℓg,<sup>1</sup> µ<sup>g</sup> *is defined as in* [\(42\)](#page-16-2)*.*

*Proof.* From Lemma [B.5,](#page-12-0) we have for any c > 0

$$\begin{aligned} \mathbb{E} [\|\mathbf{y}_{t+1} - \hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_{t+1})\|^2] &= \mathbb{E} [\|\mathbf{y}_{t+1} - \hat{\mathbf{y}}_t^*(\mathbf{x}_t) + \hat{\mathbf{y}}_t^*(\mathbf{x}_t) - \hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_{t+1})\|^2] \\ &\leq (1+c) \mathbb{E} [\|\mathbf{y}_{t+1} - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2] \\ &\quad + \left(1 + \frac{1}{c}\right) \mathbb{E} [\|\hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_{t+1}) - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2]. \end{aligned} \quad (148)$$

From Lemma [D.10,](#page-48-2) we have for any a > 0

$$\begin{aligned} \mathbb{E} \left[ \|\mathbf{y}_{t+1} - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] &\leq (1+a) \left( 1 - 2\beta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \right) \mathbb{E} \left[ \|\mathbf{y}_t - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] \\ &\quad + \left( -(1+a) \left( \frac{2\beta_t}{\mu_g + \ell_{g,1}} - \beta_t^2 \right) \right) \mathbb{E} \left[ \|\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \right] \\ &\quad + \left( 1 + \frac{1}{a} \right) \beta_t^2 \mathbb{E} \left[ \|e_t^{g_\rho}\|^2 \right]. \end{aligned} \quad (149)$$

Substituting [\(149\)](#page-49-2) into [\(148\)](#page-49-3), we get

$$\begin{aligned} & \mathbb{E} \left[ \|\mathbf{y}_{t+1} - \hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_{t+1})\|^2 \right] \\ & \leq (1+c)(1+a) \left( 1 - 2\beta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \right) \mathbb{E} \left[ \|\mathbf{y}_t - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] \\ & + \left( -(1+c)(1+a) \left( \frac{2\beta_t}{\mu_g + \ell_{g,1}} - \beta_t^2 \right) \right) \mathbb{E} \left[ \|\nabla_{\mathbf{y}} g_t, \boldsymbol{\rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \right] \\ & + (1+c)(1 + \frac{1}{a}) \beta_t^2 \mathbb{E} \left[ \|e_t^{g_t}\|^2 \right] + \left( 1 + \frac{1}{c} \right) \mathbb{E} \left[ \|\hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_{t+1}) - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right]. \end{aligned} \quad (150)$$

2756

2759 2760

2764

2766

2769

2774

2776

2779

2789 2790

2794

2796

2799 2800

Choose c = βtLµg /2 1−βtLµg and a = βtLµg 1−2βtLµg . Then, the following equations and inequalities are satisfied.

$$(1+c)(1+a)(1-2\beta_t L_{\mu_g}) = 1 - \frac{\beta_t L_{\mu_g}}{2},$$

$$(1+a)(1-2\beta_t L_{\mu_g}) = 1 - \beta_t L_{\mu_g},$$

$$(1+c)(1-\beta_t L_{\mu_g}) = 1 - \frac{\beta_t L_{\mu_g}}{2}, \quad (151)$$

$$1 + \frac{1}{a} \leq \frac{1}{\beta_t L_{\mu_g}}, \quad 1 + \frac{1}{c} \leq \frac{2}{\beta_t L_{\mu_g}},$$

where Lµ<sup>g</sup> = µgℓg,<sup>1</sup> µg+ℓg,<sup>1</sup> . Based on [\(150\)](#page-49-4) and [\(151\)](#page-50-0), we get

$$\begin{aligned} & \mathbb{E} \left[ \|\mathbf{y}_{t+1} - \hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_{t+1})\|^2 \right] - \mathbb{E} \left[ \|\mathbf{y}_t - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] \\ & \leq -\frac{\beta_t L_{\mu_g}}{2} \mathbb{E} \left[ \|\mathbf{y}_t - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] + \left( -\left( \frac{2\beta_t}{\mu_g + \ell_{g,1}} - \beta_t^2 \right) \right) \mathbb{E} \left[ \|\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \right] \\ & + \frac{2}{\beta_t L_{\mu_g}} \beta_t^2 \mathbb{E} \left[ \|e_t^{g_\rho}\|^2 \right] + \frac{2}{\beta_t L_{\mu_g}} \mathbb{E} \left[ \|\hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_{t+1}) - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right]. \end{aligned} \quad (152)$$

Next, we upper-bound the last term of the above inequality.

$$\begin{aligned} & \mathbb{E} \left[ \|\hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_{t+1}) - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] \\ & \leq 2 \left( \mathbb{E} \left[ \|\hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_{t+1}) - \hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_t)\|^2 \right] + \mathbb{E} \left[ \|\hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_t) - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] \right) \\ & \leq 2 \left( L_y^2 \mathbb{E} \left[ \|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 + \|\hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_t) - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] \right), \end{aligned} \tag{153}$$

where the second inequality is by Lemma [D.2.](#page-40-0)

Moreover, from Lemma [D.7,](#page-44-3) we get

$$\begin{aligned} \mathbb{E} \left[ \|\hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_t) - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] &\leq 3\mathbb{E} \left[ \|\hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_t) - \mathbf{y}_{t+1}^*(\mathbf{x}_t)\|^2 \right] \\ &\quad + 3\mathbb{E} \left[ \|\mathbf{y}_{t+1}^*(\mathbf{x}_t) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 \right] + 3\mathbb{E} \left[ \|\mathbf{y}_t^*(\mathbf{x}_t) - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] \\ &\leq 3\mathbb{E} \left[ \|\mathbf{y}_{t+1}^*(\mathbf{x}_t) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 \right] + \frac{6\ell_{g,1}(\rho_s^2 + \rho_r^2)}{\mu_g}. \end{aligned} \quad (154)$$

Combining [\(153\)](#page-50-1) and [\(154\)](#page-50-2) yields

$$\begin{aligned} & \mathbb{E} \left[ \|\hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_{t+1}) - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \right] \\ & \leq 2 \left( L_{\mathbf{y}}^2 \mathbb{E} \left[ \|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 \right] + 3\mathbb{E} \left[ \|\mathbf{y}_{t+1}^*(\mathbf{x}_t) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 \right] + \frac{6\ell_{g,1}(\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2)}{\mu_g} \right). \end{aligned} \quad (155)$$

Substituting [\(155\)](#page-50-3) into [\(152\)](#page-50-4) and summing over t ∈ [T], give the desired result.

# D.4. Bounds on the Zeroth-Order System Solution

Lemma D.12. *Suppose Assumptions* [B2.](#page-3-12) *and* [B3.](#page-3-10) *hold. Then, for the sequence* {(xt, yt, vt)} T <sup>t</sup>=1 *generated by Algorithm [2,](#page-5-2) we have*

$$\begin{aligned} & \mathbb{E}\|\hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1})\|^2 \\ & \leq (12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{V}}^2})d_2\mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + (12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{V}}^2})d_2\mathbb{E}\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\ & + \frac{9}{2}d_2\ell_{g,1}^2\mathbb{E}\|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + (3\ell_{f,1}^2 + \frac{3\ell_{g,1}^2}{4\rho_{\mathbf{V}}^2})d_2^2\rho_{\mathbf{r}}^2, \end{aligned}$$

*2809*

*2814*

*2816*

*2819*

*2824* where the first inequality follows from Lemma [D.6.](#page-44-2)

*2829*

*2834*

*2836*

*2854*

*2856*

*Proof.* From Lemma [D.6,](#page-44-2) we have

$$\begin{aligned} & \|\hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1})\|^2 \\ & \leq 3d_2 \ell_{f,1}^2 \|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 + \frac{3}{2} \ell_{f,1}^2 d_2^2 \rho_{\mathbf{r}}^2 \\ & \leq 6d_2 \ell_{f,1}^2 \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 6d_2 \ell_{f,1}^2 \|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 + \frac{3}{2} \ell_{f,1}^2 d_2^2 \rho_{\mathbf{r}}^2. \end{aligned} \quad (156)$$

Moreover, from [\(21a\)](#page-4-4), we have

$$\begin{aligned} & \| \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1}) \|^2 \\ &= \frac{1}{4\rho_{\mathbf{z}}^2} \| \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1} + \rho_{\mathbf{v}} \mathbf{v}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t; \bar{\mathcal{B}}_{t+1}) \|^2 \\ &\leq \frac{3}{4\rho_{\mathbf{z}}^2} d_2 \ell_{g,1}^2 \| (\mathbf{x}_{t+1}, \mathbf{y}_{t+1} + \rho_{\mathbf{v}} \mathbf{v}_{t+1}) - (\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t) \|^2 + \frac{3}{8\rho_{\mathbf{z}}^2} \ell_{g,1}^2 d_2^2 \rho_{\mathbf{r}}^2 \\ &\leq \frac{9}{4\rho_{\mathbf{z}}^2} d_2 \ell_{g,1}^2 \| \mathbf{x}_{t+1} - \mathbf{x}_t \|^2 + \frac{9}{4\rho_{\mathbf{z}}^2} d_2 \ell_{g,1}^2 \| \mathbf{y}_{t+1} - \mathbf{y}_t \|^2 \\ &+ \frac{9}{4} d_2 \ell_{g,1}^2 \| \mathbf{v}_{t+1} - \mathbf{v}_t \|^2 + \frac{3}{8\rho_{\mathbf{z}}^2} \ell_{g,1}^2 d_2^2 \rho_{\mathbf{r}}^2, \end{aligned} \tag{157}$$

From ∥a + b∥ <sup>2</sup> ≤ 2 ∥a∥ <sup>2</sup> + ∥b∥ 2 , we get

$$\begin{aligned} & \|\hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1})\|^2 \\ & \leq 2\|\hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1})\|^2 \\ & + 2\|\hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1})\|^2 \\ & \leq (12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2})d_2\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + (12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2})d_2\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\ & + \frac{9}{2}d_2\ell_{g,1}^2\|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + (3\ell_{f,1}^2 + \frac{3\ell_{g,1}^2}{4\rho_{\mathbf{v}}^2})d_2^2\rho_{\mathbf{r}}^2, \end{aligned}$$

where the second inequality follows from [\(156\)](#page-51-1) and [\(157\)](#page-51-2).

Lemma D.13. *Suppose Assumptions* [B2.](#page-3-12)*,* [B3.](#page-3-10)*,* [D1.](#page-4-12)*, and* [D3.](#page-4-14) *hold. Consider the sequence* {(xt, yt, vt)} T <sup>t</sup>=1 *generated by Algorithm [2,](#page-5-2) and define*

$$e_{t+1}^M := \nabla_{\mathbf{y}} f_{t+1, \boldsymbol{\rho}}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) - \hat{\mathbf{d}}_{t+1}^{\mathbf{y}}, \quad \text{where} \quad (158)$$

$$\begin{aligned}\tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1}) &= \frac{1}{2\rho_{\mathbf{v}}} (\nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1} + \rho_{\mathbf{v}} \mathbf{v}_{t+1}) \\ &\quad - \nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1} - \rho_{\mathbf{v}} \mathbf{v}_{t+1})).\end{aligned}\tag{159}$$

$$\begin{aligned}
& 2889 \quad \mathbb{E} \|\nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{d}}_{t+1}^{\mathbf{y}}\|^2 \\
& 2890 \quad = \mathbb{E} \|\nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{d}}_t^{\mathbf{y}} - (\hat{\mathbf{d}}_{t+1}^{\mathbf{y}} - \hat{\mathbf{d}}_t^{\mathbf{y}})\|^2 \\
& 2891 \quad = \mathbb{E} \|\nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{d}}_t^{\mathbf{y}} + \lambda_{t+1} \hat{\mathbf{d}}_t^{\mathbf{y}} \\
& 2892 \quad = \mathbb{E} \|\nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{d}}_t^{\mathbf{y}} + \lambda_{t+1} \hat{\mathbf{d}}_t^{\mathbf{y}} \\
& 2893 \quad - \lambda_{t+1} \left( \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) \right) \\
& 2894 \quad - \lambda_{t+1} \left( \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) \right) \\
& 2895 \quad - (1 - \lambda_{t+1}) \left( \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) \right) \\
& 2896 \quad - \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1}) \Big) \|^2 \\
& 2897 \quad = \mathbb{E} \|(1 - \lambda_{t+1})(\nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t) - \hat{\mathbf{d}}_t^{\mathbf{y}}) \\
& 2898 \quad + \lambda_{t+1}(\nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1})) \\
& 2899 \quad + (1 - \lambda_{t+1}) \left( \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t) \right) \\
& 2900 \quad + \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) - \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) \\
& 2901 \quad - \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) + \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1}) \Big) \|^2 \\
& 2902 \quad + (1 - \lambda_{t+1}) \left( \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t) \right) \\
& 2903 \quad + \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) - \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) \\
& 2904 \quad - \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) + \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{$$

2907 2908 Since

$$\begin{aligned} & 2909 \\ & 2910 \quad \mathbb{E} \left[ \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) \right] = \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}), \\ & 2911 \quad \mathbb{E} \left[ \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1}) \right] \\ & 2912 \\ & 2913 \quad = \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_t) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t), \\ & 2914 \end{aligned}$$

*Then, we have*

$$\begin{aligned} \mathbb{E}\|e_{t+1}^M\|^2 &\leq (1 - \lambda_{t+1})^2 \mathbb{E}\|e_t^M\|^2 + 36\mathbb{E}\|\nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ &\quad + \left( 18d_2^2 \ell_{f,1}^2 + 6(3\ell_{f,1}^2 + \frac{3\ell_{g,1}^2}{4\rho_{\mathbf{v}}^2})d_2^2 \right) \rho_{\mathbf{r}}^2 + 18d_2^2 \ell_{g,1}^2 \frac{\rho_{\mathbf{r}}^2}{\rho_{\mathbf{v}}^2} \\ &\quad + \frac{18}{\rho_{\mathbf{v}}^2} \mathbb{E}\|\nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \\ &\quad + \frac{18}{\rho_{\mathbf{v}}^2} \mathbb{E}\|\nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \\ &\quad + 6(12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2})d_2 \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 6(12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2})d_2 \mathbb{E}\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\ &\quad + 27d_2 \ell_{g,1}^2 \mathbb{E}\|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + 3(\frac{\hat{\sigma}_{gy}^2}{b\rho_{\mathbf{v}}^2} + \frac{\hat{\sigma}_{fy}^2}{b})\lambda_{t+1}^2. \end{aligned} \tag{160}$$

*Proof.* According to the definition of dˆ<sup>v</sup> t in Algorithm [2,](#page-5-2) we have

$$\begin{aligned} \hat{\mathbf{a}}_{t+1}^{\mathbf{y}} - \hat{\mathbf{a}}_t^{\mathbf{y}} &= -\lambda_{t+1}\hat{\mathbf{a}}_t^{\mathbf{y}} + \lambda_{t+1}(\hat{\nabla}_{\mathbf{y}}f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1})) \\ &\quad + (1 - \lambda_{t+1}) \left( \hat{\nabla}_{\mathbf{y}}f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) \right. \\ &\quad \left. - \hat{\nabla}_{\mathbf{y}}f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1}) \right). \end{aligned}$$

Then we have

$$\begin{aligned}
& 2916 \quad \mathbb{E} \|\nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{d}}_{t+1}^{\mathbf{y}}\|^2 \\
& 2917 \quad = (1 - \lambda_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t) - \hat{\mathbf{d}}_t^{\mathbf{y}}\|^2 \\
& 2918 \quad + \|\lambda_{t+1}(\nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1})) \\
& 2920 \quad + (1 - \lambda_{t+1}) \left( \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t) \right) \\
& 2921 \quad + (1 - \lambda_{t+1}) \left( \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t) \right) \\
& 2922 \quad + \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) - \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) \\
& 2923 \quad + \nabla_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) + \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1}) \Big) \|^2 \\
& 2924 \quad - \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) + \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1}) \Big) \|^2 \\
& 2925 \quad \leq (1 - \lambda_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t) - \hat{\mathbf{d}}_t^{\mathbf{y}}\|^2 \\
& 2926 \quad + 3(1 - \lambda_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t) \\
& 2927 \quad + \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) - \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) \\
& 2928 \quad + 3(1 - \lambda_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t) \\
& 2929 \quad + \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) - \nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) \\
& 2930 \quad - \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) + \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1})\|^2 \\
& 2931 \quad + \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}($$

2947 Then, from <sup>E</sup>∥a − <sup>E</sup>[a]∥ <sup>2</sup> = <sup>E</sup>∥a∥ <sup>2</sup> − ∥E[a]∥ 2 and Assumption [4.1,](#page-4-6) we have

2948 2949

2954

2956

then, we have

where the second inequality holds by Cauchy-Schwarz inequality.

Note that, for the last term on the right-hand side of [\(161\)](#page-53-0), from [\(21a\)](#page-4-4) and [\(159\)](#page-51-3), we have

$$\begin{aligned} & \| \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) \|^2 \\ & \leq 2 \left\| \frac{1}{2\rho_{\mathbf{v}}} (\nabla_{\mathbf{y}} g_{t+1,\rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1} + \rho_{\mathbf{v}} \mathbf{v}_{t+1}) - \hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1} + \rho_{\mathbf{v}} \mathbf{v}_{t+1}; \bar{\mathcal{B}}_{t+1})) \right\|^2 \\ & + 2 \left\| \frac{1}{2\rho_{\mathbf{v}}} (\hat{\nabla}_{\mathbf{y}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1} - \rho_{\mathbf{v}} \mathbf{v}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \nabla_{\mathbf{y}} g_{t+1,\rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1} - \rho_{\mathbf{v}} \mathbf{v}_{t+1})) \right\|^2 \\ & \leq \frac{\hat{\sigma}_{g_{\mathbf{y}}}^2}{\bar{b}\rho_{\mathbf{v}}^2}, \end{aligned}$$

where the last inequality follows from Assumption [4.1.](#page-4-6)

$$\begin{aligned} & \mathbb{E} \|\nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{a}}_{t+1}^{\mathbf{y}}\|^2 \\ & \leq (1 - \lambda_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t) - \hat{\mathbf{a}}_t^{\mathbf{y}}\|^2 \\ & + 6(1 - \lambda_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) - \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t)\|^2 \\ & + 6(1 - \lambda_{t+1})^2 \mathbb{E} \|\hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) \\ & - \hat{\nabla}_{\mathbf{y}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1})\|^2 + 3\lambda_{t+1}^2 \left( \frac{\hat{\sigma}_{g_y}^2}{\bar{b}\rho_{\mathbf{y}}^2} + \frac{\hat{\sigma}_{f_y}^2}{b} \right). \end{aligned}$$

2976

2979

2981

2984

2986 2987

2989 2990

$$\begin{aligned} & \langle \nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) \rangle^2 \\ & \leq 3 \|\nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \\ & + 3 \|\nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \\ & + 3 \|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \\ & \leq 3 \|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 + \frac{3 \rho_{\mathbf{r}}^2 d_2^2 \ell_{g, 1}^2}{2}. \end{aligned}$$

3014

3016

3018 3019

Then, from Young's inequality and Lemma [D.12,](#page-50-5) we obtain

$$\begin{aligned} & \mathbb{E}\|\nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{d}}_{t+1}^{\mathbf{y}}\|^2 \\ & \leq (1 - \lambda_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t) - \hat{\mathbf{d}}_t^{\mathbf{y}}\|^2 \\ & + 12(1 - \lambda_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{z}_t) - \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{z}_t)\|^2 \\ & + 12(1 - \lambda_{t+1})^2 \mathbb{E}\|\tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t)\|^2 \\ & + 6(12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2})d_2\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 6(12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2})d_2\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\ & + 27d_2\ell_{g,1}^2\|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + 6(3\ell_{f,1}^2 + \frac{3\ell_{g,1}^2}{4\rho_{\mathbf{v}}^2})d_2^2\rho_{\mathbf{r}}^2 + 3\lambda_{t+1}^2(\frac{\hat{\sigma}_{g_{\mathbf{y}}}^2}{b\rho_{\mathbf{v}}^2}) + \frac{\hat{\sigma}_{f_{\mathbf{y}}}^2}{b}, \end{aligned} \tag{162}$$

For the third term on the right-hand side of [\(162\)](#page-54-0), based on [\(159\)](#page-51-3), we have

$$\begin{aligned} & \|\tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_t, \mathbf{y}_t) - \tilde{\nabla}_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & \leq \frac{1}{2\rho_{\mathbf{V}}^2} \|\nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{V}} \mathbf{V}_t) - \nabla_{\mathbf{y}} g_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{V}} \mathbf{V}_t)\|^2 \end{aligned} \quad (163a)$$

$$+ \frac{1}{2\rho_{\text{V}}^2} \|\nabla_{\mathbf{y}} g_t, \rho(\mathbf{x}_t, \mathbf{y}_t - \rho_{\text{V}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\text{V}} \mathbf{v}_t)\|^2. \quad (163b)$$

For [\(163a\)](#page-54-1), we get

where the last inequality follows from Eq. [\(131\)](#page-43-0).

Similary, for [\(163b\)](#page-54-2), we have

$$\begin{aligned} & \|\nabla_{\mathbf{y}} g_t, \rho(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t+1, \rho}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \\ & \leq 3 \|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 + \frac{3\rho_{\mathbf{r}}^2 d_2^2 \ell_{g_1}^2}{2}. \end{aligned}$$

Substituting the above inequalities in [\(163\)](#page-54-3), we have

$$\begin{aligned} \|\tilde{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{x}_t, \mathbf{y}_t) - \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 &\leq \frac{3}{2\rho_{\mathbf{V}}^2} \|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{V}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{V}} \mathbf{v}_t)\|^2 \\ &+ \frac{3}{2\rho_{\mathbf{V}}^2} \|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{V}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{V}} \mathbf{v}_t)\|^2 + \frac{3\rho_{\mathbf{r}}^2 d_2^2 \ell_{g,1}^2}{2\rho_{\mathbf{V}}^2}. \end{aligned} \quad (164)$$

For the second term on the right-hand side of [\(162\)](#page-54-0), we have

$$\begin{aligned} & \|\nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & \leq 3 \|\nabla_{\mathbf{y}} f_{t+1,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & + 3 \|\nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & + 3 \|\nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & \leq 3 \|\nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_t, \mathbf{y}_t)\|^2 + \frac{3\rho_{\mathbf{r}}^2 d_{\mathbf{r}}^2 \ell_{f,1}^2}{2}, \end{aligned} \tag{165}$$

$$\begin{aligned} 3026 & \quad \mathbb{E} \|\nabla_{\mathbf{y}} f_{t+1, \rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{y}}^7 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{d}}_{t+1}^{\mathbf{y}}\|^2 \\ 3027 & \leq (1 - \lambda_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{y}} f_{t, \rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t) - \hat{\mathbf{d}}_{\mathbf{y}}^{\mathbf{y}}\|^2 \\ 3028 & + 36 \|\nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_t, \mathbf{y}_t)\|^2 + 18 \rho_{\mathbf{r}}^2 d_{\mathbf{r}}^2 \ell_{f,1}^2 \\ 3030 & + \frac{18}{\rho_{\mathbf{v}}^2} \|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \\ 3032 & + \frac{18}{\rho_{\mathbf{v}}^2} \|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 + \frac{18 \rho_{\mathbf{r}}^2 d_{\mathbf{r}}^2 \ell_{g,1}^2}{\rho_{\mathbf{v}}^2} \\ 3033 & + \frac{18}{\rho_{\mathbf{v}}^2} \|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 + \frac{18 \rho_{\mathbf{r}}^2 d_{\mathbf{r}}^2 \ell_{g,1}^2}{\rho_{\mathbf{v}}^2} \\ 3034 & + \frac{18}{\rho_{\mathbf{v}}^2} \|\nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 + \frac{18 \rho_{\mathbf{r}}^2 d_{\mathbf{r}}^2 \ell_{g,1}^2}{\rho_{\mathbf{v}}^2} \\ 3035 & + 6(12 \ell_{f,1}^2 + \frac{9 \ell_{g,1}^2}{2 \rho_{\mathbf{v}}^2}) d_2 \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 6(12 \ell_{f,1}^2 + \frac{9 \ell_{g,1}^2}{2 \rho_{\mathbf{v}}^2}) d_2 \|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\ 3036 & + 27 d_2 \ell_{g,1}^2 \|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + 6(3 \ell_{f,1}^2 + \frac{3 \ell_{g,1}^2}{4 \rho_{\mathbf{v}}^2}) d_2^2 \rho_{\mathbf{r}}^2 + 3 \lambda_{t+1}^2 (\frac{\hat{\sigma}_{g_{\mathbf{y}}}^2}{b \rho_{\mathbf{v}}^2} + \frac{\hat{\sigma}_{f_{\mathbf{y}}}^2}{b}). \\ 3038 & + 27 d_2 \ell_{g,1}^2 \|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + 6(3 \ell_{f,1}^2 + \frac{3 \ell_{g,1}^2}{4 \rho_{\mathbf{v}}^2}) d_2^2 \rho_{\mathbf{r}}^2 + 3 \lambda_{t+1}^2 (\frac{\hat{\sigma}_{g_{\mathbf{y}}}^2}{b \rho_{\mathbf{v}}^2} + \frac{\hat{\sigma}_{f_{\mathbf{y}}}^2}{b}). \\ 3040 & + 27 d_2 \ell_{g,1}^2 \|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + 6(3 \ell_{f,1}^2 + \frac{3 \ell_{g,1}^2}{4 \rho_{\mathbf{v}}^2}) d_2^2 \rho_{\mathbf{r}}^2 + 3 \lambda_{t+1}^2 (\frac{\hat{\sigma}_{g_{\mathbf{y}}}^2}{b \rho_{\mathbf{v}}^2} + \frac{\hat{\sigma}_{f_{\mathbf{y}}}^2}{b}). \end{aligned}$$

3043 3044 Lemma D.14. *Suppose Assumptions* [B3.](#page-3-10) *and* [B4.](#page-3-13) *hold. Let*

3054

3056

3057      (a)  
 3058  
 3059      
$$\mathbb{E} \left[ \left\| e_t^H \right\|^2 \right] \leq \ell_{g,2}^2 \rho_{\mathbf{v}}^2 p^4.$$
      (167a)  
 3060

3061 3062

$$\mathbb{E} \left[ \|e_t^J\|^2 \right] \leq \ell_{g,2} \rho_v^2 p^4. \quad (167b)$$

3065

3066 3067 *Proof.* For part (a): From Lemma [D.1,](#page-40-8) We have

$$\begin{aligned}
3068 \quad & \mathbb{E} \left[ \|e_t^H\| \right] = \mathbb{E} \left[ \left\| \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) \mathbf{v}_t \right\| \right] \\
3069 & \leq \frac{1}{2\rho_{\mathbf{v}}} \mathbb{E} \left[ \|\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) \rho_{\mathbf{v}} \mathbf{v}_t\| \right] \\
3070 & + \frac{1}{2\rho_{\mathbf{v}}} \mathbb{E} \left[ \|\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) \rho_{\mathbf{v}} \mathbf{v}_t\| \right] \\
3073 & \leq \ell_{g,2} \rho_{\mathbf{v}} \mathbb{E} \left[ \|\mathbf{v}_t\|^2 \right] \\
3074 & \leq \ell_{g,2} \rho_{\mathbf{v}} \mathbb{E} \left[ \|\mathbf{v}_t\|^2 \right] \\
3075 & \leq \ell_{g,2} \rho_{\mathbf{v}} \mathbb{E} \left[ \|\mathbf{v}_t\|^2 \right] \\
3076 & \leq \ell_{g,2} \rho_{\mathbf{v}} p^2, \tag{168}
\end{aligned}$$

From [\(164\)](#page-54-4), [\(165\)](#page-54-5) and [\(162\)](#page-54-0), we get

$$e_t^H := \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) \mathbf{v}_t, \quad (16\text{aa})$$

$$e_t^J := \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) \mathbf{v}_t, \quad (16\text{b})$$

*where*

$$\begin{aligned}\nabla_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) &= \frac{1}{2\rho_{\mathbf{v}}}(\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}}\mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}}\mathbf{v}_t)), \\ \nabla_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) &= \frac{1}{2\rho_{\mathbf{v}}}(\nabla_{\mathbf{x}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}}\mathbf{v}_t) - \nabla_{\mathbf{x}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}}\mathbf{v}_t)).\end{aligned}$$

*Then, for* (xt, yt, vt) *presented to Algorithm [2,](#page-5-2) we have*

*(b)*

3086 3087

3089 3090 3091

3099 3100

3104

3106

3111 *Proof.* For part (a): Let

3114

3116

3118 3119

3124

3126

3129

For part (b): From Lemma [D.1,](#page-40-8) We have

$$\begin{aligned}\mathbb{E} [\|e_t^J\|] &= \mathbb{E} \left[ \left\| \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) \mathbf{v}_t \right\| \right] \\ &\leq \frac{1}{2\rho_{\mathbf{v}}} \mathbb{E} \left[ \|\nabla_{\mathbf{x}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{x}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) \rho_{\mathbf{v}} \mathbf{v}_t\| \right] \\ &\quad + \frac{1}{2\rho_{\mathbf{v}}} \mathbb{E} \left[ \|\nabla_{\mathbf{x}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{x}\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) \rho_{\mathbf{v}} \mathbf{v}_t\| \right] \\ &\leq \ell_{g,2} \rho_{\mathbf{v}} \mathbb{E} \left[ \|\mathbf{v}_t\|^2 \right] \\ &\leq \ell_{g,2} \rho_{\mathbf{v}} p^2,\end{aligned}\tag{169}$$

where the last inequality follows from [\(8\)](#page-3-8).

Lemma D.15. *Suppose Assumptions* [B2.](#page-3-12)*,* [B3.](#page-3-10) *and* [B4.](#page-3-13) *hold. Then, for directions* dˆ<sup>v</sup> <sup>t</sup> *and* <sup>d</sup>ˆ<sup>x</sup> <sup>t</sup> *presented to Algorithm [2,](#page-5-2) and*

*(a)* d v t,<sup>ρ</sup> *defined in* [\(120b\)](#page-40-9)*, we have*

$$\mathbb{E} \left[ \left\| \hat{\mathbf{d}}_t^\mathbf{v} - \mathbf{d}_{t,\rho}^\mathbf{v} \right\|^2 \right] \leq 2\mathbb{E} \left[ \left\| e_t^M \right\|^2 \right] + 2\ell_{g,2}^2 \rho_\mathbf{v}^2 p^4 := B_t, \quad (170a)$$

*where* e<sup>M</sup> t := dˆ<sup>v</sup> <sup>t</sup> − ∇yft,ρ(xt, <sup>y</sup>t) − ∇˜ <sup>2</sup> y g<sup>t</sup> (xt, yt)*, with* ∇˜ <sup>2</sup> y gt(xt, yt) *is defined in* [\(171\)](#page-56-1)*.*

*(b)* d x t,<sup>ρ</sup> *defined in* [\(120c\)](#page-40-10)*, we have*

$$\mathbb{E} \left[ \left\| \hat{\mathbf{d}}_t^x - \mathbf{d}_{t,\rho}^x \right\|^2 \right] \leq 2\mathbb{E} \left[ \|e_t^L\|^2 \right] + 2\ell_{g,2}^2 \rho_{\mathbf{v}}^2 p^4, \quad (170b)$$

*where* e L t := dˆ<sup>x</sup> <sup>t</sup> − ∇xft,ρ(xt, <sup>y</sup>t) − ∇˜ <sup>2</sup> xyg<sup>t</sup> (xt, <sup>y</sup>t) *with* ∇˜ <sup>2</sup> xyg<sup>t</sup> (xt, yt) *is defined in* [\(176\)](#page-57-0)*.*

$$\tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) = \frac{1}{2\rho_{\mathbf{v}}} (\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)). \quad (171)$$

According to the definition of d v t,ρ in [\(120b\)](#page-40-9), we have

$$\begin{aligned}\mathbb{E} \left[ \left\| \hat{\mathbf{d}}_t^{\mathbf{v}} - \mathbf{d}_{t,\rho}^{\mathbf{v}} \right\|^2 \right] &= \mathbb{E} \left[ \left\| \hat{\mathbf{d}}_t^{\mathbf{v}} - \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) \mathbf{v} \right\|^2 \right] \\ &\leq 2\mathbb{E} \left[ \left\| \hat{\mathbf{d}}_t^{\mathbf{v}} - \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) \right\|^2 \right]\end{aligned} \quad (172a)$$

$$+ 2\mathbb{E} \left[ \left\| \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) \mathbf{v} \right\|^2 \right]. \quad (172b)$$

Next, we separately bound [\(172a\)](#page-56-2)–[\(172b\)](#page-56-3) on the RHS of the above inequality.

Bounding [\(172a\)](#page-56-2) . We have

$$2\mathbb{E} \left[ \left\| \hat{\mathbf{d}}_t^{\mathbf{v}} - \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \tilde{\nabla}_2^2 g_t(\mathbf{x}_t, \mathbf{y}_t) \right\|^2 \right] := 2\mathbb{E} \left[ \left\| e_t^M \right\|^2 \right]. \quad (173)$$

Bounding [\(172b\)](#page-56-3) . From Lemmas [D.1](#page-40-8) and [D.14,](#page-55-0) we have

$$(172b) = \mathbb{E} \left[ \|e_t^H\|^2 \right] \leq 3\ell_{g,2}^2 \rho_{\sqrt{v}}^2 p^4. \quad (174)$$

3154

3156

3159 3160 3161

3164 3165 3166

3169

3174

3176

3179

Combining [\(173\)](#page-56-4)-[\(174\)](#page-56-5) yields

$$\mathbb{E} \left[ \left\| \hat{\mathbf{d}}_t^Y - \mathbf{d}_{t,\rho}^Y \right\|^2 \right] \leq 2\mathbb{E} \left[ \left\| e_t^M \right\|^2 \right] + 2\ell_{g,2}^2 \rho_{\mathbf{v}}^2 p^4. \quad (175)$$

For part (b): Let

$$\tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) = \frac{1}{2\rho_{\mathbf{v}}} (\nabla_{\mathbf{x}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{x}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)). \quad (176)$$

According to the definition of d x t,ρ in [\(120c\)](#page-40-10), we have

$$\begin{aligned}\mathbb{E} \left[ \left\| \hat{\mathbf{d}}_t^x - \mathbf{d}_{t,\rho}^x \right\|^2 \right] &= \mathbb{E} \left[ \left\| \hat{\mathbf{d}}_t^x - \nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) \mathbf{v} \right\|^2 \right] \\ &\leq 2\mathbb{E} \left[ \left\| \hat{\mathbf{d}}_t^x - \nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) \right\|^2 \right]\end{aligned}\tag{177a}$$

$$+ 2\mathbb{E} \left[ \left\| \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) \mathbf{v}_t \right\|^2 \right]. \quad (177b)$$

Next, we separately bound [\(177a\)](#page-57-1)–[\(177b\)](#page-57-2) on the RHS of the above inequality.

Bounding [\(177a\)](#page-57-1) . We have

$$2\mathbb{E} \left[ \left\| \hat{\mathbf{d}}_t^x - \nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) \right\|^2 \right] := 2\mathbb{E} \left[ \left\| e_t^L \right\|^2 \right]. \quad (178)$$

Bounding [\(177b\)](#page-57-2) . From Lemmas [D.1](#page-40-8) and [D.14,](#page-55-0) we have

$$(177b) = \mathbb{E} \left[ \|e_t^J\|^2 \right] \leq 2\ell_{g,2}^2 \rho_{\mathbf{v}}^2 p^4. \quad (179)$$

Combining [\(178\)](#page-57-3)–[\(179\)](#page-57-4) yields

$$\mathbb{E} \left[ \left\| \hat{\mathbf{d}}_t^x - \mathbf{d}_{t,\rho}^x \right\|^2 \right] \leq 2\mathbb{E} \left[ \|e_t^L\|^2 \right] + 2\ell_{g,2}^2 \rho_{\mathbf{v}}^2 p^4.$$

Lemma D.16. *Suppose Assumptions [3.2,](#page-3-2)* [B1.](#page-3-18) *and* [B3.](#page-3-10) *hold. Set the step size* δ<sup>t</sup> *and the parameter* p *in* [\(8\)](#page-3-8)*, as*

$$\delta_t \leq \left( 2 + \frac{1}{\ell_{g,1}^2} \right) \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}}, \quad \forall t \in [T], \quad \text{and} \quad p = \frac{\ell_{f,0}}{\mu_g}. \quad (180)$$

*Then, for the sequence* {(xt, yt, vt)} T <sup>t</sup>=1 *generated by Algorithm [2,](#page-5-2) we have*

$$\mathbb{E} \left[ \|\mathbf{v}_{t+1} - \hat{\mathbf{v}}_t^*(\mathbf{x}_t)\|^2 \right] \leq (1+\hat{\alpha}) \left( 1 - \delta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \right) \mathbb{E}[\hat{\theta}_t^\mathbf{v}] + \left( 1 + \frac{1}{\hat{\alpha}} \right) \delta_t^2 B_t,$$

3194

3196

3199 3200

3204

3206

3209

3214

3216

3219

3224

3226

3229

3234

3236 Bounding K<sup>t</sup> . Let

3239

3240 3241 From Lemma [D.15,](#page-56-0) we have

3242 3243 3244 *Proof.* By setting the radius p := ℓf,<sup>0</sup> µ<sup>g</sup> in [\(8\)](#page-3-8), we have

$$\begin{aligned}\mathbb{E} \left[ \|\mathbf{v}_{t+1} - \hat{\mathbf{v}}_t^*(\mathbf{x}_t)\|^2 \right] &= \mathbb{E} \left[ \left\| \Pi_{\mathcal{Z}_p} \left[ \mathbf{v}_t - \delta_t \hat{\mathbf{d}}_t^\mathbf{y} \right] - \Pi_{\mathcal{Z}_p} \left[ \hat{\mathbf{v}}_t^*(\mathbf{x}_t) \right] \right\|^2 \right] \\ &\leq \mathbb{E} \left[ \|\mathbf{v}_t - \delta_t \hat{\mathbf{d}}_t^\mathbf{y} - \hat{\mathbf{v}}_t^*(\mathbf{x}_t)\|^2 \right] \\ &\leq (1 + \hat{\alpha}) \underbrace{\mathbb{E} \left[ \|\mathbf{v}_t - \delta_t \mathbf{d}_{t,\rho}^\mathbf{y}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t) - \hat{\mathbf{v}}_t^*(\mathbf{x}_t)\|^2 \right]}_{I_t} \\ &\quad + \left( 1 + \frac{1}{\hat{\alpha}} \right) \delta_t^2 \underbrace{\mathbb{E} \left[ \|\hat{\mathbf{d}}_t^\mathbf{y} - \mathbf{d}_{t,\rho}^\mathbf{y}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2 \right]}_{K_t},\end{aligned}\tag{181}$$

where d v t,ρ (xt, yt, vt) is defined in [\(120b\)](#page-40-9); the first inequality follows from non-expansiveness property of a projection operator.

We next bound the It, and K<sup>t</sup> terms in [\(181\)](#page-58-0), respectively.

Bounding I<sup>t</sup> . We have

$$\begin{aligned} I_t &= \mathbb{E} \left[ \|\mathbf{v}_t - \hat{\mathbf{v}}_t^*(\mathbf{x}_t)\|^2 \right] - 2\delta_t \mathbb{E} \left[ \langle \mathbf{d}_{t,\rho}^{\mathbf{v}}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t), \mathbf{v}_t - \hat{\mathbf{v}}_t^*(\mathbf{x}_t) \rangle \right] + \delta_t^2 \mathbb{E} \left[ \|\mathbf{d}_{t,\rho}^{\mathbf{v}}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2 \right] \\ &\leq \left( 1 - 2\delta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \right) \mathbb{E} \left[ \|\mathbf{v}_t - \hat{\mathbf{v}}_t^*(\mathbf{x}_t)\|^2 \right] \\ &\quad - \left( 2\delta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} - \delta_t^2 \right) \mathbb{E} \left[ \|\mathbf{d}_{t,\rho}^{\mathbf{v}}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2 \right], \end{aligned}$$

where the inequality holds since d v t,ρ in [\(120\)](#page-40-1) is the gradient of the strongly convex quadratic program <sup>1</sup> 2 v <sup>⊤</sup>∇<sup>2</sup> y gt,<sup>ρ</sup> (x, y) v+ v <sup>⊤</sup>∇yft,ρ(x, y).

Thus, we have

$$\begin{aligned} & \mathbb{E} \left[ \langle \mathbf{d}_{t,\boldsymbol{\rho}}^{\mathbf{y}}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t), \mathbf{v}_t - \hat{\mathbf{v}}_t^*(\mathbf{x}_t) \rangle \right] \\ & \geq \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \mathbb{E} \left[ \|\mathbf{v}_t - \hat{\mathbf{v}}_t^*(\mathbf{x}_t)\|^2 \right] + \frac{1}{\mu_g + \ell_{g,1}} \mathbb{E} \left[ \|\mathbf{d}_{t,\boldsymbol{\rho}}^{\mathbf{y}}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2 \right]. \end{aligned}$$

Since δ<sup>t</sup> ≤ 2 + <sup>1</sup> ℓ g,1 µgℓg,<sup>1</sup> µg+ℓg,<sup>1</sup> , then we have

$$\begin{aligned} I_t &\leq \left( 1 - 2\delta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \right) \mathbb{E} [\|\mathbf{v}_t - \hat{\mathbf{v}}_t^*(\mathbf{x}_t)\|^2] + \frac{1}{\ell_{g,1}^2} \left( \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \delta_t \right) \mathbb{E} [\|\mathbf{d}_{\mathbf{v}, \boldsymbol{\rho}}^{\mathbf{v}}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2] \\ &\leq \left( 1 - \delta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \right) \mathbb{E} [\|\mathbf{v}_t - \hat{\mathbf{v}}_t^*(\mathbf{x}_t)\|^2], \end{aligned} \quad (182)$$

where the second inequality holds since from [\(119b\)](#page-40-2), we have

$$\begin{aligned}\mathbb{E} \left[ \|\mathbf{d}_{t,\rho}^{\mathbf{y}}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2 \right] &= \mathbb{E} \left[ \|\nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}, \mathbf{y}) + \nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y}) \mathbf{v}\|^2 \right] \\ &= \mathbb{E} \left[ \|\nabla_{\mathbf{y}}^2 g_{t,\rho}(\mathbf{x}, \mathbf{y}) (\mathbf{v} - \hat{\mathbf{v}}_t^*(\mathbf{x}_t))\|^2 \right] \\ &\leq \ell_{g,1}^2 \mathbb{E} \left[ \|\mathbf{v}_t - \hat{\mathbf{v}}_t^*(\mathbf{x}_t)\|^2 \right].\end{aligned}$$

$$\tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) = \frac{1}{2\rho_{\mathbf{v}}} (\nabla_{\mathbf{y}} g_t, \rho(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_t, \rho(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)).$$

$$K_t = \mathbb{E} \left[ \|\hat{\mathbf{d}}_t^\mathbf{y} - \mathbf{d}_{t,\rho}^\mathbf{y}(\mathbf{x}_t, \mathbf{y}_t, \mathbf{v}_t)\|^2 \right] \leq B_t. \quad (183)$$

3254

3256

3258 3259 3260

3264

3266

3269

3274

3276

3279

3289 3290

3294

3296

Putting [\(182\)](#page-58-1), and [\(183\)](#page-58-2) together with Eq. [\(181\)](#page-58-0) yields the desired result.

$$\mathbb{E} \left[ \|\mathbf{v}_{t+1} - \hat{\mathbf{v}}_t^*(\mathbf{x}_t)\|^2 \right] \leq (1+\hat{\alpha}) \left( 1 - \delta \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \right) \mathbb{E} \left[ \|\mathbf{v}_t - \hat{\mathbf{v}}_t^*(\mathbf{x}_t)\|^2 \right] + \left( 1 + \frac{1}{\hat{\alpha}} \right) \delta_t^2 B_t.$$

Lemma D.17. *Suppose Assumptions [3.2](#page-3-2) and [3.3](#page-3-3) hold. Let* ˆθ v <sup>t</sup> *be defined in* [\(146\)](#page-49-1)*. Set the parameter* p *in* [\(8\)](#page-3-8) *as* p = ℓf,<sup>0</sup> µ<sup>g</sup> *Then, for any positive choice of step sizes satisfying*

*.*

$$\delta_t \leq \left( 2 + \frac{1}{\ell_{g,1}^2} \right) \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}},$$

*the sequence* {(xt, yt, vt)} T <sup>t</sup>=1 *generated by Algorithm [2](#page-5-2) guarantees the following bound:*

$$\begin{aligned} \sum_{t=1}^T \left( \mathbb{E}[\hat{\theta}_{t+1}^{\mathbf{y}}] - \mathbb{E}[\hat{\theta}_t^{\mathbf{y}}] \right) &\leq \sum_{t=1}^T \left( -\frac{L_{\mu_g}}{4} \mathbb{E}[\hat{\theta}_t^{\mathbf{y}}] + \frac{4}{L_{\mu_g}} B_t \right) \delta_t \\ &\quad + \frac{16\nu^2}{L_{\mu_g}\mu_g^2} (2L_{\mathbf{y}}^2 + 1) \sum_{t=1}^T \mathbb{E} \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 \frac{1}{\delta_t} \\ &\quad + \sum_{t=1}^T \left( \frac{96\ell_{g,1}\nu^2}{L_{\mu_g}\mu_g^3} (\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2) + \frac{48\nu^2}{L_{\mu_g}\mu_g^2} \sup_{\mathbf{x} \in \mathcal{X}} \|\mathbf{y}_{t-1}^*(\mathbf{x}) - \mathbf{y}_t^*(\mathbf{x})\|^2 \right) \frac{1}{\delta_t}, \end{aligned} \quad (184)$$

*where* B<sup>t</sup> *and* ν *are defined in Lemmas [D.15](#page-56-0) and [C.7,](#page-23-5) respectively.*

*Proof.* From Lemma [B.5,](#page-12-0) we have, for any c >´ 0

$$\begin{aligned}\mathbb{E} \left[ \|\mathbf{v}_{t+1} - \hat{\mathbf{v}}_{t+1}^*(\mathbf{x}_{t+1})\|^2 \right] &= \mathbb{E} \left[ \|\mathbf{v}_{t+1} - \hat{\mathbf{v}}_t^*(\mathbf{x}_t) + \hat{\mathbf{v}}_t^*(\mathbf{x}_t) - \hat{\mathbf{v}}_{t+1}^*(\mathbf{x}_{t+1})\|^2 \right] \\ &\leq (1 + \epsilon) \mathbb{E} \left[ \|\mathbf{v}_{t+1} - \hat{\mathbf{v}}_t^*(\mathbf{x}_t)\|^2 \right] \\ &\quad + \left( 1 + \frac{1}{\epsilon} \right) \mathbb{E} \left[ \|\hat{\mathbf{v}}_{t+1}^*(\mathbf{x}_{t+1}) - \hat{\mathbf{v}}_t^*(\mathbf{x}_t)\|^2 \right].\end{aligned}\tag{185}$$

From Lemma [D.16,](#page-57-5) we have, for any a >´ 0

$$\mathbb{E} \left[ \|\mathbf{v}_{t+1} - \hat{\mathbf{v}}_t^*(\mathbf{x}_t)\|^2 \right] \leq (1+\hat{\alpha}) \left( 1 - \delta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \right) \hat{\theta}_t^{\mathbf{y}} + \left( 1 + \frac{1}{\hat{\alpha}} \right) \delta_t^2 B_t. \quad (186)$$

Substituting [\(186\)](#page-59-1) into [\(185\)](#page-59-2), we get

$$\begin{aligned} \mathbb{E} \left[ \left\| \mathbf{v}_{t+1} - \hat{\mathbf{v}}_{t+1}^*(\mathbf{x}_{t+1}) \right\|^2 \right] &\leq (1 + \hat{\epsilon}) (1 + \hat{\alpha}) \left( 1 - \delta_t \frac{\mu_g \ell_{g,1}}{\mu_g + \ell_{g,1}} \right) \hat{\theta}_t^y \\ &\quad + (1 + \hat{\epsilon}) \left( 1 + \frac{1}{\hat{\alpha}} \right) \delta_t^2 B_t \\ &\quad + \left( 1 + \frac{1}{\hat{\epsilon}} \right) \mathbb{E} \left[ \left\| \hat{\mathbf{v}}_{t+1}^*(\mathbf{x}_{t+1}) - \hat{\mathbf{v}}_t^*(\mathbf{x}_t) \right\|^2 \right]. \end{aligned} \quad (187)$$

*3304*

*3306*

*3309* where Lµ<sup>g</sup> = µgℓg,<sup>1</sup> µg+ℓg,<sup>1</sup> .

*3314*

*3316*

*3319*

*3324*

*3329*

*3334*

*3336*

*3347 3348 3349* Then, substituting [\(192\)](#page-60-3) into [\(189\)](#page-60-0), rearranging the resulting inequality and summing over t ∈ [T], we obtain the desired result.

Choose c´ = δtLµg /4 1− δtLµg and a´ = δtLµg /2 1−δtLµg . Then, the following equations and inequalities are satisfied.

$$(1 + \hat{c})(1 + \hat{a})(1 - \delta_t L_{\mu_g}) = 1 - \frac{\delta_t L_{\mu_g}}{4},$$

$$(1 + \hat{c})\left(1 + \frac{1}{\hat{a}}\right) \leq \frac{4}{\delta_t L_{\mu_g}}, \quad (188)$$

$$1 + \frac{1}{\hat{a}} \leq \frac{2}{\delta_t L_{\mu_g}}, \quad 1 + \frac{1}{\hat{c}} \leq \frac{4}{\delta_t L_{\mu_g}},$$

Thus, we have

$$\begin{aligned} \mathbb{E} \left[ \left\| \mathbf{v}_{t+1} - \hat{\mathbf{v}}_{t+1}^*(\mathbf{x}_{t+1}) \right\|^2 \right] &\leq \left( 1 - \frac{\delta_t L_{\mu_g}}{4} \right) \hat{\theta}_t^\mathbf{v} + \frac{4}{L_{\mu_g}} \delta_t B_t \\ &\quad + \frac{4}{L_{\mu_g}} \frac{1}{\delta_t} \mathbb{E} \left[ \left\| \hat{\mathbf{v}}_{t+1}^*(\mathbf{x}_{t+1}) - \hat{\mathbf{v}}_t^*(\mathbf{x}_t) \right\|^2 \right]. \end{aligned} \quad (189)$$

We now bound the last term on the right-hand side of [\(189\)](#page-60-0). By Lemma [C.7,](#page-23-5) we have:

$$\begin{aligned} & \left\| \hat{\mathbf{v}}_{t+1}^*(\mathbf{x}_{t+1}) - \hat{\mathbf{v}}_t^*(\mathbf{x}_t) \right\|^2 \\ & \leq 2 \frac{\nu^2}{\mu_g^2} \left( \left\| \hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_{t+1}) - \hat{\mathbf{y}}_t^*(\mathbf{x}_t) \right\|^2 + \left\| \mathbf{x}_{t+1} - \mathbf{x}_t \right\|^2 \right) \\ & \leq 2 \frac{\nu^2}{\mu_g^2} \left( 2 \left\| \hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_{t+1}) - \hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_t) \right\|^2 \right. \\ & \quad \left. + 2 \left\| \hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_t) - \hat{\mathbf{y}}_t^*(\mathbf{x}_t) \right\|^2 + \left\| \mathbf{x}_{t+1} - \mathbf{x}_t \right\|^2 \right) \\ & \leq 2 \frac{\nu^2}{\mu_g^2} \left( 2 L_{\mathbf{y}}^2 \left\| \mathbf{x}_{t+1} - \mathbf{x}_t \right\|^2 + 2 \left\| \hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_t) - \hat{\mathbf{y}}_t^*(\mathbf{x}_t) \right\|^2 + \left\| \mathbf{x}_{t+1} - \mathbf{x}_t \right\|^2 \right), \end{aligned} \quad (190)$$

where the last inequality follows from Lemma [D.2.](#page-40-0)

From [\(154\)](#page-50-2), we have

$$\begin{aligned} \|\hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_t) - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 &\leq 3\|\hat{\mathbf{y}}_{t+1}^*(\mathbf{x}_t) - \mathbf{y}_{t+1}^*(\mathbf{x}_t)\|^2 \\ &\quad + 3\|\mathbf{y}_{t+1}^*(\mathbf{x}_t) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 + 3\|\mathbf{y}_t^*(\mathbf{x}_t) - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \\ &\leq 3\|\mathbf{y}_{t+1}^*(\mathbf{x}_t) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 + \frac{6\ell_{g,1}(\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2)}{\mu_g}. \end{aligned} \quad (191)$$

Plugging [\(191\)](#page-60-1) into [\(190\)](#page-60-2), we get

$$\begin{aligned} & \left\| \hat{\mathbf{v}}_{t+1}^*(\mathbf{x}_{t+1}) - \hat{\mathbf{v}}_t^*(\mathbf{x}_t) \right\|^2 \\ & \leq 4 \frac{\nu^2}{\mu_g^2} (2L_{\mathbf{y}}^2 + 1) \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 \\ & + 4 \frac{\nu^2}{\mu_g^2} \left( 3\|\mathbf{y}_{t+1}^*(\mathbf{x}_t) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 + \frac{6\ell_{g,1}(\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2)}{\mu_g} \right). \end{aligned} \quad (192)$$

*3359*

*3364*

*3366*

*3369*

*3374*

*3379*

*3384*

*3389* From ∥a + b∥ <sup>2</sup> ≤ 2 ∥a∥ <sup>2</sup> + ∥b∥ 2 , we get

*3390 3391*

*3394*

*3396*

*3403 3404* Lemma D.19. *Suppose Assumptions* [B2.](#page-3-12)*,* [B3.](#page-3-10)*,* [D2.](#page-4-16)*, and* [D4.](#page-4-17) *hold. Consider the sequence* {(xt, yt, vt)} T <sup>t</sup>=1 *generated by Algorithm [2,](#page-5-2) and define*

# D.5. Bounds on the Zeroth-Order Objective Function and its Projected Gradients

Lemma D.18. *Suppose Assumptions* [B2.](#page-3-12)*,* [B3.](#page-3-10) *hold. Then, for the sequence* {(xt, yt, vt)} T <sup>t</sup>=1 *generated by Algorithm [2,](#page-5-2) we have*

$$\begin{aligned} & \| \nabla_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1}) \|^2 \\ & \leq (12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{V}}^2})d_1 \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + (12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{V}}^2})d_1 \|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\ & + \frac{9}{2}d_1\ell_{g,1}^2 \|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + (3\ell_{f,1}^2 + \frac{3\ell_{g,1}^2}{4\rho_{\mathbf{V}}^2})d_1^2 \rho_{\mathbf{s}}^2, \end{aligned}$$

*where* ∇ˆ <sup>x</sup>ft+1 *and* ∇ˆ <sup>2</sup> xygt+1 *are defined in* [\(20b\)](#page-4-15) *and* [\(21b\)](#page-4-5)*, respectively.*

*Proof.* From Lemma [D.6,](#page-44-2) we have

$$\begin{aligned} & \|\hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1})\|^2 \\ & \leq 3d_1 \ell_{g,1}^2 \|\mathbf{z}_{t+1} - \mathbf{z}_t\|^2 + \frac{3}{2} \ell_{f,1}^2 d_1^2 \rho_s^2 \\ & \leq 6d_1 \ell_{f,1}^2 \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 6d_1 \ell_{f,1}^2 \|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 + \frac{3}{2} \ell_{f,1}^2 d_1^2 \rho_s^2. \end{aligned} \quad (193)$$

Moreover, from [\(21a\)](#page-4-4), we have

$$\begin{aligned} & \|\hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \hat{\nabla}_{\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1})\|^2 \\ &= \frac{1}{4\rho_{\mathbf{v}}^2} \|\hat{\nabla}_{\mathbf{x}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1} + \rho_{\mathbf{v}} \mathbf{v}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \hat{\nabla}_{\mathbf{x}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t; \bar{\mathcal{B}}_{t+1})\|^2 \\ &\leq \frac{3}{4\rho_{\mathbf{v}}^2} d_1 \ell_{g,1}^2 \|(\mathbf{x}_{t+1}, \mathbf{y}_{t+1} + \rho_{\mathbf{v}} \mathbf{v}_{t+1}) - (\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 + \frac{3}{8\rho_{\mathbf{v}}^2} \ell_{g,1}^2 d_1^2 \rho_{\mathbf{s}}^2 \\ &\leq \frac{9}{4\rho_{\mathbf{v}}^2} d_1 \ell_{g,1}^2 \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + \frac{9}{4\rho_{\mathbf{v}}^2} d_1 \ell_{g,1}^2 \|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\ &+ \frac{9}{4} d_1 \ell_{g,1}^2 \|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + \frac{3}{8\rho_{\mathbf{v}}^2} \ell_{g,1}^2 d_1^2 \rho_{\mathbf{s}}^2, \end{aligned} \tag{194}$$

where the first inequality follows from Lemma [D.6.](#page-44-2)

$$\begin{aligned} & \|\hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1})\|^2 \\ & \leq 2\|\hat{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \hat{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1})\|^2 \\ & + 2\|\hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1})\|^2 \\ & \leq (12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2})d_1 \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + (12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2})d_1 \|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\ & + \frac{9}{2}d_1 \ell_{g,1}^2 \|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + (3\ell_{f,1}^2 + \frac{3\ell_{g,1}^2}{4\rho_{\mathbf{v}}^2})d_1^2 \rho_{\mathbf{s}}^2, \end{aligned}$$

where the second inequality follows from [\(193\)](#page-61-1) and [\(194\)](#page-61-2).

$$e_t^L := \nabla_{\mathbf{x}, f_t, \rho}(\mathbf{x}_t, \mathbf{y}_t) + \tilde{\nabla}_{\mathbf{x}, \mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) - \hat{\mathbf{d}}_t^*, \quad \text{where} \quad (195)$$

$$\tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t) = \frac{1}{2\rho_{\mathbf{v}}} (\nabla_{\mathbf{x}} g_t, \rho(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{x}} g_t, \rho(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)). \quad (196)$$

3431 *Proof.* According to the definition of dˆ<sup>x</sup> t in Algorithm [2,](#page-5-2) we have

3438 Then, we have

$$\begin{aligned}
& 3439 & \mathbb{E} \|\nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{d}}_{t+1}^{\mathbf{x}}\|^2 \\
& 3440 & = \mathbb{E} \|\nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{d}}_t^{\mathbf{x}} - (\hat{\mathbf{d}}_{t+1}^{\mathbf{x}} - \hat{\mathbf{d}}_t^{\mathbf{x}})\|^2 \\
& 3441 & = \mathbb{E} \|\nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{d}}_t^{\mathbf{x}} + \eta_{t+1} \hat{\mathbf{d}}_t^{\mathbf{x}} \\
& 3442 & = \mathbb{E} \|\nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{d}}_t^{\mathbf{x}} + \eta_{t+1} \hat{\mathbf{d}}_t^{\mathbf{x}} \\
& 3443 & = \mathbb{E} \|\nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{d}}_t^{\mathbf{x}} + \eta_{t+1} \hat{\mathbf{d}}_t^{\mathbf{x}} \\
& 3444 & = -\eta_{t+1} (\hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1})) \\
& 3445 & = -(1 - \eta_{t+1}) \left( \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) \right) \\
& 3446 & = -\hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1}) \Big) \|^2 \\
& 3447 & = \mathbb{E} \|(1 - \eta_{t+1})(\nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{xy}}^2 g_t(\mathbf{z}_t) - \hat{\mathbf{d}}_t^{\mathbf{x}}) \\
& 3448 & + \eta_{t+1}(\nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1})) \\
& 3449 & = \mathbb{E} \|(1 - \eta_{t+1})(\nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{xy}}^2 g_t(\mathbf{z}_t) - \hat{\mathbf{d}}_t^{\mathbf{x}}) \\
& 3450 & + \eta_{t+1}(\nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1})) \\
& 3451 & + (1 - \eta_{t+1}) \left( \nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{xy}}^2 g_t(\mathbf{z}_t) \right) \\
& 3452 & + (1 - \eta_{t+1}) \left( \nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{xy}}^2 g_t(\mathbf{z}_t) \right) \\
& 3453 & + \nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_t) - \nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_t) \\
& 3454 & = -\hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_$$

3457 3458 Since

$$\begin{aligned} & ^{3459} \left[ \nabla_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; B_{t+1}) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{B}_{t+1}) \right] = \nabla_{\mathbf{x}} f_{t+1, \rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}), \\ & ^{3460} \left[ \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; B_{t+1}) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{B}_{t+1}) - \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_t; B_{t+1}) - \hat{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{B}_{t+1}) \right] \\ & ^{3461} = \nabla_{\mathbf{x}} f_{t+1, \rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \nabla_{\mathbf{x}} f_{t+1, \rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t), \end{aligned}$$

*Then, we have*

$$\begin{aligned} \mathbb{E}\|e_{t+1}^L\|^2 &\leq (1 - \eta_{t+1})^2 \mathbb{E}\|e_t^L\|^2 + 36\mathbb{E}\|\nabla_{\mathbf{x}}f_{t+1}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}}f_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ &\quad + \left( 18d_1^2 \ell_{f,1}^2 + 6(3\ell_{f,1}^2 + \frac{3\ell_{g,1}^2}{4\rho_{\mathbf{v}}^2})d_1^2 \right) \rho_{\mathbf{s}}^2 + 18d_1^2 \ell_{g,1}^2 \frac{\rho_{\mathbf{s}}^2}{\rho_{\mathbf{v}}^2} \\ &\quad + \frac{18}{\rho_{\mathbf{v}}^2} \mathbb{E}\|\nabla_{\mathbf{x}}g_{t+1}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}}\mathbf{v}_t) - \nabla_{\mathbf{x}}g_t(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}}\mathbf{v}_t)\|^2 \\ &\quad + \frac{18}{\rho_{\mathbf{v}}^2} \mathbb{E}\|\nabla_{\mathbf{x}}g_{t+1}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}}\mathbf{v}_t) - \nabla_{\mathbf{x}}g_t(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}}\mathbf{v}_t)\|^2 \\ &\quad + 6(12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2})d_1 \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 6(12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2})d_1 \mathbb{E}\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\ &\quad + 27d_1 \ell_{g,1}^2 \mathbb{E}\|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + 3(\frac{\hat{\sigma}_{g_{\mathbf{x}}}^2}{b\rho_{\mathbf{v}}^2} + \frac{\hat{\sigma}_{f_{\mathbf{x}}}^2}{b})\eta_{t+1}^2. \end{aligned} \tag{197}$$

$$\begin{aligned} \hat{\mathbf{a}}_{t+1}^x - \hat{\mathbf{a}}_t^x &= -\eta_{t+1} \hat{\mathbf{a}}_t^x + \eta_{t+1} (\hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1})) \\ &\quad + (1 - \eta_{t+1}) (\hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1})) \\ &\quad - \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1})). \end{aligned}$$

$$\begin{aligned}
3466 \quad & \mathbb{E} \|\nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{d}}_{t+1}^x\|^2 \\
3467 \quad & = (1 - \eta_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{z}_t) - \hat{\mathbf{d}}_t^x\|^2 \\
3468 \quad & + \|\eta_{t+1}(\nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1})) \\
3479 \quad & + (1 - \eta_{t+1}) \left( \nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{z}_t) \right. \\
3478 \quad & \left. + \nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) - \nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) \right. \\
3479 \quad & \left. - \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) + \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1}) \right) \|^2 \\
3476 \quad & \leq (1 - \eta_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{z}_t) - \hat{\mathbf{d}}_t^x\|^2 \\
3477 \quad & + 3(1 - \eta_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{z}_t) \\
3478 \quad & + \nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) - \nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) \\
3479 \quad & - \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) - \hat{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) + \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1})\|^2 \\
3480 \quad & + 3\eta_{t+1}^2 \mathbb{E} \|\nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_{t+1}) - \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1})\|^2 \\
3481 \quad & + 3\eta_{t+1}^2 \mathbb{E} \|\nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{z}_{t+1}) - \hat{\nabla}_{\mathbf{x}} f_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1})\|^2, \\
3482 \quad & + 3\eta_{t+1}^2 \mathbb{E} \|\tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1})\|^2, \\
3483 \quad & (198)
\end{aligned}$$

3504

3506

$$\begin{aligned}
3509 \quad & \mathbb{E}\|\nabla_{\mathbf{x}}f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{d}}_{t+1}^{\mathbf{x}}\|^2 \\
3510 \quad & \leq (1 - \eta_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{x}}f_{t,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{z}_t) - \hat{\mathbf{d}}_t^{\mathbf{x}}\|^2 \\
3511 \quad & + 12(1 - \eta_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{x}}f_{t+1,\rho}(\mathbf{z}_t) - \nabla_{\mathbf{x}}f_{t,\rho}(\mathbf{z}_t)\|^2 + 12(1 - \eta_{t+1})^2 \mathbb{E}\|\tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{z}_t)\|^2 \\
3512 \quad & + 6(12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2})d_1\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 6(12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2})d_1\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\
3514 \quad & + 6(12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2})d_1\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 6(12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2})d_1\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\
3515 \quad & + 27d_1\ell_{g,1}^2\|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + 6(3\ell_{f,1}^2 + \frac{3\ell_{g,1}^2}{4\rho_{\mathbf{v}}^2})d_1^2\rho_s^2 + 3\eta_{t+1}^2(\frac{\hat{\sigma}_{g_{\mathbf{x}}}^2}{b} + \frac{\hat{\sigma}_{f_{\mathbf{x}}}^2}{b}). \\
3516 \quad & + 27d_1\ell_{g,1}^2\|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + 6(3\ell_{f,1}^2 + \frac{3\ell_{g,1}^2}{4\rho_{\mathbf{v}}^2})d_1^2\rho_s^2 + 3\eta_{t+1}^2(\frac{\hat{\sigma}_{g_{\mathbf{x}}}^2}{b} + \frac{\hat{\sigma}_{f_{\mathbf{x}}}^2}{b}). \\
3517 \quad & \\
3518 \quad & \\
\end{aligned}$$

then, we have

where the second inequality holds by Cauchy-Schwarz inequality.

Note that for the last term on the right-hand side of [\(198\)](#page-63-0), using [\(196\)](#page-61-3) and [\(21b\)](#page-4-5), we have

$$\begin{aligned} & \|\hat{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\nabla}_{\mathbf{xy}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1})\|^2 \\ & \leq 2 \left\| \frac{1}{2\rho_{\mathbf{v}}} (\nabla_{\mathbf{x}} g_{t+1,\rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1} + \rho_{\mathbf{v}} \mathbf{v}_{t+1}) - \hat{\nabla}_{\mathbf{x}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1} + \rho_{\mathbf{v}} \mathbf{v}_{t+1}; \bar{\mathcal{B}}_{t+1})) \right\|^2 \\ & + 2 \left\| \frac{1}{2\rho_{\mathbf{v}}} (\hat{\nabla}_{\mathbf{x}} g_{t+1}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1} - \rho_{\mathbf{v}} \mathbf{v}_{t+1}; \bar{\mathcal{B}}_{t+1}) - \nabla_{\mathbf{x}} g_{t+1,\rho}(\mathbf{x}_{t+1}, \mathbf{y}_{t+1} - \rho_{\mathbf{v}} \mathbf{v}_{t+1})) \right\|^2 \\ & \leq \frac{\hat{\sigma}_{g_{\mathbf{x}}}^2}{\bar{b}\rho_{\mathbf{v}}^2}, \end{aligned}$$

where the last inequality follows from Assumption [4.1.](#page-4-6)

Then, from <sup>E</sup>∥a − <sup>E</sup>[a]∥ <sup>2</sup> = <sup>E</sup>∥a∥ <sup>2</sup> − ∥E[a]∥ 2 and Assumption [4.1,](#page-4-6) we have

$$\begin{aligned} & \mathbb{E}\|\nabla_{\mathbf{x}}f_{t+1,\rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{d}}_{t+1}^{\mathbf{x}}\|^2 \\ & \leq (1 - \eta_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{x}}f_{t,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{z}_t) - \hat{\mathbf{d}}_t^{\mathbf{x}}\|^2 \\ & + 6(1 - \eta_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{x}}f_{t+1,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t) - \nabla_{\mathbf{x}}f_{t,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{z}_t)\|^2 \\ & + 6(1 - \eta_{t+1})^2 \mathbb{E}\|\hat{\nabla}_{\mathbf{x}}f_{t+1}(\mathbf{z}_{t+1}; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}; \bar{\mathcal{B}}_{t+1}) \\ & + \hat{\nabla}_{\mathbf{x}}f_{t+1}(\mathbf{z}_t; \mathcal{B}_{t+1}) + \hat{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_t; \bar{\mathcal{B}}_{t+1})\|^2 + 3\eta_{t+1}^2 \left(\frac{\hat{\sigma}_{g_{\mathbf{x}}}^2}{b\rho_{\mathbf{x}}^2} + \frac{\hat{\sigma}_{f_{\mathbf{x}}}^2}{b}\right), \end{aligned} \tag{199}$$

Then, from Young's inequality and Lemma [D.18,](#page-61-4) we have

3526

3529

3534

3536

3543 3544 Substituting these inequalities in [\(201\)](#page-64-1), we have

3554

3556

3559 3560

3564

3566

3569

For the third term on the right-hand side of [\(199\)](#page-63-1), we have

$$\begin{aligned} & \|\tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{x}_t, \mathbf{y}_t) - \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & \leq \frac{1}{2\rho_{\mathbf{V}}^2} \|\nabla_{\mathbf{x}} g_{t+1, \rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{V}} \mathbf{v}_t) - \nabla_{\mathbf{x}} g_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{V}} \mathbf{v}_t)\|^2 \end{aligned} \quad (201a)$$

$$+ \frac{1}{2\rho_v^2} \|\nabla_{\mathbf{x}} g_t, \rho(\mathbf{x}_t, \mathbf{y}_t - \rho_v \mathbf{v}_t) - \nabla_{\mathbf{x}} g_{t+1}, \rho(\mathbf{x}_t, \mathbf{y}_t - \rho_v \mathbf{v}_t)\|^2. \quad (201b)$$

For [\(201a\)](#page-64-0), we get

$$\begin{aligned} & \|\nabla_{\mathbf{x}} g_{t+1, \boldsymbol{\rho}}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{x}} g_{t, \boldsymbol{\rho}}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \\ & \leq 3 \|\nabla_{\mathbf{x}} g_{t+1, \boldsymbol{\rho}}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{x}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \\ & + 3 \|\nabla_{\mathbf{x}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{x}} g_t(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \\ & + 3 \|\nabla_{\mathbf{x}} g_t(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{x}} g_{t, \boldsymbol{\rho}}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \\ & \leq 3 \|\nabla_{\mathbf{x}} g_t(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{x}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 + \frac{3 \rho_{\mathbf{s}}^2 d_1^2 \ell_{g_1}^2}{2}, \end{aligned}$$

where the last inequality follows from Lemma [131.](#page-43-0)

Similary, for [\(163b\)](#page-54-2), we have

$$\begin{aligned} & \|\nabla_{\mathbf{x}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{x}} g_{t+1,\rho}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \\ & \leq 3 \|\nabla_{\mathbf{x}} g_t(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{x}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)\| + \frac{3\rho_s^2 d_1^2 \ell_{g_1}^2}{2}. \end{aligned}$$

$$\begin{aligned} & \|\tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{x}_t, \mathbf{y}_t) - \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & \leq \frac{3}{2\rho_{\mathbf{v}}^2} \|\nabla_{\mathbf{x}} g_t(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{x}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \\ & + \frac{3}{2\rho_{\mathbf{v}}^2} \|\nabla_{\mathbf{x}} g_t(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{x}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 + \frac{3\rho_{\mathbf{s}}^2 d_1^2 \ell_{g,1}^2}{2\rho_{\mathbf{v}}^2}. \end{aligned} \quad (202)$$

For the second term on the right-hand side of [\(199\)](#page-63-1), we have

$$\begin{aligned} & \|\nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & \leq 3 \|\nabla_{\mathbf{x}} f_{t+1,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}} f_{t+1}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & + 3 \|\nabla_{\mathbf{x}} f_{t+1}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}} f_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & + 3 \|\nabla_{\mathbf{x}} f_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}} f_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ & \leq 3 \|\nabla_{\mathbf{x}} f_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}} f_{t+1}(\mathbf{x}_t, \mathbf{y}_t)\|^2 + \frac{3\rho_s^2 d_1^2 \ell_{f,1}^2}{2}, \end{aligned} \quad (203)$$

where the last inequality follows from Eq. [\(133\)](#page-44-0).

$$\begin{aligned}
3576 & \quad \mathbb{E} \|\nabla_{\mathbf{x}} f_{t+1, \rho}(\mathbf{z}_{t+1}) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_{t+1}(\mathbf{z}_{t+1}) - \hat{\mathbf{d}}_{t+1}^{\mathbf{x}}\|^2 \\
3577 & \leq (1 - \eta_{t+1})^2 \mathbb{E} \|\nabla_{\mathbf{x}} f_{t, \rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{x}\mathbf{y}}^2 g_t(\mathbf{z}_t) - \hat{\mathbf{d}}_t^{\mathbf{x}}\|^2 \\
3579 & + 36 \mathbb{E} \|\nabla_{\mathbf{x}} f_t(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}} f_{t+1}(\mathbf{x}_t, \mathbf{y}_t)\|^2 + 18 \rho_s^2 d_1^2 \ell_{f,1}^2 \\
3580 & + \frac{18}{\rho_{\mathbf{v}}^2} \mathbb{E} \|\nabla_{\mathbf{x}} g_t(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{x}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \\
3582 & + \frac{18}{\rho_{\mathbf{v}}^2} \mathbb{E} \|\nabla_{\mathbf{x}} g_t(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{x}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 + \frac{18 \rho_s^2 d_1^2 \ell_{g,1}^2}{\rho_{\mathbf{v}}^2} \\
3583 & + 6(12 \ell_{f,1}^2 + \frac{9 \ell_{g,1}^2}{2 \rho_{\mathbf{v}}^2}) d_1 \mathbb{E} \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 6(12 \ell_{f,1}^2 + \frac{9 \ell_{g,1}^2}{2 \rho_{\mathbf{v}}^2}) d_1 \mathbb{E} \|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\
3584 & + 6(2 \ell_{f,1}^2 + \frac{9 \ell_{g,1}^2}{2 \rho_{\mathbf{v}}^2}) d_1 \mathbb{E} \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 6(2 \ell_{f,1}^2 + \frac{9 \ell_{g,1}^2}{2 \rho_{\mathbf{v}}^2}) d_1 \mathbb{E} \|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\
3585 & + 27 d_1 \ell_{g,1}^2 \mathbb{E} \|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + 6(3 \ell_{f,1}^2 + \frac{3 \ell_{g,1}^2}{4 \rho_{\mathbf{v}}^2}) d_1^2 \rho_s^2 + 3 \eta_{t+1}^2 (\frac{\hat{\sigma}_{g_{\mathbf{x}}}^2}{b} + \frac{\hat{\sigma}_{f_{\mathbf{x}}}^2}{b}). \\
3588 & + 27 d_1 \ell_{g,1}^2 \mathbb{E} \|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + 6(3 \ell_{f,1}^2 + \frac{3 \ell_{g,1}^2}{4 \rho_{\mathbf{v}}^2}) d_1^2 \rho_s^2 + 3 \eta_{t+1}^2 (\frac{\hat{\sigma}_{g_{\mathbf{x}}}^2}{b} + \frac{\hat{\sigma}_{f_{\mathbf{x}}}^2}{b}). \\
3589 & + 27 d_1 \ell_{g,1}^2 \mathbb{E} \|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + 6(3 \ell_{f,1}^2 + \frac{3 \ell_{g,1}^2}{4 \rho_{\mathbf{v}}^2}) d_1^2 \rho_s^2 + 3 \eta_{t+1}^2 (\frac{\hat{\sigma}_{g_{\mathbf{x}}}^2}{b} + \frac{\hat{\sigma}_{f_{\mathbf{x}}}^2}{b}).
\end{aligned}$$

3594 3596 Lemma D.20. *Suppose Assumptions [3.2,](#page-3-2)* [B2.](#page-3-12)*,* [B3.](#page-3-10)*, and [3.4](#page-3-4) hold. Then, for the sequence of functions* {ft,ρ} T <sup>t</sup>=1 *defined in Eq.* [\(17\)](#page-4-1)*, we have*

3603 *Here,* V<sup>T</sup> *is defined in* [\(10\)](#page-3-0)*; and* M *is defined in Assumption [3.4.](#page-3-4)*

3604 3605 3606

3607 3608 *Proof.* Note that, we have

3624

3626 and

From [\(202\)](#page-64-2), [\(203\)](#page-64-3) and [\(200\)](#page-63-2), we get

$$\begin{aligned} & \sum_{t=1}^T (f_{t,\rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) - f_{t,\rho}(\mathbf{x}_{t+1}, \hat{\mathbf{y}}_t^*(\mathbf{x}_{t+1}))) \\ & \leq 2M + V_T + \ell_{f,1} \left( 1 + 2 \frac{\ell_{g,1}}{\mu_g} \right) T (\rho_s^2 + \rho_r^2). \end{aligned}$$

$$\begin{aligned} & \sum_{t=1}^T (f_{t,\rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) - f_{t,\rho}(\mathbf{x}_{t+1}, \hat{\mathbf{y}}_t^*(\mathbf{x}_{t+1}))) \\ &= \sum_{t=1}^T (f_{t,\rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) - f_t(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t))) \\ &+ \sum_{t=1}^T (f_t(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) - f_t(\mathbf{x}_{t+1}, \hat{\mathbf{y}}_t^*(\mathbf{x}_{t+1}))) \end{aligned} \quad (204)$$

$$(204)$$

$$+ \sum_{t=1}^T (f_t(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) - f_t(\mathbf{x}_{t+1}, \hat{\mathbf{y}}_t^*(\mathbf{x}_{t+1}))) \quad (205)$$

$$+ \sum_{t=1}^T (f_t(\mathbf{x}_{t+1}, \hat{\mathbf{y}}_t^*(\mathbf{x}_{t+1})) - f_{t,\rho}(\mathbf{x}_{t+1}, \hat{\mathbf{y}}_t^*(\mathbf{x}_{t+1}))). \quad (206)$$

From [\(127\)](#page-41-2), we have

$$(204) \leq T \frac{\ell_{f,1}(\rho_s^2 + \rho_r^2)}{2}, \quad (207)$$

$$(206) \leq T \frac{\ell_{f,1}(\rho_s^2 + \rho_r^2)}{2}. \quad (208)$$

3636

3649 For the last term of the above inequality, we have

3654

3656

3658 3659 which implies that

3660 3661 3662

3663 From [\(207\)](#page-65-3), [\(208\)](#page-65-4), and [\(210\)](#page-66-1), we get the desired result.

3664

3665 3666 3667 Lemma D.21. *Suppose that Assumptions [3.2](#page-3-2) and [3.3](#page-3-3) hold. Let* ft,<sup>ρ</sup> *be defined as in* [\(17\)](#page-4-1)*. Then, for* dˆ<sup>x</sup> <sup>t</sup> *generated by Algorithm [2,](#page-5-2) for all* t ∈ [T]*, we have*

3668 3669

3674

3675 3676 *Proof.* From ∥a + b∥ <sup>2</sup> ≤ 2 ∥a∥ <sup>2</sup> + ∥b∥ 2 , we get

3679

Moreover, from Lemma [D.7,](#page-44-3) we have

$$\begin{aligned}
 (205) &= \sum_{t=1}^T (f_t(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) - f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))) \\
 &+ \sum_{t=1}^T (f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - f_t(\mathbf{x}_{t+1}, \mathbf{y}_t^*(\mathbf{x}_{t+1}))) \\
 &+ \sum_{t=1}^T (f_t(\mathbf{x}_{t+1}, \mathbf{y}_t^*(\mathbf{x}_{t+1})) - f_t(\mathbf{x}_{t+1}, \hat{\mathbf{y}}_t^*(\mathbf{x}_{t+1}))) \\
 &\leq \ell_{f,1} \sum_{t=1}^T \|\hat{\mathbf{y}}_t^*(\mathbf{x}_t) - \mathbf{y}_t^*(\mathbf{x}_t)\| + \ell_{f,1} \sum_{t=1}^T \|\hat{\mathbf{y}}_t^*(\mathbf{x}_{t+1}) - \mathbf{y}_t^*(\mathbf{x}_{t+1})\| \\
 &+ \sum_{t=1}^T (f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - f_t(\mathbf{x}_{t+1}, \mathbf{y}_t^*(\mathbf{x}_{t+1}))) \\
 &\leq 2T\ell_{f,1} \frac{\ell_{g,1}(\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2)}{\mu_g} + \sum_{t=1}^T (f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - f_t(\mathbf{x}_{t+1}, \mathbf{y}_t^*(\mathbf{x}_{t+1}))). \tag{209}
 \end{aligned}$$

$$\begin{aligned} \sum_{t=1}^T (f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - f_t(\mathbf{x}_{t+1}, \mathbf{y}_t^*(\mathbf{x}_{t+1}))) &= f_1(\mathbf{x}_1, \mathbf{y}_1^*(\mathbf{x}_1)) - f_T(\mathbf{x}_{T+1}, \mathbf{y}_T^*(\mathbf{x}_{T+1})) \\ &\quad + \sum_{t=2}^T (f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - f_{t-1}(\mathbf{x}_t, \mathbf{y}_{t-1}^*(\mathbf{x}_t))) \\ &\leq 2M + V_T, \end{aligned}$$

$$(205) \leq 2T\ell_{f,1} \frac{\ell_{g,1}(\rho_s^2 + \rho_r^2)}{\mu_g} + 2M + V_T. \quad (210)$$

$$\mathbb{E} \left[ \left\| \hat{\mathbf{d}}_t^x - \nabla f_{t,\boldsymbol{\rho}}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) \right\|^2 \right] \leq 4\mathbb{E} \left[ \|e_t^L\|^2 \right] + 4\ell_{g,2}^2 \rho_{\mathbf{v}}^2 p^4 \\ + 2M_f^2 \left( \mathbb{E}[\hat{\theta}_t^Y] + \mathbb{E}[\hat{\theta}_t^Y] \right) := A_t, \quad (211)$$

*where* e L t *is defined in Lemma [D.15,](#page-56-0) and* ˆθ y t *,* ˆθ v <sup>t</sup> *are as defined in* [\(146\)](#page-49-1)*. Additionally,* M<sup>f</sup> *is given in Lemma [D.2.](#page-40-0)*

E

$$\begin{aligned} & \mathbb{E} \left[ \left\| \hat{\mathbf{a}}_t^x - \nabla f_{t,\rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) \right\|^2 \right] \\ & \leq 2\mathbb{E} \left[ \left\| \hat{\mathbf{a}}_t^x - \mathbf{d}_{t,\rho}^x \right\|^2 \right] \end{aligned} \quad (212a)$$

dˆx

<sup>t</sup> − ∇ft,ρ(xt, yˆ

∗ t (xt)) 2

$$+ 2\mathbb{E} \left[ \left\| \mathbf{d}_{t,\rho}^x - \nabla f_{t,\rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) \right\|^2 \right], \quad (212b)$$

3689 Moreover, from Eq. [\(123a\)](#page-40-3), we get

3690

3694

3696 Lemma D.22. *Suppose Assumptions [3.2,](#page-3-2) [3.3,](#page-3-3) and [3.4](#page-3-4) hold. Let the sequence of functions* {ft,ρ} T <sup>t</sup>=1 *be defined in*[\(17\)](#page-4-1)*, and* P<sup>X</sup> ,α<sup>t</sup> *be given in Definition [B.1.](#page-11-0) Then, for any positive choice of step sizes as* α<sup>t</sup> ≤ 1/4L<sup>f</sup> *, for all* t ∈ [T]*, Algorithm [2](#page-5-2) guarantees the following bound:*

3699 3700

3704

3706

3709

3714

3716

3718

3719 For the first term on the R.H.S of Eq. [\(216\)](#page-67-3), we have that

3724

3726

3729

3734

3736

where d x t,ρ is defined in [\(120c\)](#page-40-10). From Lemma [D.15,](#page-56-0) we have

$$(212a) \leq 4\mathbb{E} \left[ \|e_t^L\|^2 \right] + 4\ell_{g,2}^2 \rho_{\mathbf{v}}^2 p^4. \quad (213)$$

$$(212b) \leq 2M_f^2 \left( \mathbb{E}[\hat{\theta}_t^Y] + \mathbb{E}[\hat{\theta}_t^Y] \right). \quad (214)$$

Substituting [\(213\)](#page-67-1) and [\(214\)](#page-67-2) into [\(212\)](#page-66-4), we conclude the desired result.

$$\begin{aligned} & \sum_{t=1}^T (\alpha_t - L_f \alpha_t^2) \mathbb{E} \left[ \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \boldsymbol{\rho}}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \right] \\ & \leq 12M + 6V_T + \sum_{t=1}^T (6\alpha_t - 3L_f \alpha_t^2) A_t \\ & + \sum_{t=1}^T \left( 6\ell_{f,1} (1 + 2 \frac{\ell_{g,1}}{\mu_g}) + \frac{3\ell_{f,1}\ell_{g,1}}{\mu_g} (\alpha_t - L_f \alpha_t^2) \right) (\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2), \end{aligned} \quad (215)$$

*where* V<sup>T</sup> *and* A<sup>t</sup> *are respectively defined in Eq.* [\(10\)](#page-3-0) *and Lemma [D.21.](#page-66-0)*

*Proof.* Due to the L<sup>f</sup> -smoothness of f<sup>t</sup> function by Lemma [C.1,](#page-15-6) ft,<sup>ρ</sup> is L<sup>f</sup> -smooth as well. Hence,

$$\begin{aligned} & f_{t,\rho}(\mathbf{x}_{t+1}, \hat{\mathbf{y}}_t^*(\mathbf{x}_{t+1})) - f_{t,\rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) \\ & \leq \langle \nabla f_{t,\rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)), \mathbf{x}_{t+1} - \mathbf{x}_t \rangle + \frac{L_f}{2} \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 \\ & = -\alpha_t \langle \nabla f_{t,\rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)), \mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \hat{\mathbf{d}}_t^*) \rangle + \frac{L_f \alpha_t^2}{2} \left\| \mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \hat{\mathbf{d}}_t^*) \right\|^2. \end{aligned} \quad (216)$$

$$\begin{aligned} & -\mathbb{E} \left\langle \nabla f_{t,\rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)), \mathcal{P}_{\mathcal{X},\alpha_t}(\mathbf{x}_t; \hat{\mathbf{d}}_t^{\mathbf{x}}) \right\rangle \\ &= -\mathbb{E} \left\langle \hat{\mathbf{d}}_t^{\mathbf{x}}, \mathcal{P}_{\mathcal{X},\alpha_t}(\mathbf{x}_t; \hat{\mathbf{d}}_t^{\mathbf{x}}) \right\rangle \\ & - \mathbb{E} \left\langle \nabla f_{t,\rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) - \hat{\mathbf{d}}_t^{\mathbf{x}}, \mathcal{P}_{\mathcal{X},\alpha_t}(\mathbf{x}_t; \hat{\mathbf{d}}_t^{\mathbf{x}}) \right\rangle \\ & \leq -\frac{1}{2} \mathbb{E} \left[ \left\| \mathcal{P}_{\mathcal{X},\alpha_t}(\mathbf{x}_t; \hat{\mathbf{d}}_t^{\mathbf{x}}) \right\|^2 \right] + \frac{1}{2} \mathbb{E} \left[ \left\| \hat{\mathbf{d}}_t^{\mathbf{x}} - \nabla f_{t,\rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) \right\|^2 \right] \\ & \leq -\frac{1}{2} \mathbb{E} \left[ \left\| \mathcal{P}_{\mathcal{X},\alpha_t}(\mathbf{x}_t; \hat{\mathbf{d}}_t^{\mathbf{x}}) \right\|^2 \right] + \frac{A_t}{2}, \end{aligned} \tag{217}$$

where the first inequality follows from Lemma [B.8;](#page-12-1) the last inequality follows from Lemma [D.21.](#page-66-0)

Plugging the bound [\(217\)](#page-67-4) into [\(216\)](#page-67-3), we have that

$$\begin{aligned} & \mathbb{E} [f_{t,\rho}(\mathbf{x}_{t+1}, \hat{\mathbf{y}}_t^*(\mathbf{x}_{t+1})) - f_{t,\rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t))] \\ & \leq \frac{(L_f \alpha_t^2 - \alpha_t)}{2} \mathbb{E} \left[ \left\| \mathcal{P}_{\mathcal{X}, \alpha_t} \left( \mathbf{x}_t; \hat{\mathbf{d}}_t^{\mathbf{x}} \right) \right\|^2 \right] + \frac{\alpha_t A_t}{2}, \end{aligned}$$

*3745 3746* In addition, we have

*3747 3748 3749*

*3754*

*3756*

*3759 3760*

*3764*

*3769*

*3774*

*3776*

*3779*

*3784*

*3789 3790 3791*

which can be rearranged into

$$\begin{aligned} & (\alpha_t - L_f \alpha_t^2) \mathbb{E} \left[ \left\| \mathcal{P}_{\mathcal{X}, \alpha_t} \left( \mathbf{x}_t; \hat{\mathbf{d}}_t^{\mathbf{x}} \right) \right\|^2 \right] \\ & \leq 2 \mathbb{E} [f_{t, \rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) - f_{t, \rho}(\mathbf{x}_{t+1}, \hat{\mathbf{y}}_t^*(\mathbf{x}_{t+1}))] + \alpha_t A_t. \end{aligned} \quad (218)$$

$$\begin{aligned} & \mathbb{E} \left[ \left\| \mathcal{P}_{\mathcal{X}, \alpha_t} (\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))) \right\|^2 \right] \\ & \leq 3\mathbb{E} \left[ \left\| \mathcal{P}_{\mathcal{X}, \alpha_t} \left( \mathbf{x}_t; \hat{\mathbf{d}}_t^{\mathbf{x}} \right) - \mathcal{P}_{\mathcal{X}, \alpha_t} (\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t))) \right\|^2 \right] \\ & + 3\mathbb{E} \left[ \left\| \mathcal{P}_{\mathcal{X}, \alpha_t} (\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t))) - \mathcal{P}_{\mathcal{X}, \alpha_t} (\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))) \right\|^2 \right] \\ & + 3\mathbb{E} \left[ \left\| \mathcal{P}_{\mathcal{X}, \alpha_t} \left( \mathbf{x}_t; \hat{\mathbf{d}}_t^{\mathbf{x}} \right) \right\|^2 \right] \\ & \leq 3\mathbb{E} \left[ \left\| \hat{\mathbf{d}}_t^{\mathbf{x}} - \nabla f_{t, \rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) \right\|^2 \right] \\ & + 3\mathbb{E} \left[ \left\| \nabla f_{t, \rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) - \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) \right\|^2 \right] \\ & + 3\mathbb{E} \left[ \left\| \mathcal{P}_{\mathcal{X}, \alpha_t} \left( \mathbf{x}_t; \hat{\mathbf{d}}_t^{\mathbf{x}} \right) \right\|^2 \right], \end{aligned}$$

where the second inequaliy follows from non-expansiveness of the projection operator.

Then, from Lemma [D.21](#page-66-0) and Assumption [3.3,](#page-3-3) we have

$$\begin{aligned} & \mathbb{E} \left[ \left\| \mathcal{P}_{\mathcal{X}, \alpha_t} (\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))) \right\|^2 \right] \\ & \leq 3A_t + 3\ell_{f,1} \mathbb{E} \left[ \left\| \hat{\mathbf{y}}_t^*(\mathbf{x}_t) - \mathbf{y}_t^*(\mathbf{x}_t) \right\|^2 \right] + 3\mathbb{E} \left[ \left\| \mathcal{P}_{\mathcal{X}, \alpha_t} (\mathbf{x}_t; \hat{\mathbf{d}}_t^{\mathbf{x}}) \right\|^2 \right] \\ & \leq 3A_t + 3\ell_{f,1} \frac{\ell_{g,1}(\rho_s^2 + \rho_r^2)}{\mu_g} + 3\mathbb{E} \left[ \left\| \mathcal{P}_{\mathcal{X}, \alpha_t} (\mathbf{x}_t; \hat{\mathbf{d}}_t^{\mathbf{x}}) \right\|^2 \right], \end{aligned} \quad (219)$$

where the last inequality is by Lemma [D.7.](#page-44-3)

Combining [\(218\)](#page-68-0) and [\(219\)](#page-68-1) and summing over t = 1 to T, we have

$$\begin{aligned} & \sum_{t=1}^T (\alpha_t - L_f \alpha_t^2) \mathbb{E} \left[ \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \right] \\ & \leq 6 \sum_{t=1}^T (f_{t, \rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) - f_{t, \rho}(\mathbf{x}_{t+1}, \hat{\mathbf{y}}_t^*(\mathbf{x}_{t+1}))) \\ & + \frac{3\ell_{f,1}\ell_{g,1}}{\mu_g} (\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2) \sum_{t=1}^T (\alpha_t - L_f \alpha_t^2) + 3 \sum_{t=1}^T (2\alpha_t - L_f \alpha_t^2) A_t \\ & \leq 12M + 6V_T + 6\ell_{f,1} \left(1 + 2\frac{\ell_{g,1}}{\mu_g}\right) T (\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2) \\ & + \frac{3\ell_{f,1}\ell_{g,1}}{\mu_g} (\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2) \sum_{t=1}^T (\alpha_t - L_f \alpha_t^2) + 3 \sum_{t=1}^T (2\alpha_t - L_f \alpha_t^2) A_t, \end{aligned}$$

where the second inequality is due to Lemma [D.20.](#page-65-5)

3832 3833 From the update rule in Algorithm [2,](#page-5-2) we obtain

3847 3848 3849 where the first inequality is by (a+b) <sup>2</sup> ≤ 2a <sup>2</sup> + 2b 2 ; the second inequality follows from non-expansiveness of the projection operator; and the last inequality follows from Lemma [D.21.](#page-66-0)

*(a) Then, we have*

$$\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \leq 2\beta_t^2 \|e_t^{g_\rho}\|^2 + 2\beta_t^2 \|\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2,$$

*where* e g<sup>ρ</sup> t *is defined in* [\(142\)](#page-46-2)*.*

*(b) Suppose Assumptions [3.2,](#page-3-2)* [B2.](#page-3-12) *and* [B3.](#page-3-10) *hold. Then, we have*

$$\begin{aligned} \|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 &\leq 4\alpha_t^2 \|\mathcal{P}_{\mathcal{X},\alpha_t}(\mathbf{x}_t; \nabla f_{t,\boldsymbol{\rho}}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \\ &\quad + \frac{4\ell_{f,1}\ell_{g,1}\alpha_t^2(\rho_s^2 + \rho_r^2)}{\mu_g} + 2A_t\alpha_t^2, \end{aligned} \quad (220)$$

*where* A<sup>t</sup> *is defined in* [\(211\)](#page-66-5)*.*

*(c) Suppose Assumptions* [B1.](#page-3-18)*,* [B2.](#page-3-12) *and* [B3.](#page-3-10) *hold. Then, we have*

$$\begin{aligned} \|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 &\leq 2\delta_t^2 \|e_t^M\|^2 + 3d_t^2 \ell_{f,1}^2 \delta_t^2 \rho_{\mathbf{r}}^2 \\ &\quad + (12\ell_{f,0}^2 + 6\ell_{g,1}^2 p^2)\delta_t^2 + 6\ell_{g,1}^2 \frac{\delta_t^2}{\rho_{\mathbf{v}^2}} \hat{\theta}_t^Y, \end{aligned}$$

*where* e<sup>M</sup> <sup>t</sup> *and* <sup>ˆ</sup><sup>θ</sup> y <sup>t</sup> *are defined in* [\(158\)](#page-51-4) *and* [\(146\)](#page-49-1)*, respectively.*

*Proof.* For part (a): From Algorithm [2,](#page-5-2) we have

$$\begin{aligned}\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 &= \beta_t^2 \|\hat{\mathbf{d}}^\mathbf{y}\|^2 \\ &\leq 2\beta_t^2 \|\hat{\mathbf{d}}^\mathbf{y} - \nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2 + 2\beta_t^2 \|\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ &= 2\beta_t^2 \|e_t^g\|^2 + 2\beta_t^2 \|\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2,\end{aligned}\tag{221}$$

and from [\(220\)](#page-69-0), we get

For part (b):

$$\begin{aligned} \|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 &= \alpha_t^2 \left\| \mathcal{P}_{\mathcal{X}, \alpha_t} \left( \mathbf{x}_t; \hat{\mathbf{d}}_t^{\mathbf{x}} \right) \right\|^2 \\ &\leq 2\alpha_t^2 \left( \|\mathcal{P}_{\mathcal{X}, \alpha_t} \left( \mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) \right)\|^2 \right. \\ &\quad \left. + \left\| \mathcal{P}_{\mathcal{X}, \alpha_t} \left( \mathbf{x}_t; \hat{\mathbf{d}}_t^{\mathbf{x}} \right) - \mathcal{P}_{\mathcal{X}, \alpha_t} \left( \mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) \right) \right\|^2 \right) \\ &\leq 2\alpha_t^2 \left( \|\mathcal{P}_{\mathcal{X}, \alpha_t} \left( \mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) \right)\|^2 \right. \\ &\quad \left. + \left\| \hat{\mathbf{d}}_t^{\mathbf{x}} - \nabla f_{t, \rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) \right\|^2 \right) \\ &\leq 2\alpha_t^2 \left( \|\mathcal{P}_{\mathcal{X}, \alpha_t} \left( \mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) \right)\|^2 + A_t \right), \end{aligned} \tag{222}$$

3850 The first term in the above inequality can be bounded as

$$\begin{aligned}
3851 & \quad \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)))\|^2 \\
3852 & \leq 2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t))) - \mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \\
3853 & + 2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \\
3854 & \leq 2 \|\nabla f_{t, \rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) - \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))\|^2 \\
3855 & \leq 2 \|\nabla f_{t, \rho}(\mathbf{x}_t, \hat{\mathbf{y}}_t^*(\mathbf{x}_t)) - \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))\|^2 \\
3856 & + 2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \\
3857 & \leq 2 \ell_{f, 1} \|\hat{\mathbf{y}}_t^*(\mathbf{x}_t) - \mathbf{y}_t^*(\mathbf{x}_t)\|^2 + 2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \\
3858 & \leq 2 \ell_{f, 1} \frac{\ell_{g, 1}(\rho_s^2 + \rho_r^2)}{\mu_g} + 2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2, \\
3859 & \leq 2 \ell_{f, 1} \frac{\ell_{g, 1}(\rho_s^2 + \rho_r^2)}{\mu_g} + 2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2, \\
3860 & \leq 2 \ell_{f, 1} \frac{\ell_{g, 1}(\rho_s^2 + \rho_r^2)}{\mu_g} + 2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2, \\
3861 & \leq 2 \ell_{f, 1} \frac{\ell_{g, 1}(\rho_s^2 + \rho_r^2)}{\mu_g} + 2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2, \\
3862 & \leq 2 \ell_{f, 1} \frac{\ell_{g, 1}(\rho_s^2 + \rho_r^2)}{\mu_g} + 2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2, \\
3863 & \leq 2 \ell_{f, 1} \frac{\ell_{g, 1}(\rho_s^2 + \rho_r^2)}{\mu_g} + 2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2, \\
3864 & \leq 2 \ell_{f, 1} \frac{\ell_{g, 1}(\rho_s^2 + \rho_r^2)}{\mu_g} + 2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2, \\
3865 & \leq 2 \ell_{f, 1} \frac{\ell_{g, 1}(\rho_s^2 + \rho_r^2)}{\mu_g} + 2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2, \\
3866 & \leq 2 \ell_{f, 1} \frac{\ell_{g, 1}(\rho_s^2 + \rho_r^2)}{\mu_g} + 2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2, \\
3867 & \leq 2 \ell_{f, 1} \frac{\ell_{g, 1}(\rho_s^2 + \rho_r^2)}{\mu_g} + 2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2, \\
3868 & \leq 2 \ell_{f, 1} \frac{\ell_{g, 1}(\rho_s^2 + \rho_r^2)}{\mu_g} + 2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2, \\
3869 & \leq 2 \ell_{f, 1} \frac{\ell_{g, 1}(\rho_s^2 + \rho_r^2)}{\mu_g}$$

3862 3863 where the last inequality follows from Lemma [D.7.](#page-44-3)

3864 Based on [\(223\)](#page-70-0) and [\(222\)](#page-69-1), we get

3865 3866 3867

3868 3869 For part (c): Note that, we have

3881 From Assumption [B3.,](#page-3-10) Lemma [B.3](#page-11-3) and [\(8\)](#page-3-8), we have

3886 3887 Similarly, we get

3888 3889

3890 3891 Moreover, from Eq. [\(133\)](#page-44-0) and Assumption [B1.,](#page-3-18) we have

3898 Substituting [\(225\)](#page-70-1), [\(226\)](#page-70-2) and [\(227\)](#page-70-3), into [\(224\)](#page-70-4), we get

$$\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2 \leq 2\alpha_t^2 \left( 2 \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \boldsymbol{\rho}}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 + \frac{2\ell_{f, 1}\ell_{g, 1}(\rho_{\mathbf{S}}^2 + \rho_{\mathbf{F}}^2)}{\mu_g} + A_t \right).$$

$$\begin{aligned}
\|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 &= \delta_t^2 \|\hat{\mathbf{d}}_{\mathbf{v}}^{\mathbf{v}}\|^2 \\
&\leq 2\delta_t^2 \|\hat{\mathbf{d}}_{\mathbf{v}}^{\mathbf{v}} - \nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{z}_t) - \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t)\|^2 + 2\delta_t^2 \|\nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{z}_t) + \tilde{\nabla}_{\mathbf{y}}^2 g_t(\mathbf{z}_t)\|^2 \\
&= 2\delta_t^2 \|e_t^M\|^2 \\
&\quad + 2\delta_t^2 \|\nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) + \frac{1}{2\rho_{\mathbf{v}}} (\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t))\|^2 \\
&\leq 2\delta_t^2 \|e_t^M\|^2 + 6\delta_t^2 \|\nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\
&\quad + \frac{3\delta_t^2}{2\rho_{\mathbf{v}}} \|\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 + \frac{3\delta_t^2}{2\rho_{\mathbf{v}}} \|\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)\|^2.
\end{aligned} \tag{224}$$

$$\begin{aligned} \|\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 &\leq \ell_{g,1}^2 \|\mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \\ &\leq 2\ell_{g,1}^2 \|\rho_{\mathbf{v}} \mathbf{v}_t\|^2 + 2\ell_{g,1}^2 \|\mathbf{y}_t - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2 \\ &\leq 2\ell_{g,1}^2 \rho_{\mathbf{v}}^2 p^2 + 2\ell_{g,1}^2 \|\mathbf{y}_t - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2. \end{aligned} \quad (225)$$

$$\|\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t - \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \leq 2\ell_{g,1}^2 \rho_{\mathbf{v}}^2 p^2 + 2\ell_{g,1}^2 \|\mathbf{y}_t - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2. \quad (226)$$

$$\begin{aligned} \|\nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2 &\leq 2\|\nabla_{\mathbf{y}} f_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 + 2\|\nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ &\leq \frac{d_2^2 \ell_{f,1}^2 \rho_{\mathbf{r}}^2}{2} + 2\|\nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ &\leq \frac{d_2^2 \ell_{f,1}^2 \rho_{\mathbf{r}}^2}{2} + 2\ell_{f,0}^2. \end{aligned} \quad (227)$$

$$\begin{aligned} \|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 &\leq 2\delta_t^2 \|e_t^M\|^2 + 3d_{f,1}^2 \delta_t^2 \rho_{\mathbf{r}}^2 \\ &\quad + (12\ell_{f,0}^2 + 6\ell_{g,1}^2 p^2)\delta_t^2 + \frac{6\ell_{g,1}^2}{\rho_{\mathbf{v}^2}} \delta_t^2 \|\mathbf{y}_t - \hat{\mathbf{y}}_t^*(\mathbf{x}_t)\|^2. \end{aligned}$$

3908 3909 *Proof.* Since (1 − γt+1) <sup>2</sup> ≤ 1 − γt+1 and γt+1 = cγαt, from [\(143\)](#page-46-3), we have

3914

3916 Since (1 − ηt+1) <sup>2</sup> ≤ 1 − ηt+1 and ηt+1 = cηαt, from [\(197\)](#page-62-0), we have

3919

3924

3926

3929

3934

3936

#### 3944 3945 Combining the outcomes .

3954

3956

#### D.6. Proof of Theorem [4.2](#page-5-0)

$$\begin{aligned} \mathbb{E}\|e_{t+1}^{g\rho}\|^2 - \mathbb{E}\|e_t^{g\rho}\|^2 &\leq -c_\gamma \alpha_t \mathbb{E}\|e_t^{g\rho}\|^2 \\ &\quad + 12(1 - \gamma_{t+1})^2 \mathbb{E}\|\nabla_{\mathbf{y}} g_{t-1}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ &\quad + 9d_2^2 \ell_{g,1}^2 (1 - \gamma_{t+1})^2 \rho_{\mathbf{r}}^2 + 24d_2 \ell_{g,1}^2 (1 - \gamma_{t+1})^2 \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 \\ &\quad + 24d_2 \ell_{g,1}^2 (1 - \gamma_{t+1})^2 \mathbb{E}\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 + 2\frac{\hat{\sigma}_{gy}^2}{b} \gamma_{t+1}^2. \end{aligned} \quad (22.8)$$

$$\begin{aligned} \mathbb{E}\|e_{t+1}^L\|^2 - \mathbb{E}\|e_t^L\|^2 &\leq -c_\eta \alpha_t \mathbb{E}\|e_t^L\|^2 + 36\mathbb{E}\|\nabla_{\mathbf{x}} f_{t+1}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{x}} f_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ &\quad + \left( 18d_1^2 \ell_{f,1}^2 + 6(3\ell_{f,1}^2 + \frac{3\ell_{g,1}^2}{4\rho_{\mathbf{V}}^2})d_1^2 \right) \rho_{\mathbf{s}}^2 + 18d_1^2 \ell_{g,1}^2 \frac{\rho_{\mathbf{s}}^2}{\rho_{\mathbf{V}}^2} \\ &\quad + \frac{36}{\rho_{\mathbf{V}}^2} \mathbb{E}\|\nabla_{\mathbf{x}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{x}} g_t(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \\ &\quad + 6(12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{V}}^2})d_1 \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 6(12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{V}}^2})d_1 \mathbb{E}\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\ &\quad + 27\ell_{g,1}^2 d_1 \mathbb{E}\|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + 3(\frac{\hat{\sigma}_{g_{\mathbf{x}}}^2}{b\rho_{\mathbf{V}}^2} + \frac{\hat{\sigma}_{f_{\mathbf{x}}}^2}{b})\eta_{t+1}^2. \end{aligned} \quad (229)$$

Since (1 − λt+1) <sup>2</sup> ≤ 1 − λt+1 and λt+1 = cλαt, from [\(160\)](#page-52-0), we have

$$\begin{aligned} \mathbb{E}\|e_{t+1}^M\|^2 - \mathbb{E}\|e_t^M\|^2 &\leq -c_\lambda \alpha_t \mathbb{E}\|e_t^M\|^2 + 36\mathbb{E}\|\nabla_{\mathbf{y}} f_{t+1}(\mathbf{x}_t, \mathbf{y}_t) - \nabla_{\mathbf{y}} f_t(\mathbf{x}_t, \mathbf{y}_t)\|^2 \\ &\quad + \left( 18d_2^2 \ell_{f,1}^2 + 6(3\ell_{f,1}^2 + \frac{3\ell_{g,1}^2}{4\rho_{\mathbf{v}}^2})d_2^2 \right) \rho_{\mathbf{r}}^2 + 18d_2^2 \ell_{g,1}^2 \frac{\rho_{\mathbf{r}}^2}{\rho_{\mathbf{v}}^2} \\ &\quad + \frac{36}{\rho_{\mathbf{v}}^2} \mathbb{E}\|\nabla_{\mathbf{y}} g_{t+1}(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t) - \nabla_{\mathbf{y}} g_t(\mathbf{x}_t, \mathbf{y}_t + \rho_{\mathbf{v}} \mathbf{v}_t)\|^2 \\ &\quad + 6(12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2})d_2 \mathbb{E}\|\mathbf{x}_{t+1} - \mathbf{x}_t\|^2 + 6(12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2})d_2 \mathbb{E}\|\mathbf{y}_{t+1} - \mathbf{y}_t\|^2 \\ &\quad + 27d_2 \ell_{g,1}^2 \mathbb{E}\|\mathbf{v}_{t+1} - \mathbf{v}_t\|^2 + 3(\frac{\hat{\sigma}_{gy}^2}{b\rho_{\mathbf{v}}^2} + \frac{\hat{\sigma}_{fy}^2}{b})\lambda_{t+1}^2. \end{aligned} \quad (230)$$

Let

$$\begin{aligned} \Lambda &:= \Gamma \sum_{t=1}^T \left( \mathbb{E}[\hat{\theta}_{t+1}^{\mathbf{y}}] - \mathbb{E}[\hat{\theta}_t^{\mathbf{y}}] \right) + \Upsilon \sum_{t=1}^T \left( \mathbb{E}[\hat{\theta}_{t+1}^{\mathbf{y}}] - \mathbb{E}[\hat{\theta}_t^{\mathbf{y}}] \right) + \frac{1}{\Phi} \sum_{t=1}^T (\mathbb{E}\|e_{t+1}^{g_\rho}\|^2 - \mathbb{E}\|e_t^{g_\rho}\|^2) \\ &+ \frac{1}{\Psi} \sum_{t=1}^T (\mathbb{E}\|e_{t+1}^M\|^2 - \mathbb{E}\|e_t^M\|^2) + \frac{1}{\Omega} \sum_{t=1}^T (\mathbb{E}\|e_{t+1}^L\|^2 - \mathbb{E}\|e_t^L\|^2). \end{aligned}$$

3974

3976

3979

3981

3983 3984

3986 3987

3990

3994

3996

Here, we have

$$\begin{aligned} \Gamma &= \frac{11M_f^2}{L_{\mu_g}c_\beta}, \quad \Upsilon = \frac{52M_f^2}{L_{\mu_g}c_\delta}, \quad \Phi = \max \left\{ 240 \frac{d_2\ell_{g,1}^2}{L_f}, \frac{12d_2\ell_{g,1}^2 L_{\mu_g}^2 c_\beta^2}{L_f M_f^2} \right\}, \\ \Psi &= \max \left\{ 720 \frac{d_2\ell_{f,1}^2}{L_f}, 27 \frac{L_{\mu_g}}{\Upsilon L_f} \ell_{g,1}^2 d_2 c_\delta, \frac{144d_2\ell_{f,1}^2(\mu_g + \ell_{g,1})c_\beta}{L_f \Gamma}, \frac{36\ell_{f,1}^2 d_2 L_{\mu_g}^2 c_\beta^2}{L_f M_f^2} \right\}, \\ \Omega &= \max \left\{ 720 \frac{d_1\ell_{f,1}^2}{L_f}, 27 \frac{L_{\mu_g}}{\Upsilon L_f} \ell_{g,1}^2 d_1 c_\delta, \frac{144d_1\ell_{f,1}^2(\mu_g + \ell_{g,1})c_\beta}{L_f \Gamma}, \frac{36\ell_{f,1}^2 d_1 L_{\mu_g}^2 c_\beta^2}{L_f M_f^2} \right\}, \end{aligned} \quad (231)$$

with

$$\begin{aligned} c_\beta &\geq \sqrt{1760 \frac{L_{\mathbf{y}}^2 M_f^2}{L_{\mu_g}^2}}, \quad c_\delta \geq \sqrt{33280 \frac{\nu^2 M_f^2}{L_{\mu_g}^2 \mu_g^2}} (1 + 2L_{\mathbf{y}}^2), \\ c &\geq \left( \max \left\{ 4L_f, c_\beta(\mu_g + \ell_{g,1}), \frac{48L_{\mu_g}^2 d_2 \ell_{g,1}^2 c_\beta^2}{M_f^2 \Phi} \right\} \right)^3 + 1, \\ c_{\mathbf{v}} &= \max \left\{ 1080\ell_{g,1}^2, \frac{324}{M_f^2} \ell_{g,1}^4 c_\delta^2, \frac{54L_{\mu_g}^2}{M_f^2} \ell_{g,1}^2 c_\beta^2, \frac{216}{\Gamma} \ell_{g,1}^2 c_\beta(\mu_g + \ell_{g,1}) \right\} \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right), \\ c_\gamma &= \frac{26M_f^2 \Phi}{L_{\mu_g}^2}, \quad c_\eta = 26\Omega, \quad c_\lambda = \frac{10\Upsilon}{L_{\mu_g}} c_\delta \Psi. \end{aligned} \quad (232)$$

4015 4016 By adding [\(229\)](#page-71-0), [\(228\)](#page-71-1), [\(230\)](#page-71-2), [\(147\)](#page-49-5), and [\(184\)](#page-59-3), along with [\(215\)](#page-67-5) and considering the fact that α<sup>t</sup> decreases with respect to t, and by applying Lemma [D.23,](#page-68-2) we obtain:

$$\begin{aligned}
 & ^{4031} (L_{\mu_g} \beta_T - L_{\mu_g} \mu_g^T \beta_T) \\
 & ^{4032} + \sum_{t=1}^T Q(\alpha_t, \beta_t, \delta_t) \mathbb{E} \|e_t^{g_\rho}\|^2 + \sum_{t=1}^T S(\alpha_t, \beta_t, \delta_t) \mathbb{E} [\|\nabla_{\mathbf{y}} g_t, \rho(\mathbf{x}_t, \mathbf{y}_t)\|^2] \\
 & ^{4033} \\
 & ^{4034}
 \end{aligned}
 \tag{233e}$$

4046 4047 Here

$$\begin{aligned} & \sum_{t=1}^T A(\alpha_t, \beta_t, \delta_t) \mathbb{E} \left[ \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \right] + \Lambda \\ & \leq 12M + 6V_T + \sum_{t=1}^T B(\alpha_t, \beta_t, \delta_t) \hat{\theta}_t^y + \sum_{t=1}^T C(\alpha_t, \beta_t, \delta_t) \hat{\theta}_t^y \end{aligned} \quad (233a)$$

$$+ \frac{4\ell_{f,1}\ell_{g,1}}{\mu_g} \sum_{t=1}^T E(\beta_t, \delta_t, \rho_{\mathbf{v}}) \alpha_t^2(\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2) + \sum_{t=1}^T L(\alpha_t, \beta_t, \delta_t) \mathbb{E}\|e_t^L\|^2 \quad (233b)$$

$$+ \frac{8\ell_{g,2}^2 p^4 \Upsilon}{L_{\mu_g}} \sum_{t=1}^T \delta_t \rho_{\mathbf{v}}^2 + 4\ell_{g,2}^2 p^4 \sum_{t=1}^T (6\alpha_t - 3L_f \alpha_t^2 + 2\alpha_t^2 E(\beta_t, \delta_t, \rho_{\mathbf{v}})) \rho_{\mathbf{v}}^2 \quad (233c)$$

$$+ \left( \frac{12}{L_{\mu_g}} \frac{\Gamma}{\beta_T} + \frac{48\nu^2}{L_{\mu_g}\mu_g^2} \frac{\Upsilon}{\delta_T} \right) H_{2,T} + \sum_{t=1}^T M(\alpha_t, \beta_t, \delta_t) \mathbb{E} \|e_t^M\|^2 \quad (233d)$$

$$+ \sum_{t=1}^T Z(\alpha_t) (3d_2^2 \ell_{f,1}^2 \delta_t^2 \rho_{\mathbf{r}}^2 + (12\ell_{f,0}^2 + 6\ell_{g,1}^2 p^2)\delta_t^2) \quad (233f)$$

$$+ \frac{36}{\Psi} D_{\mathbf{y},T} + \frac{36}{\Omega} D_{\mathbf{x},T} + \frac{12}{\Phi} G_{\mathbf{y},T} + \frac{18}{\Psi \rho_{\mathbf{y}}^2} G_{\mathbf{v},T} + \frac{18}{\Omega \rho_{\mathbf{v}}^2} G_{\mathbf{x},T} \quad (233g)$$

$$+ 2 \sum_{t=1}^T \frac{\gamma_{t+1}^2}{\Phi} \frac{\hat{\sigma}_{yx}^2}{\bar{b}} + 3 \left( \frac{\hat{\sigma}_{yx}^2}{\bar{b} \rho_{xy}^2} + \frac{\hat{\sigma}_{xy}^2}{b} \right) \sum_{t=1}^{T+1} \frac{\lambda_{t+1}^2}{\Psi} + 3 \left( \frac{\hat{\sigma}_{yx}^2}{\bar{b} \rho_{xy}^2} + \frac{\hat{\sigma}_{fx}^2}{b} \right) \sum_{t=1}^T \frac{\eta_{t+1}^2}{\Omega} \quad (233h)$$

$$+ R(\rho_{\mathbf{v}})T\rho_{\mathbf{r}}^2 + \dot{R}(\rho_{\mathbf{v}})T\rho_{\mathbf{s}}^2 + 18d_1^2\ell_{g,1}^2\frac{T\rho_{\mathbf{s}}^2}{\Omega\rho_{\mathbf{v}}^2} + 18d_2^2\ell_{g,1}^2\frac{T\rho_{\mathbf{r}}^2}{\Psi\rho_{\mathbf{v}}^2} + \sum_{t=1}^T D(\alpha_t, \beta_t, \delta_t) (\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2). \quad (233i)$$

$$\begin{aligned}
E(\beta_t, \delta_t, \rho_{\mathbf{v}}) &:= \frac{4L_{\mathbf{y}}^2}{L_{\mu_g}} \frac{\Gamma}{\beta_t} + \frac{16\nu^2}{L_{\mu_g}\mu_g^2} (2L_{\mathbf{y}}^2 + 1) \frac{\Upsilon}{\delta_t} \\
&\quad + 24d_2 \frac{\ell_{g,1}^2}{\Phi} + 6(12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2}) \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right), \\
A(\alpha_t, \beta_t, \delta_t) &:= \alpha_t - L_f \alpha_t^2 - 4E(\beta_t, \delta_t, \rho_{\mathbf{v}}) \alpha_t^2, \\
B(\alpha_t, \beta_t, \delta_t) &:= -\frac{L_{\mu_g}}{4} \Upsilon \delta_t + 2M_f^2 (6\alpha_t - 3L_f \alpha_t^2 + 2\alpha_t^2 E(\beta_t, \delta_t, \rho_{\mathbf{v}})), \\
Z(\alpha_t) &:= 27\ell_{g,1}^2 \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right), \\
C(\alpha_t, \beta_t, \delta_t) &:= -\frac{L_{\mu_g}}{2} \Gamma \beta_t + Z(\alpha_t) 6\ell_{g,1}^2 \frac{\delta_t^2}{\rho_{\mathbf{v}}^2} + 2M_f^2 (6\alpha_t - 3L_f \alpha_t^2 + 2\alpha_t^2 E(\beta_t, \delta_t, \rho_{\mathbf{v}})).
\end{aligned} \tag{234}$$

4070 Moreover,

$$\begin{aligned}
4071 & \\
4072 & M(\alpha_t, \beta_t, \delta_t) := -\frac{\lambda_{t+1}}{\Psi} + Z(\alpha_t)2\delta_t^2 + \frac{8\Upsilon}{L_{\mu_g}}\delta_t, \\
4073 & \\
4074 & D(\alpha_t, \beta_t, \delta_t) := 6\ell_{f,1}(1 + 2\frac{\ell_{g,1}}{\mu_g}) + \frac{3\ell_{f,1}\ell_{g,1}}{\mu_g}(\alpha_t - L_f\alpha_t^2) \\
4075 & \\
4076 & + \frac{24\ell_{g,1}}{L_{\mu_g}\mu_g}\frac{\Gamma}{\beta_t} + \frac{96\ell_{g,1}\nu^2}{L_{\mu_g}\mu_g^3}\frac{\Upsilon}{\delta_t}, \\
4077 & \\
4078 & F(\alpha_t) := 24d_2\frac{\ell_{g,1}^2}{\Phi} + (72\ell_{f,1}^2 + \frac{27\ell_{g,1}^2}{\rho_{\mathbf{v}}^2})(\frac{d_2}{\Psi} + \frac{d_1}{\Omega}), \\
4079 & \\
4080 & S(\alpha_t, \beta_t, \delta_t) := -\frac{2\beta_t\Gamma}{\mu_g + \ell_{g,1}} + \beta_t^2\Gamma + 2F(\alpha_t)\beta_t^2, \\
4081 & \\
4082 & Q(\alpha_t, \beta_t, \delta_t) := \frac{2}{L_{\mu_g}}\Gamma\beta_t - \frac{\gamma_{t+1}}{\Phi} + 2F(\alpha_t)\beta_t^2, \\
4083 & \\
4084 & \\
4085 & R(\rho_{\mathbf{v}}) := 9d_2^2\frac{\ell_{g,1}^2}{\Phi} + 18d_2^2\frac{\ell_{f,1}^2}{\Psi} + 6(3\ell_{f,1}^2 + \frac{3\ell_{g,1}^2}{4\rho_{\mathbf{v}}^2})\frac{d_2^2}{\Psi}, \\
4086 & \\
4087 & \\
4088 & \dot{R}(\rho_{\mathbf{v}}) := 18d_1^2\frac{\ell_{f,1}^2}{\Omega} + 6(3\ell_{f,1}^2 + \frac{3\ell_{g,1}^2}{4\rho_{\mathbf{v}}^2})\frac{d_1^2}{\Omega}, \\
4089 & \\
4090 & L(\alpha_t, \beta_t, \delta_t) := -\frac{\eta_{t+1}}{\Omega} + 4(6\alpha_t - 3L_f\alpha_t^2 + 2\alpha_t^2E(\beta_t, \delta_t, \rho_{\mathbf{v}})). \\
4091 & \\
4092 & 
\end{aligned}
\tag{235}$$

4092

4093 4094 Note that, we have

4100 4101 which together with β<sup>t</sup> = cβαt, δ<sup>t</sup> = cδαt, we have

4114 4115 4116 4117 4118 4119 where the first inequality is by Γ = <sup>11</sup>M<sup>2</sup> Lµg c<sup>β</sup> , Υ = <sup>52</sup>M<sup>2</sup> f Lµg cδ in [\(231\)](#page-72-1), ρ 2 <sup>v</sup> = cvα<sup>t</sup> and α<sup>t</sup> ≤ 1/4L<sup>f</sup> ; the second inequality follows from c<sup>β</sup> ≥ r <sup>1760</sup> <sup>L</sup><sup>2</sup> yM<sup>2</sup> f L<sup>2</sup> µg , c<sup>δ</sup> ≥ r <sup>33280</sup> <sup>ν</sup>2M<sup>2</sup> f L<sup>2</sup> µg µ<sup>2</sup> g (1 + 2L<sup>2</sup> y ), in [\(232\)](#page-72-0); and Φ = 240 <sup>d</sup>2<sup>ℓ</sup> g,1 L<sup>f</sup> , Ψ = 720 <sup>d</sup>2<sup>ℓ</sup> L<sup>f</sup> , Ω = 720 <sup>d</sup>1<sup>ℓ</sup> f,1 L<sup>f</sup> and c<sup>v</sup> ≥ 1080ℓ 2 g,1 ( d<sup>2</sup> <sup>Ψ</sup> + d<sup>1</sup> Ω ) in [\(231\)](#page-72-1).

We then provide bounds for the terms in [\(233a\)](#page-73-0)-[\(233i\)](#page-73-1).

$$\begin{aligned} E(\beta_t, \delta_t, \rho_{\mathbf{v}}) &:= \frac{4L_{\mathbf{y}}^2}{L_{\mu_g}} \frac{\Gamma}{\beta_t} + \frac{16\nu^2}{L_{\mu_g}\mu_g^2} (2L_{\mathbf{y}}^2 + 1) \frac{\Upsilon}{\delta_t} \\ &\quad + 24d_2 \frac{\ell_{g,1}^2}{\Phi} + 6(12\ell_{f,1}^2 + \frac{9\ell_{g,1}^2}{2\rho_{\mathbf{v}}^2}) \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right), \end{aligned}$$

$$\begin{aligned} \alpha_t^2 E(\beta_t, \delta_t, \rho_{\mathbf{v}}) &= \frac{4L_{\mathbf{y}}^2}{L_{\mu_g}} \frac{\Gamma \alpha_t^2}{\beta_t} + \frac{16\nu^2}{L_{\mu_g} \mu_g^2} (2L_{\mathbf{y}}^2 + 1) \frac{\Upsilon \alpha_t^2}{\delta_t} \\ &\quad + 24d_2 \frac{\ell^2_{g,1}}{\Phi} \alpha_t^2 + (72\ell_{f,1}^2 \alpha_t^2 + \frac{27\ell_{g,1}^2}{\rho_{\mathbf{v}}^2} \alpha_t^2) \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right) \\ &\leq \frac{44L_{\mathbf{y}}^2}{L_{\mu_g}^2} M_f^2 \frac{\alpha_t}{c_{\beta}^2} + \frac{832\nu^2}{L_{\mu_g}^2 \mu_g^2} (1 + 2L_{\mathbf{y}}^2) M_f^2 \frac{\alpha_t}{c_{\delta}^2} \\ &\quad + 6 \frac{d_2 \ell_{g,1}^2}{L_f \Phi} \alpha_t + \left( \frac{18\ell_{f,1}^2}{L_f} \alpha_t + \frac{27\ell_{g,1}^2}{c_{\mathbf{v}}} \alpha_t \right) \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right) \\ &\leq \frac{\alpha_t}{8}, \end{aligned} \tag{236}$$

4132 Bounding [\(233a\)](#page-73-0) .

4133 4134 From δ<sup>t</sup> = cδαt, we have

4144 4145 4146 where the first inequality follows from [\(236\)](#page-74-0); the last inequality is by Υ = <sup>52</sup>M<sup>2</sup> f Lµg cδ in [\(231\)](#page-72-1).

4147 4148 4149

4150 4151 Thus, from β<sup>t</sup> = cβαt, δ<sup>t</sup> = cδα<sup>t</sup> and ρ 2 <sup>v</sup> = cvαt, we have

4163 4164 4165 where the first inequality follows from [\(236\)](#page-74-0); the second equality follows from Γ = <sup>11</sup>M<sup>2</sup> f Lµg c<sup>β</sup> in [\(231\)](#page-72-1); the last inequality is by c<sup>v</sup> ≥ 324 M<sup>2</sup> ℓ 4 g,1 ( d<sup>2</sup> <sup>Ψ</sup> + d<sup>1</sup> )c 2 δ .

4166 f Thus, from [\(238\)](#page-75-0) and [\(239\)](#page-75-1), we get

4167 4168 4169

4170 4171 Bounding [\(233b\)](#page-73-3) .

Moreover, we have

$$\begin{aligned} A(\alpha_t, \beta_t, \delta_t) &= \alpha_t - L_f \alpha_t^2 - 4E(\beta_t, \delta_t, \rho_{\mathbf{v}}) \alpha_t^2 \\ &\geq \alpha_t - L_f \alpha_t^2 - \frac{\alpha_t}{2} \\ &\geq \frac{\alpha_t}{4}, \end{aligned} \tag{237}$$

where the last inequality is by α<sup>t</sup> ≤ 1/4L<sup>f</sup> in [\(232\)](#page-72-0).

$$\begin{aligned}
B(\alpha_t, \beta_t, \delta_t) &= -\frac{L_{\mu_g}}{4} \Upsilon \delta_t + 2M_f^2 (6\alpha_t - 3L_f \alpha_t^2 + 2\alpha_t^2 E(\beta_t, \delta_t, \rho_{\mathbf{v}})) \\
&\leq -\frac{L_{\mu_g}}{4} \Upsilon c_\delta \alpha_t + 12M_f^2 \alpha_t - 6M_f^2 L_f \alpha_t^2 + \frac{M_f^2}{2} \alpha_t \\
&\leq \left( -\frac{L_{\mu_g}}{4} \Upsilon c_\delta + \frac{25}{2} M_f^2 \right) \alpha_t \\
&\leq -\frac{1}{2} M_f^2 \alpha_t, \tag{238}
\end{aligned}$$

From [\(234\)](#page-73-2), we obtain

$$Z(\alpha_t) = 27\ell_{g,1}^2 \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right).$$

$$\begin{aligned} C(\alpha_t, \beta_t, \delta_t) &= -\frac{L_{\mu_g}}{2} \Gamma \beta_t + 162 \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right) \ell_{g,1}^4 \frac{\delta_t^2}{\rho_{\mathbf{v}}^2} \\ &\quad + 2M_f^2 (6\alpha_t - 3L_f \alpha_t^2 + 2\alpha_t^2 E(\beta_t, \delta_t, \rho_{\mathbf{v}})) \\ &\leq -\frac{L_{\mu_g}}{2} \Gamma c_\beta \alpha_t + 162 \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right) \ell_{g,1}^4 \frac{c_\delta^2}{c_{\mathbf{v}}} \alpha_t + \frac{9}{2} M_f^2 \alpha_t \\ &= -\frac{11}{2} M_f^2 \alpha_t + 162 \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right) \ell_{g,1}^4 \frac{c_\delta^2}{c_{\mathbf{v}}} \alpha_t + \frac{9}{2} M_f^2 \alpha_t \\ &\leq -\frac{1}{2} M_f^2 \alpha_t, \end{aligned} \tag{239}$$

$$(233a) \leq \mathcal{O}(V_T). \quad (240)$$

4197 4198 where the last inequality is by c<sup>η</sup> ≥ 26Ω and [\(236\)](#page-74-0).

4208 4209 Bounding [\(233c\)](#page-73-4) .

4210 From δ<sup>t</sup> = cδα<sup>t</sup> and [\(236\)](#page-74-0), we have

4217 4218 Thus, from ρ <sup>v</sup> = cvαt, we have

4223 Bounding [\(233d\)](#page-73-5) .

4224 From [\(234\)](#page-73-2), we obtain

From [\(236\)](#page-74-0), we also obtain

$$\begin{aligned} & \frac{4\ell_{f,1}\ell_{g,1}}{\mu_g} \sum_{t=1}^T E(\beta_t, \delta_t, \rho_{\mathbf{v}}) \alpha_t^2 (\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2) \\ & \leq \frac{4\ell_{f,1}\ell_{g,1}}{\mu_g} \sum_{t=1}^T \frac{\alpha_t}{8} (\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2) \\ & = \mathcal{O} \left( \sum_{t=1}^T \alpha_t (\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2) \right). \end{aligned} \tag{241}$$

From [\(235\)](#page-74-1) and ηt+1 = cηαt, we have

$$\begin{aligned} L(\alpha_t, \beta_t, \delta_t) &= -\frac{\eta_{t+1}}{\Omega} + 4 (6\alpha_t - 3L_f\alpha_t^2 + 2\alpha_t^2 E(\beta_t, \delta_t, \rho_v)) \\ &\leq -\frac{c_\eta}{\Omega} \alpha_t + 25\alpha_t \\ &\leq -\alpha_t, \end{aligned}$$

Thus, we get

$$\sum_{t=1}^T L(\alpha_t, \beta_t, \delta_t) \mathbb{E}\|e_t^L\|^2 \leq 0. \quad (242)$$

From [\(242\)](#page-76-0) and [\(241\)](#page-76-1), we have

$$(233b) \leq \mathcal{O} \left( \sum_{t=1}^T \alpha_t (\rho_s^2 + \rho_r^2) \right). \quad (243)$$

$$\begin{aligned} & \frac{8\ell_{g,2}^2 p^4 \Upsilon}{L_{\mu_g}} \sum_{t=1}^T \delta_t \rho_{\mathbf{v}}^2 + 4\ell_{g,2}^2 p^4 \sum_{t=1}^T (6\alpha_t - 3L_f \alpha_t^2 + 2\alpha_t^2 E(\beta_t, \delta_t, \rho_{\mathbf{v}})) \rho_{\mathbf{v}}^2 \\ & \leq \frac{8\ell_{g,2}^2 p^4 \Upsilon}{L_{\mu_g}} \sum_{t=1}^T c_8 \alpha_t \rho_{\mathbf{v}}^2 + 4\ell_{g,2}^2 p^4 \sum_{t=1}^T \frac{25}{4} \alpha_t \rho_{\mathbf{v}}^2. \end{aligned}$$

$$(233c) \leq \mathcal{O}\left(\sum_{t=1}^T \alpha_t^2\right). \quad (244)$$

$$Z(\alpha_t) = 27\ell_{g,1}^2 \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right). \quad (245)$$

4246 4247 4248 4249 where the first inequality is by c<sup>λ</sup> ≥ 10Υ Lµg <sup>c</sup>δ<sup>Ψ</sup> and <sup>α</sup><sup>t</sup> <sup>≤</sup> <sup>1</sup>/4L<sup>f</sup> ; the last inequality follows from <sup>Ψ</sup> <sup>≥</sup> <sup>27</sup> <sup>L</sup>µg ΥL<sup>f</sup> ℓ g,1d2c<sup>δ</sup> and <sup>Ω</sup> ≥ <sup>27</sup> <sup>L</sup>µg ΥL<sup>f</sup> ℓ 2 g,1d1cδ.

4250 Since β<sup>t</sup> = cβα<sup>t</sup> and δ<sup>t</sup> = cδαt, we get

4258 4259 From [\(235\)](#page-74-1), we have

4260 4261 4262

4263 From [\(235\)](#page-74-1), γt+1 = cγαt, β<sup>t</sup> = cβαt, we have

4278 4279 4280 4281 4282 4283 where the first equality is by Γ = <sup>11</sup>M<sup>2</sup> f Lµg c<sup>β</sup> and ρ 2 <sup>v</sup> = cvαt; the first inequality follows from c<sup>γ</sup> ≥ 26M<sup>2</sup> <sup>f</sup> Φ L<sup>2</sup> µg ; the second inequality is by α<sup>t</sup> ≤ 1/4L<sup>f</sup> ; the last inequality follows from c<sup>v</sup> ≥ 54L µg M<sup>2</sup> f ℓ 2 g,1 ( d<sup>2</sup> <sup>Ψ</sup> + d<sup>1</sup> Ω )c 2 β , Φ ≥ 12d2ℓ g,1L µg c β LfM<sup>2</sup> f ,and Ψ ≥ 36ℓ f,1d2L µg c β LfM<sup>2</sup> f , and Ω ≥ 36ℓ 2 f,1d1L 2 µg c 2 LfM<sup>2</sup> f .

From [\(235\)](#page-74-1), λt+1 = cλα<sup>t</sup> and δ<sup>t</sup> = cδαt, we have

$$\begin{aligned} M(\alpha_t, \beta_t, \delta_t) &= -\frac{\lambda_{t+1}}{\Psi} + Z(\alpha_t)2\delta_t^2 + \frac{8\Upsilon}{L_{\mu_g}}\delta_t \\ &= -\frac{c_\lambda\alpha_t}{\Psi} + 27\ell_{g,1}^2 \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right) 2c_\delta^2\alpha_t^2 + \frac{8\Upsilon}{L_{\mu_g}}c_\delta\alpha_t \\ &\leq -\frac{2\Upsilon}{L_{\mu_g}}c_\delta\alpha_t + \frac{27}{4L_f}\ell_{g,1}^2 \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right) 2c_\delta^2\alpha_t \\ &\leq -\frac{\Upsilon}{L_{\mu_g}}c_\delta\alpha_t, \end{aligned}$$

$$\begin{aligned} (233d) &= \left( \frac{12}{L_{\mu_g}} \frac{\Gamma}{\beta_T} + \frac{48\nu^2}{L_{\mu_g}\mu_g^2} \frac{\Upsilon}{\delta_T} \right) H_{2,T} + \sum_{t=1}^T M(\alpha_t, \beta_t, \delta_t) \mathbb{E}\|e_t^M\|^2 \\ &\leq \mathcal{O}\left(\frac{H_{2,T}}{\alpha_T}\right). \end{aligned} \quad (246)$$

#### Bounding [\(233e\)](#page-73-6) .

$$F(\alpha_t) = 24d_2 \frac{\ell_{g,1}^2}{\Phi} + (72\ell_{f,1}^2 + \frac{27\ell_{g,1}^2}{\rho_{\mathbf{v}}^2})(\frac{d_2}{\Psi} + \frac{d_1}{\Omega}) \quad (247)$$

$$\begin{aligned} Q(\alpha_t, \beta_t, \delta_t) &= -\frac{\gamma_{t+1}}{\Phi} + \frac{2}{L_{\mu_g}} \Gamma \beta_t + 2F(\alpha_t) \beta_t^2 \\ &= -\frac{c_\gamma \alpha_t}{\Phi} + \frac{22M_f^2}{L_{\mu_g}^2} \alpha_t + \left( 24d_2 \frac{\ell_{g,1}^2}{\Phi} + (72\ell_{f,1}^2 + \frac{27\ell_{g,1}^2}{c_\nu \alpha_t}) \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right) \right) 2c_\beta^2 \alpha_t^2 \\ &\leq -\frac{4M_f^2}{L_{\mu_g}^2} \alpha_t + \left( 24d_2 \frac{\ell_{g,1}^2}{\Phi} \alpha_t^2 + (72\ell_{f,1}^2 \alpha_t^2 + \frac{27\ell_{g,1}^2 \alpha_t}{c_\nu}) \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right) \right) 2c_\beta^2 \\ &\leq -\frac{4M_f^2}{L_{\mu_g}^2} \alpha_t + \left( \frac{6d_2}{L_f} \frac{\ell_{g,1}^2}{\Phi} \alpha_t + \left( \frac{18}{L_f} \ell_{f,1}^2 \alpha_t + \frac{27\ell_{g,1}^2 \alpha_t}{c_\nu} \right) \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right) \right) 2c_\beta^2 \\ &\leq -\frac{M_f^2}{L_{\mu_g}^2} \alpha_t, \end{aligned} \tag{248}$$

4290 From [\(235\)](#page-74-1), β<sup>t</sup> = cβαt, ρ 2 <sup>v</sup> = cvα<sup>t</sup> and [\(247\)](#page-77-0), we have

4305 4306 4307 4308 where the first inequality follows from α<sup>t</sup> ≤ 1/cβ(µ<sup>g</sup> + ℓg,1); the second inequality is by α ≤ 1/4L<sup>f</sup> ; the last inequality is by c<sup>v</sup> ≥ 216 Γ ℓ g,1 ( d<sup>2</sup> <sup>Ψ</sup> + d<sup>1</sup> Ω )cβ(µg+ℓg,1) and Φ ≥ 24d2ℓ 2 g,1 (µg+ℓg,1) L<sup>f</sup> cβΓ , and Ψ ≥ 144d2ℓ 2 (µg+ℓg,1)c<sup>β</sup> L<sup>f</sup> Γ , and Ω ≥ 144d1ℓ 2 (µg+ℓg,1)c<sup>β</sup> L<sup>f</sup> Γ . Thus, we get

$$4309 \quad (233e) = \sum_{t=1}^T Q(\alpha_t, \beta_t, \delta_t) \mathbb{E}\|e_t^{g\rho}\|^2 + \sum_{t=1}^T S(\alpha_t, \beta_t, \delta_t) \mathbb{E} [\|\nabla_{\mathbf{y}} g_{t,\rho}(\mathbf{x}_t, \mathbf{y}_t)\|^2] \leq 0. \quad (250)$$

4312 4313 Bounding [\(233f\)](#page-73-7) .

4314 From [\(234\)](#page-73-2), we obtain

4315 4316 4317

4318 4319 Thus, from δ<sup>t</sup> = cδαt, we have

4329 Bounding [\(233g\)](#page-73-8) . From ρ 2 <sup>v</sup> = cvαt, we have

$$\begin{aligned}
S(\alpha_t, \beta_t, \delta_t) &= -\frac{2\beta_t\Gamma}{\mu_g + \ell_{g,1}} + \beta_t^2\Gamma + 2F(\alpha_t)\beta_t^2 \\
&= -\frac{2c_\beta\alpha_t\Gamma}{\mu_g + \ell_{g,1}} + c_\beta^2\alpha_t^2\Gamma + \left( 24d_2 \frac{\ell_{g,1}^2}{\Phi} + (72\ell_{f,1}^2 + \frac{27\ell_{g,1}^2}{c_{\mathbf{v}}\alpha_t}) \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right) \right) 2c_\beta^2\alpha_t^2 \\
&\leq -\frac{c_\beta\alpha_t\Gamma}{\mu_g + \ell_{g,1}} + \left( 24d_2 \frac{\ell_{g,1}^2}{\Phi} \alpha_t^2 + (72\ell_{f,1}^2 \alpha_t^2 + \frac{27\ell_{g,1}^2\alpha_t}{c_{\mathbf{v}}}) \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right) \right) 2c_\beta^2 \\
&\leq -\frac{c_\beta\alpha_t\Gamma}{\mu_g + \ell_{g,1}} + \left( \frac{6d_2}{L_f} \frac{\ell_{g,1}^2}{\Phi} \alpha_t + \left( \frac{18}{L_f} \ell_{f,1}^2 \alpha_t + \frac{27\ell_{g,1}^2\alpha_t}{c_{\mathbf{v}}} \right) \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right) \right) 2c_\beta^2 \\
&\leq -\frac{c_\beta\alpha_t\Gamma}{4(\mu_g + \ell_{g,1})}, \tag{249}
\end{aligned}$$

$$Z(\alpha_t) = 27\ell_{g,1}^2 \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right).$$

$$\begin{aligned} (233\mathbf{f}) &= \sum_{t=1}^T Z(\alpha_t) \left( 3d_2^2 \ell_{f,1}^2 \delta_t^2 \rho_{\mathbf{r}}^2 + (12\ell_{f,0}^2 + 6\ell_{g,1}^2 p^2) \delta_t^2 \right) \\ &= \sum_{t=1}^T 27\ell_{g,1}^2 \left( \frac{d_2}{\Psi} + \frac{d_1}{\Omega} \right) \left( 3d_2^2 \ell_{f,1}^2 \rho_{\mathbf{r}}^2 + (12\ell_{f,0}^2 + 6\ell_{g,1}^2 p^2) \right) c_\delta^2 \alpha_t^2 \\ &= \mathcal{O} \left( \sum_{t=1}^T (d_1 + d_2) (\alpha_t^2 \rho_{\mathbf{r}}^2 + \alpha_t^2) \right). \end{aligned} \tag{251}$$

$$\begin{aligned} (233g) &= \frac{36}{\Psi} D_{\mathbf{y},T} + \frac{36}{\Omega} D_{\mathbf{x},T} + \frac{12}{\Phi} G_{\mathbf{y},T} + \frac{36}{\Psi \rho_{\mathbf{v}}^2} G_{\mathbf{v},T} + \frac{36}{\Omega \rho_{\mathbf{v}}^2} G_{\mathbf{x},T} \\ &= \mathcal{O} \left( D_{\mathbf{y},T} + D_{\mathbf{x},T} + G_{\mathbf{y},T} + \frac{1}{\alpha_T} (G_{\mathbf{v},T} + G_{\mathbf{x},T}) \right). \end{aligned} \quad (252)$$

4345 Bounding [\(233h\)](#page-73-9) . From γt+1 = cγαt, ηt+1 = cηαt, λt+1 = cλα<sup>t</sup> and ρ 2 <sup>v</sup> = cvαt, we have

4356 Bounding [\(233i\)](#page-73-1) . From β<sup>t</sup> = cβαt, δ<sup>t</sup> = cδαt, we have

4370 4371 Moreover, we have

4384 4385 From [\(254\)](#page-79-0), [\(255\)](#page-79-1) and ρ 2 <sup>v</sup> = cvαt, we get

$$\begin{aligned}
(233h) &= 2 \sum_{t=1}^T \frac{\gamma_{t+1}^2}{\Phi} \frac{\hat{\sigma}_{gy}^2}{b} + 3 \left( \frac{\hat{\sigma}_{gy}^2}{b\rho_{\mathbf{V}}^2} + \frac{\hat{\sigma}_{fy}^2}{b} \right) \sum_{t=1}^T \frac{\lambda_{t+1}^2}{\Psi} + 3 \left( \frac{\hat{\sigma}_{gx}^2}{b\rho_{\mathbf{V}}^2} + \frac{\hat{\sigma}_{fx}^2}{b} \right) \sum_{t=1}^T \frac{\eta_{t+1}^2}{\Omega} \\
&= 2 \sum_{t=1}^T \frac{c_{\gamma}^2 \alpha_t^2}{\Phi} \frac{\hat{\sigma}_{gy}^2}{b} + 3 \left( \frac{\hat{\sigma}_{gy}^2}{b\rho_{\mathbf{V}}^2} + \frac{\hat{\sigma}_{fy}^2}{b} \right) \sum_{t=1}^{T+1} \frac{c_{\lambda}^2 \alpha_t^2}{\Psi} + 3 \left( \frac{\hat{\sigma}_{gx}^2}{b\rho_{\mathbf{V}}^2} + \frac{\hat{\sigma}_{fx}^2}{b} \right) \sum_{t=1}^T \frac{c_{\eta}^2 \alpha_t^2}{\Omega} \\
&= \mathcal{O} \left( \left( \frac{\hat{\sigma}_{gy}^2}{b} + \frac{\hat{\sigma}_{gy}^2}{b\alpha_t} + \frac{\hat{\sigma}_{fy}^2}{b} + \frac{\hat{\sigma}_{gx}^2}{b\alpha_t} + \frac{\hat{\sigma}_{fx}^2}{b} \right) \sum_{t=1}^T \alpha_t^2 \right). \tag{253}
\end{aligned}$$

$$\begin{aligned} D(\alpha_t, \beta_t, \delta_t) &= 6\ell_{f,1}(1 + 2\frac{\ell_{g,1}}{\mu_g}) + \frac{3\ell_{f,1}\ell_{g,1}}{\mu_g}(\alpha_t - L_f\alpha_t^2) + \frac{24\ell_{g,1}}{L_{\mu_g}\mu_g}\frac{\Gamma}{\beta_t} + \frac{96\ell_{g,1}\nu^2}{L_{\mu_g}\mu_g^3}\frac{\Upsilon}{\delta_t} \\ &= 6\ell_{f,1}(1 + 2\frac{\ell_{g,1}}{\mu_g}) + \frac{3\ell_{f,1}\ell_{g,1}}{\mu_g}(\alpha_t - L_f\alpha_t^2) + \frac{24\ell_{g,1}}{L_{\mu_g}\mu_g}\frac{\Gamma}{c_\beta\alpha_t} + \frac{96\ell_{g,1}\nu^2}{L_{\mu_g}\mu_g^3}\frac{\Upsilon}{c_\delta\alpha_t} \\ &= \mathcal{O}\left(\alpha_t + \frac{1}{\alpha_t}\right), \end{aligned}$$

and

$$\sum_{t=1}^T D(\alpha_t, \beta_t, \delta_t) (\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2) := \mathcal{O} \left( \sum_{t=1}^T \left( \alpha_t + \frac{1}{\alpha_t} \right) (\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2) \right). \quad (254)$$

$$R(\rho_{\mathbf{v}}) = 9d_2^2 \frac{\ell_{g,1}^2}{\Phi} + 18d_2^2 \frac{\ell_{f,1}^2}{\Psi} + 6(3\ell_{f,1}^2 + \frac{3\ell_{g,1}^2}{4\rho_{\mathbf{v}}^2}) \frac{d_2^2}{\Psi} = \mathcal{O} \left( \left(1 + \frac{1}{\rho_{\mathbf{v}}^2}\right) d_2^2 \right),$$

$$\hat{R}(\rho_{\mathbf{v}}) = 18d_1^2 \frac{\ell_{f,1}^2}{\Omega} + 6(3\ell_{f,1}^2 + \frac{3\ell_{g,1}^2}{4\rho_{\mathbf{v}}^2}) \frac{d_1^2}{\Omega} = \mathcal{O} \left( \left(1 + \frac{1}{\rho_{\mathbf{v}}^2}\right) d_1^2 \right),$$

which, implies that

$$\begin{aligned} & R(\rho_{\mathbf{v}})T\rho_{\mathbf{r}}^2 + \hat{R}(\rho_{\mathbf{v}})T\rho_{\mathbf{s}}^2 + 18d_1^2\ell_{g,1}^2\frac{T\rho_{\mathbf{s}}^2}{\Omega\rho_{\mathbf{v}}^2} + 18d_2^2\ell_{g,1}^2\frac{T\rho_{\mathbf{r}}^2}{\Psi\rho_{\mathbf{v}}^2} \\ &= \mathcal{O}\left(\left(1 + \frac{1}{\rho_{\mathbf{v}}^2}\right)T(d_1^2\rho_{\mathbf{s}}^2 + d_2^2\rho_{\mathbf{r}}^2) + \frac{T}{\rho_{\mathbf{v}}^2}(d_2^2\rho_{\mathbf{r}}^2 + d_1^2\rho_{\mathbf{s}}^2)\right). \end{aligned} \quad (255)$$

$$(233i) \leq \mathcal{O} \left( \sum_{t=1}^T \left( \alpha_t + \frac{1}{\alpha_t} \right) (\rho_s^2 + \rho_r^2) + \left(1 + \frac{1}{\alpha_T}\right) T (d_2^2 \rho_r^2 + d_1^2 \rho_s^2) \right). \quad (256)$$

Combining the outcomes [\(233i\)](#page-73-1) . Combining inequalities [\(240\)](#page-75-2), [\(243\)](#page-76-2), [\(244\)](#page-76-3), [\(246\)](#page-77-1), [\(250\)](#page-78-0), [\(251\)](#page-78-1), [\(252\)](#page-78-2), [\(253\)](#page-79-2), and [\(256\)](#page-79-3)

leads to

$$\begin{aligned} & \frac{\alpha_T}{2} \sum_{t=1}^T \mathbb{E} \left[ \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \boldsymbol{\rho}}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \right] + \Lambda \\ & \leq \mathcal{O} \left( V_T + \sum_{t=1}^T \alpha_t (\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2) + \sum_{t=1}^T \alpha_t^2 + \frac{H_{2,T}}{\alpha_T} + \sum_{t=1}^T (d_1 + d_2)(\alpha_t^2 \rho_{\mathbf{r}}^2 + \alpha_t^2) \right) \\ & + \mathcal{O} \left( D_{\mathbf{y}, T} + D_{\mathbf{x}, T} + G_{\mathbf{y}, T} + \frac{1}{\alpha_T} (G_{\mathbf{v}, T} + G_{\mathbf{x}, T}) \right) \\ & + \mathcal{O} \left( \sum_{t=1}^T \left( \frac{\hat{\sigma}_{g_{\mathbf{y}}}^2 \alpha_t^2}{\bar{b}} + \frac{\hat{\sigma}_{g_{\mathbf{y}}}^2 \alpha_t}{\bar{b}} + \frac{\hat{\sigma}_{f_{\mathbf{y}}}^2 \alpha_t^2}{b} + \frac{\hat{\sigma}_{g_{\mathbf{x}}}^2 \alpha_t}{\bar{b}} + \frac{\hat{\sigma}_{f_{\mathbf{x}}}^2 \alpha_t^2}{b} \right) \right) \\ & + \mathcal{O} \left( \sum_{t=1}^T \left( \alpha_t + \frac{1}{\alpha_t} \right) (\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2) + \left(1 + \frac{1}{\alpha_T}\right) T (d_2^2 \rho_{\mathbf{r}}^2 + d_1^2 \rho_{\mathbf{s}}^2) \right). \end{aligned}$$

From the definition of Λ in [\(102\)](#page-34-7), we have

$$\begin{aligned} -\Lambda &= \Gamma \sum_{t=1}^T (\mathbb{E}[\theta_t^\mathbf{y}] - \mathbb{E}[\theta_{t+1}^\mathbf{y}]) + \Upsilon \sum_{t=1}^T (\mathbb{E}[\theta_t^\mathbf{y}] - \mathbb{E}[\theta_{t+1}^\mathbf{y}]) + \frac{1}{\Phi} \sum_{t=1}^T (\mathbb{E}\|e_t^g\|^2 - \mathbb{E}\|e_{t+1}^g\|^2) \\ &+ \frac{1}{\Psi} \sum_{t=1}^T (\mathbb{E}\|e_t^\mathbf{y}\|^2 - \mathbb{E}\|e_{t+1}^\mathbf{y}\|^2) + \frac{1}{\Omega} \sum_{t=1}^T \left( \mathbb{E}\|e_t^f\|^2 - \mathbb{E}\|e_{t+1}^f\|^2 \right) \\ &\leq \Gamma \theta_1^\mathbf{y} + \Upsilon \theta_1^\mathbf{y} + \frac{\hat{\sigma}_{g\mathbf{y}}^2}{\Phi} + \frac{\hat{\sigma}_{g\mathbf{y}\mathbf{y}}^2 + \hat{\sigma}_{f\mathbf{y}}^2}{\Psi} + \frac{\hat{\sigma}_{g\mathbf{x}\mathbf{y}}^2 + \hat{\sigma}_{f\mathbf{x}}^2}{\Omega}. \end{aligned} \quad (257)$$

From [\(23\)](#page-4-8), we have

$$\hat{\sigma}^2 := \hat{\sigma}_{g_y}^2 + \hat{\sigma}_{g_{yy}}^2 + \hat{\sigma}_{f_y}^2 + \hat{\sigma}_{g_{xy}}^2 + \hat{\sigma}_{f_x}^2.$$

From [\(26\)](#page-5-6), we have

$$\begin{aligned} \alpha_t &= \frac{1}{(d_1 + d_2)^{3/4}(c+t)^{1/3}}, \quad \beta_t = c_\beta \alpha_t, \quad \delta_t = c_\delta \alpha_t, \quad \rho_{\mathbf{v}}^2 = c_{\mathbf{v}} \alpha_t, \\ \gamma_{t+1} &= c_\gamma \alpha_t, \quad \eta_{t+1} = c_\eta \alpha_t, \quad \lambda_{t+1} = c_\lambda \alpha_t, \quad \rho_{\mathbf{r}}^2 = \frac{1}{d_2^2 T}, \quad \rho_{\mathbf{s}}^2 = \frac{1}{d_1^2 T}, \\ b &= \frac{T^{1/3}}{(d_1 + d_2)^{3/2}}, \quad \bar{b} = \frac{T^{2/3}}{(d_1 + d_2)^{3/4}}. \end{aligned} \quad (258)$$

4476 4477 where second inequality holds because we have

Thus, using [\(257\)](#page-80-0), [\(258\)](#page-80-1), and rearranging the terms, we get

$$\begin{aligned} & \sum_{t=1}^T \mathbb{E} \left[ \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \right] \\ & \leq \frac{2}{\alpha_T} \mathcal{O} \left( V_T + \sum_{t=1}^T \alpha_t (\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2) + \sum_{t=1}^T \alpha_t^2 + \frac{H_{2,T}}{\alpha_T} + \sum_{t=1}^T (d_1 + d_2)(\alpha_t^2 \rho_{\mathbf{r}}^2 + \alpha_t^2) \right) \\ & + \frac{2}{\alpha_T} \mathcal{O} \left( D_{\mathbf{y},T} + D_{\mathbf{x},T} + G_{\mathbf{y},T} + \frac{1}{\alpha_T} (G_{\mathbf{v},T} + G_{\mathbf{x},T}) \right) \\ & + \frac{2}{\alpha_T} \mathcal{O} \left( \sum_{t=1}^T \left( \frac{\hat{\sigma}_{g_{\mathbf{y}}}^2 \alpha_t^2}{b} + \frac{\hat{\sigma}_{g_{\mathbf{y}}}^2 \alpha_t}{b} + \frac{\hat{\sigma}_{f_{\mathbf{y}}}^2 \alpha_t^2}{b} + \frac{\hat{\sigma}_{g_{\mathbf{x}}}^2 \alpha_t}{b} + \frac{\hat{\sigma}_{f_{\mathbf{x}}}^2 \alpha_t^2}{b} \right) \right) \\ & + \frac{2}{\alpha_T} \mathcal{O} \left( \sum_{t=1}^T \left( \alpha_t + \frac{1}{\alpha_t} \right) (\rho_{\mathbf{s}}^2 + \rho_{\mathbf{r}}^2) + \left(1 + \frac{1}{\alpha_T}\right) T (d_2^2 \rho_{\mathbf{r}}^2 + d_1^2 \rho_{\mathbf{s}}^2) \right) \\ & + \frac{2}{\alpha_T} \mathcal{O} \left( \theta_1^{\mathbf{y}} + \theta_1^{\mathbf{v}} + \hat{\sigma}^2 \right) \\ & \leq \mathcal{O} \left( (d_1 + d_2)^{3/4} T^{1/3} (V_T + D_{\mathbf{y},T} + D_{\mathbf{x},T} + G_{\mathbf{y},T} + \Delta_1 + \hat{\sigma}^2) \right. \\ & \quad \left. + (d_1 + d_2)^{3/2} T^{2/3} (H_{2,T} + G_{\mathbf{v},T} + G_{\mathbf{x},T}) \right), \end{aligned} \tag{259}$$

$$\begin{aligned} \sum_{t=1}^T \alpha_t^3 &= \sum_{t=1}^T \frac{1}{(d_1 + d_2)^{9/4}(c+t)} \leq \sum_{t=1}^T \frac{1}{(d_1 + d_2)^{9/4}(1+t)} \leq \frac{\log(T+1)}{(d_1 + d_2)^{9/4}}, \\ \sum_{t=1}^T \alpha_t^2 &= \sum_{t=1}^T \frac{1}{(d_1 + d_2)^{3/2}(c+t)^{2/3}} \leq \sum_{t=1}^T \frac{1}{(d_1 + d_2)^{3/2}(1+t)^{2/3}} \leq \frac{T^{1/3}}{(d_1 + d_2)^{3/2}}, \\ \sum_{t=1}^T \alpha_t &= \sum_{t=0}^T \frac{1}{(d_1 + d_2)^{3/4}(c+t)^{1/3}} \leq \sum_{t=1}^T \frac{1}{(d_1 + d_2)^{3/4}(1+t)^{1/3}} \leq \frac{3T^{2/3}}{2(d_1 + d_2)^{3/4}}, \\ \sum_{t=1}^T \frac{1}{\alpha_t} &= \sum_{t=0}^T (d_1 + d_2)^{3/4}(c+t)^{1/3} \leq \frac{3}{2}(d_1 + d_2)^{3/4}T^{4/3}. \end{aligned}$$

Then, note that, we have

$$\begin{aligned} & \frac{1}{2} \sum_{t=1}^T \mathbb{E} \left[ \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \right] \\ & \leq \sum_{t=1}^T \mathbb{E} \left[ \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \right] \\ & + \sum_{t=1}^T \mathbb{E} \left[ \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))) - \mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \right]. \end{aligned}$$

4510 From non-expansiveness of the projection operator and Lemma [D.4,](#page-42-0) we have

$$\begin{aligned} 4511 \quad & \quad \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))) - \mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \\ 4512 \quad & \leq \|\nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)) - \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))\|^2 \\ 4513 \quad & \leq \frac{(\rho_s d_1 + \rho_r d_2)^2 \ell_{f, 1}^2}{4} \\ 4514 \quad & \leq \frac{(\rho_s^2 d_1^2 + \rho_r^2 d_2^2) \ell_{f, 1}^2}{2}. \end{aligned}$$

4527 4528 Applying the upper bound in [\(259\)](#page-81-0) yields

4547 4548 This completes the proof.

This implies

$$\begin{aligned} & \frac{1}{2} \sum_{t=1}^T \mathbb{E} \left[ \left\| \mathcal{P}_{\mathcal{X}, \alpha_t} (\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))) \right\|^2 \right] \\ & \leq \sum_{t=1}^T \mathbb{E} \left[ \left\| \mathcal{P}_{\mathcal{X}, \alpha_t} (\mathbf{x}_t; \nabla f_{t, \rho}(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))) \right\|^2 \right] + \frac{T(\rho_s^2 d_1^2 + \rho_r^2 d_2^2) \ell_{f,1}^2}{2}. \end{aligned}$$

$$\begin{aligned} & \frac{1}{2} \sum_{t=1}^T \mathbb{E} \left[ \left\| \mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t))) \right\|^2 \right] \\ & \leq \mathcal{O} \left( (d_1 + d_2)^{3/4} T^{1/3} (V_T + D_{\mathbf{y}, T} + D_{\mathbf{x}, T} + G_{\mathbf{y}, T} + \Delta_1 + \hat{\sigma}^2) \right. \\ & \quad \left. + (d_1 + d_2)^{3/2} T^{2/3} (H_{2, T} + G_{\mathbf{v}, T} + G_{\mathbf{x}, T}) \right) \\ & + \frac{T(\rho_s^2 d_1^2 + \rho_r^2 d_2^2) \ell_{f,1}^2}{2}. \end{aligned}$$

Thus, from ρ 2 <sup>r</sup> = 1 d 2 <sup>2</sup>T and ρ 2 <sup>s</sup> = 1 d 2 <sup>1</sup>T , we get

$$\begin{aligned} & \sum_{t=1}^T \mathbb{E} \left[ \|\mathcal{P}_{\mathcal{X}, \alpha_t}(\mathbf{x}_t; \nabla f_t(\mathbf{x}_t, \mathbf{y}_t^*(\mathbf{x}_t)))\|^2 \right] \\ & \leq \mathcal{O} \left( (d_1 + d_2)^{3/4} T^{1/3} (V_T + D_{\mathbf{y}, T} + D_{\mathbf{x}, T} + G_{\mathbf{y}, T} + \Delta_1 + \hat{\sigma}^2) \right. \\ & \quad \left. + (d_1 + d_2)^{3/2} T^{2/3} (H_{2, T} + G_{\mathbf{v}, T} + G_{\mathbf{x}, T}) \right). \end{aligned}$$