000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054 Online bilevel optimization (OBO) is a powerful framework for machine learning problems where both outer and inner objectives evolve over time, requiring dynamic updates. Current OBO approaches rely on deterministic *window-smoothed* regret minimization, which may not accurately reflect system performance when functions change rapidly. In this work, we introduce a novel search direction and show that both first- and zerothorder (ZO) stochastic OBO algorithms leveraging this direction achieve sublinear stochastic bilevel regret without window smoothing. Beyond these guarantees, our framework enhances efficiency by: (i) reducing oracle dependence in hypergradient estimation, (ii) updating inner and outer variables alongside the linear system solution, and (iii) employing ZO-based estimation of Hessians, Jacobians, and gradients. Experiments on online parametric loss tuning and black-box adversarial attacks validate our approach.

## 1. Introduction

Bilevel optimization (BO) minimizes an outer objective dependent on an inner problem's solution. Originating in game theory (Stackelberg, 1952) and formalized in mathematical optimization (Bracken & McGill, 1973), BO finds applications in operations research, engineering, economics (Dempe, 2002), and image processing (Crockett et al., 2022). Recently, BO has gained traction in machine learning, including hyperparameter optimization (Franceschi et al., 2018), meta-learning (Finn et al., 2017), reinforcement learning (Stadie et al., 2020), and neural architecture search (Liu et al., 2018a). In the *offline setting*, BO solves the following problem:
x
∗ ∈ argminx∈Rd1 f(x, y
∗(x))
subj. to y
∗(x) = argminy∈Rd2 g(x, y), (BO)
Preliminary work. Under review by the International Conference on Machine Learning (ICML). Do not distribute.

1

# Stochastic Regret Guarantees For Online Zeroth- And First-Order Bilevel Optimization

## Anonymous Authors1 Abstract

1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

where f and g are the outer and inner objectives, and x and y are their respective optimization variables. OBO (Tarzanagh et al., 2024) addresses dynamic scenarios where objectives evolve over time, requiring the agent to update the outer decision in response to the optimal inner decision. Similar to online single-level optimization (OSO) (Zinkevich, 2003), OBO involves iterative decisionmaking without prior knowledge of outcomes (Tarzanagh et al., 2024; Lin et al., 2024; Bohne et al., 2024). Let T be the total number of rounds. Define xt *∈ X ⊂* R
d1 as the decision variable and ft : X × R
d2 → R as the outer function. Similarly, define yt ∈ R
d2 and gt : X × R
d2 → R for the inner problem, where y
∗ t
(x) = argminy∈Rd2 gt(x, y).

OBO can be seen as a *single-player* problem, where the player selects xt without knowing y
∗
t(x), using yt as an estimate based on gt. Alternatively, it can be framed as a two-player game (Stackelberg, 1952), where the leader (xt)
competes with the follower (yt), who selects y
∗ t
(x) based on limited knowledge of gt; see Section 2. This framework includes online and adversarial variants of (BO), such as online actor-critic algorithms (Zhou et al., 2020), online meta-learning (Finn et al., 2019), and online hyperparameter optimization (Lin et al., 2024). The inner and outer functions may be time-varying, adversarial, unavailable a priori, and require *nonstationary* optimization.

## 1.1. Our Contributions

This paper addresses stochastic OBO, introducing novel first1- and zeroth-order methods to minimize stochastic bilevel regret. Key contributions are summarized below. - Stochastic regret minimization without windowsmoothing. Existing OBO methods (Tarzanagh et al., 2024; Lin et al., 2024; Huang et al., 2023; Bohne et al., 2024)
rely on deterministic *window-smoothed* regret minimization, which may not accurately reflect system performance when functions change rapidly. We address these limitations by introducing a novel search direction (Section 3) and proving that both first-order and ZO methods achieve sublinear *stochastic bilevel regret without window-smoothing*
(w = 1); see Theorems 3.6 and 4.2 and Table 1. - **OBO with function value oracle feedback.** In large-scale 1First-order refers to the setting where only partial gradients of the leader objective ft are accessible, while second-order information is still required for the follower objective gt; refer to Section 3.

| OBO       | Window Size   | System       | Stochastic   | Const.      | Only Func.   | Local               |                |                          |          |
|-----------|---------------|--------------|--------------|-------------|--------------|---------------------|----------------|--------------------------|----------|
| Method    | in Regret (w) | Iters.       | Regret       | Regret Min. | Feedback     | Regret Bound        |                |                          |          |
| OAGD      | o(T)          | N.A. (Exact) | ✗            | ✗           | ✗            | T w + H1,T + H2,T   |                |                          |          |
| SOBOW     | o(T)          | O(κg log κg) | ✗            | ✗           | ✗            | T w + VT + H2,T     |                |                          |          |
| SOBBO     | o(T)          | O(κg log κg) | ✓            | ✓           | ✗            | T w σ 2 + VT + H2,T |                |                          |          |
| SOGD      | 1             | 1            | ✓            | ✓           | ✗            | T 1                 | 2 + ∆T ) + T 2 |                          |          |
| 3 (σ      | 3 ΨT          |              |              |             |              |                     |                |                          |          |
| ZO-SOGD   | 1             | 1            | ✓            | ✓           | ✓            | (d1 + d2) 3         | 1              | 2 + ∆ˆ T ) + (d1 + d2) 3 | 2 3 Ψˆ T |
| 4 T 3 (ˆσ | 2 T           |              |              |             |              |                     |                |                          |          |

Further, we have

$$f_{t}(\mathbf{x},\mathbf{y}_{t}^{*}(\mathbf{x})):=\mathbb{E}_{\xi_{t}\sim{\mathcal{D}}_{f}}\left[f_{t}(\mathbf{x},\mathbf{y}_{t}^{*}(\mathbf{x});\xi_{t})\right].$$

Here, (Df , Dg) are data distributions. Note that our setting is stochastic, and only noisy evaluations of the function, gradient, and Hessian are accessible. Unlike OSO, where true losses are revealed immediately, in OBO, the outer function ft(x, y
∗
t(x)) is unavailable for updating xt. Moreover, ft(x, y
∗ t
(x)) is typically non-convex in x, making standard regret definitions from online convex optimization (Hazan, 2016b) inapplicable.

Given a sequence {αt ∈ R++}
T
t=1, we define the following notion of *bilevel local regret*:

$$\text{BL-Reg}_{T}:=\sum_{t=1}^{T}\mathbb{E}\left[\left|\left|\mathcal{P}_{\mathcal{X},\alpha_{t}}\left(\mathbf{x}_{t};\nabla f_{t}(\mathbf{x}_{t},\mathbf{y}_{t}^{*}(\mathbf{x}_{t}))\right)\right.\right|\right|^{2}\right],\tag{2a}$$

with

$${\cal P}_{{\cal X},\alpha_{t}}\left({\bf x}_{t};\nabla f_{t}({\bf x}_{t},{\bf y}_{t}^{*}({\bf x}_{t}))\right)$$ $$=\frac{1}{\alpha_{t}}\Big{(}{\bf x}_{t}-\Pi_{{\cal X}}\big{[}{\bf x}_{t}-\alpha_{t}\nabla f_{t}({\bf x}_{t},{\bf y}_{t}^{*}({\bf x}_{t}))\big{]}\Big{)}.\tag{2b}$$

The local regret (2) compares the leader's decision xt to the stationary points x
∗
tsatisfying PX ,αt(x
∗
t; ∇ft(x
∗
t, y
∗
t(x
∗
t))) = 0. This can also be viewed as dynamic local regret, as the baseline corresponds to a stationary point of the leader's objective ft.

Previous work on (nonconvex) OBO examined unconstrained local regret using window-smoothed objectives:
Ft,w(x, y) = (1/w)Pw−1 i=0 ft−i(x, y). For w = 1 and X = R
d1, this reduces to (2). Tarzanagh et al. (2024);
Lin et al. (2024) showed that w = o(T) ensures sublinear regret under slow variations in {Ft,w}
T
t=1, while rapid changes can lead to deviations. However, smoothing may misrepresent regret (Figure 1). This paper introduces a new projection-based local regret notion (2) without smoothing, and establishes sublinear regret for constrained OBO. Online Gradient Descent (OGD). One of the most widely used algorithms for online (single-level) optimization is

## 2. Preliminaries

055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 Notation. R
d denotes the d-dimensional real space, with R 
d+ and R
d++ as its positive and negative orthants. Vectors are bold lower-case letters (e.g., x, y), with ⟨x, y⟩ for inner product and ∥·∥ for Euclidean norm. A gradient is ∇x, with
∇2xy = ∇x∇y. A function is L-smooth if its gradient is L-Lipschitz. The Euclidean projection onto a convex set X
is ΠX (z) = argminx∈X (1/2)∥x−z∥
2. The set {1*, . . . , T*}
is denoted by [T], and E[·] represents expectation. Lastly, O(·) hides problem-independent constants. Stochastic OBO Setting. Let T be the total rounds
(Tarzanagh et al., 2024). Define xt *∈ X ⊂* R
d1 as the decision variable and ft : X × R
d2 as the outer objective.

The inner decision variable and objective are yt ∈ R
d2 and gt : X × R
d2, where the optimal inner decision is:

$$\mathbf{y}_{t}^{*}(\mathbf{x})\in\operatorname*{argmin}_{\mathbf{y}\in\mathbb{R}^{d_{2}}}\left\{g_{t}(\mathbf{x},\mathbf{y}):=\underset{\zeta_{t}\sim\mathcal{D}_{g}}{\mathbb{E}}\left[g_{t}(\mathbf{x},\mathbf{y};\zeta_{t})\right]\right\}.\tag{1}$$

Table 1. Comparison of OBO algorithms based on regret window size (w), system solver iterations, stochastic regret, constrained regret minimization, function feedback settings, and local regret bounds. Here, κg denotes the condition number of gt, while VT , Hp,T , ∆T ,
ΨT , ∆ˆ T , and Ψˆ T are defined in (10), (13), and (25), respectively. The compared algorithms include OAGD (Tarzanagh et al., 2024),
SOBOW (Lin et al., 2024), and SOBBO (Bohne et al., 2024).

and black-box settings (Chen et al., 2017; Nesterov, 2005), first- and second-order information is often unavailable or costly. Constructing accurate (hyper)-gradient estimators using only function value oracles is particularly challenging due to BO's nested structure. Existing methods rely on gradient, Hessian, and Jacobian oracles, limiting scalability (Franceschi et al., 2017; Ghadimi & Wang, 2018).

We propose Algorithm 2, which estimates Hessians, Jacobians, and gradients using function value oracles, achieving sublinear local regret (Theorem 4.2).

- **OBO with one subproblem solver iteration.** A major challenge in BO is solving implicit systems to approximate the hypergradient (Ji et al., 2021; Chen et al., 2021). While efficient offline BO methods exist (Ji et al., 2021; Dagreou ´ et al., 2022), extending them to OBO is difficult due to timevarying objectives. SOBOW (Lin et al., 2024) partially addresses this using a conjugate gradient (CG) algorithm with increasing iterations (Table 1). We improve upon SOBOW by introducing Algorithms 1 and 2, which require only a single subproblem solver iteration.

OGD (Zinkevich, 2003). The procedure for OGD is as follows: For each t ∈ [T], the algorithm selects xt ∈ X ,
observes the function ft : X ⊂ R
d → R, and updates according to

$$\mathbf{x}_{t+1}=\Pi_{\mathcal{X}}\big(\mathbf{x}_{t}-\alpha_{t}\nabla f_{t}(\mathbf{x}_{t})\big),\ \ \ \alpha_{t}>0.$$ (OGD)
In the following, we adapt OGD to OBO and introduce a novel framework that requires limited feedback and can utilize ZO updates within a single-loop structure.

## 3. Stochastic Obo With Access To First And Second Order Oracles

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 To adapt OGD to OBO, Tarzanagh et al. (2024); Lin et al.

(2024); Bohne et al. (2024) developed a variant alternating between inner and outer OGD, achieving sublinear bilevel regret bounds. We introduce a new search direction that enables sublinear bilevel regret without window smoothing.

To compute the hypergradient ∇ft(x, y
∗
t(x)) where y
∗
t(x)
is defined in (1), since ∇ygt(x, y
∗
t(x)) = 0, using the implicit function theorem, yields where ∇y
∗ t
(x)∇2y gt (x, y
∗ t
(x)) + ∇2xygt (x, y
∗ t
(x)) = 0.

As the exact y
∗
t(x) is not available, we estimate the hypergradient of ft at (x, y) by

$$\tilde{\nabla}f_{t}({\bf x},{\bf y}):=\nabla_{\bf x}f_{t}({\bf x},{\bf y})+\nabla_{\bf x}^{2}g_{t}\left({\bf x},{\bf y}\right){\bf v}_{t}^{*}({\bf x}),\tag{4a}$$

where

$$\nabla_{\mathbf{y}}^{2}g_{t}\left(\mathbf{x},\mathbf{y}\right)\mathbf{v}_{t}^{*}(\mathbf{x})+\nabla_{\mathbf{y}}f_{t}(\mathbf{x},\mathbf{y})=0.$$

An accurate solution of (4b) is crucial for tight regret bounds. Tarzanagh et al. (2024) assumes an exact solution, which is restrictive in large-scale settings. To address this, Lin et al. (2024) proposed an efficient OBO algorithm with window averaging, using CG methods to solve (4b), which Algorithm 1 SOGD
Require: (x1, y1, v1) *∈ X ×* R
d2 × R
d2; T ∈ N; p ∈
R++; stepsizes {(αt, βt, δt) ∈ R
3
++}
T
t=1; parameters
{(γt, λt, ηt)}
T
t=1 ∈ (0, 1); zt := (xt, yt).

For t = 1 to T do:
S1. Draw samples Bt and B¯t with batch sizes b and ¯b.

Get search directions d y t, d v t, and d x t:
d yy t(zt; B¯t) = ∇ygt(zt; B¯t), (7a)
d y t = d yy t(zt; B¯t) + (1 − γt)(d y t−1 − d yy t(zt−1; B¯t)),
d vv t (zt; Bt) = ∇yft(zt; Bt) + ∇
2 ygtzt; B¯tvt, (7b)
d v t = d vv t (zt; Bt) + (1 − λt)(d v t−1 − d vv t (zt−1; Bt)),
d xx t (zt; Bt) = ∇xft(zt; Bt) + ∇
2 xygtzt; B¯tvt, (7c)
d x t = d xx t (zt; Bt) + (1 − ηt)(d x t−1 − d xx t (zt−1; Bt)).

S2. Update inner, system, and outer solutions:
yt+1 = yt − βtd y t, vt+1 = ΠZp
-vt − δtd v t
,
is equivalent to:
minvt∈Rd2 (1/2)∥∇2ygt (x, y) vt + ∇yft(x, y)∥

$$f_{t}(\mathbf{x},\mathbf{y})\|^{2}.\quad({\boldsymbol{\mathbf{5}}})$$

New Search Direction for OBO. Next, we introduce a novel search direction that enables both first- and ZO stochastic OBO algorithms to achieve sublinear bilevel regret without smoothing. We first state the following lemma:
Lemma 3.1. Let w = t and W = 1/η in the window-smoothed gradient ∇ˆ Ft,ν(xt, yt; Bt) =
(1/W)Pw−1 i=0 ν i∇ˆ ft−i(xt−i, yt−i; Bt−i), *where* Bt :=
{ξ1, . . . , ξb} is drawn i.i.d. from Df *. Then,*

$${\hat{\nabla}}F_{t,\nu}(\mathbf{x}_{t},\mathbf{y}_{t};{\mathcal{B}}_{t})=\sum_{j=1}^{t}\eta(1-\eta)^{t-j}{\hat{\nabla}}f_{j}(\mathbf{x}_{j},\mathbf{y}_{j};{\mathcal{B}}_{j}).$$

Furthermore, we have ∇ˆ Ft,ν(xt, yt; Bt) = dˆx t *with* dˆx t = η∇ˆ ft(xt, yt; Bt) + (1 − η)dˆx t−1, and dˆ1 =
(1/W)∇ˆ f1(x1, y1; B1) *for all* t ≥ 2.

As shown in Lemma 3.1, for a specific choice of w and W,
the time-smoothed gradient forms a recursive momentumtype search direction. However, achieving sublinear regret in stochastic OBO requires large-window smoothing (w =
o(T)). To address this, we propose the following search direction:

$$(4\mathbf{b})$$
$$\mathbf{d}_{t}^{\mathbf{x}}=\eta\nabla f_{t}(\mathbf{x}_{t},\mathbf{y}_{t};\mathcal{B}_{t})+(1-\eta)\mathbf{d}_{t-1}^{\mathbf{x}}\tag{6a}$$ $$+(1-\eta)(\nabla f_{t}(\mathbf{x}_{t},\mathbf{y}_{t};\mathcal{B}_{t})-\nabla f_{t}(\mathbf{x}_{t-1},\mathbf{y}_{t-1};\mathcal{B}_{t})).\tag{6b}$$

This direction is used for updating x, with similar updates for y and v, as discussed below and detailed in Algorithm 1.

$$\nabla f_{t}({\bf x},{\bf y}_{t}^{*}({\bf x}))=\nabla_{\bf x}f_{t}\left({\bf x},{\bf y}_{t}^{*}({\bf x})\right)\tag{3}$$ $$+\nabla{\bf y}_{t}^{*}({\bf x})\nabla_{\bf y}f_{t}\left({\bf x},{\bf y}_{t}^{*}({\bf x})\right),$$

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 The quadratic optimization formulation of (4b) in (5) leads to single-loop frameworks such as Dagreou et al. ´ (2022). Inspired by this, we present Simultaneous Online Gradient Descent (SOGD) for constrained OBO, outlined in Algorithm 1. SOGD evolves the follower's decision (inner)
variable, the linear system solution, and the leader's decision (outer) variable simultaneously at each step for given batches B := {ξ1, . . . , ξb} and B¯ := {ζ1*, . . . , ζ*¯b}, which are drawn i.i.d. from unknown distributions Df and Dg with batch sizes b and ¯b. Computing directions in S1. of Algorithm 1 does not require ∇2ygt (xt, yt) and ∇2xygt (xt, yt),
only their product with a vector, at the same cost as computing a gradient. Technically, it utilizes an auxiliary variable vt and concurrently updates yt, vt, and xt at each local iteration t. Moreover, S2. of Algorithm 1 introduces an auxiliary projection ΠZp on the ball Zp defined as follows:

$$\Pi_{\mathbb{Z}_{p}}(\mathbf{v}):=\min\left\{1,\frac{p}{||\mathbf{v}||}\right\}\mathbf{v},\tag{8}$$

where Zp := {v ∈ R
d2| ∥v∥ ≤ p}.

Unlike OAGD (Tarzanagh et al., 2024), which updates x and y in separate loops, SOGD updates both simultaneously. Compared to SOBOW (Lin et al., 2024), which uses multiple CG updates, our method employs a single OGD to update the inner solution, linear system, and outer variable.

Assumption 3.2. gt(x, y) is twice continuously differentiable and µg-strongly convex in y for all x ∈ X , t ∈ [T]. Assumption 3.3. Let z = [x; y] and z
′ = [x
′; y
′], where x, x
′ ∈ X and y, y
′ ∈ R
d2. For any z, z
′, and t ∈ [T]:
B1. ∃ ℓf,0 ∈ R+ s.t. ∥ft(z; ξ) − ft(z
′; ξ)∥ ≤ ℓf,0∥z − z
′∥;
B2. ∃ ℓf,1 ∈ R+ s.t. ∥∇ft(z; ξ) − ∇ft(z
′; ξ)∥ ≤ ℓf,1∥z − z
′∥;
B3. ∃ ℓg,1 ∈ R+s.t.∥∇gt(z; ζ) − ∇gt(z
′; ζ)∥ ≤ ℓg,1∥z − z
′∥;
B4. ∃ ℓg,2 ∈ R+ s.t.∥∇2gt(z; ζ)−∇2gt(z
′; ζ)∥ ≤ ℓg,2∥z−z
′∥.

Assumption 3.4. For any t ∈ [T], |ft(x, y
∗
t(x))| ≤ M for some finite constant M ∈ R++ and any x ∈ X .

Assumption 3.5. There exist constants σgy, σgyy, σgxy, σfy, σfysuch that, for all z = [x, y]:

C1. E∥∇ygt(z; ζ) − ∇ygt(z)∥ 2 ≤ σ 2gy , C2. E∥∇2ygt(z; ζ) − ∇2ygt(z)∥ 2 ≤ σ 2gyy , C3. E∥∇2xygt(z; ζ) − ∇2xygt(z)∥ 2 ≤ σ 2 gxy , C4. E∥∇yft(z; ξ) − ∇yft(z)∥ 2 ≤ σ 2 fy , C5. E∥∇xft(z; ξ) − ∇xft(z)∥ 2 ≤ σ 2 fx .
Throughout this paper, we define

$$\sigma^{2}:=\sigma_{g_{\bf y}}^{2}+\sigma_{g_{\bf y y}}^{2}+\sigma_{f_{\bf y}}^{2}+\sigma_{g_{\bf x y}}^{2}+\sigma_{f_{\bf x}}^{2}\,.\qquad(9)$$

Assumptions 3.2 and 3.3 are widely used in both BO (Chen et al., 2021; Ji et al., 2021) and OBO (Tarzanagh et al., 2024), and many bilevel machine learning problems satisfy it (Franceschi et al., 2018). Further, Assumption 3.4 is widely used in the study of non-convex online optimization (Hazan et al., 2017; Lin et al., 2024). Assumption 3.5 assumes that we have access to an unbiased stochastic gradient, Hessian and Jacobian with bounded variance, which is standard in the literature (Chen et al., 2021). Achieving sublinear dynamic regret is generally impossible due to arbitrary fluctuations in time-varying functions (Besbes et al., 2015). Existing analyses (Tarzanagh et al., 2024; Lin et al., 2024) bound regret by imposing regularity constraints on the comparator sequence. To achieve sublinear regret, we introduce the following regularities:
- **Path-length (of order** p) and **function variation**:
Tarzanagh et al. (2024) defines the following metrics for bilevel sequences:

$$H_{p,T}:=\sum_{t=2}^{T}\sup_{\mathbf{x}\in\mathcal{X}}\|\mathbf{y}_{t-1}^{*}(\mathbf{x})-\mathbf{y}_{t}^{*}(\mathbf{x})\|^{p},\tag{10}$$ $$V_{T}:=\sum_{t=2}^{T}\sup_{\mathbf{x}\in\mathcal{X}}|f_{t-1}(\mathbf{x},\mathbf{y}_{t-1}^{*}(\mathbf{x}))-f_{t}(\mathbf{x},\mathbf{y}_{t}^{*}(\mathbf{x}))|\,.$$

Path-length Hp,T measures changes in the follower's costs, while VT captures the smoothness of the leader's objective.

We use path-length for the follower and function variation for the leader, as the follower's objective is strongly convex (see Assumption 3.2), while the leader's is nonconvex.

- **Inner and Outer Gradient Variations**: Another regularity is the sequential difference between the individual gradients of the upper-level loss function:

$$D_{\mathbf{x},T}:=\sum_{t=2}^{T}\operatorname*{sup}_{\mathbf{x},\mathbf{y}}\left\|\nabla_{\mathbf{x}}f_{t-1}(\mathbf{x},\mathbf{y})-\nabla_{\mathbf{x}}f_{t}(\mathbf{x},\mathbf{y})\right\|^{2},$$ $$D_{\mathbf{y},T}:=\sum_{t=2}^{T}\operatorname*{sup}_{\mathbf{x},\mathbf{y}}\left\|\nabla_{\mathbf{y}}f_{t-1}(\mathbf{x},\mathbf{y})-\nabla_{\mathbf{y}}f_{t}(\mathbf{x},\mathbf{y})\right\|^{2}.$$
$$(11)$$

As in Huang et al.; Hallak et al. (2021), Dx,T and Dy,T
measure the gradient drift of ft relative to ft−1 for x and y, respectively. We further define deviations in the gradient, Hessian, and Jacobian of the lower-level objective as:

$$G_{\mathbf{y},T}:=\sum_{t=2}^{T}\|\nabla_{\mathbf{y}}g_{t-1}(\mathbf{x}_{t},\mathbf{y}_{t})-\nabla_{\mathbf{y}}g_{t}(\mathbf{x}_{t},\mathbf{y}_{t})\|^{2},$$ $$G_{\mathbf{y}\mathbf{y},T}:=\sum_{t=2}^{T}\|\nabla_{\mathbf{y}}^{2}g_{t-1}(\mathbf{x}_{t},\mathbf{y}_{t})-\nabla_{\mathbf{y}}^{2}g_{t}(\mathbf{x}_{t},\mathbf{y}_{t})\|^{2},\tag{12}$$ $$G_{\mathbf{xy},T}:=\sum_{t=2}^{T}\|\nabla_{\mathbf{xy}}^{2}g_{t-1}(\mathbf{x}_{t},\mathbf{y}_{t})-\nabla_{\mathbf{xy}}^{2}g_{t}(\mathbf{x}_{t},\mathbf{y}_{t})\|^{2}.$$
$$(13)^{\frac{1}{2}}$$

We introduce the following notations for simplicity:

$$\Delta_{T}:=E_{1}+V_{T},\;\;\Psi_{T}:=H_{2,T}+G_{T}+D_{T},$$

where (VT , Hp,T ) are defined in (10), and

$$\begin{array}{l}{{E_{1}:=\|\mathbf{y_{1}}-\mathbf{y_{1}^{*}}(\mathbf{x_{1}})\|^{2}+\|\mathbf{v_{1}}-\mathbf{v_{1}^{*}}(\mathbf{x_{1}})\|^{2},}}\\ {{G_{T}:=G_{\mathbf{y,}T}+G_{\mathbf{yy,}T}+G_{\mathbf{xy,}T},}}\\ {{D_{T}:=D_{\mathbf{x,}T}+D_{\mathbf{y,}T}.}}\end{array}$$
$$(14)$$

parameters ρ = (ρs, ρr), and batches B := {ξ1*, . . . , ξ*b}
and B¯ := {ζ1*, . . . , ζ*¯b}, drawn i.i.d. from Df and Dg, as:
By accounting for both DT and GT , we can represent the variations in the environments of OBO.

Theorem 3.6. Let {(ft, gt)}
T
t=1 be the sequence of functions presented to Algorithm 1, satisfying Assumptions 3.23.5. For all t ∈ [T]*, let*

$$\alpha_{t}=\frac{1}{(c+t)^{1/3}},\quad\beta_{t}=c_{\beta}\alpha_{t},\quad\delta_{t}=c_{\delta}\alpha_{t},\quad b=\bar{b}=1,$$ $$\gamma_{t+1}=c_{\gamma}\alpha_{t}^{2},\quad\eta_{t+1}=c_{\eta}\alpha_{t}^{2},\quad\lambda_{t+1}=c_{\lambda}\alpha_{t}^{2}.\tag{15}$$

Here, c, cβ, cδ, cγ, cη, and cλ *are specified in* (104). Algorithm 1 *guarantees:*

$${\rm BL-Reg}_{T}\leq{\cal O}\left(T^{1/3}(\sigma^{2}+\Delta_{T})+T^{2/3}\Psi_{T}\right),\tag{16}$$

where σ and (∆T , ΨT ) *are defined in* (9) and (13).

Theorem 3.6 bounds the regret of Algorithm 1 without window-smoothing, based on the regularities in (14). We note that the average dynamic regret BL-RegT/T ≤
O(T
−2/3(σ 2 + ∆T ) + T
−1/3ΨT ) remains sublinear under suitable conditions on ∆T and ΨT .

Remark 3.7 (**Stochastic Regret Guarantee for OBO and** OSO with w = 1). The additional terms in (6b) improve the average regret dependence on variance, achieving a T
−2/3σ 2 bound, better than the T
−1/2σ 2 bound for stochastic OBO (Bohne et al., 2024). This also provides the first regret bound without window-smoothing, unlike (Bohne et al., 2024; Tarzanagh et al., 2024; Lin et al., 2024; Huang et al., 2023). For OSO, our approach improves the T
−1/2σ 2 dependence from (Hallak et al., 2021).

## 4. Obo With Zeroth Order Oracles

Black-box optimization arises in machine learning when explicit gradients are unavailable (Chen et al., 2017). We study ZO-type OBO algorithms with limited access to the leader's and follower's objective values. Let s ∈ R
d1 and r ∈ R
d2 be vectors uniformly generated from the unit balls B1 and B2, respectively. Given positive smoothing parameters ρ = (ρs, ρr), we use the Gaussian smoothing function (Nesterov & Spokoiny, 2017) to define the OBO objectives:

$$f_{t,\rho}\left(\mathbf{x},\hat{\mathbf{y}}_{t}^{*}(\mathbf{x})\right)=\mathbb{E}\left[f_{t}(\mathbf{x}+\rho_{\mathbf{s}}\mathbf{s},\hat{\mathbf{y}}_{t}^{*}(\mathbf{x})+\rho_{\mathbf{r}}\mathbf{r};\xi)\right],\tag{17}$$

where Using (17), we provide methodology to approximate each term in (7) using ZO oracles. Specifically, following Shamir (2017), we estimate the gradient of a function h : R
d → R, querying at x − λs and x + λs, yielding an estimator (d/2λ) (h(x + λs) − h(x − λs)) s. Using this strategy, the finite-difference estimation of ∇gt,ρ(x, y),
denoted as ∇ˆ gt(x, y), is constructed for given smoothing

∇ˆ ygt(x, y; B¯) := d2 2¯bρr X ¯b i=1 (gt(x, y + ρrri; ζi) −gt(x, y − ρrri; ζi)) ri, (19a) ∇ˆ xgt(x, y; B¯) := d1 2¯bρs X ¯b i=1 (gt(x + ρssi, y; ζi) − gt(x − ρssi, y; ζi))si. (19b)
$$(19\mathrm{a})$$
$$(19\mathbf{b})$$
Similarly, we estimate ∇yft,ρ(x, y; B) and ∇xft,ρ(x, y; B), respectively, by ∇ˆ yft(x, y; B) := d2 2bρr X b i=1 (ft(x, y + ρrri; ξi) − ft(x, y − ρrri; ξi))ri, (20a) ∇ˆ xft(x, y; B) := d1 2bρs X b i=1 (ft(x + ρssi, y; ξi) − ft(x − ρssi, y; ξi))si. (20b)
$$(20\mathrm{a})$$
$$(20\mathrm{b})$$
Further, given a smoothing parameter ρv > 0, we can approximate the Hessian-vector product ∇2ygt,ρ(x, y)v and the Jacobian-vector product ∇2xygt,ρ(x, y)v as the finite difference between two gradients, respectively, as

∇ˆ 2ygt(x, y; B¯) := 1 2 ¯bρv X ¯b i=1 (∇ˆ ygt(x, y + ρvv; ζi) − ∇ˆ ygt(x, y − ρvv; ζi)), (21a) ∇ˆ 2xygt(x, y; B¯) := 1 2 ¯bρv X ¯b i=1 (∇ˆ xgt(x, y + ρvv; ζi) − ∇ˆ xgt(x, y − ρvv; ζi)). (21b)
$$(21\mathrm{a})$$
$$(21\mathbf{b})$$
Using (19)–(21), the first-order terms in (7) are approximated as dˆy t, dˆv t, and dˆx tin (22). The approximations in
(21a) and (21b) introduce errors in the hypergradient, which must be controlled. (21) depends on the dimension of y, as in ZO optimization (Nesterov & Spokoiny, 2017; Shamir, 2017). The projection ΠZpin (8) bounds v, controlling variance in v and x updates for convergence. Assumption 4.1. There exist constants σˆgy, σˆgx, σˆfy, σˆfx such that, for all z = [x, y], the following holds:

D1. E∥∇ˆygt(z; ζ) − ∇ygt,ρ(z)∥ 2 ≤ σˆ 2 gy , D2. E∥∇ˆxgt(z; ζ) − ∇xgt,ρ(z)∥ 2 ≤ σˆ 2 gx , D3. E∥∇ˆyft(z; ξ) − ∇yft,ρ(z)∥ 2 ≤ σˆ 2 fy , D4. E∥∇ˆxft(z; ξ) − ∇xft,ρ(z)∥ 2 ≤ σˆ 2 fx .
Assumption 4.1 is analogous to the upper bound on the variance of stochastic partial gradients discussed in Luo et al. (2020); Wang et al. (2020). We simplify the notation by introducing the following shorthand.

$$\hat{\sigma}^{2}:=\hat{\sigma}_{g_{\bf y}}^{2}+\hat{\sigma}_{g_{\bf x}}^{2}+\hat{\sigma}_{f_{\bf y}}^{2}+\hat{\sigma}_{f_{\bf x}}^{2}.\tag{23}$$

Next, we establish a regret bound for ZO-SOGD. Similar to

$${\hat{\mathbf{y}}}_{t}^{*}(\mathbf{x})\in{\underset{\mathbf{y}\in\mathbb{R}^{d_{2}}}{\operatorname{argmin}}}\left\{g_{t,\boldsymbol{\rho}}(\mathbf{x},\mathbf{y})\right\}$$
$$\begin{array}{l}\mbox{\bf E}\left[g_{t}({\bf x}+\rho_{\bf s}{\bf s},{\bf y}+\rho_{\bf r}{\bf r};\zeta)\right]\right\}.\end{array}\tag{18}$$

220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 Algorithm 2 ZO-SOGD Require: In addition to parameters in SOGD, choose ρv, ρr, ρs, ∈ R++.

For t = 1 to T do:
S1. Draw samples Bt and B¯t with batch sizes b and ¯b.

Using (19)–(21), get ZO search directions dˆy t, dˆv t, dˆx t:

$$\mathbf{d}_{t}^{\mathbf{y}}\left(\mathbf{z}_{t};{\vec{B}}_{t}\right)={\hat{\nabla}}_{\mathbf{y}}g_{t}(\mathbf{z}_{t};{\vec{B}}_{t}),$$

tzt; B¯t= ∇ˆ ygt(zt; B¯t), (22a)

$$\mathbf{i}_{t}^{y}\left(\mathbf{z}\right)$$

dˆy t = d y t(zt; B¯t) + (1 − γt)(dˆy t−1 − d y t(zt−1; B¯t)),
d vv t (zt; Bt) = ∇ˆ yft (zt; Bt) + ∇ˆ 2ygtzt; B¯t, (22b)
dˆv t = d vv t (zt; Bt) + (1 − λt)(dˆv t−1 − d vv t (zt−1; Bt)),
d xy t(zt; Bt) = ∇ˆ xft (zt; Bt) + ∇ˆ 2xygtzt; B¯t, (22c)
dˆx t = d xy t(zt; Bt) + (1 − ηt)(dˆx t−1 − d xy t(zt−1; Bt)),
S2. Update inner, system, and outer solutions:

$\mathbf{V}_{t+1}=\mathbf{V}_{t}-\beta_{t}\mathbf{d}_{t}^{\mathbf{Y}},\quad\mathbf{v}_{t+1}=\Pi_{\mathcal{Z}_{p}}\left[\mathbf{v}_{t}-\delta_{t}\mathbf{d}_{t}^{\mathbf{Y}}\right]$, $\mathbf{x}_{t+1}=\Pi_{\mathcal{X}}\left[\mathbf{x}_{t}-\alpha_{t}\mathbf{d}_{t}^{\mathbf{X}}\right]$.  
previous results, we introduce regularity conditions for the smoothed functions in (17) and (18). Inner and Outer Perturbed Gradient Variations: We define the gradient variations at the perturbed point as follows:
275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329

$$G_{\mathbf{v},T}:=\sum_{t=2}^{T}(\chi_{1t}+\chi_{2t}),\ \ G_{\mathbf{x},T}:=\sum_{t=2}^{T}(\chi_{3t}+\chi_{4t}).\tag{24}$$

where z
+
t:= (xt−1, yt−1+ρvvt−1), z
$-1$), $\mathbf{z}_t^{-}:=(\mathbf{x}_{t-1},\mathbf{y}_{t-1})$
ρvvt−1), and
$$\begin{array}{l}{{\chi_{1t}:=\|\nabla_{\mathbf{y}}g_{t}(\mathbf{z}_{t}^{+})-\nabla_{\mathbf{y}}g_{t-1}(\mathbf{z}_{t}^{+})\|^{2},}}\\ {{\chi_{2t}:=\|\nabla_{\mathbf{y}}g_{t}(\mathbf{z}_{t}^{-})-\nabla_{\mathbf{y}}g_{t-1}(\mathbf{z}_{t}^{-})\|^{2},}}\end{array}$$
$\chi_{3t}:=\|\nabla_{\bf x}g_{t}({\bf z}_{t}^{+})-\nabla_{\bf x}g_{t-1}({\bf z}_{t}^{+})\|^{2}$, $\chi_{4t}:=\|\nabla_{\bf x}g_{t}({\bf z}_{t}^{-})-\nabla_{\bf x}g_{t-1}({\bf z}_{t}^{-})\|^{2}$.  
Further, for simplicity of notation, we define

$$\begin{array}{l}{{\hat{\Delta}_{T}:=E_{1}+V_{T}+D_{T}+G_{\mathbf{y},T},}}\\ {{\hat{\Psi}_{T}:=H_{2,T}+G_{\mathbf{v},T}+G_{\mathbf{x},T},}}\end{array}$$
$\eqref{eq:walpha}$. 
where (VT , Hp,T ) and (E1, DT ) are defined in (10), and
(14), repectively. Moreover, Gy,T and (Gv,T , Gx,T ), are defined in (12) and (24), respectively.

Theorem 4.2. Let {(ft, gt)}
T
t=1 be the sequence of functions presented to Algorithm 2, satisfying Assumptions 3.23.4 and 4.1. For all t ∈ [T]*, let*

αt =1
(d1 + d2)
$${\overline{{\frac{1}{3/4}(c+t)^{1/3}}}},$$
, βt = cβαt, δt = cδαt,
γt+1 = cγαt, ηt+1 = cηαt, λt+1 = cλαt,
ρ 2v = cvαt, ρ 2 r =1 d 22T , ρ 2 s =1 d 21T , b =T 1/3 (d1 + d2) 3/2 ,¯b =T 2/3 (d1 + d2) 3/4
, (26)
where c, cβ, cδ, cγ, cη, cv, and cλ *are specified in* (232)*. Let* p = ℓf,0/µg for the set Zp *defined in* (8). Then, Algorithm 2 *guarantees:*

$$\mathrm{BL-Reg}_{T}\leq{\mathcal{O}}\left((d_{1}+d_{2})^{3/4}T^{1/3}\left({\hat{\sigma}}^{2}+{\hat{\Delta}}_{T}\right)\right)$$
$$(26)^{\frac{1}{2}}$$
$$+(d_{1}+d_{2})^{3/2}T^{2/3}\hat{\Psi}_{T}\Big)$$

where σˆ
2 and (∆ˆT , ΨˆT ) *are defined in* (23) and (25).

Theorem 4.2 bounds the regret of Algorithm 2 without window-smoothing, based on the regularities in (25). We note that the average dynamic regret BL-RegT
/T ≤
O((d1+d2)
3/4T
−2/3σˆ
2 + ∆ˆT
+(d1+d2)
3/2T
−1/3ΨˆT )
remains sublinear under suitable conditions on ∆ˆT and ΨˆT .

Remark 4.3 (**Regret Guarantee for Zeroth Order OBO**).

Theorem 4.2 provides the first regret guarantee for OBO
with access only to noisy function evaluations of the leader and follower. The dimensional dependence O(d1 + d2) in Theorem 4.2 aligns with optimal results for simpler offline min-max problems (Huang et al., 2022). The bound also depends on the sample sizes b, ¯b and smoothing parameters ρv, ρr, ρs at each iteration.

Remark 4.4 (**Improved Regret for OSO**). Our dynamic regret for single-level non-stationary optimization is O((d1 + d2)
3/4T
−2/3(ˆσ 2 + E1 + VT + DT )), improving the result in Roy et al. (2022), which is O(T
−1/2σ 2
√d). Roy et al. (2022) proposed a zeroth-order stochastic gradient descent algorithm for unconstrained, non-convex, timevarying objective functions, achieving a regret bound of O(T
−1/2σ 2√dWT ) using a two-point gradient estimator, where WT bounds the nonstationarity. Additionally, Guan et al. (2023a) showed that the local regret for standard online stochastic gradient descent with the standard two-point gradient estimator (Agarwal et al., 2010) is O(T
−1/2d
√VT ).

## 5. Experimental Results

In this section, we provide experimental results on bilevel optimization-based black-box attacks on deep neural networks and parametric loss tuning for imbalanced data.

## 5.1. Bilevel Optimization-Based Black-Box Attacks

Deep neural network classifiers are vulnerable to adversarial examples—images subtly modified to mislead the classifier. These examples can deceive classifiers even without knowledge of the model, as seen in black-box adversarial attacks
(BBAA) (Chen et al., 2017; Liu et al., 2018b; Chen et al.,

$\quad2019$)... 
330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 We first review the ZO single-level optimization for BBAA
(Chen et al., 2017). Let (a, b) denote a legitimate image a ∈ R
d with true label b ∈ {1, 2*, . . . , J*}, where J is the total number of classes. Define a
′ = a+y as an adversarial example, with y as the adversarial perturbation. Let Y :=
[−5, 5]d ⊂ R
d, and ℓ : R
d → R denote the black-box attack loss. The goal of BBAA (Chen et al., 2017) is to design y for images {ai}
m i=1 by solving:
and

where
$$g(\mathbf{x}_{t},\mathbf{y}_{t};\mathcal{D}_{t}^{\mathbb{F}})=\frac{1}{|\mathcal{D}_{t}^{\mathbb{F}}|}\sum_{i\in\mathcal{D}_{t}^{\mathbb{F}}}\ell(\mathbf{a}_{t}^{(i)}+\mathbf{y}_{t})$$ $$+\frac{1}{2}\sum_{\iota=1}^{p}e^{[\mathbf{x}_{t}]_{\iota}}[\mathbf{y}_{t}]_{\iota}^{2},\tag{29a}$$
$$f({\bf y}_{t}({\bf x}_{t});{\cal D}_{t}^{\rm val})=\frac{1}{|{\cal D}_{t}^{\rm val}|}\sum_{i\in{\cal D}_{t}^{\rm val}}\ell({\bf a}_{t}^{(i)}+{\bf y}_{t}).\tag{29b}$$
Here, {a
(i)
t }i∈Dtrt and {a
(i)
t }i∈Dval tare batches of training and validation samples at timestep t; a
(i)
tis the ith sample in that batch; and [xt]ι and [yt]ι denote the ιth component of xt and yt, respectively. We normalize the pixel values to Y. For an untargeted attack, the loss in (29) is ℓ(a
′t
) = max{Z(a
′t
)bt −
maxj̸=bt Z(a
′t)j , −κ}, where Z(a
′t)j is the prediction score for class j given input a
′
t = at + yt, and κ > 0 controls the confidence gap. In our experiments, we set κ = 0.

Eq. (28) introduces the first OBO formulation of BBAA.

Using a vector x ∈ R
d+ for hyperparameters instead of λ ∈ R++ in (27) enables finer control over model components, enhancing performance for complex models and heterogeneous data (Lorraine et al., 2020). For a fair comparison with single-level BBAA, we replace λ with a fixed

$$\min_{\mathbf{y}\in\mathcal{Y}}\quad\frac{1}{m}\sum_{i=1}^{m}\ell(\mathbf{a}_{i}+\mathbf{y})+\lambda\|\mathbf{y}\|^{2}.\tag{27}$$

Here, λ > 0 is a hyperparameter balancing attack loss minimization and ℓ2 regularization. To adapt (27) to our OBO, consider OBO for supervised learning: at each timestep t, new samples (at, bt) ∈ Dt := {Dval t, Dtr t } are received, where at ∈ R
d2is the feature vector (image) and bt ∈ R is the corresponding target. Note that the correct decision can change abruptly. We consider an S-stage scenario where (x
∗s, y
∗s(x
∗
s)) represents the best decisions for the s-th stage, for all s ∈ [S].

for the $s$-th stage, for all $s\in[\mathcal{D}]$.  $$\mathbf{x}_{s}^{*}\in\operatorname*{argmin}_{\mathbf{x}\in\mathcal{X}}\sum_{t=1}^{T_{x}}f\left(\mathbf{y}_{s}^{*}(\mathbf{x});\mathcal{D}_{t}^{\mathrm{val}}\right)$$  s.t. $$\mathbf{y}_{s}^{*}(\mathbf{x})\in\operatorname*{argmin}_{\mathbf{y}\in\mathcal{Y}}\sum_{t=1}^{T_{x}}g\left(\mathbf{x},\mathbf{y};\mathcal{D}_{t}^{\mathrm{ir}}\right),$$ (28)
vector multiplied by each component of y in (27). We compare our ZO-SOGD and ZO-SOGD (Adam) with the following competing methods in the online setting:
ZO-O-GD: A single-level method that updates yt with a fixed x at each timestep using ZO gradient descent (Nesterov & Spokoiny, 2017).

ZO-O-Adam: A single-level method that updates yt with a fixed x at each timestep using ZO Adam (Kingma & Ba, 2014; Chen et al., 2019).

ZO-O-SignSGD: A single-level method that updates yt with a fixed x at each timestep using ZO SignSGD (Bernstein et al., 2018). ZO-O-ConservSGD: A single-level method that updates yt with a fixed x at each timestep using ZO Conservative SGD (Cutkosky & Boahen, 2019). Note that ZO-SOGD (Adam) is a variant of our algorithm with an adaptive stepsize, similar to that of (Kingma & Ba, 2014). We evaluated the proposed algorithms based on runtime, test accuracy on perturbed samples, and the infinity norm of yt. Figure 2 compares the methods. The left panel shows that ZO-SOGD has similar runtime to single-level baselines, despite outer-level optimization on x. The middle panel shows that all methods' accuracy decreases as the adversarial attack y strengthens, with ZO-SOGD outperforming ZO-O-GD and ZO-O-ConservGD, and ZO-SOGD (Adam) outperforming ZO-O-Adam and all baselines. The right panel shows that the increasing infinity norm of yt over time for all methods, which reduces accuracy. However, the perturbations remain unnoticeable with a max yt no larger than 4, demonstrating that ZO-SOGD achieves effective attacks with better performance than other methods.

## 5.2. Parametric Loss Tuning For Imbalanced Data

Imbalanced datasets are common in modern machine learning, causing challenges in generalization and fairness due to underrepresented classes and sensitive attributes. Deep NNs often overfit, seeming accurate and fair during training but performing poorly during testing. A common solution is designing a parametric training loss that balances accuracy and fairness while preventing overfitting (Li et al., 2021).

We consider an optimization problem similar to (28). For a new sample (at, bt), the follower and leader incur a parametric and balanced cross-entropy loss, respectively:

$$\begin{array}{c}{{g({\bf x}_{t},{\bf y}_{t};{\cal D}_{t}^{\mathrm{dr}})=-\log\frac{e^{\gamma_{b_{t}}}[{\bf y}_{t}({\bf a}_{t})]_{b_{t}}+\Delta_{b_{t}}}{\sum_{j=1}^{J}e^{\gamma_{j}}[{\bf y}_{t}({\bf a}_{t})]_{j}+\Delta_{j}},}}\\ {{f({\bf y}_{t}({\bf x}_{t});{\cal D}_{t}^{\mathrm{val}})=-u_{b_{t}}\log\frac{e^{[{\bf y}_{t}({\bf a}_{t})]_{b_{t}}}}{\sum_{j=1}^{J}e^{[{\bf y}_{t}({\bf a}_{t})]_{j}}}.}}\end{array}$$
, and
. (30)
385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 number of samples (Li et al., 2021).

To clarify the notation in (30): yt(xt) denotes the follower yt conditioned on the leader xt, while [yt(at)]btrepresents the predicted logit for class bt on sample at. The backbone model for yt is a 4-layer CNN, leading to a nonconvex bilevel objective. We compare SOGD with the following methods:
OAGD (Tarzanagh et al., 2024): A state-of-the-art static online bilevel gradient descent method using the Neumann series for hypergradient approximation. SOBOW (Lin et al., 2024): A dynamic online bilevel gradient descent method using conjugate gradients (CG) for hypergradient approximation. We conducted experiments on the MNIST (LeCun et al., 2010). We used a batch size of 64 per timestep. We evaluated cumulative runtime, balanced accuracy, and test accuracy, where balanced accuracy is the class-specific average accuracy:

$${\frac{1}{J}}\sum_{j=1}^{J}\mathbb{P}_{{\mathbf{a}}_{t}\sim{\mathcal{D}}_{j}}\left[\operatorname{argmax}_{i}([{\mathbf{y}}_{t}({\mathbf{a}}_{t})]_{i})=j\right],$$

with Dj denoting the distribution over samples of class j (Li et al., 2021). Learning rates were tuned as βt = δt = β ∈ {0.001, 0.005, 0.01, 0.05, 0.1} and αt = α ∈ {0.0001, 0.0005, 0.001, 0.005, 0.01} for all t ∈ [T]. The parameters γt, λt, ηt were tuned as γt = λt = ηt = γ ∈
{0.9, 0.99, 0.999}. The Neumann series iterations in OAGD and CG iterations in SOBOW were set to 5. We evaluated performance over 400 timesteps in four 100timestep phases, transitioning from an imbalanced (0.4 i) to a balanced (0.8 i) distribution for each class (i = 0, 1*, . . . ,* 9).

Figure 3 (left) shows SOBOW's longer runtime due to CG complexity, while SOGD is the fastest with simultaneous updates. Figures 3 (middle, right) show accuracy gains as balance increases, with SOGD achieving competitive accuracy.

## 6. Conclusion

We introduced a novel online bilevel optimization (OBO) framework that overcomes the limitations of existing algorithms, which often rely on extensive oracle information and incur high computational costs. Our approach uses limited feedback and zeroth-order updates for efficient hypergradient estimation and simultaneous updates of decision variables, achieving sublinear bilevel regret without window smoothing. Experiments on online parametric loss tuning and black-box adversarial attacks confirm its effectiveness.

$$\mathrm{and}$$
$$(30)^{\frac{1}{2}}$$

## Impact Statements

This paper develops methods to advance online learning. While our work has societal implications, none require specific emphasis here.

Here, xt := (∆j , γj )
J
j=1 represents the logits adjustments, with j indexing the J classes, and uj is the reciprocal of the proportion of samples from the j-th class to the total

## References

Agarwal, A., Dekel, O., and Xiao, L. Optimal algorithms for online convex optimization with multi-point bandit feedback. In *Colt*, pp. 28–40. Citeseer, 2010.

Agarwal, N., Gonen, A., and Hazan, E. Learning in nonconvex games with an optimization oracle. In *Conference* on Learning Theory, pp. 18–29. PMLR, 2019.

Aghasi, A. and Ghadimi, S. Fully zeroth-order bilevel programming via gaussian smoothing. arXiv preprint arXiv:2404.00158, 2024.

Allen-Zhu, Z. and Li, Y. Neon2: Finding local minima via first-order oracles. *Advances in Neural Information* Processing Systems, 31, 2018.

Bach, F. and Perchet, V. Highly-smooth zero-th order online optimization. In *Conference on Learning Theory*, pp. 257–283. PMLR, 2016.

Bernstein, J., Wang, Y.-X., Azizzadenesheli, K., and Anandkumar, A. Signsgd: Compressed optimisation for nonconvex problems. In International Conference on Machine Learning, pp. 560–569. PMLR, 2018.

Besbes, O., Gur, Y., and Zeevi, A. Non-stationary stochastic optimization. *Operations research*, 63(5):1227–1244, 2015.

Bohne, J., Rosenberg, D., Kazantsev, G., and Polak, P. Online nonconvex bilevel optimization with bregman divergences. *arXiv preprint arXiv:2409.10470*, 2024.

Bracken, J. and McGill, J. T. Mathematical programs with optimization problems in the constraints. Operations Research, 21(1):37–44, 1973.

Bubeck, S., Stoltz, G., Szepesvari, C., and Munos, R. Online ´
optimization in x-armed bandits. *Advances in Neural* Information Processing Systems, 21, 2008.

Chen, P.-Y., Zhang, H., Sharma, Y., Yi, J., and Hsieh, C.-
J. Zoo: Zeroth order optimization based black-box attacks to deep neural networks without training substitute models. In Proceedings of the 10th ACM workshop on artificial intelligence and security, pp. 15–26, 2017.

Chen, T., Sun, Y., and Yin, W. Closing the gap: Tighter analysis of alternating stochastic gradient methods for bilevel problems. *Advances in Neural Information Processing* Systems, 34, 2021.

Chen, X., Liu, S., Xu, K., Li, X., Lin, X., Hong, M., and Cox, D. Zo-adamm: Zeroth-order adaptive momentum method for black-box optimization. Advances in neural information processing systems, 32, 2019.

Crockett, C., Fessler, J. A., et al. Bilevel methods for image reconstruction. Foundations and Trends® *in Signal* Processing, 15(2-3):121–289, 2022.

Cutkosky, A. and Boahen, K. Anytime online-to-batch conversions and the conservative algorithm. Advances in Neural Information Processing Systems, 32, 2019.

Dagreou, M., Ablin, P., Vaiter, S., and Moreau, T. A frame- ´
work for bilevel optimization that enables stochastic and global variance reduction algorithms. *arXiv preprint* arXiv:2201.13409, 2022.

Dempe, S. *Foundations of bilevel programming*. Springer Science & Business Media, 2002.

Duchi, J. C., Jordan, M. I., Wainwright, M. J., and Wibisono, A. Optimal rates for zero-order convex optimization: The power of two function evaluations. *IEEE Transactions* on Information Theory, 61(5):2788–2806, 2015.

Finn, C., Abbeel, P., and Levine, S. Model-agnostic metalearning for fast adaptation of deep networks. In International Conference on Machine Learning, pp. 1126–1135. PMLR, 2017.

Finn, C., Rajeswaran, A., Kakade, S., and Levine, S. Online meta-learning. In International Conference on Machine Learning, pp. 1920–1930. PMLR, 2019.

Flaxman, A. D., Kalai, A. T., and McMahan, H. B. Online convex optimization in the bandit setting: gradient descent without a gradient. *arXiv preprint cs/0408007*, 2004.

Franceschi, L., Donini, M., Frasconi, P., and Pontil, M.

Forward and reverse gradient-based hyperparameter optimization. In International Conference on Machine Learning, pp. 1165–1173. PMLR, 2017.

Franceschi, L., Frasconi, P., Salzo, S., Grazzi, R., and Pontil, M. Bilevel programming for hyperparameter optimization and meta-learning. In International Conference on Machine Learning, pp. 1568–1577. PMLR, 2018.

Gao, X., Li, X., and Zhang, S. Online learning with nonconvex losses and non-stationary regret. In *International* Conference on Artificial Intelligence and Statistics, pp. 235–243. PMLR, 2018.

Ghadimi, S. and Lan, G. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. *SIAM*
journal on optimization, 23(4):2341–2368, 2013.

Ghadimi, S. and Wang, M. Approximation methods for bilevel programming. *arXiv preprint arXiv:1802.02246*,
2018.

Ghadimi, S., Lan, G., and Zhang, H. Mini-batch stochastic approximation methods for nonconvex stochastic composite optimization. *Mathematical Programming*, 155 (1-2):267–305, 2016.

Goel, G., Lin, Y., Sun, H., and Wierman, A. Beyond online balanced descent: An optimal algorithm for smoothed online optimization. Advances in Neural Information Processing Systems, 32, 2019.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Ji, K., Yang, J., and Liang, Y. Bilevel optimization: Convergence analysis and enhanced design. In International Conference on Machine Learning, pp. 4882–4892. PMLR, 2021.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2014.

Kleinberg, R., Slivkins, A., and Upfal, E. Multi-armed bandits in metric spaces. In Proceedings of the fortieth annual ACM symposium on Theory of computing, pp.

681–690, 2008.

Guan, Z., Zhou, Y., and Liang, Y. On the hardness of online nonconvex optimization with single oracle feedback. In The Twelfth International Conference on Learning Representations, 2023a.

Guan, Z., Zhou, Y., and Liang, Y. Online nonconvex optimization with limited instantaneous oracle feedback. In The Thirty Sixth Annual Conference on Learning Theory, pp. 3328–3355. PMLR, 2023b.

Hallak, N., Mertikopoulos, P., and Cevher, V. Regret minimization in stochastic non-convex learning via a proximal-gradient approach. In International Conference on Machine Learning, pp. 4008–4017. PMLR, 2021.

Hansen, P., Jaumard, B., and Savard, G. New branch-andbound rules for linear bilevel programming. SIAM Journal on scientific and Statistical Computing, 13(5):1194–
1217, 1992.

Krichene, W., Balandat, M., Tomlin, C., and Bayen, A.

The hedge algorithm on a continuum. In International Conference on Machine Learning, pp. 824–832. PMLR, 2015.

LeCun, Y., Cortes, C., and Burges, C. Mnist handwritten digit database. *ATT Labs [Online]. Available:* http://yann.lecun.com/exdb/mnist, 2, 2010.

Hazan, E. Introduction to online convex optimization. Foundations and Trends® *in Optimization*, 2(3-4):157–325, 2016a. URL http://ocobook.cs.princeton.

edu/OCObook.pdf.

Li, M., Zhang, X., Thrampoulidis, C., Chen, J., and Oymak, S. Autobalance: Optimized loss functions for imbalanced data. *Advances in Neural Information Processing* Systems, 34:3163–3177, 2021.

Lin, S., Sow, D., Ji, K., Liang, Y., and Shroff, N. Nonconvex bilevel optimization with time-varying objective functions. Advances in Neural Information Processing Systems, 36, 2024.

Liu, H., Simonyan, K., and Yang, Y. Darts: Differentiable architecture search. *arXiv preprint arXiv:1806.09055*, 2018a.

Liu, S., Chen, J., Chen, P.-Y., and Hero, A. Zeroth-order online alternating direction method of multipliers: Convergence analysis and applications. In International Conference on Artificial Intelligence and Statistics, pp. 288–297. PMLR, 2018b.

Hazan, E. Introduction to online convex optimization.

Foundations and Trends in Optimization, 2(3-4):157–325, 2016b.

Hazan, E., Agarwal, A., and Kale, S. Logarithmic regret algorithms for online convex optimization. Machine Learning, 69(2):169–192, 2007.

Hazan, E., Singh, K., and Zhang, C. Efficient regret minimization in non-convex games. In International Conference on Machine Learning, pp. 1433–1441. PMLR,
2017.

Heliou, A., Martin, M., Mertikopoulos, P., and Rahier, T. ´
Online non-convex optimization with imperfect feedback. Advances in Neural Information Processing Systems, 33: 17224–17235, 2020.

Heliou, A., Martin, M., Mertikopoulos, P., and Rahier, ´
T. Zeroth-order non-convex learning via hierarchical dual averaging. In *International Conference on Machine* Learning, pp. 4192–4202. PMLR, 2021.

Lorraine, J., Vicol, P., and Duvenaud, D. Optimizing millions of hyperparameters by implicit differentiation. In International conference on artificial intelligence and statistics, pp. 1540–1552. PMLR, 2020.

Huang, F., Gao, S., Pei, J., and Huang, H. Accelerated zeroth-order and first-order momentum methods from mini to minimax optimization. Journal of Machine Learning Research, 23(36):1–70, 2022.

Luo, L., Ye, H., Huang, Z., and Zhang, T. Stochastic recursive gradient descent ascent for stochastic nonconvexstrongly-concave minimax problems. Advances in Neural Information Processing Systems, 33:20566–20577, 2020.

Huang, Y., Cheng, Y., Liang, Y., and Huang, L. Online minmax problems with non-convexity and non-stationarity. Transactions on Machine Learning Research.

Lv, Y., Hu, T., Wang, G., and Wan, Z. A penalty function method based on kuhn–tucker condition for solving linear bilevel programming. Applied Mathematics and Computation, 188(1):808–813, 2007.

Nesterov, Y. Smooth minimization of non-smooth functions.

Mathematical programming, 103:127–152, 2005.

Huang, Y., Cheng, Y., Liang, Y., and Huang, L. Online minmax problems with non-convexity and non-stationarity.

Transactions on Machine Learning Research, 2023.

Ji, K., Wang, Z., Zhou, Y., and Liang, Y. Improved zerothorder variance reduced algorithms and analysis for nonconvex optimization. In International conference on machine learning, pp. 3100–3109. PMLR, 2019.

Nesterov, Y. and Spokoiny, V. Random gradient-free minimization of convex functions. Foundations of Computational Mathematics, 17(2):527–566, 2017.

Roy, A., Balasubramanian, K., Ghadimi, S., and Mohapatra, P. Stochastic zeroth-order optimization under nonstationarity and nonconvexity. *Journal of Machine Learning* Research, 23(64):1–47, 2022.

Shalev-Shwartz, S. et al. Online learning and online convex optimization. Foundations and trends in Machine Learning, 4(2):107–194, 2011.

Shamir, O. An optimal algorithm for bandit and zero-order convex optimization with two-point feedback. *Journal of* Machine Learning Research, 18(52):1–11, 2017.

Sow, D., Ji, K., and Liang, Y. On the convergence theory for hessian-free bilevel algorithms. *Advances in Neural* Information Processing Systems, 35:4136–4149, 2022.

Stackelberg, H. v. Theory of the market economy. Oxford University Press, 1952.

Stadie, B., Zhang, L., and Ba, J. Learning intrinsic rewards as a bi-level optimization problem. In Conference on Uncertainty in Artificial Intelligence, pp. 111–120. PMLR, 2020.

Suggala, A. S. and Netrapalli, P. Online non-convex learning: Following the perturbed leader is optimal. In Algorithmic Learning Theory, pp. 845–861. PMLR, 2020.

Tarzanagh, D. A., Nazari, P., Hou, B., Shen, L., and Balzano, L. Online bilevel optimization: Regret analysis of online alternating gradient methods. In International Conference on Artificial Intelligence and Statistics, pp. 2854–2862. PMLR, 2024.

Wang, Z., Balasubramanian, K., Ma, S., and Razaviyayn, M. Zeroth-order algorithms for nonconvex minimax problems with improved complexities. *arXiv preprint* arXiv:2001.07819, 2020.

Zhang, Y., Zhou, Y., Ji, K., and Zavlanos, M. M. Boosting one-point derivative-free online optimization via residual feedback. *arXiv preprint arXiv:2010.07378*, 2020.

Zhou, W., Li, Y., Yang, Y., Wang, H., and Hospedales, T. Online meta-critic learning for off-policy actor-critic methods. Advances in Neural Information Processing Systems, 33:17662–17673, 2020.

Zinkevich, M. Online convex programming and generalized infinitesimal gradient ascent. In *Proceedings of the 20th* international conference on machine learning (icml-03), pp. 928–936, 2003.

550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604

## A. Related Work

BO was introduced in game theory by (Stackelberg, 1952) and modeled mathematically in (Bracken & McGill, 1973). Initial works (Hansen et al., 1992; Lv et al., 2007) reduced it to single-level optimization. Recently, gradient-based approaches have gained popularity for their simplicity and efficacy (Franceschi et al., 2017; Ghadimi & Wang, 2018; Ji et al., 2021; Chen et al., 2021), though they assume offline objectives.

OBO was initiated by Tarzanagh et al. (2024), proposing the OAGD method with regret bounds. (Huang et al., 2023)
developed algorithms for online minimax optimization, special cases of OBO with local regret guarantees. (Lin et al., 2024) introduced SOBOW, a single-loop optimizer using window-smoothed functions and multiple CGs for nonconvexstrongly-convex cases. Unlike these works, we propose using *projected gradient* as a more general performance measure for constrained objectives, focusing on the original functions and their regret; See Table 1 for a comparison. Single-Level Regret Minimization. Single-level online optimization predominantly focuses on convex problems, either with static or dynamic convex regret minimization (Zinkevich, 2003; Hazan, 2016a; Shalev-Shwartz et al., 2011). Nonconvex online optimization (Hazan et al., 2017; Guan et al., 2023b;a) poses greater challenges than its convex counterparts (Shalev-Shwartz et al., 2011; Zinkevich, 2003; Hazan et al., 2007; Besbes et al., 2015). Notable contributions in this field include adversarial multi-armed bandit algorithms (Bubeck et al., 2008; Heliou et al. ´ , 2020; 2021; Krichene et al., 2015) and the Follow-the-Perturbed-Leader approach (Agarwal et al., 2019; Kleinberg et al., 2008; Suggala & Netrapalli, 2020). Hazan et al. (Hazan et al., 2017) introduced window-smoothed local regret for gradient averaging in non-convex models, which Hallak et al. (Hallak et al., 2021) extended to non-smooth, non-convex problems. Inspired by their work, we employ local regret for Online Bandit Optimization (OBO) without window-smoothing.

Zeroth-Order Optimization. Single-Level ZO Optimization has been widely studied in both offline (Ghadimi & Lan, 2013; Duchi et al., 2015; Agarwal et al., 2010; Nesterov & Spokoiny, 2017) and online settings (Liu et al., 2018b; Guan et al., 2023a;b; Zhang et al., 2020; Bach & Perchet, 2016). We next review closely related work. Liu et al. (Liu et al., 2018b)
proposed ZOO-ADMM, a gradient-free online optimization algorithm utilizing ADMM. Guan et al. (Guan et al., 2023b)
studied online non-convex optimization with limited oracle feedback. Research on online non-convex optimization with bandit feedback includes work by Heliou et al. (Heliou et al. ´ , 2020), which established bounds on global static and dynamic regret using dual averaging, further refined in (Heliou et al. ´ , 2021). Gao et al. (Gao et al., 2018) extended these ideas to ZO
algorithms. Flaxman et al. (Flaxman et al., 2004) provided algorithms for bandit online optimization of convex functions using ZO gradient approximation. Our work closely relates to (Sow et al., 2022), which proposes a Hessian-free method approximating the Jacobian matrix using a ZO method based on finite differences of gradients. In contrast, our method uses function oracles to approximate both the Hessian and gradients and is derivative-free. We also point out the recent work (Aghasi & Ghadimi, 2024) on ZO stochastic algorithms for solving bilevel problems when neither the upper/lower objective values nor their unbiased gradient estimates are available. Their approach, limited to the *offline* setting, does not include numerical results, thus leaving its practical efficiency unclear.

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659

## B. Additional Preliminaries And Notations

B.1. Preliminary Lemmas We first provide several useful lemmas for the main proofs. where Definition B.1 (**Projected gradient** (Ghadimi et al., 2016)). Let X ⊂ R
d1 be a closed convex set. Then, the projected gradient for any αt > 0 and p ∈ R
d1is defined as

$${\mathcal{P}}_{\chi,\alpha_{t}}\left(\mathbf{x};\mathbf{p}\right):={\frac{1}{\alpha_{t}}}\left(\mathbf{x}-\mathbf{x}^{+}\right),$$
+, (31a)
$$(31\mathrm{a})$$
$$\mathbf{x}^{+}=\Pi_{\mathcal{X}}\left(\mathbf{x}-\alpha_{t}\mathbf{p}\right),$$
+ = ΠX (x − αtp), (31b)
and ΠX [·] denotes the orthogonal projection operator onto set X .

Lemma B.2. Goel et al. (2019, Lemma 13) If f : X → R is a µf -strongly convex function with respect to some norm *∥ · ∥*,
and x
∗is the minimizer of f *(i.e.* x
∗ = arg minx∈X f(x)), then we have ∀ x ∈ X ,

$${\frac{\mu_{f}}{2}}\|\mathbf{x}-\mathbf{x}^{*}\|^{2}\leq f(\mathbf{x})-f(\mathbf{x}^{*})\leq{\frac{1}{2\mu_{f}}}\|\nabla f(\mathbf{x})\|^{2}.$$

Lemma B.3. Suppose f(x) is L*-smooth, and* x
∗ ∈ argminx∈X f(x). Then, we can upper bound the magnitude of the

$$(311\mathbf{b})$$

gradient at any given point x ∈ R
din terms of the objective sub optimality at x*, as follows:*
Lemma B.4. *For any set of vectors* {xi}
m i=1 *with* xi ∈ R
d*, we have*

$$\left\|\sum_{i=1}^{m}{\mathbf{x}}_{i}\right\|^{2}\leq m\sum_{i=1}^{m}\left\|{\mathbf{x}}_{i}\right\|^{2}.$$
$$(32)$$
$$(33)$$

$$(34)$$

Lemma B.5. *For any* x, y ∈ R
d, the following holds for any c > 0:

$$\left\|\mathbf{x}+\mathbf{y}\right\|^{2}\leq(1+c)\|\mathbf{x}\|^{2}+\left(1+{\frac{1}{c}}\right)\|\mathbf{y}\|^{2},\ {\mathrm{~and~}}$$
2, and (33)
$$\|\mathbf{x}-\mathbf{y}\|^{2}\geq(1-c)\,\|\mathbf{x}-\mathbf{z}\|^{2}+\left(1-{\frac{1}{2}}\right)\,\|\mathbf{z}-\mathbf{y}\|^{2}\,.$$
c
2. (34)
660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 We provide a set of auxiliary lemmas that will be used in establishing the proofs for the main theorems.

Lemma B.6. Ghadimi et al. (2016, Proposition 1) Let PX ,αt
(x; p) be defined in Definition B.1. Then, for any p1 and p2 in R

d*, we*
∥PX ,αt
(x; p1) − PX ,αt
(x; p2)*∥ ≤ ∥*p1 − p2∥ .

Lemma B.7. Hazan et al. (2017, Proposition 2.4) Let PX ,αt(x; p) be the projected gradient as per Definition *B.1. For any* x, p1, p2 ∈ R
d and αt > 0 *it holds that*

$\square$
∥PX ,αt(x; p1 + p2)*∥ ≤ ∥P*X ,αt(x; p1)∥ + ∥p2∥ .

$$\langle\mathbf{p},{\mathcal{P}}_{{\mathcal{X}},\alpha_{t}}(\mathbf{x};\mathbf{p})\rangle\geq\left\|{\mathcal{P}}_{{\mathcal{X}},\alpha_{t}}(\mathbf{x};\mathbf{p})\right\|^{2}.$$

Proof. By the definition of x
+, the optimality condition of (31b) is which can be rearranged to B.2. Examples Theorem 3.6 achieves sublinear bilevel regret when the variations VT and H2,T are both o(T). Below, we provide some examples of online optimization in both single-level and bilevel settings to illustrate when this occurs.

Example B.9. Consider function ft(x) = ∥Atx − bt∥
2, where At = [1, 0; 0, 1 + 1 t
], x = bt = (1, 1). Then, VT :=
13

$$\left\langle\mathbf{p}+{\frac{1}{\alpha_{t}}}(\mathbf{x}^{+}-\mathbf{x}),\mathbf{z}-\mathbf{x}^{+}\right\rangle\geq0,\quad\forall\mathbf{z}\in{\mathcal{X}}.$$
$$\left\langle\mathbf{p},\mathbf{x}-\mathbf{x}^{+}\right\rangle\geq{\frac{1}{\alpha_{t}}}\left\langle\mathbf{x}-\mathbf{x}^{+},\mathbf{x}-\mathbf{x}^{+}\right\rangle,$$
$$\langle\mathbf{p},\mathcal{P}_{\mathcal{X},\alpha_{t}}(\mathbf{x};\mathbf{p})\rangle=\frac{1}{\alpha_{t}}\left\langle\mathbf{p},\mathbf{x}-\mathbf{x}^{+}\right\rangle\geq\frac{1}{\alpha_{t}^{2}}\left\langle\mathbf{x}-\mathbf{x}^{+},\mathbf{x}-\mathbf{x}^{+}\right\rangle$$ $$=\left\|\mathcal{P}_{\mathcal{X},\alpha_{t}}(\mathbf{x};\mathbf{p})\right\|^{2}.$$
$${\frac{1}{2L}}\|\nabla f(\mathbf{x})\|^{2}\leq f(\mathbf{x})-f(\mathbf{x}^{*})\leq{\frac{L}{2}}\|\mathbf{x}-\mathbf{x}^{*}\|^{2}.$$
2. (32)
Lemma B.8. Let PX ,αt
(x; p) be as given in Definition *B.1. Then, for any* p ∈ R
d and αt > 0*, we have* Letting z = x, we obtain PT
t=2 maxx |ft(x) − ft−1(x)| =PT
t=2 |1t 2−
1 t−1 2|. By a 2 − b 2 = (a − b)(a + b), we have 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769

$$V_{T}=\sum_{t=2}^{T}|\left(\frac{1}{t}-\frac{1}{t-1}\right)-\left(\frac{1}{t}+\frac{1}{t-1}\right)|$$ $$=\sum_{t=2}^{T}|\left(\frac{t-1-t}{t(t-1)}\right)-\left(\frac{1}{t}+\frac{1}{t-1}\right)|$$ $$=\sum_{t=2}^{T}|\left(-\frac{1}{t(t-1)}\right)-\left(\frac{1}{t}+\frac{1}{t-1}\right)|$$ $$=\sum_{t=2}^{T}|\frac{1}{t(t-1)}||\frac{t-1+t}{t(t-1)}|$$
$$=\sum_{t=2}^{T}|{\frac{2}{t(t-1)^{2}}}|.$$

Then, VT ≤PT
t=2 2 t 3 ≈R T
2 2 t 3 dt =
1 4 
−
1 T 2 . As T → ∞, VT becomes bounded and approaches a constant value, indicating that VT grows slower than T itself. Example B.10. Let ft(x) = (−
1 T
, 0, 0, 0) if t is even, and ft(x) = (0, −
1 T
, 0, 0) if t is odd. Then, VT = 
PT
t=2 maxx |ft(x) − ft−1(x)| = O(1).

Example *B.11*. Let x ∈ X = [−1, 1] ⊂ R, y ∈ R, and consider a sequence of quadratic cost functions

$$\begin{array}{l}{{f_{t}(x,y)=\frac{1}{2}\left(x+2a_{t}^{(1)}\right)^{2}+\frac{1}{2}\left(y-a_{t}^{(2)}\right)^{2},}}\\ {{g_{t}(x,y)=\frac{1}{2}y^{2}-\left(x-a_{t}^{(2)}\right)y,}}\end{array}$$

where a
(1)
t = 1/t and a
(2)
t = 1/
√t for all t ∈ [T].

We have

$$y_{t}^{*}(x)=x-a_{t}^{(2)}.$$
$-1\left[\left(\mathcal{L}\right)\right]$
We have ft(*x, y*∗
t(x)) − ft−1(*x, y*∗
t−1(x))

=
1
2
x + 2a
(1)
t
2−
$$\left(x+2a_{t-1}^{(1)}\right)^{2}\right]+\frac{1}{2}\left[\left(y_{t}^{*}(x)-a_{t}^{(2)}\right)^{2}-\left(y_{t-1}^{*}(x)-a_{t-1}^{(2)}\right)^{2}\right]$$
$$+\frac{1}{2}\left[\left((x-a_{t}^{(2)})^{2}-2(x-a_{t}^{(2)})a_{t}^{(2)}+(a_{t}^{(2)})^{2}\right)-\left((x-a_{t-1}^{(2)})^{2}-2(x-a_{t-1}^{(2)})a_{t-1}^{(2)}+(a_{t-1}^{(2)})^{2}\right)\right]$$ $$=2x\left(a_{t}^{(1)}-a_{t-1}^{(1)}-a_{t}^{(2)}+a_{t-1}^{(2)}\right)+2\left((a_{t}^{(1)})^{2}-(a_{t-1}^{(1)})^{2}+(a_{t}^{(2)})^{2}-(a_{t-1}^{(2)})^{2}\right).$$
Taking the maximum over x and using x ∈ [−1, 1] :

$$\sup_{x}|f_{t}(x,y_{t}^{*}(x))-f_{t-1}(x,y_{t-1}^{*}(x))|=2\left|a_{t}^{(1)}-a_{t-1}^{(1)}\right|+2\left|-a_{t}^{(2)}+a_{t-1}^{(2)}\right|$$
$$+\,2\left|(a_{t}^{(1)})^{2}-(a_{t-1}^{(1)})^{2}\right|+2\left|(a_{t}^{(2)})^{2}-(a_{t-1}^{(2)})^{2}\right|.$$

14

=
1 2
hx
2 + 4xa
$$a_{t}^{(1)}+4(a_{t}^{(1)})^{2}\Big)-\Big(x^{2}+4$$
2 + 4xa
$$\stackrel{(1)}{t-1}+4{\left(a_{t-1}^{(1)}\right)}^{2}\right)\Bigr]$$
Since a
(1)
t = 1/t and a
(2)
t = 1/
√t for all t ∈ [T], then we have Then, we get The series PT
t=2 2 t 2 +1 2t 3/2 +
1 t 3converges, implying VT = O(1). Moreover, we have

$$V_{T}:=\sum_{t=2}^{T}\sup_{x}|f_{t}(x,y_{t}^{*}(x))-f_{t-1}(x,y_{t-1}^{*}(x))|=\sum_{t=2}^{T}\left(\frac{2}{t^{2}}+\frac{1}{2t^{3/2}}+\frac{1}{t^{3}}\right).$$
$$H_{2,T}=\sum_{t=2}^{T}\sup_{x}\|y_{t}^{*}(x)-y_{t-1}^{*}(x)\|^{2}=\sum_{t=2}^{T}\sup_{x}\|x-a_{t}^{(2)}-x+a_{t-1}^{(2)}\|^{2}$$ $$=\sum_{t=2}^{T}|-a_{t}^{(2)}+a_{t-1}^{(2)}|^{2}=\sum_{t=2}^{T}|a_{t}^{(2)}-a_{t-1}^{(2)}|^{2}\approx\sum_{t=2}^{T}\frac{1}{4t^{3}},$$

which implies H2,T = O(1).

To achieve VT = o(T), the changes in the cost functions ft(x, y
∗
t(x)) and y
∗
t(x) should decay to zero faster than O(1/t).

For example, if the coefficients in the functions change as O(1/ta) with a > 1, then the cumulative sum over T will be o(T). When ft(x, y
∗ t
(x)) and y
∗ t
(x) decay as O(1/
√t), then the total variation grows at most as O(
√T).

## C. Proof Of Regret Bounds For Simultaneous Online Gradient Descent (Sogd)

770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 Proof Roadmap. We introduce Lemma C.2, which quantifies the error between the approximated direction of the momentum-based gradient estimator, d y t, and the true direction, ∇ygt(xt, yt), at each iteration. To bound the error of the lower-level variable, we provide Lemma C.4, which captures the gap ∥yt+1 − y
∗
t(xt)∥
2and incorporates the error introduced in Lemma C.2. Moreover, we provide Lemma C.5, which quantifies the error between the approximated direction of the momentum-based gradient estimator, d v t, and the true direction, ∇2ygt (zt) vt + ∇yft(zt), at each iteration. To bound the error of the system solution, we provide Lemma C.8, which captures the gap ∥vt+1 − v
∗
t
(xt)∥
2and incorporates the error introduced in Lemma C.5. Moreover, we provide Lemma C.9, which quantifies the error between the approximated direction of the momentum-based hypergradient estimator, d x t, and the true direction, ∇xft(zt) + ∇2xygt (zt) vt, at each iteration. We also present Lemma C.11, which provides an upper bound for the projection mapping and relates to the three errors discussed in Lemmas C.4, C.8, and C.9. Finally, by combining these lemmas and appropriately setting the parameters, we achieve the desired result.

## C.1. Proof Of Lemma 3.1

Proof. SOBOW (Lin et al., 2024) has estimated the hypergradient as the weighted average of previous ones over a sliding window of size w for a given Bt := {ξ1*, . . . , ξ*b} drawn i.i.d. from the distribution Df , as follows:

$$\mathrm{with}\ W=\sum_{j=t-w+1}^{t}(1-\eta)^{t-j}.$$
$${\hat{\nabla}}F_{t,\nu}(\mathbf{x}_{t},\mathbf{y}_{t};{\mathcal{B}}_{t})={\frac{1}{W}}\sum_{i=0}^{w-1}\nu^{i}{\hat{\nabla}}f_{t-i}(\mathbf{x}_{t-i},\mathbf{y}_{t-i};{\mathcal{B}}_{t-i}),$$

with W =Pw−1 i=0 ν i, ν ∈ (0, 1). Let ν = 1 − η for η ∈ (0, 1).

Then, the above equality is equivalent to

$${\hat{\nabla}}F_{t,\nu}(\mathbf{x}_{t},\mathbf{y}_{t};{\mathcal{B}}_{t})={\frac{1}{W}}\sum_{j=t-w+1}^{t}(1-\eta)^{t-j}{\hat{\nabla}}f_{j}(\mathbf{x}_{j},\mathbf{y}_{j};{\mathcal{B}}_{j}),$$
t−j∇ˆ fj (xj , yj ; Bj ), (35)
$$(35)$$
$$\begin{array}{l l}{{|a_{t}^{(1)}-a_{t-1}^{(1)}|\approx\frac{1}{t^{2}},}}&{{|a_{t}^{(2)}-a_{t-1}^{(2)}|\approx\frac{1}{2t^{3/2}},}}\\ {{}}&{{|(a_{t}^{(1)})^{2}-(a_{t-1}^{(1)})^{2}|\approx\frac{1}{t^{3}},}}&{{|(a_{t}^{(2)})^{2}-(a_{t-1}^{(2)})^{2}|\approx\frac{1}{t^{2}}.}}\end{array}$$

Let dˆx t:= ∇ˆ Ft,ν(xt, yt; Bt). Then (35) is equivalent to

$$\hat{\mathbf{d}}_{t}^{\mathbf{x}}={\frac{1}{W}}\hat{\nabla}f_{t}(\mathbf{x}_{t},\mathbf{y}_{t};\mathcal{B}_{t})+(1-\eta)\hat{\mathbf{d}}_{t-1}^{\mathbf{x}}-{\frac{(1-\eta)^{w}}{W}}\hat{\nabla}f_{t-w}(\mathbf{x}_{t-w},\mathbf{y}_{t-w};\mathcal{B}_{t-w}),$$

with fi(·) = 0 for all i ≤ 0. If w = t and W =
1 η
, then, we have

$$(36)$$
$$\hat{\mathbf{d}}_{t}^{\mathbf{x}}=\eta\hat{\nabla}f_{t}(\mathbf{x}_{t},\mathbf{y}_{t};\mathcal{B}_{t})+(1-\eta)\hat{\mathbf{d}}_{t-1}^{\mathbf{x}}.$$
$\square$
C.2. Bounds on the Inner Decision Variable We first provide a lemma that characterizes the Lipschitz continuity of approximate gradients, inner, and system solutions.

825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 Lemma C.1. Under Assumptions 3.2 and *3.3, for all* x, x
′ ∈ X *, and the search directions* {d x t }
T
t=1 and {d v t }
T
t=1 *generated* by Algorithm *1, we have*

$$\left\|\mathbf{d}_{t}^{\mathbf{x}}-\nabla f_{t}(\mathbf{x}_{t},\mathbf{y}_{t}^{*}(\mathbf{x}_{t}))\right\|^{2}\leq M_{f}^{2}\left(\left\|\mathbf{y}_{t}-\mathbf{y}_{t}^{*}(\mathbf{x}_{t})\right\|^{2}+\left\|\mathbf{v}_{t}-\mathbf{v}_{t}^{*}(\mathbf{x}_{t})\right\|^{2}\right),$$
2, (37a)
∥d v t ∥ 2 ≤ M2v ∥yt − y ∗ t(xt)∥ 2 + ∥vt − v ∗ t(xt)∥ ∥∇ft(x, y ∗ t (x)) − ∇ft(x ′, y ∗ t (x ′))∥ ≤ Lf ∥x − x ∥y ∗ t(x) − y ∗ t(x ′)∥ ≤ Ly ∥x − x ∥v ∗ t(x) − v ∗ t(x ′)∥ ≤ Lv ∥x − x
2, (37b)
′∥ , (37c)
′∥ , (37d)
′∥ , (37e)
where Mf , Mv, and (Ly, Lv, Lf ) *are defined in* (40), (41), and (42), respectively. Proof. We first show (37a).

Using Assumptions 3.2 and 3.3, we have ∇2ygt (xt, y
∗
t(xt)) ⪰ µg, and

$$\|\mathbf{v}_{t}^{*}(\mathbf{x}_{t})\|=\|\left(\nabla_{\mathbf{y}}^{2}g_{t}\left(\mathbf{x}_{t},\mathbf{y}_{t}^{*}(\mathbf{x}_{t})\right)\right)^{-1}\nabla_{\mathbf{y}}f_{t}\left(\mathbf{x}_{t},\mathbf{y}_{t}^{*}(\mathbf{x}_{t})\right)\|\leq{\frac{\ell_{f,0}}{\mu_{g}}}.$$
. (38)
Next, we establish (37b).

∥d x t − ∇ft(xt, y ∗ t (xt))∥ ≤ ∥∇xft(xt, yt) − ∇xft(xt, y ∗ t (xt))∥ + ∥vt∇2xygt(xt, yt) − v ∗ t (xt)∇2xygt (xt, y ∗ t (xt)) ∥ ≤ ∥∇xft(xt, yt) − ∇xft(xt, y ∗ t(xt))∥ + ∥∇2xygt(xt, yt)∥∥vt − v ∗ t (xt)∥ + ∥v ∗ t(xt)∥∥∇2xygt(xt, yt) − ∇2xygt(xt, y ∗ t(xt))∥

$$(38)$$
$$\leq\left(\ell_{f,1}+\frac{\ell_{g,2}\ell_{f,0}}{\mu_{g}}\right)\|{\bf y}_{t}-{\bf y}_{t}^{*}({\bf x}_{t})\|+\ell_{g,1}\|{\bf v}_{t}-{\bf v}_{t}^{*}({\bf x}_{t})\|\,,$$ $$\leq M_{f}^{2}\left(\|{\bf y}_{t}-{\bf y}_{t}^{*}({\bf x}_{t})\|+\|{\bf v}_{t}-{\bf v}_{t}^{*}({\bf x}_{t})\|\right),$$
(xt)∥), (39)
where  $$M_{f}:=\sqrt{2}\max\left\{\ell_{f,1}+\frac{\ell_{g,2}\ell_{f,0}}{\mu_{g}},\ell_{g,1}\right\},$$  the third inequality is by Assumption 3.3, and the last inequality follows from (38).  
, (40)
$$(39)^{\frac{1}{2}}$$
$$(40)$$
Observe that Since d v t
∗:= ∇yft(xt, y
∗
t(xt)) + ∇2ygt (xt, y
∗
t(xt)) v
∗
t(xt) = 0, we have Then, from Assumption 3.3 and (38), we have

∥d v t ∥ ≤ ℓg,2∥yt − y ∗ t(xt)∥∥v ∗ t(xt)∥ + ℓg,1∥vt − v ∗ t(xt)∥ + ℓf,1∥yt − y ∗ t(xt)∥ ≤ ℓg,2ℓf,0 µg+ ℓf,1 ∥yt − y ∗ t (xt)∥ + ℓg,1∥vt − v ∗ t (xt)∥ ≤ Mv (∥yt − y ∗ t(xt)∥ + ∥vt − v ∗ t(xt)∥),
where  $$M_{\mathbf{v}}:=\sqrt{2}\max\left\{\frac{\ell_{g,2}\ell_{f,0}}{\mu_{g}}+\ell_{f,1},\ell_{g,1}\right\}.$$  The proofs of Eqs. (37c)-(37e) follow from Tarznamh et al. (2024, Lemma 17) by setting 
. (41)
$$L_{\mathbf{y}}:={\frac{\ell_{g,1}}{\mu_{g}}},$$
$$\begin{array}{c}{{\mu_{g}}}\\ {{L_{\mathbf{v}}:=\ell_{f,1}+\frac{\ell_{g,1}\ell_{f,1}}{\mu_{g}}+\frac{\ell_{f,0}}{\mu_{g}}\left(\ell_{g,2}+\frac{\ell_{g,1}\ell_{g,2}}{\mu_{g}}\right),}}\\ {{L_{f}:=\ell_{f,1}+\frac{\ell_{g,1}(\ell_{f,1}+M_{f})}{\mu_{g}}+\frac{\ell_{f,0}}{\mu_{g}}\left(\ell_{g,2}+\frac{\ell_{g,1}\ell_{g,2}}{\mu_{g}}\right),}}\end{array}$$

$$(41)$$
$$(42)$$
$$(43)$$
$$(44)^{\frac{1}{2}}$$
where the other constants are defined in Assumption 3.3.

880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 we have:

E∥e g t+1∥ 2 ≤ (1 − γt+1) 2(1 + 48ℓ 2g,1β 2 t )E∥e g t ∥ 2 + 2γ 2 t+1 σ 2gy ¯b+ 24(1 − γt+1) 2ℓ 2g,1E∥xt+1 − xt∥ 2 + 6(1 − γt+1) 2E∥∇ygt(zt+1) − ∇ygt+1(zt+1)∥ 2 + 48(1 − γt+1) 2ℓ 2 g,1β 2 t  E∥∇ygt(xt, yt)∥ 2. (44)
Proof. From Algorithm 1, we have

$$\mathbf{d}_{t+1}^{\mathbf{y}}=\nabla_{\mathbf{y}}g_{t+1}(\mathbf{z}_{t+1};{\vec{\mathcal{B}}}_{t+1})+(1-\gamma_{t+1})(\mathbf{d}_{t}^{\mathbf{y}}-\nabla_{\mathbf{y}}g_{t+1}(\mathbf{z}_{t};{\vec{\mathcal{B}}}_{t+1})).$$

Then, we have

E∥e g t+1∥ 2 = E∥d y t+1 − ∇ygt+1(zt+1)∥ 2 = E∥∇ygt+1(zt+1; B¯t+1) + (1 − γt+1)(d y t − ∇ygt+1(zt; B¯t+1)) − ∇ygt+1(zt+1)∥ 2 = E∥(1 − γt+1)e g t + (∇ygt+1(zt+1; B¯t+1) − ∇ygt+1(zt+1)) − (1 − γt+1)∇ygt+1(zt; B¯t+1)− ∇ygt(zt)∥ 2,
17

$$e_{t}^{g}:=\mathbf{d}_{t}^{y}-\nabla_{\mathbf{y}}g_{t}(\mathbf{x}_{t},\mathbf{y}_{t}),$$
t − ∇ygt(xt, yt), (43)
Lemma C.2. Suppose Assumptions *3.5,* B3. and C1. *hold. Let* {(xt, yt, vt)}
T
t=1 be generated according to Algorithm 1.

For e g t *defined as*

∥d v t ∥ = ∥d v t − d v t ∗∥ = ∥vt∇2ygt(xt, yt) + ∇yft(xt, yt) −v ∗ t(xt)∇2ygt (xt, y ∗ t(xt)) + ∇yft(xt, y ∗ t(xt))∥ ≤ ∥ ∇2ygt(xt, yt) − ∇2ygt(xt, y ∗ t(xt))v ∗ t(xt)∥ + ∥∇2ygt(xt, yt) (vt − v ∗ t(xt)) ∥ + ∥∇yft(xt, yt) − ∇yft(xt, y ∗ t (xt))∥.
which implies that E∥e g t+1∥
2 = (1 − γt+1)
2E∥e g t ∥
2 + E∥(∇ygt+1(zt+1; B¯t+1) − ∇ygt+1(zt+1))
− (1 − γt+1)∇ygt+1(zt; B¯t+1)− ∇ygt(zt)∥
2
≤ (1 − γt+1)
2E∥e g t ∥
2 + 2γ 2 t+1E∥∇ygt+1(zt+1; B¯t+1) − ∇ygt+1(zt+1)∥
2
+ 2(1 − γt+1)
2E∥∇ygt+1(zt+1; B¯t+1)
− ∇ygt+1(zt+1) − ∇ygt+1(zt; B¯t+1) + ∇ygt(zt)∥
2
≤ (1 − γt+1)
2E∥e g t ∥
2 + 2γ 2 t+1 σ 2 gy
¯b
+ 2(1 − γt+1)
2E∥∇ygt+1(zt+1; B¯t+1)
− ∇ygt+1(zt+1) − ∇ygt+1(zt; B¯t+1) + ∇ygt(zt)∥
2, where the second inequality follows from Cauchy–Schwartz inequality and Assumption 3.5. Moreover, from Cauchy–Schwartz inequality, we have E∥e g t+1∥
2 ≤ (1 − γt+1)
2E∥e g t ∥
2 + 2γ 2 t+1 σ 2 gy
¯b
+ 6(1 − γt+1)
2E∥∇ygt(zt) − ∇ygt(zt+1)∥
2
+ 6(1 − γt+1)
2E∥∇ygt(zt+1) − ∇ygt+1(zt+1)∥
2
+ 6(1 − γt+1)
2E∥∇ygt+1(zt+1; B¯t+1) − ∇ygt+1(zt; B¯t+1)∥
2.

From Assumption B3., we have E∥∇ygt(zt+1) − ∇ygt(zt)∥
2
≤ 2E∥∇ygt(xt+1, yt+1) − ∇ygt(xt+1, yt)∥
2 + 2E∥∇ygt(xt+1, yt) − ∇ygt(xt, yt)∥
2
≤ 2ℓ 2g,1E∥xt+1 − xt∥
2 + 2ℓ 2g,1E∥yt+1 − yt∥
2
= 2ℓ 2 g,1E∥xt+1 − xt∥
2 + 2ℓ 2 g,1β 2 t E∥d y t ∥
2, and E∥∇ygt+1(zt+1; B¯t+1) − ∇ygt+1(zt; B¯t+1)∥
2
≤ 2E∥∇ygt+1(xt+1, yt+1; B¯t+1) − ∇ygt+1(xt+1, yt; B¯t+1)∥
2
+ 2E∥∇ygt+1(xt+1, yt; B¯t+1) − ∇ygt+1(xt, yt; B¯t+1)∥
2
≤ 2ℓ 2 g,1E∥xt+1 − xt∥
2 + 2ℓ 2 g,1E∥yt+1 − yt∥
2
= 2ℓ 2g,1E∥xt+1 − xt∥
2 + 2ℓ 2g,1β 2 t E∥d y t ∥
2.

From the two inequalities above, we have E∥e g t+1∥
2 ≤ (1 − γt+1)
2E∥e g t ∥
2 + 2γ 2 t+1 σ 2 gy
¯b
+ 6(1 − γt+1)
2E∥∇ygt(zt+1) − ∇ygt+1(zt+1)∥
2
+ 24(1 − γt+1)
2ℓ 2 g,1 E∥xt+1 − xt∥
2 + β 2 t E∥d y t ∥
2.

935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 Since e g t:= d y t − ∇ygt(xt, yt), we have

E∥e g t+1∥ 2 ≤ (1 − γt+1) 2E∥e g t ∥ 2 + 2γ 2 t+1 σ 2gy ¯b+ 24(1 − γt+1) 2ℓ 2g,1E∥xt+1 − xt∥ 2 + 6(1 − γt+1) 2E∥∇ygt(zt+1) − ∇ygt+1(zt+1)∥ 2 + 48(1 − γt+1) 2ℓ 2 g,1β 2 t  E∥e g t ∥ 2 + 48(1 − γt+1) 2ℓ 2 g,1β 2 t  E∥∇ygt(xt, yt)∥ 2 ≤ (1 − γt+1) 2(1 + 48ℓ 2 g,1β 2 t )E∥e g t ∥ 2 + 2γ 2 t+1 σ 2gy ¯b+ 24(1 − γt+1) 2ℓ 2 g,1E∥xt+1 − xt∥ 2 + 6(1 − γt+1) 2E∥∇ygt(zt+1) − ∇ygt+1(zt+1)∥ 2 + 48(1 − γt+1) 2ℓ 2 g,1β 2 t  E∥∇ygt(xt, yt)∥ 2.
Lemma C.3. Suppose Assumptions *3.2, and* B3. *hold. Then, for the sequence* {(xt, yt)}
T
t=1 generated by Algorithm *1, we* have

$$\mathbb{E}\left[\|\mathbf{y}_{t+1}-\mathbf{y}_{t}^{*}(\mathbf{x}_{t})\|^{2}\right]\leq(1+a)\left(1-2\beta_{t}\frac{\mu_{q}\ell_{q,1}}{\mu_{g}+\ell_{g,1}}\right)\mathbb{E}\left[\|\mathbf{y}_{t}-\mathbf{y}_{t}^{*}(\mathbf{x}_{t})\|^{2}\right]$$ $$\quad+\left(-(1+a)\left(\frac{2\beta_{t}}{\mu_{g}+\ell_{g,1}}-\beta_{t}^{2}\right)\right)\mathbb{E}\left[\|\nabla_{\mathbf{y}}g_{t}(\mathbf{x}_{t},\mathbf{y}_{t})\|^{2}\right]$$ $$\quad+(1+\frac{1}{a})\beta_{t}^{2}\mathbb{E}\left[\|e_{t}^{q}\|^{2}\right],$$

990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006
$$I(007)$$ $$I(008)$$
1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020
1021
1022 1023 1024 1025 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042
1043
1044
$$\begin{array}{l}{I026}\\ {I027}\end{array}$$
where e g t *defined in* (43) and a > 0 is a constant.

Proof. From Lemma B.5, we have where the inequality results from the strong convexity of gt by Assumption 3.2, which implies

$$\langle\nabla_{\mathbf{y}}g_{t}(\mathbf{x}_{t},\mathbf{y}_{t}),\mathbf{y}_{t}-\mathbf{y}_{t}^{*}(\mathbf{x}_{t})\rangle\geq{\frac{\mu_{g}\ell_{g,1}}{\mu_{g}+\ell_{g,1}}}\|\mathbf{y}_{t}-\mathbf{y}_{t}^{*}(\mathbf{x}_{t})\|^{2}+{\frac{1}{\mu_{g}+\ell_{g,1}}}\|\nabla_{\mathbf{y}}g_{t}(\mathbf{x}_{t},\mathbf{y}_{t})\|^{2}.$$

Substituting (46) into (45), gives the desired result. To simplify the notation in the analysis, we introduce the definitions

$\theta_{t}^{\mathbf{y}}:=\|\mathbf{y}_{t}-\mathbf{y}_{t}^{*}(\mathbf{x}_{t})\|^{2},\quad\text{and}\quad\theta_{t}^{\mathbf{y}}:=\|\mathbf{v}_{t}-\mathbf{v}_{t}^{*}(\mathbf{x}_{t})\|^{2}.$
2. (47)
$$\square$$
$$(47)$$
19

Next, we will bound the first term on the RHS of (45). We have
E -∥yt − βt∇ygt(xt, yt) − y ∗ t(xt)∥ 2= E-∥yt − y ∗ t(xt)∥ 2+ β 2 t  E -∥∇ygt(xt, yt)∥ 2 − 2βtE [⟨∇ygt(xt, yt), yt − y ∗ t(xt)⟩] ≤ 1 − 2βtµgℓg,1 µg + ℓg,1 E -∥yt − y ∗ t(xt)∥ 2 − 2βt µg + ℓg,1 − β 2 t E -∥∇ygt(xt, yt)∥
$$(45)$$

$$(46)$$

2, (46)
$$\mathbb{E}\left[\|\mathbf{y}_{t+1}-\mathbf{y}_{t}^{*}(\mathbf{x}_{t})\|^{2}\right]=\mathbb{E}\left[\|\mathbf{y}_{t}-\beta_{t}\mathbf{d}_{t}^{\mathbf{y}}-\mathbf{y}_{t}^{*}(\mathbf{x}_{t})\|^{2}\right]$$ $$\leq(1+a)\mathbb{E}\left[\|\mathbf{y}_{t}-\beta_{t}\nabla_{\mathbf{y}}g_{t}(\mathbf{x}_{t},\mathbf{y}_{t})-\mathbf{y}_{t}^{*}(\mathbf{x}_{t})\|^{2}\right]$$ $$+(1+\frac{1}{a})\beta_{t}^{2}\mathbb{E}\left[\|\mathbf{d}_{t}^{\mathbf{y}}-\nabla_{\mathbf{y}}g_{t}(\mathbf{x}_{t},\mathbf{y}_{t})\|^{2}\right].$$

2. (45)
Lemma C.4. Suppose Assumptions *3.2, and* B2., B3. *hold. Let* θ y t be defined as in (47)*. Then, for the sequence*
{(xt, yt)}
T
t=1 generated by Algorithm *1, the following bound is guaranteed:*

1045 1046 1047 1048 1049
$$\begin{array}{r}{1050}\\ {1051}\\ {1052}\\ {1053}\end{array}$$
1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081
1082
1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099 Proof. From Lemma B.5, we have for any c >´ 0 From Lemma C.3, we have for any a > 0 Substituting (50) into (49), we get Choose c´ =
βtLµg /2 1−βtLµg and a =βtLµg 1−2βtLµg
. Then, the following equations and inequalities are satisfied.

$$(50)$$
$$(51)$$
$$(52)^{\frac{1}{2}}$$
$$\begin{array}{l}{{(1+\dot{c})(1+a)\left(1-2\beta_{t}L_{\mu_{g}}\right)=1-\frac{\beta_{t}L_{\mu_{g}}}{2},}}\\ {{(1+a)\left(1-2\beta_{t}L_{\mu_{g}}\right)=1-\beta_{t}L_{\mu_{g}},}}\\ {{(1+\dot{c})\left(1-\beta_{t}L_{\mu_{g}}\right)=1-\frac{\beta_{t}L_{\mu_{g}}}{2},}}\\ {{1+\frac{1}{a}\leq\frac{1}{\beta_{t}L_{\mu_{g}}},\quad1+\frac{1}{\dot{c}}\leq\frac{2}{\beta_{t}L_{\mu_{g}}},}}\end{array}$$

20

$$\mathbb{E}\left[\|\mathbf{y}_{t+1}-\mathbf{y}_{t}^{*}(\mathbf{x}_{t})\|^{2}\right]\leq(1+a)\left(1-2\beta_{t}\frac{\mu_{g}\ell_{g,1}}{\mu_{g}+\ell_{g,1}}\right)\mathbb{E}\left[\|\mathbf{y}_{t}-\mathbf{y}_{t}^{*}(\mathbf{x}_{t})\|^{2}\right]$$ $$\quad+\left(-(1+a)\left(\frac{2\beta_{t}}{\mu_{g}+\ell_{g,1}}-\beta_{t}^{2}\right)\right)\mathbb{E}\left[\|\nabla_{\mathbf{y}}g_{t}(\mathbf{x}_{t},\mathbf{y}_{t})\|^{2}\right]$$ $$\quad+\left(1+\frac{1}{a}\right)\beta_{t}^{2}\mathbb{E}\left[\|e_{t}^{g}\|^{2}\right].$$

2. (50)
E -∥yt+1 − y ∗ t+1(xt+1)∥ 2 ≤ (1 + ´c)(1 + a) 1 − 2βtµgℓg,1 µg + ℓg,1 E -∥yt − y ∗ t (xt)∥ 2 + −(1 + ´c)(1 + a) 2βt µg + ℓg,1 − β 2 t  E -∥∇ygt(xt, yt)∥ 2 + (1 + ´c)(1 +  1 a )β 2 t  E -∥e g t ∥ 2 + 1 + 1 c´ E -∥y ∗ t+1(xt+1) − y ∗ t(xt)∥

2. (51)
$$\mathbb{E}\left[\|\mathbf{y}_{t+1}-\mathbf{y}_{t+1}^{*}(\mathbf{x}_{t+1})\|^{2}\right]=\mathbb{E}\left[\|\mathbf{y}_{t+1}-\mathbf{y}_{t}^{*}(\mathbf{x}_{t})+\mathbf{y}_{t}^{*}(\mathbf{x}_{t})-\mathbf{y}_{t+1}^{*}(\mathbf{x}_{t+1})\|^{2}\right]$$ $$\leq(1+\epsilon)\,\mathbb{E}\left[\|\mathbf{y}_{t+1}-\mathbf{y}_{t}^{*}(\mathbf{x}_{t})\|^{2}\right]$$ $$+\left(1+\frac{1}{\epsilon}\right)\mathbb{E}\left[\|\mathbf{y}_{t+1}^{*}(\mathbf{x}_{t+1})-\mathbf{y}_{t}^{*}(\mathbf{x}_{t})\|^{2}\right].$$

2. (49)
X T t=1 E[θ y t+1] − E[θ y t] ≤ − Lµg 2 X T t=1 βtE[θ y t] +  2 Lµg X T t=1 βtE-∥e g t ∥ 2+ 4L 2y Lµg X T t=1 1 βt E∥xt − xt+1∥ 2 +4 Lµg X T t=2 1 βt sup x∈X E∥y ∗ t−1 (x) − y ∗ t (x)∥ 2 +X T t=1 −2βt µg + ℓg,1 + β 2 t E -∥∇ygt(xt, yt)∥ 2, (48) µg+ℓg,1 , Ly = ℓg,1 µgis defined as in (42); H2,T is defined in (10). Moreover, e g tis defined in (43).
$$(49)$$

where Lµg =µgℓg,1