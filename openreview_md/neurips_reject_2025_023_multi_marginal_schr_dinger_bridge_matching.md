# Multi-Marginal Schrodinger Bridge Matching ¨

Anonymous Author(s)
Affiliation Address email

## Abstract

1 Understanding the continuous evolution of populations from discrete temporal 2 snapshots is a critical research challenge, particularly in fields like developmental 3 biology and systems medicine where longitudinal tracking of individual entities is 4 often impossible. Such trajectory inference is vital for unraveling the mechanisms 5 of dynamic processes. While Schrodinger Bridge (SB) offer a potent framework, ¨ 6 their traditional application to pairwise time points can be insufficient for systems 7 defined by multiple intermediate snapshots. This paper introduces Multi-Marginal 8 Schrodinger Bridge Matching (MSBM), a novel algorithm specifically designed ¨ 9 for the multi-marginal SB problem. MSBM extends iterative Markovian fitting 10 (IMF) to effectively handle multiple marginal constraints. This technique ensures 11 robust enforcement of all intermediate marginals while preserving the continuity of 12 the learned global dynamics across the entire trajectory. Empirical validations on 13 synthetic data and real-world single-cell RNA sequencing datasets demonstrate the 14 competitive or superior performance of MSBM in capturing complex trajectories 15 and respecting intermediate distributions, all with notable computational efficiency. 17 Understanding the continuous evolution of populations from discrete temporal snapshots represents 18 a significant challenge in various scientific disciplines, particularly in fields like developmental 19 biology [7, 42] and systems medicine [29] where tracking individual entities longitudinally is often 20 unfeasible. The ability to infer trajectories from such snapshot data is crucial for elucidating the 21 underlying mechanisms of dynamic processes. The Schrodinger Bridge (SB) problem, originally ¨ 22 rooted in statistical mechanics [43], has garnered substantial interest in machine learning as an 23 entropy-regularized, continuous-time formulation of optimal transport [20, 30]. It seeks to identify 24 the most probable evolutionary path between prescribed initial and terminal distributions, and has 25 been successfully employed in generative modeling [3, 4, 9, 26, 27, 37, 38, 45, 49]. 26 However, many real-world scenarios present observations or constraints at multiple time points, not 27 just at the beginning and end of a process. For instance, in single-cell RNA sequencing (scRNA-seq) 28 experiments, which are pivotal for studying complex biological processes like cell differentiation, cells 29 are typically destroyed upon measurement [6, 17, 28]. This destructive nature makes it impossible 30 to track individual cells over time, thus necessitating the inference of developmental trajectories 31 from population-level snapshots collected at several intermediate stages. Similarly, meteorological 32 systems may have partial observations across various times [11, 32]. Such situations necessitate 33 a multi-marginal generalization of the SB problem (mSBP), where the path measure must align 34 with prescribed marginal distributions at multiple intermediate time points. While the traditional 35 SB framework offers a powerful approach, its standard application to pairwise time points can 36 prove insufficient for systems characterized by multiple intermediate snapshots. Although more 37 specialized methods for mSBP have recently been developed [8, 18, 44], the direct application of 38 some multi-marginal approaches can lead to error accumulation if not carefully managed, particularly

## 16 **1 Introduction**

39 when learned controls are even slightly inaccurate. These challenges highlight the need for robust 40 and scalable solutions for the mSBP that can effectively integrate information across all observed 41 time points.

42 This paper introduces Multi-Marginal Schrodinger Bridge Matching (MSBM), a novel algorithm ¨
43 specifically developed to address the multi-marginal SB problem by building upon and extending the 44 Iterative Markovian Fitting (IMF) algoritmhs [36, 45]. MSBM is designed to effectively manage mul45 tiple marginal constraints by constructing local SBs on each interval and seamlessly integrating them. 46 This local construction strategy, underpinned by a shared global parametrization of control functions, 47 ensures the robust enforcement of all intermediate marginal distributions while crucially preserving 48 the continuity of the learned global dynamics across the entire trajectory. Empirical validations 49 conducted on synthetic datasets as well as real-world single-cell RNA sequencing data demonstrate 50 that MSBM achieves competitive or superior performance in capturing complex trajectories and 51 accurately respecting intermediate distributions, all while exhibiting notable computational efficiency. 52 Our work aims to provide a robust and scalable computational method for these multi-marginal 53 settings, addressing the critical need for consistent and tractable dynamic inference when data is 54 available as snapshots at multiple time points.

55 We summarize our contributions as follows: 56 - We extend the theoretical and algorithmic foundations of SBs, including the IMF iteration and 57 optimal control perspectives, to the challenging multi-marginal setting.

58 - We introduce an efficient modeling approach for trajectory inference, that constructs and 59 smoothly integrates local SBs across sub-intervals, inherently allows for parallelized train60 ing, leading to significant speed-ups. 61 - Through comprehensive experiments on both synthetic and real-world single-cell RNA sequenc62 ing data, we demonstrate that MSBM accurately models complex population dynamics and 63 outperforms state-of-the-art methods in both trajectory fidelity and computational speed.

Notation. Let P[0,T] denote the space of continuous functions taking values in R
d 64 on the interval
[0, T]. We use an uppercase letter P ∈ P[0,T]
65 to represent a path measure. For a path measure P ∈ P[0,T]
66 , the marginal distribution at discrete time points T = {t0*, . . . , t*k}, where 0 = t0 < t1 <
· · · < tk = T is denoted by PT ∈ PT , where we define PT as the set of measures P over R
d*×|T |* 67 .

Additionally, the conditional distribution of P, given T , is denoted by P|T ∈ P[0,T]
68 . Moreover, a 69 path measure P can be defined as mixture. For any Borel measurable set A ∈ B(Ω), P can be defined by P(A) = R
70 Rd*×|T |* P|T (A|xT )dPT (xT ), where P ∈ P0,T and P ∈ PT , and we use the shorthand 71 xT := (x1, *· · ·* , xk) and [0 : k] := {0, 1, · · · , k}. The Kullback-Leibler (KL) divergence between two probability measures µ and ν on space X is defined as DKL(µ|ν) = RX
log dµ dν 72 (X)dµ(X) when 73 µ is absolutely continuous with respect to ν (µ ≪ ν), and DKL(µ|ν) = +∞ otherwise. We will often refer to probability measures on R
d 74 and their Lebesgue densities interchangeably, under the standard assumption of absolute continuity. Finally, for a function V : [0, T] × R 75 d → R, we define the gradient and laplcaian operators with respect to x ∈ R
d 76 as ∇V and ∆V, respectively, and its 77 partial derivative with respect to time t ∈ [0, T] as ∂tV.

## 78 **2 Preliminaries** 79 **2.1 Schrodinger Bridge Matching (Sbm)** ¨

80 The Schrodinger Bridge problem (SBP) [ ¨ 16, 43] is a stochastic optimal transport problem [30] that 81 seeks the optimal transport plan for endpoint marginals ρ0 and ρT . In this paper, we focus on the dynamical representation, where a reference distribution Q ∈ P[0,T]
82 is induced by the SDEs:
dXt = ft(Xt) dt + σ dWt, X0 ∼ ρ0, (1)
where ft : R
d → R
dis a drift, σ ∈ R is a diffusion, and Wt ∈ R
d 83 is a standard Wiener process.

84 With the base reference path measure Q, the dynamic representation of the SB [20, 35, 39] is:
min P∈P[0,T ]
DKL(P|Q), subject to P0 ∼ ρ0, PT ∼ ρT . (SBP)
91 **Reciprocal Projection** R. For a given reference measure Q from (1), and a path measure P with 92 marginals specified at end points T = {0, T} the reciprocal projection is defined as:
R(P, T ) := PT Q|T = P0,T Q|0,T . (2)
93 This projection constructs a new path measure by taking the endpoint coupling P0,T from P and 94 forming a mixture of bridge process using Q conditioned on these end points. Sampling from 95 Π := R(P, T ) involves drawing end points samples (X0, XT ) ∼ P0,T and then generating a path XT
96 t between them using conditional reference measure Q|0,T which induced by following SDEs, for 97 any (x0, xT ):

$\eqref{eq:walpha}$. 
$${\cal{L}}^{*}$$
dXT
t =-ft(XT
t) + σ 2∇ log QT|t(xT |XT
t)dt + σdWt, XT0 = x0, (3)
98 If Q|0,T has tractable bridge formulation, for example, when Q is chosen as a Brownian motion 99 i.e*., d*Xt = σdWt, sampling the path at time t given the endpoints can be performed as:

$$({\boldsymbol{5}})$$
($\small\sf{6}$). 
XT
t ∼ N (1 −
t T
)X0 +
t T XT , t(1 −
t T
)σ 2, where (X0, XT ) ∼ P0,T . (4)
Intuitively, the term EQT |t
[XT |Xt = x] can be understood as a prediction of the target state X⋆
t 106 .

Flow matching [23] of Bridge matching [37] tackles the approximation X⋆T ≈ EQT |t 107 [XT |Xt = x]
by learning a drift function. This learned drift guides the evolution of X⋆
t 108 such that its terminal 109 state aligns with the target, often by regressing the drift agains a target drift derived from samples of 110 (X0, XT ) under the reference conditional bridge measure Q|0,T .

111 Building upon the projections R and M, Schrodinger Bridge Matching (SBM) methods [ ¨ 37, 45] 112 refines the path measure through an alternating iteraive procedure:
P

(2n+1) := M(P
(2n), T ), P
(2n+2) := R(P
(2n+1), T ). (7)

## 115 **3 Multi-Marginal Iterative Markovian Fitting**

116 Dynamic SB methods, as discussed in Section 2, have traditionally focused on problems defined 117 by two endpoint marginal distributions, (ρ0, ρT ). However, in real-world applications, particularly 118 in fields like developmental biology (e.g., scRNA-seq studies of cellular differentiation), systems 119 are often observed through snapshots at multiple intermediate time points, not just at the beginning 120 and end of a process. This prevalence of multi-stage data highlights a critical limitation of standard 121 SB approaches. While the theoretical extension of SB methods to handle multiple marginals has 122 been explored [1, 31], the development of robust and scalable computational methods for these 123 multi-marginal settings has lagged. Recently, methods with IPF-type objectives have been derived 124 for multi-marginal cases [8, 44]. However, challenges persist in ensuring global dynamic consistency 125 across all intervals, maintaining computational tractability as the number of marginals increases.

85 Recent advancements in dynamical optimal transport [37, 45] have introduced a novel numerical 86 methodology for solving SBP. This approach reframes SBP by decomposing its dynamical constraints 87 into the time-evolving marginal distributions Pt for all t ∈ [0, T] and the joint coupling P0,T . This optimization relies on IMF [45], a technique that iteratively refines the path measure P ∈ P[0,T]
88 . 89 IMF alternates between two projection called Markovian and Reciprocal projections to preserve the 90 correct endpoint marginals (ρ0, ρT ) throughout the optimization.

100 **Markov Projection** M. Although the reciprocal projection R in (2) preserves end point marginals 101 (ρ0, ρT ), its sampling process in (4) requires both (X0, XT ), making it non-Markovian and thus 102 ill-suited for generative modeling aimed at sampling from ρT without knowing XT . The Markov 103 projection M resolves this by projecting Π := R(P, T ) into a family of Markov process while ensuring P 104 ⋆ = Πt for all t ∈ [0, T]. Again, when Q is chosen as a Brownian motion i.e*., d*Xt = σdWt, the Markov projection of Π, P 105 ⋆ = M(Π, T ), is induced by following SDEs:

$$\begin{array}{l l}{{d{\bf X}_{t}^{\star}=\sigma v^{\star}(t,{\bf X}_{t}^{\star})d t+\sigma d{\bf W}_{t},}}&{{{\bf X}_{0}^{\star}\sim\Pi_{0},}}\\ {{\mathrm{where}}}&{{v^{\star}(t,{\bf x})=\frac{1}{T-t}\left(\mathbb{E}_{Q_{T|t}}\left[{\bf X}_{T}|{\bf X}_{t}={\bf x}\right]-{\bf x}\right).}}\end{array}$$
$$(7)$$
$$T\,\}.$$

Initialized with P
(0) = P
(0)
T Q|0,T , utilizing P
(0) T
113 is independent coupling of ρ0 and ρT along with the 114 reference conditional bridge measure Q|T . Please refer to [37, 45] for more details.

126 In this section, we extends the SBM framework−conventionally applied to problems with two 127 endpoint marginals (ρ0, ρT ) and foundational to IMF methods−to handle cases involving k + 1 multiple snapshots (ρ0, ρt1 128 , · · · , ρT ) on discrete time stamps T = {t0, t1, · · · , tk} where 0 = t0 <
t1 < · · · < tk = T
1 129 . Similar to SBP, the dynamic multi-marginal Schrodinger Bridge problem can ¨ 130 be formally defined as [10] the entropy minimization problem:
min P∈P[0,T ]
DKL(P|Q), subject to Pt ∼ ρt, ∀t ∈ T . (mSBP)

## 135 **3.1 Multi-Marginal Projection Operators**

136 To develop multi-marginal extension of SBM, we investigate how the IMF framework can be adapted 137 to scenarios with multiple snapshots (i.e., where the set of time points T has cardinality *|T |* > 2). 138 This adaptation necessitates extending the fundamental building blocks of SBM—specifically, the 139 reciprocal projection R and the Markov projection M—to handle multiple marginal constraints.

Multi-Marginal Reciprocal Projection Rmm 140 . First, we state and prove a proposition that character141 izes the reciprocal structure of conditional path measures. In particular, we focus on a mixture of 142 bridges Π = ΠT Q|T ∈ P[0,T] constrained by the marginals at multiple timestamps in T .

Proposition 1 (Reciprocal Property). *For any* xT := (x0, xt1, *· · ·* , xT ) ∈ R
d×(k+1) 143 and t ∈
144 [ti−1, ti), the marginal distribution of Q|T (·|xT ) at t *satisfies:*
Q|T (xt|xT ) = Q|ti−1,ti
(xt|xti, xti−1). (8)
Therefore, for any P ∈ P[0,T]*the reciprocal projection* Rmm 145 (P, T ) *admits the following factorization:*

$$({\mathfrak{g}})$$

Rmm(P, T ) = PT Q|T = Pt0,··· ,tkQ|t0,··· ,tk = Pt0,··· ,tkQk i=1Q|ti−1,ti
, P*-a.e.* (9)

$$d\mathbf{X}_{t}^{\star}=\left[f_{t}(\mathbf{X}_{t}^{\star})+\sigma v^{\star}(t,\mathbf{X}_{t}^{\star})\right]d t+\sigma d\mathbf{W}_{t},\quad\mathbf{X}_{0}^{\star}\sim\Pi_{0},$$
)] dt + σdWt, X⋆0 ∼ Π0, (10)
$\mathbf{v}$
$\frac{1}{2}$ 4. 
∂tρt = *−∇ ·* (v
⋆
t(x)ρt(x)) + σ 2 2 ∆ρt(x) = 0, ρt = Πt, ∀t ∈ T , (12)
where pt is marginal density of Πt*. In other words,* P
⋆
157 t = Πt for all t ∈ [0, T]. d 158 As established in Proposition 2, constructing a global diffusion process via (10) with the optimal control v
⋆(11)) yields a multi-marginal Markov projection X⋆
[0,T]
159 that is continuous over the entire time interval [0, T]. The continuity arises because the local Markov projections, X⋆
[ti−1,ti]
160 , on each sub-interval are derived from factorized conditional bridge Q|ti−1,ti 161 in (9). These bridges are

$$(10)^{\frac{1}{2}}$$
$$(11)^{\frac{1}{2}}$$
$$(12)$$
$$\mathbf{I}_{t},\quad\forall t\in{\mathcal{T}},$$
$$z=-p\tau(\Delta z)$$

$\text{2iect to}$
To find a most probable path P
mSBP 131 , the solution of mSBP under multiple constraints, we will generalize 132 the principles of SBM in Section 2.1 to the multi-marginal cases in Section 3.1. The extension of 133 dynamic SB optimality [20, 35] and the associated stochastic optimal control problem [39] to multi134 marginal settings is presented in Appendix A.

Multi-Marginal Markov Projection Mmm 152 . With the reciprocal property and factorization in (9),
153 we show that the Markov projection on multi-marginal case can be constructed by similar fashion.

154 **Proposition 2** (Multi-Marginal Markovian Projection). Let Π ∈ P[0,T] *admit factorzation in* (9)*. The* multi-marginal Markov projection of Π, P
⋆:= Mmm(Π, T ) ∈ P[0,T]
155 *, is associated with the SDE:*

_where $v^{\star}(t,\mathbf{x})=\sum_{i=1}^{k}\mathbf{1}_{[t_{i-1},t_{i})}\mathbb{E}_{\Pi_{t_{i}|t}}\left[\nabla\log\mathbb{Q}_{t_{i}|t}(\mathbf{X}_{t_{i}}|\mathbf{X}_{t})|\mathbf{X}_{t}=\mathbf{x}\right].$_
|Xt)|Xt = x. (11)
Moreover, v
⋆
156 *satisfies the Fokker-Planck equation (FPE) [40]:* 146 A key implication of the reciprocal property, detailed in Proposition 1, is that a mixture of diffusion 147 bridges constrained on T factorizes into independent segments over successive time intervals. This 148 factorization simplifies the analysis and simulation of the overall path measure. Since each segment 149 can then be treated as a standard conditional bridge process as in (3), closed-form sampling, such as in (4), can be applied independently in parallel to each subinterval {ti−1, ti}i∈[1:k]
150 . This tractability 151 is essential for developing an efficient multi-marginal SBM algorithm.

anchored by identical marginal distributions at there shared boundaries; for instance, both X⋆
[ti−1,ti]
162 and X⋆
[ti,ti+1]
is guaranteed to match the marginal distribution ρtiat time ti 163 . Consequently, these local 164 diffusion processes connect seamlessly at adjacent timestamps, resulting in a smooth and well-defined path for X⋆
[0,T]
. The well-defined nature of the global path, in conjunction with the projections Rmm 165 and Mmm 166 , is fundamental to successfully applying the SBM framework to the mSBP. Finally, the 167 uniquness condition for standard SB [45, Proposition 5] can also be extended to multi-marginal case.

Proposition 3 (Uniqueness). Let P
⋆
168 be a Markov measure which is reciprocal class of Q *satisfying* P

⋆
t = ρt for all t ∈ T *. Then,* P
⋆*is unique solution* P
mSBP 169 of the *mSBP*. Building on the projection operators Rmm 170 ,Mmm with the uniquness result of Proposition 3, we can 171 apply the iterative algorithm used in SBM algorithm [45, Algorithm 1] to the multi-marginal setting:
P

(2n+1) := Mmm(P
(2n), T ), P
(2n+2) := Rmm(P
(2n+1), T ), *|T |* > 2. (13)
172 The convergence guarantees proved for the iteration apply equally well to the multi-marginal case.

Proposition 4 (Convergence). P
(n) = P
mSBP 173 of mSBP as n ↑ ∞ with iterative procedure in *(13)*.

## 174 **3.2 Practical Implementation.**

In practice, at each iteration n of (13) we approximate the optimal control v
⋆
175 from (11) by a neural 176 network vθ. By Girsanov theorem, θ are chosen to minimize the following training objective function:
L(θ, T , ΠT ) = R T
0 EΠt,T[||σ∇ log QβT (t)|t(XβT (t)|Xt) − vθ(t, Xt)||2dt], (14)
177 where βT (t) = minu{u > t|t *∈ T } ∈* [0, T] is the most recent time point in T after time t. With 178 this notation, the SBM can be generalized to the case of multi-marginal constraints. For example, 179 when T = {0, T} then (14) reduces to the objective function described in [45].

The learned Markov control vθ
⋆ (t, xt) then ensures P
θ
⋆
180 t = Πt for all t ∈ [0, T]. Moreover, prior
181 SBM algorithms interleave forward and backward-time Markov projections to re-anchor the terminal
distribution and prevent bias between P
(n) T
182 and ΠT accumulate for each n ∈ N. In the multi-marginal
183 setting, we again build the backward-time Markov projection as in Proposition 2 by *gluing* the local
bridge reversals, so that P
⋆
184 is governed by both SDEs (10) and the corresponding backward dynamics:
$d\mathbf{Y}_{t}^{\star}=[-f_{T-t}(\mathbf{Y}_{t}^{\star})+\sigma u^{\star}(t,\mathbf{Y}_{t}^{\star})]\,dt+\sigma d\mathbf{W}_{t},\quad\mathbf{Y}_{0}^{\star}\sim\Pi_{T},$  where $u^{\star}(t,\mathbf{y})=\sum_{i=1}^{k}\mathbf{1}_{(t_{i-1},t_{i}]}(t)\mathbb{E}_{\Pi_{t|t_{i-1}}}\left[\nabla\log\mathbb{Q}_{t|t_{i-1}}(\mathbf{Y}_{t}|\mathbf{Y}_{t_{i-1}})|\mathbf{Y}_{t}=\mathbf{y}\right],$
)|Yt = y, (16)
where the backward optimal control u
⋆
185 in (16) can be approximated with neural network uϕ where ϕ 186 is chosen to minimize the following training objective function with γT (t) = maxu{u < t|t ∈ T }:

$$(15)$$
$$(16)$$
$${\mathcal{L}}(\phi,{\mathcal{T}},\Pi_{{\mathcal{T}}})=\int_{0}^{T}\mathbb{E}_{\Pi_{t,\,{\mathcal{T}}}}[||\sigma\nabla\log\mathbb{Q}_{t|\gamma{\mathcal{T}}(t)}(\mathbf{Y}_{t}|\mathbf{Y}_{\gamma{\mathcal{T}}(t)})-u_{\phi}(t,\mathbf{Y}_{t})||^{2}d t].$$

## 187 **4 Multi-Marginal Schrodinger Bridge Matching** ¨

A na¨ıve extension of the standard SBM using, multi-marginal projections Rmm and Mmm 188 in Sec 3, 189 encounters significant limitations not present in the traditional two-endpoint setting. In such an 190 extension, each iteration typically enforces marginal constraints only at the global endpoints (ρ0, ρT ).

The multi-marginal coupling Π
(n) T
191 at each iteration n of (13) is then derived by propagating the 192 projected dynamics in (10) or (15) solely from these end points ρ0 or ρT , respectively.

193 This approach leads to critical issues specific to the multi-marginal context. Firstly, if the learned controls, such as v
⋆(forward) or u
⋆
194 (backward), are even slightly inaccurate, significant biases can arise between the inferred intermediate marginals (Π(n)
t1
, *· · ·* Π
(n)
tk−1 195 ) and the target marginals
(ρt1
, · · · , ρtk−1 196 ). Secondly, these discrepancies tend to accumulate iteratively. This accumulation is exacerbated because, beyond an initialization Π(0) = P
(0)
T Q|T with P
(0) T
197 , independent joint coupling 198 of {ρt}t∈T , where the joint distribution might be informed by all prescribed data distributions, 199 the subsequent self-refinement process for the dynamics often does not directly incorporate the

$$(17)$$

## Algorithm 1 Training Of Msbm

1: **Input:** Snapshots {ρt}t∈T , bridge Q|T , N ∈ N 2: Let {P
(0)
Ti
}i∈[1:k]joint coupling of {ρt∈Ti }i∈[1:k].

3: for n ∈ {0*, . . . , N* − 1} do 4: for i ∈ {1*, . . . , k* − 1} **do in parallel**
5: Let Π
(2n)
Ti= P
(2n)
Ti 6: Estimate L(ϕ, Ti, Π
(2n)
Ti, Q|Ti
)
7: Estimate L˜(ϕ) = Pk i=1 L(ϕ, Ti, Π
(2n)
Ti, Q|Ti
)
8: uϕ⋆ = arg minϕ Pk i=1L˜(ϕ)
9: Simulate local backward SBs {P
i,(2n+1)}i∈[1:k]
10: for i ∈ {1*, . . . , k* − 1} **do in parallel** 11: Let Π
(2n+1)
Ti= P
(2n+1)
Ti 12: Estimate L(θ, Ti, Π
(2n+1)
Ti, Q|Ti
)
13: Estimate L˜(θ) = Pk i=1 L(θ, Ti, Π
(2n+1)
Ti, Q|Ti
)
14: vθ⋆ = arg minθ Pk i=1L(θ, Ti, Π
(2n+1)
Ti)
15: Simulate local forward SBs {P
i,(2n+2)
[ti−1,ti]
}
16: **end for** 17: **Output:** v
⋆
θ, u⋆ϕ Algorithm 2 Simulation of MSBM (forward)
Input: Initial ρ0, learned control vθ⋆ Sample X0 ∼ ρ0 Simulate forward SDE over [0, T]
dX⋆
t = [ft + σvθ⋆ (t, X⋆
t )] dt + σdWt, Output: Trajectory X⋆
[0,T ]

Naïve MSBM
204 To address this issue of error accumulation and ensure all marginal constraints {ρt}t∈T are satisfied, we propose a method that involves constructing local SBs on each interval [ti−1, ti 205 ] and then 206 seamlessly *gluing* them together. Instead of propagating dynamics from the global endpoints ρ0 and 207 ρT alone, our approach first establishes local SBs for each segment. The resulting local couplings 208 are then systematically integrated to satisfy all specified marginal distributions {ρt}t∈T across the 209 entire time interval [0, T]. This local construction strategy helps prevent the compounding of errors at intermediate time points while still aiming to achieve the overall multi-marginal SB solution, P
mSBP 210 .

211 The theoretical basis is provided by the following result.

Corollary 5 (Multi-Marginal Schrodinger Bridge) ¨ . *Assume a sequence of controls* {v i, ui}i∈[1:k]
212 ,
where each v i, ui*induced local SBs* P
i of SBP over local interval [ti−1, ti 213 ] *with distributions*
(ρti−1, ρti) *in a forward and backward direction, respectively. If* limt↑tiv i(t, x) = v i+1 214 (t, x) and limt↓ti−1 u i(t, x) = u i−1(t, x) *for all* i ∈ [1 : k]*, then* P
mSBP 
215 of mSBP *induced by following SDEs:*
dX⋆
t = [ft(X⋆
t) + σv⋆(t, X⋆
t)] dt + σdWt, X⋆
0 ∼ ρ0. (18a)
dY⋆
t = [−fT −t(Y⋆
t) + σu⋆(t, Y⋆
t)] dt + σdWt, Y⋆
0 ∼ ρT , (18b)
where v
⋆(t, x) = Pk i=11[ti−1,ti)(t)v i(t, x), u⋆(t, x) = Pk i=11(ti−1,ti](t)u i(t, x). (18c)
intermediate data distributions (ρt1
, · · · , ρtk−1 200 ) into its training objective except ρ0 and ρT . Without 201 explicit targets for the intermediate marginals guiding each iteration, the inferred paths between ρ0 202 and ρT can "collapse" or drift away from the desired states. Consequently, precisely satisfying all 203 intermediate constraints becomes increasingly challenging as iterations proceed.

216 Building upon Corollary 5, we introduce our Multi-Marginal Schrodinger Bridge Matching (MSBM) ¨
217 method to solve the mSBP. A cornerstone of MSBM is divide the global mSBP into local SBPs while
maintaining the continuity of the composite drift functions v
⋆and u
⋆
218 in (18c) across adjacent intervals,
which guarantees a globally continuous diffusion process inducing P
mSBP 219 . Furthermore, by explicitly
constraining each local SBs, P
i, on its corresponding marginals (ρti−1, ρti
220 ), MSBM is designed to 221 mitigate the accumulation of bias at intermediate marginals, as shown in Figure 1. 222 A key challenge of the MSBM is rigorously satisfying the continuity conditions at the boundaries of
local controls: limt↑tiv
i(t, x) = v
i+1(t, x) and limt↓ti−1 u
i(t, x) = u
i−1
223 (t, x) for all i ∈ [1 : k]. If 224 these conditions are not met, discontinuities or "kinks" can arise at the intermediate time steps. Such
kinks would imply that the overall path measure P
⋆̸= Mmm(P
⋆
225 , T ). This would, in turn, hinder the
226 optimlaity for mSBP, because, following Proposition 3, the desired continuous Markov process is a fixed point of both Rmm and Markov projections Mmm 227 under multiple time points T :
P 
$${}^{*}={\mathcal{R}}^{\mathsf{m n}}(\mathbb{P}^{*},{\mathcal{T}})={\mathcal{M}}^{\mathsf{m n}}(\mathbb{P}^{*},{\mathcal{T}}).$$
⋆, T ). (19)
228 To construct local SBs such that the continuity requirements for forming a valid global solution are 229 met, thereby preventing the aforementioned kinks and ensuring (19), our MSBM introduces a shared global parametrization vθ, uϕ for its respective local controls {v i, ui}i∈[1:k]
230 for each sub-interval, 231 where each local controls are parallel updated with following aggregate objective function:

$$\tilde{\mathcal{L}}(\theta)=\sum_{i=1}^{k}\mathcal{L}(\theta,\mathcal{T}_{i},\Pi_{\mathcal{T}_{i}}),\quad\tilde{\mathcal{L}}(\phi)=\sum_{i=1}^{k}\mathcal{L}(\phi,\mathcal{T}_{i},\Pi_{\mathcal{T}_{i}}),$$
$$(20\mathrm{a})$$

where Ti = {ti−1, ti} define sub-intervals with local coupling ΠTi 232 for end-points marginals in interval [ti−1, ti 233 ] and L is defined in (14) and (17) for forward and backward direction, respectively.

234 The MSBM training procedure, summarized in Algorithm 1, adapts the standard IMF algorithm 235 presented in [45, Algorithm 1]. A key distinction in our MSBM approach is the parallel application 236 of the IMF procedure to each local time interval, utilizing globally shared forward vθ and backward 237 uϕ across all local intervals. This parallel processing across sub-intervals contributes to a significant 238 reduction in overall training time.

## 239 **5 Related Work**

240 The solution of SBP often utilize Iterative Proportional Fitting (IPF) [19], with modern adaptations 241 learning SDE drifts for two-marginal settings [4, 9, 13, 49]. A distinct iterative approach, IMF, as 242 featured in [37, 45], offers improved stability by alternating projections onto different classes of 243 path measures. Moreover, emerging research also explores non-iterative algorithm [12, 38]. These 244 methodologies primarily concentrate on the SB problem itself, iteratively refining path measures or 245 directly computing the bridge measure. Moreover, the SB algorithm is studied under the assumption 246 that the optimal coupling is given [27, 46]. While recent studies have extended foundational SB ideas 247 to the multi-marginal setting of mSBP, research in this area remains relatively limited.

248 In multi-marginal setting, [8] extends the problem to phase space to encourage smoother trajectories 249 and introduces a novel training methodology inspired by the Bregman iteration [5] to handle multiple 250 marginal constraints. Relatedly, [44] presented an approach that, similar to our work, segments the 251 problem across intervals; they learn piecewise SBs and use likelihood-based training to iteratively 252 refine a global reference dynamic. While these methods are often IPF-based or focus on specific 253 reference refinement strategies, our MSBM extends the previous IMF-type algorithm into multi254 marginal setting and effectively handles multiple constraints. We demonstrate that our MSBM 255 framework offers substantial gains in training efficiency. This enhanced efficiency is primarily 256 attributed to its direct multi-marginal formulation that adeptly manages multiple constraints, thereby 257 circumventing the computationally intensive iterative refinements common in IPF-based methods 258 Paralleling these SB-centric developments, other significant lines of work model dynamic trajectories 259 by directly learning potential functions or velocity fields, often drawing from optimal transport 260 or continuous normalizing flows. For instance, [18, 24–26] extend SBs to incorporate potentials 261 or mean-field interactions, connecting to stochastic optimal control and earlier mean-field game 262 frameworks [22, 41]. The broader field of trajectory inference from snapshot data, crucial for 263 applications like scRNA-seq, has seen methods like [48] using CNFs with dynamic OT, and [15] 264 employing Neural ODEs on learned data manifolds. More recently, [33, 34] offer variational 265 objectives to learn dynamics from marginal samples.

## 266 **6 Experiments**

267 In this section, we empirically demonstrate the effectiveness of our MSBM. Specifically, our goal 268 is to infer a dynamic model from datasets composed of samples from marginal distributions ρt 269 observed at discrete time points. We evaluate MSBM on both synthetic datasets and real-world single270 cell RNA sequencing datasets, including human embryonic stem cells (hESC) [11] and embryoid 271 body (EB) [32]. To ensure consistency and fair comparison, our experiments follow the respective 272 experimental setups established by baseline methods. In particular, for the petal dataset, we adopt 273 the experimental setup from DMSB [8], and for the hESC dataset, we follow SBIRR [44]. For 274 the EB dataset, we perform evaluations on both 5-dim and 100-dim PCA-reduced data; here, we 275 follow the 100-dim experimental setup of DMSB and the 5-dim setup from NLSB [18]. Accordingly,

MIOFlow DMSB MSBM Groud Truth t0 t1 t2 t3 t4 traj

## 293 **6.2 Single-Cell Sequencing Data**

Figure 3: Comparison of generated population dynamics using MIOFlow, DMSB and MSBM on a 2-dim petal dataset. All trajectories are generated by simulating the dynamics from ρt0.

276 we utilize evaluation metrics consistent with previous studies, including the Sliced-Wasserstein 277 Distance (SWD)[2], Maximum Mean Discrepancy (MMD)[14], as well as the 1-Wasserstein (W1) 278 and 2-Wasserstein (W2) distances. All experimental results reported are averaged mean value over 279 three independent runs with different random seeds. We highlight the best-performing results in **bold** 280 and the second-best results in blue. Further experimental details are provided in Appendix C.

## 281 **6.1 Synthetic Data**

282 **Petal** The petal dataset [15] serves as a sim283 ple yet complex challenge because it mimics 284 the natural dynamics seen in processes such as 285 cellular differentiation, which include phenom286 ena like bifurcations and merges. We compare 287 our MSBM with MIOFlow [15] and DMSB [8]
288 in Figure 2. As shown in Figure 3, we ob289 serve that MSBM exhibits the most accurate and 290 clearly defined trajectory, closely resembling the 291 ground truth. Furthermore, Figure 2 demonstrates the evaluation results for the trajectories through 292 W2 and MMD distances, highlighting that MSBM consistently outperforms MIOFlow and DMSB.

Figure 2: Evaluation results of W2 and MMD.

t0 t1 t2 t3 t4 Time 0.00 0.02 0.04 0.06 0.08 0.10 DMSB MIOFlow MSBM
t0 t1 t2 t3 t4 Time 0.1 0.2 0.3 0.4 2 MM
D

294 We evaluated our MSBM on real-world single-cell RNA sequencing data from two sources: 1) human 295 embryonic stem cells (hESCs) [11] undergoing differentiation into definitive endoderm over a 4-day 296 period, measured at 6 distinct time points (t0:0 hours, t1:12 hours, t2:24 hours, t3:36 hours, t4:72 297 hours, and t5:96 hours); 2) embryoid body (EB) cells [32] differentiating into mesoderm, endoderm, 298 neuroectoderm, and neural crest over 27 days, with samples collected at 5 time windows (t0:0-3 days, 299 t1:6-9 days, t2:12-15 days, t3:18-21 days, and t4:24-27 days). Following the experimental setup of 300 baselines, we preprocessed these datasets using the pipeline outlined in [48], and the collected cells 301 were projected into a lower-dimensional space using principal component analysis (PCA).

Table 1: Performance on the 5dim PCA of hESC dataset. W2 is compute between test ρtiand generated ρˆtiby simulating the dynamics from test ρt0
.

| W2 ↓                     | Runtime                |       |       |
|--------------------------|------------------------|-------|-------|
| Methods                  | t1                     | t3    | hours |
| TrajectoryNet† 1.30 1.93 | 10.19                  |       |       |
| DMSB†                    | 1.10 1.51              | 15.54 |       |
| SBIRR†                   | 1.08 1.33 0.36 (0.38)∗ |       |       |
| MSBM (Ours) 1.09 1.30    | 0.09                   |       |       |
| † result from [44].      |                        |       |       |

302 **hESC** To follow the experimental setup from SBIRR [44], we 303 reduced the data to the first five principal components and excluded 304 the final time point t6 from our dataset, resulting in three train305 ing time points T = {t0, t2, t4} and two intermediate test points 306 Ttest = {t1, t3}. Our objective was to train the dynamics based on 307 the available marginals at the training points in T and interpolate 308 the intermediate test marginals at Ttest, which were not observed 309 during training. Table 1 demonstrates that our proposed MSBM
310 method performs competitively, achieving lower W2 distances.

311 **Embryoid Body** We validate our MSBM on both 5-dim and 312 100-dim PCA spaces. First, for the 5-dim experiment, we adopt the 313 experimental setup from NLSB. Given 5 observation time points T = {t0, t1, t2, t3, t4}, we divide the data using train/test splits ρ tr T
/ρte T
314 , with the goal of predicting population-level dynamics from ρ tr t0
. Similar to NLSB, we train the dynamics based on ρ tr T
315 and Table 3: Performance on the 100-dim PCA of EB dataset. MMD and SWD are computed between test ρ te ti and generated ρˆtiby simulating the dynamics from test ρ te t0
.

Figure 4: Comparison of generated population dynamics using DMSB and MSBM on a 100-dim PCA of EB dataset. The plot displays the first two principal components as the x and y axes, respectively.

| MMD ↓                                                                                                                                                                   | SWD ↓                                   |    |    |    |      |    |    |    |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|----|----|----|------|----|----|----|
| Methods                                                                                                                                                                 | Full                                    | t1 | t2 | t3 | Full | t1 | t2 | t3 |
| NLSB† [18]                                                                                                                                                              | 0.66 0.38 0.37 0.37 0.54 0.55 0.54 0.55 |    |    |    |      |    |    |    |
| MIOFlow† [15] 0.23 0.23 0.90 0.23 0.35 0.49 0.72 0.50 DMSB† [8] 0.03 0.04 0.04 0.04 0.16 0.20 0.19 0.18 MSBM 0.02 0.04 0.04 0.05 0.11 0.18 0.17 0.19 † result from [8]. |                                         |    |    |    |      |    |    |    |

DMSB MSBM Groud Truth t0 t1 t2 t3 t4 traj
evaluate the W1 distance between ρ te ti and the generated ρˆti from previous test snapshot ρ te ti−1 316 . 317 In Table 2, we find that MSBM outperforms several SB methods.

Table 2: Performance on the 5-dim PCA
of EB dataset. W1 is computed between test ρ te ti and generated ρˆti by simulating the dynamics from previous test ρ te ti−1
.

318 For the 100-dim experiment, we borrow the experimental 319 setup from DMSB, where the goal is predict population 320 dynamics given that observations are available for all time 321 points T (denoted as Full in Table 3), or when one of the snapshot is left out (denoted as ti 322 in Table 3, where snapshot ρ tr ti at ti 323 is excluded during training). The high 324 performance in this task represent the robustness of the 325 model to accurately predict population dynamics. In Ta326 ble 3, MSBM consistently yields performance improve327 ments. Moreover, as shown in Figure 4, the trajectories 328 and generated marginal distributions ρˆT in PCA space fur329 ther justifies the numerical result and highlights the variety 330 and quality of the samples produced by MSBM.

| W1 ↓                                                                                                                                                                                                                                                                                                                                                            |                          |    |    |    |      |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|----|----|----|------|
| Methods                                                                                                                                                                                                                                                                                                                                                         | t1                       | t2 | t3 | t4 | Mean |
| Neural SDE† [21]                                                                                                                                                                                                                                                                                                                                                | 0.69 0.91 0.85 0.81 0.82 |    |    |    |      |
| TrajectoryNet† [48] 0.73 1.06 0.90 1.01 0.93 IPF (GP)† [49] 0.70 1.04 0.94 0.98 0.92 IPF (NN)† [4] 0.73 0.89 0.84 0.83 0.82 SB-FBSDE† [9] 0.56 0.80 1.00 1.00 0.84 NLSB† [18] 0.68 0.84 0.81 0.79 0.78 OT-CFM† [47] 0.78 0.76 0.77 0.75 0.77 WLF-SB‡ [34] 0.63 0.79 0.77 0.75 0.73 MSBM (Ours) 0.64 0.73 0.72 0.73 0.71 † result from [18], ‡ result from [34]. |                          |    |    |    |      |

## 346 **7 Conclusion And Limitation**

337 This enhanced computational efficiency primarily originates from 338 core algorithmic differences. SBIRR, for example, utilizes maxi339 mum likelihood training, which requires extensive gradient compu340 tations and the storage of all intermediate paths. DMSB employs an 341 IPF-type objective with Bregman Iteration [5]. In contrast, MSBM 342 directly optimizes controls using an IMF-type objective, which not 343 only eliminates the need to store intermediate states but also fa344 cilitates parallel computation across sub-intervals. This approach 345 substantially promotes faster convergence of the algorithm.

Figure 5: Training time
331 **Computational Efficiency** For an fair comparison of 332 training efficiency against recent multi-marginal SB al333 gorithms, we benchmarked DMSB and SBIRR on the identical hardware configuration employed for MSBM (denoted by ∗
334 in Table 1). On the hESC dataset, MSBM achieved a runtime over 4× 335 faster than SBIRR. Furthermore, on the petal and 100-dim PCA of EB dataset, MSBM significantly 336 outperformed DSMB in training speed, with detailed results presented in Figure 5. 347 This paper revisits previously established frameworks for the SBP, extending them to the mSBP. 348 Specifically, we introduce a computationally efficient framework for mSBP, termed MSBM, which 349 builds upon existing SBM methods [37, 45]. MSBM is tailored for various trajectory inference 350 problems where snapshots of data are available at multi-marginal time steps. Through the successful 351 adaptation of the IMF algorithm to this multi-marginal setting, our approach significantly accelerates 352 training processes while ensuring accurate dynamic modeling when compared to existing methods. 353 Despite these advantages, the performance degradation of MSBM is more pronounced than that 354 of DMSB when a time point is omitted in Table 3. This may occur because the including velocity 355 term could better accommodate unknown trajectory. Furthermore, the current MSBM framework 356 is restricted to the case involving snapshot data samples, highlighting a need for enhancements to 357 address problems with continuous potentials, such mean-field games [18, 24–26].

## 358 **References**

359 [1] Aymeric Baradat and Christian Leonard. Minimizing relative entropy of path measures under ´ 360 marginal constraints. *arXiv preprint arXiv:2001.10920*, 2020. 361 [2] Nicolas Bonneel, Julien Rabin, Gabriel Peyre, and Hanspeter Pfister. Sliced and Radon Wasser- ´ 362 stein barycenters of measures. *Journal of Mathematical Imaging and Vision*, 51:22–45, 2015. 363 [3] Valentin De Bortoli, Iryna Korshunova, Andriy Mnih, and Arnaud Doucet. Schrodinger 364 bridge flow for unpaired data translation. In *The Thirty-eighth Annual Conference on Neural* 365 *Information Processing Systems*, 2024.

366 [4] Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion Schrodinger ¨ 367 bridge with applications to score-based generative modeling. In A. Beygelzimer, Y. Dauphin, 368 P. Liang, and J. Wortman Vaughan, editors, *Advances in Neural Information Processing Systems*, 369 2021. 370 [5] L.M. Bregman. The relaxation method of finding the common point of convex sets and 371 its application to the solution of problems in convex programming. *USSR Computational* 372 *Mathematics and Mathematical Physics*, 1967. 373 [6] Jason D Buenrostro, Beijing Wu, Ulrike M Litzenburger, Dave Ruff, Michael L Gonzales, 374 Michael P Snyder, Howard Y Chang, and William J Greenleaf. Single-cell chromatin accessi375 bility reveals principles of regulatory variation. *Nature*, 523(7561):486–490, 2015. 376 [7] Charlotte Bunne, Stefan G Stark, Gabriele Gut, Jacobo Sarabia Del Castillo, Mitch Levesque, 377 Kjong-Van Lehmann, Lucas Pelkmans, Andreas Krause, and Gunnar Ratsch. Learning single- ¨ 378 cell perturbation responses using neural optimal transport. *Nature methods*, 20(11):1759–1768, 379 2023.

380 [8] Tianrong Chen, Guan-Horng Liu, Molei Tao, and Evangelos Theodorou. Deep momentum 381 multi-marginal schrodinger bridge. ¨ *Advances in Neural Information Processing Systems*, 382 36:57058–57086, 2023. 383 [9] Tianrong Chen, Guan-Horng Liu, and Evangelos Theodorou. Likelihood training of schrodinger ¨
384 bridge using forward-backward SDEs theory. In International Conference on Learning Repre385 *sentations*, 2022. 386 [10] Yongxin Chen, Giovanni Conforti, Tryphon T Georgiou, and Luigia Ripani. Multi-marginal 387 schrodinger bridges. In ¨ *International Conference on Geometric Science of Information*, pages 388 725–732. Springer, 2019. 389 [11] Li-Fang Chu, Ning Leng, Jue Zhang, Zhonggang Hou, Daniel Mamott, David T Vereide, Jeea 390 Choi, Christina Kendziorski, Ron Stewart, and James A Thomson. Single-cell rna-seq reveals 391 novel regulators of human embryonic stem cell differentiation to definitive endoderm. *Genome* 392 *biology*, 17:1–20, 2016. 393 [12] Valentin De Bortoli, Iryna Korshunova, Andriy Mnih, and Arnaud Doucet. Schrodinger 394 bridge flow for unpaired data translation. *Advances in Neural Information Processing Systems*, 395 37:103384–103441, 2024. 396 [13] Wei Deng, Weijian Luo, Yixin Tan, Marin Bilos, Yu Chen, Yuriy Nevmyvaka, and Ricky T. Q. ˇ 397 Chen. Variational schrodinger diffusion models. In ¨ *Forty-first International Conference on* 398 *Machine Learning*, 2024.

399 [14] Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Scholkopf, and Alexander ¨
400 Smola. A kernel two-sample test. *The Journal of Machine Learning Research*, 13(1):723–773, 401 2012. 402 [15] Guillaume Huguet, Daniel Sumner Magruder, Alexander Tong, Oluwadamilola Fasina, Manik 403 Kuchroo, Guy Wolf, and Smita Krishnaswamy. Manifold interpolating optimal-transport flows 404 for trajectory inference. *Advances in neural information processing systems*, 35:29705–29718, 405 2022. 406 [16] Benton Jamison. The Markov processes of Schrodinger. ¨ *Zeitschrift fur Wahrscheinlichkeitsthe-* ¨ 407 *orie und verwandte Gebiete*, 32(4):323–331, 1975. 408 [17] Allon M Klein, Linas Mazutis, Ilke Akartuna, Naren Tallapragada, Adrian Veres, Victor Li, 409 Leonid Peshkin, David A Weitz, and Marc W Kirschner. Droplet barcoding for single-cell 410 transcriptomics applied to embryonic stem cells. *Cell*, 161(5):1187–1201, 2015. 411 [18] Takeshi Koshizuka and Issei Sato. Neural Lagrangian Schrodinger bridge: Diffusion modeling ¨ 412 for population dynamics. *arXiv preprint arXiv:2204.04853*, 2022. 413 [19] Solomon Kullback. Probability densities with given marginals. *The Annals of Mathematical* 414 *Statistics*, 39(4):1236–1243, 1968. 415 [20] Christian Leonard. A survey of the Schr ´ odinger problem and some of its connections with ¨ 416 optimal transport. *arXiv preprint arXiv:1308.0215*, 2013. 417 [21] Xuechen Li, Ting-Kam Leonard Wong, Ricky TQ Chen, and David Duvenaud. Scalable gradi418 ents for stochastic differential equations. In *International Conference on Artificial Intelligence* 419 *and Statistics*, pages 3870–3882. PMLR, 2020. 420 [22] Alex Tong Lin, Samy Wu Fung, Wuchen Li, Levon Nurbekyan, and Stanley J. Osher. Alternating 421 the population and control neural networks to solve high-dimensional stochastic mean-field 422 games. *Proceedings of the National Academy of Sciences*, 2021. 426 [24] Guan-Horng Liu, Tianrong Chen, Oswin So, and Evangelos Theodorou. Deep generalized 427 schrodinger bridge. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, ¨ 428 editors, *Advances in Neural Information Processing Systems*, 2022. 423 [23] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow 424 matching for generative modeling. In *The Eleventh International Conference on Learning* 425 *Representations*, 2023.

429 [25] Guan-Horng Liu, Tianrong Chen, and Evangelos A Theodorou. Deep generalized schr\" odinger 430 bridges: From image generation to solving mean-field games. *arXiv preprint arXiv:2412.20279*, 431 2024. 432 [26] Guan-Horng Liu, Yaron Lipman, Maximilian Nickel, Brian Karrer, Evangelos Theodorou, and 433 Ricky T. Q. Chen. Generalized schrodinger bridge matching. In ¨ *The Twelfth International* 434 *Conference on Learning Representations*, 2024. 435 [27] Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A Theodorou, Weili Nie, and Anima Anandkumar. I2 436 SB: Image-to-image Schrodinger bridge. ¨ *arXiv preprint arXiv:2302.05872*,
437 2023. 438 [28] Evan Z Macosko, Anindita Basu, Rahul Satija, James Nemesh, Karthik Shekhar, Melissa 439 Goldman, Itay Tirosh, Allison R Bialas, Nolan Kamitaki, Emily M Martersteck, et al. Highly 440 parallel genome-wide expression profiling of individual cells using nanoliter droplets. Cell, 441 161(5):1202–1214, 2015. 442 [29] Kenneth G Manton, XiLiang Gu, and Gene R Lowrimore. Cohort changes in active life 443 expectancy in the us elderly population: Experience from the 1982–2004 national long-term 444 care survey. *The Journals of Gerontology Series B: Psychological Sciences and Social Sciences*, 445 63(5):S269–S281, 2008. 446 [30] Toshio Mikami. *Stochastic optimal transportation: stochastic control with fixed marginals*. 447 Springer Nature, 2021. 448 [31] Abdulwahab Mohamed, Alberto Chiarini, and Oliver Tse. Schrodinger bridges with multi- ¨ 449 marginal constraints. 2021. 450 [32] Kevin R Moon, David Van Dijk, Zheng Wang, Scott Gigante, Daniel B Burkhardt, William S
451 Chen, Kristina Yim, Antonia van den Elzen, Matthew J Hirn, Ronald R Coifman, et al. Vi452 sualizing structure and transitions in high-dimensional biological data. *Nature biotechnology*,
453 37(12):1482–1492, 2019. 454 [33] Kirill Neklyudov, Rob Brekelmans, Daniel Severo, and Alireza Makhzani. Action matching: 455 Learning stochastic dynamics from samples. In Proceedings of the 40th International Confer456 *ence on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*. PMLR, 457 23–29 Jul 2023. 458 [34] Kirill Neklyudov, Rob Brekelmans, Alexander Tong, Lazar Atanackovic, Qiang Liu, and 459 Alireza Makhzani. A computational framework for solving Wasserstein Lagrangian flows. 460 *arXiv preprint arXiv:2310.10649*, 2023. 461 [35] Michele Pavon and Anton Wakolbinger. On free energy, stochastic control, and Schrodinger ¨
462 processes. In *Modeling, Estimation and Control of Systems with Uncertainty: Proceedings of a* 463 *Conference held in Sopron, Hungary, September 1990*, pages 334–348. Springer, 1991. 464 [36] Stefano Peluchetti. Non-denoising forward-time diffusions, 2022. 465 [37] Stefano Peluchetti. Diffusion bridge mixture transports, schrodinger bridge problems and ¨ 466 generative modeling. *Journal of Machine Learning Research*, 24(374):1–51, 2023. 467 [38] Stefano Peluchetti. BM$ˆ2$: Coupled schrodinger bridge matching. ¨ *Transactions on Machine* 468 *Learning Research*, 2025. 469 [39] Paolo Dai Pra. A stochastic control approach to reciprocal diffusion processes. *Applied* 470 *Mathematics and Optimization*, 23:313–329, 1991.

471 [40] Hannes Risken and Hannes Risken. *Fokker-planck equation*. Springer, 1996.

472 [41] Lars Ruthotto, Stanley J. Osher, Wuchen Li, Levon Nurbekyan, and Samy Wu Fung. A machine 473 learning framework for solving high-dimensional mean field game and mean field control 474 problems. *Proceedings of the National Academy of Sciences*, 2020. 475 [42] Geoffrey Schiebinger, Jian Shu, Marcin Tabaka, Brian Cleary, Vidya Subramanian, Aryeh 476 Solomon, Joshua Gould, Siyan Liu, Stacie Lin, Peter Berube, et al. Optimal-transport analysis 477 of single-cell gene expression identifies developmental trajectories in reprogramming. *Cell*, 478 176(4):928–943, 2019.

[43] Erwin Schrodinger. ¨ *Uber die umkehrung der naturgesetze* ¨ 479 . Verlag der Akademie der Wis480 senschaften in Kommission bei Walter De Gruyter u . . . , 1931. 481 [44] Yunyi Shen, Renato Berlinghieri, and Tamara Broderick. Multi-marginal Schrodinger bridges ¨ 482 with iterative reference refinement. *arXiv preprint arXiv:2408.06277*, 2024. 483 [45] Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion schrodinger ¨ 484 bridge matching. *Advances in Neural Information Processing Systems*, 36, 2024. 485 [46] Vignesh Ram Somnath, Matteo Pariset, Ya-Ping Hsieh, Maria Rodriguez Martinez, Andreas 486 Krause, and Charlotte Bunne. Aligned diffusion schr\" odinger bridges. *arXiv preprint* 487 *arXiv:2302.11419*, 2023. 488 [47] Alexander Tong, Kilian FATRAS, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid 489 Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based genera490 tive models with minibatch optimal transport. *Transactions on Machine Learning Research*,
491 2024. Expert Certification.

492 [48] Alexander Tong, Jessie Huang, Guy Wolf, David Van Dijk, and Smita Krishnaswamy. Trajecto493 rynet: A dynamic optimal transport network for modeling cellular dynamics. In *International* 494 *conference on machine learning*, pages 9526–9536. PMLR, 2020. 495 [49] Francisco Vargas, Pierre Thodoroff, Austen Lamacraft, and Neil Lawrence. Solving Schrodinger ¨ 496 bridges via maximum likelihood. *Entropy*, 23(9):1134, 2021.

## 497 **Neurips Paper Checklist**

498 1. **Claims** 499 Question: Do the main claims made in the abstract and introduction accurately reflect the 500 paper's contributions and scope? 501 Answer: [Yes] 502 Justification: The key claims stated in the abstract and introduction correspond appropriately 503 to the scope of the paper. 504 Guidelines:
505 - The answer NA means that the abstract and introduction do not include the claims 506 made in the paper.

507 - The abstract and/or introduction should clearly state the claims made, including the 508 contributions made in the paper and important assumptions and limitations. A No or 509 NA answer to this question will not be perceived well by the reviewers. 510 - The claims made should match theoretical and experimental results, and reflect how 511 much the results can be expected to generalize to other settings. 512 - It is fine to include aspirational goals as motivation as long as it is clear that these goals 513 are not attained by the paper. 514 2. **Limitations** 515 Question: Does the paper discuss the limitations of the work performed by the authors? 516 Answer: [Yes] 517 Justification: The conclusion section provides a discussion on the limitations. 518 Guidelines: 519 - The answer NA means that the paper has no limitation while the answer No means that 520 the paper has limitations, but those are not discussed in the paper. 521 - The authors are encouraged to create a separate "Limitations" section in their paper. 522 - The paper should point out any strong assumptions and how robust the results are to 523 violations of these assumptions (e.g., independence assumptions, noiseless settings, 524 model well-specification, asymptotic approximations only holding locally). The authors 525 should reflect on how these assumptions might be violated in practice and what the 526 implications would be. 527 - The authors should reflect on the scope of the claims made, e.g., if the approach was 528 only tested on a few datasets or with a few runs. In general, empirical results often 529 depend on implicit assumptions, which should be articulated. 530 - The authors should reflect on the factors that influence the performance of the approach. 531 For example, a facial recognition algorithm may perform poorly when image resolution 532 is low or images are taken in low lighting. Or a speech-to-text system might not be 533 used reliably to provide closed captions for online lectures because it fails to handle 534 technical jargon. 535 - The authors should discuss the computational efficiency of the proposed algorithms 536 and how they scale with dataset size. 537 - If applicable, the authors should discuss possible limitations of their approach to 538 address problems of privacy and fairness. 539 - While the authors might fear that complete honesty about limitations might be used by 540 reviewers as grounds for rejection, a worse outcome might be that reviewers discover 541 limitations that aren't acknowledged in the paper. The authors should use their best 542 judgment and recognize that individual actions in favor of transparency play an impor543 tant role in developing norms that preserve the integrity of the community. Reviewers 544 will be specifically instructed to not penalize honesty concerning limitations.

## 545 3. **Theory Assumptions And Proofs**

546 Question: For each theoretical result, does the paper provide the full set of assumptions and 547 a complete (and correct) proof? 548 Answer: [Yes] 549 Justification: Yes, we are confident that our proof and assumptions are both valid and 550 adequate. 551 Guidelines: 552 - The answer NA means that the paper does not include theoretical results. 553 - All the theorems, formulas, and proofs in the paper should be numbered and cross554 referenced. 555 - All assumptions should be clearly stated or referenced in the statement of any theorems.

556 - The proofs can either appear in the main paper or the supplemental material, but if 557 they appear in the supplemental material, the authors are encouraged to provide a short 558 proof sketch to provide intuition. 559 - Inversely, any informal proof provided in the core of the paper should be complemented 560 by formal proofs provided in appendix or supplemental material. 561 - Theorems and Lemmas that the proof relies upon should be properly referenced. 562 4. **Experimental result reproducibility** 563 Question: Does the paper fully disclose all the information needed to reproduce the main ex564 perimental results of the paper to the extent that it affects the main claims and/or conclusions 565 of the paper (regardless of whether the code and data are provided or not)? 566 Answer: [Yes] 567 Justification: Yes, all the necessary data to reproduce the results can be found in the Appendix 568 C. 569 Guidelines: 570 - The answer NA means that the paper does not include experiments. 571 - If the paper includes experiments, a No answer to this question will not be perceived 572 well by the reviewers: Making the paper reproducible is important, regardless of 573 whether the code and data are provided or not. 574 - If the contribution is a dataset and/or model, the authors should describe the steps taken 575 to make their results reproducible or verifiable.

576 - Depending on the contribution, reproducibility can be accomplished in various ways.

577 For example, if the contribution is a novel architecture, describing the architecture fully 578 might suffice, or if the contribution is a specific model and empirical evaluation, it may 579 be necessary to either make it possible for others to replicate the model with the same 580 dataset, or provide access to the model. In general. releasing code and data is often 581 one good way to accomplish this, but reproducibility can also be provided via detailed 582 instructions for how to replicate the results, access to a hosted model (e.g., in the case 583 of a large language model), releasing of a model checkpoint, or other means that are 584 appropriate to the research performed. 585 - While NeurIPS does not require releasing code, the conference does require all submis586 sions to provide some reasonable avenue for reproducibility, which may depend on the 587 nature of the contribution. For example 588 (a) If the contribution is primarily a new algorithm, the paper should make it clear how 589 to reproduce that algorithm. 590 (b) If the contribution is primarily a new model architecture, the paper should describe 591 the architecture clearly and fully. 592 (c) If the contribution is a new model (e.g., a large language model), then there should 593 either be a way to access this model for reproducing the results or a way to reproduce 594 the model (e.g., with an open-source dataset or instructions for how to construct 595 the dataset). 596 (d) We recognize that reproducibility may be tricky in some cases, in which case 597 authors are welcome to describe the particular way they provide for reproducibility. 598 In the case of closed-source models, it may be that access to the model is limited in 599 some way (e.g., to registered users), but it should be possible for other researchers 600 to have some path to reproducing or verifying the results. 601 5. **Open access to data and code** 602 Question: Does the paper provide open access to the data and code, with sufficient instruc603 tions to faithfully reproduce the main experimental results, as described in supplemental 604 material? 605 Answer: [Yes] 606 Justification: We provided our code. 607 Guidelines: 608 - The answer NA means that paper does not include experiments requiring code. 609 - Please see the NeurIPS code and data submission guidelines (https://nips.cc/ 610 public/guides/CodeSubmissionPolicy) for more details. 611 - While we encourage the release of code and data, we understand that this might not be 612 possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not 613 including code, unless this is central to the contribution (e.g., for a new open-source 614 benchmark). 615 - The instructions should contain the exact command and environment needed to run to 616 reproduce the results. See the NeurIPS code and data submission guidelines (https: 617 //nips.cc/public/guides/CodeSubmissionPolicy) for more details. 618 - The authors should provide instructions on data access and preparation, including how 619 to access the raw data, preprocessed data, intermediate data, and generated data, etc. 620 - The authors should provide scripts to reproduce all experimental results for the new 621 proposed method and baselines. If only a subset of experiments are reproducible, they 622 should state which ones are omitted from the script and why. 623 - At submission time, to preserve anonymity, the authors should release anonymized 624 versions (if applicable). 625 - Providing as much information as possible in supplemental material (appended to the 626 paper) is recommended, but including URLs to data and code is permitted.

## 627 6. **Experimental Setting/Details**

628 Question: Does the paper specify all the training and test details (e.g., data splits, hyper629 parameters, how they were chosen, type of optimizer, etc.) necessary to understand the 630 results? 631 Answer: [Yes] 632 Justification: We have included the details of the experiments. 633 Guidelines: 634 - The answer NA means that the paper does not include experiments.

635 - The experimental setting should be presented in the core of the paper to a level of detail 636 that is necessary to appreciate the results and make sense of them. 637 - The full details can be provided either with the code, in appendix, or as supplemental 638 material. 639 7. **Experiment statistical significance** 640 Question: Does the paper report error bars suitably and correctly defined or other appropriate 641 information about the statistical significance of the experiments?

642 Answer: [Yes] 643 Justification: Yes, we ran our code three times and reported the mean and standard deviations 644 in the appendix. Due to space limitations, only the mean values are presented in the main 645 text. The complete results can be found in Appendix C. 646 Guidelines: 647 - The answer NA means that the paper does not include experiments. 648 - The authors should answer "Yes" if the results are accompanied by error bars, confi649 dence intervals, or statistical significance tests, at least for the experiments that support 650 the main claims of the paper. 651 - The factors of variability that the error bars are capturing should be clearly stated (for 652 example, train/test split, initialization, random drawing of some parameter, or overall 653 run with given experimental conditions). 654 - The method for calculating the error bars should be explained (closed form formula, 655 call to a library function, bootstrap, etc.) 656 - The assumptions made should be given (e.g., Normally distributed errors).

657 - It should be clear whether the error bar is the standard deviation or the standard error 658 of the mean.

659 - It is OK to report 1-sigma error bars, but one should state it. The authors should 660 preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis 661 of Normality of errors is not verified. 662 - For asymmetric distributions, the authors should be careful not to show in tables or 663 figures symmetric error bars that would yield results that are out of range (e.g. negative 664 error rates). 665 - If error bars are reported in tables or plots, The authors should explain in the text how 666 they were calculated and reference the corresponding figures or tables in the text. 667 8. **Experiments compute resources** 668 Question: For each experiment, does the paper provide sufficient information on the com669 puter resources (type of compute workers, memory, time of execution) needed to reproduce 670 the experiments? 671 Answer: [Yes] 672 Justification: Yes, the necessary resources are included in the experimental details section. 673 Guidelines: 674 - The answer NA means that the paper does not include experiments. 675 - The paper should indicate the type of compute workers CPU or GPU, internal cluster, 676 or cloud provider, including relevant memory and storage. 677 - The paper should provide the amount of compute required for each of the individual 678 experimental runs as well as estimate the total compute. 679 - The paper should disclose whether the full research project required more compute 680 than the experiments reported in the paper (e.g., preliminary or failed experiments that 681 didn't make it into the paper). 682 9. **Code of ethics**
683 Question: Does the research conducted in the paper conform, in every respect, with the 684 NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? 685 Answer: [Yes] 686 Justification: We support the NeurIPS Code of Ethics. 687 Guidelines: 688 - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. 689 - If the authors answer No, they should explain the special circumstances that require a 690 deviation from the Code of Ethics. 691 - The authors should make sure to preserve anonymity (e.g., if there is a special consid692 eration due to laws or regulations in their jurisdiction). 693 10. **Broader impacts** 694 Question: Does the paper discuss both potential positive societal impacts and negative 695 societal impacts of the work performed? 696 Answer: [NA] 697 Justification: This paper presents work aimed at advancing the field of machine learning. 698 Our research may have various societal consequences. However, we do not believe any of 699 these require specific emphasis here. 700 Guidelines: 701 - The answer NA means that there is no societal impact of the work performed.

702 - If the authors answer NA or No, they should explain why their work has no societal 703 impact or why the paper does not address societal impact. 704 - Examples of negative societal impacts include potential malicious or unintended uses 705 (e.g., disinformation, generating fake profiles, surveillance), fairness considerations 706 (e.g., deployment of technologies that could make decisions that unfairly impact specific 707 groups), privacy considerations, and security considerations.

708 - The conference expects that many papers will be foundational research and not tied 709 to particular applications, let alone deployments. However, if there is a direct path to 710 any negative applications, the authors should point it out. For example, it is legitimate 711 to point out that an improvement in the quality of generative models could be used to 712 generate deepfakes for disinformation. On the other hand, it is not needed to point out 713 that a generic algorithm for optimizing neural networks could enable people to train 714 models that generate Deepfakes faster.

715 - The authors should consider possible harms that could arise when the technology is 716 being used as intended and functioning correctly, harms that could arise when the 717 technology is being used as intended but gives incorrect results, and harms following 718 from (intentional or unintentional) misuse of the technology. 719 - If there are negative societal impacts, the authors could also discuss possible mitigation 720 strategies (e.g., gated release of models, providing defenses in addition to attacks, 721 mechanisms for monitoring misuse, mechanisms to monitor how a system learns from 722 feedback over time, improving the efficiency and accessibility of ML). 723 11. **Safeguards** 724 Question: Does the paper describe safeguards that have been put in place for responsible 725 release of data or models that have a high risk for misuse (e.g., pretrained language models, 726 image generators, or scraped datasets)? 727 Answer: [NA] 728 Justification: We believe our paper poses no such risks. 729 Guidelines: 730 - The answer NA means that the paper poses no such risks.

731 - Released models that have a high risk for misuse or dual-use should be released with 732 necessary safeguards to allow for controlled use of the model, for example by requiring 733 that users adhere to usage guidelines or restrictions to access the model or implementing 734 safety filters. 735 - Datasets that have been scraped from the Internet could pose safety risks. The authors 736 should describe how they avoided releasing unsafe images. 737 - We recognize that providing effective safeguards is challenging, and many papers do 738 not require this, but we encourage authors to take this into account and make a best 739 faith effort.

## 740 12. **Licenses For Existing Assets**

741 Question: Are the creators or original owners of assets (e.g., code, data, models), used in 742 the paper, properly credited and are the license and terms of use explicitly mentioned and 743 properly respected? 744 Answer: [Yes] 745 Justification: Yes, the license and terms of use are noted. 746 Guidelines: 747 - The answer NA means that the paper does not use existing assets. 748 - The authors should cite the original paper that produced the code package or dataset. 749 - The authors should state which version of the asset is used and, if possible, include a 750 URL. 751 - The name of the license (e.g., CC-BY 4.0) should be included for each asset. 752 - For scraped data from a particular source (e.g., website), the copyright and terms of 753 service of that source should be provided. 754 - If assets are released, the license, copyright information, and terms of use in the 755 package should be provided. For popular datasets, paperswithcode.com/datasets 756 has curated licenses for some datasets. Their licensing guide can help determine the 757 license of a dataset. 758 - For existing datasets that are re-packaged, both the original license and the license of 759 the derived asset (if it has changed) should be provided. 760 - If this information is not available online, the authors are encouraged to reach out to 761 the asset's creators. 762 13. **New assets** 763 Question: Are new assets introduced in the paper well documented and is the documentation 764 provided alongside the assets?

765 Answer: [NA]
766 Justification: The paper does not release new assets. 767 Guidelines: 768 - The answer NA means that the paper does not release new assets. 769 - Researchers should communicate the details of the dataset/code/model as part of their 770 submissions via structured templates. This includes details about training, license, 771 limitations, etc. 772 - The paper should discuss whether and how consent was obtained from people whose 773 asset is used. 774 - At submission time, remember to anonymize your assets (if applicable). You can either 775 create an anonymized URL or include an anonymized zip file.

## 776 14. **Crowdsourcing And Research With Human Subjects**

777 Question: For crowdsourcing experiments and research with human subjects, does the paper 778 include the full text of instructions given to participants and screenshots, if applicable, as 779 well as details about compensation (if any)? 780 Answer: [NA] 781 Justification: We do not involve crowdsourcing or research with human subjects. 782 Guidelines: 783 - The answer NA means that the paper does not involve crowdsourcing nor research with 784 human subjects. 785 - Including this information in the supplemental material is fine, but if the main contribu786 tion of the paper involves human subjects, then as much detail as possible should be 787 included in the main paper.

788 - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, 789 or other labor should be paid at least the minimum wage in the country of the data 790 collector.

## 791 15. **Institutional Review Board (Irb) Approvals Or Equivalent For Research With Human**

792 **subjects** 793 Question: Does the paper describe potential risks incurred by study participants, whether 794 such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) 795 approvals (or an equivalent approval/review based on the requirements of your country or 796 institution) were obtained? 797 Answer: [NA] 798 Justification: We do not involve crowdsourcing or research with human subjects 799 Guidelines: 800 - The answer NA means that the paper does not involve crowdsourcing nor research with 801 human subjects. 802 - Depending on the country in which research is conducted, IRB approval (or equivalent) 803 may be required for any human subjects research. If you obtained IRB approval, you 804 should clearly state this in the paper.

805 - We recognize that the procedures for this may vary significantly between institutions 806 and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the 807 guidelines for their institution.

808 - For initial submissions, do not include any information that would break anonymity (if 809 applicable), such as the institution conducting the review.

## 810 16. **Declaration Of Llm Usage**

811 Question: Does the paper describe the usage of LLMs if it is an important, original, or 812 non-standard component of the core methods in this research? Note that if the LLM is used 813 only for writing, editing, or formatting purposes and does not impact the core methodology, 814 scientific rigorousness, or originality of the research, declaration is not required. 815 Answer: [NA] 816 Justification: We do not use LLM for core methodology, scientific rigorousness, or originality 817 of the research. 818 Guidelines:
819 - The answer NA means that the core method development in this research does not 820 involve LLMs as any important, original, or non-standard components.

821 - Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
822 for what should or should not be described.