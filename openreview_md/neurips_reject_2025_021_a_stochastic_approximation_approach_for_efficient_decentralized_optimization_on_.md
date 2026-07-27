# A Stochastic Approximation Approach For Efficient Decentralized Optimization On Random Networks

Anonymous Author(s)
Affiliation Address email

## Abstract 17 **1 Introduction**

1 A challenging problem in decentralized optimization is to develop algorithms with 2 fast convergence on random and time varying topologies under unreliable and 3 bandwidth-constrained communication network. This paper studies a stochastic 4 approximation approach with a Fully Stochastic Primal Dual Algorithm (FSPDA) 5 framework. Our framework relies on a novel observation that randomness in time 6 varying topology can be incorporated in a stochastic augmented Lagrangian for7 mulation, whose expected value admits saddle points that coincide with stationary 8 solutions of the decentralized optimization problem. With the FSPDA framework, 9 we develop two new algorithms supporting efficient sparsified communication on 10 random time varying topologies - FSPDA-SA allows agents to execute multiple 11 local gradient steps depending on the time varying topology to accelerate conver12 gence, and FSPDA-STORM further incorporates a variance reduction step to improve 13 sample complexity. For problems with smooth (possibly non-convex) objective 14 function, within T iterations, we show that FSPDA-SA (resp. FSPDA-STORM) finds an O(1/
√T)-stationary (resp. O(1/T2/3 15 )) solution. Numerical experiments show 16 the benefits of the FSPDA algorithms.

18 Consider n agents that communicate on an undirected and connected graph/network G = (V, E) with 19 V = [n] := {1, . . . , n}, *E ⊆ V × V*. Each agent i ∈ [n] has access to a continuously differentiable
(possibly non-convex) local objective function fi: R 20 d → R and maintains a local decision variable xi ∈ R
d. Denote x = [x
⊤ 1
, ..., x
⊤ n
]
⊤ ∈ R
nd 21 . Our aim is to tackle:
minx∈Rnd1n Pn i=1 fi(xi) s.t. xi = xj , ∀ (i, j) ∈ E. (1)
In other words, (1) seeks a x
⋆ ∈ R
dthat minimizes F(x) := (1/n)Pn i=1 22 fi(x). We are interested 23 in the stochastic optimization setting where each fi(xi) is given by (with slight abuse of notation)
fi(xi) := Eξi∼Pi
[fi(xi; ξi)] (2)
24 where Pi represents the i-th data distribution. Problem (1) is relevant to the distributed learning 25 problem especially in the decentralized case where a central server is absent. Prior works [Nedic and 26 Ozdaglar, 2009, Lian et al., 2017, Nedic et al., 2017, Qu and Li, 2017] demonstrated that *decentralized* 27 algorithms can tackle (1) efficiently through repeated message exchanges among the neighbors and 28 local stochastic gradient updates. 29 Towards an efficient decentralized algorithm for (1), an important direction is to consider a *time* 30 *varying graph topology* setting where the *active edge set* in G changes over time. This is a generic 31 setting covering cases when the communication links are unreliable, or the agents choose not to 32 communicate in a certain round (a.k.a. local updates) [Koloskova et al., 2019a, Nadiradze et al., 2021].

Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Do not distribute.

Prior Works SG TV w/o BH Rate

Prox-GPDA [Hong et al., 2017] ✗ ✗ ✓ Asympt. NEXT [Lorenzo and Scutari, 2016] ✗ ✓ ✓ Asympt.

DSGD [Koloskova et al., 2020] ✓ ✓ ✗ O(σ/√nT)

Swarm-SGD [Nadiradze et al., 2021] ✓ ✓ ✗ O(σ

2/

√T)

CHOCO-SGD [Koloskova et al., 2019a] ✓ ✗

‡ ✗ O(σ/√nT)

Decen-Scaffnew [Mishchenko et al., 2022] ✓ ✗

† ✓ O(σ/√nT)

Local-GT [Liu et al., 2024] ✓ ✗

† ✓ O(σ/√nT)

LED [Alghunaim, 2024] ✓ ✗

† ✓ O(σ/√nT)

FSPDA-SA (**This Work**) ✓ ✓ ✓ O(σ/√nT)

FSPDA-STORM (**This Work**) ✓ ✓ ✓ O(σ

2/3/T2/3)

Table 1: Comparison of decentralized algorithms for **non-convex** optimization. In the table, 'SG' is
'Stochastic Gradient', 'TV' is 'Time Varying Graph', 'w/o BH' is 'Without Bounded Heterogeneity',
and 'Rate' is the expected squared gradient norm E[∥∇F(x¯)∥
2] after T iterations. Note that σ 2is the variance of stochastic gradient. ‡CHOCO-SGD incorporates broadcast gossip as a special case of compression. †ProxSkip, Local-GT, LED consider local updates with periodic communication.

33 By assuming that a random topology is drawn at each iteration, the convergence of decentralized 34 stochastic gradient (DSGD) has been studied in [Lobel and Ozdaglar, 2010, Nadiradze et al., 2021] 35 and is later on unified by [Koloskova et al., 2020] with tighter bounds for local updates, periodic 36 sampling, etc. An alternative [Ram et al., 2010] is to analyze DSGD for the B-connectivity setting 37 which requires the union of every B consecutive time varying topologies to yield a connected graph. 38 Nevertheless, these works focused on vanilla DSGD that may have slow convergence (in transient 39 stage) and is limited to bounded data heterogeneity. The prior restrictions can be relaxed using 40 advanced algorithms such as gradient tracking [Qu and Li, 2017], EXTRA [Shi et al., 2015] and 41 primal-dual framework [Hong et al., 2017, Hajinezhad and Hong, 2019, Yi et al., 2021].

42 As noted by [Koloskova et al., 2021], analyzing the convergence of sophisticated algorithms with time 43 varying topology, such as gradient tracking [Qu and Li, 2017] is challenging due to the non-symmetric 44 product of two (or more) mixing matrices. Existing works considered various restrictions on the time varying topology G
(t) = (V, E
(t)
45 ) and/or the problem (1): [Koloskova et al., 2021, Liu et al.,
2024] studied gradient tracking with local updates that essentially takes E 46 (t) = E periodically and E 47 (t) = ∅ otherwise, also see [Mishchenko et al., 2022, Guo et al., 2023, Alghunaim, 2024] for a 48 similar result and note that such algorithms require extra synchronization overhead; [Kovalev et al.,
2021, 2024] considered a setting where G
(t)
49 is connected for any t; [Nedic et al., 2017, Li and Lin, 50 2024] focused on (accelerated) gradient tracking with deterministic gradient when F(x) is (strongly) 51 convex; [Lorenzo and Scutari, 2016] also considered deterministic gradient with possibly non-convex 52 F(x) but only provides asymptotic convergence guarantees; [Lei et al., 2018, Yau and Wai, 2023] 53 considered asymptotic convergence guarantees in the case of strictly (or strongly) convex F(x). We 54 provide a non-exhaustive list summarizing the convergence of existing works in Table 1.

55 The above discussion highlights a gap in the existing literature —
56 *Is there any algorithm that achieves fast convergence on time varying (random) topology?* 57 This paper gives an affirmative answer through developing the Fully Stochastic Primal Dual Algorithm 58 (FSPDA) framework that leads to efficient decentralized algorithms tackling (1) in its general form. 59 The framework features the design of a new stochastic augmented Lagrangian function. 60 As pointed out by [Chang et al., 2020], many decentralized algorithms (including gradient tracking) 61 can be interpreted as primal-dual algorithms finding a saddle point of the augmented Lagrangian func62 tion. However, its extension to time varying topology is not straightforward due to the inconsistency 63 in dual variables updates. To overcome this challenge, we propose a stochastic equality constrained 64 reformulation of (1) to model randomness in topology. Then, the latter yields a stochastic augmented 65 Lagrangian function. Applying stochastic approximation (SA) to solve the latter leads to the FSPDA 66 framework. Our contributions are 67 - We propose two new algorithms: (i) FSPDA-SA is derived by vanilla SA that applies primal-dual 68 stochastic gradient descent-ascent on the stochastic augmented Lagrangian, (ii) FSPDA-STORM uses 69 an additional control variate / momentum term to reduce the drift term's variance in a recursive 70 manner. Both algorithms are fully stochastic as the random time varying topology is treated as 71 a part of randomness. Additionally, our framework supports sparsified communication, i.e., the 72 agents can choose to communicate a subset of primal coordinates at each iteration.

73 - We show that after T iterations, FSPDA-SA (resp. FSPDA-STORM) finds in expectation a solution whose squared gradient norm is O(1/
√T) (resp. O(1/T2/3 74 )). The convergence analysis is derived 75 from a new Lyapunov function design that involves an unsigned inner product term and incorporates 76 a variance condition on the random time varying topologies. Interestingly, we show empirically 77 that using momentum in dual updates benefits the consensus error convergence.

78 - We also demonstrate that both FSPDA-SA and FSPDA-STORM can be implemented in a fully asyn79 chronous manner, i.e., the agents can communicate and compute at different time slots, and supports 80 local update as the algorithms allow for arbitrary time varying topology. That said, we remark that 81 the convergence rates with local updates of FSPDA-SA and FSPDA-STORM are only suboptimal.

82 We provide numerical experiments to show that FSPDA-SA and FSPDA-STORM outperform existing 83 algorithms in terms of iteration and communication complexity.

Notations. Let W ∈ R
d×d 84 be a symmetric (not necessarily positive semidefinite) matrix, the W-
weighted (semi) inner product of vectors a, b ∈ R
dis denoted as ⟨a | b⟩W := a 85 ⊤Wb. Similarly, the W-weighted (semi) norm is denoted by ∥a∥
2 86 W := ⟨a | a⟩W. The subscript notation is omitted for I-weighted inner products. For any square matrix X, (X)
†
87 denotes its pseudo inverse.

## 88 **2 The Fully Stochastic Primal Dual Algorithm (**Fspda**) Framework**

89 This section develops the FSPDA framework for tackling (1) and describes two variants of the framework leading to decentralized stochastic optimization of (1). Let Ae *∈ {−*1, 0, 1}
|E|×n 90 be an incidence matrix of G. By defining A = Ae ⊗ Id *∈ {−*1, 0, 1}
|E|d×nd 91 , we observe that the consensus 92 constraint in (1) is equivalent to Ax = 0. 93 Our first step is to model the randomness in the time varying topology using the random variable 94 (r.v.) ξa ∼ Pa. For each realization ξa, we define the random incidence matrix A(ξa) := I(ξa)A ∈
{−1, 0, 1}
|E|d×nd where I(ξa) ∈ {0, 1}
|E|d*×|E|*d 95 is a binary diagonal matrix. In addition to selecting 96 each edge of G randomly, I(ξa) selects a random subset of d coordinates. As we will see later, this 97 allows our approach to simultaneously achieve random sparsification for communication compression.

Assume that Eξa∼Pa 98 [I(ξa)] is a positive diagonal matrix, (1) is equivalent to:
minx∈Rnd1n Pn i=1 Eξi∼Pi
[fi(xi; ξi)] s.t. Eξa∼Pa
[A(ξa)] x = 0. (3)
99 Denote ξ = (ξ1, . . . , ξn, ξa), FSPDA hinges on the following *augmented Lagrangian* function of (3):
L(x,λ) := Eξ[L(x,λ; ξ)] with L(x,λ; ξ) := Pn i=1 fi(xi; ξi) + ˜η ⟨λ | A(ξa)x⟩ +
γ˜
2
∥A(ξa)x∥
2,(4)
100 where η >˜ 0, γ > ˜ 0 are penalty parameters. It can be verified that the saddle points of L(x,λ) 101 correspond to the KKT points of (1) [Bertsekas, 2016]. For brevity, in the rest of this paper, we may 102 drop the subscript in ξ whenever the notation is clear from the context.

103 FSPDA is developed from applying stochastic approximation (SA) to seek a saddle point of (4). By recognizing A(ξ) 104 ⊤A(ξ) = A⊤A(ξ), we consider the stochastic gradients:
∇xL(x,λ; ξ) := ∇f(x; ξ) + ˜ηA⊤λ + ˜γA⊤A(ξ)x, ∇λL(x,λ; ξ) := ˜ηA(ξ)x, (5)
where ∇f(x; ξ) = [∇f1(x1; ξ1); *. . .* ; ∇fn(xn; ξn)] ∈ R
nd 105 . Notice that to facilitate algorithm 106 development, we have taken a deterministic A for the term in ∇xL related to λ. Now observe the ith 107 d-dimensional block of A⊤A(ξ)x which can be aggregated within Ni(ξ) the neighborhood of the 108 ith agent as:-A⊤A(ξ)xi
=Pj∈Ni(ξ) Cij (ξ)(xj − xi), (6)
where Cij (ξ) ∈ {0, 1}
d×d 109 is diagonal and depends on the selected coordinates for the edge (*i, j*) 110 under randomness ξ. Eq. (6) *only* relies on xj from neighbor j that is connected on the time varying 111 topology G(ξ). For illustration, an example of the above random graph model is given by Figure 3 in 112 Appendix A. Importantly, (5) shows that with the stochastic augmented Lagrangian function, the time 113 varying topology can be treated implicitly as a part of the randomness in the stochastic primal-dual 114 gradients. The framework is thus described as being *fully stochastic* as in [Bianchi et al., 2021], and 115 departs from [Liu et al., 2024, Alghunaim, 2024] that treat the topology as fixed during the derivation 116 of primal-dual algorithm(s). From (5), (6), we derive two variants of FSPDA. 117 FSPDA-SA **Algorithm.** The first variant of FSPDA is derived from a direct application of stochastic 118 gradient descent-ascent (SGDA) updates. Take α > 0*, β >* 0 as the step sizes, we have x t+1 = x t − α∇xL(x t,λ t; ξ t), λ t+1 = λ t + β∇λL(x t,λ t; ξ t). (7)
Taking the variable substitution λb := A⊤ 119 λ yields the following recursion:

| FSPDA-SA: for any t ≥ 0 and any i ∈ [n], x t+1 i = x t i − α∇fi(x t i ; ξ t ) − ηλbt i + γ P j∈Ni(ξ t a ) Cij (ξ t a )(x t j − x t i i λbt+1 i = λbt i + β P j∈Ni(ξ t a ) Cij (ξ t a )(x t j − x t i   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

120 Note that x 0,λb0 121 can be initialized arbitrarily.

122 FSPDA-STORM **Algorithm.** The second variant of FSPDA reduces the variance of the stochastic 123 gradient term in (5) using the recursive momentum variance reduction technique [Cutkosky and 124 Orabona, 2019]. Herein, the key idea is to utilize a control variate in estimating the (primal-dual)
125 gradients of L(x,λ). Take *α, β >* 0 and ax, aλ ∈ [0, 1] as the momentum parameters, we have x t+1 = x t − αmtx,λ t+1 = λ t + βmtλ 126 as the primal-dual updates, and mt+1 x = ∇xL(x t+1,λ t+1; ξ t+1) + (1 − ax)(mtx − ∇xL(x t,λ t; ξ t+1)),

$$(9)$$
$$t+1\choose\lambda$$

mt+1 λ = ∇λL(x t+1,λ t+1; ξ t+1) + (1 − aλ)(mtλ − ∇λL(x t,λ t; ξ t+1)).(9)
The aim of mt+1 xis to estimate ∇xL(x t+1,λ t+1 127 ). Now, instead of the straightforward estimator
∇xL(x t+1,λ t+1; ξ t+1), we include an extra zero-mean term mtx − ∇xL(x t,λ t; ξ t+1 128 ) to reduce 129 the variance of the stochastic gradient estimation. The latter is a control variate that is computed 130 recursively. Particularly, it has been shown in [Cutkosky and Orabona, 2019] that it can effectively 131 reduce variance with a carefully designed parameter ax, provided that the stochastic gradient map 132 satisfies a mean-square Lipschitz condition. We summarize the algorithm as follows.

FSPDA-STORM: for any t ≥ 0 and any i ∈ [n],
x t+1 i = x ti − αmtx,i, (10a)
λbt+1 i = λbt i + βmtλ,i, (10b)
mt+1 x,i = (1−ax)-mtx,i + ∇fi(x t i; ξ t+1 i) − ηλbti + γPj∈Ni(ξ t+1 a ) Cij (ξ t+1 a)(x tj − x ti
)(10c)
+ ∇fi(x t+1 i; ξ t+1 i) − ηλbt+1 i + γPj∈Ni(ξ t+1 a ) Cij (ξ t+1 a)(x t+1 j − x t+1 i)
mt+1 λ,i = (1 − aλ)-mtλ,i +Pj∈Ni(ξ t+1 a ) Cij (ξ t+1 a)(x tj − x ti
)(10d)
$(1-\alpha)\left[\mathbf{m}_{\lambda},i+\mathbf{j}\mathbf{e}_{\lambda}(\mathbf{e}_{\lambda}^{t+1})\mathbf{e}_{\lambda}\right]$  $+\mathbf{j}\mathbf{e}_{\lambda}(\mathbf{e}_{\lambda}^{t+1})\mathbf{e}_{\lambda}(\mathbf{e}_{\lambda}^{t+1})(\mathbf{e}_{\lambda}^{t+1})$
t+1 i)
133 Note that to achieve the theoretical performance (see later in Sec. 3), x 0,λb0, m0x, m0λ 134 shall be initialized as x 0 i = x¯
0, λb0 i = (α/η)n
−1(∇F(x¯
0) − ∇fi(x¯
0)), m0x,i = ∇F(x¯
0), m0 135 λ,i = 0 according to (23). We remark that a simple initialization choice λb0 = m0x,i = m0 136 λ,i = 0 works well 137 in practice. 138 Both FSPDA-SA and FSPDA-STORM are decentralized algorithms that can be implemented on random 139 time varying topology, and support randomized sparisification for further communication compres140 sion. The key is to observe that in P
(8), (10), the only information required for agent i is to obtain j∈Ni(ξ ta) Cij (ξ ta
)(x tj − x ti
), and in addition Pj∈Ni(ξ ta) Cij (ξ ta
)(x t−1 j − x t−1 i 141 ) for FSPDA-STORM,
142 at iteration t.

## 143 **2.1 Implementation Details And Connection To Existing Works**

144 We discuss several features of the FSPDA algorithms and their connections to existing works. 145 **Local & Asynchronous Updates.** The *local update* scheme where each agent i is allowed to update its own local variables xi 146 ,λi for multiple iterations without a communication step is a 147 common practice in decentralized optimization [Liu et al., 2024, Li and Lin, 2024, Alghunaim, 2024, 148 Mishchenko et al., 2022]. As discussed before, such scheme can be seen as a special case of the FSPDA framework where the time varying topology E
(t)
149 is chosen such that the latter alternates between E
(t) = E and E 150 (t) = ∅. 151 Furthermore, FSPDA-SA allows for the general case of *asynchronous* updates. This is done so by taking the stochastic gradient as ∇fi(x ti; ξ t) = bi(ξ t) bi ∇fi(x ti; ξ t) such that bi(ξ t 152 ) ∈ {0, 1}
with E[bi(ξ t 153 )] = 1/bi for some constant bi > 0. Detailed discussions for a fully asynchronous 154 implementation of FSPDA-SA can be found in Appendix A.

Connection to Existing Works. Evaluating x t+2 − x t+1 155 from the FSPDA-SA sequence and observe 156 that the combination of (8a) and (8b) is equivalent to the second order recursion:

$$\begin{array}{c}{{{\bf x}^{t+2}=2\left({\bf I}-\frac{\gamma}{2}{\bf A}^{\top}{\bf A}(\xi^{t+1})\right){\bf x}^{t+1}-\left({\bf I}-(\gamma-\eta\beta){\bf A}^{\top}{\bf A}(\xi^{t})\right){\bf x}^{t}}}\\ {{\qquad-\alpha\left(\nabla{\bf f}({\bf x}^{t+1};\xi^{t+1})-\nabla{\bf f}({\bf x}^{t};\xi^{t})\right).}}\end{array}$$
$$(11)$$

This reduces the FSPDA-SA recursion into a primal-only sequence by eliminating the dual sequence λ t 157 .

158 In the deterministic optimization setting when A(ξ) ≡ A and ∇f(x; ξ) ≡ ∇f(x), (11) is equivalent to the EXTRA algorithm [Shi et al., 2015] using the mixing matrix W = I − γDiag(W1 ˜ ) + γW˜ 159 where W˜ 160 is the 0-1 adjacency matrix of G. Here, with an appropriate choice of γ, W will be doubly 161 stochastic and satisfies the convergence requirement in [Shi et al., 2015]. Similar observations have 162 been made in [Nedic et al., 2017] for the gradient tracking and DIGing algorithms. 163 On the other hand, for stochastic optimization on random networks, (11) suggests each agent to keep 164 the current and previous iterates received from neighbors in the corresponding time varying topology. 165 In this case, (11) yields an extension of the EXTRA/GT algorithms to time varying topology.

## 166 **3 Convergence Analysis Of** Fspda

167 This section presents the convergence rate analysis of FSPDA for (1). Unless otherwise specified, we 168 focus on the case with smooth but possibly non-convex objective function. Specifically, we consider:
Assumption 3.1. *Each* fi 169 is L-smooth, i.e., for i = 1*, . . . , n*,
$$\|\nabla f_{i}(\mathbf{x})-\nabla f_{i}(\mathbf{y})\|\leq L\|\mathbf{x}-\mathbf{y}\|\;\forall\;\mathbf{x},\mathbf{y}\in\mathbb{R}^{d}.$$
d. (12)
There exists f⋆ > −∞ such that fi(x) ≥ f⋆ *for any* x ∈ R
d 170 . 171 Note this implies that the global objective function F(·) is L-smooth but possibly non-convex.

172 We further assume that the random network G(ξa) is connected in expectation, yet each realization 173 G(ξa) may not be connected. Let R = E [I(ξa)], this leads to the following property concerning the expected graph Laplacian matrix A⊤RA = E-A(ξa)
⊤A
174 . Defining the matrix K := (In −
11⊤ 175 /n) ⊗ Id, we have 176 **Assumption 3.2.** *There exists* ρmax ≥ ρmin > 0 and ρ¯max ≥ ρ¯min > 0 *such that* ρminK ⪯ A⊤RA ⪯ ρmaxK and ρ¯minK ⪯ A⊤A ⪯ ρ¯maxK. (13)
177 It holds that A⊤RAK = A⊤RA = KA⊤RA. The above assumption can be satisfied if G is 178 connected [Yi et al., 2021], [Yi et al., 2018, Lemma 2] and diag(R) > 0 such that each edge is selected with a positive probability. As an important consequence, if γ ≤ ρmin/ρ2max 179 , we have
∥(I − γA⊤RA)x∥
2 K ≤ (1 − γρmin)∥x∥
2 K, ∀ x ∈ R
nd.

180 We thus observe that the operator (I − γA⊤RA) serves a similar purpose as the mixing matrix 181 in a average consensus algorithms and ρmin can be interpreted as the spectral radius of G similar

$$(12)$$

$$(13)$$

$$\mathbf{K}.$$

to [Koloskova et al., 2020, Eq. (12)]. Moreover, if we define Q := (A⊤RA)
†
182 such that it holds
QA⊤RA = A⊤RAQ = K, Assumption 3.2 implies that ρ
−1
maxK ⪯ Q ⪯ ρ
−1
183 minK.
184 Next we consider several assumptions on the noise variance of the random quantities in FSPDA:
Assumption 3.3. *For any fixed* xi ∈ R
d
185 , i ∈ [n], there exists σi ≥ 0 *such that*
Eξi∼Pi[∥∇fi(xi; ξi) − ∇fi(xi)∥
$$\mathbf{\Phi}_{i}(\mathbf{x}_{i})\|^{2}]\leq\sigma_{i}^{2}.$$
i. (14)
To simplify notations, we define σ¯
2:= (1/n)Pn
i=1 σ
2 i
186 .
Assumption 3.4. *For any fixed* x ∈ R
nd 187 , there exists σA ≥ 0 *such that*
[∥A(ξa)
⊤Ax − A⊤RAx∥
2] ≤ σ
2A∥x∥
2K. (15)
$\xi_a\!\sim\!\mathbb{P}_a\left[||A\right]$

Assumption 3.3 is standard. Meanwhile for Assumption 3.4, the variance term σ 2 188 A measures the 189 quality of the random topology G(ξa) in approximating the expected graph Laplacian A⊤RA. The latter is important as it contributes to the variance in the drift term of FSPDA. Observe that σ 2 190 A
191 decreases with the proportion of edges selected in each random subgraph G(ξa).

192 To facilitate our discussions, we define the following quanitites:
x¯
t:= 1n Pn i=1 x ti,Pn i=1 ∥x ti − x¯
t∥
2 = ∥x t∥
2K. (16)
193 **Convergence of** FSPDA-SA. We summarize the convergence rate for FSPDA-SA as follows. The proof 194 can be found in Appendix C:
Theorem 3.5. Under Assumptions 3.1, 3.2, 3.3, 3.4. Suppose that the step sizes satisfy the conditions defined in (46). Then, for any T ≥ 1 *with the random stopping iteration* T ∼ Unif{0, ..., T − 1}*, the iterates generated by* FSPDA-SA *satisfy*

$$(14)$$
$$||{\bf\frac{2}{k}}.$$
$$(15)$$
$$(16)$$
$$\mathbb{E}\left[\|\nabla F(\bar{\mathbf{x}}^{\mathsf{T}})\|^{2}\right]\leq\frac{F_{0}-f_{*}}{\alpha T/8}+8\alpha\mathbb{C}_{\sigma}\frac{\bar{\sigma}^{2}}{n},$$  $$\mathbb{E}\left[\sum_{i=1}^{n}\|\mathbf{x}_{i}^{\mathsf{T}}-\bar{\mathbf{x}}^{\mathsf{T}}\|^{2}\right]\leq\frac{F_{0}-f_{*}}{\alpha\gamma\rho_{\min}T/8}+\frac{8\alpha^{2}\mathbb{C}_{\sigma}\bar{\sigma}^{2}}{\alpha\gamma\rho_{\min}n},$$  _for any $\mathbf{a}>0$, where $F_{0}$, $\mathbb{C}_{\sigma}$ are defined in (44), (50)._
$$(17)$$
, (17)
, (18)
195
Setting a = O(n/√Tσ¯
2), α =pn/(Tσ¯
2 196 ) (and assuming σ >¯ 0), we have
$\left(18\right)^2$
$$\sqrt{n/(T\bar{\sigma}^{2})}\;(\mathrm{and~assume})$$
$\frac{1}{2}$
$\mathbf{v}=\mathbf{v}$

$$\operatorname{\mathbb{E}}\left[\|\nabla F({\bar{\mathbf{x}}}^{\mathsf{T}})\|^{2}\right]={\mathcal{O}}\left({\bar{\sigma}}/{\sqrt{n T}}\right),$$

√nT, (19)

$\frac{1}{2}$  . 
197 which is the same *asymptotic convergence rate* as a centralized SGD algorithm that takes n stochastic 198 gradient samples uniformly from each agent, i.e., linear speedup [Lian et al., 2017]. Also, using a = 1, the consensus error converges as a rate of EPn i=1 ∥x T
i − x¯
T∥
2= O(n 2σ 2Aρmax/(T ρ2min 199 ))
200 under the same step size choice used in (19). Notice that for T ≫ 1, the effect of random topology 201 only degrades the convergence of consensus error, keeping the transient rate in (19) unaffected. If the gradients are deterministic (σ¯ = 0), setting a = (L
2η∞ρmin)
1/3 202 , α = α∞ will yield a better convergence rate as E-∥∇F(x¯
T)∥
2= O(σ 4A
√
203 n/T). Without a transient phase, the error due to random graph and coordinate sparsification is persistent through σ 4 204 A in the above convergence rate.

205 We further show that the convergence of FSPDA-SA can be accelerated if the objective function of (1)
206 satisfies the Polyak-Lojasiewicz (PL) condition:
Assumption 3.6. There exists a constant µ > 0 such that 2µ(F(x) − f⋆) *≤ ∥∇*F(x)∥
2, ∀x ∈ R
d 207 .

208 Assumption 3.6 includes strongly convex functions as a special case, but also includes other non209 convex functions; see [Karimi et al., 2016]. We observe:

Corollary 3.7. Suppose the assumptions and step size conditions in Theorem 3.5 hold. Furthermore, with Assumption 3.6, there exists δ ∈ (0, 1) *such that for any* t ≥ 0,
$$\mathbb{E}_{t}[F_{t+1}-f_{\star}]\leq(1-\delta)(F_{t}-f_{\star})+\mathbb{C}_{\sigma}\alpha^{2}\bar{\sigma}^{2},$$
2/n (20)
for Ft, Cσ *defined in* (44), (70)*, and* δ = min{αµ/4, γρmin/16*, ηβ/*(3ρmin)*, η/*12}.

210

The proof can be found in Appendix C.6. By setting α = c ln(T)/(n 2 211 T) in (20), with a carefully 212 chosen c and a sufficiently large T such that α ≤ α∞, we can ensure that E
-F(x¯
T) − f⋆ + ∥x T∥
2 K
= Oσ¯
2ln(T)/(µnT)(21)
In the case of deterministic gradient, i.e., σ¯ 213 2 = 0, by setting α = α∞, (20) ensures a linear convergence rate of E-F(x¯
T) − f⋆ + ∥x T ∥
2K
= O((1 − δ)
T
214 ), which shows that the performance 215 of FSPDA-SA is on par with [Nedic et al., 2017, Xu et al., 2017], despite it only requires one round of 216 (sparsified) transmission per iteration. 217 **Convergence of** FSPDA-STORM. To exploit the benefits of control variates, we need an additional 218 assumption on the stochastic gradient map:
219 **Assumption 3.8.** Each stochastic function fi(·; ξ) is Ls*-smooth in expectation, i.e., for* i = 1*, . . . , n*,
Eξ-∥∇fi(x; ξ) − ∇fi(y; ξ)∥
2≤ L
2 s ∥x − y∥
2 ∀ x, y ∈ R
d. (22)
220 The above assumption is also known as the mean-square smoothness condition, see [Cutkosky 221 and Orabona, 2019], which is strictly stronger than Assumption 3.1. We observe the following 222 convergence guarantee for FSPDA-STORM, whose proof can be found in Appendix D.

Theorem 3.9. Under Assumptions 3.1, 3.2, 3.3, 3.4, 3.8. Suppose that the step sizes satisfy the conditions in (184) - (214). Then, for any T ≥ 1 *with the random stopping iteration* T ∼ Unif{0, ..., T − 1}*, the iterates generated by* FSPDA-STORM *satisfy*

$$\mathbb{E}\left[\|\nabla F(\bar{\mathbf{x}}^{\mathsf{T}})\|^{2}\right]\leq\frac{F_{0}-f_{\star}}{T\alpha/4}+\frac{(\mathbf{e}\cdot2a_{x}^{2}+\mathbf{f}\cdot4a_{x}^{2}n)\bar{\sigma}^{2}}{\alpha/4},$$ $$\mathbb{E}\left[\sum_{i=1}^{n}\|\mathbf{x}_{i}^{\mathsf{T}}-\bar{\mathbf{x}}^{\mathsf{T}}\|^{2}\right]\leq\frac{F_{0}-f_{\star}}{T\mathbf{a}\gamma\rho_{\min}/8}+\frac{(\mathbf{e}\cdot2a_{x}^{2}+\mathbf{f}\cdot4a_{x}^{2}n)\bar{\sigma}^{2}}{\mathbf{a}\gamma\rho_{\min}/8},$$
$$2/3\,/T^{2/3})\,.$$
α/4, (23)
, (24)
_where the constants $F_{0}$, $\mathbf{a},\mathbf{e},\mathbf{f}$ are defined in (110)._
223 Setting α = O(¯σ
−2/3T
−1/3), η = O(n), γ = O(T
−1/3), β = O(n
−1T
−2/3 224 ), ax =
O(¯σ
−4/3T
−2/3), aλ = O(T
−1/3), f = O(n
−1T
1/3 225 ) (see (111) - (117)), and initializing the algorithm such that ∥v 0∥
2K = O(T
−2/3), ∥m0x − (1/n)1
⊤⊗∇f(x 0)∥
2 = O(T
−1/3 226 ) and
∥m0x − ∇xL(x 0,λ 0)∥
2 = O(T
−1/3 227 ), we have E
-∥∇F(x¯
T)∥
2= Oσ¯
2/3/T2/3. (25)
228 In regard to the order of σ¯ and T, provided that n is small, the convergence rate of FSPDA-STORM
229 matches the lower bound [Arjevani et al., 2023] for non-convex functions under the same smoothness 230 assumption. Moreover, by the same choice of step sizes, the consensus error converges at the rate of E
Pn i=1 ∥x T
i − x¯
T∥
2= O(¯σ 2/3nρ−1 minT
−2/3 231 ). We remark that in (25), the rate remains constant as 232 n increases such that FSPDA-STORM does not offer the same *linear speedup* observed in Theorem 3.5 233 for FSPDA-SA. Nevertheless, as T ≫ 1, the rate of FSPDA-STORM will surpass that of FSPDA-SA and 234 other decentralized algorithms on time varying topologies. 235 Lastly, we provide detailed discussions on the convergence rates above, e.g., transient time, effects of 236 random topology, etc., in Appendix B.

## 237 **3.1 Insight From Analysis: Fixed Point Iteration Of** Fspda-Sa

From (8a), the following recursive relationship holds for x¯
t: using the relation 1 238 ⊤A⊤ = 0, we have x¯
t+1 = x¯
t −
α n Pn i=1 ∇fi(x ti; ξ ti). (26)
This shows that the evolution of {x¯
t 239 }t≥0 is similar to that of 'centralized' SGD applied on (1) except 240 that the local gradients are evaluated on the local iterates. However, it is still not straightforward to analyze the convergence of FSPDA-SA as the update of x tinvolves the dual variable λ t 241 which lacks 242 an intuitive interpretation for constructing the right Lyapunov function.

243 To this end, we study the fixed point(s) of (8) to gain insights. Suppose that for some t⋆, the fixed point conditions E[λ t⋆+1 | ξ
:t⋆ ] = λ t⋆ , E[x t⋆+1 | ξ
:t⋆ ] = x t⋆ 244 hold. Since R is a diagonal matrix 245 with positive diagonal elements, we observe E[λ t⋆+1 | ξ
:t⋆] = λ t⋆ ⇐⇒ RAxt⋆ = 0 ⇐⇒ Axt⋆ = 0, (27)
246 On the other hand, the primal update yields

$\mathbb{E}[\mathbf{x}^{t_{*}+1}\mid\xi^{:t_{*}}]=\mathbf{x}^{t_{*}}-\alpha\mathbf{V}\mathbf{f}(\mathbf{x}^{t_{*}})-\eta\mathbf{A}^{\top}\mathbf{\lambda}^{t_{*}}$.  $\mathbf{x}^{t_{*}}$ at the fixed point (due to (27)), by the case 
$$(28)$$
$\uparrow$  . 
t⋆. (28)
Since x
t⋆
1 = x
t⋆
2 = *· · ·* = x
n
247 at the fixed point (due to (27)), by the consensus condition across two
248 time steps, it implies
$$(29)$$
$$\mathbb{E}[\mathbf{x}^{t_{*}+1}\mid\xi^{\perp_{*}}]-\mathbf{x}^{t_{*}}=(\mathbf{1}\otimes\mathbf{I}_{d})(\tilde{\mathbf{x}}^{t_{*}+1}-\tilde{\mathbf{x}}^{t_{*}})$$ $$\iff\alpha\nabla\mathbf{f}(\mathbf{x}^{t_{*}})+\eta\mathbf{A}^{\top}\boldsymbol{\lambda}^{t_{*}}=\frac{\alpha}{n}(\mathbf{11}^{\top}\otimes\mathbf{I}_{d})\nabla\mathbf{f}(\mathbf{x}^{t_{*}})$$ $$\iff\eta\mathbf{A}^{\top}\boldsymbol{\lambda}^{t_{*}}=\alpha\left(\frac{1}{n}\mathbf{11}^{\top}-\mathbf{I}_{n}\right)\otimes\mathbf{I}_{d}\;\nabla\mathbf{f}((\mathbf{1}\otimes\mathbf{I})\tilde{\mathbf{x}}^{t_{*}}).$$
From (29), we see that λbt
249 shall converge to the difference between global and local gradient. Inspired
250 by the above, to facilitate the analysis later, we define
$$\mathbf{v}^{t}:=\mathbf{A}^{\top}\boldsymbol{\lambda}^{t}+{\frac{\alpha}{\eta}}\nabla\mathbf{f}((\mathbf{1}\otimes\mathbf{I})\bar{\mathbf{x}}^{t}),$$
t), (30)
for any t ≥ 0. In particular, we see that ∥v t∥
2 251 K measures the violation of (29) in tracking the average 252 deterministic gradient using the dual variables. The latter will be instrumental in analyzing the 253 consensus error bound, as revealed in Lemma C.2.

## 254 **4 Numerical Experiments**

255 This section reports the numerical experiments on practical performance of FSPDA. For the time 256 varying topology, we take an extreme setting where for each realization G(ξa), only one edge will 257 be selected uniformly at random from G. We evaluate the performance with the worst-agent metric, i.e., we present the training loss as maxi∈[n] F(x ti 258 ), and the stationarity/gradient-norm measure as maxi∈[n] ∥∇F(x t i)∥
2 259 . This captures the worst-case of the solutions produced by the algorithms.

Unless otherwise specified, all algorithms are initialized with x 0 i = x¯
0 260 , and for FSPDA we initialize λb0 = m0x,i = m0 261 λ,i = 0, and the stochastic gradients are estimated with a batch size of 256. In the 262 interest of space, omitted details and hyperparameters of the experiments can be found in Appendix F. 263 **MNIST Experiments.** The first set of experiments considers a moderate-scale setting of training a 264 one hidden layer feed-forward neural network with 100 hidden neurons (total number of parameters 265 d = 79,510) on the MNIST dataset with m = 60, 000 samples of 784-dimensional features. 266 In the first experiment, we consider the static topology G as an Erdos-Renyi graph with connectivity of 267 p = 0.5 and n = 10 agents. We compare the proposed FSPDA-SA, FSPDA-STORM with six benchmark 268 algorithms utilizing different types of time-varying topology. Among them, DSGD [Koloskova et al., 269 2020] and Swarm-SGD [Nadiradze et al., 2021] use the general time varying topology setting as FSPDA
270 where each edge of G(ξa) is active uniformly at random, in addition to random sparsification used 271 FSPDA-SA and adaptive quantized used in Swarm-SGD; CHOCO-SGD [Koloskova et al., 2019b] takes 272 G(ξa) as an broadcasting subgraph where one agent selects all his/her neighbors; Decen-Scaffnew 273 [Mishchenko et al., 2022], LED [Alghunaim, 2024], and K-GT [Liu et al., 2024] utilize local updates 274 where G(ξa) is either taken as an empty topology, or as the static topology G. We configure these 275 algorithms such that they have the same communication cost (in terms of bits transmitted over 276 network) *on average*. For instance, the local update algorithms (Decen-Scaffnew, LED, K-GT)
only communicate once using G every O
|E|d k 277 iterations to match the communication cost of 278 k-coordinate sparse one-edge random graph used in FSPDA.

279 The local objective function held by each agent is the cross-entropy classification loss on a local dataset with mi = 6000 samples, plus a regularization loss λ2
∥xi∥
2 with λ = 10−4 280 , where xi are the 281 weight parameters of the feed-forward neural network classifier. We split the training set into n = 10 282 disjoint sets such that each set contains only one class label and assign each set to one agent as its 283 local dataset. Note that as we do not shuffle the data samples across local datasets, the local objective 284 function held by different agents will become highly heterogeneous. 285 Fig. 1 compares the squared gradient norm, training loss, consensus error of the benchmarked algo286 rithms. We first note that both FSPDA algorithms have significantly outperformed DSGD, Swarm-SGD 287 on the general time varying topology as well as CHOCO-SGD. Meanwhile, the performance of FSPDA
288 is comparable to the local update algorithms Decen-Scaffnew, LED, K-GT. Notice that the latter

FSPDA-SA 10.0% coordinates FSPDA-STORM 6.7% coordinates K-GT LED
Decen-Scaffnew DSGD
Swarm-SGD 8-bits CHOCO-SGD 10.0% coordinates 10 10 10 11 10 12 Bits Transmitted 10 3 10 2 10 1 10 0 10 10 10 11 10 12 Bits Transmitted 10 4 10 3 10 2 Gra die nt N
orm S
qua red Conse nsu s Erro r Trai n Los s 10 0 10 10 10 11 10 12 Bits Transmitted 10 1 FSPDA-SA 10.0% coordinates FSPDA-SA 1.0% coordinates FSPDA-SA 0.1% coordinates CHOCO-SGD 10.0% coordinates CHOCO-SGD 1.0% coordinates Swarm-SGD 8-bits 0 100000 200000 Iteration 0 20 40 60 80 10 10 10 12 10 14 Bits Transmitted 0 20 40 60 80 10 10 10 12 10 14 Bits Transmitted 5.0 7.5 10.0 12.5 15.0 Con sen sus Erro r 10 10 10 12 10 14 Bits Transmitted 10 3 10 1 10 1 Tes t Ac cura cy Test Acc ura cy Trai n Loss
289 require additional synchronization steps which may not be suitable for random networks. Lastly, we notice that as T ≫ 1, FSPDA-STORM can slightly outperform FSPDA-SA due to its O(1/T2/3 290 ) rate as 291 shown in our analysis. We further expand the experiments by a series of ablation studies over data 292 heterogeneity, sparsity levels, graph topologies, gradient noise and dual momentum in Appendix E. 293 **Imagenet Experiments.** The second set of experiments consider a large-scale setting for training a 294 Resnet-50 network (total number of parameters d = 25,557,032) on the Imagenet dataset (training 295 dataset of 1,281,168 images from 100 classes, re-scaled and cropped to 256 × 256 image dimensions). 296 We consider cross-entropy classification loss plus the same L2 norm regularization loss as in the 297 previous setup. We split the dataset across a network of n = 8 nodes where the static graph G is taken 298 as the fully connected topology. The performance metrics are measured at the network average iterate x¯
t 299 . Inspired by [Loshchilov and Hutter, 2016, Eq. (5)] we adopt a cosine learning rate scheduling 300 with 5 epochs of linear warm up for every algorithm. In particular, the step sizes *α, η* of FSPDA-SA
301 are scheduled simultaneously such that αt/ηt remains constant, as illustrated in Appendix F. We 302 draw a batch of 128 samples to estimate the stochastic gradient. 303 We focus on the communication efficiency and only compare FSPDA-SA, CHOCO-SGD, Swarm-SGD 304 in this experiment due to limited resources. The results are reported in Figure 2 that compare the 305 test accuracy and training loss against iteration number and bits transmitted. When compared with 306 CHOCO-SGD, FSPDA-SA achieves almost the same accuracy using one-edge random graphs with 307 at least 100x reduction in communication cost on 100 epoch training. Also notice that further 308 compressing the communication to 0.1% sparse coordinates in FSPDA-SA requires more training 309 epochs to recover the same level of accuracy. 310 **Conclusions.** This paper proposed a fully stochastic primal dual gradient algorithm (FSPDA) frame311 work for decentralized optimization over arbitrarily time varying random networks. We utilize a new 312 stochastic augmented Lagrangian function and apply SA to search for its saddle point. We develop 313 two algorithms, one is by plain SA (FSPDA-SA), and one uses control variates for variance reduction 314 (FSPDA-STORM). We prove that both algorithms achieve state-of-the-art convergence rates, while 315 relaxing assumptions on both bounded heterogeneity and the type of time varying topologies.

## 316 **References**

317 Sulaiman A Alghunaim. Local exact-diffusion for decentralized optimization and learning. *IEEE* 318 *Transactions on Automatic Control*, 2024. 319 Yossi Arjevani, Yair Carmon, John C Duchi, Dylan J Foster, Nathan Srebro, and Blake Woodworth. 320 Lower bounds for non-convex stochastic optimization. *Mathematical Programming*, 199(1): 321 165–214, 2023. 322 Dimitri Bertsekas. *Nonlinear Programming*, volume 4. Athena Scientific, 2016. 323 Pascal Bianchi, Walid Hachem, and Adil Salim. A fully stochastic primal-dual algorithm. Optimiza324 *tion Letters*, 15(2):701–710, 2021. 325 Tsung-Hui Chang, Mingyi Hong, Hoi-To Wai, Xinwei Zhang, and Songtao Lu. Distributed learning 326 in the nonconvex world: From batch data to streaming and beyond. *IEEE Signal Processing* 327 *Magazine*, 37(3):26–38, 2020.

328 Ashok Cutkosky and Francesco Orabona. Momentum-based variance reduction in non-convex sgd.

329 *Advances in neural information processing systems*, 32, 2019. 330 Luyao Guo, Sulaiman A Alghunaim, Kun Yuan, Laurent Condat, and Jinde Cao. Revisiting decen331 tralized proxskip: Achieving linear speedup. *arXiv preprint arXiv:2310.07983*, 2023. 332 Davood Hajinezhad and Mingyi Hong. Perturbed proximal primal–dual algorithm for nonconvex 333 nonsmooth optimization. *Mathematical Programming*, 176(1):207–245, 2019. 334 Mingyi Hong, Davood Hajinezhad, and Ming-Min Zhao. Prox-pda: The proximal primal-dual 335 algorithm for fast distributed nonconvex optimization and learning over networks. In *International* 336 *Conference on Machine Learning*, pages 1529–1538. PMLR, 2017. 337 Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin 338 Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Ad339 vances and open problems in federated learning. Foundations and trends® *in machine learning*, 340 14(1–2):1–210, 2021. 341 Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear convergence of gradient and proximal342 gradient methods under the polyak-łojasiewicz condition. In *Machine Learning and Knowledge* 343 *Discovery in Databases: European Conference, ECML PKDD 2016, Riva del Garda, Italy,* 344 *September 19-23, 2016, Proceedings, Part I 16*, pages 795–811. Springer, 2016.

345 Anastasia Koloskova, Tao Lin, Sebastian U Stich, and Martin Jaggi. Decentralized deep learning with 346 arbitrary communication compression. In *International Conference on Learning Representations*, 347 2019a. 348 Anastasia Koloskova, Sebastian Stich, and Martin Jaggi. Decentralized stochastic optimization and 349 gossip algorithms with compressed communication. In *International Conference on Machine* 350 *Learning*, pages 3478–3487. PMLR, 2019b. 351 Anastasia Koloskova, Nicolas Loizou, Sadra Boreiri, Martin Jaggi, and Sebastian Stich. A unified 352 theory of decentralized sgd with changing topology and local updates. In *International Conference* 353 *on Machine Learning*, pages 5381–5393. PMLR, 2020. 354 Anastasiia Koloskova, Tao Lin, and Sebastian U Stich. An improved analysis of gradient tracking 355 for decentralized machine learning. *Advances in Neural Information Processing Systems*, 34: 356 11422–11435, 2021. 357 Dmitry Kovalev, Elnur Gasanov, Alexander Gasnikov, and Peter Richtarik. Lower bounds and optimal 358 algorithms for smooth and strongly convex decentralized optimization over time-varying networks.

359 *Advances in Neural Information Processing Systems*, 34:22325–22335, 2021.

360 Dmitry Kovalev, Ekaterina Borodich, Alexander Gasnikov, and Dmitrii Feoktistov. Lower bounds and 361 optimal algorithms for non-smooth convex decentralized optimization over time-varying networks.

362 *arXiv preprint arXiv:2405.18031*, 2024. 363 Jinlong Lei, Han-Fu Chen, and Hai-Tao Fang. Asymptotic properties of primal-dual algorithm for 364 distributed stochastic optimization over random networks with imperfect communications. *SIAM* 365 *Journal on Control and Optimization*, 56(3):2159–2188, 2018. 366 Huan Li and Zhouchen Lin. Accelerated gradient tracking over time-varying graphs for decentralized 367 optimization. *Journal of Machine Learning Research*, 25(274):1–52, 2024.

368 Xiangru Lian, Ce Zhang, Huan Zhang, Cho-Jui Hsieh, Wei Zhang, and Ji Liu. Can decentralized 369 algorithms outperform centralized algorithms? a case study for decentralized parallel stochastic 370 gradient descent. *Advances in neural information processing systems*, 30, 2017. 371 Yue Liu, Tao Lin, Anastasia Koloskova, and Sebastian U Stich. Decentralized gradient tracking with 372 local steps. *Optimization Methods and Software*, pages 1–28, 2024. 373 Ilan Lobel and Asuman Ozdaglar. Distributed subgradient methods for convex optimization over 374 random networks. *IEEE Transactions on Automatic Control*, 56(6):1291–1306, 2010.

375 Paolo Di Lorenzo and Gesualdo Scutari. Next: In-network nonconvex optimization. IEEE Transac376 *tions on Signal and Information Processing over Networks*, 2(2):120–136, 2016. 377 Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. *arXiv* 378 *preprint arXiv:1608.03983*, 2016. 379 Songtao Lu, Xinwei Zhang, Haoran Sun, and Mingyi Hong. Gnsd: A gradient-tracking based 380 nonconvex stochastic algorithm for decentralized optimization. In *2019 IEEE Data Science* 381 *Workshop (DSW)*, pages 315–321. IEEE, 2019. 382 Konstantin Mishchenko, Grigory Malinovsky, Sebastian Stich, and Peter Richtárik. Proxskip: Yes! 383 local gradient steps provably lead to communication acceleration! finally! In *International* 384 *Conference on Machine Learning*, pages 15750–15769. PMLR, 2022. 385 Giorgi Nadiradze, Amirmojtaba Sabour, Peter Davies, Shigang Li, and Dan Alistarh. Asynchronous 386 decentralized sgd with quantized and local updates. *Advances in Neural Information Processing* 387 *Systems*, 34:6829–6842, 2021. 388 Angelia Nedic and Asuman Ozdaglar. Distributed subgradient methods for multi-agent optimization. 389 *IEEE Transactions on Automatic Control*, 54(1):48–61, 2009. 390 Angelia Nedic, Alex Olshevsky, and Wei Shi. Achieving geometric convergence for distributed 391 optimization over time-varying graphs. *SIAM Journal on Optimization*, 27(4):2597–2633, 2017. 392 Shi Pu, Alex Olshevsky, and Ioannis Ch Paschalidis. A sharp estimate on the transient time of 393 distributed stochastic gradient descent. *IEEE Transactions on Automatic Control*, 67(11):5900– 394 5915, 2021. 395 Tiancheng Qin, S Rasoul Etesami, and César A Uribe. Communication-efficient decentralized local 396 sgd over undirected networks. In *2021 60th IEEE Conference on Decision and Control (CDC)*, 397 pages 3361–3366. IEEE, 2021. 398 Guannan Qu and Na Li. Harnessing smoothness to accelerate distributed optimization. *IEEE* 399 *Transactions on Control of Network Systems*, 5(3):1245–1260, 2017. 400 S Sundhar Ram, Angelia Nedic, and Venugopal V Veeravalli. Distributed stochastic subgradient ´ 401 projection algorithms for convex optimization. *Journal of optimization theory and applications*, 402 147:516–545, 2010.

403 Wei Shi, Qing Ling, Gang Wu, and Wotao Yin. Extra: An exact first-order algorithm for decentralized 404 consensus optimization. *SIAM Journal on Optimization*, 25(2):944–966, 2015. 405 Jinming Xu, Shanying Zhu, Yeng Chai Soh, and Lihua Xie. Convergence of asynchronous distributed 406 gradient methods over stochastic networks. *IEEE Transactions on Automatic Control*, 63(2):
407 434–448, 2017. 408 Chung-Yiu Yau and Hoi-To Wai. Fully stochastic distributed convex optimization on time-varying 409 graph with compression. In *2023 62nd IEEE Conference on Decision and Control (CDC)*, pages 410 145–150. IEEE, 2023. 411 Xinlei Yi, Lisha Yao, Tao Yang, Jemin George, and Karl H Johansson. Distributed optimization for 412 second-order multi-agent systems with dynamic event-triggered communication. In *2018 IEEE* 413 *Conference on Decision and Control (CDC)*, pages 3397–3402. IEEE, 2018. 414 Xinlei Yi, Shengjun Zhang, Tao Yang, Tianyou Chai, and Karl H Johansson. Linear convergence 415 of first-and zeroth-order primal–dual algorithms for distributed nonconvex optimization. *IEEE* 416 *Transactions on Automatic Control*, 67(8):4194–4201, 2021.

## 417 **Neurips Paper Checklist**

418 1. **Claims** 419 Question: Do the main claims made in the abstract and introduction accurately reflect the 420 paper's contributions and scope? 421 Answer: [Yes] 422 Justification: [NA] 423 Guidelines: 424 - The answer NA means that the abstract and introduction do not include the claims 425 made in the paper. 426 - The abstract and/or introduction should clearly state the claims made, including the 427 contributions made in the paper and important assumptions and limitations. A No or 428 NA answer to this question will not be perceived well by the reviewers. 429 - The claims made should match theoretical and experimental results, and reflect how 430 much the results can be expected to generalize to other settings. 431 - It is fine to include aspirational goals as motivation as long as it is clear that these goals 432 are not attained by the paper. 433 2. **Limitations** 434 Question: Does the paper discuss the limitations of the work performed by the authors? 435 Answer: [Yes] 436 Justification: [NA] 437 Guidelines: 438 - The answer NA means that the paper has no limitation while the answer No means that 439 the paper has limitations, but those are not discussed in the paper. 440 - The authors are encouraged to create a separate "Limitations" section in their paper. 441 - The paper should point out any strong assumptions and how robust the results are to 442 violations of these assumptions (e.g., independence assumptions, noiseless settings, 443 model well-specification, asymptotic approximations only holding locally). The authors 444 should reflect on how these assumptions might be violated in practice and what the 445 implications would be. 446 - The authors should reflect on the scope of the claims made, e.g., if the approach was 447 only tested on a few datasets or with a few runs. In general, empirical results often 448 depend on implicit assumptions, which should be articulated. 449 - The authors should reflect on the factors that influence the performance of the approach. 450 For example, a facial recognition algorithm may perform poorly when image resolution 451 is low or images are taken in low lighting. Or a speech-to-text system might not be 452 used reliably to provide closed captions for online lectures because it fails to handle 453 technical jargon. 454 - The authors should discuss the computational efficiency of the proposed algorithms 455 and how they scale with dataset size. 456 - If applicable, the authors should discuss possible limitations of their approach to 457 address problems of privacy and fairness. 458 - While the authors might fear that complete honesty about limitations might be used by 459 reviewers as grounds for rejection, a worse outcome might be that reviewers discover 460 limitations that aren't acknowledged in the paper. The authors should use their best 461 judgment and recognize that individual actions in favor of transparency play an impor462 tant role in developing norms that preserve the integrity of the community. Reviewers 463 will be specifically instructed to not penalize honesty concerning limitations.

## 464 3. **Theory Assumptions And Proofs**

465 Question: For each theoretical result, does the paper provide the full set of assumptions and 466 a complete (and correct) proof? 467 Answer: [Yes] 468 Justification: [NA] 469 Guidelines: 470 - The answer NA means that the paper does not include theoretical results. 471 - All the theorems, formulas, and proofs in the paper should be numbered and cross472 referenced. 473 - All assumptions should be clearly stated or referenced in the statement of any theorems. 474 - The proofs can either appear in the main paper or the supplemental material, but if 475 they appear in the supplemental material, the authors are encouraged to provide a short 476 proof sketch to provide intuition. 477 - Inversely, any informal proof provided in the core of the paper should be complemented 478 by formal proofs provided in appendix or supplemental material. 479 - Theorems and Lemmas that the proof relies upon should be properly referenced.

## 480 4. **Experimental Result Reproducibility**

481 Question: Does the paper fully disclose all the information needed to reproduce the main ex482 perimental results of the paper to the extent that it affects the main claims and/or conclusions 483 of the paper (regardless of whether the code and data are provided or not)?

484 Answer: [Yes]
485 Justification: [NA] 486 Guidelines: 487 - The answer NA means that the paper does not include experiments.

488 - If the paper includes experiments, a No answer to this question will not be perceived 489 well by the reviewers: Making the paper reproducible is important, regardless of 490 whether the code and data are provided or not. 491 - If the contribution is a dataset and/or model, the authors should describe the steps taken 492 to make their results reproducible or verifiable. 493 - Depending on the contribution, reproducibility can be accomplished in various ways. 494 For example, if the contribution is a novel architecture, describing the architecture fully 495 might suffice, or if the contribution is a specific model and empirical evaluation, it may 496 be necessary to either make it possible for others to replicate the model with the same 497 dataset, or provide access to the model. In general. releasing code and data is often 498 one good way to accomplish this, but reproducibility can also be provided via detailed 499 instructions for how to replicate the results, access to a hosted model (e.g., in the case 500 of a large language model), releasing of a model checkpoint, or other means that are 501 appropriate to the research performed. 502 - While NeurIPS does not require releasing code, the conference does require all submis503 sions to provide some reasonable avenue for reproducibility, which may depend on the 504 nature of the contribution. For example 505 (a) If the contribution is primarily a new algorithm, the paper should make it clear how 506 to reproduce that algorithm. 507 (b) If the contribution is primarily a new model architecture, the paper should describe 508 the architecture clearly and fully. 509 (c) If the contribution is a new model (e.g., a large language model), then there should 510 either be a way to access this model for reproducing the results or a way to reproduce 511 the model (e.g., with an open-source dataset or instructions for how to construct 512 the dataset).

513 (d) We recognize that reproducibility may be tricky in some cases, in which case 514 authors are welcome to describe the particular way they provide for reproducibility. 515 In the case of closed-source models, it may be that access to the model is limited in 516 some way (e.g., to registered users), but it should be possible for other researchers 517 to have some path to reproducing or verifying the results. 518 5. **Open access to data and code** 519 Question: Does the paper provide open access to the data and code, with sufficient instruc520 tions to faithfully reproduce the main experimental results, as described in supplemental 521 material? 522 Answer: [Yes] 523 Justification: [NA] 524 Guidelines: 525 - The answer NA means that paper does not include experiments requiring code.

526 - Please see the NeurIPS code and data submission guidelines (https://nips.cc/ 527 public/guides/CodeSubmissionPolicy) for more details.

528 - While we encourage the release of code and data, we understand that this might not be 529 possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not 530 including code, unless this is central to the contribution (e.g., for a new open-source 531 benchmark). 532 - The instructions should contain the exact command and environment needed to run to 533 reproduce the results. See the NeurIPS code and data submission guidelines (https: 534 //nips.cc/public/guides/CodeSubmissionPolicy) for more details.

535 - The authors should provide instructions on data access and preparation, including how 536 to access the raw data, preprocessed data, intermediate data, and generated data, etc. 537 - The authors should provide scripts to reproduce all experimental results for the new 538 proposed method and baselines. If only a subset of experiments are reproducible, they 539 should state which ones are omitted from the script and why. 540 - At submission time, to preserve anonymity, the authors should release anonymized 541 versions (if applicable). 542 - Providing as much information as possible in supplemental material (appended to the 543 paper) is recommended, but including URLs to data and code is permitted. 544 6. **Experimental setting/details** 545 Question: Does the paper specify all the training and test details (e.g., data splits, hyper546 parameters, how they were chosen, type of optimizer, etc.) necessary to understand the 547 results? 548 Answer: [Yes] 549 Justification: [NA] 550 Guidelines: 551 - The answer NA means that the paper does not include experiments. 552 - The experimental setting should be presented in the core of the paper to a level of detail 553 that is necessary to appreciate the results and make sense of them. 554 - The full details can be provided either with the code, in appendix, or as supplemental 555 material. 556 7. **Experiment statistical significance** 557 Question: Does the paper report error bars suitably and correctly defined or other appropriate 558 information about the statistical significance of the experiments? 559 Answer: [No]
560 Justification: Due to limited computing resources and time constraints, we are unable to 561 perform multiple runs of our algorithms and report the error bars. We will produce the error 562 bar statistics if time permits. 563 Guidelines: 564 - The answer NA means that the paper does not include experiments.

565 - The authors should answer "Yes" if the results are accompanied by error bars, confi566 dence intervals, or statistical significance tests, at least for the experiments that support 567 the main claims of the paper. 568 - The factors of variability that the error bars are capturing should be clearly stated (for 569 example, train/test split, initialization, random drawing of some parameter, or overall 570 run with given experimental conditions). 571 - The method for calculating the error bars should be explained (closed form formula, 572 call to a library function, bootstrap, etc.) 573 - The assumptions made should be given (e.g., Normally distributed errors). 574 - It should be clear whether the error bar is the standard deviation or the standard error 575 of the mean. 576 - It is OK to report 1-sigma error bars, but one should state it. The authors should 577 preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis 578 of Normality of errors is not verified.

579 - For asymmetric distributions, the authors should be careful not to show in tables or 580 figures symmetric error bars that would yield results that are out of range (e.g. negative 581 error rates). 582 - If error bars are reported in tables or plots, The authors should explain in the text how 583 they were calculated and reference the corresponding figures or tables in the text. 584 8. **Experiments compute resources** 585 Question: For each experiment, does the paper provide sufficient information on the com586 puter resources (type of compute workers, memory, time of execution) needed to reproduce 587 the experiments?

588 Answer: [Yes]
589 Justification: [NA] 590 Guidelines: 591 - The answer NA means that the paper does not include experiments. 592 - The paper should indicate the type of compute workers CPU or GPU, internal cluster, 593 or cloud provider, including relevant memory and storage. 594 - The paper should provide the amount of compute required for each of the individual 595 experimental runs as well as estimate the total compute. 596 - The paper should disclose whether the full research project required more compute 597 than the experiments reported in the paper (e.g., preliminary or failed experiments that 598 didn't make it into the paper). 599 9. **Code of ethics** 600 Question: Does the research conducted in the paper conform, in every respect, with the 601 NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? 602 Answer: [Yes] 603 Justification: [NA] 604 Guidelines: 605 - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. 606 - If the authors answer No, they should explain the special circumstances that require a 607 deviation from the Code of Ethics.

608 - The authors should make sure to preserve anonymity (e.g., if there is a special consid609 eration due to laws or regulations in their jurisdiction). 610 10. **Broader impacts**
611 Question: Does the paper discuss both potential positive societal impacts and negative 612 societal impacts of the work performed? 613 Answer: [NA] 614 Justification: [NA] 615 Guidelines: 616 - The answer NA means that there is no societal impact of the work performed. 617 - If the authors answer NA or No, they should explain why their work has no societal 618 impact or why the paper does not address societal impact. 619 - Examples of negative societal impacts include potential malicious or unintended uses 620 (e.g., disinformation, generating fake profiles, surveillance), fairness considerations 621 (e.g., deployment of technologies that could make decisions that unfairly impact specific 622 groups), privacy considerations, and security considerations. 623 - The conference expects that many papers will be foundational research and not tied 624 to particular applications, let alone deployments. However, if there is a direct path to 625 any negative applications, the authors should point it out. For example, it is legitimate 626 to point out that an improvement in the quality of generative models could be used to 627 generate deepfakes for disinformation. On the other hand, it is not needed to point out 628 that a generic algorithm for optimizing neural networks could enable people to train 629 models that generate Deepfakes faster. 630 - The authors should consider possible harms that could arise when the technology is 631 being used as intended and functioning correctly, harms that could arise when the 632 technology is being used as intended but gives incorrect results, and harms following 633 from (intentional or unintentional) misuse of the technology. 634 - If there are negative societal impacts, the authors could also discuss possible mitigation 635 strategies (e.g., gated release of models, providing defenses in addition to attacks, 636 mechanisms for monitoring misuse, mechanisms to monitor how a system learns from 637 feedback over time, improving the efficiency and accessibility of ML). 638 11. **Safeguards** 639 Question: Does the paper describe safeguards that have been put in place for responsible 640 release of data or models that have a high risk for misuse (e.g., pretrained language models, 641 image generators, or scraped datasets)? 642 Answer: [NA] 643 Justification: [NA]
644 Guidelines:
645 - The answer NA means that the paper poses no such risks. 646 - Released models that have a high risk for misuse or dual-use should be released with 647 necessary safeguards to allow for controlled use of the model, for example by requiring 648 that users adhere to usage guidelines or restrictions to access the model or implementing 649 safety filters. 650 - Datasets that have been scraped from the Internet could pose safety risks. The authors 651 should describe how they avoided releasing unsafe images. 652 - We recognize that providing effective safeguards is challenging, and many papers do 653 not require this, but we encourage authors to take this into account and make a best 654 faith effort.

## 655 12. **Licenses For Existing Assets**

656 Question: Are the creators or original owners of assets (e.g., code, data, models), used in 657 the paper, properly credited and are the license and terms of use explicitly mentioned and 658 properly respected? 659 Answer: [NA] 660 Justification: [NA] 661 Guidelines:
662 - The answer NA means that the paper does not use existing assets.

663 - The authors should cite the original paper that produced the code package or dataset. 664 - The authors should state which version of the asset is used and, if possible, include a 665 URL. 666 - The name of the license (e.g., CC-BY 4.0) should be included for each asset. 667 - For scraped data from a particular source (e.g., website), the copyright and terms of 668 service of that source should be provided.

669 - If assets are released, the license, copyright information, and terms of use in the 670 package should be provided. For popular datasets, paperswithcode.com/datasets 671 has curated licenses for some datasets. Their licensing guide can help determine the 672 license of a dataset.

673 - For existing datasets that are re-packaged, both the original license and the license of 674 the derived asset (if it has changed) should be provided. 675 - If this information is not available online, the authors are encouraged to reach out to 676 the asset's creators. 677 13. **New assets** 678 Question: Are new assets introduced in the paper well documented and is the documentation 679 provided alongside the assets? 680 Answer: [NA] 681 Justification: [NA] 682 Guidelines: 683 - The answer NA means that the paper does not release new assets. 684 - Researchers should communicate the details of the dataset/code/model as part of their 685 submissions via structured templates. This includes details about training, license, 686 limitations, etc. 687 - The paper should discuss whether and how consent was obtained from people whose 688 asset is used. 689 - At submission time, remember to anonymize your assets (if applicable). You can either 690 create an anonymized URL or include an anonymized zip file. 691 14. **Crowdsourcing and research with human subjects** 692 Question: For crowdsourcing experiments and research with human subjects, does the paper 693 include the full text of instructions given to participants and screenshots, if applicable, as 694 well as details about compensation (if any)? 695 Answer: [NA] 696 Justification: [NA] 697 Guidelines: 698 - The answer NA means that the paper does not involve crowdsourcing nor research with 699 human subjects. 700 - Including this information in the supplemental material is fine, but if the main contribu701 tion of the paper involves human subjects, then as much detail as possible should be 702 included in the main paper. 703 - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, 704 or other labor should be paid at least the minimum wage in the country of the data 705 collector.

## 706 15. **Institutional Review Board (Irb) Approvals Or Equivalent For Research With Human**

707 **subjects** 708 Question: Does the paper describe potential risks incurred by study participants, whether 709 such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) 710 approvals (or an equivalent approval/review based on the requirements of your country or 711 institution) were obtained? 712 Answer: [NA] 713 Justification: [NA] 714 Guidelines: 715 - The answer NA means that the paper does not involve crowdsourcing nor research with 716 human subjects. 717 - Depending on the country in which research is conducted, IRB approval (or equivalent) 718 may be required for any human subjects research. If you obtained IRB approval, you 719 should clearly state this in the paper. 720 - We recognize that the procedures for this may vary significantly between institutions 721 and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the 722 guidelines for their institution. 723 - For initial submissions, do not include any information that would break anonymity (if 724 applicable), such as the institution conducting the review.

725 16. **Declaration of LLM usage** 726 Question: Does the paper describe the usage of LLMs if it is an important, original, or 727 non-standard component of the core methods in this research? Note that if the LLM is used 728 only for writing, editing, or formatting purposes and does not impact the core methodology, 729 scientific rigorousness, or originality of the research, declaration is not required. 730 Answer: [NA] 731 Justification: [NA] 732 Guidelines: 733 - The answer NA means that the core method development in this research does not 734 involve LLMs as any important, original, or non-standard components.

735 - Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
736 for what should or should not be described.