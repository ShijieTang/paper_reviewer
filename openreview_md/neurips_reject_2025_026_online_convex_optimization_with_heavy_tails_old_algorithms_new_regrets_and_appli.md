# Online Convex Optimization With Heavy Tails: Old Algorithms, New Regrets, And Applications

Anonymous Author(s)
Affiliation Address email

## Abstract

1 In Online Convex Optimization (OCO), when the stochastic gradient has a finite 2 variance, many algorithms provably work and guarantee a sublinear regret. How3 ever, limited results are known if the gradient estimate has a heavy tail, i.e., the 4 stochastic gradient only admits a finite p-th central moment for some p ∈ (1, 2].

5 Motivated by it, this work examines different old algorithms for OCO (e.g., Online 6 Gradient Descent) in the more challenging heavy-tailed setting. Under the standard 7 bounded domain assumption, we establish new regrets for these classical methods 8 without any algorithmic modification. Remarkably, these regret bounds are fully 9 optimal in all parameters (can be achieved even without knowing p), suggesting 10 that OCO with heavy tails can be solved effectively without any extra operation 11 (e.g., gradient clipping). Our new results have several applications. A particularly 12 interesting one is the first provable convergence result for nonsmooth nonconvex 13 optimization under heavy-tailed noise without gradient clipping. 15 This paper studies the online learning problem with convex losses, also known as Online Convex 16 Optimization (OCO), a widely applicable framework that learns under streaming data [4, 10, 27, 35]. 17 OCO has tons of implications for both designing and analyzing algorithms in different areas, for 18 example, stochastic optimization [8, 23, 14], PAC learning [3], control theory [1, 11], etc.

19 In an OCO problem, a learning algorithm A would interact with the environment in T rounds, where 20 T ∈ N can be either known or unknown. Formally, in each round round t, the learner A first decides an output xt ∈ X from a convex feasible set X ⊆ R
d 21 , then the environment reveals a convex loss 22 function ℓt : X → R, and A incurs a loss of ℓt(xt). After T many rounds, the quantity measuring the 23 algorithm's performance is called regret, defined relative to any fixed competitor x ∈ X as follows:

$$R_{T}^{\mathsf{A}}(\mathbf{x})\triangleq\sum_{t=1}^{T}\ell_{t}(\mathbf{x}_{t})-\ell_{t}(\mathbf{x}).$$

24 In the classical setting, instead of observing full information about ℓt, the learner A is only guaranteed 25 to receive a subgradient ∇ℓt(xt) ∈ ∂ℓt(xt) at its decision, where ∂ℓt(xt) denotes the subdifferential 26 set of ℓt at xt [33]. This turns out to be enough for our purpose of minimizing the regret, since any 27 OCO problem can be reduced to an Online Linear Optimization (OLO) instance via the inequality 28 ℓt(xt) − ℓt(x) *≤ ⟨∇*ℓt(xt), xt − x⟩, which holds due to convexity. Under the standard bounded 29 domain assumption, i.e., X has a finite diameter D, many classical algorithms, e.g., Online Gradient Descent (OGD) [50], guarantee an optimal sublinear regret GD√
30 T for G-Lipschitz ℓt. Even better, 31 in the case that computing an exact subgradient is intractable, and one could only query a stochastic estimate gtsatisfying E [gt 32 | xt] ∈ ∂ℓt(xt), the OGD algorithm can still solve OCO effectively with Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Do not distribute.

## 14 **1 Introduction**

a provable (G + σ)D
√
33 T regret bound in expectation if the stochastic noise gt − ∇ℓt(xt) has a bounded second moment σ 2 34 for some σ ≥ 0, which is called the finite variance condition.

35 However, many works have pointed out that even for the easier stochastic optimization (i.e., ℓt = F 36 for a common F), the typical finite variance assumption is too optimistic and can be violated in 37 different tasks [12, 37, 45], and their observations suggest that the stochastic gradient only admits a finite p-th central moment upper bounded by σ p 38 for some p ∈ (1, 2], which is named heavy-tailed 39 noise. This new assumption generalizes the classical finite variance condition (p = 2) and becomes 40 challenging when p < 2. A particular evidence is that the famous Stochastic Gradient Descent (SGD) 41 algorithm [32] (which is exactly OGD for stochastic optimization) provably diverges [45]. 42 Though heavy-tailed stochastic optimization has been extensively studied [18, 26, 34], limited results 43 are known for OCO with heavy tails. The only work under this topic that we are aware of is [47], 44 which established a parameter-free regret bound in high probability (more discussions provided 45 later). However, their algorithm includes many nontrivial modifications like gradient clipping and 46 significantly deviates from the existing simple OCO algorithms used in practice. Especially, consider 47 OGD as an example. Though the heavy-tailed issue is known, OGD (or just think of it as SGD) still 48 works (sometimes very well) in practice even without gradient clipping and is arguably one of the 49 most popular optimizers, which seemingly contradicts the theory of unconvergence mentioned before. 50 This indicates that, for classical OCO algorithms under heavy-tailed noise, a huge gap exists between 51 the empirical convergence (or even the effective practical performance) and theoretical guarantees. 52 Therefore, we are naturally led to the following question: 53 *In what context can old OCO algorithms work under heavy tails, in what sense, and to what extent?*

## 54 **1.1 Contributions**

55 Motivated by the above question, we examine three classical algorithms for OCO: Online Gradient 56 Descent (OGD) [50], Dual Averaging (DA) [25, 43], and AdaGrad [9, 22], and answer it as follows:
Under the standard bounded domain assumption, the in-expectation regret E-R
AT(x)
57 *is finite and* 58 optimal for any A ∈ {OGD, DA, AdaGrad}*, without any algorithmic modification.* 59 In detail, our new results for heavy-tailed OCO are summarized here:
- We prove the only and the first optimal regret bound E-R
AT
(x)≲ GD√T +σDT1/p 60 , ∀x ∈ X for 61 any A ∈ {OGD, DA, AdaGrad}. Remarkably, AdaGrad can achieve this result without knowing 62 any of the Lipschitz parameter G, noise level σ, and tail index p. 63 - We extend the analysis of OGD to Online Strongly Convex Optimization with heavy tails and establish the first provable result E-R
OGD
T(x)≲
G2log T
µ +
σ pG2−p µT
2−p 64 , ∀x ∈ X , where µ > 0 is the modulus of strong convexity and T
0 65 should be read as log T. 66 Based on the new regret bounds for OCO with heavy tails, we provide the following applications: 67 - For nonsmooth convex optimization with heavy tails, we show the first optimal in-expectation rate GD/√T + *σD/T*1−1/p 68 achieved without gradient clipping, which applies to both the average 69 iterate and last iterate, demonstrating that SGD does converge once the domain is bounded. 70 - For nonsmooth nonconvex optimization with heavy tails, we show the first provable sample complexity of G2δ
−1ϵ
−3 + σ p p−1 δ
−1ϵ
−
2p−1 p−1 71 for finding a (*δ, ϵ*)-stationary point without gradient 72 clipping. Moreover, we give the first convergence result when the problem-dependent parameters 73 (like G, σ, and p) are unknown in advance.

## 74 **1.2 Discussion On [47]**

75 As noted, [47] is the only work for OCO with heavy tails, as far as we know. There are two 76 major discrepancies between them and us. First, they consider the case where the feasible set 77 X is unbounded and aim to establish a parameter-free regret bound, i.e., the regret bound has a 78 linear dependency on ∥x∥ (up to an extra polylog ∥x∥) for any competitor x ∈ X . Second, they 79 focus on high-probability rather than in-expectation analysis. As such, their regret is in the form of R
AT
(x) ≲ (G + σ) ∥x∥ T
1/p 80 , ∀x ∈ X (up to extra polylogarithmic factors) with high probability. 81 Without a doubt, their setting is harder than ours implying their bound is stronger as it can convert to an in-expectation regret E-R
A T
(x)≲ (G + σ)DT1/p 82 for any bounded domain X with a diameter D. 83 We emphasize that the motivation behind [47] differs heavily from ours. They aim to solve heavy84 tailed OCO with a new proposed method that contains many nontrivial technical tricks, including 85 gradient clipping, artificially added regularization, and solving the additional fixed-point equation.

86 However, their result cannot reflect why the existing simple OCO algorithms like OGD work in 87 practice under heavy-tailed noise. In contrast, our goal is to examine whether, when, and how the 88 classical OCO algorithms work under heavy tails, thereby filling the missing piece in the literature.

Moreover, we would like to mention two drawbacks of [47]. First, though the T
1/p 89 regret seems 90 tight as it matches the lower bound [24, 30, 41], this may not be the best, since an optimal bound should recover the standard 
√
91 T regret in the deterministic case (i.e., σ = 0), as one can imagine. 92 This suggests that their bound is not entirely optimal. Second, we remark that they require knowing 93 both problem-dependent parameters G, σ, p and time horizon T in the algorithm, which may be hard to satisfy in the online setting. In comparison, our regret bound GD√T + σDT1/p 94 is fully optimal 95 in all parameters. Importantly, AdaGrad can achieve it while oblivious to the problem information.

## 96 **2 Preliminary**

97 **Notation.** N denotes the set of natural numbers (excluding 0). [T] ≜ {1*, . . . , T*} , ∀T ∈ N. a ∧ b ≜ 98 min {*a, b*} and a ∨ b ≜ max {*a, b*}. We write a ≲ b if a ≤ Cb for a universal constant C > 0.

99 ⌊·⌋ and ⌈·⌉ respectively represent the floor and ceiling functions. ⟨·, ·⟩ denotes the Euclidean inner product and ∥·∥ ≜p⟨·, ·⟩ is the standard 2-norm. Given x ∈ R
dand D > 0, B
d 100 (x, D) is the Euclidean ball in R
dcentered at x with a radius D. In the case x = 0, we use the shorthand B
d 101 (D).

Given a nonempty closed convex set A ⊆ R
d 102 , ΠA is the Euclidean projection operator onto A. For a 103 convex function f, ∂f(x) denotes its subgradient set at x.

104 *Remark* 1. We choose the Euclidean norm only for simplicity. Extending the results in this work to 105 any general norm is straightforward. 106 This work studies OCO in the context of Assumption 1. 107 **Assumption 1.** *We consider the following series of assumptions:*
- X ⊂ R
d 108 is a nonempty closed convex set bounded by D*, i.e.,* supx,y∈X ∥x − y∥ ≤ D.

109 - ℓt : X → R *is convex for all* t ∈ [T].

110 - ℓt is G-Lipschitz on X , i.e., ∥∇ℓt(x)∥ ≤ G, ∀x ∈ X , ∇ℓt(x) ∈ ∂ℓt(x), for all t ∈ [T].

- Given a point xt ∈ X at the t*-th iteration, one can query* gt ∈ R
d 111 *satisfying* ∇ℓt(xt) ≜
E [gt| Ft−1] ∈ ∂ℓt(xt) and E-∥ϵt∥
p≤ σ p 112 for some p ∈ (1, 2] and σ ≥ 0*, where* Ft ≜
σ(g1*, . . . ,* gt 113 ) denotes the natural filtration and ϵt ≜ gt − ∇ℓt(xt) *is the stochastic noise.*
114 *Remark* 2. D is recognized as known, like ubiquitously assumed in the OCO literature. Moreover, 115 xt denotes the decision/output of the online learning algorithm by default. 116 In Assumption 1, the first three points are standard, and the fourth is the heavy-tailed noise assumption. 117 In particular, p = 2 recovers the standard finite variance condition.

## 118 **3 Old Algorithms Under Heavy Tails**

119 In this section, we revisit three classical algorithms for OCO: OGD, DA, and AdaGrad, whose regret 120 bounds are well-studied in the finite variance case but remain unknown under heavy-tailed noise. 121 The basic idea of proving these algorithms work under heavy tails is to leverage the boundness 122 property of X . We will describe it in more detail using OGD as an illustrated example. The analysis 123 of DA follows a similar way at a high level, but differs in some details. However, though AdaGrad 124 can be viewed as OGD with an adaptive stepsize, the way to utilize the boundness property is entirely 125 different. All formal proofs are deferred to the appendix due to space limitations.

## 126 **3.1 New Regret For Online Gradient Descent**

| Algorithm 1 Online Gradient Descent (OGD) [50] Input: initial point x1 ∈ X , stepsize ηt > 0 for t = 1 to T do xt+1 = ΠX (xt − ηtgt ) end for   |
|-------------------------------------------------------------------------------------------------------------------------------------------------|

127 We begin from arguably the most basic algorithm for OCO, Online Gradient Descent (OGD). 128 **A well known analysis.** The regret bound of OGD has been extensively studied [10, 27, 35]. The 129 most well known analysis is perhaps the following one: for any x ∈ X , there is

$\left\|\mathbf{x}_{t+1}-\mathbf{x}\right\|^{2}=\left\|\Pi_{\mathcal{X}}(\mathbf{x}_{t}-\eta_{t}\mathbf{g}_{t})-\Pi_{\mathcal{X}}(\mathbf{x})\right\|^{2}\leq\left\|\mathbf{x}_{t}-\eta_{t}\mathbf{g}_{t}-\mathbf{x}\right\|^{2},$
130 where the inequality holds by the nonexpansive property of ΠX . Expanding both sides and rearranging 131 terms yield that

$$\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}\rangle\leq\frac{\left\|\mathbf{x}_{t}-\mathbf{x}\right\|^{2}-\left\|\mathbf{x}_{t+1}-\mathbf{x}\right\|^{2}}{2\eta_{t}}+\frac{\eta_{t}\left\|\mathbf{g}_{t}\right\|^{2}}{2}.\tag{1}$$

If gt 132 admits a finite variance, i.e., p = 2 in Assumption 1, taking expectations on both sides, then following a standard analysis for ηt =D
(G+σ)
√t
(or ηt =D
(G+σ)
√T
133 if T is known) gives the regret

$$\mathbb{E}\left[\mathbb{R}_{T}^{\mathsf{Q G D}}(\mathbf{x})\right]\lesssim(G+\sigma)D{\sqrt{T}},\forall\mathbf{x}\in{\mathcal{X}}.$$

134 However, the step of taking expectations on the R.H.S. of (1) crucially relies on the finite variance condition of gt 135 . Therefore, one may naturally think OGD would not guarantee a finite regret if p < 2.

A less well known analysis1 136 . As discussed, the failure of the above proof under heavy-tailed noise is 137 due to (1). Therefore, if a tighter inequality than (1) exists, then it might be possible to show that 138 OGD still works for p < 2. However, does it exist?

139 Actually, there is another less well known analysis to produce a better inequality than (1). That is, 140 first showing for any x ∈ X , by the optimality condition of the update rule,

$$\langle\mathbf{g}_{t},\mathbf{x}_{t+1}-\mathbf{x}\rangle\leq\frac{\langle\mathbf{x}_{t}-\mathbf{x}_{t+1},\mathbf{x}_{t+1}-\mathbf{x}\rangle}{\eta_{t}}=\frac{\left\|\mathbf{x}_{t}-\mathbf{x}\right\|^{2}-\left\|\mathbf{x}_{t+1}-\mathbf{x}\right\|^{2}-\left\|\mathbf{x}_{t}-\mathbf{x}_{t+1}\right\|^{2}}{2\eta_{t}},$$

141 and then obtaining

$$\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}\rangle\leq{\frac{\left\|\mathbf{x}_{t}-\mathbf{x}\right\|^{2}-\left\|\mathbf{x}_{t+1}-\mathbf{x}\right\|^{2}}{2\eta_{t}}}+\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}_{t+1}\rangle-{\frac{\left\|\mathbf{x}_{t}-\mathbf{x}_{t+1}\right\|^{2}}{2\eta_{t}}}.$$
$$(3)$$
$$\left\|\right)\left\|{\boldsymbol{x}}_{t}-{\boldsymbol{x}}_{t+1}\right\|.$$

$$(4)$$
$${\mathrm{(2)}}$$

Note that (2) is tighter than (1) as ⟨gt, xt − xt+1⟩ ≤ ∥gt∥ ∥xt − xt+1∥ ≤ ηt∥gt∥
2 2 +
∥xt−xt+1∥
2 2ηt 142 , 143 where the first step is due to Cauchy-Schwarz inequality and the second one is by AM-GM inequality.

144 Handle p < 2 **in a simple way.** Though we have tightened (1) into (2), can inequality (2) help to 145 overcome heavy tails? The answer is surprisingly positive, and our solution is fairly simple. Instead 146 of directly applying AM-GM inequality in the second step, we recall gt = ∇ℓt(xt) + ϵt and use 147 triangle inequality to obtain
⟨gt, xt − xt+1⟩ ≤ ∥gt∥ ∥xt − xt+1∥ ≤ (∥∇ℓt(xt)∥ + ∥ϵt∥) ∥xt − xt+1∥ . (3)
148 On the one hand, by ∥∇ℓt(xt)∥ ≤ G and AM-GM inequality, there is

$\|\nabla\ell_{t}(\mathbf{x}_{t})\|\,\|\mathbf{x}_{t}-\mathbf{x}_{t+1}\|\leq G\,\|\mathbf{x}_{t}-\mathbf{x}_{t+1}\|\leq\eta_{t}G^{2}+\frac{\|\mathbf{x}_{t}-\mathbf{x}_{t+1}\|^{2}}{4\eta_{t}}$.  
. (4)
On the other hand, let p⋆ ≜p
d, let $\sf{p_{+}\triangleq\frac{p}{p-1}}$ and $\sf{C(p)\triangleq\frac{(4p-4)^{p-1}}{p^{p}}}$, we have 
p 149 , we have
! 1 p⋆ ∥ϵt∥ ∥xt − xt+1∥ = 4ηt p⋆  1 p⋆ ∥ϵt∥ ∥xt − xt+1∥ 1− 2 p⋆ ·  p⋆ ∥xt − xt+1∥ 2 4ηt (a) ≤ 4ηt p⋆  p p⋆∥ϵt∥ p∥xt − xt+1∥ p− 2p p⋆ p+ ∥xt − xt+1∥ 2 4ηt (b) ≤ C(p)η p−1 t ∥ϵt∥ p D2−p + ∥xt − xt+1∥ 2 4ηt , (5) where (a) is by Young's inequality and (b) is due to ∥xt − xt+1∥ ≤ D, p⋆ =p p−1 150 , and C(p) = (4p−4)p−1
$$({\boldsymbol{5}})$$
(6) $\frac{1}{2}$
p
p 151 . Next, we plug (4) and (5) back into (3), then combine with (2) to know
Next, we plug (4) and (5) back into (3), then combine with (2) to know  $\left\langle g_t,x_t-x\right\rangle\leq\frac{\left\|x_t-x\right\|^2-\left\|x_{t+1}-x\right\|^2}{2\eta_t}+\eta_t G^2+\mathbb{C}(\mathfrak{p})\eta_t^{p-1}\left\|\epsilon_t\right\|^p D^{2-p}$. 
p D2−p. (6)
Notably, the term ∥ϵt∥
p
152 has a correct exponent p. Thus, we can safely take expectations on both sides. 153 Finally, a standard analysis yields the following Theorem 1 (see Appendix A for a formal proof).
Theorem 1. *Under Assumption 1, taking* ηt =D
G
√t
∧D
σt1/p
154 in OGD *(Algorithm 1), we have*
$$\mathbb{E}\left[\mathbb{R}_{T}^{\mathsf{OGD}}(x)\right]\lesssim G D$$
T(x)≲ GD√T + σDT1/p, ∀x ∈ X .

155 As far as we know, Theorem 1 is the first and the only provable result for OGD under heavy tails.

156 Remarkably, it is not only tight in T [24, 30, 41] but also fully optimal in all parameters, in contrast to the bound (G + σ)DT1/p 157 of [47]. This reveals that OCO with heavy tails can be optimally solved 158 as effectively as the finite variance case once the domain is bounded, a classical condition adapted in 159 many existing works. 160 **Strongly convex functions.** We highlight that the above idea can also be applied to Online Strongly Convex Optimization and leads to a sublinear regret T
2−p better than T
1/p 161 . This extension can be 162 found in Appendix A.

## 163 **3.2 New Regret For Dual Averaging**

Algorithm 2 Dual Averaging (DA) [25, 43]
Input: initial point x1 ∈ X , stepsize ηt > 0 for t = 1 to T do xt+1 = ΠX (x1 − ηtPts=1 gs)
end for 164 *Remark* 3. It is known that DA is a special realization of the more general Follow-the-Regularized165 Leader (FTRL) framework [21]. To keep the work concise, we only focus on DA. The key idea to 166 prove Theorem 2 can directly extend to show new regret for FTRL under heavy-tailed noise.

167 We turn our attention to the second candidate, the Dual Averaging (DA) algorithm, which is given in Algorithm 2. Though DA coincides with OGD when X = R
d 168 and ηt = η, these two methods in 169 general are not equivalent and can have significant performance differences in practice. Therefore, it 170 is also important to understand DA under heavy tails.

176 As far as we know, Theorem 2 is the first provable and optimal regret for DA under heavy tails. It 177 guarantees the same tight bound as in Theorem 1 up to different constants.

171 Despite the proof strategies for OGD and DA are in different flavors (even for p = 2), the basic idea 172 presented before for OGD still works here, i.e., apply the boundness property of X to make the term 173 ∥ϵt∥ have a correct exponent. Armed with this thought, we can prove the following new regret bound
174 for DA under heavy-tailed noise. We refer the reader to Appendix B for its proof.
Theorem 2. *Under Assumption 1, taking* ηt =D
G
√t
∧D
σt1/p
175 in DA *(Algorithm 2), we have*
E
$${\mathfrak{L}}\left[{\mathsf{R}}_{T}^{\mathsf{D A}}({\boldsymbol{x}})\right]\,;$$

## T(X)≲ Gd√T + Σdt1/P, ∀X ∈ X . 178 **3.3 New Regret For** Adagrad

Algorithm 3 AdaGrad [9, 22]
Input: initial point x1 ∈ X , stepsize η > 0 for t = 1 to T do ηt = ηV −1/2 t where Vt =Pts=1 ∥gs∥
2 xt+1 = ΠX (xt − ηtgt)
end for 179 *Remark* 4. Algorithm 3 is also named AdaGrad-Norm (e.g., [42]). We simply call it AdaGrad. It is 180 straightforward to generalize Theorem 3 below to the per-coordinate update version.

E
-R
AdaGrad T(x)≲ GD√T + σDT1/p, ∀x ∈ X .

190 *Remark* 5. We also establish a similar result for DA with an adaptive stepsize. See Theorem 7 in 191 Appendix B for details. 192 Theorem 3 provides the first regret bound for AdaGrad under heavy tails. Impressively, it is optimal 193 even without knowing any of G, σ, and p. This surprising result once again demonstrates the power 194 of the adaptive method, indicating it is robust to an unknown environment and even heavy-tailed 195 noise, which may partially explain the favorable performance of many adaptive optimizers designed 196 based on AdaGrad like RMSProp [40] and Adam [14]. 197 We point out that the key to establishing Theorem 3 differs from the idea used before for OGD and 198 DA. Actually, Theorem 3 can be obtained in an embarrassingly simple way. It is known that AdaGrad with η = D/√
199 2 on a bounded domain guarantees the following path-wise regret

$$\sum_{t=1}^{T}\left\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}\right\rangle\lesssim D{\sqrt{\sum_{t=1}^{T}\left\|\mathbf{g}_{t}\right\|^{2}}}.$$
$$\left(7\right)$$
$\begin{matrix}T&||\mathbf{q}\\ 4&1\end{matrix}$
2. (7)

## 205 **4 Applications**

206 We provide some applications based on the new regret bounds established in Section 3. The basic 207 problem we study is optimizing a single objective F, which could be either convex or nonconvex.

## 208 **4.1 Nonsmooth Convex Optimization**

209 In this section, we consider nonsmooth convex optimization with heavy tails.

186 To handle this issue, we consider AdaGrad, a classical adaptive algorithm for OCO. As can be seen, 187 AdaGrad is just OGD with an adaptive stepsize. However, it is this adaptive stepsize that can help us 188 to overcome the above undesired point.

Theorem 3. *Under Assumption 1, taking* η = D/√
189 2 in AdaGrad *(Algorithm 3), we have* 181 Although Theorems 1 and 2 are optimal, they both suffer from an undesired point. That is, the stepsize ηt =D
G
√t
∧D
σt1/p 182 requires knowing all problem-dependent parameters. However, it may not 183 be easy to obtain them in an online setting. Especially, it heavily depends on the prior information 184 about the tail index p, which is hard to know (even approximately) in advance. In other words, they 185 both lack the adaptive property to an unknown environment.

Observe that qPT
t=1 ∥gt∥
2 ≲

qPT
t=1 ∥∇ℓt(xt)∥
2 +
qPT
t=1 ∥ϵt∥
2 ≤ G
√T +
PT
t=1 ∥ϵt∥
p 1p 200 ,
where the last step is due to ∥·∥2 *≤ ∥·∥*p 201 for any p ∈ [1, 2]. After taking expectations on both sides of
(7) and applying Hölder's inequality to obtain E
PT
t=1 ∥ϵt∥
p 1 p≤
PT
t=1 E
-∥ϵt∥
p 1p≤ σT
1 p 202 , 203 we conclude Theorem 3. To make the work self-consistent, we produce the formal proof of Theorem 204 3 in Appendix C. 210 **Convergence of the average iterate.** First, we focus on convergence in average. By the classical 211 online-to-batch conversion [3], the following corollary immediately holds.

Corollary 1. *Under Assumption 1 for* ℓt(x) = ⟨∇F(xt), x⟩ *and let* x¯T ≜
1 T
PT
212 t=1 xt*, for any* 213 A ∈ {OGD, DA, AdaGrad}*, we have*

$$\mathbb{E}\left[F({\bar{\mathbf{x}}}_{T})-F(\mathbf{x})\right]\leq{\frac{\mathbb{E}\left[\mathsf{R}_{T}^{\mathsf{A}}(\mathbf{x})\right]}{T}}\lesssim{\frac{G D}{\sqrt{T}}}+{\frac{\sigma^{\mathsf{p}}D}{T^{1-{\frac{1}{\mathsf{p}}}}}},\forall\mathbf{x}\in\mathcal{X}.$$

Proof. By convexity, F(x¯T ) − F(x) ≤
PT
t=1 F (xt)−F (x)
T ≤
R
A
T (x)
T
214 is valid for any OCO algorithm 215 A. We conclude from invoking Theorems 1, 2 and 3. 216 To the best of our knowledge, Corollary 1 gives the first and optimal convergence rate for these three 217 algorithms in stochastic optimization with heavy tails. Especially, it implies that once the domain 218 is bounded, the widely implemented SGD algorithm provably converges under heavy-tailed noise 219 without any algorithmic change considered in many prior works, e.g., gradient clipping [18, 26]. 220 We are only aware of two works [19, 41] based on Stochastic Mirror Descent (SMD) [24] that gave 221 convergence results without clipping. However, they share a common drawback, i.e., their bounds are both in the form of (G + σ)D/T1−1/p, which cannot recover the optimal rate GD/√
222 T when σ = 0. 223 Lastly, we highlight that for A = AdaGrad, Corollary 1 is not only optimal but also adaptive to the 224 tail index p. As far as we know, no result has achieved this property before. This once again evidences 225 the benefit of adaptive gradient methods. 226 **Convergence of the last iterate.** Next, we consider the more challenging last-iterate convergence, 227 which has a long history in stochastic optimization and fruitful results in the case of p = 2 (see, 228 e.g., [28, 36, 49]). However, less is known about heavy-tailed problems. So far, only two works 229 [19, 29] have established the last-iterate convergence. The former is based on SMD, and the latter 230 employs gradient clipping in SGD. Unfortunately, their rates are both in the suboptimal order
(G + σ)D/T1−1/p 231 . 232 We will provide an optimal last-iterate rate based on the following lemma, which reduces the 233 last-iterate convergence to an online learning problem.

Lemma 1 (Theorem 1 of [7]). Suppose x1, . . . , xT and y1 234 , . . . , yT *are two sequences of vectors* 235 satisfying xt ∈ X , x1 = y1 and

$$\mathbf{y}_{t+1}=\mathbf{y}_{t}+{\frac{T-t}{T}}\left(\mathbf{x}_{t+1}-\mathbf{x}_{t}\right).$$

T(xt+1 − xt). (8)
Given a convex function F(x)*, let* ℓt(x) = ⟨∇F(yt 236 ), x⟩. Then for any online learner A*, we have*

$$(8)$$
_Let $\ell_{t}(\mathbf{x})=\langle\nabla F(\mathbf{y}_{t}),\mathbf{x}\rangle$. Then for all $\mathbf{x}$._
$$F(\mathbf{y}_{T})-F(\mathbf{x})\leq{\frac{\mathsf{R}_{T}^{\mathsf{A}}(\mathbf{x})}{T}},\forall\mathbf{x}\in{\mathcal{X}}.$$

We emphasize that the stochastic gradient gtreceived by A is an estimate of ∇F(yt 237 ) instead of 238 ∇F(xt). This flexibility is due to the generality of the OCO framework. Moreover, for OGD,
suppose there is no projection step, then (8) is equivalent to yt+1 = yt −
T −t Tηtgt 239 , which can be viewed as SGD with a stepsize T −t T
240 ηt. For proof of Lemma 1, we refer the interested reader to [7].

Corollary 2. *Under Assumption 1 for* ℓt(x) = ⟨∇F(yt), x⟩*, where* yt 241 *satisfies (8), for any* A ∈ 242 {OGD, DA, AdaGrad}*, we have*

$$\mathbb{E}\left[F(\mathbf{y}_{T})-F(\mathbf{x})\right]\leq{\frac{\mathbb{E}\left[\mathbf{R}_{T}^{\mathbf{A}}(\mathbf{x})\right]}{T}}\lesssim{\frac{G D}{\sqrt{T}}}+{\frac{\sigma^{\mathsf{P}}D}{T^{1-{\frac{1}{\sigma}}}}},\forall\mathbf{x}\in\mathcal{X}.$$

243 *Proof.* Combine Lemma 1 and Theorems 1, 2 and 3 to conclude. 244 As far as we know, Corollary 2 is the first optimal last-iterate convergence rate for stochastic convex 245 optimization with heavy tails, closing the gap in existing works.

One may notice that yt 246 itself is not the decision made by the online learner and naturally may ask 247 whether xt ensures the last-iterate convergence if we simply pick ℓt = F. The answer turns out to

$\square$
248 be positive at least for OGD (which is equivalent to SGD now). However, to prove this result, we 249 rely on a technique specialized to stochastic optimization recently developed by [19, 44]. To not 250 diverge from the topic of OCO, we defer the last-iterate convergence of OGD to Appendix D, in 251 which Theorem 8 gives a general result for any stepsize ηt and Corollary 4 shows the last-iterate rate under the same stepsize ηt =D
G
√t
∧D
σt 252 1/p as in Theorem 1 before.

## 253 **4.2 Nonsmooth Nonconvex Optimization**

254 This section contains another application, nonsmooth nonconvex optimization with heavy tails. Due 255 to limited space, we will provide only the necessary background. For more details, we refer the reader 256 to [6, 13, 15, 16, 38, 39] for recent progress. We start with a new set of conditions. 257 **Assumption 2.** *We consider the following series of assumptions:*
258 - The objective F *is lower bounded by* F⋆ ≜ infx∈Rd F(x) ∈ R.

- F *is differentiable and well-behaved, i.e.,* F(x) − F(y) = R 1 0 259 ⟨∇F(y + t(x − y)), x − y⟩ dt.

- F is G*-Lipschitz on* R
d, i.e., ∥∇F(x)∥ ≤ G, ∀x ∈ R
d 260 .

- *Given* zt ∈ R
d at the t*-th iteration, one can query* gt ∈ R
d*satisfying* E [gt 261 | Ft−1] = ∇F(zt)
and E-∥ϵt∥
p≤ σ p 262 for some p ∈ (1, 2] and σ ≥ 0, where Ft *denotes the natural filtration and* 263 ϵt ≜ gt − ∇F(zt) *is the stochastic noise.*
264 *Remark* 6. The second point is a mild regularity condition introduced by [5] and becomes standard 265 in the literature [2, 17, 48]. See Definition 1 and Proposition 2 of [5] for more details. In the fourth 266 point, we use the same notation zt as in the algorithm being studied later. In fact, it can be arbitrary.

267 In nonsmooth nonconvex optimization, we aim to find a (δ, ϵ)-stationary point [46] (see the formal Definition 2 in Appendix E). This goal can be reduced to finding a point x ∈ R
d 268 such that
∥∇F(x)∥δ ≤ ϵ, where ∥∇F(x)∥δ 269 is a quantity introduced by [5] as follows.

Definition 1 (Definition 5 of [5]). Given a point x ∈ R
d 270 , a number δ > 0 and an almost-everywhere differentiable function F, define ∥∇F(x)∥δ ≜ infS⊂B(x,δ),
1 |S| Py∈S y=x 1 |S| Py∈S ∇F(y)

271  .

The only existing sample complexity under Assumption 2 is (G+σ)
p p−1 δ
−1ϵ
−
2p−1 p−1 272 in high probability 273 [17], where we only report the dominant term and hide the dependency on the failure probability.

However, on the theoretical side, their result cannot recover the optimal bound G2δ
−1ϵ
−3 274 [5] in the 275 deterministic case. On the practical side, their method also employs the gradient clipping step, which 276 introduces a new clipping parameter to tune. In fact, as stated in their Section 5, they observed in 277 experiments that their algorithm without the clipping operation (exactly the algorithm we study next) 278 still works under heavy tails. In addition, in their Section 6, they also explicitly ask whether the 279 requirement to know G and A can be removed. 280 As will be seen later, we can address these points with the new regret bounds presented before.

## 281 **4.2.1 Online-To-Nonconvex Conversion Under Heavy Tails**

Algorithm 4 Online-to-Nonconvex Conversion (O2NC) [5]
Input: initial point y0 ∈ R
d, K ∈ N, T ∈ N, online learning algorithm A.

for n = 1 to KT do Receive xn from A
yn = yn−1 + xn zn = yn−1 + snxn where sn ∼ Uniform [0, 1] i.i.d. Query a stochastic gradient gn at zn Send gn to A
end for 282 *Remark* 7. Note that O2NC is a randomized algorithm. Therefore, the definition of the natural filtration is adjusted to Fn ≜ σ(s1, g1 283 , . . . , sn, gn, sn+1) accordingly.

284 We provide the Online-to-Nonconvex Conversion (O2NC) framework in Algorithm 4, which serves 285 as a meta algorithm. Roughly speaking, Algorithm 4 reduces a nonconvex optimization problem 286 to an OCO (in fact, OLO) problem, for which the K-shifting regret (see (9)) of the online learner 287 A crucially affects the final convergence rate. However, the existing Theorem 8 of [5], a general 288 convergence result for the above reduction, cannot directly apply to heavy-tailed noise, since its proof 289 relies on the finite variance condition on gn (see Appendix E for more details).

Theorem 4. *Under Assumption 2 and let* vk ≜ −D
PkT
n=(k−1)T +1 ∇F (zn)
∥PkT
n=(k−1)T +1 ∇F (zn)∥
290 , ∀k ∈ [K] *for arbitrary* 291 D > 0*, then for any online learning algorithm* A in O2NC *(Algorithm 4), we have*

$$\left[\sum_{k=1}^{K}\frac{1}{K}\left\|\frac{1}{T}\sum_{n=(k-1)T+1}^{kT}\nabla F(\mathbf{z}_{n})\right\|\right]\lesssim\frac{F(\mathbf{y}_{0})-F_{*}}{DKT}+\frac{\mathbb{E}\left[\mathsf{R}_{T}^{\mathsf{A}}(\mathbf{v}_{1},\cdots,\mathbf{v}_{K})\right]}{DKT}+\frac{\sigma}{T^{1-\frac{1}{\rho}}}.$$
$\mathbb{E}_{\mathbb{C}}$

$$(9)$$
.
R
AT
292 (v1, *· · ·* , vK) in Theorem 4 is called K*-shifting regret* [5], defined as follows:

$$\mathcal{R}_{T}^{\mathbb{A}}\left(\mathbf{v}_{1},\ldots,\mathbf{v}_{K}\right)\triangleq\sum_{k=1}^{K}\sum_{n=(k-1)T+1}^{k T}\ell_{n}(\mathbf{x}_{n})-\ell_{n}(\mathbf{v}_{k})\quad\text{where}\quad\ell_{n}(\mathbf{x})\triangleq\left\langle\mathbf{g}_{n},\mathbf{x}\right\rangle.$$

293 Theorem 4 here provides a new and the first theoretical guarantee for O2NC under heavy tails. 294 Especially, it recovers Theorem 8 of [5] when p = 2. A remarkable point is that the O2NC algorithm 295 itself does not need any information about p. The proof of Theorem 4 can be found in Appendix E.

## 296 **4.2.2 Convergence Rates**

Theorem 4 enables us to apply the results presented in Section 3. Concretely, for X = B
d 297 (D) and 298 any A ∈ {OGD, DA, AdaGrad}, if we reset the stepsize in A after every T iterations, there will be E
-R
AT
(v1, · · · , vK)≲ GDK√T + *σDKT*1/p 299 by our new regret bounds, since vk ∈ X . With a 300 carefully picked D, we obtain the following Theorem 5. Its proof is deferred to Appendix E.

Theorem 5. *Under Assumption 2 and let* ∆ ≜ F(y0
)−F⋆ and z¯k ≜
1 T
PkT
n=(k−1)T +1 301 zn, ∀k ∈ [K],
setting any A ∈ {OGD, DA, AdaGrad} in O2NC *(Algorithm 4) with a domain* X = B
d 302 (D) for 303 D = δ/T and resetting the stepsize in A after every T *iterations, we have*

$$\mathbb{E}\left[{\frac{1}{K}}\sum_{k=1}^{K}\|\nabla F({\bar{\mathbf{z}}}_{k})\|_{\delta}\right]\stackrel{<}{\sim}{\frac{\Delta}{\delta K}}+{\frac{G}{\sqrt{T}}}+{\frac{\sigma}{T^{1-{\frac{1}{p}}}}}.$$

304 Notably, this is the first time confirming that gradient clipping is indeed unnecessary for the O2NC 305 framework, matching the experimental observation of [17]. 306 **Corollary 3.** Under the same setting of Theorem 5, suppose we have N ≥ 2 *stochastic gradient* budgets, taking K = ⌊N/T⌋ and T = ⌈N/2⌉ ∧ l(*δGN/*∆) 23m∨
l(*δσN/*∆)p 2p−1m 307 *, we have*

$$\mathbb{E}\left[\frac{1}{K}\sum_{k=1}^{K}\|\nabla F(\bar{\mathbf{z}}_{k})\|_{\delta}\right]\lesssim\frac{G}{\sqrt{N}}+\frac{\sigma}{N^{1-\frac{1}{p}}}+\frac{\Delta}{\delta N}+\frac{G^{\frac{2}{3}}\Delta^{\frac{1}{3}}}{(\delta N)^{\frac{1}{3}}}+\frac{\sigma^{\frac{p}{2p-1}}\Delta^{\frac{p-1}{2p-1}}}{(\delta N)^{\frac{p}{2p-1}}}.$$

.

308 Corollary 3 is obtained by optimizing K and T in Theorem 5. It implies a sample complexity of G2δ
−1ϵ
−3 + σ p p−1 δ
−1ϵ
−
2p−1 p−1 309 for finding a (*δ, ϵ*)-stationary point, improved over the previous bound
(G + σ)
p p−1 δ
−1ϵ
−
2p−1 p−1 310 [17]. Furthermore, leveraging the adaptive feature of AdaGrad, Corollary 5 311 in Appendix E shows how to set K and T without G, σ, and p, resulting in the first provably rate for 312 O2NC when no problem information is known in advance, which solves the problem asked by [17].

## 313 **5 Conclusion And Limitation**

314 This paper shows that three classical OCO algorithms, OGD, DA, and AdaGrad, can achieve the 315 optimal in-expectation regret under heavy tails without any algorithmic modification if the feasible 316 set is bounded, and provides some applications in stochastic optimization. The main limitation of 317 our work is that all the proof crucially relies on the bounded domain assumption, which may not 318 always be suitable in practice. Finding a weaker sufficient condition, under which the classical OCO
319 algorithms work with heavy tails provably, is a direction worth studying in the future.

## 320 **References**

321 [1] Naman Agarwal, Brian Bullins, Elad Hazan, Sham Kakade, and Karan Singh. Online control 322 with adversarial disturbances. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, 323 *Proceedings of the 36th International Conference on Machine Learning*, volume 97 of Pro324 *ceedings of Machine Learning Research*, pages 111–119. PMLR, 09–15 Jun 2019. URL 325 https://proceedings.mlr.press/v97/agarwal19c.html. 326 [2] Kwangjun Ahn and Ashok Cutkosky. Adam with model exponential moving aver327 age is effective for nonconvex optimization. In A. Globerson, L. Mackey, D. Bel328 grave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, *Advances in Neural* 329 *Information Processing Systems*, volume 37, pages 94909–94933. Curran Associates, 330 Inc., 2024. URL https://proceedings.neurips.cc/paper_files/paper/2024/file/ 331 ac8ec9b4d94c03f0af8c4fe3d5fad4fd-Paper-Conference.pdf.

332 [3] N. Cesa-Bianchi, A. Conconi, and C. Gentile. On the generalization ability of on-line learning 333 algorithms. *IEEE Transactions on Information Theory*, 50(9):2050–2057, 2004. doi: 10.1109/ 334 TIT.2004.833339. 335 [4] Nicolo Cesa-Bianchi and Gabor Lugosi. *Prediction, Learning, and Games*. Cambridge 336 University Press, 2006. 337 [5] Ashok Cutkosky, Harsh Mehta, and Francesco Orabona. Optimal stochastic non-smooth non338 convex optimization through online-to-non-convex conversion. In Andreas Krause, Emma 339 Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, 340 *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of Pro341 *ceedings of Machine Learning Research*, pages 6643–6670. PMLR, 23–29 Jul 2023. URL 342 https://proceedings.mlr.press/v202/cutkosky23a.html. 343 [6] Damek Davis, Dmitriy Drusvyatskiy, Yin Tat Lee, Swati Padmanabhan, and Guanghao Ye. A 344 gradient sampling method with complexity guarantees for lipschitz functions in high and low 345 dimensions. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, 346 *Advances in Neural Information Processing Systems*, volume 35, pages 6692–6703. Curran 347 Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/ 348 2022/file/2c8d9636f74d0207ff4f65956010f450-Paper-Conference.pdf.

349 [7] Aaron Defazio, Ashok Cutkosky, Harsh Mehta, and Konstantin Mishchenko. Optimal linear 350 decay learning rate schedules and further refinements. *arXiv preprint arXiv:2310.07831*, 2023. 351 [8] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning 352 and stochastic optimization. *Journal of Machine Learning Research*, 12(61):2121–2159, 2011. 353 URL http://jmlr.org/papers/v12/duchi11a.html. 354 [9] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning 355 and stochastic optimization. *Journal of machine learning research*, 12(7), 2011.

356 [10] Elad Hazan. Introduction to online convex optimization. *Foundations and Trends®* in 357 *Optimization*, 2(3-4):157–325, 2016. ISSN 2167-3888. doi: 10.1561/2400000013. URL
358 http://dx.doi.org/10.1561/2400000013.

359 [11] Elad Hazan and Karan Singh. Introduction to online control, 2025. URL https://arxiv.

360 org/abs/2211.09619. 361 [12] Liam Hodgkinson and Michael Mahoney. Multiplicative noise and heavy tails in stochastic 362 optimization. In *International Conference on Machine Learning*, pages 4262–4274. PMLR, 363 2021. 364 [13] Michael Jordan, Guy Kornowski, Tianyi Lin, Ohad Shamir, and Manolis Zampetakis. De365 terministic nonsmooth nonconvex optimization. In Gergely Neu and Lorenzo Rosasco, 366 editors, *Proceedings of Thirty Sixth Conference on Learning Theory*, volume 195 of Pro367 *ceedings of Machine Learning Research*, pages 4570–4597. PMLR, 12–15 Jul 2023. URL 368 https://proceedings.mlr.press/v195/jordan23a.html.

369 [14] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv preprint* 370 *arXiv:1412.6980*, 2014. 371 [15] Guy Kornowski and Ohad Shamir. Oracle complexity in nonsmooth nonconvex optimiza372 tion. *Journal of Machine Learning Research*, 23(314):1–44, 2022. URL http://jmlr.org/ 373 papers/v23/21-1507.html. 374 [16] Guy Kornowski and Ohad Shamir. On the complexity of finding small subgradients in non375 smooth optimization. In *OPT 2022: Optimization for Machine Learning (NeurIPS 2022* 376 *Workshop)*, 2022. URL https://openreview.net/forum?id=SaRQ4oTqWbP.

377 [17] Langqi Liu, Yibo Wang, and Lijun Zhang. High-probability bound for non-smooth non-convex 378 stochastic optimization with heavy tails. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, 379 Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, *Proceedings* 380 *of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of* 381 *Machine Learning Research*, pages 32122–32138. PMLR, 21–27 Jul 2024. URL https: 382 //proceedings.mlr.press/v235/liu24bo.html. 383 [18] Zijian Liu and Zhengyuan Zhou. Stochastic nonsmooth convex optimization with heavy-tailed 384 noises: High-probability bound, in-expectation rate and initial distance adaptation. *arXiv* 385 *preprint arXiv:2303.12277*, 2023. 386 [19] Zijian Liu and Zhengyuan Zhou. Revisiting the last-iterate convergence of stochastic gradient 387 methods. In *The Twelfth International Conference on Learning Representations*, 2024. URL 388 https://openreview.net/forum?id=xxaEhwC1I4.

389 [20] Zijian Liu and Zhengyuan Zhou. Nonconvex stochastic optimization under heavy-tailed noises: 390 Optimal convergence without gradient clipping. In *The Thirteenth International Conference on* 391 *Learning Representations*, 2025. URL https://openreview.net/forum?id=NKotdPUc3L. 392 [21] Brendan McMahan. Follow-the-regularized-leader and mirror descent: Equivalence theo393 rems and l1 regularization. In Geoffrey Gordon, David Dunson, and Miroslav Dudík, edi394 tors, *Proceedings of the Fourteenth International Conference on Artificial Intelligence and* 395 *Statistics*, volume 15 of *Proceedings of Machine Learning Research*, pages 525–533, Fort 396 Lauderdale, FL, USA, 11–13 Apr 2011. PMLR. URL https://proceedings.mlr.press/ 397 v15/mcmahan11b.html. 398 [22] H Brendan McMahan and Matthew Streeter. Adaptive bound optimization for online convex 399 optimization. *arXiv preprint arXiv:1002.4908*, 2010. 400 [23] H. Brendan McMahan and Matthew J. Streeter. Adaptive bound optimization for online convex 401 optimization. In *Conference on Learning Theory (COLT)*, pages 244–256. Omnipress, 2010. 402 [24] Arkadi Nemirovski and David Yudin. Problem complexity and method efficiency in optimization. 403 *Wiley-Interscience*, 1983. 404 [25] Yurii Nesterov. Primal-dual subgradient methods for convex problems. Mathematical program405 *ming*, 120(1):221–259, 2009.

406 [26] Ta Duy Nguyen, Thien H Nguyen, Alina Ene, and Huy Nguyen. Improved convergence in 407 high probability of clipped gradient methods with heavy tailed noise. In A. Oh, T. Nau408 mann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neu409 *ral Information Processing Systems*, volume 36, pages 24191–24222. Curran Associates, 410 Inc., 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/file/ 411 4c454d34f3a4c8d6b4ca85a918e5d7ba-Paper-Conference.pdf. 412 [27] Francesco Orabona. A modern introduction to online learning. *arXiv preprint arXiv:1912.13213*, 413 2019.

414 [28] Francesco Orabona. Last iterate of sgd converges (even in unbounded 415 domains). 2020. URL https://parameterfree.com/2020/08/07/ 416 last-iterate-of-sgd-converges-even-in-unbounded-domains/.

417 [29] Daniela Angela Parletta, Andrea Paudice, and Saverio Salzo. An improved analysis of the 418 clipped stochastic subgradient method under heavy-tailed noise, 2025. URL https://arxiv.

419 org/abs/2410.00573. 420 [30] Maxim Raginsky and Alexander Rakhlin. Information complexity of black-box convex op421 timization: A new look via feedback information theory. In *2009 47th Annual Allerton* 422 *Conference on Communication, Control, and Computing (Allerton)*, pages 803–510, 2009. doi: 423 10.1109/ALLERTON.2009.5394945. 424 [31] Alexander Rakhlin, Ohad Shamir, and Karthik Sridharan. Making gradient descent optimal for 425 strongly convex stochastic optimization. *arXiv preprint arXiv:1109.5647*, 2011. 426 [32] Herbert Robbins and Sutton Monro. A Stochastic Approximation Method. *The Annals of* 427 *Mathematical Statistics*, 22(3):400 - 407, 1951. doi: 10.1214/aoms/1177729586. URL https: 428 //doi.org/10.1214/aoms/1177729586. 429 [33] R Tyrrell Rockafellar. *Convex analysis*, volume 28. Princeton university press, 1997. 430 [34] Abdurakhmon Sadiev, Marina Danilova, Eduard Gorbunov, Samuel Horváth, Gauthier Gidel, 431 Pavel Dvurechensky, Alexander Gasnikov, and Peter Richtárik. High-probability bounds for 432 stochastic optimization and variational inequalities: the case of unbounded variance. In Andreas 433 Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan 434 Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*,
435 volume 202 of *Proceedings of Machine Learning Research*, pages 29563–29648. PMLR, 23–29 436 Jul 2023. URL https://proceedings.mlr.press/v202/sadiev23a.html.

437 [35] Shai Shalev-Shwartz. Online learning and online convex optimization. *Foundations and Trends®* 438 *in Machine Learning*, 4(2):107–194, 2012. ISSN 1935-8237. doi: 10.1561/2200000018. URL 439 http://dx.doi.org/10.1561/2200000018. 440 [36] Ohad Shamir and Tong Zhang. Stochastic gradient descent for non-smooth optimization: 441 Convergence results and optimal averaging schemes. In Sanjoy Dasgupta and David McAllester, 442 editors, *Proceedings of the 30th International Conference on Machine Learning*, volume 28 of 443 *Proceedings of Machine Learning Research*, pages 71–79, Atlanta, Georgia, USA, 17–19 Jun 444 2013. PMLR. URL https://proceedings.mlr.press/v28/shamir13.html. 445 [37] Umut Simsekli, Levent Sagun, and Mert Gurbuzbalaban. A tail-index analysis of stochastic 446 gradient noise in deep neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, 447 editors, *Proceedings of the 36th International Conference on Machine Learning*, volume 97 of 448 *Proceedings of Machine Learning Research*, pages 5827–5837. PMLR, 09–15 Jun 2019. URL 449 https://proceedings.mlr.press/v97/simsekli19a.html. 450 [38] Lai Tian and Anthony Man-Cho So. No dimension-free deterministic algorithm computes 451 approximate stationarities of lipschitzians. *Mathematical Programming*, 208(1):51–74, 2024.

452 [39] Lai Tian, Kaiwen Zhou, and Anthony Man-Cho So. On the finite-time complexity and prac453 tical computation of approximate stationarity concepts of Lipschitz functions. In Kamalika 454 Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, 455 *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of Pro456 *ceedings of Machine Learning Research*, pages 21360–21379. PMLR, 17–23 Jul 2022. URL 457 https://proceedings.mlr.press/v162/tian22a.html. 458 [40] Tijmen Tieleman, Geoffrey Hinton, et al. Lecture 6.5-rmsprop: Divide the gradient by a running 459 average of its recent magnitude. *COURSERA: Neural networks for machine learning*, 4(2): 460 26–31, 2012. 461 [41] Nuri Mert Vural, Lu Yu, Krishna Balasubramanian, Stanislav Volgushev, and Murat A Erdogdu. 462 Mirror descent strikes again: Optimal stochastic convex optimization under infinite noise 463 variance. In Po-Ling Loh and Maxim Raginsky, editors, *Proceedings of Thirty Fifth Conference* 464 *on Learning Theory*, volume 178 of *Proceedings of Machine Learning Research*, pages 65–102.

465 PMLR, 02–05 Jul 2022. URL https://proceedings.mlr.press/v178/vural22a.html.

466 [42] Rachel Ward, Xiaoxia Wu, and Leon Bottou. AdaGrad stepsizes: Sharp convergence over 467 nonconvex landscapes. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, *Proceedings* 468 *of the 36th International Conference on Machine Learning*, volume 97 of *Proceedings of* 469 *Machine Learning Research*, pages 6677–6686. PMLR, 09–15 Jun 2019. URL https:// 470 proceedings.mlr.press/v97/ward19a.html. 471 [43] Lin Xiao. Dual averaging method for regularized stochastic learning and online opti472 mization. In Y. Bengio, D. Schuurmans, J. Lafferty, C. Williams, and A. Culotta, edi473 tors, *Advances in Neural Information Processing Systems*, volume 22. Curran Associates, 474 Inc., 2009. URL https://proceedings.neurips.cc/paper_files/paper/2009/file/ 475 7cce53cf90577442771720a370c3c723-Paper.pdf. 476 [44] Moslem Zamani and François Glineur. Exact convergence rate of the last iterate in subgradient 477 methods. *arXiv preprint arXiv:2307.11134*, 2023. 478 [45] Jingzhao Zhang, Sai Praneeth Karimireddy, Andreas Veit, Seungyeon Kim, Sashank Reddi, 479 Sanjiv Kumar, and Suvrit Sra. Why are adaptive methods good for attention models? In 480 H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neu481 *ral Information Processing Systems*, volume 33, pages 15383–15393. Curran Associates, 482 Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/file/ 483 b05b57f6add810d3b7490866d74c0053-Paper.pdf. 484 [46] Jingzhao Zhang, Hongzhou Lin, Stefanie Jegelka, Suvrit Sra, and Ali Jadbabaie. Complexity of 485 finding stationary points of nonconvex nonsmooth functions. In Hal Daumé III and Aarti Singh, 486 editors, *Proceedings of the 37th International Conference on Machine Learning*, volume 119 487 of *Proceedings of Machine Learning Research*, pages 11173–11182. PMLR, 13–18 Jul 2020. 488 URL https://proceedings.mlr.press/v119/zhang20p.html. 489 [47] Jiujia Zhang and Ashok Cutkosky. Parameter-free regret in high probability with heavy tails. 490 In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, *Advances* 491 *in Neural Information Processing Systems*, volume 35, pages 8000–8012. Curran Associates, 492 Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/ 493 349956dee974cfdcbbb2d06afad5dd4a-Paper-Conference.pdf. 494 [48] Qinzi Zhang and Ashok Cutkosky. Random scaling and momentum for non-smooth non-convex 495 optimization. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria 496 Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, *Proceedings of the 41st International* 497 *Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*,
498 pages 58780–58799. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/
499 v235/zhang24k.html. 500 [49] Tong Zhang. Solving large scale linear prediction problems using stochastic gradient descent 501 algorithms. In *Proceedings of the twenty-first international conference on Machine learning*, 502 page 116, 2004. 503 [50] Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In 504 *Proceedings of the 20th international conference on machine learning (icml-03)*, pages 928–936, 505 2003.

## 506 **A Missing Proofs For Online Gradient Descent**

507 This section provides missing proofs for regret bounds of OGD. Before showing the formal proof, 508 we recall the following core inequality that holds for any x ∈ X given in (6):

$$\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}\rangle\leq{\frac{\left\|\mathbf{x}_{t}-\mathbf{x}\right\|^{2}-\left\|\mathbf{x}_{t+1}-\mathbf{x}\right\|^{2}}{2\eta_{t}}}+\eta_{t}G^{2}+\mathsf{C}(\mathsf{p})\eta_{t}^{p-1}\left\|\mathbf{\epsilon}_{t}\right\|^{\mathsf{p}}D^{2-\mathsf{p}}.$$
$$(10)$$

509 The key to establishing the above result is showing

$$\left\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}_{t+1}\right\rangle-{\frac{\left\|\mathbf{x}_{t}-\mathbf{x}_{t+1}\right\|^{2}}{2\eta_{t}}}\leq\eta_{t}G^{2}+\mathsf{C}(\mathsf{p})\eta_{t}^{\mathsf{p}-1}\left\|\mathbf{\epsilon}_{t}\right\|^{\mathsf{p}}D^{2-\mathsf{p}},$$

510 the proof of which is by combining (3), (4), and (5) established in the main text.

## 511 **A.1 Proof Of Theorem 1**

Proof. For any x ∈ X , sum up (10) from t = 1 to T and drop the term −
∥xT +1−x∥
2 2ηT
512 to obtain

X T t=1 ⟨gt, xt − x⟩ ≤ ∥x1 − x∥ 2 2η1 + T X−1 t=1 1 ηt+1 − 1 ηt ∥xt+1 − x∥ 2 2+X T t=1 ηtG 2 + C(p)η p−1 t ∥ϵt∥ ≤ D2 ηT +X T t=1 ηtG 2 + C(p)η p−1 t ∥ϵt∥ p D2−p, (13)
$$(11)$$
(12)  $\binom{13}{2}$  (13)  . 
$$(14)$$
p D2−p(12)
513 where the last step is due to ∥xt − x∥ ≤ D, ∀t ∈ [T] and ηt+1 ≤ ηt, ∀t ∈ [T − 1].

514 Taking expectations on both sides of (13) yields that

$$\mathbb{E}\left[R_{T}^{\mathsf{Q G D}}(x)\right]\leq\frac{D^{2}}{\eta_{T}}+\sum_{t=1}^{T}\eta_{t}G^{2}+\mathsf{C}(\mathsf{p})\eta_{t}^{\mathsf{p}-1}\sigma^{\mathsf{p}}D^{2-\mathsf{p}},$$
$\square$
E [⟨gt
, xt − x*⟩ | F*t−1] = ⟨E [gt | Ft−1] , xt − x⟩ = ⟨∇ℓt(xt), xt − x⟩ ≥ ℓt(xt) − ℓt(x), (15)
519

## 520 **A.2 Extension To Online Strongly Convex Optimization**

521 Next, we extend Theorem 1 to the strongly convex case, i.e., ∃µ > 0 such that for all t ∈ [T],
$$\frac{\mu}{2}\left\|\mathbf{x}-\mathbf{y}\right\|^{2}+\langle\nabla\ell_{t}(\mathbf{y}),\mathbf{x}-\mathbf{y}\rangle+\ell_{t}(\mathbf{y})\leq\ell_{t}(\mathbf{x}),\forall\mathbf{x},\mathbf{y}\in\mathcal{X},\nabla\ell_{t}(\mathbf{y})\in\partial\ell_{t}(\mathbf{y}).$$
522 In this setting, it is well known that OGD achieves a logarithmic regret bound when p = 2 [10, 27]. 523 Theorem 6 below provides the first provable result for p < 2.

Theorem 6. *Under Assumption 1 and additionally assuming (16), taking* ηt =
1 µt 524 in OGD *(Algorithm* 525 *1), we have*

$$\mathbb{E}\left[\mathsf{R}_{T}^{\mathsf{QGD}}(\mathbf{x})\right]\lesssim{\frac{G^{2}\left(1+\log T\right)}{\mu}}+{\frac{\sigma^{\mathsf{p}}G^{2-\mathsf{p}}}{\mu}}\times\begin{cases}T^{2-\mathsf{p}}&\mathsf{p}\in(1,2)\\ 1+\log T&\mathsf{p}=2\end{cases},\forall\mathbf{x}\in\mathcal{X}.$$

where for the L.H.S., we use E [⟨gt
$\mathbf{x})]=\mathbb{E}\left[\mathbb{E}\left[\left\langle\mathbf{g}_t,\mathbf{x}_t-\mathbf{x}\right\rangle\mid\mathcal{F}_{t-1}\right]\right]$ and  $\left.\right],\mathbf{x}_t-\mathbf{x})=\left\langle\nabla\ell_t(\mathbf{x}_t),\mathbf{x}_t-\mathbf{x}\right\rangle\geq0$
515 , xt − x*⟩ | F*t−1]] and
for the R.H.S., we use $\mathbb{E}\left[\left\|\boldsymbol{e}_{t}\right\|^{p}\right]\leq\sigma^{p}$.  Finally, we plug $\eta_{t}=\frac{D}{G\sqrt{t}}\wedge\frac{D}{\sigma^{1/p}},\forall t\in[T]$ into (14), then use $\sum_{t=1}^{T}\frac{1}{\sqrt{t}}\lesssim\sqrt{T}$ and $\sum_{t=1}^{T}\frac{1}{t^{1-1/p}}\lesssim T^{1/p}$ to conclude  $$\mathbb{E}\left[\mathsf{R}_{T}^{\mathsf{GCD}}(\boldsymbol{x})\right]\lesssim GD\sqrt{T}+\sigma DT^{1/p}.$$

14 526 Theorem 6 shows that under strongly convexity, OGD for p ∈ (1, 2) achieves a better sublinear regret T
2−pthan T
1/p 527 in Theorem 1 as 2 − p ≤ 1/p, ∀p > 0. One point we highlight here is that the stepsize ηt =
1 µt 528 is commonly used in the OCO literature and is independent of the tail index p.

529 However, in contrast to Theorem 1, we suspect Theorem 6 is not tight in T for p ∈ (1, 2). The reason 530 is that for nonsmooth strongly convex optimization with heavy tails (i.e., ℓt = F, ∀t ∈ [T] where F
is strongly convex), Theorem 6 can convert to a convergence rate only in the order of 1/Tp−1 531 , which is worse than the lower bound 1/T2−2/p 532 [45]. Therefore, we conjecture that a way to obtain a better regret bound than T
2−p 533 exists, which we leave as future work. 534 *Proof of Theorem 6.* For any x ∈ X , we take expectations on both sides of (12) to have

$$\mathbb{E}\left[\mathsf{R}_{T}^{\mathsf{QGD}}(\mathbf{x})\right]\leq\left(\frac{1}{\eta_{1}}-\mu\right)\frac{\left\|\mathbf{x}_{1}-\mathbf{x}\right\|^{2}}{2}+\sum_{t=1}^{T-1}\left(\frac{1}{\eta_{t+1}}-\frac{1}{\eta_{t}}-\mu\right)\frac{\mathbb{E}\left[\left\|\mathbf{x}_{t+1}-\mathbf{x}\right\|^{2}\right]}{2}$$ $$+\sum_{t=1}^{T}\eta_{t}G^{2}+\mathbb{C}(\mathsf{p})\eta_{t}^{p-1}\sigma^{p}D^{2-p},$$  for the LHS of all $\mathbf{x}$ is all the $t$-th order of $\mathbf{x}$, $\mathbf{x}$ is not $t$-th order but $\mathbf{x}$ is not 
$$(17)$$

535 where for the L.H.S., we follow a similar step of reasoning out (15) but instead using

$$\langle\nabla\ell_{t}(\mathbf{x}_{t}),\mathbf{x}_{t}-\mathbf{x}\rangle\geq\ell_{t}(\mathbf{x}_{t})-\ell_{t}(\mathbf{x})+{\frac{\mu}{2}}\left\|\mathbf{x}_{t}-\mathbf{x}\right\|^{2},$$

for the R.H.S., we use E-∥ϵt∥
p≤ σ p 536 .

Next, we plug ηt =
1 µt 537 , ∀t ∈ [T] into (17) to obtain

$$\mathbb{E}\left[\mathsf{R}_{T}^{\mathsf{DG}}(\boldsymbol{x})\right]\lesssim\sum_{t=1}^{T}\frac{G^{2}}{\mu t}+\frac{\sigma^{\mathsf{p}}D^{2-\mathsf{p}}}{\mu^{\mathsf{p}-1}t^{\mathsf{p}-1}}$$ $$\lesssim\frac{G^{2}\left(1+\log T\right)}{\mu}+\frac{\sigma^{\mathsf{p}}D^{2-\mathsf{p}}}{\mu^{\mathsf{p}-1}}\times\begin{cases}T^{2-\mathsf{p}}&\mathsf{p}\in(1,2)\\ 1+\log T&\mathsf{p}=2\end{cases}.$$  Lastly, it is known that if $\ell_{t}$ is $G$-Lipschitz and $\mu$-strongly convex on a domain $\mathcal{X}$ with a diameter $D$

then it satisfies D ≲
G µ 539 (e.g., see Lemma 2 of [31]). Therefore, when p ∈ (1, 2),

$$\mathbb{E}\left[\mathsf{R}_{T}^{\mathrm{\tiny{OGD}}}(\mathbf{x})\right]\lesssim{\frac{G^{2}\left(1+\log T\right)}{\mu}}+{\frac{\sigma^{\mathsf{p}}G^{2-\mathsf{p}}}{\mu}}T^{2-\mathsf{p}}.$$

540

## 541 **B Missing Proofs For Dual Averaging**

542 This section provides missing proofs for regret bounds of DA.

## 543 **B.1 Proof Of Theorem 2**

Proof. Let Lt(x) ≜
∥x−x1∥
2
2ηt−1+Pt−1
$\sum_{s=1}^{t-1}\left\langle\mathbf{g}_{s},\mathbf{x}\right\rangle,\forall t\in[T+1]$, where $\eta_{0}\triangleq\eta_{1}$. Then DA can be 
544 , x⟩, ∀t ∈ [T + 1], where η0 ≜ η1. Then DA can be
545 equivalently written as
xt = argminx∈X Lt(x), ∀t ∈ [T + 1] .

546 By Lemma 7.1 of [27], for any x ∈ X ,

By Lemma 7.1 of [27], for any $\mathbf{x}\in\Lambda$,  $$\sum_{t=1}^{T}\left\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}\right\rangle=\frac{\left\|\mathbf{x}-\mathbf{x}_{1}\right\|^{2}}{2\eta_{T}}+L_{T+1}(\mathbf{x}_{T+1})-L_{T+1}(\mathbf{x})+\sum_{t=1}^{T}L_{t}(\mathbf{x}_{t})+\left\langle\mathbf{g}_{t},\mathbf{x}_{t}\right\rangle-L_{t+1}(\mathbf{x}_{t+1})$$ $$\leq\frac{\left\|\mathbf{x}-\mathbf{x}_{1}\right\|^{2}}{2\eta_{T}}+\sum_{t=1}^{T}L_{t}(\mathbf{x}_{t})-L_{t+1}(\mathbf{x}_{t+1})+\left\langle\mathbf{g}_{t},\mathbf{x}_{t}\right\rangle,$$
$$\square$$
547 where the inequality holds by LT +1(xT +1) ≤ LT +1(x), ∀x ∈ X due to xT +1 =
548 argminx∈X LT +1(x). Note that for any t ∈ [T],

$$L_{t}(\mathbf{x}_{t})-L_{t+1}(\mathbf{x}_{t+1})+\langle\mathbf{g}_{t},\mathbf{x}_{t}\rangle$$
=Lt(xt) − Lt(xt+1) + ⟨gt, xt − xt+1⟩ + ∥xt+1 − x1∥ 2 2ηt−1 − ∥xt+1 − x1∥ 2 2ηt (a) ≤Lt(xt) − Lt(xt+1) + ⟨gt, xt − xt+1⟩ (b) ≤ ⟨gt, xt − xt+1⟩ − ∥xt − xt+1∥ 2 2ηt−1 ,
where (a) is by ηt ≤ ηt−1, ∀t ∈ [T] and (b) is holds because Lt is 1 ηt−1 549 -strongly convex and 550 xt = argminx∈X Lt(x), which together imply

$$L_{t}(\mathbf{x}_{t})-L_{t}(\mathbf{x}_{t+1})\leq\left\langle\nabla L_{t}(\mathbf{x}_{t}),\mathbf{x}_{t}-\mathbf{x}_{t+1}\right\rangle-\frac{\left\|\mathbf{x}_{t}-\mathbf{x}_{t+1}\right\|^{2}}{2\eta_{t-1}}\leq-\frac{\left\|\mathbf{x}_{t}-\mathbf{x}_{t+1}\right\|^{2}}{2\eta_{t-1}}.$$

551 Therefore, we have

$$\sum_{t=1}^{T}\left\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}\right\rangle\leq{\frac{\left\|\mathbf{x}-\mathbf{x}_{1}\right\|^{2}}{2\eta_{T}}}+\sum_{t=1}^{T}\left\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}_{t+1}\right\rangle-{\frac{\left\|\mathbf{x}_{t}-\mathbf{x}_{t+1}\right\|^{2}}{2\eta_{t-1}}}.$$
$$(18)$$
. (18)
552 By the same argument as proving (11) but replacing ηt with ηt−1, there is

$$\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}_{t+1}\rangle-\frac{\|\mathbf{x}_{t}-\mathbf{x}_{t+1}\|^{2}}{2\eta_{t-1}}\leq\eta_{t-1}G^{2}+\mathsf{C}(\mathsf{p})\eta_{t-1}^{\mathsf{p}-1}\,\|\mathbf{\epsilon}_{t}\|^{\mathsf{p}}\,D^{2-\mathsf{p}}.$$

553 As such, we know

$$\sum_{t=1}^{T}\left\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}\right\rangle\leq{\frac{\left\|\mathbf{x}-\mathbf{x}_{1}\right\|^{2}}{2\eta_{T}}}+\sum_{t=1}^{T}\eta_{t-1}G^{2}+\mathsf{C}(\mathsf{p})\eta_{t-1}^{\mathsf{p}-1}\left\|\mathbf{\epsilon}_{t}\right\|^{\mathsf{p}}D^{2-\mathsf{p}}.$$

554 Finally, following similar steps in proving Theorem 1 in Appendix A, we conclude

$$\mathbb{E}\left[R_{T}^{\mathsf{D A}}({\boldsymbol{x}})\right]\leq G D{\sqrt{T}}+\sigma D T^{1/\mathsf{p}}.$$

555

## 556 **B.2 Dual Averaging With An Adaptive Stepsize**

We show that DA with an adaptive stepsize can also achieve the optimal regret GD√T + σDT1/p 557 .

Theorem 7. *Under Assumption 1, taking* ηt = 2DV −1/2 t and Vt =Pts=1 ∥gs∥
2 558 in DA *(Algorithm* 559 *2), we have* E
-R
DA
T(x)≲ GD√T + σDT1/p, ∀x ∈ X .

560 *Proof.* For any x ∈ X , we have

$$\sum_{t=1}^{T}\left\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}\right\rangle\stackrel{(18)}{{\leq}}\frac{\left\|\mathbf{x}-\mathbf{x}_{1}\right\|^{2}}{2\eta_{T}}+\sum_{t=1}^{T}\left\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}_{t+1}\right\rangle-\frac{\left\|\mathbf{x}_{t}-\mathbf{x}_{t+1}\right\|^{2}}{2\eta_{t-1}},$$
$$(19)$$
, (19)
561 where η0 ≜ η1. On the one hand, we can use AM-GM inequality to bound

$$\left\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}_{t+1}\right\rangle-{\frac{\left\|\mathbf{x}_{t}-\mathbf{x}_{t+1}\right\|^{2}}{2\eta_{t-1}}}\leq{\frac{\eta_{t-1}\left\|\mathbf{g}_{t}\right\|^{2}}{2}}.$$

562 On the other hand, we know

$$\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}_{t+1}\rangle-\frac{\|\mathbf{x}_{t}-\mathbf{x}_{t+1}\|^{2}}{2\eta_{t-1}}\leq\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}_{t+1}\rangle\leq\|\mathbf{g}_{t}\|\,\|\mathbf{x}_{t}-\mathbf{x}_{t+1}\|\leq\|\mathbf{g}_{t}\|\,D,\tag{20}$$
563 where the second step is by Cauchy-Schwarz inequality. Therefore, for any t ≥ 2, ⟨gt, xt − xt+1⟩ − ∥xt − xt+1∥ 2 2ηt−1≤ ηt−1 ∥gt∥ 2 2∧ ∥gt∥ D (a) ≤2 2 ηt−1∥gt∥ 2 +1 ∥gt∥D (b) =2D ∥gt∥ 2 qPt−1 s=1 ∥gs∥ 2 + ∥gt∥ (c) ≤2D ∥gt∥ 2 qPts=1 ∥gs∥ 2 , (21) where (a) is due to x ∧ y ≤2 x−1+y−1 , ∀x, y > 0, (b) is by ηt−1 = √2D Pt−1 s=1∥gs∥ 2 564 , and (c) holds because of qPts=1 ∥gs∥ 2 ≤ qPt−1 s=1 ∥gs∥ 2 565 + ∥gt∥. Note that (21) is also true for t = 1 by (20).
Combine (19) and (21) and use $\|\mathbf{x}-\mathbf{x}_{1}\|\leq D$ to obtain  $$\sum_{t=1}^{T}\left\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}\right\rangle\leq\frac{D^{2}}{2\eta_{T}}+\sum_{t=1}^{T}\frac{2D\left\|\mathbf{g}_{t}\right\|^{2}}{\sqrt{\sum_{s=1}^{t}\left\|\mathbf{g}_{s}\right\|^{2}}}=\frac{D^{2}}{2\eta_{T}}+\sum_{t=1}^{T}\eta_{t}\left\|\mathbf{g}_{t}\right\|^{2},$$  which only differs from (22) by a constant. Hence, by a similar proof for (24), there is 
$$\sum_{t=1}^{T}\left\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}\right\rangle\lesssim D\left[{\sqrt{\sum_{t=1}^{T}\|\nabla\ell_{t}(\mathbf{x}_{t})\|^{2}}}+\left(\sum_{t=1}^{T}\|\mathbf{\epsilon}_{t}\|^{p}\right)^{\frac{1}{p}}\right],$$
$\square$
568 implying
$$\mathbb{E}\left[\mathbb{R}_{T}^{\mathsf{D}\mathsf{A}}({\boldsymbol{x}})\right]\lesssim G D{\sqrt{T}}+\sigma D T^{1/\mathsf{p}}.$$

## 569 570 **C Missing Proofs For** Adagrad

571 This section provides missing proofs for regret bounds of AdaGrad.

## 572 **C.1 Proof Of Theorem 3**

Proof. As mentioned, AdaGrad can be viewed as OGD with a stepsize ηt = √
η Vt
= √η Pts=1∥gs∥
2 573 .

574 Therefore, we can use (1) for AdaGrad to know for any x ∈ X ,

$$\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}\rangle\leq{\frac{\left\|\mathbf{x}_{t}-\mathbf{x}\right\|^{2}-\left\|\mathbf{x}_{t+1}-\mathbf{x}\right\|^{2}}{2\eta_{t}}}+{\frac{\eta_{t}\left\|\mathbf{g}_{t}\right\|^{2}}{2}}.$$

Sum up the above inequality from t = 1 to T and drop the term −
the term $-\frac{\left\|\pmb x_{T+1}-\pmb x\right\|^2}{2\eta r}$ to have. 
575 to have
2ηT X T t=1 ⟨gt, xt − x⟩ ≤ ∥x1 − x∥ 2 2η1 + T X−1 t=1 1 ηt+1 − 1 ηt ∥xt+1 − x∥ 2 2+X T t=1 ηt ∥gt∥ 2 2 ≤ D2 2ηT +X T t=1 ηt ∥gt∥ 2 2, (22) 576 where the last step is by ∥xt − x∥ ≤ D, ∀t ∈ [T] and ηt+1 ≤ ηt, ∀t ∈ [T − 1].
$$(22)$$
$$(23)$$
577 Next, observe that for any t ∈ [T],

$$\|\mathbf{g}_{t}\|^{2}=\frac{\eta^{2}}{\eta_{t}^{2}}-\frac{\eta^{2}}{\eta_{t-1}^{2}}=\eta^{2}\left(\frac{1}{\eta_{t}}-\frac{1}{\eta_{t-1}}\right)\left(\frac{1}{\eta_{t}}+\frac{1}{\eta_{t-1}}\right)\leq\frac{2\eta^{2}}{\eta_{t}}\left(\frac{1}{\eta_{t}}-\frac{1}{\eta_{t-1}}\right),$$

578 where 1/η0 should be read as 0. The above inequality implies

$$\sum_{t=1}^{T}{\frac{\eta_{t}\left\|\mathbf{g}_{t}\right\|^{2}}{2}}\leq\eta^{2}\sum_{t=1}^{T}{\frac{1}{\eta_{t}}}-{\frac{1}{\eta_{t-1}}}={\frac{\eta^{2}}{\eta_{T}}}.$$

. (23)
17 579 Combine (22) and (23) to have

$$\sum_{t=1}^{T}\left\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}\right\rangle\leq{\frac{D^{2}}{2\eta_{T}}}+{\frac{\eta^{2}}{\eta_{T}}}=\left({\frac{D^{2}}{2\eta}}+\eta\right){\sqrt{\sum_{t=1}^{T}\|\mathbf{g}_{t}\|^{2}}}.$$

580 Note that there is

that there is  $$\sqrt{\sum_{t=1}^{T}\left\|\mathbf{g}_{t}\right\|^{2}}\leq\sqrt{\sum_{t=1}^{T}2\left\|\nabla\ell_{t}(\mathbf{x}_{t})\right\|^{2}+2\left\|\mathbf{e}_{t}\right\|^{2}}\leq\sqrt{2\sum_{t=1}^{T}\left\|\nabla\ell_{t}(\mathbf{x}_{t})\right\|^{2}}+\sqrt{2\sum_{t=1}^{T}\left\|\mathbf{e}_{t}\right\|^{2}}$$ $$\leq\sqrt{2\sum_{t=1}^{T}\left\|\nabla\ell_{t}(\mathbf{x}_{t})\right\|^{2}}+\sqrt{2}\left(\sum_{t=1}^{T}\left\|\mathbf{e}_{t}\right\|^{\mathbf{p}}\right)^{\frac{1}{\mathbf{p}}},$$
where the last step is due to ∥·∥2 *≤ ∥·∥*p 581 for any p ∈ [1, 2]. Hence, we obtain

$$\sum_{t=1}^{T}\left\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{x}\right\rangle\leq\sqrt{2}\left(\frac{D^{2}}{2\eta}+\eta\right)\left[\sqrt{\sum_{t=1}^{T}\left\|\nabla\ell_{t}(\mathbf{x}_{t})\right\|^{2}}+\left(\sum_{t=1}^{T}\left\|\mathbf{\epsilon}_{t}\right\|^{\rho}\right)^{\frac{1}{\rho}}\right].\tag{24}$$

582 We take expectations on both sides of (24), then apply Hölder's inequality to have

$$\mathbb{E}\left[\left(\sum_{t=1}^{T}\|\epsilon_{t}\|^{\mathsf{P}}\right)^{\frac{1}{\mathsf{P}}}\right]\leq\left(\sum_{t=1}^{T}\mathbb{E}\left[\|\epsilon_{t}\|^{\mathsf{P}}\right]\right)^{\frac{1}{\mathsf{P}}}\leq\sigma T^{\frac{1}{\mathsf{P}}},$$

and finally plug in η = D/√
583 2 to conclude

$$\mathbb{E}\left[\mathbb{R}_{T}^{\mathrm{AdaGrad}}({\boldsymbol{x}})\right]\lesssim G D{\sqrt{T}}+\sigma D T^{1/\mathsf{p}}.$$

584

## 585 **D Missing Proofs For Applications: Nonsmooth Convex Optimization**

586 We prove the following last-iterate convergence result for SGD (i.e., OGD for stochastic optimization)
587 under heavy-tailed noise. The proof of Theorem 8 is inspired by [19, 44].

588 **Theorem 8.** *Under Assumption 1 for* ℓt(x) = F(x)*, for any stepsize* ηt > 0 in OGD *(Algorithm 1),*
589 *we have*

$$\mathbb{E}\left[F(\mathbf{x}_{T})-F(\mathbf{x})\right]\lesssim{\frac{D^{2}}{\sum_{t=1}^{T}\eta_{t}}}+G^{2}\sum_{t=1}^{T}{\frac{\eta_{t}^{2}}{\sum_{s=(t+1)\wedge T}^{T}\eta_{s}}}+\sigma^{p}D^{2-p}\sum_{t=1}^{T}{\frac{\eta_{t}^{p}}{\sum_{s=(t+1)\wedge T}^{T}\eta_{s}}}.$$
$$(25)$$
$$(26)$$
$$(27)$$

590 *Proof.* Given x ∈ X , we recursively define

$$\mathbf{y}_{0}\triangleq\mathbf{x}\quad{\mathrm{and}}\quad\mathbf{y}_{t}\triangleq\left(1-{\frac{w_{t-1}}{w_{t}}}\right)\mathbf{x}_{t}+{\frac{w_{t-1}}{w_{t}}}\mathbf{y}_{t-1},\forall t\in[T]\,,$$

591 in which

$$w_{t}\triangleq{\frac{\eta_{T}}{\sum_{s=t+1}^{T}\eta_{s}}},\forall t\in\{0\}\cup[T-1]\quad{\mathrm{and}}\quad w_{T}\triangleq w_{T-1}=1.$$

Equivalently, yt 592 can be written into a convex combination of x, x1*, . . . ,* xt as

$$\boldsymbol{y}_{t}=\frac{w_{0}}{w_{t}}\boldsymbol{x}+\sum_{s=1}^{t}\frac{w_{s}-w_{s-1}}{w_{t}}\boldsymbol{x}_{s},\forall t\left\{0\right\}\cup\left[T\right].\tag{1}$$  $\boldsymbol{y}_{t}=\frac{w_{0}}{w_{t}}\boldsymbol{x}+\sum_{s=1}^{t}\frac{w_{s}-w_{s-1}}{w_{t}}\boldsymbol{x}_{s},\forall t\left\{0\right\}\cup\left[T\right].$  $\boldsymbol{y}_{t}=\frac{w_{0}}{w_{t}}\boldsymbol{x}+\sum_{s=1}^{t}\frac{w_{s}-w_{s-1}}{w_{t}}\boldsymbol{x}_{s},\forall t\left\{0\right\}\cup\left[T\right].$ 
Therefore, yt 593 also falls into X and satisfies yt ∈ Ft−1.

We invoke (10) for yt 594 to obtain

$$\langle\mathbf{g}_{t},\mathbf{x}_{t}-\mathbf{y}_{t}\rangle\leq\frac{\left\|\mathbf{x}_{t}-\mathbf{y}_{t}\right\|^{2}-\left\|\mathbf{x}_{t+1}-\mathbf{y}_{t}\right\|^{2}}{2\eta_{t}}+\eta_{t}G^{2}+\mathsf{C}(\mathsf{p})\eta_{t}^{p-1}\left\|\mathbf{\epsilon}_{t}\right\|^{p}D^{2-p}.\tag{28}$$  $\mathbf{y}_{t}\in\mathbb{F}_{p}$, there is 
595 Since xt, yt ∈ Ft−1, there is E [⟨gt, xt − yt
⟩] = E [⟨E [gt | Ft−1] , xt − yt
⟩] = E [⟨∇F(xt), xt − yt
⟩] ≥ E [F(xt) − F(yt)] ,

596 where the last step is due to the convexity of F. As such, we can take expectations on both sides of
597 (28) to have E [F(xt) − F(yt)] ≤ E h∥xt − yt∥ 2i− E h∥xt+1 − yt∥ 2i 2ηt + ηtG 2 + C(p)η p−1 t σ pD2−p ≤ E hwt−1 wtxt − yt−1  2i− E h∥xt+1 − yt∥ 2i 2ηt+ ηtG 2 + C(p)η p−1 t σ pD2−p, (29)

$$(30)$$
where the second step is due to ∥xt − yt∥
$\left\|\boldsymbol{x}_t-\boldsymbol{y}_t\right\|^2\leq\left(1-\frac{w_{t-1}}{w_t}\right)\left\|\boldsymbol{x}_t-\boldsymbol{x}_t\right\|^2+\frac{w_{t-1}}{w_t}\left\|\boldsymbol{y}_t\right\|^2$
wtxt − yt−1

2
598 =
wt−1
wtxt − yt−1

2by (25) and the convexity of ∥xt *− ·∥*2
599 . Mutiply both sides of (29) by wtηt and
600 sum up from t = 1 to T to obtain
To from $t=1$ to $T$ to obtain  $$\mathbb{E}\left[\sum_{t=1}^{T}w_{t}\eta_{t}\left(F(\mathbf{x}_{t})-F(\mathbf{y}_{t})\right)\right]$$ $$\leq\frac{w_{0}\left\|\mathbf{x}_{1}-\mathbf{y}_{0}\right\|^{2}-\mathbb{E}\left[w_{T}\left\|\mathbf{x}_{T+1}-\mathbf{y}_{T}\right\|^{2}\right]}{2}+\sum_{t=1}^{T}w_{t t}\eta_{t}^{2}G^{2}+\mathsf{C}(\mathsf{p})w_{t}\eta_{t}^{\mathsf{p}}\sigma^{\mathsf{p}}D^{2-\mathsf{p}}$$ $$\leq\frac{w_{0}D^{2}}{2}+\sum_{t=1}^{T}w_{t}\eta_{t}^{2}G^{2}+\mathsf{C}(\mathsf{p})w_{t}\eta_{t}^{\mathsf{p}}\sigma^{\mathsf{p}}D^{2-\mathsf{p}}.$$

601 Now observe that

$$F(\mathbf{y}_{t})-F(\mathbf{x})\stackrel{{()}}{{\leq}}\frac{w_{0}}{w_{t}}\left(F(\mathbf{x})-F(\mathbf{x})\right)+\sum_{s=1}^{t}\frac{w_{s}-w_{s-1}}{w_{t}}\left(F(\mathbf{x}_{s})-F(\mathbf{x})\right)$$ $$=\sum_{s=1}^{t}\frac{w_{s}-w_{s-1}}{w_{t}}\left(F(\mathbf{x}_{s})-F(\mathbf{x})\right),$$

602 which implies

$$\sum_{t=1}^{T}w_{t}\eta_{t}\left(F(\mathbf{y}_{t})-F(\mathbf{x})\right)\leq\sum_{t=1}^{T}\sum_{s=1}^{t}\left(w_{s}-w_{s-1}\right)\eta_{t}\left(F(\mathbf{x}_{s})-F(\mathbf{x})\right)$$ $$=\sum_{t=1}^{T}\left(w_{t}-w_{t-1}\right)\left(\sum_{s=t}^{T}\eta_{s}\right)\left(F(\mathbf{x}_{t})-F(\mathbf{x})\right).$$  for lower bound the L.H.S. of (20) by 
603 Thus, we can lower bound the L.H.S. of (30) by

$$\sum_{t=1}^{T}w_{t}\eta_{t}\left(F(\mathbf{x}_{t})-F(\mathbf{y}_{t})\right)=\sum_{t=1}^{T}w_{t}\eta_{t}\left(F(\mathbf{x}_{t})-F(\mathbf{x})\right)-w_{t}\eta_{t}\left(F(\mathbf{y}_{t})-F(\mathbf{x})\right)$$ $$\geq\sum_{t=1}^{T}\left[w_{t}\eta_{t}-\left(w_{t}-w_{t-1}\right)\left(\sum_{s=t}^{T}\eta_{s}\right)\right]\left(F(\mathbf{x}_{t})-F(\mathbf{x})\right)$$ $$=w_{T}\eta_{T}\left(F(\mathbf{x}_{T})-F(\mathbf{x})\right),\tag{31}$$

604 where the last step is due to, for t ∈ [T − 1],

$$w_{t}\eta_{t}-(w_{t}-w_{t-1})\left(\sum_{s=t}^{T}\eta_{s}\right)\stackrel{{\eqref{eq:26}}}{{=}}\frac{\eta_{T}}{\sum_{s=t+1}^{T}\eta_{s}}\cdot\eta_{t}-\left(\frac{\eta_{T}}{\sum_{s=t+1}^{T}\eta_{s}}-\frac{\eta_{T}}{\sum_{s=t+1}^{T}\eta_{s}}\right)\left(\sum_{s=t}^{T}\eta_{s}\right)$$ $$=\frac{\eta_{T}}{\sum_{s=t+1}^{T}\eta_{s}}\cdot\eta_{t}-\frac{\eta_{T}}{\sum_{s=t+1}^{T}\eta_{s}}\cdot\eta_{t}=0,$$

and wT
(26)
605 = wT −1 = 1. 606 We plug (31) back into (30) and divide both sides by wT ηT to obtain

E [F(xT ) − F(x)] ≤w0D2 2wT ηT +X T t=1 wtη 2 t wT ηT G 2 + C(p) wtη p t wT ηT σ pD2−p (26) ≲ D2 PT t=1 ηt + G 2X T η 2 t  PT s=(t+1)∧T ηs + σ pD2−pX T η p t  PT s=(t+1)∧T ηs . t=1 t=1

607 608 Equipped with Theorem 8, we show the following anytime last-iterate convergence rate for SGD/OGD. 609 As far as we know, this is the first and the only provable result demonstrating that the last iterate of 610 SGD can converge in heavy-tailed stochastic optimization without gradient clipping. Compared to 611 Corollary 2, the difference is up to an extra logarithmic factor. Therefore, it is nearly optimal.

Corollary 4. *Under Assumption 1 for* ℓt(x) = F(x)*, taking* ηt =D
G
√t
∧D
σt1/p 612 in OGD *(Algorithm* 613 *1), we have*

$$\mathbb{E}\left[F(\mathbf{x}_{T})-F(\mathbf{x})\right]\lesssim{\frac{G D\left(1+\log T\right)}{\sqrt{T}}}+{\frac{\sigma D\left(1+\log T\right)}{T^{1-{\frac{1}{p}}}}}.$$

614 *Proof.* By Theorem 8, we have

E [F(xT ) − F(x)] ≲ D2 PT t=1 ηt + G 2X T η 2 t  PT s=(t+1)∧T ηs + σ pD2−pX T η p t  PTs=(t+1)∧T ηs t=1 t=1 =D2 PT t=1 ηt + G 2   ηT + T X −1 η 2 t  PT s=t+1 ηs ! + σ pD2−p   η p−1 T + T X −1 η p t  PT s=t+1 ηs ! . t=1 t=1 615 For any t ∈ {0} ∪ [T − 1], observe that by Cauchy-Schwarz inequality

$$(T-t)^{2}\leq\left(\sum_{s=t+1}^{T}\frac{1}{\eta_{s}}\right)\left(\sum_{s=t+1}^{T}\eta_{s}\right)\Rightarrow\frac{1}{\sum_{s=t+1}^{T}\eta_{s}}\leq\frac{\sum_{s=t+1}^{T}\frac{1}{\eta_{s}}}{(T-t)^{2}}.$$
616 Thus, there is

$$\mathbb{E}\left[F(\mathbf{x}_{T})-F(\mathbf{x})\right]\lesssim\frac{D^{2}}{T^{2}}\sum_{t=1}^{T}\frac{1}{\eta_{t}}+G^{2}\left(\eta_{T}+\sum_{t=1}^{T-1}\frac{\eta_{t}^{\mathbf{\gamma}}\sum_{s=t+1}^{T}\frac{1}{\eta_{s}}}{(T-t)^{2}}\right)$$ $$\qquad\qquad+\sigma^{\mathbf{\rho}}D^{2-\mathbf{\rho}}\left(\eta_{T}^{\mathbf{\rho}-1}+\sum_{t=1}^{T-1}\frac{\eta_{t}^{\mathbf{\rho}}\sum_{s=t+1}^{T}\frac{1}{\eta_{s}}}{(T-t)^{2}}\right).\tag{32}$$

617 We first bound

$$\sum_{t=1}^{T}{\frac{1}{\eta_{t}}}=\sum_{t=1}^{T}{\frac{G{\sqrt{t}}}{D}}\vee{\frac{\sigma t^{1/p}}{D}}\leq\sum_{t=1}^{T}{\frac{G{\sqrt{t}}}{D}}+{\frac{\sigma t^{1/p}}{D}}\stackrel{<}{\sim}{\frac{G}{D}}T^{3/2}+{\frac{\sigma}{D}}T^{1+1/p},$$