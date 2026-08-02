# Variance-Dependent Regret Lower Bounds For Contextual Bandits

Anonymous Author(s)
Affiliation Address email

## Abstract

1 Variance-dependent regret bounds for linear contextual bandits, which improve upon the classical Oe(d
√K) regret bound to Oe(d qPK
k=1 σ 2k 2 ), where d is the context dimension, K is the number of rounds, and σ 2k 3 is the noise variance in round 4 k, has been widely studied in recent years. However, most existing works focus 5 on the regret upper bounds instead of lower bounds. To our knowledge, the only 6 lower bound is from Jia et al. (2024), which proved that for any eluder dimension delu and total variance budget Λ, there exists an instance with PK
k=1 σ 2 7 k ≤ Λ
for which any algorithm incurs a variance-dependent lower bound of Ω(√
8 deluΛ).

However, this lower bound has a 
√
9 d gap with existing upper bounds. More10 over, it only considers a fixed total variance budget Λ and does not apply to a general variance sequence {σ 2 1*, . . . , σ*2 11 K}. In this paper, to overcome the limita12 tions of Jia et al. (2024), we consider the general variance sequence under two 13 settings. For a prefixed sequence, where the entire variance sequence is revealed 14 to the learner at the beginning of the learning process, we establish a variancedependent lower bound of Ω(d qPK
k=1 σ 2k 15 / log K) for linear contextual bandits.

For an adaptive sequence, where an adversary can generate the variance σ 2k 16 in 17 each round k based on historical observations, we show that when the adversary must generate σ 2k 18 before observing the decision set Dk, a similar lower bound of Ω(d qPK
k=1 σ 2k
/ log6 19 (dK)) holds. In both settings, our results match the up20 per bounds of the SAVE algorithm (Zhao et al., 2023) up to logarithmic factors.

21 Furthermore, if the adversary can generate the variance σk after observing the 22 decision set Dk, we construct a counter-example showing that it is impossible 23 to construct a variance-dependent lower bound if the adversary properly selects 24 variances in collaboration with the learner. Our lower bound proofs use a novel 25 peeling technique that groups rounds by variance magnitude. For each group, 26 we construct separate instances and assign the learner distinct decision sets. We 27 believe this proof technique may be of independent interest.

## 28 1 **Introduction**

29 We consider the linear contextual bandit problem, where each arm is represented by a feature vector 30 and the expected reward is a linear function of this feature vector with an unknown parameter vector. 31 Numerous studies have developed algorithms achieving optimal regret bounds for linear bandits 32 (Chu et al., 2011; Abbasi-Yadkori et al., 2011a). However, while these works establish minimax33 optimal regret bounds in the worst-case, they do not exploit additional problem-dependent structures. 34 Our work focuses on incorporating reward variance information into the analysis, building upon a 35 line of research studying variance-dependent regret bounds for linear bandits (Zhou et al., 2021; 36 Zhang et al., 2021; Zhou and Gu, 2022; Zhao et al., 2022; Kim et al., 2022; Zhao et al., 2023) 37 and general function approximation (Jia et al., 2024), which includes linear bandits as a special Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Do not distribute.

38 case. Notably, Zhao et al. (2023) established a near-optimal regret guarantee without requiring prior 39 knowledge of the variances: 40 **Theorem 1.1** (Theorem 2.3, Zhao et al. 2023). For any linear contextual bandit problem, the regret 41 of the SAVE algorithm in the first K rounds is upper bounded by:

$$\mathrm{Regret}(K)\leq\tilde{O}\Big(d\sqrt{\sum_{k=1}^{K}\sigma_{k}^{2}}+d\Big),$$

where d is the dimension and σ 2 k 42 is the noise variance of the selected action in round k. 43 However, most of these works have focused on developing algorithms with regret upper bound 44 guarantees, while variance-dependent lower bounds remain understudied. The only exception is 45 Jia et al. (2024), which focuses on general function classes with finite eluder dimension delu and 46 provides the following variance-dependent lower bound: 47 **Theorem 1.2** (Theorem 5.1, Jia et al. 2024). For any dimension d ≥ 2, action space size A, number 48 of rounds K ≥ 2, and total variance budget Λ ∈ [0, K], there exists a contextual bandit problem with 49 eluder dimension delu = d, action space size A, and an adversarial sequence of variances satisfying PK
k=1 σ 2 50 k ≤ Λ such that for any algorithm, the regret is lower bounded by:

## Regret(K) ≥ Ωmin(√Dλ + D, √Ak).

When restricted to the linear bandit case, where d ≥
√A, the above lower bound reduces to √
51 dΛ,
which has a gap of 
√
52 d factor compared with the upper bound in Zhao et al. (2023). Moreover, Jia 53 et al. (2024) only considers instances with a fixed budget Λ and relies on carefully designed variance sequences {σ 21, σ22*, . . . , σ*2 54 K}, failing to provide lower bounds for general variance sequences.

55 Therefore, an open question arises:
56 *Can we prove variance-dependent regret lower bounds for general variance sequences?*

## 57 1.1 **Our Contributions**

58 In this paper, we answer this question affirmatively by constructing hard-to-learn instances in several different settings. For any prefixed sequence {σ 21
, . . . , σ2K}, we achieve a Ω( e d qPK
k=1 σ 2 k 59 )
60 variance-dependent expected lower bound, which matches the upper bound in Zhao et al. (2023) 61 up to logarithmic factors and demonstrates its optimality. For general adaptive variance sequences where a weak adversary (potentially collaborating with the learner) can generate variance σ 2k 62 in each 63 round k based on historical observations, our instance provides a high-probability lower bound of Ω( e d qPK
k=1 σ 2 k 64 ), which also matches the upper bound in Zhao et al. (2023) up to logarithmic fac65 tors. To the best of our knowledge, this is the first high-probability lower bound for linear contextual 66 bandit. 67 Our construction and analysis rely on the following new techniques: 68 - A peeling technique for prefixed variance sequences that divides rounds into groups based on 69 variance magnitude. Through orthogonal decision set construction, each group only interacts with 70 its corresponding parameters, allowing us to establish separate lower bounds for different variance 71 scales and combine them effectively. 72 - A multi-instance framework that handles unknown group sizes in the adaptive setting. For each 73 variance group, we maintain multiple instances designed for different possible intervals of round 74 numbers and assign the learner to these instances in a cyclic manner, ensuring uniform visits 75 across instances and guaranteeing the visiting times of one instance matches its designed interval. 76 - A high-probability lower bound that handles adaptive group sizes through a union bound. We 77 first convert expected regret bounds to constant-probability bounds through careful variance con78 trol and auxiliary algorithms, then boost these to high-probability bounds by creating multiple 79 independent instances.

80 Furthermore, we also study the setting with a strong adversary that can generate the variance σk 81 after observing the decision set Dk. Under this scenario, we proposed a counter algorithm that can 82 collaborate with the adversary by properly selecting variance, achieving an O(d) regret even the total variance PK
k=1 σ 2 83 k = Ω(K). This implies that it is impossible to derive a variance-dependent 84 lower bound for general variance sequence with strong adversary. As a direct extension of this result, 85 we also show that it is impossible to derive a variance-dependent lower bound for stochastic linear 86 bandits, where the decision set is fixed even for a general prefixed variance sequence. 87 **Notation** We use lower case letters to denote scalars, and use lower and upper case bold face letters 88 to denote vectors and matrices respectively. We denote by [n] the set {1*, . . . , n*}. For a vector x ∈ R
dand a positive semi-definite matrix Σ ∈ R
d×d 89 , we denote by ∥x∥2 the vector's ℓ2 norm and by ∥x∥Σ =
√
90 x⊤Σx the Mahalanobis norm. For two positive sequences {an} and {bn} with 91 n = 1, 2*, . . .* , we write an = O(bn) if there exists an absolute constant C > 0 such that an ≤ Cbn 92 holds for all n ≥ 1 and write an = Ω(bn) if there exists an absolute constant C > 0 such that 93 an ≥ Cbn holds for all n ≥ 1. We use Oe(·) to further hide the polylogarithmic factors. We use 1{·}
94 to denote the indicator function.

## 95 2 **Related Work**

96 **Heteroscedastic Linear Bandits.** For linear bandit problems, the worst-case regret has been widely 97 studied (Auer, 2002; Dani et al., 2008; Li et al., 2010; Chu et al., 2011; Abbasi-Yadkori et al., 2011b; Li et al., 2019), achieving Oe(
√
98 K) bounds in the first K rounds. Recently, a series of works has 99 considered heteroscedastic variants where noise distributions vary across rounds. Kirschner and 100 Krause (2018) first formally proposed a linear bandit model with heteroscedastic noise, assuming 101 σk-sub-Gaussian noise in round k ∈ [K]. Subsequently, (Zhou et al., 2021; Zhang et al., 2021; Kim 102 et al., 2021; Zhou and Gu, 2022; Dai et al., 2022; Zhao et al., 2023; Jia et al., 2024) relaxed this to variance-based constraints where round k has variance σ 2k 103 . Among these works, Zhou et al. (2021)
and Zhou and Gu (2022) obtained near-optimal regret guarantees of Oe(d qPK
k=1 σ 2 k 104 ), but required 105 knowledge of σk after observing the reward in round k. In contrast, Zhang et al. (2021); Kim et al.

106 (2021) handled unknown variances with computationally inefficient algorithms, achieving a weaker Oe(poly(d)
qPK
k=1 σ 2 k 107 ) bound. Recently, Zhao et al. (2023) improved upon these results with an efficient algorithm (SAVE) achieving the near-optimal Oe(d qPK
k=1 σ 2k 108 ) bound without requiring 109 variance knowledge. Beyond standard linear bandits, two directions have been explored. Dai et al. 110 (2022) studied heteroscedastic sparse linear bandits, providing a framework to convert standard 111 algorithms to the sparse setting. In a different direction, Jia et al. (2024) extended the analysis 112 to contextual bandits with general function classes having finite eluder dimension, which includes 113 linear bandits as a special case, and achieved a variance-dependent regret upper bounds. 114 **Lower Bounds for Linear Contextual Bandits.** For linear contextual bandit problems, several 115 works (Dani et al., 2008; Chu et al., 2011; Li et al., 2019) have established theoretical lower bounds 116 to illustrate the fundamental difficulty in learning process. For linear bandits with finite action sets, Chu et al. (2011) established an Ω( e
√
117 dK) lower bound, matching the upper bound up to logarithmic 118 factors in the action set size and number of rounds K. For general stochastic linear bandits, Dani et al. (2008) constructed an instance with 2 Ω(d)actions and obtained an Ω(d
√
119 K) lower bound. 120 Later, Li et al. (2019) focused on linear contextual bandits, where the decision set can vary across rounds, and provided an Ω(d
√
121 K log K) lower bound. However, all these works only focus on 122 worst-case regret bounds and do not consider the heteroscedastic variance information. The only exception is Jia et al. (2024), which provided an Ω(√
123 dΛ) variance-dependent lower bound for a 124 fixed total variance budget Λ. Nevertheless, this work cannot handle general variance sequences and 125 leaves open the question of variance-dependent lower bounds in the general setting.

126 3 **Preliminaries**
127 In this work, we consider the heteroscedastic linear contextual bandit (Zhou et al., 2021; Zhang 128 et al., 2021), where the noise variance varies across rounds. Let K be the total number of rounds. In 129 each round k ∈ [K], the interaction between the learner and the environment proceeds as follows:
1. The environment generates an arbitrary decision set Dk ⊆ R
d 130 , where each element repre131 sents a feasible action that can be selected by the learner; 132 2. The learner observes Dk and selects xk ∈ Dk; 133 3. The environment generates the stochastic noise ϵk and reveals the stochastic reward rk =
⟨µ, xk⟩ + ϵk to the learner, where µ ∈ R
d 134 is the unknown weight vector for the underlying 135 linear reward function.

136 Without loss of generality, we assume the random noise ϵk in each round k satisfies:
P(|ϵk| ≤ R) = 1, E[ϵk|x1:k, ϵ1:k−1] = 0, E[ϵ 2k |x1:k, ϵ1:k−1] = σ 2k ≤ 1, ∀k ∈ [K] (3.1)
137 For any algorithm Alg and linear bandit instance M, the cumulative regret is defined as follows:

$$\operatorname{Regret}_{\mathrm{Alg}}(K,{\mathcal{M}})=\sum_{k\in[K]}\langle\mathbf{x}_{k}^{*},\boldsymbol{\mu}\rangle-\langle\mathbf{x}_{k},\boldsymbol{\mu}\rangle,\quad{\mathrm{where~}}\mathbf{x}_{k}^{*}=\operatorname*{argmax}_{\mathbf{x}\in{\mathcal{D}}_{k}}\langle\mathbf{x},\boldsymbol{\mu}\rangle.$$
$\downarrow$ . 
$\left(\mathbf{\hat{J}}_{\mu}\right)$
⟨x, µ⟩. (3.2)
138 For simplicity, we may omit the subscripts Alg and/or M when there is no ambiguity. Additionally, with a slight abuse of notation, we may use σk to represent the variance σ 2 k 139 (which is originally 140 the standard deviation) when there is no ambiguity. In this work, we focus on providing variance141 dependent lower bounds for the regret based on the variances sequence {σ1*, ..., σ*K}. We consider 142 two settings for the variance sequence {σ1*, . . . , σ*K}:
143 - **Prefixed Sequence**: The variance sequence is revealed to the learner at the beginning of 144 the learning process. 145 - **Adaptive Sequence**: An adversary (potentially collaborating with the learner) can generate 146 the variance σk in each round k based on historical observations, with the learner receiving 147 each variance at the beginning of the corresponding round. This setting can be further 148 divided into two categories based on the power of the adversary: 149 - **Weak Adversary**: The adversary must generate the variance σk before observing the 150 decision set Dk. 151 - **Strong Adversary**: The adversary can generate the variance σk after observing the 152 decision set Dk.

153 **Remark 3.1.** Unlike the typical adversarial setting focused on maximizing regret for a specific 154 algorithm, our work uses the idea of an "adversary" to represent the environment's inherent ability to 155 select the variance sequence. This "adversary" might even strategically choose variance levels (σk)
156 based on the past decision sets Dk **observed so far**, potentially leading to variance levels that could 157 temporarily improve the learner's performance or make the learning process appear easier. This 158 seeming "cooperation," however, is ultimately aimed at exploring the fundamental lower bounds on 159 regret that must hold for any learner in any environment. The key is that the variance is chosen 160 **without direct knowledge of the true underlying patterns** µ. When this "adversary" (our "strong 161 adversary") can adjust the variance based on the learner's actions (Dk), this strategic "cooperation,"
162 informed by past observations but blind to µ, becomes more effective in probing the true limits of 163 learnability and challenging our lower bound results.

## 164 4 **Variance-Dependent Lower Bound With Prefixed Variance Sequence**

165 In this section, we consider the setting where the variance sequence {σ1*, . . . , σ*K} is prefixed and 166 fully revealed to the learner at the beginning of the learning process.

167 4.1 **Main Results** 168 We establish the following theorem for the variance-dependent lower bound. 169 **Theorem 4.1.** For any dimension d > 1, prefixed sequence of variance {σ1*, ..., σ*K} satisfying PK
k=1 σ 2k ≥ 1 + 384d 2 170 and algorithm Alg, there exists a hard linear contextual bandit instance such 171 that each action a ∈ Dk in round k has variance bounded by σk. For this instance, the expected 172 regret of algorithm Alg over K rounds is lower bounded by:

$$\mathbb{E}[\mathrm{Regret}(K)]\geq\Omega{\Big(}d{\sqrt{\sum_{i=1}^{K}\sigma_{k}^{2}}}/(\log K){\Big)}.$$

173 **Remark 4.2.** For a prefixed sequence {σ1*, ..., σ*K}, Theorem 4.1 shows that any algorithm incurs a regret lower bounded of Ω( e d qPK
k=1 σ 2 k 174 ), which matches the upper bound in Zhao et al. (2023) up 175 to logarithmic factors. Compared to the lower bound in Jia et al. (2024), Theorem 4.1 focuses on the linear contextual bandit setting and achieves a 
√
176 d improvement over the standard linear bandit 177 setting. It is also worth noting that the lower bound in Jia et al. (2024) only considers instances with a fixed total variance PK
k=1 σ 2 k 178 , constructed by using constant variance in the early rounds and zero 179 variance in later rounds. In comparison, Theorem 4.1 applies to any fixed variance sequence and is 180 more flexible.

In Theorem 4.1, we require that the total variance is no less than Ω(d 2), which reduces to K ≥ Ω(d 2 181 )
182 when all variances σk = 1. A similar requirement exists in standard linear bandits, since a trivial lower bound of Ω(K) always holds for any algorithm, and the lower bound of Ω(d
√
183 K) can only be achieved when K ≥ Ω(d 2 184 ). Furthermore, for general sequences of variances with total variance smaller than O(d 2 185 ), a large number of rounds K alone is not sufficient to establish the desired 186 lower bound. The presence of early rounds with zero variance would increase the total number of 187 rounds without affecting the fundamental complexity of the problem. This observation suggests that requiring total variance no less than Ω(d 2 188 ) (or other equivalent conditions) may be necessary for 189 establishing the lower bound.

## 190 4.2 **Proof Of Theorem 4.1**

191 In this subsection, we prove the variance-dependent lower bound in Theorem 4.1. We first start 192 with a fixed variance threshold σ, and construct a class of hard-to-learn instances where actions are chosen from a hypercube action set A = {−1, 1}
d 193 , and for any action a ∈ A, the reward follows a scaled Bernoulli distribution σ ·B(1/3 +⟨µ, a⟩), where ∆ = 1/
√96K and µ *∈ {−*∆, ∆}
d 194 . In this setting, the variance for each action is upper bounded by σ 2 195 , and these instances can be represented as a linear bandit problem with feature (*σ, σ* · a) and weight vector µ 196 ′ = (1/3, µ). Based on these 197 hard-to-learn instances, we have the following variance-dependent lower bound for the regret: 198 **Lemma 4.3.** For a fixed variance threshold σ and any bandit algorithm Alg, if the weight vector µ ∈
{−∆, ∆}
dis uniformly random selected from {−∆, ∆}
d 199 , the variance in each round is bounded by σ 2, and the expected regret over K ≥ 1.5 · d 2 200 rounds is lower bounded by:

## Eµ[Regret(K)] ≥ D

√Kσ2/8
√6.

201 **Remark 4.4.** Lemma 4.3 establishes a variance-dependent lower bound for the regret with a fixed
202 variance threshold σ. When all variances are equal (σ1 = ... = σK = σ), this bound matches the
203 upper bound in Zhao et al. (2023) up to logarithmic factors. In addition, under this fixed-variance 204 setting, this lemma provides a tighter logarithmic dependency on the number of rounds K compared 205 to Theorem 4.1, though it does not extend to dynamic variances.
206 Now, for any prefixed variance sequence {σ1*, ..., σ*K}, we divide the rounds into L = ⌈log2 K⌉ + 1
207 different groups based on the range of their variance as follows:
$${\mathcal{K}}_{0}=\{k:\sigma_{k}\leq1/K\},$$
Ki = {k : 2i−1*/K < σ*k ≤ 2
i/K}, for i = 1*, . . . , L* − 1.

208 For each group Ki with i ∈ [L − 1], we construct a bandit instance Mi with weight vector µi 209 following Lemma 4.3, where:
- the variance threshold is set to be σ(i) = 2i−1 210 /K;
- the number of rounds is Ki = |Ki 211 |; 212 - the dimension is di = d/L.

213 For group K0, we construct a different type of instance M0: a d/L-armed bandit, where one ran214 domly chosen arm gives constant reward 1 while all other arms give reward 0. Note that this instance 215 in M0 can be equivalently represented as a d0 = d/L-dimensional linear bandit where actions are one-hot vectors ei 216 . 217 Based on these sub-instances, we create a combined linear bandit instance with dimension 218 d0 + d1 + ... + dL−1 = d with weight vector µ = (µ0*, ...,* µL−1): At the beginning of each round k, if round k belongs to group Ki 219 , then the learner receives the decision set Dk =
(0d0, ..., 0di−1, x, 0di+1 *, ...,* 0dL−1) : x ∈ Ai	, where 0dj 220 corresponds to a zero vector with dimension dj and Aiis the action set in the bandit instance Mi 221 . Under this construction, for any round k ∈ Ki, the reward in the combined instance coincides with that of sub-instance Mi 222 . Specifically, 223 after the learner selects action x, they receive a reward drawn from a scaled Bernoulli distribution with variance upper bounded by σ 2(i) = 2 i−1/K2 224 for i ̸= 0, and variance 0 for i = 0. Note that in all groups, the variance is bounded by σ 2 k 225 . With this construction in hand, we now proceed to 226 prove the lower bound in Theorem 4.1. 227 **Remark 4.5** (Linear Contextual Bandits vs. Stochastic Linear Bandits). In the proof of Theorem 228 4.1, we heavily rely on assigning different decision sets to rounds in the contextual bandit envi229 ronment. This approach, however, does not extend to stochastic linear bandit problems, where all 230 rounds share the same decision set. To see this limitation, consider any prefixed variance sequence 231 with σ1 = *· · ·* = σd = 0. In this case, the learner can select canonical basis of the decision set in 232 the first d rounds. Since these rounds have zero variance, the learner learns the exact rewards for 233 all actions in the decision set and incurs no regret in subsequent rounds, regardless of the values of σd+1*, . . . , σ*K. Consequently, it is impossible to establish a lower bound of Ω( e d qPK
k=1 σ 2 k 234 ) in this 235 setting. 236 *Proof of Theorem 4.1.* Due to the orthogonal construction of decision sets across different groups Ki 237 , actions in group Ki provide no information about the weight vector µj for j ̸= i. Consequently, 238 the total regret can be decomposed into the sum of regrets from each sub-instance. For each sub239 instance Mi with i ̸= 0, the regret is lower bounded by:

Eµi  X k∈Ki max x∈Dk ⟨µi, x⟩ − ⟨µi, xk⟩ ≥ I(Ki ≥ 1.5d 2 i) · di pKiσ 2(i) 8 √6 ≥ di pKiσ 2(i) 8 √6− di p1.5d 2 i · σ 2(i) 8 √6 ≥ di qPk∈Ki σ 2 k 16√6− d 2 i· σ(i) 16, (4.1) 240 where the first inequality follows from Lemma 4.3, the second inequality holds due to I(x ≥
y)
√x ≥
√x −
√y, and the last inequality follows from the definition of group Ki 241 .

242 Taking a summation of (4.1) over all groups, the total regret can be lower bounded as follows:

Eµ[Regret(K)] = L X −1 i=0 Eµi  X k∈Ki max x∈Dk ⟨µi, x⟩ − ⟨µi, xk⟩  i=1 di qPk∈Ki σ 2k 16√6− d 2 i · σ(i) 16 ≥ L X −1 i=1 d qPk∈Ki σ 2 k 16√6L− d 2 4L2 ≥ L X −1 ≥ d qPL−1 i=1 Pk∈Ki σ 2 k 16√6L− d 2 4L2
$$(4.2)$$
, (4.2)
243 where the first inequality follows from (4.1), the second inequality follows from the definition of variance threshold σ(i) and dimension di = d/L, and the last inequality holds due to Pi
√
244 p xi ≥
Pi xi 245 . In addition, for the group K0, we have

$\mathcal{K}_{0}$, we have  $$\sum_{k\in\mathcal{K}_{0}}\sigma_{k}^{2}\leq\sum_{k\in\mathcal{K}_{0}}1/K\leq1,$$
246 where the first inequality follows from the definition of group K0 and the second inequality follows 247 from |K0| ≤ K. Therefore, we have

One, we have $$\begin{aligned} \mathbb{E}_\mu[\text{Regret}(K)]&\geq \frac{d\sqrt{\sum_{i=1}^{L-1}\sum_{k\in\mathcal{K}_i}\sigma_k^2}}{16\sqrt{6}L}-\frac{d^2}{4L^2}\\ &\geq \frac{d\sqrt{\sum_{k=1}^K\sigma_k^2-1}}{16\sqrt{6}L}-\frac{d^2}{4L^2}\\ &\geq \frac{d\sqrt{\sum_{k=1}^K\sigma_k^2-1}}{32\sqrt{6}L}, \end{aligned}$$ to follow from (1.2) the second inequality follows. 
$$(4.3)$$
248 where the first inequality follows from (4.2), the second inequality follows from (4.3), and the last inequality follows from the fact that PK
k=1 σ 2k ≥ 1 + 384d 2 249 . Thus, we complete the proof of 250 Theorem 4.1.

## 251 5 **Variance-Dependent Lower Bounds With Adaptive Variance Sequence**

252 In the previous section, we focused on the setting where the variance sequence is prefixed and 253 revealed to the learner at the beginning of the learning process. In this section, we extend our 254 analysis to the setting where the variance sequence can be adaptive based on historical observations, 255 with the learner receiving the adaptive variance at the beginning of each round. 256 5.1 **Main Results**

## 257 5.1.1 **Weak Adversary**

258 We first describe the learning process and the mechanism of variance adaptation. In detail, the 259 adaptive variance process proceeds as follows: 260 1. At the beginning of each round k, a (weak) adversary selects the variance level σk based on 261 the historical observations, including actions {a1*, . . . , a*k−1}, rewards {r1*, . . . , r*k−1}, and 262 decision sets {D1, D2*, . . . ,* Dk−1}. The adversary has access to all historical information 263 but not to the underlying reward model parameters; 264 2. Given the selected variance level σk, we construct and assign a decision set Dk to the learner, where the variance of the reward for each action a ∈ Dk is bounded by σ 2 k 265 ;
266 3. The learner observes the decision set Dk and variance level σk, then determines an action 267 ak from Dk based on its historical observations and current information. After selecting the action, the learner receives a reward rk with variance bounded by σ 2k 268 . 269 **Remark 5.1.** It is worth noting that our concept of adversary differs from the weak/strong adversary 270 in Jia et al. (2024). Specifically, Jia et al. (2024) considers an adversary that attempts to hinder the learner's learning by allocating a fixed total variance budget PK
k=1 σ 2 271 k ≤ Λ across rounds to max272 imize regret. In contrast, our work considers an adversary that attempts to break the lower bounds 273 themselves by collaborating with the learner. To prevent such exploitation, we must restrict the ad274 versary from knowing the weight vector of the underlying reward model. Without this restriction, 275 the adversary could encode each entry µi of the weight vector µ through the corresponding variance σi = µi 276 , allowing the learner to learn the weight vector after d rounds.

277 Under this setting, we establish the following theorem for the variance-dependent lower bound. 278 **Theorem 5.2** (Weak Adversary). For any dimension d > 1, adaptive sequence of variances 279 {σ1*, . . . , σ*K} and algorithm Alg, there exists a hard instance such that each action a ∈ Dk in round k has variance bounded by σ 2k
. For this instance, if PK
k=1 σ 2k ≥ Ω(d 2 280 ), then with probability 281 at least 1 − 1/K, the regret of algorithm Alg over K rounds is lower bounded by:

$$\mathrm{Regret}(K)\geq\Omega{\Big(}d{\sqrt{\sum_{k=1}^{K}\sigma_{k}^{2}}}/\log^{6}(d K){\Big)}.$$

Remark 5.3. Theorem 5.2 provides a high-probability lower bound of Ωed qPK
k=1 σ 2k 282 , which 283 matches the upper bound in Zhao et al. (2023) up to logarithmic factors, albeit with looser logarith284 mic dependencies than Theorem 4.1 due to the adaptive nature of the variance sequence. Unlike 285 the expected lower bound in Theorem 4.1, for adaptive variance sequences, the cumulative variance PK
k=1 σ 2 k 286 depends on the random process and observations. This dependence makes it challenging to 287 establish an expected variance-dependent regret bound - a fundamental difficulty that does not arise for standard d
√
288 K-type lower bounds in linear contextual bandits. To the best of our knowledge, our 289 result provides the first high-probability lower bound for linear contextual bandit.

## 290 5.1.2 **Strong Adversary**

291 In Theorem 5.2, we require that for each round k ∈ [K], all actions x ∈ Dk share the same adaptive 292 variance σk. This is more restrictive than the setting in Zhao et al. (2023), where the variance can 293 differ across actions x ∈ Dk. However, extending our lower bound to action-dependent variances 294 is not possible in the adaptive setting. This limitation arises because we construct the decision 295 set Dk after the adversary chooses the variance σk, which prevents assigning specific variances to 296 individual actions x ∈ Dk. Moreover, we now consider a strong adversary that can choose σk 297 after observing the decision set Dk. The interaction between the learner and this strong adversary 298 proceeds as follows:
299 1. At the beginning of each round k, we construct and assign a decision set Dk based on 300 historical observations, including actions {a1*, . . . , a*k−1} and rewards {r1*, . . . , r*k−1}; 301 2. Given the decision set Dk in round k, the strong adversary selects the variance level σk for 302 round k. The adversary has access to all historical information but not to the underlying 303 reward model parameters; 304 3. The learner observes the decision set Dk and variance level σk, then determines an action 305 ak from Dk based on its historical observations and current information. After selecting the action, the learner receives a reward rk with variance bounded by σ 2k 306 . 307 The following theorem shows that under this setting, the adversary could cooperate with the learner 308 to break the lower bound. 309 **Theorem 5.4** (Strong Adversary). For any linear contextual bandit problem and number of rounds 310 K ≥ 2d, if we first provide the decision set Dk and then allow an adversary to choose the variance 311 σk based on the decision set Dk, there exists one such type of adversary such that, there exists an 312 algorithm whose regret in the first K rounds is upper bounded by Regret(K) ≤ d, where the total variance PK
k=1 σ 2 313 k ≥ K/2.

314 **Remark 5.5.** Theorem 5.4 highlights why Theorem 5.2 requires a weak adversary that set the vari315 ance sequence before seeing the learner's choices. If the adversary could see the decision set first, it 316 could potentially choose variances that would invalidate our lower bound. This finding underscores 317 that our construction is precise and pinpoints the exact condition under which the derived lower 318 bound holds. 319 **Remark 5.6.** It is worth noting that Jia et al. (2024) also considered the case where the adver320 sary assigns variances to actions after observing the decision set and action choice, and provided 321 a variance-dependent lower bound. However, their analysis focuses on an adversary that allocates 322 variance across rounds to maximize the regret. In contrast, our work considers an adversary that 323 attempts to break these bounds, making it more challenging to establish lower bounds for general 324 variance sequences. It is also worth noting that if the adversary's goal is to increase regret, choosing 325 a prefixed sequence is a viable strategy. This case is already covered by our Theorem 4.1 for prefixed 326 sequences, which provides a tighter lower bound than Theorem 5.2. 327 Theorem 5.4 suggests that it is impossible to derive a variance-dependent lower bound if the ad328 versary can determine the variance σk after observing the decision set Dk, which further precludes 329 establishing a lower bound when the adversary has the ability to assign action-dependent variances 330 for each action x ∈ Dk after observing the decision set Dk. This result naturally extends to stochas331 tic linear bandit problems, where the decision set D remains fixed across all rounds. In this case, 332 since the adversary knows the decision set Dk = D in advance, Theorem 5.4 directly implies: 333 **Corollary 5.7.** For any stochastic linear bandit problem with fixed decision set D and number of 334 rounds K ≥ 2d, there exists a prefixed sequence {σ1*, . . . , σ*K} such that there exists an algorithm whose regret in the first K rounds is upper bounded by: RegretAlg 335 (K) ≤ d, where the total variance PK
k=1 σ 2 336 k ≥ K/2.

## 337 5.2 **Proof Sketch Of Theorem 5.2**

338 In this section, we provide the proof sketch of Theorem 5.2. Overall, the proof follows a similar 339 structure as Theorem 4.1, where we divide the rounds into several groups based on their variance 340 magnitude and create hard instances for each group. The key idea is to calculate individual regret 341 bounds for each group and combine them for the final lower bound. However, there exist several 342 challenges when dealing with adaptive variance sequences that require careful handling.

Varying Size of Groups Ki As discussed in Section 4.2, for each group Ki 343 , we create individual instance Mi with fixed variance threshold σ(i) = 2i−1 344 /K and establish a lower bound of Ω( e dipσ 2(i)|Ki 345 |) on the expected regret. However, the construction of such instances relies on prior knowledge of the number of rounds |Ki 346 |, which can be calculated at the beginning for a pre347 fixed variance sequence {σ1*, . . . , σ*K}. In contrast, for general adaptive variance sequences, the number of rounds |Ki 348 | is not known a priori and can even be a random variable, which creates a 349 barrier in constructing these instances.

To address the unknown number of rounds |Ki 350 |, instead of constructing a single instance Mi for 351 each group, we create L instances Mi,j , where L = ⌈log2 K⌉ + 1. Each instance Mi,j is designed for a specific range of round numbers, specifically Mi,j for 2 j−1 *≤ |K*i| < 2 j 352 .

For each round k in group Ki 353 , the learner receives a decision set Di from one of the instances in 354 {Mi,1, . . . ,Mi,L} in a cyclic manner. Through this sequential assignment, the number of visits to each instance Mi,j is |Ki 355 |/L. Consequently, we expect that the instance Mi,j corresponding to the true range 2 j−1 *≤ |K*i| < 2 j provides a lower bound of Ω( e dipσ 2(i)|Ki|) = Ω( e dipσ 2(i) · 2 j 356 ),
which leads to the final lower bound of Ω( e d qPK
k=1 σ 2 k 357 ).

358 **Converting Expected Lower Bound to High-Probability Lower Bound.** Another challenge is establishing the lower bound for the triggered instance Mi,j corresponding to the true range 2 359 j−1 ≤
|Ki| < 2 j 360 . Traditional analysis of lower bounds in linear contextual bandits has focused on the 361 expected regret. However, when dealing with adaptive variance sequences, this approach becomes 362 insufficient as the adversary can dynamically adjust the variance sequence to break these bounds.

For instance, an adversary might continuously set σk = 1 until the lower bound of Ω( e d qPk i=1 σ 2 i 363 )
364 is violated at some round k, then switch to σk = 0 for all future rounds, causing the total variance sum PK
k=1 σ 2k 365 to remain unchanged. In our construction, this means all rounds could fall into group 366 KL, allowing the adversary to adaptively change the number of rounds between different intervals 2 j−1 *≤ |K*L| < 2 j 367 . Since the failure of the lower bound in any single instance ML,j leads to failure 368 of the whole construction, an expected lower bound on regret cannot guarantee robust performance 369 against adaptive sequences. This necessitates a stronger high-probability lower bound that holds 370 uniformly for all instances.

Unfortunately, an expectation of Ω( e di pσ 2(i)2j 371 ) in instance Mi,j only implies a low-probability regret Regret ≥ Ω( e dipσ 2(i)2j )≥ di·2
−j/2 372 , since the cumulative regret in Ki can be up to σ(i)· |Ki 373 | in our instance. To solve this problem, we introduce an auxiliary algorithm that automatically 374 detects the cumulative regret and switches to the standard OFUL algorithm (Abbasi-Yadkori et al.,
2011a) if the cumulative regret is larger than Ω(di pσ 2(i)2j ).

1 375 For this auxiliary algorithm, we can guarantee that the upper bound is at most Ω( e dipσ 2(i)2j 376 ) while maintaining the same probability of high regret as the original algorithm. Therefore, an expectation of Ω( e di pσ 2(i)2j 377 ) in instance Mi,j implies a constant-probability regret PRegret ≥ Ω( e di pσ 2(i)2j )
378 = Ω(1). 379 After constructing an instance with constant-probability lower bound, we boost this probability by creating Ωlog2(dK)
380 independent instances. When the learner encounters instance Mi,j , it is 381 assigned to one of these instances in a cyclic manner. Through this construction, with probability at least 1 − 1/poly(K), the final regret is lower bounded by Regret ≥ Ω( e di pσ 2(i)2j 382 ). 383 **Remark 5.8.** Unlike previous lower bounds for linear bandit problems which focus on expected 384 regret, to the best of our knowledge, our result provides the first high-probability lower bound for 385 linear contextual bandits. It is worth noting that our construction requires separate decision sets 386 across different rounds in the random assignment process. For stochastic linear bandits with a fixed 387 decision set, we can only derive a constant-probability lower bound. Moreover, for a fixed decision 388 set in stochastic linear bandit problem with covering number log N ≤ Oe(d), an algorithm can 389 randomly select one action from the covering set and perform this action in all rounds. In this case, 390 there exists a probability of 1/N = 1/ exp(d) to achieve zero regret, which precludes the possibility 391 of establishing high-probability lower bounds for large round numbers K. More details about the 392 high-probability lower bound can be found in Section 5.2.

## 393 6 **Conclusion And Future Work**

394 In this paper, we study variance-dependent lower bounds for linear contextual bandits in different 395 settings. For both prefixed and adaptive variance sequences with weak adversary, we establish tight 396 lower bounds matching the upper bounds in Zhao et al. (2023) up to logarithmic factors. We further 397 demonstrate a fundamental limitation: when a strong adversary can select variances after observ398 ing decision sets, it becomes impossible to establish meaningful variance-dependent lower bounds. 399 However, our work has focused exclusively on linear bandit settings, while Jia et al. (2024) has 400 established variance-dependent lower bounds for general function approximation with a fixed total 401 variance budget Λ. Therefore, we leave for future work the generalization of our analysis of general 402 variance sequence to contextual bandits with general function approximation.

## 403 **References**

404 ABBASI-YADKORI, Y., PAL´ , D. and SZEPESVARI ´ , C. (2011a). Improved algorithms for linear 405 stochastic bandits. In *Advances in Neural Information Processing Systems*. 406 ABBASI-YADKORI, Y., PAL´ , D. and SZEPESVARI ´ , C. (2011b). Improved algorithms for linear 407 stochastic bandits. In *NIPS*, vol. 11. 408 AUER, P. (2002). Using confidence bounds for exploitation-exploration trade-offs. *Journal of* 409 *Machine Learning Research* 3 397–422. 410 CESA-BIANCHI, N. and LUGOSI, G. (2006). *Prediction, learning, and games*. Cambridge univer411 sity press. 412 CHU, W., LI, L., REYZIN, L. and SCHAPIRE, R. (2011). Contextual bandits with linear payoff 413 functions. In *Proceedings of the Fourteenth International Conference on Artificial Intelligence* 414 *and Statistics*. JMLR Workshop and Conference Proceedings.

415 DAI, Y., WANG, R. and DU, S. S. (2022). Variance-aware sparse linear bandits. *arXiv preprint* 416 *arXiv:2205.13450* . 417 DANI, V., HAYES, T. P. and KAKADE, S. M. (2008). Stochastic linear optimization under bandit 418 feedback . 419 JIA, Z., QIAN, J., RAKHLIN, A. and WEI, C.-Y. (2024). How does variance shape the regret in 420 contextual bandits? *arXiv preprint arXiv:2410.12713* . 421 KIM, Y., YANG, I. and JUN, K.-S. (2021). Improved regret analysis for variance-adaptive linear 422 bandits and horizon-free linear mixture mdps. *arXiv preprint arXiv:2111.03289* . 423 KIM, Y., YANG, I. and JUN, K.-S. (2022). Improved regret analysis for variance-adaptive lin424 ear bandits and horizon-free linear mixture mdps. *Advances in Neural Information Processing* 425 *Systems* 35 1060–1072. 426 KIRSCHNER, J. and KRAUSE, A. (2018). Information directed sampling and bandits with het427 eroscedastic noise. In *Conference On Learning Theory*. PMLR. 428 LI, L., CHU, W., LANGFORD, J. and SCHAPIRE, R. E. (2010). A contextual-bandit approach to 429 personalized news article recommendation. In *Proceedings of the 19th international conference* 430 *on World wide web*. 431 LI, Y., WANG, Y. and ZHOU, Y. (2019). Nearly minimax-optimal regret for linearly parameterized 432 bandits. In *Conference on Learning Theory*. PMLR. 433 ZHANG, Z., YANG, J., JI, X. and DU, S. S. (2021). Improved variance-aware confidence sets for 434 linear bandits and linear mixture mdp. *Advances in Neural Information Processing Systems* 34 435 4342–4355. 436 ZHAO, H., HE, J., ZHOU, D., ZHANG, T. and GU, Q. (2023). Variance-dependent regret bounds 437 for linear bandits and reinforcement learning: Adaptivity and computational efficiency. In The 438 *Thirty Sixth Annual Conference on Learning Theory*. PMLR. 439 ZHAO, H., ZHOU, D., HE, J. and GU, Q. (2022). Bandit learning with general function classes: 440 Heteroscedastic noise and variance-dependent regret bounds. *arXiv preprint arXiv:2202.13603* . 443 ZHOU, D., GU, Q. and SZEPESVARI, C. (2021). Nearly minimax optimal reinforcement learning 444 for linear mixture markov decision processes. In *Conference on Learning Theory*. PMLR. 441 ZHOU, D. and GU, Q. (2022). Computationally efficient horizon-free reinforcement learning for 442 linear mixture mdps. *Advances in neural information processing systems* 35 36337–36349.

## 445 A **Proof Of Theorem 5.2**

446 In this section, we prove the variance-dependent lower bound for adaptive variance sequences es447 tablished in Theorem 5.2. We begin with the instance construction from Lemma 4.3 and establish 448 the following constant-probability lower bound for the regret:
Lemma A.1. For a fixed variance threshold σ, number of rounds K ≥ 1.5d 2 449 , and any bandit algorithm Alg, for the instance constructed in Lemma 4.3, with probability at least Ω1/ log(dK)
450 , 451 the regret is lower bounded by

$$\mathrm{Regret}(K)\geq{\frac{d{\sqrt{K\sigma^{2}}}}{16{\sqrt{6}}}}.$$
.
452 Based on the constant-probability lower bound, we boost this probability by creating L =
Ωlog2(dK)independent instances with dimension d 453 ′ = d/L and number of rounds K′ = K/L,
454 where each instance follows the structure in Lemma 4.3 with i.i.d. sampled weight vectors. Un455 der this construction, the total dimension of all instances is d, which can be represented as a d456 dimensional linear contextual bandit through orthogonal embedding, similar to our previous con457 struction: for instance i, we augment its actions by padding zeros in dimensions reserved for other 458 instances, ensuring actions from different instances only interact with their corresponding param459 eters. Here, we consider the case where the learner visits the instances in a cyclic manner and 460 establish the following high-probability regret lower bound for the constructed instance:
Lemma A.2. For a fixed variance threshold σ, number of rounds K ≥ 1.5d 2 461 , and any bandit algorithm Alg, with probability at least Ω1/ log(dK)
462 , the regret is lower bounded by

$\mathbf{z}^3(dK)$). 
$\downarrow$ . 
$\tau_{\mu\nu}$
$\cdot\cdot\cdot\cdot,L-1$. 

## Regret(K) ≥ Ωd √ Kσ2/ Log3(Dk).

463 With the help of this high-probability lower regret bound from Lemma A.2, we begin the proof 464 of Theorem 5.2. Following a similar framework to the fixed-variance case, we first divide the 465 rounds into groups based on their variance magnitude. Specifically, for any variance sequence 466 {σ1*, . . . , σ*K}, we partition the rounds into L = ⌈log2 K⌉ + 1 groups as follows:

$$\begin{array}{l}{{{\cal K}_{0}=\{k:\sigma_{k}\leq1/K\},}}\\ {{{\cal K}_{i}=\{k:2^{i-1}/K<\sigma_{i}^{2}\}}}\end{array}$$

i/K}, for i = 1*, . . . , L* − 1.

To address the unknown number of rounds Ki = |Ki 467 |, instead of constructing a single instance 468 Mi for each group, we create L instances Mi,j , where L = ⌈log2 K⌉ + 1. Each instance Mi,j is constructed according to Lemma A.2 with dimension d
′ = d/L2, variance σ(i) = 2i−1 469 /K and number of rounds K′ = 2j−1. For each round k in group Ki 470 , the learner receives a decision set Di 471 from one of the instances in {Mi,1, . . . ,Mi,L} in a cyclic manner. 472 *Proof of Theorem 5.2.* According to Lemma A.2, for each instance Mi,j , with probability at least 1 − 1/K3, the regret in the first 2 j−1 473 visits is lower bounded by

$$\mathrm{Regret}(2^{j-1},{\mathcal{M}}_{i,j})\geq\mathbb{I}(2^{j-1}\geq1.5d^{\prime2})\cdot\Omega\big(d^{\prime}\sqrt{2^{j-1}\sigma^{2}(i)}/\log^{3}(d^{\prime}K^{\prime})\big),$$
$$\d(d^{\prime}K^{\prime})),\qquad\quad(\mathrm{A.1})$$

where the indicator reflects the requirement that K′ = 2j−1 ≥ 1.5d
′2 474 . For simplicity, we define E
475 as the event that (A.1) holds for all instances Mi,j . By union bound, we have P(E) ≥ 1 − 1/K.

Conditioned on event E, for an adaptive sequence and each corresponding group Ki 476 , due to the cyclic visiting pattern, each instance Mi,j is visited |Ki 477 |/L times. There exists an instance Mi,j with matching interval for the round number, i.e., 2 j−1 *≤ |K*i|/L ≤ 2 j 478 . Therefore, we have

X
k∈Ki
$$\overrightarrow{\mathbf{\nabla}}\operatorname*{max}_{\mathbf{x}\in{\mathcal{D}}_{k}}\langle{\boldsymbol{\mu}}_{i},\mathbf{x}\rangle-\langle{\boldsymbol{\mu}}_{i},\mathbf{x}_{k}\rangle$$
$$\begin{array}{l}{{\geq\mathrm{Regret}(2^{j-1},\mathcal{M}_{i,j})}}\\ {{\geq\Pi(2^{j-1}\geq1.5d^{\prime2})\cdot\Omega\big(d\sqrt{2^{j-1}\sigma^{2}(i)}/\log^{3}(d^{\prime}K^{\prime})\big)}}\\ {{\geq\Pi(K_{i}\geq3d^{\prime2}L)\cdot\Omega\big(d\sqrt{K_{i}\sigma^{2}(i)}/\log^{4}(d K)\big)}}\\ {{\geq\Omega\Big(d^{\prime}\sqrt{K_{i}\sigma^{2}(i)}/\log^{3}(d K)-d^{\prime}\sqrt{3d^{\prime2}L\sigma^{2}(i)}/\log^{4}(d K)\Big)}}\end{array}$$
$$\geq\Omega\bigg{(}d^{\prime}\sqrt{\sum_{k\in{\cal K}_{i}}\sigma_{k}^{2}/\log^{4}(dK)-\sqrt{3L}d^{\prime2}\cdot\sigma(i)/\log^{4}(dK)}\bigg{)},$$ (A.2)
where the first inequality follows from 2 j−1 *≤ |K*i|/L ≤ 2 j 479 , the second inequality holds by the definition of event E, the third inequality follows from 2 j−1 *≤ |K*i|/L ≤ 2 j 480 , the fourth inequality holds due to I(x ≥ y)
√x ≥
√x −
√ 481 y, and the last inequality follows from the definition of group Ki 482 .

483 Taking a summation of (A.2) over all groups, the total regret can be lower bounded as follows:

$$\operatorname{Regret}(K)$$
= L X−1 i=0 X k∈Ki max x∈Dk ⟨µi, x⟩ − ⟨µi, xk⟩ ≥ L X−1 i=1 Ω d ′sX k∈Ki σ 2 k / log4(dK) − √3Ld′2· σ(i)/ log4(dK)  ≥ Ω  LX−1 i=1 d/L2· sX k∈Ki σ 2 k / log4(dK) − 2 √3Ld2/(L 4log4(dK)) ≥ Ω d/L2· vuut L X−1 i=1 X k∈Ki σ 2 k / log4(dK) − 2 √3Ld2/(L 4log4(dK)), (A.3)
484 where the first inequality follows from (A.2), the second inequality follows from the definition of variance threshold σ(i) and dimension d
′ = d/L2, and the last inequality holds due to Pi
√
485 p xi ≥
Pi xi 486 . In addition, for the group K0, we have

$$\sum_{k\in{\mathcal{K}}_{0}}\sigma_{k}^{2}\leq\sum_{k\in{\mathcal{K}}_{0}}1/K\leq1,$$
$$(\mathrm{A.3})$$
$$(\mathbf{A.4})$$

487 where the first inequality follows from the definition of group K0 and the second inequality follows 488 from |K0| ≤ K. Therefore, we have

$$\begin{split}&\geq\Omega\bigg{(}d/L^{2}\cdot\sqrt{\sum_{i=1}^{L-1}\sum_{k\in\mathcal{K}_{i}}\sigma_{k}^{2}/\log^{4}(d K)-2\sqrt{3L}d^{2}/(L^{4}\log^{4}(d K))}\bigg{)}\\ &\geq\Omega\bigg{(}d/L^{2}\cdot\sqrt{\sum_{i=1}^{L-1}\sum_{k\in\mathcal{K}_{i}}\sigma_{k}^{2}-1/\log^{4}(d K)-2\sqrt{3L}d^{2}/(L^{4}\log^{4}(d K))}\bigg{)}\\ &\geq\Omega\bigg{(}d\cdot\sqrt{\sum_{i=1}^{L-1}\sum_{k\in\mathcal{K}_{i}}\sigma_{k}^{2}/\log^{6}(d K)}\bigg{)},\end{split}$$  Both the results follow from (1.2) the second inequality follows from (1.4).  
489 where the first inequality follows from (A.3), the second inequality follows from (A.4), and the last inequality follows from the fact that PK
k=1 σ 2k ≥ Ω(d 2 490 ). Thus, we complete the proof of Theorem 491 5.2.

## 492 B **Proof Of Theorem 5.4**

493 In this subsection, we provide the proof of Theorem 5.4. We begin by describing a simple algorithm: 494 1. The learner maintains an explored action set A, which is initialized as empty.

495 2. For each decision set Dk in round k, if there exists an action xk not in the spanning space 496 of the explored action set A, the learner:
497 - Selects an action xk and receives reward rk; 498 - Updates the explored set: A = A ∪ {(xk, rk)}.

499 3. Otherwise, when all actions lie in the spanning space of A, the learner:
500 - Estimates the reward for each action through linear combinations of (x, r) ∈ A;
501 - Selects the action with maximum estimated reward.

502 It is worth noting that this algorithm assumes the received rewards rk have no noise to provide 503 accurate estimates in step 3. While this assumption does not hold in general, when an adversary can 504 choose the variance σk based on the decision set Dk, they can cooperate with the learner by setting: 505 - σk = 0 when step 2 is triggered (exploration); 506 - σk = 1 when step 3 is triggered (exploitation).

507 For a d-dimensional linear bandit problem, the explored action set satisfies *|A| ≤* d. This implies 508 the learner performs at most d exploration steps with zero variance, while all remaining steps have 509 variance one. Under this construction, the regret in the first K rounds is upper bounded by:

$\epsilon$) $\leq d$, . 
$\begin{array}{l}–\;1\\ =\;1\end{array}$ 2. 
RegretAlg(K) ≤ d, where the total variance PK
k=1 σ 2 510 k = K −d ≥ K/2 (since K ≥ 2d). Thus, through this cooperation between the adversary and learner, the Ω( e d qPK
k=1 σ 2k 511 ) lower bound is broken, completing the 512 proof of Theorem 5.4.

## 513 C **Proof Of Key Lemmas** 514 C.1 **Proof Of Lemma 4.3**

515 In this subsection, we provide the proof of Lemma 4.3. When the variance threshold σ = 1, our 516 construction reduces to the standard lower bound instances for linear contextual bandits (Zhou et al.,
2021). Specifically, when the number of rounds K satisfying K ≥ 1.5 · d 2 517 , Zhou et al. (2021) 518 provided the following variance-independent lower bound for these hard instances: 519 **Lemma C.1** (Lemma C.8, Zhou et al. 2021). For any bandit algorithm Alg, if the weight vector µ *∈ {−*∆, ∆}
dis drawn uniformly at random from {−∆, ∆}
d 520 , then the expected regret over K
521 rounds is lower bounded by:

$$\mathbb{E}_{\mu}[\mathrm{Regret}(K)]\geq{\frac{d{\sqrt{K}}}{8{\sqrt{6}}}}.$$
.
522 With the help of Lemma C.1, we start the proof of Lemma 4.3. 523 *Proof of Lemma 4.3.* For any algorithm Alg for linear contextual bandit with fixed variance thresh524 old σ, we construct an auxiliary algorithm Alg1 to solve the standard linear contextual bandit prob525 lem:
526 - At the beginning of each round k ∈ K, Alg1 observes the decision set Dk and sends it to 527 Alg; 528 - Alg selects action ak ∈ Dk based on the historical observations and delivers it to Alg1; 529 - Alg1 performs the action ak, receives the reward rk and sends the normalized reward σ · rk 530 to Alg. 531 Now, we consider the performance of auxiliary algorithm Alg1 for the standard linear contextual 532 bandit problem. It is worth noticing that the reward/noise in bandit instances for algorithm Alg1 and 533 algorithm Alg only differ by a scalar factor σ, therefore for each instance, we have E[RegretAlg(K)] = σ · E[RegretAlg1(K)]. (C.1)
If we randomly select a weight parameter vector µ *∈ {−*∆, ∆}
d 534 , then according to Lemma C.1, the 535 regret for Alg is lower bounded by

$$\mathbb{E}[\mathrm{Regret}_{\mathrm{Alg}}(K)]=\sigma\cdot\mathbb{E}[\mathrm{Regret}_{\mathrm{Alg}1}(K)].$$
$$\mathbb{E}_{\mu}[\mathrm{Regret}_{\mathrm{Alg}}(K)]=\sigma\cdot\mathbb{E}_{\mu}[\mathrm{Regret}_{\mathrm{Alg1}}(K)]\geq\sigma\cdot{\frac{d{\sqrt{K}}}{8{\sqrt{6}}}}={\frac{d{\sqrt{K\sigma^{2}}}}{8{\sqrt{6}}}},$$

536 where the equation holds due to (C.1) and the inequality holds due to Lemma C.1. Thus, we com537 plete the proof of Lemma 4.3.

$$(\mathbb{C}.1)$$
$$\operatorname{et}_{\mathrm{Alg1}}(K)].$$

## 538 C.2 **Proof Of Lemma A.1**

539 In this subsection, we provide the proof of Lemma A.1. We begin by recalling the OFUL algorithm 540 in Abbasi-Yadkori et al. (2011a) and its corresponding upper bound for the regret: 541 **Lemma C.2** (Theorem 3 in Abbasi-Yadkori et al. 2011a). For any linear contextual bandit problem, 542 with probability at least 1−δ, the regret for OFUL algorithm in the first K rounds is upper bounded by Regret(K) ≤ OedpK log(dK/δ)
543 .

544 It is worth noting that the reward/noise in the instance construction from Lemma 4.3 only differs by 545 a scalar factor σ from the standard bandit. Therefore, as discussed in Section C.1, the regret in these 546 two cases also only differs by a scalar factor σ. This leads to the following corollary: 547 **Corollary C.3.** For the instance construction from Lemma 4.3, there exists a constant C such that 548 with probability at least 1−δ, the regret for OFUL algorithm in the first K rounds is upper bounded by Regret(K) ≤ CdpKσ2 549 log(dK/δ).

550 With the help of Corollary C.3, we can begin the proof of Lemma A.1. 551 *Proof of Lemma A.1.* For any algorithm Alg, we construct an auxiliary algorithm Alg1 as follows:
552 - At the beginning of each round k ∈ [K], Alg1 observes the decision set Dk and sends it to 553 Alg; 554 - Alg selects action ak ∈ Dk based on the historical observations and delivers it to Alg1; 555 - Alg1 performs the action ak and receives the reward rk; 556 - Alg1 calculates the pseudo regret as:

$$\mathrm{Regret}^{\prime}(k)=\sum_{i=1}^{k}{\frac{1}{3}}+{\frac{d}{\sqrt{96K}}}-r_{k}.$$

If the pseudo regret is larger than d
√Kσ2/(8√6) + σp 557 2K log(2K/δ), Alg1 removes all 558 previous information and performs the OFUL algorithm in all future rounds. 559 Based on the construction of the instances, whatever the weight vector µ is, the optimal action 560 is to select an action in the same direction as the weight vector, obtaining an expected reward of 1/3 + d/√
561 96K. Under this scenario, with probability at least 1 − δ, for any round k ∈ [K], the difference between pseudo regret Regret′
562 (k) and true regret Regret(k) can be upper bounded by

$$\mbox{Regret}(k)-\mbox{Regret}^{\prime}(k)|=|\sum_{i=1}^{k}\epsilon_{i}|\leq\sigma\sqrt{2K\log(2K/\delta)},$$ (C.2)
$$=\,\,$$. 
563 where the inequality holds due to Lemma D.1 with the fact that the noise satisfies 564 E[ϵk|a1:k, r1:k−1] = 0 and |ϵk| ≤ σ. Thus, according to the criterion of auxiliary algorithm 565 Alg1, with probability at least 1 − δ, the regret of Alg1 before transitioning to OFUL is up to d
√Kσ2/(8√6) + 2σp 566 2K log(2K/δ). On the other hand, for the stage after transitioning to 567 OFUL, Corollary C.3 suggests that with probability at least 1 − δ, the regret is no more than CdpKσ2 568 log(*dK/δ*). Therefore, with a selection of δ = 1/K, we have P
-RegretAlg1
(K) ≥ CdpKσ2 log(dK2) + d
√Kσ2/(8√6) + 2σp2K log(2K2)≤ 2/K.

$${\overline{{)}}}]\leq2/K.$$

For simplicity, let R = CdpKσ2 log(dK2) + d
√Kσ2/(8√6) + 2σp2K log(2K2 569 ) and we have Eµ[RegretAlg1
(K)]
≤ P-RegretAlg1
(K) ≥ R· Kσ + P-RegretAlg1
(K) ≥ d
√Kσ2/(16√6)· R
+ P-RegretAlg1
(K) ≥ 0· d
√
Kσ2/(16√6)
≤ 2σ + P-RegretAlg1
(K) ≥ d
√Kσ2/(16√6)· Oe(dpKσ2 log(dK)) + d
√Kσ2/(16√6),
570 where the first inequality holds due to E[X] ≤ P(X ≥ x1) · R + P(X ≥ x2) · x1 + P(X ≥ 0) · x2 571 for 0 ≤ X ≤ R and x1 > x2 > 0, and the second inequality holds due to (C.3). Combining this 572 result with the lower bound of expected regret in Lemma 4.1, we have d
√Kσ2/(8√6) ≥ 2σ + P-RegretAlg1
(K) ≥ d
√Kσ2/(16√6)· Oe(dpKσ2 log(dK))
573 which implies that

$$\mathbb{P}\big{[}\text{Regret}_{\text{Alg}_{1}}(K)\geq d\sqrt{K\sigma^{2}}/(16\sqrt{6})\big{]}\geq\Omega(1/\log(dK)).$$ (C.4)
574 In addition, according to the criterion of auxiliary algorithm Alg1 with (C.2), with probability at 575 least 1 − δ = 1 − 1/K, Alg1 will not switch to the OFUL algorithm until the cumulative regret is larger than d
√Kσ2/(8√
576 6), which implies that

$$\mathbb{P}\big{[}\text{Regret}_{\text{Alg}}(K)\geq d\sqrt{K\,\sigma^{2}}/(16\sqrt{6})\big{]}\geq\mathbb{P}\big{[}\text{Regret}_{\text{Alg}_{1}}(K)\geq d\sqrt{K\,\sigma^{2}}/(16\sqrt{6})\big{]}-1/K$$ $$=\Omega(1/\log(dK)).$$

577 Thus, we complete the proof of Lemma A.1.

## 578 C.3 **Proof Of Lemma A.2**

579 In this subsection, we provide the proof of Lemma A.2. 580 *Proof of Lemma A.2.* Since the learner visits the instances in a cyclic manner, over all K rounds, 581 each instance Mi (i = 1, 2*, . . . , L*) is visited K′ = K/L times. As actions from different instances only interact with their corresponding parameters, according to Lemma A.1, for each instance Mi 582 ,
with probability at least Ω1/ log(dK)
583 , the regret is lower bounded by

$$\begin{array}{l}{\lceil\bot}\end{array}$$
$$\mathrm{Regret}(K^{\prime},{\mathcal{M}}_{i})\geq{\frac{d^{\prime}{\sqrt{K^{\prime}\sigma^{2}}}}{16{\sqrt{6}}}}={\frac{d{\sqrt{K\sigma^{2}}}}{16{\sqrt{6}}\cdot L^{1.5}}}.$$

584 Note that the weight vectors for each instance are independently sampled, hence the probability that at least one instance has regret no less than d
√Kσ2/16√6 · L
1.5 585 is at least

$$1-\left(1-\Omega{\big(}1/\log(d K){\big)}\right)^{L}\geq1-1/K^{3}{\color{red}\mathrm{Oing}}u e:\;???$$

586 Under this condition, the total regret can be lower bounded as:

$$\text{Regret}(K)=\sum_{i=1}^{L}\text{Regret}(K^{\prime},\mathcal{M}_{i})\geq\frac{d\sqrt{K\sigma^{2}}}{16\sqrt{6}\cdot L^{0.5}}.$$ (C.5)
$\square$
587 Thus, we obtain a high-probability lower bound and complete the proof of Lemma A.2.

## 588 D **Auxiliary Lemmas**

Lemma D.1 (Azuma–Hoeffding inequality, Cesa-Bianchi and Lugosi 2006). Let {ηk}
K
k=1 589 be a mar-590 tingale difference sequence with respect to a filtration {Gk} satisfying |ηk| ≤ R for some constant R, ηk is Gk+1-measurable, E-ηk|Gk 591 = 0. Then for any 0 *< δ <* 1, with high probability at least 592 1 − δ, we have

$$\sum_{k=1}^{K}\eta_{k}\leq R{\sqrt{2K\log(1/\delta)}}.$$

## 593 **Neurips Paper Checklist**

595 Question: Do the main claims made in the abstract and introduction accurately reflect the 596 paper's contributions and scope? 597 Answer: [Yes] 598 Justification: In both abstract and introduction, we highlight the contribution in our pa599 per. The proposed algorithm and the corresponding theoretical results are discussed in the 600 followed sections 601 Guidelines: 602 - The answer NA means that the abstract and introduction do not include the claims 603 made in the paper. 604 - The abstract and/or introduction should clearly state the claims made, including the 605 contributions made in the paper and important assumptions and limitations. A No or 606 NA answer to this question will not be perceived well by the reviewers. 607 - The claims made should match theoretical and experimental results, and reflect how 608 much the results can be expected to generalize to other settings. 609 - It is fine to include aspirational goals as motivation as long as it is clear that these 610 goals are not attained by the paper. 611 2. **Limitations** 612 Question: Does the paper discuss the limitations of the work performed by the authors? 613 Answer: [Yes] 614 Justification: We explicitly list all the necessary assumptions for our theoretical analysis. 615 Guidelines: 616 - The answer NA means that the paper has no limitation while the answer No means 617 that the paper has limitations, but those are not discussed in the paper. 618 - The authors are encouraged to create a separate "Limitations" section in their paper. 619 - The paper should point out any strong assumptions and how robust the results are to 620 violations of these assumptions (e.g., independence assumptions, noiseless settings, 621 model well-specification, asymptotic approximations only holding locally). The au622 thors should reflect on how these assumptions might be violated in practice and what 623 the implications would be. 624 - The authors should reflect on the scope of the claims made, e.g., if the approach was 625 only tested on a few datasets or with a few runs. In general, empirical results often 626 depend on implicit assumptions, which should be articulated. 627 - The authors should reflect on the factors that influence the performance of the ap628 proach. For example, a facial recognition algorithm may perform poorly when image 629 resolution is low or images are taken in low lighting. Or a speech-to-text system might 630 not be used reliably to provide closed captions for online lectures because it fails to 631 handle technical jargon. 632 - The authors should discuss the computational efficiency of the proposed algorithms 633 and how they scale with dataset size. 634 - If applicable, the authors should discuss possible limitations of their approach to ad635 dress problems of privacy and fairness. 636 - While the authors might fear that complete honesty about limitations might be used by 637 reviewers as grounds for rejection, a worse outcome might be that reviewers discover 638 limitations that aren't acknowledged in the paper. The authors should use their best 639 judgment and recognize that individual actions in favor of transparency play an impor640 tant role in developing norms that preserve the integrity of the community. Reviewers 641 will be specifically instructed to not penalize honesty concerning limitations. 642 3. **Theory assumptions and proofs**
643 Question: For each theoretical result, does the paper provide the full set of assumptions and 644 a complete (and correct) proof? 594 1. **Claims** 645 Answer: [Yes] 646 Justification: The complete set of assumptions for our analysis is presented in Section 3, 647 with the detailed proofs of all our claims provided in a later section. 648 Guidelines: 649 - The answer NA means that the paper does not include theoretical results. 650 - All the theorems, formulas, and proofs in the paper should be numbered and cross651 referenced. 652 - All assumptions should be clearly stated or referenced in the statement of any theo653 rems. 654 - The proofs can either appear in the main paper or the supplemental material, but if 655 they appear in the supplemental material, the authors are encouraged to provide a 656 short proof sketch to provide intuition. 657 - Inversely, any informal proof provided in the core of the paper should be comple658 mented by formal proofs provided in appendix or supplemental material. 659 - Theorems and Lemmas that the proof relies upon should be properly referenced. 660 4. **Experimental result reproducibility**
661 Question: Does the paper fully disclose all the information needed to reproduce the main 662 experimental results of the paper to the extent that it affects the main claims and/or conclu663 sions of the paper (regardless of whether the code and data are provided or not)? 664 Answer: [NA] 665 Justification: The paper does not include experiments. 666 Guidelines: 667 - The answer NA means that the paper does not include experiments. 668 - If the paper includes experiments, a No answer to this question will not be perceived 669 well by the reviewers: Making the paper reproducible is important, regardless of 670 whether the code and data are provided or not. 671 - If the contribution is a dataset and/or model, the authors should describe the steps 672 taken to make their results reproducible or verifiable. 673 - Depending on the contribution, reproducibility can be accomplished in various ways. 674 For example, if the contribution is a novel architecture, describing the architecture 675 fully might suffice, or if the contribution is a specific model and empirical evaluation, 676 it may be necessary to either make it possible for others to replicate the model with 677 the same dataset, or provide access to the model. In general. releasing code and data 678 is often one good way to accomplish this, but reproducibility can also be provided via 679 detailed instructions for how to replicate the results, access to a hosted model (e.g., in 680 the case of a large language model), releasing of a model checkpoint, or other means 681 that are appropriate to the research performed. 682 - While NeurIPS does not require releasing code, the conference does require all sub683 missions to provide some reasonable avenue for reproducibility, which may depend 684 on the nature of the contribution. For example 685 (a) If the contribution is primarily a new algorithm, the paper should make it clear 686 how to reproduce that algorithm. 687 (b) If the contribution is primarily a new model architecture, the paper should describe 688 the architecture clearly and fully. 689 (c) If the contribution is a new model (e.g., a large language model), then there should 690 either be a way to access this model for reproducing the results or a way to re691 produce the model (e.g., with an open-source dataset or instructions for how to 692 construct the dataset). 693 (d) We recognize that reproducibility may be tricky in some cases, in which case au694 thors are welcome to describe the particular way they provide for reproducibility. 695 In the case of closed-source models, it may be that access to the model is limited in 696 some way (e.g., to registered users), but it should be possible for other researchers 697 to have some path to reproducing or verifying the results. 698 5. **Open access to data and code** 699 Question: Does the paper provide open access to the data and code, with sufficient instruc700 tions to faithfully reproduce the main experimental results, as described in supplemental 701 material? 702 Answer: [NA] 703 Justification: The paper does not include experiments. 704 Guidelines: 705 - The answer NA means that paper does not include experiments requiring code.

706 - Please see the NeurIPS code and data submission guidelines (https://nips.cc/ 707 public/guides/CodeSubmissionPolicy) for more details.

708 - While we encourage the release of code and data, we understand that this might not 709 be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not 710 including code, unless this is central to the contribution (e.g., for a new open-source 711 benchmark). 712 - The instructions should contain the exact command and environment needed to run to 713 reproduce the results. See the NeurIPS code and data submission guidelines (https:
714 //nips.cc/public/guides/CodeSubmissionPolicy) for more details. 715 - The authors should provide instructions on data access and preparation, including how 716 to access the raw data, preprocessed data, intermediate data, and generated data, etc. 717 - The authors should provide scripts to reproduce all experimental results for the new 718 proposed method and baselines. If only a subset of experiments are reproducible, they 719 should state which ones are omitted from the script and why. 720 - At submission time, to preserve anonymity, the authors should release anonymized 721 versions (if applicable). 722 - Providing as much information as possible in supplemental material (appended to the 723 paper) is recommended, but including URLs to data and code is permitted.

## 724 6. **Experimental Setting/Details**

725 Question: Does the paper specify all the training and test details (e.g., data splits, hyper726 parameters, how they were chosen, type of optimizer, etc.) necessary to understand the 727 results? 728 Answer: [NA] 729 Justification: The paper does not include experiments. 730 Guidelines: 731 - The answer NA means that the paper does not include experiments. 732 - The experimental setting should be presented in the core of the paper to a level of 733 detail that is necessary to appreciate the results and make sense of them. 734 - The full details can be provided either with the code, in appendix, or as supplemental 735 material.

## 736 7. **Experiment Statistical Significance**

737 Question: Does the paper report error bars suitably and correctly defined or other appropri738 ate information about the statistical significance of the experiments? 739 Answer: [NA] 740 Justification: The paper does not include experiments. 741 Guidelines: 742 - The answer NA means that the paper does not include experiments.

743 - The authors should answer "Yes" if the results are accompanied by error bars, confi744 dence intervals, or statistical significance tests, at least for the experiments that support 745 the main claims of the paper. 746 - The factors of variability that the error bars are capturing should be clearly stated (for 747 example, train/test split, initialization, random drawing of some parameter, or overall 748 run with given experimental conditions).

749 - The method for calculating the error bars should be explained (closed form formula, 750 call to a library function, bootstrap, etc.) 751 - The assumptions made should be given (e.g., Normally distributed errors). 752 - It should be clear whether the error bar is the standard deviation or the standard error 753 of the mean. 754 - It is OK to report 1-sigma error bars, but one should state it. The authors should prefer755 ably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of 756 Normality of errors is not verified. 757 - For asymmetric distributions, the authors should be careful not to show in tables or 758 figures symmetric error bars that would yield results that are out of range (e.g. negative 759 error rates). 760 - If error bars are reported in tables or plots, The authors should explain in the text how 761 they were calculated and reference the corresponding figures or tables in the text. 762 8. **Experiments compute resources** 763 Question: For each experiment, does the paper provide sufficient information on the com764 puter resources (type of compute workers, memory, time of execution) needed to reproduce 765 the experiments? 766 Answer: [NA] 767 Justification: The paper does not include experiments. 768 Guidelines: 769 - The answer NA means that the paper does not include experiments. 770 - The paper should indicate the type of compute workers CPU or GPU, internal cluster, 771 or cloud provider, including relevant memory and storage. 772 - The paper should provide the amount of compute required for each of the individual 773 experimental runs as well as estimate the total compute. 774 - The paper should disclose whether the full research project required more compute 775 than the experiments reported in the paper (e.g., preliminary or failed experiments 776 that didn't make it into the paper).

777 9. **Code of ethics** 778 Question: Does the research conducted in the paper conform, in every respect, with the 779 NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? 780 Answer: [Yes] 781 Justification: The research conducted in the paper conform, in every respect, with the 782 NeurIPS Code of Ethics. 783 Guidelines: 784 - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. 785 - If the authors answer No, they should explain the special circumstances that require a 786 deviation from the Code of Ethics. 787 - The authors should make sure to preserve anonymity (e.g., if there is a special consid788 eration due to laws or regulations in their jurisdiction). 789 10. **Broader impacts** 790 Question: Does the paper discuss both potential positive societal impacts and negative 791 societal impacts of the work performed?

792 Answer: [NA]
793 Justification: The paper is a theoretical work with no societal impact. 794 Guidelines: 795 - The answer NA means that there is no societal impact of the work performed. 796 - If the authors answer NA or No, they should explain why their work has no societal 797 impact or why the paper does not address societal impact. 798 - Examples of negative societal impacts include potential malicious or unintended uses 799 (e.g., disinformation, generating fake profiles, surveillance), fairness considerations 800 (e.g., deployment of technologies that could make decisions that unfairly impact spe801 cific groups), privacy considerations, and security considerations. 802 - The conference expects that many papers will be foundational research and not tied 803 to particular applications, let alone deployments. However, if there is a direct path to 804 any negative applications, the authors should point it out. For example, it is legitimate 805 to point out that an improvement in the quality of generative models could be used to 806 generate deepfakes for disinformation. On the other hand, it is not needed to point out 807 that a generic algorithm for optimizing neural networks could enable people to train 808 models that generate Deepfakes faster. 809 - The authors should consider possible harms that could arise when the technology is 810 being used as intended and functioning correctly, harms that could arise when the 811 technology is being used as intended but gives incorrect results, and harms following 812 from (intentional or unintentional) misuse of the technology.

813 - If there are negative societal impacts, the authors could also discuss possible mitiga814 tion strategies (e.g., gated release of models, providing defenses in addition to attacks, 815 mechanisms for monitoring misuse, mechanisms to monitor how a system learns from 816 feedback over time, improving the efficiency and accessibility of ML). 817 11. **Safeguards** 818 Question: Does the paper describe safeguards that have been put in place for responsible 819 release of data or models that have a high risk for misuse (e.g., pretrained language models, 820 image generators, or scraped datasets)? 821 Answer: [NA] 822 Justification: The paper is a theoretical work and poses no such risks 823 Guidelines: 824 - The answer NA means that the paper poses no such risks. 825 - Released models that have a high risk for misuse or dual-use should be released with 826 necessary safeguards to allow for controlled use of the model, for example by re827 quiring that users adhere to usage guidelines or restrictions to access the model or 828 implementing safety filters. 829 - Datasets that have been scraped from the Internet could pose safety risks. The authors 830 should describe how they avoided releasing unsafe images. 831 - We recognize that providing effective safeguards is challenging, and many papers do 832 not require this, but we encourage authors to take this into account and make a best 833 faith effort.

## 834 12. **Licenses For Existing Assets**

835 Question: Are the creators or original owners of assets (e.g., code, data, models), used in 836 the paper, properly credited and are the license and terms of use explicitly mentioned and 837 properly respected? 838 Answer: [Yes] 839 Justification: We have described the related works, especially those work which our work 840 is based on with proper citations in corresponding sections. 841 Guidelines: 842 - The answer NA means that the paper does not use existing assets. 843 - The authors should cite the original paper that produced the code package or dataset. 844 - The authors should state which version of the asset is used and, if possible, include a 845 URL. 846 - The name of the license (e.g., CC-BY 4.0) should be included for each asset. 847 - For scraped data from a particular source (e.g., website), the copyright and terms of 848 service of that source should be provided. 849 - If assets are released, the license, copyright information, and terms of use in the 850 package should be provided. For popular datasets, paperswithcode.com/ 851 datasets has curated licenses for some datasets. Their licensing guide can help 852 determine the license of a dataset.

853 - For existing datasets that are re-packaged, both the original license and the license of 854 the derived asset (if it has changed) should be provided.