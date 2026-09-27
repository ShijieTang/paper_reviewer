# Do-Em: Density Operator Expectation Maximization

Anonymous Author(s)
Affiliation Address email

## Abstract

1 Density operators, quantum generalizations of probability distributions, are gain2 ing prominence in machine learning due to their foundational role in quantum 3 computing. Generative modeling based on density operator models (**DOMs**) is an 4 emerging field, but existing training algorithms - such as those for the Quantum 5 Boltzmann Machine - do not scale to real-world data, such as the MNIST dataset. 6 The Expectation-Maximization algorithm has played a fundamental role in enabling 7 scalable training of probabilistic latent variable models on real-world datasets. In 8 *this paper, we develop an Expectation-Maximization framework to learn latent* 9 variable models defined through **DOMs** *on classical hardware, with resources* 10 *comparable to those used for probabilistic models, while scaling to real-world* 11 *data.* However, designing such an algorithm is nontrivial due to the absence of 12 a well-defined quantum analogue to conditional probability, which complicates 13 the Expectation step. To overcome this, we reformulate the Expectation step as a 14 quantum information projection (QIP) problem and show that the Petz Recovery 15 Map provides a solution under sufficient conditions. Using this formulation, we 16 introduce the Density Operator Expectation Maximization (DO-EM) algorithm 17 - an iterative Minorant-Maximization procedure that optimizes a quantum evi18 dence lower bound. We show that the **DO-EM** algorithm ensures non-decreasing 19 log-likelihood across iterations for a broad class of models. Finally, we present 20 Quantum Interleaved Deep Boltzmann Machines (**QiDBMs**), a DOM that can 21 be trained with the same resources as a DBM. When trained with **DO-EM** under 22 Contrastive Divergence, a **QiDBM** outperforms larger classical DBMs in image 23 generation on the MNIST dataset, achieving a 40–60% reduction in the Fréchet 24 Inception Distance.

## 25 **1 Introduction**

26 Recent advances in quantum hardware and hybrid quantum-classical algorithms have fueled a surge of 27 interest in developing learning models that can operate effectively in quantum regimes [1]. Classical 28 models rely on probability distributions; quantum systems generalize these to density operators - 29 positive semi-definite, unit-trace operators on Hilbert spaces—that encode both classical uncertainty 30 and quantum coherence [2]. While there is considerable progress made in quantum supervised 31 learning, there is relatively less progress in unsuperviced learning [3]. 32 Latent variable models (LVMs) are a cornerstone of unsupervised learning, offering a principled 33 approach to modeling complex data distributions through the introduction of unobserved or hidden 34 variables [4]. These models facilitate the discovery of underlying structure in data and serve as the 35 foundation for a wide range of tasks, including generative modeling, clustering, and dimensionality 36 reduction. Classical examples such as Gaussian Mixture Models, Factor Analysis, and Hidden 37 Markov Models [5, 6] exemplify the power of latent variable frameworks in capturing dependencies 38 and variability in observed data. In recent years, LVMs have formed the conceptual backbone of 58 - A Density Operator Expectation-Maximization (**DO-EM**) algorithm is specified using 59 Quantum Information Projection in Algorithm 1. **DO-EM** guarantees log-likelihood ascent 60 in Theorem 4.4 under mild assumptions that retain a rich class of models. 61 - A Quantum Evidence Lower Bound (QELBO) for the log-likelihood is derived in Lemma 4.1 62 from a minorant-maximization perspective leveraging the Monotonicity of Relative Entropy. 63 - **DO-LVMs** are specialized to train on classical data in Section 5 using the **DO-EM** algorithm. 64 This specialization we call **CQ-LVMs**, a class of models with quantum latent variables, can 65 train real world data due to a decomposition proved in Theorem 5.1. 66 - Quantum-interleaved deep Boltzmann machines (QiDBM), a quantum analog of the DBM 67 is defined in Section 5.1. The well known Contrastive Divergence (CD) algorithm for 68 Boltzmann machines is adapted to the QiDBM, which when used with **DO-EM** algorithm in 69 Section 5.1, allows QiDBMs to be trained on MNIST-scale data.

70 - First empirical evidence of a modeling advantage when training **DO-LVMs** on standard 71 computers with real-world data is provided in Section 6. QiDBMs trained using CD on the 72 MNIST dataset achieve a 40–60% lower Fréchet Inception Distance compared to state-of73 the-art deep Boltzmann machines.

## 74 **2 Preliminaries**

Notation The ℓ 2-norm of a column vector v in a Hilbert space H is given by ||v||2 =
√
75 v†v where v
† denotes the conjugate transpose of v. The set of Hermitian (self-adjoint) operators O = O†
76 on 77 H is denoted by L(H). The positive-definite subset of L(H) is denoted by L+(H). The Kronecker 78 product between two operators is denoted A ⊗ B and their direct sum is denoted A ⊕ B [15]. The 79 identity operator on H is denoted IH. The null space of an operator A ∈ H is denoted by ker(A).

80 **Latent variable models and EM algorithm** Latent Variable Models (LVMs) [4] specify the probability distribution of random variables V =[V1*, . . . , V*dV
81 ] through a joint probability model

$$P(V{=}\operatorname{v}\mid\theta)=\sum_{h}P(V{=}\operatorname{v},H{=}\operatorname{h}\mid\theta)$$

where H = [H1*, . . . , H*dL
82 ] are unobserved random variables. Learning an LVM from data, a problem 83 of great interest in Unsupervised Learning [5], refers to estimating the model parameters θ from a dataset D = {v
(1)*, . . . ,* v
(N)
84 } consisting of i.i.d instances drawn from the LVM. Maximum likelihoodbased methods aim to maximize L(θ) = 1N
PN
i=1 ℓi(θ) where ℓi(θ) = log P(V = v(i)
85 | θ). The 86 maximization problem is not only intractable in most cases but even gradient-based algorithms, which 39 deep generative models including Variational Autoencoders [7], Generative Adversarial Networks 40 [8], and Diffusion-based models [9]. The EM algorithm [10, 11] has been instrumental in deriving 41 procedures for learning latent variables models. These algorithms are often preferred over algorithms 42 which directly maximizes likelihood.

43 The study of Density Operator-based Latent Variable Models (**DO-LVM**) remains in its early stages, 44 with foundational questions around expressivity, inference, and learning still largely unexplored 45 [12–14]. Leveraging the modeling power of **DO-LVMs** on real-world data remains a significant 46 challenge. Existing approaches rarely scale beyond 12 visible units—limited by restricted access to 47 quantum hardware, the exponential cost of simulating quantum systems, and the memory bottlenecks 48 associated with representing and optimizing **DO-LVMs** on classical devices. As a result, it is 49 currently infeasible to empirically assess whether **DO-LVMs** offer any practical advantage on real50 world datasets in terms of modeling power. EM based algorithms can provide a simpler alternative 51 to existing learning algorithms for **DO-LVMs** which directly maximizes the likelihood. However 52 deriving such algorithms in Density operator theoretic setup is extremely challenging for a variety of 53 reasons, Most notably there are operator theoretic inequalities, such as Jensen Inequality, which can 54 be directly applied to derive an Evidence lower bound(ELBO) style bound for **DO-LVMs**. Precise 55 characterization of models which are compatible with such bounds and their computational behaviour 56 remains an important area of investigation. In this paper we bridge these research gaps by making the 57 following contributions.

87 can only discover local optima, are difficult to implement because of unwieldy computations in ℓi(θ).

88 The EM algorithm [10, 11] is an alternative iterative algorithm with the scheme

$$\theta^{(k+1)}=\operatorname*{argmax}_{\theta}\frac{1}{N}\sum_{i=1}^{N}Q_{i}(\theta\mid\theta^{(k)}),\text{where}\ell_{i}(\theta)\geq Q_{i}(\theta|\theta^{(k)})\text{~and~}\ell_{i}(\theta^{(k)})=Q_{i}(\theta^{(k)}|\theta^{(k)}).$$

89 **Boltzmann machines** Boltzmann Machines (BM) are stochastic neural networks that define a 90 probability distribution over binary vectors based on the Ising model in statistical physics [16]. Due 91 to the intractability of learning in fully connected BMs, the Restricted Boltzmann Machine (RBM) 92 was introduced with no intra-layer connections, enabling efficient Gibbs sampling [17–19]. Deep 93 Boltzmann Machines (DBM) [20] stacks RBMs uisng undirected connections and allow for joint training of all layers. The joint probability of a DBM with L layers, P(v, h 1*, . . . ,* h L 94 ) is defined as

$$P(\mathbf{v},\mathbf{h}_{1},\ldots,\mathbf{h}_{d_{\mathrm{L}}})={\frac{1}{Z}}e^{-E(\mathbf{v},\mathbf{h}_{1},\ldots,\mathbf{h}_{d_{\mathrm{L}}})}$$
$$({\mathsf{D B M}})$$
)(DBM)
where E(v, h 1*, . . . ,* h L) is called the *Energy Function*, and Z =Pv,h e
−E(v,h 1*,...,*h L)
95 is the Par96 *tition Function* which is typically intractable to compute. Learning in DBMs is difficult due to 97 intractable posterior dependencies. DBMs are usually trained using variants of the Contrastive 98 Divergence algorithm [18, 21, 22]. A detailed discussion on Boltzmann machines and the Contrastive 99 Divergence algorithm is provided in the Appendix A.

## 100 **2.1 Density Operators**

101 A density operator on a Hilbert space H is a Hermitian, positive semi-definite operator with unit trace 102 [2, 23]. The set of Density operators will be denoted by P(H), and can be regarded as generalizations 103 of probability distributions. A joint density operator ρ ∈ P(HA ⊗ HB) can be *marginalized* to ρA ∈ P(HA) by the partial trace operation ρA = TrB(ρ) = PdB
i=1(IA ⊗ x
† i 104 )ρ(IA ⊗ xi) where
{xi}
dB
i=1 105 is an orthonormal basis of HB. Such a ρ is *separable* if it is a convex combination of *product* 106 *states* ρA ⊗ ρB with ρA ∈ P(HA) and ρB ∈ P(HB). 107 **Definition 2.1** (Umegaki [24] Relative Entropy). Let ω and ρ be density operators in P(HA ⊗ HB) 108 with ker(ρ) ⊆ ker(ω). Their relative entropy is given by DU(*ω, ρ*) = Tr(ω log ω) − Tr(ω log ρ).

109 Lindblad [25] showed that the relative entropy does not increase under the action of the parital trace. 110 **Theorem 2.2** (Monotonicity of Relative Entropy). For density operators ω and ρ in P(HA ⊗ HB)
111 *such that* ker(ω) ⊂ ker(ρ), DU(*ω, ρ*) ≥ DU(TrBω, TrBρ).

112 Petz [26, 27] showed that Theorem 2.2 is saturated if and only if the Petz Recovery Map reverses the 113 partial trace operation.

114 **Definition 2.3** (Petz Recovery Map). For a density operator ρ in P(HA ⊗ HB), the Petz Recovery 115 Map for the partial trace Rρ : HA → HA ⊗ HB is the map

$${\mathcal R}_{\rho}(\omega)=\rho^{1/2}\left(\left(\rho_{A}^{-1/2}\omega\rho_{A}^{-1/2}\right)\otimes\mathrm{I}_{B}\right)\rho^{1/2}.$$
$$({\mathsf{P R M}})$$

1/2. (PRM)
116 **Theorem 2.4** (Ruskai's condition). For density operators ω and ρ in P(HA ⊗ HB) *such that* 117 ker(ω) ⊂ ker(ρ), DU(TrBω, TrBρ) = DU(ω, ρ) *if and only if* log ω−log ρ = (TrBω−TrBρ)⊗IB.

118 Ruskai's condition can be interpreted as ω and ρ having the same Conditional Amplitude Operator.

119 **Definition 2.5** (Conditional Amplitude Operator[28]). The conditional amplitude operator of a 120 density operator ρ in P(HA ⊗ HB) with respect to HA is ρB|A = exp(log ρ − log ρA ⊗ IB).

121 A detailed discussion on density operators and quantum channels is provided in Appendix B.

## 122 **3 Density Operator Latent Variable Models**

123 In this section, we introduce Density Operator Latent Variable Models (**DO-LVM**) and recover 124 existing models such as the Quantum Boltzmann Machine (QBM) as special cases. We discuss the 125 computational challenges of learning such models from observations. 126 Definition 3.1 (**DO-LVM** and the Learning Problem). A Density Operator Latent Variable Model 127 (**DO-LVM**) specifies the density operator ρV ∈ P(HV) on observables in HV through a joint density 128 operator ρVL ∈ P(HV ⊗ HL) as ρV = TrL (ρVL(θ)) where the space HL is not observed. Learning 129 a **DO-LVM** is the estimation of model parameters θ when a target density operator ηV ∈ P(HV) is 130 specified. This can be achieved by maximizing the log-likelihood L(θ) = Tr (ηV log ρV(θ)). (LP)
131 Remark 3.2. Maximizing the log-likelihood of a **DO-LVM** is equivalent to minimzing DU(ηV, ρV(θ)). 132 We specialize **DO-LVMs** to classical datasets in Section 5. 133 **Hamiltonian-based models** The Hamiltonian is a Hermitian operator H ∈ L(H) representing 134 the total energy and generalizes the notion of an energy function in classical energy-based models. 135 The model is defined using Gibbs state density matrix analogous to the Boltzmann distribution:
ρ(θ) = exp(H(θ))
Z(θ) with Z(θ)=Tr exp(H(θ)) and H(θ)=Pr 136 θrHr, where Hr ∈ L(H) are Hermitian 137 operators and θr ∈ R are model parameters. The Quantum Boltzmann Machine is a Hamiltonian138 based model inspired by the transverse field Ising model [12]. In this paper, QBMm,n denotes a model 139 with m visible and n hidden units with

dden units with  ${\begin{array}{l}\mathrm{H}(\theta)=-\sum_{i=1}^{m+n}b_i\sigma_i^z-\sum_{i>j}w_{ij}\sigma_i^z\sigma_j^z-\sum_{i=1}^{m+n}\Gamma_i\sigma_i^x\\ +^n\times2^{m+n}\text{Pauli matrices defined by}\sigma_i^k=\otimes_{i=1}^{i-1}\mathrm{I}\otimes\end{array}}$. 
$$\lfloor{\boldsymbol{\theta}}\rfloor\rfloor\,.$$
$$(\mathbb{Q B M})$$

i(QBM)
where σ z iand σ x iare 2 m+n ×2 i = ⊗
j=1I⊗σ k ⊗
m+n j=i+1 140 I where k ∈
{*x, z*}, σ z=1 0 0 −1
, and σ x=( 0 1 1 0 ). A QBM is hence a **DO-LVM** with ρV(θ) = 1 141 Z(θ)TrL exp(H(θ)).

142 Setting Γi = 0 recovers the Boltzmann Machine (BM) [12]. However, the density operator representation of these classical models are plagued by their 2 m+n × 2 m+n 143 dimensionality. The memory 144 requirements for storing and updating models represented by density operators have been prohibitive 145 for QBMs to scale beyond about 12 visible units.

## 162 **4 The Do-Em Framework**

163 In this section, we develop an algorithmic framework applicable for learning **DO-LVMs** using a 164 density operator expectation maximization framework. 165 The classical ELBO is derived for each datapoint using conditional probability and Jensen's inequality. 166 This approach fails for density operators due to the absence well-defined quantum conditional 167 probability [23]. In order to derive an ELBO for **DO-LVMs**, we resort to an approach inspired by the 168 chain rule of KL-divergence [35].

169 **Lemma 4.1** (Quantum ELBO). Let J (ηV) = {η | η ∈ P(HV ⊗ HL) & TrLη = ηV} *be the set of* 170 feasible extensions for a target ηV ∈ P(HV). Then for a **DO-LVM** ρ(θ) and η ∈ J (ηV),
L(θ) ≥ QELBO(η, θ) = Tr(η log ρ(θ)) + S(η) − S(ηV). (QELBO)
146 **Need for an EM algorithm.** As probabilistic LVMs are a special case of **DO-LVMs**, the training 147 challenges they face persist in **DO-LVMs**, which also introduce new operator-theoretic difficulties. 148 Maximizing the log-likelihood of a **DO-LVM** involves operators that do not commute [13]. The 149 direct computation of gradient in Equation (LP) is significantly complicated by the partial trace [29]. 150 Due to the difficulty of working with hidden units, recent work on QBMs have focused on models 151 without hidden units [30, 14, 31, 32]. Demidik et al. [33] studied a Restricted QBM with 12 visible 152 units and 90 hidden units, the largest model studied in literature so far. Refer Appendix B for a 153 detailed survey on QBM literature. Hence, training a QBM, the most popular **DO-LVM** in literature, on 154 real-world data *remains an open challenge*. 155 Intractability of the gradient of the log-likelihood in probabilistic LVMs is addressed by the EM 156 algorithm. Classical derivations of the EM algorithm fail with density operators since there is no 157 well-defined way to construct conditional density operators [23]. An EM algorithm for density 158 operators using Conditional Amplitude Operators (CAO) was conjectured in Warmuth and Kuzmin 159 [34]. This is insufficient since the CAO does not provide a density operator [28]. In the next section, 160 we appeal to well-known results in quantum information theory to derive an ELBO and EM algorithm 161 for density operators. 171 *Proof sketch:* We provide a proof due to Theorem 2.2 in Appendix C. 172 The classical EM algorithm is a consequence of the ELBO being a minorant of the log-likelihood 173 [36, 37]. However, it is well known that Theorem 2.2 is often not saturated [38–42]. Inspired by an 174 information geometric interpretation of the EM algorithm [43], we study an instance of a quantum 175 information projection problem to saturate QELBO.

## 176 **4.1 A Quantum Information Projection Problem**

177 In this subsection we study the I-projection [35] problem for density operators and show conditions 178 when (PRM) can solve this problem. The problem of Quantum Information Projection (QIP) is stated 179 as follows. Consider a density operator ω in P(HA) and a density operator ρ in P(HA ⊗ HB), find ξ
∗
180 in P(HA ⊗ HB) such that ξ
∗ = argmin TrB(ξ)=ω DU(*ξ, ρ*). (QIP)
181 To the best of our knowledge, this problem has not been studied in literature. We know from 182 Theorem 2.2 that the theoretical minimum attained by the objective function in QIP is DU(ω, TrBρ)
183 though it is not always saturated. Inspired by this connection, we explore sufficiency conditions for 184 when PRM solves QIP.

185 Definition 4.2 (**Condition S**). Two density operators ω in P(HA) and ρ in P(HA ⊗ HB) satisfy the 186 sufficiency condition if ρ is full rank, separable, and [ω, TrB(ρ)] = 0.

187 **Theorem 4.3.** Suppose two density operators ω in P(HA) and ρ in P(HA⊗HB) such that *Condition* 188 S *is satisfied, the solution to the information projection problem* QIP is PRM.

189 *Proof sketch:* The statement holds due to the fact that [ρ, Rρ(ω)] = 0 under the conditions in the 190 theorem. Thus, ρ and Rρ(ω) obey Ruskai's condition. A detailed proof is provided Appendix C.

## 191 **4.2 Do-Em Through The Lens Of Minorant Maximization**

192 In this section, we present the Density Operator Expectation Maximization (**DO-EM**) algorithm 193 from a Minorant-Maximization perspective and discuss its advantages over direct maximization of 194 the log-likelihood. We prove that the **DO-EM** algorithm can achieve log-likelihood ascent at every 195 iteration under **Condition S**.

For a fixed θ
(old) 196 , the QELBO is maximized 197 when η is the QIP of ρ(θ) onto the set of fea198 sible extensions. This allows us to define a 199 potential minorant Q for the log-likelihood.

  **Lemma 1**.: _Let $\eta$ be a positive integer. Then $\eta$ is a positive integer._  Proof.: Let $\eta$ be a positive integer. Then $\eta$ is a positive integer.  Proof.: Let $\eta$ be a positive integer. Then $\eta$ is a positive integer.  Proof.: Let $\eta$ be a positive integer. Then $\eta$ is a positive integer.  Proof.: Let $\eta$ be a positive integer. Then $\eta$ is a positive integer.  
Algorithm 1 DO-EM

| 1: Input: Target density operator ηV and θ (0) 2: while not converged do 3: E Step: η (t) = argmin DU(η, ρ(θ (t) )) η:TrLη=ηV (t+1) = argmax (t) log ρ(θ)) 4: M Step: θ Tr(η θ   |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

200 We use Q to define the **DO-EM** algorithm in Algorithm 1. Models and QIPs that obey Ruskai's 201 condition provably achieve log-likelihood ascent under the **DO-EM** procedure.

202 **Theorem 4.4** (Q is a minorant). Let ηV be a target density matrix and ρ(θ) be a **DO-LVM** trained by the **DO-EM** *algorithm. If* ρ(θ
(t)) *and its* QIP *onto the set of feasible extensions,* η
(t)
203 , obey Ruskai's condition, then Q *is a minorant of the log-likelihood. Then,* L(θ
(t+1)) ≥ L(θ
(t)
204 )*, where* θ
(t+1) = argmaxθ Q(θ; θ
(t)
205 ). 206 *Proof sketch:* Proof using the saturation of Theorem 2.2 is in Appendix C. 207 **Corollary 4.5.** For a target density operator ηV and model ρ(θ) satisfying **Condition S***, the E step is* 208 the Petz recovery map Rρ(ηV). Moreover, such a model trained using the **DO-EM** *algorithm achieves* 209 *provable likelihood ascent at every iteration.* 210 *Proof sketch:* The proof due to Theorem 4.3 and Theorem 4.4 is given in Appendix C.

211 The **DO-EM** algorithm can be considered a density operator analog of the classical EM algorithm. 212 We recover the classical EM algorithm from **DO-EM** for discrete models if ηV and ρ(θ) are diagonal.

5 213 The E Step in **DO-EM** finds a feasible extension η whose Conditional Amplitude Operator (CAO) 214 is equal to that of the model ρ(θ). The PRM under **Condition S** is the CAO reweighted by ηV to give 215 a valid density operator. This reduces to classical E step when the CAO reduces to the conditional 216 probability and PRM reduces to Bayes rule. If the model ρ is of the form ρV ⊗ ρL, we recover the 217 conjecture in [34]. 218 A log-likelihood involving a partial trace is often intractable. The M Step in **DO-EM** algorithm 219 maximizes an expression without the partial trace. The log-likelihood of such expressions may have 220 closed-form expressions for the gradients, for example, using the Lee-Trotter-Suzuki formula [14]. 221 In the classical case, this is equivalent to the EM algorithm maximizing a sum of logarithms instead 222 of a logarithm of sums.

Corollary 4.6. *For a Hamiltonian-based model with E step solution* η
(t)
223 *, the M step reduces to* θ
(t+1) = argmaxθ Tr(η
(t)H(θ)) − log Z(θ)
224 *Proof sketch:* The proof due to properties of the matrix logarithm is given in Appendix C.

225 However, the memory footprint of **DO-LVM**s remain, preventing the application of these models 226 on real-world data. We specialize **DO-LVM**s and **DO-EM** to train on classical data and achieve 227 practical scale.

## 228 **5 Do-Em For Classical Data**

229 In this section, we specialize **DO-LVMs** and the **DO-EM** algorithm to classical datasets. We assume, for ease of presentation, that the data D = {v
(1)*, . . . ,* v
(N)
230 } is sampled from the set B =
{+1, −1}
dV .We consider a 2 dV -dimensional Hilbert space HV with standard basis B = {vi}
2 dV
i=1 231 .

232 There is a one-to-one mapping between elements of B and B. For any dataset D, there is an equivalent dataset on HV given by D = {v
(1)*, . . . ,* v
(N)
233 }. The target density operator is then ηV =
1 N
PN
i=1 viv
†
i. A **DO-LVM** on dV-dimensional binary data is therefore a 2 dV+dL × 2 dV+dL 234 matrix while the target ηV is a 2 dV × 2 235 dV matrix.

236 Specializing **Condition S** to diagonal target density operators, allows the decomposition of a DO- 237 LVM into direct sums of smaller subspaces, making the **DO-EM** algorithm computationally easier.

238 **Theorem 5.1.** If ρV is diagonal, ρ is separable if and only if ρ = ⊕iρL(i) and P(vi) = Tr(ρL(i))
with vi ∈ B. The density operator for HL for a particular vi*is then given by* 1 P (vi)
239 ρL(i).

240 *Proof sketch:* See Appendix C. 241 We call models that obey Theorem 5.1 as **CQ-LVMs** since it implies a classical visible probability 242 distribution with a quantum hidden space. QELBO can be specialized to each data point for **CQ-LVMs**.

243 **Lemma 5.2.** For diagonal ηV in P(HV), a **DO-LVM** ρ(θ) satisfies **Condition S** *if and only if it* 244 *is of the form in Theorem 5.1. The log-likelihood of these models can then expressed as* L(θ) =
1 N
PN
i=1 ℓi(θ) *where* ℓi(θ) = log P(v
(i)
245 | θ). 246 *Proof sketch:* The proof is an application of Theorem 5.1 and is given in Appendix C.

247 The decomposition of the log-likelihood into terms for each datapoint, allows the training of models 248 on real-world data since the target densit operator ηV does not have to be initialized. We now show 249 that **CQ-LVMs** are a broad class of models that include several Hamiltonian-based models.

Corollary 5.3. *A Hamiltonian-based model* ρ(θ) = e H(θ)/Z(θ) *with* H(θ) = Pr 250 θrHr is a CQ-
LVMs *if and only if* H = ⊕iHi where Hi are Hermitian operators in L(HL) and i ∈ [2dV 251 ].

252 *Proof sketch:* The proof, due to the properties block diagonal matrices, is given in Appendix C. We 253 now specialize QELBO and Algorithm 1 to **CQ-LVMs**. 254 **Lemma 5.4.** For diagonal ηV in P(HV) and a **CQ-LVM** ρ(θ)*, the log-likelihood of a data point* v
(i)
255 ∈ D, ℓi(θ) *is lower bounded by*

$$(\theta)\geq\mathrm{Tr}$$

ℓi(θ) ≥ Tr ηL log(P(v
(i)|θ)ρ
(i)
L (θ))− Tr(σL log σL)

for any density operator ηL in P(HL) *with equality if and only if* ηL = ρ
(i)
256 L (θ)*. Hence, the* PRM is 257 *given by* Rρ(ηV) = ⊕iPD(V = vi)ρL(i | θ).

258 *Proof sketch:* Application of Lemma 5.4 to Lemma 4.1. Proof is given in Appendix C.

259 This allows us to specialize Algorithm 1 to 260 Algorithm 2, enabling the implementation of 261 **DO-EM** without being restricted by the dimen262 sion of ηV. However, models such as the QBM 263 remain intractable for real-world data due to 264 the normalization term, a problem that exists 265 in classical Boltzmann machines as well.

Algorithm 2 DO-EM for **CQ-LVM**

1: **Input:** Target density operator ηV and θ
(0)
2: **while** not converged do
3: Qi(θ; θ
(k)) = Tr ρ
(i)
L (θ
(k))e
H
(i)(θ)−
$\begin{array}{c}\log Z(\theta)\\ \theta^{(t+1)}=\mbox{argmax}_{\theta}\ \frac{1}{N}\sum_{i=1}^{N}\ \mathcal{Q}_{i}(\theta;\theta)\end{array}$

## (K)) 266 **5.1 Quantum Boltzmann Machine**

267 In this section, we discuss the QBM and define variants which are amenable to implementation on 268 high-dimensional classical data. We first describe QBMs that are **CQ-LVMs**. 269 **Corollary 5.5.** A QBMm,n is a **CQ-LVM** *if and only if quantum terms on the visible units are zero.* 270 *Proof sketch:* The statement is true because of the structure of Pauli matrices which have entries 271 outside the direct sum structure if and only if i ≤ m. A detailed proof can be found in Appendix C.

272 The class of semi-quantum models studied in Demidik et al. [33] are **CQ-LVMs**. Training such a 273 QBM is intractable for real-world data since the free energy term, − log Z(θ) is intractable even for 274 classical Boltzmann machines. To achieve tractable training of QBMs, we introduce the Quantum 275 Interleaved Deep Boltzmann Machine (QiDBM) that can be trained using Contrastive Divergence with 276 a quantum Gibbs sampling step derived here. 277 A Quantum Interleaved Deep Boltzmann Machine (QiDBM) is a DBM with quantum bias terms on 278 **non-contiguous hidden layers**. We describe the Hamiltonian of a three-layered QiDBMℓ,m,n with ℓ 279 visible units and m and n hidden units respectively in the two hidden layers. For ease of presentation, 280 the quantum bias terms are present in the middle layer.

$$\mathrm{H}=-\sum_{i=1}^{\ell+m+n}b_{i}\sigma_{i}^{z}-\sum_{i=1}^{\ell}\sum_{j=1}^{m}w_{i j}^{(1)}\sigma_{i}^{z}\sigma_{\ell+j}^{z}-\sum_{i=1}^{m}\sum_{j=1}^{n}w_{i j}^{(2)}\sigma_{\ell+i}^{z}\sigma_{\ell+m+j}^{z}-\sum_{i=1}^{m}\Gamma_{i}\sigma_{\ell+i}^{x}\sigma_{\ell+j}^{z}\sigma_{\ell+m+j}^{z}\sigma_{\ell+m+j}^{z}\sigma_{\ell+m+j}^{z}\sigma_{\ell+m+j}^{z}\sigma_{\ell+m+j}^{z}\sigma_{\ell+m+j}^{z}$$
$$({\tt Q i D B M})$$

281 The quantum interleaving in a QiDBM is necessary to make the Gibbs sampling step tractable. We 282 illustrate the case of the middle layer of QiDBMℓ,m,n. If the non-quantum visible and hidden layers are fixed to v and h
(2) 283 , the hidden units of the quantum layer are conditionally independent. The Hamiltonian of the i th unit of the quantum layer L
(1) is given by HL
(1) (i|v, h
(2), θ) = −b eff i σ z−Γiσ x 284 . 285 This allows for the tractable sampling from the quantum layer using the expected values

$$\langle\sigma_{i}^{z}\rangle_{\mathbf{v,h}^{(2)}}={\frac{b_{i}^{\mathrm{eff}}}{D_{i}}}\operatorname{tanh}D_{i}{\mathrm{~and~}}\langle\sigma_{i}^{x}\rangle_{\mathbf{v,h}^{(2)}}={\frac{\Gamma_{i}}{D_{i}}}\operatorname{tanh}D_{i}$$

where Di =p(b eff i)
2 + Γ2 iand b eff i = bi +Pℓj=1 w
(1)
ij vj +Pj w
(2)
ij h
(2) j 286 . The Gibbs step for the 287 non-quantum layers is done as per the classical CD algorithm using the quantum sample from the Z 288 Pauli operator. This closed-form expression for Gibbs sampling without matrices allows CD to run 289 on a QiDBM with the same memory footprint as a DBM. See Appendix C for more details.

## 290 **6 Empirical Evaluation**

291 In this work, we propose a quantum model **CQ-LVM**, and a general EM framework, **DO-EM**, to 292 learn them. In this section, we empirically evaluate our methods through experiments to answer 293 the following questions. Details of the compute used to run all our experiments and baselines are 294 provided in Appendix D and E.

295 (Q1) **Effectiveness of DO-EM.** Is Algorithm 2, a feasible algorithm for **CQ-LVM**s compared to 296 state of the art algorithms for QBMs ?

7

Method DOEM Amin Type DBM

QiDBM DBM(L)
Type qidbm dbm 0 5 10 15 20 25 30 Epoch 0.5 1.0 1.5 2.0 2.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 Params 1e6 20 30 40 50 60 0 200 400 600 800 1000 Epoch 50 100 150 200 250 Relative Entro py FID
FID
(a)
(b)
(c)
297 (Q2) **DO-EM on Real World Data.** Does Algorithm 2 scale with the to real world data?

298 (Q3) **Performance of DO-EM.** Does Algorithm 2 provide reasonable improvement in performance 299 over classical LVMs? 300 To answer (Q1), we conduct experiments running exact computation to show that the proposed 301 algorithm is feasible and is practical to implement. 302 **Baselines** We compare our method with our implementation of Amin et al. [12] which explores an 303 alternate algorithm for training QBMs. 304 **Dataset and Metrics** We use a mixture of Bernoulli dataset introduced in Amin et al. [12] described 305 in Appendix D. We measure the efficacy of our proposed method by measuring the average relative 306 entropy during training.

307 **Results of experiment** In Figure 1a, we first observe that the relative entropy of our proposed 308 algorithm does decrease during training, validating our theoretical results and showing, to the best 309 our knowledge, the first instance of an expectation maximization algorithm with quantum bias. We 310 also observe that the performance is competitive with Amin et al. [12]. We also note that **CQ-LVM** 311 training with DO-EM is faster than Amin et al. [12] and consumes lesser memory. We provide more 312 experiments using exact computation in Appendix D. 313 To answer (Q2) and (Q3), we conduct experiments on DBMs of varying sizes with and without 314 the quantum bias term described in Section 5. We present qualitative results of our experiments in 315 Appendix D. 316 **Baselines.** We compare our proposed method with Taniguchi et al. [22], the state of the art for training 317 DBMs. We are unable to reproduce the results in their work and we report the results obtained from their official implementation1 318 using the hyper parameters described in their work. 319 **Datasets and Metrics** Following prior work [22], we perform our experiments on MNIST and 320 Binarized MNIST dataset [44] which contains 60,000 training images and 10,000 testing images of 321 size 28x28. We measure the FID [45] between 10,000 generated images and the MNIST test set 322 to assess the quality of generation. The Fréchet Inception Distance (FID) is a quantitative metric 323 used to evaluate the quality of images generated by generative models by comparing the statistical 324 distribution of their feature representations to those of real images. 325 **Experiment: Performance of DO-EM** To show the superior performance of the proposed method, 326 we compare the FID of our proposed algorithm on Binarized MNIST. We train a QiDBM and DBM 327 with 498, 588, 686, and 784 hidden units with a learning rate of 0.001 for 1000 epochs with 2 hidden 328 layers with SGD optimizer with a batch size of 600.

329 **Results of Experiments** In Figure 1c, we observe that the proposed algorithm outperforms the DBM 330 in all cases, achieving a minimum FID of 14.77 to the DBM's 42.61. This experiment shows that 331 simply adding quantum bias terms to a DBM can *improve the quality* of generations by around 65%. 332 **Experiment: DO-EM on High Dimensional Data** We run CD on 2 DBMs without quantum 333 bias terms according to Taniguchi et al. [22] and CD with quantum bias for a QiDBM on MNIST. 334 Each image corresponds to 6272 visible binary units. The QiDBM has 78.70M parameters with 2 335 hidden layers with quantum bias added to the second layer with a hidden size of 6272. Both DBMs 336 have 2 hidden layers and have 78.69M and 78.71M parameters and hidden sizes of 6272 and 6273 337 respectively. We use a learning rate of 0.001 for all experiments and train with a batch size of 600 338 with SGD optimizer for 1000 epochs. The purpose of this experiment is to show that it is feasible to 339 train large models with quantum bias terms. 340 **Results of Experiments** In Figure 1b, we observe that the proposed method outperforms both 341 classical models of similar size with a 45% reduction in FID. We observe that the FID of the model 342 converges to this value in around 400 epochs whereas both DBM models still exhibit instability after 343 500 epochs. The QiDBM achieves an FID of 62.77 whereas the classical DBMs achieve an FID of 344 111.73 and 99.17 for the smaller and larger model respectively. This experiment indicates that scaling 345 QiDBMs is feasible and provides a significant improvement in performance. In Appendix D, we 346 show the qualitative differences between generated samples of the DBM and QiDBM. We observe 347 that the generated samples from the QiDBM appear to be better than that of the DBM after only 250 348 epochs. 349 **Discussion** We design **CQ-LVM**s and implement Algorithm 1 to learn different target distributions. 350 We first show that Algorithm 1 is effective in learning **CQ-LVM**s and is competitive with the state 351 of the art in terms of reduction of relative entropy at lower running times for 10 qubits and can be 352 extended to even 20 qubits where others cannot. Next, we see that the addition of quantum bias terms 353 to a DBM when trained using Algorithm 2 shows superior generation quality compared to classical 354 DBMs with a 60% reduction of FID on Binarized MNIST. Next, we show that **QiDBMs** can learn 355 high dimensional datasets like MNIST using Algorithm 2 by scaling models upto 6272 hidden units. 356 We observe that QiDBMs also achieve better performance, with 40% lower FID compared to DBMs 357 of similar sizes. We also observe that QiDBMs converge in about half the amount of time compared 358 to DBMs.

## 359 **7 Discussion**

377 The M Step proceeds via gradient descent by the computation of the gradient given by Tr[Hrη(θ
(t))] − Tr[Hrρ(θ)]for the different terms in the Hamiltonian H =Pr 378 θrHr [14, 32].

The M Step stops when the gradients are small and an updated parameter θ
(t+1) 379 is obtained. This two380 step iterative DO-EM procedure continues until convergence. While the gradients can be estimated 381 on existing near-term quantum devices, the E step requires careful design. 382 **Limitations** We discuss the limitations of this work in Appendix F.

360 The paper makes important progress by proposing **DO-EM**, an EM Algorithm for Latent Variable 361 models defined by Density Operators, which provably achieves likelihood ascent. We propose CQ- 362 LVM, a large collection of density operator based models, where **DO-EM** applies. We show that 363 QiDBM, an instance of **CQ-LVM**, can easily scale to MNIST dataset which requires working with 364 6200+ units and outperform DBMs, thus showing that Density Operator models may yield better 365 performance. The specification of **DO-EM** is amenable to implementation on quantum devices. 366 **DO-EM on quantum devices** The E Step of the DO-EM algorithm can be implemented on a quantum 367 computer using the method developed by Gilyén et al. [46], where the quantum channel is performing 368 the partial trace operation. The goal is to prepare the Petz recovery map for the partial trace channel η 369 (t) = Rρ(ηV) using PRM. The requirements for this are (1) Quantum access to the input state ηV (2)
370 efficient state preparation of the model's density matrix ρ(θ) [47, 48] and (3) Block-encodings for the 371 model's density matrix and its marginal ρV(θ) = TrLρ(θ) [49]. Given these input assumptions, the quantum algorithm implementing PRM consists of three steps [46]: (1) applying ρ
−1/2 372 V on the state 373 ηV, (2) applying the adjoint channel which is straight-forward for the partial trace channel and can 374 be operationally achieved by preparing subsystem L in the maximally mixed state, and (3) applying ρ 1/2 on the combined system. Both ρ
−1/2 V and ρ 1/2 375 are implemented using *Quantum Singular Value* 376 *Transformation (QSVT)* techniques, leveraging block-encodings of the relevant states [49].

## 383 **References**

384 [1] J. Preskill. Quantum computing in the nisq era and beyond. *Quantum*, 2:79, 2018. 385 [2] Michael A. Nielsen and Isaac L. Chuang. Quantum Computation and Quantum Infor386 *mation: 10th Anniversary Edition*. Cambridge University Press, 2010. doi: 10.1017/ 387 CBO9780511976667. 388 [3] Yaswitha Gujju, Atsushi Matsuo, and Rudy Raymond. Quantum machine learning on near389 term quantum devices: Current state of supervised and unsupervised techniques for real-world 390 applications. *Phys. Rev. Appl.*, 21:067001, Jun 2024. 391 [4] Christopher M. Bishop. *Pattern Recognition and Machine Learning*. Springer, New York, 2006. 392 ISBN 978-0-387-31073-2. 393 [5] Michael I. Jordan, Zoubin Ghahramani, Tommi S. Jaakkola, and Lawrence K. Saul. An 394 introduction to variational methods for graphical models. *Machine Learning*, 37(2):183–233, 395 1999. doi: 10.1023/A:1007665907178. 396 [6] Zoubin Ghahramani. An introduction to hidden markov models and bayesian networks. Inter397 *national Journal of Pattern Recognition and Artificial Intelligence*, 15(01):9–42, 2001.

398 [7] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. *arXiv preprint* 399 *arXiv:1312.6114*, 2014. 400 [8] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil 401 Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In *Advances in neural* 402 *information processing systems*, volume 27, 2014. 403 [9] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In 404 *Advances in Neural Information Processing Systems*, volume 33, pages 6840–6851, 2020. 405 [10] Leonard E. Baum and Ted Petrie. Statistical Inference for Probabilistic Functions of Finite State 406 Markov Chains. *The Annals of Mathematical Statistics*, 37(6):1554 - 1563, 1966. 407 [11] A. P. Dempster, N. M. Laird, and D. B. Rubin. Maximum likelihood from incomplete data via 408 the em algorithm. *Journal of the Royal Statistical Society: Series B (Methodological)*, 39(1): 409 1–22, 1977. 410 [12] Mohammad H. Amin, Evgeny Andriyash, Jason Rolfe, Bohdan Kulchytskyy, and Roger Melko. 411 Quantum boltzmann machine. *Phys. Rev. X*, 8:021050, May 2018.

412 [13] Mária Kieferová and Nathan Wiebe. Tomography and generative training with quantum 413 boltzmann machines. *Phys. Rev. A*, 96:062327, 12 2017. 414 [14] H J Kappen. Learning quantum models from quantum or classical data. *Journal of Physics A:*
415 *Mathematical and Theoretical*, 53(21):214001, 5 2020.

416 [15] Rajendra Bhatia. *Matrix Analysis*, volume 169. Springer, 1997. ISBN 0387948465. 417 [16] David H. Ackley, Geoffrey E. Hinton, and Terrence J. Sejnowski. A learning algorithm for 418 boltzmann machines. *Cognitive Science*, 9(1):147–169, 1985. ISSN 0364-0213. 422 [18] Geoffrey E Hinton. Training products of experts by minimizing contrastive divergence. *Neural* 423 *Comput*, 14(8):1771–1800, Aug 2002. 424 [19] Miguel Á. Carreira-Perpiñán and Geoffrey Hinton. On contrastive divergence learning. In 425 Robert G. Cowell and Zoubin Ghahramani, editors, *Proceedings of the Tenth International* 426 *Workshop on Artificial Intelligence and Statistics*, volume R5 of *Proceedings of Machine* 427 *Learning Research*, pages 33–40. PMLR, 06–08 Jan 2005. 419 [17] P. Smolensky. Information processing in dynamical systems: Foundations of harmony theory. 420 In *Parallel Distributed Processing, Volume 1: Explorations in the Microstructure of Cognition:* 421 *Foundations*, chapter 6, pages 194–281. The MIT Press, 07 1986. 428 [20] Ruslan Salakhutdinov and Geoffrey Hinton. Deep boltzmann machines. In *Proceedings of the* 429 *12th International Conference on Artificial Intelligence and Statistics (AISTATS)*, volume 5 of 430 *JMLR: W&CP*, pages 448–455, Clearwater Beach, Florida, USA, 2009. JMLR. 431 [21] Ruslan Salakhutdinov and Geoffrey E. Hinton. An efficient learning procedure for deep 432 boltzmann machines. *Neural Computation*, 24(8):1967–2006, 2012. doi: 10.1162/NECO_a_ 433 00302. 434 [22] Shohei Taniguchi, Masahiro Suzuki, Yusuke Iwasawa, and Yutaka Matsuo. End-to-end training 435 of deep boltzmann machines by unbiased contrastive divergence with local mode initialization. 436 In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, 437 and Jonathan Scarlett, editors, *Proceedings of the 40th International Conference on Machine* 438 *Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 33804–33815. 439 PMLR, 23–29 Jul 2023. 440 [23] Mark M. Wilde. *Quantum Information Theory*. Cambridge University Press, nov 2016. 441 [24] H. Umegaki. Conditional expectation in an operator algebra. IV (entropy and information). 442 *Kodai Mathematical Seminar Reports* ¯ , 14:59–85, 1962.

443 [25] Göran Lindblad. Completely positive maps and entropy inequalities. *Communications in* 444 *Mathematical Physics*, 40(2):147–151, Jun 1975. 445 [26] Dénes Petz. Sufficient subalgebras and the relative entropy of states of a von neumann algebra.

446 *Communications in Mathematical Physics*, 105(1):123–131, Mar 1986.

447 [27] Dénes Petz. Sufficiency Of Channels Over von Neumann Algebras. *The Quarterly Journal of* 448 *Mathematics*, 39(1):97–108, 03 1988.

449 [28] N. J. Cerf and C. Adami. Negative entropy and information in quantum mechanics. *Phys. Rev.* 450 *Lett.*, 79:5194–5197, Dec 1997. 451 [29] Nathan Wiebe and Leonard Wossnig. Generative training of quantum boltzmann machines with 452 hidden units. *arXiv preprint arXiv:1905.09902*, 2019. 453 [30] Onno Huijgen, Luuk Coopmans, Peyman Najafi, Marcello Benedetti, and Hilbert J. Kappen. 454 Training quantum boltzmann machines with the β-variational quantum eigensolver. *arXiv* 455 *preprint arXiv:2304.08631*, 2024. 456 [31] Dhrumil Patel and Mark M. Wilde. Natural gradient and parameter estimation for quantum 457 boltzmann machines. *arXiv preprint arXiv:2410.24058*, 2024. 458 [32] Luuk Coopmans and Marcello Benedetti. On the sample complexity of quantum boltzmann 459 machine learning. *Communications Physics*, 7(1):274, 2024. 460 [33] Maria Demidik, Cenk Tüysüz, Nico Piatkowski, Michele Grossi, and Karl Jansen. Expres461 sive equivalence of classical and quantum restricted boltzmann machines. *arXiv preprint* 462 *arXiv:2502.17562*, 2025. 463 [34] Manfred K. Warmuth and Dima Kuzmin. Bayesian generalized probability calculus for density 464 matrices. *Mach. Learn.*, 78(1–2):63–101, January 2010. 465 [35] Thomas M. Cover and Joy A. Thomas. *Elements of Information Theory*. Wiley-Interscience, 466 USA, 2006.

467 [36] David R. Hunter Kenneth Lange and Ilsoon Yang. Optimization transfer using surrogate 468 objective functions. *Journal of Computational and Graphical Statistics*, 9(1):1–20, 2000. doi: 469 10.1080/10618600.2000.10474858.

470 [37] Jan de Leeuw. Block-relaxation algorithms in statistics. In Hans-Hermann Bock, Wolfgang 471 Lenski, and Michael M. Richter, editors, *Information Systems and Data Analysis*, pages 308–
472 324, Berlin, Heidelberg, 1994. Springer Berlin Heidelberg. ISBN 978-3-642-46808-7. 473 [38] Andrew Lesniewski and Mary Beth Ruskai. Monotone riemannian metrics and relative entropy 474 on noncommutative probability spaces. *Journal of Mathematical Physics*, 40(11):5702–5724, 475 11 1999. 476 [39] Mario Berta, Marius Lemm, and Mark M. Wilde. Monotonicity of quantum relative entropy 477 and recoverability. *Quantum Info. Comput.*, 15(15–16):1333–1354, November 2015. 478 [40] Mark M. Wilde. Recoverability in quantum information theory. *Proceedings of the Royal* 479 *Society A: Mathematical, Physical and Engineering Sciences*, 471(2182):20150338, oct 2015. 480 [41] Eric A Carlen and Anna Vershynina. Recovery map stability for the data processing inequality.

481 *Journal of Physics A: Mathematical and Theoretical*, 53(3):035204, jan 2020.

482 [42] Samuel S. Cree and Jonathan Sorce. Geometric conditions for saturating the data processing 483 inequality. *Journal of Physics A: Mathematical and Theoretical*, 55(13):135301, 2022. doi: 484 10.1088/1751-8121/ac5648. 485 [43] Hideitsu Hino, Shotaro Akaho, and Noboru Murata. Geometry of em and related iterative 486 algorithms. *Information Geometry*, 7(1):39–77, 2024. 487 [44] Li Deng. The mnist database of handwritten digit images for machine learning research [best of 488 the web]. *IEEE Signal Processing Magazine*, 29(6):141–142, 2012. 489 [45] Maximilian Seitzer. pytorch-fid: FID Score for PyTorch. https://github.com/mseitzer/pytorch490 fid, August 2020. Version 0.3.0.

491 [46] András Gilyén, Seth Lloyd, Iman Marvian, Yihui Quek, and Mark M. Wilde. Quantum algorithm 492 for petz recovery channels and pretty good measurements. *Phys. Rev. Lett.*, 128:220502, Jun 493 2022. 494 [47] Ersen Bilgin and Sergio Boixo. Preparing thermal states of quantum systems by dimension 495 reduction. *Phys. Rev. Lett.*, 105:170405, Oct 2010. doi: 10.1103/PhysRevLett.105.170405. 496 [48] Chi-Fang Chen, Michael J. Kastoryano, Fernando G. S. L. Brandão, and András Gilyén. 497 Quantum thermal state preparation, 2023. 498 [49] András Gilyén, Yuan Su, Guang Hao Low, and Nathan Wiebe. Quantum singular value 499 transformation and beyond: exponential improvements for quantum matrix arithmetics. In 500 *Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing*, STOC
501 2019, page 193–204, New York, NY, USA, 2019. Association for Computing Machinery. ISBN
502 9781450367059. doi: 10.1145/3313276.3316366.

## 503 **Neurips Paper Checklist**

504 1. **Claims** 505 Question: Do the main claims made in the abstract and introduction accurately reflect the 506 paper's contributions and scope? 507 Answer: [Yes] 508 Justification: All contributions tally with the abstract and introduction. 509 Guidelines: 510 - The answer NA means that the abstract and introduction do not include the claims 511 made in the paper. 512 - The abstract and/or introduction should clearly state the claims made, including the 513 contributions made in the paper and important assumptions and limitations. A No or 514 NA answer to this question will not be perceived well by the reviewers. 515 - The claims made should match theoretical and experimental results, and reflect how 516 much the results can be expected to generalize to other settings. 517 - It is fine to include aspirational goals as motivation as long as it is clear that these goals 518 are not attained by the paper. 519 2. **Limitations** 520 Question: Does the paper discuss the limitations of the work performed by the authors? 521 Answer: [Yes] 522 Justification: Limitations discussed in Appendix F. 523 Guidelines: 524 - The answer NA means that the paper has no limitation while the answer No means that 525 the paper has limitations, but those are not discussed in the paper. 526 - The authors are encouraged to create a separate "Limitations" section in their paper. 527 - The paper should point out any strong assumptions and how robust the results are to 528 violations of these assumptions (e.g., independence assumptions, noiseless settings, 529 model well-specification, asymptotic approximations only holding locally). The authors 530 should reflect on how these assumptions might be violated in practice and what the 531 implications would be. 532 - The authors should reflect on the scope of the claims made, e.g., if the approach was 533 only tested on a few datasets or with a few runs. In general, empirical results often 534 depend on implicit assumptions, which should be articulated. 535 - The authors should reflect on the factors that influence the performance of the approach. 536 For example, a facial recognition algorithm may perform poorly when image resolution 537 is low or images are taken in low lighting. Or a speech-to-text system might not be 538 used reliably to provide closed captions for online lectures because it fails to handle 539 technical jargon. 540 - The authors should discuss the computational efficiency of the proposed algorithms 541 and how they scale with dataset size. 542 - If applicable, the authors should discuss possible limitations of their approach to 543 address problems of privacy and fairness. 544 - While the authors might fear that complete honesty about limitations might be used by 545 reviewers as grounds for rejection, a worse outcome might be that reviewers discover 546 limitations that aren't acknowledged in the paper. The authors should use their best 547 judgment and recognize that individual actions in favor of transparency play an impor548 tant role in developing norms that preserve the integrity of the community. Reviewers 549 will be specifically instructed to not penalize honesty concerning limitations.

## 550 3. **Theory Assumptions And Proofs**

551 Question: For each theoretical result, does the paper provide the full set of assumptions and 552 a complete (and correct) proof? 553 Answer: [Yes] 557 - All the theorems, formulas, and proofs in the paper should be numbered and cross558 referenced. 559 - All assumptions should be clearly stated or referenced in the statement of any theorems. 560 - The proofs can either appear in the main paper or the supplemental material, but if 561 they appear in the supplemental material, the authors are encouraged to provide a short 562 proof sketch to provide intuition. 563 - Inversely, any informal proof provided in the core of the paper should be complemented 564 by formal proofs provided in appendix or supplemental material. 565 - Theorems and Lemmas that the proof relies upon should be properly referenced. 566 4. **Experimental result reproducibility** 567 Question: Does the paper fully disclose all the information needed to reproduce the main ex568 perimental results of the paper to the extent that it affects the main claims and/or conclusions 569 of the paper (regardless of whether the code and data are provided or not)? 570 Answer: [Yes] 571 Justification: Experiments are clearly laid out in Section 6 and Appendix D. Experimental 572 details for reproducibility are provided in Appendix E. Anonymous code is linked. 573 Guidelines: 574 - The answer NA means that the paper does not include experiments.

575 - If the paper includes experiments, a No answer to this question will not be perceived 576 well by the reviewers: Making the paper reproducible is important, regardless of 577 whether the code and data are provided or not. 578 - If the contribution is a dataset and/or model, the authors should describe the steps taken 579 to make their results reproducible or verifiable. 580 - Depending on the contribution, reproducibility can be accomplished in various ways. 581 For example, if the contribution is a novel architecture, describing the architecture fully 582 might suffice, or if the contribution is a specific model and empirical evaluation, it may 583 be necessary to either make it possible for others to replicate the model with the same 584 dataset, or provide access to the model. In general. releasing code and data is often 585 one good way to accomplish this, but reproducibility can also be provided via detailed 586 instructions for how to replicate the results, access to a hosted model (e.g., in the case 587 of a large language model), releasing of a model checkpoint, or other means that are 588 appropriate to the research performed. 589 - While NeurIPS does not require releasing code, the conference does require all submis590 sions to provide some reasonable avenue for reproducibility, which may depend on the 591 nature of the contribution. For example 592 (a) If the contribution is primarily a new algorithm, the paper should make it clear how 593 to reproduce that algorithm. 594 (b) If the contribution is primarily a new model architecture, the paper should describe 595 the architecture clearly and fully. 596 (c) If the contribution is a new model (e.g., a large language model), then there should 597 either be a way to access this model for reproducing the results or a way to reproduce 598 the model (e.g., with an open-source dataset or instructions for how to construct 599 the dataset). 600 (d) We recognize that reproducibility may be tricky in some cases, in which case 601 authors are welcome to describe the particular way they provide for reproducibility. 602 In the case of closed-source models, it may be that access to the model is limited in 603 some way (e.g., to registered users), but it should be possible for other researchers 604 to have some path to reproducing or verifying the results. 605 5. **Open access to data and code** 606 Question: Does the paper provide open access to the data and code, with sufficient instruc607 tions to faithfully reproduce the main experimental results, as described in supplemental 608 material? 554 Justification: Assumptions are stated clearly. Proofs provided in Appendix C. 555 Guidelines: 556 - The answer NA means that the paper does not include theoretical results. 612 - The answer NA means that paper does not include experiments requiring code. 613 - Please see the NeurIPS code and data submission guidelines (https://nips.cc/ 614 public/guides/CodeSubmissionPolicy) for more details. 615 - While we encourage the release of code and data, we understand that this might not be 616 possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not 617 including code, unless this is central to the contribution (e.g., for a new open-source 618 benchmark). 619 - The instructions should contain the exact command and environment needed to run to 620 reproduce the results. See the NeurIPS code and data submission guidelines (https: 621 //nips.cc/public/guides/CodeSubmissionPolicy) for more details. 622 - The authors should provide instructions on data access and preparation, including how 623 to access the raw data, preprocessed data, intermediate data, and generated data, etc. 624 - The authors should provide scripts to reproduce all experimental results for the new 625 proposed method and baselines. If only a subset of experiments are reproducible, they 626 should state which ones are omitted from the script and why. 627 - At submission time, to preserve anonymity, the authors should release anonymized 628 versions (if applicable). 629 - Providing as much information as possible in supplemental material (appended to the 630 paper) is recommended, but including URLs to data and code is permitted. 631 6. **Experimental setting/details** 632 Question: Does the paper specify all the training and test details (e.g., data splits, hyper633 parameters, how they were chosen, type of optimizer, etc.) necessary to understand the 634 results?

635 Answer: [Yes] 636 Justification: Details provided in Appendix E. 637 Guidelines: 638 - The answer NA means that the paper does not include experiments. 639 - The experimental setting should be presented in the core of the paper to a level of detail 640 that is necessary to appreciate the results and make sense of them. 641 - The full details can be provided either with the code, in appendix, or as supplemental 642 material. 643 7. **Experiment statistical significance** 644 Question: Does the paper report error bars suitably and correctly defined or other appropriate 645 information about the statistical significance of the experiments? 646 Answer: [Yes] 647 Justification: See Appendix D. 648 Guidelines:
649 - The answer NA means that the paper does not include experiments.

650 - The authors should answer "Yes" if the results are accompanied by error bars, confi651 dence intervals, or statistical significance tests, at least for the experiments that support 652 the main claims of the paper. 653 - The factors of variability that the error bars are capturing should be clearly stated (for 654 example, train/test split, initialization, random drawing of some parameter, or overall 655 run with given experimental conditions). 656 - The method for calculating the error bars should be explained (closed form formula, 657 call to a library function, bootstrap, etc.) 658 - The assumptions made should be given (e.g., Normally distributed errors).

659 - It should be clear whether the error bar is the standard deviation or the standard error 660 of the mean. 610 Justification: Link to anonymous code provided. Details are provided in Appendix E. 611 Guidelines: 609 Answer: [Yes] 661 - It is OK to report 1-sigma error bars, but one should state it. The authors should 662 preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis 663 of Normality of errors is not verified. 664 - For asymmetric distributions, the authors should be careful not to show in tables or 665 figures symmetric error bars that would yield results that are out of range (e.g. negative 666 error rates). 667 - If error bars are reported in tables or plots, The authors should explain in the text how 668 they were calculated and reference the corresponding figures or tables in the text. 669 8. **Experiments compute resources** 670 Question: For each experiment, does the paper provide sufficient information on the com671 puter resources (type of compute workers, memory, time of execution) needed to reproduce 672 the experiments? 673 Answer: [Yes] 674 Justification: Details provided in Appendix E. 675 Guidelines: 676 - The answer NA means that the paper does not include experiments. 677 - The paper should indicate the type of compute workers CPU or GPU, internal cluster, 678 or cloud provider, including relevant memory and storage. 679 - The paper should provide the amount of compute required for each of the individual 680 experimental runs as well as estimate the total compute. 681 - The paper should disclose whether the full research project required more compute 682 than the experiments reported in the paper (e.g., preliminary or failed experiments that 683 didn't make it into the paper). 684 9. **Code of ethics**
685 Question: Does the research conducted in the paper conform, in every respect, with the 686 NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

687 Answer: [Yes] 688 Justification: Code of ethics followed, no interventions with living beings requiring special 689 processing. Only standard datasets were used. No conflicts of interest. 690 Guidelines: 691 - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. 692 - If the authors answer No, they should explain the special circumstances that require a 693 deviation from the Code of Ethics. 694 - The authors should make sure to preserve anonymity (e.g., if there is a special consid695 eration due to laws or regulations in their jurisdiction). 696 10. **Broader impacts** 697 Question: Does the paper discuss both potential positive societal impacts and negative 698 societal impacts of the work performed? 699 Answer: [NA] 700 Justification: The paper concerns an algorithm to learn density operator latent variable 701 models and does not directly have societal impact. 702 Guidelines: 703 - The answer NA means that there is no societal impact of the work performed. 704 - If the authors answer NA or No, they should explain why their work has no societal 705 impact or why the paper does not address societal impact. 706 - Examples of negative societal impacts include potential malicious or unintended uses 707 (e.g., disinformation, generating fake profiles, surveillance), fairness considerations 708 (e.g., deployment of technologies that could make decisions that unfairly impact specific 709 groups), privacy considerations, and security considerations. 710 - The conference expects that many papers will be foundational research and not tied 711 to particular applications, let alone deployments. However, if there is a direct path to 712 any negative applications, the authors should point it out. For example, it is legitimate 713 to point out that an improvement in the quality of generative models could be used to 714 generate deepfakes for disinformation. On the other hand, it is not needed to point out 715 that a generic algorithm for optimizing neural networks could enable people to train 716 models that generate Deepfakes faster. 717 - The authors should consider possible harms that could arise when the technology is 718 being used as intended and functioning correctly, harms that could arise when the 719 technology is being used as intended but gives incorrect results, and harms following 720 from (intentional or unintentional) misuse of the technology. 721 - If there are negative societal impacts, the authors could also discuss possible mitigation 722 strategies (e.g., gated release of models, providing defenses in addition to attacks, 723 mechanisms for monitoring misuse, mechanisms to monitor how a system learns from 724 feedback over time, improving the efficiency and accessibility of ML). 725 11. **Safeguards** 726 Question: Does the paper describe safeguards that have been put in place for responsible 727 release of data or models that have a high risk for misuse (e.g., pretrained language models, 728 image generators, or scraped datasets)? 729 Answer: [NA] 730 Justification: Paper poses no such risks.

731 Guidelines:
732 - The answer NA means that the paper poses no such risks. 733 - Released models that have a high risk for misuse or dual-use should be released with 734 necessary safeguards to allow for controlled use of the model, for example by requiring 735 that users adhere to usage guidelines or restrictions to access the model or implementing 736 safety filters. 737 - Datasets that have been scraped from the Internet could pose safety risks. The authors 738 should describe how they avoided releasing unsafe images. 739 - We recognize that providing effective safeguards is challenging, and many papers do 740 not require this, but we encourage authors to take this into account and make a best 741 faith effort.

## 742 12. **Licenses For Existing Assets**

743 Question: Are the creators or original owners of assets (e.g., code, data, models), used in 744 the paper, properly credited and are the license and terms of use explicitly mentioned and 745 properly respected? 746 Answer: [Yes] 747 Justification: Sources provided in Appendix E. 748 Guidelines:
749 - The answer NA means that the paper does not use existing assets.

750 - The authors should cite the original paper that produced the code package or dataset. 751 - The authors should state which version of the asset is used and, if possible, include a 752 URL. 753 - The name of the license (e.g., CC-BY 4.0) should be included for each asset. 754 - For scraped data from a particular source (e.g., website), the copyright and terms of 755 service of that source should be provided.

756 - If assets are released, the license, copyright information, and terms of use in the 757 package should be provided. For popular datasets, paperswithcode.com/datasets 758 has curated licenses for some datasets. Their licensing guide can help determine the 759 license of a dataset.

760 - For existing datasets that are re-packaged, both the original license and the license of 761 the derived asset (if it has changed) should be provided. 762 - If this information is not available online, the authors are encouraged to reach out to 763 the asset's creators. 764 13. **New assets** 765 Question: Are new assets introduced in the paper well documented and is the documentation 766 provided alongside the assets? 767 Answer: [Yes] 768 Justification: Anonymous code contains a ReadMe file. 769 Guidelines: 770 - The answer NA means that the paper does not release new assets. 771 - Researchers should communicate the details of the dataset/code/model as part of their 772 submissions via structured templates. This includes details about training, license, 773 limitations, etc. 774 - The paper should discuss whether and how consent was obtained from people whose 775 asset is used. 776 - At submission time, remember to anonymize your assets (if applicable). You can either 777 create an anonymized URL or include an anonymized zip file. 778 14. **Crowdsourcing and research with human subjects** 779 Question: For crowdsourcing experiments and research with human subjects, does the paper 780 include the full text of instructions given to participants and screenshots, if applicable, as 781 well as details about compensation (if any)? 782 Answer: [NA] 783 Justification: No crowdsourcing or research with human subjects. 784 Guidelines: 785 - The answer NA means that the paper does not involve crowdsourcing nor research with 786 human subjects. 787 - Including this information in the supplemental material is fine, but if the main contribu788 tion of the paper involves human subjects, then as much detail as possible should be 789 included in the main paper. 790 - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, 791 or other labor should be paid at least the minimum wage in the country of the data 792 collector.

## 793 15. **Institutional Review Board (Irb) Approvals Or Equivalent For Research With Human**

794 **subjects** 795 Question: Does the paper describe potential risks incurred by study participants, whether 796 such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) 797 approvals (or an equivalent approval/review based on the requirements of your country or 798 institution) were obtained? 799 Answer: [NA] 800 Justification: No crowdsourcing or research with human subjects. 801 Guidelines: 802 - The answer NA means that the paper does not involve crowdsourcing nor research with 803 human subjects. 804 - Depending on the country in which research is conducted, IRB approval (or equivalent) 805 may be required for any human subjects research. If you obtained IRB approval, you 806 should clearly state this in the paper. 807 - We recognize that the procedures for this may vary significantly between institutions 808 and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the 809 guidelines for their institution. 810 - For initial submissions, do not include any information that would break anonymity (if 811 applicable), such as the institution conducting the review.

812 16. **Declaration of LLM usage** 813 Question: Does the paper describe the usage of LLMs if it is an important, original, or 814 non-standard component of the core methods in this research? Note that if the LLM is used 815 only for writing, editing, or formatting purposes and does not impact the core methodology, 816 scientific rigorousness, or originality of the research, declaration is not required. 817 Answer: [NA] 818 Justification: Core method development in this research does not involve LLMs. 819 Guidelines: 820 - The answer NA means that the core method development in this research does not 821 involve LLMs as any important, original, or non-standard components.

822 - Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
823 for what should or should not be described.