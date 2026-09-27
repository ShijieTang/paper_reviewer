# Analog In-Memory Training On General Non-Ideal Resistive Elements: The Impact Of Response Functions

Zhaoxian Wu∗, Quan Xiao∗, Tayfun Gokmen♯, Omobayode Fagbohungbe♯**, Tianyi Chen**†∗
∗Cornell University, New York, NY
♯IBM T. J. Watson Research Center, Yorktown Heights, NY
†Rensselaer Polytechnic Institute, Troy, NY
{zw868, qx232}@cornell.edu,
{tgokmen, Omobayode.Fagbohungbe}@us.ibm.com, tianyi.chen@cornell.edu

## Abstract

As the economic and environmental costs of training and deploying large vision or language models increase dramatically, analog in-memory computing (AIMC) emerges as a promising energy-efficient solution. However, the training perspective, especially its training dynamics, is underexplored. In AIMC hardware, the trainable weights are represented by the conductance of resistive elements and updated using consecutive electrical pulses. While the conductance changes by a constant in response to each pulse, in reality, the change is scaled by asymmetric and non-linear response functions, leading to a non-ideal training dynamics. This paper provides a theoretical foundation for gradient-based training on AIMC hardware with nonideal response functions. We demonstrate that asymmetric response functions negatively impact Analog SGD by imposing an implicit penalty on the objective. To address the issue, we propose residual learning algorithm, which provably converges exactly to a critical point by solving a bilevel optimization problem. We demonstrate that the proposed method can be extended to address other hardware imperfections, such as limited response granularity. As we know, it is the first paper to investigate the impact of a class of generic non-ideal response functions. The conclusion is supported by simulations validating our theoretical insights.

## 1 Introduction

The remarkable success of large vision and language models is underpinned by advances in modern hardware accelerators, such as GPU, TPU [1], NPU [2], and NorthPole chip [3]. However, the computational demands of training these models are staggering. For instance, training LLaMA [4] cost $2.4 million, while training GPT-3 [5] required $4.6 million, highlighting the urgent need for more efficient computing hardware. Current mainstream hardware relies on the Von Neumann architecture, in which the physical separation of memory and processing units creates a bottleneck due to frequent, costly data movement between them. In this context, the industry has turned its attention to analog in-memory computing (AIMC) accelerators based on resistive crossbar arrays [6–10], which excel at accelerating ubiquitous, computationally intensive matrix-vector multiplications (MVMs) operations. In AIMC hardware, the weights (matrices) are represented by the conductance states of the *resistive elements* in analog crossbar arrays
[11, 12], while the input and output of MVM are analog signals like voltage and current. Leveraging
∗The work was done when the authors were at Rensselaer Polytechnic Institute. The work was supported by IBM through the IBM-Rensselaer Future of Computing Research Collaboration, the National Science Foundation Projects 2401297, 2532349, and 2532653, and by the Cisco Research Award.

weight (conductance)
+ -
weight (conductance)
+ -
# of pulse cycles
# of pulse cycles
or w
− after one positive and negative pulse, respectively. The response factors q+(w) and q−(w)
are approximately the slope of the curve at w, and ∆wmin is the response granularity. **(Left)** Ideal response functions q+(w) ≡ q−(w) ≡ 1. Every point is a symmetric point. **(Right)** Asymmetric response functions q+(w) ̸= q−(w) almost everywhere expect for the symmetric point w
⋄.

Kirchhoff's and Ohm's laws, AIMC hardware achieves 10×-10,000× energy efficiency than GPU [13–15] in model inference.

Despite its high efficiency, *analog training* is considerably more challenging than *inference* since it involves frequent weight updates. Unlike digital hardware, where the weight increment can be applied to the original weight in the memory cell, the weights in AIMC hardware are changed by the so-called *pulse update*. Pulse update. When receiving electrical pulses from its peripheral circuits, the resistive elements change their conductance in response to the pulse polarity [16]. Receiving a pulse at each pulse cycle, the conductance is updated by ∆wmin · q+(w) or ∆wmin · q−(w), depending on the pulse polarity, where ∆wmin is *response granularity*, and q+(w) and q−(w) are *response functions*. Geometrically, q+(w) and q−(w) are the slopes of *response curves*; see Figure 1. All ∆wmin, q+(w), and q−(w) are element-specific parameters or functions that are set before training and hence remain fixed during training. Typically, ∆wmin is known while q+(w) and q−(w) are not.

Gradient-based training implemented by analog update. Supported by pulse update, the gradientbased training algorithms are used to optimize the weights. Consider a standard training problem with objective f(·) : R
D → R and a model parameterized by W ∈ R
D

$$W^{*}:=\operatorname*{arg\,min}_{W\in\mathbb{R}^{D}}\ f(W):=\mathbb{E}_{\xi}[f(W;\xi)]$$
$$(\mathbb{I})$$

f(W) := Eξ[f(W; ξ)] (1)
where ξ is a random data sample. Similar to stochastic gradient descent (SGD) in digital training (Digital SGD), the gradient-based training algorithm on AIMC hardware, Analog SGD, updates the weights by stochastic gradients ∇f(Wk; ξk). Digital SGD updates the weight by Wk+1 = Wk − α∇f(Wk; ξk) with learning rate α. Given a *desired update* ∆W = −α∇f(Wk; ξk), AIMC hardware implements Analog SGD by sending |[∆W]d|/∆wmin pulses to the d-th element. Ideally, q+(w) = q−(w) = 1 for every conductance states. If so, with each pulse updating [Wk]d by ∆wmin, [Wk]d is ultimately updated by about [∆W]d.

Challenges of analog training. Despite its ultra-efficiency, gradient-based training on AIMC
hardware is challenging. First, the generic response functions are *asymmetric* (i.e. q+(w) ̸≡ q−(w)), and *non-linear* [17–19]. Due to the variation of response functions and conductance states, gradients are scaled by different magnitudes across different coordinates, leading to biased gradients.

Furthermore, the response granularity ∆wmin is a constant. When the gradients or the learning rate decay below ∆wmin, pulse update no longer provides sufficient precision to perform gradient descent
[20]. Other imperfections include, but are not limited to, noisy input/output (IO) of MVM operations and analog-digital conversion error [18]. This paper aims to investigate the impact of non-ideal response functions and develop a method to mitigate their negative effects. We also discuss extending the proposed method to deal with other hardware imperfections.

## 1.1 Main Results

Complementing existing empirical studies in analog in-memory computing, this paper aims to build a rigorous theoretical foundation of analog training. By introducing bias to the gradient, the asymmetric response function plays a central role in differentiating digital and analog training. In contrast, the other non-idealities hinder the training process by causing precision-related issues. Therefore, we approach the problem progressively, beginning with a simplified case that involves only the asymmetric response functions, and extending the proposed methods to more general scenarios.

As a warm-up, building upon the pulse update mechanism, we propose the following discrete-time mathematical model to characterize the trajectory of Analog SGD
Analog SGD Wk+1 = Wk − α∇f(Wk; ξk) ⊙ F(Wk) − α|∇f(Wk; ξk)| ⊙ G(Wk) (2)
where α > 0 is the learning rate and ξk is the data sample of iteration k; *| · |* and ⊙ represent the element-wise absolute value and multiplication, respectively; and F(·) and G(·) are hardwarespecific matrix which are defined by q+(·) and q−(·). In Section 2, we will explain the underlying rationale of (2). Compared with the standard Digital SGD, the gradients in (2) are scaled by F(·)
and an extra bias term is introduced. Typically, hardware imperfections lead to non-ideal response functions, i.e., F(·) ̸≡ 1 and G(·) ̸≡ 0. Thus, we ask a natural question that Q1) *What is the impact of non-ideal response functions and how to alleviate it?*
Recently, [21] partially answers the question by showing that Analog SGD suffers from a convergence issue due to the asymmetric update, and a heuristic algorithm, Tiki-Taka [22–24], converges exactly by reducing the weight drift. However, their work is limited to a special case of linear response functions, which are in the form of q+(w) = 1 − *w/τ, q*−(w) = 1 + w/τ with hardware-specific parameter τ > 0. Given more general q+(w) and q−(w), the convergence of Tiki-Taka does not trivially hold, even though the response functions are still linear. Gap between theory for special linear and generic response functions. Consider a more generic linear response q+(w) = (1+cLin)(1− w/τ ), q−(w) = (1 − cLin)(1 + w/τ ) with a parameter cLin, which reduces to the setting in [21] when cLin = 0. Figure 2 shows the damage from a non-zero cLin to Tiki-Taka. Consistent with the conclusion in [21], Tiki-Taka significantly outperforms Analog SGD when cLin = 0. However, when cLin is perturbed from 0.1 to 0.3, Tiki-Taka degrades dramatically and even becomes worse than Analog SGD does. The modification is slight, but the convergence guarantee in [21] fails, and the convergence of Tiki-Taka is harmed significantly. This counter-example indicates a gap between the theory for special linear and generic response functions, and necessitates the study of the analog training with generic response functions and the exploration of exact convergence conditions. Ignoring other imperfections temporarily, this paper first analyzes the impact of response functions. We show that Analog SGD suffers from asymptotic error due to the mismatch between the algorithmic stationary point and physical *symmetric point*. Inspired by that, we propose a novel algorithm framework that aligns two points, overcoming the asymmetric issues. Building on that, we endeavor to extend the proposed algorithm to more practical scenarios that involve other imperfections like limited granularity and noisy readings, prompting a second critical question:
0 200 400 600 800 1000 Number of Gradient Computation (k)
10 5 10 4 10 3 10 2 10 1 10 0

```
f(
Wk)cLin = 0.0
                                                            cLin = 0.1
                                                            cLin = 0.2
                                                            cLin = 0.3
                                                            Tiki-Taka
                                                            Analog SGD

```

Figure 2: Comparison of Analog SGD and Tiki-Taka under different parameter cLin. The error plateau in the order 10−5comes from the limited response granularity ∆wmin = 10−4.

Q2) *How to extend the framework to address the limited response granularity and noisy IO issues?*
To answer this question, we propose two mechanisms to further overcome these two issues. Our contributions. This paper makes the following contributions:
C1) Building on the pulse update equation, we propose an approximate discrete-time dynamics for analog update. Enabled by this, we study the impact of response functions directly, without being limited to specific element candidates.

C2) Based on that, we show that instead of optimizing f(·), Analog SGD optimizes another penalized objective implicitly. An implicit penalty is introduced by the asymmetric response functions, which attract the weights towards symmetric points. Consequently, Analog SGD can only converge to the optimal point inexactly.

C3) We propose a novel Residual Learning theoretical framework to alleviate the asymmetric update and implicit penalty issues. Residual Learning explicitly introduces another residual array, which has a stationary point 0. This framework leads to Tiki-Taka heuristically proposed in [22] while it offers an understanding of how Tiki-Taka deals with the challenge from generic response functions. By properly zero-shifting so that the stationary and symmetric points overlap, Residual Learning provably converges to a critical point.

C4) Building on C3), we propose a variant, Residual Learning v2, tailored for more practical training scenarios. We propose introducing a digital buffer to filter out reading errors caused by IO noise. Furthermore, we propose a threshold-based transfer rule to alleviate instability caused by limited granularity.

## 1.2 Prior Art

AIMC training. Analog training has shown promising early successes with tremendous energy advantage [25, 26]. Among them, on-chip training, which performs forward, backward, and update directly on analog chips [22–24, 27, 28] is considered to be the most efficient paradigm, but it is more sensitive to hardware imperfections. Sacrificing energy efficiency for robustness, hybrid digital-analog off-chip training is proposed [29–32], which offloads some computation burden to digital components. This paper focuses on the more challenging on-chip training setting.

Energy-based model and equilibrium propagation. AIMC training leverages back-propagation to compute the gradient signals. Recently, a class of energy-based models has been studied, which performs equilibrium propagation to compute gradient signals [33–37]. Focusing on the training dynamics instead of concrete gradient computing, our work is orthogonal to them and is expected to provide insight for algorithm designs of energy-based model training.

## 2 Analog Training With Generic Response Functions

This section examines the discrete-time dynamics of analog training and introduces the challenges posed by generic response functions. After that, we introduce a family of response functions that reflect crucial physical properties that interest us. Compact formulations of analog update. We first investigate the dynamics of one element w in W ∈ R
D. This paper adopts w to represent the element of the weight Wk without specifying its index. As we discuss in Section 1, the response granularity ∆wmin is scaled by the response functions q+(w) or q−(w). Since a desired update ∆w requires a series of pulses with each scaled by approximately q+(w) or q−(w), it is sensible that the ∆w is approximately scaled by q+(w) or q−(w) as well. Accordingly, we propose that an approximate dynamics of analog update is given by w
′ ≈ Uq(w, ∆w), where Uq(w, ∆w) is defined by

$$U_{q}(w,\Delta w):=\begin{cases}w+\Delta w\cdot q_{+}(w),&\Delta w\geq0,\\ w+\Delta w\cdot q_{-}(w),&\Delta w<0.\end{cases}$$

$$({\mathfrak{I}}{\mathfrak{I}})$$
w + ∆w · q−(w), ∆w < 0.(3)
The update (3) holds at each resistive element. At the k-th iteration, We stack all the weights wk and expected increment ∆wk together into vectors Wk, ∆Wk ∈ R
D. Similarly, the response functions q+(·) and q−(·) are stacked into Q+(·) and Q−(·), respectively. Let the notation UQ(Wk, ∆W) on matrices Wk and ∆W denote the element-wise operation on Wk and ∆W, i.e. [UQ(Wk, ∆W)]d := U[Q]d([Wk]d, [∆w]d), ∀d ∈ [D] with [D] := {1, 2, · · · , D} denoting the index set. The element-wise update (3) can be expressed as Wk+1 = UQ(Wk, ∆Wk). Leveraging the symmetric decomposition [21, 22], we decompose Q−(W) and Q+(W) into symmetric component F(·) and asymmetric component G(·)
F(W) := (Q−(W) + Q+(W))/2, and G(W) := (Q−(W) − Q+(W))/2, (4)
which leads to a compact form of the Analog Update

  **Analog Update** $W_{k+1}=W_{k}+\Delta W_{k}\odot F(W_{k})-|\Delta W_{k}|\odot G(W_{k})$.  
Gradient-based training algorithms on AIMC hardware. In (5), the desired update ∆Wk varies based on different algorithms. Replacing ∆Wk with the stochastic gradient ∇f(Wk; ξk), we obtain

$$W)-Q_{+}(W))/2,$$
$${}^{(4)}$$
$$({\mathfrak{S}})$$

Response function
the dynamics of Analog SGD shown in (2). This update is reduced to the mathematical form for linear response functions in [21] as a special case; see Appendix B for details. Response function class. Before proceeding to the study of response functions, we first define the response function class that interests us. Since the behavior of resistive elements is always governed by physical laws, the function class should reflect certain crucial physical properties.

The most crucial property of the response functions is the *asymmetric update*, i.e., q−(w) ̸= q+(w)
for most of w. Specifically, if a point w
⋄satisfies q−(w
⋄) = q+(w
⋄), we say w
⋄is a symmetric point. Stacking all w
⋄into a vector W⋄ ∈ R
D. Observe that the function G(W) is large if q−(w)
and q+(w) are significantly different, while it is almost zero around W⋄. At the same time, F(W) is the average of the response functions in two directions. As we will see in Sections 3.2 and 4, the ratio
√
G(W)
F (W)
plays a critical role in the convergence behaviors.

In addition to the asymmetric update, the function class should possess other properties. First, the conductance increases upon receipt of a positive pulse, and vice versa, resulting in positive response functions. In addition, we assume that the response functions are differentiable (and hence continuous) for mathematical tractability. Taking all factors into account, we define the following class of response functions.

Definition 1 (Response function class). q+(·) and q−(·) *satisfy*
- **(Positive-definiteness)** *There exist positive constants* qmin > 0 and qmax > 0 *such that* qmin ≤
q+(w) ≤ qmax and qmin ≤ q−(w) ≤ qmax, ∀w*; and,*
- **(Differentiable)** The response functions q+(·) and q−(·) *are differentiable.*
Definition 1 covers a wide range of response functions, including but not limited to PCM, ReRAM, ECRAM, and others mentioned in Section A. Figure 3 showcases three examples from the response functions class, including linear, non-linear but monotonic, and even non-monotonic functions.

## 3 Implicit Penalty And Inexact Convergence Of Analog Sgd

This section introduces a critical impact of the response functions, *implicit penalized objective*. Affected by this, Analog SGD can only converge inexactly with a non-diminishing asymptotic error.

## 3.1 Implicit Penalty

We first give an intuition through a situation where Wk is already a critical point, i.e.,
Eξ[∇f(Wk; ξ)] = 0. Recall that stochastic gradient descent on digital hardware (Digital SGD) is stable in expectation, i.e. Eξk
[Wk+1] = Wk − Eξk
[α∇f(Wk; ξk)] = Wk. However, this does not work for Analog SGD
Eξk
[Wk+1] = Wk − Eξk
[α∇f(Wk; ξk) ⊙ F(Wk) − α|∇f(Wk; ξk)| ⊙ G(Wk)] (6)
= Wk − αEξk
[|∇f(Wk; ξk)| ⊙ G(Wk)] ̸= Wk.

Consider a simplified version that the weight is a scalar (D = 1) and the function G(W) is strictly monotonically decreasing2to help us gain intuition on the impact of the drift in (6). Recall G(W⋄) = 0 at the symmetric point W⋄. G(W) > 0 when *W > W*⋄and G(W) < 0 otherwise. Consequently,
(6) indicates that Eξk[Wk+1] < Wk when Wk > W⋄and Eξk[Wk+1] > Wk otherwise. It implies that Wk suffers from a drift tendency towards W⋄. In addition, the penalty coefficient proportional 2It happens when both q+(·) and q−(·) are strictly monotonic.

$$\mathbf{U}$$
$\downarrow$ . 
to the noise level since the drift is proportional to Eξk
[|∇f(Wk; ξk)|], which is the first moment of noise Eξk[|∇f(Wk; ξk) − Eξ[∇f(Wk; ξ)]|] in essence.

The following theorem formalizes the implicit penalty effect. Before that, we define an accumulated asymmetric function Rc(·) : R
D → R
D, whose derivative is R(W) := G(W)
F (W)
, i.e. d[Rc(W)]d d[W]d=
[R(W)]d =
[G(W)]d
[F (W)]d
. If R(W) is strictly monotonic, Rc(W) reaches its minimum at the symmetric point W⋄ where R(W⋄) = 0, so that it penalizes the weight away from the symmetric point. Theorem 1 (Implicit penalty, short version). Suppose W∗is the unique minimizer of problem (1)*. Let* Σ := Eξ[|∇f(W∗; ξ)|] ∈ R
D. Analog SGD *implicitly optimizes the following penalized objective*

$$\operatorname*{min}_{W}\;f_{\Sigma}(W):=f(W)+\langle\Sigma,R_{c}(W)\rangle\;.$$
$$(T)$$
W
fΣ(W) := f(W) + ⟨Σ, Rc(W)⟩. (7)
The full version of Theorem 1 and its proof are deferred to Appendix G. In Theorem 1, Rc(W)
plays the role of a penalty to force the weight towards a symmetric point. As shown in Appendix G, Rc(W) has a simple expression on linear response functions when cLin = 0, leading (7) to minW fΣ(W) := f(W) + Σ
2τ
∥W∥
2 which is an ℓ2 regularized objective. In addition, the implicit penalty has a coefficient proportional to the noise level Σ and inversely proportional to the dynamic range τ . It implies that the implicit penalty becomes active only when gradients are noisy, and the noise amplifies the effect.

With noisy gradients, an **implicit penalty** attracts Analog SGD towards symmetric points.

## 3.2 Inexact Convergence Of Analog Sgd Under Generic Devices

Due to the implicit penalty, Analog SGD only converges to a critical point inexactly. Before showing that, We introduce a series of assumptions on the objective, as well as noise. Assumption 1 (Objective). The objective f(W) is L*-smooth and is lower bounded by* f
∗.

Assumption 2 (Unbiasness and bounded variance). The stochastic gradient is unbiased and has bounded variance σ 2*. i.e.,* Eξ[∇f(W; ξ)] = ∇f(W) and Eξ[∥∇f(W; ξ) − ∇f(W)∥
2] ≤ σ 2.

Assumption 1–2 are standard in non-convex optimization [38]. This paper considers the average
squared norm of the gradient as the convergence metric, given by EASGD
K := 1K
PK−1
k=0 ∥∇f(Wk)∥
2.
Now, we establish the convergence of Analog SGD.
Theorem 2 (Inexact convergence of Analog SGD). Under Assumption *1–2, if the learning rate is set* as α = O(1/
√K)*, it holds that*
$$E_{K}^{A S a b}\leq O\left(\sqrt{\sigma^{2}/K}+\sigma^{2}S_{K}^{A S a b}\right)$$
K(8)
where S
ASGD
K *denotes the amplification factor given by* S
ASGD
K := 1K
PK−1 k=0

√
G(Wk)
F (Wk)

$$\frac{\left|\vec{V}_{k}\right\rangle}{\left|\vec{V}_{k}\right\rangle}\left|\right|^{2}$$
$$({\mathfrak{s}})$$
$$\mathrm{K}-1\ \big|\big|$$

$\alpha$ $\epsilon$. 
∞
.

The proof of Theorem 2 is deferred to Appendix H. Theorem 2 suggests that the convergence metric EASGD
K is upper bounded by two terms: the first term vanishes at a rate of O(pσ 2/K), which matches the Digital SGD's convergence rate [38] up to a constant; the second term contributes to the *asymptotic error* of Analog SGD, which does not vanish with the number of iterations K.

Impact of saturation/asymmetric update. The exact expression of S
ASGD
K depends on the specific noise distribution and thus is difficult to reach. However, S
ASGD
K reflects the saturation degree near the critical point W∗ when Wk converges to a neighborhood of W∗. If W∗is far from the symmetric point W⋄, S
ASGD
K becomes large, leading to a large EASGD
K and a large asymptotic error. In contrast, if W∗remains close to the symmetric point W⋄, the asymptotic error is small.

## 4 Mitigating Implicit Penalty By Residual Learning

The asymptotic error in Analog SGD is a fundamental issue that arises from the mismatch between the symmetric point and the critical point. An idealistic remedy for the inexact convergence is carefully shifting the weights to ensure the stationary point is close to a symmetric point. However, determining the appropriate shifting is challenging, as the critical point is unknown before training. Therefore, an ideal solution to address this issue is to jointly construct a sequence with a proper stationary point and a proper shift of the symmetric point.

Residual learning. Our solution overlaps the algorithmic stationary point and physical symmetric point on the special point 0. Besides the main analog array, Wk, we maintain another array, Pk, whose stationary point should be 0. A natural choice is the *residual* of the weight, P
∗(W), defined by the P that minimizes the objective f(W + γP) with a non-zero γ. Notice that P
∗(Wk) → 0 as Wk → W∗. Additionally, the goal of the main array is to minimize the residual so that the model Wk approaches optimality. This process can be formulated as the following bilevel problem, whose optimal points can be proved to be those of f(W)

Residual Learning $\min_{W\in\mathbb{R}^{D}}\|P^{*}(W)\|^{2},\quad\text{s.t.}P^{*}(W)\in\operatorname*{arg\,min}_{P\in\mathbb{R}^{D}}f(W+\gamma P)$. (9)
Now we propose a gradient-based method to solve (9). The stochastic gradient of f(W + γP) with respect to P, given by ∇P f(W + γP; ξ) = γ∇f(W + γP; ξ), is accessible with fair expense, enabling us to introduce a sequence Pk to track the residual of Wk by optimizing f(Wk + γP)

$$(W_{k};\xi_{k})\odot F(P_{k})-\alpha|\nabla f$$

Pk+1 = Pk − α∇f(W¯k; ξk) ⊙ F(Pk) − α|∇f(W¯k; ξk)| ⊙ G(Pk). (10)
where W¯k := Wk + γPk is the mixed weight. We then derive the hyper-gradient of the upper-level objective. Notice ∇∥P
∗(W)∥
2 = 2∇P
∗(W)P
∗(W). Assuming W∗is the unique minimum of f(·), we know P
∗(W) satisfies γP∗(W)+W = W∗. Taking gradient with respective to W on both sides, we have ∇P
∗(W) = −
1 γ I and hence ∇∥P
∗(W)∥
2 = −
2 γ P
∗(W). Approximating P
∗(W)
by Pk and absorbing 2γ into the learning rate β, we reach the update of the main array

$$\circ G(P_{k}).$$
$W_{k+1}=W_{k}+\beta P_{k+1}\circ F(W_{k})-\beta|P_{k+1}|\circ G(W_{k})$.  
$$(11)^{\frac{1}{2}}$$

Featuring moving the residual Pk to Wk, (11) is referred to as *transfer* process. The updates (10) and
(11) are performed alternatively until convergence. Tiki-Taka mentioned in [21] is the special case with linear response functions and γ = 0. On the response functions side, it is naturally required to let zero be a symmetric point, i.e., G(0) = 0, which can be implemented by the zero-shifting technique [39] by subtracting a reference array. Convergence properties of Residual Learning. We begin by analyzing the convergence of Residual Learningwithout considering the zero-shift first, which enables us to understand how zero-shifted response functions affect convergence.

If the optimal point W∗exists and is unique, the solution of the lower-level objective has a closed form P
∗(W) := W∗−W
γ. At that time, the upper-level objective equals ∥W∗ − W∥
2. However, the solutions of f(·) are generally non-unique, especially for non-convex objectives with multiple local minima. To ensure the existence and uniqueness of W∗, we assume the objective is strongly convex.

Assumption 3 (µ-strong convexity). The objective f(W) is µ*-strongly convex.*
Under the strongly convex assumption, the optimal point W∗is unique and hence the optimal solution of the lower-level problem in (9) is unique. Since the requirement of strong convexity is non-essential in the development of bilevel optimization [40–43], we believe the proof can be extended to more general cases and will extend it for future work.

Involving two sequences Wk and Pk, Residual Learning converges in different senses, including: (a) the residual array Pk converges to the optimal point P
∗(Wk); (b) Wk converges to the critical point of f(·) or the optimal point W∗; (c) the sum W¯k = Wk + γPk converges to a critical point where ∇f(W¯k) = 0. Taking all these into account, we define the convergence metric as

$$E_{K}^{\rm HL}:=\frac{1}{K}\sum_{k=0}^{K-1}\mathbb{E}\bigg{[}\|\nabla f(\tilde{W}_{k})\|^{2}+O(\|P_{k}-P^{*}(W_{k})\|^{2})+O(\|W_{k}-W^{*}\|^{2})\bigg{]}.\tag{12}$$

For simplicity, the constants in front of some terms in ERL
K are hidden. Now, we provide the convergence of Residual Learning with generic responses.

0 5 10 15 20 25 30 Epoch 20 40 60 80 100 0 5 10 15 20 25 30 Epoch 20 40 60 80 100 Test Accu racy
= 0.5 = 0.6 = 0.7 Analog SGD Tiki-Taka Digital SGD 28 30 95 96 97 Test Accu racy
= 0.6 = 0.7 = 0.8 Analog SGD Tiki-Taka Digital SGD
28 30 98 99
Theorem 3 (Convergence of Residual Learning). Under Assumptions *1–3, with the learning rate* α = O
p1/σ2K
, β = O(αγ3/2), it holds for Residual Learning *that*

$$E_{K}^{\underline{{{R L}}}}\leq\;O\left(\sqrt{\sigma^{2}/K}+\sigma^{2}S_{K}^{\underline{{{R L}}}}\right)$$
(13)
where S
RL
K denotes the amplification factor of Pk *given by* S
RL
K := 1K
PK
$$\left.{\begin{array}{l}{K}\\ {k=0}\end{array}}\right\|{\frac{G(P_{k})}{\sqrt{F(P_{k})}}}\left\|{\begin{array}{l}{2}\\ {1}\end{array}}\right.$$
F (Pk)
$$(13)$$
$\square$
∞
.

The proof of Theorem 3 is deferred to Appendix I. Theorem 3 claims that Residual Learning converges at the rate O
pσ 2/Kto a neighbor of critical point with radius O(σ 2S
RL
K ), which share almost the same expression with the convergence of Analog SGD. The difference lies in the amplification factor S
RL
K and S
ASGD
K , where the former depends on Pk while the latter depends on Wk.

Impact of response functions. Response function affects the Analog SGD and Residual Learning similarly. However, attributed to the residual array, constructing response functions to enable exact convergence of Residual Learning is viable.

As we have discussed, Pk tends to P
∗(Wk) which tends to 0 given Wk tends to W∗. Therefore, response functions with G(P) = 0 when P = 0 are required for the exact convergence. Assumption 4. *(Zero-shifted symmetric point)* P = 0 *is a symmetric point, i.e.* G(0) = 0. Under it and the Lipschitz continuity of the response functions, it holds directly that

√
G(Pk)
F (Pk)
∞
≤
LS∥Pk∥∞ for a constant LS ≥ 0. Consequently, when Pk → P
∗(Wk) → 0 as Wk → W∗, the asymptotic error disappears. Formally, the following corollary holds true. Corollary 1 (Exact convergence of Residual Learning). Under Assumption 4 *and the conditions* in Theorem *3, if* γ ≥ Ω(q
−2/5 min )*, it holds that* ERL
K ≤ O
pσ 2L/K.

The proof of Corollary 1 is deferred to Appendix I.5. Corollary 1 demonstrates the failure of Tiki-Taka in Figure 2. The symmetric point is w
⋄ = cLinτ in this example, which violates Assumption 4 when cLin ̸= 0 and hence introduces asymptotic error into Residual Learning.

## 5 Extension Of Residual Learning: Limited Granularity And Noisy Io

This section extends Residual Learning to practical scenarios with additional hardware imperfections. To be specific, we consider the *noisy IO* and *limited granularity* as examples. We highlight that we are not trying to diminish the importance of imperfection, but rather focus on two of the primary ones known to be crucial.

IO of resistive crossbar arrays introduces noise during the reading of Pk+1 in the transfer process (11), given by Wk+1 = Wk+β(Pk+1+εk)⊙F(Wk)−β|Pk+1+εk|⊙G(Wk) with a noise εk. It incurs the implicit penalty issues again, leading to a penalized upper-level objective ∥P
∗(W)∥
2 +⟨Σε, Rc(W)⟩,
as claimed by Theorem 1, where Σε = E[|εk|] is assumed to be a constant. To filter out the noise, we propose to use a digital buffer Hk to take a moving average of noisy Pk+1 signals by Hk+1 = (1 − β)Hk + β(Pk+1 + εk+1). (14)

$$\vartheta(P_{k+1}+\varepsilon_{k+1}).$$
$$\rho)\pi_{k}+\rho$$

| CIFAR10   |            |            |            |            |            |
|-----------|------------|------------|------------|------------|------------|
| DSGD      | ASGD       | TT/RL      | TTv2       | RLv2       |            |
| ResNet18  | 95.43±0.13 | 84.47±3.40 | 94.81±0.09 | 95.31±0.05 | 95.12±0.14 |
| ResNet34  | 96.48±0.02 | 95.43±0.12 | 96.29±0.12 | 96.60±0.05 | 96.42±0.13 |
| ResNet50  | 96.57±0.10 | 94.36±1.16 | 96.34±0.04 | 96.63±0.09 | 96.56±0.08 |
| CIFAR100  |            |            |            |            |            |
| DSGD      | ASGD       | TT/RL      | TTv2       | RLv2       |            |
| ResNet18  | 81.12±0.25 | 68.98±1.01 | 76.17±0.23 | 78.56±0.29 | 79.83±0.13 |
| ResNet34  | 83.86±0.12 | 78.98±0.55 | 80.58±0.11 | 81.81±0.15 | 82.85±0.19 |
| ResNet50  | 83.98±0.11 | 79.88±1.26 | 80.80±0.22 | 82.82±0.33 | 83.90±0.20 |

Table 1: Fine-tuning ResNet models with the *power response* on CIFAR10/100. Test accuracy is reported. DSGD, ASGD, TT/RL, TTv2, and RLv2 represent Digital SGD, Analog SGD, Residual Learning/Tiki-Taka, and Residual Learning v2, respectively.

Intuitively, with a fixed Pk+1, Hk will converge to a neighborhood of Pk+1 with radius O(β). Therefore, a sufficiently small β renders Hk a fair approximation of noiseless Pk, enabling optimizing the upper-level objective with clearer signals. After that, Hk+1 is transferred to Wk as follows Wk+1 = Wk + βHk+1 ⊙ F(Wk) − β|Hk+1| ⊙ G(Wk). (15)
Furthermore, the transfer process suffers from a constant error of O(∆wmin) due to the discrete pulse firing, each of which changes the weight by O(∆wmin). To overcome these issues, we propose introducing a threshold mechanism that does not transfer the entire Hk+1 to Wk at each iteration, as in (15). Instead, we compute an intermediate value by Hk+ 12
= (1 − β)Hk + β(Pk+1 + εk+1)
first. At each coordinate d, if the value |[Hk+ 12
]d| ≥ ∆wmin, one pulse will be fired to [Wk]d and update the digital buffer by [Hk+1]d = [Hk+ 12
]d − ∆wmin or [Hk+1]d = [Hk+ 12
]d + ∆wmin, where the sign of increment is determined by the sign of [Hk+ 12
]d. Otherwise, no transfer is triggered if the intermediate value falls below the threshold, i.e., [Hk+1]d = [Hk+ 12
]d. The proposed algorithms are referred to as Residual Learning v2.

## 6 Numerical Simulations

In this section, we verify the main theoretical results by simulations on both synthetic datasets and real datasets. We use the open source toolkit IBM Analog Hardware Acceleration Kit (AIHWKIT) [44] to simulate the behaviors of Analog SGD, Residual Learning (which reduces to Tiki-Taka). Each simulation is repeated three times, and the mean and standard deviation are reported. We consider two types of response functions in our simulations: power and exponential response functions with dynamic ranges [−*τ, τ* ] and the symmetric point being 0, as required by Corollary 1. More details, simulations, and ablation studies can be found in Appendix K. The code of our simulations is available at github.com/Zhaoxian-Wu/analog-training. FCN/CNN @ MNIST. We train a fully-connected network (FCN) and a convolutional neural network (CNN) on the MNIST dataset and compare the performance of Analog SGD and Tiki-Taka under various dynamic range τ on power responses; see the results in Figure 4. By tracking residual, Residual Learning outperforms Analog SGD and reaches comparable accuracy with Digital SGD. For both architectures, the accuracy of Residual Learning drops by < 1%. In contrast, Analog SGD takes a few epochs to achieve a noticeable increase in accuracy in FCN training, rendering a slower convergence rate than Residual Learning. In CNN training, Analog SGD's accuracy increases more slowly than Residual Learning, eventually settling at about 80%. It is consistent with the theoretical claims. ResNet @ CIFAR10/CIFAR100. We fine-tune three ResNet models with different scales on CIFAR10/CIFAR100 datasets. The power response functions are used, whose results are shown in Table 1. The results show that the Tiki-Taka outperforms Analog SGD by about 1.0% in most of the cases in ResNet34/50, and the gap even reaches about 7.0% for ResNet18 training on the CIFAR100 dataset. On top of that, we also compare the proposed Residual Learning v2 and Tiki-Taka v2.

Resnet18 Resnet34 Resnet50 0.0 0.1 0.2 0.3 0.4 76 78 80 82 0.0 0.1 0.2 0.3 0.4 93 94 95 96 Ac c ur a cy Ac cu ra cyResnet18 Resnet34 Resnet50
Both of them outperform Residual Learning since they introduce a digital buffer to filter out the reading noise. However, Residual Learning v2 outperforms Tiki-Taka v2 on the CIFAR100 dataset, demonstrating the benefit from the bilevel formulation. Ablation study on γ. We conduct simulations to study the impact of mixing coefficient γ in (10) on the CIFAR10 or CIFAR100 dataset in the ResNet training tasks. The results are presented in Figure 5, which shows that Residual Learning achieves a great accuracy gain from increasing γ from 0 to 0.1, while the gain saturates from 0.1 to 0.4. Therefore, we conclude that Residual Learning benefits from a non-zero γ, and the performance is robust to the γ selection.

## 7 Conclusions And Limitations

This paper studies the impact of a generic class of asymmetric and non-linear response functions on gradient-based training in analog in-memory computing hardware. We first formulate the dynamics of Analog Update based on the pulse update rule. Based on it, we show that Analog SGD implicitly optimizes a penalized objective and hence can only converge inexactly. To overcome this issue, we propose a Residual Learning framework which solves a bilevel optimization problem. Explicitly aligning the algorithmic stationary point and physical symmetric point, Residual Learning provably converges to the optimal point exactly. Furthermore, we demonstrate how to extend Residual Learning to overcome the noisy reading and limited update granularity issues. The efficiency of the proposed method is verified through simulations. One limitation of this work is that the current analysis considers only the three hardware imperfections. While they are known to be crucial for analog training, it is also important to extend our convergence analysis and methods to more practical scenarios involving more imperfections in future work.

## References

[1] Norm Jouppi, George Kurian, Sheng Li, Peter Ma, Rahul Nagarajan, Lifeng Nai, Nishant Patil, Suvinay Subramanian, Andy Swing, Brian Towles, et al. TPU v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings. In *Annual* International Symposium on Computer Architecture, pages 1–14, 2023.

[2] Hadi Esmaeilzadeh, Adrian Sampson, Luis Ceze, and Doug Burger. Neural acceleration for general-purpose approximate programs. In IEEE/ACM international symposium on microarchitecture, pages 449–460. IEEE, 2012.

[3] Dharmendra S Modha, Filipp Akopyan, Alexander Andreopoulos, Rathinakumar Appuswamy, John V Arthur, Andrew S Cassidy, Pallab Datta, Michael V DeBole, Steven K Esser, Carlos Ortega Otero, et al. Neural inference at the frontier of energy, space, and time. *Science*, 382(6668):329–335, 2023.

[4] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.

[5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.

[6] An Chen. A comprehensive crossbar array model with solutions for line resistance and nonlinear device characteristics. *IEEE Transactions on Electron Devices*, 60(4):1318–1326, 2013.

[7] Wilfried Haensch, Tayfun Gokmen, and Ruchir Puri. The next generation of deep learning hardware: Analog computing. *Proceedings of the IEEE*, 107(1):108–122, 2019.

[8] Vivienne Sze, Yu-Hsin Chen, Tien-Ju Yang, and Joel S Emer. Efficient processing of deep neural networks: A tutorial and survey. *Proceedings of the IEEE*, 105(12):2295–2329, 2017.

[9] Abu Sebastian, Manuel Le Gallo, Riduan Khaddam-Aljameh, and Evangelos Eleftheriou.

Memory devices and applications for in-memory computing. *Nature Nanotechnology*, 15:529– 544, 2020.

[10] Manuel Le Gallo, Riduan Khaddam-Aljameh, Milos Stanisavljevic, Athanasios Vasilopoulos, Benedikt Kersting, Martino Dazzi, Geethan Karunaratne, Matthias Brändli, Abhairaj Singh, Silvia M Mueller, et al. A 64-core mixed-signal in-memory compute chip based on phase-change memory for deep neural network inference. *Nature Electronics*, 6(9):680–693, 2023.

[11] Geoffrey W Burr, Robert M Shelby, Abu Sebastian, Sangbum Kim, Seyoung Kim, Severin Sidler, Kumar Virwani, Masatoshi Ishii, Pritish Narayanan, Alessandro Fumarola, et al. Neuromorphic computing using non-volatile memory. *Advances in Physics: X*, 2(1):89–124, 2017.

[12] J Joshua Yang, Dmitri B Strukov, and Duncan R Stewart. Memristive devices for computing.

Nature nanotechnology, 8(1):13, 2013.

[13] Shubham Jain et al. Neural network accelerator design with resistive crossbars: Opportunities and challenges. *IBM Journal of Research and Development*, 63(6):10–1, 2019.

[14] Stefan Cosemans, Bram-Ernst Verhoef, Jonas Doevenspeck, Ioannis A. Papistas, Francky Catthoor, Peter Debacker, Arindam Mallik, and Diederik Verkest. Towards 10000TOPS/W DNN inference with analog in-memory computing - a circuit blueprint, device options and requirements. In *IEEE International Electron Devices Meeting*, pages 22.2.1–22.2.4, 2019.

[15] Ioannis A Papistas, Stefan Cosemans, Bram Rooseleer, Jonas Doevenspeck, M-H Na, Arindam Mallik, Peter Debacker, and Diederik Verkest. A 22 nm, 1540 TOP/s/W, 12.1 TOP/s/mm 2 in-memory analog matrix-vector-multiplier for DNN acceleration. In IEEE Custom Integrated Circuits Conference, pages 1–2. IEEE, 2021.

[16] Tayfun Gokmen and Yurii Vlasov. Acceleration of deep neural network training with resistive cross-point devices: Design considerations. *Frontiers in neuroscience*, 10:333, 2016.

[17] Geoffrey W Burr, Robert M Shelby, Severin Sidler, Carmelo Di Nolfo, Junwoo Jang, Irem Boybat, Rohit S Shenoy, Pritish Narayanan, Kumar Virwani, Emanuele U Giacometti, et al. Experimental demonstration and tolerancing of a large-scale neural network (165 000 synapses) using phase-change memory as the synaptic weight element. IEEE Transactions on Electron Devices, 62(11):3498–3507, 2015.

[18] Sapan Agarwal, Steven J Plimpton, David R Hughart, Alexander H Hsia, Isaac Richter, Jonathan A Cox, Conrad D James, and Matthew J Marinella. Resistive memory device requirements for a neural algorithm accelerator. In *International Joint Conference on Neural Networks*, pages 929–938. IEEE, 2016.

[19] Paiyu Chen, Binbin Lin, I-Ting Wang, Tuohung Hou, Jieping Ye, Sarma Vrudhula, Jae-sun Seo, Yu Cao, and Shimeng Yu. Mitigating effects of non-ideal synaptic device characteristics for on-chip learning. In *IEEE/ACM International Conference on Computer-Aided Design*, pages 194–199. IEEE, 2015.

[20] Vinay Joshi, Manuel Le Gallo, Simon Haefeli, Irem Boybat, Sasidharan Rajalekshmi Nandakumar, Christophe Piveteau, Martino Dazzi, Bipin Rajendran, Abu Sebastian, and Evangelos Eleftheriou. Accurate deep neural network inference using computational phase-change memory.

Nature communications, 11(1):2473, 2020.

[21] Zhaoxian Wu, Tayfun Gokmen, Malte J Rasch, and Tianyi Chen. Towards exact gradient-based training on analog in-memory computing. In *Advances in Neural Information Processing* Systems, 2024.

[22] Tayfun Gokmen and Wilfried Haensch. Algorithm for training neural networks on resistive device arrays. *Frontiers in Neuroscience*, 14, 2020.

[23] Tayfun Gokmen. Enabling training of neural networks on noisy hardware. *Frontiers in Artificial* Intelligence, 4:1–14, 2021.

[24] Malte J Rasch, Fabio Carta, Omobayode Fagbohungbe, and Tayfun Gokmen. Fast and robust analog in-memory deep neural network training. *Nature Communications*, 15(1):7133–7147, 2024.

[25] Peng Yao, Huaqiang Wu, Bin Gao, Sukru Burc Eryilmaz, Xueyao Huang, Wenqiang Zhang, Qingtian Zhang, Ning Deng, Luping Shi, H-S Philip Wong, et al. Face classification using electronic synapses. *Nature communications*, 8(1):15199, 2017.

[26] Zhongrui Wang, Can Li, Peng Lin, Mingyi Rao, Yongyang Nie, Wenhao Song, Qinru Qiu, Yunning Li, Peng Yan, John Paul Strachan, et al. In situ training of feed-forward and recurrent convolutional memristor networks. *Nature Machine Intelligence*, 1(9):434–442, 2019.

[27] Yaoyuan Wang, Shuang Wu, Lei Tian, and Luping Shi. SSM: a high-performance scheme for in situ training of imprecise memristor neural networks. *Neurocomputing*, 407:270–280, 2020.

[28] Shanshi Huang, Xiaoyu Sun, Xiaochen Peng, Hongwu Jiang, and Shimeng Yu. Overcoming challenges for achieving high in-situ training accuracy with emerging memories. In Design, Automation & Test in Europe Conference & Exhibition, pages 1025–1030. IEEE, 2020.

[29] Weier Wan, Rajkumar Kubendran, Clemens Schaefer, Sukru Burc Eryilmaz, Wenqiang Zhang, Dabin Wu, Stephen Deiss, Priyanka Raina, He Qian, Bin Gao, et al. A compute-in-memory chip based on resistive random-access memory. *Nature*, 608(7923):504–512, 2022.

[30] Peng Yao, Huaqiang Wu, Bin Gao, Jianshi Tang, Qingtian Zhang, Wenqiang Zhang, J Joshua Yang, and He Qian. Fully hardware-implemented memristor convolutional neural network. Nature, 577(7792):641–646, 2020.

[31] S. R. Nandakumar, Manuel Le Gallo, Irem Boybat, Bipin Rajendran, Abu Sebastian, and Evangelos Eleftheriou. Mixed-precision architecture based on computational memory for training deep neural networks. In *IEEE International Symposium on Circuits and Systems*, pages 1–5, 2018.

[32] S. R. Nandakumar, Manuel Le Gallo, Christophe Piveteau, Vinay Joshi, Giovanni Mariani, Irem Boybat, Geethan Karunaratne, Riduan Khaddam-Aljameh, Urs Egger, Anastasios Petropoulos, Theodore Antonakopoulos, Bipin Rajendran, Abu Sebastian, and Evangelos Eleftheriou. Mixedprecision deep learning based on computational memory. *Frontiers in Neuroscience*, 14, 2020.

[33] Benjamin Scellier and Yoshua Bengio. Equilibrium propagation: Bridging the gap between energy-based models and backpropagation. *Frontiers in computational neuroscience*, 11:24, 2017.

[34] Mohamed Watfa, Alberto Garcia-Ortiz, and Gilles Sassatelli. Energy-based analog neural network framework. *Frontiers in Computational Neuroscience*, 17:1114651, 2023.

[35] Benjamin Scellier, Maxence Ernoult, Jack Kendall, and Suhas Kumar. Energy-based learning algorithms for analog computing: a comparative study. Advances in Neural Information Processing Systems, 36, 2024.

[36] Jack Kendall, Ross Pantone, Kalpana Manickavasagam, Yoshua Bengio, and Benjamin Scellier.

Training end-to-end analog neural networks with equilibrium propagation. arXiv preprint arXiv:2006.01981, 2020.

[37] Maxence Ernoult, Julie Grollier, Damien Querlioz, Yoshua Bengio, and Benjamin Scellier.

Equilibrium propagation with continual weight updates. *arXiv preprint arXiv:2005.04168*,
2020.

[38] Léon Bottou, Frank E Curtis, and Jorge Nocedal. Optimization methods for large-scale machine learning. *SIAM review*, 60(2):223–311, 2018.

[39] Hyungjun Kim, Malte J Rasch, Tayfun Gokmen, Takashi Ando, Hiroyuki Miyazoe, Jae-Joon Kim, John Rozen, and Seyoung Kim. Zero-shifting technique for deep neural network training on resistive cross-point arrays. *arXiv preprint arXiv:1907.10228*, 2019.

[40] Quan Xiao, Songtao Lu, and Tianyi Chen. A generalized alternating method for bilevel learning under the polyak-łojasiewicz condition. In *Proc. Advances in Neural Info. Process. Syst.*, 2023.

[41] Michael Arbel and Julien Mairal. Non-convex bilevel games with critical point selection maps.

In *Advances in Neural Information Processing Systems*, 2022.

[42] Han Shen, Quan Xiao, and Tianyi Chen. On penalty-based bilevel gradient descent method. In Proc. of International Conference on Machine Learning, 2023.

[43] Jeongyeol Kwon, Dohyun Kwon, Steve Wright, and Robert Nowak. On penalty methods for nonconvex bilevel optimization and first-order stochastic approximation. In Proc. of International Conference on Learning Representations, 2024.

[44] Malte J Rasch, Diego Moreda, Tayfun Gokmen, Manuel Le Gallo, Fabio Carta, Cindy Goldberg, Kaoutar El Maghraoui, Abu Sebastian, and Vijay Narayanan. A flexible and fast PyTorch toolkit for simulating training and inference on analog crossbar arrays. *IEEE International Conference* on Artificial Intelligence Circuits and Systems, pages 1–4, 2021.

[45] Geoffrey W Burr, Matthew J BrightSky, Abu Sebastian, Huai-Yu Cheng, Jau-Yi Wu, Sangbum Kim, Norma E Sosa, Nikolaos Papandreou, Hsiang-Lan Lung, Haralampos Pozidis, Evangelos Eleftheriou, and Chung H Lam. Recent Progress in Phase-Change Memory Technology. IEEE
Journal on Emerging and Selected Topics in Circuits and Systems, 6(2):146–162, 2016.

[46] Manuel Le Gallo and Abu Sebastian. An overview of phase-change memory device physics.

Journal of Physics D: Applied Physics, 53(21):213002, 2020.

[47] Jun-Woo Jang, Sangsu Park, Yoon-Ha Jeong, and Hyunsang Hwang. ReRAM-based synaptic device for neuromorphic computing. In *IEEE International Symposium on Circuits and Systems*, pages 1054–1057, 2014.

[48] Jun-Woo Jang, Sangsu Park, Geoffrey W Burr, Hyunsang Hwang, and Yoon-Ha Jeong. Optimization of conductance change in Pr1−xCaxMnO3-based synaptic devices for neuromorphic systems. *IEEE Electron Device Letters*, 36(5):457–459, 2015.

[49] Tommaso Stecconi, Valeria Bragaglia, Malte J Rasch, Fabio Carta, Folkert Horst, Donato F
Falcone, Sofieke C Ten Kate, Nanbo Gong, Takashi Ando, Antonis Olziersky, et al. Analog resistive switching devices for training deep neural networks with the novel Tiki-Taka algorithm.

Nano Letters, 24(3):866–872, 2024.

[50] Seokjae Lim, Myounghoon Kwak, and Hyunsang Hwang. Improved synaptic behavior of CBRAM using internal voltage divider for neuromorphic systems. IEEE Transactions on Electron Devices, 65(9):3976–3981, 2018.

[51] Elliot J Fuller, Scott T Keene, Armantas Melianas, Zhongrui Wang, Sapan Agarwal, Yiyang Li, Yaakov Tuchman, Conrad D James, Matthew J Marinella, J Joshua Yang, Alberto Salleo, and A Alec Talin. Parallel programming of an ionic floating-gate memory array for scalable neuromorphic computing. *Science*, 364(6440):570–574, 2019.

[52] Jianshi Tang, Douglas Bishop, Seyoung Kim, Matt Copel, Tayfun Gokmen, Teodor Todorov, SangHoon Shin, Ko-Tao Lee, Paul Solomon, Kevin Chan, et al. ECRAM as scalable synaptic cell for high-speed, low-power neuromorphic computing. In IEEE International Electron Devices Meeting, pages 13–1. IEEE, 2018.

[53] Murat Onen, Nicolas Emond, Baoming Wang, Difei Zhang, Frances M Ross, Ju Li, Bilge Yildiz, and Jesús A Del Alamo. Nanosecond protonic programmable resistors for analog deep learning.

Science, 377(6605):539–543, 2022.

[54] Seungchul Jung, Hyungwoo Lee, Sungmeen Myung, Hyunsoo Kim, Seung Keun Yoon, Soon-
Wan Kwon, Yongmin Ju, Minje Kim, Wooseok Yi, Shinhee Han, et al. A crossbar array of magnetoresistive memory devices for in-memory computing. *Nature*, 601(7892):211–216, 2022.

[55] Zhihua Xiao, Vinayak Bharat Naik, Jia Hao Lim, Yaoru Hou, Zhongrui Wang, and Qiming Shao.

Adapting magnetoresistive memory devices for accurate and on-chip-training-free in-memory computing. *Science Advances*, 10(38):eadp3710, 2024.

[56] Rui Guo, Weinan Lin, Xiaobing Yan, T Venkatesan, and Jingsheng Chen. Ferroic tunnel junctions and their application in neuromorphic networks. *Applied physics reviews*, 7(1), 2020.

[57] Panni Wang, Feng Xu, Bo Wang, Bin Gao, Huaqiang Wu, He Qian, and Shimeng Yu. Threedimensional NAND flash for vector-matrix multiplication. IEEE Transactions on Very Large Scale Integration Systems, 27(4):988–991, 2018.

[58] Yachen Xiang, Peng Huang, Runze Han, Chu Li, Kunliang Wang, Xiaoyan Liu, and Jinfeng Kang. Efficient and robust spike-driven deep convolutional neural networks based on NOR flash computing array. *IEEE Transactions on Electron Devices*, 67(6):2329–2335, 2020.

[59] Farnood Merrikh-Bayat, Xinjie Guo, Michael Klachko, Mirko Prezioso, Konstantin K Likharev, and Dmitri B Strukov. High-performance mixed-signal neurocomputing with nanoscale floatinggate memory cell arrays. *IEEE Transactions on Nneural Networks and Learning Systems*,
29(10):4782–4790, 2017.

[60] Bonan Zhang, Peter Deaville, and Naveen Verma. Statistical computing framework and demonstration for in-memory computing systems. In *ACM/IEEE Design Automation Conference*, pages 979–984, 2022.

[61] Peter Deaville, Bonan Zhang, Lung-Yen Chen, and Naveen Verma. A maximally row-parallel MRAM in-memory-computing macro addressing readout circuit sensitivity and area. In IEEE
European Solid State Circuits Conference, pages 75–78. IEEE, 2021.

[62] Jung-Hoon Lee, Dong-Hyeok Lim, Hongsik Jeong, Huimin Ma, and Luping Shi. Exploring cycle-to-cycle and device-to-device variation tolerance in mlc storage-based neural network training. *IEEE Transactions on Electron Devices*, 66(5):2172–2178, 2019.

[63] Jintao Zhang, Zhuo Wang, and Naveen Verma. In-memory computation of a machine-learning classifier in a standard 6t SRAM array. *IEEE Journal of Solid-State Circuits*, 52(4):915–924, 2017.

[64] Tayfun Gokmen, Malte J Rasch, and Wilfried Haensch. The marriage of training and inference for scaled deep learning analog hardware. In *IEEE International Electron Devices Meeting*, pages 22–3. IEEE, 2019.

[65] Corey Lammie, Athanasios Vasilopoulos, Julian Büchel, Giacomo Camposampiero, Manuel Le Gallo, Malte Rasch, and Abu Sebastian. Improving the accuracy of analog-based in-memory computing accelerators post-training. In *IEEE International Symposium on Circuits and Systems*,
pages 1–5. IEEE, 2024.

[66] Qing Jin, Zhiyu Chen, Jian Ren, Yanyu Li, Yanzhi Wang, and Kaiyuan Yang. PIM-QAT:
Neural network quantization for processing-in-memory (PIM) systems. arXiv preprint arXiv:2209.08617, 2022.

[67] Malte J Rasch, Charles Mackin, Manuel Le Gallo, An Chen, Andrea Fasoli, Frédéric Odermatt, Ning Li, S. R. Nandakumar, Pritish Narayanan, Hsinyu Tsai, et al. Hardware-aware training for large-scale and diverse deep learning inference workloads using in-memory computing-based accelerators. *Nature Communications*, 14(1):5282, 2023.

[68] Bonan Zhang, Chia-Yu Chen, and Naveen Verma. Reshape and adapt for output quantization
(RAOQ): Quantization-aware training for in-memory computing systems. In *International* Conference on Machine Learning, 2024.

[69] Beiye Liu, Hai Li, Yiran Chen, Xin Li, Qing Wu, and Tingwen Huang. Vortex: Variation-aware training for memristor x-bar. In *Proceedings of the 52nd Annual Design Automation Conference*, pages 1–6, 2015.

[70] Abhiroop Bhattacharjee, Lakshya Bhatnagar, Youngeun Kim, and Priyadarshini Panda. NEAT:
Non-linearity aware training for accurate and energy-efficient implementation of neural networks on 1t-1r memristive crossbars. *arXiv preprint arXiv:2012.00261*, 2020.

[71] Tayfun Gokmen, Murat Onen, and Wilfried Haensch. Training deep convolutional neural networks with resistive cross-point devices. *Frontiers in neuroscience*, 11:538, 2017.

[72] Zhongrui Wang, Saumil Joshi, Sergey Savel'Ev, Wenhao Song, Rivu Midya, Yunning Li, Mingyi Rao, Peng Yan, Shiva Asapu, Ye Zhuo, et al. Fully memristive neural networks for pattern classification with unsupervised learning. *Nature Electronics*, 1(2):137–145, 2018.

[73] Nanbo Gong, Malte Rasch, Soon-Cheon Seo, Arthur Gasasira, Paul Solomon, Valeria Bragaglia, Steven Consiglio, Hisashi Higuchi, Chanro Park, Kevin Brew, et al. Deep learning acceleration in 14nm CMOS compatible ReRAM array: device, material and algorithm co-optimization. In IEEE International Electron Devices Meeting, 2022.

[74] Zhaoxian Wu, Quan Xiao, Tayfun Gokmen, Hsinyu Tsai, Kaoutar El Maghraoui, and Tianyi Chen. Pipeline gradient-based model training on analog in-memory accelerators. arXiv preprint arXiv:2410.15155, 2024.

[75] Logan G Wright, Tatsuhiro Onodera, Martin M Stein, Tianyu Wang, Darren T Schachter, Zoey Hu, and Peter L McMahon. Deep physical neural networks trained with backpropagation.

Nature, 601(7894):549–555, 2022.

[76] Ali Momeni, Babak Rahmani, Benjamin Scellier, Logan G Wright, Clara C McMahon, Peter L andWanjura, Yuhang Li, Anas Skalli, Natalia G. Berloff, Tatsuhiro Onodera, Ilker Oguz, Francesco Morichetti, Philipp del Hougne, Manuel Le Gallo, Abu Sebastian, Azalia Mirhoseini, Cheng Zhang, Danijela Markovic, Daniel Brunner, Christophe Moser, Sylvain Gigan, Florian ´ Marquardt, Aydogan Ozcan, Julie Grollier, Andrea J Liu, Demetri Psaltis, Andrea Alù, and Romain Fleury. Training of physical neural networks. *arXiv preprint arXiv:2406.03372*, 2024.

[77] Demetri Psaltis, David Brady, Xiang-Guang Gu, and Steven Lin. Holography in artificial neural networks. *Nature*, 343(6256):325–330, 1990.

[78] Tyler W Hughes, Ian AD Williamson, Momchil Minkov, and Shanhui Fan. Wave physics as an analog recurrent neural network. *Science advances*, 5(12):eaay6946, 2019.

[79] Alexander N Tait, Thomas Ferreira De Lima, Ellen Zhou, Allie X Wu, Mitchell A Nahmias, Bhavin J Shastri, and Paul R Prucnal. Neuromorphic photonic networks using silicon photonic weight banks. *Scientific reports*, 7(1):7430, 2017.

[80] Nanbo Gong, T Idé, S Kim, Irem Boybat, Abu Sebastian, V Narayanan, and Takashi Ando.

Signal and noise extraction from analog memory elements for neuromorphic computing. Nature communications, 9(1):2102, 2018.

[81] Mingyi Rao, Hao Tang, Jiangbin Wu, Wenhao Song, Max Zhang, Wenbo Yin, Ye Zhuo, Fatemeh Kiani, Benjamin Chen, Xiangqi Jiang, et al. Thousands of conductance levels in memristors integrated on CMOS. *Nature*, 615(7954):823–829, 2023.

[82] Deepak Sharma, Santi Prasad Rath, Bidyabhusan Kundu, Anil Korkmaz, Damien Thompson, Navakanta Bhat, Sreebrata Goswami, R Stanley Williams, and Sreetosh Goswami. Linear symmetric self-selecting 14-bit kinetic molecular memristors. *Nature*, 633(8030):560–566, 2024.

[83] Wenhao Song, Mingyi Rao, Yunning Li, Can Li, Ye Zhuo, Fuxi Cai, Mingche Wu, Wenbo Yin, Zongze Li, Qiang Wei, et al. Programming memristor arrays with arbitrarily high precision for analog computing. *Science*, 383(6685):903–910, 2024.

[84] Shubham Jain, Hsinyu Tsai, Ching-Tzu Chen, Ramachandran Muralidhar, Irem Boybat, Martin M Frank, Stanisław Wo´zniak, Milos Stanisavljevic, Praneet Adusumilli, Pritish Narayanan, et al. A heterogeneous and programmable compute-in-memory accelerator architecture for analog-ai using dense 2-d mesh. *IEEE Transactions on Very Large Scale Integration Systems*, 31(1):114–127, 2022.

[85] Yurii Nesterov. *Introductory Lectures on Convex Optimization: A Basic Course*. Springer, 2013.

# Appendix For "Analog In-Memory Training On Non-Ideal Resistive Elements: Understanding The Impact Of Response Functions"

| Table of Contents A Literature Review                  | 17                                                           |    |    |
|--------------------------------------------------------|--------------------------------------------------------------|----|----|
| B                                                      | Relation with the result in [21]                             | 19 |    |
| C                                                      | Dynamics of Non-ideal Analog Update                          | 21 |    |
| D                                                      | Comparison of Residual Learning v2 and Tiki-Taka v2          | 22 |    |
| E                                                      | Estimation of time consumption                               | 23 |    |
| F                                                      | Useful Lemmas and Proofs                                     | 23 |    |
| F.1                                                    | Lemma 1: Properties of weighted norm                         | 23 |    |
| F.2                                                    | Lemma 2: Properties of weighted norm                         | 23 |    |
| F.3                                                    | Lemma 3: Lipschitz continuity of analog update               |    | 23 |
| F.4                                                    | Lemma 4: Element-wise product error                          |    | 24 |
| G Proof of Theorem 1: Implicit Bias of Analog Training | 24                                                           |    |    |
| H Proof of Theorem 2: Convergence of Analog SGD        | 27                                                           |    |    |
| I                                                      | Proof of Theorem 3: Convergence of Residual Learning         | 30 |    |
| I.1                                                    | Main proof                                                   | 30 |    |
| I.2                                                    | Proof of Lemma 5: Descent of sequence W¯ k                   |    | 34 |
| I.3                                                    | Proof of Lemma 6: Descent of sequence Wk                     |    | 37 |
| I.4                                                    | Proof of Lemma 7: Descent of sequence Pk                     | 38 |    |
| I.5                                                    | Proof of Corollary 1: Exact convergence of Residual Learning | 41 |    |
| J                                                      | Proof of Theorem 6: Convergence of Analog GD                 | 42 |    |
| K Simulation Details and Additional Results            | 43                                                           |    |    |
| K.1                                                    | Power and Exponential Response Functions                     | 44 |    |
| K.2                                                    | Least squares problem                                        | 44 |    |
| K.3                                                    | Classification problem                                       | 45 |    |
| K.4                                                    | Additional performance on real datasets                      | 45 |    |
| K.5                                                    | Ablation study on cycle variation                            |    | 46 |
| K.6                                                    | Ablation study on various response functions                 | 46 |    |
| L                                                      | Broader Impact                                               | 47 |    |

## A Literature Review

This section briefly reviews literature that is related to this paper, as complementary to Section 1. Training on AIMC hardware. Analog training has shown promising early successes in tasks such as face classification [25] and digit classification [26], achieving 1, 000× lower energy consumption than digital implementations. Researchers are also exploring approaches to mitigate the impact of hardware non-idealities. For example, [27, 28] proposes leveraging the momentum technique to stabilize training by reducing noise. To address other potential non-idealities, a hybrid training paradigm is also being explored. [29] leverages the chip-in-the-loop technique to train models layer-by-layer, while [30] proposes to train the backbone in the digital domain and train the last layer in the analog domain. In general, these works have provided valuable insights into analog training, shedding light on many critical technical challenges. However, their focus has largely been on experimental and simulation aspects, with limited systematic and theoretical analysis of how specific imperfections affect the training process. In our paper, we present an alternative viewpoint and novel tools to explore the effects of non-idealities. Resistive element. A series of works seeks various resistive elements that have near-constant or at least symmetric responses. The leading candidates currently include PCM [45, 46], ReRAM [47–49], CBRAM [50, 51], ECRAM [52, 53], MRAM [54, 55], FTJ [56] or flash memory [57–59]. However, a resistive element with symmetric updates may not be the best option for manufacturing. For example, although ECRAM provides almost symmetric updates, it remains less competitive than ReRAM, which offers faster response speed and lower pulse voltage [49]. The suitability of the resistive elements is evaluated using metrics across multiple dimensions, including the number of conductance states, retention, material endurance, switching energy, response speed, manufacturing cost, and cell size. Among them, this paper is only interested in the impact of response functions in the training. Imperfection of AIMC hardware. Besides the response functions, analog training suffers from all kinds of hardware imperfection, especially when the task's scale increases, like asymmetric update [17, 19], reading/writing noise [18, 60, 61], device/cycle variations [62], non-linear current response due to IR drop [18, 6, 63]. This paper mainly focuses on asymmetric response functions. However, this paper is not trying to diminish the importance of other hardware imperfections but rather focuses on one of the primary ones known to be very important [19, 16]. Hardware-aware training. For inference on AIMC hardware purposes, models pretrained on digital hardware will be programmed on analog hardware. Due to hardware imperfections, the pretrained models suffer performance drops. Hardware-aware training (HWA) is a technique designed to bridge the gap between ideal pretrained models and non-ideal programmed models. In contrast to standard training methods, hardware-aware training explicitly incorporates device-specific imperfections, such as weight drift [20], device fail [64], bounded dynamic range [65], quantization error from ADC [66–68], device variation [69], and non-linear current output [70], into the training loop. By modeling these constraints during training, the learned parameters become inherently more robust to real-world deployment conditions. It is worth highlighting that HWA is still performed on digital hardware, and the trained model will be programmed onto AIMC hardware. On the contrary, this paper considers a different, more challenging setting in which training is performed directly on analog hardware.

Gradient-based training on AIMC hardware. A series of works focuses on implementing backpropagation (BP) and gradient-based training on AIMC hardware. The seminal work [16, 71] leverages the rank-one structure of the gradient and implements Analog SGD by a stochastic pulse update scheme, *rank-update*. Rank-update significantly accelerates the gradient descent step by avoiding the O(N2)-element computation of gradients and instead using two vectors with O(N)
elements for the update, where N is the number of matrix rows and columns. To alleviate the asymmetric update issue, researchers also design various of Analog SGD variants, Tiki-Taka algorithm family [22–24]. The key components of Tiki-Taka are the introduction of a *residual* array to stabilize training. Apart from the rank-update, a hybrid scheme that performs forward and backward passes in the analog domain but computes gradients in the digital domain has been proposed in [31, 32]. Their solution, referred to as *mixed-precision update*, provides a more accurate gradient signal but requires 5×-10× higher overhead compared to the rank-update scheme [24]. Attributed to these efforts, analog training has empirically shown great promise, achieving accuracy comparable to that of digital training on chip prototypes while reducing energy consumption and training time [72, 73]. Simultaneously, the parallel acceleration solution with AIMC hardware is under exploration [74]. Despite its good performance, it remains mysterious when and why the analog training works. Theoretical foundation of gradient-based training. The closely related result comes from the convergence study of Tiki-Taka [21]. Similar to our work, they attempt to model the dynamics and provide the convergence properties of Analog SGD and Tiki-Taka. However, their work is limited to a special linear response function. Furthermore, their paper considers a simplified version of Tiki-Taka, with a hyper-parameter γ = 0 (see Section 4). As we will show empirically and

| γ                       | Generic response   | Linear response   |           |               |    |
|-------------------------|--------------------|-------------------|-----------|---------------|----|
| Tiki-Taka [21]          | = 0                | %                 | O q 1           |                |    |
| K                       | 1                  |                   |           |               |    |
| 1−33P 2max/τ2           |                    |                   |           |               |    |
| q 1                         | 1                  |                    | O q 1           | 1             |     |
| Tiki-Taka [Corollary 1] | ̸= 0                | O                 | K MRL min | K 1−P 2max/τ2 |    |

Table 2: Comparison between our paper and [21]. Mixing-coefficient γ is a hyper-parameter of Tiki-Taka. "Generic response" and "Linear response" columns are the convergence rates in the corresponding settings. K represents the number of iterations. MRL
min and P
2max/τ 2 < 1 measure the saturation while the former one reduces to the latter on linear response functions. Energy-based models and equilibrium propagation. Apart from achieving explicit gradient signals by the BP, there are also attempts to train models based on *equilibrium propagation* (EP, [33]), which provides a biologically plausible alternative to traditional BP. EP is applicable to a series of energy-based models, where the forward pass is performed by minimizing an energy function [34, 35]. The update signal in EP is computed by measuring the output difference between a free phase and an active phase. EP eliminates the need for BP non-local weight transport mechanism, making it more compatible with neuromorphic and energy-efficient hardware [36, 37]. We highlight here that the approach to attain update signals (BP or EP) is orthogonal to the update mechanism
(pulse update). Their difference lies in the objective f(Wk), which is hidden in this paper. Therefore, building upon the pulse update, our work is applicable to both BP and EP.

Physical neural network. The model executing on AIMC hardware, which leverages resistive crossbar array to accelerate MVM operation, is a concrete implementation of physical neural networks
(PNNs, [75, 76]). PNN is a generic concept of implementing neural networks via a physical system in which a set of tunable parameters, such as holographic grating [77], wave-based systems [78], and photonic networks [79]. Our work particularly focuses on training with AIMC hardware, but the methodology developed in this paper can be transferred to the study of other PNNs.

## B Relation With The Result In [21]

Similar to this paper, [21] also attempts to model the dynamics of analog training. They show that Analog SGD converges to a critical point of problem (1) inexactly with an asymptotic error, and Tiki-Taka converges to a critical point exactly. In this section, we compare our results with our results and theirs. As discussed in Section 1, [21] studies the analog training on special linear response functions

 $\text{Its points include}$ . 
$\mathbf{a}$
$$\begin{array}{l l l l}{{}}&{{}}&{{}}&{{}}\\ {{}}&{{}}&{{}}&{{}}\\ {{}}&{{}}&{{q_{+}(w)=1-\frac{w}{\tau},\quad q_{-}(w)=1+\frac{w}{\tau}.}}\end{array}$$
$$(16)$$

It can be checked that the symmetric point is 0 while the dynamic range of it is [−*τ, τ* ]. The symmetric and asymmetric components are defined by F(W) = 1 and G(W) = W
τ
, respectively. It indicates Fmax = 1. Furthermore, they assume the bounded weight saturation by assuming bounded weights, i.e., ∥Wk∥∞ ≤ Wmax, ∀k ∈ [K] with a constant Wmax < τ . Under this assumption, the lower bounds of response functions are given by

.$\mbox{ax},\,\forall k\,\subset\,\lfloor x\rfloor$  Uncitions are:  . 
$$q_{\mathrm{max}}=1+{\frac{W_{\mathrm{max}}}{\tau}},\quad q_{\mathrm{min}}=1-{\frac{W_{\mathrm{max}}}{\tau}},$$
τ, (17)
$$\min\{M(W_{k})\}=\min\{Q_{+}(W_{k})\odot Q_{-}(W_{k})\}=1-\left(\frac{\|W_{k}\|_{\infty}}{\tau}\right)^{2}$$ $$M_{\min}^{\tt{ASRO}}=\min\{M(W_{k})\}=1-\left(\frac{W_{\max}}{\tau}\right)^{2}.$$  **analyzing the convergence of Tikh. Takes with generic monotone functions**
Challenge of analyzing the convergence of Tiki-Taka **with generic response functions.** For linear response functions (16), the recursion of residual array Pk has a special structure, where the

$$(17)$$

$$(18)$$
$$(19)$$

first and the biased term can be combined

$$P_{k+1}=P_{k}-\alpha\nabla f(\bar{W}_{k};\xi_{k})-\frac{\alpha}{\tau}|\nabla f(\bar{W}_{k};\xi_{k})|\odot P_{k}$$ $$=\left(1-\frac{\alpha}{\tau}|\nabla f(\bar{W}_{k};\xi_{k})|\right)\odot P_{k}-\alpha\nabla f(\bar{W}_{k};\xi_{k})$$

which is a weighted average of Pk and ∇f(W¯k; ξk). Consequently, Pk can be interpreted as an approximation of the average gradient. From this perspective, the transfer operation can be interpreted as biased gradient descent. However, given a generic G(·), the combination is no longer viable, bringing difficulties to the analysis.

Convergence of Analog SGD. As we will show in Remark 1 at the end of Appendix H, inequality (8)
can be improved when the saturation never happens

1 K K X −1 k=0 E[∥∇f(Wk)∥ 2] (21) k=0  G(Wk) pF(Wk)  2 ≤ 4F 2max MRL min r(f(W0) − f ∗)σ 2L K+ 2Fmaxσ 2 × 1 K K X −1 ∞ , min{M(Wk)} ≤ O  r(f(W0) − f ∗)σ 2L K 1 1 − W2max/τ 2 ! + 2σ 2 × 1 K X K ∥Wk∥ 2∞/τ 2 1 − ∥Wk∥ 2∞/τ 2 k=0 which is exactly the result in [21].
$$(20)$$
$$(21)$$
Convergence of Tiki-Taka. It is shown empirically that a non-zero γ in (10) improves the training accuracy [22]. However, [21] only considers γ = 0 while this paper considers a non-zero γ. With the linear response, if we also assume the bounded saturation of Pk by letting ∥Pk∥∞ ≤ Pmax, the minimal average response function is given by MRL
min = 1 −Pmax τ2. The upper bound in Corollary 1 becomes

$$\frac{1}{K}\sum_{k=0}^{K-1}\|\nabla f(W_{k})\|^{2}\leq O\left(\frac{1}{1-P_{\max}^{2}/\tau^{2}}\sqrt{\frac{(f(W_{0})-f^{*})\sigma^{2}L}{K}}\right).$$  As a comparison, without a non-zero $\gamma$, [21] shows that convergence rate of Tikh-Taka is only 
$$\frac{1}{K}\sum_{k=0}^{K-1}\|\nabla f(W_{k})\|^{2}\leq O\left(\frac{1}{1-33P_{\mathrm{max}}^{2}/\tau^{2}}\sqrt{\frac{(f(W_{0})-f^{*})\sigma^{2}L}{K}}\right).$$
$$(22)$$
$$(23)$$
$$(24)$$

Even though it is not a completely fair comparison, since the two papers rely on different assumptions, it is still worth comparing their analyses. [21] assumes the noise should be non-zero, i.e.

[Eξ[|∇f(W; ξ)|]]d ≥ cnoiseσ, ∀d ∈ [D] holds for a non-zero constant cnoise. Instead, this paper does not make this assumption but assumes that the objective is strongly convex. As mentioned in Section 4, the strong convexity is introduced only to ensure the existence of P
∗(Wk). Therefore, we believe it can be relaxed and that the convergence rate can remain unchanged, which is left for future work. Taking that into account, we believe the comparison can provide insight into how the non-zero γ improves the convergence rate of Tiki-Taka.

Why does non-zero γ **improve the convergence rate of** Tiki-Taka? As discussed in Section 4, Pk is interpreted as a residual array that optimizes f(Wk + γP). In the ideal setting that F(W) = 1 and G(W) = 0, it can be shown that Pk converges to P
∗(Wk) if Wk is fixed and Pk is kept updated, even though the Wk ̸= W∗(hence ∇f(Wk) ̸= 0).

Instead, without a non-zero γ, [21] interprets Pk as an approximation of clear gradient by showing

$$\mathbb{E}_{\xi_{k}}[\|P_{k+1}-C\nabla f(W_{k})\|^{2}]$$ $$\leq\left(1-\frac{\beta}{C}\right)\|P_{k}-C\nabla f(W_{k})\|^{2}+O(\beta C^{\prime})\|\nabla f(W_{k})\|^{2}+\text{remainder}$$

where *C, C*′are constants depending on the resistive element and model dimension, and the "remainder" is the non-essential terms. Consider the case that Wk is fixed and (10) is kept iterating, in which