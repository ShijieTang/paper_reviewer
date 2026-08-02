# OSCILLATORY STATE-SPACE MODELS

T. Konstantin Rusch MIT tkrusch@mit.edu Daniela Rus MIT

### ABSTRACT

We propose Linear Oscillatory State-Space models (LinOSS) for efficiently learning on long sequences. Inspired by cortical dynamics of biological neural networks, we base our proposed LinOSS model on a system of forced harmonic oscillators. A stable discretization, integrated over time using fast associative parallel scans, yields the proposed state-space model. We prove that LinOSS produces stable dynamics only requiring nonnegative diagonal state matrix. This is in stark contrast to many previous state-space models relying heavily on restrictive parameterizations. Moreover, we rigorously show that LinOSS is universal, i.e., it can approximate any continuous and causal operator mapping between time-varying functions, to desired accuracy. In addition, we show that an implicit-explicit discretization of LinOSS perfectly conserves the symmetry of time reversibility of the underlying dynamics. Together, these properties enable efficient modeling of long-range interactions, while ensuring stable and accurate long-horizon forecasting. Finally, our empirical results, spanning a wide range of time-series tasks from mid-range to very long-range classification and regression, as well as long-horizon forecasting, demonstrate that our proposed LinOSS model consistently outperforms state-of-the-art sequence models. Notably, LinOSS outperforms Mamba and LRU by nearly 2x on a sequence modeling task with sequences of length 50k. Code: <https://github.com/tk-rusch/linoss>.

# 1 INTRODUCTION

State-space models [\(Gu et al., 2021;](#page-10-0) [Hasani et al., 2022;](#page-11-0) [Smith et al., 2023;](#page-12-0) [Orvieto et al., 2023\)](#page-11-1) have recently emerged as a powerful tool for learning on long sequences. These models posses the statefullness and fast inference capabilities of Recurrent Neural Networks (RNNs) together with many of the benefits of Transformers [\(Vaswani, 2017;](#page-12-1) [Devlin, 2018\)](#page-10-1), such as efficient training and competitive performance on large-scale language and image modeling tasks. For these reasons, state-space models have been successfully implemented as foundation models, surpassing Transformer-based counterparts in several key modalities, including language, audio, and genomics [\(Gu & Dao, 2023\)](#page-10-2).

Originally, state-space models have been introduced to modern sequence modelling by leveraging specific structures of the state matrix, i.e., normal plus low-rank HiPPO matrices [\(Gu et al., 2020;](#page-10-3) [2021\)](#page-10-0), allowing to solve linear recurrences via a Fast Fourier Transform (FFT). This has since been simplified to only requiring diagonal state matrices [\(Gu et al., 2022a;](#page-10-4) [Smith et al., 2023;](#page-12-0) [Orvi](#page-11-1)[eto et al., 2023\)](#page-11-1) while still obtaining similar or even better performance. However, due to the linear nature of state-space models, the corresponding state matrices need to fulfill specific structural properties in order to learn long-range interactions and produce stable predictions. Consequently, these structural requirements heavily constrain the underlying latent feature space, potentially impairing the model's expressive power.

In this article, we adopt a radically different approach by observing that forced harmonic oscillators, the basis of many systems in physics, biology, and engineering, can produce stable dynamics while at the same time seem to ensure expressive representations. Motivated by this, we propose to construct state-space models based on stable discretizations of forced linear second-order ordinary differential equations (ODEs) modelling oscillators. Our additional contributions are:

- we introduce implicit and implicit-explicit associative parallel scans ensuring fast training and inference.

- we show that our proposed state-space model yields stable dynamics and is able to learn long-range interactions only requiring nonnegative diagonal state matrix.
- we demonstrate that a symplectic discretization of our underlying oscillatory system conserves its symmetry of time reversibility.
- we rigorously prove that our proposed state-space model is a universal approximator of continuous and causal operators between time-series.
- we provide an extensive empirical evaluation of our model on a wide variety of sequential data sets with sequence lengths reaching up to 50k. Our results demonstrate that our model consistently outperforms or matches the performance of state-of-the-art state-space models, including Mamba, S4, S5, and LRU.

# 2 THE PROPOSED STATE-SPACE MODEL

Our proposed state-space model is based on the following system of forced linear second-order ODEs together with a linear readout,

$$\begin{aligned} \mathbf{y}''(t) &= -\mathbf{A}\mathbf{y}(t) + \mathbf{B}\mathbf{u}(t) + \mathbf{b}, \\ \mathbf{x}(t) &= \mathbf{C}\mathbf{y}(t) + \mathbf{D}\mathbf{u}(t), \end{aligned} \tag{1}$$

with hidden state <sup>y</sup>(t) ∈ <sup>R</sup> <sup>m</sup>, output state <sup>x</sup>(t) ∈ <sup>R</sup> q , time-dependent input signal <sup>u</sup>(t) ∈ <sup>R</sup> p , weights <sup>A</sup> ∈ <sup>R</sup> <sup>m</sup>×<sup>m</sup>, <sup>B</sup> ∈ <sup>R</sup> m×p , <sup>C</sup> ∈ <sup>R</sup> <sup>q</sup>×<sup>m</sup>, <sup>D</sup> ∈ <sup>R</sup> q×p , and bias <sup>b</sup> ∈ <sup>R</sup> <sup>m</sup>. Note that A is a diagonal matrix, i.e., with non-zero entries only on its diagonal. We further introduce an auxiliary state <sup>z</sup>(t) ∈ <sup>R</sup> <sup>m</sup>, with z = y ′ . We can thus write [\(1\)](#page-1-0) (omitting the bias b and linear readout x) equivalently as,

$$\begin{aligned}\mathbf{z}'(t) &= -\mathbf{A}\mathbf{y}(t) + \mathbf{B}\mathbf{u}(t), \\ \mathbf{y}'(t) &= \mathbf{z}(t).\end{aligned}\tag{2}$$

![](_page_1_Diagram_7.jpeg)

Figure 1: Schematic drawing of the proposed Linear Oscillatory State-Space model (LinOSS). The input sequences are processed through multiple LinOSS blocks. Each block is composed of a LinOSS layer [\(2\)](#page-1-1) (y(t) plotted using orange solid lines and z(t) using blue dashed lines) followed by a nonlinear transformation, specifically a Gated Linear Units [\(Dauphin et al., 2017\)](#page-10-5) (GLU) layer in our case. After passing through several LinOSS blocks, the latent sequences are decoded to produce the final output sequence.

# 2.1 MOTIVATION AND BACKGROUND

To demonstrate that the underlying ODE in [\(1\)](#page-1-0) models a network of forced harmonic oscillators, we begin with the scalar case by setting p = m = 1 in [\(1\)](#page-1-0). Choosing B = b = 0, we obtain the classic

ODE, y ′′ + Ay = 0, which describes simple harmonic motion with frequency A > 0, such as that of a simple pendulum [\(Guckenheimer & Holmes, 1990\)](#page-11-2). Now, allowing <sup>B</sup> ̸= 0 introduces external forcing proportional to the input signal u(t), where B modulates the effect of the forcing. Finally, setting p, m > 1 yields an uncoupled system of forced harmonic oscillators.

Neuroscience inspiration. Our approach is inspired by neurobiology, where the periodic spiking and firing of action potentials in individual neurons can be observed and analyzed as oscillatory phenomena. Furthermore, entire networks of cortical neurons exhibit behavior-dependent oscillations, reviewed in [Buzsaki & Draguhn](#page-10-6) [\(2004\)](#page-10-6). Remarkably, despite their complexity, these neural oscillations share characteristics with harmonic oscillators, as described in [Winfree](#page-12-2) [\(1980\)](#page-12-2). Building on this insight, we distill the core essence of cortical dynamics and aim to construct a machine learning model following the motion of harmonic oscillations. Finally, we note that our focus is on the theoretical and empirical aspects of the proposed oscillatory state-space model in this manuscript. However, our model can be further used in the context of computational neuroscience emulating characteristic phenomena of brain oscillations, such as frequency-varying oscillations, transient synchronization or desynchronization of discharges, entrainment, phase shifts, and resonance, while at the same time being able to learn non-trivial input-output relations.

### 2.2 COMPARISON WITH RELATED WORK

State-space models have seen continuous advancements since their introduction to modern sequence modeling in [Gu et al.](#page-10-0) [\(2021\)](#page-10-0). The original S4 model [\(Gu et al., 2021\)](#page-10-0), along with its adaptations [\(Gu et al., 2022a;](#page-10-4) [Nguyen et al., 2022;](#page-11-3) [Goel et al., 2022\)](#page-10-7), utilized FFT to solve linear recurrences. More recently, a simplified variant, S5 [\(Smith et al., 2023\)](#page-12-0), was introduced, which instead employs associative parallel scans to achieve similar computational speed. Another advancement to S4 have been Liquid Structural State-Space models [\(Hasani et al., 2022\)](#page-11-0) which exchanged the static state matrix with an input-dependent state transition module. While all aforementioned models rely on the HiPPO parameterization [\(Gu et al., 2020\)](#page-10-3), Linear Recurrent Units (LURs) [\(Orvieto et al., 2023\)](#page-11-1) demonstrated that even simpler parameterization yield state-of-the-art performance. Finally, selective state spaces have been introduced in [Gu & Dao](#page-10-2) [\(2023\)](#page-10-2) in order to further close the performance gap between state-space models and Transformers utilized in foundation models.

Oscillatory dynamics as a neural computational paradigm has been originally introduced via Coupled Oscillatory RNNs (CoRNNs) [\(Rusch & Mishra, 2021a\)](#page-12-3). In this work, it has been shown that recurrent models based on nonlinear oscillatory dynamics are able to learn long-range interactions by mitigating the exploding and vanishing gradients problem [\(Pascanu, 2013\)](#page-12-4). This approach was later refined for handling very long sequences in [Rusch & Mishra](#page-12-5) [\(2021b\)](#page-12-5), which utilized uncoupled nonlinear oscillators for fast sequence modeling by adopting a diagonal hidden-to-hidden weight matrix. Interestingly, this method resembles modern state-space models but distinguishes itself by employing nonlinear dynamics. Since then, this concept, generally termed as neural oscillators [\(Lanthaler et al., 2024\)](#page-11-4), has been extended to other areas of machine learning, e.g., Graph-Coupled Oscillator Networks (GraphCON) [\(Rusch et al., 2022\)](#page-12-6) for learning on graph-structured data, and RNNs with oscillatory dynamics combined with convolutions leading to locally coupled oscillatory RNNs [\(Keller & Welling, 2023\)](#page-11-5). Our proposed LinOSS model differs from all these methods by its explicit use of harmonic oscillators via a state-space model approach.

# 2.3 DISCRETIZATION

Our aim is to approximately solve the linear ODE system [\(2\)](#page-1-1) as fast as possible, while at the same time being able to guarantee stability over long time-scales. To this end, we suggest to leverage the following two time integration schemes.

Implicit time integration. We fix a timestep <sup>0</sup> <sup>&</sup>lt; <sup>∆</sup><sup>t</sup> ≤ <sup>1</sup> and define our proposed state-space model hidden states at time t<sup>n</sup> = n∆t as the following implicit discretization of the first order system [\(2\)](#page-1-1):

$$\begin{aligned}\mathbf{z}_n &= \mathbf{z}_{n-1} + \Delta t(-\mathbf{A}\mathbf{y}_n + \mathbf{B}\mathbf{u}_n), \\ \mathbf{y}_n &= \mathbf{y}_{n-1} + \Delta t\mathbf{z}_n.\end{aligned}$$

This can be written in matrix form by introducing x<sup>n</sup> = [zn, yn] ⊤,

$$\mathbf{M}\mathbf{x}_n = \mathbf{x}_{n-1} + \mathbf{F}_n,$$

with

$$\mathbf{M} = \begin{bmatrix} \mathbf{I} & \Delta t \mathbf{A} \\ -\Delta t \mathbf{I} & \mathbf{I} \end{bmatrix}, \quad \mathbf{F}_n = \begin{bmatrix} \Delta t \mathbf{B} \mathbf{u}_n \\ \mathbf{0} \end{bmatrix}$$

.

We can now solve our discretized oscillatory system by simply inverting matrix M and computing the induced recurrence. Note that inverting matrices typically requires O(m<sup>3</sup> ) operations (where m is the number of rows or columns of the matrix) using methods like Gauss-Jordan elimination, making it computationally expensive. However, by leveraging the Schur complement, we can obtain an explicit form of M−<sup>1</sup> that can be computed in O(m), thanks to the diagonal structure of <sup>A</sup>. More concretely,

$$M^{-1} = \begin{bmatrix} I - \Delta t^2 \mathbf{A}S & -\Delta t \mathbf{A}S \\ \Delta t S & S \end{bmatrix} = \begin{bmatrix} \mathbf{S} & -\Delta t \mathbf{A}S \\ \Delta t S & S \end{bmatrix}, \quad (3)$$

with the inverse of the Schur complement S = (I + ∆t <sup>2</sup>A) <sup>−</sup><sup>1</sup> which itself is a diagonal matrix and can thus be trivially inverted. We point out that among other choices, a straightforward condition that ensures <sup>S</sup> is well-defined is <sup>A</sup><sup>k</sup> ≥ <sup>0</sup> for all <sup>k</sup> = 1, . . . , m. The recurrence of the proposed model is then given as,

$$\mathbf{x}_n = \mathbf{M}^{\text{IM}} \mathbf{x}_{n-1} + \mathbf{F}_n^{\text{IM}}, \quad (4)$$

with MIM = M−<sup>1</sup> , and F IM <sup>n</sup> <sup>=</sup> <sup>M</sup>−<sup>1</sup>Fn. As we will show in the subsequent section, this discretization leads to a globally asymptotically stable discrete dynamical system.

Implicit-explicit time integration (IMEX). Another discretization yielding stable dynamics that, however, do not converge exponentially fast to a steady-state can be obtained by leveraging symplectic integrators. To this end, we fix again a timestep <sup>0</sup> <sup>&</sup>lt; <sup>∆</sup><sup>t</sup> ≤ <sup>1</sup> and define our proposed state-space model hidden states at time t<sup>n</sup> = n∆t as the following implicit-explicit (IMEX) discretization of the first order system [\(2\)](#page-1-1):

$$\begin{aligned}\mathbf{z}_n &= \mathbf{z}_{n-1} + \Delta t(-\mathbf{A}\mathbf{y}_{n-1} + \mathbf{B}\mathbf{u}_n), \\ \mathbf{y}_n &= \mathbf{y}_{n-1} + \Delta t\mathbf{z}_n.\end{aligned}\tag{5}$$

The only difference compared to the previous fully implicit discretization is the explicit treatment of the hidden state y (i.e., using yn−<sup>1</sup> instead of yn) in the first equation of [\(5\)](#page-3-0). As before, we can simplify this system in matrix form,

$$\mathbf{x}_n = \mathbf{M}^{\text{IMEX}} \mathbf{x}_{n-1} + \mathbf{F}_n^{\text{IMEX}}, \quad (6)$$

with

$$\mathbf{M}^{\text{IMEX}} = \begin{bmatrix} \mathbf{I} & -\Delta t \mathbf{A} \\ \Delta t \mathbf{I} & \mathbf{I} - \Delta t^2 \mathbf{A} \end{bmatrix}, \quad \mathbf{F}_n^{\text{IMEX}} = \begin{bmatrix} \Delta t \mathbf{B} u_n \\ \Delta t^2 \mathbf{B} u_n \end{bmatrix}.$$

Interestingly, ODE system [\(2\)](#page-1-1) represents a Hamiltonian system [\(Arnold, 1989\)](#page-10-8), with Hamiltonian,

$$\mathbf{H}(\mathbf{y}, \mathbf{z}, t) = \frac{1}{2} \sum_{k=1}^m \mathbf{A}_k \mathbf{y}_k^2 + \mathbf{z}_k^2 - 2 \left( \sum_{l=1}^p \mathbf{B}_{kl} \mathbf{u}(t)_l \right) \mathbf{y}_k. \quad (7)$$

The numerical approximation of this system using the previously described IMEX discretization is symplectic, i.e., it preserves a Hamiltonian close to the Hamiltonian of the continuous system. Thus, by the well-known Liouville's theorem [\(Sanz Serna & Calvo, 1994\)](#page-12-7), we know that the phase space volume of [\(2\)](#page-1-1) as well as of its symplectic approximation [\(6\)](#page-3-1) is preserved. This gives rise to invertible model architectures leading to memory efficient implementations of the backpropagation through time algorithm, similar as in [\(Rusch & Mishra, 2021b\)](#page-12-5). This denotes the most significant difference between the two different discretization schemes, i.e., the IMEX integration-based model is volume preserving, while IM integration-based model introduces dissipative terms. We will see in subsequent sections that both models have their own individual advantages depending on the underlying data. Finally, we note that stable higher-order time integration schemes (such as higherorder symplectic splitting schemes) can be used in this context as well. An example of leveraging the second-order symplectic velocity Verlet method can be found in Appendix Section [D.](#page-17-0)

### 2.4 FAST RECURRENCE VIA ASSOCIATIVE PARALLEL SCANS

Parallel (or associative) scans, first introduced in [Kogge & Stone](#page-11-6) [\(1973\)](#page-11-6) and reviewed in [Blelloch](#page-10-9) [\(1990\)](#page-10-9), offer a powerful method for drastically reducing the computational time of recurrent operations. These scans have previously been employed to enhance the training and inference speed of RNNs [\(Martin & Cundy, 2017;](#page-11-7) [Kaul, 2020\)](#page-11-8). This technique was later adapted for state-space models in [Smith et al.](#page-12-0) [\(2023\)](#page-12-0), becoming a crucial component in the development of many state-ofthe-art sequence models, including Linear Recurrent Units (LRUs) [\(Orvieto et al., 2023\)](#page-11-1) and Mamba models [\(Gu & Dao, 2023\)](#page-10-2).

A parallel scan operates on a sequence [x1, . . . , x<sup>N</sup> ] with a binary associative operation •, i.e., an operation satisfying (<sup>x</sup> • <sup>y</sup>) • <sup>z</sup> <sup>=</sup> <sup>x</sup> • (<sup>y</sup> • <sup>z</sup>) for instances x, y, and <sup>z</sup>, to return the sequence [x1, x<sup>1</sup> • <sup>x</sup>2, . . . , x<sup>1</sup> • <sup>x</sup><sup>2</sup> • · · · • <sup>x</sup><sup>N</sup> ]. Under certain assumptions, this operation can be performed in computational time proportional to ⌈log<sup>2</sup> (N)⌉. This is in stark contrast to the computational time of serial recurrence that is proportional to N. It is straightforward to check that the following operation is associative:

$$(a_1, a_2) \bullet (b_1, b_2) = (b_1 \bullet a_1, b_1 \bullet a_2 + b_2),$$

where •◦, •◦ are the matrix-matrix and matrix-vector products for matrices <sup>a</sup>1, <sup>b</sup><sup>1</sup> and vectors <sup>a</sup>2, <sup>b</sup>2. Note that both products can be computed in O(m) time leveraging <sup>2</sup> × <sup>2</sup> block matrices with only diagonal entries in each block, e.g., matrices MIM and MIMEX in [\(4\)](#page-3-2),[\(6\)](#page-3-1). Clearly, applying a parallel scan based on this associative operation on the input sequence [(M, F1),(M, F2), . . . ] yields an output sequence [(M, F1),(M<sup>2</sup> ,MF<sup>1</sup> + F2), . . . ] where the solution of the recurrent system x<sup>n</sup> = Mxn−<sup>1</sup> + F<sup>n</sup> (with initial value x<sup>0</sup> = 0) is stored in the second argument of the elements of this sequence. Thus, we can successfully apply parallel scans to both discretizations [\(4\)](#page-3-2)[\(6\)](#page-3-1) of our proposed system in order to significantly speed up computations. We refer to the application of this parallel scan to implicit formulations as implicit parallel scans. When applied to implicit-explicit formulations, we describe it as implicit-explicit parallel scans.

We term our proposed sequence model, which efficiently solves the underlying linear system of harmonic oscillators [\(2\)](#page-1-1) using fast parallel scans, as Linear Oscillatory State-Space (LinOSS) model. To differentiate between the two discretization methods, we refer to the model derived from implicit discretization as LinOSS-IM, and the model based on the implicit-explicit discretization as LinOSS-IMEX. A schematic drawing of our proposed state-space model can be seen in Fig. [1.](#page-1-2) Moreover, a full description of a multi-layer LinOSS model, including specific nonlinear building blocks, can be found in Appendix [A.](#page-13-0)

# 3 THEORETICAL INSIGHTS

### 3.1 STABILITY AND LEARNING LONG-RANGE INTERACTIONS

Hidden states of classical nonlinear RNNs are updated based on its previous hidden states pushed through a parametric function followed by a (usually) bounded nonlinear activation function such as tanh or sigmoid. This way, the hidden states are guaranteed to not blow up. However, this is not true for linear recurrences, where specific structures of the hidden-to-hidden weight matrix can lead to unstable dynamics. The following simple argument demonstrates that computing the eigenspectrum (i.e., set of eigenvalues) of the hidden-to-hidden weight matrix of a linear recurrent system suffices in order to analyse its stability properties. To this end, let us consider a general linear discrete dynamical system with external forcing given via the recurrence x<sup>n</sup> = Mxn−<sup>1</sup> + Bun, and initial value x<sup>0</sup> = 0. Assuming M is diagonalizable (if not, one can make a similar argument leveraging the Jordan normal form), i.e., there exists matrix S such that M = SΛS−<sup>1</sup> , where Λ is a diagonal matrix with the eigenvalues of M on its diagonal. The dynamics of the transformed hidden state x¯<sup>n</sup> = S <sup>−</sup><sup>1</sup>x<sup>n</sup> evolve according to x¯<sup>n</sup> = S <sup>−</sup><sup>1</sup>x<sup>n</sup> = S <sup>−</sup><sup>1</sup>Mxn−<sup>1</sup> + S <sup>−</sup><sup>1</sup>Bu<sup>n</sup> = Λx¯n−<sup>1</sup> + Bu¯ <sup>n</sup> with initial value x¯<sup>0</sup> = S <sup>−</sup><sup>1</sup>x0, where B¯ = S <sup>−</sup><sup>1</sup>B. Unrolling the dynamics (assuming x<sup>0</sup> = 0) yields,

$$\bar{\mathbf{x}}_1 = \bar{\mathbf{B}}\mathbf{u}_1, \quad \bar{\mathbf{x}}_2 = \Lambda \bar{\mathbf{B}}\mathbf{u}_1 + \bar{\mathbf{B}}\mathbf{u}_2, \quad \dots \quad \Rightarrow \mathbf{x}_n = \sum_{k=0}^{n-1} \Lambda^k \bar{\mathbf{B}}\mathbf{u}_{n-k}.$$

Clearly, if all eigenvalues Λ have magnitude less or equal than 1 the hidden states will not blow up. In addition to stability guarantees, eigenspectra with unit norm allow the model to learn long-range interactions [\(Arjovsky et al., 2016;](#page-10-10) [Gu et al., 2022b;](#page-11-9) [Orvieto et al., 2023\)](#page-11-1) by avoiding vanishing and exploding gradients [\(Pascanu, 2013\)](#page-12-4). Therefore, it is sufficient to analyse the eigenspectra of our proposed LinOSS models in order to understand their ability to generate stable dynamics and learn long-range interactions. To this end, we have the following propositions.

Proposition 3.1. *Let* <sup>M</sup>*IM* ∈ <sup>R</sup> <sup>m</sup>×<sup>m</sup> *be the hidden-to-hidden weight matrix of the implicit model LinOSS-IM* [\(4\)](#page-3-2)*. We assume that* <sup>A</sup>kk ≥ <sup>0</sup> *for all diagonal elements* <sup>k</sup> = 1, . . . , m *of* <sup>A</sup>*, and further that* ∆t > 0*. Then, the complex eigenvalues of* M*IM are given as,*

$$\lambda_j = \frac{1}{1 + \Delta t^2 \mathbf{A}_{kk}} + i(-1)^{\lceil \frac{j}{m} \rceil} \Delta t \frac{\sqrt{\mathbf{A}_{kk}}}{1 + \Delta t^2 \mathbf{A}_{kk}}, \quad \text{for all } j = 1, \dots, 2m,$$

*with* <sup>k</sup> <sup>=</sup> <sup>j</sup> *mod* <sup>m</sup>*. Moreover, the spectral radius* <sup>ρ</sup>(M*IM*) *is bounded by* <sup>1</sup>*, i.e.,* |<sup>λ</sup><sup>j</sup> | ≤ <sup>1</sup> *for all* j = 1, . . . , 2m*.*

*Proof.*

$$\begin{aligned} \det(\mathbf{M}^{\text{IM}} - \lambda \mathbf{I}) &= \begin{vmatrix} \mathbf{S} - \lambda \mathbf{I} & -\Delta t \mathbf{A} \mathbf{S} \\ \Delta t \mathbf{S} & \mathbf{S} - \lambda \mathbf{I} \end{vmatrix} = \begin{vmatrix} \mathbf{S} - \lambda \mathbf{I} & -\Delta t \mathbf{A} \mathbf{S} \\ \mathbf{0} & \mathbf{S} - \lambda \mathbf{I} + \Delta t^2 \mathbf{A} \mathbf{S}^2 (\mathbf{S} - \lambda \mathbf{I})^{-1} \end{vmatrix} \\ &= \prod_{k=1}^m (\mathbf{S}_{kk} - \lambda) \left( \mathbf{S}_{kk} - \lambda + \frac{\Delta t^2 \mathbf{A}_{kk} \mathbf{S}_{kk}^2}{\mathbf{S}_{kk} - \lambda} \right) = \prod_{k=1}^m [(\mathbf{S}_{kk} - \lambda)^2 + \Delta t^2 \mathbf{A}_{kk} \mathbf{S}_{kk}^2]. \end{aligned}$$

Setting k = j mod m, the eigenvalues of MIM are thus given as,

$$\lambda_j = \mathbf{S}_{kk} + i(-1)^{\lceil \frac{j}{m} \rceil} \Delta t \mathbf{S}_{kk} \sqrt{\mathbf{A}_{kk}}, \quad \text{for all } j = 1, \dots, 2m.$$

In particular, assuming <sup>A</sup>kk ≥ <sup>0</sup> for all <sup>k</sup> = 1, . . . , m, the magnitude of the eigenvalues <sup>λ</sup><sup>j</sup> are given as,

$$|\lambda_j|^2 = \mathbf{S}_{kk}^2 + \Delta t^2 \mathbf{S}_{kk}^2 \mathbf{A}_{kk} = \mathbf{S}_{kk}^2 (1 + \Delta t^2 \mathbf{A}_{kk}) = \mathbf{S}_{kk} \leq 1,$$
for all  $j = 1, \dots, 2m$ , with  $k = j \bmod m$ . □

This proof reveals important insights into our proposed LinOSS-IM model. First, the magnitude of the eigenvalues at initialization can be controlled by either specific initialization of A, or alternatively through a specific choice of the timestep parameter ∆t. Moreover, LinOSS-IM yields asymptotically stable dynamics for any choices of positive parameters A. This denotes a significant difference compared to previous first-order system, where the values of A (and thus its eigenvalues) have to be heavily constrained for the system to be stable. We argue that this flexibility in the parameterization of our model benefits the optimizer potentially leading to better performance in practice.

Remark 1. A straightforward adaptation of Proposition [3.1](#page-5-0) can also be derived for the implicitexplicit version of the model, LinOSS-IMEX [\(6\)](#page-3-1). The detailed proposition is given in Appendix Proposition [E.1.](#page-17-1) In particular, the analysis shows that all eigenvalues λ<sup>j</sup> of MIMEX in [\(6\)](#page-3-1) satisfy |<sup>λ</sup><sup>j</sup> | = 1. This result underscores the key distinction between the two models: LinOSS-IM incorporates dissipative terms, whereas LinOSS-IMEX denotes a conservative system. Following the argument at the beginning of Section [3.1,](#page-4-0) one can interpret the dissipative terms in LinOSS-IM as forgetting mechanisms, which are considered crucial for expressive modeling of long sequences. This makes LinOSS-IM a more flexible model version compared to LinOSS-IMEX. Finally, we note that explicit discretization schemes lead to exploding hidden states returning NaN output values in practice during training. For this reason, we focus solely on stable methods in this context.

Initialization and parameterization of weights. Proposition [3.1](#page-5-0) and Remark [1](#page-5-1) reveal that both LinOSS models exhibit stable dynamics as long as the diagonal weights A in [\(1\)](#page-1-0) are non-negative. This condition can easily be fulfilled via many different parameterizations of the diagonal matrix A. An obvious choice is to square the diagonal values, i.e., a parameterization A = Aˆ Aˆ , with diagonal matrix <sup>A</sup><sup>ˆ</sup> ∈ <sup>R</sup> <sup>m</sup>×<sup>m</sup>. Another straightforward approach is to apply the element-wise ReLU nonlinear activation function to A, i.e., A = ReLU(Aˆ ), with ReLU(x) = max(0, x) and Aˆ as before. The latter results in a LinOSS model where specific dimensions can be switched off completely by setting the corresponding weight in Aˆ to a negative value. Due to this flexibility, we decide to focus on the ReLU-parameterization of the diagonal weights A in this manuscript.

After discussing the parameterization of A, the next step is to determine an appropriate method for initializing its weights prior to training. Since we already know from Remark [1](#page-5-1) that all absolute eigenvalues of the hidden matrix MIMEX of LinOSS-IMEX are exactly one, the specific values of A are irrelevant for the model to learn long-range interactions. However, according to Proposition [3.1,](#page-5-0) A highly influences the eigenvalues of the hidden matrix MIM of LinOSS-IM [\(4\)](#page-3-2). As discussed in the previous section, high powers of the absolute eigenvalues of the hidden matrix are of particular interest when learning long-range interactions. To this end, we have the following Proposition concerning the expected powers of absolute eigenvalues for LinOSS-IM.

Proposition 3.2. *Let* {<sup>λ</sup>j} 2m <sup>j</sup>=1 *be the eigenspectrum of the hidden-to-hidden matrix* M*IM of the LinOSS-IM model* [\(4\)](#page-3-2)*. We further initialize* <sup>A</sup>kk ∼ U([0, A*max*]) *with* <sup>A</sup>*max* <sup>&</sup>gt; <sup>0</sup> *for all diagonal elements* k = 1, . . . , m *of* A *in* [\(2\)](#page-1-1)*. Then, the* N*-th moment of the magnitude of the eigenvalues are given as,*

$$\mathbb{E}(|\lambda_j|^N) = \frac{(\Delta t^2 A_{max} + 1)^{1 - \frac{N}{2}} - 1}{\Delta t^2 A_{max} (1 - \frac{N}{2})}, \quad (8)$$

*for all* j = 1, . . . , 2m*, with* k = j *mod* m*.*

The proof, detailed in Appendix Section [E.2,](#page-18-0) is a straight-forward application of the law of the unconscious statistician. Proposition [3.2](#page-6-0) demonstrates that while the expectation of |<sup>λ</sup><sup>i</sup> | N might be much smaller than 1, it is still sufficiently large for practical use-cases. For instance, even considering ∆t = Amax = 1 and an extreme sequence length of N = 100k still yields <sup>E</sup>(|<sup>λ</sup><sup>j</sup> | <sup>N</sup> ) = 1/<sup>49999</sup> ≈ <sup>2</sup> × <sup>10</sup>−<sup>5</sup> . Based on this and the fact that the values of A do not affect the eigenvalues for LinOSS-IMEX, we decide to initialize <sup>A</sup> according to <sup>A</sup>kk ∼ U([0, 1]) for both LinOSS models, while setting ∆t = 1.

### 3.2 UNIVERSALITY OF LINOSS WITHIN CONTINUOUS AND CAUSAL OPERATORS

While trainability is an important aspect of learning long-range interactions, it does not demonstrate why LinOSS is able to express complex mappings between general (i.e., not necessarily oscillatory) input and output sequences. Therefore, in this section we analyze the approximation power of our proposed LinOSS model. Following the recent work of [Lanthaler et al.](#page-11-4) [\(2024\)](#page-11-4) we show that LinOSS is universal within the class of continuous and causal operators between time-series. To this end, we consider a full LinOSS block,

$$\mathbf{z}(t) = \mathbf{W}\sigma(\tilde{\mathbf{W}}\mathbf{y}(t) + \tilde{\mathbf{b}}), \quad (9)$$

with weights <sup>W</sup> ∈ <sup>R</sup> q×m˜ , <sup>W</sup>˜ ∈ <sup>R</sup> <sup>m</sup>˜ <sup>×</sup><sup>m</sup>, bias <sup>b</sup>˜ ∈ <sup>R</sup> m˜ , element-wise nonlinear activation function σ (e.g., tanh or ReLU), and solution y(t) of the LinOSS differential equations [\(2\)](#page-1-1).

Based on this, we are now interested in approximating operators Φ : C0([0, T]; <sup>R</sup> p ) → C0([0, T]; <sup>R</sup> q ) with the full LinOSS model [\(9\)](#page-6-1), where

$$C_0([0, T]; \mathbb{R}^p) := \{\mathbf{u} : [0, T] \rightarrow \mathbb{R}^p \mid t \mapsto \mathbf{u}(t) \text{ is continuous and } \mathbf{u}(0) = \mathbf{0}\},$$

i.e., operators between continuous time-varying functions with values in R p . As pointed out in [Lanthaler et al.](#page-11-4) [\(2024\)](#page-11-4), the condition u(0) = 0 is not restrictive and can directly be generalized to the case of any initial condition <sup>u</sup>(0) = <sup>u</sup><sup>0</sup> ∈ <sup>R</sup> p simply by introducing an arbitrarily small warmup phase [−<sup>t</sup>0, 0] with <sup>t</sup><sup>0</sup> <sup>&</sup>gt; <sup>0</sup> of the oscillators to synchronize with the input signal <sup>u</sup>. In addition to these function spaces, we pose the following conditions on the underlying operator Φ,

- 1. <sup>Φ</sup> is *causal*, i.e., for any <sup>t</sup> ∈ [0, T], if <sup>u</sup>, <sup>B</sup> ∈ <sup>C</sup>0([0, T]; <sup>R</sup> p ) are two input signals, such that <sup>u</sup>|[0,t] ≡ <sup>B</sup>|[0,t] , then Φ(u)(t) = Φ(B)(t).
- 2. Φ is *continuous* as an operator

$$\Phi : (C_0([0, T]; \mathbb{R}^p), \|\cdot\|_{L^\infty}) \rightarrow (C_0([0, T]; \mathbb{R}^q), \|\cdot\|_{L^\infty}),$$

with respect to the L∞-norm on the input-/output-signals.

Theorem 3.3. *Let* Φ : C0([0, T]; <sup>R</sup> p ) → <sup>C</sup>0([0, T]; <sup>R</sup> q ) *be a causal and continuous operator. Let* <sup>K</sup> ⊂ <sup>C</sup>0([0, T]; <sup>R</sup> p ) *be compact. Then for any* ϵ > 0*, there exist hyperparameters* m*,* m˜ *, diagonal weight matrix* <sup>A</sup> ∈ <sup>R</sup> <sup>m</sup>×<sup>m</sup>*, weights* <sup>B</sup> ∈ <sup>R</sup> m×d *,* <sup>W</sup>˜ ∈ <sup>R</sup> <sup>m</sup>˜ <sup>×</sup>m*,* <sup>W</sup> ∈ <sup>R</sup> <sup>q</sup>×m˜ *and bias vectors* <sup>b</sup> ∈ <sup>R</sup> <sup>m</sup>*,* <sup>b</sup>˜ ∈ <sup>R</sup> m˜ *, such that the output* <sup>z</sup> : [0, T] → <sup>R</sup> <sup>q</sup> *of the LinOSS block* [\(9\)](#page-6-1) *satisfies,*

$$\sup_{t \in [0, T]} |\Phi(\mathbf{u})(t) - \mathbf{z}(t)| \leq \epsilon, \quad \forall \mathbf{u} \in K.$$

The proof can be found in Appendix Section [E.3.](#page-18-1) The main idea of the proof is to encode the infinite-dimensional operator Φ with a finite-dimensional operator that makes use of the structure of the LinOSS ODE system [\(1\)](#page-1-0), and that can further be expressed by a (finite-dimensional) function. This theorem rigorously shows that LinOSS can approximate any causal and continuous operator between continuous time-varying functions with values in R p to any desired accuracy.

# 4 EMPIRICAL RESULTS

In this section, we empirically test the performance of our proposed LinOSS models on a variety of challenging real-world sequential datasets, ranging from scientific datasets in genomics to practical applications in medicine. We ensure thereby a fair comparison to other state-of-the-art sequence models such as Mamba, LRU, and S5.

### 4.1 LEARNING LONG-RANGE INTERACTIONS

In the first part of the experiments, we focus on a recently proposed long-range sequential benchmark introduced in [Walker et al.](#page-12-8) [\(2024\)](#page-12-8). This benchmark focuses on six datasets from the University of East Anglia (UEA) Multivariate Time Series Classification Archive (UEA-MTSCA) [\(Bagnall et al.,](#page-10-11) [2018\)](#page-10-11), selecting those with the longest sequences for increased difficulty. The sequence lengths range thereby from 400 to almost 18k. We compare our proposed LinOSS models to recent state-ofthe-art sequence models, including state-space models such as Mamba and LRU. Table [1](#page-7-0) shows the test accuracies averaged over five random model initialization and dataset splits of all six datasets for LinOSS as well as competing methods. We note that all other results are taken from [Walker et al.](#page-12-8) [\(2024\)](#page-12-8). Moreover, we highlight that we exactly follow the training procedure described in [Walker](#page-12-8) [et al.](#page-12-8) [\(2024\)](#page-12-8) in order to ensure a fair comparison against competing models. More concretely, we use the same pre-selected random seeds for splitting the datasets into training, validation, and testing parts (using 70/15/15 splits), as well as tune our model hyperparameters only on the same pre-described grid. In fact, since LinOSS does not possess any model specific hyperparameters – unlike competing state-space models such as LRU or S5 – our search grid is lower-dimensional compared to the other models considered. As a result, the hyperparameter tuning process involves significantly fewer model instances.

Table 1: Test accuracies averaged over 5 training runs on UEA time-series classification datasets. All models are trained based on the same hyper-parameter tuning protocol in order to ensure fair comparability. The dataset names are abbreviations of the following UEA time-series datasets: Eigen-Worms (Worms), SelfRegulationSCP1 (SCP1), SelfRegulationSCP2 (SCP2), EthanolConcentration (Ethanol), Heartbeat, MotorImagery (Motor). The three best performing methods are highlighted in red<sup>1</sup> (First), blue<sup>2</sup> (Second), and violet<sup>3</sup> (Third).

| SCP1                              |    |       | SCP2 |   | Motor Avg                               |
|-----------------------------------|----|-------|------|---|-----------------------------------------|
| Seq. length 17,984 896            |    | 1,152 |      |   | 1,751 405 3,000                         |
| #Classes 5 2                      |    |       | 2    |   | 4 2 2                                   |
| NRDE 83 9 ± 7 3 80 9 ± 2 5        | 53 | 7     | ±    | 6 | 9                                       |
|                                   |    |       |      |   | 3 25 3 ± 1 8 72 9 ± 4 8 47 0 ± 5 7 60 6 |
| NCDE 75 0 ± 3 9 79 8 ± 5 6        | 53 | 0     | ±    | 2 | 8 29 9 ± 6 5                            |
|                                   |    |       |      |   | 2 73 9 ± 2 6 49 5 ± 2 8 60 2            |
| Log-NCDE 85 6 ± 5 1               |    |       |      |   |                                         |
| 3 83 1 ± 2 8                      | 53 | 7     | ±    | 4 | 1                                       |
|                                   |    |       |      |   | 3 34 4 ± 6 4                            |
|                                   |    |       |      |   | 1 75 2 ± 4 6 53 7 ± 5 3                 |
|                                   |    |       |      |   | 3 64 3                                  |
| LRU 87 8 ± 2 8                    |    |       |      |   |                                         |
| 2 82 6 ± 3 4                      | 51 | 2     | ±    | 3 | 6 21 5 ± 2 1 78 4 ± 6 7                 |
|                                   |    |       |      |   | 1 48 4 ± 5 0 61 7                       |
| S5 81 1 ± 3 7 89 9 ± 4 6          |    |       |      |   |                                         |
| 1                                 | 50 | 5     | ±    | 2 | 6 24 1 ± 4 3 77 7 ± 5 5                 |
|                                   |    |       |      |   | 2 47 7 ± 5 5 61 8                       |
| S6 85 0 ± 16 1 82 8 ± 2 7         | 49 | 9     | ±    | 9 | 4 26 4 ± 6 4 76 5 ± 8 3                 |
|                                   |    |       |      |   | 3 51 3 ± 4 7 62 0                       |
| Mamba 70 9 ± 15 8 80 7 ± 1 4      | 48 | 2     | ±    | 3 | 9 27 9 ± 4 5                            |
|                                   |    |       |      |   | 3 76 2 ± 3 8 47 7 ± 4 5 58 6            |
| LinOSS-IMEX 80 0 ± 2 7 87 5 ± 4 0 |    |       |      |   |                                         |
| 3                                 | 58 | 9     | ±    | 8 | 1                                       |
|                                   |    |       |      |   | 1 29 9 ± 1 0                            |
|                                   |    |       |      |   | 2 75 5 ± 4 3 57 9 ± 5 3                 |
|                                   |    |       |      |   | 2 65 0                                  |
| LinOSS-IM 95 0 ± 4 4              |    |       |      |   |                                         |
| 1 87 8 ± 2 6                      |    |       |      |   |                                         |
| 2                                 | 58 | 2     | ±    | 6 | 9                                       |
|                                   |    |       |      |   | 2 29 9 ± 0 6                            |
|                                   |    |       |      |   | 2 75 8 ± 3 7 60 0 ± 7 5                 |
|                                   |    |       |      |   | 1 67 8                                  |

We can see in Table [1](#page-7-0) that on average both LinOSS models outperform any other model we consider here by reaching an average (over all six datasets) accuracy of 65.0% for LinOSS-IMEX and 67.8% for LinOSS-IM. In particular, the average accuracy of LinOSS-IM is significantly higher than the two next best models, i.e., Log-NCDE reaching an average accuracy of 64.3%, and S6 reaching an accuracy of 62.0% on average. It is particularly noteworthy that LinOSS-IM yields state-of-the-art results on the two datasets with the longest sequences, namely EigenWorms and MotorImagery.

### 4.2 VERY LONG-RANGE INTERACTIONS

In this experiment, we test the performance of LinOSS in the case of very long-range interactions. To this end, we consider the PPG-DaLiA dataset, a multivariate time series regression dataset designed for heart rate prediction using data collected from a wrist-worn device [\(Reiss et al., 2019\)](#page-12-9). It includes recordings from fifteen individuals, each with approximately 150 minutes of data sampled at a maximum rate of 128 Hz. The dataset consists of six channels: blood volume pulse, electrodermal activity, body temperature, and three-axis acceleration. We follow [Walker et al.](#page-12-8) [\(2024\)](#page-12-8) and divide the data into training, validation, and test sets with a 70/15/15 split for each individual. After splitting the data, a sliding window of length 49920 and step size 4992 is applied. As in previous experiments, we apply the exact same hyperparameter tuning protocol to each model we consider here to ensure fair comparison. The test mean-squared error (MSE) of both LinOSS models as well as other competing models are shown in Table [2.](#page-8-0) We can see that both LinOSS models significantly outperform all other models. In particular, LinOSS-IM outperforms Mamba and LRU by nearly a factor of 2. This highlights the effectiveness of our proposed LinOSS models on sequential data with extreme length.

Table 2: Average test mean-squared error over 5 training runs on the PPG-DaLiA dataset. All models are trained following the same hyper-parameter tuning protocol in order to ensure fair comparability. The three best performing methods are highlighted in red<sup>1</sup> (First), blue<sup>2</sup> (Second), and violet<sup>3</sup> (Third).

| Model                          | MSE $\times 10^{-2}$                           |
|--------------------------------|------------------------------------------------|
| NRDE (Morrill et al., 2021)    | $9.90 \pm 0.97$                                |
| NCDE (Kidger et al., 2020)     | $13.54 \pm 0.69$                               |
| Log-NCDE (Walker et al., 2024) | <b><math>5.56 \pm 0.49</math></b> <sup>2</sup> |
| LRU (Orvieto et al., 2023)     | $12.17 \pm 0.49$                               |
| S5 (Smith et al., 2023)        | $12.63 \pm 1.25$                               |
| S6 (Gu & Dao, 2023)            | $12.88 \pm 2.05$                               |
| Mamba (Gu & Dao, 2023)         | $10.65 \pm 2.20$                               |
| <b>LinOSS-IMEX</b>             | <b><math>7.5 \pm 0.42</math></b> <sup>2</sup>  |
| <b>LinOSS-IM</b>               | <b><math>6.4 \pm 0.23</math></b> <sup>1</sup>  |

#### 4.3 LONG-HORIZON FORECASTING

Inspired by [Gu et al.](#page-10-0) [\(2021\)](#page-10-0), we test our proposed LinOSS on its ability to serve as a general sequence-to-sequence model, even with weak inductive bias. To this end, we focus on time-series forecasting which typically requires specialized domain-specific preprocessing and architectures. We do, however, not alter our LinOSS model nor incorporate any inductive biases. Thus, we simply follow [Gu et al.](#page-10-0) [\(2021\)](#page-10-0) by setting up LinOSS as a general sequence-to-sequence model that treats forecasting as a masked sequence-to-sequence transformation. We consider a weather prediction task introduced in [Zhou et al.](#page-12-10) [\(2021\)](#page-12-10). In this task, several climate variables are predicted into the future based on local climatological data. Here, we focus on the hardest task in [Zhou et al.](#page-12-10) [\(2021\)](#page-12-10) of predicting the future 720 timesteps (hours) based on the past 720 timesteps. Table [3](#page-9-0) shows the mean absolute error for both LinOSS models as well as other competing models. We can see that both LinOSS models outperform Transformers-based baselines as well as the other state-space models.

### 4.4 ADDITIONAL EXPERIMENTS AND ABLATIONS

Our proposed LinOSS model is the result of several design choices, such as the state matrix parameterization, state matrix initialization, and the numerical value of the discretization timestep ∆t. In this section, we empirically analyze how these choices affect the performance of LinOSS by providing ablations and sensitivity studies.

We start by evaluating the performance of LinOSS under different parameterizations and initializations of the state matrix A. Specifically, we examine LinOSS-IM within the experimental framework described in Section [4.1.](#page-7-1) For this analysis, we parameterize A using one of the two approaches proposed in Section [3.1:](#page-4-0) <sup>A</sup> <sup>=</sup> <sup>A</sup>˜ <sup>A</sup>˜ or <sup>A</sup> = ReLU(A˜ ), where <sup>A</sup>˜ ∈ <sup>R</sup> <sup>m</sup>×<sup>m</sup> is a diagonal matrix. The

Table 3: Mean absolute error on the weather dataset predicting the future 720 time steps based on the past 720 time steps. The three best performing methods are highlighted in red<sup>1</sup> (First), blue<sup>2</sup> (Second), and violet<sup>3</sup> (Third).

| Model                          | Mean Absolute Error      |
|--------------------------------|--------------------------|
| Informer (Zhou et al., 2021)   | 0.731                    |
| LogTrans (Li et al., 2019)     | 0.773                    |
| Reformer (Kitaev et al., 2020) | 1.575                    |
| LSTMa (Bahdanau et al., 2016)  | 1.109                    |
| LSTnet (Lai et al., 2018)      | 0.757                    |
| S4 (Gu et al., 2021)           | <b>0.578<sup>3</sup></b> |
| <b>LinOSS-IMEX</b>             | <b>0.508<sup>1</sup></b> |
| <b>LinOSS-IM</b>               | <b>0.528<sup>2</sup></b> |

results, presented in Appendix Table [6](#page-15-0) show that the ReLU parameterization leads to slightly better performance on average over all six datasets. However, the squared parameterization yields better performance on three out of six datasets. Thus, including different state matrix parameterization choices in the hyperparameter optimization can help achieve improved performance. We further test the performance of LinOSS-IM using a standard normal random initialization of the state matrix instead of the uniform initialization. The results of the standard normal initialization of the state matrix are shown in Appendix Table [6.](#page-15-0) We can see that this initialization yields similar performance to a uniform initialization on almost all considered datasets, except the EigenWorms dataset, where it obtains a lower mean test accuracy and a much higher standard deviation. This indicates that the performance of LinOSS-IM with a standard normal initialization for the state matrix is highly sensitive to the random seed used during initialization. This sensitivity arises because normal distributions do not have bounded support, allowing for the possibility of large matrix entries. These large entries result in small eigenvalues, which in turn lead to vanishing gradients.

Another natural question in this context concerns the sensitivity of performance to variations in the timestep ∆t used in the underlying discretization scheme. To investigate this, we train several LinOSS-IM models using different values of ∆t spanning three orders of magnitude, i.e., ranging from 10−<sup>3</sup> to 1. The average test error, along with the standard deviation, is plotted for three different datasets in Appendix [C.3.](#page-16-0) We can see that while the choice of ∆t does influence performance, the variations are not substantial.

Finally, we empirically analyze the roles of dissipation and conservation in LinOSS-IM and LinOSS-IMEX. As rigorously demonstrated in Section [3.1,](#page-4-0) LinOSS-IM introduces dissipative terms, whereas LinOSS-IMEX denotes a conservative system. While our earlier experiments examined the differences between these models using real-world data, we now focus on their performance for predicting energy-conserving dynamical systems to highlight their contrasting behaviors. To this end, we simulate the energy-conserving simple harmonic motion with various initial positions and velocities. The test error over time for both models is plotted in Appendix Fig. [2.](#page-16-1) The results show that the error in LinOSS-IM grows over time, whereas the error in LinOSS-IMEX remains constant. Notably, by the end of the time interval, LinOSS-IMEX outperforms LinOSS-IM by a factor of more than 8. This stark contrast underscores the fundamental difference between the two models: LinOSS-IM introduces dissipative terms, making it more effective for dissipative systems, while LinOSS-IMEX, being fully conservative, excels in energy-conserving systems.

# 5 CONCLUSION

In this paper, we introduce LinOSS, a state-space model based on harmonic oscillators. We rigorously show that LinOSS produces stable dynamics and is able to learn long-range interactions only requiring a nonnegative diagonal state matrix. In addition, we connect the underlying ODE system of the LinOSS model to Hamiltonian dynamics, which, discretized using symplectic integrators, perfectly preserves its symmetry of time reversibility. Moreover, we show that LinOSS is a universal approximator of continuous and causal operators between time-series. Together, these properties enable efficient modeling of long-range interactions, while ensuring stable and accurate long-horizon forecasting. Finally, we demonstrate that LinOSS outperforms state-of-the-art state-space models, such as Mamba, LRU, and S5.

## ACKNOWLEDGMENTS

The authors would like to thank Dr. Alaa Maalouf (MIT) and Dr. T. Anderson Keller (Harvard University) for their insightful feedback and constructive suggestions on an earlier version of the manuscript. This work was supported in part by the Postdoc.Mobility grant P500PT-217915 from the Swiss National Science Foundation, the Schmidt AI2050 program (grant G-22-63172), and the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

# REFERENCES


[1] Martin Arjovsky, Amar Shah, and Yoshua Bengio. Unitary evolution recurrent neural networks. In *International conference on machine learning*, pp. 1120–1128. PMLR, 2016.

[2] V. I. Arnold. *Mathematical methods of classical mechanics*. Springer Verlag, New York, 1989. Anthony Bagnall, Hoang Anh Dau, Jason Lines, Michael Flynn, James Large, Aaron Bostrom, Paul Southam, and Eamonn Keogh. The uea multivariate time series classification archive, 2018. *arXiv preprint arXiv:1811.00075*, 2018. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate, 2016. Guy E Blelloch. Prefix sums and their applications. 1990. James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL [http:](http://github.com/jax-ml/jax) [//github.com/jax-ml/jax](http://github.com/jax-ml/jax). Gyorgy Buzsaki and Andreas Draguhn. Neuronal oscillations in cortical networks. *science*, 304 (5679):1926–1929, 2004. Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. *arXiv preprint arXiv:1412.3555*, 2014. Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks. In *International conference on machine learning*, pp. 933–941. PMLR, 2017. Jacob Devlin. Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*, 2018. Karan Goel, Albert Gu, Chris Donahue, and Christopher Re. It's raw! audio generation with state- ´ space models. In *International Conference on Machine Learning*, pp. 7616–7633. PMLR, 2022. Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*, 2023. Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Re. Hippo: Recurrent memory ´ with optimal polynomial projections. *Advances in neural information processing systems*, 33: 1474–1487, 2020. Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured ´ state spaces. *arXiv preprint arXiv:2111.00396*, 2021. Albert Gu, Karan Goel, Ankit Gupta, and Christopher Re. On the parameterization and initialization ´ of diagonal state space models. *Advances in Neural Information Processing Systems*, 35:35971– 35983, 2022a.

[3] Albert Gu, Isys Johnson, Aman Timalsina, Atri Rudra, and Christopher Re. How to train ´ your hippo: State space models with generalized orthogonal basis projections. *arXiv preprint arXiv:2206.12037*, 2022b.

[4] J. Guckenheimer and P. Holmes. *Nonlinear oscillations, dynamical systems, and bifurcations of vector fields*. Springer Verlag, New York, 1990. Ramin Hasani, Mathias Lechner, Tsun-Hsuan Wang, Makram Chahine, Alexander Amini, and Daniela Rus. Liquid structural state-space models. *arXiv preprint arXiv:2209.12951*, 2022. Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). *arXiv preprint arXiv:1606.08415*, 2023. S Hochreiter. Long short-term memory. *Neural Computation MIT-Press*, 1997. Zheyuan Hu, Nazanin Ahmadi Daryakenari, Qianli Shen, Kenji Kawaguchi, and George Em Karniadakis. State-space models are accurate and efficient neural operators for dynamical systems. *arXiv preprint arXiv:2409.03231*, 2024. Shiva Kaul. Linear dynamical systems as a core computational primitive. *Advances in Neural Information Processing Systems*, 33:16808–16820, 2020. T Anderson Keller and Max Welling. Neural wave machines: learning spatiotemporally structured representations with locally coupled oscillatory recurrent neural networks. In *International Conference on Machine Learning*, pp. 16168–16189. PMLR, 2023. Patrick Kidger, James Morrill, James Foster, and Terry Lyons. Neural controlled differential equations for irregular time series. *Advances in neural information processing systems*, 33:6696–6707, 2020. Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. *arXiv preprint arXiv:2001.04451*, 2020. Peter M Kogge and Harold S Stone. A parallel algorithm for the efficient solution of a general class of recurrence equations. *IEEE transactions on computers*, 100(8):786–793, 1973. Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. In *The 41st international ACM SIGIR conference on research & development in information retrieval*, pp. 95–104, 2018. Samuel Lanthaler, T Konstantin Rusch, and Siddhartha Mishra. Neural oscillators are universal. *Advances in Neural Information Processing Systems*, 36, 2024. Shiyang Li, Xiaoyong Jin, Yao Xuan, Xiyou Zhou, Wenhu Chen, Yu-Xiang Wang, and Xifeng Yan. Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. *Advances in neural information processing systems*, 32, 2019. Eric Martin and Chris Cundy. Parallelizing linear recurrent neural nets over sequence length. *arXiv preprint arXiv:1709.04057*, 2017. Jonas Mikhaeil, Zahra Monfared, and Daniel Durstewitz. On the difficulty of learning chaotic dynamics with rnns. *Advances in Neural Information Processing Systems*, 35:11297–11312, 2022. James Morrill, Cristopher Salvi, Patrick Kidger, and James Foster. Neural rough differential equations for long time series. In *International Conference on Machine Learning*, pp. 7829–7838. PMLR, 2021. Eric Nguyen, Karan Goel, Albert Gu, Gordon Downs, Preey Shah, Tri Dao, Stephen Baccus, and Christopher Re. S4nd: Modeling images and videos as multidimensional signals with state spaces. ´ *Advances in neural information processing systems*, 35:2846–2861, 2022. Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, and Soham De. Resurrecting recurrent neural networks for long sequences. In *International Conference on Machine Learning*, pp. 26670–26698. PMLR, 2023.

[5] R Pascanu. On the difficulty of training recurrent neural networks. *arXiv preprint arXiv:1211.5063*, 2013. Attila Reiss, Ina Indlekofer, Philip Schmidt, and Kristof Van Laerhoven. Deep ppg: Large-scale heart rate estimation with convolutional neural networks. *Sensors*, 19(14):3079, 2019.

[6] T. Konstantin Rusch and Siddhartha Mishra. Coupled oscillatory recurrent neural network (cornn): An accurate and (gradient) stable architecture for learning long time dependencies. In *International Conference on Learning Representations*, 2021a. T Konstantin Rusch and Siddhartha Mishra. Unicornn: A recurrent model for learning very long time dependencies. In *International Conference on Machine Learning*, pp. 9168–9178. PMLR, 2021b. T Konstantin Rusch, Ben Chamberlain, James Rowbottom, Siddhartha Mishra, and Michael Bronstein. Graph-coupled oscillator networks. In *International Conference on Machine Learning*, pp. 18888–18909. PMLR, 2022. J.M. Sanz Serna and M.P. Calvo. *Numerical Hamiltonian problems*. Chapman and Hall, London, 1994. Jimmy T.H. Smith, Andrew Warrington, and Scott Linderman. Simplified state space layers for sequence modeling. In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=Ai8Hw3AXqks>. A Vaswani. Attention is all you need. *Advances in Neural Information Processing Systems*, 2017. Benjamin Walker, Andrew D. McLeod, Tiexin Qin, Yichuan Cheng, Haoliang Li, and Terry Lyons. Log neural controlled differential equations: The lie brackets make a difference. *International Conference on Machine Learning*, 2024. Arthur T Winfree. *The geometry of biological time*, volume 2. Springer, 1980. Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In *Proceedings of the AAAI conference on artificial intelligence*, volume 35, pp. 11106–11115, 2021.
#### Supplementary Material for: Oscillatory State-Space Models

### A FULL ARCHITECTURE DETAILS

In this section, we outline the detailed architecture of a full multi-layer LinOSS model. To this end, the full LinOSS model starts by encoding an input sequence u = [u1, u2, . . . , u<sup>N</sup> ] with <sup>u</sup><sup>i</sup> ∈ <sup>R</sup> q for all i = 1, . . . , N via an affine transformation. After that, several blocks are applied consisting of a LinOSS layer (i.e., solving [\(1\)](#page-1-0) with either IM or IMEX associative parallel scans) directly followed by a nonlinear layer using the Gaussian error linear unit activation function (GELU) [\(Hendrycks & Gimpel, 2023\)](#page-11-15), the Gated Linear Unit (GLU) [\(Dauphin et al., 2017\)](#page-10-5), i.e., GLU(x) = sigmoid(W1x) ◦<sup>W</sup>2x, where <sup>W</sup>1,<sup>2</sup> are learnable weight matrices, and a skip connection. Finally, the output of the final LinOSS block gets decoded by an affine transformation. The full LinOSS model is further presented in Algorithm [1.](#page-13-1) Note that weight matrices and bias vectors are applied parallel in time whenever possible (i.e., outside the recurrence). We are thus omitting subscript i in Algorithm [1.](#page-13-1)

Algorithm 1 Full LinOSS model

Input: Input sequence u Output: L-block LinOSS output sequence o

u

<sup>0</sup> ← <sup>W</sup>enc<sup>u</sup> <sup>+</sup> <sup>b</sup>enc <sup>▷</sup> Encode input sequence

for l = 1, . . . , L do

y

<sup>l</sup> ← solution of ODE in [\(1\)](#page-1-0) with input <sup>u</sup>

<sup>l</sup>−<sup>1</sup> via parallel scan

x

<sup>l</sup> ← Cy<sup>l</sup> <sup>+</sup> Du<sup>l</sup>−<sup>1</sup> <sup>▷</sup> Linear readout in [\(1\)](#page-1-0)

x

<sup>l</sup> ← GELU(<sup>x</sup>

l )

u

<sup>l</sup> ← GLU(<sup>x</sup>

l ) + u l−1

end for <sup>o</sup> ← <sup>W</sup>dec<sup>y</sup>

<sup>L</sup> + bdec ▷ Decode final LinOSS block output

#### A.1 SEQUENCE-TO-SEQUENCE LINOSS ARCHITECTURE FOR TIME-SERIES FORECASTING

While in sequence regression and sequence classification tasks we simply take the final output of the full LinOSS model at final time T as the model prediction, we have to slightly adapt our architecture to handle time-series forecasting problems. To this end, we follow common practice for state-space models suggested in [Gu et al.](#page-10-0) [\(2021\)](#page-10-0) and generate train, validation, and test sequences of length L<sup>1</sup> + L2, where L<sup>1</sup> is the number of the past sequence entries used for forecasting and L<sup>2</sup> are the number of future steps we aim to predict. Note that for the input sequences, we simply mask out the last L<sup>2</sup> entries of the sequence (i.e., with zero entries). Moreover, for the LinOSS output sequence, we only use the final L<sup>2</sup> entries for the future predictions.

# B TRAINING DETAILS

The code to run the experiments is implemented using the JAX auto-differentiation framework [\(Bradbury et al., 2018\)](#page-10-13). All experiments were conducted on Nvidia Tesla V100 GPUs and Nvidia RTX 4090 GPUs, with the exception of the PPG experiment, which was run on Nvidia Tesla A100 GPUs due to higher memory demands.

### B.1 HYPERPARAMETERS

The hyperparameters of the models were optimized with the same grid search approach from [Walker et al.](#page-12-8) [\(2024\)](#page-12-8) for the six datasets in Section [4.1](#page-7-1) and the PPG dataset of Section [4.2](#page-8-1) to ensure perfect comparability with competing methods, i.e., using the grid: learning rate = {0.00001, <sup>0</sup>.0001, <sup>0</sup>.001}, number of layers <sup>=</sup> {2, <sup>4</sup>, <sup>6</sup>}, number of hidden neurons <sup>=</sup> {16, <sup>64</sup>, <sup>128</sup>}, state-space dimension <sup>=</sup> {16, <sup>64</sup>, <sup>256</sup>}, include time dimension {True, False}. Note that for the weather dataset, we performed a random search instead of grid search using the same hyperparameter bounds as before, except that we increased the maximum number of LinOSS blocks from 6 to 8.

The hyperparameters yielding the best performance can be seen in Table [4](#page-14-0) for both LinOSS-IM and LinOSS-IMEX for each experiment in the main paper.

Table 4: Best hyperparameters

|           | Model |   | lr    | hidden dim | state dim | #blocks | include time |
|-----------|-------|---|-------|------------|-----------|---------|--------------|
| Worms     | IM    | 0 | 001   | 128        | 64        | 2       | True         |
|           | IMEX  | 0 | 0001  | 64         | 16        | 2       | False        |
| SCP1      | IM    | 0 | 0001  | 128        | 256       | 6       | True         |
|           | IMEX  | 0 | 0001  | 64         | 256       | 6       | False        |
| SCP2      | IM    | 0 | 0001  | 128        | 64        | 6       | False        |
|           | IMEX  | 0 | 00001 | 64         | 256       | 6       | True         |
| Ethanol   | IM    | 0 | 00001 | 16         | 16        | 4       | False        |
|           | IMEX  | 0 | 00001 | 16         | 256       | 4       | False        |
| Heartbeat | IM    | 0 | 001   | 16         | 16        | 6       | True         |
|           | IMEX  | 0 | 00001 | 64         | 16        | 2       | True         |
| Motor     | IM    | 0 | 00001 | 128        | 16        | 2       | False        |
|           | IMEX  | 0 | 0001  | 16         | 256       | 6       | True         |
| PPG       | IM    | 0 | 001   | 16         | 64        | 6       | True         |
|           | IMEX  | 0 | 0001  | 64         | 16        | 2       | True         |
| Weather   | IM    | 0 | 0006  | 64         | 32        | 8       | True         |
|           | IMEX  | 0 | 0007  | 256        | 128       | 5       | False        |

### B.2 MEMORY REQUIREMENTS, RUNTIMES, AND NUMBER OF PARAMETERS

In this section, we present the number of parameters, GPU memory usage (in MB), and run time (in seconds) for every considered model on all datasets from Section [4.1.](#page-7-1) The GPU memory and run time results for the other models are taken from [Walker et al.](#page-12-8) [\(2024\)](#page-12-8). Note that we used exactly the same GPU architecture as well as the same code and python libraries as in [Walker et al.](#page-12-8) [\(2024\)](#page-12-8) to ensure fair comparability, i.e., GPU memory usage and run time was measured on an Nvidia RTX 4090 GPU for all models. The run time was measured as the average run time for 1000 training steps. Table [5](#page-15-1) comprehensively shows the number of parameters, GPU memory usage, and run time for all models. Note that we further follow [Walker et al.](#page-12-8) [\(2024\)](#page-12-8) and report these results for the best performing model identified during the previously described hyperparameter optimization process. Both LinOSS models exhibit comparable GPU memory usage and run time performance to other state-space models. Notably, LinOSS achieves the fastest runtime on two out of six datasets and ranks as the second fastest on another two datasets.

# C ADDITIONAL EXPERIMENTS

### C.1 ON THE STATE MATRIX PARAMETERIZATION AND INITIALIZATION

In the main paper, we have focused on parameterizing the state matrix A in [\(1\)](#page-1-0) according to <sup>A</sup> = ReLU(A<sup>ˆ</sup> ), with diagonal matrix <sup>A</sup><sup>ˆ</sup> ∈ <sup>R</sup> <sup>m</sup>×<sup>m</sup>. This was due to the fact that LinoSS requires the state matrix to be nonnegative in order to produce stable dynamics. However, another viable parameterization choice would be A = Aˆ Aˆ . In this section, we test how the squared parameterization influences the performance of LinOSS on six long-range datasets taken from Section [4.1](#page-7-1) of the main paper. Table [6](#page-15-0) shows the average test accuracies of LinOSS-IM using a ReLU parameterization as well as using a squared parameterization together with the same baselines taken from the main paper. We can see that on average, the ReLU parameterization performs better. However, since the squared parameterization performs better on SCP2, Ethanol and Heartbeat (i.e., on three out of six datasets), we conclude that including the two parameterization choices in the hyperparameter optimization process will lead to even better performance.

Table 5: Number of parameters, GPU memory usage (in MB) and run time (in seconds) for every considered model on all long-range datasets from Section [4.1.](#page-7-1)

|                           | NRDE     | NCDE    | Log-NCDE | LRU    | S5     | Mamba   | S6    | LinOSS-IMEX | LinOSS-IM |
|---------------------------|----------|---------|----------|--------|--------|---------|-------|-------------|-----------|
| #parameters               | 105110   | 166789  | 37977    | 101129 | 22007  | 27381   | 15045 | 26119       | 134279    |
| GPU memory (MB) Worms     | 2506     | 2484    | 2510     | 10716  | 6646   | 13486   | 7922  | 6556        | 10654     |
| run time (s)              | 5386     | 24595   | 1956     | 94     | 31     | 122     | 68    | 37          | 90        |
| #parameters               | 117187   | 166274  | 91557    | 25892  | 226328 | 184194  | 24898 | 447944      | 991240    |
| GPU memory (MB) SCP1      | 716      | 694     | 724      | 960    | 1798   | 1110    | 904   | 4768        | 4772      |
| run time (s)              | 1014     | 973     | 635      | 9      | 17     | 7       | 3     | 42          | 38        |
| #parameters               | 200707   | 182914  | 36379    | 26020  | 5652   | 356290  | 26018 | 448072      | 399112    |
| GPU memory (MB) SCP2      | 712      | 692     | 714      | 954    | 762    | 2460    | 1222  | 4772        | 2724      |
| run time (s)              | 1404     | 1251    | 583      | 9      | 9      | 32      | 7     | 55          | 22        |
| #parameters               | 93212    | 133252  | 31452    | 76522  | 76214  | 1032772 | 5780  | 70088       | 6728      |
| GPU memory (MB) Ethanol   | 712      | 692     | 710      | 1988   | 1520   | 4876    | 938   | 4766        | 1182      |
| run time (s)              | 2256     | 2217    | 2056     | 16     | 9      | 255     | 4     | 48          | 8         |
| #parameters               | 15657742 | 1098114 | 168320   | 338820 | 158310 | 1034242 | 6674  | 29444       | 10936     |
| GPU memory (MB) Heartbeat | 6860     | 1462    | 2774     | 1466   | 1548   | 1650    | 606   | 922         | 928       |
| run time (s)              | 9539     | 1177    | 826      | 8      | 11     | 34      | 4     | 4           | 7         |
| #parameters               | 1134395  | 186962  | 81391    | 107544 | 17496  | 228226  | 52802 | 106024      | 91844     |
| GPU memory (MB) Motor     | 4552     | 4534    | 4566     | 8646   | 4616   | 3120    | 4056  | 12708       | 4510      |
| run time (s)              | 7616     | 3778    | 730      | 51     | 16     | 35      | 34    | 128         | 11        |

Table 6: Test accuracies averaged over 5 training runs on UEA time-series classification datasets. All models are trained based on the same hyper-parameter tuning protocol in order to ensure fair comparability. The dataset names are abbreviations of the following UEA time-series datasets: EigenWorms (Worms), SelfRegulationSCP1 (SCP1), SelfRegulationSCP2 (SCP2), EthanolConcentration (Ethanol), Heartbeat, MotorImagery (Motor).

|                     |                     |        | Worms |    |   |    |     | SCP1 |   |   |    |       | SCP2 |   |   |    |       | Ethanol |   |   |    |     |   |   |   |    |       | Motor |   |   |    | Avg |
|---------------------|---------------------|--------|-------|----|---|----|-----|------|---|---|----|-------|------|---|---|----|-------|---------|---|---|----|-----|---|---|---|----|-------|-------|---|---|----|-----|
| Seq. length         |                     | 17,984 |       |    |   |    | 896 |      |   |   |    | 1,152 |      |   |   |    | 1,751 |         |   |   |    | 405 |   |   |   |    | 3,000 |       |   |   |    |     |
| #Classes            |                     |        | 5     |    |   |    |     | 2    |   |   |    |       | 2    |   |   |    |       | 4       |   |   |    |     | 2 |   |   |    |       | 2     |   |   |    |     |
| NRDE                | 83                  | 9      | ±     | 7  | 3 | 80 | 9   | ±    | 2 | 5 | 53 | 7     | ±    | 6 | 9 | 25 | 3     | ±       | 1 | 8 | 72 | 9   | ± | 4 | 8 | 47 | 0     | ±     | 5 | 7 | 60 | 6   |
| NCDE                | 75                  | 0      | ±     | 3  | 9 | 79 | 8   | ±    | 5 | 6 | 53 | 0     | ±    | 2 | 8 | 29 | 9     | ±       | 6 | 5 | 73 | 9   | ± | 2 | 6 | 49 | 5     | ±     | 2 | 8 | 60 | 2   |
| Log-NCDE            | 85                  | 6      | ±     | 5  | 1 | 83 | 1   | ±    | 2 | 8 | 53 | 7     | ±    | 4 | 1 | 34 | 4     | ±       | 6 | 4 | 75 | 2   | ± | 4 | 6 | 53 | 7     | ±     | 5 | 3 | 64 | 3   |
| LRU                 | 87                  | 8      | ±     | 2  | 8 | 82 | 6   | ±    | 3 | 4 | 51 | 2     | ±    | 3 | 6 | 21 | 5     | ±       | 2 | 1 | 78 | 4   | ± | 6 | 7 | 48 | 4     | ±     | 5 | 0 | 61 | 7   |
| S5                  | 81                  | 1      | ±     | 3  | 7 | 89 | 9   | ±    | 4 | 6 | 50 | 5     | ±    | 2 | 6 | 24 | 1     | ±       | 4 | 3 | 77 | 7   | ± | 5 | 5 | 47 | 7     | ±     | 5 | 5 | 61 | 8   |
| S6                  | 85                  | 0      | ±     | 16 | 1 | 82 | 8   | ±    | 2 | 7 | 49 | 9     | ±    | 9 | 4 | 26 | 4     | ±       | 6 | 4 | 76 | 5   | ± | 8 | 3 | 51 | 3     | ±     | 4 | 7 | 62 | 0   |
| Mamba               | 70                  | 9      | ±     | 15 | 8 | 80 | 7   | ±    | 1 | 4 | 48 | 2     | ±    | 3 | 9 | 27 | 9     | ±       | 4 | 5 | 76 | 2   | ± | 3 | 8 | 47 | 7     | ±     | 4 | 5 | 58 | 6   |
| LinOSS-IM (ReLU)    | 95                  | 0      | ±     | 4  | 4 | 87 | 8   | ±    | 2 | 6 | 58 | 2     | ±    | 6 | 9 | 29 | 9     | ±       | 0 | 6 | 75 | 8   | ± | 3 | 7 | 60 | 0     | ±     | 7 | 5 | 67 | 8   |
| LinOSS-IM (squared) | 88                  | 9      | ±     | 2  | 5 | 86 | 6   | ±    | 1 | 8 | 59 | 3     | ±    | 7 | 8 | 32 | 7     | ±       | 6 | 2 | 76 | 8   | ± | 2 | 2 | 58 | 2     | ±     | 8 | 4 | 67 | 1   |
| LinOSS-IM (squared  | + Gaussian init) 74 | 4      | ±     | 31 | 3 | 87 | 3   | ±    | 1 | 7 | 60 | 7     | ±    | 4 | 1 | 31 | 4     | ±       | 4 | 8 | 73 | 9   | ± | 4 | 0 | 56 | 5     | ±     | 3 | 0 | 64 | 0   |

Another important question arises in the context of the state matrix initialization. While we argue in the main paper that initializing the state matrix using a simple random uniform distribution in [0, 1] leads to models that are able to learn long-range interactions, we are interested in exploring other choices in this context. To this end, we train LinOSS-IM models and initialize their state matrix using a standard normal distribution. Note that we need to use the squared parameterization in this case, as ReLU would lead to switching off approximately half of the dimensions. The results for the six datasets from Section [4.1](#page-7-1) of the main paper are shown in Table [6.](#page-15-0) We can see that initializing the state matrix using standard normal distribution leads to competitive results on almost all datasets except EigenWorms. Note that the standard deviation is very high on this dataset, suggesting that the performance is very sensitive to the random seed of the initialization. Therefore, the subpar performance on EigenWorms can be explained by the possibility of this initialization to produce large matrix entries that lead to small eigenvalues and thus vanishing gradients.

### C.2 DISSIPATIVE VS CONSERVATIVE LINOSS MODELS

In this section, we empirically analyze the dissipative behavior of LinOSS-IM and compare it to the conservative behavior of LinOSS-IMEX. To this end, we aim to predict an energy-conserving dynamical system. More concretely, we train our two LinOSS models to predict simple harmonic motion for different initial positions and velocities, i.e., the solution of,

$$\begin{aligned} y''(t) &= y(t), \\ y(0) &= A, \quad y'(0) = B, \end{aligned} \tag{10}$$

for A, B ∈ [0, 1]. We construct train, validation, and test sets by solving [\(10\)](#page-15-2) for uniform randomly chosen A, B, with 2000 train sequences, 500 validation sequences, and 500 test sequences. We set the stepsize for solving y(t) to ∆t = 0.1 and predict y(t) for 1000 steps, i.e., for the time interval [0, 100]. Clearly, [\(10\)](#page-15-2) is energy-conserving with the Hamiltonian given as H(y, y′ ) = y <sup>2</sup> + y ′2 .

Since state-space models are sequence-to-sequence models, we follow common practice and construct the input sequences as two-dimensional sequences of length 1000, where all entries of the first dimension are set to A and the second dimension is set to B. The resulting test mean-squared error (MSE) is shown in Fig. [2](#page-16-1) for each point in time. We can see that LinOSS-IMEX keeps the error constant, while the error for LinOSS-IM grows over time. This can be explained by the dissipative nature of LinOSS-IM, which forces the predicted trajectories to slowly converge to a steady-state, i.e., LinOSS-IM would go to zero in the asymptotic case of <sup>t</sup> → ∞.

![](_page_16_Figure_3.jpeg)

Figure 2: Mean squared error over time of LinOSS-IMEX and LinOSS-IM on predicting simple harmonic motion for different initial positions and velocities.

### C.3 ON THE SENSITIVITY OF ∆t IN LINOSS

While we set the timestep ∆t of the underlying time integration schemes to ∆t = 1 for all our experiments in this paper, it is natural to ask whether different choices of ∆t will lead to different performance. To analyze this, we train LinOSS-IM models on three datasets from Section [4.1](#page-7-1) of the main paper, i.e., SelfRegulationSCP1, Heartbeat, and MotorImagery dataset. We further vary ∆t between 10−<sup>3</sup> and 1, i.e., spanning three orders of magnitude. We plot the average test accuracy (with standard deviation) in Fig. [3](#page-16-2) for all three datasets. From this, we can conclude that while the choice of ∆t does influence performance, the variations are not substantial.

![](_page_16_Figure_7.jpeg)

Figure 3: Test accuracies (mean and standard deviation over five different seeds) of LinOSS-IM on three different datasets from Section [4](#page-7-2) of the main paper, i.e., SelfRegulationSCP1, Heartbeat, and MotorImagery, for varying values of the timestep ∆t of the underlying implicit time integration scheme of LinOSS [\(4\)](#page-3-2).

# D HIGHER-ORDER INTEGRATION SCHEMES

As outlined in the main text, higher-order discretization schemes can be readily applied in the context of oscillatory state-space models. Here, we provide an example on how to leverage the second-order velocity Verlet method as the underlying discretization method of LinOSS. To this end, applying the velocity Verlet integration scheme to our underlying system of ODEs [\(2\)](#page-1-1) yields,

$$\begin{aligned} \mathbf{y}_n &= \mathbf{y}_{n-1} + \Delta t \mathbf{z}_{n-1} + \frac{\Delta t^2}{2} (-\mathbf{A}\mathbf{y}_{n-1} + \mathbf{B}\mathbf{u}_n), \\ \mathbf{z}_n &= \mathbf{z}_{n-1} + \frac{\Delta t}{2} (-\mathbf{A}\mathbf{y}_n + \mathbf{B}\mathbf{u}_{n+1} - \mathbf{A}\mathbf{y}_{n-1} + \mathbf{B}\mathbf{u}_n), \end{aligned}$$

This can be rewritten in matrix form as,

$$\mathbf{x}_n = \mathbf{M}^{\text{VV}} \mathbf{x}_{n-1} + \mathbf{F}_n^{\text{VV}}, \quad (11)$$

with x<sup>n</sup> = [yn, zn] <sup>⊤</sup>, and,

$$\mathbf{M}^{\text{VV}} = \begin{bmatrix} I - \frac{\Delta t^2}{2} \mathbf{A} & \frac{\Delta t I}{4} \\ -\Delta t \mathbf{A} (I - \frac{\Delta t^2}{4} \mathbf{A}) & I - \frac{\Delta t^2}{2} \mathbf{A} \end{bmatrix}, \quad \mathbf{F}_n^{\text{VV}} = \begin{bmatrix} \frac{\Delta t^2}{2} \mathbf{B} u_n \\ -\frac{\Delta t^3}{4} \mathbf{A} \mathbf{B} u_n + \frac{\Delta t}{2} \mathbf{B} (u_{n+1} + u_n) \end{bmatrix}.$$

Equation [\(11\)](#page-17-2) can be efficiently solved using fast associative parallel scans, as described in Section [2.4,](#page-4-1) leading to an alternative model architecture we refer to as LinOSS-VV. The key distinction from the symplectic LinOSS-IMEX lies in the discretization order: LinOSS-VV employs a second-order scheme, yielding more accurate approximations of the underlying ODE system compared to the first-order IMEX approach. However, this comes at the cost of greater computational complexity. We aim to explore the role of higher-order discretization schemes within the LinOSS framework in future research.

# E SUPPLEMENT TO THE THEORETICAL INSIGHTS

# E.1 EIGENSPECTRUM OF LINOSS-IMEX

Proposition E.1. *Let* <sup>M</sup>*IMEX* ∈ <sup>R</sup> <sup>m</sup>×<sup>m</sup> *be the hidden-to-hidden weight matrix of the implicit-explicit model LinOSS-IMEX* [\(6\)](#page-3-1)*. We assume that* Akk > 0 *for all diagonal elements* k = 1, . . . , m *of* A*, and further that* <sup>0</sup> <sup>&</sup>lt; <sup>∆</sup><sup>t</sup> ≤ max k=1,...,m ( √ 2 Akk )*. Then, the eigenvalues of* M*IMEX are given as,*

$$\lambda_j = \frac{1}{2}(2 - \Delta t^2 \mathbf{A}_{kk}) + i(-1)^{\lceil \frac{j}{m} \rceil} \frac{1}{2} \sqrt{\Delta t^2 \mathbf{A}_{kk}(4 - \Delta t^2 \mathbf{A}_{kk})}, \quad \text{for all } j = 1, \dots, 2m,$$

*with* <sup>k</sup> <sup>=</sup> <sup>j</sup> *mod* <sup>m</sup>*. Moreover, all absolute eigenvalues of* <sup>M</sup>*IMEX are exactly* <sup>1</sup>*, i.e.,* |<sup>λ</sup><sup>j</sup> | = 1 *for all* j = 1, . . . , 2m*.*

*Proof.* Following the same procedure outlined in the proof of Proposition [3.1,](#page-5-0) the eigenvalues of MIMEX are given as,

$$\lambda_j = \frac{1}{2}(2 - \Delta t^2 \mathbf{A}_{kk}) + i(-1)^{\lceil \frac{j}{m} \rceil} \frac{1}{2} \sqrt{\Delta t^2 \mathbf{A}_{kk}(4 - \Delta t^2 \mathbf{A}_{kk})}, \quad \text{for all } j = 1, \dots, 2m,$$

with k = j mod m. To calculate their absolute value, we must consider two distinct cases.

- 1. ∆t <sup>2</sup>Akk = 4: it follows directly that |<sup>λ</sup><sup>j</sup> | <sup>2</sup> = ( <sup>1</sup> 2 (−4 + 2))<sup>2</sup> = 1.
- 2. ∆t <sup>2</sup>Akk < 4: With this assumption, we can compute the absolute value of λ<sup>j</sup> as,

$$\begin{aligned} |\lambda_j|^2 &= \left( \frac{2 - \Delta t^2 \mathbf{A}_{kk}}{2} \right)^2 + \frac{\Delta t^2}{4} \mathbf{A}_{kk} (4 - \Delta t^2 \mathbf{A}_{kk}) \\ &= \frac{4 - 4\Delta t^2 \mathbf{A}_{kk} + \Delta t^4 \mathbf{A}_{kk}^2}{4} + \frac{4\Delta t^2}{4} \mathbf{A}_{kk} - \frac{\Delta t^4 \mathbf{A}_{kk}^2}{4} \\ &= 1. \end{aligned}$$

# E.2 PROOF OF PROPOSITION [3.2](#page-6-0)

Proposition. *Let* {<sup>λ</sup>j} 2m <sup>j</sup>=1 *be the eigenspectrum of the hidden-to-hidden matrix* M*IM of the LinOSS-IM model* [\(4\)](#page-3-2)*. We further initialize* <sup>A</sup>kk ∼ U([0, A*max*]) *with* <sup>A</sup>*max* <sup>&</sup>gt; <sup>0</sup> *for all diagonal elements* k = 1, . . . , m *of* A *in* [\(2\)](#page-1-1)*. Then, the* N*-th moment of the magnitude of the eigenvalues are given as,*

$$\mathbb{E}(|\lambda_j|^N) = \frac{(\Delta t^2 A_{max} + 1)^{1 - \frac{N}{2}} - 1}{\Delta t^2 A_{max} (1 - \frac{N}{2})}, \quad (12)$$

*for all* j = 1, . . . , 2m*, with* k = j *mod* m*.*

*Proof.* By the law of the unconscious statistician together with the identity |<sup>λ</sup><sup>j</sup> | <sup>=</sup> √ S<sup>k</sup> from the proof of Proposition [3.1](#page-5-0) it follows that,

$$\begin{aligned}\mathbb{E}(|\lambda_j|^N) &= \frac{1}{A_{\max}} \int_0^{A_{\max}} (1 + \Delta t^2 x)^{-\frac{N}{2}} dx = \frac{1}{\Delta t^2 A_{\max}} \int_1^{\Delta t^2 A_{\max} + 1} u^{-\frac{N}{2}} du \\ &= \frac{(\Delta t^2 A_{\max} + 1)^{1 - \frac{N}{2}} - 1}{\Delta t^2 A_{\max} (1 - \frac{N}{2})},\end{aligned}$$

where we substituted u = ∆t 2 x + 1.

# E.3 PROOF OF THEOREM [3.3](#page-6-2)

Theorem. *Let* Φ : C0([0, T]; <sup>R</sup> p ) → <sup>C</sup>0([0, T]; <sup>R</sup> q ) *be a causal and continuous operator. Let* <sup>K</sup> ⊂ <sup>C</sup>0([0, T]; <sup>R</sup> p ) *be compact. Then for any* ϵ > 0*, there exist hyperparameters* m*,* m˜ *, diagonal weight matrix* <sup>A</sup> ∈ <sup>R</sup> <sup>m</sup>×<sup>m</sup>*, weights* <sup>B</sup> ∈ <sup>R</sup> m×d *,* <sup>W</sup>˜ ∈ <sup>R</sup> <sup>m</sup>˜ <sup>×</sup>m*,* <sup>W</sup> ∈ <sup>R</sup> <sup>q</sup>×m˜ *and bias vectors* <sup>b</sup> ∈ <sup>R</sup> <sup>m</sup>*,* <sup>b</sup>˜ ∈ <sup>R</sup> m˜ *, such that the output* <sup>z</sup> : [0, T] → <sup>R</sup> <sup>q</sup> *of the LinOSS model* [\(9\)](#page-6-1) *satisfies,*

$$\sup_{t \in [0, T]} |\Phi(\mathbf{u})(t) - \mathbf{z}(t)| \leq \epsilon, \quad \forall \mathbf{u} \in K.$$

*Proof.* We begin by considering the simple forced harmonic oscillator,

$$\mathbf{y}''(t) = -A^2\mathbf{y}(t) + \mathbf{u}(t), \quad (13)$$

where <sup>u</sup>(t) ∈ <sup>R</sup> p is an external forcing and we assume <sup>A</sup> ̸= 0. We further introduce the timewindowed sine transform for the input signal u(t),

$$\mathcal{L}_t \mathbf{u}(A) = \int_0^t \mathbf{u}(t-\tau) \sin(A\tau) d\tau.$$

Then, a straightforward calculation shows that the solution y(t) to [\(13\)](#page-18-2) computes (up to a constant) a time-windowed sine transform, i.e.,

$$y(t) = A^{-1} \int_0^t \mathbf{u}(\tau) \sin(A(t-\tau)) d\tau. \quad (14)$$

This can easily be verified by differentiating y,

$$\begin{aligned} \mathbf{y}'(t) &= \int_0^t \mathbf{u}(\tau) \cos(A(t-\tau)) d\tau + A^{-1}[\mathbf{u}(\tau) \sin(A(t-\tau))]_{\tau=t} \\ &= \int_0^t \mathbf{u}(\tau) \cos(A(t-\tau)) d\tau. \end{aligned}$$

Differentiating one more time yields,

$$\begin{aligned} \mathbf{y}''(t) &= -A \int_0^t \mathbf{u}(\tau) \sin(A(t-\tau)) d\tau + [\mathbf{u}(\tau) \cos(A(t-\tau))]_{\tau=t} \\ &= -A \int_0^t \mathbf{u}(\tau) \sin(A(t-\tau)) d\tau + \mathbf{u}(t) \\ &= -A^2 \mathbf{y}(t) + \mathbf{u}(t). \end{aligned}$$

We will now make use of the fundamental Lemma in [Lanthaler et al.](#page-11-4) [\(2024\)](#page-11-4) that provides a finitedimensional encoding of the operator Φ we wish to approximate, i.e., for any ϵ > 0, there exists <sup>N</sup> ∈ <sup>N</sup>, weights <sup>A</sup>1, . . . , A<sup>N</sup> and a continuous mapping Ψ : <sup>R</sup> <sup>p</sup>×<sup>N</sup> × [0, T<sup>2</sup>/4] → <sup>R</sup> q , such that

$$|\Phi(\mathbf{u})(t) - \Psi(\mathcal{L}_t \mathbf{u}(A_1), \dots, \mathcal{L}_t \mathbf{u}(A_N); t^2/4)| \leq \epsilon,$$

for all <sup>u</sup> ∈ <sup>K</sup> and <sup>t</sup> ∈ [0, T].

Using the result in [\(14\)](#page-18-3) we can then construct the input vector [Ltu(A1), . . . ,Ltu(A<sup>N</sup> ), t<sup>2</sup>/4]<sup>⊤</sup> ∈ R pN × [0, T<sup>2</sup>/4] based on the ODE system [\(1\)](#page-1-0) underlying our LinOSS model:

$$\begin{bmatrix} \mathcal{L}_t \mathbf{u}(A_1) \\ \mathcal{L}_t \mathbf{u}(A_2) \\ \vdots \\ \mathcal{L}_t \mathbf{u}(A_N) \\ t^2/4 \end{bmatrix} = \tilde{\mathbf{A}}\mathbf{y}(t), \quad (15)$$

where <sup>y</sup>(t) ∈ <sup>R</sup> pN solves the system,

$$\mathbf{y}''(t) = -\mathbf{A}^2\mathbf{y} + \mathbf{B}\mathbf{u}(t) + \mathbf{b}, \quad (16)$$

with

$$\mathbf{A} = \text{diag}([\underbrace{A_1, \dots, A_1}_{p\text{-times}}, \underbrace{A_2, \dots, A_2}_{p\text{-times}}, \dots, \underbrace{A_N, \dots, A_N}_{p\text{-times}}, \mathbf{0}]),$$

$$\mathbf{B} = [\underbrace{\mathbf{I}_p, \dots, \mathbf{I}_p}_{N\text{-times}}, \mathbf{0}]^\top, \quad \mathbf{b} = [0, \dots, 0, 1/2]^\top$$

where <sup>I</sup><sup>p</sup> ∈ <sup>R</sup> p×p is the identity matrix and A˜ equals A, except that A˜ pN,pN = 1. Thus, Ay˜ (t) computes exactly the input to the finite-dimensional operator Ψ. By the universal approximation theorem for ordinary neural networks there exist weight matrices W,Wˆ and bias b˜, such that,

$$|\Psi(\mathcal{L}_t\mathbf{u}(A_1), \dots, \mathcal{L}_t\mathbf{u}(A_N); t^2/4) - \mathbf{W}\sigma(\hat{\mathbf{W}}[\mathcal{L}_t\mathbf{u}(A_1), \dots, \mathcal{L}_t\mathbf{u}(A_N), t^2/4]^\top + \tilde{\mathbf{b}})| < \epsilon.$$

Thus, for every <sup>u</sup> ∈ <sup>K</sup> we have,

$$\begin{aligned} |\Phi(\mathbf{u}(t)) - \mathbf{z}(t)| &\leq |\Phi(\mathbf{u}(t)) - \Psi(\mathcal{L}_t \mathbf{u}(A_1), \dots, \mathcal{L}_t \mathbf{u}(A_N); t^2/4)| \\ &\quad + |\Psi(\mathcal{L}_t \mathbf{u}(A_1), \dots, \mathcal{L}_t \mathbf{u}(A_N); t^2/4) - \mathbf{W}\sigma(\tilde{\mathbf{W}}\mathbf{y}(t) + \tilde{\mathbf{b}})| \\ &< 2\epsilon, \end{aligned}$$

with W˜ = Wˆ A˜ . Since ϵ > 0 was arbitrary, we conclude that for any causal and continuous operator Φ : C0([0, T]; <sup>R</sup> p ) → <sup>C</sup>0([0, T]; <sup>R</sup> q ), compact set <sup>K</sup> ⊂ <sup>C</sup>0([0, T]; <sup>R</sup> p ) and ϵ > 0, there exists a LinOSS model of form [\(9\)](#page-6-1), which uniformly approximates <sup>Φ</sup> to accuracy <sup>ϵ</sup> for all <sup>u</sup> ∈ <sup>K</sup>. This completes the proof.

Remark 2. A natural question arises regarding the performance of LinOSS and other state-space models when learning chaotic dynamical systems. A key issue in the context of learning chaotic systems with recurrent models is the inevitability of exploding gradients during training, as rigorously demonstrated in [Mikhaeil et al.](#page-11-16) [\(2022\)](#page-11-16). While our universality result, Theorem [3.3,](#page-6-2) still holds assuming the assumptions stated are fulfilled, it can be seen that the Lipschitz constant of the readout MLP (i.e., approximation of Ψ in the proof of Theorem [3.3\)](#page-6-2) would blow up, thereby enabling exploding gradients. However, it is crucial to emphasize that this issue persists with all recurrent models and is not unique to LinOSS or other state-space models. Interestingly, it has already been empirically shown in [Hu et al.](#page-11-17) [\(2024\)](#page-11-17) that state-space models (i.e., models based on linear dynamics) vastly outperform chaotic RNNs such as LSTMs [\(Hochreiter, 1997\)](#page-11-18) and GRUs [\(Chung et al.,](#page-10-14) [2014\)](#page-10-14) for learning chaotic dynamical systems.