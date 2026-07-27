# High-Dimensional Neuronal Activity From Low-Dimensional Latent Dynamics: A Solvable Model

Valentin Schmutz∗
University College London WC1E 6BT London, UK v.schmutz@ucl.ac.uk

| Shuqi Wang                                                                                                                         |
|------------------------------------------------------------------------------------------------------------------------------------|
| École Polytechnique Fédérale de Lausanne                                                                                           |
| 1015 Lausanne, Switzerland shuqi.wang@epfl.ch Matteo Carandini University College London WC1E 6BT London, UK m.carandini@ucl.ac.uk |

Ali Haydaroglu˘
∗
University College London WC1E 6BT London, UK
ali.haydaroglu.20@ucl.ac.uk Yixiao Feng Shanghai Jiao Tong University 200240 Shanghai, China yf2887@nyu.edu

| Kenneth D. Harris                             |
|-----------------------------------------------|
| University College London WC1E 6BT London, UK |
| kenneth.harris@ucl.ac.uk                      |

## Abstract

Computation in recurrent networks of neurons has been hypothesized to occur at the level of low-dimensional latent dynamics, both in artificial systems and in the brain. This hypothesis seems at odds with evidence from large-scale neuronal recordings in mice showing that neuronal population activity is high-dimensional. To demonstrate that low-dimensional latent dynamics and high-dimensional activity can be two sides of the same coin, we present an analytically solvable recurrent neural network (RNN) model whose dynamics can be exactly reduced to a lowdimensional dynamical system, but generates an activity manifold that has a high linear embedding dimension. This raises the question: Do low-dimensional latents explain the high-dimensional activity observed in mouse visual cortex? Spectral theory tells us that the covariance eigenspectrum alone does not allow us to recover the dimensionality of the latents, which can be low or high, when neurons are nonlinear. To address this indeterminacy, we develop Neural Cross-Encoder (NCE), an interpretable, nonlinear latent variable modeling method for neuronal recordings, and find that high-dimensional neuronal responses to drifting gratings and spontaneous activity in visual cortex can be reduced to low-dimensional latents, while the responses to natural images cannot. We conclude that the high-dimensional activity measured in certain conditions, such as in the absence of a stimulus, is explained by low-dimensional latents that are nonlinearly processed by individual neurons.

## 1 Introduction

The mammalian cortex comprises a large number of neurons, which, in principle, should allow it to use a high-dimensional neural code to represent sensory, motor, and cognitive information.

Nevertheless, multi-neuronal recordings in nonhuman primates [1–4] have suggested that cortical
∗equal contribution populations perform computations by approximating low-dimensional dynamical systems [5, 6], with neuronal firing rates lying on a low-dimensional "neural manifold" [7]. In support of this hypothesis, low-dimensional dynamics have been inferred from multi-neuronal recordings through a wide variety of methods [8–23]; they spontaneously emerge in recurrent neural networks (RNNs) trained to solve behavioral tasks [2, 24–34]; and they appear in several theoretical models of noise-robust neuronal population dynamics [35–38]. A result that might at first sight challenge the low-dimensional dynamical systems hypothesis is that visual cortical population activity in mice has high linear dimension [39, 40] with shared neuronal covariance having a heavy-tailed eigenspectrum (see also [41] and [42] for recordings in cerebellum and across cortex, respectively). In particular, the shared covariance eigenspectrum has a power-law tail with an exponent close to 1 (α ≈ 1.04) [39] for responses to natural images and an exponent of α ≈ 1.14 for spontaneous activity [40]. Are these two views on the dimensionality of population activity compatible? Namely, can a low-dimensional dynamical system produce a neural manifold that has a high linear embedding dimension? Here, we first construct a solvable RNN model that reconciles the low- and high-dimensional perspectives on population activity by carefully disambiguating the *linear* dimension of the system before and *after* the neurons' nonlinearity, which we refer to as the pre- and post-activation dimension, respectively. This dichotomy refines the usual distinction between linear and "intrinsic" dimension [39, 43, 44], since the intrinsic dimension of a system is the same before and after any continuous, injective nonlinearity. Using the notions of pre- and post-activation linear dimensions, we show that our RNN can be exactly reduced to a low-dimensional dynamical system in the space of preactivations, making the pre-activations low-dimensional. Then, we show that these latent dynamics generate high-dimensional post-activation activity that has a power-law covariance eigenspectrum. (In this work, dimension will always refer to linear dimension, unless stated otherwise.) Before analyzing experimental recordings, we revisit the spectral theory of infinite-width neural networks (random feature kernels) [45–47] to quantitatively relate the pre-activation dimension, the neuronal activation function, and the post-activation covariance eigenspectrum. This three-way relationship tells us that high-dimensional activity is consistent with both low- and high-dimensional pre-activations. To uncover the pre-activation dimension of high-dimensional activity in visual cortex, we perform two-photon calcium recordings of tens of thousands of neurons from mouse visual cortex, and infer the pre-activation dimension using the Neural Cross-Encoder (NCE), an interpretable, nonlinear latent variable modeling method which models the activity of each neuron as a simple linear-nonlinear readout of low-dimensional latents. NCE reveals that both the responses to drifting gratings and spontaneous activity can be well approximated by low-dimensional pre-activations, but that responses to natural images cannot. This suggests that the encoding of natural images in visual cortex is already high-dimensional in the space of pre-activations.

## 2 Solvable Rnn Model

To demonstrate how high-dimensional post-activations can arise from low-dimensional pre-activation dynamics, we first present a solvable RNN model whose autonomous dynamics is low-dimensional in the space of pre-activations, but high-dimensional in the space of post-activations, with the post-activations producing a power-law covariance eigenspectrum. We consider an RNN consisting of N rate-units (neurons). The pre-activation xi of neuron i evolves according to

$$\dot{x}_{i}=-x_{i}+\frac{1}{N}\sum_{j=1}^{N}W_{i j}\phi(x_{j}),$$
$$(1)$$

Wijϕ(xj ), (1)
where Wij denotes the synaptic weight from neuron j to neuron i, and ϕ : R → R≥0 is a nonlinear activation function converting the pre-activations into post-activations (firing rates). To define the weights Wij , we randomly place neurons on a ring [48–50] by assigning to each neuron i an independent and uniformly distributed angle θi ∈ [0, 2π) (Fig. 1A). The weights Wij are then given by the following shifted cosine function:

$$W_{i j}:=J\cos(\theta_{i}-\theta_{j}-\Delta).$$
Wij := J cos(θi − θj − ∆). (2)
The shift ∆ in Eq. (2) makes the weights asymmetric, with neurons sending their strongest excitatory output to neurons located at an angle ∆ counter-clockwise (Fig. 1B). To make the model solvable,

$\eqref{eq:walpha}$. 
we assume that the activation function ϕ is the Heaviside step function Θ, i.e.,

$$\phi(x)=\Theta(x)={\begin{cases}1,&{\mathrm{if~}}x\geq0,\\ 0,&{\mathrm{if~}}x<0.\end{cases}}$$
0, if x < 0.(3) 
A B
RNN on a ring shifted cosine weights rank-2 factorization

$$({\mathfrak{I}})$$

VT
C
j i Wij Wij = J cos(θi − θj − Δ) W = U
latent dynamics post-activation eigenspectrum D E
F
κ1 U κ2
⋮
κ2 κ1

## 2.1 Low-Dimensional Pre-Activation Dynamics

The RNN model defined above has low-dimensional pre-activation dynamics because the weight matrix W has rank 2. Indeed, using an elementary trigonometric identity,2 W can be factorized as the outer product W = UVT of two N × 2-matrices (Fig. 1C), with

$$\mathbf{U}:=\begin{pmatrix}\cos(\theta_{1})&\sin(\theta_{1})\\ \vdots&\vdots\\ \cos(\theta_{N})&\sin(\theta_{N})\end{pmatrix}\quad\text{and}\quad\mathbf{V}:=J\begin{pmatrix}\cos(\theta_{1}+\Delta)&\sin(\theta_{1}+\Delta)\\ \vdots&\vdots\\ \cos(\theta_{N}+\Delta)&\sin(\theta_{N}+\Delta)\end{pmatrix}.$$
 .
Then, following Beiran et al. [51], we can reduce the N-dimensional system, Eq. (1), to a 2dimensional system describing the dynamics of the latent variables κ := U†x, where † denotes the pseudoinverse and x the N-dimensional vector of pre-activations (x1*, . . . , x*N )
T. The dynamics of the latent variables follows

$${\dot{\mathbf{x}}}=-\mathbf{\kappa}+{\frac{1}{N}}\mathbf{V}^{\mathrm{T}}\phi(\mathbf{U}\mathbf{\kappa}),$$

where ϕ is applied element-wise to the N-dimensional vector of pre-activations Uκ = x. In the equation above, the vector ϕ(Uκ) = ϕ(x) represents the joint post-activations (firing rates) of the N neurons.

Taking the number of neurons N → ∞ yields a neural field limit [52] where the sum over neurons becomes an integral over the ring,

becomes an integral over the ring,  $$\hat{\kappa}=-\kappa+\int_{\theta=0}^{2\pi}\mathbf{v}(\theta)\phi\left(\mathbf{u}(\theta)\cdot\boldsymbol{\kappa}\right)\frac{\mathrm{d}\theta}{2\pi},\tag{4}$$  with $\mathbf{u}(\theta):=(\cos(\theta),\sin(\theta))^{\mathrm{T}}$ and $\mathbf{v}(\theta):=J(\cos(\theta+\Delta),\sin(\theta+\Delta))^{\mathrm{T}}$. Equation (4) describes
the dynamics of the latent variables κ as the solution to a 2-dimensional dynamical system whose vector field involves an integral over the "circuit structure" [52] (the ring). Since ϕ is the step function, the integral over the ring in Eq. (4) can be solved, and we obtain the solvable 2-dimensional dynamical system,

$$\dot{\kappa}=-\kappa+\left(\begin{matrix}1&-1\\ 1&1\end{matrix}\right)\frac{\kappa}{\|\kappa\|},$$
$$(S)$$
, (5)
when J := π
√2 and ∆ := π/4 (derivation presented in Appendix A). Equation (5) generates a stable limit cycle over the unit circle (Fig. 1D), which implies that the latent variables κ will eventually rotate on the unit circle indefinitely. In Appendix E, we provide other examples of low-rank RNNs for which the latent dynamics can be expressed in a tractable form similar to Eq. (5). Neuronal activity, modeled here as the post-activations ϕ(Uκ), are simple *linear-nonlinear* readouts of latent variables κ (Fig. 1E). Hence, we have effectively reduced the dynamics of the RNN, Eq. (1), to a 2-dimensional latent dynamical system. We will say that neuronal activity is a linearnonlinear function of latent variables if it is given by the composition of a linear mapping (U) and an element-wise, nondecreasing nonlinear mapping (ϕ).

## 2.2 Post-Activations Produce A Power-Law Eigenspectrum

Since the dynamics of the latent variables is solvable in the large-network limit, and rotates on the unit circle, we can compute the correlation between the post-activations of two neurons. For any pair of neurons i and j, with positions θi and θj on the ring, respectively, the correlation of their post-activations Cij is, in the long-recording limit, given by

$$C_{ij}=\frac{2}{\pi}\left(\pi-|\theta_{i}-\theta_{j}|\right)-1,\tag{6}$$  $\theta_{i}$)) is the absolute angle difference $\theta_{i}$ and $\theta_{j}$ (derivation presented.  
where |θi−θj | := cos−1(cos(θi−θj )) is the absolute angle difference θi and θj (derivation presented in Appendix B). We can find the eigenvalue spectrum of post-activations by noting that as the number of recorded neurons M → ∞, the eigenvalues of the M × M correlation matrix C(M) defined by Eq. (6) converge to the eigenvalues of an integral operator. Since the angles θi are independently and uniformly sampled on circle [0, 2π), the correlation matrix C(M)is a so-called Euclidean random matrix [53], that is, a matrix whose entries are given by the pairwise distances between randomly sampled points in a given space. Writing λ
(M)
1 ≥ λ
(M)
2 *≥ · · · ≥* λ
(M)
N for the ranked eigenvalues of the matrix C(M), random matrix theory [54, 55] tells us that, as M → ∞, the scaled eigenvalues {λ
(M)
n /M}M
n=1, converge (in a ℓ2 sense) to the eigenvalues of the integral operator,

$$f(\theta)\mapsto\int_{\theta^{\prime}=0}^{2\pi}\left[\frac{2}{\pi}\underbrace{(\pi-|\theta-\theta^{\prime}|)}_{\text{0-th arc-cosine kernel}}-1\right]f(\theta^{\prime})\frac{\mathrm{d}\theta^{\prime}}{2\pi}.\tag{7}$$

The eigenvalues of this integral operator can be computed analytically. In Eq. (7), we have highlighted the presence of the 0-th arc-cosine kernel k0(*θ, θ*′) := π − |θ − θ
′| of Cho and Saul [45], which is well-known in machine learning, and whose eigenvalues have been computed in [46]. In short, by the rotational invariance of k0, we have that, for any positive integer m, the functions θ 7→ cos(mθ) and θ 7→ sin(mθ) are orthogonal eigenfunctions of the operator Eq. (7) sharing the same eigenvalue,

$${\frac{2}{\pi^{2}}}\int_{\theta=0}^{\pi}{(\pi-\theta)\cos(m\theta)\mathrm{d}\theta}={\frac{2}{\pi^{2}}}{\frac{1-(-1)^{m}}{m^{2}}}.$$

4 Using this result, we obtain that the ranked eigenvalues λ1 ≥ λ2 ≥ *. . .* of the operator Eq. (7) are given by

 $\lambda_n=\dfrac{4}{\pi^2}(2\lfloor(n-1)/2\rfloor+1)^{-2},\quad\forall n\in\mathbb{N}^*,$  in identical pairs that decay exactly as a power law with decay ax. 
$$(8)$$
that is, eigenvalues come in identical pairs that decay exactly as a power law with decay exponent α = 2 (Fig. 1F). Hence, post-activations are high-dimensional in the sense that their covariance eigenspectrum has a heavy tail [39]. Although the model presented assumed, for simplicity, a cosine lateral connectivity on the ring, Eq. (2), similar results can be derived for more general lateral connectivity; see Appendix D for an example. In summary, this solvable model shows that low-dimensional dynamics in the space of pre-activations can generate high-dimensional post-activations. The heavy tail of the covariance eigenspectrum implies that post-activations are not confined to any finite-dimensional linear subspace. Formally, the smallest vector space containing the post-activations generated by our model has the same size as the infinite-dimensional reproducing kernel Hilbert space associated with the kernel k0. We stress that, in this model, the heavy tail of the post-activation eigenspectrum is not due to noise, since we used a deterministic, non-chaotic RNN. Also, all the results presented above remain exact if the rate-units in Eq. (1) are replaced by linear-nonlinear-Poisson neurons, as spike noise cancels out in the limits we consider [38, 52].

## 3 Post-Activation Eigenspectrum Depends On Pre-Activation Dimension And Activation Function

To shed light on the relationship between the post-activation eigenspectrum, pre-activation dimension, and the activation function ϕ, we now turn to a more general setup, which allows us to relax some of the strong assumptions of the solvable model (Fig. 1E). First, we allow the number of latent variables d to be greater than 2, assuming that the latent variables, henceforth denoted by z (instead of κ),
are uniformly distributed on the unit sphere S
d−1in R
d. We assume that the pre-activations of the network are determined by passing the latent activity through a N × d feedforward weight matrix U with i.i.d. standard normal entries. In this setup (Fig. 2A), we call d the *pre-activation dimension*, as it sets the linear dimensionality of the pre-activations. In the solvable model of Sec. 2.2, for example, the pre-activation dimension was d = 2 (Fig. 1E). Finally, we replace the step function, Eq. (3), by the general rectified power activation function

$$\phi_{p,c}(x):=[\operatorname*{max}(0,x+c)]^{p},$$
$$({\mathfrak{g}})$$

where the activation parameter p ∈ R≥0 is a nonnegative real value and the bias c ∈ R. (By convention, ϕ0,c(x) := Θ(x + c).)
This setup can be analyzed within the framework of random feature kernels (see [56, Sec. 9.5]).

Denoting µd−1 the uniform probability measure on the sphere S
d−1, let us take T independent latent variable samples z1*, . . . ,* zT from µd−1, and define the N × T post-activation matrix A(N,T):=
(ϕp,c(Uz1), . . . , ϕp,c(UzT )). In the limits N → ∞ and T → ∞ taken successively, the covariance eigenspectrum of A(N,T)converges (when properly scaled) to the eigenvalue spectrum of the integral operator

$$f(\mathbf{z})\mapsto\int_{\mathbb{S}^{d-1}}K_{p,c,d}(\mathbf{z},\mathbf{z}^{\prime})f(\mathbf{z}^{\prime})\mathrm{d}\mu_{d-1}(\mathbf{z}^{\prime}),$$

where K*p,c,d* : S
d−1 × S
d−1 → R is the *random feature kernel*

$$K_{p,c,d}({\bf z},{\bf z^{\prime}}):=\mathbb{E}_{\mathbf{\xi}\sim{\mathcal{N}}({\bf0},{\bf I}_{d})}[\phi_{p,c}(\mathbf{\xi}\cdot{\bf z})\phi_{p,c}(\mathbf{\xi}\cdot{\bf z^{\prime}})]$$
$\mathbf{l}\leftrightarrow\mathbf{u}\mathbf{l}\leftrightarrow\mathbf{l}$
′)] (11)
(see [46, 56] or Appendix F for more details).

Drawing intuition from Fourier analysis, the smoothness of a function (here the kernel) should be related to the decay rate of its Fourier transform (here the eigenspectrum)—the smoother the function, the faster the decay rate of its Fourier transform. Known results on the eigenvalues of random feature kernels for the cases p = 0 and p = 1, with c = 0, confirm this intuition and show how it extends to general integers d [46]. Extrapolating those results to any nonnegative p and any real c, we get the following conjecture.

$$(10)$$
$$(11)$$

A B C
z3 z1 z2 U ⋮
Conjecture 1. For any p ∈ R≥0, c ∈ R, and any integer d ≥ 2*, the ranked eigenvalues* λ1 ≥ λ2 ≥
. . . *of the integral operator* (10) *obey the following power-law decay:*

λn ≍ n −α with α = 1 + 2p + 1 d − 1
$$(12)$$

$ /b_n=C\in(0,+\infty)$
, (12)
$$w h e r e\ a$$
where an ≍ bn *means* limn→+∞ an/bn = C ∈ (0, +∞).

To the best of our knowledge, Conjecture 1 is not a straightforward consequence of any existing result in theoretical machine learning [47, 57, 58] or harmonic analysis [59–62], hence our presentation of Eq. (12) as a conjecture. Note that when the activation parameter p is an integer, ϕp,c is p-times weakly differentiable, that is, the first p weak derivatives3 of ϕp,c are all locally integrable. This, and the fact that the bias c does not affect the decay rate, suggest a further extension of the conjecture to more general activation functions, with p replaced by the weak differentiability of the activation function.

We tested Conjecture 1 numerically by performing PCA on large post-activation matrices A(N,T).

The linear and continuous dependence of the decay exponent α on the activation parameter p predicted by Eq. (12) was confirmed in simulations (Fig. 2B). Simulations also confirmed that the bias c of the activation function does not affect the decay rate (Fig. 2C), a fact already mentioned in [63, 64]. To summarize, the spectral theory of random feature kernels suggests a three-way relationship between the power-law tail exponent of the post-activation eigenspectrum, the pre-activation dimension, and the activation function. This relationship should hold when we can consider the neurons as linearnonlinear functions of the latent vector, with weights that vary randomly and independently between neurons. The relationship suggests that, when high-dimensional neuronal activity (modeled here as post-activations) is observed [39, 40], two scenarios are possible: high-dimensional activity could arise from nonlinear transformation of low-dimensional latent states, or it could reflect pre-activations that are already high-dimensional. To distinguish these two scenarios, we propose, in what follows, a method for inferring the pre-activation dimension of neuronal activity in experimental recordings.

## 4 Latent Variable Modeling Of Neuronal Recordings

```
To estimate the input dimensionality of neuronal activity in mouse visual cortex (as defined in Sec.
3) we developed the Neural Cross-Encoder (NCE), a nonlinear generalization of Reduced Rank
Regression. Using NCE, we show that high-dimensional neuronal responses to drifting gratings
   3The k-th weak derivative of a function f : R → R is the defined as the function g ∈ L
                                                                                    1
                                                                                    loc that satisfies
RR
  
  φ(x)g(x)dx = (−1)kRR
                         
                          φ
                           (k)(x)f(x)dx, for all φ ∈ C
                                                     ∞
                                                     c (R).

```

are well-approximated by a linear-nonlinear readout of a low dimensional latent variable, whereas responses to natural images are not. Finally, we apply NCE to high-dimensional spontaneous dynamics in the cortex and find that they are well-approximated by a linear-nonlinear readout of low-dimensional latents.

## 4.1 Experimental Data

We conducted large-scale volumetric two-photon microscopy on awake, adult mice during visual stimulation and spontaneous activity. We targeted primary and higher visual cortices with a Light Beads Microscope [65], and extracted deconvolved activity traces for 19,223 ± 2,948 neurons using Suite3D [66] as described in Appendix H. Recordings were performed in three stimulus conditions: (1) responses to 320 full-field drifting grating stimuli with 2-14 repeats each; (2) responses to 1866 natural images with 2 repeats each; (3) spontaneous activity in the absence of stimuli for 10-15 minutes.

## 4.2 Neural Cross-Encoder (Nce)

The Neural Cross-Encoder (NCE) divides neurons randomly into two sets: a source set and a target set. It predicts the activity bt of the target set from the source set at via a non-linear readout of a set of latents, zt (Fig. 3A). NCE uses a multi-layer feedforward encoder E that ends in a bottleneck layer whose activity zt = E(at) represents a low-dimensional latent state estimated from the source neurons. The reason to use this rather than an autoencoder, which predicts one set of neurons from themselves, is to discard variability that is not shared across neurons. The NCE we used here has a single power-ReLU output layer, matching the setup of section 3, and a 3-layer encoder allowing flexible estimation of latent variables, so that the number of latent variables can be readily interpreted as the pre-activation dimension. We train NCE with stochastic gradient descent on source-target activity pairs as described in Appendix I. When all nonlinearities are removed, NCE becomes equivalent to Reduced Rank Regression [67]. When predicting stimulus-driven activity, we pair the activity of source and target neurons on different repeats of the same stimulus, to also discard shared variability that is not related to the stimulus [39].

Linear-nonlinear readout. The recorded activity of a set of target neurons at time t, b˜t ∈ R
B,
is modeled as a weighted sum of the latent variables, zt ∈ R
d, passed through a nondecreasing nonlinearity:

$\boldsymbol{\mathit{t)+r.}}$. 
bt = ϕp,c(Uzt) + r. (13)
Here, ϕp,c is the rectified power activation function, defined in Eq. 9, with a power parameter p that is constant across neurons, a pre-activation bias that varies across neurons encoded by an N-dimensional vector c, and a post-activation added bias r to account for non-zero baseline firing rates. The decoder parameters {p, c, U, r} are learned alongside the encoder parameters of E. The fact that the decoder, Eq. (13), has a single-layer is crucial as it allows us to interpret the latents (zt) as linear factors of the observed neurons' pre-activations (Uzt). It is this constrained decoder that allows us to infer the pre-activation dimension of neuronal activity; in comparison, a multi-layer decoder as used in
[13, 23] would infer something closer to the intrinsic dimension of neuronal activity, which is not our goal.

## 4.3 Results

NCE identifies the latent dimensionality of simulated data. To validate that NCE can identify the pre-activation latent variables, we test it on simulated data generated from the toy model in Sec. 2 with a ReLU readout (p = 1) and d = 2. NCE recovers the true latents up to a scaling and a shift (Fig 3B). Moreover, NCE can explain all of the variance in the population with only two pre-activation dimensions, while the corresponding linear model (Reduced Rank Regression) requires more dimensions (Fig 3C).

Pre-activation dimension is low for grating responses, high for natural image responses. We next consider the pre-activation dimensional of visual stimulus responses of visual cortex neurons.

To ensure that the NCE focused on the stimulus responses, and not correlated ongoing activity such

A. B. C.

at bt zt p,c(U⋅) +r Pre-Activation Dim (d)
NCE Linear 0 5T
−1 1 Gr ound Trut h z 1.0 Fra ctio n of Var iance Ex pl ain ed
 (
nor m.

)

U
Infer re d z 
(NC
E) 
0 5T
−2 2 2 32 0.3 source neuronstarget neurons pre-activation latents
as spontaneous activity or encoding of movements, the activity of the target cells and the source cells were taken from different repeats of the same stimuli. In the case of drifting gratings, for which we know there is a low-dimensional latent variable (the grating orientation), an NCE model with low-dimensional pre-activations accurately predicts neuronal responses (Fig. 4A,B). NCE requires fewer dimensions (5.5 ± 1.2, mean ± std) than the corresponding linear model (13.9 ± 3.0) to predict 95% of the explainable variance (defined as the maximum variance explained across all d in both models). On the other hand, NCE models with low pre-activation dimension are not sufficient to predict responses to natural images (Fig. 4C,D), requiring 93.9 ± 6.0 dimensions to reach the threshold, suggesting that natural images produce high-dimensional representations in the space of pre-activations (but see Limitations below). Linear models only account for a smaller fraction of the total variance (Fig. 4D), and therefore underestimate the dimensionality of natural image responses (48.0 ± 13.1). Spontaneous activity has low pre-activation dimension. Spontaneous activity is well predicted by NCE with low pre-activation dimension (Fig. 4E,F). Across all recordings, spontaneous activity of 1000 target neurons has an estimated pre-activation dimension d of 7.0±1.0 (mean ± std.), somewhat larger than grating responses but substantially lower than natural images responses (Fig. 4G,H). The linear model finds a similar dimension (7.5 ± 2.0), though its performance deteriorates at high preactivation dimensions due to overfitting, while the NCE performance remains consistent (Fig. 4F).

These results indicate that visual cortex activity can be modeled as a linear-nonlinear transformation of a latent vector, which is low-dimensional for grating responses and spontaneous activity, but high-dimensional for natural image responses. In the case of grating responses, we find latents that resemble the sine and cosine of the stimulus angle (Fig. 4A)—this is what one would expect to find if neurons follow the canonical model of simple cells in visual cortex [68]. On the other hand, during spontaneous activity, the dynamics of the latent variables are correlated with the running speed of the mouse, and are perhaps related to its arousal state (see Appendix I.4).

## 5 Summary Of Technical Contributions And Previous Works

Latent dynamics of low-rank RNNs Low-rank RNNs are tractable models of how the brain can perform computations through low-dimensional population dynamics [36, 51, 69–72]. In particular, the dynamics of certain low-rank RNNs, in the large-network limit, reduce to that of "effective circuits", i.e., dynamical systems describing the evolution of the latent variables [51, 52, 73]. A limitation of these effective dynamical systems is that the expression of the vector field involves an integral over the distribution of weights (the "circuit structure" [52]), making them somewhat opaque and costly to solve numerically in general. In this work (Sec. 2.1 and Appendix E), we prove that, in

Drifting Gratings A. C.

Natural Images H.

G.

E.

Spontaneous late nts pr edi cti on
(d 
= 3)
pr edi cti on
(d 
= 3)
pr edi cti on
(d 
= 3) 
grating spont natimg 1 0 1 1 0 2 grating (5.5±1.2) spont (7.0±1.0)

natimg (93.9±6.0)
Pre-Ac tivat ion D
im (
d) 
late nts late nts acti vity acti vity acti vity Pre-Activation Dim (d)
1 16 256 0 1 0 image identity 950 0 time (s) 300 B. D. F.

0 stimulus direction 360 Fra ctio n o f Va riance Expl ained (
norm.

)

2 32 512 0 1 Pre-Activation Dim (d)
2 32 512 0 1 Pre-Activation Dim (d)
2 32 512 0 1 NCE
Linear Fr acti on of Vari ance NCE
Linear Fracti on of Varian ce Fracti on of Varian ce grating spont natimg Exp lained 
(no rm.)
Exp lained
 (no rm.)
Exp lained
 (no rm.)
NCE
Linear Pre-Activation Dim (d)
several special cases that are beyond the case treated in [74], the integral over the weight distribution can be solved, yielding simple exact equations for the latent dynamics. Eigenvalue decay of random feature kernels In the infinite-width limit, two-layer neural networks with random input weights behave like *random feature kernels* that depend on the distribution of input weights and the activation function of the neurons in the hidden layer [56, Sec. 9.5]. This functional perspective can be generalized to deep networks [45, 75] and constitute the basis of the Neural Tangent Kernel formalism for studying learning dynamics [76]. When the activation function is the ReLU, the decay rate of the eigenvalues has been proven to be polynomial [46, 47, 58, 64, 77, 78], even when inputs are not assumed to be uniformly distributed on the sphere [58]; for general results on dot-product kernels, see [57, 61, 63, 79, 80]. In this work, we propose a simple formula that links the power-law exponent of the eigenvalue decay rate, the power of the rectified-power activation function, and the input dimensionality. This formula, which goes beyond known results [46, 47, 64], is presented as a conjecture that we test in simulations. Latent variable modeling of neuronal activity While most latent variable models of neuronal activity were originally developed for electrophysiological recordings [8–23], some are tailored for calcium recordings [81–83]. These models vary in their mechanistic interpretability: The inferred latents are either abstract variables, for example when the model's mapping from latents to neuronal activity involves a multi-layer neural network [13, 23], or they can be interpreted as linear factors of the neurons' pre-activations, as in [9, 10]. With nonlinear dimensionality reduction methods such as CEBRA [84] or Rastermap [85], latent variables are also abstract as there is no explicit mapping going from the latents to neuronal activity. In this work (Sec. 4), we developed NCE, a latent variable model for calcium recordings that models neuronal activity as an interpretable linear-nonlinear readout of latent variables. NCE also uses a cross-encoding scheme, which allows it to discard variability not shared across neurons. We demonstrate that NCE is capable of identifying a low-dimensional pre-activation space even when the recorded neuronal activity has high linear dimension.

## 6 Discussion

Dimensionality of neural systems The solvable RNN model we proposed produces cyclic population dynamics that is low-dimensional in the space of pre-activations, and high-dimensional in the space of post-activations (firing rates). Thus, an RNN can produce trajectories that are simultaneously low-dimensional and high-dimensional, depending on the variables being considered. In this work, we focused on the notion of *linear* dimensionality and adopted an infinite-dimensional Hilbert space formalism borrowed from kernel methods to characterize the linear dimensionality of firing rate trajectories in the large-network limit. Of course, the "intrinsic" dimension of neuronal activity is equal to 1, since the dynamics is periodic; this highlights the important distinction between intrinsic and linear (or "embedding") dimension of neuronal activity (see also [43, 44]). The type of high-dimensional activity our model produces is computationally relevant: It can be exploited by a downstream readout neuron to represent arbitrary periodic functions (see also [86]), or, following the random readout approach of [87, 88], it can be used to represent a Gaussian process prior over periodic functions (see Appendix G). Two definitions of high-dimensional neuronal activity have been studied in neuroscience. Throughout this work, we define high-dimensional neuronal activity to mean a covariance eigenspectrum with a heavy tail that decays strictly faster that 1/n [39]. This definition is well-suited for systems generating activity whose pairwise correlations do not converge to zero in the limit of large network size. In contrast, random chaotic RNNs––solvable models whose pairwise correlations do converge to zero––produce a different form of high-dimensional activity, where the eigenspectrum decays slower than 1/n [89], reflecting noise that is not shared between neurons. While a comprehensive comparison of these two types of high-dimensional activity is beyond the scope of this work, we mention that the latter is relevant when one wants to study the noisiness of neuronal responses [90]. Pre-activations and subthreshold membrane potentials We developed NCE to disentangle the linear dimension of neuronal activity and of the neurons' pre-activations. We show that even when activity is high-dimensional, it can be well-explained with low-dimensional pre-activations in the case of grating responses and spontaneous activity. From a biological point of view, how should we interpret the pre-activations inferred by NCE? If one assumes that the link between synaptic integration and neuronal firing is well approximated by a simple nonlinear activation function in cortical neurons, as NCE does, one could argue that pre-activations represent estimates of the neurons' synaptic inputs or subthreshold membrane potentials. This is an experimental prediction that large-scale voltage imaging of neuronal populations [91] may make testable in the near future. Limitations Training NCE, which is a non-convex optimization problem, can be challenging when neural data is limited. The duration of imaging experiments is limited to 3.5 h, which yields only a few thousand training examples per session for tens of thousands of neurons; thus, optimization can get stuck in local minima. To facilitate training, we limit to fitting only 1000 highly responsive neurons at a time, and use tools such as data augmentation and pretraining as described in Appendix I. The fact that we were not able to accurately predict neuronal responses to natural images with a low-dimensional NCE model does not necessarily exclude the possibility for such a model to exist, and one could possibly find it with a larger training set or a different model class/hyperparameter configuration. Note that this work did not revisit the problem of how to estimate the tail of the shared covariance eigenspectrum from neuronal recordings, which is discussed in [39, 40, 92, 93].

## Acknowledgments And Disclosure Of Funding

The authors thank Louis Pezon for useful discussions; Michael Krumin, Bex Terry and Charu B. Reddy for experimental support; Kimberly Ren for feedback on the manuscript. We also thank the four anonymous reviewers who have helped improve the manuscript. This work was funded by UKRI (Frontier Award EP/X022366/1 to MC), BBSRC (grant BB/W019884/1 to MC), the National Institutes of Health BRAIN initiative (grant U01NS126057 to MC), the Wellcome Trust (Investigator Award 223144/Z/21/Z to MC and KDH), and the ERC (101097874 to KDH). MC holds the GlaxoSmithKline / Fight for Sight Chair in Visual Neuroscience. AH is supported by a studentship from the Gatsby Charitable Foundation (GAT3755) and the Wellcome Trust (219627/Z/19/Z). VS is supported by a Royal Society Newton International Fellowship (NIF\R1\231927) and a fellowship from the Swiss National Science Foundation (grant no. 222150).

## References

[1] Mark M Churchland, John P Cunningham, Matthew T Kaufman, Justin D Foster, Paul Nuyujukian, Stephen I Ryu, and Krishna V Shenoy. Neural population dynamics during reaching.

Nature, 487(7405):51–56, 2012.

[2] Valerio Mante, David Sussillo, Krishna V Shenoy, and William T Newsome. Context-dependent computation by recurrent dynamics in prefrontal cortex. *Nature*, 503(7474):78–84, 2013.

[3] Abigail A Russo, Sean R Bittner, Sean M Perkins, Jeffrey S Seely, Brian M London, Antonio H
Lara, Andrew Miri, Najja J Marshall, Adam Kohn, Thomas M Jessell, et al. Motor cortex embeds muscle-like commands in an untangled population response. *Neuron*, 97(4):953–966, 2018.

[4] Evan D Remington, Devika Narain, Eghbal A Hosseini, and Mehrdad Jazayeri. Flexible sensorimotor computations through rapid reconfiguration of cortical dynamics. *Neuron*, 98(5):
1005–1019, 2018.

[5] Krishna V Shenoy, Maneesh Sahani, and Mark M Churchland. Cortical control of arm movements: a dynamical systems perspective. *Annual Review of Neuroscience*, 36(1):337–359, 2013.

[6] Saurabh Vyas, Matthew D Golub, David Sussillo, and Krishna V Shenoy. Computation through neural population dynamics. *Annual Review of Neuroscience*, 43(1):249–275, 2020.

[7] Juan A Gallego, Matthew G Perich, Lee E Miller, and Sara A Solla. Neural manifolds for the control of movement. *Neuron*, 94(5):978–984, 2017.

[8] M Yu Byron, John P Cunningham, Gopal Santhanam, Stephen I Ryu, Krishna V Shenoy, and Maneesh Sahani. Gaussian-process factor analysis for low-dimensional single-trial analysis of neural population activity. *Journal of Neurophysiology*, 102(1):614–635, 2009.

[9] Jakob H Macke, Lars Buesing, John P Cunningham, Byron M Yu, Krishna V Shenoy, and Maneesh Sahani. Empirical models of spiking in neural populations. In Advances in Neural Information Processing Systems, volume 24, 2011.

[10] Chethan Pandarinath, Daniel J. O'Shea, Jasmine Collins, Rafal Jozefowicz, Sergey D. Stavisky, Jonathan C. Kao, Eric M. Trautmann, Matthew T. Kaufman, Stephen I. Ryu, Leigh R. Hochberg, Jaimie M. Henderson, Krishna V. Shenoy, L. F. Abbott, and David Sussillo. Inferring single-trial neural population dynamics using sequential auto-encoders. *Nature Methods*, 15(10):805–815, 2018. doi: 10.1038/s41592-018-0109-9.

[11] Lea Duncker, Gergo Bohner, Julien Boussard, and Maneesh Sahani. Learning interpretable continuous-time models of latent stochastic dynamical systems. In International conference on machine learning, pages 1726–1734. PMLR, 2019.

[12] Joshua Glaser, Matthew Whiteway, John P Cunningham, Liam Paninski, and Scott Linderman.

Recurrent switching dynamical systems models for multiple interacting neural populations. In Advances in Neural Information Processing Systems, volume 33, pages 14867–14878, 2020.

[13] Ding Zhou and Xue-Xin Wei. Learning identifiable and interpretable latent models of highdimensional neural activity using pi-vae. In *Advances in Neural Information Processing Systems*,
volume 33, pages 7234–7247, 2020.

[14] Timothy D Kim, Thomas Z Luo, Jonathan W Pillow, and Carlos D Brody. Inferring latent dynamics underlying neural population activity via neural differential equations. In International Conference on Machine Learning, pages 5551–5561. PMLR, 2021.

[15] Marine Schimel, Ta-Chu Kao, Kristopher T Jensen, and Guillaume Hennequin. iLQR-VAE :
control-based learning of input-driven dynamics with applications to neural data. In International Conference on Learning Representations, 2022.

[16] Mohammad Reza Keshtkaran, Andrew R Sedler, Raeed H Chowdhury, Raghav Tandon, Diya Basrai, Sarah L Nguyen, Hansem Sohn, Mehrdad Jazayeri, Lee E Miller, and Chethan Pandarinath. A large-scale neural network training framework for generalized estimation of single-trial population dynamics. *Nature Methods*, 19(12):1572–1577, 2022.

[17] Adrian Valente, Jonathan W Pillow, and Srdjan Ostojic. Extracting computational mechanisms from neural data using low-rank RNNs. In *Advances in Neural Information Processing Systems*,
volume 35, pages 24072–24086, 2022.

[18] Shuqi Wang, Valentin Schmutz, Guillaume Bellec, and Wulfram Gerstner. Mesoscopic modeling of hidden spiking neurons. In *Advances in Neural Information Processing Systems*, volume 35, pages 23566–23579, 2022.

[19] Mikhail Genkin, Krishna V Shenoy, Chandramouli Chandrasekaran, and Tatiana A Engel. The dynamics and geometry of choice in the premotor cortex. *Nature*, pages 1–9, 2025.

[20] Timothy Doyeon Kim, Thomas Zhihao Luo, Tankut Can, Kamesh Krishnamurthy, Jonathan W.

Pillow, and Carlos D Brody. Flow-field inference from neural data using deep recurrent networks.

In *Forty-second International Conference on Machine Learning*, 2025.

[21] Aditi Jha, Diksha Gupta, Carlos Brody, and Jonathan W Pillow. Disentangling the roles of distinct cell classes with cell-type dynamical systems. In Advances in Neural Information Processing Systems, volume 37, pages 33668–33690, 2024.

[22] Christopher Langdon and Tatiana A Engel. Latent circuit inference from heterogeneous neural responses during cognitive tasks. *Nature Neuroscience*, pages 1–11, 2025.

[23] Adam Gosztolai, Robert L Peach, Alexis Arnaudon, Mauricio Barahona, and Pierre Vandergheynst. MARBLE: interpretable representations of neural population dynamics using geometric deep learning. *Nature Methods*, pages 1–9, 2025.

[24] David Sussillo and Omri Barak. Opening the black box: low-dimensional dynamics in highdimensional recurrent neural networks. *Neural Computation*, 25(3):626–649, 2013.

[25] David Sussillo, Mark M Churchland, Matthew T Kaufman, and Krishna V Shenoy. A neural network that finds a naturalistic solution for the production of muscle activity. *Nature Neuroscience*,
18(7):1025–1033, 2015.

[26] Guillaume Hennequin, Tim P Vogels, and Wulfram Gerstner. Optimal control of transient dynamics in balanced networks supports generation of complex movements. *Neuron*, 82(6):
1394–1406, 2014.

[27] Federico Carnevale, Victor de Lafuente, Ranulfo Romo, Omri Barak, and Néstor Parga. Dynamic control of response criterion in premotor cortex during perceptual detection under temporal uncertainty. *Neuron*, 86(4):1067–1077, 2015.

[28] Jonathan A Michaels, Benjamin Dann, and Hansjörg Scherberger. Neural population dynamics during reaching are better explained by a dynamical system than representational tuning. PLoS
Computational Biology, 12(11):e1005175, 2016.

[29] Alexander Rivkind and Omri Barak. Local dynamics in trained recurrent neural networks.

Physical Review Letters, 118(25):258101, 2017.

[30] Joao Barbosa, Rémi Proville, Chris C Rodgers, Michael R DeWeese, Srdjan Ostojic, and Yves Boubenec. Early selection of task-relevant features through population gating. *Nature* Communications, 14(1):6837, 2023.

[31] Christopher Langdon, Mikhail Genkin, and Tatiana A Engel. A unifying perspective on neural manifolds and circuits for cognition. *Nature Reviews Neuroscience*, 24(6):363–377, 2023.

[32] Daniel Durstewitz, Georgia Koppe, and Max Ingo Thurm. Reconstructing computational system dynamics from neural data with recurrent neural networks. *Nature Reviews Neuroscience*, 24
(11):693–710, 2023.

[33] Laura N Driscoll, Krishna Shenoy, and David Sussillo. Flexible multitask computation in recurrent networks utilizes shared dynamical motifs. *Nature Neuroscience*, 27(7):1349–1363, 2024.

[34] Harsha Gurnani, Weixuan Liu, and Bingni W Brunton. Feedback control of recurrent dynamics constrains learning timescales during motor adaptation. *bioRxiv*, pages 2024–05, 2024.

[35] Ashok Litwin-Kumar and Brent Doiron. Slow dynamics and high variability in balanced cortical networks with clustered connections. *Nature Neuroscience*, 15(11):1498–1505, 2012. doi:
10.1038/nn.3220.

[36] Francesca Mastrogiuseppe and Srdjan Ostojic. Linking connectivity, dynamics, and computations in low-rank recurrent neural networks. *Neuron*, 99(3):609–623, 2018.

[37] Brian DePasquale, David Sussillo, LF Abbott, and Mark M Churchland. The centrality of population-level factors to network computation is demonstrated by a versatile approach for training spiking networks. *Neuron*, 111(5):631–649, 2023.

[38] Valentin Schmutz, Johanni Brea, and Wulfram Gerstner. Emergent rate-based dynamics in duplicate-free populations of spiking neurons. *Physical Review Letters*, 134(1):018401, 2025.

[39] Carsen Stringer, Marius Pachitariu, Nicholas Steinmetz, Matteo Carandini, and Kenneth D
Harris. High-dimensional geometry of population responses in visual cortex. *Nature*, 571(7765):
361–365, 2019.

[40] Carsen Stringer, Marius Pachitariu, Nicholas Steinmetz, Charu Bai Reddy, Matteo Carandini, and Kenneth D Harris. Spontaneous behaviors drive multidimensional, brainwide activity.

Science, 364(6437):eaav7893, 2019.

[41] Frederic Lanore, N Alex Cayco-Gajic, Harsha Gurnani, Diccon Coyle, and R Angus Silver.

Cerebellar granule cell axons support high-dimensional representations. *Nature Neuroscience*, 24(8):1142–1150, 2021.

[42] Jason Manley, Sihao Lu, Kevin Barber, Jeffrey Demas, Hyewon Kim, David Meyer, Francisca Martínez Traub, and Alipasha Vaziri. Simultaneous, cortex-wide dynamics of up to 1 million neurons reveal unbounded scaling of dimensionality with neuron number. *Neuron*, 112
(10):1694–1709, 2024.

[43] Mehrdad Jazayeri and Srdjan Ostojic. Interpreting neural computations by examining intrinsic and embedding dimensionality of neural activity. *Current Opinion in Neurobiology*, 70:113–120, 2021.

[44] Mark D Humphries. Strong and weak principles of neural dimension reduction. *Neurons,*
Behavior, Data analysis, and Theory, 5(2):1–28, 2021.

[45] Youngmin Cho and Lawrence Saul. Kernel methods for deep learning. In Advances in Neural Information Processing Systems, volume 22, 2009.

[46] Francis Bach. Breaking the curse of dimensionality with convex neural networks. Journal of Machine Learning Research, 18(19):1–53, 2017.

[47] Alberto Bietti and Francis Bach. Deep equals shallow for ReLU networks in kernel regimes. In International Conference on Learning Representations, 2021.

[48] William Skaggs, James Knierim, Hemant Kudrimoti, and Bruce McNaughton. A model of the neural basis of the rat's sense of direction. In Advances in Neural Information Processing Systems, volume 7, 1994.

[49] Rani Ben-Yishai, R Lev Bar-Or, and Haim Sompolinsky. Theory of orientation tuning in visual cortex. *Proceedings of the National Academy of Sciences*, 92(9):3844–3848, 1995.

[50] Kechen Zhang. Representation of spatial orientation by the intrinsic dynamics of the headdirection cell ensemble: a theory. *Journal of Neuroscience*, 16(6):2112–2126, 1996.

[51] Manuel Beiran, Alexis Dubreuil, Adrian Valente, Francesca Mastrogiuseppe, and Srdjan Ostojic. Shaping dynamics with multiple populations in low-rank recurrent networks. *Neural* Computation, 33(6):1572–1615, 2021.

[52] Louis Pezon, Valentin Schmutz, and Wulfram Gerstner. Linking neural manifolds to circuit structure in recurrent networks. *bioRxiv*, pages 2024–02, 2024.

[53] Marc Mézard, Giorgio Parisi, and Anthony Zee. Spectra of euclidean random matrices. *Nuclear* Physics B, 559(3):689–701, 1999.

[54] Vladimir Koltchinskii and Evarist Giné. Random matrix approximation of spectra of integral operators. *Bernoulli*, 6(1):113–167, 2000. ISSN 1350-7265,1573-9759. doi: 10.2307/3318636.

[55] Charles Bordenave. Eigenvalues of euclidean random matrices. Random Structures & Algorithms, 33(4):515–532, 2008. doi: https://doi.org/10.1002/rsa.20228.

[56] Francis Bach. *Learning theory from first principles*. MIT press, 2024. [57] Meyer Scetbon and Zaid Harchaoui. A spectral analysis of dot-product kernels. In International Conference on Artificial Intelligence and Statistics, pages 3394–3402. PMLR, 2021.

[58] Yicheng Li, Zixiong Yu, Guhan Chen, and Qian Lin. On the eigenvalue decay rates of a class of neural-network related kernel functions defined on general domains. Journal of Machine Learning Research, 25(82):1–47, 2024.

[59] Thomas Kühn. Eigenvalues of integral operators with smooth positive definite kernels. Archiv der Mathematik, 49:525–534, 1987.

[60] M Castro and V Menegatto. Eigenvalue decay of positive integral operators on the sphere.

Mathematics of Computation, 81(280):2303–2317, 2012.

[61] Douglas Azevedo and Valdir Antonio Menegatto. Sharp estimates for eigenvalues of integral operators generated by dot product kernels on the sphere. *Journal of Approximation Theory*,
177:57–68, 2014.

[62] Thaís Jordão and V Menegatto. Estimates for fourier sums and eigenvalues of integral operators via multipliers on the sphere. *Proceedings of the American Mathematical Society*, 144(1):
269–283, 2016.

[63] Blake Bordelon and Cengiz Pehlevan. Population codes enable learning from few examples by shaping inductive bias. *Elife*, 11:e78606, 2022.

[64] Marjorie Xie, Samuel P Muscinelli, Kameron Decker Harris, and Ashok Litwin-Kumar. Taskdependent optimal representations for cerebellar learning. *Elife*, 12:e82914, 2023.

[65] Jeffrey Demas, Jason Manley, Frank Tejera, Kevin Barber, Hyewon Kim, Francisca Martínez Traub, Brandon Chen, and Alipasha Vaziri. High-speed, cortex-wide volumetric recording of neuroactivity at cellular resolution using light beads microscopy. *Nature Methods*, 18(9):
1103–1111, 2021.

[66] Ali Haydaroglu, Tinya Chang, Andrew Landau, Michael Krumin, Sam Dodgson, Liad J. ˘
Baruchin, Maria Cozan, Jingkun Guo, David Meyer, Charu Bai Reddy, Jian Zhong, Na Ji, Sylvia Schröder, Kenneth D. Harris, Alipasha Vaziri, and Matteo Carandini. Suite3D: Volumetric cell detection for two-photon microscopy. *bioRxiv*, page 2025.03.26.645628, 2025.

[67] Alan Julian Izenman. Reduced-rank regression for the multivariate linear model. Journal of Multivariate Analysis, 5(2):248–264, 1975.

[68] Matteo Carandini. What simple and complex cells compute. *The Journal of Physiology*, 577(Pt 2):463–466, December 2006. ISSN 0022-3751. doi: 10.1113/jphysiol.2006.118976.

[69] Friedrich Schuessler, Alexis Dubreuil, Francesca Mastrogiuseppe, Srdjan Ostojic, and Omri Barak. Dynamics of random recurrent networks with correlated low-rank structure. Physical Review Research, 2(1):013111, 2020.

[70] Alexis Dubreuil, Adrian Valente, Manuel Beiran, Francesca Mastrogiuseppe, and Srdjan Ostojic.

The role of population structure in computations through neural dynamics. *Nature Neuroscience*,
25(6):783–794, 2022.

[71] Yue Wan and Robert Rosenbaum. High-dimensional dynamics in low-dimensional networks.

arXiv preprint arXiv:2504.13727, 2025.

[72] Francesca Mastrogiuseppe, Joana Carmona, and Christian Machens. Stochastic activity in low-rank recurrent neural networks. *bioRxiv*, pages 2025–04, 2025.

[73] Romain Veltz and Olivier Faugeras. Local/global analysis of the stationary solutions of some neural field equations. *SIAM Journal on Applied Dynamical Systems*, 9(3):954–998, 2010.

[74] Matthijs Pals, Jakob H Macke, and Omri Barak. Trained recurrent neural networks develop phase-locked limit cycles in a working memory task. *PLOS Computational Biology*, 20(2): e1011852, 2024.

[75] Jaehoon Lee, Jascha Sohl-dickstein, Jeffrey Pennington, Roman Novak, Sam Schoenholz, and Yasaman Bahri. Deep neural networks as gaussian processes. In International Conference on Learning Representations, 2018.

[76] Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In *Advances in Neural Information Processing Systems*,
volume 31, 2018.

[77] Amnon Geifman, Abhay Yadav, Yoni Kasten, Meirav Galun, David Jacobs, and Basri Ronen.

On the similarity between the laplace and neural tangent kernels. In Advances in Neural Information Processing Systems, volume 33, pages 1451–1461, 2020.

[78] Alberto Bietti and Julien Mairal. On the inductive bias of neural tangent kernels. In Advances in Neural Information Processing Systems, volume 32, 2019.

[79] Alex Smola, Zoltán Ovári, and Robert C Williamson. Regularization with dot-product kernels.

In *Advances in Neural Information Processing Systems*, volume 13, 2000.

[80] Ha Quang Minh, Partha Niyogi, and Yuan Yao. Mercer's theorem, feature maps, and smoothing.

In *International Conference on Computational Learning Theory*, pages 154–168. Springer, 2006.

[81] Luke Y Prince, Shahab Bakhtiari, Colleen J Gillon, and Blake A Richards. Parallel inference of hierarchical latent dynamics in two-photon calcium imaging of neuronal populations. *bioRxiv*, pages 2021–03, 2021.

[82] Feng Zhu, Harrison A Grier, Raghav Tandon, Changjia Cai, Anjali Agarwal, Andrea Giovannucci, Matthew T Kaufman, and Chethan Pandarinath. A deep learning framework for inference of single-trial neural population dynamics from calcium imaging with subframe temporal resolution. *Nature Neuroscience*, 25(12):1724–1734, 2022.

[83] Tze Hui Koh, William E Bishop, Takashi Kawashima, Brian B Jeon, Ranjani Srinivasan, Yu Mu, Ziqiang Wei, Sandra J Kuhlman, Misha B Ahrens, Steven M Chase, et al. Dimensionality reduction of calcium-imaged neuronal population activity. *Nature Computational Science*, 3(1):
71–85, 2023.

[84] Steffen Schneider, Jin Hwa Lee, and Mackenzie Weygandt Mathis. Learnable latent embeddings for joint behavioural and neural analysis. *Nature*, 617(7960):360–368, May 2023.

[85] Carsen Stringer, Lin Zhong, Atika Syeda, Fengtong Du, Maria Kesa, and Marius Pachitariu.

Rastermap: a discovery method for neural population recordings. *Nature Neuroscience*, 28(1): 201–212, 2025.

[86] Soledad Gonzalo Cogno, Horst A Obenhaus, Ane Lautrup, R Irene Jacobsen, Claudia Clopath, Sebastian O Andersson, Flavio Donato, May-Britt Moser, and Edvard I Moser. Minute-scale oscillatory sequences in medial entorhinal cortex. *Nature*, 625(7994):338–344, 2024.

[87] Radford M Neal. Priors for infinite networks. *Bayesian learning for neural networks*, pages 29–53, 1996.

[88] Christopher Williams. Computing with infinite networks. In *Advances in Neural Information* Processing Systems, volume 9, 1996.

[89] David G Clark, LF Abbott, and Ashok Litwin-Kumar. Dimension of activity in random neural networks. *Physical Review Letters*, 131(11):118401, 2023.

[90] Gengshuo John Tian, Ou Zhu, Vinay Shirhatti, Charles M Greenspon, John E Downey, David J
Freedman, and Brent Doiron. Neuronal firing rate diversity lowers the dimension of population covariability. *bioRxiv*, pages 2024–08, 2024.

[91] Jian Zhong, Ryan G. Natan, Qinrong Zhang, Justin S. J. Wong, Christoph Miehl, Krishnashish Bose, Xiaoyu Lu, François St-Pierre, Su Guo, Brent Doiron, Kevin K. Tsia, and Na Ji. Faced 2.0 enables large-scale voltage and calcium imaging in vivo. page 2025.03.06.641784, 2025.

[92] Dean A. Pospisil and Jonathan W. Pillow. Revisiting the high-dimensional geometry of population responses in visual cortex. *bioRxiv*, 2024. doi: 10.1101/2024.02.16.580726.

[93] Marius Pachitariu, Lin Zhong, Alexa Gracias, Amanda Minisi, Crystall Lopez, and Carsen Stringer. A critical initialization for biological neural networks. *bioRxiv*, pages 2025–01, 2025.

[94] Tsai-Wen Chen, Trevor J. Wardill, Yi Sun, Stefan R. Pulver, Sabine L. Renninger, Amy Baohan, Eric R. Schreiter, Rex A. Kerr, Michael B. Orger, Vivek Jayaraman, Loren L. Looger, Karel Svoboda, and Douglas S. Kim. Ultrasensitive fluorescent proteins for imaging neuronal activity.

Nature, 499(7458):295–300, 2013.

[95] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network.

arXiv, (arXiv:1503.02531), 2015. arXiv:1503.02531.

[96] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv*,
(arXiv:1412.6980), 2017. arXiv:1412.6980.

[97] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers:
Surpassing human-level performance on imagenet classification. In 2015 IEEE International Conference on Computer Vision (ICCV), page 1026–1034. IEEE, 2015.

## Neurips Paper Checklist

1. **Claims**
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The claims we make in abstract and introduction result our results. Guidelines:
- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: There is a paragraph titled "Limitations" in the discussion section where we clearly state the limitations of our results. Guidelines:
- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper. - The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach.

For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. **Theory Assumptions And Proofs**

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: Our nontrivial mathematical statements are either fully proved or clearly stated as conjectures (e.g. Conjecture 1 in the main text). Guidelines:
- The answer NA means that the paper does not include theoretical results.

- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

- All assumptions should be clearly stated or referenced in the statement of any theorems. - The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. **Experimental Result Reproducibility**

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: The architecture, training procedure and hyperparameters for the Neural Cross- Encoder are described in detail. While the text descriptions are sufficient to reproduce the results, we also provide the code for *in silica* experiments. We also share the preprocessed neuronal datasets. The methods for *in vivo* experiments and the preprocessing of imaging data are described clearly to enable reproducibility. Guidelines:
- The answer NA means that the paper does not include experiments. - If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways.

For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. **Open Access To Data And Code**

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: All code required to reproduce the main experimental results is anonymized and submitted with the supplementary materials. Preprocessed neural datasets, comprised of deconvolved spontaneous activity and averaged, deconvolved stimulus responses, is also anonymized and submitted with the supplementary materials.

Guidelines:
- The answer NA means that paper does not include experiments requiring code.

- Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.

- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. **Experimental Setting/Details**

Question: Does the paper specify all the training and test details (e.g., data splits, parameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: All relevant details of the training and testing of presented models are described in the main text and appendices, including the dataset generation, hyperparameters, and optimization procedure. Guidelines:
- The answer NA means that the paper does not include experiments. - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. **Experiment Statistical Significance**

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: On Fig. 4, error bars are reported and clearly defined. The factors of variability they represent, including variability from subselection of neurons within a session and variability across session, are clearly stated. On Fig. 2B, error bars are not reported as the number of points representing single simulations (> 20 per line) already visually indicates the variance of the simulations. Guidelines:
- The answer NA means that the paper does not include experiments. - The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors). - It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. **Experiments Compute Resources**

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: All experiments are undertaken on a single workstation, with a total compute time no greater than 7 days. Further details of the workstation and compute time are provided in the appendices Guidelines:
- The answer NA means that the paper does not include experiments. - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. **Code Of Ethics**

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: The research presented in the paper conforms with the NeurIPS Code of Ethics. Guidelines:
- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. **Broader Impacts**

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA]