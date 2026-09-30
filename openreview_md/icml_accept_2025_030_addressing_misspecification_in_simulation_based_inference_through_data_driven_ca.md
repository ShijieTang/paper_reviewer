# Addressing Misspecification In Simulation-Based Inference Through Data-Driven Calibration

Antoine Wehenkel * 1 **Juan L. Gamella** * 2 3 Ozan Sener 1Jens Behrmann 1 **Guillermo Sapiro** 1 Jörn-Henrik Jacobsen 2 **Marco Cuturi** 1

## Abstract

Driven by steady progress in deep generative modeling, simulation-based inference (SBI) has emerged as the workhorse for inferring the parameters of stochastic simulators. However, recent work has demonstrated that model misspecification can compromise the reliability of SBI,
preventing its adoption in important applications where only misspecified simulators are available. This work introduces robust posterior estimation (RoPE), a framework that overcomes model misspecification with a small real-world calibration set of ground-truth parameter measurements. We formalize the misspecification gap as the solution of an optimal transport (OT) problem between learned representations of real-world and simulated observations, allowing RoPE to learn a model of the misspecification without placing additional assumptions on its nature. RoPE demonstrates how OT and a calibration set provide a controllable balance between calibrated uncertainty and informative inference, even under severely misspecified simulators. Results on four synthetic tasks and two real-world problems with groundtruth labels demonstrate that RoPE outperforms baselines and consistently returns informative and calibrated credible intervals.

## 1 **Introduction**

Many fields of science and engineering have shifted in recent years from modeling real-world phenomena through a few equations to relying instead on highly complex computer simulations. While this shift has increased model versatility and the ability to explain or replicate complex
*Equal contribution 1Apple 2Work done while being at Apple 3ETH Zürich. Correspondence to: Antoine Wehenkel <awehenkel@apple.com>.

phenomena, it has also necessitated the development of new statistical inference methods. In particular, state-of-the-art simulation-based inference (SBI, Cranmer et al., 2020) algorithms leverage neural networks to learn surrogate models of the likelihood (Papamakarios et al., 2019), likelihood ratio (Hermans et al., 2020), or posterior distribution (Papamakarios & Murray, 2016), from which one can extract confidence or credible intervals over the parameters of interest given an observation. While SBI has proven helpful when the simulator is a faithful description of the studied phenomenon, e.g., for scientific applications (Delaunoy et al., 2020; Brehmer, 2021; Lückmann, 2022; Linhart et al., 2022; Hashemi et al., 2022; Tolley et al., 2023; Avecilla et al., 2022), recent work has also highlighted the unreliability of SBI methods under model misspecification (Cannon et al., 2022; Schmitt et al., 2023). Addressing Misspecification with a Calibration Set. In this work, we target important applications of SBI in common settings where (1) the goal is to estimate a hard-tomeasure variable from indirect but readily available measurements of other variables; (2) only misspecified simulators relating these variables are available; and (3) a few ground-truth pairings of the hard-to-measure variables and the related variables are available in a *calibration* set1.

Such a setting can arise, for example, when inferring the properties of a patient's cardiovascular system from noninvasive and abundant measurements of other physiological signals (Wehenkel et al., 2023), or when developing soft sensors to monitor industrial processes in real time, where directly measuring the quantity of interest is costly and time consuming, for example, through laboratory analysis, but where related variables can be measured quickly and inexpensively (Jiang et al., 2021; Perera et al., 2023).

1 settings, the main challenge lies in the absence of a paired dataset of simulated and corresponding real outputs. To handle this knowledge gap, RoPE estimates a coupling between real and simulated observations using optimal transport (OT, Peyré et al., 2017; Villani et al., 2009). The algorithm extends neural posterior estimation (Papamakarios & Murray, 2016) and models misspecification using OT. We evaluate the performance of the algorithm on existing benchmarks from the SBI literature and introduce four new benchmarks, two of which are synthetic and two come from real physical systems. To the best of our knowledge, the latter constitute the first real-world benchmarks that directly provide a ground truth for the inferred parameters for SBI under misspecification. We conduct additional experiments to investigate the impact on RoPE's performance of varying calibration set sizes, prior misspecification, and distribution shifts, as well as various ablation studies.

## 2 **Background & Notation**

In this section, we first pose the machine learning problem we are trying to solve and then formally introduce SBI, model misspecification, and OT, as our method lies at the intersection of these fields.

We consider a simulator, S : R
k × [0, 1] → R
d, that takes in physical parameters θ ∈ Θ ⊆ R
kand a random seed ε ∈ [0, 1] to generate simulated measurements xs ∈ Xs ⊆ R
d.

The simulator is a simplified version of a real and unknown generative process p
⋆(xo) := Rp
⋆(θ)p
⋆(xo | θ)dθ that produces real-world observations xo ∈ Xo ⊆ R
d. We assume this process depends on parameters with the same physical meaning as the ones of the simulator and thus use the same notation θ. Our task is to estimate a well-calibrated and informative posterior distribution p(θ | x io) for each observation x io in a test set D, reducing uncertainty compared to the prior distribution, assuming that the prior is well-specified. To achieve our goal, we have access to: 1. the misspecified simulator S that embeds domain knowledge and generates samples whose distribution approximates p
⋆(xo | θ),
2. a prior p(θ) that approximates the marginal distribution p
⋆(θ) of parameters in the real-world, 3. a small calibration set of labeled real-world observations C := {(θ i, x io)}
nc i=1 composed of i.i.d. samples from p
⋆(θ, xo), which enables data-driven correction of the simulator's misspecification, 4.

a test set D := {x io}
no i=1 of real-world observations arising from p
⋆(xo) for which we want to estimate the posterior.

## 2.1 **Simulation-Based Inference (Sbi)**

Applying statistical inference to simulators is challenged by the absence of a tractable likelihood function (Cranmer et al., 2020). As a solution, SBI algorithms leverage modern machine learning methods to tackle inference in this likelihood-free setting (Lueckmann et al., 2021; Delaunoy et al., 2022; Glöckler et al., 2022). Among SBI algorithms, neural posterior estimation NPE (Papamakarios & Murray, 2016; Lueckmann et al., 2017; Radev et al., 2020) is a broadly applicable method that trains a conditional density estimator of p(θ | xs) from a dataset of parameter-simulation pairs. In this paper, we focus on making NPE robust to model misspecification. NPE usually parametrizes the posterior with a neural conditional density estimator (NCDE), which is composed of (1)
a neural statistic estimator (NSE), denoted by hω : Xs → R
l, that compresses observations into l-dimensional representations and, (2) a normalizing flow (NF, Papamakarios et al., 2021; Tabak & Vanden-Eijnden, 2010) that parameterizes the posterior density as pϕ(θ | hω(xs)). The parameters ϕ and ω of the NCDE are trained with stochastic gradient ascent on the expected log-posterior probability, solving the following optimization problem ϕ
⋆, ω
⋆ ∈ arg max ϕ,ω E
 θ∼p(θ) ε∼U[0,1]
-log pϕ(θ | hω(*S(θ, ε*))), (1)
where p(θ) denotes a prior over the parameters θ.

Under the assumption that the class of functions represented by the NCDE contains the true posterior, solving (1) leads to a surrogate pϕ⋆ (θ | hω⋆ (xs)) that matches exactly the posterior p(θ | xs) corresponding to the simulator. In that case, θ ⊥ xs | hω⋆ (xs), that is, the NSE hω⋆ is a sufficient statistic of xs for the parameter θ (Chen et al., 2020; Wrede et al., 2022; Chan et al., 2018). In practice, we can only approach perfect training by generating a sufficiently large number of pairs (θ, xs) and doing a search on the NCDE's architecture and training hyperparameters. To simplify notation, we denote the NCDE learned with NPE as p˜(θ | xs).

## 2.2 **Model Misspecification**

In statistics, where the model parameters do not necessarily carry real-world meaning, model misspecification generally denotes the inability of a model to reproduce the observed data distribution. Formally, a parametric model p(xo | θ)
is said to be misspecified with respect to some true datagenerating process p
⋆(xo) if the latter does not fall within the family of distributions defined by the model, i.e. ∄θ ∈
Θ : p(xo | θ) = p
⋆(xo) ∀xo (Cannon et al., 2022). In contrast, we are not necessarily interested in reproducing the observed data xo but only in inferring the parameter value θ from an observation xo. For this goal, naively using the standard definition is insufficient, as a model may be well-specified but still produce incorrect credible intervals for the parameters of interest θ. This undesired behavior may happen, for example, if the model is over-parameterized, as illustrated in Appendix A. Thus, in this work, we define model misspecification differently and align it with the setting motivated in Section 1. Intuitively, we describe model misspecification as the nontransferability of the posterior obtained from the simulator to the prediction of real-world parameters. Formally, we assume that the pairs of parameters and observations (θ, xo) are i.i.d. from an unknown distribution p
⋆(θ, xo), which implicitly defines p
⋆(θ | xo), the Bayes optimal predictor of the parameter given an observation. Under this premise, we say a simulator is misspecified if ∃S ⊆ Θ × X : ∀(θ, xo) ∈ S,
p(θ) = p
⋆(θ) and p
⋆(θ | xo) ̸= p(θ | xs = xo).

Following this definition, we frame the problem of model misspecification in SBI as a learning task where our goal is to find a good estimator of p
⋆(θ | xo). As we assume the simulator provides strong domain knowledge, we focus on the challenging settings where the dataset of labeled real observations D := {(θ i, x io)}
n i=1 that we have for learning p
⋆(θ | xo) is small. In such settings, most examples must be saved for testing and only a small subset, denoted by the calibration set C, remains available for training.

## 2.3 **Semi-Balanced Optimal Transport (Ot)**

As further motivated in Section 3, RoPE models the misspecification between simulations and real-world observations as an OT coupling. For readers unfamiliar with OT, a coupling between two distributions—e.g., p(xs) and p(xo)—is a distribution π
⋆(xs, xo) on the product space whose marginals coincide with those two distributions while minimizing an expected cost Eπ⋆ [c(xo, xs)]. The function c : Xo × Xs → R
assigns a cost to any pair (xo, xs) ∈ Xo × Xs. In our setting, we can access a limited number no of real-world observations {x io}
no i=1, which we assume are i.i.d. from the unknown distribution p
⋆(xo). Writing C := [c(x io, x js)]ij for the cost matrix between observed and simulated data, we solve the discrete semi-balanced (Rabin et al., 2014) and entropy-regularized (Frogner et al., 2015) OT problem. This formulation preserves a strict marginal constraint on the observed data, but relaxes the marginal constraint on the simulated data, thus allowing certain simulations xs to be discarded or down-weighted. Namely, given a set {x js}
ns j=1 of simulated observations, we search for the non-negative transport matrix P
⋆ ∈ Bo that satisfies the left marginal constraint,

marginal constraint,  $$B_{o}=\left\{P\in\mathbb{R}_{+}^{n_{o}\times n_{s}}:\sum_{j=1}^{n_{s}}P_{ij}=\frac{1}{n_{o}}\ \forall i=1,...,n_{o}\right\}$$  and solves  $$P^{\star}=\arg\min_{P\in\mathcal{B}_{o}}\left\langle P,C\right\rangle+\rho\,\mathrm{KL}\left(P^{T}\mathbf{1}_{n_{o}}\|\frac{\mathbf{1}_{n_{s}}}{n_{s}}\right)+\gamma\langle P,\log P\rangle,\tag{2}$$
where 1n is a vector of ones with size n and KL is the Kullback-Leibler divergence. Therefore, a larger ρ > 0 promotes a coupling that fits the marginal of simulated data more closely, and γ > 0 is a hyperparameter that encourages entropic transport matrices. This problem can be solved with a variant of the Sinkhorn algorithm (Cuturi, 2013) with efficient GPU implementations. In our experiments, we rely on OTT (Cuturi et al., 2022) to return such a coupling P
⋆, given the cost matrix C and the parameters γ and ρ, parameterized as τ = *ρ/(ρ* + γ). Setting τ = 1 amounts to a perfectly balanced transport.

## 3 **Rope: Modeling Misspecification With Ot**

$\eqref{eq:walpha}$. 
$\mathbf{\hat{x}}$ . 

In this section, we formally introduce our robust posterior estimation algorithm (RoPE) and highlight some benefits of modeling misspecification with OT. RoPE approaches the problem of misspecification as a hybrid modeling task by combining the simulator with a misspecification model learned from the few observations in the calibration set. The main modeling assumption of RoPE is xo ⊥ θ | xs, (3)
that is, given the simulated observations xs, the real observations xo contain no additional information about the parameters θ. As a consequence, we can express the posterior for real-world observations as p(θ | xo) = Rp(θ | xs)p(xs | xo)dxs, where p(θ | xs) is easily approximated with NPE. On the other hand, the conditional p(xs | xo), which can be attributed to misspecification, is what RoPE
intends to learn by estimating an OT coupling (that is then conditioned on x0, c.f. 4).

While this assumption introduces an information bottleneck, it does not prevent the method from achieving calibrated and informative posterior distributions—even if the assumption is only partially met in practice (e.g., tasks E and F in Figure 2). In fact, it acts as a regularizer, enabling the learning of a generalizable misspecification model from only a small calibration set, and it ensures that predictions remain grounded in the expert knowledge embedded in the simulator. This bottleneck can be limiting for simulators that are highly misspecified and fail to model the dependencies between parameters and observations. However, when the simulator encodes phenomena the practitioner believes to be invariant across different application environments, the assumption forestalls "shortcut learning" (Geirhos et al., 2020) from the calibration data and improves generalization. In Appendix D, we illustrate this property using real out-of-distribution data.
Intuitively, the discrete OT coupling P
⋆between the two
point clouds {x
is}
ns
i=1 and {x
is}
no
i=1 obtained from solving (2)
can be seen as an approximation of a joint distribution π
⋆in
Xo×Xs when τ = 1 (see Appendix E for further discussion).
Then, the modeled misspecification π
⋆, together with our
modeling assumption (3), defines the posterior distribution for real-world observations as
$$p(\theta\mid\mathbf{x}_{o})=\int p(\theta\mid\mathbf{x}_{s})\pi^{\star}(\mathbf{x}_{s}\mid\mathbf{x}_{o})\mathrm{d}\mathbf{x}_{s},$$
⋆(xs | xo)dxs, (4)
$$\quad(4)^{\frac{1}{2}}$$
where the posterior p(θ | xs) can be approximated very precisely with NPE (Papamakarios & Murray, 2016) as NFs are universal density estimators of continuous distributions (Wehenkel & Louppe, 2019; Draxler et al., 2024). We approximate π
⋆by computing the OT coupling P
⋆between the test set D and a set {x js}
ns j=1 of ns simulations generated by running the simulator on parameters from the given prior θ j ∼ p(θ). The cost function is defined in the next section. Thus, RoPE estimates the posterior for realworld observations as a mixture of the posteriors p˜ obtained with NPE, that is,

$$\tilde{p}(\theta\mid\mathbf{x}_{o}^{i}):=\sum_{j=1}^{n_{s}}\alpha_{ij}\tilde{p}(\theta\mid\mathbf{x}_{s}^{j}),\text{where}\alpha_{ij}=n_{o}P_{ij}^{\star}.\tag{5}$$

## 3.1 **Defining The Ot Cost Function**

In our setting, an ideal coupling would pair a real-world observation with simulations generated by the same parameters. Hence, the cost function should be insensitive to variation in the data (e.g., noise) that is independent of θ. Formally, we can write c(xo, xs) = c(ho(xo), hs(xs)),
where ho and hs are sufficient statistics for θ with respect to xo and xs, respectively. A key concern is to find a meaningful way to learn ho, the sufficient statistic for the real observations. As discussed in Appendix G, we can learn an approximate minimal sufficient statistic hω⋆ for the simulated observations with NPE. Because the simulator carries information about the true generative process, our approach is to fine-tune hω⋆ using the calibration set, which is otherwise too small to learn a representation from real-world data only. Denoting the fine-tuned neural network as gφ : Xo → R
l, the fine-tuning objective reads

$$\mathcal{L}(\varphi;\mathcal{C}):=\sum_{i=1}^{n_{e}}[\mathbf{g}_{\varphi}(\mathbf{x}_{o}^{i})-\mathbb{E}_{\varepsilon\sim\mathcal{U}[0,1]}[\mathbf{h}_{\omega^{*}}\left(S(\theta^{i},\varepsilon)\right)]|_{2},\tag{6}$$  where the expectation is approximated via a Monte-Carlo
approximation. The training of g starts from the weights ω
⋆and optimizes (6) with gradient descent. Optimizing
(6) enforces, at least on the calibration set, that g and h are close in L2 norm when applied to observations from the same parameter θ. Thus, we define the OT cost as c(xo, xs) := |gφ⋆ (xo) − hω⋆ (xs)|2, where gφ⋆ is the NSE obtained after fine-tuning (6). Figure 4 in Appendix B depicts RoPE's training and inference steps. We discuss the computational cost of RoPE in Section H.

## 3.2 **On The Benefits Of Using Optimal Transport To** Handle Misspecification

While we could have chosen other approaches to model p(xs | xo)—e.g., conditional deep generative modelsseveral attractive properties directly follow from modeling the misspecification as an OT coupling between simulated and real-world measurements. First, **a self-calibration** property: by modeling the posterior as (5), when τ = 1 (i.e., the transport is perfectly balanced), the marginal posterior distribution over the test set, i.e., *p˜(θ*) := R*p˜(θ* | xo)p
⋆(xo)dxo, converges to the prior distribution as the number of simulated observations Ns approaches infinity, as expected from a well-estimated posterior distribution. A proof and further discussion of this self-calibration property is given in Section F. Second, **a control mechanism for the** posteriors' confidence: the entropic regularization of OT not only enables fast computation of the transport coupling but also provides an effective control mechanism to balance the calibration of the posterior with its informativeness. Indeed, for small entropic regularization, the estimated posteriors have low entropy and may be overconfident, as they are sparse mixtures of a few simulation posteriors *p˜(θ* | x js).

In contrast, for large values of γ in (2), the coupling matrix becomes uniform and the corresponding posteriors tend to the prior, as p(θ | xo) ≈
1 ns Pns jp˜(θ | x js) is a Monte-Carlo approximation of Ep(xs)[˜p(θ | xs)] ≈ p(θ). Thus, the practitioner can optimize the hyper-parameter γ to find the right trade-off between calibration of the estimated posteriors, favored by higher γ, and their informativeness, favored by lower γ. Finally, **robustness to prior misspecification**: by enabling the transport to be unbalanced—that is, to discard simulated observations when τ < 1—RoPE can flexibly depart from the assumed marginal distribution of p(θ) and be robust to prior misspecification. Thus, the parameter τ can be seen as a control mechanism to account for the user's confidence in the prior distribution. In the rest of the text, we denote the method as RoPE⋆ when τ < 1 and as RoPE
when τ = 1. In Section 5.1, we provide guidance on how to set γ and τ in practice.

## 4 **Related Work**

The problem we address shares fundamental similarities with sim2real transfer learning, where the goal is to bridge the gap between simulated and real-world data. In robotics and computer vision, this challenge has been tackled through domain randomization (Tobin et al., 2017), which increases simulation diversity to improve real-world generalization, and domain adaptation techniques (Ganin et al., 2016; Long et al., 2015; Bousmalis et al., 2018) that learn domain-invariant representations. However, unlike these approaches that typically focus on point predictions, RoPE addresses the more challenging problem of transferring uncertainty quantification from simulation to reality while preserving calibration properties. The setting we consider also naturally connects to semisupervised learning (Zhu, 2005), as both involve leveraging abundant unlabeled data alongside limited labeled examples. Our setup with the calibration set resembles few-shot learning scenarios (Wang et al., 2020), where rapid adaptation occurs with minimal labeled examples. While classical semi-supervised methods focus on exploiting unlabeled data for classification or regression tasks, our approach differs in that it leverages a large set of labeled data obtained through simulation. Crucially, unlike standard semi-supervised or few-shot learning, where labeled and unlabeled data come from the same distribution, we must explicitly account for the distributional mismatch between simulated and real observations. In both likelihood-based and simulation-based inference settings, model misspecification has recently gained substantial interest from the research community. Among developed strategies, works that take inspiration from generalized Bayesian inference (Bissiri et al., 2016) are numerous (Dellaporta et al., 2022; Chérief-Abdellatif & Alquier, 2020; Matsubara et al., 2022; Pacchiardi & Dutta, 2021; Schmon et al., 2020; Gao et al., 2023; Frazier et al., 2023). In the specific context of SBI, recent works (Ward et al., 2022; Huang et al., 2023; Kelly et al., 2023) have investigated solutions to improve the robustness of existing neural-network-based SBI methods to model misspecification, detecting it at inference time (Schmitt et al., 2023). Similarly, Frazier et al.

(2020) studied the impact of misspecification on approximate Bayesian computation methods (ABC, Rubin, 1984),
introducing diagnostics to detect it and proposing strategies to make ABC robust. For the interested reader, Nott et al. (2023) review restricted likelihood methods, Bayesian modular inference, and parametric projection methods, which are standard frameworks to handle model misspecification in likelihood-based Bayesian inference. In contrast to these approaches, we frame model misspecification in SBI as a learning problem, recognizing that if the ultimate goal is to perform inference over parameters for downstream decision-making, it is essential to have a test set to empirically validate the performance of any inference procedure. RoPE leverages a small subset of this test set as a calibration set to overcome the modeled misspecification in a supervised manner.

## 5 **Experiments**

Our experiments aim to (1) empirically validate the discussion in Section 3.2, and (2) illustrate settings in which RoPE enables uncertainty quantification under model misspecification and small calibration datasets. The experiments comprise two existing benchmarks from the SBI literature, two synthetic benchmarks, and two new benchmarks from real physical systems for which both labeled data and simulators are available. While these benchmarks remain simplified versions of real-world scenarios, they represent various types of misspecification with varied parameter and observation spaces, allowing us to study RoPE's performance under diverse configurations. We briefly describe each task and provide examples of real vs. simulated observations in Figures 1 and 2. Further details about the experimental setup can be found in Appendix I.

Task A & B (synthetic): CS & SIR . We reproduce the cancer and stromal cell development (CS) and the stochastic epidemic model (SIR) benchmarks from Ward et al. (2022). We provide a description of the parameters, observations and synthetic misspecification in Appendix I.1 Task C (synthetic): Pendulum. The damped pendulum is a common benchmark for hybrid learning algorithms (Takeishi & Kalousis, 2021; Yin et al., 2021; Wehenkel et al., 2022) that leverage both domain knowledge and real-world data. The simulator outputs the horizontal position of a frictionless pendulum given its fundamental frequency ω0 ∈
R 
+ and amplitude A ∈ R
+, with randomness introduced via a phase shift and white measurement noise. As misspecified
"real-world" data, we generate observations from a damped pendulum with friction. Task D (synthetic): Hemodynamics. Following Wehenkel et al. (2023), we define the task of inferring the stroke volume (SV) and the left ventricular ejection time (LVET) from normalized arterial pressure waveforms. The simulator is a PDE solver (Melis, 2017) that produces an 8-second time-series xs sampled at 125Hz. As synthetic misspecification, the simulator assumes all arteries have constant length, whereas this parameter varies in the "real-world" data. Task E (real): Light Tunnel. We employ one of the light tunnel datasets from Gamella et al. (2025). The tunnel is an elongated chamber with a controllable light source at one end, two linear polarizers mounted on rotating frames, and a camera. Our task consists of predicting the color setting of the light source ((*R, G, B*) ∈ [0, 255]3) and the dimming effect of the polarizers α ∈ [0, 1] from the captured images. The simulator takes the parameters θ := [*R, G, B, α*] and produces an image consisting of a hexagon roughly the size of the light source, with a color equal to [*αR, αG, αB*]. Task F (real): Wind Tunnel. We employ one of the wind tunnel datasets from Gamella et al. (2025). The tunnel is a chamber with two controllable fans that push air through it, and barometers that measure air pressure at different locations. A hatch controls the area of an additional opening to the outside. The dataset is a collection of pressure curves that result from applying a short impulse to the intake fan power and measuring the change in air pressure inside the tunnel. Our inference task consists of predicting the hatch position, θ := H ∈ [0, 45] given a pressure curve. As a simulator model, we adapt the physical model given in Gamella et al. (2025, Appendix IV). Metrics. We consider two metrics to assess whether RoPE provides reliable and useful uncertainty quantification. First, given a labeled test set {(θ i, x io)}
n i=1, we compute the logposterior probability (LPP) as LPP := 1n Pn i=1 log ˜p(θ i| x io) ≈ Ep(θ,xo)[log ˜p(θ | xo)] . The LPP, also called the negative log probability of the true test parameter (NLTP), is

Task A  CS
LPP (better = higher values)
ACAUC (better = closer to zero)
Synthetic benchr SE
-0.67 uning-and OT-only ROPE
NP
-3.8
.1 b 24.8 x
−0.1 10 50 200 1000 10 50 200 1000 RR
Calibration set size Calibration set size Task B  SIR
LPP (better = higher values)
ACAUC (better = closer to zero)
Synthetic benchm
..

2.4
. Lning.ong 2.3 RPE
RePE
-0.1 X,
.749.3 Bea 10 50 200 1000 10 50 200 1000 Calibration Calibration set size set size Task C Pendulum LPP (better
= higher values ACAUC (better = closer to zero)
Synthetic benchmark 3.1 S
hvae t = A, ton NPE
RoPE
MP
-2.5 JNPE
.1 OT only Prior 3773.8
-0.1 10 200 1000 10 s 200 50 1000 Calibration set size Calibration set size Task D. Hemodynam ACAUC (better = closer to zero)
LPP (better = higher values)
Synthetic benchmark 0.22 ng only NPE
.1
-2.3 a 151.7 0.0 1000 50 200 1000 200 10 10 50 Real Simulator Calibration set size Calibration set size Task E Light Tunnel Real-world benchmark LPP (better = higher values) **ACAUC** (better = closer to zero)
SBI
J-NPE
NPE
RoPE*
HVAE
tuning-only Observations NNPE
J-NPE
J-NPE
RoPE
MLP
SBI
Prior OT-onlyOT-only NPE-RS NPE-RS
MLP
tuning-only NNPE
SBI
HVAE
Prior RoPE
RoPE*
NPE
Simulated Real Task F Wind Tunnel Real-world benchmark LPP (better = higher values) **ACAUC** (better = closer to zero)
NPE-RS
tuning-only Observations NNPE
MLP
MLP
J-NPE
RoPE*
RoPE*
OT-only RoPE
RoPE
Prior NNPE
tuning-only Prior OT-only NPE-RS
SBI
NPE
Simulated Real
Figure 2: Continuation of Figure 1 above. For task F, the ACAUC of the NPE baseline is -0.5 and not shown.

an empirical estimation of the expectation over possible observations of the negative cross entropy between the true and estimated posterior; thus, for an infinite test set, it is only maximized by the true posterior. LPP characterizes the entropy reduction on the estimation of θ achieved by a posterior estimator p˜ when given one observation, on average, over the test set. Second, the average coverage AUC (ACAUC) indicates the average calibration of k 1D credible intervals extracted from the estimated posteriors, i.e., ACAUC := 1 kn Pk j=1 Pn i=1 R 1 0 α − 1[θ ij ∈
Θp˜(θj |xio)(α)]dα, where Θp˜(θj |xio)(α) denotes the credible interval for the j th dimension of the parameter θ at level α. Its value is positive (resp. negative) if, on average over different credible levels, parameter dimensionality, and observations, the corresponding credible intervals are overconfident (resp. underconfident). The ACAUC of a perfectly specified prior distribution is zero. The integral can be efficiently approximated, as described in Appendix J. ACAUC does not capture joint calibration, as dependencies between parameters are not explicitly assessed. Alternative dependence-sensitive metrics may require larger test sets to be stable. For all experiments, we compute the LPP and ACAUC on labeled test set containing 2000 pairs (θ, xo). Baselines. As a sanity check, we compare the performance of RoPE against four reference baselines: the **prior** p(θ), which amounts to the lower bound on the LPP for any calibrated posterior estimator when the prior is well-specified; the SBI posterior, which is an NPE trained and tested on simulated data and thus provides an upper bound on the LPP for RoPE under the independence assumption xo ⊥ θ | xs (see Appendix I for more details); (NPE) a posterior estimator fitted to the simulated data and applied to the real data; and (**J-NPE**) a posterior estimator trained jointly on the pooled simulated and real observations. The latter two baselines represent some first approaches that a practitioner may consider. Furthermore, to asses how a fully supervised approach would fare if trained directly on the calibration set, we compare the performance of RoPE to MLP, which trains a neural network to predict the mean and log-variance of a Gaussian posterior distribution by maximizing the calibration set log-likelihood. We train both the MLP and J-NPE baselines in a supervised way, and we thus expect these baselines to perform strongly as the size of the calibration set becomes sufficiently large and the test data is i.i.d. We also run **NPE-RS** (Huang et al., 2023), which trains a robust version of NPE with a regularization loss that forces the distributions of NSE on simulated and test data to match. For a fair comparison with RoPE, we use the n = 2000 test examples to compute the regularization, informing NPE-RS as much as possible. We additionally run Noisy NPE (**NNPE**, Ward et al., 2022), the amortized version of RNPE introduced in the same paper, which improves the robustness of NPE by introducing a Spike and Slab error model on simulated data statistics. We also run **HVAE** (Takeishi &
Kalousis, 2021), which constitutes a strong baseline when the simulator can be made differentiable (tasks C and E) but is not directly applicable otherwise. More details about the experimental setup can be found in Appendix I.

## 5.1 **Results**

Figure 1 compares the performance of RoPE and the other methods and baselines on the six tasks we consider with a correctly specified prior. To demonstrate that applying RoPE is straightforward, we deliberately fix γ = 0.5 for RoPE and τ = 0.9 for RoPE⋆in all tasks. In Figure 3, we further study the role of these hyperparameters in optimizing performance. RoPE achieves robust posterior estimation for all tasks. As mentioned above, the SBI and prior baselines provide upper and lower bounds on the expected performance of a well-calibrated posterior estimator, under the modeling assumption made in Section 3. For all tasks, even with minimal calibration budgets, RoPE is the only method that consistently returns well-calibrated, or sometimes slightly under-confident, posterior estimates while significantly reducing uncertainty compared to the prior distribution. As the size of the calibration set increases, we see that J-NPE
and MLP adapt and their performance improves and aligns with or outperforms RoPE. This adaptability is an expected behavior in i.i.d. settings, where real-world data eventually allows finding the minimizer of empirical risk among a class of predictors. Nevertheless, these two baselines tend to be overconfident even for larger calibration sets, as highlighted by their positive ACAUC numbers, which are significantly larger than RoPE's in almost all configurations. Moreover, on task E, where posteriors are complex conditional distributions—whose entropy increases with darker images and contain non-trivial dependencies between parameters—RoPE remains the best approach, even with a calibration set containing more than 1000 examples. As an outlier, we observe that NPE trained on simulated data achieves the best results for the SIR benchmark (Task B), indicating that the misspecification of this benchmark is not a challenging test case for existing SBI methods and may not be a meaningful test for methods that cope with model misspecification. Finally, because interpreting these metrics can be difficult, we complement these numerical results with corner and calibration plots for all tasks in Appendix K.

Ablation study. RoPE combines two steps with distinct roles, shown in Figure 4, Appendix B: (1) a fine-tuning step, which improves the domain generalization of the NSE; and (2) an OT step, aiming to model the misspecification as a coupling between simulations and observations. To better understand their respective contribution to the performance of RoPE, we look at two ablated versions of our algorithm: **tuning-only** which appends the fine-tuned NSE to the NF trained on simulated data and directly applies it to the real observations without an OT step; and **OT-only**, which directly performs OT with L2-norm in the original NSE space c(xo, xs) = |hω⋆ (xo) − hω⋆ (xs)|2. In Figure 1, we observe that the results for tuning-only are poor except for Task B, where misspecification is negligible. In contrast, for tasks A, D, and F, OT-only exhibits performance on par with RoPE. Nevertheless, RoPE can significantly outperform OT-only, such as in tasks C and E where the misspecification is significant. We conclude that the OT step is crucial and fine-tuning is sometimes necessary. In practice, we recommend to first evaluate the performance of OT-only on the test set, and optimize γ before using a subset of the test samples for fine-tuning. Effect of entropic regularization—setting γ. In Figure 3a, we study the effect of entropic regularization by varying the regularization parameter γ. For all values of γ, excluding γ ≥ 5, we observe that both LPP and ACAUC consistently improve with the calibration set size. For large values of γ, the entropic regularization dominates and pushes toward a uniform mapping, resulting in posteriors that approximate the prior distribution and are barely affected by the calibration set size. These empirical results are consistent with the theoretical discussion in Subsection 3.2. As a recommendation for practitioners, our empirical evaluation suggests that values between 0.1 and 1 provide well-calibrated and precise credible intervals. Ideally, the practitioner shall keep a portion of the calibration set for validation, using it to optimize γ based on the metrics of interest. If this is not possible, we recommend employing γ = 0.5, which offers sharp and calibrated posteriors on all our benchmarks.

RoPE⋆**for prior misspecification—setting** τ . We now study the impact of prior misspecification on RoPE and its unbalanced version RoPE⋆. In Figure 3, we compare the performance of RoPE (γ = 0.5 and τ = 1) and RoPE⋆(γ *= 0.*5 and varying τ ) on extensions of Task E and C, where the ground-truth parameters of the test dataset come from distributions different to the assumed prior distributions. For task E, we observe that RoPE's performance is robust to the prior misspecification; it provides well-calibrated and informative posteriors, as is also visible in the corner plots of Figure 5 in Appendix C. While the gap between RoPE and RoPE⋆is negligible in the case of a well-specified prior (see Task E in Figure 1), under prior misspecification RoPE⋆leverages the additional flexibility in the OT solution and discards some of the simulated observations, achieving higher LPP. Similarly, for Task C in Figure 3c, when there is no prior misspecification, RoPE (i.e, τ = 1) achieves the best performance; using lower values of τ becomes preferable as prior misspecification increases. From these experiments, we recommend leveraging τ as a hyperparameter describing the confidence in the assumed prior distribution—setting its value to 0.9 offers robust performance for both well-specified and partially misspecified priors. The user shall also explore lower values when there is suspicion that the prior distribution is overly spread with respect to the correct prior.

A Entropic regularization Task E - Light Tunnel LPP
B Robustness to prior misspecification Task E - Light Tunnel C OT balance parameter Task C - Pendulum LPP
LPP
Observations Prior B
betabinomial dist. 

Simulator Prior A
uniform dist. 

ACAUC
ACAUC
min. max. min. max.

ACAUC
Simulator prior real data real data True prior ROPE **ROPE*** - 0.9 **ROPE*** - 0.5
Figure 3: (a) Effect of γ on the LPP and ACAUC scores of RoPE on the light-tunnel task for different sizes of the calibration set. The value of γ is shown by each curve. For reference, we plot the metrics achieved by the SBI posterior and prior distribution on simulated data. (b-c) Effect of τ ∈ [0.1, 1] under a prior misspecification in Task E (b); and for various levels of prior misspecification in task C (c).

## 6 **Discussion**

While Section 5 demonstrates the effectiveness of RoPE, opportunities for future work remain, which we discuss now. Curse of dimensionality. While our experiments focused on low-dimensional parameter spaces, as is common for many applications of SBI, the dimensionality of θ may impact two critical parts of RoPE. First, with each additional parameter θk+1, given xo, the NSE must encode up to K dependencies between θk+1 and the other dimensions θ1*, . . . , θ*k. While generating more simulations can address the curse of dimensionality in the simulation space, finetuning on a small calibration may no longer suffice to cope with misspecification. Second, the dimensionality of the manifold on which the NSE projects the simulated and realworld observations will grow, and finding a meaningful coupling between the two populations may require larger sample sizes. A potential solution is to focus on marginal or 2D posterior distributions and ignore higher-dimensional dependencies in p(θ | xo). Nevertheless, extending RoPE to such settings certainly opens new questions, e.g., concerning the development of better fine-tuning strategies that can leverage calibration sets with incomplete labels. Non-iid Calibration Sets. An important assumption made by RoPE is that the calibration set contains i.i.d. samples drawn from the same distribution p
⋆(θ, xo) as the test data.

However, practical constraints may lead to calibration data being collected from a different, potentially biased, distribution p˜(θ, xo). We identify two main scenarios. If p˜ and p
⋆share the same support, the fine-tuning step can still correct for the distributional shift, especially with a sufficiently large calibration set. For smaller sets, RoPE's robustness hinges on the neural statistic estimator's (NSE) ability to generalize. Moreover, the optimal transport (OT) step provides additional resilience: observations where the fine-tuned NSE performs well will be accurately matched, leading to reliable posteriors, while poorly generalized observations may cause the posterior to revert to the prior. In the more challenging scenario where p˜ and p
⋆have disjoint support, even arbitrarily large calibration sets may fail to provide relevant training examples, making fine-tuning highly dependent on out-of-distribution generalization. Here, the OT step is expected to highlight this issue, as the lack of meaningful matches will cause the transport matrix to become uniform, leading the posterior to revert to the prior. Appendix L further investigate RoPE's sensitivity to these practical challenges, on the Light Tunnel task, using a calibration set from a different prior than the test set, approximating the 'same support' scenario. Other extensions. Similar to incomplete labels, in certain applications we may only have access to noisy labels, measured with a well-modeled but noisy measurement process.

Further developing the fine-tuning stage to exploit such noisy labels would be necessary to make an approach similar to RoPE applicable. Our strategy of modeling misspecification as an OT coupling opens up several avenues to address more specific problem setups. For example, we can leverage the inductive bias in the neural network architecture of neural OT to better cope with large test sets. This appears as a promising direction to amortize the mapping between simulation and real-world data. Conclusion. Motivated by important applications where SBI is not applied due to its sensitivity to model misspecification, we have introduced RoPE, a method that jointly exploits a calibration set and optimal transport to extend neural posterior estimation for misspecified simulators. Our experiments on diverse benchmarks demonstrate RoPE's ability to estimate calibrated and informative posterior distributions for various simulators and real-world examples. Overall, we have framed model misspecification as a challenge in transferring predictive models from simulated to real-world data. Our work highlights the need for a labeled test set to validate inference quality, encouraging future research to treat misspecification as a machine learning problem.

## Acknowledgements

The authors would like to acknowledge Michal Klein for his help with OTT library and Maria Cervera, Laura Manduchi, Joe Futoma, Andy Miller and Pierre Ablin for providing useful feedback on the manuscript.

## Impact Statement

This paper presents a framework and an algorithm to address model misspecification in simulation-based inference (SBI). SBI is predominantly applied in scientific fields where complex simulators of physical phenomena are available, such as astronomy, medicine, particle physics, or climate modeling. A priori, this circumscribes the application of our algorithm to highly specialized scientific domains in the natural sciences, precluding issues such as fairness or privacy. However, its application to the scientific domain is not exempt from societal or ethical implications, particularly when computer simulations may inform research or policy decisions. In this regard, we find some properties of the algorithm particularly promising, such as uncertainty quantification and the limitation of not drawing conclusions beyond the given expert model. However, more work is needed to deeply understand the reliability of these properties and how they are affected by violations of the core assumptions, such as a well-specified prior. Such work should precede any sort of over-selling to practitioners about the benefits of the algorithm. Rather, we see our work as a contribution towards a more broad and successful application of SBI techniques; success in this endeavor, as for the establishment of any scientific tool, will require an iterative dialogue between the scientists who develop the methodology and those who use it.

## References

Avecilla, G., Chuong, J. N., Li, F., Sherlock, G., Gresham, D., and Ram, Y. Neural networks enable efficient and accurate simulation-based inference of evolutionary parameters from adaptation dynamics. *PLoS biology*, 20(5): e3001633, 2022.

Bissiri, P. G., Holmes, C. C., and Walker, S. G. A general framework for updating belief distributions. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 78(5):1103–1130, 2016.

Bousmalis, K., Irpan, A., Wohlhart, P., Bai, Y., Kelcey, M.,
Kalakrishnan, M., Downs, L., Ibarz, J., Pastor, P., Konolige, K., et al. Using simulation and domain adaptation to improve efficiency of deep robotic grasping. In 2018 IEEE international conference on robotics and automation (ICRA), pp. 4243–4250. IEEE, 2018.

Brehmer, J. Simulation-based inference in particle physics.

Nature Reviews Physics, 3(5):305–305, 2021.

Cannon, P., Ward, D., and Schmon, S. M. Investigating the impact of model misspecification in neural simulationbased inference. *arXiv preprint arXiv:2209.01845*, 2022.

Chan, J., Perrone, V., Spence, J., Jenkins, P., Mathieson, S., and Song, Y. A likelihood-free inference framework for population genetic data using exchangeable neural networks. Advances in neural information processing systems, 31, 2018.

Chen, Y., Zhang, D., Gutmann, M., Courville, A., and Zhu, Z. Neural approximate sufficient statistics for implicit models. *arXiv preprint arXiv:2010.10079*, 2020.

Chérief-Abdellatif, B.-E. and Alquier, P. Mmd-bayes: Robust bayesian estimation via maximum mean discrepancy. In Symposium on Advances in Approximate Bayesian Inference, pp. 1–21. PMLR, 2020.

Collett, E. *Field guide to polarization*. International society for optics and photonics, 2005.

Cranmer, K., Brehmer, J., and Louppe, G. The frontier of simulation-based inference. Proceedings of the National Academy of Sciences, 117(48):30055–30062, 2020.

Cuturi, M. Sinkhorn distances: Lightspeed computation of optimal transport. *Advances in neural information* processing systems, 26, 2013.

Cuturi, M., Meng-Papaxanthos, L., Tian, Y., Bunne, C.,
Davis, G., and Teboul, O. Optimal transport tools (ott): A jax toolbox for all things wasserstein. arXiv preprint arXiv:2201.12324, 2022.

Delaunoy, A., Wehenkel, A., Hinderer, T., Nissanke, S.,
Weniger, C., Williamson, A., and Louppe, G. Lightningfast gravitational wave parameter inference through neural amortization. In *Machine Learning and the Physical* Sciences. Workshop at the 34th Conference on Neural Information Processing Systems (NeurIPS), 2020.

Delaunoy, A., Hermans, J., Rozet, F., Wehenkel, A., and Louppe, G. Towards reliable simulation-based inference with balanced neural ratio estimation. Advances in Neural Information Processing Systems, 35:20025–20037, 2022.

Dellaporta, C., Knoblauch, J., Damoulas, T., and Briol, F.-X.

Robust bayesian inference for simulator-based models via the mmd posterior bootstrap. In International Conference on Artificial Intelligence and Statistics, pp. 943–970. PMLR, 2022.

Draxler, F., Wahl, S., Schnörr, C., and Köthe, U. On the universality of coupling-based normalizing flows. arXiv preprint arXiv:2402.06578, 2024.

Falkiewicz, M., Takeishi, N., Shekhzadeh, I., Wehenkel, A.,
Delaunoy, A., Louppe, G., and Kalousis, A. Calibrating neural simulation-based inference with differentiable coverage probability. *Advances in Neural Information* Processing Systems, 36, 2024.

Frazier, D. T., Robert, C. P., and Rousseau, J. Model misspecification in approximate bayesian computation: consequences and diagnostics. Journal of the Royal Statistical Society Series B: Statistical Methodology, 82(2): 421–444, 2020.

Frazier, D. T., Kohn, R., Drovandi, C., and Gunawan, D.

Reliable bayesian inference in misspecified models. arXiv preprint arXiv:2302.06031, 2023.

Frogner, C., Zhang, C., Mobahi, H., Araya, M., and Poggio, T. A. Learning with a wasserstein loss. In *Advances* in Neural Information Processing Systems, volume 28.

Curran Associates, Inc., 2015.

Gamella, J. L., Peters, J., and Bühlmann, P. Causal chambers as a real-world physical testbed for AI methodology. *Nature Machine Intelligence*, 2025. doi: 10.1038/ s42256-024-00964-x.

Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., March, M., and Lempitsky, V. Domainadversarial training of neural networks. Journal of machine learning research, 17(59):1–35, 2016.

Gao, R., Deistler, M., and Macke, J. H. Generalized bayesian inference for scientific simulators via amortized cost estimation. Advances in Neural Information Processing Systems, 36:80191–80219, 2023.

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., and Wichmann, F. A. Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11):665–673, 2020.

Glöckler, M., Deistler, M., and Macke, J. H. Variational methods for simulation-based inference. In International Conference on Learning Representations 2022, 2022.

Hashemi, M., Vattikonda, A. N., Jha, J., Sip, V., Woodman, M. M., Bartolomei, F., and Jirsa, V. K. Simulation-based inference for whole-brain network modeling of epilepsy using deep neural density estimators. *medRxiv*, pp. 2022– 06, 2022.

Hermans, J., Begy, V., and Louppe, G. Likelihood-free mcmc with amortized approximate ratio estimators. In International conference on machine learning, pp. 4239–
4248. PMLR, 2020.

Hermans, J., Delaunoy, A., Rozet, F., Wehenkel, A., and Louppe, G. A crisis in simulation-based inference? beware, your posterior approximations can be unfaithful. Transactions on Machine Learning Research, 2022.

Huang, D., Bharti, A., Souza, A., Acerbi, L., and Kaski, S. Learning robust statistics for simulation-based inference under model misspecification. arXiv preprint arXiv:2305.15871, 2023.

Jiang, Y., Yin, S., Dong, J., and Kaynak, O. A review on soft sensors for monitoring, control, and optimization of industrial processes. *IEEE Sensors Journal*, 21(11): 12868–12881, 2021. doi: 10.1109/JSEN.2020.3033153.

Kelly, R. P., Nott, D. J., Frazier, D. T., Warne, D. J., and Drovandi, C. Misspecification-robust sequential neural likelihood. *arXiv preprint arXiv:2301.13368*, 2023.

Linhart, J., Rodrigues, P. L. C., Moreau, T., Louppe, G., and Gramfort, A. Neural posterior estimation of hierarchical models in neuroscience. In GRETSI 2022-XXVIIIème Colloque Francophone de Traitement du Signal et des Images, 2022.

Long, M., Cao, Y., Wang, J., and Jordan, M. Learning transferable features with deep adaptation networks. In International conference on machine learning, pp. 97– 105. PMLR, 2015.

Lückmann, J.-M. Simulation-Based Inference for Neuroscience and Beyond. PhD thesis, Universität Tübingen, 2022.

Lueckmann, J.-M., Goncalves, P. J., Bassetto, G., Öcal, K.,
Nonnenmacher, M., and Macke, J. H. Flexible statistical inference for mechanistic models of neural dynamics. Advances in neural information processing systems, 30, 2017.

Lueckmann, J.-M., Boelts, J., Greenberg, D., Goncalves, P.,
and Macke, J. Benchmarking simulation-based inference. In International Conference on Artificial Intelligence and Statistics, pp. 343–351. PMLR, 2021.

Makkuva, A., Taghvaei, A., Oh, S., and Lee, J. Optimal transport mapping via input convex neural networks. In International Conference on Machine Learning, pp. 6672– 6681. PMLR, 2020.

Matsubara, T., Knoblauch, J., Briol, F.-X., and Oates, C. J.

Robust generalised bayesian inference for intractable likelihoods. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(3):997–1022, 2022.

Melis, A. Gaussian process emulators for 1d vascular models, 2017. URL https://etheses.whiterose. ac.uk/19175/.

Mensch, A. and Peyré, G. Online sinkhorn: Optimal transport distances from sample streams. Advances in Neural Information Processing Systems, 33:1657–1667, 2020.

Nott, D. J., Drovandi, C., and Frazier, D. T. Bayesian inference for misspecified generative models. Annual Review of Statistics and Its Application, 11, 2023.

Pacchiardi, L. and Dutta, R. Generalized bayesian likelihood-free inference using scoring rules estimators.

arXiv preprint arXiv:2104.03889, 2(8), 2021.

Papamakarios, G. and Murray, I. Fast ε-free inference of simulation models with bayesian conditional density estimation. *Advances in neural information processing* systems, 29, 2016.

Papamakarios, G., Sterratt, D., and Murray, I. Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 837–848.

PMLR, 2019.

Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., and Lakshminarayanan, B. Normalizing flows for probabilistic modeling and inference. The Journal of Machine Learning Research, 22(1):2617–2680, 2021.

Perera, Y. S., Ratnaweera, D., Dasanayaka, C. H., and Abeykoon, C. The role of artificial intelligence-driven soft sensors in advanced sustainable process industries:
A critical review. Engineering Applications of Artificial Intelligence, 121:105988, 2023.

Peyré, G., Cuturi, M., et al. Computational optimal transport.

Center for Research in Economics and Statistics Working Papers, 2017.

Rabin, J., Ferradans, S., and Papadakis, N. Adaptive color transfer with relaxed optimal transport. In 2014 IEEE international conference on image processing (ICIP), pp. 4852–4856. IEEE, 2014.

Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., and Köthe, U. Bayesflow: Learning complex stochastic models with invertible neural networks. *IEEE transactions on* neural networks and learning systems, 33(4):1452–1466, 2020.

Rubin, D. B. Bayesianly justifiable and relevant frequency calculations for the applied statistician. *The Annals of* Statistics, pp. 1151–1172, 1984.

Schmitt, M., Bürkner, P.-C., Köthe, U., and Radev, S. T.

Detecting model misspecification in amortized bayesian inference with neural networks. In DAGM German Conference on Pattern Recognition, pp. 541–557. Springer, 2023.

Schmon, S. M., Cannon, P. W., and Knoblauch, J. Generalized posteriors in approximate bayesian computation. arXiv preprint arXiv:2011.08644, 2020.

Tabak, E. G. and Vanden-Eijnden, E. Density estimation by dual ascent of the log-likelihood. Communications in Mathematical Sciences, 8(1):217–233, 2010.

Takeishi, N. and Kalousis, A. Physics-integrated variational autoencoders for robust and interpretable generative modeling. Advances in Neural Information Processing Systems, 34:14809–14821, 2021.

Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W.,
and Abbeel, P. Domain randomization for transferring deep neural networks from simulation to the real world. In 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS), pp. 23–30. IEEE, 2017.

Tolley, N., Rodrigues, P. L., Gramfort, A., and Jones, S. R.

Methods and considerations for estimating parameters in biophysically detailed neural models with simulation based inference. *bioRxiv*, pp. 2023–04, 2023.

Villani, C. et al. *Optimal transport: old and new*, volume 338. Springer, 2009.

Wang, Y., Yao, Q., Kwok, J. T., and Ni, L. M. Generalizing from a few examples: A survey on few-shot learning.

ACM computing surveys (csur), 53(3):1–34, 2020.

Ward, D., Cannon, P., Beaumont, M., Fasiolo, M., and Schmon, S. Robust neural posterior estimation and statistical model criticism. Advances in Neural Information Processing Systems, 35:33845–33859, 2022.

Wehenkel, A. and Louppe, G. Unconstrained monotonic neural networks. Advances in neural information processing systems, 32, 2019.

Wehenkel, A., Behrmann, J., Hsu, H., Sapiro, G., Louppe, G., and Jacobsen, J.-H. Robust hybrid learning with expert augmentation. Transaction on Machine Learning Research, 2022.

Wehenkel, A., Behrmann, J., Miller, A. C., Sapiro, G.,
Sener, O., Cuturi, M., and Jacobsen, J.-H. Simulationbased inference for cardiovascular models. arXiv preprint arXiv:2307.13918, 2023.

Wrede, F., Eriksson, R., Jiang, R., Petzold, L., Engblom, S., Hellander, A., and Singh, P. Robust and integrative bayesian neural networks for likelihood-free parameter inference. In 2022 International Joint Conference on Neural Networks (IJCNN), pp. 1–10. IEEE, 2022.

Yin, Y., Le Guen, V., Dona, J., de Bézenac, E., Ayed, I.,
Thome, N., and Gallinari, P. Augmenting physical models with deep networks for complex dynamics forecasting.

Journal of Statistical Mechanics: Theory and Experiment, 2021(12):124012, 2021.

Zhu, X. J. Semi-supervised learning literature survey. 2005.

## A **Model Misspecification** A.1 **Mis-Calibration Vs Misspecification**

To further elucidate the distinction between posterior calibration and model misspecification, it is essential to highlight their respective scopes and the specific challenges they address. Posterior calibration focuses on ensuring that the predicted posterior distributions accurately reflect the true uncertainty in parameter estimates given the observations, under the assumption that the simulator is well-specified. Methods such as those proposed by Falkiewicz et al. (2024); Delaunoy et al. (2022) address this by improving the alignment between the expected and actual coverage probabilities of the posterior. These approaches generally assume that the simulator faithfully represents the generative process of the observed data, enabling calibration to be evaluated and improved by leveraging simulations. While important, these methods do not account for discrepancies between the simulator and real-world data, which are precisely the scenarios we target in this work.

Model misspecification, on the other hand, arises when the simulator fails to capture the true generative process underlying the observed data. This results in systematic discrepancies that cannot be corrected solely by optimizing posterior calibration techniques. Misspecification introduces a gap between the simulated and real-world distributions, and this gap is only observable when real-world data is available. Unlike posterior calibration, addressing misspecification requires methods that can robustly leverage the simulator despite its inaccuracies, while incorporating real-world observations to mitigate the impact of the mismatch. In our work, we explicitly focus on handling model misspecification. This distinction is reflected in the design of our approach and the evaluation scenarios we consider, such as Task E, where the simulated data diverges significantly from the real-world measurements. While posterior calibration methods may perform well in a well-specified context, they are not designed to cope with such gaps. Instead, we prioritize creating predictive models that balance informativeness and robustness in the presence of misspecification, even if achieving perfect calibration remains an open and challenging problem.

## A.2 **Comparison Between Model Misspecification Definitions**

We provide a toy example to show how a simulator may be well-specified according to the standard definition of misspecification but still provide biased estimates of the target parameter when applied to real data. Consider the following setting: a noisy sensor measures some physical quantity θ, producing measurements x 1o*, . . . ,* x n o i.i.d. ∼
P

⋆, where P
⋆:= N (θ
⋆, 1) is a normal distribution centered around the 'true' value θ
⋆. Let {Pθ : θ ∈ R} be a simulator of this process with Pθ := N (µ, 1), where µ := θ + λ and λ > 0 is a fixed scalar constant, which is a misspecification in the simulator that falsely accounts for a non-existing offset in the sensor that produced the real observations x 1 o*, . . . ,* x n o .

According to the standard definition of misspecification, the simulator is well specified, as setting θ ← θ
⋆ − λ yields Pθ = P
⋆.

However, the posterior estimates we obtain with this simulator are biased with respect to the true parameter θ
⋆.

To see this, let us compute the posterior under a Gaussian prior N (θ
⋆, 1) over the parameter θ, centered on the true value θ
⋆.

Taking advantage of the conjugate prior, the posterior p(θ | x 1o*, . . . ,* x n o ) becomes

p(θ | x
1o*, . . . ,* x
n
o ) ∝ *p(θ)p*(x
1o*, . . . ,* x
n
o | θ)
= p(θ)Y
n
i=1
p(x
io | θ)
=1
√2π
exp −
1
2
(θ − θ
⋆)
2 Yn
i=1
1
√2π
exp −
1
2
(x
io − µ)
2
∝ exp −
1
2
(θ − θ
⋆)
2 −
1
2
Xn
i=1
(x
io − µ)
2
!
= exp
 
−
1
2
"
θ
2 + (θ
⋆)
2 − 2θθ⋆ +Xn
i=1
(x
io)
2 + nµ
2 − 2µXn
i=1
x
io
#!
(drop const. terms) ∝ exp −
1
2
"
θ
2 − 2θθ⋆ + nµ
2 − 2µX
n
i=1
x
io
#!
(µ = θ + λ) = exp −
1
2
"
θ
2 − 2θθ⋆ + nθ2 + nλ2 + 2nλθ − 2θX
n
i=1
x
io − 2λX
n
i=1
x
io
#!
(drop const. terms) ∝ exp −
1
2
"
θ
2 − 2θθ⋆ + nθ2 + 2nλθ − 2θ
Xn
i=1
x
io
#!
= exp
 
−
1
2
"
(n + 1)θ
2 − 2θ(θ
⋆ − nλ +Xn
i=1
x
io)
#!
= exp −1
2(n + 1)−1
"
θ
2 − 2θ
1
n + 1
(θ
⋆ − nλ +X
n
i=1
x
io)
#!
(complete square) ∝ exp

−
1
2(n + 1)−1
"
θ −
1
n + 1
(θ
⋆ − nλ +Xn
i=1
x
io)
#2
 ,
that is, a normal distribution N (*τ, γ*2) with mean

$$\tau=\left({\frac{1}{1+n}}\right)\left(\theta^{\star}-n\lambda+\sum_{i=1}^{n}{\bf x}_{o}^{i}\right)$$

and variance γ 2 = (n + 1)−1. Thus, the posterior is biased, e.g., the posterior mean τ is a biased estimator of θ
⋆ with E[θ
⋆ − τ ] = θ
⋆ − λ n n+1.

Problem setup **ROPE** algorithm simulated posterior NPE
NSE
1 Simulated data Real observations Reality
(unknown)
Calibration set fine tuning 2 Physical parameters Misspecification OT solution Test set NSE
Inference pipeline Simulated observations Simulator

## B **The Rope Algorithm**

Algorithm 1 Posterior Inference using Robust Neural Posterior Estimation (RoPE)
Input: Simulator *S(θ, ε*), prior distribution p(θ), calibration set C = {(x io, θi)}
Nc i=1, test set D = {x io}
No i=1 Output: p˜(θ | xo)∀x io ∈ D
Step 1: Neural Posterior Estimation (NPE)
Train neural network hω and conditional normalizing flow p(θ | ·) using NPE:
p, ω ˜
⋆ = arg max p,ω E θ∼π(θ)
ε∼U[0,1]
[log p(θ | hω(S(θ, ϵ)))]
Step 2: Fine-tune sufficient statistics hω⋆ **on the Calibration Set**
gψ := COPY(hω⋆ ) Ctrain, Cval = RandomSplit(C,
1 5
)
bestval = ∞ for Niter do ψ ← ψ − α∇ψ hP(θ,xo)∈C*train* |gψ(xo) − Eε[hω⋆ (S(*θ, ε*))]|2 i curval =P(θ,xo)∈Cval |gψ(xo) − Eε[hω⋆ (S(*θ, ε*))]|2 if curval < bestval **then**
bestval = curval ψ
⋆ = ψ end if end for Step 3: Generate Simulations for Test Set (Ns = No)
S = {x js}
Ns j=1, where x js ∼ S(θ j*, ε)* θ j ∼ π(θ) ε ∼ U[0, 1]
Step 4: Entropic-regularized OT

Cij =|fω⋆ (x
$$f_{\omega^{\ast}}(\mathbf{x}_{s}^{j})-g_{\psi^{\ast}}(\mathbf{x}_{o}^{i})|\quad\forall$$
$$C_{i j}$$
$\uparrow\downarrow$ . 
io)| ∀i, j ∈ {1, . . . , No} × {1*, . . . , N*s}
P
⋆ = arg min
P ∈Bo
$\left({C\left)+\rho\,KL\left({P}\right.}\right)$
T1No
1Ns
Ns
+ γ⟨P, log P⟩
$\mathbf{1}_{N_0}\mathbb{I}$
Step 5: Compute Posterior Distributions

$$p(\theta|\mathbf{x}_{o}^{i}):=\sum_{j=1}^{N_{s}}P_{i j}^{\star}{\bar{p}}\left(\theta\mid\mathbf{h}_{\omega^{\star}}(\mathbf{x}_{s}^{j})\right)$$
$$\vec{J}$$
$+\gamma\langle P,\log\gamma\rangle$
Return *p˜(θ*|x io) ∀x io ∈ D

## C **Prior Misspecification Experiments**

Prior misspecification on Task C. With this experiment we aim to better understand the role of τ when RoPE is applied with different levels of prior misspecification. We thus re-use the same setup as in Figure 1 but add prior misspecification as a mixture between the assumed prior and a much tighter uniform distribution. As the weight of the tighter uniform distribution increases, the prior gets more misspecified. The experimental setup follows closely the one in the well-specified case (see Section I.2), except calibration samples are drawn from the true prior (as this would be the case in a real-world application) and we compute the OT coupling for values of τ ∈ [0.1, 1]. The results in Figure 3b demonstrate that RoPE can be robust to prior misspecification. In particular, we observe that τ plays the expected role and that values below 1. enable RoPE to perform better when the true prior is only a subset of the prior used to generated synthetic data. Prior misspecification on Task E. In some practical settings, it is unlikely that the prior used to generate synthetic data will match the distribution of the target parameters in the real data. For this reason, we consider a semi-balanced formulation of OT, providing the flexibility to discard simulations with no corresponding real-world observations. To evaluate the effect of a misspecified prior on RoPE and RoPE⋆, we perform an experiment that would resemble its use in real applications like the ones we outline in the introduction. In such settings—e.g., inferring cardiac parameters or chemical concentrations—the target parameters are limited to a range of validity, and a likely choice for the practitioner would be to select a uniform prior over this range. To replicate this setting, we collect a new real-world dataset from the light tunnel (Task E) and train RoPE on synthetic data originating from a uniform prior, as we do for the results shown in Figure 1. However, we then apply RoPE to real data generated from a different (betabinomial) distribution over the target parameters.

Task E Light Tunnel Out-of-distribution performance Training distribution Target distribution ROPE
J -NPE
NPE ROPE
Training distribution Target distribution flipped images OT-only tuning-only rea l o bs erva tion s MLP
LPP
B
si m. ob serva tion s ACAUC

## D **Robustness To Distribution Shifts**

Figure 6: Out-of-distribution performance of RoPE and some baselines. We train RoPE and other baselines on the same light-tunnel data as in task E (training distribution), but apply it to test sets originating from a target distribution where the real-world images are flipped vertically. We compare the performance on test sets from both distributions, showing the LPP and ACAUC scores for each method. For comparison, in the right plot we show again the LPP curve (light gray, dotted) attained by RoPE under the training distribution. The performance of RoPE is barely affected as it cannot exploit any signal in the real images (xo) beyond what is encoded in the simulator, and the simulator output (xs) is invariant to the transformation we consider. Because NPE is not trained on real observations, its performance, although poor, also remains virtually unchanged. On the other hand, the performance of MLP and J-NPE drops in the target distribution, as these methods are not limited in what information they can exploit from the real observations on which they are trained, potentially learning shortcuts that are not present in the target distribution. This results demonstrate that if the simulator embeds the right invariances, our modeling assumption xo ⊥ θ | xs can be favorable to out-of-distribution generalization.

## E **Optimal Transport Coupling As A Joint Distribution**

With our conditional independence assumption, the problem of modeling p(xo | θ) reduces to modeling p(xo | xs) instead. If we assume the prior well-specified, this task is equivalent to modeling p(xo, xs) under the constraint that the corresponding marginal p(xs) = Rp(xs, xo)dxo equals Rp(θ)p(xs | *θ)dθ*. By construction, the OT coupling, π
⋆, respects the constraint on the marginals, Rπ
⋆(xs, xo)dxo = p(xs) and Rπ
⋆(xs, xo)dxs = p(xo) , and the exact instantiation π
⋆depends also on the chosen cost function which can always be defined to yield any given conditional p(xo | xs) that respects the constraint Rp(xo | xs)p(xs)dxs = p(xo). π
∗can thus model the "right" posterior, provided the right cost function is used. In the case, where the prior cannot be trusted, we suggest to use τ < 1 and relax the OT formulation. In this case, we only enforce that all elements of p(xo) are matched to a subset of the elements of p(xs). This implicitly assumes that the assumed prior p(θ) is overly conservative and covers p
⋆(θ). We believe this is a reasonable assumption as it is often easy to derive physical bounds for the parameter values and use a uniform distribution.

## F **Self-Calibration Property**

We say RoPE is self-calibrating because, by design, the posterior distribution marginalized over observations tends to the prior as the number of simulation increases, that is,

$$\int_{\mathcal{X}}\bar{p}(\theta\mid\mathbf{x}_{o})p(\mathbf{x}_{o})d\mathbf{x}_{o}=p(\theta).\tag{1}$$

This property is also called marginal calibration, and is a necessary condition for a posterior estimation method to be calibrated. Considering NPE, *p˜(θ* | xs), is marginally calibrated and observations xo are generated from the assumed prior, that is sampled from an unknown distribution p(xo) = Rp(xo | θ)p(θ), we can show RoPE is marginally calibrated. Indeed, considering the Monte-Carlo approximation of the marginalized posterior distribution over the test set Do := {x io}
No i=1, we have,

Z X p˜(θ | xo)p(xo)dxo = Ep(xo)[˜p(θ | xo)] (8) ≈1 No X No i=1 p˜(θ | x io) (9) =1 No X No i=1 X Ns j=1 NoP ⋆ ijp˜(θ | x js) (10) =X Ns j=1  "X No i=1 P ⋆ ij # p˜(θ | x js) (11) =1 Ns X Ns j=1 p˜(θ | x js) (12) ≈ p(θ), (13)
$$\left(T\right)$$
$$({\mathfrak{s}})$$
$$(9)$$
(10)  $\binom{11}{2}$  (11)  ... 
$$(12)$$
$$(13)$$
where we use the definition of the transport matrix to get PNo i=1 P
⋆
ij =
1 Ns
. The last approximation tends to be exact as the number of simulations increases, if the NPE is marginally calibrated.

## G **Learning Minimal Sufficient Statistics With Neural Posterior Estimation**

We now discuss why NPE may learn a minimal sufficient statistic under perfect training. First, under a sufficiently large validation set, NPE's objective function is only optimal on the validation set if NPE models the true posterior as defined implicitly by the prior p(θ) and the likelihood corresponding to the simulator S. This consistency has been proven in (Papamakarios & Murray, 2016) and is the motivation to use such an objective when estimating density. Second, some normalizing flows, such as autoregressive UMNN flows (Wehenkel & Louppe, 2019), are universal approximators of continuous densities. In addition, neural networks are also universal function approximators. As such, we can claim that it is always possible to parameterize the NCDE pθ(θ | hω(x)) such that the class of functions its parameters represent contains the true posterior. We directly observe that x is only used by the NCDE through hω(x). Thus, under perfect training pθ
⋆ (θ | hω⋆ (x)) = p(θ | x) and hω⋆ (x) is a sufficient statistic for θ given x under the simulator's model.

Without additional constraints, we cannot claim anything about the minimality of hω⋆ (x). Nevertheless, we can enforce the neural network hω⋆ (x) to have an information bottleneck and thus reduce the information carried. In practice, we choose the output dimension of hω⋆ (x) so that the NCDE achieves optimal performance on the test set. Because in the context of SBI
we can generate as many (simulated) samples as needed, we can obtain estimators that closely approach the simulation's posterior and a minimal sufficient statistic.