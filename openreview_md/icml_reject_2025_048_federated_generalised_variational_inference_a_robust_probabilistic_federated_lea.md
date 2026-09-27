# Federated Generalised Variational Inference: A Robust Probabilistic Federated Learning Framework

Terje Mildner 1 Oliver Hamelijnck 1 Paris Giampouras 1 **Theodoros Damoulas** 1 2

## Abstract

We introduce FEDGVI, a probabilistic Federated Learning (FL) framework that is robust to both prior and likelihood misspecification. FEDGVI addresses limitations in both frequentist and Bayesian FL by providing unbiased predictions under model misspecification, with calibrated uncertainty quantification. Our approach generalises previous FL approaches, specifically Partitioned Variational Inference (Ashman et al., 2022), by allowing robust and conjugate updates, decreasing computational complexity at the clients. We offer theoretical analysis in terms of fixed-point convergence, optimality of the cavity distribution, and provable robustness to likelihood misspecification. Further, we empirically demonstrate the effectiveness of FEDGVI in terms of improved robustness and predictive performance on multiple synthetic and real world classification data sets.

## 1. Introduction

Federated learning (FL) is a framework for the collaborative training of a global model by a collection of clients, without requiring proprietary data to be shared with a central server or other participating clients (McMahan et al., 2017). This decentralised approach allows FL to be used on applications with strict data privacy constraints, such as in finance or healthcare (Kairouz et al., 2021). However, due to the sensitive nature and complexity of these domains, both privacy and robustness to model misspecification are paramount. The frequentist formulation of FL aims to minimise a global loss function by aggregating local gradients from clients. Early works include Federated Averaging (FEDAVG, McMahan et al., 2017) which iterates between training clients lo1University of Warwick, Department of Computer Science, Coventry, United Kingdom 2University of Warwick, Department of Statistics, Coventry, United Kingdom. Correspondence to: Terje Mildner <Terje.Mildner@warwick.ac.uk>.

1 cally and averaging updates on the server. This has sparked a large body of research on issues such as communication efficiency, data privacy, and data heterogeneity across clients (Hamer et al., 2020; Malinovsky et al., 2020; Reddi et al., 2021; Chen et al., 2022; Tenison et al., 2023; Tziotis et al., 2023; Li et al., 2024; Demidovich et al., 2025). There has been some work addressing robustness to adversarial clients (Allouah et al., 2024; Bao et al., 2024) and data and system heterogeneity (Chen et al., 2022; Zhao et al., 2023; Heikkila¨ et al., 2023). However, these only provide point estimates, and do not allow principled uncertainty quantification, as required in many FL applications (Jonker et al., 2024). In contrast, Bayesian FL approaches aim to update beliefs of a global model with data partitioned across clients. This largely builds on distributed inference methods such as the Bayesian Committee Machine (Tresp, 2000), parallel MCMC (Ahn et al., 2014; Mesquita et al., 2020), or Divide&Conquer SMC (Chan et al., 2023). Expectation Propagation (Minka, 2001; Vehtari et al., 2020) is naturally applicable to the distributed setting where local sites are iteratively refined. This requires computing the cavity distribution that removes local sites from the current approximation. Partitioned Variational Inference (PVI, Bui et al.,
2018; Ashman et al., 2022) takes this idea and proposes a distributed variational inference algorithm, which has been extended through MCMC (Guo et al., 2023) and Stochastic Gradient Langevin Dynamics (SGLD) (Mekkaoui et al., 2021). Whilst these approaches quantify uncertainty, they are susceptible to model misspecification which can lead to inaccurate, overconfident predictions (Bernardo & Smith, 2000; Bissiri et al., 2016; Knoblauch et al., 2022). Current approaches to FL are inherently non-robust to model misspecification which leads to compromised performance and uncalibrated uncertainty quantification. We address these challenges by departing from the traditional Bayesian paradigm and propose a distributed Generalised Variational Inference framework that allows us to deal with model misspecification. In summary, our contributions are:
- We prove that FEDGVI is robust to likelihood misspecification (Theorem 4.12).

- We demonstrate that FEDGVI generalises standard approaches such as PVI and FEDAVG (Remarks 4.1 and 4.2) and theoretically justify the use of the cavity distribution (Theorem 4.9).

- We prove that, under suitable conditions, FEDGVI converges to Generalised Bayesian posteriors (Lemma 4.6 and Proposition 4.10) that are computationally tractable.

- We evaluate FEDGVI on a range of synthetic and realworld datasets, across multiple models, demonstrating improved robustness and predictive performance.

In Section 2 we define model misspecification and recall methods that mitigate it in the non–distributed setting. Section 3 introduces our framework, which builds on these concepts and extends them to the federated setting. We analyse the theoretical properties of FEDGVI in Section 4, including provable robustness. Finally, Section 5 studies the empirical performance and gains of FEDGVI with multiple models and real world datasets such as Bayesian Neural Networks on MNIST and FASHIONMNIST. 1

## 1.1. Related Work

Robust Frequentist Federated Learning In the frequentist setting, building on the seminal paper of McMahan et al. (2017), many approaches have aimed at mitigating challenges in FL, such as robustness to adversarial servers through secure aggregation (Chen et al., 2022), to stragglers (Tziotis et al., 2023), heterogenous data in out–of–
distribution generalisation (Tenison et al., 2023), heterogeneous and asynchronous clients (Fraboni et al., 2023), or finding weaknesses in communications (Zhu et al., 2019; Zhao et al., 2023). More recently, work on robust server aggregations achieves robustness against Byzantine clients that aim to deteriorate model performance (Allouah et al., 2024; Bao et al., 2024). However these do not allow principled uncertainty quantification. Federated Bayesian Inference Federated and distributed Bayesian methods aim to approximate the posterior as if it had been computed with the data of all clients available at a central server. Early work on distributed Bayesian inference includes Bayesian opinion pools (Genest, 1984; Carvalho et al., 2023), and the Bayesian Committee machine (Tresp, 2000), which aim to find a consensus among a collection of Bayesian beliefs. Works that aim to operationalise this in the distributed setting, where data is split IID across clients, 1Code to reproduce experiments can be found at https://
github.com/Terje-M/FedGVI.

include Expectation Propagation (Minka, 2001; Opper & Winther, 2005; Hasenclever et al., 2017; Vehtari et al., 2020), and consensus based Monte Carlo (Scott et al., 2016). In the Federated setting this assumption is often violated, as data is not split homogeneously and IID across participating devices. From this perspective, most approaches to Bayesian FL can be categorised into finding an approximate posterior through variational inference (Corinzia et al., 2021; Ashman et al., 2022; Kassab & Simeone, 2022; Heikkila et al. ¨ , 2023; Hassan et al., 2024; Vedadi et al., 2024; Swaroop et al., 2025), Markov Chain Monte Carlo (Al-Shedivat et al., 2021; Mekkaoui et al., 2021; Kotelevskii et al., 2022; Guo et al., 2023; Hasan et al., 2024), Gaussian Processes (Achituve et al., 2021), or directly learning a Bayesian neural network (Yurochkin et al., 2019; Zhang et al., 2022). Personalised or hierarchical Bayesian FL (Kotelevskii et al., 2022; Zhang et al., 2022; Kim & Hospedales, 2023; Hassan et al., 2023; 2024; Vedadi et al., 2024) allows for additional expressibility of client posteriors, especially under heterogeneity. However, none of these are inherently robust to contamination and model misspecification. Robust Bayesian Inference Although the existing Bayesian FL methods address some of the challenges of federated learning, such as communication constraints and data heterogeneity, they still aim to approximate the Bayesian posterior, which in itself is a flawed objective under model misspecification (Walker, 2013; Berk, 1966; Bernardo & Smith, 2000). In the global, non-federated case, several methods have been proposed to combat misspecification in the Bayesian setting (Grunwald ¨ , 2012), with the most promising direction being Generalised Bayesian Inference
(Hooker & Vidyashankar, 2014; Bissiri et al., 2016; Ghosh
& Basu, 2016a; Jewson et al., 2018; Miller, 2021; Alquier, 2021; Knoblauch et al., 2022; Matsubara et al., 2022). In this work we capitalise on this front and bring robustness to model misspecification in the federated setting.

## 2. Preliminaries 2.1. Notation And Model Misspecification

Let (Ω, F, P0) be a probability space where P0 is the data generating process, generating the observable random variables X1*, ..., X*n ≡ Xn 1taking values in the measurable space (Ξ, X ). Further, let Y
n 1 be observable random variables depending on Xn 1respectively, taking values in (Υ, Y).

Denote their realisations {Xi = xi, Yi = yi}
n i=1, which are assumed to be partitioned across M clients {xm, ym}Mm=1 each of size nm. Consider hypothesis measures Pθ where θ takes values in (Θ, T ), a measurable space, admitting densities pθ. We study elements of P(Θ), the set of all probability measures on (Θ, T ), starting with prior Π and updated to Q, dominated by some common measure µ, and

| Algorithm 1 FEDGVI SERVER 1: Input: π(θ), Q, Ds (0) (0)                                                                                                                                                                                 | (0)          |            |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|------------|
| 2: Define: ℓ m (θ) = 0, ℓ s                                                                                                                                                                                                             | (θ) = 0, q s | (θ) = π(θ) |
| 3: for t = 1, ..., T do 4: for m = 1, ..., M in parallel do 5: ∆ (t) m (θ) ←CLIENT(q s (θ), Q, m) (t−1) 6: end for (t) (t−1) s (θ) + PM (t) 7: Set ℓ s (θ) ← ℓ m=1 ∆ m (θ) (t) 8: Optimise q s (θ) according to Equation (7) 9: end for |              |            |

admitting densities π and q respectively. Naive Bayes updates π(θ) to qB(θ) through

$$q_{B}(\mathbf{\theta})=\pi(\mathbf{\theta})\prod_{m=1}^{M}p_{\mathbf{\theta}}(\mathbf{y}_{m};\mathbf{x}_{m})\,/Z$$
m=1 pθ(ym; xm) /Z (1)
where Z =RΘ
QM
m=1 pθ(ym; xm) Π(dθ) is the marginal likelihood. Since we do not suppose that the prior Π, nor the likelihood Pθ are well specified, i.e. P0 ∈ P/ (Θ), we are in the M–open setting (Bernardo & Smith, 2000), the model misspecified, and the Bayesian posterior inappropriate.

## 2.2. Model Misspecification

There are several different ways we can think about model misspecification under the M–open assumption. Prior Misspecification The traditional Bayesian paradigm assumes that the prior encodes the best available judgement about θ, which beyond simple settings, is never realised (Berger, 1985; Knoblauch et al., 2018). Such misspecification is common; e.g. it is standard to use zero–mean Gaussian distributions on the weights of Bayesian Neural networks. This can have dire effects, for instance Diaconis & Freedman (1986) demonstrate that multimodal priors in a location model can cause the posterior to not accumulate around P0, even when the DGP is well specified, i.e. when P0 ∈ P(Θ).

Likelihood Misspecification One such example is where the hypothesis of interest is contaminated , and an ε fraction of the data (input and/or output variables) has some unknown data source. Formalising this we follow the definition of Huber (1964): Definition 2.1 (Huber contamination). Given an ε ∈ (0, 1 2
)
and the uncontaminated distribution Pθ of inliers and some contaminating distribution G of outliers, then P0 is said to be an ε*-corrupted version of* Pθ; P0 := (1 − ε)Pθ + εG.

## 2.3. Robust Bayesian Methods

Generalised Bayesian Inference (GBI) Instead of linking the parameter and data through likelihoods, Bissiri et al.

(2016) and Miller (2021) formalised a coherent Bayesian framework using loss functions leading to Gibbs posteriors (Alquier et al., 2016). This was further utilised to deal with likelihood misspecification through robust losses, e.g Knoblauch et al. (2018). Let L : Θ × Ξ × Υ → R be such a loss, then the GBI posterior is given by:

$\left(\mathrm{I}\right)$. 
$\mathbf{a}=\mathbf{a}\cdot\mathbf{a}$. 
$$q_{\mathrm{GBI}}(\mathbf{\theta})=\pi$$
qGBI(θ) = π(θ) exp n−βPM

$$\left.{}_{,1}L(\mathbf{y}_{m};{\boldsymbol{\theta}},\mathbf{x}_{m})\right\}/Z\;\;(2)$$

| Algorithm 2 FEDGVI CLIENT (t−1)                                                                                                                                                                                                           | (t−1)                     |        |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|--------|
| 1: Input: q s                                                                                                                                                                                                                             | (θ), Q, {xm, ym}, Lm, ℓ m | (θ), D |
| 2: Optimise q \m(θ) according to Equation (3) (t) 3: Optimise q m (θ) according to Equation (4) 4: Set ∆ (t) m (θ) according to Equation (5) 5: Set ℓ (t) m (θ) ← ℓ (t−1) m (θ) + ∆(t) m (θ) (t) 6: return: Communicate ∆ m (θ) to SERVER |                           |        |

with Z =RΘ
exp{−βPM
m=1L(ym; θ, xm)}Π(dθ). Here, β ∈ R>0 is a learning rate parameter that determines how much weight we place on the observed data, similar to power posteriors in VI (Grunwald ¨ , 2012; Kallioinen et al.,
2024). This recovers qB(θ) when the loss is the negative log–likelihood and β = 1. Generalised Variational Inference (GVI) In Knoblauch et al. (2022) GBI is generalised within a variational framework that explicitly accounts for prior and likelihood misspecification. Let D : P(Θ)×P(Θ) → R+ be a divergence then the GVI posteriors are defined as:
qGVI(θ) = arg min q∈Q
Eq(θ)
-L(y M
1; θ, x M
1)+ D(q : π)	
where *Q ⊂ P*(Θ), making inference tractable. This allows for targeting a larger subspace of posteriors, and through different divergences the effect of the prior can be controlled.

## 3. **Federated Generalised Variational Inference** 3.1. Methodology

In this section, we present the proposed federated learning framework, named FEDGVI, that explicitly addresses likelihood and prior misspecification. We aim to learn a robust approximate posterior qs(θ) using partitioned observations across M clients. FEDGVI iterates consist of two steps: a) sending of the current approximate posterior to each client, which is updated through a robust variational objective, and b) aggregating the updates on the server, resulting in a robust approximate posterior; summarised in Algorithms 1 and 2. Initialisation We set the initial server posterior as the prior, q
(0)
s (θ) = π(θ), and the local and server loss approximations to be zero, ℓ
(0)
m (θ) = 0 and ℓ
(0)
s (θ) = 0 respectively; m denotes a specific client and s the server. Until Convergence For t = 1, 2*, ..., T*, we synchronously compute updates locally at each client, and accumulate these at the server to form the new global posterior q
(t)
s (θ).

Client The client receives the current approximate posterior from the server. This will be used as the prior from which a client can compute an updated posterior using their local data. First, however the information of the client's data must be removed by computing the cavity distribution. The cavity distribution acts as the local prior incorporating all previous information from all other clients and is given by:

$$q^{\backslash m}(\theta)\propto\frac{q_{s}^{(t-1)}(\theta)}{\exp\{-\ell_{m}^{(t-1)}(\theta)\}}$$
$$(3)$$

The client then computes a robust local approximate posterior with it's local data set {xm, ym} and it's loss function L
(t)
m (·), which is regularised by the divergence, D, and cavity distribution

$$q_{m}^{(t)}(\mathbf{\theta})=\operatorname*{arg\,min}_{q\in\mathcal{Q}}\mathbb{E}_{q(\mathbf{\theta})}\left[L_{m}^{(t)}(\mathbf{y}_{m};\mathbf{\theta},\mathbf{x}_{m})\right]+D(q:q^{\backslash m}).\tag{4}$$

This GVI style objective allows the client to be robust to both likelihood misspecification as well as prior misspecification arising due to the cavity. To update the global posterior at the server, the client computes the negative log ratio of the local and global posteriors. In line with existing Bayesian FL
(Ashman et al., 2022; Guo et al., 2023), we use a damping parameter τm ∈ (0, 1], which is analogous to a learning rate as in frequentist FL, to compute the update:

$$\Delta_{m}^{(t)}(\pmb{\theta})=-\tau_{m}\log\frac{q_{m}^{(t)}(\pmb{\theta})}{q_{s}^{(t-1)}(\pmb{\theta})}$$
$$({\boldsymbol{5}})$$

The client stores ℓ
(t)
m (θ) := ℓ
) :$=\ell_m^{(t-1)}(\theta)$
m (θ) + ∆(t)
m (θ) and communicates ∆
(t)
m (θ) to the server.

Server The loss at the server is updated based on the received client updates,

$$\ell_{s}^{(t)}(\mathbf{\theta})=\ell_{s}^{(t-1)}(\mathbf{\theta})+\sum_{m=1}^{M}\Delta_{m}^{(t)}(\mathbf{\theta})$$

By only incorporating clients' updates that have changed we can trivially allow for batched and asynchronous scheduling of clients. The updated loss is then used to compute the new server posterior though a GVI optimisation procedure:

$$q_{s}^{(t)}(\mathbf{\theta})=\operatorname*{arg\,min}_{q\in\mathcal{Q}}\mathbb{E}_{q(\mathbf{\theta})}\left[\ell_{s}^{(t)}(\mathbf{\theta})\right]+D_{s}(q:\pi)\quad\quad(7)$$

This posterior and loss are passed back to the clients for further refinement at the next iteration until convergence.

3.1.1. HYPERPARAMETERS
Ashman et al. (2022) set the damping parameter to τ ∝
1 M
throughout their experiments. This turns out, see Proposition 4.3, to be a reasonable choice when τ =
1 M in combination with Ds = DKL since this causes the posterior at the server to be a logarithmic opinion pool induced by an externally Bayesian pooling operator (Genest et al., 1986), ensuring stable convergence. Other hyperparameters arising from the choice of losses and divergences are dependent on the expected amount of model misspecification.

## 3.2. Robustness To Likelihood Misspecification

Within our framework we are free to choose the client side losses. We consider the Density–Power divergence based loss (Ghosh & Basu, 2016b), often referred to as β–
divergence loss Lβ, the γ–divergence based losses (Hung et al., 2018), Lγ, as well as a score matching loss, LSM,
based on the Hyvarinen divergence ( ¨ Hyvarinen ¨ , 2005; Altamirano et al., 2023). In the classification setting, we consider the generalised cross–entropy loss

$$\mathcal{L}_{GCE}^{(\delta)}(y_{i};\theta,x_{i})=\frac{(1-p_{\theta}(y=y_{i};x_{i})^{\delta})}{\delta}\tag{8}$$

for some δ ∈ (0, 1] (Zhang & Sabuncu, 2018). These losses are robust to misspecification because they have a finite supremum (see Definition 4.11). It is important to highlight that GVI and FEDGVI may underperform when using robust losses in the case of correct likelihood specification; see Knoblauch et al. (2022). We can use a Sequential Monte Carlo sampler to estimate the β or γ hyperparameters in Lβ and Lγ (Yonekura & Sugasawa, 2023) or use cross validation to select optimal parameters (Altamirano et al., 2024).

## 3.3. Robustness To Prior Misspecification

We mainly consider the weighted Kullback–Leiber divergence, 1w DKL, (Kullback & Leibler, 1951)

$$\frac{1}{w}D_{K L}(q:\pi):=\frac{1}{w}\,\mathbb{E}_{q(\mathbf{\theta})}\left[\log\frac{q(\mathbf{\theta})}{\pi(\mathbf{\theta})}\right],$$
and the Alpha-Rényi divergence, $D^{(\alpha)}_{AR}$. 
$$D_{A R}^{(\alpha)}(q:\pi):=\frac{1}{\alpha(\alpha-1)}\log\left(\mathbb{E}_{\pi(\mathbf{\theta})}\left[\left(\frac{q(\mathbf{\theta})}{\pi(\mathbf{\theta})}\right)^{\alpha}\right]\right).$$
$$(6)$$

As examined in Knoblauch et al. (2022), D
(α)
AR allows for different prior regularisation depending on how much we trust the prior by placing different weights on it. In future work it would be simple to explore other divergences such as the f–divergences, Df , (Amari, 2016; Alquier, 2021). Similarly to the losses, we can perform cross validation to select the α parameter, however as demonstrated in the ablation study (Figure 6) FedGVI performs favourably under a range of α (and δ) values.

## 4. Theoretical Results

We now present a theoretical analysis of FEDGVI. We begin by examining the relationship of FEDGVI with other FL algorithms while recovering some of them as special cases, we study the damping parameter, and examine the convergence behaviour of FEDGVI. Then, we turn our attention on robustness to likelihood misspecification, where we first study FEDGVI as distributed GBI, from which we derive a theorem on the necessity of the cavity distribution. Finally, we derive a result for computationally tractable and conjugate FEDGVI, enabling us to present the main theorem on bias–robustness of FEDGVI. Since it is an open problem where global GVI posteriors converge to under arbitrary divergences, we often have to restrict ourselves to consider the server divergence to be the Kullback–Leibler divergence. This ensures that the posterior at the server will have the structure of a GBI posterior,

$$q_{s}^{(T)}(\mathbf{\theta})\propto\exp\left\{-\sum_{m=1}^{M}\ell_{m}^{(T)}(\mathbf{\theta})\right\}\pi(\mathbf{\theta})$$

where we incorporate prior robustness and tractability through the approximate losses.

## 4.1. Recovering Existing Methods As A Special Case

By choosing specific divergences, loss functions, and variational families, we can recover existing methods as special cases of our framework, which we summarise in Figure 1: Remark 4.1. Choosing the Kullback–Leibler divergence and the negative log–likelihood as a loss function recovers the PVI algorithm of Ashman et al. (2022).

Remark 4.2. When D = Ds = 0, and Q = {δθˆ(θ) : θˆ ∈
Θ}, with δθˆ being the Dirac–delta measure at some element θˆ, we recover FEDAVG of McMahan et al. (2017).

FEDAVG
L, D = 0, {δθ},
M, Ds = 0 PVI
− log pθ, DKL, Q,
M, Ds = DKL
FEDGVI
L, D, Q*, M, D*s VI
− log pθ, DKL, Q,
M = 1, Ds = DKL
ERM
L, D = 0, {δθ}, M = 1, Ds = 0
Figure 1: We illustrate the relationship of FEDGVI— characterised by the loss L, the client divergence D, the variational family Q, the number of clients M, and the divergence at the server Ds—to Partitioned Variational Inference (PVI), Variational Inference (VI), Federated Averaging (FEDAVG), and Empirical Risk Minimisation (ERM).

4.2. Damping as a Bayesian Logarithmic Opinion Pool Choosing the damping parameter to be τ = 1/M results in a logarithmic opinion pool. In fact choosing damping parameters such that all of them sum to unity also forms a valid logarithmic opinion pool (Genest et al., 1986).

Proposition 4.3. Assume Ds = DKL*, and that* Pm τm =
1 where τm ≥ 0 ∀m, then the posterior at the server is an externally Bayesian logarithmic opinion pool of the form

$$q_{s}^{(t)}(\theta)=\frac{\prod_{m=1}^{M}\left(q_{m}^{(t)}(\theta)\right)^{\tau_{m}}}{\int_{\Theta}\prod_{m=1}^{M}\left(q_{m}^{(t)}(\theta)\right)^{\tau_{m}}d\theta},\;\theta-a.e.$$

See Appendix B.2 for the proof. This results provides a theoretical justification on the previously heuristic use of the damping parameter (as used in PVI, Ashman et al., 2022). Specifically it ensures that this selection of τ leads to a valid distribution and results in more stable convergence.

## 4.3. Fixed Points Of Fedgvi

In this section we study the properties of FEDGVI posteriors when these converge to some fixed point. Specifically, we generalise the fixed point result of PVI (Ashman et al., 2022, Property 2.3) to arbitrary losses.

Proposition 4.4. Let Ds = DKL, D =
1 w DKL, w > 0, and *Q ⊂ P*(Θ)*, then if* q
∗ s
(θ) = π(θ) exp{−ℓ
∗s
(θ)}/Zq
∗
such that ∀m ∈ [M], ∆∗m(θ) = 0*, then* q
∗s
(θ) is a local minimiser of the following GVI objective:

$$\mathbb{E}_{q(\mathbf{\theta})}\left[\sum_{m=1}^{M}L_{m}(\mathbf{y}_{m};\mathbf{\theta},\mathbf{x}_{m})\right]+{\frac{1}{w}}D_{K L}(q:\pi)\tag{9}$$

Remark 4.5. If the loss in Equation (9) is convex, then a fixed point of FEDGVI is a global minimum of GVI.

This illustrates that if FEDGVI converges, then the posterior is a (local) minimiser of the GVI objective. We refer to such distributions as fixed points. This recovers Kassab & Simeone (2022, Theorem 1) (which deals with the restricted case of Q = P(Θ)) with a novel proof; see Appendix B.3.

## 4.4. Generalised Bayesian Inference

As a consequence of Proposition 4.4 and Remark 4.1, FEDGVI will recover the GBI posterior when Q = P(Θ).

Lemma 4.6. *Assuming* Q = P(Θ), D =1β DKL *with* β > 0, Ds = DKL*, and* τ = 1*, then* FEDGVI will recover the GBI posterior after the first iteration.

$$\begin{array}{l l}{{q_{s}^{(1)}(\mathbf{\theta})=q_{G B I}(\mathbf{\theta}|\{\mathbf{x}_{m},\mathbf{y}_{m}\}_{m=1}^{M})}}\\ {{\ }}&{{=\exp\{-\beta{\sum_{m=1}^{M}}L(\mathbf{y}_{m};\mathbf{\theta},\mathbf{x}_{m})\}\pi(\mathbf{\theta})/Z}}\end{array}$$

This posterior is invariant under subsequent iterations of FEDGVI, having reached a fixed point.

Moreover, for a damping rate τ = 1/M*, the posterior at the* server converges pointwise a.e. in Θ *to the GBI posterior,*

$$q_{s}^{(T)}(\mathbf{\theta})\stackrel{T\to\infty}{\longrightarrow}q_{G B I}(\mathbf{\theta}|\{\mathbf{x}_{m},\mathbf{y}_{m}\}_{m=1}^{M}),\,\mathbf{\theta}-a.e.$$

This result, proven in Appendix B.4, is the first step towards likelihood robustness. If we were able to find the GBI posterior efficiently with some robust loss, then the posterior would be robust and computable. Here however, the loss may not vary over different iterations of FEDGVI as in Equation (4) and the normaliser may be intractable.

## 4.5. The Cavity Distribution Is Necessary

By further investigating the relationship of FEDGVI with the GBI posterior, we can extend Lemma 4.6 and derive a Theorem under which we are required to use the cavity distribution to regularise the client update. This is in contrast to both PVI, where it's use is heuristically justified, and to other Bayesian FL approaches where the previous posterior is used instead. For this we recall two natural assumptions that any such distribution must satisfy in a federated setting. Assumption 4.7. No client can have access to the data set of another client. Assumption 4.8. Each client generates their update equivalently to other clients. These assumptions combined with Lemma 4.6 lead us to the necessity of the cavity distribution. Theorem 4.9. Let the assumptions be as in Lemma 4.6 *with* τ = 1, and assume that the Assumptions 4.7 and 4.8 are satisfied, then (1.) holds if and only if (2.) *holds.*
1. FEDGVI *recovers the generalised Bayesian posterior* qGBI(θ) *which is invariant under further* FEDGVI *updates.*
2. *The cavity regularises the client optimisation problem.* This provides a principled justification for the use of the cavity distribution, as defined in Equation (3), in FEDGVI. We provide the proof in Appendix B.5.

## 4.6. Conjugate Client Updates

Before we present our main result on provable robustness to likelihood misspecification, we first show that we can find a GBI posterior under specific losses in a computationally tractable manner. Assuming that the data generating process has some exponential family distribution, where y ∼ pθ(y),

$$p_{\mathbf{\theta}}(\mathbf{y})=\exp\{\eta(\mathbf{\theta})^{\top}\phi(\mathbf{y})-A(\eta(\mathbf{\theta}))+h(\mathbf{y})\},$$

such that this is differentiable in y, by using the weighted score matching loss of Altamirano et al. (2023), L
w SM, then client updates, using the weighted KL divergence locally, are available in closed form. If we further assume that our model is Gaussian, or has the form of a squared exponential, and that the natural parameters of the DGP are η(θ) = θ, then the client approximation will have a conjugate form.

Proposition 4.10. Assume that the hypothesis pθ(y) has differentiable, exponential family distribution with η(θ) = θ, L
(t)
m = L
w t m SM*, and* D =
1 β DKL, and the variational family Q is the multivariate Gaussians, then the local posteriors at the clients are conjugate Gaussians. Moreover, Equation (7) will have closed form if Ds has closed form between Gaussian distributions. See Appendix B.6 for the proof. The loss may now depend on the client and iteration t. Most exponential family distributions satisfy the conditions of the proposition, and there are several divergences that allow closed form expressions between Gaussians, such as the Alpha–Renyi , or the ´ α, β, γ–divergences of Cichocki & Amari (2010). Further, this enables the use of intractable likelihood models.

## 4.7. Provable Robustness To Outliers

For a robust loss function at the clients, and using the weighted KL divergence at the clients and the KL divergence at the server, guarantees that after T iterations, the posterior computed at the server will also be robust to outliers. This means we can achieve robustness at the server by leveraging the robust losses that were derived for GVI. In this, we mean robustness as defined by Ghosh & Basu (2016a) and further developed in Matsubara et al. (2022). We define the empirical DGP of a client as Pnm := 1 nm Pnm i=1 δxi, and of the entire data set as Pn := 1n PM
m=1 nmPnm. When this is contaminated by some ε fraction of data centred at some adversarially chosen data point z ∈ Ξ, the misspecified DGP
is defined as P*n,ε,z* := (1 − ε)Pn + εδz.

Definition 4.11. We say that a loss L
(t)
m (θ; Pnm*,ε,z*), w.r.t.

some prior distribution π(θ), is robust to outliers, if the following hold:

$\sup_{z\in\Xi}\left|\frac{d}{d\varepsilon}L_{m}^{(t)}(\boldsymbol{\theta};\mathbb{P}_{nm,\varepsilon,z})\right|_{\varepsilon=0}\right|\leq\gamma_{(m)}^{(t)}(\boldsymbol{\theta})$, $\sup_{\boldsymbol{\theta}\in\Theta}\pi(\boldsymbol{\theta})\gamma_{(m)}^{(t)}(\boldsymbol{\theta})<\infty$, and $\sup_{\boldsymbol{\theta}\in\Theta}\pi(\boldsymbol{\theta})\gamma_{(m)}^{(t)}(\boldsymbol{\theta})\mu(d\boldsymbol{\theta})<\infty$.  
These conditions ensure that the influence of arbitrary contamination on the local posterior is not arbitrarily bad. In particular the auxiliary function γ
(t)
m ensures that the influence of an adversarial data point z on the posterior over infinitesimal contaminations, d dϵ q
(t)
m (θ; Pnm*,ϵ,z*)|ϵ=0, are finite over all θ and z. Condition 2 ensures the loss increases slowly enough for the local posterior to concentrate around the data, and condition 3 ensures the resulting posterior will be normalisable.

Theorem 4.12. Let Ds = DKL, D =
1 w DKL, Q = P(Θ),
further assume that the prior is upper bounded and the loss is lower bounded, then if ∀t ∈ [T] and ∀m ∈ [M] L
(t)
m (θ; Pnm,ε,z) *is robust, then the posterior generated by* FEDGVI *will be robust to outliers.*
The proof is in Appendix B.7. This result together with Proposition 4.10 is significant as we have robustness under intractable optimisation, and we can choose a provably robust, conjugate loss to generate robust FEDGVI posteriors, which are then computationally efficient to compute.

## 5. Experiments

We evaluate FEDGVI against several other methods, specifically PVI (Ashman et al., 2022), FEDAVG (McMahan et al., 2017), the nonparametric DSVGD (Kassab & Simeone, 2022), the distributed MCMC based DSGLD (Ahn et al., 2014), federated MCMC based FEDPA (Al-Shedivat et al., 2021), and the one shot BCM based approach β–
PREDBAYES (Hasan et al., 2024). We provide further details about experiments in Appendix D.

−6 −4 −2 0 2 4 θ 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 MLE w/o outliers MLE with outliers PVI w/o outliers PVI with outliers FedGVI SM loss FedGVI β loss Data Outliers Distributi on
We first examine the effect of misspecified likelihoods through the well known clutter problem (Minka, 2001). We generate 100 observations from a Gaussian location model that is contaminated through Definition 2.1 with ε = 0.25 Gaussian noise. The aim is to infer the location parameter θ of the uncontaminated data. We compare FEDGVI with both Lβ and LSM vs PVI with and without misspecification. We also provide the corresponding MLE results. See Figure 2. Under misspecification both the MLE and PVI fail to recover the true θ, whereas FEDGVI can easily handle different levels of contamination.

5.2. Influence Function

NLL IMQ SE β–Loss γ–Loss 0 5 10 15 20 25 Distance to True mean 0.0 0.5 1.0 1.5 2.0 Influ enc e

To demonstrate robustness to likelihood misspecification as in Theorem 4.12, we consider the influence of a single outlier at one of seven clients on the server posterior. Figure 3 demonstrates that the negative log likelihood is not robust in the federated setting, whereas different robust divergence based losses allow only limited influence of outliers on the posterior. We plot this as the divergence between the posterior, had we observed the outlier value at the true mean, against the posteriors that have the outlier be farther from the true mean, using the Fisher–Rao distance (Nielsen, 2023).

−2 0 2 4 6
−2 0 2 4 6 FedGVI PVI PVI w/o Outliers Class 0 Class 1 Outliers Class 0
We next consider a 2D logistic regression example where we generate 100 linearly separable samples from a Gaussian mixture distribution. We inject outliers generated by a third Gaussian distribution and assign them to one of the classes so that the data is no longer linearly separable. We compare FEDGVI with L
(0.7)
βand D
(1.5)
AR against PVI, both with 5 clients. Again, the target is given by PVI only trained on the uncontaminated data. As expected PVI is severely impacted by outliers, whereas FEDGVI is robust to them and closely recovers the target posterior.

2 4 6 8 10 Server Iterations t 0.64 0.66 0.68 0.70 0.72 0.74 0.76 Clas sifi ca tion A
cc ur acy PVI
FedAvg DSGLD DSVGD

| MODEL           | ACCURACY + STD.   |              |
|-----------------|-------------------|--------------|
| 10 CLIENTS      | 3 CLIENTS         |              |
| FEDAVG          | 96.64± 0.07       | 96.34 ± 0.20 |
| FEDPA           | 94.25± 0.39       | 95.31± 0.35  |
| β–PREDBAYES     | 94.90± 0.08       | 96.73± 0.08  |
| PVI             | 95.56± 0.18       | 96.68± 0.07  |
| FEDGVI DAR      | 96.36± 0.09       | 97.13 ± 0.13 |
| FEDGVI LGCE     | 97.06± 0.03       | 98.04 ± 0.07 |
| FEDGVI DAR+LGCE | 97.50± 0.07       | 98.13± 0.08  |
| VI (1 CLIENT)   | (96.96± 0.17)     |              |
| GVI (1 CLIENT)  | (98.13± 0.07)     |              |

FedGVI, D
(5) AR
FedGVI, L
(0.5) GCE
In this experiment we follow the experimental setup of Kassab & Simeone (2022) and average accuracy over 10 random 80/20 train-test splits, where the training data is split homogeneously across 2 clients. We do not add any label contamination. The results are plotted in Figure 5. The non-robust methods all eventually achieve similar accuracy, however FEDGVI is able to outperform all competing methods, which we argue is due to FEDGVI putting less weight on data points that are less likely to belong to the class.

## 5.5. Bayesian Neural Networks On Mnist And Fashion**Mnist**

Table 1: Classification accuracy (highest in bold) on uncontaminated test data after training on 10% contaminated MNIST data. We report the best performance across all server iterations. We create label contamination by adding noise to the train-

α=0.0 α=0.5 α=1.0 α=1.5 α=2.5 α=5.0 2

| δ=0.0 δ=0.2 δ=0.4 δ=0.6 δ=0.8 δ=1.0 6.16 3.14 1.92 2.26 2.55 2.83 3.68 3.09 2.03 2.01 2.24 2.48 3.46 3.02 2.08 1.95 2.08 2.28 3.32 3.93 2.33 1.96 2.04 2.12 3.16 5.21 2.81 1.83 1.91 2.13 2.63 7.04 4.19 2.05 1.89 1.95   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

3 4 7
% E
rro r 5 6
Figure 6: An ablation study on the hyperparameters of FEDGVI with L
(δ)
GCE and D
(α)
AR. We plot the maximum results achieved as percentage errors on uncontaminated test data after training 5 clients on 10% contaminated data. Table 2: Classification accuracy (highest in bold) on uncontaminated test data after training on different amounts of contaminated FASHIONMNIST data. For FEDGVI we have fixed α = 2.5 for the α−Renyi divergence. Each Method ´ has data split homogeneously across 3 Clients. We report the best performance during all server iterations. ing set while leaving the test set unchanged and evaluate performance in this. For MNIST, we add 10% of class dependent label noise, see Figure 7 and Table 1. We further carry out an ablation study on the hyperparameter selection in FEDGVI with the Alpha–Renyi divergence and the gen- ´ eralised cross entropy loss, see Figure 6. This demonstrates that FEDGVI performs well under a variety of different loss and divergence parameters. Note that α = 1 recovers the KL divergence, α = 0 the reverse KL divergence, i.e.

D
(0)
AR(q : π) = DRKL(q : π) = DKL(π : q), and that δ = 0 recovers the negative log–likelihood.

| MODEL          | CONTAMINATION   |          |          |          |
|----------------|-----------------|----------|----------|----------|
| 0%             | 10%             | 20%      | 40%      |          |
| FEDAVG         | 85.7±0.5        | 79.0±1.9 | 71.2±1.5 | 49.0±6.5 |
| FEDPA          | 88.1±0.3        | 87.4±0.2 | 86.5±0.2 | 85.4±0.5 |
| β–PREDBAYES    | 87.6±0.1        | 87.2±0.1 | 86.8±0.1 | 85.8±0.1 |
| PVI            | 86.2±0.2        | 85.1±0.1 | 84.4±0.1 | 82.8±0.1 |
| FEDGVI δ = 0.0 | 87.1±0.1        | 86.2±0.2 | 85.6±0.1 | 83.8±0.1 |
| FEDGVI δ = 0.4 | 88.7±0.2        | 88.6±0.1 | 87.0±0.4 | 78.1±0.4 |
| FEDGVI δ = 0.5 | 89.0±0.2        | 88.6±0.2 | 88.4±0.2 | 85.1±0.7 |
| FEDGVI δ = 0.8 | 88.6±0.0        | 88.4±0.1 | 88.0±0.0 | 87.2±0.1 |
| FEDGVI δ = 1.0 | 88.1±0.1        | 87.8±0.1 | 87.5±0.2 | 86.0±0.3 |

For FASHIONMNIST, in Table 2, we vary the amount of random label contamination, showcasing performance drops under different amounts of misspecification. We use an MLP, for FEDGVI and PVI with 1 hidden layer of 200

0 5 10 15 20 25 Server Iterations t 1 3 10 30 100 0.1 NLL (
10 Cli ent s)
FedGVI D
(2.5)
AR , L
(0.8) GCE
FedGVI D
(2.5)
AR , LNLL
FedGVI KL, L
(0.8) GCE
%
 Erro r
 (1 0 Cl ient s)FedGVI D
(2.5)
AR , L
(0.8) GCE
FedGVI D
(2.5)
AR , LNLL
FedGVI KL, L
(0.8) GCE
PVI KL, LNLL
GVI D
(2.5)
AR , L
(0.8) GCE
VI KL, LNLL
PVI KL, LNLL
GVI D
(2.5)
AR , L
(0.8) GCE
VI KL, LNLL
1 0 5 10 15 20 25 Server Iterations t 1
% Err or (
3 Cli ent s)
FedGVI D
(2.5)
AR , L
(0.8) GCE
PVI KL, LNLL
GVI D
(2.5)
AR , L
(0.8) GCE
VI KL, LNLL
10 FedGVI D
(2.5)
AR , L
(0.8) GCE
PVI KL, LNLL
GVI D
(2.5)
AR , L
(0.8) GCE
VI KL, LNLL
0.1 NLL (
3 Cli ents)
0 5 10 15 20 25 Server Iterations t 1 3 0 5 10 15 20 25 Server Iterations t
neurons; for FEDAVG, FEDPA, and β–PREDBAYES, two hidden layers with 100 neurons in each. Data is distributed homogeneously across clients, using 5 different, randomly chosen seeds. We demonstrate that under model misspecification, FEDGVI significantly outperforms competing FL methods. Furthermore, FEDGVI incurs no additional computational complexity when compared to PVI. This is due to the KL and Alpha-Renyi divergences having closed form ´ solutions between Multivariate Gaussians with complexity of O(1) in each other, and as we require O(1) additional, constant operations to get the GCE from the NLL.

We provide further experiments in Appendix D on the runtime of FEDGVI against PVI, learning rate selection, stability of posteriors under small perturbations in the robust loss parameters, and showing that using a single hidden layer NN for the competing methods would either negatively, or not significantly, affect their performance.

## 6. Conclusions And Future Work

We have introduced FEDGVI, a novel probabilistic approach to federated learning that is provably robust to model misspecification, and allows for faster, conjugate client updates. The theoretical analysis of FEDGVI demonstrates it's appealing properties; we easily recover existing methods as restricted cases, and characterise the convergence behaviour at fixed points of FEDGVI as solving a global GVI optimisation problem, extending existing theory. Our result on provable robustness to outliers through FEDGVI allows for closed form, conjugate posteriors that are computationally efficient, and robust to model misspecification. In deriving this, we have also shown that the cavity distribution is necessary as predictions would otherwise be overly confident and biased. The robustness of FEDGVI was further demonstrated empirically on multiple synthetic and real–world data sets, showing outperformance of existing FL methods across model architectures and misspecification levels.

An interesting future direction is to extend FedGVI within personalised FL settings (Kotelevskii et al., 2022) and hierarchical Bayesian FL through latent variables (Kim & Hospedales, 2023) as well as through the use of a structured posterior approximation (Hassan et al., 2024), in order to incorporate client level variations. Incorporating the hierarchical model structures and additional inductive biases from such settings, while maintaining conjugacy and favourable computational complexity, remain as open challenges. In future work, we further aim to address the robust Bayesian nonparametric setting of FL through FEDGVI, as well as investigate other types of robustness, including to adversarial and Byzantine attacks, by for instance using a robust aggregator in Equation (6), and addressing the open problem of provable robustness to prior misspecification in GVI.

## Acknowledgements

OH, TM and TD acknowledge support from a UKRI Turing AI Acceleration Fellowship [EP/V02678X/1] and a Turing Impact Award from the Alan Turing Institute. For the purpose of open access, the authors have applied a Creative Commons Attribution (CC-BY) license to any Author Accepted Manuscript version arising from this submission.

The authors acknowledge the University of Warwick Research Technology Platform for assistance in the research described in this paper.

## Impact Statement

This paper presents work on robust federated learning, a framework that aims to not only advance the field of machine learning, but also to develop methods that ensure the privacy of data sources, whilst aiming to achieve optimal performance even under contamination of the data. This approach, however, may discard low probability, tail events that could represent minority groups. Hence, the trade off between robustness and inclusivity is a fundamental ethical challenge for decision makers.

## References

Achituve, I., Shamsian, A., Navon, A., Chechik, G., and Fetaya, E. Personalized federated learning with Gaussian processes. In Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), Advances in Neural Information Processing Systems, 2021.

Ahn, S., Shahbaba, B., and Welling, M. Distributed stochastic gradient MCMC. In Xing, E. P. and Jebara, T. (eds.), Proceedings of the 31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pp. 1044–1052, Bejing, China, 2014.

PMLR.

Al-Shedivat, M., Gillenwater, J., Xing, E., and Rostamizadeh, A. Federated learning via posterior averaging: A new perspective and practical algorithms. In International Conference on Learning Representations, 2021.

Ali, S. M. and Silvey, S. D. A general class of coefficients of divergence of one distribution from another. Journal of the Royal Statistical Society. Series B (Methodological), 28(1):131–142, 1966.

Allouah, Y., Farhadkhani, S., Guerraoui, R., Gupta, N.,
Pinot, R., Rizk, G., and Voitovych, S. Byzantine-robust federated learning: Impact of client subsampling and local updates. In Salakhutdinov, R., Kolter, Z., Heller, K.,
Weller, A., Oliver, N., Scarlett, J., and Berkenkamp, F.

(eds.), *Proceedings of the 41st International Conference* on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 1078–1114. PMLR, 21– 27 Jul 2024.

Alquier, P. Non-exponentially weighted aggregation: Regret bounds for unbounded loss functions. In Meila, M. and Zhang, T. (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 207–218. PMLR, 18–24 Jul 2021.

Alquier, P., Ridgway, J., and Chopin, N. On the properties of variational approximations of gibbs posteriors. Journal of Machine Learning Research, 17(236):1–41, 2016.

Altamirano, M., Briol, F.-X., and Knoblauch, J. Robust and scalable Bayesian online changepoint detection. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 642–663. PMLR, 23–29 Jul 2023.

Altamirano, M., Briol, F.-X., and Knoblauch, J. Robust and conjugate Gaussian process regression. In Salakhutdinov, R., Kolter, Z., Heller, K., Weller, A., Oliver, N., Scarlett, J., and Berkenkamp, F. (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of *Proceedings of Machine Learning Research*, pp. 1155–1185. PMLR, 21–27 Jul 2024.

Amari, S.-i. *Information Geometry and Its Applications*.

Springer, Tokyo, Japan, 2016. ISBN 9784431559771.

Ashman, M., Bui, T. D., Nguyen, C. V., Markou, S., Weller, A., Swaroop, S., and Turner, R. E. Partitioned variational inference: A framework for probabilistic federated learning. *arXiv preprint arXiv:2202.12275*, 2022.

Bao, W., Wu, J., and He, J. BOBA: Byzantine-robust federated learning with label skewness. In Dasgupta, S., Mandt, S., and Li, Y. (eds.), Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pp. 892–900. PMLR, 02–04 May 2024.

Berger, J. O. *Statistical Decision Theory and Bayesian* Analysis. Springer–Verlag, New York, 1985. ISBN
9781475742862.

Berk, R. H. Limiting behavior of posterior distributions when the model is incorrect. *The Annals of Mathematical* Statistics, 37(1):51 - 58, 1966.

Bernardo, J. M. and Smith, A. F. M. *Bayesian theory*. Wiley Series in Probability and Statistics, Chichester, England, 2000. ISBN 9780470316870.

Bissiri, P. G., Holmes, C., and Walker, S. G. A general framework for updating belief distributions. *Journal of* the Royal Statistical Society. Series B (Statistical Methodology), 78(5):1103–1130, 2016.

Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. Variational inference: A review for statisticians. Journal of the American Statistical Association, 112(518):859–877, 2017.

Bui, T. D., Nguyen, C. V., Swaroop, S., and Turner, R. E.

Partitioned variational inference: A unified framework encompassing federated and continual learning. arXiv preprint arXiv:1811.11206, 2018.

Carvalho, L. M., Villela, D. A. M., Coelho, F. C., and Bastos, L. S. Bayesian inference for the weights in logarithmic pooling. *Bayesian Analysis*, 18(1):223 - 251, 2023.

Chan, R. S., Pollock, M., Johansen, A. M., and Roberts, G. O. Divide-and-conquer fusion. *Journal of Machine* Learning Research, 24(193):1–82, 2023.

Chen, W.-N., Choquette-Choo, C. A., Kairouz, P., and Suresh, A. T. The fundamental price of secure aggregation in differentially private federated learning. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of *Proceedings of Machine Learning Research*, pp. 3056–3089. PMLR, 17–23 Jul 2022.

Cichocki, A. and Amari, S.-i. Families of alpha- beta- and gamma- divergences: Flexible and robust measures of similarities. *Entropy*, 12(6):1532–1568, 2010.

Corinzia, L., Beuret, A., and Buhmann, J. M. Variational federated multi-task learning. arXiv preprint arXiv:1906.06268, 2021.

Demidovich, Y., Ostroukhov, P., Malinovsky, G., Horvath, ´
S., Taka´c, M., Richt ˇ arik, P., and Gorbunov, E. Meth- ´ ods with local steps and random reshuffling for generally smooth non-convex federated optimization. In The Thirteenth International Conference on Learning Representations, 2025.

Diaconis, P. and Freedman, D. On the consistency of Bayes estimates. *The Annals of Statistics*, 14(1):1 - 26, 1986.

Fraboni, Y., Vidal, R., Kameni, L., and Lorenzi, M. A general theory for federated optimization with asynchronous and heterogeneous clients updates. *Journal of Machine* Learning Research, 24(110):1–43, 2023.

Genest, C. A characterization theorem for externally Bayesian groups. *The Annals of Statistics*, 12(3):1100– 1105, 1984.

Genest, C., McConway, K. J., and Schervish, M. J. Characterization of externally Bayesian pooling operators. The Annals of Statistics, 14(2):487 - 501, 1986.

Ghosh, A. and Basu, A. Robust Bayes estimation using the density power divergence. Annals of the Institute of Statistical Mathematics, 68(2):413–437, 2016a.

Ghosh, A. and Basu, A. Robust estimation in generalized linear models: the density power divergence approach. TEST, 25(2):269–290, 2016b.

Grunwald, P. The safe Bayesian. In Bshouty, N. H., Stoltz, ¨
G., Vayatis, N., and Zeugmann, T. (eds.), Algorithmic Learning Theory, pp. 169–183, Berlin, Heidelberg, 2012. Springer Berlin Heidelberg.

Guo, H., Greengard, P., Wang, H., Gelman, A., Kim, Y.,
and Xing, E. Federated learning as variational inference: A scalable expectation propagation approach. In The Eleventh International Conference on Learning Representations, 2023.

Hamer, J., Mohri, M., and Suresh, A. T. FedBoost: A
communication-efficient algorithm for federated learning. In III, H. D. and Singh, A. (eds.), Proceedings of the 37th International Conference on Machine Learning, volume 119 of *Proceedings of Machine Learning Research*, pp. 3973–3983. PMLR, 2020.

Hasan, M., Zhang, G., Guo, K., Chen, X., and Poupart, P.

Calibrated one round federated learning with Bayesian inference in the predictive space. Proceedings of the AAAI Conference on Artificial Intelligence, 38(11):12313– 12321, 2024.

Hasenclever, L., Webb, S., Lienart, T., Vollmer, S., Lakshminarayanan, B., Blundell, C., and Teh, Y. W. Distributed Bayesian learning with stochastic natural gradient expectation propagation and the posterior server. Journal of Machine Learning Research, 18(1):3744–3780, 2017.

Hassan, C., Salomone, R., and Mengersen, K. Federated variational inference methods for structured latent variable models. *arXiv preprint arXiv:2302.03314*, 2023.

Hassan, C., Sutton, M., Mira, A., and Mengersen, K. Scalable vertical federated learning via data augmentation and amortized inference. *arXiv preprint arXiv:2405.04043*,
2024.

Heikkila, M., Ashman, M., Swaroop, S., Turner, R., and ¨
Honkela, A. Differentially private partitioned variational inference. *Transactions on machine learning research*,
2023(4), 2023.

Hooker, G. and Vidyashankar, A. N. Bayesian model robustness via disparities. *TEST*, 23(3):556–584, 2014.

Huber, P. J. Robust estimation of a location parameter.

Annals of Mathematical Statistics, 35:73–101, 1964.

Hung, H., Jou, Z.-Y., and Huang, S.-Y. Robust mislabel logistic regression without modeling mislabel probabilities. Biometrics, 74(1):145–154, 2018.

Hyvarinen, A. Estimation of non-normalized statistical ¨
models by score matching. Journal of Machine Learning Research, 6(24):695–709, 2005.

Jewson, J., Smith, J. Q., and Holmes, C. Principles of Bayesian inference using general divergence criteria. Entropy, 20(6):442, 2018.

Jonker, M. A., Pazira, H., and Coolen, A. C. Bayesian federated inference for estimating statistical models based on non-shared multicenter data sets. *Statistics in Medicine*, pp. 1–18, 2024.

Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., Bonawitz, K., Charles, Z., Cormode, G., Cummings, R., D'Oliveira, R. G. L., Eichner, H., Rouayheb, S. E., Evans, D., Gardner, J., Garrett, Z.,
Gascon, A., Ghazi, B., Gibbons, P. B., Gruteser, M., Har- ´ chaoui, Z., He, C., He, L., Huo, Z., Hutchinson, B., Hsu, J., Jaggi, M., Javidi, T., Joshi, G., Khodak, M., Konecny,´ J., Korolova, A., Koushanfar, F., Koyejo, S., Lepoint, T.,
Liu, Y., Mittal, P., Mohri, M., Nock, R., Ozg ¨ ur, A., Pagh, ¨
R., Qi, H., Ramage, D., Raskar, R., Raykova, M., Song, D., Song, W., Stich, S. U., Sun, Z., Suresh, A. T., Tramer, ` F., Vepakomma, P., Wang, J., Xiong, L., Xu, Z., Yang, Q., Yu, F. X., Yu, H., and Zhao, S. Advances and open problems in federated learning. *Foundations and Trends®* in Machine Learning, 14(1–2):1–210, 2021.

Kallioinen, N., Paananen, T., Burkner, P.-C., and Vehtari, A. ¨
Detecting and diagnosing prior and likelihood sensitivity with power-scaling. *Statistics and Computing*, 34(1):57, 2024.

Kassab, R. and Simeone, O. Federated generalized Bayesian learning via distributed Stein variational gradient descent. IEEE Transactions on Signal Processing, 70:2180–2192, 2022.

Katsevich, A. and Rigollet, P. On the approximation accuracy of Gaussian variational inference. The Annals of Statistics, 52(4):1384 - 1409, 2024.

Kim, M. and Hospedales, T. FedHB: Hierarchical Bayesian federated learning. *arXiv preprint arXiv:2305.04979*, 2023.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In 3rd International Conference on Learning Representations, 2015.

Knoblauch, J., Jewson, J. E., and Damoulas, T. Doubly robust Bayesian inference for non-stationary streaming data with β-divergences. In Advances in Neural Information Processing Systems, volume 31, pp. 64–75. Curran Associates, Inc., 2018.

Knoblauch, J., Jewson, J., and Damoulas, T. An optimization-centric view on Bayes' rule: Reviewing and generalizing variational inference. Journal of Machine Learning Research, 23(132):1–109, 2022.

Kotelevskii, N. Y., Vono, M., Durmus, A., and Moulines, E. FedPop: A Bayesian approach for personalised federated learning. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), *Advances in Neural Information* Processing Systems, 2022.

Kullback, S. and Leibler, R. A. On Information and Sufficiency. *The Annals of Mathematical Statistics*, 22(1):79 - 86, 1951.

LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradientbased learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

Li, H., Acharya, K., and Richtarik, P. The power of extrapo- ´
lation in federated learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Malinovsky, G., Kovalev, D., Gasanov, E., Condat, L., and Richtarik, P. From local SGD to local fixed-point methods for federated learning. In III, H. D. and Singh, A. (eds.), Proceedings of the 37th International Conference on Machine Learning, volume 119 of *Proceedings of Machine* Learning Research, pp. 6692–6701. PMLR, 2020.

Matsubara, T., Knoblauch, J., Briol, F.-X., and Oates, C. J.

Robust generalised Bayesian inference for intractable likelihoods. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(3):997–1022, 04 2022.

McMahan, B., Moore, E., Ramage, D., Hampson, S., and Arcas, B. A. y. Communication-efficient learning of deep networks from decentralized data. In Singh, A. and Zhu, J. (eds.), *Proceedings of the 20th International Conference* on Artificial Intelligence and Statistics, volume 54 of Proceedings of Machine Learning Research, pp. 1273– 1282. PMLR, 2017.

Mekkaoui, K. e., Mesquita, D., Blomstedt, P., and Kaski, S. Federated stochastic gradient Langevin dynamics. In de Campos, C. and Maathuis, M. H. (eds.), Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence, volume 161 of Proceedings of Machine Learning Research, pp. 1703–1712. PMLR, 2021.

Mesquita, D., Blomstedt, P., and Kaski, S. Embarrassingly parallel MCMC using deep invertible transformations. In Adams, R. P. and Gogate, V. (eds.), *Proceedings of* The 35th Uncertainty in Artificial Intelligence Conference, volume 115 of *Proceedings of Machine Learning* Research, pp. 1244–1252. PMLR, 2020.

Miller, J. W. Asymptotic normality, concentration, and coverage of generalized posteriors. *Journal of Machine* Learning Research, 22(168):1–53, 2021.

Minka, T. P. Expectation propagation for approximate Bayesian inference. In *Proceedings of the Seventeenth* Conference on Uncertainty in Artificial Intelligence, pp.

362–369, San Francisco, CA, USA, 2001.

Nielsen, F. An elementary introduction to information geometry. *Entropy*, 22(10):1100, 2020.

Nielsen, F. A simple approximation method for the fisher–rao distance between multivariate normal distributions. *Entropy*, 25(4), 2023.

Opper, M. and Winther, O. Expectation consistent approximate inference. *Journal of Machine Learning Research*, 6(73):2177–2204, 2005.

Pardo Llorente, L. Statistical inference based on divergence measures. Chapman & Hall/CRC, 2006. ISBN 9781584886006.

Pinski, F. J., Simpson, G., Stuart, A. M., and Weber, H.

Kullback-leibler approximation for probability measures on infinite dimensional spaces. SIAM Journal on Mathematical Analysis, 47(6):4091–4122, 2015.

Reddi, S., Charles, Z. B., Zaheer, M., Garrett, Z., Rush, K., Konecnˇ y, J., Kumar, S., and McMahan, B. Adaptive ´
federated optimization. In International Conference on Learning Representations, 2021.

Scott, S. L., Blocker, A. W., Bonassi, F. V., Chipman, H. A.,
George, E. I., and McCulloch, R. E. Bayes and big data: The consensus monte carlo algorithm. International Journal of Management Science and Engineering Management, 11:78–88, 2016.

Swaroop, S., Khan, M. E., and Doshi-Velez, F. Connecting federated ADMM to Bayes. In The Thirteenth International Conference on Learning Representations, 2025.

Tenison, I., Sreeramadas, S. A., Mugunthan, V., Oyallon, E.,
Rish, I., and Belilovsky, E. Gradient masked averaging for federated learning. *Transactions on Machine Learning* Research, 2023.

Tresp, V. A Bayesian committee machine. Neural computation, 12:2719–41, 2000.

Tziotis, I., Shen, Z., Pedarsani, R., Hassani, H., and Mokhtari, A. Straggler-resilient personalized federated learning. *Transactions on Machine Learning Research*, 2023.

Vedadi, E., Dillon, J. V., Mansfield, P. A., Singhal, K.,
Afkanpour, A., and Morningstar, W. R. Federated variational inference: Towards improved personalization and generalization. Transactions on Machine Learning Research, 2024.

Vehtari, A., Gelman, A., Sivula, T., Jylanki, P., Tran, D., ¨
Sahai, S., Blomstedt, P., Cunningham, J. P., Schiminovich, D., and Robert, C. P. Expectation propagation as a way of life: A framework for Bayesian inference on partitioned data. *Journal of Machine Learning Research*, 21(1), 2020.

Walker, S. G. Bayesian inference with misspecified models.

Journal of Statistical Planning and Inference, 143(10): 1621–1633, 2013.

Xiao, H., Rasul, K., and Vollgraf, R. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. *arXiv preprint arXiv:1708.07747*, 2017.

Yonekura, S. and Sugasawa, S. Adaptation of the tuning parameter in general Bayesian inference with robust divergence. *Statistics and Computing*, 33(2):39, 2023.

Yurochkin, M., Agarwal, M., Ghosh, S., Greenewald, K.,
Hoang, N., and Khazaeni, Y. Bayesian nonparametric federated learning of neural networks. In Chaudhuri, K. and Salakhutdinov, R. (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of *Proceedings of Machine Learning Research*, pp. 7252– 7261. PMLR, 09–15 Jun 2019.

Zellner, A. Optimal information processing and Bayes's theorem. *The American Statistician*, 42(4):278–280, 1988.

Zhang, X., Li, Y., Li, W., Guo, K., and Shao, Y. Personalized federated learning via variational Bayesian inference. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of *Proceedings of Machine Learning Research*, pp. 26293–26310. PMLR, 17–23 Jul 2022.

Zhang, Z. and Sabuncu, M. Generalized cross entropy loss for training deep neural networks with noisy labels. In Bengio, S., Wallach, H., Larochelle, H., Grauman, K., Cesa-Bianchi, N., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018.

Zhao, Z., Luo, M., and Ding, W. Deep leakage from model in federated learning. In Conference on Parsimony and Learning, volume 234 of Proceedings of Machine Learning Research, pp. 324–340. PMLR, 2023.

Zhu, L., Liu, Z., and Han, S. Deep leakage from gradients. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alche-Buc, F., Fox, E., and Garnett, R. (eds.), ´ Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

# Supplementary Material For: Federated Generalised Variational Inference: A Robust Probabilistic Federated Learning Framework

The appendix is structured as follows: Appendix A summarises the notation used throughout the paper and in the proofs. In Appendix B we present complete proofs of all theorems, propositions and lemmas given in the main paper. Appendix C clarifies the requirements of Definition 4.11, the GBI learning rate, and places FEDGVI in the broader GVI literature. Lastly, Appendix D gives additional details about the implementation of FEDGVI and additional experiments.

## A. Notation

In this section, we give definitions of the symbols used throughout the paper and the appendix.

P0 The abstract and unknown probability measure, also called data generating process, acting on some abstract measurable space (Ω, F) which gives rise to the data
{xi, yi}
n i=1 Entire data set of all clients, also written as {x n 1, yn 1 }, for xi ∈ Ξ and yi|xi ∈ Υ
{xm, ym}M
m=1 The entire set of data points split across M clients labelled m ∈ [M] := {1, 2*, ..., M*}
Ξ The data space, which is assumed to have Polish topology Υ The output space, which can be categorical such as in classification where Υ = [C], or real valued as in regression Υ = R
C , C ∈ N
Θ In the parametric setting this is the parameter space θ ∈ Θ, assumed to admit Polish topology P(Θ) The space of probability measures over the measurable space (Θ, T ). We refer to distributions in this space, where we mean distribution functions given rise to by measures in this space. Note that these need not be continuous, and could only be defined almost everywhere in θ. Q A variational family of distributions such that *Q ⊂ P*(Θ) and, in terms of distributions, Q = {q(θ|κ) ∈ P(Θ) :
κ ∈ K}, where K is a set of variational parameters π(θ) The prior distribution, given rise to by the prior measure Π on (Θ, T ) L
(t)
m (ym; θ, xm) The local loss of client m, at iteration t ∈ [T], on the local data set {xm, ym}, not necessarily the same across clients nor iterations, and associated with the parameters θ ∈ Θ ℓ
(t)
m (θ) Local loss approximation of Lm(ym; θ, xm) and the impact of the data of client m on the posterior at the server
∆
(t)
m (θ) Local update, Equation (5), that represents the change in the approximate posteriors, and the de facto change in the local loss approximation. It has associated damping parameter τ .

ℓ
(t)
s (θ) Global loss approximation of all clients aggregated at the server q
(t)
m (θ) Local posterior computed through Equation (4)
q
(t)
s (θ) Global approximate posterior after server–side optimisation step, Equation (7)
P(*L, D,* Q) The Rule of Three (Knoblauch et al., 2022) that defines a global GVI objective D Any statistical divergence D : P(Θ) × P(Θ) → R≥0 (for a detailed definition see Nielsen, 2020); Ds denotes the divergence at the server.

Eq(θ) The expectation with respect to q(θ)

## B. Proofs Of Theorems, Propositions, And Lemmas

Here, we provide the full proofs of the theorems stated in the paper. Throughout, we assume that all the losses, distributions and approximate losses, are measurable with respect to some dominating measure µ(dθ). This can be the Lebesgue measure in finite dimensional spaces, or more generally the Haar measure. For infinite dimensional measure spaces, which are of interest in the study of Bayesian inverse problems and nonparametrics, we could assume µ(dθ) to be a Gaussian measure as in Pinski et al. (2015).

## B.1. Equivalence Between The Kl Divergence And Weighted Kl Divergence

First, we present a well known auxiliary lemma that will be used throughout the proofs. It states that the weighted KL divergence is equivalent to using a tempered or weighted likelihood in the optimisation procedure, and hence lead to equivalent inference problems (Knoblauch et al., 2022; Bissiri et al., 2016). So without loss of generality, we can push the weighting term of the KL divergence inside the loss, by defining the loss to be L = w · L, which does not change the optimisation procedure. We show this result for f–divergences, which we define as in Ali & Silvey (1966) and Amari (2016).

Lemma B.1. For w > 0 the posteriors computed by the weighted f*–divergence,* D =
1 w Df and loss L, and the posterior through the f–divergence D = Df and weighted loss w · L *are equivalent, i.e.,*

$$P(L,\frac{1}{w}D_{f},\mathcal{Q})=P(w\cdot L,D_{f},\mathcal{Q})$$

Proof

P(L,  1 w Df , Q) = arg min q∈Q  Eq(θ)[L(y; θ, x)] + 1w Df (q : π)  = arg min q∈Q  Eq(θ)[L(y; θ, x)] + 1w Eq(θ) f q(θ) π(θ)  = arg min q∈Q  1 w Eq(θ) w · L(y; θ, x) + f q(θ) π(θ)  = arg min q∈Q  Eq(θ) w · L(y; θ, x) + f q(θ) π(θ)  = arg min q∈Q Eq(θ)[w · L(y; θ, x)] + Df (q : π)	:= P(w · L, Df , Q)
Therefore, when referring to the loss in the following we mean it to be the weighted loss so that we can utilise the weighted KL divergence. This easily recovers the KL–divergence for f : u *7→ −* log u.

## B.2. Proposition **4.3: A Logarithmic Opinion Pool Through Damping**

Proof Consider the server update at some iteration t, where we gather the client updates. Under the KL divergence, we then solve the server optimisation procedure as:

$$q_{s}^{(t)}(\mathbf{\theta})=\operatorname*{arg\,min}_{q\in\mathcal{Q}}\left\{\mathbb{E}_{q(\mathbf{\theta})}\left[\ell_{s}^{(t)}(\mathbf{\theta})\right]+KL(q:\pi)\right\}=\operatorname*{arg\,min}_{q\in\mathcal{Q}}\left\{\mathbb{E}_{q(\mathbf{\theta})}\left[\log\frac{q(\mathbf{\theta})}{\pi(\mathbf{\theta})\exp\{-\ell_{s}^{(t)}(\mathbf{\theta})\}}\right]\right\}.$$

we know that this is minimised at:

q (t) s (θ) ∝ π(θ) exp{−ℓ (t) s (θ)} = π(θ) exp  ( −ℓ (t−1) s(θ) −X M m=1 ∆(t) m (θ) ) m=1  q (t) m (θ) q (t−1) s (θ) !τm exp  ( −X M m=1 −τm log  q (t) m (θ) q (t−1) s (θ) ) ∝ q (t−1) s(θ)Y M ∝ π(θ) exp{−ℓ (t−1) s(θ)} | {z } ∝q (t−1) s (θ) = q (t−1) s (θ)QM m=1(q (t) m (θ))τm (q (t−1) s (θ)) PM m=1 τm
By assumption we have that PMm=1 τm = 1, therefore (q
e $(q_s^{(t-1)}(\pmb{\theta}))^{\sum_{m=1}^M\tau_m}=q_s^{(t-1)}(\pmb{\theta})$
s (θ) and:
$$q_{s}^{(t)}(\mathbf{\theta})\propto\prod_{m=1}^{M}(q_{m}^{(t)}(\mathbf{\theta}))^{\tau_{m}}$$
$$q_{s}^{(t)}(\mathbf{\theta})={\frac{\prod_{m=1}^{M}(q_{m}^{(t)}(\mathbf{\theta}))^{\tau_{m}}}{\int_{\Theta}\prod_{m=1}^{M}(q_{m}^{(t)}(\mathbf{\theta}))^{\tau_{m}}\;\mu(d\mathbf{\theta})}},\;\mu-a.e.$$
This forms an externally Bayesian logarithmic opinion pool (Genest, 1984; Genest et al., 1986).

## B.3. Proof Of Proposition 4.4

The proof of Proposition 4.4 is adapted from that for Partitioned Variational Inference in Ashman et al. (2022). We show the proof of Proposition 4.4 by comparing the derivatives with respect to the variational parameters of q(θ|κ) of the sum of local objectives with those of the global objective. This is motivated by the equivalence of a sum of local GVI objectives (from each client) with some added constants and the global GVI objective, demonstrated in Appendix B.3.1. The main proof is in Appendix B.3.2.

## B.3.1. Recovering A Global Gvi Objective From Local Objectives

First, we provide an analogue of Ashman et al. (2022, Property 2) which states that the sum of the local (client) FEDGVI
objectives and some constant, which we find to be the negative log normalising constants of the cavity and the server distributions, equals the global GVI objective. We define the following:

q (t) s(θ) = 1 Zq (t) s π(θ) exp{− X M m=1 ℓ (t) m (θ)} q \m(θ) = 1 Zq \m π(θ) exp{− X k̸=m ℓ (t) k (θ)} ∝ q (t) s (θ) exp{−ℓ (t) m (θ)} Obj(m, q(t) s) := Eq(θ)[Lm(ym; θ, xm)] + 1w DKL(q : q \m) Obj(q (t) s) := Eq(θ) "X M m=1 Lm(ym; θ, xm) # + 1 w DKL(q : π)
Then we can recover the global objective by summing over the local objectives and subtracting the log normalising constants of the cavity distributions and the current server posterior.

X
M
m=1
Obj(*m, q*(t)
s) −
1
w
(log Zq
(t)
s+PMm=1 log Zq
\m)
=X
M
m=1
Eq(θ)[Lm(ym; θ, xm)] + 1w
DKL(q : q
\m)
−
1
w
(log Zq
(t)
s
+PM
m=1 log Zq
\m)
=X
M
m=1
Eq(θ)[Lm(ym; θ, xm)] + X
M
m=1
1
w
Eq(θ)
log q(θ)
q
\m(θ)
−
1
w
(log Zq
(t)
s
+PM
m=1 log Zq
\m)
= Eq(θ)
"X
M
m=1
Lm(ym; θ, xm)
#
+
1
w
Eq(θ)
"X
M
m=1
log q(θ)
q
\m(θ)
#
−
1
w
(log Zq
(t)
s
+PM
m=1 log Zq
\m)
= Eq(θ)
"X
M
m=1
Lm(ym; θ, xm)
#
+
1
w
Eq(θ)
"
log Y
M
m=1
q(θ) exp{−ℓ
(t)
m (θ)}
q
(t)
s (θ)
#
−
1
w
(log Zq
(t)
s
+PM
m=1 log Zq
\m)
= Eq(θ)
"X
M
m=1
Lm(ym; θ, xm)
#
+
1
w
Eq(θ)
"
log q(θ) exp{−PMm=1ℓ
(t)
m (θ)}
q
(t)
s (θ)
#
−
1
w
log Zq
(t)
s
= Eq(θ)
"X
M
m=1
Lm(ym; θ, xm)
#
+
1
w
Eq(θ)
"
log q(θ)
π(θ)/Zq
(t)
s
#−
1
w
log Zq
(t)
s= Obj(q
(t)
s
)
17 Hence, by using the weighted KL divergence at the clients optimisation step, can we recover a global GVI objective by summing over the local objectives and adding some constants independent of the variational parameters of interest in the optimisation problem. We note that the added logarithms of the normalising constants are independent of κ, since these are fixed through the current posterior and cavity distribution and do not depend on the variational parameters.

B.3.2. PROPOSITION 4.4: FIXED POINTS RECOVERS A GLOBAL FIXED POINT
We denote a fixed point of the algorithm as q
∗
s
(θ|κ
∗) such that for all m ∈ [M] we have q
∗s
(θ|κ
∗) ∈ arg minq∈Q Obj(*m, q*∗
s
),
then we have the property that no update will change the posterior found. Recall: Proposition 4.4 Let D =
1 w DKL at the clients, local loss Lm and Q := {q(θ|κ) : κ ∈ K*} ⊂ P*(Θ) as a variational family.

Assume that FEDGVI *finds a fixed point* q
∗
s
(θ|κ
∗)*, such that for all clients we have that* q
∗ s
(θ|κ
∗) ∈ arg minq∈Q Obj(*m, q*∗s
).

Then, it holds that q
∗
s(θ|κ
∗) ∈ arg minq∈Q Obj(q
∗
s).

Proof First we note that we consider only the KL divergence in this proof, which is equivalent to saying we modify the loss L to be multiplied by w > 0, which results in the equivalent formulation, as shown in Knoblauch et al. (2022) where P(L, 1w DKL, Q) = P(w · *L, KL,* Q), see also Lemma B.1.

Note that the condition ∀m ∈ [M] we have that q
∗s(θ|κ
∗) ∈ arg minq∈Q Obj(*m, q*∗
s) is equivalent to requiring that
∆∗m(θ) = 0, since this means that the local loss approximations remain unchanged and hence ℓ
∗s(θ) remains unchanged.

This then implies that the posterior at the server will not change. This is the same as saying that the client optimisation step has found the global solution and hence q
∗
m(θ) and q
∗
s(θ) will be the same which implies that ∆∗m(θ) = 0.

In the following all integrals are assumed to be over the parameter space Θ, even when we don't make it explicit. We can furthermore show that we can express the derivative of the local objective as a single integral under the weighted KL divergence.

∇κObj(m, q∗ s) = ∇κ Eq(θ)[Lm(ym; θ, xm)] + DKL q :q ∗ s(θ|κ ∗) exp{−ℓ ∗m(θ|κ∗)}Z∗ qs  = ∇κ Zq(θ|κ) log 1 exp{−Lm(ym; θ, xm)} + q(θ|κ) log q(θ|κ) exp{−ℓ ∗m(θ|κ ∗)} q ∗ s (θ|κ∗)+ log Z ∗ qs µ(dθ) = ∇κ Zq(θ|κ) log q(θ|κ) exp{−ℓ ∗m(θ|κ ∗)} q ∗ s(θ|κ∗) exp{−Lm(ym; θ, xm)} µ(dθ) + ✘✘✘✘✘✘✘✘✘✘✘✘✿0 ∇κ log Z ∗ qs Zq(θ|κ) µ(dθ)
Now we first show that the fixed point is an extremum of the global objective and then that it is a minimum. We do this by first differentiating the local objective with respect to the variational parameters κ and then that the sum of the local derivatives evaluated at κ = κ
∗equal the derivative of the global objective.

∇κObj(m, q∗s) = ∇κ Zq(θ|κ) log q(θ|κ) exp{−ℓ ∗m(θ|κ ∗)} q ∗ s(θ|κ∗) exp{−Lm(ym; θ, xm)} µ(dθ) = ∇κ Zq(θ|κ)(Lm(ym; θ, xm) − ℓ ∗ m(θ|κ ∗)) µ(dθ) + ∇κ Zq(θ|κ) log q(θ|κ) q ∗ s(θ|κ∗) µ(dθ) = ∇κ Zq(θ|κ)(Lm(ym; θ, xm) − ℓ ∗ m(θ|κ ∗)) µ(dθ) +  Z(∇κq(θ|κ)) log q(θ|κ) q ∗s(θ|κ∗) µ(dθ) + Z ✘✘✘✘✘✘✘✘✿0 ∇κq(θ|κ) µ(dθ)
where first line follows since we can compose the expectation and (weighted) KL divergence and the normalising constant of the cavity distribution is constant with respect to κ. The last line follows from the fact that d dx f(x) log f(x) =
f
′(x) log f(x) + f
′(x) and that we can exchange the order of integration and differentiation. We further note that at convergence, where κ = κ
∗, that log q(θ|κ)
q
∗ s
(θ|κ∗)
κ=κ∗ = 0. Evaluating the expression above at κ = κ
∗then yields:

$$\nabla_{\kappa}\mathrm{Obj}(m,q_{s}^{*})\Big|_{\kappa=\kappa^{*}}=\nabla_{\kappa}\int q(\mathbf{\theta}|\mathbf{\kappa})(L_{m}(\mathbf{y}_{m};\mathbf{\theta},\mathbf{x}_{m})-\ell_{m}^{*}(\mathbf{\theta}|\mathbf{\kappa}^{*}))\,\mu(d\mathbf{\theta})\Big|_{\kappa=\kappa^{*}}$$

18 Summing over all these client objectives then yields the following expression:

m=1 ∇κObj(m, q∗s) κ=κ∗ =X M X M m=1 ∇κ Zq(θ|κ)(Lm(ym; θ, xm) − ℓ ∗ m(θ|κ ∗)) µ(dθ) κ=κ∗ = ∇κ Zq(θ|κ)(X M m=1 Lm(ym; θ, xm) −X M m=1 ℓ ∗ m(θ|κ ∗)) µ(dθ) κ=κ∗ = Z(∇κq(θ|κ)) log q ∗ s(θ|κ ∗) π(θ) exp{PM m=1 Lm(ym; θ, xm)} µ(dθ) κ=κ∗ + ✘✘✘✘✘✘✘✘✘✘✘✘✘✿0 ∇κ Zq(θ|κ) log Zq ∗ µ(dθ)
To compare this with a global fixed point we differentiate the global objective at q
∗, not yet assumed to be a minimiser of the global objective, with respect to the variational parameters.

$$\nabla_{\mathbf{\kappa}}\mathrm{Obj}(q_{\mathbf{\kappa}}^{*})=\nabla_{\mathbf{\kappa}}\int q(\mathbf{\theta}|\mathbf{\kappa})\log\frac{q(\mathbf{\theta}|\mathbf{\kappa})}{\pi(\mathbf{\theta})\exp\{\sum_{m=1}^{M}L_{m}(\mathbf{y}_{m};\mathbf{\theta},\mathbf{x}_{m})\}}\,\mu(d\mathbf{\theta})$$ $$=\int(\nabla_{\mathbf{\kappa}}q(\mathbf{\theta}|\mathbf{\kappa}))\log\frac{q(\mathbf{\theta}|\mathbf{\kappa})}{\pi(\mathbf{\theta})\exp\{\sum_{m=1}^{M}L_{m}(\mathbf{y}_{m};\mathbf{\theta},\mathbf{x}_{m})\}}\,\mu(d\mathbf{\theta})+\int\!\!\!\sum_{\mathbf{\kappa}}\!q(\mathbf{\theta}|\mathbf{\kappa})\widetilde{\mu(d\mathbf{\theta})}$$

Then,

∇κObj(q ∗s) κ=κ∗ = Z(∇κq(θ|κ)) log q(θ|κ) π(θ) exp{PM m=1 Lm(ym; θ, xm)} µ(dθ) κ=κ∗ =X M m=1 ∇κObj(m, q∗ s) κ=κ∗
And since q
∗ s
(θ|κ
∗) is a fixed point of each client, we have that ∇κObj(*m, q*∗
s
)κ=κ∗ = 0. Therefore,

$$\sum_{m=1}^{M}\nabla_{\kappa}\mathrm{Obj}(m,q_{s}^{*})\Big|_{\kappa=\kappa^{*}}=0\qquad\Longrightarrow\qquad\nabla_{\kappa}\mathrm{Obj}(q_{s}^{*})\Big|_{\kappa=\kappa^{*}}=0$$

This means that q
∗
s(θ|κ
∗) is an extremum of FEDGVI, and further that it is also an extremum of GVI with D =
1 w DKL. We now show that it is further a minimum of the global GVI objective. We consider the Hessian ∇∇κ and proceed like before.

∇∇κObj(m, q∗ s ) = ∇∇κ Zq(θ|κ) log q(θ|κ) exp{−ℓ ∗ m(θ|κ ∗)} q ∗ s (θ|κ∗) exp{−Lm(ym; θ, xm)} µ(dθ) = ∇∇κ Zq(θ|κ)(Lm(ym; θ, xm) − ℓ ∗ m(θ|κ ∗)) µ(dθ) + ∇∇κ Zq(θ|κ) log q(θ|κ) q ∗ s(θ|κ∗) µ(dθ) = ∇∇κ Zq(θ|κ)(Lm(ym; θ, xm) − ℓ ∗ m(θ|κ ∗)) µ(dθ) + ∇κ   Z(∇κq(θ|κ)) log q(θ|κ) q ∗s(θ|κ∗) µ(dθ) +✘ Z ✘✘✘✘✘✘✘✘✿0 ∇κ log q(θ|κ) µ(dθ)   = ∇∇κ Zq(θ|κ)(Lm(ym; θ, xm) − ℓ ∗ m(θ|κ ∗)) µ(dθ) + Z(∇∇κq(θ|κ)) log q(θ|κ) q ∗s (θ|κ∗) µ(dθ) +  Z(∇κq(θ|κ))(∇κ log q(θ|κ)) µ(dθ)

Ashman et al. (2022) point out that this last term can equivalently be expressed through it's transpose.

$$\left(\int(\nabla_{\mathbf{\kappa}}q(\mathbf{\theta}|\mathbf{\kappa}))(\nabla_{\mathbf{\kappa}}\log q(\mathbf{\theta}|\mathbf{\kappa}))\,\mu(d\mathbf{\theta})\right)^{\top}$$

19

= ∇κ Zq(θ|κ)(∇κ log q(θ|κ)) µ(dθ) +✘ Z ✘✘✘✘✘✘✘✘✘✿0 ∇∇κq(θ|κ) µ(dθ) = ∇κ Zq(θ|κ)1 q(θ|κ) µ(dθ) = 0

Evaluating this Hessian at κ = κ
∗:

$$\left.\nabla\nabla_{\kappa}\mathrm{Obj}(m,q_{s}^{*})\right|_{\kappa=\kappa^{*}}=$$
✘✘✘✘✘✘✘✘✘✘✘✘✘✘✘✿0
(∇∇κq(θ|κ)) log q(θ|κ)
q
∗s
(θ|κ∗)
µ(dθ)
κ=κ∗
Therefore, when summing over the individual Hessians of the clients, we get:
∇∇κ
Zq(θ|κ)(Lm(ym; θ, xm) − ℓ
∗m(θ|κ
∗)) µ(dθ)
κ=κ∗
+
Z
m=1
∇∇κObj(*m, q*∗s)
κ=κ∗
=X
M
X
M
m=1
∇∇κ
Zq(θ|κ)(Lm(ym; θ, xm) − ℓ
∗m(θ|κ
∗)) µ(dθ)
κ=κ∗
= ∇∇κ
Zq(θ|κ)(X
M
m=1
Lm(ym; θ, xm) −X
M
m=1
ℓ
∗m(θ|κ
∗)) µ(dθ)
κ=κ∗
= ∇∇κ
Zq(θ|κ) log q
∗
s(θ|κ
∗)
π(θ) exp{PM
m=1 Lm(ym; θ, xm)}
µ(dθ)
κ=κ∗
+
✘✘✘✘✘✘✘✘✘✘✘✘✘✿0
∇∇κ
Zq(θ|κ) log Zq
∗ µ(dθ)
=
Z(∇∇κq(θ|κ)) log q
∗
s(θ|κ
∗)
π(θ) exp{PMm=1 Lm(ym; θ, xm)}
µ(dθ)
κ=κ∗
which is a sum of positive definite matrices, and therefore, the extremum at the fixed point is a minimum.
We now compare this with the Hessian of the global objective of GVI. ∇∇κObj(q ∗ s) = ∇∇κ Zq(θ|κ) log q(θ|κ) π(θ) exp{−PMm=1 Lm(ym; θ, xm)} µ(dθ) = ∇κ  Z(∇κq(θ|κ)) log q(θ|κ) π(θ) exp{−PMm=1 Lm(ym; θ, xm)} µ(dθ) + ✘✘✘✘✘✘✘✘✘✘✘✘✿0 Z(∇κ log q(θ|κ))q(θ|κ) µ(dθ)   = Z(∇∇κq(θ|κ)) log q(θ|κ) π(θ) exp{−PM m=1 Lm(ym; θ, xm)} µ(dθ) ✘✘✘✘✘✘✘✘✘✘✘✘✘✘✘✘✿0 Z(∇κq(θ|κ))(∇κ log q(θ|κ)) µ(dθ) +
Therefore, we can see that, evaluated at κ = κ ∗, ∇∇κObj(q ∗ s) κ=κ∗ = Z(∇∇κq(θ|κ)) log q(θ|κ) π(θ) exp{−PM m=1 Lm(ym; θ, xm)} µ(dθ) κ=κ∗ = Z(∇∇κq(θ|κ)) log q ∗ s(θ|κ ∗) π(θ) exp{−PM m=1 Lm(ym; θ, xm)} µ(dθ) κ=κ∗ =X M m=1 ∇∇κObj(m, q∗ s) κ=κ∗
20