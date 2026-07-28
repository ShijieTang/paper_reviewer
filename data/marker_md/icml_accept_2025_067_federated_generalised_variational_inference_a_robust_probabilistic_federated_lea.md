---

# Federated Generalised Variational Inference: A Robust Probabilistic Federated Learning Framework

---

Terje Mildner<sup>1</sup> Oliver Hamelijnck<sup>1</sup> Paris Giampouras<sup>1</sup> Theodoros Damoulas<sup>1,2</sup>

## Abstract

We introduce FEDGVI, a probabilistic Federated Learning (FL) framework that is robust to both prior and likelihood misspecification. FEDGVI addresses limitations in both frequentist and Bayesian FL by providing unbiased predictions under model misspecification, with calibrated uncertainty quantification. Our approach generalises previous FL approaches, specifically Partitioned Variational Inference (Ashman et al., 2022), by allowing robust and conjugate updates, decreasing computational complexity at the clients. We offer theoretical analysis in terms of fixed-point convergence, optimality of the cavity distribution, and provable robustness to likelihood misspecification. Further, we empirically demonstrate the effectiveness of FEDGVI in terms of improved robustness and predictive performance on multiple synthetic and real world classification data sets.

## 1. Introduction

Federated learning (FL) is a framework for the collaborative training of a global model by a collection of clients, without requiring proprietary data to be shared with a central server or other participating clients (McMahan et al., 2017). This decentralised approach allows FL to be used on applications with strict data privacy constraints, such as in finance or healthcare (Kairouz et al., 2021). However, due to the sensitive nature and complexity of these domains, both privacy and robustness to model misspecification are paramount.

The frequentist formulation of FL aims to minimise a global loss function by aggregating local gradients from clients. Early works include Federated Averaging (FEDAVG, McMahan et al., 2017) which iterates between training clients lo-

cally and averaging updates on the server. This has sparked a large body of research on issues such as communication efficiency, data privacy, and data heterogeneity across clients (Hamer et al., 2020; Malinovsky et al., 2020; Reddi et al., 2021; Chen et al., 2022; Tenison et al., 2023; Tziotis et al., 2023; Li et al., 2024; Demidovich et al., 2025). There has been some work addressing robustness to adversarial clients (Allouah et al., 2024; Bao et al., 2024) and data and system heterogeneity (Chen et al., 2022; Zhao et al., 2023; Heikkilä et al., 2023). However, these only provide point estimates, and do not allow principled uncertainty quantification, as required in many FL applications (Jonker et al., 2024).

In contrast, Bayesian FL approaches aim to update beliefs of a global model with data partitioned across clients. This largely builds on distributed inference methods such as the Bayesian Committee Machine (Tresp, 2000), parallel MCMC (Ahn et al., 2014; Mesquita et al., 2020), or Divide&Conquer SMC (Chan et al., 2023). Expectation Propagation (Minka, 2001; Vehtari et al., 2020) is naturally applicable to the distributed setting where local sites are iteratively refined. This requires computing the cavity distribution that removes local sites from the current approximation. Partitioned Variational Inference (PVI, Bui et al., 2018; Ashman et al., 2022) takes this idea and proposes a distributed variational inference algorithm, which has been extended through MCMC (Guo et al., 2023) and Stochastic Gradient Langevin Dynamics (SGLD) (Mekkaoui et al., 2021). Whilst these approaches quantify uncertainty, they are susceptible to model misspecification which can lead to inaccurate, overconfident predictions (Bernardo & Smith, 2000; Bissiri et al., 2016; Knoblauch et al., 2022).

Current approaches to FL are inherently non-robust to model misspecification which leads to compromised performance and uncalibrated uncertainty quantification. We address these challenges by departing from the traditional Bayesian paradigm and propose a distributed Generalised Variational Inference framework that allows us to deal with model misspecification. In summary, our contributions are:

<sup>1</sup>University of Warwick, Department of Computer Science, Coventry, United Kingdom <sup>2</sup>University of Warwick, Department of Statistics, Coventry, United Kingdom. Correspondence to: Terje Mildner <Terje.Mildner@warwick.ac.uk>.

*Proceedings of the 4<sup>th</sup> International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

- • We introduce Federated Generalised Variational Inference (FEDGVI), a family of robust probabilistic algorithms for federated learning.

- • We prove that FEDGVI is robust to likelihood misspecification (Theorem 4.12).
- • We demonstrate that FEDGVI generalises standard approaches such as PVI and FEDAVG (Remarks 4.1 and 4.2) and theoretically justify the use of the cavity distribution (Theorem 4.9).
- • We prove that, under suitable conditions, FEDGVI converges to Generalised Bayesian posteriors (Lemma 4.6 and Proposition 4.10) that are computationally tractable.
- • We evaluate FEDGVI on a range of synthetic and real-world datasets, across multiple models, demonstrating improved robustness and predictive performance.

In Section 2 we define model misspecification and recall methods that mitigate it in the non-distributed setting. Section 3 introduces our framework, which builds on these concepts and extends them to the federated setting. We analyse the theoretical properties of FEDGVI in Section 4, including provable robustness. Finally, Section 5 studies the empirical performance and gains of FEDGVI with multiple models and real world datasets such as Bayesian Neural Networks on MNIST and FASHIONMNIST.<sup>1</sup>

### 1.1. Related Work

**Robust Frequentist Federated Learning** In the frequentist setting, building on the seminal paper of McMahan et al. (2017), many approaches have aimed at mitigating challenges in FL, such as robustness to adversarial servers through secure aggregation (Chen et al., 2022), to stragglers (Tziotis et al., 2023), heterogenous data in out-of-distribution generalisation (Tenison et al., 2023), heterogeneous and asynchronous clients (Fraboni et al., 2023), or finding weaknesses in communications (Zhu et al., 2019; Zhao et al., 2023). More recently, work on robust server aggregations achieves robustness against Byzantine clients that aim to deteriorate model performance (Allouah et al., 2024; Bao et al., 2024). However these do not allow principled uncertainty quantification.

**Federated Bayesian Inference** Federated and distributed Bayesian methods aim to approximate the posterior as if it had been computed with the data of all clients available at a central server. Early work on distributed Bayesian inference includes Bayesian opinion pools (Genest, 1984; Carvalho et al., 2023), and the Bayesian Committee machine (Tresp, 2000), which aim to find a consensus among a collection of Bayesian beliefs. Works that aim to operationalise this in the distributed setting, where data is split IID across clients,

include Expectation Propagation (Minka, 2001; Opper & Winther, 2005; Hasenclever et al., 2017; Vehtari et al., 2020), and consensus based Monte Carlo (Scott et al., 2016). In the Federated setting this assumption is often violated, as data is not split homogeneously and IID across participating devices. From this perspective, most approaches to Bayesian FL can be categorised into finding an approximate posterior through variational inference (Corinzia et al., 2021; Ashman et al., 2022; Kassab & Simeone, 2022; Heikkilä et al., 2023; Hassan et al., 2024; Vedadi et al., 2024; Swaroop et al., 2025), Markov Chain Monte Carlo (Al-Shedivat et al., 2021; Mekkaoui et al., 2021; Kotelevskii et al., 2022; Guo et al., 2023; Hasan et al., 2024), Gaussian Processes (Achituve et al., 2021), or directly learning a Bayesian neural network (Yurochkin et al., 2019; Zhang et al., 2022). Personalised or hierarchical Bayesian FL (Kotelevskii et al., 2022; Zhang et al., 2022; Kim & Hospedales, 2023; Hassan et al., 2023; 2024; Vedadi et al., 2024) allows for additional expressibility of client posteriors, especially under heterogeneity. However, none of these are inherently robust to contamination and model misspecification.

**Robust Bayesian Inference** Although the existing Bayesian FL methods address some of the challenges of federated learning, such as communication constraints and data heterogeneity, they still aim to approximate the Bayesian posterior, which in itself is a flawed objective under model misspecification (Walker, 2013; Berk, 1966; Bernardo & Smith, 2000). In the global, non-federated case, several methods have been proposed to combat misspecification in the Bayesian setting (Grünwald, 2012), with the most promising direction being Generalised Bayesian Inference (Hooker & Vidyashankar, 2014; Bissiri et al., 2016; Ghosh & Basu, 2016a; Jewson et al., 2018; Miller, 2021; Alquier, 2021; Knoblauch et al., 2022; Matsubara et al., 2022). In this work we capitalise on this front and bring robustness to model misspecification in the federated setting.

## 2. Preliminaries

### 2.1. Notation and Model Misspecification

Let  $(\Omega, \mathcal{F}, P_0)$  be a probability space where  $P_0$  is the data generating process, generating the observable random variables  $X_1, \dots, X_n \equiv X_1^n$  taking values in the measurable space  $(\Xi, \mathcal{X})$ . Further, let  $Y_1^n$  be observable random variables depending on  $X_1^n$  respectively, taking values in  $(\Upsilon, \mathcal{Y})$ . Denote their realisations  $\{X_i = x_i, Y_i = y_i\}_{i=1}^n$ , which are assumed to be partitioned across  $M$  clients  $\{\mathbf{x}_m, \mathbf{y}_m\}_{m=1}^M$  each of size  $n_m$ . Consider hypothesis measures  $P_\theta$  where  $\theta$  takes values in  $(\Theta, \mathcal{T})$ , a measurable space, admitting densities  $p_\theta$ . We study elements of  $\mathcal{P}(\Theta)$ , the set of all probability measures on  $(\Theta, \mathcal{T})$ , starting with prior  $\Pi$  and updated to  $Q$ , dominated by some common measure  $\mu$ , and

<sup>1</sup>Code to reproduce experiments can be found at <https://github.com/Terje-M/FedGVI>.

Algorithm 1 FEDGVI SERVER 1: Input: π(θ), Q, D<sup>s</sup>

2: Define: ℓ

(0)

<sup>m</sup> (θ) = 0, ℓ

(0)

<sup>s</sup> (θ) = 0, q

(0)

<sup>s</sup> (θ) = π(θ)

3: for t = 1, ..., T do

4: for m = 1, ..., M in parallel do

5: ∆

(t)

<sup>m</sup> (θ) ←CLIENT(q

(t−1)

<sup>s</sup> (θ), Q, m)

6: end for 7: Set ℓ

(t) <sup>s</sup> (θ) ← ℓ

(t−1)

<sup>s</sup> (θ) + P<sup>M</sup>

<sup>m</sup>=1 ∆ (t) <sup>m</sup> (θ)

8: Optimise q

(t)

<sup>s</sup> (θ) according to Equation [\(7\)](#page-3-0)

9: end for

admitting densities π and q respectively. Naive Bayes updates π(θ) to qB(θ) through

$$q_B(\boldsymbol{\theta}) = \pi(\boldsymbol{\theta}) \prod_{m=1}^M p_{\boldsymbol{\theta}}(\mathbf{y}_m; \mathbf{x}_m) / Z \quad (1)$$

where Z = R Θ Q<sup>M</sup> <sup>m</sup>=1 <sup>p</sup>θ(ym; <sup>x</sup>m) Π(dθ) is the marginal likelihood. Since we do not suppose that the prior Π, nor the likelihood P<sup>θ</sup> are well specified, i.e. P<sup>0</sup> ∈ P/ (Θ), we are in the M–open setting [\(Bernardo & Smith,](#page-9-4) [2000\)](#page-9-4), the model misspecified, and the Bayesian posterior inappropriate.

### 2.2. Model Misspecification

There are several different ways we can think about model misspecification under the M–open assumption.

Prior Misspecification The traditional Bayesian paradigm assumes that the prior encodes the best available judgement about θ, which beyond simple settings, is never realised [\(Berger,](#page-9-9) [1985;](#page-9-9) [Knoblauch et al.,](#page-11-12) [2018\)](#page-11-12). Such misspecification is common; e.g. it is standard to use zero–mean Gaussian distributions on the weights of Bayesian Neural networks. This can have dire effects, for instance [Diaconis & Freedman](#page-10-19) [\(1986\)](#page-10-19) demonstrate that multimodal priors in a location model can cause the posterior to not accumulate around P0, even when the DGP is well specified, i.e. when P<sup>0</sup> ∈ P(Θ).

Likelihood Misspecification One such example is where the hypothesis of interest is contaminated , and an ε fraction of the data (input and/or output variables) has some unknown data source. Formalising this we follow the definition of [Huber](#page-11-13) [\(1964\)](#page-11-13):

Definition 2.1 (Huber contamination). Given an ε ∈ (0, 1 2 ) and the uncontaminated distribution P<sup>θ</sup> of inliers and some contaminating distribution G of outliers, then P<sup>0</sup> is said to be an ε*-corrupted version of* Pθ; P<sup>0</sup> := (1 − ε)P<sup>θ</sup> + εG.

# 2.3. Robust Bayesian Methods

| Algorithm |          | 2 F ED GVI C LIENT                    |
|-----------|----------|---------------------------------------|
| 1:        | Input:   | q                                     |
|           |          | ( t − 1)                              |
|           |          | s ( θ ) , Q , { x m , y m } , L m , ℓ |
|           |          | ( t − 1)                              |
|           |          | m ( θ ) , D                           |
| 2:        | Optimise | q                                     |
|           |          | \ m ( θ ) according to Equation (3)   |
| 3:        | Optimise | q                                     |
|           |          | ( t )                                 |
|           |          | m ( θ ) according to Equation (4)     |
| 4:        | Set      | ∆                                     |
|           |          | ( t )                                 |
|           |          | m ( θ ) according to Equation (5)     |
| 5:        | Set      | ℓ                                     |
|           |          | ( t )                                 |
|           |          | m ( θ ) ← ℓ                           |
|           |          | ( t − 1)                              |
|           |          | m ( θ ) + ∆ ( t )                     |
|           |          | m ( θ )                               |
| 6:        | return:  | Communicate ∆                         |
|           |          | ( t )                                 |
|           |          | m ( θ ) to SERVER                     |

[\(2016\)](#page-10-7) and [Miller](#page-12-15) [\(2021\)](#page-12-15) formalised a coherent Bayesian framework using loss functions leading to Gibbs posteriors [\(Alquier et al.,](#page-9-10) [2016\)](#page-9-10). This was further utilised to deal with likelihood misspecification through robust losses, e.g [Knoblauch et al.](#page-11-12) [\(2018\)](#page-11-12). Let L : Θ × Ξ × Υ → <sup>R</sup> be such a loss, then the GBI posterior is given by:

$$q_{\text{GBI}}(\boldsymbol{\theta}) = \pi(\boldsymbol{\theta}) \exp \left\{ -\beta \sum_{m=1}^M L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) \right\} / Z \quad (2)$$

with Z = R Θ exp{−β P<sup>M</sup> <sup>m</sup>=1L(ym; <sup>θ</sup>, <sup>x</sup>m)}Π(dθ). Here, β ∈ <sup>R</sup>><sup>0</sup> is a learning rate parameter that determines how much weight we place on the observed data, similar to power posteriors in VI [\(Grunwald](#page-10-16) ¨ , [2012;](#page-10-16) [Kallioinen et al.,](#page-11-14) [2024\)](#page-11-14). This recovers qB(θ) when the loss is the negative log–likelihood and β = 1.

Generalised Variational Inference (GVI) In [Knoblauch](#page-11-6) [et al.](#page-11-6) [\(2022\)](#page-11-6) GBI is generalised within a variational framework that explicitly accounts for prior and likelihood misspecification. Let D : P(Θ)×P(Θ) → <sup>R</sup><sup>+</sup> be a divergence then the GVI posteriors are defined as:

$$q_{\text{GVI}}(\boldsymbol{\theta}) = \arg \min_{q \in \mathcal{Q}} \{ \mathbb{E}_{q(\boldsymbol{\theta})} [L(\mathbf{y}_1^M; \boldsymbol{\theta}, \mathbf{x}_1^M)] + D(q : \pi) \}$$

where Q ⊂ P(Θ), making inference tractable. This allows for targeting a larger subspace of posteriors, and through different divergences the effect of the prior can be controlled.

# 3. Federated Generalised Variational Inference

### 3.1. Methodology

In this section, we present the proposed federated learning framework, named FEDGVI, that explicitly addresses likelihood and prior misspecification. We aim to learn a robust approximate posterior qs(θ) using partitioned observations across M clients. FEDGVI iterates consist of two steps: a) sending of the current approximate posterior to each client, which is updated through a robust variational objective, and b) aggregating the updates on the server, resulting in a robust approximate posterior; summarised in Algorithms [1](#page-2-1) and [2.](#page-2-2)

Initialisation We set the initial server posterior as the prior, q (0) <sup>s</sup> (θ) = π(θ), and the local and server loss approximations to be zero, ℓ (0) <sup>m</sup> (θ) = 0 and ℓ (0) <sup>s</sup> (θ) = 0 respectively; m denotes a specific client and s the server.

Until Convergence For t = 1, 2, ..., T, we synchronously compute updates locally at each client, and accumulate these at the server to form the new global posterior q (t) <sup>s</sup> (θ).

Client The client receives the current approximate posterior from the server. This will be used as the prior from which a client can compute an updated posterior using their local data. First, however the information of the client's data must be removed by computing the cavity distribution. The cavity distribution acts as the local prior incorporating all previous information from all other clients and is given by:

$$q^{\setminus m}(\boldsymbol{\theta}) \propto \frac{q_s^{(t-1)}(\boldsymbol{\theta})}{\exp\{-\ell_m^{(t-1)}(\boldsymbol{\theta})\}} \quad (3)$$

The client then computes a robust local approximate posterior with it's local data set {xm, ym} and it's loss function L (t) <sup>m</sup> (·), which is regularised by the divergence, D, and cavity distribution

$$q_m^{(t)}(\boldsymbol{\theta}) = \arg \min_{q \in \mathcal{Q}} \mathbb{E}_{q(\boldsymbol{\theta})} \left[ L_m^{(t)}(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) \right] + D(q : q^{(m)}). \quad (4)$$

This GVI style objective allows the client to be robust to both likelihood misspecification as well as prior misspecification arising due to the cavity. To update the global posterior at the server, the client computes the negative log ratio of the local and global posteriors. In line with existing Bayesian FL [\(Ashman et al.,](#page-9-0) [2022;](#page-9-0) [Guo et al.,](#page-10-6) [2023\)](#page-10-6), we use a damping parameter τ<sup>m</sup> ∈ (0, 1], which is analogous to a learning rate as in frequentist FL, to compute the update:

$$\Delta_m^{(t)}(\boldsymbol{\theta}) = -\tau_m \log \frac{q_m^{(t)}(\boldsymbol{\theta})}{q_s^{(t-1)}(\boldsymbol{\theta})} \quad (5)$$

The client stores ℓ (t) <sup>m</sup> (θ) := ℓ (t−1) <sup>m</sup> (θ) + ∆(t) <sup>m</sup> (θ) and communicates ∆ (t) <sup>m</sup> (θ) to the server.

Server The loss at the server is updated based on the received client updates,

$$\ell_s^{(t)}(\boldsymbol{\theta}) = \ell_s^{(t-1)}(\boldsymbol{\theta}) + \sum_{m=1}^M \Delta_m^{(t)}(\boldsymbol{\theta}) \quad (6)$$

By only incorporating clients' updates that have changed we can trivially allow for batched and asynchronous scheduling of clients. The updated loss is then used to compute the new server posterior though a GVI optimisation procedure:

$$q_s^{(t)}(\boldsymbol{\theta}) = \arg \min_{q \in \mathcal{Q}} \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \ell_s^{(t)}(\boldsymbol{\theta}) \right] + D_s(q : \pi) \quad (7)$$

This posterior and loss are passed back to the clients for further refinement at the next iteration until convergence.

### 3.1.1. HYPERPARAMETERS

[Ashman et al.](#page-9-0) [\(2022\)](#page-9-0) set the damping parameter to τ ∝ M throughout their experiments. This turns out, see Proposition [4.3,](#page-4-4) to be a reasonable choice when τ = <sup>M</sup> in combination with D<sup>s</sup> = DKL since this causes the posterior at the server to be a logarithmic opinion pool induced by an externally Bayesian pooling operator [\(Genest et al.,](#page-10-20) [1986\)](#page-10-20), ensuring stable convergence. Other hyperparameters arising from the choice of losses and divergences are dependent on the expected amount of model misspecification.

### 3.2. Robustness to Likelihood Misspecification

Within our framework we are free to choose the client side losses. We consider the Density–Power divergence based loss [\(Ghosh & Basu,](#page-10-21) [2016b\)](#page-10-21), often referred to as β– divergence loss Lβ, the γ–divergence based losses [\(Hung](#page-11-15) [et al.,](#page-11-15) [2018\)](#page-11-15), Lγ, as well as a score matching loss, LSM, based on the Hyvarinen divergence ( ¨ [Hyvarinen](#page-11-16) ¨ , [2005;](#page-11-16) [Al](#page-9-11)[tamirano et al.,](#page-9-11) [2023\)](#page-9-11). In the classification setting, we consider the generalised cross–entropy loss

$$\mathcal{L}_{GCE}^{(\delta)}(y_i; \boldsymbol{\theta}, x_i) = \frac{(1 - p\boldsymbol{\theta}(y = y_i; x_i)^\delta)}{\delta} \quad (8)$$

for some δ ∈ (0, 1] [\(Zhang & Sabuncu,](#page-12-16) [2018\)](#page-12-16). These losses are robust to misspecification because they have a finite supremum (see Definition [4.11\)](#page-5-3). It is important to highlight that GVI and FEDGVI may underperform when using robust losses in the case of correct likelihood specification; see [Knoblauch et al.](#page-11-6) [\(2022\)](#page-11-6). We can use a Sequential Monte Carlo sampler to estimate the β or γ hyperparameters in L<sup>β</sup> and L<sup>γ</sup> [\(Yonekura & Sugasawa,](#page-12-17) [2023\)](#page-12-17) or use cross validation to select optimal parameters [\(Altamirano et al.,](#page-9-12) [2024\)](#page-9-12).

### 3.3. Robustness to Prior Misspecification

We mainly consider the weighted Kullback–Leiber divergence, <sup>1</sup> <sup>w</sup> <sup>D</sup>KL, [\(Kullback & Leibler,](#page-11-17) [1951\)](#page-11-17)

$$\frac{1}{w} D_{KL}(q : \pi) := \frac{1}{w} \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \log \frac{q(\boldsymbol{\theta})}{\pi(\boldsymbol{\theta})} \right],$$

and the Alpha–Renyi divergence, ´ D (α) AR,

$$D_{AR}^{(\alpha)}(q : \pi) := \frac{1}{\alpha(\alpha - 1)} \log \left( \mathbb{E}_{\pi(\boldsymbol{\theta})} \left[ \left( \frac{q(\boldsymbol{\theta})}{\pi(\boldsymbol{\theta})} \right)^{\alpha} \right] \right).$$

As examined in [Knoblauch et al.](#page-11-6) [\(2022\)](#page-11-6), D (α) AR allows for different prior regularisation depending on how much we trust the prior by placing different weights on it. In future work it would be simple to explore other divergences such as the f–divergences, D<sup>f</sup> , [\(Amari,](#page-9-13) [2016;](#page-9-13) [Alquier,](#page-9-8) [2021\)](#page-9-8). Similarly to the losses, we can perform cross validation to select the α parameter, however as demonstrated in the ablation study (Figure [6\)](#page-7-0) FedGVI performs favourably under a range of α (and δ) values.

### 4. Theoretical Results

We now present a theoretical analysis of FEDGVI. We begin by examining the relationship of FEDGVI with other FL algorithms while recovering some of them as special cases, we study the damping parameter, and examine the convergence behaviour of FEDGVI. Then, we turn our attention on robustness to likelihood misspecification, where we first study FEDGVI as distributed GBI, from which we derive a theorem on the necessity of the cavity distribution. Finally, we derive a result for computationally tractable and conjugate FEDGVI, enabling us to present the main theorem on bias–robustness of FEDGVI.

Since it is an open problem where global GVI posteriors converge to under arbitrary divergences, we often have to restrict ourselves to consider the server divergence to be the Kullback–Leibler divergence. This ensures that the posterior at the server will have the structure of a GBI posterior,

$$q_s^{(T)}(\boldsymbol{\theta}) \propto \exp \left\{ - \sum_{m=1}^M \ell_m^{(T)}(\boldsymbol{\theta}) \right\} \pi(\boldsymbol{\theta})$$

where we incorporate prior robustness and tractability through the approximate losses.

### 4.1. Recovering Existing Methods as a Special Case

By choosing specific divergences, loss functions, and variational families, we can recover existing methods as special cases of our framework, which we summarise in Figure [1:](#page-4-5) *Remark* 4.1*.* Choosing the Kullback–Leibler divergence and the negative log–likelihood as a loss function recovers the PVI algorithm of [Ashman et al.](#page-9-0) [\(2022\)](#page-9-0).

*Remark* 4.2*.* When <sup>D</sup> <sup>=</sup> <sup>D</sup><sup>s</sup> = 0, and <sup>Q</sup> <sup>=</sup> {δθ<sup>ˆ</sup>(θ) : <sup>θ</sup><sup>ˆ</sup> <sup>∈</sup> <sup>Θ</sup>}, with <sup>δ</sup>θ<sup>ˆ</sup> being the Dirac–delta measure at some element θˆ, we recover FEDAVG of [McMahan et al.](#page-11-0) [\(2017\)](#page-11-0).

![](_page_4_Diagram_20.jpeg)

Figure 1: We illustrate the relationship of FEDGVI characterised by the loss L, the client divergence D, the variational family Q, the number of clients M, and the divergence at the server Ds—to Partitioned Variational Inference (PVI), Variational Inference (VI), Federated Averaging (FEDAVG), and Empirical Risk Minimisation (ERM).

### 4.2. Damping as a Bayesian Logarithmic Opinion Pool

Choosing the damping parameter to be τ = 1/M results in a logarithmic opinion pool. In fact choosing damping parameters such that all of them sum to unity also forms a valid logarithmic opinion pool [\(Genest et al.,](#page-10-20) [1986\)](#page-10-20).

Proposition 4.3. *Assume* D<sup>s</sup> = DKL*, and that* P <sup>m</sup> τ<sup>m</sup> = 1 *where* τ<sup>m</sup> ≥ 0 ∀m*, then the posterior at the server is an externally Bayesian logarithmic opinion pool of the form*

$$q_s^{(t)}(\boldsymbol{\theta}) = \frac{\prod_{m=1}^M (q_m^{(t)}(\boldsymbol{\theta}))^{\tau_m}}{\int_{\Theta} \prod_{m=1}^M (q_m^{(t)}(\boldsymbol{\theta}))^{\tau_m} d\boldsymbol{\theta}}, \quad \boldsymbol{\theta} - a.e.$$

See Appendix [B.2](#page-15-0) for the proof. This results provides a theoretical justification on the previously heuristic use of the damping parameter (as used in PVI, [Ashman et al.,](#page-9-0) [2022\)](#page-9-0). Specifically it ensures that this selection of τ leads to a valid distribution and results in more stable convergence.

### 4.3. Fixed Points of FEDGVI

In this section we study the properties of FEDGVI posteriors when these converge to some fixed point. Specifically, we generalise the fixed point result of PVI [\(Ashman et al.,](#page-9-0) [2022,](#page-9-0) Property 2.3) to arbitrary losses.

Proposition 4.4. *Let* D<sup>s</sup> = DKL*,* D = 1 <sup>w</sup> DKL*,* w > 0*, and* Q ⊂ P(Θ)*, then if* q ∗ s (θ) = π(θ) exp{−ℓ ∗ s (θ)}/Z<sup>q</sup> ∗ *such that* ∀m ∈ [M]*,* ∆<sup>∗</sup> <sup>m</sup>(θ) = 0*, then* q ∗ s (θ) *is a local minimiser of the following GVI objective:*

$$\mathbb{E}_{q(\boldsymbol{\theta})} \left[ \sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) \right] + \frac{1}{w} D_{KL}(q : \boldsymbol{\pi}) \quad (9)$$

*Remark* 4.5*.* If the loss in Equation [\(9\)](#page-4-6) is convex, then a fixed point of FEDGVI is a global minimum of GVI.

This illustrates that if FEDGVI converges, then the posterior is a (local) minimiser of the GVI objective. We refer to such distributions as fixed points. This recovers [Kassab &](#page-11-7) [Simeone](#page-11-7) [\(2022,](#page-11-7) Theorem 1) (which deals with the restricted case of Q = P(Θ)) with a novel proof; see Appendix [B.3.](#page-16-0)

### 4.4. Generalised Bayesian Inference

As a consequence of Proposition [4.4](#page-4-7) and Remark [4.1,](#page-4-0) FEDGVI will recover the GBI posterior when Q = P(Θ).

Lemma 4.6. *Assuming* Q = P(Θ)*,* D = 1 <sup>β</sup> DKL *with* β > 0*,* D<sup>s</sup> = DKL*, and* τ = 1*, then* FEDGVI *will recover the GBI posterior after the first iteration.*

$$q_s^{(1)}(\boldsymbol{\theta}) = q_{GBI}(\boldsymbol{\theta} | \{\mathbf{x}_m, \mathbf{y}_m\}_{m=1}^M) \\ = \exp\{-\beta \sum_{m=1}^M L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\} \pi(\boldsymbol{\theta}) / L$$

*This posterior is invariant under subsequent iterations of* FEDGVI*, having reached a fixed point.*

*Moreover, for a damping rate* τ = 1/M*, the posterior at the server converges pointwise a.e. in* Θ *to the GBI posterior,*

$$q_s^{(T)}(\boldsymbol{\theta}) \xrightarrow{T \rightarrow \infty} q_{GBI}(\boldsymbol{\theta} | \{\mathbf{x}_m, \mathbf{y}_m\}_{m=1}^M), \quad \boldsymbol{\theta} = a.e.$$

This result, proven in Appendix [B.4,](#page-20-0) is the first step towards likelihood robustness. If we were able to find the GBI posterior efficiently with some robust loss, then the posterior would be robust and computable. Here however, the loss may not vary over different iterations of FEDGVI as in Equation [\(4\)](#page-3-2) and the normaliser may be intractable.

### 4.5. The Cavity Distribution is Necessary

By further investigating the relationship of FEDGVI with the GBI posterior, we can extend Lemma [4.6](#page-4-2) and derive a Theorem under which we are required to use the cavity distribution to regularise the client update. This is in contrast to both PVI, where it's use is heuristically justified, and to other Bayesian FL approaches where the previous posterior is used instead. For this we recall two natural assumptions that any such distribution must satisfy in a federated setting.

Assumption 4.7. No client can have access to the data set of another client.

Assumption 4.8. Each client generates their update equivalently to other clients.

These assumptions combined with Lemma [4.6](#page-4-2) lead us to the necessity of the cavity distribution.

Theorem 4.9. *Let the assumptions be as in Lemma [4.6](#page-4-2) with* τ = 1*, and assume that the Assumptions [4.7](#page-5-4) and [4.8](#page-5-5) are satisfied, then* (1.) *holds if and only if* (2.) *holds.*

- 1. FEDGVI *recovers the generalised Bayesian posterior* qGBI(θ) *which is invariant under further* FEDGVI *updates.*
- 2. *The cavity regularises the client optimisation problem.*

This provides a principled justification for the use of the cavity distribution, as defined in Equation [\(3\)](#page-3-1), in FEDGVI. We provide the proof in Appendix [B.5.](#page-23-0)

## 4.6. Conjugate Client Updates

Before we present our main result on provable robustness to likelihood misspecification, we first show that we can find a GBI posterior under specific losses in a computationally tractable manner. Assuming that the data generating process has some exponential family distribution, where y ∼ pθ(y),

$$p_{\boldsymbol{\theta}}(\mathbf{y}) = \exp\{\boldsymbol{\eta}(\boldsymbol{\theta})^\top \boldsymbol{\phi}(\mathbf{y}) - A(\boldsymbol{\eta}(\boldsymbol{\theta})) + h(\mathbf{y})\},$$

such that this is differentiable in y, by using the weighted score matching loss of [Altamirano et al.](#page-9-11) [\(2023\)](#page-9-11), L w SM, then client updates, using the weighted KL divergence locally, are available in closed form. If we further assume that our

model is Gaussian, or has the form of a squared exponential, and that the natural parameters of the DGP are η(θ) = θ, then the client approximation will have a conjugate form.

Proposition 4.10. *Assume that the hypothesis* pθ(y) *has differentiable, exponential family distribution with* η(θ) = θ*,* L (t) <sup>m</sup> = L w t m SM*, and* D = <sup>β</sup> DKL*, and the variational family* Q *is the multivariate Gaussians, then the local posteriors at the clients are conjugate Gaussians. Moreover, Equation* [\(7\)](#page-3-0) *will have closed form if* D<sup>s</sup> *has closed form between Gaussian distributions.*

See Appendix [B.6](#page-25-0) for the proof. The loss may now depend on the client and iteration t. Most exponential family distributions satisfy the conditions of the proposition, and there are several divergences that allow closed form expressions between Gaussians, such as the Alpha–Renyi , or the ´ α, β, γ–divergences of [Cichocki & Amari](#page-10-22) [\(2010\)](#page-10-22). Further, this enables the use of intractable likelihood models.

### 4.7. Provable Robustness to Outliers

For a robust loss function at the clients, and using the weighted KL divergence at the clients and the KL divergence at the server, guarantees that after T iterations, the posterior computed at the server will also be robust to outliers. This means we can achieve robustness at the server by leveraging the robust losses that were derived for GVI. In this, we mean robustness as defined by [Ghosh & Basu](#page-10-18) [\(2016a\)](#page-10-18) and further developed in [Matsubara et al.](#page-11-11) [\(2022\)](#page-11-11). We define the empirical DGP of a client as <sup>P</sup>n<sup>m</sup> := <sup>1</sup> n<sup>m</sup> P<sup>n</sup><sup>m</sup> <sup>i</sup>=1 δ<sup>x</sup><sup>i</sup> , and of the entire data set as <sup>P</sup><sup>n</sup> := <sup>1</sup> n P<sup>M</sup> <sup>m</sup>=1 <sup>n</sup>mPnm. When this is contaminated by some ε fraction of data centred at some adversarially chosen data point z ∈ Ξ, the misspecified DGP is defined as <sup>P</sup>n,ε,z := (1 − ε)<sup>P</sup><sup>n</sup> + εδz.

Definition 4.11. We say that a loss L (t) <sup>m</sup> (θ; <sup>P</sup>nm,ε,z), w.r.t. some prior distribution π(θ), is robust to outliers, if the following hold:

- 1. sup z∈Ξ d dε<sup>L</sup>
- (t) <sup>m</sup> (θ; <sup>P</sup>nm,ε,z) ε=0 ≤ γ
- (t)
- (m) (θ),
- 2. sup θ∈Θ π(θ)γ
- (t)
- (m) (θ) < ∞, and
- 3. Z Θ π(θ)γ
- (t)
- (m) (θ)µ(dθ) < ∞

1. 1. 
   $$\sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m, \varepsilon, z}) \right|_{\varepsilon=0} \leq \gamma_{(m)}^{(t)}(\boldsymbol{\theta}),$$
2. 2.  $\sup_{\boldsymbol{\theta} \in \Theta} \pi(\boldsymbol{\theta}) \gamma_{(m)}^{(t)}(\boldsymbol{\theta}) < \infty,$  and
3. 3.  $\int_{\Theta} \pi(\boldsymbol{\theta}) \gamma_{(m)}^{(t)}(\boldsymbol{\theta}) \mu(d\boldsymbol{\theta}) < \infty$

These conditions ensure that the influence of arbitrary contamination on the local posterior is not arbitrarily bad. In particular the auxiliary function γ (t) <sup>m</sup> ensures that the influence of an adversarial data point z on the posterior over infinitesimal contaminations, <sup>d</sup> dϵ q (t) <sup>m</sup> (θ; <sup>P</sup>nm,ϵ,z)|ϵ=0, are finite over all θ and z. Condition 2 ensures the loss increases slowly enough for the local posterior to concentrate around the data, and condition 3 ensures the resulting posterior will be normalisable.

Theorem 4.12. *Let* D<sup>s</sup> = DKL*,* D = 1 <sup>w</sup> DKL*,* Q = P(Θ)*, further assume that the prior is upper bounded and the loss is lower bounded, then if* ∀t ∈ [T] *and* ∀m ∈ [M] L (t) <sup>m</sup> (θ; <sup>P</sup>nm,ε,z) *is robust, then the posterior generated by* FEDGVI *will be robust to outliers.*

The proof is in Appendix [B.7.](#page-26-0) This result together with Proposition [4.10](#page-5-2) is significant as we have robustness under intractable optimisation, *and* we can choose a provably robust, conjugate loss to generate robust FEDGVI posteriors, which are then computationally efficient to compute.

# 5. Experiments

We evaluate FEDGVI against several other methods, specifically PVI [\(Ashman et al.,](#page-9-0) [2022\)](#page-9-0), FEDAVG [\(McMahan](#page-11-0) [et al.,](#page-11-0) [2017\)](#page-11-0), the nonparametric DSVGD [\(Kassab & Sime](#page-11-7)[one,](#page-11-7) [2022\)](#page-11-7), the distributed MCMC based DSGLD [\(Ahn](#page-9-3) [et al.,](#page-9-3) [2014\)](#page-9-3), federated MCMC based FEDPA [\(Al-Shedivat](#page-9-5) [et al.,](#page-9-5) [2021\)](#page-9-5), and the one shot BCM based approach β– PREDBAYES [\(Hasan et al.,](#page-10-14) [2024\)](#page-10-14). We provide further details about experiments in Appendix [D.](#page-34-0)

### 5.1. 1D Clutter Problem

![](_page_6_Figure_10.jpeg)

Figure 2: Robustness to outliers can be achieved through varying losses with FEDGVI, while traditional Bayesian methods fail.

We first examine the effect of misspecified likelihoods through the well known clutter problem [\(Minka,](#page-12-6) [2001\)](#page-12-6). We generate 100 observations from a Gaussian location model that is contaminated through Definition [2.1](#page-2-3) with ε = 0.25 Gaussian noise. The aim is to infer the location parameter θ of the uncontaminated data. We compare FEDGVI with both L<sup>β</sup> and LSM vs PVI with and without misspecification. We also provide the corresponding MLE results. See Figure [2.](#page-6-1) Under misspecification both the MLE and PVI fail to recover the true θ, whereas FEDGVI can easily handle different levels of contamination.

### 5.2. Influence Function

![](_page_6_Figure_3.jpeg)

Figure 3: We plot the influence of a single outlier on the server posterior. PVI is not robust to likelihood misspecification through outliers, because it uses the negative log– likelihood (NLL).

To demonstrate robustness to likelihood misspecification as in Theorem [4.12,](#page-5-0) we consider the influence of a single outlier at one of seven clients on the server posterior. Figure [3](#page-6-2) demonstrates that the negative log likelihood is not robust in the federated setting, whereas different robust divergence based losses allow only limited influence of outliers on the posterior. We plot this as the divergence between the posterior, had we observed the outlier value at the true mean, against the posteriors that have the outlier be farther from the true mean, using the Fisher–Rao distance [\(Nielsen,](#page-12-18) [2023\)](#page-12-18).

## 5.3. 2D Misspecified Logistic Regression

![](_page_6_Figure_12.jpeg)

Figure 4: Logistic Regression decision boundaries (0.2, 0.5, 0.8) for PVI without outliers, PVI with misspecification, and FEDGVI with misspecification. The synthetic data set is split homogeneously across 5 clients where PVI negatively skews the decision boundary, while FEDGVI does not.

We next consider a 2D logistic regression example where we generate 100 linearly separable samples from a Gaussian mixture distribution. We inject outliers generated by a third Gaussian distribution and assign them to one of the classes so that the data is no longer linearly separable. We compare FEDGVI with L (0.7) β and D (1.5) AR against PVI, both with 5 clients. Again, the target is given by PVI only trained on the uncontaminated data. As expected PVI is severely impacted by outliers, whereas FEDGVI is robust to them and closely recovers the target posterior.

### 5.4. Real-World Cover Type Dataset

![](_page_7_Figure_4.jpeg)

Figure 5: Results on the COVERTYPE data set. We place a Gaussian distribution over the weights and average over 10 different train/test splits; see Appendix [D](#page-34-0) for details.

In this experiment we follow the experimental setup of [Kassab & Simeone](#page-11-7) [\(2022\)](#page-11-7) and average accuracy over 10 random 80/20 train-test splits, where the training data is split homogeneously across 2 clients. We do not add any label contamination. The results are plotted in Figure [5.](#page-7-1) The non-robust methods all eventually achieve similar accuracy, however FEDGVI is able to outperform all competing methods, which we argue is due to FEDGVI putting less weight on data points that are less likely to belong to the class.

### 5.5. Bayesian Neural Networks on MNIST and FASHIONMNIST

Table 1: Classification accuracy (highest in bold) on uncontaminated test data after training on 10% contaminated MNIST data. We report the best performance across all server iterations.

|  | MODEL                     | ACCURACY + STD.                      |                                    |
|--|---------------------------|--------------------------------------|------------------------------------|
|  | 10 CLIENTS                | 3 CLIENTS                            |                                    |
|  | FEDAVG                    | 96.64 $\pm$ 0.07                     | 96.34 $\pm$ 0.20                   |
|  | FEDPA                     | 94.25 $\pm$ 0.39                     | 95.31 $\pm$ 0.35                   |
|  | $\beta$ -PREDBAYES        | 94.90 $\pm$ 0.08                     | 96.73 $\pm$ 0.08                   |
|  | PVI                       | 95.56 $\pm$ 0.18                     | 96.68 $\pm$ 0.07                   |
|  | FEDGVI $D_{AR}$           | 96.36 $\pm$ 0.09                     | 97.13 $\pm$ 0.13                   |
|  | FEDGVI $L_{GCE}$          | 97.06 $\pm$ 0.03                     | 98.04 $\pm$ 0.07                   |
|  | FEDGVI $D_{AR} + L_{GCE}$ | <b>97.50 <math>\pm</math> 0.07</b>   | <b>98.13 <math>\pm</math> 0.08</b> |
|  | VI (1 CLIENT)             | (96.96 $\pm$ 0.17)                   |                                    |
|  | GVI (1 CLIENT)            | <b>(98.13 <math>\pm</math> 0.07)</b> |                                    |

We create label contamination by adding noise to the train-

![](_page_7_Figure_2.jpeg)

Figure 6: An ablation study on the hyperparameters of FEDGVI with L (δ) GCE and <sup>D</sup> (α) AR. We plot the maximum results achieved as percentage errors on uncontaminated test data after training 5 clients on 10% contaminated data.

Table 2: Classification accuracy (highest in bold) on uncontaminated test data after training on different amounts of contaminated FASHIONMNIST data. For FEDGVI we have fixed α = 2.5 for the α−Renyi divergence. Each Method ´ has data split homogeneously across 3 Clients. We report the best performance during all server iterations.

|   |    | M   | ODEL |    |      |   |      |    |     |      | C   |     |      |     |     |      |     |     |
|---|----|-----|------|----|------|---|------|----|-----|------|-----|-----|------|-----|-----|------|-----|-----|
|   |    |     |      |    |      |   |      | 0% |     |      | 10% |     |      | 20% |     |      | 40% |     |
|   |    | F   | ED A | VG |      |   | 85.7 | ±  | 0.5 | 79.0 | ±   | 1.9 | 71.2 | ±   | 1.5 | 49.0 | ±   | 6.5 |
|   |    | F   | ED   | PA |      |   | 88.1 | ±  | 0.3 | 87.4 | ±   | 0.2 | 86.5 | ±   | 0.2 | 85.4 | ±   | 0.5 |
|   | β  | –P  | RED  | B  | AYES |   | 87.6 | ±  | 0.1 | 87.2 | ±   | 0.1 | 86.8 | ±   | 0.1 | 85.8 | ±   | 0.1 |
|   |    |     | PVI  |    |      |   | 86.2 | ±  | 0.2 | 85.1 | ±   | 0.1 | 84.4 | ±   | 0.1 | 82.8 | ±   | 0.1 |
| F | ED | GVI | δ    | =  | 0    | 0 | 87.1 | ±  | 0.1 | 86.2 | ±   | 0.2 | 85.6 | ±   | 0.1 | 83.8 | ±   | 0.1 |
| F | ED | GVI | δ    | =  | 0    | 4 | 88.7 | ±  | 0.2 | 88.6 | ±   | 0.1 | 87.0 | ±   | 0.4 | 78.1 | ±   | 0.4 |
| F | ED | GVI | δ    | =  | 0    | 5 | 89.0 | ±  | 0.2 | 88.6 | ±   | 0.2 | 88.4 | ±   | 0.2 | 85.1 | ±   | 0.7 |
| F | ED | GVI | δ    | =  | 0    | 8 | 88.6 | ±  | 0.0 | 88.4 | ±   | 0.1 | 88.0 | ±   | 0.0 | 87.2 | ±   | 0.1 |
| F | ED | GVI | δ    | =  | 1    | 0 | 88.1 | ±  | 0.1 | 87.8 | ±   | 0.1 | 87.5 | ±   | 0.2 | 86.0 | ±   | 0.3 |

ing set while leaving the test set unchanged and evaluate performance in this. For MNIST, we add 10% of class dependent label noise, see Figure [7](#page-8-0) and Table [1.](#page-7-2) We further carry out an ablation study on the hyperparameter selection in FEDGVI with the Alpha–Renyi divergence and the gen- ´ eralised cross entropy loss, see Figure [6.](#page-7-0) This demonstrates that FEDGVI performs well under a variety of different loss and divergence parameters. Note that α = 1 recovers the KL divergence, α = 0 the reverse KL divergence, i.e. D (0) AR(<sup>q</sup> : <sup>π</sup>) = <sup>D</sup>RKL(<sup>q</sup> : <sup>π</sup>) = <sup>D</sup>KL(<sup>π</sup> : <sup>q</sup>), and that δ = 0 recovers the negative log–likelihood.

For FASHIONMNIST, in Table [2,](#page-7-3) we vary the amount of random label contamination, showcasing performance drops under different amounts of misspecification. We use an MLP, for FEDGVI and PVI with 1 hidden layer of 200

![](_page_8_Figure_2.jpeg)

Figure 7: Accuracy (% Error) and Negative Log Likelihood (NLL) results when running fully connected BNNs, with a Mean–Field Gaussian distribution, on the MNIST data set with FEDGVI. The training data set is contaminated by 10% random label flipping, fixed across all repetitions. We average over five runs with random, homogeneous client splits.

neurons; for FEDAVG, FEDPA, and β–PREDBAYES, two hidden layers with 100 neurons in each. Data is distributed homogeneously across clients, using 5 different, randomly chosen seeds. We demonstrate that under model misspecification, FEDGVI significantly outperforms competing FL methods. Furthermore, FEDGVI incurs no additional computational complexity when compared to PVI. This is due to the KL and Alpha-Renyi divergences having closed form ´ solutions between Multivariate Gaussians with complexity of O(1) in each other, and as we require O(1) additional, constant operations to get the GCE from the NLL.

We provide further experiments in Appendix [D](#page-34-0) on the runtime of FEDGVI against PVI, learning rate selection, stability of posteriors under small perturbations in the robust loss parameters, and showing that using a single hidden layer NN for the competing methods would either negatively, or not significantly, affect their performance.

# 6. Conclusions and Future Work

We have introduced FEDGVI, a novel probabilistic approach to federated learning that is provably robust to model misspecification, and allows for faster, conjugate client updates. The theoretical analysis of FEDGVI demonstrates it's appealing properties; we easily recover existing methods as restricted cases, and characterise the convergence behaviour

at fixed points of FEDGVI as solving a global GVI optimisation problem, extending existing theory. Our result on provable robustness to outliers through FEDGVI allows for closed form, conjugate posteriors that are computationally efficient, and robust to model misspecification. In deriving this, we have also shown that the cavity distribution is necessary as predictions would otherwise be overly confident and biased. The robustness of FEDGVI was further demonstrated empirically on multiple synthetic and real–world data sets, showing outperformance of existing FL methods across model architectures and misspecification levels.

An interesting future direction is to extend FedGVI within personalised FL settings [\(Kotelevskii et al.,](#page-11-8) [2022\)](#page-11-8) and hierarchical Bayesian FL through latent variables [\(Kim &](#page-11-9) [Hospedales,](#page-11-9) [2023\)](#page-11-9) as well as through the use of a structured posterior approximation [\(Hassan et al.,](#page-10-13) [2024\)](#page-10-13), in order to incorporate client level variations. Incorporating the hierarchical model structures and additional inductive biases from such settings, while maintaining conjugacy and favourable computational complexity, remain as open challenges. In future work, we further aim to address the robust Bayesian nonparametric setting of FL through FEDGVI, as well as investigate other types of robustness, including to adversarial and Byzantine attacks, by for instance using a robust aggregator in Equation [\(6\)](#page-3-4), and addressing the open problem of provable robustness to prior misspecification in GVI.

## Acknowledgements

OH, TM and TD acknowledge support from a UKRI Turing AI Acceleration Fellowship [EP/V02678X/1] and a Turing Impact Award from the Alan Turing Institute. For the purpose of open access, the authors have applied a Creative Commons Attribution (CC-BY) license to any Author Accepted Manuscript version arising from this submission. The authors acknowledge the University of Warwick Research Technology Platform for assistance in the research described in this paper.

## Impact Statement

This paper presents work on robust federated learning, a framework that aims to not only advance the field of machine learning, but also to develop methods that ensure the privacy of data sources, whilst aiming to achieve optimal performance even under contamination of the data. This approach, however, may discard low probability, tail events that could represent minority groups. Hence, the trade off between robustness and inclusivity is a fundamental ethical challenge for decision makers.

## References


[1] Achituve, I., Shamsian, A., Navon, A., Chechik, G., and Fetaya, E. Personalized federated learning with Gaussian processes. In Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), *Advances in Neural Information Processing Systems*, 2021.

[2] Ahn, S., Shahbaba, B., and Welling, M. Distributed stochastic gradient MCMC. In Xing, E. P. and Jebara, T. (eds.), *Proceedings of the 31st International Conference on Machine Learning*, volume 32 of *Proceedings of Machine Learning Research*, pp. 1044–1052, Beijing, China, 2014. PMLR.

[3] Al-Shedivat, M., Gillenwater, J., Xing, E., and Rostamizadeh, A. Federated learning via posterior averaging: A new perspective and practical algorithms. In *International Conference on Learning Representations*, 2021.

[4] Ali, S. M. and Silvey, S. D. A general class of coefficients of divergence of one distribution from another. *Journal of the Royal Statistical Society. Series B (Methodological)*, 28(1):131–142, 1966.

[5] Allouah, Y., Farhadkhani, S., Guerraoui, R., Gupta, N., Pinot, R., Rizk, G., and Voitovych, S. Byzantine-robust federated learning: Impact of client subsampling and local updates. In Salakhutdinov, R., Kolter, Z., Heller, K., Weller, A., Oliver, N., Scarlett, J., and Berkenkamp, F. (eds.), *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning*, pp. 1078–1114. PMLR, 21–27 Jul 2024.

[6] Alquier, P. Non-exponentially weighted aggregation: Regret bounds for unbounded loss functions. In Meila, M. and Zhang, T. (eds.), *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pp. 207–218. PMLR, 18–24 Jul 2021.

[7] Alquier, P., Ridgway, J., and Chopin, N. On the properties of variational approximations of gibbs posteriors. *Journal of Machine Learning Research*, 17(236):1–41, 2016.

[8] Altamirano, M., Briol, F.-X., and Knoblauch, J. Robust and scalable Bayesian online changepoint detection. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pp. 642–663. PMLR, 23–29 Jul 2023.

[9] Altamirano, M., Briol, F.-X., and Knoblauch, J. Robust and conjugate Gaussian process regression. In Salakhutdinov, R., Kolter, Z., Heller, K., Weller, A., Oliver, N., Scarlett, J., and Berkenkamp, F. (eds.), *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pp. 1155–1185. PMLR, 21–27 Jul 2024.

[10] Amari, S.-i. *Information Geometry and Its Applications*. Springer, Tokyo, Japan, 2016. ISBN 9784431559771.

[11] Ashman, M., Bui, T. D., Nguyen, C. V., Markou, S., Weller, A., Swaroop, S., and Turner, R. E. Partitioned variational inference: A framework for probabilistic federated learning. *arXiv preprint arXiv:2202.12275*, 2022.

[12] Bao, W., Wu, J., and He, J. BOBA: Byzantine-robust federated learning with label skewness. In Dasgupta, S., Mandt, S., and Li, Y. (eds.), *Proceedings of The 27th International Conference on Artificial Intelligence and Statistics*, volume 238 of *Proceedings of Machine Learning Research*, pp. 892–900. PMLR, 02–04 May 2024.

[13] Berger, J. O. *Statistical Decision Theory and Bayesian Analysis*. Springer–Verlag, New York, 1985. ISBN 9781475742862.

[14] Berk, R. H. Limiting behavior of posterior distributions when the model is incorrect. *The Annals of Mathematical Statistics*, 37(1):51 – 58, 1966.

[15] Bernardo, J. M. and Smith, A. F. M. *Bayesian theory*. Wiley Series in Probability and Statistics, Chichester, England, 2000. ISBN 9780470316870.

[16] Bissiri, P. G., Holmes, C., and Walker, S. G. A general framework for updating belief distributions. *Journal of the Royal Statistical Society. Series B (Statistical Methodology)*, 78(5):1103–1130, 2016. Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518):859–877, 2017. Bui, T. D., Nguyen, C. V., Swaroop, S., and Turner, R. E. Partitioned variational inference: A unified framework encompassing federated and continual learning. *arXiv preprint arXiv:1811.11206*, 2018. Carvalho, L. M., Villela, D. A. M., Coelho, F. C., and Bastos,

[17] L. S. Bayesian inference for the weights in logarithmic pooling. *Bayesian Analysis*, 18(1):223 – 251, 2023. Chan, R. S., Pollock, M., Johansen, A. M., and Roberts,

[18] G. O. Divide-and-conquer fusion. *Journal of Machine Learning Research*, 24(193):1–82, 2023. Chen, W.-N., Choquette-Choo, C. A., Kairouz, P., and Suresh, A. T. The fundamental price of secure aggregation in differentially private federated learning. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Proceedings of Machine Learning Research*, pp. 3056–3089. PMLR, 17–23 Jul 2022. Cichocki, A. and Amari, S.-i. Families of alpha- beta- and gamma- divergences: Flexible and robust measures of similarities. *Entropy*, 12(6):1532–1568, 2010. Corinzia, L., Beuret, A., and Buhmann, J. M. Variational federated multi-task learning. *arXiv preprint arXiv:1906.06268*, 2021. Demidovich, Y., Ostroukhov, P., Malinovsky, G., Horvath, ´ S., Taka´c, M., Richt ˇ arik, P., and Gorbunov, E. Meth- ´ ods with local steps and random reshuffling for generally smooth non-convex federated optimization. In *The Thirteenth International Conference on Learning Representations*, 2025. Diaconis, P. and Freedman, D. On the consistency of Bayes estimates. *The Annals of Statistics*, 14(1):1 – 26, 1986. Fraboni, Y., Vidal, R., Kameni, L., and Lorenzi, M. A general theory for federated optimization with asynchronous and heterogeneous clients updates. *Journal of Machine Learning Research*, 24(110):1–43, 2023. Genest, C. A characterization theorem for externally Bayesian groups. *The Annals of Statistics*, 12(3):1100– 1105, 1984. Genest, C., McConway, K. J., and Schervish, M. J. Characterization of externally Bayesian pooling operators. *The Annals of Statistics*, 14(2):487 – 501, 1986. Ghosh, A. and Basu, A. Robust Bayes estimation using the density power divergence. *Annals of the Institute of Statistical Mathematics*, 68(2):413–437, 2016a. Ghosh, A. and Basu, A. Robust estimation in generalized linear models: the density power divergence approach. *TEST*, 25(2):269–290, 2016b. Grunwald, P. The safe Bayesian. In Bshouty, N. H., Stoltz, ¨ G., Vayatis, N., and Zeugmann, T. (eds.), *Algorithmic Learning Theory*, pp. 169–183, Berlin, Heidelberg, 2012. Springer Berlin Heidelberg. Guo, H., Greengard, P., Wang, H., Gelman, A., Kim, Y., and Xing, E. Federated learning as variational inference: A scalable expectation propagation approach. In *The Eleventh International Conference on Learning Representations*, 2023. Hamer, J., Mohri, M., and Suresh, A. T. FedBoost: A communication-efficient algorithm for federated learning. In III, H. D. and Singh, A. (eds.), *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pp. 3973–3983. PMLR, 2020. Hasan, M., Zhang, G., Guo, K., Chen, X., and Poupart, P. Calibrated one round federated learning with Bayesian inference in the predictive space. *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(11):12313– 12321, 2024. Hasenclever, L., Webb, S., Lienart, T., Vollmer, S., Lakshminarayanan, B., Blundell, C., and Teh, Y. W. Distributed Bayesian learning with stochastic natural gradient expectation propagation and the posterior server. *Journal of Machine Learning Research*, 18(1):3744–3780, 2017. Hassan, C., Salomone, R., and Mengersen, K. Federated variational inference methods for structured latent variable models. *arXiv preprint arXiv:2302.03314*, 2023. Hassan, C., Sutton, M., Mira, A., and Mengersen, K. Scalable vertical federated learning via data augmentation and amortized inference. *arXiv preprint arXiv:2405.04043*, 2024. Heikkila, M., Ashman, M., Swaroop, S., Turner, R., and ¨ Honkela, A. Differentially private partitioned variational inference. *Transactions on machine learning research*, 2023(4), 2023. Hooker, G. and Vidyashankar, A. N. Bayesian model robustness via disparities. *TEST*, 23(3):556–584, 2014.

[19] Huber, P. J. Robust estimation of a location parameter. *Annals of Mathematical Statistics*, 35:73–101, 1964. Hung, H., Jou, Z.-Y., and Huang, S.-Y. Robust mislabel logistic regression without modeling mislabel probabilities. *Biometrics*, 74(1):145–154, 2018. Hyvarinen, A. Estimation of non-normalized statistical ¨ models by score matching. *Journal of Machine Learning Research*, 6(24):695–709, 2005. Jewson, J., Smith, J. Q., and Holmes, C. Principles of Bayesian inference using general divergence criteria. *Entropy*, 20(6):442, 2018. Jonker, M. A., Pazira, H., and Coolen, A. C. Bayesian federated inference for estimating statistical models based on non-shared multicenter data sets. *Statistics in Medicine*, pp. 1–18, 2024. Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., Bonawitz, K., Charles, Z., Cormode, G., Cummings, R., D'Oliveira, R. G. L., Eichner, H., Rouayheb, S. E., Evans, D., Gardner, J., Garrett, Z., Gascon, A., Ghazi, B., Gibbons, P. B., Gruteser, M., Har- ´ chaoui, Z., He, C., He, L., Huo, Z., Hutchinson, B., Hsu, J., Jaggi, M., Javidi, T., Joshi, G., Khodak, M., Konecny,´ J., Korolova, A., Koushanfar, F., Koyejo, S., Lepoint, T., Liu, Y., Mittal, P., Mohri, M., Nock, R., Ozg ¨ ur, A., Pagh, ¨ R., Qi, H., Ramage, D., Raskar, R., Raykova, M., Song, D., Song, W., Stich, S. U., Sun, Z., Suresh, A. T., Tramer, ` F., Vepakomma, P., Wang, J., Xiong, L., Xu, Z., Yang, Q., Yu, F. X., Yu, H., and Zhao, S. Advances and open problems in federated learning. *Foundations and Trends® in Machine Learning*, 14(1–2):1–210, 2021. Kallioinen, N., Paananen, T., Burkner, P.-C., and Vehtari, A. ¨ Detecting and diagnosing prior and likelihood sensitivity with power-scaling. *Statistics and Computing*, 34(1):57, 2024. Kassab, R. and Simeone, O. Federated generalized Bayesian learning via distributed Stein variational gradient descent. *IEEE Transactions on Signal Processing*, 70:2180–2192, 2022. Katsevich, A. and Rigollet, P. On the approximation accuracy of Gaussian variational inference. *The Annals of Statistics*, 52(4):1384 – 1409, 2024. Kim, M. and Hospedales, T. FedHB: Hierarchical Bayesian federated learning. *arXiv preprint arXiv:2305.04979*, 2023. Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In *3rd International Conference on Learning Representations*, 2015. Knoblauch, J., Jewson, J. E., and Damoulas, T. Doubly robust Bayesian inference for non-stationary streaming data with β-divergences. In *Advances in Neural Information Processing Systems*, volume 31, pp. 64–75. Curran Associates, Inc., 2018. Knoblauch, J., Jewson, J., and Damoulas, T. An optimization-centric view on Bayes' rule: Reviewing and generalizing variational inference. *Journal of Machine Learning Research*, 23(132):1–109, 2022. Kotelevskii, N. Y., Vono, M., Durmus, A., and Moulines,
  - E. FedPop: A Bayesian approach for personalised federated learning. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), *Advances in Neural Information Processing Systems*, 2022. Kullback, S. and Leibler, R. A. On Information and Sufficiency. *The Annals of Mathematical Statistics*, 22(1):79 – 86, 1951. LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradientbased learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, 1998. Li, H., Acharya, K., and Richtarik, P. The power of extrapo- ´ lation in federated learning. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024. Malinovsky, G., Kovalev, D., Gasanov, E., Condat, L., and Richtarik, P. From local SGD to local fixed-point methods for federated learning. In III, H. D. and Singh, A. (eds.), *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pp. 6692–6701. PMLR, 2020. Matsubara, T., Knoblauch, J., Briol, F.-X., and Oates, C. J. Robust generalised Bayesian inference for intractable likelihoods. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 84(3):997–1022, 04 2022. McMahan, B., Moore, E., Ramage, D., Hampson, S., and Arcas, B. A. y. Communication-efficient learning of deep networks from decentralized data. In Singh, A. and Zhu, J. (eds.), *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics*, volume 54 of *Proceedings of Machine Learning Research*, pp. 1273– 1282. PMLR, 2017. Mekkaoui, K. e., Mesquita, D., Blomstedt, P., and Kaski,
    - S. Federated stochastic gradient Langevin dynamics. In de Campos, C. and Maathuis, M. H. (eds.), *Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence*, volume 161 of *Proceedings of Machine Learning Research*, pp. 1703–1712. PMLR, 2021.

[20] Mesquita, D., Blomstedt, P., and Kaski, S. Embarrassingly parallel MCMC using deep invertible transformations. In Adams, R. P. and Gogate, V. (eds.), *Proceedings of The 35th Uncertainty in Artificial Intelligence Conference*, volume 115 of *Proceedings of Machine Learning Research*, pp. 1244–1252. PMLR, 2020. Miller, J. W. Asymptotic normality, concentration, and coverage of generalized posteriors. *Journal of Machine Learning Research*, 22(168):1–53, 2021. Minka, T. P. Expectation propagation for approximate Bayesian inference. In *Proceedings of the Seventeenth Conference on Uncertainty in Artificial Intelligence*, pp. 362–369, San Francisco, CA, USA, 2001. Nielsen, F. An elementary introduction to information geometry. *Entropy*, 22(10):1100, 2020. Nielsen, F. A simple approximation method for the fisher–rao distance between multivariate normal distributions. *Entropy*, 25(4), 2023. Opper, M. and Winther, O. Expectation consistent approximate inference. *Journal of Machine Learning Research*, 6(73):2177–2204, 2005. Pardo Llorente, L. *Statistical inference based on divergence measures*. Chapman & Hall/CRC, 2006. ISBN 9781584886006. Pinski, F. J., Simpson, G., Stuart, A. M., and Weber, H. Kullback-leibler approximation for probability measures on infinite dimensional spaces. *SIAM Journal on Mathematical Analysis*, 47(6):4091–4122, 2015. Reddi, S., Charles, Z. B., Zaheer, M., Garrett, Z., Rush, K., Konecnˇ y, J., Kumar, S., and McMahan, B. Adaptive ´ federated optimization. In *International Conference on Learning Representations*, 2021. Scott, S. L., Blocker, A. W., Bonassi, F. V., Chipman, H. A., George, E. I., and McCulloch, R. E. Bayes and big data: The consensus monte carlo algorithm. *International Journal of Management Science and Engineering Management*, 11:78–88, 2016. Swaroop, S., Khan, M. E., and Doshi-Velez, F. Connecting federated ADMM to Bayes. In *The Thirteenth International Conference on Learning Representations*, 2025. Tenison, I., Sreeramadas, S. A., Mugunthan, V., Oyallon, E., Rish, I., and Belilovsky, E. Gradient masked averaging for federated learning. *Transactions on Machine Learning Research*, 2023. Tresp, V. A Bayesian committee machine. *Neural computation*, 12:2719–41, 2000. Tziotis, I., Shen, Z., Pedarsani, R., Hassani, H., and Mokhtari, A. Straggler-resilient personalized federated learning. *Transactions on Machine Learning Research*, 2023. Vedadi, E., Dillon, J. V., Mansfield, P. A., Singhal, K., Afkanpour, A., and Morningstar, W. R. Federated variational inference: Towards improved personalization and generalization. *Transactions on Machine Learning Research*, 2024. Vehtari, A., Gelman, A., Sivula, T., Jylanki, P., Tran, D., ¨ Sahai, S., Blomstedt, P., Cunningham, J. P., Schiminovich, D., and Robert, C. P. Expectation propagation as a way of life: A framework for Bayesian inference on partitioned data. *Journal of Machine Learning Research*, 21(1), 2020. Walker, S. G. Bayesian inference with misspecified models. *Journal of Statistical Planning and Inference*, 143(10): 1621–1633, 2013. Xiao, H., Rasul, K., and Vollgraf, R. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. *arXiv preprint arXiv:1708.07747*, 2017. Yonekura, S. and Sugasawa, S. Adaptation of the tuning parameter in general Bayesian inference with robust divergence. *Statistics and Computing*, 33(2):39, 2023. Yurochkin, M., Agarwal, M., Ghosh, S., Greenewald, K., Hoang, N., and Khazaeni, Y. Bayesian nonparametric federated learning of neural networks. In Chaudhuri, K. and Salakhutdinov, R. (eds.), *Proceedings of the 36th International Conference on Machine Learning*, volume 97 of *Proceedings of Machine Learning Research*, pp. 7252– 7261. PMLR, 09–15 Jun 2019. Zellner, A. Optimal information processing and Bayes's theorem. *The American Statistician*, 42(4):278–280, 1988. Zhang, X., Li, Y., Li, W., Guo, K., and Shao, Y. Personalized federated learning via variational Bayesian inference. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Proceedings of Machine Learning Research*, pp. 26293–26310. PMLR, 17–23 Jul 2022. Zhang, Z. and Sabuncu, M. Generalized cross entropy loss for training deep neural networks with noisy labels. In Bengio, S., Wallach, H., Larochelle, H., Grauman, K., Cesa-Bianchi, N., and Garnett, R. (eds.), *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018. Zhao, Z., Luo, M., and Ding, W. Deep leakage from model in federated learning. In *Conference on Parsimony and Learning*, volume 234 of *Proceedings of Machine Learning Research*, pp. 324–340. PMLR, 2023.

[21] Zhu, L., Liu, Z., and Han, S. Deep leakage from gradients. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alche-Buc, F., Fox, E., and Garnett, R. (eds.), ´ *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019.
# Supplementary Material for: Federated Generalised Variational Inference: A Robust Probabilistic Federated Learning Framework

The appendix is structured as follows: Appendix [A](#page-14-0) summarises the notation used throughout the paper and in the proofs. In Appendix [B](#page-14-1) we present complete proofs of all theorems, propositions and lemmas given in the main paper. Appendix [C](#page-33-0) clarifies the requirements of Definition [4.11,](#page-5-3) the GBI learning rate, and places FEDGVI in the broader GVI literature. Lastly, Appendix [D](#page-34-0) gives additional details about the implementation of FEDGVI and additional experiments.

# A. Notation

In this section, we give definitions of the symbols used throughout the paper and the appendix.

P<sup>0</sup> The abstract and unknown probability measure, also called data generating process, acting on some abstract measurable space (Ω, F) which gives rise to the data

{x<sup>i</sup> , yi} n <sup>i</sup>=1 Entire data set of all clients, also written as {<sup>x</sup> n 1 , y<sup>n</sup> <sup>1</sup> }, for <sup>x</sup><sup>i</sup> ∈ <sup>Ξ</sup> and <sup>y</sup><sup>i</sup> |x<sup>i</sup> ∈ Υ

{xm, ym}<sup>M</sup> <sup>m</sup>=1 The entire set of data points split across <sup>M</sup> clients labelled <sup>m</sup> ∈ [M] := {1, <sup>2</sup>, ..., M}

Ξ The data space, which is assumed to have Polish topology

Υ The output space, which can be categorical such as in classification where Υ = [C], or real valued as in regression Υ = R <sup>C</sup> , C ∈ <sup>N</sup>

Θ In the parametric setting this is the parameter space θ ∈ Θ, assumed to admit Polish topology

P(Θ) The space of probability measures over the measurable space (Θ, T ). We refer to distributions in this space, where we mean distribution functions given rise to by measures in this space. Note that these need not be continuous, and could only be defined almost everywhere in θ.

Q A variational family of distributions such that Q ⊂ P(Θ) and, in terms of distributions, Q = {q(θ|κ) ∈ P(Θ) : κ ∈ K}, where K is a set of variational parameters

π(θ) The prior distribution, given rise to by the prior measure Π on (Θ, T )

L (t) <sup>m</sup> (ym; θ, xm) The local loss of client m, at iteration t ∈ [T], on the local data set {xm, ym}, not necessarily the same across clients nor iterations, and associated with the parameters θ ∈ Θ

ℓ (t) <sup>m</sup> (θ) Local loss approximation of Lm(ym; θ, xm) and the impact of the data of client m on the posterior at the server

∆ (t) <sup>m</sup> (θ) Local update, Equation [\(5\)](#page-3-3), that represents the change in the approximate posteriors, and the de facto change in the local loss approximation. It has associated damping parameter τ .

ℓ (t) <sup>s</sup> (θ) Global loss approximation of all clients aggregated at the server

q (t) <sup>m</sup> (θ) Local posterior computed through Equation [\(4\)](#page-3-2)

q (t) <sup>s</sup> (θ) Global approximate posterior after server–side optimisation step, Equation [\(7\)](#page-3-0)

P(L, D, Q) The Rule of Three [\(Knoblauch et al.,](#page-11-6) [2022\)](#page-11-6) that defines a global GVI objective

D Any statistical divergence D : P(Θ) × P(Θ) → <sup>R</sup>≥<sup>0</sup> (for a detailed definition see [Nielsen,](#page-12-19) [2020\)](#page-12-19); D<sup>s</sup> denotes the divergence at the server.

<sup>E</sup>q(θ) The expectation with respect to <sup>q</sup>(θ)

# B. Proofs of Theorems, Propositions, and Lemmas

Here, we provide the full proofs of the theorems stated in the paper. Throughout, we assume that all the losses, distributions and approximate losses, are measurable with respect to some dominating measure µ(dθ). This can be the Lebesgue measure in finite dimensional spaces, or more generally the Haar measure. For infinite dimensional measure spaces, which are of interest in the study of Bayesian inverse problems and nonparametrics, we could assume µ(dθ) to be a Gaussian measure as in [Pinski et al.](#page-12-20) [\(2015\)](#page-12-20).

# B.1. Equivalence Between The KL Divergence and Weighted KL Divergence

First, we present a well known auxiliary lemma that will be used throughout the proofs. It states that the weighted KL divergence is equivalent to using a tempered or weighted likelihood in the optimisation procedure, and hence lead to equivalent inference problems [\(Knoblauch et al.,](#page-11-6) [2022;](#page-11-6) [Bissiri et al.,](#page-10-7) [2016\)](#page-10-7). So without loss of generality, we can push the weighting term of the KL divergence inside the loss, by defining the loss to be L = w · L, which does not change the optimisation procedure. We show this result for f–divergences, which we define as in [Ali & Silvey](#page-9-14) [\(1966\)](#page-9-14) and [Amari](#page-9-13) [\(2016\)](#page-9-13).

Lemma B.1. *For* w > 0 *the posteriors computed by the weighted* f*–divergence,* D = <sup>w</sup> D<sup>f</sup> *and loss* L*, and the posterior through the* f*–divergence* D = D<sup>f</sup> *and weighted loss* w · L *are equivalent, i.e.,*

$$P(L, \frac{1}{w}D_f, \mathcal{Q}) = P(w \cdot L, D_f, \mathcal{Q})$$

Proof

$$\begin{aligned}
P(L, \frac{1}{w} D_f, \mathcal{Q}) &= \arg \min_{q \in \mathcal{Q}} \left\{ \mathbb{E}_{q(\boldsymbol{\theta})} [L(\mathbf{y}; \boldsymbol{\theta}, \mathbf{x})] + \frac{1}{w} D_f(q : \pi) \right\} \\
&= \arg \min_{q \in \mathcal{Q}} \left\{ \mathbb{E}_{q(\boldsymbol{\theta})} [L(\mathbf{y}; \boldsymbol{\theta}, \mathbf{x})] + \frac{1}{w} \mathbb{E}_{q(\boldsymbol{\theta})} \left[ f\left(\frac{q(\boldsymbol{\theta})}{\pi(\boldsymbol{\theta})}\right) \right] \right\} \\
&= \arg \min_{q \in \mathcal{Q}} \left\{ \frac{1}{w} \mathbb{E}_{q(\boldsymbol{\theta})} \left[ w \cdot L(\mathbf{y}; \boldsymbol{\theta}, \mathbf{x}) + f\left(\frac{q(\boldsymbol{\theta})}{\pi(\boldsymbol{\theta})}\right) \right] \right\} \\
&= \arg \min_{q \in \mathcal{Q}} \left\{ \mathbb{E}_{q(\boldsymbol{\theta})} \left[ w \cdot L(\mathbf{y}; \boldsymbol{\theta}, \mathbf{x}) + f\left(\frac{q(\boldsymbol{\theta})}{\pi(\boldsymbol{\theta})}\right) \right] \right\} \\
&= \arg \min_{q \in \mathcal{Q}} \left\{ \mathbb{E}_{q(\boldsymbol{\theta})} [w \cdot L(\mathbf{y}; \boldsymbol{\theta}, \mathbf{x})] + D_f(q : \pi) \right\} := P(w \cdot L, D_f, \mathcal{Q})
\end{aligned}$$

Therefore, when referring to the loss in the following we mean it to be the weighted loss so that we can utilise the weighted KL divergence. This easily recovers the KL–divergence for f : u 7→ − log u.

## B.2. Proposition [4.3:](#page-4-4) A Logarithmic Opinion Pool through Damping

Proof Consider the server update at some iteration t, where we gather the client updates. Under the KL divergence, we then solve the server optimisation procedure as:

$$q_s^{(t)}(\boldsymbol{\theta}) = \arg \min_{q \in \mathcal{Q}} \left\{ \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \ell_s^{(t)}(\boldsymbol{\theta}) \right] + KL(q : \pi) \right\} = \arg \min_{q \in \mathcal{Q}} \left\{ \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \log \frac{q(\boldsymbol{\theta})}{\pi(\boldsymbol{\theta}) \exp\{-\ell_s^{(t)}(\boldsymbol{\theta})\}} \right] \right\}$$

we know that this is minimised at:

$$\begin{aligned} q_s^{(t)}(\boldsymbol{\theta}) \propto \pi(\boldsymbol{\theta}) \exp\{-\ell_s^{(t)}(\boldsymbol{\theta})\} &= \pi(\boldsymbol{\theta}) \exp\left\{-\ell_s^{(t-1)}(\boldsymbol{\theta}) - \sum_{m=1}^M \Delta_m^{(t)}(\boldsymbol{\theta})\right\} \\ &\propto \underbrace{\pi(\boldsymbol{\theta}) \exp\{-\ell_s^{(t-1)}(\boldsymbol{\theta})\}}_{\propto q_s^{(t-1)}(\boldsymbol{\theta})} \exp\left\{-\sum_{m=1}^M -\tau_m \log \frac{q_m^{(t)}(\boldsymbol{\theta})}{q_s^{(t-1)}(\boldsymbol{\theta})}\right\} \propto q_s^{(t-1)}(\boldsymbol{\theta}) \prod_{m=1}^M \left( \frac{q_m^{(t)}(\boldsymbol{\theta})}{q_s^{(t-1)}(\boldsymbol{\theta})} \right)^{\tau_m} \\ &= \frac{q_s^{(t-1)}(\boldsymbol{\theta}) \prod_{m=1}^M (q_m^{(t)}(\boldsymbol{\theta}))^{\tau_m}}{(q_s^{(t-1)}(\boldsymbol{\theta}))^{\sum_{m=1}^M \tau_m}} \end{aligned}$$

By assumption we have that P<sup>M</sup> <sup>m</sup>=1 <sup>τ</sup><sup>m</sup> = 1, therefore (<sup>q</sup> (t−1) <sup>s</sup> (θ)) P<sup>M</sup> <sup>m</sup>=1 <sup>τ</sup><sup>m</sup> = q (t−1) <sup>s</sup> (θ) and:

$$q_s^{(t)}(\boldsymbol{\theta}) \propto \prod_{m=1}^M (q_m^{(t)}(\boldsymbol{\theta}))^{\tau_m}$$

$$q_s^{(t)}(\boldsymbol{\theta}) = \frac{\prod_{m=1}^M (q_m^{(t)}(\boldsymbol{\theta}))^{\tau_m}}{\int_{\Theta} \prod_{m=1}^M (q_m^{(t)}(\boldsymbol{\theta}))^{\tau_m} \mu(d\boldsymbol{\theta})}, \quad \mu - a.e.$$

This forms an externally Bayesian logarithmic opinion pool [\(Genest,](#page-10-9) [1984;](#page-10-9) [Genest et al.,](#page-10-20) [1986\)](#page-10-20).

### B.3. Proof of Proposition [4.4](#page-4-7)

The proof of Proposition [4.4](#page-4-7) is adapted from that for Partitioned Variational Inference in [Ashman et al.](#page-9-0) [\(2022\)](#page-9-0). We show the proof of Proposition [4.4](#page-4-7) by comparing the derivatives with respect to the variational parameters of q(θ|κ) of the sum of local objectives with those of the global objective. This is motivated by the equivalence of a sum of local GVI objectives (from each client) with some added constants and the global GVI objective, demonstrated in Appendix [B.3.1.](#page-16-1) The main proof is in Appendix [B.3.2.](#page-17-0)

# B.3.1. RECOVERING A GLOBAL GVI OBJECTIVE FROM LOCAL OBJECTIVES

First, we provide an analogue of [Ashman et al.](#page-9-0) [\(2022,](#page-9-0) Property 2) which states that the sum of the local (client) FEDGVI objectives and some constant, which we find to be the negative log normalising constants of the cavity and the server distributions, equals the global GVI objective. We define the following:

$$\begin{aligned} q_s^{(t)}(\boldsymbol{\theta}) &= \frac{1}{Z_{q_s^{(t)}}} \pi(\boldsymbol{\theta}) \exp\{-\sum_{m=1}^M \ell_m^{(t)}(\boldsymbol{\theta})\} \\ q^{\setminus m}(\boldsymbol{\theta}) &= \frac{1}{Z_{q^{\setminus m}}} \pi(\boldsymbol{\theta}) \exp\{-\sum_{k \neq m} \ell_k^{(t)}(\boldsymbol{\theta})\} \propto \frac{q_s^{(t)}(\boldsymbol{\theta})}{\exp\{-\ell_m^{(t)}(\boldsymbol{\theta})\}} \\ \text{Obj}(m, q_s^{(t)}) &:= \mathbb{E}_{q(\boldsymbol{\theta})} [L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)] + \frac{1}{w} D_{KL}(q : q^{\setminus m}) \\ \text{Obj}(q_s^{(t)}) &:= \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) \right] + \frac{1}{w} D_{KL}(q : \pi) \end{aligned}$$

Then we can recover the global objective by summing over the local objectives and subtracting the log normalising constants of the cavity distributions and the current server posterior.

$$\begin{aligned} & \sum_{m=1}^M \text{Obj}(m, q_s^{(t)}) - \frac{1}{w} (\log Z_{q_s^{(t)}} + \sum_{m=1}^M \log Z_{q^{\setminus m}}) \\ &= \sum_{m=1}^M \left( \mathbb{E}_{q(\boldsymbol{\theta})} [L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)] + \frac{1}{w} D_{KL}(q : q^{\setminus m}) \right) - \frac{1}{w} (\log Z_{q_s^{(t)}} + \sum_{m=1}^M \log Z_{q^{\setminus m}}) \\ &= \sum_{m=1}^M \mathbb{E}_{q(\boldsymbol{\theta})} [L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)] + \sum_{m=1}^M \frac{1}{w} \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \log \frac{q(\boldsymbol{\theta})}{q^{\setminus m}(\boldsymbol{\theta})} \right] - \frac{1}{w} (\log Z_{q_s^{(t)}} + \sum_{m=1}^M \log Z_{q^{\setminus m}}) \\ &= \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) \right] + \frac{1}{w} \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \sum_{m=1}^M \log \frac{q(\boldsymbol{\theta})}{q^{\setminus m}(\boldsymbol{\theta})} \right] - \frac{1}{w} (\log Z_{q_s^{(t)}} + \sum_{m=1}^M \log Z_{q^{\setminus m}}) \\ &= \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) \right] + \frac{1}{w} \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \log \prod_{m=1}^M \frac{q(\boldsymbol{\theta}) \exp\{-\ell_m^{(t)}(\boldsymbol{\theta})\}}{q_s^{(t)}(\boldsymbol{\theta})} \right] - \frac{1}{w} (\log Z_{q_s^{(t)}} + \sum_{m=1}^M \log Z_{q^{\setminus m}}) \\ &= \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) \right] + \frac{1}{w} \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \log \frac{q(\boldsymbol{\theta}) \exp\{-\sum_{m=1}^M \ell_m^{(t)}(\boldsymbol{\theta})\}}{q_s^{(t)}(\boldsymbol{\theta})} \right] - \frac{1}{w} \log Z_{q_s^{(t)}} \\ &= \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) \right] + \frac{1}{w} \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \log \frac{q(\boldsymbol{\theta})}{\pi(\boldsymbol{\theta})/Z_{q_s^{(t)}}} \right] - \frac{1}{w} \log Z_{q_s^{(t)}} = \text{Obj}(q_s^{(t)}) \end{aligned}$$

Hence, by using the weighted KL divergence at the clients optimisation step, can we recover a global GVI objective by summing over the local objectives and adding some constants independent of the variational parameters of interest in the optimisation problem. We note that the added logarithms of the normalising constants are independent of κ, since these are fixed through the current posterior and cavity distribution and do not depend on the variational parameters.

# B.3.2. PROPOSITION [4.4:](#page-4-7) FIXED POINTS RECOVERS A GLOBAL FIXED POINT

We denote a fixed point of the algorithm as q ∗ s (θ|κ ∗ ) such that for all m ∈ [M] we have q ∗ s (θ|κ ∗ ) ∈ arg minq∈Q Obj(m, q<sup>∗</sup> s ), then we have the property that no update will change the posterior found. Recall:

Proposition [4.4](#page-4-7) *Let* D = 1 <sup>w</sup> DKL *at the clients, local loss* L<sup>m</sup> *and* Q := {q(θ|κ) : κ ∈ K} ⊂ P(Θ) *as a variational family. Assume that* FEDGVI *finds a fixed point* q ∗ s (θ|κ ∗ )*, such that for all clients we have that* q ∗ s (θ|κ ∗ ) ∈ arg minq∈Q Obj(m, q<sup>∗</sup> s )*. Then, it holds that* q ∗ s (θ|κ ∗ ) ∈ arg minq∈Q Obj(q ∗ s )*.*

Proof First we note that we consider only the KL divergence in this proof, which is equivalent to saying we modify the loss L to be multiplied by w > 0, which results in the equivalent formulation, as shown in [Knoblauch et al.](#page-11-6) [\(2022\)](#page-11-6) where P(L, <sup>1</sup> <sup>w</sup> <sup>D</sup>KL, <sup>Q</sup>) = <sup>P</sup>(<sup>w</sup> · L, KL, <sup>Q</sup>), see also Lemma [B.1.](#page-15-1)

Note that the condition ∀m ∈ [M] we have that q ∗ s (θ|κ ∗ ) ∈ arg minq∈Q Obj(m, q<sup>∗</sup> s ) is equivalent to requiring that ∆<sup>∗</sup> <sup>m</sup>(θ) = 0, since this means that the local loss approximations remain unchanged and hence <sup>ℓ</sup> ∗ s (θ) remains unchanged. This then implies that the posterior at the server will not change. This is the same as saying that the client optimisation step has found the global solution and hence q ∗ <sup>m</sup>(θ) and <sup>q</sup> ∗ s (θ) will be the same which implies that ∆<sup>∗</sup> <sup>m</sup>(θ) = 0.

In the following all integrals are assumed to be over the parameter space Θ, even when we don't make it explicit.

We can furthermore show that we can express the derivative of the local objective as a single integral under the weighted KL divergence.

$$\begin{aligned}\nabla_{\boldsymbol{\kappa}} \text{Obj}(m, q_s^*) &= \nabla_{\boldsymbol{\kappa}} \left\{ \mathbb{E}_{q(\boldsymbol{\theta})} [L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)] + D_{KL} \left( q : \frac{q_s^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)}{\exp\{-\ell_m^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)\} Z_{q_s^*}^*} \right) \right\} \\ &= \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \log \frac{1}{\exp\{-L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} + q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \left( \log \frac{q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \exp\{-\ell_m^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)\}}{q_s^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)} + \log Z_{q_s^*}^* \right) \mu(d\boldsymbol{\theta}) \\ &= \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \log \frac{q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \exp\{-\ell_m^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)\}}{q_s^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*) \exp\{-L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \mu(d\boldsymbol{\theta}) + \nabla_{\boldsymbol{\kappa}} \log Z_{q_s^*}^* \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \mu(d\boldsymbol{\theta}) \end{aligned}$$

Now we first show that the fixed point is an extremum of the global objective and then that it is a minimum. We do this by first differentiating the local objective with respect to the variational parameters κ and then that the sum of the local derivatives evaluated at κ = κ ∗ equal the derivative of the global objective.

$$\begin{aligned}\nabla_{\boldsymbol{\kappa}} \text{Obj}(m, q_s^*) &= \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \log \frac{q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \exp\{-\ell_m^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)\}}{q_s^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*) \exp\{-L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \mu(d\boldsymbol{\theta}) \\ &= \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) (L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) - \ell_m^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)) \mu(d\boldsymbol{\theta}) + \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \log \frac{q(\boldsymbol{\theta}|\boldsymbol{\kappa})}{q_s^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)} \mu(d\boldsymbol{\theta}) \\ &= \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) (L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) - \ell_m^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)) \mu(d\boldsymbol{\theta}) + \int (\nabla_{\boldsymbol{\kappa}} q(\boldsymbol{\theta}|\boldsymbol{\kappa})) \log \frac{q(\boldsymbol{\theta}|\boldsymbol{\kappa})}{q_s^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)} \mu(d\boldsymbol{\theta}) \\ &\quad + \int \overline{\nabla_{\boldsymbol{\kappa}} q(\boldsymbol{\theta}|\boldsymbol{\kappa})} \mu(d\boldsymbol{\theta}) \end{aligned}$$

where first line follows since we can compose the expectation and (weighted) KL divergence and the normalising constant of the cavity distribution is constant with respect to κ. The last line follows from the fact that <sup>d</sup> dx f(x) log f(x) = f ′ (x) log f(x) + f ′ (x) and that we can exchange the order of integration and differentiation. We further note that at convergence, where κ = κ ∗ , that log <sup>q</sup>(θ|κ) q ∗ (θ|κ<sup>∗</sup>) <sup>κ</sup>=κ<sup>∗</sup> = 0. Evaluating the expression above at <sup>κ</sup> <sup>=</sup> <sup>κ</sup> ∗ then yields:

$$\nabla_{\boldsymbol{\kappa}} \text{Obj}(m, q_s^*) \Big|_{\boldsymbol{\kappa}=\boldsymbol{\kappa}^*} = \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) (L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) - \ell_m^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)) \mu(d\boldsymbol{\theta}) \Big|_{\boldsymbol{\kappa}=\boldsymbol{\kappa}^*}$$

Summing over all these client objectives then yields the following expression:

$$\begin{aligned} \sum_{m=1}^M \nabla_{\boldsymbol{\kappa}} \text{Obj}(m, q_s^*) \Big|_{\boldsymbol{\kappa}=\boldsymbol{\kappa}^*} &= \sum_{m=1}^M \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) (L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) - \ell_m^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)) \mu(d\boldsymbol{\theta}) \Big|_{\boldsymbol{\kappa}=\boldsymbol{\kappa}^*} \\ &= \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \left( \sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) - \sum_{m=1}^M \ell_m^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*) \right) \mu(d\boldsymbol{\theta}) \Big|_{\boldsymbol{\kappa}=\boldsymbol{\kappa}^*} \\ &= \int (\nabla_{\boldsymbol{\kappa}} q(\boldsymbol{\theta}|\boldsymbol{\kappa})) \log \frac{q_s^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)}{\pi(\boldsymbol{\theta}) \exp\{\sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \mu(d\boldsymbol{\theta}) \Big|_{\boldsymbol{\kappa}=\boldsymbol{\kappa}^*} \\ &\quad + \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \log Z_{q^*} \mu(d\boldsymbol{\theta}) \end{aligned}$$

To compare this with a global fixed point we differentiate the global objective at q ∗ , not yet assumed to be a minimiser of the global objective, with respect to the variational parameters.

$$\begin{aligned}\nabla_{\boldsymbol{\kappa}} \text{Obj}(q_s^*) &= \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta} | \boldsymbol{\kappa}) \log \frac{q(\boldsymbol{\theta} | \boldsymbol{\kappa})}{\pi(\boldsymbol{\theta}) \exp\{\sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \mu(d\boldsymbol{\theta}) \\ &= \int (\nabla_{\boldsymbol{\kappa}} q(\boldsymbol{\theta} | \boldsymbol{\kappa})) \log \frac{q(\boldsymbol{\theta} | \boldsymbol{\kappa})}{\pi(\boldsymbol{\theta}) \exp\{\sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \mu(d\boldsymbol{\theta}) + \int \nabla_{\boldsymbol{\kappa}} q(\boldsymbol{\theta} | \boldsymbol{\kappa}) \mu(d\boldsymbol{\theta}) \end{aligned}$$

Then,

$$\nabla_{\boldsymbol{\kappa}} \text{Obj}(q_s^*) \Big|_{\boldsymbol{\kappa}=\boldsymbol{\kappa}^*} = \int (\nabla_{\boldsymbol{\kappa}} q(\boldsymbol{\theta}|\boldsymbol{\kappa})) \log \frac{q(\boldsymbol{\theta}|\boldsymbol{\kappa})}{\pi(\boldsymbol{\theta}) \exp\{\sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \mu(d\boldsymbol{\theta}) \Big|_{\boldsymbol{\kappa}=\boldsymbol{\kappa}^*} = \sum_{m=1}^M \nabla_{\boldsymbol{\kappa}} \text{Obj}(m, q_s^*) \Big|_{\boldsymbol{\kappa}=\boldsymbol{\kappa}^*}$$

And since q ∗ s (θ|κ ∗ ) is a fixed point of each client, we have that ∇κObj(m, q<sup>∗</sup> s ) <sup>κ</sup>=κ<sup>∗</sup> = 0. Therefore,

$$\sum_{m=1}^M \nabla_{\boldsymbol{\kappa}} \text{Obj}(m, q_s^*) \Big|_{\boldsymbol{\kappa}=\boldsymbol{\kappa}^*} = 0 \quad \implies \quad \nabla_{\boldsymbol{\kappa}} \text{Obj}(q_s^*) \Big|_{\boldsymbol{\kappa}=\boldsymbol{\kappa}^*} = 0$$

This means that q ∗ s (θ|κ ∗ ) is an extremum of FEDGVI, and further that it is also an extremum of GVI with D = <sup>w</sup> <sup>D</sup>KL. We now show that it is further a minimum of the global GVI objective. We consider the Hessian ∇∇<sup>κ</sup> and proceed like before.

$$\begin{aligned}\nabla \nabla_{\boldsymbol{\kappa}} \text{Obj}(m, q_s^*) &= \nabla \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \log \frac{q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \exp\{-\ell_m^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)\}}{q_s^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*) \exp\{-L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \mu(d\boldsymbol{\theta}) \\ &= \nabla \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) (L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) - \ell_m^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)) \mu(d\boldsymbol{\theta}) + \nabla \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \log \frac{q(\boldsymbol{\theta}|\boldsymbol{\kappa})}{q_s^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)} \mu(d\boldsymbol{\theta}) \\ &= \nabla \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) (L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) - \ell_m^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)) \mu(d\boldsymbol{\theta}) \\ &\quad + \nabla_{\boldsymbol{\kappa}} \left( \int (\nabla_{\boldsymbol{\kappa}} q(\boldsymbol{\theta}|\boldsymbol{\kappa})) \log \frac{q(\boldsymbol{\theta}|\boldsymbol{\kappa})}{q_s^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)} \mu(d\boldsymbol{\theta}) + \int \nabla_{\boldsymbol{\kappa}} \log q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \mu(d\boldsymbol{\theta}) \right) \\ &= \nabla \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) (L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) - \ell_m^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)) \mu(d\boldsymbol{\theta}) \\ &\quad + \int (\nabla \nabla_{\boldsymbol{\kappa}} q(\boldsymbol{\theta}|\boldsymbol{\kappa})) \log \frac{q(\boldsymbol{\theta}|\boldsymbol{\kappa})}{q_s^*(\boldsymbol{\theta}|\boldsymbol{\kappa}^*)} \mu(d\boldsymbol{\theta}) + \int (\nabla_{\boldsymbol{\kappa}} q(\boldsymbol{\theta}|\boldsymbol{\kappa})) (\nabla_{\boldsymbol{\kappa}} \log q(\boldsymbol{\theta}|\boldsymbol{\kappa})) \mu(d\boldsymbol{\theta})\end{aligned}$$

[Ashman et al.](#page-9-0) [\(2022\)](#page-9-0) point out that this last term can equivalently be expressed through it's transpose.

$$\left( \int (\nabla_{\kappa} q(\boldsymbol{\theta}|\boldsymbol{\kappa})) (\nabla_{\kappa} \log q(\boldsymbol{\theta}|\boldsymbol{\kappa})) \mu(d\boldsymbol{\theta}) \right)^{-1}$$

$$\begin{aligned} &= \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta} | \boldsymbol{\kappa}) (\nabla_{\boldsymbol{\kappa}} \log q(\boldsymbol{\theta} | \boldsymbol{\kappa})) \mu(d\boldsymbol{\theta}) + \underbrace{\int \nabla \nabla_{\boldsymbol{\kappa}} q(\boldsymbol{\theta} | \boldsymbol{\kappa}) \mu(d\boldsymbol{\theta})}_{0} \\ &= \nabla_{\boldsymbol{\kappa}} \int q(\boldsymbol{\theta} | \boldsymbol{\kappa}) \frac{1}{q(\boldsymbol{\theta} | \boldsymbol{\kappa})} \mu(d\boldsymbol{\theta}) = \mathbf{0} \end{aligned}$$

Evaluating this Hessian at κ = κ ∗ :

$$\nabla \nabla_{\boldsymbol{\kappa}} \text{Obj}(m, q_s^*) \Big|_{\boldsymbol{\kappa}=\boldsymbol{\kappa}^*} =$$

$$\nabla \nabla \kappa \int q(\boldsymbol{\theta} | \kappa) (L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) - \ell_m^*(\boldsymbol{\theta} | \kappa^*)) \mu(d\boldsymbol{\theta}) \Big|_{\kappa=\kappa^*} + \int (\nabla \nabla \kappa q(\boldsymbol{\theta} | \kappa)) \log \frac{q(\boldsymbol{\theta} | \kappa)}{q_s^*(\boldsymbol{\theta} | \kappa^*)} \mu(d\boldsymbol{\theta}) \Big|_{\kappa=\kappa^*}$$

Therefore, when summing over the individual Hessians of the clients, we get:

$$\begin{aligned} \sum_{m=1}^M \nabla \nabla \kappa \text{Obj}(m, q_s^*) \Big|_{\kappa=\kappa^*} &= \sum_{m=1}^M \nabla \nabla \kappa \int q(\boldsymbol{\theta} | \kappa) (L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) - \ell_m^*(\boldsymbol{\theta} | \kappa^*)) \mu(d\boldsymbol{\theta}) \Big|_{\kappa=\kappa^*} \\ &= \nabla \nabla \kappa \int q(\boldsymbol{\theta} | \kappa) \left( \sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) - \sum_{m=1}^M \ell_m^*(\boldsymbol{\theta} | \kappa^*) \right) \mu(d\boldsymbol{\theta}) \Big|_{\kappa=\kappa^*} \\ &= \nabla \nabla \kappa \int q(\boldsymbol{\theta} | \kappa) \log \frac{q_s^*(\boldsymbol{\theta} | \kappa^*)}{\pi(\boldsymbol{\theta}) \exp\{\sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \mu(d\boldsymbol{\theta}) \Big|_{\kappa=\kappa^*} \\ &\quad + \nabla \nabla \kappa \int q(\boldsymbol{\theta} | \kappa) \log Z_{q^*} \mu(d\boldsymbol{\theta}) \\ &= \int (\nabla \nabla \kappa q(\boldsymbol{\theta} | \kappa)) \log \frac{q_s^*(\boldsymbol{\theta} | \kappa^*)}{\pi(\boldsymbol{\theta}) \exp\{\sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \mu(d\boldsymbol{\theta}) \Big|_{\kappa=\kappa^*} \end{aligned}$$

which is a sum of positive definite matrices, and therefore, the extremum at the fixed point is a minimum.

We now compare this with the Hessian of the global objective of GVI.

$$\begin{aligned}\nabla \nabla \kappa \text{Obj}(q_s^*) &= \nabla \nabla \kappa \int q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \log \frac{q(\boldsymbol{\theta}|\boldsymbol{\kappa})}{\pi(\boldsymbol{\theta}) \exp\{-\sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \mu(d\boldsymbol{\theta}) \\ &= \nabla \kappa \left( \int (\nabla \kappa q(\boldsymbol{\theta}|\boldsymbol{\kappa})) \log \frac{q(\boldsymbol{\theta}|\boldsymbol{\kappa})}{\pi(\boldsymbol{\theta}) \exp\{-\sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \mu(d\boldsymbol{\theta}) \right. \\ &\quad \left. + \underbrace{\int (\nabla \kappa \log q(\boldsymbol{\theta}|\boldsymbol{\kappa})) q(\boldsymbol{\theta}|\boldsymbol{\kappa}) \mu(d\boldsymbol{\theta})}_0 \right) \\ &= \int (\nabla \nabla \kappa q(\boldsymbol{\theta}|\boldsymbol{\kappa})) \log \frac{q(\boldsymbol{\theta}|\boldsymbol{\kappa})}{\pi(\boldsymbol{\theta}) \exp\{-\sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \mu(d\boldsymbol{\theta}) \\ &\quad + \underbrace{\int (\nabla \kappa q(\boldsymbol{\theta}|\boldsymbol{\kappa})) (\nabla \kappa \log q(\boldsymbol{\theta}|\boldsymbol{\kappa})) \mu(d\boldsymbol{\theta})}_0 \end{aligned}$$

Therefore, we can see that, evaluated at κ = κ ∗ ,

$$\begin{aligned}\nabla \nabla_{\boldsymbol{\kappa}} \text{Obj}(q_s^*) \Big|_{\boldsymbol{\kappa}=\boldsymbol{\kappa}^*} &= \int (\nabla \nabla_{\boldsymbol{\kappa}} q(\boldsymbol{\theta} | \boldsymbol{\kappa})) \log \frac{q(\boldsymbol{\theta} | \boldsymbol{\kappa})}{\pi(\boldsymbol{\theta}) \exp\{-\sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \mu(d\boldsymbol{\theta}) \Big|_{\boldsymbol{\kappa}=\boldsymbol{\kappa}^*} \\ &= \int (\nabla \nabla_{\boldsymbol{\kappa}} q(\boldsymbol{\theta} | \boldsymbol{\kappa})) \log \frac{q_s^*(\boldsymbol{\theta} | \boldsymbol{\kappa}^*)}{\pi(\boldsymbol{\theta}) \exp\{-\sum_{m=1}^M L_m(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \mu(d\boldsymbol{\theta}) \Big|_{\boldsymbol{\kappa}=\boldsymbol{\kappa}^*} \\ &= \sum_{m=1}^M \nabla \nabla_{\boldsymbol{\kappa}} \text{Obj}(m, q_s^*) \Big|_{\boldsymbol{\kappa}=\boldsymbol{\kappa}^*}\end{aligned}$$

Hence, the Hessian of the global GVI objective is positive definite and therefore we have found a local minimum at q ∗ s (θ|κ ∗ ) through FEDGVI.

### B.4. Proof of Lemma [4.6](#page-4-2)

By combining Remark [4.1](#page-4-0) and Proposition [4.4,](#page-4-7) we can show that, under infinite computational resources, specifically if we are able to optimise over the entire space of possible distribution parametrised by θ ∈ Θ, then we are able to recover the Generalised Bayesian Posterior of [Bissiri et al.](#page-10-7) [\(2016\)](#page-10-7) in a distributed fashion by partitioning the input data and solving several smaller optimisation problems in parallel. This is achieved by using the weighted Kullback–Leibler divergence at the clients and the regular KL divergence at the server.

Under the assumption that the prior is not misspecified, we can perform distributed Bayesian updating with our framework, similar to the Bayesian Committee Machine [\(Tresp,](#page-12-4) [2000\)](#page-12-4) where we combine local posterior distributions. We aim to recover the Generalised Bayesian Posterior [\(Bissiri et al.,](#page-10-7) [2016\)](#page-10-7):

$$q_{GBI}(\theta|\boldsymbol{\kappa}) = \frac{\exp\{-\beta L(\mathbf{y}; \boldsymbol{\theta}, \mathbf{x})\} \pi(\boldsymbol{\theta})}{\int_{\Theta} \exp\{-\beta L(\mathbf{y}; \boldsymbol{\theta}, \mathbf{x})\} \pi(\boldsymbol{\theta}) \mu(d\boldsymbol{\theta})}$$

where β is some parameter that controls the learning rate from the data.

We will show that using w = β at the clients will recover this GBI posterior after a single iteration of our algorithm, and further that the algorithm shows convergence for any subsequent iteration. We assume that Q = P(Θ) and that <sup>q</sup>GBI (θ|y, <sup>x</sup>) ∈ Q. Furthermore, for simplicity we assume that the loss function <sup>L</sup>(·) is the additive across clients and that the data set is partitioned such that there are no intersections.

Proof The M clients have data sets {xm, ym}<sup>M</sup> <sup>m</sup>=1 such that <sup>x</sup><sup>k</sup> ∩ <sup>x</sup><sup>j</sup> <sup>=</sup> ∅ for all <sup>k</sup> ̸<sup>=</sup> <sup>j</sup> and we write ∪<sup>M</sup> <sup>m</sup>=1x<sup>m</sup> = x<sup>M</sup> 1 and ∪<sup>M</sup> <sup>m</sup>=1y<sup>m</sup> = y<sup>M</sup> 1 to symbolise the entire data set.

Then we can rewrite the GBI posterior as:

$$q_{GBI}(\boldsymbol{\theta} | \mathbf{y}_1^M, \mathbf{x}_1^M) = \frac{\exp\{-\beta \sum_{m=1}^M L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\} \pi(\boldsymbol{\theta})}{\int_{\Theta} \exp\{-\beta \sum_{m=1}^M L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\} \pi(\boldsymbol{\theta}) \mu(d\boldsymbol{\theta})}$$

The FEDGVI approximation then takes the following form: q (0) <sup>s</sup> (θ) = Q<sup>M</sup> <sup>m</sup>=1 exp{−ℓ (0) <sup>m</sup> (θ)}π(θ)/Z<sup>q</sup><sup>s</sup> and as we initiate ℓ (0) <sup>m</sup> (θ) = 0 we have that q (0) <sup>s</sup> (θ) = π(θ).

Then in parallel, the each client m ∈ [M] carries out their optimisation step:

The cavity distribution can be found through division as:

$$q^{\setminus m}(\boldsymbol{\theta}) \propto \frac{q_s^{(0)}(\boldsymbol{\theta})}{\exp\{-\ell_m^{(0)}(\boldsymbol{\theta})\}} = \frac{\pi(\boldsymbol{\theta})}{1} = \pi(\boldsymbol{\theta})$$

And the Generalised Variational Inference step with the cavity distribution as a local prior solves the following optimisation problem:

$$\begin{aligned} q_m^{(1)}(\boldsymbol{\theta}) &= \arg \min_{q \in \mathcal{Q}} \left\{ \mathbb{E}_{q(\boldsymbol{\theta})} [L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)] + \frac{1}{\beta} D_{KL}(q : \pi) \right\} \\ &\stackrel{(1)}{=} \arg \min_{q \in \mathcal{Q}} \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \log \frac{q(\boldsymbol{\theta})}{\pi(\boldsymbol{\theta}) \exp\{-\beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \right] \\ &\stackrel{(2)}{=} \pi(\boldsymbol{\theta}) \exp\{-\beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\} / Z_{q_m} \end{aligned}$$

Where (1) follows through the equivalence between the weighted KL divergence and the tempered loss as discussed in Appendix [B.1,](#page-15-2) and (2) follows due to the properties of a statistical divergence which is minimised when the inside of the expectation is zero and since Q = P(Θ).

This then implies that the update we send to the server is of the form:

$$\begin{aligned}\Delta_m^{(1)}(\boldsymbol{\theta}) &= -\log \frac{q_m^{(1)}(\boldsymbol{\theta})}{q_s^{(0)}(\boldsymbol{\theta})} = -\log \frac{\pi(\boldsymbol{\theta}) \exp\{-\beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}/Z_{q_m}}{\pi(\boldsymbol{\theta})} \\ &= \beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) + \log Z_{q_m}\end{aligned}$$

At the server, we can combine these such that we get:

$$\ell_s^{(1)}(\boldsymbol{\theta}) = \sum_{m=1}^M \beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) + \sum_{m=1}^M Z_{q_m} + \overbrace{\ell_s^{(0)}(\boldsymbol{\theta})}^{=0} = \beta L(\mathbf{y}_1^M; \boldsymbol{\theta}, \mathbf{x}_1^M) + \sum_{m=1}^M Z_{q_m}$$

As GBI depends on the prior and hence trusts it, we use the KL divergence at the server, which is optimal with respect to the GBI posterior [\(Zellner,](#page-12-21) [1988;](#page-12-21) [Knoblauch et al.,](#page-11-6) [2022\)](#page-11-6). Thus, the GVI objective at the server becomes:

$$\begin{aligned} q_s^{(1)}(\boldsymbol{\theta}) &= \arg \min_{q \in \mathcal{Q}} \left\{ \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \ell_s^{(1)}(\boldsymbol{\theta}) \right] + D_{KL}(q : \pi) \right\} \\ &= \arg \min_{q \in \mathcal{Q}} \left\{ \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \beta L(\mathbf{y}_1^M; \boldsymbol{\theta}, \mathbf{x}_1^M) + \sum_{m=1}^M Z_{q_m} \right] + D_{KL}(q : \pi) \right\} \\ &\stackrel{(3)}{=} \arg \min_{q \in \mathcal{Q}} \left\{ \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \beta L(\mathbf{y}_1^M; \boldsymbol{\theta}, \mathbf{x}_1^M) \right] + \sum_{m=1}^M Z_{q_m} + D_{KL}(q : \pi) \right\} \\ &= \arg \min_{q \in \mathcal{Q}} \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \log \frac{q(\boldsymbol{\theta})}{\pi(\boldsymbol{\theta}) \exp\{-\beta L(\mathbf{y}_1^M; \boldsymbol{\theta}, \mathbf{x}_1^M)\}} \right] \\ &\stackrel{(4)}{=} \pi(\boldsymbol{\theta}) \exp\{-\beta L(\mathbf{y}_1^M; \boldsymbol{\theta}, \mathbf{x}_1^M)\} / Z_{q_s^{(1)}} \end{aligned}$$

(3) follows since <sup>Z</sup><sup>q</sup><sup>m</sup> does not depend on θ, nor the variational parameters, and hence does not affect our optimisation problem. Line (4) is a result of Q = P(Θ) and the assumption that the GBI posterior is contained within this set.

This implies that the posterior that we find at the server is the Generalised Bayesian Inference posterior.

$$q_s^{(1)}(\boldsymbol{\theta}) = \pi(\boldsymbol{\theta}) \exp\{-\beta L(\mathbf{y}_1^M; \boldsymbol{\theta}, \mathbf{x}_1^M)\} / Z_{q_s^{(1)}}$$

Thereby, we have shown that FEDGVI recovers the GBI posterior under the assumptions and that this occurs after the first iteration. It remains to be shown that any further iteration steps will not change the posterior, and hence that we have recovered a fixed point as defined in Proposition [4.4.](#page-4-7)

We repeat the client optimisation steps in parallel. We first find the cavity distribution:

$$q^{|m(\theta)|} \propto \frac{q_s^{(1)}(\theta)}{\exp\{-\beta L(\mathbf{y}_m; \theta, \mathbf{x}_m)\}} \propto \frac{\pi(\theta) \exp\{-\beta \sum_{k=1}^M L(\mathbf{y}_k; \theta, \mathbf{x}_k)\}}{\exp\{-\beta L(\mathbf{y}_m; \theta, \mathbf{x}_m)\}} = \pi(\theta) \exp\{-\beta \sum_{k \neq m} L(\mathbf{y}_k; \theta, \mathbf{x}_k)\}$$

Note that we ignore the normalising constant, since, similar to the server side optimisation step before, it does not depend on the variational parameters nor θ.

The optimisation step is then given through:

$$\begin{aligned} q_m^{(2)}(\boldsymbol{\theta}) &= \arg \min_{q \in \mathcal{Q}} \left\{ \mathbb{E}_{q(\boldsymbol{\theta})} [L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)] + \frac{1}{\beta} D_{KL}(q : q^{\setminus m}) \right\} \\ &= \arg \min_{q \in \mathcal{Q}} \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \log \frac{q(\boldsymbol{\theta})}{q^{\setminus m}(\boldsymbol{\theta}) \exp\{-\beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \right] \end{aligned}$$

This statistical divergence is minimised at:

$$\begin{aligned} q_m^{(2)}(\boldsymbol{\theta}) &= q^{(m)}(\boldsymbol{\theta}) \exp\{-\beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\} / \tilde{Z} \\ &= \pi(\boldsymbol{\theta}) \exp\{-\beta \sum_{k \neq m} L(\mathbf{y}_k; \boldsymbol{\theta}, \mathbf{x}_k)\} \exp\{-\beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\} / Z_{q_m^{(2)}} \\ &= \pi(\boldsymbol{\theta}) \exp\{-\beta \sum_{m=1}^M L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\} / Z_{q_m^{(2)}} \end{aligned}$$

where we note that Z q (2) m = Z q s and we have recovered the GBI posterior we currently have as our server distribution. As a result, ∆ (2) <sup>m</sup> (θ) = −(log q (2) <sup>m</sup> (θ) − log q (1) <sup>s</sup> (θ)) = − log 1 = 0 for all m ∈ [M].

This satisfies the conditions for Proposition [4.4](#page-4-7) and hence we have achieved a fixed point, which will not change the server distribution, since:

$$\ell_s^{(2)}(\boldsymbol{\theta}) = \underbrace{\sum_{m=1}^M \underbrace{\Delta_m^{(2)}(\boldsymbol{\theta})}_{=0}}_{=0} + \ell_s^{(1)}(\boldsymbol{\theta}) = \ell_s^{(1)}(\boldsymbol{\theta}) = \beta L(\mathbf{y}_1^M; \boldsymbol{\theta}, \mathbf{x}_1^M)$$

which means that the server optimisation routine would not be different from the one during the previous iteration.

$$q_s^{(2)}(\boldsymbol{\theta}) = \arg \min_{q \in \mathcal{Q}} \left\{ \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \ell_s^{(2)}(\boldsymbol{\theta}) \right] + D_{KL}(q : \pi) \right\} = \arg \min_{q \in \mathcal{Q}} \left\{ \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \ell_s^{(1)}(\boldsymbol{\theta}) \right] + D_{KL}(q : \pi) \right\} = q_s^{(1)}(\boldsymbol{\theta})$$

And thus q (2) <sup>s</sup> (θ) = q (1) <sup>s</sup> (θ) = q ∗ GBI (θ|y<sup>M</sup> 1 , x<sup>M</sup> 1 ).

For the moreover part, we define the damping parameter δ = 1 <sup>M</sup> , and show that <sup>q</sup> (t) <sup>s</sup> (θ) → qGBI (θ|y<sup>M</sup> 1 , x<sup>M</sup> 1 ) as t → ∞. As the data here is implicit, we simplify notation by denoting the losses of a client as Lm(θ) and the GBI posterior as qGBI (θ). Furthermore, we assume that the GBI learning rate parameter β is implicitly included in each client's loss. Then by the usual modes of convergence, we show that:

$$\left| q_s^{(t)}(\boldsymbol{\theta}) - q_{GBI}(\boldsymbol{\theta}) \right| \rightarrow 0$$

Note that under KL divergences at the server and client, we will have that ℓ (t) <sup>s</sup> (θ) = P<sup>M</sup> <sup>m</sup>=1 ℓ (t) <sup>m</sup> (θ) (see proof of Remark [4.1\)](#page-4-0).

$$\left| \pi(\boldsymbol{\theta}) \exp \left\{ -\sum_m^M \ell_m^{(t)}(\boldsymbol{\theta}) \right\} - \pi(\boldsymbol{\theta}) \exp \left\{ -\sum_m^M L_m(\boldsymbol{\theta}) \right\} \right| = \pi(\boldsymbol{\theta}) \left| \exp \left\{ -\sum_m^M \ell_m^{(t)}(\boldsymbol{\theta}) \right\} - \exp \left\{ -\sum_m^M L_m(\boldsymbol{\theta}) \right\} \right|$$

This converges when the exponents are equal, hence it is sufficient to prove that ∀m ∈ [M] we have ℓ (t) <sup>m</sup> (θ) → Lm(θ).

Since for all m ∈ [M], at each iteration t we have that under the KL divergences:

$$\begin{aligned} q^{\setminus m}(\boldsymbol{\theta}) &\propto \frac{q_s^{(t-1)}(\boldsymbol{\theta})}{\exp\{-\ell_m^{(t-1)}(\boldsymbol{\theta})\}} = \frac{\pi(\boldsymbol{\theta}) \exp\{-\sum_{m=1}^M \ell_m^{(t-1)}(\boldsymbol{\theta})\}}{\exp\{-\ell_m^{(t-1)}(\boldsymbol{\theta})\}} = \pi(\boldsymbol{\theta}) \exp\left\{-\sum_{k \neq m} \ell_k^{(t-1)}(\boldsymbol{\theta})\right\} \\ q_m^{(t)}(\boldsymbol{\theta}) &\propto \exp\{L_m(\boldsymbol{\theta})\} q^{\setminus m}(\boldsymbol{\theta}) \\ \Delta_m^{(t)}(\boldsymbol{\theta}) &= -\frac{1}{M} \log \frac{\exp\{L_m(\boldsymbol{\theta})\} \pi(\boldsymbol{\theta}) \exp\left\{-\sum_{k \neq m} \ell_k^{(t-1)}(\boldsymbol{\theta})\right\}}{\pi(\boldsymbol{\theta}) \exp\{-\sum_{m=1}^M \ell_m^{(t-1)}(\boldsymbol{\theta})\}} = \frac{1}{M} L_m(\boldsymbol{\theta}) - \frac{1}{M} \ell_m^{(t-1)}(\boldsymbol{\theta}) \\ \ell_m^{(t)}(\boldsymbol{\theta}) &= \ell_m^{(t-1)}(\boldsymbol{\theta}) + \Delta_m^{(t)}(\boldsymbol{\theta}) = \frac{1}{M} L_m(\boldsymbol{\theta}) + \frac{M-1}{M} \ell_m^{(t-1)}(\boldsymbol{\theta}) \end{aligned}$$

By expansion of ℓ (t−1) <sup>m</sup> (θ), by recursively applying the definition above, we get the following closed form expression:

$$\ell_m^{(t-1)}(\boldsymbol{\theta}) = \left( \left( \frac{M-1}{M} \right) \frac{1}{M} + \left( \frac{M-1}{M} \right) \left( \frac{M-1}{M} \right) \frac{1}{M} + \dots + \left( \frac{M-1}{M} \right)^t \frac{1}{M} \ell_m^{(0)}(\boldsymbol{\theta}) \right) L_m(\boldsymbol{\theta})$$

written as a summation and recalling that ℓ (0) <sup>m</sup> (θ) = 0 by definition, we can interpret this as the series:

$$\ell_m^{(t)}(\boldsymbol{\theta}) = L_m(\boldsymbol{\theta}) \sum_{i=0}^{t-1} \frac{1}{M} \left( \frac{M-1}{M} \right)^t$$

which is a geometric series. And since <sup>M</sup>−<sup>1</sup> <sup>M</sup> <sup>∈</sup> (0, 1) by elementary analysis this converges, as <sup>t</sup> → ∞, to the limit

$$\lim_{t \rightarrow \infty} \ell_m^{(t)}(\boldsymbol{\theta}) = L_m(\boldsymbol{\theta}) \frac{1}{M} M = L_m(\boldsymbol{\theta}).$$

Therefore, as t → ∞ q (t) <sup>s</sup> (θ) → qGBI (θ) θ almost everywhere. We can only guarantee almost everywhere pointwise convergence, since integral operators such as the KL divergence only guarantee equivalence up to null sets.

Notably, the reason for using the cavity distribution instead of some other effective prior for the client optimisation step is that we want to recover the (generalised) Bayesian posterior eventually with our framework assuming that we can optimise over the entire space of probability measures that characterise their respective probability distributions. We further assume that we can find a global minimiser of any optimisation problem. Then, under these assumptions, we would like to not change the current posterior any further after recovering the GBI posterior.

We have previously shown that our algorithm achieves just this, and we can furthermore show that the cavity distribution is indeed the only choice in the client update that causes this.

# B.5. Proof of Theorem [4.9](#page-5-1)

We are interested in verifying whether the cavity distribution is necessary in Equation [\(4\)](#page-3-2). It acts to regularise the optimisation problem at the client, which we restate here, using some arbitrary probability density ρ ∈ P(Θ):

$$q_m^{(t)}(\boldsymbol{\theta}) = \arg \min_{q \in \mathcal{Q}} \left\{ \mathbb{E}_{q(\boldsymbol{\theta})} \left[ L_m^{(t)}(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) \right] + D(q : \rho) \right\}$$

where it is regularised by D(· : ρ). It is clear that this should not be the prior distribution after the server has additional information about client data available since we would not be doing anything different for subsequent updates and this would result in a Bayesian Committee Machine where each client does not learn from the others. Therefore it is imperative to ask what this 'effective prior' ρ should be? And in fact it turns out that it needs to be the cavity distribution.

We will approach this problem by considering the case where we know what we would want to target in the optimization problem and hence the sequence {q (t) <sup>s</sup> (θ)}t∈N<sup>0</sup> should converge to. We, however, have to restrict ourselves to the Federated Learning scenario and therefore any distribution that we come up with needs to satisfy the Assumptions [4.7](#page-5-4) and [4.8.](#page-5-5) For this we require the following assumption so that we are able to target the GBI posterior.

Assumption B.2. We are able to find global minimisers over the entire space of probability distributions parametrised by θ, P(Θ).

Then it turns out that this regularising distribution is uniquely described by Theorem [4.9,](#page-5-1) which we restate here.

Theorem [4.9](#page-5-1) *Let the assumptions be as in Lemma [4.6,](#page-4-2) i.e.* Q = P(Θ)*,* D = 1 <sup>β</sup> DKL *for* β > 0*,* D<sup>s</sup> = DKL*,* L (t) <sup>m</sup> = L*, and* τ<sup>m</sup> = 1*, and further assume that Assumptions [4.7](#page-5-4) and [4.8](#page-5-5) are satisfied, then the following are equivalent:*

- *1.* ∃t ∈ [T] *for which* q
- (t) <sup>s</sup> (θ) = qGBI(θ) *(a.e.) is invariant under further* FEDGVI *updates.*
- *2. The cavity distribution regularises the client optimisation problem.*

Proof (2 =⇒ 1) This is a direct consequence of Lemma [4.6](#page-4-2) and can easily be seen by iterating through the algorithm with the cavity distribution.

(1 =⇒ 2) Without loss of generality we consider the GBI posterior to be found after the first iteration. We show that the unique way that satisfies the axioms and does not change the GBI posterior at the second iteration (or any further iterations) is uniquely achieved by the cavity distribution. By the statement we have

$$q_s^{(2)}(\boldsymbol{\theta}) = \exp\{-\ell_s^{(2)}(\boldsymbol{\theta})\}\pi(\boldsymbol{\theta})/Z_s^{(2)} = \exp\{-\ell_s^{(1)}(\boldsymbol{\theta})\}\pi(\boldsymbol{\theta})/Z_s^{(1)} = q_s^{(1)}(\boldsymbol{\theta}).$$

We now need to relate this to the client updates and hence the solutions of the client optimization problem.

$$\begin{aligned}
q_s^{(2)}(\boldsymbol{\theta}) = q_s^{(1)}(\boldsymbol{\theta}) &\iff \exp\{-\ell_s^{(2)}(\boldsymbol{\theta})\}/Z_s^{(2)} = \exp\{-\ell_s^{(1)}(\boldsymbol{\theta})\}/Z_s^{(1)} \\
&\iff \ell_s^{(2)}(\boldsymbol{\theta}) + \log Z_s^{(2)} = \ell_s^{(1)}(\boldsymbol{\theta}) + \log Z_s^{(1)} \\
&\iff \ell_s^{(2)}(\boldsymbol{\theta}) = \ell_s^{(1)}(\boldsymbol{\theta}) + C, \quad C \in \mathbb{R} \\
&\iff \sum_{m=1}^M \Delta_m^{(2)}(\boldsymbol{\theta}) + \ell_s^{(1)}(\boldsymbol{\theta}) = \ell_s^{(1)}(\boldsymbol{\theta}) + C \\
&\iff \sum_{m=1}^M \Delta_m^{(2)}(\boldsymbol{\theta}) = C \\
&\iff \sum_{m=1}^M \log \frac{q_m^{(2)}(\boldsymbol{\theta})}{q_s^{(1)}(\boldsymbol{\theta})} = C \\
&\iff \prod_{m=1}^M q_m^{(2)}(\boldsymbol{\theta}) = K \left( q_s^{(1)}(\boldsymbol{\theta}) \right)^M, \quad K = e^C
\end{aligned} \tag{10}$$

Now, for some transformation operator ξ<sup>m</sup> : P(Θ) → P(Θ) acting on the information available at the client from the server in the form of the current approximate posterior, which we denote as ξm[q (1) <sup>s</sup> ](θ), that satisfies the Assumptions [4.7](#page-5-4) and [4.8,](#page-5-5) we get the client optimisation problem ∀m:

$$\begin{aligned} q_m^{(2)}(\boldsymbol{\theta}) &= \arg \min_{q \in \mathcal{P}(\Theta)} \left\{ \mathbb{E}_{q(\boldsymbol{\theta})} [L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)] + \frac{1}{\beta} D_{KL}(q : \xi_m[q_s^{(1)}]) \right\} \\ &= \arg \min_{q \in \mathcal{P}(\Theta)} \left\{ \frac{1}{\beta} \mathbb{E}_{q(\boldsymbol{\theta})} [-\beta \log \exp\{-L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}] + \frac{1}{\beta} \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \log \frac{q(\boldsymbol{\theta})}{\xi_m[q_s^{(1)}](\boldsymbol{\theta})} \right] \right\} \\ &= \arg \min_{q \in \mathcal{P}(\Theta)} \left\{ \frac{1}{\beta} \mathbb{E}_{q(\boldsymbol{\theta})} \left[ \log \frac{q(\boldsymbol{\theta})}{\exp\{-\beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\} \xi_m[q_s^{(1)}](\boldsymbol{\theta})} \right] \right\} \\ &\Rightarrow q_m^{(2)}(\boldsymbol{\theta}) = \exp\{-\beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\} \xi_m[q_s^{(1)}](\boldsymbol{\theta}) / Z_m^{(2)} \end{aligned}$$

Substituting this into Equation [\(10\)](#page-24-0) and using the definition of q (1) <sup>s</sup> (θ) we can derive a relation between the individual client approximations.

$$\begin{aligned} & \prod_{m=1}^M q_m^{(2)}(\boldsymbol{\theta}) = K \left( q_s^{(1)}(\boldsymbol{\theta}) \right)^M \\ & \prod_{m=1}^M \frac{\xi_m[q_s^{(1)}](\boldsymbol{\theta}) \exp\{-\beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}}{Z_m^{(2)}} = K(\pi(\boldsymbol{\theta}))^M \exp \left\{ -M\beta \sum_{m=1}^M L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) \right\} / \left( Z_s^{(1)} \right)^M \\ & \prod_{m=1}^M \xi_m[q_s^{(1)}](\boldsymbol{\theta})/Z_m^{(2)} = K(\pi(\boldsymbol{\theta}))^M \exp \left\{ -(M-1)\beta \sum_{m=1}^M L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m) \right\} / \left( Z_s^{(1)} \right)^M \\ & \Rightarrow \prod_{m=1}^M \xi_m[q_s^{(1)}](\boldsymbol{\theta}) \propto \prod_{m=1}^M \pi(\boldsymbol{\theta}) \exp \left\{ -\beta \sum_{k \neq m} L(\mathbf{y}_k; \boldsymbol{\theta}, \mathbf{x}_k) \right\} \propto \prod_{m=1}^M \frac{q_s^{(1)}(\boldsymbol{\theta})}{\exp\{-\beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}} \end{aligned}$$

Here, proportional '∝' means equivalent up to some constant independent of θ. To see that the cavity distribution is in fact the only choice that satisfies the above equation, we need to recall the two axioms: (Assumption [4.8\)](#page-5-5) ξm[q (1) <sup>s</sup> ](θ) needs to be generated in the same way across clients, and (Assumption [4.7\)](#page-5-4) since we are in federated learning, each client will only be able to access it's own data. This implies that we can write ξm[q (1) <sup>s</sup> ](θ) as a function of the current approximation and the client data, ξm[q (1) <sup>s</sup> ](θ) = ξ[q (1) <sup>s</sup> , ym, xm](θ).

$$\prod_{m=1}^M \xi[q_s^{(1)}, \mathbf{y}_m, \mathbf{x}_m](\boldsymbol{\theta}) \propto \prod_{m=1}^M \frac{q_s^{(1)}(\boldsymbol{\theta})}{\exp\{-\beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\}}$$

The only client that would have access to an explicit expression for the denominator would be client m, to which the data {xm, ym} belongs, and hence it must be entirely contained within that client's regularisation term ξm. Therefore, we can conclude that q (2) <sup>m</sup> (θ) = q (1) <sup>s</sup> (θ) and find a closed form for ξm[q (1) <sup>s</sup> ](θ). Note that this implies C = 0 and hence K = 1.

$$\exp\{-\beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\} \xi_m[q_s^{(1)}](\boldsymbol{\theta}) / Z_{q_m^{(2)}} = \exp\{-\sum_{m=1}^M \beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\} \pi(\boldsymbol{\theta}) / Z_{q_s^{(1)}}$$

$$\xi_m[q_s^{(1)}](\boldsymbol{\theta}) \propto \frac{\exp\{-\sum_{m=1}^M \beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\} \pi(\boldsymbol{\theta}) / Z_{q_s^{(1)}}}{\exp\{-\beta L(\mathbf{y}_m; \boldsymbol{\theta}, \mathbf{x}_m)\} / Z_{q_m^{(2)}}}$$

This is exactly the cavity distribution as described in Equation [\(3\)](#page-3-1).

This gives a justification for using the cavity distribution in our algorithm, since under the assumption that the prior is well specified, we would like to converge to the generalised Bayesian posterior distribution. Furthermore, we can note that this single step of FEDGVI recovers the principle of the Bayesian Committee Machine (BCM) of [Tresp](#page-12-4) [\(2000\)](#page-12-4) where we use generalised loss functions instead of the negative log likelihood in our formulation. Furthermore, a single pass through FEDGVI—with the divergences as described above— will recover a generalised version of the BCM irregardless of the space we optimise over.

*Remark* B.3*.* For the last two proofs we have assumed that we can find the global minimisers of the equations. This isn't strictly necessary to have since the use of the (weighted) Kullback–Leibler divergence allows us to formulate a closed form expression for what these will look like.

# B.6. Proof of Proposition [4.10](#page-5-2)

This proposition is a direct result of Proposition 3.1 in [Altamirano et al.](#page-9-11) [\(2023\)](#page-9-11) and the proof is analogous, we merely include it here for completeness. And while the stated result is in a regression setting, it can be be extended to the classification setting similar to [Altamirano et al.](#page-9-12) [\(2024\)](#page-9-12) where Gaussian Processes are considered.

We assume that each client has a data set {xi} n<sup>m</sup> <sup>i</sup>=1 of size <sup>n</sup>m. The divergence operator ∇ · <sup>f</sup>(x) is defined in the usual way as the inner product between the vector of partial derivative operators and the vector of some vector valued function f(x) as ∇ · f(x) = ⟨(∂/∂x1, ...,(∂/∂xd) <sup>⊤</sup>,(f1(x), ..., fd(x))<sup>⊤</sup>⟩, and ∇xg(x) is the Jacobian, the vector of partial derivatives of g(x). We further assume that Ξ ⊆ <sup>R</sup> d , and that pθ(x) ∈ P(Θ).

Proof The loss of some client m ∈ [M] at some arbitrary iteration t ∈ [T] is given by

$$\hat{D}(\boldsymbol{\theta}, \mathbb{P}_{n_m}) := \frac{1}{n_m} \sum_{i=1}^{n_m} \underbrace{\|w_m^{(t)\top} \nabla_{\mathbf{x}} \log p_{\boldsymbol{\theta}}(\mathbf{x}_i)\|_2^2}_{(1)} + 2 \underbrace{\nabla \cdot (w_m^{(t)} w_m^{(t)\top} \nabla_{\mathbf{x}} \log p_{\boldsymbol{\theta}}(\mathbf{x}_i))}_{(2)}$$

where ∇<sup>x</sup> log pθ(xi) = ∇xη(θ) <sup>⊤</sup>ϕ(xi) + ∇xh(xi). We can then expand the terms in the above terms which we then give equal up to an additive constant independent of θ.

$$\begin{aligned} (1) &= (w_m^{(t)})^\top (\nabla_{\mathbf{x}} \phi(\mathbf{x}_i)^\top \boldsymbol{\eta}(\boldsymbol{\theta}) + \nabla_{\mathbf{x}} h(\mathbf{x}_i))^\top (w_m^{(t)})^\top (\nabla_{\mathbf{x}} \phi(\mathbf{x}_i)^\top \boldsymbol{\eta}(\boldsymbol{\theta}) + \nabla_{\mathbf{x}} h(\mathbf{x}_i))) \\ &= (w_m^{(t)})^\top \nabla_{\mathbf{x}} \phi(\mathbf{x}_i)^\top \boldsymbol{\eta}(\boldsymbol{\theta}) + (w_m^{(t)})^\top \nabla_{\mathbf{x}} \phi(\mathbf{x}_i)^\top \boldsymbol{\eta}(\boldsymbol{\theta}) + (w_m^{(t)})^\top \nabla_{\mathbf{x}} h(\mathbf{x}_i) + (w_m^{(t)})^\top \nabla_{\mathbf{x}} h(\mathbf{x}_i) \\ &\quad + 2(w_m^{(t)})^\top \nabla_{\mathbf{x}} \phi(\mathbf{x}_i)^\top \boldsymbol{\eta}(\boldsymbol{\theta}) + ((w_m^{(t)})^\top \nabla_{\mathbf{x}} h(\mathbf{x}_i)) \\ &\stackrel{+c}{=} \boldsymbol{\eta}(\boldsymbol{\theta})^\top \nabla_{\mathbf{x}} \phi(\mathbf{x}_i) w_m^{(t)} w_m^{(t)} \nabla_{\mathbf{x}} \phi(\mathbf{x}_i)^\top \boldsymbol{\eta}(\boldsymbol{\theta}) + \boldsymbol{\eta}(\boldsymbol{\theta})^\top \nabla_{\mathbf{x}} \phi(\mathbf{x}_i) w_m^{(t)} w_m^{(t)} \nabla_{\mathbf{x}} h(\mathbf{x}_i) \end{aligned}$$

where the last line follows since the middle terms are independent of θ as long as the weight function is independent of θ.

$$(2) = \nabla \cdot (w_m^{(t)} w_m^{(t)} \top \nabla_{\mathbf{x}} \boldsymbol{\eta}(\boldsymbol{\theta}) \top \boldsymbol{\phi}(\mathbf{x}_i)) + \nabla \cdot (w_m^{(t)} w_m^{(t)} \top \nabla_{\mathbf{x}} h(\mathbf{x}_i))$$

$$\stackrel{+c}{=} \boldsymbol{\eta}(\boldsymbol{\theta})^\top (\nabla \cdot (w_m^{(t)} w_m^{(t)} \top \nabla_{\mathbf{x}} \boldsymbol{\phi}(\mathbf{x}_i)))$$

Then, this has the form Dˆ(θ, <sup>P</sup>) <sup>+</sup><sup>c</sup> <sup>=</sup> <sup>η</sup>(θ) <sup>⊤</sup>Λ (t) <sup>m</sup> η(θ) + η(θ) <sup>⊤</sup>ν (t) <sup>m</sup> , where

$$\Lambda_m^{(t)} := \frac{1}{n_m} \sum_{i=1}^{n_m} \nabla_{\mathbf{x}} \phi(\mathbf{x}_i) w_m^{(t)} w_m^{(t)} \top \nabla_{\mathbf{x}} \phi(\mathbf{x}_i) \top \quad \text{and} \quad \boldsymbol{\nu}_m^{(t)} := \frac{2}{n_m} \sum_{i=1}^{n_m} \nabla \cdot (w_m^{(t)} w_m^{(t)} \top \nabla_{\mathbf{x}} \phi(\mathbf{x}_i)).$$

The first art follows by setting q (t) <sup>m</sup> (θ) ∝ q \m (t) (θ) exp{−βnm(η(θ) <sup>⊤</sup>Λ (t) <sup>m</sup> η(θ) + η(θ) <sup>⊤</sup>ν (t) <sup>m</sup> )}. Then, if η(θ) = θ, and the local cavity distribution has the form q \m (t) (θ) ∝ exp{−<sup>1</sup> 2 (θ − µ (t) \m) <sup>⊤</sup>Σ (t) \m −1 (θ − µ (t) \m)}, then the local posterior is conjugate and is given by q (t) <sup>m</sup> (θ) ∝ exp{−<sup>1</sup> 2 (θ − µ (t) <sup>m</sup> ) <sup>⊤</sup>Σ (t) m −1 (θ − µ (t) <sup>m</sup> )}, where

$$\begin{aligned} q_m^{(t)}(\boldsymbol{\theta}) &\propto q_{(t)}^m(\boldsymbol{\theta}) \exp\{-\beta n_m(\boldsymbol{\eta}(\boldsymbol{\theta})^\top \boldsymbol{\Lambda}_m^{(t)} \boldsymbol{\eta}(\boldsymbol{\theta}) + \boldsymbol{\eta}(\boldsymbol{\theta})^\top \boldsymbol{\nu}_m^{(t)})\} \\ &\propto \exp\{-\frac{1}{2}(\boldsymbol{\theta} - \boldsymbol{\mu}_{\setminus m}^{(t)})^\top \Sigma_{\setminus m}^{(t)} - (\boldsymbol{\theta} - \boldsymbol{\mu}_{\setminus m}^{(t)})\} \exp\{-\beta n_m(\boldsymbol{\theta}^\top \boldsymbol{\Lambda}_m^{(t)} \boldsymbol{\theta} + \boldsymbol{\theta}^\top \boldsymbol{\nu}_m^{(t)})\} \\ &\propto \exp\{-\frac{1}{2}[\boldsymbol{\theta}^\top \Sigma_{\setminus m}^{(t)} - \boldsymbol{\theta} - \boldsymbol{\theta}^\top \Sigma_{\setminus m}^{(t)} - \boldsymbol{\mu}_{\setminus m}^{(t)} + 2\beta n_m \boldsymbol{\theta}^\top \boldsymbol{\Lambda}_m^{(t)} \boldsymbol{\theta} + 2\beta n_m \boldsymbol{\theta}^\top \boldsymbol{\nu}_m^{(t)}]\} \\ &\propto \exp\{-\frac{1}{2}(\boldsymbol{\theta} - \boldsymbol{\mu}_m^{(t)})^\top \Sigma_m^{(t)} - (\boldsymbol{\theta} - \boldsymbol{\mu}_m^{(t)})\} \end{aligned}$$

where in the last line, we complete the square and get parameters

$$\Sigma_m^{(t)} - \Sigma_m^{(t)} + \beta n_m \Lambda_m^{(t)} \quad \text{and} \quad \boldsymbol{\mu}_m^{(t)} := \Sigma_m^{(t)} (\Sigma_m^{(t)} - \boldsymbol{\mu}_m^{(t)} - \Lambda_m^{(t)} \boldsymbol{\nu}_m^{(t)}).$$

The moreover part can now easily be seen. The update will be quadratic in θ and hence summing these results in a quadratic function, and since the posterior will have Gaussian distribution, the expectation with respect to the posterior of this quadratic function will have closed form. Therefore, if the divergence at the server allows for closed form solutions between Multivariate Gaussians, then the entire Equation [\(7\)](#page-3-0) will have a closed form optimisation procedure that does not require sampling to approximate integrals.

Note we have implicitly used the weighted KL divergence with parameter βnm. Note also that this does not immediately follow from Lemma [4.6](#page-4-2) since the weighting function is allowed to change depending on the iteration and the client. In our experiments, we for instance use the weighting function as measuring some deviation of a data point to the cavity mean. Furthermore, the weighting function does depend on the data point, but we suppress this dependence here to lighten notation.

### B.7. Proof of Theorem [4.12](#page-5-0)

This result is more involved to prove where we show by induction that at each iteration, the posterior generated at the server is robust to outliers through the robustness of each client's loss function to outliers. To prove this result, we first consider what we mean by robustness and introduce some terminology. We consider the empirical data distribution of all clients <sup>P</sup><sup>n</sup> = 1 n P<sup>n</sup> i δ<sup>x</sup><sup>i</sup> which is perturbed by some Huber contamination with parameter <sup>ε</sup> at some adversarially chosen data point z ∈ Ξ as <sup>P</sup>n,ε,z := (1 − ε)<sup>P</sup><sup>n</sup> + εδz, where the subscript n indicates how many data points are drawn from the distribution. Note that <sup>P</sup><sup>n</sup> = 1 n P<sup>M</sup> m=1 P<sup>n</sup><sup>m</sup> <sup>i</sup>=1 δ<sup>x</sup>mi = 1 n P<sup>M</sup> <sup>m</sup>=1 <sup>n</sup>mPnm. We then write <sup>q</sup> (t) <sup>s</sup> (θ; <sup>P</sup>n,ε,z) to indicate a distribution with respect to data generated from the specified DGP. We first recall the notion of robustness introduced by [Ghosh & Basu](#page-10-18) [\(2016a\)](#page-10-18). Note that we suppress the measure µ(dθ) in the following and simply write dθ. The posterior influence is given by:

$$\text{PIF}(z, \boldsymbol{\theta}, \mathbb{P}_n) := \lim_{\varepsilon \downarrow 0} \frac{q_s^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) - q_s^{(t)}(\boldsymbol{\theta}; \mathbb{P}_n)}{\varepsilon} = \frac{d}{d\varepsilon} q_s^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})|_{\varepsilon=0}$$

where the last line follows by L'Hopital's rule. [Ghosh & Basu](#page-10-18) [\(2016a\)](#page-10-18) further show, and one can easily check, that for q (t) <sup>s</sup> (θ; <sup>P</sup>n,ε,z) = π(θ) exp{−ℓ (t) <sup>s</sup> (θ; <sup>P</sup>n,ε,z)}/ R π(θ) exp{−ℓ (t) <sup>s</sup> (θ; <sup>P</sup>n,ε,z)}dθ this is equal to:

$$\text{PIF}(z, \boldsymbol{\theta}, \mathbb{P}_n) = q_s^{(t)}(\boldsymbol{\theta}; \mathbb{P}_n) \left( -\frac{d}{d\varepsilon} \ell_s^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \Big|_{\varepsilon=0} + \int_{\Theta} \frac{d}{d\varepsilon} \ell_s^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \Big|_{\varepsilon=0} \Pi(d\boldsymbol{\theta}) \right)$$

We call loss robust if it has finite posterior influence, i.e. supθ∈<sup>Θ</sup> supz∈<sup>Ξ</sup> |PIF(z, <sup>θ</sup>, <sup>P</sup>n)| <sup>&</sup>lt; ∞. To this end, we now state a Lemma due to [Matsubara et al.](#page-11-11) [\(2022\)](#page-11-11), adapted to our notation for FEDGVI which we have rephrased into Definition [4.11.](#page-5-3) Lemma B.4 [\(Matsubara et al.](#page-11-11) [\(2022\)](#page-11-11)). *Let* q (t) (·) (θ; <sup>P</sup>n) *be a posterior computed at the server or the client with fixed* n ∈ <sup>N</sup> *with loss* ℓ (t) (·) (θ; <sup>P</sup>n) *and a prior* π(θ)*. Suppose that* ℓ (t) (·) (θ; <sup>P</sup>n) *is lower bounded and that* π(θ) *is upper bounded over* θ ∈ Θ*, for any* <sup>P</sup>n*. Then if there exists some function* γ (t) (·) : Θ → R *such that*

1. 1. 
   $$\sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} \ell^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \right|_{\varepsilon=0} \leq \gamma^{(t)}(\boldsymbol{\theta})$$
2. 2.  $\sup_{\boldsymbol{\theta} \in \Theta} \pi(\boldsymbol{\theta}) \gamma^{(t)}(\boldsymbol{\theta}) < \infty$ , and
3. 3.  $\int_{\Theta} \pi(\boldsymbol{\theta}) \gamma^{(t)}(\boldsymbol{\theta}) d\boldsymbol{\theta} < \infty$

*hold, then* q (t) (·) (θ; <sup>P</sup>n) *is globally bias–robust.*

We provide further clarification on these conditions in Appendix [C.2,](#page-33-1) and we are now able to give the proof of Theorem [4.12.](#page-5-0)

Proof By the Lemma [B.4,](#page-26-1) we need to show that

$$\sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} \ell_s^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \right|_{\varepsilon=0} \leq \gamma_s^{(t)}(\boldsymbol{\theta})$$

and that this γ (t) <sup>s</sup> (θ) satisfies conditions (2.) and (3.) of the Lemma. Per assumption we know that the clients are robust to likelihood misspecification, so we need to relate the server loss to the client posterior influence functions. To this end, we consider the loss at the server.

$$\ell_s^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) = \sum_{m=1}^M \Delta_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) + \ell_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})$$

where for each client, the update is given through Equation [\(5\)](#page-3-3)

$$\begin{aligned} \Delta_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) &= -\log \frac{q_s^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})}{q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})} = -\log \frac{\int q_s^{(m)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \exp\{-\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})\} d\boldsymbol{\theta}}{q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})} d\boldsymbol{\theta} \\ &= -\log \frac{\int \frac{q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})}{\exp\{-\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})\}} d\boldsymbol{\theta}}{q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})} \\ &= -\log \frac{\int \frac{\exp\{-\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})\}}{q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})} d\boldsymbol{\theta}}{q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})} \\ &= -\log \frac{\exp\{-\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})\}}{\exp\{-\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})\}} \\ &= -\log \frac{\int \frac{q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})}{\exp\{-\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})\}} \exp\{-\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})\} d\boldsymbol{\theta}}{q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})} \\ &= -\log \frac{\exp\{-\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})\}}{\exp\{-\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})\}} + \log Z_m^{(t)}(\mathbb{P}_{n,\varepsilon,z}) \end{aligned}$$

Therefore,

$$\begin{aligned} \ell_s^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) &= \sum_{m=1}^M -\log \frac{\exp\{-\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})\}}{\exp\{-\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})\}} + \log Z_m^{(t)}(\mathbb{P}_{n,\varepsilon,z}) + \ell_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \\ &= \sum_{m=1}^M -\log \exp\{-\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})\} + \sum_{m=1}^M \sum_{i=1}^t \log Z_m^{(i)}(\mathbb{P}_{n,\varepsilon,z}) \\ &= \sum_{m=1}^M \beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z}) + \sum_{m=1}^M \sum_{i=1}^t \log Z_m^{(i)}(\mathbb{P}_{n,\varepsilon,z}) \end{aligned}$$

We will now show by induction on t that the posterior at the server is robust.

Concretely we will show that ∀t ∈ [T], T ∈ <sup>N</sup>, and M ∈ <sup>N</sup> finite, then

$$\sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} \ell_s^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \right|_{\varepsilon=0} \leq \beta \sum_{m=1}^M n_m \sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z}) \right|_{\varepsilon=0} + \sum_{m=1}^M \sum_{i=1}^t \sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} \log Z_m^{(i)}(\mathbb{P}_{n,\varepsilon,z}) \right|_{\varepsilon=0} \leq \gamma_s^{(t)}(\boldsymbol{\theta})$$

such that this function γ (t) <sup>s</sup> (θ) satisfies the conditions of Lemma [B.4.](#page-26-1) Note that the first inequality follows by Minkowski's inequality.

We begin by considering the case where t = 1, then we have q (1−1) <sup>s</sup> (θ; <sup>P</sup>n,ε,z) = π(θ) and L (1−1) <sup>m</sup> (θ; <sup>P</sup>n,ε,z) = 0 as initialised in the algorithm.

Consider the term <sup>d</sup> dε log Z (i) <sup>m</sup> (<sup>P</sup>n,ε,z) <sup>ε</sup>=0, then we have

$$\begin{aligned} \frac{d}{d\varepsilon} \log Z_m^{(1)}(\mathbb{P}_{n,\varepsilon,z})|_{\varepsilon=0} &= \frac{\frac{d}{d\varepsilon} Z_m^{(1)}(\mathbb{P}_{n,\varepsilon,z})|_{\varepsilon=0}}{Z_m^{(1)}(\mathbb{P}_{n,\varepsilon,z})|_{\varepsilon=0}} \\ &= \frac{\int \frac{d}{d\varepsilon} q_s^{(1-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \exp\{-\beta n_m L_m^{(1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})\} \Big|_{\varepsilon=0} d\boldsymbol{\theta}}{Z_m^{(1)}(\mathbb{P}_{n_m})} \\ &= \int \frac{\frac{d}{d\varepsilon} \pi(\boldsymbol{\theta}) \exp\{-\beta n_m L_m^{(1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})\} \Big|_{\varepsilon=0} d\boldsymbol{\theta}}{Z_m^{(1)}(\mathbb{P}_{n_m})} \\ &= - \int \left( \frac{d}{d\varepsilon} \beta n_m L_m^{(1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z}) \Big|_{\varepsilon=0} \right) q_m^{(1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) d\boldsymbol{\theta} \end{aligned}$$

where the last equation follows since <sup>d</sup> dx exp{f(x)} = exp{f(x)} d dx <sup>f</sup>(x).

Consequently, using Jensen's inequality

$$\begin{aligned} & \sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} \ell_s^{(1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \right|_{\varepsilon=0} \\ & \leq \beta \sum_{m=1}^M n_m \sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} L_m^{(1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z}) \right|_{\varepsilon=0} + \sum_{m=1}^M \sup_{z \in \Xi} \left| \int \left( \frac{d}{d\varepsilon} \beta n_m L_m^{(1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z}) \right) q_m^{(1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) d\boldsymbol{\theta} \right| \\ & \leq \beta \sum_{m=1}^M n_m \left( \sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} L_m^{(1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z}) \right|_{\varepsilon=0} \right) + \int \sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} L_m^{(1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z}) \right|_{\varepsilon=0} \left| q_m^{(1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) d\boldsymbol{\theta} \right| \end{aligned}$$

Then, if L (1) <sup>m</sup> (θ; <sup>P</sup>nm,ε,z) is robust, then there exists some function γ (1) <sup>m</sup> (θ) : Θ → <sup>R</sup> such that supz∈<sup>Ξ</sup> d dεL (1) <sup>m</sup> (θ; <sup>P</sup>nm,ε,z) ε=0  <sup>≤</sup> <sup>γ</sup> (1) <sup>m</sup> (θ) and which satisfies:

$$\sup_{\boldsymbol{\theta} \in \Theta} \pi(\boldsymbol{\theta}) \gamma_m^{(1)}(\boldsymbol{\theta}) < \infty, \quad \text{and} \quad \int \pi(\boldsymbol{\theta}) \gamma_m^{(1)}(\boldsymbol{\theta}) d\boldsymbol{\theta} < \infty.$$

Substituting this into the above, we have that

$$\sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} \ell_s^{(1)}(\boldsymbol{\theta}; \mathbb{P}_{n, \varepsilon, z}) \right|_{\varepsilon=0} \leq \beta \sum_{m=1}^M n_m \left( \gamma_m^{(1)}(\boldsymbol{\theta}) + \int \gamma_m^{(1)}(\boldsymbol{\theta}) q_m^{(1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) d\boldsymbol{\theta} \right)$$

Now recall that q (1) <sup>m</sup> (θ; <sup>P</sup>nm) = π(θ) exp{−βnmL (1) <sup>m</sup> (θ; <sup>P</sup>nm)}/Z<sup>m</sup> (1)(<sup>P</sup>nm), and per the assumption we have that the loss is lower bounded and that 0 < Z<sup>m</sup> (1)(<sup>P</sup>nm) <sup>&</sup>lt; <sup>∞</sup>, therefore <sup>q</sup> (1) <sup>m</sup> (θ; <sup>P</sup>nm) ≤ π(θ) exp{−βn<sup>m</sup> infθ∈<sup>Θ</sup> L (1) <sup>m</sup> (θ; <sup>P</sup>nm)}/Z<sup>m</sup> (1)(<sup>P</sup>nm) ≤ C (1) <sup>m</sup> π(θ) so that,

$$\sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} \ell_s^{(1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \right|_{\varepsilon=0} \leq \beta \sum_{m=1}^M n_m \left( \gamma_m^{(1)}(\boldsymbol{\theta}) + C_m^{(1)} \int \gamma_m^{(1)}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} \right) =: \gamma_s^{(1)}(\boldsymbol{\theta})$$

We now verify that the conditions hold. For condition 2, we have

$$\sup_{\boldsymbol{\theta} \in \Theta} \pi(\boldsymbol{\theta}) \gamma_s^{(1)}(\boldsymbol{\theta}) \leq \beta \sum_{m=1}^M n_m \left( \left( \sup_{\boldsymbol{\theta} \in \Theta} \pi(\boldsymbol{\theta}) \gamma_m^{(1)}(\boldsymbol{\theta}) \right) + \left( \sup_{\boldsymbol{\theta} \in \Theta} \pi(\boldsymbol{\theta}) \right) C_m^{(1)} \int \gamma_m^{(1)}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} \right) < \infty$$

which follows by the assumptions on the robustness of the loss and that the prior is upper bounded, as well as the finiteness of β, nm, and C (1) <sup>m</sup> .

Condition 3 follows similar reasoning.

$$\begin{aligned} \int \gamma_s^{(1)}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} &= \int \beta \sum_{m=1}^M n_m \left( \gamma_m^{(1)}(\boldsymbol{\theta}) + C_m^{(1)} \int \gamma_m^{(1)}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} \right) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} \\ &= \beta \sum_{m=1}^M n_m \left( \int \pi(\boldsymbol{\theta}) \gamma_m^{(1)}(\boldsymbol{\theta}) d\boldsymbol{\theta} + \int \pi(\boldsymbol{\theta}) C_m^{(1)} \left( \int \gamma_m^{(1)}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} \right) d\boldsymbol{\theta} \right) \\ &= \beta \sum_{m=1}^M n_m \left( \int \pi(\boldsymbol{\theta}) \gamma_m^{(1)}(\boldsymbol{\theta}) d\boldsymbol{\theta} + C_m^{(1)} \int \gamma_m^{(1)}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} \right) < \infty \end{aligned}$$

Since the loss is robust, the integrals are finite, and since all other terms are finite, we conclude that condition 3 is also satisfied. Therefore, for t = 1 the posterior computed at the server satisfies the conditions of Lemma [B.4](#page-26-1) and is therefore globally bias–robust. It remains to be shown that this holds for all t ∈ <sup>N</sup> such that t ≤ T, i.e. is finite.

We now show by induction that if the posterior at the server is robust for t = k, then it will also be robust for t = k + 1.

$$\begin{aligned} \frac{d}{d\varepsilon} \ell_s^{(k+1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) &= \beta \sum_{m=1}^M n_m \frac{d}{d\varepsilon} L_m^{(k+1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z}) \Big|_{\varepsilon=0} + \sum_{m=1}^M \sum_{t=1}^{k+1} \frac{d}{d\varepsilon} \log Z_m^{(t)}(\mathbb{P}_{n,\varepsilon,z}) \Big|_{\varepsilon=0} \\ &= \beta \sum_{m=1}^M n_m \frac{d}{d\varepsilon} L_m^{(k+1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z}) \Big|_{\varepsilon=0} + \sum_{m=1}^M \sum_{t=1}^{k+1} \underbrace{\frac{d}{d\varepsilon} Z_m^{(t)}(\mathbb{P}_{n,\varepsilon,z}) \Big|_{\varepsilon=0}}_{(1)} \\ &= \beta \sum_{m=1}^M n_m \frac{d}{d\varepsilon} L_m^{(k+1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z}) \Big|_{\varepsilon=0} + \sum_{m=1}^M \sum_{t=1}^{k+1} \underbrace{Z_m^{(t)}(\mathbb{P}_{n,\varepsilon,z}) \Big|_{\varepsilon=0}}_{(1)} \end{aligned}$$

To show the boundedness of this, we need to consider the expansion of (1) above.

$$\frac{d}{d\varepsilon} \log Z_m^{(t)}(\mathbb{P}_{n,\varepsilon,z})|_{\varepsilon=0} = \frac{\frac{d}{d\varepsilon} Z_m^{(t)}(\mathbb{P}_{n,\varepsilon,z})|_{\varepsilon=0}}{Z_m^{(t)}(\mathbb{P}_{n,\varepsilon,z})|_{\varepsilon=0}} = \frac{\int \frac{d}{d\varepsilon} \frac{q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \exp\{-\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})\}}{\exp\{-\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})\}}|_{\varepsilon=0} d\boldsymbol{\theta}}{Z_m^{(t)}(\mathbb{P}_n)}$$

Now we consider the integral in the numerator. Using the chain rule when differentiating under the integral sign:

$$\int \frac{d}{d\varepsilon} \frac{q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \exp\{-\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})\}}{\exp\{-\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})\}} \Big|_{\varepsilon=0} d\boldsymbol{\theta}$$

$$\begin{aligned} &= \int \left[ \frac{\exp\{-\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})\}}{\exp\{-\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})\}} \frac{d}{d\varepsilon} q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \Big|_{\varepsilon=0} \right. \\ &\quad \left. + \frac{q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_n) \exp\{-\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})\}}{\exp\{-\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})\}} \frac{d}{d\varepsilon} (-\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})) \Big|_{\varepsilon=0} \right. \\ &\quad \left. - \frac{q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_n) \exp\{-\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})\}}{\exp\{-\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})\}} \frac{d}{d\varepsilon} (-\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})) \Big|_{\varepsilon=0} \right] d\boldsymbol{\theta} \end{aligned}$$

Bringing the denominator back, and recalling the definition of q (t) <sup>m</sup> (θ; <sup>P</sup>nm), then we can simplify.

$$\begin{aligned} \frac{d}{d\varepsilon} \log Z_m^{(t)}(\mathbb{P}_{n,\varepsilon,z}) \Big|_{\varepsilon=0} &= \int \left[ \frac{\left( \frac{\exp\{-\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})\}}{\exp\{-\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})\}} \right) d}{Z_m^{(t)}(\mathbb{P}_n)} \frac{d}{d\varepsilon} q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \Big|_{\varepsilon=0} \right. \\ &\quad \left. - \frac{\frac{q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_n) \exp\{-\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})\}}{\exp\{-\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})\}} \frac{d}{Z_m^{(t)}(\mathbb{P}_n)} \frac{d}{d\varepsilon} (\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})) \Big|_{\varepsilon=0}}{= q_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})} \right. \\ &\quad \left. + \frac{\frac{q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_n) \exp\{-\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})\}}{\exp\{-\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})\}} \frac{d}{Z_m^{(t)}(\mathbb{P}_n)} \frac{d}{d\varepsilon} (\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})) \Big|_{\varepsilon=0}}{= q_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})} \right] d\boldsymbol{\theta} \\ &= \int \left[ \left( \frac{\exp\{-\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})\}}{Z_m^{(t)}(\mathbb{P}_n) \exp\{-\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})\}} \frac{d}{d\varepsilon} q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \Big|_{\varepsilon=0} \right. \right. \\ &\quad \left. - q_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) \frac{d}{d\varepsilon} (\beta n_m L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})) \Big|_{\varepsilon=0} \right. \\ &\quad \left. + q_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) \frac{d}{d\varepsilon} (\beta n_m L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z})) \Big|_{\varepsilon=0} \right] d\boldsymbol{\theta} \end{aligned}$$

Consider now the derivative of the previous server posterior with respect to ε evaluated at 0, which we can write as:

$$\begin{aligned} \frac{d}{d\varepsilon} q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \Big|_{\varepsilon=0} &= \frac{d}{d\varepsilon} \frac{\pi(\boldsymbol{\theta}) \exp\{-\ell_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})\}}{Z_s^{(t-1)}(\mathbb{P}_{n,\varepsilon,z})} \Big|_{\varepsilon=0} \\ &= \pi(\boldsymbol{\theta}) \left( \frac{\exp\{-\ell_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_n)\}}{Z_s^{(t-1)}(\mathbb{P}_n)} \frac{d}{d\varepsilon} (-\ell_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})) \Big|_{\varepsilon=0} \right. \\ &\quad \left. - \frac{\exp\{-\ell_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_n)\}}{(Z_s^{(t-1)}(\mathbb{P}_n))^2} \int \pi(\boldsymbol{\theta}) \frac{d}{d\varepsilon} \exp\{-\ell_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z})\} \Big|_{\varepsilon=0} d\boldsymbol{\theta} \right) \\ &= -q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_n) \left( \frac{d}{d\varepsilon} \ell_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \Big|_{\varepsilon=0} - \int q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_n) \frac{d}{d\varepsilon} \ell_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \Big|_{\varepsilon=0} d\boldsymbol{\theta} \right) \end{aligned}$$

where we have used the definition of q (t−1) <sup>s</sup> (θ; <sup>P</sup>n) by distributing the common terms outside the brackets and for the second term, since the normalising constant does not depend on θ, we can take one of them inside the integral. Furthermore, using the fact that

$$\frac{q_m^{(t-1)}(\theta; \mathbb{P}_n) \exp\{-\beta n_m L_m^{(t)}(\theta; \mathbb{P}_{n_m})\}}{\exp\{-\beta n_m L_m^{(t-1)}(\theta; \mathbb{P}_{n_m})\}} = q_m^{(t)}(\theta; \mathbb{P}_{n_m})$$

then substituting the result for <sup>d</sup> dε q (t−1) <sup>s</sup> (θ; <sup>P</sup>n,ε,z) <sup>ε</sup>=0 into <sup>d</sup> dε log Z (t) <sup>m</sup> (<sup>P</sup>n,ε,z) <sup>ε</sup>=0, we get that:

$$\begin{aligned} \frac{d}{d\varepsilon} \log Z_m^{(t)}(\mathbb{P}_{n,\varepsilon,z})|_{\varepsilon=0} &= \int q_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n,m}) \left( -\frac{d}{d\varepsilon} \ell_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \Big|_{\varepsilon=0} + \int q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_n) \frac{d}{d\varepsilon} \ell_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \Big|_{\varepsilon=0} d\boldsymbol{\theta} \right. \\ &\quad \left. - \beta n_m \frac{d}{d\varepsilon} L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n,m,\varepsilon,z}) \Big|_{\varepsilon=0} + \beta n_m \frac{d}{d\varepsilon} L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,m,\varepsilon,z}) \Big|_{\varepsilon=0} \right) d\boldsymbol{\theta} \end{aligned}$$

Substituting this expression back into the original equation for <sup>d</sup> dε ℓ (t) <sup>s</sup> (θ; <sup>P</sup>n,ε,z)|ε=0, taking the supremum over z ∈ Ξ of the absolute value of this, and applying Minkowski's inequality, results in the following upper bound.

$$\begin{aligned} & \sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} \ell_s^{(k+1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \right| \\ & \leq \sum_{m=1}^M \beta n_m \sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} L_m^{(k+1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z}) \right|_{\varepsilon=0} + \sum_{m=1}^M \sum_{t=1}^{k+1} \left\{ \int q_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) \left[ \sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} \ell_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \right|_{\varepsilon=0} \right] \right. \\ & \quad \left. + \left( \int_{\Theta} \sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} \ell_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n,\varepsilon,z}) \right|_{\varepsilon=0} \right| q_s^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_n) d\boldsymbol{\theta} \right) + \beta n_m \sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} L_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z}) \right|_{\varepsilon=0} \right| \\ & \quad \left. + \beta n_m \sup_{z \in \Xi} \left| \frac{d}{d\varepsilon} L_m^{(t-1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m,\varepsilon,z}) \right|_{\varepsilon=0} \right] \right| d\boldsymbol{\theta} \Big\} \end{aligned}$$

By the inductive assumption ∀t ∈ [k+1], ∃γ (t−1) <sup>s</sup> (θ) such that supz∈<sup>Ξ</sup> d dε ℓ (t−1) <sup>s</sup> (θ; <sup>P</sup>n,ε,z)|ε=0  ≤ γ (t−1) <sup>s</sup> (θ). Additionally, as, by assumption, the loss is lower bounded and robust ∃γ (t) <sup>m</sup> (θ) ∀<sup>t</sup> ∈ [<sup>k</sup> + 1] such that supz∈<sup>Ξ</sup> L (t) <sup>m</sup> (θ; <sup>P</sup>nm,ε,z)|ε=0  ≤ γ (t) <sup>m</sup> (θ). Furthermore, these functions satisfy the conditions of Lemma [B.4.](#page-26-1) Note also that q (t−1) <sup>s</sup> (θ; <sup>P</sup>n) ≤ C (t−1) <sup>s</sup> π(θ), since the normalising constant of this distribution is finite and the loss is lower bounded per the inductive assumption, so we get q (t−1) <sup>s</sup> (θ; <sup>P</sup>n) ≤ π(θ) exp{− infθ∈<sup>Θ</sup> ℓ (t−1) <sup>s</sup> (θ; <sup>P</sup>n)}/Z(t−1) <sup>s</sup> (<sup>P</sup>n) ≤ C (t−1) <sup>s</sup> π(θ), as seen in similar arguments before. Utilising this, we conclude:

$$\begin{aligned} &\leq \sum_{m=1}^M \beta n_m \gamma_m^{(k+1)}(\boldsymbol{\theta}) + \sum_{m=1}^M \sum_{t=1}^{k+1} \int q_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) \left\{ \gamma_s^{(t-1)}(\boldsymbol{\theta}) + \left( \int \gamma_s^{(t-1)}(\boldsymbol{\theta}) C_s^{(t-1)} \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} \right) \right. \\ &\quad \left. + \beta n_m \gamma_m^{(t)}(\boldsymbol{\theta}) + \beta n_m \gamma_m^{(t-1)}(\boldsymbol{\theta}) \right\} d\boldsymbol{\theta} := \gamma_s^{(k+1)}(\boldsymbol{\theta}) \end{aligned}$$

We now need to show that this satisfies conditions (2) and (3) of Lemma [B.4.](#page-26-1) Let's recall what these conditions state:

$$(2) = \sup_{\boldsymbol{\theta} \in \Theta} \pi(\boldsymbol{\theta}) \gamma_s^{(k+1)}(\boldsymbol{\theta}) < \infty \quad (11)$$

$$(3) = \int \pi(\boldsymbol{\theta}) \gamma_s^{(k+1)}(\boldsymbol{\theta}) d\boldsymbol{\theta} < \infty \quad (12)$$

We first verify that condition (2) holds.

$$\begin{aligned} \sup_{\boldsymbol{\theta} \in \Theta} \pi(\boldsymbol{\theta}) \gamma_s^{(k+1)}(\boldsymbol{\theta}) &= \sup_{\boldsymbol{\theta} \in \Theta} \pi(\boldsymbol{\theta}) \left\{ \sum_{m=1}^M \beta n_m \gamma_m^{(k+1)}(\boldsymbol{\theta}) \right. \\ &\quad \left. + \sum_{m=1}^M \sum_{t=1}^{k+1} \int q_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) \left\{ \gamma_s^{(t-1)}(\boldsymbol{\theta}) + C_s^{(t-1)} \int \gamma_s^{(t-1)}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} + \beta n_m \gamma_m^{(t)}(\boldsymbol{\theta}) + \beta n_m \gamma_m^{(t-1)}(\boldsymbol{\theta}) \right\} d\boldsymbol{\theta} \right\} \\ &\leq \beta n_m \sum_{m=1}^M \sup_{\boldsymbol{\theta} \in \Theta} \pi(\boldsymbol{\theta}) \gamma_m^{(k+1)}(\boldsymbol{\theta}) + \sup_{\boldsymbol{\theta} \in \Theta} \pi(\boldsymbol{\theta}) \left\{ \sum_{m=1}^M \sum_{t=1}^{k+1} \int q_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) \left[ \gamma_s^{(t)}(\boldsymbol{\theta}) \right. \right. \\ &\quad \left. \left. + C_s^{(t-1)} \int \gamma_s^{(t-1)}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} + \beta n_m \gamma_m^{(t)}(\boldsymbol{\theta}) + \beta n_m \gamma_m^{(t-1)}(\boldsymbol{\theta}) \right] d\boldsymbol{\theta} \right\} < \infty \end{aligned}$$

Since β, nm, and M are finite, and any finite linear combination of finite terms is finite, we can easily see that the first part is finite. This follows since γ (k+1) <sup>m</sup> (θ) satisfies condition (2) of Lemma [B.4.](#page-26-1) Furthermore, since π(θ) is upper bounded, we now need to verify whether the inside of the curly brackets is finite. Since this is a finite sum, we need to verify if ∀m ∈ [M] and ∀t ∈ [k + 1], the following holds:

$$\int q_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) \left[ \gamma_s^{(t)}(\boldsymbol{\theta}) + C_s^{(t-1)} \int \gamma_s^{(t-1)}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} + \beta n_m \gamma_m^{(t)}(\boldsymbol{\theta}) + \beta n_m \gamma_m^{(t-1)}(\boldsymbol{\theta}) \right] d\boldsymbol{\theta} < \infty$$

By the inductive step, this is true ∀t ∈ [k], so we need to show that it also holds for t = k + 1. So,

$$\int q_m^{(k+1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) \left[ \gamma_s^{(k+1)}(\boldsymbol{\theta}) + C_s^{(k)} \int \gamma_s^{(k)}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} + \beta n_m \gamma_m^{(k+1)}(\boldsymbol{\theta}) + \beta n_m \gamma_m^{(k)}(\boldsymbol{\theta}) \right] d\boldsymbol{\theta} < \infty$$

Note that q (k+1) <sup>m</sup> (θ; <sup>P</sup>nm) is equal to <sup>π</sup>(θ) exp{−βnm<sup>L</sup> (k+1) <sup>m</sup> (θ; <sup>P</sup>nm)} exp{−β P <sup>i</sup≯=<sup>m</sup> niL (k) i (θ; <sup>P</sup>n<sup>i</sup> )}/Z(k+1) <sup>m</sup> Z (k) <sup>s</sup> , and since the normalising constants are finite and positive, and the losses are lower bounded, then we can write

$$\begin{aligned} q_m^{(k+1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) &\leq \pi(\boldsymbol{\theta}) \exp\{-\beta n_m \inf_{\boldsymbol{\theta} \in \Theta} L_m^{(k+1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m})\} \exp\{-\beta \sum_{i \neq m} n_i \inf_{\boldsymbol{\theta} \in \Theta} L_i^{(k)}(\boldsymbol{\theta}; \mathbb{P}_{n_i})\} / Z_m^{(k+1)} Z_s^{(k)} \\ &\leq C_m^{(k+1)} \pi(\boldsymbol{\theta}) \end{aligned}$$

where <sup>0</sup> < C(k+1) <sup>m</sup> < ∞. Thereby, we have

$$\begin{aligned} & \int q_m^{(k+1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) \left[ \gamma_s^{(k+1)}(\boldsymbol{\theta}) + C_s^{(k)} \int \gamma_s^{(k)}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} + \beta n_m \gamma_m^{(k+1)}(\boldsymbol{\theta}) + \beta n_m \gamma_m^{(k)}(\boldsymbol{\theta}) \right] d\boldsymbol{\theta} \\ & \leq C_m^{(k+1)} \int \pi(\boldsymbol{\theta}) \gamma_s^{(k+1)}(\boldsymbol{\theta}) d\boldsymbol{\theta} + C_m^{(k+1)} C_s^{(k)} \left( \int \gamma_s^{(k)}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} \right) \left( \int \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} \right) \\ & \quad + \beta n_m C_m^{(k+1)} \int \pi(\boldsymbol{\theta}) \gamma_m^{(k+1)}(\boldsymbol{\theta}) d\boldsymbol{\theta} + \beta n_m C_m^{(k+1)} \int \pi(\boldsymbol{\theta}) \gamma_m^{(k)}(\boldsymbol{\theta}) d\boldsymbol{\theta} < \infty \end{aligned}$$

This expression is finite since the individual integrals must be finite by the definition of the bounding functions γ, as these need to satisfy condition (3) of Lemma [B.4](#page-26-1) with the prior π(θ). Hence, we have shown that condition (2) holds for γ (k+1) <sup>s</sup> (θ) and Equation [\(11\)](#page-31-0) is indeed finite.

It remains to be shown that condition (3), Equation [\(12\)](#page-31-1), also holds. Using the same expression for γ (k+1) <sup>s</sup> (θ) as before, we have:

$$\begin{aligned} \int \pi(\boldsymbol{\theta}) \gamma_s^{(k+1)}(\boldsymbol{\theta}) d\boldsymbol{\theta} &= \int \pi(\boldsymbol{\theta}) \left\{ \sum_{m=1}^M \beta n_m \gamma_m^{(k+1)}(\boldsymbol{\theta}) \right. \\ &\quad \left. + \sum_{m=1}^M \sum_{t=1}^{k+1} \int q_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) \left\{ \gamma_s^{(t-1)}(\boldsymbol{\theta}) + C_s^{(t-1)} \int \gamma_s^{(t-1)}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} + \beta n_m \gamma_m^{(t)}(\boldsymbol{\theta}) + \beta n_m \gamma_m^{(t-1)}(\boldsymbol{\theta}) \right\} d\boldsymbol{\theta} \right\} d\boldsymbol{\theta} \end{aligned}$$

Since, the summations are finite, we can exchange the integrals and sums to get

$$\begin{aligned} &= \left( \sum_{m=1}^M \beta n_m \underbrace{\int \pi(\boldsymbol{\theta}) \gamma_m^{(k+1)}(\boldsymbol{\theta}) d\boldsymbol{\theta}}_{<\infty \ \forall m \in [M]} \right) + \underbrace{\int \pi(\boldsymbol{\theta}) d\boldsymbol{\theta}}_{=1} \left( \sum_{m=1}^M \sum_{t=1}^{k+1} \int q_m^{(t)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) \left\{ \gamma_s^{(t-1)}(\boldsymbol{\theta}) + C_s^{(t-1)} \int \gamma_s^{(t-1)}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} \right. \right. \\ &\quad \left. \left. + \beta n_m \gamma_m^{(t)}(\boldsymbol{\theta}) + \beta n_m \gamma_m^{(t-1)}(\boldsymbol{\theta}) \right\} d\boldsymbol{\theta} \right) \end{aligned}$$

where the first part is finite since for each γ (k+1) <sup>m</sup> (θ), we have by definition that this expression is finite as it needs to satisfy condition (3). Therefore, we need to show that the summation is finite. By the inductive step, this is true ∀t ∈ [k], and we will now show that ∀m ∈ [M] it also is finite for t = k + 1.

$$\int q_m^{(k+1)}(\boldsymbol{\theta}; \mathbb{P}_{n_m}) \left\{ \gamma_s^{(k)}(\boldsymbol{\theta}) + C_s^{(k)} \int \gamma_s^{(k)}(\boldsymbol{\theta}) \pi(\boldsymbol{\theta}) d\boldsymbol{\theta} + \beta n_m \gamma_m^{(k+1)}(\boldsymbol{\theta}) + \beta n_m \gamma_m^{(k)}(\boldsymbol{\theta}) \right\} d\boldsymbol{\theta}$$

Recall from before that q (k+1) <sup>m</sup> (θ; <sup>P</sup>nm) ≤ C (k+1) <sup>m</sup> π(θ) and hence, it is now immediate to see that by the same argument as in the proof of condition (2), this integral is finite. Therefore, condition (3) of Lemma [B.4](#page-26-1) also holds and Equation [\(12\)](#page-31-1) is true.

We conclude that all conditions of Lemma [B.4](#page-26-1) are satisfied.

Therefore, by induction, as long as we have a robust loss function (in the sense of [Ghosh & Basu,](#page-10-18) [2016a;](#page-10-18) [Matsubara et al.,](#page-11-11) [2022\)](#page-11-11) at the clients, then irregardless of the current iteration by using the weighted KL divergence at the clients and the KL divergence at the server, FEDGVI achieves global bias robustness to outliers.

Note that when assuming that q (k+1) <sup>m</sup> (θ; <sup>P</sup>nm) ≤ C (k+1) <sup>m</sup> π(θ), or similarly at the server in the uncontaminated case, we have used that the normalising constants in the well specified case are finite. This is necessary to hold, since otherwise we will not have valid distributions, and furthermore we can always choose a prior distribution that is bounded above so this will always be finite. However, this finiteness is not assumed for the normalising constants that are contaminated by the outliers, so the proof is needed to show boundedness of the posterior influence under contamination.

# C. Additional Details on FEDGVI

We present some additional details on FEDGVI in order to aid clarity and contextualise it in the broader literature.

## C.1. A Note on the Learning Rate Parameter in GBI

The β parameter comes from the power/cold/tempered posteriors of e.g. [Grunwald](#page-10-16) ¨ [\(2012\)](#page-10-16), where the likelihood in Bayesian posteriors is raised to some power of β > 0. This was originally done to add some robustness to the posterior, down– weighting observations if β < 1 and up weighting these for β > 1. Through a known result [\(Knoblauch et al.,](#page-11-6) [2022\)](#page-11-6) which we highlight in Lemma [B.1](#page-15-1) in the Appendix, this is equivalent to having a weighted Kullback–Leibler divergence, <sup>1</sup> <sup>β</sup> KL. This also allows us to define if we want to trust the prior more β < 1 or less β > 1, since up weighting the data means down weighting the prior and vice versa.

# C.2. A Note on Definition [4.11](#page-5-3)

The three conditions combined allow us to say whether the client posterior (or simply the posterior in a global, 1 Client, GBI setting) derived from such a robust loss is provably robust to Huber contamination.

From Condition 1 we are able to bound an infinitesimal change in the loss with the contaminating data point z by some auxiliary function γ, possibly infinite for some values of θ.

Condition 2 states that the product function, γ(θ)π(θ) has finite uniform norm. This ensures that this product under the worst case contamination and the worst parameter θ, is finite and hence it cannot be made arbitrarily bad, which does not hold for the negative log likelihood in general. Alternatively, the prior decays to zero faster than the auxiliary function can diverge to infinity in θ.

Condition 3 further says that γ(θ)π(θ) is finitely integrable, i.e. that this is in L 1 (Θ, µ). This, in effect bounds the normalising constant of the contaminated posterior and will ensure that this is finite.

Taking all these conditions together tells us that the product function π(θ)γ(θ) is in L 1 (Θ, µ)∩L<sup>∞</sup>(Θ, µ), and that it is in fact finite everywhere. These two conditions that are mutually independent so both Condition 2 and 3 need to hold. Equivalently, we require that γ(θ) is bounded and integrable with respect to the prior probability measure π(θ)µ(dθ) =: Π(dθ).

These conditions characterise the notion of robustness we use for Theorem [4.12,](#page-5-0) with derivation in Appendix [B.7,](#page-26-0) by considering the worst choice for the contamination z and the parameter θ with respect to small perturbations of the resulting posterior through ε. The influence of the contamination z and parameter θ on the posterior is defined as d dε q (t) <sup>m</sup> (θ; <sup>P</sup>nm,ε,z)|ε=0, which is bounded through the conditions. Our result then implies that the posterior is *'globally bias robust'*, i.e. robust to Huber contamination.

# C.3. FEDGVI in the Context of GVI and FL

When viewing GVI/GBI as an optimisation problem on the space of probability distributions P(Θ), Bayesian inference, VI, hierarchical Bayes/VI, all target a single element of this space. These methods either target the standard Bayesian posterior

explicitly, or the posterior within some variational family with closest Kullback–Leibler distance to the Bayesian one [\(Blei](#page-10-23) [et al.,](#page-10-23) [2017;](#page-10-23) [Walker,](#page-12-14) [2013\)](#page-12-14). Through GBI and GVI we are able to target different elements of a subspace of P(Θ), then simply a single point; in that regard, these approaches 'generalise' Bayes. In this paper, 'generalised' is inherited from GVI and GBI. We should note that in the FEDGVI setting, GBI and GVI allow us to generalise PVI or FEDAVG to a broader subspace of possible posteriors. Figure [8](#page-34-1) displays FEDGVI in regards to the related GVI and GBI literature as well as the FL literature as in Figure [1.](#page-4-5)

![](_page_34_Diagram_2.jpeg)

Figure 8: We illustrate the relationship of FEDGVI—characterised by the loss L, the client divergence D, the variational family Q, the number of clients M, and the divergence at the server Ds—to Generalised Variational/Bayesian Inference (GVI/GBI), Partitioned Variational Inference (PVI), Variational Inference (VI), Federated Averaging (FEDAVG), Empirical Risk Minimisation (ERM), and Bayes.

# D. Additional Details on Experiments

For reproducibility we give additional details on the experiments that we have carried out to empirically support our contributions. Code to reproduce these can be found at:

<https://github.com/Terje-M/FedGVI>.

# D.1. Normal–Location Model

We assume the following well known model for the Data Generating Process and prior, with some unspecified prior mean µπ, in order to allow for prior misspecification:

$$\theta \sim \mathcal{N}(\mu_\pi, 1^2) := \pi(\theta)$$

$$x_{1:N} | \theta \stackrel{\text{iid}}{\sim} \mathcal{N}(\theta, 1^2) := p(x_i | \theta).$$

The true Data Generating Process under model misspecification through Huber contamination is given by:

$$\theta \sim \mathcal{N}(0, 0.5^2)$$

$$x_{1:N} | \boldsymbol{\theta} \stackrel{\text{iid}}{\sim} (1 - \varepsilon) \mathcal{N}((\boldsymbol{\theta} - 2), 1^2) + \varepsilon \mathcal{N}((\boldsymbol{\theta} + 3), 0.5^2)$$

where the second term represents some ε noise fraction that is added to the data. Our aim is to find the location of the first term in the above model, while modelling out the noise from the second term.

We consider PVI where the client optimisation is given by Pm(− log p(·|θ), KL, N ), and the server optimisation step by Ps(ℓ (·) <sup>s</sup> (θ), KL, N ). Under the assumption of likelihood misspecification, we consider the following divergences and losses at the clients, while leaving the server optimisation step unchanged: The weighted Kullback–Leibler divergence <sup>1</sup> <sup>w</sup> <sup>D</sup>KL, the Alpha–Renyi divergence ´ D (α) AR, the Fisher–Rao divergence <sup>D</sup>F R, the score matching losses <sup>L</sup> (w) SM, the beta–divergence based loss L (β) <sup>B</sup> , and the gamma–divergence based loss <sup>L</sup> (γ) <sup>G</sup> . Expect for <sup>P</sup>m(<sup>L</sup> (w) SM, <sup>w</sup> <sup>D</sup>KL, <sup>N</sup> ), which allows for conjugate updates by Proposition [4.10,](#page-5-2) we have to resort to optimisation. This however does not require Monte–Carlo sampling since the divergence terms and the losses have closed forms under Gaussian distributions, see [Knoblauch et al.](#page-11-6) [\(2022\)](#page-11-6) for the remaining losses, [Pardo Llorente](#page-12-22) [\(2006\)](#page-12-22) for the KL and Alpha–Renyi divergences and ´ [Nielsen](#page-12-18) [\(2023\)](#page-12-18) for the Fisher–Rao divergence. For the optimisation, we use the Adam optimiser with a learning rate of 0.001, leaving all other parameters at their default values.

Explicit Losses and Divergences used As mentioned in Section [3.1,](#page-2-4) we employ a range of different loss functions and divergences throughout the experiments. The main one being the robust generalised cross entropy used in the real world experiments. For the synthetics, for instance in Figure [3](#page-6-2) we compare four different losses with two different implementations for the Score–Matching loss.

For this example, where we only have one sequence of data points x1:<sup>N</sup> which are assumed to be independent, the losses are:

- 1. The Negative Log Likelihood:

$$\mathcal{L}_{NLL}(x_i, p_{\boldsymbol{\theta}}) = -\log p_{\boldsymbol{\theta}}(x_i)$$

- 2. The Density–Power Divergence based loss [\(Ghosh & Basu,](#page-10-18) [2016a;](#page-10-18)[b\)](#page-10-21):

$$\mathcal{L}_B^{(\beta)}(x_i, p_{\boldsymbol{\theta}}) = -\frac{1}{\beta} p_{\boldsymbol{\theta}}(x_i)^{\beta} + \frac{1}{1+\beta} \int_{\Xi} p_{\boldsymbol{\theta}}(x)^{\beta+1} \mu(dx)$$

- 3. The Gamma divergence based loss [\(Hung et al.,](#page-11-15) [2018\)](#page-11-15):

$$\mathcal{L}_G^{(\gamma)}(x_i, p_{\boldsymbol{\theta}}) = -\frac{1}{(\gamma - 1)} p_{\boldsymbol{\theta}}(x_i)^{\gamma-1} \cdot \frac{\gamma}{(\int_{\Xi} p_{\boldsymbol{\theta}}(x)^{\gamma} \mu(dx))^{\frac{\gamma-1}{\gamma}}}$$

- 4. The weighted Score Matching Loss [\(Altamirano et al.,](#page-9-11) [2023\)](#page-9-11):

$$\mathcal{L}_{SM}^{(w_m^{(t)})}(x_i, p_{\boldsymbol{\theta}}) = \|w_m^{(t)}(x_i)^\top \nabla_x \log p_{\boldsymbol{\theta}}(x_i)\|_2^2 + 2\nabla \cdot (w_m^{(t)}(x_i)w_m^{(t)}(x_i)^\top \nabla_x \log p_{\boldsymbol{\theta}}(x_i))$$

We use two different weight functions w (t) <sup>m</sup> , where µ (t) \<sup>m</sup> is the mean of the cavity distribution:

- (a) The Squared Exponential Kernel (SE):

$$w_m^{(t)}(x_i) = \beta \exp \left\{ -\frac{(x_i - \mu_{\sqrt{m}}^{(t)})^2}{2c^2} \right\}$$

- (b) The Inverse Multi-Quadratic Kernel (IMQ):

$$w_m^{(t)}(x_i) = \beta \left( 1 + \frac{(x_i - \mu_{\sqrt{m}}^{(t)})^2}{2ac^2} \right)^{-a}$$

All the above losses have closed form objectives under the expectation with respect to the approximating distribution q(θ) and the assumed Gaussian likelihood. Furthermore, the negative log likelihood and the score matching loss admit conjugate updates under the KL divergence.

We also use different divergences, mainly:

- 1. The Kullback–Leibler divergence [\(Kullback & Leibler,](#page-11-17) [1951\)](#page-11-17):

$$D_{KL}(q : \pi) = \int_{\Theta} q(\boldsymbol{\theta}) \log \frac{q(\boldsymbol{\theta})}{\pi(\boldsymbol{\theta})} \mu(d\boldsymbol{\theta})$$

- 2. The Reverse KL divergence:

$$D_{RKL}(q : \pi) = D_{KL}(\pi : q)$$

- 3. The Alpha–Renyi divergence: ´

$$D_{AR}^{(\alpha)}(q : \pi) = \frac{1}{\alpha(1-\alpha)} \log \int_{\Theta} q(\boldsymbol{\theta})^{\alpha} \pi(\boldsymbol{\theta})^{1-\alpha} \mu(d\boldsymbol{\theta})$$

- 4. Weighted Divergences of the form:

$$\frac{1}{\beta} D(q : \pi)$$

For Gaussian distributions, these have closed form solutions.

# D.1.1. INFLUENCE FUNCTIONS

For the influence functions experiment in Figure [3,](#page-6-2) we still assume the same likelihood function, but we have a different data generating process. We generate 99 data points from the following student-t distribution with 4 degrees of freedom, mean 0 and scale 1,:

$$\begin{aligned} x_{1:99} &\sim \text{Student } T(0, 1, 4) \\ x_{100} &\sim \delta_y(x), y \in \mathbb{R} \end{aligned}$$

We place Huber contamination on the hypothesis, where we add an additional observation to one of seven clients that is increasingly farther from the true mean, and calculate the posteriors with this outlier, y. We have used the losses described previously for the posteriors and the Kullback–Leibler divergence, running all experiments to convergence. We compare the resulting distributions using the Fisher–Rao divergence [\(Nielsen,](#page-12-18) [2023\)](#page-12-18), which has closed form between two univariate Gaussians q(θ) ∼ N (µq, σ<sup>2</sup> q ) and π(θ) ∼ N (µπ, σ<sup>2</sup> π )

$$D_{FR}(q : \pi) = \sqrt{2} \log \left( \frac{1 + \Delta(\mu_q, \sigma_q : \mu_\pi, \sigma_\pi)}{1 - \Delta(\mu_q, \sigma_q : \mu_\pi, \sigma_\pi)} \right)$$

$$\Delta(a, b : c, d) := \sqrt{\frac{(c-a)^2 + (d-b)^2}{(c-a)^2 - (d+b)^2}}, \quad (a, b, c, d) \in \mathbb{R}^4 \setminus \{0\}$$

# D.2. Logistic Regression with Gaussian Design

We place a mean field Gaussian distribution over the parameters of linear model θ <sup>⊤</sup>x + b by augmenting the data to x˜ = [1, x <sup>⊤</sup>] in order to allow for non–normalised data sets. We assume that the labels, y<sup>i</sup> ∈ {0, 1}, follow a Bernoulli distribution with sigmoid probabilities:

$$\mathbf{y}_i \sim \text{Ber}(\sigma(\boldsymbol{\theta}^\top \tilde{\mathbf{x}}_i))$$

where σ(a) = (1 + e −a ) −1 is the sigmoid function. This allows us to define the likelihood as follows:

$$p(\mathbf{y}_i | \boldsymbol{\theta}, \mathbf{x}_i) = \exp\{\mathbf{y}_i \tilde{\mathbf{x}}_i^\top \boldsymbol{\theta} - \psi(\tilde{\mathbf{x}}_i^\top \boldsymbol{\theta})\}$$

where ψ(a) := log(1 + e a ), which gives rise to the sigmoid through σ(a) = ψ ′ (a) [\(Katsevich & Rigollet,](#page-11-18) [2024\)](#page-11-18). We use this exponential family form above since taking the logarithm for the negative log–likelihood is easily achieved by removing the exponential and allows for slightly faster calculations during the optimisation. Further, we assume that the prior π(θ) = N (0, Σ), where Σ is fixed but generated through sampling from a Gamma distribution and averaging over the samples. More specifically we sampled 100 samples from a Gamma distribution with ξ1:100 iid∼ Gamma(1, <sup>1</sup>/0.01), and use their mean, ¯ξ, to define Σ := ¯ξ −1 Id. This was done to ensure fairness with the Distributed Stein Variational Gradient Descent approach of [Kassab & Simeone](#page-11-7) [\(2022\)](#page-11-7), who use an Gaussian inverse Gamma prior, which we for ease of implementation forgo (the results of the experiments show that we easily match their performance, if not surpass it slightly). For the prediction, we use an approximation to the expectation with respect to the final distribution found, q (T) <sup>s</sup> (θ) ∼ N (µs, Σs) where Σ<sup>s</sup> is a diagonal matrix, as in [Ashman et al.](#page-9-0) [\(2022\)](#page-9-0).

$$p(\mathbf{y}_{\text{new}} = 1 | \tilde{\mathbf{x}}_{\text{new}}) = \mathbb{E}_{q_s^{(T)}(\boldsymbol{\theta})} [p(\mathbf{y}_{\text{new}} = 1 | \boldsymbol{\theta}, \tilde{\mathbf{x}}_{\text{new}})] \approx \sigma \left( \frac{\mu_s^\top \tilde{\mathbf{x}}_{\text{new}}}{\sqrt{1 + \pi \tilde{\mathbf{x}}_{\text{new}}^\top \Sigma_s \tilde{\mathbf{x}}_{\text{new}}}} \right)$$

This allows us to forgo Monte Carlo sampling to evaluate this expectation.

*Remark* D.1*.* Since neither GVI, nor FEDGVI targets the Bayesian posterior under different divergences or loss functions in comparison to vanilla VI, we cannot truly speak of this expectation approximating the Bayesian posterior predictive distribution, however since our aim is to find a distribution that is more valuable to a decision maker, using a FEDGVI posterior should allow us to make more informed predictions depending on what the DM wants to model. This can be better uncertainty quantification through changing the divergence, and/or better prediction accuracy through changing the loss.

## D.2.1. FURTHER EXPERIMENTS

In Figure [9](#page-37-0) we compare the predictive performance of FEDGVI with two clients against that of GVI with only one client.

![](_page_37_Figure_7.jpeg)

Figure 9: Comparing Logistic Regression with FEDGVI where the data set is split across clients, to GVI where the entire data set is available.

## D.3. Bayesian Neural Networks

The model architecture is a fully connected multi–layer perceptron with RELU activation.

# D.3.1. MNIST (LEC[UN ET AL](#page-11-19)., [1998\)](#page-11-19) DETAILS AND ADDITIONAL EXPERIMENTS

For the hyperparameters of the competing methods in the BNNs, we follow [Hasan et al.](#page-10-14) [\(2024\)](#page-10-14) in using SGD with momentum with a learning rate of 0.1 for FEDAVG, and β–PREDBAYES, and 0.01 for FEDPA. The architecture for these is a 2 hidden layer fully connected neural network, where each hidden layer has 100 neurons.

For FEDGVI and PVI, we follow the set up of [\(Ashman et al.,](#page-9-0) [2022\)](#page-9-0) in using the ADAM optimiser [\(Kingma & Ba,](#page-11-20) [2015\)](#page-11-20) with a learning rate of 0.0005, leaving all other parameters the default values in PyTorch. Here we use a fully connected NN with 1 hidden layer of 200 neurons.

The contamination maps all contaminated data points of one class to a single other class. In both cases, we carried out mini–batch optimisation.

Since we use different architectures for the BNN experiments for the MNIST data set in Table [1,](#page-7-2) we additionally report results for BNNs when we use the same Neural Network architecture and still retain superior performance of FEDGVI, see Table [3.](#page-38-0) We notice that the choosing the implementation with the two hidden layer NN for the competing methods performs better or on an equivalent level (within one standard deviation) of each other, while FEDGVI performs better on the single layer NN.

When examining their convergence behaviour under the different architectures, we further notice that the competing methods perform worse than FEDGVI, and FEDAVG and FEDPA exhibit no stability in their accuracy in the contaminated setting, see Figures [10](#page-39-0) and [11.](#page-39-1) This phenomenon occurs even in the uncontaminated case as reported in [Al-Shedivat et al.](#page-9-5) [\(2021\)](#page-9-5), where we have chosen the optimiser and learning rates as suggested in their paper, and hence we conjecture that contamination further exacerbates this.

Table 3: Classification accuracy (highest in bold) on uncontaminated test data after training on 10% contaminated MNIST data. Here, we compare the results with a fully connected Neural Network with 1 hidden layer of 200 Neurons to the results of a fully connected Neural Network with 2 hidden layers of 100 Neurons each. We report the best performance across all server iterations.

|      |     | M F ED | ODEL A  | VG      | 10    | 1 C 94.79 ± H LIENTS 0.43 | IDDEN L 3 | AYER C 91.76 ± LIENTS 0.08 | 10 96.64 | 2 C ± H IDDEN LIENTS 0.07 | L 3 96.34 | AYERS C ± LIENTS 0.20 |
|------|-----|--------|---------|---------|-------|---------------------------|-----------|----------------------------|----------|---------------------------|-----------|-----------------------|
|      |     | F ED   | PA      |         | 94.53 | ± 0.15                    | 95.74     | ± 0.08                     | 94.25    | ± 0.39                    | 95.31     | ± 0.35                |
| β    | –P  | RED    | B       | AYES    | 94.96 | ± 0.06                    | 96.67     | ± 0.07                     | 94.90    | ± 0.08                    | 96.73     | ± 0.08                |
|      |     | PVI    |         |         | 95.56 | ± 0.18                    | 96.68     | ± 0.07                     | 95.68    | ± 0.10                    | 97.31     | ± 0.08                |
| F    | ED  | GVI    | D       | AR      | 96.36 | ± 0.09                    | 97.13     | ± 0.13                     | 95.78    | ± 0.17                    | 97.24     | ± 0.05                |
| F    | ED  | GVI    | L       | GCE     | 97.06 | ± 0.03                    | 98.04     | ± 0.07                     | 96.57    | ± 0.04                    | 97.74     | ± 0.11                |
| F ED | GVI | D      | AR      | + L GCE | 97.50 | ± 0.07                    | 98.13     | ± 0.08                     | 96.77    | ± 0.10                    | 97.79     | ± 0.10                |
| VI   |     | (1 C   | LIENT   | )       |       | (96.96                    | ± 0.17)   |                            |          | (90.87                    | ± 0.50)   |                       |
| GVI  |     | (1     | C LIENT | )       |       | ( 98.13                   | ± 0.07    | )                          |          | ( 97.56                   | ± 0.05    | )                     |

Table 4: Classification accuracy (highest in bold for each learning rate) on uncontaminated test data after training on 10% contaminated MNIST data split across 3 clients. Here, we compare the results of different initialisations of FEDGVI with the Alpha–Renyi divergence and generalised cross entropy loss achieved when optimising the posteriors with different ´ learning rates of ADAM. We report the best performance after all server iterations.

| M ODEL     |       | 1 e − 2 |       | 5 e − 3 |       | L 1 e − EARNING 3 | R     | ATE 5 e − η 4 |       | 1 e − 4 |       | 5 e − 3 |
|------------|-------|---------|-------|---------|-------|-------------------|-------|---------------|-------|---------|-------|---------|
| PVI        | 96.34 | ± 0.16  | 96.50 | ± 0.18  | 96.72 | ± 0.06            | 96.76 | ± 0.07        | 96.01 | ± 0.05  | 95.39 | ± 0.06  |
| F ED GVI D |       |         |       |         |       |                   |       |               |       |         |       |         |
| (2 5)      |       |         |       |         |       |                   |       |               |       |         |       |         |
| AR         | 96.84 | ± 0.12  | 96.91 | ± 0.02  | 97.16 | ± 0.04            | 97.18 | ± 0.03        | 96.51 | ± 0.19  | 95.65 | ± 0.03  |
| F ED GVI L |       |         |       |         |       |                   |       |               |       |         |       |         |
| (0 8)      |       |         |       |         |       |                   |       |               |       |         |       |         |
| GCE        | 98.22 | ± 0.07  | 98.30 | ± 0.03  | 98.15 | ± 0.01            | 98.08 | ± 0.08        | 97.07 | ± 0.06  | 95.84 | ± 0.04  |
| F ED GVI D |       |         |       |         |       |                   |       |               |       |         |       |         |
| (2 5)      |       |         |       |         |       |                   |       |               |       |         |       |         |
| AR + L     |       |         |       |         |       |                   |       |               |       |         |       |         |
| (0         | 8)    |         |       |         |       |                   |       |               |       |         |       |         |
| GCE        | 98.31 | ± 0.10  | 98.24 | ± 0.07  | 98.23 | ± 0.06            | 98.06 | ± 0.09        | 97.50 | ± 0.01  | 96.35 | ± 0.08  |

In Table [4](#page-38-1) we compare different learning rates of ADAM with different initialisations of FEDGVI showing that we in fact underperform with the learning rate selected for the experiments in Figures [6](#page-7-0) and [7,](#page-8-0) and Table [1](#page-7-2) for the outperforming methods of FEDGVI.

In Table [5](#page-39-2) we further investigate the stability of FEDGVI posteriors when slightly varying the robustness parameter. This shows no significant variations in the accuracy achieved by FEDGVI when slightly perturbing δ = 0.8.

We also want to highlight that by not carefully selecting the hyperparameters of FEDGVI, as well as the learning rate, and keeping these constant across the BNN experiments, we have shown that you do not require extensive knowledge to adapt existing PVI approaches to FEDGVI and outperform. For instance, FEDGVI performs even better for the robust losses at a

Table 5: We fix α = 2.5 in the Alpha–Renyi divergence, and vary ´ δ, of the generalised cross entropy loss of [Zhang &](#page-12-16) [Sabuncu](#page-12-16) [\(2018\)](#page-12-16), around 0.8. We report accuracies on uncontaminated test data after training on 10% contaminated MNIST data split across 5 clients. These accuracies vary very little demonstrating stability in the FEDGVI posterior at slight perturbations in the loss parameter.

| M ODEL     |              |              |              | ( δ ) δ OF L GCE |              |              |              |
|------------|--------------|--------------|--------------|------------------|--------------|--------------|--------------|
|            | 0.75         | 0.775        | 0.79         | 0.8              | 0.81         | 0.825        | 8.85         |
| F ED GVI D |              |              |              |                  |              |              |              |
| (2 5)      |              |              |              |                  |              |              |              |
| AR + L     |              |              |              |                  |              |              |              |
| ( δ )      |              |              |              |                  |              |              |              |
| GCE        | 98.05 ± 0.02 | 98.14 ± 0.09 | 98.04 ± 0.09 | 98.06 ± 0.06     | 98.06 ± 0.05 | 97.99 ± 0.07 | 97.98 ± 0.03 |

![](_page_39_Figure_3.jpeg)

Figure 10: Accuracy on Fully Connected Neural Networks with 1 Hidden Layer. We demonstrate convergence of the different approaches examined in the first multicolumn of Table [3.](#page-38-0) The models are trained on 10% label–contaminated data, and prediction accuracy is assessed on uncontaminated test data.

![](_page_39_Figure_5.jpeg)

Figure 11: Accuracy on Fully Connected Neural Networks with 2 Hidden Layers. We demonstrate convergence of the different approaches examined in the second multicolumn of Table [3.](#page-38-0) The models are trained on 10% label–contaminated data, and prediction accuracy is assessed on uncontaminated test data.

higher learning rate, but we have shown in Table [1](#page-7-2) that it still outperforms even when not carefully selecting a learning rate. Furthermore, choosing δ = 0.6 and α = 2.5 would have performed better when varying only the robustness parameters of FedGVI, as seen in Figure [6.](#page-7-0)

Lastly, Figure [12](#page-40-0) shows that even when the loss is not available in a conjugate, closed form way, that FEDGVI still incurs no significant computational overhead through choosing the Alpha–Renyi divergence or the generalised cross entropy loss. ´

![](_page_40_Figure_1.jpeg)

Figure 12: Wall–clock times for FEDGVI iterations per client. We plot the classification error against the computation time taken per client during each server iteration, where we train 5 Clients on 10% contaminated MNIST data. If we do not state the loss or divergence in the legend, it is LNLL and DKL respectively. Here, FEDGVI outperforms PVI in terms of accuracy while having similar runtimes.

# D.3.2. FASHIONMNIST (X[IAO ET AL](#page-12-23)., [2017\)](#page-12-23) DETAILS

We vary the amount of contamination from 0.0, 0.1, 0.2, 0.4, where the contamination is random and assigns each contaminated data point a different class uniformly at random. The model architecture, prior, learning rate, and optimiser remain unchanged and are as before.

FEDGVI uses the Alpha–Renyi divergence with an alpha value of 2.5 for all, and the robust generalised cross entropy ´ loss, where δ = 0.0 indicates the negative log likelihood. Table [6](#page-40-1) specifies Table [2](#page-7-3) to a higher precision but the results are identical.

Table 6: Classification accuracy (highest in bold) on uncontaminated test data after training on different amounts of contaminated FASHIONMNIST data. Each Method has data split homogeneously across 3 Clients. We report the best performance during all server iterations for each method.

|   |    | M   | ODEL  |    |      |   |       |    |      |       | C   |      |       |     |      |       |     |      |
|---|----|-----|-------|----|------|---|-------|----|------|-------|-----|------|-------|-----|------|-------|-----|------|
|   |    |     |       |    |      |   |       | 0% |      |       | 10% |      |       | 20% |      |       | 40% |      |
|   |    | F   | ED A  | VG |      |   | 85.72 | ±  | 0.52 | 78.99 | ±   | 1.90 | 71.16 | ±   | 1.53 | 48.97 | ±   | 6.51 |
|   |    | F   | ED    | PA |      |   | 88.08 | ±  | 0.30 | 87.36 | ±   | 0.15 | 86.54 | ±   | 0.16 | 85.36 | ±   | 0.53 |
|   | β  | –P  | RED B |    | AYES |   | 87.58 | ±  | 0.13 | 87.20 | ±   | 0.12 | 86.82 | ±   | 0.07 | 85.77 | ±   | 0.10 |
|   |    |     | PVI   |    |      |   | 86.21 | ±  | 0.21 | 85.14 | ±   | 0.13 | 84.36 | ±   | 0.12 | 82.81 | ±   | 0.05 |
| F | ED | GVI | δ     | =  | 0    | 0 | 87.12 | ±  | 0.12 | 86.23 | ±   | 0.15 | 85.56 | ±   | 0.11 | 83.78 | ±   | 0.09 |
| F | ED | GVI | δ     | =  | 0    | 4 | 88.73 | ±  | 0.21 | 88.60 | ±   | 0.09 | 87.01 | ±   | 0.37 | 78.14 | ±   | 0.39 |
| F | ED | GVI | δ     | =  | 0    | 5 | 89.02 | ±  | 0.18 | 88.57 | ±   | 0.16 | 88.39 | ±   | 0.21 | 85.06 | ±   | 0.67 |
| F | ED | GVI | δ     | =  | 0    | 8 | 88.59 | ±  | 0.03 | 88.44 | ±   | 0.07 | 87.95 | ±   | 0.04 | 87.21 | ±   | 0.10 |
| F | ED | GVI | δ     | =  | 1    | 0 | 88.09 | ±  | 0.08 | 87.83 | ±   | 0.14 | 87.54 | ±   | 0.15 | 85.97 | ±   | 0.27 |