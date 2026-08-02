# Learning Time-Varying Multi-Region Brain Communications Via Scalable Markovian Gaussian Processes

Weihan Li 1 Yule Wang 1 Chengrui Li 1 **Anqi Wu** 1

## Abstract

Understanding and constructing brain communications that capture dynamic communications across multiple regions is fundamental to modern system neuroscience, yet current methods struggle to find time-varying region-level communications or scale to large neural datasets with long recording durations. We present a novel framework using Markovian Gaussian Processes to learn brain communications with timevarying temporal delays from multi-region neural recordings, named Adaptive Delay Model (ADM). Our method combines Gaussian Processes with State Space Models and employs parallel scan inference algorithms, enabling efficient scaling to large datasets while identifying concurrent communication patterns that evolve over time. This time-varying approach captures how brain region interactions shift dynamically during cognitive processes. Validated on synthetic and multi-region neural recordings datasets, our approach discovers both the directionality and temporal dynamics of neural communication. This work advances our understanding of distributed neural computation and provides a scalable tool for analyzing dynamic brain networks. Code is available at https://github.com/ BRAINML-GT/Adaptive-Delay-Model.

## 1. Introduction

Modern system neuroscience faces a significant challenge in discovering communication patterns that capture dynamic interactions across multiple regions. Understanding these time-varying communications has become increasingly critical with the advent of advanced recording technologies that 1School of Computational Science & Engineering, Georgia Institute of Technology, Atlanta, USA. Correspondence to: Anqi Wu <anqiwu@gatech.edu>.

enable simultaneous measurement of neural activity across numerous brain areas with unprecedented temporal and spatial resolution (Steinmetz et al., 2021; Siegle et al., 2021; Li et al., 2024a; Nejatbakhsh et al., 2024). These large-scale neural recordings necessitate computational methods capable of uncovering and characterizing the dynamic nature of neural communication within the brain. Latent representations offer a promising approach to building time-varying multi-region communications (Wang et al.,
2023; 2024; Zhang et al., 2024). In the brain, communication patterns between brain regions manifest at varying temporal scales: some regions exhibit fast, synchronous interactions with short delays, indicative of strong functional coupling, while others display slower interactions with longer delays, reflecting more indirect relationships. Such communications provide a framework for understanding how these diverse communication patterns evolve over time, capturing both feedforward and feedback pathways that may shift during different cognitive states (Lillicrap et al., 2020; Liu et al., 2025). Current computational approaches for modeling multiregion communications can be broadly categorized into nondelay models and delay models. Non-delay models, such as mp-srSLDS (Glaser et al., 2020) and MR-SDS (Karniol- Tambour et al.), do not explicitly incorporate temporal delays when capturing latent dynamics across regions. As a result, they can only learn the directional information of communications. Delay models, including DLAG (Gokcen et al., 2022), m-DLAG (Gokcen et al., 2024a), their approximated versions (Gokcen et al., 2024b), and MRM-GP
(Li et al., 2024b), introduce mechanisms to learn temporal delays between pairs of communications. Learning delays provides not only directional information but also insights into relative communication speeds, which can help assess the strength of functional coupling between brain regions. Although these methods have shown potential in capturing inter-region brain communications, delay models, such as DLAG and its variants, do not account for the dynamic nature of multi-region communication. They fail to model time-varying temporal dependencies and communication patterns, which are crucial for understanding neural processing. Non-delay models, such as MR-SDS and mp-srSLDS,
1 introduce dynamic message flow among regions but don't model delays between communications and assume a single communication subspace, meaning they cannot capture concurrent communications over different subspaces. MRM-GP is a specific case that integrates a Gaussian Process (GP) with a State Space Model (SSM), making it a delay model with discrete changing of phase delays. MRM- GP has two key limitations: (1) it only supports kernels that are separable across spatial and temporal domains, restricting its applicability in the frequency domain by learning phase delays, whereas temporal delays are more generalizable for neuroscience applications; (2) it assumes that the delays across all communication subspaces across regions change synchronously with hidden state transition. However, this assumption does not reflect the true brain mechanism, where different communication pathways between regions can operate asynchronously. In terms of computational efficiency, DLAG and mDLAG,
which are GP-based methods, incur an O(T
3) computational cost, where T is the number of time samples, making them challenging to scale. Approximated GP and SSM-based methods, such as MRM-GP, MR-SDS, and mpsrSLDS, reduce the cost to O(T). However, they rely on sequential inference, which remains inefficient for large neural datasets with long recording durations. Moreover, the reliance on discrete hidden states, e.g., MRM-GP, introduces inefficiencies during inference. To address these limitations, we propose the Adaptive Delay Model (ADM). It falls within the family of Markovian Gaussian process-based delay models (Li et al., 2024b), but incorporates a continuous time-varying temporal delay. Thus unlike static communication models commonly used in neuroscience, ADM can accommodate communication patterns with varying temporal characteristics, offering a more flexible and biologically relevant framework. Markovian Gaussian Process (Markovian GP) models integrate the expressive power of GPs with the computational efficiency of SSMs, facilitating scalable analysis of largescale neural recordings while discovering multiple evolving communication patterns. However, existing Markovian GP approaches either rely on single-output kernels or require multi-output kernels to be separable across spatial and temporal domains (Solin et al., 2016; Loper et al., 2021; Dowling et al., 2021; 2023; Li et al., 2024b). In this paper, we introduce a novel, universal connection between arbitrary temporally stationary GPs and SSMs, making the framework highly flexible and broadly applicable to various neuroscience problems. Additionally, we apply an advanced inference method for Markovian GP, leveraging parallel scan algorithms (Blelloch, 1990; Sarkk ¨ a & Garc ¨ ´ıa-Fernandez ´ , 2020) to significantly accelerate computation and reducing complexity to O(log T). This approach enables efficient analysis of long-duration recordings while capturing dynamic communication patterns. We validate our model using neural recordings from multiple regions of the brain during visual processing tasks (Semedo et al., 2019; Siegle et al., 2021). Our results demonstrate the method's capability to uncover how information flow patterns dynamically change across multi-region networks, offering new insights into the temporal organization of largescale neural circuits and advancing our understanding of distributed neural computation. In summary, the key contributions of our work are:
- We establish a universal connection between arbitrary temporally stationary GPs and SSMs, which has broader implications for other domains where computational efficiency is a priority.

- Our model discovers time-varying multi-region communications from latent representations of neural recordings without introducing additional discrete hidden states in the SSM. - We propose a scalable method for analyzing multi-region communications in large-scale neural data with O(log T) complexity.

## 2. Method

We begin by demonstrating how a Gaussian Process (GP) with a Factor Analysis (FA) model can be used to capture static brain communication across regions (Section 2.1). Next, we establish a universal connection between a GP and a State Space Model (SSM), referred to as the Markovian Gaussian Process (Section 2.2). Finally, we illustrate how time-varying brain communication can be modeled (Section 2.3).

## 2.1. Gaussian Process Factor Analysis For Brain Communications

The Gaussian Process Factor Analysis (GPFA) for modeling brain communication employs the multi-output Squared Exponential (MOSE) kernel (Gokcen et al., 2022):

$${\bf K}_{i j}(\tau)=\exp\left(-\frac{(\tau+\theta_{i j})^{2}}{2l^{2}}\right),\qquad\qquad(1)$$

where *i, j* represent two brain regions, τ = t − t
′is the time difference, θij is the temporal delay between regions i and j, and l is the length scale shared across all regions. This MOSE kernel allows the identification of temporal delays that characterize communication dynamics across regions.

Our goal is to learn MN latent variables, x ∈ RMN×T,
from neural recordings y ∈ R
D×Tacross N regions. Each region is associated with M latent variables. A typical assumption for x is to decompose it into two components: across-region variables and within-region variables (Gokcen et al., 2022; Li et al., 2024b). The across-region variables, x a ∈ R
maN×T, capture shared neural activity that reflects communication between brain regions. These variables exhibit the similar dynamics across regions, differing only in temporal delays. The within-region variables, x w ∈ R
mwN×T, represent neural activity unique to individual regions and are independent of other regions. Together, these components form the latent representation of neural recordings, x = [x a, x w], where ma + mw = M. The relationship between y and x is then modeled using a Factor Analysis (FA) model:
y = Cx + d + ϵ, (2)

$${\boldsymbol{y}}=\mathbf{C}{\boldsymbol{x}}+d+\epsilon,$$

where C ∈ R
D×MN is a block-diagonal matrix C =
diag{C1*, . . . ,* CN }, with each Cirepresenting the mapping from region i's latent variable to its neural recordings. Additionally, d ∈ R
D×1is a bias term, and ϵ ∼ N (0, V) represents Gaussian noise with diagonal covariance V ∈ R
D×D.

The across-region variables x aare designed to capture communication patterns among regions. For the m-th group of across-region variables, x am ∈ R
N×T, the activity from each region exhibits spatial correlations with the N −1 other regions. These variables are modeled as a Gaussian Process (GP) with the MOSE kernel, which captures the temporal delay characteristics of across-region communication. On the other hand, the m-th group of within-region variables x wm ∈ R
N×T, representing region-specific activity, is modeled independently across regions using a single-output Squared Exponential (SE) kernel:

$$\mathbf{K}^{\mathrm{single}}(\tau)=\exp\left(-{\frac{\tau^{2}}{2l^{2}}}\right).$$

Additionally, independence is assumed across different groups of both the across-region and within-region variables, with each group m governed by distinct kernel parameters. By explicitly separating the across-region and within-region latent variables, this framework offers a clear representation of across-region communication and region-specific dynamics, enabling a more interpretable analysis of multi-region neural recordings.

## 2.2. Connect Gaussian Process With State Space Model

We develop a novel universal connection between arbitrary temporally stationary GPs and SSMs, enabling us to efficiently model both across- and within-region dynamics. Gaussian Process and State-Space Approximation. The m-th group of across-region variables, x am ∈ R
N×Tare modeled as a GP with MOSE kernel:

$$\mathcal{GP}(0,\begin{bmatrix}\mathbf{K}(0)&\mathbf{K}(-1)&\dots&\mathbf{K}(-T+1)\\ \mathbf{K}(1)&\mathbf{K}(0)&\dots&\mathbf{K}(-T+2)\\ \vdots&\vdots&\ddots&\vdots\\ \mathbf{K}(T-1)&\mathbf{K}(T-2)&\dots&\mathbf{K}(0)\end{bmatrix}),\tag{4}$$

where each K(τ ) ∈ R
N×N is a MOSE kernel in Eq. 1 with an interval τ over N brain regions. Our goal is to find a statespace approximation of x am, which follows a Multi-Order SSM structure:

$$\mathbf{x}_{m,t}^{a}=\sum_{p=1}^{P}\mathbf{A}_{p}\mathbf{x}_{m,t-p}^{a}+\mathbf{q}_{t},\quad\mathbf{q}_{t}\sim\mathcal{N}(0,\mathbf{Q}),\quad(5)$$
$$(2)$$

where P represents the number of orders, A1*, . . . ,* AP ∈
R 
N×N are the transition matrices, and Q ∈ R
N×N is the process noise matrix.

Determining an State Space Model using Kernels. To estimate transition matrices and measurement using x am, we can consider the SSM in Eq. 5 as a regression model (Neumaier & Schneider, 2001):

$$\mathbf{x}_{m,t}^{a}=\mathbf{G}\mathbf{v}_{t}+\mathbf{q}_{t},\quad\mathbf{q}_{t}\sim\mathcal{N}(0,\mathbf{Q}),$$
$$(6)$$

where G ∈ R
N×NP is the regression coefficient and vt ∈
R 
NP ×1is the predictor:

$$\begin{array}{l}{{\mathbf{G}=[\mathbf{A}_{P},\mathbf{A}_{P-1},\ldots,\mathbf{A}_{1}],}}\\ {{\mathbf{v}_{t}=[\mathbf{x}_{m,t-P}^{a,\top},\mathbf{x}_{m,t-P+1}^{a,\top},\ldots,\mathbf{x}_{m,t-1}^{a,\top}]^{\top}.}}\end{array}$$
$$(7)$$
⊤.(7)
$$(3)$$

Our ultimate goal is to use K(τ ) to represent G and Q.

First, we can represent G and Q as functions of x am. Concretely, given T samples, x am,1*, . . . ,* x am,T , we define predictor matrix as V = [vP +1*, . . .* vT ]
⊤ ∈ R
NP ×(T −P )and target observation matrix as W = [x am,P +1*, . . . ,* x am,T ] ∈
R 
N×(T −P ).

Then, we can represent the regression model in Eq. 6 in the matrix form: W = GV + R, where R is the residual matrix. By doing so, we can estimate coefficient matrix G and the process noise matrix Q by least squares estimation:

$$\mathbf{G}=\mathbf{W}\mathbf{V}^{\mathsf{T}}(\mathbf{V}\mathbf{V}^{\mathsf{T}})^{-1},\quad\mathbf{Q}={\frac{\mathbf{R}\mathbf{R}^{T}}{T-P-1}},$$
$$({\mathfrak{s}})$$
, (8)
where R = W − GV denotes an estimate of the residual matrix, and its covariance is an estimate of process noise matrix Q. Now, if we can represent WV⊤ and VV⊤ with K(τ ), we will achieve the ultimate goal. Since each sample x am,t in V and W is modeled as a sample in the GP in Eq. 4. We can represent VV⊤ ∈ R
NP ×NP
and WV⊤ ∈ R
N×NP using K(τ ) as (full derivations see Appendix A):

$$\mathbf{VV}^{\top}\propto\begin{bmatrix}\mathbf{K}(0)&\mathbf{K}(-1)&\ldots&\mathbf{K}(-P+1)\\ \mathbf{K}(1)&\mathbf{K}(0)&\ldots&\mathbf{K}(-P+2)\\ \vdots&\vdots&\ddots&\vdots\\ \mathbf{K}(P-1)&\mathbf{K}(P-2)&\ldots&\mathbf{K}(0)\end{bmatrix},$$ $$\mathbf{WV}^{\top}\propto\begin{bmatrix}\mathbf{K}(P)&\mathbf{K}(P-1)\ldots&\mathbf{K}(1)\end{bmatrix}.\tag{9}$$
,
(9)
Notably, each K(τ ) ∈ R
N×N , where τ ∈ [−P + 1, P − 1],
depends only on the number of brain regions N and can be efficiently computed using the stationary kernel function employed in the GP. Furthermore, K(τ ) can represent any stationary temporal kernel, establishing a universal connection between GPs and SSMs. We apply this universal conversion to various kernels in GP regression task, see Appendix D for details. Markovian Across-region Communcations. Now, the transition matrices and the measurement matrix in Eq. 5 are uniquely determined by the kernel functions of GP by Eq. 8 and Eq. 9. Moreover, we can rewrite the SSM in Eq. 5 to an SSM with a Markovian structure, resulting in a Markovian Gaussian Process (Markovian GP) (Zhao, 2021):

$$\hat{\mathbf{x}}_{m,t}^{a}=\hat{\mathbf{A}}\hat{\mathbf{x}}_{m,t-1}^{a}+\mathbf{q}_{t},\quad\mathbf{q}_{t}\sim\mathcal{N}(0,\hat{\mathbf{Q}}),\tag{10}$$ $$\mathbf{x}_{m,t}^{a}=\hat{\mathbf{H}}\hat{\mathbf{x}}_{m,t}^{a},$$

where H ∈ R
N×NP denotes a mask matrix, Aˆ ∈
R 
NP ×NP is structured as a controllable canonical form
(Grewal & Andrews, 2014) and small constants are added to Qˆ ∈ R
NP ×NP so it matches the shape of Aˆ :

$${\hat{\mathbf{A}}}={\begin{bmatrix}\mathbf{A}_{1}&\mathbf{A}_{2}&\dots&\mathbf{A}_{P-1}&\mathbf{A}_{P}\\ \mathbf{I}_{N}&0&\dots&0&0\\ 0&\mathbf{I}_{N}&\dots&0&0\\ 0&0&\ddots&0&0\\ 0&0&\dots&\mathbf{I}_{N}&0\end{bmatrix}},$$ $${\hat{\mathbf{Q}}}={\begin{bmatrix}\mathbf{Q}&0\\ 0&\sigma\mathbf{I}_{N(P-1)}\end{bmatrix}},\quad\mathbf{H}=\begin{bmatrix}\mathbf{I}_{N}&0\end{bmatrix},$$
$$(11)$$

with IN ∈ R
N×N denoting the identity matrix and σ a small constant added for numerical stability. Notably, although we use a Markovian structure to represent a stationary GP, our method still incorporates information from multiple orders.

Markovian Within-region Neural Activity. Similarly, the state-space approximation of the m-th group of withinregion variables, x wm ∈ R
N×T, can be seen as a specific case of across-region variables. In this case, each dimension, x wm,n ∈ R
T ×1, is independently modeled as a Markovian GP with a scalar single-output SE kernel (Eq. 3).

## 2.3. Time-Varying Brain Communications

Since the SSM in Eq. 10 follows a discrete structure, we can
extend it to incorporate time-varying transition and process
noise matrices as follows:
$$\hat{\mathbf{x}}_{m,t}^{a}=\hat{\mathbf{A}}_{t}\hat{\mathbf{x}}_{m,t-1}^{a}+\mathbf{q}_{t},\quad\mathbf{q}_{t}\sim\mathcal{N}(0,\hat{\mathbf{Q}}_{t}),\tag{12}$$ $$\mathbf{x}_{m,t}^{a}=\hat{\mathbf{H}}\hat{\mathbf{x}}_{m,t}^{a}.$$
This formulation introduces a time-varying MOSE kernel, where the temporal delay parameter θ*ij,t* evolves over time.

In other words, at each time step t, we construct a Markovian GP (or SSM) as described in Eq. 10, conditioned on the MOSE kernel specific to that time t. Additionally, the length scale parameter l is held constant over time, which limits the flexibility of each Aˆ t. By sharing l across time points, we constrain the temporal evolution of the delay parameters, promoting smoother dynamics and reducing the risk of misattributing variability in the messages to fluctuations in delays. This approach enables the model to learn time-varying temporal delays, effectively capturing the dynamics of multiregion brain communications. Compared to modeling timevarying phase delays using hidden discrete states (Li et al.,
2024b), our method is more flexible, as it does not assume that each group of across-region communications, x am, undergoes simultaneous delay changes during state transitions. Importantly, this computation can be efficiently performed in vectorized form across all T time steps, ensuring minimal impact on overall computational efficiency. In the FA
model, the projection matrix C ∈ R
D×MN , the bias term d ∈ R
D×1, and the observation Gaussian noise ϵ ∈ R
D×1 remain time-invariant.

## 3. Inference

Now, having established the connection between the m-th group of across-region communications and within-region neural activity to the Markovian GP (or SSM), as described in Eq. 12, the next step is to efficiently learn the latent variables and model parameters. Our model, ADM, formulated as an SSM, offers a significant advantage: it can learn the parameters using either a sequential estimation method with complexity O(T) or a parallel computation method with complexity O(log T). On modern hardware, the parallel approach is consistently faster due to its efficient utilization of computational resources. Parameter Settings The model parameters, denoted as Θ, include the kernel parameters θ k ij,t and l kfrom each latent dimension k, which define the transition matrix Aˆ t and the process noise covariance matrix Qˆ t for each across- or within-region latent group. Additionally, the Factor Analysis (FA) parameters include the projection matrix C, the bias term d, and the diagonal matrix V. The model also has a hyperparameter P, representing the order of the autoregressive process. To better understand the effect of different P values, we generate samples from our model for various P values, as shown in Appendix C.

Parallel Scan Kalman EM Algorithm Given neural recordings y ∈ R
D×T, our goal is to estimate the latent brain communications x ∈ RMN×Talong with the model parameters Θ. To achieve this, we use the parallel scanbased Kalman Expectation-Maximization (EM) inference algorithm (Sarkk ¨ a & Garc ¨ ´ıa-Fernandez ´ , 2020), which introduces a parallel scan version of the Kalman Filter and Smoother. Specifically, in the E-step, we apply the parallel Kalman Filter and Smoother to infer the latent variables and expected log-likelihood. In the M-step, we update the kernel parameters using gradient descent and optimize the Factor Analysis (FA) parameters through closed-form linear regression. See Appendix B for details. The objective of the Kalman Filter is to compute the posterior density p(xt|y0:t), given the neural data up to time step t, while the Kalman Smoother computes the posterior density p(xt|y0:T) for all time steps. Traditionally, both filtering and smoothing are computed in O(T) time using sequential updates. However, sequential computation is often inefficient compared to parallel computation, particularly on modern hardware architectures (Chen et al., 2024). To address this inefficiency, Sarkk ¨ a & Garc ¨ ´ıa-Fernandez ´ (2020) demonstrates that the sequential updates of the Kalman Filter and Smoother can be reformulated as an associative operator, enabling the use of the parallel scan algorithm (Blelloch, 1990). Consequently, the time complexity and memory cost of our model are given by:

Time complexity: O(MN3P Memory complexity: O(MN2P
3log T), (13)
2T), (14)
where N is the number of regions, M is the group number of across- and within-region latent dynamics, and P is the SSM order in Eq. 5. The cubic cost arises from Eq. 8. However, as we will show in the experimental section, the order parameter P and N can be significantly smaller compared to T while still achieving strong generative and inference performance. Thus, the cubic cost does not pose a major computational bottleneck.

## 4. Experiment

Datasets. We evaluate our model on three datasets.

- **Synthetic Data**: We generate synthetic data that incorporate both across-region communications and within-region neural activities, along with time-varying temporal delays, to simulate dynamic brain communications characterized by both fast and slow features. - **Two Brain Regions** (Semedo et al., 2019; Zandvakili & Kohn, 2019): Simultaneous spike train recordings from a monkey's primary visual area (V1) and secondary visual cortex (V2), with a 6Hz drifting grating as the external stimulus. - **Five Brain Regions** (Siegle et al., 2021): Simultaneous spike train recordings from a mouse's primary visual cortex (VISp), rostrolateral area (VISrl), anterolateral area (VISal), posteromedial area (VISpm), and anteromedial area (VISam), with a 4Hz drifting grating as the external stimulus. Baselines. We compare our model with three methods:
- **DLAG** (Gokcen et al., 2022): A Gaussian Process Factor Analysis model with a MOSE kernel, designed for neural recordings from two brain regions. It can be used to learn both across-region and within-region latent communications. - **mDLAG** (Gokcen et al., 2024a): An extension of DLAG that supports more than two brain regions with a different inference approach. Unlike DLAG, it assumes all latent variables correspond to across-region communications and does not explicitly model within-region dynamics. - **MRM-GP** (Li et al., 2024b): An approximation of a Gaussian Process with a Cross-Spectral Mixture (CSM) kernel (Ulrich et al., 2015), formulated as an SSM with O(T) complexity. It is designed to learn frequency-based communications between two brain regions and can capture both across-region and within-region latent dynamics.

$\eqref{eq:walpha}$. 
Evaluation. We evaluate our model and baseline models by randomly splitting the data into training, validation, and testing sets with a ratio of 0.8, 0.1, and 0.1, respectively. Since all models assume a linear/Gaussian relationship between the latent variables and observed data, we assess their performance by computing the observation test log-likelihood: LL(xtest, ytest), where xtest represents the inferred test latent variables, and ytest denotes the test neural recordings. To mitigate randomness, we report the average test log-likelihood over five different random seeds.

## 4.1. Synthetic Data

In this section, we simulate a common phenomenon in neuroscience where brain region communications are dynamic (Parra & Tobar, 2017). Our goal is to evaluate our model's ability to recover time-varying temporal delays and latent variables.

Experimental setup. We generate 120 independent trials for two brain regions (N = 2) with an order of P = 5

time bin time bin
(A)
5
-1 te m po ral d el ays region 1 estimated delay true delay 1
-5 true latent region 2 true latent 10 bins
(B)
ti me in m in utes
(C)
350k 400k 450k tes t l og
-li ke lih ood 128 256 512 1 4 16 parallel sequential P=5P=4P=3P=2P=1
and T = 200 time bins. Each region contains 50 neurons, with ma = 2 groups of across-region communications and mw = 1 group of within-region variables. For across-region communications, the first group represents forward communication, characterized by a larger positive delay of 5 bins and a smaller positive delay of 1 bin during time bins 30 to 70. The second group represents feedback communication, with a larger negative delay of -5 bins and a smaller negative delay of -1 bin during time bins 130 to 170. For across-region and within-region dynamics, the length scales are set to l = 5 and l = 2.5, respectively. Note that large delays in this context are intended to represent slow communication in the brain, whereas an extremely large delay would imply an absence of communication between regions. A small delay indicates fast communication. Therefore, our data simulation is designed to reflect a scenario where region A initially has minimal effective communication with region B (delay of 5), then suddenly transmits a signal (delay of 1), followed by another period of ineffective communication (delay of 5). A delay of -5 represents communication in the opposite direction. During fitting, we set ma = 2, mw = 1, and P = 5. Results. Figure 1(A) presents the estimated and truth across-region communications, temporal delays, and ground truth over T bins for P = 5. For the estimated delay, the shaded area represents the variance across five different runs.

Our model effectively captures time-varying communications for both latent dynamics and delay. See Appendix E for within-region neural activities. Figure 1(B) shows the test log-likelihood summed over trials and time bins. The results indicate that performance remains relatively stable for P ∈ [2, 5], except for P = 1, which yields the lowest performance. Figure 1(C) compares the time costs of the sequential and parallel scan-based Kalman EM algorithms with GPU parallelization. We generate synthetic data with up to T = 600 time bins. The results demonstrate that the parallel version is significantly faster than the sequential update. Finally, Appendix G presents a more complex case with five-region synthetic dataset and evaluates the model's performance as a function of the number of regions, latent variables, and data length.

## 4.2. Two Brain Regions

In this section, we investigate the interactions between the mouse's primary visual area (V1) and secondary visual cortex (V2) in response to a 6Hz drifting grating. Additionally, we compare our model's performance and inference time with MRM-GP and DLAG. Experimental setup. We use smoothed multi-region spike trains from session 106r001p26 with an orientation of 0
◦.

This dataset consists of 400 trials, each containing 64 time bins (20 ms per bin), with 72 V1 neurons and 22 V2 neurons. The monkey begins receiving the visual stimulus (drifting gratings) at the first time bin, and the stimulus persists throughout all 64 time bins. The number of across-region and within-region latent dynamics follows previous works
(Gokcen et al., 2022; Li et al., 2024b), where ma = 2 and mw = 2. The order P = 4 is selected based on performance evaluation on the validation dataset. Results. Figure 2(A) shows the estimated across-region communications and time-varying delays from the test dataset (ten trials shown; within-region dynamics are provided in Appendix E). The first group of communications shows a shift from slower feedback (with larger absolute delays) to faster feedback (with smaller absolute delays) starting around 200 ms after stimulus onset. In contrast, the second group exhibits a periodic pattern driven by the external drifting grating stimulus, along with a change in communication direction immediately following stimulus onset. The time-varying delays suggest the following interpretation: shortly after stimulus presentation, V2 generates a strong feedback signal from V2 to V1, potentially reflecting the emergence of surprise or prediction error (Rao &
Ballard, 1999). As time progresses, both regions transition into more synchronized oscillations. Our findings on V1-V2

tem po ral del ays(A)
(B) (C)
-30 0 ms 20 ms te st lo g-li kel ihoo d P=5P=4P=3P=2P=1 20k 25k 30k 30k loglike liho od
-20 0 -15 val idati on
 

28k estimated delay zero line 26k 100ms ADMMRM-GPDLAG
V1
(D)
50 100 150 200 1 4 16 tim e i n mi nutes ADM MRM-GP DLAG
V2->V1 V2->V1 V1->V2 V2 across-region communcation 1 across-region communcation 2
interactions are similar to previous studies (Gokcen et al.,
2022; Li et al., 2024b; Gokcen et al., 2024a).

Figure 2(B) compares the test log-likelihood summed over trials and time bins, showing that our model, ADM, outperforms MRM-GP and DLAG under the same ma and mw settings. This improvement is attributed to ADM's ability to capture continuously time-varying temporal delays. Figure 2(C) presents the validation log-likelihood across different P values, with P = 4 achieving the highest value. Combined with the insights from Figures 2(A–B), this suggests that a small P value can effectively estimate model parameters and latent variables. Figure 2(D) compares the computational time of our model with MRM-GP and DLAG on spike trains of varying T, obtained by concatenating trials. The use of parallel computation significantly improves efficiency, outperforming the linear model (MRM-GP) and the cubic model (DLAG).

## 4.3. Five Brain Regions

In this section, we scale up our model to a larger neural recording spanning five regions with increased time resolution. Our objective is to investigate across-region communications and identify the time-varying meso-scale brain network, defined as the dynamic network spanning sAppendixub-brain regions, e.g., regions in visual cortex. Experimental setup. We use smoothed multi-region spike train data from the Visual Coding - Neuropixels project by the Allen Institute (Siegle et al., 2021), specifically from session 750749662. This dataset includes spike trains recorded from VISp, VISrl, VISal, VISpm, and VISam—sub-areas of the mouse visual cortex. It consists of 120 trials, T = 200 time bins (each 10 ms), and a total of 202 neurons, with external visual stimuli comprising 4 Hz drifting gratings. Following the approach in (Gokcen et al., 2022), we first apply Factor Analysis to estimate the total number of across-region and within-region latent dynamics, determining M = 4 (see Appendix F for details). We then conduct a grid search with 5-fold cross-validation to refine the number of across-region and within-region latent dynamics and the model order P.

Results. Figure 3(A) presents the ten estimated pairwise temporal delays from one group of across-region communications. See Appendix E for the estimated latent dynamics. Our results reveal consistent forward communication from VISp to downstream visual areas, such as VISrl, VISal, and VISpm, aligning with the known anatomical hierarchy of the mouse visual cortex (Siegle et al., 2021). Additionally, these forward communications exhibit time-varying dynamics. For instance, communication between VISp and VISrl transitions from slow to fast, indicating an enhanced interaction that gradually becomes more synchronous following the initial surprise response to the visual stimulus onset. In contrast, the communication between VISp and VISal shifts from fast to slow, suggesting inhibition induced by the external stimulus. Furthermore, our findings indicate that all communications involving VISam are feedback signals. This is expected, as VISam is positioned at the end of the anatomical hierarchy of the mouse visual system, consistent with the anatomical hierarchy scores reported in (Siegle et al., 2021).

Figure 3(B) depicts the meso-scale brain network corresponding to the across-region communications presented in Figure 3(A). Each node represents a region in the visual system, while directed edges indicate the directional

VISam VISam VISpm VISpm
(A) (B)
VISp VISp VISrl VISp VISal VISp VISpm VISp VISam VISrl VISal 0 20 ms ms ms 0
-20 0
-25 VISp
(C) (D)
0 10 ms
-25 ms 0 25 VISrl VISal VISal te m po ral d el ays VISrl meso-scale brain network when t=3 meso-scale brain network when t=50 VISrl VISpm VISrl VISam VISal VISpm VISal VISam VISpm VISam ms ms ms ms ms 0 25 0 0 6e5 1.9e6 2.1e6 highest 0 tes t l o g-l ike lih oo d

-50 0
-20 lo g-li ke lihoo d 
-20
-25 20 hyperparameter combinations val id ati on 5e5
-25
-40
-50 100msestimated delay zero line ADMmDLAG
communications. The length of each edge is determined by the estimated delays, reflecting the speed of communication. The figure presents two meso-scale brain networks at t = 3 and t = 50 time bins. The primary differences between these networks include changes in the speed of certain forward communications and a direction change in the communication direction between VISrl and VISal, suggesting the emergence of stimulus presentation. Figure 3(C) presents the cross-validation results for twenty hyperparameter combinations. We first determine M = 4 using Factor Analysis (see Appendix F for details), then conduct a grid search over all combinations of ma ∈ [0, 4], mw ∈ [0, 4], and P ∈ [2, 5]. The highest validation loglikelihood is achieved with ma = 3, mw = 1, and P = 5. Figure 3(D) compares the test log-likelihood, summed over trials and time bins, between our model (ADM) and mD- LAG with ma = 3 latent communication channels, where mDLAG is an extension of DLAG that supports more than two brain regions using variational inference. The results indicate that ADM provides a better fit to the data, attributed to its ability to model time-varying communications. We do not compare MRM-GP and DLAG since they are limited to two brain regions. Additionally, we skip a time cost comparison with mDLAG because it is implemented only in MATLAB, which is significantly slower than our GPU-optimized implementation.

## 5. Discussion

Summary. Our findings highlight the importance of modeling time-varying multi-region neural communications and demonstrate that the Adaptive Delay Model (ADM) effectively captures these dynamics while maintaining computational efficiency. Existing methods for studying acrossregion neural interactions can be broadly categorized into non-delay models and delay models. While non-delay models provide directional communication patterns, they fail to capture temporal delays, limiting their ability to infer the communication speed. Conversely, delay models, such as DLAG and MRM-GP, introduce delay estimation but assume static or discretely changing delays, which do not reflect the continuously evolving nature of brain communications. Our results show that ADM overcomes these limitations by incorporating a flexible, time-varying delay mechanism, enabling a more biologically relevant representation of neural interactions. Neuroscience Implications. Our results from large-scale neural recordings show that across-region communication delays are not static but change over time. Notably, we observe transitions from slow feedback and forward communication to fast forward interactions in both datasets (Section 4.2 and Section 4.3), aligning with adaptive sensory processing in the visual cortex. These findings show the importance of time-varying models in capturing the dynamic nature of brain communications. Computational Advancements. Beyond its neuroscientific implications, our model contributes to the broader field of computational modeling by bridging temporally stationary GPs with SSMs. Traditional GP-SSM connections often rely on separability assumptions in spatial and temporal kernels, limiting their flexibility. Our proposed universal connection between arbitrary temporally stationary GPs and SSMs removes this restriction. Furthermore, by leveraging parallel scan algorithms, ADM achieves an impressive computational complexity of O*(log* T), significantly improving scalability compared to existing methods.

Limitations and Future Directions. Our model has a cubic time complexity with respect to the number of brain regions N and the SSM order P. Although these values are typically much smaller than T, they can still become computational bottlenecks for specific cases. A potential solution may involve leveraging frequency domain techniques. Parnichkun et al. (2024) proposed a state-free SSM with a controllable canonical transition matrix, similar to ours in Eq. 11, and utilized the Fast Fourier Transform to achieve linear scaling in latent size. Similarly, Gokcen et al. (2024b) approximated the GP kernel in the frequency domain to reduce the computational cost to linear in latent size.

## Acknowledgement

This work is supported by National Institutes of Health BRAIN initiative (1U01NS131810).

## Impact Statement

Our work presents the Adaptive Delay Model (ADM), a scalable and biologically relevant framework for uncovering time-varying multi-region neural communications. By bridging Gaussian Processes (GPs) with State Space Models (SSMs), ADM enhances our ability to analyze large-scale neural recordings, offering insights into sensory processing, cognitive flexibility, and neural disorders. Beyond neuroscience, its adaptability to fields such as robotics, autonomous systems, and signal processing broadens its societal impact. Ethically, as machine learning models increasingly shape neuroscience research, it is crucial to ensure their responsible application, avoiding overinterpretation of inferred neural dynamics and considering the broader implications for brain-computer interfaces and neurotechnology.

## References

Balzani, E., Noel, J. P., Herrero-Vidal, P., Angelaki, D. E.,
and Savin, C. A probabilistic framework for task-aligned intra-and inter-area neural manifold estimation. *arXiv* preprint arXiv:2209.02816, 2022.

Blelloch, G. E. Prefix sums and their applications. 1990. Boots, B. Learning stable linear dynamical systems. Online]. Avail.: https://www. ml. cmu. edu/research/dappapers/dap *boots. pdf [Accessed 30 05 2016]*, 2009.

Chen, Z., Lin, C.-H., Liu, R., Xiao, J., and Dyer, E. Your contrastive learning problem is secretly a distribution alignment problem. Advances in Neural Information Processing Systems, 37:91597–91617, 2024.

Dowling, M., Sokoł, P., and Park, I. M. Hida-mat ´ \'ern kernel. *arXiv preprint arXiv:2107.07098*, 2021.

Dowling, M., Zhao, Y., and Park, I. M. Linear time gps for inferring latent trajectories from neural spike trains. arXiv preprint arXiv:2306.01802, 2023.

Glaser, J., Whiteway, M., Cunningham, J. P., Paninski, L.,
and Linderman, S. Recurrent switching dynamical systems models for multiple interacting neural populations. Advances in neural information processing systems, 33: 14867–14878, 2020.

Gokcen, E., Jasper, A. I., Semedo, J. D., Zandvakili, A.,
Kohn, A., Machens, C. K., and Yu, B. M. Disentangling the flow of signals between populations of neurons. Nature Computational Science, 2(8):512–525, 2022.

Gokcen, E., Jasper, A., Xu, A., Kohn, A., Machens, C. K.,
and Yu, B. M. Uncovering motifs of concurrent signaling across multiple neuronal populations. Advances in Neural Information Processing Systems, 36, 2024a.

Gokcen, E., Jasper, A. I., Kohn, A., Machens, C. K., and Yu, B. M. Fast multi-group gaussian process factor models.

arXiv preprint arXiv:2412.16773, 2024b.

Grewal, M. S. and Andrews, A. P. Kalman filtering: Theory and Practice with MATLAB. John Wiley & Sons, 2014.

Karniol-Tambour, O., Zoltowski, D. M., Diamanti, E. M.,
Pinto, L., Brody, C. D., Tank, D. W., and Pillow, J. W. Modeling state-dependent communication between brain regions with switching nonlinear dynamical systems. In The Twelfth International Conference on Learning Representations.

Li, C., Li, W., Wang, Y., and Wu, A. A differentiable partially observable generalized linear model with forward-backward message passing. arXiv preprint arXiv:2402.01263, 2024a.

Li, W., Li, C., Wang, Y., and Wu, A. Multi-region markovian gaussian process: An efficient method to discover directional communications across multiple brain regions. In Forty-first International Conference on Machine Learning, 2024b.

Lillicrap, T. P., Santoro, A., Marris, L., Akerman, C. J.,
and Hinton, G. Backpropagation and the brain. Nature Reviews Neuroscience, 21(6):335–346, 2020.

Liu, Y. A., Nong, Y., Feng, J., Li, G., Sajda, P., Li, Y., and Wang, Q. Phase synchrony between prefrontal noradrenergic and cholinergic signals indexes inhibitory control.

bioRxiv, pp. 2024–05, 2025.

Loper, J., Blei, D., Cunningham, J. P., and Paninski, L. A
general linear-time inference method for gaussian processes on one dimension. Journal of Machine Learning Research, 22(234):1–36, 2021.

Nejatbakhsh, A., Geadah, V., Williams, A. H., and Lipshutz, D. Comparing noisy neural population dynamics using optimal transport distances. *arXiv preprint* arXiv:2412.14421, 2024.

Neumaier, A. and Schneider, T. Estimation of parameters and eigenmodes of multivariate autoregressive models. ACM Transactions on Mathematical Software (TOMS), 27(1):27–57, 2001.

Parnichkun, R. N., Massaroli, S., Moro, A., Smith, J. T.,
Hasani, R., Lechner, M., An, Q., Re, C., Asama, H., ´ Ermon, S., et al. State-free inference of state-space models: The transfer function approach. *arXiv preprint* arXiv:2405.06147, 2024.

Parra, G. and Tobar, F. Spectral mixture kernels for multioutput gaussian processes. Advances in Neural Information Processing Systems, 30, 2017.

Rao, R. P. and Ballard, D. H. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature neuroscience*, 2(1):79–87, 1999.

Safaie, M., Chang, J. C., Park, J., Miller, L. E., Dudman, J. T., Perich, M. G., and Gallego, J. A. Preserved neural dynamics across animals performing similar behaviour. Nature, 623(7988):765–771, 2023.

Sarkk ¨ a, S. and Garc ¨ ´ıa-Fernandez, ´ A. F. Temporal paral- ´
lelization of bayesian smoothers. *IEEE Transactions on* Automatic Control, 66(1):299–306, 2020.

Semedo, J. D., Zandvakili, A., Machens, C. K., Byron, M. Y., and Kohn, A. Cortical areas interact through a communication subspace. *Neuron*, 102(1):249–259, 2019.

Siegle, J. H., Jia, X., Durand, S., Gale, S., Bennett, C.,
Graddis, N., Heller, G., Ramirez, T. K., Choi, H., Luviano, J. A., et al. Survey of spiking in the mouse visual system reveals functional hierarchy. *Nature*, 592(7852):86–92, 2021.

Solin, A. et al. Stochastic differential equation methods for spatio-temporal gaussian process regression. 2016.

Steinmetz, N. A., Aydin, C., Lebedeva, A., Okun, M., Pachitariu, M., Bauza, M., Beau, M., Bhagat, J., Bohm, ¨ C., Broux, M., et al. Neuropixels 2.0: A miniaturized high-density probe for stable, long-term brain recordings. Science, 372(6539):eabf4588, 2021.

Ulrich, K. R., Carlson, D. E., Dzirasa, K., and Carin, L. Gp kernels for cross-spectrum analysis. *Advances in neural* information processing systems, 28, 2015.

Wang, Y., Wu, Z., Li, C., and Wu, A. Extraction and recovery of spatio-temporal structure in latent dynamics alignment with diffusion models. Advances in Neural Information Processing Systems, 36:38988–39005, 2023.

Wang, Y., Li, C., Li, W., and Wu, A. Exploring behaviorrelevant and disentangled neural dynamics with generative diffusion models. *Advances in Neural Information* Processing Systems, 37:34712–34736, 2024.

Williams, A. H., Kunz, E., Kornblith, S., and Linderman, S. Generalized shape metrics on neural representations. Advances in Neural Information Processing Systems, 34: 4738–4750, 2021.

Wilson, A. and Adams, R. Gaussian process kernels for pattern discovery and extrapolation. In *International* conference on machine learning, pp. 1067–1075. PMLR,
2013.

Zandvakili, A. and Kohn, A. Paired v1-v2 neuronal spiking responses in anesthetized macaque monkey. *CRCNS. org*, 2019.

Zhang, Y., Wang, Y., Jimenez-Benet ´ o, D., Wang, Z., Az- ´
abou, M., Richards, B., Tung, R., Winter, O., Dyer, E., Paninski, L., et al. Towards a" universal translator" for neural dynamics at single-cell, single-spike resolution. Advances in Neural Information Processing Systems, 37: 80495–80521, 2024.

Zhao, Z. State-space deep gaussian processes with applications. *arXiv preprint arXiv:2111.12604*, 2021.

## A. Derivation For Vv⊤, Wv⊤**, And** Ww⊤

Let's inspect VV⊤ ∈ R
NP ×NP first. To simplify the notation, we use x to represent x am in Eq. 7. We have:

 x ⊤ 1 x ⊤ 2. . . x ⊤ P x ⊤ 2 x ⊤ 3. . . x ⊤ P +1 ............ x ⊤ T −P x ⊤ T −P +1 . . . x ⊤ T −1  VV⊤ =   x1 x2 . . . xT −P x2 x3 . . . xT −P +1 ............ xP xP +1 . . . xT −1     (15) =   x1x ⊤ 1 + · · · + xT −P x ⊤ T −P. . . x1x ⊤ P + x2x ⊤ P +1 + · · · + xT −P x ⊤ T −1 ......... xP x ⊤ 1 + · · · + xT −1x ⊤ T −P. . . xP x ⊤ P + xP +1x ⊤ P +1 + · · · + xT −1x ⊤ T −1   ,
where the first element x1x
⊤
1 ∈ R
N×N represents the auto-covariance of x1, which is essentially the kernel K(0) (Eq. 4).

In other words, since x is modeled as a stationary GP, the elements x1x
⊤
1*, . . . ,* xT −1x
⊤
T −1are all equivalent and correspond to the diagonal elements K(0) in Eq. 4. Similarly, the elements xP x
⊤
1*, . . . ,* xT −1x
⊤
T −Prepresent cross-covariances with time interval P − 1, which correspond to the off-diagonal elements K(P − 1). Therefore, we can further write Eq. 15 as:

$$\mathbf{VV}^{\top}=\begin{bmatrix}\mathbf{K}(0)+\cdots+\mathbf{K}(0)&\cdots&\mathbf{K}(-P+1)+\cdots+\mathbf{K}(-P+1)\\ \vdots&\ddots&\vdots\\ \mathbf{K}(P-1)+\cdots+\mathbf{K}(P-1)&\cdots&\mathbf{K}(0)+\cdots+\mathbf{K}(0)\end{bmatrix}$$ $$\propto\begin{bmatrix}\mathbf{K}(0)&\mathbf{K}(-1)&\cdots&\mathbf{K}(-P+1)\\ \mathbf{K}(1)&\mathbf{K}(0)&\cdots&\mathbf{K}(-P+2)\\ \vdots&\vdots&\ddots&\vdots\\ \mathbf{K}(P-1)&\mathbf{K}(P-2)&\cdots&\mathbf{K}(0)\end{bmatrix}.$$
$$(17)$$
$$(18)$$

Following the same way, we can also represent WV⊤ ∈ R
N×NP and WW⊤ ∈ R
N×N using K:

$${\bf W}{\bf V}^{\top}\propto\left[{\bf K}(P)\quad{\bf K}(P-1)\quad\ldots\quad{\bf K}(1)\right],$$ $${\bf W}{\bf W}^{\top}\propto{\bf K}(0).$$

If the computation of G in Eq.8 leads to numerical issues because VV⊤ has singular values that are nearly zero, a more numerically stable approach is to rewrite VV⊤ by Cholesky factorization:

$$\mathbf{D}=\begin{bmatrix}\mathbf{V}\mathbf{V}^{\top}&\mathbf{V}\mathbf{W}^{\top}\\ \mathbf{W}\mathbf{V}^{\top}&\mathbf{W}\mathbf{W}^{\top}\end{bmatrix}=\mathbf{L}\mathbf{L}^{\top},\quad\mathbf{L}=\begin{bmatrix}\mathbf{L}_{1}&0\\ \mathbf{L}_{2}&\mathbf{L}_{3}\end{bmatrix},\quad\mathbf{V}\mathbf{V}^{\top}=\mathbf{L}_{1}\mathbf{L}_{1}^{\top},\tag{1}$$

where D ∈ R
N(P +1)×N(P +1), WV⊤ = L2L
⊤
1, WW⊤ ∝ K(0), and L1 ∈ R
NP ×NP , L2 ∈ R
N×NP , L3 ∈ R
N×N are the sub-matrices of L. In practice, Eq. 18 factorizes D + δI with a small postive number δ to ensure the positive definite of D. Then, the estimation for G can be cast in the form of L and the measurement matrix Q is the residual covariance of residual R:

$$\hat{\mathbf{G}}=\mathbf{W}\mathbf{V}^{\top}(\mathbf{V}\mathbf{V}^{\top})^{-1}=\mathbf{L}_{2}\mathbf{L}_{1}^{-1},$$ $$\hat{\mathbf{Q}}=\frac{(\mathbf{W}-\hat{\mathbf{G}}\mathbf{V})(\mathbf{W}-\hat{\mathbf{G}}\mathbf{V})^{\top}}{T-P-1}=\frac{\mathbf{L}_{3}\mathbf{L}_{3}^{\top}}{T-P-1}.\tag{1}$$
$$(19)$$

## B. Details For Kalman Em

We consider the linear–Gaussian state–space model

$\mathbf{x}_{t}=\mathbf{A}_{t}\mathbf{x}_{t-1}+\mathbf{w}_{t},\qquad\mathbf{w}_{t}\sim\mathcal{N}(0,\mathbf{Q}_{t}),\qquad\mathbf{y}_{t}=\mathbf{H}\mathbf{x}_{t}+\mathbf{v}_{t},\qquad\mathbf{v}_{t}\sim\mathcal{N}(0,\mathbf{V}),$
where the transition matrix At and process-noise covariance Qt change with time. Let Θ = {A1:T , Q1:T , H, V} denote the full parameter set. E–step With parameters fixed at Θk, the expected complete-data log-likelihood is

Q(Θ | Θ k) = Ex|y,Θk-log p(x, y | Θ) ∝ − 1 2 X T t=1 y ⊤ t V−1yt − 2y ⊤ t V−1H xet + TrH⊤V−1HVt − 1 2 X T t=1 TrQ−1 t Ct− 2 TrQ−1 t AtC⊤ t,t−1 + TrA⊤ t Q−1 t AtCt−1 − T 2 log |V| − 12 X T t=1 log |Qt|. (20)
$$(20)$$
$$(21)$$
$$(22)$$
Here
$\widetilde{\mathbf{x}}_{t}=\mathbb{E}[\mathbf{x}_{t}\mid\mathbf{y},\Theta^{k}],\quad\mathbf{C}_{t}=\mathbb{E}[\mathbf{x}_{t}\mathbf{x}_{t}^{\top}\mid\mathbf{y},\Theta^{k}],\quad\mathbf{C}_{t,t-1}=\mathbb{E}[\mathbf{x}_{t}\mathbf{x}_{t-1}^{\top}\mid\mathbf{y},\Theta^{k}],$
which are obtained with a Kalman filter followed by a Rauch–Tung–Striebel smoother run with the time-varying parameters (Boots, 2009). The individual expectations that appear in (20) expand to

E[x ⊤ t H⊤V−1Hxt] = xe ⊤ t H⊤V−1Hxet + TrH⊤V−1HCt, (21) E[x ⊤ t Q−1 t xt] = xe ⊤ t Q−1 t xet + TrQ−1 t Ct , (22) E[x ⊤ t Q−1 t Atxt−1] = xe ⊤ t Q−1 t Atxet−1 + TrQ−1 t AtC⊤ t,t−1 , (23) E[x ⊤ t−1A⊤ t Q−1 t Atxt−1] = xe ⊤ t−1A⊤ t Q−1 t Atxet−1 + TrA⊤ t Q−1 t AtCt−1. (24)
M–step The M–step maximises (20) with respect to Θ,

$$(23)$$
$$(24)$$
$$\Theta^{k+1}=\arg\operatorname*{max}_{\Theta}Q(\Theta\mid\Theta^{k}),$$

where the paramerer sets Θ is updated either via gradient descent or in closed form, depending on the specific component.

## C. Generation Samples

To better understand the effect of different P values, we generate samples with T = 200 time bins from our model using various P values. Figure 4 shows that when P is very small (e.g., P = 1), the generated samples appear unsmooth. However, for P ≥ 2, the generated samples exhibit no noticeable visual differences.

P=1 P=2 P=3 P=4
Figure 4. Generated samples from our model with MOSE kernel when P = 1, 2, 3, 4.

| Table 1. MSE for GP regression with single-output kernels.   |           |            |           |           |           |
|--------------------------------------------------------------|-----------|------------|-----------|-----------|-----------|
| Reg-MSE / 10−1                                               | Exp       | Matern 3/2 | SE        | RQ        | SM        |
| GP                                                           | 5.7 ± 0.1 | 5.9 ± 0.2  | 3.1 ± 0.1 | 3.0 ± 0.1 | 3.0 ± 0.2 |
| SSM-Approx                                                   | 5.9 ± 0.1 | 6.2 ± 0.1  | 3.3 ± 0.1 | 3.4 ± 0.1 | 3.3 ± 0.2 |

| Table 2. MSE for GP regression with multi-output kernels.   |            |            |            |             |
|-------------------------------------------------------------|------------|------------|------------|-------------|
| Reg-MSE / 10−1                                              | MOSE       | MOSM       | CSM        | LMC         |
| GP                                                          | 7.4 ± 0.02 | 7.5 ± 0.02 | 8.2 ± 0.05 | 0.66 ± 0.02 |
| SSM-Approx                                                  | 7.6 ± 0.04 | 7.9 ± 0.08 | 7.7 ± 0.09 | 0.72 ± 0.02 |

## D. Gaussian Process Regression

To verify the universal connection between arbitrary temporally stationary Gaussian Processes (GPs) and State Space Models (SSMs), we compare GP regression performance using our SSM approximation and the standard GP. We generate samples of 300 points from a GP with added Gaussian noise as regression data. The samples are then randomly split into training
(ttrain, ytrain) and testing (ttest, ytest) sets, with 60% used for training and 40% for testing.

The kernels we evaluated are:
- **Exponential (Exp)**: Single-output with K(*t, t*′) = σ 2exp −
|t−t
′| l
.

- **Matern 3/2 (Matern)**: Single-output with K(*t, t*′) = σ 21 +
√3|t−t
′| l exp −
√3|t−t
′| l
.

- **Squared Exponential (SE)**: Single-output with K(*t, t*′) = σ 2exp −
(t−t
′)
2 2l 2
.

- **Rational Quadratic (RQ)**: Single-output with K(*t, t*′) = σ 21 + (t−t
′)
2 2αl2
−α.

- **Spectral Mixture (SM)** (Wilson & Adams, 2013): Single-output with K(*t, t*′) = PQ
q=1 σ 2 qexp −
(t−t
′)
2 2l 2q cos (ωq(t − t
′)).

- **Multi-Output Squared Exponential (MOSE)** (Gokcen et al., 2022): Multi-output with Kij (*t, t*′) =
σ 2 ij exp −
(t−t
′+δij )
2 2l 2 ij 
.

- **Multi-Output Spectral Mixture (MOSM)** (Parra & Tobar, 2017): Multi-output with Kij (*t, t*′) = PQ
q=1 σ 2 ij,q exp −
(t−t
′+δij,q)
2 2l 2 ij,q cos (ω*ij,q*(t − t
′) + ϕ*ij,q*).

- **Cross-Spectral Mixture (CSM)** (Ulrich et al., 2015): Multi-output with Kij (*t, t*′) = PQ
q=1 PR
r=1 σ r i,qσ r j,q exp −
(t−t
′)
2 2l 2 ij,q cos ω*ij,q*(t − t
′) + ϕ r ij,q.

- **Linear Model of Coregionalization (LMC)**: Multi-output with K(*t, t*′) = PQ
q=1 Bq ⊗ kq(*t, t*′), where Bq is a coregionalization matrix and kq(*t, t*′) is a single-output kernel.

The number of orders P for each kernels are as follows: - **Exponential (Exp)**: P = 1. - **Matern 3/2 (Matern)**: P = 2.

- **Squared Exponential (SE)**: P = 2. - **Rational Quadratic (RQ)**: P = 4. - **Spectral Mixture (SM)**: P = 2. - **Multi-Output Squared Exponential (MOSE)**: P = 2. - **Multi-Output Spectral Mixture (MOSM)**: P = 2.

- **Cross-Spectral Mixture (CSM)**: P = 4.

- **Linear Model of Coregionalization (LMC)**: When kq(*t, t*′) is SE kernel, P = 2.

The results are shown in Table 1 and Table 2, where our SSM approximation demonstrates regression performance comparable to GP in terms of MSE.

## E. Additional Across- And Within-Region Latent Variables

Figure 5 presents the within-region neural activity for both synthetic data and V1-V2 neural spike trains.

(A)

```
region 1
region 2
            true 
            true 

```

within-region activity for synthetic data
(B)

V1 V2 within-region activity 1 within-region activity 2
(C)

VISp VISrl VISal VISpm VISam across-region activity 1,2,3 within-region activity 1

## F. Factor Analysis For Five Regions Spike Trains

Figure 6 presents the Factor Analysis results for neural spike trains from five regions, and we select the latent size to be the largest optimal latent size across five regions, which is 4.

VSp vsr VISƏI
-40.

-98-999
-28.4
−40.

-99
-28.5 40.9 test log-like lihood
-92 ing like Thood
-28.6
..

−9.4 o - like
-28.7
-41.1
-99 11 -28.0 test
- 41.2
− 99.6
−41.3
-28.9
-1000
-41.4
-29.0
-100.2 10 10 10 B
6 8
.

2.2 o s 6 17
-
10.

i 18 size Istent
: sin Isten VISpm VSam
-53-1
-59.

−53.2 −53.3 59.2 test log-likelihood
 1956 - 1956.

test log likelihood

-59.

5–53.6
−53.7
−53.8
-59.0
-53.9 10 10 o 2 latent six size

## G. More Synthetic Data Experiments

(A)
delays between region 1 and 2delays between region 1 and 3delays between region 1 and 4 delays between region 1 and 5 delays between region 2 and 3 tempo ral del ays (ti me bi ns)
delays between region 2 and 4delays between region 2 and 5delays between region 3 and 4 delays between region 3 and 5 delays between region 4 and 5 20 time bins estimated delays ground truth
Figure 7. Estimated Pairwise Temporal Delays for Synthetic Data with Five Regions. Dashed lines indicate the ground truth delays, red lines show the estimated delays, and shaded areas represent the variance across different runs. The results demonstrate that the model accurately recovers the true delays. The increased MSE observed in Figure 8(A) is attributed to amplitude variability, which is not a practical concern, as the temporal delay patterns are well preserved.

(A) (B) 1.0 0.04 Pears on's CC
1.0 4 latents6 latents8 latents 10 latents Pearson
's CC
0.04 0.9 0.9 0.02 0.03 0.02 0.03 0.8 0.8 MSE
MSE
0.7 0.7 0.01 0.01 0.6 0.6 0.0 0.5 0.5 0.0 
(C)
2 regions3 regions4 regions5 regions 2 regions3 regions4 regions5 regions 4 latents6 latents8 latents 10 latents
(D)
1.0 0.04 correctly specified under-specifiedover-specified test lo g-likeli hood 500k 400k 300k 200k 100k 0 0.9 0.02 0.03 Pearson
's CC
0.8 MSE
0.7 0.01 0.6 0.0 0.5 200 400 600 800 time points 200 400 600 800 time points
(A)
500k 350k test log-likelih ood 450k 400k MRM-GP
ADM
mp-srSLDS
DLAG
H. Additional Results for Decoding Visual Stimuli in the Five-Region Brain Dataset

(A) 0.8 latent variablewithin latent variable test cla ssificati on accu racy 0.6 0.4 0.2 0 all dataacross 
Figure 10. Decoding Visual Stimulus Orientation from the Visual Rostrolateral Area (VISrl). We analyze neural data from five regions and extract the learned latent variables. A linear decoder is then used to classify the orientation of visual stimuli (0
◦, 90◦, and 135◦)
presented to the mouse during data collection. We evaluate decoding performance using three inputs: (1) raw observed neural activity from Visual Rostrolateral Area, (2) across-region latent variables (representing the communication subspace) of Visual Rostrolateral Area, and (3) within-region latent variables of Visual Rostrolateral Area. The results show that decoding directly from the observed neural data yields the highest test classification accuracy. Among the latent spaces, the communication subspace achieves higher accuracy than the within-region subspace. Notably, both the observed data and the communication subspace perform above random guessing, indicating that orientation information is preserved in the communication subspace of the Visual Rostrolateral Area, which is a region known to be involved in motion and spatial processing. Finally, the drop in accuracy from the observed data to the communication subspace is likely due to information loss from dimensionality reduction.