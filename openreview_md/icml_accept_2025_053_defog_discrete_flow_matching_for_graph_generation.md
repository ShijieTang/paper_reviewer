# 

Yiming Qin * 1 **Manuel Madeira** * 1 Dorina Thanou 1 **Pascal Frossard** 1

## Abstract

Graph generative models are essential across diverse scientific domains by capturing complex distributions over relational data. Among them, graph diffusion models achieve superior performance but face inefficient sampling and limited flexibility due to the tight coupling between training and sampling stages. We introduce DeFoG, a novel graph generative framework that disentangles sampling from training, enabling a broader design space for more effective and efficient model optimization. DeFoG employs a discrete flow-matching formulation that respects the inherent symmetries of graphs. We theoretically ground this disentangled formulation by explicitly relating the training loss to the sampling algorithm and showing that DeFoG faithfully replicates the ground truth graph distribution. Building on these foundations, we thoroughly investigate DeFoG's design space and propose novel sampling methods that significantly enhance performance and reduce the required number of refinement steps. Extensive experiments demonstrate state-of-the-art performance across synthetic, molecular, and digital pathology datasets, covering both unconditional and conditional generation settings. It also outperforms most diffusion-based models with just 5–10% of their sampling steps.

## 1. Introduction

Graph generation has become a fundamental task across diverse fields, from molecular chemistry to social network analysis, due to graphs' capacity to represent complex relationships and generate realistic structured data. Diffusion-based graph generative models (Niu et al., 2020; Jo et al., 2022), particularly those tailored for discrete data (Vignac et al., 2022), have emerged as compelling ap-
*Equal contribution 1EPFL, Lausanne, Switzerland. Correspondence to: Yiming Qin <yiming.qin@epfl.ch>, Manuel Madeira <manuel.madeira@epfl.ch>.

proaches, demonstrating pioneering performance in applications such as molecular generation (Irwin et al., 2024), reaction pathway design (Igashov et al., 2024), neural architecture search (Asthana et al., 2024), and combinatorial optimization (Sun & Yang, 2024). Recently, continuoustime discrete diffusion frameworks have further advanced the domain of discrete graph diffusion (Xu et al., 2024; Siraudin et al., 2024). These frameworks leverage the robustness and flexibility of continuous-time modeling, while preserving the natural alignment with the discrete structure of graphs. Despite their state-of-the-art performance, the training and sampling stages of diffusion-based models is tightly entangled, restricting sampling options to training-phase choices. Thus, optimizing components such as noise schedules or rate matrices requires re-training for each configuration, resulting in prohibitive computational costs. Consequently, these models often adopt a single configuration across graph datasets. This one-size-fits-all approach fails to accommodate the diverse structural characteristics of different datasets, leaving room for further improvement. In this work, we present DeFoG, a novel graph generative framework that disentangles the training and sampling stages (Figure 1a), addressing the inefficiencies in graph diffusion models and achieving state-of-the-art (SOTA)
performance. DeFoG leverages a discrete flow matching
(DFM) inspired formulation (Campbell et al., 2024) that we tailor to graph settings. It features a linear interpolation noising process and a continuous-time Markov chain (CTMC)-based denoising process, while ensuring node permutation equivariance and addressing the model expressivity limitations inherent to this data modality (Morris et al., 2019). We demonstrate that training-sampling decoupling not only enhances flexibility but is also provably sound. By theoretically establishing that training loss optimization leads to improved sampling dynamics, DeFoG enables faithful replication of the ground truth graph distribution. To navigate the expanded design space enabled by such disentanglement, we take a critical step by "defogging" this space. Specifically, we explore and propose various sampling methods, including time-adaptive methods and modifications to CTMC rate matrices, to better govern denoising trajectories and align with the unique characteristics of graph datasets.

1

State 2 4 1/3 1/3 1/3 3 5 2 4 1 0 0 3 5 p
(n) 0 Noising : p(n)
t|1 
=(1 → t) p(n)
0 + t p(n)
1 p(n)
0 p(n)
1 Decoupled Training & Sampling Entangled Continuous Continuous Discrete Discrete Initial Framework Discreteness Exploitation t Time x(n)
t+!t x(n)
t Rω,(n)
t !t p(n)
t+!t|t
+
Flexible Sampling SOTA
Gt Gt+t
+ =Sample DeFoG
 0 xnt Denoising p ω,(n)
1|t (·|Gt)
=
(a) Motivation of DeFoG.

(b) Overview of DeFoG.
Figure 1: (a) DeFoG enhances graph generation by introducing training-sampling decoupling, an orthogonal improvement within graph iterative refinement models, while preserving the sampling flexibility and inherent discreteness exploitation of prior SOTA models. (b) One node, x
(n), is selected to illustrate both *noising* and *denoising* processes. For *noising*,
DeFoG follows a straight path from the one-hot encoding p1 of the clean node to the initial distribution p0. For *denoising*,
a network parameterized by θ predicts the marginal distributions of the clean graph, there the node's distribution p θ,(n)
1|t(·|Gt)
is used to compute its rate matrix R
θ,(n)
t and, subsequently, its probability at the next time point t + ∆t.

Fonts:
- 32 - 24 x(n)
t x(n)
t Rω,(n)
t (xt, ·) !t Rω,(n)
t !t x(n)
t+!t p(xnt+−t| xnt , x̂
nt)
xnt + R⋅(xnt , xnt+−t| x̂
d t )−t = p(xnt+−t| xnt , x̂
n t)
Our experiments show that DeFoG achieves SOTA performance across diverse datasets, with near-saturated validity of 99.5%, 96.5%, and 90% on planar, tree, and stochastic block model (SBM) datasets, respectively. On complex molecular data, it achieves 92.8% validity on MOSES (Polykovskiy et al., 2020), surpassing the previous SOTA of 90.5%. Moreover, DeFoG achieves 95.0% and 86.5% validity on planar and SBM datasets, respectively, with only 5–10% of the sampling steps used by diffusion models. This performance surpasses all but one diffusion model on the planar dataset and ranks best on SBM, highlighting substantial efficiency gains. To further highlight the versatility of DeFoG, we also test it in conditional generation tasks for digital pathology, where it largely outperforms existing unconstrained models. Ablation studies further confirm the need of each proposed sampling method and highlight the importance of dataset-specific sampling procedures to effectively address diverse data characteristics.

xnt R*(xnt , xnt+−t| xn1 )−t p(xnt+−t| xnt + = )
Our main contributions are as follows:
- We introduce DeFoG, a novel graph generative model that effectively exploits the training-sampling decoupling enabled by its flow-based formulation, significantly enhancing sampling flexibility and efficiency.

- We provide a theoretical foundation for both our training and sampling algorithms, validating the soundness of the disentanglement framework;
- We comprehensively explore DeFoG's design space with novel training and sampling approaches, highlight critical configurations for graph data and attain state-of-the-art performance across diverse datasets.

Overall, DeFoG enables more effective graph generation with reduced computational costs under theoretical guarantees, paving the way for broader adoption of graph generative models in real-world applications.

## Xnt R*(Xnt , Xnt+−T | Xn1 )−T P(Xnt+−T | Xnt + = ) 2. Background

In generative modeling, the primary goal is to generate new data samples from the underlying distribution that produced the original data, pdata. An effective approach is to learn a mapping between a simpler distribution pϵ that can be easily sampled, and pdata.

Iterative refinement models achieve this mapping through a stochastic process over the time interval t ∈ [0, 1] for variables in discrete state spaces. For the sake of simplicity, we describe here an univariate formulation. At any time t, we consider a discrete variable with Z possible values, denoted by zt ∈ Z = {1*, . . . , Z*}. The marginal distribution of zt is represented by the vector pt ∈ ∆Z, with
∆Z =
nu ∈ R
Z |PZ
i=1 ui = 1, ui ≥ 0, ∀i o. The initial distribution is set to a predefined noise distribution, p0 = pϵ, while p1 = pdata represents the target data distribution. We refer to the mapping t : 1 → 0 as the *noising* process and t : 0 → 1 as the *denoising* process.

DFM (Campbell et al., 2024) builds upon a streamlined noising process. In particular, the noising trajectory pt|1(zt|z1) ∈ [0, 1] is defined through a simple linear interpolation starting from a chosen datapoint z1:

$$p_{t|1}(z_{t}|z_{1})=t\,\delta(z_{t},z_{1})+(1-t)\,p_{0}(z_{t}),\qquad(1)$$

where δ(zt, z1) is the Kronecker delta (1 when zt = z1).

A usual choice for the initial distribution is the uniform distribution over the state space, p0 = [1*/Z, . . . ,* 1/Z].

In the *denoising* stage, DFM leverages a CTMC formulation. In general, a CTMC is characterized by an initial distribution, p0, and a *rate matrix*, Rt ∈ R
Z×Z that governs its evolution across time t ∈ [0, 1]. Specifically, the rate matrix defines the instantaneous transition rates between states, such that:

$$p_{t+{\rm d}t|t}(z_{t+{\rm d}t}|z_{t})=\delta(z_{t},z_{t+{\rm d}t})+R_{t}(z_{t},z_{t+{\rm d}t}){\rm d}t,\tag{2}$$

where Rt(zt, zt+dt) denotes an entry in the rate matrix. Intuitively, Rt(zt, zt+dt)dt yields the probability that a transition from state zt to state zt+dt will occur in the next infinitesimal time step dt. By definition, we have Rt(zt, zt+dt) ≥ 0 for zt ̸= zt+dt. Consequently, we further have Rt(zt, zt) = −Pzt+dt̸=zt Rt(zt, zt+dt) to ensure normalization Pzt+dt pt+dt|t(zt+dt|zt) = 1. Under this definition, the marginal distribution and the rate matrix of a CTCM are related by a conservation law, the Kolmogorov equation, given by ∂tpt = R⊤
t pt. If expanded, this expression unveils the time derivative of the marginal distribution as the net balance between the inflow and outflow of probability mass at that state. Similarly to the noising process of Eq. (1), the denoising is also performed under conditioning on z1. Specifically, we consider a z1-conditional rate matrix, Rt(· , ·|z1) ∈ R
Z×Z,
that will govern the denoising in DFM. Under mild assumptions, Campbell et al. (2024) present a closed-form for a valid conditional rate matrix, i.e., a matrix that verifies the corresponding Kolmogorov equation, for zt ̸= zt+dt, defined as:

$$R_{t}^{*}(z_{t},z_{t+\mathrm{d}t}|z_{1})=\frac{\mathrm{ReLU}[\partial_{t}p_{t|1}(z_{t+\mathrm{d}t}|z_{1})-\partial_{t}p_{t|1}(z_{t}|z_{1})]}{Z_{t}^{>0}\ p_{t|1}(z_{t}|z_{1})}\tag{3}$$

and Z
>0 t = |{zt : pt|1(zt|z1) > 0}|. Again, normalization is performed for the case zt+dt = zt. Intuitively, R∗
t applies a positive rate to states needing more mass than the current state zt (details in Appendix B.5). Finally, it can be shown that Rt(zt, zt+dt) = Ep1|t(z1|zt)[Rt(zt+dt, zt|z1)],
which is employed in Eq. (2) for denoising. While the DFM paradigm enables training-sampling disentanglement, it lacks a complete formulation and empirical validation on graph data. Moreover, how to effectively leverage this disentanglement to enhance sampling performance remains underexplored, particularly for graphspecific tasks. We introduce DeFoG to address these gaps.

## 3. Defog Framework

In this section, we present DeFoG (Discrete Flow Matching on Graphs), a novel iterative refinement framework for graph generation that leverages the decoupling of training and sampling stages. We begin by describing its noising and denoising processes, highlighting how they enable this disentanglement, as illustrated in Figure 1b. We theoretically demonstrate that this flexible framework is also robust by proving that optimizing the training loss improves sampling dynamics, ensuring that DeFoG can faithfully replicate graph distributions. Then, we discuss the expanded design space enabled by DeFoG's disentanglement, which drives key improvements in graph generation performance. Finally, we establish the node permutation equivariance/invariance guarantees of DeFoG.

## 3.1. Learning Discrete Flows Over Graphs

We instantiate undirected graphs with N nodes as G = (x 1:n:N , e1:i<j:N ), where x 1:n:N = (x
(n))1≤n≤N and e 1:i<j:N = (e
(ij))1≤i<j≤N denote the node and edge sets, respectively, with x
(n) ∈ X = {1*, . . . , X*} and e
(ij) ∈
E = {1*, . . . , E*}. Following standard practice in the field
(Vignac et al., 2022; Xu et al., 2024; Siraudin et al., 2024), we consider an edge between every pair of nodes, where one of the edge categories explicitly represents the absence of an edge (i.e., a "non-existing" edge).) Noising We now define the noising process of DeFoG. According to Eq. (1), with shared initial distributions across nodes and edges, denoted as pX
0and pE
0, respectively, we formulate the noising trajectory by independently adding noise to each node and each edge:

$p_{t|1}(G_{t}|G_{1})=\prod_{n}p_{t|1}\left(x_{t}^{(n)}|x_{1}^{(n)}\right)\prod_{i<j}p_{t|1}\left(e_{t}^{(ij)}|e_{1}^{(ij)}\right).$  Different $p_{0}^{\chi}$ and $p_{0}^{\xi}$ are further discussed in Appendix C.1.  
Sampling As formulated in Sec. 2, the denoising process requires simulating a CTMC, driven by its rate matrix Rt. We start by sampling a purely noisy graph G0 from the predefined initial distribution p0(G0) = Qn pX
0(x
(n)
0)Qi<j pE0(e ij 0). Then, we progress in the denoising process by employing independent Euler steps for each node and edge, with a finite time step ∆t, i.e., we iteratively sample progressively denoised graphs from p˜t+∆t|t(Gt+∆t|Gt), given by:

$$\prod_{n}\tilde{p}^{(n)}_{t+\Delta t|t}(x^{(n)}_{t+\Delta t}|G_{t})\prod_{i<j}\tilde{p}^{(ij)}_{t+\Delta t|t}(e^{(ij)}_{t+\Delta t}|G_{t}).\tag{4}$$

Each term p˜
(n)
t+∆t|t
(x
(n)
t+∆t|Gt) corresponds to the Euler step given in Eq. (2), where the transition dynamics are governed by the rate matrix computed as:

$$R_{t}^{(n)}\left(x_{t}^{(n)},x_{t+\mathrm{d}t}^{(n)}\right)=\mathbb{E}_{p_{1|t}^{(n)}}(x_{1}^{(n)}|G_{t})\left[R_{t}^{(n)}\left(x_{t}^{(n)},x_{t+\Delta t}^{(n)}|x_{1}^{(n)}\right)\right],\tag{5}$$
and similarly for $\tilde{p}_{t+\Delta t|t}^{(ij)}(e_{t+\Delta t}^{(ij)}|G_{t})$.  
Training The rate matrix used in the denoising steps above requires the knowledge of the marginal distributions p
(n)
1|t
(·|Gt) ∈ ∆X and p
(ij)
1|t
(·|Gt) ∈ ∆E for all nodes and all edges, respectively. Both are gathered in p1|t(·|Gt) =
p 1:n:N
1|t(·|Gt), p 1:i<j:N
1|t(·|Gt)
. Each of these components consists of the clear marginal distribution prediction given a noisy graph Gt. However, the computation

| Algorithm 1 DeFoG Training                                                                                                           | Algorithm 2 DeFoG Sampling                           |                                                                                |           |
|--------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|--------------------------------------------------------------------------------|-----------|
| 1: Input: Graph dataset D = {G1 , . . . , GM} 2: while fθ not converged do 3: Sample G ∼ D 4: Sample t ∼ T 5: Sample Gt ∼ pt|1(Gt|G) | ▷ Noising                                            |                                                                                |           |
| 6:                                                                                                                                   | h ← RRWP(Gt)                                         | ▷ Extra features                                                               |           |
| 7:                                                                                                                                   | p θ (·|Gt) ← fθ(Gt, h, t)                            | ▷ Denoising prediction                                                         |           |
| 1|t                                                                                                                                  |                                                      |                                                                                |           |
| 8:                                                                                                                                   | loss ← CEλ(G, p θ (·|Gt)) 1|t                        |                                                                                |           |
| 9:                                                                                                                                   | optimizer.step(loss)                                 | 1: Input: # graphs to sample S 2: for i = 1 to S do 3: Sample N from train set | ▷ # Nodes |
| 4:                                                                                                                                   | Sample G0 ∼ p0(G0)                                   |                                                                                |           |
| 5:                                                                                                                                   | for t = 0 to 1 − ∆t with step ∆t do                  |                                                                                |           |
| 6:                                                                                                                                   | h ← RRWP(Gt)                                         | ▷ Extra features                                                               |           |
| 7:                                                                                                                                   | p θ (·|Gt) ← fθ(Gt, h, t) ▷ Denoising prediction 1|t |                                                                                |           |
| 8:                                                                                                                                   | Gt+∆t ∼ p˜t+∆t|t(Gt+∆t|Gt)                           | ▷ Eq. (4)                                                                      |           |
| 9:                                                                                                                                   | Store G1                                             |                                                                                |           |

of these terms is generally intractable. Instead, we train a neural network, parameterized by θ, to approximate them, p θ 1|t
(·|Gt). To cover different noise levels, we sample t ∼
T , where T is an arbitrary distribution over [0, 1]. In DFM, T is typically set to the uniform distribution, though alternative choices can enhance performance, as later explored in Sec. 3.2. The training loss is naturally formulated as:

$\blacksquare$

$$\mathbf{l}\cdot\mathbf{p}^{\prime}$$
LDeFoG = Et∼T ,p1(G1),pt|1(Gt|G1) CEλ(G1, p

$$r_{1},p_{1|t}^{\theta}(\cdot|G_{t})$$

where CEλ(G1, p θ 1|t(·|Gt)) is defined as:

$$-\sum_{n}\log\left(p_{1|t}^{\theta,(n)}(x_{1}^{(n)}|G_{t})\right)-\lambda\sum_{i<j}\log\left(p_{1|t}^{\theta,(i j)}(e_{1}^{(i j)}|G_{t})\right).$$

Here, λ ∈ R
+ is introduced to weight nodes and edge differently to more flexibly capture varying topologies. Decoupled Training and Sampling DeFoG exhibits a clear disentanglement of training and sampling. The training phase focuses on predicting the marginal probabilities of the clean graph p θ1|t
(·|Gt), while sampling relies on the rate matrix formulation. Importantly, the training process is agnostic to the choice of the z1-conditional rate matrix. Thus, different z1-conditional rate matrices can be employed at sampling time, such as R⋆(·, ·|z1) in Eq. (3).

This decoupling of sampling from training provides additional flexibility in DeFoG's design, which can be leveraged to further enhance performance. Notably, we further demonstrate that, upon this decoupling, optimizing De- FoG's training loss improves its sampling dynamics, ensuring the soundness of our framework. Corollary 1 (Bounded estimation error of rate matrix for graphs). Given t ∈ [0, 1] and graphs Gt, Gt+dt*, and* G1 ∼
p1(G1), there exist constants C¯0, C¯1, C¯3 > 0 *such that the* rate matrix estimation error can be upper bounded by:

$$|R_{t}(G_{t},G_{t+\mathrm{d}t})-R_{t}^{\theta}(G_{t},G_{t+\mathrm{d}t})|^{2}\leq\bar{C}_{0}+$$
$$+\,\bar{C}_{1}\,\mathbb{E}_{p_{1}(G_{1})}\left[p_{t|1}(G_{t}|G_{1})\sum_{n}-\log p_{1|t}^{\theta,(n)}(x_{1}^{(n)}|G_{t})\right]$$
$$+\,\bar{C}_{2}\,\mathbb{E}_{p_{1}(G_{1})}\,\left[p_{t|1}(G_{t}|G_{1})\sum_{i<j}-\log p_{1|t}^{\theta,(i j)}(e_{1}^{(i j)}|G_{t})\right]$$

 .
4 By taking the expectation over t ∼ T and summing over Gt, minimizing the derived upper bound of rate matrix estimation error in Theorem 1 with respect to θ corresponds directly to minimizing the loss function of DeFoG with λ = 1. Therefore, we guarantee that our training loss minimization is aligned with accurate rate matrix estimation.

$\sigma$
Upon this result, we are now in conditions of justifying De-
FoG's approximated denoising algorithm.

Corollary 2 (Bounded deviation of the generated graph distribution). Let p1 *be the marginal distribution at* t = 1 of a groundtruth CTMC, {Gt}0≤t≤1*, and* p˜1 be the marginal distribution at t = 1 of its independentdimensional Euler sampling approximation, with a maximum step size ∆t. Then, under Theorem 6, the following total variation bound holds:

$$\|p(G_{1})-p_{data}\|_{TV}\leq\bar{U}\left(XN+E\frac{N(N-1)}{2}\right)$$ $$+\bar{B}\left(XN+E\frac{N(N-1)}{2}\right)^{2}\Delta t+\ O(\Delta t),$$

where U¯ and B¯ *are constant upper bounds for the bound* from Theorem 1 and for the denoising process relative to its noising counterpart, respectively, for any t ∈ [0, 1].

The first term of the bound captures the estimation error introduced by using the neural network approximation Rθ t(Gt, Gt+dt). From Theorem 1, this term is bounded.

The remaining terms arise from the discretization of the CTMC and can be controlled by reducing the step size ∆t. Consequently, the deviation introduced by this approximation can be made arbitrarily small, ensuring that the generated distribution remains close to the groundtruth and validating our graph sampling scheme. The resulting training and sampling processes are detailed in Algs. 1 and 2. The proofs of Theorems 1 and 2 are provided in Appendix D.1.

## 3.2. Design Space Of Defog

As described in the previous section, DeFoG benefits from greater flexibility due to its training-sampling decoupling.

For example, it allows the number and size of sampling steps to be adjusted dynamically, unlike the fixed steps in discrete-time diffusion (Vignac et al., 2022), and enables adjustment of the rate matrix without retraining, addressing limitations of continuous-time diffusion graph models (Siraudin et al., 2024; Xu et al., 2024). This decoupling supports extensive, training-free performance optimization during the sampling stage, which is crucial for improving performance and reducing the number of sampling steps. Below, we propose the key components of DeFoG that are enabled by this disentanglement. Sample Distortion In DFM's sampling process, the discretization is performed using equally sized time steps (Alg. 2, line 5). However, this uniformity may fail to preserve key properties during critical intervals where finer control is needed. For instance, smaller steps are essential near the final stages of sampling to prevent sudden edge alterations that could compromise global properties such as planarity. To overcome this limitation, we propose using variable step sizes, allocating smaller, more frequent steps during these critical intervals to better capture essential graph characteristics. Specifically, we apply to each timestep t a bijective, increasing *distortion function* f defined for t ∈ [0, 1], yielding t′ = f(t). For example, the choice f(t) = 2t−t 2(referred to as *polydec*) creates monotonically decreasing step sizes, emphasizing the final stages of sampling, where error correction can be most critical. The specific distortion functions employed are described in Appendix B.1. Importantly, we can efficiently (i.e., without re-training) adjust the sample distortion adopted for each dataset to better accommodate its graphs characteristics, leading to significant performance improvements. Train Distortion Once the optimal sampling distortion is identified, it can guide training by highlighting the critical time ranges for graph generation in a specific dataset. This enables adjustments to the training distribution T , skewing it toward these ranges to focus the model's capacity on the most relevant regions. The skewed distributions are obtained by passing uniformly sampled times t through the same distortion functions. While similar strategies in other modalities, such as image generation (Esser et al., 2024), often emphasize intermediate time ranges, we find that optimal time ranges in graph generation vary across datasets.

Aligning the distortion function in training with sampling typically enhances the algorithm by focusing on critical time ranges. For instance, for larger datasets, such as drugsized molecular graphs, the *polydec* distortion function accelerates convergence significantly and provides noticeable performance improvements. Target Guidance The application of time distortions is not the sole avenue for optimizing the sampling process; the design of the conditional rate matrices also offers significant potential for improvement. One promising direction arises from the goal of better guiding the generation process toward the clean data distribution (Song et al., 2020a). This also aligns with the fundamental design of diffusion and flow matching models, which are structured to predict clean data directly and subsequently use that prediction to generate the denoising trajectory (Ho et al., 2020; Lipman et al., 2023; Vignac et al., 2022). Inspired by these principles, we propose an alternative sampling mechanism that seeks to further amplify the influence of the denoising neural network's predictions in the designed rate matrices, by setting Rt(zt, zt+dt|z1) = R∗
t(zt, zt+dt|z1) +
Rω t(zt, zt+dt|z1) for zt ̸= zt+dt, such that:

$$R_{t}^{\omega}(z_{t},z_{t+\mathrm{d}t}|z_{1})=\omega\frac{\delta(z_{t+\mathrm{d}t},z_{1})}{Z_{t}^{>0}\;p_{t|1}(z_{t}|z_{1})}.\tag{6}$$

This adjustment increases the weight of transitions in the rate matrix when zt+dt = z1, where z1 is the predicted clean data. While moderate increases in ω significantly enhances performance by steering the generation toward high confidence domains, excessively high ω leads to performance drop. This behavior is explained in Theorem 10, in Appendix B.2, where we show that target guidance introduces an O(ω) violation of the Kolmogorov equation. Consequently, finding an optimal value for ω is essential. Stochasticity Orthogonal to target guidance, there also exists unexplored potential in the design space of conditional rate matrices that preserve the Kolmogorov equation as the standard formulation of R∗
t(zt, zt+dt|z1) does not fully capture this space. As demonstrated by Campbell et al. (2024), for any rate matrix RDB
tsatisfying the detailed balance condition pt|1(zt|z1)RDB
t(zt, zt+dt|z1) =
pt|1(zt+dt|z1)RDB
t(zt+dt, zt|z1), the modified rate matrix R
η t = R∗
t + ηRDB
t, with η ∈ R
+, is also valid. Intuitively, increasing η facilitates more transitions to other states while reducing the likelihood of remaining in the same state, thereby increasing stochasticity in the trajectory of the denoising process. This approach can be interpreted as a correction mechanism, as it draws transitions back to states that would otherwise be forbidden according to the rate matrix formulation, as described in Appendix B.5. Additionally, different designs of RDB
tencode different priors for preferred transitions between states, which we investigate in detail in Appendix B.3.

## 3.3. Permutation Invariance Guarantees

Graph generative models should respect the inherent permutation symmetries of graphs. Accordingly, DeFoG's training and sampling algorithms should be independent of node ordering. This requires that both DeFoG's loss function during training and its probability of generating a specific graph during sampling be permutation invariant. We formally demonstrate these results in Theorem 3, whose proof is in Appendix D.2.1.

Lemma 3 (Node Permutation Equivariance and Invariance Properties of DeFoG). For any permutation equivariant denoising neural network, the loss function of DeFoG is permutation invariant, and its sampling probability is permutation invariant. We further describe the permutation equivariant denoising neural network of DeFoG in Sec. 5.

## 4. Related Work

Graph Generative Models Graph generation has applications across various domains, including molecular generation (Mercado et al., 2021), combinatorial optimization (Sun & Yang, 2024), and inverse protein folding (Yi et al., 2024). Existing methods for this task generally fall into two main categories. First, *autoregressive* models progressively grow the graph by inserting nodes and edges (You et al., 2018; Liao et al., 2019). Although these methods offer high flexibility in sampling and facilitate the integration of domain-specific knowledge (e.g., for molecule generation, Liu et al. (2018) perform valency checks at each iteration), they suffer from a fundamental drawback: the need to learn a node ordering (Kong et al., 2023; Han et al., 2023), or use a predefined node ordering (You et al., 2018) to avoid the overly large learning space. In contrast, *one-shot* models circumvent such limitation by predicting the entire graph in a single step, enabling the straightforward incorporation of node permutation equivariance/invariance properties. Examples of these approaches include graph-adapted versions of VAEs (Kipf & Welling, 2016), GANs (De Cao & Kipf, 2018), or normalizing flows (Liu et al., 2019). Among oneshot methods, diffusion models have gained prominence for their state-of-the-art performance, attributed to their iterative mapping between noise and data distributions. Graph Diffusion One of the initial research directions in graph diffusion sought to adapt continuous diffusion frameworks (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020b) for graph-structured data (Niu et al., 2020; Jo et al., 2022; 2024). Those however faced challenges in preserving the inherent discreteness of graphs. In response, discrete diffusion models (Austin et al., 2021) were effectively extended to the graph domain (Vignac et al., 2022; Haefeli et al., 2022), utilizing Discrete-Time Markov Chains to model the stochastic diffusion process. However, this method restricts sampling to the discrete time points used during training. To address this limitation, continuous-time discrete diffusion models incorporating CTMCs have emerged (Campbell et al., 2022), and have been recently applied to graph generation (Siraudin et al., 2024; Xu et al., 2024). Despite employing a continuoustime framework, their optimization space is constrained by training-dependent choices like fixed-rate matrices, limiting further performance gains.

Discrete Flow Matching Flow matching (FM) models emerged as a compelling alternative to diffusion models among iterative refinement generative approaches for continuous state spaces (Lipman et al., 2023; Liu et al., 2023). FM frameworks have been empirically shown to enhance performance and efficiency in image generation (Esser et al., 2024; Ma et al., 2024). To address discrete state spaces, a DFM formulation has been introduced (Campbell et al., 2024; Gat et al., 2024). This approach streamlines its diffusion counterpart by employing a linear interpolation noising process and a more flexible CTMC-based denoising process, whose rate matrices, unlike diffusion models, need not be fixed during training. While other flow-based formulations for graphs have been proposed (Eijkelboom et al., 2024), DeFoG stands out as the first DFM-based model for graphs, leveraging a training-sampling decoupled formulation for improved performance.

## 5. Experiments

This section highlights DeFoG's SOTA performance, enabled by its highly decoupled framework and effective sampling methods. We present DeFoG's performance in generating graphs with diverse topological structures and in molecular datasets with rich prior information. We also provide ablations to showcase the effectiveness and necessity of each proposed sampling method. We highlight the best result and the second-best method. Results for generation with 5-10% of steps used by previous SOTA diffusion models are also provided to demonstrate DeFoG's sampling efficiency. We show DeFoG's versatility on conditional generation for digital pathology in Appendix G.1.

1 Setup To isolate the effect of the network architecture, we build DeFoG on the best-performing graph transformer on generative tasks (Vignac et al., 2022). To enhance expressivity, we incorporate Relative Random Walk Probabilities (RRWP) (Ma et al., 2023; Siraudin et al., 2024) as node and edge features. More details on the architecture in Appendix F.1. Our ablations in Appendix G.4 show that, while RRWP features encode structural properties more efficiently and effectively than prior alternatives, DeFoG's disentangled framework is the primary driver of performance gains. Importantly, the overall architecture, together with RRWP features, is guaranteed to be permutation equivariant (see Appendix D.2.1).

## 5.1. Graph Generation Performance

Synthetic Graph Generation We evaluate DeFoG using the widely adopted Planar, SBM (Martinkus et al., 2022), and *Tree* datasets (Bergmeister et al., 2023), along with the associated evaluation methodology. In Tab. 1, we report the proportion of generated graphs that are valid, unique, 1Code at github.com/manuelmlmadeira/DeFoG.

| Planar                               | Tree           | SBM      |         |          |         |          |         |
|--------------------------------------|----------------|----------|---------|----------|---------|----------|---------|
| Model                                | Class          | V.U.N. ↑ | Ratio ↓ | V.U.N. ↑ | Ratio ↓ | V.U.N. ↑ | Ratio ↓ |
| Train set                            | -              | 100      | 1.0     | 100      | 1.0     | 85.9     | 1.0     |
| GraphRNN (You et al., 2018)          | Autoregressive | 0.0      | 490.2   | 0.0      | 607.0   | 5.0      | 14.7    |
| GRAN (Liao et al., 2019)             | Autoregressive | 0.0      | 2.0     | 0.0      | 607.0   | 25.0     | 9.7     |
| SPECTRE (Martinkus et al., 2022)     | GAN            | 25.0     | 3.0     | -        | -       | 52.5     | 2.2     |
| DiGress (Vignac et al., 2022)        | Diffusion      | 77.5     | 5.1     | 90.0     | 1.6     | 60.0     | 1.7     |
| EDGE (Chen et al., 2023)             | Diffusion      | 0.0      | 431.4   | 0.0      | 850.7   | 0.0      | 51.4    |
| BwR (EDP-GNN) (Diamant et al., 2023) | Diffusion      | 0.0      | 251.9   | 0.0      | 11.4    | 7.5      | 38.6    |
| BiGG (Dai et al., 2020)              | Autoregressive | 5.0      | 16.0    | 75.0     | 5.2     | 10.0     | 11.9    |
| GraphGen (Goyal et al., 2020)        | Autoregressive | 7.5      | 210.3   | 95.0     | 33.2    | 5.0      | 48.8    |
| HSpectre (Bergmeister et al., 2023)  | Diffusion      | 95.0     | 2.1     | 100.0    | 4.0     | 75.0     | 10.5    |
| GruM (Jo et al., 2024)               | Diffusion      | 90.0     | 1.8     | -        | -       | 85.0     | 1.1     |
| CatFlow (Eijkelboom et al., 2024)    | Flow           | 80.0     | -       | -        | -       | 85.0     | -       |
| DisCo (Xu et al., 2024)              | Diffusion      | 83.6±2.1 | -       | -        | -       | 66.2±1.4 | -       |
| Cometh (Siraudin et al., 2024)       | Diffusion      | 99.5±0.9 | -       | -        | -       | 75.0±3.7 | -       |
| DeFoG (5% steps)                     | Flow           | 95.0±3.2 | 3.2±1.1 | 73.5±9.0 | 2.5±1.0 | 86.5±5.3 | 2.2±0.3 |
| DeFoG                                | Flow           | 99.5±1.0 | 1.6±0.4 | 96.5±2.6 | 1.6±0.4 | 90.0±5.1 | 4.9±1.3 |

Table 2: Large molecule generation performance. Only iterative denoising-based methods are reported here. Respective full versions in Tab. 10 (Guacamol) and Tab. 9 (MOSES), Appendix G.3.

| Guacamol                       | MOSES   |        |         |          |       |        |           |           |           |       |       |        |
|--------------------------------|---------|--------|---------|----------|-------|--------|-----------|-----------|-----------|-------|-------|--------|
| Model                          | Val. ↑  | V.U. ↑ | V.U.N.↑ | KL div ↑ | FCD ↑ | Val. ↑ | Unique. ↑ | Novelty ↑ | Filters ↑ | FCD ↓ | SNN ↑ | Scaf ↑ |
| Training set                   | 100.0   | 100.0  | 0.0     | 99.9     | 92.8  | 100.0  | 100.0     | 0.0       | 100.0     | 0.01  | 0.64  | 99.1   |
| DiGress (Vignac et al., 2022)  | 85.2    | 85.2   | 85.1    | 92.9     | 68.0  | 85.7   | 100.0     | 95.0      | 97.1      | 1.19  | 0.52  | 14.8   |
| DisCo (Xu et al., 2024)        | 86.6    | 86.6   | 86.5    | 92.6     | 59.7  | 88.3   | 100.0     | 97.7      | 95.6      | 1.44  | 0.50  | 15.1   |
| Cometh (Siraudin et al., 2024) | 98.9    | 98.9   | 97.6    | 96.7     | 72.7  | 90.5   | 99.9      | 92.6      | 99.1      | 1.27  | 0.54  | 16.0   |
| DeFoG (10% steps)              | 91.7    | 91.7   | 91.2    | 92.3     | 57.9  | 83.9   | 99.9      | 96.9      | 96.5      | 1.87  | 0.50  | 23.5   |
| DeFoG                          | 99.0    | 99.0   | 97.9    | 97.7     | 73.8  | 92.8   | 99.9      | 92.1      | 98.9      | 1.95  | 0.55  | 14.4   |

and novel (V.U.N.), as well as the average ratio of the usual distances between graph statistics of the generated and test sets relative to the train and test sets (Ratio) to assess sample quality. As shown in Tab. 1, for the Planar dataset, DeFoG achieves the best performance across both metrics, with a nearly saturated V.U.N. value of 99.5%. On the Tree dataset, it is only surpassed by HSpectre, which leverages a local expansion procedure particularly well-suited to hierarchical structures like trees. On the SBM dataset, DeFoG achieves the highest V.U.N. score among all methods, even with just 50 steps. Molecular Graph Generation Molecular design is a prominent real-world application of graph generation. We evaluate DeFoG's performance on this task using the QM9 (Wu et al., 2018), ZINC250k (Sterling & Irwin, 2015), MOSES (Polykovskiy et al., 2020), and Guacamol (Brown et al., 2019) datasets. For QM9, we follow the dataset split and evaluation metrics from Vignac et al. (2022), presenting the results in Appendix F.2.2, Tab. 8. For ZINC250k, we provide the experimental setup in Appendix F. For the larger MOSES and Guacamol datasets, we adhere to the training setup and evaluation metrics established by Polykovskiy et al. (2020) and Brown et al. (2019), respectively, with results in Tabs. 9 and 10. As illustrated in Tab. 2, on Guacamol, it ranks best across all metrics, achieving a nearly saturated validity of 99.0%. Notably, DeFoG achieves over 90% validity with just 10% of the sampling steps, surpassing the well-established baseline DiGress with 500 steps. On MOSES, DeFoG also outperforms diffusion models, achieving SOTA validity of 92.8% while maintaining a high uniqueness of 99.9%.

## 5.2. Efficiency Improvement

We now show that DeFoG enhances both training and sampling efficiency significantly across diverse datasets. Sampling Efficiency Figure 2a highlights the cumulative benefits of sampling approaches from Sec. 3.2. Starting with a vanilla DeFoG model (initially slightly below DiGress), each optimization step, denoted by + symbols, progressively improves performance, culminating in significant gains using only 50 steps on the Planar dataset. This demonstrates that the three sampling approaches are each essential components for optimizing generation. Training Efficiency Figure 2b illustrates convergence curves for the tree and MOSES datasets. We observe that incorporating sampling distortion enhances performance

5 10 50 100 1000
# Steps 60 80 100 5 10 50 100 1000
# Steps 0.0 0.3 0.6 0.9 Val idity (
QM
9)
+++ Stochasticity
++ Target Guidance + Sample Distortion Vanilla DFM DiGress V.U
.N.

 (Planar)
(a) Sampling efficiency improvement.

Validity (MOSES)++ Train Distortion
+ Sample Distortion Vanilla DFM DiGress V.U
.N.

 (Tree
)

2 4 6 Training time (h)
0.0 0.4 0.8 10 20 Training time (h)
0.6 0.8
(b) Training efficiency improvement.
significantly beyond the vanilla implementation, making it particularly useful for generation with undertrained models in resource-constrained settings (see Appendix B.6). Additionally, applying the same optimal distortion found in sampling during training typically yields further gains in convergence (see Appendix C.2). The convergence on some graph datasets may also benefit from an appropriate initial distribution, as shown for SBM in Appendix C.1.

## 5.3. Ablations

Here, we focus on evaluating the impact of different sampling approaches. We start from the vanilla sampling setup and sweep over sample distortion, target guidance, or stochasticity independently. More details in Appendix B.4. For illustration, here we focus on the Planar dataset: Sample Distortion In Figure 3a, we observe that cos and polydec distortions which emphasize refinement at later steps, perform better on the Planar dataset. This aligns with the intuition that, unlike continuous data undergoing gradual refinement, graphs often experience abrupt transitions due to the random sampling of categorical data. These transitions can violate hard constraints, such as planarity, as categorical values shift abruptly (e.g., from 0 to 1 in onehot encoding) when t approaches 1. These later steps are thus critical for error detection and correction. On the contrary, for datasets like SBM, where properties are not deterministic and such strict constraints are absent, this refinement does not provide any advantage (see Appendix B.1). Target Guidance As shown in Figure 3b, ω improves both V.U.N. and Ratio by biasing generation toward predicted clean data. However, excessive ω skews the generated distribution to the high density regions of training set distribution, leading to higher planarity (reflected by V.U.N.) but increased divergence from test graphs (reflected by Ratio). We also observe that the Ratio only begins to worsen at 0.1 with 50 steps, compared to 0.02 with 1000 steps. This demonstrates that target guidance is particularly effective with fewer steps, where the genera-

V.U.N. Ratio Polyinc Revcos Identity Cos Polydec Sample Distortion (50 steps)
0.0 0.3 0.6 1.5 Polyinc Revcos Identity Cos Polydec Sample Distortion (1000 steps)
0.75 0.90 2.0 10 20
(a) Sample Distortion.

V.U.N. Ratio 0.0 0.010.020.05 0.1 0.2 0.3 0.4 0.5 1.0 Target Guidance ω (50 steps)
0.15 0.30 5 10 5.0 0.0 0.010.020.05 0.1 0.2 0.3 0.4 0.5 1.0 Target Guidance ω (1000 steps)
0.8 0.9 7.5 10.0
(b) Target guidance.

V.U.N. Ratio 0 0.0 5.0 10.0 25.0 50.0 100.0 200.0 Stochasticity η (1000 steps)
0.80 0.88 0.96 0.0 5.0 10.0 25.0 50.0 100.0 200.0 Stochasticity η (50 steps)
0.00 0.05 0.10 2 3 100 200
(c) Stochasticity.
tion process becomes more challenging due to larger transitions, as it steers the model toward higher-confidence regions, safeguarding generative performance. Stochasticity As shown in Figure 3c, a moderate level of stochasticity benefits both metrics, while extreme values introduce excessive noise, disrupting the generation process. This indicates a sweet spot exists between effective error correction and over-stochasticity. Furthermore, the V.U.N. of generated graphs decreases with increasing η values when more steps are utilized (drop after η = 100 for 1000 steps vs. η = 10 for 50 steps). As this approach preserves the Kolmogorov equation, it benefits from more sampling steps to mitigate simulation errors.

## 6. Conclusion

We introduce DeFoG, a novel discrete flow matching framework for graphs. This formulation enables trainingsampling decoupling, which we ground theoretically to ensure faithful graph distribution modeling. Extensive experiments demonstrate the importance of our proposed strategies in achieving state-of-the-art performance on synthetic and molecular graph generation tasks. DeFoG currently employs a simple but efficient hyperparameter search for sampling, yielding impressive results and underscoring its potential for further improvement with more advanced search algorithms. Future work will further explore purely sampling-stage methods to enhance performance. Generating high-quality graphs in even fewer steps and scaling to larger graphs also remain key challenges.

## Acknowledgements

We would like to thank Clement Vignac and Andrew ´ Campbell for the useful discussions and suggestions.

## Impact Statement

The primary objective of this paper is to advance graph generation under a more flexible framework, with applications spanning general graph generation, molecular design, and digital pathology. The ability to generate graphs with discrete labels can have broad-reaching implications for fields such as drug discovery and diagnostic technologies. While this development has the potential to bring about both positive and negative societal or ethical impacts, particularly in areas like biomedical and chemical research, we currently do not foresee any immediate societal concerns associated with the proposed methodology.

## References

Asthana, R., Conrad, J., Dawoud, Y., Ortmanns, M., and Belagiannis, V. Multi-conditioned graph diffusion for neural architecture search. In Transactions on Machine Learning Research (TMLR), 2024.

Austin, J., Johnson, D. D., Ho, J., Tarlow, D., and Van Den Berg, R. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Bergmeister, A., Martinkus, K., Perraudin, N., and Wattenhofer, R. Efficient and scalable graph generation through iterative local expansion. In *International Conference on* Learning Representations (ICLR), 2023.

Brown, N., Fiscato, M., Segler, M. H., and Vaucher, A. C.

Guacamol: benchmarking models for de novo molecular design. In Journal of Chemical Information and Modeling, 2019.

Campbell, A., Benton, J., De Bortoli, V., Rainforth, T.,
Deligiannidis, G., and Doucet, A. A continuous time framework for discrete denoising models. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Campbell, A., Yim, J., Barzilay, R., Rainforth, T., and Jaakkola, T. Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. In International Conference on Machine Learning (ICML), 2024.

Cao, Y., Chen, J., Luo, Y., and Zhou, X. Exploring the optimal choice for generative processes in diffusion models: Ordinary vs stochastic differential equations. In *Advances in Neural Information Processing Systems* (NeurIPS), 2023.

Chen, X., He, J., Han, X., and Liu, L.-P. Efficient and degree-guided graph generation via discrete diffusion modeling. In *International Conference on Machine* Learning (ICML), 2023.

Chung, H., Sim, B., and Ye, J. C. Come-closer-diffusefaster: Accelerating conditional diffusion models for inverse problems through stochastic contraction. In *IEEE* Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

Dai, H., Nazi, A., Li, Y., Dai, B., and Schuurmans, D.

Scalable deep generative modeling for sparse graphs. In International Conference on Machine Learning (ICML), 2020.

De Cao, N. and Kipf, T. Molgan: An implicit generative model for small molecular graphs. In International Conference on Machine Learning (ICML) Workshops, 2018.

Diamant, N. L., Tseng, A. M., Chuang, K. V., Biancalani, T., and Scalia, G. Improving graph generation by restricting graph bandwidth. In International Conference on Machine Learning (ICML), 2023.

Dockhorn, T., Vahdat, A., and Kreis, K. Score-based generative modeling with critically-damped langevin diffusion. In International Conference on Learning Representations (ICLR), 2022.

Eijkelboom, F., Bartosh, G., Andersson Naesseth, C.,
Welling, M., and van de Meent, J.-W. Variational flow matching for graph generation. In *Advances in Neural* Information Processing Systems (NeurIPS), 2024.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., Muller, ¨
J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for highresolution image synthesis. In *International Conference* on Machine Learning (ICML), 2024.

Gat, I., Remez, T., Shaul, N., Kreuk, F., Chen, R. T., Synnaeve, G., Adi, Y., and Lipman, Y. Discrete flow matching. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Gillespie, D. T. A general method for numerically simulating the stochastic time evolution of coupled chemical reactions. In *Journal of Computational Physics*, 1976.

Gillespie, D. T. Exact stochastic simulation of coupled chemical reactions. In The Journal of Physical Chemistry, 1977.

Gillespie, D. T. Approximate accelerated stochastic simulation of chemically reacting systems. In The Journal of Chemical Physics, 2001.

Goyal, N., Jain, H. V., and Ranu, S. Graphgen: A scalable approach to domain-agnostic labeled graph generation.

In *The Web Conference*, 2020.

Haefeli, K. K., Martinkus, K., Perraudin, N., and Wattenhofer, R. Diffusion models for graphs benefit from discrete state spaces. In Learning on Graphs Conference (LOG) Extended Abstracts, 2022.

Han, X., Chen, X., Ruiz, F. J., and Liu, L.-P. Fitting autoregressive graph generative models through maximum likelihood estimation. In Journal of Machine Learning Research (JMLR), 2023.

Ho, J. and Salimans, T. Classifier-free diffusion guidance.

In *Advances in Neural Information Processing Systems* (NeurIPS), 2021.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

Hochreiter, S. and Schmidhuber, J. Long short-term memory. In *Neural computation*, 1997.

Igashov, I., Schneuing, A., Segler, M., Bronstein, M. M.,
and Correia, B. Retrobridge: Modeling retrosynthesis with markov bridges. In International Conference on Learning Representations (ICLR), 2024.

Irwin, R., Tibo, A., Janet, J. P., and Olsson, S. Efficient 3d molecular generation with flow matching and scale optimal transport. In *International Conference on Machine* Learning (ICML) Workshops, 2024.

Jaume, G., Pati, P., Bozorgtabar, B., Foncubierta-
Rodr´ıguez, A., Feroce, F., Anniciello, A. M., Rau, T. T., Thiran, J.-P., Gabrani, M., and Goksel, O. Quantifying explainers of graph neural networks in computational pathology. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

Jin, W., Barzilay, R., and Jaakkola, T. Junction tree variational autoencoder for molecular graph generation. In International Conference on Machine Learning (ICML), 2018.

Jo, J., Lee, S., and Hwang, S. J. Score-based generative modeling of graphs via the system of stochastic differential equations. In International Conference on Machine Learning (ICML), 2022.

Jo, J., Kim, D., and Hwang, S. J. Graph generation with diffusion mixture. In International Conference on Machine Learning (ICML), 2024.

Jolicoeur-Martineau, A., Li, K., Piche-Taillefer, R., Kach- ´
man, T., and Mitliagkas, I. Gotta go fast when generating data with score-based models. In *ArXiv*, 2021.

Karras, T., Aittala, M., Aila, T., and Laine, S. Elucidating the design space of diffusion-based generative models. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Kingma, D. P. and Welling, M. Auto-encoding variational bayes. In *ArXiv*, 2013.

Kipf, T. N. and Welling, M. Variational graph autoencoders. In *ArXiv*, 2016.

Kong, L., Cui, J., Sun, H., Zhuang, Y., Prakash, B. A., and Zhang, C. Autoregressive diffusion model for graph generation. In International Conference on Machine Learning (ICML), 2023.

Kuratowski, C. Sur le probleme des courbes gauches en topologie. In *Fundamenta Mathematicae*, 1930.

Kwon, Y., Lee, D., Choi, Y.-S., Shin, K., and Kang, S.

Compressed graph representation for scalable molecular graph generation. In *Journal of Cheminformatics*, 2020.

Liao, R., Li, Y., Song, Y., Wang, S., Hamilton, W., Duvenaud, D. K., Urtasun, R., and Zemel, R. Efficient graph generation with graph recurrent attention networks. In *Advances in Neural Information Processing* Systems (NeurIPS), 2019.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

Liu, J., Kumar, A., Ba, J., Kiros, J., and Swersky, K. Graph normalizing flows. In *Advances in Neural Information* Processing Systems (NeurIPS), 2019.

Liu, Q., Allamanis, M., Brockschmidt, M., and Gaunt, A.

Constrained graph variational autoencoders for molecule design. In *Advances in Neural Information Processing* Systems (NeurIPS), 2018.

Liu, X., Gong, C., and Liu, Q. Flow straight and fast:
Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations (ICLR), 2023.

Liu, X., He, Y., Chen, B., and Zhou, M. Advancing graph generation through beta diffusion. In International Conference on Learning Representations (ICLR), 2024.

Ma, L., Lin, C., Lim, D., Romero-Soriano, A., Dokania, P. K., Coates, M., Torr, P., and Lim, S.-N. Graph inductive biases in transformers without message passing. In International Conference on Machine Learning (ICML), 2023.

Ma, N., Goldstein, M., Albergo, M. S., Boffi, N. M.,
Vanden-Eijnden, E., and Xie, S. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision (ECCV), 2024.

Madeira, M., Thanou, D., and Frossard, P. Tertiary lymphoid structures generation through graph-based diffusion. In *International Conference on Medical Image* Computing and Computer-Assisted Intervention (MIC- CAI) Workshops, 2023.

Madeira, M., Vignac, C., Thanou, D., and Frossard, P.

Generative modelling of structurally constrained graphs. In *Advances in Neural Information Processing Systems* (NeurIPS), 2024.

Madhawa, K., Ishiguro, K., Nakago, K., and Abe, M.

Graphnvp: An invertible flow model for generating molecular graphs. In *ArXiv*, 2019.

Martinkus, K., Loukas, A., Perraudin, N., and Wattenhofer, R. Spectre: Spectral conditioning helps to overcome the expressivity limits of one-shot graph generators. In International Conference on Machine Learning (ICML), 2022.

Mendez, D., Gaulton, A., Bento, A. P., Chambers, J.,
De Veij, M., Felix, E., Magari ´ nos, M. P., Mosquera, J. F., ˜ Mutowo, P., Nowotka, M., et al. Chembl: towards direct deposition of bioassay data. In *Nucleic Acids Research*, 2019.

Mercado, R., Rastemo, T., Lindelof, E., Klambauer, G., ¨
Engkvist, O., Chen, H., and Bjerrum, E. J. Graph networks for molecular design. In Machine Learning: Science and Technology, 2021.

Morris, C., Ritzert, M., Fey, M., Hamilton, W. L., Lenssen, J. E., Rattan, G., and Grohe, M. Weisfeiler and leman go neural: Higher-order graph neural networks. In AAAI Conference on Artificial Intelligence, 2019.

Nisonoff, H., Xiong, J., Allenspach, S., and Listgarten, J. Unlocking guidance for discrete state-space diffusion and flow models. In International Conference on Learning Representations (ICLR), 2025.

Niu, C., Song, Y., Song, J., Zhao, S., Grover, A., and Ermon, S. Permutation invariant graph generation via score-based generative modeling. In International Conference on Artificial Intelligence and Statistics (AIS- TATS), 2020.

Pati, P., Jaume, G., Foncubierta-Rodriguez, A., Feroce, F., Anniciello, A. M., Scognamiglio, G., Brancati, N., Fiche, M., Dubruc, E., Riccio, D., et al. Hierarchical graph representations in digital pathology. In Medical Image Analysis, 2022.

Polykovskiy, D., Zhebrak, A., Sanchez-Lengeling, B.,
Golovanov, S., Tatanov, O., Belyaev, S., Kurbanov, R., Artamonov, A., Aladinskiy, V., Veselov, M., et al. Molecular sets (moses): a benchmarking platform for molecular generation models. In Frontiers in Pharmacology, 2020.

Qin, Y., Vignac, C., and Frossard, P. Sparse training of discrete diffusion models for graph generation. In *ArXiv*, 2023.

Ruddigkeit, L., Van Deursen, R., Blum, L. C., and Reymond, J.-L. Enumeration of 166 billion organic small molecules in the chemical universe database gdb-17. In Journal of Chemical Information and Modeling, 2012.

Salimans, T. and Ho, J. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations (ICLR), 2022.

Sanchez, G., Spangher, A., Fan, H., Levi, E., and Biderman, S. Stay on topic with classifier-free guidance. In International Conference on Machine Learning (ICML), 2024.

Shaul, N., Gat, I., Havasi, M., Severo, D., Sriram, A., Holderrieth, P., Karrer, B., Lipman, Y., and Chen, R. T. Flow matching with general discrete paths: A kinetic-optimal perspective. In International Conference on Learning Representations (ICLR), 2025.

Siraudin, A., Malliaros, F. D., and Morris, C. Cometh: A
continuous-time discrete-state graph diffusion model. In ArXiv, 2024.

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning (ICML), 2015.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. In *International Conference on Learning* Representations (ICLR), 2020a.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A.,
Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations (ICLR), 2020b.

Sterling, T. and Irwin, J. J. Zinc 15–ligand discovery for everyone. In Journal of Chemical Information and Modeling, 2015.

Sun, H., Yu, L., Dai, B., Schuurmans, D., and Dai, H.

Score-based continuous-time discrete diffusion models. In International Conference on Learning Representations (ICLR), 2023.

Sun, Z. and Yang, Y. Difusco: Graph-based diffusion solvers for combinatorial optimization. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Tseng, A. M., Diamant, N., Biancalani, T., and Scalia, G.

Complex preferences for different convergent priors in discrete graph diffusion. In International Conference on Machine Learning (ICML) Workshops, 2023.

Vignac, C., Krawczuk, I., Siraudin, A., Wang, B., Cevher, V., and Frossard, P. Digress: Discrete denoising diffusion for graph generation. In International Conference on Machine Learning (ICML), 2022.

Vignac, C., Osman, N., Toni, L., and Frossard, P. Midi:
Mixed graph and 3d denoising diffusion for molecule generation. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases
(ECML/PKDD), 2023.

Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K., and Pande, V.

Moleculenet: a benchmark for molecular machine learning. In *Chemical Science*, 2018.

Wu, Z., Trevino, A. E., Wu, E., Swanson, K., Kim, H. J.,
D'Angio, H. B., Preska, R., Charville, G. W., Dalerba, P. D., Egloff, A. M., et al. Graph deep learning for the characterization of tumour microenvironments from spatial protein profiles in tissue specimens. In *Nature* Biomedical Engineering, 2022.

Xu, K., Hu, W., Leskovec, J., and Jegelka, S. How powerful are graph neural networks? In International Conference on Learning Representations (ICLR), 2019.

Xu, Y., Deng, M., Cheng, X., Tian, Y., Liu, Z., and Jaakkola, T. Restart sampling for improving generative processes. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Xu, Z., Qiu, R., Chen, Y., Chen, H., Fan, X., Pan, M., Zeng, Z., Das, M., and Tong, H. Discrete-state continuous-time diffusion for graph generation. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Yi, K., Zhou, B., Shen, Y., Lio, P., and Wang, Y. Graph `
denoising diffusion for inverse protein folding. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

You, J., Ying, R., Ren, X., Hamilton, W., and Leskovec, J. Graphrnn: Generating realistic graphs with deep autoregressive models. In International Conference on Machine Learning (ICML), 2018.

Zhang, Q. and Chen, Y. Fast sampling of diffusion models with exponential integrator. In *International Conference* on Learning Representations (ICLR), 2023.

Zhu, W., Wen, T., Song, G., Wang, L., and Zheng, B.

On structural expressive power of graph transformers. In ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2023.

## Appendix Overview

In the Appendix, we provide additional details organized as follows:
1. Appendix A: Contextualizing Related Research. 2. Appendix B: Sample Optimization. 3. Appendix C: Train Optimization.

4. Appendix D: Theoretical Results.

5. Appendix E: Conditional Generation. 6. Appendix F: Experimental Details. 7. Appendix G: Additional Results.

## A. Contextualizing Related Research

In this section, we further contextualize DeFoG within the scope of related work. We begin by introducing the methods used for comparison with DeFoG in Appendix A.1. Subsequently, we outline the key distinctions between DeFoG and existing diffusion-based graph generative models in Appendix A.2.

## A.1. Overview Of Compared Methods

In Sec. 5, we evaluate DeFoG against a diverse set of graph generative models, which we introduce below:
- GraphRNN (You et al., 2018) and GRAN (Liao et al., 2019), two pioneering autoregressive models for graph generation;
- SPECTRE (Martinkus et al., 2022), a spectrally conditioned GAN-based model for graph generation; - DiGress (Vignac et al., 2022), the first discrete diffusion model for graph generation; - EDGE (Chen et al., 2023), a discrete diffusion model leveraging graph sparsity and degree guidance for scalability. - BwR (Diamant et al., 2023), which focuses on efficient graph representations via bandwidth restriction schemes that are compatible with various graph generation models. We report its results in combination with EDP-GNN (Niu et al., 2020), which was the first graph diffusion model;
- BiGG (Dai et al., 2020), an autoregressive model that exploits graph sparsity and training parallelization to scale to larger graphs;
- GraphGen (Goyal et al., 2020), a scalable autoregressive approach utilizing graph canonization with minimum DFS
codes, notable for being domain-agnostic and inherently supporting attributed graphs;
- HSpectre (Bergmeister et al., 2023), a hierarchical graph generation method that utilizes a score-based formulation for iterative local expansion steps;
- DisCo (Xu et al., 2024) and Cometh (Siraudin et al., 2024), two continuous-time discrete diffusion models for graph generation;
- GruM (Jo et al., 2024), which employs a diffusion mixture to explicitly learn the final graph topology and structure; - CatFlow (Eijkelboom et al., 2024), which results from the instantiation of variational flow matching to graph generation.

## A.2. Defog And Graph Diffusion Models

In this section, we contextualize DeFoG in relation to existing graph diffusion models.

## A.2.1. From Continuous To Discrete State-Spaces

Early diffusion-based graph generative models extended continuous diffusion and score-based methods from image generation to graphs by relaxing adjacency matrices into continuous state-spaces (Niu et al., 2020; Jo et al., 2022). However, this approach overlooks the inherent discreteness of graph-structured data, resulting in topologically uninformed noising processes. For instance, these methods often destroy graph sparsity and generate noisy complete graphs (Vignac et al., 2022; Xu et al., 2024; Siraudin et al., 2024), making it more challenging for denoising neural networks to recover meaningful structural properties from the noisy inputs. Some recent formulations operating on continuous state-spaces have tried to overcome these limitations: GruM (Jo et al., 2024) introduces an endpoint-conditioned diffusion mixture strategy to enhance accuracy by explicitly learning final graph structures, while CatFlow (Eijkelboom et al., 2024) proposes variational flow matching to handle categorical data more effectively. Alternatively, discrete diffusion models have emerged as a more natural solution, directly preserving the discrete nature of graph data (Vignac et al., 2022; Haefeli et al., 2022). These models have demonstrated state-of-the-art performance across a variety of applications, including neural architecture search (Asthana et al., 2024), combinatorial optimization (Sun & Yang, 2024), molecular generation (Irwin et al., 2024), and reaction pathway design (Igashov et al., 2024). DeFoG aligns with this second family of methods, modeling nodes and edges in discrete state-spaces to leverage the structural properties of graph data effectively.

## A.2.2. From Discrete To Continuous Time

The initial discrete-time diffusion frameworks for graph generation (Vignac et al., 2022; Haefeli et al., 2022) were built upon Discrete Denoising Diffusion Probabilistic Models (D3PMs) (Austin et al., 2021), which operate with a fixed partitioning of time. This discretization constrains the model to denoise at specific time points and ties the sampling process to the same fixed time steps used during training, leading to a rigid coupling between the training and sampling stages. Such inflexibility in time discretization can limit the quality of generated graphs. In contrast, continuous-time discrete diffusion frameworks (Campbell et al., 2022; Sun et al., 2023) overcome these limitations by enabling the model to denoise at arbitrary time points within a continuous interval (typically between 0 and 1). This flexibility allows the time discretization strategy for sampling to be selected post-training, enabling the use of advanced sampling techniques (Jolicoeur-Martineau et al., 2021; Zhang & Chen, 2023; Salimans & Ho, 2022; Chung et al., 2022; Song et al., 2020b; Dockhorn et al., 2022) to improve generation performance. These continuous-time frameworks have been successfully extended to graph generative models (Xu et al., 2024; Siraudin et al., 2024), achieving notable improvements. DeFoG follows a continuous-time formulation, leveraging its flexibility in sampling to achieve enhanced performance while maintaining the strengths of discrete state-space modeling.

## A.2.3. From Continuous-Time Discrete Diffusion To Discrete Flow Matching

While both continuous-time discrete diffusion and discrete flow matching (DFM) share the CTMC formulation for the denoising process, they differ fundamentally in the formulation of the noising process. Continuous-time discrete diffusionbased graph generative models (Xu et al., 2024; Siraudin et al., 2024) define the noising process as a CTMC, akin to the denoising process. However, this approach imposes two significant limitations:
1. **Incomplete Coupling of Training and Sampling**: The rate matrices of the noising and denoising processes are explicitly interrelated, and the noising rate matrix must be fixed during training. This restricts the sampling stage, preventing full decoupling of training and sampling.

2. **Limited Design Space**: The noising process must be derived analytically, which is not straightforward and is only feasible for rate matrices suitable for matrix exponentiation. Additionally, the denoising rate matrix is implicitly defined during training, constraining the flexibility of the denoising trajectory at sampling time (e.g., fixing the level of stochasticity).

In contrast, DeFoG allows for direct prescription of the noising process, pt|1, without these constraints. The rate matrix for the denoising process is selected exclusively at sampling time, fully decoupling the training and sampling stages. This flexibility enables performance optimization during sampling, such as tuning the stochasticity of the denoising trajectory via RDB
t or adjusting target guidance magnitude with Rω t.

The benefits of this decoupled framework are evident in Figures 2, 7 and 8, which demonstrate that the vanilla DeFoG configuration alone does not outperform existing diffusion-based graph generative models. However, our extensive sampling optimization pipeline capitalizes on DeFoG's flexible design space to achieve state-of-the-art results. These observations align with findings in iterative refinement methods across other data modalities. For instance, Karras et al. (2022) elaborate on the benefits of stochasticity adjustment in denoising trajectories within diffusion models for image generation. For a comprehensive discussion of the differences between continuous-time discrete diffusion and DFM frameworks, see Appendix H of Campbell et al. (2024).

## A.2.4. Mixed Integration Of Continuous And Discrete State-Spaces

Integrating continuous and categorical data within graph generative models is an important challenge, as many real-world applications involve heterogeneous data types (e.g., molecular graphs containing atomic coordinates alongside categorical atom and bond types). A recent example addressing this challenge is GBD (Liu et al., 2024), which incorporates beta diffusion to jointly model both continuous and discrete variables. Similarly, DeFoG is amenable to formulations involving mixed data types by leveraging an approach akin to MiDi (Vignac et al., 2023), independently factorizing continuous and discrete variables. However, explicitly exploring this integration is beyond the scope of this work.

## B. Sample Optimization

In this section, we explore the proposed sampling optimization in more detail. We start by analysing the different time distortion functions in Appendix B.1. Next, in Appendix B.2, we prove that the proposed target guidance mechanism actually satisfies the Kolmogorov equation, thus yielding valid rate matrices and, in Appendix B.3, we provide more details about the detailed balance equation and how it widens the design space of rate matrices. In Appendix B.4, we also describe the adopted sampling optimization pipeline. Finally, in Appendix B.5, we provide more details to better clarify the dynamics of the sampling process.

## B.1. Time Distortion Functions

In Sec. 3, we explore the utilization of different *distortion functions*, i.e., functions that are used to transform time. The key motivation for employing such functions arises from prior work on flow matching in image generation, where skewing the time distribution during training has been shown to significantly enhance empirical performance (Esser et al., 2024). In practical terms, this implies that the model is more frequently exposed to specific time intervals. Mathematically, this transformation corresponds to introducing a time-dependent re-weighting factor in the loss function, biasing the model to achieve better performance in particular time ranges. In our case, we apply time distortions to the probability density function (PDF) by introducing a function f that transforms the original uniformly sampled time t, such that t′ = f(t) for t ∈ [0, 1]. These time distortion functions must satisfy certain conditions: they must be monotonic, with f(0) = 0 and f(1) = 1. Although the space of functions that satisfy these criteria is infinite, we focus on five distinct functions that yield fundamentally different profiles for the PDF of t′. Our goal is to gain intuition about which time ranges are most critical for graph generation and not to explore that function space exhaustively. Specifically:
- *Polyinc*: f(t) = t 2, yielding a PDF that decreases monotonically with t′;
- Cos: f(t) = 1−cos πt 2, creating a PDF with high density near the boundariest′ = 0 and t′ = 1, and low for intermediate t′;
- Identity: f(t) = t, resulting in a uniform PDF for t′ ∈ [0, 1]; - *Revcos*: f(t) = 2t −
1−cos πt 2, leading to high PDF density for intermediate t′and low density at the extremes t′ = 0 and t′ = 1;
- *Polydec*: f(t) = 2t − t 2, where the PDF increases monotonically with t′.

The PDF resulting from applying a monotonic function f to a random variable t is given by:

$$\phi_{t^{\prime}}(t^{\prime})=\phi_{t}(t)\left|{\frac{\mathrm{d}}{\mathrm{d}t^{\prime}}}f^{-1}(t^{\prime})\right|,$$

where ϕt(t) and ϕt
′ (t′) denote the PDFs of t and t′, respectively. In our case, ϕt(t) = 1 for t ∈ [0, 1]. The distortion functions and their corresponding PDFs are illustrated in Figure 4. One of the strategies the proposed in **sampling** optimization procedure is the use of variable step sizes throughout the denoising process. This is achieved by mapping evenly spaced time points (DeFoG's vanilla version) through a transformation that follows the same constraints as the training time distortions discussed earlier. We employ the same set of time distortion functions, again not to exhaustively explore the space of applicable functions, but to gain insight into how varying step sizes affect graph generation. The expected step sizes for each distortion can be directly inferred from Figure 4. For instance, the *polydec* function leads to progressively smaller time steps, suggesting more refined graph edits in the denoising process as t′approaches 1. Note that even though we apply the same time distortions for both training and sample stages, in each setting they have different roles: in training, the time distortions skew the PDFs from where t′is sampled, while in sampling they vary the denoising step sizes.

Distortions PDFs for Different Distortions 0.0 0.2 0.4 0.6 0.8 1.0 t 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 f(t)
0 1 2 3 4 5 Polyinc Cos Identity Revcos Polydec Polyinc Cos Identity Revcos Polydec Function valu e PD
Fs
(a)

$$\left(7\right)$$

(b)

## B.2. Target Guidance

In this section, we demonstrate that the proposed *target guidance* design for the rate matrices violates the Kolmogorov equation with an error that is linear in ω. This result indicates that a small guidance factor effectively helps fit the distribution, whereas a larger guidance factor, as shown in Figure 9, while enhancing topological properties such as planarity, increases the distance between generated and training data on synthetic datasets according to the metrics of average ratio. Similarly, for molecular datasets, this also leads to an increase in validity and a decrease in novelty by forcing the generated data to closely resemble the training data.

Lemma 10 (Rate matrices for target guidance). Let Rω t(zt, zt+dt|z1) *be defined as:*

$$R_{t}^{\omega}(z_{t},z_{t+\mathrm{d}t}|z_{1})=\omega\frac{\delta(z_{1},z_{t+\mathrm{d}t})}{Z_{t}^{>0}\,p_{t|1}(z_{t}|z_{1})}.$$
. (7)
Then, the univariate rate matrix RTG
t(zt, zt+dt|z1) = R∗
t(zt, zt+dt|z1) + Rω t(zt, zt+dt|z1) violates the Kolmogorov equation with an error of −
ω Z
>0 t when zt ̸= z1*, and an error of* ω Z
>0 t −1 Z
>0 t when zt = z1.

Proof. In the remaining of the proof, we consider the case zt ̸= z1. We consider the same assumptions as Campbell et al.

(2024):

- pt|1(zt|z1) = 0 ⇒ R∗
$$=0\Rightarrow R_{t}^{*}(z_{t},z_{t+\mathrm{d}t}|z_{1})=0;$$
$\lceil\,\varkappa_t\,\vert\,\varkappa$
$\uparrow$
- pt|1(zt|z1) = 0 ⇒ ∂tpt|1(zt|z1) = 0 ("dead states cannot ressurect").

The z1-conditioned Kolmogorov equation is given by:

$$\partial_{t}p_{t|1}(z_{t}|z_{1})=\sum_{z_{t+dt}\neq z_{t}}R_{t}(z_{t+dt},z_{t}|z_{1})p_{t+dt|1}(z_{t+dt}|z_{1})\ -\sum_{z_{t+dt}\neq z_{t}}R_{t}(z_{t},z_{t+dt}|z_{1})p_{t|1}(z_{t}|z_{1})\tag{8}$$

We denote by RHS and LHS the right-hand side and left-hand side, respectively, of Eq. (8). For the case in which pt|1(zt|z1) > 0, we have:

RHS =X zt+dt̸=zt,pt+dt|1(zt+dt|z1)>0 (R ∗ t(zt+dt, zt|z1) + R ω t(zt+dt, zt|z1))pt+dt|1(zt+dt|z1) −X zt+dt̸=zt,pt+dt|1(zt+dt|z1)>0 (R∗ t(zt, zt+dt|z1) + R ω t(zt, zt+dt|z1))pt|1(zt|z1) =X zt+dt̸=zt,pt+dt|1(zt+dt|z1)>0 R ∗ t(zt+dt, zt|z1)pt+dt|1(zt+dt|z1) − R ∗ t(zt, zt+dt|z1)pt|1(zt|z1) +X zt+dt̸=zt,pt+dt|1(zt+dt|z1)>0 R ω t(zt+dt, zt|z1)pt+dt|1(zt+dt|z1) − R ω t(zt, zt+dt|z1)pt|1(zt|z1),
For the first sum, we have:

$$\sum_{z_{t+\mathrm{\tiny{dd}}}\neq z_{t},p_{t+\mathrm{\tiny{dd}}}|1\left(z_{t+\mathrm{\tiny{dd}}},z_{t}|z_{1}\right)p_{t+\mathrm{\tiny{dd}}}|1\left(z_{t+\mathrm{\tiny{dd}}}|z_{1}\right)-R_{t}^{*}(z_{t},z_{t+\mathrm{\tiny{dd}}}|z_{1})p_{t|1}(z_{t}|z_{1})\geq0$$
$$=\partial_{t}p_{t|1}(z_{t}|z_{1}).$$

since the z1-conditioned R∗
t generates pt|1.

For the second sum, we have:

X zt+dt̸=zt,pt+dt|1(zt+dt|z1)>0 R ω t(zt+dt, zt|z1)pt+dt|1(zt+dt|z1) − R ω t(zt, zt+dt|z1)pt|1(zt|z1) = zt+dt̸=zt,pt+dt|1(zt+dt|z1)>0 ωδ(z1, zt) Z >0 t pt+dt|1(zt+dt|z1) pt+dt|1(zt+dt|z1) =X −X zt+dt̸=zt,pt+dt|1(zt+dt|z1)>0 ωδ(z1, zt+dt) Z >0 t pt|1(zt|z1) pt|1(zt|z1) =ω Z >0 t X zt+dt̸=zt,pt+dt|1(zt+dt|z1)>0 (δ(z1, zt) − δ(z1, zt+dt))
If z1 ̸= zt, we have:

$$\sum_{z_{t+dt}\neq z_{t},p_{t+dt1}(z_{t+dt}|z_{1})>0}R_{t}^{\omega}(z_{t+dt},z_{t}|z_{1})p_{t+dt|1}(z_{t+dt}|z_{1})-R_{t}^{\omega}(z_{t},z_{t+dt}|z_{1})p_{t|1}(z_{t}|z_{1})=$$ $$=\frac{\omega}{Z_{t}^{\odot}}\sum_{z_{t+dt}\neq z_{t},p_{t+dt1}(z_{t+dt}|z_{1})>0}(\delta(z_{1},z_{t})-\delta(z_{1},z_{t+dt}))$$ $$=\frac{\omega}{Z_{t}^{\odot}}\sum_{z_{t+dt}\neq z_{t},p_{t+dt1}(z_{t+dt}|z_{1})>0}(0-\delta(z_{1},z_{t+dt}))$$ $$=-\frac{\omega}{Z_{t}^{\odot}},$$

Here, we apply the property that zt ̸= z1, which indicates that δ(z1, zt) = 0 and that there exists one and only one zt+dt ∈ {zt+dt, zt+dt ̸= zt} such that zt+dt = z1, which verifies that pt+dt|1(zt+dt|z1) > 0 - a condition satisfied by any initial distribution proposed in this work when t strictly positive—the sum simplifies to −
ω Z
>0 t
.

If z1 = zt, we have:

X
= ω Z
>0 t − 1 Z
>0 t
,
zt+dt̸=zt,pt+dt|1(zt+dt|z1)>0
R ω t(zt+dt, zt|z1)pt+dt|1(zt+dt|z1) − R ω t(zt, zt+dt|z1)pt(zt|z1) = X zt+dt̸=zt,pt+dt|1(zt+dt|z1)>0 (δ(z1, zt) − δ(z1, zt+dt)) X zt+dt̸=zt,pt+dt|1(zt+dt|z1)>0 (1 − 0)
=ω
Z
>0
t
=ω
Z
>0
t
Intuition The aim of *target guidance* is to reinforce the transition rate to the state predicted by the probabilistic model, z1. The ω term is an hyperparameter used to control the target guidance magnitude.

## B.3. Detailed Balance, Prior Incorporation, And Stochasticity

Campbell et al. (2024) show that although their z1-conditional formulation of R∗
t generates pt|1, it does not span the full space of valid rate matrices - those that satisfy the conditional Kolmogorov equation (Eq. (8)). They derive sufficient conditions for identifying other valid rate matrices. Notably, they demonstrate that matrices of the form

$$R_{t}^{\eta}:=R_{t}^{*}+\eta R_{t}^{\mathrm{DB}},$$

$$(9)$$

with η ∈ R≥0and RDB
tany matrix that verifies the *detailed balance condition*:
pt|1(zt|z1)R
DB
t(zt, zt+dt|z1) = pt|1(zt+dt|z1)R
DB
t(zt+dt, zt|z1), (9)
still satisfy the Kolmogorov equation. The detailed balance condition ensures that the outflow, pt|1(zt|z1)RDB
t(zt, zt+dt|z1), and inflow, pt|1(zt+dt|z1)RDB
t(zt+dt, zt|z1), of probability mass to any given state are perfectly balanced. Under these conditions, this additive component's contribution to the Kolmogorov equation becomes null (similar to the target guidance, as shown in the proof of of Theorem 10, in Appendix B.2).

A natural question is how to choose a suitable design for RDB
tfrom the infinite space of detailed balance rate matrices. As depicted in Figure 5, this flexibility can be leveraged to incorporate priors into the denoising model by encouraging specific transitions between states. By adjusting the sparsity of the matrix entries, additional transitions beyond those prescribed by R∗
tcan be introduced. In the general case, transitions between all states are possible; in the column case, a specific state centralizes all potential transitions; and in the single-entry case, only transitions between two states are permitted. These examples merely illustrate some possibilities and do not exhaust the range of potential RDB
t designs. The matrix entries can be structured by considering the following reorganization of terms of Eq. (9):

$$R_{t}^{\mathrm{DB}}(z_{t+\mathrm{d}t},z_{t}|z_{1})=\frac{p_{t|1}(z_{t}|z_{1})}{p_{t|1}(z_{t+\mathrm{d}t}|z_{1})}R_{t}^{\mathrm{DB}}(z_{t},z_{t+\mathrm{d}t}|z_{1}).$$

Therefore, a straightforward approach is to assign the lower triangular entries of the rate matrix as RDB
t(zt, zt+dt|z1) =
pt|1(zt+dt|z1), and the upper triangular entries as RDB
t(zt+dt, zt|z1) = pt|1(zt|z1). The diagonal entries are computed last to ensure that Rt(zt, zt) = −Pzt+dt̸=zt Rt(zt, zt+dt).

We incorporated various types of priors into RDB by preserving specific rows or entries in the matrix. Specifically, we experimented with retaining the column corresponding to the state with the highest marginal distribution (Column - Max Marginal), the column corresponding to the predicted x1 states (Column - x1), and the columns corresponding to the state with the highest probability in pt|1. Additionally, we tested the approach of retaining only RDB(xt, i) where i is the state with the highest marginal distribution (Entry - Max Marginal). For instance, under the absorbing initial distribution, this state is the one to which all data is absorbed at t = 0. We note that there remains significant space for exploration by adjusting the weights assigned to different positions within RDB, as the only condition that must be satisfied is that

| −                    | p1                      | p2                   |       |     |          |
|----------------------|-------------------------|----------------------|-------|-----|----------|
| p0                   | −                       | p2                   |       |     |          |
| p0                   | p1                      | −                    |       |     |          |
| General case         | −                       | p1                   | 0     |     |          |
| p0                   | −                       | p2                   |       |     |          |
| 0                    | p1                      | −                    |       |     |          |
| Column case          | −                       | 0                    | 0     |     |          |
| 0                    | −                       | p2                   |       |     |          |
| 0                    | p1                      | −                    |       |     |          |
| Single entry case    | −                       | 0                    | 0     |     |          |
| 0                    | −                       | 0                    |       |     |          |
| 0                    | 0                       | −                    |       |     |          |
| No stochasticity     |                         |                      |       |     |          |
| Full matrix          | Preserve one column/row | Preserve two entries | All 0 | R   | sparsity |
| DB                   |                         |                      |       |     |          |
| −                    | 0.7                     | 0.1                  |       |     |          |
| 0.2                  | −                       | 0.1                  |       |     |          |
| 0.2                  | 0.7                     | −                    | −     | 0.7 | 0        |
| 0.2                  | −                       | 0.1                  |       |     |          |
| 0                    | 0.7                     | −                    | −     | 0   | 0        |
| 0                    | −                       | 0.1                  |       |     |          |
| 0                    | 0.7                     | −                    | −     | 0   | 0        |
| 0                    | −                       | 0                    |       |     |          |
| 0                    | 0                       | −                    |       |     |          |
| Space of exploration |                         |                      |       |     |          |

Figure 5: Examples of different rate matrices from the space of 3×3 matrices that satisfy the detailed balance condition. Here pi denotes pt|1(i|z1).

RDB Design - Planar RDB Design - QM9 0.50 0.75 1.00 0.98 0.98 0.99 general Column - Max Marginal Column - x1 Column - pt|1 Entry - Max Marginal general Column - Max Marginal Column - x1 Column - pt|1 Entry - Max Marginal Vali dity V.

U.

N.

0.0 5.0 10.0 25.0 50.0 100.0 200.0 η 0.00 2.00 4.00 0.0 5.0 10.0 25.0 50.0 100.0 200.0 η 0.75 0.76 0.77 Novel ty Rati o

(a) Planar dataset.

(b) QM9 dataset.
symmetrical positions adhere to a specific proportionality. However, in practice, none of the specific designs illustrated in Figure 5 showed a clear advantage over others in the settings we evaluated. As a result, we chose the general case for our experiments, as it offers the most flexibility by incorporating the least prior knowledge.

Orthogonal to the design of RDB
t, we must also consider the hyperparameter η, which regulates the magnitude of stochasticity in the denoising process. Specifically, setting η = 0 (thereby relying solely on R∗
t) minimizes the expected number of jumps throughout the denoising trajectory under certain conditions, as shown by Campbell et al. (2024) in Proposition 3.4. However, in continuous diffusion models, some level of stochasticity has been demonstrated to enhance performance (Karras et al., 2022; Cao et al., 2023; Xu et al., 2023). Conversely, excessive stochasticity can negatively impact performance. Campbell et al. (2024) propose that there exists an optimal level of stochasticity that strikes a balance between exploration and accuracy. In our experiments, we observed varied behaviors as η increases, resulting in different performance outcomes across datasets, as illustrated in Figure 10.

## B.4. Hyperparameter Optimization Pipeline

A significant advantage of flow matching methods is their inherently greater flexibility in the sampling process compared to diffusion models, as they are more disentangled from the training stage. Each of the proposed optimization strategies exposed in Sec. 3.2 expands the search space for optimal performance. However, conducting a full grid search across all those methodologies is impractical for the computational resources available. To address this challenge, our sampling optimization pipeline consists of, for each of the proposed optimization strategies, all hyperparameters are held constant at their default values except for the parameter controlling the chosen strategy, over which we perform a sweep. The optimal values obtained for each strategy are combined to form the final configuration. In Tab. 5, we present the final hyperparameter values obtained for each dataset. This pipeline is sufficient to achieve state-of-the-art performance, which reinforces the expressivity of DeFoG. We expect to achieve even better results if a more comprehensive search of the hyperparameter space was carried out.