# JOINT GRAPH REWIRING AND FEATURE DENOISING VIA SPECTRAL RESONANCE

Jonas Linkerhägner<sup>∗</sup> Cheng Shi<sup>∗</sup> Ivan Dokmanic´ Department of Mathematics and Computer Science University of Basel firstname.lastname@unibas.ch

# ABSTRACT

When learning from graph data, the graph and the node features both give noisy information about the node labels. In this paper we propose an algorithm to jointly denoise the features and rewire the graph (JDR), which improves the performance of downstream node classification graph neural nets (GNNs). JDR works by aligning the leading spectral spaces of graph and feature matrices. It approximately solves the associated non-convex optimization problem in a way that handles graphs with multiple classes and different levels of homophily or heterophily. We theoretically justify JDR in a stylized setting and show that it consistently outperforms existing rewiring methods on a wide range of synthetic and real-world node classification tasks.

# 1 INTRODUCTION

Graph neural networks (GNNs) are a powerful deep learning tool for graph-structured data, with applications in physics [\(Mandal et al., 2022;](#page-13-0) [Linkerhägner et al., 2023\)](#page-13-1), chemistry [\(Gilmer et al.,](#page-12-0) [2017\)](#page-12-0), biology [\(Gligorijevic et al., 2021\)](#page-12-1) and beyond [\(Zhou et al., 2020\)](#page-15-0). Typical tasks across ´ disciplines include graph classification [\(Duvenaud et al., 2015;](#page-11-0) [Xu et al., 2019\)](#page-15-1), node classification [\(Kipf and Welling, 2017;](#page-13-2) [Li et al., 2019\)](#page-13-3) and link prediction [\(Pan et al., 2022\)](#page-14-0).

Graph datasets contain two distinct types of information: the graph structure and the node features. The graph encodes interactions between entities and thus the classes or communities they belong to, similarly to the features. Recent work demonstrates that *rewiring* the graph by judiciously adding and removing edges may improve downstream GNN performance. That work argues that in a GNN, the graph serves not only to encode interactions but also to organize message passing computations [\(Battaglia et al., 2018\)](#page-10-0). Even when it correctly encodes interactions it may not be an effective *computational* graph—rewiring it may then facilitate information flow.

Graph rewiring methods can be categorized into preprocessing and end-to-end. Preprocessing methods rewire the graph by relating its geometric and spectral properties to information flow [\(Topping](#page-15-2) [et al., 2022;](#page-15-2) [Nguyen et al., 2023;](#page-14-1) [Karhadkar et al., 2023\)](#page-12-2). End-to-end methods [\(Giraldo et al., 2023;](#page-12-3) [Gutteridge et al., 2023;](#page-12-4) [Qian et al., 2024\)](#page-14-2) dynamically rewire the graph during training, leveraging both the graph and the node features. Unlike preprocessing methods, they do not output an improved graph which restricts their interpretability and reusability. Our focus is on the preprocessing methods.

There are thus two mechanisms that hurt performance of GNNs: (1) real-world graphs and features are noisy (the graph has spurious and missing links), and (2) geometric properties of the graph impede message passing. In this paper we focus on (1) and ask a natural question: *can simple, joint feature and graph denoising improve performance of a downstream GNN?*

We propose a new rewiring scheme that also uses node features to produce an enhanced graph. We leverage the fact that both the graph and the features are correlated with the labels. This is explicit in high-quality stylized models of graphs with features, including community models such as the contextual stochastic block model (cSBM) [\(Deshpande et al., 2018\)](#page-11-1) and neighborhood graphs on

<sup>∗</sup>These authors contributed equally.

![](_page_1_Diagram_1.jpeg)

Figure 1: Schematic overview of joint denoising and rewiring (JDR). In this example, we consider a noisy graph as it occurs in many different real-world scenarios, in the sense that it contains edges between and within classes and its node features are not fully aligned with the labels. The graph's adjacency matrix A and binary node features X are decomposed via spectral decomposition and singular value decomposition (SVD). The rewiring of A is performed by combining the information of its own eigenvectors V and the singular vectors U from X. The same applies vice versa for denoising, and both are performed iteratively K times. We synthesize the rewired graph A˜ and the denoised features <sup>X</sup>˜ by multiplying back with the final <sup>V</sup>(K) and <sup>U</sup>(K) . To get specific properties like sparsity or binarity we can perform an UPDATE step, e.g. by thresholding (as done here). The resulting denoised and rewired graph is displayed on the right. Its structure now better represents the communities and the first entry of the features indicates the class assignment.

points from low-dimensional manifolds. This fact motivates various spectral clustering and nonlinear dimensionality reduction methods [\(Shi and Malik, 2000;](#page-15-3) [Ng et al., 2001\)](#page-14-3). In the cSBM, seminal theoretical work shows that jointly leveraging the graph (stochastic block model (SBM)) and the features (a Gaussian mixture model (GMM)) improves over unsupervised clustering using either piece of information alone. However, the associated efficient inference algorithms based on belief propagation [\(Deshpande et al., 2018;](#page-11-1) [Duranthon and Zdeborová, 2023\)](#page-11-2) rely on perfect knowledge of the distribution of the cSBM and cannot be applied to arbitrary real-world data.

Our contributions are as follows:

- 1. We take inspiration from work on the cSBM to design a practical algorithm for joint graph rewiring and feature denoising, which can improve the node classification performance of any downstream GNN on real-world data. We achieve this by adapting the graph and the features so as to maximize alignment between their leading eigenspaces. If these spaces are well-aligned we say that the graph and the features are *in resonance*.
- 2. We design an alternating optimization algorithm, joint denoising and rewiring (JDR), which approximates alignment maximization on spectrally-complex real-world graph data with multiple classes, possibly homophilic or heterophilic. We prove that JDR improves alignment between the graph and the features, but also with the labels, on a stylized generative model with noise from the Gaussian orthogonal ensemble (GOE); a recent conjecture in the literature suggests that this generalizes to cSBM.
- 3. We run extensive experiments to show that JDR outperforms existing preprocessing rewiring strategies while being guided solely by denoising.

This last point suggests that although there exist graphs with topological and geometrical characteristics which make existing rewiring schemes beneficial, a greater issue in real-world graphs is noise in the sense of missing and spurious links. This is true even when the graphs correctly reflect the ground truth information. In a citation network, for example, citations that *should* exist may be missing because of incomplete scholarship. Conversely, citations that *should not* exist may be present because the authors engaged in bibliographic ornamentation. Our method is outlined in Figure [1](#page-1-0) and the code repository is available online[<sup>1</sup>](#page-1-1) .

<sup>1</sup><https://github.com/jlinki/JDR>

### 2 JOINT DENOISING AND REWIRING

#### 2.1 PRELIMINARIES

We let G = (V, E) be an undirected graph with |V| = N nodes and an adjacency matrix A. To each node we associate an F-dimensional feature vector and collect these vectors in the rows of matrix X ∈ R <sup>N</sup>×<sup>F</sup> . We make extensive use of the graph and feature spectra, namely the eigendecomposition A = V ΛV T and the SVD X = UΣW<sup>T</sup> , with eigen- and singular values ordered from largest to smallest. (As discussed below, in heterophilic graphs we order the eigenvalues of A according to their absolute value.) The graph Laplacian is L = D − A, where D is the diagonal node degree matrix. For k > 2 node classes, we use one-hot labels y ∈ {0, 1} N×k . We write [L] for the set {1, 2, . . . , L}. In the balanced two-class case, we consider nodes to be ordered so that the first half has label y<sup>i</sup> = −1 and the second half y<sup>i</sup> = 1. In semi-supervised node classification, the task is to label the nodes based on the graph (A and X) and a subset of the labels y. *Homophilic* graphs are those where nodes are more likely to connect with nodes with similar features or labels (e.g., friendship networks [\(McPherson et al., 2001\)](#page-14-4)); *heterophilic* graphs are those where nodes more likely to connect with dissimilar nodes (e.g., protein interaction networks [\(Zhu et al., 2020\)](#page-15-4)).

#### 2.2 MOTIVATION VIA THE CONTEXTUAL STOCHASTIC BLOCK MODEL

For simplicity, we first explain our method for k = 2 classes and graphs generated from the cSBM. We then extend it to real-world graphs with multiple classes and describe the full practical algorithm.

Contextual Stochastic Block Model. CSBMs [\(Deshpande et al., 2018\)](#page-11-1) extend SBMs [\(Abbe, 2018\)](#page-10-1), a community graph model, by high-dimensional node features. They have become a key generative model for studying GNNs [\(Baranwal et al., 2021;](#page-10-2) [Wu et al., 2023;](#page-15-5) [Kothapalli et al., 2023\)](#page-13-4); for further pointers see Appendix [A.2.](#page-18-0) We use cSBMs to build intuition about the graph rewiring and denoising problem. In a balanced 2-class SBM, the nodes are divided into two equal-sized communities with node labels y<sup>i</sup> ∈ {±1}. Pairs of nodes connect independently at random, with probability cin/N inside communities and cout/N across communities.

In the sparse regime [\(Abbe, 2018\)](#page-10-1), with average node degree d = O(1), it is common to parameterize probabilities as cin = d + λ √ d and cout = d − λ √ d, where |λ| can be seen as the signal-to-noise ratio (SNR) of the graph. The signal X<sup>i</sup> ∈ <sup>R</sup> <sup>F</sup> at node i comes from a GMM,

$$\mathbf{X}_i = \sqrt{\frac{\mu}{N}} \mathbf{y}_i \boldsymbol{\xi} + \frac{\boldsymbol{z}_i}{\sqrt{F}}, \quad (1)$$

where ξ ∼ N (0, I<sup>F</sup> /F) is the randomly drawn mean and z<sup>i</sup> ∼ N (0, I<sup>F</sup> ) is i.i.d. Gaussian standard noise. We set γ = N F and, following [Chien et al.](#page-11-3) [\(2021\)](#page-11-3), parameterize the graphs generated from the cSBM using ϕ = π arctan(λ √γ/µ). For <sup>ϕ</sup> <sup>→</sup> <sup>1</sup> we get homophilic behavior; for <sup>ϕ</sup> → −<sup>1</sup> we get heterophilic behavior. Close to either extreme the node features contain little information. For ϕ → 0 the graph is Erdos–Rényi and only the features contain information. ˝

Denoising and Rewiring the cSBM. In the cSBM, A and X offer different noisy views on the labels. One can show that up to a scaling and a shift, the adjacency matrix is approximately ±yy<sup>T</sup> + ZER, which means that it is approximately a rank-one matrix with labels in the range, corrupted with "Erdos–Rényi-like noise" ˝ ZER [\(Erdös and Rényi, 1959\)](#page-12-5). Another way to see this is to note that EA = 1 2N (cin + cout)11<sup>T</sup> + 1 2N (cin − cout)yy<sup>T</sup> (from the definition of the SBM). Since A is close to EA at high SNR, the eigenvectors contain information about the labels. It similarly follows directly from the definition that the feature matrix X is (up to a scaling) yu<sup>T</sup> + Z<sup>G</sup> where Z<sup>G</sup> is white Gaussian noise. It thus makes sense to use the information from X to enhance A and vice versa. [Deshpande et al.](#page-11-1) [\(2018\)](#page-11-1) show that analyzing the following optimization problem:

$$\begin{aligned} \text{maximize } & \langle \mathbf{v}, \mathbf{A}\mathbf{v} \rangle + b \langle \mathbf{v}, \mathbf{X}\mathbf{u} \rangle \\ & v \in \mathbb{R}^N, \mathbf{u} \in \mathbb{R}^F \\ \text{subject to } & \|\mathbf{v}\|_2 = \|\mathbf{u}\|_2 = 1, \langle \mathbf{v}, \mathbf{1} \rangle \leq \delta \end{aligned} \tag{2}$$

for some carefully chosen value of b allows one to characterize detection bounds in unsupervised community detection with k = 2. It is clear from the above reasoning that in the high-SNR regime (λ and µ far away from the detection threshold), the second leading eigenvector of A and the leading

![](_page_3_Figure_1.jpeg)

Figure 2: An illustration of spectral alignment and resonance. In [\(a\)](#page-3-0) we plot r = x <sup>T</sup> Ax for different noise levels in A and x ∈ {−1, 1} <sup>N</sup> , illustrated in the rows below. Without noise, x is exactly the label vector and A is block-diagonal. We apply multiplicative noise; namely, for each noise level, we flip the sign of a proportion of values, resulting in a random signal for ±0.5. We see that the value of r depends on the noise level. The maximum is achieved for zero noise when the second leading eigenvector of A and the signal x are perfectly aligned. In [\(b\)](#page-3-0), we consider a signal xˆ = Ax for different noise levels in A and x on a graph with 20 nodes; only a quarter of edges are shown to reduce clutter; the intra-class edges are grey; the inter-class edges are black. The largest norm is obtained for noise-free A and x (upper-left corner). The norm of xˆ and the separation of communities decrease along both noise axes. The inherent denoising capabilities of propagating x on a high-SNR graph [\(Ma et al., 2021b\)](#page-13-5) are also visible, particularly in the first two rows to the right.

left singular vector of X approximately coincide with the labels. The optimal v ∗ is related to those vectors and aligned with the labels, since the quadratic and the bilinear form in [\(2\)](#page-2-0) are individually maximized by the mentioned vectors. The maximizer of the linear combination of both terms therefore combines the spectral information from both matrices—the graph and the features. This suggests the following rationale for denoising: (1) We can interpret the value of [\(2\)](#page-2-0) as a measure of alignment. Since v ∗ corresponds to the labels, we can relate this measure to the quality of the label estimation. (2) We may leverage this alignment to rewire the graph and denoise the features. Namely, we could perturb A and X in a way that improves the alignment.

In real datasets, however, the optimal value of b is unknown, the scaling of X is arbitrary, and things are further complicated by having (many) more than 2 classes. Moreover, [\(2\)](#page-2-0) is computationally hard. We thus define a simple related measure of alignment which alleviates these issues.

Definition 1. *Recall the decompositions* A = V ΛV T *,* X = UΣW<sup>T</sup> *, and let* VL*,* U<sup>L</sup> *denote the first* L *columns of* V *and* U *and* ∥.∥sp *the spectral norm. We define graph–feature alignment as*

$$\text{Alignment}_L(\mathbf{A}, \mathbf{X}) = \|\mathbf{V}_L^T \mathbf{U}_L\|_{\text{sp}}. \quad (3)$$

*Remark:* The logic of this definition is that for a cSBM with high SNR and k classes, the information about labels is indeed contained in the leading L = k vectors of V and U. This follows directly by generalizing the formulation in [\(2\)](#page-2-0) to multiple classes and thus multiple eigenvectors [\(Decelle et al.,](#page-11-4) [2011;](#page-11-4) [Lesieur et al., 2017\)](#page-13-6). The quantity AlignmentL(A, X) is the cosine of the angle between the subspaces spanned by the columns of V<sup>L</sup> and UL. To denoise the features and rewire the graph, we seek to improve the alignment.

Given AlignmentL(A, X) and a graph with A<sup>0</sup> and X0, the jointly denoised graph and features are the solution to

$$\begin{aligned} & \text{maximize Alignment}_L(\mathbf{A}, \mathbf{X}) \\ & \mathbf{A}, \mathbf{X} \\ & \text{subject to } \|\mathbf{A} - \mathbf{A}_0\| \leq \delta_A, \|\mathbf{X} - \mathbf{X}_0\| \leq \delta_X. \end{aligned} \quad (4)$$

The parameters δA, δ<sup>X</sup> > 0 modulate the strength of alignment. We will show empirically that a stronger alignment indicates a better representation of the labels by A and X and thus a better graph. Figure [2](#page-3-0) visualizes this connection. It shows that the response of the graph to features is maximized when the spectra of the graph and the features are aligned. We refer to the condition where the alignment is high as spectral resonance; see Appendix [A.1.1](#page-17-0) for further discussion.

#### 2.3 JOINT DENOISING AND REWIRING ALGORITHM

Maximizing the alignment [\(4\)](#page-3-1) directly, e.g., using gradient descent, is computationally challenging. Here we propose a heuristic which alternates between spectral interpolation and graph synthesis. We later prove that the resulting algorithm indeed improves alignment, both with the labels and between the graph and the features, under a stylized noise model. The algorithm, illustrated in Figure [1,](#page-1-0) comprises three steps. In Step 1, we compute the spectral decompositions of A and X. To improve the alignment, we interpolate between the L largest eigenvectors in Step 2. Based on the new eigenvectors, we synthesize a new graph in Step 3. The three steps are iterated until a stopping criterion is met. As is standard in the rewiring literature, the hyperparameters of the algorithm are tuned on a validation set. Formalizing this results in the JDR algorithm:

#### Step 1: Decomposition

$$A = V \Lambda V^T \text{ with } V = (v_1, v_2, \dots, v_N) \text{ and } X = U \Sigma W^T \text{ with } U = (u_1, u_2, \dots, u_N)$$

Step 2: Interpolation: For every i ∈ [L],

$$\begin{aligned}\tilde{v}_i &= (1 - \eta_A)\mathbf{v}_i + \eta_A \operatorname{sign}(\langle \mathbf{v}_i, \mathbf{u}_j \rangle)\mathbf{u}_j \\ \tilde{u}_i &= (1 - \eta_X)\mathbf{u}_i + \eta_X \operatorname{sign}(\langle \mathbf{u}_i, \mathbf{v}_j \rangle)\mathbf{v}_j\end{aligned}$$

where j is chosen as argmaxj∈[L] |⟨v<sup>i</sup> ,u<sup>j</sup> ⟩| when updating v<sup>i</sup> and as argmaxj∈[L] |⟨u<sup>i</sup> , v<sup>j</sup> ⟩| when updating u<sup>i</sup> . η<sup>A</sup> and η<sup>X</sup> are hyperparameters that are tuned with a downstream algorithm on a validation set. We use sign() to handle sign ambiguities in decompositions.

Step 3: Graph Synthesis

$$\tilde{\mathbf{A}} = \tilde{\mathbf{V}} \mathbf{\Lambda} \tilde{\mathbf{V}}^T \text{ and } \tilde{\mathbf{X}} = \tilde{\mathbf{U}} \mathbf{\Sigma} \mathbf{W}^T$$

Step 4: Iterate steps K times with

$$A \leftarrow \tilde{A}$$
 and  $X \leftarrow \tilde{X}$ .

Following [\(3\)](#page-3-2), we consider the L leading eigenvectors of A and X for interpolation. Since these bases may be rotated with respect to each other (we note that [\(3\)](#page-3-2) is insensitive to relative rotations, see Appendix [A.1.2\)](#page-18-1), when updating an eigenvector of A, we interpolate it with the most similar eigenvector of X. We show empirically that this heuristic yields strong results, but also prove that it improves alignment with labels with a stylized noise model. We emphasize that the interpolation rates η<sup>A</sup> and η<sup>X</sup> are the same across different eigenvectors and iterations K. After K steps, we synthesize the final weighted dense graph <sup>A</sup>˜ <sup>=</sup> <sup>V</sup>(K)Λ<sup>V</sup> T (K) . To efficiently apply GNNs, we can enforce sparsity, e.g., via thresholding or selecting the top-k entries per node. A detailed pseudocode is given in Appendix [A.1.](#page-17-1)

An illustration. A simple edge case to illustrate how the algorithm works is when either only A or X contains information. In a cSBM with ϕ = 0, X contains all the information, so the best hyperparameter choice is η<sup>X</sup> = 0 and [\(4\)](#page-3-1) simplifies to a maximization over A. Since there are only two classes, it is sufficient to consider L = 1. From [\(2\)](#page-2-0) we know that the leading left singular vector u<sup>1</sup> of X is well-aligned with the labels. We thus replace the second leading eigenvector v<sup>2</sup> in A by u<sup>1</sup> by choosing η<sup>A</sup> = 1.0. After graph synthesis, the new v<sup>2</sup> of A˜ is not yet equal to u1, since u<sup>1</sup> was not orthogonal to the other v<sup>i</sup> . We thus repeat the three steps K times. For ϕ = ±1 all information is contained in the graph; a similar argument can then be constructed *mutatis mutandis*.

JDR Improves Alignment. We now show that JDR improves alignment, as defined in [\(4\)](#page-3-1), under a stylized cSBM-like model. In fact, we show a stronger result: that the algorithm improves alignment with the true labels. Appealing to universality arguments [\(Hu and Lu, 2023\)](#page-12-6), we study a model with a spiked Gaussian matrix A<sup>c</sup> = λ N yy<sup>T</sup> <sup>+</sup> √ N O<sup>A</sup> where O<sup>A</sup> is GOE noise instead of the binary matrix

Table 1: Comparison of state-of-the-art preprocessing rewiring approaches. Note that we refer to the computational complexity per iteration. N denotes the number of nodes, m the number of edges and dmax is the maximum node degree. Additional details on the complexity of JDR are given in Appendix [A.1.3;](#page-18-2) detailed runtime comparisons are in Appendix [A.1.4](#page-18-3)

| Method |            |    |      |       | Add edge Remove edge | Use Features Heterophilic? | Complexity |    |     |       |
|--------|------------|----|------|-------|----------------------|----------------------------|------------|----|-----|-------|
| DIGL   | (Gasteiger | et | al., | 2019) | ✓ ✗                  | ✗ ✗                        | O          |    | ( N | )     |
| FoSR   | (Karhadkar | et | al., | 2023) | ✓ ✗                  | ✗ ✓                        | O          | (  | N   | 2     |
| BORF   | (Nguyen    | et | al., | 2023) | ✓ ✓                  | ✗ ✓                        | O (        | md |     | 3     |
|        |            |    |      |       |                      |                            |            |    |     | max ) |
| JDR    | (Ours)     |    |      |       | ✓ ✓                  | ✓ ✓                        | O          |    | ( N | )     |

A, and features X = p <sup>µ</sup> N yξ<sup>T</sup> <sup>+</sup> √ F O<sup>X</sup> as defined in [1.](#page-2-1) In the cSBM context, a recent conjecture with strong empirical support states that replacing the binary A by A<sup>c</sup> leads to the same behavior in downstream tasks such as community detection [\(Deshpande et al., 2018;](#page-11-1) [Lu and Sen, 2023\)](#page-13-7) and node classification [\(Shi et al., 2024\)](#page-15-6). An iteration of JDR with L = 1 applied to this model, first interpolates between the leading eigenvector v<sup>1</sup> (A<sup>c</sup> ) = v<sup>A</sup> and leading left singular vector u<sup>1</sup> (X) = uX. Graph and feature synthesis then yields A<sup>c</sup> <sup>η</sup><sup>A</sup> <sup>=</sup> <sup>A</sup><sup>c</sup> <sup>+</sup> <sup>λ</sup><sup>1</sup> (A<sup>c</sup> ) −vAv T <sup>A</sup> + v˜Av˜ T A and Xη<sup>X</sup> = X + σ<sup>1</sup> (X) −uXw<sup>T</sup> <sup>X</sup> + u˜Xw<sup>T</sup> X . Here v˜<sup>A</sup> = (1 − ηA)v<sup>A</sup> + sign(⟨vA,uX⟩)ηAv<sup>X</sup> and u˜<sup>X</sup> = (1−ηX)u<sup>X</sup> + sign(⟨vA,uX⟩)ηXvA, where w<sup>1</sup> (X) = w<sup>X</sup> is the leading right singular vector of X. Denoting y˜ = y/ √ N, we have

Proposition 1. *Let* λ > <sup>1</sup> *and* µ > √<sup>γ</sup> *with* <sup>γ</sup> <sup>=</sup> N/F*. There exist* <sup>η</sup> 0 A, η<sup>0</sup> <sup>X</sup> ∈ (0, 1) *such that for all* η<sup>A</sup> ∈ (0, η<sup>0</sup> <sup>A</sup>) *and* η<sup>X</sup> ∈ (0, η<sup>0</sup> <sup>X</sup>)*, when* N → ∞*, we have*

$$\langle v_1(A_{\eta_A}^c), \tilde{y} \rangle^2 > \langle v_1(A^c), \tilde{y} \rangle^2 \quad \text{and} \quad \langle u_1(X_{\eta_X}), \tilde{y} \rangle^2 > \langle u_1(X), \tilde{y} \rangle^2.$$

In words, interpolation improves alignment of the largest eigenvector with the labels y for sufficiently large graphs. The proof, based on the BBP transition in the spiked covariance model [\(Baik](#page-10-3) [et al., 2005\)](#page-10-3) and the fluctuation of the leading eigenvector, can be found in Appendix [A.3.](#page-19-0) It seems challenging but quite possible to extend this argument to a binary A. One would then interpolate between u<sup>x</sup> and A's second leading eigenvector v2(A), which has similar properties to v1(A<sup>c</sup> ), especially in a dense graph regime [\(Nadakuditi and Newman, 2012\)](#page-14-5).

# 3 EXPERIMENTS

We extensively evaluate JDR on both synthetic data generated from the cSBM and real-world benchmark datasets. We follow experimental setting from [Chien et al.](#page-11-3) [\(2021\)](#page-11-3) and evaluate JDR for semisupervised node classification with different downstream GNNs. We also adopt their data splits, namely the sparse splitting 2.5%/2.5%/95% for training, validation and testing, respectively, or the dense splitting 60%/20%/20%. For the general experiments, we perform 100 runs with different random splits. For the scalability experiments, we use the experimental settings of the respective works [\(Lim et al., 2021;](#page-13-8) [Platonov et al., 2023\)](#page-14-6). We report the average accuracy and the 95%-confidence interval calculated via bootstrapping with 1000 samples. All experiments are reproducible using the code provided.

Baselines. Following recent works on rewiring, we use graph convolution network (GCN) [\(Kipf](#page-13-2) [and Welling, 2017\)](#page-13-2) as our downstream GNN. To obtain a more comprehensive picture, we additionally evaluate the performance on the more recent generalized PageRank graph neural network (GPRGNN) [\(Chien et al., 2021\)](#page-11-3). We compare our algorithm with the state-of-the-art rewiring methods first-order spectral rewiring (FoSR) [\(Karhadkar et al., 2023\)](#page-12-2), batch Ollivier-Ricci flow (BORF) [\(Nguyen et al., 2023\)](#page-14-1) and diffusion improves graph learning (DIGL) [\(Gasteiger et al., 2019\)](#page-12-7). FoSR approximates which edges should be added to maximize the spectral gap to reduce oversquashing. BORF adds edges in regions of negative curvature in the graph, which indicate bottlenecks that can lead to an oversquashing of the messages passed along these edges. A positive curvature indicates that there are so many edges in this area that messages could be oversmoothed, which is why edges are removed here. We compare computational and implementation aspects of JDR and baselines in Table [1.](#page-5-0) On the cSBM, we compare to an *optimal* algorithm, namely the approximate message passing-belief propagation (AMP-BP) algorithm [\(Duranthon and Zdeborová, 2023\)](#page-11-2). AMP-BP is asymptotically optimal (in the large dimension limit) for unsupervised or semi-supervised community detection in the cSBM. It relies on knowing the distribution of the cSBM and is thus not applicable to real-world graphs with unknown characteristics and complex features.

Hyperparameters. Unless stated otherwise, we use the hyperparameters from [Chien et al.](#page-11-3) [\(2021\)](#page-11-3) for the GNNs and optimize the hyperparameters of JDR using a mixture of grid and random search on the validation set. We use the top-64 values of A˜ to enforce sparsity and interpolation to update the features. For DIGL, FoSR and BORF, we tune their hyperparameters using a grid search, closely following the given parameter range from the original papers. For all hyperparameter searches we use GCN and GPRGNN as the downstream models on 10 runs with different random splits. A detailed list of all hyperparameters can be found in Appendix [A.7](#page-32-0) or in the code repository.

#### 3.1 RESULTS ON SYNTHETIC DATA

We first test JDR on data generated from the cSBM, as we can easily vary the SNR of the graph and the features to verify its denoising and rewiring capabilities. We focus on the sparse splitting, since for the dense splitting GPRGNN already matches the performance of AMP-BP.

![](_page_6_Figure_6.jpeg)

Figure 3: Alignment of the leading eigenspaces according to [\(3\)](#page-3-2) for graphs

from the cSBM with different ϕ. Heterophilic Regime. For ϕ < −0.25, the predictions of GCN are only slightly better than random. GPRGNN performs much better, since it can learn higher order polynomial filters to deal with heterophily. GCN+JDR outperforms the baseline by a very large margin; it handles heterophilic data well. Using JDR for GPRGNN further improves its already strong performance in this regime. Both GNNs benefit less from the denoising in the weakly heterophilic setting where they exhibit the worst performance across all ϕ. The difference between ϕ = 0 and the weakly heterophilic regime is that "optimal denoising" for ϕ = 0 is straightforward, since all the information is contained in X. We show similar findings for spectral clustering on the cSBM in Appendix [A.5.6.](#page-27-0)

Does JDR Maximize Alignment? Before discussing Figure [4,](#page-6-0) which shows the results of baselines and JDR for different values of ϕ, we verify empirically that our alternating optimization algorithm indeed approximates solutions to [\(4\)](#page-3-1). As shown in Figure [3,](#page-6-1) the quantity AlignmentL(A, X) improves significantly after running JDR, across all ϕ. As we show next, this happens simultaneously with improvements in downstream performance, which lends credence to the intuitive reasoning that motivates our algorithm. For additional alignment results on real-world data and baselines, refer to Appendix [A.5.5.](#page-26-0)

![](_page_6_Figure_10.jpeg)

Figure 4: Test accuracy on graphs from the cSBM across different ϕ. The error bars indicate the 95% confidence interval. JDR improves the performance for both GNNs across all ϕ.

Weak Graph Regime. For |ϕ| ≤ 0.25, where the SNR of the graph is very low, both GNNs perform poorly. Intuitively, when the graph is very noisy, a GNN is a suboptimal model, since it leverages the graph structure. A simple MLP baseline, using only the node features, outperforms GNNs in this setting, with all three approaches lagging far behind AMP-BP. Using JDR, we see significant improvements for both GNNs, which almost catch up with AMP-BP for ϕ = 0. Although all information was available in the node features, the GNN with JDR now clearly outperform the MLP by a very large margin. We argue that this is because in the semisupervised setting with few labels available, the GNN generalizes much better.

Table 2: Results on real-world homophilic datasets in the sparse splitting (2.5%/2.5%/95%): Mean accuracy across runs (%) ± 95% confidence interval. Best average accuracy in bold.

| Method      |    |    | Cora |    |    |    | CiteSeer |    |    |    | PubMed |    |    |    |     |    |    |    | Photo |    |
|-------------|----|----|------|----|----|----|----------|----|----|----|--------|----|----|----|-----|----|----|----|-------|----|
| GCN         | 77 | 26 | ± 0  | 35 | 67 | 16 | ± 0      | 37 | 84 | 22 | ± 0    | 09 | 84 | 42 | ± 0 | 31 | 91 | 33 | ± 0   | 29 |
| GCN+DIGL    | 79 | 27 | ± 0  | 26 | 68 | 03 | ± 0      | 33 | 84 | 60 | ± 0    | 09 | 86 | 00 | ± 0 | 24 | 92 | 00 | ± 0   | 23 |
| GCN+FoSR    | 77 | 23 | ± 0  | 34 | 67 | 03 | ± 0      | 34 | 84 | 21 | ± 0    | 09 | 84 | 34 | ± 0 | 27 | 91 | 36 | ± 0   | 28 |
| GCN+BORF    | 77 | 23 | ± 0  | 35 | 66 | 96 | ± 0      | 38 | 84 | 22 | ± 0    | 09 | 84 | 46 | ± 0 | 30 | 91 | 26 | ± 0   | 30 |
| GCN+JDR     | 79 | 96 | ± 0  | 26 | 69 | 35 | ± 0      | 28 | 84 | 79 | ± 0    | 08 | 85 | 66 | ± 0 | 36 | 92 | 52 | ± 0   | 23 |
| GPRGNN      | 79 | 65 | ± 0  | 33 | 67 | 50 | ± 0      | 35 | 84 | 33 | ± 0    | 10 | 84 | 06 | ± 0 | 48 | 92 | 01 | ± 0   | 41 |
| GPRGNN+DIGL | 79 | 77 | ± 0  | 30 | 67 | 50 | ± 0      | 35 | 84 | 72 | ± 0    | 10 | 86 | 25 | ± 0 | 28 | 92 | 31 | ± 0   | 25 |
| GPRGNN+FoSR | 79 | 22 | ± 0  | 31 | 67 | 30 | ± 0      | 38 | 84 | 32 | ± 0    | 09 | 84 | 21 | ± 0 | 46 | 92 | 07 | ± 0   | 37 |
| GPRGNN+BORF | 79 | 43 | ± 0  | 30 | 67 | 48 | ± 0      | 36 | 84 | 36 | ± 0    | 10 | 84 | 08 | ± 0 | 43 | 92 | 11 | ± 0   | 38 |
| GPRGNN+JDR  | 80 | 77 | ± 0  | 29 | 69 | 17 | ± 0      | 30 | 85 | 05 | ± 0    | 08 | 84 | 77 | ± 0 | 35 | 92 | 68 | ± 0   | 25 |

Table 3: Results on real-world heterophilic dataset in the dense splitting (60%/20%/20%): Mean accuracy across runs (%) ± 95% confidence interval. Best average accuracy in bold.

| Method      |    |    |     |    |    |     | Squirrel |    |    |    | Actor |    |    |    | Texas |    |    |    | Cornell |    |
|-------------|----|----|-----|----|----|-----|----------|----|----|----|-------|----|----|----|-------|----|----|----|---------|----|
| GCN         | 67 | 65 | ± 0 | 42 | 57 | 94  | ± 0      | 31 | 34 | 00 | ± 0   | 31 | 75 | 62 | ± 1   | 12 | 64 | 68 | ± 1     | 25 |
| GCN+DIGL    | 58 | 04 | ± 0 | 48 | 39 | 64  | ± 0      | 34 | 39 | 57 | ± 0   | 29 | 91 | 05 | ± 0   | 73 | 88 | 49 | ± 0     | 74 |
| GCN+FoSR    | 67 | 67 | ± 0 | 39 | 58 | 12  | ± 0      | 35 | 33 | 98 | ± 0   | 30 | 78 | 31 | ± 1   | 07 | 65 | 64 | ± 1     | 06 |
| GCN+BORF    | 67 | 78 | ± 0 | 43 |    | OOM |          |    | 33 | 95 | ± 0   | 31 | 76 | 66 | ± 1   | 10 | 68 | 72 | ± 1     | 11 |
| GCN+JDR     | 69 | 76 | ± 0 | 50 | 61 | 76  | ± 0      | 39 | 40 | 47 | ± 0   | 31 | 85 | 12 | ± 0   | 74 | 84 | 51 | ± 1     | 06 |
| GPRGNN      | 69 | 15 | ± 0 | 51 | 53 | 44  | ± 0      | 37 | 39 | 52 | ± 0   | 22 | 92 | 82 | ± 0   | 67 | 87 | 79 | ± 0     | 89 |
| GPRGNN+DIGL | 66 | 57 | ± 0 | 46 | 42 | 98  | ± 0      | 37 | 39 | 61 | ± 0   | 21 | 91 | 11 | ± 0   | 72 | 88 | 06 | ± 0     | 81 |
| GPRGNN+FoSR | 68 | 96 | ± 0 | 45 | 52 | 34  | ± 0      | 37 | 39 | 47 | ± 0   | 21 | 93 | 16 | ± 0   | 66 | 87 | 51 | ± 1     | 04 |
| GPRGNN+BORF | 69 | 44 | ± 0 | 56 |    | OOM |          |    | 39 | 55 | ± 0   | 20 | 93 | 53 | ± 0   | 68 | 88 | 83 | ± 1     | 06 |
| GPRGNN+JDR  | 71 | 00 | ± 0 | 50 | 60 | 62  | ± 0      | 38 | 41 | 89 | ± 0   | 24 | 93 | 85 | ± 0   | 54 | 89 | 45 | ± 0     | 84 |

GPRGNN achieving better results for ϕ → 1.0. With JDR, they become much more comparable to each other and closer to AMP-BP. Even though the hyperparameters of JDR were tuned using only GCN as a downstream model, it also improves the performance of GPRGNN for all ϕ. The general robustness to hyperparameter changes is also analyzed in detail in Appendix [A.6.](#page-30-0)

#### 3.2 RESULTS ON REAL-WORLD DATA

We evaluate JDR on five common homophilic benchmarks datasets, namely the citation graphs Cora, CiteSeer, PubMed [\(Sen et al., 2008\)](#page-14-7) and the Amazon co-purchase graphs Computers and Photo [\(McAuley et al., 2015\)](#page-14-8). For heterophilic datasets, we rely on the Wikipedia graphs Chameleon and Squirrel [\(Rozemberczki et al., 2021\)](#page-14-9), the WebKB datasets Texas and Cornell used in [Pei et al.](#page-14-10) [\(2020\)](#page-14-10) and the actor co-occurence network Actor [\(Tang et al., 2009\)](#page-15-7). To show the scalability of JDR on larger heterophilic datasets, we further report the results for the Yandex Q user network Questions [\(Platonov et al., 2023\)](#page-14-6) and the social networks Penn94 and Twitch-Gamers [\(Lim et al., 2021\)](#page-13-8). Further details about all datasets are in Appendix [A.4.](#page-21-0) Following [Chien et al.](#page-11-3) [\(2021\)](#page-11-3), we evaluate the homophilic datasets in the sparse splitting, staying close to the original setting of [Kipf and Welling](#page-13-2) [\(2017\)](#page-13-2) and the heterophilic datasets in dense splitting [\(Pei et al., 2020\)](#page-14-10). The remaining larger graphs are evaluated using their original splits. For further results and splits, see Appendix [A.5.](#page-21-1)

Homophilic Datasets. Table [2](#page-7-0) shows the results of JDR compared to the baselines. For both GNNs, JDR achieves the best results on four out of five datasets. GCN and GPRGNN with JDR achieve similar performance here, which is consistent with the findings for the homophilic cSBM. DIGL also performs strongly on the datasets and ranks first on Computers. However, with GPRGNN as a downstream model, the improvements are quite small. FoSR and BORF only marginally improve the performance of the GNNs in this setting.

Heterophilic Datasets. The results in Table [3](#page-7-1) show that GCN+JDR can catch up significantly compared to GPRGNN, but GPRGNN+JDR generally performs better. This is in line with the findings for the heterophilic cSBM. DIGL performs well on Actor, Texas and Cornell despite its inherent homophily assumption. The reason for this is the chosen smoothing kernel, which results in a graph that is evenly connected everywhere with small weights. GCN then largely ignores the graph and thus performs very similarly to an MLP, which performs already quite well on these datasets [\(Chien et al., 2021\)](#page-11-3). However, this fails for GPRGNN, which can make better use of the weak, complex graph structures. FoSR and BORF also improve performance here in most cases, but they are outperformed by JDR in all cases, often by a large margin. The out-of-memory error on Squirrel for BORF results from its computational complexity of O(md<sup>3</sup> max), because the dataset has a large number of edges m and a high maximum node degree dmax.

Table 4: Results on large datasets. Mean accuracy (ROC AUC for imbalanced Questions) across runs (%) ± 95% confidence interval. Best results in bold.

| Method   |    |    |    |     |    |    |    | Penn94 |     |    |    |     |   |     |    |
|----------|----|----|----|-----|----|----|----|--------|-----|----|----|-----|---|-----|----|
| # nodes  |    | 48 | ,  | 921 |    |    | 41 | ,      | 554 |    |    | 168 | , | 114 |    |
| # edges  |    | 0  | 15 | M   |    |    | 1  | 36     | M   |    |    | 6   | 8 | M   |    |
| GCN      | 75 | 31 | ±  | 0   | 81 | 80 | 40 | ±      | 0   | 18 | 64 | 56  | ± | 0   | 19 |
| GCN+DIGL | 73 | 35 | ±  | 0   | 64 | 74 | 70 | ±      | 0   | 32 | 61 | 64  | ± | 0   | 14 |
| GCN+FoSR | 75 | 51 | ±  | 0   | 73 | 80 | 54 | ±      | 0   | 31 | 64 | 65  | ± | 0   | 15 |
| GCN+JDR  | 77 | 52 | ±  | 0   | 63 | 82 | 30 | ±      | 0   | 61 | 65 | 14  | ± | 0   | 19 |

Larger Graphs. Scalability is a problem for preprocessing rewiring methods because applying them to large graphs requires significant amounts of memory and compute (see complexity in Table [1\)](#page-5-0). Since the decompositions needed for JDR can be truncated to the largest L vectors, it is still applicable to larger graphs. The experimental results on larger heterophilic datasets in Table [4](#page-8-0) verify this. They show that JDR can significantly

improve performance for these larger graphs, while FoSR only achieves marginal improvements. BORF ran out of memory on all of these datasets and DIGL is unable to improve due to its inherent homophily assumption. While scaling JDR to even larger graphs with millions of nodes is possible in principle, it requires more optimized and efficient implementations and is therefore left for future work.

# 4 RELATION TO PRIOR WORK

JDR is most related to preprocessing rewiring methods which we thus use as baselines. To provide a more thorough overview, we also place it within the extended literature.

Graph Rewiring. Recent work show that even when a graph correctly encodes interactions it may not be an effective *computational* graph for a GNN due to conditions such as *oversquashing* [\(Alon](#page-10-4) [and Yahav, 2021;](#page-10-4) [Di Giovanni et al., 2023\)](#page-11-5) and *oversmoothing* [\(Chen et al., 2020a\)](#page-11-6). Recently many methods have been proposed to address this, notably *graph rewiring methods*. They can be divided into preprocessing and end-to-end methods. Preprocessing methods rewire the graph using geometric and spectral properties, including curvature [\(Topping et al., 2022;](#page-15-2) [Nguyen et al., 2023;](#page-14-1) [Fesser and](#page-12-8) [Weber, 2024;](#page-12-8) [Bober et al., 2024\)](#page-10-5), expansion [\(Deac et al., 2022;](#page-11-7) [Banerjee et al., 2022\)](#page-10-6), effective resistance [\(Black et al., 2023;](#page-10-7) [Shen et al., 2024\)](#page-15-8), and spectral gap [\(Karhadkar et al., 2023\)](#page-12-2). Conceptually related is diffusion-based rewiring [\(Gasteiger et al., 2019\)](#page-12-7) that smooths the graph with a diffusion kernel. This can be interpreted as graph denoising, but is only suitable for homophilic graphs. Our approach is related to rewiring but with several key differences (see Table [1\)](#page-5-0). Our rewiring strategy aims to denoise the graph (rather than control some geometric property) with the goal to improve downstream performance, while the classical rewiring literature focuses on optimizing the graph for message passing computations.

Early end-to-end methods randomly drop edges during training to reduce oversmoothing [\(Rong](#page-14-11) [et al., 2020\)](#page-14-11). Subsequent work [\(Gutteridge et al., 2023;](#page-12-4) [Qian et al., 2024\)](#page-14-2) incorporates latent features to dynamically rewire the graph. [Ji et al.](#page-12-9) [\(2023\)](#page-12-9) use the estimated labels of a GNN to rewire the graph during training of the same GNN. [Giraldo et al.](#page-12-3) [\(2023\)](#page-12-3) use curvature information for dynamic rewiring. Graph Transformers [\(Dwivedi and Bresson, 2021;](#page-12-10) [Rampasek et al., 2022\)](#page-14-12) aim to overcome oversquashing in GNNs via global attention. In order to handle large graphs, these works still need to revert to sparse, non-global attention [\(Gabrielsson et al., 2023;](#page-12-11) [Shirzad et al., 2023\)](#page-15-9). Unlike preprocessing methods, end-to-end methods cannot output an improved graph which restricts their interpretability and reusability.

Graph Denoising. There is extensive literature on denoising signals on graphs using graph filters [\(Chen et al., 2014;](#page-11-8) [Ma et al., 2021b;](#page-13-5) [Liu et al., 2022\)](#page-13-9). However, we are interested in modifying the structure of the graph itself (rewiring), in a way that can benefit any downstream algorithm. [Dong](#page-11-9) [and Kluger](#page-11-9) [\(2023\)](#page-11-9) recently proposed a new metric to measure graph noise that correlates well with GCN performance. Based on this, they develop a method for graph denoising via self-supervised learning and link prediction. We discuss the relation to our work in detail in Appendix [A.5.7](#page-29-0) and also evaluate our rewired graphs using their ESNR metric there. More broadly, link prediction [\(Zhang](#page-15-10) [and Chen, 2018;](#page-15-10) [Pan et al., 2022\)](#page-14-0) can be seen as a tool for graph denoising; this perspective has been applied, for instance, to denoising neighborhood graphs arising in molecular imaging [\(Debarnot](#page-11-10) [et al., 2022\)](#page-11-10).

Graph Structure Learning. The aim of graph structure learning (GSL) [\(Zhu et al., 2022\)](#page-16-0) is to make GNNs more robust against adversarial perturbations of the graph or to learn a graph for data where there is no graph to start with [\(Jin et al., 2020;](#page-12-12) [Chen et al., 2020b;](#page-11-11) [Wang et al., 2024;](#page-15-11) [Zhu et al.,](#page-16-1) [2024\)](#page-16-1). [Lv et al.](#page-13-10) [\(2023\)](#page-13-10) build a neighborhood graph over features and interpolate between it and the input graph, which is a form of alignment. Unlike our method, they do not use spectral information, are unable to deal with noisy features and are only suitable for homophilic graphs where similaritybased connection rules apply. Even though both our work and GSL consider noisy graph settings, they are conceptually very different. We do not add noise to graph datasets which corresponds to a perturbation rate of 0 ("clean data") in GSL nomenclature. Instead, we acknowledge that in every real world dataset, there is noise in the graph structure and the node features, and one manifestation of this noise is a misalignment of their leading eigenspaces. We then use this to rewire the graph (and denoise features) so as to improve the overall node classification performance of downstream GNNs. Naturally, GSL methods have difficulties to improve over baselines in this setting [\(Jin et al.,](#page-12-12) [2020;](#page-12-12) [Dong and Kluger, 2023\)](#page-11-9). Our method, on the other hand, is not designed to handle strong perturbations and therefore cannot compete with GSL methods developed for specifically this purpose.

Graph Regularization. Laplacian regularization [\(Ando and Zhang, 2006\)](#page-10-8), originally stemming from semi-supervised representation learning, has been adapted by recent methods [\(Yang et al.,](#page-15-12) [2021;](#page-15-12) [Ma et al., 2021a\)](#page-13-11) to also improve the performance of GNNs. An extra loss term is added during the GNNs training, which contains additional information about the graph structure to reduce oversmoothing. Their main limiting factor is the underlying homophiliy assumption: It is assumed that connected nodes are more likely to share the same label.

# 5 CONCLUSION AND LIMITATIONS

Our experimental results clearly show that spectral resonance is a powerful principle on which to build graph rewiring (and feature denoising) algorithms. JDR consistently outperforms existing rewiring methods DIGL, FoSR and BORF on both synthetic and real-world graph datasets. The smaller performance gains of GPRGNN suggest that this more powerful GNN is already able to leverage the complementary spectra of graphs and features to some extent.

The main limitation of JDR is that it cannot be used without node features. The preprocessing rewiring methods that we compare with do not have this limitation as they only use the graph for rewiring, but in turn, they cannot take advantage of features. Since JDR is the first method to jointly denoise the graph and the features, there are no other methods to which it could be directly compared. Our experiments thus highlight what advantage features bring to rewiring.

Furthermore, our results suggest that noise in real-world graphs is an important limiting factor for the performance of GNNs. It would be interesting to see whether feature-agnostic rewiring from a denoising perspective, for example using link prediction, could be used to improve the downstream performance. A related idea that we tested but could not get to work well is to combine existing geometric rewiring algorithms with JDR. Intuitively, there should be a way to benefit from both removing noise and facilitating computation, but we have to leave that exploration for future work.

We also note that most current rewiring methods can be applied to graph level tasks, while JDR is currently limited to node classification. It is an open question how to extend the cSBM idea to graph-level problems.

#### ACKNOWLEDGMENTS

JL, CS and ID were supported by the European Research Council (ERC) Starting Grant 852821— SWING.

# REFERENCES


[1] E. Abbe. Community detection and stochastic block models: Recent developments. *Journal of Machine Learning Research*, 18(177):1–86, 2018. URL [http://jmlr.org/papers/v18/](http://jmlr.org/papers/v18/16-480.html) [16-480.html](http://jmlr.org/papers/v18/16-480.html).

[2] U. Alon and E. Yahav. On the bottleneck of graph neural networks and its practical implications. In *International Conference on Learning Representations*, 2021. URL [https://openreview.](https://openreview.net/forum?id=i80OPhOCVH2) [net/forum?id=i80OPhOCVH2](https://openreview.net/forum?id=i80OPhOCVH2).

[3] R. Ando and T. Zhang. Learning on Graph with Laplacian Regularization. In B. Schölkopf,
  - J. Platt, and T. Hoffman, editors, *Advances in Neural Information Processing Systems*, volume 19. MIT Press, 2006. URL [https://proceedings.neurips.cc/paper\\_files/paper/](https://proceedings.neurips.cc/paper_files/paper/2006/file/d87c68a56bc8eb803b44f25abb627786-Paper.pdf) [2006/file/d87c68a56bc8eb803b44f25abb627786-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2006/file/d87c68a56bc8eb803b44f25abb627786-Paper.pdf).

[4] J. Baik, G. B. Arous, and S. Péché. Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. *The Annals of Probability*, 33(5):1643 – 1697, 2005. doi: 10.1214/ 009117905000000233. URL <https://doi.org/10.1214/009117905000000233>.

[5] P. K. Banerjee, K. Karhadkar, Y. G. Wang, U. Alon, and G. Montúfar. Oversquashing in gnns through the lens of information contraction and graph expansion. In *2022 58th Annual Allerton Conference on Communication, Control, and Computing (Allerton)*, page 1–8. IEEE Press, 2022. doi: 10.1109/Allerton49937.2022.9929363. URL [https://doi.org/10.1109/](https://doi.org/10.1109/Allerton49937.2022.9929363) [Allerton49937.2022.9929363](https://doi.org/10.1109/Allerton49937.2022.9929363).

[6] A. Baranwal, K. Fountoulakis, and A. Jagannath. Graph convolution for semi-supervised classification: Improved linear separability and out-of-distribution generalization. In M. Meila and
  - T. Zhang, editors, *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pages 684–693. PMLR, 07 2021. URL <https://proceedings.mlr.press/v139/baranwal21a.html>.

[7] P. W. Battaglia, J. B. Hamrick, V. Bapst, A. Sanchez-Gonzalez, V. Zambaldi, M. Malinowski,
  - A. Tacchetti, D. Raposo, A. Santoro, R. Faulkner, C. Gulcehre, F. Song, A. Ballard, J. Gilmer,
  - G. Dahl, A. Vaswani, K. Allen, C. Nash, V. Langston, C. Dyer, N. Heess, D. Wierstra, P. Kohli,
  - M. Botvinick, O. Vinyals, Y. Li, and R. Pascanu. Relational inductive biases, deep learning, and graph networks, 2018.

[8] F. Benaych-Georges and R. R. Nadakuditi. The singular values and vectors of low rank perturbations of large rectangular random matrices. *Journal of Multivariate Analysis*, 111:120–135, 2012.

[9] F. Benaych-Georges, A. Guionnet, and M. Maida. Fluctuations of the extreme eigenvalues of finite rank deformations of random matrices. *Electronic Journal of Probability*, 16:1621–1662, 2011.

[10] M. Black, Z. Wan, A. Nayyeri, and Y. Wang. Understanding oversquashing in GNNs through the lens of effective resistance. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 2528–2547. PMLR, 7 2023. URL <https://proceedings.mlr.press/v202/black23a.html>.

[11] J. Bober, A. Monod, E. Saucan, and K. N. Webster. Rewiring networks for graph neural network training using discrete geometry. In H. Cherifi, L. M. Rocha, C. Cherifi, and M. Donduran, editors, *Complex Networks & Their Applications XII*, pages 225–236, Cham, 2024. Springer Nature Switzerland. ISBN 978-3-031-53468-3.

[12] S. Chanpuriya and C. Musco. Simplified graph convolution with heterophily. In S. Koyejo,
  - S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, *Advances in Neural Information Processing Systems*, volume 35, pages 27184–27197. Curran Associates, Inc.,

[13] 2022. URL [https://proceedings.neurips.cc/paper\\_files/paper/2022/](https://proceedings.neurips.cc/paper_files/paper/2022/file/ae07d152c51ea2ddae65aa7192eb5ff7-Paper-Conference.pdf) [file/ae07d152c51ea2ddae65aa7192eb5ff7-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/ae07d152c51ea2ddae65aa7192eb5ff7-Paper-Conference.pdf).

[14] D. Chen, Y. Lin, W. Li, P. Li, J. Zhou, and X. Sun. Measuring and relieving the over-smoothing problem for graph neural networks from the topological view. *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(04):3438–3445, 4 2020a. doi: 10.1609/aaai.v34i04.5747. URL <https://ojs.aaai.org/index.php/AAAI/article/view/5747>.

[15] S. Chen, A. Sandryhaila, J. M. F. Moura, and J. Kovacevic. Signal denoising on graphs via graph filtering. In *2014 IEEE Global Conference on Signal and Information Processing (GlobalSIP)*, pages 872–876, 2014. doi: 10.1109/GlobalSIP.2014.7032244.

[16] Y. Chen, L. Wu, and M. Zaki. Iterative deep graph learning for graph neural networks: Better and robust node embeddings. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, *Advances in Neural Information Processing Systems*, volume 33, pages 19314–19326. Curran Associates, Inc., 2020b. URL [https://proceedings.neurips.cc/paper\\_files/](https://proceedings.neurips.cc/paper_files/paper/2020/file/e05c7ba4e087beea9410929698dc41a6-Paper.pdf) [paper/2020/file/e05c7ba4e087beea9410929698dc41a6-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/e05c7ba4e087beea9410929698dc41a6-Paper.pdf).

[17] E. Chien, J. Peng, P. Li, and O. Milenkovic. Adaptive universal generalized pagerank graph neural network. In *International Conference on Learning Representations*, 2021. URL [https://](https://openreview.net/forum?id=n6jl7fLxrP) [openreview.net/forum?id=n6jl7fLxrP](https://openreview.net/forum?id=n6jl7fLxrP).

[18] E. Chien, W.-C. Chang, C.-J. Hsieh, H.-F. Yu, J. Zhang, O. Milenkovic, and I. S. Dhillon. Node feature extraction by self-supervised multi-scale neighborhood prediction. In *International Conference on Learning Representations*, 2022. URL [https://openreview.net/forum?](https://openreview.net/forum?id=KJggliHbs8) [id=KJggliHbs8](https://openreview.net/forum?id=KJggliHbs8).

[19] A. Deac, M. Lackenby, and P. Velickovi ˇ c. Expander graph propagation. In ´ *NeurIPS 2022 Workshop on Symmetry and Geometry in Neural Representations*, 2022. URL [https://openreview.](https://openreview.net/forum?id=6cthqh2qhCT) [net/forum?id=6cthqh2qhCT](https://openreview.net/forum?id=6cthqh2qhCT).

[20] V. Debarnot, V. Kishore, C. Shi, and I. Dokmanic. Manifold rewiring for unlabeled imaging. In ´ *2022 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*, pages 1–8, 2022. doi: 10.23919/APSIPAASC55919.2022.9980168.

[21] A. Decelle, F. Krzakala, C. Moore, and L. Zdeborová. Asymptotic analysis of the stochastic block model for modular networks and its algorithmic applications. *Phys. Rev. E*, 84:066106, Dec 2011. doi: 10.1103/PhysRevE.84.066106. URL [https://link.aps.org/doi/10.](https://link.aps.org/doi/10.1103/PhysRevE.84.066106) [1103/PhysRevE.84.066106](https://link.aps.org/doi/10.1103/PhysRevE.84.066106).

[22] Y. Deshpande, S. Sen, A. Montanari, and E. Mossel. Contextual stochastic block models. *Advances in Neural Information Processing Systems*, 31, 2018.

[23] F. Di Giovanni, L. Giusti, F. Barbero, G. Luise, P. Lio, and M. M. Bronstein. On over-squashing in message passing neural networks: The impact of width, depth, and topology. In A. Krause,
  - E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 7865–7885. PMLR, 07 2023. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v202/di-giovanni23a.html) [v202/di-giovanni23a.html](https://proceedings.mlr.press/v202/di-giovanni23a.html).

[24] M. Dong and Y. Kluger. Towards understanding and reducing graph structural noise for GNNs. In
  - A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 8202–8226. PMLR, 07 2023. URL [https://proceedings.](https://proceedings.mlr.press/v202/dong23a.html) [mlr.press/v202/dong23a.html](https://proceedings.mlr.press/v202/dong23a.html).

[25] O. Duranthon and L. Zdeborová. Optimal inference in contextual stochastic block models, 2023.

[26] D. K. Duvenaud, D. Maclaurin, J. Iparraguirre, R. Bombarell, T. Hirzel, A. Aspuru-Guzik, and R. P. Adams. Convolutional networks on graphs for learning molecular fingerprints. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, *Advances in Neural Information Processing Systems*, volume 28. Curran Associates, Inc., 2015. URL [https://proceedings.neurips.cc/paper\\_files/paper/2015/](https://proceedings.neurips.cc/paper_files/paper/2015/file/f9be311e65d81a9ad8150a60844bb94c-Paper.pdf) [file/f9be311e65d81a9ad8150a60844bb94c-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2015/file/f9be311e65d81a9ad8150a60844bb94c-Paper.pdf).

[27] V. P. Dwivedi and X. Bresson. A generalization of transformer networks to graphs. *AAAI Workshop on Deep Learning on Graphs: Methods and Applications*, 2021.

[28] P. Erdös and A. Rényi. On random graphs i. *Publicationes Mathematicae Debrecen*, 6:290–297, 1959.

[29] L. Fesser and M. Weber. Mitigating over-smoothing and over-squashing using augmentations of forman-ricci curvature. In S. Villar and B. Chamberlain, editors, *Proceedings of the Second Learning on Graphs Conference*, volume 231 of *Proceedings of Machine Learning Research*, pages 19:1–19:28. PMLR, 11 2024. URL [https://proceedings.mlr.press/v231/](https://proceedings.mlr.press/v231/fesser24a.html) [fesser24a.html](https://proceedings.mlr.press/v231/fesser24a.html).

[30] M. Fey and J. E. Lenssen. Fast graph representation learning with PyTorch Geometric. In *ICLR Workshop on Representation Learning on Graphs and Manifolds*, 2019.

[31] R. B. Gabrielsson, M. Yurochkin, and J. Solomon. Rewiring with positional encodings for graph neural networks. *Transactions on Machine Learning Research*, 2023. ISSN 2835-8856. URL <https://openreview.net/forum?id=dn3ZkqG2YV>.

[32] J. Gasteiger, S. Weiß enberger, and S. Günnemann. Diffusion improves graph learning. In
  - H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019. URL [https://proceedings.neurips.cc/paper\\_files/paper/2019/](https://proceedings.neurips.cc/paper_files/paper/2019/file/23c894276a2c5a16470e6a31f4618d73-Paper.pdf) [file/23c894276a2c5a16470e6a31f4618d73-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2019/file/23c894276a2c5a16470e6a31f4618d73-Paper.pdf).

[33] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl. Neural message passing for quantum chemistry. In *Proceedings of the 34th International Conference on Machine Learning - Volume 70*, ICML'17, page 1263–1272. JMLR.org, 2017.

[34] J. H. Giraldo, K. Skianis, T. Bouwmans, and F. D. Malliaros. On the trade-off between oversmoothing and over-squashing in deep graph neural networks. CIKM '23, page 566–576, New York, NY, USA, 2023. Association for Computing Machinery. doi: 10.1145/3583780.3614997. URL <https://doi.org/10.1145/3583780.3614997>.

[35] V. Gligorijevic, P. D. Renfrew, T. Kosciolek, J. K. Leman, D. Berenberg, T. Vatanen, C. Chandler, ´
  - B. C. Taylor, I. M. Fisk, H. Vlamakis, R. J. Xavier, R. Knight, K. Cho, and R. Bonneau. Structurebased protein function prediction using graph convolutional networks. *Nature Communications*, 12(1):3168, 2021. ISSN 2041-1723. doi: 10.1038/s41467-021-23303-9. URL [https://doi.](https://doi.org/10.1038/s41467-021-23303-9) [org/10.1038/s41467-021-23303-9](https://doi.org/10.1038/s41467-021-23303-9).

[36] B. Gutteridge, X. Dong, M. M. Bronstein, and F. Di Giovanni. DRew: Dynamically rewired message passing with delay. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 12252–12267. PMLR, 07 2023. URL <https://proceedings.mlr.press/v202/gutteridge23a.html>.

[37] H. Hu and Y. M. Lu. Universality laws for high-dimensional learning with random features. *IEEE Transactions on Information Theory*, 69(3):1932–1964, 2023. doi: 10.1109/TIT.2022.3217698.

[38] F. Ji, S. H. Lee, H. Meng, K. Zhao, J. Yang, and W. P. Tay. Leveraging label non-uniformity for node classification in graph neural networks. In *Proceedings of the 40th International Conference on Machine Learning*, ICML'23. JMLR.org, 2023.

[39] W. Jin, Y. Ma, X. Liu, X. Tang, S. Wang, and J. Tang. Graph structure learning for robust graph neural networks. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, KDD '20, page 66–74, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450379984. doi: 10.1145/3394486.3403049. URL <https://doi.org/10.1145/3394486.3403049>.

[40] K. Karhadkar, P. K. Banerjee, and G. Montufar. FoSR: First-order spectral rewiring for addressing oversquashing in GNNs. In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=3YjQfCLdrzz>.

[41] T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. In *International Conference on Learning Representations*, 2017. URL [https://openreview.](https://openreview.net/forum?id=SJU4ayYgl) [net/forum?id=SJU4ayYgl](https://openreview.net/forum?id=SJU4ayYgl).

[42] V. Kothapalli, T. Tirer, and J. Bruna. A neural collapse perspective on feature evolution in graph neural networks. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. URL <https://openreview.net/forum?id=sxao2udWXi>.

[43] T. Lesieur, F. Krzakala, and L. Zdeborová. Constrained low-rank matrix estimation: phase transitions, approximate message passing and applications. *Journal of Statistical Mechanics: Theory and Experiment*, 2017(7):073403, jul 2017. doi: 10.1088/1742-5468/aa7284. URL [https:](https://dx.doi.org/10.1088/1742-5468/aa7284) [//dx.doi.org/10.1088/1742-5468/aa7284](https://dx.doi.org/10.1088/1742-5468/aa7284).

[44] P. Li, I. Chien, and O. Milenkovic. Optimizing generalized pagerank methods for seed-expansion community detection. In *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019. URL [https://proceedings.neurips.cc/paper\\_files/](https://proceedings.neurips.cc/paper_files/paper/2019/file/9ac1382fd8fc4b631594aa135d16ad75-Paper.pdf) [paper/2019/file/9ac1382fd8fc4b631594aa135d16ad75-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2019/file/9ac1382fd8fc4b631594aa135d16ad75-Paper.pdf).

[45] D. Lim, F. M. Hohne, X. Li, S. L. Huang, V. Gupta, O. P. Bhalerao, and S.-N. Lim. Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods. In A. Beygelzimer,
  - Y. Dauphin, P. Liang, and J. W. Vaughan, editors, *Advances in Neural Information Processing Systems*, 2021. URL <https://openreview.net/forum?id=DfGu8WwT0d>.

[46] J. Linkerhägner, N. Freymuth, P. M. Scheikl, F. Mathis-Ullrich, and G. Neumann. Grounding graph network simulators using physical sensor observations. In *The Eleventh International Conference on Learning Representations*, 2023. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=jsZsEd8VEY) [jsZsEd8VEY](https://openreview.net/forum?id=jsZsEd8VEY).

[47] S. Liu, R. Ying, H. Dong, L. Lin, J. Chen, and D. Wu. How powerful is implicit denoising in graph neural networks, 2022.

[48] C. Lu and S. Sen. Contextual stochastic block model: Sharp thresholds and contiguity. *Journal of Machine Learning Research*, 24(54):1–34, 2023.

[49] S. Luan, C. Hua, M. Xu, Q. Lu, J. Zhu, X.-W. Chang, J. Fu, J. Leskovec, and
  - D. Precup. When do graph neural networks help with node classification? investigating the homophily principle on node distinguishability. In A. Oh, T. Naumann,
  - A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, *Advances in Neural Information Processing Systems*, volume 36, pages 28748–28760. Curran Associates, Inc., 2023. URL [https://proceedings.neurips.cc/paper\\_files/paper/2023/](https://proceedings.neurips.cc/paper_files/paper/2023/file/5ba11de4c74548071899cf41dec078bf-Paper-Conference.pdf) [file/5ba11de4c74548071899cf41dec078bf-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/5ba11de4c74548071899cf41dec078bf-Paper-Conference.pdf).

[50] S. Lv, G. Wen, S. Liu, L. Wei, and M. Li. Robust graph structure learning with the alignment of features and adjacency matrix, 2023.

[51] X. Ma, H. Chen, and G. Song. Lereg: Empower graph neural networks with local energy regularization. In *Proceedings of the 30th ACM International Conference on Information & Knowledge Management*, CIKM '21, page 1191–1201, New York, NY, USA, 2021a. Association for Computing Machinery. ISBN 9781450384469. doi: 10.1145/3459637.3482447. URL <https://doi.org/10.1145/3459637.3482447>.

[52] Y. Ma, X. Liu, T. Zhao, Y. Liu, J. Tang, and N. Shah. A unified view on graph neural networks as graph signal denoising. In *Proceedings of the 30th ACM International Conference on Information & Knowledge Management*, CIKM '21, page 1202–1211, New York, NY, USA, 2021b. ISBN 9781450384469. doi: 10.1145/3459637.3482225.

[53] Y. Ma, X. Liu, N. Shah, and J. Tang. Is homophily a necessity for graph neural networks? In *International Conference on Learning Representations*, 2022. URL [https://openreview.](https://openreview.net/forum?id=ucASPPD9GKN) [net/forum?id=ucASPPD9GKN](https://openreview.net/forum?id=ucASPPD9GKN).

[54] R. Mandal, C. Casert, and P. Sollich. Robust prediction of force chains in jammed solids using graph neural networks. *Nature Communications*, 13:4424, 07 2022. doi: 10.1038/s41467-022-31732-3.

[55] J. McAuley, C. Targett, Q. Shi, and A. van den Hengel. Image-based recommendations on styles and substitutes. In *Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval*, SIGIR '15, page 43–52, New York, NY, USA, 2015. Association for Computing Machinery. ISBN 9781450336215. doi: 10.1145/2766462.2767755. URL <https://doi.org/10.1145/2766462.2767755>.

[56] M. McPherson, L. Smith-Lovin, and J. M. Cook. Birds of a feather: Homophily in social networks. *Annual Review of Sociology*, 27:415–444, 2001. ISSN 03600572, 15452115. URL [http://](http://www.jstor.org/stable/2678628) [www.jstor.org/stable/2678628](http://www.jstor.org/stable/2678628).

[57] R. R. Nadakuditi and M. E. Newman. Graph spectra and the detectability of community structure in networks. *Physical review letters*, 108(18):188701, 2012.

[58] A. Ng, M. Jordan, and Y. Weiss. On spectral clustering: Analysis and an algorithm. In T. Dietterich,
  - S. Becker, and Z. Ghahramani, editors, *Advances in Neural Information Processing Systems*, volume 14. MIT Press, 2001. URL [https://proceedings.neurips.cc/paper\\_files/](https://proceedings.neurips.cc/paper_files/paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf) [paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf).

[59] K. Nguyen, N. M. Hieu, V. D. Nguyen, N. Ho, S. Osher, and T. M. Nguyen. Revisiting oversmoothing and over-squashing using ollivier-ricci curvature. In A. Krause, E. Brunskill, K. Cho,
  - B. Engelhardt, S. Sabato, and J. Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 25956–25979. PMLR, 7 2023. URL [https://proceedings.mlr.press/v202/](https://proceedings.mlr.press/v202/nguyen23c.html) [nguyen23c.html](https://proceedings.mlr.press/v202/nguyen23c.html).

[60] L. Pan, C. Shi, and I. Dokmanic. Neural link prediction with walk pooling. In ´ *International Conference on Learning Representations*, 2022. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=CCu6RcUMwK0) [CCu6RcUMwK0](https://openreview.net/forum?id=CCu6RcUMwK0).

[61] D. Paul. Asymptotics of sample eigenstructure for a large dimensional spiked covariance model. *Statistica Sinica*, pages 1617–1642, 2007.

[62] H. Pei, B. Wei, K. C.-C. Chang, Y. Lei, and B. Yang. Geom-gcn: Geometric graph convolutional networks. In *International Conference on Learning Representations*, 2020. URL [https://](https://openreview.net/forum?id=S1e2agrFvS) [openreview.net/forum?id=S1e2agrFvS](https://openreview.net/forum?id=S1e2agrFvS).

[63] O. Platonov, D. Kuznedelev, M. Diskin, A. Babenko, and L. Prokhorenkova. A critical look at the evaluation of GNNs under heterophily: Are we really making progress? In *The Eleventh International Conference on Learning Representations*, 2023. URL [https://openreview.](https://openreview.net/forum?id=tJbbQfw-5wv) [net/forum?id=tJbbQfw-5wv](https://openreview.net/forum?id=tJbbQfw-5wv).

[64] C. Qian, A. Manolache, K. Ahmed, Z. Zeng, G. V. den Broeck, M. Niepert, and C. Morris. Probabilistically rewired message-passing neural networks. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=Tj6Wcx7gVk) [Tj6Wcx7gVk](https://openreview.net/forum?id=Tj6Wcx7gVk).

[65] L. Rampasek, M. Galkin, V. P. Dwivedi, A. T. Luu, G. Wolf, and D. Beaini. Recipe for a general, powerful, scalable graph transformer. In A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, editors, *Advances in Neural Information Processing Systems*, 2022. URL [https://openreview.](https://openreview.net/forum?id=lMMaNf6oxKM) [net/forum?id=lMMaNf6oxKM](https://openreview.net/forum?id=lMMaNf6oxKM).

[66] Y. Rong, W. Huang, T. Xu, and J. Huang. Dropedge: Towards deep graph convolutional networks on node classification. In *International Conference on Learning Representations*, 2020. URL <https://openreview.net/forum?id=Hkx1qkrKPr>.

[67] B. Rozemberczki, C. Allen, and R. Sarkar. Multi-Scale attributed node embedding. *Journal of Complex Networks*, 9(2):cnab014, 05 2021. ISSN 2051-1329. doi: 10.1093/comnet/cnab014. URL <https://doi.org/10.1093/comnet/cnab014>.

[68] P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Galligher, and T. Eliassi-Rad. Collective classification in network data. *AI Magazine*, 29(3):93, 9 2008. doi: 10.1609/aimag.v29i3.2157. URL [https:](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2157) [//ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2157](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2157).

[69] X. Shen, P. Lio, L. Yang, R. Yuan, Y. Zhang, and C. Peng. Graph rewiring and preprocessing for graph neural networks based on effective resistance. *IEEE Transactions on Knowledge and Data Engineering*, pages 1–14, 2024. doi: 10.1109/TKDE.2024.3397692.

[70] C. Shi, L. Pan, H. Hu, and I. Dokmanic. Homophily modulates double descent generalization ´ in graph convolution networks. *Proceedings of the National Academy of Sciences*, 121(8): e2309504121, 2024. doi: 10.1073/pnas.2309504121. URL [https://www.pnas.org/doi/](https://www.pnas.org/doi/abs/10.1073/pnas.2309504121) [abs/10.1073/pnas.2309504121](https://www.pnas.org/doi/abs/10.1073/pnas.2309504121).

[71] J. Shi and J. Malik. Normalized cuts and image segmentation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(8):888–905, 2000. doi: 10.1109/34.868688.

[72] H. Shirzad, A. Velingker, B. Venkatachalam, D. J. Sutherland, and A. K. Sinop. Exphormer: Sparse transformers for graphs. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and
  - J. Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 31613–31632. PMLR, 07 2023. URL <https://proceedings.mlr.press/v202/shirzad23a.html>.

[73] J. Tang, J. Sun, C. Wang, and Z. Yang. Social influence analysis in large-scale networks. In *Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, KDD '09, page 807–816, New York, NY, USA, 2009. Association for Computing Machinery. ISBN 9781605584959. doi: 10.1145/1557019.1557108. URL [https:](https://doi.org/10.1145/1557019.1557108) [//doi.org/10.1145/1557019.1557108](https://doi.org/10.1145/1557019.1557108).

[74] J. Topping, F. D. Giovanni, B. P. Chamberlain, X. Dong, and M. M. Bronstein. Understanding oversquashing and bottlenecks on graphs via curvature. In *International Conference on Learning Representations*, 2022. URL <https://openreview.net/forum?id=7UmjRGzp-A>.

[75] J. Wang, J. Guo, Y. Sun, J. Gao, S. Wang, Y. Yang, and B. Yin. DGNN: Decoupled Graph Neural Networks with Structural Consistency between Attribute and Graph Embedding Representations, 2024.

[76] X. Wu, Z. Chen, W. W. Wang, and A. Jadbabaie. A non-asymptotic analysis of oversmoothing in graph neural networks. In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=CJd-BtnwtXq>.

[77] K. Xu, W. Hu, J. Leskovec, and S. Jegelka. How powerful are graph neural networks? In *International Conference on Learning Representations*, 2019. URL [https://openreview.net/](https://openreview.net/forum?id=ryGs6iA5Km) [forum?id=ryGs6iA5Km](https://openreview.net/forum?id=ryGs6iA5Km).

[78] H. Yang, K. Ma, and J. Cheng. Rethinking graph regularization for graph neural networks. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(5):4573–4581, 05 2021. doi: 10. 1609/aaai.v35i5.16586. URL [https://ojs.aaai.org/index.php/AAAI/article/](https://ojs.aaai.org/index.php/AAAI/article/view/16586) [view/16586](https://ojs.aaai.org/index.php/AAAI/article/view/16586).

[79] M. Zhang and Y. Chen. Link prediction based on graph neural networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018. URL [https://proceedings.neurips.cc/paper\\_files/paper/2018/](https://proceedings.neurips.cc/paper_files/paper/2018/file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf) [file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2018/file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf).

[80] J. Zhou, G. Cui, S. Hu, Z. Zhang, C. Yang, Z. Liu, L. Wang, C. Li, and M. Sun. Graph neural networks: A review of methods and applications. *AI Open*, 1:57–81, 2020. ISSN 2666-6510. doi: https://doi.org/10.1016/j.aiopen.2021.01.001.

[81] J. Zhu, Y. Yan, L. Zhao, M. Heimann, L. Akoglu, and D. Koutra. Beyond homophily in graph neural networks: Current limitations and effective designs. In H. Larochelle,
  - M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, *Advances in Neural Information Processing Systems*, volume 33, pages 7793–7804. Curran Associates, Inc., 2020. URL [https://proceedings.neurips.cc/paper\\_files/paper/2020/](https://proceedings.neurips.cc/paper_files/paper/2020/file/58ae23d878a47004366189884c2f8440-Paper.pdf) [file/58ae23d878a47004366189884c2f8440-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/58ae23d878a47004366189884c2f8440-Paper.pdf).

[82] Y. Zhu, W. Xu, J. Zhang, Y. Du, J. Zhang, Q. Liu, C. Yang, and S. Wu. A survey on graph structure learning: Progress and opportunities, 2022.

[83] Y. Zhu, R. Amor, Y. Chen, Z. Deng, L. Feng, and M. Witbrock. Robust node classification on graph data with graph and label noise. *Proceedings of the AAAI Conference on Artificial Intelligence*, 2024.
# A APPENDIX

# A.1 THE JDR ALGORITHM

|     | Algorithm   | 1 Joint Denoising and Rewiring                              |
|-----|-------------|-------------------------------------------------------------|
| 1:  | procedure   | R EWIRE ( X , A ) ▷ For D ENOISE just exchange X and A      |
| 2:  | X           | = U Σ W T                                                   |
| 3:  | A =         | V Λ V                                                       |
| 4:  | for         | i in range ( L A ) do ▷ Loop over L A eigenvectors in A     |
| 5:  |             | v a ← V [: , i ]                                            |
| 6:  |             | for j in range ( L A ) do ▷ Loop over L A eigenvectors in X |
| 7:  |             | u x ← U [: , j ]                                            |
| 8:  |             | θ ← ⟨ u x , v a ⟩ ▷ Find angle between eigenvectors         |
| 9:  |             | if   θ   >   θ max   then                                   |
| 10: |             | θ max ← θ                                                   |
| 11: |             | u                                                           |
|     |             | x ← u x                                                     |
| 12: |             | end if                                                      |
| 13: |             | end for                                                     |
| 14: |             | V ˜ [: , i ] ← (1 − η A ) v a + η A sign ( θ max ) u        |
|     |             | x ▷ Interpolation between eigenvectors                      |
| 15: | end         | for                                                         |
| 16: | A ˜ ←       | V ˜ Λ V ˜ T                                                 |
| 17: | end         | procedure                                                   |
| 18: | X ˜ , A ˜ ← | X , A                                                       |
| 19: | for i in    | range ( K ) do ▷ Main loop                                  |
| 20: | X ′         | ← D ENOISE ( X ˜ , A ˜)                                     |
| 21: | A ′         | ← R EWIRE ( X ˜ , A ˜)                                      |
| 22: | X ˜ ,       | A ˜ ← X ′                                                   |
|     |             | , A ′                                                       |
| 23: | end for     |                                                             |
| 24: | X ˜ = U     | PDATE _X ( X , X ˜ ) ▷ Sparsify and binarize if needed      |
| 25: | A ˜ = U     | PDATE _A ( A , A ˜)                                         |

#### A.1.1 LOW-DIMENSIONAL GRAPHS AND RELATION TO RESONANCE

Figure 5: Visualization of the first six eigenmodes of L of the 8×8 grid graph.

low-dimensional coordinates, We finally mention that although our algorithm is motivated by the cSBM, it could have equivalently been motivated by ubiquitous lowdimensional graphs. In such graphs, node labels are related to the which are in turn given by the eigenvectors of the graph Laplacian; this is illustrated in Figure [5.](#page-17-2) If, for example, the labels are given by the sign of the first non-constant eigenfunction (the slowest-changing normal mode), our notion of alignment with L = 1 clearly remains meaningful.

![](_page_17_Picture_6.jpeg)

This also further motivates our terminology of resonance. In a purist sense, resonance is a dynamical phenomenon where driving a system with a frequency corresponding to an eigenvalue of the Laplacian yields a diverging response. Importantly, the shape of the response is then an eigenfunction. In a broad sense, resonance signifies alignment with Laplacian eigenfunctions, which are the natural modes. For graphs, this is closely related to alignment with eigenvectors of the adjacency matrix (it is equivalent for d-regular graphs). As Figure [2b](#page-3-0) shows, maximizing alignment between feature and graph spectra indeed leads to the largest response of the graph to the features.

#### A.1.2 ROTATIONAL INVARIANCE OF ALIGNMENT

We show that the alignment measure [\(3\)](#page-3-2) is invariant to rotations of the subspaces for non-unique eigenvalues. Let A ∈ R <sup>N</sup>×<sup>N</sup> be the adjacency matrix and X ∈ <sup>R</sup> <sup>N</sup>×<sup>F</sup> the feature matrix. Let the eigendecomposition of A be

$$A = \lambda_1 U_1 U_1^T + \cdots + \lambda_p U_p U_p^T$$

where p ≤ N, U<sup>i</sup> ∈ <sup>R</sup> <sup>N</sup>×s<sup>i</sup> with s<sup>i</sup> being the multiplicity of the eigenvalue λ<sup>i</sup> , P i s<sup>i</sup> = N, U T <sup>i</sup> U<sup>i</sup> = I, and U T <sup>j</sup> U<sup>i</sup> = 0 for i ̸= j. Let similarly the SVD of X be

$$X = \sigma_1 V_1 W_1^T. + \cdots + \sigma_q V_q W_q^T,$$

where V<sup>i</sup> ∈ <sup>R</sup> N×t<sup>i</sup> , W<sup>i</sup> ∈ <sup>R</sup> F ×t<sup>i</sup> , P i t<sup>i</sup> = F, with analogous orthogonality conditions. Assume both (λi) and (σi) are sorted from largest to smallest. Assume also for simplicity that L = P<sup>p</sup> <sup>i</sup>=1 s<sup>i</sup> = P<sup>q</sup> <sup>i</sup>=1 t<sup>i</sup> so that the leading L-dimensional subspace of the graph A is spanned by the columns of the block matrix U<sup>L</sup> = [U<sup>1</sup> · · ·Up′ ]. Of course it is also spanned by the columns of <sup>U</sup>e<sup>L</sup> = [U1Q<sup>1</sup> · · ·Up′Qp′ ] where invertible matrices <sup>Q</sup><sup>i</sup> ∈ <sup>R</sup> si×s<sup>i</sup> reflect the fact that the eigensolver may return any of the infinitely many (when s<sup>i</sup> > 1) orthogonal bases for the subspaces spanned by the columns of U<sup>i</sup> . The Q<sup>i</sup> are orthogonal since UiQ<sup>i</sup> are orthogonal. Similarly the leading L-dimensional subspace of the features is spanned by the columns of V<sup>L</sup> = [V<sup>1</sup> · · · V<sup>q</sup> ′ ] but also of <sup>V</sup>e<sup>L</sup> = [V1R<sup>1</sup> · · · <sup>V</sup><sup>q</sup> ′R<sup>q</sup> ′ ] for any orthogonal (Ri). Now

$$\begin{aligned}\|\tilde{U}_L^T \tilde{V}_L\|_{\text{sp}} &= \|\text{blockdiag}(Q_1^T, \dots, Q_{p'}^T) U_L^T V_L \text{blockdiag}(R_1, \dots, R_{q'})\|_{\text{sp}} \\ &= \|U_L^T V_L\|_{\text{sp}}\end{aligned}$$

for any choice of Qs and Rs since both block-diagonal matrices are orthogonal and the spectral norm is unitarily-invariant.

#### A.1.3 COMPUTATIONAL COMPLEXITY

The complexity of JDR results mainly from the SVD and eigendecomposition, which is of order O(F Nmin(F, N)) for SVD and O(N<sup>3</sup> ) for the eigendecomposition (F = N). Since we only need the leading k eigenvectors this reduces to O(F N k). If the matrix is additionally sparse as is often the case for real-world graphs with binary node features this reduces further to O(nnz(A)k) where nnz(A) is the number of non-zero elements in A. Since usually neither the average degree d nor k is scaled by N, the complexity actually scales with O(N).

#### A.1.4 RUNTIME COMPARISON

In addition to computational complexity, we report measurements of running time of the different algorithms. We run JDR and baseline methods on the real-world datasets, using GCN as downstream model. All algorithms are run on Nvidia A100 with 80GB and we time their Python processes. We emphasize that we did not explicitly optimize the timing code and we kept the outputs and logging turned on. But this influences all methods in the same way so the relative comparisons are meaningful. The results in Table [5](#page-19-1) do not show a clear "winner". The ambiguity is especially visible on the large heterophilic graphs, where JDR is slower than DIGL, but significantly faster than FoSR on two datasets. On the Twitch-gamers dataset, on the other hand, it is faster than DIGL but not as fast as FoSR. The main reason for this is that different hyperparameters choices of the rewiring methods lead to dramatically different run times (even when applying the same method on the same dataset). For example on Computers, JDR is very fast since it only requires 3 denoising iterations, compared to 15 on Citeseer. The same holds true for FoSR which only require 5 iterations on Twitch-gamers, but 700 on Questions. So if one wants to optimize for speed, one should constrain the hyperparameters of the methods that significantly impact execution speed. Of course, there is a trade-off between any such constraint and accuracy, as our experiments on the denoising iterations of JDR in Figure [16](#page-33-0) (b) and Figure [17](#page-33-1) (b) in Appendix [A.6](#page-30-0) show.

#### A.2 CONTEXTUAL STOCHASTIC BLOCK MODELS

SBMs and GMMs are landmark theoretical models for studying clustering, classification problems and developing algorithmic tools. The cSBM [\(Deshpande et al., 2018\)](#page-11-1), a combination of the two,

Table 5: Timing experiments in seconds for different rewiring methods using GCN as downstream GNN. Smaller is better. We record the time of the preprocessing and training and evaluating the GNN on 100 random splits. The results do not show a clear winner; JDR generally requires a comparable or less time compared to baselines. For more discussion see [A.1.4.](#page-18-3)

| Dataset       | Base | DIGL | FoSR  | BORF | JDR  |
|---------------|------|------|-------|------|------|
| Cora          | 182  | 228  | 187   | 201  | 246  |
| Citeseer      | 258  | 360  | 291   | 290  | 433  |
| PubMed        | 291  | 692  | 416   | 897  | 858  |
| Computers     | 274  | 444  | 516   | 718  | 465  |
| Photo         | 213  | 299  | 220   | 801  | 330  |
| Chameleon     | 372  | 239  | 396   | 483  | 545  |
| Squirrel      | 1263 | 302  | 1282  |      | 1659 |
| Actor         | 166  | 286  | 171   | 225  | 319  |
| Texas         | 163  | 200  | 169   | 174  | 203  |
| Cornell       | 167  | 202  | 164   | 240  | 208  |
| Questions     | 93   | 804  | 11053 |      | 3707 |
| Penn94        | 164  | 232  | 3198  |      | 1779 |
| Twitch-gamers | 1579 | 6729 | 1618  |      | 5165 |

has become a key model for studying node classification problems on graphs, inspiring numerous designs of GNNs like GPRGNN [\(Chien et al., 2021\)](#page-11-3), GIANT [\(Chien et al., 2022\)](#page-11-12) or ASGC [\(Chan](#page-10-9)[puriya and Musco, 2022\)](#page-10-9). Many theoretical studies of node-level GNN problems are based on the cSBM, e.g. on double descent [\(Shi et al., 2024\)](#page-15-6), neural collapse [\(Kothapalli et al., 2023\)](#page-13-4), OOD generalization [\(Baranwal et al., 2021\)](#page-10-2), or oversmoothing [\(Wu et al., 2023\)](#page-15-5). Beyond being a standard synthetic benchmark, the cSBM is also used to to verify hypotheses about GNNs [\(Ma et al., 2022;](#page-13-12) [Luan et al., 2023\)](#page-13-13).

As for any model, the cSBM also comes with limitations. One possible limitation of our work is that cSBM assumes that the features are linear as in a GMM, which makes a linear classifier optimal. If the class boundaries are highly nonlinear, this is no longer true, and the spectrum of X may need to be "linearized", e.g. via Laplacian eigenmaps or diffusion maps. Still, the results on real-world data show that the cSBM model is already highly transferable, suggesting that the high-dimensional features in real-world graph datasets are often quite linear.

#### A.3 PROOF OF PROPOSITION [1](#page-5-1)

Notation. We order the eigenvalues and singular values from largest to smallest and denote the eigenvector associated with the eigenvalue λ<sup>j</sup> by v<sup>j</sup> (A<sup>c</sup> ). For the leading eigenvalue and eigenvectors of A<sup>c</sup> , we write λ<sup>1</sup> = λ<sup>A</sup> and v<sup>1</sup> (A<sup>c</sup> ) = vA. We use analogous notation for the singular values and corresponding singular vectors of X. For simplicity and without loss of generality we assume that the angles between these vectors are accute, i.e., ⟨vA,uX⟩,⟨y˜, vA⟩,⟨y˜,uX⟩ ≥ 0.

*Proof.* When λ > 1, based on the Baik-Ben Arous-Péché (BBP) transition [\(Baik et al., 2005;](#page-10-3) [Paul,](#page-14-13) [2007\)](#page-14-13), the leading eigenvalue of A lies outside the spectral bulk,

$$\lambda_A = \lambda + \frac{1}{\lambda} + \mathcal{O}_p\left(\frac{1}{\sqrt{N}}\right),$$

and the fluctuation of the leading eigenvector satisfies

$$\mathbf{q}_A := \lambda \left( \mathbf{v}_A - \sqrt{1 - \frac{1}{\lambda^2}} \tilde{\mathbf{y}} \right) \xrightarrow{d} \text{Haar}(\mathbb{S}_{\tilde{\mathbf{y}}^\perp}^{N-2}) \quad (5)$$

where Haar(S N−2 <sup>y</sup>˜<sup>⊥</sup> ) is the uniform distribution on the sphere orthogonal to y˜, <sup>S</sup> n−2 <sup>y</sup>˜<sup>⊥</sup> = v : v ∈ R <sup>N</sup> | v <sup>T</sup> y˜ = 0, ∥v∥ = 1 , and the convergence is in distribution as N → ∞. Similarly, for the rectangular matrix X we have [\(Benaych-Georges and Nadakuditi, 2012\)](#page-10-10),

$$\sigma_X = \sqrt{\frac{(\gamma + \mu)(1 + \mu)}{\mu}} + \mathcal{O}_p\left(\frac{1}{\sqrt{N}}\right)$$

and the fluctuation of the leading singular vector satisfies

$$\mathbf{q}_X := \sqrt{\frac{\mu(\mu + \gamma)}{\gamma(1 + \mu)}} \left( \mathbf{u}_X - \sqrt{1 - \frac{\gamma(1 + \mu)}{\mu(\mu + \gamma)} \tilde{\mathbf{y}}} \right) \xrightarrow{d} \text{Haar}(\mathbb{S}_{\tilde{\mathbf{y}}^\perp}^{N-2}).$$

To denoise A<sup>c</sup> , we adjust the leading eigenvector towards the direction of u<sup>X</sup> as

$$\tilde{\mathbf{v}}_A = (1 - \eta_A) \mathbf{v}_A + \eta_A \mathbf{u}_X$$

where η<sup>A</sup> > 0 is a small constant. The corresponding perturbation in the matrix reads

$$\begin{aligned} \mathbf{A}_{\eta_A}^c - \mathbf{A}^c &= \lambda_A(\tilde{\mathbf{v}}_A \tilde{\mathbf{v}}_A^T - \mathbf{v}_A \mathbf{v}_A^T) \\ &= \lambda_A \eta_A (\mathbf{v}_A \mathbf{u}_X^T + \mathbf{u}_X \mathbf{v}_A^T) + \mathcal{O}(\eta_A^2). \end{aligned}$$

The first-order perturbation of the leading eigenvector yields

$$v_1(\mathbf{A}_{\eta_A}^c) - v_1(\mathbf{A}^c) = \lambda_A \eta_A \sum_{j \neq 1} \frac{v_j(\mathbf{A}^c)^T (\mathbf{v}_A \mathbf{u}_X^T + \mathbf{u}_X \mathbf{v}_A^T) v_A}{\lambda_A - \lambda_j} v_j(\mathbf{A}^c) + \mathcal{O}(\eta_A^2).$$

Since v<sup>j</sup> (A<sup>c</sup> ) <sup>T</sup> v<sup>A</sup> = 0 for j > 1, we have

$$\begin{aligned} & \langle \mathbf{v}_1(\mathbf{A}_{\eta_A}^c) - \mathbf{v}_1(\mathbf{A}^c), \tilde{\mathbf{y}} \rangle \\ &= \lambda_A \eta_A \sum_{j \neq 1} \frac{\tilde{\mathbf{y}}^T \mathbf{v}_j(\mathbf{A}^c) \mathbf{v}_j(\mathbf{A}^c)^T \mathbf{u}_X \mathbf{v}_A^T \mathbf{v}_A}{\lambda_A - \lambda_j} + \mathcal{O}(\eta_A^2) \\ &= \lambda_A \eta_A c_1 \sum_{j \neq 1} \frac{(\tilde{\mathbf{y}}^T \mathbf{v}_j(\mathbf{A}^c))^2}{\lambda_A - \lambda_j} + \lambda_A \eta_A c_2 \sum_{j \neq 1} \frac{\tilde{\mathbf{y}}^T \mathbf{v}_j(\mathbf{A}^c) \mathbf{v}_j(\mathbf{A}^c)^T \mathbf{q}_X}{\lambda_A - \lambda_j} + \mathcal{O}(\eta_A^2) \end{aligned} \quad (6)$$

where c<sup>1</sup> = q 1 − γ(1+µ) µ(µ+γ) and c<sup>2</sup> = qγ(1+µ) µ(µ+γ) . From the BBP transition we know that when λ > 1 we have ⟨y˜, vA⟩ <sup>2</sup> = 1 − 1 <sup>λ</sup><sup>2</sup> <sup>+</sup> <sup>O</sup>p(1). Consequently, it follows that P <sup>j</sup≯=1 y˜ <sup>T</sup> v<sup>j</sup> (A<sup>c</sup> ) 2 = <sup>λ</sup><sup>2</sup> + <sup>O</sup>p(1). Since the edge of the bulk of spiked matrices still follows the Tracy–Widom distribution [\(Benaych-Georges et al., 2011\)](#page-10-11), i.e., λ<sup>2</sup> = 2 + O<sup>p</sup> N <sup>−</sup> <sup>2</sup> and λ<sup>N</sup> = −2 + O<sup>p</sup> N <sup>−</sup> <sup>2</sup> , we have

$$\frac{1}{\frac{\lambda}{\lambda} + \lambda + 2} + \mathcal{O}_p\left(N^{-\frac{2}{3}}\right) < \frac{1}{\lambda_A - \lambda_j} < \frac{1}{\frac{\lambda}{\lambda} + \lambda - 2} + \mathcal{O}_p\left(N^{-\frac{2}{3}}\right) \quad \text{for} \quad j > 1.$$

Therefore the first term in [\(6\)](#page-20-0) can be bounded as P j̸=1 (y˜ <sup>T</sup> v<sup>j</sup> (A<sup>c</sup> )) λA−λ<sup>j</sup> a.s > 1 λ A <sup>λ</sup> <sup>+</sup>λ+2 when <sup>N</sup> → ∞. For the second term, we note that the vector q<sup>X</sup> is independent of z := P j̸=1 y˜ <sup>T</sup> v<sup>j</sup> (A<sup>c</sup> λA−λ<sup>j</sup> v<sup>j</sup> (A<sup>c</sup> ). Each eigenvector v<sup>j</sup> (A<sup>c</sup> ) is uniformly distributed on S N−2 v<sup>⊥</sup> A and {v<sup>j</sup> (A<sup>c</sup> )} N <sup>j</sup>=2 is an orthogonal basis of R N−1 v<sup>⊥</sup> A = {v : v ∈ <sup>R</sup> <sup>N</sup> | v <sup>T</sup> v<sup>A</sup> = 0}. Therefore, for large N, y˜ <sup>T</sup> v<sup>j</sup> (A<sup>c</sup> ) is approximately independent of each element in v<sup>j</sup> (A<sup>c</sup> ). More precisely, the entries of z are of the order of O<sup>p</sup> √ 1 N , and thus P j̸=1 y˜ <sup>T</sup> v<sup>j</sup> (A<sup>c</sup> )v<sup>j</sup> (A<sup>c</sup> ) <sup>T</sup> q<sup>X</sup> λA−λ<sup>j</sup> = ⟨qX, z⟩ = O<sup>p</sup> √ N . Summarizing, we get

$$\langle \mathbf{v}_1(\mathbf{A}_{\eta_A}^c) - \mathbf{v}_1(\mathbf{A}^c), \tilde{\mathbf{y}} \rangle > \frac{\sqrt{1 - \frac{\gamma(1+\mu)}{\mu(\mu+\gamma)}}}{(\lambda + 1/\lambda)(\lambda + 1/\lambda + 2)} \eta_A + \mathcal{O}(\eta_A^2) \quad \text{when} \quad N \rightarrow \infty.$$

Table 6: Properties of the real-world benchmark datasets. For directed graphs we transform the graph to undirected in all experiments. H(G) indicates the homophily measure.

| Dataset        | Classes | Features |     |       | Nodes |         |       | Edges | Directed |   | H ( G ) |
|----------------|---------|----------|-----|-------|-------|---------|-------|-------|----------|---|---------|
| Cora           | 7       | 1 , 433  | 2   | , 708 |       | 5       | , 278 |       | False    | 0 | 810     |
| Citeseer       | 6       | 3 , 703  | 3   | , 327 |       | 4       | , 552 |       | False    | 0 | 736     |
| PubMed         | 3       | 500      | 19  | ,     | 717   | 44      | ,     | 324   | False    | 0 | 802     |
| Computers      | 10      | 767      | 13  | ,     | 752   | 245     |       | , 861 | False    | 0 | 777     |
| Photo          | 8       | 745      | 7   | , 650 |       | 119     |       | , 081 | False    | 0 | 827     |
| Chameleon      | 6       | 2 , 325  | 2   | , 277 |       | 31      | ,     | 371   | True     | 0 | 231     |
| Squirrel       | 5       | 2 , 089  | 5   | , 201 |       | 198     |       | , 353 | True     | 0 | 222     |
| Actor          | 5       | 932      | 7   | , 600 |       | 26      | ,     | 659   | True     | 0 | 219     |
| Texas          | 5       | 1 , 703  |     | 183   |       |         | 279   |       | True     | 0 | 087     |
| Cornell        | 5       | 1 , 703  |     | 183   |       |         | 277   |       | True     | 0 | 127     |
| Roman-empire   | 18      | 300      | 22  | ,     | 662   | 32      | ,     | 927   | False    | 0 | 047     |
| Amazon-ratings | 5       | 300      | 24  | ,     | 492   | 93      | ,     | 050   | False    | 0 | 380     |
| Minesweeper    | 2       | 7        | 10  | ,     | 000   | 39      | ,     | 402   | False    | 0 | 683     |
| Tolokers       | 2       | 10       | 11  | ,     | 758   | 519     |       | , 000 | False    | 0 | 595     |
| Questions      | 2       | 301      | 48  | ,     | 921   | 153     |       | , 540 | False    | 0 | 840     |
| Penn94         | 2       | 4 , 814  | 41  | ,     | 554   | 1 , 362 |       | , 229 | False    | 0 | 470     |
| Twitch-gamers  | 2       | 7        | 168 |       | , 114 | 6 , 797 |       | , 557 | False    | 0 | 545     |

#### A.4 DATASETS

Table [6](#page-21-2) shows the properties of the real-world datasets used. We also provide the homophily measure H(G) proposed in [Pei et al.](#page-14-10) [\(2020\)](#page-14-10), which we compute using the build-in function of Pytorch Geometric [\(Fey and Lenssen, 2019\)](#page-12-13). For the cSBM, following [\(Chien et al., 2021\)](#page-11-3), we choose N = 5000, F = 2000 and thus have γ = N <sup>F</sup> = 2.5. Since the threshold to recover communities in cSBM is λ <sup>2</sup> + µ <sup>2</sup>/γ > 1 [\(Deshpande et al., 2018\)](#page-11-1), we use a margin such that λ <sup>2</sup> + µ <sup>2</sup>/γ = 1 + ϵ. We choose the same ϵ = 3.25 as [Chien et al.](#page-11-3) [\(2021\)](#page-11-3) in all our experiments to be above the detection threshold and d = 5 to obtain a sparse graph to be close to the properties of real-world graphs. From the recovery threshold, we can parameterize the resulting arc of an ellipse with λ ≥ 0 and µ ≥ 0 using ϕ = arctan(λ √γ/µ). Table [7](#page-22-0) shows the parameters <sup>µ</sup> 2 and λ and the homophily measure H(G)for the different values of ϕ.

#### A.5 ADDITIONAL RESULTS

We provide a number of additional experiments which did not fit in the main text. These include more experiments on additional heterophilic datasets from [Platonov et al.](#page-14-6) [\(2023\)](#page-14-6), results for the homophilic datasets in the dense splitting, more experiments with DIGL [\(Gasteiger et al., 2019\)](#page-12-7), more alignment results and results for synthetic and real-world data using spectral clustering with and w/o JDR. The clustering experiments in particular allow an interpretation of how JDR works: Applying it to a graph increases its "spectral clusterability".

#### A.5.1 ADDITIONAL HETEROPHILIC DATASETS

In order to get a more comprehensive picture of the performance of JDR, we also test JDR on all the datasets proposed there by [Platonov et al.](#page-14-6) [\(2023\)](#page-14-6). Table [8](#page-22-1) shows the results on these datasets using their original splits and comparing DIGL, FoSR, BORF and JDR. In general, the performance increases are relatively small for all methods, but overall JDR still performs best. It achieves significant performance increases on Tolokers and Questions. For Minesweeper none of the methods is really able to improve performance. The reason for this is in the synthetic design of its graph and the features: The graph does not contain any information about the labels, as it only connects neighboring cells (it is solely a computational graph). The same is partially true for node features which indeed contain information about neighboring mines, but only for 50% of the nodes. This renders JDR unsuitable, which is also reflected in an interesting way in the experiments: We found

Table 7: Properties of the synthetic datasets generated from the cSBM with ϵ = 3.25. H(G) indicates the homophily measure.

| ϕ |       |    | 2 µ |     | λ  |   | H ( G ) |
|---|-------|----|-----|-----|----|---|---------|
| − | 1 0   | 0  | 0   | − 2 | 06 | 0 | 039     |
| − | 0 875 | 0  | 40  | − 2 | 02 | 0 | 049     |
| − | 0 75  | 1  | 56  | − 1 | 90 | 0 | 076     |
| − | 0 625 | 3  | 28  | − 1 | 71 | 0 | 119     |
| − | 0 5   | 5  | 31  | − 1 | 46 | 0 | 170     |
| − | 0 375 | 7  | 35  | − 1 | 15 | 0 | 241     |
| − | 0 25  | 9  | 07  | − 0 | 79 | 0 | 325     |
| − | 0 125 | 10 | 22  | − 0 | 40 | 0 | 408     |
| 0 | 0     | 10 | 63  | 0   | 0  | 0 | 496     |
| 0 | 125   | 10 | 22  | 0   | 40 | 0 | 583     |
| 0 | 25    | 9  | 07  | 0   | 79 | 0 | 671     |
| 0 | 375   | 7  | 35  | 1   | 15 | 0 | 751     |
| 0 | 5     | 5  | 31  | 1   | 46 | 0 | 837     |
| 0 | 625   | 3  | 28  | 1   | 71 | 0 | 879     |
| 0 | 75    | 1  | 56  | 1   | 90 | 0 | 925     |
| 0 | 875   | 0  | 40  | 2   | 02 | 0 | 955     |
| 1 | 0     | 0  | 0   | 2   | 06 | 0 | 963     |

Table 8: Comparison of DIGL, FoSR, BORF and JDR on real-world heterophilic datasets from [Platonov et al.](#page-14-6) [\(2023\)](#page-14-6): Mean accuracy (%) and ROC AUC for imbalanced Minesweeper, Tolokers and Questions± 95% confidence interval. Best average accuracy in bold. OOM indicates an out-ofmemory error.

| Method      |    |    |     |    |    |    |     |    |    |    |     |    |    |     | Tolokers |    |    |     |     |    |
|-------------|----|----|-----|----|----|----|-----|----|----|----|-----|----|----|-----|----------|----|----|-----|-----|----|
| GCN         | 78 | 64 | ± 0 | 42 | 46 | 19 | ± 0 | 58 | 90 | 08 | ± 0 | 31 | 84 | 61  | ± 0      | 59 | 75 | 31  | ± 0 | 81 |
| GCN+DIGL    | 75 | 32 | ± 0 | 61 | 45 | 92 | ± 0 | 41 | 88 | 16 | ± 0 | 57 | 81 | 62  | ± 0      | 59 | 73 | 35  | ± 0 | 64 |
| GCN+FoSR    | 78 | 58 | ± 0 | 43 | 46 | 30 | ± 0 | 44 | 90 | 07 | ± 0 | 51 | 84 | 50  | ± 0      | 47 | 75 | 51  | ± 0 | 73 |
| GCN+BORF    | 78 | 66 | ± 0 | 42 | 46 | 44 | ± 0 | 54 | 90 | 06 | ± 0 | 38 |    | OOM |          |    |    | OOM |     |    |
| GCN+JDR     | 78 | 86 | ± 0 | 48 | 46 | 47 | ± 0 | 67 | 90 | 01 | ± 0 | 32 | 84 | 73  | ± 0      | 45 | 77 | 52  | ± 0 | 63 |
| GPRGNN      | 71 | 46 | ± 0 | 29 | 45 | 84 | ± 0 | 21 | 87 | 80 | ± 0 | 51 | 72 | 01  | ± 0      | 65 | 65 | 30  | ± 1 | 01 |
| GPRGNN+DIGL | 71 | 59 | ± 0 | 37 | 46 | 43 | ± 0 | 34 | 87 | 96 | ± 0 | 50 | 73 | 09  | ± 1      | 16 | 69 | 98  | ± 0 | 49 |
| GPRGNN+FoSR | 71 | 44 | ± 0 | 30 | 45 | 94 | ± 0 | 36 | 87 | 83 | ± 0 | 58 | 72 | 72  | ± 0      | 69 | 65 | 45  | ± 0 | 68 |
| GPRGNN+BORF | 71 | 46 | ± 0 | 26 | 45 | 79 | ± 0 | 33 | 87 | 81 | ± 0 | 51 |    | OOM |          |    |    | OOM |     |    |
| GPRGNN+JDR  | 71 | 85 | ± 0 | 31 | 46 | 19 | ± 0 | 24 | 87 | 91 | ± 0 | 49 | 75 | 54  | ± 0      | 73 | 73 | 60  | ± 0 | 86 |

that the choice of hyperparameters has hardly any influence on performance. But also the results of the other rewiring methods indicate that they cannot be applied here. The graph is a standard gridgraph, which should not exhibit any interesting geometric properties and not contain any insights about the labels. Overall, this is a typical error case for any rewiring method. But it could also be discussed to what extent this dataset is an interesting *graph* dataset for node classification at all, since the connectivity does not contain any information about the labels.

#### A.5.2 HOMOPHILIC DATASETS IN THE DENSE SPLITTING

Table [9](#page-23-0) shows the results of DIGL, FoSR, BORF and JDR on real-world homophilic datasets in the *dense* splitting. The improvements of rewiring are smaller overall compared to the sparse splitting, but all four methods are able to improve it in most cases. With GCN as the downstream model, DIGL now performs best. JDR can still achieve the best result on two out of five data sets. When using GPRGNN as downstream model, JDR performs best on three out of five datasets. DIGL and FoSR are still able to achieve small performance improvements on most datasets and both rank first place on one dataset. BORF, on the other hand, is not able to improve the performance in most cases. This suggests that a more powerful GNN architecture benefits less from DIGL, FoSR or BORF, while

Table 9: Comparison of DIGL, BORF and JDR on real-world homophilic datasets using the *dense* splitting: Mean accuracy (%) ± 95% confidence interval. Best average accuracy in bold.

| Method      |    |    | Cora  |    |    |    | CiteSeer |    |    |    | PubMed |    |    |    |     |    |    |    | Photo |    |
|-------------|----|----|-------|----|----|----|----------|----|----|----|--------|----|----|----|-----|----|----|----|-------|----|
| GCN         | 88 | 14 | ± 0   | 27 | 79 | 02 | ± 0      | 25 | 86 | 14 | ± 0    | 10 | 89 | 03 | ± 0 | 12 | 94 | 07 | ± 0   | 10 |
| GCN+DIGL    | 88 | 74 | ± 0   | 28 | 79 | 13 | ± 0      | 27 | 87 | 81 | ± 0    | 09 | 90 | 34 | ± 0 | 12 | 94 | 87 | ± 0   | 10 |
| GCN+FoSR    | 88 | 09 | ± 0   | 28 | 79 | 23 | ± 0      | 25 | 86 | 14 | ± 0    | 10 | 88 | 98 | ± 0 | 12 | 94 | 04 | ± 0   | 09 |
| GCN+BORF    | 88 | 18 | ± 0   | 24 | 79 | 17 | ± 0      | 24 | 86 | 14 | ± 0    | 10 | 89 | 14 | ± 0 | 11 | 94 | 00 | ± 0   | 10 |
| GCN+JDR     | 88 | 76 | ± 0   | 25 | 80 | 25 | ± 0      | 27 | 86 | 20 | ± 0    | 10 | 88 | 93 | ± 0 | 13 | 94 | 20 | ± 0   | 08 |
| GPRGNN      | 88 | 57 | ± 0 0 | 25 | 79 | 42 | ± 0      | 30 | 89 | 16 | ± 0    | 15 | 88 | 95 | ± 0 | 18 | 94 | 49 | ± 0   | 11 |
| GPRGNN+DIGL | 88 | 49 | ± 0   | 24 | 79 | 62 | ± 0      | 29 | 88 | 89 | ± 0    | 16 | 90 | 15 | ± 0 | 14 | 94 | 27 | ± 0   | 10 |
| GPRGNN+FoSR | 88 | 37 | ± 0   | 25 | 79 | 75 | ± 0      | 31 | 89 | 28 | ± 0    | 17 | 88 | 85 | ± 0 | 19 | 94 | 50 | ± 0   | 10 |
| GPRGNN+BORF | 88 | 56 | ± 0   | 27 | 79 | 39 | ± 0      | 31 | 89 | 04 | ± 0    | 18 | 88 | 90 | ± 0 | 19 | 94 | 52 | ± 0   | 10 |
| GPRGNN+JDR  | 89 | 33 | ± 0   | 25 | 81 | 00 | ± 0      | 28 | 89 | 24 | ± 0    | 15 | 87 | 35 | ± 0 | 32 | 94 | 78 | ± 0   | 08 |

JDR can still improve it even further. The computer dataset is an exception for both downstream GNNs, JDR is not really able to improve the performance at all, while DIGL can clearly improve it.

#### A.5.3 MLP WITH JDR

We design JDR with the aim to denoise the (possibly) complementary information in graph and features. This is based on the claim that a GNN is the method of choice when both the graph and the features contain valuable information, as it can utilize both. The experiments on cSBM in the main text show the ability of JDR to transfer information between the two. This becomes visible especially in the corner cases, were either the graph or the features do not contain any information about the labels. In this case the only way to improve is by transferring label information from one source to the other. To investigate this further, we test an MLP with JDR and compare the results with only the MLP and the GNNs. Since JDR can transfer the information from the graph to the features, an MLP should be able to perform similar to the GNNs. Therefore, we tune the hyperparameters of JDR with the MLP downstream model on the synthetic cSBM datasets and the real-world datasets from the main text.

The results on the cSBM data are shown in Figure [6.](#page-24-0) They show that combining an MLP with JDR clearly outperforms GCN in the heterophilic regime and performs very similar to GPRGNN. If the GNNs are also combined with JDR, they generally again provide superior performance compared to MLP+JDR. In the weak graph regime they have a huge performance advantage of about 20%, since JDR cannot improve the feature quality for the MLP in these cases. In the heterophilic regime, MLP+JDR is still comparable to GCN+JDR, which is not the case for the homophilic regime. A similar behavior (without JDR) has been observed in the literature before, e.g. by [Ma et al.](#page-13-12) [\(2022\)](#page-13-12) and is related to the limited ability of GCN to deal with heterophilic graphs. GPRGNN, however, does not show this limitation and provides superior performance across all datasets. Notably, all these findings do directly translate to the real-world datasets. The results on the real-world data can be found in Table [10](#page-24-1) in the homophilic case and in Table [11](#page-24-2) for the heterophilic case. For the homophilic datasets, MLP+JDR shows clear performance increases but cannot beat any of the GNN or GNN+JDR baselines. On the heterophilic graph, the MLP already outperforms GCN on three out of five datasets and with JDR on all datasets. GCN+JDR regains superior performance on three of the five datasets, but only GPRGNN+JDR outperforms the MLP+JDR on all datasets. Overall, this supports the claim that a GNN like GPRGNN is the method of choice when both the graph and the features contain valuable information.

![](_page_24_Figure_1.jpeg)

Figure 6: Comparison of MLP and the GNNs on the cSBM datasets in the sparse splitting. Comparisons of MLP and MLP+JDR with the GNNs [\(a\)](#page-25-0) and GNN+JDR [\(b\)](#page-25-0). The error bars indicate the 95% confidence interval. The MLP outperforms the GNNs in the very weak graph regime. Combining it with JDR clearly beats GCN (especially in the heterophilic regime) and performs very similar to GPRGNN. If the GNNs are combined with JDR, they generally again provide superior performance compared to MLP+JDR.

Table 10: Results of MLP and JDR on real-world homophilic dataset using the sparse splitting: Mean accuracy (%) ± 95% confidence interval. Best average accuracy in bold.

| Method     |    |    | Cora |    |    |    |     |    |    |    |     |    |    |    |     |    |    |    | Photo |    |
|------------|----|----|------|----|----|----|-----|----|----|----|-----|----|----|----|-----|----|----|----|-------|----|
| MLP        | 50 | 79 | ± 0  | 73 | 50 | 29 | ± 0 | 48 | 79 | 73 | ± 0 | 13 | 73 | 17 | ± 0 | 31 | 80 | 88 | ± 0   | 33 |
| MLP+JDR    | 62 | 66 | ± 0  | 61 | 61 | 55 | ± 0 | 32 | 80 | 86 | ± 0 | 12 | 80 | 65 | ± 0 | 24 | 88 | 34 | ± 0   | 45 |
| GCN        | 77 | 26 | ± 0  | 35 | 67 | 16 | ± 0 | 37 | 84 | 22 | ± 0 | 09 | 84 | 42 | ± 0 | 31 | 91 | 33 | ± 0   | 29 |
| GCN+JDR    | 79 | 96 | ± 0  | 26 | 69 | 35 | ± 0 | 28 | 84 | 79 | ± 0 | 08 | 85 | 66 | ± 0 | 36 | 92 | 52 | ± 0   | 23 |
| GPRGNN     | 79 | 65 | ± 0  | 33 | 67 | 50 | ± 0 | 35 | 84 | 33 | ± 0 | 10 | 84 | 06 | ± 0 | 48 | 92 | 01 | ± 0   | 41 |
| GPRGNN+JDR | 80 | 77 | ± 0  | 29 | 69 | 17 | ± 0 | 30 | 85 | 05 | ± 0 | 08 | 84 | 77 | ± 0 | 35 | 92 | 68 | ± 0   | 25 |

Table 11: Results of MLP and JDR on real-world heterophilic dataset using the dense splitting: Mean accuracy (%) ± 95% confidence interval. Best average accuracy in bold.

| Method     |    |    |     |    |    |    | Squirrel |    |    |    | Actor |    |    |    | Texas |    |    |    | Cornell |    |
|------------|----|----|-----|----|----|----|----------|----|----|----|-------|----|----|----|-------|----|----|----|---------|----|
| MLP        | 49 | 07 | ± 0 | 57 | 28 | 19 | ± 0      | 40 | 38 | 54 | ± 0   | 30 | 91 | 16 | ± 0   | 79 | 88 | 19 | ± 0     | 74 |
| MLP+JDR    | 70 | 48 | ± 0 | 46 | 59 | 18 | ± 0      | 31 | 39 | 50 | ± 0   | 26 | 91 | 16 | ± 0   | 77 | 88 | 47 | ± 0     | 77 |
| GCN        | 67 | 65 | ± 0 | 42 | 57 | 94 | ± 0      | 31 | 34 | 00 | ± 0   | 31 | 75 | 62 | ± 1   | 12 | 64 | 68 | ± 1     | 25 |
| GCN+JDR    | 69 | 76 | ± 0 | 50 | 61 | 76 | ± 0      | 39 | 40 | 47 | ± 0   | 31 | 85 | 12 | ± 0   | 74 | 84 | 51 | ± 1     | 06 |
| GPRGNN     | 69 | 15 | ± 0 | 51 | 53 | 44 | ± 0      | 37 | 39 | 52 | ± 0   | 22 | 92 | 82 | ± 0   | 67 | 87 | 79 | ± 0     | 89 |
| GPRGNN+JDR | 71 | 00 | ± 0 | 50 | 60 | 62 | ± 0      | 38 | 41 | 89 | ± 0   | 24 | 93 | 85 | ± 0   | 54 | 89 | 45 | ± 0     | 84 |

![](_page_25_Figure_1.jpeg)

Figure 7: Comparison of DIGL [\(Gasteiger et al., 2019\)](#page-12-7) and JDR on the cSBM datasets in the sparse splitting. Results for [\(a\)](#page-25-0) GCN and [\(b\)](#page-25-0) GPRGNN as downstream models. The error bars indicate the 95% confidence interval. As expected, DIGL is not really able to improve the performance of the GNNs in the heterophilic regime. It achieves the greatest improvement in the weak-graph regime and for strongly homophilic graphs, especially using GCN as downstream model. Another interesting observation is that for GCN and ϕ < 0.25 the curve of MLP corresponds exactly to the one of GCN+DIGL. The reason for this is that the hyperparameters found for DIGL ensure that the graph is ignored (α = 1.0), which means that the GCN then collapses to a simple MLP. For the more powerfull GPRGNN, on the other hand, DIGL is generally hardly able to improve performance, while JDR clearly increases the performance across all ϕ.

#### A.5.4 COMBINING JDR AND DIGL

We compare our method to DIGL [\(Gasteiger et al., 2019\)](#page-12-7) in the main text. We use the personalized PageRank (PPR) diffusion kernel and the same top-64 values sparsening method as in JDR in all experiments. Figure [7](#page-25-0) shows the additional results for DIGL on the synthetic datasets from the cSBM. Table [12](#page-26-1) shows the results on the real-world homophilic datasets in the sparse splitting and Table [13](#page-26-2) on the heterophilic datasets in the dense splitting. Here, in addition to the individual results for JDR and DIGL, the results for a combination of the two methods are also shown. For this purpose, the graph was first denoised with JDR and then diffused with DIGL. To do this, we fixed the hyperparameters of JDR and then tuned the parameter α of DIGL. We think this is interesting as both methods enhance the graph in different ways and thus should be combinable. In principle, this should also be possible for a combination of JDR with BORF or FoSR, but so far we have not been able to get this to work.

Homophilic datasets. For the homophilic datasets, both DIGL and JDR can improve the results when GCN is used as a downstream model. Still, DIGL is outperformed by JDR on four out the five datasets. The two methods can be combined on three of the five data sets to achieve even better results. This gives empirical support for the assumption that the two methods use a distinct way of performing rewiring in this case and a combination therefore can further increase accuracy. The picture is somewhat different for GPRGNN as a downstream model. The improvements for DIGL are significantly smaller here, whereas JDR shows clear improvements across all datasets. This suggests that a more powerful GNN architecture benefits less from DIGL, while JDR can still improve it even further. A combination of the two methods does not lead to an increase in performance here. Although the performance is still significantly better compared to no rewiring or just DIGL, JDR alone usually performs better.

Heterophilic datasets. Since DIGL rewires the graph by adding edges between nodes with short diffusion distance, it is expected to perform poorly on the heterophilic datasets. The results using GCN show that this is only true for Chameleon and Squirrel, while for Actor, Texas and Cornell there are still considerable improvements. For the datasets Texas and Cornell, DIGL even achieve the best results. JDR, on the other hand, improves performance across datasets and GNNs. This is also in line with the finding on the cSBM in Figure [7a.](#page-25-0) However, we can also see that DIGL can not really improve performance of GPRGNN. JDR, on the other hand, can still achieve an improvement

Table 12: Comparison of DIGL and JDR on real-world homophilic dataset using the sparse splitting: Mean accuracy (%) ± 95% confidence interval. Best average accuracy in bold.

| Method          |    |    | Cora |    |    |    | CiteSeer |    |    |    | PubMed |    |    |    |     |    |    |    | Photo |    |
|-----------------|----|----|------|----|----|----|----------|----|----|----|--------|----|----|----|-----|----|----|----|-------|----|
| GCN             | 77 | 26 | ± 0  | 35 | 67 | 16 | ± 0      | 37 | 84 | 22 | ± 0    | 09 | 84 | 42 | ± 0 | 31 | 91 | 33 | ± 0   | 29 |
| GCN+DIGL        | 79 | 27 | ± 0  | 26 | 68 | 03 | ± 0      | 33 | 84 | 60 | ± 0    | 09 | 86 | 00 | ± 0 | 24 | 92 | 00 | ± 0   | 23 |
| GCN+JDR         | 79 | 96 | ± 0  | 26 | 69 | 35 | ± 0      | 28 | 84 | 79 | ± 0    | 08 | 85 | 66 | ± 0 | 36 | 92 | 52 | ± 0   | 23 |
| GCN+JDR+DIGL    | 80 | 48 | ± 0  | 26 | 69 | 19 | ± 0      | 29 | 84 | 83 | ± 0    | 10 | 84 | 78 | ± 0 | 34 | 92 | 69 | ± 0   | 22 |
| GPRGNN          | 79 | 65 | ± 0  | 33 | 67 | 50 | ± 0      | 35 | 84 | 33 | ± 0    | 10 | 84 | 06 | ± 0 | 48 | 92 | 01 | ± 0   | 41 |
| GPRGNN+DIGL     | 79 | 77 | ± 0  | 30 | 67 | 50 | ± 0      | 35 | 84 | 72 | ± 0    | 10 | 86 | 25 | ± 0 | 28 | 92 | 31 | ± 0   | 25 |
| GPRGNN+JDR      | 80 | 77 | ± 0  | 29 | 69 | 17 | ± 0      | 30 | 85 | 05 | ± 0    | 08 | 84 | 77 | ± 0 | 35 | 92 | 68 | ± 0   | 25 |
| GPRGNN+JDR+DIGL | 80 | 55 | ± 0  | 27 | 69 | 47 | ± 0      | 27 | 84 | 87 | ± 0    | 10 | 85 | 98 | ± 0 | 21 | 92 | 67 | ± 0   | 27 |

![](_page_26_Figure_3.jpeg)

Figure 8: Alignment of the leading eigenspaces according to [\(3\)](#page-3-2) for graphs from the cSBM with different ϕ. We compare JDR to DIGL and no rewiring.

across all datasets. A combination of DIGL and JDR is generally not particularly useful in this scenario, likely because DIGL has difficulties on the heterophilic datasets anyway.

Table 13: Comparison of DIGL and JDR on real-world heterophilic dataset using the dense splitting: Mean accuracy (%) ± 95% confidence interval. Best average accuracy in bold.

| Method          |    |    |     |    |    |    | Squirrel |    |    |    | Actor |    |    |    | Texas |    |    |    | Cornell |    |
|-----------------|----|----|-----|----|----|----|----------|----|----|----|-------|----|----|----|-------|----|----|----|---------|----|
| GCN             | 67 | 65 | ± 0 | 42 | 57 | 94 | ± 0      | 31 | 34 | 00 | ± 0   | 31 | 75 | 62 | ± 1   | 12 | 64 | 68 | ± 1     | 25 |
| GCN+DIGL        | 58 | 04 | ± 0 | 48 | 39 | 64 | ± 0      | 34 | 39 | 57 | ± 0   | 29 | 91 | 05 | ± 0   | 73 | 88 | 49 | ± 0     | 74 |
| GCN+JDR         | 69 | 76 | ± 0 | 50 | 61 | 76 | ± 0      | 39 | 40 | 47 | ± 0   | 31 | 85 | 12 | ± 0   | 74 | 84 | 51 | ± 1     | 06 |
| GCN+JDR+DIGL    | 66 | 06 | ± 0 | 43 | 36 | 62 | ± 0      | 29 | 40 | 30 | ± 0   | 27 | 88 | 90 | ± 0   | 73 | 88 | 06 | ± 0     | 77 |
| GPRGNN          | 69 | 15 | ± 0 | 51 | 53 | 44 | ± 0      | 37 | 39 | 52 | ± 0   | 22 | 92 | 82 | ± 0   | 67 | 87 | 79 | ± 0     | 89 |
| GPRGNN+DIGL     | 66 | 57 | ± 0 | 46 | 42 | 98 | ± 0      | 37 | 39 | 61 | ± 0   | 21 | 91 | 11 | ± 0   | 72 | 88 | 06 | ± 0     | 81 |
| GPRGNN+JDR      | 71 | 00 | ± 0 | 50 | 60 | 62 | ± 0      | 38 | 41 | 89 | ± 0   | 24 | 93 | 85 | ± 0   | 54 | 89 | 45 | ± 0     | 84 |
| GPRGNN+JDR+DIGL | 70 | 07 | ± 0 | 44 | 59 | 37 | ± 0      | 35 | 41 | 57 | ± 0   | 20 | 91 | 52 | ± 0   | 70 | 87 | 77 | ± 1     | 81 |

#### A.5.5 ALIGNMENT

Here, we give a more detailed view on how much JDR actually increases alignment on cSBM and real-world datasets compared to the baseline methods. For cSBM, we can see in Figure [8](#page-26-3) that DIGL only increases alignment in the homophilic regime. In the heterophilic regime it clearly decreases alignment. We expect this because it promotes connections among nodes at short diffusion distance. Also the random teleport probability found on these datasets is 1.0, which results in a random uniformly connected graph. Similar to the results on the real-world datasets Cornell and Texas from the main text, we can see this in the classification performance in Figure [7.](#page-25-0) In the heterophilic regime, the performance of DIGL matches exactly the MLP, while in the homophilic regime, we can see some performance increases. With GPRGNN, DIGL is not really able to improve performance at all (except for ϕ = 0). In Figure [9,](#page-27-1) we can see that JDR increases alignment across all settings and more strongly than the baseline methods on the real-world graphs (except for Citeseer).

![](_page_27_Figure_1.jpeg)

Figure 9: Alignment of the leading eigenspaces of graphs from homophilic [\(a\)](#page-27-1) and heterophilic [\(b\)](#page-27-1) real-world datasets. We compare the original graph (None) to the output of DIGL, BORF and JDR with the hyperparameters found on GCN. JDR increases the alignment in all settings and achieves the maximum alignment among rewiring algorithms in all settings except on the Citeseer dataset.

DIGL also increases alignment on many homophilic graphs and on heteophilic Texas. We would like to note that when only rewiring a graph, increasing the alignment might not always be the best thing to do: If the graph is very good, it would be the features that should be made more aligned to the graph. But only JDR is able to do this, as it also denoises the features as well. It is also interesting that DIGL decreases the alignment for Cora and PubMed, but still achieves a good performance. This indicates that DIGL in this case improves the graph in a different way than JDR. So here, it should be possible to combine both methods to achieve even better performance. And indeed, [A.5.4](#page-25-1) shows that this is exactly the case for Cora, PubMed and Photo on GCN. FoSR and BORF do not visibly change the alignment, since their modifications to the graph are usually too small.

#### A.5.6 SPECTRAL CLUSTERING

In addition to the GNNs as a downstream algorithm, we also experimente with spectral clustering (SC). Spectral clustering either works with an existing graph, or a k-nearest neighbor graph is created from given (high-dimensional) node features. Then the k largest eigenvectors of the graph are calculated (the first one is usually omitted as it is a constant vector) and their entries are then used as coordinates for a k-means clustering of the nodes into k classes. We show that JDR using the hyperparameters found with GCN as a downstream model, improves the performance of a spectral clustering algorithm acting directly on A or X. This indicates a close connection between GCNs and spectral clustering such that a good denoised graph for GCN is also a good graph for spectral clustering. Intuitively, since spectral clustering is related to the graph cut, this means that in this case the classes are connected with fewer edges, making them easier to cluster based on the cut.

Table 14: Results on real homophilic datasets using spectral clustering: Mean accuracy (%) and best result in bold. Here, all methods use the hyperparameters found using GCN as downstream algorithm.

| Method     |       | Cora |       | CiteSeer |    | Pubmed |    | Computers |    | Photo |
|------------|-------|------|-------|----------|----|--------|----|-----------|----|-------|
| SC(A)      | 33    | 83   | 24    | 16       | 58 | 94     | 37 | 35        | 30 | 58    |
| SC(A)+DIGL | 29    | 54   | 22    | 18       | 59 | 65     | 61 | 55        | 25 | 41    |
| SC(A)+FoSR | 33    | 83   | 24    | 61       | 58 | 72     | 36 | 97        | 30 | 54    |
| SC(A)+BORF | 35    | 01   | 25    | 22       | 58 | 90     | 37 | 35        | 33 | 37    |
| SC(A)+JDR  | 67.76 |      | 63.36 |          | 72 | 90     | 62 | 29        | 65 | 67    |
| SC(X)      | 29    | 76   | 45    | 57       | 60 | 45     | 28 | 53        | 48 | 46    |
| SC(X)+JDR  | 34.68 |      | 45.90 |          | 60 | 47     | 28 | 55        | 48 | 58    |

![](_page_28_Figure_1.jpeg)

Figure 10: Separate results for using spectral clustering on a rewriting only A [\(a\)](#page-28-0) and denoising only X [\(b\)](#page-28-0) compared to full JDR . Note that for ϕ ∈ {0.5, 0.625, 0.875} we had to use additional graphs generated using cSBM with an average node degree of d = 10 for spectral clustering of A to work in general and for ϕ = 0.875 also for JDR. The reason for this is that the graph is very sparse so it is not necessarily connected such that there is no guarantee that spectral clustering works. However a larger node degree does not improve the performance of spectral clustering in general, while it may for GNNs.

Table 15: Results on real heterophilic datasets using spectral clustering: Mean accuracy (%) and best result in bold. Here, all methods use the hyperparameters found using GCN as downstream algorithm.

|            |       | Chameleon |       | Squirrel |    | Actor |    | Texas |    | Cornell |
|------------|-------|-----------|-------|----------|----|-------|----|-------|----|---------|
| SC(A)      | 31    | 71        | 22    | 40       | 25 | 92    | 48 | 09    | 39 | 89      |
| SC(A)+DIGL | 32    | 06        | 22    | 69       | 25 | 91    | 43 | 72    | 40 | 44      |
| SC(A)+FoSR | 31    | 44        | 24    | 46       | 25 | 93    | 55 | 19    | 38 | 80      |
| SC(A)+BORF | 31    | 97        | OOM   |          | 25 | 97    | 56 | 83    | 43 | 17      |
| SC(A)+JDR  | 31    | 36        | 22    | 15       | 28 | 63    | 52 | 46    | 44 | 26      |
| SC(X)      | 23    | 54        | 20    | 17       | 31 | 01    | 49 | 18    | 45 | 36      |
| SC(X)+JDR  | 24.59 |           | 21.03 |          | 23 | 99    | 55 | 74    | 49 | 73      |

cSBM. Figure [10](#page-28-0) displays the results of spectral clustering with and w/o JDR. Figure [10a](#page-28-0) indicates the expected behavior that spectral clustering using A performs particularly poorly in the weak graph regime, since in this case there is hardly any information about the labels in A. By using JDR, this limitation is completely removed and the performance is close to AMP-BP across all ϕ. The rewired graph now contains more information about the labels, which was previously only available in X. For spectral clustering of X in Figure [10b,](#page-28-0) the relation is exactly the other way around. In the strong heterophilic or homophilic regime the performance is poor since most information is contained in the graph structure. Using JDR this limitation is removed and the performance becomes closer to AMP-BP across all ϕ. Although a slight denoising of X by A would be possible for ϕ = ±0.375, there is no performance advantage here and these settings now show the weakest performance across all ϕ.

Real-world Datasets. For the real world datasets, we compare the spectral clustering of A using the different rewiring methods DIGL, FoSR , BORF and JDR. For the spectral clustering of X we can only evaluate JDR. Again we use the hyperparameters found using GCN as downstream model. The results in Table [14](#page-27-2) on homophilic datasets show a significant benefit of using JDR across all datasets. FoSR, BORF and DIGL are also able to improve the performance in some settings but not very consistently. There are also performance improvements across all datasets for the spectral clustering of X with JDR, but these are significantly smaller. This indicates that the rewiring of the graph has a significantly greater influence on performance here than the denoising of the features. It also gives an indication of how JDR works on real-world data: It increases "spectral

![](_page_29_Figure_1.jpeg)

Figure 11: Separate results for rewiring only A and denoising only X compared to full JDR. Results for [\(a\)](#page-29-1) GCN and [\(b\)](#page-29-1) GPRGNN. The error bars indicate the 95% confidence interval.

![](_page_29_Figure_3.jpeg)

Figure 12: ESNR for graphs from the cSBM with different ϕ. We compare JDR to DIGL and no rewiring.

clusterability" of the graph. Table [15](#page-28-1) shows the results for the heterophlic datasets. The results here are much more inconsistent. It is striking that DIGL improves Chameleon and Squirrel, while it has actually worsened performance for GCN. BORF can improve the performance on Texas and Cornell by a large margin, although DIGL and JDR perform better with the GCN. FoSR improves the performance of Squirrel. For the results of JDR, it is worth looking at them together with the spectral clustering of X. On Chameleon and Squirrel the performance decreases for A but clearly increases for X. On Texas and Cornell it is improved in all cases, but on A not as strongly as for BORF. On Actor, the performance for X has dropped, while JDR is also the only method that really improves the result for A. To summarize, the improvements for JDR can be assigned to one of the two sources of information, either A or X, for each dataset.

# A.5.7 EVALUATING GRAPH DENOISING VIA ESNR

As described in the main text, the Graph Propensity Score (GPS) algorithm [\(Dong and Kluger,](#page-11-9) [2023\)](#page-11-9), together with the edge signal-to-noise ratio (ESNR) metric introduced in the same paper, are related to our work. Similar to the GPS algorithm, which uses the ESNR to denoise the graph, JDR denoises the graph but also the features. These two strategies are different, but they are both based on the cSBM. The main drawback of GPS is that it does not consider the possibility to also denoise the features using the graph and it is further limited to low-SNR graphs and to GCN as downstream model. We are interested in the the ESNR metric as it quantifies the noise in the graph.

In Figure [12,](#page-29-2) Table [16](#page-30-1) and Table [17](#page-30-2) we compare the achieved ESNR values of JDR with the baselines and the original graphs. The results for cSBM in Figure [12](#page-29-2) show that JDR is able to denoise the graph in all cases and DIGL in the strongly homophilic regime. However, comparing the ESNR curve to the actual GCN performance in Figure [7,](#page-25-0) there is no clear connection apart from the general trend. This was claimed to be more visible in the paper by [Dong and Kluger](#page-11-9) [\(2023\)](#page-11-9), especially for cSBM on which it is based on. The results on homophilic real-world datasets in Table [16](#page-30-1) follow

Table 16: ESNR results of the original and rewired homophilic real-world graphs using JDR and the baseline rewiring methods. Largest value in bold.

| Method |   | Cora |   | CiteSeer |   | PubMed |   | Computers |   | Photo |
|--------|---|------|---|----------|---|--------|---|-----------|---|-------|
| None   | 0 | 2964 | 0 | 1976     | 0 | 2701   | 0 | 6502      | 0 | 7145  |
| DIGL   | 0 | 6888 | 0 | 5989     | 0 | 5490   | 0 | 7368      | 0 | 7407  |
| FoSR   | 0 | 2965 | 0 | 1988     | 0 | 2700   | 0 | 6500      | 0 | 7145  |
| BORF   | 0 | 2958 | 0 | 1948     | 0 | 2700   | 0 | 6502      | 0 | 7144  |
| JDR    | 0 | 6160 | 0 | 5754     | 0 | 5291   | 0 | 7069      | 0 | 7155  |

Table 17: ESNR results of the original and rewired heterophilic real-world graphs using JDR and the baseline rewiring methods. Largest value in bold.

| Method |   | Chameleon |   | Squirrel |   | Actor |   | Texas |   | Cornell |
|--------|---|-----------|---|----------|---|-------|---|-------|---|---------|
| None   | 0 | 5199      | 0 | 4680     | 0 | 0546  | 0 | 0585  | 0 | 0388    |
| DIGL   | 0 | 5082      | 0 | 3884     | 0 | 0244  | 0 | 1442  | 0 | 1195    |
| FoSR   | 0 | 5199      | 0 | 4679     | 0 | 0546  | 0 | 0620  | 0 | 0389    |
| BORF   | 0 | 5199      |   |          | 0 | 0545  | 0 | 0593  | 0 | 0423    |
| JDR    | 0 | 4473      | 0 | 2156     | 0 | 1703  | 0 | 2729  | 0 | 2828    |

this trend. Indeed both JDR and DIGL are able to decrease the noise in the graph, but DIGL does so best on all five datasets, while the GCN results show a benefit of JDR over DIGL on four out of five datasets (see Table [2\)](#page-7-0). For the heterophilic real-world datasets the results are even more inconsistent. JDR is able to decrease the graph noise on three datasets, but two of them not being the ones where GCN performs best (see Table [3\)](#page-7-1). Also on Chameleon on Squirrel, no method is able to improve the ESNR. We suspect that this behavior occurs due to the role of features which is not captured by the ESNR. Checking the results in the paper by [Dong and Kluger](#page-11-9) [\(2023\)](#page-11-9), there are cases where the ESNR is not sensitive, e.g. on the Chameleon dataset, which suggests that in such cases, denoising the features is more beneficial than focusing only on the graph structure. In fact, this is in line with our findings in the ablations in Figure [13](#page-31-0) and Figure [14,](#page-31-1) where the denoising of the features improves more than the denoising of the graph on the Chameleon dataset. Overall, the ESNR is generally able to quantify the denoising of the graphs in most cases, but a more direct connection to GCN performance requires further research.

Finally, there are significant differences between the GPS approach by [Dong and Kluger](#page-11-9) [\(2023\)](#page-11-9) and our work, most importantly in that we consider both graph and feature information. Moreover, JDR improves performance of GNNs on real-world graph datasets where GPS does not provide any improvements (homophilic datasets). Ablations on these datasets (again Figure [13](#page-31-0) and Figure [14\)](#page-31-1) indicate that JDR still achieves the most improvement by denoising the graph rather than the features in a way that does not seem to be captured by the ESNR. We think that this is because ESNR does not consider the SNR of the features. In the experiments of [Dong and Kluger](#page-11-9) [\(2023\)](#page-11-9) with edge dropout on real-world data, the performance barely decreases for datasets like PubMed. The reason for this is that the features already contain a lot of information and making the graph noisier via edge dropout does not spoil this. This is also related to the discussion of GNN vs. MLP (see [A.5.3\)](#page-23-1), where an MLP on PubMed already performs reasonably well (see e.g. Table [10.](#page-24-1)

#### A.6 ABLATIONS

We perform several ablations of our method to investigate what happens in different scenarios and what effects changes in parameters have. First, we present our ablations of the JDR method. We show separate results for denoising only the graph JDR(A) or the features JDR(X) using the GNNs on the cSBM and real-world data. Also, we show several ablations of the hyperparameters of JDR. We therefore use a dataset created from cSBM, the homophilic dataset Cora and the heterophilic dataset Chameleon. Ablations on the real-world datasets are performed for all hyperparameters of JDR and show its robustness to change in these parameters.

![](_page_31_Figure_1.jpeg)

Figure 13: Average accuracy of GCN on all real-world datasets tested for denoising only the features JDR(X), rewiring only the graph JDR(A) and joint denoising and rewiring JDR.

![](_page_31_Figure_3.jpeg)

Figure 14: Average accuracy of GPRGNN on all real-world datasets tested for denoising only the features JDR(X), rewiring only the graph JDR(A) and joint denoising and rewiring JDR. It can be observed that for most datasets, the major improvement is achieved by JDR(A). Only for Squirrel and Chameleon it is JDR(X). In most cases using JDR on both X and A achieves the best performance.

JDR. The main motivation for these ablations is to show how much impact the denoising of A and X respectively have on the results for a dataset and how big the additional benefit is to do this jointly. Therefore we look at the results if we denoise only the graph JDR(A) or the features JDR(X). Doing this for the cSBM in Figure [11,](#page-29-1) we can observe the expected behavior, which is particularly pronounced for GCN in Figure [11a.](#page-29-1) In the weak graph regime, the performance increase results purely from denoising A, so JDR(A) achieves the same performance as JDR. The same holds for JDR(X) in the strong homophilic regime and for ϕ = −1.0. In the remaining intermediate regimes, we can often observe a performance benefit of both JDR(A) and JDR(X), which becomes much stronger when we combine both. This benefit of combining both is particularly pronounced for ϕ = −0.375, where JDR(X) alone even reduces performance, while JDR clearly improves performance. In Figure [11b,](#page-29-1) we can basically observe the same behavior, but less strongly pronounced. Moreover, it happens in several cases here, again especially in the intermediate regime, that the performance is reduced by JDR(X), but improved for the joint denoising.

Figure [13](#page-31-0) and Figure [14](#page-31-1) show the same investigation for the real-world datasets using GCN and GPRGNN, respectively. In most cases, the greater performance gain results from JDR(A) and the joint denoising performs best. Only for the datasets Chameleon for both GNNs and Squirrel for GPRGNN, the denoising of X has the greater influence. Also the case where the denoising of X reduces the performance, but a joint denoising performs best, occurs here, e.g. for Citeseer or

Table 18: Comparison for JDR using hyperparameters tuned on different downstream models. The "\*" indicates that the hyperparameters of JDR where tuned using the same GNN as downstream model, no symbol mean that the respective other GNN model was used. Results on real-world homophilic datasets using sparse splitting (2.5%/2.5%/95%): Mean accuracy (%) ± 95% confidence interval. Best average accuracy in bold.

| Method      |    |    | Cora |    |    |    | CiteSeer |    |    |    | PubMed |    |    |    |     |    |    |    | Photo |    | ↑ | Gain |
|-------------|----|----|------|----|----|----|----------|----|----|----|--------|----|----|----|-----|----|----|----|-------|----|---|------|
| GCN         | 77 | 26 | ± 0  | 35 | 67 | 16 | ± 0      | 37 | 84 | 22 | ± 0    | 09 | 84 | 42 | ± 0 | 31 | 91 | 33 | ± 0   | 29 |   |      |
| GCN+JDR*    | 79 | 96 | ± 0  | 26 | 69 | 35 | ± 0      | 28 | 84 | 79 | ± 0    | 08 | 85 | 66 | ± 0 | 36 | 92 | 52 | ± 0   | 23 | 1 | 59   |
| GCN+JDR     | 78 | 85 | ± 0  | 29 | 69 | 11 | ± 0      | 28 | 84 | 20 | ± 0    | 09 | 85 | 61 | ± 0 | 21 | 92 | 25 | ± 0   | 25 | 1 | 13   |
| GPRGNN      | 79 | 65 | ± 0  | 33 | 67 | 50 | ± 0      | 35 | 84 | 33 | ± 0    | 10 | 84 | 06 | ± 0 | 48 | 92 | 01 | ± 0   | 41 |   |      |
| GPRGNN+JDR  | 80 | 47 | ± 0  | 33 | 68 | 94 | ± 0      | 29 | 85 | 17 | ± 0    | 09 | 84 | 64 | ± 0 | 25 | 92 | 64 | ± 0   | 21 | 0 | 86   |
| GPRGNN+JDR* | 80 | 77 | ± 0  | 29 | 69 | 17 | ± 0      | 30 | 85 | 05 | ± 0    | 08 | 84 | 77 | ± 0 | 35 | 92 | 68 | ± 0   | 25 | 0 | 98   |

Cornell. Overall, this confirms that our method indeed performs *joint* denoising, especially when both graph and node contain relevant information both benefit from denoising.

Hyperparameters. In Table [18](#page-32-1) we show how the downstream GNN performs if JDR was tuned on a different downstream GNN. We use GCN and GPRGNN for this. The results show that the hyperparameters of JDR are quite robust to different GNN downstream models as it achieves similar gains using the respective other hyperparameters. Another way to show the robustness of JDR is to perform ablations of the actual hyperparameters. To do this, we first look at a data set generated from the cSBM and examine the influence of the number of denoising iterations K and the number of entries of the adjacency matrix to be retained Ak. Figure [15](#page-33-2) show the results of this study. As expected increase both results in better performance but will also increase the computational complexity. Based on this, we choose A<sup>k</sup> = 64 for all experiments as a good trade-off between computational and memory cost and accuracy over different numbers of denoising iterations. We also investigate this effect together with the rest of the hyperparameters for the real-world datasets Cora in Figure [16](#page-33-0) and Chameleon in Figure [17.](#page-33-1) We again examine the number of denoising iterations K and the number of entries of the adjacency matrix to be retained Ak. Additionally, we study the interpolation ratios η<sup>X</sup> and η<sup>A</sup> and the number of eigenvectors for the denoising L<sup>X</sup> and LA. Both are analyzed relative to the value found by random search and for both A and X at the same time. For the interpolation ratios η<sup>X</sup> and ηA, we show the influence of using only a reduced number of digits of the best found value (0 corresponds to no denoising) and for the number of eigenvectors L<sup>X</sup> and L<sup>A</sup> we test different offsets (0 corresponding to the best value found using random search). Overall, we con observe a strong robustness to changes in the hyperparameters. Only the number of denoising iterations K should not be too high for the heterophilic data set Chameleon.

#### A.7 HYPERPARAMETERS

In this section we list all the hyperparameters used for the experiments to ensure the reproducibility of the results. They are also included in the code. In all experiments we use the Adam optimizer and the standard early stopping after 200 epochs from [\(Chien et al., 2021\)](#page-11-3). Whenever we use a GCN, it uses two layers, a hidden dimension of 64 and dropout with 0.5. Whenever we use GPRGNN, we use a polynomial filter of order 10 (corresponding to 10 hops) and a hidden dimension of 64. For JDR, we always keep the 64 largest entries of the rewired adjacency matrix A˜ per node. We justify this choice by the ablation in Figure [15.](#page-33-2)

cSBM. For synthetic data from the cSBM, we generally follow the hyperparameters from [\(Chien](#page-11-3) [et al., 2021\)](#page-11-3). GCN uses a learning rate of 0.01 and weight decay with λ = 0.0005. GPRGNN also uses a λ = 0.0005 and both use ReLU non-linearity. On homophilic graphs (ϕ ≥ 0), GPRGNN uses a learning rate of 0.01, a weight initialization α = 0.1 and dropout with 0.5. For heterophilic graphs, it uses a learning rate of 0.05, α = 1.0 and dropout 0.7. The hyperparameters for JDR on the cSBM are shown in Table [22.](#page-35-0) We only tuned them using GCN as a downstream model, so for GPRGNN+JDR we use the same ones.

Real-world Datasets. For the real-world datasets, the remaining hyperparameters for GCN are displayed in Table [19](#page-34-0) and for GPRGNN in Table [21.](#page-34-1) The hyperparameters for JDR can be found

![](_page_33_Figure_1.jpeg)

Figure 15: Average accuracy of GCN on cSBM with ϕ = 0.0 for different numbers of denoise iterations and different numbers of entries A<sup>k</sup> to keep for each node in the rewired adjacency matrix. Error bars indicating the 95% confidence interval over 100 runs.

![](_page_33_Figure_3.jpeg)

Figure 16: Ablations of GCN+JDR on the homophilic dataset Cora compared to the result for GCN. The light shaded ares indicate the 95% confidence interval. Ablations on the number number of entries chosen per node for the adjacency Ak, the number of denoise iterations K, the number of interpolations digits for the η values and the number of eigenvectors L□ used. All other parameters are kept constant. In all cases we can see that JDR is quite robust to changes in all of its hyperparameters.

![](_page_33_Figure_5.jpeg)

Figure 17: Ablations of GCN+JDR on the heterophilic dataset Chameleon compared to the result for GCN. The light shaded ares indicate the 95% confidence interval. We perfrom the same ablations as for Cora. In all cases except the number of denoising iterations, we can see that JDR is quite robust to changes in all of its hyperparameters.

in Table [23](#page-35-1) and Table [24.](#page-36-0) For the rewiring method BORF, we list its hyperparameters in Table [25](#page-36-1) and Table [26.](#page-36-2) For DIGL, we always use the PPR kernel and sparsify the result by keeping the top-64 values for a weighted adjacency matrix. The values for the random-teleport probabililty α and number of iterations for FoSR are listed in Table [27](#page-37-0) and Table [28.](#page-37-1)

Table 19: Hyperparameters of GCN. All models use 2 layers, a hidden dimension of 64 and dropout with 0.5. Different type of weight decay and early stopping from [\(Gasteiger et al., 2019\)](#page-12-7) was used, if these provided a better performance then using the standard setting in [Chien et al.](#page-11-3) [\(2021\)](#page-11-3). The same holds for feature normalization, which was used by default in [Chien et al.](#page-11-3) [\(2021\)](#page-11-3) for GPRGNN.

| Dataset   |   | Lr | Normalize | X | λ 1  | λ 1 layer | Early stopping |
|-----------|---|----|-----------|---|------|-----------|----------------|
| Cora      | 0 | 01 | False     |   | 0 05 | First     | GPRGNN         |
| Citeseer  | 0 | 01 | True      | 0 | 0005 | All       | GPRGNN         |
| PubMed    | 0 | 01 | True      | 0 | 0005 | All       | GPRGNN         |
| Computers | 0 | 01 | False     | 0 | 0005 | All       | GPRGNN         |
| Photo     | 0 | 01 | False     | 0 | 0005 | All       | GPRGNN         |
| Chameleon | 0 | 05 | True      |   | 0 0  | All       | DIGL           |
| Squirrel  | 0 | 05 | True      |   | 0 0  | All       | DIGL           |
| Actor     | 0 | 01 | False     | 0 | 0005 | All       | DIGL           |
| Texas     | 0 | 05 | True      | 0 | 0005 | All       | GPRGNN         |
| Cornell   | 0 | 05 | True      | 0 | 0005 | All       | GPRGNN         |

Table 20: Hyperparameters of GCN for the larger heterophilic datasets. All models use a hidden dimension of 64, batch norm, no weight decay and the early stopping from [Chien et al.](#page-11-3) [\(2021\)](#page-11-3).

| Dataset       |   | Lr  | # layers |   | Dropout | Residuals |
|---------------|---|-----|----------|---|---------|-----------|
| Questions     | 0 | 005 | 5        | 0 | 2       | True      |
| Penn94        | 0 | 001 | 2        | 0 | 5       | False     |
| Twitch-gamers | 0 | 01  | 4        | 0 | 5       | False     |

Table 21: Hyperparameters of GPRGNN. All models use 10 hops and a hidden dimension of 64.

| Dataset       |   | Lr  | Normalize | X | α |   | λ 1  |   | Dropout | Early stopping |
|---------------|---|-----|-----------|---|---|---|------|---|---------|----------------|
| Cora          | 0 | 01  | True      | 0 | 1 | 0 | 0005 | 0 | 5       | GPRGNN         |
| Citeseer      | 0 | 01  | True      | 0 | 1 | 0 | 0005 | 0 | 5       | GPRGNN         |
| PubMed        | 0 | 05  | True      | 0 | 2 | 0 | 0005 | 0 | 5       | GPRGNN         |
| Computers     | 0 | 01  | False     | 0 | 1 | 0 | 0005 | 0 | 5       | GPRGNN         |
| Photo         | 0 | 01  | False     | 0 | 5 |   | 0 0  | 0 | 5       | GPRGNN         |
| Chameleon     | 0 | 05  | False     | 1 | 0 |   | 0 0  | 0 | 7       | DIGL           |
| Squirrel      | 0 | 05  | True      | 0 | 0 |   | 0 0  | 0 | 7       | GPRGNN         |
| Actor         | 0 | 01  | True      | 0 | 9 |   | 0 0  | 0 | 5       | GPRGNN         |
| Texas         | 0 | 05  | True      | 1 | 0 | 0 | 0005 | 0 | 5       | GPRGNN         |
| Cornell       | 0 | 05  | True      | 0 | 9 | 0 | 0005 | 0 | 5       | GPRGNN         |
| Questions     | 0 | 005 | False     | 1 | 0 |   | 0 0  | 0 | 2       | GPRGNN         |
| Penn94        | 0 | 01  | False     | 0 | 1 | 0 | 0001 | 0 | 2       | GPRGNN         |
| Twitch-gamers | 0 | 001 | False     | 0 | 5 | 0 | 0001 | 0 | 2       | GPRGNN         |

Table 22: Hyperparameters for GCN on the cSBM in the sparse splitting. For all homophilic datasets the eigenvalues are ordered by value and for all heterophilic datasets they are ordered by absolute value. In all setting we keep the 64 largest entries of the rewired adjacency matrix A˜ per node. Interpolation ratios η are rounded to three digits from the best values found by the random search.

| ϕ |       | K  | L A | L X |   | JDR η A |   | η X 1 |   | η X 2 |   | DIGL α |
|---|-------|----|-----|-----|---|---------|---|-------|---|-------|---|--------|
| − | 1 0   | 28 | −   | 10  |   | −       | 0 | 482   | 0 | 916   | 1 | 0      |
| − | 0 875 | 41 | 5   | 8   | 0 | 101     | 0 | 479   | 0 | 858   | 1 | 0      |
| − | 0 75  | 40 | 6   | 9   | 0 | 042     | 0 | 498   | 0 | 846   | 1 | 0      |
| − | 0 625 | 48 | 6   | 8   | 0 | 036     | 0 | 453   | 0 | 862   | 1 | 0      |
| − | 0 5   | 50 | 9   | 10  | 0 | 189     | 0 | 412   | 0 | 991   | 1 | 0      |
| − | 0 375 | 48 | 8   | 10  | 0 | 879     | 0 | 973   | 0 | 773   | 1 | 0      |
| − | 0 25  | 80 | 1   | 1   | 1 | 000     |   | −     |   | −     | 1 | 0      |
| − | 0 125 | 80 | 1   | 1   | 1 | 000     |   | −     |   | −     | 1 | 0      |
| 0 | 0     | 80 | 1   | 1   | 1 | 000     |   | −     |   | −     | 0 | 95     |
| 0 | 125   | 76 | 1   | −   | 0 | 650     |   | −     |   | −     | 1 | 0      |
| 0 | 25    | 33 | 1   | −   | 0 | 951     |   | −     |   | −     | 0 | 5      |
| 0 | 375   | 18 | 10  | 10  | 0 | 856     | 0 | 023   | 0 | 228   | 0 | 05     |
| 0 | 5     | 18 | 10  | 9   | 0 | 415     | 0 | 263   | 0 | 880   | 0 | 05     |
| 0 | 625   | 22 | 8   | 7   | 0 | 264     | 0 | 340   | 0 | 807   | 0 | 05     |
| 0 | 75    | 15 | 7   | 9   | 0 | 056     | 0 | 474   | 0 | 778   | 0 | 05     |
| 0 | 875   | 16 | 10  | 8   | 0 | 035     | 0 | 228   | 0 | 981   | 0 | 05     |
| 1 | 0     | 80 | −   | 1   |   | −       | 1 | 000   | 1 | 000   | 0 | 05     |

Table 23: Hyperparameters of JDR for all real-world datasets in the dense splitting. Following the findings from cSBM for all homophilic datasets the eigenvalues are ordered by value and for all heterophilic datasets they are ordered by absolute value. In all setting we keep the 64 largest entries of the rewired adjacency matrix A˜ per node. Interpolation ratios η are rounded to three digits from the best values found by the random search.

| Dataset       | K  | L A  | L X  | GNN | η A |   | η X 1 |   | η X 2 | K  | L A  | L X  | GPRGNN | η A |   | η X 1 |   | η X 2 |
|---------------|----|------|------|-----|-----|---|-------|---|-------|----|------|------|--------|-----|---|-------|---|-------|
| Cora          | 10 | 1853 | 38   | 0   | 066 | 0 | 173   | 0 | 071   | 10 | 772  | 76   | 0      | 027 | 0 | 434   | 0 | 005   |
| Citeseer      | 15 | 578  | 1330 | 0   | 460 | 0 | 173   | 0 | 049   | 4  | 1390 | 1169 | 0      | 345 | 0 | 099   | 0 | 585   |
| PubMed        | 12 | 8    | 53   | 0   | 316 | 0 | 004   | 0 | 187   | 1  | 1772 | 919  | 0      | 197 | 0 | 893   | 0 | 034   |
| Computers     | 3  | 718  | 975  | 0   | 398 | 0 | 021   | 0 | 068   | 7  | 583  | 1533 | 0      | 468 | 0 | 062   | 0 | 127   |
| Photo         | 6  | 467  | 1867 | 0   | 479 | 0 | 071   | 0 | 344   | 4  | 433  | 1719 | 0      | 413 | 0 | 115   | 0 | 231   |
| Chameleon     | 7  | 41   | 1099 | 0   | 066 | 0 | 375   | 0 | 975   | 3  | 31   | 1331 | 0      | 063 | 0 | 486   | 0 | 755   |
| Squirrel      | 2  | 4    | 1941 | 0   | 404 | 0 | 011   | 0 | 022   | 2  | 53   | 1210 | 0      | 234 | 0 | 495   | 0 | 964   |
| Actor         | 29 | 896  | 14   | 0   | 298 | 0 | 235   | 0 | 219   | 11 | 1171 | 791  | 0      | 476 | 0 | 028   | 0 | 251   |
| Texas         | 20 | 21   | 183  | 0   | 514 | 0 | 028   | 0 | 836   | 1  | 109  | 36   | 0      | 182 | 0 | 004   | 0 | 214   |
| Cornell       | 17 | 10   | 125  | 0   | 794 | 0 | 298   | 0 | 113   | 1  | 39   | 67   | 0      | 482 | 0 | 424   | 0 | 068   |
| Questions     | 8  | 248  | 284  | 0   | 218 | 0 | 199   | 0 | 841   | 2  | 89   | 2    | 0      | 974 | 0 | 106   | 0 | 311   |
| Penn94        | 20 | 60   | 71   | 0   | 445 | 0 | 005   | 0 | 902   | 5  | 172  | 1851 | 0      | 422 | 0 | 094   | 0 | 138   |
| Twitch-gamers | 5  | 7    | 5    | 0   | 235 | 0 | 286   | 0 | 806   | 5  | 2    | 2    | 0      | 165 | 0 | 329   | 0 | 003   |

Table 24: Hyperparameters of JDR for all the homophilic datasets in the sparse splitting. Following the findings from cSBM for all homophilic datasets the eigenvalues are ordered by value and for all heterophilic datasets they are ordered by absolute value. In all setting we keep the 64 largest entries of the rewired adjacency matrix A˜ per node. Interpolation ratios η are rounded to three digits from the best values found by the random search.

| Dataset   | K  | L A  | L X  | GNN | η A |   | η X 1 |   | η X 2 | K  | L A  | L X  | GPRGNN | η A |   | η X 1 |   | η X 2 |
|-----------|----|------|------|-----|-----|---|-------|---|-------|----|------|------|--------|-----|---|-------|---|-------|
| Cora      | 10 | 1853 | 38   | 0   | 066 | 0 | 173   | 0 | 071   | 10 | 772  | 76   | 0      | 027 | 0 | 434   | 0 | 005   |
| Citeseer  | 15 | 578  | 1330 | 0   | 460 | 0 | 173   | 0 | 049   | 4  | 1390 | 1169 | 0      | 345 | 0 | 099   | 0 | 585   |
| PubMed    | 12 | 8    | 53   | 0   | 316 | 0 | 004   | 0 | 187   | 1  | 1772 | 919  | 0      | 197 | 0 | 893   | 0 | 034   |
| Computers | 3  | 718  | 975  | 0   | 398 | 0 | 021   | 0 | 068   | 7  | 583  | 1533 | 0      | 468 | 0 | 062   | 0 | 127   |
| Photo     | 6  | 467  | 1867 | 0   | 479 | 0 | 071   | 0 | 344   | 4  | 433  | 1719 | 0      | 413 | 0 | 115   | 0 | 231   |

Table 25: Hyperparameters for BORF for all real-world datasets in the dense splitting. OOM indicates an out-of-memory error.

| Dataset   | # iterations | GNN # added | # removed | # iterations | GPRGNN # added | # removed |
|-----------|--------------|-------------|-----------|--------------|----------------|-----------|
| Cora      | 2            | 30          | 10        | 1            | 10             | 40        |
| Citeseer  | 3            | 30          | 40        | 3            | 10             | 50        |
| PubMed    | 2            | 0           | 30        | 3            | 20             | 40        |
| Computers | 1            | 20          | 40        | 3            | 20             | 30        |
| Photo     | 2            | 40          | 20        | 3            | 50             | 50        |
| Chameleon | 2            | 50          | 30        | 1            | 10             | 30        |
| Squirrel  |              | OOM         |           |              | OOM            |           |
| Actor     | 2            | 40          | 50        | 2            | 10             | 50        |
| Texas     | 1            | 40          | 10        | 2            | 40             | 50        |
| Cornell   | 1            | 20          | 50        | 1            | 20             | 50        |

Table 26: Hyperparameters for BORF for the homophilic real-world datasets in the sparse splitting.

| Dataset   | # iterations | GNN # added | # removed | # iterations | GPRGNN # added | # removed |
|-----------|--------------|-------------|-----------|--------------|----------------|-----------|
| Cora      | 2            | 10          | 40        | 2            | 30             | 50        |
| Citeseer  | 3            | 50          | 40        | 1            | 20             | 50        |
| PubMed    | 2            | 0           | 30        | 3            | 20             | 40        |
| Computers | 1            | 20          | 40        | 3            | 20             | 30        |
| Photo     | 3            | 0           | 50        | 3            | 10             | 20        |

Table 28: Values of the hyperparameter α of DIGL and the number of iterations (# iter) of FoSR for the homophilic real-world datasets in the sparse splitting.

| Dataset   |   | α GCN | DIGL | GPRGNN |   | α GCN | DIGL+JDR | GPRGNN | #iter GCN | FoSR GPRGNN |
|-----------|---|-------|------|--------|---|-------|----------|--------|-----------|-------------|
| Cora      | 0 | 10    | 0    | 30     | 0 | 10    | 0        | 30     | 5         | 75          |
| Citeseer  | 0 | 30    | 0    | 45     | 0 | 20    | 0        | 45     | 500       | 600         |
| PubMed    | 0 | 35    | 0    | 60     | 0 | 40    | 0        | 60     | 50        | 75          |
| Computers | 0 | 05    | 0    | 65     | 0 | 15    | 0        | 30     | 250       | 800         |
| Photo     | 0 | 20    | 0    | 50     | 0 | 10    | 0        | 50     | 5         | 500         |

Table 27: Values of the hyperparameter α of DIGL and the number of iterations (# iter) of FoSR for the real-world datasets in the dense splitting.

| Dataset       |   | α GCN | DIGL | GPRGNN |   | α GCN | DIGL+JDR | GPRGNN | #iter GCN | FoSR GPRGNN |
|---------------|---|-------|------|--------|---|-------|----------|--------|-----------|-------------|
| Cora          | 0 | 25    | 0    | 60     | 0 | 20    | 0        | 60     | 150       | 5           |
| Citeseer      | 0 | 60    | 0    | 50     | 0 | 25    | 0        | 25     | 1000      | 1000        |
| PubMed        | 0 | 60    | 0    | 50     | 0 | 60    | 0        | 65     | 10        | 5           |
| Computers     | 0 | 05    | 0    | 60     | 0 | 10    | 0        | 65     | 500       | 600         |
| Photo         | 0 | 30    | 0    | 70     | 0 | 20    | 0        | 75     | 25        | 250         |
| Chameleon     | 0 | 15    | 0    | 50     | 0 | 55    | 0        | 40     | 10        | 5           |
| Squirrel      | 0 | 05    | 0    | 15     | 0 | 10    | 0        | 20     | 5         | 10          |
| Actor         | 1 | 00    | 0    | 60     | 0 | 20    | 0        | 05     | 10        | 150         |
| Texas         | 1 | 00    | 0    | 00     | 0 | 20    | 0        | 20     | 50        | 75          |
| Cornell       | 1 | 00    | 1    | 00     | 0 | 95    | 0        | 00     | 25        | 5           |
| Questions     | 0 | 05    | 1    | 0      |   | −     |          | −      | 700       | 700         |
| Penn94        | 0 | 2     | 0    | 1      |   | −     |          | −      | 200       | 150         |
| Twitch-gamers | 0 | 15    | 0    | 25     |   | −     |          | −      | 5         | 100         |

### A.8 HARDWARE SPECIFICATIONS

Experiments on cSBM, Cora, Citeseer and Photo were conducted on an internal cluster with Nvidia Tesla V100 GPUs with 32GB of VRAM. The experiments on the remaining datasets (PubMed, Computers, Chameleon, Squirrel, Actor, Cornell and Texas) were performed using Nvidia A100 GPUs with 40GB or 80GB of VRAM. The larger VRAM is only necessary for GNN+JDR on PubMed and the larger heterophilic datasets from [Lim et al.](#page-13-8) [\(2021\)](#page-13-8); [Platonov et al.](#page-14-6) [\(2023\)](#page-14-6), because they have larger numbers of nodes in the graph (and we choose the top-64 edges per node after rewiring). Note that this could be reduced by sacrificing only a little bit of performance as shown in [A.6.](#page-30-0) One experiment of training and testing on 100 random splits typically takes about 5 min. For the standard benchmark graphs, the longest experiments with GPRGNN+JDR and a different early stopping condition take about 40 min. The experiments on the large Twitch-gamers dataset take around 60 min (for 10 splits), but similar to DIGL they require a lot of standard memory (around 500GB) while performing the decompositions.

# A.9 INSIGHTS FROM RANDOM MATRIX THEORY FOR ONE-LAYER GCNS

Following the derivation from [Shi et al.](#page-15-6) [\(2024\)](#page-15-6), we show empirically how denoising can reduce the empirical risk for a one-layer GCN without non-linearity. When the number of nodes N goes to infinity and the average node degree satisfies some assumptions, we can apply the Gaussian adjacency equivalence conjecture. This allows us to replace the binary adjacency in the cSBM with a spiked non-symmetric Gaussian random matrix without changing the training and test loss in the

![](_page_38_Figure_1.jpeg)

Figure 18: Experimental results on a non-symmetric gaussian cSBM with N = 1000 and γ = 2 with denoising of A. We plot the MSE for different µ 2 and λ = 0.0 in [18a](#page-38-0) and λ = 1.0 in [18b.](#page-38-0) Each data point is averaged over 10 independent trials and the standard deviation is indicated by the light shaded area.

limit. The equivalent adjacency reads

$$A = \frac{\lambda}{N} y y^T + \Xi_{gn} \quad (7)$$

where with Ξgn has i.i.d. centered normal entries with variance 1/N. Similarly, we build the features matrix as

$$X = \frac{\mu}{N}yu^T + \Xi_x. \quad (8)$$

Compared to the standard cSBM formulation we rescale the variables √µγ <sup>→</sup> <sup>µ</sup> and √ Fu → u. Additionally, we define α = 1/γ = F/N and for simplicity, we consider the case Itrain = I. The mean squared error (MSE) loss reads

$$L(\boldsymbol{\omega}) = \frac{1}{N} \|\mathbf{A} \mathbf{X} \boldsymbol{\omega} - \mathbf{y}\|_F^2 + \frac{r}{N} \|\boldsymbol{\omega}\|^2, \quad (9)$$

where r is the parameter for the ridge part, ω are the weights of the GCN and ∥∥<sup>F</sup> indicates the Frobenius norm. For N → ∞, the MSE concentrates, which means it is only a function of µ, λ and α. For denoising A we do

$$\mathbf{A}_{\text{den}} = \mathbf{A} + \eta_A \mathbf{X} \mathbf{X}^T. \quad (10)$$

The idea is that although this leads to more noise terms, the signal strength of yy<sup>T</sup> is increased more. Instead of a weighting of <sup>λ</sup> N yy<sup>T</sup> , we now have ( λ <sup>N</sup> +η<sup>A</sup> µ <sup>2</sup>F N )yy<sup>T</sup> . The new MSE also concentrates on a value determined by ηA. So, numerically, as shown in Figure [18,](#page-38-0) for any µ, |λ| > 0 we can always find values of η<sup>A</sup> such that the MSE is decreased. For denoising X we do

$$\mathbf{X}_{\text{den}} = \mathbf{X} + \eta_X \mathbf{A} \mathbf{X} \quad (11)$$

and show in Figure [19](#page-39-0) with the same argumentation as for A that an η<sup>X</sup> exists so that the MSE is reduced. Proof of both cases has yet to be provided and will be the subject of future work.

![](_page_39_Figure_1.jpeg)

(a) µ = 0

(b) µ = 1.0

Figure 19: Experimental results on a non-symmetric gaussian cSBM with N = 1000 and γ = 2 with denoising of X. We plot the MSE for different λ and µ = 0.0 in [19a](#page-39-0) and µ = 1.0 in [19b.](#page-39-0) Each data point is averaged over 10 independent trials and the standard deviation is indicated by the light shaded area.