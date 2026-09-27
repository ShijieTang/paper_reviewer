# Joint Graph Rewiring And Feature Denoising Via Spectral Resonance

Jonas Linkerhägner∗ Cheng Shi∗**Ivan Dokmanic´**
Department of Mathematics and Computer Science University of Basel firstname.lastname@unibas.ch

## Abstract

When learning from graph data, the graph and the node features both give noisy information about the node labels. In this paper we propose an algorithm to jointly denoise the features and rewire the graph (JDR), which improves the performance of downstream node classification graph neural nets (GNNs). JDR works by aligning the leading spectral spaces of graph and feature matrices. It approximately solves the associated non-convex optimization problem in a way that handles graphs with multiple classes and different levels of homophily or heterophily. We theoretically justify JDR in a stylized setting and show that it consistently outperforms existing rewiring methods on a wide range of synthetic and real-world node classification tasks.

## 1 Introduction

Graph neural networks (GNNs) are a powerful deep learning tool for graph-structured data, with applications in physics (Mandal et al., 2022; Linkerhägner et al., 2023), chemistry (Gilmer et al., 2017), biology (Gligorijevic et al., 2021) and beyond (Zhou et al., 2020). Typical tasks across ´ disciplines include graph classification (Duvenaud et al., 2015; Xu et al., 2019), node classification (Kipf and Welling, 2017; Li et al., 2019) and link prediction (Pan et al., 2022). Graph datasets contain two distinct types of information: the graph structure and the node features. The graph encodes interactions between entities and thus the classes or communities they belong to, similarly to the features. Recent work demonstrates that *rewiring* the graph by judiciously adding and removing edges may improve downstream GNN performance. That work argues that in a GNN, the graph serves not only to encode interactions but also to organize message passing computations (Battaglia et al., 2018). Even when it correctly encodes interactions it may not be an effective computational graph—rewiring it may then facilitate information flow.

Graph rewiring methods can be categorized into preprocessing and end-to-end. Preprocessing methods rewire the graph by relating its geometric and spectral properties to information flow (Topping et al., 2022; Nguyen et al., 2023; Karhadkar et al., 2023). End-to-end methods (Giraldo et al., 2023; Gutteridge et al., 2023; Qian et al., 2024) dynamically rewire the graph during training, leveraging both the graph and the node features. Unlike preprocessing methods, they do not output an improved graph which restricts their interpretability and reusability. Our focus is on the preprocessing methods. There are thus two mechanisms that hurt performance of GNNs: (1) real-world graphs and features are noisy (the graph has spurious and missing links), and (2) geometric properties of the graph impede message passing. In this paper we focus on (1) and ask a natural question: can simple, joint feature and graph denoising improve performance of a downstream GNN? We propose a new rewiring scheme that also uses node features to produce an enhanced graph. We leverage the fact that both the graph and the features are correlated with the labels. This is explicit in high-quality stylized models of graphs with features, including community models such as the contextual stochastic block model (cSBM) (Deshpande et al., 2018) and neighborhood graphs on
∗These authors contributed equally.

Noisy Graph Noisy Features Rewired Graph Denoised Features UPDATE UPDATE
DENOISE
REWIRE
Removed Added Class 1/2
/
Figure 1: Schematic overview of joint denoising and rewiring (JDR). In this example, we consider a noisy graph as it occurs in many different real-world scenarios, in the sense that it contains edges between and within classes and its node features are not fully aligned with the labels. The graph's adjacency matrix A and binary node features X are decomposed via spectral decomposition and singular value decomposition (SVD). The rewiring of A is performed by combining the information of its own eigenvectors V and the singular vectors U from X. The same applies vice versa for denoising, and both are performed iteratively K times. We synthesize the rewired graph A˜ and the denoised features X˜ by multiplying back with the final V(K) and U(K). To get specific properties like sparsity or binarity we can perform an UPDATE step, e.g. by thresholding (as done here). The resulting denoised and rewired graph is displayed on the right. Its structure now better represents the communities and the first entry of the features indicates the class assignment. points from low-dimensional manifolds. This fact motivates various spectral clustering and nonlinear dimensionality reduction methods (Shi and Malik, 2000; Ng et al., 2001). In the cSBM, seminal theoretical work shows that jointly leveraging the graph (stochastic block model (SBM)) and the features (a Gaussian mixture model (GMM)) improves over unsupervised clustering using either piece of information alone. However, the associated efficient inference algorithms based on belief propagation (Deshpande et al., 2018; Duranthon and Zdeborová, 2023) rely on perfect knowledge of the distribution of the cSBM and cannot be applied to arbitrary real-world data. Our contributions are as follows:
1. We take inspiration from work on the cSBM to design a practical algorithm for joint graph rewiring and feature denoising, which can improve the node classification performance of any downstream GNN on real-world data. We achieve this by adapting the graph and the features so as to maximize alignment between their leading eigenspaces. If these spaces are well-aligned we say that the graph and the features are *in resonance*.

2. We design an alternating optimization algorithm, joint denoising and rewiring (JDR), which approximates alignment maximization on spectrally-complex real-world graph data with multiple classes, possibly homophilic or heterophilic. We prove that JDR improves alignment between the graph and the features, but also with the labels, on a stylized generative model with noise from the Gaussian orthogonal ensemble (GOE); a recent conjecture in the literature suggests that this generalizes to cSBM.

3. We run extensive experiments to show that JDR outperforms existing preprocessing rewiring strategies while being guided solely by denoising.

This last point suggests that although there exist graphs with topological and geometrical characteristics which make existing rewiring schemes beneficial, a greater issue in real-world graphs is noise in the sense of missing and spurious links. This is true even when the graphs correctly reflect the ground truth information. In a citation network, for example, citations that *should* exist may be missing because of incomplete scholarship. Conversely, citations that *should not* exist may be present because the authors engaged in bibliographic ornamentation. Our method is outlined in Figure 1 and the code repository is available online1.

## 2 Joint Denoising And Rewiring 2.1 Preliminaries

We let G = (V, E) be an undirected graph with |V| = N nodes and an adjacency matrix A. To each node we associate an F-dimensional feature vector and collect these vectors in the rows of matrix X ∈ R
N×F . We make extensive use of the graph and feature spectra, namely the eigendecomposition A = V ΛV
Tand the SVD X = UΣWT, with eigen- and singular values ordered from largest to smallest. (As discussed below, in heterophilic graphs we order the eigenvalues of A according to their absolute value.) The graph Laplacian is L = D − A, where D is the diagonal node degree matrix. For k > 2 node classes, we use one-hot labels y ∈ {0, 1}
N×k. We write [L] for the set
{1, 2*, . . . , L*}. In the balanced two-class case, we consider nodes to be ordered so that the first half has label yi = −1 and the second half yi = 1. In semi-supervised node classification, the task is to label the nodes based on the graph (A and X) and a subset of the labels y. *Homophilic* graphs are those where nodes are more likely to connect with nodes with similar features or labels (e.g., friendship networks (McPherson et al., 2001)); *heterophilic* graphs are those where nodes more likely to connect with dissimilar nodes (e.g., protein interaction networks (Zhu et al., 2020)).

## 2.2 Motivation Via The Contextual Stochastic Block Model

For simplicity, we first explain our method for k = 2 classes and graphs generated from the cSBM. We then extend it to real-world graphs with multiple classes and describe the full practical algorithm. Contextual Stochastic Block Model. CSBMs (Deshpande et al., 2018) extend SBMs (Abbe, 2018), a community graph model, by high-dimensional node features. They have become a key generative model for studying GNNs (Baranwal et al., 2021; Wu et al., 2023; Kothapalli et al., 2023); for further pointers see Appendix A.2. We use cSBMs to build intuition about the graph rewiring and denoising problem. In a balanced 2-class SBM, the nodes are divided into two equal-sized communities with node labels yi *∈ {±*1}. Pairs of nodes connect independently at random, with probability cin/N inside communities and cout/N across communities. In the sparse regime (Abbe, 2018), with average node degree d = O(1), it is common to parameterize probabilities as cin = d + λ
√d and cout = d − λ
√d, where |λ| can be seen as the signal-to-noise ratio (SNR) of the graph. The signal Xi ∈ R
F at node i comes from a GMM,

$${\mathbf{X}}_{i}={\sqrt{\frac{\mu}{N}}}{\mathbf{y}}_{i}{\mathbf{\xi}}+{\frac{z_{i}}{\sqrt{F}}},$$
$$(1)$$

, (1)
where ξ ∼ N (0, IF /F) is the randomly drawn mean and zi ∼ N (0, IF ) is i.i.d. Gaussian standard noise. We set γ =
N F
and, following Chien et al. (2021), parameterize the graphs generated from the cSBM using ϕ =
2 π arctan(λ
√γ/µ). For ϕ → 1 we get homophilic behavior; for ϕ → −1 we get heterophilic behavior. Close to either extreme the node features contain little information. For ϕ → 0 the graph is Erdos–Rényi and only the features contain information. ˝ Denoising and Rewiring the cSBM. In the cSBM, A and X offer different noisy views on the labels. One can show that up to a scaling and a shift, the adjacency matrix is approximately ±yyT + ZER, which means that it is approximately a rank-one matrix with labels in the range, corrupted with "Erdos–Rényi-like noise" ˝ ZER (Erdös and Rényi, 1959). Another way to see this is to note that EA =1 2N
(cin + cout)11T +1 2N
(cin − cout)yyT(from the definition of the SBM). Since A is close to EA at high SNR, the eigenvectors contain information about the labels. It similarly follows directly from the definition that the feature matrix X is (up to a scaling) yuT + ZG where ZG is white Gaussian noise. It thus makes sense to use the information from X to enhance A and vice versa. Deshpande et al. (2018) show that analyzing the following optimization problem:

$$\operatorname*{maximize}_{\mathbf{v}\in\mathbb{R}^{N},\mathbf{u}\in\mathbb{R}^{F}}\langle\mathbf{v},\mathbf{A}\mathbf{v}\rangle+b\langle\mathbf{v},\mathbf{X}\mathbf{u}\rangle$$ $$\operatorname*{subject~to}||\mathbf{v}||_{2}=||\mathbf{u}||_{2}=1,\langle\mathbf{v},\mathbf{1}\rangle\leq\delta$$
$$\left(2\right)$$

for some carefully chosen value of b allows one to characterize detection bounds in unsupervised community detection with k = 2. It is clear from the above reasoning that in the high-SNR regime (λ and µ far away from the detection threshold), the second leading eigenvector of A and the leading

Feature Noise Amplitude Graph Noise 1 Gr ap h Noi se Amp litude Amp litude Graph Noise Feature Noise Feature Noise Fe ature No ise Graph Noise
-1
(a)
(b)
left singular vector of X approximately coincide with the labels. The optimal v
∗is related to those vectors and aligned with the labels, since the quadratic and the bilinear form in (2) are individually maximized by the mentioned vectors. The maximizer of the linear combination of both terms therefore combines the spectral information from both matrices—the graph and the features. This suggests the following rationale for denoising: (1) We can interpret the value of (2) as a measure of alignment. Since v
∗corresponds to the labels, we can relate this measure to the quality of the label estimation. (2) We may leverage this alignment to rewire the graph and denoise the features. Namely, we could perturb A and X in a way that improves the alignment. In real datasets, however, the optimal value of b is unknown, the scaling of X is arbitrary, and things are further complicated by having (many) more than 2 classes. Moreover, (2) is computationally hard. We thus define a simple related measure of alignment which alleviates these issues. Definition 1. *Recall the decompositions* A = V ΛV
T, X = UΣWT, and let VL, UL *denote the* first L columns of V and U and ∥.∥sp *the spectral norm. We define graph–feature alignment as*

$$\operatorname{Alignment}_{L}(\mathbf{A},\mathbf{X})=\|\mathbf{V}_{L}^{T}\mathbf{U}_{L}\|_{\mathrm{sp}}.$$
$\eqref{eq:walpha}$. 
L UL∥sp. (3)
Remark: The logic of this definition is that for a cSBM with high SNR and k classes, the information about labels is indeed contained in the leading L = k vectors of V and U. This follows directly by generalizing the formulation in (2) to multiple classes and thus multiple eigenvectors (Decelle et al.,
2011; Lesieur et al., 2017). The quantity AlignmentL(A, X) is the cosine of the angle between the subspaces spanned by the columns of VL and UL. To denoise the features and rewire the graph, we seek to improve the alignment.

Given AlignmentL(A, X) and a graph with A0 and X0, the jointly denoised graph and features are the solution to

$$\quad(4)$$
$$\operatorname*{maximize}_{\mathbf{A},\mathbf{X}}\operatorname{Alignment}_{L}(\mathbf{A},\mathbf{X})$$ subject to $\|\mathbf{A}-\mathbf{A}_{0}\|\leq\delta_{A},\|\mathbf{X}-\mathbf{X}_{0}\|\leq\delta_{X}$. 
The parameters δA, δX > 0 modulate the strength of alignment. We will show empirically that a stronger alignment indicates a better representation of the labels by A and X and thus a better graph. Figure 2 visualizes this connection. It shows that the response of the graph to features is maximized when the spectra of the graph and the features are aligned. We refer to the condition where the alignment is high as spectral resonance; see Appendix A.1.1 for further discussion.

## 2.3 Joint Denoising And Rewiring Algorithm

Maximizing the alignment (4) directly, e.g., using gradient descent, is computationally challenging. Here we propose a heuristic which alternates between spectral interpolation and graph synthesis. We later prove that the resulting algorithm indeed improves alignment, both with the labels and between the graph and the features, under a stylized noise model. The algorithm, illustrated in Figure 1, comprises three steps. In Step 1, we compute the spectral decompositions of A and X. To improve the alignment, we interpolate between the L largest eigenvectors in Step 2. Based on the new eigenvectors, we synthesize a new graph in Step 3. The three steps are iterated until a stopping criterion is met. As is standard in the rewiring literature, the hyperparameters of the algorithm are tuned on a validation set. Formalizing this results in the JDR algorithm:
Step 1: Decomposition A = V ΛV
T with V = (v1, v2*, . . . ,* vN ) and X = UΣWT with U = (u1,u2*, . . . ,*uN )
Step 2: Interpolation: For every i ∈ [L],
v˜i = (1 − ηA)vi + ηA sign(⟨vi,uj ⟩)uj u˜i = (1 − ηX)ui + ηX sign(⟨ui, vj ⟩)vj where j is chosen as argmaxj∈[L]|⟨vi,uj ⟩| when updating vi and as argmaxj∈[L]|⟨ui, vj ⟩| when updating ui. ηA and ηX are hyperparameters that are tuned with a downstream algorithm on a validation set. We use sign() to handle sign ambiguities in decompositions.

Step 3: Graph Synthesis

$$\operatorname{d}\mathbf{X}=\mathbf{U}\mathbf{\Sigma}\mathbf{W}^{T}\operatorname{w}$$
, $\boldsymbol{u}_N$). 
$$\bar{\mathbf{A}}=\bar{\mathbf{V}}\mathbf{\Lambda}\bar{\mathbf{V}}^{T}{\mathrm{~and~}}\bar{\mathbf{X}}=\bar{\mathbf{U}}\mathbf{\Sigma}\mathbf{W}^{T}$$

Step 4: Iterate steps K times with

A ← A˜ and X ← X˜ .

Following (3), we consider the L leading eigenvectors of A and X for interpolation. Since these bases may be rotated with respect to each other (we note that (3) is insensitive to relative rotations, see Appendix A.1.2), when updating an eigenvector of A, we interpolate it with the most similar eigenvector of X. We show empirically that this heuristic yields strong results, but also prove that it improves alignment with labels with a stylized noise model. We emphasize that the interpolation rates ηA and ηX are the same across different eigenvectors and iterations K. After K steps, we synthesize the final weighted dense graph A˜ = V(K)ΛV
T
(K)
. To efficiently apply GNNs, we can enforce sparsity, e.g., via thresholding or selecting the top-k entries per node. A detailed pseudocode is given in Appendix A.1. An illustration. A simple edge case to illustrate how the algorithm works is when either only A or X contains information. In a cSBM with ϕ = 0, X contains all the information, so the best hyperparameter choice is ηX = 0 and (4) simplifies to a maximization over A. Since there are only two classes, it is sufficient to consider L = 1. From (2) we know that the leading left singular vector u1 of X is well-aligned with the labels. We thus replace the second leading eigenvector v2 in A
by u1 by choosing ηA = 1.0. After graph synthesis, the new v2 of A˜ is not yet equal to u1, since u1 was not orthogonal to the other vi. We thus repeat the three steps K times. For ϕ = ±1 all information is contained in the graph; a similar argument can then be constructed *mutatis mutandis*. JDR Improves Alignment. We now show that JDR improves alignment, as defined in (4), under a stylized cSBM-like model. In fact, we show a stronger result: that the algorithm improves alignment with the true labels. Appealing to universality arguments (Hu and Lu, 2023), we study a model with a spiked Gaussian matrix Ac =
λ N
yyT + √
1 N
OA where OA is GOE noise instead of the binary matrix

Table 1: Comparison of state-of-the-art preprocessing rewiring approaches. Note that we refer to the computational complexity per iteration. N denotes the number of nodes, m the number of edges

and dmax is the maximum node degree. Additional details on the complexity of JDR are given in

Appendix A.1.3; detailed runtime comparisons are in Appendix A.1.4

.

Method Add edge Remove edge Use Features Heterophilic? Complexity

DIGL (Gasteiger et al., 2019) ✓ ✗ ✗ ✗ O(N) FoSR (Karhadkar et al., 2023) ✓ ✗ ✗ ✓ O(N2)

BORF (Nguyen et al., 2023) ✓ ✓ ✗ ✓ O(md3max)

JDR (Ours) ✓ ✓ ✓ ✓ O(N)

A, and features X =p µN
yξT + √
1 F
OX as defined in 1. In the cSBM context, a recent conjecture with strong empirical support states that replacing the binary A by Acleads to the same behavior in downstream tasks such as community detection (Deshpande et al., 2018; Lu and Sen, 2023) and node classification (Shi et al., 2024). An iteration of JDR with L = 1 applied to this model, first interpolates between the leading eigenvector v1 (Ac) = vA and leading left singular vector u1 (X) = uX. Graph and feature synthesis then yields AcηA = Ac + λ1 (Ac)−vAv T
A + v˜Av˜
T A

and XηX = X + σ1 (X)−uXwTX + u˜XwTX
. Here v˜A = (1 − ηA)vA + sign(⟨vA,uX⟩)ηAvX
and u˜X = (1−ηX)uX + sign(⟨vA,uX⟩)ηXvA, where w1 (X) = wX is the leading right singular vector of X. Denoting y˜ = y/
√N, we have Proposition 1. Let λ > 1 and µ > 
√γ with γ = N/F*. There exist* η 0A, η0X ∈ (0, 1) such that for all ηA ∈ (0, η0A) and ηX ∈ (0, η0X), when N → ∞*, we have*

$$\left\langle\mathbf{v}_{1}(\mathbf{A}_{\eta_{A}}^{c}),\tilde{\mathbf{y}}\right\rangle^{2}\stackrel{{a.s}}{{>}}\left\langle\mathbf{v}_{1}(\mathbf{A}^{c}),\tilde{\mathbf{y}}\right\rangle^{2}\quad\mathrm{and}\quad\left\langle\mathbf{u}_{1}(\mathbf{X}_{\eta_{X}}),\tilde{\mathbf{y}}\right\rangle^{2}\stackrel{{a.s}}{{>}}\left\langle\mathbf{u}_{1}(\mathbf{X}),\tilde{\mathbf{y}}\right\rangle^{2}.$$

In words, interpolation improves alignment of the largest eigenvector with the labels y for sufficiently large graphs. The proof, based on the BBP transition in the spiked covariance model (Baik et al., 2005) and the fluctuation of the leading eigenvector, can be found in Appendix A.3. It seems challenging but quite possible to extend this argument to a binary A. One would then interpolate between ux and A's second leading eigenvector v2(A), which has similar properties to v1(Ac),
especially in a dense graph regime (Nadakuditi and Newman, 2012).

## 3 Experiments

We extensively evaluate JDR on both synthetic data generated from the cSBM and real-world benchmark datasets. We follow experimental setting from Chien et al. (2021) and evaluate JDR for semisupervised node classification with different downstream GNNs. We also adopt their data splits, namely the sparse splitting 2.5%/2.5%/95% for training, validation and testing, respectively, or the dense splitting 60%/20%/20%. For the general experiments, we perform 100 runs with different random splits. For the scalability experiments, we use the experimental settings of the respective works (Lim et al., 2021; Platonov et al., 2023). We report the average accuracy and the 95%-confidence interval calculated via bootstrapping with 1000 samples. All experiments are reproducible using the code provided. Baselines. Following recent works on rewiring, we use graph convolution network (GCN) (Kipf and Welling, 2017) as our downstream GNN. To obtain a more comprehensive picture, we additionally evaluate the performance on the more recent generalized PageRank graph neural network (GPRGNN) (Chien et al., 2021). We compare our algorithm with the state-of-the-art rewiring methods first-order spectral rewiring (FoSR) (Karhadkar et al., 2023), batch Ollivier-Ricci flow (BORF) (Nguyen et al., 2023) and diffusion improves graph learning (DIGL) (Gasteiger et al., 2019). FoSR approximates which edges should be added to maximize the spectral gap to reduce oversquashing.

BORF adds edges in regions of negative curvature in the graph, which indicate bottlenecks that can lead to an oversquashing of the messages passed along these edges. A positive curvature indicates that there are so many edges in this area that messages could be oversmoothed, which is why edges are removed here. We compare computational and implementation aspects of JDR and baselines in Table 1. On the cSBM, we compare to an *optimal* algorithm, namely the approximate message passing-belief propagation (AMP-BP) algorithm (Duranthon and Zdeborová, 2023). AMP-BP is asymptotically optimal (in the large dimension limit) for unsupervised or semi-supervised community detection in the cSBM. It relies on knowing the distribution of the cSBM and is thus not applicable to real-world graphs with unknown characteristics and complex features. Hyperparameters. Unless stated otherwise, we use the hyperparameters from Chien et al. (2021) for the GNNs and optimize the hyperparameters of JDR using a mixture of grid and random search on the validation set. We use the top-64 values of A˜ to enforce sparsity and interpolation to update the features. For DIGL, FoSR and BORF, we tune their hyperparameters using a grid search, closely following the given parameter range from the original papers. For all hyperparameter searches we use GCN and GPRGNN as the downstream models on 10 runs with different random splits. A detailed list of all hyperparameters can be found in Appendix A.7 or in the code repository.

## 3.1 Results On Synthetic Data

We first test JDR on data generated from the cSBM, as we can easily vary the SNR of the graph and the features to verify its denoising and rewiring capabilities. We focus on the sparse splitting, since for the dense splitting GPRGNN already matches the performance of AMP-BP.

-1.0 -0.75 -0.5 -0.25 0.0 0.25 0.5 0.75 1.0 0.0 0.2 0.4 0.6 0.8 1.0 A

l i g n m e n t L

(

A, 
X

)

JDR
None
Does JDR Maximize Alignment? Before discussing Figure 4, which shows the results of baselines and JDR for different values of ϕ, we verify empirically that our alternating optimization algorithm indeed approximates solutions to (4). As shown in Figure 3, the quantity AlignmentL(A, X) improves significantly after running JDR, across all ϕ. As we show next, this happens simultaneously with improvements in downstream performance, which lends credence to the intuitive reasoning that motivates our algorithm. For additional alignment results on real-world data and baselines, refer to Appendix A.5.5. Heterophilic Regime. For ϕ < −0.25, the predictions of GCN are only slightly better than random. GPRGNN performs much better, since it can learn higher order polynomial filters to deal with heterophily. GCN+JDR outperforms the baseline by a very large margin; it handles heterophilic data well. Using JDR for GPRGNN further improves its already strong performance in this regime. Both GNNs benefit less from the denoising in the weakly heterophilic setting where they exhibit the worst performance across all ϕ. The difference between ϕ = 0 and the weakly heterophilic regime is that "optimal denoising" for ϕ = 0 is straightforward, since all the information is contained in X. We show similar findings for spectral clustering on the cSBM in Appendix A.5.6.

Figure 3: Alignment of the leading eigenspaces according to (3) for graphs from the cSBM with different ϕ.

-1.0 -0.75 -0.5 -0.25 0.0 0.25 0.5 0.75 1.0 40 50 60 70 80 90 100 Accuracy in
 %
AMP-BP
GPRGNN+JDR GPRGNN
GCN+JDR GCN MLP
Heterophilic Weak Homophilic
Weak Graph Regime. For |ϕ| ≤ 0.25, where the SNR of the graph is very low, both GNNs perform poorly. Intuitively, when the graph is very noisy, a GNN is a suboptimal model, since it leverages the graph structure. A simple MLP baseline, using only the node features, outperforms GNNs in this setting, with all three approaches lagging far behind AMP-BP. Using JDR, we see significant improvements for both GNNs, which almost catch up with AMP-BP for ϕ = 0. Although all information was available in the node features, the GNN with JDR now clearly outperform the MLP by a very large margin.

We argue that this is because in the semisupervised setting with few labels available, the GNN generalizes much better. Homophilic Regime. For ϕ > 0.25, GCN and GPRGNN perform similarly well, with

Method Cora CiteSeer PubMed Computers Photo GCN 77.26±0.35 67.16±0.37 84.22±0.09 84.42±0.31 91.33±0.29

GCN+DIGL 79.27±0.26 68.03±0.33 84.60±0.09 86.00±0.24 92.00±0.23

GCN+FoSR 77.23±0.34 67.03±0.34 84.21±0.09 84.34±0.27 91.36±0.28 GCN+BORF 77.23±0.35 66.96±0.38 84.22±0.09 84.46±0.30 91.26±0.30 GCN+JDR 79.96±0.26 69.35±0.28 84.79±0.08 85.66±0.36 92.52±0.23 GPRGNN 79.65±0.33 67.50±0.35 84.33±0.10 84.06±0.48 92.01±0.41 GPRGNN+DIGL 79.77±0.30 67.50±0.35 84.72±0.10 86.25±0.28 92.31±0.25 GPRGNN+FoSR 79.22±0.31 67.30±0.38 84.32±0.09 84.21±0.46 92.07±0.37 GPRGNN+BORF 79.43±0.30 67.48±0.36 84.36±0.10 84.08±0.43 92.11±0.38 GPRGNN+JDR 80.77±0.29 69.17±0.30 85.05±0.08 84.77±0.35 92.68±0.25

Table 3: Results on real-world heterophilic dataset in the dense splitting (60%/20%/20%): Mean accuracy across runs (%) ± 95% confidence interval. Best average accuracy in **bold**.

Method Chameleon Squirrel Actor Texas Cornell GCN 67.65±0.42 57.94±0.31 34.00±0.31 75.62±1.12 64.68±1.25 GCN+DIGL 58.04±0.48 39.64±0.34 39.57±0.29 91.05±0.73 88.49±0.74 GCN+FoSR 67.67±0.39 58.12±0.35 33.98±0.30 78.31±1.07 65.64±1.06 GCN+BORF 67.78±0.43 OOM 33.95±0.31 76.66±1.10 68.72±1.11 GCN+JDR 69.76±0.50 61.76±0.39 40.47±0.31 85.12±0.74 84.51±1.06 GPRGNN 69.15±0.51 53.44±0.37 39.52±0.22 92.82±0.67 87.79±0.89 GPRGNN+DIGL 66.57±0.46 42.98±0.37 39.61±0.21 91.11±0.72 88.06±0.81

GPRGNN+FoSR 68.96±0.45 52.34±0.37 39.47±0.21 93.16±0.66 87.51±1.04

GPRGNN+BORF 69.44±0.56 OOM 39.55±0.20 93.53±0.68 88.83±1.06 GPRGNN+JDR 71.00±0.50 60.62±0.38 41.89±0.24 93.85±0.54 89.45±0.84

GPRGNN achieving better results for ϕ → 1.0. With JDR, they become much more comparable to each other and closer to AMP-BP. Even though the hyperparameters of JDR were tuned using only GCN as a downstream model, it also improves the performance of GPRGNN for all ϕ. The general robustness to hyperparameter changes is also analyzed in detail in Appendix A.6.

## 3.2 Results On Real-World Data

We evaluate JDR on five common homophilic benchmarks datasets, namely the citation graphs Cora, CiteSeer, PubMed (Sen et al., 2008) and the Amazon co-purchase graphs Computers and Photo (McAuley et al., 2015). For heterophilic datasets, we rely on the Wikipedia graphs Chameleon and Squirrel (Rozemberczki et al., 2021), the WebKB datasets Texas and Cornell used in Pei et al. (2020) and the actor co-occurence network Actor (Tang et al., 2009). To show the scalability of JDR on larger heterophilic datasets, we further report the results for the Yandex Q user network Questions (Platonov et al., 2023) and the social networks Penn94 and Twitch-Gamers (Lim et al., 2021). Further details about all datasets are in Appendix A.4. Following Chien et al. (2021), we evaluate the homophilic datasets in the sparse splitting, staying close to the original setting of Kipf and Welling (2017) and the heterophilic datasets in dense splitting (Pei et al., 2020). The remaining larger graphs are evaluated using their original splits. For further results and splits, see Appendix A.5. Homophilic Datasets. Table 2 shows the results of JDR compared to the baselines. For both GNNs, JDR achieves the best results on four out of five datasets. GCN and GPRGNN with JDR achieve similar performance here, which is consistent with the findings for the homophilic cSBM. DIGL also performs strongly on the datasets and ranks first on Computers. However, with GPRGNN as a downstream model, the improvements are quite small. FoSR and BORF only marginally improve the performance of the GNNs in this setting. Heterophilic Datasets. The results in Table 3 show that GCN+JDR can catch up significantly compared to GPRGNN, but GPRGNN+JDR generally performs better. This is in line with the findings for the heterophilic cSBM. DIGL performs well on Actor, Texas and Cornell despite its inherent homophily assumption. The reason for this is the chosen smoothing kernel, which results in a graph that is evenly connected everywhere with small weights. GCN then largely ignores the graph and thus performs very similarly to an MLP, which performs already quite well on these datasets (Chien et al., 2021). However, this fails for GPRGNN, which can make better use of the weak, complex graph structures. FoSR and BORF also improve performance here in most cases, but they are outperformed by JDR in all cases, often by a large margin. The out-of-memory error on Squirrel for BORF results from its computational complexity of O(md3max), because the dataset has a large number of edges m and a high maximum node degree dmax.

Larger Graphs. Scalability is a problem for preprocessing rewiring methods because applying them to large graphs requires significant amounts of memory and compute (see complexity in Table 1). Since the decompositions needed for JDR
can be truncated to the largest L vectors, it is still applicable to larger graphs.

The experimental results on larger heterophilic datasets in Table 4 verify this. They show that JDR can significantly improve performance for these larger graphs, while FoSR only achieves marginal improvements. BORF ran out of memory on all of these datasets and DIGL is unable to improve due to its inherent homophily assumption. While scaling JDR to even larger graphs with millions of nodes is possible in principle, it requires more optimized and efficient implementations and is therefore left for future work.

Table 4: Results on large datasets. Mean accuracy (ROC
AUC for imbalanced Questions) across runs (%) ± 95%
confidence interval. Best results in **bold**.

Method Questions Penn94 Twitch-gamers \# nodes 48, 921 41, 554 168, 114 \# edges 0.15M 1.36M 6.8M GCN 75.31 ± 0.81 80.40 ± 0.18 64.56 ± 0.19 GCN+DIGL 73.35 ± 0.64 74.70 ± 0.*32 61*.64 ± 0.14 GCN+FoSR 75.51 ± 0.*73 80*.54 ± 0.31 64.65 ± 0.15 GCN+JDR 77.52 ± 0.*63 82*.30 ± 0.61 65.14 ± 0.19

## 4 Relation To Prior Work

JDR is most related to preprocessing rewiring methods which we thus use as baselines. To provide a more thorough overview, we also place it within the extended literature. Graph Rewiring. Recent work show that even when a graph correctly encodes interactions it may not be an effective *computational* graph for a GNN due to conditions such as *oversquashing* (Alon and Yahav, 2021; Di Giovanni et al., 2023) and *oversmoothing* (Chen et al., 2020a). Recently many methods have been proposed to address this, notably *graph rewiring methods*. They can be divided into preprocessing and end-to-end methods. Preprocessing methods rewire the graph using geometric and spectral properties, including curvature (Topping et al., 2022; Nguyen et al., 2023; Fesser and Weber, 2024; Bober et al., 2024), expansion (Deac et al., 2022; Banerjee et al., 2022), effective resistance (Black et al., 2023; Shen et al., 2024), and spectral gap (Karhadkar et al., 2023). Conceptually related is diffusion-based rewiring (Gasteiger et al., 2019) that smooths the graph with a diffusion kernel. This can be interpreted as graph denoising, but is only suitable for homophilic graphs. Our approach is related to rewiring but with several key differences (see Table 1). Our rewiring strategy aims to denoise the graph (rather than control some geometric property) with the goal to improve downstream performance, while the classical rewiring literature focuses on optimizing the graph for message passing computations. Early end-to-end methods randomly drop edges during training to reduce oversmoothing (Rong et al., 2020). Subsequent work (Gutteridge et al., 2023; Qian et al., 2024) incorporates latent features to dynamically rewire the graph. Ji et al. (2023) use the estimated labels of a GNN to rewire the graph during training of the same GNN. Giraldo et al. (2023) use curvature information for dynamic rewiring. Graph Transformers (Dwivedi and Bresson, 2021; Rampasek et al., 2022) aim to overcome oversquashing in GNNs via global attention. In order to handle large graphs, these works still need to revert to sparse, non-global attention (Gabrielsson et al., 2023; Shirzad et al., 2023). Unlike preprocessing methods, end-to-end methods cannot output an improved graph which restricts their interpretability and reusability. Graph Denoising. There is extensive literature on denoising signals on graphs using graph filters (Chen et al., 2014; Ma et al., 2021b; Liu et al., 2022). However, we are interested in modifying the structure of the graph itself (rewiring), in a way that can benefit any downstream algorithm. Dong and Kluger (2023) recently proposed a new metric to measure graph noise that correlates well with GCN performance. Based on this, they develop a method for graph denoising via self-supervised learning and link prediction. We discuss the relation to our work in detail in Appendix A.5.7 and also evaluate our rewired graphs using their ESNR metric there. More broadly, link prediction (Zhang and Chen, 2018; Pan et al., 2022) can be seen as a tool for graph denoising; this perspective has been applied, for instance, to denoising neighborhood graphs arising in molecular imaging (Debarnot et al., 2022). Graph Structure Learning. The aim of graph structure learning (GSL) (Zhu et al., 2022) is to make GNNs more robust against adversarial perturbations of the graph or to learn a graph for data where there is no graph to start with (Jin et al., 2020; Chen et al., 2020b; Wang et al., 2024; Zhu et al., 2024). Lv et al. (2023) build a neighborhood graph over features and interpolate between it and the input graph, which is a form of alignment. Unlike our method, they do not use spectral information, are unable to deal with noisy features and are only suitable for homophilic graphs where similaritybased connection rules apply. Even though both our work and GSL consider noisy graph settings, they are conceptually very different. We do not add noise to graph datasets which corresponds to a perturbation rate of 0 ("clean data") in GSL nomenclature. Instead, we acknowledge that in every real world dataset, there is noise in the graph structure and the node features, and one manifestation of this noise is a misalignment of their leading eigenspaces. We then use this to rewire the graph (and denoise features) so as to improve the overall node classification performance of downstream GNNs. Naturally, GSL methods have difficulties to improve over baselines in this setting (Jin et al., 2020; Dong and Kluger, 2023). Our method, on the other hand, is not designed to handle strong perturbations and therefore cannot compete with GSL methods developed for specifically this purpose. Graph Regularization. Laplacian regularization (Ando and Zhang, 2006), originally stemming from semi-supervised representation learning, has been adapted by recent methods (Yang et al., 2021; Ma et al., 2021a) to also improve the performance of GNNs. An extra loss term is added during the GNNs training, which contains additional information about the graph structure to reduce oversmoothing. Their main limiting factor is the underlying homophiliy assumption: It is assumed that connected nodes are more likely to share the same label.

## 5 Conclusion And Limitations

Our experimental results clearly show that spectral resonance is a powerful principle on which to build graph rewiring (and feature denoising) algorithms. JDR consistently outperforms existing rewiring methods DIGL, FoSR and BORF on both synthetic and real-world graph datasets. The smaller performance gains of GPRGNN suggest that this more powerful GNN is already able to leverage the complementary spectra of graphs and features to some extent. The main limitation of JDR is that it cannot be used without node features. The preprocessing rewiring methods that we compare with do not have this limitation as they only use the graph for rewiring, but in turn, they cannot take advantage of features. Since JDR is the first method to jointly denoise the graph and the features, there are no other methods to which it could be directly compared. Our experiments thus highlight what advantage features bring to rewiring. Furthermore, our results suggest that noise in real-world graphs is an important limiting factor for the performance of GNNs. It would be interesting to see whether feature-agnostic rewiring from a denoising perspective, for example using link prediction, could be used to improve the downstream performance. A related idea that we tested but could not get to work well is to combine existing geometric rewiring algorithms with JDR. Intuitively, there should be a way to benefit from both removing noise and facilitating computation, but we have to leave that exploration for future work. We also note that most current rewiring methods can be applied to graph level tasks, while JDR is currently limited to node classification. It is an open question how to extend the cSBM idea to graph-level problems.

## Acknowledgments

JL, CS and ID were supported by the European Research Council (ERC) Starting Grant 852821— SWING.

## References

E. Abbe. Community detection and stochastic block models: Recent developments. *Journal of* Machine Learning Research, 18(177):1–86, 2018. URL http://jmlr.org/papers/v18/ 16-480.html.

U. Alon and E. Yahav. On the bottleneck of graph neural networks and its practical implications. In International Conference on Learning Representations, 2021. URL https://openreview. net/forum?id=i80OPhOCVH2.

R. Ando and T. Zhang. Learning on Graph with Laplacian Regularization. In B. Schölkopf, J. Platt, and T. Hoffman, editors, *Advances in Neural Information Processing Systems*, volume 19. MIT Press, 2006. URL https://proceedings.neurips.cc/paper_files/paper/ 2006/file/d87c68a56bc8eb803b44f25abb627786-Paper.pdf.

J. Baik, G. B. Arous, and S. Péché. Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. *The Annals of Probability*, 33(5):1643 - 1697, 2005. doi: 10.1214/ 009117905000000233. URL https://doi.org/10.1214/009117905000000233.

P. K. Banerjee, K. Karhadkar, Y. G. Wang, U. Alon, and G. Montúfar. Oversquashing in gnns through the lens of information contraction and graph expansion. In 2022 58th Annual Allerton Conference on Communication, Control, and Computing (Allerton), page 1–8. IEEE Press, 2022. doi: 10.1109/Allerton49937.2022.9929363. URL https://doi.org/10.1109/ Allerton49937.2022.9929363.

A. Baranwal, K. Fountoulakis, and A. Jagannath. Graph convolution for semi-supervised classification: Improved linear separability and out-of-distribution generalization. In M. Meila and T. Zhang, editors, *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pages 684–693. PMLR, 07 2021. URL https://proceedings.mlr.press/v139/baranwal21a.html.

P. W. Battaglia, J. B. Hamrick, V. Bapst, A. Sanchez-Gonzalez, V. Zambaldi, M. Malinowski, A. Tacchetti, D. Raposo, A. Santoro, R. Faulkner, C. Gulcehre, F. Song, A. Ballard, J. Gilmer, G. Dahl, A. Vaswani, K. Allen, C. Nash, V. Langston, C. Dyer, N. Heess, D. Wierstra, P. Kohli, M. Botvinick, O. Vinyals, Y. Li, and R. Pascanu. Relational inductive biases, deep learning, and graph networks, 2018.

F. Benaych-Georges and R. R. Nadakuditi. The singular values and vectors of low rank perturbations of large rectangular random matrices. *Journal of Multivariate Analysis*, 111:120–135, 2012.

F. Benaych-Georges, A. Guionnet, and M. Maida. Fluctuations of the extreme eigenvalues of finite rank deformations of random matrices. *Electronic Journal of Probability*, 16:1621–1662, 2011.

M. Black, Z. Wan, A. Nayyeri, and Y. Wang. Understanding oversquashing in GNNs through the lens of effective resistance. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 2528–2547. PMLR, 7 2023. URL https://proceedings.mlr.press/v202/black23a.html.

J. Bober, A. Monod, E. Saucan, and K. N. Webster. Rewiring networks for graph neural network training using discrete geometry. In H. Cherifi, L. M. Rocha, C. Cherifi, and M. Donduran, editors, *Complex Networks & Their Applications XII*, pages 225–236, Cham, 2024. Springer Nature Switzerland. ISBN 978-3-031-53468-3.

S. Chanpuriya and C. Musco. Simplified graph convolution with heterophily. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 27184–27197. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/ file/ae07d152c51ea2ddae65aa7192eb5ff7-Paper-Conference.pdf.

D. Chen, Y. Lin, W. Li, P. Li, J. Zhou, and X. Sun. Measuring and relieving the over-smoothing problem for graph neural networks from the topological view. Proceedings of the AAAI Conference on Artificial Intelligence, 34(04):3438–3445, 4 2020a. doi: 10.1609/aaai.v34i04.5747. URL
https://ojs.aaai.org/index.php/AAAI/article/view/5747.

S. Chen, A. Sandryhaila, J. M. F. Moura, and J. Kovacevic. Signal denoising on graphs via graph filtering. In *2014 IEEE Global Conference on Signal and Information Processing (GlobalSIP)*, pages 872–876, 2014. doi: 10.1109/GlobalSIP.2014.7032244.

Y. Chen, L. Wu, and M. Zaki. Iterative deep graph learning for graph neural networks: Better and robust node embeddings. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 19314–19326. Curran Associates, Inc., 2020b. URL https://proceedings.neurips.cc/paper_files/
paper/2020/file/e05c7ba4e087beea9410929698dc41a6-Paper.pdf.

E. Chien, J. Peng, P. Li, and O. Milenkovic. Adaptive universal generalized pagerank graph neural network. In *International Conference on Learning Representations*, 2021. URL https:// openreview.net/forum?id=n6jl7fLxrP.

E. Chien, W.-C. Chang, C.-J. Hsieh, H.-F. Yu, J. Zhang, O. Milenkovic, and I. S. Dhillon. Node feature extraction by self-supervised multi-scale neighborhood prediction. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=KJggliHbs8.

A. Deac, M. Lackenby, and P. Velickovi ˇ c. Expander graph propagation. In ´ NeurIPS 2022 Workshop on Symmetry and Geometry in Neural Representations, 2022. URL https://openreview. net/forum?id=6cthqh2qhCT.

V. Debarnot, V. Kishore, C. Shi, and I. Dokmanic. Manifold rewiring for unlabeled imaging. In ´
2022 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pages 1–8, 2022. doi: 10.23919/APSIPAASC55919.2022.9980168.

A. Decelle, F. Krzakala, C. Moore, and L. Zdeborová. Asymptotic analysis of the stochastic block model for modular networks and its algorithmic applications. *Phys. Rev. E*, 84:066106, Dec 2011. doi: 10.1103/PhysRevE.84.066106. URL https://link.aps.org/doi/10. 1103/PhysRevE.84.066106.

Y. Deshpande, S. Sen, A. Montanari, and E. Mossel. Contextual stochastic block models. *Advances* in Neural Information Processing Systems, 31, 2018.

F. Di Giovanni, L. Giusti, F. Barbero, G. Luise, P. Lio, and M. M. Bronstein. On over-squashing in message passing neural networks: The impact of width, depth, and topology. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, *Proceedings of the 40th* International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 7865–7885. PMLR, 07 2023. URL https://proceedings.mlr.press/
v202/di-giovanni23a.html.

M. Dong and Y. Kluger. Towards understanding and reducing graph structural noise for GNNs. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 8202–8226. PMLR, 07 2023. URL https://proceedings. mlr.press/v202/dong23a.html.

O. Duranthon and L. Zdeborová. Optimal inference in contextual stochastic block models, 2023.

D. K. Duvenaud, D. Maclaurin, J. Iparraguirre, R. Bombarell, T. Hirzel, A. Aspuru-Guzik, and R. P. Adams. Convolutional networks on graphs for learning molecular fingerprints. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015. URL https://proceedings.neurips.cc/paper_files/paper/2015/ file/f9be311e65d81a9ad8150a60844bb94c-Paper.pdf.

V. P. Dwivedi and X. Bresson. A generalization of transformer networks to graphs. AAAI Workshop on Deep Learning on Graphs: Methods and Applications, 2021.

P. Erdös and A. Rényi. On random graphs i. *Publicationes Mathematicae Debrecen*, 6:290–297, 1959.

L. Fesser and M. Weber. Mitigating over-smoothing and over-squashing using augmentations of forman-ricci curvature. In S. Villar and B. Chamberlain, editors, Proceedings of the Second Learning on Graphs Conference, volume 231 of *Proceedings of Machine Learning Research*, pages 19:1–19:28. PMLR, 11 2024. URL https://proceedings.mlr.press/v231/ fesser24a.html.

M. Fey and J. E. Lenssen. Fast graph representation learning with PyTorch Geometric. In ICLR
Workshop on Representation Learning on Graphs and Manifolds, 2019.

R. B. Gabrielsson, M. Yurochkin, and J. Solomon. Rewiring with positional encodings for graph neural networks. *Transactions on Machine Learning Research*, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=dn3ZkqG2YV.

J. Gasteiger, S. Weiß enberger, and S. Günnemann. Diffusion improves graph learning. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc.,
2019. URL https://proceedings.neurips.cc/paper_files/paper/2019/ file/23c894276a2c5a16470e6a31f4618d73-Paper.pdf.

J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl. Neural message passing for quantum chemistry. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML'17, page 1263–1272. JMLR.org, 2017.

J. H. Giraldo, K. Skianis, T. Bouwmans, and F. D. Malliaros. On the trade-off between oversmoothing and over-squashing in deep graph neural networks. CIKM '23, page 566–576, New York, NY, USA, 2023. Association for Computing Machinery. doi: 10.1145/3583780.3614997.

URL https://doi.org/10.1145/3583780.3614997.

V. Gligorijevic, P. D. Renfrew, T. Kosciolek, J. K. Leman, D. Berenberg, T. Vatanen, C. Chandler, ´
B. C. Taylor, I. M. Fisk, H. Vlamakis, R. J. Xavier, R. Knight, K. Cho, and R. Bonneau. Structurebased protein function prediction using graph convolutional networks. *Nature Communications*,
12(1):3168, 2021. ISSN 2041-1723. doi: 10.1038/s41467-021-23303-9. URL https://doi. org/10.1038/s41467-021-23303-9.

B. Gutteridge, X. Dong, M. M. Bronstein, and F. Di Giovanni. DRew: Dynamically rewired message passing with delay. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 12252–12267. PMLR, 07 2023. URL https://proceedings.mlr.press/v202/gutteridge23a.html.

H. Hu and Y. M. Lu. Universality laws for high-dimensional learning with random features. IEEE
Transactions on Information Theory, 69(3):1932–1964, 2023. doi: 10.1109/TIT.2022.3217698.

F. Ji, S. H. Lee, H. Meng, K. Zhao, J. Yang, and W. P. Tay. Leveraging label non-uniformity for node classification in graph neural networks. In Proceedings of the 40th International Conference on Machine Learning, ICML'23. JMLR.org, 2023.

W. Jin, Y. Ma, X. Liu, X. Tang, S. Wang, and J. Tang. Graph structure learning for robust graph neural networks. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD '20, page 66–74, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450379984. doi: 10.1145/3394486.3403049. URL
https://doi.org/10.1145/3394486.3403049.

K. Karhadkar, P. K. Banerjee, and G. Montufar. FoSR: First-order spectral rewiring for addressing oversquashing in GNNs. In *The Eleventh International Conference on Learning Representations*, 2023. URL https://openreview.net/forum?id=3YjQfCLdrzz.

T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017. URL https://openreview. net/forum?id=SJU4ayYgl.

V. Kothapalli, T. Tirer, and J. Bruna. A neural collapse perspective on feature evolution in graph neural networks. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.

URL https://openreview.net/forum?id=sxao2udWXi.

T. Lesieur, F. Krzakala, and L. Zdeborová. Constrained low-rank matrix estimation: phase transitions, approximate message passing and applications. Journal of Statistical Mechanics: Theory and Experiment, 2017(7):073403, jul 2017. doi: 10.1088/1742-5468/aa7284. URL https: //dx.doi.org/10.1088/1742-5468/aa7284.

P. Li, I. Chien, and O. Milenkovic. Optimizing generalized pagerank methods for seed-expansion community detection. In *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper_files/ paper/2019/file/9ac1382fd8fc4b631594aa135d16ad75-Paper.pdf.

D. Lim, F. M. Hohne, X. Li, S. L. Huang, V. Gupta, O. P. Bhalerao, and S.-N. Lim. Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, editors, *Advances in Neural Information Processing* Systems, 2021. URL https://openreview.net/forum?id=DfGu8WwT0d.

J. Linkerhägner, N. Freymuth, P. M. Scheikl, F. Mathis-Ullrich, and G. Neumann. Grounding graph network simulators using physical sensor observations. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= jsZsEd8VEY.

S. Liu, R. Ying, H. Dong, L. Lin, J. Chen, and D. Wu. How powerful is implicit denoising in graph neural networks, 2022.

C. Lu and S. Sen. Contextual stochastic block model: Sharp thresholds and contiguity. *Journal of* Machine Learning Research, 24(54):1–34, 2023.

S. Luan, C. Hua, M. Xu, Q. Lu, J. Zhu, X.-W. Chang, J. Fu, J. Leskovec, and D. Precup. When do graph neural networks help with node classification? investigating the homophily principle on node distinguishability. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 28748–28760. Curran Associates, Inc.,
2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/ file/5ba11de4c74548071899cf41dec078bf-Paper-Conference.pdf.

S. Lv, G. Wen, S. Liu, L. Wei, and M. Li. Robust graph structure learning with the alignment of features and adjacency matrix, 2023.

X. Ma, H. Chen, and G. Song. Lereg: Empower graph neural networks with local energy regularization. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, CIKM '21, page 1191–1201, New York, NY, USA, 2021a. Association for Computing Machinery. ISBN 9781450384469. doi: 10.1145/3459637.3482447. URL
https://doi.org/10.1145/3459637.3482447.

Y. Ma, X. Liu, T. Zhao, Y. Liu, J. Tang, and N. Shah. A unified view on graph neural networks as graph signal denoising. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, CIKM '21, page 1202–1211, New York, NY, USA, 2021b. ISBN 9781450384469. doi: 10.1145/3459637.3482225.

Y. Ma, X. Liu, N. Shah, and J. Tang. Is homophily a necessity for graph neural networks? In International Conference on Learning Representations, 2022. URL https://openreview. net/forum?id=ucASPPD9GKN.

R. Mandal, C. Casert, and P. Sollich. Robust prediction of force chains in jammed solids using graph neural networks. *Nature Communications*, 13:4424, 07 2022. doi: 10.1038/s41467-022-31732-3.

J. McAuley, C. Targett, Q. Shi, and A. van den Hengel. Image-based recommendations on styles and substitutes. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR '15, page 43–52, New York, NY, USA, 2015. Association for Computing Machinery. ISBN 9781450336215. doi: 10.1145/2766462.2767755. URL https://doi.org/10.1145/2766462.2767755.

M. McPherson, L. Smith-Lovin, and J. M. Cook. Birds of a feather: Homophily in social networks.

Annual Review of Sociology, 27:415–444, 2001. ISSN 03600572, 15452115. URL http:// www.jstor.org/stable/2678628.

R. R. Nadakuditi and M. E. Newman. Graph spectra and the detectability of community structure in networks. *Physical review letters*, 108(18):188701, 2012.

A. Ng, M. Jordan, and Y. Weiss. On spectral clustering: Analysis and an algorithm. In T. Dietterich, S. Becker, and Z. Ghahramani, editors, *Advances in Neural Information Processing Systems*, volume 14. MIT Press, 2001. URL https://proceedings.neurips.cc/paper_files/ paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf.

K. Nguyen, N. M. Hieu, V. D. Nguyen, N. Ho, S. Osher, and T. M. Nguyen. Revisiting oversmoothing and over-squashing using ollivier-ricci curvature. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of *Proceedings of Machine Learning Research*,
pages 25956–25979. PMLR, 7 2023. URL https://proceedings.mlr.press/v202/ nguyen23c.html.

L. Pan, C. Shi, and I. Dokmanic. Neural link prediction with walk pooling. In ´ International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id= CCu6RcUMwK0.

D. Paul. Asymptotics of sample eigenstructure for a large dimensional spiked covariance model.

Statistica Sinica, pages 1617–1642, 2007.

H. Pei, B. Wei, K. C.-C. Chang, Y. Lei, and B. Yang. Geom-gcn: Geometric graph convolutional networks. In *International Conference on Learning Representations*, 2020. URL https:// openreview.net/forum?id=S1e2agrFvS.

O. Platonov, D. Kuznedelev, M. Diskin, A. Babenko, and L. Prokhorenkova. A critical look at the evaluation of GNNs under heterophily: Are we really making progress? In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview. net/forum?id=tJbbQfw-5wv.

C. Qian, A. Manolache, K. Ahmed, Z. Zeng, G. V. den Broeck, M. Niepert, and C. Morris. Probabilistically rewired message-passing neural networks. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id= Tj6Wcx7gVk.

L. Rampasek, M. Galkin, V. P. Dwivedi, A. T. Luu, G. Wolf, and D. Beaini. Recipe for a general, powerful, scalable graph transformer. In A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https://openreview. net/forum?id=lMMaNf6oxKM.

Y. Rong, W. Huang, T. Xu, and J. Huang. Dropedge: Towards deep graph convolutional networks on node classification. In *International Conference on Learning Representations*, 2020. URL https://openreview.net/forum?id=Hkx1qkrKPr.

B. Rozemberczki, C. Allen, and R. Sarkar. Multi-Scale attributed node embedding. Journal of Complex Networks, 9(2):cnab014, 05 2021. ISSN 2051-1329. doi: 10.1093/comnet/cnab014.

URL https://doi.org/10.1093/comnet/cnab014.

P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Galligher, and T. Eliassi-Rad. Collective classification in network data. *AI Magazine*, 29(3):93, 9 2008. doi: 10.1609/aimag.v29i3.2157. URL https: //ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2157.

X. Shen, P. Lio, L. Yang, R. Yuan, Y. Zhang, and C. Peng. Graph rewiring and preprocessing for graph neural networks based on effective resistance. *IEEE Transactions on Knowledge and Data* Engineering, pages 1–14, 2024. doi: 10.1109/TKDE.2024.3397692.

C. Shi, L. Pan, H. Hu, and I. Dokmanic. Homophily modulates double descent generalization ´
in graph convolution networks. *Proceedings of the National Academy of Sciences*, 121(8): e2309504121, 2024. doi: 10.1073/pnas.2309504121. URL https://www.pnas.org/doi/ abs/10.1073/pnas.2309504121.

J. Shi and J. Malik. Normalized cuts and image segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(8):888–905, 2000. doi: 10.1109/34.868688.

H. Shirzad, A. Velingker, B. Venkatachalam, D. J. Sutherland, and A. K. Sinop. Exphormer: Sparse transformers for graphs. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 31613–31632. PMLR, 07 2023.

URL https://proceedings.mlr.press/v202/shirzad23a.html.

J. Tang, J. Sun, C. Wang, and Z. Yang. Social influence analysis in large-scale networks. In Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '09, page 807–816, New York, NY, USA, 2009. Association for Computing Machinery. ISBN 9781605584959. doi: 10.1145/1557019.1557108. URL https: //doi.org/10.1145/1557019.1557108.

J. Topping, F. D. Giovanni, B. P. Chamberlain, X. Dong, and M. M. Bronstein. Understanding oversquashing and bottlenecks on graphs via curvature. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=7UmjRGzp-A.

J. Wang, J. Guo, Y. Sun, J. Gao, S. Wang, Y. Yang, and B. Yin. DGNN: Decoupled Graph Neural Networks with Structural Consistency between Attribute and Graph Embedding Representations, 2024.

X. Wu, Z. Chen, W. W. Wang, and A. Jadbabaie. A non-asymptotic analysis of oversmoothing in graph neural networks. In *The Eleventh International Conference on Learning Representations*,
2023. URL https://openreview.net/forum?id=CJd-BtnwtXq.

K. Xu, W. Hu, J. Leskovec, and S. Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations, 2019. URL https://openreview.net/ forum?id=ryGs6iA5Km.

H. Yang, K. Ma, and J. Cheng. Rethinking graph regularization for graph neural networks. Proceedings of the AAAI Conference on Artificial Intelligence, 35(5):4573–4581, 05 2021. doi: 10.

1609/aaai.v35i5.16586. URL https://ojs.aaai.org/index.php/AAAI/article/ view/16586.

M. Zhang and Y. Chen. Link prediction based on graph neural networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper_files/paper/2018/
file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf.

J. Zhou, G. Cui, S. Hu, Z. Zhang, C. Yang, Z. Liu, L. Wang, C. Li, and M. Sun. Graph neural networks: A review of methods and applications. *AI Open*, 1:57–81, 2020. ISSN 2666-6510. doi: https://doi.org/10.1016/j.aiopen.2021.01.001.

J. Zhu, Y. Yan, L. Zhao, M. Heimann, L. Akoglu, and D. Koutra. Beyond homophily in graph neural networks: Current limitations and effective designs. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 7793–7804. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/ file/58ae23d878a47004366189884c2f8440-Paper.pdf.

Y. Zhu, W. Xu, J. Zhang, Y. Du, J. Zhang, Q. Liu, C. Yang, and S. Wu. A survey on graph structure learning: Progress and opportunities, 2022.

Y. Zhu, R. Amor, Y. Chen, Z. Deng, L. Feng, and M. Witbrock. Robust node classification on graph data with graph and label noise. *Proceedings of the AAAI Conference on Artificial Intelligence*, 2024.

## A Appendix

A.1 THE JDR ALGORITHM

| Algorithm 1 Joint Denoising and Rewiring 1: procedure REWIRE(X, A)   | ▷ For DENOISE just exchange X and A          |                                      |
|----------------------------------------------------------------------|----------------------------------------------|--------------------------------------|
| 2:                                                                   | X = UΣWT                                     |                                      |
| 3:                                                                   | A = V ΛV T                                   |                                      |
| 4:                                                                   | for i in range(LA) do                        | ▷ Loop over LA eigenvectors in A     |
| 5:                                                                   | va ← V [:, i]                                |                                      |
| 6:                                                                   | for j in range(LA) do                        | ▷ Loop over LA eigenvectors in X     |
| 7:                                                                   | ux ← U[:, j]                                 |                                      |
| 8:                                                                   | θ ← ⟨ux, va⟩                                 | ▷ Find angle between eigenvectors    |
| 9:                                                                   | if |θ| > |θmax| then                         |                                      |
| 10:                                                                  | θmax ← θ                                     |                                      |
| 11:                                                                  | u max x ← ux                                 |                                      |
| 12:                                                                  | end if                                       |                                      |
| 13:                                                                  | end for                                      |                                      |
| 14:                                                                  | V˜ [:, i] ← (1 − ηA)va + ηAsign(θmax)u max x | ▷ Interpolation between eigenvectors |
| 15:                                                                  | end for                                      |                                      |

1: **procedure** REWIRE(X, A) ▷ For DENOISE just exchange X and A

2: X = UΣWT

3: A = V ΛV

T

4: for i in range(LA) do ▷ Loop over LA eigenvectors in A 5: va ← V [:, i] 6: for j in range(LA) do ▷ Loop over LA eigenvectors in X 7: ux ← U[:, j] 8: θ ← ⟨ux, va⟩ ▷ Find angle between eigenvectors 9: if |θ| > |θmax| **then**

10: θmax ← θ

11: u

max

x ← ux

12: **end if** 13: **end for**

14: V˜ [:, i] ← (1 − ηA)va + ηAsign(θmax)u

max

x ▷ Interpolation between eigenvectors

15: **end for**

16: A˜ ← V˜ ΛV˜ T

17: **end procedure**

18: X˜ , A˜ ← X, A

19: for i in range(K) do ▷ Main loop

20: X′ ← DENOISE(X˜ , A˜) 21: A′ ← REWIRE(X˜ , A˜) 22: X˜ , A˜ ← X′, A′

23: **end for**

24: X˜ = UPDATE_X(X, X˜ ) ▷ Sparsify and binarize if needed 25: A˜ = UPDATE_A(A, A˜)

## A.1.1 Low-Dimensional Graphs And Relation To Resonance

low-dimensional coordinates, We finally mention that although our algorithm is motivated by the cSBM, it could have equivalently been motivated by ubiquitous lowdimensional graphs. In such graphs, node labels are related to the which are in turn given by the eigenvectors of the graph Laplacian; this is illustrated in Figure 5. If, for example, the labels are given by the sign of the first non-constant eigenfunction (the slowest-changing normal mode), our notion of alignment with L = 1 clearly remains meaningful. This also further motivates our terminology of resonance. In a purist sense, resonance is a dynamical phenomenon where driving a system with a frequency corresponding to an eigenvalue of the Laplacian yields a diverging response. Importantly, the shape of the response is then an eigenfunction. In a broad sense, resonance signifies alignment with Laplacian eigenfunctions, which are the natural modes. For graphs, this is closely related to alignment with eigenvectors of the adjacency matrix (it is equivalent for d-regular graphs). As Figure 2b shows, maximizing alignment between feature and graph spectra indeed leads to the largest response of the graph to the features.

Figure 5: Visualization of the first six eigenmodes of L of the 8×8 grid graph.

## A.1.2 Rotational Invariance Of Alignment

We show that the alignment measure (3) is invariant to rotations of the subspaces for non-unique
eigenvalues. Let A ∈ R
N×N be the adjacency matrix and X ∈ R
N×F the feature matrix. Let the
eigendecomposition of A be
$$A=\lambda_{1}U_{1}U_{1}^{T}+\cdot\cdot\cdot+\lambda_{p}U_{p}U_{p}^{T}$$
where p ≤ N, Ui ∈ R
N×si with si being the multiplicity of the eigenvalue λi,Pi
si = N,
U
T
i Ui = I, and U
T
j Ui = 0 for i ̸= j. Let similarly the SVD of X be
X = σ1V1WT
1. + *· · ·* + σqVqWT
where Vi ∈ R
N×ti, Wi ∈ R
F ×ti,Pi
ti = F, with analogous orthogonality conditions. Assume both (λi) and (σi) are sorted from largest to smallest. Assume also for simplicity that
L =Pp
′
i=1 si =Pq
′
i=1 ti so that the leading L-dimensional subspace of the graph A is spanned
$\uparrow$ . 
by the columns of the block matrix UL = [U1 *· · ·*Up′ ]. Of course it is also spanned by the columns
of UeL = [U1Q1 *· · ·*Up′Qp′ ] where invertible matrices Qi ∈ R
si×sireflect the fact that the eigensolver may return any of the infinitely many (when si > 1) orthogonal bases for the subspaces spanned by the columns of Ui. The Qi are orthogonal since UiQi are orthogonal. Similarly the
leading L-dimensional subspace of the features is spanned by the columns of VL = [V1 *· · ·* Vq
′ ] but
also of VeL = [V1R1 · · · Vq
′Rq
′ ] for any orthogonal (Ri). Now
= ∥U
T
L VL∥sp
for any choice of Qs and Rs since both block-diagonal matrices are orthogonal and the spectral
norm is unitarily-invariant.
$\widetilde{V}_{L}\|_{\rm sp}=\|{\rm blockdiag}(Q_{1}^{T},\ldots,Q_{p^{\prime}}^{T})\ U_{L}^{T}V_{L}$ blockdiag$(R_{1},\cdots,R_{q^{\prime}})\|_{\rm sp}$
$$\vert\vert{\hat{U}}_{L}^{T}\rangle$$

$\parallel$ . 

## A.1.3 Computational Complexity

The complexity of JDR results mainly from the SVD and eigendecomposition, which is of order O(F Nmin(*F, N*)) for SVD and O(N3) for the eigendecomposition (F = N). Since we only need the leading k eigenvectors this reduces to O(*F N k*). If the matrix is additionally sparse as is often the case for real-world graphs with binary node features this reduces further to O(nnz(A)k) where nnz(A) is the number of non-zero elements in A. Since usually neither the average degree d nor k is scaled by N, the complexity actually scales with O(N).

## A.1.4 Runtime Comparison

In addition to computational complexity, we report measurements of running time of the different algorithms. We run JDR and baseline methods on the real-world datasets, using GCN as downstream model. All algorithms are run on Nvidia A100 with 80GB and we time their Python processes. We emphasize that we did not explicitly optimize the timing code and we kept the outputs and logging turned on. But this influences all methods in the same way so the relative comparisons are meaningful. The results in Table 5 do not show a clear "winner". The ambiguity is especially visible on the large heterophilic graphs, where JDR is slower than DIGL, but significantly faster than FoSR on two datasets. On the Twitch-gamers dataset, on the other hand, it is faster than DIGL but not as fast as FoSR. The main reason for this is that different hyperparameters choices of the rewiring methods lead to dramatically different run times (even when applying the same method on the same dataset). For example on Computers, JDR is very fast since it only requires 3 denoising iterations, compared to 15 on Citeseer. The same holds true for FoSR which only require 5 iterations on Twitch-gamers, but 700 on Questions. So if one wants to optimize for speed, one should constrain the hyperparameters of the methods that significantly impact execution speed. Of course, there is a trade-off between any such constraint and accuracy, as our experiments on the denoising iterations of JDR in Figure 16 (b) and Figure 17 (b) in Appendix A.6 show.

## A.2 Contextual Stochastic Block Models

SBMs and GMMs are landmark theoretical models for studying clustering, classification problems and developing algorithmic tools. The cSBM (Deshpande et al., 2018), a combination of the two, Table 5: Timing experiments **in seconds** for different rewiring methods using GCN as downstream GNN. Smaller is better. We record the time of the preprocessing and training and evaluating the GNN on 100 random splits. The results do not show a clear winner; JDR generally requires a comparable or less time compared to baselines. For more discussion see A.1.4.

Dataset Base DIGL FoSR BORF JDR Cora 182 228 187 201 246 Citeseer 258 360 291 290 433 PubMed 291 692 416 897 858

Computers 274 444 516 718 465

Photo 213 299 220 801 330 Chameleon 372 239 396 483 545 Squirrel 1263 302 1282 - 1659 Actor 166 286 171 225 319

Texas 163 200 169 174 203

Cornell 167 202 164 240 208 Questions 93 804 11053 - 3707 Penn94 164 232 3198 - 1779 Twitch-gamers 1579 6729 1618 - 5165

has become a key model for studying node classification problems on graphs, inspiring numerous designs of GNNs like GPRGNN (Chien et al., 2021), GIANT (Chien et al., 2022) or ASGC (Chanpuriya and Musco, 2022). Many theoretical studies of node-level GNN problems are based on the cSBM, e.g. on double descent (Shi et al., 2024), neural collapse (Kothapalli et al., 2023), OOD generalization (Baranwal et al., 2021), or oversmoothing (Wu et al., 2023). Beyond being a standard synthetic benchmark, the cSBM is also used to to verify hypotheses about GNNs (Ma et al., 2022; Luan et al., 2023). As for any model, the cSBM also comes with limitations. One possible limitation of our work is that cSBM assumes that the features are linear as in a GMM, which makes a linear classifier optimal. If the class boundaries are highly nonlinear, this is no longer true, and the spectrum of X may need to be "linearized", e.g. via Laplacian eigenmaps or diffusion maps. Still, the results on real-world data show that the cSBM model is already highly transferable, suggesting that the high-dimensional features in real-world graph datasets are often quite linear.

## A.3 Proof Of Proposition 1

Notation. We order the eigenvalues and singular values from largest to smallest and denote the eigenvector associated with the eigenvalue λj by vj (Ac). For the leading eigenvalue and eigenvectors of Ac, we write λ1 = λA and v1 (Ac) = vA. We use analogous notation for the singular values and corresponding singular vectors of X. For simplicity and without loss of generality we assume that the angles between these vectors are accute, i.e., ⟨vA,uX⟩,⟨y˜, vA⟩,⟨y˜,uX⟩ ≥ 0.

Proof. When λ > 1, based on the Baik-Ben Arous-Péché (BBP) transition (Baik et al., 2005; Paul, 2007), the leading eigenvalue of A lies outside the spectral bulk,

$$\lambda_{A}=\lambda+\frac{1}{\lambda}+\mathcal{O}_{p}\left(\frac{1}{\sqrt{N}}\right),$$

and the fluctuation of the leading eigenvector satisfies

$$\mathbf{q}_{A}:=\lambda\left(\mathbf{v}_{A}-\sqrt{1-\frac{1}{\lambda^{2}}\tilde{\mathbf{y}}}\right)\stackrel{{d}}{{\longrightarrow}}\mbox{Haar}(\mathbb{S}_{\tilde{\mathbf{y}}^{\perp}}^{N-2})\tag{5}$$

where Haar(S
N−2 y˜⊥ ) is the uniform distribution on the sphere orthogonal to y˜, S
n−2 y˜⊥ =
v : v ∈ R
N | v T y˜ = 0, ∥v∥ = 1	, and the convergence is in distribution as N → ∞. Similarly,