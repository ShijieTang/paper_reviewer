# Beyond Random Masking: When Dropout Meets Graph Convolutional Networks

Yuankai Luo<sup>1</sup>,<sup>2</sup> & Xiao-Ming Wu<sup>2</sup> & Hao Zhu<sup>3</sup>, ∗

<sup>1</sup>Beihang University, Beijing, China

<sup>2</sup>The Hong Kong Polytechnic University, Hong Kong

<sup>3</sup>Data61rCSIRO, Sydney, Australia

#### Abstract

Graph Convolutional Networks (GCNs) have emerged as powerful tools for learning on graph-structured data, yet the behavior of dropout in these models remains poorly understood. This paper presents a comprehensive theoretical analysis of dropout in GCNs, revealing that its primary role differs fundamentally from standard neural networks - preventing oversmoothing rather than co-adaptation. We demonstrate that dropout in GCNs creates dimension-specific stochastic subgraphs, leading to a form of structural regularization not present in standard neural networks. Our analysis shows that dropout effects are inherently degreedependent, resulting in adaptive regularization that considers the topological importance of nodes. We provide new insights into dropout's role in mitigating oversmoothing and derive novel generalization bounds that account for graphspecific dropout effects. Furthermore, we analyze the synergistic interaction between dropout and batch normalization in GCNs, uncovering a mechanism that enhances overall regularization. Our theoretical findings are validated through extensive experiments on both node-level and graph-level tasks across 14 datasets. Notably, GCN with dropout and batch normalization outperforms state-of-the-art methods on several benchmarks, demonstrating the practical impact of our theoretical insights.

# 1 Introduction

The remarkable success of deep neural networks across various domains has been accompanied by the persistent challenge of overfitting, where models perform well on training data but fail to generalize to unseen examples. This issue has spurred the development of numerous regularization techniques, among which dropout has emerged as a particularly effective and widely adopted approach [\(LeCun et al., 2015\)](#page-12-0). Introduced by [Srivastava et al.](#page-13-0) [\(2014\)](#page-13-0), dropout addresses overfitting by randomly "dropping out" a proportion of neurons during training, effectively creating an ensemble of subnetworks. This technique has proven highly successful in improving generalization and has become a standard tool in the deep learning toolkit. The effectiveness of dropout has prompted extensive theoretical analysis, with various perspectives offered to explain its regularization effects.

Some researchers have interpreted dropout as a form of model averaging [\(Baldi & Sadowski, 2013\)](#page-10-0), while others have analyzed it through the lens of information theory [\(Achille & Soatto, 2018\)](#page-10-1). [Wager](#page-13-1) [et al.](#page-13-1) [\(2013\)](#page-13-1) provided insights into dropout's adaptive regularization properties, and [Gal & Ghahra](#page-11-0)[mani](#page-11-0) [\(2016\)](#page-11-0) established connections between dropout and Bayesian inference. These diverse theoretical frameworks have significantly enhanced our understanding of dropout's role in mitigating overfitting in traditional neural networks. However, as the field of deep learning has expanded to encompass more complex data structures, particularly graphs, new questions have arisen regarding the applicability and behavior of established techniques. Graph Neural Networks (GNNs), especially Graph Convolutional Networks (GCNs), have demonstrated remarkable performance on tasks involving graph-structured data [\(Kipf & Welling, 2017\)](#page-11-1). Naturally, researchers and practitioners have applied dropout to GNNs, often observing beneficial effects on generalization [\(Hamilton et al.,](#page-11-2) [2017\)](#page-11-2).

<sup>∗</sup>Hao Zhu is the corresponding author and led the writing of the paper.

While dropout was originally designed to prevent co-adaptation of features in standard neural networks, our analysis reveals that its primary mechanism in GCNs is fundamentally different. We demonstrate that dropout's main contribution in GCNs is mitigating oversmoothing by maintaining feature diversity across nodes, rather than preventing co-adaptation as in standard neural networks. This finding represents a significant shift in our understanding of how regularization operates in graph neural networks. Specifically, we demonstrate that:

- Dropout in GCNs creates dimension-specific stochastic sub-graphs, leading to a unique form of structural regularization not present in standard neural networks.
- The effects of dropout are inherently degree-dependent, with differential impacts on nodes based on their connectivity, resulting in adaptive regularization that considers the topological importance of nodes in the graph.
- Dropout plays a crucial role in mitigating the oversmoothing problem rather than coadaption in GCNs, though its effects are more nuanced than previously thought.
- The generalization bounds for GCNs with dropout exhibit a complex dependence on graph properties, diverging from traditional dropout theory.
- There exists a significant interplay between dropout and batch normalization in GCNs, revealing synergistic effects that enhance the overall regularization.

Our theoretical framework not only provides deeper insights into the mechanics of dropout in graphstructured data but also yields practical implications for the design and training of GCNs. We validate our theoretical findings through extensive experiments on both node-level and graph-level tasks, demonstrating the practical relevance of our analysis. This work bridges a critical gap in the theoretical understanding of regularization in GCNs and paves the way for more principled approaches to leveraging dropout in graph representation learning. Furthermore, we validate our theoretical findings through extensive experiments, demonstrating that GCNs incorporating our insights on dropout and batch normalization outperform several state-of-the-art methods on benchmark datasets. This practical success underscores the importance of our theoretical contributions and their potential to advance the field of graph representation learning.

# 2 Related Work

Dropout in Neural Networks. Overfitting can be reduced by using dropout [Hinton et al.](#page-11-3) [\(2012\)](#page-11-3) to prevent complex co-adaptations on the training data. Since its inception, several variants have been proposed to enhance its effectiveness. DropConnect [\(Wan et al., 2013\)](#page-13-2) generalizes dropout by randomly dropping connections rather than nodes. Gaussian dropout [Srivastava et al.](#page-13-0) [\(2014\)](#page-13-0) replaces the Bernoulli distribution with a Gaussian one for smoother regularization. Curriculum dropout [\(Morerio et al., 2017\)](#page-12-1) adaptively adjusts the dropout rate during training. Theoretical interpretations of dropout have provided insights into its success. The model averaging perspective [\(Baldi](#page-10-0) [& Sadowski, 2013\)](#page-10-0) views dropout as an efficient way of approximately combining exponentially many different neural networks. The adaptive regularization interpretation [\(Wager et al., 2013\)](#page-13-1) shows how dropout adjusts the regularization strength for each feature based on its importance. The Bayesian approximation view [\(Gal & Ghahramani, 2016\)](#page-11-0) connects dropout to variational inference in Bayesian neural networks, providing a probabilistic framework for understanding its effects.

Regularization in Graph Neural Networks. Graph Neural Networks (GNNs), while powerful, are prone to overfitting and over-smoothing [\(Li et al., 2018\)](#page-12-2). Various regularization techniques [\(Yang](#page-13-3) [et al., 2021;](#page-13-3) [Rong et al., 2020;](#page-13-4) [Fang et al., 2023;](#page-11-4) [Feng et al., 2020\)](#page-11-5) have been proposed to address these issues. DropEdge [\(Rong et al., 2020\)](#page-13-4) randomly removes edges from the input graph during training, reducing over-smoothing and improving generalization. Graph diffusion-based methods [\(Gasteiger et al., 2019\)](#page-11-6) incorporate higher-order neighborhood information to enhance model robustness. Spectral-based approaches [\(Wu et al., 2019\)](#page-13-5) leverage the graph spectrum to design effective regularization strategies. Empirical studies have shown that traditional dropout can be effective in GNNs [\(Hamilton et al., 2017\)](#page-11-2), but its interaction with graph structure remains poorly understood. Some works have proposed adaptive dropout strategies for GNNs [\(Gao & Ji, 2019\)](#page-11-7), but these are primarily heuristic approaches without comprehensive theoretical grounding.

Theoretical Frameworks for GNNs. Despite the empirical success of Graph Neural Networks (GNNs), establishing theories to explain their behaviors is still an evolving field. Recent works have made significant progress in understanding over-smoothing [\(Li et al., 2018;](#page-12-2) [Zhao & Akoglu, 2019;](#page-14-0) [Oono & Suzuki, 2019;](#page-12-3) [Rong et al., 2020\)](#page-13-4), interpretability [\(Ying et al., 2019;](#page-14-1) [Luo et al., 2020;](#page-12-4) [Vu](#page-13-6) [& Thai, 2020;](#page-13-6) [Yuan et al., 2020;](#page-14-2) [2021\)](#page-14-3), expressiveness [\(Xu et al., 2018;](#page-13-7) [Chen et al., 2019;](#page-10-2) [Maron](#page-12-5) [et al., 2018;](#page-12-5) [Dehmamy et al., 2019;](#page-10-3) [Feng et al., 2022\)](#page-11-8), and generalization [\(Scarselli et al., 2018;](#page-13-8) [Du](#page-10-4) [et al., 2019;](#page-10-4) [Verma & Zhang, 2019;](#page-13-9) [Garg et al., 2020;](#page-11-9) [Zhang et al., 2020;](#page-14-4) [Oono & Suzuki, 2019;](#page-12-3) [Lv, 2021;](#page-12-6) [Liao et al., 2020;](#page-12-7) [Esser et al., 2021;](#page-11-10) [Cong et al., 2021\)](#page-10-5). Our work aims to complement these existing theoretical frameworks by focusing on the practical aspects of dropout in GNNs, a widely used regularization technique that has not been thoroughly examined from a theoretical perspective. Previous works have provided valuable insights using classical techniques such as Vapnik-Chervonenkis dimension [\(Scarselli et al., 2018\)](#page-13-8), Rademacher complexity [\(Lv, 2021;](#page-12-6) [Garg](#page-11-9) [et al., 2020\)](#page-11-9), and algorithm stability [\(Verma & Zhang, 2019\)](#page-13-9). Recent efforts [\(Oono & Suzuki,](#page-12-3) [2019;](#page-12-3) [Esser et al., 2021\)](#page-11-10) have also made strides in incorporating the transductive learning schema of GNNs into theoretical analyses. We bridge the gap between theoretical understanding and practical implementation of GNNs, offering insights into how dropout affects generalization and performance in graph-structured learning tasks.

# 3 Theoretical Framework

In this section, we develop a rigorous mathematical framework to analyze the behavior of dropout in Graph Convolutional Networks (GCNs). We begin by establishing notations and definitions, then formalize the GCN model with dropout, and finally introduce key concepts that will be central to our analysis.

#### 3.1 Notations and Definitions

Notations. Let G = (V, E, X) be an undirected graph with *n* = |V| nodes and *m* = |E| edges, where X ∈ R *n*×*d*<sup>0</sup> represents the node feature matrix with *d*<sup>0</sup> input features per node. We denote by A ∈ R *n*×*n* the adjacency matrix, D = diag(*deg*1, . . . , *degn*) the degree matrix where *deg<sup>i</sup>* = P *<sup>j</sup> Ai j*, and A˜ = D<sup>−</sup> <sup>2</sup> AD<sup>−</sup> <sup>2</sup> the normalized adjacency matrix.

Graph Convolutional Networks (GCNs). An L-layer GCN performs the following layer-wise transformation:

$$H^{(l)} = \sigma(\tilde{A}H^{(l-1)}W^{(l)}), \quad (1)$$

where H(*l*) ∈ <sup>R</sup> *n*×*d<sup>l</sup>* is the feature matrix, W(*l*) ∈ <sup>R</sup> *dl*−1×*d<sup>l</sup>* is the weight matrix, σ(·) is a non-linear activation, and H(0) = X. The feature energy measures representation smoothness:

$$E(\mathbf{H}^{(l)}) = \frac{1}{2|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} \|\mathbf{h}_i^{(l)} - \mathbf{h}_j^{(l)}\|_2^2 \quad (2)$$

Dropout in GCNs. For layer *l*, dropout applies a random mask M(*l*) ∈ <sup>R</sup> *<sup>n</sup>*×*d<sup>l</sup>* where each element *M* (*l*) *i j* is drawn independently from Bernoulli(1 − *p*). The forward pass with dropout is defined as:

$$H^{(l)} = \frac{1}{1-p} M^{(l)} \odot \sigma(\tilde{A}H^{(l-1)}W^{(l)}), \quad (3)$$

where ⊙ denotes element-wise multiplication and *p* is the dropout probability.

Batch Normalization. When incorporating batch normalization, the layer transformation becomes:

$$H^{(l)} = \sigma(\text{BN}(\tilde{A}H^{(l-1)}W^{(l)})), \quad (4)$$

where BN applies feature-wise normalization BN(X) <sup>=</sup> <sup>γ</sup> <sup>⊙</sup> √ X−µ*<sup>B</sup>* σ *B* +ϵ + β with learnable parameters γ, β and batch statistics <sup>µ</sup>*B*, σ<sup>2</sup> *B* .

![](_page_3_Diagram_1.jpeg)

Figure 1: Illustration of how dropout creates dimension-specific sub-graphs. From left to right: the original graph with complete feature vectors, the graph after applying dropout (where *x* indicates dropped features), and the resulting sub-graphs for each feature dimension. Different colors indicate different feature dimensions, and grayed-out nodes show where features are dropped, preventing message passing along those paths in the next convolution.

#### 3.2 Dimension-Specific Stochastic Sub-graphs

We demonstrate how dropout creates dimension-specific sub-graphs in Figure [1.](#page-3-0) At each iteration *t*, dropout induces dimension-specific stochastic sub-graphs G (*l*, *j*) *<sup>t</sup>* = (V, E (*l*, *j*) *t* ) with:

$$\mathcal{E}_t^{(l,j)} = \{(u, v) \in \mathcal{E} \mid M_{uj}^{(l,t)} \neq 0 \text{ and } M_{vj}^{(l,t)} \neq 0\}. \quad (5)$$

The coupling between feature dropout and graph topology is captured by the feature-topology coupling matrix:

$$\mathcal{C}_t^{(l)} = \boldsymbol{A} \odot \mathbb{1}[(\boldsymbol{M}^{(l,t)}(\boldsymbol{M}^{(l,t)})^\top) > 0], \quad (6)$$

which measures how dropout simultaneously affects connected nodes' features. This interaction manifests in each node's effective degree:

$$\deg_{i,t}^{\text{eff}} = |\{j \in \mathcal{N}(i) : \exists k, M_{ik}^{(l,t)} \neq 0 \text{ and } M_{jk}^{(l,t)} \neq 0\}| = \sum_j (C_t^{(l)})_{ij}, \quad (7)$$

representing the actual count of node i's neighbors that maintain feature connections after dropout. We consider a path P = (*v*0, . . . , *vk*) active for feature *j* when all nodes along the path retain this feature, i.e., Q*<sup>k</sup>*−<sup>1</sup> *<sup>i</sup>*=<sup>0</sup> *M* (*l*,*t*) *vi <sup>j</sup> M* (*l*,*t*) *vi*+<sup>1</sup> *j* , 0. To elucidate the specific impact of dropout on embedding features, we introduce these concepts:

Theorem 1 (Sub-graph Diversity). *The expected number of distinct sub-graphs per iteration is:*

$$\mathbb{E}[|\mathcal{E}_t^{(l,j)} | \ j = 1, \dots, d_l]] = d_l(1 - (1 - p)^{2^{|\mathcal{E}_l|}}),$$

*where d<sup>l</sup> is the number of features at layer l, p is the dropout probability, and* |E| *is the number of edges in the original graph (The complete proof is in the Appendix. [A.1\)](#page-14-5).*

This theorem reveals that dropout in GCNs leads to a rich set of sub-graphs, providing a form of structural data augmentation unique to graph-based models. The diversity of these sub-graphs increases with both the dropout probability *p* and the number of features *d<sup>l</sup>* . This suggests that higher-dimensional GCNs with moderate dropout rates can benefit from a wider range of structural variations during training, potentially leading to more robust and generalizable representations. Moreover, this mechanism allows the GCN to implicitly explore different graph structures without explicitly modifying the input graph. This could be particularly beneficial for tasks where the optimal graph structure is uncertain or where multiple relevant sub-structures exist within the data.

Theorem 2 (Expected Active Features per Path). *For a path* P *of length k, the expected number of features for which it is active is:*

$$\mathbb{E}[\text{active features for } \mathcal{P}] = d_l(1 - p)^{k+1}.$$

This theorem demonstrates that while individual long paths are unlikely to be active for any given feature, the multi-dimensional nature of GCNs allows for effective long-range information flow through the ensemble effect across features. This theoretical insight is further supported by our empirical analysis in Appendix [A.6.](#page-18-0)

#### 3.3 Degree-Dependent Nature of Dropout Effects

The interaction between dropout and the graph structure leads to a form of degree-dependent regularization in GCNs. This means that the effect of dropout varies based on the connectivity of each node, creating an adaptive regularization scheme that considers the topological importance of nodes in the graph.

Theorem 3 (Degree-Dependent Dropout Effect). *The expected e*ff*ective degree and its variance are given by:*

| $\mathbb{E}[\deg_{i,t}] = (1 - p)^2 \deg_i$ | $\text{and } \text{Var}[\deg_{i,t}] = \deg_i(1 - p)^2(1 - (1 - p)^2)$ , | (8) |
|---------------------------------------------|-------------------------------------------------------------------------|-----|
|---------------------------------------------|-------------------------------------------------------------------------|-----|

*where deg<sup>i</sup> is the original degree of node i and p is the dropout probability.*

This theorem highlights that dropout affects nodes differentially depending on their degree. Highdegree nodes, typically more influential within the graph, exhibit less variation in their effective degree due to dropout, potentially resulting in more stable representations for these important nodes. This observation is empirically confirmed in the analysis of a 2-layer GCN presented in Appendix [A.6.](#page-18-0) Consequently, the degree-dependent nature of dropout in GCNs results in adaptive regularization, where the regularization effect naturally adjusts to the local graph structure.

Corollary 4 (Relative Stability of High-Degree Nodes). *The coe*ffi*cient of variation of the e*ff*ective degree, defined as CV*[*deg<sup>e</sup>*ff *i*,*t* ] = q *Var*[*deg<sup>e</sup>*ff *i*,*t* ]/E[*deg<sup>e</sup>*ff *i*,*t* ]*, decreases with increasing node degree:*

$$CV[\deg_{i,t}^{eff}] = \frac{\sqrt{1 - (1 - p)^2}}{\sqrt{\deg_i(1 - p)}}.$$

This corollary further confirms that high-degree nodes experience relatively less variation in their effective degree due to dropout. Figure [10](#page-19-0) illustrates that the CV decreases as node degree increases. This degree-dependent effect distinguishes dropout in GCNs from its application in standard neural networks and suggests that the optimal dropout strategy for GCNs may need to consider the graph structure explicitly.

#### 3.4 Role of Dropout in Oversmoothing

Oversmoothing is a well-known issue in GCNs, where node representations become indistinguishable as the number of layers increases. Our analysis reveals that dropout plays a crucial role in this context, though its effects are more nuanced than previously thought.

Theorem 5 (Dropout and Feature Energy). *For a GCN with dropout probability p, the expected feature energy at layer l is bounded by:*

$$\mathbb{E}[E(\mathbf{H}^{(l)})] \leq \frac{deg_{\max}}{|\mathcal{E}|} \left( \frac{1}{1-p} \right)' \|\tilde{\mathbf{A}}\|_2^{2l} \prod_{i=1}^l \|\mathbf{W}^{(i)}\|_2^2 \|\mathbf{X}\|_F^2 \quad (9)$$

*where E*(X) *is the energy of the input features and* W(*i*) *are the weight matrices (The complete proof is in the Appendix[.A.2\)](#page-15-0).*

The derived bound demonstrates how dropout affects feature energy through the interplay of network depth (*l*), graph structure (through *deg*max and A˜), and weight properties (∥W(*i*) 2 2 ). Note that this analysis only provides an upper bound; the absence of a lower bound in this derivation is due to limitations in bounding certain terms. We will later show that when considering batch normalization, we can establish the existence of a lower bound, providing a more complete characterization.

# 3.5 Generalization Bounds with Graph-Specific Dropout Effects

The unique properties of dropout in GCNs, such as the creation of stochastic sub-graphs and degreedependent effects, influence how these models generalize to unseen data. Our analysis provides novel generalization bounds that explicitly account for these graph-specific dropout effects, offering insights into how dropout interacts with graph structure to influence the model's generalization capabilities.

![](_page_5_Figure_1.jpeg)

Figure 2: Feature energy vs dropout rates. Figure 3: BN feature energy vs dropout rates.

Theorem 6 (Generalization Bound for *L*-Layer GCN with Dropout). *For an L-layer GCN F with dropout probability p<sup>l</sup> at layer l and L*σ*-Lipschitz activation function* <sup>σ</sup>*, with probability at least* 1 − δ *over the training examples, the following generalization bound holds:*

$$\mathbb{E}_D[L(F(X))] - \mathbb{E}_S[L(F(X))] \leq O\left(\sqrt{\frac{\log(1/\delta)}{n}}\right) \sum_{l=1}^L L_{loss} \cdot L_l \cdot \sqrt{\frac{p_l}{(1-p_l)\chi_f(\mathcal{G})}} \|\sigma(\tilde{\mathbf{A}}\mathbf{H}^{(l-1)}\mathbf{W}^{(l)})\|_F, \quad (10)$$

*where* <sup>E</sup>*<sup>D</sup> is the expectation over the data distribution,* <sup>E</sup>*<sup>S</sup> is the expectation over the training samples, L is the loss function with Lipschitz constant Lloss, L<sup>l</sup>* = Q*<sup>L</sup> i*=*l* (*L*σ∥W(*i*) <sup>∥</sup><sup>2</sup> · ∥A˜ <sup>∥</sup>2) *is the Lipschitz constant from layer l to output,* ∥W(*i*) <sup>∥</sup><sup>2</sup> *is the spectral norm of the weight matrix at layer i,* <sup>∥</sup>A˜ <sup>∥</sup><sup>2</sup> *is the spectral norm of the normalized adjacency matrix, and* <sup>χ</sup>*f*(G) *is the fractional chromatic number of the dependency graph* G *induced by the message passing structure.*

This generalization bound reveals how the network's stability depends on the loss function's Lipschitz constant, layer-wise Lipschitz constants capturing weight effects, graph structure through <sup>χ</sup>*f*(G), feature activations, and dropout rates. This leads to several key insights: First, network depth affects stability through the layer-wise Lipschitz constants *L<sup>l</sup>* . The multiplicative accumulation of weight and graph effects (Q*<sup>L</sup> i*=*l*+1 ∥W(*i*) ∥∥A˜ ∥) suggests deeper GCNs require careful regularization as perturbations can amplify through layers. Second, the graph structure fundamentally influences stability through <sup>χ</sup>*f*(G). Since <sup>χ</sup>*f*(G) > 1 for GCNs due to message passing (versus <sup>χ</sup>*f*(G) = 1 for MLPs), GCNs gain natural regularization from their graph structure. This effect strengthens with graph connectivity since larger <sup>χ</sup>*f*(G) leads to better stability. Combined with the fact that the normalized adjacency matrix has bounded spectral norm (∥A˜ <sup>∥</sup><sup>2</sup> <sup>≤</sup> 1), this provides a built-in stabilizing mechanism unique to GNNs. Third, examining layer-specific terms reveals the interplay between weights ∥W(*l*) ∥, feature magnitudes ∥σ(AH˜ (*l*−1)W(*l*) )∥*F*, and dropout rates *p<sup>l</sup>* . The contribution of each layer to the overall bound suggests that adaptive layer-wise dropout rates might be more effective than uniform dropout, particularly when certain layers process more critical features. Finally, the bound mathematically explains the dropout rate trade-off through the term p *pl*/((1 − *pl*)χ*f*(G)). Higher dropout provides stronger regularization but increases noise, while the graph structure (through <sup>χ</sup>*f*(G)) moderates this effect. This helps explain why moderate dropout rates often work best in practice, with the optimal rate depending on the graph's connectivity patterns. This theoretical insight aligns with empirical observations that GNNs often benefit more from dropout than MLPs, as the graph structure provides additional stability through <sup>χ</sup>*f*(G) while allowing effective information flow via message passing.

### 3.6 Interaction of Dropout and Batch Normalization in GCN<sup>s</sup>

While dropout provides a powerful regularization mechanism for GCNs, its degree-dependent nature can lead to uneven regularization across nodes. Batch Normalization (BN) offers a complementary approach that can potentially address this issue and enhance the benefits of dropout. Our analysis reveals how the combination of dropout and BN creates a synergistic regularization effect that is sensitive to both graph structure and feature distributions.

Theorem 7 (Layer-wise Energy Lower Bound for GCN with Dropout and BN). *For an L-layer Graph Convolutional Network with dropout rate p, batch normalization parameters* {β (*l*) *d* , γ (*l*) *d dl d*=1 *at* *each layer l, with probability at least* (1 − δ) *L , the expected feature energy at each layer l satisfies:*

$$E(H^{(l)}) \geq \frac{p \cdot d e g_{\min}}{2|\mathcal{E}|(1-p)} \sum_{d=1}^{d_l} \Phi(\beta_d^{(l)}/\gamma_d^{(l)}) \cdot (\beta_d^{(l)})^2$$

*where l* = 1, 2, ..., *L indicates the layer, deg*min *is the minimum degree in the graph,* |E| *is the total number of edges,* Φ *is the standard normal CDF and* β (*l*) *d* , γ (*l*) *d are the BN parameters for dimension d at layer l (The complete proof is in the Appendix[.A.4\)](#page-16-0).*

Our theoretical analysis reveals a crucial interplay between dropout and batch normalization in GCNs. The lower bound on feature energy combines three essential components: (1) A graph structural term *degmin* <sup>2</sup>|E| that captures the network connectivity, (2) A dropout-induced scaling factor *<sup>p</sup>* 1−*p* that amplifies preserved features, and (3) A BN-controlled feature activation term P*<sup>d</sup><sup>l</sup> <sup>d</sup>*=<sup>1</sup> Φ(β *d* /γ(*l*) *d* )·(β (*l*) *d* ) 2 that establishes a non-zero energy floor. This interaction operates through several key mechanisms: (1) The BN shift parameters β (*l*) *d* directly contribute to feature energy through their squared magnitude, while the ratio β (*l*) *d* /γ(*l*) *d* determines the proportion of features preserved through ReLU activation via the standard normal CDF Φ. Higher positive values of this ratio increase feature preservation. (2) Dropout's *<sup>p</sup>* 1−*p* factor enhances this feature preservation effect, creating a controlled amplification that prevents feature collapse. This amplification is naturally weighted by graph connectivity, with minimum degree *degmin* ensuring baseline protection even for sparsely connected nodes. (3) The entire bound scales with the graph's minimum degree, illustrating how the mechanism adapts to the underlying graph structure, providing stronger guarantees for more densely connected graphs. This theoretical framework explains our empirical observations in Figures [2](#page-5-0) & [3,](#page-5-0) where batch normalization effectively moderates the energy dynamics in GCNs. By establishing a non-zero lower bound on feature energy, BN prevents complete feature collapse regardless of weight updates, while dropout enhances feature discrimination. Their joint application creates a specialized regularization mechanism for graph-structured data, where BN's parameter-controlled feature preservation interacts with dropout-induced sparsity to maintain robust node representations across graph topologies.

#### 3.7 Comparison with Other Dropout Variants

Various dropout mechanisms have been proposed for GNNs, each applying masks at different stages of message passing. We formally characterize these variants through their masking operations and their effects in Table [1.](#page-6-0) The key distinction of standard dropout lies in its feature-dimension-specific masking, which creates unique sub-graph structures for each feature dimension. This leads to a quadratic effect on the effective degree, providing stronger regularization than other variants. While DropNode and DropEdge apply coarse-grained masks uniformly across features, and DropMessage operates at the message level, dropout's feature-specific approach provides finer-grained control over information flow.

Table 1: Comparison of different dropout variants in GNNs. Each method is characterized by its masking operation M*d*, the resulting sub-graph formation G*<sup>t</sup>* , and expected effective degree <sup>E</sup>[*deg*eff *i*,*t* ], where *p* is the dropout probability.

| Method      | Masking |   |       | Operation |      |      |       |     |   |     |   |    |    |   |       |   | Sub-graph Formation Expected E ff ective Degree                          |
|-------------|---------|---|-------|-----------|------|------|-------|-----|---|-----|---|----|----|---|-------|---|--------------------------------------------------------------------------|
| DropNode    | M       | d | = A   | ˜((       | M    | node | ⊙     | H   |   | ( l | − | 1) | )  | W | (     | l | )                                                                        |
|             |         |   |       |           |      |      |       |     |   |     |   |    |    |   |       |   | ) d G t = ( V \ V dropped , E \ { ( i , j )   i ∈ V dropped } ) deg i    |
|             |         |   |       |           |      |      |       |     |   |     |   |    |    |   |       |   | j ∈N ( i ) (1 − p )                                                      |
| DropEdge    | M       | d | = ( M |           | edge | ⊙    | A ˜)( |     | H | (   | l | −  | 1) | W | (     | l | )                                                                        |
|             |         |   |       |           |      |      |       |     |   |     |   |    |    |   |       |   | ) d G t = ( V , E \ E dropped ) (1 − p ) deg i                           |
| DropMessage | M       | d | = A   | ˜( M      | msg  | d    | ⊙     | ( H |   | ( l | − | 1) |    | W | ( l   | ) |                                                                          |
|             |         |   |       |           |      |      |       |     |   |     |   |    |    |   |       |   | )) d G                                                                   |
|             |         |   |       |           |      |      |       |     |   |     |   |    |    |   |       |   | t = ( V , { ( i , j ) ∈ E  M msg di j , 0 } ) (1 − p ) deg i             |
| Dropout     | M       | d | = M   | f eat     | d    | ⊙    | A ˜(  | H   | ( | l − |   | 1) | W  |   | ( l ) |   |                                                                          |
|             |         |   |       |           |      |      |       |     |   |     |   |    |    |   |       | ) | d G                                                                      |
|             |         |   |       |           |      |      |       |     |   |     |   |    |    |   |       |   | t = ( V , { ( i , j ) ∈ E  M f eat di , 0 , M f eat d j , 0 } ) (1 − p ) |
|             |         |   |       |           |      |      |       |     |   |     |   |    |    |   |       |   | 2 deg i                                                                  |

### 4 Experiments

To validate our theoretical analysis, we conducted extensive experiments on a variety of datasets, considering both node-level and graph-level tasks. We implemented dropout technique on several popular GNN architectures: GCN [\(Kipf & Welling, 2017\)](#page-11-1), GraphSAGE [\(Hamilton et al., 2017\)](#page-11-2), GAT [\(Velickovi](#page-13-10) ˇ c et al., [2018\)](#page-13-10), and GatedGCN [\(Bresson & Laurent, 2017\)](#page-10-6). For each model, we ´ compared the performance with and without dropout. Our code is available at [https://github.](https://github.com/LUOyk1999/dropout-theory) [com/LUOyk1999/dropout-theory](https://github.com/LUOyk1999/dropout-theory).

Table 2: Node classification results (%). The baseline results are taken from [Deng et al.](#page-10-7) [\(2024\)](#page-10-7); [Wu](#page-13-11) [et al.](#page-13-11) [\(2023\)](#page-13-11). The top 1 st , 2 nd and 3 rd results are highlighted. "dp" denotes dropout.

|            |               | Cora   |          | CiteSeer |          | PubMed |          | Computer |          | Photo  |          | CS     |          | Physics |          | WikiCS |           | ogbn-arxiv |            |        |
|------------|---------------|--------|----------|----------|----------|--------|----------|----------|----------|--------|----------|--------|----------|---------|----------|--------|-----------|------------|------------|--------|
| # nodes    | 2,708         |        | 3,327    |          | 19,717   |        | 13,752   |          | 7,650    |        | 18,333   |        | 34,493   |         | 11,701   |        | 169,343   |            | 2,449,029  |        |
| # edges    | 5,278         |        | 4,732    |          | 44,324   |        | 245,861  |          | 119,081  |        | 81,894   |        | 247,962  |         | 216,123  |        | 1,166,243 |            | 61,859,140 |        |
| Metric     | Accuracy      | ↑      | Accuracy | ↑        | Accuracy | ↑      | Accuracy | ↑        | Accuracy | ↑      | Accuracy | ↑      | Accuracy | ↑       | Accuracy | ↑      | Accuracy  | ↑          | Accuracy   | ↑      |
| GCNII      | 85.19         | ± 0.26 | 73.20    | ± 0.83   | 80.32    | ± 0.44 | 91.04    | ± 0.41   | 94.30    | ± 0.20 | 92.22    | ± 0.14 | 95.97    | ± 0.11  | 78.68    | ± 0.55 | 72.74     | ± 0.31     | 79.42      | ± 0.36 |
| GPRGNN     | 83.17         | ± 0.78 | 71.86    | ± 0.67   | 79.75    | ± 0.38 | 89.32    | ± 0.29   | 94.49    | ± 0.14 | 95.13    | ± 0.09 | 96.85    | ± 0.08  | 78.12    | ± 0.23 | 71.10     | ± 0.12     | 79.76      | ± 0.59 |
| APPNP      | 83.32         | ± 0.55 | 71.78    | ± 0.46   | 80.14    | ± 0.22 | 90.18    | ± 0.17   | 94.32    | ± 0.14 | 94.49    | ± 0.07 | 96.54    | ± 0.07  | 78.87    | ± 0.11 | 72.34     | ± 0.24     | 78.84      | ± 0.09 |
| tGNN       | 82.97         | ± 0.68 | 71.74    | ± 0.49   | 80.67    | ± 0.34 | 83.40    | ± 1.33   | 89.92    | ± 0.72 | 92.85    | ± 0.48 | 96.24    | ± 0.24  | 71.49    | ± 1.05 | 72.88     | ± 0.26     | 81.79      | ± 0.54 |
| GraphGPS   | 82.84         | ± 1.03 | 72.73    | ± 1.23   | 79.94    | ± 0.26 | 91.19    | ± 0.54   | 95.06    | ± 0.13 | 93.93    | ± 0.12 | 97.12    | ± 0.19  | 78.66    | ± 0.49 | 70.97     | ± 0.41     | OOM        |        |
| NAGphormer | 82.12         | ± 1.18 | 71.47    | ± 1.30   | 79.73    | ± 0.28 | 91.22    | ± 0.14   | 95.49    | ± 0.11 | 95.75    | ± 0.09 | 97.34    | ± 0.03  | 77.16    | ± 0.72 | 70.13     | ± 0.55     | 73.55      | ± 0.21 |
| Exphormer  | 82.77         | ± 1.38 | 71.63    | ± 1.19   | 79.46    | ± 0.35 | 91.47    | ± 0.17   | 95.35    | ± 0.22 | 94.93    | ± 0.01 | 96.89    | ± 0.09  | 78.54    | ± 0.49 | 72.44     | ± 0.28     | OOM        |        |
| GOAT       | 83.18         | ± 1.27 | 71.99    | ± 1.26   | 79.13    | ± 0.38 | 90.96    | ± 0.90   | 92.96    | ± 1.48 | 94.21    | ± 0.38 | 96.24    | ± 0.24  | 77.00    | ± 0.77 | 72.41     | ± 0.40     | 82.00      | ± 0.43 |
| NodeFormer | 82.20         | ± 0.90 | 72.50    | ± 1.10   | 79.90    | ± 1.00 | 86.98    | ± 0.62   | 93.46    | ± 0.35 | 95.64    | ± 0.22 | 96.45    | ± 0.28  | 74.73    | ± 0.94 | 59.90     | ± 0.42     | 73.96      | ± 0.30 |
| SGFormer   | 84.50         | ± 0.80 | 72.60    | ± 0.20   | 80.30    | ± 0.60 | 92.42    | ± 0.66   | 95.58    | ± 0.36 | 95.71    | ± 0.24 | 96.75    | ± 0.26  | 80.05    | ± 0.46 | 72.63     | ± 0.13     | 81.54      | ± 0.43 |
| Polynormer | 83.25         | ± 0.93 | 72.31    | ± 0.78   | 79.24    | ± 0.43 | 93.68    | ± 0.21   | 96.46    | ± 0.26 | 95.53    | ± 0.16 | 97.27    | ± 0.08  | 80.10    | ± 0.67 | 73.46     | ± 0.16     | 83.82      | ± 0.11 |
| GCN        | 85.22         | ± 0.66 | 73.24    | ± 0.63   | 81.08    | ± 1.16 | 93.15    | ± 0.34   | 95.03    | ± 0.24 | 94.41    | ± 0.13 | 97.07    | ± 0.04  | 80.14    | ± 0.52 | 73.13     | ± 0.27     | 81.87      | ± 0.41 |
| Dirichlet  | energy 74.671 |        | 9.934    |          | 4.452    |        | 8.020    |          | 3.765    |        | 20.241   |        | 8.966    |         | 6.109    |        | 8.021     |            | 7.771      |        |
| GCN w / o  | dp 83.18      | ± 1.22 | 70.48    | ± 0.45   | 79.40    | ± 1.02 | 90.60    | ± 0.84   | 94.10    | ± 0.15 | 94.30    | ± 0.22 | 96.92    | ± 0.05  | 77.61    | ± 1.34 | 72.05     | ± 0.23     | 77.50      | ± 0.37 |
| Dirichlet  | energy 2.951  |        | 0.170    |          | 0.247    |        | 0.592    |          | 1.793    |        | 3.980    |        | 0.318    |         | 1.592    |        | 1.231     |            | 1.745      |        |
| GCN w / o  | BN 84.97      | ± 0.73 | 72.97    | ± 0.86   | 80.94    | ± 0.87 | 92.39    | ± 0.18   | 94.38    | ± 0.13 | 93.46    | ± 0.24 | 96.76    | ± 0.06  | 79.00    | ± 0.48 | 71.93     | ± 0.18     | 79.37      | ± 0.42 |
| SAGE       | 84.14         | ± 0.63 | 71.62    | ± 0.29   | 77.86    | ± 0.79 | 92.65    | ± 0.21   | 95.71    | ± 0.20 | 95.90    | ± 0.09 | 97.20    | ± 0.10  | 80.29    | ± 0.97 | 72.72     | ± 0.13     | 82.69      | ± 0.28 |
| SAGE w /   | o dp 83.06    | ± 0.80 | 69.68    | ± 0.82   | 76.40    | ± 1.48 | 90.17    | ± 0.60   | 94.90    | ± 0.17 | 95.80    | ± 0.08 | 97.06    | ± 0.06  | 78.84    | ± 1.17 | 71.37     | ± 0.31     | 79.82      | ± 0.22 |
| SAGE w /   | o BN 83.89    | ± 0.67 | 71.39    | ± 0.75   | 77.26    | ± 1.02 | 92.54    | ± 0.24   | 95.51    | ± 0.23 | 94.87    | ± 0.15 | 97.03    | ± 0.03  | 79.50    | ± 0.93 | 71.52     | ± 0.17     | 80.91      | ± 0.35 |
| GAT        | 83.92         | ± 1.29 | 72.00    | ± 0.91   | 80.48    | ± 0.99 | 93.47    | ± 0.27   | 95.53    | ± 0.16 | 94.49    | ± 0.17 | 96.73    | ± 0.10  | 80.21    | ± 0.68 | 72.83     | ± 0.19     | 80.05      | ± 0.34 |
| GAT w / o  | dp 82.58      | ± 1.47 | 71.08    | ± 0.42   | 79.28    | ± 0.58 | 92.94    | ± 0.30   | 93.88    | ± 0.16 | 94.30    | ± 0.14 | 96.42    | ± 0.08  | 78.67    | ± 0.40 | 71.52     | ± 0.41     | 77.87      | ± 0.25 |
| GAT w / o  | BN 83.76      | ± 1.32 | 71.82    | ± 0.83   | 80.43    | ± 1.03 | 92.16    | ± 0.26   | 95.05    | ± 0.49 | 93.33    | ± 0.26 | 96.57    | ± 0.20  | 79.49    | ± 0.62 | 71.68     | ± 0.36     | 78.21      | ± 0.32 |

#### 4.1 Datasets and Setup

Datasets. For node-level tasks, we used 10 datasets: Cora, CiteSeer, PubMed [\(Sen et al., 2008\)](#page-13-12), ogbn-arxiv, ogbn-products [\(Hu et al., 2020\)](#page-11-11), Amazon-Computer, Amazon-Photo, Coauthor-CS, Coauthor-Physics [\(Shchur et al., 2018\)](#page-13-13), and WikiCS [\(Mernyei & Cangea, 2020\)](#page-12-8). Cora, CiteSeer, and PubMed are citation networks, evaluated using the semi-supervised setting and data splits from [Kipf & Welling](#page-11-1) [\(2017\)](#page-11-1). Computer and Photo [\(Shchur et al., 2018\)](#page-13-13) are co-purchase networks. CS and Physics [\(Shchur et al., 2018\)](#page-13-13) are co-authorship networks. We used the standard 60%/20%/20% training/validation/test splits and accuracy as the evaluation metric [\(Chen et al., 2022;](#page-10-8) [Shirzad et al.,](#page-13-14) [2023;](#page-13-14) [Deng et al., 2024\)](#page-10-7). For WikiCS, we adopted the official splits and metrics [\(Mernyei & Cangea,](#page-12-8) [2020\)](#page-12-8). For large-scale graphs, we included ogbn-arxiv and ogbn-products with 0.16M to 2.4M nodes, using OGB's standard evaluation settings [\(Hu et al., 2020\)](#page-11-11).

For graph-level tasks, we used MNIST, CIFAR10 [\(Dwivedi et al., 2023\)](#page-11-12), and two Peptides datasets (functional and structural) [\(Dwivedi et al., 2022\)](#page-11-13). MNIST and CIFAR10 are graph versions of their image classification counterparts, constructed using 8-nearest neighbor graphs of SLIC superpixels. We follow all evaluation protocols suggested by [Dwivedi et al.](#page-11-12) [\(2023\)](#page-11-12). Peptides-func involves classifying graphs into 10 functional classes, while Peptides-struct regresses 11 structural properties. All evaluations followed the protocols in [\(Dwivedi et al., 2022\)](#page-11-13).

Baselines. Our main focus lies on the following prevalent GNNs and transformer models from Polynormer [\(Deng et al., 2024\)](#page-10-7): GCN [\(Kipf & Welling, 2017\)](#page-11-1), SAGE [\(Hamilton et al., 2017\)](#page-11-2), GAT [Velickovi](#page-13-10) ˇ c et al. [\(2018\)](#page-13-10), GCNII [\(Chen et al., 2020\)](#page-10-9), (Veli ´ ckovi ˇ [c et al., 2018\)](#page-13-10), APPNP [\(Gasteiger](#page-11-14) ´ [et al., 2018\)](#page-11-14), GPRGNN [\(Chien et al., 2020\)](#page-10-10), SGFormer [\(Wu et al., 2023\)](#page-13-11), Polynormer [\(Deng et al.,](#page-10-7) [2024\)](#page-10-7), GOAT [\(Kong et al., 2023\)](#page-12-9), NodeFormer [\(Wu et al., 2022\)](#page-13-15), NAGphormer [\(Chen et al., 2022\)](#page-10-8), G[TDwivedi & Bresson](#page-10-11) [\(2020\)](#page-10-11), SAN [Kreuzer et al.](#page-12-10) [\(2021\)](#page-12-10), MGT [Ngo et al.](#page-12-11) [\(2023\)](#page-12-11), DRew [Gutteridge](#page-11-15) [et al.](#page-11-15) [\(2023\)](#page-11-15), Graph-MLPMixer [He et al.](#page-11-16) [\(2023\)](#page-11-16), GRIT [Ma et al.](#page-12-12) [\(2023\)](#page-12-12) , GraphGPS [\(Rampa´sek](#page-13-16) ˇ [et al., 2022\)](#page-13-16), Exphormer [\(Shirzad et al., 2023\)](#page-13-14), CKGCN [\(Ma et al., 2024\)](#page-12-13), GRED [\(Ding et al.,](#page-10-12) [2024\)](#page-10-12), Graph Mamba [Behrouz & Hashemi](#page-10-13) [\(2024\)](#page-10-13). We report the performance results of baselines primarily from [\(Deng et al., 2024\)](#page-10-7), with the remaining obtained from their respective original papers or official leaderboards whenever possible, as those results are obtained by well-tuned models.

Experimental Setup. We implemented all models using the PyTorch Geometric library [\(Fey &](#page-11-17) [Lenssen, 2019\)](#page-11-17). The experiments are conducted on a single workstation with 8 RTX 3090 GPUs. For node-level tasks, we adhered to the training protocols specified in [\(Deng et al., 2024;](#page-10-7) [Luo et al.,](#page-12-14)

![](_page_8_Figure_1.jpeg)

Figure 4: Effect of dropout on feature F-norm, average pair distance, and Dirichlet energy.

[2024b;](#page-12-14)[a\)](#page-12-15), employing BN and adjusting the dropout rate between 0.1 and 0.7. In graph-level tasks, we adopted the settings from [\(Tonsho](#page-13-17) ¨ ff et al., [2023;](#page-13-17) [Luo et al., 2025\)](#page-12-16), utilizing BN with a consistent dropout rate of 0.2. All experiments were run with 5 different random seeds, and we report the mean accuracy and standard deviation. To ensure generalizability, we used Dirichlet energy [\(Cai & Wang,](#page-10-14) [2020\)](#page-10-14) as an oversmoothing metric, which is proportional to our feature energy.

#### 4.2 Node-level Classification Results

The node-level classification results in Table [2](#page-7-0) not only align with our theoretical predictions but also showcase the remarkable effectiveness of dropout. Notably, GCN with dropout and batch normalization outperforms state-of-the-art methods on several benchmarks, including Cora, CiteSeer, and PubMed. This superior performance underscores the practical significance of our theoretical insights. Consistently across all datasets, models employing dropout outperform their counterparts without it, validating our analysis that dropout provides beneficial regularization in GNNs, distinct from its effects in standard neural networks. The varying levels of improvement observed across different datasets support our theory of degree-dependent dropout effects that adapt to the graph structure. Furthermore, the consistent increase in Dirichlet energy when using dropout provides empirical evidence for our theoretical insight into dropout's crucial role in mitigating oversmoothing in GCNs, particularly evident in larger graphs. The complementary roles of dropout and batch normalization are demonstrated by the performance drop when either is removed, supporting our analysis of their synergistic interaction in GCNs.

#### 4.3 Graph-level Classification Results

Our graph-level classification results, presented in Tables [3](#page-9-0) and [4,](#page-9-0) further validate the broad applicability of our theoretical framework. First, compared to recent SOTA models, we observe that simply tuning dropout enables GNNs to achieve SOTA performance on three datasets and is competitive with the best single-model results on the remaining dataset. Second, the significant accuracy improvements on graph-level tasks such as Peptides-func and CIFAR10 highlight that our insights extend beyond node classification. The varying degrees of improvement across different graph datasets are consistent with our theory that dropout provides adaptive regularization tailored to graph properties. Third, the consistent increase in Dirichlet energy when using dropout supports our theoretical analysis of dropout's role in preserving feature diversity.

These results robustly validate our theory, showing that dropout in GCNs produces dimensionspecific stochastic sub-graphs, has degree-dependent effects, mitigates oversmoothing, and offers topology-aware regularization. Combined with batch normalization, dropout enhances GCN performance on graph-level tasks, affirming the relevance and utility of our framework and suggesting directions for improving GNN architectures.

### 4.4 Mitigating Oversmoothing Rather Than Co-adaptation

In traditional neural networks, dropout primarily prevents co-adaptation of neurons. However, our theoretical framework suggests that dropout in GCNs serves a fundamentally different purpose: mitigating oversmoothing rather than preventing co-adaptation. To validate this hypothesis, we examined how dropout affects weight matrices in a 2-layer GCN, focusing specifically on spectral norm changes (see Appendix [A.5\)](#page-17-0). We further analyzed three key metrics to quantify dropout's influence on feature representations, as shown in Figure [4.](#page-8-0) The left panel of Figure [4](#page-8-0) demonstrates that the

Table 3: Graph classification results on two peptide datasets from LRGB [\(Dwivedi et al., 2022\)](#page-11-13). Table 4: Graph classification results on two image datasets from [\(Dwivedi et al., 2023\)](#page-11-12).

| Model # graphs Avg. # nodes Avg. # edges Metric |        | Peptides-func 15,535 150.9 307.3 AP ↑ |        | Peptides-struct 15,535 150.9 307.3 MAE Model MNIST CIFAR10 # graphs 70,000 60,000 Avg. # nodes 70.6 117.6 Avg. # edges 564.5 941.1 Metric Accuracy ↑ Accuracy ↑ ↓ |
|-------------------------------------------------|--------|---------------------------------------|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GT                                              | 0.6326 | ± 0.0126                              | 0.2529 | ± 0.0016                                                                                                                                                          |
| SAN + RWSE                                      | 0.6439 | ± 0.0075                              | 0.2545 | ± 0.0012                                                                                                                                                          |
| GraphGPS                                        | 0.6535 | ± 0.0041                              | 0.2500 | ± 0.0012                                                                                                                                                          |
| MGT + WavePE                                    | 0.6817 | ± 0.0064                              | 0.2453 | ± 0.0025                                                                                                                                                          |
| DRew                                            | 0.7150 | ± 0.0044                              | 0.2536 | ± 0.0015                                                                                                                                                          |
| Exphormer                                       | 0.6527 | ± 0.0043                              | 0.2481 | ± 0.0007                                                                                                                                                          |
| Graph-MLPMixer                                  | 0.6970 | ± 0.0080                              | 0.2475 | ± 0.0015                                                                                                                                                          |
| GRIT                                            | 0.6988 | ± 0.0082                              | 0.2460 | ± 0.0012                                                                                                                                                          |
| CKGCN                                           | 0.6952 | ± 0.0068                              | 0.2477 | ± 0.0019                                                                                                                                                          |
| GRED                                            | 0.7085 | ± 0.0027                              | 0.2503 | ± 0.0019                                                                                                                                                          |
| Graph Mamba                                     | 0.6972 | ± 0.0100                              | 0.2477 | ± 0.0019                                                                                                                                                          |
| GCN                                             | 0.7015 | ± 0.0021                              | 0.2437 | ± 0.0012                                                                                                                                                          |
| Dirichlet energy                                | 9.649  |                                       | 6.121  |                                                                                                                                                                   |
| GCN w / o dp                                    | 0.6484 | ± 0.0034                              | 0.2541 | ± 0.0026                                                                                                                                                          |
| Dirichlet energy                                | 6.488  |                                       | 3.725  |                                                                                                                                                                   |
|                                                 |        |                                       |        | GT 90.831 ± 0.161 59.753 ± 0.293                                                                                                                                  |
|                                                 |        |                                       |        | SAN + RWSE                                                                                                                                                        |
|                                                 |        |                                       |        | GraphGPS 98.051 ± 0.126 72.298 ± 0.356                                                                                                                            |
|                                                 |        |                                       |        | MGT + WavePE DRew                                                                                                                                                 |
|                                                 |        |                                       |        | Exphormer 98.550 ± 0.039 74.696 ± 0.125                                                                                                                           |
|                                                 |        |                                       |        | Graph-MLPMixer 97.422 ± 0.110 73.961 ± 0.330                                                                                                                      |
|                                                 |        |                                       |        | GRIT 98.108 ± 0.111 76.468 ± 0.881                                                                                                                                |
|                                                 |        |                                       |        | CKGCN 98.423 ± 0.155 72.785 ± 0.436                                                                                                                               |
|                                                 |        |                                       |        | GRED 98.383 ± 0.012 76.853 ± 0.185                                                                                                                                |
|                                                 |        |                                       |        | Graph Mamba 98.392 ± 0.183 74.563 ± 0.379                                                                                                                         |
|                                                 |        |                                       |        | GatedGCN 98.684 ± 0.137 76.931 ± 0.367                                                                                                                            |
|                                                 |        |                                       |        | Dirichlet energy 1.119 1.541                                                                                                                                      |
|                                                 |        |                                       |        | GatedGCN w / o dp 98.235 ± 0.136 71.384 ± 0.397                                                                                                                   |
|                                                 |        |                                       |        | Dirichlet energy 0.987 0.845                                                                                                                                      |

Frobenius norm of features remains relatively stable regardless of dropout application, indicating that dropout does not uniformly scale all features. The middle panel reveals that dropout consistently doubles the average pairwise distance between nodes, helping maintain distinct node representations. Most significantly, the right panel shows that dropout substantially increases Dirichlet energy. This dramatic rise in Dirichlet energy, compared to the modest changes in Frobenius norm and pairwise distances, provides compelling evidence that dropout enhances discriminative power between connected nodes, explaining its effectiveness in preventing oversmoothing rather than simply reducing co-adaptation.

#### 4.5 Comparison with Dropout Variants

To further explore the practical impact of these different regularization techniques, we conducted hyperparameter tuning for DropEdge, DropNode, and DropMessage on the Cora, Citeseer, and Pubmed datasets. The results, summarized in Table [5,](#page-9-1) demonstrate that while these methods yield comparable performance, traditional dropout generally performs best.

Table 5: Experimental results of different regularization methods on Cora, Citeseer, and PubMed.

|     |               | Cora  | (GCN)  | CiteSeer | (GCN)  | PubMed | (GCN)  | Cora  | (SAGE) | CiteSeer | (SAGE) | PubMed | (SAGE) | Cora  | (GAT)  | CiteSeer | (GAT)  | PubMed | (GAT)  |
|-----|---------------|-------|--------|----------|--------|--------|--------|-------|--------|----------|--------|--------|--------|-------|--------|----------|--------|--------|--------|
| GNN |               | 83.18 | ± 1.22 | 70.48    | ± 0.45 | 79.40  | ± 1.02 | 83.06 | ± 0.80 | 69.68    | ± 0.82 | 76.40  | ± 1.48 | 82.58 | ± 1.47 | 71.08    | ± 0.42 | 79.28  | ± 0.58 |
| GNN | + Dropout     | 85.22 | ± 0.66 | 73.24    | ± 0.63 | 81.08  | ± 1.16 | 84.14 | ± 0.63 | 71.62    | ± 0.29 | 77.86  | ± 0.79 | 83.92 | ± 1.29 | 72.00    | ± 0.91 | 80.48  | ± 0.99 |
| GNN | + DropEdge    | 84.88 | ± 0.68 | 72.96    | ± 0.38 | 80.42  | ± 1.15 | 83.10 | ± 0.51 | 71.72    | ± 0.92 | 77.88  | ± 1.31 | 83.44 | ± 0.78 | 71.60    | ± 1.14 | 79.82  | ± 0.68 |
| GNN | + DropNode    | 84.92 | ± 0.52 | 73.08    | ± 0.39 | 80.60  | ± 0.49 | 83.42 | ± 0.58 | 71.92    | ± 0.65 | 78.06  | ± 1.09 | 83.80 | ± 0.97 | 71.30    | ± 0.87 | 79.50  | ± 0.68 |
| GNN | + DropMessage | 84.78 | ± 0.58 | 73.12    | ± 1.19 | 80.92  | ± 0.88 | 83.18 | ± 0.62 | 71.22    | ± 1.34 | 78.20  | ± 0.80 | 83.46 | ± 1.06 | 71.38    | ± 1.12 | 79.36  | ± 1.22 |

# 5 Conclusions

Our comprehensive theoretical analysis of dropout in GCNs has unveiled complex interactions between regularization, graph structure, and model performance that challenge traditional understanding. These insights not only deepen our understanding of how dropout functions in graph-structured data but also open new avenues for research and development in graph representation learning. Our findings suggest the need to reimagine regularization techniques for graph-based models, explore adaptive and structure-aware dropout strategies, and carefully balance local and global information in GCN architectures. Furthermore, the observed synergies between dropout and batch normalization point towards more holistic approaches to regularization in GNNs. As we move forward, this work lays a foundation for developing more robust and effective graph learning algorithms, with potential applications in dynamic graphs, large-scale graph sampling, and adversarial robustness. Ultimately, this research contributes to bridging the gap between the empirical success of GNNs and their theoretical foundations, paving the way for designing graph learning models.

# Acknowledgments

Hao Zhu was supported by the Science Digital Program in Commonwealth Scientific and Industrial Research Organization (CSIRO). Yuankai Luo received support from National Key R&D Program of China (2021YFB3500700), NSFC Grant 62172026, National Social Science Fund of China 22&ZD153, the Fundamental Research Funds for the Central Universities, State Key Laboratory of Complex & Critical Software Environment (SKLCCSE), and the HK PolyU Grant P0051029.

# References


[1] Alessandro Achille and Stefano Soatto. Information dropout: Learning optimal representations through noisy computation. *IEEE transactions on pattern analysis and machine intelligence*, 40 (12):2897–2905, 2018. Pierre Baldi and Peter J Sadowski. Understanding dropout. *Advances in neural information processing systems*, 26, 2013. Ali Behrouz and Farnoosh Hashemi. Graph mamba: Towards learning on graphs with state space models. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pp. 119–130, 2024. Xavier Bresson and Thomas Laurent. Residual gated graph convnets. *arXiv preprint arXiv:1711.07553*, 2017. Chen Cai and Yusu Wang. A note on over-smoothing for graph neural networks. *arXiv preprint arXiv:2006.13318*, 2020. Jinsong Chen, Kaiyuan Gao, Gaichao Li, and Kun He. Nagphormer: A tokenized graph transformer for node classification in large graphs. In *The Eleventh International Conference on Learning Representations*, 2022. Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, and Yaliang Li. Simple and deep graph convolutional networks. In *International conference on machine learning*, pp. 1725–1735. PMLR, 2020. Zhengdao Chen, Soledad Villar, Lei Chen, and Joan Bruna. On the equivalence between graph isomorphism testing and function approximation with gnns. *Advances in neural information processing systems*, 32, 2019. Eli Chien, Jianhao Peng, Pan Li, and Olgica Milenkovic. Adaptive universal generalized pagerank graph neural network. In *International Conference on Learning Representations*, 2020. Weilin Cong, Morteza Ramezani, and Mehrdad Mahdavi. On provable benefits of depth in training graph convolutional networks. *Advances in Neural Information Processing Systems*, 34:9936– 9949, 2021. Nima Dehmamy, Albert-Laszl ´ o Barab ´ asi, and Rose Yu. Understanding the representation power of ´ graph neural networks in learning graph topology. *Advances in Neural Information Processing Systems*, 32, 2019. Chenhui Deng, Zichao Yue, and Zhiru Zhang. Polynormer: Polynomial-expressive graph transformer in linear time. *arXiv preprint arXiv:2403.01232*, 2024. Yuhui Ding, Antonio Orvieto, Bobby He, and Thomas Hofmann. Recurrent distance filtering for graph representation learning. In *Forty-first International Conference on Machine Learning*, 2024. Simon S Du, Kangcheng Hou, Russ R Salakhutdinov, Barnabas Poczos, Ruosong Wang, and Keyulu Xu. Graph neural tangent kernel: Fusing graph neural networks with graph kernels. *Advances in neural information processing systems*, 32, 2019. Vijay Prakash Dwivedi and Xavier Bresson. A generalization of transformer networks to graphs. *arXiv preprint arXiv:2012.09699*, 2020.

[2] Vijay Prakash Dwivedi, Ladislav Rampa´sek, Mikhail Galkin, Ali Parviz, Guy Wolf, Anh Tuan Luu, ˇ and Dominique Beaini. Long range graph benchmark. *arXiv preprint arXiv:2206.08164*, 2022. Vijay Prakash Dwivedi, Chaitanya K Joshi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Benchmarking graph neural networks. *Journal of Machine Learning Research*, 24(43):1–48, 2023. Pascal Esser, Leena Chennuru Vankadara, and Debarghya Ghoshdastidar. Learning theory can (sometimes) explain generalisation in graph neural networks. *Advances in Neural Information Processing Systems*, 34:27043–27056, 2021. Taoran Fang, Zhiqing Xiao, Chunping Wang, Jiarong Xu, Xuan Yang, and Yang Yang. Dropmessage: Unifying random dropping for graph neural networks. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 37, pp. 4267–4275, 2023. Jiarui Feng, Yixin Chen, Fuhai Li, Anindya Sarkar, and Muhan Zhang. How powerful are k-hop message passing graph neural networks. *Advances in Neural Information Processing Systems*, 35: 4776–4790, 2022. Wenzheng Feng, Jie Zhang, Yuxiao Dong, Yu Han, Huanbo Luan, Qian Xu, Qiang Yang, Evgeny Kharlamov, and Jie Tang. Graph random neural networks for semi-supervised learning on graphs. *Advances in neural information processing systems*, 33:22092–22103, 2020. Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. *arXiv preprint arXiv:1903.02428*, 2019. Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In *international conference on machine learning*, pp. 1050–1059. PMLR, 2016. Hongyang Gao and Shuiwang Ji. Graph u-nets. In *international conference on machine learning*, pp. 2083–2092. PMLR, 2019. Vikas Garg, Stefanie Jegelka, and Tommi Jaakkola. Generalization and representational limits of graph neural networks. In *International Conference on Machine Learning*, pp. 3419–3430. PMLR, 2020. Johannes Gasteiger, Aleksandar Bojchevski, and Stephan Gunnemann. Predict then propagate: ¨ Graph neural networks meet personalized pagerank. *arXiv preprint arXiv:1810.05997*, 2018. Johannes Gasteiger, Stefan Weißenberger, and Stephan Gunnemann. Di ¨ ffusion improves graph learning. *Advances in neural information processing systems*, 32, 2019. Benjamin Gutteridge, Xiaowen Dong, Michael M Bronstein, and Francesco Di Giovanni. Drew: Dynamically rewired message passing with delay. In *International Conference on Machine Learning*, pp. 12252–12267. PMLR, 2023. Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. *Advances in neural information processing systems*, 30, 2017. Xiaoxin He, Bryan Hooi, Thomas Laurent, Adam Perold, Yann LeCun, and Xavier Bresson. A generalization of vit/mlp-mixer to graphs. In *International Conference on Machine Learning*, pp. 12724–12745. PMLR, 2023. Geoffrey E Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors. arxiv 2012. *arXiv preprint arXiv:1207.0580*, 2012. Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. *Advances in neural information processing systems*, 33:22118–22133, 2020. Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In *International Conference on Learning Representations*, 2017. URL [https:](https://openreview.net/forum?id=SJU4ayYgl) [//openreview.net/forum?id=SJU4ayYgl](https://openreview.net/forum?id=SJU4ayYgl).

[3] Kezhi Kong, Jiuhai Chen, John Kirchenbauer, Renkun Ni, C. Bayan Bruss, and Tom Goldstein. GOAT: A global transformer on large-scale graphs. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pp. 17375–17390. PMLR, 23–29 Jul 2023. URL [https:](https://proceedings.mlr.press/v202/kong23a.html) [//proceedings.mlr.press/v202/kong23a.html](https://proceedings.mlr.press/v202/kong23a.html). Devin Kreuzer, Dominique Beaini, Will Hamilton, Vincent Letourneau, and Prudencio Tossou. Re- ´ thinking graph transformers with spectral attention. *Advances in Neural Information Processing Systems*, 34:21618–21629, 2021. Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. *nature*, 521(7553):436–444, 2015. Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks for semi-supervised learning. In *Proceedings of the AAAI conference on artificial intelligence*, volume 32, 2018. Renjie Liao, Raquel Urtasun, and Richard Zemel. A pac-bayesian approach to generalization bounds for graph neural networks. *arXiv preprint arXiv:2012.07690*, 2020. Dongsheng Luo, Wei Cheng, Dongkuan Xu, Wenchao Yu, Bo Zong, Haifeng Chen, and Xiang Zhang. Parameterized explainer for graph neural network. *Advances in neural information processing systems*, 33:19620–19631, 2020. Yuankai Luo, Qijiong Liu, Lei Shi, and Xiao-Ming Wu. Structure-aware semantic node identifiers for learning on graphs. *arXiv preprint arXiv:2405.16435*, 2024a. Yuankai Luo, Lei Shi, and Xiao-Ming Wu. Classic GNNs are strong baselines: Reassessing GNNs for node classification. In *The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track*, 2024b. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=xkljKdGe4E) [xkljKdGe4E](https://openreview.net/forum?id=xkljKdGe4E). Yuankai Luo, Lei Shi, and Xiao-Ming Wu. Unlocking the potential of classic gnns for graph-level tasks: Simple architectures meet excellence. *arXiv preprint arXiv:2502.09263*, 2025. Shaogao Lv. Generalization bounds for graph convolutional neural networks via rademacher complexity. *arXiv preprint arXiv:2102.10234*, 2021. Liheng Ma, Chen Lin, Derek Lim, Adriana Romero-Soriano, Puneet K Dokania, Mark Coates, Philip Torr, and Ser-Nam Lim. Graph inductive biases in transformers without message passing. *arXiv preprint arXiv:2305.17589*, 2023. Liheng Ma, Soumyasundar Pal, Yitian Zhang, Jiaming Zhou, Yingxue Zhang, and Mark Coates. Ckgconv: General graph convolution with continuous kernels. *arXiv preprint arXiv:2404.13604*, 2024. Haggai Maron, Heli Ben-Hamu, Nadav Shamir, and Yaron Lipman. Invariant and equivariant graph networks. *arXiv preprint arXiv:1812.09902*, 2018. Peter Mernyei and C ´ at˘ alina Cangea. Wiki-cs: A wikipedia-based benchmark for graph neural net- ˘ works. *arXiv preprint arXiv:2007.02901*, 2020. Pietro Morerio, Jacopo Cavazza, Riccardo Volpi, Rene Vidal, and Vittorio Murino. Curriculum ´ dropout. In *Proceedings of the IEEE International Conference on Computer Vision*, pp. 3544– 3552, 2017. Nhat Khang Ngo, Truong Son Hy, and Risi Kondor. Multiresolution graph transformers and wavelet positional encoding for learning long-range and hierarchical structures. *The Journal of Chemical Physics*, 159(3), 2023. Kenta Oono and Taiji Suzuki. Graph neural networks exponentially lose expressive power for node classification. *arXiv preprint arXiv:1905.10947*, 2019.

[4] Ladislav Rampa´sek, Mikhail Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Do- ˇ minique Beaini. Recipe for a general, powerful, scalable graph transformer. *arXiv preprint arXiv:2205.12454*, 2022. Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang. Dropedge: Towards deep graph convolutional networks on node classification. In *International Conference on Learning Representations*, 2020. URL <https://openreview.net/forum?id=Hkx1qkrKPr>. Franco Scarselli, Ah Chung Tsoi, and Markus Hagenbuchner. The vapnik–chervonenkis dimension of graph and recursive neural networks. *Neural Networks*, 108:248–259, 2018. Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad. Collective classification in network data. *AI magazine*, 29(3):93–93, 2008. Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski, and Stephan Gunnemann. Pitfalls ¨ of graph neural network evaluation. *arXiv preprint arXiv:1811.05868*, 2018. Hamed Shirzad, Ameya Velingker, Balaji Venkatachalam, Danica J Sutherland, and Ali Kemal Sinop. Exphormer: Sparse transformers for graphs. *arXiv preprint arXiv:2303.06147*, 2023. Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. *The journal of machine learning research*, 15(1):1929–1958, 2014. Jan Tonsho ¨ ff, Martin Ritzert, Eran Rosenbluth, and Martin Grohe. Where did the gap go? reassessing the long-range graph benchmark. *arXiv preprint arXiv:2309.00367*, 2023. Petar Velickovi ˇ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li ´ o, and Yoshua ` Bengio. Graph attention networks. In *International Conference on Learning Representations*, 2018. Saurabh Verma and Zhi-Li Zhang. Stability and generalization of graph convolutional neural networks. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery* & *Data Mining*, pp. 1539–1548, 2019. Minh Vu and My T Thai. Pgm-explainer: Probabilistic graphical model explanations for graph neural networks. *Advances in neural information processing systems*, 33:12225–12235, 2020. Stefan Wager, Sida Wang, and Percy S Liang. Dropout training as adaptive regularization. *Advances in neural information processing systems*, 26, 2013. Li Wan, Matthew Zeiler, Sixin Zhang, Yann Le Cun, and Rob Fergus. Regularization of neural networks using dropconnect. In *International conference on machine learning*, pp. 1058–1066. PMLR, 2013. Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Weinberger. Simplifying graph convolutional networks. In *International conference on machine learning*, pp. 6861–6871. PMLR, 2019. Qitian Wu, Wentao Zhao, Zenan Li, David P Wipf, and Junchi Yan. Nodeformer: A scalable graph structure learning transformer for node classification. *Advances in Neural Information Processing Systems*, 35:27387–27401, 2022. Qitian Wu, Wentao Zhao, Chenxiao Yang, Hengrui Zhang, Fan Nie, Haitian Jiang, Yatao Bian, and Junchi Yan. Simplifying and empowering transformers for large-graph representations. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. URL [https://](https://openreview.net/forum?id=R4xpvDTWkV) [openreview.net/forum?id=R4xpvDTWkV](https://openreview.net/forum?id=R4xpvDTWkV). Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In *International Conference on Learning Representations*, 2018. Han Yang, Kaili Ma, and James Cheng. Rethinking graph regularization for graph neural networks. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 35, pp. 4573–4581, 2021.

[5] Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure Leskovec. Gnnexplainer: Generating explanations for graph neural networks. *Advances in neural information processing systems*, 32, 2019. Hao Yuan, Jiliang Tang, Xia Hu, and Shuiwang Ji. Xgnn: Towards model-level explanations of graph neural networks. In *Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery* & *data mining*, pp. 430–438, 2020. Hao Yuan, Haiyang Yu, Jie Wang, Kang Li, and Shuiwang Ji. On explainability of graph neural networks via subgraph explorations. In *International conference on machine learning*, pp. 12241– 12252. PMLR, 2021. Rui-Ray Zhang and Massih-Reza Amini. Generalization bounds for learning under graphdependence: a survey. *Mach. Learn.*, 113(7):3929–3959, April 2024. ISSN 0885-6125. doi: 10.1007/s10994-024-06536-9. URL <https://doi.org/10.1007/s10994-024-06536-9>. Shuai Zhang, Meng Wang, Sijia Liu, Pin-Yu Chen, and Jinjun Xiong. Fast learning of graph neural networks with guaranteed generalizability: one-hidden-layer case. In *International Conference on Machine Learning*, pp. 11268–11277. PMLR, 2020. Lingxiao Zhao and Leman Akoglu. Pairnorm: Tackling oversmoothing in gnns. *arXiv preprint arXiv:1909.12223*, 2019.
# A Appendix

### A.1 Proof of Theorem [1](#page-3-1)

*Proof.* Let's approach this proof:

Step 1: For a single feature *j*, the probability that an edge is present in the sub-graph *E* (*l*, *j*) *t* is (1− *p*) 2 , as both endpoints need to retain this feature.

Step 2: The probability that an edge is not present in E (*l*, *j*) *t* is 1 − (1 − *p*) <sup>2</sup> = *p*(2 − *p*).

Step 3: For a sub-graph to be identical to the original graph, all edges must be present. The probability of this is: ((1 − *p*) 2 ) |E| = (1 − *p*) 2|E| .

Step 4: Therefore, the probability that E (*l*, *j*) *t* is different from the original graph (i.e., unique) is 1 − (1 − *p*) 2|E| .

Step 5: Define an indicator random variable *X<sup>j</sup>* for each feature *j*:

| $X_j = \begin{cases} 1 & \text{if } \mathcal{E}_t^{(l,j)} \text{ is unique} \\ 0 & \text{otherwise} \end{cases}$ |
|------------------------------------------------------------------------------------------------------------------|
|------------------------------------------------------------------------------------------------------------------|

Step 6: We have:

$$P(X_j = 1) = 1 - (1 - p)^{2|\mathcal{E}|} [P(X_j = 0) = (1 - p)^{2|\mathcal{E}|}]$$

Step 7: The expected value of *X<sup>j</sup>* is:

$$\mathbb{E}[X_j] = 1 \cdot P(X_j = 1) + 0 \cdot P(X_j = 0) = 1 - (1 - p)^{2^{\text{el}}}.$$

Step 8: The total number of unique sub-graphs is P*<sup>d</sup><sup>l</sup> j*=1 *Xj* . By the linearity of expectation:

$$\mathbb{E}[\mathbb{E}_t^{(l,j)} \mid j = 1, \dots, d_l] = \mathbb{E}[\sum_{j=1}^{d_l} X_j] = \sum_{j=1}^{d_l} \mathbb{E}[X_j] = d_l(1 - (1 - p)^{2|\mathcal{E}_l|}).$$

# A.2 Proof of Theorem [5](#page-4-0)

*Proof.* We start with the definition of feature energy:

$$E(H^{(l)}) = \frac{1}{2|\mathcal{E}|} \sum_{i,j \in \mathcal{E}} \|\mathbf{h}_i^{(l)} - \mathbf{h}_j^{(l)}\|_2^2$$

Step 1: Taking the expectation:

$$\mathbb{E}[E(H^{(l)})] = \frac{1}{2|\mathcal{E}|} \sum_{i,j \in \mathcal{E}} \mathbb{E}[\|h_i^{(l)} - h_j^{(l)}\|_2^2].$$

.

Step 2: Since P (*i*, *<sup>j</sup>*)∈E[∥h*i*∥ <sup>2</sup> + ∥h*j*∥ 2 ] = 2 P *<sup>i</sup> degi*∥h*i*∥ 2 :

$$\begin{aligned} \frac{1}{2|\mathcal{E}|} \sum_{i,j \in \mathcal{E}} \mathbb{E}[\|\mathbf{h}_i^{(t)} - \mathbf{h}_j^{(t)}\|_2^2] &= \frac{1}{2|\mathcal{E}|} \sum_{i,j \in \mathcal{E}} \mathbb{E}[\|\frac{1}{1-p} \mathbf{M}_i^{(t)} \odot \mathbf{z}_i^{(t)} - \frac{1}{1-p} \mathbf{M}_j^{(t)} \odot \mathbf{z}_j^{(t)}\|_2^2] \\ &= \frac{1}{2|\mathcal{E}|(1-p)^2} \sum_{i,j \in \mathcal{E}} \mathbb{E}[\|\mathbf{M}_i^{(t)} \odot \mathbf{z}_i^{(t)} - \mathbf{M}_j^{(t)} \odot \mathbf{z}_j^{(t)}\|_2^2] \\ &= \frac{1}{2|\mathcal{E}|(1-p)^2} \sum_{i,j \in \mathcal{E}} [(1-p)(\|\mathbf{z}_i^{(t)}\|_2^2 + \|\mathbf{z}_j^{(t)}\|_2^2) - 2(1-p)^2(\mathbf{z}_i^{(t)})^\top \mathbf{z}_j^{(t)}] \\ &= \frac{1}{1-p} \frac{1}{|\mathcal{E}|} \sum_i deg_i \|\mathbf{z}_i^{(t)}\|_2^2 - \frac{1}{|\mathcal{E}|} \text{Tr}(\mathbf{Z}^\top \mathbf{A} \mathbf{Z}) \end{aligned}$$

where z*<sup>i</sup>* = <sup>σ</sup>( P *<sup>k</sup>* A˜ *ik*h (*l*−1) *<sup>k</sup>* <sup>W</sup>(*l*) ).

Step 3: Since *deg<sup>i</sup>* ≤ *degmax* for all *i*:

$$\frac{1}{|\mathcal{E}|} \sum_i deg_i \|z_i\|_2^2 \leq \frac{deg_{\max}}{|\mathcal{E}|} \sum_i \|z_i\|_2^2 = \frac{deg_{\max}}{|\mathcal{E}|} \|\mathbf{Z}\|_F^2.$$

Step 4: By ReLU non-negative homogeneity and submultiplicative property:

$$\|\mathbf{Z}^{(0)}\|_F^2 \leq \|\tilde{\mathbf{A}}\mathbf{H}^{(1)}\mathbf{W}^{(0)}\|_F^2 \leq \|\mathbf{W}^{(0)}\|_2^2 \|\tilde{\mathbf{A}}\|_2^2 \|\mathbf{H}^{(1)}\|_F^2.$$

Step 5: By dropout scaling with probability *p*:

$$\|\mathbf{H}^{(l-1)}\|_F^2 = \frac{1}{1-p} \|\mathbf{Z}^{(l-1)}\|_F^2$$

Step 6: By applying steps 4-5 recursively:

$$\|\mathbf{Z}^{(i)}\|_F^2 \leq (\frac{1}{1-p})^{l-1} \|\tilde{\mathbf{A}}\|_2^{2^l} \prod_{i=1}^l \|\mathbf{W}^{(i)}\|_2^2 \|\mathbf{X}\|_F^2.$$

Step 7: Combining all inequalities:

$$\mathbb{E}[E(\mathbf{H}^{(l)})] \leq \frac{deg_{\max}}{|\mathcal{E}|} \left( \frac{1}{1-p} \right)^l \|\tilde{\mathbf{A}}\|_2^{2l} \prod_{i=1}^l \|\mathbf{W}^{(i)}\|_2^2 \|\mathbf{X}\|_F^2.$$

□

A.3 Proof of Theorem [6](#page-5-1)

*Proof.* The proof proceeds in several steps:

Step 1: Dependency Graph. Let G = (V, E) be the dependency graph where vertices V represent nodes in the graph, and an edge (*i*, *j*) ∈ E exists if nodes *i* and *j* are connected through message passing via A˜. The graph G is fixed across all layers as it is determined by the structure of A˜.

Step 2: Dropout Effect as Perturbation. At layer *l* with dropout probability *p<sup>l</sup>* , let δ (*l*) be the perturbation matrix:

$$\delta^{(l)} = \frac{1}{1-p_l} M^{(l)} \odot \sigma(\tilde{A}H^{(l-1)} \mathbf{W}^{(l)}) - \sigma(\tilde{A}H^{(l-1)} \mathbf{W}^{(l)}), \quad (11)$$

where *M*(*l*) has elements drawn from Bernoulli(1 − *pl*).

Step 3: Perturbation Propagation. Let *Fl*(X) denote the network output with dropout applied up to layer *l*. With *L*σ-Lipschitz activation:

$$L_l = \prod_{i=l}^L (L_{\sigma} \|\mathbf{W}^{(i)}\|_2 \cdot \|\tilde{\mathbf{A}}\|_2) \quad (12)$$

By operator norm properties:

$$\|F_t(\mathbf{X}) - F_{t-1}(\mathbf{X})\|_F \leq L_t \|\boldsymbol{\delta}^{(t)}\|_F \quad (13)$$

Step 4: Bounding Matrix Perturbation. Let δ (*l*) *i j* denote the (*i*, *j*)-th entry of δ (*l*) . By Janson's inequality for dependent variables over G [\(Zhang & Amini, 2024\)](#page-14-6):

$$\mathbb{E} \left[ \sum_{i,j} (\delta_{ij}^{(\ell)})^2 \right] \leq \frac{1}{\chi_f(\mathcal{G})} \sum_{i,j} \mathbb{E}[(\delta_{ij}^{(\ell)})^2] \quad (14)$$

Taking the square root and using the definition of Frobenius norm:

$$\mathbb{E}[\|\delta^{(l)}\|_F] \leq \sqrt{\frac{1}{\chi_f(\mathcal{G})}} \cdot \mathbb{E}[\|\delta^{(l)}\|_F^2] \quad (15)$$

$$= \sqrt{\frac{p_l}{(1-p_l)\chi_f(\mathcal{G})}} \|\sigma(\tilde{A}H^{(l-1)}W^{(l)})\|_F \quad (16)$$

where we use <sup>E</sup>[(M(*l*) ) 2 ] = <sup>E</sup>[M(*l*) ] = 1 − *p<sup>l</sup>* .

Step 5: Loss Stability. By the Lipschitz property of the loss function:

$$\mathbb{E}[|L(F_t(x)) - L(F_{t-1}(x))|_F] \leq L_{olss} \cdot \mathbb{E}[|F_t(x) - F_{t-1}(x)|_F] \quad (17)$$

$$\leq L_{loss} \cdot L_l \cdot L_\sigma \cdot \sqrt{\frac{p_l}{(1-p_l)\chi_f(\mathcal{G})}} \|\sigma(\tilde{A}H^{(l-1)}\mathbf{W}^{(l)})\|_F \quad (18)$$

Step 6: Final Concentration Bound. Using McDiarmid's inequality and noting the impact of message passing through <sup>χ</sup>*f*(G), with probability at least 1 − δ:

$$\mathbb{E}_D[L(F(x))] - \mathbb{E}_S[L(F(x))] \leq O\left(\sqrt{\frac{\log(1/\delta)}{n}}\right) \sum_{l=1}^L L_{loss} \cdot L_l \cdot \sqrt{\frac{p_l}{(1-p_l)\chi_f(\mathcal{G})}} \|\sigma(\tilde{\mathbf{A}}\mathbf{H}^{(l-1)}\mathbf{W}^{(l)})\|_F \quad (19)$$

The bound shows that GNNs (χ*f*(G) > 1 due to message passing) achieve better stability than MLPs (χ*f*(G) = 1, no message passing), with the benefit increasing with graph connectivity. □

# A.4 Proof of Theorem [7](#page-5-2)

*Proof.* Step 1: Start with feature energy and node representation:

$$E(\mathbf{H}^{(l)}) = \frac{1}{2|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} \|\mathbf{h}_i^{(l)} - \mathbf{h}_j^{(l)}\|^2$$

$$\mathbf{h}_i^{(l)} = \frac{1}{1-p} \mathbf{M}_i^{(l)} \odot \mathbf{z}_i^{(l)}$$

Step 2: For the BN output before ReLU at layer *l*, for each feature dimension *d* ∈ {1, ..., *dl*}:

$$(Y^{(l)})_{:,d} = \text{BN}((\tilde{A}H^{(l-1)}W^{(l)})_{:,d}) = \gamma_d^{(l)} \frac{(\tilde{A}H^{(l-1)}W^{(l)})_{:,d} - \mu_d^{(l)}}{\sqrt{(\sigma_d^{(l)})^2 + \epsilon}} + \beta_d^{(l)}$$

Step 3: For ReLU activation z = max(0, y) at layer l, for each dimension d:

$$\mathbb{E}[(z_d^{(l)})^2] \geq \Phi(\beta_d^{(l)}/\gamma_d^{(l)}) \cdot (\beta_d^{(l)})^2$$

where Φ is the standard normal CDF.

Step 4: Using the BN-induced bound:

$$\begin{aligned}\|z_i^{(l)}\|^2 &= \sum_{d=1}^{d_l} (z_i^{(l)})_d^2 \\ &\geq \sum_{d=1}^{d_l} \Phi(\beta_d^{(l)} / \gamma_d^{(l)}) \cdot (\beta_d^{(l)})^2 > 0\end{aligned}$$

Step 5: For feature energy with merged terms:

$$\begin{aligned} E(\mathbf{H}^{(l)}) &= \frac{1}{2|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} \left[ \frac{1}{1-p} (\|\mathbf{z}_i^{(l)}\|^2 + \|\mathbf{z}_j^{(l)}\|^2) - 2(\mathbf{z}_i^{(l)})^T \mathbf{z}_j^{(l)} \right] \\ &\geq \frac{1}{2|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} \left[ \frac{1}{1-p} (\|\mathbf{z}_i^{(l)}\|^2 + \|\mathbf{z}_j^{(l)}\|^2) - (\|\mathbf{z}_i^{(l)}\|^2 + \|\mathbf{z}_j^{(l)}\|^2) \right] \\ &= \frac{1}{2|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} \left( \frac{1}{1-p} - 1 \right) (\|\mathbf{z}_i^{(l)}\|^2 + \|\mathbf{z}_j^{(l)}\|^2) \\ &= \frac{p}{1-p} \frac{1}{2|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} (\|\mathbf{z}_i^{(l)}\|^2 + \|\mathbf{z}_j^{(l)}\|^2) \\ &= \frac{p}{1-p} \frac{1}{2|\mathcal{E}|} \sum_i deg_i \|\mathbf{z}_i^{(l)}\|^2 \\ &\geq \frac{p \cdot deg_{\min}}{1-p} \frac{1}{2|\mathcal{E}|} \|\mathbf{Z}^{(l)}\|_F^2 \end{aligned}$$

Then with BN bound:

$$E(\mathbf{H}^{(l)}) \geq \frac{p \cdot deg_{\min}}{1-p} \frac{1}{2|\mathcal{E}|} \sum_{d=1}^d \Phi(\beta_d^{(l)}/\gamma_d^{(l)}) \cdot (\beta_d^{(l)})^2$$

□

### A.5 Effect of Dropout on Max Singular Values of the Weight Matrices

We analyze why dropout leads to larger weight matrices in terms of spectral norm ∥W∥2. Consider the gradient update for weights W(2) between layers:

$$\frac{\partial L}{\partial \mathbf{W}^{(2)}} = (\tilde{\mathbf{A}}\mathbf{H}_{drop}^{(1)})^\top \times \frac{\partial L}{\partial \mathbf{H}^{(2)}} = (\tilde{\mathbf{A}}(\mathbf{H}^{(1)} \odot \mathbf{M}^{(1)})/(1-p))^\top \times \frac{\partial L}{\partial \mathbf{H}^{(2)}} \quad (20)$$

where *p* is the dropout rate and *M*<sup>1</sup> is the dropout mask. This leads to weight updates:

$$\Delta \mathbf{W}^{(2)} = -\eta(\tilde{\mathbf{A}}\mathbf{H}_{drop}^{(1)})^{\top} \times \frac{\partial L}{\partial \mathbf{H}^{(2)}} = -\eta(\tilde{\mathbf{A}}(\mathbf{H}^{(1)} \odot \mathbf{M}^{(1)})/(1-p))^{\top} \times \frac{\partial L}{\partial \mathbf{H}^{(2)}} \quad (21)$$

The 1/(1 − *p*) scaling factor in dropout has two key effects: 1) For surviving features (where *M* (1) *i j* = 1), the gradient is amplified by 1/(1 − *p*). This leads to larger updates for these weights

during training. 2) During each iteration, different subsets of features survive, but their gradients are consistently scaled up. Over many iterations, this accumulates to larger weight values despite the unbiased expectation maintained by dropout. Specifically, with dropout rate *p* when *p* = 0.5, surviving gradients are doubled. This amplification effect compounds over training iterations. While dropout maintains unbiased expected values during forward propagation, the consistent gradient scaling during backward propagation leads to systematically larger weight magnitudes. Empirically, we observe that higher dropout rates correlate with larger spectral norms ∥W∥ 2 2 (as shown in Figure [5\)](#page-18-1), supporting this theoretical analysis. The increased weight magnitudes directly contribute to higher feature energy *E*(H(2)) during inference, as:

$$E(\mathbf{H}^{(2)}) = \frac{1}{2|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} \|\mathbf{h}_i^{(2)} - \mathbf{h}_j^{(2)}\|_2^2 \quad (22)$$

where larger weights produce more distinctive features between connected nodes, helping mitigate oversmoothing.

![](_page_18_Figure_4.jpeg)

Figure 5: Effect of dropout on max singular values of the weight matrices.

#### A.6 Empirical Validation of Theoretical Properties

In this section, we provide empirical evidence supporting the theoretical properties derived in Section 3.

Dimension-Specific Stochastic Sub-graphs. Figure [6](#page-19-1) shows how varying dropout rates impact the number of edges E*<sup>t</sup>* in stochastic sub-graphs of a 2-layer GCN, defined by Equation [3,](#page-2-0) across the Cora and Citeseer datasets. We observe that higher dropout rates correlate with fewer edges in these sub-graphs. This variation demonstrates dropout's role in GCNs as a form of structural regularization, where dimension-specific stochastic sub-graphs are generated. Each feature dimension samples a different sub-graph from the original graph at each iteration. This mechanism provides a rich set of structural variations during training, potentially enhancing the model's ability to capture diverse graph patterns. Figures [7](#page-19-1) & [8](#page-19-1) illustrate the behavior of active features along paths of length 1 and 2 within a 2-layer GCN equipped with 16 hidden dimensions, across varying dropout rates. Notably, at a dropout rate of 0.6, the average number of active features approaches zero. This characteristic also underscores the importance of multidimensional feature spaces in ensuring robust information transmission under feature dropout.

Degree-Dependent Nature of Dropout Effects. Figure [9](#page-19-0) demonstrates that dropout affects the effective degree of nodes. Figure [10](#page-19-0) illustrates that the CV decreases as node degree increases. This degree-dependent effect distinguishes dropout in GCNs from its application in standard neural networks and suggests that the optimal dropout strategy for GCNs may need to consider the graph structure explicitly. Figure [11](#page-19-0) presents empirical evidence supporting Theorem [3](#page-3-2) (Degree-Dependent Dropout Effect), which predicts that high-degree nodes experience relatively less variation in their effective degree due to dropout. The figure shows classification accuracy on the Cora

![](_page_19_Figure_1.jpeg)

Figure 6: Sub-graph size. Figure 7: Active path on Cora. Figure 8: Active path on Citeseer.

![](_page_19_Figure_3.jpeg)

Figure 9: Effective degree. Figure 10: Effective CV vs degree. Figure 11: Accuracy on Cora.

dataset broken down by node degree, demonstrating that nodes with higher degrees consistently achieve better performance. This aligns with our theoretical finding that high-degree nodes maintain more stable representations under dropout, as their effective degree has lower coefficient of variation. The observed pattern confirms that dropout naturally provides adaptive regularization that adjusts to the local graph structure, with stronger stabilizing effects for topologically important nodes.