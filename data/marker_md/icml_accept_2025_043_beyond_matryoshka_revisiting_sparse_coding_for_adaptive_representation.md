# Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation

Tiansheng Wen \* 1 2 Yifei Wang \* 3 Zequn Zeng <sup>1</sup> Zhong Peng <sup>1</sup> Yudi Su <sup>1</sup> Xinyang Liu <sup>1</sup> Bo Chen <sup>1</sup> Hongwei Liu <sup>1</sup> Stefanie Jegelka 3 4 Chenyu You <sup>2</sup>

![](_page_0_Figure_2.jpeg)

Figure 1. Overview of our proposed method. (a) Illustrative comparison between standard embeddings (dense, long) and two different compression schemes: Matryoshka representations (MRL) [\(Kusupati et al.,](#page-10-0) [2022\)](#page-10-0) with short length and our Contrastive Sparse Representation (CSR) based on sparsification. (b) Comparison of retrieval accuracy and time of different methods on ImageNet with GPUs. For CSR, we present results with the SOTA RN50 backbone from [Wightman](#page-11-0) [\(2019\)](#page-11-0) as well as the same RN50 backbone from [Kusupati et al.](#page-10-0) [\(2022\)](#page-10-0) for a fair comparison. Compared to MRL and int8 quantification (Quant Int8) methods, our sparse embedding approach CSR attains the best retrieval accuracy (very close to full representations) while being much more efficient in retrieval time, using sparse matrix multiplication on GPU. (c) Training GPU hours of CSR compared to baseline methods, where we outperform MRL on average 1-NN accuracy with much less training time.

# Abstract

Many large-scale systems rely on high-quality deep representations (embeddings) to facilitate tasks like retrieval, search, and generative modeling. Matryoshka Representation Learning (MRL) recently emerged as a solution for adaptive embedding lengths, but it requires full model retraining and suffers from noticeable performance degradations at short lengths. In this paper, we show that *sparse coding* offers a compelling alternative for achieving adaptive representation with minimal overhead and higher fidelity. We propose Contrastive Sparse Representation (CSR), a method that sparsifies pre-trained embeddings

into a high-dimensional but *selectively activated* feature space. By leveraging lightweight autoencoding and task-aware contrastive objectives, CSR preserves semantic quality while allowing flexible, cost-effective inference at different sparsity levels. Extensive experiments on image, text, and multimodal benchmarks demonstrate that CSR consistently outperforms MRL in terms of both accuracy and retrieval speed—often by large margins—while also cutting training time to a fraction of that required by MRL. Our results establish sparse coding as a powerful paradigm for adaptive representation learning in real-world applications where efficiency and fidelity are both paramount. Code is available at [this https URL.](https://github.com/neilwen987/CSR_Adaptive_Rep)

# 1. Introduction

Representation learning is at the core of deep learning [\(Le-](#page-10-1)[Cun et al.,](#page-10-1) [2015\)](#page-10-1) and high-quality representations of inputs (*e.g.*, image, text) empower numerous large-scale systems, including but not limited to search engines, vector databases,

<sup>\*</sup>Equal contribution <sup>1</sup>National Key Laboratory of Radar Signal Processing, Xidian University, Xi'an, China <sup>2</sup> Stony Brook University, New York, USA <sup>3</sup>MIT CSAIL, MA, USA <sup>4</sup>TU Munich. Correspondence to: Bo Chen <bchen@mail.xidian.edu.cn>, Hongwei Liu <hwliu@xidian.edu.cn>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

and retrieval-augmented generative AI [\(Lewis et al.,](#page-10-2) [2020\)](#page-10-2). However, the rapid growth in data volume poses significant challenges for latency-sensitive applications. It is thus desirable to develop representations of adaptive inference cost that can best trade-off between accuracy and inference speed.

Recently, a class of methods called Matryoshka Representation Learning (MRL) [\(Kusupati et al.,](#page-10-0) [2022\)](#page-10-0) has drawn a lot of attention and is now officially supported in the latest OpenAI and Google's Gemini text embedding APIs [\(Ope](#page-11-1)[nAI,](#page-11-1) [2024;](#page-11-1) [Lee et al.,](#page-10-3) [2024b\)](#page-10-3) with millions of users and applications. The idea if MRL is to train an ensemble of representations truncated at different lengths (*e.g.*, from 8 to 2048) through joint multi-task training. However, MRL deviates from standard representation learning and requires full parameter updates to the backbone; the joint training also inevitably sacrifices the quality of representations at a noticeable margin (*e.g.*, 5% drop of top-1 accuracy on ImageNet at full representation length). These limitations render MRL a costly and lossy method for adaptive representation.

In this paper, we revisit sparse coding [\(Lee et al.,](#page-10-4) [2006\)](#page-10-4) as a much faster, lightweight, and high-fidelity approach to achieve adaptive representation. As illustrated in Figure [1\(](#page-0-0)a), instead of truncating the representation length as in MRL, we leverage sparse vectors and sparse matrix factorization to attain computational efficiency. Specifically, we sparsify a full representation at different levels (characterized by K, the number of activated neurons). We find that a few numbers of activated neurons (*e.g.*, 4 to 16) can preserve the performance of a much longer dense representation (*e.g.*, 2048 dimensions). This is in sharp contrast to MRL embeddings whose quality deteriorates a lot at such extremely short lengths (>10% drop). Therefore, sparse features using sparse vector formats can be stored efficiently with only a few activated neurons. With the help of sparse matrix factorization (with native GPU support in modern deep learning libraries such as PyTorch)[<sup>1</sup>](#page-1-0) , these sparse embeddings can be used for retrieval tasks at a much higher speed with a complexity order of O(K), where K is very small. In comparison, MRL requires a longer length of representation (*e.g.* 256) to attain similar accuracy (if possible), leading to extra slower inference speed. As shown in Figure [1\(](#page-0-0)b), MRL is inferior to our method in terms of both accuracy and retrieval time by a significant margin.

Another key advantage of sparse features is that they eliminate the need to retrain the entire network. In contrast, MRL[—Kusupati et al.](#page-10-0) [\(2022\)](#page-10-0) noted—performs poorly unless full-parameter tuning. However, many existing foundation models, such as the multimodal representations in CLIP [\(Radford et al.,](#page-11-2) [2021\)](#page-11-2) and the text embeddings in NV-Embed

[\(Lee et al.,](#page-10-5) [2024a\)](#page-10-5), are pre-trained as single representations on massive Internet-scale data. Fine-tuning these models would be prohibitively expensive and would prevent leveraging pre-trained open weights. Leveraging recent advances in training sparse autoencoders (SAEs) [\(Cunningham et al.,](#page-9-0) [2023;](#page-9-0) [Gao et al.,](#page-9-1) [2024\)](#page-9-1), we can train a lightweight 2-layer MLP module for sparsifying pre-trained embeddings within a very short period of time (*e.g.*, half of an hour on ImageNet with a single GPU), which is of orders of magnitude faster than MRL, as shown in Figure [1\(](#page-0-0)c).

These pieces of evidence on accuracy, retrieval time, and training time show that sparse features are strong alternatives to MRL methods for producing high-fidelity and computationally efficient representations with a lightweight module and training cost. Our proposed method, Contrastive Sparse Representation Learning (CSR), combines contrastive retrieval and reconstructive autoencoding objectives to preserve the original feature semantics while better tailing it down to the retrieval tasks. We evaluate CSR on a range of standard embedding benchmarks, from image embedding, text embedding, to multimodal embeddings, and compare it against various state-of-the-art efficient embedding models. Extensive experiments show that CSR consistently outperforms MRL and its variants by significant margins in terms of both accuracy and efficiency. Notably, under the same compute budget, CSR rivals MRL's performance by 9%, 15%, and 7% on ImageNet classification, MTEB text retrieval, and MS COCO retrieval, respectively. Our main contributions are:

- We propose sparse coding as an alternative approach to adaptive representation learning and demonstrate its numerous advantages over the MRL approach in terms of fidelity, retrieval cost, and training cost.
- We introduce an effective learning method for sparse adaptive representation, Contrastive Sparse Representation (CSR) Learning. It combines a taskspecific sparse contrastive learning loss with a reconstructive loss to maintain overall embedding quality. This generic design consistently improves performance across different tasks like classification and retrieval.
- We conduct a detailed analysis of CSR, examining various factors and providing a fair comparison with MRL in terms of retrieval time and accuracy. We further validate CSR's effectiveness across real-world domains and benchmarks, where it achieves competitive performance against heavily trained state-of-the-art MRL models with significantly lower computational costs. On the inference side, CSR delivers a 69× speedup on ImageNet1k 1-NN tasks without compromising performance compared to quantization-based approaches.

<sup>1</sup> PyTorch's native sparse vector library can be found at [https:](https://pytorch.org/docs/stable/sparse.html) [//pytorch.org/docs/stable/sparse.html](https://pytorch.org/docs/stable/sparse.html).

#### 2. Related Work

Adaptive Representation Learning. Recent research has increasingly focused on learning *adaptive representations* that cater to multiple downstream tasks with diverse computational requirements. Early efforts explored contextbased architectural adaptations [\(Kim & Cho,](#page-10-6) [2020\)](#page-10-6), dynamic widths and depths in BERT [\(Hou et al.,](#page-10-7) [2020\)](#page-10-7), and random layer dropping during training to improve pruning robustness [\(Fan et al.,](#page-9-2) [2019\)](#page-9-2). More recently, Matryoshka Representation Learning [\(Kusupati et al.,](#page-10-0) [2022\)](#page-10-0) introduced a novel technique for creating flexible, nested substructures within embeddings, enabling fine-grained control over the trade-off between latency and accuracy. This concept has since been extended to various modalities and applications, including large language models [\(OpenAI,](#page-11-1) [2024;](#page-11-1) [Nussbaum](#page-11-3) [et al.,](#page-11-3) [2024;](#page-11-3) [Yu et al.,](#page-12-0) [2024\)](#page-12-0), diffusion models [\(Gu et al.,](#page-9-3) [2023\)](#page-9-3), and multimodal models [\(Cai et al.,](#page-9-4) [2024;](#page-9-4) [Hu et al.,](#page-10-8) [2024\)](#page-10-8). Other works have further explored token reduction in image and video processing [\(Yan et al.,](#page-11-4) [2024b;](#page-11-4) [Duggal](#page-9-5) [et al.,](#page-9-5) [2024\)](#page-9-5).

Despite these advances, existing methods often do not fully harness the capabilities of large foundation models, highlighting the need for more effective compression strategies. Our proposed *sparse compression* methodology addresses this gap by providing a lightweight, plug-and-play solution that can be readily applied on top of any foundation model – significantly reducing computational overhead while preserving representational quality.

Sparse Coding. Sparse coding serves as a powerful technique for compressing high-dimensional signals and extracting salient features [\(Wright et al.,](#page-11-5) [2010;](#page-11-5) [Zhang et al.,](#page-12-1) [2015\)](#page-12-1), with learned sparse representations often providing additional computational benefits and robustness [\(You et al.,](#page-11-6) [2024;](#page-11-6) [2025\)](#page-11-7). Prior work has induced sparsity through modifications to model design or training protocols, including modifications to attention mechanisms [\(Correia et al.,](#page-9-6) [2019\)](#page-9-6), applying Bayesian standard Gamma priors [\(Duan et al.,](#page-9-7) [2024a](#page-9-7)[;b;](#page-9-8) [Hu et al.,](#page-10-9) [2025\)](#page-10-9), incorporating discrete sparse concept layers [\(Koh et al.,](#page-10-10) [2020;](#page-10-10) [Xie et al.,](#page-11-8) [2025\)](#page-11-8), and promoting sparse activations in large language models [\(Mirzadeh](#page-11-9) [et al.,](#page-11-9) [2023;](#page-11-9) [Zhang et al.,](#page-12-2) [2024\)](#page-12-2). However, training state-ofthe-art foundation models from scratch under these sparsity constraints has proven challenging [\(Elhage et al.,](#page-9-9) [2022\)](#page-9-9), limiting their current applicability.

Meanwhile, Sparse Autoencoders have achieved notable success in improving the interpretability of foundation models [\(Cunningham et al.,](#page-9-0) [2023;](#page-9-0) [Yan et al.,](#page-11-10) [2024a\)](#page-11-10), primarily because they uncover semantic information by mapping high-dimensional data onto lower-dimensional subspaces [\(Cunningham et al.,](#page-9-0) [2023\)](#page-9-0). Building on these insights – and harnessing the inherent advantages of sparse

coding – we investigate how SAEs can be further developed to learn adaptive representations with high efficiency, expanding their applicability to a wider range of tasks.

#### 3. Method

Our proposed framework, Contrastive Sparse Representation learning (CSR), is illustrated in Figure [2.](#page-3-0) Starting from a pre-trained embedding v ∈ R d , we project it into a sparse representation space R h , selectively activating the most relevant dimensions for adaptive representation learning. We then regularize this hidden space using a reconstructionbased sparse compression loss (Section [3.2.1\)](#page-3-1). Additionally, with theoretical motivations and guarantees provided by [\(Wang et al.,](#page-11-11) [2024\)](#page-11-11), we introduce a non-negative contrastive loss to expand model capacity and feature identifiability. (Section [3.2.2\)](#page-3-2)

#### 3.1. Preliminaries

Problem Formulation. For simplicity, we first introduce our framework in the context of a classification task. Let D<sup>N</sup> db = {(x<sup>i</sup> , yi) N <sup>i</sup>=1} be a training dataset of size N, where x<sup>i</sup> ∈ X are an input sample and y<sup>i</sup> ∈ Y<sup>L</sup> are corresponding labels with L classes, We obtain an embedding v = f(x; θ<sup>f</sup> ) : X → <sup>R</sup> d . We can apply exact ℓ2-based knearest neighbor (KNN) search for classification, which has O(dN) complexity. In practice, KNN often employs highdimensional embeddings (*i.e.* d = 4096) to achieve stronger performance, but at the cost of increased computational latency. Our goal is to learn a more compact representation v ′ ∈ <sup>R</sup> <sup>m</sup> (where m ≪ d) that balances accuracy and query latency. This shortened embedding can also benefit other downstream tasks such as retrieval and clustering.

Matryoshka Representation Learning (MRL). MRL [\(Kusupati et al.,](#page-10-0) [2022\)](#page-10-0) simultaneously optimizes embeddings at multiple dimensions, as illustrated in Figure [2,](#page-3-0) to produce representations of variable size. Specifically, let M be a set of target embedding sizes. For each m ∈ M, MRL applies an additional linear classifier to the first m dimensions of the embedding vector, v1:<sup>m</sup> ∈ <sup>R</sup> <sup>m</sup>. This design ensures each truncated representation is explicitly trained via the final loss. Formally, the MRL objective is

$$\mathcal{L}_{\text{MRL}} = \sum_{m \in \mathcal{M}} c_m \mathcal{L}_{\text{CE}} \left( \mathbf{W}^{(m)} \cdot f(x_i; \theta_f)_{1:m}; y_i \right), \quad (1)$$

where W(m) ∈ <sup>R</sup> <sup>L</sup>×<sup>m</sup> is the linear classifier weights corresponding to v1:m. Each loss term is scaled by a non-negative coefficient {c<sup>m</sup> ≥ 0}m∈M. The multi-granularity arises from selecting dimensions in M, whose size is constrained to at most log(d), that is, |M| ≤ ⌊log(d)⌋. For example, [Kusupati et al.](#page-10-0) [\(2022\)](#page-10-0) choose M = {8, 16, . . . , 1024} as the nesting dimensions.

#### 3.2. Contrastive Sparse Representation

As discussed in Section [1,](#page-0-1) MRL (Equation [1\)](#page-2-0) faces two key constraints: it requires (full) training of the backbone parameters θ<sup>f</sup> and its performance often deteriorates a lot under small hidden dimensions. To overcome these limitations, we propose a new methodology that relies on the computational efficiency of *sparse vectors* for efficient retrieval. The method, named Contrastive Sparse Representation (CSR), learns a simple one-layer sparse module on top of *frozen* pretrained embedding models (with full representation size, *e.g.*, 2048) that maps dense embeddings to highly sparse embeddings with a small number of active (i.e., non-zero) dimensions (*e.g.*, 32). As a result, CSR not only saves a lot training effort, but also allow using sparse matrix multiplication at inference time to accelerate retrieval significantly. Below, we outline how we train the CSR module through a combination of sparse autoencoding (Section [3.2.1\)](#page-3-1) and sparse contrastive learning (Section [3.2.2\)](#page-3-2).

#### 3.2.1. SPARSE AUTOENCODING

Autoencoding is a long-standing unsupervised objective that extract salient features that could preserve the original data the most a reconstruction objective. In CSR, we aim at compressing dense embeddings to sparse vectors for efficient sparse retrieval while retaining most of the useful information. To achieve this goal, we adopt sparse autoencoders due to their ability to scale with large data and restore feature semantics [\(Cunningham et al.,](#page-9-0) [2023;](#page-9-0) [Yan et al.,](#page-11-10) [2024a\)](#page-11-10).

Sparse Autoencoders (SAEs). SAEs [\(Makhzani & Frey,](#page-10-11) [2013;](#page-10-11) [Cunningham et al.,](#page-9-0) [2023;](#page-9-0) [Gao et al.,](#page-9-1) [2024;](#page-9-1) [Yan et al.,](#page-11-10) [2024a\)](#page-11-10) aim to extract a sparse representation z<sup>k</sup> by learning to reconstruct the dense feature from zk. Specifically, given a pretrained dense embedding v := f(x) ∈ <sup>R</sup> d as the input, we apply a TopK SAE [\(Gao et al.,](#page-9-1) [2024\)](#page-9-1) with the following autoencoding process:

$$z_k := \sigma^+(\text{TopK}(\mathbf{W}_{\text{enc}}(f(x) - \mathbf{b}_{\text{pre}}) + \mathbf{b}_{\text{enc}})), \quad (2)$$

$$\widehat{f(x)}_k := \mathbf{W}_{\text{dec}} z_k + \mathbf{b}_{\text{pre}}, \quad (3)$$

where Wenc ∈ <sup>R</sup> h×d and Wdec ∈ <sup>R</sup> d×h are the encoder and decoder weight matrices, respectively; benc ∈ <sup>R</sup> h and bpre ∈ <sup>R</sup> d are bias terms. The function σ <sup>+</sup>(·) = max(0, ·) denotes the ReLU activation, and TopK(·) selects the top k largest elements of the input, zeroing out the rest (as in [Gao](#page-9-1) [et al.](#page-9-1) [\(2024\)](#page-9-1)). As a result, the latent z<sup>k</sup> is always a sparse non-negative vector with k active dimensions. This enables direct control over the accuracy–compute trade-off in downstream tasks, particularly under resource-constrained conditions. We formulate the loss function as follows:

$$\mathcal{L}(k) = \left\| f(x) - \widehat{f(x)}_k \right\|_2^2. \quad (4)$$

![](_page_3_Diagram_1.jpeg)

Figure 2. Overview of our proposed CSR framework. As a posttraining approach, CSR differs fundamentally from MRL by projecting embeddings into a higher-dimensional space and dynamically activating only the TopK dimensions for a compact representation. The hidden space is constrained by both reconstruction and contrastive losses, which together enhance the capacity of the sparse representation while preserving computational efficiency.

Moreover, as the hidden dimension h increases, we empirically observe that an increasing number of latent dimensions remain inactive during training – a phenomenon referred to as "dead latents". A large proportion of dead latents reduces the model's capacity and leads to performance degradation [\(Lu et al.,](#page-10-12) [2019;](#page-10-12) [Templeton et al.,](#page-11-12) [2024\)](#page-11-12). To mitigate this issue, an auxiliary loss Laux and Multi-TopK losses are proposed to mitigate this problem. The overall reconstruction loss is

$$\mathcal{L}_{\text{recon}} = \mathcal{L}(k) + \mathcal{L}(4k)/8 + \beta \mathcal{L}_{\text{aux}}, \quad (5)$$

where Laux = ||e − eˆ||<sup>2</sup> 2 , <sup>e</sup> <sup>=</sup> <sup>f</sup>(x) <sup>−</sup> <sup>f</sup>d(x), and <sup>e</sup><sup>ˆ</sup> <sup>=</sup> <sup>W</sup>dec<sup>z</sup> is the reconstruction using the top-kaux dead latents. By default, we set kaux = 512 and β = 1/32, following the setting in [Gao et al.](#page-9-1) [\(2024\)](#page-9-1). We also offer dynamic sparsity selection, with k ranging from 8 to 256, to accommodate different tasks across various modalities.

#### 3.2.2. SPARSE CONTRASTIVE LEARNING

Furthermore, we consider to incorporate an additional *sparse contrastive loss* to the representations' discriminative power. Most state-of-the-art embedding models today, *e.g.*, CLIP [\(Radford et al.,](#page-11-2) [2021\)](#page-11-2), follow a contrastive learning paradigm, which that learns to use the embeddings to distinguish between positive and negative pairs. And it applies to both supervised and unsupervised settings [\(Huang et al.,](#page-10-13) [2024\)](#page-10-13).

The loss objective can be formulated as:

$$\mathcal{L}_{\text{cl}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(z_i^T z_i)}{\exp(z_i^T z_i) + \sum_{j \neq i}^B \exp(z_i^T z_j)}. \quad (6)$$

By leveraging the non-negative nature of latent variables zi in sparse autoencoders, Equation [6](#page-4-0) can be viewed as a variant of the Non-negative Contrastive Loss (NCL) proposed in [Wang et al.](#page-11-11) [\(2024\)](#page-11-11). This interpretation enables us to draw on the theoretical guarantees of NCL, as stated in the following theorem:

Theorem 5 [\(Wang et al.](#page-11-11) [\(2024\)](#page-11-11)). *Under mild conditions, the solution* ϕ(x) *is the unique solution to the NCL objective. As a result, NCL features are identifiable and disentangled.*

Theoretically guaranteed by Theorem [5,](#page-4-1) the model is encouraged to utilize a larger number of latent dimensions to reconstruct the input data. This behavior is empirically demonstrated in Figure [6,](#page-5-0) where we observe a reduction in "dead" dimensions compared to vanilla SAE approaches.

#### 3.2.3. OVERALL TRAINING OBJECTIVE

At last, we optimize the sparse module through a combination of sparse autoencoding Lrecon and sparse contrastive learning Lncl. The former incentivizes the model to preserve original semantic information in the original representation, while the latter shapes the sparse representation to be better at discriminative tasks. The final training objective of our Contrastive Sparse Representation (CSR) method is formulated as:

$$\mathcal{L}_{\text{CSR}} = \mathcal{L}_{\text{recon}} + \gamma \mathcal{L}_{\text{ncl}}. \quad (7)$$

Here, γ is a hyperparameter that balances the two loss components and is set to 1 by default.

# 4. Empirical Analysis

In this section, we conduct a careful study on the empirical performance of the proposed CSR. All experiments in this section are conducted on ImageNet, using 1-NN accuracy [\(Johnson et al.,](#page-10-14) [2019\)](#page-10-14) as the evaluation metric. By default, we set the hidden dimension h of CSR to be h = 4d, where d is the dimension of the pretrained dense embeddings, and set the default active dimension to k = 32.

For a fair and intuitive comparison of MRL and CSR, First, we adopt the notion of *active dimension* as a surrogate metric to benchmark the retrieval time under dense (MRLtype) and sparse (CSR-type) embeddings. For example, "Active Dim = 8" denotes either a length-8 dense embedding (MRL) or a sparse embedding with TopK (k = 8) activation (CSR). Notably, we choose it because dense and sparse matrix multiplication have the same computation complexity under the same active dimension k, *i.e.*, O(k).

![](_page_4_Figure_2.jpeg)

(a) Effect of hidden dim (b) Effect of database size

Figure 3. Comparision of retrieval time based on different factors. (a) Fixed-scale scenario (1M database): Both methods achieve performance sweet spots at TopK=16, with CSR exhibiting 2.1× speedup over dense embeddings when sparsity exceeds 80%. (b) Scaling scenario (h = 8192): CSR exhibits increasingly efficient scalability from 0.5M to 10M, with performance gains accelerating at larger scales. This makes it highly practical for real-world applications involving millions of entries.

In Section [4.1,](#page-4-2) we further carefully benchmark them in practice and find that the two indeed have similar retrieval time, and sparse ones can be even slightly faster under small k.

To account for variations in retrieval time due to sample size, we establish a standardized benchmarking protocol (denoted as T ) to measure retrieval latency by default. Specifically, to simulate large-scale retrieval scenarios, we report the average retrieval time for 512 queries over an ImageNetscale database containing 1.3 million entries (equivalent to the size of the ImageNet training set). For CSR, we use a default hidden dimension of h = 16,384 and an active dimension of k = 32. All experiments are conducted in a consistent GPU environment using PyTorch [\(Paszke et al.,](#page-11-13) [2019\)](#page-11-13). To facilitate comparison, we also report the relative retrieval time of each method by normalizing it against the retrieval time of CSR under the default setup. Additional implementation details can be found in Section [E.3.](#page-18-0)

#### 4.1. Retrieval Time Comparison with MRL

In this section, we benchmark the retrieval time of MRL and CSR under the same active dimension k and analyze the impact of hidden dimension R h , database size N and sparsity k.

*(i) Active dimension.* Figure [3\(](#page-4-3)a) shows retrieval time under varying hidden dimensions, with database size fixed. We can see that the retrieval time of CSR (*i.e.*, sparse multiplication) and MRL (*i.e.*, dense multiplication) both grow with large k and remain relatively on the same level. And for smaller k, CSR shows a clearer advantage over MRL. Although CSR and MRL have similar theoretical complexity O(dk), their actual runtimes are affected by backend im-

![](_page_5_Figure_2.jpeg)

Figure 4. Performance of CSR under different sparsity levels with different sizes of backbone models. CSR achieves higher fidelity at greater sparsity levels when applied to larger backbone models (which provide better base performance), observed consistently in both ViT and ResNet architectures.

plementations. For instance, cuBLAS (used for dense ops) is highly optimized but has high launch overhead, while cuSPARSE (used for CSR) is lighter but less optimized for small k. Interestingly, we can observe that for sparse embeddings, retrieval time decreases as hidden dimension h increases. This suggests notable benefit of CSR that it can use higher latent dimensions for better expressivity while attaining faster retrieval. On the contrary, MRL with higher dense dimensions always has slower retrieval. We elaborate potential reasons on this distinction at Appendix [E.4.](#page-18-1)

*(ii) Database size.* Figure [3\(](#page-4-3)b) shows that CSR demonstrates superior scalability as the database size N increases from 0.5M to 10M. The relative efficiency gain becomes more pronounced with larger datasets, underscoring the practicality of sparse embeddings in real-world retrieval scenarios.

#### 4.2. Effect of Backbone Size

Experiment Setup. We examine fidelity versus backbone size (with different input dimension R d ), and sparsity, using fixed hidden dimension R h across architectures. For ViT, we use ViT-S/16 (d = 384) and ViT-L/16 (d = 1024) with h = 4096. For ResNet, we test RN18 (d = 512) and RN50 (d = 2048) with h = 8192. A more detailed experiment setup is provided in Section [E.1.](#page-17-0)

Analysis. Figure [4](#page-5-1) demonstrates that a larger backbone with higher input embedding dimensions improves model fidelity at equal sparsity levels. This insight is particularly significant, as larger embedding sizes generally encode richer information, thereby achieving better downstream performance. By leveraging these high-dimensional embeddings, our approach more effectively retains essential features and relationships within the data.

![](_page_5_Figure_1.jpeg)

Figure 5. Performance of CSR under different hidden dimensions and different types of backbone models (ResNet-50 (convolution) and ViT-L (Transformers)). CSR exhibits a reverse U-shape across different models and hidden dimensions. CSR's performance peaks at h = 4d (d is the input dimension size) but degrades beyond this, especially with higher sparsity.

![](_page_5_Figure_5.jpeg)

Figure 6. Comparison of dead latent fractions across loss combinations under varying sparsity constraints. Results show that even equipped with Lauxk and Multiple-TopK at extreme sparsity levels (*i.e.*, k = 8, 16, 32). CSR further alleviates this issue, outperforming baselines and demonstrating its robustness.

#### 4.3. Effect of Hidden Representation Dimension R h

Experiment Setup. We explore how hidden dimension R h effects on our model, we use ViT-Large and ResNet50 as pre-trained backbones, sweeping h from d to 16d while keeping all other parameters at their default values. Additional implementation details are provided in Section [E.2.](#page-18-2)

Analysis. Figure [5](#page-5-2) compares model performance across different hidden dimensions under varying sparsity constraints. Notably, a shift in the performance trend occurs at h = 4d. When h < 4d, performance gradually improves with increasing hidden dimension, reaching its peak at h = 4d. However, beyond this point, further increases in h lead to performance degradation, particularly under higher sparsity constraints. This trend aligns with the observations of [Gao et al.](#page-9-1) [\(2024\)](#page-9-1), which suggest that excessively large

![](_page_6_Figure_1.jpeg)

Figure 7. (Left & Middle): Results of ImageNet Top-1 accuracy (a) and 1-NN accuracy (b) across active dimensions under the same pretrained ResNet-50 backbone used in [Kusupati et al.](#page-10-0) [\(2022\)](#page-10-0). We can see that while MRL trains the whole network and CSR only uses frozen embeddings, CSR still performs consistently better across all embedding sizes and has significant margins beyond 20% at lower active dimensions (the region that yields the largest efficiency gains). (Right): *Comparison of text embedding methods at similar retrieval cost.* For CSR, we use k = 32 by default. For each task, the model is trained on three datasets and evaluated on three unseen datasets. The text embeddings learned by CSR outperformed other MRL-based baselines by significant margins across different natural language tasks at much lower training cost.

hidden dimensions may not be fully utilized, ultimately diminishing model performance. A similar pattern is observed in ResNet. Based on these findings, we set h = 4d as the default configuration for all subsequent experiments unless otherwise specified.

#### 4.4. Effect of Different Losses

Experiment Setup. We investigate how different loss functions affect model capacity, particularly in addressing the dead latent problem discussed in Section [3.2.1,](#page-3-1) using RN50 backbone with h = 4d. Other parameters are set at their default values.

Analysis. Figure [6](#page-5-0) illustrates the impact of different loss functions on model capacity. The na¨ıve SAE suffers from severe dead latents, while the inclusion of an auxiliary loss Laux and the multi-TopK loss partially mitigates this issue. Introducing a non-negative contrastive loss (NCL) further alleviates the problem, particularly at extreme sparsity levels (*e.g.*, k = 8, 16, 32). Empirical results validate the effectiveness of Theorem [5,](#page-4-1) demonstrating that representation learning with NCL promotes more orthogonal and disentangled features. This, in turn, increases the number of active dimensions and enhances overall model performance.

# 5. Benchmark Results and Analysis

We evaluated the effectiveness of our proposed CSR framework across three mainstream representation modalities: vision, language, and vision+language. For vision representation (see Section [5.1\)](#page-6-0), we conduct image classification on ImageNet-1K and evaluate performance using 1-NN accuracy, following [Kusupati et al.](#page-10-0) [\(2022\)](#page-10-0). For language

representation (see Section [5.2\)](#page-7-0), we focus on three primary tasks: text classification, text clustering, and text retrieval on the MTEB benchmark [\(Muennighoff et al.,](#page-11-14) [2022\)](#page-11-14). For multimodal representation (see Section [5.3\)](#page-7-1), we report both in-distribution and zero-shot cross-modal retrieval performance on two widely-used datasets: MS COCO [\(Lin et al.,](#page-10-15) [2014\)](#page-10-15) and Flickr30K [\(Young et al.,](#page-12-3) [2014\)](#page-12-3). Through these experiments, we aim to provide a holistic understanding of the capabilities of our proposed framework.

#### 5.1. Vision Representation Comparision

Baselines We compare our proposed method with the following baseline approaches. 1) MRL/MRL-E [\(Kusupati](#page-10-0) [et al.,](#page-10-0) [2022\)](#page-10-0): RN50 model where the fully connected layer is replaced by multiple (MRL) or a single (MRL-E) classification head(s) that take truncated input dimensions (*e.g.*, only the first 8 of the original 2048 dimensions). 2) SVD: We performed a low-rank approximation of the 1000-way classification layer of RN50, with rank = 1000. 3) Rand-LP: We compared against a linear classifier fit on randomly selected features [\(He et al.,](#page-9-10) [2020\)](#page-9-10). 4) Rand-FS: We randomly selected features extracted from RN50 for 1-NN classification.

Experiment Setup. We evaluate 1-NN accuracy and Top-1 accuracy on ImageNet1k classification, following [Kusu](#page-10-0)[pati et al.](#page-10-0) [\(2022\)](#page-10-0). For fair comparison, we used the same RN50 backbone weights as MRL (denoted as FF2048 in the original work) and trained CSR on its ImageNet1k encoded embeddings. For further implementation details, please refer to Section [B.](#page-14-0)

Table 1. Performance and efficiency of text embeddings on three natural language tasks: classification, clustering, and retrieval. We use NV-Embed-V2 as our pre-trained model, and present its performance in the first line of the table in gray. We analyze *Dataset-Specific Evaluation* results along two key dimensions: (1) Relative Retrieval Time under matched performance and ii) performance under matched retrieval efficiency. Under matched performance, CSR achieves a remarkable 61× speedup, while under matched retrieval efficiency, it improves performance by 15%, demonstrating its superior balance between speed and accuracy. The maximum values are indicated in bold, while the second-highest values are underlined. Relative Retrieval Time is calculated follows the definition in Section [4.](#page-4-4)

| Category Model              | Active Dim   | Retrieval Time | MTOPIntent | Text Classification Top-1 Acc Banking77 | (%) ↑ TweetSentiment | BiorxivP2P | Text Clustering Top-1 Acc BiorxivS2S | (%) ↑ TwentyNews | FiQA2018 | Text Retrieval NDCG@10 NFCorpus | (%) ↑ SciFACT |
|-----------------------------|--------------|----------------|------------|-----------------------------------------|----------------------|------------|--------------------------------------|------------------|----------|---------------------------------|---------------|
| Full Rep NV-Embed-V2        | 4096         | 37.6           | 93.58      | 92.20                                   | 79.73                | 53.61      | 49.60                                | 64.82            | 62.65    | 43.97                           | 77.93         |
| Stella-1.5B-v5              | 256          | 2.6            | 90.45      | 86.14                                   | 76.75                | 50.81      | 46.42                                | 60.07            | 55.59    | 36.97                           | 77.48         |
| Jina-V3                     | 256          | 2.8            | 78.81      | 84.08                                   | 73.81                | 38.14      | 34.39                                | 51.96            | 55.73    | 36.63                           | 66.63         |
| Nomic-Embed-V1.5            | 256          | 2.7            | 72.47      | 83.69                                   | 59.20                | 38.19      | 31.83                                | 48.56            | 35.00    | 32.54                           | 68.24         |
| Gecko-Embed-004(Google)     | 256          | 2.4            | 77.82      | 86.01                                   | 72.97                | 36.28      | 33.09                                | 50.60            | 55.54    | 37.81                           | 70.86         |
| Text-Embed-3-L MRL          | (OpenAI) 256 | 2.8            | 70.45      | 83.19                                   | 58.98                | 35.43      | 33.86                                | 54.24            | 50.33    | 37.94                           | 73.10         |
| Arctic-Embed-L-V2           | 256          | 2.6            | 67.69      | 80.99                                   | 59.06                | 34.25      | 34.07                                | 30.06            | 44.69    | 35.02                           | 69.51         |
| M2V-Base-Glove              | 256          | 2.4            | 59.26      | 72.39                                   | 50.02                | 32.26      | 22.34                                | 25.38            | 11.82    | 23.15                           | 50.66         |
| Jina-V3                     | 64           | 1.2            | 68.12      | 67.98                                   | 71.18                | 36.89      | 33.57                                | 50.22            | 44.18    | 33.66                           | 68.84         |
| Nomic-Embed-V1.5            | 64           | 1.6            | 62.77      | 80.63                                   | 55.23                | 34.81      | 44.61                                | 48.06            | 10.22    | 18.96                           | 36,55         |
| Potion-Base-2M              | 64           | 1.4            | 42.50      | 65.17                                   | 52.52                | 25.78      | 14.94                                | 27.07            | 32.08    | 30.72                           | 64.28         |
| Sparse SAE (w/ NV-Embed-V2) | 32           | 1.0            | 87.43      | 88.11                                   | 75.19                | 51.02      | 48.68                                | 58.63            | 49.18    | 35.14                           | 66.04         |
| CSR (w/ NV-Embed-V2)        | 32           | 1.0            | 89.86      | 91.02                                   | 78.55                | 53.49      | 49.13                                | 63.05            | 57.54    | 38.06                           | 71.17         |

Analysis. Figure [7\(](#page-6-1)a) and (b) illustrate the comparison of learned representation quality through the Top-1 and 1- NN classification accuracy of RN50 models trained and evaluated on ImageNet-1K. For linear probing results (Figure [7\(](#page-6-1)a)), reconstruction-based sparse compression methods (CSR & SAE) outperform MRL-LP (both linear probing methods) by a large margin and also surpass MRL/MRL-E (train from scratch) in lower active dim (k < 128). Furthermore, Figure [7\(](#page-6-1)a) demonstrates the superior representation quality learned by CSR, which consistently outperforms MRL across various active dimensions. CSR also surpass traditional post-hoc compression techniques (*e.g.*, SVD) and linear probes on random features by increasing the overall model total capacity while keeping active dimensions for each sample unchanged, as discussed in Section [1](#page-0-1) and Section [3.2.1.](#page-3-1) This enhanced capability allows CSR to maintain remarkable robustness, even under extrem sparsity where k = 2, 4, 8. These results highlight that the proposed CSR design can effectively compress pre-trained embeddings while leveraging the natural benefits of sparse matrix multiplication. More detailed experimental results can be found in Section [4.](#page-15-0)

#### 5.2. Text Representation Comparision

Experiment Setup. We assessed CSR on three key tasks from the MTEB benchmark, testing it across six datasets for each task. In detail, we conduct evaluations in two distinct settings: *Dataset-Specific Evaluation*, where CSR is trained and tested on different splits of the same dataset to ensure consistency, and *Task-Specific Evaluation*, where CSR is trained on one dataset and evaluated on unseen datasets within the same task to rigorously assess its generalization capabilities. We choose NV-Embed-V2 [\(Lee et al.,](#page-10-5) [2024a\)](#page-10-5) as our pre-trained model and present its performance in gray. For further experimental details, please refer to Section [C.](#page-15-1) To improve readability, we refer to CSR-K as a model with the TopK activations and so as SAE.

Analysis Table [1](#page-7-2) demonstrates the performance of CSR and baseline models across multiple tasks and datasets. CSR not only maintains the strong performance of the pre-trained model but also surpasses baselines under varying resource constraints. Taking text classification as an example, CSR achieves a 15% accuracy improvement at matched computational cost (*i.e.*, with retrieval times comparable to Jina-V3-64 and Nomic-Embed-V1.5-64) while attaining a 61x speedup when matched for performance (*i.e.*, compared to NV-Embed-V2). The results underscore CSR 's exceptional ability to maintain an optimal speed-accuracy trade-off - a critical requirement for practical deployment in large-scale retrieval systems. We further evaluate the generalization capability of CSR (with k = 32) on three unseen datasets per task, as shown in Figure [7\(](#page-6-1)c). The results demonstrate that sparse representations yield more robust performance compared to dense alternatives at same activation dimensions. These results underscore the efficacy and versatility of CSR , demonstrating its strong potential for real-world applications.

#### 5.3. MultiModal Representation Comparision

Experiment Setup. We evaluated our methods on multimodal retrieval tasks using the ViT-B-16 backbone, testing both in-distribution and zero-shot cross-modal retrieval on MS COCO [\(Lin et al.,](#page-10-15) [2014\)](#page-10-15) and Flickr30K [\(Young et al.,](#page-12-3) [2014\)](#page-12-3) datasets. For baselines, we fine-tuned MRL on these datasets (using CC3M [\(Changpinyo et al.,](#page-9-11) [2021\)](#page-9-11) for zeroshot training), following standard MRL training protocols [\(Kusupati et al.,](#page-10-0) [2022\)](#page-10-0). The performance of our backbone,

Table 2. Comparison of different methods on multi-modal retrieval tasks using two benchmark datasets, MS COCO and Flickr30k, evaluated under both in-distribution and zero-shot settings, with Recall@5 (%) as the performance metric. We use ViT-B/16 as our pre-trained model, and present its performance in the first line of the table in gray. For Zero-Shot setting, CSR is first trained on a large-scale scale, dataset-CC3M and evaluated on downstream tasks. CSR (plug-and-play) consistently outperforms ViT-B-16-MRL (fully fine-tuned) in various tasks with significant training efficiency.

| Method       | Active Dim | Trainable Parms | MS I2T | COCO T2I | In-Distribution Flickr30K I2T | T2I   | MS I2T | Zero-Shot COCO T2I | Flickr30K I2T | T2I   |
|--------------|------------|-----------------|--------|----------|-------------------------------|-------|--------|--------------------|---------------|-------|
| ViT-B-16     | 512        | 86M             | 74.42  | 86.47    | 91.92                         | 97.79 | 69.23  | 83.03              | 89.82         | 97.70 |
| ViT-B-16-MRL |            | 86M             | 67.12  | 77.53    | 80.41                         | 89.89 | 56.90  | 65.82              | 80.94         | 89.20 |
| SAE          |            | 1.1M            | 71.21  | 82.58    | 87.76                         | 95.59 | 58.22  | 67.40              | 82.44         | 86.19 |
| CSR          |            | 1.1M            | 71.41  | 83.49    | 87.98                         | 96.79 | 61.85  | 70.14              | 85.22         | 91.10 |
| ViT-B-16-MRL |            | 86M             | 64.19  | 73.02    | 77.56                         | 87.80 | 53.63  | 61.16              | 77.67         | 85.10 |
| SAE          |            | 1.1M            | 64.67  | 76.70    | 81.40                         | 91.20 | 53.20  | 63.02              | 77.54         | 85.19 |
| CSR          |            | 1.1M            | 69.34  | 81.04    | 84.05                         | 93.00 | 54.37  | 68.04              | 78.08         | 88.09 |
| ViT-B-16-MRL |            | 86M             | 62.61  | 72.43    | 74.22                         | 84.79 | 47.47  | 54.42              | 71.16         | 79.00 |
| SAE          |            | 1.1M            | 56.30  | 69.45    | 70.58                         | 81.30 | 44.48  | 53.56              | 69.58         | 82.29 |
| CSR          |            | 1.1M            | 62.75  | 78.10    | 76.44                         | 88.50 | 48.61  | 61.90              | 73.04         | 84.10 |

using the same fine-tuning procedure, is shown in gray. During training, both SAE and CSR leverage a shared sparse embedding layer for images and text. Additional experimental setup and implementation details are provided in Section [D.](#page-16-0)

Analysis. Table [2](#page-8-0) presents the multimodal retrieval task results across different methods and settings. In general, reconstruction-based methods exhibit relatively low performance degradation on both datasets. Compared to the MRL method, CSR achieves average performance gains of 4.6% and 6.8% on I2T retrieval, and 10.3% and 6.5% on T2I retrieval across the two datasets in In-Distribution Evaluation. Besides, under zero-shot scenario, CSR also surpasses MRL by 3.2% and 3.3% on I2T, and 9.2% and 3.9% on T2I, respectively. Notably, these results demonstrate CSR's potential to handle large-scale datasets (*e.g.*, CC3M-3M images, compared to ImageNet's 1M and MS COCO's 0.3M), confirming CSR's consistent superiority across various active dimensions and its scalability. SAE experiences more severe performance degradation compared to CSR, which underlines the efficacy of our design in image-text alignment. However, as the sparsity constraint becomes more stringent, the performance gap between CSR and MRL narrows. Upon further investigation, we find that CSR still suffers from the "dead latents" problem even when equipped with advanced mechanisms. Addressing the mitigation of dead latents in the alignment space remains an open challenge, leaving room for future work and study. For a detailed analysis, please refer to Section [D.4.](#page-16-1)

# 6. Conclusion & Discussion

In this paper, we introduce Contrastive Sparse Representation Learning (CSR), a generic learning framework offering a high-fidelity and flexible approach to compress embedding, surpassing existing methods like MRL in various tasks and modalities. We believe CSR paves the way for more efficient and flexible representation learning, especially in scenarios constrained by memory, latency or other computational considerations.

Our method, CSR, is orthogonal to existing acceleration techniques such as pruning [\(He et al.,](#page-9-12) [2017\)](#page-9-12), quantization [\(Jacob et al.,](#page-10-16) [2018\)](#page-10-16), and distillation [\(Hinton et al.,](#page-9-13) [2015\)](#page-9-13), which primarily target embedding generation. In contrast, CSR optimizes the post-processing stage, enabling complementary speedups with minimal performance trade-off. A current limitation of CSR, shared by other sparsity-based approaches, is the emergence of dead neurons under high sparsity, especially in multimodal settings. While techniques like contrastive loss partially mitigate this (see Figure [6\)](#page-5-0), fully resolving the issue remains an open challenge and direction for future work.

# Acknowledgements

This work was supported in part by the National Natural Science Foundation of China under Grant U21B2006; in part by the Fundamental Research Funds for the Central Universities QTZX24003 and QTZX23018; in part by the 111 Project under Grant B18039; and in part by Shaanxi Youth Innovation Team Project.

- Impact Statement This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here. References Model2vec: Turn any sentence transformer into a small fast model, 2024. Boteva, V., Gholipour, D., Sokolov, A., and Riezler,
- S. A full-text learning to rank dataset for medical information retrieval. 2016. URL [http:](http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf) [//www.cl.uni-heidelberg.de/˜riezler/](http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf) [publications/papers/ECIR2016.pdf](http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf). Cai, M., Yang, J., Gao, J., and Lee, Y. J. Matryoshka multimodal models. *arXiv preprint arXiv:2405.17430*, 2024. Casanueva, I., Temcinas, T., Gerz, D., Henderson, M., and ˇ Vulic, I. Efficient intent detection with dual sentence ´ encoders. *arXiv preprint arXiv:2003.04807*, 2020. Changpinyo, S., Sharma, P., Ding, N., and Soricut, R. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 3558–3568, 2021. Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., and Jitsev, J. Reproducible scaling laws for contrastive language-image learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 2818–2829, 2023. Correia, G. M., Niculae, V., and Martins, A. F. Adaptively sparse transformers. *arXiv preprint arXiv:1909.00015*, 2019. Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L. Sparse autoencoders find highly interpretable features in language models. *arXiv preprint arXiv:2309.08600*, 2023. Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei,
- L. Imagenet: A large-scale hierarchical image database. In *2009 IEEE conference on computer vision and pattern recognition*, pp. 248–255. Ieee, 2009. Duan, Z., Wen, T., Wang, M., Chen, B., and Zhou, M. A non-negative vae: the generalized gamma belief network. *arXiv preprint arXiv:2408.03388*, 2024a. Duan, Z., Wen, T., Wang, Y., Zhu, C., Chen, B., and Zhou, M. Contrastive factor analysis. *arXiv preprint arXiv:2407.21740*, 2024b. Duggal, S., Isola, P., Torralba, A., and Freeman, W. T. Adaptive length image tokenization via recurrent allocation. *arXiv preprint arXiv:2411.02393*, 2024. Elhage, N., Hume, T., Olsson, C., Nanda, N., Henighan, T., Johnston, S., ElShowk, S., Joseph, N., DasSarma, N., Mann, B., Hernandez, D., Askell, A., Ndousse, K., Jones, A., Drain, D., Chen, A., Bai, Y., Ganguli, D., Lovitt, L., Hatfield-Dodds, Z., Kernion, J., Conerly, T., Kravec, S., Fort, S., Kadavath, S., Jacobson, J., Tran-Johnson, E., Kaplan, J., Clark, J., Brown, T., McCandlish, S., Amodei, D., and Olah, C. Softmax linear units. *Transformer Circuits Thread*, 2022. https://transformercircuits.pub/2022/solu/index.html. Fan, A., Grave, E., and Joulin, A. Reducing transformer depth on demand with structured dropout. *arXiv preprint arXiv:1909.11556*, 2019. FitzGerald, J., Hench, C., Peris, C., Mackie, S., Rottmann, K., Sanchez, A., Nash, A., Urbach, L., Kakarala, V., Singh, R., et al. Massive: A 1m-example multilingual natural language understanding dataset with 51 typologically-diverse languages. *arXiv preprint arXiv:2204.08582*, 2022. Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., and Wu, J. Scaling and evaluating sparse autoencoders. *arXiv preprint arXiv:2406.04093*, 2024. Geigle, G., Reimers, N., Ruckl ¨ e, A., and Gurevych, I. ´ Tweac: transformer with extendable qa agent classifiers. *arXiv preprint arXiv:2104.07081*, 2021. Gu, J., Zhai, S., Zhang, Y., Susskind, J. M., and Jaitly, N. Matryoshka diffusion models. In *The Twelfth International Conference on Learning Representations*, 2023. He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. Momentum contrast for unsupervised visual representation learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 9729–9738, 2020. He, Y., Zhang, X., and Sun, J. Channel pruning for accelerating very deep neural networks. In *Proceedings of the IEEE international conference on computer vision*, pp. 1389–1397, 2017. Hinton, G., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*, 2015.

- Hoogeveen, D., Verspoor, K. M., and Baldwin, T. Cqadupstack: A benchmark data set for community questionanswering research. In *Proceedings of the 20th Australasian Document Computing Symposium (ADCS)*, ADCS '15, pp. 3:1–3:8, New York, NY, USA, 2015. ACM. ISBN 978-1-4503-4040-3. doi: 10.1145/2838931. 2838934. URL [http://doi.acm.org/10.1145/](http://doi.acm.org/10.1145/2838931.2838934) [2838931.2838934](http://doi.acm.org/10.1145/2838931.2838934). Hou, L., Huang, Z., Shang, L., Jiang, X., Chen, X., and Liu, Q. Dynabert: Dynamic bert with adaptive width and depth. *Advances in Neural Information Processing Systems*, 33:9782–9793, 2020. Hu, W., Dou, Z.-Y., Li, L. H., Kamath, A., Peng, N., and Chang, K.-W. Matryoshka query transformer for large vision-language models. *arXiv preprint arXiv:2405.19315*, 2024. Hu, X., Duan, Z., Chen, B., and Zhou, M. Enhancing uncertainty estimation and interpretability with bayesian non-negative decision layer. In *The Thirteenth International Conference on Learning Representations*, 2025. URL [https://openreview.net/forum?](https://openreview.net/forum?id=xJXq6FkqEw) [id=xJXq6FkqEw](https://openreview.net/forum?id=xJXq6FkqEw). Huang, J., Hu, Z., Jing, Z., Gao, M., and Wu, Y. Piccolo2: General text embedding with multi-task hybrid loss training. *arXiv preprint arXiv:2405.06932*, 2024. Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., and Kalenichenko, D. Quantization and training of neural networks for efficient integerarithmetic-only inference. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 2704–2713, 2018. Johnson, J., Douze, M., and Jegou, H. Billion-scale similar- ´ ity search with gpus. *IEEE Transactions on Big Data*, 7 (3):535–547, 2019. Kim, G. and Cho, K. Length-adaptive transformer: Train once with length drop, use anytime with search. *arXiv preprint arXiv:2010.07003*, 2020. Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., and Liang, P. Concept bottleneck models. In *International conference on machine learning*, pp. 5338– 5348. PMLR, 2020. Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Ramanujan, V., Howard-Snyder, W., Chen, K., Kakade, S., Jain, P., et al. Matryoshka representation learning. *Advances in Neural Information Processing Systems*, 35:30233–30249, 2022. Leclerc, G., Ilyas, A., Engstrom, L., Park, S. M., Salman, H., and Madry, A. Ffcv: Accelerating training by removing data bottlenecks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 12011–12020, 2023. LeCun, Y., Bengio, Y., and Hinton, G. Deep learning. *nature*, 521(7553):436–444, 2015. Lee, C., Roy, R., Xu, M., Raiman, J., Shoeybi, M., Catanzaro, B., and Ping, W. Nv-embed: Improved techniques for training llms as generalist embedding models. *arXiv preprint arXiv:2405.17428*, 2024a. Lee, H., Battle, A., Raina, R., and Ng, A. Efficient sparse coding algorithms. *Advances in neural information processing systems*, 19, 2006. Lee, J., Dai, Z., Ren, X., Chen, B., Cer, D., Cole, J. R., Hui, K., Boratko, M., Kapadia, R., Ding, W., et al. Gecko: Versatile text embeddings distilled from large language models. *arXiv preprint arXiv:2403.20327*, 2024b. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W.-t., Rockt ¨ aschel, ¨ T., et al. Retrieval-augmented generation for knowledgeintensive nlp tasks. *Advances in Neural Information Processing Systems*, 33:9459–9474, 2020. Li, H., Arora, A., Chen, S., Gupta, A., Gupta, S., and Mehdad, Y. Mtop: A comprehensive multilingual taskoriented semantic parsing benchmark. *arXiv preprint arXiv:2008.09335*, 2020. Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., and Zitnick, C. L. Microsoft coco: ´ Common objects in context. In *Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13*, pp. 740–
  - 755. Springer, 2014. Lu, L., Shin, Y., Su, Y., and Karniadakis, G. E. Dying relu and initialization: Theory and numerical examples. *arXiv preprint arXiv:1903.06733*, 2019. Maggie, Culliton, P., and Chen, W. Tweet sentiment extraction. [https://kaggle.com/competitions/](https://kaggle.com/competitions/tweet-sentiment-extraction) [tweet-sentiment-extraction](https://kaggle.com/competitions/tweet-sentiment-extraction), 2020. Kaggle. Maia, M., Handschuh, S., Freitas, A., Davis, B., McDermott, R., Zarrouk, M., and Balahur, A. Www'18 open challenge: financial opinion mining and question answering. In *Companion proceedings of the the web conference 2018*, pp. 1941–1942, 2018. Makhzani, A. and Frey, B. K-sparse autoencoders. *arXiv preprint arXiv:1312.5663*, 2013.

- McAuley, J. and Leskovec, J. Hidden factors and hidden topics: understanding rating dimensions with review text. In *Proceedings of the 7th ACM conference on Recommender systems*, pp. 165–172, 2013. Mirzadeh, I., Alizadeh, K., Mehta, S., Del Mundo, C. C., Tuzel, O., Samei, G., Rastegari, M., and Farajtabar,
- M. Relu strikes back: Exploiting activation sparsity in large language models. *arXiv preprint arXiv:2310.04564*, 2023. Muennighoff, N., Tazi, N., Magne, L., and Reimers, N. Mteb: Massive text embedding benchmark. *arXiv preprint arXiv:2210.07316*, 2022. Nussbaum, Z., Morris, J. X., Duderstadt, B., and Mulyar, A. Nomic embed: Training a reproducible long context text embedder. *arXiv preprint arXiv:2402.01613*, 2024. OpenAI. New embedding models and api updates. [https://openai.com/index/](https://openai.com/index/new-embedding-models-and-api-updates) [new-embedding-models-and-api-updates](https://openai.com/index/new-embedding-models-and-api-updates), 2024. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. Pytorch: An imperative style, high-performance deep learning library. *Advances in neural information processing systems*, 32, 2019. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pp. 8748–8763. PMLR, 2021. Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., and Chen, Y.-S. Carer: Contextualized affect representations for emotion recognition. In *Proceedings of the 2018 conference on empirical methods in natural language processing*, pp. 3687–3697, 2018. Sturua, S., Mohr, I., Akram, M. K., Gunther, M., Wang, B., ¨ Krimmel, M., Wang, F., Mastrapas, G., Koukounas, A., Wang, N., et al. jina-embeddings-v3: Multilingual embeddings with task lora. *arXiv preprint arXiv:2409.10173*, 2024. Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., Pearce, A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N. L., McDougall, C., MacDiarmid, M., Freeman, C. D., Sumers, T. R., Rees, E., Batson, J., Jermyn, A., Carter, S., Olah, C., and Henighan, T. Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. *Transformer Circuits Thread*, 2024. URL [https:](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) [//transformer-circuits.pub/2024/](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) [scaling-monosemanticity/index.html](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html). Wachsmuth, H., Stede, M., El Baff, R., Al Khatib, K., Skeppstedt, M., and Stein, B. Argumentation synthesis following rhetorical strategies. In *Proceedings of the 27th International Conference on Computational Linguistics*, pp. 3753–3765. Association for Computational Linguistics, 2018a. URL [http://aclweb.](http://aclweb.org/anthology/C18-1318) [org/anthology/C18-1318](http://aclweb.org/anthology/C18-1318). Wachsmuth, H., Syed, S., and Stein, B. Retrieval of the best counterargument without prior topic knowledge. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 241–251, 2018b. Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., and Hajishirzi, H. Fact or fiction: Verifying scientific claims. *arXiv preprint arXiv:2004.14974*, 2020. Wang, Y., Zhang, Q., Guo, Y., and Wang, Y. Non-negative contrastive learning. In *The Twelfth International Conference on Learning Representations*, 2024. Wightman, R. Pytorch image models. [https://github.](https://github.com/huggingface/pytorch-image-models) [com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models), 2019. Wright, J., Ma, Y., Mairal, J., Sapiro, G., Huang, T. S., and Yan, S. Sparse representation for computer vision and pattern recognition. *Proceedings of the IEEE*, 98(6): 1031–1044, 2010. Xie, Y., Zeng, Z., Zhang, H., Ding, Y., Wang, Y., Wang, Z., Chen, B., and Liu, H. Discovering fine-grained visual-concept relations by disentangled optimal transport concept bottleneck models, 2025. URL [https:](https://arxiv.org/abs/2505.07209) [//arxiv.org/abs/2505.07209](https://arxiv.org/abs/2505.07209). Yan, H., He, Y., and Wang, Y. The multi-faceted monosemanticity in multimodal representations. In *Workshop on Responsibly Building the Next Generation of Multimodal Foundational Models*, 2024a. URL [https:](https://openreview.net/forum?id=9NLRpwfLnT) [//openreview.net/forum?id=9NLRpwfLnT](https://openreview.net/forum?id=9NLRpwfLnT). Yan, W., Zaharia, M., Mnih, V., Abbeel, P., Faust, A., and Liu, H. Elastictok: Adaptive tokenization for image and video. *arXiv preprint arXiv:2410.08368*, 2024b. You, C., Mint, Y., Dai, W., Sekhon, J. S., Staib, L., and Duncan, J. S. Calibrating multi-modal representations: A pursuit of group robustness without annotations. In *2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 26140–26150. IEEE, 2024. You, C., Dai, H., Min, Y., Sekhon, J. S., Joshi, S., and Duncan, J. S. The silent majority: Demystifying memorization effect in the presence of spurious correlations, 2025. URL <https://arxiv.org/abs/2501.00961>.

Young, P., Lai, A., Hodosh, M., and Hockenmaier, J. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. *Transactions of the Association for Computational Linguistics*, 2:67–78, 2014. Yu, P., Merrick, L., Nuti, G., and Campos, D. Arctic-embed 2.0: Multilingual retrieval without compromise. *arXiv preprint arXiv:2412.04506*, 2024. Zhang, D., Li, J., Zeng, Z., and Wang, F. Jasper and stella: distillation of sota embedding models, 2025. URL <https://arxiv.org/abs/2412.19048>. Zhang, Z., Xu, Y., Yang, J., Li, X., and Zhang, D. A survey of sparse representation: algorithms and applications. *IEEE access*, 3:490–530, 2015. Zhang, Z., Song, Y., Yu, G., Han, X., Lin, Y., Xiao, C., Song, C., Liu, Z., Mi, Z., and Sun, M. Relu <sup>2</sup> wins: Discovering efficient activation functions for sparse llms. *arXiv preprint arXiv:2402.03804*, 2024.

# A. Datasets

For Image embedding Experiment:

- ImageNet-1K [\(Deng et al.,](#page-9-14) [2009\)](#page-9-14): ImageNet-1K is a large-scale visual database designed to provide researchers with a comprehensive resource for developing and evaluating computer vision models. It contains 1,000 categories, each with a diverse set of images. Specifically, the dataset includes 1,281,167 training images, 50,000 validation images, and 100,000 test images.

## For Text embedding Experiment:

Note that, all datasets mentioned below can be found at MTEB [\(Muennighoff et al.,](#page-11-14) [2022\)](#page-11-14).

- MTOPIntent [\(Li et al.,](#page-10-17) [2020\)](#page-10-17): MTOP is a multilingual dataset introduced in 2021. It comprises 100,000 annotated dialogue sentences across six languages and eleven domains. Designed to serve as a benchmark for multilingual task-oriented semantic parsing, this dataset plays a crucial role in advancing technology in this field.
- Banking77 [\(Casanueva et al.,](#page-9-15) [2020\)](#page-9-15): Dataset composed of online banking queries annotated with their corresponding intents, consisting of 13,083 customer service queries labeled with 77 intents.
- TweetSentimentExtraction [\(Maggie et al.,](#page-10-18) [2020\)](#page-10-18): Dataset from Kag gle competition. Sentiment classification of tweets as neutral, positive or negative.
- MassiveScenario [\(FitzGerald et al.,](#page-9-16) [2022\)](#page-9-16): A collection of Amazon Alexa virtual assistant utterances annotated with the associated intent. For each user utterance the label is a theme among 60 scenarios like 'music', 'weather', etc. This is a multilingual dataset with 51 available languages.
- AmazonReviews [\(McAuley & Leskovec,](#page-11-15) [2013\)](#page-11-15): A collection of Amazonreviews designed to aid research in multilingual text classification. For each review the label is the score given by their view between 0 and 4 (1-5 stars). This is a multilingual dataset with 6 available languages.
- Emotion [\(Saravia et al.,](#page-11-16) [2018\)](#page-11-16): The dataset consists of English Twitter messages categorized into basic emotions, including anger, fear, joy, love, sadness, and surprise.
- ArxivClusteringS2S, BiorxivClusteringS2S, BiorxivClusteringP2P [\(Muennighoff et al.,](#page-11-14) [2022\)](#page-11-14): The BioxivS2S dataset is created using public APIs from bioRxiv. For S2S datasets, the input text is simply the title of the paper, while for P2P the input text is the concatenation of the title and the abstract.
- TwentyNewsgroupsClustering[<sup>2</sup>](#page-13-0) : Clustering of the 20 Newsgroups dataset, given titles of article the goal is to find the newsgroup (20 in total). Contains 10 splits, each with 20 classes, with each split containing between 1,000 and 10,000 titles.
- RedditClusteringP2P [\(Muennighoff et al.,](#page-11-14) [2022\)](#page-11-14): created for MTEB using available data from Reddit posts[<sup>3</sup>](#page-13-1) . The task consists of clustering the concatenation of title+post according to their subreddit. It contains 10 splits, with 10 and 100 clusters per split and 1,000 to 100,000 posts.
- StackExchangeClustering [\(Geigle et al.,](#page-9-17) [2021\)](#page-9-17): Clustering of titles from 121 stack exchanges. Clustering of 25 splits, each with 10-50 classes, and each class with 100-1000 sentences.
- FiQA2018 [\(Maia et al.,](#page-10-19) [2018\)](#page-10-19): A dataset for aspect-based sentiment analysis and opinion-based question answering in finance.
- NFCorpus [\(Boteva et al.,](#page-9-18) [2016\)](#page-9-18): NFCorpus is a full-text English retrieval data set for Medical Information Retrieval. It contains a total of 3,244 natural language queries, with 169,756 automatically extracted relevance judgments for 9,964 medical documents.
- SciFACT [\(Wadden et al.,](#page-11-17) [2020\)](#page-11-17): A dataset of 1.4K expert-written claims, paired with evidence-containing abstracts annotated with veracity labels and rationales.

<sup>2</sup>[https://scikit-learn.org/0.19/datasets/twenty\\_newsgroups.html](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)

<sup>3</sup><https://huggingface.co/datasets/sentence-transformers/reddit-title-body>

- Arguana [\(Wachsmuth et al.,](#page-11-18) [2018b\)](#page-11-18): The dataset consists of debates from idebate.org, collected as of January 30, 2018. Each debate includes the thesis, introductory text, all points and counters, bibliography, and metadata.
- CQADupStack [\(Hoogeveen et al.,](#page-10-20) [2015\)](#page-10-20): A benchmark dataset for community question-answering research. It contains threads from twelve StackExchange subforums, annotated with duplicate question information.
- Quora Question Pairs[<sup>4</sup>](#page-14-1) : A dataset consists of over 400,000 question pairs, and each question pair is annotated with a binary value indicating whether the two questions are paraphrase of each other.

For Multimodal embedding Experiment:

- MS COCO [\(Lin et al.,](#page-10-15) [2014\)](#page-10-15): The MS COCO dataset is a large-scale object detection, segmentation, and captioning dataset. It contains images with complex scenes involving multiple objects, each annotated with labels, bounding boxes, and segmentation masks.
- Flickr30K [\(Young et al.,](#page-12-3) [2014\)](#page-12-3): The Flickr30k dataset is a collection of images with corresponding textual descriptions. Each image is annotated with multiple captions that describe the scene, objects, and actions depicted.

# B. Experiment Detail on Vision Representation.

# B.1. Evaluation Metric

We adopt 1-NN as our evaluation metric, implemented using FAISS [\(Johnson et al.,](#page-10-14) [2019\)](#page-10-14) with exact L2 search, following the setup in [\(Kusupati et al.,](#page-10-0) [2022\)](#page-10-0). This approach provides an efficient and cost-effective way to evaluate the utility of learned representations for downstream tasks, as 1-NN accuracy requires no additional training. In detail, we use the training set with 1.3M samples as the database and the validation set with 50K samples as the query set. We also report linear probing and few-shot results using Top-1 accuracy. For a holistic evaluation, different methods, Figure [1](#page-0-0) (c) presents the average 1-NN performance (active dimensions < 64).

# B.2. Baselines

We select MRL and MRL-E from [\(Kusupati et al.,](#page-10-0) [2022\)](#page-10-0) as baselines. This work introduces a novel training paradigm that learns representations of varying lengths. MRL-E is an efficient version of MRL, also proposed in [\(Kusupati et al.,](#page-10-0) [2022\)](#page-10-0).

# B.3. Implementation Detail

For a fair comparison, we selected the pre-trained ResNet50 weights, noted as FF2048 in the MRL [\(Kusupati et al.,](#page-10-0) [2022\)](#page-10-0). Additionaly, we select the ResNet50 model[<sup>5</sup>](#page-14-2) as our SOTA backbone from [Wightman](#page-11-0) [\(2019\)](#page-11-0). For image preprocessing, we adopt the same procedure as described in [Kusupati et al.](#page-10-0) [\(2022\)](#page-10-0); [Leclerc et al.](#page-10-21) [\(2023\)](#page-10-21). Consistent with [Gao et al.](#page-9-1) [\(2024\)](#page-9-1), we utilize a tied encoder-decoder structure to build the CSR framework. The implementation of CSR is based on the codebase[<sup>6</sup>](#page-14-3) provided by OpenAI. All experiments are conducted on a server equipped with 4 RTX4090 GPUs. The selection of hyperparameters are:

Table 3. Implementation details on Image experiment.

| Backbone | <i>d</i> | <i>h</i> | Ir   | epoch | Bacth Size | <i>k</i> <sub>aux</sub> | $\beta$ | $\gamma$ | $\mathbb{K}$   | Optimizer | weight decay | eps          |
|----------|----------|----------|------|-------|------------|-------------------------|---------|----------|----------------|-----------|--------------|--------------|
| ResNet50 | 2048     | 8192     | 4e-5 | 10    | 4096       | 512                     | 1/32    | 0.1      | 8,16,32...2048 | Adam      | 1e-4         | 6.25 * 1e-10 |

# B.4. 1-NN Classification Results

1-NN classification and Top-1 linear probing results are shown in Table [4](#page-15-0) and Table [5.](#page-15-2)

<sup>4</sup><https://paperswithcode.com/dataset/quora-question-pairs>

<sup>5</sup>[https://huggingface.co/timm/resnet50d.ra4\\_e3600\\_r224\\_in1k](https://huggingface.co/timm/resnet50d.ra4_e3600_r224_in1k)

<sup>6</sup>[https://github.com/openai/sparse\\_autoencoder](https://github.com/openai/sparse_autoencoder)

Table 4. 1-NN accuracy of different methods on ImageNet1k classification.

| Active Dim | Full Rep. | MRL   | MRL-E | SVD   | Rand. FS | SAE   | CSR   | SOTA Full Rep. | CSR (w/ SOTA RN50) |
|------------|-----------|-------|-------|-------|----------|-------|-------|----------------|--------------------|
| 8          |           | 62.19 | 57.45 | 19.14 | 2.36     | 67.14 | 67.78 |                | 73.84              |
| 16         |           | 67.91 | 67.05 | 46.02 | 12.06    | 68.14 | 69.17 |                | 74.39              |
| 32         |           | 69.46 | 68.60 | 60.78 | 32.91    | 68.91 | 70.15 |                | 74.53              |
| 64         |           | 70.17 | 69.61 | 67.04 | 49.91    | 69.69 | 70.94 |                | 74.62              |
| 128        |           | 70.52 | 70.12 | 69.63 | 60.91    | 69.74 | 70.99 |                | 74.65              |
| 256        |           | 70.62 | 70.36 | 70.67 | 65.75    | 70.35 | 71.31 |                | 74.73              |
| 512        |           | 70.82 | 70.74 | 71.06 | 68.77    | 71.21 | 71.29 |                | 74.88              |
| 1024       |           | 70.89 | 71.07 | 71.22 | 70.41    | 71.20 | 71.30 |                | 74.90              |
| 2048       | 71.19     | 70.97 | 71.21 | 71.21 | 71.19    | 71.24 | 71.20 | 75.19          | 74.91              |

Table 5. Top-1 classification accuracy results of different methods on ImageNet1k classification.

| Active Dim | Table 5. Top-1 Full Rep. MRL | classification MRL-E | MRL-LP | accuracy results SVD | of different Rand. | methods LP SAE | on ImageNet1k CSR SOTA Full | classification. Rep. CSR (w/ SOTA RN50) |
|------------|------------------------------|----------------------|--------|----------------------|--------------------|----------------|-----------------------------|-----------------------------------------|
| 8          | 66.63                        | 56.66                | 5.15   | 2.34                 | 4.56               | 73.46          | 73.62                       | 79.17                                   |
| 16         | 73.53                        | 71.94                | 13.79  | 7.17                 | 11.29              | 74.60          | 74.75                       | 79.72                                   |
| 32         | 75.03                        | 74.48                | 32.52  | 20.46                | 27.21              | 75.28          | 75.44                       | 79.96                                   |
| 64         | 75.82                        | 75.35                | 52.66  | 48.10                | 49.47              | 75.81          | 75.88                       | 80.16                                   |
| 128        | 76.30                        | 75.80                | 64.60  | 67.24                | 65.70              | 75.91          | 76.24                       | 80.24                                   |
| 256        | 76.47                        | 76.22                | 69.29  | 74.59                | 72.43              | 76.27          | 76.25                       | 80.31                                   |
| 512        | 76.65                        | 76.36                | 70.51  | 76.78                | 74.94              | 76.43          | 76.34                       | 80.33                                   |
| 1024       | 76.76                        | 76.48                | 70.19  | 76.87                | 76.10              | 76.59          | 76.54                       | 80.32                                   |
| 2048       | 76.87 76.80                  | 76.51                | 69.72  |                      | 76.87              | 76.66          | 76.52 80.59                 | 80.35                                   |

# C. Experiment Detail on Text Representation

# C.1. Evaluation Metric

We adopt the universal evaluation metrics used in the MTEB benchmark [\(Muennighoff et al.,](#page-11-14) [2022\)](#page-11-14). For text classification and clustering, we use Top-1 accuracy to assess model performance. For the text retrieval task, we use NDCG@10 (Normalized Discounted Cumulative Gain at 10), a metric that evaluates the quality of a ranked list of items, commonly used in information retrieval and recommendation systems.

#### C.2. Experiment Setup

We choose three main tasks on MTEB benchmark and randomly select six datasets(for each task) to measure our methods. We also design two experiment settings to evaluate the effectiveness and generalization ability of our methods.

Firstly, we introduce *Dataset-Specific Evaluation*, where CSR are trained and tested on different splits of the same dataset. We use MTOPIntent [\(Li et al.,](#page-10-17) [2020\)](#page-10-17), Banking77 [\(Casanueva et al.,](#page-9-15) [2020\)](#page-9-15) and TweetSentimentExtraction [\(Maggie et al.,](#page-10-18) [2020\)](#page-10-18) for text classification task. We use BiorxivClusteringS2S, BiorxivClusteringP2P [\(Muennighoff et al.,](#page-11-14) [2022\)](#page-11-14) and TwentyNewsgroupdClustering for text clustering. For text retrieval, we select FiQA2018 [\(Maia et al.,](#page-10-19) [2018\)](#page-10-19), NFCorpus [\(Boteva](#page-9-18) [et al.,](#page-9-18) [2016\)](#page-9-18) and SciFACT [\(Wadden et al.,](#page-11-17) [2020\)](#page-11-17).

Furthermore, we introduce *Task-Specific Evaluation*, where CSR are trained and tested on different datasets within the same task to evaluate the generalization ability of our proposed method. We construct a training dataset using the training splits of the aforementioned datasets and test on the corresponding task datasets. For classification: MassivScenario [\(FitzGerald](#page-9-16) [et al.,](#page-9-16) [2022\)](#page-9-16), AmazonRevies [\(McAuley & Leskovec,](#page-11-15) [2013\)](#page-11-15) and Emotion [\(Saravia et al.,](#page-11-16) [2018\)](#page-11-16). For clustering: ArxivClusteringS2S, RedditClusteringP2P [\(Muennighoff et al.,](#page-11-14) [2022\)](#page-11-14) and StackExchangeClustering [\(Geigle et al.,](#page-9-17) [2021\)](#page-9-17). For retrieval: Arguana [\(Wachsmuth et al.,](#page-11-19) [2018a\)](#page-11-19), CQADupStack [\(Hoogeveen et al.,](#page-10-20) [2015\)](#page-10-20) and Quora.

### C.3. Baselines

We choose several models that provide MRL embeddings on MTEB benchmark [\(Muennighoff et al.,](#page-11-14) [2022\)](#page-11-14). These models are Stella-en-1.5B-v5 [\(Zhang et al.,](#page-12-4) [2025\)](#page-12-4), Jina-V3 [\(Sturua et al.,](#page-11-20) [2024\)](#page-11-20), Nomic-Embed-V1.5 [\(Nussbaum et al.,](#page-11-3) [2024\)](#page-11-3), Gecko-Text-Embedding-004-256 [\(Lee et al.,](#page-10-3) [2024b\)](#page-10-3), OpenAI-Text-Embedding-3-L-256 [\(OpenAI,](#page-11-1) [2024\)](#page-11-1), Arctic-Embed-L-V2.0 [\(Yu](#page-12-0)

[et al.,](#page-12-0) [2024\)](#page-12-0) and Potion-Base-2M [\(min,](#page-9-19) [2024\)](#page-9-19).

## C.4. Implementation Detail

We select NV-Embed-V2 [\(Lee et al.,](#page-10-5) [2024a\)](#page-10-5) as our pre-trained model. We utilize a tied encoder-decoder structure to build the CSR framework. For text classification and clustering tasks, we use data from the same class as positive samples while the other as negative samples to calculate Equation [6.](#page-4-0) The hyperparameters are set as follows:

Table 6. Implementation details on Text experiment.

| Backbone    | $d$  | $h$  | $lr$ | epoch | Back Size | $k_{aux}$ | $\beta$ | $\gamma$ | $\mathbb{K}$ | Optimizer | weight decay | eps            |
|-------------|------|------|------|-------|-----------|-----------|---------|----------|--------------|-----------|--------------|----------------|
| NV-Embed-V2 | 4096 | 1964 | 2e-5 | 10    | 128       | 1024      | 0.9     | 1.0      | 32,64,256    | Adam      | 1e-4         | $6.25 * 1e-10$ |

# D. Experiment Detail on MultiModal Representation

# D.1. Evaluation Metric

We adopt the universal evaluation metric Recall@5 to measure performance in the MultiModal Retrieval task. This metric evaluates a model's ability to retrieve relevant items within its top 5 predictions. Calculated as the fraction of relevant items appearing in the top 5 results out of the total relevant items, a higher Recall@5 indicates better performance in capturing relevant content early in the ranked list, making it useful for recommendation systems and retrieval tasks.

## D.2. Experiment Setup

We selected ViT-B-16, trained on the DFN2B dataset[<sup>7</sup>](#page-16-2) , as our pre-trained model. For the in-distribution cross-modal retrieval experiment, we implemented MRL in the pre-trained ViT model following [Kusupati et al.](#page-10-0) [\(2022\)](#page-10-0), and fine-tuned it for 50 epochs on the MSCOCO [\(Lin et al.,](#page-10-15) [2014\)](#page-10-15) and Flickr30K [\(Young et al.,](#page-12-3) [2014\)](#page-12-3) datasets, respectively. For a fair comparison, we also fine-tuned the backbone on both datasets for 50 epochs using the same hyperparameters, which were then used for the backbone of CSR . The hyperparameters used for fine-tuning are as follows:

Table 7. Hyperparameters for fine-tuning ViT-B/16 backbone.

| Dataset   | Ir   | epoch | Batch Size | warmup | Optimizer | weight decay |  |
|-----------|------|-------|------------|--------|-----------|--------------|--|
| MS COCO   | pe-6 | 50    | 64         | 10000  | Adam      | 0.1          |  |
| Flickr30k | pe-6 | 50    | 64         | 10000  | Adam      | 0.1          |  |

For zero-shot cross-modal retrieval, we employed the same MRL fine-tuning procedure as in our in-distribution experiment, maintaining identical hyperparameters while training for 3 epochs with 2208 batch size on CC3M [\(Changpinyo et al.,](#page-9-11) [2021\)](#page-9-11).

# D.3. Implementation Detail

We select the ViT-B-16[<sup>8</sup>](#page-16-3) as our backbone from [Wightman](#page-11-0) [\(2019\)](#page-11-0). Consistent with [Gao et al.](#page-9-1) [\(2024\)](#page-9-1), we utilize a tied encoder-decoder structure to build the CSR framework. The encoder and decoder structure share between image space and text space. The implementation of CSR is based on the codebase[<sup>9</sup>](#page-16-4) and OpenCLIP [\(Cherti et al.,](#page-9-20) [2023\)](#page-9-20). The metric is evaluated through CLIP-benchmark following standard procedure. All experiments are conducted on a server equipped with 4 RTX4090 GPUs. We present detailed training parameters for the multimodal experiment in Table [8.](#page-17-1)

# D.4. Discussion On Dead Latents

Addressing the mitigation of dead latents in the alignment space remains an open challenge, leaving room for future work and study. Table [2](#page-8-0) presents the performance comparison between CSR and MRL, revealing that the gap between the two

<sup>7</sup><https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16>

<sup>8</sup><https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16>

<sup>9</sup>[https://github.com/openai/sparse\\_autoencoder](https://github.com/openai/sparse_autoencoder)

Table 8. Implementation details on MultiModal experiment.

| Dataset   | <i>d</i> | <i>h</i> | Ir   | epoch | Batch Size | <i>k</i> <sub>aux</sub> | $\frac{1}{B}$ | $\frac{K}{K}$ | Optimizer | weight decay | eps          |
|-----------|----------|----------|------|-------|------------|-------------------------|---------------|---------------|-----------|--------------|--------------|
| MS COCO   | 512      | 2048     | 24-4 | 5     | 256        | 512                     | 1/3.2         | 6,418,256,256 | Adam      | 1e-4         | 6.25 * 1e-10 |
| Flickr30k | 512      | 2048     | 24-4 | 5     | 64         | 1024                    | 1/3.2         | 6,418,256,256 | Adam      | 1e-4         | 6.25 * 1e-10 |
| CC3M      | 512      | 2048     | 24-4 | 1     | 1024       | 1024                    | 1/3.2         | 6,418,256,256 | Adam      | 0.0          | 6.25 * 1e-10 |

methods diminishes as sparsity constraints become more stringent. Further analysis indicates that CSR continues to face the "dead latents" issue despite incorporating advanced mechanisms. As shown in Figure [8,](#page-17-2) CSR exhibits a significant performance drop, corresponding to a sharp rise in dead latent dimensions. We attribute this to a technical challenge, as CSR has demonstrated robust performance in both image and text domains under similar sparsity constraints. This suggests that representations in alignment spaces may require more specialized design, presenting an opportunity for future research and improvement.

![](_page_17_Figure_4.jpeg)

Figure 8. Dead latents still exits in image-text alignment space.

# E. Empirical Analysis

#### E.1. Effect on Input Embedding Dimension R d

The implementation details are shown in Table [9.](#page-17-3) To avoid other unknown factors, we choose ViT-based[<sup>10</sup>](#page-17-4) and ResNet-based models[<sup>11</sup>](#page-17-5) following same pre-training procedure respectively. To ensure generalizability, we train the model using three different random seeds and report the mean performance in the main paper.

Table 9. Implementation details on empirical study of input embedding dimension R d

| Backbone | d    | h    | Table 9. lr | Implementation epoch | details Batch | on Size k aux | empirical β | study γ | of input K  | embedding dimension Optimizer | weight | decay | eps     |
|----------|------|------|-------------|----------------------|---------------|---------------|-------------|---------|-------------|-------------------------------|--------|-------|---------|
| ViT-L/16 | 512  | 4096 | 4e-5        | 10                   | 1024          | 512           | 1/32        | 1.0     | 8,16,64,256 | Adam                          | 1e-4   | 6.25  | * 1e-10 |
| ViT-L/16 | 1024 | 4096 | 4e-5        | 10                   | 1024          | 512           | 1/32        | 1.0     | 8,16,64,256 | Adam                          | 1e-4   | 6.25  | * 1e-10 |
| ResNet18 | 512  | 8192 | 4e-5        | 10                   | 1024          | 512           | 1/32        | 1.0     | 8,16,64,256 | Adam                          | 1e-4   | 6.25  | * 1e-10 |
| ResNet50 | 2048 | 8192 | 4e-5        | 10                   | 1024          | 512           | 1/32        | 1.0     | 8,16,64,256 | Adam                          | 1e-4   | 6.25  | * 1e-10 |

<sup>10</sup>[https://huggingface.co/timm/vit\\_small\\_patch16\\_224.augreg\\_in21k\\_ft\\_in1k](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k),[https://](https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k) [huggingface.co/timm/vit\\_large\\_patch16\\_224.augreg\\_in21k\\_ft\\_in1k](https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k)

<sup>11</sup>[https://huggingface.co/timm/resnet18.a1\\_in1k](https://huggingface.co/timm/resnet18.a1_in1k),[https://huggingface.co/timm/resnet50.a1\\_](https://huggingface.co/timm/resnet50.a1_in1k) [in1k](https://huggingface.co/timm/resnet50.a1_in1k)

Table 10. Implementation details on empirical study of hidden dimension R h

| Backbone d | h     | lr   | epoch | Batch | Size k aux | β    | γ   | K           | Optimizer | weight | decay | eps     |
|------------|-------|------|-------|-------|------------|------|-----|-------------|-----------|--------|-------|---------|
| 1024       | 1024  | 4e-5 | 10    | 1024  | 512        | 1/32 | 1.0 | 8,16,64,256 | Adam      | 1e-4   | 6.25  | * 1e-10 |
| 1024       | 2048  | 4e-5 | 10    | 1024  | 512        | 1/32 | 1.0 | 8,16,64,256 | Adam      | 1e-4   | 6.25  | * 1e-10 |
| 1024       | 4096  | 4e-5 | 10    | 1024  | 512        | 1/32 | 1.0 | 8,16,64,256 | Adam      | 1e-4   | 6.25  | * 1e-10 |
| 1024       | 8192  | 4e-5 | 10    | 1024  | 512        | 1/32 | 1.0 | 8,16,64,256 | Adam      | 1e-4   | 6.25  | * 1e-10 |
| 1024       | 16384 | 4e-5 | 10    | 1024  | 512        | 1/32 | 1.0 | 8,16,64,256 | Adam      | 1e-4   | 6.25  | * 1e-10 |
| 2048       | 2048  | 4e-5 | 10    | 1024  | 512        | 1/32 | 1.0 | 8,16,64,256 | Adam      | 1e-4   | 6.25  | * 1e-10 |
| 2048       | 4096  | 4e-5 | 10    | 1024  | 512        | 1/32 | 1.0 | 8,16,64,256 | Adam      | 1e-4   | 6.25  | * 1e-10 |
| 2048       | 8192  | 4e-5 | 10    | 1024  | 512        | 1/32 | 1.0 | 8,16,64,256 | Adam      | 1e-4   | 6.25  | * 1e-10 |
| 2048       | 16384 | 4e-5 | 10    | 1024  | 512        | 1/32 | 1.0 | 8,16,64,256 | Adam      | 1e-4   | 6.25  | * 1e-10 |
| 2048       | 32768 | 4e-5 | 10    | 1024  | 512        | 1/32 | 1.0 | 8,16,64,256 | Adam      | 1e-4   | 6.25  | * 1e-10 |

#### E.2. Effect on Hidden Representation Dimension R h

Implementation details are shown in Table [10.](#page-18-3) The pre-trained ViT-L/16[<sup>12</sup>](#page-18-4) and ResNet50 models[<sup>13</sup>](#page-18-5) can be found at timm [\(Wightman,](#page-11-0) [2019\)](#page-11-0). To ensure generalizability, we train the model using three different random seeds and report the mean performance in the main paper.

## E.3. Retrieval Time Evaluation

We employ PyTorch [\(Paszke et al.,](#page-11-13) [2019\)](#page-11-13) to measure retrieval time on ImageNet1k. The average retrieval time is computed over 2000 rounds with a batch size of 512 queries, excluding an initial 100 warm-up rounds. For the learned CSR representation, both query and key embeddings are stored in csr format, and sparse product operations are utilized for similarity computation while maintaining identical experimental settings for fair comparison.

## E.4. Understanding Retrieval Time Difference between Dense and Sparse Embeddings

Although CSR and MRL have similar theoretical complexity O(k), their actual runtimes are affected by backend implementations. For instance, cuBLAS (used for dense ops) is highly optimized but has high launch overhead, while cuSPARSE (used for CSR) is lighter but less optimized for small k. Here, we can share a preliminary insight into why sparse embeddings can be faster than dense embeddings and why it can get faster with larger hidden dimension h.

Sparse matrix multiplication benefits from zero-skipping: only overlapping non-zero entries are used. For each query, computing the i-th output only involves comparing indices of non-zero entries—an integer operation much cheaper than floating-point multiplication. As h increases and k stays small, overlap likelihood drops, reducing the number of multiplications required. In Table We empirically verify this by counting the number of multiplications under various h:

Table 11. Comparison on the number of multiplication operation between MRL (dense) and CSR (embeddings) on the default setup.

| Active Dim | MRL               | CSR ( $h = 8192$ ) | CSR ( $h = 16384$ ) | CSR ( $h = 3276$ ) |
|------------|-------------------|--------------------|---------------------|--------------------|
| 2          | $1.3 \times 10^9$ | $3.2 \times 10^5$  | $1.7 \times 10^5$   | $8.4 \times 10^4$  |
| 4          | $2.6 \times 10^9$ | $1.3 \times 10^6$  | $6.7 \times 10^5$   | $3.4 \times 10^5$  |

The number of operations in CSR is several orders of magnitude smaller than in MRL, and it decreases with larger h. This counterintuitive yet practical effect highlights the appeal of using sparse high-dimensional embeddings: they allow richer representations while improving runtime.

<sup>12</sup>[https://huggingface.co/timm/vit\\_large\\_patch16\\_224.augreg\\_in21k\\_ft\\_in1k](https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k)

<sup>13</sup>[https://huggingface.co/timm/resnet50.a1\\_in1k](https://huggingface.co/timm/resnet50.a1_in1k)