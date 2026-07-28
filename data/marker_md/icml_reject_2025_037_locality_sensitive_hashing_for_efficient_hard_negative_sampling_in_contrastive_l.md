011

014 015 016

018

024

026

034

036

038

054

# Locality-Sensitive Hashing for Efficient Hard Negative Sampling in Contrastive Learning

Anonymous Authors<sup>1</sup>

#### Abstract

Contrastive learning is a representational learning paradigm in which a neural network maps data elements to feature vectors. It improves the feature space by forming lots with an anchor and examples that are either positive or negative based on class similarity. Hard negative examples, which are close to the anchor in the feature space but from a different class, improve learning performance. Finding such examples of high quality efficiently in large, high-dimensional datasets is computationally challenging. In this paper, we propose a GPU-friendly LSH scheme that quantizes real-valued feature vectors into binary representations for approximate nearest neighbor search. We demonstrate on several datasets from both textual and visual modalities that our approach outperforms other hard negative mining strategies in terms of computational efficiency without significant performance degradation.

## 1. Introduction

Contrastive learning builds on the principle of distinguishing positive (similar) examples from negative (dissimilar) examples, and aims to learn a representation space in which similar data points are closer together than dissimilar ones. Unlike supervised classification, which relies on hard labeldefined boundaries, contrastive learning provides a learning strategy for tasks where such strict boundaries are inadequate. Scenarios for contrastive learning vary from person re-identification [\(Hermans et al.,](#page-9-0) [2017\)](#page-9-0) or face verification [\(Schroff et al.,](#page-9-1) [2015\)](#page-9-1), player re-identification [\(Zhang](#page-10-0) [et al.,](#page-10-0) [2020;](#page-10-0) [Habel et al.,](#page-8-0) [2022\)](#page-8-0), up to cross-view geolocalization [\(Deuser et al.,](#page-8-1) [2023a;](#page-8-1) [Zhu et al.,](#page-10-1) [2022;](#page-10-1) [Deuser](#page-8-2) [et al.,](#page-8-2) [2024;](#page-8-2) [2023b\)](#page-8-3), sentence and text retrieval [\(Reimers](#page-9-2) [& Gurevych,](#page-9-2) [2019;](#page-9-2) [2020\)](#page-9-3), multi-modal retrieval [\(Radford](#page-9-4)

![](_page_0_Figure_3.jpeg)

Figure 1: We compare search time relative to dataset size, showing results with ConvNeXt and Transformer models. Both use LSH-based feature encoding with varying bit sizes, along with pre-epoch HN sampling using float32 embeddings and their respective model output sizes.

[et al.,](#page-9-4) [2021;](#page-9-4) [Zhai et al.,](#page-10-2) [2023\)](#page-10-2), and product search [\(Patel](#page-9-5) [et al.,](#page-9-5) [2022;](#page-9-5) [An et al.,](#page-8-4) [2023\)](#page-8-4). These tasks exemplify the success and versatility of contrastive learning across diverse domains. An example of the need for contrastive learning in these scenarios is product search. Items of clothing may look very similar, almost identical, but belong to different categories, such as a sweater compared to a sweatshirt. In text retrieval, where sentences with different structures and vocabularies can convey the same meaning, the challenge is even greater. These use cases underscore the need for embeddings that capture nuanced similarities without enforcing rigid class separations.

Batch composition sampling strategies are crucial in contrastive learning, as they significantly impact training effectiveness [\(Wu et al.,](#page-10-3) [2017\)](#page-10-3). Research has shown that incorporating negative examples close to the anchor sample, called Hard Negative (HN), can improve learning outcomes [\(Wu](#page-10-3) [et al.,](#page-10-3) [2017;](#page-10-3) [Galanopoulos & Mezaris,](#page-8-5) [2021;](#page-8-5) [Wang et al.,](#page-10-4) [2019;](#page-10-4) [Yuan et al.,](#page-10-5) [2017;](#page-10-5) [Hermans et al.,](#page-9-0) [2017;](#page-9-0) [Cakir et al.,](#page-8-6) [2019;](#page-8-6) [Xuan et al.,](#page-10-6) [2020\)](#page-10-6). However, with modern datasets

<sup>1</sup>[Anonymous Institution, Anonymous City, Anonymous Region,](#page-9-4) [Anonymous Country. Correspondence to: Anonymous Author](#page-9-4) <[anon.email@domain.com](#page-9-4)>.

[Preliminary work. Under review by the International Conference](#page-9-4) [on Machine Learning \(ICML\). Do not distribute.](#page-9-4)

058

071

074

076

078

087 088

090 091

093 094

096

098

100

104

106

108 109 containing millions [\(Radford et al.,](#page-9-4) [2021\)](#page-9-4) to billions [\(Jia](#page-9-6) [et al.,](#page-9-6) [2021\)](#page-9-6) of samples, computing and training on all possible negative combinations is impractical. Pre-extracted HNs, though often more effective than random sampling, do not adapt to changes in the embedding space during training. This limits their effectiveness. Therefore, an efficient dynamic selection of HNs based on specific criteria is crucial to maximize training effectiveness.

A common strategy for HNs calculation [\(Wang et al.,](#page-10-4) [2019;](#page-10-4) [Yuan et al.,](#page-10-5) [2017;](#page-10-5) [Hermans et al.,](#page-9-0) [2017;](#page-9-0) [Cakir et al.,](#page-8-6) [2019;](#page-8-6) [Xuan et al.,](#page-10-6) [2020\)](#page-10-6) is the within-batch selection, based on a pre-defined criteria. The within-batch calculation is computationally effective as the HNs are selected dynamically during training. However, pre-epoch HNs sampling, which computes negatives globally, offers the advantage of a more comprehensive view of the dataset leading to higher performance [\(Deuser et al.,](#page-8-1) [2023a\)](#page-8-1). Unfortunately, this introduces significant computational complexity, making it infeasible for large datasets, as shown in Figure [1.](#page-0-0)

To address computational inefficiency of pre-epoch HN sampling, we propose a lightweight Approximated Nearest Neighbor (ANN) approach that leverages Locality-Sensitive Hashing (LSH) [\(Charikar,](#page-8-7) [2002\)](#page-8-7) to reduce search time and space costs. We store and retrieve HNs efficiently by encoding approximate embeddings in a compact binary space, enabling fast queries while maintaining a global view of the dataset. This accelerates pre-epoch sampling, boosting training efficiency without sacrificing effectiveness.

Our work first explores HN sampling methods and introduces LSH as an ANN approach. Using this foundation, we design a training process and evaluate it on six datasets spanning two modalities. We then compare HN-based performance gains against random negatives and those mined with our LSH-based method. Finally, we analyze the hardness and relevance of the mined HN in relation to real HN identified by cosine similarity.

To summarize, we contribute:

- A lightweight and efficient framework for HN sampling using LSH, offering a global view of the dataset while keeping computational costs low during training.
- A comprehensive analysis of LSH on six datasets in the context of supervised contrastive learning, demonstrating its effectiveness on dynamic embeddings during training on multiple datasets from two modalities.
- A demonstration of our method drastically reducing the complexity of training time and providing an efficient and scalable HN-sampling strategy without severe performance degradation despite its approximate nature.

## 2. Related Work

Existing mining strategies for contrastive learning fall into two categories: within-batch sampling and pre-epoch sampling. We briefly review both strategies.

#### 2.1. With-In Batch Sampling

Simo-Serra et al. refined within-batch sampling by selecting HNs based on loss values computed after the forward step [\(Simo-Serra et al.,](#page-9-7) [2015\)](#page-9-7). Samples are chosen randomly at the start of each epoch, with backward gradients computed only for high-loss cases. Similarly, triplet loss [\(Schroff et al.,](#page-9-1) [2015\)](#page-9-1) enhances HN sampling. Schroff et al. employed it in an online mining scheme, selecting HNs within a batch using ℓ<sup>2</sup> distance.

Subsequent work [\(Wu et al.,](#page-10-3) [2017\)](#page-10-3) introduces semi-HN sampling, as mining only the hardest examples can cause model collapse. Others [\(Hermans et al.,](#page-9-0) [2017\)](#page-9-0) compared multiple mining strategies for triplet loss in person re-identification, showing that selecting the hardest positive and negative within a batch outperforms prior work [\(Oh Song et al.,](#page-9-8) [2016;](#page-9-8) [Ding et al.,](#page-8-8) [2015\)](#page-8-8). Hermans et al. investigated offline hard mining as well [\(Hermans et al.,](#page-9-0) [2017\)](#page-9-0). However, selecting the hardest samples across the entire dataset led to suboptimal performance, causing model collapse with standard triplet loss and hindering training.

Yuan et al. proposed a cascaded model to identify HNs at different network stages [\(Yuan et al.,](#page-10-5) [2017\)](#page-10-5), enabling the model to focus on hard examples when they are the most difficult to distinguish, improving learning effectiveness.

Another strategy is mining informative pairs by comparing negative pairs with the hardest positive pairs and vice versa [\(Wang et al.,](#page-10-4) [2019\)](#page-10-4). Wang et al. further refined this mining strategy with a soft weighting scheme to more accurately prioritize the selected pairs [\(Wang et al.,](#page-10-4) [2019\)](#page-10-4). For positive sampling Xuan et al. found out that easiest samples can provide higher generalization as the embedding maintains intra-class variance [\(Xuan et al.,](#page-10-6) [2020\)](#page-10-6).

## 2.2. Pre-Epoch Sampling

While previous work focusses on in-batch strategies, Cakir et al. take a different approach by defining the batch composition during the training epoch [\(Cakir et al.,](#page-8-6) [2019\)](#page-8-6). They use WordNet [\(Pedersen et al.,](#page-9-9) [2004\)](#page-9-9) similarities between classes to determine which classes should be sampled together, effectively introducing harder-to-differentiate samples. In a previously mentioned study, Hermans et al. also explore offline HN mining [\(Hermans et al.,](#page-9-0) [2017\)](#page-9-0), but in line with the results of [\(Wu et al.,](#page-10-3) [2017\)](#page-10-3), they found that HNs can lead to model collapse with standard triplet loss.

While most of the previous work focused on image retrieval

114 115 116

118

124

126

128

131

134

136

138

151

154

158

160

164

Gillick et al. introduced HN sampling for entity retrieval in Natural Language Processing (NLP) [\(Gillick et al.,](#page-8-9) [2019\)](#page-8-9). They encode all mentions and entities, identifying the 10 most similar entities after each epoch. If an incorrect entity ranks higher than the correct one, it is treated as an HN. Qu et al. enhance this by adding a cross-encoder denoising mechanism to reduce false negatives [\(Qu et al.,](#page-9-10) [2020\)](#page-9-10).

Xiong et al. pioneer the use of ANN by storing embeddings in a database during training and performing sampling on asynchronously updated indices [\(Xiong et al.,](#page-10-7) [2020\)](#page-10-7). In cross-view geo-localization, Deuser et al. highlight the significant performance improvements achievable with HN sampling [\(Deuser et al.,](#page-8-1) [2023a\)](#page-8-1), and show that the InfoNCE loss [\(Oord et al.,](#page-9-11) [2018\)](#page-9-11) avoids the collapsing problems often associated with triplet loss. However, their method incurs significant computational and storage costs due to the need to compute the entire similarity matrix.

#### 2.3. Research Gap

Xiong et al. construct an index using full vector embeddings and asynchronously update the embeddings for the entire dataset every few batches, significantly increasing computational overhead and double the GPU resource requirements [\(Xiong et al.,](#page-10-7) [2020\)](#page-10-7). In contrast, we investigate whether a lower-dimensional binary representation is sufficient to retrieve high-quality HNs.

## 3. Method

#### 3.1. Preliminary

The primary goal of contrastive learning is to bring positive pairs closer together in the embedding space while pushing negative pairs farther apart. In a supervised setting, positive pairs consist of samples with the same label, while negative pairs have different labels. Previous work [\(Deuser et al.,](#page-8-1) [2023a;](#page-8-1) [Xiong et al.,](#page-10-7) [2020;](#page-10-7) [Wang et al.,](#page-10-4) [2019;](#page-10-4) [Xuan et al.,](#page-10-6) [2020;](#page-10-6) [Cakir et al.,](#page-8-6) [2019\)](#page-8-6) has shown that the selection of negative samples can significantly affect the learning process, either by speeding it up or by improving generalization. In this work we first want to analyze the theoretical properties of our approach based on ANN for the selection of HNs.

We aim to find a suitable embedding Y ⊂ R d in a ddimensional real vector space for some input data X that is parametrized by a neural network, i.e. f<sup>θ</sup> : X → Y . To do so, we first want to establish the InfoNCE loss on the embedding space, a contrastive loss function defined by [\(He](#page-9-12) [et al.,](#page-9-12) [2020\)](#page-9-12) as

$$\mathcal{L}_c(y_1, \dots, y_K) = -\log \frac{\exp\left(\frac{c^\top y_+}{\|c\| \|y_+\|} / \tau\right)}{\sum_{i=1}^K \exp\left(\frac{c^\top y_i}{\|c\| \|y_i\|} / \tau\right)}$$

on a batch of size K. c is called the anchor point, y<sup>+</sup> ∈

y1, . . . , y<sup>K</sup> serves as its positive sample from the identical class and other y<sup>i</sup> serve as its negative samples from arbitrary different classes. τ serves as a temperature parameter, controlling how concentrated the features are in the representation space. If the cosine similarity between the anchor and its positive sample, defined as

$$\text{sim}(c, y_+) = \frac{c^\top y_+}{\|c\| \|y_+\|}$$

is high, the loss decreases. Vice versa, if the similarity between the anchor and its negative samples is high, the loss increases.

During training, we iteratively sample anchors and the corresponding batches of their positive and negative samples, to calculate the derivative of the loss w.r.t. to θ, to achieve a better embedding, pushing c closer to its positive sample y<sup>+</sup> and further away from its negative samples. θ denotes the weights of the neural network.

According to [\(Schroff et al.,](#page-9-1) [2015\)](#page-9-1), a crucial step is choosing meaningful positives and negative samples for the anchor to achieve fast convergence, i.e. the similarity between anchor and positive sample is lower than between anchor and negative samples:

$$\text{sim} (c, y_+) \leq \text{sim} (c, y_i) \forall i$$

Given an anchor c, we call

$$y_- = \arg \max_{y \in Y: y \neq y_+} \text{sim}(c, y)$$

its HN sample. It is the most similar instance of another class to the anchor. Thus, the batch must include the K − 1 hardest negatives.

Since every element in the dataset can be an anchor with its corresponding batch of HNs, traditional pre-epoch sampling calculates the full similarity matrix S. Each entry Sij = sim(y<sup>i</sup> , y<sup>j</sup> ) represents the cosine similarity between the pairs of embeddings. The calculation of this matrix is computational expensive and memory-intensive due to its size which scales quadratically to dataset size M.

To mitigate these computational expensive and memoryintensive drawbacks, we employ an ANN method [\(Har-](#page-9-13)[Peled et al.,](#page-9-13) [2012;](#page-9-13) [Charikar,](#page-8-7) [2002\)](#page-8-7) resulting in reduced computational complexity and time. Our approach to ANN utilizes LSH, which we will elaborate on in the following.

#### 3.2. Locality Sensitive Hashing

During training, it is essential to query stored vectors to identify HNs after each epoch. However, storing all these vectors can become space intensive, especially as the number of data points grow. To address this challenge, we

168

171

174

176

178

194

196 197 198

200

204

206

208

211

214 215 216

218

A

P

1.00

−0.95

<sup>−</sup>0.<sup>99</sup> <sup>−</sup>0.<sup>32</sup>

0.47

1.00

No Hyperplane (Cosine Similarities)

A

P

0

1

1

1

0

0

Random Hyperplane √<sup>1</sup>

2 (1, 1) A

P

0

1

1

0

1

0

Random Hyperplane √<sup>1</sup>

5 (2, −1)

Figure 2: Illustration of the anchor (A, blue), positive (P, red), and several negatives (N, black). Left: The raw cosine similarities between the anchor and the negatives are shown, which are commonly used to determine HNs (examples that are very close to the anchor). Middle and Right: Two examples of randomly sampled hyperplanes ( √ (1, 1) and √ 1 5 (2, −1)) are provided, demonstrating how the HNs have a high probability of being mapped to the same side of the hyperplane as the anchor. The Hamming distance is defined as the number of hyperplanes separating the embeddings. Thus, higher cosine similarity corresponds to a smaller Hamming distance, enabling effective identification of HNs.

adopt a binarization approach inspired by LSH, which significantly reduces storage requirements while maintaining the ability to efficiently retrieve ANNs. Following previous work [\(Har-Peled et al.,](#page-9-13) [2012;](#page-9-13) [Datar et al.,](#page-8-10) [2004;](#page-8-10) [Andoni](#page-8-11) [et al.,](#page-8-11) [2015\)](#page-8-11), we implement this approach by sampling a random rotation (i.e. the vectors are othonormal) matrix.

$$R \in \mathbb{R}^{b \times d}$$

where d denotes the dimensionality of the embedded feature vector y ∈ Y , and b specifies the bit dimension of the encoded feature vector. The embedded dataset Y is first transformed using the random matrix R and then in every dimension centered around its mean:

$$Z = RY - \overline{RY}$$

We then convert every vector z ∈ Z into a signed vector representation zˆ:

$$\hat{z}_i = \text{sign}(z_i), \text{ where } \text{sign}(z_i) = \begin{cases} 1 & \text{if } z_i \geq 0 \\ -1 & \text{if } z_i < 0 \end{cases}$$

Following the work from Wang et al. the probability of an anchor point c and another point y to be mapped into the same bit in one dimension is [\(Wang et al.,](#page-10-8) [2015\)](#page-10-8):

$$\Pr [h_i(c) = h_i(y)] = 1 - \frac{\theta_{cy}}{\pi} = 1 - \frac{1}{\pi} \cos^{-1} \frac{c^\top y}{\|c\| \|y\|}$$

where h<sup>i</sup> converts the vector y into the binary representation zˆ<sup>i</sup> as described above and θ is the angle. This is illustrated in Figure [2](#page-3-0) where we show how the cosine similarity between the anchor and the data points affects different hyperplanes hi . The Hamming distance between c and y

$$\text{HamDist}(c, y) = \sum_{i=1}^b \mathbf{1}_{h_i(c) \neq h_i(y)}$$

is the number of bits they differ in. As all rows of R are drawn independently it corresponds to a binomial distribution with parameters <sup>θ</sup>cy π as success rate and b as number of trials. For large b this is approximating a normal distribution. The smaller the angle between the data point y and the anchor c, the more likely our method will identify y as a nearest neighbor because the Hamming distance will be smaller. In our ablation study we further investigate the design choices made during the LSH process, namely the choice for an orthonormal matrix as well as the centering.

The advantages of such encoding become evident when working with large datasets. For example, reducing embeddings from d-dimensional 32-bit floating-point vectors, where each value requires 4 bytes, to d-dimensional binary representations, where each value requires only 1 bit, results in a reduction of storage requirements by a factor of 32. This drastically reduces the memory needed for storing embeddings used in HN sampling. Additionally, storing embeddings as binary vectors enables the use of Hamming distance, i.e. the number of different bits in the vectors, for similarity search, which is highly efficient due to its reliance on bitwise operations (XOR and popcount) [\(Wang et al.,](#page-10-8) [2015\)](#page-10-8). These operations are optimized in hardware, providing significantly faster similarity computations compared to cosine similarity, especially for high-dimensional data.

226

228

231

234

236

238

254

256

258

260

264

266

268

271

Table 1: Quantitative comparison between multiple sampling methods on supervised image retrieval dataset. Results are reported for Recall@1 (R@1) and Recall@5 (R@5).

|           | Approach |             | R@1              | CVUSA R@5 | R@1   | CVACT val R@5 | R@1   | CVACT test R@5 | R@1   | VIGOR same R@5 | R@1   | VIGOR cross R@5 | R@1   | SOP R@5 | R@1   | InShop R@5 |
|-----------|----------|-------------|------------------|-----------|-------|---------------|-------|----------------|-------|----------------|-------|-----------------|-------|---------|-------|------------|
| Random    |          |             | 97.68            | 99.63     | 87.46 | 96.46         | 60.17 | 89.35          | 64.58 | 91.19          | 36.06 | 62.96           | 87.55 | 94.70   | 91.93 | 97.92      |
| BatchHard |          | (Schroff et | al., 2015) 97.64 | 99.63     | 87.28 | 96.65         | 60.68 | 89.46          | 66.75 | 92.28          | 36.31 | 63.71           | 87.85 | 94.93   | 91.93 | 97.93      |
| Pre-Epoch |          | Full        | 98.68            | 99.67     | 91.01 | 97.11         | 69.98 | 92.82          | 77.11 | 96.11          | 59.86 | 82.55           | 89.44 | 95.76   | 93.21 | 98.21      |
| Pre-Epoch |          | Incr.       | 98.53            | 99.62     | 90.42 | 97.12         | 68.71 | 92.50          | 76.39 | 96.01          | 57.97 | 81.61           | 89.78 | 95.75   | 93.07 | 98.30      |
| LSH       | 128      | (ours)      | 98.15            | 99.67     | 89.70 | 96.94         | 66.29 | 91.62          | 74.48 | 95.24          | 53.93 | 79.08           | 89.09 | 95.42   | 92.86 | 98.22      |
| LSH       | 256      | (ours)      | 98.43            | 99.65     | 90.07 | 97.02         | 67.27 | 91.93          | 75.50 | 95.59          | 55.99 | 80.53           | 89.34 | 95.54   | 93.19 | 98.13      |
| LSH       | 512      | (ours)      | 98.54            | 99.68     | 90.45 | 97.15         | 68.20 | 92.29          | 76.35 | 95.76          | 57.22 | 81.25           | 89.60 | 95.65   | 93.11 | 98.12      |
| LSH       | 1024     | (ours)      | 98.60            | 99.65     | 90.82 | 97.24         | 68.75 | 92.60          | 76.51 | 95.88          | 57.69 | 81.53           | 89.64 | 95.78   | 93.31 | 98.22      |

## 4. Evaluation

We compare our approach on six datasets, two from the domain of product search [\(Oh Song et al.,](#page-9-8) [2016;](#page-9-8) [Liu](#page-9-14) [et al.,](#page-9-14) [2016\)](#page-9-14), three from the domain of cross-view geolocalization [\(Zhu et al.,](#page-10-9) [2021;](#page-10-9) [Workman et al.,](#page-10-10) [2015;](#page-10-10) [Liu](#page-9-15) [& Li,](#page-9-15) [2019\)](#page-9-15), and one textual retrieval dataset [\(Bajaj et al.,](#page-8-12) [2016\)](#page-8-12), as prototypical retrieval tasks. We compare the results on benchmark metrics, evaluating overlap and mean positional distance while assessing LSH against pre-epoch and random sampling.

## 4.1. Training Process

Our comparison uses a Siamese CNN [\(Chopra et al.,](#page-8-13) [2005\)](#page-8-13) to encode image embeddings and a Transformer [\(Vaswani,](#page-10-11) [2017\)](#page-10-11) to generate text embeddings. To minimize computational overhead, we reuse training embeddings for preepoch sampling. Although these embeddings may not match the latest network updates, this approach remains efficient with minimal impact on performance. To handle the computational and memory requirements of highdimensional embeddings, we use LSH to project them into a lower-dimensional binary space. The dimensionality of this space is determined by a random rotation matrix with bit sizes b ∈ 128, 256, 512, 1024 for our image embeddings and b ∈ 128, 256, 512, 768 for the text embeddings. After each epoch, a binary index is created and HNs are identified by computing Hamming distances. These HNs are then used in the next epoch to efficiently construct training batches.

For datasets without predefined query and reference splits, such as SOP [\(Oh Song et al.,](#page-9-8) [2016\)](#page-9-8) or InShop [\(Liu et al.,](#page-9-14) [2016\)](#page-9-14), we use a similar approach. Within each class, the hardest positive sample is selected for each positive sample based on Hamming distance.

## 4.2. Implementation Details

We use ConvNeXt-base [\(Liu et al.,](#page-9-16) [2022\)](#page-9-16) as the CNN backbone, training with a learning rate of 1E-3 and a cosine decay schedule. As Transformer we use Distill-RoBERTabase [\(Sanh et al.,](#page-9-17) [2019\)](#page-9-17) with a learning rate of 1E-4 and a cosine decay schedule over 10 epochs. During training, we apply a weight decay of 0.01 and use label smoothing set to 0.1 to improve generalization. The InfoNCE loss [\(Oord](#page-9-11) [et al.,](#page-9-11) [2018\)](#page-9-11) is used in all experiments, with a learnable temperature parameter τ . All our experiments are conducted on a Nvidia DGX-2 system equipped with 16 Nvidia V100 GPUs and dual Intel Xeon Platinum 8168 processors.

#### 4.3. Datasets

CVUSA [\(Workman et al.,](#page-10-10) [2015\)](#page-10-10), contains images from all over the US from different locations and 35,532 pairs for training and 8,884 in the validation set.

CVACT [\(Liu & Li,](#page-9-15) [2019\)](#page-9-15) contains the same amount of data for training and validation, but in the region of Canberra, Australia, and extends further with a test set containing over 92k images. In CVUSA and CVACT, the street view is always centered on the aerial view.

VIGOR contains 90,618 aerial views and 105,214 street views with arbitrary positions within the aerial view, significantly increasing the challenge of the task [\(Zhu et al.,](#page-10-9) [2021\)](#page-10-9). These datasets provide two configurations: "cross" and "equal". In the cross setting, training data is derived from two cities, while testing is performed on the other two cities. Conversely, the same setting uses samples from all four city regions for both training and testing.

Stanford Online Products (SOP) [\(Oh Song et al.,](#page-9-8) [2016\)](#page-9-8) contains ≈ 120, 000 images with 22,634 different classes and nearly a 50:50 (training:testing) split.

InShop [\(Liu et al.,](#page-9-14) [2016\)](#page-9-14) consists of over 52k images with 7,982 different clothing types as classes.

MS Marco [\(Bajaj et al.,](#page-8-12) [2016\)](#page-8-12) focuses on textual retrieval, requiring the identification of relevant passages from a corpus containing 500,000 training examples and 8.8 million passages based on Bing queries.

278

289 290

294

296

298

300

304

306

308 309

311

314 315 316

318

324

326

328

Table 2: Quantitative comparison between multiple sampling methods on the MS MARCO dataset. Results are reported for MRR@10.

| Approach  |     |        | MRR@10 |
|-----------|-----|--------|--------|
| Random    |     |        | 20.07  |
| BatchHard |     |        | 20.59  |
| Pre-Epoch |     | Incr.  | 26.23  |
| LSH       | 128 | (ours) | 24.41  |
| LSH       | 256 | (ours) | 25.67  |
| LSH       | 512 | (ours) | 26.42  |
| LSH       | 768 | (ours) | 26.44  |
| Pre-Epoch |     | Full   | 26.24  |

#### 4.4. Sampling Strategies

For our evaluation, we compare multiple sampling methods with our proposed LSH sampling:

Random Sampling For the random sampling strategy random pairs are sampled, allowing HNs only by chance. The only filtering is done on the class level to prevent multiple instances of a class from being in a batch. This approach does not add computational overhead.

Pre-Epoch Full Sampling For pre-epoch full sampling, HNs are pre-computed before each epoch by extracting the full training dataset and selecting negatives based on cosine similarity from the similarity matrix. This sampling is the most resource intensive method, as it requires a reprocessing of the complete dataset.

Pre-Epoch Incremental Sampling For pre-epoch incremental sampling, HNs are extracted during training using saved embeddings before weight updates. This method is faster compared to *pre-epoch full sampling* but relies on embeddings that might partially not be updated yet.

BatchHard Sampling Since the loss method can influence Hard Negative Sampling (HNS) selection, we follow Schroff et al. [\(Schroff et al.,](#page-9-1) [2015\)](#page-9-1) and implement Batch-Hard for the InfoNCE loss. BatchHard calculates the loss using only the 50% hardest negatives within a batch.

## 4.5. Impact of Different Sampling Strategies

We evaluate all sampling strategies on the used datasets. As shown in Section [3.2,](#page-4-0) random sampling consistently underperforms any form of sampling. While BatchHard sampling can achieve higher performance, retaining only 50% of the HNs results in marginal improvements, since HNs are not explicitly selected. In addition, BatchHard Sampling has the disadvantage of discarding some computed results because

it artificially limits the number of HNs used.

Pre-Epoch Full Sampling, while the slowest approach, often yields the best performance since embeddings are extracted after each completed epoch. This allows the model to generalize effectively throughout the training process.

Comparing Pre-Epoch Incremental and LSH, we obtain slightly worse performance when the bit dimension is low (128 or 256) and the same performance when the bit dimension is high (512 or 1024), while being faster and requiring less space to store the vector embeddings.

Similar results are observed for text retrieval, see Section [4.3,](#page-5-0) where random sampling underperforms, and HN improves performance. Unlike vision tasks, higher LSH bit counts further boost performance over Pre-Epoch full sampling.

Additionally, we compare the speed and space costs of the pre-epoch incremental and LSH sampling for searching and calculation. Furthermore, we investigate the relationships between sampled neighbors by LSH and the actual hardest negatives determined via the cosine similarity matrix.

## 4.6. Search Speed Comparison

In Figure [1,](#page-0-0) we present a comparison of search times between our LSH-based feature encoding with different bit sizes (128, 256, 512, 768, 1024) and HN sampling using float32 vector embeddings. To ensure a fair evaluation, we used the FAISS library [\(Douze et al.,](#page-8-14) [2024\)](#page-8-14) to retrieve the top 128 Nearest Neighbor (NN) in each configuration.

Although the theoretical query time for all indices is O(n · d), the use of binary features (LSH) leads to significant speedups. For this experiment, we encoded the training data of each dataset, except CVACT, since its size is identical to that of CVUSA. For datasets with a reference-query split, we performed the search within the query set to retrieve 128 NN for each reference.

The gap between LSH-based coding and full vector embeddings remains consistent across datasets, including large datasets such as MS MARCO with over 500,000 samples, where the search time remains significantly shorter compared to indices that compute cosine similarity with full precision. The reported times represent the duration required per epoch during training for searching, resulting in a significant impact on the overall training time.

## 4.7. Neighbor Analysis

We further investigate the behavior of HNs selection when LSH is used. For this analysis, we use the MS-MARCO dataset with over 500k samples and the SOP dataset, which contains over 60k samples, to comprehensively evaluate the generalization of our approach. In our appendix results for the VIGOR dataset can be found as well. In each setting,

334

336

338

351

354

356

358

360 361

364

366

368

371

374

378

![](_page_6_Figure_1.jpeg)

Figure 3: A comparison of overlap and mean positional distance of LSH with varying bit sizes (128, 256, 512, and 1024) and random sampling. The overlap with Pre-Epoch Increment (HNs) and mean positional distance (right) on MS MARCO.

we retrieve the top 128 NNs and rank them based on their similarity to the reference feature. This allows us to compute the mean positional distance between the ANNs obtained by LSH and the actual NNs determined by cosine similarity. Furthermore, we evaluate the overlap between the retrieved neighbors and the real nearest neighbors to quantify the effectiveness of the approximation.

#### 4.7.1. NEIGHBOR OVERLAP

We compare the overlap between the examples used in a batch and those retrieved based on cosine similarity. For SOP, as shown Figure [4,](#page-6-0) the overlap with random sampling is approximately one percent. In contrast, using LSH significantly increases the overlap, and our strategy achieves around 70% overlap at the highest bit count.

Based on the evaluation presented in Section [3.2](#page-4-0) and the overlap for 54% for 512 bits, this precision appears sufficient to provide enough HNs for achieving satisfying performance. It is interesting to note that especially in lower bit regimes (128 and 256) the overlap reduces over time. This may seem counterintuitive at first, but it is consistent with the goal of the loss function. The loss function is designed to encourage negative samples that are close neighbors to become more dissimilar. As a result, it becomes increasingly difficult to identify real HNs, especially considering that the probability of selection depends on similarity, as described in Section [3.2.](#page-2-0) In the mean positional distance plot (see Figure [4\)](#page-6-0), random sampling converges to the center of the dataset. This is expected, as positional relationships do not influence its selection. In contrast, the 1024-bit LSH remains very close to zero, with fluctuations increasing as the number of bits decreases.

When analyzing the textual modality, we can observe a different behavior, as shown in Figure [3.](#page-6-1) The overlap is notably lower compared to the vision task, and the mean positional

![](_page_6_Figure_2.jpeg)

Figure 4: A comparison of LSH with varying bit sizes (128, 256, 512, and 1024) and random sampling is presented across two metrics: overlap with Pre-Epoch Increment (HNs) and mean positional distance (right) on SOP.

distance remains far from zero, even with higher bit counts. While the real HNs can still be identified, the process for the textual modality is considerably more challenging compared to the vision task. This may stem from the nature of text, which captures multiple concepts and fine-grained distinctions, unlike the broader and more cohesive concepts typical of images. Nevertheless, 512 bits still provide strong performance, even surpassing pre-epoch incremental and full sampling.

Furthermore, we compare the cosine similarity between the retrieved and actual HNs, with details in the appendix.

#### 4.7.2. NEIGHBOR HARDNESS

The question of how many HNs (HNs) are required to maintain robust performance in HN sampling is crucial to understanding the trade-offs associated with using ANN. As detailed in Section [4.7,](#page-5-1) the LSH algorithm achieves approximately 70% overlap with real HNs. This study evaluates how performance improves when more hard samples are used and hardness defines the percentage in a batch that is a HN. Figure [5](#page-7-0) shows the results of training for 10 epochs with different levels of hardness. A hardness level of 1.0 represents the setting used for pre-epoch incremental sampling where always all HNs are retrieved and 0.0 represents random sampling. We also include our LSH results based on the overlap from Figure [4.](#page-6-0) While the performance on SOP remains similar, our method significantly improves results on MS Marco, even at the same hardness levels. This demonstrates that LSH selects more effective HN, even if they are not the real true HN, compared to random selection. For VIGOR, we present the same experiment in the appendix.

394

396

![](_page_7_Figure_1.jpeg)

Figure 5: Impact of HN hardness on R@1 on SOP (left) and MRR@10 on MSMarco (right). We define hardness as the percentage of HNs within a batch and include the results from LSH based on the respective overlap.

#### 4.8. LSH Design Choices

In Section [4.7.1](#page-6-2) we showed how a smaller bit size results in less overlap with the actual NN, and as we can see in our performance evaluation, Section [3.2,](#page-4-0) less overlap results in worse benchmark scores on the dataset. We now want to investigate how the design choices of our method improve this overlap. As described in Section [3.2,](#page-2-0) we use a random rotation matrix with orthonormal vectors and center the projected features. To obtain orthonormal vectors, we use QR decomposition. As shown in Figure [6,](#page-7-1) centering consistently improves the overlap between the found NN and the actual NN, especially in lower dimensional hash spaces, by reducing the skew in the binary representations.

For the MS MARCO dataset, the use of orthonormal matrices becomes increasingly important at higher bit dimensions, preserving feature variance and improving alignment with neighbours retrieved with the cosine similarity. Without orthonormalization, overlap performance degrades, especially in high-dimensional spaces.

## 5. Conclusion

We show that LSH, through its inherent properties such as locality preservation and similarity-based retrieval, effectively approximates real NNs and enables robust HNs mining. Our experiments show that even at lower bit sizes (e.g., 256 or 512), the sampled neighbors achieve strong overlap with real NNs and generalize better than traditional methods such as BatchHard [\(Schroff et al.,](#page-9-1) [2015\)](#page-9-1) or random sampling. This confirms that LSH can serve as reliable and efficient technique for HN sampling in contrastive learning.

Further, we show that the use of LSH significantly reduces the time and space complexity associated with traditional pre-epoch sampling. The binarization process naturally re-

![](_page_7_Figure_2.jpeg)

Figure 6: Impact of our LSH design choices on the overlap for SOP and MS MARCO.

duces memory requirements, while the slower scaling of search time allows our approach to handle larger datasets more efficiently compared to exact search methods. Despite these reductions, the performance and quality of HN selection remain competitive or performs even better, further validating the practical advantages of our method.

## 6. Discussion

In our work, we focus on supervised learning and acknowledge a key challenge in finding HNs: positive examples and identified HNs may belong to the same underlying class, particularly in unsupervised settings. This overlap complicates model convergence. While this issue has been explored in related literature [\(Robinson et al.,](#page-9-18) [2020;](#page-9-18) [Chuang et al.,](#page-8-15) [2020\)](#page-8-15), we do not address it further here, leaving it as a limitation for future research.

As explained in Section [4.7.2](#page-6-3) the overall quality of our selected ANN delivers solid experimental results. Nonetheless, providing a quality guarantee for the identified HNs would be advantageous, similar to the ANN guarantees in [\(Har-Peled et al.,](#page-9-13) [2012\)](#page-9-13). Furthermore, it remains an open question whether theoretical bounds on the performance of the final embedding can be derived from such quality guarantees for the ANN.

Another interesting question remains the comparison of our LSH-based approach with other ANN methods like Product Quantization [\(Jegou et al.](#page-9-19) ´ , [2011\)](#page-9-19) or Hierarchical Navigable Small World graphs [\(Fu et al.,](#page-8-16) [2019\)](#page-8-16).

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 This work proposes a more efficient sampling approach for contrastive learning that focuses on approximating HNs to enable faster training and reducing computational cost. We demonstrate the effectiveness of the method on general retrieval tasks such as dense passage retrieval and product search. Recognizing the ethical concerns of person re-identification, we deliberately avoid using such datasets and limit our engagement with this topic to a literature review. This work aims to advance contrastive learning while promoting responsible and sustainable research practices. References An, X., Deng, J., Yang, K., Li, J., Feng, Z., Guo, J., Yang, J., and Liu, T. Unicom: Universal and compact representation learning for image retrieval. *arXiv preprint arXiv:2304.05884*, 2023. Andoni, A., Indyk, P., Laarhoven, T., Razenshteyn, I., and Schmidt, L. Practical and optimal lsh for angular distance. *Advances in neural information processing systems*, 28, 2015. Bajaj, P., Campos, D., Craswell, N., Deng, L., Gao, J., Liu, X., Majumder, R., McNamara, A., Mitra, B., Nguyen, T., et al. Ms marco: A human generated machine reading comprehension dataset. *arXiv preprint arXiv:1611.09268*, 2016. Cakir, F., He, K., Xia, X., Kulis, B., and Sclaroff, S. Deep metric learning to rank. In *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 1861–1870, 2019. doi: 10.1109/CVPR.2019.00196. Charikar, M. S. Similarity estimation techniques from rounding algorithms. In *Proceedings of the Thiry-Fourth Annual ACM Symposium on Theory of Computing*, STOC '02, pp. 380–388, New York, NY, USA, 2002. Association for Computing Machinery. ISBN 1581134959. doi: 10.1145/509907.509965. URL [https://doi.org/](https://doi.org/10.1145/509907.509965) [10.1145/509907.509965](https://doi.org/10.1145/509907.509965). Chopra, S., Hadsell, R., and LeCun, Y. Learning a similarity metric discriminatively, with application to face verification. In *2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05)*, volume 1, pp. 539–546 vol. 1, 2005. doi: 10.1109/CVPR.2005.202. Chuang, C.-Y., Robinson, J., Lin, Y.-C., Torralba, A., and Jegelka, S. Debiased contrastive learning. *Advances in neural information processing systems*, 33:8765–8775, 2020. Datar, M., Immorlica, N., Indyk, P., and Mirrokni, V. S. Locality-sensitive hashing scheme based on p-stable distributions. In *Proceedings of the Twentieth Annual Symposium on Computational Geometry*, SCG '04, pp. 253–262, New York, NY, USA, 2004. Association for Computing Machinery. ISBN 1581138857. doi: 10. 1145/997817.997857. URL [https://doi.org/10.](https://doi.org/10.1145/997817.997857) [1145/997817.997857](https://doi.org/10.1145/997817.997857). Deuser, F., Habel, K., and Oswald, N. Sample4geo: Hard negative sampling for cross-view geo-localisation. In *ICCV*, pp. 16847–16856, 2023a. Deuser, F., Habel, K., Werner, M., and Oswald, N. Orientation-guided contrastive learning for uav-view geolocalisation. In *Proceedings of the 2023 Workshop on UAVs in Multimedia: Capturing the World from a New Perspective*, pp. 7–11, 2023b. Deuser, F., Werner, M., Habel, K., and Oswald, N. Optimizing geo-localization with k-means re-ranking in challenging weather conditions. In *Proceedings of the 2nd Workshop on UAVs in Multimedia: Capturing the World from a New Perspective*, pp. 9–13, 2024. Ding, S., Lin, L., Wang, G., and Chao, H. Deep feature learning with relative distance comparison for person reidentification. *Pattern Recognition*, 48(10):2993–3003, 2015. Dosovitskiy, A. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020. Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazare, P.-E., Lomeli, M., Hosseini, L., and J ´ egou, H. ´ The faiss library. 2024. Fu, C., Xiang, C., Wang, C., and Cai, D. Fast approximate nearest neighbor search with the navigating spreadingout graph. *Proc. VLDB Endow.*, 12(5):461–474, January 2019. ISSN 2150-8097. doi: 10.14778/3303753. 3303754. URL [https://doi.org/10.14778/](https://doi.org/10.14778/3303753.3303754) [3303753.3303754](https://doi.org/10.14778/3303753.3303754). Galanopoulos, D. and Mezaris, V. Hard-negatives or nonnegatives? a hard-negative selection strategy for crossmodal retrieval using the improved marginal ranking loss. In *CVPR*, pp. 2312–2316, 2021. Gillick, D., Kulkarni, S., Lansing, L., Presta, A., Baldridge, J., Ie, E., and Garcia-Olano, D. Learning dense representations for entity retrieval. *arXiv preprint arXiv:1909.10506*, 2019. Habel, K., Deuser, F., and Oswald, N. Clip-reident: Contrastive training for player re-identification. In *Proceedings of the 5th International ACM Workshop on*

494

## Impact Statement

495 496 497 498 499 500 504 506 508 509 511 514 515 516 518 524 526 528 531 534 536 538 540 541 542 543 544 545 546 547 548 *Multimedia Content Analysis in Sports*, MMSports '22, pp. 129–135, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450394888. doi: 10.1145/3552437.3555698. URL [https://doi.](https://doi.org/10.1145/3552437.3555698) [org/10.1145/3552437.3555698](https://doi.org/10.1145/3552437.3555698). Har-Peled, S., Indyk, P., and Motwani, R. Approximate nearest neighbor: Towards removing the curse of dimensionality. *Theory of Computing*, 8(14):321–350, 2012. doi: 10.4086/toc.2012. v008a014. URL [https://theoryofcomputing.](https://theoryofcomputing.org/articles/v008a014) [org/articles/v008a014](https://theoryofcomputing.org/articles/v008a014). He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. Momentum contrast for unsupervised visual representation learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 9729–9738, 2020. Hermans, A., Beyer, L., and Leibe, B. In defense of the triplet loss for person re-identification. *arXiv preprint arXiv:1703.07737*, 2017. Jia, C., Yang, Y., Xia, Y., Chen, Y.-T., Parekh, Z., Pham, H., Le, Q., Sung, Y.-H., Li, Z., and Duerig, T. Scaling up visual and vision-language representation learning with noisy text supervision. In *International conference on machine learning*, pp. 4904–4916. PMLR, 2021. Jegou, H., Douze, M., and Schmid, C. Product quantization ´ for nearest neighbor search. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 33(1):117–128, 2011. doi: 10.1109/TPAMI.2010.57. Liu, L. and Li, H. Lending orientation to neural networks for cross-view geo-localization. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 5624–5633, 2019. Liu, Y. Roberta: A robustly optimized bert pretraining approach. *arXiv preprint arXiv:1907.11692*, 364, 2019. Liu, Z., Luo, P., Qiu, S., Wang, X., and Tang, X. Deepfashion: Powering robust clothes recognition and retrieval with rich annotations. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 1096–1104, 2016. Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., and Xie, S. A convnet for the 2020s. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 11976–11986, 2022. Oh Song, H., Xiang, Y., Jegelka, S., and Savarese, S. Deep metric learning via lifted structured feature embedding. In *CVPR*, pp. 4004–4012, 2016. Oord, A. v. d., Li, Y., and Vinyals, O. Representation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*, 2018. Patel, Y., Tolias, G., and Matas, J. Recall@ k surrogate loss with large batches and similarity mixup. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 7502–7511, 2022. Pedersen, T., Patwardhan, S., Michelizzi, J., et al. Wordnet:: Similarity-measuring the relatedness of concepts. In *AAAI*, volume 4, pp. 25–29, 2004. Qu, Y., Ding, Y., Liu, J., Liu, K., Ren, R., Zhao, W. X., Dong, D., Wu, H., and Wang, H. Rocketqa: An optimized training approach to dense passage retrieval for open-domain question answering. *arXiv preprint arXiv:2010.08191*, 2020. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pp. 8748–8763. PMLR, 2021. Reimers, N. and Gurevych, I. Sentence-bert: Sentence embeddings using siamese bert-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 11 2019. URL [https://arxiv.org/](https://arxiv.org/abs/1908.10084) [abs/1908.10084](https://arxiv.org/abs/1908.10084). Reimers, N. and Gurevych, I. Making monolingual sentence embeddings multilingual using knowledge distillation. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 11 2020. URL [https:](https://arxiv.org/abs/2004.09813) [//arxiv.org/abs/2004.09813](https://arxiv.org/abs/2004.09813). Robinson, J., Chuang, C.-Y., Sra, S., and Jegelka, S. Contrastive learning with hard negative samples. *arXiv preprint arXiv:2010.04592*, 2020. Sanh, V., Debut, L., Chaumond, J., and Wolf, T. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. *ArXiv*, abs/1910.01108, 2019. Schroff, F., Kalenichenko, D., and Philbin, J. Facenet: A unified embedding for face recognition and clustering. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 815–823, 2015. Simo-Serra, E., Trulls, E., Ferraz, L., Kokkinos, I., Fua, P., and Moreno-Noguer, F. Discriminative learning of deep convolutional feature point descriptors. In *2015 IEEE International Conference on Computer Vision (ICCV)*, pp. 118–126, 2015. doi: 10.1109/ICCV.2015.22.

549

- 551 554 556 558 560 564 566 568 571 574 576 578 580 581 582 583 584 585 586 587 588 589 590 594 596 598 600 601 602 603 604 Vaswani, A. Attention is all you need. *Advances in Neural Information Processing Systems*, 2017. Wang, J., Liu, W., Kumar, S., and Chang, S.-F. Learning to hash for indexing big data—a survey. *Proceedings of the IEEE*, 104(1):34–57, 2015. Wang, X., Han, X., Huang, W., Dong, D., and Scott, M. R. Multi-similarity loss with general pair weighting for deep metric learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 5022–5030, 2019. Wightman, R. Pytorch image models. [https://github.](https://github.com/rwightman/pytorch-image-models) [com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models), 2019. Workman, S., Souvenir, R., and Jacobs, N. Wide-area image geolocalization with aerial reference imagery. In *2015 IEEE International Conference on Computer Vision (ICCV)*, pp. 3961–3969, 2015. doi: 10.1109/ICCV.2015.
  - 451. Wu, C.-Y., Manmatha, R., Smola, A. J., and Krahenbuhl, P. Sampling matters in deep embedding learning. In *CVPR*, pp. 2840–2848, 2017. Xiong, L., Xiong, C., Li, Y., Tang, K.-F., Liu, J., Bennett, P., Ahmed, J., and Overwijk, A. Approximate nearest neighbor negative contrastive learning for dense text retrieval. *arXiv preprint arXiv:2007.00808*, 2020. Xuan, H., Stylianou, A., and Pless, R. Improved embeddings with easy positive triplet mining. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pp. 2474–2482, 2020. Yuan, Y., Yang, K., and Zhang, C. Hard-aware deeply cascaded embedding. In *Proceedings of the IEEE international conference on computer vision*, pp. 814–823, 2017. Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 11975–11986, 2023. Zhang, R., Wu, L., Yang, Y., Wu, W., Chen, Y., and Xu,
- M. Multi-camera multi-player tracking with deep player identification in sports video. *Pattern Recognition*, 102: 107260, 2020. Zhu, S., Yang, T., and Chen, C. Vigor: Cross-view image geo-localization beyond one-to-one retrieval. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 3640–3649, 2021. Zhu, S., Shah, M., and Chen, C. Transgeo: Transformer is all you need for cross-view image geo-localization. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 1162–1171, 2022.

![](_page_11_Figure_4.jpeg)

#### A. Appendix

#### A.1. Similarity Distribution Analysis

We further examine the distribution of similarities between the ANN identified by our sampling method and the real NN retrieved by cosine similarity, as shown in Figure [7.](#page-11-0) Notably, while random sampling produces a uniform similarity distribution, increasing the bit count consistently shifts the distribution toward 0.5, regardless of the dataset used. This highlights the advantage of using LSH, as the similarity in the original embedding space affects the probability of hash collisions. Even if the true NN is not found, the retrieved examples are better than those obtained by random sampling.

Figure 7: Comparison of the similarity between the retrieved approximated HNs and the actual HN retrieved by the cosine similarity for SOP (top), VIGOR(middle) and MS MARCO (bottom).

### A.2. VIGOR Analysis

Furthermore, we also compare the overlap and mean positional distance a subset of VIGOR in Figure [8.](#page-12-0) In this subset we only use the city of Seattle for training and evaluation. Similar to the other datasets the overlap declines over time as the embeddings of pairs are pushed afar from each other. We also investigate the impact of hardness during training in Figure [9,](#page-12-1) increasing hardness of the sample improves the Recall@1 on the VIGOR dataset. We further include the results achieved with LSH based on our overlap depicted in Figure [8.](#page-12-0) Similar to MS MARCO we achieve higher values of Recall@1 while

![](_page_12_Figure_1.jpeg)

Figure 8: A comparison of overlap and mean positional distance of LSH with varying bit sizes (128, 256, 512, and 1024) and random sampling. The overlap with Pre-Epoch Increment (HNs) and mean positional distance (right) on VIGOR.

the overlap remains the same.

![](_page_12_Figure_4.jpeg)

Figure 9: Impact of HN hardness on R@1 on VIGOR. We define hardness as the percentage of HNs within a batch and include the results from LSH based on the respective overlap.

## A.3. Further Implementation Details

For our image-based experiments, we apply several data augmentation techniques during training, including flipping, rotation, coarse dropout, grid dropout, and color jitter. These augmentations help improve model generalization by introducing variability into the training samples.

In supervised settings, where multiple positive pairs exist for a given label, we structure our batches to contain only one positive pair per label. This approach minimizes redundancy and reduces noise in the loss computation, ensuring a more stable training process.

We train for 40 epochs on the CVUSA, CVACT, and VIGOR datasets, which focus on cross-view geo-localization tasks. For datasets with a different retrieval structure, such as Stanford Online Products (SOP), InShop, and the MS MARCO text retrieval dataset, we limit training to 10 epochs to avoid overfitting. For the cross-view dataset, we resize the images for CVUSA and CVACT to 384 × 384 for the satellite image and 112 × 616 for the street view image, for VIGOR we use × 384 for the satellite view and 384 × 768 for the street view. In the experiments for SOP and InShop, all images are resized to 384 × 384.

718

724

726

728

![](_page_13_Diagram_6.jpeg)

754

756

758

760

764

766

#### A.4. Training Process:

Figure [10](#page-13-0) illustrates the process of encoding arbitrary input data, such as text or images, using an encoder to generate embeddings. LSH is applied to transform these embeddings into binary vectors. After each epoch, pairwise search based on Hamming distance is used to sample HNs, for the next epoch. Similar to sampling on the float32 embedding, we define the HN as the negative sample with the smallest distance to the anchor.

## A.5. Architecture Details:

For our experiments on image datasets, we use the ConvNeXt base model [\(Liu et al.,](#page-9-16) [2022\)](#page-9-16), pre-trained on ImageNet-21k, from the timm library [\(Wightman,](#page-10-12) [2019\)](#page-10-12). ConvNeXt modernizes the ResNet architecture by incorporating design principles from the Vision Transformer [\(Dosovitskiy,](#page-8-17) [2020\)](#page-8-17). The model outputs 1024-dimensional embeddings and consists of 88 million parameters.

For our experiments on the MS MARCO text retrieval dataset, we use Distill-RoBERTa-base [\(Sanh et al.,](#page-9-17) [2019\)](#page-9-17), a distilled version of RoBERTa [\(Liu,](#page-9-20) [2019\)](#page-9-20), with 82 million parameters. The hidden size of the transformer is 768. In both cases, we choose these models to allow efficient training and evaluation of our methodology. Additionally, we employ shared weights for both reference and query inputs.

Figure 10: Framework for encoding input data and leveraging LSH for binary transformation and hard negative sampling during contrastive learning.