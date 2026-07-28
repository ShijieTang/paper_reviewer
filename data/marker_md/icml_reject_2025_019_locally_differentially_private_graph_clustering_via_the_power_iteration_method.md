011

014 015 016

018

024

026

034

036

038

# Locally Differentially Private Graph Clustering via the Power Iteration Method

## Abstract

We propose a locally differentially private graph clustering algorithm. Previous works have explored this problem, including approaches that apply spectral clustering to graphs generated via the randomized response algorithm. However, these methods only achieve accurate results when the privacy budget is in Ω(log n), which is unsuitable for many practical applications. In response, we present an interactive algorithm based on the power iteration method. Given that the noise introduced by the largest eigenvector constant can be significant, we incorporate a technique to eliminate this constant. As a result, our algorithm attains local differential privacy with a constant privacy budget when the graph is well-clustered and has a minimum degree of Ω( ˜ √ n). In contrast, while randomized response has been shown to produce accurate results under the same minimum degree condition, it is limited to graphs generated from the stochastic block model. We perform experiments to demonstrate that our method outperforms spectral clustering applied to randomized response results.

## 1. Introduction

As the adoption of artificial intelligence expands, ensuring the protection of user privacy has become a critical priority. Various techniques have been proposed to tackle privacy concerns, with differential privacy emerging as a leading approach. Differential privacy, introduced in [\(Dwork,](#page-8-0) [2008\)](#page-8-0), quantifies the privacy leakage of a system using a parameter known as the privacy budget. The core idea involves introducing noise to users' data to obscure individual information while still enabling meaningful statistical analysis. The challenge of designing algorithms that can draw accurate insights from this noisy data has garnered significant attention from researchers [\(Zhu et al.,](#page-9-0) [2017\)](#page-9-0), as it is essential to balance privacy protection with the utility of the resulting analysis.

### . AUTHORERR: Missing \icmlcorrespondingauthor.

Preliminary work. Under review by the International Conference on Machine Learning (ICML). Do not distribute.

In this work, we focus on a specific variant of differential privacy known as local differential privacy (LDP) [\(Ka](#page-8-1)[siviswanathan et al.,](#page-8-1) [2011\)](#page-8-1). Unlike traditional differential privacy, which allows data collection before noise is added, LDP requires users to anonymize their data directly on their local devices before transmitting it to a central server. This approach ensures that sensitive information remains protected during transmission, as the data is already corrupted at the source. LDP has been adopted by several major companies [\(Erlingsson et al.,](#page-8-2) [2014;](#page-8-2) [Apple's Differential Privacy](#page-8-3) [Team,](#page-8-3) [2017\)](#page-8-3) in their services to safeguard user privacy while still enabling data analysis at scale.

We focus on developing LDP algorithms for social networks, where users are represented as nodes and their relationships as edges. Since these connections are considered sensitive, they are protected using privacy notions such as edge LDP [\(Qin et al.,](#page-9-1) [2017\)](#page-9-1) or node LDP [\(Ye et al.,](#page-9-2) [2020\)](#page-9-2). However, with some exceptions like [\(Zhang et al.,](#page-9-3) [2020\)](#page-9-3), node LDP is generally too stringent, making it difficult to release useful information in most applications. As a result, the majority of research in LDP has centered around the more practical edge LDP framework [\(Imola et al.,](#page-8-4) [2021\)](#page-8-4).

To protect user's information, one widely used technique is randomized response, also known as edge flipping [\(Warner,](#page-9-4) [1965;](#page-9-4) [Mangat,](#page-8-5) [1994;](#page-8-5) [Wang et al.,](#page-9-5) [2016\)](#page-9-5). In this method, before a user sends a bit vector which encodes their list of friends to a central server, each bit in the vector is flipped with a certain probability. The server aggregates the obfuscated adjacency vector to construct an obfuscated version of the graph. Although it is possible to compute various graph statistics from this obfuscated data, the accuracy of these statistics is often reduced. Algorithms designed specifically to publish particular statistics tend to offer more precise and insightful results about the graph [\(Imola et al.,](#page-8-4) [2021;](#page-8-4) [2022\)](#page-8-6).

Graph clustering illustrates how analyzing a graph obfuscated by randomized response can lead to inaccurate results. Let n be the number of nodes in the input graph. In [\(Hehir](#page-8-7) [et al.,](#page-8-7) [2022\)](#page-8-7), the authors demonstrated that spectral clustering [\(Ng et al.,](#page-9-6) [2001\)](#page-9-6) can yield accurate results with a privacy budget in O(1), provided the input graphs are generated from stochastic block models and have an average degree of Θ(√ n) [\(Holland et al.,](#page-8-8) [1983\)](#page-8-8). For general graphs, [\(Mukher](#page-9-7)[jee & Suppakitpaisarn,](#page-9-7) [2023\)](#page-9-7) showed that applying spectral clustering to randomized response data only yields accurate

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

108 109 results when the privacy budget ϵ ∈ Ω(log n), which is too large for many real-world applications. Furthermore, even for dense graphs, when ϵ ∈ o(log n), the authors identified a class of graphs for which clustering results are inaccurate.

Although numerous algorithms have been proposed for clustering under differential privacy [\(Ji et al.,](#page-8-9) [2020;](#page-8-9) [Mohamed](#page-8-10) [et al.,](#page-8-10) [2022;](#page-8-10) [Chen et al.,](#page-8-11) [2023;](#page-8-11) [Imola et al.,](#page-8-12) [2023;](#page-8-12) [Epasto](#page-8-13) [et al.,](#page-8-13) [2024;](#page-8-13) [He et al.,](#page-8-14) [2024\)](#page-8-14), relatively few have been developed specifically for publishing clustering results under edge LDP. Aside from the work mentioned in the previous paragraph, the only other algorithm we are aware of targets node LDP rather than edge LDP [\(Fu et al.,](#page-8-15) [2023\)](#page-8-15).

#### 1.1. Our Contributions

In this work, we aim to develop a dedicated algorithm for graph clustering under the edge LDP framework. Rather than using non-interactive methods like the randomized response algorithm, we propose an interactive approach, which has been shown to achieve better performance for many edge LDP tasks [\(Henzinger et al.,](#page-8-16) [2024;](#page-8-16) [Hillebrand](#page-8-17) [et al.,](#page-8-17) [2024\)](#page-8-17).

Specifically, we draw inspiration from the work in [\(Betzer](#page-8-18) [et al.,](#page-8-18) [2024\)](#page-8-18), where the authors employ multi-round interactive algorithms to compute iterative matrix multiplications for Katz centrality. Since spectral clustering can also be derived through iterative matrix multiplication using the Power Iteration Clustering (PIC) algorithm [\(Lin & Cohen,](#page-8-19) [2010;](#page-8-19) [Boutsidis et al.,](#page-8-20) [2015\)](#page-8-20), we propose extending this approach to calculate clusters via the PIC algorithm under the edge LDP framework.

Unfortunately, calculating the PIC algorithm under the edge LDP framework is not straightforward. While the goal is to compute the second eigenvector through the iterative process, the largest component of the result is the first eigenvector. In a non-private setting, the first eigenvector, being a uniform vector, does not interfere with the calculation of the PIC algorithm. However, when protecting users' sensitive information under edge LDP, noise must be added at a magnitude comparable to the largest terms. This causes the noise to dominate the result, especially as the number of iterations increases, leading to a significant loss in accuracy.

We propose a technique to eliminate the largest constant term, enabling the development of an algorithm that achieves accurate results with a constant privacy budget when the minimum degree of the input graph is Ω( ˜ √ n). Recall that randomized response is proven to yield accurate results for graphs generated by the stochastic block model when the minimum degree is Ω( ˜ √ n). Our algorithm, however, provides precise results under the same minimum degree condition but applies to general graphs, not limited to those generated by the model. This extends the applicability

of our clustering algorithm to a wider range of input graphs.

Our algorithm is computationally efficient. It requires O(log n) interactions between users and the central server, with each node having a computational complexity of O(n) per iteration. The central server also has a computational complexity of O(n) per iteration. Consequently, the total computation time of our algorithm is O(n log n). Additionally, the communication cost for each user is also O(n log n).

Compared to the spectral clustering algorithm applied to the randomized response results [\(Hehir et al.,](#page-8-7) [2022;](#page-8-7) [Mukherjee](#page-9-7) [& Suppakitpaisarn,](#page-9-7) [2023\)](#page-9-7), our iterative method is significantly more memory-efficient. In the previous approach, the server requires Θ(n 2 ) bits of memory to store the randomized response results [\(Imola et al.,](#page-8-6) [2022;](#page-8-6) [Hillebrand](#page-8-21) [et al.,](#page-8-21) [2023\)](#page-8-21). In contrast, our algorithm reduces the memory requirement to Θ(n) for both the server and the users. This improvement enables our method to handle graphs with a large number of nodes, which would be infeasible to process using the earlier algorithm.

We validate our algorithm through experiments on graphs generated using the stochastic block model [\(Holland et al.,](#page-8-8) [1983\)](#page-8-8) and the Reddit graph [\(Hamilton et al.,](#page-8-22) [2017\)](#page-8-22). Compared to applying the spectral clustering algorithm to the randomized response results [\(Hehir et al.,](#page-8-7) [2022\)](#page-8-7), our algorithm produces clustering results that are closer to those of the original spectral clustering algorithm in almost all cases. Notably, there are instances where the previous algorithm yields random outcomes, while our algorithm consistently produces results identical to the original spectral clustering.

## 2. Preliminaries

#### 2.1. Notation

Throughout this paper, we consider a graph G = (V, E) with n vertices. Let S ⊆ V represent a subset of vertices, and S denote its complement V \ S.

Let S and S ′ be two disjoint subsets of V (meaning S∩S ′ = <sup>∅</sup>). We denote by eG(S, S′ ) the number of edges in G that have one endpoint in S and the other in S ′ . For each subset S ⊆ V , let VolG(S) denote the number of edges with both endpoints in S. We refer to VolG(S) as the *volume* of S.

For S, S′ ⊆ V , the quantity dvol(S, S′ ) is defined as min(VolG(S△S ′ ) + VolG(S△S′), VolG(S△S′) + VolG(S△S ′ )). Since S△S ′ = S△S′ , this simplifies to dvol(S, S′ ) = min 2VolG(S△S ′ ), 2VolG(S△S′) . Two cuts (S, S) and (S ′ , S′) are considered similar if dvol(S, S′ ) is small. We also define the *normalized discrepancy* as

$$d_{\text{norm}}(S, S') = \frac{d_{\text{vol}}(S, S')}{\text{Vol}_G(V)}. \quad (1)$$

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

Given that dvol(S, S′ ) ≤ VolG(V ), normalization ensures that 0 ≤ dnorm(S, S′ ) ≤ 1. When S is fixed and nodes are randomly assigned to S ′ with uniform probability, dnorm(S, S′ ) tends to be close to 1.

Any real symmetric n × n matrix A has n real eigenvalues. We denote the i-th smallest eigenvalue of A as λi(A), so that λ1(A) ≥ λ2(A) ≥ · · · ≥ λn(A). The eigenvector corresponding to λi(A) is denoted by vi(A) = [νi,1, . . . , νi,n] ⊺ .

For each i ∈ [1, n], let a<sup>i</sup> = [ai,1, . . . , ai,n] ⊺ represent the adjacency list of user v<sup>i</sup> , where ai,j = 1 signifies the existence of an edge between v<sup>i</sup> and v<sup>j</sup> (i.e., (v<sup>i</sup> , v<sup>j</sup> ) ∈ E), and ai,j = 0 indicates no edge. The degree of node v<sup>i</sup> , denoted by d<sup>i</sup> , reflects the number of edges connected to v<sup>i</sup> . In the context of a locally differentially private algorithm, it is assumed that each user v<sup>i</sup> is aware only of their own adjacency vector a<sup>i</sup> , which contains sensitive personal information.

#### 2.2. Edge Local Differential Privacy

We define two adjacency lists, a and a ′ , as neighboring if they differ by exactly one bit, meaning that one can be transformed into the other by either adding or removing a single edge connected to node v<sup>i</sup> . The concept of edge local differential privacy is formalized as follows:

Definition 2.1 (ϵ-Edge LDP Query). Let ϵ > 0. A randomized query R is said to satisfy ϵ-edge local differential privacy (ϵ-edge LDP) if, for any pair of neighboring adjacency lists a and a ′ , and any possible outcome set S, <sup>P</sup> [R(a) ∈ S] ≤ e <sup>ϵ</sup><sup>P</sup> [R(a ′ ) ∈ S].

Definition 2.2 (ϵ-edge LDP Algorithm [\(Qin et al.,](#page-9-1) [2017\)](#page-9-1)). An algorithm A is said to be ϵ-edge LDP if, for any user vi , and any sequence of queries R1, . . . , R<sup>κ</sup> posed to user vi , where each query R<sup>j</sup> satisfies ϵ<sup>j</sup> -edge local differential privacy (for 1 ≤ j ≤ κ), the total privacy loss is bounded by ϵ<sup>1</sup> + · · · + ϵ<sup>κ</sup> ≤ ϵ.

If an algorithm A is ϵ-edge LDP, it is also said to have a privacy budget of ϵ. Next, we introduce a query that satisfies ϵ-edge LDP which designed to estimate a realvalued statistic based on the adjacency vector.

Definition 2.3 (Edge Local Laplacian Query [\(Hillebrand](#page-8-21) [et al.,](#page-8-21) [2023\)](#page-8-21)). Let f : {0, 1} <sup>n</sup> → <sup>R</sup> be a function defined on adjacency lists, and let a ∼ a ′ represent neighboring adjacency lists. The global sensitivity of f, denoted as ∆<sup>f</sup> , is defined as: ∆<sup>f</sup> = maxa∼a′ |f(a) − f(a ′ )|.

For any ϵ > 0, a query that returns f(a) + Lap(∆<sup>f</sup> /ϵ) is ϵ-edge LDP. Here, Lap(b) refers to noise sampled from the Laplace distribution with scale parameter b.

#### 2.3. Spectral Clustering

For a given graph G, the primary objective of clustering techniques is to identify a cut (S, S) such that the number of edges crossing between S and S, denoted by eG(S, S), is minimized, while most of the edges are concentrated within S or S. To avoid trivial cuts (such as when S contains only a single vertex), it is common to define the *conductance*, ϕG(S) = eG(S, S)/ min{VolG(S), VolG(S)}, and seek cuts that minimize ϕG(S) [\(Shi & Malik,](#page-9-8) [2000\)](#page-9-8). The conductance of the graph, denoted by ϕ(G), is given by ϕ(G) = min∅⊊S⊊<sup>V</sup> ϕG(S). Unless otherwise stated, we use S ∗ to denote the subset that achieves the minimum normalized cut, where ϕG(S ∗ ) = ϕ(G).

Let B = (bi,j )1≤i,j≤<sup>n</sup> be the transition-probability matrix of a random walk on G, given by bi,i = 0 for all i and bi,j = ai,j/d<sup>i</sup> for all i ̸= j. We have that −1 ≤ λi(B) ≤ 1 for all <sup>i</sup>, <sup>λ</sup>1(B) = 1, and <sup>v</sup>1(B) = [ √ n , √ 1 n , . . . , √ 1 n ⊺ .

Observe that when I is the identity matrix, the matrix I −B is referred to as the *random walk normalized Laplacian matrix* [\(Von Luxburg,](#page-9-9) [2007\)](#page-9-9). The eigenvectors of I − B are identical to those of B. More specifically, it is known that, for all i, vi(I − B) = vn−i(B).

The spectral clustering algorithm [\(Shi & Malik,](#page-9-8) [2000\)](#page-9-8) computes the eigenvector v2(B) = [ν1, . . . , νn] ⊺ , and then produces the cut S ′ = {v<sup>i</sup> : ν<sup>i</sup> > 0} as the clustering result. Since ϕG(S ′ ) ≤ 2 p ϕG(S<sup>∗</sup>) [\(Alon,](#page-8-23) [1986\)](#page-8-23), it is established that the cut produced by the spectral clustering algorithm achieves a low conductance. Additionally, according to [\(Peng et al.,](#page-9-10) [2015\)](#page-9-10), we have dvol(S ′ , S<sup>∗</sup> ) = O ϕ(G) λ3(B) · VolG(V ) , indicating that S ′ closely approximates S ∗ in a graph that is well-clustered.

The normalized Laplacian matrix L = (ℓi,j )1≤i,j≤n, defined by ℓi,j = −ai,j/ p di · d<sup>j</sup> for i ̸= j and ℓi,i = 1, is commonly used in spectral clustering algorithms that aim to minimize the conductance. However, in this work, we opt for the random walk normalized Laplacian matrix, as calculating spectral clustering under the normalized Laplacian is more complex in the edge LDP setting. Notably, when the desired number of clusters is two, the results of spectral clustering using the random walk normalized Laplacian matrix are at least as good as those obtained with the normalized Laplacian matrix [\(Von Luxburg,](#page-9-9) [2007\)](#page-9-9).

#### 2.4. Power Iteration Clustering

While spectral clustering can produce a cut with a small cutratio, it requires computing the eigenvector v2(B), which can be computationally expensive. To address this, the power iteration clustering algorithm [\(Lin & Cohen,](#page-8-19) [2010\)](#page-8-19) offers a more efficient method for estimating the eigenvector, significantly reducing the computation time.

Let x be a vector of length n where each element is independently drawn from a Gaussian distribution. It is known that x can be expressed as c1λ1(B)v1(B) + · · · +

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

cnλn(B)vn(B), where c1, . . . , c<sup>n</sup> are independent random variables also drawn from a Gaussian distribution. Therefore, for a sufficiently large T, applying B<sup>T</sup> to x gives:

$$\begin{aligned} B^T \mathbf{x} &= c_1 \lambda_1(B)^T \mathbf{v}_1(B) + \dots + c_n \lambda_n(B)^T \mathbf{v}_n(B) \\ &= c_1 \left[ \frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}, \dots, \frac{1}{\sqrt{n}} \right]^T + c_2 \lambda_2(B)^T \mathbf{v}_2(B) \\ &\quad + \dots + c_n \lambda_n(B)^T \mathbf{v}_n(B). \end{aligned} \quad (2)$$

When λ3(B) ≪ λ2(B), the term B<sup>T</sup> x is approximately:

$$B^T \mathbf{x} \approx c_1 \left[ \frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}, \dots, \frac{1}{\sqrt{n}} \right]^T + c_2 \lambda_2(B)^T \mathbf{v}_2(B), \quad (3)$$

meaning the order of elements in B<sup>T</sup> x closely mirrors that of v2(B). Therefore, clustering can be performed using B<sup>T</sup> x instead of v2(B), yielding results similar to those from the spectral clustering algorithm.

#### 2.5. Assumptions

We assume that the input graph has the following properties: (1) The minimum degree is at least 2 √ n log<sup>4</sup> n,

- (2) There exists a constant g such that for all i ≥ 3, λi(B) + 1 ≤ λ2(B)+1 g ,
- (3) There exists δ ≈ 1 and γ < 1 such that the components of v2(B) satisfies n i : |ν<sup>i</sup> | ≥ √ γ n o  <sup>≥</sup> <sup>δ</sup> · <sup>n</sup>, and
- (4) The number of nodes n is larger than a constant C.

Assumption (1) The first assumption is essential for any graph clustering algorithm under edge LDP with a constant privacy budget. Protecting the connections of low-degree nodes requires adding so much noise that their contributions are obscured, resulting in unstable clustering outcomes for these nodes.

Assumption (2) The second assumption is a standard prerequisite for iterative spectral clustering algorithms, such as the one presented in [\(Boutsidis et al.,](#page-8-20) [2015\)](#page-8-20). This assumption ensures the convergence of the iterative process. A comprehensive technical explanation supporting this assumption is provided in [\(Boutsidis et al.,](#page-8-20) [2015\)](#page-8-20).

Assumption (3) We demonstrate in Appendix [A](#page-10-0) that the third assumption holds when the graph is well-clustered and most nodes have a degree cluster close to the average degree of the cluster to which they belong.

Specifically, for a node i in cluster A ⊆ V , we show in Proposition [A.1](#page-10-1) that the value of ν<sup>i</sup> exceeds √ σ·c 4 · q <sup>d</sup><sup>i</sup> <sup>n</sup>·d(A) −2 q <sup>ϕ</sup>(G) 1−λ3(B) , where d(A) represents the average degree of nodes in cluster A, and c, σ ∈ R satisfy the condition that at least c|A| nodes in cluster A have degrees not less than <sup>q</sup> <sup>σ</sup> · <sup>d</sup>(A). If the graph is well-clustered, the term ϕ(G) 1−λ3(B) becomes small and can be neglected [\(Mukherjee](#page-9-7)

[& Suppakitpaisarn,](#page-9-7) [2023\)](#page-9-7). Consequently, we conclude that when d<sup>i</sup> ≥ σd(A) and there are at least c|A| nodes satisfying this condition, it follows that ν<sup>i</sup> ≥ σc 4 · √ n . Moreover, if σ and c are constants, there exist at least c|A| nodes i such that <sup>ν</sup><sup>i</sup> = Ω √ n .

We observe that the graphs generated by the stochastic block model have this property. In addition to our mathematical proof in the appendix, it is empirically demonstrated in [\(Abbe et al.,](#page-8-24) [2020\)](#page-8-24) that most of the values in the eigenvectors is in Θ(1/ √ n). Additionally, [\(Balakrishnan et al.,](#page-8-25) [2011\)](#page-8-25) shows that this assumption can be satisfied when B is a node similarity matrix with certain additional properties.

Assumption (4) The final assumption is a common requirement for most differentially private algorithms. A large user base typically allows the added noise, introduced to protect sensitive information, to average out in the results.

## 3. Our Algorithm

We describe our algorithm in Algorithm [1.](#page-4-0) One can notice that we almost have x (t) = B ·x (t−1) and x (T) = B<sup>T</sup> ·x (0) by the calculation at Lines 6 - 7. The only five differences are as follows:

Difference 1: Addition of Laplace Noise We add Laplace noise in Line 6 to protect users' information. Later, we show in Section [4.2](#page-5-0) that this noise satisfies the conditions of the edge-local Laplacian query (Definition [2.3\)](#page-2-0). Furthermore, in Section [4.3,](#page-5-1) we demonstrate that when the minimum degree is sufficiently large, the magnitude of the Laplacian noise becomes negligible compared to other terms in the calculation in Line 6.

Difference 2: Minimum Degree Estimation When B is the normalized random walk Laplacian matrix, calculating B · x (t−1) does not require knowing the degrees of other nodes. This property simplifies computations within the edge LDP setting and is the main reason we select the normalized random walk Laplacian matrix over the normalized Laplacian matrix in our clustering algorithm.

On the other hands, to bound the sensitivity, which determines the scale of the Laplace noise in Line 6, we need a lower bound on the minimum degree of the graph G. This bound is computed in Line 2 of the algorithm, using degree estimates obtained in Line 1 of the mechanism. In Appendix [C,](#page-13-0) We will show that the estimate in Line 2 overestimates the minimum degree with probability not larger than <sup>1</sup> <sup>n</sup><sup>2</sup> when ζ = 1 n . If the estimate exceeds the actual minimum degree, we add edges in Line 3 to ensure that the modified graph meets the estimated minimum degree. In Appendix [C,](#page-13-0) we further show that the variable δ exceeds

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

274

Algorithm 1 Private Power Iteration Clustering

Input: Graph G = (V, E) where V = {v1, . . . , vn} and its adjacency matrix is A = (ai,j )1≤i,j≤n, privacy budget ϵ, number of iterations T = 2 log n log g , clipping factor c, and parameter ζ = 1

n Output: A cut of G denoted by S ⊂ V

<sup>1</sup> [User i] Compute the degree of v<sup>i</sup>

, denoted by d<sup>i</sup>

. Broadcast

˜d<sup>i</sup> ← d<sup>i</sup> + Lap(10/ϵ) to all users and the server.

<sup>2</sup> [Server] Calculate δ ← min<sup>i</sup>

˜d<sup>i</sup> − ϵ log <sup>n</sup> 2ζ

. Broadcast δ

to all users.

<sup>3</sup> [User i] If d<sup>i</sup> < δ, randomly select j such that ai,j = 0, then set ai,j = 1 and increment d<sup>i</sup> by one. Repeat this

process until d<sup>i</sup> ≥ δ. <sup>4</sup> [Server] Initiate the vector x

(0) = [x

(0) 1

, . . . , x (0) <sup>n</sup> ]

<sup>⊺</sup> where

x (0) i

is chosen from the Gaussian distribution with expected value 0 and standard deviation 1. Broadcast the vector x

(0)

to all users. <sup>5</sup> for t = 1, . . . , T do <sup>6</sup> [User i] Calculate w

(t) <sup>i</sup> = 2 x (t−1) <sup>i</sup> + 2 P j ai,j x (t−1) di −

1 n P j x (t−1) <sup>j</sup> + Lap

5·T <sup>9</sup>·<sup>ϵ</sup> max<sup>j</sup> |x (t−1) δ .

<sup>7</sup> [User i] Let U = c·

5·T <sup>9</sup>·<sup>ϵ</sup> max<sup>j</sup> |x (t−1) j δ

, also let x

(t) <sup>i</sup> = U

if w (t) <sup>i</sup> > U, x

(t)

<sup>i</sup> = −U if w

(t)

<sup>i</sup> < −U, and x

(t) <sup>i</sup> =

w (t) i

otherwise. Calculate and send x

(t) i

to the server.

<sup>8</sup> [Server] Aggregate the values x

(t) i

into a vector x

(t) =

[x (t) 1

, . . . , x (t) <sup>n</sup> ] ⊺

, and broadcast this information to all

users.

<sup>9</sup> [Server] Return S ← {v<sup>i</sup>

: x (T) <sup>i</sup> > 0}.

√ n log<sup>4</sup> n with probability at least 1 − n .

Difference 3: Replacing the Random Walk with a Lazy Random Walk Recall that all eigenvalues of the matrix B lie between 1 and −1. In certain networks, such as bipartite graphs, λn(B) can be close to −1. This causes the final term in Equation [\(2\)](#page-3-0) to oscillate, preventing the calculation of B<sup>T</sup> x from converging. To address this, we propose replacing B with W = 1 2 I + 1 <sup>2</sup>B. Note that for all i, vi(W) = vi(B) and λi(W) = <sup>1</sup> 2 λi(B) + <sup>1</sup> 2 . Consequently, for all i, 0 ≤ λi(W) ≤ 1. By the second assumption in Section [2.5,](#page-3-1) which is λi(W) ≤ λ2(W) g for all i ≥ 3, we can have the approximation [\(3\)](#page-3-2) even when some λi(B) are negative. This modification leads to the first two terms of the calculation in Line 6.

Difference 4: Elimination of the Leading Eigenvector Recall Equation [\(2\)](#page-3-0). Since λ2(W) < 1, the term α2λ2(W) <sup>T</sup> v2(W) diminishes compared to the leading term as T increases. On the other hand, the size of the

Laplace noise added depends on the largest element of x (t−1), which is determined by the leading term. Hence, for larger T, the noise magnitude dominates over the term α2λ2(W) <sup>T</sup> v2(W). This causes x (T) to deviate significantly from v2(W), reducing the accuracy of the results.

To address this, we introduce the matrix W˜ = ( ˜wi,j )1≤i,j≤n, where w˜i,j = wi,j − 1/n for all i, j. We show in Appendix [B](#page-13-1) that for all i ≥ 1, λi(W˜ ) = λi+1(W) and vn(W˜ ) = v1(W). Additionally, vn(W˜ ) = v1(W) = [ √ n , √ 1 n , . . . , √ 1 n ⊺ and λn(W˜ ) = 0.

With this update, the leading term <sup>α</sup><sup>1</sup> · [ √ 1 n , √ 1 n , . . . , √ 1 n ⊺ from [\(2\)](#page-3-0) is eliminated. The term α2λ2(W˜ ) <sup>T</sup> v2(W˜ ) now becomes the leading term, and we can ensure that the Laplace noise (the fourth term of Line 6 in Algorithm [1\)](#page-4-0) is substantially smaller than the new leading term. The subtraction of the third term in the calculation at Line 6 reflects the update from W to W˜ .

Difference 5: Clipping At Line 6 of the algorithm, we apply a standard clipping method commonly used in various LDP studies, such as [\(Imola et al.,](#page-8-6) [2022\)](#page-8-6) and [\(Betzer et al.,](#page-8-18) [2024\)](#page-8-18). We notice from the proof of Lemma [D.6](#page-16-0) that when the clipping factor c satisfies c ≥ log n · log g, it holds with high probability that −U ≤ w (t) <sup>i</sup> ≤ U for all i and t. Consequently, the clipping has no impact on our theoretical results. However, in our experiments, we observed that Algorithm [1](#page-4-0) achieves optimal performance when c is set to 5, which is smaller than log n · log g.

## 4. Properties of Our Algorithm

#### 4.1. Efficiency

Computation Time The primary computational bottleneck of Algorithm [1](#page-4-0) occurs in Line 6. In this step, the per-node computational complexity for each iteration is O(n). To achieve accurate results, the required number of iterations T is given by 2 log n log <sup>g</sup> = Θ(log n), leading to an overall computational complexity of O(n log n) per node. In contrast, the central server has minimal computational demands. Its responsibilities are limited to generating the initial vector, receiving calculation results, and distributing them to all users.

Communication Cost While each user uploads only one real number x (t) i to the server at each iteration, they must download the entire vector x (t) in Line 8 of the algorithm. This results in a total communication cost of O(n log n) for each user.

Memory Consumption During iteration t, the central server and all users only need to store two vectors: x (t−1)

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

and x (t) . As a result, the memory consumption for all parties is O(n). This is a significant improvement compared to the randomized response method. Even for sparse input graphs, the randomized response mechanism flips each relationship with a constant probability, leading to a graph with Ω(n 2 ) edges. Storing such a graph, with Ω(n 2 ) edges, requires a prohibitive amount of memory on the server, making it infeasible to design an LDP algorithm for large input graphs [\(Imola et al.,](#page-8-6) [2022\)](#page-8-6). In contrast, our approach requires only O(n) memory, enabling our algorithms to handle input graphs with millions of nodes efficiently.

#### 4.2. Privacy

The following theorem discuss our algorithm's privacy.

Theorem 4.1. *Algorithm [1](#page-4-0) is* ϵ*-edge LDP.*

*Proof.* We perform T + 1 edge-local Laplacian queries to all users: one at Line 1 and T queries at Line 6. At Line 1, the degree d<sup>i</sup> has a sensitivity of one. Since the Laplace noise is set to 10/ϵ, the privacy budget for the publication at Line 1 is ϵ/10.

When any ai,j is changed, the value of x (t) i calculated at Line 6 changes by at most <sup>1</sup> <sup>2</sup> max<sup>j</sup> |x (t−1) j d<sup>j</sup> . Therefore, the sensitivity of the publication at Line 6 is <sup>1</sup> <sup>2</sup> max<sup>j</sup> |x (t−1) d<sup>j</sup> ≤ 1 <sup>2</sup> max<sup>j</sup> |x (t−1) δ . The privacy budget for each publication at Line 6 is <sup>9</sup> <sup>10</sup> · ϵ T . Since there are T publications at Line 6, the total privacy budget of Algorithm [1](#page-4-0) is <sup>ϵ</sup> <sup>10</sup>+T · 9 10 · ϵ <sup>T</sup> = ϵ.

#### 4.3. Precision

In this section, we analyze the precision of Algorithm [1.](#page-4-0) In particular, we demonstrate that the algorithm's results closely resemble those of the spectral clustering algorithm. We provide an outline of our proof sketch here, with the full proof details available in Appendix [D.](#page-14-0)

In Algorithm [1,](#page-4-0) at iteration t we compute the vector x (t) = [x (t) 1 , . . . , x (t) <sup>n</sup> ] ⊺ . The output of the algorithm is Salg = {v<sup>i</sup> | x (T) <sup>i</sup> > 0}, where T = 2 log n log g .

Let v<sup>j</sup> (W˜ ) = [vj,1, . . . , vj,n] <sup>⊺</sup> be the j'th eigenvector of W˜ , and let c1, . . . , c<sup>n</sup> ∈ <sup>R</sup> be coefficients such that x (0) = P<sup>n</sup> <sup>j</sup>=1 <sup>c</sup>jv<sup>j</sup> (W˜ ). Additionally, for all <sup>t</sup>, suppose the noise added during iteration t of the algorithm is y (t) , and that e (t) 1 , . . . , e (t) <sup>n</sup> ∈ <sup>R</sup> are coefficients such that y (t) = P<sup>n</sup> <sup>j</sup>=1 e (t) <sup>j</sup> <sup>v</sup><sup>j</sup> (W˜ ). In Lemma [D.1,](#page-14-1) we show that x (T) <sup>i</sup> = P<sup>n</sup> <sup>j</sup>=1 c˜jvj,i, where c˜<sup>j</sup> is given by c˜<sup>j</sup> = cjλ<sup>j</sup> (W˜ ) <sup>T</sup> + P<sup>T</sup> <sup>t</sup>=1 e (t) <sup>j</sup> <sup>λ</sup><sup>j</sup> (W˜ ) T −t .

In Lemma [D.6,](#page-16-0) we show that the noise generated at Line 6 of the algorithm has a small scale. Specifically, we demonstrate that the noise scale, given by <sup>5</sup><sup>T</sup> <sup>9</sup><sup>ϵ</sup> max<sup>j</sup> |x (t−1) δ , is negligible compared to the magnitude of x (t) . Consequently, the noise term y (t) does not dominate the calculation. This emphasizes the significance of removing the leading eigenvector and establishing a lower bound for the minimum degree δ.

Due to the lemma, the term P<sup>T</sup> <sup>t</sup>=1 e (t) <sup>j</sup> <sup>λ</sup><sup>j</sup> (W˜ ) T −t is negligible compared to cjλ<sup>j</sup> (W˜ ) T , and we have c˜<sup>i</sup> ≈ ciλi(W˜ ) T . Consequently, x (T) <sup>i</sup> ≈ P<sup>n</sup> <sup>j</sup>=1 <sup>c</sup>jλ<sup>j</sup> (W˜ ) <sup>T</sup> vj,i. Using techniques from [\(Boutsidis et al.,](#page-8-20) [2015\)](#page-8-20), we show that x (T) <sup>i</sup> <sup>≈</sup> <sup>c</sup>1λ1(W˜ ) <sup>T</sup> v1,i when λ<sup>j</sup> (W˜ ) ≤ λ1(W˜ ) g for all j ≥ 2. Specifically, in Theorem [D.7,](#page-18-0) we demonstrate that c1λ1(W˜ ) <sup>T</sup> v1,i  <sup>&</sup>gt; P<sup>T</sup> <sup>t</sup>=1 e T <sup>1</sup> <sup>λ</sup>1(W˜ ) <sup>T</sup> <sup>−</sup><sup>t</sup>v1,i + P<sup>n</sup> <sup>j</sup>=2 c˜jvj,i  with probability at least 0.95 − o(1). The term c1λ1(W˜ ) <sup>T</sup> v1,i dominates and determines the sign of x (T) i .

Since λ1(W˜ ) T is positive, we conclude that when c1v1,i > 0, x (t) <sup>i</sup> > 0 with high probability. Recall that the outcome of the spectral clustering algorithm is Sorig = {v<sup>i</sup> : v1,i > 0}. Thus, when c<sup>1</sup> > 0, the result Salg closely resembles Sorig with high probability. Conversely, when c<sup>1</sup> < 0, the result Salg is similar to V \ Sorig with high probability. Therefore, our algorithm is likely to produce a small dvol(Salg, Sorig). In conclusion, the results are comparable to those obtained from the spectral clustering algorithm.

## 5. Experimental Results

Evaluation Method For all experiments, we use the normalized discrepancy dnorm, as defined in [\(1\)](#page-1-0), to assess precision. Remember that when the normalized discrepancy is small, the outcome closely resembles that of the original spectral clustering algorithm, indicating a high-quality clustering result. The reported values represent the average of 10 experiments, which we consider sufficient, as the variance in precision across each set of experiments is typically small.

Input Graphs We conduct most of our experiments on graphs generated using the stochastic block model (SBM) [\(Holland et al.,](#page-8-8) [1983\)](#page-8-8). This model is chosen because it ensures that the generated graphs are well-clustered and consist of exactly two clusters. Furthermore, SBM has been widely employed in prior studies to analyze spectral clustering under local differential privacy [\(Hehir et al.,](#page-8-7) [2022\)](#page-8-7). In this model, the set of n nodes is divided into two clusters of sizes n<sup>1</sup> and n2, where n<sup>1</sup> + n<sup>2</sup> = n. Two nodes within the same cluster are connected with probability p, while nodes from different clusters are connected with probability q. While in most cases p ≫ q, this paper also considers the scenario where q > p.

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

Parameters Unless otherwise specified, we set n = 10, 000, n<sup>1</sup> = n<sup>2</sup> = 5, 000, p = 0.3, q = 0.2, the clipping factor c = 10, and the privacy budget ϵ = 1.

The value of n is chosen to be 10, 000 due to the memory requirements of the benchmark algorithm, randomized response, which requires Ω(n ) bits to store the entire graph for spectral clustering calculations. We believe that graphs of this size are sufficient to effectively demonstrate the empirical properties of our algorithm. Given the constraints of our local computational environment, handling larger graphs is not feasible. We select p = 0.3 and q = 0.2 because these values are close enough to highlight the precision of our algorithm in distinguishing clusters. We set the clipping factor c = 10, as it is the integer closest to log n log g for well-clustered graphs generated using the stochastic block model. Recall that, when c = log n log g, the clipping is applied only with small probability. The privacy budget is set to ϵ = 1 as it is a standard value commonly used in experiments of other local differential privacy algorithms [\(Hillebrand et al.,](#page-8-21) [2023\)](#page-8-21).

Benchmark To the best of our knowledge, only one graph clustering algorithm under local differential privacy has been explored in the literature. This algorithm employs the spectral clustering method on graph processed using randomized response [\(Hehir et al.,](#page-8-7) [2022\)](#page-8-7). Therefore, we select this algorithm as the benchmark for our study.

(a) Comparison across different privacy budget

(b) Comparison across different graph sizes

(c) Comparison across different graph density when ϵ = 1

(d) Comparison across different graph density when ϵ = 1.5

Figure 1. Comparison of the normalized discrepancy between our algorithm and the randomized response-based algorithm on the graphs generated from the stochastic block model. The results shown in Figures [1\(c\)](#page-6-0) and [1\(d\)](#page-6-1) represent the differences in the normalized discrepancies between the two algorithms.

Comparison across Different Privacy Budget As illustrated in Figure [1\(a\),](#page-6-2) our algorithm consistently outperforms the benchmark algorithm across all privacy budget values (ϵ). The improvement is especially notable in the range 0.8 ≤ ϵ ≤ 2, where the benchmark algorithm yields nearly random results, with a normalized discrepancy close to 1, while our algorithm produces results almost identical to the non-private spectral clustering.

Comparison across Different Graph Size Figure [1\(b\)](#page-6-3) presents a comparison with the benchmark algorithm across varying numbers of nodes (n). From the figure, we observe that while our algorithm performs poorly for small n, it achieves results identical to non-private spectral clustering when n becomes sufficiently large. This aligns with our theoretical findings, which indicate that the noise introduced by our algorithm becomes negligible as the input graph size increases.

The plot also reveals that the randomized response-based algorithm performs well only when the input graph size is small. This observation aligns with the theoretical findings of previous work [\(Mukherjee & Suppakitpaisarn,](#page-9-7) [2023\)](#page-9-7), which state that the required privacy budget must exceed Θ(log n). Consequently, larger values of n demand a higher privacy budget in the prior approach. In summary, our algorithm demonstrates greater precision for larger n, whereas the previous method performs better on very small graphs.

It is worth noting that, for the plot in Figure [1\(b\)](#page-6-3) alone, we conducted the experiment on Google Colaboratory. This was necessary because our local computing environment lacked the storage capacity for the randomized response results for graphs of that size. However, we have verified that the precision results remain consistent across different computational environments.

![](_page_6_Figure_8.jpeg)

![](_page_6_Figure_9.jpeg)

Comparison across Different Edge Density In Figures [1\(c\)](#page-6-0) and [1\(d\),](#page-6-1) we explore the impact of graph density by varying the probabilities p and q. The experiments are conducted for all pairs (p, q) ∈ {0.05, 0.1, . . . , 0.95} 2 and for ϵ ∈ {1, 1.5}. Due to the large number of experiments, the graph size is reduced to 1000 for this analysis. The results show that when p > 0.35, our algorithm consistently outperforms the randomized response-based method, achieving a smaller normalized discrepancy in these cases.

![](_page_6_Figure_13.jpeg)

![](_page_6_Figure_14.jpeg)

When p ≤ 0.35, there are instances where our algorithm performs worse than the benchmark algorithm. This occurs because the estimated minimum degree, δ, is relatively small in these cases, resulting in a larger amount of noise added in Algorithm 1. While we have theoretically shown that our algorithm can produce results comparable to original spectral clustering when δ ≥ √ n log<sup>4</sup> n (where n is the number of nodes), this analysis is valid only for large n and

394

396

does not extend to cases where n = 1000. On the other hand, as shown in [\(Mohamed et al.,](#page-8-10) [2022\)](#page-8-10), the randomized response-based algorithm performs well when q ≤ p and p is small. Consequently, in these scenarios, the randomized response method outperforms our algorithm.

We observe that when q > p, the results of both algorithms deviate from those of the original spectral clustering algorithm. This outcome arises because the input graphs are not well-clustered, leading to poor performance from both the original spectral clustering method and the two algorithms in these cases.

Computation Time Although our algorithm is designed to be executed in a distributed manner in practice, we were unable to afford the necessary computation units for handling 10,000 nodes in this experiment. As a result, all computations were performed on our server, making the computation environment different from practical scenarios. Consequently, a direct comparison of the computation times between our algorithm and the benchmark algorithm is not feasible. However, even with all computations performed on the server, the computation time for graphs with 20,000 nodes is less than 10 seconds for both algorithms, and for graphs with 1,000,000 nodes, our algorithm completes in under 1 minute. Therefore, we consider computation time to be a manageable factor for both algorithms.

Results on Reddit Graph We also conduct an experiment on the real graph called Reddit graph [\(Hamilton et al.,](#page-8-22) [2017\)](#page-8-22). We chose this graph because it is one of the largest publicly available social networks and features a clear cluster structure. To ensure that the noise added in our algorithm is not too large, we calculate a 100-core and 500-core decomposition of the graph before giving it as an input of both algorithms. The 100-core decomposition result contains 154,525 nodes and 108,024,958 edges, while the 500-core decomposition result contains 44,586 nodes, 54,984,204 edges.

We were unable to run the randomized response algorithm on this large network, even with the A100 GPU (40GB of GPU RAM) and 83.5GB of system RAM. As a result, we could not directly compare our algorithm with the previous one. Since the Reddit graph contains more than two clusters, we observed that λ3(B) + 1 is very close to λ2(B) + 1, and the value of g (defined in Section [2.5\)](#page-3-1) must be set as low as 0.005. Consequently, the number of iterations required by the algorithm, calculated as 2 log n/ log g, increases significantly to approximately 14,000. Given that the noise size is dependent on the number of iterations, this large iteration count renders the noise size unmanageable. To address this, we limited the number of iterations to 50 for this experiment.

![](_page_7_Figure_2.jpeg)

Figure 2. The normalized discrepancies of our algorithm for the graph extracted from the Reddit graph

Our results for these graphs are presented in Figure [2.](#page-7-0) For graphs generated using the SBM, we observe that when an algorithm fails to classify the graph in a particular setting, the normalized discrepancy exceeds 0.99. In contrast, our normalized discrepancy remains below 0.99 when the privacy budget is at least 4 for the 100-core decomposition and at least 1 for the 500-core decomposition. This demonstrates that our algorithm can produce meaningful clustering results under these conditions.

While the normalized discrepancy rapidly converges to 0 in graphs generated by the model, it does not converge to 0 in Figure [2.](#page-7-0) We attribute this to the Reddit graph containing more than two clusters, which results in a significant number of nodes v<sup>i</sup> with small |ν<sup>i</sup> | (as discussed in Assumption 3 in Section [2.5\)](#page-3-1). Consequently, our algorithm is unable to classify these nodes correctly.

Further Experiments In Appendix [E,](#page-20-0) we present experiments to validate the positive impact of the differences discussed in Section [3.](#page-3-3)

## 6. Conclusion and Future Work

In this paper, we propose a locally differentially private algorithm for graph clustering that is theoretically proven to work on general graphs. Unlike most prior works, which focus on non-interactive algorithms based on randomized response, we introduce an interactive algorithm leveraging power iterative clustering. Our approach demonstrates both theoretical and experimental improvements over previous methods. By this work, we believe that interactive algorithms have the potential to become a key tool for addressing graph problems under local differential privacy.

Although our algorithm is applicable to sparse graphs, our theoretical guarantees currently hold only for dense graphs. Extending the theory to sparse graphs requires an additional condition: for any eigenvector v<sup>i</sup> = [vi,1, . . . , vi,n] ⊺ , the ratio maxj,j′ vi,j <sup>v</sup>i,j′ must be small. This property, known as delocalization, has been studied in several works, such as [\(Rudelson & Vershynin,](#page-9-11) [2016\)](#page-9-11). We plan to investigate the potential of incorporating this property into our analysis.

- 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. References Abbe, E., Fan, J., Wang, K., and Zhong, Y. Entrywise eigenvector analysis of random matrices with low expected rank. *Annals of Statistics*, 48(3):1452, 2020. Alon, N. Eigenvalues and expanders. *Combinatorica*, 6(2): 83–96, 1986. Apple's Differential Privacy Team. Learning with privacy at scale. *Apple Machine Learning Research*, 2017. Balakrishnan, S., Xu, M., Krishnamurthy, A., and Singh,
  - A. Noise thresholds for spectral clustering. *Advances in Neural Information Processing Systems*, 24, 2011. Betzer, L., Suppakitpaisarn, V., and Hillebrand, Q. Publishing number of walks and Katz centrality under local differential privacy. In *UAI 2024*, 2024. Boutsidis, C., Kambadur, P., and Gittens, A. Spectral clustering via the power method - provably. In *ICML 2015*, pp. 40–48, 2015. Chen, H., Cohen-Addad, V., d'Orsi, T., Epasto, A., Imola, J., Steurer, D., and Tiegel, S. Private estimation algorithms for stochastic block models and mixture models. *Advances in Neural Information Processing Systems*, 36: 68134–68183, 2023. Dwork, C. Differential privacy: A survey of results. In *TAMC 2008*, pp. 1–19, 2008. Epasto, A., Liu, Q. C., Mukherjee, T., and Zhou, F. The power of graph sparsification in the continual release model. *arXiv preprint arXiv:2407.17619*, 2024. Erlingsson, U., Pihur, V., and Korolova, A. RAPPOR: Randomized aggregatable privacy-preserving ordinal response. In *SIGSAC 2014*, pp. 1054–1067, 2014. Fu, N., Ni, W., Zhang, S., Hou, L., and Zhang, D. GC-NLDP: A graph clustering algorithm with local differential privacy. *Computers & Security*, 124:102967, 2023. Hamilton, W., Ying, Z., and Leskovec, J. Inductive representation learning on large graphs. *Advances in neural information processing systems*, 30, 2017. He, W., Fichtenberger, H., and Peng, P. A differentially private clustering algorithm for well-clustered graphs. In *ICLR 2024*, 2024. Hehir, J., Slavkovic, A., and Niu, X. Consistent spectral clustering of network block models under local differential privacy. *Journal of Privacy and Confidentiality*, 12 (2), 2022. Henzinger, M., Sricharan, A., and Zhu, L. Tighter bounds for local differentially private core decomposition and densest subgraph. *arXiv preprint arXiv:2402.18020*, 2024. Hillebrand, Q., Suppakitpaisarn, V., and Shibuya, T. Unbiased locally private estimator for polynomials of laplacian variables. In *SIGKDD 2023*, pp. 741–751, 2023. Hillebrand, Q., Suppakitpaisarn, V., and Shibuya, T. Cycle counting under local differential privacy for degeneracybounded graphs. *arXiv preprint arXiv:2409.16688*, 2024. Holland, P. W., Laskey, K. B., and Leinhardt, S. Stochastic blockmodels: First steps. *Social networks*, 5(2):109–137, 1983. Imola, J., Murakami, T., and Chaudhuri, K. Locally differentially private analysis of graph statistics. In *USENIX Security 2021*, pp. 983–1000, 2021. Imola, J., Murakami, T., and Chaudhuri, K. Communicationefficient triangle counting under local differential privacy. In *USENIX Security 2022*, pp. 537–554, 2022. Imola, J., Epasto, A., Mahdian, M., Cohen-Addad, V., and Mirrokni, V. Differentially private hierarchical clustering with provable approximation guarantees. In *ICML 2023*, pp. 14353–14375, 2023. Ji, T., Luo, C., Guo, Y., Wang, Q., Yu, L., and Li, P. Community detection in online social networks: A differentially private and parsimonious approach. *IEEE Transactions on Computational Social Systems*, 7(1):151–163, 2020. Kasiviswanathan, S. P., Lee, H. K., Nissim, K., Raskhodnikova, S., and Smith, A. What can we learn privately? *SIAM Journal on Computing*, 40(3):793–826, 2011. Li, J. and Tkocz, T. Tail bounds for sums of independent two-sided exponential random variables. In *High Dimensional Probability IX: The Ethereal Volume*, pp. 143–154. Springer, 2023. Lin, F. and Cohen, W. W. Power iteration clustering. In *ICML 2010*, pp. 655–662, 2010. Mangat, N. S. An improved randomized response strategy. *Journal of the Royal Statistical Society: Series B (Methodological)*, 56(1):93–95, 1994. Mohamed, M. S., Nguyen, D., Vullikanti, A., and Tandon, R. Differentially private community detection for stochastic block models. In *ICML 2022*, pp. 15858–15894, 2022.

### Impact Statement

- 495 496 497 498 499 500 504 506 508 509 511 514 515 516 518 524 526 528 531 534 536 538 540 541 542 543 544 545 546 547 548 549 Mohar, B. Isoperimetric numbers of graphs. *Journal of Combinatorial Theory, Series B*, 47(3):274–291, 1989. Mukherjee, S. and Suppakitpaisarn, V. Robustness for spectral clustering of general graphs under local differential privacy. *arXiv preprint arXiv:2309.06867*, 2023. Ng, A., Jordan, M., and Weiss, Y. On spectral clustering: Analysis and an algorithm. *NIPS 2001*, 14, 2001. Peng, R., Sun, H., and Zanetti, L. Partitioning well-clustered graphs: Spectral clustering works! In *COLT 2015*, pp. 1423–1455, 2015. Qin, Z., Yu, T., Yang, Y., Khalil, I., Xiao, X., and Ren,
  - K. Generating synthetic decentralized social graphs with local differential privacy. In *CCS 2017*, pp. 425–438, 2017. Rudelson, M. and Vershynin, R. No-gaps delocalization for general random matrices. *Geometric and Functional Analysis*, 26(6):1716–1776, 2016. Shi, J. and Malik, J. Normalized cuts and image segmentation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(8):888–905, 2000. Von Luxburg, U. A tutorial on spectral clustering. *Statistics and Computing*, 17:395–416, 2007. Wang, Y., Wu, X., and Hu, D. Using randomized response for differential privacy preserving data collection. In *EDBT/ICDT Workshops*, 2016. Warner, S. L. Randomized response: A survey technique for eliminating evasive answer bias. *Journal of the American Statistical Association*, 60(309):63–69, 1965. Ye, Q., Hu, H., Au, M. H., Meng, X., and Xiao, X. Towards locally differentially private generic graph metric estimation. In *ICDE 2020*, pp. 1922–1925, 2020. Zhang, H., Latif, S., Bassily, R., and Rountev, A. Differentially-private control-flow node coverage for software usage analysis. In *USENIX Security 2020*, 2020. Zhu, T., Li, G., Zhou, W., and Philip, S. Y. Differentially private data publishing and analysis: A survey. *IEEE Transactions on Knowledge and Data Engineering*, 29 (8):1619–1638, 2017.

554

556

558

560

564

566

568

571

574

576

578

594

596

598

#### A. Eigenvector Components

In this section, we analyze the Laplacian matrix of the graph G, defined as L = I − B. For each i, let λi(L) = 1 − λi(B). It follows that λi(L) is an eigenvalue of L, and the eigenvalues are ordered as λ1(L) ≤ · · · ≤ λn(L). Moreover, the eigenvector vi(B) associated with λi(B) is also an eigenvector of L corresponding to λi(L). For simplicity, throughout this section, we denote λi(L) by λ<sup>i</sup> and vi(B) by v<sup>i</sup> = [vi,1, . . . , vi,n] ⊺ .

Proposition A.1. *Assume that*

*(i) Let* V (G) = A ⊔ B *be a bipartition of* G *with* v2,j ≥ 0 *for* v<sup>j</sup> ∈ A*,* v2,j ≤ 0 *for* v<sup>j</sup> ∈ B*. Then, the cut* (A, B) *has conductance* ϕ *satisfying* ϕ/λ<sup>3</sup> ≤ 0.12*.*

*(ii) Let* ϵ *and* c *be a constant. For a subset* S ⊆ V *and vertex* v<sup>j</sup> ∈ S*, let us call* v<sup>j</sup> *to be* (ϵ, S)*-average if* d<sup>j</sup> ≥ ϵd(S)*, where* d(S) = Vol(S)/|S| *is the average degree of the vertices in* S*. Let* A<sup>ϵ</sup> *and* B<sup>ϵ</sup> *denote the set of* (ϵ, A)*-average nodes of* A *and* (ϵ, B)*-average nodes of* B*, respectively. Assume that* |Aϵ| ≥ c|A| *and* |Bϵ| ≥ c|B|*.*

*Then,*

$$|v_{2,j}| \geq \begin{cases} \frac{\epsilon^{1/2}c}{4} \cdot \sqrt{\frac{d_j}{nd(A)}} - 2\sqrt{\frac{\phi}{\lambda_3}}, & v \in A \\ \frac{\epsilon^{1/2}c}{4} \cdot \sqrt{\frac{d_j}{nd(B)}} - 2\sqrt{\frac{\phi}{\lambda_3}}, & v \in B \end{cases} \quad (4)$$

*Consequently, for* v<sup>j</sup> ∈ A<sup>ϵ</sup> ∪ Bϵ*, which is at least* c *fraction of the vertices of* G*, we have*

$$|v_{2,j}| \geq \frac{\epsilon c}{4} \cdot \frac{1}{\sqrt{n}} - 2\sqrt{\frac{\phi}{\lambda_3}}. \quad (5)$$

*Proof.* Let us define the normalized indicator variables

$$g_A(j) = \begin{cases} \frac{d_j^{1/2}}{\text{Vol}(A)^{1/2}}, & v_j \in A \\ 0, & v_j \in B \end{cases} \quad \text{and} \quad g_B(j) = \begin{cases} 0, & v_j \in A \\ \frac{d_j^{1/2}}{\text{Vol}(B)^{1/2}}, & v_j \in B \end{cases}$$

Let the vector g<sup>A</sup> = [gA(1), . . . , gA(n)]<sup>⊺</sup> , g<sup>B</sup> = [gB(1), . . . , gB(n)]<sup>⊺</sup> , and, for any vector v, the Rayleign quotient of v = [x1, . . . , xn] ⊺ , denoted by R(v), is <sup>v</sup> <sup>⊺</sup>Lv v⊺v . We show the following regarding the Rayleigh quotients RL(gA) and RL(gB).

Claim A.2. ϕ ≥ max{RL(gA), RL(gB)}.

*Proof of Claim [A.2.](#page-10-2)* Observe that the Rayleigh quotient of L satisfies,

$$\mathcal{R}_L(\mathbf{v}) = \frac{\mathbf{v}^\top L \mathbf{v}}{\mathbf{v}^\top \mathbf{v}} = 1 - \frac{\sum_{i=1}^n \sum_{j=1}^n \frac{a_{ij}}{d_i} x_i x_j}{\sum_{i=1}^n x_i^2} = 1 - \frac{\sum_{\{i,j\} \in E} \left( \frac{1}{d_i} + \frac{1}{d_j} \right) x_i x_j}{\sum_{i=1}^n x_i^2}. \quad (6)$$

Since ∥gA∥ <sup>2</sup> = 1, we have

$$\begin{aligned}\mathcal{R}_L(g_A) &= 1 - \sum_{\{i,j\} \in E} \left( \frac{1}{d_i} + \frac{1}{d_j} \right) g_A(i)g_A(j) = 1 - \sum_{\{i,j\} \in E(A)} \left( \frac{1}{d_i} + \frac{1}{d_j} \right) \cdot \frac{\sqrt{d_i d_j}}{\text{Vol}(A)} \\ &\leq 1 - \sum_{\{i,j\} \in E(A)} \frac{2}{\text{Vol}(A)} = \frac{\text{Vol}(A) - 2e(A)}{\text{Vol}(A)} \\ &= \frac{e(A, B)}{\text{Vol}(A)} \leq \phi.\end{aligned}$$

Similarly, we have RL(gB) ≤ ϕ, completing the proof of Claim [A.2.](#page-10-2) ■

For the rest of the proof, let us denote t := ϕ/λ3. Recall that v<sup>1</sup> = [1/ √ n, . . . , 1/ √ n] ⊺ . We will make use of the following lemmas from the structure theorem (Theorem 3.1) of [\(Peng et al.,](#page-9-10) [2015\)](#page-9-10), but with a different notation and error estimates.

Lemma A.3. *Let* gˆA*,* gˆ<sup>B</sup> *be the projections of* gA*,* g<sup>B</sup> *onto the space spanned by the first two eigenvectors* {v1, v2} *of* L*. Then,*

$$\max\{\|\hat{g}_A - g_A\|^2, \|\hat{g}_B - g_B\|^2\} \leq t. \quad (7)$$

*Proof of Lemma [A.3.](#page-10-3)* Let v3, . . . , v<sup>n</sup> be normalized eigenvectors of λ3, . . . , λ<sup>n</sup> of L. Say g<sup>A</sup> = α1v<sup>1</sup> + · · · + αnv<sup>n</sup> and g<sup>B</sup> = β1v<sup>1</sup> + · · · + βnv<sup>n</sup> are representations of g<sup>A</sup> and g<sup>B</sup> in the L-eigenbasis. Clearly gˆ<sup>A</sup> = α1v<sup>1</sup> + α2v<sup>2</sup> and gˆ<sup>B</sup> = β1v<sup>1</sup> + β2v2. Then, note that as v ⊺ <sup>i</sup> v<sup>j</sup> = 0 for every i ̸= j,

$$\mathcal{R}_L(g_A) = \sum_{i=1}^n \alpha_i \mathbf{v}_i^\top \cdot \mathbf{L} \cdot \sum_{i=1}^n \alpha_i \mathbf{v}_i = \sum_{i=1}^n \alpha_i^2 \mathbf{v}_i^\top \mathbf{L} \mathbf{v}_i = \sum_{i=1}^n \alpha_i^2 \lambda_i.$$

But λ<sup>1</sup> = 0, leading us to RL(gA) ≥ α 2 <sup>2</sup>λ<sup>2</sup> + (α 2 <sup>3</sup> + · · · + α 2 n )λ<sup>3</sup> = α 2 <sup>2</sup>λ<sup>2</sup> + ∥gˆ<sup>A</sup> − gA∥ <sup>2</sup>λ<sup>3</sup> ≥ ∥gˆ<sup>A</sup> − gA∥ <sup>2</sup>λ3. Thus, ∥gˆ<sup>A</sup> − gA∥ <sup>2</sup> ≤ RL(gA)/λ<sup>3</sup> ≤ ϕ/λ<sup>3</sup> by Claim [A.2.](#page-10-2) The proof for ∥gˆ<sup>B</sup> − gB∥ 2 is exactly analogous. ■

One of the main ideas used in [\(Peng et al.,](#page-9-10) [2015\)](#page-9-10) is that if gˆ<sup>A</sup> and gˆ<sup>B</sup> are independent, then Span({v1, v2}) = Span({gˆA, gˆB}), implying that v<sup>1</sup> and v<sup>2</sup> can be written as linear combinations of the projected indicator vectors gˆ<sup>A</sup> and gˆB, say v<sup>2</sup> = η1gˆ<sup>A</sup> + η2gˆB, implying that ∥v<sup>2</sup> − η1g<sup>A</sup> − η2gB∥ is small.

Let us now continue with the argument.

Claim A.4. gˆ<sup>A</sup> and gˆ<sup>B</sup> are linearly independent.

*Proof of Claim [A.4](#page-11-0)*. By Lemma [A.3,](#page-10-3) we have ∥gˆA∥ <sup>2</sup> ≥ 1 − t and ∥gˆB∥ <sup>2</sup> ≥ 1 − t. On the other hand,

$$\begin{aligned} |\langle \hat{g}_A, \hat{g}_B \rangle| &= |\langle \hat{g}_A - g_A + g_A, \hat{g}_B - g_B + g_B \rangle| \\ &\leq |\langle \hat{g}_A - g_A, \hat{g}_B - g_B \rangle| + |\langle g_A, \hat{g}_B - g_B \rangle| + |\langle \hat{g}_A - g_A, g_B \rangle| \\ &\leq \|\hat{g}_A - g_A\| \|\hat{g}_B - g_B\| + \|\hat{g}_A - g_A\| + \|\hat{g}_B - g_B\| \\ &\leq t + 2\sqrt{t}. \end{aligned} \tag{8}$$

Since t ≤ 0.12 < 2 (2 − √ 3), we have <sup>t</sup> + 2√ t < 1 − t, implying |⟨gˆA, gˆB⟩| < ∥gˆA∥∥gˆB∥. As this implies a strict inequality in the Cauchy-Schwarz inequality, we have gˆ<sup>A</sup> ∦ gˆB. ■

As discussed earlier, Claim [A.4](#page-11-0) implies that there exist η1, η<sup>2</sup> ∈ <sup>R</sup> such that v<sup>2</sup> = η1gˆ<sup>A</sup> +η2gˆB. Suppose v ′ <sup>2</sup> = η1g<sup>A</sup> +η2gB, and η = ∥v ′ <sup>2</sup>∥ = p η 2 <sup>1</sup> + η 2 . Note that, using [\(8\)](#page-11-1),

$$\begin{aligned} 1 = \|\mathbf{v}_2\|^2 &\geq \eta_1^2 \|\hat{g}_A\|^2 + \eta_2^2 \|\hat{g}_B\|^2 - 2|\eta_1 \eta_2 \langle \hat{g}_A, \hat{g}_B \rangle| \\ &\geq \eta_1^2 (1-t) + \eta_2^2 (1-t) - (\eta_1^2 + \eta_2^2)(t + 2\sqrt{t}) \\ &= \eta^2 (1 - 2t - 2\sqrt{t}). \end{aligned}$$

Moreover, since t ≤ 0.12, we have

$$\eta^2 \leq \frac{1}{1 - 2t - 2\sqrt{t}} < 16. \quad (9)$$

Moreover, by the triangle inequality and Cauchy-Schwarz inequality,

$$\begin{aligned} \|\mathbf{v}_2 - \mathbf{v}_2'\|^2 &= \|\eta_1(\hat{g}_A - g_A) + \eta_2(\hat{g}_B - g_B)\|^2 \\ &\leq \left( |\eta_1|\sqrt{t} + |\eta_2|\sqrt{t} \right)^2 \\ &\leq 2t\eta^2 \end{aligned} \tag{10}$$

Therefore, we have that

$$2\eta\langle \mathbf{v}_2, \frac{1}{\eta}\mathbf{v}'_2 \rangle = 2\langle \mathbf{v}_2, \mathbf{v}'_2 \rangle = \|\mathbf{v}_2\|^2 + \|\mathbf{v}'_2\|^2 - \|\mathbf{v}_2 - \mathbf{v}'_2\|^2 \geq 1 + \eta^2 - 2t\eta^2,$$

leading us to

$$\langle \mathbf{v}_2, \frac{1}{\eta} \mathbf{v}'_2 \rangle \geq \frac{1+\eta^2}{2\eta} - \eta t \geq 1 - \eta t. \quad (11)$$

Basically, this means that v<sup>2</sup> is closely aligned with the normalized vector <sup>1</sup> η v ′ 2 . We now show a lemma that relates the components of two such vectors.

*666 667*

*684*

*687 688*

*690 691*

*694*

*696*

*700*

*704*

*706*

Lemma A.5. *Let* v = [u1, . . . , un] <sup>⊺</sup> *be a unit eigenvector of* L *and* v ′ = [u ′ 1 , . . . , u′ n <sup>⊺</sup> *be any unit vector with* ⟨v, v ′ ⟩ ≥ 1 − ϵ 2 *for some* ϵ > 0*. Then, for each* 1 ≤ j ≤ n*, we have*

$$|u'_j| \leq |u_j| + \epsilon.$$

*Proof of Lemma [A.5.](#page-11-2)* Let {v, z1, . . . zn−1} be a orthonormal basis of eigenvectors of L, and, for all i, let z<sup>i</sup> = [zi,1, . . . , zi,n] ⊺ . Since v ′ = ⟨v, v ′ ⟩ · v + P<sup>n</sup>−<sup>1</sup> <sup>i</sup>=1 ⟨v ′ , zi⟩ · z<sup>i</sup> , this implies that for any 1 ≤ j ≤ n,

$$\begin{aligned} |u'_j| &\leq |\langle \mathbf{v}, \mathbf{v}' \rangle| |u_j| + \sum_{i=1}^{n-1} |\langle \mathbf{v}', \mathbf{z}_i \rangle| |z_{i,j}| \\ &\leq |u_j| + \left( \sum_{i=1}^{n-1} \langle \mathbf{v}', \mathbf{z}_i \rangle^2 \right)^{1/2} \left( \sum_{i=1}^{n-1} z_{i,j}^2 \right)^{1/2} \\ &\leq |u_j| + \epsilon, \end{aligned}$$

where the last step follows from the fact that P<sup>n</sup>−<sup>1</sup> <sup>i</sup>=1 ⟨v ′ , zi⟩ <sup>2</sup> + ⟨v ′ , v⟩ <sup>2</sup> = ∥v ′∥ <sup>2</sup> = 1, and P<sup>n</sup>−<sup>1</sup> <sup>i</sup>=1 z 2 i,j + u 2 <sup>j</sup> = 1. ■

Hence, by virtue of Lemma [A.5,](#page-11-2) [\(9\)](#page-11-3) and [\(11\)](#page-11-4), we obtain

$$|v_{2,j}| \geq \frac{1}{\eta} |v'_{2,j}| - \sqrt{\eta t} = \frac{1}{\eta} |\eta_1 g_A(j) + \eta_2 g_B(j)| - \sqrt{\eta t} = \begin{cases} \frac{|\eta_1|}{\eta} \cdot \frac{d_j^{1/2}}{\text{Vol}(A)^{1/2}} - \sqrt{\eta t}, & v_j \in A \\ \frac{|\eta_2|}{\eta} \cdot \frac{d_j^{1/2}}{\text{Vol}(B)^{1/2}} - \sqrt{\eta t}, & v_j \in B \end{cases} \quad (12)$$

Finally, we need to show that min{|η1|, |η2|} ≥ ϵ 1/2 c. For this part of the proof, we shall use the assumption (ii) of our proposition.

Claim A.6. |η1| ≥ c · ϵ|A| n 1/2 and |η2| ≥ c · ϵ|B| n 1/2 .

*Proof of Claim [A.6.](#page-12-0)* Recall from the proof of Lemma [A.3](#page-10-3) that gˆ<sup>A</sup> = α1v<sup>1</sup> + α2v<sup>2</sup> and gˆ<sup>B</sup> = β1v<sup>1</sup> + β2v2. These equations, along with v<sup>2</sup> = η1gˆ<sup>A</sup> + η2gˆB, allow us to solve exactly for η<sup>1</sup> and η<sup>2</sup> as,

$$\eta_1 = \frac{\beta_1}{\alpha_2\beta_1 - \alpha_1\beta_2} \text{ and } \eta_2 = \frac{-\alpha_1}{\alpha_2\beta_1 - \alpha_1\beta_2}.$$

First, we note that |α2β<sup>1</sup> − α1β2| ≤ (α 2 <sup>1</sup> + α 2 2 ) 1/2 (β 2 <sup>1</sup> + β 2 2 ) <sup>1</sup>/<sup>2</sup> ≤ ∥gA∥∥gB∥ = 1, so it suffices to lower bound |α1| and |β1|. We have that:

$$\begin{aligned} |\alpha_1| = |\langle g_A, \mathbf{v}_1 \rangle| &= \frac{1}{\sqrt{n}} \sum_{v_j \in A} \frac{d_j^{1/2}}{\text{Vol}(A)^{1/2}} \geq \frac{1}{\sqrt{n}} \sum_{v_j \in A_\epsilon} \frac{d_j^{1/2}}{(|A|d(A))^{1/2}} \\ &\geq \frac{1}{\sqrt{n}} \cdot |A_\epsilon| \cdot \left( \frac{\epsilon}{|A|} \right)^{1/2} \\ &\geq c \cdot \left( \frac{\epsilon |A|}{n} \right)^{1/2}. \end{aligned}$$

By a similar argument, we have |β1| ≥ c · ϵ|B| n 1/2 , finishing the proof of Claim [A.6.](#page-12-0) ■

Claim [A.6,](#page-12-0) [\(12\)](#page-12-1) and η ≤ 4 leads us to, for v<sup>j</sup> ∈ A,

$$|v_{2,j}| \geq \frac{c}{4} \cdot \frac{\epsilon^{1/2} |A|^{1/2}}{n^{1/2}} \cdot \frac{d_j^{1/2}}{\text{Vol}(A)^{1/2}} - 2\sqrt{t} = \frac{c\epsilon^{1/2}}{4} \cdot \sqrt{\frac{d_j}{nd(A)}} - 2\sqrt{t},$$

which proves the inequality [\(4\)](#page-10-4) for v<sup>j</sup> ∈ A. The argument for v<sup>j</sup> ∈ B is analogous.

Finally, the inequality [\(5\)](#page-10-5) directly follows [\(4\)](#page-10-4) via the definitions of A<sup>ϵ</sup> and Bϵ.

718

724

726

728

731

734

736

738

751

754

756

758

760

764

766

### B. Elimination of the Leading Eigenvector

The following proposition shows that the third term in the calculation at Line 6 of Algorithm [1](#page-4-0) eliminates the leading eigenvector of W. Consequently, the leading eigenvector of W˜ becomes the second eigenvector of W.

Proposition B.1. *Let* W = 2 (I+D−<sup>1</sup>A) *be the lazy random walk matrix for a graph on* n *vertices. Let* J = (ji,j )1≤i,j≤<sup>n</sup> *be a matrix such that* ji,j = 1 *for all* i, j*. Define* W˜ = W − n J*. Then, for all* i ≥ 1*,* λi(W˜ ) = λi+1(W) *and* vn(W˜ ) = v1(W)*. Additionally,* <sup>v</sup>n(W˜ ) = <sup>v</sup>1(W) = [ √ n , . . . , √ n <sup>⊺</sup> *and* λn(W˜ ) = 0*.*

*Proof.* Recall that <sup>v</sup>1(W) = h √ 1 n , . . . , √ 1 n i⊺ and λ1(W) = 1. We have:

$$\tilde{W} \cdot \mathbf{v}_1(W) = W \cdot \mathbf{v}_1(W) - \frac{1}{n} J \mathbf{v}_1(W) = \mathbf{v}_1(W) - \left[ \frac{1}{\sqrt{n}}, \dots, \frac{1}{\sqrt{n}} \right]^T = \mathbf{0}.$$

Therefore, the vector h √ n , . . . , √ n i⊺ is an eigenvector of W˜ with eigenvalue 0. Since 0 is the minimum eigenvalue of W˜ , it follows that vn(W˜ ) = v1(W) and λn(W˜ ) = 0.

Next, let us consider vi(W) for i ≥ 2. Since, vi(W) ⊥ v1(W), we obtain that the sum of all elements in vi(W) is zero. Thus,

$$\tilde{W}\mathbf{v}_i(W) = W\mathbf{v}_i(W) - \frac{1}{n}J\mathbf{v}_i(W) = \lambda_i(W)\mathbf{v}_i(W).$$

This implies that, for all i ≥ 2, vi(W) is also an eigenvector of W˜ with the same eigenvalue. Consequently, as the largest eigenvalue of W becomes the smallest eigenvalue of W˜ , we have λi−1(W˜ ) = λi(W) and vi−1(W˜ ) = vi(W).

## C. Minimum Degree Estimation

We will now demonstrate that the value of δ computed in Line 2 of Algorithm [1](#page-4-0) has a low probability of overestimating the minimum degree of the input graph. This implies that, with large probability, we do not need to modify the input graph in Line 3 of the algorithm.

Proposition C.1. *With probability at least* 1 − ζ*, we have* δ < min<sup>i</sup> d<sup>i</sup> *.*

*Proof.* We have δ > min<sup>i</sup> d<sup>i</sup> only if there is ˜d<sup>i</sup> such that ˜d<sup>i</sup> − ϵ log <sup>n</sup> <sup>2</sup><sup>ζ</sup> > d<sup>i</sup> . This implies that the value sampled from the Laplace distribution at Line 1, denoted by l<sup>i</sup> is larger than <sup>10</sup> ϵ log <sup>n</sup> 2ζ . By the property of the Laplace distribution, for all i, we have that:

$$\Pr \left[ |i_i| > \frac{10}{\epsilon} \log \frac{n}{2\zeta} \right] = \frac{1}{2} \exp \left( -\frac{10}{\epsilon} \log \frac{n}{2\zeta} / \frac{10}{\epsilon} \right) = \zeta / n.$$

Then, by the union bound, the probability that there is an index i such that l<sup>i</sup> > ϵ log <sup>n</sup> 2ζ is not greater than ζ.

Suppose that ζ = 1 n . In the next proposition, we shown that δ ≥ √ n log<sup>4</sup> n with large probability.

Proposition C.2. Pr[δ < √ n log<sup>4</sup> n] ≤ 2n .

*Proof.* In Line 2 of Algorithm [1,](#page-4-0) Laplacian noise with a scale of <sup>10</sup> ϵ is added. It follows that ˜d<sup>i</sup> < d<sup>i</sup> − 20 ϵ log n if the noise added to d<sup>i</sup> is less than − 20 ϵ . This event occurs with probability

$$\frac{1}{2} \exp\left(-\frac{20/\epsilon \cdot \log n}{10/\epsilon}\right) = \frac{1}{2n^2}.$$

Using the union bound, we have:

$$\Pr \left[ \min_i \tilde{d}_i < \min_i d_i - \frac{20}{\epsilon} \log n \right] \leq \Pr \left[ \tilde{d}_i < d_i - \frac{20}{\epsilon} \log n \text{ for some } i \right] \leq \frac{1}{2n}.$$

774

776

778

794

796

800

804

806

808

Given that δ = min<sup>i</sup> ˜d<sup>i</sup> − 10 ϵ log <sup>n</sup> 2ζ , and under the assumption in Section [2.5](#page-3-1) that the minimum degree of the network is at least 2 √ n log<sup>4</sup> n, we can bound:

$$\Pr [\delta < \sqrt{n} \log^4 n] \leq \Pr \left[ \delta < \min_i d_i - \frac{20}{\epsilon} \log n - \frac{10}{\epsilon} \log \frac{n}{2\zeta} \right] \leq \frac{1}{2n},$$

for sufficiently large n.

## D. Size of Laplace Noise

In this section, we analyze the effect of adding the Laplace noise at Line 6 of the algorithm. Let the noise added by the node i at the iteration t is y (t) i . Define the vector y (t) as [y (t) 1 , . . . , y (t) <sup>n</sup> ] ⊺ . Also, for all i, t, let e (t) i be a real number such that y (t) = e (t) <sup>1</sup> <sup>v</sup>1(W˜ ) + · · · <sup>+</sup> <sup>e</sup> (t) <sup>n</sup> vn(W˜ ).

Let the initial vector denoted by x (0) = c1v1(W˜ ) + · · · + cnvn(W˜ ), and the final vector is denoted by x (T) . We obtain the following lemma by the notation.

Lemma D.1. *Let* c˜1, . . . , c˜<sup>n</sup> *be numbers such that* x (T) = ˜c1v1(W˜ ) + · · · + ˜cnvn(W˜ )*. We obtain that* c˜<sup>i</sup> = ciλi(W˜ ) <sup>T</sup> + e (1) <sup>i</sup> <sup>λ</sup>i(W˜ ) <sup>T</sup> <sup>−</sup><sup>1</sup> + · · · + e (T) i *.*

*Proof.* To prove the statement, let c (t) <sup>i</sup> <sup>=</sup> <sup>c</sup>iλi(W˜ ) <sup>t</sup> + e (1) <sup>i</sup> <sup>λ</sup>i(W˜ ) <sup>t</sup>−<sup>1</sup> + · · · + e (t) i . We proceed by induction on t to show that, for all t ≥ 0, x (t) = c (t) <sup>1</sup> <sup>v</sup>1(W˜ ) + · · · <sup>+</sup> <sup>c</sup> (t) <sup>n</sup> vn(W˜ ). When t = 0, c (0) <sup>i</sup> = c<sup>i</sup> , so the statement holds directly by the definition of the notation. Assume the statement is true for t − 1; that is, x (t−1) = c (t−1) <sup>1</sup> <sup>v</sup>1(W˜ ) + · · · <sup>+</sup> <sup>c</sup> (t−1) <sup>n</sup> vn(W˜ ). Then, for x (t) , we have

$$\mathbf{x}^{(t)} = \tilde{W} \cdot \mathbf{x}^{(t-1)} + \mathbf{y}^{(t)}.$$

Expanding this using the induction hypothesis gives

$$\mathbf{x}^{(t)} = (\mathbf{c}_1^{(t-1)} \lambda_1(\tilde{W}) + e_1^{(t)}) \mathbf{v}_1(\tilde{W}) + \cdots + (\mathbf{c}_n^{(t-1)} \lambda_n(\tilde{W}) + e_n^{(t)}) \mathbf{v}_n(\tilde{W}).$$

Thus, we obtain x (t) = c (t) <sup>1</sup> <sup>v</sup>1(W˜ ) + · · · <sup>+</sup> <sup>c</sup> (t) <sup>n</sup> vn(W˜ ), completing the induction.

From now, let vi(W˜ ) = [vi,1, . . . , vi,n] ⊺ . We will now calculate the size of each variable. Recall from Line 4 of Algorithm [1](#page-4-0) that x (0) i is sampled from the Gaussian distribution with expected value 0 and standard deviation 1.

Lemma D.2. *For each* i*, the variable* c<sup>i</sup> *is a normal random variable with mean* 0 *and standard deviation* 1*. Furthermore, for* i ̸= j*,* c<sup>i</sup> *is independent to* c<sup>j</sup> *.*

*Proof.* Since, for all i, the eigenvector vi(W˜ ) is a unit vector and c<sup>i</sup> = ⟨x (0) , vi(W˜ )⟩, we have that c<sup>i</sup> = P j vi,jx (0) j . Because c<sup>i</sup> is a linear combination of normal random variables, c<sup>i</sup> is a normal random variable. Furthermore,

$$\mathbb{E}[c_i] = v_{i,1}\mathbb{E}[x_1^{(0)}] + \cdots + v_{i,n}\mathbb{E}[x_n^{(0)}] = 0,$$

and

$$\text{Var}(c_i) = v_{i,1}^2 \text{Var}[x_1^{(0)}] + \cdots + v_{i,n}^2 \text{Var}[x_n^{(0)}] = v_{i,1}^2 + \cdots + v_{i,n}^2 = 1.$$

Since vi(W˜ ) is orthogonal to v<sup>j</sup> (W˜ ) for i ̸= j, the coefficients c<sup>i</sup> and c<sup>j</sup> , which are the dot products of x (0) with vi(W˜ ) and v<sup>j</sup> (W˜ ) respectively, are independent of each other.

Next, we give analyze the variables e (t) i . We observe that, although the random variable is a linear combination of the Laplace variables y (t) j , it is not itself Laplace-distributed.

828

831

834

836

838

854

856

858

860

864

866

868

874

876

878

*Proof.* According to Line 6 of Algorithm [1,](#page-4-0) for all t and i ̸= j, the variables y (t) i and y (t) j are independent, with <sup>E</sup>(y (t) i ) = <sup>E</sup>(y (t) j ) = 0 and Var(y (t) i ) = Var(y (t) j ). The variable e (t) i is defined as the dot product between vi(W˜ ) and y (t) . Specifically, if vi(W˜ ) = [vi,1, . . . , vi,n] ⊺ , then e (t) <sup>i</sup> = P j vi,jy (t) j . Consequently, <sup>E</sup>(e (t) i ) = P j vi,j<sup>E</sup>[y (t) j ] = 0.

Next, for i ̸= j, we examine the covariance between e (t) i and e (t) j , denoted as Cov(e (t) i , e (t) j ). Since <sup>E</sup>(e (t) i ) = <sup>E</sup>(e (t) j ) = 0, {y (t) 1 , . . . , y (t) <sup>n</sup> } are independent with mean 0, and v<sup>i</sup> is orthogonal to v<sup>j</sup> , we have:

$$\text{Cov}(e_i^{(t)}, e_j^{(t)}) = \mathbb{E} \left[ \sum_{i', j'} v_{i, i'} y_{i'}^{(t)} v_{j, j'} y_{j'}^{(t)} \right] = \sum_{i', j'} v_{i, i'} v_{j, j'} \mathbb{E}[y_{i'}^{(t)} y_{j'}^{(t)}] = \sum_k v_{i, k} v_{j, k} \mathbb{E}[(y_k^{(t)})^2] = \mathbb{E}[(y_1^{(t)})^2] \cdot \sum_k v_{i, k} v_{j, k} = 0.$$

Let C<sup>t</sup> represent the scale of the Laplace noise in Line 6 during the t-th iteration of Algorithm [1.](#page-4-0) By definition, Var(y (t) i ) = 2C 2 t for every i. The variance of e (t) i is discussed in the following lemma. Our proof draws on ideas from the paper [\(Li &](#page-8-26) [Tkocz,](#page-8-26) [2023\)](#page-8-26).

Lemma D.4. *For all* i *and* t*, the variance of* e (t) i *is* 2 · C 2 t *. Furthermore,* Pr[e (t) <sup>i</sup> ≥ √ 2C<sup>t</sup> log n] ≤ e n *.*

*Proof.* Based on the argument in the proof of Lemma [D.3,](#page-14-2) we have e (t) <sup>i</sup> = P j vi,jy (t) j . Consequently, Var(e (t) i ) = P j v 2 i,jVar(y (t) j ) for all i and t. Since y (t) j is a Laplace variable with scale C<sup>t</sup> and each vi(W˜ ) is a unit vector, it follows that Var(e (t) j ) = 2 · C 2 t .

Using the Chernoff bound and the moment generating function of the Laplacian distribution, we obtain that

$$\begin{aligned}\Pr[e_i^{(t)} \geq \sqrt{2}C_t \log n] &\leq e^{-\log n} \cdot \mathbb{E} \left[ \exp \left( \frac{\sum_j v_{i,j} y_j^{(t)}}{\sqrt{2}C_t} \right) \right] = \frac{1}{n} \prod_j \mathbb{E} \left[ \exp \left( \frac{v_{i,j} y_j^{(t)}}{\sqrt{2}C_t} \right) \right] \\ &= \frac{1}{n} \prod_j \mathbb{E} \left[ \exp \left( \frac{v_{i,j}}{\sqrt{2}} \cdot \text{Lap}(0,1) \right) \right] \\ &= \frac{1}{n} \prod_j \frac{1}{1 - \frac{1}{2} v_{i,j}^2} \\ &\leq \frac{1}{n} \exp \sum_j v_{i,j}^2 = \frac{e}{n}.\end{aligned}$$

Let h be a positive integer. We discuss the property of the vector W˜ <sup>h</sup>y (t) := [γ (h,t) 1 , . . . , γ (h,t) <sup>n</sup> ] ⊺ in the next lemma.

Lemma D.5. *For all* i, h, t*, the probability that* |γ (h,t) i | ≥ 3 √ 2 · λ1(W˜ ) h · C<sup>t</sup> · log n *is at most* 2e/n<sup>3</sup> *.*

*Proof.* From the definition of γ (h,t) i and the argument in Lemma [D.1,](#page-14-1) we find that γ (h,t) <sup>i</sup> = P j λ<sup>j</sup> (W˜ ) h · vj,i · e (t) j . According to Lemma [D.3,](#page-14-2) Cov(e (t) j , e (t) j ′ ) = 0 for j ̸= j ′ . Therefore, by Lemma [D.4,](#page-15-0)

$$\text{Var}(\gamma_i^{(h,t)}) = \sum_j \lambda_j(\tilde{W})^{2h} \cdot v_{j,i}^2 \cdot \text{Var}(e_j^{(t)}) = 2C_t^2 \cdot \sum_j \lambda_j(\tilde{W})^{2h} \cdot v_{j,i}^2 \leq 2C_t^2 \cdot \lambda_1(\tilde{W})^{2h} \cdot \sum_j v_{j,i}^2 = 2C_t^2 \cdot \lambda_1(\tilde{W})^{2h}.$$

Since e (t) j is a linear combination of Laplace variables, γ (h,t) i is also a linear combination of the Laplace variable y (t) j . Let a1, . . . , a<sup>n</sup> be real numbers such that γ (h,t) <sup>i</sup> = P j ajy (t) j . We obtain that Var(γ (h,t) i ) = 2 · C 2 t P j a <sup>j</sup> ≤ 2C t · λ1(W˜ ) 2h ,

887 888

890

894

896

898

911

914 915 916

918

924

928

and P j a 2 <sup>j</sup> <sup>≤</sup> <sup>λ</sup>1(W˜ ) 2h . Using the Chernoff bound, we obtain that

$$\begin{aligned} \Pr[\gamma_i^{(h,t)} \geq 3\sqrt{2} \cdot \lambda_1(\tilde{W})^h \cdot C_t \cdot \log n] &\leq e^{-3 \log n} \cdot \mathbb{E} \left[ \exp \left( \frac{\gamma_i^{(h,t)}}{\sqrt{2} \cdot \lambda_1(\tilde{W})^h \cdot C_t} \right) \right] \\ &\leq \frac{1}{n^3} \mathbb{E} \left[ \exp \left( \frac{\sum_j \mathbf{a}_j \cdot \text{Lap}(0, 1)}{\sqrt{2} \cdot \lambda_1(\tilde{W})^h} \right) \right] \\ &= \frac{1}{n^3} \prod_j \frac{1}{1 - \frac{\mathbf{a}_j^2}{2\lambda_1(\tilde{W})^{2h}}} \\ &\leq \frac{1}{n^3} \exp \left( \frac{1}{2\lambda_1(\tilde{W})^{2h}} \sum_j \mathbf{a}_j^2 \right) \leq \frac{1}{n^3} \exp(1). \end{aligned}$$

The lemma statement follows from the fact that the probability distribution of γh,t is symmetric about 0.

In the next lemma, we analyze the size of the noise added in the algorithm. Recall that the variable δ is the noisy minimum degree published at Line 2 of Algorithm [1.](#page-4-0) In Proposition [C.2,](#page-13-2) we show that δ ≥ √ n log<sup>4</sup> n with probability at least 1 − 1 n . We denote the event that δ ≥ √ n log<sup>4</sup> n by Eδ.

Lemma D.6. *Recall that* C<sup>t</sup> *is the scale of the noise added at Line 6 of Algorithm [1.](#page-4-0) Then,*

$$\Pr \left[ C_t \leq \frac{10}{9\epsilon} \cdot \frac{\lambda_1(\tilde{W})^{t-1}}{\sqrt{n} \log^2 n} \text{ for all } 1 \leq t \leq T \mid \mathcal{E}_\delta \right] \geq 1 - \frac{8\epsilon T^2}{n^2}.$$

*Proof.* Since x (0) i is drawn from a Gaussian distribution with mean 0 and standard deviation 1, it follows from the properties of a normal random variable that Pr[|x (0) i | ≥ log n · log g] ≤ <sup>n</sup><sup>3</sup> . By applying the union bound, we then have Pr[max<sup>i</sup> |x (0) i | ≥ log n · log g] ≤ 1 <sup>n</sup><sup>2</sup> .

We will prove this lemma by induction on the number of iterations t. For t = 1, recall from Line 6 of the algorithm that the noise y (t) i is drawn from a Laplace distribution with scale parameter <sup>5</sup>·<sup>T</sup> 9·ϵ · max<sup>i</sup> |x (t−1) i δ , where ϵ is the privacy budget and δ is the minimum degree of the input graph. In the event Eδ, the variable δ ≥ √ n log<sup>4</sup> n. Recall that we set T = 2log <sup>n</sup> log g in our algorithm. Consequently, the noise scale in the first iteration is larger than <sup>10</sup> 9ϵ log n log g · log √ <sup>n</sup>·log <sup>g</sup> <sup>n</sup> log<sup>4</sup> <sup>n</sup> = 9ϵ· √ n log<sup>2</sup> n with probability not larger than 1/n<sup>2</sup> when n is large enough.

Next, assume that, in the event Eδ, with probability not smaller than 1 − 2e·(2t−2)<sup>2</sup> <sup>n</sup><sup>2</sup> , for all t ′ < t, the noise (denoted by y (t i ) is sampled from a Laplace distribution with a scale no more than <sup>10</sup> 9ϵ · λ1(W˜ ) t ′−<sup>1</sup> √ n log<sup>2</sup> n . From our previous calculations, it follows that x (t) = W˜ <sup>t</sup>x (0) + W˜ <sup>t</sup>−<sup>1</sup>y (1) + · · · + y (t) . Let W˜ <sup>t</sup>x (0) = [x (t) 1 , . . . , x (t) <sup>n</sup> ] ⊺ and, for all t ′ ≤ t, W˜ <sup>t</sup>−<sup>t</sup> y (t ) = [y (t,t′ 1 , . . . , y (t,t′ ) <sup>n</sup> ] ⊺ . The value of max<sup>i</sup> |x (t−1) i |, which decides the noise scale of y (t) , is equal to max<sup>i</sup> x (t−1) <sup>i</sup> + tP−1 t ′=1 y (t−1,t′ ) i .

Let us first consider the vector [x (t−1) , . . . , x (t−1) <sup>n</sup> ] ⊺ . Recall that vi(W˜ ) = [vi,1, . . . , vi,n] ⊺ . By the notation, we have x (t−1) <sup>i</sup> = P j λ<sup>j</sup> (W˜ ) <sup>t</sup>−<sup>1</sup>vj,ic<sup>j</sup> .

Since, by Lemma [D.2,](#page-14-3) c<sup>j</sup> and c<sup>j</sup> ′ are independent for j ̸= j ′ , we obtain:

$$\mathbb{E}[\mathbf{x}_i^{(t-1)}] = \sum_j \lambda_j (\tilde{W})^{t-1} v_{j,i} \cdot \mathbb{E}[c_j] = 0,$$

$$\text{Var}[\mathbf{x}_i^{(t-1)}] = \sum_j \lambda_j (\tilde{W})^{2t-2} v_{j,i}^2 \text{Var}[c_j] \leq \lambda_1 (\tilde{W})^{2t-2} \text{Var} \left[ \sum_j v_{j,i} c_j \right] = \lambda_1 (\tilde{W})^{2t-2} \text{Var} \left[ x_i^{(0)} \right] = \lambda_1 (\tilde{W})^{2t-2}.$$

938

954

956

958

971

974

976

978

987 988 Also, since x (t−1) i is a linear combination of the normal random variable c<sup>j</sup> , we can conclude that x (t−1) i is also normal. By the property of the normal variable, we have Pr h |x (t−1) i | ≥ <sup>1</sup> 2 log n · log g · λ1(W˜ ) t−1 i ≤ 1 <sup>n</sup><sup>3</sup> for all i. By the union bound, Pr h max<sup>i</sup> |x (t−1) i | ≥ <sup>1</sup> 2 log n · log g · λ1(W˜ ) t−1 i ≤ 1 <sup>n</sup><sup>2</sup> .

Let us reconsider the variable γ (h,t) i from Lemma [D.5.](#page-15-1) Note that y (t−1,t′ ) <sup>i</sup> = γ (t−t ′−1,t′ ) . Let E denote the event that C<sup>t</sup> ′ ≤ 10 9ϵ · λ ′−<sup>1</sup> 1 (W˜ ) √ n log<sup>2</sup> n for all t ′ < t. In the event E and Eδ, Lemma [D.5](#page-15-1) implies that, for all i, t, t′ ,

$$\left| y_i^{(t-1, t')} \right| \geq 3\sqrt{2} \cdot \lambda_1^{t-t'-1}(\tilde{W}) \cdot \frac{10}{9\epsilon} \cdot \frac{\lambda_1^{t-1}(\tilde{W})}{\sqrt{n} \log^2 n} = \frac{30\sqrt{2}}{9\epsilon} \cdot \frac{\lambda_1^{t-2}(\tilde{W})}{\sqrt{n} \log^2 n}.$$

with probability at most <sup>2</sup><sup>e</sup> <sup>n</sup><sup>3</sup> .

By applying the union bound, we deduce that for all t, t′ ,

$$\max_i |y_i^{(t-1, t')}| \geq \frac{30\sqrt{2}}{9\epsilon} \cdot \frac{\lambda_1^{t-2}(\tilde{W})}{\sqrt{n} \log^2 n}$$

with probability at most <sup>2</sup><sup>e</sup> <sup>n</sup><sup>2</sup> . By Lemma 4.4 of [\(Mohar,](#page-9-12) [1989\)](#page-9-12), we have that <sup>λ</sup>2(B) <sup>≥</sup> <sup>0</sup> and <sup>λ</sup>1(W˜ ) <sup>≥</sup> 1 2 . When n is sufficiently large, it follows that, for all t, t′ ,

$$\max_i |y_i^{(t-1, t')}| \geq \frac{\lambda_1^{t-1}(\tilde{W}) \log g}{4 \log n} \geq \frac{30\sqrt{2}}{9\epsilon} \frac{\lambda_1^{t-2}(\tilde{W})}{\sqrt{n} \log^2 n}$$

with probability at most <sup>2</sup><sup>e</sup> <sup>n</sup><sup>2</sup> . We finally obtain

$$\Pr \left[ \sum_{t' \leq t-1} \max_i |y_i^{(t-1, t')}| \geq \frac{1}{2} \cdot \lambda_1^{t-1}(\tilde{W}) \mid \mathcal{E}, \mathcal{E}_\delta \right] \leq \frac{2et}{n^2}.$$

Because, for all i and t, the variables x (t−1) i do not depends on the scale of the Laplacian noise and the event E, we obtain that:

$$\begin{aligned} & \Pr \left[ \max_i \left| x_i^{(t-1)} \right| \geq \log n \cdot \log g \cdot \lambda_1(\tilde{W})^{t-1} \mid \mathcal{E}, \mathcal{E}_\delta \right] \\ &= \Pr \left[ \max_i \left| x_i^{(t-1)} + \sum_{t' \leq t-1} y_i^{(t-1, t')} \right| \geq \log n \cdot \log g \cdot \lambda_1(\tilde{W})^{t-1} \mid \mathcal{E}, \mathcal{E}_\delta \right] \\ &\leq \Pr \left[ \max_i \left| x_i^{(t-1)} \right| + \sum_{t' \leq t-1} \max_i \left| y_i^{(t-1, t')} \right| \geq \log n \cdot \log g \cdot \lambda_1(\tilde{W})^{t-1} \mid \mathcal{E}, \mathcal{E}_\delta \right] \\ &\leq \Pr \left[ \max_i \left| x_i^{(t-1)} \right| \geq \frac{1}{2} \log n \cdot \log g \cdot \lambda_1(\tilde{W})^{t-1} \right] + \Pr \left[ \sum_{t' \leq t-1} \max_i \left| y_i^{(t-1, t')} \right| \geq \frac{1}{2} \lambda_1(\tilde{W})^{t-1} \mid \mathcal{E}, \mathcal{E}_\delta \right] \\ &\leq \frac{(2et + 1)}{n^2}. \end{aligned}$$

In the event E and Eδ, max<sup>i</sup> |x (t−1) i | ≥ log n · log g · λ1(W˜ ) <sup>t</sup>−<sup>1</sup> with probability at most <sup>2</sup>et+1 <sup>n</sup><sup>2</sup> . In the event of E and Eδ, the noise scale at the iteration t, denoted by Ct, is at most <sup>10</sup> 9ϵ 2 log n log g log n·log g·λ1(W˜ ) t−1 √ <sup>n</sup> log<sup>4</sup> <sup>n</sup> = 10 9ϵ · λ1(W˜ ) t−1 √ n log<sup>2</sup> n with probability at

994

996

998

1000 1001 1002 1003 This completes the induction step. We can conclude that, for all t ∈ {1, . . . , T}, C<sup>t</sup> ′ ≤ 10 9ϵ · λ1(W˜ ) ′−<sup>1</sup> √ n log<sup>2</sup> n for all t ′ ≤ t with probability at least 1 − 2e(2t) <sup>n</sup><sup>2</sup> when δ ≥ √ n log<sup>4</sup> n.

1004 1005 1006 1007 1008 We will leverage the previous lemma to demonstrate that the outcome of Algorithm [1](#page-4-0) closely aligns with the results obtained through spectral clustering. Recall Lemma [D.1](#page-14-1) that the final vector x (T) <sup>i</sup> = P<sup>n</sup> <sup>j</sup>=1 <sup>c</sup>˜jvj,i when <sup>c</sup>˜<sup>j</sup> <sup>=</sup> <sup>c</sup>jλ<sup>j</sup> (W˜ ) <sup>T</sup> + P<sup>T</sup> <sup>t</sup>=1 e (t) <sup>j</sup> <sup>λ</sup><sup>j</sup> (W˜ ) T −t .

1009

1014

1016

1019

1024

1026

1029 Recall from Lemma [D.2](#page-14-3) that c<sup>i</sup> is a normal random variable with mean 0 and standard deviation 1. We obtain that:

1034

1036

least 1 − 2et+1 <sup>n</sup><sup>2</sup> . As a result,

$$\begin{aligned} \Pr \left[ C_t \geq \frac{10}{9\epsilon} \cdot \frac{\lambda_1(\tilde{W})^{t-1}}{\sqrt{n} \log^2 n} \text{ or } \bar{\mathcal{E}} \mid \mathcal{E}_\delta \right] &\leq \Pr \left[ C_t \geq \frac{10}{9\epsilon} \cdot \frac{\lambda_1(\tilde{W})^{t-1}}{\sqrt{n} \log^2 n} \text{ and } \mathcal{E} \mid \mathcal{E}_\delta \right] + \Pr[\bar{\mathcal{E}} \mid \mathcal{E}_\delta] \\ &\leq \Pr \left[ C_t \geq \frac{10}{9\epsilon} \cdot \frac{\lambda_1(\tilde{W})^{t-1}}{\sqrt{n} \log^2 n} \mid \mathcal{E}, \mathcal{E}_\delta \right] + \frac{2e(2t-2)^2}{n^2} \\ &\leq \frac{2et+1}{n^2} + \frac{2e(2t-2)^2}{n^2} \leq \frac{2e(2t)^2}{n^2}. \end{aligned}$$

Theorem D.7. *For any node* <sup>i</sup> *such that* |v1,i| ≥ √ γ n *. For large enough* n*, we obtain that*

$$\Pr \left[ \left| c_1 \lambda_1(\tilde{W})^T v_{1,i} \right| > \left| \sum_{t=1}^T e_1^T \lambda_1(\tilde{W})^{T-t} v_{1,i} + \sum_{j=2}^n \tilde{c}_j v_{j,i} \right| \right] \geq 0.95 - o(1).$$

*Proof.* We first obtain that

$$\begin{aligned} & \Pr \left[ \left| c_1 \lambda_1(\tilde{W})^T v_{1,i} \right| > \left| \sum_{t=1}^T e_1^t \lambda_1(\tilde{W})^{T-t} v_{1,i} + \sum_{j=2}^n \tilde{c}_j v_{j,i} \right| \right] \\ & \geq \Pr \left[ \left| c_1 \lambda_1(\tilde{W})^T v_{1,i} \right| > \left| \sum_{t=1}^T e_1^t \lambda_1(\tilde{W})^{T-t} v_{1,i} \right| + \left| \sum_{j=2}^n \tilde{c}_j v_{j,i} \right| \right] \\ & \geq \Pr \left[ \left| v_{1,i} \right| \left( \left| c_1 \lambda_1(\tilde{W})^T \right| - \left| \sum_{t=1}^T e_1^{(t)} \lambda_1(\tilde{W})^{T-t} \right| \right) > \left| \sum_{j=2}^n c_j v_{j,i} \lambda_j(\tilde{W})^T \right| + \left| \sum_{j=2}^n \sum_{t=1}^T e_j^{(t)} \lambda_j(\tilde{W})^{T-t} v_{j,i} \right| \right]. \end{aligned}$$

$$\Pr \left[ \left| c_1 \lambda_1(\tilde{W})^T \right| \geq \frac{\lambda_1(\tilde{W})^T}{16} \right] > 0.95. \quad (13)$$

Recall from Lemma [D.4](#page-15-0) that Pr h e (t) <sup>1</sup> ≥ √ 2C<sup>t</sup> log n i ≤ e n . Let E be the event that max<sup>t</sup> C<sup>t</sup> ≤ 10 9ϵ · λ1(W˜ ) t−1 √ n log<sup>2</sup> n and E<sup>δ</sup> be the event that δ ≥ √ n log<sup>4</sup> <sup>n</sup>. We obtain that Pr h e (t) <sup>1</sup> ≥ 10√ 2 9ϵ λ1(W˜ ) t−1 √ n log n |E, E<sup>δ</sup> i ≤ e n . Denote P<sup>T</sup> <sup>t</sup>=1 e (t) <sup>j</sup> <sup>λ</sup>1(W˜ ) <sup>T</sup> <sup>−</sup><sup>t</sup> by η. By the union bound,

$$\Pr \left[ \eta \geq \frac{\lambda_1(\tilde{W})^T}{32} \mid \mathcal{E} \right] \leq \Pr \left[ \eta \geq \frac{10\sqrt{2} \cdot T}{9\epsilon} \frac{\lambda_1(\tilde{W})^{T-1}}{\sqrt{n} \log n} \mid \mathcal{E} \right] \leq \frac{eT}{n},$$

1054

1056

1063 1064 1065 1066 1067 By the assumption that λ2(W˜ ) ≤ λ1(W˜ ) g , we have that P<sup>n</sup> <sup>j</sup>=2 <sup>c</sup>jvj,iλ<sup>j</sup> (W˜ ) T  <sup>≤</sup> λ1(W˜ ) T <sup>g</sup><sup>T</sup> · P<sup>n</sup> <sup>j</sup>=2 c<sup>j</sup> . Since P<sup>n</sup> <sup>j</sup>=2 <sup>c</sup>jvj,iλ<sup>j</sup> (W˜ ) T is a normal random variable with mean 0 and standard deviation at most √ n·λ1(W˜ ) T <sup>g</sup><sup>T</sup> , we have the following.

1068 1069

1074

1076

1079

1096 1097 1098 1099 We observe that both e (t) j and β<sup>t</sup> = P<sup>n</sup> <sup>j</sup>=2 e (t) <sup>j</sup> <sup>λ</sup><sup>j</sup> (W˜ ) <sup>T</sup> <sup>−</sup><sup>t</sup>vj,i can be written as linear combinations of y (t) 1 , . . . , y (t) <sup>n</sup> . Let b (t) 1 , . . . , b (t) <sup>n</sup> ∈ <sup>R</sup> be such that β<sup>t</sup> = P<sup>n</sup> <sup>j</sup>=1 b (t) j y (t) j . By [\(17\)](#page-19-1), Var [βt] = 2P<sup>n</sup> <sup>j</sup>=1 b <sup>j</sup>C 2 <sup>t</sup> ≤ 2 λ1(W˜ ) 2T−2t g <sup>2</sup>T−2<sup>t</sup> C t and

Pr[E | Eδ] ≥ 1 − 8eT <sup>2</sup> <sup>n</sup><sup>2</sup> . Also, by Proposition [C.2,](#page-13-2) we know that Pr[E¯ <sup>δ</sup>] ≤ 1 2n . As a result, when n is large enough,

$$\begin{aligned} \Pr \left[ \eta \geq \frac{\lambda_1(\tilde{W})^T}{32} \right] &\leq \Pr \left[ \eta \geq \frac{\lambda_1(\tilde{W})^T}{32} \text{ and } \mathcal{E} \text{ and } \mathcal{E}_\delta \right] + \Pr[\bar{\mathcal{E}} \text{ and } \mathcal{E}_\delta] + \Pr[\bar{\mathcal{E}}_\delta] \\ &\leq \Pr \left[ \eta \geq \frac{\lambda_1(\tilde{W})^T}{32} \mid \mathcal{E}, \mathcal{E}_\delta \right] + \Pr[\bar{\mathcal{E}} \mid \mathcal{E}_\delta] + \frac{1}{2n} = o(1). \end{aligned}$$

Since the distribution of η is symmetric around 0, we have that

$$\Pr \left[ |\eta| \geq \frac{\lambda_1(\tilde{W})^T}{32} \right] = o(1). \quad (14)$$

By combining [\(13\)](#page-18-1) and [\(14\)](#page-19-0) and using the fact that |v1,i| ≥ √ γ n , we obtain that:

$$\Pr \left[ |v_{1,i}| \left( |c_1 \lambda_1(\tilde{W})^T| - |\eta| \right) \geq \frac{\gamma \lambda_1(\tilde{W})^T}{32\sqrt{n}} \right] > 0.95 - o(1) \quad (15)$$

$$\Pr \left[ \left| \sum_{j=2}^n c_{jv_{j,i}\lambda_j(\tilde{W})^T} \right| \geq \frac{\sqrt{n} \log n \cdot \lambda_1(\tilde{W})^T}{g^T} \right] < \frac{1}{n^2}.$$

When T = 2 log n log g and n is large enough, we obtain that √ n log n <sup>g</sup><sup>T</sup> < γ 65√ n . Hence,

$$\Pr \left[ \left| \sum_{j=2}^n c_j v_{j,i} \lambda_j(\tilde{W})^T \right| \geq \frac{\gamma \cdot \lambda_1(\tilde{W})^T}{65\sqrt{n}} \right] < \frac{1}{n^2}. \quad (16)$$

Consider the summation P<sup>n</sup> j=2 P<sup>T</sup> <sup>t</sup>=1 e (t) <sup>j</sup> <sup>λ</sup><sup>j</sup> (W˜ ) <sup>T</sup> <sup>−</sup><sup>t</sup>vj,i. Denote the summation as βt. By Lemma [D.3,](#page-14-2) we know that <sup>E</sup>[e (t) j ] = 0 for all j, t. As a result, <sup>E</sup>[βt] = <sup>E</sup> hP<sup>n</sup> <sup>j</sup>=2 e (t) <sup>j</sup> <sup>λ</sup><sup>j</sup> (W˜ ) <sup>T</sup> <sup>−</sup><sup>t</sup>vj,ii = 0 for all t. Furthermore, by λ<sup>j</sup> (W˜ ) ≤ λ1(W˜ ) g for all j ≥ 2,

$$\begin{aligned}
\text{Var}[\beta_t] &\leq \sum_{j=2}^n \lambda_j (\tilde{W})^{2T-2t} v_{j,i}^2 \text{Var}(e_j^{(t)}) \\
&\leq \sum_{j=1}^n \frac{\lambda_1(\tilde{W})^{2T-2t}}{g^{2T-2t}} v_{j,i}^2 \text{Var}(e_j^{(t)}) \\
&= \frac{\lambda_1(\tilde{W})^{2T-2t}}{g^{2T-2t}} \text{Var} \left[ \sum_j v_{j,i} e_j^{(t)} \right] \\
&= \frac{\lambda_1(\tilde{W})^{2T-2t}}{g^{2T-2t}} \text{Var} [y_i^{(t)}] \\
&= 2 \frac{\lambda_1(\tilde{W})^{2T-2t}}{g^{2T-2t}} C_t^2. \tag{17}
\end{aligned}$$

1104

1106

1109

1111

1114

1116

1118 1119

1124

1126

1129

1134

1136

1151

P<sup>n</sup> <sup>j</sup>=1 b <sup>j</sup> ≤ λ1(W˜ ) 2T−2t g 2T−2t . Using the Chernoff bound and the moment generating function of the Laplacian distribution, we obtain that

$$\begin{aligned} \Pr \left[ \beta_t \geq \sqrt{2} \log n \cdot \frac{\lambda_1^{T-t}(\tilde{W})}{g^{T-t}} \cdot C_t \right] &\leq e^{-\log n} \cdot \mathbb{E} \left[ \exp \left( \frac{\beta_t}{\sqrt{2} \cdot \frac{\lambda_1^{T-t}(\tilde{W})}{g^{T-t}} \cdot C_t} \right) \right] \\ &\leq \frac{1}{n} \mathbb{E} \left[ \exp \left( \frac{\sum_j \mathbf{b}_j \text{Lap}(0, 1)}{\sqrt{2} \cdot \frac{\lambda_1^{T-t}(\tilde{W})}{g^{T-t}}} \right) \right] \\ &= \frac{1}{n} \prod_j \frac{1}{1 - \mathbf{b}_j^2 / \left( 2 \cdot \frac{\lambda_1^{2T-2t}(\tilde{W})}{g^{2T-2t}} \right)} \\ &\leq \frac{1}{n} \exp \left( \frac{g^{2T-2t}}{\lambda_1^{2T-2t}(\tilde{W})} \sum_j \mathbf{b}_j^2 \right) \leq \frac{e}{n}. \end{aligned}$$

Let E be the event that max<sup>t</sup> C<sup>t</sup> ≤ 9ϵ · λ1(W˜ ) t−1 √ n log<sup>2</sup> n . For large n such that log n ≥ 650√ 9ϵ · g g−1 · γ ,

$$\Pr \left[ \beta_t \geq \frac{\gamma \lambda_1(\tilde{W})^T}{65 \cdot \frac{g}{g-1} \sqrt{n} g^{T-t}} \mid \mathcal{E} \right] \leq \Pr \left[ \beta_t \geq \frac{10\sqrt{2}}{9\epsilon} \cdot \frac{\lambda_1^{T-1}(\tilde{W})}{\log n \sqrt{n} g^{T-t}} \mid \mathcal{E} \right] \leq \frac{\epsilon}{n}.$$

Recall that E<sup>δ</sup> is the event such that δ ≥ √ n log<sup>4</sup> n. By Lemma [D.6,](#page-16-0) we know that Pr[E | Eδ] = 1 − 8eT <sup>2</sup> <sup>n</sup><sup>2</sup> , and, by Proposition [C.2,](#page-13-2) we know that Pr[E¯ <sup>δ</sup>] ≤ 1 2n . As a result, when n is large enough,

$$\begin{aligned} \Pr \left[ \beta_t \geq \frac{\gamma \lambda_1(\tilde{W})^T}{65 \cdot \frac{g}{g-1} \sqrt{n} g^{T-t}} \right] &\leq \Pr \left[ \beta_t \geq \frac{\gamma \lambda_1(\tilde{W})^T}{65 \cdot \frac{g}{g-1} \sqrt{n} g^{T-t}} \text{ and } \mathcal{E} \text{ and } \mathcal{E}_\delta \right] + \Pr[\bar{\mathcal{E}} \text{ and } \mathcal{E}_\delta] + \Pr[\mathcal{E}_\delta] \\ &\leq \Pr \left[ \beta_t \geq \frac{\gamma \lambda_1(\tilde{W})^T}{65 \cdot \frac{g}{g-1} \sqrt{n} g^{T-t}} \mid \mathcal{E} \right] + \Pr[\bar{\mathcal{E}} \mid \mathcal{E}_\delta] + \frac{1}{2n} = O\left(\frac{1}{n}\right). \end{aligned}$$

By the union bound and by P<sup>T</sup> t=1 <sup>g</sup>T−<sup>t</sup> = g<sup>T</sup> P<sup>T</sup> <sup>t</sup>=1 g <sup>t</sup> = g<sup>T</sup> g <sup>T</sup> +1−1 <sup>g</sup>−<sup>1</sup> = gn<sup>2</sup>−1 gn<sup>2</sup>−n<sup>2</sup> ≤ g g−1 , we obtain that

$$\Pr \left[ \sum_{t=1}^T \beta_t \geq \frac{\gamma \lambda_1(\tilde{W})^T}{65\sqrt{n}} \right] \leq \Pr \left[ \sum_{t=1}^T \beta_t \geq \sum_{t=1}^T \frac{\gamma \lambda_1(\tilde{W})^T}{65 \cdot \frac{g}{g-1} \cdot \sqrt{n} g^{-t}} \right] = O \left( \frac{\log n}{n} \right).$$

As P<sup>T</sup> <sup>t</sup>=1 β<sup>t</sup> is a linear combination of Laplacian random variable, we know that the distribution of the summation is symmetric around 0. Hence,

$$\Pr \left[ \left| \sum_{j=2}^n \sum_{t=1}^T e_j^{(t)} \lambda_j(\tilde{W})^{T-t} v_{j,i} \right| \geq \frac{\gamma \lambda_1(\tilde{W})^T}{65\sqrt{n}} \right] = o(1). \quad (18)$$

We then obtain the lemma statement by combining [\(15\)](#page-19-2) with inequalities [\(16\)](#page-19-3) and [\(18\)](#page-20-1).

## E. Further Experiments

In this appendix, we present additional experimental results, specifically demonstrating that each modification introduced in Algorithm [1](#page-4-0) contributes to improved precision.

#### E.1. Results of using the lazy random walk matrix in the iterative spectral clustering (Difference 3)

In this subsection, we analyze the effect of performing power iteration with the lazy random walk matrix W<sup>α</sup> = αI + (1 − α)D−<sup>1</sup>A instead of the usual random walk matrix D−<sup>1</sup>A used during PIC [\(Lin & Cohen,](#page-8-19) [2010;](#page-8-19) [Boutsidis et al.,](#page-8-20) [2015\)](#page-8-20).

1159 1160 1161

1164

1174

1176

1194

1196

1199 1200

1204

1206

1209

![](_page_21_Figure_7.jpeg)

![](_page_21_Figure_1.jpeg)

Figure 3. Power iteration on BSBM(1000, 1000, 1000, 1000, 0.5, 0.2) for lazy random walk matrices W<sup>α</sup> with α ∈ {0, 0.1, . . . , 0.9}.

We start with an n-dimensional standard normal variable x (0), and iteratively obtain x (t) = W<sup>α</sup> · x (t−1) − 1 n P i x (t−1) i . This is equivalent to the PIC algorithm with k = 2 initial vectors.

For bipartite graphs, the random walk matrix W<sup>0</sup> = D−<sup>1</sup>A has −1 as an eigenvalue. Thus, for bipartite 2-clustered graphs, the performance of PIC is not good unless more initial vectors are selected. We demonstrate this by introducing a *Bipartite Stochastic Block Model* graph with two clusters, defined as follows. Given integers a<sup>i</sup> , b<sup>i</sup> and probabilities p and q, a graph G ∼ BSBM(a1, a2, b1, b2, p, q) has node set A<sup>1</sup> ⊔ A<sup>2</sup> ⊔ B<sup>1</sup> ⊔ B<sup>2</sup> with |A<sup>i</sup> | = a<sup>i</sup> and |B<sup>i</sup> | = b<sup>i</sup> , such that every pair of nodes between A<sup>i</sup> and B<sup>i</sup> is added with probability p, and A<sup>i</sup> and B<sup>j</sup> are added with probability q. This graph is bipartite with independent sets A<sup>1</sup> ∪ A<sup>2</sup> and B<sup>1</sup> ∪ B2, and when p ≫ q, admits a clear cluster structure given by the node clusters A<sup>1</sup> ∪ B<sup>1</sup> and A<sup>2</sup> ∪ B2.

Figure [3](#page-21-0) shows that for certain BSBM's, the produced clusters by iteratively multiplying W<sup>α</sup> always have discrepancy close to 1 when α is close to 0. On the other hand, when α ≈ 1, the procedure is too slow to converge since W<sup>α</sup> ≈ I. Therefore, selecting a lazy factor of α = 2 seems to be a natural choice in general when no additional information about the input graph is available.

#### E.2. Result of leading eigenvector elimination (Difference 4)

Figure 4. (Left): Heatmap of average dnorm(NonElim) − dnorm(Ours) over 20 SBMs with n<sup>1</sup> = n<sup>2</sup> = 1000, with varying probabilities p, q ∈ {0.05, 0.1, . . . , 0.95}, and privacy budget ϵ = 2.0. (Right): Discrepancy with increasing ϵ for 20 SBMs with p = 0.3, q = 0.2.

Now, we perform an experiment to investigate the effect of elimination of the leading eigenvalue of the lazy random walk matrix, which is a procedure changing the matrix W to the matrix W˜ described in Difference 4 of Section [3.](#page-3-3) For this experiment, we select ε = 2.0 and generate 20 SBM's of cluster sizes 1000 for pairs of probabilities (p, q).

We present our results in Figure [4.](#page-21-1) The figure illustrates that our algorithm, without the leading vector elimination (referred to as NonElim), consistently fails to recover the original clusters. Although not depicted in the graph, we observed that the NonElim algorithm fails to successfully identify the clusters even when the privacy budget ϵ is set as high as 20. In contrast, our proposed method successfully identifies these clusters with minimal discrepancy.