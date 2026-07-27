# Locally Differentially Private Graph Clustering Via The Power Iteration Method

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054

## Abstract

We propose a locally differentially private graph clustering algorithm. Previous works have explored this problem, including approaches that apply spectral clustering to graphs generated via the randomized response algorithm. However, these methods only achieve accurate results when the privacy budget is in Ω(log n), which is unsuitable for many practical applications. In response, we present an interactive algorithm based on the power iteration method. Given that the noise introduced by the largest eigenvector constant can be significant, we incorporate a technique to eliminate this constant. As a result, our algorithm attains local differential privacy with a constant privacy budget when the graph is well-clustered and has a minimum degree of Ω( ˜
√n). In contrast, while randomized response has been shown to produce accurate results under the same minimum degree condition, it is limited to graphs generated from the stochastic block model. We perform experiments to demonstrate that our method outperforms spectral clustering applied to randomized response results.

## 1. Introduction

As the adoption of artificial intelligence expands, ensuring the protection of user privacy has become a critical priority. Various techniques have been proposed to tackle privacy concerns, with differential privacy emerging as a leading approach. Differential privacy, introduced in (Dwork, 2008), quantifies the privacy leakage of a system using a parameter known as the privacy budget. The core idea involves introducing noise to users' data to obscure individual information while still enabling meaningful statistical analysis. The challenge of designing algorithms that can draw accurate insights from this noisy data has garnered significant attention from researchers (Zhu et al., 2017), as it is essential to balance privacy protection with the utility of the resulting analysis.

. AUTHORERR: Missing \**icmlcorrespondingauthor.**
Preliminary work. Under review by the International Conference on Machine Learning (ICML). Do not distribute.

1 In this work, we focus on a specific variant of differential privacy known as local differential privacy (LDP) (Kasiviswanathan et al., 2011). Unlike traditional differential privacy, which allows data collection before noise is added, LDP requires users to anonymize their data directly on their local devices before transmitting it to a central server. This approach ensures that sensitive information remains protected during transmission, as the data is already corrupted at the source. LDP has been adopted by several major companies (Erlingsson et al., 2014; Apple's Differential Privacy Team, 2017) in their services to safeguard user privacy while still enabling data analysis at scale. We focus on developing LDP algorithms for social networks, where users are represented as nodes and their relationships as edges. Since these connections are considered sensitive, they are protected using privacy notions such as edge LDP
(Qin et al., 2017) or node LDP (Ye et al., 2020). However, with some exceptions like (Zhang et al., 2020), node LDP is generally too stringent, making it difficult to release useful information in most applications. As a result, the majority of research in LDP has centered around the more practical edge LDP framework (Imola et al., 2021). To protect user's information, one widely used technique is randomized response, also known as edge flipping (Warner, 1965; Mangat, 1994; Wang et al., 2016). In this method, before a user sends a bit vector which encodes their list of friends to a central server, each bit in the vector is flipped with a certain probability. The server aggregates the obfuscated adjacency vector to construct an obfuscated version of the graph. Although it is possible to compute various graph statistics from this obfuscated data, the accuracy of these statistics is often reduced. Algorithms designed specifically to publish particular statistics tend to offer more precise and insightful results about the graph (Imola et al., 2021; 2022). Graph clustering illustrates how analyzing a graph obfuscated by randomized response can lead to inaccurate results.

Let n be the number of nodes in the input graph. In (Hehir et al., 2022), the authors demonstrated that spectral clustering (Ng et al., 2001) can yield accurate results with a privacy budget in O(1), provided the input graphs are generated from stochastic block models and have an average degree of Θ(√n) (Holland et al., 1983). For general graphs, (Mukherjee & Suppakitpaisarn, 2023) showed that applying spectral clustering to randomized response data only yields accurate results when the privacy budget ϵ ∈ Ω(log n), which is too large for many real-world applications. Furthermore, even for dense graphs, when ϵ ∈ o(log n), the authors identified a class of graphs for which clustering results are inaccurate. Although numerous algorithms have been proposed for clustering under differential privacy (Ji et al., 2020; Mohamed et al., 2022; Chen et al., 2023; Imola et al., 2023; Epasto et al., 2024; He et al., 2024), relatively few have been developed specifically for publishing clustering results under edge LDP. Aside from the work mentioned in the previous paragraph, the only other algorithm we are aware of targets node LDP rather than edge LDP (Fu et al., 2023).

## 1.1. Our Contributions

In this work, we aim to develop a dedicated algorithm for graph clustering under the edge LDP framework. Rather than using non-interactive methods like the randomized response algorithm, we propose an interactive approach, which has been shown to achieve better performance for many edge LDP tasks (Henzinger et al., 2024; Hillebrand et al., 2024). Specifically, we draw inspiration from the work in (Betzer et al., 2024), where the authors employ multi-round interactive algorithms to compute iterative matrix multiplications for Katz centrality. Since spectral clustering can also be derived through iterative matrix multiplication using the Power Iteration Clustering (PIC) algorithm (Lin & Cohen, 2010; Boutsidis et al., 2015), we propose extending this approach to calculate clusters via the PIC algorithm under the edge LDP framework. Unfortunately, calculating the PIC algorithm under the edge LDP framework is not straightforward. While the goal is to compute the second eigenvector through the iterative process, the largest component of the result is the first eigenvector. In a non-private setting, the first eigenvector, being a uniform vector, does not interfere with the calculation of the PIC algorithm. However, when protecting users' sensitive information under edge LDP, noise must be added at a magnitude comparable to the largest terms. This causes the noise to dominate the result, especially as the number of iterations increases, leading to a significant loss in accuracy.

We propose a technique to eliminate the largest constant term, enabling the development of an algorithm that achieves accurate results with a constant privacy budget when the minimum degree of the input graph is Ω( ˜
√n).

Recall that randomized response is proven to yield accurate results for graphs generated by the stochastic block model when the minimum degree is Ω( ˜
√n). Our algorithm, however, provides precise results under the same minimum degree condition but applies to general graphs, not limited to those generated by the model. This extends the applicability of our clustering algorithm to a wider range of input graphs. Our algorithm is computationally efficient. It requires O(log n) interactions between users and the central server, with each node having a computational complexity of O(n) per iteration. The central server also has a computational complexity of O(n) per iteration. Consequently, the total computation time of our algorithm is O(n log n). Additionally, the communication cost for each user is also O(n log n). Compared to the spectral clustering algorithm applied to the randomized response results (Hehir et al., 2022; Mukherjee & Suppakitpaisarn, 2023), our iterative method is significantly more memory-efficient. In the previous approach, the server requires Θ(n 2) bits of memory to store the randomized response results (Imola et al., 2022; Hillebrand et al., 2023). In contrast, our algorithm reduces the memory requirement to Θ(n) for both the server and the users. This improvement enables our method to handle graphs with a large number of nodes, which would be infeasible to process using the earlier algorithm.

We validate our algorithm through experiments on graphs generated using the stochastic block model (Holland et al., 1983) and the Reddit graph (Hamilton et al., 2017). Compared to applying the spectral clustering algorithm to the randomized response results (Hehir et al., 2022), our algorithm produces clustering results that are closer to those of the original spectral clustering algorithm in almost all cases. Notably, there are instances where the previous algorithm yields random outcomes, while our algorithm consistently produces results identical to the original spectral clustering.

## 2. Preliminaries 2.1. Notation

Throughout this paper, we consider a graph G = (*V, E*) with n vertices. Let S ⊆ V represent a subset of vertices, and S denote its complement V \ S.

Let S and S
′ be two disjoint subsets of V (meaning S∩S
′ =
∅). We denote by eG(*S, S*′) the number of edges in G that have one endpoint in S and the other in S
′. For each subset S ⊆ V , let VolG(S) denote the number of edges with both endpoints in S. We refer to VolG(S) as the *volume* of S.

For *S, S*′ ⊆ V , the quantity dvol(*S, S*′) is defined as min(VolG(S△S
′) + VolG(S△S′), VolG(S△S′) +
VolG(S△S
′)). Since S△S
′ = S△S′, this simplifies to dvol(*S, S*′) = min 2VolG(S△S
′), 2VolG(S△S′). Two cuts (S, S) and (S
′, S′) are considered similar if dvol(*S, S*′)
is small. We also define the *normalized discrepancy* as

$$d_{\mathrm{norm}}(S,S^{\prime})={\frac{d_{\mathrm{vol}}(S,S^{\prime})}{\mathrm{Vol}_{G}(V)}}.$$
. (1)
055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 Given that dvol(*S, S*′) ≤ VolG(V ), normalization ensures that 0 ≤ dnorm(*S, S*′) ≤ 1. When S is fixed and nodes are randomly assigned to S
′ with uniform probability, dnorm(*S, S*′) tends to be close to 1.

Any real symmetric n × n matrix A has n real eigenvalues.

We denote the i-th smallest eigenvalue of A as λi(A), so that λ1(A) ≥ λ2(A) *≥ · · · ≥* λn(A). The eigenvector corresponding to λi(A) is denoted by vi(A) = [νi,1, . . . , νi,n]
⊺
.

For each i ∈ [1, n], let ai = [ai,1, . . . , ai,n]
⊺
represent the adjacency list of user vi, where ai,j = 1 signifies the existence of an edge between vi and vj (i.e., (vi, vj ) ∈ E), and ai,j = 0 indicates no edge. The degree of node vi, denoted by di, reflects the number of edges connected to vi. In the context of a locally differentially private algorithm, it is assumed that each user viis aware only of their own adjacency vector ai, which contains sensitive personal information.

## 2.2. Edge Local Differential Privacy

We define two adjacency lists, a and a
′, as neighboring if they differ by exactly one bit, meaning that one can be transformed into the other by either adding or removing a single edge connected to node vi. The concept of edge local differential privacy is formalized as follows: Definition 2.1 (ϵ-Edge LDP Query). Let ϵ > 0. A randomized query R is said to satisfy ϵ-edge local differential privacy (ϵ-edge LDP) if, for any pair of neighboring adjacency lists a and a
′, and any possible outcome set S,
P
 [R(a) ∈ S] ≤ e ϵP
 [R(a
′) ∈ S].

Definition 2.2 (ϵ-edge LDP Algorithm (Qin et al., 2017)). An algorithm A is said to be ϵ-edge LDP if, for any user vi, and any sequence of queries R1*, . . . ,* Rκ posed to user vi, where each query Rj satisfies ϵj -edge local differential privacy (for 1 ≤ j ≤ κ), the total privacy loss is bounded by ϵ1 + *· · ·* + ϵκ ≤ ϵ.

If an algorithm A is ϵ-edge LDP, it is also said to have a privacy budget of ϵ. Next, we introduce a query that satisfies ϵ-edge LDP which designed to estimate a realvalued statistic based on the adjacency vector. Definition 2.3 (Edge Local Laplacian Query (Hillebrand et al., 2023)). Let f : {0, 1}
n → R be a function defined on adjacency lists, and let a ∼ a
′represent neighboring adjacency lists. The global sensitivity of f, denoted as ∆f ,
is defined as: ∆f = maxa∼a′ |f(a) − f(a
′)|.

For any ϵ > 0, a query that returns f(a) + Lap(∆f /ϵ) is ϵ-edge LDP. Here, Lap(b) refers to noise sampled from the Laplace distribution with scale parameter b.

## 2.3. Spectral Clustering

For a given graph G, the primary objective of clustering techniques is to identify a cut (S, S) such that the number of edges crossing between S and S, denoted by eG(S, S), is minimized, while most of the edges are concentrated within S or S. To avoid trivial cuts (such as when S contains only a single vertex), it is common to define the conductance, ϕG(S) = eG(S, S)/ min{VolG(S), VolG(S)}, and seek cuts that minimize ϕG(S) (Shi & Malik, 2000). The conductance of the graph, denoted by ϕ(G), is given by ϕ(G) = min∅⊊S⊊V ϕG(S). Unless otherwise stated, we use S
∗to denote the subset that achieves the minimum normalized cut, where ϕG(S
∗) = ϕ(G).

Let B = (bi,j )1≤i,j≤n be the transition-probability matrix of a random walk on G, given by bi,i = 0 for all i and bi,j = ai,j/di for all i ̸= j. We have that −1 ≤ λi(B) ≤ 1 for all i, λ1(B) = 1, and v1(B) = [ √
1 n
, √
1 n
, . . . , √
1 n
]
⊺
.

Observe that when I is the identity matrix, the matrix I −B is referred to as the *random walk normalized Laplacian* matrix (Von Luxburg, 2007). The eigenvectors of I − B are identical to those of B. More specifically, it is known that, for all i, vi(I − B) = vn−i(B). The spectral clustering algorithm (Shi & Malik, 2000) computes the eigenvector v2(B) = [ν1*, . . . , ν*n]
⊺
, and then produces the cut S
′ = {vi: νi > 0} as the clustering result. Since ϕG(S
′) ≤ 2pϕG(S∗) (Alon, 1986), it is established that the cut produced by the spectral clustering algorithm achieves a low conductance. Additionally, according to (Peng et al., 2015), we have dvol(S
′, S∗) =
O
ϕ(G)
λ3(B)
· VolG(V )
, indicating that S
′closely approximates S
∗in a graph that is well-clustered.

The normalized Laplacian matrix L = (ℓi,j )1≤i,j≤n, defined by ℓi,j = −ai,j/pdi· dj for i ̸= j and ℓi,i = 1, is commonly used in spectral clustering algorithms that aim to minimize the conductance. However, in this work, we opt for the random walk normalized Laplacian matrix, as calculating spectral clustering under the normalized Laplacian is more complex in the edge LDP setting. Notably, when the desired number of clusters is two, the results of spectral clustering using the random walk normalized Laplacian matrix are at least as good as those obtained with the normalized Laplacian matrix (Von Luxburg, 2007).

## 2.4. Power Iteration Clustering

While spectral clustering can produce a cut with a small cutratio, it requires computing the eigenvector v2(B), which can be computationally expensive. To address this, the power iteration clustering algorithm (Lin & Cohen, 2010) offers a more efficient method for estimating the eigenvector, significantly reducing the computation time. Let x be a vector of length n where each element is independently drawn from a Gaussian distribution. It is known that x can be expressed as c1λ1(B)v1(B) + *· · ·* +
110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164

$$i_{n}\lambda_{n}($$

cnλn(B)vn(B), where c1*, . . . , c*n are independent random variables also drawn from a Gaussian distribution. Therefore, for a sufficiently large T, applying BTto x gives:

$\overline{x}$
$\square$
= c1 ·
h√
$\left|\frac{1}{\sqrt{n}},\:1\right|$
, √
i⊺+ c2λ2(B)
T v2(B)
* $\left(\frac{1}{\sqrt{n}},\frac{1}{\sqrt{n}}\right)^{\sf T}+c_{n}$ * $c_{n}\lambda_{n}(B)^{T}{\bf v}_{n}(B)$.  
$$1+\cdots+c_{r}$$
+ · · · + cnλn(B)
T vn(B). (2)
When λ3(B) ≪ λ2(B), the term BT x is approximately:

B
$${}^{T}\mathbf{x}\approx c_{1}\left[\frac{1}{\sqrt{n}},\frac{1}{\sqrt{n}},\ldots,\frac{1}{\sqrt{n}}\right]^{\mathsf{T}}+c_{2}\lambda_{2}(B)^{T}\mathbf{v}_{2}(B),\tag{3}$$
meaning the order of elements in BT x closely mirrors that of v2(B). Therefore, clustering can be performed using BT x instead of v2(B), yielding results similar to those from the spectral clustering algorithm.

## 2.5. Assumptions

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 We assume that the input graph has the following properties:
(1) The minimum degree is at least 2
√n log4n,
(2) There exists a constant g such that for all i ≥ 3, λi(B) +
1 ≤
λ2(B)+1 g,
(3) There exists δ ≈ 1 and γ < 1 such that the components of v2(B) satisfies

ni : |νi| ≥ √
γ n o ≥ δ · n, and
(4) The number of nodes n is larger than a constant C. Assumption (1) The first assumption is essential for any graph clustering algorithm under edge LDP with a constant privacy budget. Protecting the connections of low-degree nodes requires adding so much noise that their contributions are obscured, resulting in unstable clustering outcomes for these nodes. Assumption (2) The second assumption is a standard prerequisite for iterative spectral clustering algorithms, such as the one presented in (Boutsidis et al., 2015). This assumption ensures the convergence of the iterative process. A comprehensive technical explanation supporting this assumption is provided in (Boutsidis et al., 2015). Assumption (3) We demonstrate in Appendix A that the third assumption holds when the graph is well-clustered and most nodes have a degree cluster close to the average degree of the cluster to which they belong. Specifically, for a node i in cluster A ⊆ V , we show in Proposition A.1 that the value of νi exceeds
√σ·c 4· 
q di n·d(A) 
−2 q ϕ(G)
1−λ3(B)
, where d(A) represents the average degree of nodes in cluster A, and *c, σ* ∈ R satisfy the condition that at least c|A| nodes in cluster A have degrees not less than q σ · d(A). If the graph is well-clustered, the term ϕ(G)
1−λ3(B)
becomes small and can be neglected (Mukherjee
& Suppakitpaisarn, 2023). Consequently, we conclude that when di ≥ σd(A) and there are at least c|A| nodes satisfying this condition, it follows that νi ≥
σc 4
· √
1 n
. Moreover, if σ and c are constants, there exist at least c|A| nodes i such that νi = Ω 
√
1 n
.

$T_{\rm NL}$

We observe that the graphs generated by the stochastic block model have this property. In addition to our mathematical proof in the appendix, it is empirically demonstrated in (Abbe et al., 2020) that most of the values in the eigenvectors is in Θ(1/
√n). Additionally, (Balakrishnan et al., 2011)
shows that this assumption can be satisfied when B is a node similarity matrix with certain additional properties. Assumption (4) The final assumption is a common requirement for most differentially private algorithms. A large user base typically allows the added noise, introduced to protect sensitive information, to average out in the results.

## 3. Our Algorithm

We describe our algorithm in Algorithm 1. One can notice that we almost have x
(t) = B ·x
(t−1) and x
(T) = BT·x
(0)
by the calculation at Lines 6 - 7. The only five differences are as follows:
Difference 1: Addition of Laplace Noise We add Laplace noise in Line 6 to protect users' information. Later, we show in Section 4.2 that this noise satisfies the conditions of the edge-local Laplacian query (Definition 2.3).

Furthermore, in Section 4.3, we demonstrate that when the minimum degree is sufficiently large, the magnitude of the Laplacian noise becomes negligible compared to other terms in the calculation in Line 6. Difference 2: Minimum Degree Estimation When B is the normalized random walk Laplacian matrix, calculating B · x
(t−1) does not require knowing the degrees of other nodes. This property simplifies computations within the edge LDP setting and is the main reason we select the normalized random walk Laplacian matrix over the normalized Laplacian matrix in our clustering algorithm. On the other hands, to bound the sensitivity, which determines the scale of the Laplace noise in Line 6, we need a lower bound on the minimum degree of the graph G. This bound is computed in Line 2 of the algorithm, using degree estimates obtained in Line 1 of the mechanism. In Appendix C, We will show that the estimate in Line 2 overestimates the minimum degree with probability not larger than 1 n2 when ζ =
1 n
. If the estimate exceeds the actual minimum degree, we add edges in Line 3 to ensure that the modified graph meets the estimated minimum degree. In Appendix C, we further show that the variable δ exceeds B
T x = c1λ1(B)
T v1(B) + · · · + cnλn(B)
T vn(B)
220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 Algorithm 1 Private Power Iteration Clustering Input: Graph G = (V, E) where V = {v1*, . . . , v*n} and its adjacency matrix is A = (ai,j )1≤i,j≤n, privacy budget ϵ, number of iterations T =
2 log n log g
, clipping factor c, and parameter ζ =
1 n Output: A cut of G denoted by S ⊂ V
1 **[User** i] Compute the degree of vi, denoted by di. Broadcast
˜di ← di + Lap(10/ϵ) to all users and the server.

2 **[Server]** Calculate δ ← mini
˜di −
10 ϵ log n 2ζ
. Broadcast δ to all users.

3 **[User** i] If di < δ, randomly select j such that ai,j = 0, then set ai,j = 1 and increment di by one. Repeat this process until di ≥ δ.

4 **[Server]** Initiate the vector x
(0) = [x
(0)
1*, . . . , x*
(0)
n ]
⊺ where x
(0)
iis chosen from the Gaussian distribution with expected value 0 and standard deviation 1. Broadcast the vector x
(0)
to all users.

5 for t = 1*, . . . , T* do 6 **[User** i] Calculate w
(t)
i =
1 2 x
(t−1)
i +
1 2 Pj ai,j x
(t−1) j di−
1 n Pj x
(t−1)
j + Lap 5·T
9·ϵ maxj |x
(t−1)
j| δ
.

7 **[User** i] Let U = c·
5·T
9·ϵ maxj |x
(t−1)
j| δ, also let x
(t)
i = U
if w
(t)
i > U, x
(t)
i = −U if w
(t)
i < −U, and x
(t)
i =
w
(t)
iotherwise. Calculate and send x
(t)
ito the server.

8 **[Server]** Aggregate the values x
(t)
iinto a vector x
(t) =
[x
(t)
1*, . . . , x*
(t)
n ]
⊺
, and broadcast this information to all users.

9 **[Server]** Return S ← {vi: x
(T)
i > 0}.

√n log4n with probability at least 1 −
1 n
.

Difference 3: Replacing the Random Walk with a Lazy Random Walk Recall that all eigenvalues of the matrix B lie between 1 and −1. In certain networks, such as bipartite graphs, λn(B) can be close to −1. This causes the final term in Equation (2) to oscillate, preventing the calculation of BT x from converging. To address this, we propose replacing B with W =
1 2 I +
1 2B. Note that for all i, vi(W) = vi(B) and λi(W) = 12 λi(B) + 12
. Consequently, for all i, 0 ≤ λi(W) ≤ 1. By the second assumption in Section 2.5, which is λi(W) ≤
λ2(W)
gfor all i ≥ 3, we can have the approximation (3) even when some λi(B) are negative. This modification leads to the first two terms of the calculation in Line 6.

Difference 4: Elimination of the Leading Eigenvector Recall Equation (2). Since λ2(W) < 1, the term α2λ2(W)
T v2(W) diminishes compared to the leading term as T increases. On the other hand, the size of the Laplace noise added depends on the largest element of x
(t−1), which is determined by the leading term. Hence, for larger T, the noise magnitude dominates over the term α2λ2(W)
T v2(W). This causes x
(T)to deviate significantly from v2(W), reducing the accuracy of the results.

To address this, we introduce the matrix W˜ =
( ˜wi,j )1≤i,j≤n, where w˜i,j = wi,j − 1/n for all *i, j*. We show in Appendix B that for all i ≥ 1, λi(W˜ ) = λi+1(W) and vn(W˜ ) = v1(W). Additionally, vn(W˜ ) = v1(W) =
[ √
1 n
, √
1 n
, . . . , √
1 n
]
⊺
and λn(W˜ ) = 0.

With this update, the leading term α1 · [ √
1 n
, √
1 n
, . . . , √
1 n
]
⊺

from (2) is eliminated. The term α2λ2(W˜ )
T v2(W˜ ) now becomes the leading term, and we can ensure that the Laplace noise (the fourth term of Line 6 in Algorithm 1) is substantially smaller than the new leading term. The subtraction of the third term in the calculation at Line 6 reflects the update from W to W˜ .

Difference 5: Clipping At Line 6 of the algorithm, we apply a standard clipping method commonly used in various LDP studies, such as (Imola et al., 2022) and (Betzer et al., 2024). We notice from the proof of Lemma D.6 that when the clipping factor c satisfies c ≥ log n · log g, it holds with high probability that −U ≤ w
(t)
i ≤ U for all i and t.

Consequently, the clipping has no impact on our theoretical results. However, in our experiments, we observed that Algorithm 1 achieves optimal performance when c is set to 5, which is smaller than log n · log g.

## 4. Properties Of Our Algorithm 4.1. Efficiency

Computation Time The primary computational bottleneck of Algorithm 1 occurs in Line 6. In this step, the per-node computational complexity for each iteration is O(n). To achieve accurate results, the required number of iterations T is given by 2 log n log g = Θ(log n), leading to an overall computational complexity of O(n log n) per node. In contrast, the central server has minimal computational demands. Its responsibilities are limited to generating the initial vector, receiving calculation results, and distributing them to all users. Communication Cost While each user uploads only one real number x
(t)
ito the server at each iteration, they must download the entire vector x
(t)in Line 8 of the algorithm.

This results in a total communication cost of O(n log n) for each user.

Memory Consumption During iteration t, the central server and all users only need to store two vectors: x
(t−1)
275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 and x
(t). As a result, the memory consumption for all parties is O(n). This is a significant improvement compared to the randomized response method. Even for sparse input graphs, the randomized response mechanism flips each relationship with a constant probability, leading to a graph with Ω(n 2)
edges. Storing such a graph, with Ω(n 2) edges, requires a prohibitive amount of memory on the server, making it infeasible to design an LDP algorithm for large input graphs (Imola et al., 2022). In contrast, our approach requires only O(n) memory, enabling our algorithms to handle input graphs with millions of nodes efficiently.

## 4.2. Privacy

The following theorem discuss our algorithm's privacy. Theorem 4.1. Algorithm 1 is ϵ-edge LDP. Proof. We perform T + 1 edge-local Laplacian queries to all users: one at Line 1 and T queries at Line 6. At Line 1, the degree di has a sensitivity of one. Since the Laplace noise is set to 10/ϵ, the privacy budget for the publication at Line 1 is ϵ/10.

When any ai,j is changed, the value of x
(t)
icalculated at Line 6 changes by at most 12 maxj |x
(t−1)
j| dj. Therefore, the sensitivity of the publication at Line 6 is 12 maxj |x
(t−1)
j| dj≤
1 2 maxj |x
(t−1)
j| δ. The privacy budget for each publication at Line 6 is 9 10 
·
ϵ T
. Since there are T publications at Line 6, the total privacy budget of Algorithm 1 is ϵ 10+T ·
9 10 
·
ϵ T 
= ϵ.

## 4.3. Precision

In this section, we analyze the precision of Algorithm 1. In particular, we demonstrate that the algorithm's results closely resemble those of the spectral clustering algorithm. We provide an outline of our proof sketch here, with the full proof details available in Appendix D. In Algorithm 1, at iteration t we compute the vector x
(t) =
[x
(t) 1
, . . . , x
(t)
n ]
⊺
. The output of the algorithm is Salg =
{vi| x
(T)
i > 0}, where T =
2 log n log g
.

Let vj (W˜ ) = [vj,1, . . . , vj,n]
⊺ be the j'th eigenvector of W˜ , and let c1*, . . . , c*n ∈ R be coefficients such that x
(0) =Pn j=1 cjvj (W˜ ). Additionally, for all t, suppose the noise added during iteration t of the algorithm is y
(t), and that e
(t)
1*, . . . , e*
(t)
n ∈ R are coefficients such that y
(t) =Pn j=1 e
(t)
j vj (W˜ ). In Lemma D.1, we show that x
(T)
i =Pn j=1 c˜jvj,i, where c˜j is given by c˜j =
cjλj (W˜ )
T +PT
t=1 e
(t)
j λj (W˜ )
T −t.

In Lemma D.6, we show that the noise generated at Line 6 of the algorithm has a small scale. Specifically, we demonstrate that the noise scale, given by 5T
9ϵ maxj |x
(t−1)
j| δ, is negligible compared to the magnitude of x
(t). Consequently, the noise term y
(t) does not dominate the calculation. This emphasizes the significance of removing the leading eigenvector and establishing a lower bound for the minimum degree δ.

Due to the lemma, the term PT
t=1 e
(t)
j λj (W˜ )
T −tis negligible compared to cjλj (W˜ )
T, and we have c˜i ≈
ciλi(W˜ )
T. Consequently, x
(T)
i ≈Pn j=1 cjλj (W˜ )
T vj,i.

Using techniques from (Boutsidis et al., 2015), we show that x
(T)
i ≈ c1λ1(W˜ )
T v1,i when λj (W˜ ) ≤
λ1(W˜ )
gfor all j ≥ 2. Specifically, in Theorem D.7, we demonstrate thatc1λ1(W˜ )
T v1,i
 >
PT
t=1 e T
1 λ1(W˜ )
T −tv1,i +Pn j=2 c˜jvj,i
 with probability at least 0.95 − o(1). The term c1λ1(W˜ )
T v1,i dominates and determines the sign of x
(T)
i.

Since λ1(W˜ )
Tis positive, we conclude that when c1v1,i >
0, x
(t)
i > 0 with high probability. Recall that the outcome of the spectral clustering algorithm is Sorig = {vi: v1,i > 0}. Thus, when c1 > 0, the result Salg closely resembles Sorig with high probability. Conversely, when c1 < 0, the result Salg is similar to V \ Sorig with high probability. Therefore, our algorithm is likely to produce a small dvol(Salg, Sorig).

In conclusion, the results are comparable to those obtained from the spectral clustering algorithm.

## 5. Experimental Results

Evaluation Method For all experiments, we use the normalized discrepancy dnorm, as defined in (1), to assess precision. Remember that when the normalized discrepancy is small, the outcome closely resembles that of the original spectral clustering algorithm, indicating a high-quality clustering result. The reported values represent the average of 10 experiments, which we consider sufficient, as the variance in precision across each set of experiments is typically small. Input Graphs We conduct most of our experiments on graphs generated using the stochastic block model (SBM) (Holland et al., 1983). This model is chosen because it ensures that the generated graphs are well-clustered and consist of exactly two clusters. Furthermore, SBM has been widely employed in prior studies to analyze spectral clustering under local differential privacy (Hehir et al., 2022). In this model, the set of n nodes is divided into two clusters of sizes n1 and n2, where n1 + n2 = n. Two nodes within the same cluster are connected with probability p, while nodes from different clusters are connected with probability q. While in most cases p ≫ q, this paper also considers the scenario where *q > p*.

Parameters Unless otherwise specified, we set n =
10, 000, n1 = n2 = 5, 000, p = 0.3, q = 0.2, the clipping factor c = 10, and the privacy budget ϵ = 1. The value of n is chosen to be 10, 000 due to the memory requirements of the benchmark algorithm, randomized response, which requires Ω(n 2) bits to store the entire graph for spectral clustering calculations. We believe that graphs of this size are sufficient to effectively demonstrate the empirical properties of our algorithm. Given the constraints of our local computational environment, handling larger graphs is not feasible. We select p = 0.3 and q = 0.2 because these values are close enough to highlight the precision of our algorithm in distinguishing clusters. We set the clipping factor c = 10, as it is the integer closest to log n log g for well-clustered graphs generated using the stochastic block model. Recall that, when c = log n log g, the clipping is applied only with small probability. The privacy budget is set to ϵ = 1 as it is a standard value commonly used in experiments of other local differential privacy algorithms (Hillebrand et al., 2023). Benchmark To the best of our knowledge, only one graph clustering algorithm under local differential privacy has been explored in the literature. This algorithm employs the spectral clustering method on graph processed using randomized response (Hehir et al., 2022). Therefore, we select this algorithm as the benchmark for our study.

330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384

(c) Comparison across different graph density when ϵ = 1
(d) Comparison across different graph density when ϵ *= 1.*5
Comparison across Different Privacy Budget As illustrated in Figure 1(a), our algorithm consistently outperforms the benchmark algorithm across all privacy budget values (ϵ). The improvement is especially notable in the range 0.8 ≤ ϵ ≤ 2, where the benchmark algorithm yields nearly random results, with a normalized discrepancy close to 1, while our algorithm produces results almost identical to the non-private spectral clustering. Comparison across Different Graph Size Figure 1(b) presents a comparison with the benchmark algorithm across varying numbers of nodes (n). From the figure, we observe that while our algorithm performs poorly for small n, it achieves results identical to non-private spectral clustering when n becomes sufficiently large. This aligns with our theoretical findings, which indicate that the noise introduced by our algorithm becomes negligible as the input graph size increases. The plot also reveals that the randomized response-based algorithm performs well only when the input graph size is small. This observation aligns with the theoretical findings of previous work (Mukherjee & Suppakitpaisarn, 2023), which state that the required privacy budget must exceed Θ(log n). Consequently, larger values of n demand a higher privacy budget in the prior approach. In summary, our algorithm demonstrates greater precision for larger n, whereas the previous method performs better on very small graphs. It is worth noting that, for the plot in Figure 1(b) alone, we conducted the experiment on Google Colaboratory. This was necessary because our local computing environment lacked the storage capacity for the randomized response results for graphs of that size. However, we have verified that the precision results remain consistent across different computational environments. Comparison across Different Edge Density In Figures 1(c) and 1(d), we explore the impact of graph density by varying the probabilities p and q. The experiments are conducted for all pairs (p, q) ∈ {0.05, 0.1*, . . . ,* 0.95}
2and for ϵ ∈ {1, 1.5}. Due to the large number of experiments, the graph size is reduced to 1000 for this analysis. The results show that when p > 0.35, our algorithm consistently outperforms the randomized response-based method, achieving a smaller normalized discrepancy in these cases.

When p ≤ 0.35, there are instances where our algorithm performs worse than the benchmark algorithm. This occurs because the estimated minimum degree, δ, is relatively small in these cases, resulting in a larger amount of noise added in Algorithm 1. While we have theoretically shown that our algorithm can produce results comparable to original spectral clustering when δ ≥
√n log4n (where n is the number of nodes), this analysis is valid only for large n and

(a) Comparison across different privacy budget
(b) Comparison across different graph sizes
385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 does not extend to cases where n = 1000. On the other hand, as shown in (Mohamed et al., 2022), the randomized response-based algorithm performs well when q ≤ p and p is small. Consequently, in these scenarios, the randomized response method outperforms our algorithm. We observe that when *q > p*, the results of both algorithms deviate from those of the original spectral clustering algorithm. This outcome arises because the input graphs are not well-clustered, leading to poor performance from both the original spectral clustering method and the two algorithms in these cases. Computation Time Although our algorithm is designed to be executed in a distributed manner in practice, we were unable to afford the necessary computation units for handling 10,000 nodes in this experiment. As a result, all computations were performed on our server, making the computation environment different from practical scenarios. Consequently, a direct comparison of the computation times between our algorithm and the benchmark algorithm is not feasible. However, even with all computations performed on the server, the computation time for graphs with 20,000 nodes is less than 10 seconds for both algorithms, and for graphs with 1,000,000 nodes, our algorithm completes in under 1 minute. Therefore, we consider computation time to be a manageable factor for both algorithms.

Results on Reddit Graph We also conduct an experiment on the real graph called Reddit graph (Hamilton et al., 2017).

We chose this graph because it is one of the largest publicly available social networks and features a clear cluster structure. To ensure that the noise added in our algorithm is not too large, we calculate a 100-core and 500-core decomposition of the graph before giving it as an input of both algorithms. The 100-core decomposition result contains 154,525 nodes and 108,024,958 edges, while the 500-core decomposition result contains 44,586 nodes, 54,984,204 edges. We were unable to run the randomized response algorithm on this large network, even with the A100 GPU (40GB of GPU RAM) and 83.5GB of system RAM. As a result, we could not directly compare our algorithm with the previous one. Since the Reddit graph contains more than two clusters, we observed that λ3(B) + 1 is very close to λ2(B) + 1, and the value of g (defined in Section 2.5) must be set as low as 0.005. Consequently, the number of iterations required by the algorithm, calculated as 2 log n/ log g, increases significantly to approximately 14,000. Given that the noise size is dependent on the number of iterations, this large iteration count renders the noise size unmanageable. To address this, we limited the number of iterations to 50 for this experiment.

(a) 100-core decomposition (b) 500-core decomposition
Our results for these graphs are presented in Figure 2. For graphs generated using the SBM, we observe that when an algorithm fails to classify the graph in a particular setting, the normalized discrepancy exceeds 0.99. In contrast, our normalized discrepancy remains below 0.99 when the privacy budget is at least 4 for the 100-core decomposition and at least 1 for the 500-core decomposition. This demonstrates that our algorithm can produce meaningful clustering results under these conditions. While the normalized discrepancy rapidly converges to 0 in graphs generated by the model, it does not converge to 0 in Figure 2. We attribute this to the Reddit graph containing more than two clusters, which results in a significant number of nodes vi with small |νi| (as discussed in Assumption 3 in Section 2.5). Consequently, our algorithm is unable to classify these nodes correctly. Further Experiments In Appendix E, we present experiments to validate the positive impact of the differences discussed in Section 3.

## 6. Conclusion And Future Work

In this paper, we propose a locally differentially private algorithm for graph clustering that is theoretically proven to work on general graphs. Unlike most prior works, which focus on non-interactive algorithms based on randomized response, we introduce an interactive algorithm leveraging power iterative clustering. Our approach demonstrates both theoretical and experimental improvements over previous methods. By this work, we believe that interactive algorithms have the potential to become a key tool for addressing graph problems under local differential privacy. Although our algorithm is applicable to sparse graphs, our theoretical guarantees currently hold only for dense graphs. Extending the theory to sparse graphs requires an additional condition: for any eigenvector vi = [vi,1, . . . , vi,n]
⊺
, the ratio maxj,j′
vi,j vi,j′ 
must be small. This property, known as delocalization, has been studied in several works, such as (Rudelson & Vershynin, 2016). We plan to investigate the potential of incorporating this property into our analysis.

## Impact Statement References

Abbe, E., Fan, J., Wang, K., and Zhong, Y. Entrywise eigenvector analysis of random matrices with low expected rank. *Annals of Statistics*, 48(3):1452, 2020.

Alon, N. Eigenvalues and expanders. *Combinatorica*, 6(2):
83–96, 1986.

Apple's Differential Privacy Team. Learning with privacy at scale. *Apple Machine Learning Research*, 2017.

Balakrishnan, S., Xu, M., Krishnamurthy, A., and Singh, A. Noise thresholds for spectral clustering. Advances in Neural Information Processing Systems, 24, 2011.

Betzer, L., Suppakitpaisarn, V., and Hillebrand, Q. Publishing number of walks and Katz centrality under local differential privacy. In *UAI 2024*, 2024.

Boutsidis, C., Kambadur, P., and Gittens, A. Spectral clustering via the power method - provably. In *ICML 2015*,
pp. 40–48, 2015.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476

477

478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494

Dwork, C. Differential privacy: A survey of results. In TAMC 2008, pp. 1–19, 2008.

Erlingsson, U., Pihur, V., and Korolova, A. RAPPOR:
Randomized aggregatable privacy-preserving ordinal response. In *SIGSAC 2014*, pp. 1054–1067, 2014.

Fu, N., Ni, W., Zhang, S., Hou, L., and Zhang, D. GC-
NLDP: A graph clustering algorithm with local differential privacy. *Computers & Security*, 124:102967, 2023.

He, W., Fichtenberger, H., and Peng, P. A differentially private clustering algorithm for well-clustered graphs. In ICLR 2024, 2024.

Hehir, J., Slavkovic, A., and Niu, X. Consistent spectral clustering of network block models under local differential privacy. *Journal of Privacy and Confidentiality*, 12 (2), 2022.

Henzinger, M., Sricharan, A., and Zhu, L. Tighter bounds for local differentially private core decomposition and densest subgraph. *arXiv preprint arXiv:2402.18020*, 2024.

Hillebrand, Q., Suppakitpaisarn, V., and Shibuya, T. Unbiased locally private estimator for polynomials of laplacian variables. In *SIGKDD 2023*, pp. 741–751, 2023.

Hillebrand, Q., Suppakitpaisarn, V., and Shibuya, T. Cycle counting under local differential privacy for degeneracybounded graphs. *arXiv preprint arXiv:2409.16688*, 2024.

Holland, P. W., Laskey, K. B., and Leinhardt, S. Stochastic blockmodels: First steps. *Social networks*, 5(2):109–137, 1983.

Imola, J., Murakami, T., and Chaudhuri, K. Locally differentially private analysis of graph statistics. In USENIX Security 2021, pp. 983–1000, 2021.

Imola, J., Murakami, T., and Chaudhuri, K. Communicationefficient triangle counting under local differential privacy.

In *USENIX Security 2022*, pp. 537–554, 2022.

Imola, J., Epasto, A., Mahdian, M., Cohen-Addad, V., and Mirrokni, V. Differentially private hierarchical clustering with provable approximation guarantees. In *ICML 2023*, pp. 14353–14375, 2023.

Ji, T., Luo, C., Guo, Y., Wang, Q., Yu, L., and Li, P. Community detection in online social networks: A differentially private and parsimonious approach. IEEE Transactions on Computational Social Systems, 7(1):151–163, 2020.

Kasiviswanathan, S. P., Lee, H. K., Nissim, K., Raskhodnikova, S., and Smith, A. What can we learn privately? SIAM Journal on Computing, 40(3):793–826, 2011.

Li, J. and Tkocz, T. Tail bounds for sums of independent two-sided exponential random variables. In High Dimensional Probability IX: The Ethereal Volume, pp. 143–154. Springer, 2023.

Lin, F. and Cohen, W. W. Power iteration clustering. In ICML 2010, pp. 655–662, 2010.

Mangat, N. S. An improved randomized response strategy. Journal of the Royal Statistical Society: Series B (Methodological), 56(1):93–95, 1994.

Mohamed, M. S., Nguyen, D., Vullikanti, A., and Tandon, R.

Differentially private community detection for stochastic block models. In *ICML 2022*, pp. 15858–15894, 2022.

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. Epasto, A., Liu, Q. C., Mukherjee, T., and Zhou, F. The power of graph sparsification in the continual release model. *arXiv preprint arXiv:2407.17619*, 2024.

Hamilton, W., Ying, Z., and Leskovec, J. Inductive representation learning on large graphs. Advances in neural information processing systems, 30, 2017.

Chen, H., Cohen-Addad, V., d'Orsi, T., Epasto, A., Imola, J., Steurer, D., and Tiegel, S. Private estimation algorithms for stochastic block models and mixture models. Advances in Neural Information Processing Systems, 36: 68134–68183, 2023.

Mohar, B. Isoperimetric numbers of graphs. *Journal of* Combinatorial Theory, Series B, 47(3):274–291, 1989.

Mukherjee, S. and Suppakitpaisarn, V. Robustness for spectral clustering of general graphs under local differential privacy. *arXiv preprint arXiv:2309.06867*, 2023.

Ng, A., Jordan, M., and Weiss, Y. On spectral clustering:
Analysis and an algorithm. *NIPS 2001*, 14, 2001.

Peng, R., Sun, H., and Zanetti, L. Partitioning well-clustered graphs: Spectral clustering works! In *COLT 2015*, pp. 1423–1455, 2015.

Qin, Z., Yu, T., Yang, Y., Khalil, I., Xiao, X., and Ren, K. Generating synthetic decentralized social graphs with local differential privacy. In *CCS 2017*, pp. 425–438, 2017.

Rudelson, M. and Vershynin, R. No-gaps delocalization for general random matrices. Geometric and Functional Analysis, 26(6):1716–1776, 2016.

Shi, J. and Malik, J. Normalized cuts and image segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(8):888–905, 2000.

Von Luxburg, U. A tutorial on spectral clustering. *Statistics* and Computing, 17:395–416, 2007.

Wang, Y., Wu, X., and Hu, D. Using randomized response for differential privacy preserving data collection. In EDBT/ICDT Workshops, 2016.

Warner, S. L. Randomized response: A survey technique for eliminating evasive answer bias. Journal of the American Statistical Association, 60(309):63–69, 1965.

Ye, Q., Hu, H., Au, M. H., Meng, X., and Xiao, X. Towards locally differentially private generic graph metric estimation. In *ICDE 2020*, pp. 1922–1925, 2020.

Zhang, H., Latif, S., Bassily, R., and Rountev, A.

Differentially-private control-flow node coverage for software usage analysis. In *USENIX Security 2020*, 2020.

Zhu, T., Li, G., Zhou, W., and Philip, S. Y. Differentially private data publishing and analysis: A survey. IEEE Transactions on Knowledge and Data Engineering, 29
(8):1619–1638, 2017.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549

## A. Eigenvector Components

In this section, we analyze the Laplacian matrix of the graph G, defined as L = I − B. For each i, let λi(L) = 1 − λi(B). It follows that λi(L) is an eigenvalue of L, and the eigenvalues are ordered as λ1(L) *≤ · · · ≤* λn(L). Moreover, the eigenvector vi(B) associated with λi(B) is also an eigenvector of L corresponding to λi(L). For simplicity, throughout this section, we denote λi(L) by λi and vi(B) by vi = [vi,1, . . . , vi,n]
⊺
.

Proposition A.1. Assume that
(i) Let V (G) = A ⊔ B be a bipartition of G with v2,j ≥ 0 for vj ∈ A, v2,j ≤ 0 for vj ∈ B*. Then, the cut* (A, B) has conductance ϕ satisfying ϕ/λ3 ≤ 0.12. (ii) Let ϵ and c be a constant. For a subset S ⊆ V and vertex vj ∈ S, let us call vj to be (ϵ, S)*-average if* dj ≥ ϵd(S), where d(S) = Vol(S)/|S| is the average degree of the vertices in S. Let Aϵ and Bϵ denote the set of (ϵ, A)*-average nodes* of A and (ϵ, B)-average nodes of B, respectively. Assume that |Aϵ| ≥ c|A| and |Bϵ| ≥ c|B|.

Then,

$$|v_{2,j}|\geq\left\{\begin{array}{l l}{{\frac{\epsilon^{1/2}c}{4}\cdot\sqrt{\frac{d_{j}}{n d(A)}}-2\sqrt{\frac{\phi}{\lambda_{3}}},}}&{{v\in A}}\\ {{\frac{\epsilon^{1/2}c}{4}\cdot\sqrt{\frac{d_{j}}{n d(B)}}-2\sqrt{\frac{\phi}{\lambda_{3}}},}}&{{v\in B}}\end{array}\right.$$
$$(4)$$

Consequently, for vj ∈ Aϵ ∪ Bϵ, which is at least c fraction of the vertices of G*, we have*

$$|v_{2,j}|\geq{\frac{\epsilon c}{4}}\cdot{\frac{1}{\sqrt{n}}}-2{\sqrt{\frac{\phi}{\lambda_{3}}}}.$$
$$(S)$$
. (5)
Proof. Let us define the normalized indicator variables

$$g_{A}(j)=\left\{\begin{array}{c c}{{\frac{d_{j}^{1/2}}{\mathrm{Vol}(A)^{1/2}},}}&{{v_{j}\in A}}\\ {{0,}}&{{v_{j}\in B}}\end{array}\right.\;\;\mathrm{and}\;g_{B}(j)=\left\{\begin{array}{c c}{{0,}}&{{v_{j}\in A}}\\ {{\frac{d_{j}^{1/2}}{\mathrm{Vol}(B)^{1/2}},}}&{{v_{j}\in B}}\end{array}\right.$$
.
Let the vector gA = [gA(1)*, . . . , g*A(n)]⊺, gB = [gB(1)*, . . . , g*B(n)]⊺, and, for any vector v, the Rayleign quotient of v = [x1*, . . . , x*n]
⊺
, denoted by R(v), is v
⊺Lv v⊺v
. We show the following regarding the Rayleigh quotients RL(gA) and RL(gB).

550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 Claim A.2. ϕ ≥ max{RL(gA), RL(gB)}.

Proof of Claim *A.2.* Observe that the Rayleigh quotient of L satisfies,

$${\mathcal{R}}_{L}(\mathbf{v})={\frac{\mathbf{v}^{\mathsf{T}}L\mathbf{v}}{\mathbf{v}^{\mathsf{T}}\mathbf{v}}}=1-{\frac{\sum_{i=1}^{n}\sum_{j=1}^{n}{\frac{a_{i j}}{d_{i}}}x_{i}x_{j}}{\sum_{i=1}^{n}x_{i}^{2}}}=1-{\frac{\sum_{\{i,j\}\in E}\left({\frac{1}{d_{i}}}+{\frac{1}{d_{j}}}\right)x_{i}x_{j}}{\sum_{i=1}^{n}x_{i}^{2}}}.$$
. (6)
Since ∥gA∥
2 = 1, we have

$$\mathcal{R}_{L}(g_{A})=1-\sum_{\{i,j\}\in E}\left(\frac{1}{d_{i}}+\frac{1}{d_{j}}\right)g_{A}(i)g_{A}(j)=1-\sum_{\{i,j\}\in E(A)}\left(\frac{1}{d_{i}}+\frac{1}{d_{j}}\right)\cdot\frac{\sqrt{d_{i}d_{j}}}{\text{Vol}(A)}$$
$$\begin{array}{r l}{\{i,j\}_{\in E(A)}\setminus a_{i}\quad}&{{}a_{j}\,\}\quad\mathrm{Vol}(A)\\ {\leq1-\sum_{\{i,j\}_{\in E(A)}}{\frac{2}{\mathrm{Vol}(A)}}={\frac{\mathrm{Vol}(A)-2e(A)}{\mathrm{Vol}(A)}}}\\ {={\frac{e(A,B)}{\mathrm{Vol}(A)}}\leq\phi.}\end{array}$$

$$(6)$$

Similarly, we have RL(gB) ≤ ϕ, completing the proof of Claim A.2. ■
For the rest of the proof, let us denote t := ϕ/λ3. Recall that v1 = [1/
√n, . . . , 1/
√n]
⊺
. We will make use of the following lemmas from the structure theorem (Theorem 3.1) of (Peng et al., 2015), but with a different notation and error estimates.

Lemma A.3. Let gˆA, gˆB be the projections of gA, gB onto the space spanned by the first two eigenvectors {v1, v2} of L.
Then,
$$\operatorname*{max}\{\|{\hat{g}}_{A}-g_{A}\|^{2},\|{\hat{g}}_{B}-g_{B}\|^{2}\}\leq t.$$
2} ≤ t. (7)
≤ 2tη2
$${\mathcal{R}}_{L}(g_{A})=\sum_{i=1}^{n}\alpha_{i}\mathbf{v}_{i}^{\mathsf{T}}\cdot L\cdot\sum_{i=1}^{n}\alpha_{i}\mathbf{v}_{i}=\sum_{i=1}^{n}\alpha_{i}^{2}\mathbf{v}_{i}^{\mathsf{T}}L\mathbf{v}_{i}=\sum_{i=1}^{n}\alpha_{i}^{2}\lambda_{i}.$$

$$(7)$$
But λ1 = 0, leading us to RL(gA) ≥ α 22λ2 + (α 23 + *· · ·* + α 2n
)λ3 = α 22λ2 + ∥gˆA − gA∥
2λ3 ≥ ∥gˆA − gA∥
2λ3. Thus,
∥gˆA − gA∥
2 ≤ RL(gA)/λ3 ≤ ϕ/λ3 by Claim A.2. The proof for ∥gˆB − gB∥
2is exactly analogous. ■
605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 One of the main ideas used in (Peng et al., 2015) is that if gˆA and gˆB are independent, then Span({v1, v2}) = Span({gˆA, gˆB}), implying that v1 and v2 can be written as linear combinations of the projected indicator vectors gˆA and gˆB, say v2 = η1gˆA + η2gˆB, implying that ∥v2 − η1gA − η2gB∥ is small.

Let us now continue with the argument.

Claim A.4. gˆA and gˆB are linearly independent. Proof of Claim A.4. By Lemma A.3, we have ∥gˆA∥
2 ≥ 1 − t and ∥gˆB∥
2 ≥ 1 − t. On the other hand, Since t ≤ 0.12 <
1 2
(2 −
√3), we have t + 2√t < 1 − t, implying |⟨gˆA, gˆB⟩| < ∥gˆA∥∥gˆB∥. As this implies a strict inequality in the Cauchy-Schwarz inequality, we have gˆA ∦ gˆB. ■
Moreover, since t ≤ 0.12, we have Moreover, by the triangle inequality and Cauchy-Schwarz inequality,

$$\eta^{2}\leq{\frac{1}{1-2t-2{\sqrt{t}}}}<16.$$ hy-Schwarz inequality,
< 16. (9)
$$\|\mathbf{v}_{2}-\mathbf{v}_{2}^{\prime}\|^{2}$$

′
2 = ∥η1(ˆgA − gA) + η2(ˆgB − gB)∥
2

$$\begin{array}{l}{{=\left|\eta_{1}(g_{A}-g_{A})+\eta_{2}(g_{B}-g_{B})\right|}}\\ {{\leq\left(|\eta_{1}|{\sqrt{t}}+|\eta_{2}|{\sqrt{t}}\right)^{2}}}\end{array}$$
$$(9)$$

$$(10)^{\frac{1}{2}}$$
$$(11)^{\frac{1}{2}}$$

Therefore, we have that

$2\eta\langle\mathbf{v}_{2},\frac{1}{\eta}\mathbf{v}_{2}^{\prime}\rangle=2\langle\mathbf{v}_{2},\mathbf{v}_{2}^{\prime}\rangle=\|\mathbf{v}_{2}\|^{2}+\|\mathbf{v}_{2}^{\prime}\|^{2}-\|\mathbf{v}_{2}-\mathbf{v}_{2}^{\prime}\|^{2}\geq1+\eta^{2}-2t\eta^{2}$,
leading us to
$$\langle{\bf v}_{2},{\frac{1}{\eta}}{\bf v}_{2}^{\prime}\rangle\geq{\frac{1+\eta^{2}}{2\eta}}-\eta t\geq1-\eta t.$$
Basically, this means that v2 is closely aligned with the normalized vector 1η v
′2. We now show a lemma that relates the components of two such vectors.

$$({\mathfrak{s}})$$

$$1=\|{\bf v}_{2}\|^{2}\geq\eta_{1}^{2}\|\hat{g}_{A}\|^{2}+\eta_{2}^{2}\|\hat{g}_{B}\|^{2}-2|\eta_{1}\eta_{2}\langle\hat{g}_{A},\hat{g}_{B}\rangle|$$ $$\geq\eta_{1}^{2}(1-t)+\eta_{2}^{2}(1-t)-(\eta_{1}^{2}+\eta_{2}^{2})(t+2\sqrt{t})$$ $$=\eta^{2}(1-2t-2\sqrt{t}).$$
$$|\langle\hat{g}_{A},\hat{g}_{B}\rangle|=|\langle\hat{g}_{A}-g_{A}+g_{A},\hat{g}_{B}-g_{B}+g_{B}\rangle|$$ $$\leq|\langle\hat{g}_{A}-g_{A},\hat{g}_{B}-g_{B}\rangle|+|\langle g_{A},\hat{g}_{B}-g_{B}\rangle|+|\langle\hat{g}_{A}-g_{A},g_{B}\rangle|$$ $$\leq\|\hat{g}_{A}-g_{A}\|\|\hat{g}_{B}-g_{B}\|+\|\hat{g}_{A}-g_{A}\|+\|\hat{g}_{B}-g_{B}\|$$ $$\leq t+2\sqrt{t}.$$

As discussed earlier, Claim A.4 implies that there exist η1, η2 ∈ R such that v2 = η1gˆA +η2gˆB. Suppose v
′2 = η1gA +η2gB,
and η = ∥v
′2∥ =pη 21 + η 2 2. Note that, using (8),
Proof of Lemma *A.3.* Let v3*, . . . ,* vn be normalized eigenvectors of λ3*, . . . , λ*n of L. Say gA = α1v1 + *· · ·* + αnvn and gB = β1v1 + *· · ·* + βnvn are representations of gA and gB in the L-eigenbasis. Clearly gˆA = α1v1 + α2v2 and gˆB = β1v1 + β2v2. Then, note that as v
⊺

i vj = 0 for every i ̸= j, Lemma A.5. Let v = [u1*, . . . , u*n]
⊺ be a unit eigenvector of L and v
′ = [u
′1*, . . . , u*′n]
⊺ *be any unit vector with* ⟨v, v
′⟩ ≥
1 − ϵ 2for some ϵ > 0. Then, for each 1 ≤ j ≤ n*, we have*

$$|u_{j}^{\prime}|\leq|u_{j}|+\epsilon.$$

Proof of Lemma *A.5.* Let {v, z1*, . . .* zn−1} be a orthonormal basis of eigenvectors of L, and, for all i, let zi =
[zi,1, . . . , zi,n]
⊺
. Since v
′ = ⟨v, v
′⟩ · v +Pn−1 i=1 
⟨v
′, zi⟩ · zi, this implies that for any 1 ≤ j ≤ n, where the last step follows from the fact that Pn−1 i=1 
⟨v
′, zi⟩
2 + ⟨v
′, v⟩
2 = ∥v
′∥
2 = 1, and Pn−1 i=1 z 2 i,j + u 2 j = 1. ■
Hence, by virtue of Lemma A.5, (9) and (11), we obtain

$$|v_{2,j}|\geq\frac{1}{\eta}|v^{\prime}_{2,j}|-\sqrt{\eta t}=\frac{1}{\eta}|\eta_{1}g_{A}(j)+\eta_{2}g_{B}(j)|-\sqrt{\eta t}=\left\{\begin{array}{ll}\frac{|\eta_{1}|}{\eta}\cdot\frac{d^{1/2}}{\mbox{Vol}(A)^{1/2}}-\sqrt{\eta t},&v_{j}\in A\\ \frac{|\eta_{2}|}{\eta}\cdot\frac{d^{1/2}}{\mbox{Vol}(B)^{1/2}}-\sqrt{\eta t},&v_{j}\in B\end{array}\right.\tag{12}$$

Finally, we need to show that min{|η1|, |η2*|} ≥* ϵ 1/2c. For this part of the proof, we shall use the assumption (ii) of our proposition.

  **Claim A.6.**$|\eta_{1}|\geq c\cdot\left(\frac{\epsilon|A|}{n}\right)^{1/2}$ and $|\eta_{2}|\geq c\cdot\left(\frac{\epsilon|B|}{n}\right)^{1/2}$.  
Proof of Claim *A.6.* Recall from the proof of Lemma A.3 that gˆA = α1v1 + α2v2 and gˆB = β1v1 + β2v2. These equations, along with v2 = η1gˆA + η2gˆB, allow us to solve exactly for η1 and η2 as,

$$|\alpha_{1}|=|\langle g_{A},{\bf v}_{1}\rangle|=\frac{1}{\sqrt{n}}\sum_{v_{j}\in A}\frac{d_{j}^{1/2}}{\mathrm{Vol}(A)^{1/2}}\geq\frac{1}{\sqrt{n}}\sum_{v_{j}\in A_{\epsilon}}\frac{d_{j}^{1/2}}{(|A|d(A))^{1/2}}\,,$$
vj∈Aϵ ≥1 √n · |Aϵ| ·  ϵ |A| 1/2 ≥ c · ϵ|A| n 1/2.
660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 By a similar argument, we have |β1| ≥ c ·
ϵ|B| n 1/2, finishing the proof of Claim A.6. ■
Claim A.6, (12) and η ≤ 4 leads us to, for vj ∈ A,

$$|v_{2,j}|\geq{\frac{c}{4}}\cdot{\frac{\epsilon^{1/2}|A|^{1/2}}{n^{1/2}}}\cdot{\frac{d_{j}^{1/2}}{\mathrm{Vol}(A)^{1/2}}}-2{\sqrt{t}}={\frac{c\epsilon^{1/2}}{4}}\cdot{\sqrt{\frac{d_{j}}{n d(A)}}}-2{\sqrt{t}},$$

which proves the inequality (4) for vj ∈ A. The argument for vj ∈ B is analogous. Finally, the inequality (5) directly follows (4) via the definitions of Aϵ and Bϵ.

First, we note that |α2β1 − α1β2| ≤ (α
$(\alpha_1^2+\alpha_2^2)^{1/2}(\beta_1^2+\beta_2^2)^{1/2}\leq\|g_A\|\|g_B\|$
1/2 ≤ ∥gA∥∥gB∥ = 1, so it suffices to lower bound |α1| and
|β1|. We have that:
$$\eta_{1}=\frac{\beta_{1}}{\alpha_{2}\beta_{1}-\alpha_{1}\beta_{2}}\;\mathrm{and}\;\eta_{2}=\frac{-\alpha_{1}}{\alpha_{2}\beta_{1}-\alpha_{1}\beta_{2}}.$$
$|u^{\prime}_{j}|\leq|\langle\mathbf{v},\mathbf{v}^{\prime}\rangle||u_{j}|+\sum_{i=1}^{n-1}|\langle\mathbf{v}^{\prime},\mathbf{z}_{i}\rangle||z_{i,j}|$  $$\leq|u_{j}|+\left(\sum_{i=1}^{n-1}\langle\mathbf{v}^{\prime},\mathbf{z}_{i}\rangle^{2}\right)^{1/2}\left(\sum_{i=1}^{n-1}z_{i,j}^{2}\right)^{1/2}$$ $$\leq|u_{j}|+\epsilon,$$

## B. Elimination Of The Leading Eigenvector

The following proposition shows that the third term in the calculation at Line 6 of Algorithm 1 eliminates the leading eigenvector of W. Consequently, the leading eigenvector of W˜ becomes the second eigenvector of W.

Proposition B.1. Let W =
1 2
(I+D−1A) be the lazy random walk matrix for a graph on n *vertices. Let* J = (ji,j )1≤i,j≤n be a matrix such that ji,j = 1 for all i, j*. Define* W˜ = W −
1 n J*. Then, for all* i ≥ 1, λi(W˜ ) = λi+1(W) and vn(W˜ ) = v1(W).

Additionally, vn(W˜ ) = v1(W) = [ √
1 n
, . . . , √
1 n
]
⊺ and λn(W˜ ) = 0.

Proof. Recall that v1(W) = h√
$(W)=\left[\frac{1}{\sqrt{n}},\ldots,\frac{1}{\sqrt{n}}\right]^{\mathsf{T}}$ and $\lambda_{1}(W)=1$. We have:
$\uparrow$ . 
$$\tilde{W}\cdot{\bf v}_{1}(W)=W\cdot{\bf v}_{1}(W)-\frac{1}{n}J{\bf v}_{1}(W)={\bf v}_{1}(W)-\left[\frac{1}{\sqrt{n}},\ldots,\frac{1}{\sqrt{n}}\right]^{\dagger}={\bf0}.$$
Therefore, the vector h√
1 n
, . . . , √
1 n i⊺is an eigenvector of W˜ with eigenvalue 0. Since 0 is the minimum eigenvalue of W˜ ,
it follows that vn(W˜ ) = v1(W) and λn(W˜ ) = 0.

Next, let us consider vi(W) for i ≥ 2. Since, vi(W) ⊥ v1(W), we obtain that the sum of all elements in vi(W) is zero. Thus,

$$\tilde{W}{\bf v}_{i}(W)=W{\bf v}_{i}(W)-\frac{1}{n}J{\bf v}_{i}(W)=\lambda_{i}(W){\bf v}_{i}(W).$$

This implies that, for all i ≥ 2, vi(W) is also an eigenvector of W˜ with the same eigenvalue. Consequently, as the largest eigenvalue of W becomes the smallest eigenvalue of W˜ , we have λi−1(W˜ ) = λi(W) and vi−1(W˜ ) = vi(W).

## C. Minimum Degree Estimation

We will now demonstrate that the value of δ computed in Line 2 of Algorithm 1 has a low probability of overestimating the minimum degree of the input graph. This implies that, with large probability, we do not need to modify the input graph in Line 3 of the algorithm.

Proposition C.1. With probability at least 1 − ζ, we have δ < mini di.

Proof. We have δ > mini di only if there is ˜di such that ˜di −
10 ϵ log n 2ζ > di. This implies that the value sampled from the Laplace distribution at Line 1, denoted by liis larger than 10 ϵ log n 2ζ
. By the property of the Laplace distribution, for all i, we have that:

$$\begin{array}{r c l}{{\mathrm{Pr}\left[\vert_{i}>\frac{10}{\epsilon}\log\frac{n}{2\zeta}\right]}}&{{=}}&{{\frac{1}{2}\exp\left(-\frac{10}{\epsilon}\log\frac{n}{2\zeta}/\frac{10}{\epsilon}\right)=\zeta/n.}}\end{array}$$

Then, by the union bound, the probability that there is an index i such that li >
10 ϵ log n 2ζ is not greater than ζ.

Suppose that ζ =
1 n
. In the next proposition, we shown that δ ≥
√n log4n with large probability.

Proposition C.2. Pr[δ < 
√n log4n] ≤
1 2n
.

Proof. In Line 2 of Algorithm 1, Laplacian noise with a scale of 10 ϵ is added. It follows that ˜di < di −
20 ϵ log n if the noise added to diis less than −
20 ϵ
. This event occurs with probability Using the union bound, we have:

$$\Pr\left[\min_{i}\bar{d}_{i}<\min_{i}d_{i}-\frac{20}{\epsilon}\log n\right]\leq\Pr\left[\bar{d}_{i}<d_{i}-\frac{20}{\epsilon}\log n\text{for some}i\right]\leq\frac{1}{2n}.$$
.
715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769

$\square$
14

$${\frac{1}{2}}\exp\left(-{\frac{20/\epsilon\cdot\log n}{10/\epsilon}}\right)={\frac{1}{2n^{2}}}.$$

Given that δ = mini
˜di −
10 ϵ log n 2ζ
, and under the assumption in Section 2.5 that the minimum degree of the network is at least 2
√n log4n, we can bound:

$$\operatorname*{Pr}\left[\delta<{\sqrt{n}}\log^{4}n\right]\leq\operatorname*{Pr}\left[\delta<\operatorname*{min}_{i}d_{i}-{\frac{20}{\epsilon}}\log n-{\frac{10}{\epsilon}}\log{\frac{n}{2\zeta}}\right]\leq{\frac{1}{2n}},$$
$\boxed{\text{L}}$
for sufficiently large n.

## D. Size Of Laplace Noise

770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 In this section, we analyze the effect of adding the Laplace noise at Line 6 of the algorithm. Let the noise added by the node i at the iteration t is y
(t)
i. Define the vector y
(t)as [y
(t)
1
, . . . , y
(t)
n ]
⊺
. Also, for all *i, t*, let e
(t)
ibe a real number such that y
(t) = e
(t)
1 v1(W˜ ) + *· · ·* + e
(t)
n vn(W˜ ).

Let the initial vector denoted by x
(0) = c1v1(W˜ ) + *· · ·* + cnvn(W˜ ), and the final vector is denoted by x
(T). We obtain the following lemma by the notation.

Lemma D.1. Let c˜1, . . . , c˜n *be numbers such that* x
(T) = ˜c1v1(W˜ ) + *· · ·* + ˜cnvn(W˜ )*. We obtain that* c˜i = ciλi(W˜ )
T +
e
(1)
i λi(W˜ )
T −1 + *· · ·* + e
(T)
i.

Proof. To prove the statement, let c
(t)
i = ciλi(W˜ )
t + e
(1)
i λi(W˜ )
t−1 + *· · ·* + e
(t)
i. We proceed by induction on t to show that, for all t ≥ 0, x
(t) = c
(t)
1 v1(W˜ ) + *· · ·* + c
(t)
n vn(W˜ ). When t = 0, c
(0)
i = ci, so the statement holds directly by the definition of the notation. Assume the statement is true for t − 1; that is, x
(t−1) = c
(t−1)
1 v1(W˜ ) + *· · ·* + c
(t−1)
n vn(W˜ ).

Then, for x
(t), we have x
(t) = W˜ · x
(t−1) + y
(t).

Expanding this using the induction hypothesis gives x
(t) = (c
(t−1)
1 λ1(W˜ ) + e
(t)
1)v1(W˜ ) + *· · ·* + (c
(t−1)
n λn(W˜ ) + e
(t)
n)vn(W˜ ).

Thus, we obtain x
(t) = c
(t)
1 v1(W˜ ) + *· · ·* + c
(t)
n vn(W˜ ), completing the induction.

From now, let vi(W˜ ) = [vi,1, . . . , vi,n]
⊺
. We will now calculate the size of each variable. Recall from Line 4 of Algorithm 1 that x
(0)
iis sampled from the Gaussian distribution with expected value 0 and standard deviation 1.

Lemma D.2. For each i, the variable ciis a normal random variable with mean 0 and standard deviation 1*. Furthermore,*
for i ̸= j, ci*is independent to* cj .

and
$$\mathrm{Var}(c_{i})=v_{i,1}^{2}\,\mathrm{Var}[x_{1}^{(0)}]+\cdots+v_{i,n}^{2}\,\mathrm{Var}[x_{n}^{(0)}]=v_{i,1}^{2}+\cdots v_{i,n}^{2}=1.$$
Since vi(W˜ ) is orthogonal to vj (W˜ ) for i ̸= j, the coefficients ci and cj , which are the dot products of x
(0) with vi(W˜ )
and vj (W˜ ) respectively, are independent of each other.

Next, we give analyze the variables e
(t)
i. We observe that, although the random variable is a linear combination of the Laplace variables y
(t)
j, it is not itself Laplace-distributed.

Lemma D.3. For all t and i*, we have* E[e
(t) i
] = 0. Furthermore, for all t *and all* i ̸= j, Cov(e
(t)
i, e
(t) j
) = 0.

Proof. Since, for all i, the eigenvector vi(W˜ ) is a unit vector and ci = ⟨x
(0), vi(W˜ )⟩, we have that ci =Pj vi,jx
(0)
j.

Because ciis a linear combination of normal random variables, ciis a normal random variable. Furthermore,

$\square$
$\mathbf{M}$
E[ci] = vi,1E[x
(0)
1] + · · · + vi,nE[x
(0)
n] = 0, 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879

Var(γ
(h,t) i) = X j λj (W˜ ) 2h· v 2 j,i · Var(e (t) j) = 2C 2 t ·X j λj (W˜ ) 2h· v 2 j,i ≤ 2C 2 t · λ1(W˜ ) 2h·X j v 2 j,i = 2C 2 t · λ1(W˜ ) 2h.
Since e
(t)
jis a linear combination of Laplace variables, γ
(h,t)
iis also a linear combination of the Laplace variable y
(t)
j. Let a1*, . . . ,* an be real numbers such that γ
(h,t)
i =Pj ajy
(t)
j. We obtain that Var(γ
(h,t)
i) = 2 · C
2 t Pj a 2 j ≤ 2C
2 t
· λ1(W˜ )
2h,

Cov(e (t) i, e (t) j) = E  X i ′,j′ vi,i′y (t) i ′ vj,j′y (t) j ′   =X i ′,j′ vi,i′vj,j′E[y (t) i ′ y (t) j ′ ] = X k vi,kvj,kE[(y (t) k ) 2] = E[(y (t) 1) 2] ·X k vi,kvj,k = 0.
$\square$
Let Ct represent the scale of the Laplace noise in Line 6 during the t-th iteration of Algorithm 1. By definition, Var(y
(t)
i) =
2C
2 tfor every i. The variance of e
(t)
iis discussed in the following lemma. Our proof draws on ideas from the paper (Li &
Tkocz, 2023). Lemma D.4. For all i and t*, the variance of* e
(t)
iis 2 · C
2 t*. Furthermore,* Pr[e
(t)
i ≥
√2Ct log n] ≤
e n
.

Proof. Based on the argument in the proof of Lemma D.3, we have e
(t)
i =Pjvi,jy
(t)
j. Consequently, Var(e
(t)
i) =
Pj v 2 i,jVar(y
(t)
j
) for all i and t. Since y
(t)
jis a Laplace variable with scale Ct and each vi(W˜ ) is a unit vector, it follows that Var(e
(t)
j) = 2 · C
2 t.

Using the Chernoff bound and the moment generating function of the Laplacian distribution, we obtain that

$$\operatorname*{Pr}[e_{i}^{(t)}\geq{\sqrt{2}}C_{t}\log n]\leq e^{-\log n}\cdot\mathbb{E}\left[\exp\left({\frac{\sum_{j}v_{i,j}y_{j}^{(t)}}{\sqrt{2}C_{t}}}\right)\right]=$$
=

 $:\dfrac{1}{n}\prod\limits_j\mathbb{E}\left[\exp\left(\dfrac{v_{i,j}y_j^{(t)}}{\sqrt{2}C_t}\right)\right]$  $:\dfrac{1}{n}\prod\limits_j\mathbb{E}\left[\exp\left(\dfrac{v_{i,j}}{\sqrt{2}}\cdot\text{Lap}\left(0,1\right)\right)\right]$  $:\dfrac{1}{n}\cdot\text{Lap}\left(1,\dots,1\right)$

=

=
$$\frac{1}{n}\prod_{j}\frac{1}{1-\frac{1}{2}v_{i,j}^{2}}$$
≤
$${\frac{1}{n}}\exp\sum_{j}v_{i,j}^{2}={\frac{e}{n}}.$$

Let h be a positive integer. We discuss the property of the vector W˜ hy
(t):= [γ
(h,t)
1*, . . . , γ*
(h,t)
n ]
⊺
in the next lemma.

Lemma D.5. For all i, h, t*, the probability that* |γ
(h,t)
i| ≥ 3
√2 · λ1(W˜ )
h· Ct · log n is at most 2e/n3.

Proof. From the definition of γ
(h,t)
iand the argument in Lemma D.1, we find that γ
(h,t)
i =Pjλj (W˜ )
h· vj,i · e
(t)
j.

According to Lemma D.3, Cov(e
(t) j
, e
(t) j
′ ) = 0 for j ̸= j
′. Therefore, by Lemma D.4, 16 Next, for i ̸= j, we examine the covariance between e
(t)
iand e
(t)
j, denoted as Cov(e
(t)
i, e
(t) j
). Since E(e
(t)
i) = E(e
(t) j
) = 0,
{y
(t)
1*, . . . , y*
(t)
n } are independent with mean 0, and viis orthogonal to vj , we have:
Proof. According to Line 6 of Algorithm 1, for all t and i ̸= j, the variables y
(t)
iand y
(t)
jare independent, with E(y
(t)
i) =
E(y
(t)
j) = 0 and Var(y
(t)
i) = Var(y
(t)
j). The variable e
(t)
iis defined as the dot product between vi(W˜ ) and y
(t). Specifically, if vi(W˜ ) = [vi,1, . . . , vi,n]
⊺
, then e
(t)
i =Pj vi,jy
(t)
j. Consequently, E(e
(t)
i) = Pj vi,jE[y
(t)
j] = 0.

and Pj a 2j ≤ λ1(W˜ )
2h. Using the Chernoff bound, we obtain that

$$\mathrm{Pr}\left[C_{t}\leq{\frac{10}{9\epsilon}}\cdot{\frac{\lambda_{1}({\tilde{W}})^{t-1}}{\sqrt{n}\log^{2}n}}\,{\mathrm{for~all~}}1\leq t\leq T\,\mid{\mathcal{E}}_{\delta}\right]\geq1-{\frac{8e T^{2}}{n^{2}}}.$$

Proof. Since x
(0)
iis drawn from a Gaussian distribution with mean 0 and standard deviation 1, it follows from the properties of a normal random variable that Pr[|x
(0)
i| ≥ log n · log g] ≤1 n3 . By applying the union bound, we then have Pr[maxi|x
(0)
i| ≥ log n · log g] ≤1 n2 .

We will prove this lemma by induction on the number of iterations t. For t = 1, recall from Line 6 of the algorithm that the noise y
(t)
iis drawn from a Laplace distribution with scale parameter 5·T
9·ϵ
·
maxi |x
(t−1)
i| δ, where ϵ is the privacy budget and δ is the minimum degree of the input graph. In the event Eδ, the variable δ ≥
√n log4n. Recall that we set T = 2log n log g in our algorithm. Consequently, the noise scale in the first iteration is larger than 10 9ϵ log n log g
·
log
√ n·log g n log4 n 
=10 9ϵ·
√n log2 n with probability not larger than 1/n2 when n is large enough. Next, assume that, in the event Eδ, with probability not smaller than 1 −
2e·(2t−2)2 n2 , for all t
′ < t, the noise (denoted by y
(t
′)
i) is sampled from a Laplace distribution with a scale no more than 10 9ϵ
·
λ1(W˜ )
t
′−1
√n log2 n
. From our previous calculations, it follows that x
(t) = W˜ tx
(0) + W˜ t−1y
(1) + *· · ·* + y
(t). Let W˜ tx
(0) = [x
(t) 1
, . . . , x
(t)
n ]
⊺
and, for all t
′ ≤ t, W˜ t−t
′y
(t
′) = [y
(t,t′)
1*, . . . ,* y
(t,t′)
n ]
⊺
. The value of maxi|x
(t−1)
i|, which decides the noise scale of y
(t), is equal to maxi

x
(t−1)
i +
tP−1 t
′=1 y
(t−1,t′)
i

.

Let us first consider the vector [x
(t−1)
1*, . . . ,* x
(t−1)
n ]
⊺
. Recall that vi(W˜ ) = [vi,1, . . . , vi,n]
⊺
. By the notation, we have x
(t−1)
i =Pj λj (W˜ )
t−1vj,icj .

Since, by Lemma D.2, cj and cj
′ are independent for j ̸= j
′, we obtain:

$$\begin{array}{r c l}{{\mathbb{E}[\mathbf{x}_{i}^{(t-1)}]}}&{{=}}&{{\sum_{j}\lambda_{j}({\tilde{W}})^{t-1}v_{j,i}\cdot\mathbb{E}[c_{j}]=0,}}\end{array}$$
j
$$\begin{array}{r l}{\operatorname{Var}[\mathbf{x}_{i}^{(t-1)}]}&{{}=}\end{array}$$

$$\sum_{j}\lambda_{j}(\hat{W})^{2t-2}v_{j,i}^{2}\mathrm{Var}[c_{j}]\leq\lambda_{1}(\hat{W})^{2t-2}\mathrm{Var}\left[\sum_{j}v_{j,i}c_{j}\right]=\lambda_{1}(\hat{W})^{2t-2}\mathrm{Var}\left[x_{i}^{(0)}\right]=\lambda_{1}(\hat{W})^{2t-2}.$$

880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 In the next lemma, we analyze the size of the noise added in the algorithm. Recall that the variable δ is the noisy minimum degree published at Line 2 of Algorithm 1. In Proposition C.2, we show that δ ≥
√n log4n with probability at least 1 −
1 n
.

We denote the event that δ ≥
√n log4n by Eδ.

Lemma D.6. Recall that Ct is the scale of the noise added at Line 6 of Algorithm *1. Then,*

≤1
n3
E exp Pj aj · Lap(0, 1) √2 · λ1(W˜ ) h  Y j 1 1 −a 2j 2λ1(W˜ ) 2h
$$\underline{{<}}$$
=1
n3
$$\leq$$
≤1
n3
$$\frac{1}{3}\exp\left(\frac{1}{2\lambda_{1}(\tilde{W})^{2h}}\sum_{j}a_{j}^{2}\right)\leq\frac{1}{n^{3}}\exp(1).$$
The lemma statement follows from the fact that the probability distribution of γh,t is symmetric about 0.

$$\Pr[\gamma_{i}^{(h,t)}\geq3\sqrt{2}\cdot\lambda_{1}(\bar{W})^{h}\cdot C_{t}\cdot\log n]\leq e^{-3\log n}\cdot\mathbb{E}\left[\exp\left(\frac{\gamma_{i}^{(h,t)}}{\sqrt{2}\cdot\lambda_{1}(\bar{W})^{h}\cdot C_{t}}\right)\right]$$

17

$$\begin{array}{r c l}{{\left|y_{i}^{(t-1,t^{\prime})}\right|}}&{{\geq}}&{{3\sqrt{2}\cdot\lambda_{1}^{t-t^{\prime}-1}(\bar{W})\cdot\frac{10}{9\epsilon}\cdot\frac{\lambda_{1}^{t^{\prime}-1}(\bar{W})}{\sqrt{n}\log^{2}n}=\frac{30\sqrt{2}}{9\epsilon}\cdot\frac{\lambda_{1}^{t-2}(\bar{W})}{\sqrt{n}\log^{2}n}}}\end{array}$$

with probability at most 2e n3 .

By applying the union bound, we deduce that for all *t, t*′,
935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 with probability at most 2e n2 . By Lemma 4.4 of (Mohar, 1989), we have that λ2(B) ≥ 0 and λ1(W˜ ) ≥
1 2
. When n is sufficiently large, it follows that, for all *t, t*′,
with probability at most 2e n2 . We finally obtain

$$\mathrm{Pr}\left[\sum_{t^{\prime}\leq t-1}\operatorname*{max}_{i}|y_{i}^{(t-1,t^{\prime})}|\geq{\frac{1}{2}}\cdot\lambda_{1}^{t-1}({\bar{W}})\mid{\mathcal{E}},{\mathcal{E}}_{\delta}\right]\leq{\frac{2e t}{n^{2}}}.$$
$$\frac{(2e t+1)}{n^{2}}$$

In the event E and Eδ, maxi|x
(t−1)
i| ≥ log n · log g · λ1(W˜ )
t−1 with probability at most 2et+1 n2 . In the event of E and Eδ, the noise scale at the iteration t, denoted by Ct, is at most 10 9ϵ 2 log n log g log n·log g·λ1(W˜ )
t−1
√n log4 n =
10 9ϵ
·
λ1(W˜ )
t−1
√n log2 n with probability at

$$\Pr\left[\max_{i}\left|\mathsf{x}_{i}^{(t-1)}+\sum_{t^{\prime}\leq t-1}\mathsf{y}_{i}^{(t-1,t^{\prime})}\right|\geq\log n\cdot\log g\cdot\lambda_{1}(\tilde{W})^{t-1}\mid\mathcal{E},\mathcal{E}_{\delta}\right]$$ $$\Pr\left[\max_{i}\left|\mathsf{x}_{i}^{(t-1)}\right|+\sum_{t^{\prime}\leq t-1}\max_{i}\left|\mathsf{y}_{i}^{(t-1,t^{\prime})}\right|\geq\log n\cdot\log g\cdot\lambda_{1}(\tilde{W})^{t-1}\mid\mathcal{E},\mathcal{E}_{\delta}\right]$$

≤ Pr 
$$\Pr\left[\max_{t}\left|x_{t}^{(t-1)}\right|\geq\frac{1}{2}\log n\cdot\log g\cdot\lambda_{1}(\bar{W})^{t-1}\right]+\Pr\left[\sum_{t^{\prime}\leq t-1}\max_{t}\left|y_{t}^{(t-1,t^{\prime})}\right|\geq\frac{1}{2}\lambda_{1}(\bar{W})^{t-1}\mid\mathcal{E},\mathcal{E}_{\delta}\right].$$

$$\operatorname*{max}_{i}|y_{i}^{(t-1,t^{\prime})}|\geq{\frac{\lambda_{1}^{t-1}({\bar{W}})\log g}{4\log n}}\geq{\frac{30{\sqrt{2}}}{9\epsilon}}{\frac{\lambda_{1}^{t-2}({\bar{W}})}{{\sqrt{n}}\log^{2}n}}$$

Because, for all i and t, the variables x
(t−1)
ido not depends on the scale of the Laplacian noise and the event E, we obtain that:

$$\Pr\left[\max_{i}\left|x_{i}^{(t-1)}\right|\geq\log n\cdot\log g\cdot\lambda_{1}(\tilde{W})^{t-1}\mid\mathcal{E},\mathcal{E}_{\delta}\right],$$
$$\operatorname*{max}_{i}|y_{i}^{(t-1,t^{\prime})}|\geq{\frac{30\sqrt{2}}{9\epsilon}}\cdot{\frac{\lambda_{1}^{t-2}({\bar{W}})}{\sqrt{n}\log^{2}n}}$$

Let us reconsider the variable γ
(h,t)
ifrom Lemma D.5. Note that y
(t−1,t′)
i = γ
(t−t
′−1,t′). Let E denote the event that Ct
′ ≤
10 9ϵ
·
λ t
′−1 1(W˜ )
√n log2 n for all t
′ < t. In the event E and Eδ, Lemma D.5 implies that, for all *i, t, t*′,
Also, since x
(t−1)
iis a linear combination of the normal random variable cj , we can conclude that x
(t−1)
iis also normal. By the property of the normal variable, we have Pr h|x
(t−1)
i| ≥ 12 log n · log g · λ1(W˜ )
t−1i≤
1 n3 for all i. By the union bound, Pr hmaxi|x
(t−1)
i| ≥ 12 log n · log g · λ1(W˜ )
t−1i≤
1 n2 .

least 1 −
2et+1 n2 . As a result, This completes the induction step. We can conclude that, for all t ∈ {1*, . . . , T*}, Ct
′ ≤
10 9ϵ
·
λ1(W˜ )
t
′−1
√n log2 n for all t
′ ≤ t with probability at least 1 −
2e(2t)
2 n2 when δ ≥
√n log4n.

990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 We will leverage the previous lemma to demonstrate that the outcome of Algorithm 1 closely aligns with the results obtained through spectral clustering. Recall Lemma D.1 that the final vector x
(T)
i =Pn j=1 c˜jvj,i when c˜j = cjλj (W˜ )
T +
PT
t=1 e
(t)
j λj (W˜ )
T −t.

Theorem D.7. For any node i such that |v1,i| ≥ √
γ n
. For large enough n*, we obtain that* Proof. We first obtain that Recall from Lemma D.2 that ciis a normal random variable with mean 0 and standard deviation 1. We obtain that:
Recall from Lemma D.4 that Pr he
(t)
1 ≥
√2Ct log n i≤
e n
. Let E be the event that maxt Ct ≤
10 9ϵ
·
λ1(W˜ )
t−1
√n log2 n and Eδ be the event that δ ≥
√n log4n. We obtain that Pr he
(t)
1 ≥
10√2 9ϵ λ1(W˜ )
t−1
√n log n |E, Eδ i≤
e n
. Denote PT
t=1 e
(t)
j λ1(W˜ )
T −t by η. By the union bound,

$$\mathrm{Pr}\left[\eta\geq{\frac{\lambda_{1}(\tilde{W})^{T}}{32}}|{\mathcal{E}}\right]\leq\mathrm{Pr}\left[\eta\geq{\frac{10{\sqrt{2}}\cdot T}{9\epsilon}}{\frac{\lambda_{1}(\tilde{W})^{T-1}}{{\sqrt{n}}\log n}}|{\mathcal{E}}\right]\leq{\frac{\epsilon T}{n}},$$

for sufficiently large n. Recall the event Eδ, which is the event when δ ≥
√n log4n. By Lemma D.6, we know that

c1λ1(W˜ ) Tv1,i  >  X T t=1 e T 1 λ1(W˜ ) T −tv1,i +Xn j=2 c˜jvj,i   Pr   t=1 e T 1 λ1(W˜ ) T −tv1,i  +  X n j=2 c˜jvj,i    c1λ1(W˜ ) Tv1,i  >  X T  t=1 e (t) 1 λ1(W˜ ) T −t  ! >  X n j=2 cjvj,iλj (W˜ ) T  +  X n t=1 e (t) j λj (W˜ ) T −tvj,i    .  |v1,i|   |c1λ1(W˜ ) T| −  X T j=2 X T
$\geq$ . 
≥ Pr
$$(13)$$
$$\geq$$
≥ Pr

$$\mathrm{Pr}\left[\left|c_{1}\lambda_{1}(\tilde{W})^{T}\right|\geq{\frac{\lambda_{1}(\tilde{W})^{T}}{16}}\right]>0.95.$$
> 0.95. (13)
$$\mathrm{Pr}\left[C_{t}\geq{\frac{10}{9\epsilon}}\cdot{\frac{\lambda_{1}({\bar{W}})^{t-1}}{\sqrt{n}\log^{2}n}}\;\mathrm{or}\;{\bar{\mathcal{E}}}\mid{\mathcal{E}}_{\delta}\right]$$
$\leq\quad\text{Pr}\,\bigg[C_t\geq0\bigg]$
$$\geq{\frac{10}{9\epsilon}}\cdot{\frac{\lambda_{1}(\tilde{W})^{t-1}}{\sqrt{n}\log^{2}n}}\;\mathrm{and}\;\mathcal{E}\mid\mathcal{E}_{\delta}\Bigg]+\mathrm{Pr}[\tilde{\mathcal{E}}\mid\mathcal{E}_{\delta}]$$ $$\geq{\frac{10}{9\epsilon}}\cdot{\frac{\lambda_{1}(\tilde{W})^{t-1}}{\sqrt{n}\log^{2}n}}\mid\mathcal{E},\mathcal{E}_{\delta}\Bigg]+{\frac{2e(2t-2)^{2}}{n^{2}}}$$ $$+{\frac{2e(2t-2)^{2}}{n^{2}}}\leq{\frac{2e(2t)^{2}}{n^{2}}}.$$
$ \leq\quad\Pr\begin{bmatrix}C_t\geq0\\ \\ 2et+1\\ \leq\quad\dfrac{2et+1}{n^2}\end{bmatrix}$  . 
$$\mathrm{Pr}\left[\left|c_{1}\lambda_{1}(\bar{W})^{T}v_{1,i}\right|>\left|\sum_{t=1}^{T}e_{1}^{T}\lambda_{1}(\bar{W})^{T-t}v_{1,i}+\sum_{j=2}^{n}\tilde{c}_{j}v_{j,i}\right|\right]\geq0.95-o(1).$$
19 Pr[*E | E*δ] ≥ 1 −
8eT 2 n2 . Also, by Proposition C.2, we know that Pr[E¯δ] ≤
1 2n
. As a result, when n is large enough, Since the distribution of η is symmetric around 0, we have that By combining (13) and (14) and using the fact that |v1,i| ≥ √
γ n
, we obtain that:

$$\mathrm{Pr}\left[|v_{1,i}|\left(|c_{1}\lambda_{1}(\tilde{W})^{T}|-|\eta|\right)\geq\frac{\gamma\lambda_{1}(\tilde{W})^{T}}{32\sqrt{n}}\right]>0.95-o(1)$$

By the assumption that λ2(W˜ ) ≤λ1(W˜ )
g, we have that
Pn j=2 cjvj,iλj (W˜ )
T ≤
λ1(W˜ )
T
gT ·Pn j=2 cj
. Since Pn j=2 cjvj,iλj (W˜ )
Tis a normal random variable with mean 0 and standard deviation at most
√n·λ1(W˜ )
T
gT , we have the following.

$$\mathrm{Pr}\left[\left|\sum_{j=2}^{n}c_{j}v_{j,i}\lambda_{j}(\tilde{W})^{T}\right|\geq{\frac{\sqrt{n}\log n\cdot\lambda_{1}(\tilde{W})^{T}}{g^{T}}}\right]<{\frac{1}{n^{2}}}.$$
$$(14)$$
$$(15)$$
at $\;\left|\sum_{j=2}^n c_j v_{j,i}\lambda_j\right|$
When T =
2 log n log gand n is large enough, we obtain that
√n log n gT <γ 65√n
. Hence,

$$\mathrm{Pr}\left[\left|\sum_{j=2}^{n}c_{j}v_{j,i}\lambda_{j}(\tilde{W})^{T}\right|\geq\frac{\gamma\cdot\lambda_{1}(\tilde{W})^{T}}{65\sqrt{n}}\right]<\frac{1}{n^{2}}.$$
$$(16)$$

1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081 1082 1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099 Consider the summation Pn j=2 PT
t=1 e
(t)
j λj (W˜ )
T −tvj,i. Denote the summation as βt. By Lemma D.3, we know that E[e
(t)
j] = 0 for all j, t. As a result, E[βt] = E
hPn j=2 e
(t)
j λj (W˜ )
T −tvj,ii= 0 for all t. Furthermore, by λj (W˜ ) ≤
λ1(W˜ )
g for all j ≥ 2, We observe that both e
(t)
jand βt =Pn j=2 e
(t)
j λj (W˜ )
T −tvj,i can be written as linear combinations of y
(t)
1*, . . . , y*
(t)
n .

Let b
(t) 1
, . . . , b
(t)
n ∈ R be such that βt =Pn j=1 b
(t)
jy
(t)
j. By (17), Var [βt] = 2Pn j=1 b 2 jC
2 t ≤ 2 λ1(W˜ )
2T−2t g 2T−2t C
2 tand

$$\operatorname{Var}\left[\beta_{t}\right]$$
Var [βt] ≤X
n j=2 λj (W˜ ) 2T −2tv 2 j,iVar(e (t) j) ≤X n j=1 λ1(W˜ ) 2T −2t g 2T −2tv 2 j,iVar(e (t) j ) =λ1(W˜ ) 2T −2t g 2T −2tVar  X j vj,ie (t) j   =λ1(W˜ ) 2T −2t g 2T −2tVar  hy (t) i i = 2 λ1(W˜ ) 2T −2t g 2T −2tC 2
$$(17)$$
t. (17)
$$\mathrm{Pr}\left[\eta\geq{\frac{\lambda_{1}({\tilde{W}})^{T}}{32}}\right]\quad\leq$$ $$\leq$$
≤ Pr 
$$\begin{array}{l}{{\mathrm{Pr}\left[\eta\geq\frac{\lambda_{1}(\tilde{W})^{T}}{32}\;\mathrm{and}\;\mathcal{E}\;\mathrm{and}\;\mathcal{E}_{\delta}\right]+\mathrm{Pr}[\tilde{\mathcal{E}}\;\mathrm{and}\;\mathcal{E}_{\delta}]+\mathrm{Pr}[\tilde{\mathcal{E}}_{\delta}]}}\\ {{\mathrm{Pr}\left[\eta\geq\frac{\lambda_{1}(\tilde{W})^{T}}{32}\mid\mathcal{E},\mathcal{E}_{\delta}\right]+\mathrm{Pr}[\tilde{\mathcal{E}}\mid\mathcal{E}_{\delta}]+\frac{1}{2n}=o(1).}}\end{array}$$
≤ Pr 
$$\mathrm{Pr}\left[|\eta|\geq{\frac{\lambda_{1}({\tilde{W}})^{T}}{32}}\right]=o(1).$$

#= o(1). (14)
20