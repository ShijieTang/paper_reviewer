011

014 015 016

018

024

026

034

036

038

054

# Accelerating Eigenvalue Dataset Generation via Chebyshev Subspace Filter

Anonymous Authors<sup>1</sup>

## Abstract

Eigenvalue problems are among the most important topics in many scientific disciplines. With the recent surge and development of machine learning, neural eigenvalue methods have attracted significant attention as a forward pass of inference requires only a tiny fraction of the computation time compared to traditional solvers. However, a key limitation is the requirement for large amounts of labeled data in training, including operators and their corresponding eigenvalues. To tackle this limitation, we propose a novel method, named Sorting Chebyshev Subspace Filter (SCSF), which significantly accelerates eigenvalue data generation by leveraging similarities between operators—a factor overlooked by all existing methods. Specifically, SCSF employs truncated fast Fourier transform (FFT) sorting to group operators with similar eigenvalue distributions and constructs a Chebyshev subspace filter that leverages eigenpairs from previously solved problems to assist in solving subsequent ones, reducing redundant computations. To the best of our knowledge, SCSF is the first method to accelerate eigenvalue data generation. Experimental results show that SCSF achieves up to a 6× speedup compared to various numerical solvers.

# 1. Introduction

Solving eigenvalue problems is an important challenge in fields such as quantum physics [\(Pfau et al.,](#page-9-0) [2023\)](#page-9-0), electromagnetism [\(Augenstein et al.,](#page-8-0) [2023\)](#page-8-0), and structural mechanics [\(Wen et al.,](#page-9-1) [2022\)](#page-9-1). Traditional numerical solvers, such as the Krylov-Schur algorithm [\(Stewart,](#page-9-2) [2002\)](#page-9-2), often suffer from prohibitively high computational costs when tackling complex problems. To overcome these computational challenges, recent advancements in deep learning [\(Schutt et al.](#page-9-3) ¨ , [2017;](#page-9-3) [Li et al.,](#page-8-1) [2020;](#page-8-1) [Luo et al.\)](#page-9-4) have demonstrated remarkable success as one forward pass only necessitates a tiny fraction of the computation time compared to numerical solvers, often in milliseconds.

Despite their success, data-driven approaches face a fundamental limitation: the reliance on labeled datasets. Training neural networks require large-scale labeled data, which is often generated using computationally expensive traditional methods. For example, the QM9 dataset [\(Ramakrishnan](#page-9-5) [et al.,](#page-9-5) [2014\)](#page-9-5) contains 1.34 × 10<sup>5</sup> molecular data points, each produced by solving Hamiltonian operator eigenvalue problems. These calculations typically employ traditional algorithms such as the Krylov-Schur method [\(Stewart,](#page-9-2) [2002\)](#page-9-2), whose computational costs can escalate dramatically with increasing problem complexity, like finer grid resolutions or higher accuracy requirements. This scalability issue represents a significant bottleneck for generating the labeled data needed to train deep learning models. Furthermore, the diversity of scientific problems leads to the need for a unique dataset for each scenario, which further intensifies this challenge of computational intractability. As a result, the high computational expense of generating eigenvalue data severely limits the practical application of deep learning approaches [\(Zhang et al.,](#page-9-6) [2023\)](#page-9-6).

In particular, the dataset generation process typically involves six key steps, as illustrated in the flowchart in Figure [1.](#page-1-0) Among these steps, the computation of eigenvalues of matrices is the most computationally demanding (step 4), accounting for 95% of the total processing cost [\(Hughes,](#page-8-2) [2012\)](#page-8-2). Existing data generation methods typically compute the eigenvalues of each matrix in the dataset independently. However, the operators in the dataset often share similarities, as they describe related physical phenomena, which can largely simplify and accelerate the eigenvalue-solving process. Existing approaches, however, fail to leverage these similarities, leading to significant computational redundancy. Previous works [\(Wang et al.,](#page-9-7) [2024;](#page-9-7) [Dong et al.,](#page-8-3) [2024\)](#page-8-3) on solving PDE datasets have demonstrated the potential of leveraging similarity to significantly reduce the time required for dataset generation. However, effectively exploiting matrix similarity to accelerate eigenvalue problem solving for datasets remains a significant challenge.

To address this challenge, we introduce a novel data generation approach, named Sorting Chebyshev Subspace Filter (SCSF). SCSF is designed to leverage the approximate eigenpairs of close problems to reduce redundant computa-

![](_page_1_Diagram_2.jpeg)

Figure 1: Generation process of the eigenvalue dataset: 1. Generate a set of random problem parameters. 2. Derive the corresponding operators based on these parameters. 3. Convert the operators into matrices using discretization methods. 4. Independently solve for the matrix eigenvalues using numerical solvers. 5. Obtain the matrix eigenpairs, converting them into the operator eigenpairs. 6. Assemble the dataset.

tions in the eigenvalue solving process, thereby accelerating dataset generation. Specifically, in the initial stages, SCSF employs a sorting algorithm based on truncated Fast Fourier transform (FFT), which arranges these operators efficiently, enhancing the adjacent correlation between them and laying the groundwork for sequential solving. Then, SCSF accelerates the convergence of iterations and significantly reduces computation times by constructing a Chebyshev subspace filter, which solves the problem aided by the eigenpairs identified from previous problem solutions. The core design of SCSF is to identify and exploit the close spectral distributions and invariant subspaces within these eigenvalue problems. SCSF coordinates the sequential resolution of these systems rather than treating them as discrete entities. This improved approach not only alleviates the computational demands of eigenvalue solutions but also significantly speeds up the generation of training data for related data-driven algorithms. We summarize our contributions as follow:

- To the best of our knowledge, SCSF is the first method to accelerate the eigenvalue data generation.
- By using truncated FFT sorting and the Chebyshev subspace filtered iteration, we introduce a novel approach that solves the operator problem sequentially.
- Comprehensive experiments demonstrate that SCSF substantially reduces the computational cost of eigenvalue dataset generation. As demonstrated in Figure [2,](#page-1-1) our method achieves up to a 6× speedup compared to state-of-the-art solvers.

## 2. Related work

#### 2.1. Eigenvalue Datasets and Neural Eigenvalue Methods

Eigenvalue datasets are prevalent in neural eigenvalue methods. In quantum chemistry research, eigenvalue algorithms

![](_page_1_Figure_0.jpeg)

Figure 2: Results of average computation times across various algorithms based on the number of eigenpairs solving.

are commonly used to determine key molecular features, such as orbital energy levels [\(Kittel & McEuen,](#page-8-4) [2018\)](#page-8-4). These features, which form the basis of the datasets, are obtained by solving the eigenvalues of Hamiltonian operators [\(Helgaker et al.,](#page-8-5) [2013\)](#page-8-5). Notable datasets in this domain include QM7 [\(Blum & Reymond,](#page-8-6) [2009\)](#page-8-6), QM9 [\(Ra](#page-9-5)[makrishnan et al.,](#page-9-5) [2014\)](#page-9-5), ANI-1 [\(Smith et al.,](#page-9-8) [2017\)](#page-9-8), MD17 [\(Chmiela et al.,](#page-8-7) [2017\)](#page-8-7). These datasets have been widely used to train and validate neural eigenvalue methods [\(Schutt et al.](#page-9-3) ¨ , [2017;](#page-9-3) [Bartok et al.](#page-8-8) ´ , [2017;](#page-8-8) [Rupp et al.,](#page-9-9) [2012\)](#page-9-9), thereby advancing tasks in molecular property prediction and materials design. Besides, [Luo et al.](#page-9-4) accelerates the solution of linear equations by predicting the eigenfunctions of differential operators, which requires a dataset of eigenfunctions for training.

#### 2.2. Data Generation for Eigenvalue Algorithms

Training data-driven algorithms require a large amount of labeled eigenvalue data. Typically, the generation of these high-precision data is obtained by traditional algorithms. In the field of computational mathematics, solving operator eigenvalue problems often involves utilizing various discretization methods such as finite difference methods (FDM) [\(Strikwerda,](#page-9-10) [2004\)](#page-9-10), finite element methods (FEM) [\(Hughes,](#page-8-2) [2012;](#page-8-2) [Johnson,](#page-8-9) [2012\)](#page-8-9). These discretization methods transform operator eigenvalue problems into matrix eigenvalue problems, which are then solved using the corresponding matrix algorithms. For larger matrices, the Krylov-Schur algorithm [\(Stewart,](#page-9-2) [2002\)](#page-9-2), Jacobi-Davidson [\(Sleijpen](#page-9-11) [& Van der Vorst,](#page-9-11) [2000\)](#page-9-11), and locally optimal block preconditioned conjugate gradient (LOBPCG) [\(Knyazev,](#page-8-10) [2001\)](#page-8-10) are among the most frequently employed algorithms [\(Golub &](#page-8-11) [Van Loan,](#page-8-11) [2013\)](#page-8-11).

Nonetheless, traditional methods were not designed for dataset generation, resulting in high computational costs, which have become a significant barrier to the advancement of data-driven approaches [\(Zhang et al.,](#page-9-6) [2023;](#page-9-6) [Hao et al.,](#page-8-12) [2022\)](#page-8-12). Recent data augmentation research [\(Brandstetter](#page-8-13)

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

[et al.,](#page-8-13) [2022;](#page-8-13) [Liu et al.,](#page-8-14) [2023\)](#page-8-14) has led to the development of methods that preserve symmetries and conservation laws, enhancing model generalization and data efficiency. [Wang](#page-9-7) [et al.](#page-9-7) [\(2024\)](#page-9-7); [Dong et al.](#page-8-3) [\(2024\)](#page-8-3) report acceleration in the process of solving linear equations, thereby speeding up the generation of PDE datasets. However, these improvements largely focus on neural networks or the rapid solution of linear equation-based PDEs, without discussing optimizations in the generation of eigenvalue datasets.

#### 2.3. Chebyshev Filter Technique

The Chebyshev filter technique originates from polynomial approximation theory, where the core concept involves using Chebyshev polynomials to accelerate the convergence of eigenvalues [\(Zhou & Saad,](#page-10-0) [2007\)](#page-10-0). This technique constructs a polynomial filter that selectively amplifies spectral components in a specified interval, thereby speeding up the solution of specific eigenvalues. This technique is particularly effective in dealing with sequence eigenvalue problems [\(Saad,](#page-9-12) [2011;](#page-9-12) [Zhou et al.,](#page-10-1) [2006a\)](#page-10-1) and has been applied in various contexts, such as stability analysis in electronic structure [\(Pieper et al.,](#page-9-13) [2016;](#page-9-13) [Banerjee et al.,](#page-8-15) [2016\)](#page-8-15) and quantum chemical computations [\(Mohr et al.,](#page-9-14) [2017;](#page-9-14) [Zhou](#page-10-2) [et al.,](#page-10-2) [2014;](#page-10-2) [2006b\)](#page-10-3). To further adapt this technique to the generation of operator eigenvalue datasets, we have developed a specialized sorting algorithm that transforms dataset generation into a sequential solving problem. Throughout the solving process, eigenpairs obtained from previous solutions are used to construct Chebyshev filters, accelerating subsequent solutions.

## 3. Preliminaries

#### 3.1. Discretization of Eigenvalue Problem

Our main focus is on solving matrix eigenvalues, the most time-consuming part of data generation. As depicted in Figure [1,](#page-1-0) these problems are typically solved by using numerical discretization methods such as FDM [\(Strikwerda,](#page-9-10) [2004;](#page-9-10) [LeVeque,](#page-8-16) [2002\)](#page-8-16). These discretization techniques embed the infinite-dimensional Hilbert space of operators into an appropriate finite-dimensional space, thereby transforming operator problems into matrix problems. We provide a simple example to clarify the discussed processes. A detailed account of the equation assembly process can be found in Appendix [A.](#page-11-0) Specifically, we discuss using FDM to solve the eigenvalue problem of the two-dimensional Poisson operator, transforming it into a matrix eigenvalue problem:

$$k(x, y)\nabla^2 u(x, y) = \lambda u(x, y). \quad (1)$$

[W](#page-8-13)e map the problem onto a 2 × 2 grid (i.e., N<sup>x</sup> = N<sup>y</sup> = 2 and ∆x = ∆y), where both the variable ui,j and the coefficients ki,j follow a row-major order. This setup facilitates

the derivation of the matrix eigenvalue equation:

$$\begin{bmatrix} k_{1,1} & 0 & 0 & 0 \\ 0 & k_{1,2} & 0 & 0 \\ 0 & 0 & k_{2,1} & 0 \\ 0 & 0 & k_{2,2} \end{bmatrix} \begin{bmatrix} -4 & 1 & 0 \\ 1 & -4 & 0 \\ 0 & 1 & -4 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} u_{1,1} \\ u_{1,2} \\ u_{2,1} \\ u_{2,2} \end{bmatrix} = \lambda \begin{bmatrix} u_{1,1} \\ u_{1,2} \\ u_{2,1} \\ u_{2,2} \end{bmatrix}.$$

By employing various methods to generate the parameter matrices P ,

$$P = \begin{bmatrix} k_{11} & k_{12} \\ k_{21} & k_{22} \end{bmatrix}. \quad (3)$$

Such as utilizing Gaussian random fields (GRF) or truncated polynomials, we can derive Poisson operators characterized by distinct parameters.

Typically, training a neural network requires 10<sup>3</sup> to 10<sup>5</sup> data [\(Lu et al.,](#page-8-17) [2019\)](#page-8-17). Such a multitude of eigenvalue systems, derived from the same distribution of operators, naturally exhibit a highly similarity [\(Soodhalter et al.,](#page-9-15) [2020\)](#page-9-15). It is precisely this similarity that is key to the effective acceleration of SCSF. We can conceptualize this as the task of solving a sequential series of matrix eigenvalue problems:

$$A^{(i)}v_j^{(i)} = \lambda_j^{(i)}v_j^{(i)}, \quad j = 1, \dots, L; \quad i = 1, 2, \dots, \quad (4)$$

where the matrix A(i) ∈ <sup>C</sup> <sup>n</sup>×<sup>n</sup>, the eigenvector v (i) <sup>j</sup> ∈ C <sup>n</sup> and the eigenvalue λ (i) <sup>j</sup> ∈ <sup>C</sup> vary depending on the operator. We define the eigenpairs as (Λ(i) , V (i) ), with Λ (i) = diag(λ (i) 1 , . . . , λ(i) L ) and V (i) = [v (i) 1 | · · · |v (i) L ].

#### 3.2. The Chebyshev Polynomials and Chebyshev Filter

Chebyshev filtered subspace iteration is closely related to Chebyshev orthogonal polynomials [\(Mason & Handscomb,](#page-9-16) [2002;](#page-9-16) [Rivlin,](#page-9-17) [2020\)](#page-9-17). Chebyshev polynomials are widely used due to their strong approximation capabilities. The Chebyshev polynomials Cm(t) of degree m are defined on the interval [−1, 1] and are expressed as

$$C_m(t) = \cos(m \cos^{-1}(t)), \quad |t| \leq 1. \quad (5)$$

Cm(t) commonly referred to as the Chebyshev polynomial of the first kind, satisfies the following recurrence relation:

$$C_{m+1}(t) = 2tC_m(t) - C_{m-1}(t). \quad (6)$$

For a Hermitian matrix A ∈ C <sup>n</sup>×<sup>n</sup> and vectors Y<sup>0</sup> ∈ <sup>C</sup> n×k , we use the three-term recurrence relation that defines Chebyshev polynomials in vector form:

$$C_{m+1}(Y_0) = 2AC_m(Y_0) - C_{m-1}(Y_0), \quad (7)$$

$$C_m(Y_0) \equiv C_m(A)Y_0. \quad (8)$$

The computation of Cm(Y0) and the Chebyshev filter is described in Algorithm [1.](#page-3-0) Let A′ denote the previously

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

solved related matrix, with (λ ′ i , v′ i ) in ascending order, and {λ ′ 2 , . . . , λ′ <sup>n</sup>} ∈ [α, β]. In Algorithm [1,](#page-3-0) the parameter λ is typically approximated by λ ′ 1 , while c = α+β 2 and e = β−α 2 represent the center and half-width of the interval [α, β], providing estimates for the spectral distribution of A.

Algorithm 1: Chebyshev Filter [\(Berljafa et al.,](#page-8-18) [2015\)](#page-8-18)

Input: Matrix A ∈ C <sup>n</sup>×<sup>n</sup>, vectors Y<sup>0</sup> ∈ <sup>C</sup> n×k sorted according to ascending degree specification m = (m1, . . . , mk) ∈ <sup>N</sup> k , and parameters λ, c, e ∈ R.

Output: Filtered vectors Y<sup>m</sup> = Cm(Y0), where each vector Ym,j is filtered with a Chebyshev polynomial of degree m<sup>j</sup> .

 A = (A − cIn)/e; σ<sup>1</sup> = e/(λ − c); Y<sup>1</sup> = σ1AY0; s = argmin j=1,...,k

m<sup>j</sup> ̸= 1;

<sup>5</sup> for i = 1, . . . , m<sup>k</sup> − 1 do <sup>6</sup> σi+1 = 1/(2/σ<sup>1</sup> − σi); <sup>7</sup> Yi+1,1:s−<sup>1</sup> = Yi,1:s−1;

<sup>8</sup> Yi+1,s:<sup>k</sup> = 2σi+1AYi,s:<sup>k</sup> − σi+1σiYi,s:k;

<sup>9</sup> s = argmin

j=1,...,k

m<sup>j</sup> ̸= i + 1;

## 4. Method

In this section, we introduce our novel method, named Sorting Chebyshev Subspace Filter (SCSF), a fast data generation approach that improves the efficiency of solving eigenvalue problems by leveraging intrinsic spectral correlations among operators. SCSF incorporates two key components: (1) a truncated Fast Fourier Transform (FFT)-based approach for efficiently sorting operator eigenvalue samples and (2) the Chebyshev filtered subspace iteration (ChFSI) employed for sequential solving. By integrating these components, SCSF enables effective utilization of prior spectral information, accelerating the eigenvalue data generation.

We first introduce the sorting algorithm that leverages the spectral similarities and provides the time complexity analysis in Section [4.1](#page-3-1) . Then we give an introduction to the Chebyshev filtered subspace iteration in Section [4.2.](#page-4-0) Figure [3](#page-4-1) shows the overview of our SCSF. Generally, the truncated FFT sorting algorithm ensures that successive matrices in the sequence exhibit close relations. Then ordered sequence enables the Chebyshev filtered subspace iteration (ChFSI) to effectively utilize prior information, thereby accelerating the solution process.

#### 4.1. The Sorting Algorithm

To benefit the successive solving sequence of the eigenvalue problem, we need the sorting algorithm that pulls matrices with similar spectral properties, like frequency series, close enough in the solving sequence, so that eigenvalue solving of the current matrix in sequence can be easily boosted by the last solving. A naive sorting strategy is a greedy sort that uses the similarity of spectral properties between two matrices, like frequency as the distance. And by repeatedly fetching without reservation from the remaining matrix in the data pool, we can re-organize the solving sequence so that the successive solving can benefit from the re-ordered sequence.

Algorithm 2: The Truncated FFT Sorting Algorithm

Input: Sequence of eigenvalue problems to be solved A(i) ∈ <sup>C</sup> <sup>n</sup>×<sup>n</sup>, corresponding parameter matrix P (i) ∈ <sup>C</sup> p×p , i = 1, 2, · · · , N and k is the truncation threshold for low frequencies.

Output: Sequence for eigenvalue problems seqmat. <sup>1</sup> Initialize the list with sequence seq<sup>0</sup> = {1, 2, · · · , N},

seqmat is an empty list;

<sup>2</sup> Set i<sup>0</sup> = 1 as the starting point. And remove 1 from

seq<sup>0</sup> and append 1 to seqmat;

<sup>3</sup> for i = 1, · · · , N do

<sup>4</sup> Let P

(i)

low = Trunc<sup>k</sup>

 FFT(P (i) ) 

. Perform

truncated FFT on matrix P

(i)

to extract

low-frequency information, and P

(i) low ∈ <sup>C</sup>

k×k ;

<sup>5</sup> for i = 1, · · · , N − 1 do

<sup>6</sup> Refresh dis and set it to a large number, e.g., 1000;

<sup>7</sup> for *each* j *in* seq<sup>0</sup> do

<sup>8</sup> dis<sup>j</sup> = the Frobenius norm of the difference

between P

(i0) low and P

(j) low;

<sup>9</sup> if disj<dis then

<sup>10</sup> dis = dis<sup>j</sup> and jmin = j;

<sup>11</sup> Remove jmin from seq<sup>0</sup> and append jmin to

seqmat and set i<sup>0</sup> = jmin;

<sup>12</sup> Get the sequence for eigenvalue problems seqmat;

However, the core computational cost of such a naive sorting algorithm arises from repeatedly calculating the distances between different matrices A, which is directly related to the matrix dimension—that is, the resolution of operators. Recalling Section [3.1,](#page-2-0) our eigenvalue problem, the matrix A, is generated from the parameter matrix P [\(Lu et al.,](#page-8-19) [2022;](#page-8-19) [Li et al.,](#page-8-1) [2020\)](#page-8-1). Existing works [\(Holmes,](#page-8-20) [2012;](#page-8-20) [Li](#page-8-1) [et al.,](#page-8-1) [2020\)](#page-8-1) have shown that the key variables that affect operators stem from the low-frequency components of the parameter matrices P, while high-frequency components often represent noise or irrelevant data. Based on this insight, to reduce computational overhead during sorting, we first

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

![](_page_4_Diagram_0.jpeg)

Figure 3: Algorithm Flow Diagram: a. Generation of operators to be solved. b. Discretization of operators into matrixes. c. Apply SCSF algorithm to sort matrixes, obtaining a sequence with strong correlations. d. Other algorithms independently solve the eigenvalue problems from step b. d1, d2, d3. SCSF algorithm utilizes Chebyshev subspace iterations to sequentially solve the eigenvalue problems. e. Assembly of eigenvalue pairs into a dataset. f. Amplification of the interval of interest through spectral transformation. g. Replacement of initial subspaces with previously solved invariant subspaces.

perform truncated FFT on the parameter matrices to extract the low-frequency information before sorting. We then sort by comparing the distances between these low-frequency components.

As shown in Algorithm [2,](#page-3-2) suppose we have N eigenvalue problems, the parameter matrices P (i) ∈ <sup>C</sup> p×p , and the low-frequency truncated matrices P (i) low ∈ <sup>C</sup> k×k . The computational complexity of directly using a greedy algorithm is O(N<sup>2</sup>p 2 ). Our sorting algorithm's complexity consists of two main parts: 1. FFT Computation: The complexity of FFT is O(p 2 log p) per matrix. For N matrices, this totals O(N p<sup>2</sup> log p). 2. Greedy Sorting: The subsequent greedy sorting algorithm has a complexity of O(N<sup>2</sup>k 2 ).

Overall, the total complexity is O(N<sup>2</sup>k <sup>2</sup> + N p<sup>2</sup> log p). Since k ≪ p and p ≪ N, our sorting algorithm effectively reduces computational cost compared to the naive greedy algorithm.

### 4.2. Chebyshev Filtered Subspace Iteration

In a series of eigenvalue problems, inherent correlations often exist between successive systems. We hypothesize that leveraging the eigenpairs (Λ(i−1), V (i−1)) of the previous problem A(i−1) can accelerate the iterative convergence of the subsequent system A(i) , thereby significantly enhancing computational performance. For various types of operators, the resulting eigenvalue problems produce matrices with distinct structural characteristics. These unique matrix structures align well with the ChFSI method [\(Manteuffel,](#page-9-18) [1977;](#page-9-18) [Saad,](#page-9-12) [2011;](#page-9-12) [Winkelmann et al.,](#page-9-19) [2019;](#page-9-19) [Berljafa et al.,](#page-8-18) [2015\)](#page-8-18). To verify the effectiveness of our algorithm, we focus on the most common scenario in eigenvalue problems where the operator is self-adjoint; in this case, the corresponding matrix A is Hermitian.

Algorithm [3](#page-5-0) outlines the process of ChFSI for solving the i-th eigenvalue problem A(i) (i > 1) where L eigenvalues are required. The initial approximate invariant subspace V (i−1) and spectral distribution Λ (i−1) are derived from the eigenvectors and eigenvalues of the previous problem A(i−1) in the sequence. The parameter m denotes the polynomial degree in the filter function, typically chosen between 10 and 15. For the first eigenvalue problem A(1) in the sequence, the initial iterative subspace V˜ <sup>0</sup> and initial spectrum Λ˜ <sup>0</sup> are randomly generated.

Specifically, ChFSI begins by estimating the upper bound of the eigenvalue spectrum (line 3) using a few Lanczos iterations and known approximate eigenvalues [\(Zhou & Li,](#page-10-4) [2011;](#page-10-4) [Saad,](#page-9-12) [2011\)](#page-9-12). This estimate aids in constructing the subsequent filter function. In line 5, the Chebyshev filter is applied using the vector form of Chebyshev polynomials; details can be found in the Preliminaries section [3.2.](#page-2-1) After the Chebyshev filtering step, the vector block V˜ <sup>0</sup> spanning the invariant subspace may become linearly dependent. To prevent this, orthonormalization is performed (line 6) using QR decomposition based on Householder reflectors. Line 7 computes the Rayleigh quotient of matrix A(i) using the orthonormalized V˜ 0, projecting the eigenvalue problem onto a subspace that approximates the desired eigenspace. In line 8, 9, the reduced eigenvalue problem is diagonalized, and the computed eigenvectors

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

are projected back to the original problem. At the end of the Rayleigh-Ritz step, residuals of the computed eigenvectors are calculated; converged eigenpairs are locked, and non-converged vectors are set to be filtered again (line 10). For each non-converged vector, the optimal degree of the polynomial filter is updated (line 11) based on its residual and approximate eigenvalue.

Algorithm 3: Chebyshev Filtered Subspace Iteration

Input: Eigenvalue problem A(i) , eigenpairs (Λ(i−1), V (i−1)) of the previous eigenvalue problem A(i−1) where Λ (i−1) = diag(λ (i−1) 1 . . . λ(i−1) L ). Initial filter degree m0.

Output: Wanted eigenpairs (Λ(i)

, V (i) ).

<sup>1</sup> Initialize empty arrays/matrices (Λ˜, V˜ ), set

Λ˜

<sup>0</sup> = Λ(i−1), and V˜

<sup>0</sup> = V

(i−1);

<sup>2</sup> Set the filter degrees

(m1, . . . , mL) = (m0, . . . , m0) =: m;

<sup>3</sup> Estimate the largest eigenvalue via Lanczos iteration;

<sup>4</sup> repeat

<sup>5</sup> Apply Chebyshev filter: V˜

<sup>0</sup> = Cm(V˜

<sup>0</sup>);

<sup>6</sup> Perform QR orthonormalization on [V˜ |V˜

0];

<sup>7</sup> Compute Rayleigh quotient G = V˜ †

<sup>0</sup> <sup>A</sup>(i)V˜ 0;

<sup>8</sup> Solve the reduced problem GW = WΛ˜

0;

<sup>9</sup> Update V˜

<sup>0</sup> = V˜ <sup>0</sup>W; <sup>10</sup> Lock converged eigenpairs into (Λ˜, V˜ ); <sup>11</sup> Update filter degrees (m1, . . . , mk) = m; <sup>12</sup> until *the number of converged eigenpairs* ≥ *L*;

<sup>13</sup> Return eigenpairs (Λ(i)

, V (i)

) = (Λ˜, V˜ ) ;

Assuming m is the degree of the polynomial, n is the dimension of the matrix A, and f is the number of vectors being filtered, the computational complexity per iteration comprises: 1. Chebyshev Filter: O(mnf) 2. QR Factorization: O(nf <sup>2</sup> ) 3. Rayleigh-Ritz Procedure: O(nf <sup>2</sup> + f 3 ) 4. Residuals Check: O(nf) . Since m ≫ 1, the Chebyshev filtering step is the most computationally intensive.

The acceleration effectiveness of the Chebyshev filtered subspace iteration heavily depends on selecting approximate invariant subspaces and eigenvalues that promote rapid convergence in subsequent iterations. Proper sorting amplifies their impact, reducing the number of iterations required. This underscores the critical importance of the sorting algorithm in our method.

# 5. Experiment

## 5.1. Experimental Settings

To comprehensively assess the performance of our model, denoted as SCSF, against other algorithms, we conducted

extensive experiments, each simulating the generation of an operator eigenvalue dataset. We primarily compared the average computation times across different numbers of eigenvalues solved and various matrix sizes. These tests encompassed three distinct datasets and four mainstream eigenvalue solving algorithms, with SCSF consistently delivering commendable results. The detailed data is available in the Appendix [C.1.](#page-13-0)

Baseline. As previously mentioned, our focus revolves around the eigenvalue problem of matrices derived from selfadjoint differential operators, typically consisting of large sparse Hermitian matrices. We benchmarked against the following mainstream algorithms implemented in professional libraries: 1. Eigsh from SciPy (implicitly restarted Lanczos method) [\(Virtanen et al.,](#page-9-20) [2020;](#page-9-20) [Lehoucq et al.,](#page-8-21) [1998\)](#page-8-21), 2. Locally optimal block preconditioned conjugate gradient (LOBPCG) algorithm from SLEPc [\(Knyazev,](#page-8-10) [2001;](#page-8-10) [Her](#page-8-22)[nandez et al.,](#page-8-22) [2009\)](#page-8-22), 3. Krylov-Schur (KS) algorithm from SLEPc [\(Stewart,](#page-9-2) [2002\)](#page-9-2), 4. Jacobi-Davidson (JD) algorithm from SLEPc [\(Sleijpen & Van der Vorst,](#page-9-11) [2000\)](#page-9-11). For detailed information, please refer to Appendix [B.1.](#page-12-0)

Datasets. To explore the adaptability of the algorithm across different matrix types, we delved into three distinct operator eigenvalue problem challenges: 1. Generalized Poisson operator; 2. Second-order elliptic partial differential operator; 3. Helmholtz operator. For a thorough description of the datasets and their generation, please refer to Appendix [B.2.](#page-12-1)

All experiments focus on computing the smallest L eigenvalues in absolute value and their corresponding eigenvectors, which are indicative of the operator eigenvalues and eigenfunctions. For the runtime environment and experimental parameters, refer to Appendix [B.3](#page-13-1) and [B.4.](#page-13-2) The hyperparameter analysis experiments and the time distribution of each part of SCSF can be found in Appendix [C.4](#page-15-0) and [C.3.](#page-15-1)

## 5.2. Main Experiment

Table [1](#page-6-0) showcases selected experimental data. From this table, we can infer several conclusions: Firstly, across all tests, our SCSF algorithm consistently maintained the lowest computation times. The most significant improvements appeared in the Helmholtz dataset, where SCSF demonstrated speedups of 8×, 20×, 6×, and 95× compared to Eigsh, LOBPCG, KS, and JD algorithms, respectively. These results confirm that SCSF effectively reduces inherent redundancies in sequential eigenvalue problems, substantially accelerating dataset generation.

Secondly, as the number of eigenvalues L solved per matrix increases, the speed advantage of SCSF over other algorithms becomes more pronounced. For instance, on the second-order elliptic operator dataset, when solving for 200 eigenvalues, SCSF is 2.5 times faster than the Krylov-Schur

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

Table 1: Comparison of average computation times (in seconds) for eigenvalue problems using various algorithms. The first row lists different algorithms, the first column details the datasets including matrix dimensions and solution precisions, and the second column shows the number of eigenvalues L computed for each matrix. The best algorithm is in bold. The symbol '-' denotes data not recorded due to excessive computation times.

| Dataset   | L   | Eigsh | LOBPCG | KS    | JD    | SCSF  |
|-----------|-----|-------|--------|-------|-------|-------|
| Poisson   | 200 | 14.20 | 73.03  | 23.76 | 270.2 | 12.85 |
| 2500      | 300 | 26.27 | 151.5  | 45.95 | 920.8 | 25.61 |
| 1e-12     | 400 | 36.86 | 265.3  | 72.32 | 2691  | 33.91 |
| Ellipse   | 200 | 41.82 | 139.2  | 61.77 | 414.3 | 24.08 |
| 4900      | 300 | 62.47 | 264.1  | 110.5 | 1446  | 29.88 |
| 1e-10     | 400 | 87.19 | 459.7  | 188.7 | 3386  | 34.60 |
| Helmholtz | 200 | 151.7 | 129.9  | 98.34 | 489.6 | 31.31 |
| 6400      | 400 | 253.5 | 460.4  | 283.0 | 3829  | 40.52 |
| 1e-8      | 600 | 398.8 | 1031   | 329.6 |       | 51.32 |

method and 5.5 times faster at 400 eigenvalues. This efficiency stems from SCSF inheriting approximate invariant subspaces from previous solutions, effectively leveraging available information to expand the initial search space. Consequently, SCSF requires minimal additional iterations as L increases, resulting in modest computation time growth.

Thirdly, the performance disparity across different datasets is significant. For example, on the generalized Poisson operator dataset, SCSF is only about 10% faster than Eigsh, yet it leads by 4-7 times on the Helmholtz dataset. This difference can be attributed to the numerical properties of different operators and the matrix assembly formats, which directly influence algorithmic performance.

Furthermore, the impact of the matrix dimension is significant. We conducted supplementary experiments, with the data available in the Appendix [C.2.](#page-14-0) The results are shown in Figure [4,](#page-6-1) SCSF performs noticeably better as matrix dimensions increase. Below the matrix dimension of 3600, SCSF and Eigsh show comparable efficiency. However, beyond 5000, SCSF significantly outperforms Eigsh and other algorithms. This phenomenon is analyzed from the matrix approximation of operators. For a fixed operator, its eigenvalues and eigenfunctions are fixed. Different matrix dimensions represent embedding the operator in different finite-dimensional linear spaces. For a fixed number of eigenvalues L, larger matrices more accurately approximate the true eigenvalues (the smallest L eigenvalues by absolute value) of the operator. In other words, larger matrix dimensions result in fewer errors and noise in the computed eigenvalues, allowing for a clearer demonstration of the similarities between operators. Consequently, larger matrix dimensions allow SCSF to better exploit the similarities,

![](_page_6_Figure_0.jpeg)

Figure 4: Plot of average computation time versus matrix dimension for solving 400 eigenvalues with a precision of 1e − 12 on the generalized Poisson operator dataset.

yielding superior performance.

#### 5.3. Efficacy of Chebyshev Subspace Filter

To analyze the efficacy of the Chebyshev Subspace Filter, we conducted the following experiments. After sorting, the initial vector or subspace for the existing algorithms was set to the eigenvectors from the previous problem. We compared the computational time across different methods. The experiments were conducted on the Helmholtz operator dataset, with a matrix dimension of 6400 and a solution accuracy of 1e − 8. The results are shown in Table [2.](#page-7-0)

First, the computation time for SCSF in all experiments was minimal, clearly demonstrating the efficacy of the Chebyshev subspace filter. This also highlights that the Chebyshev subspace filter is the optimal choice for leveraging problem similarity to reduce redundancy.

Second, modifying the initial setup had varying impacts on different algorithms. 1. LOBPCG: showed significant acceleration. Its underlying logic is similar to SCSF, both relying on iterative optimization of the subspace to solve the problem. The initial subspace has a considerable impact on the solution. 2. Eigsh and KS were almost unaffected. These methods start with an initial vector and solve the problem through Krylov iteration. In other words, problem similarity only impacts a vector, with little effect on the overall time. 3. JD showed a performance decline. This is because its performance is sensitive to the size of the initial subspace. Our modification altered the default dimension of the initial subspace.

## 5.4. Efficacy of Sorting Algorithms

We analyze the performance of the sorting algorithm module from two perspectives: 1. Comparing the performance of SCSF algorithm with and without the use of 'sorting' as shown in Table [3.](#page-7-1) 2. Evaluating the effectiveness of different

394

396

Table 2: Impact of initial setup modifications on average computation time (in seconds) for different algorithms. '\*' denotes the modified initial subspace version. The first-row lists algorithms, and the first column shows the number of eigenvalues L computed. The best algorithm is in bold, and '-' indicates unrecorded data due to excessive computation time.

| L   | Eigsh | Eigsh* | LOBPCG | LOBPCG* | KS    | KS*   | JD    | JD*   | SCSF (ours) |
|-----|-------|--------|--------|---------|-------|-------|-------|-------|-------------|
| 200 | 151.7 | 150.2  | 129.9  | 95.9    | 98.34 | 100.6 | 489.6 | 760.1 | 31.31       |
| 300 | 208.8 | 206.3  | 270.1  | 199.8   | 179.9 | 185.2 | 1803  | 3101  | 38.67       |
| 400 | 253.5 | 249.1  | 460.4  | 362.1   | 283.0 | 292.2 | 3829  | 6374  | 40.52       |
| 500 | 324.6 | 315.3  | 717.3  | 573.7   | 314.2 | 317.4 |       |       | 46.70       |
| 600 | 398.8 | 394.7  | 1031   | 866.0   | 329.6 | 335.7 |       |       | 51.32       |

Table 3: Performance comparison of SCSF with and without sorting. The first column lists the number of eigenvalues L computed, while subsequent columns display average computation times, average iteration counts, total Flop counts, and filter Flop counts. Experiments used the matrix dimension of 2500 and precision 1e − 12 on the generalized Poisson operator dataset.

| L   | nosort | Time (s) sort | nosort | Iteration sort | nosort | Flops sort | Filter nosort | Flops sort |
|-----|--------|---------------|--------|----------------|--------|------------|---------------|------------|
| 20  | 8.248  | 2.971         | 19.70  | 9.880          | 519.7  | 298.4      | 485.8         | 280.8      |
| 100 | 14.18  | 9.891         | 18.77  | 35.38          | 1984   | 2332       | 1798          | 1970       |
| 200 | 18.45  | 12.85         | 36.30  | 33.67          | 4459   | 3944       | 3654          | 3192       |
| 300 | 34.59  | 25.61         | 47.50  | 39.18          | 8967   | 7544       | 6985          | 5702       |
| 400 | 42.60  | 33.91         | 47.43  | 45.18          | 12022  | 11182      | 9087          | 8338       |

Table 4: Comparison of average computation times (in seconds) for different sorting algorithms, with the first column indicating dataset size. Experiments used the matrix dimension of 6400 on the Helmholtz dataset.

| Size | Greedy Total | Truncated FFT | FFT Sort Greedy | (ours) Total |
|------|--------------|---------------|-----------------|--------------|
| 10 2 |              |               |                 |              |
|      | 0.114        | 0.0016        | 0.0147          | 0.0163       |
| 10 3 |              |               |                 |              |
|      | 7.328        | 0.0164        | 1.421           | 1.438        |
| 10 4 |              |               |                 |              |
|      | 592.7        | 0.1658        | 150.9           | 151.1        |

sorting algorithms as detailed in Tables [4](#page-7-2) and [5.](#page-7-3)

Firstly, Table [3](#page-7-1) indicates that incorporating sorting can enhance SCSF computation speed to 1.3 to 2.8 times, reduce the number of iterations by 5% to 50%, and decrease total Flops by 7% to 43%. The optimization effect of sorting is more pronounced with smaller numbers of solutions (L). This is because when L is large, the inherited subspace already contains most of the necessary correlation information, diminishing the impact of sorting. Moreover, the Flops in the Filter component constitute over 70% of SCSF's computational load. A detailed time analysis of different aspects of SCSF can be found in Appendix [C.3.](#page-15-1)

Secondly, as shown in Table [4,](#page-7-2) our designed truncated FFT sorting algorithm incurs significantly lower time overhead compared to the complete greedy sorting in SKR [\(Wang](#page-9-7) [et al.,](#page-9-7) [2024\)](#page-9-7), with its benefits becoming more pronounced

Table 5: Comparison of average computation times (in seconds) and iteration counts for different sorting algorithms using SCSF. Experiments used the matrix dimension of 6400 on the Helmholtz dataset, precision 1e − 8, and targeting 400 eigenvalues.

|           | Nosort | Greedy | Ours  |
|-----------|--------|--------|-------|
| Time (s)  | 66.66  | 40.52  | 40.52 |
| Iteration | 10.4   | 5.5    | 5.5   |

as dataset size increases. In the truncated FFT sorting algorithm, the FFT contributes minimally to computational overhead but significantly reduces the time required for subsequent greedy sorting. Table [5](#page-7-3) shows SCSF solution times for matrices sorted using either greedy or truncated FFT sorting are nearly identical, highlighting its effectiveness.

## 6. Conclusions

In this paper, we introduced SCSF algorithm. To the best of our knowledge, this is the first method to accelerate eigenvalue dataset generation by reducing computational redundancy in the associated matrix eigenvalue problems. The proposed SCSF algorithm significantly reduces the computational overhead of eigenvalue dataset generation, thereby addressing a major obstacle to the application of neural networks in scientific computing.

- 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Impact Statement This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here. References Augenstein, Y., Repan, T., and Rockstuhl, C. Neural operator-based surrogate solver for free-form electromagnetic inverse design. *ACS Photonics*, 2023. Banerjee, A. S., Lin, L., Hu, W., Yang, C., and Pask, J. E. Chebyshev polynomial filtered subspace iteration in the discontinuous galerkin method for large-scale electronic structure calculations. *The Journal of chemical physics*, 145(15), 2016. Bartok, A. P., De, S., Poelking, C., Bernstein, N., Kermode, ´
  - J. R., Csanyi, G., and Ceriotti, M. Machine learning ´ unifies the modeling of materials and molecules. *Science advances*, 3(12):e1701816, 2017. Berljafa, M., Wortmann, D., and Di Napoli, E. An optimized and scalable eigensolver for sequences of eigenvalue problems. *Concurrency and Computation: Practice and Experience*, 27(4):905–922, 2015. Bers, L., John, F., and Schechter, M. *Partial differential equations*. American Mathematical Soc., 1964. Blum, L. C. and Reymond, J.-L. 970 million druglike small molecules for virtual screening in the chemical universe database GDB-13. *J. Am. Chem. Soc.*, 131:8732, 2009. Brandstetter, J., Berg, R. v. d., Welling, M., and Gupta, J. K. Clifford neural layers for pde modeling. *arXiv preprint arXiv:2209.04934*, 2022. Chmiela, S., Tkatchenko, A., Sauceda, H. E., Poltavsky, I., Sch¨"utt, K. T., and M¨"uller, K.-R. Machine learning of accurate energy-conserving molecular force fields. *Science advances*, 3(5):e1603015, 2017. Dong, H., Wang, H., Liu, H., Luo, J., and Wang, J. Accelerating pde data generation via differential operator action in solution space. *arXiv preprint arXiv:2402.05957*, 2024. Evans, L. C. *Partial differential equations*, volume 19. American Mathematical Society, 2022. Golub, G. H. and Van Loan, C. F. *Matrix computations*. JHU press, 2013. Hao, Z., Liu, S., Zhang, Y., Ying, C., Feng, Y., Su, H., and Zhu, J. Physics-informed machine learning: A survey on problems, methods and applications. *arXiv preprint arXiv:2211.08064*, 2022. Helgaker, T., Jorgensen, P., and Olsen, J. *Molecular electronic-structure theory*. John Wiley & Sons, 2013. Hernandez, V., Roman, J. E., Tomas, A., and Vidal, V. A survey of software for sparse eigenvalue problems. Technical Report STR-6, Universitat Politecnica de Val ` encia, ` 2009. Available at https://slepc.upv.es. Holmes, P. *Turbulence, coherent structures, dynamical systems and symmetry*. Cambridge university press, 2012. Hughes, T. J. *The finite element method: linear static and dynamic finite element analysis*. Courier Corporation, 2012. Johnson, C. *Numerical solution of partial differential equations by the finite element method*. Courier Corporation, 2012. Kittel, C. and McEuen, P. *Introduction to solid state physics*. John Wiley & Sons, 2018. Knyazev, A. V. Toward the optimal preconditioned eigensolver: Locally optimal block preconditioned conjugate gradient method. *SIAM journal on scientific computing*, 23(2):517–541, 2001. Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., and Anandkumar, A. Neural operator: Learning maps between function spaces. *arXiv preprint arXiv:2108.08481*, 2021. Lehoucq, R. B., Sorensen, D. C., and Yang, C. *ARPACK users' guide: solution of large-scale eigenvalue problems with implicitly restarted Arnoldi methods*. SIAM, 1998. LeVeque, R. J. *Finite volume methods for hyperbolic problems*, volume 31. Cambridge university press, 2002. Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar, A. Fourier neural operator for parametric partial differential equations. *arXiv preprint arXiv:2010.08895*, 2020. Liu, N., Yu, Y., You, H., and Tatikola, N. Ino: Invariant neural operators for learning complex physical systems with momentum conservation. In *International Conference on Artificial Intelligence and Statistics*, pp. 6822–6838. PMLR, 2023. Lu, L., Jin, P., and Karniadakis, G. E. Deeponet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators. *arXiv preprint arXiv:1910.03193*, 2019. Lu, L., Meng, X., Cai, S., Mao, Z., Goswami, S., Zhang, Z., and Karniadakis, G. E. A comprehensive and fair comparison of two neural operators (with practical extensions) based on fair data. *Computer Methods in Applied Mechanics and Engineering*, 393:114778, 2022.

504

506

508 509

511

514 515 516

518

524

526

528

531

534

536

538

- Luo, J., Wang, J., Wang, H., Geng, Z., Chen, H., Kuang, Y., et al. Neural krylov iteration for accelerating linear system solving. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*. Manteuffel, T. A. The tchebychev iteration for nonsymmetric linear systems. *Numerische Mathematik*, 28:307–327, 1977. Mason, J. C. and Handscomb, D. C. *Chebyshev polynomials*. Chapman and Hall/CRC, 2002. Mohr, S., Dawson, W., Wagner, M., Caliste, D., Nakajima, T., and Genovese, L. Efficient computation of sparse matrix functions for large-scale electronic structure calculations: The chess library. *Journal of Chemical Theory and Computation*, 13(10):4684–4698, 2017. Pfau, D., Axelrod, S., Sutterud, H., von Glehn, I., and Spencer, J. S. Natural quantum monte carlo computation of excited states. *arXiv preprint arXiv:2308.16848*, 2023. Pieper, A., Kreutzer, M., Alvermann, A., Galgon, M., Fehske, H., Hager, G., Lang, B., and Wellein, G. Highperformance implementation of chebyshev filter diagonalization for interior eigenvalue computations. *Journal of Computational Physics*, 325:226–243, 2016. Rahman, M. A., Ross, Z. E., and Azizzadenesheli, K. U-no: U-shaped neural operators. *arXiv preprint arXiv:2204.11127*, 2022. Ramakrishnan, R., Dral, P. O., Rupp, M., and von Lilienfeld,
- O. A. Quantum chemistry structures and properties of 134 kilo molecules. *Scientific Data*, 1, 2014. Rivlin, T. J. *Chebyshev polynomials*. Courier Dover Publications, 2020. Rupp, M., Tkatchenko, A., Muller, K.-R., and Von Lilien- ¨ feld, O. A. Fast and accurate modeling of molecular atomization energies with machine learning. *Physical review letters*, 108(5):058301, 2012. Saad, Y. *Numerical methods for large eigenvalue problems: revised edition*. SIAM, 2011. Schutt, K. T., Arbabzadah, F., Chmiela, S., M ¨ uller, K. R., ¨ and Tkatchenko, A. Quantum-chemical insights from deep tensor neural networks. *Nature communications*, 8 (1):13890, 2017. Sleijpen, G. L. and Van der Vorst, H. A. A jacobi–davidson iteration method for linear eigenvalue problems. *SIAM review*, 42(2):267–293, 2000. Smith, J. S., Isayev, O., and Roitberg, A. E. Ani-1: an extensible neural network potential with dft accuracy at force field computational cost. *Chemical science*, 8(4): 3192–3203, 2017. Soodhalter, K. M., de Sturler, E., and Kilmer, M. E. A survey of subspace recycling iterative methods. *GAMM-Mitteilungen*, 43(4):e202000016, 2020. Stewart, G. W. A krylov–schur algorithm for large eigenproblems. *SIAM Journal on Matrix Analysis and Applications*, 23(3):601–614, 2002. Strikwerda, J. C. *Finite difference schemes and partial differential equations*. SIAM, 2004. Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., Carey, C. J., Polat, ˙I., Feng, Y., Moore, E. W., VanderPlas, J., Laxalde, D., Perktold, J., Cimrman, R., Henriksen, I., Quintero, E. A., Harris, C. R., Archibald, A. M., Ribeiro, A. H., Pedregosa, F., van Mulbregt, P., and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17:261–272, 2020. doi: 10.1038/s41592-019-0686-2. Wang, H., Hao, Z., Wang, J., Geng, Z., Wang, Z., Li, B., and Wu, F. Accelerating data generation for neural operators via krylov subspace recycling. *arXiv preprint arXiv:2401.09516*, 2024. Wen, G., Li, Z., Azizzadenesheli, K., Anandkumar, A., and Benson, S. M. U-fno—an enhanced fourier neural operator-based deep-learning model for multiphase flow. *Advances in Water Resources*, 163:104180, 2022. Winkelmann, J., Springer, P., and Napoli, E. D. Chase: Chebyshev accelerated subspace iteration eigensolver for sequences of hermitian eigenvalue problems. *ACM Transactions on Mathematical Software (TOMS)*, 45(2):1–34, 2019. Zhang, E., Kahana, A., Turkel, E., Ranade, R., Pathak, J., and Karniadakis, G. E. A hybrid iterative numerical transferable solver (hints) for pdes based on deep operator network and relaxation methods. *arXiv preprint arXiv:2208.13273*, 2022. Zhang, X., Wang, L., Helwig, J., Luo, Y., Fu, C., Xie, Y., Liu, M., Lin, Y., Xu, Z., Yan, K., Adams, K., Weiler, M., Li, X., Fu, T., Wang, Y., Yu, H., Xie, Y., Fu, X., Strasser, A., Xu, S., Liu, Y., Du, Y., Saxton, A., Ling, H., Lawrence, H., Stark, H., Gui, S., Edwards, C., Gao, N., Ladera, ¨ A., Wu, T., Hofgard, E. F., Tehrani, A. M., Wang, R.,

- 551 554 556 558 560 564 566 568 571 574 576 578 580 581 582 583 584 585 586 587 588 589 590 594 596 598 600 601 602 603 604 Daigavane, A., Bohde, M., Kurtin, J., Huang, Q., Phung, T., Xu, M., Joshi, C. K., Mathis, S. V., Azizzadenesheli, K., Fang, A., Aspuru-Guzik, A., Bekkers, E., Bronstein, M., Zitnik, M., Anandkumar, A., Ermon, S., Lio, P., ` Yu, R., Gunnemann, S., Leskovec, J., Ji, H., Sun, J., ¨ Barzilay, R., Jaakkola, T., Coley, C. W., Qian, X., Qian, X., Smidt, T., and Ji, S. Artificial intelligence for science in quantum, atomistic, and continuum systems. *arXiv preprint arXiv:2307.08423*, 2023. Zhou, Y. and Li, R.-C. Bounding the spectrum of large hermitian matrices. *Linear Algebra and its Applications*, 435(3):480–493, 2011. Zhou, Y. and Saad, Y. A chebyshev–davidson algorithm for large symmetric eigenproblems. *SIAM Journal on Matrix Analysis and Applications*, 29(3):954–971, 2007. Zhou, Y., Saad, Y., Tiago, M. L., and Chelikowsky,
  - J. R. Parallel self-consistent-field calculations via chebyshev-filtered subspace acceleration. *Physical Review E—Statistical, Nonlinear, and Soft Matter Physics*, 74(6):066704, 2006a. Zhou, Y., Saad, Y., Tiago, M. L., and Chelikowsky, J. R. Self-consistent-field calculations using chebyshev-filtered subspace iteration. *Journal of Computational Physics*, 219(1):172–184, 2006b. Zhou, Y., Chelikowsky, J. R., and Saad, Y. Chebyshevfiltered subspace iteration method free of sparse diagonalization for solving the kohn–sham equation. *Journal of Computational Physics*, 274:770–782, 2014.

## A. From Differential Operator to Matrix Eigenvalue Problem: An Example

#### A.1. Overview

The general methodology for solving the eigenvalue problems of differential operators numerically, employing techniques such as Finite Difference Method (FDM), Finite Element Method (FEM), and Spectral Method, can be delineated through the following pivotal steps [\(Strikwerda,](#page-9-10) [2004;](#page-9-10) [Hughes,](#page-8-2) [2012;](#page-8-2) [Johnson,](#page-8-9) [2012;](#page-8-9) [LeVeque,](#page-8-16) [2002\)](#page-8-16):

- 1. Mesh Generation: This step involves dividing the domain, over which the differential operator is defined, into a discrete grid. The grid could be composed of various shapes, including squares, triangles, or more complex forms depending on the problem's geometry.
- 2. Operator Discretization: The differential operator is transformed into its discrete counterpart. Essentially, this maps the operator from an infinite-dimensional Hilbert space to a finite-dimensional representation.
- 3. Matrix Assembly: In this phase, the discretized operator is represented in a matrix form. For linear differential operators, this involves creating a system of matrix eigenvalue problems. For nonlinear operators, iterative methods akin to Newton's iteration are employed, transforming the problem into a sequence of matrix eigenvalue problems.
- 4. Applying Boundary Conditions: This involves discretizing and applying boundary conditions specific to the differential operator in question, which are then incorporated into the matrix system.
- 5. Solving the Matrix Eigenvalue Problem: This stage, often the most computationally intensive, entails solving the matrix for its eigenvalues and eigenvectors, which correspond to the eigenvalues and eigenfunctions of the original differential operator.
- 6. Obtaining the Numerical Solution: The final step involves mapping the obtained numerical solutions back onto the original domain, analyzing them for accuracy and stability, and interpreting them in the context of the initial problem.

#### A.2. Example

To illustrate how the FDM can transform the wave equation into a system of matrix eigenvalue problems, let's consider a concrete and straightforward example. Assume we aim to solve a one-dimensional wave equation's operator eigenvalue problem, expressed as

$$-\frac{d^2u}{dx^2} = \lambda u, \quad (9)$$

over the interval [0, L]. The boundary conditions are u(0) = u(L) = 0, signifying fixed-end conditions. In this context, u(x) denotes the eigenfunction, and λ represents the eigenvalue.

1. Mesh Generation: Using the central difference quotient, we divide the interval [0, L] into N + 1 evenly spaced points, including the endpoints. The distance between adjacent points is denoted as ∆x =

L N .

2. Operator Discretization: This step involves formulating the difference equation. At each interior node, which excludes the endpoints and totals N − 1 points, we apply a central difference approximation for the second derivative, represented as

$$\frac{d^2u}{dx^2} \approx \frac{u(x_{i+1}) - 2u(x_i) + u(x_{i-1})}{(\Delta x)^2} \quad (10)$$

3. Matrix Assembly: In this phase, the discretized operator is represented in a matrix form. Following the approximation, we construct the matrix A, an N − 1 × N − 1 tridiagonal matrix, crucial for the computations. The matrix A is constructed as:

$$\mathbf{A} = \frac{1}{(\Delta x)^2} \begin{bmatrix} -2 & 1 & 0 & \dots & 0 \\ 1 & -2 & 1 & \dots & 0 \\ 0 & 1 & -2 & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & -2 \end{bmatrix} \quad (11)$$

4. Applying Boundary Conditions: For the wave equation with boundary conditions u(0) = u(L) = 0, these fixed-end conditions are integrated into the matrix equation. In the FDM framework, the values at the endpoints (u<sup>0</sup> and u<sup>N</sup> ) are zero,

689 690

694

696

698

700

704

706

708 709

711

directly reflecting the boundary conditions. The impact of these conditions is encapsulated in the matrix A, affecting the entries related to u<sup>1</sup> and uN−<sup>1</sup> (the grid points adjacent to the boundaries). The tridiagonal matrix A incorporates these boundary conditions, ensuring that the computed eigenfunctions satisfy u(0) = u(L) = 0.

5. Solving the Matrix Eigenvalue Problem: The final computational step involves solving the matrix eigenvalue problem, expressed as Au = λu. This includes determining the eigenvalues λ and corresponding eigenvectors u, which are discrete approximations of the eigenfunctions of the original differential equation.

6. Obtaining the Numerical Solution: By solving the eigenvalue problem, we obtain numerical solutions that approximate the behavior of the original differential equation. These solutions reveal the eigenvalues and eigenvectors and provide insights into the physical phenomena modeled by the equation.

# B. Details of Experimental Setup

### B.1. Baseline

The baseline algorithms were implemented using the following numerical computing libraries:

- Eigsh: A SciPy (v1.14.1) implementation wrapping ARPACK's SSEUPD and DSEUPD functions, which compute eigenvalues and eigenvectors using the Implicitly Restarted Lanczos Method. Default parameters were used.
- Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG): Implemented in SLEPc (v3.21.1) with default parameters.
- Krylov-Schur (KS): Implemented in SLEPc (v3.21.1) with default parameters.
- Jacobi-Davidson (JD): Implemented in SLEPc (v3.21.1). The implementation uses 'bcgsl' as the linear equation solver, 'bjacobi' as the preconditioner, and sets the linear equation solving precision to 1e-5.

#### B.2. Dataset

#### 1. Generalized Poisson Operator

We consider two-dimensional generalized Poisson operators, which can be described by the following equation [\(Li et al.,](#page-8-1) [2020;](#page-8-1) [Rahman et al.,](#page-9-21) [2022;](#page-9-21) [Kovachki et al.,](#page-8-23) [2021;](#page-8-23) [Lu et al.,](#page-8-19) [2022\)](#page-8-19):

$$-\nabla \cdot (K(x, y) \nabla h(x, y)) = \lambda h(x, y),$$

In our experiment, K(x, y) is derived using the Gaussian Random Field (GRF) method. We convert these operators into matrices using the central difference scheme of FDM. The parameters inherent to the GRF serve as the foundation for our sort scheme.

## 2. Second-Order Elliptic Partial Differential Operator

We consider general two-dimensional second-order elliptic partial differential operators, which are frequently described by the following generic form [\(Evans,](#page-8-24) [2022;](#page-8-24) [Bers et al.,](#page-8-25) [1964\)](#page-8-25):

$$\mathcal{L}u \equiv a_{11}u_{xx} + a_{12}u_{xy} + a_{22}u_{yy} + a_1u_x + a_2u_y + a_0u = \lambda u,$$

where a0, a1, a2, a11, a12, a<sup>22</sup> are constants, and f represents the source term, depending on x, y. The variables u, ux, u<sup>y</sup> are the dependent variable and its partial derivatives. The equation is classified as elliptic if 4a11a<sup>22</sup> > a<sup>2</sup> <sup>12</sup>.

In our experiments, a11, a22, a1, a2, a<sup>0</sup> are uniformly sampled within the range (−1, 1), while the coupling term a<sup>12</sup> is sampled within (−0.01, 0.01). We then select equations that satisfy the elliptic condition to form our dataset. We convert these operators into matrices using the central difference scheme of FDM. The coefficients a0, a1, a2, a11, a12, a<sup>22</sup> serve as the foundation for our sort scheme.

#### 3. Helmholtz Operator

We consider two-dimensional Helmholtz operators, which can be described by the following equation [\(Zhang et al.,](#page-9-22) [2022\)](#page-9-22):

$$\nabla \cdot (p(x, y) \nabla u(x, y)) + k^2(x, y) = \lambda u(x, y),$$

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

Physical Contexts in which the Helmholtz operator appears: 1. Acoustics; 2. Electromagnetism; 3. Quantum Mechanics.

In Helmholtz operators, k is the wavenumber, related to the frequency of the wave and the properties of the medium in which the wave is propagating. In our experiment, p(x, y) and k(x, y) are derived using the GRF method. The parameters inherent to the GRF serve as the foundation for our sort scheme.

### B.3. Environment

To ensure consistency in our evaluations, all comparative experiments were conducted under uniform computing environments. Specifically, the environments used are detailed as follows:

- Platform: Docker version 4.33.1 (windows 11)
- Operating System: Ubuntu 22.04.3 LTS
- Processor: CPU AMD Ryzen 9 8945HS w, clocked at 4.00 GHz

## B.4. Experimental Parameter Configuration

All baseline methods were implemented using their default parameters from respective libraries.

For SCSF, the following configurations were adopted:

- The size of the inherited subspace varies according to the number of eigenvalues to be computed. Specifically, when calculating 20, 100, 200, 300, and 400 eigenvalues, the corresponding subspace sizes are set to 4, 20, 40, 60, and 80, respectively.
- The filter degree parameter m is consistently set to 20 across all experiments.

# C. Experimental Data and Supplementary Experiments

## C.1. Main Experimental Data

As shown in Tables [7,](#page-14-1) [6,](#page-13-3) [8,](#page-14-2) SCSF showed the best performance among all tested configurations

Table 6: Comparison of average computation times (in seconds) for eigenvalue problems using various algorithms on generalized Poisson operator dataset. The first row lists different algorithms, and the first column shows the number of eigenvalues L computed for each matrix. Matrix Dimension = 4900, Precision = 1e − 10.

| L   | Eigsh | LOBPCG | KS   | JD   | SCSF (ours) |
|-----|-------|--------|------|------|-------------|
| 150 | 9.15  | 46.8   | 14.9 | 138  | 7.95        |
| 200 | 14.2  | 73.0   | 23.8 | 270  | 12.9        |
| 250 | 19.8  | 109    | 34.3 | 553  | 19.0        |
| 300 | 26.3  | 152    | 45.6 | 921  | 25.7        |
| 350 | 31.5  | 203    | 58.4 | 1732 | 29.8        |
| 400 | 36.9  | 265    | 72.3 | 2691 | 33.9        |
| 450 | 42.8  | 342    | 87.3 | 3708 | 38.3        |

Table 7: Comparison of average computation times (in seconds) for eigenvalue problems using various algorithms on second-order elliptic operator dataset. The first row lists different algorithms, and the first column shows the number of eigenvalues L computed for each matrix. Matrix Dimension = 2500, Precision = 1e − 12.

| L   | Eigsh  | LOBPCG | KS     | JD      | SCSF (ours) |
|-----|--------|--------|--------|---------|-------------|
| 150 | 31.35  | 91.80  | 40.65  | 214.80  | 19.62       |
| 200 | 41.82  | 139.20 | 61.77  | 414.30  | 24.08       |
| 250 | 52.17  | 197.04 | 84.65  | 861.44  | 28.00       |
| 300 | 62.47  | 264.10 | 110.50 | 1446.00 | 29.88       |
| 350 | 74.59  | 355.18 | 147.01 | 2324.88 | 31.52       |
| 400 | 87.19  | 459.70 | 188.70 | 3386.00 | 34.60       |
| 450 | 100.28 | 577.67 | 235.56 | 4629.38 | 40.05       |

Table 8: Comparison of average computation times (in seconds) for eigenvalue problems using various algorithms on Helmholtz operator dataset. The first row lists different algorithms, and the first column shows the number of eigenvalues L computed for each matrix. Matrix Dimension = 6400, Precision = 1e − 8. The symbol '-' denotes data not recorded due to excessive computation times.

| L   | Eigsh  | LOBPCG  | KS     | JD      | SCSF (ours) |
|-----|--------|---------|--------|---------|-------------|
| 200 | 151.70 | 129.90  | 98.34  | 489.60  | 31.31       |
| 300 | 190.84 | 273.08  | 192.88 | 1601.08 | 37.78       |
| 400 | 253.50 | 460.40  | 283.00 | 3829.00 | 40.52       |
| 500 | 344.60 | 720.33  | 310.21 |         | 47.41       |
| 600 | 398.80 | 1031.00 | 329.60 |         | 51.32       |

#### C.2. Analysis of the Influence of Matrix Dimension

Table 9: Comparison of different algorithms' computation time (in seconds) for varying matrix dimensions using the generalized Poisson operator dataset. Results show average computation times for solving 400 eigenvalues with a precision of 1e − 12.

| Matrix Dimension | Eigsh   | LOBPCG  | KS      | JD      | SCSF (ours) |
|------------------|---------|---------|---------|---------|-------------|
| 2500             | 36.86   | 265.30  | 72.32   | 2691.00 | 33.91       |
| 3600             | 66.41   | 387.20  | 116.50  | 2990.00 | 65.41       |
| 4225             | 89.13   | 467.74  | 151.36  | 3548.13 | 70.79       |
| 4900             | 121.90  | 546.20  | 187.80  | 3886.00 | 74.23       |
| 5625             | 186.21  | 691.83  | 251.19  |         | 85.11       |
| 6400             | 282.80  | 860.00  | 337.70  |         | 93.86       |
| 8100             | 707.95  | 1412.54 | 707.95  |         | 114.82      |
| 10000            | 3162.28 | 2511.89 | 1995.26 |         | 158.49      |

As demonstrated in Table [9,](#page-14-3) the impact of matrix dimension on algorithm performance reveals several key insights. For matrices below dimension 3600, SCSF and Eigsh show comparable efficiency. However, SCSF's advantages become increasingly pronounced as matrix dimensions grow larger. At dimension 10000, SCSF achieves remarkable speedups: 20x faster than Eigsh, 16x faster than LOBPCG, and 13x faster than KS. This scaling behavior can be attributed to how larger matrix dimensions result in fewer errors and noise in the computed eigenvalues, allowing SCSF to better exploit similarities between problems. Additionally, the JD algorithm becomes computationally intractable at and above dimension 5625, while

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

SCSF maintains stable performance scaling even at high dimensions.

## C.3. Analysis of Computational Times for SCSF Components

Table 10: Analysis of Computational Times (in seconds) for SCSF Components

| All     | Lanczos (line3) | Filter (line 5) | QK (line 6) | RK (line 7) | Resid (line 8, 9) | Sort    |
|---------|-----------------|-----------------|-------------|-------------|-------------------|---------|
| 9.89e+0 | 4.04e-2         | 7.41e+0         | 3.12e-1     | 9.76e-1     | 7.95e-1           | 1.51e-2 |

We conducted a statistical analysis of the average time consumption for each component of the SCSF algorithm on the generalized Poisson operator dataset, with a matrix dimension of 2500 and the number of eigenvalues to be solved set to 100. The results are presented in Table [10.](#page-15-2) The notation "line x" within parentheses corresponds to line x in Algorithm [3,](#page-5-0) "ALL" denotes the total time consumption, and "sort" represents the average time required by the sorting algorithm. It is evident that the filtering process accounts for over 70% of the total time consumption, which aligns with our theoretical analysis in Section [4.2.](#page-4-0)

#### C.4. Analysis of Hyperparameters

Table 11: Average Computational Times (in seconds) of SCSF under Different Degree Parameters m.

| Deg      | 12    | 16    | 20    | 24    | 28    |
|----------|-------|-------|-------|-------|-------|
| Time (s) | 43.92 | 39.79 | 40.52 | 40.64 | 40.85 |
|          |       |       |       |       | 41.13 |
|          |       |       |       |       | 41.19 |
|          |       |       |       |       | 43.50 |
|          |       |       |       |       |       |
|          |       |       |       |       |       |
|          |       |       |       |       |       |
|          |       |       |       |       |       |

We investigated the impact of different degree parameters m on the performance of SCSF. As shown in Table [11,](#page-15-3) the experiments were conducted on the Helmholtz operator dataset with a matrix dimension of 6400, a solution accuracy of 1e-8, 400 eigenvalues to be solved, and an inherited subspace size of 80. The degree parameter m, as described in Algorithm [3,](#page-5-0) primarily controls the order of the Chebyshev polynomial. The results indicate that varying m within the range of 12 to 40 has a minimal effect on the computation time of SCSF. Therefore, as long as m is chosen within a reasonable range, its specific value does not significantly influence the performance. In the main experiments of this paper, m is fixed at 20.

Table 12: Average Computational Times (in seconds) of SCSF under Different Subspace Dimension.

| Dim      | 50    | 60    | 70    | 80    | 90    | 100   | 110   | 120   |
|----------|-------|-------|-------|-------|-------|-------|-------|-------|
| Time (s) | 42.38 | 44.35 | 42.43 | 40.52 | 39.65 | 37.43 | 38.28 | 38.58 |

We examine the influence of different inherited subspace sizes on the performance of SCSF. As presented in Table [12,](#page-15-4) the experiments are conducted on the Helmholtz operator dataset with a matrix dimension of 6400, a solution accuracy of 1e-8, 400 eigenvalues to be computed, and a degree parameter m set to 20.

The results demonstrate that as the inherited subspace size increases, the computation time of SCSF initially decreases and then rises, reaching its minimum around a size of 100. The reduction in computation time at the front end is attributed to the enriched initial subspace with more available information as the inherited subspace grows. Conversely, the increase in computation time at the back end is due to the significantly higher overhead of performing Chebyshev filtering with a larger inherited subspace.

Overall, as long as the inherited subspace size is set within a reasonable range, its impact on SCSF remains minimal. In our experiments, we consistently set the inherited subspace size to 20% of the number of eigenvalues to be computed.