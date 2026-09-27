011

014 015 016

018

024

026

034

036

038

054

# Consistent Multigroup Low-rank Approximation

### Anonymous Authors<sup>1</sup>

## Abstract

We consider the problem of consistent low-rank approximation for multigroup data: we ask for a sequence of k basis vectors such that projecting the data onto their spanned subspace treats all groups as equally as possible, by minimizing the maximum error among the groups. Additionally, we require that the sequence of basis vectors satisfies the natural consistency property: when looking for the best k vectors, the first d < k vectors are the best possible solution to the problem of finding d basis vectors. Thus, this multigroup lowrank approximation method naturally generalizes SVD and reduces to SVD for data with a single group. We give an iterative algorithm for this task that sequentially adds to the basis the vector that gives the best rank−1 projection according to the min-max criterion, and then projects the data onto the orthogonal complement of that vector. For finding the best rank−1 projection, we use primal-dual approaches or semidefinite programming. We analyze the theoretical properties of the algorithms and demonstrate empirically that the proposed methods compare favorably to existing methods for multigroup (or fair) PCA.

# 1. Introduction

Low-rank approximation techniques provide dimensionally reduced representations of data by expressing the data matrix as a linear combination of a small number of factors. Such methods are fundamental in machine learning and data science, due to the benefits they offer in terms of scalability, interpretability, and their strong mathematical foundation.

Among other methods, the singular value decomposition (SVD) holds a central position. A celebrated result states that the first d left or right singular vectors offer the best possible rank-d approximation to a matrix M in terms of Frobenius or spectral norm [\(Eckart & Young,](#page-8-0) [1936\)](#page-8-0). We call this the *consistency* property of the SVD.

In many applications, the rows of a data matrix are divided into two or more groups according to a particular attribute, e.g., gender. In such a case, using the top k right singular

vectors may not represent every group equally well, potentially resulting in inaccurate or even discriminatory outcomes. To address these concerns, previous works [\(Samadi](#page-8-1) [et al.,](#page-8-1) [2018;](#page-8-1) [Tantipongpipat et al.,](#page-8-2) [2019;](#page-8-2) [Song et al.,](#page-8-3) [2024\)](#page-8-3) have studied the problem of finding a *common* projection onto a subspace that minimizes the worst-case reconstruction error of any group. This problem is typically referred to as FAIR-PCA.

While effective, previous methods [\(Samadi et al.,](#page-8-1) [2018;](#page-8-1) [Tantipongpipat et al.,](#page-8-2) [2019;](#page-8-2) [Song et al.,](#page-8-3) [2024\)](#page-8-3) do not ensure the consistency property of the SVD, i.e., given a basis of a subspace, it is not possible to readily obtain a basis of a lower-rank subspace simply by discarding some vectors.

Building on this line of work, we introduce a *multigroup* lowrank approximation formulation which, in the spirit of the SVD, imposes the consistency property. More specifically, given a data matrix M with rows divided into groups G = {A<sup>1</sup> , . . . , A<sup>k</sup>}, we look for an orthonormal basis V for a subspace of the column space with the following properties: 1) projecting onto it minimizes the maximum possible error of any group (min-max criterion), 2) is consistent: given the best r vectors, the first d < r vectors from that solution are the best possible solution to the problem of finding d basis vectors. We call such vectors *multigroup singular vectors*.

Figure [1](#page-1-0) illustrates the concept of multigroup singular vectors. Figure 1 (a) shows the standard singular vectors (in red) and multigroup singular vectors (in green) in synthetic data. While standard singular vectors clearly favor the larger group over the smaller, multigroup singular vectors seek a more balanced representation. Figure [1](#page-1-0) (b) instead shows a comparison in the real-world compas dataset [\(Dua & Graff,](#page-8-4) [2017\)](#page-8-4). We consider the partitioning of the data into females and males. Projecting onto the multigroup singular vectors leads to a more balanced reconstruction error than projecting onto standard singular values, while giving a similar overall reconstruction error.

We empirically evaluate our method in the task of FAIR-PCA [\(Samadi et al.,](#page-8-1) [2018\)](#page-8-1). We show that it ensures the consistency property while incurring similar reconstruction error to the previous methods [\(Samadi et al.,](#page-8-1) [2018;](#page-8-1) [Tan](#page-8-2)[tipongpipat et al.,](#page-8-2) [2019;](#page-8-2) [Song et al.,](#page-8-3) [2024\)](#page-8-3). In addition, an obvious advantage of the consistency property is that it confers high efficiency and scalability: we can compute

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

108 109

![](_page_1_Figure_1.jpeg)

Figure 1: Left (a): synthetic data partitioned in two groups, as indicated by the color of the points. Singular vectors {w1, w2} and the multigroup singular vectors {v1, v2} given by our method, are also shown. Right (b): real-world compas dataset partitioned in two groups, females and males. The y-axis indicates the ratio of the average group-wise reconstruction error incurred by standard singular vectors and the multigroup singular vectors. The x-axis indicates the number of basis vectors. We additionally report the average reconstruction error across all data instances (both males and females).

each basis vector efficiently, and once the full-rank basis is computed, we can obtain lower-dimensional representations of any rank by just discarding basis vectors, as for SVD. This is in contrast to the previous approaches [\(Samadi et al.,](#page-8-1) [2018;](#page-8-1) [Tantipongpipat et al.,](#page-8-2) [2019;](#page-8-2) [Song et al.,](#page-8-3) [2024\)](#page-8-3) which require solving an independent, computationally challenging problem for any basis dimension.

The multigroup low-rank approximation problem still presents significant challenges. To ensure the consistency property holds, we construct the k-dimensional solution through an iterative process of solving simpler rank-1 problems. The more difficult aspect is proving that this procedure also yields the optimal k-dimensional result. Notably, we demonstrate that our solution is in fact optimal in the case of two groups in the data. For scenarios with more than two groups, we show that our solutions are empirically close to optimal in practice.

The contributions of this work can be summarized as follows.

- We formalize the consistent multigroup low-rank approximation problem.
- We give an iterative procedure which selects the best basis vector according to the min-max criterion, and then projects the data onto the orthogonal complement of the previously chosen vectors. The selection of the best basis vector at each iteration represents the main algorithmic challenge that we tackle.
- We theoretically analyze the formulated problems and the proposed algorithms, focusing on the two-groups case, which exhibits interesting properties.
- We describe extensive experiments on real-world datasets to demonstrate the benefits of consistent low-rank approx-

imation over previous work.

The rest of this paper is organized as follows. Section [2](#page-1-1) gives an overview of related work. Section [3](#page-2-0) gives necessary notations and definitions. Section [4](#page-2-1) describes our overall framework. Section [5](#page-3-0) formally introduces the multigroup singular vector problem (MG-SINGULARVECTOR), while Section [6](#page-4-0) proposes algorithms to solve it. We present a theoretical analysis and give an algorithm for a special case in Section [7,](#page-5-0) while Section [8](#page-5-1) contains our experimental evaluation and Section [9](#page-7-0) presents conclusions.

# 2. Related Work

We assume that the reader is familiar with singular value decomposition (SVD) and principal component analysis (PCA) (see, e.g., [\(Van Der Maaten et al.,](#page-8-5) [2009;](#page-8-5) [Eckart &](#page-8-0) [Young,](#page-8-0) [1936;](#page-8-0) [Hotelling,](#page-8-6) [1933\)](#page-8-6)).

Multigroup low-rank approximation: Fair PCA. Recently PCA has been extended to handle multigroup data. In this line of work, groups correspond to different values of a sensitive attribute (e.g., gender), and hence the proposed multigroup extension of PCA is referred to as FAIR PCA [\(Samadi et al.,](#page-8-1) [2018;](#page-8-1) [Tantipongpipat et al.,](#page-8-2) [2019;](#page-8-2) [Song](#page-8-3) [et al.,](#page-8-3) [2024\)](#page-8-3). In FAIR PCA, the goal is to retrieve a lowdimensional representation of the data that maximizes and balances variance for the groups. A similar problem has also been studied by [Zalcberg & Wiesel](#page-8-7) [\(2021\)](#page-8-7) from a signal processing perspective and by [Babu & Stoica](#page-8-8) [\(2023\)](#page-8-8). Other works have instead explored a significantly different formulation of the FAIR PCA problem. For instance, some works rely on notions of fairness such as *demographic parity* or *equal opportunity* that are adapted from supervised

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

learning [\(Olfat & Aswani,](#page-8-9) [2019;](#page-8-9) [Kleindessner et al.,](#page-8-10) [2023\)](#page-8-10). In a similar vein, [Lee et al.](#page-8-11) [\(2022\)](#page-8-11) define FAIR PCA as the problem of minimizing the maximum mean discrepancy between dimensionality-reduced conditional distributions of different classes. Instead, [Pelegrina & Duarte](#page-8-12) [\(2023\)](#page-8-12) as well as [Kamani et al.](#page-8-13) [\(2022\)](#page-8-13) formulate FAIR PCA as an optimization problem where the objective encodes the trade-off between reconstruction error and fairness.

Other multigroup dimensionality-reduction techniques. In recent years, significant attention has been devoted to algorithmic fairness and there have been efforts to extend traditional dimensionality-reduction techniques, beyond PCA. For instance, [Matakos et al.](#page-8-14) [\(2024\)](#page-8-14) and [Song et al.](#page-8-3) [\(2024\)](#page-8-3) study fair *column subset selection*, while [Louizos et al.](#page-8-15) [\(2016\)](#page-8-15) introduce the fair *variational autoencoder*.

# 3. Preliminaries

Notation. We denote matrices and vectors by bold uppercase and lowercase letters, respectively. The notation V<sup>d</sup> indicates the set of all matrices with d orthonormal columns, i.e., V<sup>d</sup> = {V ∈ <sup>R</sup> n×d : V⊤V = Id}, where I<sup>d</sup> is the d×d identity matrix.

For a matrix V ∈ Vd, we denote the ordered set of columns of V by {V} = {v1, . . . , vd}. The orthogonal complement of the span of the columns of V is denoted by V⊥. We write V:<sup>r</sup> ∈ <sup>R</sup> n×r for the matrix whose columns correspond to the first r columns of {V}. In addition, for a matrix A ∈ R <sup>a</sup>×<sup>n</sup> and matrix V ∈ <sup>R</sup> n×d that has orthonormal columns, the component of A in V<sup>⊥</sup> :r−1 is obtained as A − AV:r−1V<sup>⊤</sup> :r−1 . Finally, the first d singular values of matrix A, sorted in descending order, are denoted by σ1(A), . . . , σd(A). The Frobenius norm of a matrix A ∈ R <sup>m</sup>×<sup>n</sup> is defined as: ∥A∥<sup>F</sup> = qP<sup>m</sup> i=1 P<sup>n</sup> <sup>j</sup>=1 |aij | 2, where aij denotes the (i, j)-th element of A.

Orthogonal Projections. We briefly recall the properties of orthogonal projections. Given a matrix with orthonormal columns V ∈ R n×d , and vector x ∈ R <sup>n</sup>, the projection of x onto the column space of V is obtained as x <sup>⊤</sup>VV⊤.

Orthogonal projections satisfy the following property.

*Property* 1 (Orthogonal projection)*.* For any matrix A ∈ R <sup>m</sup>×<sup>n</sup> and a matrix V ∈ <sup>R</sup> <sup>n</sup>×<sup>d</sup> with orthonormal columns v1, . . . , vd, we have ∥AVV<sup>⊤</sup>∥ 2 <sup>F</sup> = P<sup>d</sup> <sup>i</sup>=1 ∥Aviv ⊤ i 2 F .

The proof is elementary, and we provide it in the appendix for completeness.

A fundamental property of the SVD is the *consistency* property, formally stated in the Eckart-Young-Mirsky theorem [\(Eckart & Young,](#page-8-0) [1936\)](#page-8-0). We state this pivotal theorem next. Given a matrix M ∈ R <sup>m</sup>×<sup>n</sup> and its singular value decomposition M = UΣV⊤, then for any

d = rank(Xd) ≤ rank(M) we have that the reconstruction error

$$\|\mathbf{M} - \mathbf{X}_d\|_{\xi}$$

is minimized by X<sup>d</sup> = MV:dV<sup>⊤</sup> :d , i.e. the projection of M onto the first d singular vectors. Here ξ denotes either the Frobenius norm (ξ = F) or the spectral norm (ξ = 2).

## 4. Overview of the Method

In this section we describe the multigroup low-rank approximation method. Our fundamental building block is the concept of a *multigroup* singular vector. A multigroup singular vector is rank-1 projection of the data, that takes all groups into account. Given a method to compute such a vector, it is fairly simple to obtain a *consistent* set of multigroup singular vectors by iteratively removing the component of the data that lies in the span of that vector.

First, we formally define the concept of a multigroup singular vector. We call the problem of finding such a vector MG-SINGULARVECTOR, and it represents the main algorithmic focus of this paper. Then, we give an algorithm for computing a consistent set of such vectors.

Multigroup singular vector. We seek to find a vector, such that the resulting rank-1 projection minimizes the maximum loss incurred by any group. The idea of minimizing the maximum per-group loss is inspired by the egalitarian rule in algorithmic fairness [\(Martinez et al.,](#page-8-16) [2020\)](#page-8-16). Assume an input matrix M ∈ R <sup>m</sup>×<sup>n</sup> with rows divided into groups G = {A<sup>1</sup> , . . . , A<sup>k</sup>}. A multigroup singular vector is the vector v minimizing the maximum loss over all groups of the difference between the largest singular value, σ1(A<sup>i</sup> ), and the rank-1 projection of A<sup>i</sup> using v. That is, v minimizes the loss L(M, v) defined as

$$\mathcal{L}(\mathbf{M}, \mathbf{v}) = \max_{\mathbf{A}^i \in \mathcal{G}} \{ \sigma_1^2(\mathbf{A}^i) - \|\mathbf{A}^i \mathbf{v} \mathbf{v}^\top\|_F^2 \}. \quad (1)$$

Recall that σ1(A<sup>i</sup> ), corresponds to the maximum norm of any rank-1 projection of A<sup>i</sup> . Subtracting from σ1(A<sup>i</sup> ) helps avoid bad minima of the minimization problem in Equation [1,](#page-2-2) by taking into account the best achievable rank-1 representation of every group. Since this loss function is a rank-1 version of the *marginal loss* of [Samadi et al.](#page-8-1) [\(2018\)](#page-8-1), we refer to Appendix [A](#page-9-0) and [Samadi et al.](#page-8-1) [\(2018\)](#page-8-1) for a broader discussion on this.

Computing a set of multigroup singular vectors. We are now ready to describe the iterative algorithm for obtaining a sequence of consistent multigroup singular vectors. The algorithm works as follows. We solve the rank-1 problem in Equation [1](#page-2-2) to obtain the multigroup singular vector v1. Given v1, we project the groups in G onto {v1} <sup>⊥</sup>, the orthogonal complement of v1. We repeat the same process on

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

Algorithm 1 Multigroup Orthonormalization

1: Input: Matrices {A<sup>1</sup> , . . . , A<sup>k</sup>}, rank d. 2: Initialize r ← 1, V ← ∅ 3: while r <= d do 4: v<sup>r</sup> ← MG-SINGULARVECTOR (A1, . . . , Ak) 5: A<sup>i</sup> ← A<sup>i</sup> − Aivrv ⊤ r 6: V ← V ∪ v<sup>r</sup> 7: r ← r + 1 8: end while return V

{v1} <sup>⊥</sup> to obtain a new vector v2. Repeating this process d times, we obtain an orthonormal basis V = {v1, . . . , vd}.

The whole iterative process is summarized in Algorithm [1.](#page-3-1) Step [1](#page-3-1) of the algorithm corresponds to a call to a subroutine, described in the following sections, that solves the MG-SINGULARVECTOR problem. Step [1](#page-3-1) illustrates the projection onto the orthogonal complement of the solution vector to the MG-SINGULARVECTOR problem.

Quality of the solutions. Since our algorithm iteratively produces orthonormal vectors, Property [1](#page-2-3) and a simple inductive argument imply that the loss for a solution of d dimensions is the sum of d losses of rank-1 solutions. Thus, the quality of our solution depends only on our ability to solve the rank-1 problem. Indeed, assume we are in step r of Algorithm [1:](#page-3-1) we have {V} = {v1, . . . , vr−1}, and we seek {V′} = {V} ∪ {vr}. Since ∥A<sup>i</sup>V′V′⊤∥ F , can be decomposed as ∥A<sup>i</sup>V′V′⊤∥ 2 <sup>F</sup> <sup>=</sup> ∥A<sup>i</sup>VV<sup>⊤</sup>∥ 2 <sup>F</sup> + ∥A<sup>i</sup>vrv ⊤ <sup>r</sup> ∥ 2 F , the problem reduces to solving a sequence of rank-1 problems. We will refer to the total error P<sup>d</sup> <sup>i</sup>=1 L(M, vi), as the *incremental error*.

In the following sections, we will show that, for the case of two groups, we can in fact efficiently solve the rank-1 problem to optimality, and for the general case we will give an approximate algorithm which works well in practice.

Complexity. The overall time complexity is O(dℓ), where O(ℓ) is the complexity of solving MG-SINGULARVECTOR that is discussed later.

# 5. The Multigroup Singular Vector Problem

As anticipated in Section [4,](#page-2-1) solving the MG-SINGULARVECTOR problem represents the crucial algorithmic challenge to be addressed for multigroup low-rank approximation. In this section, we study the properties of the MG-SINGULARVECTOR problem, as well as of its dual problem, which is more amenable to optimization. Leveraging the insights gained in the present section, in the next section we introduce algorithms to solve the MG-SINGULARVECTOR problem.

Next, we formalize the MG-SINGULARVECTOR problem. *Problem* 1 (MG-SINGULARVECTOR)*.* Given a matrix M ∈ R <sup>m</sup>×<sup>n</sup> with rows divided into groups G = {A<sup>1</sup> , . . . , A<sup>k</sup>}, find the vector v satisfying

$$\begin{aligned} & \min_{\mathbf{v} \in \mathbb{R}^n, z \in \mathbb{R}} z \\ & \text{s.t. } \sigma_1^2(\mathbf{A}^i) - \|\mathbf{A}^i \mathbf{v} \mathbf{v}^\top\|_F^2 \leq z \text{ for all } \mathbf{A}^i \in \mathcal{G} \\ & \text{and } \|\mathbf{v}\|_2^2 = 1. \end{aligned}$$

We use the term *constraint functions* hi(v) for the left-hand sides of the constraints in the problem:

$$h_i(\mathbf{v}) = \sigma_1^2(\mathbf{A}^i) - \|\mathbf{A}^i \mathbf{v} \mathbf{v}^\top\|_F^2.$$

Convexity analysis. Problem [1](#page-3-2) is not a convex problem. To see this, note that −∥Avv<sup>⊤</sup>∥ 2 <sup>F</sup> = −v <sup>⊤</sup>A⊤Av, and since −A⊤A is a negative semidefinite matrix, the corresponding quadratic forms are concave functions. Each hi(v) consists of such a quadratic form and an affine transformation (which does not impact convexity), and is thus concave. Minimization problems over concave functions are non-convex and not straightforward to solve.

We also note that the constraint functions h<sup>i</sup> are continuous functions supported on the unit hypersphere. Importantly, all their minima are at zero, since the incremental loss attains its minimum at 0. Given these observations, we can prove that for any optimal solution v ∗ , z<sup>∗</sup> , two groups attain exactly the same error, while other groups smaller or equal error.

Theorem 5.1. *For an optimal solution* v ∗ , z<sup>∗</sup> *to Problem [1](#page-3-2) we have:*

$$z^* = h_i(\mathbf{v}^*) = h_j(\mathbf{v}^*) \geq h_k(\mathbf{v}^*),$$

*for some* i ̸= j *and for all* k ̸= i, j*.*

*Proof.* We first prove that there exist two groups such that z <sup>∗</sup> = hi(v ∗ ) = h<sup>j</sup> (v ∗ ). Assume for the sake of contradiction that v ∗ is an optimal solution such that z <sup>∗</sup> = hi(v ∗ ) > h<sup>j</sup> (v ∗ ) for all j. Then, this implies that hi(v ∗ ) > 0, and v ∗ cannot be a minimizer of h<sup>i</sup> since the minima of h<sup>l</sup> for all l, are at hl(v) = 0. Thus we can locally move to a nearby solution v<sup>ϵ</sup> such that hi(vϵ) < hi(v ∗ ) and at the same time h<sup>j</sup> (vϵ) ≤ hi(vϵ) for all j. This contradicts the fact that v ∗ is an optimal solution. Additionally it must be that hk(v ∗ ) ≤ hi(v ∗ ) = h<sup>j</sup> (v ∗ ) since hi(v ∗ ) = h<sup>j</sup> (v ∗ ) attain the optimal value z ∗ .

As we stated before, Problem [1](#page-3-2) is easier when there are two groups. The proof hints at an interesting geometric intuition for why that is the case. In this setting, we have two quadratic constraint functions h<sup>1</sup> and h2, and as Theorem [5.1](#page-3-3) suggests, the candidate optimal solutions v lie at

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

the intersection points of two ellipsoids determined by the quadratic equation h<sup>1</sup> = h2. Thus, it suffices to start from the minimum of either h<sup>1</sup> or h<sup>2</sup> (note that this minimum is found by setting v to the leading eigenvector of either group 1 or 2) and follow the direction of steepest descent of the objective function, to find the global minimum (due to symmetry it can be either v <sup>∗</sup> or −v ∗ ). We will formalize this intuition by characterizing the KKT points, in Section [7,](#page-5-0) where we show that in the two-group case the problem enjoys strong duality. Indeed, this is not surprising, as several strong results exist for non-convex problems with two quadratic constraints (we refer to [\(Boyd & Vandenberghe,](#page-8-17) [2004\)](#page-8-17), Appendix B). We now proceed to define the dual problem for the general case.

The dual problem. To study Problem [1,](#page-3-2) it is useful to consider the *dual* problem. An advantage of the dual problem over the primal problem (Problem [1\)](#page-3-2) is that it leads to an objective function with a gradient that is more "informative" and more convenient for gradient-based methods. Methods such as Frank-Wolfe [\(Frank et al.,](#page-8-18) [1956\)](#page-8-18), use the gradient to determine the search direction in the feasible region.

As already mentioned, we show that for k = |G| = 2 Problem [1](#page-3-2) exhibits strong duality (i.e., the optimal value of the primal problem equals the optimal value of the dual problem), despite being non-convex.

However, even for |G| > 2, the dual problem is still useful in practice, since we can assess the quality of our solution by evaluating the difference between the primal and dual optimal objective values.

To formulate the dual problem we will consider the Lagrangian function corresponding to Problem [1.](#page-3-2) The Lagrangian is obtained by adding the problem constraints to the objective, along with the dual variables, which correspond to the Lagrange *multipliers*. In particular, the Lagrangian function associated with Problem [1](#page-3-2) is:

$$\mathcal{H}(\mathbf{v}, z, \boldsymbol{\mu}, \lambda) = z + \sum_{i=1}^k \mu_i(h_i(\mathbf{v}) - z) + \lambda(\|\mathbf{v}\|_2^2 - 1),$$

where we denote µ = [µ1, . . . , µk]. Further, let

$$\mathbf{A}(\boldsymbol{\mu}) = \sum_{i=1}^k \mu_i (\mathbf{A}^i)^\top \mathbf{A}^i,$$

and define s = [σ 2 1 (A<sup>1</sup> ), . . . , σ<sup>2</sup> 1 (A<sup>k</sup> )]. The dual problem associated with Problem [1](#page-3-2) is the following.

*Problem* 2 (MG-SINGULARVECTOR-DUAL)*.*

max µ∈R<sup>k</sup> µ

<sup>⊤</sup>s − λmax(A(µ))

s.t. 
$$\mathbf{1}^\top \boldsymbol{\mu} = 1$$
 (2)

$$\mu \geq 0. \quad (3)$$

Here, λmax denotes the maximum eigenvalue. A detailed derivation of Problem [2](#page-4-1) is given in the appendix.

Problem [2](#page-4-1) is convex and has an interesting interpretation as a parametric eigenvalue problem: the solution vector v is the leading eigenvector of the optimal convex combination A(µ), determined by the coefficients µ.

Uniqueness. Later on, we define the solution v which we obtain from the dual as a function of µ, which requires uniqueness. However, in general, v is not unique, as A(µ) may have repeated eigenvalues. This is not a problem in practice since real data contain noise, which leads to distinct eigenvalues [\(Kato,](#page-8-19) [1966\)](#page-8-19). In any case, it is always possible to slightly perturb the data to avoid ill-conditioned scenarios with repeated eigenvalues.

# 6. Algorithms for Multigroup Singular Vector

In this section we present two algorithms for MG-SINGULARVECTOR. The first algorithm solves the dual problem MG-SINGULARVECTOR-DUAL, which is a convex optimization problem with linear constraints, using the Frank-Wolfe algorithm. The second one solves a semidefinite programming (SDP) relaxation of the primal problem.

Frank-Wolfe. The Frank-Wolfe algorithm is a widely-used iterative algorithm for solving constrained convex optimization problems [\(Pokutta,](#page-8-20) [2023\)](#page-8-20). In each iteration, the algorithm linearizes the objective function, and moves towards its minimizer , while staying inside the feasible region.

The Frank-Wolfe algorithm is particularly easy to use for Problem [2,](#page-4-1) as the dual constraints are almost trivial to satisfy and thus the only computationally challenging aspect for the algorithm is the computation of the gradient ∇g of the dual objective, g(µ) = µ <sup>⊤</sup>s − λmax(A(µ)), which involves computing the gradient of λmax(A(µ)).

Denoting for brevity λ(µ) = λmax(A(µ)), we have that λ(µ) is an eigenvalue of A(µ) and hence:

$$\mathbf{A}(\mu)\mathbf{v}(\mu) = \lambda(\mu)\mathbf{v}(\mu), \quad (4)$$

where v(µ) is the eigenvector corresponding to λ(µ). Taking the gradient and using the product rule, we have:

$$(\mathbf{A}^i)^\top \mathbf{A}^i \mathbf{v}(\boldsymbol{\mu}) + \mathbf{A}(\boldsymbol{\mu}) \nabla \mathbf{v}(\boldsymbol{\mu}) = \nabla \lambda(\boldsymbol{\mu}) \mathbf{v}(\boldsymbol{\mu}) + \lambda(\boldsymbol{\mu}) \nabla \mathbf{v}(\boldsymbol{\mu}). \quad (5)$$

To simplify the gradient, we use the constraint v(µ) <sup>⊤</sup>v(µ) = 1. This gives:

$$\nabla \mathbf{v}(\boldsymbol{\mu})^\top \mathbf{v}(\boldsymbol{\mu}) + \mathbf{v}(\boldsymbol{\mu}) \nabla \mathbf{v}(\boldsymbol{\mu}) = 0,$$

i.e., v(µ) is orthogonal to its gradient. Therefore, multiplying equation [5](#page-4-2) with v(µ) <sup>⊤</sup>, we obtain:

$$\begin{aligned} & \mathbf{v}(\boldsymbol{\mu})^\top (\mathbf{A}^i)^\top \mathbf{A}^i \mathbf{v}(\boldsymbol{\mu}) + \lambda(\boldsymbol{\mu}) \mathbf{v}(\boldsymbol{\mu})^\top \nabla \mathbf{v}(\boldsymbol{\mu}) \\ &= \nabla \lambda(\boldsymbol{\mu}) \mathbf{v}(\boldsymbol{\mu})^\top \mathbf{v}(\boldsymbol{\mu}) + \lambda(\boldsymbol{\mu}) \mathbf{v}(\boldsymbol{\mu})^\top \nabla \mathbf{v}(\boldsymbol{\mu}), \end{aligned}$$

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

Algorithm 2 Frank-Wolfe for MG-SINGULARVECTOR-DUAL

- 1: Input: Matrices A<sup>1</sup> , . . . , A<sup>k</sup> , convergence tolerance ϵ. 2: Initialize: Set µ
- (0) = [1, 0, . . . , 0], s = [σ 2 1 (A<sup>1</sup> ), . . . , σ<sup>2</sup> 1 (A<sup>k</sup> 3: t ← 0 4: repeat 5: v(µ
  - (t) ) ← x s.t. A(µ
- (t) )x = λmaxx 6: ∇g(µ
  - (t) )<sup>i</sup> ← s<sup>i</sup> + v(µ
- (t) ) <sup>⊤</sup>(A<sup>i</sup> ) <sup>⊤</sup>A<sup>i</sup>v(µ
- (t) ) 7: s
  - (t) ← arg maxy:1⊤y=1,y≥<sup>0</sup> y <sup>⊤</sup>∇g(µ
- (t) ) 8: γ<sup>t</sup> ← <sup>2</sup> t+2 9: µ (t+1) ← (1 − γt)µ
  - (t) + γts
- (t) 10: t ← t + 1 11: until ∥µ
- (t) − µ (t−1)∥ < ϵ 12: return µ
  - (t) , v(µ
- (t) )

which simplifies to (∇λ(µ))<sup>i</sup> = v(µ) <sup>⊤</sup>(A<sup>i</sup> ) <sup>⊤</sup>A<sup>i</sup>v(µ). Putting everything together, we conclude that:

$$(\nabla g)_i = \mathbf{s}_i - \mathbf{v}(\boldsymbol{\mu})^\top (\mathbf{A}^i)^\top \mathbf{A}^i \mathbf{v}(\boldsymbol{\mu}). \quad (6)$$

Algorithm [1](#page-3-1) contains the pseudocode of the Frank-Wolfe method for Problem [2.](#page-4-1) The algorithm proceeds as follows. It starts with an initial feasible solution µ (0). In each step, line 5 solves the maximum eigenvalue problem associated with the given parameter vector µ (t) . Line 6 computes the gradient. Line 7 solves a linear maximization problem over the simplex defined by constraints [2](#page-4-3) and [3](#page-4-4) in Problem [2.](#page-4-1) Lines 8-9 describe standard parameter update steps of the Frank-Wolfe algorithm. Finally, the returned solution is the v(µ) for the final update and the corresponding dual solution µ. The complexity of the algorithm is dominated by the maximum eigenvalue step (Line 5) which can be handled using a fast Lanczos implementation. Thus the overall complexity is O(tn<sup>2</sup> ) where t is the number of iterations until convergence.

Algorithm [2](#page-5-2) solves Problem [2](#page-4-1) optimally, as it is a convex problem. However, the value g(µ) of the dual objective is only a lower bound on the primal objective (i.e., Problem [1\)](#page-3-2) i.e., there can be a non-zero duality gap.

Semidefinite programming. We also solve MG-SINGULARVECTOR through a semidefinite programming (SDP) relaxation [\(Boyd & Vandenberghe,](#page-8-17) [2004\)](#page-8-17). Since SDP solvers come with an O(n 6 ) running time, this algorithm is expected to be significantly slower than Algorithm [2.](#page-5-2) However, since for |G| > 2 we are not guaranteed to solve MG-SINGULARVECTOR exactly, the SDP relaxation may offer a solution that is close to rank-1, and hence close to optimal. As a consequence, this approach can be useful in settings where accuracy is more important than efficiency. The pseudocode of the SDP for solving Problem [1](#page-3-2) is provided in the appendix (Algorithm [3\)](#page-11-0) where we also present

experiments demonstrating more accurate approximation of the primal optimum compared to Algorithm [2.](#page-5-2)

## 7. Algorithm and Analysis for Two Groups

Often, the data are divided into exactly two groups, e.g., on the basis of sex. As mentioned, in this particular case, we are able to solve Problem [1](#page-3-2) optimally and, moreover, the optimal solution equalizes the loss.

Algorithm. We give a novel algorithm dedicated to the two-group case, which outperforms competitors (such as the Frank-Wolfe algorithm) in this setting, but cannot be conveniently extended to address the setting of more than two groups. The algorithm relies on the observation that for |G| = 2 there exists a unique feasible µ for the dual, which satisfies Theorem [5.1.](#page-3-3) We can find such a µ using a fast root-finding approach, such as Brent's method [\(Brent,](#page-8-21) [1971\)](#page-8-21). This algorithm is evaluated in our experiments and its details are in the appendix (see Lemma [E.1\)](#page-13-0).

Theoretical Analysis. The case of |G| = 2 has interesting theoretical properties, which we present here. All the proofs can be found in the appendix. First, we observe that, as a consequence of Theorem [5.1,](#page-3-3) it holds that h1(v ∗ ) = h2(v ∗ ) for any optimal solution v ∗ to Problem [1.](#page-3-2)

Furthermore, leveraging the KKT conditions (see, e.g., [\(Kuhn & Tucker,](#page-8-22) [2013\)](#page-8-22)) to characterize the optimal solutions to Problem [2](#page-4-1) leads to the following theorem.

Theorem 7.1. *For* |G| = 2*, the optimal solution to* MG-SINGULARVECTOR *can be computed in polynomial time.*

The proof relies on a simple idea. Since Problem [2](#page-4-1) is convex, it has a unique maximum, which can be found in polynomial time (for example, using the approach based on the Frank-Wolfe algorithm). Such a unique maximum can be characterized by the KKT conditions. Then, to complete the proof, it suffices to show that Problems [1](#page-3-2) and [2](#page-4-1) attain strong duality, i.e., f(v ∗ ) = z <sup>∗</sup> = g(µ ∗ ), where v ∗ and µ ∗ are optimal solutions to Problems [1](#page-3-2) and [2,](#page-4-1) respectively.

Our analysis reveals interesting properties which are described in the following lemmas and proved in the appendix. Lemma 7.2. *For* |G| = 2*, the SDP relaxation in Algorithm [3](#page-11-0) is tight.*

Finally, the following lemma related to Algorithm [1](#page-3-1) follows. Lemma 7.3. *For* |G| = 2*, an optimal solution of Algorithm [1](#page-3-1) is such that the total error for the two groups is equal.*

# 8. Experiments

This section presents our experimental evaluation, which aims at assessing the performance of our method (Algorithm

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

Table 1: Dataset statistics. For each dataset, we report the number of columns (n), the number of groups (|G|), and the number of rows and rank by group.

| Dataset       | Columns ( n ) | G     | Group       | Rows    | Group     | Ranks |
|---------------|---------------|-------|-------------|---------|-----------|-------|
| heart         | 14            | 2     | 201,        | 96      | 13,       | 13    |
| german        | 63            | 2     | 690,        | 310     | 49,47     |       |
| credit        | 25            | 2     | 18 112, 11  | 888     | 24,       | 24    |
| student       | 58            | 2     | 383,        | 266     | 42,       | 42    |
| adult         | 109           | 2     | 21 790, 10  | 771     | 98,       | 98    |
| compas        | 189           | 2     | 619,        | 100     | 165,      | 71    |
| communities   | 104           | 2     | 1 685,      | 309     | 101,      | 101   |
| recidivism    | 227           | 2     | 1 923,      | 310     | 175,      | 113   |
| compas-3      | 189           | 3     | 241, 240,   | 238     | 115, 110, | 97    |
| communities-4 | 104           | 4 90, | 1 571, 218, | 115 90, | 99, 103,  | 103   |

[1\)](#page-3-1) in the FAIR-PCA task. Exploring other applications is left to future work. We refer to the multigroup singular vectors output by Algorithm [1](#page-3-1) as MULTIGROUP SVS.

The experiments consider both the two-group case, where our methods are supported by optimality guarantees, and the case of more than two groups, where the optimality guarantees no longer hold, but we observe that, in practice, the gap between primal and dual solutions is consistently small and hence they are close to optimal (see Appendix [D\)](#page-10-0). The results show that our method can offer significant advantages over recent methods for FAIR-PCA.

## 8.1. Settings

Next, we illustrate the datasets, metrics, baselines, parameter settings and experimental setup used in our experiments.

### Datasets.

- Datasets with two groups, We use the juvenile recidivism data (recidivism) from Catalunya [\(Tolan et al.,](#page-8-23) [2019\)](#page-8-23) and various datasets from the UCI machine learning repository [\(Dua & Graff,](#page-8-4) [2017\)](#page-8-4): "heartcleveland" (heart), "german-credit" (german), "creditcard" (credit), "student performance" (student), "adult" (adult), "compas-recidivism" (compas), "communities" (communities). Group membership is based on sex, except for "communities " where groups determined by racial composition (caucasian majority or not).
- Datasets with more than two groups. We consider the "compas-recidivism" dataset partitioned into three groups according to age (compas-3), and the "communities" dataset partitioned into four groups, namely "blacks", "hispanics", "asians" and "caucasians", according to the dominant ethnicity (communities-4).

Data are processed by removing protected attributes, onehot encoding categorical variables, and standardizing columns. Table [1](#page-6-0) shows summary statistics of the datasets.

[pipat et al.,](#page-8-2) [2019\)](#page-8-2) and against the BICRITERIA algorithm [\(Song et al.](#page-8-3) [\(2024\)](#page-8-3), Algorithm 3). Given target rank d, FAIR-PCA-SDP and BICRITERIA return a rank-d projection matrix P = UΛU⊤, where U ∈ <sup>R</sup> n×d is obtained through SVD. In our experiments, we evaluate the consistency property by comparing the loss when using U:<sup>r</sup> of FAIR-PCA-SDP and BICRITERIA against V:<sup>r</sup> retrieved by Algorithm [1,](#page-3-1) for all r < d.

Metrics and parameters. To evaluate the performance of our method and FAIR-PCA-SDP, we monitor the marginal loss (introduced in [Samadi et al.](#page-8-1) [\(2018\)](#page-8-1)), incremental loss, (see Section [4\)](#page-2-1), and the standard L<sup>2</sup> reconstruction loss.

Both the marginal and incremental losses quantify the deviation from the optimal reconstruction, whereas the L<sup>2</sup> reconstruction loss does not account for such optimal reconstruction. As the BICRITERIA algorithm [\(Song et al.,](#page-8-3) [2024\)](#page-8-3) is designed to optimize the L<sup>2</sup> reconstruction loss only, it is not competitive with our method and FAIR-PCA-SDP in terms of marginal and incremental loss.

We show the variation of each loss in the groups as a function of the (target) reconstruction rank d, which we vary from 1 to 8. Finally, we measure runtimes in seconds.

Experimental setup. Our implementation is written in python. In the two-groups case, the singular vectors for multigroup data are obtained by the tailored algorithm based on the root-finding procedure, while for more than two groups, they are obtained by the Frank-Wolfe algorithm.

Experiments are executed on a compute node with 32 cores and 256GB of RAM. The (anonymized) source code is available online [<sup>1</sup>](#page-6-1) .

## 8.2. Results for Two-group Data

Figure [2](#page-7-1) (top) shows the different losses incurred by our method and the baselines in the compas datasets as a function of the target rank d. Due to the space limitations, analogous results for all the other datasets are presented in the appendix (Figure [3\)](#page-14-0). The figure highlights the crucial advantage of MULTIGROUP SVS: the incremental loss is the same in both groups for all values of the rank parameter lower than the input target rank (8), meaning that fairness is also pursued in the lower-dimensional subspaces. In particular, the incremental loss is considerably smaller for MULTI-GROUP SVS than for FAIR-PCA-SDP. On the other hand, the marginal loss optimized by FAIR-PCA-SDP is never significantly smaller for FAIR-PCA-SDP than for MULTI-GROUP SVS, but tends to be smaller for MULTIGROUP SVS. Finally, the reconstruction loss is consistently comparable for FAIR-PCA-SDP and MULTIGROUP SVS, but tends

<sup>1</sup>[https://anonymous.4open.science/r/](https://anonymous.4open.science/r/multigroupSVs-F716/) [multigroupSVs-F716/](https://anonymous.4open.science/r/multigroupSVs-F716/)

394

396

![](_page_7_Figure_1.jpeg)

Figure 2: compas dataset with two groups (top) and compas-3 dataset with three groups (bottom). Marginal, incremental, and reconstruction loss by rank. Different marker symbols indicate different groups.

Table 2: Runtimes of MULTIGROUP SVS, FAIR-PCA-SDP and BICRITERIA (in seconds) in all datasets (d = 8).

| Dataset       | M ULTIGROUP SV S | FAIR-PCA-SDP | B ICRITERIA |
|---------------|------------------|--------------|-------------|
| heart         | 0.009            | 0.022        | 0.016       |
| german        | 0.1              | 0.9          | 0.021       |
| credit        | 0.23             | 0.084        | 0.053       |
| student       | 0.067            | 0.64         | 0.031       |
| adult         | 2.16             | 9.13         | 0.2         |
| compas        | 0.71             | 143.15       | 0.053       |
| communities   | 0.28             | 8.62         | 0.035       |
| recidivism    | 1.28             | 357.59       | 0.061       |
| compas-3      | 2.54             | 124.11       | 0.019       |
| communities-4 | 1.23             | 11.16        | 0.024       |

to be larger for BICRITERIA. Unlike the incremental and marginal losses, the reconstruction loss can be highly unbalanced since both MULTIGROUP SVS and FAIR-PCA-SDP do not seek to balance the reconstruction loss, but rather the distance to the best possible approximation.

In addition, Table [2](#page-7-2) shows the runtime of MULTIGROUP SVS and the baselines in the different datasets. BICRITE-RIA is typically the fastest algorithm. However, it is not competitive with MULTIGROUP SVS and FAIR-PCA-SDP in terms of performance, even for reconstruction loss. On the other hand, FAIR-PCA-SDP becomes slow as dataset size increases, and MULTIGROUP SVS is generally faster that FAIR-PCA-SDP, often by orders of magnitude. MULTI-GROUP SVS always deliver high-quality results in terms of all the metrics under consideration in less than three seconds.

## 8.3. Results for more than two groups

In case there are more than two groups, the problems solved by the algorithms under comparison become NP-hard, and the algorithms drop the optimality guarantees.

As Figure [2](#page-7-1) (bottom) shows for the compas-3 dataset, MULTIGROUP SVS consistently yield a more balanced lowdimensional data representation than FAIR-PCA-SDPand BICRITERIA as the rank increases. This observation suggests that MULTIGROUP SVS provides an effective heuristic for multigroup dimensionality reduction with an arbitrary number of groups |G| > 2. Results of the same experiments for the communities-4 dataset, given in Figure [4](#page-15-0) in the appendix, lead to analogous observations. Figures [2](#page-7-1) (bottom) and [4](#page-15-0) also show that, as discussed, when there are more than two groups in the data, there is no guarantee of equality in the marginal and incremental losses associated with different groups.

Nevertheless, in the appendix (Figures [5](#page-15-1) and [6\)](#page-15-2), we present experimental results demonstrating that the gap between primal and dual solutions in practice tends to be negligible, and thus our solutions tend to be close to optimal.

To conclude, Table [2](#page-7-2) also reports the runtimes in the experiments with more than two groups, which confirm the trends observed in the two-group case.

## 9. Conclusion

We have introduced the problem of consistent multigroup low-rank approximation that, given a dataset partitioned into groups, asks for a sequence of orthonormal vectors such that projecting the data onto their spanned subspace minimizes the maximum error across groups, and such that any subsequence is also an optimal solution of smaller length.

We have proposed efficient and theoretically well-founded methods to compute the desired sequence of vectors. Extensive experiments highlight the advantages of our methods over existing approaches.

- 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Babu, P. and Stoica, P. Fair principal component analysis (pca): minorization-maximization algorithms for fair pca, fair robust pca and fair sparse pca. *arXiv preprint arXiv:2305.05963*, 2023. Boyd, S. and Vandenberghe, L. *Convex optimization*. Cambridge university press, 2004. Brent, R. P. An algorithm with guaranteed convergence for finding a zero of a function. *The computer journal*, 14(4): 422–425, 1971. Dua, D. and Graff, C. UCI machine learning repository, 2017. URL <http://archive.ics.uci.edu/ml>. Eckart, C. and Young, G. The approximation of one matrix by another of lower rank. *Psychometrika*, 1(3):211–218, 1936. Ehiwario, J. and Aghamie, S. Comparative study of bisection, newton-raphson and secant methods of root-finding problems. *IOSR Journal of Engineering*, 4(4):01–07, 2014. Frank, M., Wolfe, P., et al. An algorithm for quadratic programming. *Naval research logistics quarterly*, 3(1-2): 95–110, 1956. Hotelling, H. Analysis of a complex of statistical variables into principal components. *Journal of educational psychology*, 24(6):417, 1933. Kamani, M. M., Haddadpour, F., Forsati, R., and Mahdavi,
  - M. Efficient fair principal component analysis. *Machine Learning*, pp. 1–32, 2022. Kato, T. *Perturbation Theory for Linear Operators*. 1966. Kleindessner, M., Donini, M., Russell, C., and Zafar, M. B. Efficient fair pca for fair representation learning. In *International Conference on Artificial Intelligence and Statistics*, pp. 5250–5270. PMLR, 2023. Kuhn, H. W. and Tucker, A. W. Nonlinear programming. In *Traces and emergence of nonlinear programming*, pp. 247–258. Springer, 2013. Lee, J., Kim, G., Olfat, M., Hasegawa-Johnson, M., and Yoo, C. D. Fast and efficient mmd-based fair pca via optimization over stiefel manifold. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, pp. 7363–7371, 2022. Louizos, C., Swersky, K., Li, Y., Welling, M., and Zemel,
- R. S. The variational fair autoencoder. In *4th International Conference on Learning Representations, ICLR, San Juan, Puerto Rico*, 2016. Martinez, N., Bertran, M., and Sapiro, G. Minimax pareto fairness: a multi objective perspective. In *Proceedings of the 37th International Conference on Machine Learning*, ICML'20. JMLR.org, 2020. Matakos, A., Ordozgoiti, B., and Thejaswi, S. Fair column subset selection. In *Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, KDD '24, pp. 2189–2199, New York, NY, USA, 2024. Association for Computing Machinery. Olfat, M. and Aswani, A. Convex formulations for fair principal component analysis. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 33, pp. 663– 670, 2019. Pelegrina, G. D. and Duarte, L. T. A novel approach for fair principal component analysis based on eigendecomposition. *IEEE Transactions on Artificial Intelligence*, 2023. Pokutta, S. The frank-wolfe algorithm: A short introduction. *Jahresbericht der Deutschen Mathematiker-Vereinigung*, 2023. Samadi, S., Tantipongpipat, U., Morgenstern, J., Singh, M., and Vempala, S. The price of fair pca: One extra dimension. In *NeuRIPS*, NIPS'18, pp. 10999–11010. Curran Associates Inc., 2018. Schiffer, M. Hadamard's formula and variation of domainfunctions. *American Journal of Mathematics*, 68(3):417– 448, 1946. Song, Z., Vakilian, A., Woodruff, D., and Zhou, S. On socially fair low-rank approximation and column subset selection. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024. Tantipongpipat, U. T., Samadi, S., Singh, M., Morgenstern, J., and Vempala, S. Multi-criteria dimensionality reduction with applications to fairness. In *NIPS*, Red Hook, NY, USA, 2019. Curran Associates Inc. Tolan, S., Miron, M., Gomez, E., and Castillo, C. Why ´ machine learning may lead to unfairness: Evidence from risk assessment for juvenile justice in catalonia. In *International Conference on Artificial Intelligence and Law*, pp. 83–92, 2019. Van Der Maaten, L., Postma, E. O., Van Den Herik, H. J., et al. Dimensionality reduction: A comparative review. *Journal of Machine Learning Research*, 10(66-71):13, 2009. Zalcberg, G. and Wiesel, A. Fair principal component analysis and filter design. *IEEE Transactions on Signal Processing*, 69:4835–4842, 2021.

## References


[1] 504

[2] 506

508 509

[4] 511

514 515 516

[6] 518

[7] 524

[8] 526

[9] 528

[10] 531

[11] 534

[12] 536

[13] 538
## A. Loss Functions

In this section, we discuss widely used loss functions.

An equivalence that we will frequently use is σi(M) = ∥Mviv ⊤ i ∥<sup>F</sup> , where v<sup>i</sup> is the i-th singular vector of M.

#### A.1. Reconstruction Error

A commonly used and natural loss function for a given group is the reconstruction error.

Definition A.1 (Reconstruction Error). Given matrix A ∈ R <sup>a</sup>×<sup>n</sup> and an n × d matrix V ∈ Vd, the reconstruction error of A using V is

$$\mathcal{L}_{\text{rec}}(\mathbf{A}, \mathbf{V}) \triangleq \|\mathbf{A} - \mathbf{A}\mathbf{V}\mathbf{V}^\top\|_F^2 = \sum_{i=1}^n \sigma_i^2(\mathbf{A}) - \|\mathbf{A}\mathbf{V}\mathbf{V}^\top\|_F^2,$$

where the equivalence holds due to the properties of projection matrices.

However, the reconstruction error has a serious limitation when considering multiple groups [\(Samadi et al.,](#page-8-1) [2018\)](#page-8-1). To explain this, imagine that we are given a data matrix with two groups M = {A, B}, and WA, W<sup>B</sup> the corresponding minimizers of Lrec(A, ·) and Lrec(B, ·), for some rank d. We can obtain W<sup>A</sup> and W<sup>B</sup> from the SVD of A and B accordingly. Now consider that Lrec(A,W<sup>A</sup>) = ℓ<sup>A</sup> >> Lrec(B,W<sup>B</sup>) = ℓB, i.e., the best possible rank-d reconstruction error for B is much better than the best possible reconstruction error for A. We can see that this puts a lower bound of ℓ<sup>A</sup> to the loss. This means that any improvement to the reconstruction error of B, beyond ℓA, cannot improve the objective. This may be considered unfair to group B, since it suffers from a high reconstruction error only due to the fact that group A cannot be reconstructed well enough in a rank d subspace.

## A.2. Marginal Loss

[Tantipongpipat et al.](#page-8-2) [\(2019\)](#page-8-2) consider a family of problems under the term *multicriteria dimensional reduction*, where the task is to find a subspace that takes into account various groups present in the data, in a balanced manner.

*Problem* 3 ((f, g)-Multicriteria dimension reduction)*.* For each group A<sup>i</sup> , associate a function f<sup>i</sup> : V<sup>d</sup> → <sup>R</sup> that denotes the group's objective value for a particular V ∈ R n×d , and an aggregation function g : R <sup>k</sup> → <sup>R</sup>. Find V ∈ V<sup>d</sup> which optimizes

$$\min_{\mathbf{V} \in \mathcal{V}_d} g(f_1(\mathbf{V}\mathbf{V}^\top), f_2(\mathbf{V}\mathbf{V}^\top), \dots, f_k(\mathbf{V}\mathbf{V}^\top)).$$

[Samadi et al.](#page-8-1) [\(2018\)](#page-8-1) introduced the marginal loss, described next. Assume that we are given a matrix M with groups {A<sup>1</sup> , . . . , A<sup>k</sup>}. For some group A<sup>g</sup> , the singular values are σ1(A<sup>g</sup> ), . . . , σn(A<sup>g</sup> ). Given an n × d matrix V ∈ Vd, the *marginal* error of group A<sup>g</sup> using V is as follows.

Definition A.2 (FAIR-PCA loss).

$$\mathcal{L}_{\text{marg}}(\mathbf{A}^g, \mathbf{V}) \triangleq \sum_{i=1}^d \sigma_i^2(\mathbf{A}^g) - \|\mathbf{A}^g \mathbf{V} \mathbf{V}^\top\|_F^2.$$

For more information on the marginal loss, we refer the reader to [Samadi et al.](#page-8-1) [\(2018\)](#page-8-1) and [Tantipongpipat et al.](#page-8-2) [\(2019\)](#page-8-2).

## A.3. Consistency Makes Parity More Challenging

A motivating factor for using the marginal error objective in FAIR-PCA is that it ensures equal loss, when two groups are present in the data, i.e. Lmarg(A, V<sup>∗</sup> ) = Lmarg(B, V<sup>∗</sup> ) (see Theorem 4.5 in [Samadi et al.](#page-8-1) [\(2018\)](#page-8-1))

However, the consistency requirement means neither the reconstruction error nor the marginal loss can guarantee parity of loss while meeting the consistency requirements.

As already noticed, we are interested in minimizing the loss of projecting the groups in G using the common projection V:dV<sup>⊤</sup> for all values of d. Observe that V:dV<sup>⊤</sup> is an orthogonal projection.

:d :d *Observation* 1*.* Assume that Algorithm [1](#page-3-1) is executed on an instance with two groups A ∈ R <sup>a</sup>×<sup>n</sup> and B ∈ <sup>R</sup> <sup>b</sup>×<sup>n</sup>, where the loss function L is instead either Lrec or Lmarg. Then for optimal solution V<sup>∗</sup> ∈ <sup>R</sup> n×d it may hold that

$$\mathcal{L}(\mathbf{A}, \mathbf{V}^*(\mathbf{V}^*)^\top) \neq \mathcal{L}(\mathbf{B}, \mathbf{V}^*(\mathbf{V}^*)^\top)$$

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

To see why this holds for L = Lrec, assume that we have a solution of MULTIGROUP SV<sup>S</sup> of rank d, and that L(A, V:d) = L(B, V:d). The vector vd+1 lies in the orthogonal complement of V:d, V<sup>⊥</sup> :d . We denote the component of A in the orthogonal complement of V:<sup>d</sup> as Ad+1.

$$\text{If: } \|\mathbf{A}\|_F^2 - \|\mathbf{A}_{d+1}\mathbf{x}\mathbf{x}^\top\|_F^2 < \|\mathbf{B}\|_F^2 - \sigma_1^2(\mathbf{B}_{d+1}) \quad \forall \|\mathbf{x}\|_2^2 = 1,$$

or vice versa, then necessarily either L(A, V<sup>∗</sup> ) < L(B, V<sup>∗</sup> ) or L(A, V<sup>∗</sup> ) > L(B, V<sup>∗</sup> ),

For the marginal error Lmarg, assume again that L(A, V:d) = L(B, V:d), and we are seeking a vector vd+1 in V<sup>⊥</sup> :d . In order for L(A, Vd+1) = L(B, Vd+1) to hold, according to Property [1,](#page-2-3) we must have:

$$\sum_{i=1}^{d+1} (\sigma_i^2(\mathbf{A}) - \|\mathbf{A}\mathbf{v}_i\mathbf{v}_i^\top\|_F^2) = \sum_{i=1}^{d+1} (\sigma_i^2(\mathbf{B}) - \|\mathbf{B}\mathbf{v}_i\mathbf{v}_i^\top\|_F^2).$$

Since by hypothesis the equality holds for the summands up to the d-th, then the equality needs to hold also for i = d + 1. However if:

$$\sigma_{d+1}^2(\mathbf{A}) - \|\mathbf{A}_{d+1}\mathbf{x}\mathbf{x}^\top\|_F^2 < \sigma_{d+1}^2(\mathbf{B}) - \sigma_1^2(\mathbf{B}_{d+1}) \quad \forall \mathbf{x} : \|\mathbf{x}\|_2^2 = 1$$

or vice versa, then again either L(A, V<sup>∗</sup> ) < L(B, V<sup>∗</sup> ) or L(A, V<sup>∗</sup> ) > L(B, V<sup>∗</sup> ).

# B. Derivation of the Dual of Problem [1](#page-3-2)

The dual objective is obtained as:

$$g(\boldsymbol{\mu}, \boldsymbol{\lambda}) = \inf_{\mathbf{v}, z} \mathcal{H}(\mathbf{v}, z, \boldsymbol{\mu}, \boldsymbol{\lambda}).$$

First, notice that, grouping the terms containing z together we can see that the coefficient of z is 1 − P <sup>i</sup>∈G µ<sup>i</sup> . The infimum of H involves taking the derivative of H with respect to z and setting to zero.

$$\frac{\partial \mathcal{H}}{\partial z} = 0 \implies \sum_{i \in \mathcal{G}} \mu_i = 1$$

Since its coefficient is zero, we can effectively delete z from the lagrangian without changing the optimal solution. However, the infimum of the lagrangian w.r.t. v is particularly interesting. Rearranging the terms, we observe that the infimum involves the quadratic form: v ⊤( P <sup>i</sup>∈G −µi(A<sup>i</sup> ) <sup>⊤</sup>A<sup>i</sup> + λI)v. In general, the infimum of this expression is −∞, unless the matrix ( P <sup>i</sup>∈G −µi(A<sup>i</sup> ) <sup>⊤</sup>A<sup>i</sup> + λI) is positive semi-definite. We set A(µ) = P <sup>i</sup>∈G µi(A<sup>i</sup> ) <sup>⊤</sup>A<sup>i</sup> and thus equivalently we write:

$$-\mathbf{A}(\boldsymbol{\mu}) + \boldsymbol{\lambda}\mathbf{I} \succeq 0$$

We observe that the matrix A(µ) is a convex combination (since 0 ≤ µ<sup>i</sup> and P <sup>i</sup> µ<sup>i</sup> = 1 ) of positive semidefinite matrices, thus its negation is negative semidefinite. It follows that the primal minimization problem is bounded from below when λ = λmax(A(µ)) . We define s = [σ 2 1 (A<sup>1</sup> ), . . . , σ<sup>2</sup> 1 (A<sup>k</sup> )]. Putting everything together, we obtain the dual problem:

$$\max_{\mu \in \mathbb{R}^k} \mu^\top \mathbf{s} - \lambda_{max}(\mathbf{A}(\mu))$$
s.t.  $\mathbf{1}^\top \mu = 1$  (7)
 $\mu \geq 0.$  (8)

# C. SDP

Algorithm [3](#page-11-0) contains the pseudocode of SDP to solve Problem [1.](#page-3-2)

# D. Additional Experiment Results

Algorithm 3 MG-SINGULARVECTOR-SDP

1: Input: Matrices [A<sup>1</sup>

, . . . , A<sup>k</sup>

2: X ∈ R

<sup>n</sup>×<sup>n</sup> ← Solve:

$$\begin{aligned} \min_{z \in \mathbb{R}} z & \quad (9) \\ \text{s.t. } \sigma_1^2(\mathbf{A}^i) - \text{Tr}(\mathbf{A}^i \mathbf{X}) & \leq z \quad \text{for } \mathbf{A}^i \in \mathcal{G} \\ \begin{bmatrix} \mathbf{X} & \mathbf{x} \\ \mathbf{x}^\top & 1 \end{bmatrix} & \succeq 0 \quad , \text{Tr}(\mathbf{X}) \leq 1, \end{aligned}$$

3: X =

P<sup>n</sup>

<sup>j</sup>=1 λjxjx

⊤ j

4: Output: x<sup>1</sup> ∈ <sup>R</sup>

n

#### D.1. Two groups

Figure [3](#page-14-0) shows the different metrics being monitored in our experiments (i.e., the marginal loss, the incremental loss and the reconstruction loss) as a function of reconstruction (target) rank in all considered two-group datasets except the compas dataset, for which results are provided in Figure [2](#page-7-1) in the main text.

The findings of the experiments presented in Figure [3](#page-14-0) largely corroborate the findings presented in the main text (Figure [2\)](#page-7-1) for the compas dataset.

## D.2. More than two groups

Figure [4](#page-15-0) displays marginal, incremental and reconstruction loss by rank in the communities-4 dataset partitioned into four groups. Again, the results for the communities-4 are consistent with and confirm the results seen in in Figure [2](#page-7-1) for the compas-3 dataset.

Empirical Duality gap. In the case of more than two groups, the proposed methods are heuristics as they are not guaranteed to retrieve an optimal solution. In particular, there can be a discrepancy between the optimum of the primal and the one of the dual. Such discrepancy is known as *duality gap*. We note that we can compare the value of the dual objective g, at the obtained solutions for Algorithms [2](#page-5-2) and [3,](#page-11-0) and also the primal objective for the corrsponing solution vector v = v(µ) from Algorithm [2](#page-5-2) and v = x<sup>1</sup> from Algorithm [3,](#page-11-0) by computing f = max(h1(v), . . . , hk(v)). We call the difference |f − g|, *empirical duality gap*, as it gives us an empirical estimate of how far away from optimality are our solutions (a zero empirical duality gap means that the particular primal-dual solution pair is optimal).

In practice, as shown in Figure [5,](#page-15-1) such empirical duality gap is typically narrow. In particular, Figure [5](#page-15-1) shows the value of the primal and dual objective in the compas-3 dataset with three groups, communities-4 dataset with four groups as well as in a synthetic dataset (gaussian-3) consisting of three groups, each of size 50 × 10 and with entries independent and identically distributed according to a standard Gaussian distribution. The difference between the primal and dual objective is generally limited and often negligible.

The results presented in Figure [5](#page-15-1) are obtained by resorting to the Frank-Wolfe procedure to solve the dual problem, i.e., Algorithm [2.](#page-5-2)

The Frank-Wolfe algorithm is the algorithm of choice because of its simplicity and efficiency. However, solving MG-SINGULARVECTOR by the semidefinite programming relaxation (Algorithm [3\)](#page-11-0) yields an even smaller duality gap, as demonstrated in Figure [6](#page-15-2) for the same datasets considered in Figure [5.](#page-15-1)

# E. Proofs

689 690

694

696

698

700

704

706

708 709

711

#### E.1. Proof of Property [1](#page-2-3)

*Proof.* We have that:

$$\|\mathbf{AVV}^\top\|_F^2 = \|\mathbf{Av}_1\mathbf{v}_1^\top + \dots + \mathbf{Av}_d\mathbf{v}_d^\top\|_F^2.$$

The result follows from orthogonality, i.e., v ⊤ <sup>i</sup> v<sup>j</sup> = 0 for all i, j ∈ [1, . . . , d]. This implies that:

$$\|\mathbf{A}\mathbf{v}_1\mathbf{v}_1^\top + \dots + \mathbf{A}\mathbf{v}_d\mathbf{v}_d^\top\|_F^2 = \|\mathbf{A}\mathbf{v}_1\mathbf{v}_1^\top\| + \dots + \|\mathbf{A}\mathbf{v}_d\mathbf{v}_d^\top\|_F^2 = \sum_{i=1}^d \|\mathbf{A}\mathbf{v}_i\mathbf{v}_i^\top\|_F^2$$

## E.2. Proof of Orthonormalization Argument

*Proof.* Following an inductive argument (where the induction is on d), we can prove that V = {v1, . . . , vd} is indeed an orthonormal basis.

Base case. For d = 1, we can choose an arbitrary unit vector v1. Note that v<sup>1</sup> is in the orthogonal complement of the subspace spanned by the 0. Since v<sup>1</sup> is a unit vector, it forms an orthormal basis of its span {v1}.

Inductive hypothesis At step k − 1, we have a k − 1-dimensional orthonormal basis Vk−<sup>1</sup> = {v1, . . . , vk−1}.

Inductive step At step k, we project the data onto the orthogonal complement of vk−<sup>1</sup> and we select v<sup>k</sup> in such subspace. The orthogonal complement of vk−<sup>1</sup> is which also orthogonal to the space spanned by vk−2, and so on. Thus, v<sup>k</sup> is orthogonal to all vectors v<sup>j</sup> for j < k and V = {v1, . . . , vk} must be an orthonormal basis, which completes the proof.

## E.3. Proof of Theorem [7.1](#page-5-3)

*Proof.* Since we are in the case |G| = 2, we can consider a simplified formulation. We notice that µ<sup>2</sup> = 1 − µ<sup>1</sup> and set µ<sup>1</sup> = µ and µ<sup>2</sup> = 1 − µ. We also set A<sup>1</sup> = A, A<sup>2</sup> = B and C(µ) = µA⊤A + (1 − µ)B⊤B. Thus, Problem [2](#page-4-1) becomes:

$$\max_{\mu \in \mathbb{R}} \mu s_1 + (1 - \mu) s_2 - \lambda_{max}(\mathbf{C}(\mu)), \quad \mu \in [0, 1]. \quad (10)$$

We can now perform the standard KKT analysis. The dual lagrangian is:

$$\mathcal{H}_D(\mu, \xi_1, \xi_2) = g(\mu) + \xi_1\mu + \xi_2(1 - \mu).$$

The stationarity condition is:

$$\frac{\partial}{\partial \mu} \mathcal{H}_D(\mu^*, \xi_1, \xi_2) = \frac{\partial}{\partial \mu} g(\mu^*) + \xi_1 - \xi_2 = 0.$$

Additionally, the complementary slackness condition requires that ξ1µ = 0 and ξ2(1 − µ) = 0. To see this, first recall the duality between MG-SINGULARVECTOR and MG-SINGULARVECTOR-DUAL, from which we know that µ<sup>1</sup> = µ and µ<sup>2</sup> = 1 − µ are the associated multipliers with constraints h<sup>A</sup> − z and h<sup>B</sup> − z of MG-SINGULARVECTOR. From Theorem [5.1](#page-3-3) we know that h<sup>A</sup> − z = 0 and h<sup>B</sup> − z = 0 and thus from complementary slackness we can infer that µ can be neither 0 or 1. Similarly, complementary slackness between µ and ξ<sup>1</sup> and ξ<sup>2</sup> indicates that ξ<sup>1</sup> = ξ<sup>2</sup> = 0.

Thus, stationarity simply reduces to <sup>∂</sup> ∂µg(µ ∗ ) = 0. From this and using equation [6,](#page-5-4) it follows that:

$$s_1 - s_2 - \mathbf{v}^\top (\mu^*) (\mathbf{A}^\top \mathbf{A} - \mathbf{B}^\top \mathbf{B}) \mathbf{v}(\mu^*) = 0. \quad (11)$$

Therefore, v(µ ∗ ) leads to equal loss between the two groups. Additionally, this stationary point is a global maximum of g. To see this, we take the second derivative of g:

$$\frac{\partial^2 g}{\partial \mu^2} = -\frac{\partial^2}{\partial \mu^2} \lambda_{max}(\mathbf{C}(\mu)).$$

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

768

The Hadamard second variation formula [\(Schiffer,](#page-8-24) [1946\)](#page-8-24), gives us an analytical expression for the second derivative of λmax:

$$\frac{\partial^2}{\partial \mu^2} \lambda_{max}(\mathbf{C}(\mu)) = \mathbf{v}(\mu)^\top \frac{\partial^2 \mathbf{C}(\mu)}{\partial \mu^2} \mathbf{v}(\mu) + 2 \sum_{j \neq max} \frac{|\mathbf{v}(\mu)^\top \frac{\partial \mathbf{C}(\mu)}{\partial \mu} \mathbf{v}_j(\mu)|}{\lambda_{max} - \lambda_j(\mu)}. \quad (12)$$

where λ<sup>j</sup> , v<sup>j</sup> are eigenvalue-eigenvector pairs corresponding to smaller eigenvalues. The first term of Equation [12](#page-13-1) vanishes (C(µ) is only linearly dependent on µ), while the numerator and denominator in the second term are trivially positive (since C(µ) is positive semidefinite and λmax > λ<sup>j</sup> . An important thing to note is that we have assumed simple spectrum. From this we can conclude that <sup>∂</sup> g ∂µ<sup>2</sup> < 0, i.e., the function is concave, and thus has a unique maximum, at µ ∗ . At µ ∗ , we have that:

$$\begin{aligned} g(\mu^*) &= s_1 - \mathbf{v}(\mu^*)^\top \mathbf{A}^\top \mathbf{A} \mathbf{v}(\mu^*) \\ &= s_2 - \mathbf{v}(\mu^*)^\top \mathbf{B}^\top \mathbf{B} \mathbf{v}(\mu^*). \end{aligned}$$

As v(µ ∗ ) is also a feasible point of Problem [1,](#page-3-2) with some value z, we have that g(µ ∗ ) = z and since the primal is always lower bounded by the dual, we conclude that strong duality holds.

Lemma E.1. *Define* q(µ) = s<sup>1</sup> − s<sup>2</sup> − v <sup>⊤</sup>(µ)(A⊤A − B⊤B)v(µ)*. Then,* µ ∗ *is a root of* q(µ) *and additionally* q(µ) *is monotone with respect to* µ

The fact that µ ∗ is a root of q(µ) follows directly from Equation [11.](#page-12-0) The monotonicity follows from ∂q ∂µ = − ∂ ∂µ<sup>2</sup> λmax(C(µ)) > 0. This has an interesting consequence for the problem under investigation when |G| = 2. The fact that a unique root exists in µ ∈ (0, 1) and the monotonicity mean that we can resort to a root-finding algorithm (such as Brent's method [\(Brent,](#page-8-21) [1971\)](#page-8-21) or the bisection method [\(Ehiwario & Aghamie,](#page-8-25) [2014\)](#page-8-25)) to locate the optimal µ ∗ . In fact, as we show in the experiments, such an algorithm is highly effective for MG-SINGULARVECTOR, when |G| = 2. By default, we use the aforementioned Brent's method for finding the unique root µ ∈ (0, 1).

Note that a similar approach based on root-finding algorithms cannot be applied to the case of more than two groups and there is no obvious way to extend this approach to the general case.

## E.4. Proof of Lemma [7.2](#page-5-5)

*Proof.* Using a Schur complement [\(Boyd & Vandenberghe,](#page-8-17) [2004\)](#page-8-17), we can rewrite Problem [2](#page-4-1) as:

$$\max_{\mu \in \mathbb{R}^k} \gamma$$
s.t.  $\begin{bmatrix} -\mathbf{A}(\mu) + \lambda \mathbf{I} & 0 \\ 0 & \mu^\top \mathbf{s} - \gamma \end{bmatrix} \succeq 0$ 

$$\mathbf{1}^\top \mu = 1$$

$$\mu \geq 0.$$

To complete the proof, it suffices to notice that the SDP relaxation illustrated in Algorithm [3](#page-11-0) is the dual problem to this problem (with dual variable X). From our previous duality results it follows that strong duality exists between these two SDPs. Then, we can conclude that the SDP in Algorithm [3](#page-11-0) solves Problem [1](#page-3-2) to optimality.

## E.5. Proof of Lemma [7.3](#page-5-6)

*Proof.* Observe that V = {v1, . . . , vd} is a matrix with orthonormal columns since it is constructed using Algorithm [1.](#page-3-1) Hence, we can invoke Property [1](#page-2-3) along with Theorem [5.1](#page-3-3) to obtain the result. Namely, after running Algorithm [1,](#page-3-1) we obtain <sup>V</sup> <sup>=</sup> {v1, . . . , <sup>v</sup>d}, which gives a total error of P<sup>d</sup> <sup>i</sup>=1 <sup>L</sup>(A, <sup>v</sup>i) for group <sup>A</sup> and a total error of P<sup>d</sup> <sup>i</sup>=1 L(B, vi) for group B. We know that L(A, vi) = L(B, vi) for any i ∈ {1, . . . , d} due to Theorem [5.1.](#page-3-3) The lemma then follows.

As for time complexity, it suffices to consider that the optimal rank-1 solutions of MG-SINGULARVECTOR for two groups can be obtained in polynomial time O(ℓ), as stated in Theorem [7.1.](#page-5-3) Then Property [1](#page-2-3) implies that we need total time O(dℓ) to obtain an optimal solution.

774

776

778

794

796

800

804

806

808

![](_page_14_Figure_2.jpeg)

Figure 3: Real-world datasets with two groups. Marginal, incremental and reconstruction loss by rank. Different marker symbols indicate different groups.

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

![](_page_15_Figure_1.jpeg)

Figure 4: communities-4 dataset with four groups. Marginal, incremental and reconstruction loss by rank. Different marker signs indicate different groups.

![](_page_15_Figure_3.jpeg)

Figure 5: Real-world and syntethic data. Primal and dual optimal objective values as a function of rank for the solution relying on the Frank-Wolfe algorithm.

![](_page_15_Figure_5.jpeg)

Figure 6: Real-world and synthetic data. Duality gap as a function of rank for the solutions relying on the Frank-Wolfe (FW) and semidefinite programming solver (SDP).