011

014 015 016

018

024

026

034

036

038

054

# STNet: Spectral Transformation Network for Solving Operator Eigenvalue Problem

Anonymous Authors<sup>1</sup>

## Abstract

Operator eigenvalue problems play a critical role in various scientific fields and engineering applications, yet numerical methods are hindered by the curse of dimensionality. Recent deep learning methods provide an efficient approach to address this challenge by iterative updating neural networks. These methods' performance relies heavily on the spectral distribution of the given operator: larger gaps between the operator's eigenvalues will improve precision, thus tailored spectral transformations that leverage the spectral distribution can enhance their performance. Based on this observation, we propose the Spectral Transformation Network (STNet). During each iteration, STNet uses approximate eigenvalues and eigenfunctions to perform spectral transformations on the original operator, turning it into an equivalent but easier problem. Specifically, we employ deflation projection to exclude the subspace corresponding to already solved eigenfunctions, thereby reducing the search space and avoiding converging to existing eigenfunctions. Additionally, our filter transform magnifies eigenvalues in the desired region and suppresses those outside, further improving performance. Extensive experiments demonstrate that STNet consistently outperforms existing learning-based methods, achieving state-of-the-art performance in accuracy.

# 1. Introduction

The operator eigenvalue problem is a prominent focus in many scientific fields [\(Elhareef & Wu,](#page-8-0) [2023;](#page-8-0) [Buchan et al.,](#page-8-1) [2013;](#page-8-1) [Cuzzocrea et al.,](#page-8-2) [2020;](#page-8-2) [Pfau et al.,](#page-9-0) [2023\)](#page-9-0) and engineering applications [\(Diao et al.,](#page-8-3) [2023;](#page-8-3) [Chen & Chan,](#page-8-4) [2000\)](#page-8-4). However, traditional numerical methods are constrained by the curse of dimensionality, as the computational complexity increases quadratically or even cubically with the mesh size [\(Watkins,](#page-9-1) [2007\)](#page-9-1).

A promising alternative is using neural networks to approxi-

![](_page_0_Figure_3.jpeg)

Figure 1: Absolute error results of eigenvalues for the Fokker-Planck operator computed using various algorithms, the x axis represents the operator dimension.

mate eigenfunctions [\(Pfau et al.,](#page-9-2) [2018\)](#page-9-2). These approaches reduce the number of parameters by replacing the matrix representation with a parametric nonlinear representation via neural networks. By designing appropriate loss functions, it updates parameters to approximate the desired operator eigenfunctions. These methods only require sampling specific regions without designing discretization mesh, significantly reducing the algorithm design cost and unnecessary approximation errors [\(He et al.,](#page-8-5) [2022\)](#page-8-5). Moreover, neural networks generally exhibit stronger expressiveness than linear matrix representations, requiring far fewer sampling points for the same problem compared to traditional methods [\(Nguyen et al.,](#page-9-3) [2020\)](#page-9-3).

Despite these advantages, the performance of such methods strongly depends on the operator's spectral distribution: if the target eigenvalues differs greatly to each other, the algorithm converges much more faster; otherwise, it may suffer from inefficient iterations. To improve convergence, spectral transformations can be designed based on the spectral distribution, reformulating the original problem into an equivalent but more tractable one. However, since the

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

![](_page_1_Diagram_1.jpeg)

Figure 2: Comparison of the eigenfunctions of the 2D Harmonic operator computed by STNet and the Ground Truth.

real spectrum of the operator is initially unknown, existing approaches do not optimize spectral properties through such transformations.

To address this limitation, we propose the Spectral Transformation Network (STNet). By exploiting approximate eigenvalues and eigenvectors learned during the iterative process, STNet applies spectral transformations to the original operator, modifying its spectral distribution and thereby converting it into an equivalent problem that converges more easily. Concretely, we employ deflation projection to remove the subspace corresponding to already computed eigenfunctions. This not only narrows the search space but also prevents subsequent eigenfunctions from collapsing into the same subspace. Meanwhile, our filter transform amplifies eigenvalues within the target region and suppresses those outside it, promoting rapid convergence to the desired eigenvalues. Extensive experiments demonstrate that STNet significantly surpasses existing methods based on deep learning, achieving state-of-the-art performance in accuracy. Figure [2](#page-1-0) presents the results obtained by STNeton the 2D Harmonic operator eigenvalue problem, alongside the ground truth, demonstrating our method's capability to accurately solve eigenvalue problems.

# 2. Related work

Recent advancements in applying neural networks to eigenvalue problems have shown promising results. Innovations such as spectral inference networks (SpIN) [\(Pfau et al.,](#page-9-2) [2018\)](#page-9-2), which model eigenvalue problems as kernel problem optimizations solved via neural networks. Neural eigenfunctions (NeuralEF) [\(Deng et al.,](#page-8-6) [2022\)](#page-8-6), which significantly reduces computational costs by optimizing the costly orthogonalization steps, are noteworthy. Neural singular value decomposition (NeuralSVD) employs truncated singular value decomposition for low-rank approximation to enhance the orthogonality required in learning functions [\(Ryu et al.,](#page-9-4) [2024\)](#page-9-4).

Another class of algorithms originates from optimizing the Rayleigh quotient. The deep Ritz method (DRM) utilizes the Rayleigh quotient for computing the smallest eigenvalues, demonstrating significant potential [\(Yu et al.,](#page-9-5) [2018\)](#page-9-5). Several studies have employed the Rayleigh quotient to construct variation-free functions, achieved through physics-informed neural networks (PINNs) [\(Ben-Shaul et al.,](#page-8-7) [2023;](#page-8-7) [2020\)](#page-8-8). Extensions of this approach include enhanced loss functions with regularization terms to improve the learning accuracy of the smallest eigenvalues [\(Jin et al.,](#page-8-9) [2022\)](#page-8-9). Additionally, [Han et al.](#page-8-10) [\(2020\)](#page-8-10) reformulate the eigenvalue problem as a fixed-point problem of the semigroup flow induced by the operator, solving it using the diffusion Monte Carlo method. The power method neural network (PMNN) integrates the power method with PINNs, using an iterative process to approximate the exact eigenvalues [\(Yang et al.,](#page-9-6) [2023\)](#page-9-6) closely. While PMNN has proven effective in solving for a single eigenvalue [\(Yang et al.,](#page-9-6) [2023\)](#page-9-6), it has yet to be developed for computing multiple distinct eigenvalues simultaneously.

Furthermore, in the field of computational chemistry, research on specialized model architectures for specific operators, such as the Hamiltonian, focuses on developing novel neural network ansatzes [\(Carleo & Troyer,](#page-8-11) [2017;](#page-8-11) [Schutt et al.](#page-9-7) ¨ , [2017;](#page-9-7) [Choo et al.,](#page-8-12) [2020;](#page-8-12) [Pfau et al.,](#page-9-8) [2020;](#page-9-8) [Hermann et al.,](#page-8-13) [2020;](#page-8-13) [Gerard et al.,](#page-8-14) [2022;](#page-8-14) [Hermann et al.,](#page-8-15) [2023\)](#page-8-15). These architectures are designed to embed physical inductive biases better, enhancing expressivity. Additionally, there are studies employing neural networks for Quantum Monte Carlo (QMC) methods to tackle related problems in quantum chemistry [\(Cuzzocrea et al.,](#page-8-2) [2020;](#page-8-2) [Entwistle et al.,](#page-8-16) [2023;](#page-8-16) [Pfau et al.,](#page-9-0) [2023\)](#page-9-0).

# 3. Preliminaries

## 3.1. Operator Eigenvalue Problem

We primarily focus on the eigenvalue problems of differential operators, such as <sup>∂</sup> ∂x + ∂ ∂y , ∆, etc. Mathematically, an operator L : H<sup>1</sup> → H<sup>2</sup> is a mapping between two Hilbert spaces. Considering a self-adjoint operator L defined on a domain Ω ⊂ R <sup>D</sup>, the operator eigenvalue problem can be expressed in the following form [\(Evans,](#page-8-17) [2022\)](#page-8-17):

$$\mathcal{L}v = \lambda v \quad \text{in } \Omega, \quad (1)$$

where Ω ⊆ R <sup>D</sup> serves as the domain; v is the eigenfunction and λ is the eigenvalue. Typically, it is often necessary to solve for multiple eigenvalues, λ<sup>i</sup> , i = 1, . . . , L.

## 3.2. Power Method

The power method is a classical algorithm designed to approximate the eigenvalue of an operator L in the vicinity of a given shift σ. By applying the shift σ (often chosen as an

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

approximation to the target eigenvalue), the original eigenvalue problem is effectively transformed into an equivalent problem for the new operator (L − σI) −1 . In each iteration, the current approximate solution is multiplied by this new operator, thereby amplifying the component associated with the eigenvalue closest to σ. This iterative procedure converges to the desired eigenvalue. The pseudocode is shown below [\(Golub & Van Loan,](#page-8-18) [2013\)](#page-8-18):

Algorithm 1 Power Method for the Operator L

1: Input: Operator L, shift σ, initial guess v 0 , maximum iterations kmax, and convergence threshold ϵ. 2: Output: Eigenvalue λ near σ. 3: v <sup>0</sup> = v <sup>0</sup>/∥v <sup>0</sup>∥ . 4: for k = 1 to kmax do 5: v <sup>k</sup> = p <sup>k</sup>/∥p <sup>k</sup>∥ and solve (L − σI) p <sup>k</sup> = v k−1 . 6: if ∥v <sup>k</sup> − v <sup>k</sup>−<sup>1</sup>∥ < ϵ then 7: λ = ⟨v k ,Lv k ⟨v <sup>k</sup>,v<sup>k</sup>⟩ and break. 8: end if 9: end for

In each iteration, solving the linear system (L − σI) p <sup>k</sup> = v k−1 is equivalent to applying the operator (L − σI) −1 to v k−1 . Afterward normalizing v <sup>k</sup> helps maintain numerical stability. Convergence is typically assessed by evaluating the error ∥v <sup>k</sup> −v <sup>k</sup>−<sup>1</sup>∥, ensuring that the final solution meets the desired accuracy. The fundamental reason for the convergence of the power method lies in the repeated application of (L − σI) −1 , which progressively magnifies the component of v k in the direction of the eigenfunction with eigenvalue closest to σ. For a more detailed introduction to the power method, please refer to the Appendix [A.1.](#page-10-0)

## 3.3. Deflation Projection

The deflation technique plays a critical role in solving eigenvalue problems, particularly when multiple distinct eigenvalues need to be computed. Deflation projection is an effective deflation strategy that utilizes known eigenvalues and corresponding eigenfunctions to modify the structure of the operator, thereby simplifying the computation of remaining eigenvalues [\(Saad,](#page-9-9) [2011\)](#page-9-9).

The core idea of deflation projection is to construct an operator P, often defined as P(u) = ⟨u, v1⟩v<sup>1</sup> where v<sup>1</sup> is a known eigenfunction. This operator is then used to modify the original operator L into a new operator:

$$\mathcal{B} = \mathcal{L} - \lambda_1 \mathcal{P}. \quad (2)$$

In B, the eigenvalue λ<sup>1</sup> associated with v<sup>1</sup> is effectively removed from the spectrum of L. Additional details on deflation projection can be found in Appendix [A.2.](#page-10-1)

#### 3.4. Filter Transform

The filter transform is widely used in numerical linear algebra to enhance the accuracy of eigenvalue computations [\(Saad,](#page-9-9) [2011\)](#page-9-9). By constructing a suitable filter function F(L), the operator L undergoes a spectral transformation that amplifies the target eigenvalues and suppresses the irrelevant ones. The filter transform can effectively highlight the desired spectral region without altering the corresponding eigenfunctions [\(Watkins,](#page-9-1) [2007\)](#page-9-1). Further details on the filter transform can be found in Appendix [A.3.](#page-11-0)

## 4. Method

#### 4.1. Problem Formulation

We consider the operator eigenvalue problem for a differential operator L defined on a domain Ω ⊂ R <sup>D</sup>. Our goal is to approximate the L eigenvalues λ<sup>i</sup> near a given shift σ and their corresponding eigenfunctions v<sup>i</sup> , satisfying

$$\mathcal{L} v_i = \lambda_i v_i, \quad i = 1, 2, \dots, L. \quad (3)$$

To achieve this, we employ L neural networks parameterized by θ<sup>i</sup> . Each neural network NNL(·; θi) maps the domain Ω into <sup>R</sup>, providing an approximation of the eigenfunction v<sup>i</sup> :

$$NN_{\mathcal{L}}(\cdot; \theta_i) : \Omega \rightarrow \mathbb{R}, \quad i = 1, 2, \dots, L. \quad (4)$$

In order to represent both the functions and the operators numerically, we discretize Ω by uniformly randomly sampling N points:

$$S \equiv \{\mathbf{x}_j = (x_j^1, \dots, x_j^D) \mid \mathbf{x}_j \in \Omega, j = 1, 2, \dots, N\}, \quad (5)$$

Correspondingly, each neural network NNL(·; θi) output a vector Y<sup>i</sup> ∈ <sup>R</sup> <sup>N</sup> , which approximate the values of the eigenfunction v˜i(·) = NNL(·; θi) at these sampled points:

$$\tilde{v}_i(\mathbf{x}_j) \equiv \mathbf{Y}_i(j), \quad i = 1, 2, \dots, L, \quad j = 1, 2, \dots, N. \quad (6)$$

The approximate eigenvalues λ˜ <sup>i</sup> are then obtained by applying L to the computed eigenfunctions v˜<sup>i</sup> :

$$\tilde{\lambda}_i \equiv \frac{\langle \tilde{v}_i, \mathcal{L}\tilde{v}_i \rangle}{\langle \tilde{v}_i, \tilde{v}_i \rangle}, \quad i = 1, 2, \dots, L. \quad (7)$$

Here, the differential operator L acts on the functions via automatic differentiation. We iteratively update the neural network parameters θ<sup>i</sup> using gradient descent, aiming to

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

minimize the overall residual. Specifically, we formulate the following optimization problem:

$$\min_{\theta_i \in \Theta} \frac{1}{N} \sum_{i=1}^L \sum_{j=1}^N [\tilde{v}_i(\mathbf{x}_j) - v_i(\mathbf{x}_j)]^2, \quad (8)$$

, where Θ denotes the parameter space of the neural networks. This approach does not require any training data, as it relies solely on satisfying the differential operator eigenvalue equations over the domain Ω. Finally, this procedure provides approximations λ˜ <sup>i</sup> of the true eigenvalues λ<sup>i</sup> , i = 1, . . . , L.

#### 4.2. Spectral Transformation Network

Inspired by the power method and the power method neural network [\(Yang et al.,](#page-9-6) [2023\)](#page-9-6), we propose STNet to solve eigenvalue problems, as shown in Figure [3.](#page-4-0) In STNet, we replace the function v k from the k-th iteration of the power method with v˜ k i (x) ≡ NNL(x; θ k i ), where each neural network is implemented via a multilayer perceptron. Since neural networks cannot directly implement the inverse operator (L −σI) −1 , we enforce (L −σI)˜v <sup>k</sup> ≈ v˜ k−1 through a suitable loss function. The updated parameters θ k <sup>i</sup> → θ k+1 i then yield v˜ <sup>k</sup>+1 = NNL(x; θ k+1 i ). Algorithm [2](#page-3-0) shows the detailed procedure of STNet.

Classical power method convergence is closely related to the spectral distribution of the operator, which is unknown initially and thus difficult to optimize against directly. However, as the iterative process starts, we can get additional information—such as already computed eigenvalues and eigenfunctions. Using these results for the spectral transformation of the original operator can greatly improve subsequent power-method iterations. In our pseudocode [2,](#page-3-0) we introduce two modules to enhance performance:

- Deflation Projection uses already computed eigenvalues and eigenfunctions to construct a projection that excludes the previously resolved subspace, preventing convergence to known eigenfunctions and reducing the search space.
- Filter Transform employs approximate eigenvalues to construct a spectral transformation (filter function) that enlarges the target eigenvalue region and suppresses others, boosting the efficiency of STNet.

#### 4.2.1. DEFLATION PROJECTION

Suppose we have already approximated the eigenvalues λ˜ <sup>1</sup>, λ˜ <sup>2</sup>, . . . , λ˜ <sup>i</sup>−<sup>1</sup> and their corresponding eigenfunctions v˜1, v˜2, . . . , v˜i−1. To compute the i-th eigenfunction, we focus on the residual subspace orthogonal to the subspace spanned by these previously computed eigenfunctions.

Algorithm 2 Spectral Transformation Network

1: Input: Operator L over domain Ω ⊂ R <sup>D</sup>, shift σ, number of sampling points N, number of eigenvalues L, learning rate η, convergence threshold ϵ, maximum iterations kmax. 2: Output: Eigenvalues λ˜ i , i = 1, . . . , L. 3: Uniformly randomly sample N points {xj} in Ω to form dataset S. 4: Randomly initialize the network parameters θ 0 i , as well as the normalized v˜<sup>i</sup> , and set λ˜ <sup>i</sup> = σ, i = 1, . . . , L. 5: for k = 1 to kmax do 6: v˜ k i (x<sup>j</sup> ) = NNL(x<sup>j</sup> ; θ k i ), x<sup>j</sup> ∈ S. 7: L ′ <sup>i</sup> = Di(L), i = 1, . . . , L // Deflation Projection 8: L ′′ <sup>i</sup> = Fi(L ′ ), i = 1, . . . , L // Filter Transform 9: u˜ k i (x<sup>j</sup> ) = <sup>L</sup> ′′ v˜ k (x<sup>j</sup> ) ∥L′′ v˜ k (x<sup>j</sup> )∥ , i = 1, . . . , L. 10: Loss<sup>k</sup> <sup>i</sup> = N P<sup>N</sup> <sup>j</sup>=1[˜v k−1 i (x<sup>j</sup> ) − u˜ k i (x<sup>j</sup> )]<sup>2</sup> , i = 1, . . . , L. 11: θ k+1 <sup>i</sup> = θ k <sup>i</sup> − <sup>η</sup> ∇<sup>θ</sup><sup>i</sup> Loss<sup>k</sup> i , i = 1, . . . , L // Parameter Update 12: for i = 1 to L do 13: if Loss<sup>k</sup> <sup>i</sup> < ϵ<sup>i</sup> then 14: ϵ<sup>i</sup> = Loss<sup>k</sup> i , λ˜ <sup>i</sup> = ⟨v˜ k i ,Lv˜ k i ⟨v˜ i ,v˜ k i , v˜<sup>i</sup> = ˜v k i . 15: end if 16: end for 17: if ϵ<sup>i</sup> < ϵ for all i then 18: Convergence achieved; break. 19: else 20: Update deflation projection and filter function: D<sup>i</sup> , F<sup>i</sup> , i = 1, . . . , L. 21: end if 22: end for

The deflated projection is then defined as

$$D_i(\mathcal{L}) \equiv \mathcal{L} - \mathcal{Q}_{i-1} \Sigma_{i-1} \mathcal{Q}_{i-1}^*. \quad (9)$$

Here Qi−<sup>1</sup> maps each vector (α1, . . . , αi−1) ∈ <sup>R</sup> i−1 to the function P<sup>i</sup>−<sup>1</sup> <sup>k</sup>=1 α<sup>k</sup> v˜k, thus reconstructing functions from the span of {v˜1, . . . , v˜i−1}. Q<sup>∗</sup> i−1 is the adjoint of Qi−1. And Σi−<sup>1</sup> is a diagonal operator that scales each v˜<sup>k</sup> by its corresponding eigenvalue λ˜ k.

By employing the deflation projection, the gradient descent search space of the neural network is constrained to be orthogonal to the subspace spanned by {v˜1, v˜2, . . . , v˜i−1}. This projection prevents the neural network output NNL(θi) from converging to the invariant subspace formed by known eigenfunctions, thereby enhancing the orthogonality among the outputs of different neural networks NNL(θ1), . . . , NNL(θi−1). On one hand, this reduction

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

![](_page_4_Diagram_1.jpeg)

![](_page_4_Diagram_0.jpeg)

Figure 3: Overview of the STNet. (a) Introduction to the inputs and outputs. (b) STNet comprises multiple neural networks, each tasked with predicting distinct eigenvalues. If the accuracy of the solution reaches the expectation, then STNet will output the result.

in the search space accelerates the convergence toward the eigenfunctions v<sup>i</sup> ; On the other hand, it improves the orthogonality among the neural network outputs, which reduces the error in predicting the eigenfunction v˜<sup>i</sup> .

In practice, we use the approximate eigenvalues and eigenfunctions with the smallest error in iterations to construct the deflation projection. This allows us to update adaptively, ensuring that the method remains effective when calculating more eigenfunctions.

#### 4.2.2. FILTER TRANSFORM

During the iterative process, we can obtain approximate eigenvalues λ˜ i , and assume the corresponding true eigenvalues lie within [λ˜ <sup>i</sup>−ξ, λ˜ <sup>i</sup>+ξ], where ξ is a tunable parameter, typically ξ = 0.1 or 1. We employ a rational function-based filter transform on the original operator to simultaneously amplify the eigenvalues in these intervals and thus improve convergence performance. Specifically, we transform

$$\mathcal{L} \rightarrow \prod_{i_0=0}^{i-1} [(\mathcal{L} - (\tilde{\lambda}_{i_0} - \xi)I)(\mathcal{L} - (\tilde{\lambda}_{i_0} + \xi)I)]^{-1}. \quad (10)$$

By contrast, the basic power method shift-invert strategy, L → (L − σI) −1 , can be viewed as a special case of this more general construction. In STNet, we simulate the inverse operator via a suitably designed loss function. Therefore, the corresponding pseudocode filter function F removes the inverse, namely:

$$F_i(\mathcal{L}) = \prod_{i_0=0}^{i-1} [(\mathcal{L} - (\tilde{\lambda}_{i_0} - \xi)I)(\mathcal{L} - (\tilde{\lambda}_{i_0} + \xi)I)]. \quad (11)$$

When λ<sup>i</sup> lies within [λ˜ <sup>i</sup> − ξ, λ˜ <sup>i</sup> + ξ], the poles λ˜ <sup>i</sup> ± ξ make ∥Fi(vi)∥ sufficiently large for the corresponding eigenvector v<sup>i</sup> . This repeated amplification causes that direction to dominate in the subsequent iterations, while eigenvalues outside those intervals are gradually suppressed. Consequently, the method converges more efficiently to the desired eigenvalues.

# 5. Experiments

We conducted comprehensive experiments to evaluate STNet, focusing on:

- Solving multiple eigenvalues in the Harmonic eigenvalue problem.
- Solving the principal eigenvalue in the Schrodinger ¨ oscillator equation.
- Solving zero eigenvalues in the Fokker-Planck equation.
- Comparative experiment with traditional algorithms.
- The ablation experiments.

Baselines: For these experiments, we selected three learning-based methods for computing operator eigenvalues as our baselines: 1. PMNN [\(Yang et al.,](#page-9-6) [2023\)](#page-9-6); 2. NeuralEF [\(Deng et al.,](#page-8-6) [2022\)](#page-8-6); 3. NeuralSVD [\(Ryu et al.,](#page-9-4) [2024\)](#page-9-4). In the comparative experiments with traditional algorithms, we chose the finite difference method (FDM) [\(LeVeque,](#page-9-10) [2007\)](#page-9-10).

Experiment Settings: To ensure consistency, all experiments were conducted under the same computational conditions. For further details on the experimental environment

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

and algorithm parameters, please refer to Appendices [B.1](#page-12-0) and [B.2.](#page-12-1)

#### 5.1. Harmonic Eigenvalue Problem

Harmonic eigenvalue problems are common in fields such as structural dynamics and acoustics, and can be mathematically expressed as follows [\(Yang et al.,](#page-9-6) [2023;](#page-9-6) [Morgan &](#page-9-11) [Zeng,](#page-9-11) [1998\)](#page-9-11):

$$\begin{cases} -\Delta v = \lambda v, & \text{in } \Omega, \\ v = 0, & \text{on } \partial\Omega. \end{cases} \quad (12)$$

Here ∆ denotes the Laplacian operator. We consider the domain Ω = [0, 1]<sup>D</sup> where D represents the dimension of the operator, and the boundary conditions are Dirichlet. In this setting, the eigenvalue problem has analytical solutions, with eigenvalues and corresponding eigenfunctions given by:

$$\lambda_{n_1, \dots, n_D} = \pi^2 \sum_{k=1}^D n_k^2, \quad n_k \in \mathbb{N}^+ \quad (13)$$

$$u_{n_1, \dots, n_D}(x_1, \dots, x_k) = \prod_{k=1}^D \sin(n_k \pi x_k).$$

These experiments aim to calculate the smallest four eigenvalues of the Harmonic operator in 1, 2 and 5 dimensions. Since the PMNN model only computes the principal eigenvalue and cannot compute multiple eigenvalues simultaneously, it is not considered for comparison. NeuralEF, due to cumulative errors in its iterative orthogonalization process, experiences numerical instability in 2 and 5 dimensions, thus no data is available for these dimensions.

Firstly, as demonstrated in Table [1,](#page-6-0) the accuracy of STNet on all tasks is significantly better than that of existing methods. This enhancement primarily stems from the deflation projection. It effectively excludes solved invariant subspaces during the multi-eigenvalue solution process, thereby preserving the accuracy of multiple eigenvalues. This strongly validates the efficacy of our algorithm.

Secondly, in 5-dimension, STNet consistently maintains a precision improvement of at least three orders of magnitude. As shown in Table [2,](#page-5-0) this is largely due to the STNet computed eigenpairs having smaller residuals (defined as ||Lv − λv||2, see Appendix [B.3](#page-13-0) for details), indicating that STNet can effectively solve for accurate eigenvalues and eigenfunctions simultaneously.

Table 2: Residual comparison for eigenpairs of STNet and NeuralSVD for solving 5-dimensional Harmonic eigenvalue problems. The first row indicates the eigenpair index.

| Index     | ( v 1 , λ 1 ) | ( v 2 , λ 2 ) | ( v 3 , λ 3 ) | ( v 4 , λ 4 ) |
|-----------|---------------|---------------|---------------|---------------|
| NeuralSVD | 5.924e+0      | 5.920e+0      | 5.921e+0      | 5.920e+0      |
| STNet     | 4.864e-4      | 3.060e-3      | 5.980e-3      | 4.447e-3      |

Additionally, Table [1](#page-6-0) reveals that in the process of solving multiple eigenvalues, the errors for subsequent eigenvalues tend to be significantly higher than those for earlier ones. NeuralEF and NeuralSVD exhibit relatively stable error change, and But STNet shows fluctuations (for instance, errors for λ<sup>2</sup> and λ<sup>3</sup> at dimension five are smaller than those for λ1). This variability primarily arises because NeuralEF and NeuralSVD employ a uniform grid to acquire data points, whereas STNet uses uniform random sampling. While uniform random sampling inherently introduces some degree of randomness, it offers a significant advantage in high-dimensional settings. Specifically, a uniform grid necessitates an exponentially growing number of sampling points, num<sup>D</sup>, where num represents the number of grid points per dimension and D denotes the operator dimension. In contrast, uniform random sampling is not subject to this constraint, making it more scalable for high-dimensional problems.

#### 5.2. Schrodinger Oscillator Equation ¨

The Schrodinger oscillator equation is a common problem ¨ in quantum mechanics, and its time-independent form is expressed as follows:

$$-\frac{1}{2}\Delta\psi + V\psi = E\psi, \quad \text{in } \Omega = [0, 1]^D, \quad (14)$$

where ψ is the wave function, ∆ represents the Laplacian operator indicating the kinetic energy term, V is the potential energy within Ω, and E denotes the energy eigenvalue [\(Ryu et al.,](#page-9-4) [2024;](#page-9-4) [Griffiths & Schroeter,](#page-8-19) [2018\)](#page-8-19). This equation is formulated in natural units, simplifying the constants involved. Typically, the potential V (x1, . . . , xD) = 1 2 P<sup>D</sup> <sup>k</sup>=1 x 2 k characterizes a multidimensional quadratic potential. The principal eigenvalue E<sup>0</sup> and corresponding eigenfunction ψ<sup>0</sup> are given by:

$$E_0 = \frac{D}{2}, \quad \psi_0(x_1, \dots, x_D) = \prod_{k=1}^D \left( \frac{1}{\pi} \right)^{\frac{1}{4}} e^{-\frac{x_k^2}{2}}. \quad (15)$$

This experiment focuses on calculating the ground states of

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

Table 1: Absolute error comparison for eigenvalues of Harmonic operators. The first row lists the methods, the second row lists eigenvalue indexs, and the first column lists the operator dimensions. The most accurate method is in bold.

|         |        | NeuralEF      |        |        | NeuralSVD     |        |         | STNet         |        |
|---------|--------|---------------|--------|--------|---------------|--------|---------|---------------|--------|
|         | λ 1    | λ 2 λ 3       | λ 4    | λ 1    | λ 2 λ 3       | λ 4    | λ 1     | λ 2 λ 3       | λ 4    |
| Dim = 1 | 1.4e-1 | 2.9e+1 7.9e+1 | 1.4e+2 | 1.0e-1 | 4.1e+1 1.0e+0 | 1.4e+2 | 6.3e-10 | 1.7e-1 6.3e-1 | 1.6e+1 |
| Dim = 2 |        |               |        | 5.5e-2 | 2.1e-1 1.5e-1 | 2.6e+1 | 1.0e-5  | 3.0e-2 6.8e-2 | 1.0e-1 |
| Dim = 5 |        |               |        | 2.5e-1 | 2.9e+1 2.9e+1 | 2.9e+1 | 2.3e-4  | 9.5e-5 6.2e-5 | 1.3e-3 |

Table 3: Absolute error comparison for the principal eigenvalues of oscillator operators. The first row lists the methods, and the first column lists the operator dimensions. The most accurate method is in bold.

| Method |     | PMNN    | NeuralEF | NeuralSVD | STNet   |
|--------|-----|---------|----------|-----------|---------|
| Dim    | = 1 | 1.17e-6 | 2.57e-2  | 2.53e-2   | 3.62e-7 |
| Dim    | = 2 | 9.07e-5 | 7.55e-2  | 4.01e-1   | 2.35e-6 |
| Dim    | = 5 | 3.92e-1 | 3.97e-1  | 4.37e+0   | 3.23e-1 |

the Schrodinger equation in one, two, and five dimensions, ¨ i.e. the smallest principal eigenvalues.

Firstly, as shown in Table [3,](#page-6-1) the STNet achieves significantly higher precision than existing algorithms in computing the principal eigenvalues of the oscillator operator.

Furthermore, the accuracy of STNet surpasses that of PMNN. Both are designed based on the concept of the power method. When solving for the principal eigenvalue, the deflation projection loss may be considered inactive. This outcome suggests that the filter transform significantly enhances the accuracy.

## 5.3. Fokker-Planck Equation

The Fokker-Planck equation is central to statistical mechanics and is extensively applied across diverse fields such as thermodynamics, particle physics, and financial mathematics [\(Yang et al.,](#page-9-6) [2023;](#page-9-6) [Jordan et al.,](#page-8-20) [1998;](#page-8-20) [Frank,](#page-8-21) [2005\)](#page-8-21). It can be mathematically formulated as follows:

$$-\Delta v - V \cdot \nabla v - \Delta V v = \lambda v, \quad \text{in } \Omega = [0, 2\pi]^D,$$

$$V(x) = \sin \left( \sum_{i=1}^D c_i \cos(x_i) \right). \quad (16)$$

Here V (x) is a potential function with each coefficient c<sup>i</sup> varying within [0.1, 1], λ the eigenvalue, and v the eigenfunction. When the boundary conditions are periodic, there are multiple zero eigenvalues.

The eigenvalue at zero significantly impacts the numerical stability of the algorithm during iterative processes. This experiment investigates the computation of two zero eigenvalues for the Fokker-Planck equations with different parameters in 1, 2, and 5 dimensions. Due to the inherent limitation of the PMNN method, which can only compute a single eigenvalue, we restrict our analysis to calculating one eigenvalue when employing this approach.

As indicated in Table [4,](#page-7-0) the STNet algorithm significantly outperforms existing methods in computing the zero eigenvalues of the Fokker-Planck operator, effectively solving cases where the eigenvalue is zero. It is mainly due to the filter function, which performs a spectral transformation on the operator, converting the zero eigenvalue into other eigenvalues that are easier to calculate without changing the eigenvector.

#### 5.4. Comparative Experiment with Traditional Algorithms

This experiment compares the accuracy of STNet and the traditional finite difference method (FDM) with a central difference scheme under identical point distributions (6 × 10<sup>4</sup> points) [\(LeVeque,](#page-9-10) [2007\)](#page-9-10). Both methods compute the four smallest eigenvalues of the 5D harmonic operator.

As shown in Table [5,](#page-7-1) STNet significantly outperforms FDM in accuracy. While FDM's precision depends on grid density, requiring exponentially more grid points and parameters with increasing dimensionality, STNet employs uniform random sampling instead of fixed grids. Leveraging neural networks' expressive power, STNet achieves higher accuracy with fewer parameters by effectively approximating eigenfunctions.

Traditional algorithms and neural network-based algorithms each have their own applicable domains. In low-dimensional scenarios, traditional algorithms significantly outperform neural network-based algorithms in terms of computational speed, and their accuracy can be improved by increasing

394

396

Table 4: Absolute error comparison for the principal eigenvalues of Fokker-Planck operators across algorithms. The first row lists the methods, the second row lists eigenvalue index, the first column lists the Fokker-Planck parameter and the second column lists the operator dimensions. The most accurate method is in bold.

| Method  | PMNN    | NeuralEF |         | NeuralSVD |         |         | STNet   |
|---------|---------|----------|---------|-----------|---------|---------|---------|
| c i Dim | λ 1     | λ 1      | λ 2     | λ 1       | λ 2     | λ 1     | λ 2     |
| 1       | 1.16e+0 | 4.98e-2  | 1.05e+0 | 7.19e-1   | 1.02e+0 | 1.17e-3 | 8.75e-3 |
| 2       | 1.11e+0 | 6.71e-2  | 1.57e+0 | 3.33e-1   | 1.03e+0 | 5.26e-6 | 5.14e-2 |
| 5       | 1.17e+0 | 2.11e+0  | 9.17e+0 | 2.11e+0   | 4.82e+0 | 3.90e-3 | 1.29e-1 |
| 1       | 8.60e-1 | 5.21e-1  | 5.95e-1 | 2.73e-1   | 3.19e-1 | 3.86e-2 | 2.33e-1 |
| 2       | 8.30e-1 | 6.58e-1  | 8.45e-1 | 2.75e-1   | 3.94e-1 | 1.99e-2 | 3.91e-2 |
| 5       | 7.58e-1 | 7.71e-1  | 1.02e+0 | 2.01e-1   | 3.08e-1 | 5.64e-2 | 2.67e-2 |

Table 5: Absolute error comparison for eigenvalues of 5D Harmonic operators. The first column lists the methods, and the second column lists eigenvalue indexes

| Method | λ 1     | λ 2     | λ 3     | λ 4     |
|--------|---------|---------|---------|---------|
| FDM    | 4.05e-1 | 1.61e+0 | 1.61e+0 | 1.61e+0 |
| STNet  | 2.31e-4 | 9.54e-5 | 6.21e-5 | 1.39e-3 |

the number of grid points. However, in high-dimensional problems, the number of required grid points grows exponentially with the dimensionality. For instance, while a 2D problem requires a 100<sup>2</sup> grid, its 5D counterpart would need 100<sup>5</sup> grid points. In such cases, enhancing accuracy by increasing the number of grid points becomes impractical. Neural network-based algorithms, on the other hand, offer an effective solution to these high-dimensional challenges.

#### 5.5. Ablation Experiments

We conducted ablation experiments to validate the effectiveness of the deflation projection and filter transform modules. As shown in Table [6,](#page-7-2) the results for "w/o F" indicate that removing the filter transform significantly reduces solution accuracy. In the cases of "w/o D" and "w/o F and D," while the residuals remain small, the absolute errors for λ<sup>2</sup> and λ<sup>3</sup> are notably larger compared to λ1. This suggests that without the deflation projection module, the network converges exclusively to the first eigenfunction v<sup>1</sup> corresponding to λ1, failing to capture subsequent eigenfunctions. These findings underscore the critical roles of both modules: the filter transform enhances accuracy through spectral transformation. The deflation projection removes the subspace of already solved eigenfunctions from the search space, enabling the computation of multiple eigenvalues.

Additionally, experiments detailing the performance of STNet as a function of model depth, model width, and

Table 6: A comparison of different settings of STNet for the 2-dimensional Harmonic eigenvalue problem. "w/o" denotes the absence of a specific module, "F" represents the filter transform module, and "D" indicates the deflation projection module.

|             | Index |   |     |   | λ Absolute Error | Residual |
|-------------|-------|---|-----|---|------------------|----------|
| (           | v     | 1 | , λ | 1 | ) 1.02e-5        | 4.12e-3  |
| (           | v     | 2 | , λ | 2 | ) 3.04e-2        | 1.24e+1  |
| (           | v     | 3 | , λ | 3 | ) 6.76e-1        | 1.43e+1  |
| w/o F       |       |   |     |   |                  |          |
| (           | v     | 1 | , λ | 1 | ) 6.73e-5        | 1.35e-2  |
| (           | v     | 2 | , λ | 2 | ) 5.10e-2        | 4.72e+1  |
| (           | v     | 3 | , λ | 3 | ) 1.06e-1        | 1.70e+2  |
| w/o D       |       |   |     |   |                  |          |
| (           | v     | 1 | , λ | 1 | ) 1.42e-5        | 4.12e-3  |
| (           | v     | 2 | , λ | 2 | ) 2.96e+1        | 7.09e-3  |
| (           | v     | 3 | , λ | 3 | ) 2.97e+1        | 1.09e-2  |
| w/o F and D |       |   |     |   |                  |          |
| (           | v     | 1 | , λ | 1 | ) 6.73e-5        | 1.35e-2  |
| (           | v     | 2 | , λ | 2 | ) 2.96e+1        | 1.45e-2  |
| (           | v     | 3 | , λ | 3 | ) 2.97e+1        | 1.37e-2  |

the number of sampling points are provided in Appendix [C.](#page-14-0)

# 6. Conclusions

In this paper, we present STNet, a novel learning-based approach for solving operator eigenvalue problems. By leveraging approximate eigenvalues and eigenvectors obtained during iteration, STNet employs spectral transformations to reformulate the original operator, altering its spectral distribution to create an equivalent problem with improved convergence properties. Experimental results show that STNet outperforms existing algorithms in accuracy across a wide range of operator eigenvalue problems.

- 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Impact Statement This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here. References Banerjee, A. S., Lin, L., Hu, W., Yang, C., and Pask, J. E. Chebyshev polynomial filtered subspace iteration in the discontinuous galerkin method for large-scale electronic structure calculations. *The Journal of chemical physics*, 145(15), 2016. Ben-Shaul, I., Bar, L., and Sochen, N. Solving the functional eigen-problem using neural networks. *arXiv preprint arXiv:2007.10205*, 2020. Ben-Shaul, I., Bar, L., Fishelov, D., and Sochen, N. Deep learning solution of the eigenvalue problem for differential operators. *Neural Computation*, 35(6):1100–1134, 2023. Buchan, A., Pain, C., Fang, F., and Navon, I. A pod reducedorder model for eigenvalue problems with application to reactor physics. *International Journal for Numerical Methods in Engineering*, 95(12):1011–1032, 2013. Carleo, G. and Troyer, M. Solving the quantum many-body problem with artificial neural networks. *Science*, 355 (6325):602–606, 2017. Chen, Q. and Chan, Y. Integral finite element method for dynamical analysis of elastic–viscoelastic composite structures. *Computers & Structures*, 74(1):51–64, 2000. Choo, K., Mezzacapo, A., and Carleo, G. Fermionic neuralnetwork states for ab-initio electronic structure. *Nature communications*, 11(1):2368, 2020. Cuzzocrea, A., Scemama, A., Briels, W. J., Moroni, S., and Filippi, C. Variational principles in quantum monte carlo: The troubled story of variance minimization. *Journal of chemical theory and computation*, 16(7):4203–4212, 2020. Deng, Z., Shi, J., and Zhu, J. Neuralef: Deconstructing kernels by deep neural networks. In *International Conference on Machine Learning*, pp. 4976–4992. PMLR, 2022. Diao, H., Li, H., Liu, H., and Tang, J. Spectral properties of an acoustic-elastic transmission eigenvalue problem with applications. *Journal of Differential Equations*, 371: 629–659, 2023. Elhareef, M. H. and Wu, Z. Physics-informed neural network method and application to nuclear reactor calculations: A pilot study. *Nuclear Science and Engineering*, 197(4):601–622, 2023. Entwistle, M. T., Schatzle, Z., Erdman, P. A., Hermann, J., ¨ and Noe, F. Electronic excited states in deep variational ´ monte carlo. *Nature Communications*, 14(1):274, 2023. Evans, L. C. *Partial differential equations*, volume 19. American Mathematical Society, 2022. Frank, T. D. *Nonlinear Fokker-Planck equations: fundamentals and applications*. Springer Science & Business Media, 2005. Gerard, L., Scherbela, M., Marquetand, P., and Grohs, P. Gold-standard solutions to the schrodinger equation using ¨ deep learning: How much physics do we need? *Advances in Neural Information Processing Systems*, 35:10282– 10294, 2022. Golub, G. H. and Van Loan, C. F. *Matrix computations*. JHU press, 2013. Griffiths, D. J. and Schroeter, D. F. *Introduction to quantum mechanics*. Cambridge university press, 2018. Han, J., Lu, J., and Zhou, M. Solving high-dimensional eigenvalue problems using deep neural networks: A diffusion monte carlo like approach. *Journal of Computational Physics*, 423:109792, 2020. He, C., Hu, X., and Mu, L. A mesh-free method using piecewise deep neural network for elliptic interface problems. *Journal of Computational and Applied Mathematics*, 412: 114358, 2022. Hermann, J., Schatzle, Z., and No ¨ e, F. Deep-neural-network ´ solution of the electronic schrodinger equation. ¨ *Nature Chemistry*, 12(10):891–897, 2020. Hermann, J., Spencer, J., Choo, K., Mezzacapo, A., Foulkes,
  - W. M. C., Pfau, D., Carleo, G., and Noe, F. Ab initio ´ quantum chemistry with neural-network wavefunctions. *Nature Reviews Chemistry*, 7(10):692–709, 2023. Jin, H., Mattheakis, M., and Protopapas, P. Physicsinformed neural networks for quantum eigenvalue problems. In *2022 International Joint Conference on Neural Networks (IJCNN)*, pp. 1–8. IEEE, 2022. Jordan, R., Kinderlehrer, D., and Otto, F. The variational formulation of the fokker–planck equation. *SIAM journal on mathematical analysis*, 29(1):1–17, 1998. Kohn, W. Nobel lecture: Electronic structure of matter—wave functions and density functionals. *Reviews of Modern Physics*, 71(5):1253, 1999.

- 495 496 497 498 499 500 504 506 508 509 511 514 515 516 518 524 526 528 531 534 536 538 540 541 542 543 544 545 546 547 548 LeVeque, R. J. *Finite difference methods for ordinary and partial differential equations: steady-state and timedependent problems*. SIAM, 2007. Miao, C.-Q. and Wu, W.-T. On relaxed filtered krylov subspace method for non-symmetric eigenvalue problems. *Journal of Computational and Applied Mathematics*, 398: 113698, 2021. Morgan, R. B. and Zeng, M. Harmonic projection methods for large non-symmetric eigenvalue problems. *Numerical linear algebra with applications*, 5(1):33–55, 1998. Nguyen, T., Raghu, M., and Kornblith, S. Do wide and deep networks learn the same things? uncovering how neural network representations vary with width and depth. *arXiv preprint arXiv:2010.15327*, 2020. Pfau, D., Petersen, S., Agarwal, A., Barrett, D. G., and Stachenfeld, K. L. Spectral inference networks: Unifying deep and spectral learning. *arXiv preprint arXiv:1806.02215*, 2018. Pfau, D., Spencer, J. S., Matthews, A. G., and Foulkes, W.
  - M. C. Ab initio solution of the many-electron schrodinger ¨ equation with deep neural networks. *Physical review research*, 2(3):033429, 2020. Pfau, D., Axelrod, S., Sutterud, H., von Glehn, I., and Spencer, J. S. Natural quantum monte carlo computation of excited states. *arXiv preprint arXiv:2308.16848*, 2023. Ryu, J. J., Xu, X., Erol, H., Bu, Y., Zheng, L., and Wornell,
  - G. W. Operator svd with neural networks via nested lowrank approximation. *arXiv preprint arXiv:2402.03655*, 2024. Saad, Y. *Numerical methods for large eigenvalue problems: revised edition*. SIAM, 2011. Salas, P., Giraud, L., Saad, Y., and Moreau, S. Spectral recycling strategies for the solution of nonlinear eigenproblems in thermoacoustics. *Numerical Linear Algebra with Applications*, 22(6):1039–1058, 2015. Schutt, K., Kindermans, P.-J., Sauceda Felix, H. E., Chmiela, ¨ S., Tkatchenko, A., and Muller, K.-R. Schnet: A ¨ continuous-filter convolutional neural network for modeling quantum interactions. *Advances in neural information processing systems*, 30, 2017. Van Beeumen, R. Rational krylov methods for nonlinear eigenvalue problems, 2015. Watkins, D. S. *The matrix eigenvalue problem: GR and Krylov subspace methods*. SIAM, 2007. Winkelmann, J., Springer, P., and Napoli, E. D. Chase: Chebyshev accelerated subspace iteration eigensolver for sequences of hermitian eigenvalue problems. *ACM Transactions on Mathematical Software (TOMS)*, 45(2):1–34, 2019. Yang, Q., Deng, Y., Yang, Y., He, Q., and Zhang, S. Neural networks based on power method and inverse power method for solving linear eigenvalue problems. *Computers & Mathematics with Applications*, 147:14–24, 2023. Yu, B. et al. The deep ritz method: a deep learning-based numerical algorithm for solving variational problems. *Communications in Mathematics and Statistics*, 6(1):1–12, 2018.

554

556

558

560

564 <sup>6</sup> if δ < ϵ then

566

568

571

574

576

578

594

596

598

# A. Background Knowledge and Relevant Analysis

## A.1. Convergence Analysis of the Power Method

Suppose A ∈ R <sup>n</sup>×<sup>n</sup> and V <sup>−</sup><sup>1</sup>AV = diag(λ1, . . . , λn) with V = v<sup>1</sup> · · · v<sup>n</sup> . Assume that |λ1| > |λ2| ≥ · · · ≥ |λn|. The pseudocode for the power method is shown below [\(Golub & Van Loan,](#page-8-18) [2013\)](#page-8-18):

Algorithm 1: Power method for finding the largest principal eigenvalue of the matrix A

<sup>1</sup> Given A ∈ R

<sup>n</sup>×<sup>n</sup> an n × n matrix, an arbitrary unit vector x

(0) ∈ <sup>R</sup>

<sup>n</sup>, the maximum number of iterations kmax, and the

stopping criterion ϵ. <sup>2</sup> for k = 1, 2, . . . , k*max* do

<sup>3</sup> Compute y

(k) = Ax(k−1)

.

<sup>4</sup> Normalize x (k) = y ∥y(k)∥ . <sup>5</sup> Compute the difference δ = ∥x (k) − x (k−1)∥.

(k)

<sup>7</sup> Record the largest principal eigenvalue using the Rayleigh quotient,

$$\lambda^{(k)} = \frac{\langle \mathbf{x}^{(k)}, \mathbf{A}\mathbf{x}^{(k)} \rangle}{\langle \mathbf{x}^{(k)}, \mathbf{x}^{(k)} \rangle}.$$

The stopping criterion is met, the iteration can be stopped.

Let us examine the convergence properties of the power iteration. If

$$\mathbf{x}^{(0)} = a_1 \mathbf{v}_1 + a_2 \mathbf{v}_2 + \cdots + a_n \mathbf{v}_n$$

and v<sup>1</sup> ̸= 0, then

$$\mathbf{A}^k \mathbf{x}^{(0)} = a_1 \lambda_1^k \left( \mathbf{v}_1 + \sum_{j=2}^n \frac{a_j}{a_1} \left( \frac{\lambda_j}{\lambda_1} \right)^k \mathbf{v}_j \right).$$

Since x (k) ∈ span{A<sup>k</sup>x (0)}, we conclude that

$$\text{dist}\left(\text{span}\{\mathbf{x}^{(k)}\}, \text{span}\{\mathbf{v}_1\}\right) = O\left(\left(\frac{\lambda_2}{\lambda_1}\right)^k\right).$$

It is also easy to verify that

$$|\lambda_1 - \lambda^{(k)}| = O\left(\left(\frac{\lambda_2}{\lambda_1}\right)^k\right).$$

Since λ<sup>1</sup> is larger than all the other eigenvalues in modulus, it is referred to as the largest principal eigenvalue. Thus, the power method converges if λ<sup>1</sup> is the largest principal and if x (0) has a component in the direction of the corresponding dominant eigenvector x1.

In practice, the effectiveness of the power method largely depends on the ratio |λ2|/|λ1|, as this ratio determines the convergence rate. Therefore, applying specific spectral transformations to the matrix to increase this ratio can significantly accelerate the convergence of the power method.

## A.2. Deflation Projection Details

Consider the scenario where we have determined the largest modulus eigenvalue, λ1, and its corresponding eigenvector, v1, utilizing an algorithm such as the power method. These algorithms consistently identify the eigenvalue of the largest modulus from the given matrix along with an associated eigenvector. We ensure that the vector v<sup>1</sup> is normalized such that

∥v1∥<sup>2</sup> = 1. The task then becomes computing the subsequent eigenvalue, λ2, of the matrix A. A traditional approach to address this is through what is commonly known as a deflation procedure. This technique involves a rank-one modification to the original matrix, aimed at shifting the eigenvalue λ<sup>1</sup> while preserving all other eigenvalues intact. The modification is designed in such a way that λ<sup>2</sup> emerges as the eigenvalue with the largest modulus in the adjusted matrix. Consequently, the power method can be reapplied to this updated matrix to extract the eigenvalue-eigenvector pair λ2, v2.

When the invariant subspace requiring deflation is one-dimensional, consider the following Proposition [A.1.](#page-11-1) The propositions and proofs below are derived from [Saad](#page-9-9) [\(2011\)](#page-9-9) P90.

Proposition A.1. *Let* v<sup>1</sup> *be an eigenvector of* A *of norm 1, associated with the eigenvalue* λ<sup>1</sup> *and let* A<sup>1</sup> ≡ A − σv1v H 1 *. Then the eigenvalues of* A<sup>1</sup> *are* λ˜ <sup>1</sup> = λ<sup>1</sup> − σ *and* λ˜ <sup>j</sup> = λ<sup>j</sup> , j = 2, 3, . . . , n*. Moreover, the Schur vectors associated with* λ˜ <sup>j</sup> , j = 1, 2, 3, . . . , n *are identical with those of* A*.*

*Proof.* Let AV = V R be the Schur factorization of A, where R is upper triangular and V is orthonormal. Then we have

$$\mathbf{A}_1 \mathbf{V} = [\mathbf{A} - \sigma v_1 v_1^\top] \mathbf{V} = \mathbf{V} \mathbf{R} - \sigma v_1 e_1^\top = \mathbf{V} [\mathbf{R} - \sigma e_1 e_1^\top].$$

Here, e<sup>1</sup> is the first standard basis vector. The result follows immediately.

According to Proposition [A.1,](#page-11-1) once the eigenvalue λ<sup>1</sup> and eigenvector v<sup>1</sup> are known, we can define the deflation projection matrix P<sup>1</sup> = I − λ1v1v ⊤ 1 to compute the remaining eigenvalues and eigenvectors.

When deflating with multiple vectors, let q1, q2, . . . , q<sup>j</sup> be a set of Schur vectors associated with the eigenvalues λ1, λ2, . . . , λ<sup>j</sup> . We denote by Q<sup>j</sup> the matrix of column vectors q1, q2, . . . , q<sup>j</sup> . Thus, Q<sup>j</sup> ≡ [q1, q2, . . . , q<sup>j</sup> ] is an orthonormal matrix whose columns form a basis of the eigenspace associated with the eigenvalues λ1, λ2, . . . , λ<sup>j</sup> . An immediate generalization of Proposition [A.1](#page-11-1) is the following [\(Saad,](#page-9-9) [2011\)](#page-9-9) P94.

Proposition A.2. *Let* Σ<sup>j</sup> *be the* j × j *diagonal matrix* Σ<sup>j</sup> = *diag*(σ1, σ2, . . . , σ<sup>j</sup> )*, and* Q<sup>j</sup> *an* n × j *orthogonal matrix consisting of the Schur vectors of* A *associated with* λ1, . . . , λ<sup>j</sup> *. Then the eigenvalues of the matrix*

$$A_j \equiv A - Q_j \Sigma_j Q_j^\top,$$

*are* λ˜ <sup>i</sup> = λ<sup>i</sup> − σ<sup>i</sup> *for* i ≤ j *and* λ˜ <sup>i</sup> = λ<sup>i</sup> *for* i > j*. Moreover, its associated Schur vectors are identical with those of* A*.*

*Proof.* Let AU = UR be the Schur factorization of A. We have

$$\mathbf{A}_j \mathbf{U} = [\mathbf{A} - \mathbf{Q}_j \mathbf{\Sigma}_j \mathbf{Q}_j^\top] \mathbf{U} = \mathbf{U} \mathbf{R} - \mathbf{Q}_j \mathbf{\Sigma}_j \mathbf{E}_j^\top,$$

where E<sup>j</sup> = [e1, e2, . . . , e<sup>j</sup> ]. Hence

$$\mathbf{A}_j \mathbf{U} = \mathbf{U} [\mathbf{R} - \mathbf{E}_j \Sigma_j \mathbf{E}_j^\top]$$

and the result follows.

According to Proposition [A.2,](#page-11-2) if A is a normal matrix and the eigenvalues λ1, . . . , λ<sup>j</sup> along with their corresponding eigenvectors v1, . . . , v<sup>j</sup> are known, we can construct the deflation projection matrix P<sup>j</sup> = I − VjΣjV ⊤ j to compute the remaining eigenvalues and eigenvectors. Here, Σ<sup>j</sup> = diag(σ1, σ2, . . . , σ<sup>j</sup> ) and V<sup>j</sup> = [v1, v2, . . . , v<sup>j</sup> ].

#### A.3. Filtering Technique

The primary objective of filtering techniques is to manipulate the eigenvalue distribution of a matrix through spectral transformations [\(Saad,](#page-9-9) [2011\)](#page-9-9). This enhances specific eigenvalues of interest, facilitating their recognition and computation by iterative solvers. Filter transformation functions, F(x), typically fall into two categories:

- 1. Polynomial Filters, expressed as P(x), such as the Chebyshev filter [\(Miao & Wu,](#page-9-12) [2021;](#page-9-12) [Banerjee et al.,](#page-8-22) [2016\)](#page-8-22).
- 2. Rational Function Filters, often denoted as P(x)/Q(x), such as the shift-invert method [\(Van Beeumen,](#page-9-13) [2015;](#page-9-13) [Watkins,](#page-9-1) [2007\)](#page-9-1). Below we describe this strategy in detail.

689 690

694

696

698

700

704

706

708 709

711

714

Shift-Invert Strategy The shift-invert strategy applies the transformation (A − σI) −1 to the matrix A, where σ is a scalar approximating a target eigenvalue, termed as shift. This operation transforms each eigenvalue λ of A into <sup>1</sup> λ−σ , amplifying those eigenvalues close to σ in the transformed matrix, making them larger and more distinguishable [\(Watkins,](#page-9-1) [2007\)](#page-9-1).

For instance, consider the power method, where the convergence rate is primarily governed by the ratio of the matrix's largest modulus eigenvalue to its second largest. Suppose matrix A has three principal eigenvalues: λ<sup>1</sup> = 10, λ<sup>2</sup> = 3, and λ<sup>3</sup> = 2. Our objective is to compute λ1, the largest eigenvalue. In the original matrix A, the convergence rate of the power method hinges on the spectral gap ratio, defined as:

$$\text{Spectral Gap Ratio} = \frac{\lambda_1}{\lambda_2} \approx 3.33$$

Applying the shift-invert transformation with σ = 9.5 strategically selected close to λ1, the new eigenvalues µ are recalculated as:

$$\mu_i = \frac{1}{\lambda_i - \sigma}$$

This results in transformed eigenvalues:

$$\mu_1 = 2, \quad \mu_2 \approx -0.133, \quad \mu_3 \approx -0.125$$

Under this transformation, µ<sup>1</sup> = 2 emerges as the dominant eigenvalue in the new matrix, with the other eigenvalues significantly smaller. Consequently, the new spectral gap ratio escalates to:

$$\text{New Spectral Gap Ratio} = \frac{2}{0.133} \approx 15.04$$

This enhanced spectral gap notably accelerates the convergence of the power method in the new matrix configuration.

Filtering techniques are often synergized with techniques like the implicit restarts of Krylov algorithms [\(Watkins,](#page-9-1) [2007;](#page-9-1) [Golub & Van Loan,](#page-8-18) [2013\)](#page-8-18), employing matrix operation optimizations to minimize the computational demands of evaluating matrix functions. These methods enable more precise localization and computation of multiple eigenvalues spread across the spectral range, particularly vital in physical [\(Salas et al.,](#page-9-14) [2015;](#page-9-14) [Banerjee et al.,](#page-8-22) [2016\)](#page-8-22) and materials science [\(Kohn,](#page-8-23) [1999\)](#page-8-23) simulations where these eigenvalues frequently correlate with the system's fundamental properties [\(Winkelmann et al.,](#page-9-15) [2019\)](#page-9-15).

# B. Details of Experimental Setup

## B.1. Experimental Environment

To ensure consistency in our evaluations, all comparative experiments were conducted under uniform computing environments. Specifically, the environments used are detailed as follows:

- CPU: 72 vCPU AMD EPYC 9754 128-Core Processor
- GPU: NVIDIA GeForce RTX 4090D (24GB)

## B.2. Experimental Parameters

- NeuralSVD and NeuralEF: (Using the original paper settings)
  - Optimizer: RMSProp with a learning rate scheduler.
  - Learning rate: 1e-4, batch size: 128
  - Neural Network Architecture: layers = [128,128,128]
  - Laplacian regularization set to 0.01, with evaluation frequency every 10000 iterations.
  - Fourier feature mapping enabled with a size of 1024 and scale of 0.1.
  - Neural network structure: hidden layers of 128,128,128 using softplus as the activation function.
  - For the 1-dimensional problem, the number of points is 20, 000, with 400, 000 iterations. For the 2-dimensional problem, the number of points is 40, 000 = 200 × 200, also with 400, 000 iterations. For the 5-dimensional problem, the number of points is 59, 049 = 9<sup>5</sup> , with 500, 000 iterations.

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

- STNet
  - Optimizer: Adam
  - Learning rate: 1e-4
  - Neural Network Architecture: Assuming d is the dimension of the problem. For d = 1 or 2, layers = [d, 20, 20, 20, 20, 1] (For Harmonic operator d=2, layers = [d, 20, 20, 20, 1]). For d=5, layers = [d, 40, 40, 40, 40, 1]. For else case, layers = [d, 40, 40, 40, 40, 1].
  - For the 1-dimensional problem, the number of points is 20, 000, with 400, 000 iterations. For the 2-dimensional problem, the number of points is 40, 000 = 200 × 200, also with 400, 000 iterations. For the 5-dimensional problem, the number of points is 59, 049 = 9<sup>5</sup> , with 500, 000 iterations.

#### B.3. Error Metrics

- Absolute Error: We employ absolute error to estimate the bias of the output eigenvalues of the model:

Absolute Error = 
$$|\tilde{\lambda} - \lambda|$$
. (17)

Here λ˜ represents the eigenvalue predicted by the model, while λ denotes the true eigenvalue.

- Residual Error: To further analyze the error in eigenpair (˜v, λ˜) predictions, we use the following metric:

$$\text{Residual Error} = \|\mathcal{L}\tilde{v} - \tilde{\lambda}\tilde{v}\|_2. \quad (18)$$

Here, v˜ represents the eigenfunction predicted by the model. When λ˜ is the true eigenvalue and v˜ is the true eigenfunction, the Residual Error equals 0.

774

776

778

794

796

800

804

806

808

# C. Analysis of Hyperparameters

## Model Depth:

Table 7: Consider the 2-dimensional Harmonic problem, with the fixed layer width of 20, and compare the performance of STNet at different model layers. Other experimental details are the same as Appendix [B.2.](#page-12-1)

| Layer | Index |   |     |   | λ Absolute Error | Residual |
|-------|-------|---|-----|---|------------------|----------|
| (     | v     | 1 | , λ | 1 | ) 1.02e-5        | 4.56e-3  |
| (     | v     | 2 | , λ | 2 | ) 3.04e-2        | 2.56e+1  |
| ( 3   | v     | 3 | , λ | 3 | ) 6.76e-2        | 6.99e+1  |
| (     | v     | 4 | , λ | 4 | ) 1.00e-1        | 2.12e+3  |
| (     | v     | 1 | , λ | 1 | ) 1.42e-5        | 4.12e-3  |
| (     | v     | 2 | , λ | 2 | ) 2.96e-1        | 1.24e+1  |
| ( 4   | v     | 3 | , λ | 3 | ) 4.17e-1        | 1.43e+1  |
| (     | v     | 4 | , λ | 4 | ) 2.00e+1        | 2.17e+5  |
| (     | v     | 1 | , λ | 1 | ) 4.36e-6        | 4.12e-3  |
| (     | v     | 2 | , λ | 2 | ) 8.63e-1        | 3.12e+1  |
| ( 5   | v     | 3 | , λ | 3 | ) 1.98e+0        | 1.58e+3  |
| (     | v     | 4 | , λ | 4 | ) 8.94e+1        | 2.09e+3  |
| (     | v     | 1 | , λ | 1 | ) 1.06e-5        | 9.56e-3  |
| (     | v     | 2 | , λ | 2 | ) 8.21e-1        | 2.00e+1  |
| ( 6   | v     | 3 | , λ | 3 | ) 1.17e+0        | 9.90e+3  |
| (     | v     | 4 | , λ | 4 | ) 3.81e+1        | 7.53e+4  |

# Model Width:

Table 8: Consider the 2-dimensional Harmonic problem, with the fixed layer depth of 3, and compare the performance of STNet at different model widths. Other experimental details are the same as Appendix [B.2.](#page-12-1)

| Width | Index |   |     |   | λ Absolute Error | Residual |
|-------|-------|---|-----|---|------------------|----------|
| (     | v     | 1 | , λ | 1 | ) 1.68e-6        | 1.26e-3  |
| (     | v     | 2 | , λ | 2 | ) 3.82e-1        | 2.36e+0  |
| ( 10  | v     | 3 | , λ | 3 | ) 7.54e-1        | 1.20e+2  |
| (     | v     | 4 | , λ | 4 | ) 1.71e-1        | 2.49e+3  |
| (     | v     | 1 | , λ | 1 | ) 1.42e-5        | 4.12e-3  |
| (     | v     | 2 | , λ | 2 | ) 2.96e-1        | 1.24e+1  |
| ( 20  | v     | 3 | , λ | 3 | ) 4.17e-1        | 1.43e+1  |
| (     | v     | 4 | , λ | 4 | ) 2.00e+1        | 2.17e+5  |
| (     | v     | 1 | , λ | 1 | ) 3.26e-5        | 2.25e-2  |
| (     | v     | 2 | , λ | 2 | ) 1.50e+0        | 2.10e+1  |
| ( 30  | v     | 3 | , λ | 3 | ) 1.59e+0        | 8.21e+3  |
| (     | v     | 4 | , λ | 4 | ) 3.52e+2        | 2.77e+5  |
| (     | v     | 1 | , λ | 1 | ) 1.57e-5        | 2.06e-2  |
| (     | v     | 2 | , λ | 2 | ) 2.67e+0        | 5.03e+1  |
| ( 40  | v     | 3 | , λ | 3 | ) 7.93e+1        | 5.76e+3  |
| (     | v     | 4 | , λ | 4 | ) 1.50e+2        | 1.47e+4  |

## The Number of Points:

Table 9: Consider the 2-dimensional Harmonic problem and compare the performance of STNet at different numbers of points. Other experimental details are the same Appendix [B.2.](#page-12-1)

| Number Index |   |     |   | λ Absolute Error | Residual |
|--------------|---|-----|---|------------------|----------|
| ( v          | 1 | , λ | 1 | ) 1.11e-5        | 3.19e-3  |
| ( v          | 2 | , λ | 2 | ) 1.25e+0        | 3.22e+0  |
| ( v          | 3 | , λ | 3 | ) 1.61e+0        | 1.27e+2  |
| ( v          | 1 | , λ | 1 | ) 4.40e-5        | 7.09e-3  |
| ( v          | 2 | , λ | 2 | ) 3.58e-1        | 2.71e+0  |
| ( v          | 3 | , λ | 3 | ) 1.70e-1        | 5.62e+1  |
| ( v          | 1 | , λ | 1 | ) 1.42e-5        | 4.12e-3  |
| ( v          | 2 | , λ | 2 | ) 2.96e-1        | 1.24e+1  |
| ( v          | 3 | , λ | 3 | ) 4.17e-1        | 1.43e+1  |
| ( v          | 1 | , λ | 1 | ) 4.94e-6        | 6.63e-3  |
| ( v          | 2 | , λ | 2 | ) 2.53e-1        | 2.46e+1  |
| ( v          | 3 | , λ | 3 | ) 3.73e-1        | 1.50e+3  |

The influence of model depth, model width, and the number of points on STNet is illustrated in Tables [7,](#page-14-1) [8,](#page-14-2) and [9,](#page-15-0) respectively. Experimental results indicate that STNet is relatively unaffected by changes in model depth and model width. However, it is significantly influenced by the number of points, with performance improving as more points are used.