011

014 015 016

018

024

026

034

036

038

054

# SymMaP: Improving Computational Efficiency in Linear Solvers through Symbolic Preconditioning

Anonymous Authors<sup>1</sup>

## Abstract

Matrix preconditioning is a critical technique to accelerate the solving of linear systems, where performance heavily depends on the selection of preconditioning parameters. Traditional parameter selection approaches often define fixed constants for specific scenarios. However, they rely on domain expertise and fail to consider the instance-wise features for individual problems, limiting their performance. In contrast, machine learning (ML) approaches, though promising, are hindered by high inference costs and limited interpretability. To combine the strengths of both approaches, we propose a symbolic discovery framework—namely, Symbolic Matrix Preconditioning (SymMaP)—to learn efficient symbolic expressions for preconditioning parameters. Specifically, we employ a large neural network to search the high-dimensional discrete space for expressions that can accurately predict the optimal parameters. The learned expression allows for high inference efficiency and excellent interpretability (expressed in concise symbolic formulas), making it simple and reliable for deployment. Experimental results show that SymMaP consistently outperforms traditional strategies across various benchmarks.

# 1. Introduction

Linear systems are foundational in the machine learning, physics, engineering, and other scientific fields [\(Leon et al.,](#page-9-0) [2006;](#page-9-0) [LeVeque,](#page-9-1) [2007\)](#page-9-1). Since analytical solutions are often unavailable, efficient numerical algorithms become essential [\(Demmel,](#page-8-0) [1997\)](#page-8-0). Matrix preconditioning, a key technique in this domain, accelerates iterative solvers and improves computational stability [\(Trefethen & Bau,](#page-9-2) [2022;](#page-9-2) [Chen,](#page-8-1) [2005\)](#page-8-1). For instance, the successive over-relaxation (SOR) method optimizes convergence by integrating Gauss-Seidel iterations with a weighted update scheme governed by the over-relaxation factor ω [\(Golub & Van Loan,](#page-8-2) [2013\)](#page-8-2).

The effectiveness of matrix preconditioning depends on

key parameters, such as ω in SOR. Selecting ω > 1 can accelerate convergence, while ω < 1 may stabilize the process. This trade-off makes ω a critical parameter, directly influencing the performance of preconditioning.

Traditional parameter selection strategies often rely on domain expertise to define fixed constants for specific scenarios. However, (challenge 1) different problem parameters often require distinct optimal preconditioning parameters. Traditional strategies ignore instance-wise features—specific characteristics of individual problems, such as equation coefficients. This limits their adaptability to varying problem instances and tasks.

In contrast, machine learning (ML) approaches hold great promise but come with other challenges. First, (challenge 2) ML inference,while efficient on GPUs, performs poorly in CPU-only environments due to limited parallel processing capabilities. This is particularly problematic in linear system solver deployments, where GPU resources are often unavailable. Second, (challenge 3) the "black-box" nature of many ML techniques hinders a deeper understanding of the learned policies, raising concerns about their reliability.

In light of this, a natural solution is to combine the reliability and superior performance of these two paradigms. We propose a symbolic discovery framework—namely, Symbolic Matrix Preconditioning (SymMaP)—to learn efficient symbolic expressions for preconditioning parameters. The framework consists of three main steps. SymMaP first begins by applying grid search to identify the optimal preconditioning parameters based on task-specific performance metrics. Next, the framework performs a risk-seeking search in the high-dimensional discrete space of symbolic expressions, evaluating the best-found symbolic expression using a risk-seeking strategy. Finally, these symbolic expressions can be directly integrated into the modern solvers for linear systems, significantly improving computational efficiency.

The key contributions and advantages of SymMaP are summarized as follows:

- We propose a symbolic discovery framework, SymMaP, to learn efficient symbolic expressions for preconditioning parameters.

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

- SymMaP exhibits excellent generalization, making it adaptable to a wide range of preconditioning methods and optimization objectives.
- The symbolic expressions derived by SymMaP are both interpretable and easy to integrate into solver environments, offering a practical and transparent approach to enhancing the performance of linear system solvers.

## 2. Preliminaries

#### 2.1. Matrix Preconditioning Technique

Matrix preconditioning is a technique employed to accelerate the convergence of iterative solvers and enhance the stability of algorithms. It is generally employed in solving linear systems [\(Chen,](#page-8-1) [2005;](#page-8-1) [Golub & Van Loan,](#page-8-2) [2013\)](#page-8-2), which are typically expressed in the form:

$$Ax = b. \quad (1)$$

The fundamental idea of preconditioning is to transform the original problem into an equivalent one with better numerical properties. Specifically, this technique involves finding a preconditioner M that approximates either the inverse of A or some form conducive to iterative solutions [\(Chen,](#page-8-1) [2005\)](#page-8-1). Consequently, the original [\(1\)](#page-1-0) is transformed into

$$MAx = Mb. \quad (2)$$

$$MAx = Mb. \quad (2)$$

There are generally two optimization objectives: 1. to accelerate the convergence of iterations by altering the spectral distribution of the matrix A. 2. to reduce the condition number of the matrix A, thereby lessening its ill-conditioning and enhancing the stability of iteration. Some common preconditioning techniques include the Jacobi, Gauss-Seidel, SOR [\(Young,](#page-9-3) [1954\)](#page-9-3), algebraic multigrid (AMG) [\(Trotten](#page-9-4)[berg et al.,](#page-9-4) [2000\)](#page-9-4), etc.

#### 2.2. Prefix Notation

Prefix notation is a mathematical format where every operator precedes its operands, eliminating the need for parentheses required in conventional infix notation and simplifying symbolic manipulation. This representation is particularly advantageous in symbolic regression, as it allows mathematical expressions to be expressed as sequences of tokens that can be easily processed by neural networks.

In this notation, operators can be unary (e.g., sin, cos) or binary (e.g., +, −, ×, ÷), while operands can be constants or variables [\(Landajuela et al.,](#page-8-3) [2021\)](#page-8-3). Each prefix expression uniquely corresponds to a symbolic tree structure, facilitating the conversion back to the original mathematical expression [\(Lample & Charton,](#page-8-4) [2019\)](#page-8-4).

![](_page_1_Figure_1.jpeg)

Figure 1: Variation in iteration counts and computation times under different SOR preconditioning parameters applied to a linear system from a second-order elliptic PDE.

The sequential nature of prefix notation aligns well with the architecture of recurrent neural networks (RNNs), which process information step by step. Unlike infix notation which may require variable-length look-ahead to determine the next valid token, prefix notation allows RNN to generate expressions through an auto-regressive process where each decision is well-defined based on previous tokens, and by removing the need for parentheses, it reduces the vocabulary size of possible tokens, which greatly enhances model training efficiency.

## 3. Motivation

The selection of matrix preconditioning parameters significantly affects their effectiveness [\(Chen,](#page-8-1) [2005\)](#page-8-1). To design appropriate algorithmic prediction parameters, we first analyze the optimization space for preconditioning parameter selection and investigate the existence of optimal parameters. Next, we analyze the unique challenges present in this scenario. Finally, to address these challenges, we design a symbolic discovery framework to select the parameters.

#### 3.1. Motivation for Optimizing Preconditioning Parameters

As illustrated in figure [1,](#page-1-1) the choice of relaxation factors ω significantly impacts the iteration count and computation time, when solving a second-order elliptic partial differential equation (PDE) [\(Evans,](#page-8-5) [2022\)](#page-8-5) using SOR preconditioning [\(Golub & Van Loan,](#page-8-2) [2013\)](#page-8-2). There exists an optimal parameter ωop that minimizes the computation time, with specific details available in the Appendix [B.2.](#page-11-0)

To further analyze the optimization space of preconditioning parameters, we evaluate the impact of various parameter selection strategies on preconditioning performance.

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

![](_page_2_Figure_0.jpeg)

Figure 2: Ratio of average computation times at various tolerances to default parameter times under different SOR parameter selection schemes, evaluated on the second-order elliptic PDE dataset.

As shown in Figure [2,](#page-2-0) the 'Optimal Parameter ωop' represents the parameter that minimizes computation time in each experiment, serving as the theoretical upper limit of our optimization. The 'Optimal Fixed Constant' refers to a fixed constant that minimizes average computation time, and 'Default Parameter' corresponds to the default setting of ω = 1 in the portable extensible toolkit for scientific computation (PETSc) [\(Balay et al.,](#page-8-6) [2024\)](#page-8-6). The gap between the optimal fixed constant and the optimal parameter highlights significant potential for optimizing preconditioning parameter selection, motivating this paper. The performance of our SymMaP algorithm approaches the optimal parameter, demonstrating its accuracy in learning the optimal parameter expression.

#### 3.2. Challenges in Predicting Efficient Preconditioning Parameters

We aim to develop a universal framework for predicting efficient parameters. However, the context of solving linear systems imposes specific challenges:

(C1) Strong Generalization Capability: Real-world scientific computing scenarios vary significantly. For instance, the choice of PDE grid form can lead to significant variations in matrix structure [\(Johnson,](#page-8-7) [2009\)](#page-8-7), resulting in distinct optimal parameters. Moreover, preconditioning addresses various optimization goals, such as reducing computational time, the number of iterations, and lowering condition numbers [\(Chen,](#page-8-1) [2005\)](#page-8-1). This necessitates that parameter prediction algorithms possess robust generalization capabilities: they should take problem scenarios and features as inputs while applying them to different preconditioning methods and optimization goals.

(C2) Computational Efficiency: linear system solver typically relies on Krylov subspace methods implemented in low-level libraries optimized for CPU architectures, such as PETSc [\(Balay et al.,](#page-8-6) [2024\)](#page-8-6), LAPACK [\(Anderson et al.,](#page-8-8) [1999\)](#page-8-8). Algorithms like generalized minimal residual method (GMRES) [\(Saad & Schultz,](#page-9-5) [1986\)](#page-9-5) and conjugate gradient (CG) [\(Greenbaum,](#page-8-9) [1997\)](#page-8-9) iteratively compute the matrix's invariant subspace, favoring single-threaded or limited multithreaded execution modes. Preconditioning techniques aim to accelerate these solvers without significant additional computational overhead, often adopting implicit iterative formats (e.g., SOR [\(Chen,](#page-8-1) [2005\)](#page-8-1)) or utilizing low-cost matrix decompositions (e.g., AMG [\(Trottenberg et al.,](#page-9-4) [2000\)](#page-9-4)). Therefore, any parameter prediction algorithms must be compatible with CPU environments and seamlessly integrated into existing algorithm libraries. At the same time, it must maintain low computational costs to preserve the performance benefits of preconditioning.

(C3) Algorithmic Transparency: Algorithms in scientific computing often require rigorous analysis under mathematical theories. Opaque prediction algorithms could confuse researchers. For instance, the relaxation factor ω in SOR needs to avoid being too close to 0 or 2 in some scenarios [\(Agarwal,](#page-8-10) [2000\)](#page-8-10). This is an issue that opaque algorithms cannot avoid in advance. Moreover, interpretable algorithms can guide researchers to conduct further studies and reveal the underlying mathematical structures of problems. Therefore, these pose challenges to the transparency and interpretability of the parameter prediction algorithms.

#### 3.3. Symbolic Discovery to Preconditioning Parameter Selection

Symbolic discovery extracts mathematical expressions from data, establishing relationships between problem features and optimal preconditioning parameters. Its integration into matrix preconditioning overcomes parameter selection challenges through a generalizable, efficient, and transparent approach.

Firstly, symbolic discovery can accommodate various types of input parameters and can specifically tailor symbolic expression learning for different preconditioning methods and optimization goals [\(Cranmer et al.,](#page-8-11) [2020\)](#page-8-11), thereby meeting the requirement for broad applicability in scientific computing tasks (C1).

Secondly, the explicit expressions derived are computationally lightweight and can be quickly evaluated at runtime. They integrate seamlessly into existing CPU-based algorithm libraries like PETSc [\(Balay et al.,](#page-8-6) [2024\)](#page-8-6) with almost no overhead (C2).

Thirdly, the symbolic discovery provides transparent and interpretable expressions [\(Rudin,](#page-9-6) [2019\)](#page-9-6), allowing researchers

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

![](_page_3_Diagram_0.jpeg)

Figure 3: Illustration of how SymMaP discovers efficient symbolic expressions for preconditioning parameters. Part 1 demonstrates the acquisition of optimal parameters and dataset generation; Part 2 illustrates the training process of the RL-based deep symbolic discovery framework; Part 3 shows how the sequential model generates symbolic policies; Part 4 presents the deployment of symbolic expressions.

to understand the influence of parameters within existing theoretical frameworks and identify potential numerical stability issues. This interpretability fosters trust in the algorithm's predictions and supports further theoretical exploration (C3).

## 4. Method

This study focuses on enhancing the performance of parameterized preconditioners in solving linear systems derived from parameterized PDEs. Specifically, we investigate preconditioners with continuous parameters, such as the relaxation factor in SOR, while excluding those with discrete parameters like the level of fill-in ICC or ILU factorization.

We introduce a novel framework, SymMaP, for symbolic discovery in matrix preconditioning. As shown in Figure [3,](#page-3-0) we first obtain the optimal preconditioning parameters through a grid search to construct a training dataset. Then we employ an RNN to generate symbolic expressions in prefix notation, which are then evaluated for their fitness. The RNN is trained using a reward function based on the performance of the generated expressions. By optimizing the RNN parameters to maximize this reward function, we generate symbolic expressions that approximate the relationship between the problem's feature parameters (PDE parameters) and the optimal preconditioning parameters. Finally, we deploy the learned symbolic expressions into linear system solvers. The detailed steps are as follows and pseudocode is

provided in the Appendix [C.](#page-13-0)

## 4.1. Input Features and Training Data Generation

Input Features. In the context of solving parameterized PDEs, which frequently arise in linear systems, we consider feature parameters that characterize the equations. For instance, a second-order elliptic PDE can be expressed as: a11uxx + a12uxy + a22uyy + a1u<sup>x</sup> + a2u<sup>y</sup> + a0u = f, where the coefficients a11, a12, a22, . . . represent the feature parameters of PDE (see Appendix [D.1](#page-14-0) for details). These feature parameters, denoted as x<sup>i</sup> , serve as input features for the symbolic discovery process in SymMaP.

Training Data Generation. For each linear system, we determine optimal preconditioning parameters through an adaptive grid search. Using SOR preconditioning as an example, we optimize the relaxation factor ω within [0, 2] to minimize computation time (or condition number). The search process involves: 1. Conducting an initial coarse grid search (step size: 0.01) to evaluate computation time (or condition number) for each ω. 2. Identifying candidate regions with optimal performance. 3. Performing a refined grid search (step size: 0.001) within these regions.

This process yields the required training dataset, where each data point contains: 1. the problem feature parameters x<sup>i</sup> ; 2. the optimal preconditioning parameters y<sup>i</sup> . i = 1, 2, . . . , n, and n is the number of data, typically set to n = 1200, with 1000 allocated for the training set and 200 for the test set.

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

#### 4.2. The Generation of Symbolic Expressions

Token Library. For SymMaP, we define the library L of mathematical operators and operands as {+, −, ×, ÷,sqrt, exp, log, pow, 1.0}. Although other operators such as poly, sin and cos are frequently used [\(Udrescu & Tegmark,](#page-9-7) [2020\)](#page-9-7), we decided to exclude them because they offer limited explanatory power in matrix preconditioning and significantly increase the time and memory consumption during training.

After converting the mathematical expressions into prefix notation, we use this tokenized representation as a preorder traversal of the expression tree [\(Zaremba & Sutskever,](#page-9-8) [2014\)](#page-9-8). In each iteration, the RNN receives a pair consisting of a parent node and a sibling node as inputs. Then the RNN outputs a categorical distribution over all possible next tokens. The parent node refers to the last incomplete operator that requires additional operands to form a complete expression. The sibling node, in the context of a binary operator, represents the operand that has already been processed and incorporated into the expression. In cases where no parent or sibling node is applicable, they are designated as empty nodes. This structured input method enables the RNN to maintain contextual awareness and effectively predict the sequence of tokens that form valid mathematical expressions.

The Sequential Model. During the generation of a single symbolic expression, the RNN emits a categorical distribution for each "next token" at each step. This distribution is represented as a vector ψ (i) θ , where i denotes the i-th step and θ represents the parameters of the RNN. The elements of the vector correspond to the probabilities of each token, conditioned on the previously selected tokens in the traversal [\(Petersen et al.,](#page-9-9) [2019\)](#page-9-9):

$$\psi_{\boldsymbol{\theta}}^{(i)}(\boldsymbol{\tau}_i) = p(\boldsymbol{\tau}_i | \boldsymbol{\tau}_{1:i-1}; \boldsymbol{\theta}). \quad (3)$$

Here, τ<sup>i</sup> denotes the index of the token selected at the ith step. The probability of generating the entire symbolic expression τ is then the product of the conditional probabilities of all tokens [\(Petersen et al.,](#page-9-9) [2019;](#page-9-9) [Landajuela et al.,](#page-8-3) [2021\)](#page-8-3):

$$p(\tau|\theta) = \prod_{i=1}^N \psi_{\theta}^{(i)}(\tau_i). \quad (4)$$

Optimization of Constants: The library L incorporates a 'constant token,' which allows for the inclusion of various constant placeholders within sampled expressions. These placeholders serve as the parameters ξ in the symbolic expression. We seek to find the optimal values

of these parameters by maximizing the reward function: ξ <sup>∗</sup> = arg max<sup>ξ</sup> <sup>R</sup>(<sup>τ</sup> ; ξ), utilizing a nonlinear optimization method. This optimization is executed within each sampled expression as an integral part of computing the reward, prior to each training iteration.

#### 4.3. The Reward Function

Once a symbolic expression is fully generated (i.e., the symbolic tree reaches all its leaf nodes), we evaluate its fitness by calculating the normalized root-mean-square error (NRMSE), a metric frequently used in genetic programming symbolic discovery [\(Schmidt & Lipson,](#page-9-10) [2009\)](#page-9-10). It is defined as

$$\text{NRMSE} = \frac{1}{\sigma_y} \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}, \quad (5)$$

where yˆ<sup>i</sup> = τ (xi) is the predicted value for the i-th sample, x<sup>i</sup> is the problem feature parameter, y<sup>i</sup> is the optimal preconditioning parameter, σ<sup>y</sup> is the standard deviation of the target values y, and n is the number of data. To bound this fitness value between 0 and 1, we apply a squashing function:

$$R(\boldsymbol{\tau}) = \frac{1}{1 + \text{NRMSE}}. \quad (6)$$

Our objective is to maximize R(τ ), thereby minimizing the NRMSE and improving the accuracy of the generated expressions.

### 4.4. The Training Algorithm

Although the objective function is well-defined, it is important to note that R(τ ) is not a deterministic value but a random variable dependent on the RNN's parameters θ. Therefore, the key challenge is to establish an appropriate criterion for evaluating this random variable, and then apply gradient-based optimization methods accordingly.

Risk-seeking Policy. It is natural to consider the expectation of the reward function, i.e., <sup>E</sup>τ∼p(τ;θ) [R(τ )], as the objective function to optimize. We can apply the 'log-integral' trick [\(Williams,](#page-9-11) [1992\)](#page-9-11) and obtain

$$\nabla_{\boldsymbol{\theta}} \mathbb{E}_{\boldsymbol{\tau} \sim \mathbf{p}(\boldsymbol{\tau}; \boldsymbol{\theta})} [R(\boldsymbol{\tau})] = \mathbb{E}_{\boldsymbol{\tau} \sim \mathbf{p}(\boldsymbol{\tau}; \boldsymbol{\theta})} [R(\boldsymbol{\tau}) \nabla_{\boldsymbol{\theta}} \log \mathbf{p}(\boldsymbol{\tau}; \boldsymbol{\theta})]. \quad (7)$$

Thus, even though the expectation of the reward function is not directly differentiable with respect to θ, we can approximate the gradient using the sample mean.

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

Table 1: Comparison of average computation times (seconds) for SOR with different ω selections, and tolerance is 1e-7. SymMaP 1 and 2 are the two learned expressions that achieved the highest reward function scores, with the best-performing method highlighted in bold.

| Dataset      | Matrix size | No precondition | PETSc default ω = 1 | Fixed constant ω = 0 2 | Fixed constant ω = 1 8 | Optimal constant | SymMaP 1      | SymMaP 2      |
|--------------|-------------|-----------------|---------------------|------------------------|------------------------|------------------|---------------|---------------|
| Biharmonic   | 4 2 × 10 3  | 7.67            | 2.04                | 4.86                   | 1.60                   | 1.31             | 1.24          | 1.26          |
| Darcy Flow   | 1 0 × 10 4  | 33.1            | 13.5                | 17.5                   | 9.91                   | 9.54             | 8.50          | 8.60          |
| Elliptic PDE | 4 0 × 10 4  | 31.3            | 21.0                | 21.4                   | 17.5                   | 16.6             | 15.8          | 16.3          |
| Poisson      | 2 3 × 10 3  | 4.12 × 10 − 2   | 1.95 × 10 − 2       | 2.15 × 10 − 2          | 1.95 × 10 − 2          | 1.38 × 10 − 2    | 1.35 × 10 − 2 | 1.36 × 10 − 2 |
| Thermal      | 2 8 × 10 3  | 2.23 × 10 − 1   | 5.98 × 10 − 2       | 2.07 × 10 − 1          | 1.18 × 10 − 1          | 5.94 × 10 − 2    | 5.76 × 10 − 2 | 5.91 × 10 − 2 |

In the context of symbolic regression, model performance is often driven by a few exceptional results that outperform others by a significant margin [\(Petersen et al.,](#page-9-9) [2019;](#page-9-9) [Tamar](#page-9-12) [et al.,](#page-9-12) [2015\)](#page-9-12). With this in mind, we adopt a risk-seeking policy, which aims to maximize:

$$J(\boldsymbol{\theta}, \varepsilon) = \mathbb{E}_{\boldsymbol{\tau} \sim \mathbf{p}(\boldsymbol{\tau}, \boldsymbol{\theta})}[R(\boldsymbol{\tau}) | R(\boldsymbol{\tau}) > Q(\boldsymbol{\theta}, \varepsilon)]. \quad (8)$$

Here, ε is the risk factor, typically ε = 0.05, Q(θ, ε) is the (1 − ε)-quantile of the reward distribution under parameter θ, i.e.

$$Q(\boldsymbol{\theta}, \varepsilon) = \inf\{q \in \mathbb{R} | \text{CDF}(R(\boldsymbol{\tau}); \boldsymbol{\theta}) \geq 1 - \varepsilon\}, \quad (9)$$

where CDF(R(τ ); θ) refers to the cumulative distribution function. From this, the gradient of J(θ, ε) can be derived as [\(Petersen et al.,](#page-9-9) [2019\)](#page-9-9):

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}, \varepsilon) = \mathbb{E}_{\boldsymbol{\tau} \sim \mathbf{p}(\boldsymbol{\tau}; \boldsymbol{\theta})} \left[ \left( R(\boldsymbol{\tau}) - Q(\boldsymbol{\theta}, \varepsilon) \right) \cdot \nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{\tau}; \boldsymbol{\theta}) \middle| R(\boldsymbol{\tau}) > Q(\boldsymbol{\theta}, \varepsilon) \right]. \quad (10)$$

This gradient can be estimated using Monte Carlo sampling:

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}, \varepsilon) \approx \hat{g} \triangleq (11)$$

$$\frac{1}{\varepsilon N} \sum_{i=1}^N (R(\boldsymbol{\tau}^{(i)}) - \tilde{Q}(\boldsymbol{\theta}, \varepsilon)) \nabla_{\boldsymbol{\theta}} \log \mathbf{p} \cdot \mathbb{1}_{R(\boldsymbol{\tau}^{(i)}) > \tilde{Q}(\boldsymbol{\theta}, \varepsilon)},$$

Q˜(θ, ε) is the empirical (1 − ε)-quantile of the reward function. By concentrating on the top ε percentile of samples, SymMaP emphasizes optimizing the best-performing solutions in preconditioning, thereby obtaining the optimal symbolic expressions for preconditioning parameters.

## 4.5. Deployment in Linear System Solver

learned formula is exceptionally concise and incurs minimal computational cost. Therefore, we directly compile the learned policy into a lightweight shared object using a simple script and then integrate it into the linear system solver package (e.g., PETSc).

# 5. Experiments

We conducted comprehensive experiments to evaluate the SymMaP framework, organized into three primary sections: 1. Assessment of three different preconditioners and optimization goals across various datasets to determine the effectiveness of SymMaP algorithm, 2. Analysis of associated computational cost and the interpretability of the learned symbolic expressions, 3. Ablation studies of SymMaP.

Preconditioners: We considered three different preconditioners and various optimization metrics: 1. SOR preconditioner with the relaxation factor ω [\(Golub & Van Loan,](#page-8-2) [2013\)](#page-8-2); 2. SSOR preconditioner with the relaxation factor ω [\(Golub & Van Loan,](#page-8-2) [2013\)](#page-8-2); 3. AMG preconditioner with the threshold parameters θ<sup>T</sup> [\(Trottenberg et al.,](#page-9-4) [2000\)](#page-9-4).

Datasets: We investigated linear systems derived from five distinct PDE classes: 1. Darcy Flow Problems [\(Li et al.,](#page-9-13) [2020\)](#page-9-13), 2. Second-order Elliptic PDEs [\(Evans,](#page-8-5) [2022\)](#page-8-5), 3. Biharmonic Equations [\(Barrata et al.,](#page-8-12) [2023\)](#page-8-12), 4. Thermal Problems [\(Wang et al.,](#page-9-14) [2024\)](#page-9-14). 5. Poisson Equations [\(Wang](#page-9-14) [et al.,](#page-9-14) [2024\)](#page-9-14). All cases except biharmonic equations yield symmetric matrices. Notably, the non-symmetric matrices from biharmonic equations are incompatible with SSOR and AMG preconditioning techniques.

Baselines: We compared SymMaP against various parameter selection methods for preconditioning. Specifically, the comparison involved the following scenarios: 1. No matrix preconditioning, 2. Default parameters in PETSc [\(Balay](#page-8-6) [et al.,](#page-8-6) [2024\)](#page-8-6), 3. Fixed constants, 4. Optimized fixed constants.

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

Table 2: Comparison of average computation times (seconds) for SSOR with different ω selections, and tolerance is 1e-7. SymMaP 1 and 2 are the two learned expressions that achieved the highest reward function scores, with the best-performing method highlighted in bold.

| Dataset      | Matrix size | No precondition | PETSc default ω = 1 | Fixed constant ω = 0 2 | Fixed constant ω = 1 8 | Optimal constant | SymMaP 1      | SymMaP 2      |
|--------------|-------------|-----------------|---------------------|------------------------|------------------------|------------------|---------------|---------------|
| Darcy Flow   | 4 9 × 10 3  | 4.18            | 0.488               | 0.757                  | 1.09                   | 0.448            | 0.412         | 0.523         |
| Elliptic PDE | 4 0 × 10 4  | 23.9            | 10.5                | 14.7                   | 8.72                   | 8.68             | 7.70          | 7.74          |
| Poisson      | 2 3 × 10 3  | 2.12 × 10 − 2   | 1.02 × 10 − 2       | 1.93 × 10 − 2          | 9.91 × 10 − 3          | 9.89 × 10 − 3    | 9.10 × 10 − 3 | 9.92 × 10 − 3 |
| Thermal      | 2 8 × 10 3  | 2.34 × 10 − 1   | 2.69 × 10 − 2       | 5.08 × 10 − 2          | 9.87 × 10 − 2          | 2.24 × 10 − 2    | 2.13 × 10 − 2 | 2.14 × 10 − 2 |

Table 3: Comparison of average condition numbers for preconditioned matrices using different threshold parameter θ<sup>T</sup> selections in AMG. SymMaP 1 and 2 are the two learned expressions that achieved the highest reward function scores, with the best-performing method highlighted in bold.

| Dataset      | Matrix size | No precondition | PETSc default θ T = 0 | Fixed constant θ T = 0 2 | Fixed constant θ T = 0 8 | Optimal constant | SymMaP 1 | SymMaP 2 |
|--------------|-------------|-----------------|-----------------------|--------------------------|--------------------------|------------------|----------|----------|
| Darcy Flow   | 1 0 × 10 4  | 752862          | 8204                  | 19146                    | 11426                    | 7184             | 4824     | 5786     |
| Elliptic PDE | 4 0 × 10 4  | 6792            | 184.6                 | 205.4                    | 212.5                    | 182.8            | 168.8    | 170.3    |
| Poisson      | 1 0 × 10 4  | 1242            | 4.55                  | 68.85                    | 68.85                    | 4.55             | 3.72     | 3.72     |
| Thermal      | 2 8 × 10 3  | 7325            | 11.9                  | 627.2                    | 627.2                    | 9.91             | 9.71     | 9.71     |

were uniformly implemented using the C-based PETSc library [\(Balay et al.,](#page-8-6) [2024\)](#page-8-6) to maintain evaluation consistency. The experiments were conducted within PETSc's GMRES linear solver framework [\(Golub & Van Loan,](#page-8-2) [2013\)](#page-8-2), with condition numbers computed through the built-in function KSPComputeExtremeSingularValues.

Details on preconditioners, the mathematical forms of datasets, and the runtime environment are available in Appendices [B,](#page-10-0) [D.1,](#page-14-0) and [D.2,](#page-15-0) respectively. Information on the generation of training datasets for the following experiments and parameters of the SymMaP algorithm are outlined in Appendices [D.3](#page-15-1) and [D.4.](#page-16-0) The generated dataset and training time are available in Appendix [D.5.](#page-16-1) For an introduction to related work, see Appendix [A.](#page-10-1)

### 5.1. Main Experiments

In these experiments, as shown in Tables [1,](#page-5-0) [2,](#page-6-0) [3,](#page-6-1) we optimized relaxation factors ω in both SOR and SSOR preconditioning, and threshold parameters θ<sup>T</sup> in AMG preconditioning. For SOR and SSOR, we identified ω values that minimize computation time, forming the training dataset for SymMaP to learn symbolic expressions that optimize computational times for solutions. Similarly, for AMG, we selected θ<sup>T</sup> values that minimize the condition number of preconditioned matrices. Partial symbolic expressions can be found in Appendix [E.1.](#page-17-0)

Experimental results indicate that SymMaP consistently outperforms all others across all experimental tasks. For SOR, Table [1](#page-5-0) shows that SymMaP reduces computation times by up to 40% compared to PETSc's default settings and by 10% against the optimal constants. In SSOR, Table [2](#page-6-0) shows that it cuts computation time and iteration counts by up to 27%, over PETSc's defaults, and achieves reductions of 11% in time compared to optimal constants. For AMG, Table [3](#page-6-1) shows that SymMaP lowers the condition number by up to 40% relative to PETSc's defaults and 32% against the optimal constants.

These results highlight SymMaP's ability to effectively derive high-performance symbolic expressions for various preconditioning parameters, showcasing its broad applicability and strong generalization across different preconditioning tasks.

## 5.2. Comparison with Neural Network Performance

To evaluate the deployment overhead and prediction performance of SymMaP, we compared it with a basic multilayer perceptron (MLP) architecture. The MLP implementation consists of three fully connected layers, taking PDE parameters as input and generating preconditioning parameters as output. We employed ReLU activation functions and trained the model using mean squared error (MSE) between predicted and optimal parameters as the loss function. Both

394

396

Table 4: Comparison of the runtime required for symbolic expression and MLP to predict the SOR relaxation factor and the subsequent average solution time for linear systems, using the Darcy flow dataset with a matrix size of 10<sup>3</sup> and tolerance is 1e-5.

|        | Runtime (s) | Solution time (s) |
|--------|-------------|-------------------|
| MLP    | 5.1e-5      | 7.1e-1            |
| Symbol | 1.1e-5      | 7.1e-1            |

symbolic expression and MLP were executed in a CPU environment to simulate a modern solver environment.

Table 5: Partial symbolic expressions learned from some of the main experiments

| Precondition | Dataset      | Symbolic expression                           |
|--------------|--------------|-----------------------------------------------|
| SOR          | Biharmonic   | 1 0 + 1 0 / (4 0 + 1 0 /x 2 ) + 1 0 /x 1      |
| SOR          | Elliptic PDE | 1 0 + 1 0 / ( x 2 + 1 0 + 1 0 / ( x 2 + 4 0)) |
| SOR          | Darcy Flow   | 1 0 + 1 0 / ( x 4 + 1 0)                      |
| SSOR         | Elliptic PDE | 1 0 + 1 0 / ( x 2 + 1 2)                      |
| AMG          | Elliptic PDE | ( x 1 x 3 + 1) / 7                            |

As shown in Table [4,](#page-7-0) the runtime of symbolic expressions learned by SymMaP was only 20% of that of the MLP, primarily due to the poor performance of neural networks in a pure CPU environment, highlighting SymMaP's computational efficiency. Furthermore, the average solution times for parameters predicted by both symbolic expressions and MLP were closely matched. This demonstrates that symbolic expressions possess equivalent expressive capabilities to neural networks in this scenario, effectively approximating the optimal parameter expressions.

#### 5.3. Interpretable analysis

In Table [5,](#page-7-1) we report a subset of the learned symbolic expressions, with the mathematical significance of the related symbols detailed in Appendix [E.2.](#page-18-0) More symbolic expressions can be found in Appendix [E.1.](#page-17-0) These symbolic expressions are notably more concise and selective, not utilizing all candidate parameters and symbols, which aids researchers in analyzing their underlying relationships.

For instance, in the context of SOR and SSOR preconditioning, empirical evidence suggests that smaller relaxation factors should be chosen when diagonal components are relatively small. Our experimental findings corroborate this: for the second-order elliptical PDE dataset, the symbolic expressions derived for SOR and SSOR preconditioning depend solely on x2, with larger x<sup>2</sup> values leading to smaller predicted relaxation factors, exemplified by 1.0 + <sup>1</sup>.<sup>0</sup> (x2+1.2) . Here, x<sup>2</sup> represents the coupling coefficient of the elliptical

Table 6: Ablation study examining the selection of mathematical operators, comparing the effects on preconditioning and training times. The first column lists the selected operators, the second column shows the condition numbers of preconditioned matrices derived from AMG parameter predictions on the Darcy flow dataset (lower is better), and the third column displays SymMaP training times.

| Functionset                                              | Condition number | time(s) |
|----------------------------------------------------------|------------------|---------|
| + , − , × , ÷ , poly                                     | 6803.8           | 15351   |
| + , − , × , ÷ , sqrt , exp , log , pow , 1 0             | 7086.9           | 703.17  |
| + , − , × , ÷ , sqrt , exp , log , sin , cos , pow , 1 0 | 7172.6           | 635.82  |
| + , − , ÷ , 1 0 , pow                                    | 7241.8           | 703.26  |
| + , − , × , ÷ , sqrt , pow , 1 0                         | 7271.1           | 746.80  |
| + , − , × , ÷ , pow , 1 0                                | 7301.4           | 702.46  |

PDE, which directly influences the relative size of the nondiagonal components of the generated matrix, whereas other coefficients have minimal impact. As the coupling coefficient increases, the relative numerical of the non-diagonal components increases, and the diagonal components reduce correspondingly, aligning with empirical observations.

These experimental outcomes demonstrate that SymMaP can derive interpretable and efficient symbolic expressions for parameters, further aiding researchers in understanding and exploring the underlying mathematical principles.

#### 5.4. Ablation Experiments

We conducted an ablation study using SymMaP to evaluate the impact of different mathematical operator selections, as described in Table [6.](#page-7-2) In the main experiments, We utilized the operator set {+, −, ×, ÷,sqrt, exp, log, pow, 1.0} listed in the second row.

The results indicate that this selection of operators achieves a balance between predictive performance and training time efficiency, meeting our expectations. Furthermore, experiments detailing the performance of SymMaP about variations in learning rate, batch size, and dataset size are documented in Appendix [E.3.](#page-19-0)

## 6. Conclusions

In this paper, we propose SymMaP, a deep symbolic discovery framework designed for predicting efficient matrix preconditioning parameters. Experiments show that SymMaP can predict high-performance parameters and is applicable across a variety of preconditioning and optimization objectives. Additionally, SymMaP is easy to deploy with virtually no computational cost. We are confident in the symbolic model's immense potential for broad real-world applications, especially in matrix preconditioning.

- 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. References Agarwal, R. P. *Difference equations and inequalities: theory, methods, and applications*. CRC Press, 2000. Anderson, E., Bai, Z., Bischof, C., Blackford, S., Demmel, J., Dongarra, J., Du Croz, J., Greenbaum, A., Hammarling, S., McKenney, A., and Sorensen, D. *LAPACK Users' Guide*. Society for Industrial and Applied Mathematics, Philadelphia, PA, third edition, 1999. ISBN 0-89871-447- 8 (paperback). Balay, S., Abhyankar, S., Adams, M. F., Benson, S., Brown, J., Brune, P., Buschelman, K., Constantinescu, E. M., Dalcin, L., Dener, A., Eijkhout, V., Faibussowitsch, J., Gropp, W. D., Hapla, V., Isaac, T., Jolivet, P., Karpeev, D., Kaushik, D., Knepley, M. G., Kong, F., Kruger, S., May, D. A., McInnes, L. C., Mills, R. T., Mitchell, L., Munson, T., Roman, J. E., Rupp, K., Sanan, P., Sarich, J., Smith, B. F., Zampini, S., Zhang, H., Zhang, H., and Zhang, J. PETSc Web page. <https://petsc.org/>, 2024. URL <https://petsc.org/>. Barrata, I. A., Dean, J. P., Dokken, J. S., Habera, M., HALE, J., Richardson, C., Rognes, M. E., Scroggs, M. W., Sime, N., and Wells, G. N. Dolfinx: The next generation fenics problem solving environment. 2023. Bers, L., John, F., and Schechter, M. *Partial differential equations*. American Mathematical Soc., 1964. Chen, K. *Matrix preconditioning techniques and applications*. Number 19. Cambridge University Press, 2005. Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Pham, H., Dong, X., Luong, T., Hsieh, C.-J., Lu, Y., et al. Symbolic discovery of optimization algorithms. *Advances in neural information processing systems*, 36, 2024. Ciarlet, P. G. and Raviart, P.-A. A mixed finite element method for the biharmonic equation. In *Mathematical aspects of finite elements in partial differential equations*, pp. 125–145. Elsevier, 1974. Cranmer, M., Sanchez Gonzalez, A., Battaglia, P., Xu, R., Cranmer, K., Spergel, D., and Ho, S. Discovering symbolic models from deep learning with inductive biases. *Advances in neural information processing systems*, 33: 17429–17442, 2020. Demmel, J. W. *Applied numerical linear algebra*. SIAM, 1997. Driscoll, T. A., Hale, N., and Trefethen, L. N. Chebfun guide, 2014. Evans, L. C. *Partial differential equations*, volume 19. American Mathematical Society, 2022. Glowinski, R. and Pironneau, O. Numerical methods for the first biharmonic equation and for the two-dimensional stokes problem. *SIAM review*, 21(2):167–212, 1979. Golub, G. H. and Van Loan, C. F. *Matrix computations*. JHU press, 2013. Gotz, M. and Anzt, H. Machine learning-aided numerical ¨ linear algebra: Convolutional neural networks for the efficient preconditioner generation. In *2018 IEEE/ACM 9th Workshop on Latest Advances in Scalable Algorithms for Large-Scale Systems (scalA)*, pp. 49–56. IEEE, 2018. Greenbaum, A. *Iterative methods for solving linear systems*. SIAM, 1997. Greenfeld, D., Galun, M., Basri, R., Yavneh, I., and Kimmel,
  - R. Learning to optimize multigrid pde solvers. In *International Conference on Machine Learning*, pp. 2415–2423. PMLR, 2019. Hsieh, J.-T., Zhao, S., Eismann, S., Mirabella, L., and Ermon, S. Learning neural pde solvers with convergence guarantees. *arXiv preprint arXiv:1906.01200*, 2019. Johnson, C. *Numerical solution of partial differential equations by the finite element method*. Courier Corporation, 2009. Koric, S. and Abueidda, D. W. Data-driven and physicsinformed deep learning operators for solution of heat conduction equation with parametric heat source. *International Journal of Heat and Mass Transfer*, 203:123809, 2023. Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., and Anandkumar, A. Neural operator: Learning maps between function spaces. *arXiv preprint arXiv:2108.08481*, 2021. Lample, G. and Charton, F. Deep learning for symbolic mathematics. *arXiv preprint arXiv:1912.01412*, 2019. Landajuela, M., Petersen, B. K., Kim, S., Santiago, C. P., Glatt, R., Mundhenk, N., Pettit, J. F., and Faissol, D. Discovering symbolic policies with deep reinforcement learning. In *International Conference on Machine Learning*, pp. 5979–5989. PMLR, 2021.

## Impact Statement

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

Leon, S. J., De Pillis, L. G., and De Pillis, L. G. *Linear algebra with applications*. Pearson Prentice Hall Upper Saddle River, NJ, 2006. LeVeque, R. J. *Finite difference methods for ordinary and partial differential equations: steady-state and timedependent problems*. SIAM, 2007. Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar, A. Fourier neural operator for parametric partial differential equations. *arXiv preprint arXiv:2010.08895*, 2020. Lu, L., Meng, X., Cai, S., Mao, Z., Goswami, S., Zhang, Z., and Karniadakis, G. E. A comprehensive and fair comparison of two neural operators (with practical extensions) based on fair data. *Computer Methods in Applied Mechanics and Engineering*, 393:114778, 2022. Luz, I., Galun, M., Maron, H., Basri, R., and Yavneh, I. Learning algebraic multigrid using graph neural networks. In *International Conference on Machine Learning*, pp. 6489–6499. PMLR, 2020. Mankowitz, D. J., Michi, A., Zhernov, A., Gelmi, M., Selvi, M., Paduraru, C., Leurent, E., Iqbal, S., Lespiau, J.-B., Ahern, A., et al. Faster sorting algorithms discovered using deep reinforcement learning. *Nature*, 618(7964): 257–263, 2023. Petersen, B. K., Landajuela, M., Mundhenk, T. N., Santiago, C. P., Kim, S. K., and Kim, J. T. Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients. *arXiv preprint arXiv:1912.04871*, 2019. Poli, R., Langdon, W., and McPhee, N. A field guide to genetic programming (with contributions by jr koza)(2008). *Published via http://lulu. com*, 2008. Rahman, M. A., Ross, Z. E., and Azizzadenesheli, K. U-no: U-shaped neural operators. *arXiv preprint arXiv:2204.11127*, 2022. Rudin, C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature machine intelligence*, 1(5):206– 215, 2019. Ruge, J. W. and Stuben, K. Algebraic multigrid. In ¨ *Multigrid methods*, pp. 73–130. SIAM, 1987. Saad, Y. *Iterative methods for sparse linear systems*. SIAM, 2003. Saad, Y. and Schultz, M. H. Gmres: A generalized minimal residual algorithm for solving nonsymmetric linear systems. *SIAM Journal on scientific and statistical computing*, 7(3):856–869, 1986. Schmidt, M. and Lipson, H. Distilling free-form natural laws from experimental data. *science*, 324(5923):81–85, 2009. Sharma, R., Farimani, A. B., Gomes, J., Eastman, P., and Pande, V. Weakly-supervised deep learning of heat transport via physics informed loss. *arXiv preprint arXiv:1807.11374*, 2018. Stanaityte, R. *ILU and Machine Learning Based Preconditioning for the Discretized Incompressible Navier-Stokes Equations*. PhD thesis, University of Houston, 2020. Taghibakhshi, A., MacLachlan, S., Olson, L., and West, M. Optimization-based algebraic multigrid coarsening using reinforcement learning. *Advances in neural information processing systems*, 34:12129–12140, 2021. Tamar, A., Glassner, Y., and Mannor, S. Policy gradients beyond expectations: Conditional value-at-risk. Citeseer, 2015. Trefethen, L. N. and Bau, D. *Numerical linear algebra*. SIAM, 2022. Trottenberg, U., Oosterlee, C. W., and Schuller, A. *Multigrid*. Elsevier, 2000. Udrescu, S.-M. and Tegmark, M. Ai feynman: A physicsinspired method for symbolic regression. *Science Advances*, 6(16):eaay2631, 2020. Wang, H., Hao, Z., Wang, J., Geng, Z., Wang, Z., Li, B., and Wu, F. Accelerating data generation for neural operators via krylov subspace recycling. *arXiv preprint arXiv:2401.09516*, 2024. Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine learning*, 8:229–256, 1992. Young, D. Iterative methods for solving partial difference equations of elliptic type. *Transactions of the American Mathematical Society*, 76(1):92–111, 1954. Zaremba, W. and Sutskever, I. Learning to execute. *arXiv preprint arXiv:1410.4615*, 2014. Zhang, E., Kahana, A., Turkel, E., Ranade, R., Pathak, J., and Karniadakis, G. E. A hybrid iterative numerical transferable solver (hints) for pdes based on deep operator network and relaxation methods. *arXiv preprint arXiv:2208.13273*, 2022.

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

#### A. Related work

#### A.1. Machine Learning for Algorithm Discovery

Machine learning has the potential to uncover implicit rules beyond human intuition from training data, enabling the construction of algorithms that outperform handcrafted programs. Approaches to algorithm discovery in machine learning encompass symbolic discovery, program search, and more. Specifically, program search focuses on optimizing the computational processes of algorithms. For example, [Mankowitz et al.](#page-9-15) [\(2023\)](#page-9-15) explores the discovery of faster sorting algorithms, while [Chen et al.](#page-8-13) [\(2024\)](#page-8-13) investigates efficient optimization algorithms.

In contrast, symbolic discovery aims to search within the space of small mathematical expressions rather than computational streams [\(Petersen et al.,](#page-9-9) [2019;](#page-9-9) [Landajuela et al.,](#page-8-3) [2021\)](#page-8-3). This approach is analogous to an extreme form of model distillation, where knowledge extracted from black-box neural networks is distilled into explicit mathematical expressions. Traditional methods for symbolic discovery have relied on evolutionary algorithms, including genetic programming [\(Poli et al.,](#page-9-16) [2008\)](#page-9-16). Recently, deep learning has emerged as a powerful tool in this domain, offering enhanced representational capacity and new avenues for solving symbolic discovery problems [\(Schmidt & Lipson,](#page-9-10) [2009;](#page-9-10) [Cranmer et al.,](#page-8-11) [2020\)](#page-8-11).

## A.2. Neural Networks for Matrix Preconditioning

Recent studies have explored the use of neural networks to improve matrix preconditioning techniques. [\(Greenfeld et al.,](#page-8-14) [2019;](#page-8-14) [Luz et al.,](#page-9-17) [2020;](#page-9-17) [Taghibakhshi et al.,](#page-9-18) [2021\)](#page-9-18) demonstrate the effectiveness of neural networks in refining multigrid preconditioning algorithms, thus streamlining the computational process. [\(Gotz & Anzt](#page-8-15) ¨ , [2018\)](#page-8-15) utilized Convolutional Neural Networks (CNNs) for the optimization of block Jacobi preconditioning algorithms, while [\(Stanaityte,](#page-9-19) [2020\)](#page-9-19) developed corresponding Incomplete Lower-Upper Decomposition (ILU) preconditioning algorithms leveraging machine learning insights. Although these algorithms achieved impressive results, they still face challenges such as limited interpretability and reduced computational efficiency when deployed in pure CPU environments. This paper attempts to address these issues by incorporating symbolic discovery into the framework.

## B. Detailed introduction of matrix preconditioning

## B.1. Overview of Matrix Preconditioning Methods

- Jacobi Method: The Jacobi preconditioner utilizes only the diagonal elements of a matrix to precondition a linear system. By approximating the inverse of the diagonal matrix, this method is computationally simple and effective for systems with strong diagonal dominance. However, its convergence rate can be slow, and its performance diminishes for poorly conditioned or weakly diagonally dominant matrices. The Jacobi method is typically used as a baseline for comparison with more sophisticated preconditioners [\(Saad,](#page-9-20) [2003\)](#page-9-20).
- Gauss-Seidel (GS) Method: The Gauss-Seidel preconditioner improves upon the Jacobi method by considering both the lower triangular and diagonal parts of the matrix in a sequential manner. Unlike the Jacobi method, which updates all variables simultaneously, the GS method updates each variable in sequence using the most recent values. This leads to faster convergence, especially for diagonally dominant matrices. However, the GS method can still struggle with poorly conditioned systems, and its forward-only approach can limit performance in some applications [\(Saad,](#page-9-20) [2003\)](#page-9-20).
- Successive Over-Relaxation (SOR): The SOR method builds on the Gauss-Seidel method by introducing a relaxation factor ω to accelerate convergence. This factor allows for over-relaxation (ω > 1) or under-relaxation (ω < 1), tuning the method for faster performance on certain types of problems. SOR can significantly reduce the number of iterations needed for convergence compared to both the Jacobi and GS methods, but choosing the optimal relaxation factor is problem-dependent [\(Young,](#page-9-3) [1954\)](#page-9-3).
- Symmetric Successive Over-Relaxation (SSOR): SSOR is a symmetric version of the SOR method, where relaxation is applied in both forward and backward sweeps of the matrix. This bidirectional process improves stability and is well-suited for use with iterative solvers like the conjugate gradient method, which requires symmetric preconditioners. SSOR's symmetry ensures that the preconditioner maintains the properties needed for efficient and stable convergence, making it a popular choice for symmetric positive-definite systems [\(Golub & Van Loan,](#page-8-2) [2013\)](#page-8-2).
- Algebraic Multigrid (AMG): AMG is an advanced preconditioning technique designed to handle large, sparse systems of linear equations, especially those arising from the discretization of partial differential equations. Unlike traditional

methods, AMG operates on multiple levels of the matrix structure, coarsening the matrix to form a hierarchy of smaller systems that are easier to solve. Solutions on the coarser grids are then interpolated back to the finer grids. This multilevel approach makes AMG highly efficient for large-scale problems, as it can dramatically reduce the number of iterations needed to achieve convergence. AMG is often used in combination with methods like SSOR or Gauss-Seidel as a smoother on each grid level, and it is particularly effective in cases where the problem exhibits a multiscale nature [\(Ruge & Stuben](#page-9-21) ¨ , [1987\)](#page-9-21).

Relationship Among Jacobi, GS, and SOR Methods: The Jacobi method is the simplest of the three, using only diagonal information. The GS method improves upon the Jacobi method by using both diagonal and lower triangular matrix elements to achieve faster convergence. SOR further refines the GS method by introducing a relaxation factor to optimize the update process. Both the GS and SOR methods can be seen as iterative improvements on the Jacobi method, with SOR offering a more flexible and potentially faster alternative by adjusting the relaxation factor. SSOR extends SOR symmetrically, making it suitable for use in more advanced iterative solvers like the conjugate gradient method [\(Saad,](#page-9-20) [2003;](#page-9-20) [Golub & Van Loan,](#page-8-2) [2013\)](#page-8-2).

#### B.2. Parameters in Matrix Preconditioning

The choice of preconditioning parameters significantly influences the effectiveness of the preconditioning process, especially in the iterative solving of linear systems [\(Chen,](#page-8-1) [2005\)](#page-8-1). Below, we discuss three specific preconditioning techniques—SOR, SSOR, and AMG—focusing particularly on how their key parameters affect the preconditioning results.

#### B.2.1. RELAXATION FACTOR ω IN SOR AND SSOR METHODS

In the SOR preconditioning method, the relaxation factor ω is a critical parameter that determines the acceleration of iteration. SOR evolves from the Gauss-Seidel method by introducing ω to speed up convergence. The SOR iteration formula is given by:

$$\mathbf{x}^{(k+1)} = (\mathbf{D} + \omega\mathbf{L})^{-1} \left[ (1 - \omega)\mathbf{D}\mathbf{x}^{(k)} + \omega\mathbf{b} - \omega\mathbf{U}\mathbf{x}^{(k)} \right], \quad (12)$$

where D, L, and U are the diagonal, strictly lower triangular, and strictly upper triangular parts of the matrix A, respectively [\(Golub & Van Loan,](#page-8-2) [2013\)](#page-8-2).

The SSOR preconditioning method can be represented by the following formula:

$$M_{\text{SSOR}} = \frac{1}{\omega(2-\omega)} (\mathbf{D} - \omega \mathbf{U}) \mathbf{D}^{-1} (\mathbf{D} - \omega \mathbf{L}), \quad (13)$$

where MSSOR constitutes the preconditioner, and D, L, U, and ω are defined similarly to their roles in the SOR method. This symmetrical formulation enhances the stability and effectiveness of the preconditioning, particularly benefiting symmetric positive-definite matrices by optimizing the convergence properties of the iterative solver [\(Golub & Van Loan,](#page-8-2) [2013\)](#page-8-2).

The choice of ω directly impacts the speed of convergence and the condition number of the matrix. Different problems and scenarios often require different choices of ω, which typically need to be determined based on the specific properties of the problem and through numerical experimentation [\(Golub & Van Loan,](#page-8-2) [2013\)](#page-8-2). In the PETSc library, the default relaxation factor ω for both SOR and SSOR is set to 1, at which point SOR degenerates to GS preconditioning.

## B.2.2. THRESHOLD PARAMETERS θ<sup>T</sup> IN AMG

In the AMG method, the threshold parameter θ<sup>T</sup> determines whether the non-zero elements of the matrix are "strong" enough to be considered in the construction of a coarse grid during the multigrid process. This parameter is crucial for establishing the connectivity between coarse and fine grids in the hierarchical multilevel structure [\(Ruge & Stuben](#page-9-21) ¨ , [1987\)](#page-9-21).

The AMG method solves the equation system through multiple levels of grids, each corresponding to a coarser version of the original problem. During this process, the threshold parameter is used to determine whether a given non-zero matrix element is strong enough to keep the corresponding grid points connected during coarsening.

- A lower threshold often leads to more elements being considered as strong connections, which might increase the complexity of the coarse grid but can help preserve the essential features of the original problem, thus improving the efficiency and convergence of the multigrid method.
- A higher threshold might result in fewer strong connections, thereby reducing the complexity of the coarse grid. However, this can weaken the effectiveness of the AMG method, especially in maintaining the features of the original problem.

Different values of θ<sup>T</sup> directly influence the condition number of the preconditioned matrix. Selecting the appropriate threshold parameter typically involves considering the specific structure and features of the problem, and adjustments are made through experimental fine-tuning to achieve the optimal balance [\(Trottenberg et al.,](#page-9-4) [2000\)](#page-9-4). In the PETSc library, the default threshold parameter θ<sup>T</sup> is set to 0.

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

## C. Algorithm Pseudocode

Algorithm 1 RNN-based Symbolic Discovery Process

Input: RNN with parameter θ, the library of tokens L τ ← [ ] parent(0), sibling(0) ← empty node x<sup>0</sup> ← parent(0)||sibling(0) // x is the concatenation of parent and sibling nodes h<sup>0</sup> ← 0 // Initialize hidden state of RNN for t = 1, 2, · · · do (ψt, ht) ← RNN(xt−1, ht−1; θ) // ψ<sup>t</sup> is the categorical distribution of the next token ψ<sup>t</sup> ← ApplyConstraint(ψt,L, τ ) // Regularize the distribution Sample token τ<sup>t</sup> ∼ ψ<sup>t</sup> if *Arity*(τt) > 0 then // Arity(τi) denotes the number of operands of τ<sup>i</sup> parent(t) ← τ<sup>t</sup> sibling(t) ← empty node else // When Arity(τt) = 0, go back to the last incomplete operator node count ← 0 for i = t, t − 1, . . . , 1 do // Backward iteration count ← count + Arity(τi) −1 if *count = 0* then parent(t) ← τ<sup>i</sup> sibling(t) ← τi+1 break if *count* = −1 then // The expression sequence is complete break x<sup>t</sup> ← parent(t)||sibling(t) Output: Prefix expression sequence τ

Algorithm 2 Deep Symbolic Optimization for Matrix Preconditioning Parameter

Input :RNN with initial parameter θ0, the library of tokens L, batch size N, iteration number J, risk factor ε, and learning rate α θ ← θ<sup>0</sup> j ← 0 repeat for i = 1, 2, . . . , N do τ (i) ← SymbolicDiscover(θ,L) ξ <sup>∗</sup> ← {ξ in τ as constant placeholder : R(τ ; ξ)} // Constant optimization τ (i) ← ReplaceConstant(τ (i) , ξ<sup>∗</sup> ) Compute gˆ<sup>1</sup> using τ (i) and θ // See Eq. [\(11\)](#page-5-1) Compute gˆ<sup>2</sup> as entropy gradient θ ← θ + α(ˆg<sup>1</sup> + ˆg2) // Update the parameter Train model: update p<sup>θ</sup> via PPO by optimizing J(θ; ϵ)

until j = J *or convergence*

Output :The best symbolic expression τ ∗

774

776

778

794

796

800

804

806

808

#### D. Experiment Settings

#### D.1. Datasets

#### 1. Darcy Flow Problem

We consider two-dimensional Darcy flows, which can be described by the following equation [\(Li et al.,](#page-9-13) [2020;](#page-9-13) [Rahman et al.,](#page-9-22) [2022;](#page-9-22) [Kovachki et al.,](#page-8-16) [2021;](#page-8-16) [Lu et al.,](#page-9-23) [2022\)](#page-9-23):

$$-\nabla \cdot (K(x, y) \nabla h(x, y)) = f,$$

where K is the permeability field, h is the pressure, and f is a source term which can be either a constant or a space-dependent function.

In our experiment, K(x, y) is generated using truncated Chebyshev polynomials. We convert the Darcy flow problem into a system of linear equations using the central difference scheme of Finite Difference Methods (FDM) [\(LeVeque,](#page-9-1) [2007\)](#page-9-1). The coefficients of the Chebyshev polynomials serve as input features for our symbolic discovery framework.

#### 2. Second-order Elliptic Partial Differential Equation

We consider general two-dimensional second-order elliptic partial differential equations, which are frequently described by the following generic form [\(Evans,](#page-8-5) [2022;](#page-8-5) [Bers et al.,](#page-8-17) [1964\)](#page-8-17):

$$\mathcal{L}u \equiv a_{11}u_{xx} + a_{12}u_{xy} + a_{22}u_{yy} + a_1u_x + a_2u_y + a_0u = f,$$

where a0, a1, a2, a11, a12, a<sup>22</sup> are constants, and f represents the source term, depending on x, y. The variables u, ux, u<sup>y</sup> are the dependent variable and its partial derivatives. The equation is classified as elliptic if 4a11a<sup>22</sup> > a<sup>2</sup> <sup>12</sup>.

In our experiments, a11, a22, a1, a2, a<sup>0</sup> are uniformly sampled within the range (−1, 1), while the coupling term a<sup>12</sup> is sampled within (−0.01, 0.01). We then select equations that satisfy the elliptic condition to form our dataset. Similar to the approach with the Darcy flow problem, we convert the PDE into a system of linear equations using the central difference scheme of FDM. The coefficients a0, a1, a2, a11, a12, a<sup>22</sup> serve as input features for our symbolic discovery framework. When discussing SSOR preconditioning, we set a<sup>1</sup> and a<sup>2</sup> to zero to ensure the resulting matrix remains symmetric.

#### 3. Biharmonic Equation

We consider the biharmonic equation, a fourth-order elliptic equation, defined on a domain Ω ⊂ R 2 . The equation is expressed as follows [\(Ciarlet & Raviart,](#page-8-18) [1974;](#page-8-18) [Glowinski & Pironneau,](#page-8-19) [1979;](#page-8-19) [Barrata et al.,](#page-8-12) [2023\)](#page-8-12):

$$\nabla^4 u = f \quad \text{in } \Omega = [0, a] \times [0, b],$$

where ∇<sup>4</sup> ≡ ∇<sup>2</sup>∇<sup>2</sup> represents the biharmonic operator and f = 4.0π 4 sin(πx) sin(πy) is the prescribed source term.

In our experiments, we construct the dataset by varying the solution domain Ω = [0, a] × [0, b]. We utilize the discontinuous Galerkin finite element method from the FEniCS library to transform this problem into a system of linear equations [\(Barrata](#page-8-12) [et al.,](#page-8-12) [2023\)](#page-8-12). The parameters a, b of the domain serve as input features for our symbolic discovery framework.

#### 4. Poisson Equation

We consider a two-dimensional Poisson equation, which can be described by the following equation [\(Wang et al.,](#page-9-14) [2024;](#page-9-14) [Hsieh et al.,](#page-8-20) [2019;](#page-8-20) [Zhang et al.,](#page-9-24) [2022\)](#page-9-24):

$$\nabla^2 u = f \quad \text{in } \Omega = [0, 1]^2.$$

Physical Contexts in which the Poisson Equation Appears: 1. Electrostatics; 2. Gravitation; 3. Fluid Dynamics.

In our experiments, we address the Poisson equation within a square domain , where both the boundary conditions on all four sides and the source term f on the equation's left-hand side are generated using third-order truncated Chebyshev polynomials. The finite difference method with a central difference scheme is employed to discretize the equation into a linear system. The Chebyshev coefficients serve as parameters for our symbolic discovery framework [\(Driscoll et al.,](#page-8-21) [2014\)](#page-8-21).

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

We consider a two-dimensional thermal steady state equation, which can be described by the following equation [\(Wang](#page-9-14) [et al.,](#page-9-14) [2024;](#page-9-14) [Sharma et al.,](#page-9-25) [2018;](#page-9-25) [Koric & Abueidda,](#page-8-22) [2023\)](#page-8-22):

$$\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0,$$

where T is the temperature. We examine the steady-state thermal equation in thermodynamics. As with the previous equation, we still solve this equation in the square domain. The boundary temperatures on the left and right boundaries are determined by random values ranging from -100 to 0 and 0 to 100, respectively. The top and bottom boundary temperature functions are generated by third-order truncated Chebyshev polynomials. The boundary temperature and the coefficients of the Chebyshev polynomials serve as parameters for our symbolic discovery framework.

#### D.2. Environment

To ensure consistency in our evaluations, all comparative experiments were conducted under uniform computing environments. Specifically, the environments used are detailed as follows:

- 1. Environment (Env1):
  - Platform: Windows11 version 22631.4169, WSL
  - Operating System: Ubuntu 22.04.3
  - CPU Processor: AMD Ryzen 9 5900HX with Radeon Graphics CPU, clocked at 3.30GHz
- 2. Environment (Env2):
  - Platform & Operating System: Ubuntu 18.04.4 LTS
  - CPU Processor: Intel(R) Xeon(R) Gold 6246R CPU at 3.40GHz
  - GPU Processor: GeForce RTX 3090 24GB
  - Library: CUDA Version 11.3

Speed tests for solving linear systems were performed in Env 1, while all training related to symbolic discovery was conducted in Env 2.

#### D.3. Training Data Generation

We employed an adaptive grid search to generate the training dataset. Initially, we traversed a coarse grid, sampling every 0.05, and from this dataset, we selected the three points with the smallest values. Subsequently, we conducted a finer grid search around these points, sampling every 0.001, to identify the point with the minimum value, which we designated as our optimal parameter. Particularly, after experimental validation confirmed the dataset's convexity, we utilized a binary search sampling method for a dataset derived from the second-order elliptic equation's SOR preconditioning. Starting with points at 0.0, 1.0, and 2.0, we compared these values. If the value at 0.0 was lowest, we computed at 0.5; if at 2.0, then at 1.5; and if at 1.0, then at both 0.5 and 1.5. This process was repeated until achieving a minimum point with a precision of 0.001.

For SOR preconditioning, we evaluated second-order elliptic equations, Darcy flow equations, and biharmonic equations, with solution time as the metric for optimal preprocessing parameters, achieved by minimizing solution time using the previously described grid method. In SSOR preconditioning, applied to second-order elliptic and Darcy flow equations, we utilized a hybrid metric that combined normalized computation time and iteration counts, aiming to simultaneously optimize both iteration counts and solution times. For AMG preconditioning, also examined with second-order elliptic and Darcy flow equations, we used the condition number of the preconditioned matrix as the metric, where a lower value indicates better performance.

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

### D.4. Parameters of SymMAP

Experimental Setup. SymMAP is implemented using the LSTM architecture with one layer and 32 units. More details about the hyperparameters are provided in Table [7.](#page-16-2)

Table 7: Hyperparameters of SymMAP (Default Model)

|         | Hyperparameter |          |                | Value     |
|---------|----------------|----------|----------------|-----------|
| Number  | of             | LSTM     | layers         | 1         |
| Number  |                | of LSTM  | units          | 32        |
| Number  | of             | training | samples        | 2,000,000 |
|         | Batch          | size     |                | 1,000     |
|         | Risk           | factor   | ε              | 0.05      |
| Minimal | expression     |          | length         | 4         |
| Maximal | expression     |          | length         | 64        |
|         | Learning       |          | rate           | 0.0005    |
| Weight  | of entropy     |          | regularization | 0.03      |

Restricting searching space. We employ specific constraints within our framework to streamline the exploration of expression spaces effectively and ensure they remain within practical and manageable bounds:

- 1. Bounds on expression length. To strike a balance between complexity and manageability, we set boundaries for expression lengths: a minimum of 4 and a maximum of 64 characters. This ensures that expressions are neither overly trivial nor excessively complicated.
- 2. Constant combination. We restrict expressions such that the operands of any binary operator are not both constants. This is out of the simple intuition that, if both operands are constants, the combination of the two can be precomputed and replaced with a single constant.
- 3. Inverse operator exclusion. We preclude unary operators from having their inverses as children to avoid redundant computations and meaningless expressions, such as in log(exp(x)).
- 4. Trigonometric Constraints. Expressions involving trigonometric operators should not include descendants within their formulation. For instance, sin(x + cos(x)) is restricted because it combines trigonometric operators in a way that is uncommon in scientific contexts.

## D.5. Computational Time for Related Algorithms

- Dataset Generation Time:
  - Darcy Flow Problem: 40 hours
  - Second-order Elliptic Partial Differential Equation: 40 hours
  - Biharmonic Equation: 100 hours
  - Poisson Equation: 6 hours
  - Thermal Problem: 6 hours
- SymMAP Execution Time: For each run, 1000 iterations are performed.
  - Without polynomials in the Token Library: approximately 800 seconds.
  - With polynomials in the Token Library: approximately 2600 seconds.

# E. Experimental Data and Supplementary Experiments

## E.1. Symbolic Expressions from Main Experiments

This section documents some of the learned expressions from the main experiments, corresponding to "SymMaP1" in Tables [1,](#page-5-0) [2,](#page-6-0) and [3.](#page-6-1)

- Second-order elliptic PDE problem, AMG preconditioning:

$$\frac{x_1 x_3 + 1.0}{x_1 + 7.0}$$

Parameter meanings: x1-x<sup>6</sup> represent the coefficients a11, a12, a22, a1, a2, and a<sup>0</sup> in the second-order elliptic equation.

- Second-order elliptic PDE problem, SSOR preconditioning:

$$\begin{aligned} & (2x_2 - \frac{1}{4x_1 + 2x_3})(-0.21785x_1^3 - 63.6118x_1^2x_2 + 0.206541x_1^2x_3 - 0.235667x_1^2x_4 \\ & \quad + 0.269472x_1^2 - 967.517x_1x_2^2 - 61.2291x_1x_2x_3 + 1.68205x_1x_2x_4 + 5.07925x_1x_2 \\ & \quad - 0.0221322x_1x_3^2 - 0.454257x_1x_3x_4 - 0.0693756x_1x_3 + 0.411528x_1x_4^2 + 0.0311608x_1x_4 \\ & \quad - 7.53439x_1 + 9506.4x_3^2 - 468.735x_3^2x_3 - 154.885x_3^2x_4 + 410.223x_3^2 - 25.5913x_3x_3^2 \\ & \quad + 7.92627x_2x_3x_4 + 5.30828x_2x_3 + 3.82512x_2x_4^2 - 7.0487x_2x_4 - 0.612507x_2 - 0.180432x_3^3 \\ & \quad - 0.0462734x_3^2x_4 + 0.310649x_3^2 + 0.257121x_3x_4^2 - 0.0962336x_3x_4 - 3.86012x_3 + 0.13906x_4^3 \\ & \quad - 0.389893x_4^2 + 0.3144x_4 - 0.0111835) \end{aligned}$$

Parameter meanings: x1-x<sup>4</sup> represent the coefficients a11, a12, a22, and a<sup>0</sup> in the second-order elliptic equation.

- Darcy flow problem, SSOR preconditioning:

$$\frac{1.0}{x_1(x_{14} + x_4) + 1.0}$$

Parameter meanings: x1-x<sup>16</sup> represent the 16 coefficients of a second-order truncated Chebyshev polynomial in two dimensions, ordered as follows: 1, x, x , x , y, xy, x y, x y, y , xy<sup>2</sup> , x y , x y , y , xy<sup>3</sup> , x y , and x y .

- Darcy flow problem, AMG preconditioning:

$$1.0 + \frac{1.0}{1.0 + \frac{1.0}{x_{16} + x_3 + x_9^2 + 2.0}}$$

Parameter meanings: x1-x<sup>16</sup> represent the 16 coefficients as described above.

- Biharmonic Equation, SOR preconditioning:

$$\left( \frac{1.0x_2}{-4x_2 - 1.0 + \frac{1.0}{x_2}} + 1.0 \right)^4$$

Parameter meanings: x<sup>1</sup> and x<sup>2</sup> represent the length and width of the equation's boundary, respectively.

- Biharmonic Equation, AMG preconditioning:

$$1.0 + \frac{1.0}{3.0 + \frac{1.0x_1+1.0}{x_2}} + \frac{1.0}{x_2}$$

994

996

998

1014

1016

1019

1024

1026

1029

1034

1036

- Poisson Equation, SOR preconditioning:

$$\sqrt{\exp\left(\frac{1.0}{x_3 + \exp(2 \exp(x_1^2))}\right)}$$

Parameter meanings: x1-x<sup>8</sup> represent the coefficients of two second-order truncated Chebyshev polynomials for the boundary functions.

- Poisson Equation, SSOR preconditioning:

$$1.15024107160485 (0.106506978919201x_2^2 + 1)^{1/16}$$

Parameter meanings: x1-x<sup>8</sup> represent the coefficients of two second-order truncated Chebyshev polynomials for the boundary functions.

- Poisson Equation, AMG preconditioning:

$$\frac{1.0}{x_8^2 + 7.0}$$

Parameter meanings: x1-x<sup>8</sup> represent the coefficients of two second-order truncated Chebyshev polynomials for the boundary functions.

- Thermal problem, SOR preconditioning:

$$\exp \left( \frac{0.778800783071405}{(1 - x_6)^{1/4}} \right)^{1/4}$$

Parameter meanings: x1-x<sup>4</sup> represent the coefficients of the Chebyshev polynomial for the boundary temperature function on the upper and lower boundaries, while x<sup>5</sup> and x<sup>6</sup> represent the coefficients for the boundary temperature function on the left and right boundaries.

- Thermal problem, SSOR preconditioning:

$$1.0 - \frac{1.0}{\log(4.0(1 - 0.5x_6)^2)}$$

Parameter meanings: x1-x<sup>4</sup> represent the coefficients of the Chebyshev polynomial for the boundary temperature function on the upper and lower boundaries, while x<sup>5</sup> and x<sup>6</sup> represent the coefficients for the boundary temperature function on the left and right boundaries.

- Thermal problem, AMG preconditioning:

1.0

---

2.71828182845905 
$$\exp(0.1353352583236613x_6) + 8.154845548537714$$

Parameter meanings: x1-x<sup>4</sup> represent the coefficients of the Chebyshev polynomial for the boundary temperature function on the upper and lower boundaries, while x<sup>5</sup> and x<sup>6</sup> represent the coefficients for the boundary temperature function on the left and right boundaries.

## E.2. Interpretable Analysis Details

As shown in Table [8,](#page-19-1) the variables are defined as follows: in the first row, x<sup>1</sup> and x<sup>2</sup> represent the size of the boundary for PDE solutions; in the second row, x<sup>2</sup> represents the coefficient of a second-order coupling term; in the third row, x<sup>4</sup> is the coefficient of the fourth x-term multiplied by the first y-term in a two-dimensional Chebyshev polynomial; in the fourth row, x<sup>2</sup> again denotes the coefficient of a second-order coupling term; in the fifth row, x1x<sup>3</sup> signifies the coefficient of a second-order non-coupling term.

1054

1056

1074

1076

1079

Table 8: Symbolic expressions learned from the main experiments

| Precondition | Dataset      | Symbolic expression                           |
|--------------|--------------|-----------------------------------------------|
| SOR          | Biharmonic   | 1 0 + 1 0 / (4 0 + 1 0 /x 2 ) + 1 0 /x 1      |
| SOR          | Elliptic PDE | 1 0 + 1 0 / ( x 2 + 1 0 + 1 0 / ( x 2 + 4 0)) |
| SOR          | Darcy Flow   | 1 0 + 1 0 / ( x 4 + 1 0)                      |
| SSOR         | Elliptic PDE | 1 0 + 1 0 / ( x 2 + 1 2)                      |
| AMG          | Elliptic PDE | ( x 1 x 3 + 1) / 7                            |

#### E.3. Analysis of Hyperparameters

The performance of SymMaP is primarily influenced by the learning rate of the RNN, batch size, and dataset size. We conducted experiments to study the impact of these hyperparameters.

## Symbolic Learning RNN Parameters:

Table 9: Performance comparison of SymMaP under various symbolic learning RNN parameters (lower condition numbers are preferable). The experiment focuses on optimizing AMG preconditioning coefficients in the Darcy Flow dataset.

| Learning Rate Batch Size | Condition number | Training time(s) |
|--------------------------|------------------|------------------|
| 500                      | 6780             | 1173.09          |
| 1000                     | 5168             | 863.51           |
| 2000                     | 6898             | 522.80           |
| 500                      | 5935             | 1104.16          |
| 1000                     | 11774            | 676.40           |
| 2000                     | 5935             | 505.85           |
| 500                      | 4718             | 1026.45          |
| 1000                     | 5935             | 703.17           |
| 2000                     | 5935             | 549.36           |
| 500                      | 12228            | 1324.00          |
| 1000                     | 7508             | 837.18           |
| 2000                     | 6884             | 603.62           |

Results in Table [9](#page-19-2) indicate that an appropriate combination of RNN learning rate and batch size can enhance performance.

#### Dataset size:

Table 10: Performance comparison of SymMaP across varying dataset sizes (lower condition numbers indicate better performance). The experiment evaluates the optimization of AMG preconditioning coefficients for the Darcy Flow dataset.

| Dataset size | Condition number | Training time (s) |
|--------------|------------------|-------------------|
| 10           | 7032             | 669.68            |
| 50           | 6980             | 737.80            |
| 100          | 4892             | 812.02            |
| 500          | 3811             | 699.54            |
| 1000         | 5345             | 703.17            |