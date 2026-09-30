# Symmap: Improving Computational Efficiency In Linear Solvers Through Symbolic Preconditioning

## Anonymous Authors1 Abstract

Matrix preconditioning is a critical technique to accelerate the solving of linear systems, where performance heavily depends on the selection of preconditioning parameters. Traditional parameter selection approaches often define fixed constants for specific scenarios. However, they rely on domain expertise and fail to consider the instance-wise features for individual problems, limiting their performance. In contrast, machine learning (ML) approaches, though promising, are hindered by high inference costs and limited interpretability. To combine the strengths of both approaches, we propose a symbolic discovery framework—namely, Symbolic Matrix Preconditioning (**SymMaP**)—to learn efficient symbolic expressions for preconditioning parameters. Specifically, we employ a large neural network to search the high-dimensional discrete space for expressions that can accurately predict the optimal parameters. The learned expression allows for high inference efficiency and excellent interpretability (expressed in concise symbolic formulas), making it simple and reliable for deployment. Experimental results show that SymMaP consistently outperforms traditional strategies across various benchmarks.

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054

## 1. Introduction

Linear systems are foundational in the machine learning, physics, engineering, and other scientific fields (Leon et al., 2006; LeVeque, 2007). Since analytical solutions are often unavailable, efficient numerical algorithms become essential (Demmel, 1997). Matrix preconditioning, a key technique in this domain, accelerates iterative solvers and improves computational stability (Trefethen & Bau, 2022; Chen, 2005). For instance, the successive over-relaxation (SOR) method optimizes convergence by integrating Gauss- Seidel iterations with a weighted update scheme governed by the over-relaxation factor ω (Golub & Van Loan, 2013). The effectiveness of matrix preconditioning depends on 1 key parameters, such as ω in SOR. Selecting ω > 1 can accelerate convergence, while ω < 1 may stabilize the process. This trade-off makes ω a critical parameter, directly influencing the performance of preconditioning. Traditional parameter selection strategies often rely on domain expertise to define fixed constants for specific scenarios. However, (**challenge 1**) different problem parameters often require distinct optimal preconditioning parameters. Traditional strategies ignore instance-wise features—specific characteristics of individual problems, such as equation coefficients. This limits their adaptability to varying problem instances and tasks. In contrast, machine learning (ML) approaches hold great promise but come with other challenges. First, (**challenge** 2) ML inference,while efficient on GPUs, performs poorly in CPU-only environments due to limited parallel processing capabilities. This is particularly problematic in linear system solver deployments, where GPU resources are often unavailable. Second, (**challenge 3**) the "black-box" nature of many ML techniques hinders a deeper understanding of the learned policies, raising concerns about their reliability.

In light of this, a natural solution is to combine the reliability and superior performance of these two paradigms. We propose a symbolic discovery framework—namely, Symbolic Matrix Preconditioning (**SymMaP**)—to learn efficient symbolic expressions for preconditioning parameters. The framework consists of three main steps. SymMaP first begins by applying grid search to identify the optimal preconditioning parameters based on task-specific performance metrics. Next, the framework performs a risk-seeking search in the high-dimensional discrete space of symbolic expressions, evaluating the best-found symbolic expression using a risk-seeking strategy. Finally, these symbolic expressions can be directly integrated into the modern solvers for linear systems, significantly improving computational efficiency. The key contributions and advantages of SymMaP are summarized as follows:
- We propose a symbolic discovery framework, SymMaP,
to learn efficient symbolic expressions for preconditioning parameters.

- SymMaP exhibits excellent generalization, making it adaptable to a wide range of preconditioning methods and optimization objectives.

- The symbolic expressions derived by SymMaP are both interpretable and easy to integrate into solver environments, offering a practical and transparent approach to enhancing the performance of linear system solvers.

## 2. Preliminaries 2.1. Matrix Preconditioning Technique

Matrix preconditioning is a technique employed to accelerate the convergence of iterative solvers and enhance the stability of algorithms. It is generally employed in solving linear systems (Chen, 2005; Golub & Van Loan, 2013),
which are typically expressed in the form:

$$A x=b.$$
Ax = b. (1)
$$M A\mathbf{x}=M\mathbf{b}.$$
MAx = M b. (2)
There are generally two optimization objectives: 1. to accelerate the convergence of iterations by altering the spectral distribution of the matrix A. 2. to reduce the condition number of the matrix A, thereby lessening its ill-conditioning and enhancing the stability of iteration. Some common preconditioning techniques include the Jacobi, Gauss-Seidel, SOR (Young, 1954), algebraic multigrid (AMG) (Trottenberg et al., 2000), etc.

## 2.2. Prefix Notation

055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 Prefix notation is a mathematical format where every operator precedes its operands, eliminating the need for parentheses required in conventional infix notation and simplifying symbolic manipulation. This representation is particularly advantageous in symbolic regression, as it allows mathematical expressions to be expressed as sequences of tokens that can be easily processed by neural networks.

In this notation, operators can be unary (e.g., sin, cos) or binary (e.g., +, −, ×, ÷), while operands can be constants or variables (Landajuela et al., 2021). Each prefix expression uniquely corresponds to a symbolic tree structure, facilitating the conversion back to the original mathematical expression (Lample & Charton, 2019).

10 2 Iterations op Time 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 SOR preconditioning parameter 10 1 10 2 10 3 10 4 10 5 10 1 Ite ra ti o n s Ti m e( s)
10 0 optimal parameter 10 1 10 2
The sequential nature of prefix notation aligns well with the architecture of recurrent neural networks (RNNs), which process information step by step. Unlike infix notation which may require variable-length look-ahead to determine the next valid token, prefix notation allows RNN to generate expressions through an auto-regressive process where each decision is well-defined based on previous tokens, and by removing the need for parentheses, it reduces the vocabulary size of possible tokens, which greatly enhances model training efficiency.

## 3. Motivation

The selection of matrix preconditioning parameters significantly affects their effectiveness (Chen, 2005). To design appropriate algorithmic prediction parameters, we first analyze the optimization space for preconditioning parameter selection and investigate the existence of optimal parameters. Next, we analyze the unique challenges present in this scenario. Finally, to address these challenges, we design a symbolic discovery framework to select the parameters.

## 3.1. Motivation For Optimizing Preconditioning Parameters

As illustrated in figure 1, the choice of relaxation factors ω significantly impacts the iteration count and computation time, when solving a second-order elliptic partial differential equation (PDE) (Evans, 2022) using SOR preconditioning (Golub & Van Loan, 2013). There exists an optimal parameter ωop that minimizes the computation time, with specific details available in the Appendix B.2.

To further analyze the optimization space of preconditioning parameters, we evaluate the impact of various parameter selection strategies on preconditioning performance.

The fundamental idea of preconditioning is to transform the original problem into an equivalent one with better numerical properties. Specifically, this technique involves finding a preconditioner M that approximates either the inverse of A or some form conducive to iterative solutions (Chen, 2005). Consequently, the original (1) is transformed into As shown in Figure 2, the 'Optimal Parameter ωop' represents the parameter that minimizes computation time in each experiment, serving as the theoretical upper limit of our optimization. The 'Optimal Fixed Constant' refers to a fixed constant that minimizes average computation time, and 'Default Parameter' corresponds to the default setting of ω = 1 in the portable extensible toolkit for scientific computation (PETSc) (Balay et al., 2024). The gap between the optimal fixed constant and the optimal parameter highlights significant potential for optimizing preconditioning parameter selection, motivating this paper. The performance of our SymMaP algorithm approaches the optimal parameter, demonstrating its accuracy in learning the optimal parameter expression.

## 3.2. Challenges In Predicting Efficient Preconditioning Parameters

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 We aim to develop a universal framework for predicting efficient parameters. However, the context of solving linear systems imposes specific challenges: (C1) Strong Generalization Capability: Real-world scientific computing scenarios vary significantly. For instance, the choice of PDE grid form can lead to significant variations in matrix structure (Johnson, 2009), resulting in distinct optimal parameters. Moreover, preconditioning addresses various optimization goals, such as reducing computational time, the number of iterations, and lowering condition numbers (Chen, 2005). This necessitates that parameter prediction algorithms possess robust generalization capabilities: they should take problem scenarios and features as inputs while applying them to different preconditioning methods and optimization goals.

94%
100% 100% 100% 100%
100%
87% 87%
92%
86%
90%
Ti m e Ra ti o 83%
78%
73%
85% 85%
80%
78%
73%
Optimal Parameter op SymMaP Optimal Fixed Constant Default Parameter 70%
1e-5 1e-6 1e-7 1e-8 Tolerance (2 Norm)
60%
(C2) Computational Efficiency: linear system solver typically relies on Krylov subspace methods implemented in low-level libraries optimized for CPU architectures, such as PETSc (Balay et al., 2024), LAPACK (Anderson et al., 1999). Algorithms like generalized minimal residual method (GMRES) (Saad & Schultz, 1986) and conjugate gradient (CG) (Greenbaum, 1997) iteratively compute the matrix's invariant subspace, favoring single-threaded or limited multithreaded execution modes. Preconditioning techniques aim to accelerate these solvers without significant additional computational overhead, often adopting implicit iterative formats (e.g., SOR (Chen, 2005)) or utilizing low-cost matrix decompositions (e.g., AMG (Trottenberg et al., 2000)). Therefore, any parameter prediction algorithms must be compatible with CPU environments and seamlessly integrated into existing algorithm libraries. At the same time, it must maintain low computational costs to preserve the performance benefits of preconditioning. (C3) Algorithmic Transparency: Algorithms in scientific computing often require rigorous analysis under mathematical theories. Opaque prediction algorithms could confuse researchers. For instance, the relaxation factor ω in SOR
needs to avoid being too close to 0 or 2 in some scenarios (Agarwal, 2000). This is an issue that opaque algorithms cannot avoid in advance. Moreover, interpretable algorithms can guide researchers to conduct further studies and reveal the underlying mathematical structures of problems. Therefore, these pose challenges to the transparency and interpretability of the parameter prediction algorithms.

## 3.3. Symbolic Discovery To Preconditioning Parameter Selection

Symbolic discovery extracts mathematical expressions from data, establishing relationships between problem features and optimal preconditioning parameters. Its integration into matrix preconditioning overcomes parameter selection challenges through a generalizable, efficient, and transparent approach. Firstly, symbolic discovery can accommodate various types of input parameters and can specifically tailor symbolic expression learning for different preconditioning methods and optimization goals (Cranmer et al., 2020), thereby meeting the requirement for broad applicability in scientific computing tasks (C1). Secondly, the explicit expressions derived are computationally lightweight and can be quickly evaluated at runtime. They integrate seamlessly into existing CPU-based algorithm libraries like PETSc (Balay et al., 2024) with almost no overhead (C2). Thirdly, the symbolic discovery provides transparent and interpretable expressions (Rudin, 2019), allowing researchers to understand the influence of parameters within existing theoretical frameworks and identify potential numerical stability issues. This interpretability fosters trust in the algorithm's predictions and supports further theoretical exploration (C3).

## 4. Method

This study focuses on enhancing the performance of parameterized preconditioners in solving linear systems derived from parameterized PDEs. Specifically, we investigate preconditioners with continuous parameters, such as the relaxation factor in SOR, while excluding those with discrete parameters like the level of fill-in ICC or ILU factorization. We introduce a novel framework, SymMaP, for symbolic discovery in matrix preconditioning. As shown in Figure 3, we first obtain the optimal preconditioning parameters through a grid search to construct a training dataset. Then we employ an RNN to generate symbolic expressions in prefix notation, which are then evaluated for their fitness. The RNN is trained using a reward function based on the performance of the generated expressions. By optimizing the RNN parameters to maximize this reward function, we generate symbolic expressions that approximate the relationship between the problem's feature parameters (PDE parameters) and the optimal preconditioning parameters. Finally, we deploy the learned symbolic expressions into linear system solvers. The detailed steps are as follows and pseudocode is 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219

1. Training Data Generation Sampled Token 2. Evalutate Expressions and Train the Deep Symbolic Model 3. Sequential Model Parent
& Sibling RNN
dataset initial parameter parameter preconditioner log update log prefix expression sequence generationg expression adaptive grid search comput ation tim e optimal parameter constant optimization R L 
tr ain er const parameter calculate dataset const NO
expression Yes or 4. Deployment in linear solvers Symbolic Experssion Tree Token Library log compilled to library integrated to linear solver const const const exp log

provided in the Appendix C.

## 4.1. Input Features And Training Data Generation

Input Features. In the context of solving parameterized PDEs, which frequently arise in linear systems, we consider feature parameters that characterize the equations. For instance, a second-order elliptic PDE can be expressed as: a11uxx + a12uxy + a22uyy + a1ux + a2uy + a0u = f, where the coefficients a11, a12, a22*, . . .* represent the feature parameters of PDE (see Appendix D.1 for details). These feature parameters, denoted as xi, serve as input features for the symbolic discovery process in SymMaP.

Training Data Generation. For each linear system, we determine optimal preconditioning parameters through an adaptive grid search. Using SOR preconditioning as an example, we optimize the relaxation factor ω within [0, 2] to minimize computation time (or condition number). The search process involves: 1. Conducting an initial coarse grid search (step size: 0.01) to evaluate computation time (or condition number) for each ω. 2. Identifying candidate regions with optimal performance. 3. Performing a refined grid search (step size: 0.001) within these regions. This process yields the required training dataset, where each data point contains: 1. the problem feature parameters xi; 2. the optimal preconditioning parameters yi. i = 1, 2*, . . . , n*,
and n is the number of data, typically set to n = 1200, with 1000 allocated for the training set and 200 for the test set.

220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274

## 4.2. The Generation Of Symbolic Expressions

Token Library. For SymMaP, we define the library L of mathematical operators and operands as {+, −, ×, ÷,sqrt, exp, log, pow, 1.0}. Although other operators such as poly, sin and cos are frequently used (Udrescu & Tegmark, 2020), we decided to exclude them because they offer limited explanatory power in matrix preconditioning and significantly increase the time and memory consumption during training. After converting the mathematical expressions into prefix notation, we use this tokenized representation as a preorder traversal of the expression tree (Zaremba & Sutskever, 2014). In each iteration, the RNN receives a pair consisting of a parent node and a sibling node as inputs. Then the RNN outputs a categorical distribution over all possible next tokens. The parent node refers to the last incomplete operator that requires additional operands to form a complete expression. The sibling node, in the context of a binary operator, represents the operand that has already been processed and incorporated into the expression. In cases where no parent or sibling node is applicable, they are designated as empty nodes. This structured input method enables the RNN to maintain contextual awareness and effectively predict the sequence of tokens that form valid mathematical expressions. The Sequential Model. During the generation of a single symbolic expression, the RNN emits a categorical distribution for each "next token" at each step. This distribution is represented as a vector ψ
(i)
θ, where i denotes the i-th step and θ represents the parameters of the RNN. The elements of the vector correspond to the probabilities of each token, conditioned on the previously selected tokens in the traversal (Petersen et al., 2019):

$$\psi_{\theta}^{(i)}(\mathbf{\tau}_{i})=\mathbf{p}(\mathbf{\tau}_{i}|\mathbf{\tau}_{1:i-1};\mathbf{\theta}).$$

Here, τi denotes the index of the token selected at the ith step. The probability of generating the entire symbolic expression τ is then the product of the conditional probabilities of all tokens (Petersen et al., 2019; Landajuela et al., 2021):

$$p(\tau|\theta)=\prod_{i=1}^{N}\psi_{\theta}^{(i)}(\tau_{i}).$$

(τi). (4)
Optimization of Constants: The library L incorporates a 'constant token,' which allows for the inclusion of various constant placeholders within sampled expressions. These placeholders serve as the parameters ξ in the symbolic expression. We seek to find the optimal values of these parameters by maximizing the reward function: ξ
∗ = arg maxξ R(τ ; ξ), utilizing a nonlinear optimization method. This optimization is executed within each sampled expression as an integral part of computing the reward, prior to each training iteration.

## 4.3. The Reward Function

Once a symbolic expression is fully generated (i.e., the symbolic tree reaches all its leaf nodes), we evaluate its fitness by calculating the normalized root-mean-square error (NRMSE), a metric frequently used in genetic programming symbolic discovery (Schmidt & Lipson, 2009). It is defined as

$${\mathrm{NRMSE}}={\frac{1}{\sigma_{y}}}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_{i}-{\hat{y}}_{i})^{2}}},$$
$$\mathbf{(5)}$$

where yˆi = τ (xi) is the predicted value for the i-th sample, xiis the problem feature parameter, yiis the optimal preconditioning parameter, σy is the standard deviation of the target values y, and n is the number of data. To bound this fitness value between 0 and 1, we apply a squashing function:

$R(\tau)=\frac{1}{1+\text{NRMSE}}$.  
Our objective is to maximize R(τ ), thereby minimizing the NRMSE and improving the accuracy of the generated expressions.

## 4.4. The Training Algorithm

$$({\mathfrak{I}})$$

Although the objective function is well-defined, it is important to note that R(τ ) is not a deterministic value but a random variable dependent on the RNN's parameters θ. Therefore, the key challenge is to establish an appropriate criterion for evaluating this random variable, and then apply gradient-based optimization methods accordingly. Risk-seeking Policy. It is natural to consider the expectation of the reward function, i.e., Eτ∼p(τ;θ)[R(τ )], as the objective function to optimize. We can apply the 'log-integral' trick (Williams, 1992) and obtain

$$(4)$$
$$\nabla_{\mathbf{\theta}}\,\mathbb{E}_{\mathbf{\tau}\sim\mathbf{p}(\mathbf{\tau};\mathbf{\theta})}[R(\mathbf{\tau})]=\mathbb{E}_{\mathbf{\tau}\sim\mathbf{p}(\mathbf{\tau};\mathbf{\theta})}[R(\mathbf{\tau})\nabla_{\mathbf{\theta}}\log\mathbf{p}(\mathbf{\tau};\mathbf{\theta})].\tag{7}$$
$$\begin{array}{l}{{\mathsf{I})].}}\\ {{(7)}}\end{array}$$

Thus, even though the expectation of the reward function is not directly differentiable with respect to θ, we can approximate the gradient using the sample mean.

In the context of symbolic regression, model performance is often driven by a few exceptional results that outperform others by a significant margin (Petersen et al., 2019; Tamar et al., 2015). With this in mind, we adopt a risk-seeking policy, which aims to maximize:

$$J(\theta,\varepsilon)=\mathbb{E}_{\tau\sim p(\tau;\theta)}[R(\tau)|R(\tau)>Q(\theta,\varepsilon)].\tag{8}$$

Here, ε is the risk factor, typically ε = 0.05, Q(θ, ε) is the
(1 − ε)-quantile of the reward distribution under parameter θ, i.e.

$$Q(\mathbf{\theta},\varepsilon)=\inf\{q\in\mathbb{R}|\text{CDF}(R(\mathbf{\tau});\mathbf{\theta})\geq1-\varepsilon\},\tag{9}$$

where CDF(R(τ ); θ) refers to the cumulative distribution function. From this, the gradient of J(θ, ε) can be derived as (Petersen et al., 2019):

$$\begin{array}{c}{{\nabla_{\mathbf{\theta}}J(\mathbf{\theta},\varepsilon)=\mathbb{E}_{\mathbf{\tau}\sim\mathbf{p}(\mathbf{\tau};\mathbf{\theta})}\left[\left(R(\mathbf{\tau})-Q(\mathbf{\theta},\varepsilon)\right)\right.}}\\ {{\left.\cdot\nabla_{\mathbf{\theta}}\log\mathbf{p}(\mathbf{\tau};\mathbf{\theta})\right|R(\mathbf{\tau})>Q(\mathbf{\theta},\varepsilon)\right].}}\end{array}$$
$$(10)$$

This gradient can be estimated using Monte Carlo sampling:

$$\nabla_{\theta}J(\theta,\varepsilon)\approx{\hat{g}}\triangleq$$
$$\begin{array}{l}\mbox{$\theta J(\theta,\varepsilon)\approx\hat{g}\stackrel{{\mbox{\scriptsize$\Delta$}}}{{=}}$}\\ \mbox{$\frac{1}{\varepsilon N}\sum_{i=1}^{N}(R(\tau^{(i)})-\tilde{Q}(\theta,\varepsilon))\nabla_{\theta}\log p\cdot\mathbb{1}_{R(\tau^{(i)})>\tilde{Q}(\theta,\varepsilon)}$}\end{array}$$

Q˜(θ, ε) is the empirical (1 − ε)-quantile of the reward function. By concentrating on the top ε percentile of samples, SymMaP emphasizes optimizing the best-performing solutions in preconditioning, thereby obtaining the optimal symbolic expressions for preconditioning parameters.

## 4.5. Deployment In Linear System Solver

After the training process, we obtained a symbolic expression for predicting the preconditioning parameter. The 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 learned formula is exceptionally concise and incurs minimal computational cost. Therefore, we directly compile the learned policy into a lightweight shared object using a simple script and then integrate it into the linear system solver package (e.g., PETSc).

## 5. Experiments

We conducted comprehensive experiments to evaluate the SymMaP framework, organized into three primary sections:
1. Assessment of three different preconditioners and optimization goals across various datasets to determine the effectiveness of SymMaP algorithm, 2. Analysis of associated computational cost and the interpretability of the learned symbolic expressions, 3. Ablation studies of SymMaP.

$$(11)^{\frac{1}{2}}$$

Preconditioners: We considered three different preconditioners and various optimization metrics: 1. SOR preconditioner with the relaxation factor ω (Golub & Van Loan, 2013); 2. SSOR preconditioner with the relaxation factor ω (Golub & Van Loan, 2013); 3. AMG preconditioner with the threshold parameters θT (Trottenberg et al., 2000).

Datasets: We investigated linear systems derived from five distinct PDE classes: 1. Darcy Flow Problems (Li et al., 2020), 2. Second-order Elliptic PDEs (Evans, 2022), 3. Biharmonic Equations (Barrata et al., 2023), 4. Thermal Problems (Wang et al., 2024). 5. Poisson Equations (Wang et al., 2024). All cases except biharmonic equations yield symmetric matrices. Notably, the non-symmetric matrices from biharmonic equations are incompatible with SSOR and AMG preconditioning techniques. Baselines: We compared SymMaP against various parameter selection methods for preconditioning. Specifically, the comparison involved the following scenarios: 1. No matrix preconditioning, 2. Default parameters in PETSc (Balay et al., 2024), 3. Fixed constants, 4. Optimized fixed constants. Experiment Settings: All preconditioning procedures

| Dataset      | Matrix size   | No        | PETSc default   | Fixed constant   | Fixed constant   | Optimal constant   | SymMaP 1   | SymMaP 2   |
|--------------|---------------|-----------|-----------------|------------------|------------------|--------------------|------------|------------|
| precondition | ω = 1         | ω = 0.2   | ω = 1.8         |                  |                  |                    |            |            |
| Biharmonic   | 4.2 × 103     | 7.67      | 2.04            | 4.86             | 1.60             | 1.31               | 1.24       | 1.26       |
| Darcy Flow   | 1.0 × 104     | 33.1      | 13.5            | 17.5             | 9.91             | 9.54               | 8.50       | 8.60       |
| Elliptic PDE | 4.0 × 104     | 31.3      | 21.0            | 21.4             | 17.5             | 16.6               | 15.8       | 16.3       |
| Poisson      | 2.3 × 103     | 4.12×10−2 | 1.95×10−2       | 2.15×10−2        | 1.95×10−2        | 1.38×10−2          | 1.35×10−2  | 1.36×10−2  |
| Thermal      | 2.8 × 103     | 2.23×10−1 | 5.98×10−2       | 2.07×10−1        | 1.18×10−1        | 5.94×10−2          | 5.76×10−2  | 5.91×10−2  |

Darcy Flow 4.9 × 103 4.18 0.488 0.757 1.09 0.448 **0.412** 0.523

Elliptic PDE 4.0 × 104 23.9 10.5 14.7 8.72 8.68 **7.70** 7.74

| Dataset      | Matrix size   | No        | PETSc default   | Fixed constant   | Fixed constant   | Optimal constant   | SymMaP 1   | SymMaP 2   |
|--------------|---------------|-----------|-----------------|------------------|------------------|--------------------|------------|------------|
| precondition | ω = 1         | ω = 0.2   | ω = 1.8         |                  |                  |                    |            |            |
| Poisson      | 2.3 × 103     | 2.12×10−2 | 1.02×10−2       | 1.93×10−2        | 9.91×10−3        | 9.89×10−3          | 9.10×10−3  | 9.92×10−3  |
| Thermal      | 2.8 × 103     | 2.34×10−1 | 2.69×10−2       | 5.08×10−2        | 9.87×10−2        | 2.24×10−2          | 2.13×10−2  | 2.14×10−2  |

Table 3: Comparison of average condition numbers for preconditioned matrices using different threshold parameter θT
selections in AMG. SymMaP 1 and 2 are the two learned expressions that achieved the highest reward function scores, with the best-performing method highlighted in bold.

330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 were uniformly implemented using the C-based PETSc library (Balay et al., 2024) to maintain evaluation consistency. The experiments were conducted within PETSc's GMRES linear solver framework (Golub & Van Loan, 2013), with condition numbers computed through the built-in function KSPComputeExtremeSingularValues.

Details on preconditioners, the mathematical forms of datasets, and the runtime environment are available in Appendices B, D.1, and D.2, respectively. Information on the generation of training datasets for the following experiments and parameters of the SymMaP algorithm are outlined in Appendices D.3 and D.4. The generated dataset and training time are available in Appendix D.5. For an introduction to related work, see Appendix A.

| Dataset      | Matrix size   | No       | PETSc default   | Fixed constant   | Fixed constant   | Optimal constant   | SymMaP 1   | SymMaP 2   |
|--------------|---------------|----------|-----------------|------------------|------------------|--------------------|------------|------------|
| precondition | θT = 0        | θT = 0.2 | θT = 0.8        |                  |                  |                    |            |            |
| Darcy Flow   | 1.0 × 104     | 752862   | 8204            | 19146            | 11426            | 7184               | 4824       | 5786       |
| Elliptic PDE | 4.0 × 104     | 6792     | 184.6           | 205.4            | 212.5            | 182.8              | 168.8      | 170.3      |
| Poisson      | 1.0 × 104     | 1242     | 4.55            | 68.85            | 68.85            | 4.55               | 3.72       | 3.72       |
| Thermal      | 2.8 × 103     | 7325     | 11.9            | 627.2            | 627.2            | 9.91               | 9.71       | 9.71       |

## 5.1. Main Experiments

In these experiments, as shown in Tables 1, 2, 3, we optimized relaxation factors ω in both SOR and SSOR preconditioning, and threshold parameters θT in AMG preconditioning. For SOR and SSOR, we identified ω values that minimize computation time, forming the training dataset for SymMaP to learn symbolic expressions that optimize computational times for solutions. Similarly, for AMG, we selected θT values that minimize the condition number of preconditioned matrices. Partial symbolic expressions can be found in Appendix E.1.

Experimental results indicate that SymMaP consistently outperforms all others across all experimental tasks. For SOR, Table 1 shows that SymMaP reduces computation times by up to 40% compared to PETSc's default settings and by 10% against the optimal constants. In SSOR, Table 2 shows that it cuts computation time and iteration counts by up to 27%, over PETSc's defaults, and achieves reductions of 11% in time compared to optimal constants. For AMG,
Table 3 shows that SymMaP lowers the condition number by up to 40% relative to PETSc's defaults and 32% against the optimal constants. These results highlight SymMaP's ability to effectively derive high-performance symbolic expressions for various preconditioning parameters, showcasing its broad applicability and strong generalization across different preconditioning tasks.

## 5.2. Comparison With Neural Network Performance

To evaluate the deployment overhead and prediction performance of SymMaP, we compared it with a basic multilayer perceptron (MLP) architecture. The MLP implementation consists of three fully connected layers, taking PDE parameters as input and generating preconditioning parameters as output. We employed ReLU activation functions and trained the model using mean squared error (MSE) between predicted and optimal parameters as the loss function. Both Table 4: Comparison of the runtime required for symbolic expression and MLP to predict the SOR relaxation factor and the subsequent average solution time for linear systems, using the Darcy flow dataset with a matrix size of 103and tolerance is 1e-5.

| Runtime (s)   | Solution time (s)   |        |
|---------------|---------------------|--------|
| MLP           | 5.1e-5              | 7.1e-1 |
| Symbol        | 1.1e-5              | 7.1e-1 |

symbolic expression and MLP were executed in a CPU environment to simulate a modern solver environment.

| Precondition   | Dataset      | Symbolic expression                   |
|----------------|--------------|---------------------------------------|
| SOR            | Biharmonic   | 1.0 + 1.0/(4.0 + 1.0/x2) + 1.0/x1     |
| SOR            | Elliptic PDE | 1.0 + 1.0/(x2 + 1.0 + 1.0/(x2 + 4.0)) |
| SOR            | Darcy Flow   | 1.0 + 1.0/(x4 + 1.0)                  |
| SSOR           | Elliptic PDE | 1.0 + 1.0/(x2 + 1.2)                  |
| AMG            | Elliptic PDE | (x1x3 + 1)/7                          |

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 As shown in Table 4, the runtime of symbolic expressions learned by SymMaP was only 20% of that of the MLP, primarily due to the poor performance of neural networks in a pure CPU environment, highlighting SymMaP's computational efficiency. Furthermore, the average solution times for parameters predicted by both symbolic expressions and MLP were closely matched. This demonstrates that symbolic expressions possess equivalent expressive capabilities to neural networks in this scenario, effectively approximating the optimal parameter expressions.

## 5.3. Interpretable Analysis

In Table 5, we report a subset of the learned symbolic expressions, with the mathematical significance of the related symbols detailed in Appendix E.2. More symbolic expressions can be found in Appendix E.1. These symbolic expressions are notably more concise and selective, not utilizing all candidate parameters and symbols, which aids researchers in analyzing their underlying relationships. For instance, in the context of SOR and SSOR preconditioning, empirical evidence suggests that smaller relaxation factors should be chosen when diagonal components are relatively small. Our experimental findings corroborate this: for the second-order elliptical PDE dataset, the symbolic expressions derived for SOR and SSOR preconditioning depend solely on x2, with larger x2 values leading to smaller predicted relaxation factors, exemplified by 1.0 + 1.0
(x2+1.2) .

Here, x2 represents the coupling coefficient of the elliptical Table 6: Ablation study examining the selection of mathematical operators, comparing the effects on preconditioning and training times. The first column lists the selected operators, the second column shows the condition numbers of preconditioned matrices derived from AMG parameter predictions on the Darcy flow dataset (lower is better), and the third column displays SymMaP training times. PDE, which directly influences the relative size of the nondiagonal components of the generated matrix, whereas other coefficients have minimal impact. As the coupling coefficient increases, the relative numerical of the non-diagonal components increases, and the diagonal components reduce correspondingly, aligning with empirical observations. These experimental outcomes demonstrate that SymMaP can derive interpretable and efficient symbolic expressions for parameters, further aiding researchers in understanding and exploring the underlying mathematical principles.

| Functionset                                    | Condition number   | time(s)   |
|------------------------------------------------|--------------------|-----------|
| +, −, ×, ÷, poly                               | 6803.8             | 15351     |
| +, −, ×, ÷, sqrt, exp, log, pow, 1.0           | 7086.9             | 703.17    |
| +, −, ×, ÷, sqrt, exp, log, sin, cos, pow, 1.0 | 7172.6             | 635.82    |
| +, −, ÷, 1.0, pow                              | 7241.8             | 703.26    |
| +, −, ×, ÷, sqrt, pow, 1.0                     | 7271.1             | 746.80    |
| +, −, ×, ÷, pow, 1.0                           | 7301.4             | 702.46    |

## 5.4. Ablation Experiments

We conducted an ablation study using SymMaP to evaluate the impact of different mathematical operator selections, as described in Table 6. In the main experiments, We utilized the operator set {+, −, ×, ÷,sqrt, exp, log, pow, 1.0} listed in the second row. The results indicate that this selection of operators achieves a balance between predictive performance and training time efficiency, meeting our expectations. Furthermore, experiments detailing the performance of SymMaP about variations in learning rate, batch size, and dataset size are documented in Appendix E.3.

## 6. Conclusions

In this paper, we propose SymMaP, a deep symbolic discovery framework designed for predicting efficient matrix preconditioning parameters. Experiments show that SymMaP can predict high-performance parameters and is applicable across a variety of preconditioning and optimization objectives. Additionally, SymMaP is easy to deploy with virtually no computational cost. We are confident in the symbolic model's immense potential for broad real-world applications, especially in matrix preconditioning.

## Impact Statement

Agarwal, R. P. *Difference equations and inequalities: theory,*
methods, and applications. CRC Press, 2000.

Anderson, E., Bai, Z., Bischof, C., Blackford, S., Demmel, J., Dongarra, J., Du Croz, J., Greenbaum, A., Hammarling, S., McKenney, A., and Sorensen, D. LAPACK Users' Guide. Society for Industrial and Applied Mathematics, Philadelphia, PA, third edition, 1999. ISBN 0-89871-4478 (paperback).

Barrata, I. A., Dean, J. P., Dokken, J. S., Habera, M., HALE,
J., Richardson, C., Rognes, M. E., Scroggs, M. W., Sime, N., and Wells, G. N. Dolfinx: The next generation fenics problem solving environment. 2023.

Bers, L., John, F., and Schechter, M. Partial differential equations. American Mathematical Soc., 1964.

Chen, K. Matrix preconditioning techniques and applications. Number 19. Cambridge University Press, 2005.

Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Pham, H., Dong, X., Luong, T., Hsieh, C.-J., Lu, Y., et al. Symbolic discovery of optimization algorithms. Advances in neural information processing systems, 36, 2024.

Cranmer, M., Sanchez Gonzalez, A., Battaglia, P., Xu, R.,
Cranmer, K., Spergel, D., and Ho, S. Discovering symbolic models from deep learning with inductive biases. Advances in neural information processing systems, 33:
17429–17442, 2020.

Demmel, J. W. *Applied numerical linear algebra*. SIAM,
1997.

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

Driscoll, T. A., Hale, N., and Trefethen, L. N. Chebfun guide, 2014.

Evans, L. C. *Partial differential equations*, volume 19.

American Mathematical Society, 2022.

Glowinski, R. and Pironneau, O. Numerical methods for the first biharmonic equation and for the two-dimensional stokes problem. *SIAM review*, 21(2):167–212, 1979.

Golub, G. H. and Van Loan, C. F. *Matrix computations*.

JHU press, 2013.

Gotz, M. and Anzt, H. Machine learning-aided numerical ¨
linear algebra: Convolutional neural networks for the efficient preconditioner generation. In *2018 IEEE/ACM*
9th Workshop on Latest Advances in Scalable Algorithms for Large-Scale Systems (scalA), pp. 49–56. IEEE, 2018.

Greenbaum, A. *Iterative methods for solving linear systems*.

SIAM, 1997.

Greenfeld, D., Galun, M., Basri, R., Yavneh, I., and Kimmel, R. Learning to optimize multigrid pde solvers. In International Conference on Machine Learning, pp. 2415–2423. PMLR, 2019.

Hsieh, J.-T., Zhao, S., Eismann, S., Mirabella, L., and Ermon, S. Learning neural pde solvers with convergence guarantees. *arXiv preprint arXiv:1906.01200*, 2019.

Johnson, C. Numerical solution of partial differential equations by the finite element method. Courier Corporation, 2009.

Koric, S. and Abueidda, D. W. Data-driven and physicsinformed deep learning operators for solution of heat conduction equation with parametric heat source. International Journal of Heat and Mass Transfer, 203:123809, 2023.

Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., and Anandkumar, A. Neural operator: Learning maps between function spaces. arXiv preprint arXiv:2108.08481, 2021.

Lample, G. and Charton, F. Deep learning for symbolic mathematics. *arXiv preprint arXiv:1912.01412*, 2019.

Landajuela, M., Petersen, B. K., Kim, S., Santiago, C. P.,
Glatt, R., Mundhenk, N., Pettit, J. F., and Faissol, D. Discovering symbolic policies with deep reinforcement learning. In International Conference on Machine Learning, pp. 5979–5989. PMLR, 2021.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Ciarlet, P. G. and Raviart, P.-A. A mixed finite element method for the biharmonic equation. In *Mathematical* aspects of finite elements in partial differential equations, pp. 125–145. Elsevier, 1974.

## References

Balay, S., Abhyankar, S., Adams, M. F., Benson, S., Brown, J., Brune, P., Buschelman, K., Constantinescu, E. M., Dalcin, L., Dener, A., Eijkhout, V., Faibussowitsch, J., Gropp, W. D., Hapla, V., Isaac, T., Jolivet, P., Karpeev, D., Kaushik, D., Knepley, M. G., Kong, F., Kruger, S., May, D. A., McInnes, L. C., Mills, R. T., Mitchell, L., Munson, T., Roman, J. E., Rupp, K., Sanan, P., Sarich, J., Smith, B. F., Zampini, S., Zhang, H., Zhang, H., and Zhang, J. PETSc Web page. https://petsc.org/, 2024. URL https://petsc.org/.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Leon, S. J., De Pillis, L. G., and De Pillis, L. G. Linear algebra with applications. Pearson Prentice Hall Upper Saddle River, NJ, 2006.

LeVeque, R. J. Finite difference methods for ordinary and partial differential equations: steady-state and timedependent problems. SIAM, 2007.

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar, A. Fourier neural operator for parametric partial differential equations. *arXiv preprint arXiv:2010.08895*, 2020.

Lu, L., Meng, X., Cai, S., Mao, Z., Goswami, S., Zhang, Z., and Karniadakis, G. E. A comprehensive and fair comparison of two neural operators (with practical extensions) based on fair data. Computer Methods in Applied Mechanics and Engineering, 393:114778, 2022.

Luz, I., Galun, M., Maron, H., Basri, R., and Yavneh, I.

Learning algebraic multigrid using graph neural networks. In *International Conference on Machine Learning*, pp. 6489–6499. PMLR, 2020.

Mankowitz, D. J., Michi, A., Zhernov, A., Gelmi, M., Selvi, M., Paduraru, C., Leurent, E., Iqbal, S., Lespiau, J.-B., Ahern, A., et al. Faster sorting algorithms discovered using deep reinforcement learning. *Nature*, 618(7964): 257–263, 2023.

Petersen, B. K., Landajuela, M., Mundhenk, T. N., Santiago, C. P., Kim, S. K., and Kim, J. T. Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients. arXiv preprint arXiv:1912.04871, 2019.

Poli, R., Langdon, W., and McPhee, N. A field guide to genetic programming (with contributions by jr koza)(2008).

Published via http://lulu. com, 2008.

Rahman, M. A., Ross, Z. E., and Azizzadenesheli, K.

U-no: U-shaped neural operators. arXiv preprint arXiv:2204.11127, 2022.

Rudin, C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature machine intelligence*, 1(5):206– 215, 2019.

Ruge, J. W. and Stuben, K. Algebraic multigrid. In ¨ Multigrid methods, pp. 73–130. SIAM, 1987.

Saad, Y. *Iterative methods for sparse linear systems*. SIAM,
2003.

Saad, Y. and Schultz, M. H. Gmres: A generalized minimal residual algorithm for solving nonsymmetric linear systems. SIAM Journal on scientific and statistical computing, 7(3):856–869, 1986.

Schmidt, M. and Lipson, H. Distilling free-form natural laws from experimental data. *science*, 324(5923):81–85, 2009.

Sharma, R., Farimani, A. B., Gomes, J., Eastman, P., and Pande, V. Weakly-supervised deep learning of heat transport via physics informed loss. arXiv preprint arXiv:1807.11374, 2018.

Stanaityte, R. ILU and Machine Learning Based Preconditioning for the Discretized Incompressible Navier-Stokes Equations. PhD thesis, University of Houston, 2020.

Taghibakhshi, A., MacLachlan, S., Olson, L., and West, M.

Optimization-based algebraic multigrid coarsening using reinforcement learning. *Advances in neural information* processing systems, 34:12129–12140, 2021.

Tamar, A., Glassner, Y., and Mannor, S. Policy gradients beyond expectations: Conditional value-at-risk. Citeseer, 2015.

Trefethen, L. N. and Bau, D. *Numerical linear algebra*.

SIAM, 2022.

Trottenberg, U., Oosterlee, C. W., and Schuller, A. Multigrid. Elsevier, 2000.

Udrescu, S.-M. and Tegmark, M. Ai feynman: A physicsinspired method for symbolic regression. Science Advances, 6(16):eaay2631, 2020.

Wang, H., Hao, Z., Wang, J., Geng, Z., Wang, Z., Li, B.,
and Wu, F. Accelerating data generation for neural operators via krylov subspace recycling. arXiv preprint arXiv:2401.09516, 2024.

Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine* learning, 8:229–256, 1992.

Young, D. Iterative methods for solving partial difference equations of elliptic type. Transactions of the American Mathematical Society, 76(1):92–111, 1954.

Zaremba, W. and Sutskever, I. Learning to execute. *arXiv* preprint arXiv:1410.4615, 2014.

Zhang, E., Kahana, A., Turkel, E., Ranade, R., Pathak, J.,
and Karniadakis, G. E. A hybrid iterative numerical transferable solver (hints) for pdes based on deep operator network and relaxation methods. arXiv preprint arXiv:2208.13273, 2022.

## A. Related Work A.1. Machine Learning For Algorithm Discovery

Machine learning has the potential to uncover implicit rules beyond human intuition from training data, enabling the construction of algorithms that outperform handcrafted programs. Approaches to algorithm discovery in machine learning encompass symbolic discovery, program search, and more. Specifically, program search focuses on optimizing the computational processes of algorithms. For example, Mankowitz et al. (2023) explores the discovery of faster sorting algorithms, while Chen et al. (2024) investigates efficient optimization algorithms. In contrast, symbolic discovery aims to search within the space of small mathematical expressions rather than computational streams (Petersen et al., 2019; Landajuela et al., 2021). This approach is analogous to an extreme form of model distillation, where knowledge extracted from black-box neural networks is distilled into explicit mathematical expressions. Traditional methods for symbolic discovery have relied on evolutionary algorithms, including genetic programming (Poli et al., 2008). Recently, deep learning has emerged as a powerful tool in this domain, offering enhanced representational capacity and new avenues for solving symbolic discovery problems (Schmidt & Lipson, 2009; Cranmer et al., 2020).

## A.2. Neural Networks For Matrix Preconditioning

Recent studies have explored the use of neural networks to improve matrix preconditioning techniques. (Greenfeld et al., 2019; Luz et al., 2020; Taghibakhshi et al., 2021) demonstrate the effectiveness of neural networks in refining multigrid preconditioning algorithms, thus streamlining the computational process. (Gotz & Anzt ¨ , 2018) utilized Convolutional Neural Networks (CNNs) for the optimization of block Jacobi preconditioning algorithms, while (Stanaityte, 2020) developed corresponding Incomplete Lower-Upper Decomposition (ILU) preconditioning algorithms leveraging machine learning insights. Although these algorithms achieved impressive results, they still face challenges such as limited interpretability and reduced computational efficiency when deployed in pure CPU environments. This paper attempts to address these issues by incorporating symbolic discovery into the framework.

## B. Detailed Introduction Of Matrix Preconditioning

B.1. Overview of Matrix Preconditioning Methods
- **Jacobi Method**: The Jacobi preconditioner utilizes only the diagonal elements of a matrix to precondition a linear system. By approximating the inverse of the diagonal matrix, this method is computationally simple and effective for systems with strong diagonal dominance. However, its convergence rate can be slow, and its performance diminishes for poorly conditioned or weakly diagonally dominant matrices. The Jacobi method is typically used as a baseline for comparison with more sophisticated preconditioners (Saad, 2003).

- **Gauss-Seidel (GS) Method**: The Gauss-Seidel preconditioner improves upon the Jacobi method by considering both the lower triangular and diagonal parts of the matrix in a sequential manner. Unlike the Jacobi method, which updates all variables simultaneously, the GS method updates each variable in sequence using the most recent values. This leads to faster convergence, especially for diagonally dominant matrices. However, the GS method can still struggle with poorly conditioned systems, and its forward-only approach can limit performance in some applications (Saad, 2003).

- **Successive Over-Relaxation (SOR)**: The SOR method builds on the Gauss-Seidel method by introducing a relaxation factor ω to accelerate convergence. This factor allows for over-relaxation (ω > 1) or under-relaxation (ω < 1), tuning the method for faster performance on certain types of problems. SOR can significantly reduce the number of iterations needed for convergence compared to both the Jacobi and GS methods, but choosing the optimal relaxation factor is problem-dependent (Young, 1954).

- **Symmetric Successive Over-Relaxation (SSOR)**: SSOR is a symmetric version of the SOR method, where relaxation is applied in both forward and backward sweeps of the matrix. This bidirectional process improves stability and is well-suited for use with iterative solvers like the conjugate gradient method, which requires symmetric preconditioners. SSOR's symmetry ensures that the preconditioner maintains the properties needed for efficient and stable convergence, making it a popular choice for symmetric positive-definite systems (Golub & Van Loan, 2013).

550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604
- **Algebraic Multigrid (AMG)**: AMG is an advanced preconditioning technique designed to handle large, sparse systems of linear equations, especially those arising from the discretization of partial differential equations. Unlike traditional methods, AMG operates on multiple levels of the matrix structure, coarsening the matrix to form a hierarchy of smaller systems that are easier to solve. Solutions on the coarser grids are then interpolated back to the finer grids. This multilevel approach makes AMG highly efficient for large-scale problems, as it can dramatically reduce the number of iterations needed to achieve convergence. AMG is often used in combination with methods like SSOR or Gauss-Seidel as a smoother on each grid level, and it is particularly effective in cases where the problem exhibits a multiscale nature (Ruge & Stuben ¨ , 1987).

Relationship Among Jacobi, GS, and SOR Methods: The Jacobi method is the simplest of the three, using only diagonal information. The GS method improves upon the Jacobi method by using both diagonal and lower triangular matrix elements to achieve faster convergence. SOR further refines the GS method by introducing a relaxation factor to optimize the update process. Both the GS and SOR methods can be seen as iterative improvements on the Jacobi method, with SOR offering a more flexible and potentially faster alternative by adjusting the relaxation factor. SSOR extends SOR symmetrically, making it suitable for use in more advanced iterative solvers like the conjugate gradient method (Saad, 2003; Golub & Van Loan, 2013).

## B.2. Parameters In Matrix Preconditioning

The choice of preconditioning parameters significantly influences the effectiveness of the preconditioning process, especially in the iterative solving of linear systems (Chen, 2005). Below, we discuss three specific preconditioning techniques—SOR, SSOR, and AMG—focusing particularly on how their key parameters affect the preconditioning results.

## B.2.1. Relaxation Factor Ω In Sor And Ssor Methods

In the SOR preconditioning method, the relaxation factor ω is a critical parameter that determines the acceleration of iteration. SOR evolves from the Gauss-Seidel method by introducing ω to speed up convergence. The SOR iteration formula is given by:

$$\mathbf{x}^{(k+1)}=(\mathbf{D}+\omega\mathbf{L})^{-1}\left[(1-\omega)\mathbf{D}\mathbf{x}^{(k)}+\omega\mathbf{b}-\omega\mathbf{U}\mathbf{x}^{(k)}\right],\tag{13}$$

where D, L, and U are the diagonal, strictly lower triangular, and strictly upper triangular parts of the matrix A, respectively (Golub & Van Loan, 2013). The SSOR preconditioning method can be represented by the following formula:

$$(12)$$
$$M_{\mathrm{SSOR}}=\frac{1}{\omega(2-\omega)}(D-\omega U)D^{-1}(D-\omega L),$$
$$(13)$$
(D − ωU)D−1(D − ωL), (13)
605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 where MSSOR constitutes the preconditioner, and D, L, U, and ω are defined similarly to their roles in the SOR method. This symmetrical formulation enhances the stability and effectiveness of the preconditioning, particularly benefiting symmetric positive-definite matrices by optimizing the convergence properties of the iterative solver (Golub & Van Loan, 2013).

The choice of ω directly impacts the speed of convergence and the condition number of the matrix. Different problems and scenarios often require different choices of ω, which typically need to be determined based on the specific properties of the problem and through numerical experimentation (Golub & Van Loan, 2013). In the PETSc library, the default relaxation factor ω for both SOR and SSOR is set to 1, at which point SOR degenerates to GS preconditioning.

## B.2.2. Threshold Parameters Θt In Amg

In the AMG method, the threshold parameter θT determines whether the non-zero elements of the matrix are "strong" enough to be considered in the construction of a coarse grid during the multigrid process. This parameter is crucial for establishing the connectivity between coarse and fine grids in the hierarchical multilevel structure (Ruge & Stuben ¨ , 1987). The AMG method solves the equation system through multiple levels of grids, each corresponding to a coarser version of the original problem. During this process, the threshold parameter is used to determine whether a given non-zero matrix element is strong enough to keep the corresponding grid points connected during coarsening.

- A lower threshold often leads to more elements being considered as strong connections, which might increase the complexity of the coarse grid but can help preserve the essential features of the original problem, thus improving the efficiency and convergence of the multigrid method.

- A higher threshold might result in fewer strong connections, thereby reducing the complexity of the coarse grid.

However, this can weaken the effectiveness of the AMG method, especially in maintaining the features of the original problem.

Different values of θT directly influence the condition number of the preconditioned matrix. Selecting the appropriate threshold parameter typically involves considering the specific structure and features of the problem, and adjustments are made through experimental fine-tuning to achieve the optimal balance (Trottenberg et al., 2000). In the PETSc library, the default threshold parameter θT is set to 0.

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714

## C. Algorithm Pseudocode

Algorithm 1 RNN-based Symbolic Discovery Process Input: RNN with parameter θ, the library of tokens L
τ ← [ ] parent(0), sibling(0) ← empty node x0 ← parent(0)||sibling(0) // x is the concatenation of parent and sibling nodes h0 ← 0 // Initialize hidden state of RNN
for t = 1, 2, *· · ·* do
(ψt, ht) ← RNN(xt−1, ht−1; θ) // ψt is the categorical distribution of the next token ψt ← ApplyConstraint(ψt,L, τ ) // Regularize the distribution Sample token τt ∼ ψt if *Arity*(τt) > 0 **then**
// Arity(τi) denotes the number of operands of τi parent(t) ← τt sibling(t) ← empty node else
// When Arity(τt) = 0, go back to the last incomplete operator node count ← 0 for i = t, t − 1*, . . . ,* 1 do
// Backward iteration count ← count + Arity(τi) −1 if *count = 0* **then**
parent(t) ← τi sibling(t) ← τi+1 **break**
715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769 if *count* = −1 **then**
// The expression sequence is complete break xt ← parent(t)||sibling(t)
Output: Prefix expression sequence τ Algorithm 2 Deep Symbolic Optimization for Matrix Preconditioning Parameter Input :RNN with initial parameter θ0, the library of tokens L, batch size N, iteration number J, risk factor ε, and learning rate α θ ← θ0 j ← 0 repeat for i = 1, 2*, . . . , N* do τ
(i) ← SymbolicDiscover(θ,L)
ξ
∗ ← {ξ in τ as constant placeholder : R(τ ; ξ)} // Constant optimization τ
(i) ← ReplaceConstant(τ
(i), ξ∗)
Compute gˆ1 using τ
(i)and θ // See Eq. (11)
Compute gˆ2 as entropy gradient θ ← θ + α(ˆg1 + ˆg2) // Update the parameter Train model: update pθ via PPO by optimizing J(θ; ϵ)
until j = J *or convergence* Output :The best symbolic expression τ
∗

## D. Experiment Settings

D.1. Datasets 1. Darcy Flow Problem We consider two-dimensional Darcy flows, which can be described by the following equation (Li et al., 2020; Rahman et al., 2022; Kovachki et al., 2021; Lu et al., 2022):

$$-\nabla\cdot(K(x,y)\nabla h(x,y))=f,$$

where K is the permeability field, h is the pressure, and f is a source term which can be either a constant or a space-dependent function.

In our experiment, K(*x, y*) is generated using truncated Chebyshev polynomials. We convert the Darcy flow problem into a system of linear equations using the central difference scheme of Finite Difference Methods (FDM) (LeVeque, 2007). The coefficients of the Chebyshev polynomials serve as input features for our symbolic discovery framework.

2. Second-order Elliptic Partial Differential Equation We consider general two-dimensional second-order elliptic partial differential equations, which are frequently described by the following generic form (Evans, 2022; Bers et al., 1964):
Lu ≡ a11uxx + a12uxy + a22uyy + a1ux + a2uy + a0u = f, 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 where a0, a1, a2, a11, a12, a22 are constants, and f represents the source term, depending on x, y. The variables u, ux, uy are the dependent variable and its partial derivatives. The equation is classified as elliptic if 4a11a22 > a212.

In our experiments, a11, a22, a1, a2, a0 are uniformly sampled within the range (−1, 1), while the coupling term a12 is sampled within (−0.01, 0.01). We then select equations that satisfy the elliptic condition to form our dataset. Similar to the approach with the Darcy flow problem, we convert the PDE into a system of linear equations using the central difference scheme of FDM. The coefficients a0, a1, a2, a11, a12, a22 serve as input features for our symbolic discovery framework. When discussing SSOR preconditioning, we set a1 and a2 to zero to ensure the resulting matrix remains symmetric.

## 3. Biharmonic Equation

We consider the biharmonic equation, a fourth-order elliptic equation, defined on a domain Ω ⊂ R
2. The equation is expressed as follows (Ciarlet & Raviart, 1974; Glowinski & Pironneau, 1979; Barrata et al., 2023):

$$\nabla^{4}u=f\quad\mathrm{in}\;\Omega=[0,a]\times[0,b],$$

where ∇4 ≡ ∇2∇2represents the biharmonic operator and f = 4.0π 4sin(πx) sin(πy) is the prescribed source term.

In our experiments, we construct the dataset by varying the solution domain Ω = [0, a] × [0, b]. We utilize the discontinuous Galerkin finite element method from the FEniCS library to transform this problem into a system of linear equations (Barrata et al., 2023). The parameters *a, b* of the domain serve as input features for our symbolic discovery framework.

## 4. Poisson Equation

We consider a two-dimensional Poisson equation, which can be described by the following equation (Wang et al., 2024; Hsieh et al., 2019; Zhang et al., 2022):
∇2u = f in Ω = [0, 1]2.

Physical Contexts in which the Poisson Equation Appears: 1. Electrostatics; 2. Gravitation; 3. Fluid Dynamics. In our experiments, we address the Poisson equation within a square domain , where both the boundary conditions on all four sides and the source term f on the equation's left-hand side are generated using third-order truncated Chebyshev polynomials. The finite difference method with a central difference scheme is employed to discretize the equation into a linear system. The Chebyshev coefficients serve as parameters for our symbolic discovery framework (Driscoll et al., 2014). 5. Thermal Problem

## D.2. Environment

To ensure consistency in our evaluations, all comparative experiments were conducted under uniform computing environments. Specifically, the environments used are detailed as follows: 1. Environment (Env1):
- Platform: Windows11 version 22631.4169, WSL - Operating System: Ubuntu 22.04.3 - CPU Processor: AMD Ryzen 9 5900HX with Radeon Graphics CPU, clocked at 3.30GHz 2. Environment (Env2):
- Platform & Operating System: Ubuntu 18.04.4 LTS - CPU Processor: Intel(R) Xeon(R) Gold 6246R CPU at 3.40GHz - GPU Processor: GeForce RTX 3090 24GB
- Library: CUDA Version 11.3 Speed tests for solving linear systems were performed in Env 1, while all training related to symbolic discovery was conducted in Env 2.

## D.3. Training Data Generation

825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 where T is the temperature. We examine the steady-state thermal equation in thermodynamics. As with the previous equation, we still solve this equation in the square domain. The boundary temperatures on the left and right boundaries are determined by random values ranging from -100 to 0 and 0 to 100, respectively. The top and bottom boundary temperature functions are generated by third-order truncated Chebyshev polynomials. The boundary temperature and the coefficients of the Chebyshev polynomials serve as parameters for our symbolic discovery framework. We consider a two-dimensional thermal steady state equation, which can be described by the following equation (Wang et al., 2024; Sharma et al., 2018; Koric & Abueidda, 2023):

$${\frac{\partial^{2}T}{\partial x^{2}}}+{\frac{\partial^{2}T}{\partial y^{2}}}=0,$$

We employed an adaptive grid search to generate the training dataset. Initially, we traversed a coarse grid, sampling every 0.05, and from this dataset, we selected the three points with the smallest values. Subsequently, we conducted a finer grid search around these points, sampling every 0.001, to identify the point with the minimum value, which we designated as our optimal parameter. Particularly, after experimental validation confirmed the dataset's convexity, we utilized a binary search sampling method for a dataset derived from the second-order elliptic equation's SOR preconditioning. Starting with points at 0.0, 1.0, and 2.0, we compared these values. If the value at 0.0 was lowest, we computed at 0.5; if at 2.0, then at 1.5; and if at 1.0, then at both 0.5 and 1.5. This process was repeated until achieving a minimum point with a precision of 0.001. For SOR preconditioning, we evaluated second-order elliptic equations, Darcy flow equations, and biharmonic equations, with solution time as the metric for optimal preprocessing parameters, achieved by minimizing solution time using the previously described grid method. In SSOR preconditioning, applied to second-order elliptic and Darcy flow equations, we utilized a hybrid metric that combined normalized computation time and iteration counts, aiming to simultaneously optimize both iteration counts and solution times. For AMG preconditioning, also examined with second-order elliptic and Darcy flow equations, we used the condition number of the preconditioned matrix as the metric, where a lower value indicates better performance.

## D.4. Parameters Of Symmap

Experimental Setup. SymMAP is implemented using the LSTM architecture with one layer and 32 units. More details about the hyperparameters are provided in Table 7.

Table 7: Hyperparameters of SymMAP (Default Model)

| Hyperparameter                   | Value     |
|----------------------------------|-----------|
| Number of LSTM layers            | 1         |
| Number of LSTM units             | 32        |
| Number of training samples       | 2,000,000 |
| Batch size                       | 1,000     |
| Risk factor ε                    | 0.05      |
| Minimal expression length        | 4         |
| Maximal expression length        | 64        |
| Learning rate                    | 0.0005    |
| Weight of entropy regularization | 0.03      |

Restricting searching space. We employ specific constraints within our framework to streamline the exploration of expression spaces effectively and ensure they remain within practical and manageable bounds:
1. **Bounds on expression length.** To strike a balance between complexity and manageability, we set boundaries for expression lengths: a minimum of 4 and a maximum of 64 characters. This ensures that expressions are neither overly trivial nor excessively complicated.

2. **Constant combination.** We restrict expressions such that the operands of any binary operator are not both constants.

This is out of the simple intuition that, if both operands are constants, the combination of the two can be precomputed and replaced with a single constant.

3. **Inverse operator exclusion.** We preclude unary operators from having their inverses as children to avoid redundant computations and meaningless expressions, such as in log(exp(x)).

4. **Trigonometric Constraints.** Expressions involving trigonometric operators should not include descendants within their formulation. For instance, sin(x + cos(x)) is restricted because it combines trigonometric operators in a way that is uncommon in scientific contexts.

880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 D.5. Computational Time for Related Algorithms
- **Dataset Generation Time:**
- Darcy Flow Problem: 40 hours - Second-order Elliptic Partial Differential Equation: 40 hours - Biharmonic Equation: 100 hours - Poisson Equation: 6 hours - Thermal Problem: 6 hours
- **SymMAP Execution Time:** For each run, 1000 iterations are performed.

- Without polynomials in the Token Library: approximately 800 seconds. - With polynomials in the Token Library: approximately 2600 seconds.

## E. Experimental Data And Supplementary Experiments

E.1. Symbolic Expressions from Main Experiments This section documents some of the learned expressions from the main experiments, corresponding to "SymMaP1" in Tables 1, 2, and 3.

- Second-order elliptic PDE problem, AMG preconditioning:

$$\frac{x_{1}x_{3}+1.0}{x_{1}+7.0}$$
$$-\;61$$

Parameter meanings: x1-x6 represent the coefficients a11, a12, a22, a1, a2, and a0 in the second-order elliptic equation.

- Second-order elliptic PDE problem, SSOR preconditioning:

$$(2x_2-\frac{1}{4x_1+2x_3})(-0.21785x_2^2)$$ $$-0.200\sqrt{2x_1^2-x_3^2}$$
$$-\frac{}{})(-0.21785x_{1}^{3}-63.6118x_{1}^{2}x_{2}+0.206541x_{1}^{2}x_{3}-0.235667x_{1}^{2}x_{4}$$
$\frac{1}{11}$
+ 0.269472x 2 1 − 967.517x1x 2 2 − 61.2291x1x2x3 + 1.*68205*x1x2x4 + 5.07925x1x2
− 0.0221322x1x 2 3 − 0.454257x1x3x4 − 0.0693756x1x3 + 0.411528x1x 2 4 + 0.0311608x1x4
− 7.53439x1 + 9506.4x 3 2 − 468.735x 2 2x3 − 154.885x 2 2x4 + 410.223x 2 2 − 25.5913x2x 2 3
+ 7.92627x2x3x4 + 5.30828x2x3 + 3.82512x2x 24 − 7.0487x2x4 − 0.612507x2 − 0.180432x 33
− 0.0462734x 23x4 + 0.310649x 23 + 0.257121x3x 24 − 0.0962336x3x4 − 3.86012x3 + 0.13906x 34
− 0.389893x 24 + 0.3144x4 − 0.0111835)

$\mathbf{a}$
Parameter meanings: x1-x4 represent the coefficients a11, a12, a22, and a0 in the second-order elliptic equation.

- Darcy flow problem, SSOR preconditioning:

$$893x_{4}^{2}+0.3144x$$
$$\frac{1.0}{x_{1}(x_{14}+x_{4})+1.0}$$

Parameter meanings: x1-x16 represent the 16 coefficients of a second-order truncated Chebyshev polynomial in two dimensions, ordered as follows: 1, x, x 2, x 3, y, xy, x 2y, x 3y, y 2, xy2, x 2y 2, x 3y 2, y 3, xy3, x 2y 3, and x 3y 2.

- Darcy flow problem, AMG preconditioning:
935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 Parameter meanings: x1-x16 represent the 16 coefficients as described above.

- Biharmonic Equation, SOR preconditioning:
Parameter meanings: x1 and x2 represent the length and width of the equation's boundary, respectively.

- Biharmonic Equation, AMG preconditioning:

$$1.0+{\frac{1.0}{3.0+{\frac{1.0x_{1}+1.0}{x_{2}}}}}+{\frac{1.0}{x_{2}}}$$
$$\left({\frac{1.0x_{2}}{-4x_{2}-1.0+{\frac{1.0}{x_{2}}}}}+1.0\right)^{4}$$

Parameter meanings: x1 and x2 represent the length and width of the equation's boundary, respectively.

$$1.0+{\frac{1.0}{1.0+{\frac{1.0}{x_{16}+x_{3}+x_{9}^{2}+2.0}}}}$$

- Poisson Equation, SOR preconditioning:
Parameter meanings: x1-x8 represent the coefficients of two second-order truncated Chebyshev polynomials for the boundary functions.

- Poisson Equation, SSOR preconditioning:
1.15024107160485 0.106506978919201x 2 2 + 11/16 Parameter meanings: x1-x8 represent the coefficients of two second-order truncated Chebyshev polynomials for the boundary functions.

- Poisson Equation, AMG preconditioning:
$$1.0$$
Parameter meanings: x1-x8 represent the coefficients of two second-order truncated Chebyshev polynomials for the
boundary functions.
$\overline{x_8^2+7.0}$. 
- Thermal problem, SOR preconditioning:

$$\exp\left({\frac{0.778800783071405}{(1-x_{6})^{1/4}}}\right)^{1/4}$$

Parameter meanings: x1-x4 represent the coefficients of the Chebyshev polynomial for the boundary temperature function on the upper and lower boundaries, while x5 and x6 represent the coefficients for the boundary temperature function on the left and right boundaries.

- Thermal problem, SSOR preconditioning:

$$1.0-{\frac{1.0}{\log\left(4.0(1-0.5x_{6})^{2}\right)}}$$

Parameter meanings: x1-x4 represent the coefficients of the Chebyshev polynomial for the boundary temperature function on the upper and lower boundaries, while x5 and x6 represent the coefficients for the boundary temperature function on the left and right boundaries.

- Thermal problem, AMG preconditioning:

## 1.0

2.71828182845905 exp(0.135335283236613x6) + 8.15484548537714 Parameter meanings: x1-x4 represent the coefficients of the Chebyshev polynomial for the boundary temperature function on the upper and lower boundaries, while x5 and x6 represent the coefficients for the boundary temperature function on the left and right boundaries.

## E.2. Interpretable Analysis Details

990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037

## 1038

1039 1040 1041 1042 1043 1044 As shown in Table 8, the variables are defined as follows: in the first row, x1 and x2 represent the size of the boundary for PDE solutions; in the second row, x2 represents the coefficient of a second-order coupling term; in the third row, x4 is the coefficient of the fourth x-term multiplied by the first y-term in a two-dimensional Chebyshev polynomial; in the fourth row, x2 again denotes the coefficient of a second-order coupling term; in the fifth row, x1x3 signifies the coefficient of a second-order non-coupling term.

$$\sqrt{\exp\left({\frac{1.0}{x_{3}+\exp(2\exp(x_{1}^{2}))}}\right)}$$

## E.3. Analysis Of Hyperparameters

The performance of SymMaP is primarily influenced by the learning rate of the RNN, batch size, and dataset size. We conducted experiments to study the impact of these hyperparameters.

## Symbolic Learning Rnn Parameters:

Table 9: Performance comparison of SymMaP under various symbolic learning RNN parameters (lower condition numbers are preferable). The experiment focuses on optimizing AMG preconditioning coefficients in the Darcy Flow dataset.

| Learning Rate   | Batch Size   | Condition number   | Training time(s)   |
|-----------------|--------------|--------------------|--------------------|
| 500             | 6780         | 1173.09            |                    |
| 0.01            | 1000         | 5168               | 863.51             |
| 2000            | 6898         | 522.80             |                    |
| 500             | 5935         | 1104.16            |                    |
| 0.001           | 1000         | 11774              | 676.40             |
| 2000            | 5935         | 505.85             |                    |
| 500             | 4718         | 1026.45            |                    |
| 0.0005          | 1000         | 5935               | 703.17             |
| 2000            | 5935         | 549.36             |                    |
| 500             | 12228        | 1324.00            |                    |
| 0.0001          | 1000         | 7508               | 837.18             |
| 2000            | 6884         | 603.62             |                    |

1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081

1082

1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099

Results in Table 9 indicate that an appropriate combination of RNN learning rate and batch size can enhance performance.

## Dataset Size:

Table 10: Performance comparison of SymMaP across varying dataset sizes (lower condition numbers indicate better performance). The experiment evaluates the optimization of AMG preconditioning coefficients for the Darcy Flow dataset.

| Dataset size   | Condition number   | Training time (s)   |
|----------------|--------------------|---------------------|
| 10             | 7032               | 669.68              |
| 50             | 6980               | 737.80              |
| 100            | 4892               | 812.02              |
| 500            | 3811               | 699.54              |
| 1000           | 5345               | 703.17              |

Table 10 demonstrates that increasing the dataset size enhances the performance of symbolic expressions learned by SymMaP,
as expected.

| Precondition   | Dataset      | Symbolic expression                   |
|----------------|--------------|---------------------------------------|
| SOR            | Biharmonic   | 1.0 + 1.0/(4.0 + 1.0/x2) + 1.0/x1     |
| SOR            | Elliptic PDE | 1.0 + 1.0/(x2 + 1.0 + 1.0/(x2 + 4.0)) |
| SOR            | Darcy Flow   | 1.0 + 1.0/(x4 + 1.0)                  |
| SSOR           | Elliptic PDE | 1.0 + 1.0/(x2 + 1.2)                  |
| AMG            | Elliptic PDE | (x1x3 + 1)/7                          |