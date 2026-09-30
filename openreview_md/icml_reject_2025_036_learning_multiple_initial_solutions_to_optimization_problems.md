# Learning Multiple Initial Solutions To Optimization Problems

## Anonymous Authors1 Abstract

Sequentially solving similar optimization problems under strict runtime constraints is essential for many applications, such as robot control, autonomous driving, and portfolio management. The performance of local optimization methods in these settings is sensitive to the initial solution: poor initialization can lead to slow convergence or suboptimal solutions. To address this challenge, we propose learning to predict *multiple* diverse initial solutions given parameters that define the problem instance. We introduce two strategies for utilizing multiple initial solutions: (i) a singleoptimizer approach, where the most promising initial solution is chosen using a selection function, and (ii) a multiple-optimizers approach, where several optimizers, potentially run in parallel, are each initialized with a different solution, with the best solution chosen afterward. Notably, by including a default initialization among predicted ones, the cost of the final output is guaranteed to be equal or lower than with the default initialization. We validate our method on three optimal control benchmark tasks: cart-pole, reacher, and autonomous driving, using different optimizers: DDP, MPPI, and iLQR. We find significant and consistent improvement with our method across all evaluation settings and demonstrate that it efficiently scales with the number of initial solutions required.

## 1. Introduction

Many applications, ranging from trajectory optimization in robotics and autonomous driving to portfolio management in finance, require solving similar optimization problems sequentially under tight runtime constraints (Paden et al., 2016; Ye et al., 2020; Mugel et al., 2022). The performance 1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054 of local optimizers in these contexts is often highly sensitive to the initial solution provided, where poor initialization can result in suboptimal solutions or failure to converge within the allowed time (Michalska & Mayne, 1993; Scokaert et al.,
1999). The ability to consistently generate high-quality initial solutions is, therefore, essential for ensuring both performance and safety guarantees. Conventional methods for selecting these initial solutions typically rely on heuristics or warm-starting, where the solution from a previously solved, related problem instance is reused. More recently, learning-based solutions have also been proposed, where neural networks are used to predict an initial solution. However, in more challenging cases, where the optimization landscape is highly non-convex or when consecutive problem instances rapidly change, predicting a single good initial solution is inherently difficult. To this end, we propose Learning Multiple Initial Solutions (MISO) (Figure 1), in which we train a neural network to predict *multiple* initial solutions. Our approach facilitates two key settings: (i) a single-optimizer method, where a selection function leverages prior knowledge of the problem instance to identify the most promising initial solution, which is then supplied to the optimizer; and (ii) a multipleoptimizers method, where multiple initial solutions are generated jointly to support the execution of several optimizers, potentially running in parallel, with the best solution chosen afterward. More specifically, our neural network receives a parameter vector that characterizes the problem instance and outputs K candidate initial solutions. The network is trained on a dataset of problem instances paired with (near-)optimal solutions and is evaluated on previously unseen instances. Crucially, the network is designed not only to predict *good* initial solutions—those close to the optimal—but also to ensure that these solutions are sufficiently diverse, potentially spanning all underlying modes of the problem in hand. To actively encourage this multimodality, we implement training strategies such as a winner-takes-all loss that penalizes only the candidate with the lowest loss, a dispersion-based loss term to promote dispersion among solutions, and a combination of both. Notably, any existing initialization strategy can be combined with MISO, by simply including the existing initial solution 1

Learning Multiple Initial Solutions to Optimization Problems 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 Pri or work Neural Network Problem Parameters Initial Solution **Optimizer**
Solution This work Sing le O
pti mi zer Best Initial Solution Problem Parameters Neural Network Optimizer **Solution**
Candidate Evaluator Mult. Initial Solutions M
ult iple Op ti mi ze rs Best Solution Mult. Initial Solutions Multiple Optimizers Problem Parameters Neural Network Multiple Solutions Candidate Evaluator
among the predicted ones; and by design, MISO is guaranteed to be equal or better than the default initialization.

to prevent mode collapse and ensure that the predicted solutions remain multimodal.

We evaluate MISO across three distinct local optimization algorithms applied to separate robot control tasks: First-order Box Differential Dynamic Programming (DDP), which utilizes first-order linearization for the cart-pole swing-up task; Model Predictive Path Integral (MPPI) control, a samplingbased method, for the reacher task; and the Iterative Linear Quadratic Regulator (iLQR), a trajectory optimization algorithm, for an autonomous driving task. Our results show that MISO significantly outperforms existing initialization methods that rely on heuristics, learn to predict a single initial solution or use ensembles of independently learned models. In summary, our key contributions are as follows:
1. We present a novel framework for predicting *multiple* initial solutions for optimizers.

3. We design and implement specific training objectives 4. We apply our framework to three distinct sequential optimization tasks and perform extensive evaluation.

## 2. Related Work

Learning for optimization. Advancements in machine learning have introduced numerous learning-based approaches to optimization problems (Sun et al., 2019). Early work by Gregor & LeCun (2010) replaced components of classical convex optimization algorithms with neural networks. More recent works aim to replace optimization methods entirely with end-to-end neural networks (OpenAI et al., 2020; Mirowski et al., 2017) or generate new optimization algorithms (Chen et al., 2022b) for specific classes of problems. Other works enhance optimization-based control algorithms (Sacks & Boots, 2022), learn constraints (Fajemisin et al., 2024), or learn objective functions and system dynamics (Lenz et al., 2015; Wahlstrom et al. ¨ , 2015; Tamar et al., 2017; Hafner et al., 2019; Nagabandi et al., 2018; Xiao et al., 2022).

Learning initial solutions. Previous studies have proposed heuristic approaches to generate initial solutions for optimizers (Johnson et al., 2015; Marcucci & Tedrake, 2. We introduce two distinct strategies for utilizing the predicted initial solutions: (i) *single-optimizer*, where the most promising solution is chosen based on a selection function, and (ii) *multiple-optimizers*, where multiple optimizers are initialized, potentially in parallel, with the best solution chosen afterward.

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 2020). More recently, learning-based methods for initializing optimizers have gained attention in various fields, aiming to enhance both computational efficiency and resulting solutions quality. In mixed-integer programming, neural networks have enhanced solver performance by predicting variable assignments (Nair et al., 2020), branching decisions (Sonnerat et al., 2021), and integer variables (Bertsimas & Stellato, 2021). Baker (2019) employed Random Forests to predict solutions for AC optimal power flow problems. Kang et al. (2024) utilized nearest neighbor search to warm-start tight convex relaxations in nonconvex trajectory optimization problems. In robot control, neural networks were used to predict initializations for trajectory optimizers or Model Predictive Control (MPC) (Chen et al., 2022a; Wang & Ba, 2019; Lembono et al., 2020). An exciting line of recent work developed *differentiable* optimization algorithms, which allow jointly learning objectives, constraints, and initializations by backpropagating through the optimization process (Amos et al., 2018; East et al., 2019; Karkus et al., 2022; Sambharya et al., 2023). In contrast, we learn multiple initializations instead of one, and we do so without strong assumptions about the task or the optimizer.

Notably, Bouzidi et al. (2023) used multiple initializations by repurposing a motion prediction model and Bezier curve ´ fitting for a downstream MPC; however, this approach is specifically tailored for autonomous driving, incorporating a dedicated motion prediction module. Parallel optimizers. Leveraging parallelism has a long history in optimization research (Betts & Huffman, 1991). With recent advances in parallel computing hardware, such as GPUs, methods that execute multiple optimizers in parallel have also emerged. For example, Sundaralingam et al. (2023) introduced cuRobo, a GPU-accelerated method combining L-BFGS and particle-based optimization for robotic manipulators. Similarly, Huang et al. (2024) utilized massive parallel GPU computation for efficient inverse kinematics and trajectory optimization. de Groot et al. (2024) proposed a topology-driven method that plans for multiple evasive maneuvers in parallel. Barcelos et al. (2024) focused on initializing parallel optimizers through rough paths. However, these works have not utilized learning. Lembono et al. (2020) explored learning-based strategies for initializing trajectory optimizers based on a database of previous solutions and ensemble-learned models, particularly in manipulation and humanoid control tasks. In contrast, we propose a single neural network to generate multiple initializations, which, as shown in our experiments, significantly outperforms the ensemble-based approach.

## 3. Initializing Optimizers

Problem setup. In the most general form, we need to solve instances of a parameterized optimization problem,

$$\begin{array}{l l}{{\mathbf{x}^{\star}(\psi)=\arg\operatorname*{min}_{\mathbf{x}}J(\mathbf{x};\psi)}}&{{\mathrm{s.t.}}}&{{\mathbf{g}(\mathbf{x};\psi)\leq\mathbf{0},}}\\ {{}}&{{\mathbf{h}(\mathbf{x};\psi)=\mathbf{0}.}}\end{array}\tag{1}$$

where x ∈ R
n is the variable vector to be optimized, J is the objective function, g and h are collections of inequality and equality constraints, and ψ ∈ R
m is a parameter vector that defines the problem instance, e.g., parameters of the objective function and constraints that differ across problem instances. A local optimization algorithm, Opt, attempts to find an optimum of J, namely,

$${\hat{\mathbf{x}}}^{\star}=\mathbf{Opt}(J,\psi,t_{\mathrm{lim}};\mathbf{x}^{\mathrm{init}}),$$

where x init is initial solution provided to the optimizer, and tlim is the runtime limit.

Heuristic methods. A common choice of the initial solution, xinit, is the solution to a previously solved similar problem instance, referred to as a *warm-start*. For example, in optimal control the warm-start is typically the solution from the previous timestep, shifted and padded with zeros, x w.s.:= {{x cand t+k
}
H−2 k=1 
, 0} (Otta et al., 2015). This heuristic often works well in practice, however, it can struggle when large changes in the problem instance, ψ, occur between consecutive time steps, leading to significant shifts in the optimal solution. For example, in autonomous driving, abrupt events like a traffic light switch or the sudden appearance of a pedestrian might drastically alter the reference trajectory or constraints. In such cases, the previous solution becomes a poor initialization, and the optimizer may fail to find a good solution within the allocated time frame.

## 4. Learning Multiple Initial **Solutions**

The main idea of MISO is to train a single neural network to predict multiple initial solutions to an optimization problem, such that the initial solutions cover promising regions of the optimization landscape, eventually allowing a local optimizer to find a solution close to the global optimum. The key questions are then how to design a multi-output predictor; how to utilize multiple initial solutions in existing optimizers; and how to train the predictor to output a diverse set of initial solutions. In the following, we discuss our proposed solutions to these questions, illustrate the need for multimodality with a toy example, and discuss applications to optimal control.

## 4.1. Multi-Output Predictor

Our multi-output predictor is a standard Transformer model–
chosen for its simplicity and clarity (see Appendix A.13

$$\{{\hat{\mathbf{x}}}_{k}^{\mathrm{init}}\}_{k=1}^{K}=f(\psi;\theta),$$

where θ are the learned parameters of the network. We train the network on a dataset of problem instances and their corresponding (near-)optimal solutions, {(ψi, x
⋆ i
)}
n i=1. Such dataset can be generated offline, for example, by running a slow yet globally optimal solver, or allowing the same local optimizer to run with longer time limits, potentially many times from different initial solutions.

## 4.2. Optimization With Multiple Initial Solutions

We propose two distinct settings to leverage multiple initial solutions: *single-optimizer* and *multiple-optimizers*. The resulting frameworks are illustrated in Fig. 1. Single optimizer. In the single-optimizer setting we run a single instance of the optimizer with the most promising initial solution, xˆ
⋆ = Opt(J, ψ, tlimit; xˆ
init). We introduce a selection function, Λ, which, given a set of candidate solutions and the problem instance ψ, returns the most promising candidate, xˆ
init = Λ({xˆ
init k}
K
k=1, ψ). A reasonable choice for Λ used in our experiments is selecting the candidate that minimizes the objective function the optimizer aims to minimize, i.e., Λ := arg mink J(xˆ
init k; ψ).

However, other criteria–such as robustness, constraint satisfaction, or domain-specific requirements–may be more appropriate in certain scenarios; see Appendix A.10 for details and examples.

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 Multiple optimizers. In the multiple-optimizers setting, we assume multiple instances of the optimizer can be executed in parallel. We then initialize each optimizer with a different initial solution, x
⋆
k =
Optk
(J, ψ, tlimit; xˆ
init k), k ∈ {1*, . . . , K*}. To select a single solution from the outputs of the optimizers, we can use the same selection function Λ, as in the previous case, e.g., the solution that minimizes the objective function. Guarantees. Our framework can be trivially generalized to allow a different number of optimizers and initial solution predictions, as well as using a heterogeneous set of optimization methods. To maintain performance guarantees, one may include traditional initialization methods, such as warm-start heuristics, as part of the set of initializations. MISO is guaranteed to improve over the existing default strategy by design. In the single-optimizer setting the best initial solution is always equal or better than the default according to the selection function Λ; and in the multipleoptimizer setting the final solution is equal or better than using only the default initialization.

## 4.3. Training Strategies

for a discussion on alternative architectures). It takes the problem instance, ψ, as input and outputs K initial solutions for the optimization problem, The ultimate goal is to predict multiple initial solutions so that the downstream optimizer can find a solution close to the global optima, i.e., J(xˆ
⋆; ψ) ≈ J(x
⋆; ψ). Training a neural network directly for this objective is not feasible in general. Instead, we propose proxy training objectives that combine two terms: a regression term that encourages outputs to be close to the global optimum, e.g.,
Lreg(xˆ
init k, x
⋆) = ∥xˆinit − x
⋆∥, where *∥ · ∥* is a distance metric; along with a *diversity* term that promotes outputs being different from each other, thereby covering various regions of the solution space. An illustrative example is in Sect. 4.4. In the following, we present three simple training strategies promoting diversity and preventing mode collapse. We discuss alternative formulations, with probabilistic modeling and reinforcement learning, in Sect. 7. Pairwise distance loss. A simple method to encourage the model's outputs to differ from each other is to penalize the pairwise distance between all outputs. The overall loss combines this dispersion-promoting term with the regression loss,

$${\mathcal{L}}_{\mathrm{PD}}={\frac{1}{K}}\sum_{k=1}^{K}{\mathcal{L}}_{\mathrm{reg}}({\hat{\mathbf{x}}}_{k}^{\mathrm{init}},\mathbf{x}^{\star})+\alpha_{K}{\frac{1}{K}}\sum_{k=1}^{K}{\mathcal{L}}_{\mathrm{PD},k}({\hat{\mathbf{x}}}_{k}^{\mathrm{init}}),$$
$${\mathcal{L}}_{\mathrm{PD},k}={\frac{1}{K-1}}\sum_{\begin{array}{l}{k^{\prime}=1}\\ {k^{\prime}\neq k}\end{array}}^{K}\|{\hat{\mathbf{x}}}_{k}^{\mathrm{init}}-{\hat{\mathbf{x}}}_{k^{\prime}}^{\mathrm{init}}\|,$$

where αK is a hyperparameter that balances the trade-off between accuracy and dispersion.

Winner-takes-all loss. A more interesting way to encourage multimodality is to select the best-predicted output at training time and only minimize the regression loss for this specific prediction,

$${\mathcal{L}}_{\mathrm{WTA}}=\operatorname*{min}_{k}\{{\mathcal{L}}_{\mathrm{reg}}({\hat{\mathbf{x}}}_{k}^{\mathrm{init}},\mathbf{x}^{*})\}.$$
k
Intuitively, the model only needs one of its outputs to be close to the ground truth, while the other predictions are not penalized for deviating, potentially aligning with different regions of the underlying distribution. Similar losses have been used, e.g., in multiple-choice learning (Guzman-Rivera et al., 2012). One advantage of this approach is that it is hyperparameter-free. Mixture loss. Lastly, we consider a combination of the previous two approaches to potentially enhance performance, as it provides some measure of dispersion we can tune,

$${\mathcal{L}}_{\mathrm{MIX}}=\operatorname*{min}_{k}\left\{{\mathcal{L}}_{\mathrm{reg}}({\hat{\mathbf{x}}}_{k}^{\mathrm{init}},\mathbf{x}^{\star})+\alpha_{K}\Phi\left({\mathcal{L}}_{\mathrm{PD},k}({\hat{\mathbf{x}}}_{k}^{\mathrm{init}})\right)\right\},$$

4 here, Φ is an upper-bounded function, such as min or tanh, designed to limit the contribution of the pairwise distance term. Beyond the losses above, MISO could be integrated with other training paradigms, such as reinforcement learning or probabilistic modeling. We discuss these options in Sect. 7 but differ investigation to future work.

220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 To illustrate the advantage of using a single model with multimodal outputs compared to regression models or ensembles of regressors, we examine a straightforward onedimensional optimization problem aimed at minimizing the cost function c(x) shown in Fig. 2 (top). The function features two global minima, denoted as A and C, with a local minimum located between them at B. Applying our learning framework to this simple problem, the dataset of optimal solutions includes instances of A and C. A single-output regression model has no means to distinguish the two modes and inevitably learns to predict the mean of examples in the dataset, somewhere near B. Consequently, the local optimizer is likely to converge to the suboptimal local minimum at B. Constructing an ensemble of such models to generate multiple initial solutions does not mitigate this issue, as each ensemble member tends to be biased toward the mean of the two modes near B. We implemented the optimization problem and showed the predictions for different training strategies in Fig. 2 (bottom). Details are in Appendix A.5. Indeed, an ensemble of singleoutput predictors fails to predict a global optimum, while our multi-output predictor succeeds with winner-takes-all and mixture losses. While the problem considered here is purposefully simplistic, the existence of local minima is the key challenge in most optimization problems.

## 4.5. Application To Optimal Control

MISO is applicable to a broad class of sequential optimization problems; however, for the sake of evaluation, we focus on optimal control problems. Optimal control has a wide range of applications, e.g., in robotics, autonomous driving, and many other domains with strict runtime requirements, and due to the complexity induced by constraints and non-convex costs, local optimization algorithms are highly sensitive to the initial solution. In optimal control the optimization variable x represents a trajectory defined as a sequence of states and control inputs over discrete time steps: τ = {st,ut}t=1:H. Here, st ∈ S and ut ∈ U denote the state and control input at time step t ∈ Z
+, and H ∈ Z
+ is the optimization horizon. The constraints involve adhering to the system dynamics fd(st+1, st,ut) = 0, starting from an initial state s0 = scurr, where scurr represents the system's current state.

The problem instance parameters ψ encompass the initial state s0, and other domain-specific variables that parameterize the objective function or constraints, such as target states, reference trajectories, obstacle positions, friction coefficients, temperature, etc. A specific property of optimal control problems is that the relationship between optimization variables, states st and controls ut, are defined by the dynamics constraint fd; and the initial state s0 is given. Therefore, a sequence of controls uniquely defines an (initial) solution. We can leverage this property by learning to predict only a sequence of controls instead of the full optimization variable of statecontrol sequences. Further, one can define the training loss over either control, state, or state-control sequences and backpropagate gradients through the dynamics constraint as long as it is differentiable. In our experiments, we use state-control loss by default as we found it to improve both our and baseline learning methods. In Appendix A.7, we show that our conclusions hold with control-only loss as well.

(a) Cart-pole (b) Reacher (c) Driving

## 5. Experimental Setup

Tasks. We evaluated our method on the three robot control benchmark tasks shown in Fig. 3, each employing a distinct local optimization algorithm. **Cart-pole.** This task involves balancing a pole upright while moving a cart toward a randomly selected target position (Barto et al., 1983), using a first-order box Differential Dynamic Programming (DDP) optimizer (Amos et al., 2018). **Reacher.** In this task, a two-link planar robotic arm needs to reach a target placed at a random positon (Tassa et al., 2018), using a Model Predictive Path Integral (MPPI) optimizer (Williams et al., 2015). **Autonomous Driving.** Based on the nuPlan benchmark (Caesar et al., 2021), this task focuses on trajectory tracking in complex urban environments by following a reference trajectory generated by a Predictive Driver Model (PDM) planner (Dauner et al., 2023), using the Iterative Linear Quadratic Regulator (iLQR) optimizer (Li & Todorov, 2004). Further details are in Appendix A.1 and Appendix A.2. Baselines. We compare MISO to a range of alternative methods to provide single or multiple initial solutions. For a single initial solution, we considered: **Warm-start**, the default method that uses the optimizer output from the last problem instance; **Regression**, a single-output regression model (the K = 1 version of MISO); **Oracle Proxy**, optimization with unlimited runtime, which we also used to generate our training data. For methods that generate multiple initial solutions, we considered: Warm-start with perturbations, which extends the warm-start approach by adding Gaussian noise to the optimizer output from the last problem instance; **Regression with perturbations**, where Gaussian noise is introduced to the predictions of the single-output regression model; **Multi-output regression**, a naive multioutput regression model without a diversity-promoting objective; and **Ensemble**, which trains multiple single-output neural networks with different random initializations. Finally, we assessed variants of our proposed method with the different training losses discussed in Sect. 4.3: **pairwise** distance, **winner-takes-all**, and mix. Evaluation settings. We employ two evaluation modes. (i) **One-off**, where the optimization task is treated as an isolated problem with the objective of finding the minimum of a given function. This mode serves as the default configuration for training neural networks, where data is replayed to the model, and the optimizer's solution is recorded but not executed. Methods are assessed by the mean cost of the optimizer's output over problem instances. (ii) Sequential, which involves solving a series of related optimization problems, executing each proposed solution, and starting the subsequent optimization from the resulting state. This setting simulates real-world conditions where the optimizer continuously interacts with a dynamic environment in a closed loop. Astute readers may notice parallels to the open-loop/closed-loop paradigms in control theory; these connections are discussed in greater depth in Appendix A.11 and Appendix A.12 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 We evaluate performance by taking the mean cost over problems in a sequence, and then the mean over sequences. To account for the additional time required to predict initial solutions, we assumed that all models perform inference in under 0.85ms, which was the case for all methods on both CPU and GPU, except for the ensemble (see Appendix A.6). In the autonomous driving task, we then reduced the runtime allocated to the optimizer accordingly. Implementation details. To generate the training data, we first create a set of problem instances by sequentially executing the optimizer initialized with the default warm-start strategy. The problem instances are then fed again to an "oracle" version of the optimizer with a significantly increased runtime limit, and the resulting solutions are recorded. After training, evaluation is done on a separate unseen set of problem instances. All experiments are conducted on an Intel Core i9-13900KF CPU and an NVIDIA RTX 4090 GPU. Further implementation details, including hyperparameters and training procedures, are in Appendix A.3 and Appendix A.4.

## 6. Results

Our main results for optimization with different initial solutions are reported in Table 1 and Table 2 for single optimizer and multiple optimizers settings, respectively. Figure 4 shows the effect of the number of predicted initial solutions. Figure 5 provides qualitative results. More detailed results, including inference times, are in the Appendix. Single optimizer. In the single-optimizer setting, Table 1, we first observe that even one learned initialization outperforms heuristic solutions (regression vs. warm-start), in almost all settings, and in particular in the most challenging autonomous driving task. We then examine the impact of generating multiple initial solutions. Perturbationsbased methods show some improvement over their singleinitialization counterparts in most cases, and ensembles of independently learned models perform consistently better than single models. Finally, our proposed multi-output methods demonstrate substantial improvements over all baselines because they can learn to predict diverse multimodal initial solutions. Specifically, MISO winner-takes-all or MISO mix achieve the lowest mean costs across all tasks. Considering the pairwise distance term alone proves insufficient to ensure adequate diversity, whereas incorporating it with MISO winner-takes-all often boosts performance, yet, its effectiveness varies, which underscores the challenge of selecting optimal hyperparameters. As expected, improvements are consistently larger in the more important sequential optimization setting, where errors over time compound.

| One-Off Optimization        | Sequential Optimization   |             |             |               |             |             |               |
|-----------------------------|---------------------------|-------------|-------------|---------------|-------------|-------------|---------------|
| Method                      | K                         | Reacher     | Cart-pole   | Driving       | Reacher     | Cart-pole   | Driving       |
| Single Optimizer Warm-start | 1                         | 13.48 ±0.88 | 11.69 ±0.84 | 283.86 ±37.91 | 13.48 ±0.88 | 11.69 ±0.84 | 283.86 ±37.91 |
| Regression                  | 1                         | 13.40 ±0.88 | 11.19 ±0.80 | 74.23 ±7.69   | 19.56 ±0.52 | 6.18 ±0.47  | 70.62 ±7.38   |
| Warm-start w. perturb       | 32                        | 13.46 ±0.88 | 11.64 ±0.83 | 145.01 ±23.01 | 14.71 ±0.93 | 16.29 ±0.44 | 164.75 ±22.84 |
| Regression w. perturb       | 32                        | 13.38 ±0.88 | 11.16 ±0.80 | 67.69 ±8.01   | 15.28 ±0.58 | 5.74 ±0.47  | 66.75 ±6.56   |
| Multi-output regression     | 32                        | 13.41 ±0.88 | 11.21 ±0.80 | 70.25 ±8.75   | 18.49 ±0.55 | 6.62 ±0.45  | 78.74 ±8.99   |
| Ensemble                    | 32                        | 13.39 ±0.88 | 10.94 ±0.79 | 47.22 ±4.71   | 8.40 ±0.40  | 3.55 ±0.34  | 52.59 ±4.81   |
| MISO pairwise dist.         | 32                        | 13.41 ±0.88 | 11.22 ±0.80 | 66.06 ±7.48   | 19.20 ±0.49 | 6.07 ±0.45  | 71.90 ±7.90   |
| MISO winner-takes-all       | 32                        | 13.36 ±0.88 | 10.48 ±0.77 | 30.17 ±2.24   | 2.72 ±0.21  | 0.83 ±0.06  | 30.75 ±2.15   |
| MISO mix                    | 32                        | 12.74 ±0.86 | 10.48 ±0.77 | 33.95 ±2.39   | 2.44 ±0.20  | 0.79 ±0.04  | 33.38 ±2.21   |
| Oracle Proxy                | 1                         | 13.43 ±0.88 | 11.01 ±0.80 | 41.94 ±4.31   | 6.88 ±0.58  | 4.54 ±0.71  | 26.52 ±2.00   |

| One-Off Optimization                      | Sequential Optimization   |             |             |               |            |            |               |
|-------------------------------------------|---------------------------|-------------|-------------|---------------|------------|------------|---------------|
| Method                                    | K                         | Reacher     | Cart-pole   | Driving       | Reacher    | Cart-pole  | Driving       |
| Multiple Optimizers Warm-start w. perturb | 32                        | 13.41 ±0.88 | 10.93 ±0.79 | 155.53 ±24.33 | 5.89 ±0.50 | 6.68 ±0.59 | 162.13 ±34.10 |
| Regression w. perturb                     | 32                        | 13.34 ±0.88 | 11.12 ±0.80 | 64.88 ±6.84   | 3.53 ±0.27 | 5.36 ±0.48 | 62.07 ±6.39   |
| Multi-output regression                   | 32                        | 13.34 ±0.88 | 11.21 ±0.80 | 70.29 ±9.13   | 3.31 ±0.26 | 6.38 ±0.43 | 70.71 ±8.18   |
| Ensemble                                  | 32                        | 13.34 ±0.88 | 10.65 ±0.78 | 45.44 ±4.64   | 3.08 ±0.23 | 2.21 ±0.20 | 49.08 ±5.29   |
| MISO pairwise dist.                       | 32                        | 13.34 ±0.88 | 11.22 ±0.80 | 67.62 ±7.58   | 3.42 ±0.27 | 6.09 ±0.47 | 71.33 ±8.13   |
| MISO winner-takes-all                     | 32                        | 13.34 ±0.88 | 10.29 ±0.76 | 30.87 ±2.30   | 2.21 ±0.16 | 0.76 ±0.05 | 30.48 ±2.07   |
| MISO mix                                  | 32                        | 12.72 ±0.86 | 10.29 ±0.76 | 33.52 ±2.35   | 1.56 ±0.14 | 0.63 ±0.02 | 34.85 ±2.64   |

330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 Multiple optimizers. When considering the multipleoptimizers setting, we observe the same trend. Learningbased methods outperform heuristic ones, and multi-output approaches yield further enhancements. As expected, the use of multiple optimizers leads to consistently better results compared to the single-optimizer setting due to increased exploration of the solution space. Scaling with the number of initial solutions. Figure 4 shows that our method scales effectively and consistently with the number of predicted initial solutions K, and outperforms other approaches across varying values of K. Importantly, as K increases, the inference time for ensemble approaches grows, whereas MISO remains almost constant (see Appendix A.6). We further evaluate mode diversity in Appendix A.8, and find that, in line with our conclusions, all MISO outputs remain useful even when K increases. Performance guarantees. To evaluate the guarantees discussed in Sect. 4.2, we added the warm-start initial solution to the set of candidates of MISO winner-takes-all and MISO
mix. We observed in all problem instances the cost of the best initial solution to be equal or lower than the cost of the warm start in the single optimizer setting; and the cost of the final solution to be lower or equal than for the warm start in the multiple optimizer setting. The mean costs remained mostly similar to MISO, and improved slightly in some cases. Detailed results are in Appendix A.15.

Qualitative results. Figure 5 (left) depicts the optimizer's output trajectories with different initial solutions for the autonomous driving task. In this scenario, the high-level planner abruptly alters the reference path, which could happen, e.g., because of a newly detected pedestrian. The change in reference path makes the previous solution (warm-start) a poor initialization, and the optimizer converges to a local minimum that minimizes control effort but is far from the desired path. Regression and model ensemble also fail to predict a good initial solution. In contrast, MISO winnertakes-all adapts to this sudden reference change and closely follows the reference path. Figure 5 (right) depicts MISO's initial solutions for the cart-pole task. The different outputs capture different modes of the solution space (maintaining balance while moving, swinging leftward, and swinging rightward), showing MISO's ability to generate diverse and multimodal solutions. Summary. Overall, our methods significantly outperform the other baselines in both settings. The consistent superiority of the MISO mix and MISO winner-takes-all methods

(a) Single Optimizer (b) Multiple Optimizers
across different tasks and configurations underscores the advantages of using learning-based multi-output strategies for generating initial solutions. These findings demonstrate that promoting diversity among multiple initializations is crucial for improving optimization outcomes, especially when combined with multiple optimizers.

## 7. Conclusions And Future Work

We introduced Learning Multiple Initial Solutions (MISO), a novel framework for learning multiple diverse initial solutions that significantly enhance the reliability and efficiency of local optimization algorithms across various settings. Extensive experiments in optimal control demonstrated that our method consistently outperforms baseline approaches and scales efficiently with the number of initializations.

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 Limitations. Our approach is not without limitations. First, to train a useful model, we rely on the coverage and quality of the training data, as the method does not directly interact with the optimizer or the underlying objective function. Second, the underlying assumption of our regression loss is that initial solutions closer to the global optimum increase the likelihood of successful optimization may not hold in complex optimization landscapes with intricate constraints. Third, in highly complex optimization problems where each solution constitutes a high-dimensional and intricate structure, accurately learning initial solution candidates can become exceedingly challenging, potentially diminishing the effectiveness of our approach.

Future work. There are several promising directions for future research. To address the aforementioned limitations, one may simply incorporate the optimization objective into the model training loss, thus creating a direct link to the final optimization goal. Alternatively, using reinforcement learning (RL) to train MISO is a particularly exciting opportunity. By framing the problem in an RL context, e.g., where the reward is the negative cost of the optimizer's final solution, models would be directly trained to maximize the probability of the optimizer finding the global optima and may learn to specialize to the specific optimizer. One challenge would be computational, as RL would require running the optimizer numerous times during training. Other extensions of our approach include probabilistic modeling, e.g., Gaussian mixture models, variational autoencoders, or diffusion models; however, preventing mode collapse and promoting diversity, and inference overhead (see Appendix A.14), would remain a challenge. Future work may explore alternative selection functions, such as risk measures or criteria based on stability, robustness, exploration, or other domain-specific metrics; as well as using a heterogeneous set of parallel optimizers. Finally, we are excited about various possible applications in optimal control and beyond, where sequences of similar optimization problems need to be solved, for example, localization and mapping in robotics, financial optimization, traffic routing optimization, or even training neural networks with different initial weights, e.g., for meta-learning.

## References

Impact statement. This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

Chen, T., Chen, X., Chen, W., Heaton, H., Liu, J., Wang, Z., and Yin, W. Learning to optimize: A primer and a benchmark. *Journal of Machine Learning Research*, 23 (189):1–59, 2022b.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., and Song, S. Diffusion policy: Visuomotor policy learning via action diffusion, 2024. URL https://arxiv.org/abs/2303.04137.

Amos, B., Jimenez, I., Sacks, J., Boots, B., and Kolter, J. Z.

Differentiable MPC for end-to-end planning and control.

In *Advances in Neural Information Processing Systems*,
pp. 8299–8310, 2018.

Dauner, D., Hallgarten, M., Geiger, A., and Chitta, K. Parting with misconceptions about learning-based vehicle motion planning. In *Conference on Robot Learning*, pp. 1268–1281, 2023.

Baker, K. Learning warm-start points for ac optimal power flow. In *2019 IEEE 29th International Workshop on* Machine Learning for Signal Processing (MLSP), pp. 1– 6, 2019.

de Groot, O., Ferranti, L., Gavrila, D., and Alonso-Mora, J. Topology-driven parallel trajectory optimization in dynamic environments. *arXiv preprint arXiv:2401.06021*,
2024.

Barcelos, L., Lai, T., Oliveira, R., Borges, P., and Ramos, F. Path signatures for diversity in probabilistic trajectory optimisation. The International Journal of Robotics Research, pp. 02783649241233300, 2024.

East, S., Gallieri, M., Masci, J., Koutnik, J., and Cannon, M.

Infinite-horizon differentiable model predictive control. In *International Conference on Learning Representations*, 2019.

Barto, A. G., Sutton, R. S., and Anderson, C. W. Neuronlike adaptive elements that can solve difficult learning control problems. IEEE Transactions on Systems, Man, and Cybernetics, SMC-13(5):834–846, 1983. doi: 10.1109/ TSMC.1983.6313077.

Fajemisin, A. O., Maragno, D., and den Hertog, D. Optimization with constraint learning: A framework and survey. *European Journal of Operational Research*, 314 (1):1–14, 2024.

Bertsimas, D. and Stellato, B. The voice of optimization.

Machine Learning, 110(2):249–277, 2021.

Gregor, K. and LeCun, Y. Learning fast approximations of sparse coding. In *Proceedings of the 27th international* conference on international conference on machine learning, pp. 399–406, 2010.

Betts, J. T. and Huffman, W. P. Trajectory optimization on a parallel processor. *Journal of Guidance, Control, and* Dynamics, 14(2):431–439, 1991.

Guzman-Rivera, A., Batra, D., and Kohli, P. Multiple choice learning: Learning to produce multiple structured outputs.

Advances in neural information processing systems, 25, 2012.

Bouzidi, M.-K., Yao, Y., Goehring, D., and Reichardt, J. Learning-aided warmstart of model predictive control in uncertain fast-changing traffic. arXiv preprint arXiv:2310.02918, 2023.

Hafner, D., Lillicrap, T., Fischer, I., Villegas, R., Ha, D.,
Lee, H., and Davidson, J. Learning latent dynamics for planning from pixels. In *International conference on* machine learning, pp. 2555–2565. PMLR, 2019.

Cadene, R., Alibert, S., Soare, A., Gallouedec, Q., Zouitine, A., and Wolf, T. Lerobot: State-of-the-art machine learning for real-world robotics in pytorch. https: //github.com/huggingface/lerobot, 2024.

Huang, H., Sundaralingam, B., Mousavian, A., Murali, A.,
Goldberg, K., and Fox, D. Diffusionseeder: Seeding motion optimization with diffusion for rapid motion planning. In *8th Annual Conference on Robot Learning*, 2024.

Caesar, H., Kabzan, J., Tan, K. S., Fong, W. K., Wolff, E. M., Lang, A. H., Fletcher, L., Beijbom, O., and Omari, S. NuPlan: A closed-loop ml-based planning benchmark for autonomous vehicles. In Conference on Computer Vision and Pattern Recognition (CVPR) ADP3 Workshop, 2021.

Johnson, T. C., Kirches, C., and Wachter, A. An active-set method for quadratic programming based on sequential hot-starts. *SIAM Journal on Optimization*, 25(2):967–994, 2015.

Chen, S. W., Wang, T., Atanasov, N., Kumar, V., and Morari, M. Large scale model predictive control with neural networks and primal active sets. *Automatica*, 135:109947, 2022a.

Kang, S., Xu, X., Sarva, J., Liang, L., and Yang, H. Fast and certifiable trajectory optimization. In 16th International Workshop on the Algorithmic Foundations of Robotics (WAFR), 2024.

Karkus, P., Ivanovic, B., Mannor, S., and Pavone, M. Diffstack: A differentiable and modular control stack for autonomous vehicles. In *6th Annual Conference on Robot* Learning, 2022.

Otta, P., Santin, O., and Havlena, V. Measured-state driven warm-start strategy for linear mpc. In 2015 European Control Conference (ECC), pp. 3132–3136. IEEE, 2015.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Paden, B., Cˇ ap, M., Yong, S. Z., Yershov, D., and Frazzoli, ´
E. A survey of motion planning and control techniques for self-driving urban vehicles. IEEE Transactions on Intelligent Vehicles, 1(1):33–55, 2016.

Lembono, T. S., Paolillo, A., Pignat, E., and Calinon, S.

Memory of motion for warm-starting trajectory optimization. *IEEE Robotics and Automation Letters*, 5(2):2594–
2601, 2020.

Sacks, J. and Boots, B. Learning to optimize in model predictive control. In International Conference on Robotics and Automation (ICRA), pp. 10549–10556, 2022.

Lenz, I., Knepper, R. A., and Saxena, A. Deepmpc: Learning deep latent features for model predictive control. In Robotics: Science and Systems, volume 10, 2015.

Sambharya, R., Hall, G., Amos, B., and Stellato, B. Learning to warm-start fixed-point optimization algorithms. arXiv preprint arXiv:2309.07835, 2023.

Li, W. and Todorov, E. Iterative linear quadratic regulator design for nonlinear biological movement systems. In First International Conference on Informatics in Control, Automation and Robotics, volume 2, pp. 222–229. SciTePress, 2004.

Scokaert, P. O., Mayne, D. Q., and Rawlings, J. B. Suboptimal model predictive control (feasibility implies stability). IEEE Transactions on Automatic Control, 44(3):648–654, 1999.

Marcucci, T. and Tedrake, R. Warm start of mixed-integer programs for model predictive control of hybrid systems. IEEE Transactions on Automatic Control, 66(6):2433– 2448, 2020.

Sonnerat, N., Wang, P., Ktena, I., Bartunov, S., and Nair, V. Learning a large neighborhood search algorithm for mixed integer programs. *arXiv preprint* arXiv:2107.10201, 2021.

Michalska, H. and Mayne, D. Q. Robust receding horizon control of constrained nonlinear systems. IEEE transactions on automatic control, 38(11):1623–1633, 1993.

Sun, S., Cao, Z., Zhu, H., and Zhao, J. A survey of optimization methods from a machine learning perspective. IEEE transactions on cybernetics, 50(8):3668–3681, 2019.

Mirowski, P., Pascanu, R., Viola, F., Soyer, H., Ballard, A. J., Banino, A., Denil, M., Goroshin, R., Sifre, L., Kavukcuoglu, K., Kumaran, D., and Hadsell, R. Learning to navigate in complex environments. *International* Conference on Learning Representations (ICLR), 2017.

Sundaralingam, B., Hari, S. K. S., Fishman, A., Garrett, C., Van Wyk, K., Blukis, V., Millane, A., Oleynikova, H., Handa, A., Ramos, F., et al. Curobo: Parallelized collision-free minimum-jerk robot motion generation. arXiv preprint arXiv:2310.17274, 2023.

Mugel, S., Kuchkovsky, C., Sanchez, E., Fern ´ andez- ´
Lorenzo, S., Luis-Hita, J., Lizaso, E., and Orus, R. Dy- ´ namic portfolio optimization with real datasets using quantum processors and quantum-inspired tensor networks. *Physical Review Research*, 4(1):013006, 2022.

Tamar, A., Thomas, G., Zhang, T., Levine, S., and Abbeel, P. Learning from the hindsight plan—episodic mpc improvement. In *International Conference on Robotics and* Automation (ICRA), pp. 336–343, 2017.

Nagabandi, A., Kahn, G., Fearing, R. S., and Levine, S.

Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning. In 2018 IEEE international conference on robotics and automation (ICRA), pp. 7559–7566. IEEE, 2018.

Tassa, Y., Mansard, N., and Todorov, E. Control-limited differential dynamic programming. In IEEE International Conference on Robotics and Automation (ICRA), pp. 1168–1175. IEEE, 2014.

Nair, V., Bartunov, S., Gimeno, F., Von Glehn, I., Lichocki, P., Lobov, I., O'Donoghue, B., Sonnerat, N.,
Tjandraatmadja, C., Wang, P., et al. Solving mixed integer programs using neural networks. *arXiv preprint* arXiv:2012.13349, 2020.

Tassa, Y., Doron, Y., Muldal, A., Erez, T., Li, Y., Casas, D.

d. L., Budden, D., Abdolmaleki, A., Merel, J., Lefrancq, A., et al. Deepmind control suite. arXiv preprint arXiv:1801.00690, 2018.

OpenAI, Andrychowicz, M., Baker, B., Chociej, M.,
Jozefowicz, R., McGrew, B., Pachocki, J., Petron, A., ´ Plappert, M., Powell, G., Ray, A., Schneider, J., Sidor, S., Tobin, J., Welinder, P., Weng, L., and Zaremba, W. Learning dexterous in-hand manipulation. *The International* Journal of Robotics Research, 39(1):3–20, 2020.

Wahlstrom, N., Sch ¨ on, T. B., and Deisenroth, M. P. From ¨
pixels to torques: Policy learning with deep dynamical models. *arXiv preprint arXiv:1502.02251*, 2015.

Wang, T. and Ba, J. Exploring model-based planning with policy networks. *arXiv preprint arXiv:1906.08649*, 2019.

Zhao, T. Z., Kumar, V., Levine, S., and Finn, C. Learning fine-grained bimanual manipulation with low-cost hardware, 2023. URL https://arxiv.org/abs/ 2304.13705.

550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 Williams, G., Aldrich, A., and Theodorou, E. Model predictive path integral control using covariance variable importance sampling. *arXiv preprint arXiv:1509.01149*, 2015.

Xiao, X., Zhang, T., Choromanski, K., Lee, E., Francis, A.,
Varley, J., Tu, S., Singh, S., Xu, P., Xia, F., et al. Learning model predictive controllers with real-time attention for real-world navigation. *arXiv preprint arXiv:2209.10780*, 2022.

Ye, Y., Pei, H., Wang, B., Chen, P.-Y., Zhu, Y., Xiao, J., and Li, B. Reinforcement-learning based portfolio management with augmented asset movement prediction states. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 1112–1119, 2020.

## A. Appendix A.1. Detailed Descriptions Of Baseline Optimizers

This subsection provides detailed descriptions of the optimization algorithms used in our evaluations: First-order Box Differential Dynamic Programming (DDP), Model Predictive Path Integral (MPPI), and the Iterative Linear Quadratic Regulator (iLQR). These algorithms were selected due to their widespread use and effectiveness in solving optimal control problems across various domains. First-order Box Differential Dynamic Programming (DDP). Building on the work of (Tassa et al., 2014), (Amos et al., 2018) introduced a simplified version of Box-DDP that utilizes first-order linearization instead of second-order derivatives. This approach, termed "first-order Box-DDP," reduces computational complexity while maintaining the ability to handle box constraints on both the state and control spaces.

Model Predictive Path Integral (MPPI). MPPI (Williams et al., 2015) is a sampling-based model predictive control algorithm that iteratively refines control inputs using stochastic sampling. Starting from the current state and a prior solution, it generates a set of randomly perturbed control sequences, simulates their trajectories, and evaluates them using a cost function. The control inputs are then updated based on a weighted average, favoring lower-cost trajectories. We use the implementation from https://github.com/UM-ARM-Lab/pytorch_mppi.

Iterative Linear Quadratic Regulator (iLQR). iLQR (Li & Todorov, 2004) is a trajectory optimization algorithm that refines control sequences iteratively by linearizing system dynamics and approximating the cost function quadratically around a nominal trajectory. It alternates between a forward pass, simulating the system trajectory, and a backward pass, computing optimal control updates. We use the implementation provided in the nuPlan simulator.

## A.2. Detailed Task Descriptions And Hyperparameters

This subsection provides detailed descriptions of the tasks used in our experiments: cart-pole, reacher, and autonomous driving. For each task, we outline the system dynamics, control inputs, and the specific hyperparameters employed in our evaluations. Cart-pole. The cart-pole task (Barto et al., 1983) involves a cart-pole system tasked with swinging the pole upright while moving the cart to a randomly selected target position along the rail. The goal is to balance the pole vertically and simultaneously reach the target cart position. The system is characterized by the state vector st ∈ R
4, which includes the pole angle θ, pole angular velocity ˙θ, cart position x, and cart velocity x˙. The control input is a single force applied to the cart, ut ∈ R.

Hyperparameters. The mass of the cart is mc = 1.0 kg, the mass of the pole is mp = 0.3 kg, and the length of the pole is l = 0.5 m. Gravity is set to g = −9.81 m/s2. Control inputs are bounded by umin = −5.5 N and umax = 5.5 N, with a time step of ∆t = 0.1 s and nsub steps = 2 physics sub-steps per control step. Each episode has a maximum length of Tenv = 50 steps. Both optimizers use goal weights of [0.1, 0.01, 1.0, 0.01] for the state variables and a control weight of 0.0001. The prediction horizon is set to H = 10. For the online optimizer, we set lqr iter = 2 and max linesearch iter = 1.

In the oracle optimizer, we use lqr iter = 10 and max linesearch iter = 3. The initial state s0 ∈ R
4is sampled as follows: x0 ∼ U(−2, 2) m, x˙ 0 ∼ U(−1, 1) m/s, θ0 ∼ U(−
π 2
,
π 2
)rad, and ˙θ0 ∼ U(−
π 4
,
π 4
)rad/s. The goal state is defined by a target cart position xgoal ∼ U(−2, 2) m, while the rest of the state variables (pole angle and velocities) are set to zero, ensuring the goal is to bring the pole upright and bring the system to rest. Reacher. The Reacher task (Tassa et al., 2018) involves a two-link planar robot arm tasked with reaching a randomly positioned target. The goal is to move the end-effector to the target position in the plane. The system is characterized by the state vector st ∈ R
4, which includes the joint angles θ1, θ2 and angular velocities ˙θ1,˙θ2. The control inputs, ut ∈ R
2, are torques applied to each joint.

Hyperparameters. The simulation uses a time step of ∆t = 0.02 s, with joint damping set to 0.01 and motor gear ratios of 0.05. The control inputs are constrained by umin = [−1, −1] and umax = [1, 1]. The wrist joint has a limited range of [−160◦, 160◦]. Each episode is limited to Tenv = 250 steps. Both optimizers are set with a control noise covariance σ 2 = 1 × 10−3, a temperature parameter λ = 1 × 10−4, and a prediction horizon H = 10. The online optimizer uses 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 num samples = 3, while the oracle optimizer uses num samples = 50. The target position is generated by sampling θtarget ∼ U(0, 2π) and rtarget ∼ U(0.05, 0.20) m, then set as posx = rtarget cos(θtarget) and posy = rtarget sin(θtarget).

Autonomous Driving. The autonomous driving task, based on the nuPlan benchmark (Caesar et al., 2021), evaluates the performance of motion planning algorithms in complex urban environments. Our focus is on the control (tracking) layer, which is responsible for accurately following the planned trajectories. The task involves navigating a vehicle through a series of scenarios with varying traffic conditions, obstacles, and road layouts. The goal is to execute safe, efficient, and comfortable trajectories while adhering to traffic rules and avoiding collisions. The system is characterized by the state vector st ∈ R
5, which includes the vehicle's position (x, y), orientation ϕ, velocity v, and steering angle δ. The control inputs, ut ∈ R
2, are acceleration a and steering angle rate ˙δ. We use the state-of-the-art Predictive Driver Model (PDM)
planner (Dauner et al., 2023) to generate the reference trajectories rt.

Hyperparameters. The prediction horizon is set to H = 40 with a discretization time step of ∆t = 0.2 s. The cost function is weighted with state cost diagonal entries [1.0, 1.0, 10.0, 0.0, 0.0] for the position, heading, velocity, and steering angle, respectively, and input cost diagonal entries [1.0, 10.0] for acceleration and steering angle rate. The maximum acceleration is constrained to 3.0 m/s2, the maximum steering angle is 60◦, and the maximum steering angle rate is 0.5 rad/s.

A minimum velocity threshold for linearization is set at 0.01 m/s. The online optimizer is limited to a maximum solve time of max solve time = 5 ms, while the oracle optimizer allows for max solve time = 50 ms. For a fair comparison, we keep the total runtime limit fixed, including both initialization and optimization. The total runtime limit for warm-start and perturbation methods is 5 ms. For learning-based methods, the inference time for our neural networks is between 0.6 ms and 0.7 ms on a GPU, and 0.7 ms to 0.8 ms on a CPU. For simplicity, we allocate 0.85 ms for model inference and run the optimizer for the remaining 4.15 ms. We have not performed any inference optimization for our models (e.g., TensorRT).

## A.3. Network Architecture And Training Details

This section provides a brief overview of the network architecture, the data collection process, and the training procedures used in our experiments. We summarize the design of our base Transformer model, outline the methods used to generate and preprocess the training data, and detail the key training methodologies and hyperparameters employed. Network Architecture. The base model is a standard Transformer architecture with absolute positional embedding, a linear decoder layer, and output scaling. The core Transformer architecture remains standard, with task-specific configurations.

The input to the network is the concatenated sequence of the warm-start state trajectory error, τ w.s.

e, defined as the difference between the reference trajectory, ψ = τr, (in the autonomous driving task) or the goal state, ψ = xg (in the cart-pole and reacher tasks), and the warm-start trajectory, τ w.s.

x. The warm-start control trajectory is τ w.s.

u = {{u cand t+k
}
H−2 k=1 
, 0}. The network predicts K control trajectories, {τˆ
init u,k }
K
k=1, for the next optimization step. Each environment's configuration is described in Table 3.

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714

| Table 3. Configuration of the Transformer model for each environment. Parameter Description Reacher Cart-pole Driving n layer Number of layers 4 4 4 n head Number of heads 2 2 2 n embd Embedding dimension 64 64 64 dropout Dropout rate 0.1 0.1 0.1 src dim Input dimension 8 5 7 src len Sequence length 10 9 40 out dim Output dimension 2 1 2   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Data Collection and Preprocessing. In all experiments, the training data is generated by (1) unrolling an optimizer using a warm-start initialization policy and recording its inputs and outputs, and (2) replaying the same scenarios using an oracle optimizer—essentially the same optimizer with enhanced capabilities, such as more optimization steps or additional sampled trajectories—and logging its inputs and outputs. The resulting mapping may be seen as a filter, refining (near-)optimal initial solutions into optimal ones. For each task, 500,000 instances were collected.

Training. Prior to training, all features were standardized to ensure consistent input scaling. We used the AdamW optimizer and applied gradient norm clipping. The models were trained using standard settings without any complex modifications. All hyperparameters are detailed in Table 4.

715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769

| Table 4. Training hyperparameters for each environment.   |                        |         |           |         |
|-----------------------------------------------------------|------------------------|---------|-----------|---------|
| Parameter                                                 | Description            | Reacher | Cart-pole | Driving |
| epochs                                                    | Epochs                 | 125     | 125       | 125     |
| batch size                                                | Batch size             | 1024    | 1024      | 1024    |
| lr                                                        | Learning rate          | 0.001   | 0.0003    | 0.0001  |
| weight decay                                              | Weight decay           | 0.0001  | 0.0001    | 0.0001  |
| grad norm clip                                            | Gradient norm clipping | 2.0     | 2.0       | 2.0     |
| control loss weight                                       | Control loss weight    | 100.0   | 1.0       | 5.0     |
| state loss weight                                         | State loss weight      | 0       | 0.01      | 0.005   |
| pairwise loss weight                                      | Pairwise loss weight   | 0.1     | 0.01      | 0.1     |

## A.4. Descriptions Of Baseline Methods

This section introduces the baseline methods used in our experiments in more detail and discusses their implementation specifics. These baselines serve as reference points to evaluate our proposed methods' performance and understand the benefits and limitations of different initialization strategies in optimization algorithms. Warm-start (K **= 1).** A common technique involves shifting the previous solution forward by one time step and padding it with zeros. Assuming the system does not exhibit rapid changes in this interval, the previous solution should retain local, feasible information. Oracle Proxy (K **= 1).** This proxy serves two purposes: (1) estimating the gap between a real-time-constrained optimizer and an unrestricted one and (2) providing a proxy for a mapping worth learning. For each optimization algorithm, a suitable heuristic is defined. In DDP, the oracle is allowed more iterations to converge; in MPPI, it has a larger sample budget; and in iLQR, it is given more time to perform optimization iterations. Regression (K **= 1).** This approach involves training a neural network to approximate the oracle's mapping. Unlike the oracle heuristic, which is impractical for real-time use, the trained neural network requires only a single forward pass. Warm-start with Perturbations (K >1). We utilize the warm-start technique as another baseline by duplicating the proposed initial solution K times and adding Gaussian noise. While this introduces some form of dispersion, the resulting initial solutions are neither guaranteed to be feasible nor to ensure any level of optimality. Regression with Perturbations (K >1). Similar to the warm-start with perturbation, after predicting an initial solution using the neural network, we duplicate it K times and add perturbations.

Ensemble (K >1). An ensemble of K separate neural networks leverages the idea that networks initialized with different weights during training will often produce different predictions. The main drawbacks of this approach are (1) the need to train K neural networks and (2) the requirement to run K forward passes, which may be impractical for real-time deployment. Multi-Output Regression (K >1). A naive approach to predicting K initializations from a single network involves calculating the loss for each prediction and summing the losses. However, since there is no explicit multimodality objective, these models are prone to mode collapse. MISO Pairwise Distance (K >1). One way to mitigate mode collapse is to introduce an additional term in the loss function, such as the pairwise distance between predictions. While this approach requires weight tuning and selecting an appropriate norm, a significant challenge lies in understanding how effectively this term promotes multimodality in practice.

MISO Winner-Takes-All (K >1). This approach updates only the best-performing mode based on the loss of each prediction. Although no explicit dispersion objective is included, multimodality is indirectly encouraged by maintaining multiple active modes while refining only the best one and not penalizing the others.

MISO Mix (K >1). Lastly, we combine the pairwise distance term with the Winner-Takes-All approach. While this allows for greater refinement, it adds complexity due to additional hyperparameters and the need to apply further operations—such as clamping distances—to ensure the loss won't diverge.

## A.5. Illustrative Example

This example provides a simplified scenario to illustrate the behavior of local optimization algorithms in a controlled, low-dimensional setting with non-convex and multimodal objective function. The system dynamics are defined by the linear equation xt+1 = xt + ut, where xt ∈ R represents the state and ut ∈ [−1, 1] is the constrained control input at time step t. The initial state is x0 = 0, and the optimization horizon is set to H = 5. The objective is to find the optimal control trajectory τ
⋆
u = {ut}
H−1 t=0 that minimizes the cumulative cost function τ
⋆
u = arg minτu PH−1 k=0 c(xt+k), where the resulting state trajectory τx = {xt}
H
t=0 is obtained by unrolling τu from x0. The cost function, c(x) = (x 2 + 0.05)(x+ 1.5)2(x−2)2, is non-convex and multimodal, featuring two global minima at x
⋆1 = −1.5 and x
⋆2 = 2.

Figure 6. Mean inference time for varying values of K for the ensemble and multi-output models on CPU and CUDA
This experiment benchmarks the execution time of two model architectures: an ensemble of K single-output models and 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824

| Table 5. Comparison of different methods for predicting the optimal control trajectories Method xH+1 τˆu Ensemble 0.03 -0.02, 0.03, 0.00, 0.01, 0.01 0.02 -0.01, 0.02, -0.01, 0.01, 0.01 MISO pairwise dist. -0.24 -0.12, -0.05, -0.04, -0.02, 0.01 0.24 0.08, 0.05, 0.04, 0.03, 0.03 MISO winner-takes-all -1.49 -0.88, -0.65, 0.00, 0.02, 0.02 2.01 0.99, 0.94, 0.05, 0.01, 0.01 MISO mix -1.52 -0.90, -0.68, 0.00, 0.03, 0.03 2.05 0.99, 0.95, 0.07, 0.02, 0.02 Optimal -1.50 -1.00, -0.50, 0.00, 0.00, 0.00 2.00 1.00, 1.00, 0.00, 0.00, 0.00   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

The results from Table 5 provide a comparison of different methods for predicting optimal control trajectories. The Ensemble method, which combines multiple single-output predictions, yields the least accurate results, with final states close to zero. Due to the dispersion term, MISO pairwise distance is a bit further from zero but still far from either optimum. On the other hand, MISO winner-takes-all and MISO mix successfully predict both optimal sequences with high fidelity and thus are able to reach either global optimum. Overall, the results suggest that methods specifically tailored for capturing multimodality, such as the winner-takes-all and mixed strategies, are more effective than their single-output regression counterparts, particularly in non-convex environments. a single multi-output model producing K outputs. Both models are based on the Transformer architecture used in the autonomous driving environment. The experiments were conducted on an Intel Core i9-13900KF CPU and an NVIDIA RTX 4090 GPU, measuring the mean inference time over 1000 runs across five random seeds. GPU operations were synchronized before timing to ensure accurate measurements. The results, shown in Fig. 6, display the mean inference time in milliseconds as K increases for both the ensemble and multi-output models on CPU and GPU. The multi-output model exhibits minimal sensitivity to the increase in K on both the CPU and GPU, indicating that this architecture scales efficiently, maintaining a low overhead even as the number of outputs grows. In contrast, the ensemble model's inference time increases significantly with K, suggesting that managing multiple models introduces overhead that scales poorly as K grows. In applications with strict runtime constraints, such as the autonomous driving environment, the ensemble approach becomes impractical as K increases. Conversely, the multi-output model remains a viable option, even at larger values of K, making it the preferred choice for time-sensitive scenarios.

## A.7. State Loss

825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 An additional challenge in learning control policies is addressing *compounding errors*—small inaccuracies in the predicted control trajectory τu that, when unrolled, cause significant deviations in the state trajectory τx. Even if most elements of τu are accurate, errors in the initial steps can cause the state τx to drift, leading to further divergence as the system evolves. To mitigate compounding errors, we introduce a regression loss not only over the control trajectory τu but also over the resulting state trajectory τx. In a supervised learning setting, this requires a model of the system dynamics, which can either be known or learned.

Let τˆu = {uˆt}
H−1 t=0 denote the predicted control trajectory, and τ
⋆
u = {u
⋆
t }
H−1 t=0 denote the target control trajectory.

Similarly, let τˆx = {xˆt}
H
t=1 be the predicted state trajectory obtained by unrolling the predicted controls through the system dynamics starting from the initial state x0, i.e.,

$${\hat{\mathbf{x}}}_{t+1}=f({\hat{\mathbf{x}}}_{t},{\hat{\mathbf{u}}}_{t}),\quad{\mathrm{with~}}{\hat{\mathbf{x}}}_{0}={\mathbf{x}}_{0},$$
xˆt+1 = f(xˆt,uˆt), with xˆ0 = x0, (2)
and let τ
⋆
x = {x
⋆
t }
H
t=1 be the target state trajectory.

We define the control loss as

$${\mathcal{L}}_{\mathrm{control}}={\frac{1}{H}}\sum_{t=0}^{H-1}\left\|{\hat{\mathbf{u}}}_{t}-\mathbf{u}_{t}^{\star}\right\|^{2},$$
2, (3)
and the state loss as Our total loss function combines these two components:

$${\mathcal{L}}_{\mathrm{state}}={\frac{1}{H}}\sum_{t=1}^{H}\left\|{\hat{\mathbf{x}}}_{t}-\mathbf{x}_{t}^{\star}\right\|^{2}.$$
2. (4)
L = Lcontrol + λLstate, (5)
where λ is a weighting factor that balances the contributions of the control loss and the state loss.

By incorporating the state loss Lstate, we encourage the predicted control trajectory to produce a state trajectory that remains close to the target state trajectory, thereby mitigating compounding errors during rollout. As we show in Table 6, incorporating state trajectory loss helps mitigate these types of error accumulation and improve long-horizon trajectory accuracy. More specifically, we see that (1) as the prediction horizon increases, from H = 9
(Cart-pole) to H = 40 (Driving), so does the difference between using and not using state loss, (2) The gap between One-Off and Sequential also increases thus we do not generalize as well, (3) For the single-output regression model, the difference is even greater.

Table 6 shows a comparison between using and not using state loss (SL) across One-Off and Sequential Optimization settings. While both strategies benefit from including the state loss, the improvement is more profound in the Sequential

## A.8. Mode Frequency

880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 Figure 7 presents a heatmap illustrating the percentage of selections for each specific mode (output) in MISO winner-takes-all by the selection function, which in this context chooses the output with the lowest resulting cost. Although a few modes appear to dominate, all other modes remain active, i.e., with a frequency greater than zero, indicating that there are problem instances where these less frequent modes identify the optimal action, thereby contributing to the overall performance improvements. Additionally, Fig. 8 compares the distribution of selections between MISO winner-takes-all and MISO mix with the expected, approximately uniform distribution of the Ensemble method.

(a) Cart-pole (Single Optimizer) (b) Driving (Single Optimizer)
(a) Cart-pole (Single Optimizer) (b) Driving (Single Optimizer)
Optimization setting. This is because it involves executing each solution and starting the next optimization from the resulting state, causing the compounding errors to accumulate. The One-Off Optimization setting also benefits from state loss, but the impact is less apparent and thus was not marked in the table.

| Table 6. Mean Cost Comparison for One-Off and Sequential Optimization (With and Without State Loss) One-Off Optimization Sequential Optimization   |         |             |             |                           |             |                        |                           |                           |
|----------------------------------------------------------------------------------------------------------------------------------------------------|---------|-------------|-------------|---------------------------|-------------|------------------------|---------------------------|---------------------------|
| Method                                                                                                                                             | K       | Cart-pole   | Driving     | Cart-pole                 | Driving     |                        |                           |                           |
| No SL                                                                                                                                              | With SL | No SL       | With SL     | No SL                     | With SL     | No SL                  | With SL                   |                           |
| Regression                                                                                                                                         | 1       | 11.88 ±0.79 | 11.19 ±0.80 | 440.97 ±74.92             | 74.23 ±7.69 | 11.93 ±0.29 6.18 ±0.47 | 590.50 ±89.06 70.62 ±7.38 |                           |
| MISO WTA                                                                                                                                           | 32      | 10.62 ±0.77 | 10.48 ±0.77 | 128.32 ±22.55 30.17 ±2.24 | 1.86 ±0.20  | 0.83 ±0.06             | 151.4 ±20.70              | 30.75 ±2.15               |
| MISO mix                                                                                                                                           | 32      | 10.63 ±0.77 | 10.48 ±0.77 | 153.37 ±21.8              | 33.95 ±2.39 | 2.28 ±0.26             | 0.79 ±0.04                | 212.70 ±28.02 33.38 ±2.21 |
| MISO WTA                                                                                                                                           | 32      | 10.25 ±0.75 | 10.29 ±0.76 | 144.51 ±19.51 30.87 ±2.30 | 0.97 ±0.06  | 0.76 ±0.05             | 158.71 ±20.94 30.48 ±2.07 |                           |
| MISO mix                                                                                                                                           | 32      | 10.24 ±0.75 | 10.29 ±0.76 | 196.09 ±31.43 33.52 ±2.35 | 1.14 ±0.12  | 0.63 ±0.02             | 214.56 ±28.02 34.85 ±2.64 |                           |

935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989

## A.9. Mean Cost Sequential Optimization

(a) Cart-pole: Single Optimizer (b) Cart-pole: Multiple Optimizers
(c) Reacher: Single Optimizer (d) Reacher: Multiple Optimizers
Figure 9 shows the mean cost of each method with varying values of K, for both cart-pole and reacher environments. Both showcase that our method scales effectively and consistently with the number of predicted initial solutions and outperforms other approaches across varying values of K.

## A.10. Selection Function Λ

A.10.1. ALTERNATIVE SELECTION CRITERIA
In our framework, the selection function Λ plays a critical role in choosing the most promising initial solution from the set of candidates predicted by our model. While using the objective function J as Λ is a natural and effective choice, alternative choices for Λ can be advantageous in certain scenarios.

990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 Constraint Satisfaction: In some applications, especially those that are safety-critical, it is essential to ensure that certain constraints are satisfied by the initial solution, even if it means accepting a higher value of J. In such cases, Λ can be designed to prioritize solutions that satisfy these constraints. For example:
Λ(xˆ
init k, ψ) = J(xˆ
init k, ψ) + β C(xˆ
init k, ψ), (6)
where C(xˆ
init k
, ψ) measures the degree of constraint violation, and β is a weighting factor that penalizes constraint violations.

Robustness Measures: Λ can incorporate robustness criteria, selecting initial solutions less sensitive to model uncertainties or external disturbances. For instance, it could favor solutions that maintain performance across various scenarios. Contextual Adaptation: The selection function can adapt based on the problem instance ψ. For example, in varying environmental conditions, Λ could prioritize more conservative or aggressive solutions depending on the context or operational requirements.

## A.10.2. Learning The Selection Function

Instead of hand-crafting Λ, it can be learned from data. One approach is to model Λ as a parameterized function, such as a neural network, and train it jointly with the predictor model or separately. The learning objective could be to maximize the overall performance of the optimizer when initialized with the selected solutions, potentially incorporating criteria like safety, robustness, or energy efficiency.

A.10.3. EXAMPLES
Safety-Critical Control: In autonomous driving, safety constraints such as maintaining a safe distance from obstacles are crucial. Λ can prioritize trajectories that ensure safety over those that simply minimize time or fuel consumption. For example, it can assign infinite cost to any solution violating safety constraints, effectively excluding unsafe options.

Adaptive Behavior: In robotics, Λ can select initial solutions that favor energy efficiency when the robot's battery is low or prioritize speed when tasks are time-sensitive. By incorporating the robot's current state or mission objectives into Λ, the system can adapt its behavior accordingly.

## A.11. Evaluation Modes: One-Off And Sequential

In our experiments, we assess the performance of the methods using two evaluation modes: One-off Evaluation: In the one-off evaluation, problem instances are uniformly sampled from the evaluation dataset (disjoint from the training dataset). Each method is tested on the same set of independently sampled instances, ensuring a fair comparison across methods. The optimizer solves each problem instance independently, without any interaction with the environment or influence from previous solutions. This evaluation mode focuses on the optimizer's ability to find high-quality solutions for individual problems in isolation. Sequential Evaluation: In sequential evaluation, the optimizer interacts with the environment across a series of time steps. Starting from an initial state sampled from the evaluation dataset (disjoint from the training dataset), at each time step the optimizer adjusts its decisions based on the evolving state. This mode evaluates the optimizer's performance in a dynamic, real-time setting, highlighting its ability to manage evolving states and adapt over time.

## A.12. Sequential Vs. Closed-Loop And One-Off Vs. Open-Loop

In control settings, the terms *sequential* and *closed-loop* evaluations are often used interchangeably. However, in the context of general optimization problems, the notion of "closed-loop" may not always be applicable, as there may be no dynamic environment or feedback mechanism involved. Therefore, we adopt more general terminology—referring to *sequential* evaluations—to encompass scenarios where decisions are made in a sequence but without necessarily involving feedback from an environment. On the other hand, the distinction between *one-off* and *open-loop* evaluations is subtle yet significant. In a *one-off* evaluation, we assess the optimizer's performance on individual problem instances without any interaction with an environment. This means the optimizer solves a static problem, and we can directly compare different methods on the same set of instances. In contrast, open-loop control involves sequentially executing actions in an environment without feedback.

## A.13. Alternative Neural Backbone Architectures

We chose a standard Transformer backbone primarily for its simplicity and clarity. Our focus is not on comparing different sequence-based architectures, e.g., Action Chunking Transformers (Zhao et al., 2023), LSTM variants, or hierarchical models, but on introducing a multi-output framework that can be integrated into a wide range of neural backbones to generate multiple diverse initial solutions. Consequently, replacing the Transformer with, say, an Action Chunking Transformer or another advanced architecture would still benefit from the same multi-output concept and diversity-promoting objectives, with minimal implementation changes.

## A.14. Sampling-Based Architectures

1045 1046 1047 1048 1049 1050 1051

## 1052 1053

1054 1055 1056 1057 1058 1059 1060

## 1061

1062 1063 1064 1065 1066 1067 1068 1069 1070

## 1071

1072 1073 1074 1075 1076 1077 1078 1079 1080 1081 1082 1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099 Methods that rely on sampling multiple solutions at inference time—such as diffusion models—are particularly ill-suited for real-time settings with strict runtime constraints. Generating K distinct samples typically entails either multiple iterative steps per sample or repeated forward passes, causing inference time to scale poorly with K. This overhead is incompatible with scenarios like autonomous driving, where any significant increase in computation time can undermine real-time performance. For instance, a Diffusion Policy (Chi et al., 2024) took1 378.50 milliseconds to produce a single sequence for the autonomous driving task (length 40 and dimension 2), compared to *K >>*1 sequences in just 0.85 milliseconds using a multi-output regressor.

## A.15. Experiments On Combining Miso With Warm-Start

In Table 7 and Table 8 we report mean cost results for combining MISO predicted initial solution candidates with the heuristic warm-start initial solution. Specifically MISO⋆ denotes a setting where the warm-start is added as an extra candidate. Results show that MISO⋆consistently improves over warm-start. Further, including warm-start as one of the candidate initial solutions can sometimes lead to additional improvements (MISO vs. MISO⋆).

Table 7. Results for combining MISO with warm-start for the single optimizer setting.

One-Off Optimization Sequential Optimization

Method K **Reacher Cart-pole Driving Reacher Cart-pole Driving**

Single Optimizer Warm-start 1 13.48 ±0.88 11.69 ±0.84 283.86 ±37.91 13.48 ±0.88 11.69 ±0.84 283.86 ±37.91 MISO winner-takes-all 32 13.36 ±0.88 10.48 ±0.77 30.17 ±2.24 2.72 ±0.21 0.83 ±0.06 30.75 ±2.15 MISO mix 32 12.74 ±0.86 10.48 ±0.77 33.95 ±2.39 2.44 ±0.20 0.79 ±0.04 33.38 ±2.21 MISO⋆ winner-takes-all 32 13.36 ±0.88 10.47 ±0.77 29.95 ±2.04 2.70 ±0.21 0.82 ±0.05 29.62 ±1.98 MISO⋆ mix 32 12.74 ±0.86 10.47 ±0.77 33.19 ±2.39 2.30 ±0.19 0.77 ±0.04 33.82 ±2.48

Table 8. Results for combining MISO with warm-start for the multiple optimizers setting.

One-Off Optimization Sequential Optimization

Method K **Reacher Cart-pole Driving Reacher Cart-pole Driving** Multiple Optimizers Warm-start w. perturb 32 13.41 ±0.88 10.93 ±0.79 155.53 ±24.33 5.89 ±0.50 6.68 ±0.59 162.13 ±34.10 MISO winner-takes-all 32 13.34 ±0.88 10.29 ±0.76 30.87 ±2.30 2.21 ±0.16 0.76 ±0.05 30.48 ±2.07 MISO mix 32 12.72 ±0.86 10.29 ±0.76 33.52 ±2.35 1.56 ±0.14 0.63 ±0.02 34.85 ±2.64 MISO⋆ winner-takes-all 32 13.34 ±0.88 10.29 ±0.76 30.62 ±2.24 2.19 ±0.17 0.76 ±0.05 29.71 ±1.96 MISO⋆ mix 32 12.72 ±0.86 10.28 ±0.76 33.26 ±2.29 1.53 ±0.14 0.63 ±0.02 33.61 ±2.36

1Mean over 1000 runs using Cadene et al. (2024) implementation, on an NVIDIA RTX 4090 GPU.