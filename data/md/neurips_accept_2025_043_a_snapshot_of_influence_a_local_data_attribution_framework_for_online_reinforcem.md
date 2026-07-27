# A Snapshot Of Influence: A Local Data Attribution Framework For Online Reinforcement Learning

| Yuzheng Hu∗ UIUC                                     |                |
|------------------------------------------------------|----------------|
| Urbana, IL 61801                                     |                |
| yh46@illinois.edu                                    | Fan Wu∗ UIUC   |
| Urbana, IL 61801                                     |                |
| fanw6@illinois.edu                                   | Haotian Ye     |
| Stanford University Stanford, CA 94305               |                |
| haotianye@stanford.edu                               |                |
| David Forsyth UIUC Urbana, IL 61801 daf@illinois.edu | James Zou      |
| Stanford University Stanford, CA 94305               |                |
| jamesz@stanford.edu                                  | Nan Jiang UIUC |
| Urbana, IL 61801                                     |                |
| nanjiang@illinois.edu                                |                |

| Jiaqi W. Ma† UIUC    |
|----------------------|
| Urbana, IL 61801     |
| jiaqima@illinois.edu |

Han Zhao†
UIUC
Urbana, IL 61801 hanzhao@illinois.edu

## Abstract

Online reinforcement learning (RL) excels in complex, safety-critical domains but suffers from sample inefficiency, training instability, and limited interpretability. Data attribution provides a principled way to trace model behavior back to training samples, yet existing methods assume fixed datasets, which is violated in online RL where each experience both updates the policy and shapes future data collection. In this paper, we initiate the study of data attribution for online RL, focusing on the widely used Proximal Policy Optimization (PPO) algorithm. We start by establishing a *local* attribution framework, interpreting model checkpoints with respect to the records in the recent training buffer. We design two target functions, capturing agent action and cumulative return respectively, and measure each record's contribution through gradient similarity between its training loss and these targets. We demonstrate the power of this framework through three concrete applications: diagnosis of learning, temporal analysis of behavior formation, and targeted intervention during training. Leveraging this framework, we further propose an algorithm, iterative influence-based filtering (IIF), for online RL training that iteratively performs experience filtering to refine policy updates. Across standard RL benchmarks (classic control, navigation, locomotion) to RLHF for large language models, IIF reduces sample complexity, speeds up training, and achieves higher returns. Together, these results open a new direction for making online RL more interpretable, efficient, and effective.

## 1 Introduction

Reinforcement learning (RL) has achieved remarkable success across a wide range of decision-making tasks, from game playing [Mnih et al., 2015, Silver et al., 2016] to robotic control [Andrychowicz et al., 2020] and the alignment of large language models (LLMs) [Ouyang et al., 2022]. Among its
∗Equal contribution †Equal advising variants, online RL, which continuously alternates between data collection and policy updates (e.g., A3C [Mnih et al., 2016], PPO [Schulman et al., 2017]), is well-suited to real-time, adaptive, and safety-critical domains such as autonomous driving, as it enables on-the-fly correction of mistakes and rapid adaptation to non-stationary environments [Sallab et al., 2017, Andrychowicz et al., 2020]. However, modern online RL faces several challenges, including sample inefficiency, high variance, and training instability, often requiring millions of interactions for convergence and yielding inconsistent performance across runs [Henderson et al., 2018, Yu, 2018, Dulac-Arnold et al., 2019]. These challenges, together with their deployment in high-stakes domains, necessitate a deeper understanding of the operational mechanisms of online RL. To this end, prior work has explored various methods for RL interpretability [Milani et al., 2024, Cheng et al., 2025]. While useful, these methods often lack the fine-grained explanations necessary for effective interventions or have limited applicability (see Sec. 6 for a detailed review of related work). Addressing these limitations requires exploring new paradigms. In recent years, *data attribution* [Deng et al., 2025] has emerged as a powerful approach for machine learning interpretability, offering a complementary perspective by tracing model behaviors back to training data. This framework further benefits downstream applications such as data selection [Xia et al., 2024], bias mitigation [Wang et al., 2024], fact tracing [Chang et al., 2025], among others. However, applying data attribution to online RL is non-trivial. In online RL, agents continuously interact with their environment; each collected experience not only contributes to policy updates but also influences future rollouts collected by the evolving policy. This violates the core assumptions of traditional data attribution methods, which are designed for static datasets and fixed objectives. In this work, we address this gap by presenting the first study of data attribution for online RL, specifically focusing on the widely used Proximal Policy Optimization (PPO) algorithm [Schulman et al., 2017]. Our contributions are threefold: 1. A principled and flexible framework (Sec. 3). We propose a local data attribution framework for online RL, interpreting model checkpoints w.r.t. the records from the recent training buffer. We define the attribution entity as the atomic unit in PPO training, design two target functions that capture agent actions and cumulative returns, and measure each record's influence through gradient similarity between its training loss and the target.

2. Fresh insights into learning (Sec. 4). We demonstrate the power of our framework through three applications: a) *diagnosis of learning*: we show records most harmful for learning feature inaccurate advantage estimates; b) *temporal analysis of behavior formation*: we reveal an intriguing phase transition of critical records in shaping agent behaviors; c) *targeted intervention*: we show that removing records with the most negative influences can effectively improve model training.

3. Improved training (Sec. 5). Building on the targeted intervention, we further develop an iterative influence-based filtering algorithm (IIF) that significantly improves standard online RL training. Across standard RL benchmarks to modern RLHF for large language models, IIF consistently improves *sample efficiency*, reduces *computational cost*, and enhances *final performance*.

## 2 Preliminaries 2.1 Online Reinforcement Learning

We consider the online RL setting, where an agent learns to maximize long-term returns by interacting with the environment. The environment E is modeled as a Markov Decision Process (MDP) defined by the tuple (S, A*, P, R, γ, d*0), where S is the state space, A the action space, P the transition function, R the reward function, γ ∈ [0, 1] the discount factor, and d0 ∈ P(S) the initial state distribution. At timestep t, the agent observes st, takes action at, receives reward rt, and transitions to st+1. Online RL typically proceeds in alternating **training rounds** of data collection and model training (Fig. 1). In round k, the data collection phase involves the agent executing the current policy πθ
(k) , sampling experiences over multiple episodes to accumulate n transition records in a rollout buffer B(k). Each record contains the raw transition (st, at, rt) and several computed quantities, including the action log probability log πθ
(k) (at|st), estimated value vt, and advantage estimate Aˆt.

Model parameters are then updated iteratively starting from θ
(k)
0 = θ
(k): at optimization step j, training on the mini-batch B
(k)
jdrawn from B(k) updates parameters from θ
(k)
jto θ
(k)
j+1. In this paper,

B(k)
Env E
Rollout buffer Training round 
… (s0,a0,r0, log ⇡0,v0, Aˆ0)
(sT ,aT ,rT , log ⇡T ,vT , AˆT )
Alternating learning cycle Sampling experiences to fill episodes Batching Batching ✓(k+1)
B(k)
0 …
Model ✓(k)
Record batch Model Record batch
✓
(k)
j B(k) m PPO **training** updates Measuring influence Local Data Attribution Validation gradient Per-record training gradient Inner product
=> Record influence r✓LPPO(✓
(k)
j ,zi)
Target functions f 2 {f action,freturn}
Ii r✓f(✓
(k)
j )
we focus on Proximal Policy Optimization (PPO), a widely used, effective algorithm in various applications [Berner et al., 2019, Andrychowicz et al., 2020, Ouyang et al., 2022]. Proximal policy optimization (PPO) [Schulman et al., **2017].** PPO is a policy gradient method for online RL that optimizes a clipped surrogate function. The core PPO objective, which is typically

combined with a value function loss and an entropy bonus during optimization, is defined as:  $\mathcal{L}^\text{PPO}(\theta)=\mathbb{E}_{(s,a)\sim\mathbb{E}_s^{[\theta]}}\left[\min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta(s)}(a|s)}\hat{A}(s,a),\text{clip}\left(\frac{\pi_\theta(a|s)}{\pi_{\theta(s)}(a|s)},1-\epsilon,1+\epsilon\right)\hat{A}(s,a)\right)\right],$  where $\epsilon$ is a hyperparameter that limits policy changes between rounds and promotes stable learning.  

## 2.2 Data Attribution

Data attribution, which quantifies the influence of individual training samples on model behavior, has become increasingly important in machine learning [Grosse et al., 2023, Wang et al., 2023, Zheng et al., 2024]. Common techniques include influence functions [Koh and Liang, 2017], Data Shapley [Ghorbani and Zou, 2019], SGD-influence [Hara et al., 2019], TracIn [Pruthi et al., 2020], and TRAK [Park et al., 2023]. We focus on TracIn due to its conceptual simplicity, relative efficiency, and widespread use in recent works [Xie et al., 2024, Xia et al., 2024, Lin et al., 2024]. TracIn [Pruthi et al., **2020].** TracIn measures the cumulative change in a *target function* f(θ)
resulting from the optimization steps involving a specific training sample zi. Formally, consider training a model parameterized by θ on a training set {zi}
n i=1 P
by minimizing the empirical loss n i=1 ℓ(*θ, z*i) using stochastic gradient descent (SGD). At step j, with parameters θj , learning rate ηj , and mini-batch Bj , a first-order Taylor expansion of f(θ) around θj gives:

$f(\theta_{j})-f(\theta_{j+1})\approx\nabla_{\theta}f(\theta_{j})\,\cdot\,(\theta_{j}-\theta_{j+1})=\eta_{j}\sum_{i\in\mathcal{B}_{j}}\nabla_{\theta}f(\theta_{j})\,\cdot\,\nabla_{\theta}\ell(\theta_{j},z_{i})$.  

Accumulating these contributions over the relevant training iterations yields the TracIn score for zi:
TracIn(zi) = X

is over the relevant training iterations yet $\mathbb{Z}_{i}$) = $\sum_{j}\eta_{j}\;\nabla_{\theta}f(\theta_{j})\;\cdot\;\nabla_{\theta}\ell(\theta_{j},z_{i})$ = $j$: $z_{i}$$\in$$\mathbb{B}_{j}$

## 3 A Local Data Attribution Framework For Online Rl

Online RL presents unique challenges for data attribution, due to the way data interacts with model parameters during learning. To tackle this challenge, we introduce a *local* attribution framework tailored to *local* policy optimization inherent in online RL.

Challenges. The key feature of online RL is *the circular dependency between data and model*—
earlier experiences drive policy updates, and updated policies produce new experiences to learn from. The dependency of data on model (red arrows in Fig. 2) is unique to online RL and cannot be addressed by existing attribution methods. Current data attribution methods include Figure 2: Twofold data influence: driving policy updates, shaping future data collection.

retraining-based (e.g., Ghorbani and Zou [2019]) and *gradient-based*, with the latter further divided into *static* and *dynamic* [Hammoudeh and Lowd, 2024]. Retraining-based methods require training the model once for each of the records being evaluated, which is computationally expensive in any setting and particularly prohibitive in RL. Static methods implicitly assume model parameters are obtained from solving an empirical risk minimization problem over a fixed dataset, which is violated in the non-stationary, sequential data setting here. While dynamic methods (e.g., TracIn) capture the temporal dependencies of training data influences on model parameters, they still fail to account for this key effect of *data-model dependency*. If we compute influence scores using the original formulas from standard supervised learning, they capture only the impact on parameter updates, ignoring the extra *channel* of influences through future data generation. As a result, the scores may deviate significantly from the true influence we seek to measure. Furthermore, quantifying influences through this channel is challenging because sampling is stochastic and non-differentiable.

## 3.1 A Framework Of Local Data Attribution

Our local data attribution framework addresses the circular data-model dependency. Online RL
involves a *local policy optimization* structure, i.e., round k optimizes on a fixed buffer B(k) of onpolicy data. Thus, each round serves as a natural unit of analysis. Our framework operates at this level, examining how records in B(k)contributes to the updates from θ
(k)to θ
(k+1). This circumvents the challenges in tracing influence through the complex, cascading, and non-differentiable dependencies across the training history. Below, we detail the three key components of our framework. Entity of attribution. We consider attribution to individual training records in the rollout buffer, zi = (si, ai, ri, log πi, vi, Aˆi), collected from the environment using the current policy θ
(k). These records form the *atomic* unit used in PPO updates and provide a natural granularity for attribution. Target functions. Training data influence is usually reflected through the impact on model behaviors. Here we focus on two core aspects of an RL agent: agent action and cumulative return. Agent action: To identify records influencing the agent's decision to take a specific action a at state s, we define a straightforward target function:
f action(θ) := log πθ(a | s).

Cumulative return: We aim to understand which experience records contribute positively or negatively to the agent's ability to maximize cumulative return. Formally, the ideal quantity is the expected return J(θ) = Eτ∼πθ[R(τ )], where R(τ ) = PT −1 t=0 rt and trajectories τ are sampled by executing πθ. However, using J(θ) directly poses two fundamental challenges. *First*, unlike supervised learning with a fixed validation set, the data distribution in online RL is inherently policy-dependent. This intertwining of policy and evaluation means no fixed, universal validation set exists. *Second*, raw returns R(τ ) exhibit high variance, leading to noisy influence estimates. To address these challenges, we introduce a stable surrogate objective based on a reference policy π ref and advantage estimates Aˆref:
f return(θ) := Eτ∼πref,(s,a)∼τ hlog πθ(a | s)Aˆref(*s, a*)
i.

This target function is structurally equivalent to the objective of REINFORCE with a baseline [Sutton and Barto, 2018, Section 13.4]. By sampling from π ref, we obtain a fixed evaluation distribution; using advantage estimates significantly reduces variance compared to raw returns. Maximizing f return(θ) encourages increasing the probability of better-than-average actions and decreasing worsethan-average ones, capturing the essence of improving expected return while being tractable.

For attribution in round k, we set the reference policy π ref = πθ
(k) , i.e., the policy snapshot at the beginning of the round. This is a key design choice of our *contextual* framework, which enables

data generation Zooming in …
{(si,ai,ri, log ⇡i,vi, Aˆ ✓ i)} (0) ✓(1) **sample**
B(0)
B(1)
Buffer
…
…
Model sample train Influence from B(0)
✓(0) ✓(1)
B(1)…
B(2)
parameter updates
✓(3) ✓(2)
us to ask: For the agent at its current stage of training, which experiences will be most helpful or harmful for the next update? Unlike a fixed, off-distribution reference that may provide misleading signals due to mismatch with the agent's current state, our dynamic reference evolves with training, providing a stable and relevant basis for meaningful evaluation and attribution. Furthermore, since the training rollout buffer B(k)is collected under πθ
(k) , we can directly use it as the validation dataset.

We provide further discussions on this design choice in Sec. 4.3 and Sec. 5.1. We note that one key contribution in our framework is the design of *tractable yet meaningful* target functions, particularly f return, which can be reused in future work with alternative attribution methods.

Remark 1 (Use cases of the two target functions). *The two target functions have different use cases.*
f action is mainly for diagnosis: understanding why the agent takes a specific action at a specific state
(Sec. *4.2). On the other hand,* f return *assesses contribution to overall performance, which makes it* suitable for both analysis (Sec. 4.1) and algorithmic policy improvement (Sec. 5).

$I_{i}$ := . 
Method of attribution. We adapt TracIn to our online RL setting. For record ziin the rollout buffer
B(k), we compute its *influence score* by summing over the optimization steps j within round k:
Ii:=X
the its _infinite score_ by summing over the optimization steps $f$ with $\sum_{j:z_{i}\in\mathcal{B}_{j}^{(k)}}\left\langle\nabla_{\theta}f(\theta_{j}^{(k)}),\nabla_{\theta}\mathcal{L}^{\text{PPO}}(\theta_{j}^{(k)},z_{i})\right\rangle,\quad\text{where}f\in\left\{f^{\text{action}},f^{\text{action}}\right\}$.  
action, freturn	.
Here, ∇θf(θ
(k)
j) is the gradient of the target function evaluated at θ
(k)
j, and ∇θL
PPO(θ
(k)
j, zi) is the per-sample gradient of the PPO training objective for record zi. We also discuss two design choices in Sec. 5.1 which substantially reduce the computational and storage costs of the vanilla TracIn. Finally, we clarify how to interpret the computed influence scores. Records with positive influence benefit behavior formation or learning, whereas those with negative influence *harm* it. We refer to records with the most positive influence as *top records* and those with the most negative influence as bottom records; these terms will be used throughout the remainder of the paper.

Remark 2 (Extension to other online RL algorithms). While we focus on PPO in our study, our framework readily extends to other online RL algorithms. For on-policy methods2such as TRPO [Schulman et al., 2015] and A3C [Mnih et al., 2016], the adaptation only requires modifying the per-sample loss gradient. For offline methods like DQN [Mnih et al., 2013], we need to additionally change the target function to the Bellman error. In all cases, our attribution framework reveals whether training records help or hinder learning at the agent's current state. A key distinction is that, on-policy methods allow direct validation with current data, whereas off-policy methods require sampling fresh rollouts.

## 4 Applications Of Local Data Attribution

We now illustrate the practical value of our framework. The framework delivers fresh insights for RL researchers and practitioners, enabling key applications such as diagnosis of learning, temporal analysis of agent behavior formation, and targeted interventions during training. We demonstrate these capabilities through extensive empirical studies spanning a range of RL environments and tasks. Experimental setup. We perform evaluation on a diverse suite of RL environments—navigation (FrozenLake and MiniGrid), classic control (Acrobot and LunarLander), driving (Highway), and locomotion (BipedalWalker)—covering discrete and continuous state and action spaces with varying complexity and reward structures. We defer descriptions of environments to Appendix A.1 and PPO training setups to Appendix A.2. Our code is at https://github.com/LDAORL/LDA-ORL.

## 4.1 Diagnosis Of Learning: What Features Bottom Records?

In this section, we analyze the bottleneck that hinders learning in online RL. Specifically, we examine the bottom records for f return and uncover a consistent pattern across training rounds
(additional examples in Appendix B.1): these bottom records are characterized by inaccurate advantage estimates, echoing observations in the literature [Ilyas et al., 2018]. Fig. 3(a–b) illustrates two examples. In FrozenLake, bottom records include poor actions receiving high positive Aˆ and good actions receiving negative Aˆ. Similarly, in MiniGrid, the agent drifts from the goal but receives positive Aˆ. These instances of *misleading* advantage estimates harm the learning.

2For GRPO [Shao et al., 2024], which uses a group-relative baseline rather than value-function baseline, the target function needs to be adjusted as well.

(a) harmful records in FrozenLake (b) harmful records in MiniGrid
(c-d) analysis in FrozenLake 0 200 400 600 800 0.00 0.25 0.50 0.75 a b s(A
A)
rank correlation = -0.67 0 200 400 600 800 0.2 0.0 0.2 A

A

Record ID (sorted by decreasing influence)
Figure 3: **(a-b) Examples of bottom records**. (a) Bottom 100 records in FrozenLake at k = 5, aggregated over (s, a) for demonstration: arrow indicates action, green/red for positive/negative Aˆ. (b)
Selected records among bottom 20 in MiniGrid at k = 5: ▼–agent, ■–goal, gray area–the limited egocentric observation, yellow arrows–agent action in {turn left, turn right, forward}; all records shown are of positive Aˆ. **(c-d) These records are harmful due to their inaccurate advantage**
estimates. We sort records by decreasing influence (top on the left). (c) y axis is |A¯ − Aˆ|; points with same/opposite signs for Aˆ and A¯ colored green/red; top/bottom 20% region shaded green/red, and the intermediate in gray. (d) The product A¯ · Aˆ versus record rank, showing a strong negative correlation.

We conduct quantitative analysis to characterize what constitutes "inaccurate" advantage estimates.

We approximate the true advantage Aπ(*s, a*) using Monte Carlo (MC) rollouts from each (*s, a*),
averaging over multiple trajectories (details in Appendix B.4). We refer to this as the MC estimate, denoted by A¯, and compare it with the advantage estimate Aˆ. We perform analysis in FrozenLake.

Our analysis reveals two key aspects of "inaccuracy": (1) **Sign mismatch**: A significant proportion of bottom records exhibit opposite signs for the advantage estimate Aˆ and the MC estimate A¯ (marked by red points in Fig. 3(c)). (2) **Large magnitude errors**: These records also have large |A¯−Aˆ|. Together, sign flips and large magnitude errors generate strong but misleading learning signals. Indeed, the Spearman rank correlation [Spearman, 1904] between each record's influence and the product A¯ · Aˆ
is strongly negative (Fig. 3(d)), confirming that misaligned advantages drive harmful gradient steps.

## 4.2 Temporal Analysis Of Behavior Formation: Phase Transition Of Top Records

We investigate the reinforcement of a specific behavior (a at s), characterized by a monotonic increase in π(a|s). We track the evolution of top records w.r.t. f action across training rounds, which are critical in shaping the agent's behavior. Our analysis reveals an intriguing three-stage phase transition (Fig. 4).

Target behavior semantically similar , same , positive # semantically similar , different , **negative** #
No clear patterns same , **positive** #
action-advantage association + semantic clustering **+ influence saturation**
1. **Initial association**: Initially, top records highlight patterns based on simple action-advantage association: they manifest target action paired with positive Aˆ, or alternative actions paired with negative Aˆ (see Appendix B.2 for examples). The agent's behavior in this phase is reinforced through this naive association, largely ignoring the context of *state*. This basic association persists throughout training, even as more complex relationships are learned.

2. **Semantic clustering**: As learning progresses, the agent develops more nuanced representations.

As a result, a pattern of *semantic clustering* develops alongside the initial action-advantage association. Top records in this phase demonstrate action-advantage association *within* states semantically similar to the target state, indicating the agent has learned to generalize across similar situations.

3. **Influence saturation**: In the final phase where learning approaches convergence, influence scores for most records stabilize near zero and become dominated by noise. Due to this noise, the top records appear less structured, though the action-advantage association still persists.

We quantify these phases by analyzing the *roughness* (normalized Dirichlet energy) [Von Luxburg, 2007] of a similarity graph, a measure closely related to the graph Laplacian [Chung, 1997]. In this graph, nodes represent records, values are (L∞-normalized) influence scores ˜Ii, edge weights wij capture semantic similarity and decay with embedding distance (details in Appendix B.2). Roughness, computed as Pwij (I˜i−I˜j )
2/Pwij , is low when semantically similar records have similar influence; this captures the *clustering* effect. We track roughness across training rounds. As Fig. 4 shows, roughness remains high in Phase 1, indicating influence scores are largely uncorrelated with semantic similarity. It then significantly drops in Phase 2, representing the formation of semantically meaningful *clusters* of records with similar influences. In Phase 3, roughness remains low due to the settling of clustering, but exhibits minor fluctuations due to influence scores dominated by noise upon convergence.

## 4.3 Targeted Interventions During Training: Filtering Amplifies Policy Gain

Sec. 4.1 demonstrates that our framework can identify harmful training records, thereby opening possibilities for targeted interventions. As a sanity check, we apply a simple intervention procedure within *a single training round* to verify if removing these records yields performance gains.

Our procedure is straightforward: in round k, we identify records in B(k) with negative influence scores w.r.t. f return, remove them, and re-train the agent on the filtered dataset starting from θ
(k). Fig. 5 shows that this consistently improves performance throughout learning and across environments.

0 5 10 15 20 0.00 0.05 FrozenLake 0 5 10 15 0 100 Acrobot
∆
 
re tu rn
Round Round Figure 5: Boxplots of ∆ **return for single round interventions in two environments**; red dashed line for zero ∆. We intervene for each round *independently*. The ∆ return is computed as the difference between the test return of the model trained on the *filtered* dataset and the *original* dataset. Results are shown for 3 random seeds. Additional results can be found in Appendix B.3.

A reader may ask: how can f return be meaningful when it relies on on-policy data with potentially inaccurate advantage estimates, unlike clean validation data used in traditional data attribution for supervised learning? Despite potential noise in individual records, the aggregated signal from f return is reasonably robust. This arises from the close alignment of f return with the PPO objective: effective PPO
updates on the training buffer implies a reliable f return for attribution, enabling our intervention to clear away misleading records while retaining beneficial ones. This can be seen as *purifying* the learning signal, thereby *amplifying* the improvement achieved by PPO. More discussions are in Appendix B.3.

## 5 Iterative Influence-Based Filtering For Online Rl Training

Standard online RL algorithms typically treat all collected experiences uniformly. However, as our analysis in Sec. 4.1 has shown, some records can be harmful for learning. This likely contributes to the notorious *sample inefficiency* of online RL, a challenge widely acknowledged [Yu, 2018]. Given this, a natural question arises: *can we leverage the local data attribution framework to tackle this challenge?* We propose Iterative Influence-Based Filtering (IIF), building on the single-round interventions in Sec. 4.3. IIF filters records based on their computed influence scores, uses the resulting improved policy to sample new data, and repeats the cycle. This creates a loop for iterative refinement. We detail the algorithm below and showcase its effectiveness in traditional RL environments and RLHF for LLMs.

## 5.1 Algorithm And Designs

Algorithm 1: Iterative Influence-Based Filtering (IIF) for Online RL Define: E: environment. n: \# records in a rollout buffer. p ∈ (0, 1]: percentage of negative records to drop.

1 **Function** Update(model):
▷ Stage I: sampling 2 B ← CollectTransitions(E, model, n) ▷ collect transitions into buffer B
▷ Stage II: Filtering 3 I ← ComputeInfluence(model, B) ▷ compute influence for each record 4 Bfiltered ← DiscardBottomRecords(*B, I, p)* ▷ drop bottom records
▷ Stage III: training 5 **return** PPOUpdate(model, Bfiltered)
6 for iter = 1 to T do 7 model ← Update(model)
Alg. 1 outlines IIF. Compared to standard PPO, IIF introduces an additional step of filtering (in red) between data collection and training. We further highlight the desiderata and IIF's design choices. Sample efficiency. We aim to reduce the environment interactions required to reach a given performance level. To achieve this, IIF reuses the original rollout buffer B(k)as the validation set for influence calculation, incurring no extra sampling overhead. Furthermore, by selectively filtering bottom records, IIF accelerates learning, thus further reducing the total interactions needed. Computational cost. We aim to keep the overhead of influence calculation small. This is achieved through two design choices. (1) Instead of iterating over all intermediate checkpoints, we compute the influence scores for the entire rollout buffer B(k)in round k via ∇θf(θ
(k)), ∇θL
PPO(θ
(k), zi),
using only the initial parameter θ
(k). This saves a full training pass and excessive forward/backward calculations. (2) We implement an efficient "ghost dot product" following Wang et al. [2025a]. Final performance. We aim to improve the policy's final performance compared to standard training. IIF fulfills this through identifying and filtering out harmful records. IIF employs a hyperparameter, p, which determines the amount of records to discard. We evaluate various p's and report the best in Fig. 6. We observe that removing all negative-influence records (p = 100%) as in Wang et al. [2025a] is often suboptimal, likely due to the non-additivity of sample influence [Hu et al., 2024]. Full ablation and recommendations for the choice of p are in Appendix B.6.

## 5.2 Experiments In Traditional Rl Environments

Experimental setup. We evaluate IIF on the diverse set of RL environments introduced in Sec. 4.

Baselines: We compare IIF with standard PPO and a random filtering baseline (dropping a similar fraction of records). We additionally investigate an advantage based filtering heuristic in Appendix B.4 motivated by the characterization of bottom records in Sec. 4.1, as well as a TD error based heuristic in Appendix B.5 inspired by the Prioritized Experience Replay algorithm [Schaul et al., 2016]. Metrics: We quantify sample efficiency by the reduction in training rounds required for IIF to match standard training. For a performance level v (measured by test return), let mstd(v) and mIIF(v) be the earliest training rounds where standard training and IIF achieve performance at least v, respectively.

The reduction at v is defined as (1 − mIIF(v)/mstd(v)) × 100%. We report two metrics: SEave, the mean reduction over a list of strictly increasing performance levels reached by standard training, and SEpeak, the reduction at its peak. We measure computational cost by runtime; we similarly define RTpeak as the reduction of runtime at the performance peak. Model performance is measured by the average test return over multiple episodes. See Appendix A.2 for further details on experimental setups.

Results. Fig. 6(a) presents the test returns for each environment; Fig. 6(b) summarizes the efficiency and runtime metrics. We report a detailed breakdown of runtime in Appendix B.9. Our key findings

(a) Test returns over training rounds. IIF (Ours) Standard Random 0 10 20 30 0.00 0.25 0.50 0.75 1.00FrozenLake MiniGrid 5 10 15 20 200 150 100 Acrobot 0 10 20 30 40 0.2 0.4 0.6 Return 0 5 10 15 20 25 15 20 25 Highway 0 20 40 60 80 100 200 100 0 100 200LunarLander 0 50 100 150 100 0 100 200 BipedalWalker Retur n

Round Round Round

(b) Improvement in sample efficiency and runtime

FrozenLake Acrobot MiniGrid Highway LunarLander BipedalWalker

SEave (↑) 34.0% ± 2.0% 36.7% ± 6.5% 65.8% ± 3.3% 37.7% ± 6.1% 26.0% ± 1.8% 31.0% ± 8.7%

SEpeak (↑) 19.2% ± 5.9% 48.5% ± 0.8% 61.7% ± 4.1% 55.1% ± 2.9% 39.7% ± 3.7% 26.2% ± 8.0% RTpeak (↑) 29.5% ± 2.9% 55.2% ± 1.0% 69.1% ± 1.7% 59.9% ± 0.7% 44.9% ± 2.5% 29.2% ± 0.7%

Figure 6: (a) **Test returns over rounds for IIF vs. baselines.** IIF speeds up learning and improves performance. Results are averaged over 5 random seeds. For Acrobot, we omit early rounds where returns rise from -500 to -200 for better visualization. (b) **Sample efficiency and runtime metrics.**

are summarized as follows: 1) IIF achieves substantial sample efficiency gains, showing a 20-67% reduction in training rounds required to match the standard training performance across environments. 2) The computational overhead of IIF is negligible, and offset by the reduced optimization time (see Appendix B.9), leading to significant improvement in runtime. 3) IIF's final performance exceeds standard training in almost every environment. These observed gains stem from effective data attribution rather than mere data reduction: random filtering performs significantly worse than original training.

## 5.3 Extending Iif To Rlhf For Large Language Models

As the final part, we apply IIF to improve Reinforcement Learning from Human Feedback (RLHF).3 Compared to standard PPO, RLHF introduces several key differences. First, the atomic unit shifts
(a) Training Reward (↑)
from state-action records to prompt-generation pairs, where each generation is a *trajectory* (or sequence) of tokens. Second, RLHF incorporates *dual* reward sources: a reward model evaluating the final generation, and a per-token KL divergence penalty to constrain deviation from a reference model. To accommodate these differences, we adapt IIF for RLHF by employing a sequence-level objective:
f seq(θ) = Ex∼Dval,y∼πref(·|x)
hlog πθ(y | x)Aˆref
−1(x, y)
i, where x is a prompt drawn from the validation set Dval P 
, y the generation, log πθ(y|x) =
ilog πθ(yi|x, y0*, . . . , y*i−1) the log-probability of the sequence y given x, and Aˆref
−1the advantage estimate at the last token. This objective emphasizes the reward model's feedback at the last token. Experimental results: toxicity mitigation. We consider the task of detoxifying LLMs using RLHF [Hugging Face, 2023], using gpt-neo-2.7B [Black et al., 2021] as our base

0 20 40 60 80 100 3.0 3.5 4.0 4.5 IIF (Ours) Standard Random Round
(b) Test toxicity (↓) on a different test set, evaluated using a different toxicity detector.

0 20 40 60 80 100 0.05 0.10 0.15 Round
model. Fig. 7 illustrates the effectiveness of our approach. We defer detailed experimental setups to Appendix A.3 and additional results (e.g., comparisons with using the target function f return) in Appendix B.11.

We further highlight IIF's substantial gains in *computational efficiency*. IIF filters out negativeinfluence records (∼50% of all), effectively *halving* the optimization time per round. Furthermore, IIF accelerates learning, requiring less than *half* the number of rounds to surpass standard training, significantly enhancing sample efficiency. The overhead of influence calculation is minimal. Collectively, these factors result in an ∼4× reduction in total runtime (detailed breakdown in Appendix B.12).

## 6 Related Work

Interpretability in reinforcement learning has become a central research theme because real-world deployment requires agents that are trustworthy and reliable [Arulkumaran et al., 2017, Sutton and Barto, 2018, Milani et al., 2024, Cheng et al., 2025]. Early studies emphasize *feature*-level explanations: they highlight regions of the observation space that most influence an agent's decisions, often through saliency maps or attention heatmaps [Zahavy et al., 2016, Greydanus et al., 2018, Mott et al., 2019, Atrey et al., 2020, Puri et al., 2020]. A complementary thread seeks *policy*-level explanations. These works approximate learned policies with human-interpretable rules [Verma et al., 2018, Soares et al., 2020], design transparent architectures [Topin et al., 2021, Demircan et al., 2025], or dissect reward functions to clarify action choices [Juozapaitis et al., 2019, Liu and Zhu, 2025]. More recently, researchers have probed how entire training *trajectories* shape behavior [Deshmukh et al., 2023].

Zooming in further, identifying critical *states* offers a finer-grained view of decision making. Several approaches address offline settings [Guo et al., 2021, Yu et al., 2023, Liu et al., 2023, Rishav et al., 2025]. Closer to our focus are methods that target online RL such as lazy-MDP [Jacq et al., 2022], StateMask [Cheng et al., 2023] and RICE [Cheng et al., 2024]. Lazy-MDP augments the action space with a "lazy" action and penalizes non-lazy choices; states where the agent still acts are interpreted as important. However, this approach requires modifying the training pipeline. StateMask and RICE
train an auxiliary mask network alongside the policy, forcing random actions in selected states while keeping returns roughly unchanged; masked states are deemed non-critical. Nevertheless, these methods crucially rely on the policy being sufficiently developed, which limits their applicability when agents are still learning in complex environments.

Moving beyond these constraints, our work introduces data attribution as a principled lens for interpretability in online RL. This approach closes a key methodological gap in the literature, delivers fresh insights for RL researchers and practitioners, and informs more efficient and effective training.

## 7 Conclusion And Limitations

This work pioneers data attribution for online RL by introducing a local attribution framework that addresses the circular dependency between data and model. The framework provides finegrained insights into how training records shape model behaviors and offers a principled approach to enhancing the interpretability, efficiency, and effectiveness of online RL. We discuss a few limitations. Optimizers. Our framework leverages TracIn, which is designed for SGD [Hammoudeh and Lowd, 2024]. However, adaptive optimizers like Adam [Kingma and Ba, 2015] are prevalent in modern RL [Asadi et al., 2023] and LLMs [Zhao et al., 2025]. In this work, we follow Wang et al. [2025b]
and employ SGD as a proxy for Adam. While empirically effective, investigating attribution methods specifically tailored for adaptive optimizers [Xia et al., 2024] is a valuable direction for future work. RL algorithms. Extending our framework to other online RL algorithms, particularly those used for LLMs like GRPO [Shao et al., 2024, DeepSeek-AI, 2025, Yu et al., 2025], is a promising avenue. Technically, our framework should generalize provided the attribution entity and per-sample gradients are well-defined. On the application side, leveraging attribution as a principled tool for improving LLM reasoning offers an intriguing alternative to existing data selection methods [Li et al., 2025, Shi et al., 2025, Xu et al., 2025, Wang et al., 2025c] that are largely based on heuristics.

Counterfactual interpretation. Finally, our local attribution framework, while powerful, lacks a clear counterfactual interpretation. This limitation partly stems from TracIn itself, but primarily from the fundamental difficulty of tracking causal effects across the circular data-model dependency inherent in online RL, as discussed in Sec. 3. We encourage future work to tackle this open problem.

## Acknowledgements

We thank the anonymous NeurIPS 2025 reviewers for their constructive feedback. YH thanks Haozhe Si for assistance in setting up an NVIDIA instance. YH and JM thank Huazheng Wang and Kaiqing Zhang for helpful discussions on variance reduction. YH and HZ are partially supported by NSF IIS Grant No.2416897 and the NVIDIA Academic Grant Program. HZ also acknowledges support from a Google Research Scholar Award. Nan Jiang acknowledges funding support from NSF CNS-2112471, NSF CAREER IIS-2141781, Google Scholar Award, and Sloan Fellowship.

## References

O. M. Andrychowicz, B. Baker, M. Chociej, R. Jozefowicz, B. McGrew, J. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, et al. Learning dexterous in-hand manipulation. The International Journal of Robotics Research, 39(1):3–20, 2020.

K. Arulkumaran, M. P. Deisenroth, M. Brundage, and A. A. Bharath. Deep reinforcement learning:
A brief survey. *IEEE Signal Processing Magazine*, 34(6):26–38, 2017.

K. Asadi, R. Fakoor, and S. Sabach. Resetting the optimizer in deep rl: An empirical study. Advances in Neural Information Processing Systems, 36:72284–72324, 2023.

A. Atrey, K. Clary, and D. Jensen. Exploratory not explanatory: Counterfactual analysis of saliency maps for deep reinforcement learning. In *International Conference on Learning Representations*, 2020. URL https://openreview.net/forum?id=rkl3m1BFDB.

S. Bae, J. Hong, M. Y. Lee, H. Kim, J. Nam, and D. Kwak. Online difficulty filtering for reasoning oriented reinforcement learning. *arXiv preprint arXiv:2504.03380*, 2025.

C. Berner, G. Brockman, B. Chan, V. Cheung, P. D˛ebiak, C. Dennison, D. Farhi, Q. Fischer, S. Hashme, C. Hesse, et al. Dota 2 with large scale deep reinforcement learning. arXiv preprint arXiv:1912.06680, 2019.

S. Black, G. Leo, P. Wang, C. Leahy, and S. Biderman. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow, Mar. 2021. URL https://doi.org/10.5281/ zenodo.5297715. If you use this software, please cite it using these metadata.

T. A. Chang, D. Rajagopal, T. Bolukbasi, L. Dixon, and I. Tenney. Scalable influence and fact tracing for large language model pretraining. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=gLa96FlWwn.

Z. Cheng, X. Wu, J. Yu, W. Sun, W. Guo, and X. Xing. Statemask: Explaining deep reinforcement learning through state mask. *Advances in Neural Information Processing Systems*, 36:62457–62487, 2023.

Z. Cheng, X. Wu, J. Yu, S. Yang, G. Wang, and X. Xing. RICE: Breaking through the training bottlenecks of reinforcement learning with explanation. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=PKJqsZD5nQ.

Z. Cheng, J. Yu, and X. Xing. A survey on explainable deep reinforcement learning. arXiv preprint arXiv:2502.06869, 2025.

M. Chevalier-Boisvert, B. Dai, M. Towers, R. de Lazcano, L. Willems, S. Lahlou, S. Pal, P. S. Castro, and J. Terry. Minigrid & miniworld: Modular & customizable reinforcement learning environments for goal-oriented tasks. *CoRR*, abs/2306.13831, 2023.

F. R. Chung. *Spectral graph theory*, volume 92. American Mathematical Soc., 1997. N. Das, S. Chakraborty, A. Pacchiano, and S. R. Chowdhury. Active preference optimization for sample efficient RLHF. In ICML 2024 Workshop on Theoretical Foundations of Foundation Models, 2024. URL https://openreview.net/forum?id=uSCvfYNn0s.

DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

C. Demircan, T. Saanum, A. K. Jagadish, M. Binz, and E. Schulz. Sparse autoencoders reveal temporal difference learning in large language models. In *The Thirteenth International Conference* on Learning Representations, 2025. URL https://openreview.net/forum?id=2tIyA5cri8.

J. Deng, Y. Hu, P. Hu, T.-W. Li, S. Liu, J. T. Wang, D. Ley, Q. Dai, B. Huang, J. Huang, C. Jiao, H. A. Just, Y. Pan, J. Shen, Y. Tu, W. Wang, X. Wang, S. Zhang, S. Zhang, R. Jia, H. Lakkaraju, H. Peng, W. Tang, C. Xiong, J. Zhao, H. Tong, H. Zhao, and J. W. Ma. A Survey of Data Attribution: Methods, Applications, and Evaluation in the Era of Generative AI. Aug. 2025. URL https://hal.science/hal-05230469.

S. V. Deshmukh, A. Dasgupta, B. Krishnamurthy, N. Jiang, C. Agarwal, G. Theocharous, and J. Subramanian. Explaining RL decisions with trajectories. In *The Eleventh International Conference on* Learning Representations, 2023. URL https://openreview.net/forum?id=5Egggz1q575.

G. Dulac-Arnold, D. Mankowitz, and T. Hester. Challenges of real-world reinforcement learning, 2019. URL https://openreview.net/forum?id=S1xtR52NjN.

S. Gehman, S. Gururangan, M. Sap, Y. Choi, and N. A. Smith. Realtoxicityprompts: Evaluating neural toxic degeneration in language models. *arXiv preprint arXiv:2009.11462*, 2020.

A. Ghorbani and J. Zou. Data shapley: Equitable valuation of data for machine learning. In International conference on machine learning, pages 2242–2251. PMLR, 2019.

S. Greydanus, A. Koul, J. Dodge, and A. Fern. Visualizing and understanding atari agents. In International conference on machine learning, pages 1792–1801. PMLR, 2018.

R. Grosse, J. Bae, C. Anil, N. Elhage, A. Tamkin, A. Tajdini, B. Steiner, D. Li, E. Durmus, E. Perez, et al. Studying large language model generalization with influence functions. *arXiv preprint* arXiv:2308.03296, 2023.

W. Guo, X. Wu, U. Khan, and X. Xing. Edge: Explaining deep reinforcement learning policies.

Advances in Neural Information Processing Systems, 34:12222–12236, 2021.

Z. Hammoudeh and D. Lowd. Training data influence analysis and estimation: A survey. Machine Learning, 113(5):2351–2403, 2024.

S. Hara, A. Nitanda, and T. Maehara. Data cleansing for models trained with sgd. Advances in Neural Information Processing Systems, 32, 2019.

P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger. Deep reinforcement learning that matters. In *Proceedings of the AAAI conference on artificial intelligence*, volume 32, 2018.

E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA:
Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

Y. Hu, P. Hu, H. Zhao, and J. Ma. Most influential subset selection: Challenges, promises, and beyond. *Advances in Neural Information Processing Systems*, 37:119778–119810, 2024.

Hugging Face. Detoxifying a language model using ppo. https://huggingface.co/docs/trl/
en/detoxifying_a_lm, 2023. TRL documentation (v0.17.0), accessed May 8, 2025.

A. Ilyas, L. Engstrom, S. Santurkar, D. Tsipras, F. Janoos, L. Rudolph, and A. Madry. A closer look at deep policy gradients. *arXiv preprint arXiv:1811.02553*, 2018.

A. Jacq, J. Ferret, O. Pietquin, and M. Geist. Lazy-mdps: Towards interpretable rl by learning when to act. In *Proceedings of the International Foundation for Autonomous Agents and Multiagent* Systems, pages 669–677, 2022.

Z. Juozapaitis, A. Koul, A. Fern, M. Erwig, and F. Doshi-Velez. Explainable reinforcement learning via reward decomposition. In *IJCAI/ECAI Workshop on explainable artificial intelligence*, 2019.

D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In The Third International Conference on Learning Representations, 2015.

P. W. Koh and P. Liang. Understanding black-box predictions via influence functions. In International conference on machine learning, pages 1885–1894. PMLR, 2017.

E. Leurent. An environment for autonomous driving decision-making. https://github.com/
eleurent/highway-env, 2018.

X. Li, H. Zou, and P. Liu. Limr: Less is more for rl scaling. *arXiv preprint arXiv:2502.11886*, 2025.

H. Lin, J. Long, Z. Xu, and W. Zhao. Token-wise influential training data retrieval for large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 841–860, 2024.

H. Liu, M. Zhuge, B. Li, Y. Wang, F. Faccio, B. Ghanem, and J. Schmidhuber. Learning to identify critical states for reinforcement learning from videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1955–1965, 2023.

S. Liu and M. Zhu. UTILITY: Utilizing explainable reinforcement learning to improve reinforcement learning. In *The Thirteenth International Conference on Learning Representations*, 2025. URL
https://openreview.net/forum?id=Tk1VQDadfL.

S. Milani, N. Topin, M. Veloso, and F. Fang. Explainable reinforcement learning: A survey and comparative review. *ACM Computing Surveys*, 56(7):1–36, 2024.

V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller.

Playing atari with deep reinforcement learning. *arXiv preprint arXiv:1312.5602*, 2013.

V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al. Human-level control through deep reinforcement learning. *nature*, 518(7540):529–533, 2015.

V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu.

Asynchronous methods for deep reinforcement learning. In International conference on machine learning, pages 1928–1937. PmLR, 2016.

A. Mott, D. Zoran, M. Chrzanowski, D. Wierstra, and D. Jimenez Rezende. Towards interpretable reinforcement learning using attention augmented agents. Advances in neural information processing systems, 32, 2019.

W. Muldrew, P. Hayes, M. Zhang, and D. Barber. Active preference learning for large language models. In *Forty-first International Conference on Machine Learning*, 2024. URL https:
//openreview.net/forum?id=CTgEV6qgUy.

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

S. M. Park, K. Georgiev, A. Ilyas, G. Leclerc, and A. Madry. Trak: Attributing model behavior at scale. In *International Conference on Machine Learning*, pages 27074–27113. PMLR, 2023.

G. Pruthi, F. Liu, S. Kale, and M. Sundararajan. Estimating training data influence by tracing gradient descent. *Advances in Neural Information Processing Systems*, 33:19920–19930, 2020.

N. Puri, S. Verma, P. Gupta, D. Kayastha, S. Deshmukh, B. Krishnamurthy, and S. Singh. Explain your move: Understanding agent actions using specific and relevant feature attribution. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id= SJgzLkBKPB.

A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann. Stable-baselines3:
Reliable reinforcement learning implementations. *Journal of Machine Learning Research*, 22 (268):1–8, 2021. URL http://jmlr.org/papers/v22/20-1364.html.

R. Rishav, S. Nath, V. Michalski, and S. E. Kahou. Behaviour discovery and attribution for explainable reinforcement learning. *arXiv preprint arXiv:2503.14973*, 2025.

A. E. Sallab, M. Abdou, E. Perot, and S. Yogamani. Deep reinforcement learning framework for autonomous driving. *arXiv preprint arXiv:1704.02532*, 2017.

T. Schaul, J. Quan, I. Antonoglou, and D. Silver. Prioritized experience replay. In International Conference on Learning Representations (ICLR), 2016. URL http://arxiv.org/abs/1511. 05952.

J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015.

J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. High-dimensional continuous control using generalized advantage estimation. In Proceedings of the International Conference on Learning Representations (ICLR), 2016.

J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.

Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al.

Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv* preprint arXiv:2402.03300, 2024.

Y. Shen, H. Sun, and J.-F. Ton. Reviving the classics: Active reward modeling in large language model alignment. *arXiv preprint arXiv:2502.04354*, 2025.

T. Shi, Y. Wu, L. Song, T. Zhou, and J. Zhao. Efficient reinforcement finetuning via adaptive curriculum learning. *arXiv preprint arXiv:2504.05520*, 2025.

D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van Den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, et al. Mastering the game of go with deep neural networks and tree search. *nature*, 529(7587):484–489, 2016.

E. Soares, P. P. Angelov, B. Costa, M. P. G. Castro, S. Nageshrao, and D. Filev. Explaining deep learning models through rule-based approximation and visualization. *IEEE Transactions on Fuzzy* Systems, 29(8):2399–2407, 2020.

C. Spearman. The proof and measurement of association between two things. The American Journal of Psychology, 15(1):72–101, 1904.

R. S. Sutton and A. G. Barto. *Reinforcement Learning: An Introduction*. The MIT Press, Cambridge, MA, second edition, 2018. URL http://incompleteideas.net/book/the-book-2nd.html.

N. Topin, S. Milani, F. Fang, and M. Veloso. Iterative bounding mdps: Learning interpretable policies via non-interpretable methods. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 35, pages 9923–9931, 2021.

M. Towers, A. Kwiatkowski, J. Terry, J. U. Balis, G. D. Cola, T. Deleu, M. Goulão, A. Kallinteris, M. Krimmel, A. KG, R. Perez-Vicente, A. Pierré, S. Schulhoff, J. J. Tai, H. Tan, and O. G.

Younis. Gymnasium: A standard interface for reinforcement learning environments, 2024. URL
https://arxiv.org/abs/2407.17032.

A. Verma, V. Murali, R. Singh, P. Kohli, and S. Chaudhuri. Programmatically interpretable reinforcement learning. In *International conference on machine learning*, pages 5045–5054. PMLR, 2018.

B. Vidgen, T. Thrush, Z. Waseem, and D. Kiela. Learning from the worst: Dynamically generated datasets to improve online hate detection. In ACL, 2021.

U. Von Luxburg. A tutorial on spectral clustering. *Statistics and computing*, 17:395–416, 2007. L. von Werra, Y. Belkada, L. Tunstall, E. Beeching, T. Thrush, N. Lambert, S. Huang, K. Rasul, and Q. Gallouédec. Trl: Transformer reinforcement learning. https://github.com/huggingface/ trl, 2020.

H. Wang, Z. Wu, and J. He. Fairif: Boosting fairness in deep learning via influence functions with validation set sensitive attributes. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining, pages 721–730, 2024.

J. T. Wang, P. Mittal, D. Song, and R. Jia. Data shapley in one training run. In The Thirteenth International Conference on Learning Representations, 2025a. URL https://openreview.

net/forum?id=HD6bWcj87Y.

J. T. Wang, D. Song, J. Zou, P. Mittal, and R. Jia. Capturing the temporal dependence of training data influence. In *The Thirteenth International Conference on Learning Representations*, 2025b. URL https://openreview.net/forum?id=uHLgDEgiS5.

S.-Y. Wang, A. A. Efros, J.-Y. Zhu, and R. Zhang. Evaluating data attribution for text-to-image models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 7192–7203, 2023.

Y. Wang, Q. Yang, Z. Zeng, L. Ren, L. Liu, B. Peng, H. Cheng, X. He, K. Wang, J. Gao, W. Chen, S. Wang, S. S. Du, and Y. Shen. Reinforcement learning for reasoning in large language models with one training example. *arXiv preprint arxiv:2504.20571*, 2025c.

M. Xia, S. Malladi, S. Gururangan, S. Arora, and D. Chen. LESS: Selecting influential data for targeted instruction tuning. In *Forty-first International Conference on Machine Learning*, 2024. URL https://openreview.net/forum?id=PG5fV50maR.

T. Xie, H. Li, A. Bai, and C.-J. Hsieh. Data attribution for diffusion models: Timestep-induced bias in influence estimation. *Transactions on Machine Learning Research*, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=P3Lyun7CZs.

Y. E. Xu, Y. Savani, F. Fang, and Z. Kolter. Not all rollouts are useful: Down-sampling rollouts in llm reinforcement learning. *arXiv preprint arXiv:2504.13818*, 2025.

J. Yu, W. Guo, Q. Qin, G. Wang, T. Wang, and X. Xing. {*AIRS*}: Explanation for deep reinforcement learning based security applications. In *32nd USENIX Security Symposium (USENIX Security 23)*, pages 7375–7392, 2023.

Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, T. Fan, G. Liu, L. Liu, X. Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. *arXiv preprint arXiv:2503.14476*, 2025.

Y. Yu. Towards sample efficient reinforcement learning. In *IJCAI*, pages 5739–5743, 2018. T. Zahavy, N. Ben-Zrihem, and S. Mannor. Graying the black box: Understanding dqns. In International conference on machine learning, pages 1899–1908. PMLR, 2016.

R. Zhao, D. Morwani, D. Brandfonbrener, N. Vyas, and S. M. Kakade. Deconstructing what makes a good optimizer for autoregressive language models. In *The Thirteenth International Conference on* Learning Representations, 2025. URL https://openreview.net/forum?id=zfeso8ceqr.

X. Zheng, T. Pang, C. Du, J. Jiang, and M. Lin. Intriguing properties of data attribution on diffusion models. In *The Twelfth International Conference on Learning Representations*, 2024. URL https://openreview.net/forum?id=vKViCoKGcB.

## Neurips Paper Checklist

1. **Claims**
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: We have tried our best to ensure that the abstract and introduction accurately reflect the paper's contributions and scope. Guidelines:
- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: The limitations are discussed in Sec. 7. Guidelines:
- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper. - The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach.

For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. **Theory Assumptions And Proofs**

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [NA] Justification: The paper does not include theoretical results. Guidelines:
- The answer NA means that the paper does not include theoretical results. - All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

- All assumptions should be clearly stated or referenced in the statement of any theorems. - The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. **Experimental Result Reproducibility**

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We provide detailed information on the experimental setups in Appendix A. Our code is also also publicly available at https://github.com/LDAORL/LDA-ORL. Guidelines:
- The answer NA means that the paper does not include experiments.

- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways.

For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. **Open Access To Data And Code**

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: Our code is publicly available at https://github.com/LDAORL/LDA-ORL. Guidelines:
- The answer NA means that paper does not include experiments requiring code.

- Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.

- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. **Experimental Setting/Details**

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: The details of the experiments are discussed in Appendix A. Guidelines:
- The answer NA means that the paper does not include experiments. - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. **Experiment Statistical Significance**

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: We use 3 random seeds for all experiments; we include error bars in all reported results (Figs. 5 to 7) in the main paper as well as more results in Appendix B (Figs. 11 to 14 and tables 2, 5 and 6). Guidelines:
- The answer NA means that the paper does not include experiments.

- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors). - It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. **Experiments Compute Resources**

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: The information on the compute resources is provided in Appendix C. Guidelines:
- The answer NA means that the paper does not include experiments. - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. **Code Of Ethics**

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: We have reviewed the NeurIPS Code of Ethics and confirm that the research conducted in this paper adheres to its principles. Guidelines:
- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. **Broader Impacts**

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: The paper conducts fundamental research aimed at understanding the role of data in online RL, and leverages this understanding to improve RL training. We do not anticipate any immediate societal impact. Guidelines:
- The answer NA means that there is no societal impact of the work performed. - If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. **Safeguards**

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: The paper does not pose such risks. Guidelines:
- The answer NA means that the paper poses no such risks. - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. **Licenses For Existing Assets**

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: We have cited the RL environments / datasets, models, code frameworks, and included their licenses in Appendix A. Guidelines:
- The answer NA means that the paper does not use existing assets. - The authors should cite the original paper that produced the code package or dataset. - The authors should state which version of the asset is used and, if possible, include a URL.

- The name of the license (e.g., CC-BY 4.0) should be included for each asset. - For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.