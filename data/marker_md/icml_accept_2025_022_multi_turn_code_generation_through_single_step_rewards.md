# Multi-Turn Code Generation Through Single-Step Rewards

Arnav Kumar Jain \* 1 2 Gonzalo Gonzalez-Pumariega \* 3 Wayne Chen <sup>3</sup> Alexander M Rush <sup>3</sup> Wenting Zhao † <sup>3</sup> Sanjiban Choudhury † <sup>3</sup>

# Abstract

We address the problem of code generation from multi-turn execution feedback. Existing methods either generate code without feedback or use complex, hierarchical reinforcement learning to optimize multi-turn rewards. We propose a simple yet scalable approach, µCODE, that solves multi-turn code generation using only single-step rewards. Our key insight is that code generation is a one-step recoverable MDP, where the correct code can be recovered from any intermediate code state in a single turn. µCODE iteratively trains both a generator to provide code solutions conditioned on multi-turn execution feedback and a verifier to score the newly generated code. Experimental evaluations show that our approach achieves significant improvements over the stateof-the-art baselines. We provide analysis of the design choices of the reward models and policy, and show the efficacy of µCODE at utilizing the execution feedback. Our code is available [here.](https://github.com/portal-cornell/muCode)

# 1. Introduction

Software engineers often iteratively refine their code based on execution errors. A common strategy for machine code generation is thus to repair code using execution feedback at test time [\(Chen et al.,](#page-9-0) [2024;](#page-9-0) [Wang et al.,](#page-10-0) [2024b;](#page-10-0) [Zhao](#page-11-0) [et al.,](#page-11-0) [2024\)](#page-11-0). However, prompting alone is insufficient as it cannot teach how to recover from all possible errors within a limited context.

We need to train models that can learn from execution feedback during training. Existing approaches fall into either single-turn or multi-turn settings. In the single-turn setting, methods either train without execution feedback [\(Zelikman](#page-11-1) [et al.,](#page-11-1) [2022\)](#page-11-1) or perform one-step corrections [\(Welleck et al.,](#page-10-1)

[2023;](#page-10-1) [Ni et al.,](#page-10-2) [2024\)](#page-10-2). However, these struggle to iteratively correct errors over multiple turns. Multi-turn approaches, on the other hand, rely on complex reinforcement learning (RL) [\(Gehring et al.,](#page-9-1) [2024a;](#page-9-1) [Kumar et al.,](#page-9-2) [2024b;](#page-9-2) [Zhou et al.,](#page-11-2) [2024\)](#page-11-2) to optimize long-term rewards. While effective in principle, these methods suffer from sparse learning signals which makes learning inefficient.

Our key insight is that code generation is *a one-step recoverable Markov Decision Process (MDP), implying that the correct code can be recovered from any intermediate state in a single step*. This allows us to greedily maximize a onestep reward instead of relying on complex multi-step reward optimization. As a result, this reduces the problem from reinforcement learning, which requires exploration and credit assignment, to imitation learning, where the model simply learns to mimic correct code, leading to a more stable and efficient training process.

We propose µCODE, a simple and scalable approach for multi-turn code generation from execution feedback. During training, µCODE follows an *expert iteration* [\(Anthony](#page-8-0) [et al.,](#page-8-0) [2017\)](#page-8-0) framework with a *local search expert*, enabling iterative improvement of both the generator and the expert. The process begins by rolling out the current code generator to collect interaction data with execution feedback. A single-step verifier is then trained on this data and utilized to guide a local search expert in refining the code and generating training labels. Finally, the generator is fine-tuned using these labels. Given recent trends of test-time scaling in generating high quality solutions [\(Brown et al.,](#page-9-3) [2024;](#page-9-3) [Snell et al.,](#page-10-3) [2024;](#page-10-3) [Wu et al.,](#page-11-3) [2024\)](#page-11-3), µCODE also uses the learned verifier for inference-time scaling. Here, µCODE samples N trajectories; at each step, µCODE picks the best code solution ranked by the learned verifier.

The key contributions of this work are as follows:

- 1. A novel framework, µCODE, for training code generators and verifiers through multi-turn execution feedback. We add theoretical analysis of performance bounds using the property of one-step recoverability for this task.
- 2. We propose a *multi-turn Best-of-N (BoN) approach* for inference-time scaling and present benefits of learned verifier to select the code solution at each turn.
- 3. Our approach µCODE outperforms leading multi-turn

<sup>\*</sup>Equal contribution <sup>1</sup>Mila- Quebec AI Institute <sup>2</sup>Universite de ´ Montreal ´ <sup>3</sup>Cornell University. Correspondence to: Arnav <arnavkumar.jain@mila.quebec>, Gonzalo <gg387@cornell.edu>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

![](_page_1_Diagram_1.jpeg)

fier In multi-turn code generation, an agent iteratively refines a program to maximize its correctness on private test cases. Given an initial problem prompt x, at each turn t, the agent generates a complete code snippet y<sup>t</sup> and executes it on a set of public tests. The outcomes o<sup>t</sup> from these tests serve as observations that guide subsequent refinements. This process continues until the agent generates a code snippet y<sup>t</sup> that passes all public tests, at which point the episode terminates, or until the maximum number of turns T is reached without success. The first successful code, yt, is then evaluated on private tests to compute the correctness score C(x, yt) ∈ {0, 1}.

Figure 1. (a) We define the task of multi-turn code generation where for an initial problem x, the generator π<sup>θ</sup> provides a solution y1. This solution is evaluated with the public test to get execution feedback o1. At a turn t, the generator is conditioned on the history to generate solution y<sup>t</sup> ∼ πθ(.|x, y<t, o<t). The rollout ends when the turn limit is reached or the public tests pass upon which the solution is executed on private tests. Since, the agents can generate the optimal solution at any turn, this is a 1-step recoverable process. (b) Training loop of our method µCODE – which comprises of a generator and a learned verifier. During each iteration, rollouts are collected using π<sup>θ</sup> and we train a verifier R<sup>ϕ</sup> to rank candidate solutions for a prompt. The verifier R<sup>ϕ</sup> is then used to construct a local expert and relabel the collected rollouts. Lastly, the generator is fine-tuned with this expert dataset.

approaches on MBPP [\(Austin et al.,](#page-9-4) [2021\)](#page-9-4), HumanEval [\(Chen et al.,](#page-9-5) [2021\)](#page-9-5) and CodeContests [\(Li et al.,](#page-9-6) [2022a\)](#page-9-6) benchmarks. Our ablations show that learned verifiers aid in learning better generators and show promising scaling law trends with higher inference budgets.

# 2. Background

We model this as a Markov Decision Process (MDP),

where the state is the interaction history s<sup>t</sup> = {x, y1, o1, . . . , yt−1, ot−1} where s<sup>1</sup> = {x}, and the action is the code snippet yt. The oracle reward is defined as R(st, yt) = R(x, yt) = C(x, yt) if y<sup>t</sup> passes all public and private tests (terminating the episode), or 0 otherwise.

During training, given a dataset of problem prompts D, the goal is to find a generator πθ(yt|x, y1, o1, . . . , yt−1, ot−1), that maximizes the cumulative discounted reward R(x, yt):

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y_t \sim \pi_\theta(\cdot | s_t)} \left[ \sum_{t=1}^T \gamma^t R(x, y_t) \right], \quad (1)$$

where γ ∈ [0, 1) is the discount factor. As shown in Eq. [1,](#page-1-0) the objective optimizes for a policy to generate the correct solution with as few turns as possible. However, at any step t, the agent can generate the correct code solution y<sup>t</sup> = y ⋆ such that C(x, y<sup>⋆</sup> ) = 1 (as shown in Fig. [1](#page-1-1) (a)) – a *one-step recoverable* process. In the following section, we describe µCODE, a simple and scalable framework that leverages the one-step recoverability and reduces the problem of reinforcement learning to imitation learning.

Algorithm 1 µCODE: Training

- input Initial generator π0, multi-turn code environment E, and max iterations M 1: for iteration i = 1 . . . M do 2: Rollout generator π<sup>θ</sup> in multi-turn environment E to collect datapoints D<sup>i</sup> ← {(x, st, yt, ot))} 3: Aggregate data D ← D ∪ D<sup>i</sup> 4: Train a verifier R<sup>i</sup> ϕ (x, y) on D 5: Construct a local search expert using verifier π i ⋆
- (x) = arg maxy∈D(x) βOR(x, y) + βLRϕ(x, y) 6: Relabel data D with π i ⋆
- (x) to get D<sup>i</sup> ⋆ 7: Train π i <sup>θ</sup> with fine-tuning (FT) on D<sup>i</sup> ⋆ 8: end for output Best generator π<sup>θ</sup> and verifier R<sup>ϕ</sup>

# 3. µCODE: Multi-turn Code Generation

We propose µCODE, a simple and scalable algorithm for multi-turn code generation using execution feedback. µCODE follows an *expert iteration* [\(Anthony et al.,](#page-8-0) [2017\)](#page-8-0) framework with a *local search expert*. µCODE iteratively trains two components – a *learned verifier* R<sup>ϕ</sup> to score code snippets (Section [3.2\)](#page-2-0), and a *generator* π<sup>θ</sup> to imitate local search with the verifier (Section [3.3\)](#page-2-1). This iterative process allows the generator and expert to bootstrap off each other, leading to continuous improvement. At inference time, both the generator and verifier are used as BoN search to select and execute code (Section [3.4\)](#page-3-0). Finally, we analyze the performance of µCODE in Section [3.5.](#page-3-1)

# 3.1. The µCODE Algorithm

Algorithm [1](#page-2-2) presents the iterative training procedure. At an iteration i, the current generator π<sup>θ</sup> is rolled out in the multi-turn code environment E to generate interaction data D<sup>i</sup> ← {(x, st, yt, rt)}. Every turn t in D<sup>i</sup> includes the prompt x, interaction history st, code generated y<sup>t</sup> and the correctness score from the oracle verifier r<sup>t</sup> = R(x, yt). This data is then aggregated D ← D ∪ D<sup>i</sup> . The learned verifier R<sup>i</sup> ϕ is trained on the aggregated data D. An expert is created using R<sup>i</sup> ϕ to perform local search to find the optimal action π i ⋆ (x) = arg maxy∈D(x) R<sup>i</sup> ϕ (x, y), where D(x) are all the code completions for a given prompt x. The expert π i ⋆ (x) relabels the data D with the optimal action. The generator π i θ is then trained via fine-tuning (FT) on D. This process iterates M times, and the best generator and verifier pair on the validation dataset are returned.

### 3.2. Training Verifier

The learned verifier provides dense scores to code solutions for a given problem. At train time, this is used by the expert to perform local search to obtain optimal code. At inference time, the verifier is used for multi-turn BoN [\(3.4\)](#page-3-0) for efficient search. The learned verifier has two distinct advantages over process reward functions typically used in multi-turn RL: (1) It is conditioned only on the initial prompt and the current solution, and is not dependent on previous states (2) It is trained via supervised learning on oracle reward labels. We explore two different losses:

Binary Cross-Entropy loss (BCE): The nominal way to train the verifier is to directly predict the oracle reward labels [\(Cobbe et al.,](#page-9-7) [2021\)](#page-9-7) as given by:

$$\mathcal{L}_{\text{BCE}}(\phi) = -\mathbb{E}_{(x,y,r) \sim \mathcal{D}}[r \log R_\phi(x,y) - (1-r) \log R_\phi(x,y)] \quad (2)$$

Bradley Terry Model (BT): Since the goal of the verifier is to relatively rank code solutions rather than predict absolute reward, we create a preference dataset and then train with a Bradley Terry loss [\(Ouyang et al.,](#page-10-4) [2022\)](#page-10-4). For every prompt x, we create pairs of correct y <sup>+</sup> (where r = 1) and incorrect y <sup>−</sup> (where r = 0) code and define the following loss:

$$\mathcal{L}_{BT}(\phi) = -\mathbb{E}_{(x,y^+,y^-) \sim \mathcal{D}}[\log \sigma(R_\phi(x, y^+) - R_\phi(x, y^-))]. \quad (3)$$

where σ(.) is the sigmoid function. We hypothesize that BT is strictly easier to optimize as the verifier has to only focus on relative performance. This is also consistent with observations made for training process reward models, where the advantage function is easier to optimize than the absolute Q function [\(Setlur et al.,](#page-10-5) [2024\)](#page-10-5).

### 3.3. Training Generator

µCODE comprises a generator π<sup>θ</sup> trained to produce code solutions conditioned on the initial problem and execution observations from previous turns. Given a dataset D, µCODE iteratively trains the generator to find the optimal code solution labeled using the local expert over the learned verifier. For this step, µCODE extracts all code solutions from D for every problem x. An expert is then created by picking the best solution, y ⋆ , which achieves the highest score using the learned verifier Rϕ(x, y) when combined with the output of oracle verifier R(x, y) and is given by

$$y^* = \pi_*(x) = \arg \max_{y \in \mathcal{D}(x)} \beta_{\mathcal{O}} R(x, y) + \beta_{\mathcal{L}} R_{\phi}(x, y), \quad (4)$$

where β<sup>O</sup> = 1.0 and β<sup>L</sup> = 0.1 denote the weights for oracle and learned rewards. Note that for creating the training dataset, we also use the ground labels from oracle verifier as they are available to the agent. The combination of both verifiers yielded better performance in our experiments. Using this expert dataset, we relabel D with the optimal solutions:

$$\mathcal{D}_\star = \{(x, s_t, y^\star) \mid (x, s_t) \sim \mathcal{D}\}, \quad (5)$$

where D<sup>⋆</sup> represents the expert dataset. The generator π<sup>θ</sup> is then trained via fine-tuning (FT) on this expert dataset D⋆.

Algorithm 2 µCODE: Inference loop

input Generator πθ, learned verifier Rϕ, turn limit T, number of rollouts N, public tests, and private tests 1: Set s<sup>1</sup> = {x}, t = 1 2: while true do 3: Generate N rollouts {y n t } N <sup>n</sup>=1 ∼ πθ(.|st) 4: Choose best solution y ∗ <sup>t</sup> = arg max<sup>n</sup> Rϕ(x, y<sup>n</sup> t ) 5: Execute y ∗ t to get execution feedback o<sup>t</sup> 6: if y ∗ <sup>t</sup> passes public tests or t = T then 7: break; 8: end if 9: Update state st+1 = {st, y<sup>∗</sup> t , ot} and increment t 10: end while output Return y ∗ to execute on public and private tests

#### 3.4. Inference: Multi-turn Best-of-N

At inference time, the goal is to generate a code solution with a fixed inference budget – denoting the number of times generators can provide one complete solution. In this work, we propose to leverage the learned verifier to improve search and code generations over successive turns with *multi-turn Best-of-N* (BoN). To achieve this, µCODE uses a natural extension of BoN to the multi-turn setting. At each turn, the generator produces N one-step rollouts {y n t } N <sup>n</sup>=1 ∼ πθ(.|st) and the learned verifier picks the most promising code solution among these candidates using

$$y_t^* = \arg \max_n R_\phi(x, y_t^n). \quad (6)$$

The selected code y ∗ t is executed in the environment over public tests to obtain the execution feedback ot. This solution and the feedback is provided as context to the generator at the next turn to repeat this procedure. The search ends once y ∗ <sup>t</sup> passes all public tests or when the turn limit is reached. Consequently, even if Rϕ(·) grants a high score to a code solution, inference continues until the solution has successfully cleared all public tests, thus mitigating potential errors by Rϕ(·). The final response y ∗ t is then passed through the oracle verifier to check its correctness. Algorithm [2](#page-3-2) describes our propose procedure of multi-turn BoN search. We found it beneficial to use the reward model trained with samples of the latest generator π<sup>θ</sup> (see Table [1\)](#page-4-0).

### 3.5. Analysis

µCODE effectively treats multi-turn code generation as an interactive imitation learning problem by collecting rollouts from a learned policy and re-labeling them with an expert. It circumvents the exploration burden of generic reinforcement learning which has exponentially higher sample complexity [\(Sun et al.,](#page-10-6) [2017\)](#page-10-6). We briefly analyze why this problem is amenable to imitation learning and prove performance bounds for µCODE.

Definition 3.1 (One-Step Recoverable MDP). A MDP M = (S, A, P, R, γ) with horizon T is *one-step recoverable* if the advantage function of the optimal policy π ∗ , defined as A<sup>∗</sup> (s, a) = Q<sup>∗</sup> (s, a)−V ∗ (s), is uniformly bounded for all (s, a), i.e. −1 ≤ A<sup>∗</sup> (s, a) ≤ 0.

Code generation is one-step recoverable MDP. Multiturn code generation satisfies one-step recoverability because the optimal policy π ∗ (yt|st) depends only on the problem prompt x and not the interaction history s<sup>t</sup> = (x, y1, o1, . . . , yt−1, ot−1). Since the correctness of a code snippet y<sup>t</sup> is fully determined by x, the optimal Q-function satisfies Q<sup>∗</sup> (st, yt) = R(x, yt), where R(x, yt) ∈ {0, 1}. The optimal value function is V ∗ (st) = max<sup>y</sup><sup>t</sup> R(x, yt), so the advantage function simplifies to A<sup>∗</sup> (st, yt) = R(x, yt) − max<sup>y</sup> ′ <sup>t</sup> <sup>R</sup>(x, y′ t ) ≤ 0.

Code generation enables efficient imitation learning. There are two challenges to applying interactive imitation learning [\(Ross et al.,](#page-10-7) [2011;](#page-10-7) [Ross & Bagnell,](#page-10-8) [2014\)](#page-10-8) – (1) Existence of expert policies or value functions, and (2) Recoverability of expert from arbitrary states. First, for code generation, the expert is simply the one-step reward maximizer arg max<sup>y</sup> R(x, y). We can efficiently estimate Rϕ(x, y) to compute the expert, without needing to compute value function backups. Second, even if the learner fails to imitate the expert at any given state, the expert can perfectly recover from the next state. This results in the best possible performance bounds for imitation learning, which we formalize below.

Theorem 3.2 (Performance bound for µCODE). *For a onestep recoverable MDP* M *with horizon* T*, running* N *iterations of* µCODE *yields at least one policy* π *such that*

$$J(\pi^*) - J(\pi) \leq O(T(\epsilon + \gamma(N))). \quad (7)$$

*where* π ∗ *is the expert policy,* ϵ *is the realizability error, and* γ(N) *is the average regret.*

Proof is in Appendix [A.1.](#page-12-0) The bound O(ϵT) is much better than the worst-case scenario of O(ϵT<sup>2</sup> ) for unrecoverable MDPs [\(Swamy et al.,](#page-10-9) [2021\)](#page-10-9). Thus, µCODE exploits the structure of multi-turn code generation to enable imitation learning, bypassing the need for hierarchical credit assignment. More generally, this analysis suggests that for any task where the optimal action is history-independent and recoverable in one step, reinforcement learning can be reduced to efficient imitation learning without loss of performance.

# 4. Experiments

Through our experiments, we aim to analyze (1) How does µCODE compare to leading state-of-the-art methods? (2) Does a learned verifier facilitate training a better generator? (3) Can the use of a learned verifier improve multi-turn BoN

search at inference time? (4) Does the test-time search show scaling law trends? (5) Which loss function works better for learning a verifier for µCODE?

#### 4.1. Setup

Models. The generator model in µCODE is initialized with Llama-3.2-1B-Instruct or Llama-3.1-8B-Instruct [\(Dubey](#page-9-8) [et al.,](#page-9-8) [2024\)](#page-9-8). The learned verifiers are initialized with the same models as generators and have a randomly initialized linear layer to predict a scalar score [\(Stiennon et al.,](#page-10-10) [2020\)](#page-10-10).

Datasets. We conduct experiments on MBPP [\(Austin](#page-9-4) [et al.,](#page-9-4) [2021\)](#page-9-4) and HumanEval [\(Chen et al.,](#page-9-5) [2021\)](#page-9-5) where the agent needs to generate code solutions in Python given natural language descriptions. We train the methods on the MBPP training set which comprises 374 problems and evaluate on the MBPP test set and HumanEval (HE) dataset which have 500 and 164 problems. We also compare methods on the DeepMind CodeContests dataset (CC, [Li et al.](#page-9-6) [\(2022a\)](#page-9-6)) where we train on 1000 problems sampled from the training set and evaluate on the 165 problems in the test set. We further describe the prompts and the split of public and private tests in Appendix [C.1](#page-14-0) and [C.2.](#page-16-0) For training, we trained RFT and µCODE for 2 iterations in MBPP and HumanEval datasets and for 1 iteration on CodeContests.

Baselines. We compare µCODE with single and multiturn baselines. For single and multi-turn settings, we report metrics with *Llama-3.2-1B Instruct* and *Llama-3.1- 8B-Instruct* as base models. We also compare with rejection finetuning (RFT) where we collect multiple rollouts and filter trajectories with a correct solution for finetuning (FT). For multi-turn RFT, given a positive rollout {x, y1, o1, ..., y<sup>T</sup> , o<sup>T</sup> }, the model is finetuned over all the sub-trajectories {s<sup>i</sup> , yi+1} T −1 <sup>i</sup>=0 . For the multi-turn BoN search at inference time, we used the verifier learned from the last iteration for µCODE and trained a verifier with generated rollouts for the base and RFT models. Lastly, for multi-turn BoN search at inference time, we pick the best code solution with a hybrid approach where a solution passing public tests is preferred followed by ranking solutions with a learned verifier.

Metrics. We evaluate the methods by comparing the *BoN* accuracy. The generator is allowed upto T = 3 turns and the final solution is used for evaluation over private tests. The *BoN@1* evaluates the agent at producing the solution in the first attempt and is obtained via greedy decoding with a temperature of 0. The *BoN* accuracy measures the ability of verifiers to leverage test-time compute by generating N candidate solutions in parallel at each turn. At each turn, the verifier ranks N = 5 solutions (unless stated otherwise) provided by the generator. For the BoN performance, we

| Method   | N | MBPP | Llama-3.2-1B HE Single-Turn | MBPP | Llama-3.1-8B HE | CC   |
|----------|---|------|-----------------------------|------|-----------------|------|
| Instruct | 1 | 35.1 | 25.6                        | 52.1 | 59.8            | 3.6  |
| RFT      | 1 | 35.7 | 34.1                        | 53.7 | 54.9            | –    |
| Instruct | 1 | 35.1 | 31.1                        | 60.3 | 59.7            | 4.8  |
| +BoN     | 5 | 47.3 | 35.7                        | 69.7 | 62.9            | 13.8 |
| RFT      | 1 | 31.1 | 31.7                        | 58.9 | 61.2            | 7.2  |
| +BoN     | 5 | 46.7 | 34.1                        | 68.4 | 62.8            | 14.9 |
| µ C ODE  | 1 | 37.9 | 35.4                        | 62.1 | 60.9            | 7.9  |
| +BoN     | 5 | 51.1 | 41.5                        | 70.6 | 63.8            | 16.3 |

Table 1. Comparison of our method µCODE with baselines across MBPP, HumanEval, and CodeContests datasets. N = 1 denotes generating solutions with 0 temperature. The Best-of-N (BoN) accuracy is computed with N = 5 candidate solutions at each where the public tests and learned verifier are used for selection. We observe that µCODE outperforms competing methods based on Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct models. The best performance for each dataset and model-sized is highlighted in bold and similar performances (within 1%) are underlined.

sample with a temperature of 0.7.

### 4.2. Results

In Table [1,](#page-4-0) we compare our proposed algorithm µCODE with the baselines. We first evaluate the generators using code generated via greedy sampling for each problem (*BoN* with N = 1). Our approach µCODE outperforms RFT across both benchmarks with 1B-sized model demonstrating the efficacy of using one-step recoverability and learned verifiers. To highlight, our method µCODE with the 1B model outperforms baselines by 2.2% and 1.3% on MBPP and HumanEval datasets. Interestingly, the performance of RFT drops when compared to the base Instruct model for the multi-turn setting which can be attributed to the fact that finetuning dataset consists of sub-trajectories with incorrect code solution at non-terminal steps. For the 8B-sized variant, we observe similar trends where we see that all algorithms benefit from the multi-turn BoN search at inference time. Additionally, µCODE performs better than both single and multi turn baselines across benchmarks.

With more inference budget and multi-turn BoN search at test-time, where the learned verifier and outcomes of public tests are used to select the best solution at each turn, we observe that every method performs better with the test-time search procedure by upto 13%. For the 1B-sized models, our method µCODE significantly outperforms baselines by 4.4% on MBPP and by 5.8% on Humaneval datasets. For the 8B-sized variant, µCODE performs better than the

baselines across datasets. On the more challenging task of CodeContests, all methods benefit from the BoN search and observe performance gains of upto 2×. On this benchmark, µCODE outperforms the RFT and base model baselines by 1.4%. We further investigate these trends with a Qwen model in Appendix [B.1](#page-13-0) and additional unit tests on MBPP and HumanEval benchmarks in Appendix [B.2.](#page-13-1)

#### 4.3. Analysis

To delve deeper into the improvements, we conduct a component-wise ablation study where we 1) check the effect of one-step recoverability and the learned verifier for creating the local expert [\(4.3.1\)](#page-5-0), 2) compare different verifiers for multi-turn BoN search at test-time [\(4.3.2\)](#page-5-1), 3) test the efficacy of agents at utilizing execution feedback [\(4.3.3\)](#page-6-0), 4) assess scaling behaviors at inference time with number of candidate generations (N) at each turn [\(4.3.4\)](#page-6-1), and 5) study different loss functions to train the verifiers [\(4.3.5\)](#page-7-0).

4.3.1. WHAT MAKES A GOOD GENERATOR IN µCODE?

| Method     | One-step Verifier | MBPP | HE   |
|------------|-------------------|------|------|
| RFT        | ✗ Oracle          | 46.7 | 36.5 |
| RFT LV     | ✗ Learned         | 49.0 | 38.9 |
| µ C ODE OV | ✓ Oracle          | 48.2 | 38.0 |
| µ C ODE LV | ✓ Learned         | 48.5 | 39.1 |
| µ C ODE    | ✓ Both            | 51.1 | 41.5 |

Table 2. Comparison of using learned verifier (LV) and relabeling with one-step recoverability (One-Step) with the 1B-sized model. We observe that FT with the learned verifier RFTLV performs better than FT with the oracle verifier scores RFT. Moreover, µCODE performs best when from both verifiers are used for relabeling.

Firstly, we compare different verifiers for training to demonstrate the benefits of using dense rewards obtained from learned verifiers. The RFT baseline uses the oracle verifier to filter trajectories and does not relabel subsequences for FT. In this experiment, we compare RFT to RFTLV where the learned verifier selects the top K rollouts for each prompt ranked via the learned verifier for FT. We use K = 3 in our experiments. We observe in Table [2](#page-5-2) that finetuning with the learned verifier outperforms the RFT baseline. In contrast to RFT, which lacks training data for prompts with no correct solutions, RFTLV effectively trains the generator to ascend the dense rewards obtained from the learned verifier.

Secondly, we present the advantages of relabeling subtrajectories with our insight of one-step recoverability. We extend the baselines with our proposed relabeling strategy described in Sec. [3.3.](#page-2-1) We call these methods µCODEOV (β<sup>O</sup> = 1, β<sup>L</sup> = 0) and CODELV (β<sup>O</sup> = 0, β<sup>L</sup> = 1) depending on the verifier used for relabeling. Table [2](#page-5-2) shows that µCODEOV outperforms the baseline RFT presenting the

| Approach Base | MBPP | Llama-3.2-1B HE | MBPP | Llama-3.1-8B HE |
|---------------|------|-----------------|------|-----------------|
| Random        | 31.7 | 24.6            | 58.0 | 57.9            |
| LV            | 30.3 | 29.4            | 62.5 | 61.0            |
| PT            | 46.4 | 33.4            | 68.4 | 60.7            |
| PT+LV RFT     | 47.3 | 35.7            | 69.7 | 62.9            |
| Random        | 31.1 | 27.4            | 58.9 | 57.7            |
| LV            | 33.2 | 29.6            | 62.2 | 61.3            |
| PT            | 46.8 | 36.8            | 67.4 | 61.4            |
| PT+LV         | 46.7 | 36.5            | 68.4 | 62.8            |
| µ C ODE       |      |                 |      |                 |
| Random        | 37.5 | 31.5            | 61.5 | 58.4            |
| LV            | 43.3 | 36.5            | 64.8 | 60.6            |
| PT            | 49.4 | 39.7            | 69.4 | 61.4            |
| PT+LV         | 51.1 | 41.5            | 70.6 | 63.8            |

Table 3. Comparing BoN with different ways of picking solutions at each turn for multi-turn BoN search using the 1B sized model. The hierarchical approach of using public test and learned verifier (PT+LV) outperforms picking solutions with only using either public tests (PT) or the learned verifier (LV). The best performance for each dataset and model-size is highlighted in bold and similar performances (within 1%) are underlined.

benefits of relabeling. The performance is similar for the setting where the learned verifier is used. To leverage the best of both worlds, µCODE uses a linear combination of scores from learned and oracle verifier for relabeling which outperforms each variant by around 2%.

### 4.3.2. DOES LEARNED VERIFIER AID BON SEARCH?

We study the effect of different verifiers for ranking the candidate solutions to pick the best solution for multi-turn BoN search at inference time. We test with *Random* strategy where the policy randomly picks from the N solutions. We compare to using the outcomes on public tests (PT) that picks any solution that passes the public test. Note that this involves evaluating all generated solutions at every turn with all the given public tests. We also compare to selecting a solution based on scores obtained via the learned verifier only (LV). This is crucial as in certain applications such privileged information like public tests are not available and the agents can benefit from learned verifiers to improve search during inference. Lastly, we compare with the combination of public tests and using the dense scores obtained from the learned verifier to break ties at each turn (PT+LV).

In Table [3,](#page-5-3) we compare the baselines and µCODE with different verifiers at inference-time. We observe that LV outperforms *Random* strategy which shows that a learned verifier

![](_page_6_Figure_1.jpeg)

Figure 2. Compares the BoN performance at each turn (with 6 turns) on HumanEval, MBPP datasets. We also present results on a partially observable version of MBPP where we remove the public tests from the prompt to test the efficacy of methods at incorporating execution feedback– MBPP (POMDP). We observe that all methods improve performance with more turns on MBPP and HumanEval datasets. However, on MBPP (POMDP) the performance drops at first turn and µCODE closes the gap with performance on MBPP compared to the baselines demonstrating its ability to incorporate execution feedback to improve code solutions at each turn.

indeed selects better solutions amongst the candidates. Furthermore, using the outcome of public tests (PT) performed better than using learned verifiers (LV) and performs similarly on the HumanEval datset for 8B-sized models. We believe that this gap can be further reduced by learning more powerful verifiers with larger datasets. Interestingly, the hierarchical approach (PT+LV) that uses the learned verifier to break ties on the outcomes of public tests performs best across methods and datasets. We hypothesize that using learned verifiers is beneficial in two scenarios. Firstly, if multiple solutions pass the public tests, then the learned verifier can filter out incorrect solutions which may not pass private tests. Secondly, if all candidate solutions are incorrect, then the learned verifier chooses the most promising solution at each turn. This is crucial as picking a better solution with the learned verifier can lead to more relevant feedback for recovering the true solution.

#### 4.3.3. CAN µCODE UTILIZE EXECUTION FEEDBACK?

We evaluate the ability of the trained generator to utilize execution feedback and improve the code response across turns. We report the BoN accuracy till a turn t, which denotes the accuracy of obtaining a correct solution within t turns. Note that the agents were trained with rollouts of upto 3 turns and we report results with 6 turns for this experiment. In Fig. [2](#page-6-2) (left and middle), we present the results with 1Bsized models where we observe that BoN accuracy improves with successive turns across the benchmarks.

To further understand if µCODE learns to utilize execution feedback, we curated another test dataset from MBPP where we removed the sample unit tests provided in the prompt, and call it MBPP (POMDP). Removing public unit test information from the prompt makes the information from execution feedback essential and is similar to a partially observable MDP setting. In Fig. [2](#page-6-2) (right), we compare the

BoN accuracy over this evaluation dataset for µCODE and the baselines. We observe that the performance at first turn drops for all methods by upto 13%. With the execution feedback at each turn, µCODE improves the code solutions leading to gains of 15.9% from turn 1 to turn 6 and matches its performance on the MBPP dataset. In contrast, the base model and RFT were unable to close this gap in this partially observable setting and performed worse by around 6% that the MBPP dataset. This demonstrates the ability of µCODE to recover better code solutions at each turn by utilizing the execution feedback at each turn, a trend not observed for the Instruct model and the RFT baselines.

## 4.3.4. DOES µCODE SCALE WITH INFERENCE BUDGET?

![](_page_6_Figure_6.jpeg)

Figure 3. Test-time scaling with different values of candidate solutions N at each turn. The candidate solutions are obtained from the 1B-sized generator of µCODE. We observe that the BoN performance improves with larger values of N on both datasets.

In the multi-turn setting, the number of candidate solutions can rise exponentially with the number of turns. To avoid this, µCODE uses the learned verifier during inference to select the most promising candidate among N candidates at each turn, leading to a linearly increasing number of calls to the generator. In this experiment, we study the inferencetime scaling behaviors of µCODE where we scale the number of candidate generations N at each turn. Figure [3](#page-6-3) plots the BoN with different values of N (1 ≤ N ≤ 11). With more inference time budget, we observe that the performance improves with larger number of candidates at each turn on both datasets. The BoN accuracy plateaus with N ≥ 5 for HumanEval dataset where for MBPP dataset we still observe some performance gains with larger N.

#### 4.3.5. LOSS FUNCTION FOR VERIFIER

As described in [3.2,](#page-2-0) we compare against different loss functions for training the verifier. For this experiment, we first generate multiple single step rollouts and label them via oracle verifier. Given oracle labels, we train verifiers with two loss functions – BCE and BT. During inference, the learned verifier picks the best ranked solution among the N solutions provided by the generator. Similar to [\(Cobbe et al.,](#page-9-7) [2021\)](#page-9-7), we report the BoN plot with different values of N obtained by first sampling N candidate solutions, choosing the top-ranked solution using the learned verifier, and then evaluating the solution against public and private tests. We calculate this metric over multiple samples for each value of N. In Figure [4,](#page-7-1) we observe that the verifier trained with BT loss consistently outperforms the verifier trained on BCE loss on both MBPP and HumanEval.

Figure 4. Comparison between BCE and BT loss function for training the verifier. We train the verifiers on samples generated by the base model (Llama-3.2-1B-Instruct). The learned verifier then ranks the candidate solutions from base model and the BoN performance of selected solution is reported. The verifier trained with BT loss performs better increasing value of N.

### 4.3.6. QUALITATIVE RESULT

Figure [5](#page-8-1) presents a qualitative example of multi-turn Bestof-N search with µCODE. Through this example, we demonstrate the advantages of dense scores from the learned verifier at facilitating efficient search across turns. We generate N = 5 code solutions at each turn and show the top 3 ranked solutions using the dense scores. At the first turn, we observe that the last solution y 3 1 is less accurate than the other 2 solutions y 1 1 and y 2 1 . The top ranked solution is used to collect the environment feedback, upon which the generator comes up with N new candidate solutions. Upon

the top 3 solutions, the last two snippets are similar to the candidates from the previous turn. However, the top ranked solution is a novel solution and is more accurate as the generated code learns to extract a single digit and multiply it. With the execution feedback, µCODE generates 2 correct responses– y 1 3 and y 2 3 and learned verifier chooses one of them compared to the incorrect response y 3 3 .

# 5. Related Work

Prompting To Solve Multi Step Tasks A common framework for tackling multi-step tasks with LLMs is promptingbased agentic systems. Self-Debugging [\(Chen et al.,](#page-9-9) [2023b\)](#page-9-9) asks the LLM to iteratively improve code by providing execution feedback while CodeT [\(Chen et al.,](#page-9-10) [2022\)](#page-9-10) asks the LLM to generate test cases. AlphaCodium [\(Ridnik et al.,](#page-10-11) [2024\)](#page-10-11) first reflects on input instructions, generates and filters from multiple code generations, and finally iterates on public and self-generated test cases. MapCoder [\(Islam et al.,](#page-9-11) [2024\)](#page-9-11) incorporates four agents to generate example problems, plans and code, and then perform debugging. However, prompting-based agents yield limited improvements.

Training LLMs for Multi Step Tasks Some work has explored explicitly training critics or reward models for multi-step reasoning tasks. In the coding domain, CodeRL [\(Le et al.,](#page-9-12) [2022\)](#page-9-12) trains a token-level critic to aid in code generation and to perform inference-time search. CodeRL's mechanics are similar to our method, but their generator is not trained for multi-step: CodeRL trains a "code repairer" which conditions on one erroneous code completion while our generator incorporates multiple. ARCHER [\(Zhou et al.,](#page-11-2) [2024\)](#page-11-2), which frames multi-step tasks via a two-level hierarchical MDP, where the higher level MDP considers completions as actions and the lower level MDP considers tokens as actions. Another line of work utilizes Monte Carlo Tree Search (MCTS) methods for training: rStar-Math [\(Guan](#page-9-13) [et al.,](#page-9-13) [2025\)](#page-9-13) trains a policy preference model to boost small LMs' math abilities to match or exceed large reasoningbased LMs and ReST-MCTS [\(Zhang et al.,](#page-11-4) [2024\)](#page-11-4) trains a process reward model (PRM) similarly to Math-Shepherd [\(Wang et al.,](#page-10-12) [2024a\)](#page-10-12). Although µCODE's BoN search resembles a tree search, our key insight that multi-step code generation resembles a one-step recoverable MDP allows us to collect training trajectories much more efficiently. Finally, some work has explored using verifiers only during inference time. In "Let's Verify Step by Step" [\(Lightman](#page-10-13) [et al.,](#page-10-13) [2023\)](#page-10-13), the authors demonstrate that PRMs trained on erroneous math solutions annotated by humans outperform outcome reward models for filtering multiple inference time generations. Meanwhile, AlphaCode [\(Li et al.,](#page-10-14) [2022b\)](#page-10-14) trains a test generator to evaluate multiple code solutions.

![](_page_7_Figure_8.jpeg)

Other works omit learning a critic or reward model alto-

![](_page_8_Diagram_1.jpeg)

Figure 5. A qualitative example of multi-turn BoN search using dense rewards obtained via the learned verifier in µCODE. Here, we show the top 3 ranked solutions at each turn t where Rϕ(x, y<sup>i</sup> <sup>t</sup>) ≥ Rϕ(x, y t ) for i < j. We observe that the learned verifier selects the better solution (in orange) at each turn. The selected solution is passed to public tests to retrieve execution feedback for the generator to improve the next code solution. The selected solution at each turn is better than the last (less errors highlighted in yellow), with the final solution passing all tests. Note that there are 2 correct solutions at the final turn.

gether. In the coding domain, RLEF [\(Gehring et al.,](#page-9-14) [2024b\)](#page-9-14) derives rewards only on the executor's result on test cases and syntax checkers, and PPOCoder [\(Shojaee et al.,](#page-10-15) [2023\)](#page-10-15) additionally considers semantic and syntactic alignment, generated via data flow graphs and abstract syntax trees respectively, with a reference solution. The "oracle" rewards in these methods may not be informative for training, and in the case of PPOCoder, require complex constructs. We empirically show that having a reward model is beneficial by comparing µCODE against the RFT baseline. Meanwhile, SCoRe [\(Kumar et al.,](#page-9-15) [2024a\)](#page-9-15) splits training into a "generator" and "correction" phase, thus restricting the total number of turns to 2. RISE [\(Qu et al.,](#page-10-16) [2024\)](#page-10-16) generates recovery steps via a more powerful LLM or by selecting a sampled completion via the oracle rewards. Both methods are less efficient than µCODE, which doesn't require generating corrections beyond generating training trajectories. Finally, FireAct [\(Chen et al.,](#page-9-16) [2023a\)](#page-9-16) and LEAP [\(Choudhury & Sodhi,](#page-9-17) [2024\)](#page-9-17) FT ReAct style agents while RL4VLM [\(Zhai et al.,](#page-11-5) [2024\)](#page-11-5) and GLAM [\(Carta et al.,](#page-9-18) [2024\)](#page-9-18) studies training LLMs with interactive environment feedback.

# 6. Conclusion

We present µCODE, a simple and scalable method for multiturn code generation through single-step rewards. µCODE models code generation as a one-step recoverable MDP and learns to iteratively improve code with a learned verifier to guide the search. Experimental results demonstrate that µCODE outperforms methods using oracle verifiers by a large margin. We acknowledge the following limitations of this paper. Due to a limited budget, we were only able to train models with up to eight-billion parameters. It is possible that the conclusions made in this paper do not

generalize to models of larger scales. Additionally, we train models on MBPP, whose training set has only 374 examples. However, we hypothesize that more training examples will lead to better performance. Finally, our datasets are only in Python, and our findings might not generalize to other programming languages.

# Impact Statement

The proposed method for training code agents has the potential to streamline software development processes by automating routine coding tasks, thereby reducing human labor and accelerating production timelines. However, these advances will also introduce bugs, which can propagate at scale if no proper quality control is in place.

# Acknowledgements

AJ is supported by Calcul Quebec, Canada Ex- ´ cellence Research Chairs (CERC), and Fonds de Recherche du Quebec (FRQ) scholarship (DOI assigned: ´ https://doi.org/10.69777/350253) program. The authors are also grateful to Mila (mila.quebec) IDT and Digital Research Alliance of Canada for computing resources. AMR is supported in part by NSF CAREER #2037519 and NSF #2242302. SC is supported in part by Google Faculty Research Award, OpenAI SuperAlignment Grant, ONR Young Investigator Award, NSF RI #2312956, and NSF FRR#2327973.

# References


[1] Anthony, T., Tian, Z., and Barber, D. Thinking fast and slow with deep learning and tree search, 2017. URL

[2] <https://arxiv.org/abs/1705.08439>. Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al. Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*, 2021. Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., Re, C., and Mirhoseini, A. Large language monkeys: ´ Scaling inference compute with repeated sampling. *arXiv preprint arXiv:2407.21787*, 2024. Carta, T., Romac, C., Wolf, T., Lamprier, S., Sigaud, O., and Oudeyer, P.-Y. Grounding large language models in interactive environments with online reinforcement learning, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2302.02662) [2302.02662](https://arxiv.org/abs/2302.02662). Chen, B., Zhang, F., Nguyen, A., Zan, D., Lin, Z., Lou, J.-G., and Chen, W. Codet: Code generation with generated tests, 2022. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2207.10397) [2207.10397](https://arxiv.org/abs/2207.10397). Chen, B., Shu, C., Shareghi, E., Collier, N., Narasimhan, K., and Yao, S. Fireact: Toward language agent finetuning, 2023a. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2310.05915) [2310.05915](https://arxiv.org/abs/2310.05915). Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto,
  - H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba,

[3] W. Evaluating large language models trained on code, 2021. Chen, X., Lin, M., Scharli, N., and Zhou, D. Teaching large ¨ language models to self-debug, 2023b. URL [https:](https://arxiv.org/abs/2304.05128) [//arxiv.org/abs/2304.05128](https://arxiv.org/abs/2304.05128). Chen, X., Lin, M., Scharli, N., and Zhou, D. Teaching ¨ large language models to self-debug. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=KuPixIqPiq) [id=KuPixIqPiq](https://openreview.net/forum?id=KuPixIqPiq). Choudhury, S. and Sodhi, P. Better than your teacher: Llm agents that learn from privileged ai feedback, 2024. URL <https://arxiv.org/abs/2410.05434>. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021. Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024. Gehring, J., Zheng, K., Copet, J., Mella, V., Cohen, T., and Synnaeve, G. Rlef: Grounding code llms in execution feedback with reinforcement learning. *arXiv preprint arXiv:2410.02089*, 2024a. Gehring, J., Zheng, K., Copet, J., Mella, V., Cohen, T., and Synnaeve, G. Rlef: Grounding code llms in execution feedback with reinforcement learning, 2024b. URL <https://arxiv.org/abs/2410.02089>. Guan, X., Zhang, L. L., Liu, Y., Shang, N., Sun, Y., Zhu, Y., Yang, F., and Yang, M. rstar-math: Small llms can master math reasoning with self-evolved deep thinking, 2025. URL <https://arxiv.org/abs/2501.04519>. Islam, M. A., Ali, M. E., and Parvez, M. R. Mapcoder: Multi-agent code generation for competitive problem solving, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2405.11403) [2405.11403](https://arxiv.org/abs/2405.11403). Kakade, S. and Langford, J. Approximately optimal approximate reinforcement learning. In *Proceedings of the Nineteenth International Conference on Machine Learning*, pp. 267–274, 2002. Kumar, A., Zhuang, V., Agarwal, R., Su, Y., Co-Reyes,
  - J. D., Singh, A., Baumli, K., Iqbal, S., Bishop, C., Roelofs, R., Zhang, L. M., McKinney, K., Shrivastava, D., Paduraru, C., Tucker, G., Precup, D., Behbahani, F., and Faust, A. Training language models to self-correct via reinforcement learning, 2024a. URL <https://arxiv.org/abs/2409.12917>. Kumar, A., Zhuang, V., Agarwal, R., Su, Y., Co-Reyes, J. D., Singh, A., Baumli, K., Iqbal, S., Bishop, C., Roelofs, R., et al. Training language models to self-correct via reinforcement learning. *arXiv preprint arXiv:2409.12917*, 2024b. Le, H., Wang, Y., Gotmare, A. D., Savarese, S., and Hoi,
  - S. C. H. Coderl: Mastering code generation through pretrained models and deep reinforcement learning, 2022. URL <https://arxiv.org/abs/2207.01780>. Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Dal Lago, A., Hubert, T., Choy, P., de Masson d'Autume, C., Babuschkin, I., Chen, X., Huang, P.-S., Welbl, J., Gowal,

[4] S., Cherepanov, A., Molloy, J., Mankowitz, D., Sutherland Robson, E., Kohli, P., de Freitas, N., Kavukcuoglu, K., and Vinyals, O. Competition-level code generation with alphacode. *arXiv preprint arXiv:2203.07814*, 2022a. Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Dal Lago, A., Hubert, T., Choy, P., de Masson d'Autume, C., Babuschkin, I., Chen, X., Huang, P.-S., Welbl, J., Gowal, S., Cherepanov, A., Molloy, J., Mankowitz,

[5] D. J., Sutherland Robson, E., Kohli, P., de Freitas, N., Kavukcuoglu, K., and Vinyals, O. Competitionlevel code generation with alphacode. *Science*, 378 (6624):1092–1097, December 2022b. ISSN 1095-9203. doi: 10.1126/science.abq1158. URL [http://dx.doi.](http://dx.doi.org/10.1126/science.abq1158) [org/10.1126/science.abq1158](http://dx.doi.org/10.1126/science.abq1158). Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let's verify step by step, 2023. URL [https:](https://arxiv.org/abs/2305.20050) [//arxiv.org/abs/2305.20050](https://arxiv.org/abs/2305.20050). Muennighoff, N., Liu, Q., Zebaze, A., Zheng, Q., Hui, B., Zhuo, T. Y., Singh, S., Tang, X., von Werra, L., and Longpre, S. Octopack: Instruction tuning code large language models. *arXiv preprint arXiv:2308.07124*, 2023. Ni, A., Allamanis, M., Cohan, A., Deng, Y., Shi, K., Sutton, C., and Yin, P. NExt: Teaching large language models to reason about code execution. In *Forty-first International Conference on Machine Learning*, 2024. URL [https:](https://openreview.net/forum?id=B1W712hMBi) [//openreview.net/forum?id=B1W712hMBi](https://openreview.net/forum?id=B1W712hMBi). Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright,

[6] C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., and Lowe, R. Training language models to follow instructions with human feedback, 2022. URL [https:](https://arxiv.org/abs/2203.02155) [//arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155). Qu, Y., Zhang, T., Garg, N., and Kumar, A. Recursive introspection: Teaching language model agents how to selfimprove, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2407.18219) [2407.18219](https://arxiv.org/abs/2407.18219). Ridnik, T., Kredo, D., and Friedman, I. Code generation with alphacodium: From prompt engineering to flow engineering, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2401.08500) [2401.08500](https://arxiv.org/abs/2401.08500). Ross, S. and Bagnell, J. A. Reinforcement and imitation learning via interactive no-regret learning. *arXiv preprint arXiv:1406.5979*, 2014. Ross, S., Gordon, G., and Bagnell, D. A reduction of imitation learning and structured prediction to no-regret online learning. In *Proceedings of the fourteenth international conference on artificial intelligence and statistics*, pp. 627–635. JMLR Workshop and Conference Proceedings, 2011. Setlur, A., Nagpal, C., Fisch, A., Geng, X., Eisenstein, J., Agarwal, R., Agarwal, A., Berant, J., and Kumar, A. Rewarding progress: Scaling automated process verifiers for llm reasoning. *arXiv preprint arXiv:2410.08146*, 2024. Shojaee, P., Jain, A., Tipirneni, S., and Reddy, C. K. Execution-based code generation using deep reinforcement learning, 2023. URL [https://arxiv.org/](https://arxiv.org/abs/2301.13816) [abs/2301.13816](https://arxiv.org/abs/2301.13816). Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling llm testtime compute optimally can be more effective than scaling model parameters. *arXiv preprint arXiv:2408.03314*, 2024. Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 3008–3021. Curran Associates, Inc., 2020. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf) [cc/paper\\_files/paper/2020/file/](https://proceedings.neurips.cc/paper_files/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf) [1f89885d556929e98d3ef9b86448f951-Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf). [pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf). Sun, W., Venkatraman, A., Gordon, G. J., Boots, B., and Bagnell, J. A. Deeply aggrevated: Differentiable imitation learning for sequential prediction. In *International conference on machine learning*, pp. 3309–3318. PMLR, 2017. Swamy, G., Choudhury, S., Bagnell, J. A., and Wu, S. Of moments and matching: A game-theoretic framework for closing the imitation gap. In *International Conference on Machine Learning*, pp. 10022–10032. PMLR, 2021. Wang, P., Li, L., Shao, Z., Xu, R. X., Dai, D., Li, Y., Chen, D., Wu, Y., and Sui, Z. Math-shepherd: Verify and reinforce llms step-by-step without human annotations, 2024a. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2312.08935) [2312.08935](https://arxiv.org/abs/2312.08935). Wang, X., Chen, Y., Yuan, L., Zhang, Y., Li, Y., Peng, H., and Ji, H. Executable code actions elicit better LLM agents. In *Forty-first International Conference on Machine Learning*, 2024b. URL [https://openreview.](https://openreview.net/forum?id=jJ9BoXAfFa) [net/forum?id=jJ9BoXAfFa](https://openreview.net/forum?id=jJ9BoXAfFa). Welleck, S., Lu, X., West, P., Brahman, F., Shen, T., Khashabi, D., and Choi, Y. Generating sequences

[7] by learning to self-correct. In *The Eleventh International Conference on Learning Representations*, 2023. URL [https://openreview.net/forum?](https://openreview.net/forum?id=hH36JeQZDaO) [id=hH36JeQZDaO](https://openreview.net/forum?id=hH36JeQZDaO). Wu, Y., Sun, Z., Li, S., Welleck, S., and Yang, Y. Inference scaling laws: An empirical analysis of computeoptimal inference for problem-solving with language models. *arXiv preprint arXiv:2408.00724*, 2024. Zelikman, E., Wu, Y., Mu, J., and Goodman, N. Star: Bootstrapping reasoning with reasoning. *Advances in Neural Information Processing Systems*, 35:15476–15488, 2022. Zhai, Y., Bai, H., Lin, Z., Pan, J., Tong, S., Zhou, Y., Suhr, A., Xie, S., LeCun, Y., Ma, Y., and Levine, S. Finetuning large vision-language models as decision-making agents via reinforcement learning, 2024. URL [https:](https://arxiv.org/abs/2405.10292) [//arxiv.org/abs/2405.10292](https://arxiv.org/abs/2405.10292). Zhang, D., Zhoubian, S., Hu, Z., Yue, Y., Dong, Y., and Tang, J. Rest-mcts\*: Llm self-training via process reward guided tree search, 2024. URL [https://arxiv.](https://arxiv.org/abs/2406.03816) [org/abs/2406.03816](https://arxiv.org/abs/2406.03816). Zhao, W., Jiang, N., Lee, C., Chiu, J. T., Cardie, C., Galle,´ M., and Rush, A. M. Commit0: Library generation from scratch. *arXiv preprint arXiv:2412.01769*, 2024. Zheng, L., Yin, L., Xie, Z., Sun, C., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., Barrett, C., and Sheng, Y. Sglang: Efficient execution of structured language model programs, 2024. URL <https://arxiv.org/abs/2312.07104>. Zhou, Y., Zanette, A., Pan, J., Levine, S., and Kumar, A. Archer: Training language model agents via hierarchical multi-turn rl. *arXiv preprint arXiv:2402.19446*, 2024.
## A. Proofs

### A.1. Proof of Theorem 3.2

The proof relies on two important results.

The first is the Performance Difference Lemma (PDL) [\(Kakade & Langford,](#page-9-19) [2002\)](#page-9-19) which states that the performance difference between any two policies can be expressed as the sum of advantages.

$$J(\pi) - J(\pi') = \sum_{t=1}^T \mathbb{E}_{s_t \sim d_t^\pi} \left[ \sum_{a_t} A^{\pi'}(s_t, a_t) \pi(a_t | s_t) \right] \quad (8)$$

where s<sup>t</sup> ∼ d π t is the induced state distribution by π, and A<sup>π</sup> ′ (st, at) = Q<sup>π</sup> ′ (st, at) − V π ′ (st) is the advantage w.r.t. π ′ .

We apply the PDL between the expert π ∗ and the learner π

$$J(\pi^*) - J(\pi) = \sum_{t=1}^T \mathbb{E}_{s_t \sim d_t^\pi} \left[ \sum_{a_t} A^*(s_t, a_t) (\pi^*(a_t|s_t) - \pi(a_t|s_t)) \right] \quad (9)$$

where the result follows from P a<sup>t</sup> A<sup>⋆</sup> (st, at)π ⋆ (at|st) = 0

According to the one-step recoverable MDP definition, −1 ≤ A<sup>⋆</sup> (s, a) ≤ 0 for all (s, a). Hence we can bound the performance difference as

$$\begin{aligned} J(\pi^*) - J(\pi) &= \sum_{t=1}^T \mathbb{E}_{s_t \sim d_t^\pi} \left[ \sum_{a_t} A^*(s_t, a_t) (\pi^*(a|s_t) - \pi(a|s_t)) \right] \\ &\leq \|A^*(\cdot, \cdot)\|_\infty \sum_{t=1}^T \mathbb{E}_{s_t \sim d_t^\pi} \|\pi(\cdot|h_t) - \pi^*(\cdot|s_t)\|_1 \quad (\text{Holder's Inequality}) \\ &\leq \sum_{t=1}^T \mathbb{E}_{s_t \sim d_t^\pi} \|\pi(\cdot|s_t) - \pi^*(\cdot|s_t)\|_1 \quad (\text{One step recoverability}) \end{aligned}$$

The second result we use us from interactive imitation learning DAGGER [\(Ross et al.,](#page-10-7) [2011\)](#page-10-7) that reduces imitation learning to no-regret online learning. DAGGER shows that with π ⋆ as the expert teacher guarantees that after N iterations, it will find at least one policy

$$\mathbb{E}_{s \sim d^*} \|\pi(\cdot | s) - \pi^*(\cdot | s)\|_1 \leq \mathbb{E}_{s \sim d^*} \|\pi_{\text{class}}(\cdot | s) - \pi^*(\cdot | s)\|_1 + \gamma(N) \quad (10)$$

where γ(N) is the average regret, and d π is the time average distribution of states induced by policy π, πclass is the best policy in policy class.

Plugging this in we have

$$\begin{aligned} J(\pi^*) - J(\pi) &\leq \sum_{t=1}^T \mathbb{E}_{s_t \sim d_t^\pi} \|\pi(\cdot | s_t) - \pi^*(\cdot | s_t)\|_1 \\ &\leq \sum_{t=1}^T \mathbb{E}_{s_t \sim d_t^\pi} \|\pi_{\text{class}}(\cdot | s_t) - \pi^*(\cdot | s_t)\|_1 + \gamma(N) \quad \text{From (10)} \\ &\leq T(\epsilon + \gamma(N)) \end{aligned}$$

# B. Additional Results

### B.1. Qwen Model

| Method   | N | MBPP | HE   |
|----------|---|------|------|
| Instruct | 1 | 53.8 | 64.5 |
| +BoN     | 5 | 60.9 | 70.3 |
| RFT      | 1 | 57.0 | 66.6 |
| +BoN     | 5 | 58.0 | 71.3 |
| µ C ODE  | 1 | 59.0 | 70.5 |
| +BoN     | 5 | 63.1 | 74.0 |

Table 4. Comparison of our method µCODE with baselines across MBPP, HumanEval using the Qwen-2.5-1.5B-Instruct model.

# B.2. Additional private tests

| Method   | N | MBPP+ | HE+  |
|----------|---|-------|------|
| Instruct | 1 | 43.2  | 25.7 |
| +BoN     | 5 | 49.9  | 31.9 |
| RFT      | 1 | 44.3  | 26.7 |
| +BoN     | 5 | 50.0  | 34.3 |
| µ C ODE  | 1 | 48.7  | 32.4 |
| +BoN     | 5 | 55.1  | 40.0 |

Table 5. Comparison of our method µCODE with baselines across MBPP+, HumanEval+ using the Llama-3.2-1B-Instruct model. MBPP+ and HumanEval+ introduce 35x and 80x more tests than their original counterparts respectively. Note that MBPP+ has a higher score than MBPP because there are 30% fewer problems

# C. Hyperparameters

| Model           | Generator  | Verifier   |
|-----------------|------------|------------|
| Training Epochs | 2          | 2          |
| Learning Rate   | 5 × 10 − 7 | 1 × 10 − 6 |
| Batch Size      | 32         | 64         |
| Max seq length  | 4096       | 2048       |

Table 6. Hyperparameters for SFT and RM training.

## C.0.1. TRAINING PARAMETERS

Table [6](#page-13-2) contains hyperparameters for training the generator and reward model on both models (Llama-3.1-8B-Instruct and Llama-3.2-1B-Instruct) and datasets (MBPP and HumanEval). We perform 2 iterations of training with µCODE, starting from the base model each iteration. All training runs were on machines with either 4 RTX 6000 Ada Generation GPUs for 1B models with 48 GB of memory per GPU or 4 H100 GPUs for 8B models with 80 GB of memory per GPU.

## C.0.2. INFERENCE PARAMETERS

We use SGLang [\(Zheng et al.,](#page-11-6) [2024\)](#page-11-6) to serve our models for inference. Greedy experiments use temperature 0 with flags *–disable-radix-cache –max-running-request 1* to ensure deterministic results while BoN search experiments use a temperature of 0.7. All experiments are capped to 1000 tokens per completion per turn.

#### C.1. Prompts

## C.1.1. SINGLE STEP PROMPT

Immediately below is the prompt template to generate 1 code completion in a single-step method or to generate the 1st step in a multi-step method. Below the prompt templates are examples of the code prompt and public tests for HumanEval, MBPP, and CodeContests.

Single Step Prompt Write a Python function implementation for the following prompt: \{prompt\} Your code should satisfy these tests: \{test\} Please follow the following instructions: - Reason about the problem and any base cases before writing the code. - You must return the implementation code in the following format: '''python <CODE GOES HERE> ''' - You must only return a single code block since we only parse the first code block. - Do not include any tests in your code - we will run the suite and return any error feedback. - Include relevant import statements.

## HumanEval Prompt Example

from typing import List

def has\_close\_elements(numbers: List[float], threshold: float) -> bool:

""" Check if in given list of numbers, are any two numbers closer to each other

than

given threshold.

>>> has\_close\_elements([1.0, 2.0, 3.0], 0.5)

False

>>> has\_close\_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)

True """

## HumanEval Test Example

def check(has\_close\_elements): assert has\_close\_elements([1.0, 2.0, 3.0], 0.5) == False assert has\_close\_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True check(has\_close\_elements)

# MBPP Prompt Example

Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].

### MBPP Test Example

assert min\_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8 assert min\_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12 assert min\_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16

#### CodeContests Prompt Example

Provide a Python solution for the following competitive programming question: Mr. Chanek has an array a of n integers. The prettiness value of a is denoted as: \$\$\$\Sigma\_{i=1}ˆ{n} {\Sigma\_{j=1}ˆ{n} {\gcd(a\_i, a\_j) \cdot \gcd(i, j)}}\$\$\$ where \gcd(x, y) denotes the greatest common divisor (GCD) of integers x and y. In other words, the prettiness value of an array a is the total sum of \gcd(a\_i, a\_j ) \cdot \gcd(i, j) for all pairs (i, j). Help Mr. Chanek find the prettiness value of a, and output the result modulo 10ˆ9 + 7! Input The first line contains an integer n (2 \leq n \leq 10ˆ5). The second line contains n integers a\_1, a\_2, ..., a\_n (1 \leq a\_i \leq 10ˆ5). Output Output an integer denoting the prettiness value of a modulo 10ˆ9 + 7. Example Input 5 3 6 2 1 4 Output 77 Your code should be enclosed in triple backticks like so: '''python YOUR CODE HERE '''. Use the backticks for your code only.

#### CodeContests Test Example

# Input fed through stdin and output checked against stdout {'input': ['5\n54883 59286 71521 84428 60278\n', '2\n83160 83160\n'], 'output': ['1027150\n', '415800\n']}

# C.1.2. FEEDBACK PROMPT

Immediately below is the prompt template for how we provide feedback in multi-step methods. The feedback only consists of executor error traces, and we provide an example from HumanEval.

### Multi-Step Feedback Prompt

Feedback: \{feedback\}

## HumanEval Multi-Step Feedback Prompt

Traceback (most recent call last): File "test.py", line 18, in <module> assert has\_close\_elements([1.0, 2.0, 3.0], 0.5) == False ˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆˆ AssertionError

## C.2. Public Private Tests

We choose a public-private test split for HumanEval and MBPP to ensure that naively passing the public tests does not guarantee private test success. For HumanEval, we use a single test from the code prompt's docstring as the public test and the remaining tests along with the official test suite as private tests. For ease of parsing, we utilize a processed version of HumanEval, [HumanEvalPack](https://huggingface.co/datasets/bigcode/humanevalpack) [\(Muennighoff et al.,](#page-10-17) [2023\)](#page-10-17). For MBPP, we use a single test from the official test suite as the public test, and the remaining tests and any "challenge test list" tests as private tests.