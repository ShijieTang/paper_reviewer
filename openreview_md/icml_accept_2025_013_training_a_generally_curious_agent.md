# 

Fahim Tajwar * 1 **Yiding Jiang** * 1 Abitha Thankaraj 1 Sumaita Sadia Rahman 2 J Zico Kolter 1 **Jeff Schneider** 1 Russ Salakhutdinov 1

## Abstract

Efficient exploration is essential for intelligent systems interacting with their environment, but existing language models often fall short in scenarios that require strategic information gathering. In this paper, we present PAPRIKA, a fine-tuning approach that enables language models to develop general decision-making capabilities that are not confined to particular environments. By training on synthetic interaction data from different tasks that require diverse strategies, PAPRIKA teaches models to explore and adapt their behavior on a new task based on environment feedback incontext without more gradient updates. Experimental results show that models fine-tuned with PAPRIKA can effectively transfer their learned decision-making capabilities to entirely unseen tasks without additional training. Unlike traditional training, our approach's primary bottleneck lies in sampling useful interaction data instead of model updates. To improve sample efficiency, we propose a curriculum learning strategy that prioritizes sampling trajectories from tasks with high learning potential. These results suggest a promising path towards AI systems that can autonomously solve novel sequential decisionmaking problems that require interactions with the external world.

## 1. Introduction

Large language models (LLMs) are considered to be a promising foundation for autonomous agents, systems capable of achieving goals independently with minimal human supervision or intervention. A crucial requirement for such systems is the ability to interact effectively with external environments and gather the information necessary to achieve their objectives. This capability can be formalized as solving
*Equal contribution 1CMU 2North Carolina State University.

Correspondence to: Fahim Tajwar <ftajwar@cs.cmu.edu>.

sequential decision-making problems or performing reinforcement learning (RL) with language models as the agent. However, two challenges hinder the development of these interactive capabilities. First, most naturally occurring data lacks the structure and context needed to model interactions.

Second, directly deploying models into the real world to collect interaction data can produce critical errors, which is expensive and potentially risky. Given the impracticality of direct deployment in the wild, a natural alternative is to generate interaction data synthetically. Although generating synthetic data for every possible problem is infeasible, LLMs possess the capacity for incontext learning (ICL), which allows them to adapt to new tasks with minimal demonstrations (Brown et al., 2020). Instead of teaching the model to do all the interaction tasks that we care about, we should instead teach the model incontext reinforcement learning (Laskin et al., 2022) so that the model can solve new problems without being trained on them a priori. It shifts the focus from training the model on particular problems to training it on the general process of solving problems. This paradigm shares similarities with the supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF) stages of training a language model (vs pretraining) where only a relatively small number of examples is needed to produce a model that can generate responses to a wide range of queries that they are not trained on. Our approach is also closely related to the principles of meta reinforcement learning (Beck et al., 2023). In this work, we explore the feasibility of teaching LLMs to perform in-context RL that generalizes across different tasks, with the specific goal of training a curious agent with general information gathering capability. A popular notion of curiosity is *intrinsic motivation* which has been used to train agents with an exploration bonus not necessarily related to the success of any particular task (Schmidhuber, 1991; 2007). Our work differs from this notion of curiosity in that we do not leverage intrinsic motivation. Instead, we train our agents to explore and interact with an entirely unseen environment to gather information that is needed for completing the task at hand. PAPRIKA can be thought of as a form of *amortized exploration*, since our goal is to learn good exploration strategies from trajectories from many different environments to make exploration on a new 1

20 Questions Customer Service Wordle You are playing the game 20 questions.

[...]
I just moved to a new place and my wifi is not working.

Crane You are playing wordle.

[...]
Is the topic an animal?

Could you tell me your address?

42 Binary Boulevard, Circuit City, APT #314 What is the serial number of your router?

No.

Bears Construct preference pairs
&
perform policy optimization Is the topic a person?

... ... ...

Training Phase Test Phase
problem more efficient (see Appendix A for more details). We begin by designing a diverse suite of textual decisionmaking tasks that require active information gathering and decision-making based on interaction outcomes. Using a base model, we generate interaction trajectories and assign scores based on their success in achieving the tasks' objectives. We then apply a sequential variant of Direct Preference Optimization (Rafailov et al., 2024b, DPO) to increase the relative likelihood of successful trajectories. Unlike traditional training where computational costs are dominated by model updates, our approach's primary bottleneck lies in sampling useful interaction data. To improve sample efficiency, we propose a curriculum learning strategy that prioritizes sampling trajectories from tasks with high learning potential.

We refer to the overall framework as PAPRIKA1. Our results demonstrate that training on different subsets of these tasks improves the performance of the model on unseen tasks. More broadly, our result highlights the potential of using synthetic data to learn in-context RL which would equip LLMs with the capability to interact with the world and solve different decision-making problems without requiring task-specific fine-tuning.

## 2. Preliminary

Many decision making problems can be formalized as a partially observable Markov decision process (POMDP). We assume each task, τ , is a POMDP although we will not draw on the details of the POMDP formalism in this work.

1The name is inspired by the movie "Paprika" (2006), where a dream detective navigates vast and strange dream worlds to solve different mysteries.

As a concrete example, guessing the word "apple" would be a task in 20 questions. We will use *group* (or *task group*,
used interchangeably), G = {τ1, τ2*, . . . , τ*|G|}, to refer to a high-level grouping of different tasks (e.g., the game 20 questions would be a group). Tasks in a group should share similar strategies but it is not always true that they share the same optimal policy as such constraints may be overly stringent. From the agent's perspective, each task is a black box function that takes in the agent's action at (and possibly the whole interaction history) and outputs an observation ot. Both at and ot are strings. In a game of 20 questions, at could be "Is the word an animal?" and the ot could be "No.". In other words, each task employs an environment that the agent interacts with to obtain intermediate observations. An episode contains the agent's interaction trajectory within a single task. Unlike the conventional RL structure, we will assume that the transition-level reward is either 0 or must be inferred from ot, and that the individual tasks can flexibly implement different observation spaces and termination conditions. An episode terminates when the agent achieves the objective of the task or when the maximum number of interactions allowed within the task is reached.

We will use h = (o0, a0, . . . , oH, aH) to denote an episode of length H, ht = (ot, at) to denote a single step of h, and hp:q = (op, ap, . . . , oq, aq) to denote a slice of h similar to array slicing. At the end of an episode, the environment emits a single score, r(h), that evaluates the performance of the agent. Let π denote the LLM agent and h ∼ π ◦ τ denote sampling a trajectory from task τ using policy π. The performance of a policy on a group would be:
Perf(G) = 1 |G| Pτ∈G Eh∼π◦τ [r(h)]. The agent is trained on a finite set of groups, Gtrain, and the goal is to perform well on unseen groups, Gtest.

| Table 1. Summary of the task groups used by PAPRIKA.   |               |              |               |                   |          |
|--------------------------------------------------------|---------------|--------------|---------------|-------------------|----------|
| Task Group                                             | # Train Tasks | # Test Tasks | Maximum Turns | Env Feedback      | Uses COT |
| Twenty questions                                       | 1499          | 367          | 20            | LLM generated     | ✗        |
| Guess my city                                          | 500           | 185          | 20            | LLM generated     | ✗        |
| Wordle                                                 | 1515          | 800          | 6             | Hardcoded program | ✓        |
| Cellular automata                                      | 1000          | 500          | 6             | Hardcoded program | ✓        |
| Customer service                                       | 628           | 200          | 20            | LLM generated     | ✗        |
| Murder mystery                                         | 203           | 50           | 20            | LLM generated     | ✗        |
| Mastermind                                             | 1000          | 500          | 12            | Hardcoded program | ✓        |
| Battleship                                             | 1000          | 200          | 20            | Hardcoded program | ✓        |
| Minesweeper                                            | 1000          | 200          | 20            | Hardcoded program | ✓        |
| Bandit best arm selection                              | 81            | 1            | 21            | Hardcoded program | ✓        |

## 3. P**Aprika**

The goal of our paper is to develop a scalable method to instill better strategic exploration and sequential decisionmaking capabilities into LLMs. Prior works (Krishnamurthy et al., 2024) have shown that LLMs can perform poorly on even the simple decision making task of multi-armed bandits. Nie et al. (2024) has since then demonstrated that LLMs can be taught to perform better on bandits after finetuning them on synthetic trajectories generated by known algorithms such as UCB. However, this idea is limited in scope for three reasons: (1) we want LLMs to perform strategic exploration and decision making in more complex settings, (2) for most tasks, there is no known algorithm like UCB to generate good synthetic trajectories from, (3) it can be infeasible to collect data for all tasks that we care about. We aim to solve these issues using our method, PAPRIKA. First, we design a suite of complex decision-making tasks that require strategic information gathering to succeed. Next, we show that in the absence of known good algorithms, existing LLMs can generate trajectories with better decision making behaviors through diversity-encouraging sampling. We then finetune the LLMs to prefer higher performing trajectories (in a fashion similar to STaR (Zelikman et al., 2022)) and show that this leads to better decision making abilities at test-time. More importantly, these behaviors often generalize to unseen task groups without additional training. Finally, we propose a general curriculum learning algorithm that can dynamically choose which subset of tasks to train on next to improve data efficiency of such training methods. We next describe each component of PAPRIKA.

## 3.1. Task Design

The first component of PAPRIKA is to design a set of task groups that we can evaluate and train LLMs on. The task groups we want should have the following desired properties: (1) they are purely text based, (2) they require multiturn interaction, where the agents have to both understand prior history in its context and choose actions that maximize the probability of success in the future, (3) they are partially observable, i.e., the observations do not capture the full state or hidden information, so the agents must simultaneously explore to reveal more information and exploit to solve the task efficiently, (4) they are diverse and require different strategies to succeed. With these requirements in mind, we design 10 task groups in our paper. On all of them, we employ an LLM as the agent that is given a task it needs to solve through sequential interaction with the task-specific environment, which provides both observations for intermediate timesteps given the agent's actions and also a task reward at the end of an episode. For tasks requiring general knowledge about the world to generate intermediate observations, we employ another LLM (typically GPT-4o-mini) as the environment. For tasks that have rule-based observations and rewards, we find that using hardcoded programs as the verifier/observation generator is more reliable than LLMs, similar to DeepSeek- AI et al. (2025). In order to prevent reward hacking, we also use either another LLM or a hardcoded program as a judge to filter out unsuccessful trajectories that got incorrectly labeled as successful by the task environment (see Appendix D for more on environment hacking). We also find that for task groups requiring complex reasoning, letting the agent think using chain-of-thought (COT) prompting (Wei et al., 2022; Kojima et al., 2022) before generating a final answer improves its performance significantly, similar to ReAct (Yao et al., 2023). We provide a brief description of our task groups here, please refer to Table 1 for their summary and Appendix B for more details. Following prior work (Abdulhai et al., 2023), we include classic guessing games like *twenty questions* and guess my city in our list of task groups. They require guessing a secret topic as quickly as possible by asking a sequence of questions and observing the answers. We also employ *Wordle* and *Mastermind*, where the agent needs to guess a secret 5letter word and 4-digit code respectively. The environments for these task groups provide feedback in terms of similarity between the guess and the target word/code, and the agent needs to refine their guesses in future turns to maximize information gathering. We design *customer service* and murder mystery as dynamic text-based task groups: an LLM plays the role of the task environment, which is provided with the criterion for task success and generates dynamic intermediate observations based on this criterion. A desirable capability in LLMs is to code and refine based on interpreter feedback. To simulate this process with a toy case, we design *Cellular Automata*, where the agent needs to make inferences about the transition rule in 1D elementary cellular automata (Wolfram, 1983; Cook et al., 2004) by observing inputs and outputs. The agent receives the outputs generated from their predicted transition rule and they have to refine their predictions based on it. Next, we incorporate *Minesweeper* and *Battleship* based on classical games, which require the agent to interact with 2D grids to find hidden items within a fixed number of turns and refine their guesses based on per-turn observations.

Finally, we incorporate a modified version of the multiarmed bandit (Slivkins, 2024) task group from prior works (Krishnamurthy et al., 2024; Nie et al., 2024) with the following distinctions: (1) we let the agent employ chainof-thought reasoning before choosing arms so that they can transfer good strategies learned from other tasks, (2) we let the agent interact with the task environment in a multiturn way, (3) instead of reducing regret, we work on the bandit best arm selection (Audibert & Bubeck, 2010; Wang et al., 2024a) problem, where we let the agent choose arms and observe rewards for a fixed number of turns and then measure its accuracy in deciding the arm with the highest reward. This is done to reduce computational cost over generating COTs for a large number of turns, since the difference in regret between different models is not meaningful when the number of turns is not large enough.

## 3.2. Dataset Construction

In order to learn from these task groups, we must first generate data from them. It is crucial that the data we generate are diverse which would allow the model to learn different strategies without the risk of overfitting. We accomplish this by generating a large number of trajectories at a high temperature with Min-p sampling (Nguyen et al., 2024). Min-p sampling works by using an adaptive threshold pscaled ∝ pmax, where pmax is the highest probability predicted by the model on the next token, to truncate the vocabulary to tokens that have a probability larger than pscaled and sample from them - this enables us to generate diverse yet coherent trajectories at a higher temperature. We note that training data generation for PAPRIKA could be improved by adopting more advanced methods for guiding exploration such as Murty et al. (2024); Yang et al. (2024); however, we opt for sampling with high temperature for its simplicity and leave these other options for future work. For each task in a set of chosen tasks (e.g., uniformly sampled), we generate nsample trajectories and then construct a preference pair (hw, hl) where hw is the highest scoring trajectory (trajectory that succeeds and does so at the fewest number of turns) and hlis randomly sampled from the lower scoring (failed or takes substantially more turns to succeed)
trajectories. We choose hl randomly instead of choosing the worst one to increase the diversity of our dataset. We treat hw and hl as proxies for desirable and undesirable behaviors. A dataset D =
nh w, hl(i)oN
i=1 is a collection of such trajectory pairs.

## 3.3. Optimization

Supervised fine-tuning. If we take the winning episodes as the expert behavior, then we can discard the losing episode and maximize the likelihood of winning episodes:

$$\mathcal{L}_{\text{SFT}}(\mathcal{D}_{\text{SFT}})=-\mathbb{E}_{\mathcal{D}_{\text{SFT}}}\left[\frac{1}{\sum_{t=0}^{\left|h_{w}\right|}\left|a_{t}^{w}\right|}\sum_{t=0}^{\left|h_{w}\right|}\log\pi_{\theta}\left(a_{t}^{w}\mid h_{:t}^{w}\right)\right]\tag{1}$$

where DSFT is the dataset used for supervised fine-tuning and |a| is the number of tokens for the agent response (discarding the environment generation). This is akin to rejection sampling fine-tuning (Gulcehre et al., 2023; Dong et al., 2023; Mukobi et al., 2023) seen in prior work. Direct preference optimization. A popular approach for finetuning LLMs is DPO (Rafailov et al., 2024b) where one directly optimizes the Bradley-Terry model (Bradley & Terry, 1952) for preferences. In our setting, each trajectory consists of multiple rounds of interactions so the original DPO objective does not apply. We instead use a multi-turn version of DPO introduced in Rafailov et al. (2024a):

$$\mathcal{L}_{\text{DPO}}(\mathcal{D}_{\text{DPO}})=-\mathbb{E}_{\mathcal{D}_{\text{DPO}}}\Bigg{[}\log\sigma\Bigg{(}\sum_{t=0}^{|h^{w}|}\beta\log\frac{\pi_{\theta}(a_{t}^{w}\mid h_{:t}^{w})}{\pi_{\text{ref}}(a_{t}^{w}\mid h_{:t}^{w})}\Bigg{)}$$ $$-\sum_{t=0}^{|h^{t}|}\beta\log\frac{\pi_{\theta}(a_{t}^{t}\mid h_{:t}^{t})}{\pi_{\text{ref}}(a_{t}^{t}\mid h_{:t}^{t})}\Bigg{)}\Bigg{]}\tag{2}$$

where DDPO is the preference dataset, a w tand a ltare the action tokens generated by the model at turn t in the preferred and dispreferred trajectories, h w and h l, respectively. πref is the reference policy, for which we use the initial model.

The main difference with standard DPO here is that we only calculate the loss on the action tokens - the log probability ratios of the environment generated tokens are not included in the loss. We note that we use DPO because it is less compute intensive. DPO allows us to decouple the data collection and policy improvement steps and offload them on different machines. However, in principle, one could also employ online RL with more resources. Following prior work that shows the efficacy of online RL compared to offline algorithms (Xu et al., 2024; Tajwar et al., 2024), we expect doing PAPRIKA with online RL would lead to even stronger results. Combining objectives. Finally, prior works have noted DPO having the unintended effect of reducing the probability of preferred trajectories as well, known as unintentional unalignment (Razin et al., 2024), which can affect model performance. The RPO objective (Pang et al., 2024), by combining SFT and DPO loss, has shown promising results in mitigating this issue. Formally, the RPO loss is:

$${\cal L}_{\rm RPO}({\cal D}_{\rm DPO})={\cal L}_{\rm DPO}({\cal D}_{\rm DPO})+\alpha{\cal L}_{\rm SFT}({\cal D}_{\rm DPO})\tag{3}$$

where α is a hyper-parameter. Following Pang et al. (2024), we set α to be 1.0 for the rest of this paper.

## 3.4. Scalable Online Curriculum Learning

The core idea of PAPRIKA is to fine-tune the model on a large number of decision making problems to acquire general decision making ability. It is relatively easy to design a large number of tasks, but it is harder to decide which task to train on. A major obstacle is that different tasks may have a large range of difficulty. Unlike pretraining where the model can generally make progress on any given sample (i.e., decrease next-token prediction loss), an RL agent cannot make meaningful progress without collecting good experience. As such, if a task is too difficult for the current model, the model would not generate trajectories with meaningful learning signals. Since generating a trajectory is expensive, it stands to reason that we want to prioritize the tasks where the model can make meaningful progress, which is a form of curriculum learning (Bengio et al., 2009). Without additional assumptions, the only way to know whether a task would yield good learning signals is to actually perform a rollout in that task, which is expensive. In fact, in this particular scenario, the major cost for training is actually data generation rather than model updates. As such, this naive approach would not save us time or computation. A desideratum for an efficient curriculum is the ability to know whether certain tasks will yield data with learning signals without actually performing the rollout. A natural assumption is that similar tasks would have similar levels of learning signal. These groupings can be obtained through Algorithm 1 Task selection with UCB
1: **Input:** Number of arms K, number of samples C, number of rounds T, model π 2: **Initialize:** sk = 0, nk = 0, Buffer 3: for each round t = 1, 2*, . . . , T* do 4: Compute θk =
sk nk
+
q2 log PK
k=1 nk nkfor each k 5: Select k
⋆ = arg maxk θk 6: Sample τ from group k
⋆
7: Sample C trajectories from τ and add to Buffer 8: Compute an estimate for νˆπ(τ ) using Eq 4 9: Update: sk⋆ = sk⋆ + ˆνπ(τ ), nk⋆ = nk⋆ + 1 10: **end for** 11: Construct D from Buffer and train the model π meta data or prior knowledge.2 Measuring learning potential. We will use h ∼ π ◦ τ to denote sampling one episode from the task τ using the policy π. The average performance of π on τ is Rπ(τ ) = Eh∼π◦τ [r(h)] and the variance is σ 2 π(τ ) =
Eh∼π◦τ-(r(h) − Rπ(τ ))2. Based on these, we can define:

$$\nu_{\pi}(\tau)=\frac{\sqrt{\sigma_{\pi}^{2}(\tau)}}{R_{\pi}(\tau)}.$$
(4)  $\frac{1}{2}$ .............................. 
This quantity is known as the coefficient of variation in statistics, a dimensionless quantity that measures the population's variability relative to the mean. We argue that this quantity is an ideal measure of the learning potential for a single task. DPO requires a pair of positive and negative samples 3. Intuitively, the pair should be sufficiently different so the model can tell the two apart - for example, prior work (Pal et al., 2024) has shown that DPO suffers when the edit distance between preferred and dispreferred responses is not large enough. Variance naturally measures the possibility of getting diverse trajectories from sampling. On the other hand, different tasks could have vastly different reward scales. Without loss of generality, if we assume that all rewards are positive, the average reward of each task is a measurement of the reward scale. Normalizing the standard deviation with the reward scale allows us to compare different tasks directly.

Av g S
ucce ss Rat e (
%)
Twenty Questions Av g S
ucce ss Rat e (
%)
Mastermind Av g S
ucce ss Rat e (
%)
Cellular Automata Av g S
ucce ss Rat e (
%)
Battleship Av g S
ucce ss Rat e (
%)
Minesweeper 30 36 42 48 54 0 2 4 6 8 4 7 10 13 16 2 8 14 20 26 8 11 14 17 20 Avg S
uc ces s Ra te (%
)
Customer Service Avg S
uc ces s Ra te (%
)
Murder Mystery Avg S
uc ces s Ra te (%
)
Wordle Avg S
uc ces s Ra te (%
)
Guess My City Avg S
uc ces s Ra te (%
)

Bandit Best Arm Selection 70 74 78 82 86 50 55 60 65 70 4 11 18 25 32 30 40 50 60 70 40 52 64 76 88 Llama-3.1-8B-Instruct Paprika (Llama-3.1-8B-Instruct) Gemma-3-12B-IT Paprika (Gemma-3-12B-IT) gpt-4o-mini
of νπ(τ ) for all tasks in the group G. Given a collection of K groups (G1*, . . . , G*K), a reasonable objective would be to maximize the learning potential of the tasks sampled. This problem can be formulated as a multi-armed bandit (MAB). Many algorithms for MAB exist; for simplicity, we choose the Upper Confidence Bound (Auer, 2000, UCB). We conduct the task selection in a sequential manner using the original UCB algorithm, but we expect a batched variant of UCB could be used to parallelize the experience collection. Each action corresponds to a group of tasks, and we then uniformly sample one task from the chosen group to evaluate the model performance with C rollouts. These statistics are then used to update the mean estimate of that group. After a sufficient amount of episodes are sampled, we construct the dataset and train the model with objectives in Section 3.3. See Algorithm 3.4 for the pseudocode.

Note. An important role of νπ is to make different task groups comparable. The specific selection algorithms could likely be replaced with other more sophisticated online learning methods. More importantly, recent breakthroughs such as OpenAI et al. (2024b) and DeepSeek-AI et al. (2025) mark the beginning of applying RL to a broad range of reasoning problems. Moving forward, we anticipate a proliferation of different RL tasks for LLMs. In this emerging paradigm, a scalable meta algorithm for selecting which tasks to train on will be essential, and we believe PAPRIKA's curriculum learning approach will be a promising foundation for future algorithms.

## 4. Empirical Results

In this section, we will present the results of our empirical study to answer the following research questions: (1) Can training on self-generated trajectories from a diverse range of task groups equip LLMs with sequential decision making capabilities that generalize to unseen task groups without the need to train on them? (2) Can curriculum learning improve the data efficiency of our training mechanism? (3) Finally, does PAPRIKA hurt the model's regular abilities, and can fine-tuning on existing multiturn interaction data that do not have any sequential decision making structure also improve these capabilities? We first describe our experimental setup, and then report our empirical observations.

Experimental Setup. For experiments in this paper, we use Llama-3.1-8B-Instruct (MetaAI et al., 2024) and Gemma-3-12B-IT (Gemma-Team et al., 2025) models. For data generation, we use Min-p sampling (Nguyen et al.,
2024) with temperature 1.5 and Min-p parameter 0.3, as we saw that this setting consistently generated diverse training data that resulted in higher test-time accuracy. For each task in the training split, we generate nsample = 20 trajectories to construct our training dataset (except for mastermind, where we sample nsample = 100 trajectories per task). After filtering, this results in 17,181 training trajectories for supervised fine-tuning and 5,260 trajectory pairs for RPO over all task groups. Unless explicitly mentioned otherwise, we use learning rate of 10−6for supervised fine-tuning and 2 × 10−7for RPO. We use batch size 32 for all training runs. We generally always run supervised fine-tuning first and then further fine-tune with the RPO objective to obtain the final model unless explicitly mentioned otherwise. We use an AdamW optimizer (Loshchilov & Hutter, 2019) with a cosine annealing learning rate scheduler and warmup ratio 0.04 (Loshchilov & Hutter, 2017) to train all our models. During evaluation, in order to account for variability of both the environment and the agent, we generate 4 trajectories for each task in the test set and report the average success rate (we also report pass@4 success rates in Appendix I). We use Min-p sampling with parameter 0.3 for evaluation. Default temperature for evaluation is set to 0.7. Finally,

Avg S ucces s Ra te (
%)Twenty Questions Avg S
ucces s Ra te (
%)Mastermind Avg S
ucces s Ra te (
%)Cellular Automata Avg S ucces s Ra te (
%)Battleship Avg S
ucces s Ra te (
%)Minesweeper 30 33 36 39 42 0 2 4 6 8 6 7 8 9 10 2 4 6 8 10 8 10 12 14 16 Avg S ucces s Ra te (
%)Customer Service Avg S
ucces s Ra te (
%)Murder Mystery Avg S
ucces s Ra te (
%)Wordle Avg S ucces s Ra te (
%)Guess My City Avg S
ucces s Ra te (
%)
Bandit Best Arm Selection 72 73 74 75 76 54 58 62 66 70 4 7 10 13 16 30 35 40 45 50 40 50 60 70 80 Llama-3.1-8B-Instruct Paprika (Full) Paprika (LOO) Paprika (Single Task Group)
for task groups with hardcoded feedback mechanism, we consider a failure to follow formatting instructions to be a failure in the task.

PAPRIKA **improves LLM decision making abilties.** We motivate this question by looking into the toy task group of bandit best arm selection more closely. This task requires strategic use of the fixed sampling budget (20) to quickly discard arms that are unlikely to have a high mean reward, and use most of the sampling budget on the few top arms to decide the best arm among them. Previous work (Nie et al., 2024) has shown that training on synthetic trajectories from optimal bandit algorithms can significantly improve LLMs' performance on them. Contrary to that, we show that LLMs can learn generalizable strategies from other decision making task groups that then transfer to this bandit group, without needing an optimal algorithm to generate synthetic trajectories. Figure 3 shows that PAPRIKA improves average success rate of Llama-3.1-8B-Instruct from 42.25% to 62.25% on the bandit task after only seeing trajectories from other task groups.

Motivated by this, we next study whether PAPRIKA can also improve performance on more complex tasks. Figure 2 shows our main findings: PAPRIKA, when trained on a dataset consisting of filtered trajectories from all 10 task groups, improves the success rate of both Llama-3.18B-Instruct and Gemma-3-12B-It models (see Appendix I for complete results). Averaged across all 10 task groups, PAPRIKA increases the Llama-3.1-8B-Instruct model's performance by 47% of its original success rate after training with only about 22,500 trajectories.

to entirely different groups of tasks. We saw already that PAPRIKA (LOO) improved the success rate on the bandit group without the need to train on it, now we explore this possibility for more complex decision making tasks. To do so, we perform a set of leave-one-out (LOO) experiments: we randomly choose one group (e.g., 20 questions) from our set of task groups, train the LLM on trajectories generated from every other group, and test the resulting model's performance on the left-out group. Additionally, we run an experiment where for each task group, we train and test the LLM on only this single group (using separate splits). We use Llama-3.1-8B-Instruct for this set of experiments.

Figure 3 shows our results: remarkably, we observe that the LOO models can match or sometimes even exceed the performance of group-specific training, demonstrating genuine cross-task group generalization. Concretely, PAPRIKA (LOO) improves success rate on 9 out of 10 task groups compared to the initial model. Moreover, PAPRIKA (full), trained on all 10 task groups, outperform PAPRIKA (Single Task Group) in 7 out of 10 task groups, showing that the model learns better in-group strategies when it observes trajectories from other task groups. Note that we do not expect PAPRIKA (LOO) to always generalize to a new task group. While PAPRIKA (LOO) generalizes better to some task groups vs others (e.g., the improvement on mastermind is minimal), and for some task groups there is no transfer at all or negative transfer (wordle), we hypothesize that scaling up the number of task groups could keep improving LLMs' zero-shot decision-making abilities. Overall, these results demonstrate that PAPRIKA is a potentially scalable solution for teaching LLMs how to do in-context RL.

PAPRIKA **can teach LLMs generalizable strategies.**
The next important question we want to study is whether the strategies learned by PAPRIKA can zero-shot transfer Curriculum learning can improve data efficiency of PA-
PRIKA. The biggest bottleneck of PAPRIKA is the time required to generate a large number of trajectories for each.

Average Success Rate Pass@4 Success Rate Per Group Success Rate 45 0 1 2 3 Training Round 60 65 70 75 Easy Medium Hard Task Groups 0 15 30 45 60 Av g Su cc ess R
at e ( %)
S
ucc es s Rate 
(
%
)

S
ucc es s Rate 
(
%
)

40 0 1 2 3 Training Round 35 Curriculum Uniform Llama-3.1-8B-Instruct
Some tasks are naturally harder than others, which means that spending an equal sampling budget on the harder tasks gives us a smaller learning signal. We study a curriculum learning version of PAPRIKA where we have a grouping over our tasks according to task difficulty. For this, we use GPT-4o-mini to classify the tasks in twenty questions into 3 categories: easy, medium, and hard. This results in 477 easy, 726 medium, and 296 hard topics in the train split and 127 easy, 172 medium, and 68 hard topics in the test split. Next, we run the curriculum learning algorithm described in Section 3.4 for 3 rounds on a Llama-3.1-8B-Instruct model: at each round, we sample 250 tasks from the train set according to Section 3.4. We use the number of turns it took the agent to solve a task across multiple trajectories as a proxy for reward in Equation (4) to calculate νπ (see Appendix H
for more details). 20 trajectories are generated for each task using the previous round's model checkpoint and we train that checkpoint on the resulting dataset (for DPO, we use the prior round's checkpoint instead of the initial model as the reference policy). We compare our curriculum against the baseline of sampling 250 tasks uniformly at random from the train set at each round. Figure 4 shows our results: after three rounds of training, our curriculum outperforms uniform sampling by 1.4% and 3.3% at average and pass@4 accuracy respectively.

## 4.1. Analysis

PAPRIKA **improves LLMs' task efficiency.** In this section, we want to analyze the sequential decision-making abilities learned by PAPRIKA beyond just success rate on individual task groups. Note that our tasks are designed in a way such that an agent capable of better strategic exploration would solve them faster, eg., an agent capable of asking better yes/no questions would guess the secret topic using fewer number of turns. We leverage this property of our tasks and conduct both quantitative and qualitative analysis on the behaviors of the regular instruct model and PAPRIKA - (1) Figure 7 shows that PAPRIKA reduces the average number of turns it takes for the agent to solve tasks, implying that PAPRIKA is choosing more optimal actions at intermediate steps, (2) Appendix K shows qualitative difference between the behavior of the regular instruct model and PAPRIKA on twenty questions and wordle, with PAPRIKA
generally generating more sensible responses.

PAPRIKA **does not hurt LLMs' regular capabilities.** We have demonstrated the efficacy of PAPRIKA in instilling decision making capabilities into LLMs efficiently. However, to scale up PAPRIKA, one would potentially use online reinforcement learning on such decision making tasks, and an important question is whether PAPRIKA fine-tuning would hurt the LLM's regular capabilities which would hinder scaling it up. To study this question, we run a set of standard evaluations (see Appendix I.12) on our PAPRIKA fine-tuned model and compare its performance against Llama-3.1-8B- Instruct. Table 2 shows our findings: PAPRIKA does not result in any noticeable performance degradation.

## 5. Related Works

LLM alignment. Alignment or post-training is a crucial step for creating helpful LLM assistant. Existing posttraining pipeline typically involves instruction tuning and then reinforcement learning from human feedback (Christiano et al., 2017, RLHF) where one either performs RL against a reward model trained on human preference data via Proximal Policy Optimization (Schulman et al., 2017, PPO)
or sidesteps reward model training via Direct Preference Optimization (Rafailov et al., 2024b, DPO). Most methods

| Model                 | MT-Bench    | AlpacaEval   | GPQA       | Math (Hard)   | MMLU-Pro   | IFEval     |
|-----------------------|-------------|--------------|------------|---------------|------------|------------|
| Llama-3.1-8B-Instruct | 7.88        | 33.6         | 33.5       | 24.6          | 46.7       | 84.4       |
| + PAPRIKA             | 8.14 (0.03) | 33.5 (0.3)   | 32.8 (1.5) | 25.3 (0.3)    | 46.2 (0.1) | 85.4 (0.3) |

focus on *single-turn* interactions where the model generates a single response to a query. We focus on the *multi-turn* setting where the agent has to interact with an environment iteratively, similar to Rafailov et al. (2024a). There are a few existing environments and datasets that focus on multiturn interactions (Abdulhai et al., 2023; Sun et al., 2023; Kwan et al., 2024; Wang et al., 2024b). LMRL-Gym (Abdulhai et al., 2023) implements a suite of textual RL environment, some of which we build on. Concurrent work such as Narayanan et al. (2024) has designed environments based on scientific tasks (such as molecule cloning and protein stability) for LLMs to interact with and showed that behavior cloning and expert iteration (Anthony et al., 2017; 2019; Havrilla et al., 2024) can improve an LLM's multiturn interaction capabilities on these tasks. Most of these environments focus on interactions with humans. Rather than any particular task, we focus on evaluating LLMs' general ability to solve sequential decision making problems where the agent needs to explore and exploit. In-context reinforcement learning. In-context learning (ICL) is the ability where LLMs can learn a new task from a small number of demonstrations without any gradient update (Brown et al., 2020). Existing ICL usually focuses on a single-turn interaction. We focus on in-context reinforcement learning (Laskin et al., 2022; Raparthy et al., 2023; Lee et al., 2024; Lin et al., 2024) instead. Existing work in this field has focused on environments where RL is conventionally applied (e.g., grid world, bandits, and maze) (Monea et al., 2025), and the training data are generated by either random policies or pre-existing RL algorithms. In comparison, we focus on diverse environments and study how well the decision making abilities generalize to completely new environments. Concurrent work has also studied improving LLMs' information seeking abilities (Li et al., 2025) for medical reasoning, whereas we work on general information seeking abilities applicable to a diverse range of tasks. Moreover, Harris & Slivkins (2025) has studied using an LLM to assist a decision-making agent navigate explorationexploitation tradeoff, whereas we use an LLM directly as the decision making agent and teach it this capability. Curriculum learning in RL. Curriculum learning (Bengio et al., 2009) shows the data to the model in a nonuniform order. This idea is inspired by the fact that humans tend to learn skills in a sequential order (Skinner, 1958),
and is particularly appealing for RL because learning easier tasks first could build scaffold toward solving difficult tasks that the agent could not solve otherwise (Andrychowicz et al., 2017; Florensa et al., 2017; Fang et al., 2019; Portelas et al., 2020a). Concurrent work such as Foster & Foerster (2025) has studied curriculum learning for training LLMs to improve their reasoning capabilities. While their work requires generating rollouts per each example to determine the learnability, we show that given access to some grouping metadata, one can design an effective curriculum using only a constant number of rollouts generated from each task group. Another related line of work is environment design, where a second process controls the distribution over different environments or directly generates environments in a procedural manner to maximize various notions of learning progress (Wang et al., 2019; Dennis et al., 2020; Jiang et al., 2021b;a; Bruce et al., 2024). Since this is a field of extensive existing literature, we refer the interested reader to Portelas et al. (2020b) for a comprehensive survey.

## 6. Discussion

In this paper, we presented a scalable fine-tuning method to improve multi-turn decision making abilities of LLMs. Moreover, we showed that the strategies learned by the LLM from our method can generalize zero-shot to unseen tasks. There are a few limitations to our approach. Firstly, we use rejection sampling on self-generated data to teach the model better behaviors. In order to get good performance, the starting model need to exhibit good behavior within a reasonable generation budget, so PAPRIKA would perform worse in the absence of a good base model. Next, we use offline preference tuning algorithms to train our models due to the lack of computational resources. A possible future direction for our work is to run online RL on diverse tasks instead: due to its recent success in other domains (DeepSeek-AI et al., 2025), we expect it will give a larger improvement in LLMs' in-context RL capabilities. Our environments, despite being designed with the help of GPT-4o-mini, required a lot of human effort for implementation. A new axis of improvement can be training an LLM to scalably generate suitable tasks that can then be used to train the agent. Finally, the performance of our curriculum learning algorithm heavily depends on the quality of the task group clusters which is not ideal, and one can study possible improvements of this algorithm. We leave these directions for future work.

## Impact Statement References

Our work can be used to train large language models that have better strategic exploration and decision making capabilities, which can have potential impact in the real world if agentic systems become wide spread. Our experiments are conducted in relatively simple and controlled environments and it is an open question what kind of impacts truly agentic systems will have on society. Other than that, this paper presents work whose goal is to advance the field of Machine Learning. There are many potential overall societal consequences of our work, none of which we feel must be specifically highlighted here.

## Reproducibility Statement

We provide sufficient details about our implementation, hyperparameters, environment design and dataset construction in the main paper and the appendix to effectively reproduce the results in this paper. Our code, training dataset and models can be found via the project website: https://paprika-llm.github.io/

## Acknowledgement

This work was supported in part by the U.S. Army Futures Command under Contract No. W519TC-23-C-0030. Moreover, it has greatly benefited from using the Delta advanced computing and data resource supported by the National Science Foundation (OAC 2005572) and the State of Illinois, as part of ACCESS-approved compute grants (Boerner et al.,
2023). Subsequent larger scale experiments on Gemma3-12B-IT models were run using Bridges-2 (Brown et al., 2021) at Pittsburgh Supercomputing Center through AC-
CESS allocation CIS240901 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants \#2138259, \#2138286, \#2138307, \#2137603, and \#2138296. The authors thank Brandon Pusateri, Jillian Lehosky and Greg Bauer from ACCESS Support Staff for their incredible help at approving supplements and renewals for ACCESS compute grants throughout this project. Moreover, the work would not have finished so quickly without the help of Brett Bode from NCSA Delta Support Staff, who provided the authors critical help about properly utilizing the Delta cluster. FT and YJ gratefully acknowledge Samuel Sokota, Daman Arora, Andrea Zanette, Yuda Song, Gaurav Ghosal, Yutong He, So Yeon Min, Kevin Li, Wen-Tse Chen, Xintong Duan and other members of Russ, Auton, Locus and AIRe lab for feedback received on an earlier versions of this work. FT greatly benefited from his discussions with Prof. Aviral Kumar and his lab's computational resources. YJ gratefully acknowledges the support of the Google PhD Fellowship.

Abdulhai, M., White, I., Snell, C., Sun, C., Hong, J., Zhai, Y., Xu, K., and Levine, S. Lmrl gym: Benchmarks for multi-turn reinforcement learning with language models. arXiv preprint arXiv:2311.18232, 2023.

Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Pieter Abbeel, O.,
and Zaremba, W. Hindsight experience replay. Advances in neural information processing systems, 30, 2017.

Anthony, T., Tian, Z., and Barber, D. Thinking fast and slow with deep learning and tree search, 2017. URL
https://arxiv.org/abs/1705.08439.

Anthony, T., Nishihara, R., Moritz, P., Salimans, T., and Schulman, J. Policy gradient search: Online planning and expert iteration without search trees, 2019. URL
https://arxiv.org/abs/1904.03646.

Audibert, J.-Y. and Bubeck, S. Best Arm Identification in Multi-Armed Bandits. In *COLT 2010 - Proceedings*, pp. 13 p., Haifa, Israel, June 2010. URL https://enpc. hal.science/hal-00654404.

Auer, P. Using upper confidence bounds for online learning.

In Proceedings 41st annual symposium on foundations of computer science, pp. 270–279. IEEE, 2000.

Auer, P., Cesa-Bianchi, N., and Fischer, P. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47:235–256, 2002.

Beck, J., Vuorio, R., Liu, E. Z., Xiong, Z., Zintgraf, L., Finn, C., and Whiteson, S. A survey of meta-reinforcement learning. *arXiv preprint arXiv:2301.08028*, 2023.

Bengio, Y., Louradour, J., Collobert, R., and Weston, J.

Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pp. 41–48, 2009.

Boerner, T. J., Deems, S., Furlani, T. R., Knuth, S. L.,
and Towns, J. Access: Advancing innovation: Nsf's advanced cyberinfrastructure coordination ecosystem: Services & support. In Practice and Experience in Advanced Research Computing 2023: Computing for the Common Good, PEARC '23, pp. 173–176, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450399852. doi: 10.1145/ 3569951.3597559. URL https://doi.org/10. 1145/3569951.3597559.

Bradley, R. A. and Terry, M. E. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Brown, S. T., Buitrago, P., Hanna, E., Sanielevici, S.,
Scibek, R., and Nystrom, N. A. Bridges-2: A platform for rapidly-evolving and data intensive research. In Practice and Experience in Advanced Research Computing 2021: Evolution Across All Dimensions, PEARC
'21, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450382922. doi: 10. 1145/3437359.3465593. URL https://doi.org/ 10.1145/3437359.3465593.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D.,
Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners.

Advances in neural information processing systems, 33:
1877–1901, 2020.

Bruce, J., Dennis, M. D., Edwards, A., Parker-Holder, J.,
Shi, Y., Hughes, E., Lai, M., Mavalankar, A., Steigerwald, R., Apps, C., et al. Genie: Generative interactive environments. In *Forty-first International Conference on* Machine Learning, 2024.

Burda, Y., Edwards, H., Storkey, A., and Klimov, O. Exploration by random network distillation. *arXiv preprint* arXiv:1810.12894, 2018.

Chen, J., Qadri, R., Wen, Y., Jain, N., Kirchenbauer, J.,
Zhou, T., and Goldstein, T. Genqa: Generating millions of instructions from a handful of prompts, 2024. URL https://arxiv.org/abs/2406.10323.

Chen, R. Y., Sidor, S., Abbeel, P., and Schulman, J.

Ucb exploration via q-ensembles. arXiv preprint arXiv:1706.01502, 2017.

Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Cook, M. et al. Universality in elementary cellular automata.

Complex systems, 15(1):1–40, 2004.

Cotˆ e, M.-A., ´ Akos K ´ ad´ ar, Yuan, X., Kybartas, B., Barnes, ´
T., Fine, E., Moore, J., Tao, R. Y., Hausknecht, M., Asri, L. E., Adada, M., Tay, W., and Trischler, A. Textworld: A learning environment for text-based games, 2019. URL https://arxiv.org/abs/1806.11532.

Dao, T. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Re, C. FlashAt- ´
tention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J.,
Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https:// arxiv.org/abs/2501.12948.

Dennis, M., Jaques, N., Vinitsky, E., Bayen, A., Russell, S., Critch, A., and Levine, S. Emergent complexity and zero-shot transfer via unsupervised environment design. Advances in neural information processing systems, 33: 13049–13061, 2020.

Dong, H., Xiong, W., Goyal, D., Zhang, Y., Chow, W.,
Pan, R., Diao, S., Zhang, J., SHUM, K., and Zhang, T.

RAFT: Reward ranked finetuning for generative foundation model alignment. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https: //openreview.net/forum?id=m7p5O7zblY.

Dubois, Y., Li, X., Taori, R., Zhang, T., Gulrajani, I., Ba, J., Guestrin, C., Liang, P., and Hashimoto, T. B. Alpacafarm: A simulation framework for methods that learn from human feedback, 2023.

Dubois, Y., Galambosi, B., Liang, P., and Hashimoto, T. B.

Length-controlled alpacaeval: A simple way to debias automatic evaluators. *arXiv preprint arXiv:2404.04475*,
2024.

Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., and Kiela, D. Kto: Model alignment as prospect theoretic optimization, 2024. URL https://arxiv.org/abs/ 2402.01306.

Eysenbach, B., Gupta, A., Ibarz, J., and Levine, S. Diversity is all you need: Learning skills without a reward function. arXiv preprint arXiv:1802.06070, 2018.

Fang, M., Zhou, T., Du, Y., Han, L., and Zhang, Z.

Curriculum-guided hindsight experience replay. Advances in neural information processing systems, 32, 2019.

Florensa, C., Held, D., Wulfmeier, M., Zhang, M., and Abbeel, P. Reverse curriculum generation for reinforcement learning. In *Conference on robot learning*, pp. 482– 495. PMLR, 2017.

Foster, T. and Foerster, J. Learning to reason at the frontier of learnability, 2025. URL https://arxiv.org/ abs/2502.12272.

Gemma-Team, Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Rame, A., ´ Riviere, M., Rouillard, L., Mesnard, T., et al. Gemma 3 ` technical report, 2025. URL https://arxiv.org/
abs/2503.19786.

Gulcehre, C., Paine, T. L., Srinivasan, S., Konyushkova, K.,
Weerts, L., Sharma, A., Siddhant, A., Ahern, A., Wang, M., Gu, C., Macherey, W., Doucet, A., Firat, O., and de Freitas, N. Reinforced self-training (rest) for language modeling, 2023. URL https://arxiv.org/abs/ 2308.08998.

Jiang, M., Grefenstette, E., and Rocktaschel, T. Prioritized ¨
level replay. In International Conference on Machine Learning, pp. 4940–4950. PMLR, 2021b.

Jiang, Y., Kolter, J. Z., and Raileanu, R. On the importance of exploration for generalization in reinforcement learning. *Advances in Neural Information Processing Systems*, 36:12951–12986, 2023b.

Harris, K. and Slivkins, A. Should you use your large language model to explore or exploit?, 2025. URL https://arxiv.org/abs/2502.00225.

Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35: 22199–22213, 2022.

Hausknecht, M., Ammanabrolu, P., Cotˆ e, M.-A., and Yuan, ´
X. Interactive fiction games: A colossal adventure, 2020a.

URL https://arxiv.org/abs/1909.05398.

Krishnamurthy, A., Harris, K., Foster, D. J., Zhang, C.,
and Slivkins, A. Can large language models explore incontext?, 2024. URL https://arxiv.org/abs/ 2403.15371.

Hausknecht, M., Ammanabrolu, P., Cotˆ e, M.-A., and Yuan, ´
X. Interactive fiction games: A colossal adventure, 2020b.

URL https://arxiv.org/abs/1909.05398.

Kwan, W.-C., Zeng, X., Jiang, Y., Wang, Y., Li, L., Shang, L., Jiang, X., Liu, Q., and Wong, K.-F. Mt-eval: A multiturn capabilities evaluation benchmark for large language models. *arXiv preprint arXiv:2401.16745*, 2024.

Havrilla, A., Du, Y., Raparthy, S. C., Nalmpantis, C., Dwivedi-Yu, J., Zhuravinskyi, M., Hambro, E., Sukhbaatar, S., and Raileanu, R. Teaching large language models to reason with reinforcement learning, 2024. URL
https://arxiv.org/abs/2403.04642.

Laskin, M., Wang, L., Oh, J., Parisotto, E., Spencer, S.,
Steigerwald, R., Strouse, D., Hansen, S., Filos, A., Brooks, E., et al. In-context reinforcement learning with algorithm distillation. *arXiv preprint arXiv:2210.14215*, 2022.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset, 2021.

URL https://arxiv.org/abs/2103.03874.

Lee, J., Xie, A., Pacchiano, A., Chandak, Y., Finn, C.,
Nachum, O., and Brunskill, E. Supervised pretraining can learn in-context reinforcement learning. Advances in Neural Information Processing Systems, 36, 2024.

Hurst, A., Lerer, A., Goucher, A. P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Li, S. S., Mun, J., Brahman, F., Ilgen, J. S., Tsvetkov, Y., and Sap, M. Aligning llms to ask good questions a case study in clinical reasoning, 2025. URL https://arxiv.

org/abs/2502.14860.

Jansen, P., Cotˆ e, M.-A., Khot, T., Bransom, E., Mishra, ´
B. D., Majumder, B. P., Tafjord, O., and Clark, P. Discoveryworld: A virtual environment for developing and evaluating automated scientific discovery agents, 2024. URL https://arxiv.org/abs/2406.06769.

Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I.,
Guestrin, C., Liang, P., and Hashimoto, T. B. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/ alpaca_eval, 5 2023.

Jansen, P. A. A systematic survey of text worlds as embodied natural language environments, 2021. URL https:// arxiv.org/abs/2107.04132.

Lin, L., Bai, Y., and Mei, S. Transformers as decision makers: Provable in-context reinforcement learning via supervised pretraining. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=yN4Wv17ss3.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C.,
Chaplot, D. S., de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux, M.- A., Stock, P., Scao, T. L., Lavril, T., Wang, T., Lacroix, T., and Sayed, W. E. Mistral 7b, 2023a. URL https: //arxiv.org/abs/2310.06825.

Loshchilov, I. and Hutter, F. Sgdr: Stochastic gradient descent with warm restarts, 2017. URL https: //arxiv.org/abs/1608.03983.

Jiang, M., Dennis, M., Parker-Holder, J., Foerster, J., Grefenstette, E., and Rocktaschel, T. Replay-guided adversarial ¨
environment design. Advances in Neural Information Processing Systems, 34:1884–1897, 2021a.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization, 2019. URL https://arxiv.org/abs/
1711.05101.

MetaAI, Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A.,
Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., et al. The llama 3 herd of models, 2024. URL https:// arxiv.org/abs/2407.21783.

Monea, G., Bosselut, A., Brantley, K., and Artzi, Y. Llms are in-context bandit reinforcement learners, 2025. URL https://arxiv.org/abs/2410.05362.

Mukobi, G., Chatain, P., Fong, S., Windesheim, R., Kutyniok, G., Bhatia, K., and Alberti, S. Superhf: Supervised iterative learning from human feedback, 2023. URL
https://arxiv.org/abs/2310.16763.

Murty, S., Manning, C. D., Shaw, P., Joshi, M., and Lee, K. BAGEL: Bootstrapping agents by guiding exploration with language. In Forty-first International Conference on Machine Learning, 2024. URL https: //openreview.net/forum?id=VsvfSMI5bs.

Narayanan, S., Braza, J. D., Griffiths, R.-R., Ponnapati, M.,
Bou, A., Laurent, J., Kabeli, O., Wellawatte, G., Cox, S., Rodriques, S. G., and White, A. D. Aviary: training language agents on challenging scientific tasks, 2024. URL https://arxiv.org/abs/2412.21154.

Nguyen, M., Baker, A., Neo, C., Roush, A., Kirsch, A., and Shwartz-Ziv, R. Turning up the heat: Min-p sampling for creative and coherent llm outputs. *arXiv preprint* arXiv:2407.01082, 2024.

Nie, A., Su, Y., Chang, B., Lee, J. N., Chi, E. H., Le, Q. V.,
and Chen, M. Evolve: Evaluating and optimizing llms for exploration, 2024. URL https://arxiv.org/ abs/2410.06238.

OpenAI, Achiam, J., Adler, S., Agarwal, S., Ahmad, L.,
Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., et al. Gpt-4 technical report, 2024a. URL https:
//arxiv.org/abs/2303.08774.

OpenAI, Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-
Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. Openai o1 system card. *arXiv preprint* arXiv:2412.16720, 2024b.

Osband, I., Blundell, C., Pritzel, A., and Van Roy, B. Deep exploration via bootstrapped dqn. Advances in neural information processing systems, 29, 2016.

Pal, A., Karkhanis, D., Dooley, S., Roberts, M., Naidu, S.,
and White, C. Smaug: Fixing failure modes of preference optimisation with dpo-positive, 2024. URL https:// arxiv.org/abs/2402.13228.

Pang, R. Y., Yuan, W., Cho, K., He, H., Sukhbaatar, S.,
and Weston, J. Iterative reasoning preference optimization, 2024. URL https://arxiv.org/abs/2404.

19733.

Pathak, D., Agrawal, P., Efros, A. A., and Darrell, T.

Curiosity-driven exploration by self-supervised prediction. In *International conference on machine learning*, pp. 2778–2787. PMLR, 2017.

Pathak, D., Gandhi, D., and Gupta, A. Self-supervised exploration via disagreement. In International conference on machine learning, pp. 5062–5071. PMLR, 2019.

Portelas, R., Colas, C., Hofmann, K., and Oudeyer, P.-Y.

Teacher algorithms for curriculum learning of deep rl in continuously parameterized environments. In Conference on Robot Learning, pp. 835–853. PMLR, 2020a.

Portelas, R., Colas, C., Weng, L., Hofmann, K., and Oudeyer, P.-Y. Automatic curriculum learning for deep rl: A short survey. *arXiv preprint arXiv:2003.04664*, 2020b.

Qwen, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B.,
Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., et al. Qwen2.5 technical report, 2025. URL https://arxiv.org/ abs/2412.15115.

Rafailov, R., Hejna, J., Park, R., and Finn, C. From r to q
∗: Your language model is secretly a q-function, 2024a.

URL https://arxiv.org/abs/2404.12358.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization:
Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024b.

Raparthy, S. C., Hambro, E., Kirk, R., Henaff, M., and Raileanu, R. Generalization to new sequential decision making tasks with in-context learning. *arXiv preprint* arXiv:2312.03801, 2023.

Razin, N., Malladi, S., Bhaskar, A., Chen, D., Arora, S., and Hanin, B. Unintentional unalignment: Likelihood displacement in direct preference optimization, 2024. URL https://arxiv.org/abs/2410.08847.

Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y.,
Dirani, J., Michael, J., and Bowman, S. R. Gpqa: A
graduate-level google-proof q&a benchmark, 2023. URL https://arxiv.org/abs/2311.12022.

Schmidhuber, J. Curious model-building control systems. In Proc. international joint conference on neural networks, pp. 1458–1463, 1991.

Schmidhuber, J. Godel machines: Fully self-referential ¨
optimal universal self-improvers. In Artificial general intelligence, pp. 199–226. Springer, 2007.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Sharma, A., Gu, S., Levine, S., Kumar, V., and Hausman, K.

Dynamics-aware unsupervised discovery of skills. arXiv preprint arXiv:1907.01657, 2019.

Skinner, B. F. Reinforcement today. *American Psychologist*,
13(3):94, 1958.

Slivkins, A. Introduction to multi-armed bandits, 2024.

URL https://arxiv.org/abs/1904.07272.

Sokal, R. and Rohlf, F. Biometry : the principles and practice of statistics in biological research / robert r. sokal and f. james rohlf, 04 2013.

Sun, Y., Liu, C., Huang, J., Song, R., Zhang, F., Zhang, D., Wang, Z., and Gai, K. Parrot: Enhancing multi-turn chat models by learning to ask questions. arXiv preprint arXiv:2310.07301, 2023.

Sutton, R. S., Barto, A. G., et al. Reinforcement learning:
An introduction, volume 1. MIT press Cambridge, 1998.

Tajwar, F., Singh, A., Sharma, A., Rafailov, R., Schneider, J., Xie, T., Ermon, S., Finn, C., and Kumar, A. Preference fine-tuning of llms should leverage suboptimal, on-policy data, 2024. URL https://arxiv.org/abs/2404. 14367.

Thompson, W. R. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika*, 25(3-4):285–294, 1933.

Wang, P.-A., Tzeng, R.-C., and Proutiere, A. Best arm identification with fixed budget: A large deviation perspective, 2024a. URL https://arxiv.org/abs/ 2312.12137.

Wang, R., Lehman, J., Clune, J., and Stanley, K. O. Paired open-ended trailblazer (poet): Endlessly generating increasingly complex and diverse learning environments and their solutions. *arXiv preprint arXiv:1901.01753*, 2019.

Wang, R., Jansen, P., Cotˆ e, M.-A., and Ammanabrolu, P. Sci- ´
enceworld: Is your agent smarter than a 5th grader?, 2022. URL https://arxiv.org/abs/2203.07540.

Wang, X., Wang, Z., Liu, J., Chen, Y., Yuan, L., Peng, H.,
and Ji, H. MINT: Evaluating LLMs in multi-turn interaction with tools and language feedback. In The Twelfth International Conference on Learning Representations, 2024b. URL https://openreview.net/forum?

id=jp3gWrMuIZ.

Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., Li, T., Ku, M., Wang, K., Zhuang, A., Fan, R., Yue, X., and Chen, W. Mmlu-pro: A more robust and challenging multitask language understanding benchmark, 2024c. URL https://arxiv.org/abs/2406.01574.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Wolfram, S. Statistical mechanics of cellular automata.

Reviews of modern physics, 55(3):601, 1983.

Xu, S., Fu, W., Gao, J., Ye, W., Liu, W., Mei, Z., Wang, G., Yu, C., and Wu, Y. Is dpo superior to ppo for llm alignment? a comprehensive study, 2024. URL https: //arxiv.org/abs/2404.10719.

Yang, Z., Li, P., Yan, M., Zhang, J., Huang, F., and Liu, Y. React meets actre: Autonomous annotation of agent trajectories for contrastive self-training. In First Conference on Language Modeling, 2024. URL https: //openreview.net/forum?id=0VLBwQGWpA.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. React: Synergizing reasoning and acting in language models, 2023. URL https://arxiv. org/abs/2210.03629.

Zelikman, E., Wu, Y., Mu, J., and Goodman, N. D. Star:
Bootstrapping reasoning with reasoning, 2022. URL https://arxiv.org/abs/2203.14465.

Zhao, W., Ren, X., Hessel, J., Cardie, C., Choi, Y., and Deng, Y. Wildchat: 1m chatgpt interaction logs in the wild, 2024.

URL https://arxiv.org/abs/2405.01470.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z.,
Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., and Stoica, I. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023. URL https: //arxiv.org/abs/2306.05685.

Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., Zhou, D., and Hou, L. Instruction-following evaluation for large language models, 2023. URL https: //arxiv.org/abs/2311.07911.

## A. Note On Curiosity

The concept of curiosity has been used in many different machine learning contexts. A popular notion of curiosity is intrinsic motivation, where the agent is driven by an exploration bonus that is not necessarily related to the task to be achieved (Schmidhuber, 1991; 2007). Many works build on this notion to handle problems with sparse reward or no reward at all (Pathak et al., 2017; Eysenbach et al., 2018; Burda et al., 2018; Sharma et al., 2019; Pathak et al., 2019). The curiosity in this work differs from intrinsic motivation in that we focus on gathering only the information required to solve a given task rather than all the knowable information. This is closer in spirit to the original exploration-exploitation trade-off in reinforcement learning (Sutton et al., 1998; Auer et al., 2002; Thompson, 1933). The goal is to explore to the extent that the problem can be solved but not over-explore at the cost of efficiency. Most existing works based on this principle are tabula rasa (Osband et al., 2016; Chen et al., 2017). This class of exploration algorithms has been shown to improve the generalization ability of non-LLM-based RL agents (Jiang et al., 2023b). PAPRIKA differs from these approaches by learning good exploration strategies from trajectories from many different environments to make exploration on a new problem more efficient. This can be thought of as a form of *amortized exploration*.

## B. Details On Task Design B.1. Summary Of Task Groups

Twenty questions: Twenty questions challenges the agent to identify a secret topic by asking up to 20 yes-or-no questions. The goal is to guess the topic in as few questions as possible by interpreting previous answers and strategizing to maximize information gained. Twenty questions has been studied in prior benchmarks such as LMRL-Gym (Abdulhai et al., 2023): here we expand upon their environment with a more diverse and difficult set of secret topics. Our secret topics come from a diverse range of scenarios, including famous people, historical events, scientific concepts, locations, etc. Each secret topic corresponds to a task, and we have generated a set of 1499 train and 367 test tasks. In order to generate a diverse set of topics, we use prompting techniques from GenQA (Chen et al., 2024) on GPT-4o-mini. The topics to guess in our training and test sets are distinct from one another and also the set of topics included in LMRL-Gym (159 topics), which use as an additional evaluation set. We use GPT-4o-mini (Hurst et al., 2024; OpenAI et al., 2024a) as the task environment to provide yes/no answers at every turn, and also as a judge to make sure task success label is correct. We use strict string matching to make sure the intermediate observations are only 'yes', 'no' or 'Goal reached'. We also maintain train and test set separation to accurately test generalization unlike previous works. Guess my city: Following LMRL-Gym, this task group requires the agent to guess a secret city after asking a maximum of 20 questions. But unlike twenty questions, the questions here can be broader than just yes/no questions, for example,
"*What is your city most popular for?*" so long as the answer to the question does not reveal the name of the city directly. We generated a train set of 500 and test set of 185 distinct cities using GPT-4o-mini and GenQA (Chen et al., 2024) prompting techniques. In addition, we also evaluated our models on the list of 91 cities from LMRL-Gym, which does not overlap with our training and test set. We maintain train and test set separation. Customer service: In this task group, we test for efficient directed exploration —- the LLM must act as a support agent who asks maximally informative questions to diagnose problems and minimize the number of interactions needed to resolve the customer's query. To do so, we simulate realistic troubleshooting scenarios ranging from electronic device issues to automobile maintenance. We use GPT-4o-mini to simulate a customer with limited technical expertise, and use another LLM to act as a customer service agent whose role is to listen to the responses from the customer and suggest a sequence of actions that lead to solving the customer's problem in as few turns as possible. The customer service troubleshooting scenarios are generated by GPT-4o-mini, using prompting techniques from GenQA. Murder mystery: Text-based interactive fiction (IF) environments can be a good benchmark to test LLMs' decision making and interaction abilities. Inspired by Hausknecht et al. (2020a), we design our murder mystery task group, where an LLM is given a crime scene with a possible list of suspects, witnesses, and clues, and it needs to take actions to uncover more information to successfully determine the culprit. The environments provided in Hausknecht et al. (2020a) proved difficult to incorporate directly in our setup, since they have a predefined list of valid actions and uses text-based parsing on the LLM generation to match against the list, making it difficult for LLMs to play the games. Instead, we use GPT-4o-mini to simulate the environment that can provide dynamic feedback to the agent's actions. The murder mystery scenarios are generated by GPT-4o-mini, using prompting techniques from GenQA.

Wordle: Wordle tests an LLM's deductive reasoning abilities. The agent must guess a secret 5-letter word within 6 attempts. After each guess, the environment provides feedback for each letter: correct letter in correct position, correct letter in wrong position, or letter not in the word. The agent must use this feedback strategically to maximize information gained with each guess. We found that LLMs like GPT-4o-mini cannot generate accurate environment feedback for Wordle, so we use hardcoded rules to generate it instead. We also saw that prompting the LLM agent to do chain-of-thought reasoning before outputting its final guess significantly improves its performance, so we use that here unlike the environments above. The secret words are generated by looking at 5-letter words from an English dictionary. Cellular Automata: A key trait of LLM agents is the ability to code and refine based on interpreter feedback. To model this, we create a cellular automata-based environment. Here, a binary string (e.g., 1010) represents cells, and a transition rule defines a cell's next state based on itself and its neighbors (e.g., 100: 1 means a 0 cell with 1 and 0 neighbors turns into 1). We randomly select a transition rule (one of 256) and up to three input strings and their corresponding outputs generated by the transition rule. The LLM must infer the rule by analyzing input-output pairs. If its guess generates correct outputs, it wins; otherwise, it gets feedback and can refine its guess. The task ends in failure if the correct rule isn't found within six turns. We use chain-of-thought prompting for the agent and a hardcoded program to generate environment feedback. The tasks are generated by sampling transition rules and inputs randomly. Mastermind: Similar to Wordle, Mastermind challenges agents to deduce a 4-digit secret code within 12 turns. After each guess, environment feedback indicates two values: the number of digits that are correct and in the right position (exact matches), and the number of digits that appear in the code but in wrong positions (partial matches). Agents must use this feedback to iteratively refine subsequent guesses. We use chain-of-thought prompting for the agent and a hardcoded program to generate environment feedback. The tasks are generated by randomly sampling (without replacement) secret codes from all possible 10,000 four digit codes. Battleship: Battleship tests an LLM's ability to balance exploration and exploitation. The environment features a 2D square grid where three ships are hidden: a carrier (5 cells), a battleship (4 cells), and a destroyer (2 cells). Ships are placed horizontally or vertically. At each turn, the agent targets one cell with a missile. The environment environment reports either a hit (including the ship type) or a miss. A ship sinks when all its cells are hit. The agent must sink all ships within 20 turns. This environment environment requires grid exploration to locate ships and once located, exploitation in the form of targeted attacks to sink them. We use chain-of-thought prompting for the agent and a hardcoded program to generate environment feedback. The tasks are generated by randomly choosing the ship locations at each iteration. Minesweeper: We include minesweeper to test an LLM's sequential logical reasoning ability. The agent interacts with a 2D rectangular grid containing hidden mines. At each turn, the agent reveals one cell. The first move is always safe since mines are placed afterwards. If a mine is revealed, the task ends in failure. To win, the agent must reveal all mine-free cells within 20 turns. When a cell is revealed, it displays a number indicating how many mines are in adjacent cells. If a revealed cell has no adjacent mines (shown as '0'), all neighboring mine-free cells are automatically revealed. We use chain-of-thought prompting for the agent and a hardcoded program to generate environment feedback. The tasks are generated by randomly placing mines in the 2D grid at each generation. Bandit Best Arm Selection: Multi-arm bandits are a classic test for an agent's ability to perform sequential decision making - LLMs have been tested on this task in prior works such as Krishnamurthy et al. (2024); Nie et al. (2024). In this environment, an LLM is presented with a hypothetical scenario where it can select arms at every turn and observe the reward chosen from a Bernoulli distribution with a fixed but unknown mean attached to that arm. We created a modified version of their environment with three key distinctions: 1) prior works operated on bandits in a single-turn fashion: at each turn, LLMs were given the problem setup and history of past interactions within a single user prompt and asked to choose the next arm. Instead, our design employs multi-turn interactions, where the task description is given in the first turn, and later turns only provide rewards for the selected arm. 2) Prior works required the LLM to output only the chosen arm, whereas we employ chain-of-thought (COT) prompting to let the LLM think before it chooses an arm. 3) Instead of minimizing regret over a long time horizon, we instead work on the bandit best arm selection problem, where the LLM gets to choose arms and observe rewards for 20 turns, and then is prompted to choose what it thinks is the arm with the highest mean reward. This is done mainly to control for context length when employing COT, as we could not run inference for more than 20 turns without running into computational issues, and the observed regret between multiple models is too small if horizon length is 20. We randomize the arm rewards at every iteration. For evaluation, we use the same bandit description as Krishnamurthy et al. (2024), for training, we use GPT-4o-mini to generate 81 diverse scenarios that are similar to it but has randomly chosen arm names and hypothetical scenarios. We also note that if the two best arms have very close mean reward (for example, 0.7 and 0.65), then it can be very difficult to identify the best arm within 20 turns. Following Krishnamurthy et al. (2024); Nie et al. (2024), we set the mean reward of the best arm to be above a certain threshold over the mean rewards of the other arms. Finally, all the task instructions for the agents, task environments and LLM-judges were written by GPT-4o-mini, which we report next for the sake of reproducibility.

## B.1.1. Note On Task Prompts

We provide the task information in the first user prompt given to the agent. The system prompt for the agent on all task groups remains the same: "You are a helpful assistant.". Our initial experiments suggested that giving the task instruction in the first user prompt was more fruitful than providing it in the system prompt, though we suggest further investigation of this phenomenon.

## B.2. Note On Text-Based Games

The goal of PAPRIKA is to train an LLM agent to be better at information-seeking, and to test whether these informationseeking behavior learned from a few task groups also generalizes to a new domain. To do so, we design our own task groups that require gather information to succeed. While a lot of the task groups resemble text-based games, our focus is not on them; rather text-based games are simpler information-seeking tasks that can be solved and learned reliably by language models of 8-12B parameter range, and we expect these ideas to extend to much more complicated domains given sufficiently powerful initial models. Text-based games are an active area of research, and we would like to mention some related works here. Hausknecht et al. (2020b) utilizes interactive fiction games as a testbed for studying language based autonomous agents and their ability to handle dynamic action spaces. While our 'Murder Mystery' task group is inspired by Hausknecht et al. (2020b), particularly Detective, we choose to implement it separately instead of using their task environment directly, primarily due to their implementation relying on a manual parser to extract action from the LLM's generation and relying on it to take steps in the environment. The LLMs we experimented with had difficulty outputting responses in the exact format their task environment required, and we found using GPT-4o-mini to simulate the task environment to be easier while also providing more dynamic environment responses. Future work can try to directly incorporate games from Hausknecht et al. (2020b) into PAPRIKA. Similarly, text-based task groups from Cotˆ e et al. ´ (2019); Wang et al. (2022); Jansen et al. (2024) can provide a further set of rich environments to train and test PAPRIKA-based agents on. This is a growing field with many interesting directions, we direct the readers to Jansen (2021) for a comprehensive study.

| Table 3. Summary of the initial state received by the agent, the action, and the observation spaces on all 10 task groups.   |                                                                                                      |                                                                                                                                   |                           |           |
|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|---------------------------|-----------|
| Task Group                                                                                                                   | Initial State                                                                                        | Action                                                                                                                            | Intermediate Observations |           |
| Twenty Questions                                                                                                             | User message describing the task, and the type of the secret topic (e.g., 'a concept' or 'a famous person') that the agent needs to guess within 20 questions                                                                                                      | A yes/no question about the secret topic the agent needs to guess                                                                 | GPT-4o-mini               | generated |
| yes/no answer                                                                                                                |                                                                                                      |                                                                                                                                   |                           |           |
| Guess My City                                                                                                                | User message describing the                                                                          | An                                                                                                                                | open-ended                | question  |
| task                                                                                                                         | about                                                                                                | the secret                                                                                                                        | city                      | the       |
| model needs to guess                                                                                                         | GPT-4o-mini generated answer                                                                                                      |                                                                                                                                   |                           |           |
| Customer Service                                                                                                             | User message describing the task, which includes a description of the problem the customer is facing                                                                                                      | A customer service agent's troubleshooting question that aims to identify the action that will resolve the customer's problem                                                                                                                                   | GPT-4o-mini simulated customer response                           |           |
| Murder Mystery                                                                                                               | User message describing the task and the particular muder mystery scenario, including the victim, supsects and witnesses of the crime scenario                                                                                                      | An action the detective takes in the game, aimed at identifying the perpetrator (for example, asking a suspect where they were)                                                                                                                                   | GPT-4o-mini               | simulated |
| game                                                                                                                         | environment,                                                                                         | e.g,                                                                                                                              |                           |           |
| the suspect answering the detective's questions                                                                              |                                                                                                      |                                                                                                                                   |                           |           |
| Wordle                                                                                                                       | User message describing the Wordle game, nature of intermediate feedback and rules the agent needs to follow                                                                                                      | Step-by-step                                                                                                                      | thinking                  | fol           |
| lowed by a 5 letter word guess                                                                                               | Feedback related to the common letters between the correct word and the guess                                                                                                      |                                                                                                                                   |                           |           |
| Cellular Automata                                                                                                            | User message describing the task, and example inputs and corresponding outputs from which the agent needs to deduce the transition rule                                                                                                      | Step-by-step                                                                                                                      | thinking                  | and       |
| then Deduction about the transition rules in a 1D cellular automaton                                                                                                                              | Inputs, true outputs, outputs generated by the transition rule the agent deduced                     |                                                                                                                                   |                           |           |
| Mastermind                                                                                                                   | User message describing the                                                                          | Step-by-step thinking about                                                                                                       |                           |           |
| rules of the task                                                                                                            | past guesses and observed feedback, followed by a 4 digit secret code guess                          | Feedback related to the common digits between the correct code and the guess made by the agent                                                                                                                                   |                           |           |
| Battleship                                                                                                                   | User message describing the task, and the initial board state of the game                            | Whether the cell contains a ship and what type of ship, followed by the current state of the board given the agent's past actions |                           |           |
| Minesweeper                                                                                                                  | User message describing the task and the initial state of the grid                                   | A particular cell that                                                                                                            | the                       |           |
| agent wants to hit with a missile                                                                                                                              | Whether the cell is mine free, followed by the current state of the grid following rules of the game |                                                                                                                                   |                           |           |
| Bandit Best Arm Selection                                                                                                    | User message describing the bandit game in text, including the number of turns and the possible arm names from which the agent needs to pick arm with the best reward                                                                                                      | A particular cell that                                                                                                            | the                       |           |
| agent wants to declare 'minefree' A particular arm in the multiarmed bandit problem that the agent chooses                                                                                                                              | Observed reward from choosing that arm, sampled from its corresponding Bernoulli distribution with fixed but unknown mean                                                                                                      |                                                                                                                                   |                           |           |

B.3. Comparison of action and observation spaces between the task groups Table 3 shows a summary of how the task groups differ from each other.

## B.4. Details Of Individual Task Groups B.4.1. Twenty Questions

For twenty questions, we provide the LLM agent with general instructions about the task, and the type of hidden topic (e.g., person, location, food etc.) that it needs to guess in the first user prompt. An example is given below.

## Twenty Questions Agent Prompt

You are playing a game of 20 Questions. Your goal is to guess the name of a thing or person by asking up to 20 yes-or-no questions. After each question, you will receive an answer: 'Yes' or 'No.' Use the answers provided to refine your guesses. Here are your instructions:
- You can ask only yes-or-no questions. - After receiving each answer, you should adapt your questions based on the new information. - Your goal is to guess the topic in as few questions as possible. - If you're confident, you can make a guess before reaching 20 questions.

The game starts now. You are trying to guess a clothing. Ask your first question!

We use another LLM (usually GPT-4o-mini unless explicitly mentioned otherwise) to simulate the task environment that provides yes/no answers. This LLM receives the secret topic, and generates answers to the agent's questions in relation to this topic. An example system prompt for the task environment is given below.

## Twenty Questions Environment Prompt

You are the environment for a game of 20 Questions. You will be given a topic (e.g., cat, deer, Abraham Lincoln),
and your role is to answer 'Yes' or 'No' to questions about the topic. Respond strictly with 'Yes' or 'No' only, based on the truth of the matter. Do not provide any additional information. If the user guesses the correct answer, respond with 'Goal reached.' Enforce a strict match before saying 'Goal reached.', for example, if the user guesses
'Is the occupation Engineer?' and the topic was software engineering, say 'No'. However, if the user guesses 'Is the occupation a type of engineering?', say 'Yes'. Wait for the user's questions and respond accordingly. For example:
- User: 'Is this person alive?' - You: 'No' - User: 'Is this a living creature?' - You: 'Yes' The player is trying to guess a clothing, and the correct answer is Gloves.

To prevent task environment hacking, we additionally perform string matching to ensure the response from the task environment is yes, no or 'Goal reached'. We discard trajectories where the task environment responds with something else everytime within 5 attempts. Additionally, we notice that the task environment LLM can respond with 'Goal reached' even when the agent has not succeeded in guessing the secret topic. We use an additional API call to GPT-4o-mini as a judge to filter these trajectories. An example system prompt for this LLM judge is as follows:

## Twenty Questions Judge Prompt

| need to judge if it successfully guesses the correct topic. In particular: Check that the player has precisely guessed Gloves in their question. Reply with <VALID> or <NOTVALID>. Reply with <VALID> if and only if the given question successfully ends the game by guessing the correct clothing, which is: Gloves, otherwise reply with <NOTVALID>. Example 1: Topic to Guess: Car Player: Is the invention a car? Answer: <VALID> Example 2: Topic to Guess: Car Player: Does the invention have wheels? Answer: <NOTVALID> Example 3: Topic to Guess: Apple, Inc. Player: Does this company produce IPhones? Answer: <NOTVALID> The conversation begins here: Agent: Based on the fact that the clothing is worn for warmth and on the hands, I'm going to guess that the clothing is a glove. (End of Agent Turn) Now judge whether the player has successfully guessed the correct clothing, which is Gloves in this particular game.   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|