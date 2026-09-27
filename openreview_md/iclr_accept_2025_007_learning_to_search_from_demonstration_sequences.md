# Learning To Search From Demonstration Sequences

Dixant Mittal1,2 Liwei Kang1 **Wee Sun Lee**1 1National University of Singapore 2 Moovita
{dixant,kang,leews}@comp.nus.edu.sg

## Abstract

Search and planning are essential for solving many real-world problems. However, in numerous learning scenarios, only action-observation sequences, such as demonstrations or instruction sequences, are available for learning. Relying solely on supervised learning with these sequences can lead to sub-optimal performance due to the vast, unseen search space encountered during training. In this paper, we introduce Differentiable Tree Search Network (D-TSN), a novel neural network architecture that learns to construct search trees from just sequences of demonstrations by performing gradient descent on a best-first search tree construction algorithm. D-TSN enables the joint learning of submodules, including an encoder, value function, and world model, which are essential for planning. To construct the search tree, we employ a stochastic tree expansion policy and formulate it as another decision-making task. Then, we optimize the tree expansion policy via REINFORCE with an effective variance reduction technique for the gradient computation. D-TSN can be applied to problems with a known world model or to scenarios where it needs to jointly learn a world model with a latent state space. We study problems from these two scenarios, including Game of 24, 2D grid navigation, and Procgen games, to understand when D-TSN is more helpful. Through our experiments, we show that D-TSN is effective, especially when the world model with a latent state space is jointly learned. The code is available at https://github. com/dixantmittal/differentiable-tree-search-network.

## 1 Introduction

Search and planning are critical components in a wide range of complex tasks. It has long been important in areas such as robotics and games (Mohanan & Salgoankar, 2018; Duarte et al., 2020), and has recently gained prominence in the area of large language models as inference-time solutions for improving reasoning and agentic abilities (Yao et al., 2023; Hao et al., 2023; Zhou et al., 2024). In many of these domains, step-by-step demonstrations are available in the form of recorded expert demonstrations, medical treatment records, and instruction/solution manuals. However, constructing a search tree would require data that is not directly available from the demonstration sequences - this can be impractical or very expensive to obtain. Hence, it would be desirable to learn to search from only demonstration sequences. Search and planning can be effectively executed if a simulator or a world model is available. When such a world model is not available, we may learn the world model and then use it for these purposes (Kaiser et al., 2020; Hafner et al., 2020; Ha & Schmidhuber, 2018). However, when separately learned from demonstration sequences, such world models often suffer from compounding errors, and cannot cover the complete state space. Using such world models for search can be unreliable, particularly if the search expands into parts of the space that are not seen during training (Asadi et al.,
2018).

One way to increase the reliability of the world model is to perform end-to-end learning (Farquhar et al., 2018), where the world model is trained with the search algorithm. One of the advantages of end-to-end learning is that the search algorithm will learn to avoid going into states where the world model is inaccurate since this would result in lower performance. We begin by showing that, when the search tree has a fixed structure, the loss function for training is continuous with respect to the 1 model parameters, justifying the use of a fixed-structure search tree with gradient-based learning methods. However, if the search tree structure can change during training, the loss function may no longer be continuous. This is because a small change in parameters could result in the construction of a different search tree and bring a large change in the corresponding loss function. To alleviate this issue, we propose employing a stochastic tree expansion policy and optimizing the expected loss, ensuring the continuity of loss function in the parameter space. We further propose formulating the search tree expansion as another decision-making task with the goal of minimizing the prediction error progressively during the tree expansion and refining the tree expansion policy via REINFORCE (Williams, 1992). The use of the REINFORCE algorithm introduces another challenge; REINFORCE usually has high variance in its gradient estimates. To handle that, we propose the use of a baseline for variance reduction, adopting the telescoping sum trick used in Guez et al. (2018) for constructing the baseline. Employing these techniques allows us to train a novel neural network, Differentiable Tree Search Network (D-TSN), that embeds the structure of a best-first search algorithm into its architecture.

D-TSN can use a simulator or world model, if available, to learn to perform search. We study this scenario by fine-tuning a large language model to play the Game of 24 (Yao et al., 2023). D-TSN can also be used in more difficult scenarios where the world model is not available and only sequences of demonstrations are provided. We study two problems in this scenario, a small-scale 2D grid navigation problem and Procgen games (Cobbe et al., 2020) which are substantially more complex. Through our experiments, we show that end-to-end learning for doing search and planning from only sequences of demonstrations is possible and useful, especially when the world model needs to be jointly learned.

## 2 Related Works

Search and planning are crucial in various domains. For example, in language model reasoning, recent works integrated search algorithms to enhance the reasoning capabilities of language models. These techniques employ in-context queries (Yao et al., 2023; Xie et al., 2023; Hao et al., 2023; Zhou et al., 2024) or additional value heads (Liu et al., 2023) to guide the search process during inference, enhancing model's reasoning ability by looking at different possible solutions. In gaming, notable successes (Silver et al., 2016; Schrittwieser et al., 2020; Nasiriany et al., 2019) have combined deep neural networks with search algorithms to achieve superhuman performance. One line of recent works has focused on embedding search inductive bias into network architectures. Neural Admissible Relaxation (NEAR) (Shah et al., 2020) develops approximately admissible heuristics for the A* algorithm. Neural A* Search (Yonetani et al., 2021) integrates the A* algorithm into network structures, learning a cost function from a gridworld map. Similarly, MCTSnets (Guez et al., 2018) incorporate the framework of Monte Carlo Tree Search (MCTS) into network architectures, guiding the search using parameterized memory embeddings stored in a tree structure. However, these methods rely on a known world model for planning, which limits their application when the world model is unknown. Another notable work, TreeQN (Farquhar et al., 2018), incorporates search inductive bias into the network by fully expanding a search tree to a fixed depth while jointly learning a world model, which allows it to tackle problems where the world model is unknown. However, the full expansion mechanism of TreeQN leads to a shallow search tree, which limits its application in complex problems with longer trajectories. Our setting of learning from sequences of demonstrations is closely related to Offline Reinforcement Learning (Prudencio et al., 2024; Levine et al., 2020), where an agent learns its policy solely from a fixed dataset of experiences without further interactions with the environment. The Offline-RL paradigm is appealing in many real-world applications such as education, healthcare, and robotics, where active data collection is often infeasible (Singh et al., 2022; Singla et al., 2021; Liu et al., 2019). In our work, we develop a method that learns to search in an end-to-end fashion solely from sequences of demonstrations, such as text solutions to reasoning problems and expert trajectories on navigation and games.

## 3 Differentiable Tree Search Network

Differentiable Tree Search Network (D-TSN) is a neural network design that incorporates the algorithmic inductive bias of a best-first search algorithm into the network structure. It learns from sequences of demonstrations to construct search trees by composing submodules, that include an encoder, a value function, a reward function, and a transition function. If some submodules are readily available, e.g. a world model or simulator, they can be directly used in D-TSN. When these submodules are not available, they can be jointly optimized with the search algorithm. This joint optimization allows the learned imperfect world model to be useful for online search and makes the submodules robust against errors in the world model. We start by describing the version of D-TSN where all submodules are learned.

## 3.1 Learnable Submodules

D-TSN comprises several learnable submodules that function as subroutines in a best-first search algorithm and dynamically construct a computation graph. An illustration is provided in Figure 4.

The *Encoder* module (Eθ) transforms the actual state st into a latent state ht = Eθ(st), facilitating online search within a latent space. The *Transition* module (Tθ) approximates the environment's transition function, using ht and action at to predict the subsequent latent state, ht+1 = Tθ(ht, at), of the transition. The *Reward* module (Rθ) approximates the environment's reward function, predicting the reward, rt = Rθ(ht, at), for the transition based on ht and at. The *Value* module (Vθ) maps latent state ht to its estimated state value, Vθ(ht). Dividing the network into these submodules reduces total learnable parameters and injects a strong search inductive bias into the network architecture, preventing overfitting to an arbitrary function that may align with the limited available training data. 3.2 TREE SEARCH IN LATENT SPACE
The search begins by encoding the input state s0 into its latent state h0, then proceeds through expansion and backup phases. During expansion, the search tree expands iteratively for a set number of expansion steps, where each step expands a node in the search tree. During backup, Q-values at the root node are recursively computed using the Bellman equation across expanded nodes. Each node represents a latent state reachable from the root, and branches represent actions taken. A candidate set O is maintained during the expansion phase, representing nodes eligible for further expansion. The D-TSN algorithm is detailed in Algorithm 1.

Expansion Phase Each search iteration begins by evaluating the path value, V¯ (N), of the candidate nodes. The path value is defined as the cumulative sum of rewards from the root node to a particular leaf node N, in addition to the value of the leaf node predicted by the value module (Vθ), i.e.

V¯ (N) = Rθ(h0, a0) + ... + Vθ(hN ) (1)
A naive implementation of the search selects the node Nˆ with the highest total path value for expansion; however, for a differentiable search, the node Nˆ is sampled from the candidates using a distribution constructed by applying softmax over the path values of the candidates. Expansion of node Nˆ is carried out by simulating every action on the node using the transition module (Tθ).

Simultaneously, the associated reward, Rθ(hNˆ , a), is also computed. The resulting latent states are added to the tree as children of node Nˆ. Additionally, they are added to the candidate set O for subsequent expansions, while Nˆ is excluded from the set. This can be represented as:

$$\bar{V}(N)={\mathcal{R}}_{\theta}(h_{0},a_{0})+...+{\mathcal{V}}_{\theta}(h_{N})$$
$$(\mathbf{l})$$
$$O\gets O\cup\{h_{a}|\,h_{a}={\mathcal{T}}_{\theta}(h_{\hat{N}},a);\forall a\in A\}-\hat{N}$$
$$(2)$$

Backup Phase The expansion phase is followed by the *Backup phase*. In this phase, values of all tree nodes are recursively updated using the Bellman equation as follows:
Q(*N, a*) = Rθ(hN , a) + V (N
′), where N
′ = Tθ(hN , a) (3)

$$V(N^{\prime})=\begin{cases}\mathcal{V}_{\theta}(h_{N^{\prime}}),&N^{\prime}\text{is a leaf node}\\ \max_{a}Q(N^{\prime},a),&\text{otherwise}\end{cases}$$

After the backup phase, Q-values at the root node are returned as the final output of the online search.

An illustration of the constructing the tree is shown in Figure 1.

3.3 CONSTRUCTION OF COMPUTATION GRAPH Throughout the expansion and backup phases, illustrated in Appendix Figures 5 and 6 respectively, a dynamic computation graph is constructed where the output Q-values depend on the combination of all the submodules, i.e. Encoder, Transition, Reward, and Value modules. During training, the output

$$({\mathfrak{I}})$$
$$(4)$$

Expansion Backup
= ( , )
( , ) = + ( ) ( , ) 
( )= ( )
( ) = ( , )
( ) = + + ( )
Sum of rewards to Value of leaf state 
( , ) ( , ) = ( , )
Sample a leaf node from softmax( ( ), ( ), ( )) to expand Expansion Value of nodes are updated as maximum Q value of children nodes Backup
Q-function is evaluated by a loss function and is optimized using gradient-based optimizers such as Stochastic Gradient Descent (SGD). These optimizers backpropagate the gradient of this loss through the entire computation graph and update the parameters of submodules in an end-to-end process. An illustration of the process is provided in Appendix Figure 7. 3.4 DISCONTINUITY OF THE LOSS FUNCTION A pivotal aspect of optimizing D-TSN's parameters through gradient descent is ensuring that the loss function applied to the output Q-values is *continuous in the network's parameter space*. We initiate this discussion by showing that the loss function, when applied to the Q-values computed by expanding a search tree, is continuous in the parameter space, provided that the search tree has a fixed structure. Theorem 3.1. Given a set of parameterized modules that are continuous in the parameter space θ, the Q-values computed by expanding a search tree with a fixed structure, through the composition of these modules and backpropagation of child values using addition and max operations, are continuous in the parameter space θ. When the tree structure is not fixed, the continuity of the Q-values is not guaranteed. (For a detailed proof, refer to Appendix B.) Theorem 3.1 establishes that the Q-values calculated by a fixed-structure tree expansion are continuous in the network's parameter space. The theorem applies to TreeQN (Farquhar et al., 2018), which performs a full tree expansion to a fixed depth, and consequently, the loss function applied to its output Q-values is also continuous, contributing to TreeQN's success in gradient-based optimization. However, in the case of naive implementation of D-TSN, as outlined in Section 3.2, the search tree is constructed by expanding only those paths that are likely to represent the optimal trajectory from the root node. This results in the output Q-values and the corresponding loss function being dependent on the specific tree, τ , constructed during the search:
L(*s, a*) = L(Qθ(*s, a*|τ )) (5)
When the network parameters are changed slightly, this naive implementation could generate a different tree structure, which, in turn, would induce a large change in the loss function and affect its continuity in the parameter space. 3.5 ENABLING CONTINUITY IN THE LOSS FUNCTION To overcome the discontinuity issue observed in the naive implementation of D-TSN, we employ a stochastic tree expansion policy. This approach allows us to optimize the expectation of the loss function, defined as:

$$L(s,a)={\mathcal{L}}(Q_{\theta}(s,a|\tau))$$
$$L(s,a)=\mathbb{E}_{\tau}\Big[\mathcal{L}(Q_{\theta}(s,a|\tau))\Big]=\sum_{\tau}\pi_{\theta}(\tau)\mathcal{L}(Q_{\theta}(s,a|\tau))$$

The expected loss in Equation (6) is continuous in the parameter space θ and can be optimized using gradient-based optimization techniques. 3.6 STOCHASTIC TREE EXPANSION POLICY In order to compute the expected loss in Equation (6), let us represent a partial search tree after t iterations as τt. The output Q-values, denoted as Qθ(*s, a*|τ ), depend on the final tree τ sampled after

$$(6)$$

T iterations. We can define a stochastic tree expansion policy πθ(τt) that takes a tree τt as input and outputs a distribution over the candidate nodes, facilitating stochastic selection of the node for further expansion and generating the tree τt+1. We compute the stochastic tree expansion policy by taking softmax over the path value, as defined in Equation (1), of each candidate node as follows:

$$\pi_{\theta}(n|\tau_{t})={\frac{\exp({\bar{V}}(n))}{\sum_{n^{\prime}\in{\cal O}(\tau_{t})}\exp({\bar{V}}(n^{\prime}))}}$$

The gradient of the expected loss (Schulman et al., 2015) in Equation (6) can be computed as follows (See Appendix C for the derivation):

$$\nabla_{\theta}L(s,a)=\mathbb{E}_{\tau}\Big{[}\mathcal{L}(Q_{\theta}(s,a|\tau))\sum_{t=1}^{T}\nabla_{\theta}\log\pi_{\theta}(n_{t}|\tau_{t})+\nabla_{\theta}\mathcal{L}(Q_{\theta}(s,a|\tau))\Big{]}\tag{8}$$
$$\left(7\right)$$

3.7 REDUCING VARIANCE USING TELESCOPIC SUM The REINFORCE term of the gradient in Equation (8) usually has high variance due to the difficulty of credit assignment (Pignatelli et al., 2024) in a reinforcement learning type objective; the second part of the gradient equation is the usual optimization of a loss function, so we expect it to be reasonably well behaved. To reduce the variance of the first part of the gradient, we take inspiration from the telescoping sum trick used in Guez et al. (2018).

Let us denote the loss after t iterations as Lt = L(Qθ(*s, a*|τt)). The objective is to minimize the loss after T iterations, represented as LT . Assuming that L0 = 0, we can rewrite LT as a telescoping sum:

$$L_{T}=L_{T}-L_{0}=\sum_{t=1}^{T}L_{t}-L_{t-1}\tag{1}$$
$$({\mathfrak{g}})$$
$$(10)$$

Now, we define a reward term, rt, for selecting node n during the t th iteration as the reduction in the loss value after the t th iteration, i.e. rt = Lt − Lt−1. Further, let us represent the return (or rewards-to-go) from iteration t to the final iteration T as Rt, which can be computed as:

$$R_{t}=\sum_{i=t}^{T}r_{i}=L_{T}-L_{t-1}\tag{1}$$
$$(11)$$

Given this, the REINFORCE term from Equation (8) can be reformulated as PT
t ∇θ log πθ(nt|τt)Rt, where LT serves as a baseline that helps in reducing variance. Consequently, the final gradient estimate of the loss in Equation (6) is expressed as:

$$\nabla_{\theta}L(s,a)=\mathbb{E}_{\tau}\Big[\sum_{t}^{T}\nabla_{\theta}\log\pi_{\theta}(n_{t}|\tau_{t})R_{t}+\nabla_{\theta}\mathcal{L}(Q_{\theta}(s,a|\tau))\Big]$$

For empirical evaluations, we use a single sample estimate Schulman et al. (2015) of the expected gradient in Equation (11).

3.8 LOSS FUNCTIONS
In this section, we define a series of loss functions that are used to train D-TSN. Given a dataset of trajectories where each trajectory is denoted as {(si, ai, ri, Qi)}
T
i=0, the primary objective is to make the computed Q-values closely approximate the observed Q-values for corresponding states and actions. To achieve this, we minimize the mean squared error between the predicted and observed Q-values. This loss, denoted as LQ, is defined as:

$\mathcal{L}_{Q}=\mathbb{E}_{(s,a,Q)\sim\mathcal{D}}(Q_{\theta}(s,\ a)-Q)^{2}$
To avoid Q-values for out-of-distribution actions getting overestimated, we incorporate the CQL (Kumar et al., 2020) loss, which encourages the agent to adhere to actions observed within the training data distribution. This loss, LD, is defined as:

$${\mathcal{L}}_{\mathcal{D}}=\mathbb{E}_{(s,a)\sim{\mathcal{D}}}{\Big(}\log\sum_{a^{\prime}}\exp(Q_{\theta}(s,\,a^{\prime}))-Q_{\theta}(s,\,a){\Big)}$$
$$(12)$$
$$(13)$$

Additionally, we incorporate self-supervised consistency loss functions (Schwarzer et al., 2021; Ye
et al., 2021) to ensure consistency in the transition and reward networks. Consider actual states si and si+1, where si+1 is obtained by taking action aiin state si. Their corresponding latent state representations are denoted as hi and hi+1. Here, hi = Eθ(si) and hi+1 = Eθ(si+1). Now, we can use the transition module to predict another latent representation of state si+1, represented as
h¯i+1 = Tθ(hi, ai). To ensure that the transition function, Tθ, provides consistent predictions for the
transitions in the latent space, we minimize the squared error between the latent representations hi+1
and $h_{i+1}$.  $$\mathcal{L}_{\mathcal{T}_{\theta}}=\mathbb{E}_{(s_{i},\ a_{i},\ s_{i+1})\sim\mathcal{D}}(\bar{h}_{i+1}-h_{i+1})^{2}\tag{14}$$  In a similar vein, we minimize the mean squared error between the predicted reward $\mathcal{R}_{\theta}(h_{i},a_{i})$ and 
the actual reward observed riin the training dataset D.
2(15)
$\mathcal{L}_{\mathcal{R}_{\theta}}=\mathbb{E}_{(s_{i},\,a_{i},\,r_{i})\sim\mathcal{D}}(\mathcal{R}_{\theta}(h_{i},a_{i})-r_{i})^{2}$  (13), (14) and (15), we define the loss function 
Combining Equations (12), (13), (14) and (15), we define the loss function for training D-TSN as

$$(14)$$
$$(15)$$
$${\mathcal{L}}_{\mathrm{D-TSN}}=\mathbb{E}_{\tau}\left[\lambda_{1}{\mathcal{L}}_{Q}+\lambda_{2}{\mathcal{L}}_{{\mathcal{D}}}\right]+\lambda_{3}{\mathcal{L}}_{{\mathcal{T}}_{\theta}}+\lambda_{4}{\mathcal{L}}_{{\mathcal{R}}_{\theta}}$$
$$(16)$$

where τ denotes the search tree. Additional details on these loss functions are presented in Appendix D.4.

## 4 Experiments 4.1 D-Tsn With Known World Model

We take an initial step towards learning to perform tree search from instruction sequences for LLMs by first demonstrating that LLMs can be effectively fine-tuned when the world model is known. We study Game of 24 (Yao et al., 2023), a reasoning problem where the transition dynamics and rules are well-defined and readily available. The task is that given four cards drawn from a deck of poker, use basic arithmetic operations to combine the card values to reach 24. The problem can be formulated as follows: a state is a list of current numbers, initially 4; the actions involve selecting an operation as well as two numbers from the current numbers; the transition will calculate the operation and update the list of current numbers by removing the selected two and adding the calculated result; the reward is one when 24 is the only number in the list, and zero otherwise. Since language models can directly process token sequences, we do not need an Encoder in this problem. We use a large language model, Llama3-8B (Dubey et al., 2024), with an additional value head as the value function Vθ.

4.1.1 BASELINE We compare this version of D-TSN with a supervised fine-tuned (SFT) language model. For a fair comparison, the supervised fine-tuned model is also allowed to perform search during inference, where we use the log probability of the model generating the sequence as its value:

$$\mathcal{V}_{\theta}(s)=\frac{1}{N}\sum_{i=1}^{N-1}\log p(s_{i+1}|s_{1\dots i})\tag{1}$$
$$(17)$$

where si are tokens in the sequence, p(·|·) is the conditional probability of generating the next token. The log probability is divided by N to normalize the effect of sequence length. 4.1.2 TRAINING DATASET We collected all valid Game of 24 problems and their solutions through an exhaustive search of all combinations, then randomly selected 530 problems for evaluation. The remaining 527 problems have 16k valid solutions, from which we randomly sampled a subset for training. We format the text solutions as following format: "[2,3,7,7]->(2*7=14)->[3,7,14]->(14+7=21)->[3,21]->(3+21=24)-
>[24]", where *[...]* represents current numbers, and *(...)* represents operations. The operations combine two numbers with an operator, resulting in a new number. We train D-TSN to select operands and operators, and use the transition function to handle the computation of the current numbers after the operation, i.e. given a state *"[2,3,7,7]->(2*"* and an action to select "7" as the next operand, the next state given by the transition function is *"[2,3,7,7]->(2*7=14)->[3,7,14]->("*. This would allow us to focus more on the difficult part of the solution, which is deciding which numbers to use in the operation and which operator to use. Such transition can be easily obtained in practice as we observe that the SFT model never makes mistakes on these positions in its generation.

## 4.1.3 Results

We train D-TSN using 8 search iterations in training and compare the resulting value function with a supervised fine-tuned model, as described in Equation (17), to guide the search during evaluation. Table 1: Comparison between D-TSN and SFT on Game of 24, with varying amount of training trajectories, using metrics *Success Rate* - whether the generated expression is valid and results in 24. D-TSN is trained and evaluated with 8 search iterations, SFT is evaluated with 8 search iterations.

| #Training Trajectories   |       |       |       |
|--------------------------|-------|-------|-------|
| Method                   | 1k    | 2k    | 10k   |
| D-TSN (n_itr=8)          | 31.46 | 38.25 | 45.24 |
| SFT+Search               | 12.45 | 15.09 | 21.32 |

To compare D-TSN with a supervised fine-tuned model under different amounts of available data, we sample 1k, 2k, and 10k trajectories to train both methods. Table 1 shows that D-TSN can better guide the search during evaluation when compared to the SFT model. On a closer look, we notice that the log-likelihood of the SFT model as a value function often fails to recognize promising states, assigning a lower value to them, even when the states have already reached 24. As a result, tokens chosen at each step often do not lead to a correct solution. In contrast, D-TSN better recognizes these states and assigns high values to them, guiding each step to select tokens that lead to a correct solution.

4.2 D-TSN WITH JOINTLY LEARNED WORLD MODEL
Next, we study two problems where we jointly learn a world model in D-TSN. The states in these problems are represented as pixels of images.

Navigation This is a 2D grid-based navigation task designed to quantitatively and qualitatively visualize the agent's generalization capabilities. The environment is a 20 × 20 grid featuring a central hall. At the start of each episode, a robot is randomly positioned inside this hall, while its destination is set outside. We present two distinct scenarios: one with a single exit and another with two exits from the central hall. Training is done only in the two exit scenarios. The single exit scenario, which requires a longer-horizon planning to reach the goal, is used to test generalization. Procgen Procgen (Cobbe et al., 2019) is a collection of 16 procedurally generated, game-like environments, specifically designed to evaluate an agent's generalization capability, differentiating it from Atari 2600 games (Mnih et al., 2013). Further details on these domains are presented in Appendix D.1. We follow the Offline-RL paradigm and train D-TSN using sequences of demonstrations. Please refer to Appendix D.2 for a detailed explanation of the learning framework.

## 4.2.1 Training Datasets

We use a behavior policy, which can be *optimal or sub-optimal*, to collect demonstration sequences for training. An optimal policy generates a dataset with lower noise and a cleaner training signal, leading to a stable learning process. In contrast, a sub-optimal policy produces a noisier dataset, which consequently restricts the quality of the policy that the agent can learn. The choice of the behavior policy depends on domain-specific requirements and the resources available for data collection. We use an optimal behavior policy for Navigation and a sub-optimal policy for Procgen.

The training dataset, D, consists of trajectories generated using the behavior policy πB, where each trajectory is defined as a series of T tuples, each comprising the state observed, action taken, reward observed, and the corresponding Q-value of the observed state, denoted as {(si, ai, ri, Qi)}
T
i=0.

The Q-value for state si can be computed by adding the rewards obtained in the trajectory from timestep i onwards, i.e. Qi =PT
j=i rj . We limit the number of trajectories to 1000 for each domain to evaluate the sample complexity and generalization capabilities of each method.

## 4.2.2 Baselines

We benchmark D-TSN against the following prominent baselines: Model-free Q-network This allows us to assess the significance of incorporating inductive biases into the neural network architecture. This model is trained using the loss defined as:
LQ-net = λ1LQ + λ2LD (18)
Model-based Search In this baseline, we utilize the submodules defined for D-TSN, but the world models and the value module are trained independently of each other. During evaluation, we employ the best-first search, akin to D-TSN, utilizing the independently trained modules. Through this baseline, we assess the benefits derived from the joint optimization of the world model and the search algorithm. For our evaluations, we perform 10 search iterations for each input state. To train this model, we compute the Q-value, Qθ, *without performing the search* and optimize the loss defined as:
LSearch = λ1LQ + λ2LD + λ3LTθ + λ4LRθ(19)
TreeQN This comparison helps highlight the advantages of an advanced search algorithm, used in D-TSN, that can execute a deeper search while maintaining similar computational constraints. For evaluations, we adhere to a depth of 2 for TreeQN, as described in Farquhar et al. (2018), for both Procgen and navigation domains. Notably, the memory footprint would increase exponentially for larger depth, which prohibits us to use a larger depth. This model is trained using the loss (as discussed in Farquhar et al. (2018)) defined as:
LTreeQN = λ1LQ + λ2LD + λ3LRθ(20)
Every method is trained with the same datasets using their respective loss functions. A more comprehensive discussion of the baselines and implementation details can be found in Appendix D.6. Table 2: Comparison of D-TSN with the baselines on Navigation using metrics *Success Rate* and Collision Rate.

Table 3: Comparison of D-TSN with the baselines on Procgen using metrics *Mean Scores* and *Mean* Z-score. Standard error intervals are reported, based on 1000 evaluation runs per game.

Games Model-free Model-based TreeQN **D-TSN** 

Q-network Search

bigfish 21.67±0.53 16.36±0.50 20.38±0.52 21.76±**0.53**

bossfight 8.83±**0.18** 7.33±0.19 8.35±0.19 8.50±0.19

caveflyer 2.05±0.12 3.52±0.15 3.57±0.15 3.71±**0.15** chaser 5.77±0.16 5.31±0.15 6.37±0.17 6.77±**0.17** climber 2.85±0.15 4.93±0.17 4.13±0.16 5.62±**0.18**

coinrun 5.18±0.16 6.33±0.15 5.01±0.16 6.40±**0.15**

dodgeball 1.05±0.06 4.26±0.17 4.70±0.18 5.30±**0.19** fruitbot 14.09±0.37 10.76±0.36 15.43±**0.35** 14.00±0.37

heist 0.87±0.09 2.24±0.13 1.91±0.12 2.43±**0.14**

jumper 3.21±0.15 3.97±0.15 3.74±0.15 4.65±**0.16** leaper 7.78±0.13 6.36±0.15 7.85±0.13 8.20±**0.12**

maze 2.02±0.13 2.26±0.13 2.50±0.14 3.30±**0.15**

miner 1.41±0.02 1.35±0.05 2.06±**0.06** 1.71±0.06 ninja 5.07±0.16 5.07±0.16 5.05±0.16 6.61±**0.15** plunder 13.77±**0.36** 9.74±0.32 12.47±0.35 12.75±0.35

starpilot 15.74±0.39 14.83±0.38 17.91±**0.42** 16.42±0.41

Mean Z-Score - 0.10 0.27 **0.35**

4.2.3 RESULTS

Navigation We compare D-TSN against the baselines using success rate and collision rate as

evaluation metrics, where success rate refers to the fraction of test levels completed by the agent, and collision rate refers to the fraction of levels failed due to collision with a wall. Results are shown in

Table 2.

We observe that D-TSN outperforms the baselines on both navigation scenarios. Notably, when agents are trained on data from the 2-exit scenarios but are tested in the 1-exit scenario, D-TSN, with its powerful inductive bias, retains its performance. In stark contrast, the Model-free Q-network and TreeQN experience a substantial performance decline, which underscores their limited generalization ability. Model-based Search also registers a minor decrease in the success rate, reinforcing the importance of jointly optimizing the world model for enhanced robustness.

| Solver               | Success        | Absolute   |                       |            |             |        |       |
|----------------------|----------------|------------|-----------------------|------------|-------------|--------|-------|
| Rate                 | Collision Rate |            |                       |            |             |        |       |
| Navigation (2 exits) |                |            |                       |            |             |        |       |
| Model-free Q-network | 94.5% (± 0.2%) | 4.4%       |                       |            |             |        |       |
| Model-based Search   | 93.2% (± 0.3%) | 6.7%       |                       |            |             |        |       |
| TreeQN               | 95.4% (± 0.2%) | 3.8%       |                       |            |             |        |       |
| D-TSN                | 99.0% (± 0.1%) | 0.7%       |                       |            |             |        |       |
| Navigation (1 exit)  |                |            |                       |            |             |        |       |
| Model-free Q-network | 47.1% (± 0.5%) | 50.2%      |                       |            |             |        |       |
| Model-based Search   | 86.9% (± 0.3%) | 12.4%      |                       |            |             |        |       |
| TreeQN               | 51.8% (± 0.5%) | 39.2%      |                       |            |             |        |       |
| D-TSN                | 99.3% (± 0.1%) | 0.2%       | Games                 | Model-free | Model-based | TreeQN | D-TSN |
| Q-network            | Search         |            |                       |            |             |        |       |
| bigfish              | 21.67±0.53     | 16.36±0.50 | 20.38±0.52 21.76±0.53 |            |             |        |       |
| bossfight            | 8.83±0.18      | 7.33±0.19  | 8.35±0.19             | 8.50±0.19  |             |        |       |
| caveflyer            | 2.05±0.12      | 3.52±0.15  | 3.57±0.15             | 3.71±0.15  |             |        |       |
| chaser               | 5.77±0.16      | 5.31±0.15  | 6.37±0.17             | 6.77±0.17  |             |        |       |
| climber              | 2.85±0.15      | 4.93±0.17  | 4.13±0.16             | 5.62±0.18  |             |        |       |
| coinrun              | 5.18±0.16      | 6.33±0.15  | 5.01±0.16             | 6.40±0.15  |             |        |       |
| dodgeball            | 1.05±0.06      | 4.26±0.17  | 4.70±0.18             | 5.30±0.19  |             |        |       |
| fruitbot             | 14.09±0.37     | 10.76±0.36 | 15.43±0.35 14.00±0.37 |            |             |        |       |
| heist                | 0.87±0.09      | 2.24±0.13  | 1.91±0.12             | 2.43±0.14  |             |        |       |
| jumper               | 3.21±0.15      | 3.97±0.15  | 3.74±0.15             | 4.65±0.16  |             |        |       |
| leaper               | 7.78±0.13      | 6.36±0.15  | 7.85±0.13             | 8.20±0.12  |             |        |       |
| maze                 | 2.02±0.13      | 2.26±0.13  | 2.50±0.14             | 3.30±0.15  |             |        |       |
| miner                | 1.41±0.02      | 1.35±0.05  | 2.06±0.06             | 1.71±0.06  |             |        |       |
| ninja                | 5.07±0.16      | 5.07±0.16  | 5.05±0.16             | 6.61±0.15  |             |        |       |
| plunder              | 13.77±0.36     | 9.74±0.32  | 12.47±0.35 12.75±0.35 |            |             |        |       |
| starpilot            | 15.74±0.39     | 14.83±0.38 | 17.91±0.42 16.42±0.41 |            |             |        |       |
| Mean Z-Score         | -              | 0.10       | 0.27                  | 0.35       |             |        |       |

Procgen We evaluate the performance of D-TSN and the baselines on all Procgen suite games, comparing mean Z-score and head-to-head wins. We also report the mean scores across 1000 episodes obtained by these methods in Table 3. As Procgen games have different scales, we use model-free Q-network as baseline to compute a normalized score1, Z-score = (µπ −µB)/σB, where µπ and µB
1We use Z-score because the baseline normalized score used in prior works Badia et al. (2020); Kaiser et al.

(2020); Mittal et al. (2023) is problematic in our experiments, as described in Appendix D.7.

represent the mean scores obtained by the agent policy and the baseline policy respectively and σB
represents the standard deviation of the scores obtained by the baseline policy. It is important to note that the training data for Procgen was generated using a sub-optimal behavior policy, leading to noisy training signals in comparison to the relatively noise-free data in the Navigation domain. Despite this, D-TSN reports a higher mean Z-score, averaged across the 16 Procgen games, particularly in climber, coinrun, jumper, and ninja, which require long-term planning. This underscores the stronger inductive bias of D-TSN. In Procgen, Model-based Search lags behind both TreeQN and D-TSN, highlighting that for complex environments, joint optimization enables learning a robust world model. In the head-to-head comparison listed in Table 4, D-TSN won in 13 games against the Model-free Q-network, 16 games against Model-based Search, and 13 games against TreeQN.

Table 5: Comparison of D-TSN with its variants to evaluate the role of Auxiliary losses, REIN-
FORCE term and *Telescoping Sum*.

| Navigation                | Procgen             |              |              |
|---------------------------|---------------------|--------------|--------------|
| Solver                    | (1-exit)            |              |              |
| Baseline                  | Tree Search Network |              |              |
| (Games won / Total games) |                     |              |              |
| Model-free Q-network      | 13 / 16             |              |              |
| Model-based Search        | 16 / 16             |              |              |
| TreeQN                    | 13 / 16             | Success Rate | Mean Z-score |
| D-TSN                     | 99.3% (± 0.1%)      | 0.31         |              |
| w/o Telescoping Sum       | 98.5% (± 0.1%)      | 0.28         |              |
| w/o REINFORCE term        | 97.7% (± 0.2%)      | 0.29         |              |
| w/o Auxiliary losses      | 91.1% (± 0.3%)      | -            |              |

## 4.3 Ablation Studies 4.3.1 The Impact Of The Reinforce Term And The Telescoping Sum Trick

In this study, we assess the impact of both the REINFORCE term and the Telescoping Sum trick on D-TSN's performance. The results, presented in Table 5, show notable differences. Without the telescoping sum, D-TSN sees a modest decrease in the success rate for Navigation (1-exit), moving from 99.3% to 98.5%. Similarly, the mean Z-score for Procgen dips to 0.28. The omission of the REINFORCE term also marks a decline, with the Navigation (1-exit) success rate landing at 97.7% and the mean Z-score for Procgen dipping to 0.29.

## 4.3.2 The Contribution Of Auxiliary Losses

In this study, we explore the contribution of auxiliary losses to D-TSN's performance. As outlined in Table 5, the absence of these auxiliary losses leads to a more pronounced decline in the performance. Specifically, the success rate in the Navigation (1-exit) task drops significantly to 91.1% compared to 99.3% with auxiliary losses.

## 4.3.3 Enhancing Performance Through Deeper Search

In this study, we explore the benefits of deeper search in the D-TSN framework by increasing search iterations in Navigation and Game of 24 problems. For Navigation, we maintain an identical number of iterations in the training and evaluation phases to prevent any distribution shifts in the world model. For Game of 24, we can use a larger search iteration during evaluation, as the world model remains fixed. Table 6: Comparison of D-TSN trained with different number of *search iterations* ('n_itr') to evaluate the performance gain by performing deeper searches.

Table 7: Comparison of D-TSN using different number of search iterations during evaluation. All methods are trained with 2k trajectories.

| ing deeper searches.   | #Search iterations in evaluation   |            |       |       |       |
|------------------------|------------------------------------|------------|-------|-------|-------|
| 1                      | 4                                  | 8          | 16    | 128   |       |
| D-TSN (n_itr=1)        | 20.75                              | 31.51      | 38.16 | 49.76 | 89.29 |
| D-TSN (n_itr=4)        | 32.22                              | 41.13      | 49.67 | 90.24 |       |
| D-TSN (n_itr=8)        | 38.25                              | 49.91      | 91.42 |       |       |
| D-TSN (n_itr=16)       | 49.58                              | 92.38      |       |       |       |
| Solver                 | Navigation                         | Navigation |       |       |       |
| (2-exits)              | (1-exit)                           |            |       |       |       |
| D-TSN (n_itr=5)        | 97.4%                              | 96.6%      |       |       |       |
| D-TSN (n_itr=10)       | 99.0%                              | 99.3%      |       |       |       |
| D-TSN (n_itr=20)       | 99.5%                              | 99.1%      |       |       |       |

The results, as listed in Table 6 and Table 7 indicate that a higher number of iterations leads to an improvement in the success rate in both Navigation and Game of 24. For Navigation, where the world model is not known, the improvement suggests that D-TSN is able to exploit the jointly learned world model effectively, and can potentially be scaled up further given additional computational resources. For Game of 24, we see performance consistently improve as the number of search iterations during evaluation increases, highlighting the importance of search. Notably, the performance reaches around 90% for D-TSN when there are sufficient search iterations during evaluation, suggesting that the value function learned by D-TSN can recognize the promising states and guide the generation to those states. We also observe that the number of search iterations performed during training does not affect the performance much in Game of 24. One explanation is that we are using a ground-truth world model, and performing a search using actual states. The REINFORCE term in Equation (11) would act similarly to the CQL loss in Equation (13). Learning a world model jointly in D-TSN is also possible in principle in language model tasks, but scaling considerations become important to manage the large action space and observation space. We leave this exploration for future works.

## 4.3.4 Robustness Of The World Model

A key challenge in employing learned world models for search is addressing the compounding errors, which impact the accuracy and effectiveness of search. This study shows that the joint optimization of both the world model and the search algorithm compensates for these inaccuracies, ensuring the world model is usable in deeper online searches. Table 8: Comparison of D-TSN and Model-based Search to highlight the robustness of the world model when performing deeper searches ('n_itr' refers to number of search iterations).

| Solver                        | Navigation   | Navigation   |
|-------------------------------|--------------|--------------|
| (2-exits)                     | (1-exit)     |              |
| D-TSN (n_itr=10)              | 99.0%        | 99.3%        |
| D-TSN (n_itr=20)              | 99.3%        | 99.4%        |
| D-TSN (n_itr=50)              | 99.7%        | 99.6%        |
| Model-based Search (n_itr=10) | 93.2%        | 86.9%        |
| Model-based Search (n_itr=20) | 91.1%        | 84.6%        |
| Model-based Search (n_itr=50) | 89.5%        | 80.4%        |

As demonstrated in Table 8, the world model trained jointly with the search algorithm consistently outperforms the independently trained model, especially as the number of iterations increases.

## 5 Conclusion

In this paper, we introduce Differentiable Tree Search Network (D-TSN), a novel framework that learns to conduct differentiable tree search from just sequences of demonstrations. D-TSN can leverage a known world model, or jointly learn a world model end-to-end with other search submodules. D-TSN conducts a best-first style search, which allows it to search deeper and explore more promising states. A naive incorporation of best-first search could lead to discontinuity of the loss in the parameter space, an issue we address by employing a stochastic tree expansion policy. We optimize the expected loss function using a REINFORCE-style objective and propose a telescoping sum trick to reduce the variance of the gradient for this expected loss. We evaluate D-TSN in two scenarios: 1) when the world model is known - we experiment with a language model reasoning task, Game of 24; 2) when the world model is not known - we experiment with 2D grid navigation and Procgen games. Through our experiments, we show that D-TSN is effective, outperforming baselines in both scenarios, especially when the world model is jointly learned. Nonetheless, the strength of the current implementation of D-TSN is currently limited to deterministic decision-making problems with a discrete action space. We have demonstrated the method on latent world models with 2D visual observations but have yet to do so on language observations. To cater to a broader spectrum of decision-making problems, there is also a need to revamp the transition model to manage stochastic world scenarios, which include common usage of language and visual language models. We aim to address these problems in our future works.

## Acknowledgments

This research is supported by the Ministry of Education, Singapore, under its Academic Research Fund Tier 1 (A-8001814-00-00).

## References

Kavosh Asadi, Dipendra Misra, and Michael L. Littman. Lipschitz continuity in model-based reinforcement learning. In Jennifer G. Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of *Proceedings of Machine Learning Research*, pp. 264–273. PMLR, 2018. URL http://proceedings.mlr.press/v80/asadi18a.html.

Adrià Puigdomènech Badia, Bilal Piot, Steven Kapturowski, Pablo Sprechmann, Alex Vitvitskyi, Zhaohan Daniel Guo, and Charles Blundell. Agent57: Outperforming the atari human benchmark. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of *Proceedings of Machine Learning Research*, pp. 507–517.

PMLR, 2020. URL http://proceedings.mlr.press/v119/badia20a.html.

Karl Cobbe, Christopher Hesse, Jacob Hilton, and John Schulman. Leveraging procedural generation to benchmark reinforcement learning. *arXiv preprint arXiv:1912.01588*, 2019.

Karl Cobbe, Christopher Hesse, Jacob Hilton, and John Schulman. Leveraging procedural generation to benchmark reinforcement learning. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of *Proceedings of* Machine Learning Research, pp. 2048–2056. PMLR, 2020. URL http://proceedings. mlr.press/v119/cobbe20a.html.

Karl Cobbe, Jacob Hilton, Oleg Klimov, and John Schulman. Phasic policy gradient. Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, 139:2020–2027, 2021. URL http://proceedings.mlr.press/v139/
cobbe21a.html.

Fernando Fradique Duarte, Nuno Lau, Artur Pereira, and Luis Paulo Reis. A survey of planning and learning in games. *Applied Sciences*, 10(13):4529, 2020.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. The llama 3 herd of models. *CoRR*, abs/2407.21783, 2024. doi: 10.48550/ARXIV.2407.21783. URL https:
//doi.org/10.48550/arXiv.2407.21783.

Gregory Farquhar, Tim Rocktäschel, Maximilian Igl, and Shimon Whiteson. Treeqn and atreec:
Differentiable tree-structured models for deep reinforcement learning. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. URL https://openreview.net/ forum?id=H1dh6Ax0Z.

Arthur Guez, Théophane Weber, Ioannis Antonoglou, Karen Simonyan, Oriol Vinyals, Daan Wierstra, Rémi Munos, and David Silver. Learning to search with mctsnets. In International Conference on Machine Learning, pp. 1822–1831. PMLR, 2018.

David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada, pp. 2455–2467, 2018. URL https://proceedings.neurips.cc/paper/2018/hash/ 2de5d16682c3c35007e4e92982f1a2ba-Abstract.html.

Danijar Hafner, Timothy P. Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control:
Learning behaviors by latent imagination. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum?id=S1lOTC4tDS.

Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. Reasoning with language model is planning with world model. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), *Proceedings of the 2023 Conference on Empirical Methods in* Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pp. 8154–8173. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.EMNLP-MAIN.507. URL https://doi.org/10.18653/v1/2023.emnlp-main.507.

Lukasz Kaiser, Mohammad Babaeizadeh, Piotr Milos, Blazej Osinski, Roy H. Campbell, Konrad Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, Afroz Mohiuddin, Ryan Sepassi, George Tucker, and Henryk Michalewski. Model based reinforcement learning for atari. In *8th International Conference on Learning Representations, ICLR 2020, Addis Ababa,* Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/ forum?id=S1xCPJHtDB.

Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/
0d2b2061826a5df3221116a5085a6052-Abstract.html.

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. *CoRR*, abs/2005.01643, 2020. URL https:// arxiv.org/abs/2005.01643.

Jiacheng Liu, Andrew Cohen, Ramakanth Pasunuru, Yejin Choi, Hannaneh Hajishirzi, and Asli Celikyilmaz. Don't throw away your value model! generating more preferable text with valueguided monte-carlo tree search decoding, 2023.

Siqi Liu, Kee Yuan Ngiam, and Mengling Feng. Deep reinforcement learning for clinical decision support: A brief survey. *CoRR*, abs/1907.09475, 2019. URL http://arxiv.org/abs/ 1907.09475.

Dixant Mittal, Siddharth Aravindan, and Wee Sun Lee. Expose: Combining state-based exploration with gradient-based online search. In Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems, AAMAS '23, pp. 1345–1353, Richland, SC, 2023.

International Foundation for Autonomous Agents and Multiagent Systems. ISBN 9781450394321.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin A. Riedmiller. Playing atari with deep reinforcement learning. *CoRR*,
abs/1312.5602, 2013. URL http://arxiv.org/abs/1312.5602.

Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Maria-Florina Balcan and Kilian Q. Weinberger (eds.), Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016, volume 48 of *JMLR Workshop and Conference Proceedings*, pp. 1928–1937. JMLR.org, 2016. URL http://proceedings.mlr.press/v48/mniha16.html.

M. G. Mohanan and Ambuja Salgoankar. A survey of robotic motion planning in dynamic environments. *Robotics Auton. Syst.*, 100:171–185, 2018. doi: 10.1016/J.ROBOT.2017.10.011. URL https://doi.org/10.1016/j.robot.2017.10.011.

Soroush Nasiriany, Vitchyr Pong, Steven Lin, and Sergey Levine. Planning with goalconditioned policies. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d'Alché-Buc, Emily B. Fox, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pp.

14814–14825, 2019. URL https://proceedings.neurips.cc/paper/2019/hash/
c8cc6e90ccbff44c9cee23611711cdc4-Abstract.html.

Eduardo Pignatelli, Johan Ferret, Matthieu Geist, Thomas Mesnard, Hado van Hasselt, and Laura Toni. A survey of temporal credit assignment in deep reinforcement learning. *Trans. Mach. Learn.* Res., 2024, 2024. URL https://openreview.net/forum?id=bNtr6SLgZf.

Rafael Figueiredo Prudencio, Marcos Ricardo Omena Albuquerque Máximo, and Esther Luna Colombini. A survey on offline reinforcement learning: Taxonomy, review, and open problems. IEEE Trans. Neural Networks Learn. Syst., 35(8):10237–10257, 2024. doi: 10.1109/TNNLS.2023. 3250269. URL https://doi.org/10.1109/TNNLS.2023.3250269.

Walter Rudin. *Principles of Mathematical Analysis*. McGraw-Hill New York, 3d ed. edition, 1976. ISBN 007054235. URL http://www.loc.gov/catdir/toc/mh031/75017903.

html.

Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. *Nature*, 588(7839):604–609, 2020.

John Schulman, Nicolas Heess, Theophane Weber, and Pieter Abbeel. Gradient estimation using stochastic computation graphs. In Corinna Cortes, Neil D. Lawrence, Daniel D. Lee, Masashi Sugiyama, and Roman Garnett (eds.), *Advances in Neural Information Processing Systems 28:*
Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada, pp. 3528–3536, 2015. URL https://proceedings.neurips.cc/ paper/2015/hash/de03beffeed9da5f3639a621bcab5dd4-Abstract.html.

Max Schwarzer, Ankesh Anand, Rishab Goel, R. Devon Hjelm, Aaron C. Courville, and Philip Bachman. Data-efficient reinforcement learning with self-predictive representations. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=uCQfPZwRaUu.

Ameesh Shah, Eric Zhan, Jennifer J. Sun, Abhinav Verma, Yisong Yue, and Swarat Chaudhuri. Learning differentiable programs with admissible neural heuristics. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 342285bb2a8cadef22f667eeb6a63732-Abstract.html.

David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Vedavyas Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy P. Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. Mastering the game of go with deep neural networks and tree search. *Nature*, 529(7587):484–489, 2016. doi:
10.1038/nature16961. URL https://doi.org/10.1038/nature16961.

Bharat Singh, Rajesh Kumar, and Vinay Pratap Singh. Reinforcement learning in robotic applications: a comprehensive survey. *Artif. Intell. Rev.*, 55(2):945–990, 2022. doi: 10.1007/ S10462-021-09997-9. URL https://doi.org/10.1007/s10462-021-09997-9.

Adish Singla, Anna N. Rafferty, Goran Radanovic, and Neil T. Heffernan. Reinforcement learning for education: Opportunities and challenges. *CoRR*, abs/2107.08828, 2021. URL https:
//arxiv.org/abs/2107.08828.

Richard S Sutton and Andrew G Barto. *Reinforcement learning: An introduction*. MIT press, 2018. Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine learning*, 8(3):229–256, 1992.

Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan, Junxian He, and Michael Qizhe Xie. Self-evaluation guided beam search for reasoning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 -
16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/
81fde95c4dc79188a69ce5b24d63010b-Abstract-Conference.html.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. CoRR, abs/2305.10601, 2023. doi: 10.48550/ARXIV.2305.10601. URL https://doi.org/10. 48550/arXiv.2305.10601.

Weirui Ye, Shaohuai Liu, Thanard Kurutach, Pieter Abbeel, and Yang Gao. Mastering atari games with limited data. In Marc'Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan (eds.), Advances in Neural Information Processing Systems 34:
Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pp. 25476–25488, 2021. URL https://proceedings.neurips.cc/ paper/2021/hash/d5eca8dc3820cad9fe56a3bafda65ca1-Abstract.html.

Ryo Yonetani, Tatsunori Taniai, Mohammadamin Barekatain, Mai Nishimura, and Asako Kanezaki.

Path planning using neural a* search. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of *Proceedings of Machine Learning Research*, pp. 12029–12039. PMLR, 2021. URL http://proceedings.mlr.press/v139/yonetani21a.html.

Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning, acting, and planning in language models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024.

OpenReview.net, 2024. URL https://openreview.net/forum?id=njwv9BsGHF.

# A Differentiable Tree Search Network Algorithm

A.1 DIFFERENTIABLE TREE SEARCH NETWORK PSEUDO-CODE Algorithm 1: Differentiable Tree Search (D-TSN)
Input: Input state, s*root* Result: Q-values, Q(sroot, a)
hroot ← Eθ(s*root*) ; //Encode s0 to its latent state h0 noderoot ← *initialise*(hroot) ; //Initialize root node Open ← {node*root*} ; //Initialize candidate set *Open*
// **Expansion phase**
itr ← 0; repeat foreach node ∈ *Open* do hnode ← getLatent(*node*) ; //Get latent state of node V¯ (node) ← sumOfRewards(*node*) + Vθ(h*node*) ; //Compute path values π*tree* ← softmaxn V¯ (n)
; //Compute the tree expansion policy node∗ ← sample(π*tree*) ; //Sample the node to expand foreach a ∈ *Actions*; do hchild ← Tθ(hnode∗ , a) ; //Compute the next latent state rchild ← Rθ(hnode∗ , a) ; //Compute reward for the transition createNode(hchild, r*child*) ; //Create the child node Open ← Open ∪ {childa|childa = getChild(node∗, a); ∀a ∈ A} − *node*∗; //Update the open set itr ← itr + 1; until itr *< MAX_ITR*;
// **Backup phase**
foreach node ∈ T ree*, iterating from leaf nodes to the root node;* do if node *is a leaf;* **then**
hnode ← getLatent(*node*) ; //Get latent state of *node* V (*node*) = Vθ(h*node*) ; //Compute value using Value module else foreach a ∈ *Actions*; do nodechild[a] ← getChild(*node, a*) ; //Get child of *node* that corresponds to action a hnode ← getLatent(*node*) ; //Get latent state of node rnode ← Rθ(hnode, a) ; //Get reward using Reward module Q(node, a) ← rnode + Vnode*child*[a]

V (*node*) ← maxa Q(*node, a*)
return Q(node*root*) ; //Return Q-value of the root node

## A.2 Pseudo-Code Explanation

D-TSN initializes the search by converting the input state to its latent representation and proceeds in two phases: Expansion and Backup as detailed in the following sections. A.2.1 INITIALIZATION
Given an input state sroot, the algorithm begins by encoding this state into its latent representation hroot using an encoder Eθ. This latent representation serves as the root node of the search tree.

## A.2.2 Expansion Phase

- The algorithm initiates a set of candidate nodes, termed 'Open', starting with the root node.

- In each iteration, the algorithm considers every node in the 'Open' set, retrieves its latent state, and computes an interim value V¯ (node), which combines the cumulative rewards of the node's path with a value estimation from the value module Vθ.

- Using the path values of nodes in 'Open', the tree expansion policy, πtree, is computed. From this policy, a node, *node*∗, is sampled for expansion.

- Each possible action from *node*∗results in the creation of a child node. This is achieved by leveraging the transition module Tθ to predict the latent state of the child and the reward module Rθ to determine the associated reward. After expansion, *node*∗is removed from
'Open' and its children are added.

- This expansion process continues until a predetermined number of iterations, MAX_ITR, is reached.

## A.2.3 Backup Phase

- Starting from the leaf nodes, the algorithm backpropagates value estimates to the root.

- For each leaf node, its value is directly computed from the value module Vθ. For non-leaf nodes, the Q-value for each action is estimated by combining the reward for that action with the value of the corresponding child node.

- The value of a non-leaf node is set to the maximum Q-value among its actions.

## A.2.4 Output

The algorithm finally returns the Q-values associated with the root node, Q(*node*root), providing an estimation of the value of taking each action from the initial state. This explanation provides a high-level view of the D-TSN algorithm's operation. By breaking down the search process into expansion and backup phases, the pseudo-code highlights how D-TSN incrementally builds the search tree and then consolidates value estimates back to the root.

## B Continuity Of The Loss Function

In this section, we show that the Q-function represented by a search tree with fixed structure is continuous in network's parameter space. Suppose we have two functions f(x) and g(x) which are continuous at any point c in their domains. Lemma B.1. **Continuity of Composition***: The composition of two continuous functions, denoted as* f(g(x))*, retains continuity. (Theorem 4.7 in Rudin (1976))* Lemma B.2. **Continuity of Sum operation**: The result of adding two continuous functions, expressed as f(x) + g(x)*, is a continuous function. (Theorem 4.9 in Rudin (1976))*
Lemma B.3. **Continuity of Max operation**: Applying Max over two continuous functions, expressed as max(f(x), g(x)), results in a function that is continuous. Proof. Consider a function h(x) = max(f(x), g(x)), and we aim to demonstrate that h(x) is continuous. Now, h(x) can be expressed as a combination of continuous functions:

$$h(x)={\frac{f(x)+g(x)+|f(x)-g(x)|}{2}}$$

Since sums and absolute values of continuous functions are continuous (Rudin, 1976), h(x) is continuous. Hence, the maximum of two continuous functions is also a continuous function. Rewriting the Theorem 3.1 with proof below: Theorem B.4. *Given a set of parameterised modules that are continuous in the parameter space* θ, the Q-function computed by expanding a fixed search tree by composing these modules and backpropagating the children values using addition and max operations is continuous in the parameter space θ*. When the tree structure is not fixed, the continuity of the Q-values is not guaranteed.* Proof. Let us represent the set of parameters, encoder module, transition module, reward module, and value module respectively as θ, Eθ, Tθ, Rθ, and Vθ. These submodules are assumed to be composed of simple neural network architectures, comprising linear and convolutional layers, with the Rectified Linear Unit (ReLU) serving as the activation function. These submodules, therefore, are continuous within the parameter space. We can subsequently rewrite the Q-value as the output of a full tree expansion as follows:

Q(s (21) 0, a0) = Q(h0, a0)
$$(21)$$
$$\begin{array}{c}{{Q(s_{0},a_{0})=Q(h_{0},a_{0})}}\\ {{=r_{0}+V(h_{1})}}\end{array}$$

 $\begin{array}{l}\text{(22)}\\ \text{(23)}\\ \text{(24)}\\ \text{(25)}\end{array}$  . 
$$(26)$$
where,

$$h_{0}=\mathcal{E}_{\theta}(s_{0})$$ $$r_{t}=\mathcal{R}_{\theta}(h_{t},a_{t})$$ $$h_{t+1}=\mathcal{T}_{\theta}(h_{t},a_{t})$$ $$Q(h_{t},a_{t})=\mathcal{R}_{\theta}(h_{t},a_{t})+V(h_{t+1})$$ $$V(h_{t})=\begin{cases}\mathcal{V}_{\theta}(h_{t})&\text{if$h_{t}$is a leaf}\\ \max_{a}\left(Q(h_{t},a)\right)&\text{otherwise}\end{cases}$$

Given that Eθ, Rθ, and Tθ are continuous, h0, rt, and ht+1 in Equations (22) and (24) are similarly continuous (derived from Lemma B.1).

Further, V (ht) in Equation (26) can either be Vθ(ht), if ht is a leaf node, or maxa(Q(ht, a)) otherwise. In the first scenario, V (ht) retains continuity by assumption. In the second scenario, if Q(ht, at) is continuous, then V (ht) remains continuous as per Lemma B.3. Now, we show that Q(ht, at) is continuous using a recursive argument that for any node in the search tree, if the Q-values of all its child nodes are continuous, then its Q-value is also continuous.

Q-value of an internal tree node ht can be written as Q(ht, at) = Rθ(ht, at) + V (ht+1), where ht+1 = Tθ(ht, at) is the child node of ht. Considering the base case, when the ht+1 is a leaf node, then Q(ht, at) = Rθ(ht, at) + Vθ(ht+1), which is continuous as per Lemma B.2. Consequently, V (ht) maintains continuity. Applying this logic recursively, the Q-value Q(ht, at) of all the tree nodes maintains continuity.

Thus, we can decompose the Q-value at the root node, Q(s0, a0), as a composition of continuous functions, ensuring that the output Q-value, Q(s0, a0), is continuous.

Next, we show that the continuity of Q-function is not guaranteed when the structure of the search tree is not fixed by giving an example.

Consider a simple environment with two possible actions a1 and a2, and a state is represented by a 2-D vector. We parameterize the reward, value, transition functions as simple matrices: Ri ∈ R
2 is a reward function that maps a state to a real number representing the reward for doing action ai at the state; V ∈ R
2is the value function that maps a state to a real number representing the value of a state; Ti ∈ R
2×2is a transition function that maps a state to a next state after doing action ai. Consider a search tree with root node s ∈ R
2and a child node x ∈ R
2 derived by doing action a1. x has two child node n1 and n2. The path value of n1, defined by Equation (1), is V¯ (n1) = Rθ(s, a1) + Rθ(x, a1) + Vθ(n1) = RT
1 s + RT
1 x + V
T T1x. Similarly, the path value of n2 is V¯ (n2) = Rθ(*s, a*1) + Rθ(*x, a*2) + Vθ(n2) = RT1 s + RT
2 x + V
T T2x.

Consider we choose between n1 and n2 for the next expansion of the tree. When RT
1 x + V
T T1x >
RT2 x + V
T T2x, n1 is expanded, two children of n1 has value RT
1 T1x + V
T T1T1x and RT
2 T1x +
V
T T2T1x, value of x after backup is the maximum of value of n2 and the values of two children of n1, which is max{RT
2 x + V
T T2x, max{RT
1 T1x + V
T T1T1*x, R*T2 T1x + V
T T2T1x, }}.

When RT
1 x + V
T T1*x < R*T
2 x + V
T T2x, n2 is expanded, two children of n2 has value RT
1 T2x +
V
T T1T2x and RT
2 T2x + V
T T2T2x, value of x after backup is maximum of value of n1 and the values of two children of n2, which is max{RT
1 x + V
T T1x, max{RT
1 T2x + V
T T1T2*x, R*T
2 T2x +
V
T T2T2x, }}.

… …
Let's assign the following values to the mentioned symbols:

$$x=\begin{bmatrix}2\\ 2\end{bmatrix};\quad V=\begin{bmatrix}\theta\\ 1\end{bmatrix};\quad R_{1}=\begin{bmatrix}0\\ 1\end{bmatrix};\quad R_{2}=\begin{bmatrix}1\\ 0\end{bmatrix};\quad T_{1}=\begin{bmatrix}0&2\\ 1&0\end{bmatrix};\quad T_{2}=\begin{bmatrix}0&1\\ 1&2\end{bmatrix}.$$

We have RT1 x + V
T T1x = 4θ + 4, and RT
2 x + V
T T2x = 2θ + 8. When θ > 2, n1 will be expanded, and when θ < 2, n2 will be expanded, see Figure 2 for a visualization of the example. We have

$$\lim_{\theta\to a^{+}}Q(s,a_{1})$$ $$=R_{1}^{T}s+\lim_{\theta\to2^{+}}\max(R_{2}^{T}x+V^{T}T_{2}x,\max(R_{1}^{T}T_{1}x+V^{T}T_{1}T_{1}x,R_{2}^{T}T_{1}x+V^{T}T_{2}T_{1}x,))\tag{27}$$ $$=16$$
$$\lim_{\theta\to2^{-}}Q(s,a_{1})$$ $$=R_{1}^{T}s+\lim_{\theta\to2^{-}}\max(R_{1}^{T}x+V^{T}T_{1}x,\max(R_{1}^{T}T_{2}x+V^{T}T_{1}T_{2}x,R_{2}^{T}T_{2}x+V^{T}T_{2}T_{2}x,))\tag{28}$$ $$=32$$

Combining Equation (27) and Equation (28) we have that Q(*r, a*1) is discontinuous at θ = 2.

## C Derivation Of The Gradient Of The Expected Loss Function

Let us represent the Q-values predicted by D-TSN as Qθ(*s, a*|τ ), which depends on the final tree τ sampled after T trials of the online search. Let us denote the corresponding loss function on this output Q-value as L
Qθ(*s, a*|τ )
. Our objective is to compute the gradient of the expected loss value, averaging over trees sampled. The gradient of expected loss, considering the expectation over the sampled trees, is derived as:

∇θL(s, a) = ∇θEτ L Qθ(s, a|τ )  = ∇θ X τ πθ(τ )L Qθ(s, a|τ )  =X τ ∇θ πθ(τ )L Qθ(s, a|τ )  =X τ L Qθ(s, a|τ ) ∇θπθ(τ ) +X τ πθ(τ )∇θL Qθ(s, a|τ )  =X τ πθ(τ )L Qθ(s, a|τ ) ∇θ log πθ(τ ) +X τ πθ(τ )∇θL Qθ(s, a|τ )  = Eτ L Qθ(s, a|τ ) ∇θ log πθ(τ ) + ∇θL Qθ(s, a|τ )  = Eτ L Qθ(s, a|τ ) ∇θ logY T t=1 πθ(nt|τt) + ∇θL Qθ(s, a|τ )  = Eτ L Qθ(s, a|τ ) X T t=1 ∇θ log πθ(nt|τt) + ∇θL Qθ(s, a|τ )  Leveraging the telescoping sum trick, as elaborated in Section 3.7, the gradient of the expected loss
can be rewritten as a lower-variance estimate:
$$\nabla_{\theta}L(s,a)=\mathbb{E}_{\tau}\bigg[\sum_{t=1}^{T}\nabla_{\theta}\log\pi_{\theta}(n_{t}|\tau_{t})R_{t}+\nabla_{\theta}\mathcal{L}\Big(Q_{\theta}(s,a|\tau)\Big)\bigg]$$
where

$R_{t}=\sum_{i=t}^{T}r_{i}=\mathcal{L}_{T}-\mathcal{L}_{t-1}$  $\mathcal{L}_{t}=\mathcal{L}\Big{(}Q_{\theta}(s,a|\tau_{t})\Big{)},\text{representing the loss value after the}t^{th}\text{search iteration.}$
In practice, we utilize the single-sample estimate for the expected gradient, as elaborated in Schulman et al. (2015)

## D Experiments

D.1 TEST DOMAINS We use a grid navigation task and the Procgen games to study D-TSN under the scenario when world models are not available. A visualization of these two tasks can be viewed in Figure 3. D.1.1 NAVIGATION The grid navigation task serves as a foundational test that mimics the challenges a robot might face when navigating in a 2D grid environment. This environment provides both a quantitative metric and a qualitative visualization to understand an agent's capacity to generalize its policy. Specifically, this task involves a 20 × 20 grid with a central hall. At the beginning of each episode, the robot is positioned at a random point within this central hall. Simultaneously, a goal position is sampled randomly at a location outside the hall, challenging the robot to find its way out and reach this target. There are two variations of this task. The first provides the robot with a single exit from the central hall, while the second offers two exits. The single-exit hall scenario is similar to the two-exit scenario but requires a longer-horizon planning to successfully evade the walls to exit the hall and reach the goal.

Navigation - 2 exits a Procgen Games Navigation - 1 exit VALUI
INPUT
ENCODIS
LATENT
  STATE
LATENT
PREDICTED
STATE
MODULE
STATE
MODULE
VALUE
L
Vo V i ( h )
h LATENT
TRANSITION
PREDICTED LATENT
LATENT
REWARD
MODULE
PREDICTED
STATE
MODULI
next state STATE
REWARDS
h T
R
h t+1 Rh, a)
h a

## D.1.2  P Rocgen

Procgen is a unique suite consisting of 16 game-like environments, each of which is procedurally generated.  This means that they are designed to present slightly altered levels every time they are played. Such design intricacy makes Procgen an ideal choice to test an agent's generalization capabilities. It stands in contrast to other commonly used testing suites, like the renowned Atari 200 games (Mnih et al., 2013; 2016; Badia et al., 2020). The diverse array of environments within Procgen emphasizes the pivotal role of robust policy learning. The environmental diversity in Procgen