# Optionzero: Planning With Learned Options

Po-Wei Huang1,2, Pei-Chiun Peng1,2, Hung Guei1**, Ti-Rong Wu**1†
1Institute of Information Science, Academia Sinica, Taiwan 2Department of Computer Science, National Yang Ming Chiao Tung University, Taiwan

## Abstract

Planning with options - a sequence of primitive actions - has been shown effective in reinforcement learning within complex environments. Previous studies have focused on planning with predefined options or learned options through expert demonstration data. Inspired by MuZero, which learns superhuman heuristics without any human knowledge, we propose a novel approach, named *OptionZero*. OptionZero incorporates an *option network* into MuZero, providing autonomous discovery of options through self-play games. Furthermore, we modify the dynamics network to provide environment transitions when using options, allowing searching deeper under the same simulation constraints. Empirical experiments conducted in 26 Atari games demonstrate that OptionZero outperforms MuZero, achieving a 131.58% improvement in mean human-normalized score. Our behavior analysis shows that OptionZero not only learns options but also acquires strategic skills tailored to different game characteristics. Our findings show promising directions for discovering and using options in planning. Our code is available at https://rlg.iis.sinica.edu.tw/papers/optionzero.

## 1 Introduction

Reinforcement learning is a decision-making process in which an agent interacts with environments by selecting actions at each step to maximize long-term rewards. Actions, commonly referred to as primitive action, advance the state by one step each. While this granularity allows precise control at each time step, it can lead to inefficiencies in scenarios where predictable sequences of actions are beneficial. For example, in a maze navigation task, it is more efficient to choose a sequence of actions - such as following a straightforward path until reaching a new junction - rather than deciding an action at each time step. This approach reduces the frequency of decision-making and accelerates the learning process. To address these challenges, the concept of *options* (Sutton et al., 1999) has emerged, providing a framework for executing temporally extended actions based on the current state. Options bridge single-step decision-making and strategic long-term planning, not only speeding up the learning process to handle complex scenarios but also simplifying the decisionmaking by reducing the frequency of choices an agent must consider. Previous works have proposed adopting the concept of options by either predefining options or learning from expert demonstration data (Sharma et al., 2016; Durugkar et al., 2016; de Waard et al., 2016; Gabor et al., 2019; Czechowski et al., 2021; Kujanpa¨a et al., 2023; 2024). However, the pre- ¨ defined options often rely on a deep understanding of specific environments, and expert data may not be available for every environment, making it difficult to generalize these methods to other environments. Moreover, when planning with options, previous methods require recurrently executing each action within the option to obtain the next states (de Waard et al., 2016) or verifying whether the subgoal can be reached through primitive actions (Czechowski et al., 2021; Kujanpa¨a et al., ¨
2023; 2024). This increases the computational cost when executing longer options during planning, especially in scenarios where environment transitions are expensive. Inspired by the success of MuZero (Schrittwieser et al., 2020), which employs a learned dynamics network to simulate the environment transitions during planning and achieves superhuman performance from scratch without requiring any human knowledge, this paper proposes a novel approach,
†Corresponding author: tirongwu@iis.sinica.edu.tw 1 named *OptionZero*. We modify the MuZero algorithm by integrating an *option network* that predicts the most likely option for each state. During training, OptionZero autonomously discovers options through self-play games and utilizes them during planning, eliminating the requirement for designing options in advance. Furthermore, OptionZero improves the dynamics network to efficiently simulate environment transitions across multiple states with options, significantly reducing the overhead for iterative examination of internal states. We conduct experiments on Atari games, which are visually complex environments with relatively small frame differences between states, making them suitable for learning options. Our results show that using options with maximum lengths of 3 and 6, OptionZero achieved mean human-normalized scores of 1054.30% and 1025.56%, respectively. In contrast, MuZero achieves a score of only 922.72%. In addition, we provide a comprehensive behavior analysis to examine the options learned and used during planning. Interestingly, the adoption of options varies across different games, aligning with the unique characteristics of each game. This demonstrates that OptionZero effectively discovers options tailored to the specific game states and challenges of each environment. In conclusion, our findings suggest that OptionZero not only discovers options without human knowledge but also maintains efficiency during planning. This makes OptionZero easily applicable to other applications, further extending the versatility of the MuZero algorithm.

## 2 Related Works

Numerous studies have explored the concepts of *options* in reinforcement learning. For example, de Waard et al. (2016) incorporated options from a predefined option set into Monte Carlo tree search
(MCTS) and extended it to focus exploration on higher valued options during planning. Sharma et al.

(2016) proposed using two policies for planning: one determines which primitive action to use, and the other determines how many times to repeat that action. Durugkar et al. (2016) explored the effects of repetition and frequency by statistics in Atari games. Vezhnevets et al. (2016) introduced a method which learns options through end-to-end reinforcement learning. Lakshminarayanan et al. (2017) proposed a method that allows agents to dynamically adjust rates of repeated actions. Bacon et al. (2017) derived an option-critic framework, which learns a policy over options and a policy within options. The option policy not only determines how to select and execute an action within options but also learns when to terminate the option. Kim et al. (2023) proposed to adaptively integrate multiple exploration strategies for options based on the option-critic framework. Riemer et al. (2020) introduced a parameter-sharing approach for deep option learning. Young & Sutton (2023) discovered options by learning the option policies and integrated them with a Monte Carlo search. Jinnai et al. (2019) formalized the problem of selecting the optimal option set, and produced an algorithm for discovering the suboptimal option set for planning. Veeriah et al. (2021) proposed a meta-gradient approach for discovering reusable, task-independent options. In addition, several works have studied subgoals, which represent a target state to achieve after several time steps, either segmented by predefined time step intervals or predicted dynamically by a learned network. For example, Gabor et al. (2019) used predefined subgoals for planning in MCTS. Czechowski et al. (2021) introduced a Subgoal Search method to obtain fixed-length subgoals with a low-level policy that predicts primitive actions for reaching subgoals. Kujanpa¨a et al. (2023) proposed Hierarchical ¨ Imitation Planning with Search (HIPS), which learns subgoals from expert demonstration data. Kujanpa¨a et al. (2024) extended HIPS to HIPS- ¨ ϵ, adding a low-level (primitive action) search to the high-level (subgoal) search, guaranteeing that subgoals are reachable. In summary, these previous works either adopt predefined options, learn subgoals from expert data, or do not incorporate options in MCTS planning. Compared to these works, our goal is to automatically discover options without relying on predefined options or expert data and to use options during planning.

## 3 Muzero

MuZero (Schrittwieser et al., 2020) is based on the foundation of AlphaZero (Silver et al., 2018),
distinguishing itself by learning environment transitions using neural networks. This allows MuZero to plan in advance without extra interaction with the environment, which is particularly advantageous in environments where such interactions are computationally expensive. Consequently, MuZero has achieved success in a wide range of domains (Schrittwieser et al., 2020; Danihelka et al., 2022; Antonoglou et al., 2021; Hubert et al., 2021; Mandhane et al., 2022; Wang et al., 2023). For planning, MuZero adopts Monte Carlo tree search (MCTS) (Kocsis & Szepesvari, 2006; ´ Coulom, 2007; Browne et al., 2012), integrating three distinct networks: representation, dynamics, and *prediction*. Specifically, for an observation xt at time step t, the search determines an action at+1 using multiple simulations, each consisting of three phases: selection, expansion, and backup.

The selection starts from the hidden state root node s 0, selecting child nodes recursively until an unexpanded leaf node s lis reached. For each non-leaf node s k, the child node s k+1 (corresponding to action a k+1) is selected according to the highest PUCT (Rosin, 2011; Silver et al., 2017) score:

$$Q(s^{k},a^{k+1})+P(s^{k},a^{k+1})\times\frac{\sqrt{\sum_{b}N(s^{k},b)}}{1+N(s^{k},a^{k+1})}\times c_{puct},\tag{1}$$

where Q(s k, ak+1) is the estimated Q-value, P(s k, ak+1) is the prior probability, N(s k, ak+1) is the visit counts, and c*puct* is a constant for exploration. In the expansion phase, to expand the leaf node s l, the dynamics network gθ is applied to perform the environmental transition: s l, rl = gθ(s l−1, al),
where r lis the immediate reward. Note that when l = 0, the representation network hθ is used to initialize the root node: s 0 = hθ(xt). Then, the prediction network fθ is applied to evaluate its policy and value: p l, vl = fθ(s l), where p lis used for P(s l, a) and v lis the estimated value for s l.

The backup phase uses the obtained value v lto update the statistics Q(s k, ak+1) and N(s k, ak+1):

$$Q(s^{k},a^{k+1}):=\frac{N(s^{k},a^{k+1})\times Q(s^{k},a^{k+1})+G^{k+1}}{N(s^{k},a^{k+1})+1}\text{and}N(s^{k},a^{k+1}):=N(s^{k},a^{k+1})+1,\tag{2}$$

where Gk+1 =Pl−k−1 τ=0 γ τr k+1+τ + γ l−kv lis the cumulative reward discounted by a factor γ.

During training, MuZero continuously performs self-play and optimization. The self-play process collects game trajectories, including xt, πt, at+1, ut+1, and zt for all time steps. For each xt, MCTS is conducted to produce the search policy πt. Then, an action at+1 ∼ πt is applied to the environment, obtaining an immediate reward ut+1 and moving forward to the next observation xt+1. In addition, zt is the n-step return. The optimization process updates the networks by sampling records from collected trajectories. For each sampled record xt, the process uses the networks to unroll it for K steps to obtain s k t with corresponding p k t, v k t, and r k tfor 0 ≤ k ≤ K, where s 0 t = hθ(xt) and s k t, rk t = gθ(s k−1 t, at+k) for k > 0. Then, all networks are jointly updated using

$$L_{t}=\sum_{k=0}^{K}l^{p}(\pi_{t+k},p_{t}^{k})+\sum_{k=0}^{K}l^{v}(z_{t+k},v_{t}^{k})+\sum_{k=1}^{K}l^{r}(u_{t+k},r_{t}^{k})+c||\theta||^{2},\tag{3}$$

where l pis the policy loss, l vis value loss, l ris the reward loss, and c||θ||2is the L2 normalization.

## 4 Optionzero 4.1 Option Network

Options are the generalization of actions to include temporally extended actions, which is applied interchangeably with primitive actions (Sutton et al., 1999; Bacon et al., 2017). In this context, options on Markov decision process (MDP) form a special case of decision problem known as a semi-Markov decision process (SMDP). Given a state st at time step t and an option length L,
we enumerate all possible options, denoted as ot+1 = {at+1, at+2*, ..., a*t+L}, by considering every sequence of primitive actions starting at st. When executing the option ot+1, we obtain a sequence of states and actions st, at+1, st+1, at+2, ..., st+L−1, at+L, st+L. Ideally, the probability of selecting each option can be calculated by multiplying the probabilities of each primitive action within the option, as illustrated in Figure 1a. For example, when L = 4, the probability of option o1 = {a1, a2, a3, a4} for s0 is P(a1) × P(a2) × P(a3) × P(a4) = 0.8 4 = 0.4096, where P(ai) is the probability of selecting action ai. A naive approach to obtaining the option probabilities involves using a policy network to evaluate all possible states from st to st+L. However, this approach is computationally expensive, and the number of options grows exponentially as the option length L increases, making it infeasible to generate all options. In practice, since most options occur infrequently due to their lower probabilities, our primary interest lies in the *dominant option*. The dominant option, o1 = {a1, a2*, ..., a*l}, is defined such that

0 0
′
0.2 0.8 (1)
  
 0 0.8 0.2
  
 **0.75** 0 0.25
⋮
(1
′ ) 0.75
(2
′ ) 0.75
(3
′ ) 0.75
⋮
0.8 (2)
⋮⋮ 0.8 (3)
0.8 (4)
0 0
′
 0 **0.64** 0.36
 **0.5625** 0 0.4375 0 **0.512** 0.488 1
′
= 1
′
, 2
′
= {, }
1 = 1, 2, 3
= {, , }
 0.4219 0 **0.5781**
⋮
(b)
 0 0.4096 **0.5904**
⋮
(a)
Πli=1P(ai) > 0.5 ∧ Π
l+1 i=1P(ai) ≤ 0.5, where Πli=1P(ai) is the cumulative product of probabilities and 1 ≤ l ≤ L. For example, in Figure 1a, the dominant option for s0 is o1 = {a1, a2, a3} because P(a1) × P(a2) × P(a3) = 0.512 and P(a1) × P(a2) × P(a3) × P(a4) = 0.4096, and the dominant option for s
′0is o
′1 = {a
′1, a′2}. This indicates that the length of the dominant option can vary, depending on how long the cumulative probabilities remain above the threshold of 0.5. In addition, this design ensures that there is only one dominant option for each state s, effectively preventing exponential growth in the number of possible options. Next, we incorporate the *option network* into the prediction network in MuZero, denoted as Ω*, p, v* = fθ(s), which predicts an additional option output, Ω, for predicting the dominant option at state s. Given the maximum option length L, the option network produces L distributions, Ω = {ω1, ω2*, ..., ω*L}, which are used to derive the dominant option, o1 = {a
∗
1, a∗2*, ..., a*∗
l}, where a
∗
i = arg maxa ωi(a). Each ωi represents the conditional cumulative product probability of selecting a sequence of actions from a
∗
1to a
∗
i, i.e, ωi(a
∗
i) = Πij=1P(a
∗
j). Furthermore, a virtual action, called stop, is introduced to provide a termination condition. This *stop* action is the sum of probabilities for all actions except a
∗, defined as ω(*stop*) = 1 − ω(a
∗). To derive the dominant option from Ω, we progressively examine each ωi from ω1 to ωL, selecting a
∗
ias a
∗
i = arg maxa ωi(a)
until i = L or a
∗
ibecomes a *stop* action. We provide an example for obtaining the dominant options for state s0 and s
′0, as shown in Figure 1b. This method allows for determining the dominant option at any state s without recurrently evaluating future states, reducing the computational costs.

## 4.2 Planning With Dominant Option In Mcts

This subsection describes the modifications to MCTS implemented in OptionZero to incorporate planning with the dominant option. For simplicity, we will use *option* to represent the *dominant* option in the rest of this paper. The planning generally follows the MuZero but with two modifications, including the network architecture and MCTS. For the network architecture, we add an additional option output to the prediction network, denoted as Ω
k, pk, vk = fθ(s k), where Ω
k, p k, and v kare the option distribution, policy distribution, and value at state s k, respectively. Note that we use superscript s kinstead of subscript sk in this subsection. This is because s krepresents the hidden state, obtained after unrolling k steps by the dynamics network from the initial hidden state s 0. In contrast, sk denotes the actual observed state in the environment at time step k. As illustrated in the previous section, we can derive the option o kfrom Ω
k. The dynamics network, denoted as s k+l, rk+1,k+l = gθ(s k, Ak+1), is modified to predict the next hidden state s k+land the accumulated discounted reward r k+1,k+l upon executing a composite action Ak+1 at s k. The composite action, Ak+1, can be either a primitive action a k+1 or an option o k+1 with the length l. The accumulated discounted reward r k+1,k+lis computed as Pli=1 γ i−1r k+i,k+i, where r i,i represents the single immediate reward obtained by applying a iat state s i−1. Note that the dynamics network supports unrolling the option directly, eliminating the need to recurrently evaluate each subsequent state from s kto s k+l.

Next, we demonstrate the incorporation of options within MCTS. The search tree retains the structure of the original MCTS but includes edges for options that can bypass multiple nodes directly, as shown in Figure 2. This adaptation integrates options subtly while preserving the internal actions within options, allowing the tree to traverse states using either a primitive action or an option. Each

0 0 Selection Expansion Backup primitive selection 0

0

2 1 0 primitive expansion 

2 1

2 1 option expansion option selection 3 3 3
edge within the tree is associated with statistics {N(s, A), Q(s, A), P(s, A), R(s, A)}, representing its visit counts, estimated Q-value, prior probability, and reward, respectively. Moreover, for nodes that possess both a primitive edge and an option edge, the statistics of the primitive edge are designed to include those of the option edge. For example, if the tree traverses the node via the option edge, the visit counts for both the primitive and option edges are incremented. This ensures the statistics remain consistent with MuZero when only primitive edges are considered within the tree. We illustrate the modifications made to each phase of MCTS in the following.

Selection. For any node s k, the selection of the next child node includes two stages: primitive selection and *option selection*. The primitive selection only considers primitive child nodes and remains consistent with MuZero by selecting the next action a k+1 based on the PUCT score based on equation 1. If the selected action a k+1 matches the first action in option o k+1, we then proceed with the option selection to determine whether to select this primitive action a k+1 or option o k+1.

Option selection is similar to the primitive selection, using the PUCT score to compare both the primitive and option nodes. Since the option node is a successor node of the primitive node, the statistics for the primitive node need to be adjusted to exclude contributions from the option node in the option selection. We select either primitive or option nodes based on the higher PUCT score, which is calculated as follows:

$$\begin{cases}Q(s^{k},o^{k+1})+P(s^{k},o^{k+1})\times\frac{\sqrt{\sum_{i=1}^{n}N(o^{k},b)}}{1+N(o^{k},o^{k+1})}\times c_{puct}&\text{if$option node},\\ \hat{Q}(s^{k},a^{k+1})+\hat{P}(s^{k},a^{k+1})\times\frac{\sqrt{\sum_{i=1}^{n}N(o^{k},b)}}{1+N(o^{k},o^{k+1})-N(o^{k},o^{k+1}))}\times c_{puct}&\text{if$primitive node}.\end{cases}\tag{4}$$

The P˜(s k, ak+1) = max(0, P(s k, ak+1) − P(s k, ok+1)) ensures that the prior remains non-negative. The adjusted estimated Q-value, Q˜(s k, ak+1), is calculated as N(s k,ak+1)Q(s k,ak+1)−N(s k,ok+1)Q(s k,ok+1)
N(sk,ak+1)−N(sk,ok+1). Note that Pb N(s k, b) is the total visit counts for selecting a k+1 and o k+1, which is equivalent to N(s k, ak+1) because the statistics of the primitive node already include the statistics of option node. The selection process begins at the root node s 0 until an unevaluated node s lis reached, as shown in Figure 2.

Expansion. Assume the last two node in the selection path is s m and s l, where s m is the parent node of s l. To expand node s l, we derive r m+1,l, Ω
l, pl, vl using the dynamics and prediction network.

The reward r m+1,l is from s m to s land used to initialize the edge R(s m, Am+1) = r m+1,l. The edge of all primitive child nodes are initialized as {N(s l, al+1) = R(s l, al+1) = Q(s l, al+1) = 0}
and P(s l, al+1) = p l. Then, if the length of option o l+1 derived from Ω
lis larger than 1, we expand the internal nodes following the action within the option. The statistics of each edge are initialized as 0 since these internal nodes are unevaluated. For the option node, the edge is initialized as {N(s l, ol+1) = R(s l, ol+1) = Q(s l, ol+1) = 0} and P(s l, ol+1) = ω l.

Backup. The backup phase updates the visit counts and estimated Q-value from s l back to s 0. Considering that s l may be accessed through various selection paths from s 0, all edges on the possible paths from s 0to s l must be updated. This ensures that both the visited count and estimated Q-value of all nodes remain consistent within the search, regardless of the selection path chosen. We first obtain the l − k-step estimate of the cumulative discounted reward as Gk = r k+1,l + γ l−kv l, where r k+1,l is the discounted reward from s kto s land v lis the value at state s l. Since not all edges have been evaluated, we calculate r k+1,l by using r 1,l−r 1,k γk , where r 1,k and r 1,l represent discounted rewards from the root node s 0to s kand s l, respectively. Then, we update the estimated Q-value of each edge, Q(s k, Ak+1), using a similar approach as introduced in equation 2:

), using a similar approach as introduced in equation 2.1  ${Q(s^k,\mathcal{A}^{k+1}):=\frac{N(s^k,\mathcal{A}^{k+1})\times Q(s^k,\mathcal{A}^{k+1})+G^{k+1}}{N(s^k,\mathcal{A}^{k+1})+1},}$  ${N(s^k,\mathcal{A}^{k+1}):=N(s^k,\mathcal{A}^{k+1})+1,}$  . 
$$(S)$$
During planning, the MCTS performs a fixed number of simulations, each including the above three phases. Upon the search completed, MCTS selects a child node from the root node s 0 based on probabilities proportional to their visit counts and performs the composite action in the environment. Overall, the additional complexity introduced by OptionZero, including the costs for the option network and maintaining statistics for option edges, is negligible compared to the original MuZero.

## 4.3 Training Optionzero

We describe the optimization process using the self-play trajectory in OptionZero, as shown in Figure 3. For better illustration, we utilize three additional symbols, including O, U, τ , and τˆ.

Given a state st at time step t, Oi represents the i-th executed composite action starting from st, Uiis defined as the discounted reward obtained after executing Oi, τi denotes the action sequence length of Oi, and τˆi =Pij=1 τj is the accumulated length from O1 to Oi. For example, in Figure 3, from the perspective of st, we can obtain O1 = ot+1 = {at+1, at+2}, O2 = {at+3}, with corresponding discounted rewards U1 = ut+1 + γut+2, U2 = ut+3, action sequence lengths τ1 = 2, τ2 = 1, and accumulated lengths τˆ1 = 2, τˆ2 = 3. Then, the observed discounted reward U1 at st is calculated as Pτ1−1 i=0 γ iut+1+i, aggregating the observed rewards provided by the environment with a discount factor γ. The n-step return zt is calculated as U1 + γ τˆ1 U2 + ... + γ τˆT−1 UT + γ τˆT vt+ˆτT, where τˆT = n. Note that vt+n is not always available, as st+n may be bypassed when options are executed.

Consequently, we identify the smallest T such that τˆT ≥ n, ensuring that the step count for the n-step return approximates n as closely as possible. In Figure 3, if n = 5, since st+5 is skipped by option, we then approximate the n-step return by using vt+6 as zt = U1 + γ 2U2 + γ 3U3 + γ 4U4 + γ 6vt+6.

Next, we describe the training target for both the policy and option network. The search policy distribution πt is calculated in the same manner as in MuZero. For the option network, given an option length L at state st, we examine its subsequent states to derive the training target, Φt = {ϕt, ϕt+1*, ..., ϕ*t+L−1}. Each ϕiis a one-hot vector corresponding to the training target for ωi. Specifically, for any state s, if the option network predicts an option o = {a1, a2*, ..., a*l} that exactly matches the composite action O = {a1, a2*, ..., a*l} executed in the environment, then the option learns the action sequence, i.e., ϕi = onehot(ai+1) for 0 ≤ i ≤ l − 1. Conversely, if o ̸= O, then the option learns to *stop*, i.e., ϕi = onehot(*stop*). We iterate this process to set each ϕ from st to st+L−1. If ϕt+iis set to learn *stop*, subsequent ϕt+j should follow, i.e., ϕt+j = onehot(*stop*) for i ≤ j ≤ L − 1. Note that if the length of predicted option oiis zero, oiis defined as {ai+1},
where ai+1 = arg maxa pi(a) is determined according to the policy network. This method ensures that the option network eventually learns the cumulative probability of the dominant option, as described in subsection 4.1. Figure 3 shows an example of setting the training target for the option network. If ot+2 ̸= O2, then the option network learns {at+1, at+2, stop}, {stop, stop, *stop*}, and {at+4, at+5, at+6}, for st, st+2, and st+3, respectively.

1 = +1, +2 2 = {+3} 3 = {+4} 4 = {+5, +6}
⋯
 +2 +3 +4 +6 1 = +1 + +2 2 = +3 3 = +4 4 = +5 + +6
 +1 +2 +3 +4 +5 +6 onehot onehot onehot onehot +4 onehot +5 onehot +6 onehot +1 onehot +2 onehot 1 = 2 2 = 1 3 = 1 4 = 2
+1 = 1 +3 ≠ 2 +4 = 3 +5 = 4
⋯
Φ Φ+2 Φ+3
⋯
During the optimization phase, the sampled state st is trained with K unrolling steps, where each step can be either a primitive action or an option. This enables the dynamics network to learn the environment transitions that incorporate options. The loss is modified from equation 3 as follows:

$$L_{t}=\sum_{k=0}^{K}l^{p}(\pi_{t+\hat{\pi}_{k}},p_{t}^{k})+\sum_{k=0}^{K}l^{v}(z_{t+\hat{\pi}_{k}},v_{t}^{k})+\sum_{k=1}^{K}l^{v}(\mathcal{U}_{k},r_{t}^{k})+\sum_{k=0}^{K}l^{v}(\Phi_{t+\hat{\pi}_{k}},\Omega_{t}^{k})+c||\theta||^{2},\tag{6}$$

where τˆ0 = 0. Note that the option loss l oincludes L cross-entropy losses.

## 5 Experiment 5.1 Optionzero In Gridworld

We first train OptionZero in *GridWorld*, a toy environment where the objective is to navigate an agent through a grid map with walls from a start position (S) to a goal (G) via the shortest possible route. The maximum option length is set to nine. Other training details are provided in Appendix A. Figure 4 shows the options learned by the option network at four stages of training: 25%, 50%, 75%, and 100% completion. It can be observed that the learning behavior of OptionZero evolves distinctly across different stages. In the early stage (25%), the model mainly relies on primitive actions, identifying options only when approaching the goal. In the middle stages (50% and 75%), the model begins to establish longer options, progressively learning options with lengths from two up to nine. In the final stage (100%), the model has learned the optimal shortest path using options. Notably, using only primitive actions, the optimal path requires an agent to take at least 30 actions. In contrast, OptionZero achieves this with just four options, accelerating the training process by approximately 7.5 times in this example. This substantial reduction highlights OptionZero's efficacy, especially in more complex environments. This experiment also shows that the option network can progressively learn and refine options during training, without requiring predefined options.

(a) 25% (b) 50% (c) 75% (d) 100%

## 5.2 Optionzero In Atari Games

Next, we evaluate OptionZero on *Atari* games, which are commonly used for investigating options (Sharma et al., 2016; de Waard et al., 2016; Durugkar et al., 2016; Bacon et al., 2017; Vezhnevets et al., 2016; Kim et al., 2023; Lakshminarayanan et al., 2017; Riemer et al., 2020) due to their visually complex environments and subtle frame differences between states, making training with primitive actions inefficient. We train three OptionZero models, denoted as ℓ1, ℓ3, and ℓ6, each configured with maximum option lengths L = 1, 3, and 6, respectively. Detailed experiment setups are provided in Appendix B. The model ℓ1 serves as a baseline, identical to MuZero, where no options are used during training. In addition, we adopt the frameskip technique (Mnih et al., 2015), commonly used in training on Atari games, set to 4. Namely, this results in a frame difference between 24 states when executing an option of length 6. This requires OptionZero to strategically utilize options when necessary, rather than indiscriminately.

Table 1 shows the results of 26 Atari games. Both ℓ3 and ℓ6 outperform the baseline ℓ1 in mean and median human-normalized scores, with ℓ3 achieving the best performance at 1054.30% and 391.69%, representing improvements of 131.58% and 63.29% over ℓ1. Overall, 20 out of 26 games perform better than ℓ1 for ℓ3, and 17 for ℓ6. There are 12 games where scores consistently increase as the option length increases. For example, in *up n down*, the scores rise by 63810.83 and 79503.9 from ℓ1 to ℓ6. Conversely, there are only four games where scores decrease as the option length increases. For example, in *bank heist*, scores drop by 156.54 and 511.26, respectively. We find that this decline is likely due to the difficulty of the dynamics network in learning environment transitions (Vries et al., 2021; He et al., 2024; Guei et al., 2024) in games with more complex action spaces. As the option length increases, the number of possible option combinations grows. Although we focus on learning the dominant option, the dynamics network still needs to learn across all dominant options. In games like *bank heist*, which offers a wide range of strategic possibilities for different option combinations, the learning complexity for the dynamics network increases. Nevertheless, most of the games still improve when training with options, demonstrating that options enable more effective planning.

| Game            | Random   | Human     | OptionZero   |            |            |
|-----------------|----------|-----------|--------------|------------|------------|
| ℓ1              | ℓ3       | ℓ6        |              |            |            |
| alien           | 128.30   | 6,371.30  | 2,437.30     | 2,900.07   | 3,748.73   |
| amidar          | 11.79    | 1,540.43  | 780.26       | 820.77     | 862.17     |
| assault         | 166.95   | 628.89    | 18,389.88    | 19,302.04  | 21,593.53  |
| asterix         | 164.50   | 7,536.00  | 177,128.50   | 188,999.00 | 187,716.00 |
| bank heist      | 21.70    | 644.50    | 1,097.63     | 950.13     | 906.53     |
| battle zone     | 3,560.00 | 33,030.00 | 53,326.67    | 53,583.33  | 39,556.67  |
| boxing          | -1.46    | 9.61      | 97.71        | 95.09      | 96.00      |
| breakout        | 1.77     | 27.86     | 371.30       | 375.58     | 364.11     |
| chopper command | 644.00   | 8,930.00  | 43,951.67    | 60,181.67  | 45,518.67  |
| crazy climber   | 9,337.00 | 32,667.00 | 110,634.00   | 114,390.00 | 128,455.67 |
| demon attack    | 208.25   | 3,442.85  | 103,823.17   | 117,270.57 | 109,092.33 |
| freeway         | 0.17     | 25.61     | 29.46        | 31.06      | 31.45      |
| frostbite       | 90.80    | 4,202.80  | 3,183.40     | 3,641.10   | 6,047.97   |
| gopher          | 250.00   | 2,311.00  | 70,985.27    | 68,240.60  | 63,951.47  |
| hero            | 1,580.30 | 25,839.40 | 13,568.20    | 19,073.18  | 19,919.22  |
| jamesbond       | 33.50    | 368.50    | 8,155.50     | 13,276.67  | 8,571.17   |
| kangaroo        | 100.00   | 2,739.00  | 8,929.67     | 12,294.00  | 13,951.33  |
| krull           | 1,151.90 | 2,109.10  | 10,255.37    | 10,098.83  | 9,587.57   |
| kung fu master  | 304.00   | 20,786.80 | 66,304.67    | 68,528.33  | 69,452.33  |
| ms pacman       | 197.80   | 15,375.05 | 3,695.60     | 4,952.37   | 4,706.63   |
| pong            | -17.95   | 15.46     | 19.37        | 15.49      | 17.39      |
| private eye     | 662.78   | 64,169.07 | 116.83       | 90.76      | 83.24      |
| qbert           | 159.38   | 12,085.00 | 17,155.50    | 30,748.42  | 36,328.08  |
| road runner     | 200.00   | 6,878.00  | 26,971.33    | 32,786.67  | 21,699.67  |
| seaquest        | 215.50   | 40,425.80 | 3,592.53     | 5,606.63   | 6,754.50   |
| up n down       | 707.20   | 9,896.10  | 217,021.60   | 280,832.43 | 360,336.33 |
| Mean (%)        | 0.00     | 100.00%   | 922.72%      | 1054.30%   | 1025.56%   |
| Median (%)      | 0.00     | 100.00%   | 328.40%      | 391.69%    | 329.77%    |

## 5.3 Option Utilization And Behavior Analysis

This subsection analyzes how options are applied to better understand the planning process of OptionZero. Table 2 presents the average percentages of primitive actions (% a) and options (% o),
and the distribution of different option lengths (% l) across 26 games. In addition, columns "¯l", "%
Rpt.", and "% NRpt." show the average action sequence length executed in games, the proportions of options that repeat a single primitive action or involve more than one action types, respectively. Detailed statistics for each game are provided in Appendix D. From the table, we observe that primitive actions are generally the majority, accounting for over 60% in both ℓ3 and ℓ6. This is because Atari uses a frameskip of four, which means that each primitive action already spans across four states. The use of frameskip four is well-established in previous research, and our experiments further corroborate these findings. However, there are still nearly 30% of states that can adopt options. When comparing the use of options, it is notable that ℓ6 applies options less frequently, with a percentage of 30.57% compared to 37.62% in ℓ3. However, the average action sequence length for ℓ6 (2.03) is longer than that of ℓ3 (1.69). This is because action sequences that involve taking two consecutive three-step options in ℓ3 are merged into a single six-step option in ℓ6, resulting in a lower usage rate of options in statistics. In summary, our findings reveal that OptionZero strategically learns to utilize options as well as employ primitive actions at critical states instead of indiscriminately utilizing longer options. Next, among the different option lengths used, we observe that generally the longer option lengths are preferred. This suggests that if a state already has applicable options, it is likely these options will be extended further, resulting in a trend towards longer options. This behavior is consistent with the

| % a   | % o    | % 2    | % 3   | % 4    | % 5   | % 6   | ¯l     | % Rpt.   | % NRpt.   |        |
|-------|--------|--------|-------|--------|-------|-------|--------|----------|-----------|--------|
| ℓ3    | 62.38% | 37.62% | 6.23% | 31.39% | -     | -     | -      | 1.69     | 75.94%    | 24.06% |
| ℓ6    | 69.43% | 30.57% | 8.55% | 3.52%  | 1.86% | 0.99% | 15.64% | 2.03     | 74.12%    | 25.88% |

Table 2: Proportions of options with different lengths and options with repeated primitive actions for ℓ3 and ℓ6 in Atari games.

gradual increase in option lengths observed in gridworld as described in subsection 5.1, illustrating the capability of OptionZero to effectively discover and extend longer options when beneficial. Finally, we investigate the composition of primitive actions in options. From Table 2, approximately 75% of options consist of repeated primitive actions, similar to the findings in Sharma et al. (2016). For example, in *freeway*, a game where players control chickens across a traffic-filled highway from bottom to top, the most commonly used options by OptionZero are sequences of repeated Up actions (*U-U-U* in ℓ3 and *U-U-U-U-U-U* in ℓ6), guiding the chicken to advance upwards. In addition, OptionZero prefers repeated *Noop* actions, strategically pausing to let cars pass before proceeding. On the other hand, some games still require options composed of diverse combinations of primitive action. For example, in *crazy climber*, a game where players control the left and right side of the body to climb up to the top, OptionZero utilizes options consisting of non-repeated actions. These options often interleave Up and *Down* actions to coordinate the movements of the player's hands and feet, respectively. More interestingly, OptionZero also learns to acquire options involving combination skills under specific circumstances. In *hero*, only 4.60% of options involve non-repeated actions. Although the chance is small, such options are crucial during the game. For example, as depicted in Figure 5, the agent executes a series of strategically combined options, including landing from the top, planting a bomb at the corner to destroy a wall, swiftly moving away to avoid injury from the blast, and then skillfully times its movement to the right while firing after the wall is demolished. It is worth noting that there are a total of 24 primitive actions executed from Figure 5a to 5e, but only four options are executed in practice, showing that using option provides effective planning. In conclusion, our results demonstrate that OptionZero is capable of learning complex action sequences tailored to specific game dynamics, effectively discovering the required combinations whether the options are shorter, longer, or contain repeated actions. We have provided the top frequency of options used in each game in the Appendix D.5.

(a) *RF-RF-RF-RF-D-D* (b) *D-L-L-L-L-L* (c) *L-L-RF-RF-RF-RF* (d) *RF-RF-RF-RF-RF-RF* (e) *RF-RF-RF-RF-RF-RF*

## 5.4 Option Utilization In The Search

We further investigate the options used during planning. Table 3 lists the results for ℓ1, ℓ3, and ℓ6, including the proportions of search trees that consist of at least one option edge is expanded in MCTS ("% in Tree"), the proportions of simulations that at least one option has been selected during search ("% in Sim."), the average tree depth, the median tree depth, and the maximum tree depth. Detailed statistics for each game are provided in Appendix D.3. The results show that approximately 90% of search trees expand options, but only around 30% of search trees choose options during selection. Considering the nature of exploration in MCTS, it is reasonable that not all simulations will incorporate options. Surprisingly, there are still certain game states for which the search process does not use options at all. Especially in *hero*, From ℓ3 to ℓ6, the proportion of search trees utilizing options decreases from 74.43% to 54.39%, showing that there are numerous game states where options are not required. However, the performance remains consistent, suggesting that the planning could concentrate on applying options in certain states. Note that the less frequent use of options does not cause undesirable results; eventually, the search behaves similarly to that of MuZero.

| % in Tree   | % in Sim.   | Avg. tree depth   | Median tree depth   | Max tree depth   |        |
|-------------|-------------|-------------------|---------------------|------------------|--------|
| ℓ1          | 0.00%       | 0.00%             | 14.52               | 12.58            | 48.54  |
| ℓ3          | 91.43%      | 28.94%            | 20.74               | 18.23            | 121.46 |
| ℓ6          | 87.48%      | 22.28%            | 24.92               | 19.35            | 197.58 |

Finally, we compare the tree depths of the MCTS process with and without options. It is naturally considered that applying options provides a deeper tree, which helps in identifying longer future state sequences for better planning and avoiding pitfalls. From the statistics, the average search tree depths generally increase as the maximum option length increases, rising by 6.22 from ℓ1 to ℓ3 and by 10.4 from ℓ1 to ℓ6. Interestingly, there are counterexamples where the average depth decreases, such as *hero*. Although the average tree depth decreases in *hero* (22.30, 17.06, and 12.15 for ℓ1, ℓ3, and ℓ6), the performance is improved, as shown in Table 1. Furthermore, by comparing the median tree depth (19, 10, and 7) and maximum tree depth (50, 147, and 276) in *hero*, it can be derived that the model learns to perform deep searches depending on whatever the state requires. Ultimately, whether to conduct a deeper or shallower search tree is learned by OptionZero automatically. For the maximum tree depth, the baseline ℓ1 approaches the simulation budget of 50 nodes, meaning the search process may continuously exploit the same promising branch. When integrating with options, although the maximum depths increase, they do not always approach the simulation budgets of 150 and 300. The average numbers of maximum depths are 48.54, 121.46, and 192.27, equivalent to 97.08%, 80.97%, and 64.09% of the budgets, reflecting that the maximum depth is converging. This observation suggests that using an option length of 3 or 6 is sufficient in Atari games.

## 6 Discussion

This paper presents *OptionZero*, a method that integrates options into the well-known MuZero algorithm. OptionZero autonomously discovers options through self-play games and efficiently simulates environment transitions across multiple states with options during planning, which not only eliminates the requirement for obtaining options in advance but also reduces the overhead for examining consecutive states during planning. The empirical results on Atari games demonstrate a significant improvement of 131.58% in mean human-normalized scores, and the behavior analysis reveals that OptionZero effectively discovers options tailored to the specific challenges of each environment. In conclusion, our findings suggest that OptionZero not only discovers options without human knowledge but also maintains efficiency during planning. This makes OptionZero easily applicable to other applications, further extending the versatility of the MuZero algorithm. As OptionZero builds upon MuZero, it can be easily applied to various environments. For example, when applied to two-player games, OptionZero is expected to discover optimal strategies for both players at specific states. In strategic games such as StarCraft, our approach can learn skillfully combined options, enhancing further explainability and facilitating human learning, as illustrated in subsection 5.3. OptionZero can also be integrated with Sampled MuZero (Hubert et al., 2021) to support environments with complex action spaces, like robotic environments. Nevertheless, our experiments show that OptionZero does not improve performance in all games, especially in environments with numerous option types or visually complex observations, the dynamics network might struggle to learn well. Future work could explore integrating OptionZero with other dynamics models, such as S4 (Gu et al., 2021) or Dreamer (Hafner et al., 2020). Finally, the current design of the option networks requires a predefined maximum option length. Dynamically extending this maximum option length could be a promising direction for future work. We hope our approach and findings provide promising directions in planning with reinforcement learning for future researchers.

## Ethics Statement

We do not foresee any ethical issues in this research work. All data are generated by our programs.

## Reproducibility Statement

For reproducing the work, we have provided the details of the proposed algorithm in Section 4 and Appendix A, and the setup of training in Appendix B. The source code, scripts for processing behavior analysis, and trained models are available at https://rlg.iis.sinica.edu.tw/papers/optionzero.

## Acknowledgement

This research is partially supported by the National Science and Technology Council (NSTC) of the Republic of China (Taiwan) under Grant Number NSTC 113-2221-E-001-009-MY3, NSTC 1132634-F-A49-004, and NSTC 113-2221-E-A49-127.

## References

Ioannis Antonoglou, Julian Schrittwieser, Sherjil Ozair, Thomas K. Hubert, and David Silver.

Planning in Stochastic Environments with a Learned Model. In International Conference on Learning Representations, October 2021. URL https://openreview.net/forum?id= X6D9bAHhBQ1.

Pierre-Luc Bacon, Jean Harb, and Doina Precup. The Option-Critic Architecture. Proceedings of the AAAI Conference on Artificial Intelligence, 31(1), February 2017. URL https://ojs. aaai.org/index.php/AAAI/article/view/10916.

Cameron B. Browne, Edward Powley, Daniel Whitehouse, Simon M. Lucas, Peter I. Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis, and Simon Colton. A Survey of Monte Carlo Tree Search Methods. IEEE Transactions on Computational Intelligence and AI in Games, 4(1):1–43, March 2012. URL https://ieeexplore.ieee.org/ document/6145622.

Xinlei Chen and Kaiming He. Exploring Simple Siamese Representation Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15750–15758, 2021.

Remi Coulom. Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search. In ´ Computers and Games, Lecture Notes in Computer Science, pp. 72–83, Berlin, Heidelberg, 2007. Springer.

Konrad Czechowski, Tomasz Odrzygo´zd´ z, Marek Zbysi ´ nski, Micha ´ \l Zawalski, Krzysztof Olejnik, Yuhuai Wu, Łukasz Kucinski, and Piotr Miło ´ s. Subgoal Search For Complex Reasoning ´
Tasks. In *Advances in Neural Information Processing Systems*, volume 34, pp. 624–638. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper_files/ paper/2021/hash/05d8cccb5f47e5072f0a05b5f514941a-Abstract.html.

Ivo Danihelka, Arthur Guez, Julian Schrittwieser, and David Silver. Policy improvement by planning with Gumbel. In *International Conference on Learning Representations*, April 2022. URL
https://openreview.net/forum?id=bERaNdoegnO.

Maarten de Waard, Diederik M. Roijers, and Sander C.J. Bakkes. Monte Carlo Tree Search with options for general video game playing. In 2016 IEEE Conference on Computational Intelligence and Games (CIG), pp. 1–8, September 2016.

Ishan P. Durugkar, Clemens Rosenbaum, Stefan Dernbach, and Sridhar Mahadevan. Deep Reinforcement Learning With Macro-Actions, June 2016. URL http://arxiv.org/abs/ 1606.04615.

Thomas Gabor, Jan Peter, Thomy Phan, Christian Meyer, and Claudia Linnhoff-Popien. Subgoal-
Based Temporal Abstraction in Monte-Carlo Tree Search. In *Proceedings of the Twenty-Eighth* International Joint Conference on Artificial Intelligence, pp. 5562–5568, Macao, China, August 2019. International Joint Conferences on Artificial Intelligence Organization. URL https:// www.ijcai.org/proceedings/2019/772.

Albert Gu, Karan Goel, and Christopher Re. Efficiently Modeling Long Sequences with Structured State Spaces. In *International Conference on Learning Representations*, October 2021. URL https://openreview.net/forum?id=uYLFoz1vlAC.

Hung Guei, Yan-Ru Ju, Wei-Yu Chen, and Ti-Rong Wu. Interpreting the learned model in muzero planning. *arXiv preprint arXiv:2411.04580*, 2024.

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to Control: Learning Behaviors by Latent Imagination. In Eighth International Conference on Learning Representations, April 2020. URL https://iclr.cc/virtual_2020/poster_S1lOTC4tDS. html.

Jinke He, Thomas M Moerland, Joery A de Vries, and Frans A Oliehoek. What model does muzero learn? In Ulle Endriss, Francisco S. Melo, Kerstin Bach, Alberto Bugarin-Diz, Jose M. Alonso- Moral, Senen Barro, and Fredrik Heintz (eds.), ECAI 2024 - 27th European Conference on Artificial Intelligence, Including 13th Conference on Prestigious Applications of Intelligent Systems, PAIS 2024, Proceedings, Frontiers in Artificial Intelligence and Applications, pp. 1599–1606. IOS Press, 2024. doi: 10.3233/FAIA240666.

Matteo Hessel, Ivo Danihelka, Fabio Viola, Arthur Guez, Simon Schmitt, Laurent Sifre, Theophane Weber, David Silver, and Hado Van Hasselt. Muesli: Combining Improvements in Policy Optimization. In *Proceedings of the 38th International Conference on Machine Learning*, pp. 4214–4226. PMLR, July 2021.

Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Mohammadamin Barekatain, Simon Schmitt, and David Silver. Learning and Planning in Complex Action Spaces. In Proceedings of the 38th International Conference on Machine Learning, pp. 4476–4486. PMLR, July 2021. URL https://proceedings.mlr.press/v139/hubert21a.html.

Yuu Jinnai, David Abel, David Hershkowitz, Michael Littman, and George Konidaris. Finding Options that Minimize Planning Time. In *Proceedings of the 36th International Conference on* Machine Learning, pp. 3120–3129. PMLR, May 2019. URL https://proceedings.mlr. press/v97/jinnai19a.html.

Woojun Kim, Jeonghye Kim, and Youngchul Sung. LESSON: Learning to Integrate Exploration Strategies for Reinforcement Learning via an Option Framework. In Proceedings of the 40th International Conference on Machine Learning, pp. 16619–16638. PMLR, July 2023.

Levente Kocsis and Csaba Szepesvari. Bandit Based Monte-Carlo Planning. In ´ European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases, volume 2006, pp. 282–293, September 2006.

Kalle Kujanpa¨a, Joni Pajarinen, and Alexander Ilin. Hierarchical Imitation Learning with Vector ¨
Quantized Models. In *Proceedings of the 40th International Conference on Machine Learning*,
pp. 17896–17919. PMLR, July 2023. URL https://proceedings.mlr.press/v202/ kujanpaa23a.html.

Kalle Kujanpa¨a, Joni Pajarinen, and Alexander Ilin. Hybrid Search for Efficient Planning with ¨
Completeness Guarantees. *Advances in Neural Information Processing Systems*, 36, February 2024. URL https://papers.nips.cc/paper_files/paper/2023/hash/ 46d26daeb05fbbcfe5f3d8f7ca756e16-Abstract-Conference.html.

Aravind Lakshminarayanan, Sahil Sharma, and Balaraman Ravindran. Dynamic Action Repetition for Deep Reinforcement Learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 31(1), February 2017.

Amol Mandhane, Anton Zhernov, Maribeth Rauh, Chenjie Gu, Miaosen Wang, Flora Xue, Wendy Shang, Derek Pang, Rene Claus, Ching-Han Chiang, Cheng Chen, Jingning Han, Angie Chen, Daniel J. Mankowitz, Jackson Broshear, Julian Schrittwieser, Thomas Hubert, Oriol Vinyals, and Timothy Mann. MuZero with Self-competition for Rate Control in VP9 Video Compression, February 2022. URL http://arxiv.org/abs/2202.06626.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G.

Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement learning. *Nature*, 518(7540):529–533, February 2015. URL https://www.nature.com/ articles/nature14236.

Matthew Riemer, Ignacio Cases, Clemens Rosenbaum, Miao Liu, and Gerald Tesauro. On the Role of Weight Sharing During Deep Option Learning. Proceedings of the AAAI Conference on Artificial Intelligence, 34(04):5519–5526, April 2020.

Christopher D. Rosin. Multi-armed bandits with episode context. Annals of Mathematics and Artificial Intelligence, 61(3):203–230, March 2011. URL http://link.springer.com/10. 1007/s10472-011-9258-6.

Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, and David Silver. Mastering Atari, Go, chess and shogi by planning with a learned model. Nature, 588(7839):604–609, December 2020. URL https://www.nature.com/articles/ s41586-020-03051-4.

Sahil Sharma, Aravind S. Lakshminarayanan, and Balaraman Ravindran. Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning. In *International Conference on* Learning Representations, November 2016. URL https://openreview.net/forum? id=B1GOWV5eg.

David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel, and Demis Hassabis. Mastering the game of Go without human knowledge. *Nature*, 550(7676):354–359, October 2017. URL https://www.nature.com/articles/nature24270.

David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, and Demis Hassabis. A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419):1140–1144, December 2018. URL https://www.science.org/doi/10.1126/science.aar6404.

Richard S. Sutton, Doina Precup, and Satinder Singh. Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. *Artificial Intelligence*, 112(1):181–211, August 1999. URL https://www.sciencedirect.com/science/article/pii/ S0004370299000521.

Vivek Veeriah, Tom Zahavy, Matteo Hessel, Zhongwen Xu, Junhyuk Oh, Iurii Kemaev, Hado P van Hasselt, David Silver, and Satinder Singh. Discovery of Options via Meta-Learned Subgoals. In *Advances in Neural Information Processing Systems*, volume 34, pp. 29861–29873. Curran Associates, Inc., 2021.

Alexander Vezhnevets, Volodymyr Mnih, Simon Osindero, Alex Graves, Oriol Vinyals, John Agapiou, and koray kavukcuoglu. Strategic Attentive Writer for Learning Macro-Actions. In *Advances in Neural Information Processing Systems*, volume 29. Curran Associates, Inc., 2016. URL https://papers.nips.cc/paper_files/paper/2016/hash/ c4492cbe90fbdbf88a5aec486aa81ed5-Abstract.html.

Joery De Vries, Ken Voskuil, Thomas M. Moerland, and Aske Plaat. Visualizing MuZero Models.

In *ICML 2021 Workshop on Unsupervised Reinforcement Learning*, July 2021. URL https: //openreview.net/forum?id=UGkzkpAq_8.

Pengming Wang, Mikita Sazanovich, Berkin Ilbeyi, Phitchaya Mangpo Phothilimthana, Manish Purohit, Han Yang Tay, Ngan V ˆ u, Miaosen Wang, Cosmin Paduraru, Edouard Leurent, Anton ˜ Zhernov, Po-Sen Huang, Julian Schrittwieser, Thomas Hubert, Robert Tung, Paula Kurylowicz, Kieran Milan, Oriol Vinyals, and Daniel J. Mankowitz. Optimizing Memory Mapping Using Deep Reinforcement Learning, October 2023. URL http://arxiv.org/abs/2305.07440.

Ti-Rong Wu, Hung Guei, Pei-Chiun Peng, Po-Wei Huang, Ting Han Wei, Chung-Chin Shih, and Yun-Jui Tsai. Minizero: Comparative analysis of alphazero and muzero on go, othello, and atari games. *IEEE Transactions on Games*, 17(1):125–137, 2025. doi: 10.1109/TG.2024.3394900.

Weirui Ye, Shaohuai Liu, Thanard Kurutach, Pieter Abbeel, and Yang Gao. Mastering Atari Games with Limited Data. In *Advances in Neural Information Processing Systems*, volume 34, pp.

25476–25488. Curran Associates, Inc., 2021. URL https://proceedings.neurips.

cc/paper/2021/hash/d5eca8dc3820cad9fe56a3bafda65ca1-Abstract.

html.

Kenny Young and Richard S. Sutton. Iterative Option Discovery for Planning, by Planning, December 2023. URL http://arxiv.org/abs/2310.01569.

## A Implementation Details

In this section, we detail our OptionZero implementation, which is built upon a publicly available MuZero framework (Wu et al., 2025).

## A.1 Mcts Details

The MCTS implementation mainly follows that introduced in Section 4.2, with minor details described below. Dirichlet noise To encourage exploration, in MuZero, Dirichlet noise is applied to the root node. Similarly, in OptionZero, since option can also be executed in the environment, we apply Dirichlet noise to both primitive selection and option selection at the root node. Default estimated Q value For primitive selection, we follow the default estimated Q value for Atari games in the framework (Wu et al., 2025) that enhances exploration:

$$\hat{Q}(s)=\begin{cases}\frac{Q_{\Sigma}(s)}{N_{\Sigma}(s)}&N_{\Sigma}(s)>0\\ 1&N_{\Sigma}(s)=0,\end{cases}$$
$$(8)$$
$$(7)$$

where NΣ(s) = Pb 1N(s,b)>0, QΣ(s) = Pb 1N(s,b)>0Q(s, b), and 1N(s,b)>0 is the characteristic function that only considers primitive child nodes with non-zero visit counts. For option selection, since the contributions of option child node are included in the statistics of its corresponding predecessor primitive child node, we use a default estimated Q value that incorporates a virtually losing outcome:

$$\hat{Q}(s)=\frac{Q_{(}s,a)\times N(s,a)}{N(s,a)+1},$$

where N(*s, a*) is the visit counts of the primitive child node, and Q(*s, a*) is the mean value of the primitive child node.

## A.2 Mcts Complexity

The complexity of the modified MCTS remains the same as the original, with additional minor computational costs in introducing a new network head to predict and use the dominant option. Specifically, in the selection phase, the only added step is comparing the PUCT scores of option child nodes and primitive child nodes, as in equation 4. In the expansion phase, the option policy Ω is evaluated along with policy p and value v. Since most network weights of the option policy head are shared with the rest of the network, the impact on runtime is negligible. While more nodes are initially expanded, each simulation evaluates only one node at a time. In the backup phase, as the statistics of option edges can be easily derived from primitive edges, only the statistics of primitive edges are maintained in practice, eliminating the additional cost of updating all possible option edges.

## A.3 Gridworld Environment

The implementation is also built upon the same framework (Wu et al., 2025), with a custom Grid- World environment added. The reward of the environment is defined as follows: the initial total reward is 200 points, and for each action or option taken, one point is deducted from the reward.

## A.4 Network Architecture

The network architecture follows a structure similar to MuZero. As discussed in Section 4, the option network is incorporated into the prediction network. Specifically, besides the policy head, we add additional L − 1 option heads for predicting Ω = {ω2, ω3*, ..., ω*L}, initialized to predict the stop. Note that there is no need for extra prediction of ω1, since we can directly get the first action of the dominant option from policy head by choosing a
∗1 = arg maxa p(a). Additionally, the dynamics network is modified to simulate the environment transitions of executing both primitive actions and options. By extending the original action input, the action sequence input to the dynamics network is encoded into a fixed number of L planes for supporting options with different lengths, with the lth plane corresponding to the lth move inside an option. Note that when *l < L*, the subsequent planes are set to zero, representing no moves. Atari games We additionally adopt the state consistency (Ye et al., 2021). Therefore, the SimSiam (Chen & He, 2021) architecture is included to calculate the consistency loss. GridWorld The network architecture generally follows the architecture tailored for Atari games. However, in the design of the representation network, we removed the down-sampling mechanism, adopting a setup similar to MuZero for board games as in Wu et al. (2025).

## B Training Optionzero

In this section, we describe the details for training OptionZero models used in the experiments. The experiments are conducted on machines with 24 CPU cores and four NVIDIA GTX 1080 Ti GPUs. For the training configurations, we generally follow those in MuZero, where the hyperparameters are listed in Table 4.

| Parameter                      | Atari     | Grid world   |
|--------------------------------|-----------|--------------|
| Optimizer                      | SGD       |              |
| Optimizer: learning rate       | 0.1       |              |
| Optimizer: momentum            | 0.9       |              |
| Optimizer: weight decay        | 0.0001    |              |
| Discount factor                | 0.997     |              |
| Priority exponent (α)          | 1         |              |
| Priority correction (β)        | 0.4       |              |
| Bootstrap step (n-step return) | 5         |              |
| MCTS simulation                | 50        |              |
| Softmax temperature            | 1         |              |
| Frames skip                    | 4         | -            |
| Frames stacked                 | 4         | -            |
| Iteration                      | 300       | 400          |
| Training steps                 | 60k       | 80k          |
| Batch size                     | 512       | 1024         |
| # Blocks                       | 2         | 1            |
| Replay buffer size             | 1M frames | 8k games     |
| Max frames per episode         | 108k      | -            |
| Dirichlet noise ratio          | 0.25      | 0.3          |

Atari games For Atari games, each setting is trained for 3 runs on each game, with each model taking approximately 22 hours to complete. Since we introduce an additional head to predict option, the training time slightly increases as the max option length increases. For ℓ1, the training time is approximately 21.89 hours. For ℓ3 and ℓ6, the training times increase to around 22.28 hours and 22.95 hours, representing increases of 1.8% and 4.8%, respectively. The performance is measured based on the average score of the latest 100 completed games in each run during the training (Hessel et al., 2021). The training curves are shown in Figure 6. GridWorld In this toy example, we aim to clearly show that the length of the learned options can be extended as the training time increases. For training, we use a maximum option length L = 9, fixing the goal position and selecting random starting points. As for the evaluation, we fix the starting point and the goal as shown in Figure 4. The evaluation also uses 50 MCTS simulations, and the original softmax function is replaced with max selection.

0k 10k 20k 30k 40k 50k 60k 0.0k 0.6k 1.2k 1.8k 2.4k 3.0k 3.6k 4.2k 4.8k alien 0k 10k 20k 30k 40k 50k 60k 0.0k 0.1k 0.3k 0.5k 0.6k 0.8k 0.9k amidar 0k 10k 20k 30k 40k 50k 60k 0k 3k 6k 9k 12k 15k 18k 21k 24k assault 0k 10k 20k 30k 40k 50k 60k 0k 30k 60k 90k 120k 150k 180k 210k 240k 270k asterix 0k 10k 20k 30k 40k 50k 60k 0.0k 0.1k 0.3k 0.5k 0.6k 0.8k 0.9k 1.1k 1.2k bank heist 0k 10k 20k 30k 40k 50k 60k 0k 8k 16k 24k 32k 40k 48k 56k 64k 72k battle zone 0k 10k 20k 30k 40k 50k 60k
 -30
 -15 0 15 30 45 60 75 90 105 boxing 0k 10k 20k 30k 40k 50k 60k 0 80 160 240 320 400 480 breakout 0k 10k 20k 30k 40k 50k 60k 0k 15k 30k 45k 60k 75k 90k chopper command 0k 10k 20k 30k 40k 50k 60k 0k 20k 40k 60k 80k 100k 120k 140k crazy climber 0k 10k 20k 30k 40k 50k 60k 0k 15k 30k 45k 60k 75k 90k 105k 120k demon attack 0k 10k 20k 30k 40k 50k 60k
 -5 0 5 10 15 20 25 30 freeway 0k 10k 20k 30k 40k 50k 60k 0k 1k 2k 3k 4k 5k 6k 7k 8k frostbite 0k 10k 20k 30k 40k 50k 60k 0k 10k 20k 30k 40k 50k 60k 70k 80k 90k gopher 0k 10k 20k 30k 40k 50k 60k 0.0k 2.5k 5.0k 7.5k 10.0k 12.5k 15.0k 17.5k 20.0k hero 0k 10k 20k 30k 40k 50k 60k 0.0k 2.5k 5.0k 7.5k 10.0k 12.5k 15.0k 17.5k 20.0k jamesbond 0k 10k 20k 30k 40k 50k 60k 0k 2k 4k 6k 8k 10k 12k 14k kangaroo 0k 10k 20k 30k 40k 50k 60k 0k 4k 8k 12k 16k 20k 24k 28k krull 0k 10k 20k 30k 40k 50k 60k 0k 10k 20k 30k 40k 50k 60k 70k 80k kung fu master 0k 10k 20k 30k 40k 50k 60k 0.0k 0.8k 1.6k 2.4k 3.2k 4.0k 4.8k 5.6k ms pacman 0k 10k 20k 30k 40k 50k 60k

 -24 -18

 -12
 -6 0 6 12 18 24 pong 0k 10k 20k 30k 40k 50k 60k
 -1.5k 0.0k 1.5k 3.0k 4.5k 6.0k 7.5k 9.0k 10.5k private eye 0k 10k 20k 30k 40k 50k 60k 0k 6k 12k 18k 24k 30k 36k 42k 48k 54k qbert 0k 10k 20k 30k 40k 50k 60k 0k 5k 10k 15k 20k 25k 30k 35k 40k road runner 0k 10k 20k 30k 40k 50k 60k 0k 1k 2k 3k 4k 5k 6k 7k seaquest 0k 10k 20k 30k 40k 50k 60k 0k 50k 100k 150k 200k 250k 300k 350k up n down 1 3 6 n- 3

## C Ablation Study For Optionzero

For the ablation study, we train OptionZero with a maximum option length L = 3, but disable the execution of options in the environment, using them solely for MCTS planning, denoted as nℓ3. Since all composite actions are primitive actions, we define each oi as {ai+1}, where ai+1 = arg maxa pi(a) and train the option network according to the policy network. According to the results shown in Table 5, n-ℓ3 achieves a mean human-normalized score of 1008.15%, which is 85.43% higher than the baseline ℓ1, indicating that OptionZero still enhances MCTS planning with options without executing them. Notably, in games that require precise step-by-step predictions, such as *gopher*, n-ℓ3 outperforms ℓ3, indicating that planning for every step remains crucial for certain games. However, in games that benefit from bypassing unimportant frames, such as *seaquest*, the performance of n-ℓ3 is only comparable to baseline.

| OptionZero        |          |           |            |            |            |
|-------------------|----------|-----------|------------|------------|------------|
| Game              | Random   | Human     | ℓ1         | ℓ3         | n-ℓ3       |
| alien             | 128.30   | 6,371.30  | 2,437.30   | 2,900.07   | 3,523.20   |
| amidar            | 11.79    | 1,540.43  | 780.26     | 820.77     | 848.97     |
| assault           | 166.95   | 628.89    | 18,389.88  | 19,302.04  | 19,378.79  |
| asterix           | 164.50   | 7,536.00  | 177,128.50 | 188,999.00 | 202,183.33 |
| bank heist        | 21.70    | 644.50    | 1,097.63   | 950.13     | 1,081.10   |
| battle zone       | 3,560.00 | 33,030.00 | 53,326.67  | 53,583.33  | 65,660.00  |
| boxing            | -1.46    | 9.61      | 97.71      | 95.09      | 94.58      |
| breakout          | 1.77     | 27.86     | 371.30     | 375.58     | 427.18     |
| chopper command   | 644.00   | 8,930.00  | 43,951.67  | 60,181.67  | 79,340.33  |
| crazy climber     | 9,337.00 | 32,667.00 | 110,634.00 | 114,390.00 | 122,865.67 |
| demon attack      | 208.25   | 3,442.85  | 103,823.17 | 117,270.57 | 104,351.00 |
| freeway           | 0.17     | 25.61     | 29.46      | 31.06      | 30.93      |
| frostbite         | 90.80    | 4,202.80  | 3,183.40   | 3,641.10   | 3,923.63   |
| gopher            | 250.00   | 2,311.00  | 70,985.27  | 68,240.60  | 73,338.67  |
| hero              | 1,580.30 | 25,839.40 | 13,568.20  | 19,073.18  | 14,181.65  |
| jamesbond         | 33.50    | 368.50    | 8,155.50   | 13,276.67  | 7,172.17   |
| kangaroo          | 100.00   | 2,739.00  | 8,929.67   | 12,294.00  | 11,175.33  |
| krull             | 1,151.90 | 2,109.10  | 10,255.37  | 10,098.83  | 17,420.13  |
| kung fu master    | 304.00   | 20,786.80 | 66,304.67  | 68,528.33  | 67,735.67  |
| ms pacman         | 197.80   | 15,375.05 | 3,695.60   | 4,952.37   | 4,762.23   |
| pong              | -17.95   | 15.46     | 19.37      | 15.49      | 20.01      |
| private eye       | 662.78   | 64,169.07 | 116.83     | 90.76      | 94.71      |
| qbert             | 159.38   | 12,085.00 | 17,155.50  | 30,748.42  | 22,321.75  |
| road runner       | 200.00   | 6,878.00  | 26,971.33  | 32,786.67  | 23,784.67  |
| seaquest          | 215.50   | 40,425.80 | 3,592.53   | 5,606.63   | 3,378.60   |
| up n down         | 707.20   | 9,896.10  | 217,021.60 | 280,832.43 | 238,409.40 |
| Normalized Mean   | 0.00     | 100.00 %  | 922.72 %   | 1054.30 %  | 1008.15 %  |
| Normalized Median | 0.00     | 100.00 %  | 328.40 %   | 391.69 %   | 341.19 %   |

## D In-Depth Behavior Analysis

In this experiment section, we conduct detailed analysis for ℓ3 and ℓ6 in 26 Atari games.

## D.1 Options Applied In Games

We present the statistics in all 26 Atari games conducted for the behavior analysis in Section 5.3. Specifically, we provide the numbers of options types, option usages, proportions of options with repeated actions, and the average option lengths for ℓ3 and ℓ6 in Table 6 and Table 7, respectively.

The columns "\# {a}" and "\# {o}" show the numbers of available primitive actions and the numbers of the options recorded during evaluation, columns "% a" and "% o" show the proportions of actions and options applied during the game, columns "% Rpt." and "% NRpt." show the proportions of options with repeated primitive actions and options with more than one action type, and column "¯l" shows the average options length (including primitive action). We also provide the proportions of options with different lengths for both ℓ3 and ℓ6 in Table 8.

Game # {a} # {o} % a % o % Rpt. % NRpt. ¯l

alien 18 185 67.66% 32.34% 94.54% 5.46% 1.61 amidar 10 187 65.51% 34.49% 98.43% 1.57% 1.65 assault 7 139 78.84% 21.16% 57.20% 42.80% 1.30 asterix 9 163 92.82% 7.18% 90.74% 9.26% 1.10 bank heist 18 138 41.93% 58.07% 7.60% 92.40% 2.05 battle zone 18 217 88.89% 11.11% 95.33% 4.67% 1.18 boxing 18 240 39.53% 60.47% 50.54% 49.46% 2.12 breakout 4 64 79.00% 21.00% 84.03% 15.97% 1.32 chopper command 18 242 76.08% 23.92% 92.28% 7.72% 1.41 crazy climber 9 158 50.93% 49.07% 25.09% 74.91% 1.94 demon attack 6 153 76.70% 23.30% 88.74% 11.26% 1.38 freeway 3 30 39.20% 60.80% 95.04% 4.96% 2.19 frostbite 18 273 58.10% 41.90% 94.14% 5.86% 1.81 gopher 8 325 51.44% 48.56% 68.60% 31.40% 1.88 hero 18 346 81.72% 18.28% 95.40% 4.60% 1.34 jamesbond 18 376 51.34% 48.66% 67.24% 32.76% 1.90 kangaroo 18 230 29.84% 70.16% 70.54% 29.46% 2.36 krull 18 182 67.72% 32.28% 54.58% 45.42% 1.54 kung fu master 14 536 38.48% 61.52% 70.51% 29.49% 2.15 ms pacman 9 181 65.01% 34.99% 94.70% 5.30% 1.67 pong 6 159 24.57% 75.43% 76.98% 23.02% 2.47 private eye 18 233 93.60% 6.40% 78.74% 21.26% 1.08 qbert 6 105 50.04% 49.96% 97.84% 2.16% 1.97 road runner 18 144 85.95% 14.05% 65.92% 34.08% 1.23 seaquest 18 340 74.88% 25.12% 70.92% 29.08% 1.38 up n down 6 116 52.06% 47.94% 88.89% 11.11% 1.91 Average - - 62.38% 37.62% 75.94% 24.06% 1.69

| Game            | # {a}   | # {o}   | % a    | % o    | % Rpt.   | % NRpt.   | ¯l   |
|-----------------|---------|---------|--------|--------|----------|-----------|------|
| alien           | 18      | 411     | 69.28% | 30.72% | 94.13%   | 5.87%     | 2.30 |
| amidar          | 10      | 318     | 67.62% | 32.38% | 97.19%   | 2.81%     | 2.22 |
| assault         | 7       | 367     | 78.71% | 21.29% | 58.78%   | 41.22%    | 1.35 |
| asterix         | 9       | 199     | 92.68% | 7.32%  | 86.80%   | 13.20%    | 1.10 |
| bank heist      | 18      | 588     | 48.17% | 51.83% | 11.88%   | 88.12%    | 2.28 |
| battle zone     | 18      | 513     | 91.54% | 8.46%  | 95.21%   | 4.79%     | 1.24 |
| boxing          | 18      | 568     | 48.46% | 51.54% | 40.29%   | 59.71%    | 2.77 |
| breakout        | 4       | 132     | 81.21% | 18.79% | 85.92%   | 14.08%    | 1.28 |
| chopper command | 18      | 351     | 82.75% | 17.25% | 87.56%   | 12.44%    | 1.40 |
| crazy climber   | 9       | 724     | 60.69% | 39.31% | 26.15%   | 73.85%    | 2.58 |
| demon attack    | 6       | 301     | 82.64% | 17.36% | 88.16%   | 11.84%    | 1.34 |
| freeway         | 3       | 118     | 45.38% | 54.62% | 90.66%   | 9.34%     | 3.45 |
| frostbite       | 18      | 708     | 66.80% | 33.20% | 86.23%   | 13.77%    | 2.45 |
| gopher          | 8       | 692     | 56.12% | 43.88% | 64.71%   | 35.29%    | 2.01 |
| hero            | 18      | 576     | 89.58% | 10.42% | 85.05%   | 14.95%    | 1.39 |
| jamesbond       | 18      | 735     | 66.08% | 33.92% | 86.88%   | 13.12%    | 2.30 |
| kangaroo        | 18      | 718     | 40.00% | 60.00% | 64.44%   | 35.56%    | 3.43 |
| krull           | 18      | 679     | 60.31% | 39.69% | 45.89%   | 54.11%    | 2.07 |
| kung fu master  | 14      | 1386    | 53.09% | 46.91% | 53.40%   | 46.60%    | 2.53 |
| ms pacman       | 9       | 219     | 77.13% | 22.87% | 94.95%   | 5.05%     | 1.77 |
| pong            | 6       | 741     | 36.82% | 63.18% | 61.09%   | 38.91%    | 3.71 |
| private eye     | 18      | 488     | 97.04% | 2.96%  | 77.05%   | 22.95%    | 1.06 |
| qbert           | 6       | 450     | 62.79% | 37.21% | 92.84%   | 7.16%     | 2.52 |
| road runner     | 18      | 225     | 96.19% | 3.81%  | 90.55%   | 9.45%     | 1.10 |
| seaquest        | 18      | 621     | 82.41% | 17.59% | 76.38%   | 23.62%    | 1.38 |
| up n down       | 6       | 226     | 71.65% | 28.35% | 84.88%   | 15.12%    | 1.84 |
| Average         | -       | -       | 69.43% | 30.57% | 74.12%   | 25.88%    | 2.03 |

Table 8: Proportions of options with different lengths for ℓ3 and ℓ6 in 26 Atari games.

| Game            | ℓ3     | ℓ6     |        |        |        |        |       |       |        |
|-----------------|--------|--------|--------|--------|--------|--------|-------|-------|--------|
| % 1             | % 2    | % 3    | % 1    | % 2    | % 3    | % 4    | % 5   | % 6   |        |
| alien           | 67.66% | 4.03%  | 28.31% | 69.28% | 4.24%  | 1.47%  | 0.97% | 0.60% | 23.44% |
| amidar          | 65.51% | 3.52%  | 30.97% | 67.62% | 7.09%  | 2.36%  | 1.71% | 0.60% | 20.61% |
| assault         | 78.84% | 11.85% | 9.30%  | 78.71% | 11.74% | 6.61%  | 1.90% | 0.63% | 0.42%  |
| asterix         | 92.82% | 4.02%  | 3.17%  | 92.68% | 5.99%  | 0.62%  | 0.29% | 0.13% | 0.28%  |
| bank heist      | 41.93% | 11.46% | 46.62% | 48.17% | 20.37% | 10.52% | 6.66% | 4.31% | 9.96%  |
| battle zone     | 88.89% | 4.34%  | 6.77%  | 91.54% | 3.16%  | 1.33%  | 0.53% | 0.22% | 3.23%  |
| boxing          | 39.53% | 8.56%  | 51.91% | 48.46% | 13.74% | 6.25%  | 2.73% | 1.77% | 27.05% |
| breakout        | 79.00% | 9.96%  | 11.04% | 81.21% | 13.30% | 3.81%  | 0.77% | 0.29% | 0.63%  |
| chopper command | 76.08% | 7.09%  | 16.83% | 82.75% | 9.00%  | 2.16%  | 1.08% | 1.32% | 3.70%  |
| crazy climber   | 50.93% | 4.63%  | 44.44% | 60.69% | 6.37%  | 3.30%  | 1.44% | 0.71% | 27.49% |
| demon attack    | 76.70% | 8.23%  | 15.07% | 82.64% | 9.64%  | 3.46%  | 1.70% | 0.69% | 1.87%  |
| freeway         | 39.20% | 2.61%  | 58.19% | 45.38% | 4.93%  | 1.91%  | 0.98% | 0.50% | 46.29% |
| frostbite       | 58.10% | 2.83%  | 39.08% | 66.80% | 3.29%  | 1.67%  | 0.92% | 0.63% | 26.70% |
| gopher          | 51.44% | 9.24%  | 39.32% | 56.12% | 22.73% | 6.51%  | 3.45% | 1.56% | 9.64%  |
| hero            | 81.72% | 2.46%  | 15.82% | 89.58% | 2.57%  | 0.66%  | 0.24% | 0.17% | 6.78%  |
| jamesbond       | 51.34% | 7.10%  | 41.55% | 66.08% | 6.68%  | 3.14%  | 1.27% | 0.99% | 21.83% |
| kangaroo        | 29.84% | 4.47%  | 65.68% | 40.00% | 8.47%  | 4.77%  | 3.47% | 1.51% | 41.78% |
| krull           | 67.72% | 10.98% | 21.30% | 60.31% | 12.44% | 9.58%  | 5.16% | 2.45% | 10.06% |
| kung fu master  | 38.48% | 7.74%  | 53.77% | 53.09% | 14.46% | 5.03%  | 3.53% | 1.60% | 22.28% |
| ms pacman       | 65.01% | 2.93%  | 32.07% | 77.13% | 7.24%  | 2.22%  | 0.86% | 0.25% | 12.31% |
| pong            | 24.57% | 3.74%  | 71.69% | 36.82% | 6.98%  | 3.08%  | 3.04% | 1.84% | 48.24% |
| private eye     | 93.60% | 4.67%  | 1.73%  | 97.04% | 1.84%  | 0.45%  | 0.11% | 0.07% | 0.49%  |
| qbert           | 50.04% | 2.62%  | 47.34% | 62.79% | 4.75%  | 3.52%  | 1.62% | 0.82% | 26.50% |
| road runner     | 85.95% | 5.41%  | 8.64%  | 96.19% | 1.69%  | 0.49%  | 0.27% | 0.13% | 1.23%  |
| seaquest        | 74.88% | 12.27% | 12.85% | 82.41% | 9.62%  | 2.79%  | 1.07% | 0.66% | 3.45%  |
| up n down       | 52.06% | 5.16%  | 42.77% | 71.65% | 9.96%  | 3.85%  | 2.70% | 1.33% | 10.50% |
| Average         | 62.38% | 6.23%  | 31.39% | 69.43% | 8.55%  | 3.52%  | 1.86% | 0.99% | 15.64% |