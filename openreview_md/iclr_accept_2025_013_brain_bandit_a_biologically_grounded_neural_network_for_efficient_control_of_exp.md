# Brain Bandit: A Biologically Grounded Neu- Ral Network For Efficient Control Of Explo- Ration

Chen Jiang1∗
, Jiahui An2,1, Yating Liu3,2,1, **Ni ji**2,1†
1Chinese Institute for Brain Research, Beijing 2Chinese Academy of Medical Sciences & Peking Union Medical College 3China Agricultural University chen.jiang3@mail.mcgill.ca, {anjiahui, liuyating, niji}@cibr.ac.cn

## Abstract

How to balance between exploration and exploitation in an uncertain environment is a central challenge in reinforcement learning. In contrast, humans and animals have demonstrated superior exploration efficiency in novel environments. To understand how the brain's neural network controls exploration under uncertainty, we analyzed the dynamical systems model of a biological neural network that controls explore-exploit decisions during foraging. Mathematically, this model (named the Brain Bandit Net, or BBN) is a special type of stochastic continuous Hopfield network. We show through theory and simulation that BBN can perform posterior sampling of action values with a tunable bias towards or against uncertain options. We then demonstrate that, in multi-armed bandit (MAB) tasks, BBN can generate probabilistic choice behavior with a flexible uncertainty bias resembling human and animal choice patterns. In addition to its high efficiency in MAB tasks, BBN can also be embedded with reinforcement learning algorithms to accelerate learning in MDP tasks. Altogether, our findings reveal the theoretical foundation for efficient exploration in biological neural networks and propose a general, brain-inspired algorithm for enhancing exploration in RL. The code is available at https://github.com/Chen-Ginger/BrainBandit

## 1 Introduction

The explore-exploit (E-E) dilemma, originally described in the context of animal foraging (Stephens & Krebs, 1986; Charnov, 1976), has become an important problem across many fields including psychology, neuroscience and reinforcement learning (RL)(Addicott et al., 2017). Despite the development of numerous algorithms, sample-efficient exploration in RL remains difficult for complex, sparse-reward tasks (Sutton & Barto, 2018). Meanwhile, studies in humans and animals have revealed a diverse array of exploration strategies (Wilson et al., 2021; Schulz & Gershman, 2019). In addition, excitingly, recent research has begun to reveal the biological neural networks that give rise to the rich and flexible exploration behaviors (Costa et al., 2019; Tomov et al., 2020; Hogeveen et al., 2022; Costa & Averbeck, 2020). Based on recent findings in the biological neural network that controls exploration, we built the Brain Bandit Network (BBN), a stochastic Hopfield network for controlling exploratory action selection under input uncertainty. We show theoretically that the BBN model can perform Bayesian posterior sampling while implementing a tunable bias that ranges from optimistic, neutral, and conservative in the face of uncertainty. Our main contributions are four-fold:
1. We propose a biologically grounded, scalable network model for solving the E-E dilemma. 2. We analytically show that BBN implements a hybrid between Bayesian posterior sampling and uncertainty-directed exploration.

∗Current address: McGill University. †Corresponding author.

3. We show that BBN can closely approximate human and animal behavior in bandit tasks under a variety of conditions.

4. We show that BBN can drive highly efficient exploration in bandit and MDP tasks, promising further application to more complex RL problems.

## 2 Background And Related Work 2.1 The Exploration Problem In Reinforcement Learning

The domain of efficient exploration in reinforcement learning focuses on balancing immediate rewards (exploitation) and information gathering for future rewards (exploration). A classic example is the Multi-Armed Bandit (MAB) problem, introduced by (Robbins, 1952) in 1952 and widely used to model this tradeoff (Lai & Robbins, 1985; Berry & Fristedt, 1985; Agrawal, 1995; Auer et al., 1995; Sutton & Barto, 1999). Conventional methods inject noise into action selection (Sutton & Barto, 1999), but these dithering algorithms can be inefficient. Alternative methods like Upper Confidence Bound (UCB) employ optimism in the face of uncertainty (OFU) by biasing for uncertain choices (Lai & Robbins, 1985; Agrawal, 1995; Auer et al., 1995). Thompson sampling (Thompson, 1933) makes decisions based on posterior samples rather than optimistic estimates. Optimistic Thompson Sampling (O-TS), combining UCB and Thompson sampling, reshapes the posterior distribution optimistically and exhibits strong empirical and theoretical performance (Chapelle & Li, 2011; May et al., 2012). More recent methods leverage deep networks to learn the exploration bonus (Zhou et al., 2020; Ban et al., 2022) or the posterior variance (Zhang et al., 2021).

## 2.2 Biological Solutions To The Explore-Exploit Dilemma

Early work on the explore-exploit tradeoff, rooted in Optimal Foraging Theory and the Marginal Value Theorem (Stephens & Krebs, 1986; Charnov, 1976), suggests that animals achieve nearoptimal balance between exploiting known resources and exploring uncertain options. Cognitive scientists have used bandit tasks to study this tradeoff in humans and animals (Addicott et al., 2017; Cohen et al., 2007; Wang et al., 2023; Beron et al., 2022). Two main strategies emerge: random exploration, involving stochastic action choices, and directed exploration, leveraging uncertainty to guide actions (Wilson et al., 2021; Schulz & Gershman, 2019). Humans and animals often combine these strategies flexibly, adjusting based on task horizon, option novelty, developmental stage, and mental state (Gershman, 2018; Bartumeus et al., 2016; Wilson et al., 2014; Cockburn et al., 2022; Mizell et al., 2024; Schulz et al., 2019; Addicott et al., 2017; Fan et al., 2023; Waltz et al., 2020). Additionally, they exhibit persistent exploration, repeating previous choices regardless of value (Beron et al., 2022; Laurie et al., 2024). These strategies resemble algorithms like Thompson sampling and Optimism in the Face of Uncertainty (OFU), but with key differences (Wilson et al., 2021). To understand the brain's solution to the E-E dilemma, neuroscientists have identified neurobiological mechanisms that control explore-exploit decisions (Daw et al., 2006; Costa et al., 2019; Hogeveen et al., 2022). Recent studies in *C. elegans* (Flavell et al., 2013; Ji et al., 2021) have revealed a compact recurrent network governing the transitions between behavioral states analogous to exploration and exploitation (Fig. 1). This minimal network provides a unique opportunity to explore the algorithmic principles the brain uses to solve the E-E problem.

## 3 Model 3.1 The Brain-Inspired Bandit Network (Bbn) Is A Stochastic Continuous Hopfield Network

To model the biological neural network that controls E-E decisions during foraging (Fig. 8 (Ji et al., 2021)), we define a set of N neurons whose temporal dynamics are described by the following stochastic differential equations (or Langevin equations):

$$\tau_{i}{\frac{d x_{i}}{d t}}=-\gamma_{i}x_{i}+\sum_{j\neq i}^{N}w_{i j}f(x_{j})+b_{i}+\bar{I}_{i}+\sigma_{i}d W(t)$$
wijf(xj ) + bi + ¯Ii + σidW(t) (1)
Where f(x) = 1 1+e−n(x−k), wi,j < 0, and dW(t) is the Wiener process. Here, wijf(xj ) represents the inhibitory interaction between neurons; biis the baseline activity of neuron i; Ii and σidW(t) are the deterministic and the stochastic components of the external input, respectively. σiis the standard deviation of the Wiener noise. We term this type of stochastic continuous Hopfield network with all negative weights the Brain-inspired Bandit Network (BBN), for reasons that will become clear later.

Assuming approximately symmetric weights *i.e., w*ij = wji 1, the deterministic part of the model is essentially a continuous Hopfield network (Hopfield, 1982; 1984) with exclusively inhibitory connections. It is hence associated with a Hopfield energy or Lyapunov function of the form:

$$E=\left\{-\frac{1}{2}\sum_{i,j,i\neq j}^{N}w_{i j}f\left(x_{i}\right)f\left(x_{j}\right)+\sum_{i}^{N}\left[x_{i}f\left(x_{i}\right)-\int_{0}^{x_{i}}f(x)d x\right]-\sum_{i}^{N}b_{i}f\left(x_{i}\right)\right\}\;.$$
$$\left(2\right)$$
$$-\left\{\sum_{i}^{N}{\bar{I}}_{i}f\left(x_{i}\right)\right\}=E^{i n t}-E^{e x t}$$

Here, we have decomposed the Hopfield energy E into Eint, dependent only on internal network parameters, and Eext, which embodies influence from external inputs Ii. With suitable parameters
(see Appendix B.1), the model can have up to N local energy minima or attractor states exhibiting winner-take-all dynamics (Fig. 1 and Fig. 12). Stochastic noise induces transitions between these attractor states, consistent with experimental findings in foraging networks (Ji et al., 2021).

## 3.2 The Bbn Implements Bayesian Posterior Sampling

Hinton and Sejnowski (Hinton & Sejnowski, 1983) have demonstrated that a discrete Hopfield network with stochastically activating units (i.e. an Ising network) can implement Bayesian inference by sampling from the posterior distribution. Here we extend this conclusion to continuous Hopfield networks. Briefly, using Kramers' escape theory (Kramers, 1940; Langer, 1968; Hanggi et al., ¨ 1990), we can approximately compute the mean first passage time (MFPT), defined here as the expected time to leave an attractor state A and crossing the nearby saddle point S, as:

$$\langle\tau_{A}\rangle=\frac{2\pi\gamma}{\omega_{b}}\frac{\prod_{i}^{\prime}\omega_{i}^{S}}{\prod_{i}\omega_{i}^{A}}*\exp\left(\frac{\Delta E_{A}}{D_{A}}\right)$$
(3)

$$(3)$$

Where γ is the friction coefficient (equivalent to τ in Eq. 1, and ω A
iare the angular frequencies
(i.e. eigenvalues of the Hessian matrix) at the center (i.e. energy minimum) of the attractor. ωb and ω S
iare the angular frequencies of the saddle point, with ωb associated specifically with the unstable mode. ∆EA is the energy difference between the saddle point and the center of the attractor and ∆EA→S = ES − EA. DA is the diffusion constant, which in thermodynamics scales with the magnitude of the stochastic noise.

The equilibrium probability of the network being in a given attractor state A1 can be approximated by its stability, measured via the MFPT, relative to the other attractors. This translates to:

$$P_{A_{1}}\cong\frac{\langle\tau_{A1}\rangle}{\sum_{1}^{N}\langle\tau_{Aj}\rangle}=\frac{1}{1+\sum_{2}^{N}\left\{\frac{\alpha_{i}}{\alpha_{1}}\exp\left(\frac{\Delta E_{A1}}{D_{Aj}}-\frac{\Delta E_{A1}}{D_{A1}}\right)\right\}},\quad\text{where}\alpha_{i\in\{1,\ldots,N\}}=\frac{\prod_{j}^{\prime}\omega_{j}^{S_{i}}}{\omega_{k}\prod_{j}\omega_{j}^{A_{i}}}\tag{4}$$

Assuming identical biophysical parameters and inputs for all neurons, the angular frequencies ω Ai j of the N attractors are permutations of each other and there is a single saddle point defined by x =
1 γ Nwf(x) +b+ ¯I. This leads to α1 = αj , ∀i. Further, by substituting ∆EA = (ES −Eint A ) +Eext A
into Eq. 4, we have:

$$P_{A_{1}}\cong\frac{1}{1+\sum_{2}^{N}\left\{\exp\left(\left[\frac{E_{S}-E_{A_{j}}^{\mathrm{w}}}{D_{A_{j}}}-\frac{E_{S}-E_{A_{1}}^{\mathrm{w}}}{D_{A_{1}}}\right]+\left[\frac{E_{A_{j}}^{\mathrm{w}}}{D_{A_{j}}}-\frac{E_{A_{1}}^{\mathrm{w}}}{D_{A_{1}}}\right]\right)\right\}}$$
$$\quad(5)$$

Now if we define the probability of an attractor state in the absence of external inputs as its prior probability as: P
prior Ai = exp ∆Eint Ai /DAi, and the probability of the state given input data (e.g.

sensory evidence) as: P¯I | Ai
= exp (Eext Ai /DAi), we have:

$$P\left(A_{1}\mid I\right)\cong\frac{1}{1+\sum_{2}^{N}\left\{\left(P_{A j}^{\mathrm{prior}}\,/P_{A1}^{\mathrm{prior}}\,\right)*\left[P\left(I\mid A_{j}\right)/P\left(I\mid A_{1}\right)\right]\right\}}$$
$$(6)$$

Eq. 6 reveals a close connection between the Hopfield-energy-based formulation of attractor state probability and Bayesian inference. Specifically, if we consider PAi as the probability of a hypothesis i being true or a decision i being optimal, then Eq. 6 essentially computes the Bayesian posterior of i given external evidence.

## 3.3 The Bbn Can Exhibit Optimistic, Neutral, Or **Conservative** Biases On Input Uncertainty

In Kramers' theory, the diffusion constant D from thermal fluctuations is typically isotropic (Σ = σ 2I, D = σ 2). However, in our model, input to each neuron can have different levels of uncertainty, making the overall noise anisotropic. Recent studies (Zhu et al., 2018; Yang et al., 2023) show that anisotropic noise affects escape efficiency (the rate at which model leaves one of its attractor states (i.e. 1/MFPT)) by interacting with local attractor curvature. Starting from a local energy minimum at x0, the model evolves as:

$$\langle E\left(x_{t}\right)\rangle\cong E\left(x_{0}\right)-\int_{0}^{t}\left\langle\nabla E^{T}\nabla E\right\rangle+{\frac{t}{2}}\left\langle\mathrm{Tr}\left(H_{0}\Sigma\right)\right\rangle$$

Here, H0 is the Hessian matrix evaluated at the attractor bottom, and Σ is the noise covariance matrix. Since both matrices are diagonal in our model, the escape efficiency is highest when the dimensions of largest input noise align with those of highest curvature. To capture this effect, we define an isotropic noise Σ = σ 2I that yields the same efficiency as Σ:

$$\mathrm{Tr}(\mathbf{H}_{i}\mathbf{\Sigma})=2\overline{{{\sigma}}}_{i}^{2}\,\mathrm{Tr}(\mathbf{H}_{i})=\mathrm{Tr}(\mathbf{H}_{i}\mathbf{\Sigma}),\quad\mathrm{~where~}\overline{{{\sigma}}}_{i}^{2}=\frac{\mathrm{Tr}(\mathbf{H}_{i}\mathbf{\Sigma})}{\mathrm{Tr}(\mathbf{H}_{i})}=D_{i}^{e\mathrm{ff}}$$
i(8)
$$\left(7\right)$$
$$(8)$$

Here, Deff irepresents the effective diffusion constant and Hi = PHj = HA, ∀i where P is a permutation matrix. Substituting Eq. 8 into Eq. 4 , we have:

$=\dfrac{1}{1+\exp\left\{2\,\text{Tr}(\pmb{H}_A)\Delta E_A\left(\frac{1}{\text{Tr}(\pmb{H}_A^T\pmb{\Sigma})}-\frac{1}{\text{Tr}(\pmb{H}_A)}\right)\right\}}$
$$(9)$$
Tr(HAΣ)
o (9)
Figure 2: **BBN implements Bayesian posterior sampling with a tunable bias towards, neutral** to, or against input uncertainty. (a) Sigmoidal dependence of attractor state probability on the difference in mean input values. (b) Slope of the state probability curve in (a) as a function of total input uncertainty (defined as pσ 2 1 + σ 2 2) for the three types of networks. (c) Intercept of the state probability curve as a function of relative input uncertainty (defined as σ1 − σ2).

While all N attractors have equal energy and share a common set of angular frequencies, their Hessian matrices are non-identical and can interact differently with non-isotropic noise (i.e.,Σ ̸=
cI). If PA1 corresponds to the attractor state with the highest input noise, the following scenarios can occur (assuming j ̸= 1):
1. Tr(H1Σ) < Tr(HjΣ) and PA1 > PAj (**Optimistic**). 2. Tr(H1Σ) = Tr(HjΣ) and PA1 = PAj (**Neutral**). 3. Tr(H1Σ) > Tr(HjΣ) and PA1 < PAj (**Conservative**).

These regimes are termed as Optimistic, **Neutral**, and **Conservative**, respectively. Fig. 2 illustrates the input dependence of attractor state probabilities under these three regimes. Parameter sensitivity analyses (Fig. 3(a-b) and Fig. 11 in Appendix A) reveal that the three parameter regimes span a wide range of parameter combinations, obviating the need for fine-tuning. By adjusting the baseline activity b, synaptic threshold k, or inhibitory synaptic weight w - either individually or in pairs - one can flexibly modulate the uncertainty bias from highly optimistic
(PA1 → 1) to neutral (PA1 =
1 N
) to highly conservative (PA1 → 0).

## 3.4 (Optimistic) Uncertainty Bias Is Preserved In Higher Dimensions

The theoretical analysis above predicts that the uncertainty bias of BBN should scale well to high dimensions. To verify this empirically, we progressively increased network dimension by adding more neurons, while keeping all network parameters in Eq. 1 unchanged. Strikingly, for a BBN that is optimistic at N = 2, scaling up to N = 10 did not alter its optimistic bias (Fig. 3(c)). In contrast, a BBN that is neutral in 2D became mildly optimistic as N increased, while a conservative BBN becomes mildly optimistic at N > 5. Thus, with increasing network dimension, the model exhibits a tendency to bias towards attractor states with higher input uncertainty. To understand this empirical phenomenon, we examined state-transition dynamics near the saddle point for a perfectly neutral 3D BBN (i.e., Hi = cI, ∀i) (Fig. 13). With isotropic noise, the network exhibited equal probability of entering any attractor state. However, with highly anisotropic noise, it preferentially entered the attractor state along the dimension of highest noise, creating a bias towards high-uncertainty states. This makes conservative bias harder to maintain and optimistic bias more prominent in high-dimensional models (Fig. 14. To incorporate this effect into our theoretical framework, we need to combine escape rates analysis (Kramers, 1940; Zhu et al., 2018) with theory of dynamics around saddle points (Daneshmand et al., 2018)—a challenge we aim to address in future work.

## 4 Experimental Evaluation 4.1 Uncertainty-Aware Exploration In Multi-Armed Bandit Task

Given BBN's ability to infer and sample from a posterior distribution with a tunable uncertainty bias, a natural application of BBN is to control action choice given external, uncertain evidence. We thus adapted the BBN model to play multi-armed bandit (MAB) games and compared its performance with classic bandit algorithms.

## 4.1.1 Running Bbn In Bandit Games

To make the BBN model play bandit games, we (1) define a BBN model with N neurons, each corresponding to one of the N bandit arms; (2) pick network parameters that yield "optimistic" exploration for a 2-D BBN, and simply apply the same parameters to all neurons in the N-D model;
(3) prior to each bandit trial, assign network input I by sampling from the reward memory buffer and numerically simulate the network for T steps using the Runge-Kutta method; (4) at the end of simulation, select the arm a whose corresponding neuron has the highest activation value; (5) collect the reward ra and add it to the memory buffer for arm a; (6) repeat (3)-(5) for the next trial till the game ends. The pseudocode along with detailed task parameters are presented in Appendix B.1.

## 4.1.2 Bbn Implements Uncertainty-Aware Posterior Sampling

To reveal BBN's exploration strategies, we examined the dependence of choice probability on total and relative reward uncertainty for BBN agents with optimistic, neutral, or conservative biases, as well as classic algorithms Thompson Sampling (TS) and Upper Confidence Bound (UCB). As shown in Fig. 4 (a-b), TS exhibited a constant intercept regardless of relative uncertainty (RU) and a decreasing slope with increasing total uncertainty (TU), indicating sensitivity only to total uncertainty; UCB exhibited a constant slope with varying TU and an increasing intercept with increasing RU, indicating sensitivity only to relative uncertainty. In contrast, BBN with optimistic parameters showed variation in both slope and intercept with changes in TU and RU. These results indicate that BBN implements a hybrid algorithm combining posterior sampling (like TS) with tunable bias towards high uncertainty (akin to UCB).

## 4.1.3 Efficient Exploration In Bandit Tasks

We next compared the empirical performance of BBN-driven exploration in comparison against UCB, Thompson sampling, and Optimistic Thompson Sampling (OTS, (Hu et al., 2023)) in both 2-armed bandit and 3-armed bandit games. Each agent played 10,000 game blocks of 20 trials each in 2-armed bandit games and 30 trials each in 3-armed bandit games. Fig. 5 (a-b) presents the probability of choosing the optimal arm as trial number increases. BBN (with optimistic parameters) consistently outperformed other algorithms in 2-armed bandits and topped the performance in 3-armed bandit games. The other 'hybrid' algorithm, OTS, performed close to BBN in 3-armed bandits, but did poorly in 2-armed bandits.

## 4.2 Bbn Closely Approximates Bandit Choice Behavior In Humans And Animals

The results above indicate that BBN exhibits similar hybrid strategies as previously reported in humans (Wilson et al., 2014; Gershman, 2018). We thus asked whether BBN can accurately model human and animal choice patterns in bandit tasks. We first compiled several publicly available datasets of humans playing bandit games (detailed list in Appendix C). We performed optimization on two network parameters b and k to minimize the difference between the choice probability curves output by BBN and in the human datasets. As shown in Fig. 6 (a-b), BBN can closely fit to both the intercept and the slope of human choice probability curves. In contrast, Thompson sampling failed to fit to the diverse intercepts across human groups, and UCB consistently yielded slopes that are much higher than those observed in human data. We next extended the above analyses to a dataset in which mice played switching blocks of 2armed bandit games (Beron et al., 2022). In this dataset, the reward for each arm is sampled from a Bernoulli distribution. In addition, the mean reward for each arm is not static, with a small probability (0.02) of being reversed before each trial starts. Based on results from (Beron et al., 2022), we used the last five rewards as inputs to the BBN model to drive choice behavior. As shown in Fig. 6 (c-d), parameter-tuned BBN generated choice and switching behavior that closely approximated those exhibited in the mice study.

## 4.3 Efficient Exploration In Mdp Problems

Building on the strong performance of BBN in MAB tasks, we next applied BBN to MDP problems. Unlike bandit problems with immediate rewards and no state transitions, MDP tasks require sequential decision-making under delayed rewards and unknown transition probabilities (Bellman, 1966; Bertsekas, 2012). Among existing methods, UCRL2 (Auer et al., 2008) extends OFU to MDPs, while PSRL (Strens, 2000; Osband et al., 2013) generalizes posterior sampling to RL. Hybrid algorithms like Optimistic Thompson Sampling (OTS) (Agrawal & Jia, 2017; Tiapkin et al., 2022; Hu et al., 2023) aim to improve exploration efficiency but face challenges such as computational cost and uncertainty estimation.

We consider a finite-horizon MDP with state space S, action space A, horizon H, rewards r lsa, and transition probabilities Psa conditioned on states s, actions a, and step l. The expected total return at step l under policy π can be estimated iteratively using the Bellman equation:

$$Q_{s a}^{t+1}=\mu_{s a}+\sum_{s^{\prime}a^{\prime}}\pi_{s^{\prime}a^{\prime}}P_{s a s^{\prime}}Q_{s^{\prime}a^{\prime}}^{t}$$

where µ = E(r) is the mean reward. Estimating uncertainty in Q-values remains an open issue in RL. Donoghue et al. (O'Donoghue et al., 2018) proposed the Uncertainty Bellman Equation (UBE) to provide an upper bound on the variance of Q-value posteriors. For tabular state space, this method effectively propagates local variance estimates to global value uncertainty.

## 4.3.1 Running Bbn In Mdp Tasks

To apply BBN to drive action-selection in MDP tasks, we (1) define a BBN model with N neurons, each corresponding to one of the N discrete actions, select network parameters that belong to the "optimistic" regime for a 2D network; (2) initialize state-action values to i.i.d. Gaussian distributions; (3) sample input values for each neuron from the distributions of state-action values and perform numerical simulation of the BBN network for T steps using the Runge-Kutta method; (4) at the end of the simulation, select action a whose corresponding neuron has the highest activation value; (5) collect the reward ra and move to the next state ; (6) Repeat (3)-(5) till the episode ends;
(7) Update the distribution of state-action values using the Uncertainty Bellman Equation (UBE) algorithm(O'Donoghue et al., 2018). (8) repeat (3)-(7) for the next episode till the game ends. We present the pseudo-code for the Algorithm 2 in Appendix B.4. We first compared the exploration efficiency of the BBN-based algorithm (UBE BBN) on the SixArms (Strehl & Littman, 2008) task, with additional implementation details presented in Appendix B.5. We compare our model to PSRL(Osband et al., 2013), UCRL2 (Auer et al., 2008) and OTS-MDP (Hu et al., 2023). We also specifically tested the role of BBN by replacing it with UCB (UBE UCB) or Thompson sampling (UBE TS). In PSRL, we maintain a Gaussian distribution for the rewards and a Dirichlet distribution for the transition probabilities. In the OTS-MDP and BBN models, we follow(Hu et al., 2023) and limit our uncertainty estimation to the reward r for simplicity. As shown in Fig. 5 (c), the cumulative regret is lowest in UBE-BBN, which demonstrates the potential of BBN in promoting highly efficient exploration. Figure 7: **BBN-enhanced RL agent exhibits efficient exploration in the FourRooms task.** (a) The FourRooms environment. The agent starts at the red point and can receive a reward only at the blue point. (b) The percent of grids covered (i.e. the coverage rate) by agents driven by various exploration algorithms over the period of training. (c) Display of visitation counts over the course of training. (d) Visitation counts for the UBE-BBN agent with or without action persistence. (e) Number of episodes taken till first reaching the reward state for different agents. Pink and purple are the UBE-BBN agents with and without action persistence respectively. Blue is PSRL and green is UBE UCB We next evaluated the exploration efficiency of BBN on sparse-reward MDP tasks, specifically the FourRooms task. In this task, an N-by-N grid world is divided into four compartments connected by narrow passages (Fig. 7 (a)). The agent starts from the upper left corner (red dot) and explores the environment to learn state-action values. First, we conducted reward-free exploration by assuming no rewards at any state. Exploration efficiency was measured as the coverage rate (ratio of visited states to total states) over episodes. Fig. 7 (b) shows that UBE-BBN achieved the fastest coverage rate among all methods. Fig. 7 (c) provides examples of cumulative visitation counts for each method during training. We then varied the environment size and repeated the experiments. UBE- BBN scaled well with grid size, while other algorithms faltered (Fig. 19 in Appendix E). Additional comparisons with more methods in different conditions are in Fig. 20-23 in Appendix E. Trajectories (visitation counts in a single episode) in Fig. 24 reveal that UBE-BBN excelled in extended deep exploration, covering hard-to-reach states effectively. Finally, we enhanced action persistence in UBE-BBN by allowing the BBN model to inherit activity states from the previous step (Fig. 25). This modification leveraged the Hopfield network's persistence property, instilling action correlation within episodes. As shown in Fig. 25, adding persistence further boosted UBE-BBN's exploration efficiency in the FourRooms task at large grid sizes.

Parameter sensitivity in MDP tasks: We additionally performed parameter sensitivity analysis for the SixArms and FourRooms task (as shown in Fig. 18 in Appendix E.1) and demonstrated that a broad range of "optimistic" network parameters yielded high performance on these tasks.

Hence, optimistic BBN generally delivers good performance in these MDP tasks without requiring parameter fine-tuning.

## 5 Discussion

We have demonstrated both theoretically and empirically that the BBN architecture can drive flexible and efficient exploration in ways similar to humans and animals. However, several limitations and open questions remain regarding its practical application. **First**, simulating the stochastic differential equations incurs high computational costs. This issue may be circumvented by analytically computing the attractor probabilities using Eq. 4 or by employing neuromorphic hardware. Second, given the development of many hybrid TS and OFU methods in the RL community (Hu et al., 2023; Tiapkin et al., 2022; Agrawal & Jia, 2017), it's intriguing to consider what gives rise to BBN's superior performance. One possibility is that BBN, as a system of coupled Langevin equations, effectively implements Langevin sampling of the posterior distribution. Langevin sampling has been shown to enjoy faster mixing and convergence rates than other sampling methods and is particularly well-suited for approximate Bayesian inference (Welling & Teh, 2011). **Third**, the current BBN algorithm lacks the ability to estimate uncertainty associated with state-action values, relying instead on a separate algorithm (in this case, the UBE) to generate value distributions. How biological neural networks compute and encode uncertainty remains an outstanding question, especially in sequential decision settings. Recent studies have suggested that a distributed population code (Dehaene et al., 2021) or a spatiotemporal activity pattern could encode uncertainty levels (Savin & Deneve, 2014). We hope future experimental and theoretical studies will provide more insights into ` how the brain estimates and utilizes uncertainty. **Lastly**, given that humans and animals can flexibly modulate their uncertainty bias in a context-dependent manner, a valuable extension for the BBN algorithm would be to integrate contextual information into the network input. Expanding the BBN model to include upstream neurons found in the biological foraging network might help implement context-dependent E-E decisions (Fig. 8).

## References

Merideth A Addicott, John M Pearson, Maggie M Sweitzer, David L Barack, and Michael L Platt.

A primer on foraging and the explore/exploit trade-off for psychiatry research. Neuropsychopharmacology, 42(10):1931–1939, 2017.

Rajeev Agrawal. Sample mean based index policies by o (log n) regret for the multi-armed bandit problem. *Advances in applied probability*, 27(4):1054–1078, 1995.

Shipra Agrawal and Randy Jia. Optimistic posterior sampling for reinforcement learning: worstcase regret bounds. *Advances in Neural Information Processing Systems*, 30, 2017.

Peter Auer, Nicolo Cesa-Bianchi, Yoav Freund, and Robert E Schapire. Gambling in a rigged casino:
The adversarial multi-armed bandit problem. In Proceedings of IEEE 36th annual foundations of computer science, pp. 322–331. IEEE, 1995.

Peter Auer, Thomas Jaksch, and Ronald Ortner. Near-optimal regret bounds for reinforcement learning. *Advances in neural information processing systems*, 21, 2008.

Yikun Ban, Yuchen Yan, Arindam Banerjee, and Jingrui He. EE-net: Exploitation-exploration neural networks in contextual bandits. In *International Conference on Learning Representations*, 2022.

URL https://openreview.net/forum?id=X_ch3VrNSRg.

Frederic Bartumeus, Daniel Campos, William S Ryu, Roger Lloret-Cabot, Vicenc¸ Mendez, and Jordi ´
Catalan. Foraging success under uncertainty: search tradeoffs and optimal space use. Ecology letters, 19(11):1299–1313, 2016.

Richard Bellman. Dynamic programming. *science*, 153(3731):34–37, 1966.

Celia C Beron, Shay Q Neufeld, Scott W Linderman, and Bernardo L Sabatini. Mice exhibit stochastic and efficient action switching during probabilistic decision making. Proceedings of the National Academy of Sciences, 119(15):e2113961119, 2022.

Donald A Berry and Bert Fristedt. Bandit problems: sequential allocation of experiments (monographs on statistics and applied probability). *London: Chapman and Hall*, 5(71-87):7–7, 1985.

Dimitri Bertsekas. *Dynamic programming and optimal control: Volume I*, volume 4. Athena scientific, 2012.

Olivier Chapelle and Lihong Li. An empirical evaluation of thompson sampling. Advances in neural information processing systems, 24, 2011.

Eric L Charnov. Optimal foraging, the marginal value theorem. *Theoretical population biology*, 9
(2):129–136, 1976.

Tianping Chen and Shun Ichi Amari. Stability of asymmetric hopfield networks. IEEE Transactions on Neural Networks, 12(1):159–163, 2001.

Jeffrey Cockburn, Vincent Man, William A Cunningham, and John P O'Doherty. Novelty and uncertainty regulate the balance between exploration and exploitation through distinct mechanisms in the human brain. *Neuron*, 110(16):2691–2702, 2022.

Jonathan D Cohen, Samuel M McClure, and Angela J Yu. Should i stay or should i go? how the human brain manages the trade-off between exploitation and exploration. *Philosophical Transactions of the Royal Society B: Biological Sciences*, 362(1481):933–942, 2007.

Vincent D Costa and Bruno B Averbeck. Primate orbitofrontal cortex codes information relevant for managing explore–exploit tradeoffs. *Journal of Neuroscience*, 40(12):2553–2561, 2020.

Vincent D Costa, Andrew R Mitz, and Bruno B Averbeck. Subcortical substrates of explore-exploit decisions in primates. *Neuron*, 103(3):533–545, 2019.

Hadi Daneshmand, Jonas Kohler, Aurelien Lucchi, and Thomas Hofmann. Escaping saddles with stochastic gradients. In *International Conference on Machine Learning*, pp. 1155–1164. PMLR, 2018.

Nathaniel D Daw, John P O'doherty, Peter Dayan, Ben Seymour, and Raymond J Dolan. Cortical substrates for exploratory decisions in humans. *Nature*, 441(7095):876–879, 2006.

Guillaume P Dehaene, Ruben Coen-Cagli, and Alexandre Pouget. Investigating the representation of uncertainty in neuronal circuits. *PLOS Computational Biology*, 17(2):e1008138, 2021.

Haoxue Fan, Samuel J Gershman, and Elizabeth A Phelps. Trait somatic anxiety is associated with reduced directed exploration and underestimation of uncertainty. *Nature Human Behaviour*, 7(1):
102–113, 2023.

Steven W Flavell, Navin Pokala, Evan Z Macosko, Dirk R Albrecht, Johannes Larsch, and Cornelia I
Bargmann. Serotonin and the neuropeptide pdf initiate and extend opposing behavioral states in c. elegans. Cell, 154(5):1023–1035, 2013.

Samuel J Gershman. Deconstructing the human algorithms for exploration. *Cognition*, 173:34–42, 2018.

Samuel J Gershman. Uncertainty and exploration. *Decision*, 6(3):277, 2019. Peter Hanggi, Peter Talkner, and Michal Borkovec. Reaction-rate theory: fifty years after kramers. ¨
Reviews of modern physics, 62(2):251, 1990.

Geoffrey E Hinton and Terrence J Sejnowski. Optimal perceptual inference. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, volume 448, pp. 448–453. Citeseer, 1983.

Jeremy Hogeveen, Teagan S Mullins, John D Romero, Elizabeth Eversole, Kimberly Rogge-
Obando, Andrew R Mayer, and Vincent D Costa. The neurocomputational bases of exploreexploit decision-making. *Neuron*, 110(11):1869–1879, 2022.

John J Hopfield. Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the national academy of sciences*, 79(8):2554–2558, 1982.

John J Hopfield. Neurons with graded response have collective computational properties like those of two-state neurons. *Proceedings of the national academy of sciences*, 81(10):3088–3092, 1984.

Bingshan Hu, Tianyue H Zhang, Nidhi Hegde, and Mark Schmidt. Optimistic thompson samplingbased algorithms for episodic reinforcement learning. In *Uncertainty in Artificial Intelligence*, pp. 890–899. PMLR, 2023.

Ni Ji, Gurrein K Madan, Guadalupe I Fabre, Alyssa Dayan, Casey M Baker, Talya S Kramer, Ijeoma Nwabudike, and Steven W Flavell. A neural circuit for flexible control of persistent behavioral states. *Elife*, 10:e62889, 2021.

Hendrik Anthony Kramers. Brownian motion in a field of force and the diffusion model of chemical reactions. *Physica*, 7(4):284–304, 1940.

Tze Leung Lai and Herbert Robbins. Asymptotically efficient adaptive allocation rules. Advances in applied mathematics, 6(1):4–22, 1985.

JS Langer. Theory of nucleation rates. *Physical Review Letters*, 21(14):973, 1968. Veldon-James Laurie, Akram Shourkeshti, Cathy S Chen, Alexander B Herman, Nicola M Grissom, and R Becket Ebitz. Persistent decision-making in mice, monkeys, and humans. *bioRxiv*, pp. 2024–05, 2024.

Kiyotoshi Matsuoka. Stability conditions for nonlinear continuous neural networks with asymmetric connection weights. *Neural networks*, 5(3):495–500, 1992.

Benedict C May, Nathan Korda, Anthony Lee, David S Leslie, and Nicolo Cesa-Bianchi. Optimistic bayesian sampling in contextual-bandit problems. *Journal of Machine Learning Research*, 13(6), 2012.

Jack-Morgan Mizell, Siyu Wang, Alec Frisvold, Lily Alvarado, Alex Farrell-Skupny, Waitsang Keung, Caroline E Phelps, Mark H Sundman, Mary-Kathryn Franchetti, Ying-hui Chou, et al. Differential impacts of healthy cognitive aging on directed and random exploration. Psychology and Aging, 39(1):88, 2024.

Ian Osband, Daniel Russo, and Benjamin Van Roy. (more) efficient reinforcement learning via posterior sampling. *Advances in Neural Information Processing Systems*, 26, 2013.

Brendan O'Donoghue, Ian Osband, Remi Munos, and Volodymyr Mnih. The uncertainty bellman equation and exploration. In *International conference on machine learning*, pp. 3836–3845, 2018.

Herbert Robbins. Some aspects of the sequential design of experiments. 1952. Cristina Savin and Sophie Deneve. Spatio-temporal representations of uncertainty in spiking neural `
networks. *Advances in neural information processing systems*, 27, 2014.

Eric Schulz and Samuel J Gershman. The algorithmic architecture of exploration in the human brain.

Current opinion in neurobiology, 55:7–14, 2019.

Eric Schulz, Charley M Wu, Azzurra Ruggeri, and Bjorn Meder. Searching for rewards like a child ¨
means less generalization and more directed exploration. *Psychological science*, 30(11):1561–
1572, 2019.

David W Stephens and John R Krebs. *Foraging theory*, volume 6. Princeton university press, 1986. Alexander L Strehl and Michael L Littman. An analysis of model-based interval estimation for markov decision processes. *Journal of Computer and System Sciences*, 74(8):1309–1331, 2008.

Malcolm Strens. A bayesian framework for reinforcement learning. In *ICML*, volume 2000, pp.

943–950, 2000.

Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. *Robotica*, 17(2):
229–235, 1999.

Richard S Sutton and Andrew G Barto. *Reinforcement learning: An introduction*. MIT press, 2018.

William R Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika*, 25(3-4):285–294, 1933.

Daniil Tiapkin, Denis Belomestny, Daniele Calandriello, Eric Moulines, Remi Munos, Alexey Naumov, Mark Rowland, Michal Valko, and Pierre Menard. Optimistic posterior sampling for re- ´
inforcement learning with few samples and tight guarantees. Advances in Neural Information Processing Systems, 35:10737–10751, 2022.

Momchil S Tomov, Van Q Truong, Rohan A Hundia, and Samuel J Gershman. Dissociable neural correlates of uncertainty underlie different exploration strategies. *Nature communications*, 11(1): 2371, 2020.

James A Waltz, Robert C Wilson, Matthew A Albrecht, Michael J Frank, and James M Gold. Differential effects of psychotic illness on directed and random exploration. Computational Psychiatry (Cambridge, Mass.), 4:18, 2020.

Siyu Wang, Blake Gerken, Julia R Wieland, Robert C Wilson, and Jean-Marc Fellous. The effects of time horizon and guided choices on explore–exploit decisions in rodents. Behavioral neuroscience, 137(2):127, 2023.

Max Welling and Yee W Teh. Bayesian learning via stochastic gradient langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11), pp. 681–688. Citeseer, 2011.

Robert C Wilson, Andra Geana, John M White, Elliot A Ludvig, and Jonathan D Cohen. Humans use directed and random exploration to solve the explore–exploit dilemma. Journal of experimental psychology: General, 143(6):2074, 2014.

Robert C Wilson, Elizabeth Bonawitz, Vincent D Costa, and R Becket Ebitz. Balancing exploration and exploitation with information and randomization. *Current opinion in behavioral sciences*, 38: 49–56, 2021.

Ning Yang, Chao Tang, and Yuhai Tu. Stochastic gradient descent introduces an effective landscapedependent regularization favoring flat solutions. *Physical Review Letters*, 130(23):237101, 2023.

Wojciech K Zajkowski, Malgorzata Kossut, and Robert C Wilson. A causal role for right frontopolar cortex in directed, but not random, exploration. *Elife*, 6:e27430, 2017.

Weitong Zhang, Dongruo Zhou, Lihong Li, and Quanquan Gu. Neural thompson sampling, 2021.

URL https://arxiv.org/abs/2010.00827.

Dongruo Zhou, Lihong Li, and Quanquan Gu. Neural contextual bandits with ucb-based exploration, 2020. URL https://arxiv.org/abs/1911.04462.

Zhanxing Zhu, Jingfeng Wu, Bing Yu, Lei Wu, and Jinwen Ma. The anisotropic noise in stochastic gradient descent: Its behavior of escaping from sharp minima and regularization effects. arXiv preprint arXiv:1803.00195, 2018.

APPENDIX

## A

SUPPLEMENTAL FIGURES

a. Biological neural model b. 3-dimensional BBN model Action 3 SN
SN
SN
Sensory Neurons IN
Interneurons X1 Motor 5-HT
PDF
command neurons X1 xz Dwell Roam Action 1 Action 2
(xxploit)
(xplore)
(myopically optimal)
States distribution Hopfield energy Phase plane 20 20 11 xx 0.4
−9.5 8 l X2 15 15 Optimistic
 
.0.3
-10.0 ity 6 1 10 Den 0.2
-10.5 a 5 5 0.1
-11.0 2
-11.5 0.0 o o o o ntwork state 20 10 15 20 10 5 15 20 x1 X1 20 10 20 0.3 xx 8 15 15
-95 x x 2 Neutral 6 Density 2 ' 10 1" 10
-10.0 4 0.1 5 5
-10.5 2 o o
-11.0 0.0 o 20 10 15 20 10 5 20 0 5 10 15 network state X1 X1 20 10 20 0.5 I X1 8
−8.5 Conservative 0.4 IX2 15 15 6'
−9.0 50.3 111 10 2 Der 4
−9.5 0.2 5 5 0.1 2
-10.0 0.0
-10.5 0 o o
  
10 is 20 5 10 15 20 o 20 x1 x1 Intercept shift Slope shift 1.0 1.0 0 = 3.5.02 = 2.5 0 = 3,0, = 2 choice probability
 
4 0.8 0 = 2.5, σ 2 = 1.5 choice probability 0 = 2.00 = 1 Optimistic 0.6 0.4 0 = 2.0.0) = 0.5 of = 1.75, of = 0.75 0.00.2 of = 1.5.0] = 1.0 0 = 1.25, 0 = 1.25 0.0 0.0
-1.5 -1.0 -0.5 0.0 0.5
-1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 1.0 1.5 value difference value difference 1.0 1.0 0 = 2.0.0) = 0.5 0 = 3.5, = 2.5 0 = 3 , 0 > = 2 0 1 = 1.75, 6 2 = 0.75 choice probability
 
choice probability
 12 0 1 = 2.5.02 = 1.5 1 = 1.5,0 1 = 1.0 2,0, m 1 0 = 1.25, of n 1.25 Neutral 0.0 0.0
-1.5 -1.0 -0.5 0.0 0.5 1.0 1.5
-1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 value difference value difference 1.0 1.0 0 = 2.0, = 0.5 0 = 3, 5, 0, = 2,5
0, = 3, 0, = 2 0 1 = 1.75, 6 1 = 0.75 choice probability
 
choice probability
 999   999   999 0 = 2.5.0 2 = 1.5
{ = 1.5,0} = 1.0 Conservative 0 = 2,00 = 1 of = 1.25, of = 1.25 0.0 0.0
−1.5 −1.0 −0.5  0.0 0.5 1.0 1.5
-1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 value difference value difference 5.0 4
- optimistic 3 4.5 neutral 2 4.0 conservative 3.5 l intercept slope
 3.0
 5.

0 2.5
-1 2.0
-2 1.5
-3 1.0 2.5 3.0 4 1.0 3.5 4.0 0.0 0.2 0.4 0.6 0.8 total uncertainty relative uncertainty 1.11.0 4.0 1.1.0 1.11.0 1.1.1.1.1.0 0.7 s
-0.8
-0.8 0.00.00.00.00.00.00.00.00.00.00.00.00.00.00.00 30 0.6 9.99.0 ol 0.00 0,5 0.00 2.

Theory 0's
M
n
>
2, 2,0 0.4 0.0.4
-0.4 0.4 1.5 0.2 5.5.5.5.5.5 02 102 0.0 1.0 0.2 ols.

3.0.1 0,00 0.0 20 os 0'2

990 08 000

O'S
5.0 7.0 3.5 5.0 80 9.99.0 1.11.0 1.13.0 15.0 6.5 1.1.0 ses k k k 1.

1.0 1.0 4.0 10.5 0.00.7 3.5
-0.0
-0.1
-0.5 9.999.999.

0.6 3.0 0.00.6 0.00.6
-0.6 0.5 2.5 se Simulation u 04
-0.4 2. 2.0 0.00.4
-0.4 16.0 1.5 ol of the content of the 102 02 0.2 4,54,54,55 1,0 0.2
.

6.5 f 0.00.00 0,00 ol and the content of the 3.0 1 5.0 3.5 5.0 8.0 00

OOTO
7.0 9.0 xx s 11.0 15.0 1.11.0 1.0 2.0 k k Neutral Optimistic Conservative
(b = 6.25)
( b = 5.5)
(b = 5)
 10 tate dynamic 10 10 s X3 5 X3 5 X3 o 0 0 o o o 5 o o 0 10 x 2 5 5 s 10 X2 10 x 10 10 10 115 15 15 15 X1 X1 x1
+ Nin(o)
1.4
+ Nin(o)
•• Nin(a)
State probability
       (theory)
NARIO
--
--
as D. B
+ Nodanial
--- 1/n
--- 1/n
--- 1/n 0.6 0.6 0.6 0,4 0.4 0.7 0.2 0.2 0.0 a.o 00 10 10 1.0 Mn(0)
- Molol Nintel M
Maxiaxi ate probability
(simulation)
Nax(q)
0.

0.

Median(o)
- Median/s/
+ Nedarko,
-- 1/n
--. In 1/n as 0.

0.6 a 0,00 0.

a o 0.2 0.0 0.0 0.0 Network dimension (n)

## B Method Details

B.1 PARAMETER SELECTION

| Parameter   | Definition                 | Suggested range                             |
|-------------|----------------------------|---------------------------------------------|
| w           | inhibitory weights         | [2, 4], increase to make states more stable |
| b           | activity baseline          | [5.5, 7], increase makes network optimistic |
| k           | threshold of sigmoid       | [6, 8], increase makes network conservative |
| n           | slope of sigmoid           | [1, 2], increase amplifies uncertainty bias |
| γ           | leak current or decay rate | 0.5                                         |
| τ           | time constant              | 1                                           |

Table 1: Internal Model Parameters

| Parameter   | Definition        | Suggested range                    |
|-------------|-------------------|------------------------------------|
| ¯I          | mean of input     | scale raw input values to [-2, 2]  |
| σ 2         | variance of input | scale raw input variance to [0, 2] |

Table 2: External Parameters

| Parameter   | Definition             | Suggested range                                |
|-------------|------------------------|------------------------------------------------|
| N           | number of neurons      | typically equal to number of actions choices   |
| T           | total simulation steps | [400, 1000]                                    |
| dt          | step length            | 0.1 or 0.2 if using suggested parameter ranges |

In this section, we list the primary parameters used in BBN (Tables 1, 2, 3) and provide a principled way to determine optimal parameters for new environments. Based on our experience and past literature (May et al., 2012; Hu et al., 2023; Agrawal & Jia, 2017), optimistic bias generally promotes efficient exploration. In addition, our sensitivity analysis on MDP tasks (Fig. 18) showed that a broad range of "optimistic" parameters yielded high performance, obviating the need for extensive fine-tuning. Further, we have shown that network parameters that yield optimistic bias for a 2D BBN preserve such bias in higher dimensions (Fig. 3(c) and Fig. 12). Thus, the steps to set up a N-dimensional BBN model are:
(1) Define a BBN model with N interconnected neurons;
(2) Select internal network parameters 1 from the "optimistic" regime based on sensitivity analysis results presented in (Fig. 3(a-b) and Fig. 11), or use the parameter ranges suggested below as a starting point; (3) Verify that the 2D network has two attractors and exhibits optimistic bias by numerically simulating the model under anisotropic 2D Gaussian noise with µ = [0, 0],σ = [1, 0.1]); tune the parameters if necessary using the tips provided below; (4) Apply these parameters to all neurons in the ND network; (5) Scale the input to the network (typically past rewards or Q-values) to a range that permits the existence of multiple attractors (use suggested range or verify empirically). We found that simulation step number of T=400 is sufficient for bandit and MDP tasks t. Below are sample network dynamics in the first episode of a 2-armed bandit game. Multiple transitions occurred between the attractor states, reflecting equal state probability as expected for equal uncertainty for the two arms.