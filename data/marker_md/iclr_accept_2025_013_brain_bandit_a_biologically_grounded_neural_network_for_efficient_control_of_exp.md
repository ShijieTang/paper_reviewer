# BRAIN BANDIT: A BIOLOGICALLY GROUNDED NEU-RAL NETWORK FOR EFFICIENT CONTROL OF EXPLO-RATION

Chen Jiang<sup>1</sup><sup>∗</sup> , Jiahui An<sup>2</sup>,<sup>1</sup> , Yating Liu<sup>3</sup>,2,<sup>1</sup> , Ni ji<sup>2</sup>,1†

<sup>1</sup>Chinese Institute for Brain Research, Beijing

<sup>2</sup>Chinese Academy of Medical Sciences & Peking Union Medical College

<sup>3</sup>China Agricultural University

chen.jiang3@mail.mcgill.ca, {anjiahui, liuyating, niji}@cibr.ac.cn

### ABSTRACT

How to balance between exploration and exploitation in an uncertain environment is a central challenge in reinforcement learning. In contrast, humans and animals have demonstrated superior exploration efficiency in novel environments. To understand how the brain's neural network controls exploration under uncertainty, we analyzed the dynamical systems model of a biological neural network that controls explore-exploit decisions during foraging. Mathematically, this model (named the Brain Bandit Net, or BBN) is a special type of stochastic continuous Hopfield network. We show through theory and simulation that BBN can perform posterior sampling of action values with a tunable bias towards or against uncertain options. We then demonstrate that, in multi-armed bandit (MAB) tasks, BBN can generate probabilistic choice behavior with a flexible uncertainty bias resembling human and animal choice patterns. In addition to its high efficiency in MAB tasks, BBN can also be embedded with reinforcement learning algorithms to accelerate learning in MDP tasks. Altogether, our findings reveal the theoretical foundation for efficient exploration in biological neural networks and propose a general, brain-inspired algorithm for enhancing exploration in RL. The code is available at<https://github.com/Chen-Ginger/BrainBandit>

# 1 INTRODUCTION

The explore-exploit (E-E) dilemma, originally described in the context of animal foraging [\(Stephens](#page-12-0) [& Krebs, 1986;](#page-12-0) [Charnov, 1976\)](#page-10-0), has become an important problem across many fields including psychology, neuroscience and reinforcement learning (RL)[\(Addicott et al., 2017\)](#page-10-1). Despite the development of numerous algorithms, sample-efficient exploration in RL remains difficult for complex, sparse-reward tasks [\(Sutton & Barto, 2018\)](#page-12-1). Meanwhile, studies in humans and animals have revealed a diverse array of exploration strategies [\(Wilson et al., 2021;](#page-13-0) [Schulz & Gershman, 2019\)](#page-12-2). In addition, excitingly, recent research has begun to reveal the biological neural networks that give rise to the rich and flexible exploration behaviors [\(Costa et al., 2019;](#page-10-2) [Tomov et al., 2020;](#page-12-3) [Hogeveen](#page-11-0) [et al., 2022;](#page-11-0) [Costa & Averbeck, 2020\)](#page-10-3). Based on recent findings in the biological neural network that controls exploration, we built the Brain Bandit Network (BBN), a stochastic Hopfield network for controlling exploratory action selection under input uncertainty. We show theoretically that the BBN model can perform Bayesian posterior sampling while implementing a tunable bias that ranges from optimistic, neutral, and conservative in the face of uncertainty.

Our main contributions are four-fold:

- 1. We propose a biologically grounded, scalable network model for solving the E-E dilemma.
- 2. We analytically show that BBN implements a hybrid between Bayesian posterior sampling and uncertainty-directed exploration.

<sup>∗</sup>Current address: McGill University.

<sup>†</sup>Corresponding author.

- 3. We show that BBN can closely approximate human and animal behavior in bandit tasks under a variety of conditions.
- 4. We show that BBN can drive highly efficient exploration in bandit and MDP tasks, promising further application to more complex RL problems.

### 2 BACKGROUND AND RELATED WORK

#### 2.1 THE EXPLORATION PROBLEM IN REINFORCEMENT LEARNING

The domain of efficient exploration in reinforcement learning focuses on balancing immediate rewards (exploitation) and information gathering for future rewards (exploration). A classic example is the Multi-Armed Bandit (MAB) problem, introduced by [\(Robbins, 1952\)](#page-12-4) in 1952 and widely used to model this tradeoff [\(Lai & Robbins, 1985;](#page-11-1) [Berry & Fristedt, 1985;](#page-10-4) [Agrawal, 1995;](#page-10-5) [Auer et al.,](#page-10-6) [1995;](#page-10-6) [Sutton & Barto, 1999\)](#page-12-5). Conventional methods inject noise into action selection [\(Sutton &](#page-12-5) [Barto, 1999\)](#page-12-5), but these dithering algorithms can be inefficient. Alternative methods like Upper Confidence Bound (UCB) employ optimism in the face of uncertainty (OFU) by biasing for uncertain choices [\(Lai & Robbins, 1985;](#page-11-1) [Agrawal, 1995;](#page-10-5) [Auer et al., 1995\)](#page-10-6). Thompson sampling [\(Thomp](#page-12-6)[son, 1933\)](#page-12-6) makes decisions based on posterior samples rather than optimistic estimates. Optimistic Thompson Sampling (O-TS), combining UCB and Thompson sampling, reshapes the posterior distribution optimistically and exhibits strong empirical and theoretical performance [\(Chapelle & Li,](#page-10-7) [2011;](#page-10-7) [May et al., 2012\)](#page-11-2). More recent methods leverage deep networks to learn the exploration bonus [\(Zhou et al., 2020;](#page-13-1) [Ban et al., 2022\)](#page-10-8) or the posterior variance [\(Zhang et al., 2021\)](#page-13-2).

#### 2.2 BIOLOGICAL SOLUTIONS TO THE EXPLORE-EXPLOIT DILEMMA

Early work on the explore-exploit tradeoff, rooted in Optimal Foraging Theory and the Marginal Value Theorem [\(Stephens & Krebs, 1986;](#page-12-0) [Charnov, 1976\)](#page-10-0), suggests that animals achieve nearoptimal balance between exploiting known resources and exploring uncertain options. Cognitive scientists have used bandit tasks to study this tradeoff in humans and animals [\(Addicott et al., 2017;](#page-10-1) [Cohen et al., 2007;](#page-10-9) [Wang et al., 2023;](#page-12-7) [Beron et al., 2022\)](#page-10-10). Two main strategies emerge: random exploration, involving stochastic action choices, and directed exploration, leveraging uncertainty to guide actions [\(Wilson et al., 2021;](#page-13-0) [Schulz & Gershman, 2019\)](#page-12-2). Humans and animals often combine these strategies flexibly, adjusting based on task horizon, option novelty, developmental stage, and mental state [\(Gershman, 2018;](#page-11-3) [Bartumeus et al., 2016;](#page-10-11) [Wilson et al., 2014;](#page-12-8) [Cockburn et al., 2022;](#page-10-12) [Mizell et al., 2024;](#page-12-9) [Schulz et al., 2019;](#page-12-10) [Addicott et al., 2017;](#page-10-1) [Fan et al., 2023;](#page-11-4) [Waltz et al., 2020\)](#page-12-11). Additionally, they exhibit persistent exploration, repeating previous choices regardless of value [\(Beron](#page-10-10) [et al., 2022;](#page-10-10) [Laurie et al., 2024\)](#page-11-5). These strategies resemble algorithms like Thompson sampling and Optimism in the Face of Uncertainty (OFU), but with key differences [\(Wilson et al., 2021\)](#page-13-0).

To understand the brain's solution to the E-E dilemma, neuroscientists have identified neurobiological mechanisms that control explore-exploit decisions [\(Daw et al., 2006;](#page-11-6) [Costa et al., 2019;](#page-10-2) [Hogeveen et al., 2022\)](#page-11-0). Recent studies in *C. elegans* [\(Flavell et al., 2013;](#page-11-7) [Ji et al., 2021\)](#page-11-8) have revealed a compact recurrent network governing the transitions between behavioral states analogous to exploration and exploitation (Fig. [1\)](#page-2-0). This minimal network provides a unique opportunity to explore the algorithmic principles the brain uses to solve the E-E problem.

### 3 MODEL

#### 3.1 THE BRAIN-INSPIRED BANDIT NETWORK (BBN) IS A STOCHASTIC CONTINUOUS HOPFIELD NETWORK

To model the biological neural network that controls E-E decisions during foraging (Fig. [8](#page-14-0) [\(Ji et al.,](#page-11-8) [2021\)](#page-11-8)), we define a set of N neurons whose temporal dynamics are described by the following stochastic differential equations (or Langevin equations):

$$\tau_i \frac{dx_i}{dt} = -\gamma_i x_i + \sum_{j \neq i}^N w_{ij} f(x_j) + b_i + \bar{I}_i + \sigma_i dW(t) \quad (1)$$

![](_page_2_Figure_1.jpeg)

Figure 1: The Brain-inspired Bandit Network (BBN) (a) Architecture of the 2-D BBN model. (b) Hopfield energy (or Lyapunov function) plotted over the state space of BBN. Heatmap represents the Hopfield energy; red and green curves are the nullclines, and white dots represent simulated network states. (c) Evolution of network states over time

Where f(x) = <sup>1</sup> 1+e−n(x−k) , wi,j < 0, and dW(t) is the Wiener process. Here, wijf(x<sup>j</sup> ) represents the inhibitory interaction between neurons; b<sup>i</sup> is the baseline activity of neuron i; I<sup>i</sup> and σidW(t) are the deterministic and the stochastic components of the external input, respectively. σ<sup>i</sup> is the standard deviation of the Wiener noise. We term this type of stochastic continuous Hopfield network with all negative weights the Brain-inspired Bandit Network (BBN), for reasons that will become clear later. Assuming approximately symmetric weights i.e., wij = wji [1](#page-2-1) , the deterministic part of the model is essentially a continuous Hopfield network [\(Hopfield, 1982;](#page-11-9) [1984\)](#page-11-10) with exclusively inhibitory connections. It is hence associated with a Hopfield energy or Lyapunov function of the form:

$$E = \left\{ -\frac{1}{2} \sum_{i,j,i \neq j}^N w_{ij} f(x_i) f(x_j) + \sum_i^N \left[ x_i f(x_i) - \int_0^{x_i} f(x) dx \right] - \sum_i^N b_i f(x_i) \right\} \\ - \left\{ \sum_i^N \bar{L}_i f(x_i) \right\} = E^{int} - E^{ext} \quad (2)$$

Here, we have decomposed the Hopfield energy E into Eint, dependent only on internal network parameters, and Eext, which embodies influence from external inputs I<sup>i</sup> . With suitable parameters (see Appendix [B.1\)](#page-18-0), the model can have up to N local energy minima or attractor states exhibiting winner-take-all dynamics (Fig. [1](#page-2-0) and Fig. [12\)](#page-17-0). Stochastic noise induces transitions between these attractor states, consistent with experimental findings in foraging networks [\(Ji et al., 2021\)](#page-11-8).

### 3.2 THE BBN IMPLEMENTS BAYESIAN POSTERIOR SAMPLING

Hinton and Sejnowski [\(Hinton & Sejnowski, 1983\)](#page-11-11) have demonstrated that a discrete Hopfield network with stochastically activating units (i.e. an Ising network) can implement Bayesian inference by sampling from the posterior distribution. Here we extend this conclusion to continuous Hopfield networks. Briefly, using Kramers' escape theory [\(Kramers, 1940;](#page-11-12) [Langer, 1968;](#page-11-13) [Hanggi et al.,](#page-11-14) ¨ [1990\)](#page-11-14), we can approximately compute the mean first passage time (MFPT), defined here as the expected time to leave an attractor state A and crossing the nearby saddle point S, as:

$$\langle \tau_A \rangle = \frac{2\pi\gamma}{\omega_b} \frac{\prod_i' \omega_i^S}{\prod_i \omega_i^A} * \exp \left( \frac{\Delta E_A}{D_A} \right) \quad (3)$$

<sup>1</sup>While the original Hopfield network study [\(Hopfield, 1984\)](#page-11-10) required weight symmetry to prove absolute stability of the energy (or Lyapunov) function. Later work [\(Matsuoka, 1992;](#page-11-15) [Chen & Amari, 2001\)](#page-10-13) have shown that the global convergence of the Hopfield energy function still holds for networks with asymmetric weights.

Where γ is the friction coefficient (equivalent to τ in Eq. [1,](#page-1-0) and ω A i are the angular frequencies (i.e. eigenvalues of the Hessian matrix) at the center (i.e. energy minimum) of the attractor. ω<sup>b</sup> and ω S i are the angular frequencies of the saddle point, with ω<sup>b</sup> associated specifically with the unstable mode. ∆E<sup>A</sup> is the energy difference between the saddle point and the center of the attractor and ∆EA→<sup>S</sup> = E<sup>S</sup> − EA. D<sup>A</sup> is the diffusion constant, which in thermodynamics scales with the magnitude of the stochastic noise.

The equilibrium probability of the network being in a given attractor state A<sup>1</sup> can be approximated by its stability, measured via the MFPT, relative to the other attractors. This translates to:

$$P_{A_1} \cong \frac{\langle \tau_{A1} \rangle}{\sum_1^N \langle \tau_{Aj} \rangle} = \frac{1}{1 + \sum_2^N \left\{ \frac{\alpha_j}{\alpha_1} \exp\left(\frac{\Delta E_{Aj}}{D_{Aj}} - \frac{\Delta E_{A1}}{D_{A1}}\right)\right\}}, \quad \text{where } \alpha_i \in \{1, \dots, N\} = \frac{\prod_j \omega_j^{S_i}}{\omega_b \prod_j \omega_j^{A_i}} \quad (4)$$

Assuming identical biophysical parameters and inputs for all neurons, the angular frequencies ω A<sup>i</sup> j of the N attractors are permutations of each other and there is a single saddle point defined by x = <sup>γ</sup> Nwf(x) +b<sup>+</sup> ¯I. This leads to <sup>α</sup><sup>1</sup> <sup>=</sup> <sup>α</sup><sup>j</sup> , <sup>∀</sup>i. Further, by substituting <sup>∆</sup>E<sup>A</sup> = (E<sup>S</sup> <sup>−</sup>Eint <sup>A</sup> ) +Eext A into Eq. [4,](#page-3-0) we have:

$$P_{A_1} \cong \frac{1}{1 + \sum_2^N \left\{ \exp \left( \left[ \frac{E_S - E_{Aj}^{\text{int}}}{D_{Aj}} - \frac{E_S - E_{A1}^{\text{int}}}{D_{A1}} \right] + \left[ \frac{E_{Aj}^{\text{ext}}}{D_{Aj}} - \frac{E_{A1}^{\text{ext}}}{D_{A1}} \right] \right) \right\}} \quad (5)$$

Now if we define the probability of an attractor state in the absence of external inputs as its prior probability as: P prior Ai = exp ∆Eint Ai /DAi , and the probability of the state given input data (e.g. sensory evidence) as: P ¯I | A<sup>i</sup> = exp (Eext Ai /DAi), we have:

$$P(A_1 | \mathbf{I}) \cong \frac{1}{1 + \sum_2^N \left\{ \left( P_{Aj}^{\text{prior}} / P_{A1}^{\text{prior}} \right) * [P(\mathbf{I} | A_j) / P(\mathbf{I} | A_1)] \right\}} \quad (6)$$

Eq. [6](#page-3-1) reveals a close connection between the Hopfield-energy-based formulation of attractor state probability and Bayesian inference. Specifically, if we consider PAi as the probability of a hypothesis i being true or a decision i being optimal, then Eq. [6](#page-3-1) essentially computes the Bayesian posterior of i given external evidence.

#### 3.3 THE BBN CAN EXHIBIT *OPTIMISTIC*, *NEUTRAL*, OR *CONSERVATIVE* BIASES ON INPUT UNCERTAINTY

In Kramers' theory, the diffusion constant D from thermal fluctuations is typically isotropic (Σ = σ 2 I, D = σ 2 ). However, in our model, input to each neuron can have different levels of uncertainty, making the overall noise anisotropic. Recent studies [\(Zhu et al., 2018;](#page-13-3) [Yang et al., 2023\)](#page-13-4) show that anisotropic noise affects escape efficiency (the rate at which model leaves one of its attractor states (i.e. 1/MFPT)) by interacting with local attractor curvature. Starting from a local energy minimum at x0, the model evolves as:

$$\langle E(x_t) \rangle \cong E(x_0) - \int_0^t \langle \nabla E^T \nabla E \rangle + \frac{t}{2} \langle \text{Tr}(\mathbf{H}_0 \mathbf{\Sigma}) \rangle \quad (7)$$

Here, H<sup>0</sup> is the Hessian matrix evaluated at the attractor bottom, and Σ is the noise covariance matrix. Since both matrices are diagonal in our model, the escape efficiency is highest when the dimensions of largest input noise align with those of highest curvature. To capture this effect, we define an isotropic noise Σ = σ 2 I that yields the same efficiency as Σ:

$$\text{Tr}(\mathbf{H}_i \overline{\mathbf{\Sigma}}) = 2\overline{\sigma_i}^2 \text{Tr}(\mathbf{H}_i) = \text{Tr}(\mathbf{H}_i \mathbf{\Sigma}), \quad \text{where } \overline{\sigma_i}^2 = \frac{\text{Tr}(\mathbf{H}_i \mathbf{\Sigma})}{\text{Tr}(\mathbf{H}_i)} = D_i^{\text{eff}} \quad (8)$$

Here, Deff i represents the effective diffusion constant and H<sup>i</sup> = PH<sup>j</sup> = HA, ∀i where P is a permutation matrix. Substituting Eq. [8](#page-3-2) into Eq. [4](#page-3-0) , we have:

$$P_{A1} = \frac{1}{1 + \exp \left\{ 2 \text{Tr}(\mathbf{H}_A) \Delta E_A \left( \frac{1}{\text{Tr}(\mathbf{H}_A^T \mathbf{\Sigma})} - \frac{1}{\text{Tr}(\mathbf{H}_A \mathbf{\Sigma})} \right) \right\}} \quad (9)$$

![](_page_4_Figure_3.jpeg)

Figure 2: BBN implements Bayesian posterior sampling with a tunable bias towards, neutral to, or against input uncertainty. (a) Sigmoidal dependence of attractor state probability on the difference in mean input values. (b) Slope of the state probability curve in (a) as a function of total input uncertainty (defined as p σ 2 <sup>1</sup> + σ 2 2 ) for the three types of networks. (c) Intercept of the state probability curve as a function of relative input uncertainty (defined as σ<sup>1</sup> − σ2).

While all N attractors have equal energy and share a common set of angular frequencies, their Hessian matrices are non-identical and can interact differently with non-isotropic noise (i.e.,Σ ̸= cI). If PA<sup>1</sup> corresponds to the attractor state with the highest input noise, the following scenarios can occur (assuming j ̸= 1):

1. Tr(H1Σ) < Tr(HjΣ) and PA<sup>1</sup> > PAj (Optimistic). 2. Tr(H1Σ) = Tr(HjΣ) and PA<sup>1</sup> = PAj (Neutral). 3. Tr(H1Σ) > Tr(HjΣ) and PA<sup>1</sup> < PAj (Conservative).

These regimes are termed as Optimistic, Neutral, and Conservative, respectively. Fig. [2](#page-4-0) illustrates the input dependence of attractor state probabilities under these three regimes.

Parameter sensitivity analyses (Fig. [3\(](#page-5-0)a-b) and Fig. [11](#page-16-0) in Appendix [A\)](#page-14-1) reveal that the three parameter regimes span a wide range of parameter combinations, obviating the need for fine-tuning. By adjusting the baseline activity b, synaptic threshold k, or inhibitory synaptic weight w — either individually or in pairs — one can flexibly modulate the uncertainty bias from highly optimistic (PA<sup>1</sup> → 1) to neutral (PA<sup>1</sup> = 1 N ) to highly conservative (PA<sup>1</sup> → 0).

#### 3.4 (OPTIMISTIC) UNCERTAINTY BIAS IS PRESERVED IN HIGHER DIMENSIONS

The theoretical analysis above predicts that the uncertainty bias of BBN should scale well to high dimensions. To verify this empirically, we progressively increased network dimension by adding more neurons, while keeping all network parameters in Eq. [1](#page-1-0) unchanged. Strikingly, for a BBN that is optimistic at N = 2, scaling up to N = 10 did not alter its optimistic bias (Fig. [3\(](#page-5-0)c)). In contrast, a BBN that is neutral in 2D became mildly optimistic as N increased, while a conservative BBN becomes mildly optimistic at N > 5. Thus, with increasing network dimension, the model exhibits a tendency to bias towards attractor states with higher input uncertainty.

To understand this empirical phenomenon, we examined state-transition dynamics near the saddle point for a perfectly neutral 3D BBN (i.e., H<sup>i</sup> = cI, ∀i) (Fig. [13\)](#page-18-1). With isotropic noise, the network exhibited equal probability of entering any attractor state. However, with highly anisotropic noise, it preferentially entered the attractor state along the dimension of highest noise, creating a bias towards high-uncertainty states. This makes conservative bias harder to maintain and optimistic bias more prominent in high-dimensional models (Fig. [14.](#page-18-2) To incorporate this effect into our theoretical framework, we need to combine escape rates analysis [\(Kramers, 1940;](#page-11-12) [Zhu et al., 2018\)](#page-13-3) with theory

![](_page_5_Figure_1.jpeg)

Figure 3: Parameter dependence and multi-dimensional model. (a) Theoretically derived and (b) numerically simulated attractor state probability as a function of network parameters *b* and *k*. The color scale corresponds to the probability that the network samples the attractor state driven by the highest input uncertainty, which is an indicator of the network's uncertainty bias. (c) Equilibrium attractor state probabilities in high dimensional BBN models. Three colored lines correspond to attractor states driven by the highest (orange), median (green), or lowest (blue) levels of input uncertainty. The network parameters remain unchanged as the dimensionality N increases.

of dynamics around saddle points [\(Daneshmand et al., 2018\)](#page-10-14)—a challenge we aim to address in future work.

### 4 EXPERIMENTAL EVALUATION

#### 4.1 UNCERTAINTY-AWARE EXPLORATION IN MULTI-ARMED BANDIT TASK

Given BBN's ability to infer and sample from a posterior distribution with a tunable uncertainty bias, a natural application of BBN is to control action choice given external, uncertain evidence. We thus adapted the BBN model to play multi-armed bandit (MAB) games and compared its performance with classic bandit algorithms.

#### 4.1.1 RUNNING BBN IN BANDIT GAMES

To make the BBN model play bandit games, we (1) define a BBN model with N neurons, each corresponding to one of the N bandit arms; (2) pick network parameters that yield "optimistic" exploration for a 2-D BBN, and simply apply the same parameters to all neurons in the N-D model; (3) prior to each bandit trial, assign network input I by sampling from the reward memory buffer and numerically simulate the network for T steps using the Runge-Kutta method; (4) at the end of simulation, select the arm a whose corresponding neuron has the highest activation value; (5) collect the reward r<sup>a</sup> and add it to the memory buffer for arm a; (6) repeat (3)-(5) for the next trial till the game ends. The pseudocode along with detailed task parameters are presented in Appendix [B.1.](#page-19-0)

#### 4.1.2 BBN IMPLEMENTS UNCERTAINTY-AWARE POSTERIOR SAMPLING

To reveal BBN's exploration strategies, we examined the dependence of choice probability on total and relative reward uncertainty for BBN agents with optimistic, neutral, or conservative biases, as well as classic algorithms Thompson Sampling (TS) and Upper Confidence Bound (UCB). As shown in Fig. [4](#page-6-0) (a-b), TS exhibited a constant intercept regardless of relative uncertainty (RU) and a decreasing slope with increasing total uncertainty (TU), indicating sensitivity only to total uncertainty; UCB exhibited a constant slope with varying TU and an increasing intercept with increasing RU, indicating sensitivity only to relative uncertainty. In contrast, BBN with optimistic parameters showed variation in both slope and intercept with changes in TU and RU. These results indicate that BBN implements a hybrid algorithm combining posterior sampling (like TS) with tunable bias towards high uncertainty (akin to UCB).

![](_page_6_Figure_1.jpeg)

Figure 4: Exploratory behavior of BBN, Thompson sampling and UCB in 2-armed bandit games (a) Slope of the choice probability curve as a function of total uncertainty. (b) Intercept of the choice probability curve as a function of relative uncertainty.

#### 4.1.3 EFFICIENT EXPLORATION IN BANDIT TASKS

We next compared the empirical performance of BBN-driven exploration in comparison against UCB, Thompson sampling, and Optimistic Thompson Sampling (OTS, [\(Hu et al., 2023\)](#page-11-16)) in both 2-armed bandit and 3-armed bandit games. Each agent played 10,000 game blocks of 20 trials each in 2-armed bandit games and 30 trials each in 3-armed bandit games. Fig. [5](#page-6-1) (a-b) presents the probability of choosing the optimal arm as trial number increases. BBN (with optimistic parameters) consistently outperformed other algorithms in 2-armed bandits and topped the performance in 3-armed bandit games. The other 'hybrid' algorithm, OTS, performed close to BBN in 3-armed bandits, but did poorly in 2-armed bandits.

![](_page_6_Figure_5.jpeg)

Figure 5: BBN achieves efficient exploration in both bandit tasks. (a) The probability of choosing optimal action over trials in 2-armed bandit games. (b) The probability of choosing optimal action over trials in 3-armed bandit games. (c) Cumulative regret in the SixArms (MDP task, see Fig. [16\)](#page-23-0)

### 4.2 BBN CLOSELY APPROXIMATES BANDIT CHOICE BEHAVIOR IN HUMANS AND ANIMALS

The results above indicate that BBN exhibits similar hybrid strategies as previously reported in humans [\(Wilson et al., 2014;](#page-12-8) [Gershman, 2018\)](#page-11-3). We thus asked whether BBN can accurately model human and animal choice patterns in bandit tasks. We first compiled several publicly available datasets of humans playing bandit games (detailed list in Appendix [C\)](#page-23-1). We performed optimization on two network parameters b and k to minimize the difference between the choice probability curves output by BBN and in the human datasets. As shown in Fig. [6](#page-7-0) (a-b), BBN can closely fit to both the intercept and the slope of human choice probability curves. In contrast, Thompson sampling failed to fit to the diverse intercepts across human groups, and UCB consistently yielded slopes that are much higher than those observed in human data.

We next extended the above analyses to a dataset in which mice played switching blocks of 2 armed bandit games [\(Beron et al., 2022\)](#page-10-10). In this dataset, the reward for each arm is sampled from a Bernoulli distribution. In addition, the mean reward for each arm is not static, with a small probability (0.02) of being reversed before each trial starts. Based on results from [\(Beron et al., 2022\)](#page-10-10),

![](_page_7_Figure_1.jpeg)

Figure 6: The choice pattern of BBN closely approximates humans and animals in MAB tasks. (a-b) BBN-fitted versus actual slope and intercept values extracted from human data. (c-d) The probability of choosing the optimal arm and switching to another arm upon block transition in mice playing the 2-armed bandit game.

we used the last five rewards as inputs to the BBN model to drive choice behavior. As shown in Fig. [6](#page-7-0) (c-d), parameter-tuned BBN generated choice and switching behavior that closely approximated those exhibited in the mice study.

### 4.3 EFFICIENT EXPLORATION IN MDP PROBLEMS

Building on the strong performance of BBN in MAB tasks, we next applied BBN to MDP problems. Unlike bandit problems with immediate rewards and no state transitions, MDP tasks require sequential decision-making under delayed rewards and unknown transition probabilities [\(Bellman, 1966;](#page-10-15) [Bertsekas, 2012\)](#page-10-16). Among existing methods, UCRL2 [\(Auer et al., 2008\)](#page-10-17) extends OFU to MDPs, while PSRL [\(Strens, 2000;](#page-12-12) [Osband et al., 2013\)](#page-12-13) generalizes posterior sampling to RL. Hybrid algorithms like Optimistic Thompson Sampling (OTS) [\(Agrawal & Jia, 2017;](#page-10-18) [Tiapkin et al., 2022;](#page-12-14) [Hu](#page-11-16) [et al., 2023\)](#page-11-16) aim to improve exploration efficiency but face challenges such as computational cost and uncertainty estimation.

We consider a finite-horizon MDP with state space S, action space A, horizon H, rewards r l sa, and transition probabilities Psa conditioned on states s, actions a, and step l. The expected total return at step l under policy π can be estimated iteratively using the Bellman equation:

$$Q_{sa}^{t+1} = \mu_{sa} + \sum_{s'a'} \pi_{s'a'} P_{sas'} Q_{s'a'}^t$$

where µ = <sup>E</sup>(r) is the mean reward. Estimating uncertainty in Q-values remains an open issue in RL. Donoghue et al. [\(O'Donoghue et al., 2018\)](#page-12-15) proposed the Uncertainty Bellman Equation (UBE) to provide an upper bound on the variance of Q-value posteriors. For tabular state space, this method effectively propagates local variance estimates to global value uncertainty.

### 4.3.1 RUNNING BBN IN MDP TASKS

To apply BBN to drive action-selection in MDP tasks, we (1) define a BBN model with N neurons, each corresponding to one of the N discrete actions, select network parameters that belong to the "optimistic" regime for a 2D network; (2) initialize state-action values to i.i.d. Gaussian distributions; (3) sample input values for each neuron from the distributions of state-action values and perform numerical simulation of the BBN network for T steps using the Runge-Kutta method; (4) at the end of the simulation, select action a whose corresponding neuron has the highest activation value; (5) collect the reward r<sup>a</sup> and move to the next state ; (6) Repeat (3)-(5) till the episode ends; (7) Update the distribution of state-action values using the Uncertainty Bellman Equation (UBE) algorithm[\(O'Donoghue et al., 2018\)](#page-12-15). (8) repeat (3)-(7) for the next episode till the game ends. We present the pseudo-code for the Algorithm [2](#page-22-0) in Appendix [B.4.](#page-20-0)

We first compared the exploration efficiency of the BBN-based algorithm (UBE BBN) on the SixArms [\(Strehl & Littman, 2008\)](#page-12-16) task, with additional implementation details presented in Appendix [B.5.](#page-22-1) We compare our model to PSRL[\(Osband et al., 2013\)](#page-12-13), UCRL2 [\(Auer et al., 2008\)](#page-10-17) and OTS-MDP [\(Hu et al., 2023\)](#page-11-16). We also specifically tested the role of BBN by replacing it with UCB (UBE UCB) or Thompson sampling (UBE TS). In PSRL, we maintain a Gaussian distribution for the rewards and a Dirichlet distribution for the transition probabilities. In the OTS-MDP and BBN models, we follow[\(Hu et al., 2023\)](#page-11-16) and limit our uncertainty estimation to the reward r for simplicity. As shown in Fig. [5](#page-6-1) (c), the cumulative regret is lowest in UBE-BBN, which demonstrates the potential of BBN in promoting highly efficient exploration.

#### 4.3.2 GRID WORLD

![](_page_8_Figure_4.jpeg)

Figure 7: BBN-enhanced RL agent exhibits efficient exploration in the FourRooms task. (a) The FourRooms environment. The agent starts at the red point and can receive a reward only at the blue point. (b) The percent of grids covered (i.e. the coverage rate) by agents driven by various exploration algorithms over the period of training. (c) Display of visitation counts over the course of training. (d) Visitation counts for the UBE-BBN agent with or without action persistence. (e) Number of episodes taken till first reaching the reward state for different agents. Pink and purple are the UBE-BBN agents with and without action persistence respectively. Blue is PSRL and green is UBE UCB

We next evaluated the exploration efficiency of BBN on sparse-reward MDP tasks, specifically the FourRooms task. In this task, an N-by-N grid world is divided into four compartments connected by narrow passages (Fig. [7](#page-8-0) (a)). The agent starts from the upper left corner (red dot) and explores the environment to learn state-action values. First, we conducted reward-free exploration by assuming no rewards at any state. Exploration efficiency was measured as the coverage rate (ratio of visited states to total states) over episodes. Fig. [7](#page-8-0) (b) shows that UBE-BBN achieved the fastest coverage rate among all methods. Fig. [7](#page-8-0) (c) provides examples of cumulative visitation counts for each method during training. We then varied the environment size and repeated the experiments. UBE-BBN scaled well with grid size, while other algorithms faltered (Fig. [19](#page-25-0) in Appendix [E\)](#page-24-0). Additional comparisons with more methods in different conditions are in Fig. [20](#page-26-0)[-23](#page-29-0) in Appendix [E.](#page-24-0) Trajectories (visitation counts in a single episode) in Fig. [24](#page-30-0) reveal that UBE-BBN excelled in extended deep exploration, covering hard-to-reach states effectively. Finally, we enhanced action persistence in UBE-BBN by allowing the BBN model to inherit activity states from the previous step (Fig. [25\)](#page-31-0). This modification leveraged the Hopfield network's persistence property, instilling action correlation within episodes. As shown in Fig. [25,](#page-31-0) adding persistence further boosted UBE-BBN's exploration efficiency in the FourRooms task at large grid sizes.

Parameter sensitivity in MDP tasks: We additionally performed parameter sensitivity analysis for the SixArms and FourRooms task (as shown in Fig. [18](#page-24-1) in Appendix [E.1\)](#page-24-2) and demonstrated that a broad range of "optimistic" network parameters yielded high performance on these tasks. Hence, optimistic BBN generally delivers good performance in these MDP tasks without requiring parameter fine-tuning.

# 5 DISCUSSION

We have demonstrated both theoretically and empirically that the BBN architecture can drive flexible and efficient exploration in ways similar to humans and animals. However, several limitations and open questions remain regarding its practical application. First, simulating the stochastic differential equations incurs high computational costs. This issue may be circumvented by analytically computing the attractor probabilities using Eq. [4](#page-3-0) or by employing neuromorphic hardware. Second, given the development of many hybrid TS and OFU methods in the RL community [\(Hu et al.,](#page-11-16) [2023;](#page-11-16) [Tiapkin et al., 2022;](#page-12-14) [Agrawal & Jia, 2017\)](#page-10-18), it's intriguing to consider what gives rise to BBN's superior performance. One possibility is that BBN, as a system of coupled Langevin equations, effectively implements Langevin sampling of the posterior distribution. Langevin sampling has been shown to enjoy faster mixing and convergence rates than other sampling methods and is particularly well-suited for approximate Bayesian inference [\(Welling & Teh, 2011\)](#page-12-17). Third, the current BBN algorithm lacks the ability to estimate uncertainty associated with state-action values, relying instead on a separate algorithm (in this case, the UBE) to generate value distributions. How biological neural networks compute and encode uncertainty remains an outstanding question, especially in sequential decision settings. Recent studies have suggested that a distributed population code [\(Dehaene et al., 2021\)](#page-11-17) or a spatiotemporal activity pattern could encode uncertainty levels [\(Savin &](#page-12-18) [Deneve, 2014\)](#page-12-18). We hope future experimental and theoretical studies will provide more insights into ` how the brain estimates and utilizes uncertainty. Lastly, given that humans and animals can flexibly modulate their uncertainty bias in a context-dependent manner, a valuable extension for the BBN algorithm would be to integrate contextual information into the network input. Expanding the BBN model to include upstream neurons found in the biological foraging network might help implement context-dependent E-E decisions (Fig. [8\)](#page-14-0).

# REFERENCES


[1] Merideth A Addicott, John M Pearson, Maggie M Sweitzer, David L Barack, and Michael L Platt. A primer on foraging and the explore/exploit trade-off for psychiatry research. *Neuropsychopharmacology*, 42(10):1931–1939, 2017. Rajeev Agrawal. Sample mean based index policies by o (log n) regret for the multi-armed bandit problem. *Advances in applied probability*, 27(4):1054–1078, 1995. Shipra Agrawal and Randy Jia. Optimistic posterior sampling for reinforcement learning: worstcase regret bounds. *Advances in Neural Information Processing Systems*, 30, 2017. Peter Auer, Nicolo Cesa-Bianchi, Yoav Freund, and Robert E Schapire. Gambling in a rigged casino: The adversarial multi-armed bandit problem. In *Proceedings of IEEE 36th annual foundations of computer science*, pp. 322–331. IEEE, 1995. Peter Auer, Thomas Jaksch, and Ronald Ortner. Near-optimal regret bounds for reinforcement learning. *Advances in neural information processing systems*, 21, 2008. Yikun Ban, Yuchen Yan, Arindam Banerjee, and Jingrui He. EE-net: Exploitation-exploration neural networks in contextual bandits. In *International Conference on Learning Representations*, 2022. URL [https://openreview.net/forum?id=X\\_ch3VrNSRg](https://openreview.net/forum?id=X_ch3VrNSRg). Frederic Bartumeus, Daniel Campos, William S Ryu, Roger Lloret-Cabot, Vicenc¸ Mendez, and Jordi ´ Catalan. Foraging success under uncertainty: search tradeoffs and optimal space use. *Ecology letters*, 19(11):1299–1313, 2016. Richard Bellman. Dynamic programming. *science*, 153(3731):34–37, 1966. Celia C Beron, Shay Q Neufeld, Scott W Linderman, and Bernardo L Sabatini. Mice exhibit stochastic and efficient action switching during probabilistic decision making. *Proceedings of the National Academy of Sciences*, 119(15):e2113961119, 2022. Donald A Berry and Bert Fristedt. Bandit problems: sequential allocation of experiments (monographs on statistics and applied probability). *London: Chapman and Hall*, 5(71-87):7–7, 1985. Dimitri Bertsekas. *Dynamic programming and optimal control: Volume I*, volume 4. Athena scientific, 2012. Olivier Chapelle and Lihong Li. An empirical evaluation of thompson sampling. *Advances in neural information processing systems*, 24, 2011. Eric L Charnov. Optimal foraging, the marginal value theorem. *Theoretical population biology*, 9 (2):129–136, 1976. Tianping Chen and Shun Ichi Amari. Stability of asymmetric hopfield networks. *IEEE Transactions on Neural Networks*, 12(1):159–163, 2001. Jeffrey Cockburn, Vincent Man, William A Cunningham, and John P O'Doherty. Novelty and uncertainty regulate the balance between exploration and exploitation through distinct mechanisms in the human brain. *Neuron*, 110(16):2691–2702, 2022. Jonathan D Cohen, Samuel M McClure, and Angela J Yu. Should i stay or should i go? how the human brain manages the trade-off between exploitation and exploration. *Philosophical Transactions of the Royal Society B: Biological Sciences*, 362(1481):933–942, 2007. Vincent D Costa and Bruno B Averbeck. Primate orbitofrontal cortex codes information relevant for managing explore–exploit tradeoffs. *Journal of Neuroscience*, 40(12):2553–2561, 2020. Vincent D Costa, Andrew R Mitz, and Bruno B Averbeck. Subcortical substrates of explore-exploit decisions in primates. *Neuron*, 103(3):533–545, 2019. Hadi Daneshmand, Jonas Kohler, Aurelien Lucchi, and Thomas Hofmann. Escaping saddles with stochastic gradients. In *International Conference on Machine Learning*, pp. 1155–1164. PMLR, 2018.

[2] Nathaniel D Daw, John P O'doherty, Peter Dayan, Ben Seymour, and Raymond J Dolan. Cortical substrates for exploratory decisions in humans. *Nature*, 441(7095):876–879, 2006. Guillaume P Dehaene, Ruben Coen-Cagli, and Alexandre Pouget. Investigating the representation of uncertainty in neuronal circuits. *PLOS Computational Biology*, 17(2):e1008138, 2021. Haoxue Fan, Samuel J Gershman, and Elizabeth A Phelps. Trait somatic anxiety is associated with reduced directed exploration and underestimation of uncertainty. *Nature Human Behaviour*, 7(1): 102–113, 2023. Steven W Flavell, Navin Pokala, Evan Z Macosko, Dirk R Albrecht, Johannes Larsch, and Cornelia I Bargmann. Serotonin and the neuropeptide pdf initiate and extend opposing behavioral states in

[3] c. elegans. *Cell*, 154(5):1023–1035, 2013. Samuel J Gershman. Deconstructing the human algorithms for exploration. *Cognition*, 173:34–42, 2018. Samuel J Gershman. Uncertainty and exploration. *Decision*, 6(3):277, 2019. Peter Hanggi, Peter Talkner, and Michal Borkovec. Reaction-rate theory: fifty years after kramers. ¨ *Reviews of modern physics*, 62(2):251, 1990. Geoffrey E Hinton and Terrence J Sejnowski. Optimal perceptual inference. In *Proceedings of the IEEE conference on Computer Vision and Pattern Recognition*, volume 448, pp. 448–453. Citeseer, 1983. Jeremy Hogeveen, Teagan S Mullins, John D Romero, Elizabeth Eversole, Kimberly Rogge-Obando, Andrew R Mayer, and Vincent D Costa. The neurocomputational bases of exploreexploit decision-making. *Neuron*, 110(11):1869–1879, 2022. John J Hopfield. Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the national academy of sciences*, 79(8):2554–2558, 1982. John J Hopfield. Neurons with graded response have collective computational properties like those of two-state neurons. *Proceedings of the national academy of sciences*, 81(10):3088–3092, 1984. Bingshan Hu, Tianyue H Zhang, Nidhi Hegde, and Mark Schmidt. Optimistic thompson samplingbased algorithms for episodic reinforcement learning. In *Uncertainty in Artificial Intelligence*, pp. 890–899. PMLR, 2023. Ni Ji, Gurrein K Madan, Guadalupe I Fabre, Alyssa Dayan, Casey M Baker, Talya S Kramer, Ijeoma Nwabudike, and Steven W Flavell. A neural circuit for flexible control of persistent behavioral states. *Elife*, 10:e62889, 2021. Hendrik Anthony Kramers. Brownian motion in a field of force and the diffusion model of chemical reactions. *Physica*, 7(4):284–304, 1940. Tze Leung Lai and Herbert Robbins. Asymptotically efficient adaptive allocation rules. *Advances in applied mathematics*, 6(1):4–22, 1985. JS Langer. Theory of nucleation rates. *Physical Review Letters*, 21(14):973, 1968. Veldon-James Laurie, Akram Shourkeshti, Cathy S Chen, Alexander B Herman, Nicola M Grissom, and R Becket Ebitz. Persistent decision-making in mice, monkeys, and humans. *bioRxiv*, pp. 2024–05, 2024. Kiyotoshi Matsuoka. Stability conditions for nonlinear continuous neural networks with asymmetric connection weights. *Neural networks*, 5(3):495–500, 1992. Benedict C May, Nathan Korda, Anthony Lee, David S Leslie, and Nicolo Cesa-Bianchi. Optimistic bayesian sampling in contextual-bandit problems. *Journal of Machine Learning Research*, 13(6), 2012.

[4] Jack-Morgan Mizell, Siyu Wang, Alec Frisvold, Lily Alvarado, Alex Farrell-Skupny, Waitsang Keung, Caroline E Phelps, Mark H Sundman, Mary-Kathryn Franchetti, Ying-hui Chou, et al. Differential impacts of healthy cognitive aging on directed and random exploration. *Psychology and Aging*, 39(1):88, 2024. Ian Osband, Daniel Russo, and Benjamin Van Roy. (more) efficient reinforcement learning via posterior sampling. *Advances in Neural Information Processing Systems*, 26, 2013. Brendan O'Donoghue, Ian Osband, Remi Munos, and Volodymyr Mnih. The uncertainty bellman equation and exploration. In *International conference on machine learning*, pp. 3836–3845, 2018. Herbert Robbins. Some aspects of the sequential design of experiments. 1952. Cristina Savin and Sophie Deneve. Spatio-temporal representations of uncertainty in spiking neural ` networks. *Advances in neural information processing systems*, 27, 2014. Eric Schulz and Samuel J Gershman. The algorithmic architecture of exploration in the human brain. *Current opinion in neurobiology*, 55:7–14, 2019. Eric Schulz, Charley M Wu, Azzurra Ruggeri, and Bjorn Meder. Searching for rewards like a child ¨ means less generalization and more directed exploration. *Psychological science*, 30(11):1561– 1572, 2019. David W Stephens and John R Krebs. *Foraging theory*, volume 6. Princeton university press, 1986. Alexander L Strehl and Michael L Littman. An analysis of model-based interval estimation for markov decision processes. *Journal of Computer and System Sciences*, 74(8):1309–1331, 2008. Malcolm Strens. A bayesian framework for reinforcement learning. In *ICML*, volume 2000, pp. 943–950, 2000. Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. *Robotica*, 17(2): 229–235, 1999. Richard S Sutton and Andrew G Barto. *Reinforcement learning: An introduction*. MIT press, 2018. William R Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika*, 25(3-4):285–294, 1933. Daniil Tiapkin, Denis Belomestny, Daniele Calandriello, Eric Moulines, Remi Munos, Alexey Naumov, Mark Rowland, Michal Valko, and Pierre Menard. Optimistic posterior sampling for re- ´ inforcement learning with few samples and tight guarantees. *Advances in Neural Information Processing Systems*, 35:10737–10751, 2022. Momchil S Tomov, Van Q Truong, Rohan A Hundia, and Samuel J Gershman. Dissociable neural correlates of uncertainty underlie different exploration strategies. *Nature communications*, 11(1): 2371, 2020. James A Waltz, Robert C Wilson, Matthew A Albrecht, Michael J Frank, and James M Gold. Differential effects of psychotic illness on directed and random exploration. *Computational Psychiatry (Cambridge, Mass.)*, 4:18, 2020. Siyu Wang, Blake Gerken, Julia R Wieland, Robert C Wilson, and Jean-Marc Fellous. The effects of time horizon and guided choices on explore–exploit decisions in rodents. *Behavioral neuroscience*, 137(2):127, 2023. Max Welling and Yee W Teh. Bayesian learning via stochastic gradient langevin dynamics. In *Proceedings of the 28th international conference on machine learning (ICML-11)*, pp. 681–688. Citeseer, 2011. Robert C Wilson, Andra Geana, John M White, Elliot A Ludvig, and Jonathan D Cohen. Humans use directed and random exploration to solve the explore–exploit dilemma. *Journal of experimental psychology: General*, 143(6):2074, 2014.

[5] Robert C Wilson, Elizabeth Bonawitz, Vincent D Costa, and R Becket Ebitz. Balancing exploration and exploitation with information and randomization. *Current opinion in behavioral sciences*, 38: 49–56, 2021. Ning Yang, Chao Tang, and Yuhai Tu. Stochastic gradient descent introduces an effective landscapedependent regularization favoring flat solutions. *Physical Review Letters*, 130(23):237101, 2023. Wojciech K Zajkowski, Malgorzata Kossut, and Robert C Wilson. A causal role for right frontopolar cortex in directed, but not random, exploration. *Elife*, 6:e27430, 2017. Weitong Zhang, Dongruo Zhou, Lihong Li, and Quanquan Gu. Neural thompson sampling, 2021. URL <https://arxiv.org/abs/2010.00827>. Dongruo Zhou, Lihong Li, and Quanquan Gu. Neural contextual bandits with ucb-based exploration, 2020. URL <https://arxiv.org/abs/1911.04462>. Zhanxing Zhu, Jingfeng Wu, Bing Yu, Lei Wu, and Jinwen Ma. The anisotropic noise in stochastic gradient descent: Its behavior of escaping from sharp minima and regularization effects. *arXiv preprint arXiv:1803.00195*, 2018.
# APPENDIX

# A SUPPLEMENTAL FIGURES

![](_page_14_Diagram_3.jpeg)

Figure 8: (a) A biological neural network in *C. elegans* that controls the exploration state (roaming) and exploitation state (dwelling). (b) Architecture of the 3-D BBN model.

![](_page_14_Figure_5.jpeg)

Figure 9: Stability analysis on the three types of BBN models that generate *optimistic*, *neutral* or *conservative* bias to uncertainty.

![](_page_15_Figure_1.jpeg)

Figure 10: Slope and intercept shift. (Left column) The slope decreases as the total uncertainty increases while relative uncertainty is kept unchanged. (Right column) The intercept increases (optimistic), stays unchanged (neutral), or decreases (conservative) as relative uncertainty increases and total uncertainty kept unchanged.

![](_page_16_Figure_1.jpeg)

Figure 11: Attractor state probability as a function of network parameters Warm colors indicate a higher chance of finding the network in the state that receives greater input uncertainty.

![](_page_17_Figure_1.jpeg)

Figure 12: Uncertainty bias in multi-dimensional BBN models. (Top row) State dynamics of 3-D BBN model with conservative, neutral and optimistic uncertainty biases. The concentration of state dynamics reveal the three attractor states, which are visited with different relative proportion in the three types of BBNs. Input noise is strongest along the Z-direction is the largest and lowest along the X-direction. (Middle row) Probability of attractor states with the highest (orange), median (green), and lowest (blue) input uncertainty as the network scales from 2D to 10-D under the conservative, neutral, or optimistic parameter regimes, computed from numerical simulations. For the same type of network, internal model parameters are kept the same as the dimensionality increases. The dotted curve indicates perfectly equal partition of probability among all N states. (Bottom row) Theoretically predicted state probability for the same network models presented above in the middle row, presented in the same format.

![](_page_18_Figure_1.jpeg)

Figure 13: State entry dynamics near the saddle point for a 3D BBN. Red points show simulated state dynamics when the network is initialized from the saddle point. Green circle denotes the saddle point, green crosses denote the attractor centers, and the projected 2D histogram reveal the relative occupancy of the three attractors (pink indicates high state probability).

![](_page_18_Figure_3.jpeg)

Figure 14: Theoretical vs. simulated attractor state probability in multi-dimensional BBNs.

# B METHOD DETAILS

### B.1 PARAMETER SELECTION

| Parameter | Definition                 | Suggested range                             |
|-----------|----------------------------|---------------------------------------------|
| w         | inhibitory weights         | [2, 4], increase to make states more stable |
| b         | activity baseline          | [5.5, 7], increase makes network optimistic |
| k         | threshold of sigmoid       | [6, 8], increase makes network conservative |
| n         | slope of sigmoid           | [1, 2], increase amplifies uncertainty bias |
| γ         | leak current or decay rate | 0.5                                         |
| τ         | time constant              | 1                                           |

Table 1: Internal Model Parameters

| Parameter | Definition        | Suggested range                    |
|-----------|-------------------|------------------------------------|
| ¯ I       | mean of input     | scale raw input values to [-2, 2]  |
| 2         | variance of input | scale raw input variance to [0, 2] |

Table 2: External Parameters

| Parameter | Definition             | Suggested range                                |
|-----------|------------------------|------------------------------------------------|
| N         | number of neurons      | typically equal to number of actions choices   |
| T         | total simulation steps | [400, 1000]                                    |
| dt        | step length            | 0.1 or 0.2 if using suggested parameter ranges |

Table 3: tab: Hyperparameters

In this section, we list the primary parameters used in BBN (Tables [1,](#page-18-3) [2,](#page-18-4) [3\)](#page-19-1) and provide a principled way to determine optimal parameters for new environments.

Based on our experience and past literature [\(May et al., 2012;](#page-11-2) [Hu et al., 2023;](#page-11-16) [Agrawal & Jia, 2017\)](#page-10-18), optimistic bias generally promotes efficient exploration. In addition, our sensitivity analysis on MDP tasks (Fig. [18\)](#page-24-1) showed that a broad range of "optimistic" parameters yielded high performance, obviating the need for extensive fine-tuning. Further, we have shown that network parameters that yield optimistic bias for a 2D BBN preserve such bias in higher dimensions (Fig. [3\(](#page-5-0)c) and Fig. [12\)](#page-17-0). Thus, the steps to set up a N-dimensional BBN model are:

(1) Define a BBN model with N interconnected neurons;

(2) Select internal network parameters [1](#page-18-3) from the "optimistic" regime based on sensitivity analysis results presented in (Fig. [3\(](#page-5-0)a-b) and Fig. [11\)](#page-16-0), or use the parameter ranges suggested below as a starting point;

(3) Verify that the 2D network has two attractors and exhibits optimistic bias by numerically simulating the model under anisotropic 2D Gaussian noise with µ = [0, 0],σ = [1, 0.1]); tune the parameters if necessary using the tips provided below;

(4) Apply these parameters to all neurons in the ND network;

(5) Scale the input to the network (typically past rewards or Q-values) to a range that permits the existence of multiple attractors (use suggested range or verify empirically).

We found that simulation step number of T=400 is sufficient for bandit and MDP tasks t. Below are sample network dynamics in the first episode of a 2-armed bandit game. Multiple transitions occurred between the attractor states, reflecting equal state probability as expected for equal uncertainty for the two arms.

![](_page_19_Figure_11.jpeg)

Figure 15: State dynamics of BBN in a two-armed bandit game

### B.2 RUNNING BBN IN BANDIT GAMES

To make the BBN model play bandit games, we

- (1) define a BBN model with N neurons, each corresponding to one of the N bandit arms;
- (2) pick network parameters that yield "optimistic" exploration for a 2-D BBN, and simply apply the parameters to all neurons in the N-D model;
- (3) at each trial, sample input I from the reward memory buffer and numerical simulation of the network for T steps using the Runge-Kutta method;
- (4) at the end of the simulation, select the arm a whose corresponding neuron has the highest activation value;
- (5) collect the reward r<sup>a</sup> and add it to memory buffer for arm a;
- (6) repeat (3)-(5) for the next trial till game ends.

### B.3 RUNNING BBN IN MDP TASKS

Here we consider the tabular case MDP, so the states and Q-values are parameterized as entries in a lookup table, where each state-action pair maps to a Q-value. To implement BBN in action selection, the agent needs to estimate the uncertainty of Q-values for BBN's input. However, how to estimate uncertainty of the cumulative rewards in MDP tasks remains an open issue in the RL community because the choice of an action affects both the current immediate reward and subsequent state transfer. [O'Donoghue et al.](#page-12-15) [\(2018\)](#page-12-15) gave an upper bound on the variance of posterior distribution of the Q-values by proposing the Uncertainty Bellman Equation (UBE), which connects the uncertainty at any time-step to the expected uncertainties at subsequent time-steps. We leverage the upper bound on the variance by UBE to obtain uncertainty estimation for Q-values.

Here we present detailed steps to apply BBN to drive action selection in MDP tasks: (1) define a BBN model with N neurons, each corresponding to one of the N discrete actions, select network

- parameters that belong to the "optimistic" regime for a 2D network;
- (2) initialize state-action values to i.i.d. Gaussian distributions;
- (3) sample input values for each neuron from the distributions of state-action values and perform numerical simulation of the BBN network for T steps using the Runge-Kutta method;
- (4) at the end of the simulation, select action a whose corresponding neuron has the highest activation value;
- (5) collect the reward r<sup>a</sup> and move to the next state ;
- (6) Repeat (3)-(5) till the episode ends;
- (7) Update the distribution of state-action values using the uncertainty bellman equation (UBE) algorithm[\(O'Donoghue et al., 2018\)](#page-12-15).
- (8) repeat (3)-(7) for next episode till game ends. We present the pseudo-code for the Algorithm [2](#page-22-0) in Appendix [B.4.](#page-20-0) The pseudocode along with detailed task parameters are presented below.

### B.4 PSEUDOCODES

Algorithm 1: BBN for multi-armed bandit games

Input :

The horizon of the multi-armed bandit game H;

The number of arms A ;

The total simulation steps for BBN model T;

Output:

The selected arm a at each trial h; Initialize the model parameter for BBN model;

Initialize the value for each neuron x<sup>i</sup>

;

for *h = 1, 2, ..., H* do for *t = 1, 2, ..., T* do

sample I<sup>i</sup> from reward history for each arm a<sup>i</sup>

τi dx<sup>i</sup>

dt ← −γix<sup>i</sup> +

P N j̸=i

wijf(x<sup>j</sup> ) + b<sup>i</sup> + I<sup>i</sup>

;

x<sup>i</sup> ← x<sup>i</sup> + dx<sup>i</sup>

;

end

select an arm a ← argmax(xi); receive a reward r<sup>a</sup> ∼ N (µa, σ<sup>2</sup>

a );

add r<sup>a</sup> to reward history of arm a;

end

Algorithm 2: UBE-BBN for MDP tasks

Input : The horizon of the MDP task H; The maximum episode τ ; The number of total states S ; The number of actions A ; The total simulation steps for BBN model T; Output: The selected action a at each timestep t; Initialize the model parameter for BBN model; Initialize the value for each neuron x<sup>i</sup> ; for *iter = 1, 2, ...,* τ do for *h = 1, 2, ..., H* do s ← current state ; for *t = 1, 2, ..., T* do sample I<sup>i</sup> from Qsi ∼ N (Qˆ si, varQˆ si) ; τi dx<sup>i</sup> dt ← −γix<sup>i</sup> + P N j̸=i wijf(x<sup>j</sup> ) + b<sup>i</sup> + I<sup>i</sup> ; x<sup>i</sup> ← x<sup>i</sup> + dx<sup>i</sup> ; end select an action a ← argmax(xi); receive a reward rsa ∼ N (µsa, σ<sup>2</sup> sa); move to next state s ′ ; update Pˆ sas′ ; end update Q values using dynamic programming: for *h = H, H-1, ..., 1* do for *s* ∈ *S* do for *a* ∈ *A* do µˆsa ← Ersa; Qˆ<sup>h</sup> sa ← µˆsa + P s ′a′ π<sup>s</sup> ′a′Pˆ sas′Q h+1 s ′a′ ; employ the Uncertainty Bellman Equation (UBE): varQˆ<sup>h</sup> sa ← varµˆsa + P s ′a′ π<sup>s</sup> ′a′Psas′varQˆh+1 s ′a′ ; end end end end

### B.5 BANDIT AND MDP TASK PARAMETERS

Bandit parameters for performance comparison We chose to use Gaussian bandits where reward values are sampled from N (µ<sup>i</sup> , σ<sup>2</sup> i ). For 2-armed bandit games, the reward mean µ for both arms are sampled from a Gaussian distribution N (0, 1 ) at the beginning of each block. The reward variance is 9 and 4 respectively. For 3-armed bandit games, the reward mean µ for all arms are sampled from a Gaussian distribution N (0, 1 2 ) at the beginning of each block. The reward variance are 9,1,0.25 respectively. Note that while we chose to use Gaussian bandits here, the model can be extended to non-Gaussian input distributions and performs well empirically in non-Gaussian (e.g. Bernoulli) bandit tasks.

Bandit parameters for fitting to mice data We follow the bandit parameters in [\(Beron et al.,](#page-10-10) [2022\)](#page-10-10). The mean rewards of the Bernoulli bandits are 0.8 and 0.2 respectively.

SixArms SixArms[\(Strehl & Littman, 2008\)](#page-12-16) consists of seven states and six actions. The agent starts in state 0. We consider episodic case, so the state is reset every 20 steps. A transition is of the form (a, p, r), where a is action, p is the transition probability, and r is the reward for taking the transition.

![](_page_23_Diagram_2.jpeg)

Figure 16: SixArms.

For more detailed parameters for each algorithm used in our experiments, please refer to our code: <https://github.com/anonymousforICLR/BrainBandit>

# C DATASETS FOR MODEL FITTING

Gershman19 is from [\(Gershman, 2019\)](#page-11-18). In their experiment, participants were given a choice between two arms, labeled either as "safe" (S) or "risky" (R). The safe arms always return deterministic rewards, while the risky arms sample rewards from a Gaussian distribution. There are four types of bandit settings: RS, SR, RR, and SS, which are denoted by compound labels (e.g., "SR" denotes trials in which the left arm is safe and the right arm is risky). The reward mean µ for both risky arms and safe arms are sampled from a Gaussian distribution N (0, 10<sup>2</sup> ) at the beginning of each block. The reward variance for risky arms is 16, and for safe arm is 0. By comparing the slope and intercept of the choice probability curve for each type, we can quantify the degree of randomness and preference for uncertainty.

Fan23[\(Fan et al., 2023\)](#page-11-4) further explored the relationship between trait somatic anxiety and different exploration strategies in decision-making. They used the same experimental design as Gershman19[\(Gershman, 2019\)](#page-11-18) and evaluated the anxiety for each individual. In Fig [6](#page-7-0) (a-b), the slope and intercept of human data in [\(Gershman, 2019\)](#page-11-18) are drawn directly from the paper. And for humans with high or low anxiety, we split the 40% of the population with the highest "somatic anxiety" score and the 40% with the lowest "somatic anxiety" score in the collected data from [\(Fan et al.,](#page-11-4) [2023\)](#page-11-4), and then performed probit regression respectively.

Mizell24 from [\(Mizell et al., 2024\)](#page-12-9) involved younger adults (ages 18–25) and older adults (ages 65–74) making decisions between two virtual slot machines to measure exploration behaviors called Horizon Task. The rewards are sampled from a Gaussian distribution. Participants first completed instructed trials, sampling the slot machines under two conditions: unequal information (one drawn from one machine and three from the other) and equal information (two drawn from each machine). They then made free choices in either a short horizon (one choice) or a long horizon (six choices) condition. The task assessed directed exploration (choosing the more informative option) and random exploration (choosing the lower reward option). We use unequal information condition of the collected data to fit our model.

Zajkowsk17 is from [\(Zajkowski et al., 2017\)](#page-13-5). Participants also performed a Horizon Task, where they made explore-exploit decisions between two virtual slot machines under two conditions: unequal information and equal information. The task involved 160 games, each consisting of 5 or 10 choices, with the key manipulation being the horizon length: short (5 choices) or long (10 choices). Continuous theta-burst transcranial magnetic stimulation (TMS) was used to selectively inhibit the right frontopolar cortex (RFPC) when participants performed the Horizon Task. We use unequal information condition of the collected data to fit our model.

### D FURTHER RESULTS ON BANDIT TASKS

### D.1 LIMITED MEMORY BUFFER SIZE

BBN doesn't need all the past experience in the memory buffer. For example, in the experiment of fitting to mice behavior, we only used the last 5 reward histories since the reward for each bandit will change over time. We also performed additional experiments to test if the limited memory buffer would hurt the performance in bandit tasks. We limited the buffer size to 8 for each arm. Fig [17](#page-24-3) shows BBN with limited memory buffer size still consistently outperforms other methods.

![](_page_24_Figure_4.jpeg)

Figure 17: BBN with limited memory buffer size achieve similar efficient exploration in bandit tasks

# E FURTHER RESULTS ON MDP TASKS

### E.1 PARAMETER SENSITIVITY ANALYSIS

![](_page_24_Figure_8.jpeg)

Figure 18: Parameter sensitivity analysis of UBE-BBN with different parameter combinations evaluated in two MDP tasks. Performance in the SixArms task was evaluated by the cumulated regret of the agent, while performance in the FourRooms grid world task was evaluated by the coverage rate. a broad range of "optimistic network parameters" generally yielded high performance on these tasks.

### E.2 PERFORMANCE ON VARIATIONS OF GRID WORLD TASKS

As shown in Fig[.19,](#page-25-0) UBE-BBN yields fastest coverage rate among all the methods on different environments. Fig[.20](#page-26-0) gives examples of cumulative visitation counts for more algorithms during training. Only UBE-BBN covers all states with less than 450 episodes.

![](_page_25_Figure_3.jpeg)

Figure 19: Learning curves on different sizes of FourRooms environments.

Fig[.24](#page-30-0) shows the trajectories of agents, which are the visitation counts in a single episode. As shown, ϵ-greedy, UCRL2, UBE-TS only perform exploration around the starting state, failing to do "deep" exploration. PSRL, OTS-MDP and UBE-UCB can perform "deep" exploration, but they all act deterministically, so they will be stuck at a certain state. UBE-BBN is also driven by uncertainty like UBE-UCB to perform "deep" exploration, but with stochastic sampling of action choices, it will not be stuck at a certain state.

Action persistence further boosts BBN performance. BBN with persistence refers to taking neuron values at the end of last step as the starting point for the next step, while BBN without persistence refers to initializing neuron values at each step. We compare the different behavior of the BBN model with and without persistence across four different grid sizes: 15×15, 19×19, 23×23, and 103×103. The results presented here show the trajectories during the first episode of exploration, and the exploration length corresponds to the number of states in the grid world. As shown in Fig. [25,](#page-31-0) for the same exploration length, the BBN model with persistence explores a larger portion of the grid world.

![](_page_26_Figure_1.jpeg)

Figure 20: Comparison of exploration efficiency across different exploration algorithms in Four Rooms) (a) Visitation counts in reward free setting; (b) Number of episodes until first encounter of the reward state.

![](_page_27_Figure_1.jpeg)

Figure 21: Trajectories (visitation counts in a single episode) of UBE-TS, UCRL2, and UBE-BBN in expanded Four Rooms task with reward

![](_page_28_Figure_1.jpeg)

Figure 22: Comparison of visitation counts across algorithms in a Four Rooms game with reward and penalty).

![](_page_29_Figure_1.jpeg)

Figure 23: Comparison of visitation counts across algorithms in a Nine Rooms game.

![](_page_30_Figure_1.jpeg)

Figure 24: Agent trajectories (visualized through visitation counts) in single episodes over the course of training.

![](_page_31_Figure_1.jpeg)

Figure 25: Trajectories (visitation counts in a single episode) of BBN with/without persistence.