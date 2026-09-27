# Heterogeneity In Multi-Agent Reinforcement Learning

Anonymous Author(s)
Affiliation Address email

## Abstract

1 *Heterogeneity* is a fundamental property in multi-agent reinforcement learning 2 (MARL), which is closely related not only to the functional differences of agents, 3 but also to policy diversity and environmental interactions. However, the MARL 4 field currently lacks a rigorous definition and deeper understanding of heterogeneity. 5 This paper systematically discusses heterogeneity in MARL from the perspectives 6 of definition, *quantification*, and *utilization*. First, based on an agent-level modeling 7 of MARL, we categorize heterogeneity into five types and provide mathematical 8 definitions. Second, we define the concept of heterogeneity distance and propose 9 a practical quantification method. Third, we design a heterogeneity-based multi10 agent dynamic parameter sharing algorithm as an example of the application of our 11 methodology. Case studies demonstrate that our method can effectively identify 12 and quantify various types of agent heterogeneity. Experimental results show that 13 the proposed algorithm, compared to other parameter sharing baselines, has better 14 interpretability and stronger adaptability. The proposed methodology will help the 15 MARL community gain a more comprehensive and profound understanding of 16 heterogeneity, and further promote the development of practical algorithms. 18 Multi-agent reinforcement learning (MARL) has achieved success in various real-world applications, 19 such as swarm robotic control [Kalashnikov et al., 2018], autonomous driving [Zhou et al., 2021], 20 and large language model fine-tuning [Ma et al., 2024]. However, most MARL studies focus on 21 policy learning for homogeneous multi-agent systems (MAS), overlooking in-depth discussions 22 of heterogeneous multi-agent scenarios [Ning and Xie, 2024]. *Heterogeneity* is a common phe23 nomenon in multi-agent systems. For example, in nature, different species of fish collaborate to 24 find food [Burns et al., 2019]; in human society, diverse teams demonstrate higher intelligence 25 and resilience [Dall'Anese et al., 2013, Young, 1993]; and in artificial systems, aerial drones and 26 ground vehicles cooperate to monitor forest fires [Lwowski et al., 2017]. Heterogeneity can enhance 27 system functionality, reduce costs, and improve robustness, but effectively leveraging heterogeneity 28 remains a key challenge in multi-agent system [Bennett, 2024]. As an approach of learning through 29 environmental interactions, MARL can effectively enable multi-agent systems to learn collabora30 tive policies. Hence, exploring heterogeneity from a reinforcement learning perspective would 31 significantly broaden the applicability of MARL. 32 In the current MARL field, although some works explicitly or implicitly mention agent hetero33 geneity, only a few focus on its definition and identification. Regarding explicit discussion of 34 heterogeneity, studies have explored communication issues [Seraj et al., 2021], credit assign35 ment [Yu et al., 2024], and zero-shot generalization [Guo et al., 2024] in heterogeneous MARL. 36 However, these works limit their focus to agents with clear functional differences and lack def37 initions of agent heterogeneity. On the other hand, many studies explore policy diversity in 38 MARL. Some encourage agents to learn distinguishable behaviors based on identity or trajec-

## 17 **1 Introduction**

Overall 39 tory information [Jiang and Lu, 2021, Li et al., 2021], some works group agents using specific 40 metrics [Wang et al., 2021, Christianos et al., 2021], and some quantify policy differences [Bettini 41 et al., 2023b, Hu et al., 2024] and design algorithms to control policy diversity [Bettini et al., 2024]. 42 However, these works do not adequately address 43 where policy diversity originates or how it fundamen44 tally relates to agent differences. In terms of defin45 ing and classifying heterogeneity in MARL, [Bettini 46 et al., 2023a] divides heterogeneity into physical and 47 behavioral types but lacks a mathematical definition.

48 [Seraj et al., 2021] provides extended POMDP for 49 heterogeneous MARL settings, but do not classify or 50 define heterogeneity. Others introduce the concept 51 of local transition heterogeneity [Yu et al., 2024], but 52 does not cover all elements of MARL. Overall, het53 erogeneity is not only a characteristic that exists in 54 MAS with traditional functional differences, but also 55 a fundamental property across the entire MARL field. 56 Currently, there is still a lack of systematic analysis 57 *of agent heterogeneity from the MARL perspective*. 58 To fill the aforementioned gaps, we conduct a se59 ries of studies on defining, quantifying, and utilizing 60 heterogeneity from the perspective of MARL, the phi61 losophy of our study can be found in Figure 1. Our 62 contributions are summarized as follows: 63 - **Defining Heterogeneity:** Based on an agent-level model of MARL, we categorize heterogeneity 64 into observation heterogeneity, response transition heterogeneity, effect transition heterogeneity, 65 objective heterogeneity, and policy heterogeneity, and provide corresponding definitions. 66 - **Quantifying Heterogeneity:** We define the heterogeneity distance, and propose a quantification 67 method based on representation learning, applicable to both model-free and model-based settings. 68 Additionally, we give the concept of meta-transition heterogeneity to quantify agents' comprehensive 69 heterogeneity. 70 - **Utilizing Heterogeneity:** We develop a multi-agent dynamic parameter-sharing algorithm based on 71 heterogeneity quantification, which offers better interpretability and fewer task-specific hyperparame72 ters compared to other related parameter-sharing algorithms.

Defining *Heterogeneity*

```
· Policy-Het
· Objective-Het
· Effect-Het
                             Het
                         in MARL
                                    Quantifying Heterogeneity

```

· Observation-Het
· Response-Het

```
· Het-based Dynamic
 Parameter Sharing
                           · Heterogeneity
                               Distance
                    · Model-based / free
                          Measuring
                  · Representation
                      Learning

```

· Other *Approaches…*
Utilizing *Heterogeneity* Diversity-based MARL
Figure 1: Our Philosophy. We aim to systematically discuss heterogeneity in MARL, establishing methodologies for defining, quantifying and utilizing heterogeneity.

## 79 **2 Preliminaries**

80 **Primal Problem of MARL.** In this paper, we use Partially Observable Markov Game 81 (POMG) [Littman, 1994, Kochenderfer et al., 2022] as the general model for the primal problem of MARL.1 82 To better study agent heterogeneity, we adopt an agent-level modeling approach 83 similar to that in [Seraj et al., 2021, Gronauer and Diepold, 2022]. A POMG is defined as an 8-tuple, 84 represented as follows:
POMG := ⟨N, {S
i}i∈N , {O
i}i∈N , {A
i}i∈N , {Ω
i}i∈N , {T i}i∈N , {ri}i∈N , γ⟩, (1)
73 In this paper, we adopt a discussion approach that progresses *from theory to practice* and *from* 74 *general to specific*. The overall structure is organized as follows: Section 2 introduces the agent-level 75 modeling of the MARL primal problem; Section 3 provides the classification and definition of 76 heterogeneity in MARL; Section 4 proposes the method for quantifying heterogeneity and presents 77 case studies; Section 5 describes the dynamic parameter-sharing algorithm; Section 6 provides the 78 related experimental results; and Section 7 summarizes the entire paper.

Among all elements in equation 1, N is the set of all agents, {S
i 85 }i∈N is the global state space which can be factored as {S
i}i∈N = ×i∈N S
i × S
E, where S
iis the state space of an agent i, and S
E 86 is the environmental state space, corresponding to all the non-agent components. {Oi}i∈N = ×i∈N Oi 87 is 1POMG is an extension of POMDP for multi-agent settings, with the basic extension path being MDP →
POMDP → POMG [Sun et al., 2023]. Please refer to Appendix C to see a more detailed explanation of POMG.

$$\pi_{i}^{*}=\arg\operatorname*{max}_{\hat{\pi}}\mathbb{E}_{\hat{\pi}}\left[\sum_{k=0}^{\infty}\gamma^{k}\sum_{i\in N}r_{t+k}^{i}\left|{\hat{s}}_{t}={\hat{s}}_{0}\right|,\right.$$
$$\left(2\right)$$
, (2)
105 where γ is the discount factor, and the expectation is taken over the trajectories induced by the joint 106 policy πˆ starting from the initial global state sˆ0.

## 107 **3 Definition And Taxonomy Of Heterogeneity In Marl**

127 Specifically, these five types of heterogeneity and their related definitions are as follows: 128 - *Observation heterogeneity* describes the differences of agents in observing global information. The 129 relevant elements include the agent's observation space and observation function. 130 **Definition 1.** Agents i and j are observation heterogeneous if the following conditions do not hold at the same time: ➀ Oi = Oj; ➁ ∀sˆ ∈ {S
i}i∈N , Ω
i(·|sˆ) = Ωj 131 (·|sˆ). 108 **Heterogeneity in MAS.** Our goal is to define agent heterogeneity from the perspective of MARL. 109 Before achieving this, we need to discuss heterogeneity in MAS across various disciplines. Early 110 studies [Dudek et al., 1996, Parker, 2000] define heterogeneity as differences in physical structure 111 or functionality of agents, which aligns with common understanding. Later work [Panait and 112 Luke, 2005] describes heterogeneity as differences in agent behavior, further expanding its meaning.

113 Recently, [Bennett, 2024] points out that heterogeneity may be a complex phenomenon, related 114 not only to the inherent properties of agents, but also to their interactions with environment. Thus, 115 heterogeneity in MARL should not be limited to inherent functional differences of agents, but should 116 also fully consider various coupling effects of agents within the environment.

117 **Heterogeneity in MARL.** In the context of MARL, the fundamental modeling of MARL and its 118 primal problem provides considerable convenience for defining heterogeneity. This modeling clearly specifies all MARL elements, delineating the boundaries of the problem discussion 2 119 and ensuring 120 the completeness of the discussion.

121 We focus on the heterogeneity *among agents* within a same POMG. As discussed in Section 2, the 122 function in a POMG can connect agent-level elements. Therefore, we categorize agent heterogeneity 123 into five types centered around the functions. This approach not only avoids overly redundant 124 classification but also ensures comprehensive coverage of each agent-level element. Regarding 125 definition, the condition for heterogeneity is obtained by *taking the negation of the necessary and* 126 *sufficient conditions for homogeneity.*
the joint observation space and {Ai}i∈N = ×i∈N Aiis the joint action space of all agents. {Ω
i 88 }i∈N
is the set of observation functions. {T i}i∈N = (T
1, *· · ·* , T
|N|, T
E 89 ) is the collection of all agents' 90 transitions and the environmental transition. Finally, {ri}i∈N is the set of reward functions of all 91 agents and γ is the discount factor.

92 Here, we give the independent and dependent variables for each function and their notation. At each time step t, an agent i receives an observation o it ∼ Ω
i(·|sˆt), where sˆt ∈ {S
i 93 }i∈N is the global state at time t. Then, agent i makes a decision based on its observation, resulting in an action a it ∼ πi(·|o it 94 ).

The environment then collects actions from all agents to form the global action aˆt = (a 1 t
, . . . , a |N| t 95 ). 96 We assume that the local state transition of agent i is influenced by the global state and global action, so its local state transitions to a new state s it+1 ∼ T i 97 (·|sˆt, aˆt). Similarly, the states of other agents and the environment also transition, yielding the next global state sˆt+1 = (s 1 t+1*, . . . , s* |N| t+1, sE
t+1 98 ) ∼
(T
1(·|sˆt, aˆt)*, . . . ,* T
|N|(·|sˆt, aˆt), T
E(·|sˆt, aˆt)) = {T i 99 }i∈N (·|sˆt, aˆt). At the same time, all agents receive rewards, with the reward for a specific agent i given by r it ∼ r i 100 (·|sˆt, aˆt).

101 The objective of MARL is to solve POMG by finding an optimal joint policy that maximizes the cumulative reward for all agents. We denote the individual optimal policy for agent i as π
∗
i 102 and the optimal joint policy as πˆ
∗, which can be expressed as πˆ
∗ = (π
∗
1
, . . . , π∗
|N| 103 ). The optimal joint policy 104 for a POMG can be obtained through the following equation: 132 - *Response transition heterogeneity* describes the differences of agents in how their state transitions 133 are affected by global environment components (*environment-to-self*). The relevant elements include 134 the agent's state space and local state transition function.

135 **Definition 2.** Agents i and j are response transition heterogeneous if the following conditions do not hold at the same time: ➀ S
i = S
j; ➁ ∀sˆ ∈ {S
i}i∈N , aˆ ∈ {Ai}i∈N , T
i(·|s, ˆ aˆ) = T
j 136 (·|s, ˆ aˆ). 137 - *Effect transition heterogeneity* describes the differences of agents in how their states and actions 138 impact global state transitions (*self-to-environment*). The relevant elements include the agent's action 139 space, state space, and global state transition function. 140 **Definition 3.** Agents i and j are effect transition heterogeneous if the following conditions do not hold at the same time: ➀ S
i = S
j; ➁ Ai = Aj; ➂ ∀s
′ ∈ S
−i, a′ ∈ A−i, s ∈ S
i, a ∈ Ai 141 ,
T
−i(·|s
′, s, a′, a) = T
−j(·|s
′*, s, a*′
142 , a).

In the above definition, S
−i = ×k∈N,k̸=iS
k × S
E 143 represents the joint state space of all agents except agent i, reflecting the influence of the agent on other states. Similarly, A−i 144 denotes the joint action space excluding agent i, and T
−i 145 is the collection of state transitions excluding agent i. 146 - *Objective heterogeneity* describes the differences of agents in the objective they aim to achieve. The 147 relevant element is the agent's reward function. 148 **Definition 4.** Agents i and j are objective heterogeneous if the following condition do not hold:
➀ ∀sˆ ∈ {S
i}i∈N , aˆ ∈ {Ai}i∈N , r i(·|s, ˆ aˆ) = r j 149 (·|s, ˆ aˆ).

## 159 **4 Quantifying Heterogeneity In Marl** 160 **4.1 Heterogeneity Distance Based On Representation Learning**

161 **Heterogeneity Distance.** In this section, we present the method to quantify the above five types of 162 heterogeneity. According to the definition, each type of heterogeneity corresponds to a core function 163 which connects relevant elements in the heterogeneity type. Therefore, we quantify the differences in these core functions to characterize the degree of heterogeneity.3 164 To make the quantification results 165 simpler and more practical, we propose the concept of heterogeneity distance.

166 Let the core function corresponding to a certain heterogeneity type F be denoted as y ∼ F(·|x). The 167 formula for calculating the F-heterogeneous distance between two agents i and j is given by:
168 where X is the space of independent variables, p(x) is the probability density function, and D[*· ∥ ·*] 169 is a measure that quantifies the difference between distributions. The core idea of heterogeneity 170 distance is to examine the cumulative differences between two agents' functions throughout the space 171 of independent variables, which captures any potential local differences. When the independent 172 variables x consist of multiple factors, the above integral becomes a multivariate integral. Based 173 on Equation 3, we provide the specific expressions for quantifying all heterogeneous distances in 174 Appendix F and discuss the properties of heterogeneous distance below.

$$d_{i j}^{F}=\int_{x\in X}D[F_{i}(\cdot|x)\parallel F_{j}(\cdot|x)]\cdot p(x)\,d x,$$
D[Fi(·|x) ∥ Fj (·|x)] · p(x) dx, (3)
150 - *Policy heterogeneity* describes the differences of agents in their autonomous decision-making based 151 on observations. The relevant elements include the observation space, action space, and policy. 152 **Definition 5.** Agents i and j are policy heterogeneous if the following conditions do not hold at the same time: ➀ Oi = Oj; ➁ Ai = Aj; ➂ ∀o ∈ Oi 153 , πi(·|o) = πj (·|o).

154 In the five types of heterogeneity mentioned above, we assume that all functions follow the Markov 155 property, making them independent of the agent's trajectory. Therefore, the first four types of 156 heterogeneity can be considered environment-related, which reflect the heterogeneity in the MARL 157 primal problem. The last type describes the policy heterogeneity of agents before, during, and after 158 training, which reflects the heterogeneity of optimization objectives (policies) in the primal problem.

(I) Sample *Collection*
(II) Learning Representational *Distribution* Model-Based Case Encoder 8%
9 Sample
:$(;|<)
< **Model** " ;;
8%(9|;, <)
=&(;'|9, <)
log A ; < ≥ C(! )|+,- log =& ; 9, < − E./[8% 9|;, < ||A(9|<)]
Decoder
=&

Sample Pool Collect Environment Interaction
< <
Sample Pool
<
(III) Computing Heterogeneity *Distance*
;$ or H
Agent "
Model-Free Case Sample Pool Encoder 8%
<
Sample A$(9|<) A0(9|<)
Encoder 8%
9 Sample
:$(;|<)
Model " H **Decoder**
=&
;
H, <
8%(9|H, <) =&(;|9, H, <)
Agent . ;0 or I
<, ;
<
Sample Pool log A ; H, < ≥ C(! )|$,- log =& ; 9, H, < − E./[8% 9|H, < ||A(9|H, <)]
J$0 1 = L
-∈3 E[A$(9|<), A0(9|<)]A(<)J<
Figure 2: The method of measuring heterogeneity distance based on representation learning.

$$E L B O_{\mathrm{model-based}}=\mathbb{E}_{f_{\phi}(z|y,x)}\left[\log g_{\omega}(y|z,x)\right]-D_{K L}\left[f_{\phi}(z|y,x)\parallel p(z|x)\right],$$

202 where fϕ and gω represent the encoder and decoder, respectively, and p(z|x) is the prior conditional 203 latent distribution. We designed the relevant loss based on ELBO, including a reconstruction term 204 and a prior-matching term. The derivation for this part can be found in Appendix H.

184 For the first issue, our approach is to **standardize the original distributions**. By learning a rep185 resentation mapping, for all independent variables x, a measurable distribution pi(z|x) is used to 186 capture the characteristics of the original distribution Fi(y|x), replacing the original one for measure 187 computation. For the second issue, our approach is **sampling based on the interaction between** 188 **agents and the environment**. Instead of simply traversing the space or using random policy explo189 ration for sampling, we construct a sample pool using trajectories from the training phase of MARL. 190 This significantly reduces computational load and filters out excessive marginal spaces that interfere 191 with MARL, benefiting the use of heterogeneity distance in subsequent MARL tasks (Section 5). 192 Combining these ideas, we propose a practical method as shown in Figure 2. 193 **In the first step**, the agents interact with the environment during MARL training to build a sample 194 pool. Notably, the sample pool data is shuffled to ensure that the learned function follows the Markov 195 property (independent of historical information), similar to the original function. 196 **In the second step**, the representational distributions are learned. We discuss this in both model197 based and model-free settings, corresponding to cases where the function is known and unknown, 198 respectively. We adopt the conditional variational autoencoder (CVAE) framework [Sohn et al., 2015] 199 for representation learning. In the model-based case, CVAE performs a reconstruction task [Lopez200 Martin et al., 2017]. The optimization goal is to maximize the likelihood of the reconstructed variable 201 log p(y|x). Through derivation, we obtain the evidence lower bound (ELBO) as:
179 **Practical Method.** Although the heterogeneity distance has a simple form, some issues may arise dur180 ing practical computation. First, computing the distribution distance via sampling is computationally 181 complex, while computing the distance using analytical solutions requires knowing the distribution type. In real-world scenarios, the distributions may be unknown or of different types 4 182 . Second, the 183 independent variable space may be very large, making traversal-based computation infeasible.

Proposition 1. (Properties of Heterogeneity Distance) ➀ *Symmetry*: d F
ij = d F
ji 175 ; ➁ *Non-negativity*:
d F
ij ≥ 0; ➂ *Identity of indiscernibles*: d F 176 ij = 0 if and only if agents i and j are F-homogeneous; ➃
Triangle inequality: d F
ij ≤ d F
ik + d F
kj 177 (*i, j, k* ∈ N). This proposition holds as long as the measure D 178 satisfies ➀➁➂➃. The proof is provided in Appendix E.

$\eqref{eq:walpha}$. 
Scenario-v1 Scenario-v2 **Scenario-v3**
Scenario-v4 Scenario-v5 Scenario-v6 L L L
L L
Obs-Het Response-Het L L
Obs-Het Response-Het L
Obs-Het Response-Het L
Obs-Het Response-Het Obs-Het Response-Het Obs-Het *Response-Het* Effect-Het *Objective-Het* Effect-Het Objective-Het Effect-Het *Objective-Het* Effect-Het Objective-Het Effect-Het *Objective-Het* Effect-Het Objective-Het Meta-Het Meta-Het Meta-Het Meta-Het

$$({\boldsymbol{5}})$$

Meta-Het Meta-Het
205 In the model-free case, CVAE essentially performs a prediction task [Zhang et al., 2021], capturing 206 the model characteristics of each agent. The optimization goal is to maximize the likelihood of 207 the predicted variable y given conditions i and x, where i is the agent ID. Similarly, we derive the 208 corresponding ELBO:

$$ELBO_{\mathrm{model-free}}=\mathbb{E}_{f_{\phi}(z|i,x)}\left[\log g_{\omega}(y|z,i,x)\right]-D_{K L}\left[f_{\phi}(z|i,x)\parallel p(z|i,x)\right].$$

## 221 **4.2 Case Study**

209 **In the third step**, the heterogeneity distances for multi-agents are computed. For each x, we obtain 210 the distribution representation using the encoder in either the model-based or model-free manner. 211 The distance under a specific x is computed using the *Wasserstein distance* [Vaserstein, 1969] of the 212 prior distribution (*standard Gaussian*). The heterogeneity distance is then calculated via multi-rollout Monte Carlo sampling. In practice, we parallelize this operation 5 213 , enabling simultaneous computation 214 of distances between all agents on GPUs, significantly improving computational efficiency. 215 **Meta-Transition.** The aforementioned method can quantify the heterogeneity of agents for specific 216 types. In practical applications, researchers may also want to quantify the **comprehensive** heterogene217 ity of agents to enable operations such as grouping. To this end, we give the *Meta-Transition* model 218 (see Appendix G for details). By measuring the differences between meta-transitions, the comprehen219 sive heterogeneity related to environment can be quantified. We refer to this as the meta-transition 220 heterogeneity distance. 222 We design a multi-agent spread scenario for case study. In the basic scenario, there are two groups, 223 each with two agents, and their goal is to move close to randomly generated landmarks. Based 224 on the basic scenario, we create 6 extended versions to show the quantitative results of different 225 types of heterogeneity and meta-transition heterogeneity. As shown in Figure 3, the first 4 versions 226 correspond to the 4 environment-related types of heterogeneity, while the last 2 versions represent 5Our code is provided in the supplementary material.

227 cases where multiple types of heterogeneity exist. We use the model-based manner to compute the 228 four heterogeneity distance matrices mentioned above, and the model-free manner to compute the 229 meta-heterogeneity distance matrix for the agents. 230 The results show that for each type of heterogeneity, our method can accurately capture and identify 231 the differences. For meta-transition heterogeneity, the distance between agents in the same group is 232 much smaller than that in different groups. Moreover, as the number of heterogeneity types increases, 233 the distance between different groups also increases. These results demonstrate the effectiveness of 234 our method for various environment-related heterogeneities. 235 We further quantify the policy heterogeneity dis236 tance (*Policy-Het*) and meta-transition hetero237 geneity distance (*Meta-Het*) of agents during the 238 training process. We select two algorithms at the 239 extreme ends of parameter sharing: fully parame240 ter sharing (FPS) and no parameter sharing (NPS) 241 for training in the above scenarios. Figure 4 242 shows the measurement results at 500 and 1500 243 updates. From the *Policy-Het* results, the policy 244 distance can effectively reveal the evolutionary 245 relationship of agent policy differences in MARL.

246 From the *Meta-Het* results, the comprehensive 247 agent heterogeneity measurement remains con248 sistent across different learning algorithms, and 249 can identify environmental heterogeneous char250 acteristics in scenarios more rapidly compared to 251 policy evolution.

## 252 **5 Multi-Agent Dynamic Parameter** 253 **Sharing Based On Heterogeneity** 254 **Quantification: An Application**

Meta-Het Policy-Het Meta-Het *Policy-Het* Meta-Het *Policy-Het* Updates Reward Meta-Het *Policy-Het*
Figure 4: Meta-transition heterogeneity and policy heterogeneity distance matrices during training in our case study.

255 Based on the case study in Section 4.2, the pro256 posed method can not only accurately quantify all 257 types of heterogeneity, but also the comprehen258 sive heterogeneity among agents. Additionally, the method is independent of the parameter-sharing 259 type used in MARL and can be deployed online, thereby further enhancing its practicality. In 260 this section, we provide a practical application of our methodology to demonstrate its potential in 261 empowering MARL.

262 We select parameter sharing in MARL as our application context. As a common technique in 263 MARL, parameter sharing can reduce computational consumption while improving sample utilization 264 efficiency [KIM and Sung, 2023], but its excessive use may inhibit agents' policy heterogeneity 265 expression [Hu et al., 2024]. Many works have attempted to find a balance between parameter sharing 266 and policy heterogeneity [Li et al., 2024b]. However, existing approaches suffer from two main 267 problems: *poor interpretability*, unable to explain why policy heterogeneity is necessary and to what 268 extent; and *poor adaptability*, manifested by numerous task-specific hyperparameters and inability 269 to dynamically adapt policy training. (For a more detailed discussion of these algorithms, see the 270 experimental section 6.1) 271 To address these issues, we propose a Heterogeneity-based multi-agent Dynamic Parameter Sharing 272 algorithm (HetDPS) with two core ideas(More details can be found in Appendix I): 273 ♠ **Grouping agents for parameter sharing through heterogeneity distances**. We utilize distance274 based clustering methods to group agents, thus avoiding the introduction of task-specific hyperpa275 rameters like group number [Christianos et al., 2021, Li et al., 2024a] or fusion thresholds [Hu et al., 276 2024]. The heterogeneity distance matrices also enhance the algorithm's interpretability.

277 ♣ **Periodically quantifying heterogeneity and modifying agents' parameter sharing paradigm**. 278 This can help the sample pool become more aligned with policy training. This approach can also 279 help policies escape local optima [Lyle et al., 2024], the effectiveness of such a mechanism has been 280 verified in the MARL domain [Li et al., 2024b], and even in broader RL areas such as large model 281 fine-tuning [Noukhovitch et al., 2023, Ma et al., 2024].

## 282 **6 Experiments**

283 In the experimental section, we conduct comprehensive comparisons between HetDPS and other 284 parameter sharing algorithms. Beyond performance comparisons, we also analyze the heterogeneity 285 characteristics of each MARL task with our proposed methodology, to demonstrate the algorithm's 286 interpretability. Additionally, we conduct hyperparameter experiments and efficiency and resource 287 consumption experiments to show the adaptability and practicality of HetDPS.

## 288 **6.1 Experimental Setups**

289 Environments. Partical290 based Multi-agent Spread291 ing [Hu et al., 2024] is a 292 typical environment in the 293 policy diversity domain. In 294 this environment, multiple 295 agents are randomly gen296 erated in the center of the 297 map, while multiple land298 marks are randomly gen299 erated near the periphery.

300 Both agents and landmarks 301 have various colors, and 302 agents need to move to land303 marks with matching colors. 304 Additionally, agents need to 305 form tight formations when 306 they reach the vicinity of landmarks. We employ 4 typical tasks, corresponding to different 307 numbers and color distributions, as detailed in Table 1. **The StarCraft Multi-Agent Challenge** 308 **(SMAC)** [Samvelyan et al., 2019] is a popular MARL benchmark, where multiple ally units controlled 309 by MARL algorithms aim to defeat enemy units controlled by the game's built-in AI.

15a_3c 30a_3c 15a_5c 30a_5c Meta-Het Matrix Meta-Het Matrix Meta-Het Matrix Meta-Het *Matrix* Figure 5: Results on Partical-based Multi-agent Spreading.

310 **Baselines and training.** We compare HetDPS with other 311 parameter sharing baselines, as listed in Table 2. We 312 analyze these baselines along three dimensions: parame313 ter sharing paradigm, adaptability, and relationship with 314 heterogeneity utilization. As seen from the table, cur315 rent methods can not effectively utilize heterogeneity. Al316 though some methods implicitly use certain heterogeneity 317 quantification results, the elements they involve are not 318 comprehensive. MADPS, as the only method that explic319 itly uses policy distance for dynamic grouping, relies on 320 the assumption that policy learning can effectively capture 321 heterogeneity, which lacks practicality. We use official implementations of the baselines where 322 available. For further discussion on related work and experiments in this paper, see the supplementary 323 materials.

Table 1: Task information for particlebased multi-agent spreading.

Task Agent Type Distribution 15a_3c 5 − 5 − 5 30a_3c 10 − 10 − 10 15a_5c 3 − 3 − 3 − 3 − 3 30a_5c 3 − 3 − 3 − 12 − 9

## 324 **6.2 Results**

325 **Performance and interpretability.** We tested the performance of all comparison algorithms in the 326 two environments mentioned above. The reward curves and corresponding heterogeneity distance 327 matrices are shown in Figure 5 and Figure 6. From the reward curve results, we can see that HetDPS 328 achieves either optimal or comparable results in all tasks above.

329 We quantified the meta-transition heterogeneity distances for all tasks. The results show that our 330 heterogeneity quantification results in the Multi-agent Spreading scenario are highly consistent with

| Method                          | Paradigm        | Adaptive   | Relation to Heterogeneity Utilization                                             |
|---------------------------------|-----------------|------------|-----------------------------------------------------------------------------------|
| NPS                             | No Sharing      | No         | None                                                                              |
| FPS                             | Full Sharing    | No         | None                                                                              |
| FPS+id                          | Full Sharing    | No         | None                                                                              |
| Kaleidoscope [Li et al., 2024b] | Partial Sharing | Yes        | No utilization, increases agent policy heterogeneity as the bias                  |
| SePS [Christianos et al., 2021] | Group Sharing   | No         | Implicitly utilizes objective heterogeneity and response transition heterogeneity |
| AdaPS [Li et al., 2024a]        | Group Sharing   | Yes        | Implicitly utilizes objective heterogeneity and response transition heterogeneity |
| MADPS [Hu et al., 2024]         | Group Sharing   | Yes        | Explicitly utilizes policy heterogeneity only                                     |
| HetDPS (ours)                   | Group Sharing   | Yes        | Explicitly utilizes heterogeneity, leveraging heterogeneous distance              |

Table 3: Training efficiency metrics across different methods. Results are normalized with respect to the FPS method, and averaged across all tasks.

| NPS            | FPS    | FPS+id   | Kaleidoscope   | SePS   | AdaPS   | MADPS   | HetDPS (ours)   |        |
|----------------|--------|----------|----------------|--------|---------|---------|-----------------|--------|
| Training Speed | 0.952x | 1.000x   | 0.992x         | 0.974x | 0.986x  | 0.614x  | 0.539x          | 0.712x |

331 the type distribution in Table 1. This demonstrates the effectiveness of our method in identifying 332 agent heterogeneity. Additionally, we made some interesting discoveries in the SMAC environment. 333 We found that in simpler tasks like *3s5z* and MMM, the agent heterogeneity quantification results 334 often do not closely match the original agent types. In MMM, agents even tend toward homogeneous 335 policies to improve training efficiency. However, in more difficult tasks such as *3s5z_vs_3s6z* and 336 *MMM2*, agents' quantification results closely match their original types to achieve better coordination. 337 This confirms our view that agent heterogeneity is related not only to the agents' original functional 338 attributes but also to how agents interact with the environment. 339 **Cost Analysis.** We conducted an experiment to investigate training efficiency. The experimental 340 results are shown in Table 3. The results indicate that although our method introduces periodic 341 heterogeneity quantification, it does not significantly reduce algorithm efficiency.

## 342 **7 Conclusion**

343 Heterogeneity manifests in 344 various aspects of MARL. 345 It is not only related to 346 the inherent properties of 347 agents themselves but also 348 to the coupling factors aris349 ing from agent-environment 350 interactions. Consequently, 351 agents that appear homoge352 neous may develop hetero353 geneity under environmen354 tal influences. In this paper, 355 we categorize heterogene356 ity in MARL into five types 357 and provide respective defi358 nitions. Meanwhile, we pro359 pose methods for quantify360 ing these heterogeneity types and conduct case studies. Under our theoretical framework, policy 361 diversity is merely a manifestation of policy heterogeneity, fundamentally originating from the 362 division of labor necessitated by agents' environmental heterogeneity (*cause*), serving as an inductive 363 bias (*result*) for solving optimal joint policies. Thus, we introduce the quantification of heterogeneity 364 as prior knowledge into multi-agent parameter-sharing learning. The result is HetDPS, an algorithm 365 with strong interpretability and adaptability. HetDPS is not the endpoint of our research, but rather a 366 starting point for heterogeneity applications. We believe that by systematically studying the definition, 367 quantification, and application of heterogeneity, future MARL research will more profoundly under368 stand the complex collaboration mechanisms between agents, and pave the way for more intelligent 369 and adaptive collective decision-making systems.

3s5z 3s5z_vs_3s6z MMM MMM2 Meta-Het Matrix Meta-Het Matrix Meta-Het Matrix Meta-Het *Matrix* Figure 6: Results on StarCraft Multi-Agent Challenge.

## 370 **References**

371 Richard Bellman. A markovian decision process. *Journal of mathematics and mechanics*, pages 372 679–684, 1957. 373 Chris Bennett. *Heterogeneity in multi-agent systems*. PhD thesis, University of Bristol, 2024. 374 Daniel S Bernstein, Robert Givan, Neil Immerman, and Shlomo Zilberstein. The complexity of 375 decentralized control of markov decision processes. *Mathematics of operations research*, 27(4):
376 819–840, 2002.

377 Matteo Bettini, Ajay Shankar, and Amanda Prorok. Heterogeneous multi-robot reinforcement 378 learning. In *AAMAS*, 2023a. 379 Matteo Bettini, Ajay Shankar, and Amanda Prorok. System neural diversity: Measuring behavioral 380 heterogeneity in multi-agent learning. *arXiv preprint arXiv:2305.02128*, 2023b. 381 Matteo Bettini, Ryan Kortvelesy, and Amanda Prorok. Controlling behavioral diversity in multi-agent 382 reinforcement learning. In *International Conference on Machine Learning*, pages 3611–3636. 383 PMLR, 2024. 384 Alicia L Burns, Alexander DM Wilson, and Ashley JW Ward. Behavioural interdependence in a 385 shrimp-goby mutualism. *Journal of Zoology*, 308(4):274–279, 2019.

386 Anthony Rocco Cassandra. *Exact and approximate algorithms for partially observable Markov* 387 *decision processes*. Brown University, 1998. 388 Filippos Christianos, Georgios Papoudakis, Muhammad A Rahman, and Stefano V Albrecht. Scaling 389 multi-agent reinforcement learning with selective parameter sharing. In *International Conference* 390 *on Machine Learning*, pages 1989–1998. PMLR, 2021.

391 Emiliano Dall'Anese, Hao Zhu, and Georgios B Giannakis. Distributed optimal power flow for smart 392 microgrids. *IEEE Transactions on Smart Grid*, 4(3):1464–1475, 2013. 393 Gregory Dudek, Michael RM Jenkin, Evangelos Milios, and David Wilkes. A taxonomy for multi394 agent robotics. *Autonomous Robots*, 3:375–397, 1996. 395 Sven Gronauer and Klaus Diepold. Multi-agent deep reinforcement learning: a survey. *Artificial* 396 *Intelligence Review*, 55(2):895–943, 2022.

397 Xudong Guo, Daming Shi, Junjie Yu, and Wenhui Fan. Heterogeneous multi-agent reinforcement 398 learning for zero-shot scalable collaboration. *arXiv preprint arXiv:2404.03869*, 2024. 399 Tianyi Hu, Zhiqiang Pu, Xiaolin Ai, Tenghai Qiu, and Jianqiang Yi. Measuring policy distance 400 for multi-agent reinforcement learning. In *Proceedings of the 23rd International Conference on* 401 *Autonomous Agents and Multiagent Systems*, pages 834–842, 2024.

402 Jiechuan Jiang and Zongqing Lu. The emergence of individuality. In *International Conference on* 403 *Machine Learning*, pages 4992–5001. PMLR, 2021. 404 Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore. Reinforcement learning: A survey. 405 *Journal of artificial intelligence research*, 4:237–285, 1996. 406 Leslie Pack Kaelbling, Michael L Littman, and Anthony R Cassandra. Planning and acting in partially 407 observable stochastic domains. *Artificial intelligence*, 101(1-2):99–134, 1998.

408 Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian Ibarz, Alexander Herzog, Eric Jang, Deirdre 409 Quillen, Ethan Holly, Mrinal Kalakrishnan, Vincent Vanhoucke, et al. Scalable deep reinforcement 410 learning for vision-based robotic manipulation. In *Conference on robot learning*, pages 651–673. 411 PMLR, 2018. 412 WOOJUN KIM and Youngchul Sung. Parameter sharing with network pruning for scalable multi413 agent deep reinforcement learning. In *The 22nd International Conference on Autonomous Agents* 414 *and Multiagent Systems (AAMAS)*. AAMAS, 2023. 415 Mykel J Kochenderfer, Tim A Wheeler, and Kyle H Wray. *Algorithms for decision making*. MIT 416 press, 2022. 417 Chenghao Li, Tonghan Wang, Chengjie Wu, Qianchuan Zhao, Jun Yang, and Chongjie Zhang. 418 Celebrating diversity in shared multi-agent reinforcement learning. *Advances in Neural Information* 419 *Processing Systems*, 34:3991–4002, 2021. 420 Dapeng Li, Na Lou, Bin Zhang, Zhiwei Xu, and Guoliang Fan. Adaptive parameter sharing for 421 multi-agent reinforcement learning. In *ICASSP 2024-2024 IEEE International Conference on* 422 *Acoustics, Speech and Signal Processing (ICASSP)*, pages 6035–6039. IEEE, 2024a. 423 Xinran Li, Ling Pan, and Jun Zhang. Kaleidoscope: Learnable masks for heterogeneous multi-agent 424 reinforcement learning. In *The Thirty-eighth Annual Conference on Neural Information Processing* 425 *Systems*, 2024b. 426 Michael L Littman. Markov games as a framework for multi-agent reinforcement learning. In 427 *Machine learning proceedings 1994*, pages 157–163. Elsevier, 1994.

428 Manuel Lopez-Martin, Belen Carro, Antonio Sanchez-Esguevillas, and Jaime Lloret. Conditional 429 variational autoencoder for prediction and feature recovery applied to intrusion detection in iot. 430 *Sensors*, 17(9):1967, 2017. 431 Jonathan Lwowski, Patrick Benavidez, John J Prevost, and Mo Jamshidi. Task allocation using 432 parallelized clustering and auctioning algorithms for heterogeneous robotic swarms operating on a 433 cloud network. *Autonomy and artificial intelligence: A threat or savior?*, pages 47–69, 2017. 434 Clare Lyle, Zeyu Zheng, Khimya Khetarpal, James Martens, Hado P van Hasselt, Razvan Pascanu, 435 and Will Dabney. Normalization and effective learning rates in reinforcement learning. *Advances* 436 *in Neural Information Processing Systems*, 37:106440–106473, 2024. 437 Hao Ma, Tianyi Hu, Zhiqiang Pu, Liu Boyin, Xiaolin Ai, Yanyan Liang, and Min Chen. Coevolving 438 with the other you: Fine-tuning llm with sequential cooperative multi-agent reinforcement learning. 439 *Advances in Neural Information Processing Systems*, 37:15497–15525, 2024. 440 Dung Nguyen, Phuoc Nguyen, Svetha Venkatesh, and Truyen Tran. Learning to transfer role 441 assignment across team sizes. *arXiv preprint arXiv:2204.12937*, 2022.

442 Zepeng Ning and Lihua Xie. A survey on multi-agent reinforcement learning and its application.

443 *Journal of Automation and Intelligence*, 3(2):73–91, 2024. 444 Michael Noukhovitch, Samuel Lavoie, Florian Strub, and Aaron C Courville. Language model 445 alignment with elastic reset. *Advances in Neural Information Processing Systems*, 36:3439–3461, 446 2023. 447 Frans A Oliehoek, Christopher Amato, et al. *A concise introduction to decentralized POMDPs*, 448 volume 1. Springer, 2016.

449 Liviu Panait and Sean Luke. Cooperative multi-agent learning: The state of the art. *Autonomous* 450 *agents and multi-agent systems*, 11:387–434, 2005. 451 Lynne E Parker. Lifelong adaptation in heterogeneous multi-robot teams: Response to continual 452 variation in individual robot performance. *Autonomous Robots*, 8:239–267, 2000. 453 Mikayel Samvelyan, Tabish Rashid, Christian Schroeder de Witt, Gregory Farquhar, Nantas Nardelli, 454 Tim G. J. Rudner, Chia-Man Hung, Philiph H. S. Torr, Jakob Foerster, and Shimon Whiteson. The 455 StarCraft Multi-Agent Challenge. *CoRR*, abs/1902.04043, 2019.

456 Esmaeil Seraj, Zheyuan Wang, Rohan Paleja, Matthew Sklar, Anirudh Patel, and Matthew Gombolay. 457 Heterogeneous graph attention networks for learning diverse communication. *arXiv preprint* 458 *arXiv:2108.09568*, 2021.

459 Kihyuk Sohn, Honglak Lee, and Xinchen Yan. Learning structured output representation using deep 460 conditional generative models. *Advances in neural information processing systems*, 28, 2015. 461 Matthijs TJ Spaan. Partially observable markov decision processes. In *Reinforcement learning:* 462 *State-of-the-art*, pages 387–414. Springer, 2012. 463 Lijun Sun, Yu-Cheng Chang, Chao Lyu, Ye Shi, Yuhui Shi, and Chin-Teng Lin. Toward multi-target 464 self-organizing pursuit in a partially observable markov game. *Information Sciences*, 648:119475, 465 2023. 466 Leonid Nisonovich Vaserstein. Markov processes over denumerable products of spaces, describing 467 large systems of automata. *Problemy Peredachi Informatsii*, 5(3):64–72, 1969. 468 T Wang, T Gupta, B Peng, A Mahajan, S Whiteson, and C Zhang. Rode: learning roles to decompose 469 multi- agent tasks. In *Proceedings of the International Conference on Learning Representations*.

470 OpenReview, 2021. 471 H Peyton Young. The evolution of conventions. *Econometrica: Journal of the Econometric Society*, 472 pages 57–84, 1993. 473 Xiaoyang Yu, Youfang Lin, Xiangsen Wang, Sheng Han, and Kai Lv. Ghq: grouped hybrid q-learning 474 for cooperative heterogeneous multi-agent reinforcement learning. *Complex & Intelligent Systems*, 475 10(4):5261–5280, 2024. 476 Chen Zhang, Riccardo Barbano, and Bangti Jin. Conditional variational autoencoder for learned 477 image reconstruction. *Computation*, 9(11):114, 2021. 478 Ming Zhou, Jun Luo, Julian Villella, Yaodong Yang, David Rusu, Jiayu Miao, Weinan Zhang, 479 Montgomery Alban, Iman Fadakar, Zheng Chen, et al. Smarts: An open-source scalable multi480 agent rl training school for autonomous driving. In *Conference on robot learning*, pages 264–285. 481 PMLR, 2021.

## 482 **A Limitations**

483 Although our proposed heterogeneity distance can effectively quantify agent heterogeneity and 484 identify various potential heterogeneities, there remain some limitations in its practical implementa485 tion. One limitation is in scaling with the number of agents. Typically, the heterogeneity distance 486 quantification algorithm outputs a heterogeneity distance matrix for the entire multi-agent system, with a computational complexity of O(N2 487 ). When the number of agents increases significantly, 488 matrix computation becomes costly. However, if only studying heterogeneity between specific agents 489 in the MAS is required, the method remains effective. One only needs to remove data from other 490 agents during CVAE training and sampling computation. 491 Additionally, the practical algorithms for heterogeneity quantification are built on the assumption 492 that agent-related variables are vectors. If certain agent variables, such as observation inputs, are 493 multimodal, operations like padding in the proposed algorithm become difficult to implement. But 494 this does not affect the correctness of the theory. As the relevant theory still holds in this situation, 495 additional tricks are needed for practical calculation implementation.

## 496 **B Broader Impacts**

497 Our work systematically analyzes heterogeneity in MARL, which has strong correlations with a 498 series of works in MARL. Under our theoretical framework, research on agent policy diversity in 499 MARL can be categorized within the domain of policy heterogeneity. Our work can give a new 500 perspective for studying policy diversity. Our proposed quantification methods can not only help 501 these works with policy evolution analysis but also explain the relationship between policy diversity 502 and agent heterogeneity. Furthermore, our proposed HetDPS, as an application case, can also be 503 classified among parameter sharing-based works. 504 Additionally, some traditional heterogeneous MARL works can be categorized within environment505 related heterogeneity domains. Our quantification and definition methods are orthogonal to these 506 works, which can fully utilize our proposed methodology for further advancement. For instance, 507 observation heterogeneity quantification can be used to enhance agents' ability to aggregate hetero508 geneous observation information; transition heterogeneity quantification can help design intrinsic 509 rewards to assist heterogeneous multi-agents in learning cooperative policies. 510 In conclusion, our work not only expands the scope of heterogeneity in MARL but also closely 511 connects with many current hot topics, contributing to the further development of these works.

## 512 **C An Introduction To Pomg**

513 Partially Observable Markov Game (POMG) is essentially an extension of Partially Observable 514 Markov Decision Process (POMDP), which in turn extends Markov Decision Process (MDP).

515 MDP [Bellman, 1957, Kaelbling et al., 1996] is a mathematical framework that describes sequential 516 decision-making by a single agent in a fully observable environment. In an MDP, the agent can 517 fully observe the environment's state, select actions based on the current state, and aim to maximize 518 cumulative rewards. Compared to MDP, the key extension of POMDP [Kaelbling et al., 1998, 519 Cassandra, 1998] is the consideration of partial observability, making it suitable for modeling both 520 single-agent partially observable problems [Spaan, 2012] and multi-agent problems [Bernstein et al., 521 2002, Oliehoek et al., 2016]. In multi-agent POMDPs, agents typically operate in a fully cooperative 522 mode, where their rewards are usually team-shared. 523 The key extension of POMG over POMDP lies in modeling mixed game relationships among multiple 524 agents. Unlike POMDP, agents in POMG do not share a common reward function; instead, each 525 agent has its own (agent-level) reward function, making POMG more general [Sun et al., 2023, 526 Gronauer and Diepold, 2022]. This design enables POMG to handle competitive, cooperative, and 527 mixed interaction scenarios, better reflecting the complexity of real-world multi-agent systems. The 528 logical relationships among Markov decision processes and their variants are illustrated in Figure 7 529 and Figure 8. As shown in these figures, POMG is the most general framework for modeling original 530 problems in the MARL domain. For these reasons, we chose POMG as the foundation for discussing 531 heterogeneity in MARL.

## 532 **D Other Potential Types Of Heterogeneity In Marl**

533 Benefiting from the reinforcement learning modeling based on POMG, we have clearly defined the 534 boundaries of heterogeneity discussed in this paper. In fact, within the realm of unconventional 535 multi-agent systems, there might be other types of heterogeneity. 536 For instance, agents may have different length of decision timesteps, with some agents inclined 537 towards long-term high-level decisions, while others tend to make short-term low-level decisions. 538 Agents may also have different discount factors, some works try to assign varying discount factors to 539 different agents during algorithm training [Nguyen et al., 2022], to encourage agents to develop "my540 *opic*" or "*far-sighted*" policy behaviors, thereby promoting agent cooperation. However, differences 541 in discount factors are more reflective of algorithmic design variations rather than environmental 542 distinctions, and thus fall outside the scope of this paper. Moreover, there may be heterogeneity 543 among agents regarding communication, agents might have different communication channels due 544 to hardware variations. However, the establishment of communication protocols aims to enable 545 agents to receive more information when making decisions, potentially overcoming non-stationarity 546 and partial observability issues [Gronauer and Diepold, 2022]. These communication messages are 547 essentially mappings of global information processed in the environment, which are then input into 548 the action-related network modules. From this perspective, agent communication can be modeled as 549 a more generalized observation function that maps global information to local observations for agent 550 decision-making, and communication heterogeneity can be categorized under observation hetero551 geneity. From a learning perspective, agents might also have heterogeneous available knowledge, 552 such as differences in initial basic policies or variations in supplementary knowledge accessible 553 during execution phase. Moreover, heterogeneity might extend beyond abstract issues, including 554 computational resource differences among agents during learning.

555 Overall, even from the perspective of multi-agent reinforcement learning, heterogeneity in multi-agent 556 systems remains a domain with extensive discussion space, warranting further subsequent research.

## 557 **E Properties Of Heterogeneity Distance**

558 **Recap.** The heterogeneity distance between two agents in Section 4 can be computed as follows:

 ${d_{ij}^F=\int_{x\in X}D[F_i(\cdot|x),F_j(\cdot|x)]\cdot p(x)\,dx,}$  Dependent variables, ${p(x)}$ is the probability density function, and ${D}$. 
$$(6)$$
559 where X is the space of independent variables, p(x) is the probability density function, and D[·, ·] is 560 a measure that quantifies the difference between distributions.

Proposition 1. (Properties of Heterogeneity Distance) ➀ *Symmetry*: d F
ij = d F
ji 561 ; ➁ *Non-negativity*:
d F
ij ≥ 0; ➂ *Identity of indiscernibles*: d F 562 ij = 0 if and only if agents i and j are F-homogeneous; ➃
Triangle inequality: d F
ij ≤ d F
ik + d F
kj 563 (*i, j, k* ∈ N). This proposition holds as long as the measure D 564 satisfies Property ➀➁➂➃. 565 **Proof.** It can be proven that when D satisfies Property ➀➁➂➃, heterogeneity distance also satisfies 566 Property ➀➁➂➃.

567 *1) Proof of Symmetry:*

_Day of Symmetry._  $$d_{ij}^{F}=\int_{x\in X}D\left[F_{i}(\cdot|x),F_{j}(\cdot|x)\right]\cdot p(x)dx=\int_{x\in X}W\left[F_{j}(\cdot|x),F_{j}(\cdot|x)\right]\cdot p(x)dx=d_{ji}^{F}.$$
ji. (7)
568 *2) Proof of Non-negativity:*

_-negativity._  $$d_{ij}^{F}=\int_{x\in X}D\left[F_{i}(\cdot|x),F_{j}(\cdot|x)\right]\cdot p(x)dx\geq\int_{x\in X}0\cdot p(x)dx=0.$$
569 *3) Proof of Identicals of indiscernibility (necessary conditions):*

if agent i and agent j are F-homogeneous, then we have: X(i) = X(j), ∀x ∈ X = X(i)
570 , Fi(·|x) =
571 Fj (·|x),
$$d_{ij}^{F}=\int_{x\in X}D\left[F_{i}(\cdot|x),F_{j}(\cdot|x)\right]\cdot p(x)dx\tag{9}$$ $$=\int_{x\in X}D\left[F_{i}(\cdot|x),F_{i}(\cdot|x)\right]\cdot p(x)dx$$ $$=\int_{x\in X}0\cdot p(x)dx$$ $$=0.$$
$$\quad(7)$$
$$(8)$$
572 *4) Proof of Identicals of indiscernibility (sufficient conditions):*

$$\begin{array}{r}{d_{i j}^{F}=0\ {\xrightarrow{\mathrm{Pop}\cdot\mathbb{Z}_{\diamond}}}\ D\left[F_{i}(\cdot|x),F_{i}(\cdot|x)\right]=0,\forall x\in X^{(i)}o r X^{(j)}}\\ {{\xrightarrow{\mathrm{Prop}\cdot\mathbb{Z}o r D}}\ F_{i}(\cdot|x)=F_{i}(\cdot|x),\forall x\in X,X=X^{(i)}=X^{(j)},}\end{array}$$
$$(10)$$

573 then we have agent i and agent j are F-homogeneous. 574 *5) Proof of Triangle Inequality:*

$$d_{ij}^{E}=\int_{x\in X}D\left[F_{i}(\cdot|x),F_{j}(\cdot|x)\right]\cdot p(x)\,dx$$ $$\leq\int_{x\in X}\left(D\left[F_{i}(\cdot|x),F_{k}(\cdot|x)\right]+D\left[F_{k}(\cdot|x),F_{j}(\cdot|x)\right]\right)\cdot p(x)\,dx$$ $$=\int_{x\in X}D\left[F_{i}(\cdot|x),F_{k}(\cdot|x)\right]\cdot p(x)\,dx+\int_{x\in X}D\left[F_{k}(\cdot|x),F_{j}(\cdot|x)\right]\cdot p(x)\,dx$$ $$=d_{ik}^{E}+d_{kj}^{E}.$$
$$(11)$$

575 In this paper, we choose the *Wasserstein Distance* [Vaserstein, 1969] as the metric to quantify the 576 distance between distributions, which satisfies the property ➀➁➂➃ [Bettini et al., 2023b].

577 **Discussion.** In practical computation, we adopt a representation learning-based approach to find 578 an alternative latent variable distribution pi(z|x) to replace the original distribution Fi(y|x) for 579 quantification. It can be easily proved that when using latent variable distributions to compute 580 heterogeneous distances, these distances still satisfy properties ➀, ➁, and ➃ (following the same 581 proof method as above).

In the model-based case, pi(z|x) = fϕ(yi 582 , x), where fϕ represents the encoder of the CVAE. When 583 two agents have the same independent and dependent variables (identical agent functions), their latent 584 variable distributions are also identical. In this case, it is straightforward to prove that property ➂ still 585 holds under the model-based case.

586 In the model-free case, pi(z|x) = fϕ(*i, x*). Due to the lack of an environment model, even agents with 587 identical mappings may learn different representation distributions through their encoders, thus not 588 satisfying property ➂. However, as demonstrated in Section 4.2, although we cannot strictly determine agent homogeneity using d F 589 ij = 0, the heterogeneity distances measured between homogeneous 590 agents in the model-free case are sufficiently small. Moreover, the model-free manner is adequate 591 to distinguish between homogeneous and heterogeneous agents, and still maintains the ability to 592 quantify the degree of heterogeneity (as shown in Sections 4.2 and 6).

## 593 **F More Details Of Computing Heterogeneity Distance**

594 Here, we present five formulas for calculating heterogeneity distances, corresponding to the five types 595 of heterogeneity discussed in this paper. 596 Regarding **observation heterogeneity**, its relevant elements include the agent's observation space 597 and observation function. For two agents i and j, let their observation heterogeneity distance be denoted as d Ω
ij 598 . The corresponding calculation formula is:

$$d_{i j}^{\Omega}=\int_{\hat{s}\in\{S^{i}\}_{i\in N}}D\left[\Omega_{i}(\cdot|\hat{s}),\Omega_{j}(\cdot|\hat{s})\right]\cdot p(\hat{s})\,d\hat{s},$$
$$(12)$$
$$(13)$$

599 where D[·, ·] represents a measure of distance between two distributions, and p(·) is the probability 600 density function (this notation applies to subsequent equations). Here, sˆ denotes the global state,
{S
i 601 }i∈N represents the global state space, and Ωi and Ωj are the observation functions of agents i 602 and j, respectively.

603 Regarding **response transition heterogeneity**, its relevant elements include the agent's action space, 604 state space, and global state transition function. For two agents i and j, let their response transition heterogeneity distance be denoted as d T
ij 605 . The corresponding calculation formula is:

$$d_{i j}^{T}=\int_{\hat{a}\in\{S^{i}\}_{i\in N}}\int_{\hat{a}\in\{A^{i}\}_{i\in N}}D\left[\mathcal{T}^{i}(\cdot|\hat{s},\hat{a}),\mathcal{T}^{j}(\cdot|\hat{s},\hat{a})\right]\cdot p(\hat{s},\hat{a})\,d\hat{a}d\hat{s},$$

606 where p(·, ·) represents the joint probability density function. sˆ and aˆ denote the global state and global action respectively, {S
i}i∈N and {Ai 607 }i∈N represent the global state space and global action 608 space, and Ti and Tj are the local state transition functions of agents i and j, respectively. 609 Regarding **effect transition heterogeneity**, its relevant elements include the agent's action space, state space, and global state transition function. For convenience, we denote S
−i = ×k∈N,k̸=iS
k × S
E 610 as the joint state space of all agents except agent i, A−i = ×k∈N,k̸=iAk 611 as the joint action space of all agents except agent i, and T
−i 612 as the collection of state transitions excluding agent i. For two agents i and j, let their effect transition heterogeneity distance be denoted as d T
−
ij 613 . The corresponding 614 calculation formula is:

$$d_{ij}^{T^{-}}=\int_{s^{\prime}\in S^{(-)}}\int_{s\in A^{i}}\int_{a^{\prime}\in A^{(-)}}\int_{a\in A^{i}}D\left[\mathcal{T}^{-i}(\cdot|x),\mathcal{T}^{-j}(\cdot|x)\right]\cdot p(x)dada^{\prime}dsds^{\prime},\tag{14}$$

where for convenience, we denote x = (s
′*, s, a*′
615 , a), and p is the joint probability density function.

616 The calculation of effect transition heterogeneity distance differs from the previous two types of 617 heterogeneity distances in two significant ways. The first difference lies in its introduction of agent618 level elements as variables rather than global variables. When two agents have different agent-level 619 variable spaces, it becomes challenging to calculate the heterogeneity distance under this definition. 620 The second difference is that it involves a quadruple integral, making its computational complexity 621 much higher than the single or double integrals of the previous two distances.

622 These two differences make the calculation of effect transition heterogeneity distance more chal623 lenging. Fortunately, through our proposed meta-transition model, we can simplify the calculation 624 of effect transition heterogeneity distance to a double integral that only involves the agent's local 625 states and actions. Additionally, the distance measurement through representation learning also 626 reduces the constraints on the similarity of agents' variable spaces. Even when two agents have 627 different variable spaces (for example, one agent's local state space is 10-dimensional while another's 628 is 20-dimensional), we can still process the variable inputs through techniques like padding and then 629 map them to the same dimension using encoder networks. This demonstrates that the approach based 630 on representation learning and meta-transition significantly extends the applicability of heterogeneity 631 distance measurement, which also holds true in the quantification of heterogeneous types discussed 632 below. 633 Regarding **objective heterogeneity**, its relevant element is the agent's reward function. For two agents i and j, let their objective heterogeneity distance be denoted as d r ij 634 . The corresponding 635 calculation formula is:

$$d^{r}_{ij}=\int_{\hat{s}\in\{S^{i}\}_{i\in N}}\int_{\hat{a}\in\{A^{i}\}_{i\in N}}D\left[r^{i}(\cdot|\hat{s},\hat{a}),r^{j}(\cdot|\hat{s},\hat{a})\right]\cdot p(\hat{s},\hat{a})\,d\hat{a}d\hat{s},\tag{15}$$  $\hat{a}$ is the $i$th order and $N$th order in fraction $\hat{a}$ and $\hat{a}$ denotes the $i$th order but $\hat{a}$
636 where p(·, ·) represents the joint probability density function. sˆ and aˆ denote the global state and global action respectively, {S
i}i∈N and {Ai 637 }i∈N represent the global state space and global action 638 space, and ri and rj are the reward functions of agents i and j, respectively. 639 Regarding **policy heterogeneity distance**, its relevant elements include the agent's observation space, 640 action space, and policy function. For two agents i and j, let their policy heterogeneous distance be denoted as d π ij 641 . The corresponding calculation formula is:

$$d_{i j}^{\pi}=\int_{o\in O^{i}}D\left[\pi_{i}(\cdot|o),\pi_{j}(\cdot|o)\right]\cdot p(o)\,d o,$$
$$(16)$$
D [πi(·|o), πj (·|o)] · p(o) do, (16)
642 where D[·, ·] represents a measure of distance between two distributions, and p(·) is the probability density function. Here, o denotes the observation, Oi 643 represents the observation space, and πi and πj 644 are the policy functions of agents i and j, respectively.

## 645 **G Meta-Transition And Its Heterogeneity Distance**

646 To quantify an agent's comprehensive heterogeneity, we introduce the concept of meta-transition. 647 Meta-transition is a modeling approach that explores an agent's own attributes from its perspec648 tive. Our goal is to quantify an agent's comprehensive heterogeneity using only the agent's local 649 information (as global information is typically difficult to obtain in practical MARL scenarios).

650 Based on this, we provide the definition of meta-transition. Let the meta-transition of agent i be denoted as Mi. It is a mapping Mi: Si × Ai → Si × R × Ωi 651 . At time step t, the inputs of meta-transition are the agent's local state s itand local action a it 652 , and the outputs are the next time step's local state s it+1, the next time step's local observation o it+1 653 , and the current time step's reward r i 654 t based on the state and action.

655 We explain why the above relationship can reflect all agent-level elements in POMG. The input local 656 state and local action of meta-transition actually correspond to the inverse mapping to the global 657 state and global action. This inverse mapping potentially restores the local state and action to global 658 information, and then obtains the next time step's global state according to the global state transition 659 function, which is mapped to local observation through the observation function. Therefore, this 660 process reflects the agent's effect transition heterogeneity and observation heterogeneity. Additionally, 661 the potential global state and global action also determine the agent's local state and corresponding 662 reward at the next time step, which reflect the agent's response transition heterogeneity and objective 663 heterogeneity, respectively. 664 It is worth noting that meta-transition is not a function that actually exists in POMG, but an implicitly 665 defined mapping. We aim to quantify this mapping difference to capture the agent's comprehensive 666 heterogeneity. Therefore, meta-transition heterogeneity is quantified in a model-free manner. 667 Moreover, meta-transition is not limited to the aforementioned form. It can be transformed into dif668 ferent forms according to the modular settings of independent and dependent variables. For example, 669 by removing the agent's reward, meta-transition can reflect the agent's observation heterogeneity, 670 response transition heterogeneity, and effect transition heterogeneity. 671 After determining the input and output of meta-transition, the relevant heterogeneity distance can 672 be calculated using the same model-free method as before. Since meta-transition involves multiple 673 variables, and the dimensions between these variables may differ significantly (for example, the 674 dimension of reward is 1, while the dimension of observation might be 100), directly fitting with 675 deep networks may struggle to capture information corresponding to low-dimensional variables. We 676 address this issue through a dimension replication trick. In practice, we typically replicate the reward 677 dimension to be similar to the dimensions of observation or action, ensuring that the autoencoder 678 network can capture information related to objective heterogeneity during learning.

## 679 **H Derivation Of Elbo**

680 The Evidence Lower Bound (ELBO) of the likelihood can be derived as follows:

$$\log p(y|x)=\log\int p(y,z|x)dz$$ (a) $$=\log\int\frac{p(y,z|x)f_{\phi}(z|y,x)}{f_{\phi}(z|y,x)}dz$$ (b) $$=\log\mathbb{E}_{f_{\phi}(z|y,x)}\left[\frac{p(y,z|x)}{f_{\phi}(z|y,x)}\right]$$ (c) $$\geq\mathbb{E}_{f_{\phi}(z|y,x)}\left[\log\frac{p(y,z|x)}{f_{\phi}(z|y,x)}\right]$$ (d) $$=ELBO_{\text{model-based}},$$
$$(17)$$

681 where fϕ(z|*y, x*) represents the posterior probability distribution of the latent variable generated 682 by the encoder, and p(*y, z*|x) denotes a joint probability distribution concerning the customized 683 feature and latent variable, conditioned on o. Throughout the derivation of the formula, (a) employs 684 the properties of the joint probability distribution, (b) multiplies both numerator and denominator 685 by fϕ(z|*y, x*), (c) applies the definition of mathematical expectation, and (d) invokes the Jensen's 686 inequality.

687 Considering that the ELBO includes an unknown joint probability distribution, we can further 688 decompose it by using the posterior probability distributions from the encoder and decoder:

ELBOmodel-based = Efϕ(z|y,x) log p(y, z|x) fϕ(z|y, x)  = Efϕ(z|y,x) log gω(c|z, x)p(z|x) fϕ(z|y, x) (a) = Efϕ(z|y,x)[log gω(c|z, x)] + Efϕ(z|y,x) log p(z|x) fϕ(z|y, x) (b) = Efϕ(z|y,x)[log gω(c|z, x)] | {z } reconstruction term − DKL [fϕ(z|y, x)∥p(z|x)] | {z } prior matching term , (c)
$$(18)$$
689 where fϕ(z|*y, x*) and gω(c|*z, x*) are the posteriors from the encoder and decoder, respectively. The 690 conditional joint probability distribution p(*y, z*|x) is a imaginary construct in mathematical terms and 691 lacks practical significance. It can be formulated using the probability chain rule, constructed from the 692 posterior distribution of the customized feature and the prior distribution of the latent variable (step 693 (a)). Step (b) decomposes the expectation, and step (c) applies the definition of the KL divergence. 694 Thus, the ELBO can be decomposed into a reconstruction term of the customized feature, and a prior 695 matching term of the posterior and the prior. By maximizing the ELBO, the reconstruction likelihood 696 can be maximized while minimizing the KL divergence between the posterior and the prior. In the 697 model-free case, the same approach can be used to derive the ELBO and corresponding loss function.

## 698 **I Details Of Hetdps**

699 HetDPS is a novel algorithm designed to efficiently manage the allocation of neural network parame700 ters across multiple agents in MARL. This algorithm leverages the Wasserstein distance matrix to 701 cluster agents based on their similarities, and subsequently assigns them to suitable neural networks. 702 The pseudocode of HetDPS is shown in Algorithm 1. 703 The algorithm begins by computing the affinity matrix from the Wasserstein distance matrix, which 704 is then used as input to the Affinity Propagation clustering algorithm. This process yields a new 705 set of cluster assignments for the agents. If it is the first time the algorithm is executed, the cluster 706 assignments are directly used as network assignments. 707 In subsequent iterations, the algorithm compares the new cluster assignments with the previous ones 708 to determine the optimal network assignments. This is achieved by constructing an overlap matrix 709 that captures the similarity between the old and new cluster assignments. Based on the number of old 710 and new clusters, the algorithm handles three distinct cases: 711 1. Equal number of old and new clusters: In this scenario, the algorithm establishes a one-to-one 712 mapping between the old and new clusters using the Hungarian algorithm. It then constructs a 713 mapping from old clusters to networks and assigns each agent to a network based on its new cluster 714 assignment. 715 2. More new clusters than old clusters: When the number of new clusters exceeds the number of old 716 clusters, the algorithm handles network splitting. It uses the Hungarian algorithm to find the best 717 matching between old and new clusters and establishes a mapping from new clusters to old clusters. 718 For new clusters without a clear match, the algorithm either finds the most similar old cluster or 719 identifies the closest network. It then executes a splitting operation to copy parameters from the 720 source network to the new network. 721 3. More old clusters than new clusters: In this case, the algorithm handles network merging. It uses 722 the Hungarian algorithm to find the best matching between old and new clusters and establishes a 723 mapping from old clusters to new clusters. For each new cluster, it identifies the networks to be 724 merged and executes a merging operation based on the specified merge mode (majority, random, 725 average, or weighted). The algorithm then assigns each agent to a network based on its new cluster 726 assignment. 727 HetDPS offers a flexible and efficient approach to managing neural network parameters in multi-agent 728 systems. By dynamically adjusting network assignments based on agent similarities, the algorithm 729 enables effective parameter sharing and reduces the need for redundant computations.

## Algorithm 1 Hetdps

1: Initialize policies and parameter sharing paradigm 2: for episode = 1 to maxEpisodes do 3: Interact with environment to collect data 4: Add data to reinforcement learning (RL) sample pool 5: Add data to heterogeneity distance sample pool 6: if episode % trainingPeriod = 0 **then** 7: Update policies using RL sample pool 8: **end if**
9: if episode % quantizationPeriod = 0 **then**
10: Compute heterogeneity distance matrix D (Section 4) 11: Cluster agents using Affinity Propagation on D 12: if no previous clustering exists **then** 13: Assign networks to agents based on clusters 14: Copy network parameters as needed 15: **else** 16: Compute maximum overlap matching between current and previous clusters 17: if number of clusters unchanged **then** 18: Map new clusters to previous networks 19: **else if** new clusters > previous clusters **then**
20: Split networks: copy parameters for unmatched clusters 21: **else** 22: Merge networks: combine parameters based on merge mode 23: **end if** 24: Assign networks to agents 25: **end if** 26: **end if** 27: **end for**

## 730 **Neurips Paper Checklist**

731 1. **Claims** 732 Question: Do the main claims made in the abstract and introduction accurately reflect the 733 paper's contributions and scope? 734 Answer: [Yes] 735 Justification: Both the abstract and introduction clearly state that our main contribution and 736 scope: this work systematically establishes a theoretical framework for heterogeneity in 737 multi-agent reinforcement learning, advancing both theoretical development and practical 738 applications in this field. 739 Guidelines: 740 - The answer NA means that the abstract and introduction do not include the claims 741 made in the paper. 742 - The abstract and/or introduction should clearly state the claims made, including the 743 contributions made in the paper and important assumptions and limitations. A No or 744 NA answer to this question will not be perceived well by the reviewers.

745 - The claims made should match theoretical and experimental results, and reflect how 746 much the results can be expected to generalize to other settings. 747 - It is fine to include aspirational goals as motivation as long as it is clear that these goals 748 are not attained by the paper. 749 2. **Limitations** 750 Question: Does the paper discuss the limitations of the work performed by the authors? 751 Answer: [Yes] 752 Justification: We discuss the limitations in Section A. 753 Guidelines: