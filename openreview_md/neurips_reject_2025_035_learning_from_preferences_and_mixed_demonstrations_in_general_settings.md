# Learning From Preferences And Mixed Demonstrations In General Settings

Anonymous Author(s)
Affiliation Address email

## Abstract 16 **1 Introduction**

1 Reinforcement learning is a general method for learning in sequential settings, 2 but it can often be difficult to specify a good reward function when the task is 3 complex. In these cases, preference feedback or expert demonstrations can be 4 used instead. However, existing approaches utilising both together are either 5 ad-hoc or rely on domain-specific properties. Building upon previous work, we 6 develop a mathematical framework for learning from human data and based on 7 this we introduce LEOPARD: Learning Estimated Objectives from Preferences 8 And Ranked Demonstrations. LEOPARD can simultaneously learn from a broad 9 range of data, including negative/failed demonstrations, to effectively learn reward 10 functions in general domains. It does this by modelling the human feedback as 11 reward-rational partial orderings over available trajectories. We find that when a 12 limited amount of preference and demonstration feedback is available, LEOPARD 13 outperforms baselines by a significant margin. Furthermore, we use LEOPARD to 14 investigate learning from many types of feedback compared to just a single one, 15 and find that a combination of feedback types is often beneficial. 17 Reinforcement Learning (RL) is a branch of machine learning where an agent learns a behavioural 18 policy by interacting with an environment and receiving rewards. These rewards are determined by 19 a reward function that mathematically encodes the objective of the agent. For real-world practical 20 applications of RL, such as robotics or Large Language Model (LLM) finetuning, the specification of 21 the reward function poses a difficult challenge. Two popular RL subfields try to solve this problem by 22 leveraging human data in order to learn what the reward function should be, typically by optimising a 23 parameterised function such as a neural network. 24 Inverse RL (IRL) utilises human-provided demonstrations of the correct behaviour and tries to learn a 25 reward function for which only the demonstrations, or similar behaviour, are near-optimal (Ng et al., 26 2000; Ziebart et al., 2008; Wulfmeier et al., 2015). RL from Human Feedback (RLHF) presents the 27 human with pairs of agent–behaviour examples. For each pair, the human decides which piece of 28 behaviour is better, and the reward function is trained to re-produce this preference (Christiano et al., 29 2017). Both methods iterate between reward model and agent training. For more details on IRL 30 and RLHF, see Sections 2.1 and 2.2, respectively. For many applications it might be possible and 31 desirable to generate and learn from both of these feedback types, rather than committing to a single 32 one. The current standard approach is to first train on demonstrations and then finetune the resulting 33 model with preferences (Ibarz et al., 2018; Palan et al., 2019; Bıyık et al., 2022). Some methods 34 have been proposed to more effectively leverage the information encoded in both the preferences and 35 demonstrations, but this is still largely ad-hoc or specific to certain domains (Krasheninnikov et al., 36 2021; Mehta & Losey, 2023; Brown et al., 2019). We discuss these methods further in Section 2.3.

Step 1a:
Teacher provides demonstrations of good and bad behaviour, .

Step 1b:
Teacher samples agent's attempts and provides pairwise preference feedback.

Step 3:
Train agent via RL.

Go to step 1b and repeat.

Environment Step 2:
Feedback is used to train parameters for a reward function.

Mixed Demonstrations LEOPARD Reward Model Encoding RRPO Loss Minimisation Human Teacher Preferences Sample Agent Agent Trajectories
64 In summary, we make the following contributions: 65 1. We introduce RRPO, a practical and general framework for interpreting human feedback.

66 2. We introduce LEOPARD, an effective and scalable method for learning from preferences, 67 and positive/negative ranked demonstrations. 37 In an attempt to solve this problem for general domains—and for many types of feedback including 38 preferences and demonstrations—Jeon et al. (2020) propose Reward-Rational Choice (RRC). This 39 frames the human feedback data as Boltzmann-Rational choices according to a probability distribution 40 which has been induced by some unknown true reward function. Learning the reward function can 41 then be cast as a supervised learning problem where we try to replicate these choices. Unfortunately, 42 RRC is often difficult to implement in practice. For example, in the case of demonstration feedback, 43 they treat it as a choice over all possible behaviours. This space is incredibly difficult to optimise over 44 if it is very large and our reward function is non-linear, as is often the case for practical problems. 45 Additionally, it cannot encode multiple selections for the 'optimal choice', nor can it encode more 46 complex relationships between behaviours such as rankings or dis-preference. 47 To address these limitations, we introduce a new mathematical framework which frames the human 48 feedback as *reward-rational partial orderings* over trajectories (RRPO). These partial orderings are 49 then encoded by sets of Boltzmann-Rational choices, analogous to the Plackett-Luce ranking model 50 (Marden, 1996). From this we derive LEOPARD: Learning Estimated Objectives from Preferences 51 And Ranked Demonstrations, which is outlined in Figure 1. In addition to preferences and ranked 52 (positive) demonstrations, LEOPARD can also learn from ranked negative/failed demonstrations. 53 Preferences are interpreted as they are in RRC, but positive demonstrations are interpreted as 54 being preferred to the agent's current and future behaviour, or the opposite in the case of negative 55 demonstrations. Demonstration rankings, if available, are also cleanly translated into partial orderings. 56 LEOPARD can utilise a wide range of feedback types simultaneously, making it effective at learn57 ing useful reward functions in general environments. We find that when preference and positive 58 demonstration feedback is available, it outperforms the standard baseline of performing DeepIRL on 59 the demonstration data, and then finetuning using preferences. It also beats Adversarial Imitation 60 Learning with Preferences (AILP), another preference and positive demonstration learning algorithm.

61 Additionally, when only positive demonstration feedback is available, LEOPARD outperforms or 62 matches DeepIRL and AILP due to its ability to exploit ranking data. Finally, we use LEOPARD to 63 investigate learning from a variety of feedback types, compared to learning from a single one. 68 3. We investigate learning from many types of feedback vs focussing on only a single one.

## 69 **2 Related Work And Background** 70 **2.1 Demonstration-Based Rl**

71 A popular paradigm for learning from demonstrations is Inverse RL (IRL), where the demonstrations 72 are used to learn a reward function (Ng et al., 2000). This overcomes many issues of behavioural 73 cloning, which aims to directly mimic the given demonstrations (Bratko et al., 1995). Many current 74 methods for IRL are based on the principle of *maximum (causal) entropy* (MaxEnt; MCE), established 75 by Ziebart et al. (2008, 2010). This learns a reward function that captures the fact that the human 76 demonstrations are optimal, but beyond this, it tries to have as much uncertainty about the reward 77 dynamics as possible. Assuming a deterministic environment simplifies MCE into MaxEnt, and 78 this assumption has been used to extend this class of methods into settings with high-dimensional 79 observation spaces, e.g. DeepIRL (Wulfmeier et al., 2015). Advanced extensions of DeepIRL have 80 been proposed, leveraging methods such as importance sampling (Finn et al., 2016), or GAN-style 81 architectures (Fu et al., 2018). For a more comprehensive introduction to MCE and its derivatives, see 82 Gleave & Toyer (2022). Our proposed algorithm does not reduce to a MaxEnt-derived method in the 83 demonstration only case, but is still inspired by the principle and is of a similar form. Bayesian meth84 ods in IRL have also been explored (Ramachandran & Amir, 2007; Brown et al., 2020), highlighting 85 how a probabilistic framing of the inverse learning problem can be useful.

## 86 **2.2 Preference-Based Rl**

87 RLHF (Christiano et al., 2017) uses preferences—pairwise comparisons of agent behaviour—to learn 88 a reward function for high-dimensional RL environments via the Bradley-Terry preference model 89 (Bradley & Terry, 1952):

$$P_{\mathrm{RLHF}}(\tau_{a}\succ\tau_{b}|\theta)=\frac{\exp(R_{\theta}(\tau_{a}))}{\exp(R_{\theta}(\tau_{a}))+\exp(R_{\theta}(\tau_{b}))},$$
$$(1)$$

exp(Rθ(τa)) + exp(Rθ(τb)), (1)
where Rθ is a parameterised reward function and τa and τb are trajectory-fragments1 90 . A 3-step 91 iterative procedure is used: sampling of new comparisons of recent agent behaviour, fitting the reward 92 model to the comparison dataset, and training of the policy on the learnt reward function. The reward 93 model is fitted by minimising the average negative log-likelihood of the preference model across all 94 pairs of trajectory-fragments. Wirth et al. (2017) provides a survey of other preference based RL 95 methods prior to RLHF. 96 Recently, RLHF has been used for instruction and safety-finetuning large language models (LLMs) 97 into chat systems (Ouyang et al., 2022; Bai et al., 2022; Bahrini et al., 2023). These are referred to 98 as 'PPO-based' to disambiguate them from other methods which finetune LLMs from preferences 99 without learning a reward function, such as DPO (Rafailov et al., 2024). Often the LLM is trained on 100 demonstrations via behavioural cloning before PPO/DPO. Concerns for the safety, reliability, and 101 misuse of LLMs has led to a plethora of research on how best to utilise human preferences/rankings 102 to train these models (Cao et al., 2024; Chaudhari et al., 2024). Despite this, there is a broad lack of 103 principled use of other feedback types for LLM safety and finetuning.

## 104 **2.3 Combining Demonstrations And Preference Feedback**

105 As mentioned in the case for LLMs, demonstration and preference feedback are typically combined by 106 pre-training on the demonstration data using IRL/behavioural-cloning methods, and then finetuning 107 the resulting reward model on preferences using RLHF (Ibarz et al., 2018; Palan et al., 2019; Bıyık 108 et al., 2022). This works well in practice, but it is unclear how to add in further reward information, 109 such as negative demonstrations or the relative rankings of demonstrations. Additionally, information 110 that is present only in the demonstrations might be forgotten or never used, especially if strong 111 regularisation is applied to the reward model, or the RL policy does not sufficiently explore when 112 training on the demonstrations. 113 More sophisticated combinations of preferences and demonstrations have been considered. Krashenin114 nikov et al. (2021) sampled trajectories according to reward functions optimal for the preferences, and 115 applied MCE-IRL. This approach is computationally expensive and limited to linear reward functions 116 over tabular MDPs. Mehta & Losey (2023) combine preferences and demonstrations alongside 117 corrections (Bajcsy et al., 2017), but leverage domain-specific properties of robotics and encode 118 their demonstrations using trajectory-space perturbations. This method is not applicable outside of 119 robotics, and loses information about how demonstrations are better than most of trajectory-space, not 120 just better than nearby trajectories. Brown et al. (2019) and Brown & Niekum (2019) both subsample 121 ranked demonstrations to produce preferences for training the reward model, giving good results 122 but still losing information about how those demonstrations might be preferred to other trajectories.

123 Taranovic et al. (2022) combines a novel preference loss with adversarial imitation learning. This 124 is the closest to our work, and so we test against it as a baseline. We also note that none of these 125 methods can be easily extended to other types of feedback.

## 126 **2.4 Learning From Other Types Of Feedback** 148 **3 Method** 156 **3.1 Reward Rational Partial Orderings**

157 To ensure the general applicability of our theoretical formalisms, we assume that only the trajectories 158 our reward optimisation procedure has access to are provided directly. These could be generated 2They refer to these as 'failed demonstrations'.

127 Other types of feedback have been explored in isolation, such as negative demonstrations (Xie et al.,
2019),2 128 improvements (Jain et al., 2015), off-signals (Hadfield-Menell et al., 2017a), natural language 129 (Matuszek et al., 2012), proxy reward functions (Hadfield-Menell et al., 2017b), rankings (Myers 130 et al., 2022), scalar feedback (Knox & Stone, 2008; Wilde et al., 2021), and even the initial state (Shah 131 et al., 2019). Of these, Myers et al. (2022) is most similar to our work, as they use a Plackett-Luce 132 model to to interpret rankings to train a reward model. We differ by considering many more types 133 of feedback, showing how they can also be interpreted as orderings, and then use this to learn from 134 preferences and mixed demonstrations. 135 Jeon et al. (2020) interpret many of types of feedback as part of an overarching formalism, reward136 *rational (implicit) choice* (RRC), providing a mathematical theory for reward learning that combines 137 different types of feedback. RRC interprets each piece of human feedback as a Boltzmann-Rational 138 choice C from some (possibly implicit) set of choices D with rationality coefficient β. A grounding 139 function, ψ, maps choices to distributions over trajectories. The expected reward over these distribu140 tions gives the value for each choice under the Boltzmann-Rational model, according to some reward 141 function Rθ. For a deterministic ψ simplifies to:

$$P_{\mathrm{RRC}}(C|D,\theta)={\frac{\exp(\beta R_{\theta}(\psi(C)))}{\sum_{C^{\prime}\in{\mathcal{D}}}\exp(\beta R_{\theta}(\psi(C^{\prime})))}}.$$
$$(2)$$

PC′∈D exp(βRθ(ψ(C′))). (2)
142 Many of the formalisms of feedback in RRC, such as demonstrations, are not generally directly 143 applicable as they naively require a large—possibly infinite—set of choices. Practical applications 144 may rely on finite state-spaces, linear reward functions, unbounded surrogate loss functions, or 145 sampling-based methods, each with their own pros and cons. We take inspiration from RRC, but 146 show that formulating feedback as orderings leads to some more natural interpretations for mixed 147 demonstrations without the need for such additional methods. 149 We propose LEOPARD, a method for learning from preferences, positive demonstrations, negative 150 demonstrations, and partial rankings over the given demonstrations. It is practical, flexible, and 151 applicable to many environments. The aim is that a practitioner can give any and all feedback possible 152 to the learning algorithm, and this feedback can be continuously learnt from and added to. First, we 153 develop a general mathematical framework, reward-rational partial ordering (RRPO), extending that 154 of deterministic reward-rational choice (RRC, Jeon et al. (2020)). Then, we apply this to the specific 155 case of learning from preferences and mixed demonstrations.

159 during the agent's training or provided by the human in the case of demonstrations. This is assumed 160 as sensible/relevant trajectories could sit on an unknown manifold in (a high-dimensional) observation
space, crippling random-sampling based approaches. 161 3 We'd expect that reward functions capturing
162 complex desirable behaviour would not be linear, but that they could at least be approximated 163 sufficiently by some differentiable parameterised function. 164 Our key insight is to interpret human feedback as a set of Boltzmann-Rational choices encoding 165 strict partial orderings over the trajectory-fragments we have direct access to, where a fragment 166 is a contiguous subsequence of a trajectory. For each item in the partial order, we 'choose' that 167 element out of a set containing itself and all elements strictly less than it. This is analogous to the 168 Plackett-Luce ranking model (Marden, 1996), and is equivalent when the ordering can be viewed as 169 a total ordering embedded in some larger set. Similar to RRC, each partial ordering is assumed to 170 be independent given the reward function. Since a partial order may encode a single element being 171 greater than all others with no other relations, this generalises deterministic choices of RRC.
172 Formally, let D = {τi}i be the set of all possible fragments of trajectories we have access to, 173 C = {<j}j the set of human feedback, and Rθ our non-linear reward function parameterised by θ. Note that <i 174 is used to denote some partial ordering i. We define the likelihood of θ under RRPO as
175 follows:
$$P_{\mathrm{RRPO}}({\mathcal{C}}|{\mathcal{D}},\theta)=\prod_{(\tau_{i},<_{j})\in{\mathcal{D}}\times{\mathcal{C}}}P(<_{j}|\tau_{i}),$$
P(<j |τi), (3)
$$({\mathfrak{I}})$$
$$P(<_{j}|\tau_{i})=\frac{e^{\beta_{j}R_{\theta}(\tau_{i})}}{e^{\beta_{j}R_{\theta}(\tau_{i})}+\sum_{\tau_{k}\in\mathcal{D}}\mathbf{1}_{\tau_{k}<_{j}\tau_{i}}e^{\beta_{j}R_{\theta}(\tau_{k})}},$$
$$(4)$$
$\left(5\right)$. 
, (4)

## 190 **3.2 Leopard**

191 Whilst we can apply the framework above to many types of feedback, we now focus on the case of 192 combining preferences with mixed demonstrations. By mixed demonstrations, we mean ones which 193 may be positive, negative and, within these two groups, we may have access to the relative rankings 194 of each demonstration.

A pairwise preference of τa ≻ τb is simply interpreted as a partial ordering with only τb < τa.

4 195 Posi196 tive demonstrations are interpreted as a single partial ordering that prefers all positive demonstrations 197 to any agent trajectories and encodes the relative rankings of the positive demonstrations themselves. 198 Negative demonstrations are interpreted likewise, but these partial orderings prefer agent trajectories 199 over the negative demonstrations.

Formally, let Dpos, <pos 200 , and Dneg, <neg be the sets of trajectories and partial orderings encoding 201 rankings from positive and negative demonstrations, respectively. Let Dagent be the set of trajectories 3For example, consider the space of all images vs ones which are plausible 3D scenes. 4By interpreting each preference as its own partial ordering, we avoid potential issues of symmetry and non-transitivity.

176 where βj is the rationality coefficient for feedback j. βs should be equal if the type and source of 177 feedback is the same, e.g. two pairwise preferences given by the same person. Note that when the 178 partial orderings are sparse, many terms of the product become unity and can be ignored. We perform 179 gradient descent on the negative-log of Equation (4) combined with a regularising term, giving the 180 loss function below:
LRRPO(θ) = − log PRRPO(C|D, θ) + LSmooth(D, θ). (5)
181 The smoothing term penalises the first derivative of the reward function over trajectories and leads 182 to better shaped reward functions that are easier for the RL agent to learn from. It is inspired by 183 previous work (Finn et al., 2016), and empirically we found it works well. Specific details are given 184 in Section A.1.3.

185 A nice property of LRRPO is that when minimised it faithfully represents the partial orderings. More 186 precisely, upper bounds on the loss give rise to lower bounds on all reward differences between 187 fragments that are related by some partial ordering. This is stated formally and proved in Theorem D.1 188 of Appendix D. As a special case, if the loss is below log 2 then all reward differences must have the 189 correct sign, i.e. the reward function induces an ordering compatible with all the partial orderings.

$$(6)$$
$$\begin{array}{l}{{{\mathcal{C}}=\{{\leq}_{\mathrm{Demo}}\}\cup{\mathcal{C}}_{\mathrm{Pref}},}}\\ {{{\mathcal{D}}=\bigcup\{{\mathcal{D}}_{\mathrm{pos}},{\mathcal{D}}_{\mathrm{neg}},{\mathcal{D}}_{\mathrm{agent}},{\mathcal{D}}_{\mathrm{pref}}\},}}\end{array}$$
$$\left(7\right)$$

C = {<Demo*} ∪ C*Pref, (6)
D =[{Dpos, Dneg, Dagent, Dpref}, (7)
<Demo= <pos ∪ <neg ∪ {τn < τa < τp, |(τn, τa, τp) ∈ Dneg × Dagent × Dpos}, CPref = {{τb < τa}|(τa, τb) ∈ P}, Dpref =[ (τa,τb)∈P {τa, τb}.

## 210 **4 Experiments** 211 **4.1 Experimental Setup**

202 sampled from the agent's behaviour. Let P = {(τa, τb)i}i be the set of ordered pairs of trajectory203 fragments in which the first is preferred, and Rθ our parameterised reward function. Then we optimise the loss function, Equation (5), with:5 204 205 where the demonstration and preference partial orderings are given by: 206 Like in the case for RLHF, our dependencies on agent behaviour means we need to iterate between sampling new preferences, optimising for Equation (5), and training the agent's policy.6 207 Our algorithm 208 is illustrated in Figure 1 and the full training procedure is given in Algorithm 1 in Appendix A, along 209 with details on reward model training. 212 We test our method on several environments against common baselines in order to evaluate its perfor213 mance across a broad variety of domains. Additionally, we also vary the proportions and amounts 214 of different types of feedback used for learning to investigate the effects of this on performance. 215 In order to reduce the cost of testing our method and facilitate hyperparameter tuning with many 216 repetitions, we synthetically generate preferences, demonstrations, and their rankings. We generate 217 preferences by sampling using the sigmoid of the reward difference between the two fragments under 218 comparison as the probability of preference. We generate demonstrations by training an agent on 219 the ground truth reward function and then sampling its trajectories, with their ground truth reward 220 determining their relative rankings. For further details, see Section A.2. For each combination of 221 environment, algorithm, and amount of feedback, we run 16 random seeds and report the average 222 results with 1-σ standard error. Standard errors are computed via the typical method of dividing the 223 empirical variance by the square root of the sample size. 224 We evaluate on four environments from the Gymnasium (Towers et al., 2024) test suite: Half Cheetah 225 (MuJoCo), Cliff Walking (Toy Text), Lunar Lander (Box2D), and Ant (MuJoCo). This covers a range 226 of continuous and discrete observation and action spaces, reward sparsities, and overall complexities.

227 We require a finite horizon to reduce complications from the preference and demonstration learning, so 228 some environments required modification. These and other environment details and hyperparameters 229 are given in Appendix B. 230 In order to get a good number of preferences and demonstrations to test with, we see how many 231 preferences or positive demonstrations LEOPARD needs to get good performance in the single feedback type case, and then use a normalised weighted combination of these.7 232 This allows us to be 233 confident there is enough feedback for learning, but not so much that it's too easy.

1 2 3 4 5 6 7 8 Iteration 1000 0 1000 2000 3000 4000 5000 6000 1 2 3 4 5 6 7 8 Iteration 2000 1500 1000 500 0 500 Mean Gr ound Truth Reward LEOPARD DeepIRL then RLHF, best AILP, best Mean Gr ound Truth Reward LEOPARD
DeepIRL then RLHF, best AILP, best
(a) Half Cheetah, ndemos = 4, nprefs = 256
(b) Cliff Walking, ndemos = 2, nprefs = 64 1 2 3 4 5 6 7 8 Iteration 3000 2500 2000 1500 1000 500 0 1 2 3 4 5 6 7 8 Iteration 1000 0 1000 2000 3000 Mean G
round Tr uth Rew ard LEOPARD
DeepIRL then RLHF, best AILP, best Mean G
round Tr uth Rew ard LEOPARD
DeepIRL then RLHF, best AILP, best
Figure 2: Comparison of LEOPARD with the baselines of AILP and DeepIRL followed by RLHF, when positive demonstrations and preferences are available. The lines denote the mean of the ground truth reward function, with shaded standard errors across 16 random seeds, against algorithm iterations—alternations between optimising the reward model and the agent. Solid lines are smoothed means for clarity, dashed lines give raw values. A breakdown of the performance of the baseline methods for different reward model training epochs per iteration is given in Figures 7 and 8.

## 234 **4.2 Leopard Vs Baselines**

235 In Figure 2 we compare LEOPARD against Adversarial Imitation Learning with Preferences (AILP,
Taranovic et al. (2022))8 236 and a standard pipeline of training on demonstrations with DeepIRL and 237 then preference finetuning with RLHF. 238 We see that without exception LEOPARD outperforms both baselines by a considerable margin. 239 Since LEOPARD can utilise all the data all the time, preferences can be used to aid early exploration, 240 and demonstrations can continue to be trained against even in the latter stages. Rankings over 241 demonstrations provide an additional information source the baselines are unable to make use 242 of. Additionally, as it trains the reward model to rough convergence each iteration it allows for 243 adequate learning without over-fitting, and does not require tuning a 'reward model training epochs' 244 hyperparameter. 245 When training the reward model with LEOPARD, we keep training until the loss has loosely converged 246 (see Section A.1.2 for details). This is not possible with DeepIRL as the maximum-entropy 'loss' 247 function is not bounded from below, thus the number of training epochs for the reward model is fixed. 248 We try a variety of values and compare against the best, for a full breakdown see Figure 7. For AILP, 249 we try using both our dynamic stopping and a fixed number of training epochs again comparing 250 against the best, see Figure 8 in Appendix C for a breakdown of these results. 251 Whilst not the focus of our algorithm, we additionally show that with only positive demonstrations 252 LEOPARD either beats or performs similarly to the baselines. This is shown in Appendix C, Figure 6, 253 with the breakdowns of DeepIRL and AILP's results for different numbers of training epochs given 254 in Figures 9 and 10 respectively. 255 Table 2 in Appendix C gives a numerical breakdown of final scores for each algorithm in each 256 environment, including the different settings of AILP and DeepIRL. 257 Note that for the analysis of the Cliff Walking environment, outliers have been removed These were 258 due to excessively large negative rewards from walking off the cliff many times before learning this 259 was bad. A detailed breakdown is given in Appendix C, Table 4. 260 **4.3 Learning from a Mixture of Feedback Types**

1 2 3 4 5 6 7 8 Iteration 1500 1000 500 0 500 1000 1 2 3 4 5 6 7 8 Iteration 0 1000 2000 3000 4000 5000 6000 Mean Groun d Truth Rewa rd Mean Groun d Truth Rewa rd 512 Prefs 256 Prefs, 4 Pos Demos 256 Prefs, 2 Pos Demos, 2 Neg Demos 8 Pos Demos 4 Pos Demos, 4 Neg Demos 128 Prefs 64 Prefs, 2 Pos Demos 64 Prefs, 1 Pos Demo, 1 Neg Demo 4 Pos Demos 2 Pos Demos, 2 Neg Demos
(a) Half Cheetah
(b) Cliff Walking 1 2 3 4 5 6 7 8 Iteration 600 500 400 300 200 100 0 Mean Grou nd Trut h Rew ard 1024 Prefs 512 Prefs, 4 Pos Demos 512 Prefs, 2 Pos Demos, 2 Neg Demos 8 Pos Demos 4 Pos Demos, 4 Neg Demos 1 2 3 4 5 6 7 8 Iteration 1000 0 1000 2000 3000 Mean Grou nd Trut h Rew ard 512 Prefs 256 Prefs, 4 Pos Demos 256 Prefs, 2 Pos Demos, 2 Neg Demos 8 Pos Demos 4 Pos Demos, 4 Neg Demos

(c) Lunar Lander
(d) Ant
Figure 3: Comparison of LEOPARD's performance when varying types of feedback are available. The lines denote the mean of the ground truth reward function, with shaded standard errors across 16 random seeds, against algorithm iterations—alternations between optimising the reward model and the agent. Solid lines are smoothed means for clarity, dashed lines give raw values.

261 In Figure 3 we investigate the performance of LEOPARD when learning from a variety of different 262 feedback proportions. Final scores are detailed in Appendix C, Table 3. The results are somewhat 263 mixed and noisy, but we see that preferences combined with positive demonstrations consistently 264 performs well.

## 265 **5 Discussion** 266 **5.1 Generality Of Rrpo**

267 Reward-rational preference orderings, the basis of LEOPARD, are a generalisation of the deterministic 268 reward-rational choice framework (Jeon et al., 2020), but offers several distinct advantages. Recall 269 that RRC frames the human feedback as a choice over some set, and then maps elements of that set 270 into distributions over trajectories. Instead, RRPO maps the human feedback directly into a set of 271 partial orderings. These two approaches have differing flexibility, and different feedback types might 272 lend themselves more readily to one or the other. However, as RRPO is explicit in its construction 273 that it operates only over directly-accessible trajectories, it becomes much more general in a practical 274 sense. 275 Furthermore, RRPO does not assume any particular properties about the space of reward functions, 276 nor the space of trajectories. In general, one can think of optimal trajectories as a small part of 277 some feasible-trajectory manifold, which itself is a small part in a larger trajectory feature space.

278 Methods which rely on domain-specific properties of these spaces, such as linearity or computable 279 perturbations, inherently limit themselves from being more broadly applied. For example, Mehta & 280 Losey (2023) leverages inverse kinematics models to interpret demonstration feedback (alongside 281 preferences) in robotics domains. Whilst effective for this application, it renders the broader method 282 impossible outside of robotics. RRPO and LEOPARD on the other hand, could be easily applied to 283 environments very different to the ones we have tested on. For example, they could be used for Large 284 Language Model (LLM) finetuning.

## 285 **5.2 Limitations And Future Work**

286 Whilst we have tested LEOPARD on a range of environments with differently structured observation 287 and action spaces, a more comprehensive study would investigate an even wider range of tasks, such 288 as more complex robotics, Atari games, and even LLM finetuning. Furthermore, with additional 289 resources, it would be instructive to more closely interrogate how performance depends on the 290 proportions of different feedback used for learning. For instance, future work could vary the feedback 291 proportions with greater precision and then fit and analyse simple predictive models on this. 292 Additionally, there are other methods that seek to learn from both preference and demonstration data, 293 or even negative/failed demonstrations, as detailed in Sections 2.3 and 2.4. Whilst these are less 294 general in application than LEOPARD; a comparison of performance would still be interesting. We 295 have chosen the baselines of AILP and 'DeepIRL followed by RLHF' to test against as they have 296 similar simplicity and generality to our own method, as well as the latter being common practice. 297 We introduce RRPO as a theoretical backdrop for LEOPARD, however our investigation of its 298 properties and encodings for many types of feedback is limited. Due to its similarity to RRC and the 299 Placket-Luce choice model, we do not see this as a critical failing, as it will inherit many properties 300 from those models, and deterministic RRC formulations can be trivially encoded under RRPO. 301 Nevertheless, there are likely important theoretical properties and applications of RRPO that are of 302 relevance to reward learning that ought to be investigated.

## 303 **6 Conclusion**

304 We have shown that LEOPARD can perform effective reward inference, learning from many sources 305 of reward information simultaneously. It is more effective than standard baselines for learning 306 from preferences and demonstrations, and can additionally incorporate more information such as 307 demonstration rankings and negative/failed demonstrations. We have also investigated how many 308 sources of reward information could be more beneficial than relying on only large amounts of a single 309 type. The generality and simplicity of our method makes it very powerful and applicable to important 310 current problems such as high dimensional robotics, and LLM finetuning. Furthermore, it opens the 311 door to exploring the use of a much wider range of feedback in many RL settings.

## 312 **References**

313 Bahrini, A., Khamoshifar, M., Abbasimehr, H., Riggs, R. J., Esmaeili, M., Majdabadkohne, R. M., and 314 Pasehvar, M. Chatgpt: Applications, opportunities, and threats. In *2023 Systems and Information* 315 *Engineering Design Symposium (SIEDS)*, pp. 274–279. IEEE, 2023. 316 Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., 317 Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from 318 human feedback. *arXiv preprint arXiv:2204.05862*, 2022.

319 Bajcsy, A., Losey, D. P., O'malley, M. K., and Dragan, A. D. Learning robot objectives from physical 320 human interaction. In *Conference on robot learning*, pp. 217–226. PMLR, 2017. 321 Bıyık, E., Losey, D. P., Palan, M., Landolfi, N. C., Shevchuk, G., and Sadigh, D. Learning reward 322 functions from diverse sources of human feedback: Optimally integrating demonstrations and 323 preferences. *The International Journal of Robotics Research*, 41(1):45–67, 2022.

324 Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, 325 A., VanderPlas, J., Wanderman-Milne, S., and Zhang, Q. JAX: composable transformations of 326 Python+NumPy programs, 2018. URL http://github.com/jax-ml/jax. 327 Bradley, R. A. and Terry, M. E. Rank analysis of incomplete block designs: I. the method of paired 328 comparisons. *Biometrika*, 39(3/4):324–345, 1952. 329 Bratko, I., Urbanciˇ c, T., and Sammut, C. Behavioural cloning: phenomena, results and problems. ˇ 330 *IFAC Proceedings Volumes*, 28(21):143–149, 1995. 331 Brown, D., Goo, W., Nagarajan, P., and Niekum, S. Extrapolating beyond suboptimal demonstrations 332 via inverse reinforcement learning from observations. In *International conference on machine* 333 *learning*, pp. 783–792. PMLR, 2019. 334 Brown, D., Niekum, S., and Petrik, M. Bayesian robust optimization for imitation learning. *Advances* 335 *in Neural Information Processing Systems*, 33:2479–2491, 2020. 336 Brown, D. S. and Niekum, S. Deep bayesian reward learning from preferences. *arXiv preprint* 337 *arXiv:1912.04472*, 2019. 338 Cao, B., Lu, K., Lu, X., Chen, J., Ren, M., Xiang, H., Liu, P., Lu, Y., He, B., Han, X., et al. Towards 339 scalable automated alignment of llms: A survey. *arXiv preprint arXiv:2406.01252*, 2024. 340 Chaudhari, S., Aggarwal, P., Murahari, V., Rajpurohit, T., Kalyan, A., Narasimhan, K., Deshpande, 341 A., and da Silva, B. C. Rlhf deciphered: A critical analysis of reinforcement learning from human 342 feedback for llms. *arXiv preprint arXiv:2404.08555*, 2024. 343 Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. Deep reinforcement 344 learning from human preferences. *Advances in neural information processing systems*, 30, 2017.

345 DeepMind, Babuschkin, I., Baumli, K., Bell, A., Bhupatiraju, S., Bruce, J., Buchlovsky, P., Budden, 346 D., Cai, T., Clark, A., Danihelka, I., Dedieu, A., Fantacci, C., Godwin, J., Jones, C., Hemsley, R., 347 Hennigan, T., Hessel, M., Hou, S., Kapturowski, S., Keck, T., Kemaev, I., King, M., Kunesch, M., 348 Martens, L., Merzic, H., Mikulik, V., Norman, T., Papamakarios, G., Quan, J., Ring, R., Ruiz, F., 349 Sanchez, A., Sartran, L., Schneider, R., Sezener, E., Spencer, S., Srinivasan, S., Stanojevic, M., ´ 350 Stokowiec, W., Wang, L., Zhou, G., and Viola, F. The DeepMind JAX Ecosystem, 2020. URL
351 http://github.com/google-deepmind.

352 Finn, C., Levine, S., and Abbeel, P. Guided cost learning: Deep inverse optimal control via policy 353 optimization. In *International conference on machine learning*, pp. 49–58. PMLR, 2016. 354 Fu, J., Luo, K., and Levine, S. Learning robust rewards with adversarial inverse reinforcement 355 learning, 2018. URL https://arxiv.org/abs/1710.11248.

356 Gleave, A. and Toyer, S. A primer on maximum causal entropy inverse reinforcement learning, 2022.

357 URL https://arxiv.org/abs/2203.11409. 358 Hadfield-Menell, D., Dragan, A., Abbeel, P., and Russell, S. The off-switch game. In *Workshops at* 359 *the Thirty-First AAAI Conference on Artificial Intelligence*, 2017a.

360 Hadfield-Menell, D., Milli, S., Abbeel, P., Russell, S. J., and Dragan, A. Inverse reward design.

361 *Advances in neural information processing systems*, 30, 2017b. 362 Heek, J., Levskaya, A., Oliver, A., Ritter, M., Rondepierre, B., Steiner, A., and van Zee, M. Flax: A
363 neural network library and ecosystem for JAX, 2024. URL http://github.com/google/
364 flax. 365 Ibarz, B., Leike, J., Pohlen, T., Irving, G., Legg, S., and Amodei, D. Reward learning from human 366 preferences and demonstrations in atari. *Advances in neural information processing systems*, 31, 367 2018. 368 Jain, A., Sharma, S., Joachims, T., and Saxena, A. Learning preferences for manipulation tasks from 369 online coactive feedback. *The International Journal of Robotics Research*, 34(10):1296–1313, 370 2015. 371 Jeon, H. J., Milli, S., and Dragan, A. Reward-rational (implicit) choice: A unifying formalism for 372 reward learning. *Advances in Neural Information Processing Systems*, 33:4415–4426, 2020. 373 Knox, W. B. and Stone, P. Tamer: Training an agent manually via evaluative reinforcement. In *2008* 374 *7th IEEE international conference on development and learning*, pp. 292–297. IEEE, 2008. 375 Krasheninnikov, D., Shah, R., and van Hoof, H. Combining reward information from multiple 376 sources. *arXiv preprint arXiv:2103.12142*, 2021. 377 Marden, J. I. *Analyzing and modeling rank data*. CRC Press, 1996. 378 Matuszek, C., FitzGerald, N., Zettlemoyer, L., Bo, L., and Fox, D. A joint model of language and 379 perception for grounded attribute learning. *arXiv preprint arXiv:1206.6423*, 2012.

380 Mehta, S. A. and Losey, D. P. Unified learning from demonstrations, corrections, and preferences 381 during physical human-robot interaction. *ACM Transactions on Human-Robot Interaction*, 2023. 382 Myers, V., Biyik, E., Anari, N., and Sadigh, D. Learning multimodal rewards from rankings. In 383 *Conference on robot learning*, pp. 342–352. PMLR, 2022.

384 Ng, A. Y., Russell, S., et al. Algorithms for inverse reinforcement learning. In Icml, volume 1, pp. 2, 385 2000. 386 Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., 387 Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. 388 *Advances in neural information processing systems*, 35:27730–27744, 2022. 389 Palan, M., Landolfi, N. C., Shevchuk, G., and Sadigh, D. Learning reward functions by integrating 390 human demonstrations and preferences. *arXiv preprint arXiv:1906.08928*, 2019. 391 Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference 392 optimization: Your language model is secretly a reward model. *Advances in Neural Information* 393 *Processing Systems*, 36, 2024. 394 Raffin, A. Rl baselines3 zoo. https://github.com/DLR-RM/rl-baselines3-zoo, 395 2020. 396 Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., and Dormann, N. Stable-baselines3: 397 Reliable reinforcement learning implementations. *Journal of Machine Learning Research*, 22 398 (268):1–8, 2021. URL http://jmlr.org/papers/v22/20-1364.html. 399 Ramachandran, D. and Amir, E. Bayesian inverse reinforcement learning. In *IJCAI*, volume 7, pp. 400 2586–2591, 2007.

401 Shah, R., Krasheninnikov, D., Alexander, J., Abbeel, P., and Dragan, A. Preferences implicit in the 402 state of the world. *arXiv preprint arXiv:1902.04198*, 2019. 403 Taranovic, A., Kupcsik, A. G., Freymuth, N., and Neumann, G. Adversarial imitation learning with 404 preferences. In *The Eleventh International Conference on Learning Representations*, 2022. 405 Todorov, E., Erez, T., and Tassa, Y. Mujoco: A physics engine for model-based control. In *2012* 406 *IEEE/RSJ International Conference on Intelligent Robots and Systems*, pp. 5026–5033. IEEE, 407 2012. doi: 10.1109/IROS.2012.6386109. 408 Towers, M., Kwiatkowski, A., Terry, J., Balis, J. U., De Cola, G., Deleu, T., Goulão, M., Kallinteris, 409 A., Krimmel, M., KG, A., et al. Gymnasium: A standard interface for reinforcement learning 410 environments. *arXiv preprint arXiv:2407.17032*, 2024.

411 Wilde, N., Bıyık, E., Sadigh, D., and Smith, S. L. Learning reward functions from scale feedback.

412 *arXiv preprint arXiv:2110.00284*, 2021. 413 Wirth, C., Akrour, R., Neumann, G., Fürnkranz, J., et al. A survey of preference-based reinforcement 414 learning methods. *Journal of Machine Learning Research*, 18(136):1–46, 2017. 415 Wulfmeier, M., Ondruska, P., and Posner, I. Deep inverse reinforcement learning. *CoRR,* 416 *abs/1507.04888*, 2015. 417 Xie, X., Li, C., Zhang, C., Zhu, Y., and Zhu, S.-C. Learning virtual grasp with failed demonstrations 418 via bayesian inverse reinforcement learning. In *2019 IEEE/RSJ International Conference on* 419 *Intelligent Robots and Systems (IROS)*, pp. 1812–1817. IEEE, 2019. 420 Ziebart, B. D., Maas, A. L., Bagnell, J. A., Dey, A. K., et al. Maximum entropy inverse reinforcement 421 learning. In *Aaai*, volume 8, pp. 1433–1438. Chicago, IL, USA, 2008.

422 Ziebart, B. D., Bagnell, J. A., and Dey, A. K. Modeling interaction via the principle of maximum 423 causal entropy. 2010.

## 424 **A Algorithm Details**

425 The full algorithm for LEOPARD is given in Algorithm 1. Initialisations follow standard neural 426 network initialisation methods. RandomRollouts generates trajectories by sampling random actions 427 and resetting the environment when necessary. TrainAgent uses the standard SAC algorithm for 428 when the action space is continuous, and PPO when it's discrete. For both algorithms we use the 429 implementations provided by Stable Baselines3 (Raffin et al., 2021). It uses the learnt reward function 430 to generate rewards for the RL procedure. Hyperparameters used for SAC and PPO are those given in 431 RL Baselines3 Zoo (Raffin, 2020), except for Lunar Lander where we use an entropy bonus of 0.05 432 instead of 0. Details on TrainRewardModel and GetPreferences are given in Sections A.1 and A.2.1 433 respectively. The generation of the demonstrations and their rankings is detailed in Section A.2.2.

| Algorithm 1 LEOPARD Input: niters   | Number of iterations to perform          |
|-------------------------------------|------------------------------------------|
| nrollout-steps                      | Number of environment rollout steps      |
| nprefs                              | Number of preferences to sample          |
| Dpos                                | Positive demonstrations                  |
| <pos                                | Positive demonstrations partial ordering |
| Dneg                                | Negative demonstrations                  |
| <neg                                | Negative demonstrations partial ordering |
| Output: π                           | Trained agent policy                     |

Output:

π Trained agent policy Rθ Learnt reward function

nrollout-steps-per-iter ← ⌊nrollout-steps/(niters + 1)⌋ nprefs-per-iter ← ⌊nprefs/niters⌋ Dagent ← ∅ {Agent trajectory pool}
P ← ∅ {Preferences dataset} π ← InitialiseAgent()
Rθ ← InitialiseRewardFunction() Dnew-trajectories ← RandomRollouts(nrollout-steps-per-iter) for i = 1 to niters do P ← P ∪ GetPreferences(nprefs-per-iter, Dnew-trajectories, Dagent) Dagent ← Dagent ∪ Dnew-trajectories Rθ ← TrainRewardModel(Rθ, Dpos, <pos, Dneg, <neg, Dagent,P) π, Dnew-trajectories ← TrainAgent(π, Rθ, nrollout-steps-per-iter)
end for

## 434 **A.1 Reward Model Training**

435 The reward model is trained by optimising the loss function Equation (5) with the AdamW optimiser.

Batches of Dpos, Dneg, Dagent 436 , and P are sampled independently, and then encoded via Equations (6)
and (7). Since we want to respect the relative proportions of each data source9 437 but also have 438 independent batch sizes, normalisation of the loss across the batch is slightly involved. This is 439 detailed in Section A.1.1. Instead of training for a fixed number of steps / epochs, training steps are 440 taken until a stopping condition is reached, as detailed in Section A.1.2. Together these procedures could result in varying coverages for each data source, from potentially many epochs on one,10 441 to 442 only sampling a small fraction of another.

## 443 **A.1.1 Loss Normalisation Across Batch**

444 As we want our gradient steps to be roughly unity in magnitude and independent of the batch size, 445 we need to normalise it. Typically, this is very easy in supervised learning—one can simply take 446 an average across the batch—but this is not the case for Equation (5). Expansion of the gradient of 447 the loss with respect to θ, and noting our reward function operates at the level of transitions within 448 trajectories, reveals the normalising factor of each data source (note this assumes a fixed length of 449 fragments for each partial ordering):

$$\sum_{(\tau_{i},<_{j})\in{\mathcal{D}}\times{\mathcal{C}}}\mathrm{Length}(\tau_{i})\cdot\mathbf{1}_{\exists\tau_{k}\in{\mathcal{D}}.\tau_{k}\neq\tau_{i}\land\tau_{k}<_{j}\tau_{i}}.$$

450 The loss term of each data source is first divided by this factor evaluated on the batch—so that they 451 are all at most unity in magnitude—and then combined in a weighted sum where the weights are 452 the factors evaluated on the whole dataset for that source divided by the sum of these dataset-level factors. Some data sources, namely Dagent 453 , are treated as 'in-excess', and their dataset-level factor is made proportional to another data source, e.g. Dpos 454 .

## 455 **A.1.2 Stopping Conditions**

456 Generally, the reward function loss from poorly-fitted demonstration rankings are much higher than 457 poorly fitted preferences. This is because trajectories are typically longer than trajectory-fragments 458 and demonstrations generate more '<' comparisons than a preference. However, the distribution of 459 demonstrations are typically quite far from that of the agent trajectories, which the preferences have 460 been generated over. This makes it much easier for the reward function to separate the demonstrations 461 from agent behaviour and thus achieve a low loss on the demonstration ordering, than it does for it to 462 get low loss on all the preference orderings. 463 The consequence of the above two facts is that if we were training on just the demonstrations, we'd 464 want to do at most a few epochs (to learn fast and avoid overfitting), but if we were training on just 465 the preferences we might want to do more (as learning is slower and overfitting less of a potential 466 issue). Thus, as the amount of data in each dataset varies in each iteration, it does not make sense to 467 have a pre-specified number of training steps, and instead a stopping condition should be used. 468 Our stopping condition simply checks if the training loss has loosely converged. At each step we 469 check if the change in training loss is less than 10% of the last step's training loss. If this occurs 3 470 times in a row, we stop training the reward model for that iteration, and return to agent training. There 471 is a hard limit of 256 epochs on the smallest data source, though this is rarely reached. Empirically 472 this strikes the balance between learning the most from the small amount of data, and avoiding 473 overfitting.

## 474 **A.1.3 Smoothness Loss**

475 In addition to our negative log-likelihood loss term for optimising RRPO, we also have a loss term 476 based on the smoothness of the reward function over trajectories, as seen in Equation (5). This is 477 defined as proportional to the mean-squared first derivative in reward with respect to environment step for all full trajectories.11 478 Concretely:

$$\mathcal{L}_{\text{Smooth}}(\mathcal{D},\theta)=\mu_{\text{smoothed}}\frac{1}{|\mathcal{D}_{\text{Full}}|}\sum_{\tau_{i}^{(n)}\in\mathcal{D}_{\text{Full}}}\frac{1}{n-1}\sum_{k=1}^{n-1}(R_{\theta}(s_{k-1},a_{k-1},s_{k})-R_{\theta}(s_{k},a_{k},s_{k+1}))^{2},\tag{1}$$
DFull = {τi|τi ∈ D, ∀τj̸=i ∈ D. τi ̸⊂ τj}, (9)
$$\begin{array}{l}{{{\mathcal{D}}_{\mathrm{Full}}=\{\tau_{i}|\tau_{i}\in{\mathcal{D}},\forall\tau_{j\neq i}\in{\mathcal{D}}.\;\tau_{i}\not\in\tau_{j}\},}}\\ {{{\tau_{i}^{(n)}}=\{(s_{0},a_{0},s_{1}),...,(s_{n-1},a_{n-1},s_{n})\}.}}\end{array}$$
i = {(s0, a0, s1), ...,(sn−1, an−1, sn)}. (10)
(8)
$$(10)^{\frac{1}{2}}$$

479 We set µsmooth to 0.1 based on early empirical results.

## 480 **A.2 Synthetic Feedback** 481 **A.2.1 Preferences**

482 In Algorithm 1, the GetPreferences function randomly samples trajectory fragments for comparison, 483 with a bias to sampling from new trajectories. We are using a synthetic oracle which uses the ground 484 truth reward function to noisily generate preferences, simulating the imperfect human rationality. 485 More specifically, for each sampled pair of fragments, the sigmoid of their reward difference is used 486 as the parameter for a Bernoulli random variable which is then sampled to generate the preference.

## 487 **A.2.2 Demonstrations**

488 To create demonstrations for our tasks, we simply train an agent on the ground truth reward function 489 (or its negation in the case of negative demonstrations). Several agents are trained, and the best few, 490 nselected, are picked. From these agents, we create a list of their trajectories, ordering from their latest 491 attempts to their first, and interleaving each agent together with the best agent first. For training 492 an agent from feedback, if n demonstrations are being used, the first n demonstrations from this 493 list are provided. Rankings are generated automatically based on the ground truth reward of each demonstration, making <pos and <neg total orders.12 494 The ground truth reward per agent step and 495 number selected, nselected, of all demonstrations trained are given in Figures 4 and 5 for positive and 496 negative demonstrations respectively.

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 Agent Step 1e6 500 0 500 1000 1500 2000 2500 3000 0 100000 200000 300000 400000 500000 Agent Step 8000 6000 4000 2000 0 Ground Trut h Rewar d Ground Trut h Rewar d

(a) Half Cheetah, nselected = 4
(b) Cliff Walking, nselected = 4 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 Agent Step 1e6 1500 1000 500 0 500 1000 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 Agent Step 1e6 0 1000 2000 3000 4000 5000 Ground Truth Reward Ground Truth Reward 
(c) Lunar Lander, nselected = 8
(d) Ant, nselected = 8
Figure 4: Ground truth reward vs agent steps for the positive demonstrations that were trained in every environment. We also state how many were selected as good examples to be used for demonstration learning.

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 Agent Step 1e6 4000 3500 3000 2500 2000 1500 1000 500 5000 10000 15000 20000 25000 30000 Agent Step 10000 8000 6000 4000 2000 Ground Tr uth Rew ard Ground Tr uth Rew ard
(a) Half Cheetah, nselected = 8
(b) Cliff Walking, nselected = 8 0 50000 100000 150000 200000 250000 300000 350000 400000 Agent Step 1600 1500 1400 1300 1200 1100 1000 0 50000 100000 150000 200000 250000 300000 350000 400000 Agent Step 2500 2000 1500 1000 500 Ground Trut h Rewar d Ground Trut h Rewar d

## 497 **B Experiment And Environment Details**

498 Here we give details on versions / modifications made for each environment, as well as environment499 specific hyperparameters summarised in Table 1. We used niters = 8 and 16 random seeds for all 500 runs.

Table 1: Environment specific hyperparameters. 'Trajectory Length' refers to the fixed time horizon for that environment, 'Preference Fragment Length' is the length of the contiguous trajectory subsequences that are used to generate preferences. Both are measured in environment timesteps.

Environment Trajectory Length Preference Fragment Length nrollout-steps

Half Cheetah 1k 32 2M Cliff Walking 250 16 256k Lunar Lander 250 32 8M Ant 1k 32 4M

## 501 **B.1 Half Cheetah**

502 The v4 version is used out-of-the-box.

## 503 **B.2 Cliff Walking**

504 The v0 version is modified to have a fixed horizon of 250 timesteps and a custom reward function. 505 The standard version has a reward of -1 every timestep with the episode terminating when the end is 506 reached. Walking off the cliff gives -100 reward and returns the agent to the start. Our fixed horizon 507 version of this is the same except reaching the end state does not terminate the environment, and 508 instead grants 5 reward per timestep spent there. This was based on what lead to good learning with 509 PPO and access to the reward function directly. 510 As the reward function is sparse, for sampling preferences only, a shaped version of it is used to 511 simulate human intuition on what behaviours are closer to optimal. The penalty for walking off cliffs 512 remains the same, but otherwise the agent receives a weighted reward of -1 and 5 depending on how 513 close in L1 norm it is to the start/end state respectively.

## 514 **B.3 Lunar Lander**

515 The v2 version is modified to have a fixed horizon of 250 timesteps and a custom reward function. 516 The reward function used is mostly the same as in the Gymnasium version, except instead of 517 terminating on game over or the lander not being awake (i.e. landed), a -1 or +1 reward is issued each 518 timestep respectively.

## 519 **B.4 Ant**

520 V4 version with terminate_when_unhealthy=False so that there are more maximum length 521 trajectories.

## 522 **C Supplementary Results**

1 2 3 4 5 6 7 8 Iteration 1500 1000 500 0 500 1000 Mean G round Tr uth Rew ard LEOPARD DeepIRL, best AILP, best Mean G round Tr uth Rew ard LEOPARD DeepIRL, best AILP, best 1 2 3 4 5 6 7 8 Iteration 1000 0 1000 2000 3000 4000 5000
(a) Half Cheetah, ndemos = 8
(b) Cliff Walking, ndemos = 4 1 2 3 4 5 6 7 8 Iteration 600 500 400 300 200 100 0 LEOPARD DeepIRL, best AILP, best LEOPARD DeepIRL, best AILP, best 1 2 3 4 5 6 7 8 Iteration 1000 0 1000 2000 3000 Mean Ground Trut h Reward Mean Ground Trut h Reward
Figure 6: Comparison of LEOPARD with the baselines of AILP and DeepIRL when only positive demonstrations are available. The lines denote the mean of the ground truth reward function, with shaded standard errors across 16 random seeds, against algorithm iterations—alternations between optimising the reward model and the agent. Solid lines are smoothed means for clarity, dashed lines give raw values. A breakdown of the performance of the baseline methods for different reward model training epochs per iteration is given in Figures 9 and 10.

| be fixed for DeepIRL. Best in column for section. Method RM epochs   | Final Ground Truth Reward ± std error   |               |              |              |             |
|----------------------------------------------------------------------|-----------------------------------------|---------------|--------------|--------------|-------------|
| per iter                                                             | Half Cheetah                            | Cliff Walking | Lunar Lander | Ant          |             |
| LEOPARD (ours)                                                       | Dynamic                                 | 5650 ± 386    | 670 ± 116    | -140 ± 49.8  | 2630 ± 322  |
| AILP                                                                 | Dynamic                                 | 3.49 ± 105    | -249 ± 6.09  | -684 ± 31.8  | -1130 ± 142 |
| AILP                                                                 | 1                                       | 14.1 ± 234    | -266 ± 116   | -2010 ± 506  | -237 ± 110  |
| AILP                                                                 | 2                                       | 25.1 ± 226    | -172 ± 74.2  | -2270 ± 507  | -300 ± 117  |
| AILP                                                                 | 4                                       | -129 ± 35.9   | -181 ± 85.5  | -1930 ± 501  | 150 ± 131   |
| AILP                                                                 | 8                                       | -87.0 ± 38.4  | -180 ± 70.0  | -813 ± 340   | 148 ± 55.0  |
| DeepIRL then RLHF                                                    | 1                                       | -389 ± 223    | -46.8 ± 125  | -2340 ± 548  | -766 ± 216  |
| DeepIRL then RLHF                                                    | 2                                       | 189 ± 312     | 1.34 ± 163   | -2200 ± 537  | -803 ± 259  |
| DeepIRL then RLHF                                                    | 4                                       | 224 ± 205     | -61.7 ± 115  | -2000 ± 467  | -792 ± 221  |
| DeepIRL then RLHF                                                    | 8                                       | 1540 ± 374    | -91.7 ± 103  | -1720 ± 548  | -927 ± 192  |
| LEOPARD (ours)                                                       | Dynamic                                 | 5020 ± 555    | 580 ± 199    | -34.4 ± 25.7 | 3000 ± 390  |
| AILP                                                                 | Dynamic                                 | -45.0 ± 236   | 554 ± 146    | -215 ± 16.1  | -489 ± 178  |
| AILP                                                                 | 1                                       | -88.3 ± 9.15  | 381 ± 131    | -99.5 ± 5.45 | 555 ± 37.1  |
| AILP                                                                 | 2                                       | -61.5 ± 47.1  | 330 ± 156    | -131 ± 9.33  | 450 ± 54.8  |
| AILP                                                                 | 4                                       | -118 ± 6.08   | 205 ± 133    | -180 ± 12.3  | 300 ± 79.1  |
| AILP                                                                 | 8                                       | -96.2 ± 6.36  | -72.2 ± 93.2 | -214 ± 8.62  | 268 ± 59.4  |
| DeepIRL                                                              | 1                                       | 1470 ± 318    | 828 ± 92.2   | -575 ± 194   | -295 ± 230  |
| DeepIRL                                                              | 2                                       | 1610 ± 264    | 769 ± 111    | -164 ± 98.6  | 1320 ± 426  |
| DeepIRL                                                              | 4                                       | 1290 ± 216    | 849 ± 102    | -159 ± 18.0  | 1780 ± 399  |
| DeepIRL                                                              | 8                                       | 1790 ± 162    | 528 ± 105    | -219 ± 21.3  | 1340 ± 319  |

| iteration see Figure 3. Best in column. Feedback types   | Final Ground Truth Reward ± std error   |              |              |            |
|----------------------------------------------------------|-----------------------------------------|--------------|--------------|------------|
| Half Cheetah                                             | Cliff Walking                           | Lunar Lander | Ant          |            |
| Preferences                                              | 4960 ± 574                              | -252 ± 2.22  | -163 ± 19.7  | 1510 ± 491 |
| Positive demonstrations                                  | 5020 ± 555                              | 580 ± 199    | -34.4 ± 25.7 | 3000 ± 390 |
| Preferences and positive demos                           | 5650 ± 386                              | 670 ± 116    | -140 ± 49.8  | 2630 ± 322 |
| Positive and negative demos                              | 2870 ± 609                              | 883 ± 79.0   | -169 ± 107   | 754 ± 339  |
| Prefs, pos and neg demos                                 | 3640 ± 603                              | 514 ± 133    | -120 ± 11.3  | 1580 ± 296 |

Mea n G
rou nd Tr uth Rew ard Epochs=1 Epochs=2 Epochs=4 Epochs=8 1 2 3 4 5 6 7 8 Iteration 2000 1500 1000 500 0 1 2 3 4 5 6 7 8 Iteration 1000 500 0 500 1000 1500 Mea n G
rou nd Tr uth Rew ard Epochs=1 Epochs=2 Epochs=4 Epochs=8
(a) Half Cheetah, ndemos = 4, nprefs =
256
(b) Cliff Walking, ndemos = 2, nprefs =
64 1 2 3 4 5 6 7 8 Iteration 3000 2500 2000 1500 1000 Me an Grou nd Trut h Re war d Epochs=1 Epochs=2 Epochs=4 Epochs=8 Me an Gro und Trut h Re ward Epochs=1 Epochs=2 Epochs=4 Epochs=8 1 2 3 4 5 6 7 8 Iteration 1600 1400 1200 1000 800 600 Mea n G
rou nd Tru th Rew ard Dynamic Stopping Epochs=1 Epochs=2 Epochs=4 Epochs=8 1 2 3 4 5 6 7 8 Iteration 1400 1200 1000 800 600 400 200 1 2 3 4 5 6 7 8 Iteration 300 200 100 0 100 200 Mea n G
rou nd Tru th Rew ard Dynamic Stopping Epochs=1 Epochs=2 Epochs=4 Epochs=8
(a) Half Cheetah, ndemos = 4, nprefs =
256
(b) Cliff Walking, ndemos = 2, nprefs =
64 1 2 3 4 5 6 7 8 Iteration 2500 2000 1500 1000 500 1 2 3 4 5 6 7 8 Iteration 1250 1000 750 500 250 0 250 500 Me an Gro und Trut h Re war d Me an Gro un d Trut h Re ward Dynamic Stopping Epochs=1 Epochs=2 Epochs=4 Epochs=8 Dynamic Stopping Epochs=1 Epochs=2 Epochs=4 Epochs=8 1 2 3 4 5 6 7 8 Iteration 1000 500 0 500 1000 1500 2000 1 2 3 4 5 6 7 8 Iteration 1500 1000 500 0 500 1000 Mean G
rou nd Tr uth Rew ard Epochs=1 Epochs=2 Epochs=4 Epochs=8 M
ea n G
rou nd Tr uth Rew ard Epochs=1 Epochs=2 Epochs=4 Epochs=8
(a) Half Cheetah, ndemos = 8
(b) Cliff Walking, ndemos = 4 Me an Grou nd Trut h Re war d Epochs=1 Epochs=2 Epochs=4 Epochs=8 Me an Grou nd Trut h Re war d Epochs=1 Epochs=2 Epochs=4 Epochs=8 1 2 3 4 5 6 7 8 Iteration 800 700 600 500 400 300 200 100 1 2 3 4 5 6 7 8 Iteration 500 0 500 1000 1500 2000 1 2 3 4 5 6 7 8 Iteration 1250 1000 750 500 250 0 250 500 750 1 2 3 4 5 6 7 8 Iteration 400 300 200 100 0 100 200 Mean G
rou nd Tru th Rew ard Dynamic Stopping Epochs=1 Epochs=2 Epochs=4 Epochs=8 Me an Gr oun d Tr uth Rew ard Dynamic Stopping Epochs=1 Epochs=2 Epochs=4 Epochs=8
(a) Half Cheetah, ndemos = 8
(b) Cliff Walking, ndemos = 4 1 2 3 4 5 6 7 8 Iteration 700 600 500 400 300 200 100 1 2 3 4 5 6 7 8 Iteration 600 400 200 0 200 400 600 Me an Gro und Trut h Re war d Me an Gro und Trut h Re war d Dynamic Stopping Epochs=1 Epochs=2 Epochs=4 Epochs=8 Dynamic Stopping Epochs=1 Epochs=2 Epochs=4 Epochs=8

(d) Ant, ndemos = 8