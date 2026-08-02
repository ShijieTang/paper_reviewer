# Quantifying First-Order Markov Breakdowns In Noisy Reinforcement Learning: A Causal Discovery Approach

| Anonymous Author(s) Affiliation Address email   |
|-------------------------------------------------|

## Abstract

1 Reinforcement learning (RL) methods often assume that each new observation 2 fully captures the environment's state, ensuring Markovian (one-step) transitions. 3 Real-world deployments, however, frequently violate this assumption due to partial 4 observability or noise in sensors and actuators. This paper introduces a systematic 5 methodology for diagnosing such violations, combining a partial correlation based 6 causal discovery procedure (PCMCI) with a newly proposed Markov Violation 7 score (MVS). The MVS quantifies multi-step dependencies that emerge when noise 8 or incomplete state information disrupts the Markov property. 9 Classic control tasks (CartPole, Pendulum, Acrobot) are used to assess how targeted 10 noise and dimension omissions affect both RL performance and the measured 11 Markov consistency. Contrary to expectations, heavy observation noise often fails 12 to induce strong multi-lag dependencies in certain tasks (e.g., Acrobot). Dimension13 dropping experiments further reveal that omitting certain state variables (e.g., 14 angular velocities in CartPole and Pendulum) substantially degrades returns and 15 elevates MVS, while other dimensions can be removed with negligible effect. 16 These findings highlight the importance of identifying and safeguarding the most 17 causally critical dimensions to maintain effective one-step learning. By bridg18 ing partial correlation tests and RL performance metrics, the proposed approach 19 uniquely pinpoints when and where the Markov property breaks. This frame20 work offers a principled tool for designing robust policies, guiding representation 21 learning, and handling partial observability in real-world RL tasks. All code and 22 experimental logs are publicly available for reproducibility (URL omitted for 23 double-blind review).

## 24 **1 Introduction**

25 Reinforcement learning (RL) typically assumes that observations fully capture the environment's 26 state, ensuring one-step (Markovian) transitions [Sutton and Barto, 1998]. In practice, however, 27 partial observability or sensor noise frequently undermines this assumption [Wisniewski et al., 2024], 28 leading to multi-step dependencies and degraded convergence. While many RL algorithms tolerate 29 mild noise, moderate or poorly structured perturbations often disrupt Markovian structure and erode 30 policy performance. 31 A key challenge lies in *diagnosing* when (and why) the Markov property ceases to hold. Standard 32 metrics (e.g., final returns) do not reveal whether an environment is effectively "non-Markovian" from 33 the agent's perspective. To address this gap, the present work introduces a *Markov Violation Score* 34 (MVS) derived from partial correlation tests via PCMCI [Runge, 2022]. This score detects significant Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Do not distribute.

35 lag-2+ dependencies, indicating multi-step effects that deviate from the single-step (first-order) 36 assumption. 37 A systematic investigation examines how specific perturbations impact both policy performance and 38 MVS in three classic control tasks—CartPole-v1, *Pendulum-v1*, and *Acrobot-v1*:
39 - **Noise Injection.** Gaussian noise and autoregressive noises are applied to observation 40 dimensions at varying levels, revealing which features are critical for stable control. 41 - **Dimension Dropping.** Entire observation dimensions are removed, forcing learning under 42 incomplete information. Some dropped dimensions cause mild performance degradation, 43 whereas others induce severe instability and high MVS.

44 - **Markov Violation Analysis.** In each scenario, PCMCI is used to detect higher-lag correla45 tions (lag-2+). Surges in multi-step links typically coincide with sharp performance drops, 46 signaling that first-order Markov assumptions no longer hold.

47 The results highlight that not all state dimensions contribute equally to preserving Markovian structure.

48 Corrupting or omitting a *critical* variable can produce large multi-step dependencies and abrupt policy 49 collapse, whereas a less influential dimension may have negligible effect. In addition, tasks exhibit 50 distinct thresholds of robustness: some degrade abruptly under moderate noise, whereas others (e.g., 51 Acrobot) handle multi-lag correlations without catastrophic failure.

52 **Paper Organization.** Sections 2–3.2 discuss related work on partial observability and causal 53 discovery, then introduce the Markov property, PCMCI, and the proposed MVS. Section 5 describes 54 the experimental design (baseline runs, noise injection, dimension dropping) and presents findings on 55 policy performance and Markov consistency. Section 6 addresses limitations and explores directions 56 for future research, and Section 7 concludes the paper.

## 57 **2 Related Works**

58 Real-world reinforcement learning (RL) often encounters partial observability and noisy signals 59 that deviate from the ideal Markov property [wie, 2012]. Much work in *robust RL* aims to handle 60 disturbances in transitions or observations [Panaganti et al., 2022, Liu et al., 2022b], using adversarial 61 training [Pinto et al., 2017] or domain randomization [Wang et al., 2019, Li et al., 2021, Wang et al., 62 2020] for noisy perception. Other studies introduce noise directly into observations or actions [Hollen63 stein et al., 2024, Hollenstein et al., Igl et al., 2019], but most evaluations rely on final-return metrics 64 and lack a principled way to detect multi-step dependencies that arise when Markov assumptions fail. 65 Another branch of *partially observable RL* explores how unobserved variables break Markovian 66 structure [Lauri et al., 2023]. Under POMDPs and related frameworks, latent variables [Liu et al., 67 2022a] often model environment dynamics [Zhu et al., 2020, Yu et al., Shi et al., 2020]. Though such 68 methods can mitigate certain noise types (e.g., Gaussian or autoregressive (AR) noise ), they seldom 69 track *which* dimensions or episodes are most critical to preserving (or violating) first-order dynamics. 70 Moreover, a single metric to capture multi-lag correlations remains elusive. 71 Meanwhile, other research [Ota et al., 2020] has *increased* input dimensionality to improve sample 72 efficiency and final performance, reinforcing the need to preserve crucial state information in expanded 73 feature spaces. However, these approaches do not pinpoint *which* dimensions are indispensable for 74 maintaining a Markovian process. 75 To fill these gaps, the present work applies PCMCI's causal discovery tests [Runge, 2022] to detect 76 higher-lag partial correlations and quantify Markov violations. Building on robust RL's concern with 77 sensor/actuator noise—and partial-observability research on hidden factors—this paper proposes a 78 *Markov Violation Score* (MVS) that aggregates multi-step links beyond first-order transitions. Unlike 79 prior causal-discovery [Zeng et al., 2023] or partial-observation works, the MVS offers a single 80 interpretable value indicating how strongly the Markov property breaks under dimension-dropping or 81 other perturbations. This framework thus moves beyond final-return comparisons to identify *which* 82 omitted dimensions or noise distributions most severely degrade first-order RL learning.

## 83 **3 Preliminaries** 84 **3.1 Markov Property And Markov Decision Processes**

A discrete-time stochastic process {Xt}∞
t=0 85 satisfies the *Markov property* if, at every time step t, 86 the future state Xt+1 is conditionally independent of all prior states {X0, X1*, . . . , X*t−1} given the 87 current state Xt. Formally, PXt+1 | Xt, Xt−1*, . . . , X*0= PXt+1 | Xt.

88 Intuitively, this means the present state fully encapsulates all relevant information from the past. In a 89 reinforcement learning (RL) context, we typically apply the Markov property to a state variable St. If 90 the environment truly satisfies this property, then PSt+1 = s
′, Rt+1 = r | St = s, At = a, . . . , S0, A0= PSt+1 = s
′, Rt+1 = r | St = *s, A*t = a, 91 which ensures that only the current state St and action At determine the distribution over next states 92 St+1 and rewards Rt+1. However, if noise or partial observability reduce the completeness of St, 93 higher-order (multi-lag) dependencies may arise. This violates the first-order Markov assumption 94 and can complicate RL methods that rely on single-step dynamics.

## 95 **3.1.1 Conditional Independence And The Pcmci Framework**

96 Two variables X and Y are said to be *conditionally independent* given a set of variables Z if P(X | *Y, Z*) = P(X | Z).

97 In an ideal Markov process, once the current state St is known, the future state St+1 becomes 98 independent of all past states {S0*, . . . , S*t−1}. However, noise or partial observability can introduce 99 multi-lag dependencies, causing St+1 to depend on earlier states St−2, St−3*, . . .*. To detect such 100 higher-order effects, one can examine *partial correlations*, which measure linear associations between 101 X and Y after conditioning on Z. Significant partial correlations at lag ≥ 2 indicate a breakdown of 102 the first-order Markov property.

103 Constraint-based causal discovery methods, such as the **PC algorithm** [Spirtes et al., 2001], itera104 tively test for conditional independence and remove edges in a candidate causal graph. *Momentary* 105 *Conditional Independence (MCI)* extends this testing to time-series data by conditioning on momen106 tary and past information at each time step. Building on MCI, **PCMCI** [Runge, 2022] combines 107 partial-correlation-based tests with the PC procedure to handle high-dimensional time series. In an 108 RL setting, detecting edges at lag 2 or beyond via PCMCI offers direct evidence that single-step 109 conditioning on St alone is insufficient, thus revealing violations of the Markov property. 110 **Relevance to RL and Markov Violations.** In RL, St+1 often depends on (St, At) only. Noise or 111 partial observability can generate dependence on St−2, St−3*, . . .* beyond St−1. By applying PCMCI 112 to agent trajectories, one can quantify the severity of these multi-lag links. Such diagnosis helps 113 explain policy breakdowns and suggests solutions like state augmentation or sensor fusion [Laskin 114 et al., 2020].

## 115 **3.2 Pcmci And The Markov Property**

116 In a strictly Markovian environment, no significant causal links appear at lags beyond one. When 117 PCMCI detects higher-lag correlations, it indicates missing information in St. After training, rollouts 118 were collected to apply PCMCI across St−1, St−2*, . . .* to find significant partial correlations at k ≥ 2. 119 The *Markov Violation Score* (Section 4) summarizes these multi-lag dependencies. Higher scores 120 typically signal greater departure from first-order dynamics, aligning with observed performance 121 drops.

## 122 **4 Markov Violation Score**

123 As noted in Section 3.2, PCMCI can reveal higher-lag dependencies that indicate violations of the 124 first-order Markov property. This section introduces the *Markov Violation Score* (MVS), which 125 quantifies how severely one-step assumptions are broken.

| Child                           | Parent   | Lag    | p-val   | Part. Corr   |
|---------------------------------|----------|--------|---------|--------------|
| Variable 0 has 6 link(s): 0 2 0 | 0.00000  | -0.833 |         |              |
| 0                               | 3        | 0      | 0.00000 | -0.621       |
| 0                               | 1        | 0      | 0.00000 | 0.566        |
| 0                               | 0        | -1     | 0.00000 | 0.423        |
| 0                               | 1        | -1     | 0.00000 | 0.109        |
| 0                               | 2        | -1     | 0.00000 | 0.079        |

Variable 0 has 6 link(s):

0 2 0 0.00000 -0.833 0 3 0 0.00000 -0.621 0 1 0 0.00000 0.566 0 0 -1 0.00000 0.423 0 1 -1 0.00000 0.109 0 2 -1 0.00000 0.079

Table 1: An example of PCMCI results for CartPole showing no significant edges (p-value threshold

was 0.05) at lag ≤ −2, consistent with first-order Markov structure in the unperturbed setting.

126 **Defining the MVS.** Consider N total variables (e.g., state components), a maximum lag τmax, and a significance threshold αlevel. For each variable pair (*i, j*) and lag |k| ≥ 2, let val 127 (*i,j,k*) be the partial correlation at lag k, and let p(i,j,k)be its p-value. The indicator Ip(*i,j,k*) ≤ αlevel 128 is 1 if the p-value 129 is below αlevel and 0 otherwise. The MVS then is

$$\mathrm{MVS}\;=\;\frac{\sum_{i=1}^{N}\sum_{j=1}^{N}\sum_{k=2}^{r_{\mathrm{max}}}(k-1)\;\|{\bf val}_{(i,j,k)}\|\;\big[-\ln\big({\bf p}_{(i,j,k)}\big)\big]\;\mathbb{I}\big({\bf p}_{(i,j,k)}\leq\alpha_{\mathrm{level}}\big)}{N^{2}\;\sum_{k=2}^{r_{\mathrm{max}}}(k-1)}$$
,
130 where (k − 1) weights longer lags more heavily. If no lag|k| ≥ 2 links are detected, then MVS = 0.

| Child Var                     | Parent Var   | Lag     | p-value   | Partial Corr   |
|-------------------------------|--------------|---------|-----------|----------------|
| Variable 0 has 4 link(s): 0 0 | -1           | 0.00000 | 0.663     |                |
| 0                             | 3            | -3      | 0.00000   | -0.281         |
| 0                             | 2            | -3      | 0.00000   | -0.078         |
| 0                             | 1            | 0       | 0.03875   | -0.003         |

Variable 0 has 4 link(s): 0 0 -1 0.00000 0.663 0 3 -3 0.00000 -0.281 0 2 -3 0.00000 -0.078 0 1 0 0.03875 -0.003

Table 2: Example PCMCI results (α threshold was 0.05) for a noisy CartPole run with MVS > 0.

131 A nonzero MVS indicates multi-step dependencies that degrade performance in one-step RL algo132 rithms. Larger scores correlate with more severe Markov violations, whereas MVS = 0 means no 133 multi-lag links survive thresholding and the system remains effectively first-order.

## 134 **5 Experiments And Results**

135 This section explores how noise injection and dimension manipulation impact both the Markovian 136 structure of classic RL environments and final policy performance. The following subsections detail 137 the experimental setup, baseline (no-modification) runs, the effects of i.i.d. and autoregressive (AR) 138 noise, and the consequences of dropping specific dimensions. Each analysis leverages both episode 139 returns and the proposed Markov Violation Score (MVS) to reveal whether multi-lag dependencies 140 emerge under different perturbations.

## 141 **5.1 Experimental Setup**

142 Jobs ran under Python 3.11 on six AWS EC2 c7i.4xlarge instances (16 vCPU, 32 GiB RAM,
143 AMI ami-00c257e12d6828491); each instance completed an identical slice of the sweep in 12 h, 144 yielding an effective 72 CPU-hours.

Every task used stable-baselines3 PPO defaults: two 64-unit TANH layers, Adam (3×10−4 145 ), dis146 count γ = 0.99, GAE λ = 0.95, entropy 0, value-loss 0.5, clip 0.2, mini-batch 64, four optimisation 147 epochs, and no hyper-parameter tuning, ensuring identical settings across baseline, noise, and drop 148 variants.

149 Training horizons followed common benchmarks—50 k steps for *CartPole* and *Acrobot*, 450 k for 150 *Pendulum*. After training, 1–2 k extra transitions per run were gathered to compute PCMCI partial 151 correlations and the Markov-Violation Score (MVS).

## 152 **5.2 Random-Seed Protocol And Significance Estimates**

153 Each EC2 worker chose one baseline seed uniformly from {0*, . . . ,* 1000}; the selected value initialised 154 the simulator, network weights, and (where relevant) the noise generator, and was re-used across the 155 baseline, Gaussian, AR, and drop variants for that environment. The six workers therefore produced 156 six statistically independent runs per condition. Learning-curve plots report the mean over n= 6 runs, with whiskers showing the 95 % confidence interval CI95 = 1.96 σ/√
157 n, where σ is the across-seed 158 standard deviation. 159 For Markov-violation analysis, every trained policy generated three additional roll-outs under fresh 160 rollout seeds; PCMCI statistics from those roll-outs were fused with Fisher's method and then 161 averaged over the six baseline seeds, yielding a single MVS ± CI95 per setting. All CSV logs 162 produced on the cloud were downloaded and aggregated offline; an appendix script reproduces the 163 merge. Unless noted otherwise, figures and tables follow this protocol.

## 164 **5.3 Baseline Performance**

165 A no-modification "baseline" was trained for each environment to verify that the default tasks exhibit 166 effectively Markovian structure. In all three domains, the baseline (indicated by black curves in 167 subsequent figures) converged quickly and maintained top returns, with PCMCI detecting negligible 168 lag-≥ 2 correlations (i.e., MVS ≈ 0). This outcome confirms that the unaltered state representations 169 of CartPole-v1, *Pendulum-v1*, and *Acrobot-v1* largely satisfy first-order Markov assumptions.

## 170 **5.4 Gaussian Noise Injection**

171 To evaluate how i.i.d. Gaussian perturbations affect both policy performance and Markov consistency, each observation dimension o
(i)
tis augmented by independent draws η
(i)
t ∼ N (*µ, σ*2 172 ). The agent is then trained on oe
(i)
t = o
(i)
t + η
(i) t 173 for the targeted dimension i. Figure 1 shows mean learning curves for three noise levels across CartPole, Pendulum, and Acrobot. Small variance (σ 174 2 = 0.02) leaves returns near baseline, whereas larger noise (σ 175 2 = 1.0−2.0) sharply degrades performance 176 when critical angles or velocities are corrupted; Acrobot remains comparatively robust. Aggregated 177 learning curves *with 95 % confidence envelopes*, computed from the full set of seeded runs, are 178 provided in Appendix Fig. 7.

## 179 **5.4.1 State-Space Noise Effects And Mvs**

Although elevated noise (σ 180 2 ≥ 1.0) clearly degrades rewards (Figure 3), the Markov property remains 181 fairly intact in i.i.d. Gaussian settings: PCMCI rarely uncovers strong lag-≥ 2 correlations unless the 182 variance is extremely high. As illustrated in Figure 2a, changes in MVS remain modest for i.i.d. noise 183 in CartPole, revealing that episodic returns can drop substantially even while MVS hovers near zero. 184 These observations suggest that purely independent noise often fails to violate first-order structure, 185 motivating the introduction of correlated (AR) disturbances to elicit stronger multi-lag dependencies.

## 186 **5.5 Autoregressive Noise Injection**

187 To induce more pronounced deviations from the first-order Markov assumption, *autoregressive* (AR)
188 noise is introduced. Let {zt} be a one-dimensional AR(p) process,

$$z_{t+1}\;=\;\sum_{\ell=0}^{p-1}\rho_{\ell}\,z_{t-\ell}+\epsilon_{t},\quad\epsilon_{t}\sim{\mathcal N}(0,\sigma^{2}).$$

189 This hidden variable zt+1 is added to designated *observation* dimensions each step, coupling consec190 utive states and frequently generating lag-≥ 2 dependencies. Experiments varying p and ρ0 confirm 191 that higher AR orders and larger coefficients correlate with elevated MVS and significant perfor192 mance degradation (Figure 4). Thus, while i.i.d. Gaussian noise alone might not break the single-step 193 property, AR noise reliably induces multi-lag correlations and accentuates the link between MVS and 194 poorer returns.

(a) CartPole, µ = 0, σ 2 = 0.02 (b) CartPole, µ = 0, σ 2 = 1.0
(c) Pendulum, µ = 0, σ 2 = 1.0 (d) Pendulum, µ = 0, σ 2 = 2.0
(e) Acrobot, µ = 0, σ 2 = 0.02 (f) Acrobot, µ = 0, σ 2 = 2.0
(a) CartPole (b) Acrobot
(a) CartPole (b) Acrobot
(a) Pendulum returns (b) Pendulum learning curves
(c) CartPole returns (d) CartPole learning curves
(e) Acrobot returns (f) Acrobot learning curves

## 195 **5.6 Dimension Dropping**

196 Motivated by the i.i.d. Gaussian noise results (Figure 1), which hinted that certain dimensions were 197 more critical, the next step was to drop each dimension entirely and observe any performance shifts. 198 This is akin to senor malfunction or total loss of signal in real world systems. Surprisingly, removing 199 some dimensions (e.g., cart position in CartPole or various joint components in Acrobot) produced 200 negligible changes, revealing a degree of redundancy in those tasks. In contrast, omitting more 201 pivotal features—such as pole angle or angular velocity in CartPole, or the angular velocity in 202 Pendulum—triggered substantial performance drops and elevated Markov Violation Scores. These 203 findings confirm that although certain state variables can be safely excluded, others are indispensable 204 for first-order RL methods to operate effectively.

## 205 **5.7 Overall Analysis: Correlating Mvs And Policy Performance**

206 Collectively, these experiments show how MVS correlates with (and often predicts) policy break207 down. In *CartPole* and *Pendulum*, large perturbations to crucial dimensions (e.g., pole angles or 208 angular velocities) often raise MVS and reduce returns drastically; by contrast, *Acrobot* exhibits 209 greater redundancy, tolerating moderate distortions or dropped variables without catastrophic failure. 210 Monitoring MVS alongside standard reward curves thus flags emergent multi-lag dependencies in 211 non-Markov settings (e.g., under AR noise or dimension-critical omissions). Such insights can guide 212 robust controller design and inform representation learning, ensuring that the most causally pivotal 213 features remain intact for stable, first-order RL.

## 214 **6 Limitations And Future Work**

215 These experiments illuminate how noise and dimension manipulations can undermine Markov 216 properties in standard control tasks, yet several constraints remain. Only three benchmarks (*CartPole*, 217 Pendulum, *Acrobot*) were examined, limiting applicability to more complex domains. All studies were 218 conducted with PPO—chosen for its empirical stability—which leaves open whether value-based 219 agents (e.g., DQN [Mnih et al., 2013]), entropy-regularised actor–critics such as SAC [Haarnoja et al.,
220 2018], or model-based planners exhibit comparable sensitivities. Noise and dimension perturbations 221 were deliberately simple, and the Markov-Violation Score (MVS) currently relies on linear partial 222 correlations; richer, nonlinear tests could reveal subtler dependencies. Real-world sensor faults 223 and actuator delays also remain unexplored. Future work will target higher-dimensional domains 224 (e.g., multi-joint robotics) that may expose new forms of Markov violation, extend the study to 225 alternative algorithms to test algorithm-level generality, integrate adaptive mitigation (recurrent, 226 Bayesian, or active dimension selection) to suppress MVS spikes, and validate findings in hardware 227 where noise processes are more complex. Planned ablations include injecting irrelevant white-noise 228 channels, evaluating MVS across random/early/final policies, and hard-muting a sensor mid-training 229 to investigate whether MVS can serve as a real-time anomaly detector. *MVS likewise offers a* 230 *lightweight safety diagnostic for real robots, though treating a low score as a blanket guarantee* 231 *could prove risky.*

## 232 **7 Conclusion**

233 This work examined how partial observability and noise injection affect Markovian assumptions 234 in reinforcement learning, with a particular focus on detecting multi-lag dependencies through the 235 Markov Violation Score (MVS). Classic control tasks demonstrated that certain dimensions—such 236 as pole angles in CartPole or angular velocity in Pendulum—are pivotal for preserving first-order 237 dynamics, whereas other variables can be removed with minimal impact. Independent Gaussian 238 noise often degraded performance yet did not necessarily induce strong lag-≥ 2 correlations, while 239 autoregressive processes consistently triggered higher MVS values and more severe policy break240 downs. Dimension-dropping experiments further revealed that some tasks, like Acrobot, retain 241 robustness under omitted variables, whereas others rely heavily on specific state components. These 242 findings highlight the utility of partial-correlation tests for diagnosing Markov violations, indicating 243 the potential for adaptive or model-based methods to mitigate these effects. Extending MVS-based 244 diagnostics to higher-dimensional domains and real-world sensor data offers a promising avenue for 245 developing more robust and generalizable RL algorithms.

## 246 **References**

247 Partially Observable Markov Decision Processes. In Marco Wiering and Martijn Van Otterlo, editors, 248 *Reinforcement Learning: State-of-the-Art*, volume 12 of *Adaptation, Learning, and Optimization*, 249 pages 387–414. Springer Berlin Heidelberg, Berlin, Heidelberg, 2012. ISBN 978-3-642-27644-6 250 978-3-642-27645-3. doi: 10.1007/978-3-642-27645-3. URL https://link.springer.com/ 251 chapter/10.1007/978-3-642-27645-3_12. 252 Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft Actor-Critic: Off-Policy 253 Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, August 2018. URL 254 http://arxiv.org/abs/1801.01290. arXiv:1801.01290 [cs].

255 Jakob Hollenstein, Sayantan Auddy, Matteo Saveriano, Erwan Renaudo, and Justus Piater. Action 256 Noise in Off-Policy Deep Reinforcement Learning: Impact on Exploration and Performance. 257 Jakob Hollenstein, Georg Martius, and Justus Piater. Colored Noise in PPO: Improved Exploration 258 and Performance through Correlated Action Sampling. *Proceedings of the AAAI Conference on* 259 *Artificial Intelligence*, 38(11):12466–12472, March 2024. ISSN 2374-3468, 2159-5399. doi: 260 10.1609/aaai.v38i11.29139. URL http://arxiv.org/abs/2312.11091. arXiv:2312.11091 261 [cs]. 262 Maximilian Igl, Kamil Ciosek, Yingzhen Li, Sebastian Tschiatschek, Cheng Zhang, Sam Devlin, and 263 Katja Hofmann. Generalization in Reinforcement Learning with Selective Noise Injection and 264 Information Bottleneck. In *Advances in Neural Information Processing Systems*, volume 32. Cur265 ran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper_files/paper/ 266 2019/hash/e2ccf95a7f2e1878fcafc8376649b6e8-Abstract.html.

267 Misha Laskin, Kimin Lee, Adam Stooke, Lerrel Pinto, Pieter Abbeel, and Aravind Srinivas. Rein268 forcement Learning with Augmented Data. In *Advances in Neural Information Processing Systems*,
269 volume 33, pages 19884–19895. Curran Associates, Inc., 2020. URL https://proceedings.

270 neurips.cc/paper/2020/hash/e615c82aba461681ade82da2da38004a-Abstract.html. 271 Mikko Lauri, David Hsu, and Joni Pajarinen. Partially Observable Markov Decision Processes 272 in Robotics: A Survey. *IEEE Transactions on Robotics*, 39(1):21–40, February 2023. ISSN 273 1552-3098, 1941-0468. doi: 10.1109/TRO.2022.3200138. URL https://ieeexplore.ieee. 274 org/document/9899480/. 275 Kevin Li, Abhishek Gupta, Ashwin Reddy, Vitchyr Pong, Aurick Zhou, Justin Yu, and Sergey 276 Levine. MURAL: Meta-Learning Uncertainty-Aware Rewards for Outcome-Driven Reinforcement 277 Learning, July 2021. URL http://arxiv.org/abs/2107.07184. arXiv:2107.07184 [cs]. 278 Qinghua Liu, Alan Chung, Csaba Szepesvari, and Chi Jin. When Is Partially Observable Rein279 forcement Learning Not Scary? In Proceedings of Thirty Fifth Conference on Learning The280 ory, pages 5175–5220. PMLR, June 2022a. URL https://proceedings.mlr.press/v178/ 281 liu22f.html. ISSN: 2640-3498.

282 Zijian Liu, Qinxun Bai, Jose Blanchet, Perry Dong, Wei Xu, Zhengqing Zhou, and Zhengyuan Zhou. 283 Distributionally Robust $Q$-Learning. In *Proceedings of the 39th International Conference on* 284 *Machine Learning*, pages 13623–13643. PMLR, June 2022b. URL https://proceedings.mlr. 285 press/v162/liu22a.html. ISSN: 2640-3498. 286 Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan 287 Wierstra, and Martin Riedmiller. Playing Atari with Deep Reinforcement Learning, December 288 2013. URL http://arxiv.org/abs/1312.5602. arXiv:1312.5602 [cs]. 289 Kei Ota, Tomoaki Oiki, Devesh Jha, Toshisada Mariyama, and Daniel Nikovski. Can Increasing 290 Input Dimensionality Improve Deep Reinforcement Learning? In *Proceedings of the 37th* 291 *International Conference on Machine Learning*, pages 7424–7433. PMLR, November 2020. URL 292 https://proceedings.mlr.press/v119/ota20a.html. ISSN: 2640-3498.

293 Kishan Panaganti, Zaiyan Xu, Dileep Kalathil, and Mohammad Ghavamzadeh. Robust Reinforce294 ment Learning using Offline Data, October 2022. URL http://arxiv.org/abs/2208.05129.

295 arXiv:2208.05129 [cs]. 296 Lerrel Pinto, James Davidson, Rahul Sukthankar, and Abhinav Gupta. Robust Adversarial Rein297 forcement Learning. In Proceedings of the 34th International Conference on Machine Learn298 ing, pages 2817–2826. PMLR, July 2017. URL https://proceedings.mlr.press/v70/ 299 pinto17a.html. ISSN: 2640-3498. 300 Jakob Runge. Discovering contemporaneous and lagged causal relations in autocorrelated 301 nonlinear time series datasets, January 2022. URL http://arxiv.org/abs/2003.03685. 302 arXiv:2003.03685 [stat]. 303 Chengchun Shi, Runzhe Wan, Rui Song, Wenbin Lu, and Ling Leng. Does the Markov Decision Pro304 cess Fit the Data: Testing for the Markov Property in Sequential Decision Making. In *Proceedings* 305 *of the 37th International Conference on Machine Learning*, pages 8807–8817. PMLR, November 306 2020. URL https://proceedings.mlr.press/v119/shi20c.html. ISSN: 2640-3498.

307 Peter Spirtes, Clark Glymour, and Richard Scheines. Causation, prediction, and search. In *Causation,* 308 *prediction, and search*. MIT press, 2001. 309 Richard S Sutton and Andrew G Barto. *Reinforcement Learning: An Introduction*. The MIT Press, 310 Cambridge, MA, 1998. 311 Jingkang Wang, Yang Liu, and Bo Li. Reinforcement Learning with Perturbed Rewards. *Proceedings* 312 *of the AAAI Conference on Artificial Intelligence*, 34(04):6202–6209, April 2020. ISSN 2374-3468, 313 2159-5399. doi: 10.1609/aaai.v34i04.6086. URL https://ojs.aaai.org/index.php/AAAI/ 314 article/view/6086.

315 Yuhui Wang, Hao He, and Xiaoyang Tan. Robust Reinforcement Learning in POMDPs with In316 complete and Noisy Observations, February 2019. URL http://arxiv.org/abs/1902.05795.

317 arXiv:1902.05795 [cs]. 318 Mariusz Wisniewski, Paraskevas Chatzithanos, Weisi Guo, and Antonios Tsourdos. Benchmarking 319 Deep Reinforcement Learning for Navigation in Denied Sensor Environments, October 2024. URL 320 http://arxiv.org/abs/2410.14616. arXiv:2410.14616 [cs]. 321 Shuguang Yu, Shuxing Fang, Ruixin Peng, Zhengling Qi, Fan Zhou, and Chengchun Shi. Two-way 322 Deconfounder for Off-policy Evaluation in Causal Reinforcement Learning. 323 Yan Zeng, Ruichu Cai, Fuchun Sun, Libo Huang, and Zhifeng Hao. A Survey on Causal Reinforce324 ment Learning, June 2023. URL http://arxiv.org/abs/2302.05209. arXiv:2302.05209 325 [cs].

326 Shengyu Zhu, Ignavier Ng, and Zhitang Chen. Causal Discovery with Reinforcement Learning, June 327 2020. URL http://arxiv.org/abs/1906.04477. arXiv:1906.04477 [cs].

## 328 **A Implementation & Reproducibility Details**

329 **Hardware.** Training jobs were distributed over six AWS EC2 c7i.4xlarge instances (16 vCPU, 330 32 GiB RAM, 4th-Gen Xeon; AMI ami-00c257e12d6828491). Each instance executed an identical 331 slice of the sweep and finished in ≈ 12 h, yielding 6 × 12 = 72 CPU-hours of compute. Post332 processing and figure generation were run locally on an Apple M3 Pro laptop. 333 **Software environment.** 334 - Python 3.11.2 335 - stable-baselines3 2.3.0 (policy optimisation)
336 - gymnasium 0.29.1 (environments) 337 - tigramite 5.2.3 (PCMCI causal discovery) 338 - numpy 1.26.4, scipy 1.12, matplotlib 3.8 339 A fresh conda (or venv) install can be reproduced with: 340 conda create -n markov python=3.11 -y 341 conda activate markov 342 pip install stable-baselines3==2.3.0 gymnasium==0.29.1 \ 343 tigramite==5.2.3 matplotlib==3.8 344 **Script entry-points.** All functionality is exposed through a single orchestrator: 345 python markovianess/main.py \# run the full pipeline 346 python markovianess/main.py --env CartPole-v1 \# one environment only 347 The orchestrator reads a human-readable config.json file (specified with –config_path) that lists 348 (i) environments, *(ii)* training budgets, and *(iii)* noise/perturbation grids. An abridged example is 349 shown below (the full version is included in the supplementary ZIP): 350 { 351 "environments": [ 352 {"name":"CartPole-v1", "time_steps":30000, 353 "observations":["CartPos","CartVel","PoleAngle","PoleAngVel"], 354 "n_envs":1} 355 ], 356 "noise_strategies": { 357 "gaussian": [ 358 {"mean":0.0, "variance":0.01}, 359 {"mean":0.0, "variance":0.05}
360 ],
361 "auto_regressive": { 362 "AR(1)":[{"alphas":[0.9], "sigma":0.1}], 363 "AR(2)":[{"alphas":[0.9,0.1], "sigma":0.1}] 364 } 365 } 366 }
367 **Training hyper-parameters.** Across all conditions the PPO defaults from stable-baselines3 were used: two 64-unit TANH layers, Adam learning-rate 3 × 10−4 368 , discount γ = 0.99, GAE 369 λ = 0.95, clip 0.2, entropy 0, value-loss 0.5, mini-batch 64, and four optimisation epochs per update. 370 **No hyper-parameter tuning** was performed.

371 **Seed protocol.** Each EC2 worker drew one baseline seed uniformly from {0*, . . . ,* 1000}; that seed 372 initialised Gymnasium, network weights, NumPy, Python's random, and (where applicable) the noise 373 generator, and was re-used for all perturbations of the same environment on that worker. Accordingly, 374 every condition (baseline, each noise level, each dimension drop) has six statistically independent 375 replicas. Rollout seeds for PCMCI diagnostics were re-sampled independently for every analysis run. 376 **Running time.** A full sweep over all three environments (CartPole, Pendulum, *Acrobot*) and all 377 perturbation grids completes in ≈ 12 h wall-clock, matching the single instance runtime thanks to 378 six-way parallelism.

## 379 **Top-Level Workflow (One Ec2 Worker).**

380 for env in [CartPole, Pendulum, Acrobot]: 381 seed = run_baseline(env) \# clean PPO + PCMCI 382 run_noised_gaussian(env, seed) \# i.i.d. obs noise 383 run_noised_auto_regressive(env, seed) \# AR(p) obs noise 384 run_dropped(env, seed) \# drop one obs dimension 385 collect_results(env) \# rewards, MVS, plots 386 Conceptually, each call above does 387 1. wrap the Gymnasium environment with the requested perturbation, 388 2. train PPO for the budget in config.json, 389 3. record the reward curve, 390 4. run 3−5 extra roll-outs, estimate MVS with PCMCI, and 391 5. save all CSV files and figures under results/$ENV/. 392 All perturbations reuse the *same* baseline seed and hyper-parameters, so reward–vs–MVS comparisons 393 are fair and reproducible.

## 394 **B Additional Plots**

CartPole-v1, σ 2 = 0.02  (mean ± 95% Cl)
CartPole-v1, σ 2 = 1.0 (mean ± 95% CI)
300 300 250 250 200 200 Reward 150 150 100 100 50 50 0 0 600 100 200 300 400 200 300 500 0 0 100 400 Episode Episode
(a) CartPole, µ = 0, σ 2 = 0.02
(b) CartPole, µ = 0, σ 2 = 1.0 Pendulum-vl, σ 2 = 1.0  (mean ± 95% Cl)
Pendulum-vl, σ 2 = 2.0  (mean ± 95% Cl)
0 o
-250
-250
−500
−500
−750
−750
-1000
-1000
-1250
-1250
-1500
-1500 1000 2000 3000 4000 5000 1000 2000 3000 4000 5000 o 0 Episode Episode
(c) Pendulum, µ = 0, σ 2 = 1.0
(d) Pendulum, µ = 0, σ 2 = 2.0 Acrobot-v1, σ 2 = 0.02  (mean ± 95 % CI)
Acrobot-v1, σ 2 = 2.0  (mean ± 95 % CI)
-100
−100
-200
-200
−300
−300
-400
-400
−500
−500 200 o 100 300 400 100 200 300 400 0 Episode Episode
(e) Acrobot, μ = 0, σ 2 = 0.02
(f) Acrobot, μ = 0, σ 2 = 2.0
(a) **CartPole-v1** (b) **Pendulum-v1** (c) **Acrobot-v1** (a) **CartPole-v1** (b) **Pendulum-v1** (c) **Acrobot-v1**

## 395 **Neurips Paper Checklist**

396 1. **Claims**
397 **Question:** Do the main claims made in the abstract and introduction accurately reflect the 398 paper's contributions and scope?

399 **Answer:** [Yes] 400 **Justification:** The abstract and §1 make exactly three claims: (i) introduction of a Markov401 *Violation Score (MVS)* that condenses multi-lag dependence into a single scalar; (ii) an 402 empirical study demonstrating a clear, link between MVS and PPO performance across 403 graded noise levels; and (iii) causal-discovery–guided dimension ablations showing that 404 removing low-influence variables leaves rewards largely unaffected, whereas injecting 405 noise into high-influence ones severely degrades learning. No results beyond these three 406 contributions are asserted. 407 2. **Limitations** 408 Question: Does the paper discuss the limitations of the work performed by the authors? 409 Answer: [Yes] 410 Justification: §6 details task breadth, algorithm coverage (PPO only), linear-test assumptions, 411 and real-world realism as explicit limitations. 412 3. **Theory assumptions and proofs** 413 Question: For each theoretical result, does the paper provide the full set of assumptions and 414 a complete (and correct) proof?

415 Answer: [NA] 416 Justification: The paper is empirical and introduces no formal theorems.

## 417 4. **Experimental Result Reproducibility**

418 Question: Does the paper fully disclose all the information needed to reproduce the main 419 experimental results? 420 Answer: [Yes] 421 Justification: §5.1 lists all PPO default hyper-parameters from stablebaseline-3, training 422 budgets and environments; §5.2 gives the seed protocol; a full JSON config and scripts are 423 linked in the public repository and Supplementary review materials contains full source 424 code and config files and results. 425 5. **Open access to data and code** 426 Question: Does the paper provide open access to the data and code? 427 Answer: [Yes] 428 Justification: A GitHub link (anonymised for review) contains all source code, configs, 429 and plotting scripts; all tasks use the open-source Gymnasium benchmark environments. 430 Supplementary review materials contains full source code and config files and results needed 431 for reproduction. 432 6. **Experimental setting/details** 433 Question: Does the paper specify all the training and test details necessary to understand the 434 results? 435 Answer: [Yes] 436 Justification: Hyper-parameters (§5.1), seed choice and CI computation (§5.2), and PCMCI 437 settings (config JSON in repo) are fully enumerated. 438 7. **Experiment statistical significance** 439 Question: Does the paper report error bars or other significance information? 440 Answer: [Yes]
441 Justification: §5.2 explains that each condition is run with six seeds and reports mean 442 ±95% CI; tables in are reported in source code as csv.

## 443 8. **Experiments Compute Resources**

444 **Question:** For each experiment, does the paper provide sufficient information on the 445 computer resources (type of compute workers, memory, time of execution) needed to 446 reproduce the experiments?

447 **Answer:** [Yes]
448 **Justification:** All training jobs were executed on six AWS EC2 c7i.4xlarge instances 449 (16 vCPU 4th-Gen Xeon, 32 GiB RAM, no GPU). Each instance handled one-sixth of the 450 sweep and finished in ∼12 h wall-clock, totalling 6 × 12 = 72 CPU-hours for the full 451 study. Stand-alone timings: a 50 k-step CartPole/Acrobot run completes in ≈3 min; the 450 452 k-step Pendulum run in ≈40 min on a single instance. Post-processing and plotting were 453 performed locally on an Apple M3 Pro laptop (12-core CPU, 18 GiB RAM) and require 454 < 2 GB memory. All hardware details and runtimes are reported in §5.1.

455 9. **Code of ethics** 456 Question: Does the research conform with the NeurIPS Code of Ethics? 457 Answer: [Yes] 458 Justification: The work uses only publicly available simulators and releases code under an 459 open-source license; no privacy-sensitive or human data are involved. 460 10. **Broader impacts** 461 Question: Does the paper discuss both positive and negative societal impacts? 462 Answer: [Yes] 463 Justification: §6 highlights MVS as a potential safety diagnostic for real robots (positive) 464 and warns of over-confidence if mis-interpreted (negative). 465 11. **Safeguards** 466 Question: Are safeguards described for high-risk data/models? 467 Answer: [NA] 468 Justification: No high-risk data or large generative models are released. 469 12. **Licenses for existing assets** 470 Question: Are existing assets properly credited and licensed? 471 Answer: [Yes] 472 Justification: Gymnasium, Stable-Baselines3, and Tigramite (PCMCI) are cited with their 473 MIT or BSD licenses. 474 13. **New assets** 475 Question: Are new assets introduced in the paper well documented? 476 Answer: [NA] 477 Justification: The study releases only code; no new datasets or models are introduced. 478 14. **Crowdsourcing and research with human subjects** 479 Question: Does the paper include full instructions and screenshots? 480 Answer: [NA] 481 Justification: No human-subject or crowdsourcing data were collected. 482 15. **Institutional review board (IRB) approvals** 483 Question: Does the paper describe participant risks and IRB approval? 484 Answer: [NA] 485 Justification: Not applicable—no human subjects. 486 16. **Declaration of LLM usage** 487 Question: Does the paper describe usage of LLMs if relevant? 488 Answer: [NA] 489 Justification: Large language models are not part of the methodology.