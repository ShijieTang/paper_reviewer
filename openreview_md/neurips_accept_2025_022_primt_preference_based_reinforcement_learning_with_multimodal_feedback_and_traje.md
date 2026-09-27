# Primt: Preference-Based Reinforcement Learning With Multimodal Feedback And Trajectory Synthesis From Foundation Models

Ruiqi Wang1∗ Dezhong Zhao1,2∗ Ziqin Yuan1∗ **Tianyu Shao**1 Guohua Chen2 Dominic Kao1,3 Sungeun Hong4 Byung-Cheol Min1,5 1Purdue University, West Lafayette, IN, USA
2Beijing University of Chemical Technology, Beijing, China 3University of Illinois Urbana-Champaign, Champaign, IL, USA
4Sungkyunkwan University, Seoul, South Korea 5Indiana University Bloomington, Bloomington, IN, USA

## Abstract

Preference-based reinforcement learning (PbRL) has emerged as a promising paradigm for teaching robots complex behaviors without reward engineering. However, its effectiveness is often limited by two critical challenges: the reliance on extensive human input and the inherent difficulties in resolving query ambiguity and credit assignment during reward learning. In this paper, we introduce PRIMT, a PbRL framework designed to overcome these challenges by leveraging foundation models (FMs) for multimodal synthetic feedback and trajectory synthesis. Unlike prior approaches that rely on single-modality FM evaluations, PRIMT employs a hierarchical neuro-symbolic fusion strategy, integrating the complementary strengths of large language models and vision-language models in evaluating robot behaviors for more reliable and comprehensive feedback. PRIMT also incorporates foresight trajectory generation, which reduces early-stage query ambiguity by warm-starting the trajectory buffer with bootstrapped samples, and hindsight trajectory augmentation, which enables counterfactual reasoning with a causal auxiliary loss to improve credit assignment. We evaluate PRIMT on 2 locomotion and 6 manipulation tasks on various benchmarks, demonstrating superior performance over FM-based and scripted baselines. Website at https://primt25.github.io/.

## 1 Introduction

Reinforcement learning (RL) has shown great success in various robotics domains [1–4], yet it remains reliant on carefully designed reward functions. In many practical scenarios, designing an informative reward function is highly challenging, as task objectives are often implicit and multifaceted [5]. Preference-based RL (PbRL) [6–8] has emerged as a promising alternative to address this challenge by learning reward models from human comparative feedback over robot trajectories, providing a more intuitive means of aligning robotic systems with human intent [9–12]. Nevertheless, the extensive human input required for preference labeling restricts the scalability of PbRL [13].

To mitigate this bottleneck, recent work has explored leveraging foundation models (FMs), e.g.,
large language models (LLMs) and vision-language models (VLMs), as synthetic feedback sources, drawing on their broad world knowledge [14–17]. Compared to using FMs to design dense reward functions [18–20] or provide auxiliary contrastive signals [21–24], incorporating them as evaluators within PbRL offers a potentially more efficient and robust paradigm.

∗Equal contribution. Corresponding authors: wang5357@purdue.edu; minb@iu.edu

Off-Policy RL
Foresight Trajectory **Generation**
Hierarchical Neuro-Symbolic Preference **Fusion**
Keyframe Visual Rendering Crowd Check VLMs
( , ) ...

(

) (
)
: 
 ≈ 

: . 

Intra Fusion Bootstrap Trajectories LLM
Task Context
(

, 
)
Textual Projection Environment
( , )
(

)
TCP:[, 
 **,...,**, 
 ]
Grip:[,   **,...,**,   ]
Obj:[, 
**,...,**, 
]
Tgt:[, 
 **,...,**, 
 ]
(
)
TCP:[, 
 **,...,**, 
 ]
Grip:[, 
 **,...,**, 
 ]
Obj:[, 
 **,...,**, 
 ]
Tgt:[, 
 **,...,**, 
 ]
Crowd Check Warm-start Initialization
: 
 > 

: . 

Intra Fusion
... 

LLMs Policy Trajectory Buffer Rollout Sampling

Training Reward Learning with Causal Auxiliary **Loss**
Feature Extraction PSL-based InterFusion Trajectory **Context**
Hindsight Trajectory **Augmentation**
Identify the preference cause Action *Abduction* Minimal intervention to key causal **steps**
Prediction Predict outcome under change SCM-based Reasoning Counterfactual Trajectories Reward Model
(

, 

, , 
)
: 
 > 

(

, 
, )
LLM
However, obtaining reliable and high-quality FM feedback remains challenging, primarily due to the dominant reliance on *single-modality* evaluation. LLM-based approaches [15, 16] interpret structured textual projections of trajectories, such as sequential arrays of state-action pairs, enabling sophisticated temporal reasoning over procedural logic and motion progression [25, 26]. However, these textual descriptions can be abstract or incomplete, making LLMs prone to hallucinations of key events, especially when inferring fine-grained spatial interactions [27, 28]. On the other hand, VLM-based methods [14] analyze spatial cues from visual renderings of robot trajectories, such as final-state images or intermediate frames, effectively capturing spatial goal completion [29]. Yet these methods often overlook subtle temporal dynamics within the trajectory [30, 31]. Consequently, relying on either modality alone risks incomplete or unreliable feedback (see Appendix A for more analysis), highlighting the need for a more comprehensive multimodal evaluation framework. Furthermore, even if feedback from FMs reaches human-expert-level quality, PbRL still faces two intrinsic challenges: i) Query ambiguity: trajectory pairs often exhibit uniformly low quality in early training stages. This happens when they are generated from random or weakly optimized policies and lack task-relevant variations, making it hard to elicit meaningful preferences [7, 32];
and ii) Credit assignment: even when reliable *trajectory-level* preferences are available, it often remains difficult to accurately attribute the observed preference differences to specific states or actions
[33–35]. Without effective *state-action-level* credit assignment, the learned reward model may result in misaligned behaviors in subsequent RL training [33]. Meanwhile, FMs have shown strong abilities in planning [36, 37], control [38, 39], and causal reasoning [40, 41]. These advances lead us to the following question: Can FMs move beyond serving as passive preference providers to be actively leveraged to mitigate query ambiguity and improve credit assignment in PbRL?

In this paper, we propose PRIMT, a foundation model–driven framework for PbRL designed to address key challenges in synthetic feedback quality, query ambiguity, and credit assignment. PRIMT, as illustrated in Fig. 1, comprises two core components: i) *Multimodal feedback fusion*, which enhances synthetic feedback quality by combining the complementary advantages of LLM- and VLM-based evaluations. Rather than directly feeding multimodal trajectory representations into generic multimodal FMs, PRIMT adopts a hierarchical neuro-symbolic preference fusion strategy. It first performs intra-modal fusion to produce modality-specific labels and confidence estimations. Then, inter-modal fusion is conducted using probabilistic soft logic (PSL) [42], which infers the final preference label via structured and interpretable reasoning over multimodal evaluation outputs and trajectory context, enabling robust aggregation of preference beliefs from heterogeneous sources.

ii) *Bidirectional trajectory synthesis*, which leverages FMs to actively enhance reward learning in PbRL. In the foresight phase, LLMs generate diverse task-aligned trajectories to initialize the trajectory buffer. Unlike prior work [16, 36] that assumes FM-generated trajectories are optimal, we treat them as semantically meaningful anchors for informative comparisons, reducing early-stage query ambiguity. In the hindsight phase, LLMs are prompted to generate counterfactual trajectories via causal reasoning based on the structural causal model (SCM) [43]. When a clear preference is detected, the model identifies causal steps and applies minimal edits to reverse the preference, producing counterfactuals that highlight critical distinctions. To better exploit these samples for credit assignment, we introduce a causal-aware auxiliary loss that enforces reward separation at edited steps while ensuring consistency elsewhere. This enables more precise preference attribution, thereby improving the efficiency and generalization of the learned reward model in downstream RL training. Our key contributions are summarized as follows: - We present PRIMT, a general FM-driven framework for zero-shot PbRL, which leverages foundation models not only as synthetic teachers to eliminate human annotation but also as active agents to facilitate preference reward learning.

- We introduce a hierarchical neuro-symbolic preference fusion strategy that combines the complementary strengths of LLMs and VLMs for multimodal evaluation of robot trajectories, improving the reliability and quality of synthetic feedback.

- We propose foresight trajectory generation to bootstrap early-stage query informativeness, and hindsight trajectory augmentation via counterfactual reasoning, coupled with a causal auxiliary loss, to improve credit assignment in reward learning.

- We conduct extensive experiments across 2 locomotion and 6 manipulation tasks from the DMC [44], MetaWorld [45], and ManiSkill [46] benchmarks, demonstrating that PRIMT consistently outperforms state-of-the-art baselines. Ablation studies provide insight into component effectiveness, and we further validate PRIMT's real-world applicability on a Kinova Jaco robot.

## 2 Related Works And Preliminaries

Foundation Models as Rewards for RL Foundation models refer to large-scale pre-trained models with strong generalization and reasoning capabilities across tasks [47]. Recent work has explored leveraging FMs to address the reward engineering challenge in RL. One line of research uses coding LLMs to directly generate executable reward functions [18–20]. Another employs VLMs as contrastive reward signals [21–24]; for example, RoboCLIP [24] rewards agents by aligning trajectory images with task descriptions or demonstrations. However, such explicit FM-based reward signals are often noisy and high-variance [14]. Recent approaches have adopted the PbRL paradigm, using FMs as synthetic evaluators to generate trajectory-level preference labels and train reward models: PrefCLM [15] and RL-SaLLM-F [16] leverage LLMs to analyze numerical state-action sequences, while RL-VLM-F [14] uses VLMs to assess final-state images of robot trajectories. These approaches have shown improved performance over FM-generated scalar rewards. Our work builds on this PbRL- with-FM direction but introduces two key innovations. First, PRIMT adopts a multimodal evaluation scheme that combines VLM and LLM perspectives via hierarchical neuro-symbolic fusion, improving the robustness and quality of synthetic supervision. Second, rather than using FMs solely for passive evaluation, we actively incorporate them to facilitate reward learning via trajectory synthesis.

Preference-based RL PbRL aims to learn a reward model rψ from human comparative feedback over pairs of robot trajectories [7]. A trajectory σ is defined as a sequence of states and actions
{(s1, a1). . . ,(sT , aT )} with a length of T. The annotator provides a preference label Υ *∈ {−*1, 0, 1}
for each pair (σ A, σB), where Υ = 1 indicates that σ A is preferred, 0 means σ B is preferred, and
−1 denotes indecision. A preference predictor is constructed using the Bradley-Terry model [48] to estimate the preference probabilities. The likelihood that σ A is preferred over σ B is computed as:

.$$ P_{\psi}[\sigma^A\succ\sigma^B]=\dfrac{\exp\left(\sum_{t=1}^T r_{\psi}(s_t^A,a_t^A)\right)}{\exp\left(\sum_{t=1}^T r_{\psi}(s_t^A,a_t^A)\right)+\exp\left(\sum_{t=1}^T r_{\psi}(s_t^B,a_t^B)\right)}$$  odel is trained to align with human reference by minimizing a cross entry. 
 (1)
The reward model is trained to align with human preferences by minimizing a cross-entropy objective over a collected preference dataset D = {(σ A, σB, Υ)} as:
LPref = −E(σA,σB,Υ)∼D -I{*Υ = 1*} log Pψ[σ B ≻ σ A] + I{Υ = 0} log Pψ[σ A ≻ σ B](2)
Training alternates between reward learning and RL-based policy optimization with learned reward.

$$(1)$$

Query Ambiguity PbRL relies on informative preference queries to train effective reward models. However, trajectory pairs often exhibit low task-relevant diversity, leading to query ambiguity [8]. This ambiguity is especially pronounced in early training stages, when trajectories are uniformly lowquality and incoherent due to randomly initialized policies [49]. Prior works address this by selecting maximally distinguishable or uncertain samples [8, 32, 50] or initializing the reward model with expert demonstrations [51, 52]. More recent works employ LLMs to revise ambiguous trajectories during training, assuming that the edited versions are task-complete and preferred [16]. In contrast, we proactively initialize the replay buffer with LLM-generated trajectories that are diverse and task-aligned. Unlike prior work, we do not assume these are optimal, but use them as preference anchors to support more informative and efficient early-stage evaluation. Credit Assignment Another core challenge in PbRL is the granularity mismatch between trajectorylevel preference supervision and the desired state-action-level reward signal [33]. This mismatch introduces uncertainty in attributing credit to specific decisions, impairing both the alignment and generalization of the learned reward model [53]. Prior work mitigates this issue by training transformerbased world models that estimate state importance [33, 54, 55], or by collecting additional human annotations to highlight key moments [34]. In contrast, our approach requires neither extra supervision nor architectural changes. Inspired by causal counterfactual reasoning [43, 56–58], we prompt LLMs to generate hindsight-based counterfactual trajectories by minimally editing key decision points in the preferred trajectory to reverse the preference. By asking, "What minimal change would make this trajectory less preferred?", we obtain contrastive examples that expose the underlying reasons for preference. To effectively leverage these counterfactuals in reward learning, we introduce a causal-aware auxiliary loss. It enforces reward separation at edited points while maintaining consistency in the unedited parts, leading to more precise credit assignment.

## 3 Methodology

In this section, we present PRIMT: PReference-based reInforcement learning with Multimodal feedback and Trajectory synthesis from foundation models. An overview of the PRIMT is illustrated in Fig. 1. Detailed prompts with example outputs for each component are included in Appendix C.

## 3.1 Multimodal Feedback Generation And Fusion

Trajectory Preprocessing Given a trajectory pair (σ A, σB) sampled from the trajectory buffer, we first obtain their textual projections *text*(σ A) and *text*(σ B) for LLM-based evaluation, following [16]. These projections organize each trajectory into dimension-specific sequences, capturing structured temporal patterns across state and action components in a format that enhances semantic interpretability. For VLM-based evaluation, instead of using all frames or final-state images as in prior work [14], we propose a hybrid keyframe extraction method to capture both low-level motion cues and high-level behavior transitions while avoiding visual overload: i) near-zero velocity detection identifies frames where the robot motion is minimal, typically marking subgoal completions or transitional pauses [59]; ii) smoothing residual peaks detect high-curvature or abrupt motion transitions by comparing the raw trajectory to its smoothed version, capturing key motion shifts [60]; and iii) change point detection segments the trajectory into semantically coherent phases to identify structural high-level task changes [61]. We take the union of the selected frames from all three methods, together with the first and last steps of the trajectory, to form the final keyframe sets *kvis*(σ A) and kvis(σ B). Full details are provided in Appendix B.

Intra-modal Preference Fusion We then query LLM and VLM separately with corresponding textual projections and keyframe sequences, along with a brief task description, to elicit preference judgments. Each query follows a structured three-step chain-of-thought (CoT) prompt: i) analyze each trajectory in terms of its effectiveness in achieving the task goal; ii) based on this analysis, output a preference label; and iii) verify the decision and assign a confidence score from 0 to 1, reflecting the preference certainty. To mitigate variance and improve the reliability of intra-modal labels, we adopt a crowd-check mechanism, querying LLM or VLM multiple times with randomly permuted trajectory orderings. This produces K predictions from each feedback modality M ∈ {LLM, VLM},
each consisting of a preference label Υ
(k)
M *∈ {−*1, 0, 1} and a confidence score C
(k)
M ∈ [0, 1]. We then aggregate these judgments into a final modality-specific preference label ΥM via major voting as:

$$\Upsilon_{M}=\operatorname*{argmax}_{l\in\{-1,0,1\}}\sum_{k=1}^{K}\mathbb{I}(\Upsilon_{M}^{(k)}=l)$$

To estimate and calibrate the confidence CM associated with the final label, we compute a weighted combination of two complementary signals: i) the average confidence C¯M among N judgments that agree with the final label, and ii) the label consistency ratio C˙M representing vote agreement:

$$\vec{\mathcal{C}}_{M}=\frac{1}{N}\sum_{k=1}^{K}\mathcal{C}_{M}^{(k)}\cdot\mathbb{I}(\Upsilon_{M}^{(k)}=\Upsilon_{M});\ \ \hat{\mathcal{C}}_{M}=\frac{1}{K}\sum_{k=1}^{K}\mathbb{I}(\Upsilon_{M}^{(k)}=\Upsilon_{M})\tag{4}$$
$$({\mathfrak{I}})$$
$$(5)$$

$\left(6\right)$. 
The final confidence CM is then computed as:

$${\mathcal{C}}_{M}=\alpha\cdot{\vec{\mathcal{C}}}_{M}+(1-\alpha)\cdot{\dot{\mathcal{C}}}_{M}$$
CM = α · C¯M + (1 − α) · C˙M (5)
where α ∈ [0, 1] is a balancing hyperparameter (typically set to 0.5). This formulation ensures that the final confidence reflects both internal certainty and stability under input perturbations, thereby improving the robustness of modality-specific confidence estimation. Inter-modal Preference Fusion The next step is to integrate modality-specific preference labels into a unified decision. This process is non-trivial, as it must consider multiple factors, including intramodal uncertainty, cross-modal conflicts, and trajectory context that reflects the relative difficulty of visual versus textual evaluation. Intuitively, one might define heuristic rules for each factor, for example, favoring the VLM label when the visual difference between trajectories is high, or trusting the label with higher confidence. Yet, such heuristics are brittle and hard to generalize: the conditions involved are often continuous rather than binary, and the interactions among rules can be complex. To efficiently model these latent dependency structures among inputs, heuristics, and decisions, we employ Probabilistic Soft Logic (PSL) [42], a probabilistic framework representing entities of interest as logical *atoms* interconnected by weighted first-order logic *rules*. Specifically, we define four rules to guide inter-modal preference fusion: i) Agreement Rule: If the VLM and LLM agree on the same preference label Υ and at least one modality reports high confidence, the agreed label is used as the final decision:
∀Υ, M : IsAgree(Υ) ∧ ConfHigh(M) → FinalLabel(Υ) (6)
Here, IsAgree(Υ) is a binary indicator set to 1 if both VLM and LLM predict the same label Υ, and 0 otherwise; ConfHigh(M) is a continuous atom representing the modality-specific confidence score CM ∈ [0, 1]; and FinalLabel(Υ) ∈ [0, 1] is the output atom to be inferred by PSL, representing the final soft confidence assigned to label Υ *∈ {−*1, 0, 1}. ii) Conflict Resolution Rules: When modality-specific labels conflict, we resolve the disagreement by considering the associated confidence and trajectory context (detailed rationale can be found in Appendix A.2). Specifically, we prioritize the VLM prediction if the visual discriminability between trajectories and VLM confidence is high:
∀Υ : ¬IsAgree(Υ) ∧ VLMLabel(Υ) ∧ ConfHigh(VLM) ∧ VDHigh∧ → FinalLabel(Υ) (7)
Likewise, if the LLM predicts a label Υ with high confidence and the temporal discriminability of the trajectory pair is high, we prioritize the LLM prediction:
∀Υ : ¬IsAgree(Υ) ∧ LLMLabel(Υ) ∧ ConfHigh(LLM) ∧ TDHigh → FinalLabel(Υ) (8)
Here, VLMLabel(Υ) and LLMLabel(Υ) are indicators set to 1 if the modality predicts label Υ, and 0 otherwise. The atom VDHigh captures the visual discriminability between the two trajectories as:

$\downarrow$ . 
$$\mathsf{V D H i g h}=\rho\left(\mathcal{W}(f(k v i s(\sigma^{A})),f(k v i s(\sigma^{B})))\right)$$
B)))(9)
where f(·) denotes the CLIP encoder applied to keyframe sets *kvis*(·), W denotes the Wasserstein distance, and ρ(·) is a sigmoid function used for normalization.

Similarly, TDHigh captures temporal discriminability based on trajectory volatility differences:

$$\mathrm{TDHigh}:=\rho\left(|\mathrm{TrjVol}(\sigma^{A})-\mathrm{TrjVol}(\sigma^{B})|\right)$$
B)(10)
$$(\mathbb{Q})$$
$$(10)$$

where TrjVol(·) measures the state-action volatility of a trajectory, defined as the mean L2 norm of second-order finite differences:

$$\mathrm{TrjVol}(\sigma)=\frac{1}{T-2}\sum_{t=2}^{T-1}\left\|(s_{t+1},a_{t+1})-2(s_{t},a_{t})+(s_{t-1},a_{t-1})\right\|_{2}.$$
$$(11)$$
$$(12)^{\frac{1}{2}}$$
$\frac{1}{2}-1$). 
. (11)
iii) Indecision Rule: When both modalities exhibit low confidence, we assign the indecision label:

$$\lnot\mathrm{ConfHigh}(\mathrm{VLM})\land$$

¬ConfHigh(VLM) ∧ ¬ConfHigh(LLM) → FinalLabel(−1) (12)
During PSL inference, each logical atom is instantiated with data and grounded into either an observed input variable X (e.g., IsAgree, ConfHigh, TDHigh) or an output variable Y (e.g., FinalLabel). Valid substitutions of these atoms within rule templates yield a set of ground rules. Each ground rule induces one or more hinge-loss potentials, relaxed from the logical clauses using Łukasiewicz continuous-valued semantics. Formally, each potential takes the form:

$$\phi(Y,X)=[\operatorname*{max}(0,\ell(Y,X))]^{p}$$
$$(13)^{\frac{1}{2}}$$

ϕ(*Y, X*) = [max(0, ℓ(*Y, X*))]p(13)
where ℓ is a linear function in PSL representing the distance to satisfaction of the corresponding ground rule, and p ∈ {1, 2} controls whether the penalty is linear or quadratic. Given observed variables X and target variables Y , PSL defines a hinge-loss Markov random field and performs inference by solving a convex constrained optimization problem (more details of PSL inference are provided in Appendix D):

$$Y^{*}=\arg\min_{Y}\sum_{i=1}^{m}w_{i},\phi_{i}(Y,X)\quad\text{s.t.}\sum_{\Upsilon\in\{-1,0,1\}}\text{FinalLabel}(\Upsilon)=1\tag{14}$$

where m is the number of instantiated potentials, ϕi denotes the i th potential function, and wiis the weight assigned to the corresponding rule template. Unlike standard PSL formulations, we impose a one-hot constraint over the final label atoms to reflect the single-label nature in PbRL. By encoding structured dependencies among modality-specific outputs and trajectory-level context, PSL facilitates robust and adaptive integration of complementary cues from multiple feedback modalities, effectively managing uncertainties and cross-modal conflicts.

## 3.2 Bidirectional Trajectory Synthesis

Foresight Trajectory Generation Prior to PbRL training, we employ LLMs to generate bootstrapped trajectories that exhibit diverse, semantically meaningful, and task-aligned behaviors, providing a warm-start initialization for the trajectory buffer. Inspired by structured code-generation paradigms [36, 38], we adopt a three-step CoT strategy: i) generate a high-level, multi-step action plan from the task specification; ii) translate each step into executable code snippets that implement concrete motion primitives; and iii) execute these programs under varied initial conditions (e.g., robot start positions) and strategy parameters (e.g., height to approach the target) to synthesize a diverse set of plausible trajectories. Compared to directly prompting LLMs to generate low-level trajectory arrays [16], our method improves physical feasibility and semantic coherence by grounding trajectory synthesis in program logic. The generated trajectories are considered as bootstrapped demonstrations rather than optimal ones, subsequently evaluated by our multimodal feedback module. Combined with strategic sampling schemes, such as uncertainty-based sampling [8], these trajectories serve as informative preference anchors when paired with exploration trajectories, reducing ambiguity in early-stage preference queries and accelerating reward learning. Hindsight Trajectory Augmentation with Causal Auxiliary Loss During PbRL training, whenever a clear preference is identified by the multimodal feedback module, we prompt LLMs to perform hindsight reasoning to generate counterfactual variants of the preferred trajectory. This process follows a three-step reasoning pattern based on the structural causal model [43]:
i) Abduction: Identify the causal rationale behind the observed preference by extracting a set of critical causal steps T
∗in the preferred trajectory σ
∗that contribute to the preference. To assist this process, we provide the step indices corresponding to keyframes in *kvis*(σ
∗) as reference candidates, though the selected causal steps are not necessarily limited to them. ii) Action: Select a key step t
∗ ∈ T
∗for minimal intervention, generating a counterfactual trajectory σ
∗
cf that reverses the original preference. This involves modifying some critical state-action features at the selected step, such as introducing a small gripper delay or adding a local end-effector position perturbation. The rest of the trajectory remains identical to the original while we allow the LLMs to apply light smoothing to the immediate neighbors (e.g., 2-3 steps before and after the intervention) to ensure physical continuity and avoid abrupt state transitions. Multiple counterfactual variants can be generated through repeated LLM sampling, providing a diverse set of sub-preferred alternatives. Following the minimal edit principle [62, 63], we filter the generated counterfactuals based on the L1 distance between the edited state-action pairs and the original, ensuring a small deviation threshold. iii) Prediction: Pair each counterfactual variant with the originally preferred trajectory and feed them into the LLM-based intra-modal fusion module to verify whether the counterfactual is sub-preferred, i.e., satisfying the preference condition (σ
∗ ≻ σ
∗
cf ). Only counterfactuals that meet this criterion are stored and used for reward learning. Through this hindsight reasoning process, we now have counterfactual trajectories of the preferred trajectory that share a common structure except at minimally edited steps, because of which their preference outcomes diverge. As such, we can assume that the edited steps are responsible for the observed preference signal, which the reward model should learn to correctly attribute. Given this, we introduce a causal auxiliary loss that encourages discriminability at the edited steps while maintaining consistency elsewhere to guide the model to focus on causal differences that drive preferences:

$$\mathcal{L}_{\text{cf}}^{\text{avg}}=\underbrace{\sum_{t=1}^{T}H_{t}\cdot\log\left(1+\exp\left(r_{\psi}(s_{t}^{ef})-r_{\psi}(s_{t}^{*})\right)\right)}_{\text{i)actual constant loss}}+\underbrace{\sum_{t=1}^{T}(1-H_{t})\cdot\left\|r_{\psi}(s_{t}^{*})-r_{\psi}(s_{t}^{ef})\right\|_{2}^{2}}_{\text{ii)reward consistency loss}}\tag{15}$$
$$(16)^{\frac{1}{2}}$$

where Ht is a binary mask indicating the edited steps, i.e., Ht = 1 for edited time steps and Ht = 0 otherwise. The first term encourages the model to assign higher rewards to the preferred trajectory at casual steps, while the second enforces consistent rewards on unchanged regions. This auxiliary loss is combined with the trajectory-level preference loss as in Eq. 2, forming the final loss for reward learning with the generated counterfactuals:

$${\mathcal{L}}_{\mathrm{final}}={\mathcal{L}}_{\mathrm{pref}}+\lambda_{\mathrm{cf}}\cdot{\mathcal{L}}_{\mathrm{cf}}^{\mathrm{aux}}$$
cf (16)
where λcf is a weight used to scale the auxiliary loss to the same magnitude as the primary preference loss. This integrated loss formulation enables the reward model to capture trajectory-level preferences while providing more precise state-action-level credit attributions.

## 4 Experiments 4.1 Setup

We evaluate PRIMT across a diverse set of tasks, including 2 locomotion tasks: *Hopper Stand* and *Walker Walk* from DeepMind Control (DMC) suite [44], as well as 6 articulation or rigid body manipulation tasks: Button Press, *Door Open*, and *Sweep Into* from MetaWorld suite [45], and PickSingleYCB, *StackCube*, and *PegInsertionSide* from ManiSkill suite [46]. Detailed task descriptions are provided in Appendix E. We compare PRIMT against the following baselines: - **RL-VLM-F [14]:** This baseline utilizes VLM to analyze visual renderings of trajectories to provide preference labels, representing a state-of-the-art VLM-based method.

- **RL-SaLLM-F [16]:** This model employs LLM to provide preference labels based on textual descriptions of trajectories, and to modify ambiguous trajectories by generating self-improved alternatives, which are assumed to be preferred when paired with the original ones. This denotes a LLM-based method with trajectory augmentation for addressing query ambiguity.

- **PrefCLM [15]:** This baseline leverages crowdsourced LLMs to provide evaluation feedback for improved feedback quality, presenting another state-of-the-art LLM-based method.

- **PrefMul:** We build this baseline by directly providing the multimodal trajectory inputs used in PRIMT to a multimodal FM for evaluation, representing a naive approach to multimodal feedback.

- **PrefGT:** This baseline uses expert-designed reward functions provided by the benchmarks in a scripted teacher manner [8] to provide preference labels. This should, in theory, serve as an upper-bound oracle of PbRL performance on each task.

We also build several ablation models to assess the impact of each PRIMT component: - **w/o.Intra:** without the crowd-check mechanism and intra-modal preference fusion module. - **w/o.Inter:** without the inter-modal preference fusion module, directly selecting the modalityspecific label with highest confidence as the final label.

- **w/o.ForeGen:** without the foresight trajectory generation module. - **w/o.HindAug:** without the hindsight trajectory augmentation module. - **w/o.CauAux:** without the causal auxiliary loss for counterfactuals in reward learning. To eliminate the impact of non-model factors, we use the same trajectory inputs and CoT prompts as in PRIMT for preference label elicitation across all FM-based baselines and ablation models. The only exception is PrefCLM, which relies heavily on direct access to the environment code; for this baseline, we follow the original settings from the source paper. For all FM-based methods, we use gpt-4o as the LLM backbone. For the PbRL backbone, we use PEBBLE [7] with the uncertaintybased sampling schedule [8] for all methods, along with a consistent set of hyperparameters for the RL-based policy learning phase with SAC. This design ensures that the only difference between methods lies in the preference reward learning, allowing for a more controlled comparison.

We evaluate all baselines across all tasks and conduct ablation studies on the *Door Open* and PickS- ingleYCB tasks, each with five random seed runs to ensure statistical robustness. For manipulation tasks, we report the success rate, whereas for locomotion tasks, we use the episodic return provided by the benchmarks. Further implementation details are provided in Appendix F. 4.2 Does PRIMT Learn Effective Rewards and Policies that Prompt Task Performance?

We first investigate whether PRIMT
can learn effective reward models that lead to policies capable of solving complex tasks. Fig. 2 shows the learning curves of all methods across 8 tasks. We observe that PRIMT consistently outperforms all FM-based baselines that rely on single-modality feedback, demonstrating superior final performance and faster convergence. However, the naive multimodal feedback method, PrefMul, performs poorly. We attribute this to the fact that directly feeding multimodal trajectory inputs to multimodal FMs without appropriate fusion can overwhelm the Figure 3: Learning curves of PRIMT and ablation models on the Door Open and *PickSingleYCB* tasks.

models, potentially even degrading performance. This highlights the need for carefully designed fusion strategies to fully leverage the complementary strengths of multimodal signals, as implemented in the hierarchical neuro-symbolic fusion strategy of PRIMT. Moreover, each component in this hierarchical design is crucial: as shown in Fig. 3, removing either intra-modal or inter-modal fusion significantly degrades task performance. This drop is especially severe when inter-modal fusion is removed, underscoring the critical role of PSL-based reasoning in PRIMT. Unlike simple rule-based fusion in w/o.Inter, PSL-based fusion more effectively captures intra-modal uncertainty, cross-modal conflicts, and the influence of trajectory context, enabling more robust preference integration from heterogeneous sources. Notably, PRIMT is competitive with the oracle PrefGT, even surpassing it on the *Sweep Into* and *Peg Insertion Side* tasks, and generally achieving faster early-stage learning, while other FM- based methods fall behind. This suggests that while PrefGT benefits from fine-grained oracle preference labels, PRIMT narrows this gap by improving the quality of synthetic feedback and leveraging trajectory synthesis to address inherent challenges in PbRL. As shown in Fig. 3, foresight generation appears to contribute more to the acceleration of early-stage learning, as the w/o.ForeGen variant reaches a similar final performance but learns more slowly in the initial stages. In contrast, hindsight augmentation plays a critical role in achieving high final performance, as evidenced by the significant performance drop when either w/o.HindAug or w/o.CauAux is removed. Interestingly, while w/o.CauAux can still benefit from the counterfactual trajectory augmentation, its performance is significantly worse, indicating that the causal auxiliary loss we designed can more effectively leverage these counterfactuals in reward learning, capturing state-action-level preference causation.

## 4.3 **Does Primt Improve The Quality Of Synthetic Feedback And Mitigate Query Ambiguity?**

We next examine the distribution of synthetic feedback generated by PRIMT to assess its impact on preference label quality and query informativeness. We calculate accuracy by comparing the synthetic preference labels with those in PrefGT and record the preference decisions. Fig. 4 shows the percentages of correct labeling, incorrect labeling, and preference indecision for PRIMT, w/o.ForeGen, w/o.Intra, w/o.Inter, and RL-VLM-F (left to right). We observe that PRIMT consistently produces more accurate preference labels compared to the baseline and ablations, confirming the effectiveness of the hierarchical fusion design in improving label quality. Additionally, PRIMT significantly reduces indecision rates compared to both the baseline and w/o.ForeGen, indicating that foresight generation effectively mitigates query ambiguity.

It is worth noting that we could not directly compare indecision rates with RL-SaLLM-F, another method that uses LLMs to address query ambiguity, because it inherently eliminates indecision by always treating self-augmented trajectories as preferable. However, as shown in Fig. 2, this baseline struggles with early-stage learning on tasks from the ManiSkill, which involve high-dimensional state and action spaces. We attribute this to the difficulty of directly generating optimal trajectories at the low level in such tasks, making the assumption that generated trajectories are always preferable highly misleading. This highlights the advantages of our foresight generation approach, which addresses query ambiguity from the outset by initializing diverse, bootstrapped trajectories as potential preference anchors, and our code-generation paradigm, which improves the trajectory sample quality.

## 4.4 Does Primt Enhance Credit Assignment In The Reward Model?

We further investigate how the learned reward models align with the task progress at state-action level. Fig. 4 shows the normalized reward outputs from the learned reward models of PRIMT, its two ablations without full hindsight trajectory augmentation, and the baseline RL-VLM-F, along with the normalized ground-truth reward values on the same trajectories. We observe that PRIMT produces more aligned reward patterns that closely reflect task progress, while the baselines and ablations exhibit either noisy signals or high variance, indicating that their learned reward models struggle to accurately assign rewards at the state-action level, even if they capture trajectory-level preferences. This demonstrates that PRIMT, particularly its hindsight augmentation and causal auxiliary loss, enables more precise state-action-level credit assignment in reward learning. Extra quantitative results using the R2coefficient are provided in Appendix G.8.

## 4.5 Additional Experimental Results

We conducted additional analyses to provide further validation insights of the proposed framework. These include: qualitative evaluations of the trajectory synthesis module (Appendices G.3 and G.4), visualization of policy outcomes across different methods (Appendix G.5), comparison with the dense-reward RL baseline (Appendix G.7), and preliminary experiments on a more complex dual-arm manipulation task (Appendix G.6). We also performed ablation studies on the influence of the foundation model backbone (Appendix G.2). The results show good potential for more accessible deployment with smaller or cheaper foundation models: with GPT-4o-mini, we achieved a 94% reduction in cost with a 16–26% drop in performance, leading to a 13× improvement in cost–performance efficiency.

## 4.6 Real-World Deployment

We further demonstrated the effectiveness of PRIMT on a Kinova Jaco robot in block lifting and stacking tasks (Fig. 5). Detailed experimental settings and results are provided in Appendix G.9.

Block Lifting Block **Stacking**

## 5 Conclusion And Future Work

In this work, we presented PRIMT, a method that leverages foundation models for multimodal synthetic evaluation and trajectory synthesis to reduce human effort and address query ambiguity and credit-assignment challenges in preference-based reinforcement learning (PbRL). We demonstrated the advantages of PRIMT across a wide range of locomotion and manipulation tasks, achieving superior task performance, higher-quality synthetic feedback, and more accurately aligned reward models. A limitation of this study involves the increased cost associated with foundation-model usage due to the multimodal components of the framework. To provide a clearer understanding of the resource requirements, we conducted a detailed analysis of computational usage and the corresponding cost–performance trade-offs, presented in Appendix H.1. We observe that PRIMT strikes a good balance between performance and cost-effectiveness, providing a practical path toward scalable preference learning. Looking forward, while PRIMT was evaluated in single-agent robotic domains, extending its key components, such as multimodal evaluation and trajectory synthesis, to non-robotic and multiagent settings represents an exciting direction for future work. Further discussion of assumptions, limitations, and broader impacts is provided in Appendix H.

## Acknowledgments

The authors would like to thank Jinjin Cai from Purdue University for insightful comments on the figures and visualizations. This work was partially supported by the Ministry of Science and ICT (MSIT), Korea, under the Global Research Support Program in the Digital Field (RS-2024-00425354), supervised by the Institute for Information & Communications Technology Planning & Evaluation (IITP). The support applies to authors Hong and Min.

## References

[1] Xu Wang, Sen Wang, Xingxing Liang, Dawei Zhao, Jincai Huang, Xin Xu, Bin Dai, and Qiguang Miao.

Deep reinforcement learning: A survey. *IEEE Transactions on Neural Networks and Learning Systems*, 35 (4):5064–5078, 2022.

[2] Ruiqi Wang, Dezhong Zhao, Arjun Gupte, and Byung-Cheol Min. Initial task allocation in multi-human multi-robot teams: An attention-enhanced hierarchical reinforcement learning approach. IEEE Robotics and Automation Letters, 2024.

[3] Jens Kober, J Andrew Bagnell, and Jan Peters. Reinforcement learning in robotics: A survey. The International Journal of Robotics Research, 32(11):1238–1274, 2013.

[4] Wonse Jo, Ruiqi Wang, Baijian Yang, Daniel Foti, Mo Rastgaar, and Byung-Cheol Min. Cognitive load-based affective workload allocation for multihuman multirobot teams. IEEE Transactions on Human- Machine Systems, 2024.

[5] Abhishek Gupta, Aldo Pacchiano, Yuexiang Zhai, Sham Kakade, and Sergey Levine. Unpacking reward shaping: Understanding the benefits of reward engineering on sample complexity. Advances in Neural Information Processing Systems, 35:15281–15295, 2022.

[6] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. *Advances in neural information processing systems*, 30, 2017.

[7] Kimin Lee, Laura M Smith, and Pieter Abbeel. Pebble: Feedback-efficient interactive reinforcement learning via relabeling experience and unsupervised pre-training. In *International Conference on Machine* Learning, pages 6152–6163. PMLR, 2021.

[8] K Lee, L Smith, A Dragan, and P Abbeel. B-pref: Benchmarking preference-based reinforcement learning.

Neural Information Processing Systems (NeurIPS), 2021.

[9] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. *arXiv preprint arXiv:2204.05862*, 2022.

[10] Ruiqi Wang, Dezhong Zhao, Dayoon Suh, Ziqin Yuan, Guohua Chen, and Byung-Cheol Min. Personalization in human-robot interaction through preference-based action representation learning. In IEEE International Conference on Robotics and Automation (ICRA), 2025.

[11] Xiaofei Wang, Kimin Lee, Kourosh Hakhamaneshi, Pieter Abbeel, and Michael Laskin. Skill preferences:
Learning to extract and execute robotic skills from human feedback. In *Conference on Robot Learning*, pages 1259–1268. PMLR, 2022.

[12] Ruiqi Wang, Weizheng Wang, and Byung-Cheol Min. Feedback-efficient active preference learning for socially aware robot navigation. In *2022 IEEE/RSJ International Conference on Intelligent Robots and* Systems (IROS), pages 11336–11343. IEEE, 2022.

[13] Jongjin Park, Younggyo Seo, Jinwoo Shin, Honglak Lee, Pieter Abbeel, and Kimin Lee. Surf: Semisupervised reward learning with data augmentation for feedback-efficient preference-based reinforcement learning. *arXiv preprint arXiv:2203.10050*, 2022.

[14] Yufei Wang, Zhanyi Sun, Jesse Zhang, Zhou Xian, Erdem Biyik, David Held, and Zackory Erickson.

Rl-vlm-f: reinforcement learning from vision language foundation model feedback. In Proceedings of the 41st International Conference on Machine Learning, pages 51484–51501, 2024.

[15] Ruiqi Wang, Dezhong Zhao, Ziqin Yuan, Ike Obi, and Byung-Cheol Min. Prefclm: Enhancing preferencebased reinforcement learning with crowdsourced large language models. IEEE Robotics and Automation Letters, 2025.

[16] Songjun Tu, Jingbo Sun, Qichao Zhang, Xiangyuan Lan, and Dongbin Zhao. Online preference-based reinforcement learning with self-augmented feedback from large language model. *24th International* Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025), 2025.

[17] Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Ren Lu, Thomas Mesnard, Johan Ferret, Colton Bishop, Ethan Hall, Victor Carbune, and Abhinav Rastogi. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. 2023.

[18] Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Eureka: Human-level reward design via coding large language models. In *The Twelfth International Conference on Learning Representations*.

[19] David Venuto, Mohammad Sami Nur Islam, Martin Klissarov, Doina Precup, Sherry Yang, and Ankit Anand. Code as reward: Empowering reinforcement learning with vlms. In Forty-first International Conference on Machine Learning.

[20] Tianbao Xie, Siheng Zhao, Chen Henry Wu, Yitao Liu, Qian Luo, Victor Zhong, Yanchao Yang, and Tao Yu. Text2reward: Reward shaping with language models for reinforcement learning. In The Twelfth International Conference on Learning Representations.

[21] Yecheng Jason Ma, Vikash Kumar, Amy Zhang, Osbert Bastani, and Dinesh Jayaraman. Liv: Languageimage representations and rewards for robotic control. In *International Conference on Machine Learning*, pages 23301–23320. PMLR, 2023.

[22] Parsa Mahmoudieh, Deepak Pathak, and Trevor Darrell. Zero-shot reward specification via grounded natural language. In *International Conference on Machine Learning*, pages 14743–14752. PMLR, 2022.

[23] Juan Rocamonde, Victoriano Montesinos, Elvis Nava, Ethan Perez, and David Lindner. Vision-language models are zero-shot reward models for reinforcement learning. *arXiv preprint arXiv:2310.12921*, 2023.

[24] Sumedh Sontakke, Jesse Zhang, Séb Arnold, Karl Pertsch, Erdem Bıyık, Dorsa Sadigh, Chelsea Finn, and Laurent Itti. Roboclip: One demonstration is enough to learn robot policies. Advances in Neural Information Processing Systems, 36:55681–55693, 2023.

[25] Ruyang Liu, Chen Li, Haoran Tang, Yixiao Ge, Ying Shan, and Ge Li. St-llm: Large language models are effective temporal learners. In *European Conference on Computer Vision*, pages 1–18. Springer, 2024.

[26] Siheng Xiong, Ali Payani, Ramana Kompella, and Faramarz Fekri. Large language models can learn temporal reasoning. *arXiv preprint arXiv:2401.06853*, 2024.

[27] Fangjun Li, David C Hogg, and Anthony G Cohn. Advancing spatial reasoning in large language models:
An in-depth evaluation and enhancement using the stepgame benchmark. In *Proceedings of the AAAI* Conference on Artificial Intelligence, volume 38, pages 18500–18507, 2024.

[28] Hongjie Zhang, Hourui Deng, Jie Ou, and Chaosheng Feng. Mitigating spatial hallucination in large language models for path planning via prompt engineering. *Scientific Reports*, 15(1):8881, 2025.

[29] Fiona Luo. Vision-language models for robot success detection. In *Proceedings of the AAAI Conference* on Artificial Intelligence, volume 38, pages 23750–23752, 2024.

[30] Xi Ding and Lei Wang. Do language models understand time? *arXiv preprint arXiv:2412.13845*, 2024. [31] Tonko EW Bossen, Andreas Møgelmose, and Ross Greer. Can vision-language models understand and interpret dynamic gestures from pedestrians? pilot datasets and exploration towards instructive nonverbal commands for cooperative autonomous vehicles. *arXiv preprint arXiv:2504.10873*, 2025.

[32] Xuening Feng, Zhaohui JIANG, Timo Kaufmann, Eyke Hüllermeier, Paul Weng, and Yifei Zhu. Comparing comparisons: Informative and easy human feedback with distinguishability queries. In ICML 2024 Workshop on Models of Human Feedback for AI Alignment, 2024.

[33] Mudit Verma and Katherine Metcalf. Hindsight priors for reward learning from human preferences. In The Twelfth International Conference on Learning Representations.

[34] Simon Holk, Daniel Marta, and Iolanda Leite. Polite: Preferences combined with highlights in reinforcement learning. In *2024 IEEE International Conference on Robotics and Automation (ICRA)*, pages 2288–2295. IEEE, 2024.

[35] Xinran Liang, Katherine Shu, Kimin Lee, and Pieter Abbeel. Reward uncertainty for exploration in preference-based reinforcement learning. In *International Conference on Learning Representations*, 2021.

[36] Teyun Kwon, Norman Di Palo, and Edward Johns. Language models as zero-shot trajectory generators.

IEEE Robotics and Automation Letters, 2024.

[37] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. Do as i can, not as i say: Grounding language in robotic affordances. *arXiv preprint arXiv:2204.01691*, 2022.

[38] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 9493–9500. IEEE, 2023.

[39] Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. Voxposer: Composable 3d value maps for robotic manipulation with language models. In *Conference on Robot Learning*, pages 540–562. PMLR, 2023.

[40] Emre Kiciman, Robert Ness, Amit Sharma, and Chenhao Tan. Causal reasoning and large language models:
Opening a new frontier for causality. *Transactions on Machine Learning Research*, 2023.

[41] Norman Di Palo, Arunkumar Byravan, Leonard Hasenclever, Markus Wulfmeier, Nicolas Heess, and Martin Riedmiller. Towards a unified agent with foundation models. *arXiv preprint arXiv:2307.09668*, 2023.

[42] Stephen H Bach, Matthias Broecheler, Bert Huang, and Lise Getoor. Hinge-loss markov random fields and probabilistic soft logic. *Journal of Machine Learning Research*, 18(109):1–67, 2017.

[43] Judea Pearl. Causal inference. *Causality: objectives and assessment*, pages 39–58, 2010. [44] Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al. Deepmind control suite. arXiv preprint arXiv:1801.00690, 2018.

[45] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine.

Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In *Conference* on robot learning, pages 1094–1100. PMLR, 2020.

[46] Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, Xinyue Wei, Yunchao Yao, et al. Maniskill2: A unified benchmark for generalizable manipulation skills. arXiv preprint arXiv:2302.04659, 2023.

[47] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S
Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. *arXiv preprint arXiv:2108.07258*, 2021.

[48] Ralph A. Bradley and Milton E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. *Biometrika*, 39(3/4):324–345, 1952.

[49] Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, et al. Open problems and fundamental limitations of reinforcement learning from human feedback. *arXiv preprint arXiv:2307.15217*, 2023.

[50] Erdem Bıyık, Malayandi Palan, Nicholas C Landolfi, Dylan P Losey, and Dorsa Sadigh. Asking easy questions: A user-friendly approach to active reward learning. *arXiv preprint arXiv:1910.04365*, 2019.

[51] Erdem Bıyık, Dylan P Losey, Malayandi Palan, Nicholas C Landolfi, Gleb Shevchuk, and Dorsa Sadigh.

Learning reward functions from diverse sources of human feedback: Optimally integrating demonstrations and preferences. *The International Journal of Robotics Research*, 41(1):45–67, 2022.

[52] Malayandi Palan, Nicholas C Landolfi, Gleb Shevchuk, and Dorsa Sadigh. Learning reward functions by integrating human demonstrations and preferences. *arXiv preprint arXiv:1906.08928*, 2019.

[53] Nan Rosemary Ke, Anirudh Goyal ALIAS PARTH GOYAL, Olexa Bilaniuk, Jonathan Binas, Michael C
Mozer, Chris Pal, and Yoshua Bengio. Sparse attentive backtracking: Temporal credit assignment through reminding. *Advances in neural information processing systems*, 31, 2018.

[54] Changyeon Kim, Jongjin Park, et al. Preference transformer: Modeling human preferences using transformers for rl. In *The Eleventh International Conference on Learning Representations*, 2022.

[55] Dezhong Zhao, Ruiqi Wang, Dayoon Suh, Taehyeon Kim, Ziqin Yuan, Byung-Cheol Min, and Guohua Chen. Prefmmt: Modeling human preferences in preference-based reinforcement learning with multimodal transformers. *arXiv preprint arXiv:2409.13683*, 2024.

[56] Thomas Mesnard, Theophane Weber, Fabio Viola, Shantanu Thakoor, Alaa Saade, Anna Harutyunyan, Will Dabney, Thomas S Stepleton, Nicolas Heess, Arthur Guez, et al. Counterfactual credit assignment in model-free reinforcement learning. In *International Conference on Machine Learning*, pages 7654–7664. PMLR, 2021.

[57] Mengyue Yang, Quanyu Dai, Zhenhua Dong, Xu Chen, Xiuqiang He, and Jun Wang. Top-n recommendation with counterfactual user preference simulation. In *Proceedings of the 30th ACM International* Conference on Information & Knowledge Management, pages 2342–2351, 2021.

[58] Sahil Verma, John Dickerson, and Keegan Hines. Counterfactual explanations for machine learning: A
review. *arXiv preprint arXiv:2010.10596*, 2(1):1, 2020.

[59] Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Perceiver-actor: A multi-task transformer for robotic manipulation. In *Conference on Robot Learning*, pages 785–799. PMLR, 2023.

[60] Baris Akgun, Maya Cakmak, Karl Jiang, and Andrea L Thomaz. Keyframe-based learning from demonstration: Method and evaluation. *International Journal of Social Robotics*, 4:343–355, 2012.

[61] Charles Truong, Laurent Oudre, and Nicolas Vayatis. Selective review of offline change point detection methods. *Signal Processing*, 167:107299, 2020.

[62] Yash Goyal, Ziyan Wu, Jan Ernst, Dhruv Batra, Devi Parikh, and Stefan Lee. Counterfactual visual explanations. In *International Conference on Machine Learning*, pages 2376–2384. PMLR, 2019.

[63] Andi Peng, Aviv Netanyahu, Mark K Ho, Tianmin Shu, Andreea Bobu, Julie Shah, and Pulkit Agrawal.

Diagnosis, feedback, adaptation: A human-in-the-loop framework for test-time policy adaptation. In International Conference on Machine Learning, pages 27630–27641. PMLR, 2023.

[64] Yuke Zhu, Josiah Wong, Ajay Mandlekar, Roberto Martín-Martín, Abhishek Joshi, Kevin Lin, Soroush Nasiriany, and Yifeng Zhu. robosuite: A modular simulation framework and benchmark for robot learning. In *arXiv preprint arXiv:2009.12293*, 2020.

## Neurips Paper Checklist

1. **Claims**
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The scope of this work is preference-based reinforcement learning (PbRL) for robotics. The abstract and introduction clearly state our core contributions: (1) introducing PRIMT, a framework for PbRL enhanced by foundation models; (2) proposing a hierarchical neuro-symbolic fusion strategy for multimodal feedback integration; and (3) incorporating bidirectional trajectory synthesis to address query ambiguity and credit assignment. These claims are directly supported by experimental results across diverse robotic manipulation and locomotion tasks. Key assumptions are explicitly discussed in both the main paper and appendix. Guidelines:
- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: The paper explicitly discusses limitations in Section H. Guidelines:
- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper.

- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. **Theory Assumptions And Proofs**

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [NA] Justification: This paper does not include formal theoretical results or proofs.

## Guidelines:

- The answer NA means that the paper does not include theoretical results. - All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced. - All assumptions should be clearly stated or referenced in the statement of any theorems. - The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. **Experimental Result Reproducibility**

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: The paper provides detailed descriptions of the experimental setup, including benchmark environments, backbone algorithms, and evaluation metrics. We disclose the architecture and training settings for both the reward model and policy (e.g., SAC), and we include tables listing hyperparameters for reward learning and query sampling (Appendix F). The implementation details of each baseline and ablation component are also explained. While access to specific APIs (e.g., GPT-4o) is required, we provide all prompts and sampling configurations to support reproduction. Guidelines:
- The answer NA means that the paper does not include experiments. - If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g.,
with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. **Open Access To Data And Code**

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: The benchmarks we have used are all publicly available. We plan to publicly release the full codebase via GitHub upon acceptance of the paper. Guidelines:
- The answer NA means that paper does not include experiments requiring code. - Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/
guides/CodeSubmissionPolicy) for more details.

- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://nips.cc/public/
guides/CodeSubmissionPolicy) for more details.

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. **Experimental Setting/Details**

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: We provide detailed descriptions of the training and evaluation setup in both the main paper and Appendix F. Additional tables summarize the settings used for SAC and reward learning across tasks. These details are sufficient to interpret and reproduce our results. Guidelines:
- The answer NA means that the paper does not include experiments. - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

7. **Experiment statistical significance**
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: We report mean and standard deviation across multiple independent runs for all learning curves. Guidelines:
- The answer NA means that the paper does not include experiments. - The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors). - It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

8. **Experiments compute resources**
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: We stated that all experiments were conducted on a workstation equipped with five NVIDIA RTX 4090 GPUs. Guidelines:
- The answer NA means that the paper does not include experiments. - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. **Code Of Ethics**

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: We have carefully reviewed the Code of Ethics and affirm that our work fully complies with its principles. Guidelines:
- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. **Broader Impacts**

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [Yes] Justification: We have included a detailed impact statement in Section H.2.

Guidelines:
- The answer NA means that there is no societal impact of the work performed.

- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses (e.g.,
disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies
(e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. **Safeguards**

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

## Answer: [Na]

Justification: Our paper does not involve the release of any models or datasets that pose a high risk of misuse. We do not release any large pretrained generative models, scraped image datasets, or other resources with potential dual-use concerns. All models used (e.g., LLMs or VLMs) are accessed through existing APIs with their own safety mechanisms in place, and no additional deployment or distribution is carried out by the authors. Guidelines:
- The answer NA means that the paper poses no such risks. - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. **Licenses For Existing Assets**

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: We use publicly available datasets and pretrained models, such as the MetaWorld benchmark (MIT license) and pre-trained large language models (e.g., GPT-4 via the OpenAI API) All benchmarks and models used are properly cited in the main text, and their licenses and terms of use have been followed.

Guidelines:
- The answer NA means that the paper does not use existing assets. - The authors should cite the original paper that produced the code package or dataset. - The authors should state which version of the asset is used and, if possible, include a URL. - The name of the license (e.g., CC-BY 4.0) should be included for each asset. - For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. **New Assets**

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not release new assets. Guidelines:
- The answer NA means that the paper does not release new assets. - Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

- The paper should discuss whether and how consent was obtained from people whose asset is used.

- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

14. **Crowdsourcing and research with human subjects**
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:
- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

15. **Institutional review board (IRB) approvals or equivalent for research with human subjects**
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects Guidelines:
- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. **Declaration Of Llm Usage**

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [Yes]

Justification: As described in the paper, we use foundation models, specifically LLMs and VLMs, as integral components in our preference-based reinforcement learning (PbRL) framework. LLMs are employed to synthesize structured feedback and to generate counterfactual trajectories through causal reasoning, both of which directly impact the reward learning pipeline. Guidelines:
- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

- Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.